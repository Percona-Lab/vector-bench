"""Latency accumulation, percentile reporting, and what a window achieved.

Percentiles are computed from every recorded sample rather than from a running
approximation. At the query counts used here (10k test vectors per run) keeping
all samples costs a few megabytes and removes a whole class of "is the p99
real?" doubt. `nearest-rank` is used rather than interpolation so a reported p99
is always a latency that actually occurred.

Keeping every span is also what makes `WindowConcurrency` possible: how many
queries a multi-client window actually had outstanding, and how many it had in
service, are both sums over the spans this module already holds. They are here
rather than in `harness/workloads/clients.py`, where the pool that reports them
lives, for two reasons -- `span_disposition`'s definition of "inside the
window" has to be the SAME one the percentiles and the throughput use, and
`clients.py` was 846 lines against this repository's 800-line ceiling with the
policy inlined.

`SERVED_CONCURRENCY_FLOOR` and `scaling_status` followed the quantity they
judge INTO this file for that same reason, and have now followed the ceiling
back OUT of it: three rounds of measured rationale took this file to 844
lines, so the policy is in `scaling.py` and re-exported here. The argument did
not change -- the ceiling did. This file keeps the quantity; that one judges
it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence

#: Key under a concurrency record's `extra` counting the queries that were
#: still running when the measured window closed. One per client at most --
#: a client issues its queries one at a time -- and they are not counted,
#: because the wall clock in the denominator stopped at the window's end.
INFLIGHT_DISCARDED = "inflight_discarded"

#: Key counting the queries that had already started when the window opened.
#: Zero on every run, and structurally so: the window opens at the instant the
#: clients' barrier releases, which precedes every measured query. It is
#: recorded rather than assumed because the first attempt at this field was not
#: zero at all -- the window opened when the parent thread next read the clock,
#: which six of eight clients beat to their first query. A number here means
#: the window's two ends disagree about when it began.
PREWINDOW_DISCARDED = "prewindow_discarded"


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "queries_executed": self.count,
            "latency_mean_ms": round(self.mean_ms, 4),
            "latency_p50_ms": round(self.p50_ms, 4),
            "latency_p95_ms": round(self.p95_ms, 4),
            "latency_p99_ms": round(self.p99_ms, 4),
        }


def percentile(sorted_samples: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. `q` in [0, 1]. Input must already be sorted."""
    if not sorted_samples:
        return float("nan")
    if q <= 0:
        return sorted_samples[0]
    if q >= 1:
        return sorted_samples[-1]
    rank = math.ceil(q * len(sorted_samples))
    return sorted_samples[min(max(rank - 1, 0), len(sorted_samples) - 1)]


def summarize(samples_s: Sequence[float]) -> LatencyStats:
    """Summarize latencies given in seconds; results are in milliseconds."""
    if not samples_s:
        return LatencyStats(0, float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"))
    ms = sorted(s * 1000.0 for s in samples_s)
    return LatencyStats(
        count=len(ms),
        mean_ms=sum(ms) / len(ms),
        p50_ms=percentile(ms, 0.50),
        p95_ms=percentile(ms, 0.95),
        p99_ms=percentile(ms, 0.99),
        min_ms=ms[0],
        max_ms=ms[-1],
    )


class Timer:
    """Context manager returning elapsed wall time in seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - (self._start or time.perf_counter())


class Sample(NamedTuple):
    """One query's latency, with the instants bracketing it when known.

    `started` and `finished` are `time.perf_counter()` readings and are None
    for a sample recorded by duration alone. That is not a degenerate case: it
    is what the filtered and churn workloads produce, and they measure to
    completion rather than to a deadline, so there is no window for them to
    fall outside of and nothing for a cutoff to compare against.

    `seconds` is derived once, at construction, rather than recomputed from the
    instants on the way out. Two spellings of one duration is how they come to
    disagree.
    """

    seconds: float
    started: Optional[float] = None
    finished: Optional[float] = None


def span_disposition(started: Optional[float], finished: Optional[float],
                     window_start: Optional[float],
                     cutoff: Optional[float]) -> Optional[str]:
    """Why a query does not belong to a measured window, or None if it does.

    ONE definition, used both by `merge_collectors` to decide which latencies
    to count and by the concurrency workload to decide which returned ids to
    score. Those two decisions have to be the same decision: the window's QPS
    and the recall printed beside it would otherwise describe different sets of
    queries, and `extra.query_reuse` -- executed over distinct -- would come
    out below 1.0 for a window in which nothing was repeated.

    The cutoff is tested first, so a query that both started before the window
    and outlived it counts once, as in-flight. Every span therefore lands in
    exactly one of three buckets and the two discard counts plus the kept
    count add up to what was recorded.
    """
    if cutoff is not None and finished is not None and finished > cutoff:
        # `>` and not `>=`: a query that completed at the instant the clock
        # stopped was paid for inside the window the denominator measures.
        return INFLIGHT_DISCARDED
    if (window_start is not None and started is not None
            and started < window_start):
        return PREWINDOW_DISCARDED
    return None


class WindowConcurrency(NamedTuple):
    """How many queries a measured window actually had in it, two ways.

    Both are derived from the spans the window already keeps, by the SAME
    `span_disposition` call that decides which latencies and which returned
    ids count -- so they describe exactly the queries the QPS beside them
    counts. Neither needs a distributional assumption and neither needs a
    field the record does not already carry.

    `outstanding` is Little's law computed exactly: the integral of the
    occupancy function over the window, divided by the window. Sum the spans
    and you have the area; there is no estimator in it. It answers "how many
    queries did the clients have in flight on average".

    `served` is Little's law with the MEDIAN span rather than the mean. It
    answers a different question -- "how many queries were in SERVICE on
    average" -- and the two diverge by exactly the time a span spends waiting
    for Python's interpreter lock instead of waiting for the engine.

    **Which one a guard must use was settled by measurement, not by
    preference, and the answer is not the exact one.** Five synthetic drivers
    of identical nominal query cost (0.325 ms, the smoke run's observed p50)
    were run through the real `ClientPool` at 1 and 4 clients
    (CPython 3.12.3, 3 windows of 2 s per point, median window reported;
    `tests/test_g1_client_concurrency.py` rebuilds four of the five regimes
    and asserts the verdicts rather than the decimals):

        driver                     efficiency   outstanding/n   served/n
        sleep (releases the lock)        1.00           0.99        0.96
        saturated server                 0.28           1.00        0.98
        atomic C call holding the lock   0.27           0.99        0.22
        Python spin holding the lock     0.25           0.98        0.25
        half released, half held         0.52           1.00        0.96

    `outstanding/n` is 0.98-1.00 on all five, across efficiencies from 0.25 to
    1.00. It is BLIND, and the reason is mechanical rather than a matter of
    noise: a thread that is waiting for the lock is still inside
    `driver.query`, because the span's start instant is read in Python and the
    interpreter can drop the lock at any bytecode boundary after it. The wait
    lands inside the span, the stretched spans tile the wall, and the sum
    approaches the client count however little work was served. The signature
    in the same rows is a mean far above the median -- 1.025 ms against
    0.227 ms for the atomic driver -- and a p99 of 16.7 ms where the p50 did
    not move.

    `served/n` separates the two regimes by a factor of four: 0.22-0.25 where
    the pool served about one query, 0.96-0.98 where it served four. A
    genuinely saturated server raises the WHOLE distribution, median
    included -- 0.389 ms to 1.377 ms at four clients here -- so its `served`
    stays at the client count and it is not flagged. That is the distinction
    a guard has to get right.

    **THAT VERDICT WAS ABOUT THREADS AND IS RETIRED AT OPS VERSION 9.**
    Every row above that separates the two columns is a lock-HOLDING driver,
    and the blindness is mechanical only while the clients share one
    interpreter lock. They have been separate OS processes since version 6,
    so a span can no longer stretch while its client waits for a lock some
    other client holds, and `outstanding` is again what it always was: the
    exact time-average occupancy, with no estimator in it. Keeping the median
    as the guard kept only its own bias -- under a heavy tail `served`
    undercounts by `p50 / mean`, which on `full-20260825` put MariaDB 12.3 at
    32 clients (p50 34 ms, p99 1,043 ms) at a fraction of its real occupancy,
    and printed a one-client "share of the window inside the call" of 74-75%
    where `qps x mean latency` is 1.00 for all six engines. So since version 9
    `scaling.scaling_status` reads `outstanding` first, and `served` survives
    as the median ESTIMATE beside it and as `scaling.tail_dominated`'s input.
    The table stays because it is the evidence for why the thread pool needed
    the median and why the process pool does not.

    `None`, never 0.0, when the window counted no query: zero is the claim
    "the clients had nothing in flight", which cannot be true of a window
    that executed queries, and it is the value a guard would read as
    serialization on a window nobody measured.
    """

    outstanding: Optional[float]
    served: Optional[float]


def window_concurrency(collectors: Sequence["LatencyCollector"],
                       wall_seconds: float,
                       cutoff: Optional[float] = None,
                       window_start: Optional[float] = None
                       ) -> WindowConcurrency:
    """The concurrency a window achieved, from the spans it already kept.

    Beside `merge_collectors` and not inside it, because `merge_collectors`
    returns `Record` columns and these two are not columns -- they belong in
    the concurrency record's `extra` the way the client-CPU figures do. In
    this module rather than in the workload, because the kept-span rule has to
    be the SAME rule: one `span_disposition` call decides which spans the QPS
    counts, which latencies the percentiles cover, which ids recall scores and
    now which spans these two sum. Four definitions of "inside the window" is
    how a published recall comes to describe a different query set from the
    throughput beside it, which this module's own history records.

    The median is `percentile(sorted, 0.50)` -- the same nearest-rank function
    and the same definition `latency_p50_ms` is reported with -- so a reader
    can check `served` against the record with one multiplication:
    `queries_executed x latency_p50_ms / 1000 / window`. An interpolating
    median here would make that arithmetic disagree with the published p50 and
    the check would read as a bug in the guard.
    """
    kept: List[float] = []
    for c in collectors:
        for sample in c._samples:
            if span_disposition(sample.started, sample.finished,
                                window_start, cutoff) is None:
                kept.append(sample.seconds)
    if not kept or wall_seconds <= 0:
        return WindowConcurrency(None, None)
    kept.sort()
    return WindowConcurrency(
        outstanding=sum(kept) / wall_seconds,
        served=len(kept) * percentile(kept, 0.50) / wall_seconds,
    )


# The scaling POLICY lives in `scaling.py`: the floor, the statuses, the two
# estimators of the share of clients in service, and the rule over them. It
# was here, beside the quantity it judges, until three rounds of measured
# rationale took this file to 844 lines against the repository's 800-line
# ceiling -- the same ceiling that moved the policy out of `clients.py` and
# into this file in the first place. `scaling.py`'s docstring has the history.
#
# RE-EXPORTED AND NOT SHARED, which is the trap. `harness/workloads/` and
# `report/` reach these names through this module and keep working; a
# MUTATION aimed at this module does not, because `scaling_status` reads
# `scaling.py`'s globals. Patch `harness.metrics.scaling`.
from .scaling import (CREDIBLE_BASELINE_MEDIAN_RATIO,  # noqa: F401
                      CREDIBLE_EFFICIENCY_CEILING,
                      SCALING_CONCURRENCY_NOT_ACHIEVED, SCALING_NOTES,
                      SCALING_OK, SCALING_STATUSES, SCALING_TAIL_DOMINATED,
                      SERVED_CONCURRENCY_FLOOR, achieved_fraction,
                      baseline_is_credible, scaling_note, scaling_status,
                      served_fraction, tail_dominated, throughput_fraction)


class LatencyCollector:
    """Collects per-query latencies, optionally discarding a warm-up prefix.

    Warm-up matters here: the first queries against a freshly started engine pay
    for cold graph caches (MHNSW's per-TABLE_SHARE cache, VIDX's share cache,
    PostgreSQL's shared_buffers). Including them in a steady-state QPS figure
    understates every engine, and understates the ones with larger caches most.
    """

    def __init__(self, warmup: int = 0):
        self.warmup = max(0, warmup)
        self._samples: List[Sample] = []
        self._seen = 0
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    def add(self, seconds: float) -> None:
        """Record a query by its duration alone.

        What the build, filtered and churn workloads call. Those loops run to
        completion, so a duration is the whole truth about the query and there
        is no window it could fall outside.
        """
        self._record(Sample(seconds))

    def add_span(self, started: float, finished: float) -> None:
        """Record a query by the two instants that bracket it.

        What a fixed-duration window needs, because whether a query counts
        depends on WHEN it ran and not only on how long it took. Both
        arguments must be `time.perf_counter()` readings on the same clock as
        the window's own start and cutoff.

        The cost of keeping the instants is two more floats per query -- at a
        fast engine's 20,000 q/s over a 20 s window that is tens of megabytes
        in the harness container, which holds the dataset in NumPy anyway and
        is deliberately a different container from the one whose memory the
        benchmark measures.
        """
        self._record(Sample(finished - started, started, finished))

    def _record(self, sample: Sample) -> None:
        """The one place a sample enters, so the warm-up prefix is discarded
        identically however the caller spelled the query."""
        self._seen += 1
        if self._seen > self.warmup:
            self._samples.append(sample)

    def stop(self) -> None:
        self._wall_end = time.perf_counter()

    @property
    def wall_seconds(self) -> float:
        if self._wall_start is None:
            return 0.0
        end = self._wall_end if self._wall_end is not None else time.perf_counter()
        return end - self._wall_start

    @property
    def measured_seconds(self) -> float:
        """Total time attributable to the measured (post-warm-up) queries."""
        return sum(s.seconds for s in self._samples)

    def qps(self, clients: int = 1) -> float:
        """Throughput over the measured window.

        With multiple clients the wall clock is shared, so QPS is measured
        queries divided by wall time — not by the sum of per-query latencies,
        which would count concurrent work multiple times.
        """
        wall = self.wall_seconds
        if wall <= 0 or not self._samples:
            return 0.0
        if clients > 1:
            return len(self._samples) / wall
        return len(self._samples) / max(self.measured_seconds, 1e-12)

    def stats(self) -> LatencyStats:
        return summarize([s.seconds for s in self._samples])

    def merge(self, other: "LatencyCollector") -> None:
        self._samples.extend(other._samples)
        self._seen += other._seen


def merge_collectors(collectors: Sequence[LatencyCollector],
                     wall_seconds: float,
                     cutoff: Optional[float] = None,
                     window_start: Optional[float] = None) -> Dict[str, float]:
    """Combine per-client collectors into one concurrency data point.

    `window_start` and `cutoff` are the instants the measured window opened
    and closed. Given them, only the queries that both began and ended inside
    that window are counted -- which is the only set of queries the wall time
    in the denominator actually covers.

    WHY THAT MATTERS, because the correction looks like a rounding. A
    fixed-duration window ends by setting a flag; each client then finishes the
    query it is holding and records it, so the numerator used to include one
    query per client that the denominator did not.

    The relative error is that one query over the queries a client did finish:
    `1 / queries-completed-per-client`, which in published record fields is
    `inflight_discarded / queries_executed` and is about
    `per-query latency / window length`. **The client count cancels.** Each
    client contributes one spurious query and roughly `window / latency` real
    ones, so the ratio is the same at 1 client as at 32 -- measured flat at
    2.564% across 1, 2, 4, 8, 16 and 32 clients over a two-second window of
    50 ms queries. What the error does depend on is a SLOW ENGINE OVER A SHORT
    WINDOW: it is 0.005% for a 1.4 ms query in a 30-second window and 2.5% for
    a 50 ms query in two seconds, whatever the concurrency.

    It also broke Little's law, reporting more queries in flight on average
    than there were clients to hold them, which no client count can do.

    Both bounds are optional and default to off, so a caller that measures to
    completion -- the filtered and churn workloads -- keeps every sample. A
    sample recorded by duration alone is never dropped: it carries no instants
    to compare, and dropping it would empty those workloads' percentiles rather
    than trim them.

    Returns the latency summary plus `qps` and the two discard counts, which
    the concurrency workload moves into the record's `extra`. A window in which
    nothing finished in time reports `queries_executed` 0, a QPS of 0 and no
    latency figures at all -- the honest answer for a configuration whose
    queries outlast the window, rather than one in-flight query's throughput.
    """
    kept: List[float] = []
    discarded = {INFLIGHT_DISCARDED: 0, PREWINDOW_DISCARDED: 0}
    for c in collectors:
        for sample in c._samples:
            reason = span_disposition(sample.started, sample.finished,
                                      window_start, cutoff)
            if reason is None:
                kept.append(sample.seconds)
            else:
                discarded[reason] += 1

    stats = summarize(kept)
    out = stats.as_dict()
    if not kept:
        # None, not the NaN `summarize([])` returns. `json.dumps` writes a NaN
        # as the bare literal `NaN`, which is not JSON: Python's `json.loads`
        # accepts it as a documented extension and the browser's `JSON.parse`
        # rejects it, so a records file holding one reads back perfectly in
        # `report/loaders.py` and throws in the web UI. One consumer silently
        # fine and another broken is this repository's signature fault, and a
        # measurement framework should not have a path that produces it.
        #
        # Trimming the window newly created that path: before it, every client
        # completed at least one query, so the sample set was never empty.
        # Now a configuration whose queries outlast the window keeps nothing.
        #
        # Nulled HERE and not in `summarize`, which predates this window and
        # is also what the build, filtered and churn workloads reach through:
        # those loops run to completion, so an empty sample set there means
        # nothing was asked of the engine at all, and widening the change
        # widens the blast radius for no gain.
        #
        # Derived from `as_dict`'s own keys rather than from a list of field
        # names kept beside it, so a field added to `LatencyStats` is nulled
        # here by default instead of being the one that still reports NaN.
        out = {key: None for key in out}
        # The count is the one figure an empty window does know, and 0 is a
        # measurement rather than a missing one.
        out["queries_executed"] = 0
    # Zero completions over the window, which is what was observed. Unlike a
    # null it plots, and unlike a NaN it serialises.
    out["qps"] = len(kept) / wall_seconds if wall_seconds > 0 else 0.0
    out.update(discarded)
    return out
