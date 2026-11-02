"""Concurrency scaling: QPS and latency percentiles as client count rises.

This is the dimension that separates a database from an ANN library, and where
the three engines' cache designs diverge most:

* MariaDB MHNSW keeps one graph cache per TABLE_SHARE, bounded by
  `mhnsw_max_cache_size`, shared across sessions.
* AliSQL VIDX keeps a shared cache for read-only transactions plus a
  per-transaction cache for read-write ones, bounded by `vidx_hnsw_cache_size`.
* pgvector has no vector-specific cache at all; graph pages come through
  PostgreSQL's `shared_buffers` like any other index pages.

Those are three different answers to "what happens when 32 clients traverse the
same graph at once", and single-client QPS cannot reveal any of them.

Each client gets its own connection and a disjoint slice of the query set, so no
two clients issue the same query at the same instant — otherwise part of what
was measured would be one client hitting a cache another client just warmed.

Two things this workload records that are easy to leave out, and both of which
made the chart above misleading until they were added:

1. **Recall.** The ids every query returned used to be discarded, so a
   concurrency point had QPS and latency but no quality figure beside them.
   Two engines at `ef_search=100` can sit at recall 0.98 and 0.80 — that is
   the normal spread of these implementations at a shared ef, not a pathology
   — and a throughput chart with no recall column presents them as comparable
   when one is simply doing less work per query.

2. **The size of the query pool.** The pool used to be whatever
   `ops.max_queries` was, a key that exists to cap the cost of the filtered
   and churn workloads (500 in `main.yml`). Each client cycles its own shard
   for the whole window, so at 32 clients that is about fifteen queries per
   client repeated for thirty seconds, and every graph cache in the
   comparison sees a hot set small enough to hold entirely. `ops.
   concurrency_queries` separates the two, defaulting to the whole test set.

3. **How busy the clients themselves were.** Each client is a forked OS
   process, and they share one container's cpuset and serialize vectors into
   it. For a fast engine at 16-32 clients the client container can run out of
   CPU before the server does, and a throughput curve that flattens for that
   reason is indistinguishable from one that flattens because the engine
   stopped scaling -- except that the first is a property of this harness and
   the second is the finding the chart exists to report. Every window
   therefore records the client container's own cgroup CPU over the window,
   as `extra.client_cpu_seconds`, `extra.client_cpus`,
   `extra.client_cpu_util` and the derived `extra.client_bound`; `fork` never
   changes cgroup membership, so that one counter still covers every client.
   The report draws a bound point hollow and the run says so on stderr while
   it happens. See `clients.CLIENT_BOUND_UTIL` for what that flag can and
   cannot see -- it catches a saturated cpuset and not a ceiling that costs
   no CPU, and the difference has been measured. Since
   `OPS_MEASUREMENT_VERSION` 6 a shared interpreter lock is no longer one of
   those ceilings: the clients hold one interpreter each.

4. **Whether the pool ran the queries its client count implies.** The
   `scaling_efficiency` this workload publishes divides an N-client
   throughput by N times a one-client throughput, and that is a statement
   about the engine ONLY IF N queries were in flight. A live `smoke` run of
   MariaDB measured 2,920.7 q/s at one client and 2,927.5 at four with every
   latency percentile unchanged, which is about one query in service and an
   efficiency of 0.25 that is a property of the client. Every window
   therefore records `extra.outstanding_concurrency` (the exact time-average
   number outstanding), `extra.served_concurrency` (the median-span estimate
   of how many were in service) and `extra.scaling_status`; where the status
   is `concurrency_not_achieved` the report renders no efficiency figure at
   all and names the point in its Validity section. The quantity is computed
   from the spans the window already keeps, so it is engine- and
   library-agnostic and catches serialization whatever produced it -- which a
   CPU threshold cannot. See `clients.SERVED_CONCURRENCY_FLOOR`. Since ops
   measurement version 9 the status reads `outstanding_concurrency` first
   (the median estimate undercounts heavy tails and is now only the
   `tail_dominated` note), each client count runs one discarded warm-up
   window before its first repeat, and an efficiency divided by a cold
   one-client baseline is withheld (`extra.efficiency_reliability`).

PR 4 §4.3 item 3 moved the client, the barrier, the window trimming and the CPU
accounting into `clients.py`, because churn's readers need all four and a
second implementation of them would start out agreeing and then drift. Nothing
about what this workload measures changed, and the existing concurrency suite
passing untouched is what says so.
"""

from __future__ import annotations

import sys
import threading  # noqa: F401 - an injected instrument; see `_instruments`
import time
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy

from ..datasets import (RECALL_DISTANCE_THRESHOLD, Dataset,
                        cached_ground_truth, recall_at_k, recall_threshold)
from ..drivers.base import IndexSpec
from ..metrics.latency import (INFLIGHT_DISCARDED,  # noqa: F401 - instrument
                               PREWINDOW_DISCARDED, merge_collectors)
from ..metrics.scaling import (baseline_is_credible, scaling_note,
                               scaling_status)
from ..metrics.records import PHASE_CONCURRENCY, Record
from ..metrics.resources import own_cpu_count, read_own_cpu_usec  # noqa: F401
# PR 4 §4.3 item 3 moved the client, the barrier, the window trimming and the
# CPU accounting into `clients.py` so churn's readers use the same code rather
# than a second implementation that would start out agreeing and then drift.
# Re-exported under the names they had here because that is how the existing
# tests and `report/` reach them, and "concurrency tests unchanged" is the
# only thing that proves the lift changed nothing.
from .clients import (CLIENT_BOUND_UTIL,  # noqa: F401
                      SCALING_CONCURRENCY_NOT_ACHIEVED,
                      SERVED_CONCURRENCY_FLOOR, Answer, ClientPool,
                      DriverFactory, Instruments, client_cpu, shard_queries)
from .context import RunContext
from .profile_window import ProfileWindow
from .server_waits import REASON_KEY, WAITS_KEY, WaitCapture

_client_cpu = client_cpu
_shard_queries = shard_queries


def _instruments() -> Instruments:
    """This module's own clocks and counters, bundled for the pool.

    Built on every call rather than once at import, and that is the whole
    point: five concurrency tests replace `threading`, `merge_collectors` or
    `read_own_cpu_usec` ON THIS MODULE to calibrate a window against a known
    quantity, and a bundle built at import time would hold the originals.
    Looking the four names up here means a `monkeypatch.setattr` on this
    module still reaches the pool -- which is what keeps "concurrency tests
    unchanged" a real statement about the lift rather than one the lift
    quietly invalidated.
    """
    return Instruments(threading=threading, merge=merge_collectors,
                       cpu_usec=read_own_cpu_usec, cpu_count=own_cpu_count)

#: `max_queries` sentinel meaning "every query the dataset ships". The pool is
#: a property of the dataset, not of the cap the filtered and churn workloads
#: need, so the default asks for all of it rather than a number someone has to
#: remember to raise.
WHOLE_TEST_SET = 0

#: How far two windows' recall may differ before the run says so on stderr.
#: Recall is a property of the index and the ef, not of the window, so two
#: windows of the same configuration measuring different recall means the
#: offsets-to-truth mapping is wrong or the index moved underneath the
#: measurement. Either way the QPS beside it is not what it claims to be.
RECALL_SPREAD_LIMIT = 0.005

class _Window(NamedTuple):
    """One measured window's results.

    A named tuple rather than a bare tuple because `run()` sorts these by
    throughput, takes the median and then reads four other fields off it: with
    positional access that is `s[0]` in the sort, `s[2]` in the spread check
    and a five-way unpack in between, and adding a field means finding all
    three. `merged` deliberately holds nothing but `Record` columns, so
    `Record(**merged)` stays valid -- the discard counts belong in `extra` and
    are therefore carried here instead.
    """

    qps: float
    merged: Dict[str, float]
    recall: Optional[float]
    recall_queries: int
    inflight_discarded: int
    prewindow_discarded: int
    #: The client container's own CPU over this window, the size of the cpuset
    #: it was measured against, and their ratio. All three are carried rather
    #: than the ratio alone: the ratio is a cgroup counter divided by a wall
    #: clock and a CPU count, each of which can be wrong while the quotient
    #: stays plausibly between 0 and 1, so the record keeps the numerator and
    #: the denominator beside it.
    client_cpu_seconds: Optional[float]
    client_cpus: int
    client_cpu_util: Optional[float]
    #: How many queries this window actually had outstanding and in service,
    #: and whether a scaling efficiency over it is reportable at all. Carried
    #: on the sample rather than recomputed from the median window later, for
    #: the reason the recall is: "the status of the median window" has to be
    #: true by construction and not by a second indexing expression that can
    #: drift into being some other window's.
    outstanding_concurrency: Optional[float]
    served_concurrency: Optional[float]
    scaling_status: Optional[str]
    #: Ops version 9. The id-overlap recall over the same queries, kept beside
    #: the distance-threshold `recall` that is published -- see
    #: `datasets.recall_threshold`.
    recall_id_overlap: Optional[float] = None
    #: The window's two ends on the WALL clock (epoch seconds), so the report
    #: can join the server's `mem-*.jsonl` CPU and pressure samples -- which
    #: are stamped `time.time()` on the host -- to exactly this window. The
    #: perf_counter instants the spans use are not comparable across hosts.
    window_started_at: Optional[float] = None
    window_ended_at: Optional[float] = None
    #: The server's own wait profile around this window, as the two record
    #: keys `server_waits.WaitCapture.finish` returns. Carried on the sample
    #: for the reason the recall is: the waits beside the median window's
    #: QPS must be that window's.
    server_waits: Optional[Dict[str, object]] = None


#: `extra.efficiency_reliability` values. A cold one-client baseline inflates
#: every efficiency divided by it and `CREDIBLE_EFFICIENCY_CEILING` cannot see
#: it; `scaling.baseline_is_credible` can, and since ops version 9 a point it
#: flags publishes NO efficiency rather than a number about the baseline.
EFFICIENCY_CREDIBLE = "credible"
EFFICIENCY_BASELINE_NOT_CREDIBLE = "baseline_not_credible"
EFFICIENCY_UNCHECKED = "unchecked"


def _efficiency_verdict(efficiency: Optional[float],
                        baseline_p50: Optional[float],
                        p50: Optional[float],
                        clients: int) -> Tuple[Optional[float], Optional[str],
                                               Optional[float]]:
    """(efficiency to publish, reliability, p50_1 / p50_n).

    One client publishes its tautological 1.0: it IS the baseline. A median
    that got FASTER by more than `CREDIBLE_BASELINE_MEDIAN_RATIO` as clients
    were added is impossible for a warm baseline -- a queue raises the
    median, a stalled client leaves it -- so the efficiency is withheld and
    the status is judged without it.
    """
    if efficiency is None:
        return None, None, None
    ratio = (baseline_p50 / p50 if baseline_p50 and p50 else None)
    if clients == 1:
        return efficiency, EFFICIENCY_CREDIBLE, ratio
    credible = baseline_is_credible(baseline_p50, p50)
    if credible is False:
        return None, EFFICIENCY_BASELINE_NOT_CREDIBLE, ratio
    if credible is None:
        return efficiency, EFFICIENCY_UNCHECKED, ratio
    return efficiency, EFFICIENCY_CREDIBLE, ratio


def run(ctx: RunContext, driver_factory: DriverFactory, dataset: Dataset,
        index: IndexSpec, ef_search: int,
        client_counts: Sequence[int] = (1, 2, 4, 8, 16, 32),
        duration_s: float = 20.0, warmup: int = 50,
        max_queries: int = WHOLE_TEST_SET, repeats: int = 1,
        indexed_rows: Optional[int] = None,
        profile_window_clients: Optional[int] = None,
        profile_ack_dir: Optional[str] = None,
        wait_sampler: bool = False) -> None:
    """Measure QPS and latency at each client count.

    Each point runs for a fixed duration rather than a fixed query count, so a
    slow configuration cannot stretch the run indefinitely and every point is
    measured over an identical window.

    `repeats` measures each point more than once and records the median window
    with the spread beside it. Every figure in this framework was a single
    sample until a Valkey point was re-measured by accident, five hours later
    on the same build and configuration, and came back 21% higher at sixteen
    and thirty-two clients. Nothing else had ever been repeated, so nothing
    else had an error bar, and two engines whose results differ by less than
    that were being presented as ordered.

    Concurrency is where this is worth paying for: it is the headline finding,
    it is the noisiest measurement here, and a point costs a minute. Repeating
    every point three times adds about a quarter of an hour per engine to a
    run that takes days.

    `max_queries` caps the query pool; `WHOLE_TEST_SET` asks for all of it,
    which is the default and what `ops.concurrency_queries` resolves to when a
    profile does not set the key.

    `indexed_rows` must be the number of rows actually loaded, exactly as the
    filtered and churn workloads need it: a profile that loaded a subset of the
    training set would otherwise have its recall scored against neighbours
    pointing at rows the engine never received, understating every engine at
    once in a way that reads as a result.

    `profile_window_clients` names the client count that gets one extra,
    profiled window after its measured repeats -- never one of them; see
    `_profiled_window`. `wait_sampler` lets a sampling engine (pgvector)
    sample its wait events DURING the window, which is off by default because
    it is load the other engines' windows do not carry.
    """
    queries = _query_pool(dataset.test, max_queries)

    # Once per run, not once per window and not once per client count: the
    # answer depends on the dataset, k and how many rows were loaded, none of
    # which move across the sweep, and recomputing it over a subset corpus is
    # minutes of brute force.
    truth = cached_ground_truth(ctx.cache_dir, dataset, ctx.k,
                                indexed_rows=indexed_rows)

    # One driver held open for metadata, so version/size lookups do not open and
    # discard a connection per data point.
    meta_driver = driver_factory()
    meta_driver.connect()
    meta_driver._index = index
    try:
        common = ctx.record_defaults(meta_driver, dataset, index)
    finally:
        meta_driver.close()

    baseline_qps: Optional[float] = None
    baseline_p50: Optional[float] = None
    corpus = dataset.train

    for clients in client_counts:
        if clients > len(queries):
            print(f"[concurrency] skipping {clients} clients: only "
                  f"{len(queries)} queries available")
            continue

        # Ops version 9: ONE DISCARDED WINDOW, same client count and length,
        # before the first measured repeat. `full-20260825` spread +/-22-24%
        # across repeats at ONE client, from a cold first repeat: the
        # per-client `warmup` gives the one-client point the fewest warmed
        # queries of the sweep, and that point is the baseline every
        # efficiency divides by. Recorded (`extra.warmup_window_qps`) so a
        # reader can see what it absorbed, and never a candidate for the
        # median -- it is not in `samples`.
        print(f"[concurrency] {clients} client(s): one discarded warm-up "
              f"window of {duration_s:.0f}s", flush=True)
        # Not scored: its throughput is all the record keeps of it, and a
        # recall nobody publishes is a distance computation over every
        # answer of a 20 s window for nothing.
        warm = _one_window(
            driver_factory, queries, clients, ctx.k, ef_search, index,
            warmup, duration_s, truth, score=False)

        samples = []
        for attempt in range(max(1, repeats)):
            label = (f" (run {attempt + 1} of {repeats})" if repeats > 1 else "")
            print(f"[concurrency] {clients} client(s){label}: warming up "
                  f"({warmup} queries each), then measuring for "
                  f"{duration_s:.0f}s", flush=True)
            # The server's wait counters, read on a side connection
            # immediately before and after the window and never inside it.
            # See `server_waits.py`; a snapshot that took a second must not
            # move the window's QPS, and the suite pins that it does not.
            # (pgvector's opt-in sampler is the one exception, and it is
            # disclosed on every record -- see `wait_sampler`.)
            capture = WaitCapture(driver_factory, meta_driver,
                                  sampler=wait_sampler).begin()
            try:
                sample = _one_window(
                    driver_factory, queries, clients, ctx.k, ef_search, index,
                    warmup, duration_s, truth, corpus, dataset.metric)
            finally:
                waits = capture.finish()
            samples.append(sample._replace(server_waits=waits))

        # The perf profile gets its OWN window, after the measured repeats
        # and outside `samples`, exactly as the version-9 warm-up window is
        # outside them: perf sampling every server thread at 99 Hz with call
        # graphs is load on the machine being measured, so a profiled window
        # must never be a candidate for the published median. Same client
        # count, same length; its throughput is recorded so the cost of the
        # profiler is visible beside the figures it did not touch.
        profile_extra: Dict[str, object] = {}
        if (profile_window_clients is not None
                and clients == int(profile_window_clients)):
            profile_extra = _profiled_window(
                profile_ack_dir, driver_factory, queries, clients, ctx.k,
                ef_search, index, warmup, duration_s, truth)

        # The median window, not an average of several: latency percentiles
        # from one measured window mean something, and the mean of several
        # engines' percentiles means nothing. Recall rides inside the sample
        # tuple rather than being selected by a second indexing expression, so
        # "the recall of the median window" is true by construction and cannot
        # drift into being some other window's recall.
        samples.sort(key=lambda s: s.qps)
        window = samples[len(samples) // 2]
        qps, merged, recall = window.qps, window.merged, window.recall
        recall_queries = window.recall_queries
        spread = (((samples[-1].qps - samples[0].qps) / qps * 100.0)
                  if qps else 0.0)

        recall_spread = _check_recall_spread(samples, clients)

        # How many times the average query was repeated inside the window.
        # A cache-sensitive throughput number cannot be judged without it: the
        # same QPS over a pool of 10,000 and a pool of 15 are measurements of
        # two different things.
        executed = merged.get("queries_executed") or 0
        reuse = (executed / recall_queries) if recall_queries else None

        p50 = merged.get("latency_p50_ms")
        if clients == 1:
            baseline_qps = qps
            baseline_p50 = p50
        raw_efficiency = ((qps / (clients * baseline_qps)) if baseline_qps
                          else None)
        efficiency, reliability, p50_ratio = _efficiency_verdict(
            raw_efficiency, baseline_p50, p50, clients)
        # Recomputed here rather than taken from the window, because only
        # this loop knows the one-client baseline: a credible efficiency term
        # joins `outstanding`, the primary figure since version 9. See
        # `metrics/scaling.scaling_status`.
        status = scaling_status(window.served_concurrency, clients,
                                efficiency,
                                outstanding=window.outstanding_concurrency)
        note = scaling_note(window.served_concurrency,
                            window.outstanding_concurrency)

        # The median window's, by the same rule as its recall: the figure
        # beside a throughput number has to come from the window that
        # produced it. `None` -- not False -- when there was no counter to
        # read, because "the clients were not the limit" and "nobody looked"
        # are different statements and only one of them is a measurement.
        client_bound = (None if window.client_cpu_util is None
                        else window.client_cpu_util > CLIENT_BOUND_UTIL)
        if client_bound:
            _warn_client_bound(clients, window.client_cpu_util,
                               window.client_cpus, window.client_cpu_seconds)

        # Whether the pool served as many queries as it had clients. Taken
        # from the SAME median window, by the same rule as its recall and its
        # client CPU: a status computed from one window beside a throughput
        # figure from another would be a caveat about a measurement nobody
        # published.
        if status == SCALING_CONCURRENCY_NOT_ACHIEVED:
            _warn_concurrency_not_achieved(
                clients, window.served_concurrency,
                window.outstanding_concurrency, window.client_cpu_seconds,
                window.client_cpu_util)

        _print_point(common["engine"], clients, ctx.k, qps, recall, merged,
                     efficiency, spread, len(samples), status)

        ctx.recorder.write(Record(
            **common,
            phase=PHASE_CONCURRENCY,
            ef_search=ef_search,
            clients=clients,
            qps=round(qps, 2),
            # An existing Record field, not a new one: this is the same
            # quantity the recall/QPS path and the filtered and churn
            # workloads already report, so it belongs in the column the report
            # already reads rather than in `extra`.
            recall_at_k=round(recall, 6) if recall is not None else None,
            **merged,
            extra={
                "scaling_efficiency": round(efficiency, 4) if efficiency is not None else None,
                # Whether the efficiency above was divided by a warm baseline,
                # and the median ratio that says so -- see
                # `_efficiency_verdict`. `None` at a point with no baseline.
                "efficiency_reliability": reliability,
                "baseline_p50_ratio": (round(p50_ratio, 4)
                                       if p50_ratio is not None else None),
                # The discarded window before the first repeat. Never
                # selected; recorded so the cold-start cost is visible.
                "warmup_window_qps": round(warm.qps, 2),
                "warmup_window_s": duration_s,
                # Which recall rule `recall_at_k` is, and the other rule's
                # figure over the same queries. Ops version 9: one name, one
                # definition -- ann-benchmarks' distance threshold.
                "recall_kind": RECALL_DISTANCE_THRESHOLD,
                "recall_id_overlap": (round(window.recall_id_overlap, 6)
                                      if window.recall_id_overlap is not None
                                      else None),
                # The median window's wall-clock ends, for joining the
                # server's CPU and pressure samples to it.
                "window_started_at": window.window_started_at,
                "window_ended_at": window.window_ended_at,
                "duration_s": duration_s,
                "warmup_queries_per_client": warmup,
                "repeats": len(samples),
                # Every sample, so a reader can see the spread rather than
                # trust a summary of it.
                "qps_samples": [round(s.qps, 1) for s in samples],
                "qps_spread_pct": round(spread, 1),
                # How many distinct queries the recall above covers. A window
                # too short for one pass over a shard leaves part of the pool
                # unmeasured, and a recall over 200 of 10,000 queries is a
                # different claim from one over all of them.
                "recall_queries": recall_queries,
                "recall_spread": (round(recall_spread, 6)
                                  if recall_spread is not None else None),
                # The pool the clients drew from and how hard they reused it,
                # so a reader can see what the engines' caches were shown.
                "query_pool": len(queries),
                "query_reuse": round(reuse, 2) if reuse is not None else None,
                # Queries the window cut off, and the evidence that both ends
                # of the window are honest. `inflight_discarded` is at most
                # one per client on any real driver -- the one each was
                # holding when the clock stopped -- and those are the queries
                # the QPS above no longer counts. Not a hard bound: the stop
                # instant is read before the flag is set, so a query cheaper
                # than that gap can be discarded several times over. Measured
                # against a zero-latency synthetic driver at 3 clients, 6
                # discarded against 2,097-2,950 executed (0.20-0.29%). Their share of it is
                # `inflight_discarded / queries_executed`, about
                # `latency / window` and independent of the client count: a
                # few per cent for a slow engine over a short window, 0.005%
                # for a 1.4 ms query over thirty seconds.
                # `prewindow_discarded` is zero by
                # construction, because the window opens at the barrier
                # release and no measured query can precede that, so a number
                # here means the window's two ends disagree about when it
                # began.
                INFLIGHT_DISCARDED: window.inflight_discarded,
                PREWINDOW_DISCARDED: window.prewindow_discarded,
                # Whose ceiling this point shows. The clients are forked OS
                # processes sharing one container's cpuset; for a fast
                # engine at 16-32 clients they can run out of CPU first, and
                # a curve that flattens for that reason looks exactly like
                # one that flattens because the engine stopped scaling. The
                # numerator and the denominator are recorded beside the
                # ratio so a reader can see which of the three moved.
                "client_cpu_seconds": (round(window.client_cpu_seconds, 3)
                                       if window.client_cpu_seconds is not None
                                       else None),
                "client_cpus": window.client_cpus,
                "client_cpu_util": (round(window.client_cpu_util, 4)
                                    if window.client_cpu_util is not None
                                    else None),
                "client_bound": client_bound,
                # Whether the pool actually ran `clients` queries at once,
                # which is the premise `scaling_efficiency` above is
                # meaningless without. `outstanding_concurrency` is the exact
                # time-average number outstanding and `served_concurrency`
                # the median-span estimate; `served / outstanding` is exactly
                # `p50 / mean`. Since version 9 `scaling_status` is keyed to
                # the FIRST: it was rejected while the clients were threads,
                # because a span waiting on the interpreter lock still counted
                # as flight, and that cause left with version 6. The median
                # estimate is kept as evidence and drives `scaling_note`
                # (`tail_dominated`), a note and never a refusal. Both are
                # recorded, never the status alone: the status is a
                # comparison against a floor, and a reader has to be able to
                # see how far from it the point sat.
                "outstanding_concurrency": (
                    round(window.outstanding_concurrency, 3)
                    if window.outstanding_concurrency is not None else None),
                "served_concurrency": (
                    round(window.served_concurrency, 3)
                    if window.served_concurrency is not None else None),
                "scaling_status": status,
                "scaling_note": note,
                # What the SERVER waited on over the median window, from its
                # own counters -- the evidence that tells a server that
                # serializes from one that is saturated, which the CPU and
                # pressure columns cannot. `None` beside a reason when the
                # engine could not say; never an empty profile for "unknown".
                WAITS_KEY: (window.server_waits or {}).get(WAITS_KEY),
                REASON_KEY: (window.server_waits or {}).get(
                    REASON_KEY, "not captured"),
                # The extra, never-selected window perf profiled, when this
                # client count was the designated one.
                **profile_extra,
            },
        ))


def _profiled_window(ack_dir: Optional[str], driver_factory: DriverFactory,
                     queries, clients: int, k: int, ef_search: int,
                     index: IndexSpec, warmup: int, duration_s: float,
                     truth: numpy.ndarray) -> Dict[str, object]:
    """One unscored window bracketed by the perf handshake.

    The handshake -- the begin mark, the wait of up to `ACK_TIMEOUT_S` for
    the orchestrator's ack, and the end mark -- is outside `_one_window`, so
    it is outside the barrier-opened span even of this window. Returns the
    record keys: the window's QPS and whether perf was acknowledged. An
    unacknowledged window says so: either the orchestrator never answered or
    perf could not start, and `profile.json` (when written) says which.
    """
    print(f"[concurrency] {clients} client(s): one extra window for the perf "
          f"profile, not measured", flush=True)
    profile = ProfileWindow(ack_dir, clients)
    profile.begin()
    try:
        window = _one_window(driver_factory, queries, clients, k, ef_search,
                             index, warmup, duration_s, truth, score=False)
    finally:
        profile.end()
    return {"profile_window_qps": round(window.qps, 2), "profile_window": {
        "duration_s": duration_s,
        "acknowledged": bool(profile.acknowledged),
        # The orchestrator's own word on perf: "recording", or
        # "unavailable: <why>". Said even when nobody answered.
        "perf": (profile.perf_status if profile.acknowledged else
                 "no acknowledgement: the orchestrator did not answer, so no "
                 "perf profile covers this window"),
    }}


def _query_pool(test: numpy.ndarray, max_queries: int) -> numpy.ndarray:
    """The queries the clients may draw from.

    `WHOLE_TEST_SET` is a sentinel, not a cap: treating it as one leaves the
    pool empty, every client count skipped by the rule below and a silently
    empty concurrency chart. The whole point of `ops.concurrency_queries` is
    that the pool stops being whatever `ops.max_queries` happened to be — a
    key that exists to bound the filtered and churn workloads' cost — so the
    default has to mean "all of them" rather than a number a profile has to
    remember to raise.

    MUST RETURN A PREFIX OF `test`, and this is load-bearing rather than
    incidental. `run()` scores results with `truth[offsets]`, where `offsets`
    are positions into the pool this returns and `truth` is indexed by
    position into the full test set. Those two agree only while position *i*
    of the pool is position *i* of `test`. Narrow the pool any other way — a
    random sample, a stratified draw, every other query — and recall is
    silently computed against the wrong rows: the mapping failure
    `_shard_queries` warns about, one level up and with nothing in the record
    to show it happened. A non-prefix pool therefore has to translate its own
    positions back to test-set positions and hand those to `_shard_queries`.
    """
    if max_queries <= WHOLE_TEST_SET:
        return test
    return test[:max_queries]


def _window_recall(results: Dict[int, Answer], truth: numpy.ndarray,
                   k: int, queries: Optional[numpy.ndarray] = None,
                   corpus: Optional[numpy.ndarray] = None,
                   metric: Optional[str] = None
                   ) -> Tuple[Optional[float], int]:
    """Recall over the queries this window actually executed, and how many.

    Only the queries present in `results` are scored. A window shorter than
    one pass over a shard leaves the rest of the pool unexecuted, and padding
    those with empty result sets would report a recall collapse that is the
    window's length rather than the engine's quality.

    The caller has already dropped the answers whose queries fell outside the
    measured window, so this counts the same queries the QPS beside it counts.

    Given the query vectors, the corpus and the metric this is the
    distance-threshold recall (`datasets.recall_threshold`) -- what the
    record publishes since ops version 9. Without them it falls back to the
    id overlap, `_window_recall_id_overlap`.
    """
    offsets = sorted(results)
    if not offsets:
        return None, 0
    if queries is None or corpus is None or metric is None:
        return _window_recall_id_overlap(results, truth, k)
    returned = [results[offset].ids for offset in offsets]
    picks = numpy.array(offsets, dtype=numpy.int64)
    return (recall_threshold(returned, truth[picks], queries[picks], corpus,
                             metric, k), len(offsets))


def _window_recall_id_overlap(results: Dict[int, Answer],
                              truth: numpy.ndarray,
                              k: int) -> Tuple[Optional[float], int]:
    """The pre-version-9 rule: share of the true top-k ids returned."""
    offsets = sorted(results)
    if not offsets:
        return None, 0
    returned = [results[offset].ids for offset in offsets]
    want = truth[numpy.array(offsets, dtype=numpy.int64)]
    return recall_at_k(returned, want, k), len(offsets)


def _warn_nothing_finished(clients: int, duration_s: float,
                           inflight: int) -> None:
    """Say so when a window completed no query at all.

    Reachable only for a configuration whose queries outlast the window, which
    no shipped profile sets up — but the record it writes has no latency
    figures and a QPS of zero, and a gap in a chart discovered a day later is
    the expensive way to learn that `ops.concurrency_duration_s` was shorter
    than one query. Loud, and on stderr, like the recall-spread check: this one
    is a profile fault rather than a framework bug, so it says what to change.
    """
    print(
        f"[concurrency] NOTHING FINISHED: no query completed inside the "
        f"{duration_s}s window at {clients} client(s); {inflight} were still "
        f"running when it closed. The record carries a QPS of 0 and no "
        f"latency figures, because there is nothing to report one over. "
        f"Raise ops.concurrency_duration_s above this configuration's query "
        f"latency — a window shorter than a single query cannot measure "
        f"throughput.",
        file=sys.stderr, flush=True)


def _warn_client_bound(clients: int, util: float, cpus: int,
                       cpu_seconds: Optional[float]) -> None:
    """Say so when the clients, and not the engine, may be the ceiling.

    On stderr and while the run is happening, like the recall-spread check,
    because the remedy is a different machine or a different client count and
    neither is available to someone reading the report a day later. The point
    is still recorded: it is a real measurement of this harness against this
    engine, and the flag says what it is evidence about.
    """
    print(
        f"[concurrency] CLIENT CPU-BOUND: at {clients} client(s) the client "
        f"container used {util:.0%} of its {cpus} CPU(s)"
        + (f" ({cpu_seconds:.1f} CPU-seconds)" if cpu_seconds is not None
           else "")
        + f", above the {CLIENT_BOUND_UTIL:.0%} limit. The throughput at this "
        f"point is bounded by the clients as well as by the engine, so it is "
        f"not evidence about where the engine saturates. The report draws it "
        f"hollow. Re-measure with more client CPUs if the point matters.",
        file=sys.stderr, flush=True)


def _warn_concurrency_not_achieved(clients: int, served: Optional[float],
                                   outstanding: Optional[float],
                                   cpu_seconds: Optional[float],
                                   util: Optional[float]) -> None:
    """Say so when the pool never ran the queries its client count implies.

    On stderr and while the run is happening, like the client-bound and
    recall-spread checks, and for a stronger reason than either: this is the
    one condition that makes the headline figure of this workload
    unpublishable, and the operator reading the log is the only person who can
    still do something about it before the run ends.

    **THE CAUSE THIS WARNING WAS WRITTEN FOR IS GONE AND THE LINE SAYS SO.**
    Until `OPS_MEASUREMENT_VERSION` 6 the clients were threads in one
    interpreter and the expected cause was a client library holding the
    interpreter lock across its socket wait; a `smoke` run produced exactly
    that on three engines. Each client is a forked OS process now, so that
    cause cannot reach this pool, and an operator who reads the old wording
    goes looking for a client library that is not the problem.

    **The one-core heuristic is DROPPED rather than restated, because it was
    calibrated for threads.** It read "roughly one core of client CPU points
    at the clients, roughly none points at the engine" -- which held only
    because one interpreter lock capped every client's CPU at about one core
    between them. N client processes that are genuinely busy burn about N
    cores, so the same reading means the opposite of what it used to, and no
    replacement threshold has been measured here. What is printed instead is
    the raw pair: the CPU-seconds and the utilisation, so an operator can
    compare them against the client count themselves. Printed even when the
    counter is absent, as `n/a`, because "no counter" and "no CPU" are not
    the same reading and this is the line that would otherwise conflate them.
    """
    served_text = f"{served:.2f}" if served is not None else "n/a"
    outstanding_text = (f"{outstanding:.2f}" if outstanding is not None
                        else "n/a")
    cores_text = f"{cpu_seconds:.1f} CPU-seconds" if cpu_seconds is not None \
        else "no readable CPU counter"
    util_text = f"{util:.0%}" if util is not None else "n/a"
    print(
        f"[concurrency] CONCURRENCY NOT ACHIEVED: at {clients} client(s) the "
        f"pool had {outstanding_text} quer(ies) outstanding on average "
        f"(the exact mean-based figure; {served_text} by the median "
        f"estimate), and either that is below the "
        f"{SERVED_CONCURRENCY_FLOOR:.0%} of {clients} this figure needs or "
        f"neither the median estimate nor the throughput shows the clients' "
        f"queries being served at once. The scaling efficiency at this point "
        f"is NOT a property of the engine and the report refuses to render "
        f"it. The clients are separate OS processes, so the cause this check "
        f"was written for -- clients serialized on Python's interpreter lock "
        f"-- is no longer reachable; what is left is the engine itself "
        f"answering one query at a time, or a fault in this pool. A skewed "
        f"service time alone no longer lands here (ops version 9): it is "
        f"noted `tail_dominated` instead. The client container spent "
        f"{cores_text} ({util_text} of its CPUs) over this window; compare "
        f"that against {clients} client(s) rather than against a threshold.",
        file=sys.stderr, flush=True)


def _ms(value: Optional[float]) -> str:
    """A latency for the operator's line, or `n/a` where there is none.

    None arrives from a window that completed no query, which is the one case
    with no percentiles to print; formatting it with `:7.3f` raises.
    """
    return f"{value:7.3f}ms" if value is not None else f"{'n/a':>7}  "


def _print_point(engine: str, clients: int, k: int, qps: float,
                 recall: Optional[float], merged: Dict[str, float],
                 efficiency: Optional[float], qps_spread: float,
                 windows: int, scaling_status: Optional[str] = None) -> None:
    """The operator's one line per data point.

    Separate from `run` because it is presentation, and because every one of
    these figures is optional in a different way: efficiency is undefined
    until the one-client baseline exists, recall is None when no query
    completed, and a spread means nothing from a single window. Formatting
    that beside the measurement is what pushed `run` past the length this
    repository asks of a function.

    An efficiency the report will refuse is printed as `refused` here rather
    than as its number. The live log is what the operator reads while the run
    happens, and the smoke run that exposed this printed `efficiency=0.25` on
    a line whose every other figure said the pool had one client's worth of
    work in it. A number on this line that the report will not publish is the
    same number read twice with two different meanings.
    """
    eff_text = f"{efficiency:.2f}" if efficiency is not None else "n/a"
    if scaling_status == SCALING_CONCURRENCY_NOT_ACHIEVED:
        eff_text = "refused"
    spread_text = f" spread={qps_spread:.1f}%" if windows > 1 else ""
    recall_text = f"{recall:.4f}" if recall is not None else "n/a"
    print(
        f"[concurrency] {engine:>8} clients={clients:>3} "
        f"qps={qps:11,.1f} "
        f"recall@{k}={recall_text} "
        f"p50={_ms(merged['latency_p50_ms'])} "
        f"p95={_ms(merged['latency_p95_ms'])} "
        f"p99={_ms(merged['latency_p99_ms'])} "
        f"efficiency={eff_text}{spread_text}"
    )


def _check_recall_spread(samples: Sequence[_Window],
                         clients: int) -> Optional[float]:
    """How far the windows' recall figures spread, complaining if too far.

    None, not 0.0, for a single window. A lone sample has no spread in the
    same sense that it has no error bar, and 0.0 would claim the stronger
    thing — that two windows were compared and agreed exactly, which is
    precisely the evidence this field exists to carry. `report/render.py`
    already draws that distinction for `qps_spread_pct` by consulting
    `repeats`; this keeps the recall field honest without a reader having to.
    """
    recalls = [s.recall for s in samples if s.recall is not None]
    if len(recalls) < 2:
        return None
    spread = max(recalls) - min(recalls)
    if spread > RECALL_SPREAD_LIMIT:
        # Loud, and on stderr, because this is not measurement noise. Recall
        # is a property of the index and the ef, so two windows of one
        # configuration disagreeing about it means either the query-to-truth
        # mapping is wrong or the index moved underneath the measurement — and
        # in both cases the QPS recorded beside it is not measuring what the
        # record says it is.
        print(
            f"[concurrency] BUG: recall varied by {spread:.4f} across "
            f"{len(recalls)} windows at {clients} client(s), above the "
            f"{RECALL_SPREAD_LIMIT} limit. Recall cannot depend on the "
            f"window, so either the query-to-ground-truth mapping is wrong "
            f"or the index changed under the measurement; the QPS recorded "
            f"beside it is suspect.",
            file=sys.stderr, flush=True)
    return spread


def _one_window(driver_factory: DriverFactory, queries, clients: int, k: int,
                ef_search: int, index: IndexSpec, warmup: int,
                duration_s: float, truth: numpy.ndarray,
                corpus: Optional[numpy.ndarray] = None,
                metric: Optional[str] = None,
                score: bool = True) -> _Window:
    """One warm-up and one measured window.

    Recall is computed here, inside the window that produced the results, so
    the median window's throughput and the recall reported beside it are the
    same window's numbers rather than two independently chosen ones.

    The window is a closed interval `[wall_start, stop_at]` and only the
    queries that both began and ended inside it are counted -- in latency, in
    throughput and in recall. See `merge_collectors` for what counting the
    query each client was still holding did to the QPS.
    """
    # Everything from the barrier to the trimmed result set now lives in
    # `clients.ClientPool`, because churn's readers need the same machinery and
    # §4.3 item 3 asks for one implementation of it. What stays here is what is
    # concurrency's alone: the fixed duration, the recall scoring and the
    # repeats.
    pool = ClientPool(driver_factory, queries, clients, k, ef_search, index,
                      warmup, instruments=_instruments())
    pool.start(f"{clients}-client warm-up", prefix="concurrency")
    # The window is exactly this sleep: the pool's start instant was read in
    # the barrier action, before any client was released, and its stop instant
    # is read before the flag is set. See `clients.ClientPool`. The two wall
    # readings bracket it to within one call, far inside the server
    # sampler's 0.25 s interval they are joined against.
    started_at = time.time()
    time.sleep(duration_s)
    ended_at = time.time()
    window = pool.stop()

    merged = window.merged
    if not merged["queries_executed"]:
        _warn_nothing_finished(clients, duration_s, window.inflight_discarded)

    results = window.results
    if score:
        recall, recall_queries = _window_recall(results, truth, k, queries,
                                                corpus, metric)
        overlap, _ = _window_recall_id_overlap(results, truth, k)
    else:
        recall, recall_queries, overlap = None, 0, None

    return _Window(
        qps=window.qps,
        inflight_discarded=window.inflight_discarded,
        prewindow_discarded=window.prewindow_discarded,
        merged=merged,
        recall=recall,
        recall_queries=recall_queries,
        client_cpu_seconds=window.client_cpu_seconds,
        client_cpus=window.client_cpus,
        client_cpu_util=window.client_cpu_util,
        outstanding_concurrency=window.outstanding_concurrency,
        served_concurrency=window.served_concurrency,
        scaling_status=window.scaling_status,
        recall_id_overlap=overlap,
        window_started_at=round(started_at, 3),
        window_ended_at=round(ended_at, 3),
    )
