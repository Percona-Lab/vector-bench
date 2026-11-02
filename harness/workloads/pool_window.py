"""What identifies a client pool's window, and what a ratio across two needs.

`PoolSpec`, `PoolWindow` and `window_ratio` are one unit and are here rather
than in `harness/workloads/clients.py` for the reason `metrics/latency.py`
and `churn_readers.py` exist: `clients.py` stood at 791 lines against this
repository's 800-line ceiling before G3 made the clients OS processes, and
there was no version of that change which fitted. They are the three pieces
`report/` and `churn.py` read rather than the pool machinery that produces
them -- a `PoolWindow` is passed around, pinned by `tests/test_t4b_readers.py`
field by field, and divided by `window_ratio`, which exists only to refuse
when the two windows' `PoolSpec`s disagree.

`CLIENT_BOUND_UTIL` came with them, and that is better placement than it had.
It is read in exactly one place -- `PoolWindow.client_bound` below -- and
`metrics/latency.py` makes the same argument for `SERVED_CONCURRENCY_FLOOR`:
the threshold belongs beside the quantity it judges. **One consequence has to
be written down, because this repository has been caught by it twice.** A
module that re-exports a name does not SHARE it, so rebinding
`clients.CLIENT_BOUND_UTIL` reaches nothing; the constant `client_bound`
compares against is this module's. Nothing patches it today and
`tests/test_g3_process_clients.py` asserts the asymmetry, so a future
mutation aimed at `clients.CLIENT_BOUND_UTIL` is recorded as dead rather
than reported as a guard that held -- exactly as
`tests/test_g1_client_concurrency.py` does for `SERVED_CONCURRENCY_FLOOR`.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

from .client_process import Answer, PoolError
from ..metrics.latency import scaling_note, scaling_status, served_fraction

#: Share of the client container's CPUs above which a point is reported as
#: client-bound. Not a measured constant and not claimed to be one: it is the
#: level at which this harness stops being able to say whose ceiling a figure
#: shows. Client processes serializing vectors do not scale smoothly to the
#: last percent of a cpuset, so a point at 0.8 is already one where they may
#: be contributing, and the honest report is "this number is not evidence about
#: the engine" rather than a correction factor nobody can justify.
#:
#: SUFFICIENT, NOT NECESSARY. See this module's docstring for the measurement:
#: raising or lowering 0.80 does not help, because the quantity itself is blind
#: to a GIL ceiling -- and for the regime that measurement was taken in, which
#: is the numpy-bound one and not the socket-bound one a real driver runs in.
#:
#: THAT BLINDNESS IS NOW A MEASUREMENT AND NOT AN ARGUMENT. Four GIL-holding
#: clients at 0.325 ms per query delivered a quarter of this pool's throughput
#: -- `scaling_efficiency` 0.26 against 1.02 for the same cost released --
#: while `client_cpu_util` read **0.177** against this 0.80 and the flag
#: stayed `False`. The 0.177 is SUPPLIED: that counter is a cgroup's and reads
#: the whole machine outside a container.
#: `metrics/latency.SERVED_CONCURRENCY_FLOOR` is the quantity that sees it.
#:
#: Kept unchanged by the move to client PROCESSES. A client container that
#: genuinely runs out of CPU is still a real case and is the case this flag
#: catches; what the move removed is the OTHER case, the one it never saw.
CLIENT_BOUND_UTIL = 0.80


class PoolSpec(NamedTuple):
    """What identifies one client pool, for comparing two of its windows.

    This exists because of the shape of PR 4 §4.3 item 5's headline number:
    `p99 under writes / baseline p99`, a ratio of two separately measured
    percentiles. This repository has published two wrong ratios of exactly
    that shape -- the filter-strategy ratio divided a narrow point from one
    ops unit by a wide point from another, and `charts.churn_retention` pairs
    records from two different maintenance states -- and both read as findings
    rather than as bugs. So a ratio across two windows is computed only when
    the two windows agree on every axis that could make them measurements of
    different things, and this tuple is that list.

    `unit` carries the ops unit: (M, build mode, storage engine, resource
    pass). It is `()` for the concurrency workload, which takes no such ratio.
    It is deliberately a tuple the CALLER builds rather than a field list this
    module knows, because `harness/` must not import `report/` -- see
    `churn.OPS_UNIT_KEYS`, which names the axes and is pinned against
    `report.generate.OPS_UNIT_FIELDS` by a test.
    """

    clients: int
    #: How many distinct queries the clients drew from. Two windows over pools
    #: of 10,000 and of 15 queries are measurements of two different things:
    #: every graph cache in this comparison is sensitive to the size of the
    #: hot set, which is the whole reason `ops.concurrency_queries` exists.
    queries: int
    k: int
    ef_search: int
    unit: Tuple[Any, ...] = ()


class PoolWindow(NamedTuple):
    """One measured window of one pool.

    `merged` holds nothing but `Record` columns, so `Record(**merged)` stays
    valid; the discard counts and the throughput are carried separately for
    that reason rather than left inside it.
    """

    qps: float
    merged: Dict[str, float]
    inflight_discarded: int
    prewindow_discarded: int
    #: The client container's own CPU over this window, the size of the cpuset
    #: it was measured against, and their ratio. All three, never the ratio
    #: alone: the ratio is a cgroup counter divided by a wall clock and a CPU
    #: count, each of which can be wrong while the quotient stays plausibly
    #: between 0 and 1.
    client_cpu_seconds: Optional[float]
    client_cpus: int
    client_cpu_util: Optional[float]
    #: The wall the throughput above is over, kept because a reader cannot
    #: otherwise tell a busy window from a short one.
    wall_s: float
    #: Which pool measured this. Two windows with different specs are not two
    #: samples of one quantity -- see `PoolSpec`.
    spec: PoolSpec
    #: Query position -> that query's answer, trimmed to this window. Empty
    #: when the pool was built with `collect_results=False`, which is what the
    #: churn readers do: §4.3 item 6 says they compute no recall, because the
    #: corpus is mid-change and no ground truth describes it.
    results: Dict[int, Answer]
    #: The two concurrency figures, over exactly the spans the QPS above
    #: counts: the exact time-average number of queries the clients had
    #: OUTSTANDING, and the median-span estimate of how many were in SERVICE.
    #: `metrics/latency.WindowConcurrency` has the measurement that says why
    #: both are kept and why only the second can carry a guard.
    #:
    #: Defaulted rather than required, and the default is the honest absence:
    #: `tests/test_t4b_readers.py` hand-builds a `PoolWindow` to exercise
    #: `window_ratio`'s refusals, which are a property of `PoolSpec` alone.
    #: A None there says nobody measured it. `stop` passes both explicitly.
    outstanding_concurrency: Optional[float] = None
    served_concurrency: Optional[float] = None

    @property
    def p99_ms(self) -> Optional[float]:
        return self.merged.get("latency_p99_ms")

    @property
    def queries_executed(self) -> int:
        return int(self.merged.get("queries_executed") or 0)

    @property
    def client_bound(self) -> Optional[bool]:
        """Whether the clients may have been part of the ceiling.

        `None` -- not False -- when there was no counter to read, because "the
        clients were not the limit" and "nobody looked" are different
        statements and only one of them is a measurement.

        **WHAT THIS CANNOT SEE, as a measurement rather than an argument.**
        Four clients were run through this pool on a driver costing 0.325 ms
        per query and holding the interpreter lock for it, against the same
        driver releasing the lock -- the pair that
        `tests/test_g1_client_concurrency.py` rebuilds. The releasing one
        scaled -- `scaling_efficiency` 1.02 at four clients -- and the
        holding one delivered 0.26. On that 0.26 point `client_cpu_util` read
        **0.177** (SUPPLIED: that counter is a cgroup's and reads the whole
        machine outside a container, so only a containerised run makes it)
        against `CLIENT_BOUND_UTIL` of **0.80**, and this property returned
        `False` -- three of four clients' throughput went missing and the flag
        said the clients were fine. Not a threshold merely set too high: a
        lock ceiling PREVENTS the CPU consumption the quotient looks for,
        since a thread waiting its turn is not running and costs the counter
        nothing, and moving 0.80 into (0, 0.177] would fire on every point of
        every run. Read `scaling_status` instead, computed from the spans.
        This flag is kept because it catches a different and real case -- a
        client container that genuinely runs out of CPU at 16-32 clients with
        a driver that does release the lock, which `scaling_status` cannot.

        The clients are processes now, so the case above can no longer be
        reached through Python's own lock -- but the paragraph stays, because
        the figures in it are what a reader needs to understand why this flag
        is not the instrument, and because a client LIBRARY that serializes
        inside one process is not the only way a pool fails to achieve its
        concurrency. This flag's own case is unaffected by the move: the
        cgroup counter it divides still covers every client, since fork does
        not change cgroup membership.
        """
        if self.client_cpu_util is None:
            return None
        return self.client_cpu_util > CLIENT_BOUND_UTIL

    @property
    def served_fraction(self) -> Optional[float]:
        """`served_concurrency` as a share of this pool's client count.

        NOT recursive, though it reads like it: a class body is not an
        enclosing scope for the functions defined in it, so the call below
        resolves to `metrics/latency.served_fraction` through this module's
        globals and never to this property. Said out loud because the obvious
        "fix" -- renaming one of them -- would be undoing a deliberate choice
        to keep the window's view of the quantity under the quantity's own
        name. `scaling_status` below is the same shape.
        """
        return served_fraction(self.served_concurrency, self.spec.clients)

    @property
    def scaling_status(self) -> Optional[str]:
        """Whether a scaling efficiency over this window is reportable.

        `metrics/latency.scaling_status` holds the rule, the floor and the
        two statuses, and carries the measurement that chose the floor. This
        property is the window's view of it and nothing else, so there is one
        definition a test can move.

        `outstanding` is passed as the primary figure since ops version 9 --
        the exact span sum, no longer the median estimate alone, for the
        reason `metrics/scaling.scaling_status` gives. No efficiency: the
        window does not know the one-client baseline `concurrency.run`
        divides by, so that caller recomputes the status with it.
        """
        return scaling_status(self.served_concurrency, self.spec.clients,
                              outstanding=self.outstanding_concurrency)

    @property
    def scaling_note(self) -> Optional[str]:
        """`tail_dominated` or None: the note `metrics/scaling.scaling_note`
        computes from the two figures, never a refusal."""
        return scaling_note(self.served_concurrency,
                            self.outstanding_concurrency)


def window_ratio(numerator: PoolWindow, denominator: PoolWindow,
                 field: str) -> Optional[float]:
    """One field's ratio across two windows of the SAME pool.

    The whole value of this function is the refusal. PR 4 §4.3 item 5's
    headline number is `p99 under writes / baseline p99`, and a ratio of two
    separately measured quantities is the shape that has produced two
    published-wrong results in this repository: the filter-strategy ratio
    divided a narrow point from one ops unit by a wide point from another, and
    `charts.churn_retention` pairs records from two different maintenance
    states. Both looked like findings. So the two windows must agree on every
    axis in `PoolSpec` -- client count, query pool, k, ef_search and the ops
    unit -- and a caller that crosses any one of them gets a `PoolError`
    naming the field rather than a plausible number.

    `None` when either window has no value for `field`, which is what a window
    that completed no query records. A ratio over a missing percentile would
    be a `TypeError` at best and a 0.0 published as "no slowdown" at worst.
    """
    if numerator.spec != denominator.spec:
        differing = [name for name, a, b
                     in zip(PoolSpec._fields, numerator.spec, denominator.spec)
                     if a != b]
        raise PoolError(
            f"refusing to divide two windows measured by different pools: "
            f"they differ on {', '.join(differing)} "
            f"({numerator.spec} vs {denominator.spec}). A ratio across two "
            f"pools is not a measurement of either.")
    top = numerator.merged.get(field)
    bottom = denominator.merged.get(field)
    if not top or not bottom:
        return None
    return top / bottom


__all__ = ["CLIENT_BOUND_UTIL", "PoolSpec", "PoolWindow", "window_ratio"]
