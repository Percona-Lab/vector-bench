"""G1: whether a multi-client window achieved the concurrency it claims.

**This is the first defect in this project found by a live run rather than by
reading.** A `smoke` pass of MariaDB on fashion-mnist-784 at M=16 measured
2,920.7 q/s at one client and 2,927.5 q/s at four -- flat across a four-fold
rise -- with every latency percentile unchanged (p50 0.325 ms, p95
0.442/0.439, p99 0.552/0.544). A saturated server raises latency; a CPU-bound
client raises latency; neither happened. `qps x p50` is 0.95, so about one
query was in service while four clients ran, and the `scaling_efficiency` of
0.25 printed beside it is a property of the client. Those five numbers are
SUPPLIED -- no engine is started here and nothing in this file reproduces
them.

What this file measures is everything downstream of them: that the harness now
records how many queries a window actually had in it, that it refuses to
publish a scaling efficiency when the answer is "about one", and that it does
NOT refuse one when the engine itself is the bottleneck.

**The quantity the refusal uses is not the obvious one, and that was settled
by measurement.** Two candidates come out of the same spans:

* `outstanding_concurrency` -- sum of in-window span durations over the wall.
  Exact: Little's law with no estimator in it.
* `served_concurrency` -- counted queries times their median span, over the
  wall. Robust: Little's law with a median.

Measured against five synthetic drivers of identical per-query cost, the
exact one reads **0.84-1.00** of the client count on all ten readings of both
pools, across efficiencies from 0.249 to 1.009 -- including the ones
delivering a quarter of their throughput. The reason is not the lock: a span
brackets `driver.query`, so it covers whatever the query spent WAITING as
well as whatever it spent being served, and the stretched spans tile the
window either way. Under threads the wait was the interpreter lock; under
processes it is a queue in the engine or its socket. The robust one separates
the two regimes by a factor of four. Both are recorded; only the second
carries the guard. `TestTheExactFigureIsBlindAndTheRobustOneIsNot` is that
measurement, run here rather than quoted; an earlier revision of this
paragraph quoted the thread arm's 0.98-1.00 as if it were both.

**Mutation discipline.** Every guard below names the regression it catches and
was checked by injecting that regression AT RUN TIME -- monkeypatch, a
subclass, or a verbatim copy of the pre-change body run over the same input --
never by editing a file on disk. CPython invalidates a `.pyc` on (whole-second
mtime, size), so a same-size rewrite inside one second can leave stale
bytecode valid and measure the previous version. Honest and mutant values are
in each docstring and were read back out of the records or the rendered text
the code produced, never described.

**R2, for the one repair.** `report/render._efficiency_cell` replaces an
expression that published the number unconditionally. Its test runs that
pre-change expression, copied verbatim, over the very same record and reports
what it published beside what the new form publishes. A repair whose old form
also passes is not a repair.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, List

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import IndexSpec                            # noqa: E402
from harness.metrics import records                                   # noqa: E402
from harness.metrics import latency as latency_mod                    # noqa: E402
from harness.metrics import scaling as scaling_mod                    # noqa: E402
from harness.metrics.latency import (LatencyCollector,                # noqa: E402
                                     merge_collectors, window_concurrency)
from harness.workloads import churn as cw                             # noqa: E402
from harness.workloads import churn_readers as crw                    # noqa: E402
from harness.workloads import clients as cl                           # noqa: E402
from harness.workloads import concurrency as conc                     # noqa: E402
from harness.workloads.context import RunContext                      # noqa: E402
from report import render as render_mod                               # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES, N_ROWS,      # noqa: E402
                               index_spec, tiny_dataset)
# The pool G3 retired, kept because it is the only thing in the tree that can
# still produce the shape `scaling_status` refuses -- see that module.
from tests.retired_thread_pool import thread_window                   # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:  # pragma: no cover - import probe
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

needs_matplotlib = pytest.mark.skipif(
    not HAVE_MPL, reason="matplotlib is not installed on this host")

#: Wall clock each synthetic query costs, identical across every driver family
#: below. **That identity is the whole experiment**: the families differ only
#: in what they do with the interpreter lock while they spend it, so any
#: difference in what the pool achieves is attributable to the lock and not to
#: the work.
#:
#: 0.5 ms rather than the smoke run's 0.325 ms, and the choice is bounded at
#: both ends by things that were measured here. It has to stay WELL BELOW the
#: interpreter's switch interval (5 ms by default): a lock-holding thread
#: yields after that long, so the share of queries whose span absorbs a lock
#: wait is roughly `QUERY_S / switchinterval`, and once that share passes a
#: half the MEDIAN span rises too and the pool stops being distinguishable
#: from a saturated server. At 1 ms the separation was already gone -- the
#: four-client lock-holding sweep measured p50 4.796 ms against a 0.5 ms
#: nominal cost and read as `ok`. It also has to stay well ABOVE the pool's
#: own per-query bookkeeping, which is microseconds, or a scaling driver's
#: `served` falls for a reason that is not the lock.
QUERY_S = 0.0005

#: The measured window. Long enough that a one-client pool completes several
#: hundred queries -- a median over a handful of spans is the box's rather
#: than the driver's -- and short enough that four driver families times two
#: client counts stay inside a test run.
WINDOW_S = 0.6

#: Client counts every end-to-end test sweeps. One is the efficiency
#: denominator; four is the count the live run exposed.
CLIENT_COUNTS = (1, 4)


# ---------------------------------------------------------------------------
# Fixtures and the four synthetic drivers
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="g1-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


@pytest.fixture
def dataset():
    return tiny_dataset()


def written(ctx: RunContext) -> List[Dict[str, Any]]:
    return list(records.read_records(ctx.recorder.path))


def loaded(dataset) -> FakeDriver:
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                        ef_search=32, ef_half=64.0)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train[:N_ROWS], list(range(N_ROWS)))
    return driver


def _spin(seconds: float) -> None:
    """Burn `seconds` of CPU without a call that releases the lock.

    A pure-Python deadline loop. Other threads run only when the
    interpreter's switch interval expires, and they then have to take the lock
    back -- which is the regime a client library that does not release it
    around its socket wait puts this pool in.
    """
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def releasing(_clients: int) -> Callable[[], None]:
    """A query that RELEASES the lock for its whole cost.

    `time.sleep` is the stdlib's lock-releasing wait. This is the regime a
    driver is in if its client library drops the lock around the socket, and
    the pool must scale in it.
    """
    return lambda: time.sleep(QUERY_S)


def holding(_clients: int) -> Callable[[], None]:
    """A query that HOLDS the lock for its whole cost."""
    return lambda: _spin(QUERY_S)


def saturated(_clients: int) -> Callable[[], None]:
    """A query against a server that IS the bottleneck.

    The queue is the SERVER's, so a query waits as long as there are queries
    in flight, and the client is blocked on the socket for all of it. The
    depth is counted here rather than taken from the pool's client count,
    because the sweep builds one factory for every point: a cost fixed at the
    largest client count made the one-client window pay it too, and the
    modelled efficiency came out 1.00 instead of 1/4 -- the row would have
    protected nothing.

    With n clients looping, each sees depth n and waits n x QUERY_S, so each
    manages 1/(n x QUERY_S) and the pool manages 1/QUERY_S whatever n is:
    flat throughput and an efficiency of about 1/n. That is exactly the shape
    the live run showed -- but every client really is waiting on the engine,
    and **this is the case the guard must not fire on**.

    **THE QUEUE IS A `multiprocessing.Value` AND WAS A `threading.Lock`, AND
    THAT CHANGE IS THE MOST DANGEROUS LINE IN G3's FOLLOW-UP.** G3 made the
    clients OS processes. A `threading.Lock` and a `[0]` are per-process
    objects, so under fork every client would have seen its own queue at
    depth one, every query would have cost the base latency, the pool would
    have scaled -- and the row would have passed while asserting nothing
    about the case the whole threshold exists to protect. Measured after the
    change: efficiency 0.272 at four clients with `served / clients` 1.004,
    which is a server that IS the bottleneck and is correctly not flagged.
    The counter is created BEFORE any fork, so every client process shares
    the one queue and the regime still models one server.
    """
    depth = multiprocessing.get_context("fork").Value("i", 0)

    def spend():
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        try:
            time.sleep(QUERY_S * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


def half_and_half(_clients: int) -> Callable[[], None]:
    """Half released, half held: a partial-serialization control.

    Here so the threshold is not calibrated against two extremes alone. It
    halves the throughput at four clients and still achieves the concurrency,
    because the lock contention lands in the latency rather than in the number
    of queries in service -- which is what the refusal is about.
    """
    def spend():
        time.sleep(QUERY_S / 2.0)
        _spin(QUERY_S / 2.0)
    return spend


DRIVERS = {"releasing": releasing, "holding": holding,
           "saturated": saturated, "half": half_and_half}

#: Which families the LIVE pool must be able to run at the client count.
#: Every key of `DRIVERS` appears here, which is what makes a family added
#: above a test failure rather than a row nothing runs -- asserted by
#: `tests/test_g4_pool_teardown.py::TestEveryDriverFamilyIsStillClassified`.
#:
#: **ALL FOUR, SINCE G3, AND THAT IS THE CHANGE RATHER THAN A LOOSENING.**
#: `holding` was in `SERIALIZES` until the clients became OS processes: it
#: models the defect by holding the interpreter lock, and separate
#: interpreters do not share one. The regime still serializes a THREAD pool
#: and `tests/retired_thread_pool.py` keeps that demonstrable -- see
#: `SERIALIZES_UNDER_THREADS` below and the class that uses it. Deleting the
#: row instead would have thrown away the only thing in the tree that can
#: still produce the shape the guard exists for.
SCALES = ("releasing", "saturated", "half", "holding")

#: The families that serialize the RETIRED thread pool, which is where the
#: refusal is still demonstrated end to end. Empty for the live pool by
#: construction: `concurrency_not_achieved` is now close to unreachable
#: through Python's own lock, which is why the guard has become a tripwire
#: rather than an expected finding -- `tests/retired_thread_pool.py`'s
#: docstring has the argument.
SERIALIZES_UNDER_THREADS = ("holding",)

#: **`SERIALIZES = ()` is GONE rather than kept empty.** It used to name the
#: families that serialize the live pool and it carried the comment "Named
#: here so a family added above has to be classified" -- which stopped being
#: true twice over: no `parametrize` referenced it any more, so an added
#: family was classified by `SCALES` alone, and an empty tuple parametrizes
#: nothing, so even a reference would have run no test. An empty collection
#: that reads as a classification is the dead-probe shape this file records
#: elsewhere, so the classification duty it claimed is now discharged by an
#: assertion that every `DRIVERS` key is in `SCALES`.


#: What every synthetic query returns. Fixed ids rather than the fake's real
#: neighbours, and **that is load-bearing rather than a shortcut**:
#: `FakeDriver.query` runs a NumPy scan of the corpus, NumPy releases the
#: interpreter lock inside it, and that release sits INSIDE the span the
#: instrument sums. Measured: with the fake's own body left in the timed
#: region, a four-client lock-holding pool had every span interrupted at the
#: NumPy call, its p50 rose from 1 ms to 4.796 ms and the pool read `ok` --
#: the lock-holding model was not holding the lock. Recall is 0.31 on these
#: ids and no assertion in this file reads it.
SYNTHETIC_IDS = list(range(K))


def factory_for(base: FakeDriver, spend: Callable[[], None]):
    """A driver factory whose clones spend `spend()` and return fixed ids.

    The cost is bound on the INSTANCE rather than on a subclass, and the
    pattern is `tests/test_workloads.py::_counting`'s: `FakeDriver.clone`
    returns a plain `FakeDriver`, so a subclass's override would not reach the
    clients the pool actually builds. `Client.run` calls `driver.query(...)`,
    which finds the instance attribute first.

    Inside the timed region deliberately: `clients.Client.run` brackets
    `driver.query` alone, so a cost paid around the call would not appear in
    the spans the instrument sums -- and the spans are the whole input. See
    `SYNTHETIC_IDS` for why the fake's own query body is NOT called in there.
    """
    def clone():
        client = base.clone()

        def query(vector, k):
            spend()
            return SYNTHETIC_IDS[:k]

        client.query = query
        return client
    return clone


def run_sweep(ctx, dataset, base, family, counts=CLIENT_COUNTS):
    """One concurrency sweep of one driver family, through the real workload.

    One factory for the whole sweep, as a real run has: the drivers above
    therefore have to behave correctly at every client count the sweep visits
    rather than at a count handed to them. `saturated` is the one that cared,
    and `saturated`'s docstring records what a count-at-construction version
    measured.
    """
    conc.run(ctx, factory_for(base, DRIVERS[family](max(counts))),
             dataset, index_spec(dataset), ef_search=32,
             client_counts=counts, duration_s=WINDOW_S, warmup=2,
             max_queries=N_QUERIES, indexed_rows=N_ROWS)
    return written(ctx)


def pool_window(base, family, clients, queries, window_s=WINDOW_S):
    """One `ClientPool` window, driven directly rather than through `run`.

    Used where the assertion is about the window object and not about the
    record, so the test does not pay for ground truth and recall scoring.
    """
    pool = cl.ClientPool(factory_for(base, DRIVERS[family](clients)),
                         queries, clients, K, 32,
                         IndexSpec(dim=queries.shape[1], m=16,
                                   metric="euclidean"),
                         warmup=2)
    pool.start(f"{family} {clients}-client warm-up", prefix="g1")
    time.sleep(window_s)
    return pool


# ---------------------------------------------------------------------------
# The instrument itself, over hand-built spans
# ---------------------------------------------------------------------------

class TestTheInstrumentIsTheArithmeticItClaims:
    """`window_concurrency` over spans chosen by hand, so the expected value
    is a sum and a median rather than a measurement."""

    @staticmethod
    def _collector(spans):
        """A collector holding exactly `spans`, as (started, finished)."""
        collector = LatencyCollector(warmup=0)
        for started, finished in spans:
            collector.add_span(started, finished)
        return collector

    def test_outstanding_is_the_span_sum_over_the_wall(self):
        """Four 1 s queries inside a 2 s window is two outstanding.

        Honest: 2.0. A mutant dividing by the span COUNT instead of the wall
        -- the shape of a mean-latency implementation -- publishes 1.0 over
        the same input, and the assertion below reads 2.0 back out.
        """
        spans = [(0.0, 1.0), (0.0, 1.0), (1.0, 2.0), (1.0, 2.0)]
        got = window_concurrency([self._collector(spans)], 2.0,
                                 window_start=0.0, cutoff=2.0)
        assert got.outstanding == pytest.approx(2.0)
        # Not the mean latency, and not the count over the wall: both are 1.0
        # and 2.0 respectively on this input, so a wrong denominator is
        # distinguishable here.
        assert got.outstanding != pytest.approx(1.0)

    def test_served_is_the_count_times_the_median_over_the_wall(self):
        """The median, not the mean, and the input separates them.

        Three spans of 0.1 s and one of 10 s in a 10 s window: the mean is
        2.575 s and the median is 0.1 s. Honest `served` is
        4 x 0.1 / 10 = 0.04; a mutant using the mean publishes 1.03 -- a
        twenty-six-fold difference, and it is the difference between seeing a
        serialized pool and not.
        """
        spans = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.0, 10.0)]
        got = window_concurrency([self._collector(spans)], 10.0,
                                 window_start=0.0, cutoff=10.0)
        assert got.served == pytest.approx(4 * 0.1 / 10.0)
        durations = [f - s for s, f in spans]
        mean_form = len(durations) * statistics.mean(durations) / 10.0
        assert mean_form == pytest.approx(got.outstanding)
        assert got.served != pytest.approx(mean_form)

    def test_the_median_is_the_one_the_record_publishes_as_p50(self):
        """`served` has to be checkable as `queries_executed x p50 / window`.

        A reader given a refusal must be able to redo it from the published
        columns. An interpolating median here would disagree with
        `latency_p50_ms` -- which is nearest-rank -- and the check would read
        as a bug in the guard. Four spans: nearest-rank p50 is the second
        value, an interpolating one is the mean of the middle two.
        """
        spans = [(0.0, 0.010), (0.0, 0.020), (0.0, 0.040), (0.0, 0.080)]
        collector = self._collector(spans)
        merged = merge_collectors([collector], 1.0, cutoff=1.0,
                                  window_start=0.0)
        got = window_concurrency([collector], 1.0, cutoff=1.0,
                                 window_start=0.0)
        rebuilt = (merged["queries_executed"] * merged["latency_p50_ms"]
                   / 1000.0 / 1.0)
        assert got.served == pytest.approx(rebuilt, rel=1e-6)
        # Non-vacuity: the interpolating median would be 0.030 and is not what
        # either side of the equality above used.
        assert merged["latency_p50_ms"] == pytest.approx(20.0)

    def test_both_figures_are_none_rather_than_zero_for_an_empty_window(self):
        """Zero is the claim "the clients had nothing in flight".

        That cannot be true of a window that executed queries, and a 0.0 here
        would be read as total serialization by a guard looking for exactly
        that. None propagates to a status of None, which the report renders as
        an em dash rather than as a refusal.
        """
        empty = window_concurrency([self._collector([])], 1.0)
        assert empty.outstanding is None and empty.served is None
        degenerate = window_concurrency(
            [self._collector([(0.0, 0.1)])], 0.0)
        assert degenerate.outstanding is None and degenerate.served is None

    def test_the_window_rule_is_the_merges_rule_and_not_a_second_one(self):
        """One `span_disposition` decides which spans both of them count.

        Two definitions of "inside the window" is how a published recall comes
        to describe a different query set from the throughput beside it, which
        this repository has already done. The input holds one in-window span,
        one that outlived the cutoff and one that started before the window:
        the merge keeps one, and so must this.

        Honest: `queries_executed` 1 and `served` 0.1. A mutant summing every
        span regardless -- the obvious implementation -- publishes `served`
        0.3 and `outstanding` 2.1 over the same input.
        """
        spans = [(0.1, 0.2),      # inside
                 (0.5, 1.5),      # finishes after the cutoff
                 (-0.5, 0.3)]     # started before the window opened
        collector = self._collector(spans)
        merged = merge_collectors([collector], 1.0, cutoff=1.0,
                                  window_start=0.0)
        got = window_concurrency([collector], 1.0, cutoff=1.0,
                                 window_start=0.0)
        assert merged["queries_executed"] == 1
        assert merged["inflight_discarded"] == 1
        assert merged["prewindow_discarded"] == 1
        assert got.served == pytest.approx(0.1, rel=1e-6)
        assert got.outstanding == pytest.approx(0.1, rel=1e-6)
        # The mutant, run here rather than described: no bounds, so every
        # span counts. Honest `outstanding` 0.1 against a mutant 1.9, and
        # honest `served` 0.1 against a mutant 2.4 -- the median of
        # (0.1, 0.8, 1.0) times three.
        untrimmed = window_concurrency([collector], 1.0)
        assert untrimmed.outstanding == pytest.approx(1.9, rel=1e-6)
        assert untrimmed.served == pytest.approx(2.4, rel=1e-6)

    def test_several_clients_collectors_are_one_window(self):
        """The merge is over one shared wall, and so is this.

        Four clients each holding one 1 s span across a 1 s window is four
        outstanding, not one per collector.
        """
        collectors = [self._collector([(0.0, 1.0)]) for _ in range(4)]
        got = window_concurrency(collectors, 1.0, window_start=0.0,
                                 cutoff=1.0)
        assert got.outstanding == pytest.approx(4.0)
        assert got.served == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The status, over a hand-built window
# ---------------------------------------------------------------------------

class TestTheStatusIsTheComparisonItClaims:
    """`PoolWindow.scaling_status` against the floor, with no pool involved.

    Hand-built, and said so: which side of a floor a number falls on is a
    property of the comparison, not of any engine. The end-to-end classes
    below measure the real thing.
    """

    @staticmethod
    def _window(served, clients=4):
        kwargs = {field: None for field in cl.PoolWindow._fields}
        kwargs["served_concurrency"] = served
        kwargs["spec"] = cl.PoolSpec(clients=clients, queries=100, k=K,
                                     ef_search=32)
        return cl.PoolWindow(**kwargs)

    def test_a_pool_that_served_its_clients_is_ok(self):
        assert self._window(4.0).scaling_status == cl.SCALING_OK
        assert self._window(3.9).scaling_status == cl.SCALING_OK

    def test_a_pool_that_served_about_one_is_refused(self):
        window = self._window(0.95)
        assert (window.scaling_status
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)
        assert window.served_fraction == pytest.approx(0.2375)

    def test_the_floor_is_exactly_the_boundary_and_not_approximately(self):
        """On the floor is `ok`; a hair below it is not.

        Stated as a pair because a `<=` where a `<` belongs moves the verdict
        for exactly one value, and that is the edit a reader would make while
        tidying.
        """
        on = cl.SERVED_CONCURRENCY_FLOOR * 4
        assert self._window(on).scaling_status == cl.SCALING_OK
        assert (self._window(on * 0.999).scaling_status
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)

    def test_an_unmeasured_window_is_none_rather_than_ok(self):
        """None is "nobody could look", which is not "the pool was fine".

        The same distinction `client_bound` draws for an unreadable CPU
        counter, and the report has to render the two differently or the
        absence reads as a pass.
        """
        assert self._window(None).scaling_status is None
        assert self._window(None).served_fraction is None

    def test_the_fraction_is_against_the_client_count(self):
        """The denominator, read back: the same served figure is fine at one
        client and refused at eight.

        A mutant dividing by the query count, or by nothing at all, cannot
        produce both of these from one input.
        """
        assert self._window(0.95, clients=1).scaling_status == cl.SCALING_OK
        assert (self._window(0.95, clients=8).scaling_status
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)

    def test_the_floor_is_load_bearing_and_not_decoration(self):
        """Runtime mutation of the floor, with both values read back.

        Honest (`SERVED_CONCURRENCY_FLOOR` = 0.5): a window serving 0.95 of
        four clients is `concurrency_not_achieved`. Mutant (floor rebound to
        0.0 **on `harness.metrics.scaling`**): the SAME window reads `ok`.
        `scaling.scaling_status` looks the constant up in its own module
        globals on every call, so the rebinding reaches it -- a floor captured
        at import time, or copied into a default argument, would make the
        mutant pass unchanged.

        **The target moved from `latency` to `scaling` when the policy did.**
        `latency.py` re-exports the floor and a re-export does not share the
        name, so the old spelling of this probe is now DEAD -- which is what
        `test_rebinding_the_re_export_reaches_nothing` below records.
        """
        window = self._window(0.95)
        honest = window.scaling_status
        original = scaling_mod.SERVED_CONCURRENCY_FLOOR
        try:
            scaling_mod.SERVED_CONCURRENCY_FLOOR = 0.0
            mutant = window.scaling_status
        finally:
            scaling_mod.SERVED_CONCURRENCY_FLOOR = original
        assert honest == cl.SCALING_CONCURRENCY_NOT_ACHIEVED
        assert mutant == cl.SCALING_OK
        assert honest != mutant
        assert scaling_mod.SERVED_CONCURRENCY_FLOOR == original

    def test_rebinding_the_re_export_reaches_nothing(self):
        """The dead probe this layout creates, asserted rather than left to
        be discovered by an agent reporting a clean mutant.

        `clients.py` re-exports `SERVED_CONCURRENCY_FLOOR`, and a re-export
        does not SHARE the name: the rule reads it out of
        `harness.metrics.scaling`. Honest and mutant are the SAME status here,
        which is the signature of a mutation that measured nothing -- so it is
        recorded as such instead of being mistaken for a guard that holds.

        **THERE ARE TWO DEAD SPELLINGS NOW AND BOTH ARE READ BACK.** The
        policy moved to `scaling.py`, so `harness.metrics.latency` re-exports
        it too and a mutation aimed there is equally dead -- it was a LIVE
        target before the split, which is exactly the kind of probe a move
        can silently neuter.
        """
        window = self._window(0.95)
        honest = window.scaling_status
        for module in (cl, latency_mod):
            original = module.SERVED_CONCURRENCY_FLOOR
            try:
                module.SERVED_CONCURRENCY_FLOOR = 0.0
                dead = window.scaling_status
            finally:
                module.SERVED_CONCURRENCY_FLOOR = original
            assert honest == dead == cl.SCALING_CONCURRENCY_NOT_ACHIEVED, \
                module.__name__
            assert module.SERVED_CONCURRENCY_FLOOR == original
        # And every spelling agrees in VALUE, so the re-exports are not second
        # constants that could drift from the one the rule reads.
        assert (cl.SERVED_CONCURRENCY_FLOOR
                == latency_mod.SERVED_CONCURRENCY_FLOOR
                == scaling_mod.SERVED_CONCURRENCY_FLOOR)


# ---------------------------------------------------------------------------
# The measurement: four driver families through the real pool
# ---------------------------------------------------------------------------

class TestTheExactFigureIsBlindAndTheRobustOneIsNot:
    """The reason the refusal is keyed to `served_concurrency`.

    `outstanding_concurrency` is the exact time-average number of queries
    outstanding -- the integral of the occupancy function over the window,
    with no estimator in it. It is also blind to the defect this file exists
    for, and that is a measurement rather than an argument: a span brackets
    `driver.query`, so it covers whatever the query spent WAITING as well as
    whatever it spent being served, and the stretched spans tile the wall
    until the sum approaches the client count however little was served.
    On the retired arm below the wait is the interpreter lock, which the
    span's start instant cannot exclude because it is read in Python and the
    lock can be dropped at any bytecode boundary after it; under processes
    the blindness is the same and the wait is a queue in the engine.

    Both figures are read off the same window here, so the pair is one
    measurement and not two runs that might differ for another reason.
    """

    @pytest.fixture(scope="class")
    def queries(self):
        return tiny_dataset().test[:N_QUERIES]

    def _window(self, dataset, family, clients, queries):
        return pool_window(loaded(dataset), family, clients, queries).stop()

    def test_the_serializing_pool_is_outstanding_at_its_client_count(
            self, dataset, queries):
        """The blindness, in the one case that matters -- on the RETIRED arm.

        Four lock-holding clients: `served_concurrency` lands near one and the
        status refuses, while `outstanding_concurrency` sits near four. Both
        are read off the same window, so this cannot be two different runs.

        **Measured through `tests/retired_thread_pool.py` and no longer
        through the live pool, and that is G3 rather than a weakening.** The
        live clients are OS processes, so a driver that models the defect by
        holding the interpreter lock no longer serializes them -- measured,
        `served / clients` 0.243 under threads against 0.982 under processes
        for the identical regime. The shape this test is about has become
        unreachable through Python's own lock, which is precisely why the
        retired pool is kept: a tripwire nothing can trip cannot be
        distinguished from a broken one. `tests/test_g3_process_clients.py`
        reports both arms side by side.
        """
        window = thread_window(factory_for(loaded(dataset),
                                           DRIVERS["holding"](4)),
                               queries, 4, K, WINDOW_S)
        served, outstanding = window["served"], window["outstanding"]
        assert (latency_mod.scaling_status(served, 4)
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED), window
        # The margin measured on this box was a factor of four. Half of the
        # client count is the floor, so 2.0 is the line; the exact figure sat
        # at 0.84-1.00 of the client count on every run.
        assert served < 2.0, served
        assert outstanding > 2.0, outstanding
        # The pair, as a relation rather than as two absolutes: the exact
        # figure is at least twice the robust one, which is the signature of
        # time spent waiting for a lock rather than for the engine. Measured
        # at about three to four times.
        assert outstanding > 2.0 * served, window

    def test_the_live_pool_no_longer_reaches_that_state(self, dataset,
                                                        queries):
        """The other half of the row above, so the move is a measurement.

        The SAME regime through the live pool: `served_concurrency` comes back
        at its client count and the status is `ok`. Honest (processes) and
        mutant (threads) are the two arms, over one driver built once, so the
        pair cannot be two runs that differed for another reason.

        This is also the assertion that would catch a revert to threads, which
        is the first thing the tripwire is for.
        """
        driver = factory_for(loaded(dataset), DRIVERS["holding"](4))
        live = cl.ClientPool(driver, queries, 4, K, 32,
                             IndexSpec(dim=queries.shape[1], m=16,
                                       metric="euclidean"), warmup=2)
        live.start("g1 holding, live pool", prefix="g1")
        time.sleep(WINDOW_S)
        window = live.stop()
        retired = thread_window(driver, queries, 4, K, WINDOW_S)
        assert window.scaling_status == cl.SCALING_OK, window
        assert window.served_concurrency > 2.0, window.served_concurrency
        # Not an identical pair, which is the signature of a dead probe.
        assert window.served_concurrency > 2.0 * retired["served"], (
            window.served_concurrency, retired["served"])

    @pytest.mark.parametrize("family", ["releasing", "saturated", "holding"])
    def test_the_exact_figure_does_not_separate_the_regimes(
            self, dataset, queries, family):
        """One assertion, three regimes, and it passes for all three.

        That is the point: the exact figure reads within a factor of two of
        the client count whether the pool delivered four clients' work
        (`releasing`), one client's work against a saturated server
        (`saturated`) or one client's work because the clients took turns
        (`holding`). A guard built on it would fire on none of them.
        """
        window = self._window(dataset, family, 4, queries)
        assert window.outstanding_concurrency > 2.0, (
            family, window.outstanding_concurrency)


class TestTheGuardFiresOnSerializationAndNotOnASaturatedServer:
    """The whole sweep, through `concurrency.run`, with the records read back.

    Four driver families of IDENTICAL per-query cost, so the only difference
    between them is what they do with the interpreter lock -- and, for
    `saturated`, that the cost rises with the client count the way a real
    queue's does.
    """

    @staticmethod
    def _point(rows, clients):
        matching = [r for r in rows if r["clients"] == clients]
        assert len(matching) == 1, [r["clients"] for r in rows]
        return matching[0]

    @pytest.mark.parametrize("family", SERIALIZES_UNDER_THREADS)
    def test_a_serializing_pool_is_refused(self, ctx, dataset, family,
                                           monkeypatch):
        """Honest: the record carries `concurrency_not_achieved` and the
        report publishes no efficiency. The `scaling_efficiency` the record
        still holds was 0.21-0.30 across runs, which is why nothing here pins
        it: the refusal is the assertion, not the decimal.

        **The serialization is supplied by the RETIRED pool rather than by the
        live one, because the live one cannot produce it any more.** The
        workload's own window is replaced for the duration with a reading
        taken from `tests/retired_thread_pool.py` over the same driver family
        -- so what travels into the record and out to the report is a real
        measurement of a real serializing pool, and only the question of which
        pool produced it has changed. Patched on
        `harness.workloads.clients`, which is where `ClientPool.stop` reads
        the name; patching `harness.metrics.latency` would reach nothing, and
        `TestTheMedianIsWhatMakesTheGuardWork` asserts that asymmetry.
        """
        retired = thread_window(factory_for(loaded(dataset),
                                            DRIVERS[family](4)),
                                tiny_dataset().test[:N_QUERIES], 4, K,
                                WINDOW_S)
        assert retired["served"] < 2.0, retired
        real = cl.window_concurrency

        def from_the_retired_pool(collectors, wall_seconds, cutoff=None,
                                  window_start=None):
            measured = real(collectors, wall_seconds, cutoff=cutoff,
                            window_start=window_start)
            if measured.served is None:
                return measured
            # BOTH figures since ops version 9, when the guard moved to
            # `outstanding`: the retired pool's own `outstanding` stayed at
            # the client count (the lock wait sat inside its spans, which
            # is why the median was used), so the splice now models what a
            # serializing pool looks like to the version-9 rule -- clients
            # that did not keep their queries in flight.
            return measured._replace(served=retired["served"],
                                     outstanding=retired["served"])

        monkeypatch.setattr(cl, "window_concurrency", from_the_retired_pool)
        rows = run_sweep(ctx, dataset, loaded(dataset), family)
        four = self._point(rows, 4)["extra"]
        assert four["scaling_status"] == cl.SCALING_CONCURRENCY_NOT_ACHIEVED
        assert four["served_concurrency"] < 2.0, four
        # The efficiency is still IN the record. Deleting it would lose the
        # evidence for the refusal; the report is what declines to print it.
        assert four["scaling_efficiency"] is not None

    @pytest.mark.parametrize("family", SCALES)
    def test_a_pool_that_achieves_its_concurrency_is_not_refused(
            self, ctx, dataset, family):
        """**The case the threshold exists to protect.**

        `saturated` is a server that IS the bottleneck: its efficiency falls
        to about 1/clients, which is the shape a reader would call an engine
        ceiling, and it is one. It must not be flagged. `releasing` scales and
        `half` halves its throughput to lock contention that lands in the
        latency rather than in the number of queries in service.

        A mutant that dropped the median from the quantity -- using the
        single-client median, or a fixed service time -- fires on `saturated`,
        and that is the failure mode this row is here to catch.
        """
        rows = run_sweep(ctx, dataset, loaded(dataset), family)
        four = self._point(rows, 4)["extra"]
        assert four["scaling_status"] == cl.SCALING_OK, four
        assert four["served_concurrency"] > 2.0, four

    def test_the_saturated_server_really_is_the_low_efficiency_case(
            self, ctx, dataset):
        """Non-vacuity for the row above.

        If `saturated` happened to scale, "not flagged" would be trivially
        true and the protection would be untested. So the same record's
        efficiency is read back and shown to be low: the guard is being
        offered a point that LOOKS exactly like the defect and is declining
        it. Measured here at 0.26-0.31 against a modelled 1/4.
        """
        rows = run_sweep(ctx, dataset, loaded(dataset), "saturated")
        four = self._point(rows, 4)["extra"]
        assert four["scaling_efficiency"] < 0.5, four
        assert four["scaling_status"] == cl.SCALING_OK, four

    def test_one_client_is_never_refused_by_its_own_baseline(self, ctx,
                                                             dataset):
        """The efficiency denominator must survive the guard.

        At one client the efficiency is 1.00 by construction. A floor that
        fired there would refuse every point of every sweep, including the one
        the others are divided by. Measured at 0.89-1.00 of one client across
        every family.
        """
        for family in DRIVERS:
            rows = run_sweep(ctx, dataset, loaded(dataset), family,
                             counts=(1,))
            one = rows[-1]["extra"]
            assert one["scaling_status"] == cl.SCALING_OK, (family, one)
            assert one["scaling_efficiency"] == 1.0

    def test_the_record_carries_both_figures_and_the_status(self, ctx,
                                                           dataset):
        """All three keys, on every point, rounded as the record rounds.

        The status alone would be a verdict a reader cannot check. Both
        figures beside it is the convention the client-CPU columns already
        follow: the numerator and the denominator travel with the quotient.
        """
        rows = run_sweep(ctx, dataset, loaded(dataset), "releasing")
        for row in rows:
            extra = row["extra"]
            for key in ("outstanding_concurrency", "served_concurrency",
                        "scaling_status"):
                assert key in extra, (row["clients"], sorted(extra))
            assert extra["outstanding_concurrency"] > 0.0
            assert extra["served_concurrency"] > 0.0
            assert extra["scaling_status"] in cl.SCALING_STATUSES
            # Rounded to three decimals, like every other derived figure in
            # this record: a raw float would carry the box's last digits into
            # a published JSONL.
            assert extra["served_concurrency"] == round(
                extra["served_concurrency"], 3)

    def test_the_operator_is_told_while_the_run_is_happening(
            self, ctx, dataset, capsys, monkeypatch):
        """On stderr, named, and with the efficiency printed as `refused`.

        The remedy is a different client library or a different measurement
        and neither is available to someone reading the report a day later.
        The live log that exposed this printed `efficiency=0.25` on a line
        whose every other figure said the pool had one client's worth of work
        in it, so the stdout line is asserted too.

        The serializing reading is SUPPLIED by `retired_thread_pool`, for the
        reason `test_a_serializing_pool_is_refused` gives: the live pool
        cannot produce it any more. The operator message is the tripwire's
        output, so it has to stay exercised by a real serializing window
        rather than by a hand-written float.
        """
        retired = thread_window(
            factory_for(loaded(dataset), DRIVERS["holding"](4)),
            tiny_dataset().test[:N_QUERIES], 4, K, WINDOW_S)
        assert retired["served"] < 2.0, retired
        real = cl.window_concurrency

        def from_the_retired_pool(collectors, wall_seconds, cutoff=None,
                                  window_start=None):
            measured = real(collectors, wall_seconds, cutoff=cutoff,
                            window_start=window_start)
            if measured.served is None:
                return measured
            # BOTH figures since ops version 9, when the guard moved to
            # `outstanding`: the retired pool's own `outstanding` stayed at
            # the client count (the lock wait sat inside its spans, which
            # is why the median was used), so the splice now models what a
            # serializing pool looks like to the version-9 rule -- clients
            # that did not keep their queries in flight.
            return measured._replace(served=retired["served"],
                                     outstanding=retired["served"])

        monkeypatch.setattr(cl, "window_concurrency", from_the_retired_pool)
        run_sweep(ctx, dataset, loaded(dataset), "holding")
        captured = capsys.readouterr()
        assert "CONCURRENCY NOT ACHIEVED" in captured.err, captured.err
        assert "4 client(s)" in captured.err, captured.err
        # Both readings named, because the measurement cannot separate them.
        assert "interpreter lock" in captured.err
        assert "one query at a time" in captured.err
        # And the per-point line does not print a number the report refuses.
        refused = [line for line in captured.out.splitlines()
                   if "clients=  4" in line]
        assert refused, captured.out
        assert "efficiency=refused" in refused[0], refused[0]

    def test_a_pool_that_achieves_its_concurrency_says_nothing(self, ctx,
                                                              dataset,
                                                              capsys):
        """The other branch, which is what makes the warning a measurement.

        A diagnostic that printed on every run would train an operator to
        skip it, after which it is also unread on the run where it matters.
        """
        run_sweep(ctx, dataset, loaded(dataset), "releasing")
        captured = capsys.readouterr()
        assert "CONCURRENCY NOT ACHIEVED" not in captured.err
        assert "efficiency=refused" not in captured.out


# ---------------------------------------------------------------------------
# The report refuses, and a reader can see what it refused
# ---------------------------------------------------------------------------

def _record(**over):
    """One concurrency record as the workload now writes it."""
    extra = {
        "scaling_efficiency": 0.25, "duration_s": 20,
        "warmup_queries_per_client": 50, "repeats": 1,
        "qps_samples": [2927.5], "qps_spread_pct": 0.0,
        "recall_queries": 10_000, "recall_spread": None,
        "query_pool": 10_000, "query_reuse": 5.9,
        "inflight_discarded": 4, "prewindow_discarded": 0,
        "client_cpu_seconds": 14.2, "client_cpus": 4,
        "client_cpu_util": 0.177, "client_bound": False,
        "outstanding_concurrency": 3.95, "served_concurrency": 0.951,
        "scaling_status": cl.SCALING_CONCURRENCY_NOT_ACHIEVED,
    }
    extra.update(over.pop("extra", {}))
    record = {"phase": "concurrency", "engine": "mariadb", "dataset": "d",
              "resource_pass": "normalized", "storage_engine": "InnoDB",
              "m": 16, "ef_search": 100, "clients": 4, "qps": 2927.5,
              "recall_at_k": 0.9986, "latency_p50_ms": 0.325,
              "latency_p95_ms": 0.439, "latency_p99_ms": 0.544,
              "queries_executed": 58_550}
    record.update(over)
    record["extra"] = extra
    return record


def _table_cells(table: str) -> List[List[str]]:
    """A rendered Markdown table as rows of cells.

    Cells rather than a substring search: "0.25" appears in several columns of
    this table and in the raw record, so a membership test against the whole
    fragment is satisfied by the number turning up somewhere else.
    """
    rows = []
    for line in table.strip().splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c and set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _plain(text: str) -> str:
    """Prose with its whitespace and emphasis normalised.

    A guard a reflow defeats is not a guard: these documents are rewrapped by
    hand and by editors, so a pattern matched against raw text fails on a line
    break that changed nothing. Emphasis markers go too, because moving a
    `**` is the other edit that changes no meaning.
    """
    return re.sub(r"\s+", " ", text.replace("**", "").replace("*", ""))


class TestTheReportRefusesTheFigure:
    def _cells(self, *rows):
        return _table_cells(
            render_mod._concurrency_table({"concurrency": list(rows)}))

    def test_the_efficiency_cell_is_a_refusal_and_not_a_number(self):
        """Honest (new form): the cell reads the refusal. The record's own
        0.25 appears nowhere in that row's efficiency column."""
        cells = self._cells(_record())
        header, body = cells[0], cells[1:]
        cell = body[0][header.index("Scaling efficiency")]
        assert cell == render_mod.SCALING_REFUSED_CELL
        assert "0.25" not in cell

    def test_the_pre_change_expression_published_the_number(self):
        """R2: the OLD form passes the mutant the NEW form catches.

        The expression this replaced, copied verbatim from the pre-change
        `_concurrency_table`:

            eff = extra.get("scaling_efficiency")
            _fmt(eff, 2) if eff is not None else "—"

        Run over the very same record it publishes **"0.25"**; the new form
        publishes **"⊘ not measurable"**. A repair whose old form also
        refuses is not a repair, and this is the line that says which it is.
        """
        extra = _record()["extra"]
        eff = extra.get("scaling_efficiency")
        old_form = render_mod._fmt(eff, 2) if eff is not None else "—"
        new_form = render_mod._efficiency_cell(extra)
        assert old_form == "0.25"
        assert new_form == render_mod.SCALING_REFUSED_CELL
        assert old_form != new_form

    def test_a_point_that_achieved_its_concurrency_still_publishes(self):
        """The other branch. A refusal that fired on every point would blank
        the column and read as a rendering fault."""
        ok = _record(extra={"scaling_status": cl.SCALING_OK,
                            "served_concurrency": 3.89})
        cells = self._cells(ok)
        header, body = cells[0], cells[1:]
        assert body[0][header.index("Scaling efficiency")] == "0.25"

    def test_a_record_predating_the_status_renders_as_it_always_did(self):
        """The key absent altogether, which is every archived run.

        Absence is not a refusal: treating it as one would blank the
        efficiency column of every run in the results tree, on a regeneration
        that measured nothing new.
        """
        old = _record()
        for key in ("scaling_status", "served_concurrency",
                    "outstanding_concurrency"):
            del old["extra"][key]
        cells = self._cells(old)
        header, body = cells[0], cells[1:]
        assert body[0][header.index("Scaling efficiency")] == "0.25"
        assert body[0][header.index("Queries in service")] == "—"

    def test_the_evidence_sits_beside_the_refusal(self):
        """What was served against what was offered, in the row itself.

        A refusal a reader cannot check is indistinguishable from a bug in
        the refusal, which is the argument the Client CPU column's numerator
        and denominator are kept for.
        """
        cells = self._cells(_record())
        header, body = cells[0], cells[1:]
        cell = body[0][header.index("Queries in service")]
        # Ops version 9: the exact mean-based figure first, the median
        # estimate second -- the order the guard now reads them in.
        assert cell.startswith("3.95 of 4 (mean)"), cell
        assert "0.95 median est." in cell, cell
        assert render_mod.SCALING_REFUSED_MARK in cell

    def test_every_body_cell_is_filled(self):
        """A column added to the header and not to the row shifts every cell
        after it, which renders as a plausible table of wrong numbers."""
        cells = self._cells(_record(), _record(clients=1))
        header, body = cells[0], cells[1:]
        assert len(body) == 2
        for row in body:
            assert len(row) == len(header), list(zip(header, row))

    def test_the_run_specific_note_names_the_points(self):
        notes = render_mod._concurrency_notes({"concurrency": [_record()]})
        assert "No scaling efficiency is published for 1 point(s)" in notes
        assert "at 4 clients" in notes

    def test_a_clean_run_gets_no_such_note(self):
        clean = _record(extra={"scaling_status": cl.SCALING_OK,
                               "served_concurrency": 3.89})
        notes = render_mod._concurrency_notes({"concurrency": [clean]})
        assert "No scaling efficiency is published" not in notes
        # The column's definition stays, because that is not a run-specific
        # remark.
        assert "**Queries in service**" in notes


class TestTheValiditySectionCarriesIt:
    """§2 Validity, not only the footnote under the table.

    Validity is the section a reader is told to read before the charts, and
    the concurrency chart is this report's headline. A caveat that lives only
    under the table it qualifies is a caveat read after the conclusion has
    been drawn.
    """

    def _section(self, *rows):
        return render_mod._validity_section(
            {"warnings": []}, {"concurrency": list(rows)})

    def test_the_section_appears_and_names_the_point(self):
        text = self._section(_record())
        assert "Scaling efficiencies this report will not publish" in text
        cells = _table_cells(text)
        header = cells[0]
        assert "Efficiency withheld" in header
        row = cells[1]
        assert row[header.index("Clients")] == "4"
        # The withheld figure is shown HERE and nowhere else, so the refusal
        # is checkable rather than merely asserted.
        assert row[header.index("Efficiency withheld")] == "0.25"
        # Renamed and reordered at ops version 9: mean-based first.
        assert header.index("In service (mean)") < header.index(
            "Median estimate")
        assert row[header.index("In service (mean)")] == "3.95"
        assert row[header.index("Median estimate")] == "0.95"

    def test_the_section_states_the_two_readings_and_the_discriminator(self):
        text = _plain(self._section(_record()))
        for claim in ("interpreter lock", "answers one query at a time",
                      "Client CPU"):
            assert claim in text, claim

    def test_the_section_says_the_throughput_is_still_real(self):
        """The QPS and the percentiles are not withdrawn.

        Only the division by the client count has no basis, and a reader who
        took the whole row as unmeasured would discard a real result.
        """
        text = _plain(self._section(_record()))
        assert "QPS and the latency percentiles are left in place and are real" \
            in text

    def test_a_clean_run_still_reports_no_validity_problems(self):
        """The regression this guards is specific and was nearly shipped.

        `_validity_section` chooses its "no validity problems were detected"
        ending by `parts` being empty, so a section that appended an empty
        string on a clean run would retire that sentence on every report --
        silently, because the section it replaced renders as nothing.
        """
        clean = _record(extra={"scaling_status": cl.SCALING_OK,
                               "served_concurrency": 3.89})
        text = self._section(clean)
        assert "No validity problems were detected" in text
        assert "will not publish" not in text

    def test_the_empty_string_mutant_breaks_that_ending(self):
        """The mutation that proves the line above is load-bearing.

        Honest: a clean run's Validity section ends with "No validity
        problems were detected". Mutant (`_scaling_refused_section` forced to
        return a space, which is what an unconditional append of an empty
        section amounts to): the same input renders **without** that
        sentence. Patched at run time on the module the renderer looks the
        name up in.
        """
        clean = _record(extra={"scaling_status": cl.SCALING_OK,
                               "served_concurrency": 3.89})
        honest = self._section(clean)
        original = render_mod._scaling_refused_section
        try:
            render_mod._scaling_refused_section = lambda summary: " "
            mutant = self._section(clean)
        finally:
            render_mod._scaling_refused_section = original
        assert "No validity problems were detected" in honest
        assert "No validity problems were detected" not in mutant


class TestOneConditionIsReadByEveryRenderer:
    def test_the_table_the_notes_and_the_validity_section_agree(self):
        """Three readers, one predicate.

        Each deciding the condition for itself is how a chart comes to
        disagree with the table beside it, which reads as a rendering bug
        rather than as the measurement it is.
        """
        summary = {"concurrency": [_record()]}
        assert len(render_mod._scaling_refused(summary)) == 1
        cells = _table_cells(render_mod._concurrency_table(summary))
        header, body = cells[0], cells[1:]
        assert (body[0][header.index("Scaling efficiency")]
                == render_mod.SCALING_REFUSED_CELL)
        assert "will not publish" in render_mod._validity_section(
            {"warnings": []}, summary)

    def test_the_harness_and_the_report_spell_the_status_the_same(self):
        """The string is declared in two modules, which is how they drift.

        Not because the import would fail: `tests/test_config.py`'s
        `TestReportNarrowing` records that the report container mounts
        `report/` AND `harness/`, and `report/render.py` already imports
        `harness.metrics.stats` -- inside function bodies, deliberately,
        because `harness.workloads.clients` pulls numpy and every driver
        module in to obtain one string. So the declaration is duplicated on
        purpose and this is the pin that keeps the two equal. The direction
        that genuinely cannot import is the other one, which is why
        `churn_readers.OPS_UNIT_KEYS` is re-declared there: the ops harness
        container mounts `harness/` alone.
        """
        assert (render_mod.SCALING_CONCURRENCY_NOT_ACHIEVED
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)

    def test_the_workload_compares_against_the_same_string(self):
        """And `concurrency.py` imports the name rather than retyping it.

        A from-imported copy does not share the original -- rebinding
        `clients.SCALING_CONCURRENCY_NOT_ACHIEVED` would not reach it -- so
        what matters is that the VALUE agrees, which is what is asserted.
        """
        assert (conc.SCALING_CONCURRENCY_NOT_ACHIEVED
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)
        assert (crw.SCALING_CONCURRENCY_NOT_ACHIEVED
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)


class TestTheChartDoesNotPlotARefusedEfficiency:
    """The efficiency panel is drawn against a linear-scaling reference line.

    A point placed on it asserts that N clients delivered this share of N
    clients' work. For a pool that served about one query that assertion is
    false whatever marker it carries, so the point is omitted rather than
    hollowed -- a hollow point still sits on the line to be read off.
    """

    def _axes(self, rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
        from report import charts
        captured = {}
        original = charts._save

        def spy(fig, out_dir, stem):
            captured["axes"] = list(fig.axes)
            return original(fig, out_dir, stem)

        charts._save = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                charts.concurrency(rows, "d", d, "t")
        finally:
            charts._save = original
            plt.close("all")
        return captured

    def _series(self, refused_at=()):
        return [_record(
            clients=c, qps=1000.0 * (1 if c in refused_at else c),
            extra={"scaling_efficiency": 1.0 / c if c in refused_at else 1.0,
                   "served_concurrency": 0.95 if c in refused_at else float(c),
                   "scaling_status": (cl.SCALING_CONCURRENCY_NOT_ACHIEVED
                                      if c in refused_at else cl.SCALING_OK)})
                for c in (1, 2, 4, 8)]

    @staticmethod
    def _xs(ax):
        out = []
        for line in ax.lines:
            xs = list(line.get_xdata())
            # The linear-scaling reference is an axhline with no client xs.
            if len(xs) > 1 and max(xs) > 1:
                out.append([float(x) for x in xs])
        return out

    @needs_matplotlib
    def test_a_refused_point_is_absent_from_the_efficiency_panel(self):
        """Honest: the efficiency panel plots 1, 2, 8 and not 4. The
        throughput panel plots all four, because that IS a measurement."""
        captured = self._axes(self._series(refused_at=(4,)))
        throughput, _latency, efficiency = captured["axes"]
        assert self._xs(throughput) == [[1.0, 2.0, 4.0, 8.0]]
        assert self._xs(efficiency) == [[1.0, 2.0, 8.0]]

    @needs_matplotlib
    def test_without_a_refusal_the_panel_plots_every_point(self):
        """Non-vacuity: the omission follows the status, not the series."""
        captured = self._axes(self._series())
        _throughput, _latency, efficiency = captured["axes"]
        assert self._xs(efficiency) == [[1.0, 2.0, 4.0, 8.0]]

    @needs_matplotlib
    def test_no_hollow_ring_is_drawn_on_a_refused_efficiency(self):
        """A point can be both client-bound and refused, and the refusal wins
        on this panel: a ring would put the omitted point back on the line.

        Honest: no offsets on the efficiency panel. A mutant that hollowed it
        instead of omitting it draws one at (4, 0.25).
        """
        rows = self._series(refused_at=(4,))
        for row in rows:
            if row["clients"] == 4:
                row["extra"]["client_bound"] = True
                row["extra"]["client_cpu_util"] = 0.95
        captured = self._axes(rows)
        throughput, _latency, efficiency = captured["axes"]
        eff_offsets = [list(map(float, p))
                       for coll in efficiency.collections
                       for p in coll.get_offsets()]
        assert eff_offsets == []
        # The throughput panel still rings it: a client-bound throughput is a
        # measurement with a caveat, which is a different thing.
        qps_offsets = [list(map(float, p))
                       for coll in throughput.collections
                       for p in coll.get_offsets()]
        assert qps_offsets == [[4.0, 1000.0]]


# ---------------------------------------------------------------------------
# The client-CPU flag is kept, and says what it cannot see
# ---------------------------------------------------------------------------

class TestTheClientBoundFlagIsKeptAndDocumentsItsBlindness:
    """§3. The flag is not deleted -- it catches a different, real case -- but
    its docstring has to carry the measurement that says what it misses, or
    the next reader trusts it for this."""

    @staticmethod
    def _window(util):
        kwargs = {field: None for field in cl.PoolWindow._fields}
        kwargs["client_cpu_util"] = util
        return cl.PoolWindow(**kwargs)

    def test_the_flag_still_fires_on_a_saturated_cpuset(self):
        """Kept, not retired. A client container that genuinely runs out of
        CPU is a real case and `scaling_status` cannot name it as one."""
        assert self._window(0.93).client_bound is True
        assert self._window(0.1).client_bound is False
        assert self._window(None).client_bound is None

    def test_the_measured_blindness_is_read_back_out_of_the_flag(self):
        """0.177 against 0.80, called rather than quoted.

        Four clients on a lock-holding driver delivered 0.26 of linear while
        the client container's utilisation read 0.177. **That utilisation is
        SUPPLIED**: `read_own_cpu_usec` opens `/sys/fs/cgroup/cpu.stat`, which
        on a bare host is the root cgroup -- 2.421 cores measured while the
        reading process slept -- so no test here can produce it and nothing
        below claims to. What IS measured is the flag's answer at that input:
        **False**, read out of the property rather than asserted about it, so
        the point where three of four clients' throughput went missing is
        unmarked.
        """
        assert self._window(0.177).client_bound is False
        assert 0.177 < cl.CLIENT_BOUND_UTIL

    def test_the_docstring_carries_that_measurement(self):
        """R7: a claim in a comment is part of the deliverable.

        Both numbers and the direction, so a reader who finds the flag has
        the measurement beside it rather than a general warning. Whitespace
        normalised, because a reflow changes nothing and a guard a reflow
        defeats is not a guard.
        """
        doc = _plain(cl.PoolWindow.client_bound.__doc__ or "")
        assert "0.177" in doc
        assert "0.80" in doc
        assert "scaling_status" in doc
        # And it says the flag is kept for a reason rather than left behind.
        assert re.search(r"kept because it catches a different", doc, re.I)

    def test_the_threshold_comment_points_at_the_quantity_that_sees_it(self):
        """A reader who concludes "raise the threshold" has read it wrong, so
        the constant's own comment has to name the alternative."""
        # In `pool_window.py` since G3 split `clients.py`: the constant went
        # beside `PoolWindow.client_bound`, its only reader, which is the
        # placement `metrics/scaling.py` argues for `SERVED_CONCURRENCY_FLOOR`
        # -- that policy left `metrics/latency.py` when `latency.py` hit the
        # same 800-line ceiling that split `clients.py`. `clients.py`
        # re-exports it and that re-export does NOT share the name --
        # `tests/test_g3_process_clients.py` pins the asymmetry, and
        # `TestTheStatusIsTheComparisonItClaims` above pins the matching one
        # for the floor, where `latency.py` is now a dead patch target too.
        source = open(os.path.join(
            VB_ROOT, "harness", "workloads", "pool_window.py")).read()
        block = source.split("CLIENT_BOUND_UTIL = 0.80")[0]
        tail = _plain(block[-2200:])
        assert "0.177" in tail
        assert "SERVED_CONCURRENCY_FLOOR" in tail


# ---------------------------------------------------------------------------
# The churn reader path
# ---------------------------------------------------------------------------

class TestTheChurnReadersCarryItToo:
    """§4. `churn_readers` builds the same pool, and
    `extra.reinsert_rows_per_s` is a version-5 quantity measured against
    concurrent readers -- so a reader pool that served one read at a time
    leaves that rate describing an unknown amount of read load. Until version
    6 the readers were threads in the WRITER's own interpreter, which made it
    worse: readers that serialized starved the writer in the client and the
    rate became a measurement of Python. Each reader is a forked OS process
    now, so the dependency is what is left and the attribution is not.
    """

    @staticmethod
    def _window(served, clients=4, p99=10.0, outstanding=None):
        """A hand-built window. `outstanding` defaults to `served` -- a
        narrow service time, where the two figures agree -- because since
        ops version 9 a low `served` beside a full `outstanding` is
        `tail_dominated` and not a refusal; a reader pool that starved is one
        whose clients did not keep their reads in flight."""
        kwargs = {field: None for field in cl.PoolWindow._fields}
        kwargs.update(
            qps=100.0, wall_s=30.0, inflight_discarded=0,
            prewindow_discarded=0, client_cpus=4, client_cpu_util=0.18,
            client_cpu_seconds=21.6, results={},
            merged={"latency_p99_ms": p99, "latency_p50_ms": 1.0,
                    "queries_executed": 3000},
            served_concurrency=served,
            outstanding_concurrency=(served if outstanding is None
                                     else outstanding),
            spec=cl.PoolSpec(clients=clients, queries=100, k=K, ef_search=32))
        return cl.PoolWindow(**kwargs)

    def test_both_reader_windows_carry_the_figures(self):
        """Under both prefixes, because the ratio divides one window by the
        other and a reader cannot see from two percentiles that only one of
        the two pools was running concurrently."""
        extra = crw._during_writes_extra(
            self._window(3.9, p99=5.0), self._window(0.95, p99=20.0),
            12.0, 2000)
        for prefix in ("reader_baseline", "reader_under_writes"):
            assert f"{prefix}_served_concurrency" in extra
            assert f"{prefix}_achieved_concurrency" in extra
            assert f"{prefix}_scaling_status" in extra
        assert extra["reader_baseline_scaling_status"] == cl.SCALING_OK
        assert (extra["reader_under_writes_scaling_status"]
                == cl.SCALING_CONCURRENCY_NOT_ACHIEVED)

    def test_the_records_own_status_is_the_during_writes_windows(self):
        """Unprefixed, beside `client_bound`: every published figure on this
        record comes from the during-writes window, so that is the window
        whose status the record asserts."""
        extra = crw._during_writes_extra(
            self._window(3.9), self._window(0.95), 12.0, 2000)
        assert extra["scaling_status"] == cl.SCALING_CONCURRENCY_NOT_ACHIEVED
        assert extra["served_concurrency"] == 0.95
        assert extra["outstanding_concurrency"] == 0.95

    def test_the_operator_is_told_what_it_means_for_the_write_rate(self,
                                                                   capsys):
        """The warning names the writer, not only the reads.

        `writer_rows_per_s_with_readers` is the version-5 quantity, and the
        live run recorded 12.0 rows/s against a modelled ~150. A diagnostic
        about the reads alone would leave the published write rate looking
        unaffected.
        """
        extra = crw._during_writes_extra(
            self._window(3.9), self._window(0.95), 12.0, 2000)
        crw._print_during_writes("fake", 0.10, extra)
        err = capsys.readouterr().err
        assert "READERS DID NOT ACHIEVE CONCURRENCY" in err, err
        assert "12.0 rows/s" in err, err
        assert "starve the writer inside the client" in err, err

    def test_a_concurrent_reader_pool_says_nothing(self, capsys):
        extra = crw._during_writes_extra(
            self._window(3.9), self._window(3.8), 150.0, 2000)
        crw._print_during_writes("fake", 0.10, extra)
        assert "READERS DID NOT ACHIEVE" not in capsys.readouterr().err

    def test_the_step_records_write_rate_carries_the_readers_status(self):
        """`reader_contention_shape`, which is the point of item 4.

        `extra.reinsert_rows_per_s` is published on the STEP record, not on
        the reader record, so the status that decides whether it means
        anything has to reach it there.
        """
        shape = crw.reader_contention_shape(self._window(0.95))
        # Ops version 9 carries the mean-based figure the status is judged on
        # and the heavy-tail note beside the median estimate.
        assert shape == {"reader_outstanding_concurrency": 0.95,
                         "reader_served_concurrency": 0.95,
                         "reader_scaling_status":
                             cl.SCALING_CONCURRENCY_NOT_ACHIEVED,
                         "reader_scaling_note": None}

    def test_a_heavy_tailed_reader_window_is_noted_and_not_refused(self):
        """#78 review finding 2: the F2 artefact on the reader path. A pool
        whose four readers kept their reads in flight (outstanding 3.9) but
        whose median sat far below its mean (served 0.95) used to be refused,
        and the writer rate beside it marked invalid -- for one engine's tail
        and not another's. Honest (v9): `ok` + `tail_dominated`, on the step
        record and on the during-writes record. Mutant (the pre-v9 rule, the
        median alone): `concurrency_not_achieved`."""
        window = self._window(0.95, outstanding=3.9)
        shape = crw.reader_contention_shape(window)
        assert shape["reader_scaling_status"] == cl.SCALING_OK
        assert shape["reader_scaling_note"] == "tail_dominated"
        extra = crw._during_writes_extra(self._window(3.9), window, 12.0,
                                         2000)
        assert extra["scaling_status"] == cl.SCALING_OK
        assert extra["scaling_note"] == "tail_dominated"
        assert extra["reader_under_writes_scaling_note"] == "tail_dominated"
        from harness.metrics import scaling
        assert scaling.scaling_status(0.95, 4) == \
            cl.SCALING_CONCURRENCY_NOT_ACHIEVED

    def test_a_step_with_no_reader_pool_carries_no_key_at_all(self):
        """`{}` and not a null, which is `_reader_shape`'s own convention.

        A `reader_scaling_status: null` written onto a record from a run that
        had no reader pool would claim this code looked at something that
        never existed, and `tests/test_t4a_churn.py` pins the key set those
        records have.
        """
        assert crw.reader_contention_shape(None) == {}

    def test_the_churn_step_record_carries_it_end_to_end(self, ctx, dataset):
        """Through `churn.run` with a real reader pool, read off the record.

        The readers here hold the interpreter lock, which is the regime the
        live run was in. The assertion is the key's presence and agreement
        with the reader record, not the decimal: on this fake the same churn
        configuration's reader QPS has been measured over a factor of 2.79.
        """
        driver = loaded(dataset)
        import harness.workloads.churn as churn_mod
        original = churn_mod.READER_BASELINE_WINDOW_S
        churn_mod.READER_BASELINE_WINDOW_S = 0.3
        try:
            cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=32,
                   fractions=(0.10,), max_queries=N_QUERIES,
                   indexed_rows=N_ROWS,
                   driver_factory=factory_for(driver, holding(4)),
                   readers=4)
        finally:
            churn_mod.READER_BASELINE_WINDOW_S = original
        rows = written(ctx)
        steps = [r for r in rows
                 if r.get("churn_fraction")
                 and not (r.get("extra") or {}).get("during_writes")]
        during = [r for r in rows
                  if (r.get("extra") or {}).get("during_writes")]
        assert during, [r.get("phase") for r in rows]
        assert steps, [r.get("phase") for r in rows]
        step_extra = steps[0]["extra"]
        assert "reader_scaling_status" in step_extra, sorted(step_extra)
        # One status, two records: the write rate and the read figures must
        # not disagree about the regime they were measured in.
        assert (step_extra["reader_scaling_status"]
                == during[0]["extra"]["scaling_status"])
        assert step_extra["reinsert_rows_per_s"] is not None


class TestTheDuringWritesTableSaysIt:
    def _rows(self, status):
        return [{
            "engine": "mariadb", "dataset": "d", "resource_pass": "normalized",
            "m": 16, "build_mode": "incremental", "storage_engine": "InnoDB",
            "churn_fraction": 0.10, "readers": 4,
            "baseline_p99_ms": 0.55, "under_writes_p99_ms": 0.60,
            "p99_ratio": 1.09, "baseline_qps": 2920.0,
            "under_writes_qps": 2819.0, "writer_rows_per_s": 12.0,
            "served_concurrency": 0.95, "outstanding_concurrency": 3.95,
            "scaling_status": status, "client_bound": False,
            "baseline_empty": False, "under_writes_empty": False,
        }]

    def test_the_writer_rate_and_the_ratio_both_carry_the_mark(self):
        text = render_mod._churn_during_writes_table(
            {"churn_during_writes": self._rows(
                cl.SCALING_CONCURRENCY_NOT_ACHIEVED)})
        cells = _table_cells(text)
        header, row = cells[0], cells[1]
        mark = render_mod.SCALING_REFUSED_MARK
        assert mark in row[header.index("Writer rate")]
        assert mark in row[header.index("p99 under writes")]
        # Mean-based first since ops version 9, the median as an estimate.
        assert row[header.index("Reads in service")] == \
            f"3.95 of 4 (mean) · 0.95 median est. {render_mod.SCALING_REFUSED_MARK}"

    def test_a_tail_dominated_reader_row_is_marked_and_the_writer_is_not(self):
        """#78 review finding 2, in the table: a v9 reader window that kept
        its reads in flight with a heavy tail carries the note and leaves the
        writer rate unmarked."""
        rows = self._rows(cl.SCALING_OK)
        rows[0]["scaling_note"] = "tail_dominated"
        text = render_mod._churn_during_writes_table(
            {"churn_during_writes": rows})
        cells = _table_cells(text)
        header, row = cells[0], cells[1]
        cell = row[header.index("Reads in service")]
        assert render_mod.TAIL_DOMINATED_MARK in cell
        assert render_mod.SCALING_REFUSED_MARK not in cell
        assert render_mod.SCALING_REFUSED_MARK not in row[
            header.index("Writer rate")]

    def test_the_footnote_explains_the_writer_specifically(self):
        text = _plain(render_mod._churn_during_writes_table(
            {"churn_during_writes": self._rows(
                cl.SCALING_CONCURRENCY_NOT_ACHIEVED)}))
        assert "starve the writer inside the client" in text
        assert "reader_scaling_status" in text

    def test_a_concurrent_reader_pool_gets_no_mark_and_no_footnote(self):
        text = render_mod._churn_during_writes_table(
            {"churn_during_writes": self._rows(cl.SCALING_OK)})
        cells = _table_cells(text)
        header, row = cells[0], cells[1]
        assert render_mod.SCALING_REFUSED_MARK not in row[
            header.index("Writer rate")]
        assert "did not run" not in _plain(text)


# ---------------------------------------------------------------------------
# The prose, and whether the guards on it fire on correct writing
# ---------------------------------------------------------------------------

#: Each claim the report's own footnote has to keep making, with rewordings
#: that MUST pass and tidied-up forms that MUST NOT.
#:
#: The rejects are the non-vacuity half: a pattern keyed to the right answer
#: cannot see a wrong one. The accepts are what keeps the guard usable -- a
#: pattern tight enough to demand one sentence gets deleted by the next
#: author, and then it is also absent on the edit it existed to catch.
PROSE_CLAIMS = {
    # **The claim this pins was NARROWED, and the pattern REQUIRES the
    # narrowing.** What was measured is a saturated server whose service time
    # is DETERMINISTIC -- every calibration driver has a fixed per-query cost
    # -- so "a genuinely saturated server is not flagged" claimed more than
    # the calibration supports, and a server skewed enough to put its median
    # below its mean can still be refused.
    #
    # **An earlier revision of this row made the qualifier OPTIONAL while its
    # comment claimed the pattern had been narrowed with the claim.** It had
    # not: `(?: with [a-z ]{0,40})?` matched the unnarrowed sentence, the bare
    # "A saturated server is not flagged.", and `docs/05-methodology.md`'s own
    # unnarrowed copy -- so a revert to the general claim would have passed
    # the guard that exists to catch it, and `accepts` listed the unnarrowed
    # form as legitimate. `deterministic` is now mandatory and the unnarrowed
    # forms are reject rows, which is what makes this a guard rather than a
    # sentence about one.
    "a saturated server is not flagged": dict(
        # `[^.\n]` and not `[^.]`: a negated class matches a newline, and
        # `test_a_reflow_does_not_defeat_the_guard` below is the thing that
        # has to stay demonstrable -- that `_plain` normalisation is what
        # makes a WRAPPED sentence match. The real input is always normalised
        # first, so excluding the newline costs nothing and keeps that
        # demonstration honest.
        pattern=r"saturated server\b[^.\n]{0,60}?\bdeterministic\b"
                r"[^.\n]{0,60}?\bis (?:deliberately )?not flagged",
        accepts=[
            "A saturated server with deterministic service time is not "
            "flagged, and that is the case the threshold protects.",
            "A saturated server whose service time is deterministic is not "
            "flagged.",
            "A saturated server with a deterministic per-query cost is "
            "deliberately not flagged here.",
        ],
        rejects=[
            # The claim reversed.
            "A saturated server is flagged the same way.",
            "A saturated server is also refused, because its efficiency "
            "falls.",
            # The form a lazy wildcard between the noun and the verb would
            # have accepted: it reverses the claim and keeps the words.
            "A saturated server is flagged, and only a lock-bound client is "
            "not flagged.",
            # And the three the OLD pattern accepted: the claim widened back
            # to the general property the calibration does not support. These
            # are the mutants that make requiring `deterministic` a guard.
            "A genuinely saturated server is not flagged, and that is the "
            "case the threshold was chosen to protect.",
            "A saturated server is not flagged: a real queue raises the "
            "median too.",
            "Note that a genuinely saturated server is deliberately not "
            "flagged here.",
        ],
    ),
    "the efficiency needs the concurrency to have happened": dict(
        pattern=r"describes the engine only if\s+N queries were in flight",
        accepts=[
            "Scaling efficiency divides an N-client throughput by N times a "
            "one-client throughput, so it describes the engine only if N "
            "queries were in flight.",
            "It describes the engine only if N queries were in flight — "
            "otherwise the division has no basis.",
            "The ratio describes the engine only if N queries were in "
            "flight at the time.",
        ],
        rejects=[
            "Scaling efficiency describes the engine whatever the pool was "
            "doing.",
            "It always describes the engine, because the throughput is "
            "measured.",
        ],
    ),
}


class TestTheFootnoteKeepsSayingIt:
    """R7 and the prose-guard rule: two or three legitimate rewordings against
    each guard, and a form that reverses the claim must fail it."""

    @pytest.fixture(scope="class")
    def notes(self):
        return _plain(render_mod._concurrency_notes(
            {"concurrency": [_record()]}))

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_the_footnote_still_makes_the_claim(self, notes, name):
        assert re.search(PROSE_CLAIMS[name]["pattern"], notes, re.I), name

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_the_pattern_accepts_a_legitimate_rewording(self, name):
        claim = PROSE_CLAIMS[name]
        assert len(claim["accepts"]) >= 2
        for wording in claim["accepts"]:
            assert re.search(claim["pattern"], _plain(wording),
                             re.I | re.S), (
                f"{name}: the pattern rejects a legitimate rewording, which "
                f"makes it a guard that fires on correct prose:\n  {wording}")

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_the_pattern_refuses_the_reversed_form(self, name):
        claim = PROSE_CLAIMS[name]
        assert claim["rejects"]
        for wording in claim["rejects"]:
            assert not re.search(claim["pattern"], _plain(wording),
                                 re.I | re.S), (
                f"{name}: the pattern accepts a form that reverses the "
                f"claim, so it cannot see the edit it exists to "
                f"catch:\n  {wording}")

    def test_a_reflow_does_not_defeat_the_guard(self, notes):
        """The normalisation itself, asserted.

        A prose guard a line break beats is not a guard, and that has bitten
        this repository three times in one day. The patterns above are matched
        against `_plain`-normalised text; this shows the normalisation is what
        makes a wrapped sentence match.
        """
        # The NARROWED sentence, wrapped: the example has to be a form the
        # pattern accepts once normalised, and the pattern now requires
        # `deterministic`.
        wrapped = ("A saturated server with\ndeterministic service time\n"
                   "   is not flagged, and that\nis the point.")
        pattern = PROSE_CLAIMS["a saturated server is not flagged"]["pattern"]
        assert not re.search(pattern, wrapped, re.I)
        assert re.search(pattern, _plain(wrapped), re.I)
        assert notes == re.sub(r"\s+", " ", notes)

    def test_the_methodology_document_carries_the_mechanism(self):
        """§5. Measured, read and supplied, each marked as such.

        `docs/05-methodology.md` is where a reader goes for what is measured.
        The claim that must stay marked as READ is whether a client library
        releases the lock: no test here starts a server, so nothing here can
        measure it, and the 2026-10-14 run that settled it is SUPPLIED rather
        than reproduced.

        **Re-pointed by G3.** This used to assert that the document said
        running the clients as processes "has not been done". It has been
        done, so that pair of assertions would now pin prose that is false --
        which is the failure mode this class exists to catch, arriving from
        the code side instead of the prose side. What replaces them is the
        other half of the same discipline: the document must say the change
        was made, must still mark the per-library question as read rather than
        measured, and must carry the cost list as measurements.
        """
        doc = _plain(open(os.path.join(
            VB_ROOT, "docs", "05-methodology.md")).read())
        assert "What was read rather than measured" in doc
        assert "psycopg3" in doc and "MariaDB Connector/Python" in doc
        assert "no measurement in this repository settles it" in doc
        # The change itself, and that it is stated as done rather than
        # proposed.
        assert "The clients are processes, and this is what it cost" in doc
        assert "None of that is attempted here" not in doc, (
            "docs/05 still says the change was not attempted, which is now "
            "false prose about shipped code")
        # And the three cost claims a reader is most likely to act on, each
        # labelled with how it is known.
        assert "fork and not spawn, and it is forced rather than preferred" \
            in doc
        assert "The memory cost this section used to state was wrong" in doc
        assert "becomes `forkserver` in 3.14" in doc
        # The consequence for how the status should be read, because a
        # tripwire that is documented as a finding will be misread.
        assert "has therefore become a **tripwire**" in doc or \
            "has therefore become a tripwire" in doc


# ---------------------------------------------------------------------------
# The two mutations that are awkward to express as a changed input
# ---------------------------------------------------------------------------

class TestTheMedianIsWhatMakesTheGuardWork:
    """The instrument swapped at run time, inside the pool that calls it.

    `clients.py` reaches `window_concurrency` as a module-level name, so
    rebinding it on `harness.workloads.clients` reaches `ClientPool.stop`.
    Rebinding it on `harness.metrics.latency` would NOT -- a module that
    re-exports a name does not share it -- and that asymmetry is asserted
    below, because a mutation aimed at the wrong module is a dead probe that
    reports a clean result.
    """

    @staticmethod
    def _mean_based(collectors, wall_seconds, cutoff=None, window_start=None):
        """`served` computed from the MEAN span instead of the median.

        Which makes it equal to `outstanding` -- the exact figure -- and
        therefore blind: this is the implementation a reader would write from
        "Little's law over the spans we already have".
        """
        from harness.metrics.latency import (WindowConcurrency,
                                             span_disposition)
        kept = []
        for c in collectors:
            for sample in c._samples:
                if span_disposition(sample.started, sample.finished,
                                    window_start, cutoff) is None:
                    kept.append(sample.seconds)
        if not kept or wall_seconds <= 0:
            return WindowConcurrency(None, None)
        exact = sum(kept) / wall_seconds
        return WindowConcurrency(outstanding=exact, served=exact)

    def test_a_mean_based_instrument_stops_seeing_the_defect(self, dataset):
        """Honest and mutant values, read off ONE window by two instruments.

        Honest (`window_concurrency`, the median form): a four-client
        lock-holding window reports `concurrency_not_achieved`, and the
        report's efficiency cell reads "⊘ not measurable". Mutant (the
        mean-based form above, which is what a reader would write from
        "Little's law over the spans we already have"): the SAME SPANS report
        `ok` and the cell reads the number.

        **One window read twice rather than two windows, which is stronger
        than the version this replaces.** That version ran the live pool
        twice with `clients.window_concurrency` rebound in between; since G3
        the live clients are processes and a lock-holding driver no longer
        serializes them, so there is no live window with the shape to read.
        `tests/retired_thread_pool.py` produces it and hands back its
        collectors, its wall and its two window instants -- so the two
        instruments are compared over identical input and any difference is
        the arithmetic rather than two runs of a loaded box.

        `test_patching_the_defining_module_would_not_have_reached_it` below
        still pins the binding asymmetry that made the old spelling
        necessary, and `test_a_serializing_pool_is_refused` still rebinds
        `clients.window_concurrency` for real.
        """
        retired = thread_window(
            factory_for(loaded(dataset), DRIVERS["holding"](4)),
            tiny_dataset().test[:N_QUERIES], 4, K, WINDOW_S)
        args = (retired["collectors"], retired["wall"])
        kwargs = {"cutoff": retired["cutoff"],
                  "window_start": retired["window_start"]}
        honest = cl.window_concurrency(*args, **kwargs)
        mutant = self._mean_based(*args, **kwargs)
        honest_status = latency_mod.scaling_status(honest.served, 4)
        mutant_status = latency_mod.scaling_status(mutant.served, 4)
        assert honest_status == cl.SCALING_CONCURRENCY_NOT_ACHIEVED, honest
        assert mutant_status == cl.SCALING_OK, mutant
        # Not an identical pair, which is the signature of a dead probe.
        assert honest_status != mutant_status
        # And the mutant's `served` is its own `outstanding`, which is the
        # mechanism rather than a coincidence of this window.
        assert mutant.served == pytest.approx(mutant.outstanding)
        assert honest.served != pytest.approx(honest.outstanding)
        # Carried through to what a reader sees.
        assert render_mod._efficiency_cell(
            {"scaling_efficiency": 0.25,
             "scaling_status": honest_status}
        ) == render_mod.SCALING_REFUSED_CELL
        assert render_mod._efficiency_cell(
            {"scaling_efficiency": 0.25,
             "scaling_status": mutant_status}) == "0.25"

    def test_patching_the_defining_module_would_not_have_reached_it(self):
        """The dead-probe trap, asserted rather than remembered.

        `clients.py` holds its own binding, so the pool keeps calling the real
        function however `harness.metrics.latency` is rebound. A mutation
        aimed there measures nothing and reports a clean result.
        """
        original = latency_mod.window_concurrency
        try:
            latency_mod.window_concurrency = self._mean_based
            assert cl.window_concurrency is not latency_mod.window_concurrency
            assert cl.window_concurrency is original
        finally:
            latency_mod.window_concurrency = original


class TestTheChartFilterIsLoadBearing:
    @needs_matplotlib
    def test_disabling_the_status_match_puts_the_point_back(self):
        """Runtime mutation of the constant the chart compares against.

        Honest: the efficiency panel plots 1, 2 and 8. Mutant
        (`charts.SCALING_CONCURRENCY_NOT_ACHIEVED` rebound to a string no
        record carries, which makes the filter inert): the same records plot
        1, 2, 4 and 8, with the refused point back on the linear-scaling
        reference line.

        Rebound on `report.charts` and not on `report.render`: `charts` does
        `from .render import ...`, so it holds its own binding. The second
        assertion shows that patching `render` reaches nothing, which is the
        probe that would have reported a clean result.
        """
        from report import charts
        chart_test = TestTheChartDoesNotPlotARefusedEfficiency()
        rows = chart_test._series(refused_at=(4,))
        honest = chart_test._xs(chart_test._axes(rows)["axes"][2])
        original = charts.SCALING_CONCURRENCY_NOT_ACHIEVED
        try:
            charts.SCALING_CONCURRENCY_NOT_ACHIEVED = "never-matches-anything"
            mutant = chart_test._xs(chart_test._axes(rows)["axes"][2])
        finally:
            charts.SCALING_CONCURRENCY_NOT_ACHIEVED = original
        assert honest == [[1.0, 2.0, 8.0]]
        assert mutant == [[1.0, 2.0, 4.0, 8.0]]
        assert honest != mutant

    @needs_matplotlib
    def test_patching_render_does_not_reach_the_chart(self):
        from report import charts
        original = render_mod.SCALING_CONCURRENCY_NOT_ACHIEVED
        try:
            render_mod.SCALING_CONCURRENCY_NOT_ACHIEVED = "never-matches"
            assert charts.SCALING_CONCURRENCY_NOT_ACHIEVED == original
        finally:
            render_mod.SCALING_CONCURRENCY_NOT_ACHIEVED = original
