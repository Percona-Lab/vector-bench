"""T4b: the shared client pool, and readers querying while churn writes.

PR 4 §4.3 and the §4.4 bullet in scope, against `tests/fake_driver.py`. No
engine is started.

**What this file is really about is one ratio.** §4.3 item 5's headline number
is `p99 under writes / baseline p99`, and a ratio of two separately measured
quantities is the shape that has produced two published-wrong results in this
repository. The filter-strategy ratio divided a narrow point from one ops unit
by a wide point from another: 0.20x "pre-filter" and 1.00x "post-filter,
abandons the query" measured separately, against a pooled 50.0x "post-filter,
exhaustive" that neither unit produced. `charts.churn_retention` pairs records
from two different maintenance states. Both read as findings and both were
bugs.

So the guards here are about the STRUCTURE of the pairing and not about any
decimal. The ratio's two percentiles are measured by one pool whose identity
`clients.window_ratio` checks before dividing, and both land in one record, so
nothing downstream has to pair two records. Every test below that touches the
ratio injects a crossing of one of those boundaries and reports what the
published number became.

**Why no test here asserts a ratio's value.** It is a ratio of two wall-clock
percentiles. The same churn configuration's QPS on this fake has been measured
between 6,028 and 16,811 -- a factor of 2.79 -- while its seeded recall
reproduced exactly 9 times out of 9. A test that pinned 1.37 would be pinning
the box. What is pinned is that both percentiles came from one pool, one ops
unit and one configuration, and that a mutation crossing any of those changes
what gets published.

**Mutation discipline.** Every guard names the regression it catches and was
checked by injecting that regression at RUN TIME -- monkeypatch, subclass, or
a verbatim copy of the pre-fix body run over the same records -- never by
editing a file on disk. CPython invalidates a `.pyc` on (whole-second mtime,
size), so a same-size rewrite inside one second can leave stale bytecode valid
and measure the previous version. Honest and mutant values are in each
docstring and were read back out of the records or the rendered text the code
produced.

**R2, for the three report defects.** Those are repairs rather than additions,
so each one's test runs the PRE-FIX body -- copied verbatim into this file, over
the very same records -- and reports what it published beside what the fixed
form publishes. A repair whose old form also passes is not a repair.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import records                                   # noqa: E402
from harness.workloads import churn as cw                             # noqa: E402
from harness.workloads import churn_readers as crw                    # noqa: E402
from harness.workloads import clients as cl                           # noqa: E402
from harness.workloads import concurrency as conc                     # noqa: E402
from harness.workloads.context import RunContext                      # noqa: E402
from report import charts as charts_mod                               # noqa: E402
from report import churn_state                                        # noqa: E402
from report import generate as gen                                    # noqa: E402
from report import render as render_mod                               # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES, N_ROWS,      # noqa: E402
                               index_spec, tiny_dataset)

#: Latency the fake spends per query in the reader tests. **Not zero, and that
#: is load-bearing.** `FakeDriver` returns instantly by default, so every
#: latency percentile comes out exactly 0.0 -- and a p99 ratio over a zero
#: denominator is an arithmetic accident, not a slowdown figure. 1.5 ms is also
#: close to the 1.4 ms this framework has measured for a real single-client
#: query, so the window arithmetic is exercised at a realistic scale.
READER_LATENCY_S = 0.0015

#: A baseline reader window short enough for a test and long enough to hold
#: several queries per reader at the latency above. The shipped value is 30 s
#: (`churn_readers.READER_BASELINE_WINDOW_S`); `TestTheDeclarationsAgree` pins
#: that one, and this is only what the end-to-end tests run with.
TEST_BASELINE_WINDOW_S = 0.4

#: Readers in the end-to-end tests. Four, the shipped default, and it has to be
#: at most `N_QUERIES` or `_reader_count` declines the pool -- which would make
#: every assertion below pass against no reader pool at all.
TEST_READERS = 4


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="t4b-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


@pytest.fixture
def dataset():
    return tiny_dataset()


@pytest.fixture
def short_window(monkeypatch):
    """Shrink the reader baseline window for the duration of one test.

    Patched on `harness.workloads.churn`, not on `churn_readers`: `churn.run`
    reads the name out of its OWN module globals, so patching the module that
    defines it would not reach the call. That asymmetry is the module-level
    binding trap this repository has been bitten by, and getting it wrong here
    would silently run the real 30 s window per fraction.
    """
    monkeypatch.setattr(cw, "READER_BASELINE_WINDOW_S", TEST_BASELINE_WINDOW_S)
    return TEST_BASELINE_WINDOW_S


def written(ctx: RunContext) -> List[Dict[str, Any]]:
    return list(records.read_records(ctx.recorder.path))


def during_writes(ctx: RunContext) -> List[Dict[str, Any]]:
    return [r for r in written(ctx)
            if (r.get("extra") or {}).get("during_writes")]


def steps(ctx: RunContext) -> List[Dict[str, Any]]:
    """The churn step records: not the baseline, not the reader pool's."""
    return [r for r in written(ctx)
            if r.get("churn_fraction") and not (r.get("extra") or {})
            .get("during_writes")]


def run_churn(ctx, dataset, driver, readers=TEST_READERS, factory=None,
              **kwargs):
    """Churn one fraction with a reader pool, on the tiny corpus."""
    cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
           fractions=kwargs.pop("fractions", (0.10,)),
           max_queries=N_QUERIES, indexed_rows=N_ROWS,
           driver_factory=factory if factory is not None else driver.clone,
           readers=readers, **kwargs)


#: Wall clock the test writer spends per re-insert slice. `_reinsert` cuts the
#: rows into fifty slices, so this is the length of the during-writes window:
#: 50 x 8 ms = 0.4 s, matching the baseline window above.
#:
#: **Not optional.** `FakeDriver.insert_rows` returns instantly, so without it
#: the re-insert finishes before any reader completes a query, the
#: during-writes window closes empty, and every latency field on the record is
#: None -- which is exactly the state in which a ratio cannot be computed and
#: the whole §4.3 item 5 record says nothing. `load_latency_s` does not help:
#: the fake spends that inside `load()`, which the churn path never calls.
WRITE_SLICE_S = 0.008


class SlowWriter(FakeDriver):
    """A fake whose re-insert slices cost wall clock, like a real write path.

    Subclassed in the test rather than added to `tests/fake_driver.py`, which
    is not this task's file and sits at 790 lines of an 800-line maximum. The
    sleep is inside `insert_rows` and not around the call, for the reason the
    fake's own `load_latency_s` is inside `load()`: `_reinsert` times the
    slices it issues, and a cost paid outside them would not appear in
    `insert_seconds`.
    """

    def insert_rows(self, ids, vectors, attrs) -> None:
        time.sleep(WRITE_SLICE_S)
        super().insert_rows(ids, vectors, attrs)


def reader_driver(dataset, cls=SlowWriter, latency_s=READER_LATENCY_S,
                  **kwargs):
    """A loaded fake whose queries and writes both cost measurable wall clock.

    The readers are `clone()`s of it, and `FakeDriver.clone` returns a plain
    `FakeDriver` -- so the readers do NOT inherit the slow write path, which
    is what we want: only the writer should pay for it.

    `latency_s` is overridable to zero for the tests that count queries rather
    than time them: a window that must complete a 1.5 ms query to be non-empty
    is a window that completes nothing when the box is busy.
    """
    driver = cls(dim=dataset.dim, metric=dataset.metric,
                 latency_s=latency_s, **kwargs)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, numpy.arange(N_ROWS) % 100)
    return driver


#: The fork context every cross-process fixture below is built from. G3 made
#: the churn readers OS processes, which retired a technique two fixtures here
#: were built on: put a dict or a `threading.Event` in the test, have the
#: reader touch it, read it back. Under fork a child gets a SNAPSHOT, so the
#: plain-object form fails silently rather than loudly. Each converted fixture
#: records what it measured before the conversion.
MP_FORK = multiprocessing.get_context("fork")


def _no_constants(name):
    raise AssertionError(f"record contains the non-JSON constant {name!r}")


def pool_clients() -> List[Any]:
    """Every live client PROCESS belonging to a pool.

    **This was `pool_clients()` and it had become a dead probe**, which is
    worse than no probe because it reads as coverage. G3 made `clients.Client`
    an OS process rather than a `threading.Thread`, so
    `[t for t in threading.enumerate() if isinstance(t, cl.Client)]` -- the
    old body, verbatim -- can only ever be empty, and the four
    `pool_clients() == []` assertions below held for a reason that had nothing
    to do with leaks. `tests/test_g3_process_clients.py` pins that fact so it
    stays visible.

    A leaked process matters MORE than a leaked thread did: a thread died with
    the interpreter, while a leaked client goes on querying the server that
    the next configuration is about to measure -- which would put one engine's
    concurrency figure on top of the previous one's leftovers.

    `multiprocessing.active_children()` rather than a class check, because it
    reaps finished children as a side effect and is the list the runtime
    itself keeps. Filtered to this repo's clients by class, so pytest's own
    helpers (and any `multiprocessing` a plugin runs) are not counted.
    """
    return [p for p in multiprocessing.active_children()
            if isinstance(p, cl.Client) and p.is_alive()]


# ---------------------------------------------------------------------------
# §4.3 item 3 -- the lift out of concurrency.py changed nothing
# ---------------------------------------------------------------------------

class TestTheLiftIsOneImplementation:
    """§4.3 item 3: `_Client` and the barrier move to `clients.py` so churn and
    concurrency share one implementation, "including PR 1's window trimming and
    CPU accounting".

    The acceptance column says "concurrency tests unchanged", and that is the
    real gate -- it is not asserted here, because a test in this file cannot
    prove it. What is asserted here is the thing a passing concurrency suite
    would NOT catch: that the two workloads reach the same objects rather than
    two copies that happen to agree today.
    """

    def test_the_two_workloads_use_one_client_class(self):
        """Not "both have a client" -- the SAME class object.

        Two modules each holding a `Client` that behaves identically today is
        exactly the drift §4.3 item 3 exists to prevent, and no behavioural
        assertion can tell it from one shared class.
        """
        assert crw.ClientPool is cl.ClientPool
        assert conc.ClientPool is cl.ClientPool
        # The thread class the pool builds, reached through the pool rather
        # than through a second import, so this cannot pass against a pool
        # that builds something else.
        pool = cl.ClientPool(lambda: None, numpy.zeros((4, 2)), 2, K, 64,
                             None, 0)
        assert all(isinstance(w, cl.Client) for w in pool._workers)

    def test_concurrency_still_exposes_the_lifted_names(self):
        """`report/render.py` and three test files reach these through the
        concurrency module. Identity, not equality: a re-export that had
        quietly become a second function would satisfy a behavioural check."""
        assert conc._client_cpu is cl.client_cpu
        assert conc._shard_queries is cl.shard_queries
        assert conc.CLIENT_BOUND_UTIL == cl.CLIENT_BOUND_UTIL == 0.80

    def test_the_cpu_instruments_are_resolved_per_call(self, dataset,
                                                       monkeypatch):
        """The mutation that a passing concurrency suite would catch, injected
        here so the reason is written down beside it.

        Five concurrency tests calibrate a window by replacing
        `read_own_cpu_usec` ON the concurrency module. The pool lives in
        `clients.py`, so if `_instruments()` were a module-level constant
        built at import time it would hold the original and the patch would
        reach nothing -- the exact "rebinding a module attribute does not reach
        a captured reference" failure that gave three mutants a green suite in
        an earlier task.

        Honest: the pool reads the patched counter and reports
        `client_cpu_util` == 42.0, because the injected counter advances 42
        CPU-seconds per CPU per wall second. Mutant (`_instruments` frozen at
        import, injected below as a bundle built from the real functions): the
        patch is not seen and the same window reports util 0.0270.

        **42 and not 1.0, deliberately.** A utilisation of 1.0 is a value the
        REAL counter can also produce on a busy box -- `read_own_cpu_usec`
        reads this process's cgroup, which outside a container is the root
        cgroup and therefore counts every other process on the machine. A
        first version of this test asserted `mutant < honest / 2` and failed
        at 17.48 busy cores of 20. No cgroup can report more CPU than it has
        CPUs, so a util above 1.0 cannot come from the real counter under any
        load, and the gap between the two readings is now a property of the
        wiring rather than of the box.
        """
        cpus = len(os.sched_getaffinity(0))
        impossible = 42
        monkeypatch.setattr(
            conc, "read_own_cpu_usec",
            lambda: int(time.perf_counter() * 1e6) * cpus * impossible)
        driver = reader_driver(dataset)
        pool = cl.ClientPool(driver.clone, dataset.test, 2, K, 64,
                             index_spec(dataset), 0,
                             instruments=conc._instruments())
        pool.start("probe", prefix="t4b")
        time.sleep(0.15)
        honest = pool.stop()

        frozen = cl.Instruments(threading=threading,
                                cpu_usec=cl.read_own_cpu_usec,
                                cpu_count=cl.own_cpu_count,
                                merge=cl.merge_collectors)
        pool = cl.ClientPool(driver.clone, dataset.test, 2, K, 64,
                             index_spec(dataset), 0, instruments=frozen)
        pool.start("probe", prefix="t4b")
        time.sleep(0.15)
        mutant = pool.stop()

        # 42.0 by construction: the injected counter advances exactly
        # `cpus * 42` CPU-seconds per wall second.
        assert honest.client_cpu_util == pytest.approx(
            float(impossible), rel=0.05), f"honest util {honest.client_cpu_util}"
        # The mutant reads the real counter instead, and a cgroup cannot
        # report more CPU than it has CPUs however busy the machine is. So
        # this bound holds under any load, and the gap is the whole point:
        # the five concurrency calibration tests would otherwise be measuring
        # the box rather than their own injected quantity.
        assert mutant.client_cpu_util is not None
        assert mutant.client_cpu_util <= 1.05, mutant.client_cpu_util

    def test_the_client_side_barrier_timeout_is_the_one_that_binds(
            self, dataset, monkeypatch):
        """`BARRIER_TIMEOUT_S` is documented as not binding, so prove it.

        The parent waits `BARRIER_TIMEOUT_S` at the barrier and each client
        waits `CLIENT_BARRIER_TIMEOUT_S`; a client that breaks the barrier
        from its own side makes the parent's wait raise at once. As shipped
        that is 120 against 300, so the parent's 300 cannot be reached — and
        before this task the client's wait was a bare `timeout=120` with the
        300 commented as "the only thing that stops a stuck driver holding the
        machine", which was the wrong number.

        Measured by holding the parent at 6.0 s and varying only the client's:

        * client 0.3 s → the barrier broke after **0.30 s**
        * client 1.2 s → the barrier broke after **1.20 s**
        * inverted, client 6.0 s and parent 1.5 s → broke after **1.50 s**

        So the break tracks whichever is smaller, the parent's 6.0 s was never
        reached in the first two runs, and the shipped pair makes that the
        client's 120.
        """
        assert cl.CLIENT_BARRIER_TIMEOUT_S == 120.0
        assert cl.BARRIER_TIMEOUT_S == 300.0
        assert cl.CLIENT_BARRIER_TIMEOUT_S < cl.BARRIER_TIMEOUT_S, (
            "the parent's wait is now the smaller of the two, so it binds and "
            "the comment on BARRIER_TIMEOUT_S is no longer true")
        # Named, not a bare literal: a future edit cannot reintroduce a
        # number that disagrees with the comment without failing here.
        #
        # **Both halves, since G3 split the file.** `clients.py` names the
        # three constants where it builds the `Timeouts` bundle and
        # `client_process.py` reads the bundle rather than a number of its
        # own -- and the bundle is read from `clients`' globals at pool
        # construction precisely so the `monkeypatch` below still reaches the
        # client that waits. An `import` in `client_process.py` would have
        # frozen all three and turned the measurement underneath this test
        # into a reading of the shipped 120 s against itself;
        # `tests/test_g3_process_clients.py` carries the honest/mutant pair
        # for that.
        from harness.workloads import client_process as cp
        pool_src = open(cl.__file__).read()
        client_src = open(cp.__file__).read()
        bundle = pool_src.split("def _timeouts()")[1].split(
            "def _open_window")[0]
        for name in ("BARRIER_TIMEOUT_S", "CLIENT_BARRIER_TIMEOUT_S",
                     "JOIN_TIMEOUT_S"):
            assert name in bundle, name
        assert "self._ready.wait(timeout=self._timeouts.client_barrier_s)" \
            in client_src
        for stale in ("timeout=120)", "timeout=300)", "Timeouts(300"):
            assert stale not in pool_src, stale
            assert stale not in client_src, stale

        base = reader_driver(dataset, cls=FakeDriver, latency_s=0.0)

        def broke_after(client_timeout, parent_timeout):
            """One pool whose first client hangs in warm-up, timed.

            The hang state is built PER CALL, not shared across calls. A
            shared counter failed: a pool whose `start()` raises leaves its
            clients to notice the stop flag, and one of them consumed the
            next call's "first query" slot and hung in its place -- so the
            second pool reached its barrier intact and the timing assertion
            compared a real reading against None. Each call owning its own
            event and counter makes the three readings independent.
            """
            released = threading.Event()
            seen = {"n": 0}
            lock = threading.Lock()

            class _Hanging(FakeDriver):
                """Its first warm-up query never returns until released."""

                def query(self, vector, k):
                    with lock:
                        seen["n"] += 1
                        mine = seen["n"]
                    if mine == 1:
                        released.wait(timeout=30)
                    return super().query(vector, k)

            def factory():
                sibling = base.clone()
                sibling.__class__ = _Hanging
                return sibling

            monkeypatch.setattr(cl, "CLIENT_BARRIER_TIMEOUT_S",
                                client_timeout)
            monkeypatch.setattr(cl, "BARRIER_TIMEOUT_S", parent_timeout)
            pool = cl.ClientPool(factory, dataset.test, 3, K, 64,
                                 index_spec(dataset), warmup=2)
            started = time.perf_counter()
            try:
                pool.start("probe", prefix="t4b")
                return None
            except cl.PoolError:
                return time.perf_counter() - started
            finally:
                released.set()
                for worker in pool._workers:
                    worker.join(timeout=5)

        # Parent fixed, client varied: the break tracks the client.
        short = broke_after(0.3, 6.0)
        longer = broke_after(1.2, 6.0)
        assert short is not None and longer is not None
        assert short == pytest.approx(0.3, abs=0.25), short
        assert longer == pytest.approx(1.2, abs=0.35), longer
        assert longer > short, (short, longer)
        # And inverted, the parent's smaller wait is what fires instead --
        # the case the shipped 120 < 300 cannot reach.
        inverted = broke_after(6.0, 1.5)
        assert inverted == pytest.approx(1.5, abs=0.4), inverted

    def test_the_window_is_still_trimmed_by_one_predicate(self, dataset):
        """PR 1's trimming survived the lift, and it is one decision.

        A window ends by setting a flag; each client then finishes the query it
        is holding, and `prewindow_discarded` is zero by construction because
        the window opens in the barrier action and no measured query can
        precede it. A number in the second field means the window's two ends
        disagree about when it began.

        **The invariant asserted here is the partition, not a per-client
        bound.** `concurrency.py:293` says `inflight_discarded` is "at most
        one per client -- the one each was holding when the clock stopped",
        and that is true of a real driver and NOT true in general: the stop
        instant is read before the flag is set, so a client whose query costs
        less than that gap completes several inside it. Measured here at a
        zero-latency query path, three clients: `inflight_discarded` 4 against
        2,877 executed. The claim holds at this framework's measured 1.4 ms
        query and fails at a microsecond one, so what a test can assert is
        that every span lands in exactly one of the three buckets.
        """
        # INSTANT queries here, not the 1.5 ms the ratio tests use. This
        # test is about counts rather than percentiles, and a window that
        # must complete a 1.5 ms query to be non-empty is a window that
        # completes nothing when two cores of foreign work arrive in it --
        # measured, at 17.48 busy cores of 20. Three attempts on top of that,
        # because the identity below holds for an empty window too (every
        # span lands in a discard bucket) and the non-vacuity check at the end
        # should not be a coin flip.
        driver = reader_driver(dataset, latency_s=0.0)
        executed = []
        for _ in range(3):
            pool = cl.ClientPool(driver.clone, dataset.test, 3, K, 64,
                                 index_spec(dataset), 0)
            pool.start("probe", prefix="t4b")
            time.sleep(0.3)
            window = pool.stop()
            # Zero BY CONSTRUCTION: the window opens in the barrier action,
            # before any client is released, so no measured query can precede
            # it. A number here means the window's two ends disagree about
            # when it began.
            assert window.prewindow_discarded == 0
            # One predicate decided every span: kept, in flight, or
            # pre-window, and the three partition what the collectors hold.
            # This is the invariant; see the docstring for why the per-client
            # bound in `concurrency.py:293` is not one.
            total = sum(len(w.collector._samples) for w in pool._workers)
            assert (window.queries_executed + window.inflight_discarded
                    + window.prewindow_discarded) == total
            executed.append(window.queries_executed)
            if window.queries_executed:
                break
        assert executed[-1] > 0, (
            f"three 0.3 s windows completed no query at all: {executed}. "
            f"The identity above still held, but it would have held against "
            f"an empty window, so this assertion is what stops it being "
            f"vacuous.")


# ---------------------------------------------------------------------------
# §4.3 items 1, 2, 4, 6 -- the pool runs where and when it should
# ---------------------------------------------------------------------------

class TestTheReaderPoolRunsAroundTheWrite:
    def test_a_during_writes_record_is_written_with_both_halves(
            self, ctx, dataset, short_window):
        """The record §4.3 item 5 asks for, with the figures it asks for.

        Asserted key by key rather than through a key-set equality, because
        the interesting property is that the numerator and the denominator are
        both present and both populated: a record carrying one of them is one
        from which no ratio can be computed, and a ratio computed from one
        would have been the defect.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver)

        rows = during_writes(ctx)
        assert len(rows) == 1, [r.get("notes") for r in written(ctx)]
        row = rows[0]
        extra = row["extra"]
        assert extra["during_writes"] is True
        assert extra["readers"] == TEST_READERS
        # The READER count, which is what this record's throughput is over.
        assert row["clients"] == TEST_READERS
        assert row["churn_fraction"] == 0.10
        for key in ("reader_baseline_qps", "reader_baseline_p50_ms",
                    "reader_baseline_p95_ms", "reader_baseline_p99_ms",
                    "reader_under_writes_qps", "reader_under_writes_p99_ms"):
            assert extra[key] is not None, key
        assert extra["reader_p99_ratio"] is not None
        assert extra["writer_rows_per_s_with_readers"] > 0
        # Both halves really ran: a window that completed nothing would
        # record 0 here and a QPS of 0 beside it.
        assert extra["reader_baseline_queries_executed"] > 0
        assert extra["reader_under_writes_queries_executed"] > 0
        # The column and the prefixed key are the same number, which is what
        # lets the report recompute the ratio from one record.
        assert row["latency_p99_ms"] == extra["reader_under_writes_p99_ms"]

    def test_readers_compute_no_recall_and_the_record_is_valid_json(
            self, ctx, dataset, short_window):
        """§4.3 item 6, and the NaN trap that comes with it.

        The readers score nothing: the corpus is mid-change, so no ground
        truth describes it. `recall_at_k` is therefore absent, not 0.0 -- a
        zero would read as a recall collapse. The trap is what an unguarded
        `recall_at_k` over an empty list returns: `nan`, which `json.dumps`
        writes as the bare literal `NaN`. Python's `json.loads` accepts that
        as a documented extension and a browser's `JSON.parse` rejects it, so
        a records file holding one reads back perfectly in `report/loaders.py`
        and throws in the web UI.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver)

        row = during_writes(ctx)[0]
        assert "recall_at_k" not in row, row.get("recall_at_k")
        assert "recall_queries" not in row["extra"]
        with open(ctx.recorder.path) as fh:
            for line in fh:
                json.loads(line, parse_constant=_no_constants)

    def test_the_pool_starts_after_the_delete_and_stops_with_the_write(
            self, ctx, dataset, short_window):
        """§4.3 item 4, measured off the driver rather than from the clock.

        The reader drivers are clones sharing one store, so a reader that was
        running during the delete would have queried a corpus the writer had
        not finished changing. The order is proved by recording, per reader
        query, how many rows the store held: every query the during-writes
        pool made saw at least the post-delete count, and the baseline pool's
        queries all saw the full pre-delete corpus.

        Honest: baseline readers see 2,000 live rows, during-writes readers
        see between 1,800 and 2,000 as the writer puts them back. Mutant (the
        pool started before `driver.delete_ids`): the during-writes pool's
        minimum becomes 2,000 as well, so the two windows measure the same
        corpus and the ratio stops being about writes at all.
        """
        # A manager dict and a shared string, not a plain dict and a plain
        # dict, because G3 made the readers OS PROCESSES. The reader's `query`
        # runs in a child, so its appends never reached the parent -- measured,
        # `seen` came back `{}` and this test failed with `KeyError:
        # 'baseline'` rather than with a wrong number. And `phase` is written
        # by the WRITER in the parent and read by the readers in children, so
        # it has to cross the fork in the other direction too: a plain dict
        # left every reader on a pre-fork snapshot saying "baseline" for the
        # whole run.
        manager = MP_FORK.Manager()
        seen = manager.dict()
        lock = manager.Lock()
        phase = manager.dict(name="baseline")

        class _WatchingReader(FakeDriver):
            """A READER that records how many rows it could see.

            On the reader and not on the writer, deliberately: the writer also
            runs `_measure` before and after the churn, and those queries
            would land in the same log and swamp the distinction being
            measured. `clone()` builds a plain `FakeDriver`, so the class is
            reassigned on the sibling -- both are `FakeDriver` subclasses with
            no extra slots, and the sibling keeps the shared store.
            """

            def query(self, vector, k):
                live = int((~self._store.rows.tombstoned).sum())
                with lock:
                    # `manager.dict` does not mutate a nested list in place:
                    # the proxy ships a copy, so the append has to be written
                    # back through the proxy or it is lost in the child. That
                    # is the same class of silence as the plain dict was.
                    where = phase["name"]
                    seen[where] = list(seen.get(where, [])) + [live]
                return super().query(vector, k)

        writer = reader_driver(dataset)
        real_delete = writer.delete_ids

        def delete_and_switch(ids):
            real_delete(ids)
            phase["name"] = "during"

        writer.delete_ids = delete_and_switch

        def watching_factory():
            sibling = writer.clone()
            sibling.__class__ = _WatchingReader
            return sibling

        run_churn(ctx, dataset, writer, factory=watching_factory)

        assert seen["baseline"], "the baseline pool answered no query"
        assert seen["during"], "the during-writes pool answered no query"
        # Every baseline query saw the whole corpus; the delete had not run.
        assert min(seen["baseline"]) == N_ROWS
        # And every during-writes query saw a corpus the delete had already
        # cut into. 10% of 2,000 is 200, so the floor is 1,800.
        assert min(seen["during"]) < N_ROWS
        assert min(seen["during"]) >= N_ROWS - 200

    def test_readers_off_restores_the_write_only_rate(self, ctx, dataset):
        """§4.3 item 7. `readers=0` writes no during-writes record at all, and
        the step record says `readers: 0` rather than going quiet.

        The explicit zero is the point. A missing key would mean both "the
        profile turned them off" and "this record predates PR 4", and only one
        of those is a choice someone made -- `ops.churn_readers: 0` is the
        write-only re-insert rate and `config/profiles/dev.yml` runs it.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver, readers=0)

        assert during_writes(ctx) == []
        step = steps(ctx)[0]
        assert step["extra"]["readers"] == 0
        assert step["extra"]["reinsert_rows_per_s"] > 0
        assert not [k for k in step["extra"] if k.startswith("reader_")]

    def test_no_factory_leaves_the_step_record_exactly_as_it_was(
            self, ctx, dataset):
        """A caller with no `driver_factory` could not have had readers under
        any setting, so its record carries no `readers` key at all.

        That is not tidiness: `tests/test_t4a_churn.py` pins the exact `extra`
        key set of a completed step, and a `readers: 0` written onto a record
        from a run that had no pool to measure would be this change claiming a
        measurement it did not make.
        """
        driver = reader_driver(dataset)
        cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               fractions=(0.10,), max_queries=N_QUERIES, indexed_rows=N_ROWS)
        step = steps(ctx)[0]
        assert "readers" not in step["extra"]
        assert during_writes(ctx) == []

    def test_a_pool_larger_than_the_query_set_is_declined_loudly(
            self, ctx, dataset, capsys):
        """`shard_queries` hands client *c* the strided slice
        `queries[c::readers]`, which is EMPTY once *c* exceeds the pool size,
        and such a client divides by zero on its first iteration. Declined
        with a printed reason, the way the concurrency workload skips a client
        count it has too few queries for.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver, readers=N_QUERIES + 1)
        assert during_writes(ctx) == []
        out = capsys.readouterr().out
        assert "no reader pool" in out
        assert f"{N_QUERIES} quer" in out
        # The request is recorded beside the resolved count, so `readers: 0`
        # here is distinguishable from a profile that asked for none.
        step = steps(ctx)[0]
        assert step["extra"]["readers"] == 0
        assert step["extra"]["readers_requested"] == N_QUERIES + 1


# ---------------------------------------------------------------------------
# §4.3 item 5 -- the headline ratio, and every boundary it must not cross
# ---------------------------------------------------------------------------

class TestTheRatioCannotPairTwoDifferentThings:
    """The core of this task. See the module docstring for the two ratios this
    repository has already published wrong."""

    def _window(self, clients=2, queries=8, k=K, ef_search=64, unit=("a",),
                p99=10.0):
        """A `PoolWindow` with a chosen identity and a chosen p99.

        Hand-built rather than measured, and said so: these tests are about
        which pairs `window_ratio` will divide, which is a property of the
        spec comparison and not of any engine. The end-to-end tests above
        measure the real thing.
        """
        return cl.PoolWindow(
            qps=100.0,
            merged={"latency_p99_ms": p99, "queries_executed": 50},
            inflight_discarded=0, prewindow_discarded=0,
            client_cpu_seconds=1.0, client_cpus=4, client_cpu_util=0.25,
            wall_s=30.0,
            spec=cl.PoolSpec(clients=clients, queries=queries, k=k,
                             ef_search=ef_search, unit=tuple(unit)),
            results={})

    @pytest.mark.parametrize("field,crossing", [
        ("clients", {"clients": 8}),
        ("queries", {"queries": 500}),
        ("k", {"k": 100}),
        ("ef_search", {"ef_search": 400}),
        ("unit", {"unit": ("b",)}),
    ])
    def test_crossing_any_pool_axis_refuses_instead_of_dividing(
            self, field, crossing):
        """One case per axis, and the axis is named in the refusal.

        Honest (identical specs): 20.0 / 10.0 publishes 2.0. Mutant (one axis
        crossed): `PoolError` naming that axis, and no number is published at
        all. Both values are read back below rather than described.
        """
        under = self._window(p99=20.0)
        same = self._window(p99=10.0)
        honest = cl.window_ratio(under, same, "latency_p99_ms")
        assert honest == 2.0, honest

        crossed = self._window(p99=10.0, **crossing)
        with pytest.raises(cl.PoolError) as caught:
            cl.window_ratio(under, crossed, "latency_p99_ms")
        message = str(caught.value)
        # Anchored on the delimiter, not on the bare field name: "k" is a
        # substring of every other field's name and of half the prose.
        assert f"they differ on {field} " in message or \
            f"they differ on {field}," in message or \
            message.split("they differ on ")[1].startswith(field), message
        assert "not a measurement of either" in message

    def test_a_zero_or_missing_percentile_is_not_a_ratio(self):
        """`FakeDriver` times can be exactly 0.0, and a window that completed
        no query records no percentile at all.

        Honest (10.0 over 5.0): 2.0. Mutant-in-the-data (10.0 over 0.0): None,
        not `ZeroDivisionError` and not a number. A 0.0 denominator published
        as a ratio would read as "no slowdown".
        """
        assert cl.window_ratio(self._window(p99=10.0), self._window(p99=5.0),
                               "latency_p99_ms") == 2.0
        assert cl.window_ratio(self._window(p99=10.0), self._window(p99=0.0),
                               "latency_p99_ms") is None
        empty = self._window()
        empty = empty._replace(merged={"queries_executed": 0})
        assert cl.window_ratio(self._window(p99=10.0), empty,
                               "latency_p99_ms") is None

    def test_the_workload_pairs_two_windows_of_one_pool(
            self, ctx, dataset, short_window):
        """End to end: both halves of the record carry one pool identity.

        `extra.reader_pool` is the agreed spec, and `extra.reader_ops_unit` is
        the ops unit from the record's own provenance. Asserted against the
        record's columns rather than against literals, so this cannot pass
        against a pool that agreed with itself about the wrong thing.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver)
        row = during_writes(ctx)[0]
        extra = row["extra"]
        assert extra["reader_pool"] == {
            "clients": TEST_READERS, "queries": N_QUERIES, "k": K,
            "ef_search": 64}
        assert extra["reader_ops_unit"] == {
            "m": row["m"], "build_mode": row["build_mode"],
            "storage_engine": row.get("storage_engine"),
            "resource_pass": row["resource_pass"]}

    def test_a_baseline_from_a_different_pool_fails_the_record(
            self, ctx, dataset, short_window, capsys):
        """The regression the guard exists for, injected at run time.

        `_reader_pool` is the single construction site for both halves. The
        mutant gives the baseline call a different reader count -- the shape of
        the filter-strategy bug, a numerator and a denominator from two
        configurations -- and the guard refuses rather than publishing it.

        Honest: one during-writes record carrying a ratio. (0.95x on the run
        this was written against; the value is a ratio of wall clocks and is
        deliberately not pinned.)
        Mutant: zero during-writes records, `READER POOL FAILED` on stderr
        naming `clients`, and `extra.readers_failed` on the step record. The
        post-churn measurement is written either way, which is the part that
        makes this a failed RECORD rather than a failed workload.

        The patch goes on `churn_readers`, not on `churn`: `_reader_window`
        resolves `_reader_pool` in its own module, so patching the name
        `churn.py` re-exported would have widened the OTHER half. Both
        spellings cross the boundary and both are refused, but only this one
        crosses the half the docstring claims -- and `calls["n"] == 1` below is
        what says the patch landed where it was aimed rather than being
        inert.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver)
        honest = during_writes(ctx)
        assert len(honest) == 1
        honest_ratio = honest[0]["extra"]["reader_p99_ratio"]
        assert honest_ratio is not None

        # A second run of the same configuration, with the baseline pool
        # built one client wider.
        recorder = records.Recorder(ctx.recorder.path + ".mutant")
        mutant_ctx = RunContext(
            run_id=ctx.run_id, resource_pass=ctx.resource_pass,
            recorder=recorder, engine=ctx.engine, engine_tag=ctx.engine_tag,
            march=ctx.march, k=ctx.k, cache_dir=ctx.cache_dir)
        real_pool = crw._reader_pool
        calls = {"n": 0}

        def widen_the_baseline(factory, queries, readers, *args, **kwargs):
            calls["n"] += 1
            return real_pool(factory, queries, readers + 1, *args, **kwargs)

        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(crw, "_reader_pool", widen_the_baseline)
                run_churn(mutant_ctx, dataset, reader_driver(dataset))
            mutant_rows = [r for r in records.read_records(recorder.path)
                           if (r.get("extra") or {}).get("during_writes")]
            mutant_steps = [r for r in records.read_records(recorder.path)
                            if r.get("churn_fraction")
                            and not (r.get("extra") or {}).get("during_writes")]
        finally:
            recorder.close()

        # Exactly the baseline half went through the patched name. Zero
        # would mean the patch was inert and the assertions below would be
        # passing against an unmutated run.
        assert calls["n"] == 1, calls
        assert mutant_rows == [], (
            f"the mutant published a ratio: "
            f"{[r['extra'].get('reader_p99_ratio') for r in mutant_rows]}")
        err = capsys.readouterr().err
        assert "READER POOL FAILED" in err
        assert "they differ on clients" in err, err
        assert mutant_steps, "the post-churn record must survive"
        assert "clients" in mutant_steps[0]["extra"]["readers_failed"]

    def test_the_report_recomputes_the_ratio_from_one_record(self):
        """Nothing downstream pairs two records, so nothing can pair the wrong
        two.

        Two during-writes records from two ops units with deliberately
        different latencies. The published ratio of each is its OWN pair, and
        changing one record's baseline moves only that record's ratio.

        Honest: unit m=16 publishes 2.0 and unit m=32 publishes 1.1. Mutant
        (the classic form: divide every record by the first baseline seen,
        written out below over the very same records): m=16 publishes 2.0 and
        m=32 publishes **8.8** -- 88.0 ms over the OTHER unit's 10.0 ms
        baseline -- against the 1.1x its own pair measured, a gap of 7.7.
        Deterministic arithmetic, so these are the numbers the assertions
        below carry; an earlier revision of this docstring said 8.0 and 7.3,
        copied from the sibling cross-check test where 7.3 is the right
        figure.
        """
        recs = [
            _during_writes_record(m=16, baseline_p99=10.0, under_p99=20.0),
            _during_writes_record(m=32, baseline_p99=80.0, under_p99=88.0),
        ]
        rows = {r["m"]: r for r in churn_state.during_writes_rows(recs)}
        assert rows[16]["p99_ratio"] == pytest.approx(2.0)
        assert rows[32]["p99_ratio"] == pytest.approx(1.1)

        # The pre-fix form, verbatim in shape: one baseline for the dataset.
        pooled_baseline = (recs[0]["extra"]["reader_baseline_p99_ms"])
        mutant = {r["m"]: r["latency_p99_ms"] / pooled_baseline for r in recs}
        assert mutant[16] == pytest.approx(2.0)
        assert mutant[32] == pytest.approx(8.8), mutant
        # The number the fixed form publishes for that unit is not the pooled
        # one, and the gap is the size of the defect.
        assert abs(mutant[32] - rows[32]["p99_ratio"]) > 7.0

    def test_two_ops_units_stay_two_distinguishable_rows(self):
        """The identity columns, which is the defect T3c fixed for filtered
        search and this reproduces for the reader pool.

        Honest: two rows whose (engine, dataset, pass, M, build mode, storage,
        churn) prefixes differ -- `m` 16 and 32 -- carrying 2.0 and 1.1.
        Mutant (`m` dropped from `OPS_UNIT_FIELDS` at run time, T3c's own
        injection): both rows report `m: None`, their identity prefixes become
        byte-identical, and the table publishes 2.0 and 1.1 under one identity
        -- which a reader can only read as noise or as a duplicate row, and
        both readings are false.
        """
        recs = [
            _during_writes_record(m=16, baseline_p99=10.0, under_p99=20.0),
            _during_writes_record(m=32, baseline_p99=80.0, under_p99=88.0),
        ]

        def identities(rows):
            return [(r["engine"], r["dataset"], r["resource_pass"], r["m"],
                     r["build_mode"], r["storage_engine"],
                     r["churn_fraction"]) for r in rows]

        honest = churn_state.during_writes_rows(recs)
        honest_ids = identities(honest)
        assert len(set(honest_ids)) == 2, honest_ids
        assert sorted(r["m"] for r in honest) == [16, 32]

        original = gen.OPS_UNIT_FIELDS
        try:
            gen.OPS_UNIT_FIELDS = tuple(f for f in original if f != "m")
            mutant = churn_state.during_writes_rows(recs)
        finally:
            gen.OPS_UNIT_FIELDS = original
        mutant_ids = identities(mutant)
        assert len(set(mutant_ids)) == 1, mutant_ids
        assert [r["m"] for r in mutant] == [None, None]
        # And the two different ratios are still both published, now under
        # that one identity.
        assert sorted(round(r["p99_ratio"], 2) for r in mutant) == [1.1, 2.0]

    def test_a_record_whose_two_halves_disagree_is_flagged_not_published(self):
        """The cross-check on the harness's own stamped ratio.

        The report divides the record's two percentiles; the harness also
        stamps its own `extra.reader_p99_ratio`. If a future change stamped a
        ratio from the wrong pair while writing percentiles from the right one
        -- which is precisely the published-wrong shape -- the two would
        disagree and neither is reportable.

        Honest (stamped 2.0, recomputed 2.0): `ratio_disagrees` False and the
        rendered table carries no warning marker. Mutant (stamped 7.3, the
        same percentiles): `ratio_disagrees` True, the cell gains `⚠`, and the
        footnote telling a reader to go to the raw keys appears.
        """
        honest_rec = _during_writes_record(m=16, baseline_p99=10.0,
                                           under_p99=20.0, stamped=2.0)
        mutant_rec = _during_writes_record(m=16, baseline_p99=10.0,
                                           under_p99=20.0, stamped=7.3)
        honest = churn_state.during_writes_rows([honest_rec])[0]
        mutant = churn_state.during_writes_rows([mutant_rec])[0]
        assert honest["ratio_disagrees"] is False
        assert mutant["ratio_disagrees"] is True
        # Both recompute the same ratio; only the stamp differs.
        assert honest["p99_ratio"] == mutant["p99_ratio"] == 2.0

        honest_table = render_mod._churn_during_writes_table(
            {"churn_during_writes": [honest]})
        mutant_table = render_mod._churn_during_writes_table(
            {"churn_during_writes": [mutant]})
        assert "⚠" not in honest_table
        assert "⚠" in mutant_table
        assert "do not describe each other" in mutant_table


# ---------------------------------------------------------------------------
# §4.4 -- readers stop cleanly, and an exception fails the record
# ---------------------------------------------------------------------------

class TestReadersStopWithoutHanging:
    """§4.4's bullet, treated as a floor.

    Taken literally, "readers stop cleanly when the writer exhausts its
    budget" is satisfied by a workload that returns -- which it would even if
    it left four reader PROCESSES querying the engine for the rest of the run.
    So the assertion here is on the live client processes, via
    `pool_clients()`, plus a wall clock bound: a hang is the failure mode, and
    a test that can only fail by timing out tells whoever reads the log
    nothing about what hung.

    **This docstring said THREADS, identified by class, and the code below has
    asserted on processes since G3.** That is the dead-probe wording
    `pool_clients()`'s own docstring records retiring -- the thread form "can
    only ever be empty" and "reads as coverage" -- so describing a process
    assertion in it was the same mistake one level up. A leaked process
    matters more than a leaked thread did: a thread died with the interpreter
    and a client goes on querying the server the next configuration measures.
    """

    def test_the_budget_ending_the_write_also_ends_the_readers(
            self, ctx, dataset, short_window):
        """Honest: the writer stops on its budget, no reader PROCESS survives
        the call, and no during-writes record is written -- the corpus is
        missing rows, so `writer_rows_per_s_with_readers` would describe a
        write that never finished. The stall record states `readers: 4`, so
        the JSONL still says what the stalled write was contending with.
        """
        before = pool_clients()
        assert before == [], f"a pool leaked from an earlier test: {before}"
        driver = reader_driver(dataset, load_latency_s=0.05)
        started = time.perf_counter()
        run_churn(ctx, dataset, driver, budget_s=0.01)
        elapsed = time.perf_counter() - started

        assert pool_clients() == [], "a reader process outlived the call"
        # Comfortably inside `clients.BARRIER_TIMEOUT_S` (300 s) and
        # `JOIN_TIMEOUT_S` (120 s): a hang would have to blow one of those,
        # and this bound distinguishes "returned" from "eventually returned".
        assert elapsed < 30.0, elapsed

        stalled = [r for r in written(ctx)
                   if (r.get("extra") or {}).get("reinsert_completed") is False]
        assert len(stalled) == 1
        assert stalled[0]["extra"]["readers"] == TEST_READERS
        assert during_writes(ctx) == []

    def test_a_reader_exception_fails_the_record_and_not_the_workload(
            self, ctx, dataset, short_window, capsys):
        """Honest (readers that answer): one during-writes record, and the
        post-churn record beside it.

        Mutant (a reader whose `query` raises on its tenth call, injected as a
        driver factory rather than by editing anything): zero during-writes
        records, `READER POOL FAILED` on stderr naming the exception, the
        failure recorded in `extra.readers_failed` on the step record, and the
        post-churn measurement still written -- which is the decision
        `_record_maintenance` already documents for a repair path that fails
        and `_reinsert` for a write path that dies.

        It raises on the TENTH call and not the first, deliberately: a reader
        that dies before the barrier is caught by the barrier's own release
        path, and the failure worth testing is one that happens while the
        writer is working and the window is open.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver)
        assert len(during_writes(ctx)) == 1

        # A `multiprocessing.Event` and not a `threading.Event`, because the
        # WRITER sets it in the parent and the READERS test it in child
        # processes. With a thread Event the readers held a pre-fork copy that
        # was never set: measured, zero readers failed, one during-writes
        # record was written where this test requires none, and the mutant arm
        # asserted nothing at all.
        writing = MP_FORK.Event()

        class _FailingReader(FakeDriver):
            def query(self, vector, k):
                if writing.is_set():
                    raise RuntimeError("reader connection reset by peer")
                return super().query(vector, k)

        class _ArmingWriter(SlowWriter):
            """Arms the reader failure once the re-insert is under way.

            Gated on the write rather than on a call count, and that is the
            interesting case rather than a convenience: a reader that dies
            before the barrier is caught by the barrier's own release path,
            and the failure worth testing is one that happens while the window
            is open and the writer is working. A call-count threshold would
            also have landed in the baseline window, whose own guard is
            exercised by `test_a_failing_baseline_pool_does_not_kill_the_run`.
            """

            def insert_rows(self, ids, vectors, attrs) -> None:
                writing.set()
                super().insert_rows(ids, vectors, attrs)

        writer = reader_driver(dataset, cls=_ArmingWriter)

        def failing_factory():
            sibling = writer.clone()
            sibling.__class__ = _FailingReader
            return sibling

        recorder = records.Recorder(ctx.recorder.path + ".mutant")
        mutant_ctx = RunContext(
            run_id=ctx.run_id, resource_pass=ctx.resource_pass,
            recorder=recorder, engine=ctx.engine, engine_tag=ctx.engine_tag,
            march=ctx.march, k=ctx.k, cache_dir=ctx.cache_dir)
        started = time.perf_counter()
        try:
            run_churn(mutant_ctx, dataset, writer, factory=failing_factory)
            rows = list(records.read_records(recorder.path))
        finally:
            recorder.close()
        elapsed = time.perf_counter() - started

        assert elapsed < 30.0, f"the failing pool hung for {elapsed:.1f}s"
        assert pool_clients() == [], "a failed reader process outlived the call"
        assert [r for r in rows
                if (r.get("extra") or {}).get("during_writes")] == []
        err = capsys.readouterr().err
        assert "READER POOL FAILED" in err
        assert "reset by peer" in err
        step = [r for r in rows if r.get("churn_fraction")][0]
        assert "reset by peer" in step["extra"]["readers_failed"]
        # The measurement the reader pool is not part of survived.
        assert step["recall_at_k"] is not None


    def test_a_failing_baseline_pool_does_not_kill_the_run(
            self, ctx, dataset, short_window, capsys):
        """The other half of the pool, and it was a real gap.

        §4.3 item 5 measures the baseline BEFORE the deletes, in its own pool.
        Until this test existed that pool's failure propagated straight out of
        `churn.run`, so a reader whose connection died in the 30 s baseline
        window took the whole churn workload with it -- the post-churn recall
        and QPS included, which on a real corpus is a night of ingest thrown
        away to lose a diagnostic.

        Honest (readers that answer): one during-writes record.
        Mutant (every reader raises on its first query, so the baseline pool
        cannot even reach the barrier with a live client): zero during-writes
        records, `READER POOL FAILED` on stderr, `extra.readers_failed` on the
        step record, and the post-churn record present with a real recall.
        """
        class _DeadReader(FakeDriver):
            def query(self, vector, k):
                raise RuntimeError("reader could not connect")

        writer = reader_driver(dataset)

        def dead_factory():
            sibling = writer.clone()
            sibling.__class__ = _DeadReader
            return sibling

        started = time.perf_counter()
        run_churn(ctx, dataset, writer, factory=dead_factory)
        elapsed = time.perf_counter() - started

        assert elapsed < 30.0, f"the failing baseline hung for {elapsed:.1f}s"
        assert pool_clients() == []
        assert during_writes(ctx) == []
        err = capsys.readouterr().err
        assert "READER POOL FAILED" in err
        assert "could not connect" in err
        step = steps(ctx)[0]
        assert "could not connect" in step["extra"]["readers_failed"]
        assert step["recall_at_k"] is not None


class TestAnEmptyReaderWindowIsNotAZero:
    """A during-writes window shorter than one reader query.

    `clients.client_cpu` already makes the argument for its own figure:
    "Zero is the claim 'the clients did nothing', which is the one thing that
    cannot be true of a window that executed queries." The during-writes
    window is exactly as long as the re-insert takes, so a write path that
    returns faster than one read closes it empty — and the record's `qps`
    column is then 0.0, which the table published beside a real baseline with
    nothing to say the window never opened. `concurrency` has
    `_warn_nothing_finished` for the same state.

    Unreachable on every shipped profile: `main` and `smoke` re-insert tens of
    thousands of rows and the pool runs for all of it. Reachable at once on a
    synthetic driver, which is where it was found.
    """

    def _empty_window_run(self, ctx, dataset):
        """A 20 ms reader query against an INSTANT write path."""
        driver = reader_driver(dataset, cls=FakeDriver, latency_s=0.02)
        run_churn(ctx, dataset, driver)
        rows = during_writes(ctx)
        assert len(rows) == 1, [r.get("notes") for r in written(ctx)]
        return rows[0]

    def test_the_record_keeps_the_evidence(self, ctx, dataset, short_window,
                                           capsys):
        """The 0.0 is in the record, honestly, with what produced it.

        Measured: `reader_under_writes_window_s` 0.002,
        `reader_under_writes_queries_executed` 0,
        `reader_under_writes_p99_ms` None, `qps` 0.0, against a baseline that
        counted 96 queries at a p99 of 20.855 ms. The ratio is None rather
        than a number, which `window_ratio` already guaranteed.
        """
        row = self._empty_window_run(ctx, dataset)
        extra = row["extra"]
        assert extra["reader_under_writes_queries_executed"] == 0
        assert extra["reader_under_writes_p99_ms"] is None
        assert extra["reader_p99_ratio"] is None
        assert row["qps"] == 0.0
        # The baseline half really ran, so this is one empty window and not a
        # pool that never worked.
        assert extra["reader_baseline_queries_executed"] > 0
        assert extra["reader_baseline_p99_ms"] > 0
        # And the operator was told while it was happening.
        err = capsys.readouterr().err
        assert "NOTHING FINISHED" in err
        assert "during-writes window" in err
        assert "the write finished before a single read did" in err

    def test_the_table_renders_a_dash_and_says_why(self, ctx, dataset,
                                                   short_window):
        """Honest vs the pre-fix render, over the very same record.

        Honest: `Reader QPS: 192 → —` and a footnote naming the window length.
        Old form (the `baseline_empty`/`under_writes_empty` flags cleared, so
        the cells fall back to `_fmt` as they did before): `192 → 0`, no
        footnote — a throughput collapse where there was no window.

        The p99 cell was already a dash either way, because `_fmt(None)` gives
        one; the QPS column was the only published zero, which is why it is
        the one asserted here.
        """
        row = churn_state.during_writes_rows(
            [self._empty_window_run(ctx, dataset)])[0]
        assert row["under_writes_empty"] is True
        assert row["baseline_empty"] is False

        fixed = render_mod._churn_during_writes_table(
            {"churn_during_writes": [row]})
        cells = [line for line in fixed.splitlines()
                 if line.startswith("| fake")]
        assert len(cells) == 1, fixed
        # Anchored on the delimiters: a bare "→ —" would also match a p99
        # cell, and " 0 |" is a substring of " 10 |".
        assert "→ — |" in cells[0], cells[0]
        assert "→ 0 |" not in cells[0], cells[0]
        assert "closed before a single reader query finished" in fixed
        assert "reader_under_writes_queries_executed" in fixed

        mutant_row = dict(row)
        mutant_row["baseline_empty"] = None
        mutant_row["under_writes_empty"] = None
        mutant = render_mod._churn_during_writes_table(
            {"churn_during_writes": [mutant_row]})
        mutant_cells = [line for line in mutant.splitlines()
                        if line.startswith("| fake")]
        assert "→ 0 |" in mutant_cells[0], mutant_cells[0]
        assert "closed before a single reader query finished" not in mutant

    def test_a_record_with_no_count_is_not_a_record_with_an_empty_window(
            self):
        """The `== 0`-not-falsy choice, which had a comment and no test.

        `churn_state` reads the emptiness flags as
        `extra.get("reader_..._queries_executed") == 0`. A missing key means a
        harness that did not record the count -- every churn record written
        before PR 4 §4.3, and anything `tests/synthetic.py` builds -- and it
        is not a claim that the window was empty.

        Honest: a record with both count keys absent gives
        `baseline_empty=False, under_writes_empty=False`, and the table prints
        its numbers with no footnote. Mutant (`not extra.get(...)`, injected
        by evaluating the same expressions over the same record): `True, True`
        -- a dash in both cells and a footnote claiming "the window closed
        before a single reader query finished" about a record that never
        recorded a count. **831 tests pass under that mutant**, which is why
        this exists.
        """
        record = _during_writes_record()
        for key in ("reader_baseline_queries_executed",
                    "reader_under_writes_queries_executed"):
            record["extra"].pop(key, None)
        extra = record["extra"]
        assert "reader_baseline_queries_executed" not in extra
        assert "reader_under_writes_queries_executed" not in extra

        row = churn_state.during_writes_rows([record])[0]
        assert row["baseline_empty"] is False
        assert row["under_writes_empty"] is False
        table = render_mod._churn_during_writes_table(
            {"churn_during_writes": [row]})
        assert "closed before a single reader query finished" not in table
        assert "→ — |" not in table, table

        # The mutant's own arithmetic, over this very record: `not None` is
        # True, so both flags flip and the footnote appears.
        assert (not extra.get("reader_baseline_queries_executed")) is True
        assert (not extra.get(
            "reader_under_writes_queries_executed")) is True
        mutant_row = dict(row)
        mutant_row["baseline_empty"] = not extra.get(
            "reader_baseline_queries_executed")
        mutant_row["under_writes_empty"] = not extra.get(
            "reader_under_writes_queries_executed")
        mutant = render_mod._churn_during_writes_table(
            {"churn_during_writes": [mutant_row]})
        assert "closed before a single reader query finished" in mutant
        assert "→ — |" in mutant

        # And the guard still fires on a real zero, so this is not a test
        # that simply disabled it.
        zeroed = _during_writes_record()
        zeroed["extra"]["reader_under_writes_queries_executed"] = 0
        zeroed["extra"]["reader_baseline_queries_executed"] = 50
        zero_row = churn_state.during_writes_rows([zeroed])[0]
        assert zero_row["under_writes_empty"] is True
        assert zero_row["baseline_empty"] is False

    def test_a_full_window_is_unaffected(self, ctx, dataset, short_window):
        """The guard must not blank a real measurement.

        Same workload with a write path that takes measurable time: both
        windows count queries, neither flag is set, and the QPS cell carries
        two numbers.
        """
        driver = reader_driver(dataset)
        run_churn(ctx, dataset, driver)
        row = churn_state.during_writes_rows(during_writes(ctx))[0]
        assert row["under_writes_empty"] is False
        assert row["baseline_empty"] is False
        table = render_mod._churn_during_writes_table(
            {"churn_during_writes": [row]})
        assert "closed before a single reader query finished" not in table
        assert "→ — |" not in table, table


# ---------------------------------------------------------------------------
# The three report defects. R2: the old form is run over the same records.
# ---------------------------------------------------------------------------

def _churn_record(fraction, qps, recall, state=None, index_bytes=250_000,
                  engine="mariadb", storage="InnoDB", during=None):
    extra: Dict[str, Any] = {}
    if state is not None:
        extra["maintenance_state"] = state
        if state == "after":
            extra["maintenance"] = "VACUUM (ANALYZE)"
            extra["maintenance_s"] = 12.5
    if during is not None:
        extra.update(during)
    return {"phase": "churn", "engine": engine, "dataset": "d",
            "storage_engine": storage, "m": 16, "build_mode": "post",
            "resource_pass": "normalized", "churn_fraction": fraction,
            "qps": qps, "recall_at_k": recall, "index_bytes": index_bytes,
            "extra": extra}


def _during_writes_record(m=16, baseline_p99=10.0, under_p99=20.0,
                          stamped=None, readers=4, fraction=0.10):
    """A `during_writes` record as `churn_readers` writes one.

    Hand-built, and said so: these are report-side tests about which rows get
    published under which identity, and a measured record would make the
    percentiles depend on the box. The end-to-end tests above write the real
    thing and assert the two halves are present and populated.
    """
    return {
        "phase": "churn", "engine": "mariadb", "dataset": "d",
        "storage_engine": "InnoDB", "m": m, "build_mode": "post",
        "resource_pass": "normalized", "churn_fraction": fraction,
        "clients": readers, "qps": 900.0, "latency_p99_ms": under_p99,
        "extra": {
            "during_writes": True, "readers": readers,
            "reader_baseline_p99_ms": baseline_p99,
            "reader_under_writes_p99_ms": under_p99,
            "reader_baseline_qps": 1200.0,
            "reader_p99_ratio": (under_p99 / baseline_p99 if stamped is None
                                 else stamped),
            "writer_rows_per_s_with_readers": 55.0,
        },
    }


class TestTheChurnTableTellsTwoMeasurementsApart:
    """Defect 1: two mutually contradictory rows under one identity.

    Measured on the fake, 9 of 9 runs of one configuration: two 10% rows whose
    engine, dataset, storage and fraction columns were byte-identical,
    carrying recalls 0.8960 and 0.9980 and index sizes 275.0 KiB and
    250.0 KiB. `identities distinguishable: False`.
    """

    def _pair(self):
        return [_churn_record(0.0, 7445.0, 1.0),
                _churn_record(0.10, 7233.0, 0.8960, state="before",
                              index_bytes=275_000),
                _churn_record(0.10, 9100.0, 0.9980, state="after",
                              index_bytes=250_000)]

    @staticmethod
    def _old_identities(summary):
        """The pre-fix `_churn_table`'s identity columns, verbatim in shape.

        The four columns the old header named, read off the same records the
        fixed form renders, so "the old form does not catch the mutant" is a
        value and not a claim.
        """
        return [(r.get("engine"), r.get("dataset"),
                 r.get("storage_engine"),
                 f"{(r.get('churn_fraction') or 0):.0%}")
                for r in summary["churn"]]

    def test_the_two_states_are_distinguishable_rows(self):
        """Honest: the 10% rows differ in the `Repair` column -- `before` and
        `after VACUUM (ANALYZE) (12.5 s)` -- so a reader can tell which recall
        belongs to which state.

        Old form: both 10% rows carry the identity
        ('mariadb', 'd', 'InnoDB', '10%'), byte-identical, with recalls 0.8960
        and 0.9980 beside them. Distinguishable: False.
        """
        summary = {"churn": self._pair()}
        old = self._old_identities(summary)
        assert len(old) == 3 and len(set(old)) == 2, old
        assert old[1] == old[2], "the old identities must collide"

        table = render_mod._churn_table(summary)
        header = table.strip().splitlines()[0]
        assert "| Repair |" in header, header
        rows = [line for line in table.splitlines()
                if line.startswith("| MariaDB")]
        assert len(rows) == 3
        tens = [r for r in rows if "| 10% |" in r]
        assert len(tens) == 2
        assert tens[0] != tens[1], tens
        assert "| before |" in tens[0]
        assert "VACUUM (ANALYZE)" in tens[1]
        assert "12.5 s" in tens[1]
        # The baseline names no state and must not be given one.
        baseline = next(r for r in rows if "| 0% |" in r)
        assert "| — |" in baseline

    def test_the_reader_pool_is_not_a_row_in_this_table(self):
        """Its `qps` is aggregate throughput over four clients against a
        corpus mid-change, and it carries no recall. In this table it would be
        a third 10% row whose QPS is several times the others' and whose
        recall cell is blank.

        Honest: 3 rows. Old form (no exclusion, run below over the same
        records): 4 rows, the extra one carrying 0.0 recall-column text and a
        QPS from a different kind of measurement.
        """
        recs = self._pair() + [_during_writes_record()]
        summary = {"churn": recs}
        assert len(self._old_identities(summary)) == 4
        table = render_mod._churn_table(summary)
        rows = [line for line in table.splitlines()
                if line.startswith("| MariaDB")]
        assert len(rows) == 3, rows
        # And it is published in its own table instead.
        published = churn_state.during_writes_rows(recs)
        assert len(published) == 1
        assert published[0]["readers"] == 4


class TestTheChurnChartDrawsCurvesNotSegments:
    """Defect 2: `charts.churn` plotted x = [0.0, 0.1, 0.1, 0.25, 0.25] as one
    series, 9 of 9 runs, so each fraction was a vertical segment."""

    def _records(self):
        return [_churn_record(0.0, 7445.0, 1.0),
                _churn_record(0.10, 7233.0, 0.8960, state="before"),
                _churn_record(0.10, 9100.0, 0.9980, state="after"),
                _churn_record(0.25, 6800.0, 0.8100, state="before"),
                _churn_record(0.25, 8900.0, 0.9900, state="after")]

    @staticmethod
    def _old_series(recs, dataset="d"):
        """The pre-fix grouping, verbatim: `series_key` alone, no state."""
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for r in recs:
            if r.get("dataset") == dataset and r.get("churn_fraction") is not None:
                groups.setdefault(charts_mod.series_key(r), []).append(r)
        return {k: sorted(r["churn_fraction"] for r in v)
                for k, v in groups.items()}

    def test_each_state_is_its_own_series(self, monkeypatch):
        """Honest: two series, each x = [0.0, 0.1, 0.25] -- one point per
        fraction, which is what a curve is.

        Old form: one series, x = [0.0, 0.1, 0.1, 0.25, 0.25]. Both duplicated
        x values are in it, which is the vertical segment.
        """
        recs = self._records()
        old = self._old_series(recs)
        assert len(old) == 1
        assert list(old.values())[0] == [0.0, 0.10, 0.10, 0.25, 0.25]

        plotted: Dict[Any, List[float]] = {}

        def capture(self, xs, ys, **kwargs):
            plotted[kwargs.get("label") or len(plotted)] = list(xs)
            return []

        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture)
        charts_mod.churn(recs, "d", str(_tmp()), "churn-d")

        xs = sorted(plotted.values())
        assert len(xs) == 4, plotted            # two panels x two series
        for series in xs:
            assert series == [0.0, 0.10, 0.25], plotted
        # The legend names the state, since this figure holds two of them.
        labels = [k for k in plotted if isinstance(k, str)]
        assert any("before repair" in k for k in labels), labels
        assert any("after repair" in k for k in labels), labels

    def test_the_reader_pool_is_not_a_third_point(self):
        """Honest: the during-writes record is excluded, so the series keep
        one point per fraction. Old form (no exclusion): the 10% x value
        appears three times."""
        recs = self._records() + [_during_writes_record()]
        old = self._old_series(recs)
        assert list(old.values())[0].count(0.10) == 3
        from report.churn_state import churn_measurements
        kept = [r for r in churn_measurements(recs)]
        assert len(kept) == len(recs) - 1


class TestTheRetentionRatioPairsOneState:
    """Defect 3: `charts.churn_retention` returned each fraction's pair in
    sorted order, so x = [0.1, 0.1, 0.25, 0.25] and before/after could not be
    told apart by position -- 9 of 9 runs.

    The ratios above 1.0 this chart can produce are NOT claimed to be fixed
    here and no mechanism is asserted for them: on the fake, which runs no
    vacuum at all, 2 of 9 runs had both points at a fraction above 1.0, so the
    tombstone-laden measurement outran the baseline too. It is jitter in a
    50-query QPS figure that ranged 6,028 to 16,811 across those runs.
    """

    def _records(self):
        return [_churn_record(0.0, 1000.0, 1.0),
                _churn_record(0.10, 400.0, 0.8960, state="before"),
                _churn_record(0.10, 1100.0, 0.9980, state="after"),
                _churn_record(0.25, 300.0, 0.8100, state="before"),
                _churn_record(0.25, 1050.0, 0.9900, state="after")]

    @staticmethod
    def _old_retention(recs, dataset="d"):
        """The pre-fix body, verbatim, over the same records."""
        baseline: Dict[Any, float] = {}
        after: Dict[Any, List[Tuple[float, float]]] = {}
        for r in recs:
            if (r.get("phase") != "churn" or r.get("dataset") != dataset
                    or not r.get("qps")):
                continue
            key = (charts_mod.series_key(r)
                   + (r.get("resource_pass"), r.get("build_mode")))
            if (r.get("churn_fraction") or 0) == 0:
                baseline[key] = r["qps"]
            else:
                after.setdefault(key, []).append(
                    (r["churn_fraction"], r["qps"]))
        return {key: [(f, q / baseline[key]) for f, q in sorted(points)]
                for key, points in after.items() if baseline.get(key)}

    def test_one_point_per_fraction_and_it_is_the_unrepaired_one(self):
        """Honest: x = [0.1, 0.25] with retentions 0.40 and 0.30 -- the index
        as the churn left it, against the pre-churn baseline the axis names.

        Old form: x = [0.1, 0.1, 0.25, 0.25] with retentions
        [0.40, 1.10, 0.30, 1.05]. Two of those four are above 1.0 on an axis
        labelled "QPS as a fraction of pre-churn (higher is better)", and
        nothing in the returned list says which of each pair is which.
        """
        recs = self._records()
        old = list(self._old_retention(recs).values())[0]
        assert [f for f, _ in old] == [0.10, 0.10, 0.25, 0.25]
        assert [round(v, 2) for _, v in old] == [0.40, 1.10, 0.30, 1.05]
        assert len([v for _, v in old if v > 1.0]) == 2

        fixed = list(charts_mod.churn_retention(recs, "d").values())[0]
        assert [f for f, _ in fixed] == [0.10, 0.25]
        assert [round(v, 2) for _, v in fixed] == [0.40, 0.30]

        # The repaired state is not lost: `charts.churn` and the churn table
        # both carry it. Asserted so this exclusion cannot become a deletion.
        table = render_mod._churn_table({"churn": recs})
        assert "VACUUM (ANALYZE)" in table

    def test_the_key_shape_is_unchanged(self):
        """`tests/test_config.py:1705` looks `churn_retention`'s result up by
        `series_key(r) + (resource_pass, build_mode)`, and that file is not
        this task's to edit. Splitting this chart's series by maintenance
        state would have widened the key and broken it -- which is why the
        repaired state is EXCLUDED here and split in `charts.churn` instead.
        """
        recs = self._records()
        result = charts_mod.churn_retention(recs, "d")
        before = next(r for r in recs
                      if (r.get("extra") or {}).get("maintenance_state")
                      == "before")
        key = (charts_mod.series_key(before)
               + (before.get("resource_pass"), before.get("build_mode")))
        assert key in result, list(result)

    def test_a_reader_pool_record_would_publish_a_spurious_retention(self):
        """The new wrong result this change would have introduced.

        Honest: the during-writes record is excluded, so the series is
        [0.40, 0.30] -- one point per fraction, each a single-client
        measurement over a single-client baseline. Old form, over the same
        records: a third point appears at 0.10 from an aggregate reader QPS
        divided by a single-client pre-churn baseline, on an axis where 1.0
        means "no degradation".

        **The magnitude is a property of the reader cost model, not a
        constant, and the 0.90 this test's own fixture produces is the
        fixture's arithmetic rather than a measurement.** Measured end to end
        over 9 real churn runs each: readers whose query releases the GIL (a
        `latency_s` sleep, which is what a real driver blocked on a socket
        does) give a spurious retention of **3.74 to 4.26**, 9 of 9 above 1.0;
        readers that are numpy-bound in-process (`latency_s = 0`) give **0.89
        to 2.62**, 5 of 9 above 1.0, with the single-client baseline itself
        ranging 4,336 to 12,076 QPS over those runs. So the hazard is "a
        retention that is not a retention", and the number below is the
        fixture's, asserted only to pin that the extra point appears at all.
        """
        pool_record = _during_writes_record()
        pool_record["qps"] = 900.0
        recs = self._records() + [pool_record]
        old = list(self._old_retention(recs).values())[0]
        assert [f for f, _ in old] == [0.10, 0.10, 0.10, 0.25, 0.25]
        # `sorted(points)` orders ties by QPS, so the reader pool's point is
        # found by value rather than by index. 0.90 is this fixture's
        # 900 / 1000 -- see the docstring for what the real range is and why
        # no single number belongs here.
        tens = [round(v, 2) for f, v in old if f == 0.10]
        assert 0.90 in tens, tens
        # Three points at one x where the fixed form has one: that is the
        # structural claim, and it holds whatever the magnitude turns out to
        # be on a given reader cost model.
        assert len(tens) == 3, tens
        fixed = list(charts_mod.churn_retention(recs, "d").values())[0]
        assert [round(v, 2) for _, v in fixed] == [0.40, 0.30]


def _tmp():
    import tempfile
    return tempfile.mkdtemp(prefix="t4b-charts-")


# ---------------------------------------------------------------------------
# The declarations that must agree across the container boundary
# ---------------------------------------------------------------------------

class TestTheDeclarationsAgree:
    """`harness/` must not import `report/` or `orchestrator/`, and the
    orchestrator runs on a host whose only Python dependency is pyyaml. So
    four constants are declared more than once, exactly as `DEFAULT_EF_MODE`
    and `DEFAULT_FILTERED_BUDGET_S` already are, and this is the test that
    pins each pair together."""

    def test_the_reader_default_is_the_same_number_in_three_places(self):
        from orchestrator import estimate as est_mod
        from orchestrator import ops_pass
        assert crw.DEFAULT_CHURN_READERS == 4
        assert ops_pass.DEFAULT_CHURN_READERS == crw.DEFAULT_CHURN_READERS
        assert (est_mod.DEFAULT_CHURN_READERS_ESTIMATE
                == crw.DEFAULT_CHURN_READERS)

    def test_the_baseline_window_is_the_same_number_in_two_places(self):
        from orchestrator import estimate as est_mod
        assert crw.READER_BASELINE_WINDOW_S == 30.0
        assert (est_mod.CHURN_READER_BASELINE_WINDOW_S
                == crw.READER_BASELINE_WINDOW_S)

    def test_the_ops_unit_axes_are_the_report_s_own(self):
        """The ratio's guard holds the ops unit constant, and "the ops unit"
        has to mean the same tuple on both sides of the container boundary or
        the harness is guarding a narrower thing than the report groups on.

        Equality and not a subset: a harness tuple MISSING an axis is a guard
        that lets a crossing through, and a harness tuple with an extra one
        refuses pairs the report would happily combine.
        """
        assert cw.OPS_UNIT_KEYS == gen.OPS_UNIT_FIELDS

    def test_the_flag_reaches_the_harness_and_the_orchestrator(self):
        from harness.main import parse_args
        base = ["--engine", "fake", "--host", "h", "--port", "1",
                "--dataset", "d", "--m", "16", "--run-id", "r",
                "--output", "/tmp/o.jsonl"]
        assert parse_args(base).churn_readers == crw.DEFAULT_CHURN_READERS
        assert parse_args(base + ["--churn-readers", "0"]).churn_readers == 0

        from orchestrator.ops_pass import harness_args
        source = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "harness", "main.py")).read()
        assert "readers=args.churn_readers" in source
        assert "driver_factory=driver_factory" in source
        del harness_args

    def test_a_cleared_profile_key_means_the_default_not_zero(self):
        """`ops.churn_readers: null` and `ops.churn_readers: 0` are different
        requests, unlike the budget keys where both mean "no cap".

        Honest: `null` emits 4, the default, because a cleared key is a
        profile that wants whatever the harness default is. Mutant (the
        `_budget_arg` idiom applied here, computed below): `null` emits 0,
        which silently turns the reader pool off -- and that changes what
        `extra.reinsert_rows_per_s` measures while the record still looks the
        same.
        """
        from orchestrator.ops_pass import _budget_arg, _churn_readers
        assert _churn_readers({}) == 4
        assert _churn_readers({"churn_readers": None}) == 4
        assert _churn_readers({"churn_readers": 0}) == 0
        assert _churn_readers({"churn_readers": 8}) == 8
        # What the budget idiom would have published for the same input.
        assert _budget_arg({"churn_readers": None}, "churn_readers", 4) == "0"

    def test_the_two_shipped_profiles_state_their_choice(self):
        """`main` measures the pool and `dev` runs the write-only pass. Both
        are explicit, so §4.3 item 7's readers=0 rate is a thing a shipped
        profile produces rather than a setting nobody runs."""
        from orchestrator.config import load_profile
        assert load_profile("main")["ops"]["churn_readers"] == 4
        assert load_profile("dev")["ops"]["churn_readers"] == 0
        # `smoke` is left on the default, so the gate profile exercises it.
        assert "churn_readers" not in load_profile("smoke")["ops"]


# ---------------------------------------------------------------------------
# Prose that is part of the deliverable (R7)
# ---------------------------------------------------------------------------

class TestTheProseSaysWhatTheCodeDoes:
    """Five defects in this repository were "code right, prose wrong", and two
    fix rounds in a row on the previous task were blocked on prose alone."""

    def _read(self, *parts):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return open(os.path.join(root, *parts)).read()

    def test_the_measurement_version_block_names_pr_4(self):
        """The version-5 block says why PR 4 sits inside version 5, and the
        claim it makes is checkable: no archived record carries a version at
        all.

        **The constant is 8, and this docstring once said 5 two lines above
        an assertion that it was 6.** G3 bumped it to 6, the one-client
        scaling status bumped it to 7, and version 8 -- the ops harness
        withdrawing a storage engine from the engines that have none --
        bumped it again; what this test is about is unchanged every time,
        because the question is whether the VERSION 5 BLOCK still states
        PR 4's three facts, not what the harness stamps today. Three times
        now the tail assertion has had to move with a bump, which is what
        makes it a tripwire rather than decoration: it is what brings
        whoever bumps the constant back here to re-read that block, and the
        three substring assertions above are what they confirm. Loosening it
        to `>=` would be the one change that silently retires it.

        **And the end of the block was being found by a string that no longer
        occurs.** It split on `"OPS_MEASUREMENT_VERSION = 5"`, which the bump
        left at zero occurrences, so `split(...)[0]` returned the whole rest
        of the file and the three substring assertions below scanned every
        later version block too. Measured against that widened scan, one of
        the three was defeated by it: `reinsert_rows_per_s` also occurs after
        the version-6 heading, so deleting it from the version-5 block would
        have passed, while `during_writes` and `2,134 ops records` occur
        nowhere later and would still have failed. One guard in three is
        enough to fix. The block now ends where the version-6 heading begins,
        and that heading's absence raises here rather than widening the scan.
        """
        source = self._read("harness", "metrics", "records.py")
        after_5 = source.split("VERSION 5 ALSO COVERS PR 4")[1]
        assert "VERSION 6 --" in after_5, (
            "the version-6 heading is gone, so the version-5 block has no "
            "end and this test would scan the rest of the file")
        block = after_5.split("VERSION 6 --")[0]
        assert "during_writes" in block
        assert "reinsert_rows_per_s" in block
        assert "2,134 ops records" in block
        # `>= 8` since version 9 (ops v9 bumped for F2-F11, a later change):
        # what this pins is that THIS change did not bump, not the value.
        assert records.OPS_MEASUREMENT_VERSION >= 8

    def test_a_version_5_figure_is_still_refused_against_an_older_one(self):
        """The guard the "no archived record carries a 5" argument rests on.

        The block's claim that PR 4 can sit inside version 5 depends on there
        being no version 5 figure anywhere for a readers-contended one to be
        incomparable with -- which was measured directly against the archived
        corpus (239 ops record files, 2,134 ops records, every one of them
        with `extra.ops_measurement_version` absent and therefore implicit
        version 1). That corpus lives outside the repository, so it is not
        asserted here: `tests/test_config.py:1528` is the one archived-run
        test and it skips when the tree is absent, and a second skipping test
        would be a second thing a reader has to decide whether to trust.

        What IS asserted is the property that makes the absence matter: 5 is
        not in `OPS_MEASUREMENT_EQUIVALENT_VERSIONS`, so a version 5 record
        and a version 1 one are refused rather than charted together. If that
        ever stopped holding, the archive's 2,134 version-1 records would
        start loading beside reader-contended version 5 ones and the argument
        in the block would be wrong rather than merely unverified here.
        """
        # `>= 8` since version 9 (ops v9 bumped for F2-F11, a later change):
        # what this pins is that THIS change did not bump, not the value.
        assert records.OPS_MEASUREMENT_VERSION >= 8
        assert 5 not in records.OPS_MEASUREMENT_EQUIVALENT_VERSIONS
        assert not records.ops_versions_measure_the_same_thing([1, 5])
        # And the pair the archive actually contains stays comparable.
        assert records.ops_versions_measure_the_same_thing([1, 2])

    def test_churn_state_does_not_claim_an_interface_it_lacks(self):
        """Item 1 of the fix round, pinned so it cannot drift back.

        `report/churn_state.py`'s docstring previously said `generate.py`
        "re-exports every public name, because `from report.generate import
        ...` is how the existing callers and tests reach them". None of its
        six public names were re-exported, and the precedent it cited behaves
        the opposite way: `report.loaders` really does keep every name of
        `report.measurement_versions` bound, and
        `test_the_public_names_are_still_importable_from_loaders` asserts the
        identity.

        The docstring now states the difference and why it is deliberate. Both
        halves are checked here: that the names are genuinely not re-exported,
        and that the cited precedent genuinely is.
        """
        import report.generate as generate_mod
        from report import loaders, measurement_versions
        for name in ("churn_measurements", "during_writes_rows",
                     "is_during_writes", "maintenance_state"):
            assert not hasattr(generate_mod, name), (
                f"report.generate re-exports {name}, so churn_state's "
                f"docstring is wrong in the other direction now")
            assert hasattr(churn_state, name), name
        # The precedent the docstring contrasts itself with.
        assert (loaders.declared_ops_version
                is measurement_versions.declared_ops_version)
        doc = churn_state.__doc__ or ""
        assert "does **not** re-export these names" in doc
        assert "a re-export here" in doc

    def test_the_inflight_bound_is_qualified_at_both_sites(self):
        """The per-client bound is true of a real driver and not in general.

        Measured against a zero-latency synthetic driver at 3 clients: 6
        discarded against 2,097-2,950 executed, 0.20-0.29%. The stop instant
        is read before the stop flag is set, so a query cheaper than that gap
        can be discarded several times over.

        Both sites are in this task's ownership and both are asserted, because
        one of them is a reader-facing footnote in the rendered report and
        fixing only the code comment would leave the published text wrong.
        """
        # Anchored on the clause rather than on the whole sentence: the
        # comment in the workload wraps between "at most" and "one per
        # client", so a sentence-level assertion would fail on a reflow that
        # changed nothing.
        workload = self._read("harness", "workloads", "concurrency.py")
        assert "one per client on any real driver" in workload
        assert "discarded several times over" in workload
        footnote = self._read("report", "render.py")
        assert "one per client on any real driver" in footnote
        assert "several times over" in footnote
        # And the rendered footnote really carries it, not just the source.
        from report.render import _concurrency_notes
        notes = _concurrency_notes({"concurrency": [{
            "engine": "mariadb", "dataset": "d", "clients": 32,
            "qps": 5820.0, "extra": {"scaling_efficiency": 0.43}}]})
        assert "on any real driver" in notes
        assert "several times over" in notes

    def test_the_methodology_states_what_the_ratio_is_and_is_not(self):
        doc = self._read("docs", "05-methodology.md")
        block = doc.split("**Reads are measured while the writer")[1].split(
            "## 3. Fairness")[0]
        assert "ops.churn_readers" in block
        assert "30 s before that step's deletes" in block
        assert "Both percentiles come from one pool" in block
        assert "Readers compute no recall" in block
        assert "ratio of wall clocks" in block
        # §4.3 item 7's pass has to be findable from the docs.
        assert "ops.churn_readers: 0" in block

    def test_the_qps_spread_is_quoted_as_a_range_not_a_single_sample(self):
        """R11, in the one form a string check can actually enforce.

        **Renamed from `test_no_numeric_claim_about_a_wall_clock_is_made_bare`,
        which promised far more than it delivered.** It checked that "6,028"
        was accompanied by "16,811" and that two fixed phrases existed, and it
        would not have caught any of the three wrong figures a review found in
        this task's prose: a line count of 907 that was really 981, a
        retention of "about four" that ranges 0.89-4.26 depending on the
        reader cost model, and a docstring saying 8.0/7.3 where its own
        assertions said 8.8/7.7. A test cannot check arbitrary arithmetic in
        prose; what it can check is that a spread is quoted as a spread, and
        that the two disclaimers that carry the R11 reasoning are still there.
        The two tests below cover the figures that CAN be made executable.
        """
        for parts in (("harness", "workloads", "churn_readers.py"),
                      ("report", "churn_state.py"),
                      ("report", "charts.py")):
            source = self._read(*parts)
            if "6,028" in source:
                assert "16,811" in source, parts
        readers = self._read("harness", "workloads", "churn_readers.py")
        assert "not a claim this code can make" in readers
        charts = self._read("report", "charts.py")
        assert "no mechanism is asserted for them" in charts
        # The retention figure is a range on both cost models, never a
        # single number. Matched against a whitespace-normalised copy of the
        # source: the second range wraps across a line in the docstring, and
        # pinning its exact indentation made a harmless reflow -- an editor
        # rewrapping a paragraph -- fail a test about a claim it did not
        # touch. The claim is that each figure is a RANGE; the line it
        # happens to break on is not part of it.
        state = " ".join(self._read("report", "churn_state.py").split())
        assert "**3.74 to 4.26**" in state
        assert "**0.89 to 2.62**" in state

    #: The phrase that marks a line-count figure as history rather than as a
    #: live claim. `churn.py` uses it to keep "907 and 107 over" -- the count
    #: before the module grew its own header -- on the record, and the
    #: no-restatement guard below exempts the clause it introduces.
    HISTORY_CLAUSE = "an earlier revision said"

    def test_the_line_count_the_split_is_justified_by_is_the_real_one(self):
        """Item 2, made executable so neither figure can drift again.

        **The figures are no longer quoted in the prose, and that is the
        fix.** `churn.py` and `churn_readers.py` used to justify the split
        with three numbers each, and those six restatements went stale three
        times: 907 and "107 over" before the module grew its own header
        docstring, 1,119/1,055/255 when an unrelated four-line addition moved
        `churn.py`, and again when PR 5 §5.1 item 4 added two interval keys.
        Two separate tasks spent a round correcting them.

        The numbers were never the argument. What a reader acts on is "the
        inlined file would be badly over the repo's 800-line ceiling", and
        what they need to be told about is that claim ceasing to be true. So
        the prose now cites this test and this test keeps the one load-bearing
        assertion, `inline > 800`. The splice is still computed and still
        parsed -- the claim is about a real file, not an arithmetic
        possibility -- and both files are still checked for the citation, so
        the prose cannot quietly drop the reference and leave the
        justification unsupported.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        churn_src = open(os.path.join(root, "harness", "workloads",
                                      "churn.py")).read()
        readers_src = open(os.path.join(root, "harness", "workloads",
                                        "churn_readers.py")).read()

        marker = "# PR 4 §4.3's reader pool."
        end = "from .context import RunContext\n"
        assert churn_src.count(marker) == 1 and churn_src.count(end) == 1
        body_start = ("#: Readers running while the writer re-inserts, per "
                      "PR 4 §4.3 item 2.")
        assert readers_src.count(body_start) == 1
        body = readers_src[readers_src.index(body_start):]

        spliced = (churn_src[:churn_src.index(marker)]
                   + churn_src[churn_src.index(end):])
        spliced = spliced.replace(end, end + "\n" + body, 1)
        inline = len(spliced.splitlines())
        together = (len(churn_src.splitlines())
                    + len(readers_src.splitlines()))

        # The spliced file is real code, not a line-count stunt.
        import ast
        ast.parse(spliced)

        # THE assertion. Everything above exists to make this one mean
        # something: if the two files ever shrink enough that inlining them
        # would fit inside the ceiling, the split is no longer justified by
        # the reason both files give, and whoever notices should be this
        # test rather than a reviewer.
        assert inline > 800, (
            f"splicing `churn_readers.py` back into `churn.py` gives "
            f"{inline} lines, inside the repo's 800-line maximum -- so the "
            f"justification both files state for the split no longer holds. "
            f"Either the split has a different reason now, in which case say "
            f"so in both files, or it can be undone.")

        # Both files still CITE this test rather than restating its
        # arithmetic. Anchored on the test's own name, so a file that drops
        # the reference fails here instead of leaving a bare claim with
        # nothing behind it.
        for source, where in ((churn_src, "churn.py"),
                              (readers_src, "churn_readers.py")):
            assert "800-line maximum badly" in source, where
            assert "test_the_line_count_the_split_is_justified_by_is_the" \
                "_real_one" in source, (
                f"{where} no longer points at the test that computes the "
                f"splice, so its claim about the 800-line ceiling is "
                f"unsupported prose again.")
            # And no figure is back. Checked by PATTERN, not against
            # today's value: an earlier version of this assertion compared
            # against `f"{together:,} lines"`, which only catches a
            # restatement that happens to be CORRECT -- and a correct
            # restatement is the harmless case. Measured: reintroducing
            # "the two files are 1,151 lines" passed that form, because
            # 1,151 was not the current count. The rot is a stale figure,
            # so the guard has to catch any figure.
            #
            # Three shapes, because a thousands separator is not the only
            # way to write one of these figures and an earlier version of
            # this pattern exempted two of the three that actually went
            # stale. Measured by injection into the passage:
            #
            #   "the two files are 1,147 lines"  -> caught by \d,\d{3}
            #   "the two files are 1147 lines"   -> needs \d{3,4}\s+lines
            #   "which is 282 over"              -> needs \d{3,4}\s+over
            #
            # "282 over" was one of the three figures that rotted, so the
            # separator-only form left the live spelling of a real defect
            # legal.
            #
            # The history clause is exempted BY ITS LABEL, not by digit
            # shape: `churn.py` correctly records that "an earlier revision
            # said 907 and 107 over", and "107 over" is indistinguishable
            # from a live restatement by pattern alone. Keying the exemption
            # to the sentence that declares itself history is the only way to
            # keep that legal while catching the same spelling used as a
            # current claim.
            marker = "800-line maximum badly"
            passage = source[source.index(marker):
                             source.index(marker) + 900]
            # Unwrapped before scanning, and that is not a nicety: in
            # `churn.py` this passage is a COMMENT, so the history clause
            # reads "... three times -- an\n# earlier revision said 907 ...".
            # The label straddles a line wrap, so a raw search for it found
            # nothing, the exemption did not apply, and the guard flagged the
            # history it is supposed to permit. Measured on the real file.
            # Same defect as the one `tests/test_t5b_stats.py` guards against
            # in rendered prose, met from the other side.
            flat = re.sub(r"\s+", " ", re.sub(r"\s*#\s*", " ", passage))
            scanned = flat
            if self.HISTORY_CLAUSE in scanned:
                head, _, tail = scanned.partition(
                    self.HISTORY_CLAUSE)
                # The labelled clause runs to the end of its dash-delimited
                # aside; everything after it is live prose again.
                _history, _, rest = tail.partition(" -- ")
                scanned = head + rest
            stale = re.search(r"\d,\d{3}|\d{3,4}\s+(?:over|lines)", scanned)
            assert stale is None, (
                f"{where} quotes a line count ({stale.group(0)!r}) in the "
                f"paragraph justifying the split. Those figures went stale "
                f"three times; cite this test instead of restating them. If "
                f"this is history rather than a claim, label it with "
                f"{self.HISTORY_CLAUSE!r} as `churn.py` does.")
        # `together` is still computed, because the assertion above needs it
        # to be a real measurement of the two real files.
        assert together > inline > 800, (together, inline)

    def test_the_maintenance_key_counts_are_the_ones_report_actually_has(self):
        """Item 5, made executable for the same reason.

        `churn._record_maintenance` now states how many times each of the four
        PR 4 keys occurs in `report/` -- 25, 14, 1 and 0 -- and that
        `repair_path`'s single occurrence is inside a docstring rather than a
        read. All four were 0 at the base of this task, which is what made the
        pair of records indistinguishable in the first place.

        The first two figures were 20 and 13 until PR 5 §5.1 item 6 grouped
        the churn table's tie column by maintenance state -- a degraded index
        tied to a repaired one would be this task's defect wearing §5.1's
        clothes -- and §5.2 cited the same pairing defect in the module
        header of `report/crosscheck.py`. The paragraph in `churn.py` says so
        too; this test is what stops either of them drifting from the
        directory.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import glob
        counts = {}
        for key in ("maintenance_state", "repair_path", "index_sync",
                    "maintenance"):
            total = 0
            for path in glob.glob(os.path.join(root, "report", "*.py")):
                total += open(path).read().count(key)
            counts[key] = total
        # `maintenance` counts `maintenance_state` too, which is why it is the
        # larger figure rather than a separate set of sites.
        # 26 rather than 25 since §1's per-engine table reads the resource
        # key `maintenance_bytes` in `report/provenance.py`: a substring hit,
        # not a churn read, and `churn.py`'s paragraph says so.
        assert counts == {"maintenance": 26, "maintenance_state": 14,
                          "repair_path": 1, "index_sync": 10}, counts
        source = open(os.path.join(root, "harness", "workloads",
                                   "churn.py")).read()
        block = source.split("**`repair_path` and `index_sync_s` are still")[1]
        for key, n in (("maintenance", 26), ("maintenance_state", 14),
                       ("repair_path", 1), ("index_sync", 10)):
            assert f"`{key}` {n}" in block, (key, n, block[:400])
