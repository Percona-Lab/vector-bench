"""The filtered budget can stop a query in flight, not only the next one.

percona-mariadb-20261111-r2 lost MariaDB's whole sift ops unit to ONE
filtered query. On `ctag` at 1% a warm-up query ran for over 240 s against a
60 s warm-up share; server RSS climbed from 2.3 GiB to the 16 GiB container
limit and the kernel killed the server (exit 137). The per-point budget was
checked only between queries, so nothing could stop it.

Each query is now capped server-side at the time left in its loop, and only
the engine's own timeout error is treated as "the budget stopped it". These
tests pin both halves: the cap reaches the wire in each engine's spelling,
and the workload turns a timeout into a budget-stopped point while every
other error still fails the unit.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional
from unittest import mock

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import datasets as ds                                  # noqa: E402
from harness.datasets import FILTER_CTAG, FILTER_TAG                 # noqa: E402
from harness.drivers.base import ConnectionSpec, RowAttrs            # noqa: E402
from harness.drivers.mysql_family import (                           # noqa: E402
    ER_QUERY_TIMEOUT_MYSQL, ER_STATEMENT_TIMEOUT_MARIADB, AliSQLDriver,
    MariaDBDriver, PerconaDriver)
from harness.drivers.postgres import PostgresDriver                  # noqa: E402
from harness.metrics import records                                  # noqa: E402
from harness.workloads import filtered as fw                         # noqa: E402
from harness.workloads.context import RunContext                     # noqa: E402
from tests.fake_driver import FakeDriver, index_spec, tiny_dataset   # noqa: E402
from tests.test_t3c_filtered import TickingClock, two_column_driver  # noqa: E402

K = 10


class Recorder:
    def __init__(self):
        self.sql: List[str] = []

    def execute(self, sql, params=None, **kw):
        self.sql.append(sql)


class ServerError(Exception):
    def __init__(self, errno=None, sqlstate=None):
        super().__init__(f"errno={errno} sqlstate={sqlstate}")
        self.errno = errno
        self.sqlstate = sqlstate


def _wired(cls):
    driver = cls(ConnectionSpec(host="h", port=1))
    driver._cur = Recorder()
    return driver


# ---------------------------------------------------------------------------
# The drivers: each engine's spelling, and its own error only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, armed, lifted", [
    (MariaDBDriver, "SET SESSION max_statement_time = 2.500",
     "SET SESSION max_statement_time = 0.000"),
    (AliSQLDriver, "SET SESSION max_execution_time = 2500",
     "SET SESSION max_execution_time = 0"),
    (PerconaDriver, "SET SESSION max_execution_time = 2500",
     "SET SESSION max_execution_time = 0"),
    (PostgresDriver, "SET statement_timeout = 2500",
     "SET statement_timeout = 0"),
])
def test_each_engine_arms_and_lifts_the_cap_in_its_own_spelling(
        cls, armed, lifted):
    driver = _wired(cls)
    assert driver.set_statement_timeout(2.5) is True
    assert driver.set_statement_timeout(None) is True
    assert driver.set_statement_timeout(0) is True
    assert driver._cur.sql == [armed, lifted, lifted]


@pytest.mark.parametrize("cls", [AliSQLDriver, PerconaDriver, PostgresDriver])
def test_a_sub_millisecond_cap_rounds_up_rather_than_becoming_no_cap(cls):
    """0 means OFF in milliseconds. Rounding 0.0004 s down would turn the
    smallest cap into none at all, the exact failure this exists to stop."""
    driver = _wired(cls)
    driver.set_statement_timeout(0.0004)
    assert driver._cur.sql[-1].endswith("= 1")


@pytest.mark.parametrize("cls, own, foreign", [
    (MariaDBDriver, ServerError(errno=ER_STATEMENT_TIMEOUT_MARIADB),
     ServerError(errno=ER_QUERY_TIMEOUT_MYSQL)),
    (PerconaDriver, ServerError(errno=ER_QUERY_TIMEOUT_MYSQL),
     ServerError(errno=ER_STATEMENT_TIMEOUT_MARIADB)),
    (AliSQLDriver, ServerError(errno=ER_QUERY_TIMEOUT_MYSQL),
     ServerError(errno=1213)),
    (PostgresDriver, ServerError(sqlstate="57014"),
     ServerError(sqlstate="57P01")),
])
def test_only_the_engine_s_own_timeout_error_counts(cls, own, foreign):
    driver = _wired(cls)
    assert driver.is_statement_timeout(own) is True
    assert driver.is_statement_timeout(foreign) is False
    # A reset connection, which is what the OOM kill looked like.
    assert driver.is_statement_timeout(
        ConnectionError("Connection reset by peer")) is False


def test_an_engine_without_a_cap_says_so_and_sends_nothing():
    driver = FakeDriver(dim=4)
    assert driver.set_statement_timeout(5.0) is False
    assert driver.is_statement_timeout(ServerError(errno=1969)) is False


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------

class CappedFake(FakeDriver):
    """A fake whose server enforces a cap, and stops chosen queries with it.

    `stall_at` is the 1-based query_filtered call the "server" stops; every
    arming is logged so a test can check what the cap was set to.
    """

    TIMEOUT = ServerError(errno=ER_STATEMENT_TIMEOUT_MARIADB)

    def __init__(self, *a, stall_at: Optional[int] = None,
                 fail_with: Optional[BaseException] = None, **kw):
        super().__init__(*a, **kw)
        self.stall_at = stall_at
        self.fail_with = fail_with or self.TIMEOUT
        self.caps: List[Optional[float]] = []
        self.filtered_calls = 0

    def set_statement_timeout(self, seconds):
        self.caps.append(seconds)
        return True

    def is_statement_timeout(self, exc):
        return exc is self.TIMEOUT

    def query_filtered(self, vector, k, threshold, column=FILTER_TAG):
        self.filtered_calls += 1
        if self.filtered_calls == self.stall_at:
            raise self.fail_with
        return super().query_filtered(vector, k, threshold, column=column)


@pytest.fixture
def ctx(tmp_path):
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="e30", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


def _capped(dataset, cache_dir, **kw) -> CappedFake:
    """`two_column_driver`'s load, on a fake that enforces a cap."""
    rows = len(dataset.train)
    attrs = RowAttrs(
        tag=ds.filter_values(cache_dir, dataset, FILTER_TAG, indexed_rows=rows),
        ctag=ds.filter_values(cache_dir, dataset, FILTER_CTAG,
                              indexed_rows=rows),
    )
    driver = CappedFake(dim=dataset.dim, metric=dataset.metric, **kw)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, attrs)
    return driver


def _run(ctx, driver, dataset, *, warmup, budget_s=100.0, step=1.0,
         max_queries=50):
    clock = TickingClock(driver, step=step)
    with mock.patch.object(fw.time, "time", clock):
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
               selectivities=(0.50,), columns=(FILTER_TAG,),
               max_queries=max_queries, warmup=warmup,
               indexed_rows=len(dataset.train), budget_s=budget_s)
    rows = list(records.read_records(ctx.recorder.path))
    assert len(rows) == 1
    return rows[0]


def test_a_timed_out_measured_query_ends_the_point_as_budget_bound(ctx):
    """Honest: queries 1..4 answered, the 5th stopped by the cap -- four
    executed, `time_bounded`, one timeout, and the cap lifted afterwards.
    Mutant (the timeout re-raised): the run raises instead of recording.
    Mutant (the stopped query counted): five latencies, not four."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir, stall_at=5)
    row = _run(ctx, driver, dataset, warmup=0)
    assert row["queries_executed"] == 4
    assert row["extra"]["time_bounded"] is True
    assert row["extra"]["statement_timeouts"] == 1
    assert row["extra"]["statement_timeout_armed"] is True
    assert driver.caps[-1] is None, "the cap must be lifted after the point"


def test_the_cap_is_the_time_left_and_never_below_the_floor(ctx):
    """Each query is armed with what remains of the budget, shrinking as
    the clock advances, floored at MIN_STATEMENT_TIMEOUT_S."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir)
    _run(ctx, driver, dataset, warmup=0, budget_s=10.0, step=4.0)
    armed = [c for c in driver.caps if c is not None]
    assert armed[0] == pytest.approx(10.0)
    assert armed[1] == pytest.approx(6.0)
    assert armed[2] == pytest.approx(2.0)
    assert all(c >= fw.MIN_STATEMENT_TIMEOUT_S for c in armed)


def test_a_timed_out_warm_up_query_truncates_only_the_warm_up(ctx):
    """The incident's shape: the WARM-UP query was the one that ran away.
    The warm-up stops, the measured loop still runs, and the point is not
    marked budget-bound -- `warmup_executed` says what happened, as it does
    when the warm-up's share elapses between queries."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir, stall_at=3)
    row = _run(ctx, driver, dataset, warmup=20, budget_s=1_000.0,
               max_queries=10)
    assert row["extra"]["warmup_executed"] == 2
    assert row["extra"]["statement_timeouts"] == 1
    assert row["extra"]["time_bounded"] is False
    assert row["queries_executed"] == 10


def test_any_other_error_still_fails_the_unit(ctx):
    """A reset connection is a dead server, not a slow point. Recording it
    as budget-bound would publish an OOM kill as a measurement."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir, stall_at=2,
                     fail_with=ConnectionError("Connection reset by peer"))
    with pytest.raises(ConnectionError):
        _run(ctx, driver, dataset, warmup=0)


def test_no_budget_leaves_the_session_alone(ctx):
    """`filtered_budget_s: 0` means no cap; arming a cap of infinity would
    be a statement for nothing, and some engines refuse the value."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir)
    row = _run(ctx, driver, dataset, warmup=2, budget_s=0)
    assert driver.caps == []
    assert row["extra"]["statement_timeout_armed"] is False
    assert row["extra"]["statement_timeouts"] == 0


def test_an_engine_without_a_cap_records_that_it_had_none(ctx):
    dataset = tiny_dataset()
    driver = two_column_driver(dataset, ctx.cache_dir)
    row = _run(ctx, driver, dataset, warmup=0)
    assert row["extra"]["statement_timeout_armed"] is False
    assert row["extra"]["statement_timeouts"] == 0


def test_no_query_is_allowed_more_than_the_ceiling_whatever_is_left(ctx):
    """The review's finding: "time left" alone gave a measured query most of
    a 300 s budget, about the 240 s the incident's query needed to OOM the
    server. Mutant (ceiling dropped): the first cap is 300."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir)
    _run(ctx, driver, dataset, warmup=0, budget_s=300.0, step=0.01,
         max_queries=3)
    armed = [c for c in driver.caps if c is not None]
    assert armed and max(armed) == pytest.approx(fw.MAX_STATEMENT_TIMEOUT_S)


def test_the_cap_is_lifted_after_a_warm_up_timeout_too(ctx):
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir, stall_at=1)
    _run(ctx, driver, dataset, warmup=5, max_queries=5)
    assert driver.caps[-1] is None


def test_the_cap_is_armed_outside_the_latency_span(ctx):
    """Arming is one more statement; inside the span it would be charged to
    the engine. Order of events: arm, then the span's clock, then the query.
    Mutant (arm moved after `perf_counter`): the order below breaks."""
    dataset = tiny_dataset()
    driver = _capped(dataset, ctx.cache_dir)
    events: List[str] = []
    real_arm = driver.set_statement_timeout
    real_query = driver.query_filtered

    def arm(seconds):
        events.append("arm" if seconds is not None else "lift")
        return real_arm(seconds)

    def query(*a, **kw):
        events.append("query")
        return real_query(*a, **kw)

    real_pc = fw.time.perf_counter

    def pc():
        events.append("clock")
        return real_pc()

    driver.set_statement_timeout = arm
    driver.query_filtered = query
    with mock.patch.object(fw.time, "perf_counter", pc):
        _run(ctx, driver, dataset, warmup=0, max_queries=3)
    queries = [i for i, e in enumerate(events) if e == "query"]
    assert len(queries) == 3
    for i in queries:
        # The span's start is the last thing before the query, so nothing --
        # in particular not the arming SET -- sits inside the span.
        assert events[i - 1] == "clock", events[max(0, i - 4):i + 1]
        assert "arm" in events[:i]
    assert events.count("arm") == 3


@pytest.mark.parametrize("module, name", [
    ("harness.drivers.myvector", "MYVECTOR_DIALECT"),
    ("harness.drivers.villagesql", "VILLAGESQL_DIALECT"),
])
def test_dialects_without_a_verified_cap_declare_none(module, name):
    """MyVector and VillageSQL decline the filtered workload; a cap nobody
    checked against those servers must not be switched on by inheritance."""
    import importlib
    dialect = getattr(importlib.import_module(module), name)
    assert dialect.statement_timeout == ""
    assert dialect.statement_timeout_errno is None
