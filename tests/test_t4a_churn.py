"""T4a: the churn driver hooks and the new churn sequence.

PR 4 §4.1 and §4.2, against `tests/fake_driver.py` and mocked cursors. No
engine is started.

**Why §4.4's prescribed tests are not the tests here.** Its first bullet asks
for "FakeDriver: tombstone penalty lowers recall after delete; `maintain()`
restores it". That is a fixture supplying the thing under test: the penalty and
the reclaim are both modelled *inside the fake*, so the assertion "recall came
back" holds whether or not `churn.run` calls `maintain()` at the point §4.2
puts it -- the fake's own model produces the number either way. It is built and
asserted below (`TestTheRepairPathIsMeasured`) because the recall story should
be legible, but the load-bearing guards are two others:

* the **call order**, measured off a recording driver whose log interleaves the
  hooks with the measurements and the record writes, so "called once" cannot
  stand in for "called before the baseline" (`TestTheSequenceIsTheOneSpecified`);
* the **recorded durations**, each proved against a hook made to consume a
  *different* amount of wall clock. Every timed quantity PR 4 asks for is
  seconds-valued, the records round to three decimals, and the fake returns
  instantly -- so `extra.index_sync_s == 0.0` is satisfied by a hook that ran
  and was fast, by a hook that never ran, and by a hook that returned the
  wrong thing alike. `index_sync_s` is given 0.25 s and `maintenance_s`
  0.15 s, two values apart by more than their jitter, so a swap fails and a
  missing call fails (`TestTheRecordedDurationsAreTheOnesSpent`).

Its second bullet -- "a driver without hooks produces exactly today's records
plus `repair_path`" -- is taken literally and then tightened: the guard below
asserts the **exact** `extra` key set, because "plus repair_path" is satisfied
by a record that also grew three keys nobody asked for.

**Mutation discipline.** Every guard names the regression it catches and was
checked by injecting that regression at run time (subclass / monkeypatch),
never by editing a file on disk -- CPython invalidates a `.pyc` on (whole-second
mtime, size), so a same-size rewrite inside one second can leave stale bytecode
valid and measure the previous version. Honest and mutant values are recorded
in each docstring and were read back out of the records the workload wrote.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds                                   # noqa: E402
from harness.datasets import FILTER_CTAG, FILTER_TAG, Dataset         # noqa: E402
from harness.drivers import mongo as mongo_mod                        # noqa: E402
from harness.drivers import postgres as pg_mod                        # noqa: E402
from harness.drivers import valkey as valkey_mod                      # noqa: E402
from harness.drivers.base import (ConnectionSpec, EngineDriver,       # noqa: E402
                                  RowAttrs)
from harness.metrics import records                                   # noqa: E402
from harness.metrics import stats                                     # noqa: E402
from harness.workloads import churn as cw                             # noqa: E402
from harness.workloads.context import RunContext                      # noqa: E402
from tests.fake_driver import (DIM, FakeDriver, K, N_QUERIES,         # noqa: E402
                               N_ROWS, index_spec, tiny_dataset)

#: What `wait_index_synced` and `maintain` are each made to spend in the
#: sequence tests. Deliberately unequal and deliberately far apart: equal
#: durations would let an implementation that recorded one hook's elapsed time
#: under the other's key pass, which is the mistake a pair of 0.0s hides
#: completely.
SYNC_SPEND_S = 0.25
MAINTAIN_SPEND_S = 0.15

#: Slack on a `time.sleep`-based duration. `sleep` never returns early but can
#: return late; 0.2 s is far more than the few milliseconds measured here and
#: still less than the 0.10 s gap between the two durations above, so the upper
#: bounds cannot be satisfied by the other hook's number.
SLEEP_SLACK_S = 0.2

#: The `extra` keys a completed churn step carried before PR 4, plus the one
#: `Recorder.write` stamps. Spelled out rather than captured from a run,
#: because a key set captured from the code under test asserts nothing.
PRE_PR4_STEP_KEYS = frozenset({
    "rows_churned",
    "delete_seconds",
    "insert_seconds",
    "reinsert_completed",
    "reinsert_rows_per_s",
    "recall_drop_vs_baseline",
    records.OPS_MEASUREMENT_VERSION_KEY,
})

#: What PR 4 adds to every completed step, hooks or no hooks: §4.2 item 4's
#: wait, §4.2 item 5's stamp, and §4.1 item 4's repair path.
PR4_UNCONDITIONAL_KEYS = frozenset({
    "index_sync_s", "maintenance_state", "repair_path",
})

#: What only a step that actually ran a repair path carries, per §4.2 item 6.
PR4_MAINTENANCE_KEYS = frozenset({"maintenance", "maintenance_s"})

#: What PR 5 §5.1 item 4 adds to every churn record that measured anything:
#: "Write extra.recall_ci95 and extra.qps_ci95 on calibration, filtered and
#: churn records." Unconditional on the engine, like PR 4's three above, so it
#: is folded into the exact-key assertions rather than being special-cased
#: there -- the point of those assertions is that a record cannot grow a key
#: nobody declared, and the way to keep that true is to declare the key.
#:
#: Taken from `harness.metrics.stats` rather than spelled as two literals: a
#: typo in a literal here would silently assert a key the harness never
#: writes, and the frozenset would still have the right size.
PR5_UNCONDITIONAL_KEYS = frozenset({stats.RECALL_CI_KEY, stats.QPS_CI_KEY})

#: What ops measurement version 9 adds to every completed step: the writer
#: pool and the load it is compared with, the accepted and indexed rates, the
#: post-delete wait, which recall rule `recall_at_k` is with the other one
#: beside it, and when the index size was read. Declared, for the reason
#: PR 5's keys are: the exact-key assertions below exist so a record cannot
#: grow a key nobody declared. (`index_vector_format` and
#: `index_includes_vectors` are absent here only because the fake's
#: `capabilities()` does not declare them; every real driver does.)
V9_STEP_KEYS = frozenset({
    "reinsert_writers", "ingest_threads",
    "reinsert_rows_per_s_accepted", "reinsert_rows_per_s_indexed",
    "index_sync_after_delete_s",
    "recall_kind", "recall_id_overlap", "size_read_at",
})

#: And to the baseline: the wait before it, the recall rule, the size read.
V9_BASELINE_KEYS = frozenset({
    "index_sync_before_baseline_s", "recall_kind", "recall_id_overlap",
    "size_read_at",
})

#: Every driver class the harness can instantiate, taken from the registry
#: rather than listed here.
#:
#: A literal tuple was the first form and it has a hole in exactly one
#: direction: a SIXTH driver class that overrode `maintain()` and inherited
#: `repair_path = "none"` would leave every assertion below passing, because
#: the class it contradicts is the one nobody added to the list. Drift by
#: omission is invisible to a loop over a list of the things you remembered.
#: `_driver_classes()` is also what `get_driver` dispatches on, so a class
#: absent from it is a class the harness cannot be asked for -- which makes
#: this the right definition of "every driver" rather than merely a
#: convenient one.
DRIVER_CLASSES = tuple(sorted(pg_mod._driver_classes().values(),
                              key=lambda cls: cls.__name__))


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="t4a-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


@pytest.fixture
def dataset():
    return tiny_dataset()


def written(ctx: RunContext) -> List[Dict]:
    return list(records.read_records(ctx.recorder.path))


def true_columns(cache_dir: str, dataset: Dataset) -> Tuple[numpy.ndarray,
                                                            numpy.ndarray]:
    """Both filter columns over the whole corpus, from the dataset module.

    The same `filter_values` call the filtered workload's ground truth is
    built from, which is what makes "the engine holds the right ctag" a claim
    about agreement with the yardstick rather than about agreement with
    whatever churn happened to compute.
    """
    rows = len(dataset.train)
    return (ds.filter_values(cache_dir, dataset, FILTER_TAG, indexed_rows=rows),
            ds.filter_values(cache_dir, dataset, FILTER_CTAG, indexed_rows=rows))


def two_column_driver(dataset: Dataset, cache_dir: str,
                      cls=FakeDriver, **kwargs) -> FakeDriver:
    """A loaded driver whose `tag` and `ctag` genuinely differ.

    `fake_driver.loaded_driver` loads a bare array, which `RowAttrs.coerce`
    mirrors into `ctag`; a corpus loaded that way cannot tell a churn that
    re-inserted the real cluster ids from one that mirrored, because both
    columns already agree everywhere.
    """
    tags, ctags = true_columns(cache_dir, dataset)
    driver = cls(dim=dataset.dim, metric=dataset.metric, **kwargs)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, RowAttrs(tag=tags, ctag=ctags))
    return driver


class _HooklessDriver(FakeDriver):
    """A FakeDriver with the three churn hooks taken back off.

    This is the "driver without hooks" of §4.4's second bullet -- MariaDB,
    AliSQL and (for `maintain`) Valkey and Percona Search. Written as three
    explicit rebindings to the base class's own functions rather than as a
    driver built from scratch, so what runs here is exactly the no-op default
    `harness/drivers/base.py` ships and not a second implementation of it.
    """

    prepare_churn = EngineDriver.prepare_churn
    wait_index_synced = EngineDriver.wait_index_synced
    maintain = EngineDriver.maintain


class _TracingDriver(FakeDriver):
    """A FakeDriver that logs what the workload did to it, in order.

    The log is the point. PR 4 §4.2 fixes the sequence -- `prepare_churn`
    before the baseline, `wait_index_synced` after the re-insert and before
    the post-churn measurement, `maintain` between the two records -- and a
    per-hook counter cannot distinguish that sequence from any permutation of
    it. Queries are collapsed into one `("measure", n)` event per run of
    consecutive queries, with the count, so two measurements with nothing
    between them do not read as one.
    """

    def __init__(self, *args, **kwargs):
        self.trace: List[Tuple[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _event(self, name: str, detail: Any = None) -> None:
        self.trace.append((name, detail))

    def query(self, vector, k: int):
        if self.trace and self.trace[-1][0] == "measure":
            self.trace[-1] = ("measure", self.trace[-1][1] + 1)
        else:
            self._event("measure", 1)
        return super().query(vector, k)

    def delete_ids(self, ids):
        self._event("delete", len(ids))
        return super().delete_ids(ids)

    def insert_rows(self, ids, vectors, attrs):
        if self.trace and self.trace[-1][0] == "insert":
            self.trace[-1] = ("insert", self.trace[-1][1] + len(ids))
        else:
            self._event("insert", len(ids))
        return super().insert_rows(ids, vectors, attrs)

    def prepare_churn(self) -> None:
        self._event("prepare_churn")
        return super().prepare_churn()

    def wait_index_synced(self, expected_rows: int) -> float:
        self._event("wait_index_synced", int(expected_rows))
        return super().wait_index_synced(expected_rows)

    def maintain(self):
        self._event("maintain")
        return super().maintain()


class _TracingRecorder(records.Recorder):
    """A Recorder that logs each write into a shared trace.

    Appended to the *driver's* trace so the record writes are interleaved with
    the engine calls in one ordered list. Two separate logs could not answer
    "was the before record written before `maintain()` ran?", which is the
    property that keeps a completed measurement on disk when a repair path
    fails.
    """

    def __init__(self, path: str, trace: List[Tuple[str, Any]]):
        super().__init__(path)
        self._trace = trace

    def write(self, record) -> None:
        state = (record.extra or {}).get("maintenance_state")
        self._trace.append(("record", state or record.notes))
        return super().write(record)


def run_churn(ctx: RunContext, driver, dataset: Dataset, **kwargs) -> None:
    """`churn.run` with this suite's shared arguments.

    `indexed_rows=N_ROWS` because the fixture loads the whole corpus and the
    ctag cache key is the row count; omitting it would cluster a different
    number of rows and relabel them.
    """
    options = dict(fractions=(0.10,), max_queries=N_QUERIES,
                   indexed_rows=N_ROWS)
    options.update(kwargs)
    cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512, **options)


# ---------------------------------------------------------------------------
# §4.1 -- the hooks and their defaults
# ---------------------------------------------------------------------------

class TestTheDefaultsAreNoOps:
    """§4.1's "optional methods with no-op defaults", checked on the base
    class itself rather than through a driver that might override them."""

    def test_the_three_defaults_do_nothing_and_say_so(self):
        assert EngineDriver.prepare_churn.__doc__
        hookless = _HooklessDriver(dim=DIM)
        assert hookless.prepare_churn() is None
        # Exactly 0.0, and the float matters: `churn.run` rounds this into a
        # record, and a default returning None would reach `round()` as a
        # TypeError inside a workload rather than here.
        waited = hookless.wait_index_synced(1_000_000)
        assert isinstance(waited, float) and waited == 0.0
        assert hookless.maintain() is None

    def test_wait_index_synced_ignores_its_argument_rather_than_refusing_it(
            self):
        """The default is for engines with no asynchronous index at all, so
        every row count is as synced as every other. Checked across three
        orders of magnitude because a default that validated its argument
        would make an engine that has nothing to wait for raise."""
        hookless = _HooklessDriver(dim=DIM)
        assert [hookless.wait_index_synced(n) for n in (0, 1, 10 ** 9)] == \
            [0.0, 0.0, 0.0]


class TestTheDeclaredRepairPathMatchesTheHook:
    """`extra.repair_path` is a declared class attribute, and the one way it
    can lie is by disagreeing with `maintain()`.

    A record carrying `maintenance: "VACUUM (ANALYZE)"` beside
    `repair_path: "none"` would say in one key that a repair ran and in
    another that the engine has none, and a report reading either would be
    right. The pair is therefore asserted against each other rather than
    each against a literal.
    """

    def test_the_base_default_is_none(self):
        assert EngineDriver.repair_path == "none"

    def test_the_registry_is_what_is_checked_and_it_is_not_empty(self):
        """The loops below iterate `DRIVER_CLASSES`, and a loop over an empty
        tuple passes every assertion in it. Two different assertions are
        needed here and an earlier revision of this docstring credited the
        wrong one with both jobs:

          * the count pin catches DIVERGENCE -- `DRIVER_CLASSES` built from
            something other than the registry, the empty tuple included. It
            cannot catch a registry that lost an entry, because
            `known_drivers()` derives from the same dict, so a shrunken
            registry shrinks both sides and `len == len` still holds.
          * the `>= 5` literal is what catches that SHRINKAGE. It is a
            hard-coded floor on purpose: the harness drives five
            architectures today and losing one is a regression, not a
            configuration.

        Measured on both, because which assertion fires is the whole point
        and an earlier revision of this docstring had them the wrong way
        round. `DRIVER_CLASSES = ()`: fails the COUNT PIN at `0 == 5` and the
        `>= 5` line never executes. The registry losing `MongoDriver`: passes
        the count pin at `4 == 4` and fails the `>= 5` LITERAL. Either way
        both loops below still pass, their bodies executing 0 and 4 times --
        which is why this test exists rather than being folded into them.
        """
        assert len(DRIVER_CLASSES) == len(pg_mod.known_drivers()), (
            f"{len(DRIVER_CLASSES)} classes against "
            f"{len(pg_mod.known_drivers())} in known_drivers(): "
            f"DRIVER_CLASSES is not the registry")
        assert len(DRIVER_CLASSES) >= 5, DRIVER_CLASSES
        assert pg_mod.PostgresDriver in DRIVER_CLASSES

    def test_exactly_the_drivers_that_override_maintain_declare_a_path(self):
        """Honest: pgvector and Percona Server are the `(overrides, declares)`
        pairs; the rest are `(inherits, "none")`.
        Mutant (`PostgresDriver.repair_path` left at the inherited "none"):
            pgvector reads (True, "none") and this fails -- which is the
            record that would have said "this engine has no repair path"
            beside a `VACUUM (ANALYZE)` it had just run.

        Stated as a biconditional rather than as a list, which is why this one
        did NOT have to change when Percona arrived: a driver that implements
        `maintain()` and forgets `repair_path`, or declares a path it cannot
        run, fails here whatever its name is.
        """
        for cls in DRIVER_CLASSES:
            overrides = cls.maintain is not EngineDriver.maintain
            declares = cls.repair_path != "none"
            assert overrides == declares, (
                f"{cls.__name__}: maintain overridden={overrides} but "
                f"repair_path={cls.repair_path!r}")

    def test_exactly_the_two_repairing_drivers_name_their_path(self):
        """The drivers that have a repair path, and what each calls it.

        **This used to read "only pgvector has one", and it fired when
        `PerconaDriver` landed** -- which is the direction it should fire in.
        Percona Server is the second engine here with a repair path and the
        only one in the MySQL family: `ALTER TABLE t1 DROP KEY vi, ADD VECTOR
        KEY vi (…)`, which rebuilds the graph from the live rows and so
        collects the tombstones a DELETE leaves behind. Returning `None` from
        `maintain()` makes the churn workload write one record per fraction
        instead of a before/after pair.

        **That last sentence used to end "losing the measurement this engine
        is uniquely able to supply", and that is now false at the pinned
        version** -- see `PerconaDriver.declare_bulk_vector_index_build`. The
        ALTER is the same statement
        `capabilities.bulk_vector_index_build: false` records as killing the
        server; measured, it returned in 1-4 ms and changed nothing in all 20
        before/after pairs, so the pair was the degraded index measured twice
        and published as a repair. The driver now withdraws the repair path
        per instance when the host says so, and **this assertion is unaffected
        because it reads the CLASS attribute**, which still declares the
        mechanism the engine has. Declaring a mechanism and being able to
        execute it at a given pin are two different claims, and this test is
        about the first.

        `==`, not `in`: "none" is a substring of plenty, and this repo has
        already shipped seven assertions where a substring was satisfied by a
        longer string. The expected map is exhaustive for the same reason --
        a third repairing driver must fail here rather than pass.
        """
        # MyVector is the third: neither of its incremental paths can remove
        # a deleted row from the graph, so its repair is a full
        # `myvector_index_build` (`MyVectorDriver.maintain`).
        expected = {"PostgresDriver": "vacuum_analyze",
                    "PerconaDriver": "index-replace",
                    "MyVectorDriver": "index-rebuild"}
        assert len(DRIVER_CLASSES) >= 6, DRIVER_CLASSES
        for cls in DRIVER_CLASSES:
            want = expected.get(cls.__name__, "none")
            assert cls.repair_path == want, (cls.__name__, cls.repair_path)
        # Non-vacuity: both names really are present among the drivers, so
        # the map above is not describing classes that are not there.
        found = {cls.__name__ for cls in DRIVER_CLASSES}
        assert set(expected) <= found, set(expected) - found


# ---------------------------------------------------------------------------
# §4.1 item 1 -- pgvector, against a mocked cursor
# ---------------------------------------------------------------------------

class _FakeCursor:
    """Records every statement, optionally raising on one of them."""

    def __init__(self, raise_on: Optional[str] = None):
        self.statements: List[str] = []
        self.raise_on = raise_on

    def execute(self, sql: str, *args, **kwargs) -> None:
        self.statements.append(sql)
        if self.raise_on is not None and self.raise_on in sql:
            raise RuntimeError("canceling statement due to lock timeout")


def pg_driver(raise_on: Optional[str] = None):
    driver = pg_mod.PostgresDriver(ConnectionSpec(host="h", port=5432))
    cursor = _FakeCursor(raise_on=raise_on)
    driver._cur = cursor
    return driver, cursor


SET_AUTOVACUUM_OFF = "ALTER TABLE t1 SET (autovacuum_enabled = false)"
RESET_AUTOVACUUM = "ALTER TABLE t1 RESET (autovacuum_enabled)"
VACUUM_ANALYZE = "VACUUM (ANALYZE) t1"


class TestPgvectorTakesAutovacuumOffAndGivesItBack:
    """§4.1 item 1 and §4.4's third bullet.

    **Every assertion here is on a whole statement string, never a substring,
    and the reason is specific rather than hygiene:** `"RESET (autovacuum_
    enabled)"` CONTAINS `"SET (autovacuum_enabled)"` -- "RESET" ends in "SET"
    -- so `assert "SET (autovacuum_enabled" in sql` is satisfied by the
    statement that turns the option back ON. A substring check here would pass
    an implementation that never disabled anything.
    """

    def test_prepare_churn_disables_autovacuum_on_the_table(self):
        driver, cursor = pg_driver()
        driver.prepare_churn()
        assert cursor.statements == [SET_AUTOVACUUM_OFF]

    def test_maintain_vacuums_analyzes_and_then_resets(self):
        """Honest: two statements, in that order, and the description the
        record carries is the statement text.
        Mutant (`RESET` dropped): statements == [VACUUM_ANALYZE] and the table
            keeps `autovacuum_enabled = false` for as long as it exists,
            because that is a persistent TABLE option and not a session SET.
        """
        driver, cursor = pg_driver()
        what = driver.maintain()
        assert cursor.statements == [VACUUM_ANALYZE, RESET_AUTOVACUUM]
        assert what == "VACUUM (ANALYZE)"

    def test_autovacuum_is_restored_when_the_vacuum_raises(self):
        """§4.4 bullet 3, and the acceptance line "autovacuum restored on
        error". A claim in a docstring is part of the deliverable, so the
        claim is proved by raising rather than by reading the `finally`.

        Honest: the RuntimeError surfaces AND `RESET` ran, last.
        Mutant (`try/finally` replaced by two plain statements): the
            RuntimeError still surfaces, `cursor.statements` is
            [VACUUM_ANALYZE] alone, and `RESET_AUTOVACUUM in statements` is
            False -- measured by subclassing `PostgresDriver` with the
            unprotected body at run time.
        """
        driver, cursor = pg_driver(raise_on="VACUUM")
        with pytest.raises(RuntimeError):
            driver.maintain()
        assert cursor.statements == [VACUUM_ANALYZE, RESET_AUTOVACUUM]
        # Last, not merely present: a RESET issued before the VACUUM would
        # satisfy membership and restore nothing.
        assert cursor.statements[-1] == RESET_AUTOVACUUM

    def test_a_failed_repair_is_not_reported_as_a_repair_that_ran(self):
        """The exception is re-raised rather than swallowed into a `None`.

        A bare `except Exception: return None` here would leave the churn
        workload writing one record per fraction and the report saying
        pgvector has no repair path, which is how a bug becomes a published
        result. Honest: RuntimeError out. Mutant (`except Exception: return
        None`): `maintain()` returns None and nothing anywhere says the
        VACUUM failed.
        """
        driver, _cursor = pg_driver(raise_on="VACUUM")
        with pytest.raises(RuntimeError):
            driver.maintain()

    def test_a_failed_prepare_is_not_swallowed_either(self):
        """A DDL failure here means the measurements that follow are not
        taken in the state they claim. `explain_uses_vector_index` swallows
        exceptions because a plan check must not end a run; this is the setup
        the measurement is defined in terms of."""
        driver, _cursor = pg_driver(raise_on="ALTER TABLE")
        with pytest.raises(RuntimeError):
            driver.prepare_churn()


# ---------------------------------------------------------------------------
# §4.1 item 2 -- Valkey delegates
# ---------------------------------------------------------------------------

class TestValkeyDelegatesToTheExistingWait:
    def test_it_passes_the_row_count_as_expected_docs_and_returns_the_wait(
            self):
        """§4.1 item 2 says "delegate to the existing
        `_wait_for_mutations(expected_docs)`", and a delegation is a claim
        about a call: which function, which argument, whose return value.

        Honest: one call, `expected_docs=4321`, and 7.5 comes back out.
        Mutant (`self._wait_for_mutations()` with the argument dropped):
            calls == [{}] and the wait settles on two equal readings of the
            document count instead of comparing against the number the
            caller knows to expect -- the exact trap
            `_wait_for_mutations`' own docstring was written about.
        """
        calls: List[Dict[str, Any]] = []
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))

        def fake_wait(expected_docs=None, timeout_s=None,
                      docs_expectation=None):
            calls.append({"expected_docs": expected_docs,
                          "timeout_s": timeout_s,
                          "docs_expectation": docs_expectation})
            return 7.5

        driver._wait_for_mutations = fake_wait
        assert driver.wait_index_synced(4321) == 7.5
        # `timeout_s` is now passed rather than defaulted, and the value is
        # pinned against the constant rather than against a literal 900.0 --
        # the point of passing it is that the hook can compare the returned
        # elapsed time against it, which a default it never sees cannot do.
        # C3 added `docs_expectation`: the delegation now also says which way
        # it expects the document count to move, because the comparison this
        # wait applies is the delete path's unless the caller says otherwise.
        # Still exact dict equality, and no weaker for the extra key --
        # injected into the driver's call and run: `expected_docs + 1` fails
        # here, half the ceiling fails here, and `DOCS_AT_MOST` fails here.
        # The stub's `timeout_s` default is `None` rather than 900.0 for the
        # same reason `docs_expectation`' is: while it was 900.0 it equalled
        # `INDEX_SYNC_TIMEOUT_S`, so OMITTING the argument recorded the right
        # number anyway and this assertion passed -- measured, that mutant
        # passed before the default changed and fails after it.
        assert calls == [{"expected_docs": 4321,
                          "timeout_s": valkey_mod.INDEX_SYNC_TIMEOUT_S,
                          "docs_expectation": valkey_mod.DOCS_EQUAL}]
        assert valkey_mod.INDEX_SYNC_TIMEOUT_S > 0
        assert valkey_mod.DOCS_EQUAL != valkey_mod.DOCS_AT_MOST

    def test_a_settled_wait_publishes_what_it_rested_on(self):
        """§4.1 item 2's delegation is weaker than its name, and the record is
        where that has to be visible. Valkey would otherwise publish a bare
        `index_sync_s` with nothing beside it, in the engine whose writes have
        actually stopped returning mid-churn and cost two runs.

        Honest: a note naming the queue, the comparison the insert path now
            waits on and the delete path's one it replaced, which `churn.run`
            carries into the record as `extra.index_sync_note`.
        Mutant (`index_sync_note` never set -- the shipped state before this
            round, 0 occurrences of the name in `valkey.py`): `None`, the key
            is absent from the record, and a reader has a 0.0 with no way to
            tell it from a synchronous engine or an unimplemented hook.

        The wait is stubbed here, so the note carries no readings; C3's own
        file covers the rendered observation. What this one holds is the
        delegation's prose contract.
        """
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        driver._wait_for_mutations = lambda expected_docs=None, **kw: 3.25
        assert driver.index_sync_note is None
        assert driver.wait_index_synced(2000) == 3.25

        note = driver.index_sync_note
        assert note, "a wait with no reason beside it is the thing this fixes"
        # Anchored on delimiters: "3.2s" is a substring of "13.2s", and
        # "2,000" of "12,000".
        assert "settled in 3.2s" in note, note
        assert "<= 2,000`" in note, note
        assert "mutation queue" in note and "delete path" in note, note
        # And it must NOT say the queue is the only signal left. That was the
        # round's first claim, and C3 made it false outright: the insert path
        # passes `DOCS_EQUAL`, so the document count binds on every poll of
        # every re-insert, not only while the index overcounts.
        assert "only signal in force" not in note, note

    def test_a_timed_out_wait_says_so_rather_than_reading_as_settled(self):
        """`_wait_for_mutations` signals its timeout only by printing to
        stderr: the value it RETURNS on giving up is a plain elapsed time,
        indistinguishable from a successful long wait. So the ceiling is
        passed explicitly and compared against.

        Honest: a note saying the index had not absorbed the re-insert and
            that what follows measures an index still catching up -- the same
            thing `MongoDriver`'s equivalent says.
        Mutant (one unconditional note, this round's first form): with the
            wait returning 900.0 the record carried `index_sync_s: 900.0`
            beside "waited 900.0s on FT.INFO's mutation queue, which was the
            only signal in force" -- naming none of timeout / timed out /
            still catching up / WARNING / had not, measured. That is worse
            than the bare zero it replaced: ambiguous became wrong, in a
            state this engine is known to reach.
        """
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        timeout = valkey_mod.INDEX_SYNC_TIMEOUT_S
        driver._wait_for_mutations = (
            lambda expected_docs=None, timeout_s=None, **kw: timeout_s)
        assert driver.wait_index_synced(2000) == timeout

        note = driver.index_sync_note
        assert "gave up" in note, note
        assert "still catching up" in note, note
        assert "settled in" not in note, note

    def test_the_document_half_still_binds_when_the_index_overcounts(self):
        """What a bare `expected_docs` means, which is the DELETE path's
        comparison and the default of this wait.

        **Read this as a statement about `DOCS_AT_MOST`, not about the insert
        path.** C3 made the direction an argument: `wait_index_synced` now
        passes `DOCS_EQUAL` and the count binds there (see
        `tests/test_c3_valkey_sync.py`). What is measured below is the
        default, which stays `num_docs <= expected_docs` precisely so that
        this call -- and `scripts/verify-valkey-churn.py`, and the two
        `test_config.py` callers -- keep the behaviour they were written
        against. Under that comparison the count is dead while the index
        UNDERCOUNTS and binds while it overcounts, which is what these two
        numbers say.

        Honest: 1 poll at `num_docs` 1,800 against an expected 2,000 (the
            comparison is already true, so the queue decides), and 4 polls at
            2,481 against the same 2,000, which is the 481-document overcount
            `_wait_for_mutations`' own docstring was written about. There the
            count is what bound it.
        Mutant (the default flipped to `DOCS_EQUAL`, which is the shape that
            would need no third argument at the insert call site): 12 polls
            and 16 polls -- a comparison no caller of this signature asked
            for, measured.
        """
        def polls_for(num_docs: int, expected: int) -> int:
            driver = valkey_mod.ValkeyDriver(
                ConnectionSpec(host="h", port=6379))
            seen = {"n": 0}

            def ft_info():
                seen["n"] += 1
                # Queue drained throughout, so the count is the only thing
                # that can still end -- or fail to end -- the wait. The
                # overcount decays so the wait terminates.
                return {"mutation_queue_size": 0,
                        "num_docs": max(num_docs - (seen["n"] - 1) * 200, 0)}

            driver._ft_info = ft_info
            with mock.patch.object(valkey_mod.time, "sleep", lambda _s: None):
                driver._wait_for_mutations(expected_docs=expected,
                                           timeout_s=60.0)
            return seen["n"]

        assert polls_for(1_800, 2_000) == 1
        assert polls_for(2_481, 2_000) == 4

    def test_the_row_count_is_an_int_whatever_the_workload_passes(self):
        """`count_rows()` is an int today, but the hook's argument reaches a
        `num_docs <= expected_docs` comparison against a value parsed out of
        FT.INFO. A numpy int64 compares fine; a float from a future caller
        would too, and then the heartbeat's `{expected_docs:,}` format would
        print 2,000.0. Coerced at the boundary instead."""
        seen: List[Any] = []
        driver = valkey_mod.ValkeyDriver(ConnectionSpec(host="h", port=6379))
        driver._wait_for_mutations = lambda expected_docs=None, **kw: (
            seen.append(expected_docs) or 0.0)
        driver.wait_index_synced(numpy.int64(2000))
        assert seen == [2000] and type(seen[0]) is int


# ---------------------------------------------------------------------------
# §4.1 item 3 -- Percona Search, against a mocked collection
# ---------------------------------------------------------------------------

class _FakeCollection:
    """Enough of a pymongo collection for the sync probe.

    `hits_after` is how many polls return nothing before the newest row
    appears; `None` means it never appears, which is the timeout path.
    """

    def __init__(self, doc_id: int, vector, hits_after: Optional[int] = 0):
        self.doc_id = int(doc_id)
        self.vector = vector
        self.hits_after = hits_after
        self.aggregates = 0
        self.find_ones = 0
        self.inserted: List[Dict[str, Any]] = []

    def find_one(self, query, projection=None):
        self.find_ones += 1
        if int(query["_id"]) != self.doc_id:
            return None
        return {"_id": self.doc_id,
                "embedding": mongo_mod.encode_vector(self.vector)}

    def aggregate(self, pipeline):
        self.aggregates += 1
        if self.hits_after is None or self.aggregates <= self.hits_after:
            return iter(())
        return iter([{"_id": self.doc_id}])

    def insert_many(self, docs, ordered=True):
        self.inserted.extend(docs)


@pytest.fixture
def mongo_driver(monkeypatch):
    """A MongoDriver with no server and no pymongo.

    `Binary` is absent on a host built from the documented prerequisites --
    `tests/test_config.py:1956` skips for exactly that -- so `encode_vector`
    would hand `None(payload, 9)` a call. Patched to the identity on the
    payload, which is what BinData is on the wire anyway, so `decode_vector`
    reads back what `encode_vector` wrote.
    """
    monkeypatch.setattr(mongo_mod, "Binary", lambda payload, subtype: payload)
    monkeypatch.setattr(mongo_mod, "SYNC_POLL_S", 0.01)
    driver = mongo_mod.MongoDriver(ConnectionSpec(host="h", port=27017))
    driver._index = index_spec(tiny_dataset())
    return driver


class TestPerconaSearchWaitsForMongotToCatchUp:
    """§4.1 item 3. **The premise is unverified here and that is expected.**
    "`insert_rows` does not appear to wait for mongot to catch up" needs a
    live server to confirm and is the human task H1; what is tested is the
    wait itself -- that it polls, that it stops, and that a zero carries a
    reason rather than being a bare zero nobody can interpret.
    """

    def test_encode_and_decode_round_trip(self, mongo_driver):
        """`wait_index_synced` searches for the newest row's own vector, which
        means reading that vector back out of mongod. A decode that dropped
        the wrong number of header bytes would shift every float by one byte
        and the probe would search for a vector no document holds -- it would
        then time out, which reads as "mongot is slow"."""
        vector = numpy.arange(DIM, dtype=numpy.float32) / 3.0
        back = mongo_mod.decode_vector(mongo_mod.encode_vector(vector))
        assert numpy.array_equal(back, vector)

    def test_a_zero_wait_comes_with_a_reason_when_nothing_was_inserted(
            self, mongo_driver):
        """H1's acceptance is "index_sync_s non-zero or the reason documented",
        and a bare 0.0 documents nothing: it is what a synchronous engine
        records, what an unimplemented hook records, and what a hook that
        found the row immediately records.

        Honest: 0.0 with a non-empty `index_sync_note`, which `churn.run`
        carries into the record as `extra.index_sync_note`.
        Mutant (`index_sync_note` left None): the record has a 0.0 and no key
            beside it, and the three cases are indistinguishable.
        """
        assert mongo_driver._newest_inserted_id is None
        assert mongo_driver.wait_index_synced(2000) == 0.0
        note = mongo_driver.index_sync_note
        assert note and "no document to wait for" in note

    def test_it_polls_until_mongot_returns_the_newest_row(self, mongo_driver):
        """Honest: three aggregates (two empty, one hit), a non-zero wait, and
        a note naming the row and the poll count.
        Mutant (the loop replaced by a single probe returning 0.0): one
            aggregate, 0.0 seconds, and the measurement that follows is taken
            against an index holding none of the re-inserted rows.
        """
        vector = numpy.ones(DIM, dtype=numpy.float32)
        coll = _FakeCollection(doc_id=2_199, vector=vector, hits_after=2)
        mongo_driver._coll = coll
        mongo_driver._newest_inserted_id = 2_199

        waited = mongo_driver.wait_index_synced(2_200)
        assert coll.aggregates == 3
        assert waited >= 2 * 0.01, waited
        # Anchored on the delimiters either side, instances 9 and 10 of the
        # shape that has bitten this repo ten times. Measured: `"2,199" in
        # note` is satisfied by a note about row **12,199**, and
        # `"3 poll(s)" in note` by one reporting **13 polls** -- both of which
        # a wait that probed the wrong row, or polled four times as long as it
        # says, would produce.
        note = mongo_driver.index_sync_note
        assert "row 2,199 " in note, note
        assert "(3 poll(s))" in note, note

    def test_a_first_poll_hit_is_labelled_as_an_upper_bound(self,
                                                            mongo_driver):
        """A wait that found the row immediately has measured a bound on the
        lag, not the lag. Said in the note, because the alternative is a
        record with a near-zero `index_sync_s` that a reader takes for
        "mongot keeps up"."""
        coll = _FakeCollection(doc_id=7, vector=numpy.zeros(DIM, numpy.float32),
                               hits_after=0)
        mongo_driver._coll = coll
        mongo_driver._newest_inserted_id = 7
        mongo_driver.wait_index_synced(8)
        assert coll.aggregates == 1
        assert "upper bound" in mongo_driver.index_sync_note

    def test_the_timeout_is_a_recorded_outcome_rather_than_a_hang(
            self, mongo_driver, monkeypatch):
        """Honest: it returns the time it spent, warns, and the note says the
        measurement that follows is of an index still catching up.
        Mutant (`while True` with no timeout branch): the call never returns
            and the run holds the machine until someone notices -- the exact
            failure two lost Valkey runs were.
        """
        monkeypatch.setattr(mongo_mod, "SYNC_TIMEOUT_S", 0.05)
        coll = _FakeCollection(doc_id=11, vector=numpy.zeros(DIM, numpy.float32),
                               hits_after=None)
        mongo_driver._coll = coll
        mongo_driver._newest_inserted_id = 11

        waited = mongo_driver.wait_index_synced(12)
        assert waited >= 0.05, waited
        assert "still catching up" in mongo_driver.index_sync_note

    def test_a_probe_error_is_treated_as_not_yet_synced_and_not_as_a_crash(
            self, mongo_driver, monkeypatch):
        """mongot answers with an error while it is rebuilding or
        unreachable, which is the state being waited for. A poll that raised
        would turn "not caught up yet" into a dead run."""
        monkeypatch.setattr(mongo_mod, "SYNC_TIMEOUT_S", 0.05)

        class Angry(_FakeCollection):
            def aggregate(self, pipeline):
                self.aggregates += 1
                raise RuntimeError("PlanExecutor error: index is rebuilding")

        coll = Angry(doc_id=3, vector=numpy.zeros(DIM, numpy.float32))
        mongo_driver._coll = coll
        mongo_driver._newest_inserted_id = 3
        waited = mongo_driver.wait_index_synced(4)
        assert waited >= 0.05
        assert coll.aggregates >= 2

    def test_a_missing_document_is_loud_rather_than_a_silent_wait(
            self, mongo_driver):
        """The caller has just written this id. Its absence means the write
        did not land, and every number taken afterwards describes a corpus
        nobody loaded -- so this raises instead of waiting out the timeout and
        recording a number."""
        coll = _FakeCollection(doc_id=5, vector=numpy.zeros(DIM, numpy.float32))
        mongo_driver._coll = coll
        mongo_driver._newest_inserted_id = 99  # not the one the collection has
        with pytest.raises(RuntimeError, match="no embedding for row 99"):
            mongo_driver.wait_index_synced(100)

    def test_insert_rows_remembers_the_highest_id_it_wrote(self,
                                                           mongo_driver):
        """`max`, not "the last element": churn hands over a slice of a list
        that is ascending today, and a probe for the wrong id would wait out
        the whole timeout and then report an index that was never behind.

        Honest: 42 from [7, 42, 13]. Mutant (`ids[-1]`): 13, and the probe
        waits for a row mongot already had.
        """
        coll = _FakeCollection(doc_id=0, vector=numpy.zeros(DIM, numpy.float32))
        mongo_driver._coll = coll
        vectors = numpy.zeros((3, DIM), dtype=numpy.float32)
        mongo_driver.insert_rows([7, 42, 13], vectors,
                                 numpy.array([1, 2, 3], dtype=numpy.int64))
        assert mongo_driver._newest_inserted_id == 42
        assert len(coll.inserted) == 3


# ---------------------------------------------------------------------------
# §4.2 -- the sequence, which is the load-bearing guard
# ---------------------------------------------------------------------------

class TestTheSequenceIsTheOneSpecified:
    def test_the_hooks_land_between_the_right_measurements(self, ctx,
                                                           dataset, tmp_path):
        """§4.2 items 1-6 as one ordered trace, over TWO fractions.

        This is the guard §4.4's first bullet is not. The recall story it
        prescribes is produced by the fake's own model and holds whatever
        `churn.run` does with the hooks; this fails unless each hook is called
        at the point §4.2 puts it.

        Honest (read back out of `driver.trace`) -- ops measurement
        version 9 added the wait before the baseline and the one after the
        deletes, because Percona Search's mongot was measured before it had
        caught up at both points on `full-20260825`:

            prepare_churn
            wait_index_synced 2000
            measure 70   record 'baseline before any churn'
            delete 200   wait_index_synced 1800   insert 200
            wait_index_synced 2000
            measure 70   record 'before'
            maintain
            measure 70   record 'after'
            delete 300   wait_index_synced 1700   insert 300
            wait_index_synced 2000
            measure 70   record 'before'
            maintain
            measure 70   record 'after'

        Mutants, each injected at run time by reordering the calls in a
        subclass of the workload's own steps:
          * `prepare_churn` moved after the baseline -> the first two events
            swap and the baseline is measured in a state the later numbers
            are not comparable with.
          * `wait_index_synced` moved after the post-churn measurement -> the
            measurement it exists to protect happens first, and the record
            still carries a plausible `index_sync_s`.
          * the before record written after `maintain()` -> the 'before' and
            'maintain' events swap, and a repair path that raises takes a
            finished measurement with it.
          * `prepare_churn` called per fraction -> it appears twice.
        """
        trace: List[Tuple[str, Any]] = []
        recorder = _TracingRecorder(str(tmp_path / "trace.jsonl"), trace)
        local = RunContext(
            run_id="t4a-trace", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x", k=K,
            cache_dir=ctx.cache_dir)
        driver = two_column_driver(
            dataset, ctx.cache_dir, cls=_TracingDriver, ef_half=1.0,
            ef_search=512, tombstone_penalty=1.0,
            index_sync_s=0.0, maintain_latency_s=0.0,
            maintain_description="reclaimed the tombstones")
        driver.trace = trace
        try:
            run_churn(local, driver, dataset, fractions=(0.10, 0.25))
        finally:
            recorder.close()

        per_measure = N_QUERIES + min(20, N_QUERIES)
        step = lambda rows: [                              # noqa: E731
            ("delete", rows), ("wait_index_synced", N_ROWS - rows),
            ("insert", rows),
            ("wait_index_synced", N_ROWS),
            ("measure", per_measure), ("record", "before"),
            ("maintain", None),
            ("measure", per_measure), ("record", "after"),
        ]
        assert trace == (
            [("prepare_churn", None),
             ("wait_index_synced", N_ROWS),
             ("measure", per_measure),
             ("record", "baseline before any churn")]
            + step(200) + step(300)
        )

    def test_prepare_churn_runs_once_however_many_fractions(self, ctx,
                                                            dataset):
        """§4.2 item 1 says "once before the baseline". Counted over three
        fractions, because once-per-fraction and once-per-run agree at one
        fraction -- which is what `config/profiles` mostly declares and so
        what a one-fraction test would bless.

        Read off `FakeDriver.churn_hook_calls`, the fake's own ordered log,
        rather than off the tracing subclass: the log is what
        `tests/fake_driver.py` advertises for exactly this, and reading it
        here means the two are checked against each other.

        Honest: one `prepare_churn`, first, and three `wait_index_synced`
            calls after it. Mutant (the call moved inside the fraction loop):
            four, measured -- one before the baseline and one per fraction.
        """
        driver = two_column_driver(dataset, ctx.cache_dir, cls=_TracingDriver,
                                   ef_half=1.0, ef_search=512)
        run_churn(ctx, driver, dataset, fractions=(0.10, 0.25, 0.50))
        hooks = [name for name, _arg in driver.churn_hook_calls]
        assert hooks.count("prepare_churn") == 1, driver.churn_hook_calls
        assert hooks[0] == "prepare_churn"
        # Three fractions, seven waits since ops version 9: one before the
        # baseline and two per fraction (after the deletes, after the
        # re-insert). The once-per-run hook and the per-fraction hooks are
        # counted in the same list, so a change that made one of them follow
        # the other's schedule shows up here.
        assert hooks.count("wait_index_synced") == 7
        # And the two logs agree, which is what makes either trustworthy.
        assert [e for e in driver.trace if e[0] == "prepare_churn"] == \
            [("prepare_churn", None)]

    def test_the_wait_is_told_the_row_count_the_corpus_should_have(self, ctx,
                                                                   dataset):
        """§4.2 item 4 passes `n_rows`, the count before any churn, because
        after a complete step the corpus is whole again. Mutant (`count` --
        the number of rows churned): 200 reaches a Valkey wait whose condition
        is `num_docs <= expected_docs`, which a 2,000-document index can never
        satisfy, so every churn step would spend the full 900 s timeout and
        then warn that the index had not caught up.
        """
        driver = two_column_driver(dataset, ctx.cache_dir, cls=_TracingDriver,
                                   ef_half=1.0, ef_search=512)
        run_churn(ctx, driver, dataset, fractions=(0.10, 0.25))
        waits = [e[1] for e in driver.trace if e[0] == "wait_index_synced"]
        # Before the baseline, then per step: after the deletes (the corpus
        # minus the churned rows), after the re-insert (whole again).
        assert waits == [N_ROWS, N_ROWS - 200, N_ROWS, N_ROWS - 300, N_ROWS]
        assert driver.count_rows() == N_ROWS


class TestTheRecordedDurationsAreTheOnesSpent:
    def test_index_sync_s_and_maintenance_s_are_each_their_own_hook(
            self, ctx, dataset):
        """Finding 1, which is the one that makes this suite worth running.

        Both quantities are seconds, both round to three decimals in the
        record, and a fake that returns instantly makes 0.0 the answer for
        "the hook ran and was fast", "the hook never ran" and "the hook
        returned the wrong thing". So the two hooks are made to spend
        DIFFERENT amounts of wall clock and each recorded value is asserted
        against its own.

        Honest: index_sync_s in [0.25, 0.45] and maintenance_s in
            [0.15, 0.35], and the first strictly greater than the second.
        Mutants:
          * `extra.index_sync_s` hard-coded to 0.0 -> 0.0, outside the band.
          * the two values swapped -> 0.15 and 0.25, and the strict
            inequality fails. A `>= 0` or `is not None` assertion passes all
            three of these, which is why neither is used.
          * `driver.wait_index_synced` never called and 0.0 written ->
            0.0, outside the band.
        """
        driver = two_column_driver(
            dataset, ctx.cache_dir, ef_half=1.0, ef_search=512,
            tombstone_penalty=1.0, index_sync_s=SYNC_SPEND_S,
            maintain_latency_s=MAINTAIN_SPEND_S,
            maintain_description="reclaimed the tombstones")
        run_churn(ctx, driver, dataset)

        rows = written(ctx)
        before = [r for r in rows
                  if r["extra"].get("maintenance_state") == "before"]
        after = [r for r in rows
                 if r["extra"].get("maintenance_state") == "after"]
        assert len(before) == len(after) == 1

        synced = before[0]["extra"]["index_sync_s"]
        maintained = after[0]["extra"]["maintenance_s"]
        assert SYNC_SPEND_S <= synced <= SYNC_SPEND_S + SLEEP_SLACK_S, synced
        assert MAINTAIN_SPEND_S <= maintained <= MAINTAIN_SPEND_S + \
            SLEEP_SLACK_S, maintained
        # The gap is wider than the slack, so this cannot be satisfied by the
        # other hook's number under any jitter these two sleeps produce.
        assert synced > maintained

        # The wait is recorded on the "before" record, which is the one it
        # protects; the "after" record carries the same step's value, because
        # the index was only waited for once.
        assert after[0]["extra"]["index_sync_s"] == synced
        # And the maintenance keys appear only where maintenance happened.
        assert PR4_MAINTENANCE_KEYS.isdisjoint(before[0]["extra"])

    def test_the_sync_reason_travels_with_the_number(self, ctx, dataset):
        """A zero `index_sync_s` is only interpretable with a reason beside
        it, and the reason is an optional driver attribute the workload reads
        through `getattr` -- the same shape `_write_shape` already uses for
        Valkey's write batch.

        Honest: `extra.index_sync_note` is the driver's note.
        Mutant (`_sync_shape` returning {}): the key is absent and a 0.0 in
            the record is indistinguishable from an unimplemented hook.
        """
        class Noting(FakeDriver):
            index_sync_note = "mongot already had the row at the first poll"

        driver = two_column_driver(dataset, ctx.cache_dir, cls=Noting,
                                   ef_half=1.0, ef_search=512)
        run_churn(ctx, driver, dataset)
        step = written(ctx)[1]
        assert step["extra"]["index_sync_note"] == \
            "mongot already had the row at the first poll"

    def test_an_engine_with_nothing_to_say_carries_no_note(self, ctx,
                                                           dataset):
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=1.0,
                                   ef_search=512)
        run_churn(ctx, driver, dataset)
        assert "index_sync_note" not in written(ctx)[1]["extra"]


# ---------------------------------------------------------------------------
# §4.4 bullet 2 -- the hookless driver
# ---------------------------------------------------------------------------

class TestADriverWithoutHooksRecordsWhatItAlwaysDid:
    def test_exactly_todays_records_plus_the_unconditional_keys(self, ctx,
                                                                dataset):
        """§4.4's second bullet, taken literally and then tightened.

        "Exactly today's records plus `repair_path`" is read as "the same
        number of records, and the same `extra` keys, plus what §4.2 adds
        unconditionally" -- `index_sync_s` (item 4), `maintenance_state`
        (item 5) and `repair_path` (§4.1 item 4). The literal reading of the
        bullet contradicts §4.2 items 4 and 5, which are not conditional on
        the engine; this is reported rather than guessed at.

        The assertion is on the EXACT key set, because "plus repair_path" is
        also satisfied by a record that quietly grew four other keys.

        Honest: 2 records for one fraction (baseline + step), the step's keys
            exactly PRE_PR4 | PR4_UNCONDITIONAL, `repair_path == "none"`,
            `maintenance_state == "before"`, `index_sync_s == 0.0`.
        Mutant (the maintenance branch gated on truthiness instead of on
            `isinstance(what, str)`): still 2 records here, because the base
            default returns None -- but 3 for a plain FakeDriver, which is
            what `TestTheTypeGate` below measures.
        """
        driver = two_column_driver(dataset, ctx.cache_dir, cls=_HooklessDriver,
                                   ef_half=1.0, ef_search=512,
                                   tombstone_penalty=1.0)
        run_churn(ctx, driver, dataset)

        rows = written(ctx)
        assert len(rows) == 2
        assert [r["churn_fraction"] for r in rows] == [0.0, 0.10]
        assert rows[0]["notes"] == "baseline before any churn"

        step = rows[1]
        assert set(step["extra"]) == PRE_PR4_STEP_KEYS | \
            PR4_UNCONDITIONAL_KEYS | PR5_UNCONDITIONAL_KEYS | V9_STEP_KEYS
        assert step["extra"]["repair_path"] == "none"
        assert step["extra"]["maintenance_state"] == "before"
        assert step["extra"]["index_sync_s"] == 0.0
        # The numbers it always recorded are still there and still numbers.
        assert step["extra"]["rows_churned"] == 200
        assert step["extra"]["reinsert_completed"] is True
        assert step["recall_at_k"] is not None

    def test_the_baseline_record_is_untouched(self, ctx, dataset):
        """§4.2 item 2: "Baseline measurement (unchanged)". The baseline
        carries no `extra` of its own today beyond the version stamp, and
        stamping a `maintenance_state` on it would invite a report to plot a
        pre-churn point on a maintenance series."""
        driver = two_column_driver(dataset, ctx.cache_dir, cls=_HooklessDriver,
                                   ef_half=1.0, ef_search=512)
        run_churn(ctx, driver, dataset)
        baseline = written(ctx)[0]
        # PR 5 §5.1 item 4 puts the two interval keys on every churn record,
        # the baseline included -- and it has to, because the baseline is the
        # x=0 point every churn fraction is measured against, so a report
        # that drew error bars on the churned points and none on the baseline
        # would show the comparison as exact at the one place it is not.
        # `maintenance_state` is still absent, which is what this test was
        # named for.
        assert set(baseline["extra"]) == {
            records.OPS_MEASUREMENT_VERSION_KEY} | PR5_UNCONDITIONAL_KEYS \
            | V9_BASELINE_KEYS
        assert "maintenance_state" not in baseline["extra"]


class TestTheTypeGateOnMaintain:
    def test_a_non_string_return_is_not_a_repair_path(self, ctx, dataset):
        """`maintain()` is typed `Optional[str]` by §4.1 and the value lands
        in `extra.maintenance`, which a human reads. The gate is that type and
        not truthiness.

        `tests/fake_driver.py`'s `maintain()` predates PR 4 and returns the
        COUNT of rows it reclaimed -- 400 at `tests/test_fake_driver.py:274`,
        100 at `tests/test_t3b_drivers.py:1999`, neither of which this task
        may edit.

        Honest: 2 records, no `maintenance` key, and
            `tests/test_workloads.py`'s churn tests keep their record counts.
        Mutant (`if what:` truthiness): 3 records, the third carrying
            `maintenance: 200` -- an integer under a key documented as a short
            description -- and `test_records_a_baseline_and_each_churn_step`
            fails on `[0.0, 0.10, 0.25]`, measured.
        """
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=1.0,
                                   ef_search=512, tombstone_penalty=1.0)
        assert driver.maintain_description is None
        run_churn(ctx, driver, dataset)

        rows = written(ctx)
        assert len(rows) == 2
        assert PR4_MAINTENANCE_KEYS.isdisjoint(rows[1]["extra"])
        assert "after" not in [r["extra"].get("maintenance_state")
                               for r in rows]

    def test_an_empty_description_is_not_a_repair_path_either(self, ctx,
                                                              dataset):
        """`""` is a str and is falsy, so the two halves of the gate disagree
        about it. A record carrying `maintenance: ""` says a repair ran and
        declines to say which, so the empty string is treated as "no repair
        path" and that choice is pinned rather than left to whichever half of
        the condition is written first."""
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=1.0,
                                   ef_search=512, tombstone_penalty=1.0,
                                   maintain_description="")
        run_churn(ctx, driver, dataset)
        rows = written(ctx)
        assert len(rows) == 2
        assert PR4_MAINTENANCE_KEYS.isdisjoint(rows[1]["extra"])


# ---------------------------------------------------------------------------
# §4.4 bullet 1 -- the recall story, built rather than hoped for
# ---------------------------------------------------------------------------

class TestTheRepairPathIsMeasured:
    def test_tombstones_cost_recall_and_maintenance_gives_it_back(self, ctx,
                                                                  dataset):
        """§4.4's first bullet. **The mechanism here is supplied by the
        fixture**: `tombstone_penalty` and the reclaim both live in
        `tests/fake_driver.py`, so this says the records carry the model's own
        numbers in the right places -- not that the workload called the hooks
        at the right time, which is `TestTheSequenceIsTheOneSpecified`'s job.

        What it does add over that: the two records are stamped, they are
        distinguishable, and the "after" recall is a different number from
        the "before" one, which is what a report needs in order to have
        anything to plot. A pair of identical recalls would mean the second
        measurement was a copy of the first.

        Honest (2,000 rows, 10% churned, penalty 1.0, ef_half 1.0, read out
            of the records): baseline recall 1.0, before 0.896, after 0.998;
            index_bytes 256,000 -> 281,600 -> 256,000. The drop is the
            tombstones and it comes back when they go.
        Mutant (`maintain()` never called by the workload): two records,
            recall stuck at 0.896, index_bytes stuck at 281,600, and the
            report has no repaired point to plot.
        """
        driver = two_column_driver(
            dataset, ctx.cache_dir, ef_half=1.0, ef_search=512,
            tombstone_penalty=1.0,
            maintain_description="reclaimed the tombstones")
        run_churn(ctx, driver, dataset)

        baseline, before, after = written(ctx)
        assert [r["extra"].get("maintenance_state")
                for r in (baseline, before, after)] == \
            [None, "before", "after"]
        assert before["recall_at_k"] < baseline["recall_at_k"], (
            baseline["recall_at_k"], before["recall_at_k"])
        assert after["recall_at_k"] > before["recall_at_k"], (
            before["recall_at_k"], after["recall_at_k"])
        # Back to the baseline, not merely upward: the tombstones were the
        # whole cost in this model, so anything short of it would mean the
        # reclaim ran against a state the churn had changed in another way.
        assert after["recall_at_k"] == pytest.approx(
            baseline["recall_at_k"], abs=0.02)
        assert after["extra"]["maintenance"] == "reclaimed the tombstones"
        assert after["extra"]["repair_path"] == "fake_maintain"
        # The index shrinks when the tombstones go, which is the other half
        # of what the churn phase reports.
        #
        # **This is a fixture artifact and NOT a substitute for the ordering
        # trace**, which matters because it is the kind of thing a later
        # reader "simplifies" by deleting the trace test. It does happen to
        # fire on one reordering -- moving the whole before-record write past
        # `maintain()` re-evaluates `driver.index_bytes()` after the reclaim
        # and this reads `assert 256000 < 256000`. But it fires only here:
        # `FakeDriver.index_bytes()` is `len(rows.ids) * m * 8` and counts
        # tombstones, while `maintain()` physically drops them from
        # `rows.ids`. Real pgvector reports `pg_relation_size(INDEX)`, and
        # `VACUUM (ANALYZE)` returns pages to the free space map WITHOUT
        # truncating an HNSW index, so on a live run the two sizes are
        # typically equal and that reordering would move nothing. It also
        # cannot fire at `tombstone_penalty=0`. And `assert 256000 < 256000`
        # tells a maintainer nothing about ordering, where the trace failure
        # names the two events that swapped.
        assert after["index_bytes"] < before["index_bytes"]

    def test_the_after_record_states_its_own_drop_from_the_baseline(
            self, ctx, dataset):
        """Honest: the "after" record's `recall_drop_vs_baseline` is computed
        from ITS own recall. Mutant (the "before" record's value copied
        across): the repaired record claims the degraded record's drop, and
        a reader plotting drop-vs-fraction sees maintenance achieve nothing.
        """
        driver = two_column_driver(
            dataset, ctx.cache_dir, ef_half=1.0, ef_search=512,
            tombstone_penalty=1.0, maintain_description="vacuumed")
        run_churn(ctx, driver, dataset)
        baseline, before, after = written(ctx)
        for row in (before, after):
            assert row["extra"]["recall_drop_vs_baseline"] == pytest.approx(
                baseline["recall_at_k"] - row["recall_at_k"], abs=1e-6)
        assert after["extra"]["recall_drop_vs_baseline"] < \
            before["extra"]["recall_drop_vs_baseline"]


# ---------------------------------------------------------------------------
# The churn re-insert's second filter column
# ---------------------------------------------------------------------------

class TestTheReinsertedRowsCarryTheirOwnCtag:
    def test_each_reinserted_row_gets_the_cluster_its_vector_belongs_to(
            self, ctx, dataset):
        """The churn mirror, which was a real latent defect.

        `assign_tags(n_rows)[original_ids]` handed to `RowAttrs.coerce` gave
        every re-inserted row `ctag = tag` instead of the cluster id its
        vector belongs to, so a `ctag`-filtered measurement taken after a
        churn would have been scored against ground truth computed from the
        real clustering. It was latent because `main.py` runs filtered
        (`:305`) before churn (`:317`) in fixed source order, and because the
        failure signature -- the two filter modes agreeing -- reads as a
        finding rather than as a bug.

        **The mapping from a re-inserted row back to its original is recovered
        from the VECTOR, not from churn's RNG.** Re-deriving the victim
        selection with the same seed and the same expressions would be the
        test recomputing the implementation's own arithmetic; matching the
        stored vector against `dataset.train` is an independent answer to
        "which row is this?".

        Honest: 200 of 200 re-inserted rows carry the true `ctag`, and 196 of
            200 carry a `ctag` that DIFFERS from their `tag` -- so the
            agreement is not vacuous. Corpus-wide the two columns differ on
            1,983 of 2,000 rows.
        Mutant (`RowAttrs.coerce(assign_tags(n_rows)[original_ids])`
            restored): 200 of 200 carry `ctag == tag`, 4 of 200 match the
            true ctag -- the ones that coincide by chance -- and
            `differs_from_tag` is 0.
        """
        true_tags, true_ctags = true_columns(ctx.cache_dir, dataset)
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=1.0,
                                   ef_search=512)
        run_churn(ctx, driver, dataset)

        rows = driver._store.rows
        fresh = numpy.nonzero(rows.ids >= N_ROWS)[0]
        assert len(fresh) == 200, len(fresh)

        # vector -> original row index, built from the dataset rather than
        # from anything the workload computed.
        origin = {dataset.train[i].tobytes(): i for i in range(N_ROWS)}
        assert len(origin) == N_ROWS, "the fixture has duplicate vectors"

        matched_ctag = 0
        differs_from_tag = 0
        for position in fresh:
            original = origin[rows.vectors[position].tobytes()]
            assert rows.tags[position] == true_tags[original]
            if rows.ctags[position] == true_ctags[original]:
                matched_ctag += 1
            if rows.ctags[position] != rows.tags[position]:
                differs_from_tag += 1

        assert matched_ctag == len(fresh), (matched_ctag, len(fresh))
        # Two-sided: without this, a mirror that happened to equal the true
        # clustering would satisfy the line above. `tag` is `id % 100` and
        # `ctag` is a nearest-pivot id, so they agree only by coincidence.
        assert differs_from_tag > len(fresh) * 0.9, differs_from_tag

    def test_the_bare_array_path_still_exists_and_still_mirrors(self):
        """`_reinsert` is called directly with a bare array by
        `tests/test_config.py` and `tests/test_t3b_drivers.py`, and by two
        diagnostic scripts that have no cache directory to compute a cluster
        assignment from. That path still coerces, and coercion still mirrors
        -- which is why `run` does not use it.

        Pinned so the mirror cannot quietly become the normal path again: if
        this ever has to change, the call sites are named here.
        """
        seen: List[RowAttrs] = []

        class Watching(FakeDriver):
            def insert_rows(self, ids, vectors, attrs):
                seen.append(attrs)
                return super().insert_rows(ids, vectors, attrs)

        driver = Watching(dim=DIM)
        driver.connect()
        written_rows, _s = cw._reinsert(
            driver, list(range(10)), numpy.zeros((10, DIM), numpy.float32),
            numpy.arange(10), budget_s=60.0)
        assert written_rows == 10
        assert seen and all(isinstance(a, RowAttrs) for a in seen)
        for attrs in seen:
            assert numpy.array_equal(attrs.tag, attrs.ctag)
            # A copy, not one buffer under two names.
            assert attrs.tag is not attrs.ctag


# ---------------------------------------------------------------------------
# The churn budget: `ops.churn_budget_s: null` and `: 0`
# ---------------------------------------------------------------------------

class _SlowInsertDriver(FakeDriver):
    """A FakeDriver whose re-insert costs real time per slice.

    Needed because the budget is a wall-clock deadline: against an instant
    write path a budget of 0 and a budget of 1800 both complete, so the
    conversion under test would be unobservable.
    """

    slice_cost_s = 0.01

    def insert_rows(self, ids, vectors, attrs):
        time.sleep(self.slice_cost_s)
        return super().insert_rows(ids, vectors, attrs)


class TestANonPositiveChurnBudgetMeansNoCap:
    @pytest.mark.parametrize("declared", [0.0, -1.0])
    def test_the_whole_reinsert_runs_rather_than_none_of_it(self, ctx,
                                                            dataset,
                                                            declared):
        """The enforcement half of the `ops.churn_budget_s: null` defect.

        `null` reached `parse_args` as the string "None" and killed the ops
        unit with SystemExit(2), taking concurrency and filtered with churn
        because `harness/main.py` wraps all three in one `except`. The
        orchestrator now emits "0" for that spelling -- and `0` had to stop
        meaning "a deadline that fires before the first slice", which is the
        same trap T3c found in the filtered loop, where every point recorded
        ZERO queries.

        Honest (budget 0.0 and -1.0, 50 slices of 10 ms): 200 of 200 rows
            re-inserted, `reinsert_completed` True, recall recorded.
        Mutant (`effective_budget = budget_s`, the conversion removed): 0 rows
            written, `reinsert_completed` False, `recall_at_k` absent, and the
            record says the re-insert did not complete within the budget --
            measured for both spellings.
        """
        driver = two_column_driver(dataset, ctx.cache_dir,
                                   cls=_SlowInsertDriver, ef_half=1.0,
                                   ef_search=512)
        run_churn(ctx, driver, dataset, budget_s=declared)

        rows = written(ctx)
        assert len(rows) == 2
        step = rows[1]
        assert step["extra"]["reinsert_completed"] is True
        assert step["extra"]["rows_churned"] == 200
        assert step["recall_at_k"] is not None
        assert driver.count_rows() == N_ROWS

    def test_a_positive_budget_still_stops_and_still_states_itself(self, ctx,
                                                                   dataset):
        """The other direction, so "no cap" is not implemented by removing
        the cap. The DECLARED value is what the record states -- not `inf`,
        which is neither valid JSON nor a value any profile wrote.

        Honest: fewer than 200 rows written, `churn_budget_s == 0.05`.
        Mutant (`effective_budget` recorded instead of `budget_s`): the record
            carries `Infinity`, which `json.loads` accepts and no JSON parser
            outside Python does.
        """
        driver = two_column_driver(dataset, ctx.cache_dir,
                                   cls=_SlowInsertDriver, ef_half=1.0,
                                   ef_search=512)
        run_churn(ctx, driver, dataset, budget_s=0.05)

        rows = written(ctx)
        stopped = rows[1]
        assert stopped["extra"]["reinsert_completed"] is False
        assert stopped["extra"]["rows_reinserted"] < \
            stopped["extra"]["rows_expected"]
        assert stopped["extra"]["churn_budget_s"] == 0.05
        assert stopped["extra"]["repair_path"] == "none"
        assert "recall_at_k" not in stopped
        # No `index_sync_s` on a stopped step: the corpus is missing rows, so
        # waiting for `n_rows` would spend the hook's whole timeout -- 900 s
        # for Valkey -- and a 0.0 recorded here would read as "the index was
        # ready". Pinned, because the alternative is a plausible-looking key.
        assert "index_sync_s" not in stopped["extra"]
        # Since ops version 9 the wait also runs before the baseline and
        # after the deletes, both of which DID happen on a stopped step --
        # what must not run is the post-re-insert wait for `n_rows`.
        waits = [arg for name, arg in driver.churn_hook_calls
                 if name == "wait_index_synced"]
        assert waits == [N_ROWS, N_ROWS - 200], waits
        # Raw text, because a float that reached the file as `Infinity` would
        # be read back by `json.loads` as a float and compare fine.
        assert "Infinity" not in open(ctx.recorder.path).read()

    def test_both_off_spellings_reach_the_workload_as_a_number(self):
        """The orchestrator's end, end to end: profile value -> emitted
        argument -> argparse -> the value `churn.run` receives.

        Honest: `null` and `0` both emit "0", which parses to 0.0 and the
            workload reads as no cap; an absent key emits the default.
        Mutant (`str(ops.get("churn_budget_s", 1800))` restored): `null`
            emits "None" and `parse_args` raises SystemExit(2) -- asserted
            below, because the failure is a dead ops unit and not a wrong
            number.
        """
        from harness.main import parse_args
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        resources = load_resources("normalized")

        def emitted(ops):
            argv = ops_pass.harness_args({"name": "t", "k": 10, "ops": ops},
                                         16, "mariadb", None, "normalized",
                                         resources)
            return argv[argv.index("--churn-budget") + 1]

        base = {"workloads": ["churn"]}
        assert emitted({**base, "churn_budget_s": None}) == "0"
        assert emitted({**base, "churn_budget_s": 0}) == "0"
        assert emitted({**base, "churn_budget_s": 900}) == "900"
        # 7200 since ops measurement version 9 (was 1800).
        assert emitted(base) == "7200"

        args = parse_args([
            "--engine", "e", "--host", "h", "--port", "1", "--dataset", "d",
            "--m", "16", "--run-id", "r", "--output", "/tmp/o",
            "--churn-budget", "0"])
        assert args.churn_budget == 0.0

        with pytest.raises(SystemExit) as caught:
            parse_args([
                "--engine", "e", "--host", "h", "--port", "1",
                "--dataset", "d", "--m", "16", "--run-id", "r",
                "--output", "/tmp/o", "--churn-budget", "None"])
        assert caught.value.code == 2

    def test_the_emitted_default_matches_the_workloads_own(self):
        """Two files now spell 1800: `ops_pass.DEFAULT_CHURN_BUDGET_S` and
        `churn.DEFAULT_CHURN_BUDGET_S`. They cannot import each other -- the
        orchestrator runs on the host and the harness in a container -- so
        they are pinned against each other here instead."""
        from orchestrator import ops_pass
        assert float(ops_pass.DEFAULT_CHURN_BUDGET_S) == \
            cw.DEFAULT_CHURN_BUDGET_S
