"""E29 -- Percona Server's graph was cold when calibration began.

Percona loads its HNSW graph lazily: `vec_runtime_load` at the pin reads only
the entry point and faults every other node in as a search reaches it. The
server team stated (2026-11-04) that `ALTER TABLE ... ADD VECTOR KEY` -- a
`post` build -- evicts the table as part of the ALTER. So
`Innodb_hnsw_memory_used` on the `index_build` record read 0 in 95 of 95
post builds (20k, 60k and 1M rows), against 207,407,472 B after an
incremental build of the same 60k corpus, and the first calibration grid
point after a post build was measured while the graph was being paged in.

WHAT THE FIX IS, by symbol:

* `EngineDriver.warm_graph_after_build` -- False on the base class, True on
  `PerconaDriver` alone. A declaration, so no engine name enters the harness.
* `harness/workloads/graph_warmup.py` -- `plan`, `top_ef`, `warm`: one
  discarded pass over calibration's query slice at the top ef any later phase
  can use, then a re-read of the driver's status gauges.
* `build.run(..., warmup=...)` -- runs it after build timing and the
  after-build readings, before the records are written, and puts the result
  under `extra.graph_warmup` on the `index_build` record.
* `harness/main.py` -- builds the plan and stamps
  `extra.graph_warmup_status` on every record after the build.

WHAT THESE TESTS DO NOT ESTABLISH: that a real Percona Server's
`Innodb_hnsw_memory_used` rises after the pass, or by how much. The gauge
values below are invented by the fakes. The first ops run on the build host
settles that; look at `extra.graph_warmup.server_status` on
`ops-percona-*.jsonl`'s `index_build` records.

PROBE STYLE as in `tests/test_e20_server_status_capture.py`: fakes, no
server, run-time mutants injected with `mock.patch.object` rather than by
editing files on disk.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import EngineDriver                       # noqa: E402
from harness.drivers.mysql_family import PerconaDriver              # noqa: E402
from harness.drivers.postgres import _driver_classes                # noqa: E402
from harness.metrics import records                                 # noqa: E402
from harness.workloads import build as build_workload               # noqa: E402
from harness.workloads import graph_warmup                          # noqa: E402
from harness.workloads.context import RunContext                    # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES,            # noqa: E402
                               index_spec, tiny_dataset)

GAUGE = "Innodb_hnsw_memory_used"

#: Bytes the fake graph "faults in" per warm-up query. Invented, non-round,
#: non-zero: an assertion against it cannot pass on a default.
BYTES_PER_QUERY = 4_099_997

#: The calibration grid the main() tests pass, and an `--ef-search` above
#: it. TOP is therefore not a grid value, so a calibration record at TOP
#: would be a scored warm-up query, and a warm-up at max(GRID) would be the
#: wrong ef.
GRID = (8, 18, 98, 198)
FIXED_EF = 400
TOP = 400

#: A cap below the test split, so "capped as calibration caps them" is
#: observable: 30 of 50.
MAX_QUERIES = 30


class Lazy(FakeDriver):
    """A fake that behaves like Percona after a post build: the gauge reads 0
    until queries fault nodes in, then rises with every query. Records the
    order of what happened, which is where the ordering assertions live."""

    name = "lazy"
    warm_graph_after_build = True
    status_gauges = (GAUGE,)

    def __init__(self, *args, fail_at: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: List[tuple] = []
        self.queries_seen = 0
        self.fail_at = fail_at

    def create_index(self, index):
        self.events.append(("create_index",))
        super().create_index(index)

    def set_ef_search(self, ef_search: int) -> None:
        self.events.append(("ef", int(ef_search)))
        super().set_ef_search(ef_search)

    def query(self, vector, k: int):
        if self.queries_seen == self.fail_at:
            raise RuntimeError("Lost connection to server during query")
        self.queries_seen += 1
        self.events.append(("query", self._ef_search))
        return super().query(vector, k)

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        self.events.append(("status",))
        return {str(n): self.queries_seen * BYTES_PER_QUERY for n in names}


class Plain(Lazy):
    """Everything `Lazy` records, with the declaration every other engine
    has: no warm-up wanted."""

    name = "plain"
    warm_graph_after_build = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build(tmp_path, driver, dataset, build_mode="post", warmup="default"):
    if warmup == "default":
        warmup = graph_warmup.plan(dataset.test, MAX_QUERIES, GRID, FIXED_EF)
    recorder = records.Recorder(str(tmp_path / "ops.jsonl"))
    ctx = RunContext(run_id="e29", resource_pass="normalized",
                     recorder=recorder, engine=driver.name, engine_tag="pin",
                     march="x86-64-v3", k=K, cache_dir=str(tmp_path / "cache"))
    try:
        returned = build_workload.run(
            ctx, driver, dataset, index_spec(dataset, build_mode=build_mode),
            warmup=warmup)
    finally:
        recorder.close()
    written = list(records.read_records(recorder.path))
    assert [r["phase"] for r in written] == [records.PHASE_INGEST,
                                             records.PHASE_BUILD]
    return returned, written[1]


def _driver(cls, dataset, **kwargs):
    """A connected fake that bulk-builds after the load, as Percona's `post`
    default does: `build.run` asks `capabilities()["incremental_index"]`,
    and `build_mode: incremental` overrides it for the other parametrisation.
    """
    kwargs.setdefault("capabilities", {"incremental_index": False})
    d = cls(dim=dataset.dim, metric=dataset.metric, ef_half=0.0, **kwargs)
    d.connect()
    return d


def _main(tmp_path, monkeypatch, driver, dataset, workloads):
    from harness import main as main_mod
    monkeypatch.setattr(main_mod, "load_dataset", lambda d, n: dataset)
    # The same instance every time: `filtered` and calibration use main's own
    # driver and nothing here asks the factory for a second one, so the
    # event log is one log.
    monkeypatch.setattr(main_mod, "get_driver", lambda *a, **kw: driver)
    out = str(tmp_path / "ops.jsonl")
    argv = ["--engine", driver.name, "--host", "h", "--port", "0",
            "--dataset", dataset.name, "--datasets-dir", str(tmp_path),
            "--m", "16", "--run-id", "run-0", "--output", out,
            "--cache-dir", str(tmp_path / "cache"),
            "--ef-search", str(FIXED_EF),
            "--ef-grid", ",".join(str(e) for e in GRID),
            "--max-queries", str(MAX_QUERIES),
            "--selectivities", "0.50", "--filter-columns", "tag",
            "--workloads", workloads]
    assert main_mod.main(argv) == 0
    return list(records.read_records(out))


@pytest.fixture
def dataset():
    return tiny_dataset()


# ---------------------------------------------------------------------------
# 1. Who declares it
# ---------------------------------------------------------------------------

class TestOnlyPerconaWarms:
    def test_the_base_class_declares_no_warm_up(self):
        assert EngineDriver.warm_graph_after_build is False

    def test_percona_is_the_only_driver_in_the_registry_that_warms(self):
        """Derived from `_driver_classes()`, so a driver added later is
        covered without editing this list; the named superset is the
        non-vacuity check for a registry that shrank."""
        declared = {name: cls.warm_graph_after_build
                    for name, cls in sorted(_driver_classes().items())}
        print(f"DECLARED = {declared}")
        assert declared["PerconaDriver"] is True
        assert [n for n, v in declared.items() if v] == ["PerconaDriver"]
        assert set(declared) >= {"MariaDBDriver", "AliSQLDriver",
                                 "PostgresDriver", "MongoDriver",
                                 "ValkeyDriver", "PerconaDriver",
                                 "MyVectorDriver", "VillageSQLDriver"}

    def test_it_is_a_class_attribute_so_both_build_modes_get_it(self):
        """Symmetry within Percona: the post-vs-incremental comparison is the
        one this repairs, so neither side may be treated differently."""
        assert "warm_graph_after_build" in vars(PerconaDriver)
        assert "warm_graph_after_build" not in PerconaDriver.__init__.__code__.co_names


# ---------------------------------------------------------------------------
# 2. Which queries, which ef
# ---------------------------------------------------------------------------

class TestThePlan:
    def test_the_ef_is_the_top_of_the_grid_or_ef_search(self):
        assert graph_warmup.top_ef((10, 1000, 100), 100) == 1000
        assert graph_warmup.top_ef((10, 64), 400) == 400
        # Order-independent, as `ascending_grid` makes calibration.
        assert graph_warmup.top_ef((1000, 10), 40) == 1000

    def test_the_queries_are_calibrations_slice(self, dataset):
        """`calibrate.run` takes `dataset.test[:max_queries]`; the plan must
        be that slice, not a sample of its own."""
        p = graph_warmup.plan(dataset.test, MAX_QUERIES, GRID, FIXED_EF)
        assert len(p.queries) == MAX_QUERIES < len(dataset.test)
        assert (p.queries == dataset.test[:MAX_QUERIES]).all()
        assert p.ef_search == TOP


# ---------------------------------------------------------------------------
# 3. Through build.run: the record
# ---------------------------------------------------------------------------

class TestTheBuildRecord:
    @pytest.mark.parametrize("mode", ["post", "incremental"])
    def test_a_warming_driver_records_both_readings(self, tmp_path, dataset,
                                                    mode):
        driver = _driver(Lazy, dataset)
        returned, built = _build(tmp_path, driver, dataset, build_mode=mode)
        extra = built["extra"]
        warm = extra[graph_warmup.EXTRA_KEY]
        print(f"after build = {extra['server_status']}, warm-up = {warm}")
        # The after-build reading keeps its meaning: taken before any query.
        assert extra["server_status"] == {GAUGE: 0}
        assert warm["server_status"] == {GAUGE: MAX_QUERIES * BYTES_PER_QUERY}
        assert warm["status"] == graph_warmup.STATUS_OK
        assert warm["queries"] == MAX_QUERIES
        assert warm["ef_search"] == TOP
        assert warm["effective_ef"] == TOP
        assert isinstance(warm["wall_s"], float) and warm["wall_s"] >= 0
        assert returned == warm

    def test_the_warm_up_runs_after_the_build_and_its_reading(
            self, tmp_path, dataset):
        driver = _driver(Lazy, dataset)
        _build(tmp_path, driver, dataset)
        ev = driver.events
        first_query = ev.index(("query", TOP))
        assert ev.index(("create_index",)) < first_query
        # The after-build status read precedes every warm-up query, and the
        # re-read follows the last one.
        statuses = [i for i, e in enumerate(ev) if e == ("status",)]
        assert len(statuses) == 2
        assert statuses[0] < first_query
        last_query = max(i for i, e in enumerate(ev) if e[0] == "query")
        assert statuses[1] > last_query
        assert [e for e in ev if e[0] == "query"] == \
            [("query", TOP)] * MAX_QUERIES

    def test_warm_up_time_is_not_build_time(self, tmp_path, dataset):
        """A per-query delay ONLY at the warm-up's ef, so the pass costs
        measurable time and nothing else does. `build_wall_s` for a post
        build is the create_index timer, which must not contain it."""
        driver = _driver(Lazy, dataset)
        real_query = Lazy.query

        def slow_at_top(self, vector, k):
            if self._ef_search == TOP:
                import time
                time.sleep(0.01)
            return real_query(self, vector, k)

        with mock.patch.object(Lazy, "query", slow_at_top):
            _ret, built = _build(tmp_path, driver, dataset)
        warm_s = built["extra"][graph_warmup.EXTRA_KEY]["wall_s"]
        print(f"build_wall_s={built['build_wall_s']} warm-up wall_s={warm_s}")
        assert warm_s >= 0.01 * MAX_QUERIES * 0.9
        assert built["build_wall_s"] < 0.1 < warm_s

    def test_a_driver_that_does_not_declare_it_is_untouched(
            self, tmp_path, dataset):
        """No query during the build, no key -- the other engines' records
        do not move."""
        driver = _driver(Plain, dataset)
        returned, built = _build(tmp_path, driver, dataset)
        assert returned is None
        assert graph_warmup.EXTRA_KEY not in built["extra"]
        assert [e for e in driver.events if e[0] == "query"] == []

    def test_no_plan_means_no_warm_up_even_for_a_declaring_driver(
            self, tmp_path, dataset):
        """`build.run`'s default, which every caller but main() uses."""
        driver = _driver(Lazy, dataset)
        returned, built = _build(tmp_path, driver, dataset, warmup=None)
        assert returned is None
        assert graph_warmup.EXTRA_KEY not in built["extra"]

    def test_the_real_percona_driver_warms_through_build_run(
            self, tmp_path, dataset):
        """The real class, wired to `test_e20`'s recording cursor: the
        declaration reaches `build.run` and the pass sends SELECTs at the
        top ef. The cursor answers no rows, which is fine -- nothing is
        scored."""
        from tests.test_e20_server_status_capture import cursor_of, wired
        driver = wired(PerconaDriver, status={GAUGE: 0})
        _ret, built = _build(tmp_path, driver, dataset)
        warm = built["extra"][graph_warmup.EXTRA_KEY]
        assert warm["status"] == graph_warmup.STATUS_OK
        assert warm["queries"] == MAX_QUERIES
        sent = cursor_of(driver).statements
        assert any(f"= {TOP}" in s and "ef_search" in s.lower() for s in sent)
        assert sum(1 for s in sent if s.startswith("SELECT id FROM")) \
            == MAX_QUERIES


# ---------------------------------------------------------------------------
# 4. A failed warm-up keeps the build record
# ---------------------------------------------------------------------------

class TestAFailedWarmUpKeepsTheBuild:
    def test_the_record_is_written_and_says_how_far_it_got(
            self, tmp_path, dataset):
        driver = _driver(Lazy, dataset, fail_at=4)
        returned, built = _build(tmp_path, driver, dataset)
        warm = built["extra"][graph_warmup.EXTRA_KEY]
        print(f"failed warm-up = {warm}")
        assert built["phase"] == records.PHASE_BUILD
        assert built["index_bytes"] > 0
        assert warm["status"] == graph_warmup.STATUS_FAILED
        assert warm["queries"] == 4
        assert warm["queries_planned"] == MAX_QUERIES
        assert "Lost connection" in warm["error"]
        # The partial reading is still taken.
        assert warm["server_status"] == {GAUGE: 4 * BYTES_PER_QUERY}
        assert built["extra"]["server_status"] == {GAUGE: 0}
        assert returned["status"] == graph_warmup.STATUS_FAILED

    def test_an_empty_query_set_is_a_skip_with_a_reason(self, tmp_path,
                                                         dataset):
        driver = _driver(Lazy, dataset)
        empty = graph_warmup.WarmupPlan(queries=dataset.test[:0],
                                        ef_search=TOP)
        _ret, built = _build(tmp_path, driver, dataset, warmup=empty)
        warm = built["extra"][graph_warmup.EXTRA_KEY]
        assert warm["status"] == graph_warmup.STATUS_SKIPPED
        assert warm["reason"]


# ---------------------------------------------------------------------------
# 5. Through main(): before calibration, unscored, stamped afterwards
# ---------------------------------------------------------------------------

class TestTheHarnessWiring:
    def test_the_warm_up_precedes_calibration_and_is_not_scored(
            self, tmp_path, monkeypatch, dataset):
        driver = _driver(Lazy, dataset)
        rows = _main(tmp_path, monkeypatch, driver, dataset, "build,filtered")
        ev = driver.events
        warm_queries = [i for i, e in enumerate(ev) if e == ("query", TOP)]
        assert len(warm_queries) == MAX_QUERIES
        # Calibration begins at the bottom of the grid, after every warm-up
        # query.
        first_calibration = ev.index(("ef", min(GRID)))
        assert max(warm_queries) < first_calibration
        calib = [r for r in rows if r["phase"] == records.PHASE_CALIBRATION]
        assert calib, "calibration did not run"
        # Unscored: no calibration point at the warm-up's ef, and no record
        # of any phase at it.
        assert TOP not in {r.get("ef_search") for r in rows
                           if r["phase"] != records.PHASE_FILTERED}
        assert {r["phase"] for r in rows} == {
            records.PHASE_INGEST, records.PHASE_BUILD,
            records.PHASE_CALIBRATION, records.PHASE_FILTERED}

    def test_records_after_the_build_say_the_graph_was_warmed(
            self, tmp_path, monkeypatch, dataset):
        driver = _driver(Lazy, dataset)
        rows = _main(tmp_path, monkeypatch, driver, dataset, "build,filtered")
        key = graph_warmup.RUN_EXTRA_KEY
        for r in rows:
            if r["phase"] in (records.PHASE_INGEST, records.PHASE_BUILD):
                assert key not in r["extra"], r["phase"]
            else:
                assert r["extra"][key] == graph_warmup.STATUS_OK, r["phase"]
        built = next(r for r in rows if r["phase"] == records.PHASE_BUILD)
        assert built["extra"][graph_warmup.EXTRA_KEY]["ef_search"] == TOP

    def test_no_engine_but_a_declaring_one_carries_any_of_it(
            self, tmp_path, monkeypatch, dataset):
        driver = _driver(Plain, dataset)
        rows = _main(tmp_path, monkeypatch, driver, dataset, "build,filtered")
        for r in rows:
            assert graph_warmup.RUN_EXTRA_KEY not in r["extra"]
            assert graph_warmup.EXTRA_KEY not in r["extra"]
        # And no query at TOP was ever sent.
        assert ("query", TOP) not in driver.events

    def test_a_build_only_unit_warms_too(self, tmp_path, monkeypatch,
                                         dataset):
        """`percona-bulk-repro` and `percona-bulk-1m` declare
        `workloads: [build]`. They calibrate nothing, and still get the
        after-warm-up memory figure."""
        driver = _driver(Lazy, dataset)
        rows = _main(tmp_path, monkeypatch, driver, dataset, "build")
        assert [r["phase"] for r in rows] == [records.PHASE_INGEST,
                                              records.PHASE_BUILD]
        warm = rows[1]["extra"][graph_warmup.EXTRA_KEY]
        assert warm["status"] == graph_warmup.STATUS_OK
        assert warm["ef_search"] == TOP
        assert warm["server_status"][GAUGE] > 0


# ---------------------------------------------------------------------------
# 6. The measurement version
# ---------------------------------------------------------------------------

class TestVersionBump:
    def test_version_eleven_is_refused_beside_ten(self):
        assert records.OPS_MEASUREMENT_VERSION >= 11
        assert not records.ops_versions_measure_the_same_thing([10, 11])
        source = open(records.__file__).read()
        block = source.split("# VERSION 11 --")[1].split(
            "\nOPS_MEASUREMENT_VERSION = ")[0]
        for token in ("warm_graph_after_build", "graph_warmup",
                      "graph_warmup_status", "vec_runtime_load",
                      "refuse_mixed_measurement_versions",
                      "ANN_MEASUREMENT_VERSION is NOT bumped"):
            assert token in block, token
