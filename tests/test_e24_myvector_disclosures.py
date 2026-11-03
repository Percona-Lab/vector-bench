"""MyVector, round two: what the report and the records must SAY about it.

`tests/test_e23_myvector_engine.py` pins the driver. This file pins the
consequences a reader sees:

* a declined workload is named per engine in the rendered report -- in the
  section it is absent from, and in Validity -- rather than the engine simply
  vanishing from a mixed run's filtered tables;
* the structural asymmetries in MyVector's query path (two statements per
  query, a scalar cosine) are on the records and footnoted under the QPS,
  latency and concurrency sections and in Validity;
* the ann overlay module is exercised with a scripted cursor, not grepped;
* the ops path really passes `--ef-construction` for myvector;
* `main()` really stamps the declines and the facts onto the records.

Rendered sections, not helpers: the review that produced this file found the
helper right and the page wrong.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from typing import Any, Dict, List
from unittest import mock

import numpy
import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness.drivers import myvector as mv  # noqa: E402
from harness.drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from tests.test_e23_myvector_engine import ScriptedCursor, status_text  # noqa: E402

REASON = mv.FILTERED_DECLINED
FACTS = {"round_trips_per_query": 2,
         "distance_impl": mv.MyVectorDriver.DISTANCE_IMPL["Cosine"],
         "index_sync_semantics": mv.INDEX_SYNC_NOTE,
         "memory_note": mv.MEMORY_NOTE}


# ---------------------------------------------------------------------------
# A small mixed run: pgvector measured filtered search, MyVector declined it
# ---------------------------------------------------------------------------

def _base(engine: str, phase: str, **kw) -> Dict[str, Any]:
    record = {"run_id": "e24", "engine": engine, "phase": phase,
              "dataset": "glove-100-angular", "m": 16, "build_mode": "post",
              "resource_pass": "normalized", "k": 10,
              "timestamp": "2026-10-30T00:00:00Z"}
    record.update(kw)
    return record


def _mixed_records() -> List[Dict[str, Any]]:
    mine = {"declined_workloads": {"filtered": REASON}, **FACTS}
    return [
        _base("pgvector", "filtered", clients=1, qps=500.0, recall_at_k=0.9,
              ef_search=40, selectivity=0.1, filter_column="tag",
              extra={"filter_column": "tag"}),
        _base("pgvector", "concurrency", clients=1, qps=900.0,
              recall_at_k=0.95, ef_search=40),
        _base("myvector", "index_build", build_wall_s=1.0, rows=1000,
              extra=dict(mine)),
        _base("myvector", "concurrency", clients=1, qps=800.0,
              recall_at_k=0.95, ef_search=40, extra=dict(mine)),
    ]


def _manifest() -> Dict[str, Any]:
    return {"run_id": "e24", "status": "completed",
            "config": {"profile": {"name": "e24", "ops": {
                "workloads": ["build", "concurrency", "filtered"]}}},
            "engines": {}}


def _render(records, html=False) -> str:
    from report import generate, render
    summary = generate.summarize(records, _manifest())
    fn = render.render_html if html else render.render_markdown
    return fn(_manifest(), summary, records, {}, "E24")


def _section(markdown: str, heading: str) -> str:
    body = markdown.split(heading, 1)[1]
    return body.split("\n---\n", 1)[0]


class TestDeclinedWorkloadIsRendered:
    def test_mixed_run_names_the_decliner_in_the_filtered_section(self):
        page = _render(_mixed_records())
        section = _section(page, "## 8. Filtered (hybrid) search")
        assert "MyVector" in section and "declined: `filtered`" in section
        assert "no filtered vector search" in section

    def test_validity_lists_every_decline(self):
        section = _section(_render(_mixed_records()), "## 2. Validity")
        assert "Workloads a driver declined" in section
        assert "declined: `filtered`" in section

    def test_not_measured_does_not_blame_the_profile_for_a_decline(self):
        records = [r for r in _mixed_records() if r["engine"] == "myvector"]
        section = _section(_render(records), "## 8. Filtered (hybrid) search")
        assert "requested `filtered`" in section
        assert "which does not include" not in section
        assert "declined: `filtered`" in section

    def test_declined_plus_failed_names_only_the_decliner(self):
        """pgvector's ops phase FAILED and MyVector DECLINED. The sentence is
        about MyVector alone: it must not claim every engine declined, and it
        must not say anything about why pgvector is absent -- Validity's
        failed-phase table does that."""
        from report import generate, render
        records = [r for r in _mixed_records() if r["engine"] == "myvector"]
        manifest = _manifest()
        manifest["phases"] = [{"phase": "ops", "engine": "pgvector",
                               "dataset": "glove-100-angular",
                               "status": "failed", "exit_code": 1,
                               "resource_pass": "normalized"}]
        summary = generate.summarize(records, manifest)
        page = render.render_markdown(manifest, summary, records, {}, "E24")
        section = _section(page, "## 8. Filtered (hybrid) search")
        assert "every engine" not in section
        assert "MyVector declined it" in section
        assert "pgvector" not in section.lower()
        assert "PostgreSQL" not in section

    def test_each_section_names_only_the_engine_that_declined_it(self):
        """Two engines declined DIFFERENT workloads. `if True` in place of the
        workload filter would name both in both sections."""
        records = [
            _base("myvector", "index_build", build_wall_s=1.0, rows=1000,
                  extra={"declined_workloads": {"filtered": REASON}}),
            _base("zengine", "index_build", build_wall_s=1.0, rows=1000,
                  extra={"declined_workloads": {"churn": "no churn here"}}),
        ]
        manifest = _manifest()
        manifest["config"]["profile"]["ops"]["workloads"] = [
            "build", "filtered", "churn"]
        from report import generate, render
        summary = generate.summarize(records, manifest)
        page = render.render_markdown(manifest, summary, records, {}, "E24")
        filtered = _section(page, "## 8. Filtered (hybrid) search")
        churn = _section(page, "## 9. Churn")
        assert "MyVector declined it" in filtered
        assert "zengine" not in filtered
        assert "zengine declined it" in churn
        assert "MyVector" not in churn and "myvector" not in churn

    def test_the_html_page_says_it_too(self):
        page = _render(_mixed_records(), html=True)
        assert "declined: <code>filtered</code>" in page \
            or "declined: `filtered`" in page

    def test_a_run_without_declines_renders_no_such_note(self):
        records = [r for r in _mixed_records() if r["engine"] == "pgvector"]
        page = _render(records)
        assert "declined" not in page.lower()
        assert "Structural asymmetries in the query path" not in page


class TestStructuralAsymmetriesAreRendered:
    def test_footnoted_under_qps_concurrency_and_in_validity(self):
        page = _render(_mixed_records())
        for heading in ("## 2. Validity", "## 4. Recall vs throughput",
                        "## 7. Concurrency scaling"):
            section = _section(page, heading)
            assert "Structural asymmetries in the query path" in section, heading
            assert "2 statements per query" in section, heading
            assert "scalar (computeCosineDistance)" in section, heading

    def test_l2_is_not_flagged_as_scalar(self):
        records = _mixed_records()
        for r in records:
            if r["engine"] == "myvector":
                r["extra"]["distance_impl"] = mv.MyVectorDriver.DISTANCE_IMPL["L2"]
        section = _section(_render(records), "## 4. Recall vs throughput")
        assert "2 statements per query" in section
        assert "computes this corpus's distance" not in section

    def test_the_driver_declares_the_facts_per_metric(self):
        d = mv.MyVectorDriver(ConnectionSpec(host="h", port=1))
        cos = d.query_path_facts(IndexSpec(dim=4, m=16, metric="angular"))
        l2 = d.query_path_facts(IndexSpec(dim=4, m=16, metric="euclidean"))
        assert cos == FACTS
        assert l2["distance_impl"].startswith("simd")

    def test_no_other_driver_declares_any(self):
        from harness.drivers.postgres import _driver_classes
        spec = IndexSpec(dim=4, m=16, metric="angular")
        # VillageSQL declares its text binding and preview status, pinned in
        # tests/test_e26_villagesql_engine.py.
        for name, cls in _driver_classes().items():
            if name not in ("MyVectorDriver", "VillageSQLDriver"):
                assert cls(ConnectionSpec(host="h", port=1)) \
                    .query_path_facts(spec) == {}, name

    def test_the_ann_loader_carries_the_facts_into_extra(self):
        """HDF5 attributes arrive as numpy scalars; both spellings convert,
        and an engine that sets neither gets no key at all."""
        from report import disclosures
        got = disclosures.query_path_extra(
            {"round_trips_per_query": numpy.int64(2),
             "distance_impl": numpy.bytes_(FACTS["distance_impl"].encode())})
        assert got == {k: FACTS[k] for k in ("round_trips_per_query",
                                             "distance_impl")}
        assert disclosures.query_path_extra({"M": 16}) == {}

    def test_the_loader_calls_it_on_every_ann_record(self):
        source = open(os.path.join(VB_ROOT, "report", "loaders.py")).read()
        body = source.split("def load_annb_results", 1)[1]
        assert '**_disclosures.query_path_extra(attrs),' in body


# ---------------------------------------------------------------------------
# main() stamps both onto the records it writes
# ---------------------------------------------------------------------------

class TestMainStampsTheRecords:
    def test_declines_and_facts_reach_every_record(self, tmp_path):
        from harness import main as main_mod
        from tests.fake_driver import FakeDriver, tiny_dataset

        class Decliner(FakeDriver):
            declined_workloads = (("filtered", REASON),)

            def query_path_facts(self, index):
                return dict(FACTS)

        def factory(*a, **kw):
            return Decliner(ConnectionSpec(host="h", port=1))

        out = tmp_path / "ops.jsonl"
        argv = ["--engine", "myvector", "--driver", "MyVectorDriver",
                "--host", "h", "--port", "3306", "--dataset", "tiny",
                "--m", "16", "--run-id", "e24", "--output", str(out),
                "--cache-dir", str(tmp_path / "cache"),
                "--workloads", "build,filtered"]
        with mock.patch.object(main_mod, "load_dataset",
                               lambda *a, **kw: tiny_dataset()), \
             mock.patch.object(main_mod, "get_driver", factory):
            assert main_mod.main(argv) == 0
        records = [json.loads(line) for line in out.read_text().splitlines()]
        assert records, "main() wrote nothing"
        assert not [r for r in records if r["phase"] == "filtered"]
        for r in records:
            assert r["extra"]["declined_workloads"] == {"filtered": REASON}
            assert r["extra"]["round_trips_per_query"] == 2
            assert r["extra"]["distance_impl"] == FACTS["distance_impl"]


# ---------------------------------------------------------------------------
# The ops path builds at the ann path's ef_construction
# ---------------------------------------------------------------------------

class TestOpsPassesEfConstruction:
    def _argv(self, ann=None):
        from orchestrator import ops_pass
        profile = {"name": "t", "k": 10, "ops": {}, "ann": ann or {}}
        return ops_pass.harness_args(profile, 16, "myvector", None,
                                     "normalized", {})

    def test_default_is_200_like_the_ann_render(self):
        argv = self._argv()
        assert argv[argv.index("--ef-construction") + 1] == "200"

    def test_the_profile_key_is_honoured(self):
        argv = self._argv({"myvector_ef_construction": 64})
        assert argv[argv.index("--ef-construction") + 1] == "64"


# ---------------------------------------------------------------------------
# The ann overlay module, run against a scripted cursor
# ---------------------------------------------------------------------------

def _load_module():
    from tests.test_e1_percona_design import ALGORITHMS, _load_overlay
    _load_overlay()
    pkg = types.ModuleType("ann_benchmarks.algorithms.myvector")
    pkg.__path__ = [os.path.join(ALGORITHMS, "myvector")]
    sys.modules[pkg.__name__] = pkg
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "ann_benchmarks.algorithms.myvector.module",
            os.path.join(ALGORITHMS, "myvector", "module.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = was
    return mod


@pytest.fixture
def annmod():
    return _load_module()


def _algo(annmod, **cursor):
    """A MyVector instance without __init__, which would start a server."""
    algo = object.__new__(annmod.MyVector)
    algo._m, algo._metric, algo._metric_fn = 16, "L2", "l2"
    algo._ef_construction = 200
    algo.dialect = annmod.dataclasses.replace(annmod.MYVECTOR,
                                              ef_construction=200)
    algo._storage_engine = "InnoDB"
    algo._insert_threads = 1
    algo._data_dir = "/nonexistent"
    algo._status = {}
    algo._ef_search = 40
    algo._plan_verified = None
    algo._cur = ScriptedCursor(**cursor)
    return algo


class TestAnnModule:
    X = numpy.ones((1000, 4), dtype="<f4")

    def test_fit_sizes_builds_and_verifies(self, annmod):
        algo = _algo(annmod, status=status_text(rows=1000))
        algo.fit(self.X)
        sql = algo._cur.statements
        create = next(s for s in sql if s.startswith("CREATE TABLE t1"))
        assert "size=2000" in create and "M=16" in create and "ef=200" in create
        assert "dist=L2" in create
        assert "CALL mysql.myvector_index_build('ann.t1.v', 'id')" in sql
        assert any("myvector_index_status" in s for s in sql)
        assert algo._status["M"] == "16"

    def test_fit_refuses_an_error_row(self, annmod):
        algo = _algo(annmod, build="Error loading config file for index build.")
        with pytest.raises(RuntimeError, match="did not succeed"):
            algo.fit(self.X)

    def test_fit_refuses_a_status_mismatch(self, annmod):
        algo = _algo(annmod, status=status_text(m=32, rows=1000))
        with pytest.raises(RuntimeError, match="'M'"):
            algo.fit(self.X)

    def test_query_binds_then_searches(self, annmod):
        algo = _algo(annmod)
        algo.query(numpy.ones(4, dtype="<f4"), 10)
        bind, search = algo._cur.statements[-2:]
        assert bind == "SET @vb_q = X'" + "0000803f" * 4 + "'"
        assert "MYVECTOR_IS_ANN('ann.t1.v','id',@vb_q,'nn=10,ef_search=40')" \
            in search
        assert search.startswith("SELECT ")

    def test_get_additional_is_hdf5_storable_and_carries_the_facts(self, annmod):
        algo = _algo(annmod)
        algo._status = {"M": "16"}
        algo._index_bytes, algo._build_seconds, algo._ingest_seconds = 0, 1.0, 2.0
        extra = algo.get_additional()
        for key, value in extra.items():
            assert value is None or isinstance(
                value, (str, int, float, bool)), (key, type(value))
        assert extra["round_trips_per_query"] == 2
        assert extra["distance_impl"] == mv.MyVectorDriver.DISTANCE_IMPL["L2"]

    def test_the_two_distance_tables_agree(self, annmod):
        assert annmod.DISTANCE_IMPL == mv.MyVectorDriver.DISTANCE_IMPL


# ---------------------------------------------------------------------------
# LOW-1: the estimator's capability read does not swallow everything
# ---------------------------------------------------------------------------

class TestRunsFilteredSearchFailsLoudly:
    def test_myvector_is_not_charged_and_others_are(self):
        from orchestrator import estimate
        assert estimate._runs_filtered_search("myvector") is False
        assert estimate._runs_filtered_search("pgvector") is True

    def test_an_unknown_engine_is_charged_quietly(self, capsys):
        from orchestrator import estimate
        assert estimate._runs_filtered_search("e24-no-such-engine") is True
        assert capsys.readouterr().err == ""

    def test_an_unreadable_config_is_charged_and_said(self, monkeypatch, capsys):
        from orchestrator import config, estimate

        def broken(name):
            raise PermissionError("denied")
        monkeypatch.setattr(config, "load_engine", broken)
        assert estimate._runs_filtered_search("myvector") is True
        assert "denied" in capsys.readouterr().err

    def test_a_programming_error_is_not_swallowed(self, monkeypatch):
        from orchestrator import config, estimate

        def bug(name):
            raise KeyError("bug")
        monkeypatch.setattr(config, "load_engine", bug)
        with pytest.raises(KeyError):
            estimate._runs_filtered_search("myvector")


# ---------------------------------------------------------------------------
# LOW-2: the server tree is pinned by commit, and the script asserts it
# ---------------------------------------------------------------------------

SERVER_COMMIT = "008e09c2834b98143a8c067d4d225c90953050cf"


class TestServerCommitIsPinned:
    def test_the_config_names_the_server_commit(self):
        import yaml
        cfg = yaml.safe_load(open(os.path.join(
            VB_ROOT, "config", "engines", "myvector.yml")))
        assert cfg["source"]["server_tag"] == "mysql-9.7.2"
        assert cfg["source"]["server_commit"] == SERVER_COMMIT

    def test_prepare_sources_refuses_a_tag_that_moved(self):
        script = open(os.path.join(VB_ROOT, "scripts", "prepare-sources.sh")).read()
        body = script.split("prepare_myvector() {", 1)[1].split("\n}\n", 1)[0]
        assert 'source.server_commit' in body
        assert '[[ "$server_sha" == "$server_pin" ]]' in body



# ---------------------------------------------------------------------------
# MEDIUM-1: index_sync_s is labelled as not an incremental sync
# ---------------------------------------------------------------------------

class TestIndexSyncIsLabelled:
    def _records(self):
        records = _mixed_records()
        records.append(_base("myvector", "churn", churn_fraction=0.1,
                             recall_at_k=0.88, qps=4000.0, ef_search=40,
                             clients=1, rows=1000,
                             extra={**FACTS, "index_sync_s": 2.5,
                                    "maintenance_state": "before",
                                    "repair_path": "index-rebuild",
                                    "declined_workloads": {"filtered": REASON}}))
        return records

    def test_churn_section_and_validity_say_it(self):
        page = _render(self._records())
        for heading in ("## 9. Churn", "## 2. Validity"):
            section = _section(page, heading)
            assert "is not one quantity across engines" in section, heading
            assert "WHOLE graph file is rewritten" in section, heading
            assert "Not comparable" in section, heading

    def test_the_driver_stamps_it(self):
        d = mv.MyVectorDriver(ConnectionSpec(host="h", port=1))
        facts = d.query_path_facts(IndexSpec(dim=4, m=16, metric="euclidean"))
        assert facts["index_sync_semantics"] == mv.INDEX_SYNC_NOTE
        # Its own key: `index_sync_note` is the churn workload's per-record
        # key, which Valkey fills with what its wait observed.
        assert "index_sync_note" not in facts

    def test_each_churn_record_gets_the_note_and_the_split(self, monkeypatch):
        from harness.workloads import churn
        from tests.test_e23_myvector_engine import driver
        monkeypatch.setattr(mv.MyVectorDriver, "_next_second",
                            staticmethod(lambda: None))
        d = driver(status=status_text(rows=1100).replace(
                       "Element Data Size : 88", "Element Data Size : 100"),
                   refresh="SUCCESS: refreshed, rows : 100.")
        d._expected_elements, d._pending = 1000, 100
        d.wait_index_synced(1000)
        extra = churn._sync_shape(d)
        assert extra["index_sync_note"] == mv.INDEX_SYNC_NOTE
        assert extra["index_sync_detail"] == {
            "refreshed_rows": 100, "graph_elements_saved": 1100,
            "graph_bytes_saved_floor": 110000}



# ---------------------------------------------------------------------------
# The memory finding: graph outside the buffer pool, said in the report
# ---------------------------------------------------------------------------

class TestMemoryPlacementIsStated:
    def test_build_section_and_validity_say_it_with_the_floor(self):
        records = _mixed_records()
        for r in records:
            if r["engine"] == "myvector" and r["phase"] == "index_build":
                r["extra"]["myvector_graph_bytes_floor"] = 3 * 1024 ** 3
        page = _render(records)
        for heading in ("## 5. Index build cost", "## 2. Validity"):
            section = _section(page, heading)
            assert "OUTSIDE innodb_buffer_pool_size" in section, heading
            assert "at least 3.00 GiB" in section, heading

    def test_the_floor_is_element_size_times_rows(self):
        status = mv.parse_status(status_text(rows=60000).replace(
            "Element Data Size : 88", "Element Data Size : 3276"))
        assert mv.graph_bytes_floor(status) == 3276 * 60000
        assert mv.graph_bytes_floor({}) is None

    def test_the_build_record_carries_it(self):
        from tests.test_e23_myvector_engine import driver
        d = driver(status=status_text(capacity=2500, rows=1000))
        d.create_index(d._index)
        assert d.capabilities()["myvector_graph_bytes_floor"] == 88 * 1000

    def test_a_small_graph_is_not_printed_as_zero(self):
        from report import disclosures
        records = [_base("myvector", "index_build",
                         extra={"memory_note": "x.",
                                "myvector_graph_bytes_floor": 816_000})]
        text = disclosures.memory_lines(records, str)
        assert "0.00 GiB" not in text and "0.8 MiB" in text
