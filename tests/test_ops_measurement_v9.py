"""Ops measurement version 9: the items that have no older test class to
extend.

`full-20260825` published six things its records could not back -- see the
VERSION 9 block in `harness/metrics/records.py` for the list. The F2/F7
concurrency changes are tested where the earlier rules were
(`test_skewed_latency_is_not_serialization.py`, `test_workloads.py`,
`test_g1_client_concurrency.py`) and the churn sequence in
`test_t4a_churn.py`; what is here is new machinery: the distance-threshold
recall, the cross-check's refusal to mix rules, the churn writer pool and
indexed rate, the index-size settle and provenance, the sampler's pressure
fields and their join to a window, mongod's WiredTiger cache flag, and the
report cells that read them.

No engine is started and no number here comes from one.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List

import numpy
import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import datasets as ds                                    # noqa: E402
from harness.drivers import mongo as mongo_mod                         # noqa: E402
from harness.drivers.base import ConnectionSpec, RowAttrs              # noqa: E402
from harness.metrics import records, scaling                           # noqa: E402
from harness.workloads import build as build_workload                  # noqa: E402
from harness.workloads import churn as cw                              # noqa: E402
from harness.workloads.context import RunContext                       # noqa: E402
from report import crosscheck, render, server_windows                  # noqa: E402
from report.loaders import knn_recall                                  # noqa: E402
from tests.fake_driver import (N_QUERIES, N_ROWS, FakeDriver,          # noqa: E402
                               index_spec, tiny_dataset)


@pytest.fixture
def ctx(tmp_path):
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="v9", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=10,
            cache_dir=str(tmp_path / "cache"))
    finally:
        recorder.close()


def written(ctx) -> List[Dict[str, Any]]:
    return list(records.read_records(ctx.recorder.path))


# ---------------------------------------------------------------------------
# F8: one recall definition
# ---------------------------------------------------------------------------

def _ann_distance(metric: str, a, b) -> float:
    """`ann_benchmarks/distance.py`'s two metrics, spelled as it spells them
    and computed pair by pair as `runner.run_individual_query` does."""
    a = numpy.asarray(a, dtype=numpy.float64)
    b = numpy.asarray(b, dtype=numpy.float64)
    if metric == ds.ANGULAR:
        return float(1 - numpy.dot(a, b)
                     / ((numpy.sum(a ** 2) ** 0.5) * (numpy.sum(b ** 2) ** 0.5)))
    return float(numpy.sum((a - b) ** 2) ** 0.5)


class TestOneRecallDefinition:
    """`datasets.recall_threshold` is `report/loaders.py:knn_recall`'s rule."""

    @pytest.mark.parametrize("metric", (ds.EUCLIDEAN, ds.ANGULAR))
    def test_it_agrees_with_knn_recall_on_a_thousand_random_cases(self,
                                                                  metric):
        """1,000 cases: random corpora with DUPLICATED rows (exact ties at
        the k-th distance), random answers that mix true neighbours, near
        misses, duplicates, unknown ids and short lists. Each case is scored
        by both, with `knn_recall` fed ann-benchmarks' own pair-wise
        distances -- so this is the rule compared with the rule, not one
        function compared with itself."""
        rng = numpy.random.default_rng(20261101)
        pick = random.Random(9)
        k = 10
        for case in range(1000):
            base = rng.normal(size=(40, 6)).astype("<f4")
            corpus = numpy.concatenate([base, base[:15]])   # 15 exact twins
            query = rng.normal(size=(1, 6)).astype("<f4")
            dist = numpy.array([_ann_distance(metric, query[0], v)
                                for v in corpus])
            order = numpy.argsort(dist, kind="stable")
            truth = order[:k][None, :]
            pool = list(order[:25]) + [-1, 999]
            n = pick.choice((k, k, k, 3, 0))
            got = [int(pick.choice(pool)) for _ in range(n)]
            ours = ds.recall_threshold([got], truth, query, corpus, metric, k)
            run_d = [[dist[g] if 0 <= g < len(corpus) else numpy.inf
                      for g in got]]
            theirs = knn_recall(numpy.sort(dist)[None, :], run_d, k)
            assert ours == pytest.approx(theirs, abs=1e-12), (case, got)

    def test_a_tie_separates_the_two_rules(self):
        """The constructed case: the 10th neighbour has an exact twin under
        another id. Returning the twin is correct by distance and wrong by
        id."""
        corpus = numpy.array([[float(i), 0.0] for i in range(10)]
                             + [[9.0, 0.0]], dtype="<f4")
        query = numpy.zeros((1, 2), dtype="<f4")
        truth = numpy.arange(10)[None, :]
        got = [list(range(9)) + [10]]
        assert ds.recall_threshold(got, truth, query, corpus, ds.EUCLIDEAN,
                                   10) == 1.0
        assert ds.recall_at_k(got, truth, 10) == pytest.approx(0.9)

    def test_a_short_answer_scores_what_it_found_over_k(self):
        corpus = numpy.array([[float(i)] for i in range(20)], dtype="<f4")
        query = numpy.zeros((1, 1), dtype="<f4")
        truth = numpy.arange(10)[None, :]
        got = [[0, 1, 2]]
        ours = ds.recall_threshold(got, truth, query, corpus, ds.EUCLIDEAN,
                                   10)
        theirs = knn_recall(numpy.arange(20, dtype=float)[None, :],
                            [[0.0, 1.0, 2.0]], 10)
        assert ours == theirs == pytest.approx(0.3)
        assert ds.recall_at_k(got, truth, 10) == pytest.approx(0.3)

    def test_a_reinserted_id_is_scored_by_its_original_vector(self):
        corpus = numpy.array([[float(i)] for i in range(12)], dtype="<f4")
        query = numpy.zeros((1, 1), dtype="<f4")
        truth = numpy.arange(10)[None, :]
        # Row 3 was deleted and came back as id 500.
        got = [[0, 1, 2, 500, 4, 5, 6, 7, 8, 9]]
        assert ds.recall_threshold(got, truth, query, corpus, ds.EUCLIDEAN,
                                   10, row_of={500: 3}) == 1.0
        assert ds.recall_threshold(got, truth, query, corpus, ds.EUCLIDEAN,
                                   10) == pytest.approx(0.9)

    def test_the_docstring_no_longer_claims_ann_benchmarks_parity(self):
        doc = ds.recall_at_k.__doc__
        assert "NOT ann-benchmarks' definition" in doc
        assert "Matches ann-benchmarks' definition" not in doc


class TestTheCrossCheckComparesOneRule:
    def _pair(self, kind=None):
        ann = {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
               "resource_pass": "normalized", "m": 16, "ef_search": 40,
               "recall_at_k": 0.95, "qps": 900.0, "storage_engine": "InnoDB"}
        ops = {"phase": "ef_calibration", "engine": "mariadb",
               "dataset": "d", "resource_pass": "normalized", "m": 16,
               "ef_search": 40, "recall_at_k": 0.95, "qps": 700.0,
               "storage_engine": "InnoDB", "extra": {}}
        if kind:
            ops["extra"]["recall_kind"] = kind
        return ann, ops

    def test_a_pre_v9_ops_record_is_read_as_id_overlap(self):
        _ann, ops = self._pair()
        assert crosscheck.recall_kind(ops) == crosscheck.RECALL_ID_OVERLAP
        assert crosscheck.recall_kind(_ann) == \
            crosscheck.RECALL_DISTANCE_THRESHOLD

    def test_mixed_kinds_are_not_compared_and_the_coverage_says_why(self):
        out = crosscheck.consistency_checks(list(self._pair()))
        assert out["matches"] == []
        assert [c["status"] for c in out["coverage"]] == [
            crosscheck.COVERAGE_RECALL_KIND]

    def test_matching_kinds_are(self):
        out = crosscheck.consistency_checks(
            list(self._pair(ds.RECALL_DISTANCE_THRESHOLD)))
        assert len(out["matches"]) == 1

    def test_the_row_builder_refuses_a_mixed_pair_outright(self):
        ann, ops = self._pair()
        with pytest.raises(ValueError, match="refusing to compare"):
            crosscheck._append_crosscheck_row(
                [], {"engine": "mariadb", "build_mode": None}, "InnoDB",
                ann, ops, 1, 1)

    def test_the_two_spellings_agree(self):
        assert crosscheck.RECALL_DISTANCE_THRESHOLD == \
            ds.RECALL_DISTANCE_THRESHOLD
        assert crosscheck.RECALL_ID_OVERLAP == ds.RECALL_ID_OVERLAP


# ---------------------------------------------------------------------------
# F9: churn writers and the indexed rate
# ---------------------------------------------------------------------------

def _loaded(dataset, cls=FakeDriver, **kwargs) -> FakeDriver:
    driver = cls(dim=dataset.dim, metric=dataset.metric, ef_search=512,
                 ef_half=1.0, **kwargs)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, RowAttrs(tag=numpy.arange(N_ROWS) % 100,
                                        ctag=numpy.arange(N_ROWS) % 100))
    return driver


class _LaggingMongot(FakeDriver):
    """mongot five seconds behind every re-insert, without spending them:
    the wait RETURNS 5.0, which is what a real wait that polled for five
    seconds returns."""

    def wait_index_synced(self, expected_rows: int) -> float:
        self.churn_hook_calls.append(("wait_index_synced", int(expected_rows)))
        return 5.0


class TestTheChurnWritesLikeTheLoad:
    def _run(self, ctx, driver, dataset, **kwargs):
        cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
               fractions=(0.10,), max_queries=N_QUERIES,
               indexed_rows=N_ROWS, readers=0, **kwargs)
        return [r for r in written(ctx) if r.get("churn_fraction")][0]

    def test_a_lagging_index_has_an_indexed_rate_below_its_accepted_one(
            self, ctx):
        dataset = tiny_dataset()
        step = self._run(ctx, _loaded(dataset, cls=_LaggingMongot), dataset)
        extra = step["extra"]
        assert extra["index_sync_s"] == 5.0
        assert extra["reinsert_rows_per_s_accepted"] == \
            extra["reinsert_rows_per_s"]
        assert (extra["reinsert_rows_per_s_indexed"]
                < extra["reinsert_rows_per_s_accepted"])
        assert extra["reinsert_rows_per_s_indexed"] == pytest.approx(
            200 / (extra["insert_seconds"] + 5.0), rel=1e-2)
        assert extra["index_sync_after_delete_s"] == 5.0

    def test_a_synchronous_engine_has_equal_rates(self, ctx):
        dataset = tiny_dataset()
        extra = self._run(ctx, _loaded(dataset), dataset)["extra"]
        assert extra["reinsert_rows_per_s_indexed"] == pytest.approx(
            extra["reinsert_rows_per_s_accepted"], rel=0.05)

    def test_the_writer_count_defaults_to_the_ingest_threads(self, ctx):
        """Honest: four writer connections, opened from the factory, and the
        corpus whole afterwards. Mutant (the default left at one): the record
        says 1 against an ingest of 4, the shape `full-20260825` published."""
        dataset = tiny_dataset()
        driver = _loaded(dataset)
        opened = []

        def factory():
            clone = driver.clone()
            opened.append(clone)
            return clone

        extra = self._run(ctx, driver, dataset, driver_factory=factory,
                          ingest_threads=4)["extra"]
        assert extra["reinsert_writers"] == 4
        assert extra["ingest_threads"] == 4
        assert len(opened) == 4
        assert driver.count_rows() == N_ROWS
        assert extra["reinsert_completed"] is True

    def test_without_a_factory_it_is_one_writer_and_says_so(self, ctx):
        dataset = tiny_dataset()
        extra = self._run(ctx, _loaded(dataset), dataset,
                          ingest_threads=8)["extra"]
        assert extra["reinsert_writers"] == 1
        assert extra["ingest_threads"] == 8

    def test_the_report_withholds_the_ratio_when_the_counts_differ(self):
        ingest = [{"phase": "ingest", "engine": "e", "dataset": "d",
                   "resource_pass": "p", "m": 16, "build_mode": "post",
                   "storage_engine": None, "ingest_rows_per_s": 100.0}]
        row = dict(ingest[0], phase="churn",
                   extra={"reinsert_rows_per_s_accepted": 50.0,
                          "reinsert_writers": 1, "ingest_threads": 8})
        assert render._reinsert_ratio_cell(row, ingest) == \
            "withheld (1 vs 8 writers)"
        row["extra"]["reinsert_writers"] = 8
        assert render._reinsert_ratio_cell(row, ingest) == "0.50x"
        # A pre-v9 record cannot show equal counts.
        assert render._reinsert_ratio_cell(
            dict(row, extra={"reinsert_rows_per_s": 50.0}), ingest) == "—"

    def test_the_budget_default_is_two_hours_everywhere(self):
        from harness.main import parse_args
        from orchestrator import ops_pass
        assert cw.DEFAULT_CHURN_BUDGET_S == 7200.0
        assert float(ops_pass.DEFAULT_CHURN_BUDGET_S) == 7200.0
        args = parse_args(["--engine", "valkey", "--dataset", "d",
                           "--m", "16", "--run-id", "r", "--host", "h",
                           "--port", "6379", "--output", "/tmp/o.jsonl"])
        assert args.churn_budget == 7200.0


class TestMongotIsProbedForWhatTheStepChanged:
    @pytest.fixture(autouse=True)
    def _no_pymongo(self, monkeypatch):
        """`Binary` is absent without pymongo; the identity on the payload,
        as `test_t4a_churn.py`'s `mongo_driver` fixture does."""
        monkeypatch.setattr(mongo_mod, "Binary",
                            lambda payload, subtype: payload)
        monkeypatch.setattr(mongo_mod, "SYNC_TIMEOUT_S", 5.0)

    class _Coll:
        """Returns `deleted` for the first `stale` probes, then not."""

        def __init__(self, deleted: int, stale: int):
            self.deleted, self.stale, self.calls = deleted, stale, 0

        def aggregate(self, pipeline):
            self.calls += 1
            return iter([{"_id": self.deleted}] if self.calls <= self.stale
                        else [{"_id": 1}])

    def test_a_deleted_row_is_waited_out(self, monkeypatch):
        monkeypatch.setattr(mongo_mod, "SYNC_POLL_S", 0.0)
        driver = mongo_mod.MongoDriver(ConnectionSpec(host="h", port=1))
        driver._index = index_spec(tiny_dataset())
        driver._coll = self._Coll(deleted=42, stale=3)
        driver.expect_index_state(absent=(42, numpy.ones(16, "<f4")))
        driver.wait_index_synced(1800)
        assert driver._coll.calls == 4
        assert "stopped returning deleted row 42 " in driver.index_sync_note
        # The hint is consumed: the next wait has nothing to look for.
        assert driver._sync_absent is None

    def test_a_failed_probe_is_not_read_as_absence(self, monkeypatch):
        monkeypatch.setattr(mongo_mod, "SYNC_POLL_S", 0.0)
        driver = mongo_mod.MongoDriver(ConnectionSpec(host="h", port=1))
        driver._index = index_spec(tiny_dataset())
        calls = {"n": 0}

        class Flaky:
            def aggregate(self, pipeline):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("mongot rebuilding")
                return iter([])

        driver._coll = Flaky()
        driver.expect_index_state(absent=(42, numpy.ones(16, "<f4")))
        driver.wait_index_synced(1800)
        assert calls["n"] == 3


# ---------------------------------------------------------------------------
# F10: index size, read after it settles, with its format
# ---------------------------------------------------------------------------

class TestTheIndexSizeIsReadAfterItSettles:
    def test_a_shrinking_mongot_directory_settles_to_the_smaller_figure(
            self, monkeypatch):
        """15 GiB right after the build, merging down to 7 over 60 s, then
        still. Honest: 7 GiB after the 90 s reading agrees with the 60 s one,
        and the first reading is kept. Mutant (one immediate read): 15."""
        gib = 1024 ** 3
        now = {"t": 0.0}
        monkeypatch.setattr(mongo_mod, "_clock", lambda: now["t"])
        monkeypatch.setattr(mongo_mod, "_sleep",
                            lambda s: now.__setitem__("t", now["t"] + s))
        driver = mongo_mod.MongoDriver(ConnectionSpec(host="h", port=1))

        def shrinking():
            t = now["t"]
            size = 15 * gib - int((8 * gib) * min(t, 60.0) / 60.0)
            return {"vec": size}

        monkeypatch.setattr(driver, "index_bytes_breakdown", shrinking)
        size, seconds, detail = driver.settle_index_bytes()
        assert size == 7 * gib
        assert seconds == 90.0
        assert detail["settled"] is True
        assert detail["first_bytes"] == 15 * gib

    def test_the_build_record_carries_both_readings_and_the_format(
            self, ctx):
        dataset = tiny_dataset()

        class Settling(FakeDriver):
            def capabilities(self):
                return {**super().capabilities(),
                        "index_vector_format": "int16",
                        "index_includes_vectors": True}

            def settle_index_bytes(self):
                return self.index_bytes() // 2, 30.0, {
                    "method": "directory_stable", "settled": True}

        driver = Settling(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset))
        build = next(r for r in written(ctx) if r["phase"] == "index_build")
        extra = build["extra"]
        assert build["index_bytes"] * 2 == extra["index_bytes_after_build"]
        assert extra["size_read_at"] == "after_settle"
        assert extra["index_size_settle_s"] == 30.0
        assert extra["index_vector_format"] == "int16"
        assert extra["index_includes_vectors"] is True

    def test_the_formats_are_the_verified_ones_and_unknown_elsewhere(self):
        from harness.drivers import mysql_family, postgres, valkey, myvector
        assert mysql_family.MariaDBDriver.index_vector_format == "int16"
        assert mysql_family.AliSQLDriver.index_vector_format == "int16"
        assert postgres.PostgresDriver.index_vector_format == "float32"
        for cls in (mysql_family.PerconaDriver, valkey.ValkeyDriver,
                    myvector.MyVectorDriver, mongo_mod.MongoDriver):
            assert cls.index_vector_format == "unknown", cls
            assert cls.index_includes_vectors is None, cls

    def test_every_ops_record_carries_quantization(self, ctx):
        dataset = tiny_dataset()
        spec = index_spec(dataset)
        spec.quantization = "scalar"
        common = ctx.record_defaults(FakeDriver(dim=dataset.dim), dataset,
                                     spec)
        assert common["quantization"] == "scalar"
        rec = records.Record(**common, phase="concurrency")
        assert rec.to_dict()["quantization"] == "scalar"


# ---------------------------------------------------------------------------
# F4: mongod's WiredTiger cache from the resource pass
# ---------------------------------------------------------------------------

class TestMongodsCacheIsSetByTheFramework:
    class _CPU:
        arch = "x86_64"; hybrid = False; efficiency_cpus = []
        performance_cpus = list(range(40)); logical_cpus = 80
        physical_cores = 40; threads_per_core = 2; model = "Xeon"

    class _Info:
        cpu = None; total_ram_bytes = 201365635072

    @pytest.mark.parametrize("resource_pass", ("normalized", "tuned"))
    def test_the_rendered_args_carry_the_flag(self, resource_pass):
        from orchestrator.config import (GB, engine_resolution, load_engine,
                                         load_resources, resolve_resources,
                                         server_args)
        info = self._Info()
        info.cpu = self._CPU()
        res = load_resources(resource_pass)
        res["memory"]["server_limit_gb"] = 64
        resolved = resolve_resources(res, "mongodb", info)
        assert resolved.wiredtiger_cache_bytes == resolved.buffer_bytes > 0
        args = server_args(load_engine("mongodb"), resource_pass, resolved)
        flag = [a for a in args if a.startswith("--wiredTigerCacheSizeGB=")]
        assert flag == [f"--wiredTigerCacheSizeGB="
                        f"{resolved.wiredtiger_cache_bytes / GB:.2f}"], args
        entry = engine_resolution(load_engine("mongodb"), resource_pass,
                                  resolved)
        assert "wiredtiger_cache_gb" in entry["applied"]
        assert entry["resolved"]["wiredtiger_cache_bytes"] == \
            resolved.wiredtiger_cache_bytes

    def test_no_other_engine_gets_one(self):
        from orchestrator.config import load_resources, resolve_resources
        info = self._Info()
        info.cpu = self._CPU()
        res = load_resources("normalized")
        res["memory"]["server_limit_gb"] = 64
        assert resolve_resources(res, "mariadb",
                                 info).wiredtiger_cache_bytes == 0


# ---------------------------------------------------------------------------
# F11: server CPU and pressure per window
# ---------------------------------------------------------------------------

class TestTheSamplerRecordsPressure:
    TEXT = ("some avg10=1.50 avg60=0.80 avg300=0.10 total=123456\n"
            "full avg10=0.25 avg60=0.10 avg300=0.00 total=7890\n")

    def test_the_fields_parse(self):
        from orchestrator.docker_ctl import pressure_fields
        out = pressure_fields("cpu", self.TEXT)
        assert out == {"cpu_pressure_some_avg10": 1.5,
                       "cpu_pressure_some_total": 123456.0,
                       "cpu_pressure_full_avg10": 0.25,
                       "cpu_pressure_full_total": 7890.0}

    def test_a_missing_file_records_none_not_zero(self):
        from orchestrator.docker_ctl import pressure_fields
        out = pressure_fields("io", None)
        assert set(out) == {"io_pressure_some_avg10", "io_pressure_some_total",
                            "io_pressure_full_avg10", "io_pressure_full_total"}
        assert all(v is None for v in out.values())
        # A file with no `full` line leaves only those keys None.
        half = pressure_fields("cpu", self.TEXT.splitlines()[0])
        assert half["cpu_pressure_full_total"] is None
        assert half["cpu_pressure_some_total"] == 123456.0

    def test_the_sampler_writes_them_from_one_exec(self, monkeypatch):
        from orchestrator import docker_ctl

        class Proc:
            stdout = "@@cpu\n" + self.TEXT + "@@io\n"

        monkeypatch.setattr(docker_ctl, "exec_in", lambda *a, **k: Proc())
        sampler = docker_ctl.MemorySampler("c", "/dev/null")
        out = sampler._cgroup_pressure()
        assert out["cpu_pressure_some_total"] == 123456.0
        assert out["io_pressure_some_total"] is None


class TestTheServerIsJoinedToEachWindow:
    def _samples(self):
        # 2 cores busy, CPU stalled 25% of the time, IO never measured.
        return [{"t": 100.0 + i, "cpu_seconds": 2.0 * i,
                 "cpu_pressure_some_total": 250_000.0 * i,
                 "io_pressure_some_total": None} for i in range(30)]

    def test_the_figures_are_differences_over_the_window(self):
        fig = server_windows.window_figures(self._samples(), 105.0, 125.0,
                                            server_cpus=8)
        assert fig["cpu_cores"] == pytest.approx(2.0)
        assert fig["cpu_util"] == pytest.approx(0.25)
        assert fig["cpu_pressure"] == pytest.approx(0.25)
        assert fig["io_pressure"] is None

    def test_a_pre_v9_record_joins_nothing(self):
        assert server_windows.window_figures(self._samples(), None,
                                             None) is None

    def test_the_concurrency_table_shows_server_beside_client(self):
        record = {"phase": "concurrency", "engine": "mariadb",
                  "dataset": "d", "resource_pass": "tuned", "m": 16,
                  "build_mode": "post", "clients": 4, "qps": 100.0,
                  "extra": {"window_started_at": 105.0,
                            "window_ended_at": 125.0,
                            "client_cpu_util": 0.3}}
        from report.loaders import memory_stem
        windows = server_windows.server_windows(
            [record], {memory_stem(record): self._samples()},
            {"engines_resolved": {"mariadb": {"tuned": {
                "resolved": {"server_cpu_count": 8}}}}})
        table = render._concurrency_table(
            {"concurrency": [record], "server_windows": windows})
        lines = [line for line in table.splitlines() if line.startswith("|")]
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        row = [c.strip() for c in lines[2].strip("|").split("|")]
        assert header.index("Server CPU") == header.index("Client CPU") + 1
        # #77's calibration columns are still the last two.
        assert header[-2:] == ["Calibrated ef", "Calibration recall"]
        assert row[header.index("Server CPU")] == "2.0 cores (25%)"
        assert row[header.index("Server CPU pressure")] == "25%"


# ---------------------------------------------------------------------------
# F2 in the report, and the #75 review's warm-up item
# ---------------------------------------------------------------------------

class TestTheReportReadsTheNewFields:
    def test_the_note_constant_is_the_harness_one(self):
        assert render.TAIL_DOMINATED == scaling.SCALING_TAIL_DOMINATED

    def test_a_tail_dominated_point_is_marked_and_not_refused(self):
        extra = {"outstanding_concurrency": 3.9, "served_concurrency": 1.2,
                 "scaling_status": "ok", "scaling_note": "tail_dominated",
                 "scaling_efficiency": 0.97}
        cell = render._served_cell(extra, 4)
        assert cell.startswith("3.90 of 4 (mean)")
        assert "1.20 median est." in cell
        assert render.TAIL_DOMINATED_MARK in cell
        assert render.SCALING_REFUSED_MARK not in cell
        assert render._efficiency_cell(extra) == "0.97"

    def test_nothing_claims_the_guard_catches_a_one_query_server(self):
        """#78 review finding 3. Under the v9 rule a server answering one
        query at a time looks saturated (outstanding ~N, served ~N,
        efficiency ~1/N) and is published `ok`. The three places a reader
        meets the rule say so, and say what does see it. Checked as
        vocabulary over whitespace-normalised text, so a reflow passes and a
        deletion does not."""
        def plain(text):
            return " ".join(text.replace("*", "").split())

        notes = plain("".join(render._CONCURRENCY_NOTES))
        doc = plain(scaling.scaling_status.__doc__)
        with open(os.path.join(VB_ROOT, "docs", "05-methodology.md")) as fh:
            methodology = plain(fh.read())
        for where, text in (("render", notes), ("scaling", doc),
                            ("docs/05", methodology)):
            lowered = text.lower()
            assert "client-side starvation" in lowered, where
            assert "one query at a time" in lowered, where
            assert "server cpu" in lowered, where

    def test_a_warm_up_that_did_not_settle_is_in_validity(self):
        from report import generate
        rec = {"phase": "recall_qps", "engine": "valkey", "dataset": "d",
               "resource_pass": "tuned", "m": 16, "ef_search": 10,
               "recall_at_k": 0.9, "qps": 100.0,
               "extra": {"index_warmup_queries": 5000,
                         "index_warmup_converged": False,
                         "index_warmup_seconds": 12.0}}
        short = dict(rec, engine="mariadb",
                     extra={"index_warmup_queries": 200,
                            "index_warmup_converged": True})
        fine = dict(rec, engine="pgvector",
                    extra={"index_warmup_queries": 600,
                           "index_warmup_converged": True})
        flags = generate.index_warmup_flags([rec, short, fine])
        assert [f["engine"] for f in flags] == ["mariadb", "valkey"]
        text = render._index_warmup_section({"index_warmup_flags": flags})
        assert "Index warm-ups that did not settle" in text
        assert "**no**" in text
        assert render._index_warmup_section({}) == ""


# ---------------------------------------------------------------------------
# #78 review: the writer pool's merge, through every REAL driver class
# ---------------------------------------------------------------------------

def _real_classes():
    from harness.drivers.postgres import _driver_classes
    return _driver_classes()


def _stubbed(cls, plan=None):
    """A real driver instance with its connection stubbed out: `connect`,
    `close` and `insert_rows` replaced ON THE INSTANCE, so the class -- its
    properties, its attributes, its `__init__` -- is the real one."""
    driver = cls(ConnectionSpec(host="h", port=1))
    driver.connect = lambda: None
    driver.close = lambda: setattr(driver, "_closed", True)

    def insert_rows(ids, vectors, attrs):
        if plan is not None:
            plan(driver, ids)
    driver.insert_rows = insert_rows
    return driver


class TestTheWriterMergeSurvivesEveryRealDriver:
    """#78 review finding 1: `_inherit` wrote through
    `ValkeyDriver.write_batch_used`, a read-only property, and raised
    AttributeError in `parallel_reinsert`'s `finally` -- after the rows were
    written, so the step wrote no record and the remaining writers'
    connections leaked."""

    @pytest.mark.parametrize("name", sorted(_real_classes()))
    def test_parallel_reinsert_completes_and_closes_every_writer(self, name):
        from harness.workloads.churn_writes import parallel_reinsert
        cls = _real_classes()[name]

        def plan(driver, ids):
            # Whatever per-connection state the real class keeps, set the
            # way its own insert path sets it.
            if "_newest_inserted_id" in vars(driver):
                driver._newest_inserted_id = max(int(i) for i in ids)
            if "_write_batch" in vars(driver):
                driver._write_batch = 1000

        main = _stubbed(cls)
        writers = []

        def factory():
            writers.append(_stubbed(cls, plan))
            return writers[-1]

        written, _s = parallel_reinsert(
            main, list(range(100)), numpy.zeros((100, 4), dtype="<f4"),
            numpy.zeros(100), 60.0, 4, factory)
        assert written == 100
        assert len(writers) == 4
        assert all(getattr(w, "_closed", False) for w in writers)
        if "_newest_inserted_id" in vars(main):
            assert main._newest_inserted_id == 99

    def test_no_inherited_name_is_a_property_or_a_slot_on_any_driver(self):
        """The audit, executable: every attribute `INHERITED` merges is a
        plain instance attribute wherever a real driver class has it."""
        from harness.workloads.churn_writes import INHERITED
        for name, cls in _real_classes().items():
            # `abc.ABC` declares an empty `__slots__`; what matters is that
            # every class below it gives its instances a `__dict__`, which is
            # what `_inherit` reads and writes.
            assert "__dict__" in dir(cls(ConnectionSpec(host="h", port=1))), \
                name
            for attr in INHERITED:
                assert not isinstance(getattr(cls, attr, None), property), \
                    (name, attr)

    def test_a_reduced_valkey_batch_survives_the_merge(self):
        """One writer fell back to one row per write; three kept 1,000. The
        main driver still holds the load's 1,000. Honest: the merged batch is
        1 and `_write_shape` says `write_batch_reduced`. Mutant (max): 1,000
        and no caveat -- the re-insert compared as if pipelined."""
        from harness.drivers.valkey import ValkeyDriver
        from harness.workloads.churn_writes import (_write_shape,
                                                    parallel_reinsert)
        main = _stubbed(ValkeyDriver)
        assert main.write_batch_used == main.load_batch_rows
        writers = []

        def plan(driver, ids):
            driver._write_batch = 1 if driver is writers[0] else 1000

        def factory():
            writers.append(_stubbed(ValkeyDriver, plan))
            return writers[-1]

        parallel_reinsert(main, list(range(200)),
                          numpy.zeros((200, 4), dtype="<f4"),
                          numpy.zeros(200), 60.0, 4, factory)
        assert main.write_batch_used == 1
        assert _write_shape(main) == {"write_batch_rows": 1,
                                      "write_batch_reduced": True}

    def test_a_writer_that_fails_to_connect_does_not_leak_the_others(self):
        from harness.drivers.mysql_family import MariaDBDriver
        from harness.workloads.churn_writes import parallel_reinsert
        made = []

        def factory():
            driver = _stubbed(MariaDBDriver)
            made.append(driver)
            if len(made) == 3:
                def refuse():
                    raise ConnectionError("refused")
                driver.connect = refuse
            return driver

        with pytest.raises(ConnectionError):
            parallel_reinsert(_stubbed(MariaDBDriver), list(range(10)),
                              numpy.zeros((10, 4), dtype="<f4"),
                              numpy.zeros(10), 60.0, 4, factory)
        assert len(made) == 3
        assert all(getattr(d, "_closed", False) for d in made)
