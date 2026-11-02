"""`ops.build_selection: frontier` -- the ops pass measures the frontier's build.

A reviewer of the published benchmark wrote that comparing engines at one M
is meaningless, because an identically named parameter is a different amount
of work in a different algorithm. The headline is the Pareto frontier over
each engine's swept grid; these tests hold that, under `frontier`, the ops
pass (build cost, concurrency, filtered, churn) measures the build that
frontier is fastest with at the ops calibration target, says so on every
record and in the manifest, and never substitutes another build silently.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from orchestrator import ops_build_selection as sel  # noqa: E402
from orchestrator import ops_pass  # noqa: E402
from orchestrator.manifest import Manifest  # noqa: E402

DATASET = "glove-100-angular"
TARGET = 0.95


def _point(recall: float, qps: float, *, m: int = 16, ef_search: int = 40,
           ef_construction: Optional[int] = None,
           quantization: Optional[str] = None,
           engine: str = "pgvector",
           index_used: Optional[bool] = True) -> Dict[str, Any]:
    parts = ["angular", "M", str(m)]
    if quantization is not None:
        parts += ["quantization", quantization]
    if ef_construction is not None:
        parts += ["efConstruction", str(ef_construction)]
    stem = "_".join(parts + [str(ef_search)])
    return {"phase": "recall_qps", "engine": engine, "dataset": DATASET,
            "resource_pass": "tuned", "recall_at_k": recall, "qps": qps,
            "m": m, "ef_search": ef_search,
            "ef_construction": ef_construction, "quantization": quantization,
            "build_mode": "post", "storage_engine": None,
            "vector_index_used": index_used,
            "extra": {"source_file": f"{DATASET}/10/{engine}/{stem}.hdf5"}}


def _curve(m: int, rows, **kw) -> List[Dict[str, Any]]:
    return [_point(r, q, m=m, ef_search=ef, **kw) for r, q, ef in rows]


# ---------------------------------------------------------------------------
# The rule (report/ops_build_choice.py, run inside the bench image)
# ---------------------------------------------------------------------------

class TestChoice:

    @pytest.fixture(autouse=True)
    def _numpy(self):
        pytest.importorskip("numpy")

    def _choose(self, points, target=TARGET):
        from report.ops_build_choice import choose_ops_build
        return choose_ops_build(points, target)

    def test_picks_the_build_fastest_at_the_target(self):
        # M=16 reaches 0.95 at 500 QPS, M=32 at 800, M=8 never does but is
        # fastest overall -- below the target it does not compete.
        points = (_curve(8, [(0.80, 2000.0, 10), (0.90, 1500.0, 40)])
                  + _curve(16, [(0.90, 900.0, 10), (0.951, 500.0, 40),
                                (0.99, 200.0, 200)])
                  + _curve(32, [(0.93, 1000.0, 10), (0.96, 800.0, 40),
                                (0.995, 300.0, 200)]))
        choice = self._choose(points)
        assert choice["status"] == "selected"
        assert choice["build"]["m"] == 32
        assert choice["point"]["qps"] == 800.0
        assert choice["point"]["recall"] == 0.96
        assert choice["builds_considered"] == 3
        assert choice["builds_reaching_target"] == 2

    def test_an_exact_qps_tie_breaks_as_the_report_breaks_it(self):
        """`builds._fastest_then_smallest`: size, then M, then efC, then the
        signature -- not recall, which the headline does not consult either,
        so the two never name different builds for the same points."""
        from report import builds
        # Equal QPS: the smaller M, even at the lower recall.
        points = (_curve(16, [(0.951, 500.0, 40)])
                  + _curve(32, [(0.97, 500.0, 40)]))
        assert self._choose(points)["build"]["m"] == 16
        best = min([p for p in points if p["recall_at_k"] >= TARGET],
                   key=builds._fastest_then_smallest)
        assert best["m"] == 16
        # Whatever the input order.
        assert self._choose(list(reversed(points)))["build"]["m"] == 16
        # A measured index size outranks M.
        sized = (_curve(16, [(0.96, 500.0, 40)])
                 + _curve(32, [(0.96, 500.0, 40)]))
        for p in sized:
            p["index_bytes"] = 9_000 if p["m"] == 16 else 4_000
        assert self._choose(sized)["build"]["m"] == 32
        assert min(sized, key=builds._fastest_then_smallest)["m"] == 32
        # Then the smaller ef_construction.
        points = (_curve(16, [(0.96, 500.0, 40)], ef_construction=400)
                  + _curve(16, [(0.96, 500.0, 40)], ef_construction=64))
        assert self._choose(points)["build"]["ef_construction"] == 64

    def test_below_target_takes_the_highest_recall_and_says_so(self):
        points = (_curve(16, [(0.71, 260.0, 10), (0.892, 150.0, 400)],
                         quantization="scalar", engine="mongodb")
                  + _curve(16, [(0.90, 140.0, 40), (0.93, 120.0, 120)],
                           quantization="none", engine="mongodb"))
        choice = self._choose(points)
        assert choice["status"] == "below_target"
        assert choice["build"]["quantization"] == "none"
        assert choice["point"]["recall"] == 0.93
        assert choice["builds_reaching_target"] == 0

    def test_a_full_table_scan_never_wins(self):
        scan = _curve(16, [(1.0, 50.0, 40)], index_used=False)
        index = _curve(32, [(0.90, 400.0, 40)])
        choice = self._choose(scan + index)
        assert choice["build"]["m"] == 32
        assert choice["status"] == "below_target"

    def test_nothing_usable_is_none(self):
        assert self._choose([]) is None
        assert self._choose([_point(0.9, None)]) is None

    def test_quantization_is_read_from_the_file_name_as_a_fallback(self):
        points = [_point(0.97, 100.0, quantization="none", engine="mongodb")]
        points[0]["quantization"] = None
        assert self._choose(points)["build"]["quantization"] == "none"

    # -- ranked on the primary storage engine and build mode only ----------

    def _choose_on(self, points, storage=None, mode=None):
        from report.ops_build_choice import choose_ops_build
        return choose_ops_build(points, TARGET, storage_engine=storage,
                                build_mode=mode)

    def test_a_myisam_curve_does_not_choose_the_innodb_build(self):
        innodb = _curve(16, [(0.96, 500.0, 40)], engine="mariadb")
        myisam = _curve(32, [(0.97, 1200.0, 40)], engine="mariadb")
        for p in innodb:
            p["storage_engine"] = "InnoDB"
        for p in myisam:
            p["storage_engine"] = "MyISAM"
        choice = self._choose_on(innodb + myisam, storage="InnoDB")
        assert choice["build"] == {"m": 16, "ef_construction": None,
                                   "quantization": None}
        assert choice["selected_on"] == {"storage_engine": "InnoDB",
                                         "build_mode": None}
        assert choice["point"]["storage_engine"] == "InnoDB"
        assert choice["builds_considered"] == 1
        # Case-insensitive, as MariaDB reports it.
        assert self._choose_on(innodb + myisam,
                               storage="innodb")["build"]["m"] == 16

    def test_an_incremental_curve_does_not_choose_the_post_build(self):
        post = _curve(16, [(0.96, 500.0, 40)], ef_construction=200)
        incremental = _curve(32, [(0.96, 900.0, 40)], ef_construction=200)
        for p in incremental:
            p["build_mode"] = "incremental"
        choice = self._choose_on(post + incremental, mode="post")
        assert choice["build"]["m"] == 16
        assert choice["selected_on"]["build_mode"] == "post"
        assert self._choose_on(post + incremental,
                               mode="incremental")["build"]["m"] == 32

    def test_points_without_the_axis_pass_the_filter(self):
        # pgvector's ann records name no storage engine in the MySQL sense,
        # Percona's name no build mode: neither is filtered away.
        points = _curve(16, [(0.96, 500.0, 40)])
        for p in points:
            p["storage_engine"] = None
            p["build_mode"] = None
        assert self._choose_on(points, storage="InnoDB",
                               mode="incremental")["build"]["m"] == 16

    def test_builds_considered_counts_distinct_m_efc_quantization(self):
        # One M in two storage engines, unfiltered: one build, not two, and
        # the two curves pool into one frontier.
        a = _curve(16, [(0.96, 500.0, 40)])
        b = _curve(16, [(0.97, 600.0, 80)])
        for p in a:
            p["storage_engine"] = "InnoDB"
        for p in b:
            p["storage_engine"] = "MyISAM"
        choice = self._choose_on(a + b)
        assert choice["builds_considered"] == 1
        assert set(choice["build"]) == {"m", "ef_construction",
                                        "quantization"}

    def test_nothing_on_the_primary_curve_is_none(self):
        points = (_curve(16, [(0.96, 500.0, 40)], engine="mariadb")
                  + _curve(32, [(0.97, 400.0, 40)], engine="mariadb"))
        points[0]["storage_engine"] = "MyISAM"
        points[1]["storage_engine"] = "Aria"
        assert self._choose_on(points, storage="InnoDB") is None

    def test_an_axis_the_grid_did_not_sweep_is_not_filtered(self):
        # One value only: the stamp, not a sweep -- MyISAM-only grid points
        # still compete, and the record says no filter was applied.
        points = _curve(16, [(0.96, 500.0, 40)], engine="mariadb")
        points[0]["storage_engine"] = "MyISAM"
        choice = self._choose_on(points, storage="InnoDB", mode="post")
        assert choice["build"]["m"] == 16
        assert choice["selected_on"] == {"storage_engine": None,
                                         "build_mode": None}


class TestContainerEntryPoint:
    """`main` reads only the files the ann phase named, and prints one line."""

    def test_only_the_named_files_are_read(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        from report import loaders, ops_build_choice
        tree = tmp_path / "annb" / DATASET / "10" / "pgvector"
        tree.mkdir(parents=True)
        for name in ("named.hdf5", "stale.hdf5"):
            (tree / name).write_text("x")
        seen: List[str] = []

        def fake_load(root, datasets_dir, run_id):
            for base, _dirs, files in os.walk(root):
                seen.extend(os.path.relpath(os.path.join(base, f), root)
                            for f in files)
            return _curve(32, [(0.96, 800.0, 40)])

        monkeypatch.setattr(loaders, "load_annb_results", fake_load)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ops_build_choice.main([
                "--annb-dir", str(tmp_path / "annb"), "--datasets-dir", "d",
                "--engine", "pgvector", "--dataset", DATASET,
                "--target-recall", "0.95", "--files", json.dumps(
                    [f"{DATASET}/10/pgvector/named.hdf5",
                     "../escape.hdf5"])])
        assert rc == 0
        assert seen == [f"{DATASET}/10/pgvector/named.hdf5"]
        line = [l for l in out.getvalue().splitlines()
                if l.startswith(sel.RESULT_MARKER)][-1]
        payload = json.loads(line[len(sel.RESULT_MARKER):])
        assert payload["choice"]["build"]["m"] == 32
        assert payload["points"] == 1


class TestTheTwoSidesAgree:

    def test_markers_and_statuses_are_spelled_alike(self):
        source = open(os.path.join(VB_ROOT, "report",
                                   "ops_build_choice.py")).read()
        assert f'RESULT_MARKER = "{sel.RESULT_MARKER}"' in source
        assert f'STATUS_SELECTED = "{sel.STATUS_SELECTED}"' in source
        assert f'STATUS_BELOW_TARGET = "{sel.STATUS_BELOW_TARGET}"' in source

    def test_report_side_does_not_import_the_orchestrator(self):
        source = open(os.path.join(VB_ROOT, "report",
                                   "ops_build_choice.py")).read()
        assert "from orchestrator" not in source
        assert "import orchestrator" not in source

    def test_target_is_read_from_one_place(self):
        """The selection chases the recall the calibration then targets."""
        profile = {"ops": {"target_recall": 0.9}}
        assert ops_pass.ops_target_recall(profile) == 0.9
        assert ops_pass.ops_target_recall({}) == ops_pass.DEFAULT_TARGET_RECALL
        args = ops_pass.harness_args(profile, 16, "mariadb", None, "tuned", {})
        assert args[args.index("--target-recall") + 1] == "0.9"
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ops_build_selection.py")).read()
        assert "DEFAULT_TARGET_RECALL" not in source
        assert "0.95" not in source.split('"""', 2)[2].replace(
            "frontier@0.95", "")


# ---------------------------------------------------------------------------
# The host side: selection, fallback, persistence, resume
# ---------------------------------------------------------------------------

def _manifest(tmp_path, *, ann: Optional[Dict[str, Any]] = None,
              engine: str = "pgvector") -> Manifest:
    manifest = Manifest(str(tmp_path / "run"), "sel-1")
    manifest.set_engine(engine, str(tmp_path / "sources"), "r", "b",
                        {"runtime": "x", "bench": "y"},
                        presentation={"reference_builds": {
                            "tuned": {"m": 16, "ef_construction": 200}}})
    if ann is not None:
        manifest.add_phase("ann", engine, DATASET, ann.get("status",
                                                           "completed"),
                           "t0", "t1", {"resource_pass": "tuned",
                                        "files": ann["files"]})
    return manifest


FILES = {"written": [{"path": f"{DATASET}/10/pgvector/a.hdf5"}],
         "skipped": [{"path": f"{DATASET}/10/pgvector/b.hdf5"}]}
PROFILE = {"ops": {"build_selection": "frontier"},
           "ann": {"pgvector_ef_construction": 200}}


def _payload(m=32, efc=400, status="selected", recall=0.96, qps=800.0):
    return {"points": 6, "files": 2, "choice": {
        "status": status, "target_recall": TARGET, "signature": "sig",
        "builds_considered": 3,
        "builds_reaching_target": 2 if status == "selected" else 0,
        "build": {"m": m, "ef_construction": efc, "quantization": None,
                  "storage_engine": None, "build_mode": "post"},
        "point": {"recall": recall, "qps": qps, "ef_search": 40,
                  "source_file": "x.hdf5"}}}


class _Chooser:
    def __init__(self, payload):
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return self.payload


def _ensure(manifest, chooser, profile=PROFILE, engine="pgvector",
            resources=None, resource_pass="tuned", ops_units_done=False):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        selection = sel.ensure_selection(
            manifest, engine=engine, dataset=DATASET,
            resource_pass=resource_pass, profile=profile,
            resources=resources or {}, chooser=chooser,
            ops_units_done=ops_units_done)
    return selection, out.getvalue(), err.getvalue()


@pytest.fixture
def any_grid(monkeypatch):
    """The grid check is exercised on its own, against a real render."""
    monkeypatch.setattr(sel, "grid_problem", lambda *a, **kw: None)


@pytest.mark.usefixtures("any_grid")
class TestHostSelection:

    def test_mode_defaults_to_reference_and_refuses_typos(self):
        assert sel.build_selection_mode({}) == "reference"
        assert sel.build_selection_mode(PROFILE) == "frontier"
        with pytest.raises(ValueError):
            sel.build_selection_mode({"ops": {"build_selection": "pareto"}})

    def test_selected_build_is_recorded_with_its_source(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        chooser = _Chooser(_payload())
        selection, out, _ = _ensure(manifest, chooser)
        assert chooser.calls == [{"files": sorted(
            [f"{DATASET}/10/pgvector/a.hdf5", f"{DATASET}/10/pgvector/b.hdf5"]),
            "target_recall": TARGET}]
        assert selection["status"] == "selected"
        assert selection["source"] == "frontier@0.95"
        assert selection["reference_build"] == {"m": 32,
                                                "ef_construction": 400}
        assert "frontier@0.95" in out
        saved = json.load(open(manifest.path))
        assert saved["ops_build_selection"][f"tuned/pgvector/{DATASET}"][
            "build"]["m"] == 32
        presentation = saved["engines"]["pgvector"]["presentation"]
        assert presentation["reference_builds_by_dataset"]["tuned"][
            DATASET] == {"build": {"m": 32, "ef_construction": 400},
                         "source": "frontier@0.95", "status": "selected"}
        # The declared per-pass build is left as it was.
        assert presentation["reference_builds"]["tuned"] == {
            "m": 16, "ef_construction": 200}

    def test_below_target_is_used_and_warned(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        selection, _, err = _ensure(manifest, _Chooser(
            _payload(status="below_target", recall=0.93)))
        assert selection["status"] == "below_target"
        assert selection["source"] == "below_target@0.95"
        assert selection["build"]["m"] == 32
        assert "below_target" in err and "0.93" in err
        assert any("below_target" in w for w in manifest.data["warnings"])

    @pytest.mark.parametrize("ann", [
        None,                                   # ann did not run here
        {"files": FILES, "status": "failed"},   # ann failed
        {"files": {"written": [], "skipped": []}},  # wrote nothing
    ])
    def test_no_ann_falls_back_to_the_reference_build_loudly(self, tmp_path,
                                                             ann):
        manifest = _manifest(tmp_path, ann=ann)
        chooser = _Chooser(_payload())
        selection, _, err = _ensure(manifest, chooser)
        assert chooser.calls == []
        assert selection["status"] == "fallback_reference"
        assert selection["source"] == "fallback_reference"
        assert selection["build"] == {"m": 16, "ef_construction": 200}
        assert "fallback_reference" in err and "no completed ann" in err
        assert manifest.data["warnings"]

    @pytest.mark.parametrize("payload", [None, {"points": 0, "files": 2,
                                                "choice": None}])
    def test_unreadable_or_empty_results_fall_back(self, tmp_path, payload):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        selection, _, err = _ensure(manifest, _Chooser(payload))
        assert selection["status"] == "fallback_reference"
        assert "fallback_reference" in err

    def test_resume_reuses_the_stored_choice(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        first, _, _ = _ensure(manifest, _Chooser(_payload(m=32)))
        # A new process on the same run directory: the manifest is reloaded,
        # and a re-read of the ann results would now pick M=24.
        resumed = Manifest(str(tmp_path / "run"), "sel-1")
        again = _Chooser(_payload(m=24))
        second, out, _ = _ensure(resumed, again)
        assert again.calls == []
        assert second["build"]["m"] == 32
        assert second == first
        assert "reusing" in out

    def test_presentation_entries_survive_a_rewritten_engine_entry(
            self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        _ensure(manifest, _Chooser(_payload()))
        # What cmd_run does on the next pass or a resume.
        manifest.set_engine(
            "pgvector", str(tmp_path / "sources"), "r", "b",
            {"runtime": "x", "bench": "y"},
            presentation={"reference_builds": {},
                          **sel.presentation_entries(manifest.data,
                                                     "pgvector")})
        assert manifest.data["engines"]["pgvector"]["presentation"][
            "reference_builds_by_dataset"]["tuned"][DATASET]["build"][
                "m"] == 32
        assert sel.presentation_entries(manifest.data, "valkey") == {}

    def test_latest_completed_ann_attempt_is_the_one_used(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": {
            "written": [{"path": "old.hdf5"}], "skipped": []}})
        manifest.add_phase("ann", "pgvector", DATASET, "completed", "t2",
                           "t3", {"resource_pass": "tuned", "files": FILES})
        manifest.add_phase("ann", "pgvector", DATASET, "failed", "t4", "t5",
                           {"resource_pass": "tuned", "files": {
                               "written": [{"path": "broken.hdf5"}]}})
        assert sel.ann_phase_files(manifest.data, "pgvector", DATASET,
                                   "tuned") == [
            f"{DATASET}/10/pgvector/a.hdf5", f"{DATASET}/10/pgvector/b.hdf5"]
        assert sel.ann_phase_files(manifest.data, "pgvector", DATASET,
                                   "normalized") is None


class TestTheOpsPassBuildsTheChoice:

    def _selection(self, **build):
        return {"mode": "frontier", "status": "selected",
                "source": "frontier@0.95", "target_recall": TARGET,
                "build": build, "ann_point": {"qps": 1.0}, "reason": None}

    def test_m_values_follow_the_choice(self):
        assert sel.ops_m_values(None, [8, 16]) == [8, 16]
        assert sel.ops_m_values(self._selection(m=32), [16]) == [32]
        # Percona Search's chosen build names no M: the declared values stay.
        assert sel.ops_m_values(self._selection(quantization="none"),
                                [16]) == [16]

    def test_reference_mode_command_line_is_unchanged(self):
        profile = {"ops": {}, "ann": {}}
        plain = ops_pass.harness_args(profile, 16, "pgvector", None, "tuned",
                                      {})
        same = ops_pass.harness_args(profile, 16, "pgvector", None, "tuned",
                                     {}, build=None, build_selection=None)
        assert plain == same
        assert "--ops-build-selection" not in plain
        assert plain[plain.index("--ef-construction") + 1] == "200"

    def test_chosen_ef_construction_and_quantization_reach_the_harness(self):
        selection = self._selection(m=32, ef_construction=400)
        args = ops_pass.harness_args(
            {}, 32, "pgvector", None, "tuned", {},
            build=sel.ops_build(selection),
            build_selection=sel.disclosure(selection))
        assert args[args.index("--ef-construction") + 1] == "400"
        stamped = json.loads(args[args.index("--ops-build-selection") + 1])
        assert stamped["status"] == "selected"
        assert stamped["source"] == "frontier@0.95"
        assert stamped["build"]["ef_construction"] == 400
        # Valkey: the ops pass passes no efC in reference mode; it must now.
        args = ops_pass.harness_args(
            {}, 16, "valkey", None, "tuned", {},
            build=sel.ops_build(self._selection(m=16, ef_construction=64)))
        assert args[args.index("--ef-construction") + 1] == "64"
        # Percona Search: the frontier's quantization replaces the pass's.
        resources = {"extras": {"mongodb_quantization": ["scalar", "none"]}}
        args = ops_pass.harness_args(
            {}, 16, "mongodb", None, "tuned", resources,
            build=sel.ops_build(self._selection(quantization="none")))
        assert args[args.index("--quantization") + 1] == "none"
        args = ops_pass.harness_args({}, 16, "mongodb", None, "tuned",
                                     resources)
        assert args[args.index("--quantization") + 1] == "scalar"

    def test_every_ops_record_carries_the_selection(self, tmp_path):
        from harness.main import parse_args
        from harness.metrics.records import (PHASE_BUILD, Record, Recorder)
        selection = sel.disclosure(self._selection(m=32))
        base = ["--engine", "pgvector", "--host", "h", "--port", "1",
                "--dataset", DATASET, "--m", "32", "--run-id", "r"]
        args = parse_args(base + ["--output", str(tmp_path / "o.jsonl"),
                                  "--ops-build-selection",
                                  json.dumps(selection)])
        assert args.ops_build_selection == selection
        with pytest.raises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parse_args(base + ["--output", "o",
                                   "--ops-build-selection", "[1]"])
        recorder = Recorder(str(tmp_path / "o.jsonl"))
        recorder.declare_run_extra(ops_build_selection=selection)
        recorder.write(Record(run_id="r", phase=PHASE_BUILD,
                              engine="pgvector", dataset=DATASET,
                              timestamp="t"))
        recorder.close()
        line = json.loads(open(tmp_path / "o.jsonl").read())
        assert line["extra"]["ops_build_selection"]["source"] == \
            "frontier@0.95"

    def test_harness_declares_it_before_any_record(self):
        source = open(os.path.join(VB_ROOT, "harness", "main.py")).read()
        declared = source.index(
            "declare_run_extra(ops_build_selection=args.ops_build_selection)")
        assert declared < source.index("driver = driver_factory()")


# ---------------------------------------------------------------------------
# The report labels the chosen build
# ---------------------------------------------------------------------------

class TestReportLabelsTheChosenBuild:

    def test_per_dataset_choice_wins_and_carries_its_source(self):
        from report import builds
        manifest = {"engines": {"pgvector": {"presentation": {
            "reference_builds": {"tuned": {"m": 16, "ef_construction": 200}},
            "reference_builds_by_dataset": {"tuned": {DATASET: {
                "build": {"m": 32, "ef_construction": 400},
                "source": "frontier@0.95", "status": "selected"}}}}}}}
        points = (_curve(16, [(0.96, 500.0, 40)], ef_construction=200)
                  + _curve(32, [(0.96, 800.0, 40)], ef_construction=400))
        build, source = builds.reference_build(points, [], manifest,
                                               "pgvector", DATASET, "tuned")
        assert build == {"m": 32, "ef_construction": 400}
        assert source == "manifest: frontier@0.95"
        # Another dataset in the same pass keeps the declared build.
        build, source = builds.reference_build(points, [], manifest,
                                               "pgvector", "sift-128",
                                               "tuned")
        assert build == {"m": 16, "ef_construction": 200}
        assert source == "manifest"

    def test_the_page_says_how_each_build_was_chosen(self):
        from report import render
        notes = render._chosen_build_notes({
            "pgvector": {"reference_source": "manifest: frontier@0.95",
                         "reference_build_label": "M=32 efC=400"},
            "mongodb": {"reference_source": "manifest: below_target@0.95",
                        "reference_build_label": "quantization=none"},
            "valkey": {"reference_source": "manifest: fallback_reference",
                       "reference_build_label": "M=16 efC=200"},
            "mariadb": {"reference_source": "manifest",
                        "reference_build_label": "M=16"}})
        assert "`M=32 efC=400` -- chosen from this run's frontier" in notes
        assert "fastest at recall 0.95" in notes
        assert "NO swept build reached recall 0.95" in notes
        assert "ann phase left no frontier" in notes
        # A declared build is not described as chosen.
        assert f"**{render._label('mariadb')}**" not in notes
        assert notes.count("\n- ") == 3


# ---------------------------------------------------------------------------
# Pre-run refusal, profiles, estimate, version
# ---------------------------------------------------------------------------

class TestRefusal:

    def _problem(self, profile):
        from orchestrator.config import load_resources, merge_resource_overrides
        from orchestrator.reference_build import reference_build_problem
        resources = merge_resource_overrides(load_resources("normalized"),
                                             profile)
        return reference_build_problem("mariadb", profile, resources,
                                       "normalized")

    def _profile(self, **ops):
        return {"name": "t", "datasets": [DATASET], "k": 10,
                "ann": {"enabled": True, "m_values": [16, 32],
                        "ef_search": [10, 40]},
                "ops": {"enabled": True, "m_values": [8], **ops}}

    def test_m_values_mismatch_is_refused_only_in_reference_mode(self):
        assert "does not build" in self._problem(self._profile())
        assert self._problem(self._profile(build_selection="frontier")) is None

    def test_frontier_without_an_ann_phase_is_refused(self):
        profile = self._profile(build_selection="frontier")
        profile["ann"]["enabled"] = False
        problem = self._problem(profile)
        assert problem and "disables the ann phase" in problem

    def test_unknown_mode_is_refused(self):
        problem = self._problem(self._profile(build_selection="best"))
        assert problem and "build_selection" in problem

    def test_frontier_still_refuses_a_reference_build_off_the_grid(self):
        profile = self._profile(build_selection="frontier")
        profile["ann"]["m_values"] = [32]
        assert "not in the normalized ann grid" in self._problem(profile)


class TestProfiles:

    def _load(self, name):
        return yaml.safe_load(open(os.path.join(
            VB_ROOT, "config", "profiles", f"{name}.yml")))

    def test_tuned_complete_measures_the_frontier_build(self):
        assert sel.build_selection_mode(self._load("tuned-complete")) == \
            "frontier"

    def test_every_other_profile_keeps_reference(self):
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "config",
                                                   "profiles"))):
            if name == "tuned-complete.yml":
                continue
            profile = self._load(name[:-len(".yml")])
            assert sel.build_selection_mode(profile) == "reference", name


class TestEstimate:

    def test_reference_mode_has_no_bound(self):
        from orchestrator.estimate import estimate_load_hours
        profile = {"ann": {"m_values": [16]}, "ops": {"m_values": [16, 32]}}
        est = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                  ["tuned"], ["ann", "ops"])
        assert est["ops_build_bound"] is None
        assert est["ops_m_values"] == 2

    def test_frontier_is_charged_one_build_at_the_largest_m(self):
        from orchestrator.estimate import estimate_load_hours
        base = {"ann": {"m_values": [8, 16, 48]}, "ops": {"m_values": [16]}}
        frontier = {**base, "ops": {"m_values": [16, 32],
                                    "build_selection": "frontier"}}
        ref = estimate_load_hours(base, ["mariadb"], ["glove-100-angular"],
                                  ["normalized"], ["ops"])
        est = estimate_load_hours(frontier, ["mariadb"],
                                  ["glove-100-angular"], ["normalized"], ["ops"])
        from orchestrator.estimate_grid import m_cost_factor
        factor = m_cost_factor("mariadb", 48)          # ~M^1.1, as #82 prices
        assert est["ops_build_bound"]["m"] == 48
        assert est["ops_build_bound"]["engines"]["mariadb"] == {
            "m": 48, "ingest_factor": round(factor, 3)}
        assert est["ops_m_values"] == 1
        assert est["ingest_hours"] == pytest.approx(
            ref["ingest_hours"] * factor, rel=0.01)

    def test_the_bound_reads_the_rendered_grid_overlay_included(self):
        """`m_values_by_engine` from a select-builds overlay or the tuned
        extras is what the run renders, so it is what the bound charges."""
        from orchestrator.estimate import frontier_ops_bound
        profile = {"ann": {"m_values": [16]},
                   "ops": {"build_selection": "frontier"},
                   "resources": {"extras": {
                       "m_values_by_engine": {"mariadb": [8, 32]}}}}
        bound = frontier_ops_bound(profile, ["mariadb", "alisql"], ["tuned"])
        assert bound["engines"]["mariadb"]["m"] == 32
        assert bound["engines"]["alisql"]["m"] == 16
        assert bound["engines"]["alisql"]["ingest_factor"] == 1.0

    def test_the_estimate_says_the_build_is_unknown(self):
        from orchestrator.estimate import print_load_estimate
        profile = {"ann": {"m_values": [16, 32]},
                   "ops": {"build_selection": "frontier"}}
        out = io.StringIO()
        with redirect_stdout(out):
            print_load_estimate(profile, ["mariadb"],
                                ["dbpedia-openai-1000k-angular"], ["normalized"],
                                ["ann", "ops"])
        assert "unknown until each ann phase" in out.getvalue()
        assert "M=32" in out.getvalue()


class TestVersionBump:

    def test_version_ten_is_refused_beside_nine(self):
        from harness.metrics import records
        assert records.OPS_MEASUREMENT_VERSION >= 10
        assert not records.ops_versions_measure_the_same_thing([9, 10])
        assert 10 not in records.OPS_MEASUREMENT_EQUIVALENT_VERSIONS
        source = open(records.__file__).read()
        block = source.split("# VERSION 10 --")[1].split(
            "\nOPS_MEASUREMENT_VERSION = ")[0]
        for token in ("ops.build_selection", "ops_build_selection",
                      "fallback_reference", "below_target",
                      "refuse_mixed_measurement_versions"):
            assert token in block, token

    def test_report_refuses_a_directory_holding_nine_and_ten(self):
        from report.measurement_versions import \
            refuse_mixed_measurement_versions
        recs = [{"phase": "concurrency",
                 "extra": {"ops_measurement_version": v}} for v in (9, 10)]
        from report.measurement_versions import MixedMeasurementVersions
        with pytest.raises(MixedMeasurementVersions):
            refuse_mixed_measurement_versions(recs, source="t")


class TestRunUnitWiring:
    """`cmd_run` -> `_run_unit` builds the chosen build, and only it."""

    def _run(self, monkeypatch, tmp_path, profile, selection):
        import argparse
        from orchestrator import cli
        captured: List[List[str]] = []

        class FakeRun:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_harness(self, args, output, memory_timeseries=None):
                captured.append(list(args))
                return 0

        monkeypatch.setattr(cli.ops_pass, "OpsRun", FakeRun)
        paths = {"run_dir": str(tmp_path),
                 "checkpoints": str(tmp_path / "cp")}
        rc = cli._run_unit("ops", "pgvector", DATASET, profile,
                           cli.load_engine("pgvector"), {}, None, "tuned",
                           paths, "r", argparse.Namespace(),
                           ops_selection=selection)
        assert rc == 0
        return captured

    def test_frontier_selection_reaches_every_unit(self, monkeypatch,
                                                   tmp_path):
        profile = {"ops": {"m_values": [8, 16], "build_selection": "frontier"},
                   "ann": {"pgvector_build_modes": ["post"]}}
        selection = {"mode": "frontier", "status": "selected",
                     "source": "frontier@0.95", "target_recall": TARGET,
                     "build": {"m": 32, "ef_construction": 400},
                     "ann_point": None, "reason": None}
        calls = self._run(monkeypatch, tmp_path, profile, selection)
        assert len(calls) == 1
        args = calls[0]
        assert args[args.index("--m") + 1] == "32"
        assert args[args.index("--ef-construction") + 1] == "400"
        assert "--ops-build-selection" in args

    def test_reference_mode_builds_m_values_as_before(self, monkeypatch,
                                                      tmp_path):
        profile = {"ops": {"m_values": [8, 16]},
                   "ann": {"pgvector_build_modes": ["post"]}}
        calls = self._run(monkeypatch, tmp_path, profile, None)
        assert [c[c.index("--m") + 1] for c in calls] == ["8", "16"]
        assert all("--ops-build-selection" not in c for c in calls)

    def test_no_selection_is_made_in_reference_mode(self, tmp_path):
        from orchestrator import cli
        manifest = _manifest(tmp_path, ann={"files": FILES})
        assert cli._ops_selection(manifest, "ops", "pgvector", DATASET,
                                  {"ops": {}}, {}, {}, None, "tuned",
                                  {}) is None
        assert cli._ops_selection(manifest, "ann", "pgvector", DATASET,
                                  PROFILE, {}, {}, None, "tuned", {}) is None
        assert "ops_build_selection" not in manifest.data

    def test_frontier_passes_the_primary_curve_and_measured_units(
            self, monkeypatch, tmp_path):
        from orchestrator import cli
        seen: Dict[str, Any] = {}

        def fake_default_chooser(**kw):
            seen["chooser"] = kw
            return lambda **_: None

        def fake_ensure(manifest, **kw):
            seen["ensure"] = kw
            return {"status": "selected"}

        monkeypatch.setattr(sel, "default_chooser", fake_default_chooser)
        monkeypatch.setattr(sel, "ensure_selection", fake_ensure)
        cp = tmp_path / "cp.json"
        cp.write_text(json.dumps([f"tuned/mariadb/{DATASET}/ops/m16-post"]))
        profile = {"ops": {"build_selection": "frontier",
                           "storage_engines": ["InnoDB", "MyISAM"]}}
        cli._ops_selection(None, "ops", "mariadb", DATASET, profile,
                           cli.load_engine("mariadb"), {}, None, "tuned",
                           {"checkpoints": str(cp)})
        assert seen["chooser"]["storage_engine"] == "InnoDB"
        assert seen["chooser"]["build_mode"] == "post"
        assert seen["ensure"]["ops_units_done"] is True
        cli._ops_selection(None, "ops", "pgvector", DATASET,
                           {"ops": {"build_selection": "frontier"},
                            "ann": {"pgvector_build_modes": ["incremental",
                                                             "post"]}},
                           cli.load_engine("pgvector"), {}, None, "normalized",
                           {"checkpoints": str(cp)})
        assert seen["chooser"]["storage_engine"] is None
        assert seen["chooser"]["build_mode"] == "incremental"
        assert seen["ensure"]["ops_units_done"] is False


# ---------------------------------------------------------------------------
# MEDIUM-1: every way the selection container can fail ends in a visible
# fallback, never in a traceback that fails the unit with nothing recorded
# ---------------------------------------------------------------------------

class TestContainerFailures:

    def _patch(self, monkeypatch, *, run=None, exists=True):
        removed: List[str] = []
        monkeypatch.setattr(sel.docker_ctl, "remove",
                            lambda name: removed.append(name))
        if callable(exists):
            monkeypatch.setattr(sel.docker_ctl, "image_exists", exists)
        else:
            monkeypatch.setattr(sel.docker_ctl, "image_exists",
                                lambda image: exists)
        if run is not None:
            monkeypatch.setattr(sel.docker_ctl, "run_foreground", run)
        return removed

    def _chooser(self, tmp_path):
        return sel.default_chooser(
            engine="pgvector", dataset=DATASET, resource_pass="tuned",
            resolved=None, paths={"datasets": str(tmp_path),
                                  "annb_results": str(tmp_path / "annb")},
            engine_cfg={"image": {"bench": "vector-bench/pgvector-bench"}},
            vb_root=VB_ROOT)

    def _call(self, chooser):
        err = io.StringIO()
        with redirect_stderr(err):
            result = chooser(files=["a.hdf5"], target_recall=TARGET)
        return result, err.getvalue()

    @pytest.fixture(autouse=True)
    def _tree(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sel, "annb_results_dir",
                            lambda paths, p, r: str(tmp_path / "annb"))

    def _raising(self, exc):
        def run(spec, **kw):
            raise exc
        return run

    def test_image_missing(self, monkeypatch, tmp_path):
        removed = self._patch(monkeypatch, exists=False,
                              run=self._raising(AssertionError("ran")))
        result, err = self._call(self._chooser(tmp_path))
        assert "no bench image" in result["error"]
        assert "no bench image" in err
        assert removed == []

    def test_docker_unavailable_while_checking_the_image(self, monkeypatch,
                                                        tmp_path):
        def exists(image):
            raise FileNotFoundError("docker")
        self._patch(monkeypatch, exists=exists)
        result, _ = self._call(self._chooser(tmp_path))
        assert "cannot ask docker" in result["error"]

    def test_nonzero_exit(self, monkeypatch, tmp_path):
        removed = self._patch(monkeypatch, run=lambda spec, **kw: 3)
        result, err = self._call(self._chooser(tmp_path))
        assert result["error"] == "the selection container exited 3"
        assert removed and removed[0].startswith("vb-ops-build-")

    @pytest.mark.parametrize("lines, expected", [
        ([sel.RESULT_MARKER + "{not json"], "not JSON"),
        ([sel.RESULT_MARKER + "[1, 2]"], "not a JSON object"),
        (["Traceback ...", "boom"], "printed no result line"),
    ])
    def test_unparsable_output(self, monkeypatch, tmp_path, lines,
                               expected):
        def run(spec, timeout, stream, sink):
            sink.extend(lines)
            return 0
        self._patch(monkeypatch, run=run)
        result, _ = self._call(self._chooser(tmp_path))
        assert expected in result["error"]

    def test_timeout_removes_the_container(self, monkeypatch, tmp_path):
        import subprocess
        removed = self._patch(monkeypatch, run=self._raising(
            subprocess.TimeoutExpired(["docker", "run"], 1800)))
        result, _ = self._call(self._chooser(tmp_path))
        assert "timed out after 1800 s" in result["error"]
        assert removed and removed[0].startswith("vb-ops-build-")

    @pytest.mark.parametrize("exc", [FileNotFoundError("docker"),
                                     PermissionError("docker.sock"),
                                     sel.docker_ctl.DockerError("daemon")])
    def test_exec_error_removes_the_container(self, monkeypatch, tmp_path,
                                              exc):
        removed = self._patch(monkeypatch, run=self._raising(exc))
        result, _ = self._call(self._chooser(tmp_path))
        assert type(exc).__name__ in result["error"]
        assert removed

    def test_a_failing_remove_is_reported_not_raised(self, monkeypatch,
                                                     tmp_path):
        import subprocess
        self._patch(monkeypatch, run=lambda spec, **kw: 1)

        def remove(name):
            raise subprocess.TimeoutExpired(["docker", "rm"], 120)
        monkeypatch.setattr(sel.docker_ctl, "remove", remove)
        result, err = self._call(self._chooser(tmp_path))
        assert result["error"] and "docker rm -f" in err

    def test_the_success_path_parses_the_marked_line(self, monkeypatch,
                                                     tmp_path):
        captured: Dict[str, Any] = {}

        def run(spec, timeout, stream, sink):
            captured["spec"] = spec
            sink.extend(["noise", sel.RESULT_MARKER + json.dumps(_payload())])
            return 0
        removed = self._patch(monkeypatch, run=run)
        chooser = sel.default_chooser(
            engine="mariadb", dataset=DATASET, resource_pass="tuned",
            resolved=None, paths={"datasets": str(tmp_path)},
            engine_cfg={"image": {"bench": "img"}}, vb_root=VB_ROOT,
            storage_engine="InnoDB", build_mode="post")
        result, _ = self._call(chooser)
        assert result["choice"]["build"]["m"] == 32
        command = captured["spec"].command
        assert command[command.index("--storage-engine") + 1] == "InnoDB"
        assert command[command.index("--build-mode") + 1] == "post"
        assert removed == []

    def test_a_failure_ends_in_a_visible_fallback(self, monkeypatch,
                                                  tmp_path, any_grid):
        import subprocess
        self._patch(monkeypatch, run=self._raising(
            subprocess.TimeoutExpired(["docker"], 1800)))
        manifest = _manifest(tmp_path, ann={"files": FILES})
        selection, _, err = _ensure(manifest, self._chooser(tmp_path))
        assert selection["status"] == "fallback_reference"
        assert "timed out" in selection["reason"]
        assert "fallback_reference" in err
        assert any("timed out" in w for w in manifest.data["warnings"])


# ---------------------------------------------------------------------------
# MEDIUM-2: a stored choice is reused only while it still describes the run
# ---------------------------------------------------------------------------

def _add_ann(manifest, files=FILES, status="completed"):
    manifest.add_phase("ann", "pgvector", DATASET, status, "t-new", "t-end",
                       {"resource_pass": "tuned", "files": files})


@pytest.mark.usefixtures("any_grid")
class TestStoredChoiceLifetime:

    def test_a_fallback_is_not_reused_once_ann_has_completed(self, tmp_path):
        manifest = _manifest(tmp_path)                  # no ann attempt
        first, _, _ = _ensure(manifest, _Chooser(_payload()))
        assert first["status"] == "fallback_reference"
        _add_ann(manifest)                              # --resume ran ann
        chooser = _Chooser(_payload(m=32))
        second, out, _ = _ensure(manifest, chooser)
        assert len(chooser.calls) == 1
        assert second["status"] == "selected"
        assert "a completed ann attempt now exists" in out
        assert manifest.data["ops_build_selection"][
            f"tuned/pgvector/{DATASET}"]["status"] == "selected"

    def test_a_fallback_is_reused_while_ann_is_still_missing(self, tmp_path):
        manifest = _manifest(tmp_path)
        first, _, _ = _ensure(manifest, _Chooser(_payload()))
        _add_ann(manifest, status="failed")
        chooser = _Chooser(_payload())
        second, out, _ = _ensure(manifest, chooser)
        assert chooser.calls == [] and second == first
        assert "reusing" in out

    def test_a_choice_is_rechosen_after_the_ann_phase_ran_again(self,
                                                                 tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        first, _, _ = _ensure(manifest, _Chooser(_payload(m=32)))
        assert first["ann_attempt"]["attempt"] == 1
        _add_ann(manifest, files={"written": [{"path": "new.hdf5"}]})
        chooser = _Chooser(_payload(m=24))
        second, out, _ = _ensure(manifest, chooser)
        assert chooser.calls[0]["files"] == ["new.hdf5"]
        assert second["build"]["m"] == 24
        assert second["ann_attempt"]["attempt"] == 2
        assert "has run again" in out

    def test_a_failed_later_attempt_does_not_invalidate_the_choice(
            self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        first, _, _ = _ensure(manifest, _Chooser(_payload(m=32)))
        _add_ann(manifest, status="failed")
        chooser = _Chooser(_payload(m=24))
        second, _, _ = _ensure(manifest, chooser)
        assert chooser.calls == [] and second == first

    def test_a_rewritten_file_list_is_a_different_attempt(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        _ensure(manifest, _Chooser(_payload(m=32)))
        manifest.data["phases"][-1]["files"] = {
            "written": [{"path": "edited.hdf5"}]}
        chooser = _Chooser(_payload(m=24))
        second, _, _ = _ensure(manifest, chooser)
        assert len(chooser.calls) == 1 and second["build"]["m"] == 24

    def test_a_changed_build_under_measured_units_is_refused(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        _ensure(manifest, _Chooser(_payload(m=32)))
        _add_ann(manifest, files={"written": [{"path": "new.hdf5"}]})
        with pytest.raises(sel.SelectionRefused, match="already measured"):
            _ensure(manifest, _Chooser(_payload(m=24)), ops_units_done=True)
        # The stored choice is untouched by the refusal.
        assert manifest.data["ops_build_selection"][
            f"tuned/pgvector/{DATASET}"]["build"]["m"] == 32
        # The same build re-chosen is not a change, measured units or not.
        same, _, _ = _ensure(manifest, _Chooser(_payload(m=32)),
                             ops_units_done=True)
        assert same["ann_attempt"]["attempt"] == 2


class TestChoiceOutsideTheGrid:
    """Against a real `render_config`, not a stub."""

    PROFILE = {"name": "t", "datasets": [DATASET], "k": 10,
               "ann": {"enabled": True, "m_values": [16, 32],
                       "ef_search": [10, 40]},
               "ops": {"enabled": True, "build_selection": "frontier"}}

    def _resources(self):
        from orchestrator.config import load_resources, merge_resource_overrides
        return merge_resource_overrides(load_resources("normalized"),
                                        self.PROFILE)

    def _manifest(self, tmp_path):
        manifest = Manifest(str(tmp_path / "run"), "grid-1")
        manifest.add_phase("ann", "mariadb", DATASET, "completed", "t0",
                           "t1", {"resource_pass": "normalized",
                                  "files": FILES})
        return manifest

    def _go(self, manifest, m):
        return _ensure(manifest, _Chooser(_payload(m=m, efc=None)),
                       profile=self.PROFILE, engine="mariadb",
                       resources=self._resources(),
                       resource_pass="normalized")

    def test_a_build_in_the_grid_is_accepted(self, tmp_path):
        selection, _, _ = self._go(self._manifest(tmp_path), 32)
        assert selection["reference_build"] == {"m": 32}
        assert sel.grid_problem("mariadb", self.PROFILE, self._resources(),
                                "normalized", {"m": 16}) is None

    def test_a_fresh_choice_outside_the_grid_is_refused(self, tmp_path):
        with pytest.raises(sel.SelectionRefused, match="not in the normalized"):
            self._go(self._manifest(tmp_path), 48)

    def test_a_stored_choice_the_grid_moved_away_from_is_refused(
            self, tmp_path, monkeypatch):
        manifest = self._manifest(tmp_path)
        self._go(manifest, 32)
        monkeypatch.setitem(self.PROFILE["ann"], "m_values", [16])
        with pytest.raises(sel.SelectionRefused, match="m=32"):
            self._go(manifest, 32)

    def test_an_overlay_restricts_the_grid_the_choice_must_lie_in(self):
        """Stage 2 (`run --overlay`) renders exactly the kept builds; a
        choice the full product would contain but the overlay dropped is
        refused. The overlay reaches `resources` through the profile's
        `resources:` block, as `cmd_run` merges it."""
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        from orchestrator import profile_overlay
        extras = {"m_values_by_engine": {"pgvector": [4, 16, 48]},
                  "pgvector_ef_construction": [64, 200, 400],
                  "ann_builds_by_engine": {"pgvector": [
                      {"M": 4, "efConstruction": 64},
                      {"M": 16, "efConstruction": 200},
                      {"M": 48, "efConstruction": 400}]}}
        selection = {"stage1": {"run_id": "s1"}, "kept": {}}
        selection["overlay_sha256"] = profile_overlay.overlay_digest(
            extras, selection)
        profile, _ = profile_overlay.apply_overlay(
            load_profile("tuned-complete"),
            {"overlay_of": "tuned-complete", "resources": {"extras": extras},
             "build_selection": selection})
        assert sel.build_selection_mode(profile) == "frontier"
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        for kept in ({"m": 4, "ef_construction": 64},
                     {"m": 48, "ef_construction": 400}):
            assert sel.grid_problem("pgvector", profile, resources, "tuned",
                                    kept) is None
        problem = sel.grid_problem("pgvector", profile, resources, "tuned",
                                   {"m": 4, "ef_construction": 400})
        assert problem and "not in the tuned ann grid" in problem
        # And the pre-run check select-builds and run share still passes.
        from orchestrator.reference_build import reference_build_problem
        assert reference_build_problem("pgvector", profile, resources,
                                       "tuned") is None

    def test_a_fallback_is_not_grid_checked_here(self, tmp_path):
        # reference_build_problem already checks it before the run.
        manifest = Manifest(str(tmp_path / "run"), "grid-2")
        selection, _, _ = self._go(manifest, 48)
        assert selection["status"] == "fallback_reference"


class TestEstimateLabels:

    def test_percona_search_is_not_charged_the_m_factor(self):
        from orchestrator.estimate import (frontier_ops_bound,
                                           ops_ingest_factor)
        from orchestrator.estimate_grid import m_cost_factor
        bound = frontier_ops_bound({"ann": {"m_values": [16, 48]},
                                    "ops": {"build_selection": "frontier"}},
                                   ["mongodb", "pgvector", "mariadb"],
                                   ["normalized"])
        assert ops_ingest_factor(bound, "mongodb") == 1.0
        # pgvector's rate is the load alone; the bulk build is in neither
        # figure, as the per-build ann block says of the same engine.
        assert ops_ingest_factor(bound, "pgvector") == 1.0
        assert ops_ingest_factor(bound, "mariadb") == round(
            m_cost_factor("mariadb", 48), 3)
        assert ops_ingest_factor(None, "mariadb") == 1.0

    def test_the_line_says_efc_is_not_bounded(self):
        from orchestrator.estimate import print_load_estimate
        out = io.StringIO()
        with redirect_stdout(out):
            print_load_estimate({"ann": {"m_values": [16, 32]},
                                 "ops": {"build_selection": "frontier"}},
                                ["mariadb"], ["dbpedia-openai-1000k-angular"],
                                ["normalized"], ["ann", "ops"])
        text = " ".join(out.getvalue().split())
        assert "not an upper bound" in text
        assert "priced as the per-build block prices M" in text
        assert "mariadb M=32" in text

    def test_both_estimate_blocks_print_coherently(self, capsys):
        """`cmd_run` prints the headline (with the ops-build line) and then
        #82's per-build ann block; under frontier both appear, the M price is
        one rule, and the headline drops its "Reduce ann.m_values" advice for
        the block that names the real multiplier."""
        from orchestrator.estimate import (estimate_load_hours,
                                           print_load_estimate)
        from orchestrator.estimate_grid import ann_grid_prints, print_ann_grid
        profile = {"ann": {"m_values": [16]},
                   "ops": {"build_selection": "frontier"},
                   "resources": {"extras": {
                       "m_values_by_engine": {"mariadb": [8, 16, 48]}}}}
        args = (profile, ["mariadb"], ["dbpedia-openai-1000k-angular"],
                ["tuned"], ["ann", "ops"])
        prints = ann_grid_prints(*args)
        assert prints
        print_load_estimate(*args, axis_hint=not prints)
        print_ann_grid(*args, estimate_load_hours(*args))
        text = " ".join(capsys.readouterr().out.split())
        assert "ops build: unknown until each ann phase" in text
        assert "mariadb M=48" in text
        assert "ann builds the rendered grid asks for" in text
        assert "Reduce ann.m_values" not in text


# ---------------------------------------------------------------------------
# End to end with #80's headline: the frontier-chosen ops build IS the
# headline build at the calibration target, so no ◇ there
# ---------------------------------------------------------------------------

class TestTheChosenBuildIsTheHeadlineBuild:

    @pytest.fixture(autouse=True)
    def _numpy(self):
        pytest.importorskip("numpy")

    def _run(self):
        from report import generate
        from report.ops_build_choice import choose_ops_build
        from tests.test_e26_report_build_labels import (
            DATASET as E26_DATASET, _mongodb_records, _pgvector_records,
            _valkey_records)
        records = (_mongodb_records() + _pgvector_records()
                   + _valkey_records())
        engines: Dict[str, Any] = {}
        chosen: Dict[str, Any] = {}
        for engine in ("mongodb", "pgvector", "valkey"):
            points = [r for r in records if r["engine"] == engine]
            choice = choose_ops_build(points, TARGET)
            selection = sel._from_choice(engine, choice, TARGET)
            chosen[engine] = selection
            engines[engine] = {"presentation": {
                # The declared build, which the old headline used and which
                # is NOT the frontier's fastest for any of the three.
                "reference_builds": {"tuned": {"m": 16,
                                               "ef_construction": 200}},
                sel.PRESENTATION_KEY: {"tuned": {E26_DATASET: {
                    "build": selection["reference_build"],
                    "source": selection["source"],
                    "status": selection["status"]}}}}}
        manifest = {"run_id": "r", "phases": [], "engines": engines,
                    "config": {"resource_pass": "tuned",
                               "profile": {"name": "t"}}}
        summary = generate.summarize(records, manifest)
        per_engine = summary["per_dataset"][E26_DATASET]["tuned"]
        return summary, per_engine, chosen

    def test_no_mismatch_mark_at_the_target(self):
        summary, per_engine, chosen = self._run()
        for engine, entry in per_engine.items():
            assert chosen[engine]["status"] == "selected", engine
            assert entry["reference_source"] == "manifest: frontier@0.95"
            assert entry["qps_at_recall_95_is_ops_build"] is True, engine
            # The ops-build table's 0.95 figure is the headline's.
            assert entry["reference_qps_at_recall_95"] == \
                entry["qps_at_recall_95"]
        assert chosen["mongodb"]["reference_build"] == {"quantization": "none"}
        assert chosen["pgvector"]["reference_build"] == {
            "m": 16, "ef_construction": 400}

    def test_the_page_carries_no_mark_at_95_and_says_how_it_was_chosen(self):
        from report import frontier_render, render
        summary, per_engine, _ = self._run()
        page = render._headline_tables(summary)
        headline = page.split(frontier_render.OPS_TABLE_HEADING)[0]
        for engine in per_engine:
            row = next(l for l in headline.splitlines()
                       if l.startswith(f"| {render._label(engine)} "))
            cell_95 = row.strip("|").split("|")[2]
            assert frontier_render.OPS_MISMATCH_MARK not in cell_95, engine
        assert "chosen from this run's frontier as the build fastest at " \
               "recall 0.95" in page
        assert "The ops build was chosen per engine" in page


# ---------------------------------------------------------------------------
# Every engine gets a frontier choice from the axis values its ann module
# really stamps, through the primary curve `cmd_run` really passes
# ---------------------------------------------------------------------------

#: (storage_engine, build_mode) exactly as each overlay module writes them:
#: vb_mysql's MySQL family writes its ENGINE= and no build mode; pgvector
#: `heap`, Valkey `memory` and both their build_mode; Percona Search
#: `wiredTiger` and `async` on every point. VillageSQL's module is a
#: vb_mysql subclass too (`_storage_engine` defaults to InnoDB, no build
#: mode), while its ops primary is `incremental`
#: (`capabilities.bulk_vector_index_build: false`) -- points without the
#: axis must still pass that filter.
STAMPS = {"mariadb": ("InnoDB", None), "mariadb123": ("InnoDB", None),
          "alisql": ("InnoDB", None), "percona": ("InnoDB", None),
          "myvector": ("InnoDB", None), "villagesql": ("InnoDB", None),
          "pgvector": ("heap", "post"),
          "valkey": ("memory", "post"), "mongodb": ("wiredTiger", "async")}

#: Two builds per engine on the axis the tuned pass sweeps for it; the
#: second is the faster at 0.95 and must be chosen.
TWO_BUILDS = {
    "mariadb": ({"m": 16}, {"m": 32}), "mariadb123": ({"m": 16}, {"m": 32}),
    "alisql": ({"m": 16}, {"m": 32}),
    "percona": ({"m": 16, "ef_construction": 200},
                {"m": 16, "ef_construction": 400}),
    "myvector": ({"m": 16, "ef_construction": 200},
                 {"m": 16, "ef_construction": 400}),
    "villagesql": ({"m": 16, "ef_construction": 200},
                   {"m": 16, "ef_construction": 400}),
    "pgvector": ({"m": 16, "ef_construction": 200},
                 {"m": 16, "ef_construction": 400}),
    "valkey": ({"m": 16, "ef_construction": 200},
               {"m": 16, "ef_construction": 400}),
    "mongodb": ({"m": 16, "quantization": "scalar"},
                {"m": 16, "quantization": "none"}),
}


class TestEveryEngineGetsAFrontierChoice:

    @pytest.fixture(autouse=True)
    def _numpy(self):
        pytest.importorskip("numpy")

    def _primary(self, engine, profile_name="tuned-complete"):
        from orchestrator import cli
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        profile = load_profile(profile_name)
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        return cli._ops_primary_curve(engine, profile,
                                      cli.load_engine(engine), resources,
                                      "tuned")

    def _points(self, engine, extra_modes=()):
        storage, mode = STAMPS[engine]
        slow, fast = TWO_BUILDS[engine]
        points = []
        for build, rows in ((slow, [(0.90, 900.0, 10), (0.951, 400.0, 40)]),
                            (fast, [(0.93, 950.0, 10), (0.96, 700.0, 40)])):
            for recall, qps, ef in rows:
                p = _point(recall, qps, m=build["m"], ef_search=ef,
                           ef_construction=build.get("ef_construction"),
                           quantization=build.get("quantization"),
                           engine=engine)
                p["storage_engine"], p["build_mode"] = storage, mode
                points.append(p)
        for other in extra_modes:
            # The same two builds measured in another mode, faster still --
            # which must not decide the build the primary mode is measured at.
            for build in (slow,):
                p = _point(0.99, 5000.0, m=build["m"],
                           ef_construction=build.get("ef_construction"),
                           quantization=build.get("quantization"),
                           engine=engine)
                p["storage_engine"], p["build_mode"] = storage, other
                points.append(p)
        return points

    @pytest.mark.parametrize("engine", sorted(STAMPS))
    def test_a_selected_choice_from_realistic_points(self, engine):
        from report.ops_build_choice import choose_ops_build
        storage, mode = self._primary(engine)
        choice = choose_ops_build(self._points(engine), TARGET,
                                  storage_engine=storage, build_mode=mode)
        assert choice is not None, (engine, storage, mode)
        assert choice["status"] == "selected"
        fast = TWO_BUILDS[engine][1]
        for axis, value in fast.items():
            assert choice["build"][axis] == value, (engine, axis)
        assert choice["point"]["qps"] == 700.0

    def test_percona_search_async_points_survive_the_post_primary(self):
        """The reviewer's reproduction: one `async` point, primary `post`."""
        from report.ops_build_choice import choose_ops_build
        assert self._primary("mongodb")[1] == "post"
        point = _point(0.97, 100.0, quantization="none", engine="mongodb")
        point["storage_engine"], point["build_mode"] = "wiredTiger", "async"
        choice = choose_ops_build([point], TARGET, None, "post")
        assert choice and choice["status"] == "selected"

    @pytest.mark.parametrize("engine", ["pgvector", "valkey"])
    def test_a_swept_build_mode_is_still_filtered(self, engine):
        from report.ops_build_choice import choose_ops_build
        storage, mode = self._primary(engine)
        choice = choose_ops_build(
            self._points(engine, extra_modes=("incremental",)), TARGET,
            storage_engine=storage, build_mode=mode)
        assert choice["build"] == {
            **{"m": None, "ef_construction": None, "quantization": None},
            **TWO_BUILDS[engine][1]}
        assert choice["selected_on"]["build_mode"] == "post"


@pytest.mark.usefixtures("any_grid")
class TestTheContainerIsRetriedOnce:

    def test_a_transient_failure_does_not_store_a_fallback(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        answers = [{"error": "the selection container exited 125"},
                   _payload(m=32)]
        calls = []

        def chooser(**kw):
            calls.append(kw)
            return answers.pop(0)
        selection, _, err = _ensure(manifest, chooser)
        assert len(calls) == 2
        assert selection["status"] == "selected"
        assert "retrying the build selection once" in err

    def test_two_failures_fall_back_naming_the_second(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        answers = [{"error": "first"}, {"error": "timed out after 1800 s"}]
        calls = []

        def chooser(**kw):
            calls.append(kw)
            return answers.pop(0)
        selection, _, _ = _ensure(manifest, chooser)
        assert len(calls) == 2
        assert selection["status"] == "fallback_reference"
        assert "timed out" in selection["reason"]

    def test_an_empty_answer_is_not_retried(self, tmp_path):
        manifest = _manifest(tmp_path, ann={"files": FILES})
        chooser = _Chooser({"points": 0, "files": 2, "choice": None})
        selection, _, _ = _ensure(manifest, chooser)
        assert len(chooser.calls) == 1
        assert selection["status"] == "fallback_reference"
