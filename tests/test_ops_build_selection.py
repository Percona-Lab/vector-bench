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

    def test_tie_on_qps_goes_to_higher_recall_then_cheaper_build(self):
        # Equal QPS at the target: higher recall wins.
        points = (_curve(16, [(0.951, 500.0, 40)])
                  + _curve(32, [(0.97, 500.0, 40)]))
        assert self._choose(points)["build"]["m"] == 32
        # Equal QPS and recall: the smaller M, whatever the input order.
        points = (_curve(32, [(0.96, 500.0, 40)])
                  + _curve(16, [(0.96, 500.0, 40)]))
        assert self._choose(points)["build"]["m"] == 16
        assert self._choose(list(reversed(points)))["build"]["m"] == 16
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


def _ensure(manifest, chooser, profile=PROFILE, engine="pgvector"):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        selection = sel.ensure_selection(
            manifest, engine=engine, dataset=DATASET, resource_pass="tuned",
            profile=profile, resources={}, chooser=chooser)
    return selection, out.getvalue(), err.getvalue()


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
                                  ["tuned"], ["ops"])
        est = estimate_load_hours(frontier, ["mariadb"],
                                  ["glove-100-angular"], ["tuned"], ["ops"])
        assert est["ops_build_bound"] == {"m": 48, "ingest_factor": 3.0}
        assert est["ops_m_values"] == 1
        assert est["ingest_hours"] == pytest.approx(
            ref["ingest_hours"] * 3.0, rel=0.01)

    def test_the_estimate_says_the_build_is_unknown(self):
        from orchestrator.estimate import print_load_estimate
        profile = {"ann": {"m_values": [16, 32]},
                   "ops": {"build_selection": "frontier"}}
        out = io.StringIO()
        with redirect_stdout(out):
            print_load_estimate(profile, ["mariadb"],
                                ["dbpedia-openai-1000k-angular"], ["tuned"],
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
