"""E17 -- a report must describe the run it is a report of.

THE DEFECT
----------
`results/annb/<pass>/<fingerprint>/` is shared across runs on purpose. The
fingerprint keys on the resource pass's declared knobs and is deliberately
engine-invariant, so one seven-engine run writes one tree and draws one
seven-engine recall chart; making it engine-specific fragments the tree, and a
six-engine smoke run then produced a recall chart with one engine in it. That
invariant is not in question here and nothing below touches the tree layout.

What the shared tree costs is that it is not a record of any one run, and the
report read all of it. Measured on `percona-20261021-134142`, a run whose
manifest holds exactly one key under `engines`, whose `phases` are two
`percona` entries both `completed`, whose profile declares
`engines: [percona]` and `datasets: [fashion-mnist-784-euclidean]`, and which
whose two phases took 2.1 s and 110.6 s against a 113 s wall clock:

* `report/records.jsonl` carried **396 records across seven engines and three
  corpora** -- alisql 18, mariadb 18, mariadb123 18, mongodb 26, percona 91,
  pgvector 150, valkey 75;
* 388 of them were read out of HDF5 files in the shared tree, written on
  2026-10-14, -16, -17, -19 and -20, and **every one was stamped
  `run_id: "percona-20261021-134142"`**;
* §1 "Engines under test" named Percona Server alone -- correctly, from the
  manifest -- directly above a §4 comparing it against six engines it had not
  been run beside;
* `pareto-gist-960-euclidean.png` and `pareto-glove-100-angular.png` were
  drawn for a profile whose `datasets:` names neither.

Nothing crashed. The report was credible, and it was a comparison of seven
engines over three corpora presented as the output of a two-phase,
one-engine, one-corpus run.

WHAT IS PINNED HERE
-------------------
* `TestANarrowedRunIsReportedNarrow` -- the headline: a manifest naming one
  engine beside a tree holding three reports one, and names the two it left
  out rather than dropping them quietly.
* `TestAWideRunIsStillReportedWide` -- the converse. Over-narrowing is the
  regression fragmenting the tree caused twice, and the archived bundles say
  the scope cannot come from the profile at all: `percona-20261020-140151`
  declares `engines: [percona]` and genuinely started and completed all
  seven, while both `full-20261015-215006` bundles declare **nothing** (the
  `full` profile has no `engines:` key) and one of them started percona alone
  where the other started seven. Only the `engines` block and `phases[]` tell
  those apart — and a failed unit counts, because `full-20261015-215006` has
  9 failed percona phases and an engine scoped out on `status` would vanish
  from the charts instead of surfacing as a named failure.
* `TestTheCorpusHasTheSameDefect` -- the gist and glove charts; why
  `config.profile.datasets` is not a union term either; and the one case that
  choice costs, paid in the open as a named row rather than by admitting a
  file no run id can be read off.
* `TestTheTwoStalenessNoticesAreDifferentFacts` -- a cached result for an
  engine the run DID ask for is stale and `--force` recomputes it; a result
  for an engine the run never asked for is not stale, and no flag makes it
  part of the run. One notice each.
* `TestAnUnnarrowableManifestIsNotNarrowedToNothing` -- the legacy class.
  Reading too much is the defect; reading nothing is worse.
* `TestRegeneratingFromAnArchivedRecordsFile` -- `--from-records` over the
  mixed file the defective generator already wrote. Load-bearing rather than
  incidental: none of the four archived bundles carries an ann tree (no
  `annb/`, not one `.hdf5`), so off the measuring host this is the only route
  to a report, and the only records file in the bundle is the contaminated
  one.
* `TestTheArchivedPerconaBundle` -- the bundle above, by its record census,
  asserting what `report/records.jsonl` holds for it now.

No test here reads `results/`, which is gitignored: every run directory and
ann results tree below is built under `tmp_path`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import records as records_mod  # noqa: E402
from report import charts, generate, loaders, render  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Skips the one test that reads the chart directory. `report/charts.py`
#: guards its matplotlib import and the stack lives in the bench images, so a
#: host built from the documented prerequisites draws nothing -- and an empty
#: directory would satisfy that test for the wrong reason.
needs_matplotlib = pytest.mark.skipif(
    not charts.have_matplotlib(),
    reason="reads drawn charts; matplotlib lives in the bench images")

#: The pass and fingerprint every run on one host at one resource pass shares.
#: Lifted from `percona-20261021-134142`, which read `annb/tuned/a9349676/` --
#: the same tree the seven-engine run before it had written.
PASS = "tuned"
FINGERPRINT = "a9349676"

#: The corpus the percona profile declares, and two the shared tree also held.
FASHION = "fashion-mnist-784-euclidean"
GIST = "gist-960-euclidean"
GLOVE = "glove-100-angular"

SIFT = "sift-128-euclidean"

ALL_SEVEN = ("alisql", "mariadb", "mariadb123", "mongodb", "percona",
             "pgvector", "valkey")

#: `new/full-20261015-215006`'s phase log, in shape: percona over four
#: corpora beside six engines over fashion-mnist alone. 10 `(engine, corpus)`
#: pairs where the product of the two axes is 7 x 4 = 28, which is the only
#: archived bundle where the two readings of "what this run measured" differ
#: at all -- and the reason the scope is the pairs and not the product.
NEW_FULL_PHASES = tuple(
    [("percona", corpus, "completed")
     for corpus in (FASHION, GLOVE, SIFT, GIST)]
    + [(engine, FASHION, "completed")
       for engine in ALL_SEVEN if engine != "percona"])

#: Epoch seconds, before the run below starts. Every file in the shared tree
#: in the reference bundle was older than the run that reported it, which is
#: what the pre-existing staleness notice is for.
MEASURED_EARLIER = 1_760_000_000.0
#: After the run started, so the staleness notice cannot see it. A file a
#: profile running beside this one -- or after it -- left in the same
#: fingerprint tree looks exactly like this.
MEASURED_LATER = 1_800_000_000.0
RUN_STARTED_AT = "2026-10-21T13:41:42Z"
RUN_ID = "percona-20261021-134142"

#: The new §2 heading, and the one it must not be confused with.
OUT_OF_SCOPE_HEADING = "### Recall results from other runs in the shared tree"
STALE_HEADING = "### Recall results that predate this run"

OUT_OF_SCOPE_HEADER = ["Engine", "Dataset", "Result files",
                       "Why it is not part of this run"]
STALE_HEADER = ["Engine", "Dataset", "Measured before this run by"]
HEADLINE_HEADER = ["Engine", "QPS @ recall≥0.90", "QPS @ recall≥0.95",
                   "QPS @ recall≥0.99", "Recall range", "Points measured"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _manifest(engines: Sequence[str],
              *,
              profile_engines: Optional[Sequence[str]] = None,
              profile_datasets: Sequence[str] = (FASHION,),
              phases: Optional[Sequence[Tuple[str, str, str]]] = None,
              fingerprint: Optional[str] = FINGERPRINT) -> Dict[str, Any]:
    """A manifest of the shape `orchestrator/manifest.py` writes.

    `engines` becomes the `engines` block, which `Manifest.set_engine` writes
    one key at a time, once per engine the resolved list named -- so it is the
    run's own record of what it configured and started, and it carries an
    explicit `--engines` narrowing for free.

    `profile_engines` defaults to `engines` and is passed separately to build
    the one shape that matters most: `percona-20261020-140151` declares
    `engines: [percona]` in its profile and holds all seven keys in its block,
    because it predates the fix that made the profile key restrict a run.
    """
    if phases is None:
        phases = [(engine, dataset, "completed")
                  for engine in engines for dataset in profile_datasets]
    config: Dict[str, Any] = {
        "profile": {
            "name": "percona",
            "description": "Percona Server alone",
            "engines": list(engines if profile_engines is None
                            else profile_engines),
            "datasets": list(profile_datasets),
            "ops": {"workloads": ["concurrency", "filtered"]},
        },
        "resource_pass": PASS,
        "resolved_resources": {"name": PASS, "server_cpuset": "0-75",
                               "server_memory_bytes": 120819381043,
                               "build_threads": 8},
        "ops_measurement_version": records_mod.OPS_MEASUREMENT_VERSION,
    }
    if fingerprint is not None:
        config["ann_fingerprint"] = fingerprint
        config["passes"] = [{"resource_pass": PASS,
                             "ann_fingerprint": fingerprint,
                             "resolved_resources":
                                 config["resolved_resources"]}]
    return {
        "run_id": RUN_ID,
        "started_at": RUN_STARTED_AT,
        "finished_at": "2026-10-21T13:43:35Z",
        "status": "completed",
        "host": {"cpu": {}},
        "engines": {engine: {"source": {"tag": f"{engine}-tag",
                                        "commit": "0" * 40},
                             "build": {"march": "x86-64-v3",
                                       "build_type": "RelWithDebInfo"}}
                    for engine in engines},
        "phases": [{"phase": "ann", "engine": engine, "dataset": dataset,
                    "status": status, "resource_pass": PASS,
                    "exit_code": 0 if status == "completed" else 1,
                    "duration_s": 2.1}
                   for engine, dataset, status in phases],
        "warnings": [],
        "config": config,
    }


def _recall(engine: str, dataset: str, *, ef_search: int = 100,
            qps: float = 100.0, recall: float = 0.95,
            mtime: float = MEASURED_EARLIER) -> Dict[str, Any]:
    """One record of the shape `loaders.load_annb_results` writes.

    `extra.source_file` and `extra.source_mtime` are both present because
    every record read out of the shared tree carries them -- 388 of the
    reference bundle's 396 did -- and the first is what lets the report say
    how many FILES it left out rather than how many rows.
    """
    return {
        "run_id": RUN_ID, "phase": "recall_qps", "engine": engine,
        "dataset": dataset, "resource_pass": PASS, "k": 10,
        "m": 16, "ef_search": ef_search, "qps": qps,
        "recall_at_k": recall, "clients": 1, "storage_engine": "InnoDB",
        "metric_space": "euclidean", "march": "x86-64-v3",
        "vector_index_used": True, "timestamp": "2026-10-20T00:00:00Z",
        "extra": {
            "source_file":
                f"{dataset}/10/{engine}/euclidean_M_16_{ef_search}.hdf5",
            "source_mtime": mtime,
        },
    }


def _ops(engine: str, dataset: str, phase: str = "concurrency",
         **fields: Any) -> Dict[str, Any]:
    """An ops record, stamped with the version the manifest declares.

    Stamped, because the version guard refuses a record set whose version
    disagrees with the manifest's before anything else in the generator runs.
    """
    extra = dict(fields.pop("extra", None) or {})
    extra[records_mod.OPS_MEASUREMENT_VERSION_KEY] = \
        records_mod.OPS_MEASUREMENT_VERSION
    record: Dict[str, Any] = {
        "run_id": RUN_ID, "engine": engine, "phase": phase,
        "dataset": dataset, "resource_pass": PASS, "m": 16,
        "build_mode": "incremental", "storage_engine": "InnoDB", "k": 10,
        "clients": 8, "qps": 500.0, "recall_at_k": 0.95, "ef_search": 100,
        "timestamp": "2026-10-21T13:42:00Z", "extra": extra,
    }
    record.update(fields)
    return record


class _TreeCapture:
    """Stands in for `loaders.load_annb_results` and hands back one tree.

    h5py lives in the bench images and not on a host built from the documented
    prerequisites, so the suite cannot write real HDF5 and read it back. What
    is substituted is the loader's OUTPUT, which is what the narrowing acts
    on: the defect is entirely in which of these records reach a chart.
    """

    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)
        self.calls: List[Tuple[str, str]] = []

    def __call__(self, annb_dir, datasets_dir, run_id,
                 resource_pass_default="unknown"):
        self.calls.append((annb_dir, resource_pass_default))
        return [dict(r) for r in self.records]


def _shared_tree(*engines: str,
                 datasets: Sequence[str] = (FASHION,)) -> List[Dict[str, Any]]:
    """Three ef_search points per (engine, corpus), so a count is a count."""
    return [_recall(engine, dataset, ef_search=ef,
                    qps=300.0 - 50.0 * i, recall=0.90 + 0.01 * i)
            for engine in engines for dataset in datasets
            for i, ef in enumerate((10, 40, 160))]


#: `percona-20261021-134142`'s own record census: (engine, corpus, records)
#: as counted in that bundle's `report/records.jsonl`. 388 of its 396 records
#: carried `extra.source_file`, which is what makes them tree reads rather
#: than this run's own measurements.
#:
#: Module level rather than a method on the class that asserts the outcome,
#: because two paths have to be driven over the same census -- the live tree
#: and `--from-records` -- and the archived bundles carry no ann tree at all,
#: so the second is the only one an operator has off-host.
BUNDLE_CENSUS = (
    ("alisql", FASHION, 18),
    ("mariadb", FASHION, 18),
    ("mariadb123", FASHION, 18),
    ("mongodb", FASHION, 26),
    ("pgvector", FASHION, 150),
    ("valkey", FASHION, 75),
    ("percona", FASHION, 35),
    ("percona", GIST, 24),
    ("percona", GLOVE, 24),
)

#: The 8 records that run produced itself, by phase.
BUNDLE_OPS_CENSUS = (("ingest", 1), ("index_build", 1), ("ef_calibration", 2),
                     ("concurrency", 2), ("filtered", 2))


def _bundle_tree() -> List[Dict[str, Any]]:
    return [_recall(engine, dataset, ef_search=10 + i)
            for engine, dataset, count in BUNDLE_CENSUS
            for i in range(count)]


def _bundle_own() -> List[Dict[str, Any]]:
    return [_ops("percona", FASHION, phase=phase, ef_search=10 + i)
            for phase, count in BUNDLE_OPS_CENSUS for i in range(count)]


def _write_run_dir(tmp_path, manifest: Dict[str, Any],
                   ops: Sequence[Dict[str, Any]]) -> str:
    """A run directory `generate.main` accepts, beside the annb root.

    One `ops-*.jsonl` per engine, because `load_ops_records` relabels every
    record with the engine its filename names.
    """
    run_dir = tmp_path / "results" / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for record in ops:
        stem = (f"ops-{record['engine']}-{record['dataset']}-{PASS}"
                f"-m16-incremental.jsonl")
        by_file.setdefault(stem, []).append(record)
    for stem, rows in by_file.items():
        with open(run_dir / stem, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    with open(run_dir / "run-manifest.json", "w") as fh:
        json.dump(manifest, fh)
    return str(run_dir)


def _report(tmp_path, manifest, tree, monkeypatch, *,
            ops: Optional[Sequence[Dict[str, Any]]] = None,
            argv: Sequence[str] = ()) -> Tuple[int, str, List[Dict[str, Any]]]:
    """`generate.main` end to end with the ann loader captured.

    Everything else is the shipped code: the manifest is read off disk, the
    ops records go through `load_ops_records`, the report is rendered, and
    `report/records.jsonl` is written -- which is the file `webui/records.py`
    prefers over the raw ops records, so what lands in it is part of the
    defect rather than a by-product of it.
    """
    monkeypatch.setattr(loaders, "load_annb_results", _TreeCapture(tree))
    run_dir = _write_run_dir(
        tmp_path, manifest,
        ops if ops is not None else [_ops("percona", FASHION)])
    annb = tmp_path / "results" / "annb" / PASS / FINGERPRINT
    annb.mkdir(parents=True, exist_ok=True)
    datasets = tmp_path / "datasets"
    datasets.mkdir(exist_ok=True)
    code = generate.main([
        "--run-dir", run_dir,
        "--annb-results", str(tmp_path / "results" / "annb"),
        "--datasets-dir", str(datasets), *argv])
    return code, _written(run_dir, "report.md"), _rows(run_dir)


def _written(run_dir: str, name: str) -> str:
    path = os.path.join(run_dir, "report", name)
    return open(path).read() if os.path.exists(path) else ""


def _rows(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "report", "records.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def _section(markdown: str, heading: str) -> str:
    """The body under `heading`, up to the next heading or rule.

    Scoped rather than searched whole, because the new notice NAMES the
    engines it excluded: `"MariaDB" not in markdown` is false of a CORRECT
    report, so a test asserting it would pin the opposite of the requirement.
    """
    if heading not in markdown:
        return ""
    body = markdown.split(heading, 1)[1]
    for nxt in ("\n### ", "\n## ", "\n---"):
        body = body.split(nxt, 1)[0]
    return body


def _table_rows(markdown: str, header: Sequence[str]) -> List[List[str]]:
    """Cells of the Markdown table whose header row is exactly `header`.

    Whole cells, not substrings: `"percona" in report` is satisfied by the run
    id in the first line of the document, and the engine label this repository
    gives Percona Server by default IS the bare key, so a substring test for
    it proves nothing at all.
    """
    want = list(header)
    rows: List[List[str]] = []
    collecting = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            collecting = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells == want:
            collecting = True
            continue
        if collecting and set("".join(cells)) <= {"-"}:
            continue
        if collecting:
            rows.append(cells)
    return rows


def _column(markdown: str, header: Sequence[str], index: int = 0) -> List[str]:
    return sorted({row[index] for row in _table_rows(markdown, header)})


def _engines_in(records: Sequence[Dict[str, Any]], phase: str) -> List[str]:
    return sorted({r.get("engine") for r in records
                   if r.get("phase") == phase})


def _datasets_in(records: Sequence[Dict[str, Any]], phase: str) -> List[str]:
    return sorted({r.get("dataset") for r in records
                   if r.get("phase") == phase})


def _label(engine: str) -> str:
    return render._label(engine)


# ---------------------------------------------------------------------------
# The headline
# ---------------------------------------------------------------------------

class TestANarrowedRunIsReportedNarrow:
    """A manifest naming one engine, beside a tree holding three."""

    def _run(self, tmp_path, monkeypatch):
        return _report(
            tmp_path, _manifest(["percona"]),
            _shared_tree("percona", "mariadb", "pgvector"), monkeypatch)

    def test_the_report_carries_one_engine(self, tmp_path, monkeypatch):
        """Honest: the three-engine tree yields percona's recall alone.

        Mutant -- drop the narrowing -- three engines, which is the defect:
        nine records in, nine out, two of the engines never started.
        """
        code, _markdown, written = self._run(tmp_path, monkeypatch)
        assert code == 0
        assert _engines_in(written, "recall_qps") == ["percona"]

    def test_section_four_compares_only_what_ran(self, tmp_path, monkeypatch):
        """The artifact, not the record list. §4 is the headline comparison,
        and it printed a MariaDB row under a §1 naming Percona alone."""
        _code, markdown, _written = self._run(tmp_path, monkeypatch)
        assert _column(markdown, HEADLINE_HEADER) == [_label("percona")]

    def test_it_names_what_it_excluded(self, tmp_path, monkeypatch):
        """A gap is stated, not quietly removed.

        Mutant -- narrow in silence -- a report identical to one produced on a
        machine that had never measured another engine, which is a different
        claim about the tree it read.
        """
        _code, markdown, _written = self._run(tmp_path, monkeypatch)
        section = _section(markdown, OUT_OF_SCOPE_HEADING)
        assert section, markdown
        assert _column(markdown, OUT_OF_SCOPE_HEADER) == \
            [_label("mariadb"), _label("pgvector")]

    def test_it_says_how_many_result_files_belonged_to_other_runs(
            self, tmp_path, monkeypatch):
        """Two engines at three ef_search points: six files, named as a total
        in the prose as well as broken down in the table, because the total is
        the figure an operator checks against the directory."""
        _code, markdown, _written = self._run(tmp_path, monkeypatch)
        assert "6 result files" in _section(markdown, OUT_OF_SCOPE_HEADING)

    def test_it_counts_result_files_and_not_rows(self):
        """The claim is about the tree's FILES. One file per engine must say
        two, not the six a three-point sweep would."""
        from report import scope
        tree = (_shared_tree("percona")
                + [_recall("mariadb", FASHION), _recall("pgvector", FASHION)])
        _kept, dropped = scope.narrow_to_measured_scope(
            tree, _manifest(["percona"]))
        assert sum(d["files"] for d in dropped) == 2
        assert sum(d["records"] for d in dropped) == 2

    def test_the_phase_name_is_the_harness_constant_and_not_a_copy(
            self, monkeypatch):
        """A second spelling of `PHASE_RECALL` here would turn the narrowing
        OFF silently if the harness renamed it: `admits` would stop
        recognising a single record, and every fixture in this file writes the
        literal itself, so the suite would stay green over a report that had
        gone back to publishing seven engines.

        Proved by moving the constant rather than by comparing two strings: a
        stale copy survives an equality assertion against a value a test also
        hardcodes, and does not survive this.
        """
        from report import scope
        assert scope.phase_recall() == records_mod.PHASE_RECALL
        monkeypatch.setattr(records_mod, "PHASE_RECALL", "recall_renamed")
        assert scope.phase_recall() == "recall_renamed"
        manifest = _manifest(["percona"])
        renamed = dict(_recall("mariadb", FASHION), phase="recall_renamed")
        kept, dropped = scope.narrow_to_measured_scope([renamed], manifest)
        assert kept == [] and len(dropped) == 1
        # and the old literal is now just another ops phase, kept whatever it
        # names -- which is what makes the assertion above about the import
        # and not about the predicate.
        kept, dropped = scope.narrow_to_measured_scope(
            [_recall("mariadb", FASHION)], manifest)
        assert len(kept) == 1 and dropped == []

    def test_the_runs_own_ops_records_are_never_narrowed(self, tmp_path,
                                                         monkeypatch):
        """Path B has no shared tree. Every `ops-*.jsonl` is inside the run
        directory, written by this run, so it is this run's own measurement
        whatever the manifest's engine block does or does not say -- and
        nothing else holds a copy of it."""
        _code, _markdown, written = _report(
            tmp_path, _manifest(["percona"]), _shared_tree("percona"),
            monkeypatch,
            ops=[_ops("percona", FASHION), _ops("mariadb", FASHION)])
        assert _engines_in(written, "concurrency") == ["mariadb", "percona"]

    def test_the_narrowing_does_not_invent_a_silent_recall_failure(
            self, tmp_path, monkeypatch):
        """An engine whose ann phase completed and wrote nothing is a real
        failure the report already names. Excluding another engine's records
        must not make an in-scope engine look like one."""
        _code, markdown, _written = self._run(tmp_path, monkeypatch)
        assert "Recall phases that ran and measured nothing" not in markdown


# ---------------------------------------------------------------------------
# The converse
# ---------------------------------------------------------------------------

class TestAWideRunIsStillReportedWide:
    """Over-narrowing is the regression, and it has happened twice.

    `percona-20261020-140151` is why this class cannot be skipped: it ran
    before the engines fix, so its `engines` block holds all seven keys while
    its profile still declares `engines: [percona]`. Its seven-engine report
    is CORRECT. Any narrowing keyed off the profile turns the one bundle whose
    breadth is real into the one bundle the fix breaks.
    """

    def _wide(self):
        return _manifest(ALL_SEVEN, profile_engines=["percona"])

    def test_a_seven_engine_manifest_keeps_all_seven(self, tmp_path,
                                                     monkeypatch):
        _code, _markdown, written = _report(
            tmp_path, self._wide(), _shared_tree(*ALL_SEVEN), monkeypatch)
        assert _engines_in(written, "recall_qps") == sorted(ALL_SEVEN)

    def test_nothing_is_excluded_so_nothing_is_announced(self, tmp_path,
                                                         monkeypatch):
        """The notice must be absent when there is nothing to notice, or it
        becomes a line every reader learns to skip."""
        _code, markdown, _written = _report(
            tmp_path, self._wide(), _shared_tree(*ALL_SEVEN), monkeypatch)
        assert OUT_OF_SCOPE_HEADING not in markdown

    def test_the_profile_key_alone_would_have_dropped_six(self):
        """Non-vacuity for the class: the profile really does name one engine
        here, so the seven above came from what the run DID."""
        from report import scope
        manifest = self._wide()
        assert manifest["config"]["profile"]["engines"] == ["percona"]
        assert scope.measured_engines(manifest) == sorted(ALL_SEVEN)

    def test_an_engine_only_the_phase_log_names_is_kept(self):
        """`engines` is written before a unit runs and `phases[]` after one
        finishes, so neither alone is the whole run. A hand-edited or
        truncated block must not cost an engine the phase log records."""
        from report import scope
        manifest = _manifest(["percona"])
        manifest["phases"].append(
            {"phase": "ann", "engine": "valkey", "dataset": FASHION,
             "status": "completed", "resource_pass": PASS})
        assert scope.measured_engines(manifest) == ["percona", "valkey"]

    def test_a_profile_that_declares_no_engines_does_not_mean_all_of_them(
            self):
        """Both `full-20261015-215006` bundles: the `full` profile has no
        `engines:` key, so `config.profile.engines` is `None`, and one of the
        two started percona alone while the other started all seven. Reading
        "declared nothing" as "every known engine" would re-admit the whole
        contamination for exactly the profiles that cost the most to run.
        Only the block and the log tell the two apart."""
        from report import scope
        narrow = _manifest(["percona"])
        narrow["config"]["profile"].pop("engines")
        wide = _manifest(ALL_SEVEN)
        wide["config"]["profile"].pop("engines")
        assert scope.measured_engines(narrow) == ["percona"]
        assert scope.measured_engines(wide) == sorted(ALL_SEVEN)

    def test_the_orchestrator_still_records_the_resolved_engine_list(self):
        """Why `orchestrator/` needs no change at all, pinned.

        The whole case for reading the scope out of the manifest rather than
        taking a flag is that `Manifest.set_engine` is called once per engine
        in the RESOLVED list -- the one `--engines` narrowed -- and before any
        unit of that engine runs. Nothing asserted either half, and both are
        load-bearing: move the call below the unit loops and a run that failed
        early stops recording the engines it configured; derive it from
        `KNOWN_ENGINES` instead of the loop variable and a narrowed run
        records seven.

        Written the way `tests/test_config.py`'s
        `test_cli_still_calls_set_config_before_the_checkpoint_skip` is, and
        for the same stated reason: `orchestrator/manifest.py` already carries
        a comment saying a call moving below the unit loop would make a notice
        useless with no test to say so. This is that test for `set_engine`.
        """
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "cli.py")).read()
        resolved = source.index('engines = [e.strip() for e in')
        loop = source.index("for engine in engines:")
        call = source.index("manifest.set_engine(")
        checkpoint = source.index("[skip]")
        # Resolved first, then the loop, then the call, then the unit skip.
        # This half is the load-bearing one and is robust to formatting.
        assert resolved < loop < call < checkpoint, (resolved, loop, call,
                                                     checkpoint)
        # And the call takes the loop variable, not a list of its own. Matched
        # on any whitespace: the first version of this pinned the exact
        # 16-space continuation indent, so a reformat or a rename of the loop
        # variable would have reddened `master` for no defect at all -- which
        # is the brittleness `tests/test_e16_citation_ratchet.py` spends its
        # preamble arguing against, in a test written to prevent one.
        assert re.match(r"manifest\.set_engine\(\s*engine\s*,",
                        source[call:call + 80]), source[call:call + 80]

    def test_an_engine_whose_every_unit_failed_stays_in_scope(self):
        """`full-20261015-215006` has 16 percona phases, 9 of them failed and
        nothing else started. `add_phase` records a unit whatever its exit
        code, and an engine scoped out on `status` would vanish from the
        charts instead of surfacing as §2's "Phases that FAILED" row -- which
        is the one line a reader of a part-failed run needs."""
        from report import scope
        manifest = _manifest(["percona"],
                             phases=[("percona", FASHION, "failed")])
        manifest["engines"] = {}
        assert scope.measured_engines(manifest) == ["percona"]


# ---------------------------------------------------------------------------
# Pairs, not the product of the axes
# ---------------------------------------------------------------------------

class TestTheScopeIsPairsAndNotTheProduct:
    """The one shape where "what this run measured" has two readings.

    `new/full-20261015-215006` ran percona over four corpora and six other
    engines over fashion-mnist alone: 10 `(engine, corpus)` pairs against a
    product of 7 x 4 = 28. Every one of the four archived bundles reads the
    same tree, `annb/tuned/a9349676`, because the fingerprint keys on the
    resource pass and not on the profile -- so a second profile depositing a
    `pgvector x gist-960-euclidean` file beside this run's results is this
    host's normal state rather than a hypothesis.

    The product admits that file. It would be drawn on a chart as this run's
    own, with **no §2 row** -- both of its halves are in scope -- and with no
    staleness row either when the file postdates the run, which a file written
    by a profile running concurrently or afterwards does.
    """

    def _manifest_new_full(self, engines: Sequence[str] = ALL_SEVEN):
        return _manifest(engines,
                         profile_engines=None,
                         profile_datasets=(FASHION, GLOVE, SIFT, GIST),
                         phases=list(NEW_FULL_PHASES))

    def _own(self):
        """The ten pairs the run really measured, three ef points each."""
        return [_recall(engine, corpus, ef_search=ef, qps=300.0 - 50.0 * i,
                        recall=0.90 + 0.01 * i)
                for engine, corpus, _status in NEW_FULL_PHASES
                for i, ef in enumerate((10, 40, 160))]

    def test_the_two_readings_really_do_differ_here(self):
        """Non-vacuity. Without this the class could be asserting a
        distinction that no manifest in the tree can express."""
        from report import scope
        manifest = self._manifest_new_full()
        found = scope.scope_of(manifest)
        assert len(found.pairs) == 10
        assert len(found.engines) * len(found.datasets) == 28
        assert found.unpaired == frozenset()

    def test_an_intruder_inside_the_product_is_excluded(self, tmp_path,
                                                        monkeypatch):
        """Honest: pgvector on gist is not a pair this run measured, so it
        goes. Mutant -- scope on the two axes independently -- it is admitted,
        drawn on gist's chart as this run's, and §2 says nothing because both
        halves pass."""
        intruder = _recall("pgvector", GIST, mtime=MEASURED_EARLIER)
        _code, _markdown, written = _report(
            tmp_path, self._manifest_new_full(), self._own() + [intruder],
            monkeypatch)
        assert ("pgvector", GIST) not in {
            (r.get("engine"), r.get("dataset")) for r in written}
        # and every pair the run did measure survives: 10 x 3 points.
        assert len([r for r in written
                    if r.get("phase") == "recall_qps"]) == 30

    def test_the_intruder_is_named_with_the_reason_that_fits_it(
            self, tmp_path, monkeypatch):
        """Not "did not measure this engine" and not "this corpus" -- the run
        measured both. A row claiming either would contradict §1."""
        _code, markdown, _written = _report(
            tmp_path, self._manifest_new_full(),
            self._own() + [_recall("pgvector", GIST)], monkeypatch)
        rows = _table_rows(markdown, OUT_OF_SCOPE_HEADER)
        assert [[r[0], r[1], r[3]] for r in rows] == [
            [_label("pgvector"), GIST, "not this engine on this corpus"]]

    def test_the_notice_does_not_claim_the_pair_it_is_excluding(
            self, tmp_path, monkeypatch):
        """§2's opening sentence, and the defect it was shipped with.

        It was built as "every engine in the report ON every corpus in the
        report" -- the PRODUCT of the two axes -- while the narrowing two
        paragraphs below it is the phase log's PAIRS. On this shape the
        product is 28 and the report carries 10, so the sentence claimed
        pgvector on gist directly above its own table row saying the run never
        measured pgvector on gist. In the published report, and rendering only
        when something was excluded -- which is exactly when a reader is
        checking the claim against the table.

        Honest: the sentence names pgvector on fashion-mnist and no other
        pgvector pair. Mutant -- build it from `summary["engines"]` and
        `summary["datasets"]` -- it names gist beside pgvector and contradicts
        the row below.
        """
        _code, markdown, _written = _report(
            tmp_path, self._manifest_new_full(),
            self._own() + [_recall("pgvector", GIST)], monkeypatch)
        sentence = [line for line in _section(
            markdown, OUT_OF_SCOPE_HEADING).splitlines()
            if "recall results in this report" in line]
        assert len(sentence) == 1, markdown
        covers = sentence[0]
        # The pair count, and every pair, in a claim a reader can check
        # against the table below it.
        assert "10 engine/corpus combinations" in covers
        assert f"{_label('pgvector')} on `{FASHION}`" in covers
        assert f"{_label('percona')} on `{FASHION}`, `{GIST}`" in covers
        # And the claim the product made and this must not.
        assert f"{_label('pgvector')} on `{GIST}`" not in covers

    def test_the_sentence_is_exactly_as_wide_as_the_tables(self, tmp_path,
                                                           monkeypatch):
        """The property, rather than one spelling of it: every pair §4 draws
        is claimed, and nothing else is."""
        _code, markdown, written = _report(
            tmp_path, self._manifest_new_full(),
            self._own() + [_recall("pgvector", GIST)], monkeypatch)
        drawn = {(r["engine"], r["dataset"]) for r in written
                 if r.get("phase") == "recall_qps"}
        section = _section(markdown, OUT_OF_SCOPE_HEADING)
        covers = [line for line in section.splitlines()
                  if "recall results in this report" in line][0]
        for engine, corpus in drawn:
            assert f"`{corpus}`" in covers and _label(engine) in covers
        assert f"{len(drawn)} engine/corpus combinations" in covers

    def test_both_halves_of_the_intruder_are_individually_in_scope(self):
        """What makes the exclusion specifically about the pair: the mutant
        this test guards against is not a typo, it is a whole design."""
        from report import scope
        manifest = self._manifest_new_full()
        assert "pgvector" in scope.measured_engines(manifest)
        assert GIST in scope.measured_datasets(manifest)
        assert not scope.scope_of(manifest).admits("pgvector", GIST)

    def test_the_staleness_notice_would_not_have_caught_it(self, tmp_path,
                                                           monkeypatch):
        """The second half of why the product is unsafe. A file written after
        this run started -- by a profile running beside it or after it -- is
        not stale, so §2's pre-existing notice is silent about it and the
        exclusion census is the only thing that can speak."""
        fresh = _recall("pgvector", GIST, mtime=MEASURED_LATER)
        _code, markdown, _written = _report(
            tmp_path, self._manifest_new_full(), self._own() + [fresh],
            monkeypatch)
        # By PAIR, not by engine: pgvector's own fashion-mnist results are in
        # scope and are themselves cache hits, so the engine is in the stale
        # table for a different and entirely correct reason. Asserting on the
        # engine alone would be asserting the wrong thing and would pass only
        # by accident of which rows the fixture happens to carry.
        #
        # WHICH ASSERTION DOES THE WORK. The first one is satisfied before the
        # mtime is ever consulted -- the scope excludes the pair, so no stale
        # row for it can exist whatever its age -- so on its own it argues the
        # claim rather than establishing it. The census assertion at the end is
        # the discriminating one: it is what fails when the pair is admitted,
        # which is why mutant G (the product of the axes instead of the pairs)
        # kills this test. The middle assertion is the positive control that
        # keeps the first from being vacuous about the table existing at all.
        stale = {(row[0], row[1])
                 for row in _table_rows(markdown, STALE_HEADER)}
        assert (_label("pgvector"), GIST) not in stale
        assert (_label("pgvector"), FASHION) in stale
        assert (_label("pgvector"), GIST) in {
            (row[0], row[1])
            for row in _table_rows(markdown, OUT_OF_SCOPE_HEADER)}

    def test_an_engine_the_log_never_reached_gets_the_runs_corpora(
            self, tmp_path, monkeypatch):
        """The union term. An engine the orchestrator configured and whose
        units the phase log does not record -- a run that died before the
        first one returned, or a truncated manifest -- has no pair to be had,
        and the corpora it would have been measured on are this run's own."""
        from report import scope
        manifest = self._manifest_new_full()
        manifest["engines"]["mariadb"] = {"source": {}, "build": {}}
        manifest["phases"] = [p for p in manifest["phases"]
                              if p["engine"] != "mariadb"]
        found = scope.scope_of(manifest)
        assert found.unpaired == frozenset({"mariadb"})
        assert found.admits("mariadb", GIST)
        assert found.admits("mariadb", FASHION)
        # and it is still bounded by the run's corpora, not by anything.
        assert not found.admits("mariadb", "dbpedia-openai-100k-angular")

    def test_an_engine_in_the_log_with_no_corpus_beside_it_is_unpaired(self):
        """The widening in the union term, which had no probe.

        The term is `engines - {e : (e, d) in pairs}` and not
        `engines - engines_in_phases`. The difference is one shape: a phase
        entry carrying an engine and NO corpus. The narrower form leaves that
        engine out of `unpaired` because the log mentions it, while no pair
        can be formed for it either -- so every one of its records is
        excluded, which is the over-narrowing this whole change must not
        cause.

        Unreachable from today's orchestrator, and deliberately guarded
        anyway: `add_phase` has one production caller, inside
        `for dataset: for phase:`, so a corpus is always in hand. What this
        covers is a hand-edited, truncated or imported manifest -- and
        substituting the narrower formula passed the entire suite, so the
        decision was argued in a docstring and asserted nowhere.
        """
        from report import scope
        manifest = self._manifest_new_full()
        manifest["phases"] = [p for p in manifest["phases"]
                              if p["engine"] != "pgvector"]
        manifest["phases"].append(
            {"phase": "ann", "engine": "pgvector", "status": "completed",
             "resource_pass": PASS})
        found = scope.scope_of(manifest)
        assert found.unpaired == frozenset({"pgvector"})
        assert found.admits("pgvector", FASHION) is True
        assert found.admits("pgvector", GIST) is True
        # Still bounded by the run's own corpora rather than by nothing.
        assert found.admits("pgvector", "dbpedia-openai-100k-angular") is False

    def test_the_union_term_does_not_relax_a_paired_engine(self):
        """It is a union of the pairs with ONE engine's row, not a promotion
        of every engine to the product. pgvector has pairs, so it keeps
        them."""
        from report import scope
        manifest = self._manifest_new_full()
        manifest["engines"]["mariadb"] = {"source": {}, "build": {}}
        manifest["phases"] = [p for p in manifest["phases"]
                              if p["engine"] != "mariadb"]
        found = scope.scope_of(manifest)
        assert found.admits("pgvector", FASHION)
        assert not found.admits("pgvector", GIST)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

class TestTheCorpusHasTheSameDefect:
    """`pareto-gist-960-euclidean.png`, for a profile naming one corpus.

    Percona's own curves for gist and glove were already in the tree from
    earlier runs -- 24 records each in the reference bundle -- so engine
    narrowing alone still leaves two charts and two §4 tables describing
    corpora this run never loaded.
    """

    def _three_corpora(self, tmp_path, monkeypatch):
        return _report(
            tmp_path, _manifest(["percona"]),
            _shared_tree("percona", datasets=(FASHION, GIST, GLOVE)),
            monkeypatch)

    def test_a_corpus_the_run_never_asked_for_is_dropped(self, tmp_path,
                                                         monkeypatch):
        """The in-scope ENGINE on an out-of-scope corpus, which is the case
        engine narrowing cannot catch."""
        _code, _markdown, written = self._three_corpora(tmp_path, monkeypatch)
        assert _datasets_in(written, "recall_qps") == [FASHION]

    def test_the_excluded_corpus_is_named(self, tmp_path, monkeypatch):
        _code, markdown, _written = self._three_corpora(tmp_path, monkeypatch)
        assert _column(markdown, OUT_OF_SCOPE_HEADER, 1) == [GIST, GLOVE]

    @needs_matplotlib
    def test_no_chart_is_drawn_for_it(self, tmp_path, monkeypatch):
        """`pareto-gist-960-euclidean.png` and
        `pareto-glove-100-angular.png` were both in the published bundle."""
        self._three_corpora(tmp_path, monkeypatch)
        chart_dir = os.path.join(str(tmp_path / "results" / RUN_ID),
                                 "report", "charts")
        drawn = sorted(os.listdir(chart_dir))
        assert [name for name in drawn if FASHION in name], drawn
        assert not [name for name in drawn if GIST in name or GLOVE in name]

    def test_a_failed_phase_does_not_lose_its_corpus(self):
        """`add_phase` records a failure, so a corpus whose phase failed is
        still a corpus this run measured -- and whatever recall it did produce
        belongs in the report, beside the Validity row saying it failed."""
        from report import scope
        manifest = _manifest(["percona"], profile_datasets=(FASHION, GIST),
                             phases=[("percona", FASHION, "completed"),
                                     ("percona", GIST, "failed")])
        assert scope.measured_datasets(manifest) == [FASHION, GIST]

    def test_the_declared_list_is_not_a_second_opinion(self):
        """`config.profile.datasets` is the corpus-side twin of
        `config.profile.engines`: a declaration the orchestrator overrides,
        not a record of the run. `--datasets` narrows a run and leaves the
        declaration alone, so a union would publish the corpora the flag
        excluded straight out of the shared tree -- this module's own defect
        surviving its fix.

        Honest: the corpus the phase log records. Mutant -- union the
        declaration -- both, and `run --profile full --datasets fashion` gets
        three other corpora's curves from whoever measured them last.
        """
        from report import scope
        manifest = _manifest(["percona"], profile_datasets=(FASHION, GIST),
                             phases=[("percona", FASHION, "completed")])
        assert scope.measured_datasets(manifest) == [FASHION]

    def test_a_corpus_the_phase_log_never_reached_is_excluded_and_named(
            self, tmp_path, monkeypatch):
        """The cost of the line above, paid deliberately and in the open.

        A run killed inside its second corpus has no `phases[]` entry for it,
        so whatever ann-benchmarks had already written for it is excluded --
        and NAMED, with the count and the reason. The alternative is to admit
        it, and nothing in the shared tree distinguishes a half-finished file
        this run wrote from one another run wrote weeks ago: there is no run
        id in the tree. A row saying four files were left out beats
        publishing another run's curve.
        """
        _code, markdown, written = _report(
            tmp_path,
            _manifest(["percona"], profile_datasets=(FASHION, GIST),
                      phases=[("percona", FASHION, "completed")]),
            _shared_tree("percona", datasets=(FASHION, GIST)), monkeypatch)
        assert _datasets_in(written, "recall_qps") == [FASHION]
        assert _column(markdown, OUT_OF_SCOPE_HEADER, 1) == [GIST]

    def test_the_declared_list_is_the_fallback_when_nothing_finished(self):
        """A run killed before its first unit returned has no phase log at
        all. There is then no per-run record to prefer, and narrowing to what
        the profile asked for beats not narrowing at all."""
        from report import scope
        manifest = _manifest(["percona"], profile_datasets=(FASHION, GIST),
                             phases=[])
        assert scope.measured_datasets(manifest) == [FASHION, GIST]

    def test_a_corpus_only_the_phase_log_names_is_kept(self):
        """The other direction: a resumed run directory can hold a corpus the
        profile has since stopped declaring."""
        from report import scope
        manifest = _manifest(["percona"], profile_datasets=(FASHION,),
                             phases=[("percona", FASHION, "completed"),
                                     ("percona", GLOVE, "completed")])
        assert scope.measured_datasets(manifest) == [FASHION, GLOVE]


# ---------------------------------------------------------------------------
# The four reasons
# ---------------------------------------------------------------------------

class TestEveryReasonSendsTheReaderToTheRightPlace:
    """All four §2 reason codes, because three of them were unguarded.

    Only the `pair` phrase was asserted anywhere, and swapping
    `OUT_OF_SCOPE_ENGINE` with `OUT_OF_SCOPE_DATASET` passed the whole suite.
    Under that mutant percona's own gist curves read "this run did not measure
    this engine" -- sending a reader after an engine that plainly ran, since
    §1 lists it -- and MariaDB on the run's own corpus read "did not measure
    this corpus", which §1 contradicts just as plainly. Those are the two
    misdirections the codes exist to prevent, so each phrase is pinned to the
    shape that produces it.

    One manifest produces all four: a run whose phase log is
    `percona x fashion-mnist` and `pgvector x gist`.
    """

    MANIFEST_PHASES = [("percona", FASHION, "completed"),
                       ("pgvector", GIST, "completed")]

    def _manifest(self):
        return _manifest(["percona", "pgvector"],
                         profile_datasets=(FASHION, GIST),
                         phases=list(self.MANIFEST_PHASES))

    #: (engine, corpus, the code, the phrase §2 must print for it).
    CASES = (
        # Neither the engine nor the corpus appears in the run.
        ("mariadb", SIFT, "both", "neither this engine nor this corpus"),
        # The corpus ran; the engine never did.
        ("mariadb", FASHION, "engine", "this run did not measure this engine"),
        # The engine ran; the corpus never did.
        ("percona", SIFT, "dataset", "this run did not measure this corpus"),
        # Both ran, never together.
        ("pgvector", FASHION, "pair", "not this engine on this corpus"),
    )

    @pytest.mark.parametrize("engine,corpus,code,phrase", CASES)
    def test_the_reason_is_the_one_that_fits(self, engine, corpus, code,
                                             phrase, tmp_path, monkeypatch):
        from report import scope
        manifest = self._manifest()
        assert scope.scope_of(manifest).reason(engine, corpus) == code
        _code, markdown, _written = _report(
            tmp_path, manifest,
            [_recall(e, d) for e, d, _s in self.MANIFEST_PHASES]
            + [_recall(engine, corpus)], monkeypatch)
        rows = [row for row in _table_rows(markdown, OUT_OF_SCOPE_HEADER)
                if row[1] == corpus and row[0] == _label(engine)]
        assert [row[3] for row in rows] == [phrase], markdown

    def test_the_four_cases_really_are_four(self):
        """Non-vacuity: a parametrisation whose cases collapsed to one code
        would assert the same thing four times."""
        assert len({code for _e, _d, code, _p in self.CASES}) == 4
        assert len({phrase for _e, _d, _c, phrase in self.CASES}) == 4

    def test_every_code_the_module_defines_has_a_phrase(self):
        """The mapping must be total, or an unmatched code falls through to a
        generic phrase and the row still renders -- a silent regression."""
        from report import scope
        codes = {scope.OUT_OF_SCOPE_ENGINE, scope.OUT_OF_SCOPE_DATASET,
                 scope.OUT_OF_SCOPE_BOTH, scope.OUT_OF_SCOPE_PAIR}
        assert set(render._OUT_OF_SCOPE_REASON) == codes
        assert len(set(render._OUT_OF_SCOPE_REASON.values())) == 4


# ---------------------------------------------------------------------------
# Two notices, two facts
# ---------------------------------------------------------------------------

class TestTheTwoStalenessNoticesAreDifferentFacts:
    """`--force` is the remedy for one of these and nonsense for the other.

    ann-benchmarks skips a configuration that already has a result file and
    reports success, so an engine the run DID ask for can be served entirely
    from cache: percona's own `ef_construction` sweep in the reference bundle
    was a cache hit from 2026-10-20, and the existing notice is right about
    it. There is no `--force` that makes MariaDB part of a percona-only run.
    """

    def _both(self, tmp_path, monkeypatch):
        return _report(
            tmp_path, _manifest(["percona"]),
            _shared_tree("percona", "mariadb"), monkeypatch)

    def test_an_in_scope_cache_hit_is_still_reported_as_stale(
            self, tmp_path, monkeypatch):
        _code, markdown, _written = self._both(tmp_path, monkeypatch)
        assert STALE_HEADING in markdown
        assert _column(markdown, STALE_HEADER) == [_label("percona")]

    def test_the_stale_notice_does_not_claim_the_excluded_engine(
            self, tmp_path, monkeypatch):
        """Before the fix MariaDB appeared here -- "measured before this run
        by 1.8 days" -- which reads as a cache hit an operator can refresh.
        It was not: nothing in the run asked for it."""
        _code, markdown, _written = self._both(tmp_path, monkeypatch)
        assert _label("mariadb") not in _column(markdown, STALE_HEADER)

    def test_the_new_notice_says_force_is_not_the_remedy(self, tmp_path,
                                                         monkeypatch):
        """Whole phrases, not the bare flag name. Both sections mention
        `--force` and that is the point: one offers it and the other says it
        does not apply, so what discriminates them is the direction of the
        sentence. Asserting the flag's absence would make the notice's
        clearest line a failure.
        """
        _code, markdown, _written = self._both(tmp_path, monkeypatch)
        section = _section(markdown, OUT_OF_SCOPE_HEADING)
        assert "`--force` is not the remedy" in section
        assert "with `--force` to recompute" not in section

    def test_the_stale_notice_offers_it(self, tmp_path, monkeypatch):
        """The converse, and the non-vacuity of the pair: the remedy really is
        in the document, so its direction in each section is a property of
        that section rather than of the file."""
        _code, markdown, _written = self._both(tmp_path, monkeypatch)
        section = _section(markdown, STALE_HEADING)
        assert "with `--force` to recompute" in section
        assert "`--force` is not the remedy" not in section

    def test_the_two_notices_are_both_present_and_distinct(
            self, tmp_path, monkeypatch):
        """Two facts about one tree, in one report, saying different things.
        A single merged notice would have to offer one remedy for both."""
        _code, markdown, _written = self._both(tmp_path, monkeypatch)
        assert STALE_HEADING in markdown
        assert OUT_OF_SCOPE_HEADING in markdown


# ---------------------------------------------------------------------------
# The legacy class
# ---------------------------------------------------------------------------

class TestAnUnnarrowableManifestIsNotNarrowedToNothing:
    """Reading too much is the defect. Reading nothing is worse.

    An archived manifest can carry an empty `engines` block and an empty
    `phases[]`. There is then no record of what the run measured, and a scope
    derived from it is empty -- which, applied, deletes the entire recall half
    of the report.
    """

    def _blank(self):
        manifest = _manifest(["percona"])
        manifest["engines"] = {}
        manifest["phases"] = []
        manifest["config"]["profile"].pop("engines")
        manifest["config"]["profile"].pop("datasets")
        return manifest

    def test_an_empty_scope_narrows_nothing(self, tmp_path, monkeypatch):
        _code, _markdown, written = _report(
            tmp_path, self._blank(), _shared_tree("percona", "mariadb"),
            monkeypatch)
        assert _engines_in(written, "recall_qps") == ["mariadb", "percona"]

    def test_it_says_so_rather_than_narrowing_in_silence(
            self, tmp_path, monkeypatch, capsys):
        """The bargain `_warn_about_unrecorded_passes` already strikes:
        stdout, because a report in this state is exactly what it was before
        this change and §2's staleness notice is still the guard that
        fires."""
        _report(tmp_path, self._blank(), _shared_tree("percona"), monkeypatch)
        assert "cannot narrow" in capsys.readouterr().out

    def test_engines_alone_still_narrow_the_engines(self, tmp_path,
                                                    monkeypatch):
        """Half a manifest is not no manifest. A block naming the engines with
        nothing saying which corpora must still narrow the engines.

        One of a pair; the corpus direction is the next test, and the two have
        to be asserted separately because a single "does the manifest say
        anything" flag satisfies this one and fails that one.
        """
        manifest = _manifest(["percona"])
        manifest["phases"] = []
        manifest["config"]["profile"].pop("datasets")
        _code, _markdown, written = _report(
            tmp_path, manifest,
            _shared_tree("percona", "mariadb", datasets=(FASHION, GIST)),
            monkeypatch)
        assert _engines_in(written, "recall_qps") == ["percona"]
        assert _datasets_in(written, "recall_qps") == [FASHION, GIST]

    def test_corpora_alone_still_narrow_the_corpora(self, tmp_path,
                                                    monkeypatch):
        """The mirror, and the one a collapsed flag loses.

        An archived manifest can carry an empty `engines` block and an empty
        phase log beside a profile that still declares its corpora -- a run
        that died between `set_config`, which saves, and the first
        `set_engine`; a truncated copy; an imported third-party bundle. With
        one "the manifest says something" flag that state admits EVERYTHING,
        including another run's curve for a corpus this one never asked for.

        Measured across the two commits of this change on these very inputs:
        two independent axes keep `(percona, fashion)` and exclude
        `(percona, gist)` as a corpus exclusion; one flag keeps both and
        reports nothing -- while the generator's own stdout still prints the
        corpus scope it has stopped applying.
        """
        manifest = _manifest(["percona"])
        manifest["engines"] = {}
        manifest["phases"] = []
        manifest["config"]["profile"]["datasets"] = [FASHION]
        _code, markdown, written = _report(
            tmp_path, manifest,
            _shared_tree("percona", "mariadb", datasets=(FASHION, GIST)),
            monkeypatch)
        # The corpus axis narrows, and the engine axis -- which knows nothing
        # -- narrows nothing, so mariadb's fashion-mnist records stay.
        assert _datasets_in(written, "recall_qps") == [FASHION]
        assert _engines_in(written, "recall_qps") == ["mariadb", "percona"]
        # Named, and named as a CORPUS exclusion: with no engine knowledge
        # every exclusion here is the corpus's doing, and a row blaming the
        # engine would accuse one the manifest never mentioned either way.
        assert [[row[1], row[3]] for row in
                _table_rows(markdown, OUT_OF_SCOPE_HEADER)] == [
            [GIST, "this run did not measure this corpus"],
            [GIST, "this run did not measure this corpus"]]

    def test_the_two_axes_are_known_independently(self):
        """The flag behind both tests above, asserted directly so a collapse
        is a failure here rather than a surprise two surfaces away."""
        from report import scope
        engines_only = _manifest(["percona"])
        engines_only["phases"] = []
        engines_only["config"]["profile"].pop("datasets")
        corpora_only = _manifest(["percona"])
        corpora_only["engines"] = {}
        corpora_only["phases"] = []
        neither = _manifest(["percona"])
        neither["engines"] = {}
        neither["phases"] = []
        neither["config"]["profile"].pop("datasets")

        first = scope.scope_of(engines_only)
        assert (first.engines_known, first.datasets_known) == (True, False)
        second = scope.scope_of(corpora_only)
        assert (second.engines_known, second.datasets_known) == (False, True)
        assert second.known is True
        third = scope.scope_of(neither)
        assert (third.engines_known, third.datasets_known) == (False, False)
        assert third.known is False


# ---------------------------------------------------------------------------
# The two places the census has to surface other than §2
# ---------------------------------------------------------------------------

class TestTheCensusReachesEverySurface:
    """§2 is rendered twice and skipped once, and all three have to agree."""

    def test_the_html_page_carries_the_notice_too(self, tmp_path,
                                                  monkeypatch):
        """`render_html` builds §2 from the same fragment and then runs it
        through its own table converter. A notice that renders in the Markdown
        and not on the page is the shape this file has shipped before, and the
        page is what anybody actually opens."""
        _report(tmp_path, _manifest(["percona"]),
                _shared_tree("percona", "mariadb"), monkeypatch)
        page = _written(str(tmp_path / "results" / RUN_ID), "report.html")
        assert "Recall results from other runs in the shared tree" in page
        assert _label("mariadb") in page

    def test_narrowing_to_an_excluded_corpus_is_refused_by_name(
            self, tmp_path, monkeypatch):
        """`--datasets` runs AFTER the narrowing, so its refusal lists what
        this run measured rather than what the machine has ever measured.
        Before the fix this succeeded and produced a whole report about a
        corpus the run never loaded."""
        code, _markdown, _written_ = _report(
            tmp_path, _manifest(["percona"]),
            _shared_tree("percona", datasets=(FASHION, GIST)), monkeypatch,
            argv=["--datasets", GIST])
        assert code == 2

    def test_a_run_with_nothing_of_its_own_says_so_on_the_way_out(
            self, tmp_path, monkeypatch, capsys):
        """The one state where the census has nowhere else to go: no records
        means no report, so §2 never renders. Reachable -- an ann-only profile
        whose own phase wrote nothing leaves a run directory with no ops
        records beside a tree full of another run's results, and that is the
        shape that published seven engines under a one-engine manifest."""
        code, markdown, _written_ = _report(
            tmp_path, _manifest(["percona"]), _shared_tree("mariadb"),
            monkeypatch, ops=[])
        assert code == 1
        assert markdown == ""
        err = capsys.readouterr().err
        assert "belonged to another run" in err
        assert "no records found" in err


# ---------------------------------------------------------------------------
# --from-records
# ---------------------------------------------------------------------------

class TestRegeneratingFromAnArchivedRecordsFile:
    """The mixed file the defective generator already wrote and shipped.

    **This is the load-bearing path, not a corner case.** None of the four
    archived bundles carries an ann tree: no `annb/` directory and not one
    `.hdf5` anywhere in them -- `.cache/` holds only filtered-ground-truth
    `.npy` files. The generator's own `--from-records` comment says why ("the
    ann-benchmarks HDF5 tree is not copied with it. The merged records file is
    self-contained"). So off the machine that measured it there is no route
    but `--from-records`, and the only records file in the bundle is the
    contaminated one.

    `report/records.jsonl` is also what `webui/records.py` prefers over the
    raw ops records whenever it exists, and this path REWRITES it -- so a
    rebuild that admitted the mixed set would launder it into the file every
    later reader prefers, under a report that no longer showed it.
    """

    def _rebuild(self, tmp_path, records, manifest=None):
        manifest = manifest or _manifest(["percona"])
        run_dir = _write_run_dir(tmp_path, manifest, [])
        archive = tmp_path / "archived-records.jsonl"
        with open(archive, "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        datasets = tmp_path / "datasets"
        datasets.mkdir(exist_ok=True)
        code = generate.main(["--run-dir", run_dir,
                              "--from-records", str(archive),
                              "--datasets-dir", str(datasets)])
        return code, _written(run_dir, "report.md"), _rows(run_dir)

    def test_a_rebuild_narrows_the_mixed_set(self, tmp_path):
        code, _markdown, rows = self._rebuild(
            tmp_path,
            _shared_tree("percona", "mariadb", "pgvector")
            + [_ops("percona", FASHION)])
        assert code == 0
        assert _engines_in(rows, "recall_qps") == ["percona"]

    def test_a_rebuild_names_what_it_excluded(self, tmp_path):
        _code, markdown, _rows_ = self._rebuild(
            tmp_path,
            _shared_tree("percona", "mariadb") + [_ops("percona", FASHION)])
        assert _column(markdown, OUT_OF_SCOPE_HEADER) == [_label("mariadb")]

    def test_a_rebuild_does_not_launder_the_mixed_set_forward(self, tmp_path):
        """The step that matters. The rebuild REWRITES `records.jsonl`, so a
        rebuild that kept the mixed set would hand it to the web UI again,
        under a report that no longer showed it."""
        _code, _markdown, rows = self._rebuild(
            tmp_path,
            _shared_tree("percona", "mariadb", "pgvector")
            + [_ops("percona", FASHION)])
        assert not [r for r in rows if r.get("engine") != "percona"]

    def test_a_rebuild_keeps_an_ops_record_for_an_engine_not_in_the_block(
            self, tmp_path):
        """Symmetry with the live path. An ops record in an archived file came
        out of the run directory, so the same reasoning applies: it is this
        run's own measurement and nothing else holds a copy."""
        _code, _markdown, rows = self._rebuild(
            tmp_path,
            _shared_tree("percona")
            + [_ops("percona", FASHION), _ops("mariadb", FASHION)])
        assert _engines_in(rows, "concurrency") == ["mariadb", "percona"]

    def test_a_seven_engine_archive_under_a_seven_engine_manifest_keeps_all(
            self, tmp_path):
        """The converse on this path too, and it needs its own test: the
        narrowing here reads the manifest at `--run-dir`, not the archive, so
        a rebuild of `percona-20261020-140151` -- seven engines in its block,
        `engines: [percona]` in its profile -- must come back with seven.
        """
        code, markdown, rows = self._rebuild(
            tmp_path,
            _shared_tree(*ALL_SEVEN) + [_ops("percona", FASHION)],
            manifest=_manifest(ALL_SEVEN, profile_engines=["percona"]))
        assert code == 0
        assert _engines_in(rows, "recall_qps") == sorted(ALL_SEVEN)
        assert OUT_OF_SCOPE_HEADING not in markdown

    def test_the_archived_bundle_rebuilds_to_forty_three_records(self,
                                                                tmp_path):
        """The command the operator actually has off-host, over the census
        the bundle actually holds: 396 records in, 43 out."""
        code, _markdown, rows = self._rebuild(
            tmp_path, _bundle_tree() + _bundle_own())
        assert code == 0
        assert len(rows) == 43
        assert _engines_in(rows, "recall_qps") == ["percona"]
        assert _datasets_in(rows, "recall_qps") == [FASHION]


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

class TestTheArchivedPerconaBundle:
    """`percona-20261021-134142`, by its own record census.

    Every count below was taken from that bundle's `report/records.jsonl`:
    396 records, seven engines, three corpora, 388 carrying
    `extra.source_file`, 8 ops records the run wrote itself. What this class
    asserts is what `report/records.jsonl` holds for that manifest now.
    """

    def test_the_fixture_reproduces_the_published_396(self):
        """Non-vacuity: the fixture is the bundle, not a smaller thing."""
        assert len(_bundle_tree()) == 388
        assert len(_bundle_own()) == 8

    def test_the_report_now_carries_forty_three_records(self, tmp_path,
                                                        monkeypatch):
        """35 percona recall points on fashion-mnist plus the run's own 8 ops
        records. 305 of the 388 tree records belonged to six other engines and
        48 more to two corpora this run never loaded."""
        _code, _markdown, written = _report(
            tmp_path, _manifest(["percona"]), _bundle_tree(), monkeypatch,
            ops=_bundle_own())
        assert len(written) == 43
        assert _engines_in(written, "recall_qps") == ["percona"]
        assert _datasets_in(written, "recall_qps") == [FASHION]

    def test_it_accounts_for_every_one_of_the_353_it_dropped(self):
        from report import scope
        kept, dropped = scope.narrow_to_measured_scope(
            _bundle_tree() + _bundle_own(), _manifest(["percona"]))
        assert len(kept) == 43
        assert sum(d["records"] for d in dropped) == 353
        assert sorted(d["engine"] for d in dropped) == [
            "alisql", "mariadb", "mariadb123", "mongodb", "percona",
            "percona", "pgvector", "valkey"]

    def test_the_notice_names_all_eight_combinations(self, tmp_path,
                                                     monkeypatch):
        _code, markdown, _written = _report(
            tmp_path, _manifest(["percona"]), _bundle_tree(), monkeypatch,
            ops=_bundle_own())
        assert len(_table_rows(markdown, OUT_OF_SCOPE_HEADER)) == 8
        assert "353 result files" in _section(markdown,
                                              OUT_OF_SCOPE_HEADING)


# ---------------------------------------------------------------------------
# F5: every recall number has an ann attempt of this run behind it
# ---------------------------------------------------------------------------

from report import provenance as provenance_mod  # noqa: E402
from report import provenance_render  # noqa: E402

UNPROVENANCED_HEADING = provenance_render.UNPROVENANCED_HEADING
UNPROVENANCED_HEADER = provenance_render.UNPROVENANCED_HEADER
DBPEDIA = "dbpedia-openai-1000k-angular"


def _files(records: Sequence[Dict[str, Any]], kind: str = "written",
           m: Optional[int] = 16) -> Dict[str, List[Dict[str, Any]]]:
    """An ann attempt's `files` block, as `ann_provenance.phase_files`
    writes it, listing exactly these records' source files."""
    listed = [{"path": r["extra"]["source_file"], "mtime": MEASURED_LATER,
               "M": m} for r in records]
    return {"written": listed if kind == "written" else [],
            "skipped": listed if kind == "skipped" else []}


def _attempt(engine: str, dataset: str, files=None, *, phase: str = "ann",
             attempt: int = 1) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"phase": phase, "engine": engine,
                             "dataset": dataset, "status": "completed",
                             "resource_pass": PASS, "exit_code": 0,
                             "duration_s": 2.1, "attempt": attempt}
    if files is not None:
        entry["files"] = files
    return entry


class TestEveryRecallNumberHasAnAttemptBehindIt:
    """A pair the phase log names is not enough: the attempt it names has to
    be the one that wrote the file. `full-20260825` passed the pair check and
    published M=16 recall under a manifest whose ann attempts wrote M=6."""

    def _with_log(self, phases):
        manifest = _manifest(["percona", "pgvector"])
        manifest["phases"] = phases
        return manifest

    def test_an_ann_file_whose_phase_is_absent_is_excluded_and_named(
            self, tmp_path, monkeypatch):
        """pgvector is in the engines block -- configured -- and no attempt
        of it ever returned. `scope.py` admits it on the run's corpora (it is
        `unpaired`); with file lists in the log, nothing of this run wrote
        those files, so they leave §4 and appear in §2."""
        own = _shared_tree("percona")
        manifest = self._with_log([_attempt("percona", FASHION, _files(own))])
        code, markdown, written = _report(
            tmp_path, manifest, own + _shared_tree("pgvector"), monkeypatch)
        assert code == 0
        assert _engines_in(written, "recall_qps") == ["percona"]
        assert _column(markdown, HEADLINE_HEADER) == [_label("percona")]
        assert _column(markdown, UNPROVENANCED_HEADER) == [_label("pgvector")]
        assert "3 result file(s)" in _section(markdown, UNPROVENANCED_HEADING)

    def test_an_ops_only_phase_no_longer_admits_ann_files(
            self, tmp_path, monkeypatch):
        """Before: an ops phase for (pgvector, fashion) made the pair
        "measured", and any ann file for it in the shared tree was drawn as
        this run's. An ops phase writes no ann file; it vouches for none."""
        own = _shared_tree("percona")
        manifest = self._with_log([
            _attempt("percona", FASHION, _files(own)),
            _attempt("pgvector", FASHION, phase="ops")])
        _code, markdown, written = _report(
            tmp_path, manifest, own + _shared_tree("pgvector"), monkeypatch)
        assert _engines_in(written, "recall_qps") == ["percona"]
        rows = _table_rows(markdown, UNPROVENANCED_HEADER)
        assert [r[0] for r in rows] == [_label("pgvector")]
        assert "no ann phase" in rows[0][-1]

    def test_the_ops_only_case_is_also_refused_without_file_lists(self):
        """The ops-only rule does not depend on the new record: an archived
        log naming only an ops phase for a pair still vouches for no ann
        file of it."""
        manifest = self._with_log([
            _attempt("percona", FASHION),
            _attempt("pgvector", FASHION, phase="ops")])
        kept, missing, _info = provenance_mod.split_by_provenance(
            _shared_tree("percona", "pgvector"), manifest)
        assert _engines_in(kept, "recall_qps") == ["percona"]
        assert [r["engine"] for r in missing] == ["pgvector"]

    def test_the_25_august_shape(self):
        """Regression fixture. The MySQL-family ann attempt listed is the
        08-31 one, which wrote M=6 files; the M=16 files beside them on disk
        are what the published recall table used. M=16 goes to
        `unprovenanced_ann`, M=6 stays as measured."""
        def mariadb(m, ef):
            record = _recall("mariadb", DBPEDIA, ef_search=ef)
            record["m"] = m
            record["extra"]["source_file"] = (
                f"{DBPEDIA}/10/mariadb/angular_M_{m}_engine_InnoDB_{ef}.hdf5")
            return record
        m6 = [mariadb(6, ef) for ef in (10, 20, 40)]
        m16 = [mariadb(16, ef) for ef in (10, 20, 40)]
        manifest = _manifest(["mariadb"], profile_datasets=(DBPEDIA,))
        manifest["phases"] = [
            _attempt("mariadb", DBPEDIA, phase="ops"),
            _attempt("mariadb", DBPEDIA, _files(m6, m=6))]
        kept, missing, info = provenance_mod.split_by_provenance(
            m6 + m16, manifest)
        assert sorted(r["m"] for r in kept) == [6, 6, 6]
        assert all(r["extra"]["ann_provenance"] == "measured" for r in kept)
        assert missing == [{
            "resource_pass": PASS, "engine": "mariadb", "dataset": DBPEDIA,
            "reason": provenance_mod.NOT_LISTED, "records": 3, "files": 3,
            "m": [16]}]
        assert info["counts"]["measured"] == 3

    def test_a_cache_hit_is_admitted_and_labelled_reused(
            self, tmp_path, monkeypatch):
        """ann-benchmarks skipped these files on the second attempt. They are
        a measurement of the configuration asked for, so they are drawn --
        and §2 says whose they are, the way `stale_ann` does."""
        own = _shared_tree("percona")
        manifest = self._with_log([
            _attempt("percona", FASHION, _files(own), attempt=1),
            _attempt("percona", FASHION, _files(own, kind="skipped"),
                     attempt=2)])
        kept, missing, info = provenance_mod.split_by_provenance(
            own, {**manifest, "phases": manifest["phases"][1:]})
        assert missing == []
        assert {r["extra"]["ann_provenance"] for r in kept} == {"reused"}
        assert info["reused"][0]["files"] == 3
        _code, markdown, written = _report(tmp_path, {
            **manifest, "phases": manifest["phases"][1:]}, own, monkeypatch)
        assert _column(markdown, HEADLINE_HEADER) == [_label("percona")]
        assert provenance_render.REUSED_HEADING in markdown
        assert "**Reused:**" in markdown

    def test_an_archived_manifest_labels_its_recall_unverified(
            self, tmp_path, monkeypatch):
        """No file lists: nothing can vouch for or against a file. Refusing
        would delete the recall half of every report written before this
        change; the numbers are shown and called what they are."""
        own = _shared_tree("percona")
        _code, markdown, written = _report(
            tmp_path, _manifest(["percona"]), own, monkeypatch)
        assert _column(markdown, HEADLINE_HEADER) == [_label("percona")]
        recall = [r for r in written if r["phase"] == "recall_qps"]
        assert {r["extra"]["ann_provenance"] for r in recall} == \
            {"unverified"}
        assert provenance_render.UNVERIFIED_HEADING in markdown
        section4 = markdown.split("## 4. Recall vs throughput", 1)[1]
        assert "**Unverified:** 3 of 3" in section4.split("\n## ", 1)[0]
        assert UNPROVENANCED_HEADING not in markdown

    def test_a_second_attempt_of_the_same_unit_does_not_hide_the_first(
            self):
        """`add_phase` appends now; the first attempt's written files are
        still the provenance of the records they hold, whatever the second
        attempt did."""
        own = _shared_tree("percona")
        manifest = self._with_log([
            _attempt("percona", FASHION, _files(own), attempt=1),
            _attempt("percona", FASHION, {"written": [], "skipped": []},
                     attempt=2)])
        kept, missing, _info = provenance_mod.split_by_provenance(
            own, manifest)
        assert missing == []
        assert {r["extra"]["ann_provenance"] for r in kept} == {"measured"}


class TestALaterAttemptDecidesHowAUnitEnded:
    """`add_phase` keeps every attempt now. A unit that failed and was then
    re-run successfully is not a failed unit, and §2 must not say it is."""

    def test_a_fixed_failure_is_not_reported(self):
        manifest = _manifest(["percona"])
        failed = dict(manifest["phases"][0], status="failed", exit_code=1,
                      attempt=1)
        manifest["phases"] = [failed, dict(manifest["phases"][0],
                                           attempt=2)]
        summary = generate.summarize([], manifest)
        assert summary["failed_phases"] == []

    def test_a_failure_after_a_success_is_reported(self):
        manifest = _manifest(["percona"])
        done = dict(manifest["phases"][0], attempt=1)
        manifest["phases"] = [done, dict(done, status="failed", exit_code=1,
                                         attempt=2)]
        summary = generate.summarize([], manifest)
        assert len(summary["failed_phases"]) == 1
