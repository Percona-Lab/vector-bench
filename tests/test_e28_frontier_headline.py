"""§4's headline is the frontier over every build swept, and says so honestly.

THE REVIEW. A reader of run `full-20260825` (dbpedia-openai, 990k x 1536)
wrote two things. "Comparing engines at M=16 makes no sense -- different
algorithms, identically named parameters do not mean identical work; what
must be compared is the Pareto frontier on the qps-recall chart." And: "you
use the words 'Pareto frontier' but only vary ef_search at M=16. A Pareto
frontier means exploring the whole parameter space and selecting the best."

PR #77 had gone the other way -- the headline was the reference (ops) build
only -- because the same run's headline gave Percona Search 138 / 124 / 88
from the unquantized index with nothing saying so. These tests hold both:

* The headline QPS-at-recall figures are the frontier over every build
  swept in one (dataset, resource pass), each naming its build; a figure from
  another build than the ops pass measured carries a mark and a note; the
  swept space is disclosed per engine; an engine measured at one build is
  called a curve, never a frontier; the reference figures remain as the
  secondary table.
* Each frontier build's build time and index size are shown, but only
  sizes measured on the server; and per floor, the smallest build that
  reaches it sits beside the fastest.
* Old records -- no build time, no size, no top-level quantization, no
  source file -- still render.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from report import (charts, frontier, frontier_render, generate,  # noqa: E402
                    loaders, render)
from tests.test_e26_report_build_labels import (  # noqa: E402
    DATASET, REFERENCE, _manifest, _mongodb_records, _pgvector_records,
    _recall, _valkey_records)


def _entry(summary, engine, resource_pass="tuned"):
    return summary["per_dataset"][DATASET][resource_pass][engine]


def _summary(records=None, manifest=None):
    records = records if records is not None else (
        _mongodb_records() + _pgvector_records() + _valkey_records())
    return generate.summarize(records, manifest if manifest is not None
                              else _manifest(REFERENCE))


def _row(table: str, label: str) -> List[str]:
    line = next(l for l in table.splitlines() if l.startswith(f"| {label} "))
    return [c.strip() for c in line.strip("|").split("|")]


def _headline(table: str) -> str:
    """The headline table and its notes, without the ops-build view."""
    return table.split(frontier_render.OPS_TABLE_HEADING)[0]


def _with_cost(records: List[Dict[str, Any]], costs: Dict[Any, Dict[str, Any]]
               ) -> List[Dict[str, Any]]:
    """Copy `build_wall_s`/`index_bytes` onto every point of a build, keyed by
    (m, ef_construction, quantization) -- the way one ann result file's
    attributes reach every point measured on that index."""
    out = []
    for r in records:
        params = loaders.build_params(r)
        key = (params["m"], params["ef_construction"], params["quantization"])
        out.append({**r, **costs.get(key, {})})
    return out


# ---------------------------------------------------------------------------
# The headline is the whole-sweep frontier, every figure named
# ---------------------------------------------------------------------------

class TestTheHeadlineIsTheFrontierOverEveryBuild:

    def test_the_headline_takes_the_fastest_build_and_names_it(self):
        """Honest: 610 from efC=400 at 0.95. Mutant (headline from the
        reference build, PR #77's rule): 520 from efC=200."""
        entry = _entry(_summary(), "pgvector")
        assert entry["qps_at_recall_95"] == 610.0
        assert entry["qps_at_recall_95_build_label"] == "M=16 efC=400"
        assert entry["reference_qps_at_recall_95"] == 520.0

    def test_percona_search_gets_its_unquantized_figures_labelled(self):
        """The 25 Aug numbers, now with the build under each one."""
        entry = _entry(_summary(), "mongodb")
        assert [entry[f"qps_at_recall_{f}"] for f in (90, 95, 99)] == \
            [138.0, 124.0, 88.0]
        assert entry["qps_at_recall_90_build_label"] == "quantization=none"

    def test_best_is_kept_as_an_alias_of_the_headline(self):
        """A reader of the earlier layout, whose headline was the reference build, looked for the sweep under `best_`."""
        entry = _entry(_summary(), "valkey")
        for floor in (90, 95, 99):
            assert (entry[f"best_qps_at_recall_{floor}"]
                    == entry[f"qps_at_recall_{floor}"])

    def test_the_rendered_headline_cell_names_its_build(self):
        table = render._headline_tables(_summary())
        row = _row(_headline(table), render._label("pgvector"))
        assert "610 @0.968" in row[2] and "M=16 efC=400" in row[2]

    def test_points_and_range_span_the_whole_sweep(self):
        entry = _entry(_summary(), "mongodb")
        assert entry["points"] == 7
        assert entry["min_recall"] == pytest.approx(0.71)
        assert entry["max_recall"] == pytest.approx(0.997)

    def test_resource_passes_are_never_one_frontier(self):
        """A faster build in the OTHER pass must not reach this pass's
        headline: a frontier across two machines is a frontier of nothing."""
        tuned = _recall("pgvector", 0.95, 500.0, ef_search=40,
                        ef_construction=200)
        normalized = _recall("pgvector", 0.96, 900.0, ef_search=40,
                             ef_construction=400,
                             resource_pass="normalized")
        summary = generate.summarize([tuned, normalized], {})
        assert _entry(summary, "pgvector")["qps_at_recall_95"] == 500.0
        assert _entry(summary, "pgvector",
                      "normalized")["qps_at_recall_95"] == 900.0

    def test_the_floor_chart_bars_carry_the_build_and_the_mark(self):
        """Matplotlib-free: the label helper the bar text is drawn from."""
        entry = _entry(_summary(), "mongodb")
        top, build = charts.floor_bar_labels(entry, 90)
        assert top == f"138 {frontier_render.OPS_MISMATCH_MARK}"
        assert build == "quantization=none"
        assert charts.floor_bar_labels({}, 90) == ("not reached", None)

    def test_the_recall_floor_gaps_use_the_whole_sweep(self):
        """A gap means EVERY configuration swept is above the floor. With the
        unquantized build in the headline, Percona Search has points from
        0.71 up, so no floor of it is a gap."""
        summary = _summary()
        assert not [g for g in summary["recall_floor_gaps"]
                    if g["engine"] == "mongodb"]


# ---------------------------------------------------------------------------
# A headline figure from another build than the ops one is marked
# ---------------------------------------------------------------------------

class TestTheOpsBuildMismatchIsSaidNextToTheFigure:

    def test_the_flag_is_false_where_the_build_differs(self):
        entry = _entry(_summary(), "pgvector")
        assert entry["qps_at_recall_95_is_ops_build"] is False

    def test_the_flag_is_true_where_the_frontier_is_the_ops_build(self):
        """pgvector at 0.90: efC=200 at 700 beats efC=400 at 690."""
        entry = _entry(_summary(), "pgvector")
        assert entry["qps_at_recall_90_build_label"] == "M=16 efC=200"
        assert entry["qps_at_recall_90_is_ops_build"] is True

    def test_the_cell_carries_the_mark_only_where_it_differs(self):
        table = _headline(render._headline_tables(_summary()))
        row = _row(table, render._label("pgvector"))
        mark = frontier_render.OPS_MISMATCH_MARK
        assert mark not in row[1]
        assert mark in row[2]

    def test_the_note_names_the_ops_build(self):
        """Concurrency and churn describe the ops build; the note says which
        one that is, so a reader does not join two indexes' figures."""
        table = _headline(render._headline_tables(_summary()))
        notes = table.split(frontier_render.OPS_MISMATCH_MARK, 1)[1]
        assert "quantization=scalar" in notes
        assert "M=16 efC=200" in notes

    def test_an_unknown_ops_build_is_not_called_a_mismatch(self):
        """Two builds, no manifest, no ops record: whether the headline is
        the ops build cannot be known, and "different" would be a guess."""
        records = [_recall("mariadb", 0.95, 400.0, ef_search=10, m=8),
                   _recall("mariadb", 0.97, 300.0, ef_search=10, m=16)]
        summary = generate.summarize(records, {})
        entry = _entry(summary, "mariadb")
        assert entry["qps_at_recall_90_is_ops_build"] is None
        table = _headline(render._headline_tables(summary))
        row = _row(table, render._label("mariadb"))
        assert frontier_render.OPS_MISMATCH_MARK not in row[1]
        assert "ops build is not recorded" in table

    def test_the_mark_is_not_a_glyph_already_in_use(self):
        """`†` and `‡` already mean "above the floor" and "never reached"."""
        assert frontier_render.OPS_MISMATCH_MARK not in (
            render.FLOOR_MARK, render.UNREACHED_MARK)


# ---------------------------------------------------------------------------
# The swept space, and "curve" where only ef_search varied
# ---------------------------------------------------------------------------

def _single_build_records() -> List[Dict[str, Any]]:
    return [_recall("mariadb", 0.95, 400.0, ef_search=10),
            _recall("mariadb", 0.98, 250.0, ef_search=40),
            _recall("mariadb", 0.99, 120.0, ef_search=100)]


class TestASingleBuildIsACurveNotAFrontier:

    def test_the_swept_space_knows_it_is_one_build(self):
        entry = _entry(generate.summarize(_single_build_records(), {}),
                       "mariadb")
        assert entry["swept_space"]["single_build"] is True
        assert entry["sweep_builds"] == 1

    def test_the_swept_table_says_curve_and_not_explored(self):
        table = render._headline_tables(
            generate.summarize(_single_build_records(), {}))
        section = table.split("**Swept space**")[1]
        row = _row(section, render._label("mariadb"))
        assert "curve at `M=16`" in row[2]
        assert "not explored" in row[2]
        assert "frontier" not in row[2]

    def test_a_multi_build_engine_is_called_a_frontier(self):
        table = render._headline_tables(_summary())
        section = table.split("**Swept space**")[1]
        assert "frontier over 2 builds" in _row(
            section, render._label("pgvector"))[2]

    def test_the_section_intro_does_not_say_pareto_frontier_for_curves(self):
        """Honest: a run that varied only ef_search says "not a Pareto
        frontier". Mutant (the old fixed sentence): "Each engine's Pareto
        frontier is the best throughput ... across the whole parameter
        sweep" -- the exact claim the reviewer refuted."""
        intro = frontier_render.section_intro(
            generate.summarize(_single_build_records(), {}))
        assert "not a Pareto frontier" in intro
        assert "build space was not explored" in intro
        assert "Each engine's Pareto frontier" not in intro

    def test_a_mixed_run_names_the_curves(self):
        intro = frontier_render.section_intro(
            _summary(_single_build_records() + _pgvector_records()))
        assert "recall–throughput curve at that build" in intro

    def test_a_fully_swept_run_says_neither(self):
        intro = frontier_render.section_intro(_summary(_pgvector_records()))
        assert "curve" not in intro

    def test_the_report_text_no_longer_hardcodes_the_claim(self):
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        assert "Each engine's Pareto frontier is the best" not in source

    def test_line_kind(self):
        entry = _entry(generate.summarize(_single_build_records(), {}),
                       "mariadb")
        assert frontier_render.line_kind(entry) == \
            "recall–throughput curve at `M=16`"
        assert frontier_render.line_kind({"sweep_builds": 3}) == \
            "Pareto frontier over 3 builds"

    def test_the_chart_legend_says_curve_for_one_build(self):
        label = charts.pareto_line_label("MariaDB", _single_build_records())
        assert label == "MariaDB — curve at M=16"
        assert charts.pareto_line_label("pgvector",
                                        _pgvector_records()) == "pgvector"


class TestTheSweptSpaceIsDisclosed:

    def test_varied_and_fixed_axes(self):
        space = frontier.swept_space(_pgvector_records())
        text = frontier.spell_swept(space)
        assert "efC ∈ {200, 400}" in text
        assert "ef_search 40–200 (3 values)" in text
        assert "fixed: M=16" in text

    def test_quantization_is_read_from_archived_records(self):
        """A records.jsonl from before the loader copied the attribute
        carries it only in `extra.parameters` and the file name."""
        space = frontier.swept_space(_mongodb_records(top_level_quant=False))
        assert space["axes"]["quantization"] == ["none", "scalar"]

    def test_mongot_is_not_said_to_have_a_fixed_m(self):
        """Its M is a file-name label, not a knob (`builds.build_label`)."""
        space = frontier.swept_space(_mongodb_records())
        assert "m" not in space["axes"]
        assert "M=" not in frontier.spell_swept(space)

    def test_the_build_count_comes_from_the_signature(self):
        """Two result files are two builds even when no named column
        differs -- the axis this list has never heard of."""
        a = _recall("percona", 0.9, 100.0, ef_search=10)
        b = dict(a, extra={"source_file": "d/10/percona/angular_M_16_new_"
                                          "axis_1_10.hdf5"})
        assert frontier.swept_space([a, b])["builds"] == 2


# ---------------------------------------------------------------------------
# Build cost on the frontier
# ---------------------------------------------------------------------------

COSTS = {(16, 200, None): {"build_wall_s": 300.0, "index_bytes": 2 * 2**30},
         (16, 400, None): {"build_wall_s": 700.0, "index_bytes": 3 * 2**30}}


class TestBuildCostOnTheFrontier:

    def test_every_frontier_build_has_its_time_and_size(self):
        entry = _entry(_summary(_with_cost(_pgvector_records(), COSTS)),
                       "pgvector")
        rows = {r["build_label"]: r for r in entry["frontier_builds"]}
        assert rows["M=16 efC=400"]["time_to_queryable_s"] == 700.0
        assert rows["M=16 efC=400"]["time_source"] == "ann"
        assert rows["M=16 efC=400"]["index_bytes"] == 3 * 2**30
        assert rows["M=16 efC=400"]["is_ops_build"] is False
        assert rows["M=16 efC=200"]["is_ops_build"] is True

    def test_only_builds_on_the_frontier_are_listed(self):
        """Valkey's efC=64 point (0.80, 900 q/s) is the fastest anywhere, so
        it is on the frontier; both efC=200 points are dominated by efC=400's
        (0.95, 850) and (0.99, 640), so that build contributes nothing."""
        entry = _entry(_summary(), "valkey")
        labels = {r["build_label"] for r in entry["frontier_builds"]}
        assert labels == {"M=16 efC=64", "M=16 efC=400"}

    def test_a_zero_size_is_not_measured(self):
        """`vb_mysql` returns 0 when its glob matched nothing."""
        records = _with_cost(_pgvector_records(),
                             {(16, 400, None): {"index_bytes": 0}})
        entry = _entry(_summary(records), "pgvector")
        row = next(r for r in entry["frontier_builds"]
                   if r["build_label"] == "M=16 efC=400")
        assert row["index_bytes"] is None
        table = frontier_render.build_cost_table({"pgvector": entry})
        assert "not measured" in table

    def test_the_ops_record_supplies_a_missing_size_and_says_so(self):
        ops = {"phase": "index_build", "engine": "pgvector",
               "dataset": DATASET, "resource_pass": "tuned", "m": 16,
               "ef_construction": 200, "build_mode": "post",
               "index_bytes": 5 * 2**30, "extra": {}}
        entry = _entry(_summary(_pgvector_records() + [ops]), "pgvector")
        row = next(r for r in entry["frontier_builds"]
                   if r["build_label"] == "M=16 efC=200")
        assert row["index_bytes"] == 5 * 2**30
        assert row["index_bytes_source"] == "ops"
        table = frontier_render.build_cost_table({"pgvector": entry})
        assert "5.0 GiB (ops)" in table

    def test_two_matching_ops_records_are_not_chosen_between(self):
        ops = {"phase": "index_build", "engine": "pgvector",
               "dataset": DATASET, "resource_pass": "tuned", "m": 16,
               "ef_construction": 200, "index_bytes": 5 * 2**30, "extra": {}}
        records = (_pgvector_records()
                   + [dict(ops, build_mode="post"),
                      dict(ops, build_mode="incremental",
                           index_bytes=6 * 2**30)])
        entry = _entry(_summary(records), "pgvector")
        row = next(r for r in entry["frontier_builds"]
                   if r["build_label"] == "M=16 efC=200")
        assert row["index_bytes"] is None

    def test_valkey_size_is_labelled_resident(self):
        records = [dict(r, storage_engine="memory", index_bytes=2**30)
                   for r in _valkey_records()]
        entry = _entry(_summary(records), "valkey")
        table = frontier_render.build_cost_table({"valkey": entry})
        assert "1.0 GiB resident" in table

    def test_the_loader_never_uses_ann_benchmarks_index_size(
            self, tmp_path, monkeypatch):
        """Upstream's `index_size` is the client process's RSS delta. A
        result file with it and without the module's own `index_bytes`
        loads with no size at all, never the client figure."""
        from tests import test_e18_ann_build_timing_reaches_the_records as e18
        monkeypatch.setitem(sys.modules, "h5py",
                            __import__("types").SimpleNamespace(
                                File=e18._StubH5File))
        datasets = tmp_path / "datasets"
        datasets.mkdir()
        e18._write_truth(str(datasets))
        attrs = e18._non_mysql_attrs("pgvector", 1.0, 2.0, 3.0, M=16)
        attrs.pop("index_bytes", None)
        attrs["index_size"] = 123456.0
        e18._write_result(str(tmp_path / "annb"), attrs)
        [record] = loaders.load_annb_results(str(tmp_path / "annb"),
                                             str(datasets), "r")
        assert record["index_bytes"] is None
        assert record["build_wall_s"] == 3.0


# ---------------------------------------------------------------------------
# The smallest build that reaches each floor
# ---------------------------------------------------------------------------

class TestTheSmallestBuildThatReachesEachFloor:

    def test_by_size_the_smaller_slower_build_wins(self):
        """At 0.95 the fastest is efC=400 (610, 3 GiB); the smallest that
        reaches it is efC=200 (520, 2 GiB)."""
        entry = _entry(_summary(_with_cost(_pgvector_records(), COSTS)),
                       "pgvector")
        assert entry["smallest_qps_at_recall_95"] == 520.0
        assert entry["smallest_qps_at_recall_95_build_label"] == "M=16 efC=200"
        assert entry["smallest_qps_at_recall_95_ranked_by"] == "index size"
        assert entry["smallest_qps_at_recall_95_index_bytes"] == 2 * 2**30

    def test_only_builds_that_reach_the_floor_compete(self):
        """At 0.99 only efC=400 gets there (0.993), however big it is."""
        entry = _entry(_summary(_with_cost(_pgvector_records(), COSTS)),
                       "pgvector")
        assert entry["smallest_qps_at_recall_99_build_label"] == "M=16 efC=400"

    def test_without_sizes_it_ranks_by_m(self):
        records = [_recall("mariadb", 0.95, 900.0, ef_search=10, m=32),
                   _recall("mariadb", 0.95, 400.0, ef_search=10, m=8)]
        entry = _entry(generate.summarize(records, {}), "mariadb")
        assert entry["smallest_qps_at_recall_90_build_label"] == "M=8"
        assert entry["smallest_qps_at_recall_90_ranked_by"] == "M, then efC"
        assert entry["qps_at_recall_90_build_label"] == "M=32"

    def test_a_partial_size_census_falls_back_to_m(self):
        """Ranking a measured size against "not measured" would rank by the
        gap in the instrument."""
        records = _with_cost(_pgvector_records(),
                             {(16, 400, None): {"index_bytes": 2**20}})
        entry = _entry(_summary(records), "pgvector")
        assert entry["smallest_qps_at_recall_95_ranked_by"] == "M, then efC"
        assert entry["smallest_qps_at_recall_95_build_label"] == "M=16 efC=200"

    def test_a_tie_without_sizes_is_said_not_decided(self):
        """Percona Search's two quantizations share M and efC and carry no
        size here, so "smallest" cannot be told; the cell says tie. (Both
        reach 0.90 in this fixture; on the 25 Aug run scalar did not.)"""
        records = [_recall("mongodb", 0.93, 200.0, ef_search=100,
                           quantization="scalar"),
                   _recall("mongodb", 0.95, 150.0, ef_search=100,
                           quantization="none")]
        entry = _entry(generate.summarize(records, {}), "mongodb")
        assert entry["smallest_qps_at_recall_90_ranked_by"] == frontier.TIED
        table = frontier_render.smallest_table({"mongodb": entry})
        assert "(tie)" in table and "cannot be told apart" in table

    def test_the_table_is_printed_beside_the_headline(self):
        table = render._headline_tables(
            _summary(_with_cost(_pgvector_records(), COSTS)))
        section = table.split("**Smallest build that reaches each floor**")[1]
        row = _row(section, render._label("pgvector"))
        assert "520" in row[2] and "M=16 efC=200" in row[2]
        assert "2.0 GiB" in row[2]
        assert table.index("**Smallest build") < table.index(
            frontier_render.OPS_TABLE_HEADING)


# ---------------------------------------------------------------------------
# Old records still render
# ---------------------------------------------------------------------------

class TestOldRecordsStillRender:

    def _archived(self) -> List[Dict[str, Any]]:
        """Shaped like a records.jsonl merged long before this change: no
        source file, no quantization, no build time, no size, no pass."""
        return [{"phase": "recall_qps", "engine": "mariadb",
                 "dataset": DATASET, "recall_at_k": r, "qps": q, "m": m,
                 "ef_search": ef, "k": 10, "extra": {}}
                for r, q, m, ef in ((0.93, 500.0, 16, 10),
                                    (0.97, 300.0, 16, 40),
                                    (0.96, 450.0, 32, 10))]

    def test_summarize_and_render(self):
        summary = generate.summarize(self._archived(), {})
        entry = summary["per_dataset"][DATASET]["unknown"]["mariadb"]
        assert entry["qps_at_recall_95"] == 450.0
        assert entry["qps_at_recall_95_build_label"] == "M=32"
        table = render._headline_tables(summary)
        assert "not measured" in table
        assert "M ∈ {16, 32}" in table

    def test_an_entry_built_without_the_new_fields_renders(self):
        """A hand-built entry (the shape tests and older callers make) has
        no swept space, no flags and no smallest figures."""
        entry = {"points": 2, "min_recall": 0.9, "max_recall": 0.99,
                 "qps_at_recall_90": 100.0, "qps_at_recall_90_recall": 0.95}
        table = render._headline_tables(
            {"per_dataset": {"d": {None: {"mariadb": entry}}}})
        assert "100 @0.950" in table
        assert frontier_render.OPS_MISMATCH_MARK not in table
        assert frontier_render.line_kind(entry) == "Pareto frontier"

    def test_the_whole_page_renders(self):
        summary = generate.summarize(self._archived(), {"run_id": "old"})
        page = render.render_markdown({"run_id": "old"}, summary,
                                      self._archived(), {}, "old")
        assert "## 4. Recall vs throughput" in page


# ---------------------------------------------------------------------------
# The archived full-20260825 shapes: ops records stamped with defaults
# ---------------------------------------------------------------------------

def _ops_index_build(engine: str, **fields: Any) -> Dict[str, Any]:
    """An `index_build` record as the 25 Aug ops pass wrote it: `post` and
    `InnoDB` stamped on every engine, knob or not, and no manifest
    declaration above it."""
    extra = fields.pop("extra", {})
    record = {"phase": "index_build", "engine": engine, "dataset": DATASET,
              "resource_pass": "tuned", "m": 16, "ef_construction": None,
              "build_mode": "post", "storage_engine": "InnoDB",
              "extra": extra}
    record.update(fields)
    return record


def _archived_mongot() -> List[Dict[str, Any]]:
    """mongot ann points say `async`/`wiredTiger`; its ops record says
    `post`/`InnoDB`. Sizes and times only on the ops record."""
    points = []
    for r in _mongodb_records(top_level_quant=False):
        points.append({**r, "build_mode": "async",
                       "storage_engine": "wiredTiger"})
    ops = _ops_index_build(
        "mongodb", build_wall_s=100.142, ingest_wall_s=14.344,
        index_bytes=16238987031,
        extra={"quantization": "scalar", "ef_construction_tunable": False,
               "separable_build": True})
    return points + [ops]


class TestTheArchivedOpsRecordsMatchTheirBuilds:

    def _mongot(self):
        return _entry(generate.summarize(_archived_mongot(), {}), "mongodb")

    def test_the_scalar_ops_build_matches_its_eight_points(self):
        """Honest: the scalar curve IS the ops build. Mutant (compare the
        stamped `post` against `async`): 0 matches, "Ops build: no" and
        "no recall point was measured at its ops build"."""
        entry = self._mongot()
        assert entry["reference_points"] == 4
        assert entry["reference_max_recall"] == pytest.approx(0.892)
        rows = {r["build_label"]: r for r in entry["frontier_builds"]}
        assert rows["quantization=scalar"]["is_ops_build"] is True
        assert rows["quantization=none"]["is_ops_build"] is False

    def test_the_ops_record_supplies_time_and_size_despite_the_stamps(self):
        """Storage was not swept for mongot, so the stamped `InnoDB` does
        not stop the match; the time is the ops time to a queryable index,
        load included (14.3 + 100.1 s), never the bare `CREATE INDEX`."""
        rows = {r["build_label"]: r for r in self._mongot()["frontier_builds"]}
        scalar = rows["quantization=scalar"]
        assert scalar["time_to_queryable_s"] == pytest.approx(114.486)
        assert scalar["time_source"] == "ops"
        assert scalar["index_bytes"] == 16238987031
        assert scalar["index_bytes_source"] == "ops"
        assert rows["quantization=none"]["time_to_queryable_s"] is None
        table = frontier_render.build_cost_table({"mongodb": self._mongot()})
        assert "114 s (ops)" in table

    def test_the_ops_build_notes_do_not_say_no_point_was_measured(self):
        page = render._headline_tables(
            generate.summarize(_archived_mongot(), {}))
        assert "no recall point was measured" not in page

    def test_an_incremental_ops_time_is_not_doubled(self):
        """MariaDB built on INSERT: the ops time is `build_wall_s` alone."""
        points = [_recall("mariadb", 0.95, 400.0, ef_search=10, m=16),
                  _recall("mariadb", 0.93, 500.0, ef_search=10, m=6)]
        points = [{**p, "build_mode": None, "storage_engine": "InnoDB"}
                  for p in points]
        ops = _ops_index_build("mariadb", build_wall_s=13041.322,
                               ingest_wall_s=13041.322,
                               extra={"separable_build": False})
        entry = _entry(generate.summarize(points + [ops], {}), "mariadb")
        rows = {r["build_label"]: r for r in entry["frontier_builds"]}
        assert rows["M=16"]["time_to_queryable_s"] == pytest.approx(13041.322)
        assert rows["M=6"]["time_to_queryable_s"] is None

    def test_valkey_at_the_server_default_is_cannot_be_said(self):
        """`{m: 16, ef_construction: None}`: the ops pass left efC to the
        server and nothing records what that was. Honest: no mark, a note
        saying so. Mutant (None compared as a value): every floor marked
        and "efC=server default" claimed as a different build."""
        ops = _ops_index_build("valkey",
                               extra={"ef_construction_tunable": True})
        summary = generate.summarize(_valkey_records() + [ops], {})
        entry = _entry(summary, "valkey")
        for floor in (90, 95, 99):
            assert entry[f"qps_at_recall_{floor}_is_ops_build"] is None
        assert all(r["is_ops_build"] is None
                   for r in entry["frontier_builds"])
        page = render._headline_tables(summary)
        table = _headline(page)
        assert frontier_render.OPS_MISMATCH_MARK not in table
        assert "server default" in table and "cannot be said" in table
        # The ops-build view says why it is empty without claiming nothing
        # was measured.
        assert "no recall point was measured" not in page

    def test_a_floor_only_one_build_reaches_is_not_called_ranked(self):
        entry = _entry(_summary(_with_cost(_pgvector_records(), COSTS)),
                       "pgvector")
        assert entry["smallest_qps_at_recall_99_ranked_by"] == frontier.ONLY

    def test_a_declared_build_that_matches_nothing_is_cannot_be_said(self):
        manifest = _manifest({"pgvector": {"m": 16, "ef_construction": 128}})
        entry = _entry(_summary(_pgvector_records(), manifest), "pgvector")
        assert entry["qps_at_recall_95_is_ops_build"] is None


class TestEqualSizesAreATie:
    """The 25 Aug run's pgvector efC 64/200/400 all read 8,110,088,192 bytes;
    AliSQL M=6 and M=16 both 4,156,555,264. Size decided nothing there."""

    def _entry(self):
        records = _with_cost(_pgvector_records(),
                             {(16, 200, None): {"index_bytes": 8110088192},
                              (16, 400, None): {"index_bytes": 8110088192}})
        return _entry(_summary(records), "pgvector")

    def test_the_rule_says_sizes_were_equal(self):
        entry = self._entry()
        assert entry["smallest_qps_at_recall_95_ranked_by"] == \
            frontier.BY_SIZE_THEN_M
        table = frontier_render.smallest_table({"pgvector": entry})
        assert "(=size)" in table
        row = _row(table, render._label("pgvector"))
        assert "(=size)" in row[2] and "(=size)" not in row[3]

    def test_the_cost_table_flags_the_shared_size(self):
        table = frontier_render.build_cost_table({"pgvector": self._entry()})
        assert "7.6 GiB =" in table and "one reading per run" in table


class TestAnExactQpsTieIsDeterministic:

    def test_the_smaller_build_wins_a_tie(self):
        """Two builds at exactly the same QPS above the floor: the smaller
        is printed, whichever order the result files were walked in."""
        a = _recall("mariadb", 0.95, 400.0, ef_search=10, m=32)
        b = _recall("mariadb", 0.95, 400.0, ef_search=10, m=8)
        for order in ([a, b], [b, a]):
            entry = _entry(generate.summarize(order, {}), "mariadb")
            assert entry["qps_at_recall_90_build_label"] == "M=8"


def test_nothing_added_imports_the_orchestrator():
    """report/ runs in a bench image that mounts report/ and harness/ only."""
    for name in ("frontier.py", "frontier_render.py"):
        source = open(os.path.join(VB_ROOT, "report", name)).read()
        assert "orchestrator" not in "\n".join(
            l for l in source.splitlines()
            if l.lstrip().startswith(("import ", "from ")))
