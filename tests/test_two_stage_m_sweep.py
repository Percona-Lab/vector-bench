"""The two-stage frontier: explore every build knob at 90k, select, re-measure.

A reviewer of full-20260825 asked for the frontier over M = 4..48 rather than
M=16 alone. `config/profiles/frontier-explore.yml` measures that grid at
90,000 rows; `select-builds` (`orchestrator/build_selection.py`) keeps the
builds at or near the recall floors plus their M neighbours and writes an
overlay that `run --profile tuned-complete --overlay` measures at 990,000.

What these tests hold, because each is a way the instrument could quietly
measure something other than what it says:

* a value an engine cannot take is refused, and a combination it cannot build
  is dropped by name, never clamped (`orchestrator/ann_knobs.py`);
* the stage-1 grid is the one the profile header describes, per engine;
* the estimate sees M=48 and every build the grid renders;
* the selection rule keeps what its docstring says -- winners, near points,
  both neighbours, edges, the reference build, the closure -- and nothing
  else, and its frontier is the report's;
* the overlay merges exactly as a profile's `resources:` block does and may
  touch nothing but build axes;
* the stage-2 manifest records the provenance, and a resume cannot swap it.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from types import SimpleNamespace

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import memory_fit  # noqa: E402
from orchestrator import ann_knobs, build_selection as sel  # noqa: E402
from orchestrator import cli as cli_mod  # noqa: E402
from orchestrator import profile_overlay, select_builds  # noqa: E402
from orchestrator.ann_config import result_definitions  # noqa: E402
from orchestrator.ann_pass import (ann_m_values, excluded_ann_builds,  # noqa: E402
                                   render_config)
from orchestrator.config import (available_profiles, load_profile,  # noqa: E402
                                 load_resources, merge_resource_overrides)
from orchestrator.estimate_grid import (ann_grid_estimate,  # noqa: E402
                                        m_cost_factor)
from orchestrator.manifest import Manifest, SelectionChanged  # noqa: E402
from orchestrator.reference_build import reference_build_problem  # noqa: E402

EXPLORE = "frontier-explore"
REVIEWER_GRID = [4, 6, 8, 12, 16, 32, 48]
EFC = [64, 200, 400]
M_ENGINES = ("mariadb", "mariadb123", "alisql", "pgvector", "valkey",
             "percona", "myvector", "villagesql")
DATASET = "dbpedia-openai-100k-angular"


def _resources(profile, resource_pass="tuned"):
    return merge_resource_overrides(load_resources(resource_pass), profile)


def _builds(engine, profile, resource_pass="tuned"):
    body = render_config(engine, profile, _resources(profile, resource_pass),
                         resource_pass)
    return {json.dumps(sel.build_of_grid(args[-1]), sort_keys=True)
            for _name, args, _q in result_definitions(body)}


# --- engine limits ------------------------------------------------------------

class TestEngineLimitsAreRefusedNotClamped:

    @pytest.mark.parametrize("engine,m", [("mariadb", 2), ("mariadb123", 201),
                                          ("alisql", 2), ("pgvector", 101),
                                          ("percona", 1), ("myvector", 1)])
    def test_an_m_outside_the_engines_range_refuses_the_render(self, engine, m):
        profile = load_profile("tuned-complete")
        profile["ann"]["m_values"] = [m]
        resources = _resources(profile, "normalized")
        with pytest.raises(ValueError, match=rf"M={m} is outside"):
            render_config(engine, profile, resources, "normalized")

    @pytest.mark.parametrize("engine,efc", [("pgvector", 1001),
                                            ("pgvector", 3),
                                            ("percona", 4097)])
    def test_an_ef_construction_outside_the_range_refuses(self, engine, efc):
        profile = load_profile("tuned-complete")
        resources = _resources(profile)
        resources["extras"][f"{engine}_ef_construction"] = [200, efc]
        with pytest.raises(ValueError, match=rf"efConstruction={efc}"):
            render_config(engine, profile, resources, "tuned")

    def test_the_run_refuses_before_loading_anything(self):
        """`reference_build_problem` is what `cmd_run` calls before the first
        unit, and it must surface the refusal rather than swallow it."""
        profile = load_profile("tuned-complete")
        profile["ann"]["m_values"] = [2, 16]
        problem = reference_build_problem("mariadb", profile,
                                          _resources(profile, "normalized"),
                                          "normalized")
        assert problem and "M=2 is outside" in problem

    def test_pgvector_m48_at_efc64_is_dropped_by_name_and_nothing_else(self):
        builds = _builds("pgvector", load_profile(EXPLORE))
        assert '{"M": 48, "efConstruction": 64}' not in builds
        assert '{"M": 48, "efConstruction": 200}' in builds
        assert '{"M": 32, "efConstruction": 64}' in builds    # 64 >= 2*32
        assert len(builds) == 20
        dropped = excluded_ann_builds("pgvector", load_profile(EXPLORE),
                                      _resources(load_profile(EXPLORE)),
                                      "tuned")
        assert [(m, e) for m, e, _ in dropped] == [(48, 64)]
        assert "2 * m" in dropped[0][2]

    @pytest.mark.parametrize("ms,efcs,match", [([48], [64, 200],
                                                "ef_construction=64"),
                                               ([48, 16], [64], "M=48")])
    def test_a_value_buildable_nowhere_is_refused_not_dropped(self, ms, efcs,
                                                              match):
        """Dropping a whole ef_construction (or M) leaves no combination to
        name it, so it is refused instead."""
        with pytest.raises(ValueError, match=match):
            ann_knobs.efc_run_groups("post_build", {
                "M": ms, "efConstruction": efcs}, "pgvector", [[10]])

    def test_hnswlib_engines_drop_an_ef_construction_below_m(self):
        """hnswlib raises it to M silently; the label would lie."""
        assert ann_knobs.excluded_builds("myvector", [100], [64])[0][:2] == (
            100, 64)
        assert ann_knobs.excluded_builds("valkey", [48], [64]) == []

    def test_a_grid_with_no_hole_renders_exactly_as_before(self):
        """Every shipped profile but the explorer renders one run group per
        build mode, as it did before the split existed, so nothing it ever
        wrote is renamed and no reader of run-group names changes."""
        for name in available_profiles():
            if name == EXPLORE:
                continue
            profile = load_profile(name)
            for resource_pass in ("normalized", "tuned"):
                resources = _resources(profile, resource_pass)
                for engine in cli_mod.KNOWN_ENGINES:
                    groups = render_config(engine, profile, resources,
                                           resource_pass)["float"]["any"][0][
                        "run_groups"]
                    assert not any("_efc" in g for g in groups), (
                        name, resource_pass, engine, sorted(groups))

    def test_the_percona_bounds_match_the_overlay_module(self):
        """Repeated in ann_knobs because the overlay is not importable on the
        host; this is what stops the two copies drifting."""
        path = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms", "percona",
                            "module.py")
        source = open(path).read()
        found = {name: int(re.search(rf"^{name} = (\d+)$", source,
                                     re.M).group(1))
                 for name in ("MIN_M", "MAX_M", "MIN_EF_CONSTRUCTION",
                              "MAX_EF_CONSTRUCTION")}
        bounds = ann_knobs.KNOB_BOUNDS["percona"]
        assert bounds["M"] == (found["MIN_M"], found["MAX_M"])
        assert bounds["efConstruction"] == (found["MIN_EF_CONSTRUCTION"],
                                            found["MAX_EF_CONSTRUCTION"])

    @pytest.mark.parametrize("relpath,pattern,expected", [
        ("sources/pgvector-v0.8.6/src/hnsw.h",
         r"#define HNSW_MIN_M\s+(\d+)", 2),
        ("sources/pgvector-v0.8.6/src/hnsw.h",
         r"#define HNSW_MAX_M\s+(\d+)", 100),
        ("sources/mariadb-mariadb-11.8.8/sql/vector_mhnsw.cc",
         r"THDVAR_UINT\(default_m,[^;]*?nullptr, nullptr, \d+, (\d+), \d+", 3),
        ("sources/mariadb-mariadb-11.8.8/sql/vector_mhnsw.cc",
         r"THDVAR_UINT\(default_m,[^;]*?nullptr, nullptr, \d+, \d+, (\d+)", 200),
    ])
    def test_the_bounds_match_the_pinned_source_when_it_is_exported(
            self, relpath, pattern, expected):
        """`sources/` is generated and gitignored, so this runs only on a
        machine that has exported the pins -- which is every machine that
        builds the images these bounds describe."""
        path = os.path.join(VB_ROOT, relpath)
        if not os.path.exists(path):
            pytest.skip(f"{relpath} not exported")
        match = re.search(pattern, open(path).read(), re.S)
        assert match and int(match.group(1)) == expected


# --- the stage-1 profile --------------------------------------------------------

class TestTheExploreProfile:

    def test_every_engine_with_a_degree_knob_sweeps_the_reviewers_grid(self):
        profile = load_profile(EXPLORE)
        for engine in M_ENGINES:
            assert ann_m_values(engine, profile, _resources(profile),
                                "tuned") == REVIEWER_GRID, engine

    def test_the_tuned_m_pin_is_cleared_not_inherited(self):
        """null, not {} -- the deep-merge invariant that once halved m-sweep."""
        resources = _resources(load_profile(EXPLORE))
        assert resources["extras"]["m_values_by_engine"] is None

    @pytest.mark.parametrize("engine,count", [
        ("mariadb", 7), ("mariadb123", 7), ("alisql", 7), ("pgvector", 20),
        ("valkey", 21), ("percona", 21), ("myvector", 21),
        # 21 less vsql_vector's six out-of-range combinations
        # (ef_construction > 32 x M at M=4, 6, 8, 12).
        ("villagesql", 15), ("mongodb", 2)])
    def test_build_count_per_engine_is_the_headers(self, engine, count):
        assert len(_builds(engine, load_profile(EXPLORE))) == count

    def test_ef_construction_engines_sweep_all_three(self):
        extras = _resources(load_profile(EXPLORE))["extras"]
        for engine in ("pgvector", "valkey", "percona", "myvector",
                       "villagesql"):
            assert extras[f"{engine}_ef_construction"] == EFC

    def test_percona_search_sweeps_both_quantizations_at_one_m_label(self):
        profile = load_profile(EXPLORE)
        assert _builds("mongodb", profile) == {
            '{"M": 16, "quantization": "scalar"}',
            '{"M": 16, "quantization": "none"}'}

    def test_publishable_with_two_runs(self):
        profile = load_profile(EXPLORE)
        assert profile["publishable"] is True and profile["runs"] >= 2

    @pytest.mark.parametrize("engine", cli_mod.KNOWN_ENGINES)
    def test_every_reference_build_is_in_the_grid(self, engine):
        profile = load_profile(EXPLORE)
        assert reference_build_problem(engine, profile, _resources(profile),
                                       "tuned") is None

    def test_m_sweep_is_left_as_the_one_axis_slice(self):
        """The explorer is a new profile; m-sweep still pins every other knob
        so the 2026-08-28 results it is cited for stay reproducible."""
        extras = _resources(load_profile("m-sweep"))["extras"]
        assert extras["pgvector_ef_construction"] == [200]
        assert load_profile("m-sweep")["ann"]["m_values"] == [6, 8, 16, 24, 32]


# --- the estimate and the memory check at M=48 ---------------------------------

class TestTheEstimateSeesTheGrid:

    def _grid(self):
        profile = load_profile(EXPLORE)
        return ann_grid_estimate(profile, cli_mod.KNOWN_ENGINES,
                                 profile["datasets"], ["tuned"], ["ann"])

    def test_every_engine_is_priced_with_its_own_build_count(self):
        rows = self._grid()["engines"]
        assert rows["percona"]["builds"] == 21
        assert rows["pgvector"]["builds"] == 20
        assert rows["mariadb"]["builds"] == 7
        assert rows["mongodb"]["builds"] == 2
        assert all(rows[e]["largest_m"] == 48 for e in M_ENGINES)

    def test_m48_costs_more_than_m16_where_the_load_builds_the_graph(self):
        assert m_cost_factor("mariadb", 48) == pytest.approx(3 ** 1.1)
        assert m_cost_factor("alisql", 4) == pytest.approx(0.25 ** 1.1)
        assert m_cost_factor("pgvector", 48) == 1.0

    def test_the_grid_costs_more_than_the_published_line_charges(self):
        grid = self._grid()
        assert grid["extra_hours"] > 0
        rows = grid["engines"]
        # 21 builds against the line's 7 reloads, each priced by M.
        assert rows["percona"]["hours"] > 2.5 * rows["percona"]["charged_hours"]
        assert rows["myvector"]["hours"] is None      # no rate: named, not 0

    def _printed(self, capsys, engines=("percona", "mariadb", "myvector")):
        from orchestrator.estimate import (estimate_load_hours,
                                           print_load_estimate)
        from orchestrator.estimate_grid import ann_grid_prints, print_ann_grid
        profile = load_profile(EXPLORE)
        args = (profile, list(engines), profile["datasets"], ["tuned"],
                ["ann"])
        prints = ann_grid_prints(*args)
        print_load_estimate(*args, axis_hint=not prints)
        print_ann_grid(*args, estimate_load_hours(*args))
        return capsys.readouterr().out

    def test_the_block_names_the_axis_that_multiplies(self, capsys):
        out = self._printed(capsys)
        assert "21 build(s) = M 7 x efConstruction 3" in out
        assert "Reduce\n    ann.m_values" not in out

    def test_the_headline_is_corrected_where_it_is_low(self, capsys):
        from orchestrator.estimate import estimate_load_hours
        from orchestrator.estimate_grid import ann_grid_estimate
        out = self._printed(capsys)
        profile = load_profile(EXPLORE)
        engines = ["percona", "mariadb", "myvector"]
        head = estimate_load_hours(profile, engines, profile["datasets"],
                                   ["tuned"], ["ann"])
        extra = ann_grid_estimate(profile, engines, profile["datasets"],
                                  ["tuned"], ["ann"])["extra_hours"]
        assert f"low by at least ~{extra:.1f} h" in out
        assert (f"plan against ~{head['typical_total_hours'] + extra:.1f} h "
                f"typical") in out

    def test_what_is_still_unpriced_is_named(self, capsys):
        out = self._printed(capsys)
        assert "bulk index builds of myvector" in out
        assert "recall/QPS sweep" in out
        assert "any ann load at all for myvector" in out

    def test_the_published_block_is_unchanged_by_default(self, capsys):
        """docs/07's sample block is `print_load_estimate` with its default
        arguments; the advice is only withheld when cmd_run says the
        per-build block follows."""
        from orchestrator.estimate import print_load_estimate
        profile = load_profile("main")
        print_load_estimate(profile, ["mariadb"], profile["datasets"],
                            ["normalized"], ["ann", "ops"])
        assert "Reduce\n    ann.m_values" in capsys.readouterr().out

    def test_cmd_run_prints_it(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        run = source[source.index("def cmd_run"):]
        assert run.index("print_ann_grid(") < run.index("_check_free_disk(")
        assert "axis_hint=not grid_prints" in run

    def test_memory_refusal_checks_m48_for_myvector(self):
        profile = load_profile(EXPLORE)
        resolved = SimpleNamespace(buffer_bytes=0, server_memory_bytes=10**8)
        refusal = cli_mod._memory_refusal(
            "myvector", {"capabilities": {"memory_fit": "myvector"}},
            DATASET, "ann", profile, _resources(profile), resolved, "tuned")
        assert refusal and refusal["m"] == 48
        assert "largest of [4, 6, 8, 12, 16, 32, 48]" in refusal["reason"]
        assert memory_fit.element_bytes(1536, 48) == (
            4 * 1536 + 2 * 48 * 4 + 4 + 8 + memory_fit.PER_ELEMENT_OVERHEAD)


# --- the selection rule -----------------------------------------------------------

def _point(build, ef, recall, qps, engine="mariadb"):
    return {"engine": engine, "build": build, "ef_search": ef,
            "recall": recall, "qps": qps}


def _kept(result):
    return {json.dumps(k["build"], sort_keys=True): k["reasons"]
            for k in result["kept"]}


def _mgrid(ms=REVIEWER_GRID):
    return [{"M": m} for m in ms]


def _measured(grid, engine="mariadb"):
    """One dominated, far-from-every-floor point per build: measured at
    stage 1 (so it can be kept as a neighbour) and chosen by nothing."""
    return [_point(b, 10, 0.5, 1.0, engine) for b in grid]


class TestTheSelectionRule:

    def test_winner_is_the_fastest_point_at_or_above_each_floor(self):
        points = [_point({"M": 8}, 40, 0.905, 900.0),
                  _point({"M": 16}, 40, 0.96, 700.0),
                  _point({"M": 32}, 40, 0.995, 300.0),
                  _point({"M": 32}, 10, 0.80, 1000.0)]
        kept = _kept(sel.select_engine("mariadb", points, _mgrid(), None,
                                       near=0.0))
        assert any("winner at floor 0.90" in r for r in kept['{"M": 8}'])
        assert any("winner at floor 0.95" in r for r in kept['{"M": 16}'])
        assert any("winner at floor 0.99" in r for r in kept['{"M": 32}'])

    def test_near_is_plus_or_minus_0015_of_a_floor_and_no_wider(self):
        assert sel.NEAR_RECALL == 0.015
        points = [_point({"M": 4}, 10, 0.70, 5000.0),     # far: frontier only
                  _point({"M": 8}, 10, 0.886, 3000.0),    # 0.014 under 0.90
                  _point({"M": 12}, 10, 0.884, 2900.0),   # beaten by M=8
                  _point({"M": 32}, 10, 0.999, 100.0)]
        result = sel.select_engine("mariadb", points, _mgrid([4, 8, 32]),
                                   None)
        kept = _kept(result)
        assert any("near floor 0.90" in r for r in kept['{"M": 8}'])
        # M=4's frontier point at 0.70 is near no floor and not a winner.
        assert not any("near" in r or "winner" in r
                       for r in kept.get('{"M": 4}', []))

    def test_a_frontier_point_0016_from_every_floor_is_not_near(self):
        points = [_point({"M": 6}, 10, 0.884, 3000.0),
                  _point({"M": 16}, 40, 0.999, 100.0)]
        kept = _kept(sel.select_engine("mariadb", points, _mgrid([6, 16]),
                                       None))
        assert not any("near floor" in r for r in kept.get('{"M": 6}', []))

    def test_an_interior_kept_m_brings_both_neighbours(self):
        points = [_point({"M": 12}, 40, 0.95, 500.0)] + _measured(_mgrid())
        kept = _kept(sel.select_engine("mariadb", points, _mgrid(), None))
        assert set(kept) == {'{"M": 8}', '{"M": 12}', '{"M": 16}'}
        assert kept['{"M": 8}'] == ['M neighbour of kept build {"M": 12}']

    @pytest.mark.parametrize("edge,neighbour,side", [(4, 6, "below"),
                                                     (48, 32, "above")])
    def test_an_edge_m_has_one_neighbour_and_says_so(self, edge, neighbour,
                                                     side):
        points = [_point({"M": edge}, 40, 0.95, 500.0)] + _measured(_mgrid())
        result = sel.select_engine("mariadb", points, _mgrid(), None)
        assert set(_kept(result)) == {f'{{"M": {edge}}}',
                                      f'{{"M": {neighbour}}}'}
        assert len(result["edges"]) == 1
        assert f"M={edge}" in result["edges"][0] and side in result["edges"][0]
        assert "may continue" in result["edges"][0]

    def test_an_illegal_next_m_is_named_as_illegal_not_unexplored(self):
        """pgvector at ef_construction 64 has no M=48 (2*m rule): M=32 at 64
        has no upper neighbour because M=48 cannot be built there, not
        because nobody explored it."""
        grid = [{"M": m, "efConstruction": e} for m in REVIEWER_GRID
                for e in EFC if not e < 2 * m]
        points = ([_point({"M": 32, "efConstruction": 64}, 40, 0.95, 500.0,
                          "pgvector")] + _measured(grid, "pgvector"))
        result = sel.select_engine("pgvector", points, grid, None)
        assert '{"M": 16, "efConstruction": 64}' in set(_kept(result))
        edge = result["edges"][0]
        assert "M=48 is in the grid but cannot be built" in edge
        assert "2 * m" in edge and "largest" not in edge

    def test_a_neighbour_with_no_stage_1_points_is_dropped_with_reason(self):
        points = [_point({"M": 12}, 40, 0.95, 500.0)] + _measured(
            _mgrid([12, 16]))
        result = sel.select_engine("mariadb", points, _mgrid(), None)
        assert '{"M": 8}' not in _kept(result)
        pruned = {json.dumps(p["build"]): p["reason"]
                  for p in result["pruned"]}
        assert pruned['{"M": 8}'].startswith(
            'M neighbour of kept build {"M": 12}, dropped: no stage-1 point')

    def test_an_unreached_floor_keeps_the_highest_recall_and_records_it(self):
        points = [_point({"M": 6}, 800, 0.93, 50.0),
                  _point({"M": 4}, 800, 0.88, 80.0)]
        result = sel.select_engine("mariadb", points, _mgrid([4, 6, 8]), None)
        assert result["floors_unreached"] == [0.95, 0.99]
        assert any("floor 0.99 unreached" in r
                   for r in _kept(result)['{"M": 6}'])

    def test_the_reference_build_is_always_kept(self):
        points = [_point({"M": 4}, 40, 0.95, 500.0)]
        result = sel.select_engine("mariadb", points, _mgrid(), {"M": 32})
        assert sel.REASON_REFERENCE in _kept(result)['{"M": 32}']

    def test_the_kept_builds_are_exact_and_no_product_is_added(self):
        """Lists alone multiply: (4, 64) and (48, 400) would bring (4, 400)
        and (48, 64) too. The exact list is what reaches the renderer."""
        grid = [{"M": m, "efConstruction": e} for m in (4, 16, 48)
                for e in EFC]
        points = [_point({"M": 4, "efConstruction": 64}, 40, 0.95, 500.0,
                         "valkey"),
                  _point({"M": 48, "efConstruction": 400}, 40, 0.99, 400.0,
                         "valkey")] + _measured(grid, "valkey")
        result = sel.select_engine("valkey", points, grid, None)
        kept = set(_kept(result))
        assert '{"M": 48, "efConstruction": 64}' not in kept
        assert '{"M": 4, "efConstruction": 400}' not in kept
        assert {json.dumps(b, sort_keys=True) for b in result["builds"]} == kept

    def test_the_renderer_builds_exactly_the_named_builds(self):
        profile = load_profile("tuned-complete")
        resources = _resources(profile)
        resources["extras"]["m_values_by_engine"] = {"pgvector": [4, 16, 48]}
        resources["extras"]["pgvector_ef_construction"] = [64, 200, 400]
        resources["extras"]["ann_builds_by_engine"] = {"pgvector": [
            {"M": 4, "efConstruction": 64}, {"M": 16, "efConstruction": 200},
            {"M": 48, "efConstruction": 400}]}
        body = render_config("pgvector", profile, resources, "tuned")
        builds = {json.dumps(sel.build_of_grid(a[-1]), sort_keys=True)
                  for _n, a, _q in result_definitions(body)}
        assert builds == {'{"M": 4, "efConstruction": 64}',
                          '{"M": 16, "efConstruction": 200}',
                          '{"M": 48, "efConstruction": 400}'}

    def test_the_renderer_refuses_a_named_build_outside_its_grid(self):
        profile = load_profile("tuned-complete")
        resources = _resources(profile)
        resources["extras"]["ann_builds_by_engine"] = {"pgvector": [
            {"M": 32, "efConstruction": 200}]}
        with pytest.raises(ValueError, match="does not contain"):
            render_config("pgvector", profile, resources, "tuned")

    def test_a_named_build_keeps_its_result_file_name(self):
        """Restricting must not rename a file: stage 2 reuses whatever the
        shared tree already holds for a build tuned-complete measured."""
        profile = load_profile("tuned-complete")
        resources = _resources(profile)
        full = {(n, json.dumps(a, sort_keys=True), json.dumps(q))
                for n, a, q in result_definitions(render_config(
                    "pgvector", profile, resources, "tuned"))}
        resources["extras"]["ann_builds_by_engine"] = {"pgvector": [
            {"M": 16, "efConstruction": 200}]}
        cut = {(n, json.dumps(a, sort_keys=True), json.dumps(q))
               for n, a, q in result_definitions(render_config(
                   "pgvector", profile, resources, "tuned"))}
        assert cut and cut < full

    def test_everything_else_is_pruned_with_a_reason(self):
        points = [_point({"M": 16}, 40, 0.95, 500.0),
                  _point({"M": 48}, 40, 0.93, 10.0)]   # dominated by M=16
        result = sel.select_engine("mariadb", points, _mgrid(), None)
        pruned = {json.dumps(p["build"]): p["reason"] for p in result["pruned"]}
        assert "no frontier point" in pruned['{"M": 48}']     # measured
        assert pruned['{"M": 4}'] == sel.REASON_NO_POINTS     # never measured

    def test_mongot_m_is_a_label_not_an_axis(self):
        points = [_point({"M": 16, "quantization": "scalar"}, 40, 0.89, 900.0,
                         "mongodb"),
                  _point({"M": 16, "quantization": "none"}, 40, 0.95, 500.0,
                         "mongodb")]
        grid = [{"M": 16, "quantization": q} for q in ("scalar", "none")]
        result = sel.select_engine("mongodb", points, grid,
                                   {"quantization": "scalar"})
        assert result["lists"] == {"quantization": ["scalar", "none"]}
        assert set(_kept(result)) == {'{"quantization": "none"}',
                                      '{"quantization": "scalar"}'}

    def test_the_frontier_is_the_reports(self):
        pytest.importorskip("numpy")
        from report.loaders import pareto_frontier
        points = [_point({"M": m}, ef, r, q)
                  for m, ef, r, q in [(4, 10, 0.5, 900), (4, 40, 0.7, 600),
                                      (8, 10, 0.6, 1000), (8, 40, 0.8, 300),
                                      (16, 10, 0.8, 400), (16, 40, 0.95, 200),
                                      (32, 40, 0.95, 250), (32, 80, 0.99, 50)]]
        theirs = pareto_frontier([{"recall_at_k": p["recall"], "qps": p["qps"],
                                   "id": i} for i, p in enumerate(points)])
        ours = sel.frontier([{**p, "id": i} for i, p in enumerate(points)])
        assert [p["id"] for p in ours] == [p["id"] for p in theirs]

    def test_a_record_build_falls_back_to_its_file_name(self):
        record = {"m": None, "extra": {"source_file":
                  "tuned/x/10/percona/angular_M_8_efConstruction_400_engine_InnoDB_40.hdf5"}}
        assert sel.build_of_record(record) == {"M": 8, "efConstruction": 400}


# --- stage 1 on disk, the overlay, the manifest -----------------------------------

def _stage1(tmp_path, scans=False):
    """A synthetic frontier-explore run directory: manifest + records.jsonl."""
    run_dir = tmp_path / "frontier-explore-20261101-000000"
    (run_dir / "report").mkdir(parents=True)
    profile = load_profile(EXPLORE)
    (run_dir / "run-manifest.json").write_text(json.dumps({
        "run_id": run_dir.name,
        "config": {"profile": profile, "ann_fingerprint": "f00dfeed"}}))
    records = []

    def add(engine, m, efc=None, q=None):
        for ef in (10, 20, 40, 80, 120, 200, 400, 800):
            cap = 1 - 0.5 / m - (0.05 if efc == 64 else 0) - (
                0.03 if q == "scalar" else 0)
            recall = cap * (1 - math.exp(-ef / (m + 6.0)))
            qps = 1e4 / ef ** 0.8 / m ** 0.5 / ((efc or 200) / 200) ** 0.1
            records.append({
                "phase": "recall_qps", "engine": engine, "dataset": DATASET,
                "resource_pass": "tuned", "m": m, "ef_construction": efc,
                "quantization": q, "ef_search": ef,
                "recall_at_k": round(recall, 5), "qps": qps,
                "vector_index_used": True})

    for m in REVIEWER_GRID:
        add("mariadb", m)
        for efc in EFC:
            if efc >= 2 * m:
                add("pgvector", m, efc)
    add("mongodb", 16, q="scalar")
    add("mongodb", 16, q="none")
    if scans:
        records.append({**records[0], "ef_search": 9999, "recall_at_k": 1.0,
                        "qps": 1e9, "vector_index_used": False})
    (run_dir / "report" / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    return str(run_dir)


class TestSelectBuildsEndToEnd:

    def test_a_scan_never_reaches_the_frontier(self, tmp_path):
        points, notes = sel.load_points(_stage1(tmp_path, scans=True))
        assert notes["points_dropped_scan"] == 1
        assert not any(p["qps"] == 1e9 for p in points)

    def test_missing_records_names_the_command_that_writes_them(self,
                                                                tmp_path):
        with pytest.raises(FileNotFoundError, match="run-benchmark.sh report"):
            sel.load_points(str(tmp_path))

    def test_an_engine_stage_1_left_without_points_is_named(self, tmp_path):
        run_dir = _stage1(tmp_path)
        path = os.path.join(run_dir, "run-manifest.json")
        manifest = json.load(open(path))
        manifest["phases"] = [{"phase": "ann", "engine": "alisql",
                               "status": "failed"}]
        json.dump(manifest, open(path, "w"))
        block = select_builds.select(run_dir)["build_selection"]
        assert "alisql" in block["unselected"]
        assert "tuned-complete's own grid" in block["unselected"]["alisql"]

    def test_the_overlay_renders_exactly_the_kept_builds(self, tmp_path):
        document = select_builds.select(_stage1(tmp_path))
        assert select_builds.check(document) == []

    def test_provenance_carries_everything_the_brief_asks_for(self, tmp_path):
        run_dir = _stage1(tmp_path)
        block = select_builds.select(run_dir)["build_selection"]
        assert block["stage1"]["run_id"] == os.path.basename(run_dir)
        assert block["stage1"]["run_dir"] == os.path.abspath(run_dir)
        assert block["stage1"]["dataset"] == DATASET
        assert len(block["grid_explored"]["pgvector"]) == 20
        assert block["stage1_excluded"]["pgvector"][0]["build"] == {
            "M": 48, "efConstruction": 64}
        for engine in ("mariadb", "pgvector", "mongodb"):
            explored = len(block["grid_explored"][engine])
            kept = {json.dumps(k["build"], sort_keys=True)
                    for k in block["kept"][engine]}
            pruned = {json.dumps(p["build"], sort_keys=True)
                      for p in block["pruned"][engine]}
            assert not kept & pruned
            assert len(kept | pruned) >= explored
            assert all(k["reasons"] for k in block["kept"][engine])
            assert all(p["reason"] for p in block["pruned"][engine])
        transfer = block["transfer_check"]
        assert transfer["tolerance"] == 0.02
        row = transfer["stage1_points"]["mariadb"][0]
        assert set(row) == {"build", "ef_search", "recall", "qps"}

    def test_the_cli_writes_an_overlay_run_can_apply(self, tmp_path, capsys):
        out = str(tmp_path / "stage2.yml")
        rc = cli_mod.main(["select-builds", "--from", _stage1(tmp_path),
                           "--out", out])
        assert rc == 0
        overlay = profile_overlay.load_overlay(out)
        merged, selection = profile_overlay.apply_overlay(
            load_profile("tuned-complete"), overlay)
        assert selection["stage1"]["profile"] == EXPLORE
        assert "--overlay" in capsys.readouterr().out
        assert "GENERATED" in open(out).read().splitlines()[0]


class TestTheOverlayMerges:

    def _overlay(self, extras):
        selection = {"stage1": {"run_id": "s1"}, "kept": {}}
        selection["overlay_sha256"] = profile_overlay.overlay_digest(
            extras, selection)
        return {"overlay_of": "tuned-complete",
                "resources": {"extras": extras},
                "build_selection": selection}

    def test_lists_edited_apart_from_their_provenance_are_refused(self):
        overlay = self._overlay({"pgvector_ef_construction": [200]})
        overlay["resources"]["extras"]["pgvector_ef_construction"] = [64]
        with pytest.raises(ValueError, match="digest"):
            profile_overlay.apply_overlay(load_profile("tuned-complete"),
                                          overlay)

    def test_provenance_edited_apart_from_its_grid_is_refused(self):
        overlay = self._overlay({"pgvector_ef_construction": [200]})
        overlay["build_selection"]["kept"] = {"pgvector": [{"build": {}}]}
        with pytest.raises(ValueError, match="digest"):
            profile_overlay.apply_overlay(load_profile("tuned-complete"),
                                          overlay)

    def test_an_overlay_without_provenance_is_refused(self):
        overlay = {"overlay_of": "tuned-complete",
                   "resources": {"extras": {"pgvector_ef_construction": [200]}}}
        with pytest.raises(ValueError, match="no build_selection"):
            profile_overlay.apply_overlay(load_profile("tuned-complete"),
                                          overlay)

    @pytest.mark.parametrize("extras,match", [
        ({"pgvector_ef_construction": 200}, "must"),
        ({"m_values_by_engine": {"mariadb": 16}}, "must"),
        ({"mongodb_quantization": [1]}, "must"),
        ({"mongodb_quantization": ["int8"]}, "must"),
        ({"valkey_ef_construction": []}, "must"),
        ({"m_values_by_engine": {"mariadb": [2]}}, "M=2 is outside"),
        ({"m_values_by_engine": {"maraidb": [16]}}, "unknown engine"),
        ({"pgvector_ef_construction": [2000]}, "efConstruction=2000"),
        ({"ann_builds_by_engine": {"pgvector": [{"ef": 64}]}}, "naming only"),
    ])
    def test_a_bad_value_is_refused_at_load_not_a_traceback(self, extras,
                                                            match):
        with pytest.raises(ValueError, match=match):
            profile_overlay.apply_overlay(load_profile("tuned-complete"),
                                          self._overlay(extras))

    def test_the_quantization_enumeration_is_the_modules(self):
        path = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms", "mongodb",
                            "module.py")
        found = re.search(r"if self\._quantization not in \(([^)]*)\)",
                          open(path).read())
        assert found
        assert tuple(re.findall(r'"(\w+)"', found.group(1))) == \
            ann_knobs.QUANTIZATIONS

    def test_render_refuses_an_overlay_on_the_normalized_pass(self, tmp_path,
                                                              capsys):
        path = tmp_path / "o.yml"
        path.write_text(yaml.safe_dump(self._overlay(
            {"pgvector_ef_construction": [200]})))
        rc = cli_mod.main(["render", "--profile", "tuned-complete",
                           "--overlay", str(path)])
        assert rc == 2
        assert "--resource-pass tuned" in capsys.readouterr().err

    def test_it_replaces_the_lists_it_names_and_keeps_the_rest(self):
        merged, _ = profile_overlay.apply_overlay(
            load_profile("tuned-complete"),
            self._overlay({"m_values_by_engine": {"mariadb": [8, 16, 32]},
                           "pgvector_ef_construction": [200]}))
        resources = _resources(merged)
        assert ann_m_values("mariadb", merged, resources, "tuned") == [8, 16,
                                                                       32]
        # Not named by the overlay: the tuned pass's own pin survives,
        # dict-into-dict, as a profile's resources block always merged.
        assert ann_m_values("alisql", merged, resources, "tuned") == [6, 16]
        assert resources["extras"]["pgvector_ef_construction"] == [200]
        assert resources["extras"]["valkey_ef_construction"] == EFC

    def test_the_target_profile_is_not_modified(self):
        target = load_profile("tuned-complete")
        before = json.dumps(target, sort_keys=True)
        profile_overlay.apply_overlay(
            target, self._overlay({"pgvector_ef_construction": [64]}))
        assert json.dumps(target, sort_keys=True) == before

    def test_a_stage_1_with_two_storage_engines_is_refused(self, tmp_path):
        run_dir = _stage1(tmp_path)
        path = os.path.join(run_dir, "run-manifest.json")
        manifest = json.load(open(path))
        manifest["config"]["profile"]["resources"]["extras"][
            "mariadb_storage_engines"] = ["InnoDB", "MyISAM"]
        json.dump(manifest, open(path, "w"))
        with pytest.raises(ValueError, match="storage engines apart"):
            select_builds.select(run_dir)

    @pytest.mark.parametrize("overlay,match", [
        ({"overlay_of": "tuned-complete", "runs": 1}, "may not set"),
        ({"overlay_of": "main", "resources": {}}, "written for profile"),
        ({"overlay_of": "tuned-complete",
          "resources": {"memory": {"host_fraction": 0.9}}}, "only `extras`"),
        ({"overlay_of": "tuned-complete",
          "resources": {"extras": {"mariadb_storage_engines": ["MyISAM"]}}},
         "may not set"),
    ])
    def test_anything_but_build_axes_is_refused(self, overlay, match):
        with pytest.raises(ValueError, match=match):
            profile_overlay.apply_overlay(load_profile("tuned-complete"),
                                          overlay)

    def test_run_refuses_an_overlay_on_the_normalized_pass(self, tmp_path,
                                                           monkeypatch,
                                                           capsys):
        path = tmp_path / "o.yml"
        path.write_text(yaml.safe_dump(self._overlay(
            {"pgvector_ef_construction": [200]})))
        monkeypatch.setattr(cli_mod.docker_ctl, "docker_available",
                            lambda: True)
        rc = cli_mod.main(["run", "--profile", "tuned-complete",
                           "--resource-pass", "both", "--overlay", str(path)])
        assert rc == 2
        assert "--resource-pass tuned" in capsys.readouterr().err


class TestTheManifestRecordsTheSelection:

    BLOCK = {"stage1": {"run_id": "frontier-explore-1"}, "kept": {}}

    def test_recorded_once_and_unchanged_on_an_identical_resume(self,
                                                                tmp_path):
        manifest = Manifest(str(tmp_path), "stage2")
        assert manifest.set_build_selection(dict(self.BLOCK)) == "recorded"
        data = Manifest.load(str(tmp_path))
        assert data["build_selection"]["stage1"]["run_id"] == \
            "frontier-explore-1"
        again = Manifest(str(tmp_path), "stage2")
        assert again.set_build_selection(dict(self.BLOCK)) == "unchanged"

    def test_a_resume_with_a_different_selection_is_refused(self, tmp_path):
        Manifest(str(tmp_path), "stage2").set_build_selection(dict(self.BLOCK))
        other = {**self.BLOCK, "stage1": {"run_id": "another"}}
        with pytest.raises(SelectionChanged, match="another"):
            Manifest(str(tmp_path), "stage2").set_build_selection(other)

    def test_a_resume_without_the_overlay_is_refused(self, tmp_path):
        Manifest(str(tmp_path), "stage2").set_build_selection(dict(self.BLOCK))
        with pytest.raises(SelectionChanged, match="no build selection"):
            Manifest(str(tmp_path), "stage2").set_build_selection(None)

    def test_a_resolved_run_that_died_before_a_phase_cannot_acquire_one(
            self, tmp_path):
        manifest = Manifest(str(tmp_path), "plain")
        manifest.data["engines_resolved"] = {"mariadb": {"tuned": {}}}
        with pytest.raises(SelectionChanged):
            manifest.set_build_selection(dict(self.BLOCK))

    def test_units_measured_without_one_cannot_acquire_one(self, tmp_path):
        manifest = Manifest(str(tmp_path), "plain")
        manifest.add_phase("ann", "mariadb", DATASET, "completed", "a", "b",
                           {"resource_pass": "tuned"})
        with pytest.raises(SelectionChanged):
            manifest.set_build_selection(dict(self.BLOCK))

    def test_a_plain_run_records_nothing(self, tmp_path):
        manifest = Manifest(str(tmp_path), "plain")
        assert manifest.set_build_selection(None) == "none"
        assert "build_selection" not in Manifest.load(str(tmp_path))
