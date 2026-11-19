"""Percona built at MariaDB MHNSW's build effort (`percona-efc-matched`).

The profile exists to take build effort out of the Percona-vs-MHNSW
comparison, so the failure that matters is silent: a "matched" curve built at
some other ef_construction, an M paired with the wrong value by upstream's
cartesian product, an ops index left at 200 under a reference build that says
32, or result files that collide with the efC=200 ones in the shared ann tree.
Each test below pins one of those.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import ann_pass, mhnsw_matched, ops_pass  # noqa: E402
from orchestrator.ann_config import _combinations, result_stems  # noqa: E402
from orchestrator.cli import VB_ROOT  # noqa: E402
from orchestrator.config import (load_profile, load_resources,  # noqa: E402
                                 merge_resource_overrides)
from orchestrator.reference_build import (  # noqa: E402
    ops_build_problems, reference_build_problem, resolve_reference_build)

PROFILE = "percona-efc-matched"
MARIADB_TAGS = ("mariadb-11.8.8", "mariadb-12.3.2")


def _render(engine, profile, resource_pass="normalized"):
    resources = merge_resource_overrides(load_resources(resource_pass),
                                         profile)
    return ann_pass.render_config(engine, profile, resources, resource_pass)


def _builds(body):
    """(M, efConstruction) of every definition the body asks upstream for."""
    out = set()
    for algo in body["float"]["any"]:
        for group in algo["run_groups"].values():
            for combo in _combinations(group, "arg_groups", "args"):
                args = combo[0] if isinstance(combo, list) else combo
                out.add((int(args["M"]), int(args["efConstruction"])))
    return out


class TestTheFormula:
    """max(10, 2*M), read from `sql/vector_mhnsw.cc` at both pins."""

    @pytest.mark.parametrize("m,width", [(3, 10), (4, 10), (5, 10), (6, 12),
                                         (8, 16), (16, 32), (24, 48),
                                         (32, 64), (48, 96)])
    def test_layer_zero_width(self, m, width):
        assert mhnsw_matched.mhnsw_build_width(m) == width

    @pytest.mark.parametrize("tag", MARIADB_TAGS)
    def test_the_three_symbols_are_still_in_the_pinned_source(self, tag):
        """The vendor checkout is a read-only input beside the repo
        (`VB_REPO_MARIADB`); `git show` reads a tag without touching it.
        Skipped where it is absent, since the host promises no vendor tree."""
        repo = os.environ.get("VB_REPO_MARIADB",
                              os.path.join(os.path.dirname(VB_ROOT), "server"))
        if not os.path.isdir(os.path.join(repo, ".git")):
            pytest.skip(f"no MariaDB checkout at {repo}")
        done = subprocess.run(
            ["git", "-C", repo, "show", f"{tag}:sql/vector_mhnsw.cc"],
            capture_output=True, text=True)
        if done.returncode != 0:
            pytest.skip(f"{tag} not in {repo}")
        src = done.stdout
        assert (f"static constexpr uint ef_construction= "
                f"{mhnsw_matched.MHNSW_EF_CONSTRUCTION};") in src
        assert "return (layer ? 1 : 2) * M;" in src
        assert "ef= std::max(ef_construction, ef);" in src


class TestTheShippedProfile:
    def test_every_m_it_builds_is_matched_to_the_formula(self):
        profile = load_profile(PROFILE)
        by_m = mhnsw_matched.efc_by_m(profile)
        ms = set(profile["ann"]["m_values"]) | set(profile["ops"]["m_values"])
        assert ms <= set(by_m)
        for m, efc in by_m.items():
            assert efc == mhnsw_matched.mhnsw_build_width(m), m

    def test_it_runs_percona_beside_mariadb123(self):
        assert set(load_profile(PROFILE)["engines"]) == {"percona",
                                                         "mariadb123"}

    @pytest.mark.parametrize("resource_pass", ["normalized", "tuned"])
    def test_the_ann_grid_pairs_each_m_with_its_own_width_only(
            self, resource_pass):
        profile = load_profile(PROFILE)
        builds = _builds(_render("percona", profile, resource_pass))
        assert builds == {(m, mhnsw_matched.mhnsw_build_width(m))
                          for m in profile["ann"]["m_values"]}

    def test_its_result_files_never_collide_with_the_efc200_ones(self):
        """ann-benchmarks names a result file from the arguments alone, and
        the tree is shared with `main`: a matched curve must be a new file,
        never one `main` already wrote (which would be skipped and reused)."""
        matched = result_stems(_render("percona", load_profile(PROFILE)))
        default = result_stems(_render("percona", load_profile("main")))
        assert matched and default
        assert not matched & default
        assert any("efConstruction_32" in stem and "M_16" in stem
                   for _, stem in matched)

    def test_mariadb123_renders_the_cells_main_already_has(self):
        """Same M and ef_search grid as `main`, so its files are shared."""
        assert (result_stems(_render("mariadb123", load_profile(PROFILE)))
                == result_stems(_render("mariadb123", load_profile("main"))))

    @pytest.mark.parametrize("m", [8, 16, 32])
    def test_the_ops_build_gets_the_same_value(self, m):
        profile = load_profile(PROFILE)
        argv = ops_pass.harness_args(profile, m, "percona", None,
                                     "normalized", {}, "incremental")
        assert (argv[argv.index("--ef-construction") + 1]
                == str(mhnsw_matched.mhnsw_build_width(m)))

    @pytest.mark.parametrize("resource_pass", ["normalized", "tuned"])
    def test_the_reference_build_is_the_matched_one_and_in_the_grid(
            self, resource_pass):
        profile = load_profile(PROFILE)
        resources = merge_resource_overrides(load_resources(resource_pass),
                                             profile)
        assert resolve_reference_build("percona", profile, resources,
                                       resource_pass) == {
            "m": 16, "ef_construction": 32}
        for engine in profile["engines"]:
            assert reference_build_problem(engine, profile, resources,
                                           resource_pass) is None
            assert ops_build_problems(engine, profile) == []


class TestEveryOtherProfileIsUnchanged:
    def test_main_still_builds_percona_at_200_on_both_paths(self):
        profile = load_profile("main")
        assert _builds(_render("percona", profile)) == {
            (m, 200) for m in profile["ann"]["m_values"]}
        argv = ops_pass.harness_args(profile, 16, "percona", None,
                                     "normalized", {}, "post")
        assert argv[argv.index("--ef-construction") + 1] == "200"
        resources = merge_resource_overrides(load_resources("normalized"),
                                             profile)
        assert resolve_reference_build("percona", profile, resources,
                                       "normalized") == {
            "m": 16, "ef_construction": 200}

    def test_a_scalar_percona_ef_construction_reaches_the_ops_build(self):
        """It used to stop at the ann render: the ops index stayed at the
        dialect's 200 whatever `ann.percona_ef_construction` said."""
        profile = load_profile("main")
        profile["ann"]["percona_ef_construction"] = 400
        assert ops_pass.ops_ef_construction("percona", profile, 16) == 400


class TestRefusals:
    def _profile(self, **ann):
        profile = load_profile(PROFILE)
        profile["ann"].update(ann)
        return profile

    def test_an_m_missing_from_the_map_is_refused_not_defaulted(self):
        profile = self._profile(m_values=[8, 16, 24])
        with pytest.raises(ValueError, match=r"\[24\]"):
            _render("percona", profile)
        with pytest.raises(ValueError, match="M=24"):
            ops_pass.ops_ef_construction("percona", profile, 24)

    def test_both_keys_set_is_refused(self):
        profile = self._profile(percona_ef_construction=200)
        with pytest.raises(ValueError, match="both set"):
            _render("percona", profile)

    def test_a_tuned_sweep_beside_the_map_is_refused(self):
        profile = load_profile(PROFILE)
        profile["resources"] = {"extras": {"percona_ef_construction":
                                           [50, 200]}}
        with pytest.raises(ValueError, match="extras.percona_ef_construction"):
            _render("percona", profile, "tuned")

    def test_the_ops_value_needs_an_m_under_a_map(self):
        with pytest.raises(ValueError, match="needs the M"):
            ops_pass.ops_ef_construction("percona", load_profile(PROFILE))

    def test_an_ops_m_missing_from_the_map_is_refused_before_the_run(self):
        profile = load_profile(PROFILE)
        profile["ops"]["m_values"] = [12]
        problems = ops_build_problems("percona", profile)
        assert problems and "M=12" in problems[0]
