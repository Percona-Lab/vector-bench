"""The `best-of-both` profile: Percona and MariaDB 12.3 each at its own best.

The failures that matter here are silent ones: a "best" Percona curve that
quietly fell back to ef_construction 200 because the pass ignored the sweep, an
ops pass measuring the declared M=16 reference build instead of the frontier
build, a grid that does not reach the ef_search Percona needs, or one engine
paying for performance_schema in a pass meant to show each at its best.
Each test pins one of those.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import ann_pass, config as config_mod, ops_pass  # noqa: E402
from orchestrator.ann_config import _combinations  # noqa: E402
from orchestrator.config import (load_profile, load_resources,  # noqa: E402
                                 merge_resource_overrides)
from orchestrator.reference_build import (  # noqa: E402
    ops_build_problems, reference_build_problem)

PROFILE = "best-of-both"
ENGINES = ("percona", "mariadb123")
MYSQL_FAMILY = ("mariadb", "mariadb123", "alisql", "percona", "myvector",
                "villagesql", "villagesql84")


def _resources(pass_name="tuned"):
    return merge_resource_overrides(load_resources(pass_name),
                                    load_profile(PROFILE))


def _builds(engine):
    """(M, efConstruction or None) of every build the tuned render asks for."""
    body = ann_pass.render_config(engine, load_profile(PROFILE), _resources(),
                                  "tuned")
    out = set()
    for algo in body["float"]["any"]:
        for group in algo["run_groups"].values():
            for combo in _combinations(group, "arg_groups", "args"):
                args = combo[0] if isinstance(combo, list) else combo
                out.add((int(args["M"]), args.get("efConstruction")))
    return out


def _query_args(engine):
    body = ann_pass.render_config(engine, load_profile(PROFILE), _resources(),
                                  "tuned")
    return [group["query_args"] for algo in body["float"]["any"]
            for group in algo["run_groups"].values()]


class TestTheProfile:
    def test_it_measures_the_two_engines_on_the_tuned_pass(self):
        profile = load_profile(PROFILE)
        assert tuple(profile["engines"]) == ENGINES
        assert profile["default_resource_pass"] == "tuned"

    def test_the_ops_pass_measures_each_engines_frontier_build(self):
        """`reference` would put every engine's concurrency figures on M=16,
        the build this profile exists to look past."""
        ops = load_profile(PROFILE)["ops"]
        assert ops["build_selection"] == "frontier"
        # The default, not a declared key: tests/test_calibrate.py keeps
        # every shipped profile on the default target.
        assert "target_recall" not in ops
        assert ops_pass.ops_target_recall(load_profile(PROFILE)) == 0.95


class TestTheGrid:
    def test_percona_sweeps_its_build_effort_from_the_server_default_up(self):
        assert _builds("percona") == {(m, efc) for m in (16, 32)
                                      for efc in (200, 400, 800)}

    def test_mariadb_sweeps_the_m_values_that_owned_its_floors(self):
        """efc-matched-20261119: M=16 owned 0.90, M=32 owned 0.95 and 0.99."""
        assert {m for m, _ in _builds("mariadb123")} == {16, 32}

    @pytest.mark.parametrize("engine", ENGINES)
    def test_ef_search_reaches_what_percona_needed_at_efc_200(self, engine):
        """percona-mariadb123-20261116: 0.958 at ef_search 640, M=16."""
        for args in _query_args(engine):
            assert max(args[0]) >= 640, args

    @pytest.mark.parametrize("engine", ENGINES)
    def test_the_reference_build_is_in_the_grid(self, engine):
        """The fallback when frontier selection cannot choose; `run` refuses
        a reference build the grid does not contain."""
        assert reference_build_problem(engine, load_profile(PROFILE),
                                       _resources(), "tuned") is None
        assert ops_build_problems(engine, load_profile(PROFILE)) == []


class TestPerformanceSchemaInTheTunedPass:
    @pytest.fixture(scope="class")
    def resolved(self):
        from harness.metrics import sysinfo
        info = sysinfo.collect()
        return {e: config_mod.resolve_resources(load_resources("tuned"), e,
                                                info)
                for e in MYSQL_FAMILY}

    @pytest.mark.parametrize("engine", MYSQL_FAMILY)
    def test_off_for_every_mysql_family_engine(self, engine, resolved):
        args = config_mod.server_args(config_mod.load_engine(engine), "tuned",
                                      resolved[engine])
        assert "--performance-schema=OFF" in args

    @pytest.mark.parametrize("engine", MYSQL_FAMILY)
    def test_the_opt_in_wait_instruments_still_switch_it_on(self, engine,
                                                             resolved):
        """The server keeps the last value of a repeated option, so ON must
        come after the pass's OFF."""
        args = config_mod.server_args(config_mod.load_engine(engine), "tuned",
                                      resolved[engine], wait_instruments=True)
        assert (args.index("--performance-schema=ON")
                > args.index("--performance-schema=OFF"))

    @pytest.mark.parametrize("engine", MYSQL_FAMILY)
    def test_the_normalized_pass_still_runs_each_server_as_shipped(self,
                                                                   engine):
        """A deliberate scope: the normalized pass leaves server defaults
        alone, and efc-matched-20261119 is reported with that difference."""
        flags = config_mod.load_engine(engine)["server"].get("normalized") or []
        assert not any("performance-schema" in f for f in flags)
