"""VillageSQL on its MySQL 8.4 base (`villagesql84`): a second server line for
the same vsql_vector commit.

What this pins is that the two VillageSQL engines differ in the SERVER and in
nothing else a run could mistake for an implementation difference -- and that
the two collisions the web UI's validator refuses (a shared ann_constructor, a
shared image) cannot happen even though both engines' images are tagged with
the same vsql-vector commit. The behaviour of the extension itself (text
binding, the plan refusal, the ef_construction rule) is pinned once, in
tests/test_e26_villagesql_engine.py; here it is only checked to reach the
second engine unchanged.
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

EXTENSION_PIN = "93ed00873dd65f64c0788c80576d5cf3fba39831"
SERVER_TAG_84 = "publish/mysql-8.4_0.0.7"
SERVER_PIN_84 = "689c74a9f88c91ae035f57e0ae8915a168274112"


def _read(rel: str) -> str:
    with open(os.path.join(VB_ROOT, rel)) as fh:
        return fh.read()


def _cfg(name: str) -> dict:
    return yaml.safe_load(_read(f"config/engines/{name}.yml"))


@pytest.fixture(scope="module")
def v97():
    return _cfg("villagesql")


@pytest.fixture(scope="module")
def v84():
    return _cfg("villagesql84")


class TestOnlyTheServerDiffers:
    def test_the_same_extension_a_different_server(self, v97, v84):
        assert v84["alias_of"] == "villagesql"
        assert v84["source"]["tag"] == v97["source"]["tag"] == EXTENSION_PIN
        assert v84["source"]["server_tag"] == SERVER_TAG_84
        assert v84["source"]["server_commit"] == SERVER_PIN_84
        assert v84["source"]["server_commit"] != v97["source"]["server_commit"]
        assert v84["source"]["upstream"] == v97["source"]["upstream"]
        assert (v84["source"]["server_upstream"]
                == v97["source"]["server_upstream"])

    @pytest.mark.parametrize("key", ["build", "server", "storage_engines",
                                     "capabilities"])
    def test_build_flags_capabilities_and_server_flags_are_identical(
            self, v97, v84, key):
        """A different cmake flag, server flag or capability would make the
        server line no longer the only variable between the two curves."""
        assert v84[key] == v97[key]

    def test_runtime_differs_only_in_identity_and_presentation(self, v97, v84):
        identity = {"order", "ann_constructor", "chart", "label"}
        a = {k: v for k, v in v97["runtime"].items() if k not in identity}
        b = {k: v for k, v in v84["runtime"].items() if k not in identity}
        assert a == b
        assert v84["runtime"]["default_selected"] is False
        assert v84["build"]["march"] == "x86-64-v3"


class TestTheTwoCollisions:
    def test_a_distinct_ann_constructor(self, v97, v84):
        assert v84["runtime"]["ann_constructor"] == "VillageSQL84"
        assert (v84["runtime"]["ann_constructor"]
                != v97["runtime"]["ann_constructor"])

    def test_distinct_image_names_although_the_tags_are_equal(self, v97, v84):
        """build-images.sh tags both images `<name>:<source.tag>`, and the tag
        is the shared extension commit -- so only the name keeps an 8.4 build
        from overwriting the 9.7 image under the tag a manifest recorded."""
        assert v84["source"]["tag"] == v97["source"]["tag"]
        assert v84["image"]["runtime"] == "vector-bench/villagesql84-runtime"
        assert v84["image"]["bench"] == "vector-bench/villagesql84-bench"
        assert not ({v84["image"]["runtime"], v84["image"]["bench"]}
                    & {v97["image"]["runtime"], v97["image"]["bench"]})

    def test_the_web_ui_validator_accepts_it(self):
        from webui import engines as wengines
        from orchestrator.config import CONFIG_DIR
        parsed = _cfg("villagesql84")
        assert wengines._collision_errors(
            CONFIG_DIR, "villagesql84", parsed, parsed["runtime"]) == []
        assert not ({parsed["image"]["runtime"], parsed["image"]["bench"]}
                    & set(wengines._other_images(CONFIG_DIR, "villagesql84")))

    def test_the_rendered_ann_config_names_its_own_module_and_image(self):
        from orchestrator import ann_pass
        profile = {"ann": {"m_values": [8, 16], "ef_search": [10, 40]}}
        a = ann_pass.render_config("villagesql", profile, {}, "normalized")
        b = ann_pass.render_config("villagesql84", profile, {}, "normalized")
        ea, eb = a["float"]["any"][0], b["float"]["any"][0]
        assert eb["constructor"] == "VillageSQL84"
        assert eb["module"] == "ann_benchmarks.algorithms.villagesql84"
        assert eb["docker_tag"] == "vector-bench/villagesql84-bench"
        assert eb["name"] == "villagesql84"
        # The grid is the 9.7 engine's, value for value.
        assert eb["run_groups"] == ea["run_groups"]


class TestTheExtensionRulesReachIt:
    def test_the_driver_is_villagesqls(self):
        from harness.drivers import villagesql as vs
        from harness.drivers.postgres import _driver_table
        assert _driver_table()["villagesql84"] is vs.VillageSQLDriver

    def test_the_ef_construction_rule_and_bounds_are_the_extensions(self):
        from orchestrator import ann_knobs
        assert (ann_knobs.KNOB_BOUNDS["villagesql84"]
                == ann_knobs.KNOB_BOUNDS["villagesql"])
        assert (ann_knobs.excluded_builds("villagesql84", [4, 6, 8, 16],
                                          [64, 200, 400])
                == ann_knobs.excluded_builds("villagesql", [4, 6, 8, 16],
                                             [64, 200, 400]))

    def test_ann_and_ops_build_at_one_ef_construction_from_one_key(self):
        from orchestrator import ann_pass, ops_pass
        profile = {"ann": {"m_values": [16], "ef_search": [10],
                           "villagesql_ef_construction": 64}, "ops": {}}
        body = ann_pass.render_config("villagesql84", profile, {}, "normalized")
        group = body["float"]["any"][0]["run_groups"]["innodb"]
        assert group["arg_groups"] == [{"M": [16], "efConstruction": [64]}]
        args = ops_pass.harness_args({**profile, "name": "t", "k": 10}, 16,
                                     "villagesql84", None, "normalized", {})
        assert args[args.index("--ef-construction") + 1] == "64"

    def test_both_declined_workloads_and_the_build_mode(self, v84):
        from orchestrator import cli, estimate
        assert estimate._runs_churn("villagesql84") is False
        assert estimate._runs_filtered_search("villagesql84") is False
        assert cli._ops_build_modes("villagesql84", {}, v84) == ["incremental"]

    def test_opt_in(self):
        from orchestrator import cli
        assert "villagesql84" in cli.KNOWN_ENGINES
        assert "villagesql84" not in cli.DEFAULT_ENGINES

    @pytest.mark.parametrize("profile_name", ["smoke", "m-sweep",
                                              "frontier-explore"])
    def test_the_reference_build_resolves_as_the_9_7_engines_does(
            self, profile_name):
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        from orchestrator.reference_build import (reference_build_problem,
                                                  resolve_reference_build)
        profile = load_profile(profile_name)
        for resource_pass in ("normalized", "tuned"):
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            assert (resolve_reference_build("villagesql84", profile, resources,
                                            resource_pass)
                    == resolve_reference_build("villagesql", profile,
                                               resources, resource_pass))
            assert reference_build_problem("villagesql84", profile, resources,
                                           resource_pass) is None

    def test_m_sweep_gives_it_the_same_m_list(self):
        from orchestrator.config import load_profile
        by_engine = load_profile("m-sweep")["ann"]["m_values_by_engine"]
        assert by_engine["villagesql84"] == by_engine["villagesql"]


class TestTheAnnModule:
    def test_it_is_the_9_7_class_under_its_own_name(self):
        from tests.test_e26_villagesql_engine import _load_module
        import importlib.util
        from tests.test_e1_percona_design import ALGORITHMS
        base = _load_module()
        spec = importlib.util.spec_from_file_location(
            "ann_benchmarks.algorithms.villagesql84.module",
            os.path.join(ALGORITHMS, "villagesql84", "module.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        was = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = was
        cls = mod.VillageSQL84
        assert issubclass(cls, base.VillageSQL)
        assert cls.vb_engine == "villagesql84"
        # Nothing but the name and the label is overridden: the plan check,
        # the DDL and the text binding are the 9.7 module's own.
        assert set(vars(cls)) - {"__module__", "__doc__", "__qualname__",
                                 "__firstlineno__", "__static_attributes__"} \
            == {"vb_engine", "__str__"}


class TestTheBuildScripts:
    def test_prepare_sources_runs_the_routine_on_its_own_config(self):
        script = _read("scripts/prepare-sources.sh")
        assert 'villagesql|villagesql84) prepare_villagesql "$ENGINE" ;;' in script
        # Per-engine source record, so the manifest's engines.villagesql84
        # .source names the 8.4 server rather than whichever ran last.
        assert '"$VB_SOURCES/${engine}.source.json"' in script
        assert 'local cfg="$VB_CONFIG/engines/$engine.yml"' in script

    def test_build_images_has_a_branch_and_names_the_engine(self):
        script = _read("scripts/build-images.sh")
        assert "villagesql84) build_engine villagesql84 ;;" in script
        assert '--build-arg "VB_ENGINE_NAME=${engine}"' in script
        assert "ARG VB_ENGINE_NAME=villagesql" in _read(
            "docker/villagesql/Dockerfile")
