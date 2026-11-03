"""Tests for adding and editing an engine from the web UI.

The failures worth catching here are the ones that would produce a plausible
config that measures the wrong thing: two engines sharing an ann-benchmarks
constructor (and therefore one results tree), two sharing an image tag, or an
engine naming a driver nothing can instantiate.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui import engines as engines_mod  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def restore_registry():
    """The registry is a process-wide cache; a test must not leave its own in it."""
    from orchestrator import engines as registry_mod
    yield
    registry_mod.registry(refresh=True)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A copy of the real engine configs, so validation sees real neighbours."""
    from orchestrator import engines as registry_mod
    import orchestrator.config as config_mod

    root = tmp_path / "config"
    (root / "engines").mkdir(parents=True)
    for name in ("mariadb", "pgvector"):
        shutil.copy(os.path.join(VB_ROOT, "config", "engines", f"{name}.yml"),
                    root / "engines" / f"{name}.yml")

    def load_engine(name):
        with open(root / "engines" / f"{name}.yml") as fh:
            return yaml.safe_load(fh) or {}

    # engines.py binds load_engine at import, so patching config_mod alone is
    # not enough; and the registry must look at this directory, not the real one.
    monkeypatch.setattr(config_mod, "load_engine", load_engine)
    monkeypatch.setattr(registry_mod, "load_engine", load_engine)
    monkeypatch.setattr(registry_mod, "ENGINES_DIR", str(root / "engines"))
    return str(root)


class TestNaming:
    @pytest.mark.parametrize("name", [
        "percona", "mysql8", "pg17",
    ])
    def test_accepts_plain_names(self, config_dir, name):
        assert engines_mod.engine_path(config_dir, name) is not None

    @pytest.mark.parametrize("name", [
        "", "A", "x", "with-dash", "with_underscore", "../escape", "has space",
        "Uppercase", "9leading", "x" * 40,
    ])
    def test_rejects_the_rest(self, config_dir, name):
        assert engines_mod.engine_path(config_dir, name) is None


class TestClone:
    def test_produces_a_valid_config(self, config_dir):
        text, errors = engines_mod.clone(config_dir, "mariadb", "percona")
        assert errors == []
        found, warnings, parsed = engines_mod.validate(config_dir, "percona", text)
        assert found == []
        assert parsed["name"] == "percona"

    def test_renames_images_and_constructor(self, config_dir):
        text, _errors = engines_mod.clone(config_dir, "mariadb", "percona")
        parsed = yaml.safe_load(text)
        assert parsed["image"]["runtime"] == "vector-bench/percona-runtime"
        assert parsed["image"]["bench"] == "vector-bench/percona-bench"
        assert parsed["runtime"]["ann_constructor"] == "Percona"

    def test_keeps_the_driver(self, config_dir):
        text, _errors = engines_mod.clone(config_dir, "mariadb", "percona")
        assert yaml.safe_load(text)["runtime"]["driver"] == "MariaDBDriver"

    def test_keeps_the_explanation(self, config_dir):
        """The configs are mostly rationale; a yaml round-trip would drop it."""
        text, _errors = engines_mod.clone(config_dir, "mariadb", "percona")
        assert "# " in text
        assert "MHNSW" in text

    def test_refuses_an_existing_name(self, config_dir):
        _text, errors = engines_mod.clone(config_dir, "mariadb", "pgvector")
        assert any("already exists" in e for e in errors)

    def test_refuses_an_unknown_base(self, config_dir):
        _text, errors = engines_mod.clone(config_dir, "nope", "percona")
        assert any("no such engine" in e for e in errors)

    def test_refuses_a_bad_name(self, config_dir):
        _text, errors = engines_mod.clone(config_dir, "mariadb", "Bad Name")
        assert errors


class TestValidation:
    def _clone(self, config_dir):
        text, _ = engines_mod.clone(config_dir, "mariadb", "percona")
        return text

    def test_constructor_collision_is_refused(self, config_dir):
        """ann-benchmarks keys result files on the constructor name.

        Two engines sharing one would share a results tree, and the second
        would report the first's recall numbers. This is why mariadb123 is a
        separate constructor rather than a retagged mariadb.
        """
        text = self._clone(config_dir).replace(
            "ann_constructor: Percona", "ann_constructor: MariaDB")
        errors, _w, _p = engines_mod.validate(config_dir, "percona", text)
        assert any("already used by mariadb" in e for e in errors)

    def test_image_collision_is_refused(self, config_dir):
        text = self._clone(config_dir).replace(
            "vector-bench/percona-runtime", "vector-bench/mariadb-runtime")
        errors, _w, _p = engines_mod.validate(config_dir, "percona", text)
        assert any("already used by mariadb" in e for e in errors)

    def test_unknown_driver_is_refused_and_explains(self, config_dir):
        text = self._clone(config_dir).replace(
            "driver: MariaDBDriver", "driver: ElasticDriver")
        errors, _w, _p = engines_mod.validate(config_dir, "percona", text)
        assert errors
        message = " ".join(errors)
        assert "ElasticDriver" in message
        assert "harness/drivers/" in message

    def test_name_mismatch_is_refused(self, config_dir):
        errors, _w, _p = engines_mod.validate(config_dir, "other", self._clone(config_dir))
        assert any("expected 'other'" in e for e in errors)

    @pytest.mark.parametrize("key", ["driver", "ann_constructor", "port",
                                     "data_mount", "probe"])
    def test_missing_runtime_key_is_refused(self, config_dir, key):
        parsed = yaml.safe_load(self._clone(config_dir))
        parsed["runtime"].pop(key)
        errors, _w, _p = engines_mod.validate(config_dir, "percona",
                                              yaml.safe_dump(parsed))
        assert any(f"runtime.{key} is required" in e for e in errors)

    @pytest.mark.parametrize("port", [0, -1, 70000, "3306"])
    def test_bad_port_is_refused(self, config_dir, port):
        parsed = yaml.safe_load(self._clone(config_dir))
        parsed["runtime"]["port"] = port
        errors, _w, _p = engines_mod.validate(config_dir, "percona",
                                              yaml.safe_dump(parsed))
        assert any("runtime.port" in e for e in errors)

    def test_probe_must_be_argv(self, config_dir):
        parsed = yaml.safe_load(self._clone(config_dir))
        parsed["runtime"]["probe"] = "sh -c true"
        errors, _w, _p = engines_mod.validate(config_dir, "percona",
                                              yaml.safe_dump(parsed))
        assert any("list of strings" in e for e in errors)

    def test_broken_yaml_is_refused(self, config_dir):
        errors, _w, _p = engines_mod.validate(config_dir, "percona", "name: [oops\n")
        assert any("YAML error" in e for e in errors)

    def test_non_mapping_is_refused(self, config_dir):
        errors, _w, _p = engines_mod.validate(config_dir, "percona", "- a\n- b\n")
        assert any("must be a YAML mapping" in e for e in errors)


class TestWrite:
    def test_writes_and_is_readable_back(self, config_dir):
        text, _ = engines_mod.clone(config_dir, "mariadb", "percona")
        ok, errors, _w = engines_mod.write(config_dir, "percona", text)
        assert ok and errors == []
        assert engines_mod.read(config_dir, "percona")["parsed"]["name"] == "percona"

    def test_refuses_to_write_an_invalid_config(self, config_dir):
        ok, errors, _w = engines_mod.write(config_dir, "percona", "name: percona")
        assert not ok and errors
        assert not os.path.exists(os.path.join(config_dir, "engines", "percona.yml"))

    def test_read_missing_is_none(self, config_dir):
        assert engines_mod.read(config_dir, "nope") is None


class TestListing:
    def test_lists_the_real_engines(self):
        """In presentation order, which is `runtime.order` and not the
        alphabetical order the directory scan returns.

        A value assertion, so an engine config added to config/engines/
        without this line being updated fails here rather than quietly
        appearing in the web UI's Engines tab. `percona` is seventh by
        `order: 70`; alphabetically it would sit between mongodb and
        pgvector.
        """
        listed = engines_mod.listing(os.path.join(VB_ROOT, "config"))
        names = [e["name"] for e in listed]
        assert names == ["mariadb", "alisql", "pgvector", "mariadb123",
                         "mongodb", "valkey", "percona", "myvector",
                         "villagesql"]

    def test_carries_what_the_table_shows(self):
        listed = engines_mod.listing(os.path.join(VB_ROOT, "config"))
        first = listed[0]
        for key in ("driver", "ann_constructor", "port", "tag", "label",
                    "color", "group", "images"):
            assert key in first

    def test_drivers_are_the_ones_the_harness_has(self):
        from harness.drivers.postgres import _driver_classes
        assert engines_mod.available_drivers() == sorted(_driver_classes())
