"""The engine configs do not describe SQL, and nothing may make them again.

`config/engines/*.yml` used to carry an `sql:` block -- `create_table`,
`insert`, `query`, `query_filtered`, `set_ef_search`, `index_plan_marker` --
introduced in mariadb.yml and mariadb123.yml by a comment claiming it was
"Consumed by the ops-harness driver and by the doc generator so the manual
instructions never drift from the code". Neither consumer existed. There is no
doc generator anywhere in the repo, and the harness cannot read `config/` at
all: its container mounts `harness/` alone, which is why `_driver_table()` says
so in its own docstring. The only code anywhere that touched the key was
`webui/engines.py`, which tested whether it was *present* and warned "no sql
block: SQL-family drivers read their dialect from it" when it was not.

Unread for long enough, the blocks drifted twice:

  * mariadb, mariadb123 and alisql declared `CREATE TABLE ... (id, v)` with no
    `tag` column, while the `query_filtered` three lines below in the same
    block filtered on `tag`; pgvector declared `tag int`, nullable, where the
    driver creates `tag int NOT NULL`.
  * when a second filter column arrived, no block mentioned `ctag` at all,
    though every live driver creates it with an index beside `tag`'s.

So the blocks were deleted rather than repaired: repairing them would have
produced a second description of the schema that still nothing executed, with a
test as its only reader -- and a copy whose only reader is a test is the same
decoration one step further on. These tests hold the deletion in place. They
would have caught the `ctag` drift at its cause: with them in the tree the
block that drifted could not have existed to drift, because nothing read it.

Each guard carries its positive control. "An ignored key changes nothing" and
"no file declares the key" are both negative claims, and a scan that looked in
the wrong directory, or a comparison between two empty bundles, satisfies them
without measuring anything.

Two limits, stated rather than papered over:

  * §1 forbids **the key**, not the idea. A block that came back under another
    name -- `statements:`, `ddl:` -- would pass it. What makes that survivable
    is §2: a new block changes nothing the host derives, so it would arrive
    inert, and anything wiring a reader to it has to get past §2 as well.
  * §4 is an argument, not a fence. Its path scan matches `"config/engines"`
    as written and would miss `os.path.join("config", "engines")` or
    `"config/" + "engines"`; its import scan would miss
    `importlib.import_module("yaml")`. It is here to say *why* §1 and §2 are
    safe to rely on -- the harness genuinely has no config to read -- and not
    as the thing doing the enforcing. §1 is that.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import re
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES_DIR = os.path.join(VB_ROOT, "config", "engines")
HARNESS_DIR = os.path.join(VB_ROOT, "harness")

#: Drivers that never spoke SQL, and so never had a block to lose. The old
#: warning exempted exactly these two by name.
NON_SQL_DRIVERS = ("MongoDriver", "ValkeyDriver")

#: A block shaped like the ones that were removed, with every value replaced by
#: something a reader would be unable to hide. If anything consulted it, a
#: derived artifact would carry `c1_sentinel` or would lose a filter column.
HOSTILE_SQL = """
sql:
  session_setup:
    - "SET SESSION sql_mode = 'c1_sentinel'"
  global_setup:
    - "SET GLOBAL c1_sentinel = ON"
  create_table: |
    CREATE TABLE {table} (
      id INT PRIMARY KEY,
      v VECTOR({dim}) NOT NULL
    ) ENGINE={engine}
  insert: "INSERT INTO {table} (id, v) VALUES (%s, %s)"
  query_filtered: "SELECT id FROM {table} WHERE c1_sentinel < %s LIMIT {k}"
  set_ef_search: "SET c1_sentinel_ef = {ef_search}"
  index_plan_marker: "c1_sentinel_marker"
  metrics:
    angular: c1_sentinel
    euclidean: c1_sentinel
"""


# ---------------------------------------------------------------------------
# Reading the configs
# ---------------------------------------------------------------------------

def _docstring_ids(tree):
    """`id()` of every docstring node in `tree`.

    Docstrings are string constants like any other, and in this repo they are
    where the reasoning lives -- `_driver_table()`'s own docstring explains
    that the harness cannot read `config/engines/*.yml`. A scan for strings
    that *claim* something therefore has to exclude the strings that explain
    it, or it flags the explanation and not the claim.
    """
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            found.add(id(first.value))
    return found


def string_literals(path: str):
    """Every string constant in `path` that is not a docstring."""
    tree = ast.parse(open(path).read())
    skip = _docstring_ids(tree)
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in skip]


def engine_configs(directory=None):
    """Every `<name>.yml` under `directory`, parsed, keyed by engine name.

    `directory` resolves to `ENGINES_DIR` when it is called rather than when
    it is defined, so the mutation harness can point the guards below at a
    temporary copy of the engine directory and read the answer back.
    """
    directory = directory or ENGINES_DIR
    names = sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".yml"))
    out = {}
    for name in names:
        with open(os.path.join(directory, f"{name}.yml")) as fh:
            out[name] = yaml.safe_load(fh) or {}
    return out


def engines_declaring_sql(directory=None):
    """Engine names whose config carries a top-level `sql:` key."""
    return [name for name, config in engine_configs(directory).items()
            if "sql" in config]


def sql_family(configs):
    """The engines whose statements are built by a SQL driver.

    Derived from `runtime.driver` rather than listed, so §1's guard picks up a
    fifth MySQL fork on its own and refuses a block in it.

    The *control* below is a different matter, and does not inherit that: it
    names today's four deliberately, because an assertion that pins "what was
    scanned" has to name something, and a subset check would be satisfied by a
    scan that had quietly lost two of them. So adding a fork does mean editing
    one line here. Measured with a legitimate `percona.yml` carrying no `sql:`
    block: the guard passes and `test_the_scan_is_looking_at_the_real_engine_
    directory` fails, naming the five it found.
    """
    return sorted(
        name for name, config in configs.items()
        if (config.get("runtime") or {}).get("driver") not in NON_SQL_DRIVERS)


@pytest.fixture
def engines_copy(tmp_path):
    """A writable copy of the real engine directory."""
    target = tmp_path / "engines"
    shutil.copytree(ENGINES_DIR, target)
    return str(target)


@pytest.fixture(autouse=True)
def restore_registry():
    """The registry is a process-wide cache; a test must not leave its own in it.

    Module-wide rather than per-class: §1 refreshes it to find out which
    engines the orchestrator loads, and §3 points it at a temporary directory.
    """
    from orchestrator import engines as registry_mod
    yield
    registry_mod.registry(refresh=True)


# ---------------------------------------------------------------------------
# §1 -- the blocks are gone and stay gone
# ---------------------------------------------------------------------------

class TestNoEngineConfigDeclaresSQL:
    def test_no_engine_config_declares_an_sql_block(self):
        """The guard the drift needed and did not have.

        Honest: no engine declares one, so the list is empty. Mutant (the
        block put back into mariadb.yml): `['mariadb']`.
        """
        declaring = engines_declaring_sql()
        assert declaring == [], (
            f"{declaring} declare an `sql:` block. Nothing reads it -- the "
            f"harness container cannot see config/ -- so it is a second "
            f"description of the schema that drifts unobserved. The "
            f"statements belong in harness/drivers/.")

    def test_the_scan_is_looking_at_the_real_engine_directory(self):
        """Positive control for the assertion above.

        A scan pointed at an empty or wrong directory returns `[]` too. This
        pins what was scanned: every engine the orchestrator would load, with
        the SQL-family configs the blocks used to live in among them.

        `percona` is in the list without ever having carried a block. It is
        the first SQL-family engine config written AFTER the blocks were
        removed, and it is in scope for the assertion above for exactly that
        reason: the risk this file guards against is a block coming BACK, and
        a new config written from an old example is the likeliest way.
        """
        configs = engine_configs()
        from orchestrator import engines as registry_mod
        loaded = sorted(registry_mod.registry(refresh=True))
        assert sorted(configs) == loaded, (
            f"scanned {sorted(configs)} but the orchestrator loads {loaded}")
        family = sql_family(configs)
        assert family == ["alisql", "mariadb", "mariadb123", "myvector",
                          "percona", "pgvector", "villagesql"], (
            f"the configs the blocks lived in are {family}")
        # And every one of them parsed to something, so `"sql" in config` was
        # asked of a real mapping rather than of an empty one.
        for name, config in configs.items():
            assert isinstance(config.get("runtime"), dict), name

    def test_the_scan_reports_a_block_that_is_there(self, engines_copy):
        """Positive control for the scan itself: the named regression.

        Honest: `[]`. Mutant (HOSTILE_SQL appended to mariadb.yml in a copy of
        the directory, read back off disk): `['mariadb']`.
        """
        assert engines_declaring_sql(engines_copy) == []
        path = os.path.join(engines_copy, "mariadb.yml")
        with open(path, "a") as fh:
            fh.write(HOSTILE_SQL)
        assert engines_declaring_sql(engines_copy) == ["mariadb"]


# ---------------------------------------------------------------------------
# §2 -- and would change nothing if one returned
# ---------------------------------------------------------------------------

def _machine():
    """A fixed machine description, so `server_args` renders deterministically."""
    info = type("I", (), {})()
    info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                              "performance_cpus": list(range(40)),
                              "efficiency_cpus": [], "logical_cpus": 80,
                              "physical_cores": 40, "threads_per_core": 2,
                              "model": "Xeon"})()
    info.total_ram_bytes = int(192 * 1024 ** 3)
    return info


def derived(name: str, config):
    """Everything the host side turns an engine config into.

    `ann_pass.render_config` is absent on purpose: it is handed an engine
    *name*, a profile and a resource pass, never the engine config, so it
    could not read an `sql:` block if one existed.
    """
    from orchestrator.config import load_resources, resolve_resources, server_args
    from orchestrator.engines import _build

    machine = _machine()
    bundle = {"runtime": dataclasses.asdict(_build(name, config, 0))}
    for resource_pass in ("normalized", "tuned"):
        resolved = resolve_resources(load_resources(resource_pass), name, machine)
        bundle[resource_pass] = server_args(config, resource_pass, resolved)
    return bundle


class TestNothingDerivesAnythingFromIt:
    @pytest.mark.parametrize("name", ["mariadb", "mariadb123", "alisql",
                                      "pgvector"])
    def test_an_sql_block_changes_nothing_the_host_derives(self, name):
        """Honest: the two bundles are identical, so the diff is empty.

        Mutant (`server_args` made to append `sql.index_plan_marker`, which is
        what "something started reading it" looks like): the diff becomes
        `['normalized', 'tuned']` and both carry a final
        `'--index-plan-marker=c1_sentinel_marker'`.
        """
        with open(os.path.join(ENGINES_DIR, f"{name}.yml")) as fh:
            text = fh.read()
        without = yaml.safe_load(text)
        with_block = yaml.safe_load(text + HOSTILE_SQL)
        assert "sql" in with_block and "sql" not in without

        plain, carrying = derived(name, without), derived(name, with_block)
        differing = sorted(k for k in plain if plain[k] != carrying[k])
        assert differing == [], (
            f"{name}: adding an `sql:` block moved {differing}; "
            f"{[(k, plain[k], carrying[k]) for k in differing]}")

    def test_a_key_that_is_read_does_move_the_bundle(self):
        """Positive control: the bundle is not constant and not empty.

        Without this, a `derived()` that returned `{}` would satisfy the
        equality above for every engine.
        """
        with open(os.path.join(ENGINES_DIR, "mariadb.yml")) as fh:
            config = yaml.safe_load(fh)
        plain = derived("mariadb", config)
        assert plain["normalized"], "server_args rendered nothing to compare"

        moved = yaml.safe_load(yaml.safe_dump(config))
        moved["server"]["normalized"].append("--c1-sentinel-flag")
        changed = derived("mariadb", moved)
        assert "--c1-sentinel-flag" in changed["normalized"]
        assert changed["normalized"] != plain["normalized"]


# ---------------------------------------------------------------------------
# §3 -- the validator no longer claims otherwise
# ---------------------------------------------------------------------------

@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A copy of the real engine configs, so validation sees real neighbours."""
    from orchestrator import engines as registry_mod
    import orchestrator.config as config_mod

    root = tmp_path / "config"
    (root / "engines").mkdir(parents=True)
    for name in ("mariadb", "pgvector"):
        shutil.copy(os.path.join(ENGINES_DIR, f"{name}.yml"),
                    root / "engines" / f"{name}.yml")

    def load_engine(name):
        with open(root / "engines" / f"{name}.yml") as fh:
            return yaml.safe_load(fh) or {}

    monkeypatch.setattr(config_mod, "load_engine", load_engine)
    monkeypatch.setattr(registry_mod, "load_engine", load_engine)
    monkeypatch.setattr(registry_mod, "ENGINES_DIR", str(root / "engines"))
    return str(root)


class TestTheValidatorSaysNothingAboutSQL:
    def test_a_block_present_or_absent_is_not_remarked_on(self, config_dir):
        """The warning that was removed, stated as the value it moved.

        Honest: `([], [])` for both texts -- the validator has nothing to say
        either way. Mutant (the old branch restored): the config without a
        block warns `['no sql block: SQL-family drivers read their dialect
        from it']` while the one with a block warns `[]`, so the two stop
        matching and the message is quoted in the failure.
        """
        from webui import engines as engines_mod

        text, errors = engines_mod.clone(config_dir, "mariadb", "percona")
        assert errors == []
        without = engines_mod.validate(config_dir, "percona", text)
        with_block = engines_mod.validate(config_dir, "percona",
                                          text + HOSTILE_SQL)
        assert (without[0], without[1]) == ([], []), without
        assert (with_block[0], with_block[1]) == ([], []), with_block
        assert "sql" in with_block[2] and "sql" not in without[2]

    def test_the_validator_still_reports_what_it_is_for(self, config_dir):
        """Positive control: `([], [])` above is a live answer, not a stub.

        A `validate` that returned two empty lists unconditionally would pass
        the test above for every input.
        """
        from webui import engines as engines_mod

        text, _errors = engines_mod.clone(config_dir, "mariadb", "percona")
        broken = text.replace("driver: MariaDBDriver", "driver: C1Driver")
        errors, warnings, _parsed = engines_mod.validate(config_dir, "percona",
                                                         broken)
        assert any("C1Driver" in e for e in errors), errors
        assert warnings == []

    def test_no_message_in_the_module_claims_a_driver_reads_a_config(self):
        """The prose, not just the behaviour.

        Every operator-facing string in webui/engines.py, so a rewording of
        the old warning cannot slip back in under a different branch. The
        explanation of the removal lives in a `#` comment there, which never
        becomes an AST node at all, so it is outside the scan for free --
        `string_literals`' docstring exclusion is load-bearing for §4's
        harness scan, where real docstrings do name `config/engines/*.yml`,
        and not for this one.
        """
        import webui.engines as engines_mod

        literals = string_literals(engines_mod.__file__)
        assert literals, "no strings found; the scan is looking at nothing"
        assert any("runtime.driver" in s for s in literals), (
            "the scan is not seeing this module's operator-facing messages")
        offending = [s for s in literals
                     if re.search(r"\bsql\b", s, re.IGNORECASE)
                     or "dialect" in s.lower()]
        assert offending == [], (
            f"webui/engines.py still tells an operator about engine-config "
            f"SQL: {offending}")


# ---------------------------------------------------------------------------
# §4 -- why §1 and §2 are safe to rely on
# ---------------------------------------------------------------------------

def _imported_modules(directory: str):
    """Top-level module names imported by every .py under `directory`."""
    found, files = set(), []
    for root, _dirs, names in os.walk(directory):
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            files.append(path)
            for node in ast.walk(ast.parse(open(path).read())):
                if isinstance(node, ast.Import):
                    found.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    found.add((node.module or "").split(".")[0])
    return found, files


class TestTheHarnessCannotReadAConfig:
    def test_nothing_under_harness_imports_a_yaml_parser(self):
        """The structural reason an `sql:` block could never have run.

        harness/ is bind-mounted into the bench container on its own; there is
        no config/ beside it and no parser to read one with. The host decides
        and passes `--driver`.

        Honest: the imports of harness/ do not include `yaml`. Mutant (`import
        yaml` injected into harness/drivers/mysql_family.py in a copy of the
        tree): `yaml` is in the set and the failure names the file.
        """
        imported, files = _imported_modules(HARNESS_DIR)
        assert len(files) > 1 and "numpy" in imported, (
            f"the scan read {len(files)} file(s) and found {sorted(imported)}; "
            f"it is not reading the harness")
        assert "yaml" not in imported, (
            "a module under harness/ imports yaml, so it could parse an "
            "engine config. The container has no config/ to parse.")

    def test_no_path_under_harness_names_the_engine_directory(self):
        """A path string is all it would take, even without a parser.

        Docstrings excluded: several modules under harness/ mention
        `config/engines/*.yml` precisely to say they must not read it.

        Honest: no non-docstring string under harness/ names the directory.
        Mutant (`CONFIG = "config/engines"` added to a copy of
        harness/drivers/mysql_family.py): one entry, naming the file and the
        string.
        """
        offending, scanned = [], 0
        for root, _dirs, names in os.walk(HARNESS_DIR):
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                scanned += 1
                for literal in string_literals(path):
                    if "config/engines" in literal:
                        offending.append((os.path.relpath(path, VB_ROOT),
                                          literal))
        assert scanned > 1, f"the scan read {scanned} file(s)"
        assert offending == [], offending

    @pytest.mark.parametrize("driver_class,expected", [
        ("MariaDBDriver", "SET mhnsw_ef_search = 64"),
        ("AliSQLDriver", "SET vidx_hnsw_ef_search = 64"),
        ("PostgresDriver", "SET hnsw.ef_search = 64"),
    ])
    def test_the_dialect_is_the_drivers_own(self, driver_class, expected):
        """What the removed warning asserted, measured.

        "SQL-family drivers read their dialect from it" -- they do not. A
        driver is constructed from a class name and a `ConnectionSpec`; there
        is no argument an engine config could arrive through. These are the
        three different statements the three drivers put on the wire, so the
        assertion cannot be satisfied by one shared constant.
        """
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.postgres import _driver_classes

        class Recorder:
            def __init__(self):
                self.sql = []

            def execute(self, sql, params=None, **kw):
                self.sql.append(sql)

        driver = _driver_classes()[driver_class](
            ConnectionSpec(host="h", port=1))
        driver._cur = Recorder()
        driver.set_ef_search(64)
        assert driver._cur.sql == [expected], (
            f"{driver_class} put {driver._cur.sql} on the wire, not "
            f"[{expected!r}]; the knob it sets is the driver module's own "
            f"constant and no config declares it")
