"""E13 -- Percona's churn "repair" was a silent no-op and the report published
it as a repair.

WHAT WAS MEASURED, and the only thing here that is not checkable from the
repo: `/home/corvin/AI_WORK/VECTOR_RESEARCH/PS/full-20261015-215006`,
`ops-percona-*.jsonl`, phase `churn`. `PerconaDriver.maintain()` issues

    ALTER TABLE t1 DROP KEY vi, ADD <clause from SHOW CREATE TABLE>,
    ALGORITHM=INPLACE

and in that run it returned in **1-4 ms** and changed nothing (the rows below
are fashion-mnist-784-euclidean at 60,000 rows):

    churn  maintenance_state  maintenance_s  recall@10  index_bytes
    0.10   before             --             0.9598     281,018,368
    0.10   after              0.001          0.9598     281,018,368
    0.25   before             --             0.95345    318,767,104
    0.25   after              0.001          0.95345    318,767,104

**Twenty** such pairs -- 10 ops units x 2 churn fractions -- across three
datasets, every one of the 20 identical in recall to five decimals and in
`index_bytes` TO THE BYTE. `maintenance_s` was 0.001 seventeen times, 0.002
twice and 0.004 twice. The 268 MiB graph those `index_bytes` describe is
fashion-mnist-784's at 60,000 rows, NOT sift's; sift-128 at 1,000,000 rows is
among the same 20 and its graph is 808-1,428 MiB, where a genuine rebuild
costs on the order of the 26-80 minute incremental load. A 1-4 ms DDL that
drops and re-adds either graph has not rebuilt it. The report then published
all 20 of those rows as `after ALTER TABLE (index replace) (0.0 s)`: the
degraded index, measured twice, presented as a repair result.

THE DEFECT ON THIS SIDE is not the server's refusal to rebuild. It is that
the statement `maintain()` issues is **the same statement** that
`capabilities.bulk_vector_index_build: false` in `config/engines/percona.yml`
records as killing the server at the pinned `source.tag`. PR #36 (merged,
`a89b536`) stopped the *build mode* default from reaching that path and did
not touch this one, so ten published rows measured the degraded index twice.

The fix: the driver declares **no repair path** when the host says the bulk
vector index build is broken -- `repair_path = "none"`, the base class default
at `harness/drivers/base.py:258`, and `maintain()` returning `None`. The churn
workload needs no change: `_record_maintenance` returns early on a non-`str`,
so only the "before" record is written, which is the shape the five engines
with no repair path already use.

The capability lives in the engine config and `harness/` must not import
`orchestrator` -- each runs in a container that mounts only itself -- so the
decision is made on the host and passed in as `--no-bulk-vector-index-build`,
the way `--driver` already is.

PROBE STYLE, per this suite's convention (see
`tests/test_e9_percona_bulk_build_crash.py`): every decision test reports its
HONEST and MUTANT value rather than a bare pass/fail, and an identical pair is
called out as a dead probe. The mutants are injected at RUN TIME -- a copy of
the engine-config dict with one key flipped, or the same declaration hook
called with the opposite argument -- never by editing a file on disk: CPython
invalidates a `.pyc` on (whole-second mtime, size), so a same-size rewrite
inside one second can leave stale bytecode valid and measure the previous
version. Nothing here starts a container, a server or a build; the driver is
exercised against a recording cursor.
"""

from __future__ import annotations

import ast
import copy
import os
import sys
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import (ConnectionSpec, EngineDriver,       # noqa: E402
                                  IndexSpec)
from harness.drivers.mysql_family import (AliSQLDriver, MariaDBDriver,  # noqa: E402
                                          PerconaDriver)
from harness.metrics import records                                   # noqa: E402
from harness.workloads import churn as cw                             # noqa: E402
from harness.workloads.context import RunContext                      # noqa: E402
from orchestrator import ops_pass                                     # noqa: E402
from tests.fake_driver import (DIM, FakeDriver, K, N_QUERIES,         # noqa: E402
                               N_ROWS, index_spec, tiny_dataset)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAPABILITY = "bulk_vector_index_build"
FLAG = "--no-bulk-vector-index-build"
HOOK = "declare_bulk_vector_index_build"

def _other_engines():
    """Every engine but Percona, from the registry rather than from memory.

    This was a hard-coded six-tuple under a comment claiming it came from the
    registry -- which would have been exactly the drift-by-omission hole the
    claim was meant to close: an engine added to `config/engines/` would have
    been silently exempt from "nothing else moved".
    """
    from orchestrator.cli import KNOWN_ENGINES
    # VillageSQL declares its bulk build broken too (an intermittent latch
    # deadlock aborts the server); its flag is pinned in
    # tests/test_e26_villagesql_engine.py.
    return tuple(sorted(set(KNOWN_ENGINES) - {"percona", "villagesql"}))


OTHER_ENGINES = _other_engines()


def _engine_cfg(engine: str = "percona") -> Dict[str, Any]:
    from orchestrator.config import load_engine
    return load_engine(engine)


def _with_capability(cfg: Dict[str, Any], value: bool) -> Dict[str, Any]:
    """The engine config as it would read if the server were fixed."""
    mutant = copy.deepcopy(cfg)
    mutant.setdefault("capabilities", {})[CAPABILITY] = value
    return mutant


# ---------------------------------------------------------------------------
# A recording cursor, so the SQL the driver sends -- or does not send -- is
# the thing under test
# ---------------------------------------------------------------------------

class RecordingCursor:
    """Records every statement and answers `SHOW CREATE TABLE` like a server.

    Deliberately self-contained rather than imported from
    `tests/test_e2_percona_wiring.py`: the claim this file makes is "no
    statement reached the cursor", and it should not be able to break because
    another test file reshaped its fixture.

    The answer is keyed on the STATEMENT rather than queued, because
    `maintain()` is called once per churn fraction and a queue drained by the
    ALTER and the COMMIT in between would hand the second call a `None` and
    send it down the "could not read the index back" fall-back -- a different
    code path, reached by accident, in the arm that is supposed to be today's
    behaviour.
    """

    def __init__(self, show_create: Optional[Any] = None):
        self.statements: List[str] = []
        self._show_create = show_create
        self._last: Any = None

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self._last = (self._show_create
                      if str(sql).upper().startswith("SHOW CREATE TABLE")
                      else None)

    def executemany(self, sql, rows):
        self.statements.append(sql)

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last or []

    def close(self):
        pass

    @property
    def sql(self) -> str:
        return "\n".join(self.statements)


#: What a live server answers `SHOW CREATE TABLE` with. The options differ
#: from the spec below on purpose -- `maintain()` is supposed to rebuild from
#: the SERVER's option list -- which also means a mutant that reaches the
#: ALTER is visible in the statement text and not only in its presence.
SHOW_CREATE = (
    "t1",
    "CREATE TABLE `t1` (\n"
    "  `id` bigint unsigned NOT NULL,\n"
    "  `v` vector(8) NOT NULL,\n"
    "  PRIMARY KEY (`id`),\n"
    "  VECTOR KEY `vi` (`v`) USING `hnsw` "
    "(`M`=24, `ef_construction`=320, `metric`=`cosine`)\n"
    ") ENGINE=InnoDB",
)


def percona_driver(bulk_build: Optional[bool] = None) -> PerconaDriver:
    """A `PerconaDriver` wired to a recording cursor.

    `bulk_build=None` declares nothing, which is the case that has to be
    today's behaviour exactly: a hand-run harness, a `run-benchmark.sh lab`
    script and every test written before this change construct the driver
    with no capability at all.
    """
    driver = PerconaDriver(ConnectionSpec(host="h", port=3306))
    driver._cur = RecordingCursor(SHOW_CREATE)
    driver._index = IndexSpec(dim=8, m=16, metric="euclidean",
                              build_mode="post")
    if bulk_build is not None:
        driver.declare_bulk_vector_index_build(bulk_build)
    return driver


def statements(driver) -> List[str]:
    """Every statement the driver sent, which is where the proof lives."""
    return list(driver._cur.statements)


# ---------------------------------------------------------------------------
# One reading of one fact, on the host
# ---------------------------------------------------------------------------

class TestTheHostReadsTheCapabilityOnce:

    def test_the_capability_decides_and_absence_reads_as_working(self):
        from orchestrator.cli import _bulk_vector_index_build
        cfg = _engine_cfg()
        honest = _bulk_vector_index_build(cfg)
        mutant = _bulk_vector_index_build(_with_capability(cfg, True))
        print(f"HONEST (percona.yml as it is) = {honest}")
        print(f"MUTANT (capability flipped)   = {mutant}")
        assert honest != mutant, (
            f"DEAD PROBE: the reading did not move when the capability was "
            f"flipped; both arms returned {honest}")
        assert honest is False
        assert mutant is True
        # And the default-when-absent, which is what keeps six engines still.
        assert _bulk_vector_index_build({}) is True
        assert _bulk_vector_index_build({"capabilities": None}) is True

    def test_the_key_is_read_in_exactly_one_place(self):
        """The drift guard, and the reason `_bulk_vector_index_build` exists.

        `_ops_build_modes` read `caps.get("bulk_vector_index_build", True)`
        TWICE, with the default spelled out in both; the ops harness's flag
        would have been a third. A default that appears N times is a default
        that can disagree with itself, and this repo's worst failures are
        exactly that shape -- a fact read two ways, each plausible.
        """
        with open(os.path.join(REPO, "orchestrator", "cli.py")) as fh:
            src = fh.read()
        # BOTH quote styles, and as an AST walk over string constants rather
        # than a text count: the double-quote-only version of this test would
        # not have seen a second reader spelled `caps.get('bulk_vector_...')`,
        # which is the same key and the same drift.
        tree = ast.parse(src)
        literals = [n for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and n.value == CAPABILITY]
        reads = len(literals)
        print(f"{CAPABILITY!r} string constants in cli.py = {reads} "
              f"(line(s) {[n.lineno for n in literals]}); "
              f"raw text count, both quote styles = "
              f"{src.count(chr(34) + CAPABILITY + chr(34))} double + "
              f"{src.count(chr(39) + CAPABILITY + chr(39))} single")
        assert reads == 1, (
            f"the capability key should be spelled once, inside "
            f"`_bulk_vector_index_build`; found {reads} string constants, so "
            f"two readings of one fact can now drift")
        enclosing = [n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef)
                     and any(c is literals[0] for c in ast.walk(n))]
        assert [n.name for n in enclosing] == ["_bulk_vector_index_build"], (
            f"the one reader must be `_bulk_vector_index_build`; it is in "
            f"{[n.name for n in enclosing]}")

    def test_the_build_mode_default_still_comes_off_the_same_reading(self):
        """The two consumers must not be able to disagree about one engine.

        `maintain()`'s ALTER and `create_index`'s ALTER are the same
        statement, so an engine whose bulk build is broken must both lose the
        `post` default and withdraw its repair path -- from one key.
        """
        from orchestrator.cli import _bulk_vector_index_build, _ops_build_modes
        cfg = _engine_cfg()
        for value, want_modes in ((False, ["incremental"]), (True, ["post"])):
            variant = _with_capability(cfg, value)
            modes = _ops_build_modes("percona", {"ops": {}}, variant)
            reading = _bulk_vector_index_build(variant)
            print(f"capability={value}: reading={reading}, modes={modes}")
            assert reading is value
            assert modes == want_modes


# ---------------------------------------------------------------------------
# The flag, which is how the decision crosses the container boundary
# ---------------------------------------------------------------------------

class TestTheDecisionIsPassedInAndNotImported:

    def test_the_harness_does_not_import_the_orchestrator(self):
        """Stated as an invariant in CLAUDE.md and enforced elsewhere; re-read
        here because this change is the one that was tempted to break it. The
        capability lives in `config/engines/*.yml`, which the harness
        container does not mount.

        Checked on the IMPORT STATEMENTS via the AST rather than on the text:
        both files I changed now mention `orchestrator/cli.py` in prose,
        because the comment's whole job is to say where the decision is made,
        and a substring test would either fail on that or be widened until it
        proved nothing.
        """
        for name in ("main.py", os.path.join("drivers", "mysql_family.py")):
            path = os.path.join(REPO, "harness", name)
            with open(path) as fh:
                tree = ast.parse(fh.read())
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported += [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            offenders = [mod for mod in imported
                         if mod.split(".")[0] == "orchestrator"]
            print(f"harness/{name}: orchestrator imports = {offenders}")
            assert not offenders, (
                f"harness/{name} imports {offenders}; it runs in a container "
                f"that mounts only harness/, so this is a "
                f"ModuleNotFoundError right after the server comes up")

    def _argv(self, bulk_build: Optional[bool] = None) -> List[str]:
        kwargs = ({} if bulk_build is None
                  else {"bulk_vector_index_build": bulk_build})
        return ops_pass.harness_args({"name": "t", "k": 10, "ops": {}}, 16,
                                     "percona", None, "normalized", {},
                                     **kwargs)

    def test_the_flag_is_emitted_only_when_the_bulk_build_is_broken(self):
        honest = FLAG in self._argv(False)
        mutant = FLAG in self._argv(True)
        print(f"HONEST (bulk build broken) = {FLAG} emitted: {honest}")
        print(f"MUTANT (bulk build works)  = {FLAG} emitted: {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the emission did not move with the parameter")
        assert honest is True
        assert mutant is False

    def test_absence_of_the_parameter_is_todays_behaviour(self):
        """The default matters more than it looks: every existing caller of
        `harness_args` -- `tests/test_config.py`, `tests/test_estimate.py`,
        `tests/test_t3c_filtered.py`, `tests/test_t4a_churn.py` -- omits it,
        and so does any hand-built call. They are all about engines whose bulk
        build works, and they must emit exactly what they emitted before.
        """
        assert FLAG not in self._argv()

    @pytest.mark.parametrize("engine", OTHER_ENGINES)
    def test_no_other_engine_gets_the_flag(self, engine):
        """Derived from each engine's own config, not from a literal: an
        engine that declares nothing reads as "the bulk build works".
        """
        from orchestrator.cli import _bulk_vector_index_build
        cfg = _engine_cfg(engine)
        assert CAPABILITY not in (cfg.get("capabilities") or {}), engine
        argv = ops_pass.harness_args(
            {"name": "t", "k": 10, "ops": {}}, 16, engine, None, "normalized",
            {}, bulk_vector_index_build=_bulk_vector_index_build(cfg))
        print(f"{engine}: {FLAG} emitted: {FLAG in argv}")
        assert FLAG not in argv, engine

    @pytest.mark.parametrize("engine,want_flag",
                             [("percona", True), ("mariadb", False)])
    def test_run_unit_puts_the_flag_on_the_real_command_line(
            self, engine, want_flag, tmp_path):
        """**BEHAVIOURAL. The source-string version of this test could not
        fail on the break its own subject has, and that was measured.**

        It asserted `"bulk_build = _bulk_vector_index_build(engine_cfg)" in
        src` -- and that line occurs TWICE in `cli.py`, once inside
        `_ops_build_modes` and once inside `_run_unit`. Replacing the
        `_run_unit` one with `bulk_build = True` left the whole suite
        byte-identical (3264 passed, 5 skipped, exit 0) while percona's argv
        lost the flag entirely. `--build-mode incremental` still comes off
        `_ops_build_modes`, so the bulk build was still avoided and the ONLY
        thing that changed was the churn repair: `declare(True)` ->
        `repair_path = "index-replace"` -> the ALTER -> `after ALTER TABLE
        (index replace) (0.0 s)` published again, on `main`, `full` and
        `smoke`, all of which run churn on all seven engines. The exact
        defect this change exists to remove, silently, with a green suite.

        This repo has already paid for that lesson **in this same function**:
        `tests/test_t3c_filtered.py` converted its axis pin from a substring
        test to an AST walk because a genuine fourth looped axis was once
        added to `_run_unit` with all three original loop strings intact and
        the suite stayed green.

        So this asserts on what the runner RECEIVES. `OpsRun` is a fake whose
        `run_harness` records the argv and returns 0 -- no container, no
        network, sub-second, modelled on
        `tests/test_t3c_filtered.py::test_run_unit_puts_the_flags_on_the_real
        _command_line`.

        Honest: percona's every unit carries the flag; mariadb's carries none.
        Mutant (`_run_unit`'s `bulk_build` hard-wired to True, or the keyword
            dropped from the `harness_args` call): percona's units lose it and
            this fails naming the engine.
        """
        import argparse

        from orchestrator import cli as cli_mod
        from orchestrator.config import load_engine, load_profile, \
            load_resources

        seen = []

        class FakeRun:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_harness(self, args, output, **kw):
                seen.append(list(args))
                return 0

        paths = {"run_dir": str(tmp_path),
                 "checkpoints": str(tmp_path / "checkpoints")}
        with mock.patch.object(ops_pass, "OpsRun", FakeRun):
            rc = cli_mod._run_unit(
                "ops", engine, "glove-100-angular", load_profile("main"),
                load_engine(engine), load_resources("normalized"), None,
                "normalized", paths, "run-0",
                argparse.Namespace(force=False), checkpoints=None)
        assert rc == 0
        assert seen, "run_harness was never reached"
        emitted = [FLAG in argv for argv in seen]
        print(f"{engine}: {len(seen)} unit(s), {FLAG} present = {emitted}")
        assert all(present is want_flag for present in emitted), (
            f"{engine}: expected {FLAG} present={want_flag} on every unit, "
            f"got {emitted}")
        # Non-vacuity: the units really were built, with the build mode the
        # capability also decides, so a run that emitted nothing at all
        # cannot pass this.
        for argv in seen:
            assert "--build-mode" in argv
        if engine == "percona":
            modes = {argv[argv.index("--build-mode") + 1] for argv in seen}
            assert modes == {"incremental"}, modes

    def test_the_reading_is_derived_inside_the_run_loop_not_assumed(self):
        """The structural half, scoped by AST to `_run_unit` alone.

        `bulk_build = _bulk_vector_index_build(engine_cfg)` appears in two
        functions, so a file-level substring test matches the wrong one. This
        walks `_run_unit`'s own body and requires both halves there: the call
        that derives the value, and the keyword that hands it on.
        """
        with open(os.path.join(REPO, "orchestrator", "cli.py")) as fh:
            tree = ast.parse(fh.read())
        run_unit = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "_run_unit")
        derives = [n for n in ast.walk(run_unit)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name)
                   and n.func.id == "_bulk_vector_index_build"]
        passes = [kw for n in ast.walk(run_unit)
                  if isinstance(n, ast.Call)
                  for kw in n.keywords
                  if kw.arg == "bulk_vector_index_build"]
        print(f"_run_unit: derives={len(derives)}, passes={len(passes)}")
        assert len(derives) == 1, (
            "_run_unit must derive the reading from the engine config "
            f"itself; found {len(derives)} call(s)")
        assert len(passes) == 1, (
            "the derived value has to reach `ops_pass.harness_args`; found "
            f"{len(passes)} keyword(s)")
        handed_on = passes[0].value
        assert isinstance(handed_on, ast.Name), (
            f"expected the derived local to be handed on, got "
            f"{ast.dump(handed_on)} -- a literal here is the mutant that "
            f"survived the substring version of this test")

    def test_the_harness_accepts_the_flag_and_defaults_to_off(self):
        from harness.main import parse_args
        base = ["--engine", "percona", "--dataset", "d", "--m", "16",
                "--run-id", "r", "--host", "h", "--port", "3306",
                "--output", "/tmp/out.jsonl"]
        honest = parse_args(base + [FLAG]).no_bulk_vector_index_build
        mutant = parse_args(base).no_bulk_vector_index_build
        print(f"HONEST (flag given)   = {honest}")
        print(f"MUTANT (flag omitted) = {mutant}")
        assert honest != mutant, "DEAD PROBE: the flag changes nothing"
        assert honest is True and mutant is False

    def test_main_applies_the_flag_to_the_driver_it_built(self, tmp_path):
        """**BEHAVIOURAL, and the source-string version it replaced was a
        hole big enough to drive the whole defect back through.**

        The hop under test is `main()` -> `driver_factory()` ->
        `declare_bulk_vector_index_build`. `get_driver` takes only the
        connection spec and the driver class name, so the capability is
        applied after construction, through `getattr`.

        The version of this test that read two substrings out of
        `driver_factory`'s source was measured to survive
        `if declare is not None and False:` -- both strings still present, the
        flag still emitted and still parsed, the declaration never applied,
        and the whole suite identical at 3264 passed / 5 skipped / exit 0.
        **Nothing else in this repo exercises `driver_factory` behaviourally:**
        the four places that patch `get_driver` and run `main()` supply fakes
        with no hook, so `getattr` returns None and the `if` never mattered.

        No server and no container: a `PerconaDriver` subclass stubs the six
        methods `main()`'s no-build path touches and keeps the real
        `maintain()`, `repair_path` and declaration hook -- the three things
        under test. `--workloads ""` gives `workloads = []`, so
        `_should_calibrate` is False and no workload runs; `main()` returns 0
        after connect/count_rows/set_ef_search/explain/close.

        Honest (flag given):   driver.repair_path == "none"
        Mutant (flag omitted): driver.repair_path == "index-replace"
        """
        from harness import main as main_mod

        built = []

        class StubPercona(PerconaDriver):
            """The real declaration path; everything else stubbed.

            A subclass rather than a mock, so `declare_bulk_vector_index_build`,
            `repair_path` and `maintain` are the shipped implementations and
            not a test's restatement of them.
            """

            def connect(self):
                self._cur = RecordingCursor(SHOW_CREATE)

            def close(self):
                pass

            def server_version(self):
                return "percona-stub"

            def count_rows(self):
                return N_ROWS

            def set_ef_search(self, ef):
                pass

            def explain_uses_vector_index(self, vector, k):
                return True

        def factory(*a, **kw):
            driver = StubPercona(ConnectionSpec(host="h", port=3306))
            built.append(driver)
            return driver

        def run(with_flag):
            del built[:]
            argv = ["--engine", "percona", "--driver", "PerconaDriver",
                    "--host", "h", "--port", "3306", "--dataset", "tiny",
                    "--m", "16", "--run-id", "e13",
                    "--output", str(tmp_path / f"ops-{with_flag}.jsonl"),
                    "--cache-dir", str(tmp_path / f"cache-{with_flag}"),
                    "--workloads", ""]
            if with_flag:
                argv.append(FLAG)
            with mock.patch.object(main_mod, "load_dataset",
                                   lambda *a, **kw: tiny_dataset()), \
                 mock.patch.object(main_mod, "get_driver", factory):
                rc = main_mod.main(argv)
            assert rc == 0, f"main() exited {rc} with flag={with_flag}"
            assert built, "get_driver was never reached"
            return built[0]

        honest = run(True)
        mutant = run(False)
        print(f"HONEST (flag given)   = repair_path "
              f"{honest.repair_path!r}, "
              f"bulk_vector_index_build={honest.bulk_vector_index_build}")
        print(f"MUTANT (flag omitted) = repair_path "
              f"{mutant.repair_path!r}, "
              f"bulk_vector_index_build={mutant.bulk_vector_index_build}")
        assert honest.repair_path != mutant.repair_path, (
            "DEAD PROBE: the driver main() built came out the same either "
            "way, so this test does not observe the declaration being "
            "applied")
        assert honest.repair_path == "none"
        assert honest.bulk_vector_index_build is False
        assert honest.maintain() is None
        assert statements(honest) == [], statements(honest)
        assert mutant.repair_path == "index-replace"
        assert mutant.bulk_vector_index_build is True
        assert mutant.maintain() == "ALTER TABLE (index replace)"

    def test_the_declaration_is_applied_inside_the_driver_factory(self):
        """Structural, and scoped to the nested `def` by AST walk rather than
        by substring, for the reason `tests/test_t3c_filtered.py`'s axis pin
        was converted to an AST walk: a substring test on a function's source
        cannot tell which copy of a repeated line it matched.

        What this adds over the behavioural test above is the PLACE. Every
        client connection goes through `driver_factory` -- the writer, the
        concurrency pool and the churn readers -- and applying it anywhere
        else would leave the readers' drivers disagreeing with the writer's
        while `main()`'s own driver still looked right.
        """
        with open(os.path.join(REPO, "harness", "main.py")) as fh:
            tree = ast.parse(fh.read())
        factory = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "driver_factory")
        calls = [n for n in ast.walk(factory) if isinstance(n, ast.Call)]
        getattrs = [c for c in calls
                    if isinstance(c.func, ast.Name) and c.func.id == "getattr"
                    and any(isinstance(a, ast.Constant) and a.value == HOOK
                            for a in c.args)]
        print(f"getattr({HOOK!r}) call sites inside driver_factory = "
              f"{len(getattrs)}")
        assert len(getattrs) == 1, (
            f"expected exactly one {HOOK} lookup inside driver_factory; "
            f"found {len(getattrs)}")
        # And it is guarded by nothing but the None check -- an `and False`
        # or any extra condition would be a second test on that `if`.
        guards = [n.test for n in ast.walk(factory) if isinstance(n, ast.If)]
        assert len(guards) == 1, f"expected one if in driver_factory, {guards}"
        assert isinstance(guards[0], ast.Compare), (
            "the declaration must be guarded by the None check alone; a "
            "compound condition is how `and False` hid from the source-string "
            "version of this test")

    def test_every_engine_that_declares_it_has_a_driver_that_can_act_on_it(
            self):
        """The wiring fault, caught in the suite instead of in a run.

        `harness/main.py` applies the capability through `getattr`, so a
        driver without the hook silently ignores it. There is deliberately no
        runtime warning -- it would fire once per client connection, up to 32
        in the concurrency pool, for what is a wiring fault -- and this is the
        check that replaces it. If a second engine ever declares the
        capability false, it fails here rather than publishing a repair that
        did not happen.
        """
        from harness.drivers.postgres import _driver_classes
        from orchestrator import engines as engines_mod
        from orchestrator.cli import KNOWN_ENGINES
        classes = _driver_classes()
        declaring, actionable = [], []
        for engine in sorted(KNOWN_ENGINES):
            cfg = _engine_cfg(engine)
            if (cfg.get("capabilities") or {}).get(CAPABILITY, True):
                continue
            declaring.append(engine)
            cls = classes[engines_mod.get(engine).driver]
            if hasattr(cls, HOOK):
                actionable.append(engine)
        print(f"engines declaring {CAPABILITY}: false = {declaring}")
        print(f"of those, drivers carrying {HOOK}() = {actionable}")
        assert declaring == ["percona", "villagesql"], declaring
        assert actionable == declaring, (
            f"{sorted(set(declaring) - set(actionable))} declare the bulk "
            f"build broken but their driver cannot act on it, so the "
            f"declaration reaches nothing and whatever repair that driver "
            f"runs is still published as a repair")


# ---------------------------------------------------------------------------
# The driver withdraws the repair path, and sends nothing
# ---------------------------------------------------------------------------

class TestTheDriverDeclaresNoRepairPath:

    def test_the_repair_path_is_withdrawn_on_the_instance(self):
        honest = percona_driver(False).repair_path
        mutant = percona_driver(True).repair_path
        print(f"HONEST (bulk build broken) = repair_path {honest!r}")
        print(f"MUTANT (bulk build works)  = repair_path {mutant!r}")
        assert honest != mutant, (
            "DEAD PROBE: the repair path did not move with the declaration")
        assert honest == EngineDriver.repair_path == "none", (
            "it has to be the base class's own default, which is the value "
            "the five engines with no repair path already record")
        assert mutant == "index-replace"

    def test_the_class_still_declares_the_mechanism_it_has(self):
        """The class attribute is NOT changed, and that is deliberate.

        `index-replace` IS this engine's repair mechanism -- `docs/10-rollout
        .md` names it as one of the two engines that have one, and
        `tests/test_e3_rollout_doc.py` and `tests/test_t4a_churn.py` assert
        it off the class. What the pinned server cannot do is EXECUTE it. The
        instance carries what this run could actually measure; the class
        carries what the engine is.
        """
        assert PerconaDriver.repair_path == "index-replace"
        assert MariaDBDriver.repair_path == "none"
        assert AliSQLDriver.repair_path == "none"
        withdrawn = percona_driver(False)
        assert withdrawn.repair_path == "none"
        assert type(withdrawn).repair_path == "index-replace"
        assert "repair_path" in withdrawn.__dict__, (
            "the withdrawal has to be an instance attribute shadowing the "
            "class one, not a mutation of the class -- a class-level write "
            "would leak into every other driver instance in the process")

    def test_maintain_returns_none_and_the_alter_never_reaches_the_cursor(
            self):
        honest = percona_driver(False)
        mutant = percona_driver(True)
        honest_what, mutant_what = honest.maintain(), mutant.maintain()
        print(f"HONEST (bulk build broken) = maintain() -> {honest_what!r}, "
              f"{len(statements(honest))} statement(s)")
        print(f"MUTANT (bulk build works)  = maintain() -> {mutant_what!r}, "
              f"{len(statements(mutant))} statement(s)")
        assert (honest_what, statements(honest)) != \
               (mutant_what, statements(mutant)), "DEAD PROBE"
        assert honest_what is None, (
            "a non-None return is what `churn.py` reads as a repair having "
            "run, and the whole defect is that something was read that way")
        assert statements(honest) == [], (
            f"nothing may reach the cursor -- not the ALTER and not the "
            f"SHOW CREATE TABLE that prepares it; got "
            f"{statements(honest)}")
        assert "ALTER" not in honest._cur.sql.upper()
        assert "ADD VECTOR KEY" not in honest._cur.sql.upper()
        # The mutant is the behaviour that must survive untouched for a fixed
        # server, down to the statement text.
        assert mutant_what == "ALTER TABLE (index replace)"
        alter = [s for s in statements(mutant) if s.startswith("ALTER")]
        assert len(alter) == 1 and "DROP KEY vi" in alter[0]
        assert "ADD VECTOR KEY" in alter[0]

    def test_declaring_nothing_is_todays_behaviour_exactly(self):
        """The class-level default, which is what a hand-run harness and a
        `lab` script get. It is the same reading the host gives an absent
        capability, so the diagnostic path needs no flag and no new profile
        key to attempt the repair on purpose.
        """
        bare = percona_driver(None)
        assert bare.bulk_vector_index_build is True
        assert bare.repair_path == "index-replace"
        assert bare.maintain() == "ALTER TABLE (index replace)"
        assert any(s.startswith("ALTER") for s in statements(bare))

    def test_the_declaration_is_reversible(self):
        """Not a hypothetical: the hook is called once per driver instance and
        a reader pool builds several. A withdrawal that could not be undone
        would make the hook order-dependent.
        """
        driver = percona_driver(False)
        assert driver.repair_path == "none"
        driver.declare_bulk_vector_index_build(True)
        assert driver.repair_path == "index-replace"
        assert driver.maintain() == "ALTER TABLE (index replace)"

    def test_the_two_mysql_family_engines_have_no_such_hook(self):
        """MariaDB and AliSQL have no repair path to withdraw, so they must
        not acquire a hook that implies they do.
        """
        for cls in (MariaDBDriver, AliSQLDriver):
            assert not hasattr(cls, HOOK), cls.__name__
            assert cls.repair_path == "none", cls.__name__


# ---------------------------------------------------------------------------
# No "after" record is written at all -- through the real churn workload
# ---------------------------------------------------------------------------

class _PerconaRepairFake(FakeDriver):
    """A loadable fake whose repair DECISION is the real driver's.

    `PerconaDriver` cannot be driven through `churn.run` without a server --
    it loads and queries over the MySQL protocol -- and a fake that simply
    returned `None` from `maintain()` would be a fixture asserting itself.
    So the two things under test, `repair_path` and `maintain()`, are
    delegated to a genuine `PerconaDriver` wired to a recording cursor, and
    everything the workload needs in order to load and measure comes from
    `FakeDriver`. The cursor is then the evidence: it holds every statement
    the real driver sent.
    """

    def __init__(self, *args, bulk_build: Optional[bool] = None, **kwargs):
        self.percona = percona_driver(bulk_build)
        super().__init__(*args, **kwargs)
        # After `super().__init__`, which sets `repair_path` from the fake's
        # own knob. The workload stamps this onto the "before" record BEFORE
        # `maintain()` runs, so it has to be the real driver's answer.
        self.repair_path = self.percona.repair_path

    def maintain(self):
        self.churn_hook_calls.append(("maintain", None))
        return self.percona.maintain()


def _loaded(dataset, bulk_build: Optional[bool]) -> _PerconaRepairFake:
    from harness import datasets as ds
    driver = _PerconaRepairFake(dim=dataset.dim, metric=dataset.metric,
                                bulk_build=bulk_build)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, ds.assign_tags(N_ROWS))
    return driver


def _run_churn(tmp_path, dataset, bulk_build: Optional[bool],
               fractions: Sequence[float] = (0.10, 0.25)):
    """One churn phase, and everything it wrote."""
    recorder = records.Recorder(str(tmp_path / f"ops-{bulk_build}.jsonl"))
    ctx = RunContext(run_id="e13", resource_pass="normalized",
                     recorder=recorder, engine="percona", engine_tag="pin",
                     march="x86-64-v3", k=K,
                     cache_dir=str(tmp_path / f"cache-{bulk_build}"))
    driver = _loaded(dataset, bulk_build)
    try:
        cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
               fractions=fractions, max_queries=N_QUERIES,
               indexed_rows=N_ROWS)
    finally:
        recorder.close()
    written = list(records.read_records(recorder.path))
    return driver, written


def _states(written) -> List[Optional[str]]:
    return [(r.get("extra") or {}).get("maintenance_state") for r in written
            if (r.get("extra") or {}).get("maintenance_state")]


@pytest.fixture
def dataset():
    return tiny_dataset()


class TestNoPostMaintenanceRecordIsWrittenAtAll:
    """The published artifact is the thing that was wrong, so it is the thing
    asserted on.

    Not "the `after` record carries a sentinel" and not "the report hides
    it": there must be NO second record. `churn.py`'s `_record_maintenance`
    returns before it measures anything when `maintain()` gives back a
    non-`str`, which is the same path mariadb, alisql, mongodb and valkey
    already take.
    """

    def test_the_after_record_is_absent_and_the_cursor_saw_nothing(
            self, tmp_path, dataset):
        driver, written = _run_churn(tmp_path, dataset, False)
        states = _states(written)
        print(f"HONEST (bulk build broken) = maintenance_state values "
              f"{states}, cursor statements {statements(driver.percona)}")
        assert states, "non-vacuity: the churn phase wrote no step records"
        assert set(states) == {"before"}, states
        assert "after" not in states, (
            "an `after` record is the degraded index measured a second time "
            "and labelled repaired")
        # Exactly one step record per fraction, the shape an engine with no
        # repair path produces.
        assert len(states) == 2, states
        assert statements(driver.percona) == [], (
            f"the ALTER must never reach the cursor; got "
            f"{statements(driver.percona)}")
        for record in written:
            extra = record.get("extra") or {}
            if "maintenance_state" not in extra:
                continue
            assert extra["repair_path"] == "none", extra
            assert "maintenance" not in extra, (
                f"`maintenance` names what ran, and nothing ran: {extra}")
            assert "maintenance_s" not in extra, (
                f"a duration for a statement that was never issued: {extra}")

    def test_the_mutant_writes_exactly_what_was_published_before(
            self, tmp_path, dataset):
        """THE PROBE'S OTHER ARM, and the measured defect reproduced.

        With the capability granted, the same workload over the same corpus
        writes an `after` record per fraction carrying
        `maintenance: "ALTER TABLE (index replace)"` -- which is the row the
        42-hour run published ten times. An identical pair with the test
        above would mean neither is measuring the fix.
        """
        driver, written = _run_churn(tmp_path, dataset, True)
        states = _states(written)
        alters = [s for s in statements(driver.percona)
                  if s.startswith("ALTER")]
        print(f"MUTANT (bulk build works) = maintenance_state values "
              f"{states}, ALTERs sent {len(alters)}")
        assert states.count("after") == 2, states
        after = [r for r in written
                 if (r.get("extra") or {}).get("maintenance_state") == "after"]
        for record in after:
            extra = record["extra"]
            assert extra["maintenance"] == "ALTER TABLE (index replace)"
            assert extra["repair_path"] == "index-replace"
            assert "maintenance_s" in extra
        assert alters, "the ALTER has to reach the cursor in this arm"

    def test_the_before_record_is_unchanged_by_the_withdrawal(
            self, tmp_path, dataset):
        """What the fix must NOT cost: the pre-maintenance measurement is the
        churn result and it is written before `maintain()` is called at all,
        which is what keeps a completed measurement on disk when a repair
        path fails. Withdrawing the repair must not take the churn number
        with it.

        ONE fraction, and the reason is the fake rather than the fix: its
        recall comes off a seeded generator, and the mutant arm consumes
        extra draws measuring the repaired state. From the second fraction on
        the two arms' "before" numbers legitimately diverge, so comparing
        them would be asserting on the fixture's RNG stream. The first
        fraction's record is written before `maintain()` is reached in either
        arm, which is exactly the property under test.
        """
        broken, written_broken = _run_churn(tmp_path, dataset, False,
                                            fractions=(0.10,))
        works, written_works = _run_churn(tmp_path, dataset, True,
                                          fractions=(0.10,))
        before_broken = [r for r in written_broken
                         if (r.get("extra") or {}).get("maintenance_state")
                         == "before"]
        before_works = [r for r in written_works
                        if (r.get("extra") or {}).get("maintenance_state")
                        == "before"]
        print(f"before-records: broken={len(before_broken)}, "
              f"works={len(before_works)}")
        assert len(before_broken) == len(before_works) == 1
        for a, b in zip(before_broken, before_works):
            assert a["churn_fraction"] == b["churn_fraction"]
            assert a["recall_at_k"] == b["recall_at_k"], (
                "the churn measurement itself is seeded and must be "
                "identical; only the repair record may differ")
            assert a["extra"]["rows_churned"] == b["extra"]["rows_churned"]


# ---------------------------------------------------------------------------
# The evidence lives where the code is
# ---------------------------------------------------------------------------

class TestTheDeclarationCanBeAuditedAndRetired:

    def test_the_driver_records_why_it_withdrew_the_repair(self):
        """R7, as `tests/test_e9_percona_bulk_build_crash.py` applies it to
        the engine config: a bare early return would be deleted by the next
        person who wondered why, so the run, the number and the condition for
        restoring it have to be written where the code is.
        """
        doc = getattr(PerconaDriver, HOOK).__doc__
        assert doc
        for needle in ("full-20261015-215006", "1-4 ms", "281,018,368",
                       "20 pairs", "five decimals", "ALGORITHM=INPLACE",
                       "fashion-mnist-784", "sift-128",
                       CAPABILITY, "no repair path"):
            assert needle in doc, (
                f"{needle!r} is missing from {HOOK}()'s docstring; without "
                f"it the withdrawal cannot be audited or retired")

    def test_the_evidence_does_not_carry_the_three_corrected_numbers(self):
        """The three figures a review found wrong, pinned so they cannot come
        back. All three understated the evidence, which is why none of them
        was caught by reading the argument.

        * "ten pairs" -- there are 20 (10 ops units x 2 fractions).
        * "returned in 1 ms" -- 1-4 ms; 0.001 x17, 0.002 x2, 0.004 x2.
        * "sift-128 ... a 268 MiB HNSW graph" -- 268 MiB (281,018,368 B) is
          fashion-mnist-784 at 60,000 rows. sift-128 at 1,000,000 rows is
          808-1,428 MiB. The 26-80 minute figure IS sift's and is correct;
          only the graph size was misattributed.
        """
        doc = getattr(PerconaDriver, HOOK).__doc__
        module_doc = sys.modules[__name__].__doc__
        with open(os.path.join(REPO, "harness", "drivers",
                               "mysql_family.py")) as fh:
            source = fh.read()
        head = source.index("def maintain(self)")
        runtime_print = source[head:source.index("clause =", head)]
        for where, text in (("the hook's docstring", doc),
                            ("this module's docstring", module_doc),
                            ("maintain()'s runtime print", runtime_print)):
            for stale in ("ten pairs", "in 1 ms", "returned in 1 ms"):
                assert stale not in text, (
                    f"{stale!r} is back in {where}; it understates the "
                    f"measurement")
        # And the one that is a misattribution rather than a smaller number:
        # 268 MiB must not be claimed for sift anywhere.
        assert "sift-128 at 1,000,000\n          rows, where a genuine " \
               "rebuild of a 268 MiB" not in doc
        for text in (doc, module_doc):
            flat = " ".join(text.split())
            assert "268 MiB" not in flat or "fashion-mnist-784" in flat, (
                "268 MiB is fashion-mnist-784's graph; naming it without "
                "naming that dataset is the misattribution this pins")

    def test_maintain_says_the_gate_exists_before_the_statement(self):
        doc = PerconaDriver.maintain.__doc__
        assert doc and CAPABILITY in doc, (
            "`maintain()`'s docstring describes a rebuild that no longer "
            "always happens; it has to name the gate")

    def test_the_cited_base_class_default_is_where_it_says(self):
        """A line citation written in this change, re-derived rather than
        trusted: six have rotted in this series.
        """
        doc = getattr(PerconaDriver, HOOK).__doc__
        assert "harness/drivers/base.py:258" in doc, (
            "the citation moved or was dropped; re-derive it with grep -n")
        with open(os.path.join(REPO, "harness", "drivers", "base.py")) as fh:
            line = fh.read().splitlines()[257]
        print(f"base.py:258 = {line!r}")
        assert 'repair_path: str = "none"' in line, (
            f"base.py:258 no longer holds the cited declaration: {line!r}")
