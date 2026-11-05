"""VillageSQL + vsql_vector (built from vsql-vector-pk): the engine's wiring
and the behaviours that keep it honest.

No server is started here. What is pinned are the things that fail SILENTLY
against a real build:

* with plain vsql-vector (the engine's first pin, measured 2026-10-30) the
  server answers every KNN query by sorting a full scan, and a driver that
  merely warned would publish exact brute force as ANN recall/QPS -- so the
  driver and the ann module check the plan and REFUSE a query that does not
  use the index, whichever pin is built;
* vectors are sent as text, so the encoding must round-trip float32 exactly
  or every stored vector is silently perturbed;
* the extension refuses ef_construction < M and > 32 x M, so a profile's
  200 at M=6 is not buildable -- the render drops it by name and the driver
  and module refuse it, because a clamped build under the requested label is
  a result file naming a build that never existed;
* DELETE is rejected, and the bulk CREATE INDEX intermittently aborts the
  server, which the configuration and the orchestrator must route around.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any, List, Optional

import numpy
import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness.drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from harness.drivers import villagesql as vs  # noqa: E402

EXTENSION_PIN = "93ed00873dd65f64c0788c80576d5cf3fba39831"
SERVER_PIN = "533ab4b58d2744b79bb45e41f44fa928f190a655"

#: EXPLAIN FORMAT=TREE as the engine's first pin (plain vsql-vector) printed
#: it for `SELECT id FROM t1 ORDER BY L2_DISTANCE(v, '[..]') LIMIT 10` (vector
#: shortened): a sort over a table scan, the brute force the guard refuses.
MEASURED_BRUTE_FORCE_PLAN = (
    "-> Limit: 10 row(s)\n    -> Sort: vsql_vector.l2_distance(t1.v,'[-0.93,"
    "0.87]'), limit input to 10 row(s) per chunk  (cost=505 rows=5000)\n"
    "        -> Table scan on t1  (cost=505 rows=5000)\n")
#: What the pinned build printed when it drove the index, for a 201-row
#: table and LIMIT 5 (`explain_access_path.cc`: "Custom index distance scan
#: on " + key name).
INDEX_PLAN = ("-> Limit: 5 row(s)  (cost=20.4 rows=5)\n    -> Custom index "
              "distance scan on vi  (cost=20.4 rows=201)\n")


def _read(rel: str) -> str:
    with open(os.path.join(VB_ROOT, rel)) as fh:
        return fh.read()


def _config() -> dict:
    return yaml.safe_load(_read("config/engines/villagesql.yml"))


class ScriptedCursor:
    def __init__(self, plan: str = MEASURED_BRUTE_FORCE_PLAN,
                 file_sizes: Optional[List[int]] = None):
        self.statements: List[str] = []
        self.params: List[Any] = []
        self.many: List[List[tuple]] = []
        self.plan = plan
        self.file_sizes = list(file_sizes or [])
        self._rows: List[tuple] = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self.params.append(params)
        s = sql.upper()
        if s.startswith("EXPLAIN"):
            self._rows = [(self.plan,)]
        elif "INNODB_TABLESPACES" in s:
            self._rows = [(self.file_sizes.pop(0),)] if self.file_sizes else []
        elif s.startswith("SELECT ID FROM T1"):
            self._rows = [(i,) for i in range(10)]
        else:
            self._rows = []

    def executemany(self, sql, rows):
        self.statements.append(sql)
        self.many.append(list(rows))

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None

    def close(self):
        pass


def driver(build_mode="incremental", metric="euclidean", m=16, dim=4,
           ef_construction=200, **cursor):
    d = vs.VillageSQLDriver(ConnectionSpec(host="h", port=1))
    d._cur = ScriptedCursor(**cursor)
    d._index = IndexSpec(dim=dim, m=m, metric=metric,
                         ef_construction=ef_construction,
                         build_mode=build_mode)
    return d


# ---------------------------------------------------------------------------
# §1 wiring
# ---------------------------------------------------------------------------

class TestWiring:
    def test_the_config_pins_both_trees_by_commit(self):
        source = _config()["source"]
        assert source["tag"] == EXTENSION_PIN
        assert source["server_commit"] == SERVER_PIN
        assert source["server_tag"] == "publish/mysql-9.7_0.0.7"
        assert source["upstream"].endswith("/vsql-vector-pk.git")

    def test_the_driver_is_registered_by_class_name(self):
        from harness.drivers.postgres import _driver_classes, _driver_table
        assert _driver_classes()["VillageSQLDriver"] is vs.VillageSQLDriver
        assert _driver_table()["villagesql"] is vs.VillageSQLDriver
        assert _config()["runtime"]["driver"] == "VillageSQLDriver"

    def test_the_declined_workloads_agree_with_the_config(self):
        """The estimator reads the config, the harness reads the driver."""
        caps = _config()["capabilities"]
        declined = dict(vs.VillageSQLDriver.declined_workloads)
        assert set(declined) == {"filtered", "churn"}
        assert caps["filtered_search"] is False and caps["churn"] is False
        assert vs.DELETE_REJECTED in declined["churn"]

    def test_main_skips_both_and_stamps_them(self):
        from harness import main as hmain
        run, declined = hmain._apply_declined(
            vs.VillageSQLDriver(ConnectionSpec(host="h", port=1)),
            ["build", "concurrency", "filtered", "churn"])
        assert run == ["build", "concurrency"]
        assert set(declined) == {"filtered", "churn"}

    def test_the_bulk_build_is_declared_broken_so_incremental_is_the_default(self):
        from orchestrator import cli
        cfg = _config()
        assert cfg["capabilities"]["bulk_vector_index_build"] is False
        assert cli._ops_build_modes("villagesql", {}, cfg) == ["incremental"]

    def test_ann_and_ops_build_at_one_ef_construction(self):
        from orchestrator import ann_pass, ops_pass
        profile = {"ann": {"m_values": [8, 16], "ef_search": [10, 40]},
                   "ops": {}}
        body = ann_pass.render_config("villagesql", profile, {}, "normalized")
        group = body["float"]["any"][0]["run_groups"]["innodb"]
        assert group["arg_groups"] == [{"M": [8, 16], "efConstruction": [200]}]
        assert body["float"]["any"][0]["constructor"] == "VillageSQL"
        args = ops_pass.harness_args({**profile, "name": "t", "k": 10}, 16,
                                     "villagesql", None, "normalized", {})
        assert args[args.index("--ef-construction") + 1] == "200"

    def test_the_estimator_charges_neither_declined_workload(self):
        from orchestrator import estimate
        assert estimate._runs_churn("villagesql") is False
        assert estimate._runs_filtered_search("villagesql") is False
        assert estimate._runs_churn("pgvector") is True
        assert estimate._runs_churn("e26-no-such-engine") is True


# ---------------------------------------------------------------------------
# §2 DDL
# ---------------------------------------------------------------------------

class TestSchema:
    def test_incremental_declares_the_index_in_create_table(self):
        d = driver(build_mode="incremental", metric="angular")
        d.create_schema(d._index)
        ddl = d._cur.statements[-1]
        assert "v SVECTOR(4) NOT NULL" in ddl
        assert ("INDEX vi (v hnsw_cosine) USING EXTENDED(hnsw) "
                "WITH (M = 16, ef_construction = 200)") in ddl
        assert ddl.rstrip().endswith("ENGINE=InnoDB")
        assert d.capabilities()["incremental_index"] is True

    def test_post_leaves_it_out_and_create_index_adds_it(self):
        d = driver(build_mode="post", metric="euclidean",
                   file_sizes=[1_000_000, 9_000_000])
        d.create_schema(d._index)
        assert "USING EXTENDED" not in d._cur.statements[-1]
        assert d.capabilities()["incremental_index"] is False
        d.create_index(d._index)
        assert ("CREATE INDEX vi ON t1 (v hnsw_l2) USING EXTENDED(hnsw) "
                "WITH (M = 16, ef_construction = 200)") in d._cur.statements
        assert d.index_bytes() == 8_000_000

    def test_incremental_create_index_is_a_noop_and_reports_no_size(self):
        d = driver(build_mode="incremental")
        before = list(d._cur.statements)
        d.create_index(d._index)
        assert d._cur.statements == before
        assert d.index_bytes() == 0

    @pytest.mark.parametrize("m", [7, 8, 16, 128])
    def test_a_buildable_ef_construction_is_sent_as_asked(self, m):
        d = driver(m=m)
        d.create_schema(d._index)
        assert "ef_construction = 200)" in d._cur.statements[-1]
        assert d.capabilities()["ef_construction"] == 200

    @pytest.mark.parametrize("m,what", [(6, "ef_construction=200"),
                                        (4, "ef_construction=200"),
                                        (129, "M=129")])
    def test_an_unbuildable_one_is_refused_not_clamped(self, m, what):
        """200 at M=6 used to become 192 under a label saying 200."""
        d = driver(m=m)
        with pytest.raises(ValueError, match=rf"refuses {what}"):
            d.create_schema(d._index)
        assert not any("CREATE TABLE" in s for s in d._cur.statements)

    def test_effective_ef_is_floored_at_k(self):
        d = driver()
        assert d.effective_ef(5, 10) == 10
        assert d.effective_ef(40, 10) == 40


# ---------------------------------------------------------------------------
# §3 text binding
# ---------------------------------------------------------------------------

class TestTextBinding:
    def test_the_literal_round_trips_every_float32_exactly(self):
        rng = numpy.random.RandomState(7)
        v = numpy.concatenate([
            rng.normal(size=500).astype("<f4"),
            numpy.array([0.0, -0.0, 1e-30, 3.4e38, 255.0, 1 / 3],
                        dtype="<f4")])
        text = vs.text_literal(v)
        assert text.startswith("[") and text.endswith("]")
        back = numpy.array([numpy.float32(t) for t in text[1:-1].split(",")],
                           dtype="<f4")
        assert back.tobytes() == v.tobytes()

    def test_only_number_characters_reach_the_statement(self):
        text = vs.text_literal(numpy.random.RandomState(1).normal(size=64))
        assert set(text) <= set("0123456789.-+e[],")

    @pytest.mark.parametrize("bad", [numpy.nan, numpy.inf, -numpy.inf])
    def test_a_non_finite_element_is_refused(self, bad):
        with pytest.raises(ValueError):
            vs.text_literal([1.0, bad])

    def test_load_and_reinsert_bind_text_not_bytes(self):
        d = driver()
        x = numpy.arange(12, dtype="<f4").reshape(3, 4)
        d.load(x, numpy.array([1, 2, 3]))
        d.insert_rows([10], x[:1], numpy.array([4]))
        rows = [r for batch in d._cur.many for r in batch]
        assert len(rows) == 4
        assert all(isinstance(r[-1], str) and r[-1].startswith("[")
                   for r in rows)
        assert rows[0] == (0, 1, 1, "[0.0,1.0,2.0,3.0]")


# ---------------------------------------------------------------------------
# §4 the plan guard
# ---------------------------------------------------------------------------

class TestBruteForceIsRefused:
    def test_the_measured_plan_is_refused_before_any_query_runs(self):
        d = driver()
        with pytest.raises(vs.BruteForceRefused):
            d.query(numpy.ones(4), 10)
        assert not any(s.startswith("SELECT id FROM t1")
                       for s in d._cur.statements)
        assert d._cur.statements[-1].startswith("EXPLAIN FORMAT=TREE ")

    def test_a_plan_on_the_index_is_measured_with_the_vector_inlined(self):
        d = driver(plan=INDEX_PLAN)
        assert d.query(numpy.ones(4), 10) == list(range(10))
        sql = d._cur.statements[-1]
        assert sql == ("SELECT id FROM t1 ORDER BY "
                       "L2_DISTANCE(v, '[1.0,1.0,1.0,1.0]') LIMIT 10")
        assert d._cur.params[-1] is None

    def test_the_marker_must_name_this_index(self):
        d = driver(plan=INDEX_PLAN.replace("on vi", "on other_index"))
        assert d.explain_uses_vector_index(numpy.ones(4), 10) is False

    def test_the_check_is_once_per_driver(self):
        d = driver(plan=INDEX_PLAN)
        d.query(numpy.ones(4), 10)
        d.query(numpy.ones(4), 10)
        assert sum(s.startswith("EXPLAIN") for s in d._cur.statements) == 1

    def test_cosine_orders_by_the_function_its_profile_binds(self):
        d = driver(metric="angular", plan=INDEX_PLAN)
        d.query(numpy.ones(4), 5)
        assert "ORDER BY COSINE_DISTANCE(v, '[" in d._cur.statements[-1]

    def test_filtered_binds_the_threshold_and_whitelists_the_column(self):
        d = driver(plan=INDEX_PLAN)
        d.query_filtered(numpy.ones(4), 10, 7, column="ctag")
        assert d._cur.statements[-1].startswith(
            "SELECT id FROM t1 WHERE ctag < %s ORDER BY")
        assert d._cur.params[-1] == (7,)
        with pytest.raises(ValueError):
            d.query_filtered(numpy.ones(4), 10, 7, column="tag; DROP")

    def test_delete_is_refused_with_the_servers_reason(self):
        d = driver()
        with pytest.raises(NotImplementedError, match="not supported yet"):
            d.delete_ids([1, 2])
        assert not d._cur.statements
        d.delete_ids([])      # nothing to do is not an error


# ---------------------------------------------------------------------------
# §5 disclosures
# ---------------------------------------------------------------------------

def _label(engine: str) -> str:
    return {"villagesql": "VillageSQL", "pgvector": "pgvector"}.get(engine,
                                                                    engine)


class TestDisclosures:
    def test_the_driver_declares_both_facts(self):
        facts = vs.VillageSQLDriver(ConnectionSpec(host="h", port=1)) \
            .query_path_facts(IndexSpec(dim=4, m=16, metric="euclidean"))
        assert facts["vector_binding"].startswith("text")
        assert facts["extension_status"].startswith("preview")

    def test_the_report_states_them_for_villagesql_alone(self):
        from report import disclosures
        facts = vs.VillageSQLDriver(ConnectionSpec(host="h", port=1)) \
            .query_path_facts(IndexSpec(dim=4, m=16, metric="euclidean"))
        records = [
            {"engine": "villagesql", "phase": "index_build", "extra": facts},
            {"engine": "pgvector", "phase": "index_build", "extra": {}},
        ]
        text = disclosures.structural_lines(records, _label)
        assert "**VillageSQL** receives every vector as a text literal" in text
        assert "preview feature" in text
        assert "pgvector" not in text

    def test_an_engine_that_declares_nothing_renders_nothing(self):
        from report import disclosures
        records = [{"engine": "pgvector", "phase": "index_build",
                    "extra": {"vector_binding": "binary float32"}}]
        assert disclosures.structural_lines(records, _label) == ""

    def test_the_ann_attributes_reach_the_record(self):
        from report import disclosures
        got = disclosures.query_path_extra(
            {"vector_binding": b"text literal ('[a,b,...]', parsed by strtof)",
             "extension_status": "preview (x)"})
        assert got == {"vector_binding":
                       "text literal ('[a,b,...]', parsed by strtof)",
                       "extension_status": "preview (x)"}
        assert disclosures.query_path_extra({}) == {}


# ---------------------------------------------------------------------------
# §6 the ann overlay module, run against a scripted cursor
# ---------------------------------------------------------------------------

def _load_module():
    from tests.test_e1_percona_design import ALGORITHMS, _load_overlay
    _load_overlay()
    pkg = types.ModuleType("ann_benchmarks.algorithms.villagesql")
    pkg.__path__ = [os.path.join(ALGORITHMS, "villagesql")]
    sys.modules[pkg.__name__] = pkg
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "ann_benchmarks.algorithms.villagesql.module",
            os.path.join(ALGORITHMS, "villagesql", "module.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = was
    return mod


@pytest.fixture
def annmod():
    return _load_module()


def _algo(annmod, m=16, ef_construction=200, **cursor):
    """A VillageSQL instance without __init__, which would start a server."""
    algo = object.__new__(annmod.VillageSQL)
    algo._m, algo._metric, algo._metric_fn = m, "l2", "l2"
    algo._ef_construction = annmod.checked_ef_construction(m, ef_construction)
    algo.dialect = annmod.dataclasses.replace(
        annmod.VILLAGESQL, ef_construction=algo._ef_construction)
    algo._storage_engine = "InnoDB"
    algo._insert_threads = 1
    algo._data_dir = "/nonexistent"
    algo._dim = 0
    algo._ef_search = None
    algo._plan_verified = None
    algo._build_seconds = algo._ingest_seconds = 0.0
    algo._cur = ScriptedCursor(**cursor)
    return algo


class TestAnnModule:
    def test_fit_declares_the_index_inline_and_loads_text(self, annmod):
        algo = _algo(annmod, m=6, ef_construction=192)
        algo.fit(numpy.ones((3, 4), dtype="<f4"))
        create = next(s for s in algo._cur.statements
                      if s.startswith("CREATE TABLE t1"))
        assert "v SVECTOR(4) NOT NULL" in create
        assert ("INDEX vi (v hnsw_l2) USING EXTENDED(hnsw) "
                "WITH (M = 6, ef_construction = 192)") in create
        assert not any(s.startswith("CREATE INDEX")
                       for s in algo._cur.statements)
        rows = [r for batch in algo._cur.many for r in batch]
        assert rows[0] == (0, 0, "[1.0,1.0,1.0,1.0]")

    def test_a_brute_force_plan_is_refused_not_measured(self, annmod):
        algo = _algo(annmod)
        algo._dim = 4
        with pytest.raises(RuntimeError, match="Refusing to measure"):
            algo.set_query_arguments(40)
        assert algo._cur.statements[0] == \
            "SET SESSION vsql_vector.ef_search = 40"
        assert algo._cur.statements[1].startswith("EXPLAIN FORMAT=TREE ")

    def test_an_index_plan_is_measured(self, annmod):
        algo = _algo(annmod, plan=INDEX_PLAN)
        algo._dim = 4
        algo.set_query_arguments(80)
        assert algo._plan_verified is True
        assert algo.query(numpy.ones(4), 10) == list(range(10))
        assert algo._cur.statements[-1] == (
            "SELECT id FROM t1 ORDER BY L2_DISTANCE(v, '[1.0,1.0,1.0,1.0]') "
            "LIMIT 10")

    def test_the_records_carry_the_ef_construction_built(
            self, annmod, monkeypatch):
        monkeypatch.setattr(annmod.VBMySQLBase, "get_additional",
                            lambda self: {})
        algo = _algo(annmod, m=8)
        extra = algo.get_additional()
        assert extra["ef_construction"] == 200
        # No second, "requested" value any more: nothing is clamped, so the
        # label and the build are one number.
        assert "ef_construction_requested" not in extra
        assert extra["vector_binding"].startswith("text")
        assert extra["extension_status"].startswith("preview")

    def test_the_module_and_driver_encode_identically(self, annmod):
        v = numpy.random.RandomState(3).normal(size=32)
        assert annmod.text_literal(v) == vs.text_literal(v)
        for m, efc in ((4, 128), (8, 200), (16, 16), (16, 512)):
            assert annmod.checked_ef_construction(m, efc) == \
                vs.checked_ef_construction(m, efc) == efc
        for m, efc in ((4, 129), (6, 200), (16, 15), (16, 513)):
            for fn in (annmod.checked_ef_construction,
                       vs.checked_ef_construction):
                with pytest.raises(ValueError):
                    fn(m, efc)

    @pytest.mark.parametrize("m", [1, 129])
    def test_the_module_refuses_an_m_outside_the_extensions_range(
            self, annmod, m):
        with pytest.raises(ValueError, match=rf"M={m}"):
            annmod.checked_ef_construction(m, max(m, 2))


class TestTheBulkBuildDeclaration:
    def test_the_host_flag_reaches_the_driver_and_the_build_still_runs(
            self, capsys):
        d = driver(build_mode="post", file_sizes=[0, 0])
        assert d.bulk_vector_index_build is True
        d.declare_bulk_vector_index_build(False)
        assert d.bulk_vector_index_build is False
        d.create_index(d._index)
        assert any(s.startswith("CREATE INDEX vi ON t1")
                   for s in d._cur.statements)
        assert "declares this bulk build broken" in capsys.readouterr().out

    def test_the_orchestrator_sends_the_flag_for_villagesql(self):
        from orchestrator import cli, ops_pass
        assert cli._bulk_vector_index_build(_config()) is False
        argv = ops_pass.harness_args(
            {"name": "t", "k": 10, "ops": {}}, 16, "villagesql", None,
            "normalized", {}, bulk_vector_index_build=False)
        assert "--no-bulk-vector-index-build" in argv


# ---------------------------------------------------------------------------
# §6 what master's engine contract asks of it: knob bounds, reference build
# ---------------------------------------------------------------------------

#: vsql-vector-pk as `scripts/prepare-sources.sh` fetches it: a bare-object
#: repository (`git fetch --depth 1 origin <sha>`, then `git archive` into the
#: build context) with NO working tree, so the pinned files are read from the
#: commit object, never from a checkout. Generated and gitignored; the
#: source half skips only when the repository or the pinned commit is absent.
EXTENSION_REPO = os.path.join(VB_ROOT, "sources", "vsql-vector-git")


def _pinned_source(relpath: str) -> str:
    import subprocess
    if not os.path.isdir(EXTENSION_REPO):
        pytest.skip(f"vsql-vector not fetched into {EXTENSION_REPO}")
    try:
        out = subprocess.run(
            ["git", "-C", EXTENSION_REPO, "show",
             f"{EXTENSION_PIN}:{relpath}"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:
        if "bad object" in out.stderr or "invalid object" in out.stderr \
                or "unknown revision" in out.stderr:
            pytest.skip(f"{EXTENSION_PIN[:12]} not in {EXTENSION_REPO}")
        # The commit is there and the path is not: a re-pin moved the file,
        # which is exactly what this test is for -- fail, do not skip.
        raise AssertionError(f"git show {EXTENSION_PIN[:12]}:{relpath} "
                             f"failed: {out.stderr.strip()}")
    return out.stdout


class TestKnobBoundsAndReferenceBuild:
    def test_the_bounds_match_the_overlay_module_and_the_driver(self, annmod):
        """`ann_knobs` cannot import either; this stops the copies drifting."""
        from orchestrator import ann_knobs
        bounds = ann_knobs.KNOB_BOUNDS["villagesql"]
        for mod in (annmod, vs):
            assert bounds["M"] == (mod.MIN_M, mod.MAX_M)
            assert mod.MAX_EF_CONSTRUCTION_FACTOR == \
                ann_knobs.VILLAGESQL_EFC_FACTOR
        assert bounds["efConstruction"] == (
            annmod.MIN_M, annmod.MAX_M * annmod.MAX_EF_CONSTRUCTION_FACTOR)

    @pytest.mark.parametrize("relpath,pattern,expected", [
        ("src/index/hnsw/storage.h", r"MAX_M = (\d+);", 128),
        ("src/index/hnsw/storage.h", r"MAX_EF_CONSTRUCTION_FACTOR = (\d+);",
         32),
        ("src/index/hnsw/storage.h", r"DEFAULT_EF_CONSTRUCTION = (\d+);", 200),
        ("src/index/hnsw/storage.cc", r"if \(M < (\d+)\)", 2),
    ])
    def test_the_bounds_match_the_pinned_source(self, relpath, pattern,
                                                expected):
        import re
        match = re.search(pattern, _pinned_source(relpath))
        assert match and int(match.group(1)) == expected

    def test_the_combination_rules_are_the_ones_the_source_applies(self):
        """`Options::validate`, statement by statement: M above MAX_M, an
        ef_construction below M, and one above M x MAX_EF_CONSTRUCTION_FACTOR
        are each refused. A re-pin that dropped or moved any of them would
        leave `excluded_builds` dropping builds the extension now accepts, or
        rendering ones it refuses."""
        import re
        source = _pinned_source("src/index/hnsw/storage.cc")
        body = re.search(r"bool Options::validate\(.*?\n}\n", source, re.S)
        assert body, "Options::validate moved or was renamed"
        body = body.group(0)
        assert re.search(r"if \(M > MAX_M\)", body)
        assert re.search(r"if \(ef_construction < M\)", body)
        assert re.search(r"max_ef_construction\s*=\s*M\s*\*\s*"
                         r"MAX_EF_CONSTRUCTION_FACTOR;", body)
        assert re.search(r"if \(ef_construction > max_ef_construction\)",
                         body)

    def test_out_of_range_combinations_are_dropped_by_name(self):
        from orchestrator import ann_knobs
        dropped = ann_knobs.excluded_builds("villagesql", [4, 6, 8, 16, 48],
                                            [64, 200, 400])
        assert [(m, e) for m, e, _ in dropped] == [
            (4, 200), (4, 400), (6, 200), (6, 400), (8, 400)]
        assert "32 * M" in dropped[0][2]
        # ef_construction < M: legal value by value, refused together.
        assert [(m, e) for m, e, _ in ann_knobs.excluded_builds(
            "villagesql", [100], [64])] == [(100, 64)]

    @pytest.mark.parametrize("m", [1, 129])
    def test_an_m_outside_the_range_refuses_the_render(self, m):
        from orchestrator import ann_pass
        profile = {"ann": {"m_values": [m], "ef_search": [10]}}
        with pytest.raises(ValueError, match=rf"M={m} is outside"):
            ann_pass.render_config("villagesql", profile, {}, "normalized")

    def test_an_m_buildable_at_no_pinned_ef_construction_is_refused(self):
        """M=6 at the one ef_construction 200: dropping it would remove the
        value from the grid without a combination left to name it."""
        from orchestrator import ann_pass
        profile = {"ann": {"m_values": [6, 16], "ef_search": [10]}}
        with pytest.raises(ValueError, match="M=6 is buildable at none"):
            ann_pass.render_config("villagesql", profile, {}, "normalized")

    @pytest.mark.parametrize("profile_name", ["smoke", "main",
                                              "tuned-complete", "m-sweep",
                                              "frontier-explore"])
    def test_the_reference_build_resolves_and_is_in_the_grid(self,
                                                             profile_name):
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        from orchestrator.reference_build import (reference_build_problem,
                                                  resolve_reference_build)
        # frontier-explore pins 64: the one value every M in its normalized
        # grid can take (see the profile).
        want = 64 if profile_name == "frontier-explore" else 200
        profile = load_profile(profile_name)
        for resource_pass in ("normalized", "tuned"):
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            assert resolve_reference_build(
                "villagesql", profile, resources, resource_pass) == {
                    "m": 16, "ef_construction": want}
            assert reference_build_problem("villagesql", profile, resources,
                                           resource_pass) is None

    def test_select_builds_can_write_its_ef_construction_axis(self):
        from orchestrator.profile_overlay import OVERLAY_EXTRAS
        assert "villagesql_ef_construction" in OVERLAY_EXTRAS


# ---------------------------------------------------------------------------
# §7 opt-in, the ops pre-flight, and the plan-check cache
# ---------------------------------------------------------------------------

class TestOptIn:
    def test_it_is_registered_but_not_default_selected(self):
        from orchestrator import cli, engines as engines_mod
        assert _config()["runtime"]["default_selected"] is False
        assert engines_mod.get("villagesql").default_selected is False
        assert "villagesql" in cli.KNOWN_ENGINES
        assert "villagesql" not in cli.DEFAULT_ENGINES
        # villagesql84, the same extension on the MySQL 8.4 server line, is
        # opt-in too (tests/test_e30_villagesql84_engine.py).
        assert cli.OPT_IN_ENGINES == ("villagesql", "villagesql84")
        # Nothing else is opt-in: the default list is the registry minus them.
        assert list(cli.DEFAULT_ENGINES) == [
            e for e in cli.KNOWN_ENGINES
            if e not in ("villagesql", "villagesql84")]

    @pytest.mark.parametrize("value", ["false", 0, "no"])
    def test_a_non_boolean_flag_is_refused(self, value):
        """"false" is truthy; read as true it would put the engine back into
        every default run."""
        from orchestrator import engines as engines_mod
        with pytest.raises(ValueError, match="default_selected"):
            engines_mod._default_selected("x", {"default_selected": value})

    def test_absent_means_selected(self):
        from orchestrator import engines as engines_mod
        assert engines_mod._default_selected("x", {}) is True


class TestOpsPreflight:
    def test_an_unbuildable_ops_combination_is_named(self):
        from orchestrator.reference_build import ops_build_problems
        problems = ops_build_problems(
            "villagesql", {"ops": {"m_values": [6, 16]}})
        assert len(problems) == 1
        assert "M=6 at ef_construction=200" in problems[0]
        assert "32 * M" in problems[0]

    def test_an_m_outside_the_range_is_named(self):
        from orchestrator.reference_build import ops_build_problems
        problems = ops_build_problems("villagesql",
                                      {"ops": {"m_values": [129]}})
        assert any("M=129 is outside" in p for p in problems)

    def test_every_shipped_profile_passes(self):
        from orchestrator import cli
        from orchestrator.config import available_profiles, load_profile
        from orchestrator.reference_build import ops_build_problems
        for name in available_profiles():
            for engine in cli.KNOWN_ENGINES:
                assert ops_build_problems(engine, load_profile(name)) == [], (
                    name, engine)

    def test_not_applied_when_ops_is_off_or_frontier_chooses(self):
        from orchestrator.reference_build import ops_build_problems
        assert ops_build_problems("villagesql", {"ops": {
            "enabled": False, "m_values": [6]}}) == []
        assert ops_build_problems("villagesql", {"ops": {
            "build_selection": "frontier", "m_values": [6]}}) == []


class TestThePlanCheckCache:
    def test_it_is_keyed_on_the_query_shape(self):
        """An unfiltered plan check must not vouch for the filtered query."""
        d = driver(plan=INDEX_PLAN)
        d._require_index(numpy.ones(4), 10)
        assert d._plan_checked == {False: True}
        d._cur = ScriptedCursor()       # the brute-force plan
        with pytest.raises(vs.BruteForceRefused):
            d._require_index(numpy.ones(4), 10, tag_threshold=5)

    def test_a_new_schema_forgets_it(self):
        d = driver(plan=INDEX_PLAN)
        d._require_index(numpy.ones(4), 10)
        d.create_schema(d._index)
        assert d._plan_checked == {}
