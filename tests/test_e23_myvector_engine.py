"""MyVector (askdba/myvector v1.26.9 on MySQL 9.7.2): the engine's wiring and
the driver behaviours that keep it honest.

No server is started here. The things pinned are the ones that fail SILENTLY
against a real server -- an option spelling the plugin does not match falls
back to a default without a word, a failed index build returns a row of text
rather than an SQL error, a deleted row stays in the graph -- so a test that
could only see a crash would see none of them.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, List, Optional

import numpy
import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness.drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from harness.drivers import myvector as mv  # noqa: E402

PIN = "384d9df7ab8a2afee9c8f99704e5a7ea90d32730"


def _read(rel: str) -> str:
    with open(os.path.join(VB_ROOT, rel)) as fh:
        return fh.read()


def _config() -> dict:
    return yaml.safe_load(_read("config/engines/myvector.yml"))


# ---------------------------------------------------------------------------
# A cursor that answers the plugin's procedures like v1.26.9 does
# ---------------------------------------------------------------------------

def status_text(m=16, dim=4, dist="L2", capacity=2500, rows=1000,
                type_="HNSW") -> str:
    """What `HNSWMemoryIndex::getStatus` prints, line for line."""
    return ("\nVector Index : ann.t1.v\nType : " + type_ + "\nDimension : "
            + str(dim) + "\nDistance : " + dist + "\nMax. Capacity : "
            + str(capacity) + "\nM = " + str(m) + "\nElement Data Size : 88"
            "\nCurrent Rows : " + str(rows) + "\nSearches : 0\n")


class ScriptedCursor:
    def __init__(self, build="SUCCESS: Index created & saved at (binlog.000002 157), rows : 1000.",
                 status: Optional[str] = None, count=1000, explain=None,
                 refresh="SUCCESS: rows : 0."):
        self.statements: List[str] = []
        self.params: List[Any] = []
        self.build = build
        self.status = status if status is not None else status_text()
        self.count = count
        self.explain = explain or []
        self.refresh = refresh
        self._rows: List[tuple] = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self.params.append(params)
        s = sql.upper()
        if "MYVECTOR_INDEX_BUILD" in s:
            self._rows = [(self.build,)]
        elif "MYVECTOR_INDEX_REFRESH" in s:
            self._rows = [(self.refresh,)]
        elif "MYVECTOR_INDEX_STATUS" in s:
            self._rows = [(self.status,)]
        elif s.startswith("SELECT COUNT(*) FROM T1"):
            self._rows = [(self.count,)]
        elif s.startswith("SELECT COUNT(*) FROM INFORMATION_SCHEMA"):
            self._rows = [(1,)]
        elif s.startswith("EXPLAIN"):
            self._rows = list(self.explain)
        else:
            self._rows = []

    def executemany(self, sql, rows):
        self.statements.append(sql)

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None

    def nextset(self):
        return None

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_second_boundary_wait(monkeypatch):
    """`_next_second` sleeps up to a second per build; not under test here
    except where a test asks for it."""
    monkeypatch.setattr(mv.MyVectorDriver, "_next_second", staticmethod(lambda: None))


def driver(build_mode="post", metric="euclidean", m=16, dim=4, **cursor):
    d = mv.MyVectorDriver(ConnectionSpec(host="h", port=1))
    d._cur = ScriptedCursor(**cursor)
    d._index = IndexSpec(dim=dim, m=m, metric=metric, ef_construction=200,
                         build_mode=build_mode)
    return d


# ---------------------------------------------------------------------------
# §1 the engine is registered everywhere a run reaches
# ---------------------------------------------------------------------------

class TestWiring:
    def test_the_config_pins_the_release_commit_and_the_server(self):
        source = _config()["source"]
        assert source["tag"] == PIN
        assert source["release"] == "v1.26.9"
        assert source["server_tag"].startswith("mysql-9.7.")

    def test_the_driver_is_registered_by_class_name(self):
        from harness.drivers.postgres import _driver_classes, _driver_table
        assert _driver_classes()["MyVectorDriver"] is mv.MyVectorDriver
        assert _driver_table()["myvector"] is mv.MyVectorDriver
        assert _config()["runtime"]["driver"] == "MyVectorDriver"

    def test_myvector_has_one_build_mode_and_it_is_post(self):
        """No build-mode axis: the binlog listener that would make an
        incremental mode is dead on this server, so the orchestrator runs
        `post` alone and the driver refuses `incremental` with the reason."""
        from orchestrator import cli
        assert "myvector" not in cli._OPS_BUILD_MODE_KEYS
        assert cli._ops_build_modes("myvector", {}, _config()) == ["post"]
        d = driver("incremental")
        with pytest.raises(ValueError, match="binlog listener"):
            d.create_schema(d._index)
        assert d._cur.statements == []

    def test_both_march_readers_know_the_image(self):
        """`march: unknown` on a record is the failure the readers exist for."""
        assert "/opt/myvector/.march" in _read("harness/main.py")
        assert "/opt/myvector/.march" in _read(
            "overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py")
        assert "/opt/myvector/.march" in _read("docker/myvector/Dockerfile")

    def test_the_build_scripts_know_the_engine(self):
        assert "myvector)   prepare_myvector" in _read("scripts/prepare-sources.sh")
        assert "myvector)   build_engine myvector" in _read("scripts/build-images.sh")

    def test_the_webui_copy_of_colour_and_label_matches_the_yaml(self):
        runtime = _config()["runtime"]
        js = _read("webui/static/fields.js")
        assert f'myvector: "{runtime["chart"]["color"]}"' in js
        assert f'myvector: "{runtime["label"]}"' in js

    def test_the_ann_overlay_module_exists_under_the_constructor_name(self):
        module = _read("overlay/ann-benchmarks/ann_benchmarks/algorithms/myvector/module.py")
        assert "class MyVector(" in module
        assert _config()["runtime"]["ann_constructor"] == "MyVector"

    def test_render_config_emits_ef_construction_pinned_in_normalized(self):
        from orchestrator import ann_pass
        profile = {"ann": {"m_values": [16], "ef_search": [10, 40]}}
        body = ann_pass.render_config("myvector", profile, {}, "normalized")
        algo = body["float"]["any"][0]
        assert algo["constructor"] == "MyVector"
        groups = algo["run_groups"]
        assert list(groups) == ["innodb"]
        assert groups["innodb"]["arg_groups"][0]["efConstruction"] == [200]

    def test_estimate_names_myvector_as_unmeasured(self):
        from orchestrator import estimate
        profile = {"datasets": ["glove-100-angular"],
                   "ops": {"m_values": [16]}, "ann": {"m_values": [16]}}
        est = estimate.estimate_load_hours(profile, ["myvector"],
                                           ["glove-100-angular"],
                                           ["normalized"], ["ann", "ops"])
        assert "myvector" in est["engines_without_ingest_rate"]


# ---------------------------------------------------------------------------
# §2 the DDL uses the spellings the plugin actually matches
# ---------------------------------------------------------------------------

class TestDDL:
    @pytest.mark.parametrize("metric,dist", [("euclidean", "L2"),
                                             ("angular", "Cosine")])
    def test_the_column_comment_is_the_exact_case_the_plugin_matches(self, metric, dist):
        d = driver(metric=metric, m=24)
        ddl = d._create_table_ddl(d._index, with_index=False)
        comment = re.search(r"COMMENT '([^']*)'", ddl).group(1)
        assert comment.startswith("MYVECTOR Column |")
        opts = dict(kv.split("=") for kv in comment.split("|", 1)[1].split(","))
        # `M`, never `m` -- getIntOption("M", 16) ignores the other case.
        assert opts["M"] == "24" and "m" not in opts
        assert opts["dist"] == dist
        assert opts["type"] == "HNSW"
        assert opts["ef"] == "200"
        # No `online=Y`: the listener is dead on this server. `track=` is the
        # refresh path's timestamp, and the column it names must exist.
        assert "online" not in opts
        assert opts["track"] == "updatets"
        assert "v VECTOR(4) NOT NULL COMMENT" in ddl
        assert ("updatets TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP "
                "ON UPDATE CURRENT_TIMESTAMP") in ddl

    def test_the_key_is_int_like_the_other_family_tables(self):
        d = driver()
        assert "id INT PRIMARY KEY" in d._create_table_ddl(d._index, with_index=False)

    def test_other_family_drivers_keep_a_bare_vector_column(self):
        from harness.drivers.mysql_family import MariaDBDriver
        m = MariaDBDriver(ConnectionSpec(host="h", port=1))
        ddl = m._create_table_ddl(IndexSpec(dim=4, m=16, metric="euclidean"),
                                  with_index=True)
        assert "  v VECTOR(4) NOT NULL,\n" in ddl
        assert "COMMENT" not in ddl


# ---------------------------------------------------------------------------
# §3 the build is verified, not assumed
# ---------------------------------------------------------------------------

class TestBuild:
    def test_post_build_sizes_the_graph_then_calls_the_procedure(self):
        d = driver(status=status_text(capacity=2500))
        d.create_index(d._index)
        sql = d._cur.statements
        alter = [s for s in sql if s.startswith("ALTER TABLE t1 MODIFY COLUMN")]
        assert alter and "size=2500" in alter[0] and "ALGORITHM=INSTANT" in alter[0]
        # The ALTER redefines the vector column only, not the tracking one.
        assert "updatets" not in alter[0].split("COMMENT", 1)[0]
        assert "TIMESTAMP" not in alter[0]
        assert "CALL mysql.myvector_index_build('ann.t1.v', 'id')" in sql
        assert any("myvector_index_status" in s for s in sql)
        assert d.capabilities()["myvector_status"]["M"] == "16"

    @pytest.mark.parametrize("text", [
        "Error in new connection to build vector index : Can't connect.",
        "Error loading config file for index build.",
        "ERROR: index built but could not be saved to disk (see the server log)",
    ])
    def test_an_error_row_is_a_failed_build(self, text):
        d = driver(build=text)
        with pytest.raises(RuntimeError, match="did not succeed"):
            d.create_index(d._index)

    def test_a_short_build_is_refused(self):
        d = driver(build="SUCCESS: Index created & saved at (binlog.000002 4), rows : 7.")
        with pytest.raises(RuntimeError, match="indexed 7 rows of 1000"):
            d.create_index(d._index)

    @pytest.mark.parametrize("kwargs,key", [
        ({"m": 16}, None),
        ({"m": 32}, "M"),            # a mis-cased M builds the default
        ({"type_": "KNN"}, "Type"),  # an unknown type falls back to KNN
        ({"dim": 8}, "Dimension"),   # a stale index object from a prior table
        ({"rows": 999}, "Current Rows"),
        ({"dist": "cosine"}, "Distance"),
    ])
    def test_status_mismatch_is_refused(self, kwargs, key):
        d = driver(status=status_text(**{"capacity": 2500, **kwargs}))
        if key is None:
            d.create_index(d._index)
            return
        with pytest.raises(RuntimeError, match=re.escape(f"'{key}'")):
            d.create_index(d._index)

    def test_status_parser_reads_both_separators(self):
        parsed = mv.parse_status(status_text(m=12, rows=5))
        assert parsed["M"] == "12" and parsed["Current Rows"] == "5"
        assert parsed["Max. Capacity"] == "2500"

    def test_drop_schema_drops_the_plugin_index_before_the_database(self):
        d = driver()
        d.drop_schema()
        sql = d._cur.statements
        drop_idx = sql.index("CALL mysql.myvector_index_drop('ann.t1.v')")
        drop_db = sql.index("DROP DATABASE IF EXISTS ann")
        assert drop_idx < drop_db

    def test_capabilities_say_bulk_build(self):
        assert driver("post").capabilities()["incremental_index"] is False


# ---------------------------------------------------------------------------
# §4 the query is one the rewrite will actually rewrite
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_carries_ef_and_k_and_no_bound_parameter(self):
        d = driver()
        d.set_ef_search(77)
        assert d._cur.statements == []   # no session variable exists
        d.query(numpy.ones(4, dtype="<f4"), 10)
        bind, sql = d._cur.statements[-2:]
        # The vector travels in a SET the rewrite never sees: inline, a 784-dim
        # literal killed mysqld in the first smoke run.
        assert bind == "SET @vb_q = X'" + "0000803f" * 4 + "'"
        assert "MYVECTOR" not in bind
        assert sql.startswith("SELECT ")
        assert "MYVECTOR_IS_ANN('ann.t1.v','id',@vb_q," in sql
        assert "X'" not in sql
        assert "'nn=10,ef_search=77')" in sql
        assert sql.endswith("ORDER BY myvector_row_distance(id) LIMIT 10")
        assert d._cur.params[-1] is None and d._cur.params[-2] is None

    def test_the_ann_module_binds_the_vector_the_same_way(self):
        module = _read(
            "overlay/ann-benchmarks/ann_benchmarks/algorithms/myvector/module.py")
        assert 'cur.execute(f"SET @vb_q = {hex_literal(v)}")' in module
        assert "'id',@vb_q," in module

    def test_the_ann_module_reports_status_as_a_string(self):
        """ann-benchmarks writes get_additional() into HDF5 attributes, which
        cannot hold a dict; a dict there lost the first smoke run's sweep."""
        module = _read(
            "overlay/ann-benchmarks/ann_benchmarks/algorithms/myvector/module.py")
        assert 'extra["myvector_status"] = json.dumps(' in module
        assert 'extra["myvector_status"] = dict(' not in module

    def test_the_ann_module_does_not_wait_out_the_post_shutdown_hang(self):
        """mysqld with this plugin logs 'Shutdown complete' and never exits;
        the shared done() would idle 300 s per configuration on it."""
        module = _read(
            "overlay/ann-benchmarks/ann_benchmarks/algorithms/myvector/module.py")
        assert "    def done(self) -> None:" in module
        assert '"Shutdown complete" in fh.read()' in module

    def test_hex_literal_cannot_contain_a_parenthesis(self):
        # 0x28 '(' and 0x29 ')' as float32 bytes: the rewrite counts parens
        # over raw text, so a binary literal holding them would derail it.
        v = numpy.frombuffer(b"\x28\x29\x28\x29" * 2, dtype="<f4")
        lit = mv.hex_literal(v)
        assert lit == "X'2829282928292829'"
        assert "(" not in lit and ")" not in lit

    def test_query_without_ef_is_refused(self):
        with pytest.raises(RuntimeError, match="set_ef_search"):
            driver().query(numpy.ones(4), 10)

    def test_effective_ef_is_hnswlib_max_ef_k(self):
        assert driver().effective_ef(5, 10) == 10
        assert driver().effective_ef(40, 10) == 40

    def test_explain_requires_the_rewrite_derived_table(self):
        row = (1, "PRIMARY", "myvector_ann", None, "ALL", None, None, None,
               None, 2, 100.0, "Table function: json_table")
        d = driver(explain=[row])
        d.set_ef_search(40)
        assert d.explain_uses_vector_index(numpy.ones(4), 10) is True
        d2 = driver(explain=[(1, "SIMPLE", "t1", None, "ALL") + (None,) * 7])
        d2.set_ef_search(40)
        assert d2.explain_uses_vector_index(numpy.ones(4), 10) is False


# ---------------------------------------------------------------------------
# §5 filtered search is declined, with the reason, before any record
# ---------------------------------------------------------------------------

class TestFilteredDeclined:
    def test_the_driver_declares_it(self):
        declined = dict(mv.MyVectorDriver.declined_workloads)
        assert set(declined) == {"filtered"}
        assert "v1.26.9" in declined["filtered"]

    def test_query_filtered_refuses_rather_than_post_filtering(self):
        d = driver()
        d.set_ef_search(40)
        with pytest.raises(NotImplementedError):
            d.query_filtered(numpy.ones(4), 10, 5)
        assert not any("MYVECTOR_IS_ANN" in s for s in d._cur.statements)

    def test_no_other_driver_declines_anything(self):
        # VillageSQL declines its own two, pinned in
        # tests/test_e26_villagesql_engine.py.
        from harness.drivers.postgres import _driver_classes
        for name, cls in _driver_classes().items():
            if name not in ("MyVectorDriver", "VillageSQLDriver"):
                assert tuple(cls.declined_workloads) == (), name

    def test_config_capability_agrees(self):
        assert _config()["capabilities"]["filtered_search"] is False

    def test_main_skips_and_stamps_declined_workloads(self):
        from harness import main as hmain
        kept, declined = hmain._apply_declined(
            mv.MyVectorDriver(ConnectionSpec(host="h", port=1)),
            ["build", "concurrency", "filtered", "churn"])
        assert kept == ["build", "concurrency", "churn"]
        assert set(declined) == {"filtered"}

        class Plain:
            declined_workloads = ()
        assert hmain._apply_declined(Plain(), ["filtered"]) == (["filtered"], {})


# ---------------------------------------------------------------------------
# §6 churn: inserts reach the graph through the binlog, deletes never do
# ---------------------------------------------------------------------------

class TestChurn:
    def test_repair_is_a_full_rebuild(self):
        d = driver(status=status_text(capacity=2500, rows=1000))
        d._capacity = 2500
        assert d.repair_path == "index-rebuild"
        assert d.maintain() == "myvector_index_build (full rebuild)"
        assert "CALL mysql.myvector_index_build('ann.t1.v', 'id')" in d._cur.statements

    def test_sync_is_a_verified_refresh(self):
        """The graph keeps deleted rows, so 1,000 rows plus a churn of 100 is
        1,100 elements, and that is what the refresh must leave."""
        d = driver(status=status_text(rows=1100),
                   refresh="SUCCESS: Index created & saved at (x 1), rows : 100., "
                           "Vector Index refreshed at : now")
        d._expected_elements, d._pending = 1000, 100
        assert d.wait_index_synced(1000) >= 0.0
        assert ("CALL mysql.myvector_index_refresh('ann.t1.v', 'id')"
                in d._cur.statements)
        assert (d._expected_elements, d._pending) == (1100, 0)

    def test_a_short_refresh_is_refused(self):
        d = driver(status=status_text(rows=1100),
                   refresh="SUCCESS: Index created & saved at (x 1), rows : 60.")
        d._expected_elements, d._pending = 1000, 100
        with pytest.raises(RuntimeError, match="indexed 60 rows of 100"):
            d.wait_index_synced(1000)

    def test_a_refresh_that_leaves_the_wrong_element_count_is_refused(self):
        d = driver(status=status_text(rows=1000),
                   refresh="SUCCESS: refreshed, rows : 100.")
        d._expected_elements, d._pending = 1000, 100
        with pytest.raises(RuntimeError, match="Current Rows"):
            d.wait_index_synced(1000)

    def test_insert_beyond_capacity_is_refused_before_the_server_sees_it(self):
        d = driver()
        d._capacity, d._expected_elements = 1050, 1000
        with pytest.raises(RuntimeError, match="capacity 1,050"):
            d.insert_rows(list(range(100)), numpy.ones((100, 4)), [0] * 100)
        assert not any(s.startswith("INSERT") for s in d._cur.statements)

    def test_insert_within_capacity_advances_the_expected_count(self):
        d = driver()
        d._capacity, d._expected_elements = 2500, 1000
        d.insert_rows(list(range(100)), numpy.ones((100, 4)), [0] * 100)
        assert (d._expected_elements, d._pending) == (1000, 100)


# ---------------------------------------------------------------------------
# §7 the image and the entrypoint
# ---------------------------------------------------------------------------

class TestImage:
    def test_error_log_matches_the_entrypoint_default(self):
        entry = _read("docker/myvector/entrypoint-myvector.sh")
        assert f'LOG_FILE="${{VB_LOG_FILE:-{_config()["runtime"]["error_log"]}}}"' in entry

    def test_the_plugin_cnf_is_never_world_readable(self):
        entry = _read("docker/myvector/entrypoint-myvector.sh")
        assert "umask 077" in entry and 'chmod 0600 "$CNF_FILE"' in entry

    def test_binlog_is_left_as_percona_has_it(self):
        """No binlog requirement (the listener is not used), so the server
        logs exactly as the other engine on this server line does."""
        import yaml as _y
        percona = _y.safe_load(_read("config/engines/percona.yml"))["server"]
        mine = _config()["server"]
        assert not any("binlog" in f or "log-bin" in f for f in mine["common"])
        for p in ("normalized", "tuned"):
            assert (("--skip-log-bin" in mine[p])
                    == ("--skip-log-bin" in percona[p])), p

    def test_install_plugin_is_stripped_because_plugin_load_add_loads_it(self):
        docker = _read("docker/myvector/Dockerfile")
        assert "grep -v '^INSTALL PLUGIN'" in docker
        assert "--plugin-load-add=myvector.so" in _read(
            "docker/myvector/entrypoint-myvector.sh")


# ---------------------------------------------------------------------------
# §8 the report says why a declined workload has no chart
# ---------------------------------------------------------------------------

class TestReportNamesTheDecline:
    def test_a_declined_workload_is_not_called_unrequested(self):
        from report.generate import _why_absent
        records = [{"engine": "myvector", "dataset": "d", "phase": "index_build",
                    "extra": {"declined_workloads": {"filtered": "no filter"}}},
                   {"engine": "pgvector", "dataset": "d", "phase": "index_build",
                    "extra": {}}]
        assert _why_absent(records, "d", "filtered") == \
            "declined by the driver for myvector"
        assert _why_absent(records, "d", "churn") == \
            "not in this profile's workloads"
        assert _why_absent(records, "other", "filtered") == \
            "not in this profile's workloads"

    def test_the_engine_notes_document_the_dead_listener(self):
        notes = _read("docs/04-engine-notes.md")
        assert "## MyVector" in notes
        assert "myvector_index_refresh" in notes
        assert "Binlog fetch failed" in notes
