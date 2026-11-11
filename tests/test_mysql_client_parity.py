"""The MySQL-family client does the same work on every server.

Every MySQL-family engine -- MariaDB, mariadb123, AliSQL, Percona Server,
MyVector, VillageSQL, villagesql84 -- is driven through one client, MariaDB
Connector/Python, so that a difference in the numbers belongs to the server.
That holds only if the client also BEHAVES the same per server, and under
1.1.10 it did not. Counted on 2026-11-09 against Percona Server 9.7 and
MariaDB 11.8 (the servers' Com_stmt_* counters and a packet capture of the
client, 1,000 identical KNN queries and one 500-row batch):

    1.1.10, `%s`        Percona  close+prepare+execute per query, 2 round trips
                        MariaDB  close+prepare+execute per query, 1 round trip
    2.0.0rc2, binary,   both     1 execute per query, prepared once, 1 trip
    `?`
    executemany, 500    Percona 500 executes, MariaDB 1 -- under both versions

So the fix has four parts, and each is pinned here because each one, undone
alone, silently brings the asymmetry back without failing a single query:

  1. `binary=True` on every MySQL-family connection, in both measurement paths;
  2. `?` placeholders -- 2.0's binary path sends the SQL unaltered, so `%s`
     is a syntax error, and 1.1's `%s` rewrite is what defeated statement reuse;
  3. one multi-row INSERT per batch, never `executemany`;
  4. Connector/Python 2.0.0rc2's pure-Python implementation on every image.

None of this starts a server: the counts above were measured, and what is
checked here is that the code still sends what was measured.
"""

from __future__ import annotations

import ast
import os
import re
import sys

import numpy
import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness.drivers import mysql_family  # noqa: E402
from harness.drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from harness.drivers.mysql_family import (AliSQLDriver,  # noqa: E402
                                          MariaDBDriver, PerconaDriver)
from harness.drivers.myvector import MyVectorDriver  # noqa: E402
from harness.drivers.villagesql import VillageSQLDriver  # noqa: E402

OVERLAY = os.path.join(VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks")

#: Every source file that speaks to a MySQL-family server through the
#: connector. A new MySQL-family driver or ann module belongs here.
MYSQL_FAMILY_SOURCES = (
    "harness/drivers/mysql_family.py",
    "harness/drivers/myvector.py",
    "harness/drivers/villagesql.py",
    "overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
    *(f"overlay/ann-benchmarks/ann_benchmarks/algorithms/{e}/module.py"
      for e in ("mariadb", "mariadb123", "alisql", "percona", "myvector",
                "villagesql", "villagesql84")),
)

#: The images that install requirements-mysql.txt.
MYSQL_FAMILY_IMAGES = ("mariadb", "alisql", "percona", "myvector",
                       "villagesql")

MYSQL_DRIVERS = (MariaDBDriver, AliSQLDriver, PerconaDriver, MyVectorDriver,
                 VillageSQLDriver)


def _source(rel: str) -> str:
    return open(os.path.join(VB_ROOT, rel), encoding="utf-8").read()


def _code_strings(rel: str):
    """Every string constant in the file's CODE -- docstrings excluded.

    Docstrings and comments are where the old `%s` and `executemany` are
    explained, and they are allowed to explain them.
    """
    tree = ast.parse(_source(rel))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.value


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.many = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def executemany(self, sql, rows):
        self.many.append((sql, rows))

    def fetchall(self):
        return []

    def fetchone(self):
        return (0,)


# ---------------------------------------------------------------------------
# 1. binary=True on every connection
# ---------------------------------------------------------------------------

class TestEveryConnectionIsBinary:
    def test_the_ops_harness_connection_options(self):
        kw = mysql_family.connect_kwargs(ConnectionSpec(host="h", port=1))
        assert kw["binary"] is True
        assert (kw["host"], kw["port"], kw["autocommit"]) == ("h", 1, True)

    def test_every_ops_connect_goes_through_connect_kwargs(self):
        """Two `_connector.connect(` sites exist -- the driver's own and each
        load worker's -- and a third written out by hand would be one
        connection without `binary=True`. Counted, so the count is a claim."""
        src = _source("harness/drivers/mysql_family.py")
        sites = re.findall(r"_connector\.connect\((.*)\)$", src, re.MULTILINE)
        assert sites == ["**connect_kwargs(self.spec)"] * 2, sites
        assert src.count("_connector.connect(") == 2
        for rel in ("harness/drivers/myvector.py",
                    "harness/drivers/villagesql.py"):
            assert "_connector.connect(" not in _source(rel), rel

    def test_the_ann_connection_options(self):
        from tests.test_e1_percona_design import _load_overlay
        vb, _ = _load_overlay()
        obj = object.__new__(vb.VBMySQLBase)
        obj._socket = "/s"
        kw = vb.VBMySQLBase._conn_kwargs(obj)
        assert kw["binary"] is True
        # Loopback TCP: 2.0.0rc2 cannot log in over the unix socket to a
        # caching_sha2_password server (`_conn_kwargs` has the trace).
        assert (kw["host"], kw["port"]) == ("127.0.0.1", 3306)
        assert "unix_socket" not in kw

    def test_every_ann_connect_goes_through_conn_kwargs(self):
        for rel in MYSQL_FAMILY_SOURCES:
            if not rel.startswith("overlay"):
                continue
            src = _source(rel)
            sites = re.findall(r"_connector\.connect\((.*)\)$", src,
                               re.MULTILINE)
            # A call wrapped over several lines would escape the pattern.
            assert len(sites) == src.count("_connector.connect("), rel
            for args in sites:
                assert args == "**self._conn_kwargs()", (rel, args)


# ---------------------------------------------------------------------------
# 2. `?` placeholders, never `%s`
# ---------------------------------------------------------------------------

class TestNoFormatPlaceholders:
    @pytest.mark.parametrize("rel", MYSQL_FAMILY_SOURCES)
    def test_no_percent_s_in_any_statement(self, rel):
        offenders = [s for s in _code_strings(rel) if "%s" in s]
        assert not offenders, (rel, offenders)

    @pytest.mark.parametrize("cls", MYSQL_DRIVERS)
    def test_the_select_each_driver_sends(self, cls):
        driver = cls(ConnectionSpec(host="h", port=3306))
        driver._index = IndexSpec(dim=8, m=16, metric="euclidean",
                                  build_mode="incremental")
        # VillageSQL inlines the query vector as a text literal (its KNN
        # recognition needs a constant), so its `_select` takes one.
        kw = {"vector": numpy.ones(8)} if cls is VillageSQLDriver else {}
        sql = driver._select(10, False, **kw)
        assert "%s" not in sql, (cls, sql)
        filtered = driver._select(10, True, **kw)
        assert "%s" not in filtered and "?" in filtered, (cls, filtered)

    def test_the_insert_statement(self):
        assert mysql_family.INSERT_ROW == "(?, ?, ?, ?)"
        assert "%s" not in mysql_family.INSERT_HEAD


# ---------------------------------------------------------------------------
# 3. one multi-row INSERT per batch, never executemany
# ---------------------------------------------------------------------------

class TestOneStatementPerBatch:
    @pytest.mark.parametrize("rel", MYSQL_FAMILY_SOURCES)
    def test_no_executemany_call(self, rel):
        tree = ast.parse(_source(rel))
        calls = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "executemany"]
        assert not calls, (rel, calls)

    def test_a_batch_is_one_execute_with_every_row_flattened(self):
        cur = FakeCursor()
        rows = [(i, i % 100, i % 7, b"v%d" % i) for i in range(500)]
        mysql_family.insert_batch(cur, rows)
        assert not cur.many
        assert len(cur.calls) == 1
        sql, params = cur.calls[0]
        assert sql == (mysql_family.INSERT_HEAD
                       + ", ".join(["(?, ?, ?, ?)"] * 500))
        assert sql.count("?") == len(params) == 2000
        assert params[:4] == [0, 0, 0, b"v0"]
        assert params[-4:] == [499, 99, 499 % 7, b"v499"]

    def test_an_empty_batch_sends_nothing(self):
        cur = FakeCursor()
        mysql_family.insert_batch(cur, [])
        assert cur.calls == [] and cur.many == []

    def test_the_load_sends_full_batches_and_one_tail(self):
        driver = object.__new__(MariaDBDriver)
        driver.name = "mariadb"
        cur = FakeCursor()
        vectors = numpy.zeros((1203, 4), dtype=numpy.float32)
        driver._load_range(cur, vectors, mysql_family.RowAttrs.coerce(
            numpy.zeros(1203, dtype=numpy.int64)), 0, 0, 1203)
        inserts = [(s, p) for s, p in cur.calls
                   if s.startswith(mysql_family.INSERT_HEAD)]
        assert [len(p) // 4 for _, p in inserts] == [500, 500, 203]
        assert cur.calls[-1] == ("COMMIT", None)

    def test_the_ann_batch_is_one_execute(self):
        from tests.test_e1_percona_design import _load_overlay
        vb, _ = _load_overlay()
        cur = FakeCursor()
        vb.insert_batch(cur, [(i, i % 100, b"x") for i in range(3)])
        assert cur.calls == [(
            "INSERT INTO t1 (id, tag, v) VALUES (?, ?, ?), (?, ?, ?), (?, ?, ?)",
            [0, 0, b"x", 1, 1, b"x", 2, 2, b"x"])]
        vb.insert_batch(cur, [])
        assert len(cur.calls) == 1


# ---------------------------------------------------------------------------
# 4. Connector/Python 2.0.0rc2, pure Python, on every image
# ---------------------------------------------------------------------------

class TestTheClientIsPinned:
    def _requirements(self):
        lines = _source("docker/_shared/requirements-mysql.txt").splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    def test_the_connector_is_pinned_without_extras(self):
        """`mariadb[binary]` would install the wheel that cannot authenticate
        to MySQL 8+, and `mariadb[c]` a compiled client the other images may
        not get; either is a different client per engine."""
        reqs = self._requirements()
        assert "mariadb==2.0.0rc2" in reqs
        assert not [r for r in reqs if r.startswith("mariadb[")
                    or r.startswith("mariadb-")], reqs
        assert "--no-binary mariadb" not in reqs

    def test_the_prepared_statement_cache_dependency_is_installed(self):
        """2.0 caches prepared statements in a `cachetools.LRUCache` and, if
        the import fails, sets the cache to None and silently re-prepares
        every query (`impl/client/base_client.py`) -- the very bug this
        client was adopted to remove, back with no error. It must be pinned,
        not left to whatever pip resolves as a transitive dependency."""
        assert any(r.startswith("cachetools==") for r in self._requirements())

    def test_every_dependency_is_pinned(self):
        for req in self._requirements():
            assert "==" in req, req

    @pytest.mark.parametrize("engine", MYSQL_FAMILY_IMAGES)
    def test_every_image_build_asserts_the_pure_python_client(self, engine):
        dockerfile = _source(f"docker/{engine}/Dockerfile")
        assert "requirements-mysql.txt" in dockerfile
        assert "v.startswith('2.0.0rc2') and v.endswith('+native')" in \
            dockerfile, engine

    @pytest.mark.parametrize("rel", (
        "harness/drivers/mysql_family.py",
        "overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py"))
    def test_the_implementation_is_chosen_before_the_import(self, rel):
        """2.0 reads MARIADB_PYTHON_CONNECTOR once, at import time."""
        src = _source(rel)
        choose = src.index(
            'os.environ.setdefault("MARIADB_PYTHON_CONNECTOR", "python")')
        assert choose < src.index("import mariadb as _connector"), rel


# ---------------------------------------------------------------------------
# 5. What 2.0 changed that 1.1 did not do: the read timeout and deadlocks
# ---------------------------------------------------------------------------

class _Deadlock(Exception):
    errno = 1213


class _OtherError(Exception):
    errno = 1062


class DeadlockingCursor(FakeCursor):
    """Fails the first `fails` executes with `error`, then succeeds."""

    def __init__(self, fails, error=_Deadlock):
        super().__init__()
        self.fails = fails
        self.error = error

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self.fails:
            self.fails -= 1
            raise self.error("Deadlock found when trying to get lock")


def _both_paths():
    from tests.test_e1_percona_design import _load_overlay
    vb, _ = _load_overlay()
    return (("ops", mysql_family, (1, 2, 3, b"x")),
            ("ann", vb, (1, 2, b"x")))


class TestTheReadTimeoutIsNotTwoPointOhsDefault:
    """2.0 defaults socket_timeout to 30 s; run percona-mariadb-20261110 lost
    every ops unit to it (Percona's index build at exactly 30.0 s, MariaDB's
    1% filtered query). `0` does not disable it over TCP -- measured: a
    `SELECT SLEEP(33)` fails with no option and with `read_timeout=0`, and
    succeeds with 86400."""

    def test_the_ops_connection(self):
        kw = mysql_family.connect_kwargs(ConnectionSpec(host="h", port=1))
        assert kw["read_timeout"] == mysql_family.READ_TIMEOUT_S >= 86400

    def test_the_ann_connection(self):
        from tests.test_e1_percona_design import _load_overlay
        vb, _ = _load_overlay()
        obj = object.__new__(vb.VBMySQLBase)
        obj._socket = "/s"
        kw = vb.VBMySQLBase._conn_kwargs(obj)
        assert kw["read_timeout"] == vb.READ_TIMEOUT_S >= 86400


class TestADeadlockedBatchIsReRun:
    """8 parallel ann loaders deadlocked MariaDB (errno 1213) in the same
    run. Every batch is one autocommit statement, rolled back whole, so a
    retry writes exactly the failed rows."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda s: None)

    @pytest.mark.parametrize("path", ("ops", "ann"))
    def test_a_deadlock_is_retried_with_the_same_statement(self, path):
        _, mod, row = dict((p[0], p) for p in _both_paths())[path]
        cur = DeadlockingCursor(fails=2)
        mod.insert_batch(cur, [row, row])
        assert len(cur.calls) == 3
        assert cur.calls[0] == cur.calls[1] == cur.calls[2]

    @pytest.mark.parametrize("path", ("ops", "ann"))
    def test_the_retries_are_bounded(self, path):
        _, mod, row = dict((p[0], p) for p in _both_paths())[path]
        cur = DeadlockingCursor(fails=100)
        with pytest.raises(_Deadlock):
            mod.insert_batch(cur, [row])
        assert len(cur.calls) == mod.DEADLOCK_RETRIES + 1

    @pytest.mark.parametrize("path", ("ops", "ann"))
    def test_any_other_error_is_not_retried(self, path):
        _, mod, row = dict((p[0], p) for p in _both_paths())[path]
        cur = DeadlockingCursor(fails=1, error=_OtherError)
        with pytest.raises(_OtherError):
            mod.insert_batch(cur, [row])
        assert len(cur.calls) == 1

    def test_both_paths_agree(self):
        (_, ops, _), (_, ann, _) = _both_paths()
        assert ops.ER_LOCK_DEADLOCK == ann.ER_LOCK_DEADLOCK == 1213
        assert ops.DEADLOCK_RETRIES == ann.DEADLOCK_RETRIES
        assert ops.READ_TIMEOUT_S == ann.READ_TIMEOUT_S
