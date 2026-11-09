"""T3b: the second filter column, proved in every driver without a server.

Four client stacks are involved and none of them is installed on the host, so
every test here drives a real driver class with its connection faked and reads
back the SQL, the command argument vector, the BSON pipeline or the hash
mapping that the driver built. That is the whole point: the statements are
where this change lives, and a statement can be wrong in ways a running server
would answer happily -- a `ctag` column that is written but never indexed, a
predicate on `tag` sent for a measurement recorded as `ctag`, a value tuple
whose two scalars are in the wrong order.

Three habits run through the file, each because of a way an assertion in this
repo has looked like coverage and was not.

**Nothing asserts a substring where a longer string would satisfy it.** The
two column names are `tag` and `ctag`, and the index names are `tag_idx` and
`ctag_idx`: `"ctag" in sql` is satisfied by `ctag_idx`, `"tag" in sql` is
satisfied by `ctag`, and `"WHERE tag <" in sql` is satisfied by a query that
appends a second predicate. So the SQL assertions are `re.fullmatch` against a
literal, and the "this column is absent" assertions use `\\b` word boundaries,
which do not fire inside `ctag` (both `c` and `t` are word characters, so
there is no boundary between them).

**No fixture hands back what it was given.** The fakes record; they never echo.
No expected statement is produced by calling the builder under test, which
would assert `x == x`.

**No assertion runs over an empty collection.** A whitelist test over an empty
candidate list and a schema test over zero parsed columns both pass while
testing nothing, so the parsers and the candidate lists assert their own size
first.

`TestTheOldFormsDoNotCatchThese` at the bottom closes the loop the other way:
it reconstructs the pre-T3b shape of each guarded thing and asserts that the
old form passes the mutant the new form refuses. A repair that the old code
would also have survived is not a repair.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.drivers import base as base_mod  # noqa: E402
from harness.drivers import mongo as mongo_mod  # noqa: E402
from harness.drivers import mysql_family as mysql_mod  # noqa: E402
from harness.drivers import postgres as pg_mod  # noqa: E402
from harness.drivers import valkey as valkey_mod  # noqa: E402
from harness.drivers.base import (ConnectionSpec, IndexSpec,  # noqa: E402
                                  RowAttrs, checked_filter_column)
from harness.metrics import records as records_mod  # noqa: E402
from harness.workloads import build as build_workload  # noqa: E402
from harness.workloads.context import RunContext  # noqa: E402

from .fake_driver import (DIM, K, N_ROWS, FakeDriver,  # noqa: E402
                          index_spec, loaded_driver, tiny_dataset)

TAG = ds.FILTER_TAG
CTAG = ds.FILTER_CTAG
COLUMNS = ds.FILTER_COLUMNS

# Module scope, not a class attribute, because the whitelist has to be
# exercised through every driver path and not only against
# `checked_filter_column` directly. That gap was real and this tuple was in it:
# the first version of this file kept the list beside the unit test of the
# validator and pushed only `"ctag_idx"` through the drivers, so pgvector's
# `if column else ""` truthiness bypass -- which let `""`, `None` and `0`
# through to an UNFILTERED select while the other four drivers raised -- was
# invisible to all 1,516 tests.
#
# Each entry is a shape some weaker check would have let through: the index
# names (a substring test), the case variants (a lowercasing test), the
# whitespace variants (a strip-then-compare test), the qualified name (a regex
# for one or more word characters), the real columns of the table, the falsy
# values (a truthiness test), and the non-strings a hand-edited profile or a
# JSON round-trip can produce.
REFUSED_COLUMNS = (
    "ctag_idx", "tag_idx", "ctag_x", "xtag",
    "Tag", "TAG", "CTag",
    "tag ", " tag", "\ttag", "tag\n",
    "tag;--", "tag OR 1=1", "tag) --", "t1.tag", "tag, ctag", "tag,ctag",
    "", "v", "id", "embedding", "tags", "ctags",
    None, 0, 1, True, ("tag",), ["tag"], {"tag": 1},
)

#: The falsy members, called out because they are the ones a truthiness test
#: cannot see and the ones that produced the defect above.
FALSY_REFUSED = ("", None, 0)

# Two thresholds that are never equal and never zero, so "the other column's
# threshold was used" and "the threshold was dropped" are both visible.
TAG_T = 7
CTAG_T = 23


# ---------------------------------------------------------------------------
# Fakes. They record; they do not answer.
# ---------------------------------------------------------------------------

class RecordingCursor:
    """A DB-API-ish cursor that keeps every statement it was handed.

    `plan` is what `fetchall()` returns, so an EXPLAIN assertion can be given
    a plan that does or does not name the vector index. It is deliberately
    *not* derived from the statement: a cursor that echoed the SQL back as its
    own plan would make `explain_uses_vector_index` return True for any string
    containing the index name, including the statement it had just sent.
    """

    def __init__(self, plan=(), scalar=0):
        self.statements = []        # list of (sql, params)
        self.many = []              # list of (sql, rows)
        self.copies = []            # list of (sql, rows) for psycopg COPY
        self.kwargs = []            # list of execute() keyword dicts
        self._plan = [(row,) for row in plan]
        self._scalar = scalar

    # -- the recording surface ------------------------------------------
    def execute(self, sql, params=None, **kw):
        self.statements.append((sql, params))
        self.kwargs.append(dict(kw))

    def executemany(self, sql, rows):
        self.many.append((sql, list(rows)))

    def copy(self, sql):
        return _RecordingCopy(self, sql)

    def fetchall(self):
        return list(self._plan)

    def fetchone(self):
        return (self._scalar,)

    def close(self):
        pass

    # -- what the assertions read ---------------------------------------
    @property
    def sql(self):
        return [statement for statement, _ in self.statements]

    def only_statement(self):
        assert len(self.statements) == 1, self.statements
        return self.statements[0]


class _RecordingCopy:
    def __init__(self, cursor, sql):
        self._cursor = cursor
        self._sql = sql
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cursor.copies.append((self._sql, self._rows))
        return False

    def write_row(self, row):
        self._rows.append(tuple(row))


class RecordingValkey:
    """Enough of a Valkey client to record FT.CREATE, FT.SEARCH and HSET."""

    def __init__(self, search_reply=None, info=None):
        self.commands = []          # list of argument tuples
        self.hsets = []             # list of (key, mapping)
        self._search_reply = search_reply if search_reply is not None else [0]
        self._info = info if info is not None else {}

    def execute_command(self, *args):
        self.commands.append(tuple(args))
        if args[0] == "FT.SEARCH":
            return list(self._search_reply)
        if args[0] == "FT.INFO":
            flat = []
            for key, value in self._info.items():
                flat += [key, value]
            return flat
        return "OK"

    def pipeline(self, transaction=False):
        return _RecordingPipe(self)

    def info(self, *a):
        return self._info

    def dbsize(self):
        return len(self.hsets)

    def close(self):
        pass

    @property
    def ft_create(self):
        for args in self.commands:
            if args[0] == "FT.CREATE":
                return args
        raise AssertionError(f"no FT.CREATE was sent; got {self.commands}")

    @property
    def searches(self):
        return [args for args in self.commands if args[0] == "FT.SEARCH"]


class _RecordingPipe:
    def __init__(self, conn):
        self._conn = conn

    def hset(self, key, mapping=None):
        self._conn.hsets.append((key, dict(mapping or {})))

    def execute(self):
        return []


class RecordingCollection:
    """A pymongo-ish collection that records definitions and documents."""

    def __init__(self):
        self.index_definitions = []
        self.documents = []
        self.pipelines = []

    def create_search_index(self, definition):
        self.index_definitions.append(definition)

    def insert_many(self, docs, ordered=True):
        self.documents.extend(docs)

    def aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        return []

    def list_search_indexes(self):
        return [{"name": mongo_mod.INDEX_NAME, "status": "READY"}]


class RecordingDatabase:
    def __init__(self):
        self.commands = []

    def command(self, *args, **kw):
        self.commands.append((args, kw))
        return {"stages": [{"$vectorSearch": {}}]}

    def create_collection(self, name):
        pass

    def drop_collection(self, name):
        pass


# ---------------------------------------------------------------------------
# Driver construction. No `connect()` anywhere: none of the four client
# libraries is installed on the host, which is the constraint that makes these
# tests runnable at all.
# ---------------------------------------------------------------------------

def spec():
    return ConnectionSpec(host="fake", port=0)


def mysql_driver(cls=None, plan=("vi",), metric="euclidean"):
    driver = (cls or mysql_mod.MariaDBDriver)(spec())
    driver._cur = RecordingCursor(plan=plan)
    driver._index = IndexSpec(dim=DIM, m=16, metric=metric)
    return driver


def pg_driver(cls=None, plan=(f"Index Scan using {pg_mod.INDEX}",),
              metric="euclidean"):
    driver = (cls or pg_mod.PostgresDriver)(spec())
    driver._cur = RecordingCursor(plan=plan)
    driver._index = IndexSpec(dim=DIM, m=16, metric=metric)
    return driver


class FakePymongoModule:
    """Just enough of `pymongo` for `_insert_range` to open its own client.

    It deliberately does: one `MongoClient` per worker is how that driver
    keeps the thread count a measure of write concurrency rather than a
    setting on a shared pool. So the stub has to route that fresh client back
    to the *same* collection the driver's `_coll` points at -- a stub that
    handed back a second, empty recorder would leave the documents somewhere
    the test does not read, and `len(docs) == rows` below is what would catch
    that rather than a silent pass.
    """

    def __init__(self, collection):
        self._collection = collection

    def MongoClient(self, *args, **kwargs):
        return _FakeMongoClient(self._collection)


class _FakeMongoClient:
    def __init__(self, collection):
        self._collection = collection
        self.admin = _FakeAdmin()

    def __getitem__(self, name):
        return _FakeMongoDatabase(self._collection)

    def close(self):
        pass


class _FakeMongoDatabase:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, name):
        return self._collection


class _FakeAdmin:
    def command(self, *args, **kw):
        return {"ok": 1}


def mongo_driver(cls=None, metric="euclidean"):
    driver = (cls or mongo_mod.MongoDriver)(spec())
    driver._coll = RecordingCollection()
    driver._db = RecordingDatabase()
    driver._index = IndexSpec(dim=DIM, m=16, metric=metric)
    # Restored by the `mongo_module_restored` fixture, which pins the
    # attribute for teardown before any test touches it.
    mongo_mod.pymongo = FakePymongoModule(driver._coll)
    return driver


def valkey_driver(cls=None, metric="euclidean", **conn_kwargs):
    driver = (cls or valkey_mod.ValkeyDriver)(spec())
    driver._conn = RecordingValkey(**conn_kwargs)
    driver._index = IndexSpec(dim=DIM, m=16, metric=metric)
    # The bulk and churn write paths open connections of their own, on leashes
    # of their own; one recorder stands in for all of them so a test can see
    # every HSET from every path in one list.
    driver._write_connection = lambda timeout_s=None: driver._conn
    return driver


#: Name -> a factory that returns a driver with its connection recorded, for
#: the tests that must hold for every engine. A dict rather than four separate
#: tests because item 1 and item 2 are claims about *all* the drivers, and the
#: failure they exist to catch is one driver being left behind -- which a test
#: written against three of them cannot see.
DRIVER_FACTORIES = {
    "mariadb": mysql_driver,
    "alisql": lambda: mysql_driver(cls=mysql_mod.AliSQLDriver),
    "pgvector": pg_driver,
    "mongodb": mongo_driver,
    "valkey": valkey_driver,
}

ALL_DRIVER_CLASSES = {
    "mariadb": mysql_mod.MariaDBDriver,
    "alisql": mysql_mod.AliSQLDriver,
    "pgvector": pg_mod.PostgresDriver,
    "mongodb": mongo_mod.MongoDriver,
    "valkey": valkey_mod.ValkeyDriver,
    "fake": FakeDriver,
}


@pytest.fixture(autouse=True)
def mongo_module_restored(monkeypatch):
    """Put `mongo.pymongo` back after every test in this module.

    `mongo_driver()` replaces it with a stub, and it is a module global rather
    than something a driver instance owns. Setting it to its current value
    here is what records the original for monkeypatch to restore, so no test
    in the rest of the suite can be handed this file's stub.
    """
    monkeypatch.setattr(mongo_mod, "pymongo", mongo_mod.pymongo,
                        raising=False)


@pytest.fixture(autouse=True)
def bson_binary(monkeypatch):
    """`bson.binary.Binary` is None on the host, and `encode_vector` calls it.

    pymongo lives only in the bench image, so the Mongo driver's own encoder
    raises TypeError on a developer machine. Standing in for `Binary` with the
    payload itself is what `tests/test_config.py` already does; the payload is
    what the assertions here care about, and the subtype tag is pymongo's
    business rather than this driver's. Left alone where pymongo is present,
    so inside the image these tests exercise the real encoder.
    """
    if mongo_mod.Binary is None:
        monkeypatch.setattr(mongo_mod, "Binary",
                            lambda payload, subtype: bytes(payload))


def vector():
    return numpy.arange(DIM, dtype=numpy.float32) + 1.0


def distinct_attrs(n: int) -> RowAttrs:
    """Two columns that disagree on every row.

    The whole file would be vacuous without this. If `tag` and `ctag` held the
    same values, a driver that read one where it should have read the other,
    or wrote one into the other's column, would produce identical output and
    every assertion below would pass.
    """
    attrs = RowAttrs(
        tag=numpy.arange(n, dtype=numpy.int64) % 10,
        ctag=900 + (numpy.arange(n, dtype=numpy.int64) % 10),
    )
    assert not numpy.array_equal(attrs.tag, attrs.ctag)
    # Disjoint ranges, so a value found in the wrong column cannot be mistaken
    # for a correct one by coincidence.
    assert set(attrs.tag.tolist()).isdisjoint(set(attrs.ctag.tolist()))
    return attrs


# ---------------------------------------------------------------------------
# Parsers for the statements the drivers build. Each refuses to return an
# empty answer, because a schema assertion over zero parsed columns is the
# vacuity shape that has bitten this repo.
# ---------------------------------------------------------------------------

_MYSQL_INDEX = re.compile(
    r"^\s*(PRIMARY KEY|UNIQUE KEY|FULLTEXT KEY|SPATIAL KEY|VECTOR INDEX|KEY)"
    r"(?:\s+\w+)?\s*\(([^)]*)\)", re.IGNORECASE)
_MYSQL_INLINE_PK = re.compile(r"^\s*(\w+)\s+\w+[^,]*?\bPRIMARY KEY\b",
                              re.IGNORECASE)


def mysql_index_kinds(ddl: str):
    """column -> set of index kinds declared on it, from a CREATE TABLE.

    Index *parity* is the claim item 2 makes, and parity is a relationship
    between two columns rather than the presence of a line. Asserting that
    `ctag_idx` appears somewhere in the DDL would pass a schema that gave
    `tag` a KEY and `ctag` a UNIQUE KEY, or one that indexed `ctag` and
    dropped the index on `tag`.
    """
    kinds = {}
    for line in ddl.splitlines():
        inline = _MYSQL_INLINE_PK.match(line)
        if inline:
            kinds.setdefault(inline.group(1), set()).add("PRIMARY KEY")
            continue
        found = _MYSQL_INDEX.match(line)
        if not found:
            continue
        kind = found.group(1).upper()
        for column in found.group(2).split(","):
            kinds.setdefault(column.strip(), set()).add(kind)
    assert kinds, f"parsed no indexes at all out of:\n{ddl}"
    return kinds


_MYSQL_COLUMN = re.compile(r"^\s*(\w+)\s+(INT|VECTOR\(\d+\))\b(.*)$",
                           re.IGNORECASE)


def mysql_column_types(ddl: str):
    """column -> (type, rest-of-declaration), from a CREATE TABLE."""
    out = {}
    for line in ddl.splitlines():
        found = _MYSQL_COLUMN.match(line)
        if found:
            out[found.group(1)] = (found.group(2).upper(),
                                   found.group(3).strip().rstrip(","))
    assert out, f"parsed no columns at all out of:\n{ddl}"
    return out


_PG_CREATE_INDEX = re.compile(
    r"CREATE INDEX (\w+) ON \w+(?: USING (\w+))? \(([^)]*)\)", re.IGNORECASE)


def pg_index_kinds(statements):
    """column -> set of index methods, from recorded CREATE INDEX statements."""
    kinds = {}
    for sql in statements:
        found = _PG_CREATE_INDEX.search(sql)
        if not found:
            continue
        method = (found.group(2) or "btree").lower()
        for column in found.group(3).split(","):
            kinds.setdefault(column.strip().split()[0], set()).add(method)
    assert kinds, f"parsed no CREATE INDEX out of {statements}"
    return kinds


def mysql_inserted_rows(cur):
    """Every row a MySQL-family driver wrote through `cur`, in order.

    The MySQL-family write path is one multi-row INSERT per batch
    (`mysql_family.insert_batch`), never `executemany`, so the rows are the
    flat parameter list of each INSERT cut back into `INSERT_ROW`-sized
    tuples. `cur.many` staying empty is asserted, because a row that went
    through `executemany` would be one this helper does not see.
    """
    assert not cur.many, "MySQL-family writes must not use executemany"
    width = mysql_mod.INSERT_ROW.count("?")
    rows = []
    for sql, params in cur.statements:
        if sql.startswith(mysql_mod.INSERT_HEAD):
            assert len(params) % width == 0, sql
            rows.extend(tuple(params[i:i + width])
                        for i in range(0, len(params), width))
    return rows


def insert_columns_and_placeholders(sql: str):
    """(column names, placeholders) for an INSERT ... VALUES statement."""
    found = re.search(r"\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)", sql,
                      re.IGNORECASE)
    assert found, f"not an INSERT ... VALUES: {sql!r}"
    columns = [c.strip() for c in found.group(1).split(",")]
    placeholders = [p.strip() for p in found.group(2).split(",")]
    assert columns and placeholders
    return columns, placeholders


def copy_columns(sql: str):
    found = re.search(r"COPY \w+ \(([^)]*)\) FROM STDIN", sql, re.IGNORECASE)
    assert found, f"not a COPY: {sql!r}"
    columns = [c.strip() for c in found.group(1).split(",")]
    assert columns
    return columns


def ft_create_schema(args):
    """field -> declared type, from an FT.CREATE argument vector."""
    args = [str(a) for a in args]
    start = args.index("SCHEMA") + 1
    out = {}
    i = start
    while i + 1 < len(args):
        field, kind = args[i], args[i + 1]
        if kind == "VECTOR":
            break
        out[field] = kind
        i += 2
    assert out, f"parsed no scalar fields out of {args}"
    return out


# ===========================================================================
# The whitelist (§3.2, the "column from a whitelist, never interpolated from
# input" clause that appears in the MySQL and PostgreSQL rows of the table)
# ===========================================================================

class _ColumnThatLiesWhenFormatted(str):
    """Compares equal to a legal column and formats as something else.

    Not a contrivance for its own sake: it is the smallest object that tells
    "the validator checked the name" apart from "the validator returned the
    name that reaches the query string". Those are the same thing only if the
    value compared and the value interpolated are the same object, which for a
    `str` subclass they need not be.
    """

    def __format__(self, spec):
        return "tag OR 1=1"

    def __str__(self):
        return "tag OR 1=1"


class TestTheColumnIsNeverInterpolatedFromInput:
    def test_what_is_returned_is_the_whitelist_s_own_string(self):
        """Regression: `checked_filter_column` returning its argument.

        Honest: `f"{checked_filter_column(liar)}"` == 'tag'.
        Mutant (`return column` instead of the matched element):
        'tag OR 1=1' -- and that is what lands in `WHERE {column} < %s`.
        """
        liar = _ColumnThatLiesWhenFormatted(TAG)
        # The premise: it really does compare equal, so a membership check
        # alone cannot refuse it.
        assert liar == TAG and liar in COLUMNS
        assert f"{liar}" == "tag OR 1=1"

        checked = checked_filter_column(liar)
        assert f"{checked}" == TAG
        assert str(checked) == TAG
        assert type(checked) is str

    def test_every_legal_column_survives(self):
        assert len(COLUMNS) >= 2, "a whitelist of one column proves nothing"
        for column in COLUMNS:
            assert checked_filter_column(column) == column

    def test_everything_else_is_refused(self):
        assert len(REFUSED_COLUMNS) > 10
        # The near-misses have to be in the list, or this is a test that only
        # refuses obvious nonsense.
        for required in ("ctag_idx", "tag_idx", "Tag", "tag ", "t1.tag", ""):
            assert required in REFUSED_COLUMNS
        # And the falsy ones, which are the entries the driver-level loop
        # below exists for.
        for falsy in FALSY_REFUSED:
            assert falsy in REFUSED_COLUMNS and not falsy
        for candidate in REFUSED_COLUMNS:
            with pytest.raises(ValueError):
                checked_filter_column(candidate)

    def test_the_refusal_names_the_whole_legal_set(self):
        """Not `"tag" in message`, which `"ctag"` satisfies.

        The message has to be useful for the mistake that actually happens --
        a column name that is nearly right -- so it states the legal set.
        """
        with pytest.raises(ValueError) as caught:
            checked_filter_column("ctag_idx")
        message = str(caught.value)
        assert str(list(COLUMNS)) in message
        assert repr("ctag_idx") in message

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_the_checked_name_is_what_reaches_the_statement(self, name):
        """The spec clause, pinned where the spec puts it: at the driver.

        `test_what_is_returned_is_the_whitelist_s_own_string` above proves
        `checked_filter_column` hands back the tuple's element rather than the
        caller's object -- one level BELOW the statement. It does not prove
        any driver uses the returned value: a driver that called the whitelist
        for its side effect and then interpolated its own argument would pass
        it. Measured: giving all five drivers exactly that shape
        (`checked_filter_column(column); where = f"WHERE {column} < %s"`)
        failed ZERO of 1,516 tests, with the mysql statement reading
        '... WHERE tag OR 1=1 < %s ORDER BY ...' against an honest
        '... WHERE tag < %s ORDER BY ...'.

        The assertion is an equality between two runs through the public API
        -- the liar and the plain string -- rather than a literal per dialect,
        because the property is dialect-independent: the object passed in must
        not be able to change the bytes sent. Not a tautology; under the
        mutant the two differ.

        Defence in depth rather than a live hole: YAML cannot produce an
        object with a lying `__format__`, so the column a run really passes
        is always an ordinary string. It is the clause §3.2 states, and this
        file already owned the discriminating object.
        """
        plain = DRIVER_FACTORIES[name]()
        plain.query_filtered(vector(), K, TAG_T, column=TAG)
        honest = last_predicate(name, plain)

        liar = DRIVER_FACTORIES[name]()
        liar.query_filtered(vector(), K, TAG_T,
                            column=_ColumnThatLiesWhenFormatted(TAG))
        sent = last_predicate(name, liar)

        assert sent == honest, f"{name}: {sent!r} != {honest!r}"
        assert "OR 1=1" not in sent, f"{name}: {sent!r}"
        assert re.search(r"\btag\b", sent), f"{name}: {sent!r}"

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_the_checked_name_reaches_the_explain_statement_too(self, name):
        """And the plan check, which is the path that swallows exceptions."""
        plain = DRIVER_FACTORIES[name]()
        plain.explain_uses_vector_index(vector(), K, tag_threshold=TAG_T,
                                        column=TAG)
        honest = last_predicate(name, plain)

        liar = DRIVER_FACTORIES[name]()
        liar.explain_uses_vector_index(
            vector(), K, tag_threshold=TAG_T,
            column=_ColumnThatLiesWhenFormatted(TAG))
        assert last_predicate(name, liar) == honest, name

    def test_the_whitelist_is_the_dataset_module_s_own_tuple(self):
        """Regression: a second literal list of legal columns in base.py.

        Identity, not equality: an equal copy drifts the moment one of the two
        is edited, and the symptom is a column the dataset module computes
        ground truth for and the drivers refuse -- or the reverse, which is
        worse, because then a predicate is built on a column whose values
        nobody assigned.
        """
        assert base_mod.FILTER_COLUMNS is ds.FILTER_COLUMNS
        assert base_mod.FILTER_TAG is ds.FILTER_TAG
        assert base_mod.FILTER_CTAG is ds.FILTER_CTAG


# ===========================================================================
# Where the check sits. §3.5's bullet asks for "the column whitelist rejects
# anything except tag and ctag"; the discriminating question is whether every
# path that builds a predicate goes through it -- there are two in
# postgres.py alone.
# ===========================================================================

def last_predicate(name, driver):
    """The predicate the driver most recently put on the wire, as a string.

    One per dialect: SQL text, a JSON-dumped `$vectorSearch.filter`, or the
    FT.SEARCH query. Reads what the recorder captured rather than asking the
    driver to rebuild it, so an assertion on the result is an assertion about
    what would have been sent.
    """
    if name in ("mariadb", "alisql", "pgvector"):
        return driver._cur.sql[-1]
    if name == "mongodb":
        # The one driver whose plan check does not go through the same object
        # as its query: `aggregate` lands on the collection and `explain`
        # lands on the database, so both recorders are consulted and the more
        # recent one wins.
        if driver._db.commands:
            args, _kw = driver._db.commands[-1]
            pipeline = args[0]["explain"]["pipeline"]
        elif driver._coll.pipelines:
            pipeline = driver._coll.pipelines[-1]
        else:
            raise AssertionError("mongo driver sent nothing")
        return json.dumps(pipeline[0]["$vectorSearch"]["filter"])
    return driver._conn.searches[-1][2]


def predicate_paths(name, driver):
    """Every call on `driver` that can produce a filtered predicate.

    Returned as callables taking a column, so a test can run all of them
    against the same hostile input. Enumerated per driver rather than guessed,
    because the whole risk is a path nobody listed.
    """
    paths = {
        "query_filtered":
            lambda column: driver.query_filtered(vector(), K, TAG_T,
                                                 column=column),
        "explain_uses_vector_index":
            lambda column: driver.explain_uses_vector_index(
                vector(), K, tag_threshold=TAG_T, column=column),
    }
    if name in ("mariadb", "alisql"):
        paths["_select"] = lambda column: driver._select(
            K, True, column=column)
    if name == "pgvector":
        paths["_select"] = lambda column: driver._select(
            True, column=column)
    if name == "mongodb":
        paths["_pipeline"] = lambda column: driver._pipeline(
            vector(), K, TAG_T, column)
    if name == "valkey":
        paths["_filter_expr"] = lambda column: driver._filter_expr(column,
                                                                   TAG_T)
    return paths


def sent_commands(name, driver):
    """How much the driver has put on the wire, as one comparable number."""
    if name in ("mariadb", "alisql", "pgvector"):
        cursor = driver._cur
        return len(cursor.statements) + len(cursor.many) + len(cursor.copies)
    if name == "mongodb":
        return (len(driver._coll.pipelines) + len(driver._coll.documents)
                + len(driver._db.commands)
                + len(driver._coll.index_definitions))
    return len(driver._conn.commands) + len(driver._conn.hsets)


class TestEveryPredicatePathIsGuarded:
    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_a_bad_column_is_refused_on_every_path(self, name):
        """Regression: the check added to `query_filtered` only.

        Honest: each path raises ValueError and sends nothing (0 commands).
        Mutant (`checked_filter_column` replaced by the identity): no path
        raises, and each one puts a statement on the wire carrying
        `ctag_idx` -- 3 or 4 commands depending on the driver.
        """
        paths = predicate_paths(name, DRIVER_FACTORIES[name]())
        assert len(paths) >= 3, f"{name}: only {sorted(paths)} enumerated"
        for label in paths:
            driver = DRIVER_FACTORIES[name]()
            before = sent_commands(name, driver)
            with pytest.raises(ValueError):
                predicate_paths(name, driver)[label]("ctag_idx")
            # The refusal is before the wire, not after it. A check that sat
            # below the execute would still raise -- and would already have
            # sent the predicate.
            assert sent_commands(name, driver) == before, (
                f"{name}.{label} sent something before refusing")

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_every_refused_column_is_refused_on_every_path(self, name):
        """The whole candidate list, not one element of it.

        The sibling test above pushed only `"ctag_idx"` through the drivers,
        and `REFUSED_COLUMNS` was exercised only against
        `checked_filter_column` directly. That gap hid a live defect for a
        review cycle: `postgres._select` read `if column else ""`, so the
        three FALSY candidates skipped the predicate and the whitelist with
        it and produced an UNFILTERED select, while the other four drivers
        raised on the same input. Measured at the time: `pg._select("")` ->
        'SELECT id FROM t1 ORDER BY embedding <-> %s LIMIT %s' against
        `pg._select("tag")` -> '... WHERE tag < %s ORDER BY ...'. Giving all
        five drivers the same truthiness bypass failed ZERO of 1,516 tests.

        So every candidate goes through every path of every driver. 31
        candidates x 3-4 paths x 5 drivers, and it runs in milliseconds
        because nothing opens a connection.
        """
        assert len(REFUSED_COLUMNS) > 10
        for label in predicate_paths(name, DRIVER_FACTORIES[name]()):
            for candidate in REFUSED_COLUMNS:
                driver = DRIVER_FACTORIES[name]()
                before = sent_commands(name, driver)
                with pytest.raises(ValueError):
                    predicate_paths(name, driver)[label](candidate)
                assert sent_commands(name, driver) == before, (
                    f"{name}.{label} sent something for {candidate!r}")

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_a_falsy_column_is_not_quietly_an_unfiltered_query(self, name):
        """The defect above, stated as the thing that made it dangerous.

        A refusal is one outcome; returning a DIFFERENT, well-formed,
        unfiltered query is another, and only the second one gets recorded as
        a measurement. So this asserts the shape of the failure and not only
        that one occurred: whatever `query_filtered` does with a falsy
        column, it does not put a predicate-free statement on the wire.
        """
        for falsy in FALSY_REFUSED:
            driver = DRIVER_FACTORIES[name]()
            with pytest.raises(ValueError):
                driver.query_filtered(vector(), K, TAG_T, column=falsy)
            assert sent_commands(name, driver) == 0, (
                f"{name}: a falsy column produced a statement")

    # (driver, a call offering a column in the FLAG's position, a call
    # offering it positionally where `column` used to sit, and the correct
    # call). Only the two SQL builders take a flag, which is why this is a
    # two-entry table rather than a parametrisation over DRIVER_FACTORIES:
    # `valkey._filter_expr(column, threshold)` takes the column FIRST, so a
    # misplaced one lands in the whitelist (`_filter_expr(7, "ctag")` ->
    # ValueError: 7 is not a filter column), and `mongo._pipeline(vector, k,
    # tag_threshold, column)` puts it after the threshold, so the same
    # mistake hits `int("ctag")` -> ValueError. Both measured; neither needs
    # a marker.
    FLAG_POSITION_CALLS = (
        ("mariadb", mysql_driver,
         lambda d, c: d._select(K, c),
         lambda d, c: d._select(K, True, c),
         lambda d, c: d._select(K, True, column=c)),
        ("pgvector", pg_driver,
         lambda d, c: d._select(c),
         lambda d, c: d._select(True, c),
         lambda d, c: d._select(True, column=c)),
    )

    def test_a_column_offered_as_the_flag_is_refused(self):
        """Regression: the last way this file can measure `tag` while a
        record says `ctag`.

        A caller who forgets the flag writes `_select("ctag")`, and a
        non-empty string is truthy, so before this guard that returned
        `WHERE tag < %s` -- a well-formed filtered query on the wrong column,
        with no error anywhere. Measured, honest vs mutant:
          pg._select("ctag")     -> TypeError naming the flag
                                 vs 'SELECT id FROM t1 WHERE tag < %s ...'
          my._select(10, "ctag") -> TypeError naming the flag
                                 vs 'SELECT id FROM t1 WHERE tag < %s ...'

        Worth stating because it is the reason this test asserts TWO shapes
        rather than one: making `column` keyword-only does NOT catch this.
        Measured against exactly that signature, `_select("ctag")` still
        returned `WHERE tag < %s`. The `*` stops a column arriving in
        `column`'s position; the `isinstance(filtered, bool)` check stops one
        arriving in the flag's. Both are asserted below, so deleting either
        one fails here.
        """
        assert len(self.FLAG_POSITION_CALLS) == 2
        for name, factory, as_flag, positional, correct in \
                self.FLAG_POSITION_CALLS:
            # A column where the flag goes: the runtime guard.
            with pytest.raises(TypeError) as caught:
                as_flag(factory(), CTAG)
            message = str(caught.value)
            assert "flag" in message and repr(CTAG) in message, message
            # And it names the way out, because the next person to hit this
            # is reading the traceback and nothing else.
            assert f"column={CTAG!r}" in message, message

            # A column passed positionally: the keyword-only marker.
            with pytest.raises(TypeError) as caught:
                positional(factory(), CTAG)
            assert "positional" in str(caught.value), str(caught.value)

            # And the correct call is untouched and really does use `ctag`.
            sql = correct(factory(), CTAG)
            assert re.search(r"\bctag\b", sql), sql
            assert re.search(r"\btag\b", sql) is None, sql

    def test_a_truthy_non_bool_flag_cannot_produce_a_predicate(self):
        """The guard is about the type, not about the string `"ctag"`.

        `1`, a non-empty tuple and an ndarray are all truthy, and any of them
        arriving in the flag's position means a caller got the signature
        wrong. The honest outcome is the same TypeError for all of them; the
        mutant (no guard) builds a filtered statement for every one.
        """
        truthy = (CTAG, TAG, 1, "yes", ("tag",), [TAG])
        for name, factory, as_flag, _positional, _correct in \
                self.FLAG_POSITION_CALLS:
            for value in truthy:
                with pytest.raises(TypeError):
                    as_flag(factory(), value)
        # `True` and `False` are the only accepted values, and both work.
        assert "WHERE" in mysql_driver()._select(K, True)
        assert "WHERE" not in mysql_driver()._select(K, False)
        assert "WHERE" in pg_driver()._select(True)
        assert "WHERE" not in pg_driver()._select(False)

    def test_the_other_two_builders_refuse_the_same_mistake_already(self):
        """Why only two drivers are in the table above, measured rather than
        asserted by inspection.

        `valkey._filter_expr` takes the column first, so a misplaced column
        lands in the whitelist; `mongo._pipeline` takes it after the
        threshold, so the mistake hits `int()`. Neither needs a keyword-only
        marker, and recording that here is what stops someone adding one to
        all four on the grounds of symmetry.
        """
        with pytest.raises(ValueError):
            valkey_driver()._filter_expr(TAG_T, CTAG)
        with pytest.raises(ValueError):
            mongo_driver()._pipeline(vector(), K, CTAG)

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_the_two_query_paths_share_one_builder(self, name):
        """Regression: `explain` building its own copy of the predicate.

        Two copies means two places to add the whitelist to, and the one that
        gets missed is the plan check -- which runs once per configuration and
        whose output goes into the report's Validity section, so a wrong
        column there is a claim about the wrong query.

        Honest: patching the single builder is observed by both paths, 2 calls.
        Mutant (`explain_uses_vector_index` restored to its pre-T3b body,
        which formatted its own string): 1 call.
        """
        driver = DRIVER_FACTORIES[name]()
        builders = {
            "mariadb": "_select", "alisql": "_select", "pgvector": "_select",
            "mongodb": "_pipeline", "valkey": "_filter_expr",
        }
        attribute = builders[name]
        original = getattr(driver, attribute)
        seen = []

        def spy(*args, **kw):
            seen.append((args, kw))
            return original(*args, **kw)

        setattr(driver, attribute, spy)
        driver.query_filtered(vector(), K, TAG_T, column=CTAG)
        driver.explain_uses_vector_index(vector(), K, tag_threshold=TAG_T,
                                         column=CTAG)
        assert len(seen) == 2, (
            f"{name}: {attribute} was called {len(seen)} times, so one of the "
            f"two paths builds its predicate itself")

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_a_driver_cannot_keep_a_whitelist_of_its_own(self, name,
                                                         monkeypatch):
        """Regression: a per-driver tuple of legal columns.

        `checked_filter_column` reads `FILTER_COLUMNS` as a module global at
        call time, so narrowing it here reaches whichever validator the driver
        really uses. A driver with its own list would carry on accepting
        `ctag` and this would fail.

        Honest: with the whitelist narrowed to ('tag',), every path refuses
        `ctag`. Mutant (a driver that checks `column in ("tag", "ctag")`
        itself): `ctag` is accepted and the predicate is built.
        """
        monkeypatch.setattr(base_mod, "FILTER_COLUMNS", (TAG,))
        # The premise, so this does not pass because the patch did nothing.
        with pytest.raises(ValueError):
            checked_filter_column(CTAG)
        for label, call in predicate_paths(
                name, DRIVER_FACTORIES[name]()).items():
            with pytest.raises(ValueError):
                call(CTAG)


# ===========================================================================
# §3.2 item 1 -- the interface, in base.py and in every implementation
# ===========================================================================

class TestTheInterfaceItem1:
    def test_the_base_signatures_are_what_the_spec_writes(self):
        """`query_filtered(vector, k, threshold, column="tag")` and
        `explain_uses_vector_index(..., column="tag")`."""
        parameters = inspect.signature(
            base_mod.EngineDriver.query_filtered).parameters
        assert list(parameters) == ["self", "vector", "k", "threshold",
                                    "column"]
        assert parameters["column"].default == TAG
        assert parameters["threshold"].default is inspect.Parameter.empty

        explain = inspect.signature(
            base_mod.EngineDriver.explain_uses_vector_index).parameters
        assert list(explain) == ["self", "vector", "k", "tag_threshold",
                                 "column"]
        assert explain["column"].default == TAG
        assert explain["tag_threshold"].default is None

    @pytest.mark.parametrize("name", sorted(ALL_DRIVER_CLASSES))
    def test_every_implementation_takes_the_column(self, name):
        """Regression: the column plumbed into three drivers and not the
        fourth.

        That driver would accept `column=` only if it inherited the abstract
        method -- none do -- and otherwise raise TypeError, which is loud. The
        quiet version is the next test.
        """
        cls = ALL_DRIVER_CLASSES[name]
        for method, third in (("query_filtered", "threshold"),
                              ("explain_uses_vector_index", "tag_threshold")):
            parameters = inspect.signature(
                getattr(cls, method)).parameters
            assert list(parameters)[:4] == ["self", "vector", "k", third], (
                f"{name}.{method}: {list(parameters)}")
            assert "column" in parameters, f"{name}.{method}"
            assert parameters["column"].default == TAG, f"{name}.{method}"

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_the_column_reaches_the_wire(self, name):
        """Regression: a driver that accepts `column=` and ignores it.

        A signature test cannot see this, and it is the worst failure in PR 3:
        the record would say `filter_column=ctag` while the engine answered a
        `tag` predicate, so the correlated case would be reported as costing
        nothing.

        Honest: the command carries `ctag` and not a bare `tag`. Mutant
        (`column` dropped on the floor inside the builder): the command
        carries `tag`.
        """
        driver = DRIVER_FACTORIES[name]()
        driver.query_filtered(vector(), K, CTAG_T, column=CTAG)
        sent = last_predicate(name, driver)
        assert re.search(r"\bctag\b", sent), f"{name}: {sent!r}"
        # \b does not fire between the 'c' and the 't' of "ctag", so this is
        # not satisfied by the ctag predicate itself.
        assert re.search(r"\btag\b", sent) is None, f"{name}: {sent!r}"

    @pytest.mark.parametrize("name", sorted(ALL_DRIVER_CLASSES))
    def test_the_call_shapes_filtered_py_already_uses_still_bind(self, name):
        """The compatibility claim T3c depends on, checked rather than hoped.

        `harness/workloads/filtered.py` is T3c's file and was not edited here.
        It calls `query_filtered(q, k, threshold)` positionally at lines 110
        and 125 and `explain_uses_vector_index(q, k, tag_threshold=t)` by
        keyword at line 104. Both have to keep binding, or the suite breaks on
        a file this task may not touch.
        """
        cls = ALL_DRIVER_CLASSES[name]
        inspect.signature(cls.query_filtered).bind(
            object(), vector(), K, TAG_T)
        inspect.signature(cls.explain_uses_vector_index).bind(
            object(), vector(), K, tag_threshold=TAG_T)

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_a_three_argument_call_still_filters_on_tag(self, name):
        """And it filters on `tag`, not on whatever the default happened to
        become. A default of `ctag` would silently relabel every filtered
        measurement taken before T3c lands."""
        driver = DRIVER_FACTORIES[name]()
        driver.query_filtered(vector(), K, TAG_T)
        sent = last_predicate(name, driver)
        assert re.search(r"\btag\b", sent), f"{name}: {sent!r}"
        assert re.search(r"\bctag\b", sent) is None, f"{name}: {sent!r}"


# ===========================================================================
# §3.2 item 2 -- index parity, as a relationship rather than a line
# ===========================================================================

class TestIndexParityItem2:
    def test_mysql_family_indexes_both_columns_the_same_way(self):
        driver = mysql_driver()
        driver.create_schema(IndexSpec(dim=DIM, m=16, metric="euclidean"))
        ddl = driver._cur.sql[0]
        kinds = mysql_index_kinds(ddl)
        assert kinds.get(TAG), f"no index on {TAG} at all: {kinds}"
        assert kinds[TAG] == kinds[CTAG], (
            f"index kinds differ: {TAG}={kinds.get(TAG)} "
            f"{CTAG}={kinds.get(CTAG)}")
        # And the declarations match too: a NOT NULL column and a nullable one
        # are not the same thing to an optimiser deciding on a range scan.
        types = mysql_column_types(ddl)
        assert types[TAG] == types[CTAG] == ("INT", "NOT NULL")

    def test_pgvector_indexes_both_columns_the_same_way(self):
        driver = pg_driver()
        driver.create_schema(IndexSpec(dim=DIM, m=16, metric="euclidean",
                                       build_mode="post"))
        kinds = pg_index_kinds(driver._cur.sql)
        assert kinds.get(TAG) == {"btree"}
        assert kinds[TAG] == kinds[CTAG]
        # The HNSW index is not in this set in post mode, which is what makes
        # the equality above a statement about the scalar indexes only.
        assert "hnsw" not in kinds.get(TAG, set())

    def test_mongo_declares_a_filter_path_for_both_columns(self):
        driver = mongo_driver()
        driver._wait_until_ready = lambda: 0.0
        driver.create_index(IndexSpec(dim=DIM, m=16, metric="euclidean"))
        fields = driver._coll.index_definitions[0]["definition"]["fields"]
        filters = {f["path"] for f in fields if f["type"] == "filter"}
        assert filters == set(COLUMNS), filters
        # Equality, not a superset: a path declared here that the dataset
        # module does not compute values for is an index field that will
        # always be absent from the documents.
        vectors = [f for f in fields if f["type"] == "vector"]
        assert len(vectors) == 1

    def test_valkey_declares_both_columns_numeric(self):
        driver = valkey_driver()
        driver._wait_for_backfill = lambda: 0.0
        driver._used_memory = lambda: 0
        driver._index_exists = lambda: False
        driver.create_index(IndexSpec(dim=DIM, m=16, metric="euclidean"))
        schema = ft_create_schema(driver._conn.ft_create)
        assert schema == {TAG: "NUMERIC", CTAG: "NUMERIC"}, schema

    def test_dropping_one_column_s_index_is_what_parity_catches(self):
        """Rule 2, inline: the mutant, and the honest value beside it.

        Honest: `{'KEY'} == {'KEY'}`.
        Mutant (a MariaDB driver whose DDL omits `KEY ctag_idx (ctag)`):
        `{'KEY'} != set()`, and a test that merely looked for the string
        `ctag` in the DDL would still pass, because the column declaration
        `ctag INT NOT NULL` is still there.
        """
        class OneIndexShort(mysql_mod.MariaDBDriver):
            def create_schema(self, index):
                self._index = index
                self._cur.execute(
                    f"CREATE TABLE {mysql_mod.TABLE} (\n"
                    f"  id INT PRIMARY KEY,\n"
                    f"  tag INT NOT NULL,\n"
                    f"  ctag INT NOT NULL,\n"
                    f"  v VECTOR({index.dim}) NOT NULL,\n"
                    f"  KEY tag_idx (tag),\n"
                    f"  VECTOR INDEX vi (v) M={index.m} DISTANCE=euclidean\n"
                    f") ENGINE=InnoDB")

        driver = mysql_driver(cls=OneIndexShort)
        driver.create_schema(IndexSpec(dim=DIM, m=16, metric="euclidean"))
        ddl = driver._cur.sql[0]
        kinds = mysql_index_kinds(ddl)
        assert kinds[TAG] == {"KEY"}
        assert kinds.get(CTAG, set()) == set()
        assert kinds[TAG] != kinds.get(CTAG, set())
        # The weaker assertion this replaces, shown passing on the mutant.
        assert "ctag" in ddl


# ===========================================================================
# §3.2 table, row by row: the schema change and the query change
# ===========================================================================

class TestMySQLFamilyStatements:
    def test_the_filtered_select_is_exactly_one_predicate(self):
        """Regression: a second predicate appended, or the threshold dropped.

        `re.fullmatch`, because `"WHERE ctag < ?" in sql` is satisfied by a
        statement that also carries `AND tag < 0`.
        """
        driver = mysql_driver()
        for column in COLUMNS:
            sql = driver._select(K, True, column=column)
            assert re.fullmatch(
                rf"SELECT id FROM t1 WHERE {column} < \? "
                rf"ORDER BY vec_distance_euclidean\(v, \?\) LIMIT {K}",
                sql), sql

    def test_the_unfiltered_select_mentions_neither_column(self):
        """Regression: the new column leaking into the unfiltered query, which
        would change what every unfiltered QPS figure measures."""
        sql = mysql_driver()._select(K, False)
        assert "WHERE" not in sql
        assert re.search(r"\bctag\b", sql) is None, sql
        assert re.search(r"\btag\b", sql) is None, sql

    def test_the_insert_names_both_columns_with_matching_placeholders(self):
        """Regression: a column added to the list and not to the values.

        Honest: 4 columns, 4 placeholders, filter columns in FILTER_COLUMNS
        order. Mutant (a column added to the list only): 4 and 3.
        """
        columns, placeholders = insert_columns_and_placeholders(
            mysql_mod.INSERT_HEAD + mysql_mod.INSERT_ROW)
        assert len(columns) == len(placeholders) == 2 + len(COLUMNS)
        assert set(placeholders) == {"?"}
        assert [c for c in columns if c in COLUMNS] == list(COLUMNS)
        assert columns[0] == "id" and columns[-1] == "v"

    def test_load_writes_each_column_s_own_values(self):
        """Regression: `ctag` written from the tag array, or the two swapped.

        Honest: row 3 is `(3, 3, 903, <bytes>)`. Mutant (`RowAttrs.row`
        returning `(tag, tag)`): `(3, 3, 3, <bytes>)`. Mutant (the two
        swapped): `(3, 903, 3, <bytes>)`.
        """
        rows = 12
        attrs = distinct_attrs(rows)
        driver = mysql_driver()
        driver.load(numpy.zeros((rows, DIM), dtype=numpy.float32), attrs)
        sent = mysql_inserted_rows(driver._cur)
        assert len(sent) == rows
        for i, row in enumerate(sent):
            assert len(row) == 2 + len(COLUMNS)
            assert row[0] == i
            assert row[1] == int(attrs.tag[i])
            assert row[2] == int(attrs.ctag[i])
            assert isinstance(row[3], bytes)

    def test_the_load_and_the_reinsert_send_the_same_statement(self):
        """Regression: the column added to one write path only.

        Character-identical, which is the property the shared `INSERT_HEAD`
        and `INSERT_ROW` give. Honest: equal. Mutant (`insert_rows` restored to its pre-T3b
        three-column statement): the two differ, and the re-insert fails
        against a four-column table -- after the delete has already happened.
        """
        rows = 5
        attrs = distinct_attrs(rows)
        vectors = numpy.zeros((rows, DIM), dtype=numpy.float32)

        bulk = mysql_driver()
        bulk.load(vectors, attrs)
        churn = mysql_driver()
        churn.insert_rows(list(range(100, 100 + rows)), vectors, attrs)

        def first_insert(cur):
            return next(sql for sql, _ in cur.statements
                        if sql.startswith(mysql_mod.INSERT_HEAD))

        assert first_insert(bulk._cur) == first_insert(churn._cur)
        assert first_insert(bulk._cur) == (
            mysql_mod.INSERT_HEAD + ", ".join([mysql_mod.INSERT_ROW] * rows))
        # And the re-insert kept its own ids while using the shared statement.
        reinserted = mysql_inserted_rows(churn._cur)
        assert [row[0] for row in reinserted] == list(range(100, 105))
        assert [row[2] for row in reinserted] == [int(t) for t in attrs.ctag]

    def test_explain_sends_the_same_predicate_the_query_does(self):
        """Regression: the plan checked for one column and the measurement
        taken on the other -- the Validity section would then vouch for a
        query nobody ran.

        The prefix is `EXPLAIN FORMAT=TRADITIONAL ` and not `EXPLAIN `, which
        is what this assertion caught when it changed. `Dialect.plan_marker`
        is a value in the `type` column and only the tabular format has one;
        `explain_format` defaults to TRADITIONAL, so a bare EXPLAIN worked,
        but the variable is settable and reachable through `VB_SERVER_ARGS`
        and under TREE or JSON the marker vanishes -- recording a false
        negative about a server that did use the index. The SELECT the two
        paths share is still asserted to be byte-identical, which is what
        this test is actually about.
        """
        for column in COLUMNS:
            driver = mysql_driver()
            driver.query_filtered(vector(), K, CTAG_T, column=column)
            queried = driver._cur.sql[-1]
            driver.explain_uses_vector_index(vector(), K,
                                             tag_threshold=CTAG_T,
                                             column=column)
            explained = driver._cur.sql[-1]
            assert explained == "EXPLAIN FORMAT=TRADITIONAL " + queried

    def test_both_dialects_get_the_same_treatment(self):
        """MariaDB and AliSQL differ only in the Dialect values, and a
        column added to one path in a way that reached only one of them would
        be a difference between two engines that this file exists to keep
        identical."""
        statements = {}
        for cls in (mysql_mod.MariaDBDriver, mysql_mod.AliSQLDriver):
            driver = mysql_driver(cls=cls)
            driver.create_schema(IndexSpec(dim=DIM, m=16, metric="euclidean"))
            statements[cls.__name__] = mysql_index_kinds(driver._cur.sql[0])
        assert len(statements) == 2
        values = list(statements.values())
        assert values[0] == values[1]
        assert values[0][TAG] == values[0][CTAG]


class TestPostgresStatements:
    def test_the_table_declares_ctag_not_null(self):
        driver = pg_driver()
        driver.create_schema(IndexSpec(dim=DIM, m=16, metric="euclidean",
                                       build_mode="post"))
        ddl = driver._cur.sql[0]
        assert re.search(r"\bctag int NOT NULL\b", ddl), ddl
        assert re.search(r"\btag int NOT NULL\b", ddl), ddl

    def test_the_scalar_index_names_follow_the_columns(self):
        assert pg_mod.SCALAR_INDEXES == {
            TAG: "t1_tag_idx", CTAG: "t1_ctag_idx"}
        # Distinct names, and neither is a substring of the HNSW index's name
        # -- `explain_uses_vector_index` tests `INDEX in plan`, so a scalar
        # index whose name contained it would read as the vector index.
        assert len(set(pg_mod.SCALAR_INDEXES.values())) == len(COLUMNS)
        for name in pg_mod.SCALAR_INDEXES.values():
            assert pg_mod.INDEX not in name

    def test_the_copy_names_both_columns_in_order(self):
        columns = copy_columns(pg_mod.COPY_SQL)
        assert [c for c in columns if c in COLUMNS] == list(COLUMNS)
        assert columns == ["id", TAG, CTAG, "embedding"]

    def test_copy_writes_a_row_as_wide_as_the_column_list(self):
        """Regression: a column added to the COPY header and not to
        `write_row`, which raises inside the copy block -- on the bulk path,
        after an unknown number of rows have already streamed."""
        rows = 9
        attrs = distinct_attrs(rows)
        driver = pg_driver()
        driver.load(numpy.zeros((rows, DIM), dtype=numpy.float32), attrs)
        sql, written = driver._cur.copies[0]
        assert sql == pg_mod.COPY_SQL
        assert len(written) == rows
        width = len(copy_columns(sql))
        for i, row in enumerate(written):
            assert len(row) == width
            assert (row[0], row[1], row[2]) == (
                i, int(attrs.tag[i]), int(attrs.ctag[i]))

    def test_both_write_paths_name_the_same_columns_in_the_same_order(self):
        """The property `COPY_SQL` and `INSERT_SQL` exist for, now that C4 has
        split the two write paths across two statement KINDS.

        The bulk-mode load streams a COPY; churn's re-insert sends batched
        INSERTs, because it writes into a table whose HNSW index already
        exists and was otherwise getting Postgres's bulk fast-path while the
        MySQL family re-inserted through `executemany`. One shared string can
        no longer hold the two column lists together, so this does: same
        columns, same order, same table, across two statements. A column
        added to one and not the other fails against a four-column table --
        on the churn path, after the delete has already happened.

        Honest: both lists are `['id', 'tag', 'ctag', 'embedding']`. Mutant
        (`INSERT_SQL`'s two scalars swapped): `['id', 'ctag', 'tag',
        'embedding']` against the COPY's unchanged order, while both
        statements still contain both column names and would both satisfy a
        substring test. Mutant (the re-insert mirroring `ctag` from the tag
        array): the ctag assertion below reads `[0, 1, 2, 3]` where the
        fixture's ctag values are `[900, 901, 902, 903]`.
        """
        rows = 4
        attrs = distinct_attrs(rows)
        vectors = numpy.zeros((rows, DIM), dtype=numpy.float32)
        bulk = pg_driver()
        bulk.load(vectors, attrs)
        churn = pg_driver()
        churn.insert_rows([50, 51, 52, 53], vectors, attrs)
        copy_sql, _streamed = bulk._cur.copies[0]
        insert_sql, written = churn._cur.many[0]
        assert copy_columns(copy_sql) == \
            insert_columns_and_placeholders(insert_sql)[0]
        # The table, which the old form got for free from comparing the two
        # statement strings and a parsed column list does not cover.
        assert copy_sql.startswith("COPY t1 (")
        assert insert_sql.startswith("INSERT INTO t1 (")
        assert [row[0] for row in written] == [50, 51, 52, 53]
        assert [row[2] for row in written] == \
            [int(t) for t in attrs.ctag]

    def test_prepare_is_kept_on_the_filtered_path_for_each_column(self):
        """§3.2: "keep `prepare=True` per column"."""
        for column in COLUMNS:
            driver = pg_driver()
            driver.query_filtered(vector(), K, TAG_T, column=column)
            assert driver._cur.kwargs[-1] == {"binary": True, "prepare": True}

    def test_the_two_columns_do_not_share_a_prepared_statement(self):
        """The live bug F-3 asks about: one cached plan serving two
        predicates.

        psycopg3 keys its prepared statements on the query; whatever the key
        is, two statements that differ cannot collide. So the property to pin
        is that they differ -- and differ in the predicate, not merely
        somewhere.

        Honest: two distinct statements, each naming only its own column.
        Mutant (a builder that ignored `column`): one statement, used for
        both, so the `ctag` measurement runs a `tag` plan.
        """
        driver = pg_driver()
        texts = []
        for column in COLUMNS:
            driver.query_filtered(vector(), K, TAG_T, column=column)
            texts.append(driver._cur.sql[-1])
        assert len(set(texts)) == len(COLUMNS)
        for column, text in zip(COLUMNS, texts):
            assert re.fullmatch(
                rf"SELECT id FROM t1 WHERE {column} < %s "
                rf"ORDER BY embedding <-> %s LIMIT %s", text), text

    def test_the_unfiltered_statement_is_unchanged_by_the_new_column(self):
        driver = pg_driver()
        driver.query(vector(), K)
        sql = driver._cur.sql[-1]
        assert sql == "SELECT id FROM t1 ORDER BY embedding <-> %s LIMIT %s"
        assert re.search(r"\bc?tag\b", sql) is None, sql


class TestMongoStatements:
    def test_the_filter_is_the_whole_stage_value_for_the_chosen_path(self):
        """Regression: the filter built on a fixed path, or merged with the
        other column's.

        Dict equality, not `"ctag" in str(stage)`: membership in a stringified
        pipeline is satisfied by a stage that filters on both columns, which
        would halve the selectivity of every measurement.
        """
        driver = mongo_driver()
        stage = driver._pipeline(vector(), K, CTAG_T, CTAG)[0]["$vectorSearch"]
        assert stage["filter"] == {CTAG: {"$lt": CTAG_T}}
        stage = driver._pipeline(vector(), K, TAG_T, TAG)[0]["$vectorSearch"]
        assert stage["filter"] == {TAG: {"$lt": TAG_T}}

    def test_an_unfiltered_pipeline_has_no_filter_at_all(self):
        stage = mongo_driver()._pipeline(vector(), K)[0]["$vectorSearch"]
        assert "filter" not in stage
        assert set(stage) == {"index", "path", "queryVector", "numCandidates",
                              "limit"}

    def test_documents_carry_both_columns_with_their_own_values(self):
        rows = 6
        attrs = distinct_attrs(rows)
        driver = mongo_driver()
        driver.load(numpy.zeros((rows, DIM), dtype=numpy.float32), attrs)
        docs = driver._coll.documents
        assert len(docs) == rows
        for i, doc in enumerate(docs):
            assert set(doc) == {"_id", TAG, CTAG, "embedding"}
            assert doc[TAG] == int(attrs.tag[i])
            assert doc[CTAG] == int(attrs.ctag[i])

    def test_the_reinsert_writes_the_same_fields_as_the_load(self):
        rows = 3
        attrs = distinct_attrs(rows)
        vectors = numpy.zeros((rows, DIM), dtype=numpy.float32)
        bulk = mongo_driver()
        bulk.load(vectors, attrs)
        churn = mongo_driver()
        churn.insert_rows([70, 71, 72], vectors, attrs)
        assert set(bulk._coll.documents[0]) == set(churn._coll.documents[0])
        assert [d[CTAG] for d in churn._coll.documents] == \
            [int(t) for t in attrs.ctag]

    def test_the_filter_path_is_declared_before_it_is_used(self):
        """F-2: adding a filter path later rebuilds the index, so a path the
        pipeline can ask for must already be in the definition."""
        driver = mongo_driver()
        driver._wait_until_ready = lambda: 0.0
        driver.create_index(IndexSpec(dim=DIM, m=16, metric="euclidean"))
        fields = driver._coll.index_definitions[0]["definition"]["fields"]
        declared = {f["path"] for f in fields if f["type"] == "filter"}
        for column in COLUMNS:
            stage = driver._pipeline(vector(), K, 1, column)[0]["$vectorSearch"]
            assert set(stage["filter"]) <= declared


class TestValkeyCommands:
    def test_the_filter_expression_is_exact(self):
        """Regression: a lost `(`, which turns the exclusive upper bound
        inclusive and shifts this engine's selectivity by one whole bucket or
        one whole cluster against the other five."""
        driver = valkey_driver()
        assert driver._filter_expr(TAG, TAG_T) == f"@tag:[-inf ({TAG_T}]"
        assert driver._filter_expr(CTAG, CTAG_T) == f"@ctag:[-inf ({CTAG_T}]"

    def test_the_search_query_carries_the_filter_and_nothing_else(self):
        driver = valkey_driver()
        driver.query_filtered(vector(), K, CTAG_T, column=CTAG)
        query = driver._conn.searches[-1][2]
        assert re.fullmatch(
            rf"@ctag:\[-inf \({CTAG_T}\]=>\[KNN {K} @embedding \$vec "
            rf"EF_RUNTIME \d+\]", query), query

    def test_the_unfiltered_search_filters_on_nothing(self):
        driver = valkey_driver()
        driver.query(vector(), K)
        query = driver._conn.searches[-1][2]
        assert query.startswith("*=>[KNN ")
        assert re.search(r"\bc?tag\b", query) is None, query

    def test_every_hset_site_in_this_file_writes_both_fields(self):
        """Regression: the field added to the load and the churn write and
        missed in the stall probe.

        The probe is the site nobody runs on purpose, and its HSETs land in
        the same keyspace as the measurement -- so a probe that wrote a hash
        without `ctag` would leave rows the `ctag` index cannot match, and the
        symptom would be a filtered recall shortfall attributed to the engine.

        Checking one site would not see it: the mutation below breaks only the
        third. Honest: all three mappings hold both columns. Mutant (only
        `_probe_write` reverted to `{TAG_FIELD: ...}`): sites 1 and 2 pass,
        site 3 is missing `ctag`.

        **"in this file" is load-bearing, and the name used to say "all
        three".** The spec says "three HSET sites (lines ~356, ~559, ~736)"
        and those three line numbers are right, but the count is not: there
        is a FOURTH at `scripts/verify-valkey-churn.py:92-95`, a raw
        `pipe.hset` with `TAG_FIELD` and `VECTOR_FIELD` only, in a script
        that calls `driver.create_index()` (so its index declares `ctag
        NUMERIC`) and then writes ctag-less hashes into the same `v:`
        keyspace alongside ctag-carrying ones from `driver.insert_rows`. No
        published number moves -- that tool issues no filtered search
        anywhere -- and `scripts/` is not this task's to change, so the
        boundary is named here and pinned by the sibling test below rather
        than quietly assumed. `probe-valkey-filter.py` needs nothing: it
        writes through `driver.load`.
        """
        rows = 4
        attrs = distinct_attrs(rows)
        vectors = numpy.zeros((rows, DIM), dtype=numpy.float32)
        ids = list(range(rows))
        wanted = {TAG, CTAG, valkey_mod.VECTOR_FIELD}

        sites = {}

        bulk = valkey_driver()
        bulk._write_range(vectors, attrs, 0, 0, rows)
        sites["_write_range"] = bulk._conn.hsets

        churn = valkey_driver()
        churn._write_batch_once(ids, vectors, attrs)
        sites["_write_batch_once"] = churn._conn.hsets

        probe = valkey_driver()
        probe._probe_write(ids, vectors, attrs)
        sites["_probe_write"] = probe._conn.hsets

        assert len(sites) == 3
        for label, hsets in sites.items():
            assert len(hsets) == rows, f"{label} wrote {len(hsets)} hashes"
            for i, (_key, mapping) in enumerate(hsets):
                assert set(mapping) == wanted, f"{label} row {i}: {mapping}"
                assert mapping[TAG] == int(attrs.tag[i]), label
                assert mapping[CTAG] == int(attrs.ctag[i]), label

    def test_this_file_has_exactly_the_three_hset_sites_it_claims(self):
        """So a fourth in-file site cannot be added spelling the fields by
        hand, which is how the three drifted apart in the first place.

        Counted by AST over `pipe.hset(...)` calls, and each one's `mapping=`
        must splat `attrs.fields(...)`. A fourth site that listed `tag` and
        `embedding` the way the lab script does would fail here rather than
        in a recall figure.
        """
        import ast
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "harness", "drivers", "valkey.py")
        tree = ast.parse(open(path).read())
        sites = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "hset"):
                continue
            mapping = next((kw.value for kw in node.keywords
                            if kw.arg == "mapping"), None)
            assert isinstance(mapping, ast.Dict), node.lineno
            # `**attrs.fields(i)` appears as a None key in ast.Dict.
            splats = [v for k, v in zip(mapping.keys, mapping.values)
                      if k is None]
            assert len(splats) == 1, f"line {node.lineno}: {len(splats)} splats"
            assert isinstance(splats[0], ast.Call)
            assert splats[0].func.attr == "fields", node.lineno
            sites.append(node.lineno)
        assert len(sites) == 3, f"expected 3 HSET sites, found {sites}"

    def test_the_schema_and_the_hash_fields_are_the_same_names(self):
        """Regression: an index declared over `ctag` and a hash written with
        some other spelling, which valkey-search reports as neither an error
        nor a match."""
        driver = valkey_driver()
        driver._wait_for_backfill = lambda: 0.0
        driver._used_memory = lambda: 0
        driver._index_exists = lambda: False
        driver.create_index(IndexSpec(dim=DIM, m=16, metric="euclidean"))
        declared = set(ft_create_schema(driver._conn.ft_create))

        attrs = distinct_attrs(2)
        driver._write_range(numpy.zeros((2, DIM), dtype=numpy.float32),
                            attrs, 0, 0, 2)
        written = set(driver._conn.hsets[0][1]) - {valkey_mod.VECTOR_FIELD}
        assert declared == written == set(COLUMNS)

    def test_ef_construction_still_finds_the_attribute_count(self):
        """A second NUMERIC field moved everything after it along, and the
        attribute count is patched by `args.index("8")` -- a positional trick
        that a new field could have broken silently, leaving HNSW told it has
        8 arguments when it was given 10."""
        driver = valkey_driver()
        driver._wait_for_backfill = lambda: 0.0
        driver._used_memory = lambda: 0
        driver._index_exists = lambda: False
        driver.create_index(IndexSpec(dim=DIM, m=16, metric="euclidean",
                                      ef_construction=200))
        args = [str(a) for a in driver._conn.ft_create]
        assert args[args.index("HNSW") + 1] == "10"
        assert "8" not in args[:args.index("HNSW") + 2]


# ===========================================================================
# RowAttrs itself
# ===========================================================================

class TestRowAttrs:
    def test_a_bare_array_is_mirrored_and_a_row_attrs_is_kept(self):
        bare = RowAttrs.coerce(numpy.array([4, 5, 6]))
        assert numpy.array_equal(bare.tag, bare.ctag)
        both = distinct_attrs(3)
        assert RowAttrs.coerce(both) is both

    def test_the_mirror_is_a_copy_and_not_one_buffer_under_two_names(self):
        """Regression: `coerce` returning `cls(tag=values, ctag=values)`.

        That was the shipped form for one review cycle and nothing caught it,
        which is why this test exists rather than only the fix. Equal values
        are the point of mirroring; a shared buffer is not, and the two are
        indistinguishable until someone writes through one -- at which point
        `tag` and `ctag` change together, silently, in a repo whose stated
        default is immutability.

        Honest: `coerce([1,2,3]).tag is .ctag` -> False, and a slice of the
        result has two independent bases. Mutant (`ctag=values`): True, and
        `attrs[a:b]` yields two views of one array.
        """
        bare = RowAttrs.coerce(numpy.array([1, 2, 3]))
        assert bare.tag is not bare.ctag
        assert numpy.array_equal(bare.tag, bare.ctag)
        # Independent in fact, not merely distinct objects: writing through
        # one must not be visible through the other.
        bare.ctag[0] = 99
        assert bare.tag[0] == 1
        # And the same after slicing, which is what churn and Valkey do.
        window = RowAttrs.coerce(numpy.arange(10))[2:5]
        assert window.tag is not window.ctag
        window.ctag[0] = 77
        assert window.tag[0] == 2

    def test_slicing_keeps_the_columns_aligned(self):
        """Regression: slicing the two arrays with different bounds, so a row
        carries another row's cluster id.

        Honest: `attrs[2:5].ctag` == ctag[2:5] == [902, 903, 904].
        Mutant (`ctag=self.ctag` unsliced in `__getitem__`): length 10
        against a tag of length 3, and Valkey's HSET loop silently pairs row
        2's vector with row 0's cluster id.
        """
        attrs = distinct_attrs(10)
        window = attrs[2:5]
        assert len(window) == 3
        assert numpy.array_equal(window.tag, attrs.tag[2:5])
        assert numpy.array_equal(window.ctag, attrs.ctag[2:5])
        assert window.ctag.tolist() == [902, 903, 904]

    def test_two_columns_of_different_lengths_are_refused_at_construction(
            self):
        """Regression: a mis-sliced pair failing at whichever row runs out
        first, which is a write that is half done rather than not started.

        Honest: ValueError naming both counts. Mutant (no `__post_init__`):
        `row(4)` on a 3-element ctag raises IndexError from inside the Valkey
        pipeline loop, three rows into the batch.
        """
        with pytest.raises(ValueError) as caught:
            RowAttrs(tag=numpy.arange(10), ctag=numpy.arange(3))
        message = str(caught.value)
        assert "10" in message and "3" in message
        assert TAG in message and CTAG in message

    def test_row_and_fields_agree_with_each_other_and_with_the_order(self):
        attrs = distinct_attrs(5)
        assert attrs.row(3) == (int(attrs.tag[3]), int(attrs.ctag[3]))
        assert attrs.fields(3) == {TAG: int(attrs.tag[3]),
                                   CTAG: int(attrs.ctag[3])}
        # `row` is positional and `fields` is keyed; the invariant that keeps
        # the SQL engines and the document engines writing the same thing is
        # that both follow FILTER_COLUMNS.
        assert tuple(attrs.fields(3)[c] for c in COLUMNS) == attrs.row(3)

    def test_column_goes_through_the_whitelist(self):
        attrs = distinct_attrs(3)
        assert numpy.array_equal(attrs.column(TAG), attrs.tag)
        assert numpy.array_equal(attrs.column(CTAG), attrs.ctag)
        with pytest.raises(ValueError):
            attrs.column("ctag_idx")

    @pytest.mark.parametrize("name", sorted(DRIVER_FACTORIES))
    def test_the_bare_array_callers_still_work(self, name):
        """`tests/test_config.py` drives `ValkeyDriver.insert_rows` with a
        plain list, and the two `scripts/` diagnostics call `load()` with one
        array. None of those files is this task's to edit."""
        rows = 3
        vectors = numpy.zeros((rows, DIM), dtype=numpy.float32)
        driver = DRIVER_FACTORIES[name]()
        driver.load(vectors, numpy.array([1, 2, 3]))
        driver = DRIVER_FACTORIES[name]()
        driver.insert_rows([9, 10, 11], vectors, [1, 2, 3])


# ===========================================================================
# build.py: both columns loaded in one build, from the same source the
# yardstick is computed from
# ===========================================================================

@pytest.fixture
def dataset():
    return tiny_dataset()


@pytest.fixture
def ctx(tmp_path):
    recorder = records_mod.Recorder(str(tmp_path / "ops-t3b.jsonl"))
    try:
        yield RunContext(run_id="t3b", resource_pass="normalized",
                         recorder=recorder, engine="fake", engine_tag="none",
                         march="x86-64-v3", k=K,
                         cache_dir=str(tmp_path / "cache"))
    finally:
        recorder.close()


class TestBuildLoadsBothColumns:
    def test_the_two_columns_the_engine_received_are_not_the_same_array(
            self, ctx, dataset):
        """The assertion the whole of PR 3 rests on.

        If `build.run` mirrored `tag` into `ctag` -- which is exactly what
        `RowAttrs.coerce` does, and what the churn path still does -- then
        both filter modes would be measured on the same predicate and the
        comparison would report that correlation costs nothing.

        Honest: 1,983 of 2,000 rows disagree between the two columns.
        Mutant (`build.run` passing `RowAttrs.coerce(tags)`): 0 of 2,000, and
        every assertion in this file that does not compare the two columns
        still passes.
        """
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        rows = driver._store.rows
        assert len(rows.tags) == len(rows.ctags) == N_ROWS
        disagree = int((rows.tags != rows.ctags).sum())
        assert disagree > N_ROWS // 2, (
            f"only {disagree} of {N_ROWS} rows differ between the two "
            f"columns; they are measuring the same predicate")

    def test_the_loaded_columns_are_the_ones_ground_truth_will_use(
            self, ctx, dataset):
        """Not `x == x`: the right-hand side is the yardstick's own source.

        `datasets.filter_values` is what the filtered workload calls to build
        the mask it computes exact ground truth over. Comparing the engine's
        column against that call, with the arguments a run really makes, is
        comparing the engine against the yardstick -- which is the
        disagreement that would score every engine against the wrong rows.
        """
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        rows = driver._store.rows
        for column, loaded in ((TAG, rows.tags), (CTAG, rows.ctags)):
            expected = ds.filter_values(ctx.cache_dir, dataset, column,
                                        indexed_rows=N_ROWS)
            assert numpy.array_equal(loaded, expected), column

    def test_a_subset_build_clusters_the_subset_it_loaded(self, ctx, dataset):
        """Regression: `indexed_rows` taken from the corpus rather than from
        what was loaded.

        The pivots are drawn from the rows given, so clustering 2,000 rows and
        clustering the first 500 produce different labels for those 500 --
        measured, 497 of 500 differ. A build that passed the full row count
        would load labels the ground truth for a 500-row index never sees.

        Honest: the 500 loaded labels equal `filter_values(..., 500)`.
        Mutant (`indexed_rows=len(dataset.train)`): they equal
        `filter_values(..., 2000)[:500]` instead, and 497 of 500 are wrong.
        """
        subset = 500
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset),
                           subset_rows=subset)

        loaded = driver._store.rows.ctags
        assert len(loaded) == subset
        right = ds.filter_values(ctx.cache_dir, dataset, CTAG,
                                 indexed_rows=subset)
        wrong = ds.filter_values(ctx.cache_dir, dataset, CTAG,
                                 indexed_rows=N_ROWS)[:subset]
        assert numpy.array_equal(loaded, right)
        # The premise: the two candidates really do differ, or this test
        # cannot tell them apart.
        assert not numpy.array_equal(right, wrong)

    def test_the_cluster_assignment_is_charged_to_neither_timer(
            self, ctx, dataset, monkeypatch):
        """Regression: the ctag computation moved inside the measured window.

        It is cached and small, but "small" is not "outside the measurement".
        A slow `filter_values` must not move `ingest_wall_s`, which build.py
        takes from the driver's own `LoadResult`.

        Honest: with a 50 ms delay injected into `filter_values`,
        `ingest_wall_s` stays at the driver's figure. Mutant (the call moved
        below `driver.load`, or wrapped around it): `ingest_wall_s` absorbs
        the 50 ms.
        """
        real = ds.filter_values
        import time as _time

        def slow(*args, **kw):
            _time.sleep(0.05)
            return real(*args, **kw)

        monkeypatch.setattr(build_workload, "filter_values", slow)
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        ingest = list(records_mod.read_records(ctx.recorder.path))[0]
        assert ingest["phase"] == records_mod.PHASE_INGEST
        assert ingest["ingest_wall_s"] < 0.05, ingest["ingest_wall_s"]


# ===========================================================================
# churn.py: the one call this task owns in that file
# ===========================================================================

class TestChurnPassesOneStructure:
    def test_the_reinsert_hands_the_driver_a_row_attrs(self):
        """§3.2's opening clause: one structure, so load and insert_rows stay
        in sync. The re-insert is the second of the two.

        Honest: the driver is handed a `RowAttrs`. Mutant (the bare array left
        in place): a numpy array, which the driver would coerce anyway -- so
        this does not change the numbers, and that is exactly why it is worth
        pinning: the mirror has to be visible at the call site rather than
        happening inside five drivers.
        """
        from harness.workloads import churn as churn_mod

        seen = []

        class Recorder(FakeDriver):
            def insert_rows(self, ids, vectors, attrs):
                seen.append(attrs)
                return super().insert_rows(ids, vectors, attrs)

        driver = Recorder(dim=DIM)
        driver.connect()
        written, _seconds = churn_mod._reinsert(
            driver, list(range(10)), numpy.zeros((10, DIM), numpy.float32),
            numpy.arange(10), budget_s=60.0)
        assert written == 10
        assert seen, "nothing was re-inserted"
        for attrs in seen:
            assert isinstance(attrs, RowAttrs)

    def test_the_mirror_is_documented_where_it_happens(self):
        """A claim in a comment is part of the deliverable, and this one is
        load-bearing: the re-inserted rows carry `ctag = tag`, which is wrong
        for any `ctag` measurement taken after a churn. `main.py` runs
        filtered before churn, so nothing does that today -- and the next
        person to reorder them needs to find this written down."""
        source = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "harness", "workloads", "churn.py")).read()
        assert "RowAttrs.coerce" in source
        assert "ctag = tag" in source

    def test_main_still_runs_filtered_before_churn(self):
        """The ordering the compromise above depends on. If this stops being
        true, the churn mirror becomes a live wrong number rather than a
        latent one."""
        source = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "harness", "main.py")).read()
        assert (source.index("filtered_workload.run(")
                < source.index("churn_workload.run("))


# ===========================================================================
# §3.2 item 3 -- the measurement version
# ===========================================================================

class TestTheMeasurementVersionItem3:
    def test_the_version_moved_past_the_calibration_bump(self):
        """Regression: the column added and the version left at 4, so an
        ingest figure from a three-column table and one from a four-column
        table load onto the same chart.

        `>= 5` rather than `== 5`: pinning the literal makes the next bump
        fail for no reason, which is how a version constant acquires a test
        that has to be edited to be satisfied.
        """
        assert records_mod.OPS_MEASUREMENT_VERSION >= 5, (
            "PR 2 took this to 4; PR 3 widens every row by a column, which "
            "moves ingest throughput, table size and -- for Valkey and "
            "Percona Search -- index size")

    def test_the_new_version_is_not_claimed_to_match_an_older_one(self):
        """F-4: the equivalence set is derived from two named constants and
        must still be exactly {1, 2}. A bump that changed what a number means
        cannot enter it, and this one did."""
        assert (records_mod.OPS_MEASUREMENT_VERSION
                not in records_mod.OPS_MEASUREMENT_EQUIVALENT_VERSIONS)
        assert set(records_mod.OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == {1, 2}
        assert not records_mod.ops_versions_measure_the_same_thing(
            [4, records_mod.OPS_MEASUREMENT_VERSION])
        assert not records_mod.ops_versions_measure_the_same_thing(
            [3, records_mod.OPS_MEASUREMENT_VERSION])
        # And the two that are equivalent still are, so the bump did not
        # disturb the only pair this scheme says is comparable.
        assert records_mod.ops_versions_measure_the_same_thing([1, 2])

    def test_a_three_column_run_and_a_four_column_one_cannot_share_a_chart(
            self, tmp_path):
        """The guard end to end, through the report's own loader.

        Honest: `load_ops_records` refuses, naming version 4 and version 5.
        Mutant (`OPS_MEASUREMENT_VERSION` left at 4): both records carry 4,
        the loader returns two records, and a rows-per-second bar chart gets
        one bar per engine built from two different table widths.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records

        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps({
            "run_id": "r", "engines": {}, "phases": [], "warnings": [],
            "config": {"resource_pass": "normalized",
                       "ops_measurement_version": 4}}))
        path = str(run / "ops-mariadb-glove-100-angular-normalized-m16.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps({
                "run_id": "r", "engine": "mariadb",
                "phase": records_mod.PHASE_INGEST,
                "dataset": "glove-100-angular", "m": 16,
                "rows": 1_183_514, "ingest_rows_per_s": 151.0,
                "resource_pass": "normalized",
                "timestamp": "2026-01-01T00:00:00Z",
                "extra": {"ops_measurement_version": 4}}) + "\n")
        with records_mod.Recorder(path) as recorder:
            recorder.write(records_mod.Record(
                run_id="r", engine="mariadb",
                phase=records_mod.PHASE_INGEST,
                dataset="glove-100-angular",
                timestamp="2026-01-01T01:00:00Z", m=16,
                rows=1_183_514, ingest_rows_per_s=148.0,
                resource_pass="normalized"))

        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(str(run))
        message = str(caught.value)
        assert "version 4" in message
        assert f"version {records_mod.OPS_MEASUREMENT_VERSION}" in message
        assert "two different measurements" in message


# ===========================================================================
# The fake has to keep implementing the interface it stands in for
# ===========================================================================

class TestTheFakeTracksTheRealInterface:
    def test_a_predicate_reads_its_own_column(self):
        """Regression, and the one that made this fake useful at all: it held
        one tag array under every column name, so a test that asked for `ctag`
        and got `tag` could not tell.

        Honest: the two columns select disjoint row sets, so the result sets
        differ. Mutant (`_column_values` returning `rows.tags` always): the
        two calls return the same ids and every `column=` test in the repo
        becomes a no-op.
        """
        dataset = tiny_dataset()
        # Mirror images under the same threshold: `tag < 2` selects exactly
        # the first half of the corpus and `ctag < 2` exactly the second, so
        # the two predicates have disjoint answers by construction rather than
        # by hope. 50 is the "does not qualify" value in both.
        tag = numpy.full(N_ROWS, 50, dtype=numpy.int64)
        ctag = numpy.full(N_ROWS, 50, dtype=numpy.int64)
        tag[:N_ROWS // 2] = 1
        ctag[N_ROWS // 2:] = 1
        assert int((tag < 2).sum()) == int((ctag < 2).sum()) == N_ROWS // 2
        assert not ((tag < 2) & (ctag < 2)).any()
        driver = loaded_driver(dataset, attrs=RowAttrs(tag=tag, ctag=ctag),
                              ef_half=0.0, ef_search=N_ROWS)

        by_tag = driver.query_filtered(dataset.test[0], K, 2, column=TAG)
        by_ctag = driver.query_filtered(dataset.test[0], K, 2, column=CTAG)
        assert by_tag and by_ctag
        assert set(by_tag).isdisjoint(set(by_ctag))
        for row in by_tag:
            assert tag[row] < 2 and ctag[row] >= 2
        for row in by_ctag:
            assert ctag[row] < 2 and tag[row] >= 2

    def test_the_fake_s_whitelist_is_the_dataset_module_s(self):
        """Regression: the fake accepting a column the drivers refuse, or
        refusing one they accept -- either way a workload test would prove
        something about a column the engines do not have."""
        driver = FakeDriver()
        assert driver._tag_columns == COLUMNS
        assert inspect.signature(FakeDriver.__init__
                                 ).parameters["tag_columns"].default is COLUMNS

    def test_an_unknown_column_is_refused_before_anything_is_measured(self):
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=0.0)
        for bad in ("category", "ctag_idx", "Tag", ""):
            with pytest.raises(ValueError):
                driver.query_filtered(dataset.test[0], K, 10, column=bad)
            with pytest.raises(ValueError):
                driver.explain_uses_vector_index(dataset.test[0], K,
                                                 tag_threshold=10, column=bad)
        # And both legal columns are accepted on both paths.
        for column in COLUMNS:
            driver.query_filtered(dataset.test[0], K, 10, column=column)
            assert driver.explain_uses_vector_index(
                dataset.test[0], K, tag_threshold=10, column=column) is True

    def test_delete_and_maintain_keep_the_columns_in_step(self):
        """Regression: `maintain()` reclaiming rows from three arrays and not
        the fourth, which would pair surviving vectors with purged cluster
        ids."""
        dataset = tiny_dataset()
        attrs = RowAttrs(
            tag=numpy.arange(N_ROWS, dtype=numpy.int64) % 10,
            ctag=900 + numpy.arange(N_ROWS, dtype=numpy.int64) % 10)
        driver = loaded_driver(dataset, attrs=attrs, ef_half=0.0)
        driver.delete_ids(list(range(100)))
        assert driver.maintain() == 100
        rows = driver._store.rows
        assert len(rows.ids) == len(rows.tags) == len(rows.ctags) == \
            N_ROWS - 100
        assert numpy.array_equal(rows.tags, attrs.tag[100:])
        assert numpy.array_equal(rows.ctags, attrs.ctag[100:])

    def test_a_clone_sees_the_same_columns(self):
        """The concurrency workload clones; a clone reading a different store
        would measure a different table."""
        dataset = tiny_dataset()
        attrs = distinct_attrs(N_ROWS)
        driver = loaded_driver(dataset, attrs=attrs, ef_half=0.0)
        sibling = driver.clone()
        assert sibling._tag_columns == driver._tag_columns
        assert numpy.array_equal(sibling._store.rows.ctags, attrs.ctag)


# ===========================================================================
# Rule 2, in one place: the old form has to pass the mutant the new form
# catches. A guard that the previous code would also have survived is not a
# guard, it is a restatement.
# ===========================================================================

class TestTheOldFormsDoNotCatchThese:
    def test_the_old_select_interpolated_whatever_it_was_handed(self):
        """Pre-T3b `_select` built `WHERE tag < %s` with no column at all; the
        shape the whitelist replaces is the obvious next edit, which is to
        format the argument straight in.

        Honest (`checked_filter_column`): ValueError, nothing built.
        Mutant (the straight f-string): 'SELECT id FROM t1 WHERE tag; DROP
        TABLE t1 -- < %s ORDER BY ...'.
        """
        def old_select(k, filtered, column=TAG):
            where = f"WHERE {column} < %s " if filtered else ""
            return (f"SELECT id FROM t1 {where}"
                    f"ORDER BY vec_distance_euclidean(v, %s) LIMIT {int(k)}")

        hostile = "tag; DROP TABLE t1 --"
        mutant = old_select(K, True, hostile)
        assert hostile in mutant
        assert "DROP TABLE" in mutant

        driver = mysql_driver()
        with pytest.raises(ValueError):
            driver._select(K, True, column=hostile)

    def test_the_old_explain_built_its_own_predicate(self):
        """So a whitelist added to `query_filtered` alone would have left the
        plan check interpolating freely.

        Honest: both paths refuse. Mutant (the pre-T3b explain body): the
        statement is built and sent, carrying the hostile column, and
        `query_filtered` next to it still refuses -- which is how a one-sided
        repair looks from the outside.
        """
        class PreT3bExplain(pg_mod.PostgresDriver):
            def explain_uses_vector_index(self, vector, k,
                                          tag_threshold=None, column=TAG):
                sql = (f"SELECT id FROM t1 WHERE {column} < %s "
                       f"ORDER BY embedding <-> %s LIMIT %s")
                self._cur.execute("EXPLAIN (FORMAT TEXT) " + sql,
                                  (int(tag_threshold), vector, k))
                return True

        mutant = pg_driver(cls=PreT3bExplain)
        assert mutant.explain_uses_vector_index(
            vector(), K, tag_threshold=1, column="ctag_idx") is True
        assert "ctag_idx" in mutant._cur.sql[-1]
        # The same call on the real driver.
        real = pg_driver()
        with pytest.raises(ValueError):
            real.explain_uses_vector_index(vector(), K, tag_threshold=1,
                                           column="ctag_idx")
        assert real._cur.statements == []

    def test_a_substring_check_passes_a_schema_that_drops_an_index(self):
        """The weaker form of the parity test, shown surviving the mutant.

        `"ctag_idx" in ddl` is the assertion a reader would reach for, and it
        is satisfied by a DDL that declares the index and not by the one that
        matters -- but worse, `"ctag" in ddl` is satisfied by the column
        declaration alone, with no index at all. Both are demonstrated here
        against a DDL the parity test refuses.
        """
        mutant_ddl = (
            "CREATE TABLE t1 (\n"
            "  id INT PRIMARY KEY,\n"
            "  tag INT NOT NULL,\n"
            "  ctag INT NOT NULL,\n"
            "  v VECTOR(16) NOT NULL,\n"
            "  KEY tag_idx (tag),\n"
            "  VECTOR INDEX vi (v) M=16 DISTANCE=euclidean\n"
            ") ENGINE=InnoDB")
        assert "ctag" in mutant_ddl                 # passes the mutant
        assert "ctag" in mutant_ddl.replace("ctag_idx", "")
        kinds = mysql_index_kinds(mutant_ddl)
        assert kinds[TAG] != kinds.get(CTAG, set())  # the parity test refuses

    def test_a_mirrored_build_passes_every_test_that_ignores_the_two_columns(
            self, ctx, dataset):
        """Why `test_the_two_columns_the_engine_received_are_not_the_same_array`
        exists. A build that mirrored `tag` into `ctag` produces a schema, an
        INSERT and a predicate that are all correct, and only a comparison of
        the two columns' values can see that the measurement is vacuous.

        Honest: 1,983 of 2,000 rows differ. Mutant: 0.
        """
        real = build_workload.filter_values

        def tag_for_everything(cache_dir, dataset_, column, **kw):
            return real(cache_dir, dataset_, TAG, **kw)

        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.filter_values = tag_for_everything
        try:
            build_workload.run(ctx, driver, dataset, index_spec(dataset))
        finally:
            build_workload.filter_values = real

        rows = driver._store.rows
        # The mutant: a perfectly well-formed, entirely uninformative load.
        assert numpy.array_equal(rows.tags, rows.ctags)
        assert int((rows.tags != rows.ctags).sum()) == 0
        assert len(rows.ctags) == N_ROWS
