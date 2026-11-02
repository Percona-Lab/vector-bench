"""Ops-harness driver for MyVector, the hnswlib vector-search PLUGIN for MySQL.

Pinned to askdba/myvector v1.26.9 (commit 384d9df7) on MySQL 9.7.2, built
in-tree (docker/myvector/Dockerfile). Every behaviour this module depends on is
cited against that tree by symbol, because several of them contradict the
release's own documentation and the documentation is not what runs.

WHY IT IS A DRIVER OF ITS OWN AND NOT A `MySQLDialect`
------------------------------------------------------
The wire protocol, the table, the bulk load, the deletes and the sizing of the
base table are the MySQL family's, and are inherited. What is not:

* **The index is not a server index.** It is an hnswlib graph the plugin holds
  in memory and writes to files in `myvector_index_dir`. It is DECLARED by a
  column comment (`MYVECTOR Column |type=HNSW,...`) and BUILT by a stored
  procedure, `mysql.myvector_index_build('db.table.col', 'pk')`, which opens a
  connection of its own back to the server (`BuildMyVectorIndexSQL`, reading
  `myvector.cnf`) and REPORTS FAILURE AS A RESULT ROW, NOT AS AN SQL ERROR. A
  driver that only executed the CALL would time a build that did not happen.
* **The search is a query rewrite.** An audit plugin rewrites
  `WHERE MYVECTOR_IS_ANN('db.t.col','pk', vec, opts)` into
  `pk IN (SELECT ... JSON_TABLE(myvector_ann_set(...)))` before the parser sees
  the statement (`rewriteMyVectorIsANN`). `ef_search` is not a session variable
  but an option inside that call, and it mutates the SHARED index
  (`HNSWMemoryIndex::setSearchEffort`), so it is passed on every query.
* **Options are matched exactly and unknown spellings fall back silently**
  (`HNSWMemoryIndex::HNSWMemoryIndex`: `getIntOption("M", 16)`,
  `getOption("dist") == "Cosine"`). `m=32` builds M=16; `dist=cosine` builds an
  L2 index. So the driver reads the effective settings back with
  `MYVECTOR_INDEX_STATUS` after every build and refuses a mismatch, rather than
  trusting its own DDL.
* **Writes after the build reach the index only when the driver says so** --
  see `wait_index_synced` for why that is `myvector_index_refresh` and not the
  binlog listener the release documents.
* **There is no filtered search** -- see `FILTERED_DECLINED`.
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy

from .base import (DATABASE, FILTER_TAG, TABLE, ConnectionSpec, IndexSpec,
                   RowAttrs, checked_filter_column)
from .mysql_family import MySQLDialect, MySQLFamilyDriver

#: `db.table.column`, the name every MyVector procedure and UDF takes.
VECTOR_COLUMN = f"{DATABASE}.{TABLE}.v"
#: The primary-key column the index maps graph labels to.
ID_COLUMN = "id"
#: The row-change timestamp `myvector_index_refresh` selects on. Its definition
#: is the one the plugin's own DDL rewrite appends for `track=`
#: (`rewriteMyVectorColumnDef`), so the table is what `MYVECTOR(...,track=...)`
#: would have produced.
TRACK_COLUMN = "updatets"
TRACK_COLUMN_DDL = (f"{TRACK_COLUMN} TIMESTAMP NOT NULL DEFAULT "
                    f"CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")

#: Dataset metric -> the ONLY spelling the plugin honours. Case matters:
#: `dist=cosine` silently builds L2 (docs/LIMITATIONS.md, #119).
#:
#: `Cosine` and not `CosineNorm`: `CosineNorm` maps to hnswlib's
#: InnerProductSpace, which is cosine only for unit vectors, and the corpora
#: here are not normalised. `Cosine` is `AngularDistanceSpace`, whose distance
#: function `computeCosineDistance` is a plain scalar loop -- no SIMD, unlike
#: the L2 path's `hnswlib::L2Space`. That is the vendor's implementation, so it
#: is what is measured, and docs/04-engine-notes.md says so.
METRIC_NAMES: Dict[str, str] = {"angular": "Cosine", "euclidean": "L2"}

#: hnswlib's `max_elements` is fixed at construction and `addPoint` THROWS past
#: it ("The number of elements exceeds the specified limit") -- inside mysqld,
#: from a UDF, where an uncaught exception terminates the server. Deleted rows
#: are never removed from the graph at this release (see `maintain`), so every
#: churn re-insert consumes a slot for good. The capacity is therefore the
#: loaded row count times this factor, which covers a cumulative churn of half
#: the corpus; the shipped profiles churn at most 25%. `insert_rows` refuses
#: BEFORE sending a row that would overflow it, so the failure is a Python
#: error with a reason rather than a dead server.
CAPACITY_FACTOR = 1.5
CAPACITY_MARGIN = 1000

#: Default ef_construction when a profile names none: the plugin's own default
#: for `ef=` (`HNSWMemoryIndex::HNSWMemoryIndex`, `getIntOption("ef", 100)`).
#: The orchestrator passes 200 (`ops_pass.harness_args`), the value every
#: other engine with the knob is built at; this is what a by-hand run gets.
DEFAULT_EF_CONSTRUCTION = 100

FILTERED_DECLINED = (
    "MyVector v1.26.9 has no filtered vector search. myvector_ann_set takes "
    "(column, id column, query vector[, options]) and its options are nn and "
    "ef_search only; MYVECTOR_IS_ANN is rewritten to `id IN (<k nearest>)`, so "
    "a `WHERE tag < t` beside it filters the k candidates AFTER the search and "
    "returns fewer than k rows (docs/LIMITATIONS.md, 'No filtered search'). "
    "Measuring that would publish post-filtered recall as filtered search.")

INCREMENTAL_DECLINED = (
    "MyVector has no working incremental build on this server: its online "
    "maintenance is a binlog listener (`myvector_binlog_loop`) that reads with "
    "a 1 s MYSQL_OPT_READ_TIMEOUT and treats any fetch error whose text lacks "
    "'timed out' as fatal; against MySQL 9.7.2 the idle-timeout error text is "
    "empty, so the listener exits after its first idle second and never "
    "restarts (observed in this image: 'Binlog fetch failed: ' then 'Exiting "
    "binlog func'). The only build is the post-load myvector_index_build.")

#: What `index_sync_s` means for this engine, stamped on every record.
#:
#: A refresh is not an incremental sync. `myvector_open_index_impl` (action
#: `refresh`) runs `BuildMyVectorIndexSQL`, which ends in
#: `vi->saveIndex(myvector_index_dir, "build")` -- a FULL rewrite of the graph
#: file -- and then calls `saveIndex(dir, "refresh")` itself, an incremental
#: checkpoint on top. So the figure is (select + insert the changed rows) +
#: (write the whole graph) + (checkpoint), and grows with the corpus. The
#: procedure is one call and its parts cannot be timed separately from SQL.
INDEX_SYNC_NOTE = (
    "is one myvector_index_refresh: the changed rows are inserted and then "
    "the WHOLE graph file is rewritten (BuildMyVectorIndexSQL's "
    "saveIndex(..., \"build\")) plus a checkpoint, so it grows with the "
    "corpus, not with the re-inserted rows; the parts cannot be timed apart "
    "from SQL. Not comparable to another engine's wait for an asynchronous "
    "index to catch up.")

#: Where this engine's graph lives, stamped on every record so the report can
#: say it beside the memory figures. See docs/04-engine-notes.md, "Memory".
MEMORY_NOTE = (
    "keeps its whole HNSW graph in mysqld's own heap (hnswlib, no paging), "
    "OUTSIDE innodb_buffer_pool_size and not bounded by any knob; it counts "
    "against the same server-container memory limit as every engine's "
    "memory. Its build also holds the entire (id, vector) table client-side "
    "inside mysqld while inserting (BuildMyVectorIndexSQL: mysql_store_result), "
    "so build peak is roughly buffer pool + graph + one more copy of the "
    "vectors.")

#: `Key : value` and `Key = value` lines of `HNSWMemoryIndex::getStatus`.
_STATUS_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z .]*?)\s*[:=]\s*(.*?)\s*$")
#: `SUCCESS: Index created & saved at (binlog.000002 157), rows : 1000.`
_BUILD_ROWS = re.compile(r"rows\s*:\s*(\d+)")


def hex_literal(vector) -> str:
    """The query vector as an SQL hex literal, `X'...'`.

    Binary whatever the connection character set, and `[0-9a-f]` only. The
    second property mattered while the vector sat inside `MYVECTOR_IS_ANN(...)`
    itself -- the rewrite finds the end of the call by COUNTING PARENTHESES
    over the raw text, quotes included (`rewriteMyVectorIsANN`: the `ob`
    counter), so an escaped binary string holding 0x28 or 0x29 would derail
    it. It now goes in a `SET` (`MyVectorDriver._bind_query_vector` says why)
    and the property is kept anyway: a literal that cannot contain a
    parenthesis cannot reintroduce that failure if someone inlines it again.
    """
    return "X'" + numpy.asarray(vector, dtype="<f4").tobytes().hex() + "'"


def parse_status(text: str) -> Dict[str, str]:
    """`MYVECTOR_INDEX_STATUS`'s text, as `{key: value}`.

    Keys are what `HNSWMemoryIndex::getStatus` prints: `Vector Index`,
    `Type`, `Dimension`, `Distance`, `Max. Capacity`, `M`, `Element Data
    Size`, `Current Rows`, `Searches`.
    """
    out: Dict[str, str] = {}
    for line in str(text).splitlines():
        match = _STATUS_LINE.match(line)
        if match:
            out[match.group(1).strip()] = match.group(2).strip()
    return out


def graph_bytes_floor(status: Dict[str, str]) -> Optional[int]:
    """Level-0 graph bytes the plugin holds: `Element Data Size` x rows.

    hnswlib's `size_data_per_element_` is the vector plus the level-0 link
    list plus the label (3,276 B at 784 dims and M=16, measured), so this
    product is the bulk of the graph. It is a FLOOR: upper-level link lists,
    the per-element lock and level arrays and the label map are not in it.
    """
    try:
        return int(status["Element Data Size"]) * int(status["Current Rows"])
    except (KeyError, TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


MYVECTOR_DIALECT = MySQLDialect(
    name="myvector",
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    global_setup=(),
    # No server variable: ef_search travels inside each MYVECTOR_IS_ANN call.
    set_ef_search="",
    # The derived table the rewrite introduces. Present in a plan only when the
    # annotation was rewritten into a search; see `explain_uses_vector_index`.
    index_name="myvector_ann",
    index_file_globs=(),
    metric_names=METRIC_NAMES,
    # INT, the MariaDB/AliSQL key. The plugin's labels are `KeyTypeInteger`
    # and the build reads the key back with `atol`, so INT and BIGINT both
    # work for the build and refresh used here.
    pk_type="INT",
    ef_construction=DEFAULT_EF_CONSTRUCTION,
)


class MyVectorDriver(MySQLFamilyDriver):
    """MySQL 9.7 + the MyVector plugin. See the module docstring."""

    dialect = MYVECTOR_DIALECT
    name = "myvector"

    #: The repair is a full rebuild by the vendor's own procedure. Declared
    #: rather than inferred, as `EngineDriver.repair_path` requires.
    repair_path = "index-rebuild"

    declined_workloads: Tuple[Tuple[str, str], ...] = (
        ("filtered", FILTERED_DECLINED),)

    #: Bulk build only: `IndexSpec.build_mode` "post" builds the graph after
    #: the load; there is no write-path maintenance to measure.
    incremental_index = False

    def __init__(self, spec: ConnectionSpec,
                 dialect: Optional[MySQLDialect] = None):
        super().__init__(spec, dialect)
        self._ef_search: Optional[int] = None
        #: hnswlib's element count as the driver expects it to be: rows built,
        #: plus every row re-inserted since. Deletes do not lower it.
        self._expected_elements = 0
        #: Rows written since the last build or refresh, which the next
        #: refresh must report having indexed.
        self._pending = 0
        self._capacity = 0
        self._status: Dict[str, str] = {}
        self._build_result = ""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _column_options(self, index: IndexSpec, capacity: int) -> str:
        """The plugin's option list, in the exact spellings it matches.

        `M`, never `m`; `ef` is ef_construction; `dist` from `METRIC_NAMES`;
        `track=` names the timestamp `myvector_index_refresh` selects on. No
        `online=Y`: see `INCREMENTAL_DECLINED`. No `threads=`: build
        parallelism is the server variable `myvector_index_bg_threads`, which
        the engine config sets per resource pass so the value is in the
        manifest's server flags.
        """
        ef = (index.ef_construction if index.ef_construction is not None
              else self.dialect.ef_construction)
        return (f"type=HNSW,dim={int(index.dim)},size={int(capacity)},"
                f"M={int(index.m)},ef={int(ef)},"
                f"dist={self.dialect.metric_names[index.metric]},"
                f"track={TRACK_COLUMN}")

    def _vector_column(self, index: IndexSpec, capacity: int = 0,
                       with_track: bool = True) -> str:
        """`VECTOR(n)` with the plugin's comment, then the tracking column.

        The comment starts `MYVECTOR Column |` because the build procedure
        refuses anything else (`MYVECTOR_INDEX_INTERNAL`: `LOCATE("MYVECTOR
        COLUMN", colinfo) <> 1`). The `MYVECTOR(...)` column TYPE would be
        rewritten into exactly this by the plugin's DDL rewrite
        (`rewriteMyVectorColumnDef`), tracking column included, but that
        rewrite has a 128-character option limit and is matched literally;
        writing the result directly avoids both. The tracking column costs
        every row 4 bytes the other MySQL-family tables do not carry.
        """
        column = (f"v VECTOR({int(index.dim)}) NOT NULL COMMENT "
                  f"'MYVECTOR Column |{self._column_options(index, capacity)}'")
        return f"{column},\n  {TRACK_COLUMN_DDL}" if with_track else column

    def drop_schema(self) -> None:
        """Drop the plugin's index object before the database under it.

        `DROP DATABASE` removes the table and leaves the plugin's in-memory
        index alive under the same name. The next build then finds it
        (`myvector_open_index_impl`: `g_indexes.get(vecid)`) and rebuilds the
        EXISTING object, whose dimension, M and ef were fixed when it was
        constructed -- so a second build at another M would silently build at
        the first one's. `myvector_index_drop` closes and deletes it.
        """
        try:
            if self._table_present():
                self._call(f"CALL mysql.myvector_index_drop('{VECTOR_COLUMN}')")
        except Exception as exc:
            print(f"[{self.name}] myvector_index_drop before DROP DATABASE "
                  f"failed (continuing): {exc}")
        super().drop_schema()

    def _table_present(self) -> bool:
        self._cur.execute(
            "SELECT COUNT(*) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (DATABASE, TABLE))
        row = self._cur.fetchone()
        return bool(row and int(row[0]))

    def create_schema(self, index: IndexSpec) -> None:
        if index.build_mode == "incremental":
            raise ValueError(INCREMENTAL_DECLINED)
        self._index = index
        ddl = self._create_table_ddl(index, with_index=False)
        print(f"[{self.name}] {ddl}")
        self._cur.execute(ddl)

    def create_index(self, index: IndexSpec) -> None:
        """The bulk build, after the load: one `myvector_index_build`."""
        self._index = index
        self._build(self.count_rows())

    def capabilities(self) -> Dict[str, Any]:
        """A bulk build, plus what the server says it built.

        `myvector_status` is the effective configuration read back after the
        last build, so the build record says what was built rather than what
        was asked for.
        """
        caps: Dict[str, Any] = {"incremental_index": False,
                                "ef_construction_tunable": True,
                                **self.index_format()}
        if self._status:
            caps["myvector_status"] = dict(self._status)
            floor = graph_bytes_floor(self._status)
            if floor is not None:
                caps["myvector_graph_bytes_floor"] = floor
        if self._build_result:
            caps["myvector_build_result"] = self._build_result
        return caps

    # ------------------------------------------------------------------
    # The build and its verification
    # ------------------------------------------------------------------

    def _call(self, sql: str) -> List[Tuple[Any, ...]]:
        """Execute a CALL and return the rows of every result set.

        A procedure answers with one or more result sets and then a status;
        all of them have to be consumed before the connection will take
        another statement, and the one that carries the build's outcome is a
        result row.
        """
        self._cur.execute(sql)
        rows: List[Tuple[Any, ...]] = []
        while True:
            try:
                rows.extend(tuple(r) for r in (self._cur.fetchall() or []))
            except Exception:
                pass
            try:
                more = self._cur.nextset()
            except Exception:
                more = None
            if not more:
                break
        return rows

    def _set_capacity(self, capacity: int) -> None:
        """Rewrite the column comment with hnswlib's `size`.

        Here rather than in CREATE TABLE because the row count is not known
        when the schema is created. A comment-only change, so INSTANT: no
        row is touched and nothing is rebuilt.
        """
        assert self._index is not None, "create_schema must run first"
        sql = (f"ALTER TABLE {TABLE} MODIFY COLUMN "
               f"{self._vector_column(self._index, capacity, with_track=False)}"
               f", ALGORITHM=INSTANT")
        self._cur.execute(sql)
        self._capacity = int(capacity)

    def _build(self, rows: int) -> None:
        """`myvector_index_build`, then proof that it built what was asked."""
        capacity = max(self._capacity,
                       int(math.ceil(rows * CAPACITY_FACTOR)) + CAPACITY_MARGIN)
        if capacity != self._capacity:
            self._set_capacity(capacity)
        sql = f"CALL mysql.myvector_index_build('{VECTOR_COLUMN}', '{ID_COLUMN}')"
        print(f"[{self.name}] {sql}")
        result = " | ".join(_as_text(c) for row in self._call(sql) for c in row)
        self._build_result = result
        print(f"[{self.name}] build returned: {result}")
        self.check_build_result(result, rows)
        self._expected_elements = rows
        self._pending = 0
        self.verify_status(expected_rows=rows)
        self._next_second()

    @staticmethod
    def _next_second() -> None:
        """Wait out the second the build or refresh recorded as its watermark.

        `myvector_index_refresh` selects `unix_timestamp(track) > previous AND
        <= now` (`BuildMyVectorIndexSQL`), at one-second resolution, and the
        previous watermark is the second the last build or refresh STARTED. A
        row written later in that same second carries that same timestamp and
        no refresh will ever select it. Waiting for the clock to leave the
        second closes that hole; it is part of the maintenance and is timed
        with it, and it costs at most one second.
        """
        now = time.time()
        time.sleep(max(0.0, math.floor(now) + 1.0 - now) + 0.01)

    @staticmethod
    def check_build_result(result: str, rows: int) -> None:
        """Refuse a build or refresh whose text is not a success for `rows`.

        The procedure signals nothing on failure: a missing or unreadable
        `myvector.cnf`, a refused connection, a failed save -- each comes back
        as a row of text (`BuildMyVectorIndexSQL`, `myvector_open_index_impl`)
        and the CALL succeeds.
        """
        if not result.startswith("SUCCESS"):
            raise RuntimeError(
                f"myvector index build/refresh did not succeed: {result!r}")
        match = _BUILD_ROWS.search(result)
        if match and int(match.group(1)) != int(rows):
            raise RuntimeError(
                f"myvector index build/refresh indexed {match.group(1)} rows "
                f"of {rows}: {result!r}")

    def index_status(self) -> Dict[str, str]:
        rows = self._call(f"CALL mysql.myvector_index_status('{VECTOR_COLUMN}')")
        text = "\n".join(_as_text(c) for row in rows for c in row)
        return parse_status(text)

    def verify_status(self, expected_rows: Optional[int] = None
                      ) -> Dict[str, str]:
        """Read the effective settings back and refuse any mismatch.

        What is checked and why each can differ from the DDL:
        * `Type` HNSW -- an unrecognised type falls back to brute-force KNN
          (`VectorIndexCollection::open`, "using KNN").
        * `M` -- `getIntOption("M", 16)`: a mis-cased key is the default.
        * `Dimension`, `Max. Capacity` -- the object may predate this table
          (see `drop_schema`).
        * `Distance` -- printed AS WRITTEN, so this only confirms our
          spelling arrived; that the spelling is one the constructor matches
          is `METRIC_NAMES`' job.
        * `Current Rows` -- hnswlib's element count, against what was loaded
          (plus re-inserts, after a refresh; deletes never lower it).

        `ef` (ef_construction) is NOT printed by `getStatus` at this release,
        so it is the one option this cannot confirm.
        """
        assert self._index is not None
        status = self.index_status()
        self._status = status
        want = {
            "Type": "HNSW",
            "M": str(int(self._index.m)),
            "Dimension": str(int(self._index.dim)),
            "Distance": self.dialect.metric_names[self._index.metric],
        }
        if self._capacity:
            want["Max. Capacity"] = str(self._capacity)
        if expected_rows is not None:
            want["Current Rows"] = str(int(expected_rows))
        wrong = {k: (status.get(k), v) for k, v in want.items()
                 if status.get(k) != v}
        if wrong:
            raise RuntimeError(
                f"MyVector built a different index from the one declared "
                f"(key: (effective, declared)): {wrong}; full status {status}")
        print(f"[{self.name}] index status verified: {status}")
        return status

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    attrs: Union[RowAttrs, Sequence[int], numpy.ndarray]
                    ) -> None:
        """Re-insert, after refusing anything the graph has no room for."""
        after = self._expected_elements + self._pending + len(ids)
        if self._capacity and after > self._capacity:
            raise RuntimeError(
                f"re-inserting {len(ids):,} rows would put {after:,} elements "
                f"into a MyVector graph built with capacity {self._capacity:,} "
                f"(deleted rows are never removed at this release). hnswlib "
                f"throws past its capacity inside mysqld, which kills the "
                f"server; refusing here instead. Raise CAPACITY_FACTOR.")
        super().insert_rows(ids, vectors, attrs)
        self._pending += len(ids)

    # ------------------------------------------------------------------
    # Churn
    # ------------------------------------------------------------------

    def wait_index_synced(self, expected_rows: int) -> float:
        """Bring the re-inserted rows into the graph with `myvector_index_refresh`.

        **Why a refresh and not the binlog listener the release documents.**
        `docs/ONLINE_INDEX_UPDATES.md` describes `online=Y` as real-time index
        maintenance from the binary log. On this build it does not survive:
        see `INCREMENTAL_DECLINED` -- the listener exits after its first idle
        second, long before a churn step begins, and nothing restarts it. The
        release's other documented incremental path is a `track=` timestamp
        column and an explicit `myvector_index_refresh`, which reads the rows
        changed since the last build or refresh and adds them. That is what
        runs here, once per churn step after the re-insert, and its duration
        is what the churn record carries as `index_sync_s`.

        The refresh ADDS rows; it never removes one. See `maintain`.

        Verified, not assumed: the refresh must report exactly the rows
        re-inserted since the last one, and the graph's element count must be
        the running total -- rows built plus every re-insert, because deleted
        rows are never subtracted. `expected_rows` (the TABLE's count) is not
        that number and is not used.
        """
        started = time.perf_counter()
        sql = (f"CALL mysql.myvector_index_refresh('{VECTOR_COLUMN}', "
               f"'{ID_COLUMN}')")
        print(f"[{self.name}] {sql}")
        result = " | ".join(_as_text(c) for row in self._call(sql) for c in row)
        print(f"[{self.name}] refresh returned: {result}")
        self.check_build_result(result, self._pending)
        refreshed = self._pending
        self._expected_elements += self._pending
        self._pending = 0
        self.verify_status(expected_rows=self._expected_elements)
        self._next_second()
        # Per churn record, through the hook `churn._sync_shape` already reads
        # (`index_sync_note` / `index_sync_detail`): what the figure covers.
        # The rows the refresh indexed and the graph it then wrote out are the
        # closest this can get to the refresh/save split -- the procedure is
        # one call and its two halves cannot be timed apart from SQL.
        self.index_sync_note = INDEX_SYNC_NOTE
        self.index_sync_detail = {
            "refreshed_rows": int(refreshed),
            "graph_elements_saved": int(self._expected_elements),
            "graph_bytes_saved_floor": graph_bytes_floor(self._status),
        }
        return time.perf_counter() - started

    def maintain(self) -> Optional[str]:
        """Rebuild the graph from the table's live rows.

        Neither incremental path the release has can remove a row. The
        refresh selects changed rows by timestamp, and a deleted row has no
        row to select; the binlog listener, when it runs at all, dispatches
        TABLE_MAP and WRITE_ROWS events and nothing else (`myvector_binlog_loop`,
        `myvector_table_op` -> `insertVector`), which contradicts
        docs/ONLINE_INDEX_UPDATES.md ("processes WRITE_ROWS, UPDATE_ROWS, and
        DELETE_ROWS events"). So a DELETEd row stays in the graph; a search
        still returns its id; the rewritten `id IN (...)` then drops it, and
        the query returns fewer than k rows. That is the degradation churn
        measures here.

        The only remedy is to rebuild, which is what this runs --
        `myvector_index_build` drops the graph and reads the table again
        (`myvector_open_index_impl`, action `build`). It takes a READ lock on
        the table for its duration (`LOCK TABLES ... READ` in
        `BuildMyVectorIndexSQL`): readers continue, writers wait.
        """
        self._build(self.count_rows())
        return "myvector_index_build (full rebuild)"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    #: What each metric's distance runs on in the plugin: `Cosine` is
    #: `AngularDistanceSpace` -> `computeCosineDistance`, a plain loop; `L2`
    #: is hnswlib's `L2Space`, whose kernels are SIMD.
    DISTANCE_IMPL = {"Cosine": "scalar (computeCosineDistance)",
                     "L2": "simd (hnswlib L2Space)"}

    def query_path_facts(self, index: IndexSpec) -> Dict[str, Any]:
        """Two statements per query, and the distance path for this metric.

        Both are inside every timed query on both measurement paths: the
        `SET @vb_q` round trip (`_bind_query_vector` says why it exists) and
        the distance kernel. Recorded so the report states them.
        """
        dist = self.dialect.metric_names.get(index.metric, "")
        facts: Dict[str, Any] = {"round_trips_per_query": 2,
                                 "index_sync_semantics": INDEX_SYNC_NOTE,
                                 "memory_note": MEMORY_NOTE}
        if dist in self.DISTANCE_IMPL:
            facts["distance_impl"] = self.DISTANCE_IMPL[dist]
        return facts

    def set_ef_search(self, ef_search: int) -> None:
        """Remembered, and sent with every query. No statement is issued."""
        self._ef_search = int(ef_search)

    def effective_ef(self, ef: int, k: int) -> int:
        """`max(ef, k)`: hnswlib's `searchKnn` searches at `std::max(ef_, k)`."""
        return max(int(ef), int(k))

    def _bind_query_vector(self, vector) -> None:
        """Put the query vector in a session variable: `SET @vb_q = X'...'`.

        **TWO STATEMENTS PER QUERY, and the reason is a server crash.** The
        obvious single statement carries the vector inline in the
        `MYVECTOR_IS_ANN(...)` call. With a 100-dim vector (a 0.9 KB
        statement) that works; with fashion-mnist's 784 dims (6.4 KB) mysqld
        dies on the statement and writes nothing to its error log --
        reproduced on this image three times, and it is what killed the first
        smoke run. The same search with the vector in a user variable works at
        784 dims. A statement that does not contain `MYVECTOR` is never
        touched by the rewrite (`myvector_query_rewrite` returns at its first
        `strstr`), so the long literal goes in a `SET`, where it is inert.

        The cost is a second round trip per query, which no other engine here
        pays. It is the plugin's design that charges it -- the vector has to
        reach a text rewrite somehow -- and it is stated in
        docs/04-engine-notes.md beside the other MyVector asymmetries.
        A hex literal rather than a bound parameter so the value is binary
        whatever the connection character set is.
        """
        self._cur.execute(f"SET @vb_q = {hex_literal(vector)}")

    def _ann_sql(self, vector, k: int) -> str:
        """The nearest-neighbour SELECT, over `@vb_q`.

        Starts with `SELECT` and contains no comment or leading whitespace,
        because the rewrite fires only on a statement whose first character
        is in `CcAaSsEe` and which matches `^SELECT\\s+` or `^EXPLAIN\\s+`
        (`myvector_query_rewrite`). `ef_search` is always sent: the value
        is set on the shared index by whichever query last named one, so a
        query that omitted it would search at another session's setting.

        `ORDER BY myvector_row_distance(id)` restores nearest-first order,
        which an `IN` list does not have; the distances are the ones the
        search computed (`tls_distances`), not recomputed.
        """
        if self._ef_search is None:
            raise RuntimeError("set_ef_search must run before querying MyVector")
        options = f"nn={int(k)},ef_search={int(self._ef_search)}"
        return (f"SELECT {ID_COLUMN} FROM {TABLE} WHERE MYVECTOR_IS_ANN("
                f"'{VECTOR_COLUMN}','{ID_COLUMN}',@vb_q,"
                f"'{options}') ORDER BY myvector_row_distance({ID_COLUMN}) "
                f"LIMIT {int(k)}")

    def query(self, vector, k: int) -> List[int]:
        sql = self._ann_sql(vector, k)
        self._bind_query_vector(vector)
        self._cur.execute(sql)
        return [row[0] for row in self._cur.fetchall()]

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        checked_filter_column(column)
        raise NotImplementedError(FILTERED_DECLINED)

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        """Whether the statement was rewritten into a MyVector search.

        A MyVector query cannot fall back to a scan the way the other engines'
        can: an un-rewritten `MYVECTOR_IS_ANN` is an unknown function and an
        unloaded index fails `myvector_ann_set_init` -- both errors, not slow
        exact answers. What CAN go wrong silently is the rewrite not happening
        and something else answering, so the plan must contain the derived
        table the rewrite introduces, `myvector_ann` (the `table` column of
        the tabular EXPLAIN, compared exactly rather than as a substring).
        """
        column = checked_filter_column(column)
        if tag_threshold is not None:
            print(f"[{self.name}] no filtered plan to check: {FILTERED_DECLINED}")
            return False
        try:
            sql = "EXPLAIN FORMAT=TRADITIONAL " + self._ann_sql(vector, k)
            self._bind_query_vector(vector)
            self._cur.execute(sql)
            rows = [tuple(r) for r in self._cur.fetchall()]
        except Exception as exc:
            print(f"[{self.name}] EXPLAIN failed: {exc}")
            return False
        tables = [str(r[2]) for r in rows if len(r) > 2]
        used = self.dialect.index_name in tables
        if not used:
            plan = " | ".join(str(c) for r in rows for c in r)
            print(f"[{self.name}] WARNING: the MyVector search is NOT in the "
                  f"plan (no {self.dialect.index_name!r} table). Plan: {plan}")
        return used

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def index_bytes(self) -> int:
        """The graph files in `myvector_index_dir`.

        The entrypoint puts that directory beside the data directory in the
        same volume (`/var/lib/vbench/myvector-index`), so from the harness's
        read-only view it is `<data_dir>/../myvector-index`. The files are
        `ann.t1.v.hnsw.index` and the incremental-checkpoint files beside it
        (`HNSWMemoryIndex::dropIndex` names all four).
        """
        if not self.spec.data_dir:
            return 0
        base = os.path.join(os.path.dirname(self.spec.data_dir.rstrip("/")),
                            "myvector-index")
        total = 0
        try:
            names = os.listdir(base)
        except OSError:
            return 0
        for name in names:
            if name.startswith(f"{VECTOR_COLUMN}.hnsw.index"):
                try:
                    total += os.stat(os.path.join(base, name)).st_size
                except OSError:
                    pass
        return total
