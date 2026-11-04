"""Ops-harness driver for PostgreSQL + pgvector.

PostgreSQL necessarily uses a different client stack (psycopg3) from the
MySQL-family engines. That asymmetry cannot be removed and is stated in
docs/05-methodology.md; psycopg3's binary protocol is the nearest equivalent to
Connector/C's, so it is what is used here.

The substantive difference from MHNSW and VIDX is that pgvector builds its HNSW
index as a bulk operation after the data is loaded, rather than incrementally on
INSERT. `build_mode` selects which of the two comparisons is being made -- and
with it how this driver ingests, because in incremental mode the ingest is part
of the build being compared: bulk mode loads through COPY, incremental mode
through batched INSERT. Argued at length beside `INSERT_SQL`.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy

from .engine_waits import postgres_delta, postgres_sample
from .base import (DATABASE, FILTER_COLUMNS, FILTER_CTAG, FILTER_TAG, TABLE,
                   ConnectionSpec, EngineDriver, IndexSpec, LoadResult,
                   RowAttrs, checked_filter_column)

try:
    import psycopg
    import pgvector.psycopg
except ImportError:  # pragma: no cover - only the bench image has these
    psycopg = None

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

INDEX = f"{TABLE}_embedding_idx"
OPCLASS = {"angular": "vector_cosine_ops", "euclidean": "vector_l2_ops"}
OPERATOR = {"angular": "<=>", "euclidean": "<->"}

#: B-tree index name per filter column, derived so the two columns cannot end
#: up with one index between them. `create_schema` creates every entry.
SCALAR_INDEXES = {column: f"{TABLE}_{column}_idx" for column in FILTER_COLUMNS}

#: The COPY the BULK-mode load streams, and now its only caller. Two write
#: paths shared this string -- the bulk load and churn's re-insert -- because
#: each had held its own copy of it and PR 3 had to add a column to both. C4
#: ended that sharing from the other end: the incremental load and churn's
#: re-insert both go through `INSERT_SQL` below, leaving COPY on the one path
#: where it is the honest ingest -- the load whose index is built afterwards.
#:
#: The width hazard is unchanged for the path that is left: a COPY whose
#: column list and `write_row` tuple disagree raises inside the copy block,
#: which on the bulk path happens after an unknown number of rows have
#: already been streamed.
#:
#: What moved rather than went away is the PARITY requirement. It used to be
#: "one string, so the two write paths cannot disagree about a column list".
#: It is now "two strings, which must name the same columns in the same
#: order". What holds them together is `tests/test_t3b_drivers.py`'s
#: `test_both_write_paths_name_the_same_columns_in_the_same_order`, which
#: parses the column list out of each statement and compares the two. A
#: divergence fails against a four-column table -- on the churn path, after
#: the delete has already happened.
COPY_SQL = (
    f"COPY {TABLE} (id, {FILTER_TAG}, {FILTER_CTAG}, embedding) FROM STDIN"
)

#: The INSERT the INCREMENTAL-mode load and churn's re-insert both go
#: through. Three figures this repo publishes depend on these not being
#: COPYs: pgvector's incremental `build_wall_s`, its `ingest_rows_per_s` in
#: that mode, and churn's `reinsert_rows_per_s`.
#:
#: In incremental mode `create_schema` creates the HNSW index BEFORE the load,
#: so the graph is maintained row by row as the rows arrive and the load time
#: IS the build time -- `workloads/build.py` records `load.wall_seconds` as
#: `build_wall_s` and says so in its own docstring. That number is then set
#: against MHNSW's and VIDX's, which are ingested through
#: `mysql_family.INSERT_SQL` in batches (`_load_range`). COPY is Postgres's
#: bulk fast-path: one statement streaming rows, with no per-row statement to
#: parse and no per-row result to return. Driving the incremental load through
#: it charged pgvector for a cheaper ingest than every engine it was being
#: compared against, and the difference landed inside build.py's second claim,
#: "pgvector's incremental build costs X versus MHNSW's Y" -- the ingest
#: mechanism was part of X and not part of Y.
#:
#: BULK mode keeps COPY, and that distinction is part of the correction rather
#: than an omission in it. There `create_index` runs AFTER the load, so the
#: load is pure ingest with no graph work inside it, and COPY is what anyone
#: loading a corpus into pgvector actually writes; using INSERT there would
#: measure an ingest nobody uses, and would turn build.py's FIRST claim --
#: "pgvector's bulk build is N times cheaper than its own incremental build"
#: -- into a comparison of two statement kinds.
#:
#: Found by a fork rather than here: villagesql/vector-bench, branch
#: `tomas/villagesql-build-fairness`, commit 96bdcb711, which measured its
#: 60k-row pgvector incremental build at ~208 s on COPY against ~245 s on
#: batched INSERT. Those are their figures on their hardware, recorded as the
#: reason to expect this correction to MOVE the number -- and to move it
#: against pgvector -- not as a result of this repo. Nothing here has been run
#: against a live engine.
#:
#: **What this controls for and what it cannot**, stated plainly because a
#: vendor build-cost comparison is published off these numbers and a reader
#: has to know which difference they are looking at.
#:
#: CONTROLLED, after this change: the statement kind -- every engine's rows
#: now arrive as INSERTs against a live index on the paths where the graph is
#: maintained during the write; the rows per batch, 500 on both sides; and
#: therefore the number of client/server ROUND TRIPS per row.
#:
#: NOT CONTROLLED, and not controllable here: the SERVER-SIDE EXECUTIONS per
#: row, and the wire encoding they come from. psycopg3's `executemany` sends
#: one execution per row inside a pipeline, and a pipeline saves round trips
#: rather than executions; Connector/C can turn a 500-row batch into a
#: single bulk operation. "The same batch size" therefore means the two
#: clients flush at the same points, not that the server is asked to do the
#: same number of things. The two client stacks are two client stacks, which
#: the module docstring names as the asymmetry no config removes.
#:
#: Which way the residual leans, because a publication needs that and not
#: just its existence: it now plausibly runs AGAINST pgvector. Connector/C
#: can collapse a batch into one execution and psycopg3 cannot, so this
#: correction removes a pgvector advantage and may overshoot by the size of
#: that difference. The residual gap between pgvector's incremental build
#: and MHNSW's is therefore per-row graph-build cost PLUS a client-library
#: term of unknown size and probably unfavourable sign. Before this change
#: it was per-row graph-build cost plus a bulk fast-path: a much larger term
#: whose sign was certainly in pgvector's favour. Anyone quoting the
#: difference should quote it as the first of those, and should read
#: `report/render.py:_known_asymmetries`' "Client stacks differ for
#: PostgreSQL" row -- "its client overhead differs by an unmeasured amount"
#: -- as load-bearing rather than decorative.
INSERT_SQL = (
    f"INSERT INTO {TABLE} (id, {FILTER_TAG}, {FILTER_CTAG}, embedding) "
    f"VALUES (%s, %s, %s, %s)"
)

#: Rows per `executemany`, copied from the engines this measurement is
#: compared against rather than chosen here: `mysql_family._load_range`
#: batches at 500 and its `insert_rows` chunks at 500. A batch of a different
#: size is a different number of client/server round trips for the same rows,
#: which is the cost this whole change exists to stop charging to one side
#: only. `tests/test_c4_pgvector_ingest.py` reads the number out of
#: `mysql_family._load_range`'s signature and fails if the two drift, because
#: nothing else here would notice: the fork this was ported from used 1000,
#: stating that 1000 was the MySQL-family batch, and 1000 against 500 would
#: have put back part of the asymmetry it was removing.
INSERT_BATCH_ROWS = 500


#: The table option `prepare_churn` turns off and `maintain` gives back.
#: Named once because the two statements have to agree about the spelling and
#: they are forty lines apart; `SET (...)` and `RESET (...)` disagreeing is a
#: table left with autovacuum off for the life of the container.
AUTOVACUUM_OPTION = "autovacuum_enabled"

#: What `maintain()` reports it ran. The statement text rather than a verb,
#: because it lands in `extra.maintenance` and is read by a human comparing
#: engines -- "VACUUM (ANALYZE)" says what to go and read about; "vacuumed"
#: does not say whether the planner statistics moved too.
MAINTENANCE_DESCRIPTION = "VACUUM (ANALYZE)"


class PostgresDriver(EngineDriver):
    name = "pgvector"
    incremental_index = False
    # The one engine in the set with a repair path the churn workload runs.
    # MHNSW and VIDX would need OPTIMIZE TABLE, which rebuilds the whole
    # vector index at 1M rows -- a separate experiment, not a churn step --
    # and mongot merges its Lucene segments on its own schedule.
    repair_path = "vacuum_analyze"

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        self._cur = None
        self._index: Optional[IndexSpec] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self):
        conn = psycopg.connect(
            host=self.spec.host, port=self.spec.port,
            user=self.spec.user, dbname=self.spec.database,
            password=self.spec.password or None,
            autocommit=True, connect_timeout=30,
        )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pgvector.psycopg.register_vector(conn)
        # JIT compiles the distance expression per query; at these row counts
        # that costs more than it saves, and vector deployments routinely
        # disable it for exactly this reason.
        conn.execute("SET jit = off")
        return conn

    def connect(self) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg/pgvector are not installed")
        self._conn = self._open()
        self._cur = self._conn.cursor()

    def close(self) -> None:
        for closer in (self._cur, self._conn):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        self._cur = self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def drop_schema(self) -> None:
        self._cur.execute(f"DROP TABLE IF EXISTS {TABLE}")

    def create_schema(self, index: IndexSpec) -> None:
        self._index = index
        self._cur.execute(
            f"CREATE TABLE {TABLE} (id int PRIMARY KEY, tag int NOT NULL, "
            f"ctag int NOT NULL, embedding vector({index.dim}))"
        )
        # Keep vectors inline. TOASTed vectors would add a detoast to every
        # distance comparison, which pgvector's own documentation warns against.
        self._cur.execute(
            f"ALTER TABLE {TABLE} ALTER COLUMN embedding SET STORAGE PLAIN"
        )
        # Looped over SCALAR_INDEXES rather than written out twice: both
        # columns get the same kind of index, which is what makes the
        # planner's choice between them comparable. A second hand-written
        # CREATE INDEX line is where one column quietly ends up unindexed, and
        # pgvector is the engine where that matters most -- without a B-tree
        # the predicate is evaluated as a filter above the index scan, which
        # changes how many candidates `hnsw.iterative_scan` has to walk.
        for column in FILTER_COLUMNS:
            self._cur.execute(
                f"CREATE INDEX {SCALAR_INDEXES[column]} ON {TABLE} ({column})"
            )
        if index.build_mode == "incremental":
            self.create_index(index)

    def create_index(self, index: IndexSpec) -> None:
        self._index = index
        ef_construction = index.ef_construction or 200
        sql = (
            f"CREATE INDEX {INDEX} ON {TABLE} USING hnsw "
            f"(embedding {OPCLASS[index.metric]}) "
            f"WITH (m = {index.m}, ef_construction = {ef_construction})"
        )
        print(f"[pgvector] {sql}")
        self._cur.execute(sql)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load(self, vectors: numpy.ndarray,
             attrs: Union[RowAttrs, Sequence[int], numpy.ndarray],
             threads: int = 1, start_id: int = 0) -> LoadResult:
        total = len(vectors)
        threads = max(1, threads)
        attrs = RowAttrs.coerce(attrs)
        # Which ingest this load is allowed to use -- the one decision in this
        # file that moves a published build number.
        #
        # `is not None` rather than a truthiness test on `self._index`, for
        # the reason `_select`'s docstring gives at length: that function read
        # `if column else ""` for a review cycle and silently returned an
        # unfiltered query. An `IndexSpec` is never falsy today, so this is
        # habit rather than a live bug -- and habit is what the last one cost.
        #
        # The None branch is DEFENSIVE, and labelled so rather than
        # attributed to a caller, because it has none in this tree:
        # `build.run` calls `create_schema` before `load`, `main.py`'s
        # no-build path assigns `driver._index` directly (`main.py:281`), and
        # nothing in `scripts/` constructs this driver at all. What it buys
        # is that a future caller -- or a test -- loading without a schema
        # gets COPY, which is the right answer when there is no HNSW index to
        # maintain, instead of an AttributeError.
        #
        # One coupling worth naming: `capabilities()` reports
        # `incremental_index: False`, so `build.run`'s own `incremental` flag
        # reduces to exactly this condition (`workloads/build.py:93-95`). The
        # record that says "incremental: build cost is the load cost" and the
        # driver that chose INSERT are decided by one test, so there is no
        # configuration in which they disagree.
        incremental = (self._index is not None
                       and self._index.build_mode == "incremental")
        start = time.perf_counter()

        def insert_range(lo: int, hi: int, cur, report: bool = False) -> None:
            span = hi - lo
            began = time.perf_counter()
            next_report = began + PROGRESS_INTERVAL_S
            rows: List[Tuple[Any, ...]] = []
            for i in range(lo, hi):
                rows.append((start_id + i, *attrs.row(i), vectors[i]))
                if len(rows) >= INSERT_BATCH_ROWS:
                    cur.executemany(INSERT_SQL, rows)
                    # Rebound rather than cleared, as `mysql_family` does and
                    # for its reason: the list was handed to the driver, and
                    # emptying the object it was given bets on executemany
                    # having finished reading it.
                    rows = []
                    if report:
                        now = time.perf_counter()
                        if now >= next_report:
                            done = i - lo + 1
                            rate = done / max(now - began, 1e-9)
                            eta = (span - done) / max(rate, 1e-9)
                            print(f"[pgvector]   {done:,}/{span:,} rows, "
                                  f"{rate:,.0f} rows/s (INSERT), "
                                  f"ETA {eta / 60:.1f} min", flush=True)
                            next_report = now + PROGRESS_INTERVAL_S
            if rows:
                cur.executemany(INSERT_SQL, rows)
            # No COMMIT: `_open` connects with autocommit, so each batch has
            # already committed. `mysql_family._load_range` ends with one
            # against an autocommit connection too, so both paths commit per
            # batch -- the difference between them is the statement count, not
            # how much durability either side was charged for.

        def copy_range(lo: int, hi: int, cur, report: bool = False) -> None:
            span = hi - lo
            began = time.perf_counter()
            next_report = began + PROGRESS_INTERVAL_S
            with cur.copy(COPY_SQL) as copy:
                for i in range(lo, hi):
                    copy.write_row((start_id + i, *attrs.row(i), vectors[i]))
                    if report and (i - lo) % 5000 == 0:
                        now = time.perf_counter()
                        if now >= next_report:
                            done = i - lo + 1
                            rate = done / max(now - began, 1e-9)
                            eta = (span - done) / max(rate, 1e-9)
                            print(f"[pgvector]   {done:,}/{span:,} rows, "
                                  f"{rate:,.0f} rows/s, ETA {eta / 60:.1f} min",
                                  flush=True)
                            next_report = now + PROGRESS_INTERVAL_S

        # The correction, in one line; `INSERT_SQL` above is the argument for
        # it. Chosen once and used by both the single- and the multi-threaded
        # path below, because `load_threads: 8` is live in two profiles and a
        # port that switched only the single-threaded call site would leave
        # those runs measuring COPY under a record that says INSERT.
        load_range = insert_range if incremental else copy_range

        if threads == 1:
            load_range(0, total, self._cur, report=True)
        else:
            bounds = [(total * i // threads, total * (i + 1) // threads)
                      for i in range(threads)]

            def worker(bound: Tuple[int, int]) -> None:
                lo, hi = bound
                conn = self._open()
                try:
                    load_range(lo, hi, conn.cursor(), report=(lo == 0))
                finally:
                    conn.close()

            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(worker, bounds))

        elapsed = time.perf_counter() - start
        return LoadResult(rows=total, wall_seconds=elapsed, threads=threads)

    def delete_ids(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self._cur.execute(
            f"DELETE FROM {TABLE} WHERE id = ANY(%s)", ([int(i) for i in ids],)
        )

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    attrs: Union[RowAttrs, Sequence[int], numpy.ndarray]
                    ) -> None:
        attrs = RowAttrs.coerce(attrs)
        # Batched INSERT rather than COPY, and NOT conditional on
        # `build_mode`, which is the one thing to understand before changing
        # this back. Churn runs after `build.run`, so the HNSW index exists by
        # the time anything re-inserts -- in BOTH modes, because incremental
        # creates it in `create_schema` and bulk creates it in `create_index`
        # before churn is reached. Every re-inserted row therefore has its
        # graph work done inside this write, exactly as MHNSW's and VIDX's
        # does, and they re-insert through `executemany(INSERT_SQL, ...)` at
        # the same batch size. This path had no honest claim on the bulk
        # fast-path at all: unlike `load` in bulk mode, there is no later
        # CREATE INDEX for it to be the cheap prelude to.
        #
        # A second reason the first does not cover: `--workloads churn`
        # without `build` is a supported invocation -- `main.py`'s no-build
        # branch, which refuses to run unless the table already holds rows
        # from an earlier one. No schema is created in that process;
        # `main.py:281` assigns `driver._index` straight from the profile. A
        # `build_mode` gate here would therefore choose the write mechanism
        # from a DECLARED mode while the table's actual state -- a live HNSW
        # index left behind by that earlier invocation -- is not knowable
        # from it, and in `post` mode would stream a COPY into that index, in
        # the one configuration that has no build record to notice.
        #
        # The figure it was distorting is `churn.py`'s `reinsert_rows_per_s`,
        # and where that surfaces is narrower than "the report compares it",
        # so it is worth stating as measured: `report/render.py:_churn_table`
        # carries no rate column, and `_churn_notes` tabulates the rate per
        # engine only for engines whose re-insert did not finish. For a
        # completed churn it travels in the record's `extra` --
        # `records.jsonl`, the export bundle, anything reading records. The
        # unfairness was live in the data either way, which is reason enough.
        #
        # No COMMIT, for the reason `insert_range` gives: `_open` connects
        # with autocommit, so each batch has already committed.
        total = len(ids)
        for begin in range(0, total, INSERT_BATCH_ROWS):
            end = min(begin + INSERT_BATCH_ROWS, total)
            # `ids[i]` paired with `attrs.row(i)` and `vectors[i]`: the id
            # comes from the caller's list, the scalars and the vector from
            # position `i` in the arrays it was handed alongside it. Churn
            # slices all three together (`RowAttrs.__getitem__`) precisely so
            # that this positional pairing is the right one.
            self._cur.executemany(INSERT_SQL, [
                (int(ids[i]), *attrs.row(i), vectors[i])
                for i in range(begin, end)
            ])

    def count_rows(self) -> int:
        self._cur.execute(f"SELECT count(*) FROM {TABLE}")
        return int(self._cur.fetchone()[0])

    # ------------------------------------------------------------------
    # Churn
    # ------------------------------------------------------------------

    def prepare_churn(self) -> None:
        """Take autovacuum off the table, so the churn measurement is ours.

        Autovacuum runs on its own schedule against its own thresholds. A
        churn that deletes 10% of a million rows crosses those thresholds, so
        whether the tombstones are still in the heap when recall is
        re-measured depends on when the daemon last looked -- and the record
        said nothing about it either way. Two runs of the same profile could
        therefore differ by the whole size of the effect PR 4 measures, with
        the difference attributed to the engine.

        Deliberately not wrapped in a try: a DDL statement that fails here has
        not put the table in the state the following measurements claim, and a
        swallowed failure would publish a churn curve labelled "autovacuum
        off" that was taken with autovacuum on. `explain_uses_vector_index`
        swallows exceptions because a plan check must not end a run; this is
        not a check, it is the setup the measurement is defined in terms of.
        """
        self._cur.execute(
            f"ALTER TABLE {TABLE} SET ({AUTOVACUUM_OPTION} = false)"
        )

    def maintain(self) -> Optional[str]:
        """VACUUM (ANALYZE) the table, then give autovacuum back.

        ANALYZE as well as VACUUM because the churn changes the statistics the
        planner uses to decide between the HNSW index and a scan, and a VACUUM
        that reclaimed the tombstones while leaving stale statistics would
        measure the repaired index through a planner still describing the old
        table.

        **The RESET is in a `finally` and that is a tested promise, not a
        tidiness.** `prepare_churn` turns off a persistent TABLE option: it
        survives the session, the connection and this process, so a VACUUM
        that raises -- disk full, a lock timeout, the server restarting under
        it -- would otherwise leave `t1` with autovacuum off for as long as
        the table exists, and the next thing to measure it would be measuring
        a table nobody configured that way. The exception itself is re-raised:
        a repair path that failed must not be recorded as a repair that ran.

        One gap worth knowing about, because it is the half this `finally`
        cannot close: if the WORKLOAD raises between `prepare_churn` and here
        -- a failed delete, a re-insert that dies -- `maintain` is never
        called and the option stays off. It is bounded rather than fixed, and
        the binding is worth naming precisely, because the obvious argument is
        the weaker one. Inside the unit, `build.run` drops and recreates the
        table and `main.py` runs churn last, so nothing in the same unit
        measures it afterwards -- but that says nothing about any LATER unit.
        What covers those is `orchestrator/ops_pass.py:OpsRun.teardown()`,
        which removes the Docker volume and its bind directory -- the whole
        PostgreSQL data directory -- per ops unit, the volume being named
        `{run_id}-{engine}-m{M}-{build_mode}[-{storage}]-data`. The table and
        its options do not survive the unit at all, so no later unit, no later
        resource pass and no orchestrated re-run can measure a table with
        autovacuum still off. Closing the gap properly, for a server someone
        keeps by hand, needs a fourth hook, which PR 4 §4.1 does not define.
        """
        try:
            self._cur.execute(f"VACUUM (ANALYZE) {TABLE}")
        finally:
            self._cur.execute(
                f"ALTER TABLE {TABLE} RESET ({AUTOVACUUM_OPTION})"
            )
        return MAINTENANCE_DESCRIPTION

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        self._cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")

    def _op(self) -> str:
        assert self._index is not None, "create_schema must run before querying"
        return OPERATOR[self._index.metric]

    def _select(self, filtered: bool = False, *,
                column: str = FILTER_TAG) -> str:
        """The one SELECT both the query path and the EXPLAIN path run.

        Before PR 3 the filtered string was written out twice -- once in
        `query_filtered`, once in `explain_uses_vector_index` -- and adding a
        whitelisted column name to a predicate that exists in two places means
        two checks, of which the second is the one nobody notices has been
        bypassed. One builder makes "the column is checked before it reaches
        SQL" a property of the file rather than a claim about two call sites.

        **`filtered` is an explicit flag and not an `Optional[str]` sentinel,
        and that is a bug fix rather than a style preference.** This read
        `if column else ""` for one review cycle, which is truthiness: `""`,
        `None` and `0` all skipped the predicate *and the whitelist with it*,
        so `_select("")` returned an UNFILTERED select while the other four
        drivers raised ValueError on the same input. `query_filtered` would
        then have sent three parameters against two placeholders -- loud, but
        loud about placeholder counts rather than about a column name, and
        `explain_uses_vector_index` swallows exceptions, so the plan check
        would have come back False for a query nobody asked for. A `None`
        sentinel fixes the falsy-string half and leaves `None` itself a legal
        argument to a function whose whole job is to refuse illegal ones; a
        boolean cannot be mistaken for a column at all.

        So the column is whitelisted unconditionally, on both branches, which
        is also what `mysql_family._select` does and argues for -- the two
        drivers now have the same shape rather than nearly the same shape.

        **`column` is keyword-only**, and the `*` above is not decoration:
        without it `_select(True, "ctag")` and `_select(True, "tag")` are two
        positional booleans-then-string calls that read alike, and the
        argument that decides which column a measurement names would be
        identified by position. Named at every call site, it cannot be
        confused with the flag. Deleting the `*` is a silent regression, which
        is why the runtime guard below exists as well as the marker.

        Parameter order follows the string: threshold, vector, k when
        filtered; vector, k when not.
        """
        # `filtered` really is a flag, and this says so at runtime because
        # the keyword-only marker above does not. Measured: with `column`
        # keyword-only and `filtered` still positional, `_select("ctag")`
        # returns `WHERE tag < %s` -- a well-formed filtered query on the
        # WRONG column, silently, because a non-empty string is truthy. The
        # marker stops a column reaching here in `column`'s position; only
        # this stops one reaching here in the FLAG's position, which is the
        # shape a caller who forgets the flag actually writes. Two lines for
        # the one remaining way this file can measure `tag` while a record
        # says `ctag`, and it survives someone deleting the `*`.
        if not isinstance(filtered, bool):
            raise TypeError(
                f"_select's filtered argument is a flag, not a column; got "
                f"{filtered!r}. Did you mean column={filtered!r}?")
        checked = checked_filter_column(column)
        where = f"WHERE {checked} < %s " if filtered else ""
        return (
            f"SELECT id FROM {TABLE} {where}"
            f"ORDER BY embedding {self._op()} %s LIMIT %s"
        )

    def query(self, vector, k: int) -> List[int]:
        self._cur.execute(
            self._select(False), (vector, k), binary=True, prepare=True,
        )
        return [row[0] for row in self._cur.fetchall()]

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        # `prepare=True` is kept per column, which is safe for the reason the
        # spec asks about: the two columns produce two different statement
        # texts, so no cache keyed on anything derived from the statement can
        # serve one column's plan to the other. psycopg3 keys its prepared
        # statements on the query and its parameter types, but the property
        # relied on here is the weaker one and holds whatever the key is --
        # two statements that differ in their WHERE clause are two statements.
        # The arrangement that would collide is one statement text with the
        # column bound as a parameter, and PostgreSQL does not accept a
        # parameter where an identifier belongs, so that shape is unreachable
        # rather than merely avoided.
        #
        # Kept rather than dropped because dropping it would change the
        # measurement: these queries are timed one at a time in a loop, so
        # whether each one is re-planned is part of the per-query cost the
        # filtered workload reports. Whatever that costs, it has to be the
        # same for `tag` and for `ctag`, and it has to be the same as it was
        # before this column existed.
        self._cur.execute(
            self._select(True, column=column),
            (int(threshold), vector, k),
            binary=True, prepare=True,
        )
        return [row[0] for row in self._cur.fetchall()]

    def set_iterative_scan(self, mode: str) -> None:
        """pgvector 0.8 iterative index scan mode: off | relaxed_order | strict_order.

        Materially changes filtered-search recall: without it, a selective filter
        can exhaust the ef_search candidate list before finding k qualifying rows
        and simply return fewer results. Recorded per measurement so a filtered
        number is never ambiguous about which mode produced it.
        """
        self._cur.execute(f"SET hnsw.iterative_scan = {mode}")

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        # Checked here, above the try, and the placement is the point. The
        # body below swallows every exception on purpose -- a plan check that
        # fails must not end a run -- so a column rejected inside the try
        # would come back as a plain False, which every caller reads as a
        # statement about the engine. It would be a statement about the
        # caller. The measurement that followed would then fail anyway, one
        # step later in `query_filtered`, with this check already recorded in
        # the report's Validity section as an index that was not used.
        column = checked_filter_column(column)
        try:
            if tag_threshold is None:
                sql = self._select(False, column=column)
                params: Tuple[Any, ...] = (vector, k)
            else:
                sql = self._select(True, column=column)
                params = (int(tag_threshold), vector, k)
            self._cur.execute("EXPLAIN (FORMAT TEXT) " + sql, params)
            plan = " ".join(str(r[0]) for r in self._cur.fetchall())
        except Exception as exc:
            print(f"[pgvector] EXPLAIN failed: {exc}")
            return False
        # Substring, and `INDEX` is `t1_embedding_idx` while the scalar
        # indexes are `t1_tag_idx` and `t1_ctag_idx`: neither of those
        # contains the other, so a plan that used only the B-tree does not
        # read as having used the HNSW index.
        used = INDEX in plan
        if not used:
            print(
                f"[pgvector] WARNING: HNSW index NOT used "
                f"(k={k}, filtered={tag_threshold is not None}, "
                f"column={column!r}). Plan: {plan}"
            )
        return used

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def index_bytes(self) -> int:
        try:
            self._cur.execute("SELECT pg_relation_size(%s)", (INDEX,))
            return int(self._cur.fetchone()[0])
        except Exception:
            return 0

    def table_bytes(self) -> int:
        try:
            self._cur.execute("SELECT pg_table_size(%s)", (TABLE,))
            return int(self._cur.fetchone()[0])
        except Exception:
            return 0

    def server_version(self) -> str:
        try:
            self._cur.execute(
                "SELECT version() || ' / pgvector ' || "
                "coalesce((SELECT extversion FROM pg_extension WHERE extname='vector'),'?')"
            )
            return str(self._cur.fetchone()[0])
        except Exception:
            return "unknown"

    #: Postgres keeps no cumulative per-event wait time, so the capture
    #: samples `pg_stat_activity` once a second from a sampler process in the
    #: harness container, on its own connection -- never one a client holds, which
    #: would make the sampler a participant in the window it describes.
    wait_sample_interval_s = 1.0

    def wait_snapshot(self):
        if self._cur is None:
            return None, "no connection"
        return postgres_sample(self._cur)

    def wait_delta(self, before, after):
        return postgres_delta(before, after, self.wait_sample_interval_s)

    #: `HnswElementTupleData` carries the `Vector data` (src/hnsw.h at
    #: v0.8.6), and this driver indexes the `vector` type, which is float32.
    index_vector_format = "float32"
    index_includes_vectors = True

    def capabilities(self) -> Dict[str, Any]:
        return {"incremental_index": False, "ef_construction_tunable": True,
                **self.index_format()}


DRIVERS = {}


def _driver_classes() -> Dict[str, Any]:
    """Driver class name -> class.

    This is what the harness can actually drive: one entry per architecture,
    not per engine. Several engines share a driver -- mariadb and mariadb123 are
    the same server at different tags -- which is what makes adding a variant a
    config change rather than a code change.
    """
    from .mongo import MongoDriver
    from .mysql_family import AliSQLDriver, MariaDBDriver, PerconaDriver
    from .myvector import MyVectorDriver
    from .villagesql import VillageSQLDriver
    from .valkey import ValkeyDriver

    return {
        "MariaDBDriver": MariaDBDriver,
        "AliSQLDriver": AliSQLDriver,
        "PostgresDriver": PostgresDriver,
        "MongoDriver": MongoDriver,
        "ValkeyDriver": ValkeyDriver,
        # A new architecture only because four of its differences are
        # behaviour rather than dialect values -- `capabilities()` varies with
        # the build mode, `effective_ef` clamps, `maintain()` runs a real
        # repair path, and the index leaves the table definition in bulk-build
        # mode. Everything else about it is a `MySQLDialect`.
        "PerconaDriver": PerconaDriver,
        # An architecture of its own: the index is a plugin's hnswlib graph,
        # declared by a column comment, built by a stored procedure that
        # reports failure as a result row, searched through a query rewrite
        # and brought up to date by an explicit refresh procedure. See
        # harness/drivers/myvector.py.
        "MyVectorDriver": MyVectorDriver,
        # An architecture of its own: an extension's custom type and custom
        # index, text-only vector input, an inlined query literal, and a plan
        # guard that refuses to time a full scan. See
        # harness/drivers/villagesql.py.
        "VillageSQLDriver": VillageSQLDriver,
    }


def _driver_table() -> Dict[str, Any]:
    """Engine name -> driver class, for callers that still think in engines.

    Kept deliberately small and orchestrator-free. This package runs inside a
    container that mounts harness/ and nothing else, so it cannot read
    config/engines/*.yml and must not try: importing the orchestrator from here
    is a ModuleNotFoundError after the server has already started.

    The orchestrator knows which driver an engine wants and passes it by name,
    so this is only a convenience for the engines that shipped with the harness.
    """
    classes = _driver_classes()
    return {
        "mariadb": classes["MariaDBDriver"],
        # Same server software at a different tag; the driver is identical.
        "mariadb123": classes["MariaDBDriver"],
        "alisql": classes["AliSQLDriver"],
        "pgvector": classes["PostgresDriver"],
        "mongodb": classes["MongoDriver"],
        "valkey": classes["ValkeyDriver"],
        "percona": classes["PerconaDriver"],
        "myvector": classes["MyVectorDriver"],
        "villagesql": classes["VillageSQLDriver"],
        # The same extension on the MySQL 8.4 server line; the driver is identical.
        "villagesql84": classes["VillageSQLDriver"],
    }


def known_drivers() -> Tuple[str, ...]:
    """Driver class names this harness can instantiate."""
    return tuple(sorted(_driver_classes()))


def known_engines() -> Tuple[str, ...]:
    """Engine names this harness recognises without being told a driver.

    Not a constraint on --engine any more: the orchestrator validates the name
    against config/engines/*.yml and passes --driver, so an engine added there
    needs no edit here. Kept because it is the honest answer to "what does this
    harness know about on its own".
    """
    return tuple(_driver_table())


def get_driver(engine: str, spec: ConnectionSpec,
               driver: Optional[str] = None) -> EngineDriver:
    """A driver instance, chosen by driver class name when one is given.

    The orchestrator reads the driver out of config/engines/*.yml and passes it,
    which is what lets a new engine be added as configuration: this side never
    needs a new entry. Falling back to the engine name keeps the harness usable
    on its own for the engines it shipped with.
    """
    if driver:
        classes = _driver_classes()
        if driver not in classes:
            raise ValueError(f"unknown driver: {driver} "
                             f"(expected one of {sorted(classes)})")
        return classes[driver](spec)

    table = _driver_table()
    if engine not in table:
        raise ValueError(
            f"unknown engine: {engine} (expected one of {sorted(table)}, "
            f"or pass --driver to name one of {sorted(_driver_classes())})")
    return table[engine](spec)
