"""Percona Server native InnoDB HNSW vector index, driven by ann-benchmarks.

New module — upstream ann-benchmarks has no Percona Server support.

REVIEW STATUS: never run against a server. Written to be reviewed.
docs/09-adding-percona-server.md carries the evidence for every claim below and
§7 lists what only a live server can settle.

Designed against percona/percona-server at
56124b03fdc5954e2401ab4d2c7bf7cb3a767158, in the `9.7` series the vector work
was merged into. Every file:line citation in this module was derived on the
earlier `vector-mvp` pin and has not been re-derived here: the series was
rebased into `9.7`, so the paths hold and the line numbers are unverified at
this pin. Paths beginning `mysql-test/` are MTR tests and their
recorded output — a `.result` file is the server's own output, which is why they
are cited in preference to Docs/percona-hnsw.md wherever both say the same thing.

WHY THIS MODULE OVERRIDES METHODS INSTEAD OF JUST SUPPLYING A `Dialect`
----------------------------------------------------------------------
`mariadb/module.py` and `alisql/module.py` are pure `Dialect` values, because
MariaDB MHNSW and AliSQL VIDX agree on the two statements `vb_mysql.py` spells
out. Percona agrees with neither, in four places, and three of them are not
expressible as a `Dialect` field today:

1. THE PRIMARY KEY TYPE. `vb_mysql.Dialect.create_table` (vb_mysql.py:96) writes
   `id INT PRIMARY KEY`. Percona refuses a vector index on that table:

       mysql-test/suite/percona_vector/r/index_syntax.result:192-196
         CREATE TABLE t1 ( id INT UNSIGNED PRIMARY KEY, … VECTOR KEY … )
         ERROR 42000: This version of MySQL doesn't yet support 'vector indexes
         on tables lacking a BIGINT UNSIGNED single column primary key'

   The same error is recorded for a signed `BIGINT PRIMARY KEY` (:186-190), for
   a composite key of two BIGINT UNSIGNED columns (:198-204), and for an
   `ALTER TABLE … CHANGE id id INT UNSIGNED` on an already-indexed table
   (:284-285). So it is exactly `BIGINT UNSIGNED`, exactly one column. This is a
   schema difference, not an index-clause difference, which is why
   `create_table` is overridden whole.

2. THE INDEX CLAUSE. MariaDB/AliSQL take `VECTOR INDEX vi (v) M=n DISTANCE=m`.
   Percona takes `VECTOR KEY vi (v) TYPE hnsw (M = n, ef_construction = e,
   metric = m)` — a parenthesised option list after a TYPE token. An option
   value is a bare identifier or an integer and never a quoted string, so it is
   `metric = cosine`; `metric = 'cosine'` is a parse error and `M = "6"` is too
   (mysql-test/suite/percona_vector/t/index_syntax.test:363).

3. THE DISTANCE EXPRESSION. `vb_mysql._query_sql` (vb_mysql.py:403) writes
   `ORDER BY vec_distance_<metric>(v, ?)` — one function per metric, MariaDB's
   spelling. Percona has a single `DISTANCE()` taking the metric as a third
   argument, and the arity is fixed at three:

       sql/item_create.cc:1403   {"DISTANCE", SQL_FN(Item_func_vector_distance, 3)}

   so `DISTANCE(v, q)` is not a two-argument form of the same call.

4. THE PLAN MARKER. See `_verify_plan` below — the shared implementation's
   substring test would pass on a plan that did not use the index.

`_image_march` is a fifth, smaller one, and the most dangerous kind: it would
not fail, it would report a wrong value as a measured one. See its override.

docs/09 §4 names the `Dialect` fields the follow-on task should add to
`vb_mysql.py` so items 2, 3 and 4 collapse into values and this file shrinks to
the shape of `mariadb/module.py`. Nothing here edits `vb_mysql.py`.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Dict, List

import numpy

from ...vb_mysql import TABLE, Dialect, VBMySQLBase, to_binary_f32

# Percona's own compiled-in defaults, so that an unset knob here reproduces what
# the server would have chosen for itself rather than a value we invented:
#
#   storage/innobase/include/vec0vec.h:55   constexpr int default_M = 25;
#   storage/innobase/include/vec0vec.h:56   constexpr int default_ef_construction = 200;
#   storage/innobase/include/vec0vec.h:64-65  min_M = 2, max_M = 200
#   storage/innobase/include/vec0vec.h:71-72  min_ef_construction = 1,
#                                             max_ef_construction = 4096
#
# Note that default_M is 25 and not MariaDB's or AliSQL's default — the M values
# this benchmark sweeps are supplied by the profile for every engine, so the
# default is never used in a measured run, but it is what a reader reproducing a
# single query by hand will get.
DEFAULT_EF_CONSTRUCTION = 200
MIN_EF_CONSTRUCTION = 1
MAX_EF_CONSTRUCTION = 4096
MIN_M = 2
MAX_M = 200

# The EXPLAIN access type the vector index path reports. From the server's own
# recorded output:
#
#   mysql-test/suite/percona_vector/r/optimizer.result:95-96
#     id select_type table partitions type        possible_keys key …  Extra
#     1  SIMPLE      t2    NULL       vector_ann  v             v   …  Using where
#
# and, on the same file, a query that does not take the path (:127-128):
#
#     1  SIMPLE      t2    NULL       ALL         NULL          NULL …  Using filesort
PLAN_ACCESS_TYPE = "vector_ann"


@dataclasses.dataclass
class PerconaDialect(Dialect):
    """`Dialect` plus the one build knob Percona exposes and MariaDB does not.

    `ef_construction` is a field rather than a constructor argument to
    `create_table` because `create_table`'s signature is fixed by its caller,
    `VBMySQLBase.fit` (vb_mysql.py:269), which this module does not override.
    `PerconaServer.__init__` therefore builds a per-instance dialect with
    `dataclasses.replace`, which returns a new object and leaves the
    module-level `PERCONA` singleton untouched — the alternative, assigning to
    the shared instance's field, would make two concurrently constructed
    algorithms fight over one value.
    """

    ef_construction: int = DEFAULT_EF_CONSTRUCTION

    def create_table(self, dim: int, m: int, metric: str,
                     storage_engine: str) -> str:
        """`CREATE TABLE` in Percona's spelling.

        Differences from `Dialect.create_table`, each argued in this module's
        docstring: the primary key is `BIGINT UNSIGNED`, and the index is a
        `VECTOR KEY … TYPE hnsw (…)` with a parenthesised option list.

        `tag` keeps the name, type and position it has in the shared
        implementation. It is the filter column the ann pass's `base_args`
        never uses but the table has always carried, and changing it here would
        make the Percona table a different table rather than the same table on
        a different server.

        InnoDB is the only storage engine this can be: the vector index is
        InnoDB's, implemented in storage/innobase. `storage_engine` is still
        honoured rather than ignored, so that a profile asking for something
        else gets a server error naming the engine instead of silently getting
        InnoDB — the same reason AliSQL's module does not hard-code it.
        """
        return (
            f"CREATE TABLE {TABLE} (\n"
            f"  id BIGINT UNSIGNED PRIMARY KEY,\n"
            f"  tag INT NOT NULL,\n"
            f"  v VECTOR({dim}) NOT NULL,\n"
            f"  VECTOR KEY {self.index_name} (v) TYPE hnsw "
            f"(M = {int(m)}, ef_construction = {int(self.ef_construction)}, "
            f"metric = {metric})\n"
            f") ENGINE={storage_engine}"
        )


PERCONA = PerconaDialect(
    name="percona",
    # Nothing to enable. Unlike AliSQL, whose VIDX ships disabled behind
    # `vidx_disabled`, and unlike AliSQL's READ COMMITTED requirement, Percona's
    # HNSW index needs no global switch and no isolation level: no test in the
    # pinned tree's percona_vector suite sets either to make vector DML work.
    global_setup=(),
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    # SESSION and GLOBAL, default 40, range 1..100000, integers only — the
    # server's own recorded output:
    #   mysql-test/suite/sys_vars/r/innodb_hnsw_ef_search_basic.result:3-5
    #       @@global = @@session = 40
    #   :21-24   SET SESSION … = 200 leaves the global at 40
    #   :31-38   1 and 100000 are accepted verbatim
    #   :39-44   0 is clamped to 1 with "Truncated incorrect … value: '0'"
    #   :45-50   100001 is clamped to 100000, same warning
    #   :52-57   1.1, 1e1 and 'AUTO' are ER_WRONG_TYPE_FOR_VAR
    # and the index search path reads it per query:
    #   mysql-test/suite/percona_vector/t/search.test:127-130
    #       "# ef_search is a session knob." / SET SESSION innodb_hnsw_ef_search = 200
    set_ef_search="SET SESSION innodb_hnsw_ef_search = {ef_search}",
    index_name="vi",
    # Percona keeps the graph in an InnoDB auxiliary table per vector index,
    # named percona_vec_<type>_<table_id>_<index_id> (Docs/percona-hnsw.md §19).
    # The ids are not knowable in advance, so match the prefix — which is what
    # the suite itself does when it checks the files are cleaned up:
    #   mysql-test/suite/percona_vector/t/insert.test:82
    #       --list_files $MYSQLD_DATADIR/test percona_vec_*.ibd
    index_file_globs=("percona_vec_*.ibd",),
    # Left of the colon is ann-benchmarks' metric name; right is Percona's.
    # Only these two exist as INDEX metrics. The others `DISTANCE()` accepts
    # cannot be indexed at all, which the server says in as many words:
    #   mysql-test/suite/percona_vector/r/index_syntax.result:789-791  euclidean_squared
    #   :797-802  dot
    #   :803-808  manhattan
    #     ERROR 42000: This version of MySQL doesn't yet support 'HNSW indexes
    #     on anything but the EUCLIDEAN or COSINE metric'
    # The five names `DISTANCE()` itself resolves are euclidean,
    # euclidean_squared, cosine, dot, manhattan — vector-common/vector_constants.h:63-69.
    # Lowercase is used because an option value is a bare identifier and the DD
    # stores the canonical lowercase form (…/r/dd_params_resolved.result:55 shows
    # `metric = EUCLIDEAN` coming back as `` `metric`=`euclidean` ``).
    metric_names={"angular": "cosine", "euclidean": "euclidean"},
    # Percona's optimizer CAN decline the index, but never for cost reasons, and
    # the distinction matters for what this flag is protecting against.
    # vector-common/vector_constants.h:40-47 gives vector index access
    # `prohibitive_cost = 1e100`, "Chosen so that it always loses against any
    # other access method, since a vector index is only ever activated for
    # ORDER BY ... LIMIT". So activation is a shape match, not a cost decision —
    # the opposite of AliSQL, which costs the index against a full scan and
    # picks the scan at larger LIMITs. The shapes that fall off the index are
    # recorded in …/t/search.test:49-68: no LIMIT, ORDER BY … DESC, a metric the
    # graph was not built for, DISTANCE() only in WHERE, and both arguments
    # non-constant. Each returns exact results slowly, which is the failure this
    # check exists to make loud.
    verify_index_used=True,
    force_index_hint="FORCE INDEX (vi)",
)


class PerconaServer(VBMySQLBase):
    dialect = PERCONA
    vb_engine = "percona"

    def __init__(self, metric: str, method_param: Dict[str, Any]):
        # Read before super().__init__, which is what consumes `dialect`.
        ef_construction = int(method_param.get("efConstruction",
                                               DEFAULT_EF_CONSTRUCTION))
        if not MIN_EF_CONSTRUCTION <= ef_construction <= MAX_EF_CONSTRUCTION:
            # Refused here rather than left to the server, because the server's
            # refusal arrives as "Illegal index construction parameter value: 0"
            # (mysql-test/suite/percona_vector/r/dd_params_resolved.result:74-83)
            # with no mention of which parameter, after the image has started and
            # a configuration has been attributed to this point.
            raise ValueError(
                f"efConstruction must be in "
                f"[{MIN_EF_CONSTRUCTION}, {MAX_EF_CONSTRUCTION}] "
                f"(storage/innobase/include/vec0vec.h:71-72); got "
                f"{ef_construction}")
        m = int(method_param["M"])
        if not MIN_M <= m <= MAX_M:
            raise ValueError(
                f"M must be in [{MIN_M}, {MAX_M}] "
                f"(storage/innobase/include/vec0vec.h:64-65); got {m}")
        # A new dialect object, not a mutation of the shared one.
        self.dialect = dataclasses.replace(PERCONA,
                                           ef_construction=ef_construction)
        self._ef_construction = ef_construction
        super().__init__(metric, method_param)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _query_sql(self, k: int, force_index: bool = False) -> str:
        """The nearest-neighbour SELECT, in Percona's spelling.

        `DISTANCE(v, ?, '<metric>')`, with the metric as a mandatory third
        argument (sql/item_create.cc:1403 registers it with arity 3). The metric
        string is matched case-insensitively — vector-common/vector_constants.h:73-82
        resolves it with `my_strnncoll` on `my_charset_latin1`, and the suite
        uses both spellings (`'EUCLIDEAN'` in …/t/search.test:24, `'euclidean'`
        in …/t/index_build.test:79) — so the uppercase form here is for the
        reader, not for the parser.

        The ORDER BY must be ascending and there must be a LIMIT, or the index
        is not used at all (…/t/search.test:49-55). Both are structural here.
        """
        hint = f" {self.dialect.force_index_hint}" if force_index else ""
        return (
            f"SELECT id FROM {TABLE}{hint} "
            f"ORDER BY DISTANCE(v, ?, '{self._metric_fn.upper()}') LIMIT {int(k)}"
        )

    def _verify_plan(self, k: int = 10) -> bool:
        """Check the plan took the vector path, by access type and not by name.

        The shared implementation (vb_mysql.py:416) asks whether
        `dialect.index_name` appears anywhere in the joined EXPLAIN text. On
        MariaDB and AliSQL that is sound. On Percona it is a check that passes
        on a plan which did not use the index, and that is **observed in the
        vendor's own recorded output**, not inferred:

          mysql-test/suite/percona_vector/r/optimizer.result:491-492
            id select_type table partitions type possible_keys key  … Extra
            1  SIMPLE      t1    NULL       ALL  v1            NULL … Using filesort

        The only index on that table is a `VECTOR KEY`, the access type is
        `ALL` and `key` is `NULL` with `Using filesort` — a full table scan —
        and the plan text still contains the index name, in `possible_keys`. So
        `index_name in plan` returns True for a brute-force scan. `:610` shows
        the other shape, the vector index offered and a different one taken
        (`type=eq_ref, possible_keys=u,v, key=u`), and the optimizer trace at
        `:471` gives the mechanism:
        `{"cause": "vector", "index": "v", "usable": false}` sitting in
        `potential_range_indexes`.

        A filtered query — `WHERE tag < ? ORDER BY DISTANCE(…) LIMIT k`, which
        is what the ops harness runs — is exactly that second shape.

        That is the vacuity this repo keeps finding: a substring satisfied by
        something other than the thing being asserted. Here it is the
        brute-force scan published as an ANN search.

        The access type cannot be satisfied that way. `vector_ann` appears only
        when the vector path was activated, and activation is a shape match
        rather than a cost comparison — vector-common/vector_constants.h:40-47
        prices vector index access at 1e100 precisely "so that it always loses
        against any other access method". A plan that declined it reports `ALL`
        with `key=NULL` and `Using filesort` (…/r/optimizer.result:127-128).

        So this requires BOTH: the access type, and the index named somewhere.
        The access type is the load-bearing half; the name is kept so that a
        future second vector index on the table could not satisfy the check for
        the wrong one.

        `FORMAT=TRADITIONAL` IS NOT DECORATION. `vector_ann` is a value in the
        `type` column, which only the tabular format has. `explain_format`
        defaults to TRADITIONAL, so a bare `EXPLAIN` works today — but the
        variable is settable and reachable through `VB_SERVER_ARGS`, and under
        `TREE` or `JSON` the string disappears and this method returns False
        about a server that *did* use the index. That records a false negative
        in the report's Validity section and invites someone to go looking for
        an engine bug. Pinning the format costs nothing and removes the whole
        failure mode. No MTR test could have caught it, because MTR output is
        always TRADITIONAL — the suite spells the keyword out anyway at
        …/r/optimizer.result:488, which is where this spelling comes from.
        """
        probe = self._zero_probe()
        try:
            self._cur.execute(
                "EXPLAIN FORMAT=TRADITIONAL " + self._query_sql(k), (probe,))
            rows = self._cur.fetchall()
        except Exception as exc:  # pragma: no cover - depends on server build
            print(f"[vb] WARNING: could not EXPLAIN the query plan: {exc}",
                  file=sys.stderr)
            return False

        plan = " | ".join(str(col) for row in rows for col in row)
        used = PLAN_ACCESS_TYPE in plan and self.dialect.index_name in plan
        if used:
            print(f"[vb] plan check OK: {PLAN_ACCESS_TYPE} access on "
                  f"{self.dialect.index_name} ({plan})", file=sys.stderr)
        else:
            print(
                "[vb] WARNING: the vector index is NOT in the query plan. "
                "This measures a brute-force scan, not ANN search.\n"
                f"[vb] looked for access type {PLAN_ACCESS_TYPE!r} and key "
                f"{self.dialect.index_name!r}\n"
                f"[vb] plan: {plan}",
                file=sys.stderr,
            )
        return used

    def _zero_probe(self) -> bytes:
        """A valid all-zero query vector of the table's width.

        Named rather than inlined so that `_verify_plan` reads as one thought.
        Zero is a legal vector here: Field_vector::store refuses NaN and Inf
        (sql/field.cc:7107-7115) and nothing else.
        """
        return to_binary_f32(numpy.zeros(self._dim_from_table(), dtype="<f4"))

    def _dim_from_table(self) -> int:
        """The indexed column's dimensionality, from the server.

        `VECTOR_DIM()` exists on this server — sql/item_create.cc:1660 registers
        `{"VECTOR_DIM", SQL_FN(Item_func_vector_dim, 1)}` — and asking a row is
        a direct answer. The shared implementation instead divides
        `information_schema.COLUMNS.CHARACTER_OCTET_LENGTH` by 4 and returns 1
        when that column is NULL (vb_mysql.py:436), and whether a VECTOR column
        populates it on this base is not something this module can establish
        without a server. A silent 1 would not raise: it would make
        `_warm_cache` warm nothing and `_verify_plan` report False, so the run
        would record "the vector index was not used" about a server that was
        never asked a well-formed question.

        Falls back to the shared implementation on any failure, including an
        empty table, so this is strictly additional.
        """
        try:
            self._cur.execute(f"SELECT VECTOR_DIM(v) FROM {TABLE} LIMIT 1")
            row = self._cur.fetchone()
            if row and row[0]:
                return int(row[0])
        except Exception:
            pass
        return super()._dim_from_table()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    # `_image_march()` IS DELIBERATELY NOT OVERRIDDEN HERE, and this comment is
    # the record of why the override that used to sit at this spot was removed
    # rather than kept.
    #
    # It read `/opt/percona/.march` first and then the shared pair, which was
    # correct for the ann path and could never be correct for the other one.
    # There are TWO independent `-march` readers in two different containers:
    # this one, and `harness/main.py:_detect_march`, which runs in the harness
    # container — it mounts `harness/` and not `overlay/`, so no module here
    # can reach it, and `orchestrator/ops_pass.py` passes no `--march` for the
    # `or` to fall back from. Percona's OPS records would have carried
    # `march: "unknown"` with the ann records looking fine, which is the worst
    # of the three possible states: half the run measured and nothing saying
    # which half.
    #
    # `/opt/percona/.march` is now in both tuples — `vb_mysql.py:_image_march`
    # and `harness/main.py:_detect_march` — so the engine after this one finds
    # the paths where they belong instead of hitting the same thing twice.

    def get_additional(self) -> Dict[str, Any]:
        """The shared record, plus the build knob only this engine exposes.

        `ef_construction` is reported for every point rather than only when a
        profile sweeps it, because its absence from MariaDB and AliSQL is a
        documented structural asymmetry of this comparison and a Percona curve
        that did not say which value built it could not be read beside them.
        """
        extra = super().get_additional()
        extra["ef_construction"] = self._ef_construction
        return extra

    def __str__(self) -> str:
        return (f"PerconaServer(M={self._m}, "
                f"ef_construction={self._ef_construction}, "
                f"engine={self._storage_engine}, ef_search={self._ef_search})")
