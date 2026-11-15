"""Ops-harness driver for VillageSQL + vsql_vector (SVECTOR, HNSW custom index).

VillageSQL is a MySQL fork -- 9.7.2 for `villagesql`, 8.4 for `villagesql84`,
which this driver serves unchanged (every statement below is the same) --
driven through the same MariaDB Connector/Python as every other MySQL-family
engine here, so connection handling, the threaded load and sizing come from
`MySQLFamilyDriver`. Four things differ, and each is forced by the server or
the extension at the pins: VillageSQL 0.0.8 (`publish/mysql-9.7_0.0.8` =
`9999524a` for `villagesql`, `publish/mysql-8.4_0.0.8` = `6c49b098` for
`villagesql84`) with vsql-vector-pk `0.0.4` (`3c798c74`) on both:

1. **The column and the index are an extension's.** `v SVECTOR(n)` is a
   custom type; the index is a custom index, `INDEX vi (v hnsw_<metric>) USING
   EXTENDED(hnsw) WITH (M = m, ef_construction = e)` inline, or the same body
   after `CREATE INDEX vi ON t1` for a bulk build. The metric is chosen by the
   key part's index PROFILE (`hnsw_l2`, `hnsw_cosine`, ...), and the ORDER BY
   function must be the one that profile binds (`L2_DISTANCE`,
   `COSINE_DISTANCE`).

2. **Vectors travel as every other MySQL-family engine's do**: packed
   little-endian float32, through `MySQLFamilyDriver`'s multi-row INSERT and
   as the query's bound parameter. The server hands a binary value to the
   type's `from_binary` (villagesql-server #1499), which for SVECTOR is the
   raw float32 array, its length checked against the column's dimension
   (`svector_from_binary` in src/vector.cc, vsql-vector-pk #39). The KNN
   recognition accepts the bound query operand (villagesql-server #1298), and
   the statement is prepared once rather than on every execute (#1530). None
   of the four is in 0.0.7, so this path needs VillageSQL 0.0.8 and
   vsql-vector-pk 0.0.4 or later.

3. **The plan is checked, and a query that does not use the index is
   refused.** The server drives a distance scan only from an index whose
   storage props include HAS_ROW_REF (`FindCustomKnnIndexOnField`, gate added
   by server PR #1269). vsql-vector-pk's HNSW declares it, and at this pin
   EXPLAIN FORMAT=TREE shows `Custom index distance scan on vi`. Plain
   vsql-vector's does not, and with it every query was a filesort over a
   table scan with recall exactly 1.000 (this engine's first pin). CLAUDE.md's
   "every engine can fall back to a full table scan" is therefore checked
   rather than assumed: `query` and `query_filtered` EXPLAIN once per
   connection and raise `BruteForceRefused` rather than time a brute-force
   scan under an ANN label.

4. **Two workloads are declined.** DELETE on a table with a custom index is
   rejected, so churn cannot run; and a WHERE beside the KNN ORDER BY is
   applied after the scan, to its top max(ef_search, LIMIT) candidates, so a
   selective predicate returns fewer than k rows. Both are
   `declined_workloads`, so the report names them and the reason instead of
   the engine vanishing from those sections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (DATABASE, FILTER_TAG, TABLE, ConnectionSpec, IndexSpec,
                   checked_filter_column)
from .mysql_family import MySQLDialect, MySQLFamilyDriver, to_binary_f32

#: The EXPLAIN FORMAT=TREE text the server prints for a distance scan
#: (`sql/join_optimizer/explain_access_path.cc`: "Custom index distance scan
#: on " + key name). TREE and not TRADITIONAL because TRADITIONAL has no row
#: for it: on the pinned build TRADITIONAL shows `type ALL ... Using filesort`
#: and TREE shows `Sort: vsql_vector.l2_distance(...)` over a table scan.
PLAN_MARKER = "Custom index distance scan on"

#: metric key (`MySQLDialect.metric_names` value) -> (index profile, ORDER BY
#: function). The profile names are the extension's (`HNSW_*_PROFILE` in
#: src/vector.cc); the function must be the one the profile binds, or the
#: server does not recognise the ORDER BY as a KNN request.
PROFILES: Dict[str, Tuple[str, str]] = {
    "l2": ("hnsw_l2", "L2_DISTANCE"),
    "cosine": ("hnsw_cosine", "COSINE_DISTANCE"),
}

#: `vsql_vector.ef_search`'s bounds and default (`DEFAULT_EF_SEARCH`,
#: `MIN_EF_SEARCH`, `MAX_EF_SEARCH` in src/index/hnsw/storage.h).
DEFAULT_EF_SEARCH = 40
MAX_EF_SEARCH = 65536

#: `Options::DEFAULT_EF_CONSTRUCTION` and the extension's validation, which
#: refuses ef_construction below M or above 32 x M
#: (`MAX_EF_CONSTRUCTION_FACTOR`).
DEFAULT_EF_CONSTRUCTION = 200
MAX_EF_CONSTRUCTION_FACTOR = 32
#: `Options::validate`'s M floor and `Options::MAX_M` (storage.cc / .h).
MIN_M = 2
MAX_M = 128

DELETE_REJECTED = (
    "DELETE on a table with a custom index (USING EXTENDED) is not supported "
    "yet.")
CHURN_DECLINED = (
    "VillageSQL rejects every DELETE on a table carrying a custom index "
    f"(the server's error: \"{DELETE_REJECTED}\"), so rows cannot be churned. "
    "Re-inserting without deleting would measure a different workload under "
    "this one's label.")
FILTERED_DECLINED = (
    "vsql_vector applies a WHERE beside the KNN ORDER BY after the index "
    "search: the HNSW scan returns its top max(ef_search, LIMIT) candidates "
    "and the predicate filters those (vsql-vector-pk README, 'Filtering with "
    "a WHERE clause'), so a selective predicate returns fewer than k rows "
    "even when enough matching rows exist. Measuring that would publish "
    "post-filtered recall as filtered search.")

VILLAGESQL_DIALECT = MySQLDialect(
    name="villagesql",
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    global_setup=(),
    # A per-session extension variable (vsql-vector #76), so one connection's
    # ef never leaks into another's -- the concurrency pool relies on that.
    set_ef_search="SET SESSION vsql_vector.ef_search = {ef_search}",
    index_name="vi",
    # The graph lives in extension-owned pages inside t1.ibd; there is no
    # file of its own to glob. See `index_bytes`.
    index_file_globs=(),
    metric_names={"angular": "cosine", "euclidean": "l2"},
    index_clause=("INDEX {name} (v {profile}) USING EXTENDED(hnsw) "
                  "WITH (M = {m}, ef_construction = {ef_construction})"),
    plan_marker=PLAN_MARKER,
    # `{metric_upper}` is "L2" or "COSINE", so this is the function the
    # index profile binds (PROFILES), with the query vector bound.
    distance_expr="{metric_upper}_DISTANCE(v, ?)",
    ef_construction=DEFAULT_EF_CONSTRUCTION,
)


class BruteForceRefused(RuntimeError):
    """A query whose plan does not use the vector index, refused unmeasured."""


def checked_ef_construction(m: int, ef_construction: Optional[int]) -> int:
    """The ef_construction a build at `m` is given, or ValueError.

    The extension refuses ef_construction < M and > 32 x M
    (`Options::validate`, src/index/hnsw/storage.cc). This driver used to
    CLAMP into that range -- 200 became 192 at M=6 -- and record both values,
    but every label upstream of the driver (the ann result file, the frontier
    choice, the ops build selection) still said 200, and a build that is not
    the one its label names is the failure `orchestrator/ann_knobs.py`
    exists to prevent. The orchestrator drops those combinations from the
    render (`ann_knobs.excluded_builds`), so reaching this refusal means a
    caller bypassed it; it is refused here before the server refuses it with
    nothing better than "Custom type operation failed".
    """
    ef = int(ef_construction if ef_construction is not None
             else DEFAULT_EF_CONSTRUCTION)
    m = int(m)
    if not MIN_M <= m <= MAX_M:
        raise ValueError(f"vsql_vector refuses M={m}: it must lie in "
                         f"[{MIN_M}, {MAX_M}]")
    if not m <= ef <= MAX_EF_CONSTRUCTION_FACTOR * m:
        raise ValueError(
            f"vsql_vector refuses ef_construction={ef} at M={m}: it must lie "
            f"in [M, {MAX_EF_CONSTRUCTION_FACTOR} x M] = "
            f"[{m}, {MAX_EF_CONSTRUCTION_FACTOR * m}]")
    return ef


class VillageSQLDriver(MySQLFamilyDriver):
    dialect = VILLAGESQL_DIALECT
    name = "villagesql"
    declined_workloads: Tuple[Tuple[str, str], ...] = (
        ("filtered", FILTERED_DECLINED),
        ("churn", CHURN_DECLINED),
    )

    #: Whether the host says the bulk `CREATE INDEX` is safe to run. False at
    #: this pin (`capabilities.bulk_vector_index_build` in the engine config:
    #: an intermittent latch deadlock aborts the server), which is why the
    #: orchestrator's default build mode here is `incremental`.
    bulk_vector_index_build: bool = True

    def __init__(self, spec: ConnectionSpec,
                 dialect: Optional[MySQLDialect] = None):
        super().__init__(spec, dialect)
        #: Plan verdict per query shape, keyed on "filtered?": the filtered
        #: query has a WHERE the unfiltered check never saw, so one verdict
        #: must not vouch for the other. Reset by `create_schema`, because a
        #: new table (another M, another build mode) is a new plan.
        self._plan_checked: Dict[bool, bool] = {}
        self._bytes_before_index = 0
        self._bytes_after_index = 0

    # ------------------------------------------------------------------
    # Facts every record carries
    # ------------------------------------------------------------------

    def query_path_facts(self, index: IndexSpec) -> Dict[str, Any]:
        return {
            "extension_status": ("preview (vsql_allow_preview_extensions=ON; "
                                 "dev/unstable extension ABI)"),
        }

    def capabilities(self) -> Dict[str, Any]:
        """The build mode decides `incremental_index`, as for Percona.

        The graph is maintained on INSERT when the index is declared in
        CREATE TABLE, and built over the loaded rows by CREATE INDEX
        otherwise (which can crash the server at this pin; see
        `declare_bulk_vector_index_build`). `harness/workloads/build.py` reads
        this flag to decide whether `create_index` runs at all.
        """
        incremental = (self._index is None
                       or self._index.build_mode == "incremental")
        caps: Dict[str, Any] = {"incremental_index": incremental,
                                "ef_construction_tunable": True}
        if self._index is not None:
            caps["ef_construction"] = checked_ef_construction(
                self._index.m, self._index.ef_construction)
        return caps

    def effective_ef(self, ef: int, k: int) -> int:
        """`max(k, ef)`: the scan searches at that width
        (`svector::hnsw` scan begin: `std::max<uint32_t>(k, read_ef_search())`)."""
        return max(int(ef), int(k))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _vector_column(self, index: IndexSpec) -> str:
        return f"v SVECTOR({int(index.dim)}) NOT NULL"

    def _profile(self, index: IndexSpec) -> Tuple[str, str]:
        return PROFILES[self.dialect.metric_names[index.metric]]

    def _index_clause(self, index: IndexSpec, metric: str) -> str:
        profile, _function = PROFILES[metric]
        return self.dialect.index_clause.format(
            name=self.dialect.index_name, profile=profile, m=int(index.m),
            ef_construction=checked_ef_construction(index.m,
                                                   index.ef_construction))

    def create_schema(self, index: IndexSpec) -> None:
        self._index = index
        self._plan_checked = {}
        ddl = self._create_table_ddl(
            index, with_index=index.build_mode == "incremental")
        print(f"[{self.name}] {ddl}")
        self._cur.execute(ddl)

    def declare_bulk_vector_index_build(self, supported: bool) -> None:
        """The host's reading of `capabilities.bulk_vector_index_build`.

        There is no repair path to withdraw (churn is declined), so all this
        changes is what `create_index` says before it runs a bulk build a
        profile asked for explicitly: the statement is still issued -- that
        request is how a newer pin gets re-tested -- but the phase log names
        the known crash first, so a lost connection is not a mystery.
        """
        self.bulk_vector_index_build = bool(supported)

    def create_index(self, index: IndexSpec) -> None:
        """The bulk build: `CREATE INDEX vi ON t1 (...) USING EXTENDED(hnsw)`.

        A no-op in incremental mode. The tablespace size is read either side
        so `index_bytes` has something better than zero to report.
        """
        self._index = index
        if index.build_mode == "incremental":
            return
        profile, _function = self._profile(index)
        sql = (f"CREATE INDEX {self.dialect.index_name} ON {TABLE} "
               f"(v {profile}) USING EXTENDED(hnsw) WITH (M = {int(index.m)}, "
               f"ef_construction = "
               f"{checked_ef_construction(index.m, index.ef_construction)})")
        self._bytes_before_index = self._tablespace_bytes()
        if not self.bulk_vector_index_build:
            print(f"[{self.name}] WARNING: the engine config declares this "
                  f"bulk build broken at this pin (an intermittent latch "
                  f"deadlock in the extension storage ABI aborts the server); "
                  f"running it because the profile asked for build_mode "
                  f"'post'.")
        print(f"[{self.name}] {sql}")
        self._cur.execute(sql)
        self._bytes_after_index = self._tablespace_bytes()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def delete_ids(self, ids: Sequence[int]) -> None:
        """Refused before the server refuses it, with the server's reason.

        Churn is declined, so nothing in the harness reaches this; a lab
        script that does gets the measured error rather than a statement the
        server is known to reject.
        """
        if not ids:
            return
        raise NotImplementedError(f"villagesql: {DELETE_REJECTED}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _plan(self, vector, k: int, tag_threshold: Optional[int],
              column: str) -> str:
        filtered = tag_threshold is not None
        self._cur.execute(
            "EXPLAIN FORMAT=TREE " + self._select(k, filtered, column=column),
            (int(tag_threshold), to_binary_f32(vector)) if filtered
            else (to_binary_f32(vector),))
        return " | ".join(str(c) for row in self._cur.fetchall() for c in row)

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        """True only when the plan is a distance scan on `vi`.

        Marker AND index name, the conjunction the base class uses: the marker
        is the access path, the name rules out another custom index.
        """
        column = checked_filter_column(column)
        try:
            plan = self._plan(vector, k, tag_threshold, column)
        except Exception as exc:
            print(f"[{self.name}] EXPLAIN failed: {exc}")
            return False
        used = f"{PLAN_MARKER} {self.dialect.index_name}" in plan
        if not used:
            print(f"[{self.name}] WARNING: vector index NOT used "
                  f"(k={k}, filtered={tag_threshold is not None}, "
                  f"column={column!r}); looked for "
                  f"{PLAN_MARKER + ' ' + self.dialect.index_name!r}. "
                  f"Plan: {plan[:400]}")
        return used

    def _require_index(self, vector, k: int,
                       tag_threshold: Optional[int] = None,
                       column: str = FILTER_TAG) -> None:
        """Refuse to time a query the server would answer by a full scan.

        Checked once per driver and query shape (each concurrency client is
        its own driver, so each checks its own connection) with the vector
        about to be queried. A refusal fails the workload loudly, which the report lists
        under failed phases -- the honest outcome -- where a warning would
        let an exact scan be published as an ANN recall/QPS point.
        """
        filtered = tag_threshold is not None
        if filtered not in self._plan_checked:
            self._plan_checked[filtered] = self.explain_uses_vector_index(
                vector, k, tag_threshold, column)
        if not self._plan_checked[filtered]:
            raise BruteForceRefused(
                "villagesql: the query plan is not a 'Custom index distance "
                "scan on vi' -- the server would answer by sorting a full "
                "table scan, which is exact and is not ANN search. Refusing "
                "to measure it. The scan needs an index that declares "
                "HAS_ROW_REF (vsql-vector-pk's HNSW; plain vsql-vector's "
                "does not), built against a server with the KNN scan.")

    def query(self, vector, k: int) -> List[int]:
        self._require_index(vector, k)
        return super().query(vector, k)

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        # Declined (`declined_workloads`); kept correct for a lab script.
        self._require_index(vector, k, threshold, column)
        return super().query_filtered(vector, k, threshold, column)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _tablespace_bytes(self) -> int:
        try:
            self._cur.execute(
                "SELECT FILE_SIZE FROM information_schema.INNODB_TABLESPACES "
                "WHERE NAME = ?", (f"{DATABASE}/{TABLE}",))
            row = self._cur.fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception:
            return 0

    def index_bytes(self) -> int:
        """Growth of t1.ibd across the bulk build; 0 for an incremental one.

        The HNSW pages live in extension-owned segments inside the table's
        own tablespace, and the server's catalog does not count them:
        `innodb_index_stats` reports `size = 1` page for `vi` and
        `INNODB_INDEXES.PAGE_NO` is -1. The tablespace delta across CREATE
        INDEX is the graph's on-disk footprint, rounded up to InnoDB's extent
        allocation. An incremental build interleaves graph pages with row
        pages and has no such boundary, so it reports 0 rather than a guess.
        """
        return max(0, self._bytes_after_index - self._bytes_before_index)

    def table_bytes(self) -> int:
        """The whole tablespace: rows, SVECTOR storage and the graph."""
        size = self._tablespace_bytes()
        return size if size else super().table_bytes()
