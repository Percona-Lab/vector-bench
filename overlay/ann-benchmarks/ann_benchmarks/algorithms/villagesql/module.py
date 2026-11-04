"""VillageSQL + vsql_vector (SVECTOR, HNSW custom index, built from
vsql-vector-pk), driven by ann-benchmarks.

New module; upstream ann-benchmarks has no VillageSQL support. The ops-harness
driver, `harness/drivers/villagesql.py`, carries the evidence for each
behaviour relied on here; this module follows it rather than restating it and
shares none of its code, because the bench container mounts `overlay/` and the
harness container mounts `harness/`, and neither can import the other.

What differs from `VBMySQLBase`, all of it forced by the server or the
extension:

* `v SVECTOR(n)` and a custom index declared in CREATE TABLE,
  `INDEX vi (v hnsw_<metric>) USING EXTENDED(hnsw) WITH (M = m,
  ef_construction = e)`, maintained on every INSERT -- so `build_seconds` is
  the load, as for MHNSW, VIDX and Percona. NOT the bulk `CREATE INDEX` over
  a loaded table, although the server accepts it: it can crash the server
  (`capabilities.bulk_vector_index_build` in config/engines/villagesql.yml
  has the evidence).
* Vectors are text (`'[a,b,...]'`, shortest float32 repr per element). The
  query vector is inlined, because the server's KNN recognition needs a
  constant query operand.
* `vsql_vector.ef_search` is a per-session variable, set on every connection
  that queries.
* **The plan is checked and a full scan is REFUSED, not measured.** The
  server drives a distance scan only from an index that stores row
  references; vsql-vector-pk's does, plain vsql-vector's does not, and with
  it every query was a filesort over a table scan with recall 1.000.
  `set_query_arguments` raises when EXPLAIN FORMAT=TREE lacks "Custom index
  distance scan on vi", so no recall/QPS point is written for a brute-force
  scan.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional

import numpy

from ...vb_mysql import (DATABASE, INSERT_BATCH, PROGRESS_INTERVAL_S, TABLE,
                         Dialect, VBMySQLBase)

# vsql-vector-pk 93ed008, src/index/hnsw/storage.h (`Options::DEFAULT_M`,
# `DEFAULT_EF_CONSTRUCTION`, `MAX_M`, `MAX_EF_CONSTRUCTION_FACTOR`) and
# `Options::validate` in storage.cc (the M floor of 2). Repeated as
# `KNOB_BOUNDS["villagesql"]` in orchestrator/ann_knobs.py, which cannot
# import this file; tests/test_e26_villagesql_engine.py fails if they differ.
DEFAULT_EF_CONSTRUCTION = 200
MIN_M = 2
MAX_M = 128
MAX_EF_CONSTRUCTION_FACTOR = 32
PLAN_MARKER = "Custom index distance scan on vi"
METRIC_NAMES = {"angular": "cosine", "euclidean": "l2"}
PROFILES = {"l2": ("hnsw_l2", "L2_DISTANCE"),
            "cosine": ("hnsw_cosine", "COSINE_DISTANCE")}


def text_literal(vector) -> str:
    """Same encoding as `harness/drivers/villagesql.py:text_literal`."""
    arr = numpy.asarray(vector, dtype="<f4")
    if not numpy.all(numpy.isfinite(arr)):
        raise ValueError("SVECTOR cannot carry a non-finite element")
    return "[" + ",".join(str(x) for x in arr) + "]"


def checked_ef_construction(m: int, ef_construction: int) -> int:
    """`harness/drivers/villagesql.py:checked_ef_construction`: the extension
    refuses ef_construction < M and > 32 x M, and a build whose result file
    names a value it was not built with is refused rather than clamped. The
    render never emits one (`orchestrator/ann_knobs.py` excluded_builds)."""
    ef, m = int(ef_construction), int(m)
    if not MIN_M <= m <= MAX_M:
        raise ValueError(f"vsql_vector refuses M={m}: it must lie in "
                         f"[{MIN_M}, {MAX_M}]")
    if not m <= ef <= MAX_EF_CONSTRUCTION_FACTOR * m:
        raise ValueError(
            f"vsql_vector refuses ef_construction={ef} at M={m}: it must lie "
            f"in [M, {MAX_EF_CONSTRUCTION_FACTOR} x M] = "
            f"[{m}, {MAX_EF_CONSTRUCTION_FACTOR * m}]")
    return ef


@dataclasses.dataclass
class VillageSQLDialect(Dialect):
    """`Dialect` plus the build knob, set per instance with
    `dataclasses.replace` as `percona/module.py` does."""

    ef_construction: int = DEFAULT_EF_CONSTRUCTION

    def create_table(self, dim: int, m: int, metric: str,
                     storage_engine: str) -> str:
        """The table WITH the HNSW index, maintained on every INSERT."""
        profile, _function = PROFILES[metric]
        return (
            f"CREATE TABLE {TABLE} (\n"
            f"  id INT PRIMARY KEY,\n"
            f"  tag INT NOT NULL,\n"
            f"  v SVECTOR({int(dim)}) NOT NULL,\n"
            f"  INDEX vi (v {profile}) USING EXTENDED(hnsw) "
            f"WITH (M = {int(m)}, ef_construction = {int(self.ef_construction)})\n"
            f") ENGINE={storage_engine}"
        )


VILLAGESQL = VillageSQLDialect(
    name="villagesql",
    global_setup=(),
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    set_ef_search="SET SESSION vsql_vector.ef_search = {ef_search}",
    index_name="vi",
    index_file_globs=(),
    metric_names=METRIC_NAMES,
    verify_index_used=True,
)


class VillageSQL(VBMySQLBase):
    dialect = VILLAGESQL
    vb_engine = "villagesql"

    def __init__(self, metric: str, method_param: Dict[str, Any]):
        m = int(method_param["M"])
        self._ef_construction = checked_ef_construction(
            m, method_param.get("efConstruction", DEFAULT_EF_CONSTRUCTION))
        self._dim = 0
        self.dialect = dataclasses.replace(
            VILLAGESQL, ef_construction=self._ef_construction)
        super().__init__(metric, method_param)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def fit(self, X: numpy.ndarray) -> None:
        """CREATE TABLE with the index, then the load, which builds the graph.

        `VBMySQLBase.fit` sets `build_seconds` to the load time, which is the
        build for an index maintained on INSERT.
        """
        self._dim = int(X.shape[1])
        super().fit(X)

    def _measure_index_bytes(self) -> int:
        """0: the graph has no file of its own (harness/drivers/villagesql.py:
        index_bytes). The pages are inside t1.ibd, interleaved with the rows
        an incremental build writes beside them, and the catalog reports one
        page for `vi`; a guess here would chart as a measurement."""
        return 0

    def _insert_serial(self, cur, X: numpy.ndarray, offset: int = 0,
                       label: str = "") -> None:
        """`VBMySQLBase._insert_serial` with the vector as text."""
        sql = f"INSERT INTO {TABLE} (id, tag, v) VALUES (%s, %s, %s)"
        total = len(X)
        started = time.time()
        next_report = started + PROGRESS_INTERVAL_S
        rows: List[Any] = []
        for i, embedding in enumerate(X):
            idx = offset + i
            rows.append((idx, idx % 100, text_literal(embedding)))
            if len(rows) >= INSERT_BATCH:
                cur.executemany(sql, rows)
                cur.execute("COMMIT")
                rows = []
                now = time.time()
                if now >= next_report:
                    rate = (i + 1) / max(now - started, 1e-9)
                    print(f"[vb] {label}{i + 1:,}/{total:,} rows, "
                          f"{rate:,.0f} rows/s", file=sys.stderr)
                    next_report = now + PROGRESS_INTERVAL_S
        if rows:
            cur.executemany(sql, rows)
        cur.execute("COMMIT")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _ann_sql(self, v, k: int) -> str:
        _profile, function = PROFILES[self._metric]
        return (f"SELECT id FROM {TABLE} "
                f"ORDER BY {function}(v, '{text_literal(v)}') LIMIT {int(k)}")

    def set_query_arguments(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)
        self._cur.execute(self.dialect.set_ef_search.format(ef_search=self._ef_search))
        if self._plan_verified is None:
            self._plan_verified = self._verify_plan()
        if not self._plan_verified:
            raise RuntimeError(
                "villagesql: the query plan is not a 'Custom index distance "
                "scan on vi'; the server would answer by sorting a full table "
                "scan, which is exact and is not ANN search. Refusing to "
                "measure it (see harness/drivers/villagesql.py).")
        # After the plan check, so a refused full scan is not warmed first:
        # once per built index, then the per-configuration top-up
        # (vb_warmup.py, ANN_MEASUREMENT_VERSION 3).
        self._warm_index_once()
        self._warm_cache()

    def _warm_cache(self) -> None:
        from ...vb_mysql import WARMUP_QUERIES
        if WARMUP_QUERIES <= 0 or self._dim <= 0:
            return
        rng = numpy.random.RandomState(0)
        for _ in range(WARMUP_QUERIES):
            try:
                self._cur.execute(self._ann_sql(rng.normal(size=self._dim), 10))
                self._cur.fetchall()
            except Exception:
                return

    def _verify_plan(self, k: int = 10) -> bool:
        probe = numpy.ones(max(self._dim, 1), dtype="<f4")
        try:
            self._cur.execute("EXPLAIN FORMAT=TREE " + self._ann_sql(probe, k))
            rows = self._cur.fetchall()
        except Exception as exc:
            print(f"[vb] WARNING: could not EXPLAIN the query plan: {exc}",
                  file=sys.stderr)
            return False
        plan = " | ".join(str(c) for r in rows for c in r)
        used = PLAN_MARKER in plan
        print(f"[vb] plan check {'OK' if used else 'FAILED'}: {plan[:400]}",
              file=sys.stderr)
        return used

    def query(self, v, n: int) -> List[int]:
        self._cur.execute(self._ann_sql(v, n))
        return [row[0] for row in self._cur.fetchall()]

    def batch_query(self, X, n: int) -> None:
        threads = self._insert_threads

        def worker(chunk):
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"USE {DATABASE}")
            cur.execute(self.dialect.set_ef_search.format(ef_search=self._ef_search))
            out = []
            for v in chunk:
                cur.execute(self._ann_sql(v, n))
                out.append([row[0] for row in cur.fetchall()])
            cur.close()
            conn.close()
            return out

        chunks = [X[i::threads] for i in range(threads)]
        pool = ThreadPool(threads)
        try:
            results = pool.map(worker, chunks)
        finally:
            pool.close()
            pool.join()
        merged: List[Optional[List[int]]] = [None] * len(X)
        for c, chunk_results in enumerate(results):
            for j, res in enumerate(chunk_results):
                merged[c + j * threads] = res
        self._batch_results = [r if r is not None else [] for r in merged]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_additional(self) -> Dict[str, Any]:
        extra = super().get_additional()
        extra["ef_construction"] = self._ef_construction
        extra["build_seconds"] = round(self._build_seconds, 3)
        extra["vector_binding"] = ("text literal ('[a,b,...]', "
                                   "parsed by strtof)")
        extra["extension_status"] = ("preview (vsql_allow_preview_extensions=ON; "
                                     "dev/unstable extension ABI)")
        return extra

    def __str__(self) -> str:
        return (f"VillageSQL(M={self._m}, ef_construction={self._ef_construction}, "
                f"ef_search={self._ef_search})")
