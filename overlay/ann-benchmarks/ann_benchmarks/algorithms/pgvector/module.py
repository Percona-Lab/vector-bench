"""PostgreSQL + pgvector HNSW, driven by ann-benchmarks.

Refreshed from the module in MariaDB's fork: PostgreSQL 17 instead of 14, the
server started through the vector-bench entrypoint instead of `service
postgresql start`, an explicit `tag` column so the ops harness can run filtered
search against the same schema, and index build timed separately from load.

One asymmetry deserves naming here rather than in a footnote. pgvector builds
its HNSW index as a bulk operation *after* the data is loaded, while MariaDB
MHNSW and AliSQL VIDX build theirs incrementally as rows arrive. Bulk build is
substantially cheaper per vector. Comparing pgvector's bulk build against the
others' incremental build measures two different operations.

`build_mode` therefore selects which comparison is being made:

  post         load, then CREATE INDEX  — pgvector's idiomatic path, fastest
  incremental  CREATE INDEX, then load  — mirrors what MHNSW and VIDX must do

The report presents both, because "pgvector builds faster" and "pgvector builds
faster when allowed to bulk-build" are different claims.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy

from ..base.module import BaseANN
from ...vb_warmup import IndexWarmupMixin

try:  # pragma: no cover - exercised only inside the bench image
    import psycopg
    import pgvector.psycopg
except ImportError:  # pragma: no cover
    psycopg = None

SERVER_START_TIMEOUT_S = int(os.environ.get("VB_SERVER_START_TIMEOUT", "120"))
SERVER_STOP_TIMEOUT_S = int(os.environ.get("VB_SERVER_STOP_TIMEOUT", "300"))
ENTRYPOINT = os.environ.get("VB_ENTRYPOINT", "/usr/local/bin/vb-entrypoint")

TABLE = "t1"
INDEX = "t1_embedding_idx"
DATABASE = os.environ.get("POSTGRES_DB", "ann")
SOCKET_DIR = "/var/run/postgresql"

OPCLASS = {"angular": "vector_cosine_ops", "euclidean": "vector_l2_ops"}
OPERATOR = {"angular": "<=>", "euclidean": "<->"}



# Queries run before each configuration is timed. The first measured point
# otherwise pays for a cold graph cache and lands slower than the next one,
# which inverts the low end of the recall/QPS curve: on a 990k x 1536 corpus
# both MySQL-family engines reported ef_search=10 as slower than ef_search=20,
# which cannot happen once the cache is warm.
WARMUP_QUERIES = int(os.environ.get("VB_WARMUP_QUERIES", "30"))


class PGVector(IndexWarmupMixin, BaseANN):
    def __init__(self, metric: str, method_param: Dict[str, Any]):
        if psycopg is None:
            raise RuntimeError(
                "psycopg/pgvector are not installed in this image; "
                "the pgvector-bench image is required to run this module"
            )
        if metric not in OPCLASS:
            raise RuntimeError(f"unsupported metric for pgvector: {metric}")

        self._metric = metric
        self._m = int(method_param["M"])
        self._ef_construction = int(method_param.get("efConstruction", 200))
        self._build_mode = method_param.get("build_mode", "post")
        if self._build_mode not in ("post", "incremental"):
            raise RuntimeError(f"unknown build_mode: {self._build_mode}")

        self._ef_search: Optional[int] = None
        self._index_bytes = 0
        self._load_seconds = 0.0
        self._build_seconds = 0.0
        self._plan_verified: Optional[bool] = None
        self._dim: Optional[int] = None
        self._batch_results: List[List[int]] = []

        self._query_sql = (
            f"SELECT id FROM {TABLE} ORDER BY embedding {OPERATOR[metric]} %s LIMIT %s"
        )

        self._server = None
        self._start_server()
        self._conn = self._connect()
        self._cur = self._conn.cursor()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        if not os.path.exists(ENTRYPOINT):
            raise RuntimeError(f"entrypoint not found at {ENTRYPOINT}")
        print(f"[vb] starting postgres: {ENTRYPOINT} server", file=sys.stderr)
        print(f"[vb] VB_SERVER_ARGS={os.environ.get('VB_SERVER_ARGS', '')}", file=sys.stderr)
        self._server = subprocess.Popen(
            [ENTRYPOINT, "server"], stdout=sys.stderr, stderr=sys.stderr
        )

        deadline = time.time() + SERVER_START_TIMEOUT_S
        last_error = None
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"postgres exited during startup with code {self._server.returncode}"
                )
            try:
                conn = psycopg.connect(
                    host=SOCKET_DIR, dbname=DATABASE, user="postgres", connect_timeout=2
                )
                conn.close()
                print("[vb] postgres is up", file=sys.stderr)
                return
            except Exception as exc:  # psycopg.OperationalError and friends
                last_error = exc
            time.sleep(0.5)
        raise TimeoutError(
            f"postgres did not become ready within {SERVER_START_TIMEOUT_S}s: {last_error}"
        )

    def _connect(self):
        conn = psycopg.connect(
            host=SOCKET_DIR, dbname=DATABASE, user="postgres", autocommit=True
        )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pgvector.psycopg.register_vector(conn)
        # JIT compilation of the distance expression adds per-query latency that
        # swamps the index work at these row counts, and is off in most vector
        # deployments for exactly that reason.
        conn.execute("SET jit = off")
        return conn

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: numpy.ndarray) -> None:
        self._keep_warmup_sample(X)
        dim = int(X.shape[1])
        self._dim = dim
        cur = self._cur

        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cur.execute(
            f"CREATE TABLE {TABLE} (id int PRIMARY KEY, tag int NOT NULL, "
            f"embedding vector({dim}))"
        )
        # Keep vectors inline rather than TOASTed, as pgvector's own docs advise
        # for benchmark-sized vectors; TOAST would add a detoast per comparison.
        cur.execute(f"ALTER TABLE {TABLE} ALTER COLUMN embedding SET STORAGE PLAIN")

        if self._build_mode == "incremental":
            print("[vb] creating index BEFORE load (incremental mode)", file=sys.stderr)
            build_start = time.time()
            self._create_index()
            self._build_seconds = time.time() - build_start
            self._load_seconds = self._copy_rows(X)
            # In incremental mode the graph is maintained during the load, so the
            # honest build cost is the load time plus the empty-index creation.
            self._build_seconds += self._load_seconds
        else:
            self._load_seconds = self._copy_rows(X)
            print("[vb] creating index AFTER load (post mode)", file=sys.stderr)
            build_start = time.time()
            self._create_index()
            self._build_seconds = time.time() - build_start

        cur.execute(f"ANALYZE {TABLE}")
        self._index_bytes = self._measure_index_bytes()
        print(
            f"[vb] load {self._load_seconds:.1f}s, build {self._build_seconds:.1f}s, "
            f"index {self._index_bytes:,} bytes",
            file=sys.stderr,
        )

    def _copy_rows(self, X: numpy.ndarray) -> float:
        print(f"[vb] copying {len(X):,} x {X.shape[1]} vectors", file=sys.stderr)
        start = time.time()
        with self._cur.copy(f"COPY {TABLE} (id, tag, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, i % 100, embedding))
        elapsed = time.time() - start
        print(
            f"[vb] copy complete in {elapsed:.1f}s "
            f"({len(X) / max(elapsed, 1e-9):,.0f} rows/s)",
            file=sys.stderr,
        )
        return elapsed

    def _create_index(self) -> None:
        sql = (
            f"CREATE INDEX {INDEX} ON {TABLE} USING hnsw "
            f"(embedding {OPCLASS[self._metric]}) "
            f"WITH (m = {self._m}, ef_construction = {self._ef_construction})"
        )
        print(f"[vb] {sql}", file=sys.stderr)
        self._cur.execute(sql)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_query_arguments(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)
        self._cur.execute(f"SET hnsw.ef_search = {self._ef_search}")
        if self._plan_verified is None:
            self._plan_verified = self._verify_plan()
        self._warm_index_once()   # once per built index; see vb_warmup.py
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Run throwaway queries so the timed ones do not pay for a cold cache.

        Random vectors, not the benchmark's query set: warming with the vectors
        about to be measured would prime the exact graph regions they need.
        """
        if WARMUP_QUERIES <= 0 or not self._dim:
            return
        rng = numpy.random.RandomState(0)
        for _ in range(WARMUP_QUERIES):
            try:
                self.query(rng.normal(size=self._dim).astype(numpy.float32), 10)
            except Exception:
                return

    def _verify_plan(self, k: int = 10) -> bool:
        """Confirm the planner picked the HNSW index rather than a seq scan.

        The same trap as on the MySQL side: a sequential scan gives exact results
        at terrible speed, which reads as "accurate but slow" instead of "the
        index was not used".
        """
        try:
            self._cur.execute(f"SELECT embedding FROM {TABLE} LIMIT 1")
            row = self._cur.fetchone()
            if row is None:
                return False
            self._cur.execute("EXPLAIN (FORMAT TEXT) " + self._query_sql, (row[0], k))
            plan = " ".join(str(r[0]) for r in self._cur.fetchall())
        except Exception as exc:  # pragma: no cover
            print(f"[vb] WARNING: could not EXPLAIN the query plan: {exc}", file=sys.stderr)
            return False

        used = INDEX in plan or "Index Scan" in plan
        if used:
            print("[vb] plan check OK: HNSW index in use", file=sys.stderr)
        else:
            print(
                "[vb] WARNING: the HNSW index is NOT in the query plan. "
                f"This measures a sequential scan, not ANN search.\n[vb] plan: {plan}",
                file=sys.stderr,
            )
        return used

    def query(self, v, n: int) -> List[int]:
        self._cur.execute(self._query_sql, (v, n), binary=True, prepare=True)
        return [row[0] for row in self._cur.fetchall()]

    def batch_query(self, X, n: int) -> None:
        from multiprocessing.pool import ThreadPool

        try:
            threads = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover
            threads = os.cpu_count() or 1
        threads = max(1, min(8, threads))

        def worker(chunk):
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"SET hnsw.ef_search = {self._ef_search}")
            out = []
            for v in chunk:
                cur.execute(self._query_sql, (v, n), binary=True, prepare=True)
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

    def get_batch_results(self):
        return self._batch_results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _measure_index_bytes(self) -> int:
        try:
            self._cur.execute("SELECT pg_relation_size(%s)", (INDEX,))
            return int(self._cur.fetchone()[0])
        except Exception:  # pragma: no cover
            return 0

    def get_memory_usage(self) -> float:
        return self._index_bytes / 1024.0

    def get_additional(self) -> Dict[str, Any]:
        return {
            "engine": "pgvector",
            "resource_pass": os.environ.get("VB_RESOURCE_PASS", "unknown"),
            "engine_version": self._server_version(),
            "storage_engine": "heap",
            "M": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "metric": self._metric,
            "build_mode": self._build_mode,
            "index_bytes": self._index_bytes,
            "load_seconds": round(self._load_seconds, 3),
            "build_seconds": round(self._build_seconds, 3),
            "vector_index_used": self._plan_verified,
            "march": self._image_march(),
            **self._warmup_attrs(),
        }

    def _server_version(self) -> str:
        try:
            self._cur.execute(
                "SELECT version() || ' / pgvector ' || "
                "(SELECT extversion FROM pg_extension WHERE extname='vector')"
            )
            return str(self._cur.fetchone()[0])
        except Exception:  # pragma: no cover
            return "unknown"

    @staticmethod
    def _image_march() -> str:
        try:
            with open("/opt/pgvector-artifacts/.march") as fh:
                return fh.read().strip()
        except OSError:
            return "unknown"

    def __str__(self) -> str:
        return (
            f"PGVector(m={self._m}, ef_construction={self._ef_construction}, "
            f"build={self._build_mode}, ef_search={self._ef_search})"
        )

    def done(self) -> None:
        try:
            self._cur.close()
            self._conn.close()
        except Exception:
            pass
        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(SERVER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._server.kill()
            self._server = None
