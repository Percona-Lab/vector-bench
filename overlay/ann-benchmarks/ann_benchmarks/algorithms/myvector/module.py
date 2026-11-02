"""MyVector (askdba/myvector v1.26.9, MySQL plugin, hnswlib), driven by
ann-benchmarks.

New module; upstream ann-benchmarks has no MyVector support. The ops-harness
driver, `harness/drivers/myvector.py`, carries the evidence for each behaviour
relied on here, cited by symbol against the pinned plugin tree; this module
follows it rather than restating it, and deliberately shares none of its code
because the bench container mounts `overlay/` and the harness container mounts
`harness/`, and neither can import the other.

What differs from `VBMySQLBase`, all of it forced by the plugin:

* The index is declared in a column COMMENT, sized there (hnswlib's capacity is
  fixed at construction), and built by `CALL mysql.myvector_index_build(...)`
  after the load. The build reports failure as a result row, so its text is
  checked, and the effective settings are read back with
  `myvector_index_status` because unknown option spellings fall back silently.
* The query is `WHERE MYVECTOR_IS_ANN(...)`, which an audit plugin rewrites
  before parsing. The vector is NOT inlined: a 784-dim vector literal inside
  the call (a 6.4 KB statement) kills mysqld on this build, so it is set into
  `@vb_q` by a statement the rewrite never touches and the search names the
  variable -- two round trips per query (`MyVectorDriver._bind_query_vector`
  has the evidence). `ef_search` is an option of each call, not a session
  variable.
* `build_seconds` is the build CALL alone and `ingest_seconds` the load alone,
  the split pgvector's module reports, because this engine has the same
  load-then-build shape.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional

import numpy

from ...vb_mysql import DATABASE, TABLE, Dialect, VBMySQLBase

VECTOR_COLUMN = f"{DATABASE}.{TABLE}.v"
DEFAULT_EF_CONSTRUCTION = 200
#: The exact spellings `HNSWMemoryIndex::HNSWMemoryIndex` matches.
METRIC_NAMES = {"angular": "Cosine", "euclidean": "L2"}
#: Must equal `MyVectorDriver.DISTANCE_IMPL`; tests/test_e23 pins the two.
DISTANCE_IMPL = {"Cosine": "scalar (computeCosineDistance)",
                 "L2": "simd (hnswlib L2Space)"}
#: hnswlib's capacity. ann-benchmarks never deletes or re-inserts, so a small
#: margin over the corpus is enough here (the ops driver needs more for churn).
CAPACITY_MARGIN = 1000
INDEX_DIR = os.environ.get("VB_MYVECTOR_INDEX_DIR", "/var/lib/vbench/myvector-index")
_STATUS_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z .]*?)\s*[:=]\s*(.*?)\s*$")


def hex_literal(vector) -> str:
    return "X'" + numpy.asarray(vector, dtype="<f4").tobytes().hex() + "'"


def _text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


@dataclasses.dataclass
class MyVectorDialect(Dialect):
    ef_construction: int = DEFAULT_EF_CONSTRUCTION
    capacity: int = 0

    def create_table(self, dim: int, m: int, metric: str,
                     storage_engine: str) -> str:
        options = (f"type=HNSW,dim={int(dim)},size={int(self.capacity)},"
                   f"M={int(m)},ef={int(self.ef_construction)},dist={metric}")
        return (
            f"CREATE TABLE {TABLE} (\n"
            f"  id INT PRIMARY KEY,\n"
            f"  tag INT NOT NULL,\n"
            f"  v VECTOR({dim}) NOT NULL COMMENT 'MYVECTOR Column |{options}'\n"
            f") ENGINE={storage_engine}"
        )


MYVECTOR = MyVectorDialect(
    name="myvector",
    global_setup=(),
    session_setup=("SET SESSION default_storage_engine = InnoDB",),
    set_ef_search="",
    index_name="myvector_ann",
    index_file_globs=(),
    metric_names=METRIC_NAMES,
    verify_index_used=True,
)


class MyVector(VBMySQLBase):
    dialect = MYVECTOR
    vb_engine = "myvector"

    def __init__(self, metric: str, method_param: Dict[str, Any]):
        self._ef_construction = int(method_param.get("efConstruction",
                                                     DEFAULT_EF_CONSTRUCTION))
        self.dialect = dataclasses.replace(
            MYVECTOR, ef_construction=self._ef_construction)
        self._status: Dict[str, str] = {}
        super().__init__(metric, method_param)

    # -- build ---------------------------------------------------------

    def _call(self, sql: str) -> List[tuple]:
        self._cur.execute(sql)
        rows: List[tuple] = []
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
                return rows

    def fit(self, X: numpy.ndarray) -> None:
        self.dialect = dataclasses.replace(
            self.dialect, capacity=int(len(X)) + CAPACITY_MARGIN)
        super().fit(X)          # CREATE TABLE + load; times the load only
        sql = f"CALL mysql.myvector_index_build('{VECTOR_COLUMN}', 'id')"
        print(f"[vb] {sql}", file=sys.stderr)
        started = time.time()
        result = " | ".join(_text(c) for r in self._call(sql) for c in r)
        self._build_seconds = time.time() - started
        print(f"[vb] build returned in {self._build_seconds:.1f}s: {result}",
              file=sys.stderr)
        if not result.startswith("SUCCESS"):
            raise RuntimeError(f"myvector_index_build did not succeed: {result!r}")
        self._verify_status(len(X), int(X.shape[1]))
        self._index_bytes = self._measure_index_bytes()

    def _verify_status(self, rows: int, dim: int) -> None:
        text = "\n".join(_text(c) for r in self._call(
            f"CALL mysql.myvector_index_status('{VECTOR_COLUMN}')") for c in r)
        status: Dict[str, str] = {}
        for line in text.splitlines():
            match = _STATUS_LINE.match(line)
            if match:
                status[match.group(1).strip()] = match.group(2).strip()
        self._status = status
        want = {"Type": "HNSW", "M": str(self._m), "Dimension": str(dim),
                "Distance": self._metric, "Current Rows": str(rows)}
        wrong = {k: (status.get(k), v) for k, v in want.items()
                 if status.get(k) != v}
        if wrong:
            raise RuntimeError(f"MyVector built a different index than declared "
                               f"(effective, declared): {wrong}")
        print(f"[vb] index status verified: {status}", file=sys.stderr)

    def _measure_index_bytes(self) -> int:
        total = 0
        try:
            for name in os.listdir(INDEX_DIR):
                if name.startswith(f"{VECTOR_COLUMN}.hnsw.index"):
                    total += os.stat(os.path.join(INDEX_DIR, name)).st_size
        except OSError:
            pass
        return total

    # -- query ---------------------------------------------------------

    @staticmethod
    def _bind(cur, v) -> None:
        cur.execute(f"SET @vb_q = {hex_literal(v)}")

    def _ann_sql(self, v, k: int) -> str:
        return (f"SELECT id FROM {TABLE} WHERE MYVECTOR_IS_ANN("
                f"'{VECTOR_COLUMN}','id',@vb_q,"
                f"'nn={int(k)},ef_search={int(self._ef_search)}') "
                f"ORDER BY myvector_row_distance(id) LIMIT {int(k)}")

    def set_query_arguments(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)
        if self._plan_verified is None:
            self._plan_verified = self._verify_plan()
        self._warm_index_once()   # once per built index; see vb_warmup.py
        self._warm_cache()

    def _warm_cache(self) -> None:
        from ...vb_mysql import WARMUP_QUERIES
        rng = numpy.random.RandomState(0)
        dim = int(self._status.get("Dimension", "0") or 0)
        if WARMUP_QUERIES <= 0 or dim <= 0:
            return
        for _ in range(WARMUP_QUERIES):
            try:
                self._bind(self._cur, rng.normal(size=dim))
                self._cur.execute(self._ann_sql(None, 10))
                self._cur.fetchall()
            except Exception:
                return

    def _verify_plan(self, k: int = 10) -> bool:
        dim = int(self._status.get("Dimension", "1") or 1)
        try:
            self._bind(self._cur, numpy.ones(dim))
            self._cur.execute("EXPLAIN FORMAT=TRADITIONAL "
                              + self._ann_sql(None, k))
            rows = self._cur.fetchall()
        except Exception as exc:
            print(f"[vb] WARNING: could not EXPLAIN the query plan: {exc}",
                  file=sys.stderr)
            return False
        used = any(len(r) > 2 and str(r[2]) == "myvector_ann" for r in rows)
        plan = " | ".join(str(c) for r in rows for c in r)
        print(f"[vb] plan check {'OK' if used else 'FAILED'}: {plan}",
              file=sys.stderr)
        return used

    def query(self, v, n: int) -> List[int]:
        self._bind(self._cur, v)
        self._cur.execute(self._ann_sql(v, n))
        return [row[0] for row in self._cur.fetchall()]

    def batch_query(self, X, n: int) -> None:
        threads = self._insert_threads

        def worker(chunk):
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"USE {DATABASE}")
            out = []
            for v in chunk:
                self._bind(cur, v)
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

    # -- reporting -----------------------------------------------------

    def get_additional(self) -> Dict[str, Any]:
        extra = super().get_additional()
        extra["ef_construction"] = self._ef_construction
        extra["build_seconds"] = round(self._build_seconds, 3)
        # A JSON string and not the dict: ann-benchmarks stores every key here
        # as an HDF5 attribute, and a dict has no HDF5 type -- the first smoke
        # run lost a finished query sweep to exactly that TypeError.
        extra["myvector_status"] = json.dumps(self._status, sort_keys=True)
        # The structural asymmetries the report must state, with the same keys
        # and values the ops driver stamps (`MyVectorDriver.query_path_facts`):
        # both statements of every query are inside ann-benchmarks' timer.
        extra["round_trips_per_query"] = 2
        extra["distance_impl"] = DISTANCE_IMPL.get(self._metric, "")
        return extra

    def done(self) -> None:
        """SHUTDOWN, then stop the process once the server says it is done.

        **With the plugin loaded, mysqld does not exit after a clean
        shutdown.** It writes `Shutdown complete` to its error log and then
        stays alive indefinitely with one thread left -- observed on this
        image, still running 120 s after the line. The plugin's
        online-update queue consumers (`vector_q_thread_fn`) are detached
        threads blocked forever in `EventsQ::dequeue`, which is the likely
        reason. The shared `done()` waits `SERVER_STOP_TIMEOUT_S` (300 s) for
        an exit that never comes and then terminates it, so every MyVector
        configuration paid five idle minutes in the first smoke run.

        This waits for the line instead: by then InnoDB is shut down and the
        index files are written, so terminating the process loses nothing.
        Without the line it falls back to the shared behaviour exactly.
        """
        from ...vb_mysql import SERVER_STOP_TIMEOUT_S
        log_file = os.environ.get("VB_LOG_FILE", "/var/lib/vbench/myvector.err")
        try:
            offset = os.path.getsize(log_file)
        except OSError:
            offset = 0
        try:
            if self._cur is not None:
                self._cur.execute("SHUTDOWN")
        except Exception:
            pass
        if self._server is None:
            return
        deadline = time.time() + SERVER_STOP_TIMEOUT_S
        while time.time() < deadline and self._server.poll() is None:
            try:
                with open(log_file, errors="replace") as fh:
                    fh.seek(offset)
                    if "Shutdown complete" in fh.read():
                        print("[vb] server logged 'Shutdown complete'; "
                              "stopping the process it leaves behind",
                              file=sys.stderr)
                        break
            except OSError:
                pass
            time.sleep(0.5)
        if self._server.poll() is None:
            self._server.terminate()
            try:
                self._server.wait(30)
            except Exception:
                self._server.kill()
                self._server.wait(30)
        self._server = None

    def __str__(self) -> str:
        return (f"MyVector(M={self._m}, ef_construction={self._ef_construction}, "
                f"ef_search={self._ef_search})")
