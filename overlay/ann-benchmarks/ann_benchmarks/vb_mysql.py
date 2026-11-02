"""Shared ann-benchmarks driver for the MySQL-family vector engines.

MariaDB (MHNSW) and AliSQL (VIDX) are close enough that one driver serves both:
same wire protocol, same client library, same `VECTOR(N)` column type, same
"raw little-endian float32 binary string" value encoding. They differ in which
session variables enable and tune the index, and in whether the optimizer can
silently decline to use it.

Keeping them on one code path is deliberate. If MariaDB were driven through
Connector/C and AliSQL through some other client, part of any measured
difference would belong to the clients rather than to the servers.

Subclasses supply a `Dialect`; everything else is shared.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Sequence

import numpy

from .algorithms.base.module import BaseANN
from .vb_warmup import IndexWarmupMixin

try:  # pragma: no cover - exercised only inside the bench images
    import mariadb as _connector
except ImportError:  # pragma: no cover
    _connector = None


SERVER_START_TIMEOUT_S = int(os.environ.get("VB_SERVER_START_TIMEOUT", "120"))
SERVER_STOP_TIMEOUT_S = int(os.environ.get("VB_SERVER_STOP_TIMEOUT", "300"))
ENTRYPOINT = os.environ.get("VB_ENTRYPOINT", "/usr/local/bin/vb-entrypoint")
# How often a long load reports progress, in seconds.
PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))
# Rows per executemany. The ops harness uses the same value; matching them
# keeps the two measurement paths comparable on ingest.
INSERT_BATCH = int(os.environ.get("VB_INSERT_BATCH", "500"))

TABLE = "t1"
DATABASE = "ann"


def to_binary_f32(vector) -> bytes:
    """Encode a vector the way both engines store it: packed little-endian f32.

    MariaDB's VECTOR and AliSQL's Field_vector both accept a binary string of
    exactly 4*dim bytes. Using it avoids the text round-trip that
    VEC_FROMTEXT('[...]') would impose on AliSQL only, which would otherwise
    show up as an AliSQL ingest penalty that belongs to the client, not the
    server.
    """
    return numpy.asarray(vector, dtype="<f4").tobytes()



# Queries run before each configuration is timed. The first measured point
# otherwise pays for a cold graph cache and lands slower than the next one,
# which inverts the low end of the recall/QPS curve: on a 990k x 1536 corpus
# both MySQL-family engines reported ef_search=10 as slower than ef_search=20,
# which cannot happen once the cache is warm.
#
# Thirty was not enough: full-20260825 still measured ef_search 10 below 20 on
# every MySQL-family curve (1.69x-2.13x). The one-time index warm-up in
# vb_warmup.py now does the heavy warming; this per-configuration top-up stays.
WARMUP_QUERIES = int(os.environ.get("VB_WARMUP_QUERIES", "30"))


@dataclass
class Dialect:
    """Everything that differs between MariaDB and AliSQL."""

    name: str
    # Statements run once on the first connection, before any DDL.
    global_setup: Sequence[str] = field(default_factory=tuple)
    # Statements run on every connection.
    session_setup: Sequence[str] = field(default_factory=tuple)
    # SET statement template for the search-width knob.
    set_ef_search: str = ""
    # Index name used in DDL, and the marker that must appear in EXPLAIN.
    index_name: str = "vi"
    # Glob patterns (relative to the database directory) matching index storage.
    index_file_globs: Sequence[str] = field(default_factory=tuple)
    # Metric name as it appears in DDL and in the distance function suffix.
    metric_names: Dict[str, str] = field(default_factory=dict)
    # Whether this engine's optimizer may silently fall back to a full scan.
    verify_index_used: bool = False
    force_index_hint: str = ""

    def create_table(self, dim: int, m: int, metric: str, storage_engine: str) -> str:
        return (
            f"CREATE TABLE {TABLE} (\n"
            f"  id INT PRIMARY KEY,\n"
            f"  tag INT NOT NULL,\n"
            f"  v VECTOR({dim}) NOT NULL,\n"
            f"  VECTOR INDEX {self.index_name} (v) M={m} DISTANCE={metric}\n"
            f") ENGINE={storage_engine}"
        )


class VBMySQLBase(IndexWarmupMixin, BaseANN):
    """Base ann-benchmarks algorithm for MariaDB and AliSQL."""

    dialect: Dialect  # set by subclasses

    # ------------------------------------------------------------------
    # Construction / server lifecycle
    # ------------------------------------------------------------------

    def __init__(self, metric: str, method_param: Dict[str, Any]):
        if _connector is None:
            raise RuntimeError(
                "the 'mariadb' Python connector is not installed in this image; "
                "the *-bench image is required to run this module"
            )

        self._m = int(method_param["M"])
        self._storage_engine = method_param.get("engine", "InnoDB")
        self._ef_search: Optional[int] = None
        self._index_bytes = 0
        self._build_seconds = 0.0
        self._ingest_seconds = 0.0
        self._plan_verified: Optional[bool] = None
        self._batch_results: List[List[int]] = []

        try:
            self._metric = self.dialect.metric_names[metric]
        except KeyError:
            raise RuntimeError(f"unsupported metric for {self.dialect.name}: {metric}")
        self._metric_fn = self._metric.lower()

        self._socket = os.environ.get("VB_SOCKET") or self._default_socket()
        self._data_dir = os.environ.get("VB_DATA_DIR", "/var/lib/vbench/data")
        self._insert_threads = self._resolve_insert_threads()

        self._server = None
        self._start_server()
        self._conn = self._connect()
        self._cur = self._conn.cursor()
        self._apply_global_setup(self._cur)

    @classmethod
    def engine_name(cls) -> str:
        """The engine this class represents, which is not always the dialect.

        MariaDB123 reuses MariaDB's dialect so the two versions cannot drift
        apart in configuration, but they are separate engines. Reporting
        dialect.name labelled every 12.3 result as "mariadb".
        """
        return getattr(cls, "vb_engine", None) or cls.dialect.name

    def _default_socket(self) -> str:
        return f"/var/run/vbench/{self.dialect.name}.sock"

    @staticmethod
    def _resolve_insert_threads() -> int:
        """Threads used to load the training set.

        Ingest speed is not what this pass measures — the ops harness measures it
        properly and under controlled conditions. Here we only need the load to
        finish in reasonable time, so parallelism is allowed but always recorded
        in get_additional() so a reader can tell how a table was populated.
        """
        env = os.environ.get("VB_INSERT_THREADS", "0")
        try:
            requested = int(env)
        except ValueError:
            requested = 0
        if requested > 0:
            return requested
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            available = os.cpu_count() or 1
        return max(1, min(8, available))

    def _start_server(self) -> None:
        """Start the engine through the image entrypoint.

        Reusing vb-entrypoint rather than re-implementing startup here means the
        benchmark and the by-hand instructions in docs/03-running-manually.md
        exercise exactly the same initialisation and the same flags.
        """
        if not os.path.exists(ENTRYPOINT):
            raise RuntimeError(f"entrypoint not found at {ENTRYPOINT}")

        # Remove a stale socket so the readiness check cannot pass instantly
        # against a socket left behind by a previous, dead server.
        try:
            os.unlink(self._socket)
        except FileNotFoundError:
            pass
        except OSError:
            pass

        print(f"[vb] starting {self.dialect.name}: {ENTRYPOINT} server", file=sys.stderr)
        print(f"[vb] VB_SERVER_ARGS={os.environ.get('VB_SERVER_ARGS', '')}", file=sys.stderr)
        self._server = subprocess.Popen(
            [ENTRYPOINT, "server"], stdout=sys.stderr, stderr=sys.stderr
        )

        deadline = time.time() + SERVER_START_TIMEOUT_S
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"{self.dialect.name} server exited during startup "
                    f"with code {self._server.returncode}"
                )
            if os.path.exists(self._socket):
                try:
                    probe = _connector.connect(**self._conn_kwargs())
                    probe.close()
                    print(f"[vb] {self.dialect.name} is up", file=sys.stderr)
                    return
                except _connector.Error:
                    # The socket appears before --init-file has finished creating
                    # the bench account, so auth failures here are expected until
                    # startup completes.
                    pass
            time.sleep(0.5)
        raise TimeoutError(
            f"{self.dialect.name} did not become ready within {SERVER_START_TIMEOUT_S}s "
            f"(socket {self._socket})"
        )

    def _conn_kwargs(self) -> Dict[str, Any]:
        """Connection parameters.

        The server runs with grants enabled (see the entrypoint: --skip-grant-tables
        would implicitly disable networking on MySQL 8), so a real account is used
        even over the unix socket. Identical for both MySQL-family engines.
        """
        return {
            "unix_socket": self._socket,
            "user": os.environ.get("VB_DB_USER", "bench"),
            "password": os.environ.get("VB_DB_PASSWORD", "bench"),
            "autocommit": True,
        }

    def _connect(self):
        conn = _connector.connect(**self._conn_kwargs())
        cur = conn.cursor()
        for stmt in self.dialect.session_setup:
            cur.execute(stmt)
        cur.close()
        return conn

    def _apply_global_setup(self, cur) -> None:
        for stmt in self.dialect.global_setup:
            cur.execute(stmt)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: numpy.ndarray) -> None:
        self._keep_warmup_sample(X)
        dim = int(X.shape[1])
        cur = self._cur

        cur.execute(f"DROP DATABASE IF EXISTS {DATABASE}")
        cur.execute(f"CREATE DATABASE {DATABASE}")
        cur.execute(f"USE {DATABASE}")
        for stmt in self.dialect.session_setup:
            cur.execute(stmt)

        ddl = self.dialect.create_table(dim, self._m, self._metric, self._storage_engine)
        print(f"[vb] {ddl}", file=sys.stderr)
        cur.execute(ddl)

        print(
            f"[vb] loading {len(X):,} x {dim} vectors "
            f"with {self._insert_threads} thread(s)",
            file=sys.stderr,
        )
        start = time.time()
        if self._insert_threads > 1:
            self._insert_parallel(X)
        else:
            self._insert_serial(cur, X)
        self._ingest_seconds = time.time() - start
        print(
            f"[vb] load complete in {self._ingest_seconds:.1f}s "
            f"({len(X) / max(self._ingest_seconds, 1e-9):,.0f} rows/s)",
            file=sys.stderr,
        )

        # Both engines maintain the HNSW graph incrementally on INSERT, so there
        # is no separate build step to time here. Build cost is measured
        # properly by the ops harness, which can separate load from build.
        self._build_seconds = self._ingest_seconds
        self._index_bytes = self._measure_index_bytes()
        print(
            f"[vb] index storage: {self._index_bytes:,} bytes", file=sys.stderr
        )

    def _insert_serial(self, cur, X: numpy.ndarray, offset: int = 0,
                       label: str = "") -> None:
        """Load rows in batches inside explicit transactions.

        Batching is not an optimisation here, it is the difference between a
        usable load and an unusable one. One `execute` per row costs a network
        round-trip and — with autocommit on — a transaction commit per row,
        which on a 1.2M-row dataset measured about 5 rows/s per thread against
        767 rows/s for the batched path the ops harness uses. Same server, same
        client, same data; the loop was the whole difference.
        """
        sql = f"INSERT INTO {TABLE} (id, tag, v) VALUES (%s, %s, %s)"
        total = len(X)
        started = time.time()
        next_report = started + PROGRESS_INTERVAL_S
        rows: List[Any] = []

        for i, embedding in enumerate(X):
            idx = offset + i
            rows.append((idx, idx % 100, to_binary_f32(embedding)))
            if len(rows) >= INSERT_BATCH:
                cur.executemany(sql, rows)
                cur.execute("COMMIT")
                rows = []
                now = time.time()
                if now >= next_report:
                    rate = (i + 1) / max(now - started, 1e-9)
                    eta = (total - i - 1) / max(rate, 1e-9)
                    print(
                        f"[vb] {label}{i + 1:,}/{total:,} rows, {rate:,.0f} rows/s, "
                        f"ETA {eta / 60:.1f} min",
                        file=sys.stderr,
                    )
                    next_report = now + PROGRESS_INTERVAL_S
        if rows:
            cur.executemany(sql, rows)
        cur.execute("COMMIT")

    def _insert_parallel(self, X: numpy.ndarray) -> None:
        total = len(X)
        threads = self._insert_threads
        bounds = [(total * i // threads, total * (i + 1) // threads) for i in range(threads)]

        def worker(bound):
            lo, hi = bound
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"USE {DATABASE}")
            # Same batched path as the serial loader; see _insert_serial for
            # why per-row inserts are not viable at this scale.
            self._insert_serial(cur, X[lo:hi], offset=lo,
                                label="shard 0: " if lo == 0 else "")
            cur.close()
            conn.close()

        pool = ThreadPool(threads)
        try:
            pool.map(worker, bounds)
        finally:
            pool.close()
            pool.join()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_query_arguments(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)
        self._cur.execute(self.dialect.set_ef_search.format(ef_search=self._ef_search))
        # Verify once per configuration that the optimizer is actually using the
        # vector index. AliSQL costs the index against a full scan and will pick
        # the scan for large LIMITs; that yields perfect recall at terrible QPS
        # and would quietly poison the comparison.
        if self.dialect.verify_index_used and self._plan_verified is None:
            self._plan_verified = self._verify_plan()
        # Once per built index, before the first timed configuration; see
        # vb_warmup.py for the cold first point that made it necessary.
        self._warm_index_once()
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Run throwaway queries so the timed ones do not pay for a cold cache.

        Random vectors rather than the benchmark's own query set, which the
        module never sees and must not: warming with the vectors about to be
        measured would prime exactly the graph regions those queries need and
        flatter the result.
        """
        if WARMUP_QUERIES <= 0:
            return
        try:
            dim = self._dim_from_table()
        except Exception:
            return
        rng = numpy.random.RandomState(0)
        for _ in range(WARMUP_QUERIES):
            probe = to_binary_f32(rng.normal(size=dim).astype("<f4"))
            try:
                self._cur.execute(self._query_sql(10), (probe,))
                self._cur.fetchall()
            except Exception:
                return

    def _query_sql(self, k: int, force_index: bool = False) -> str:
        hint = f" {self.dialect.force_index_hint}" if force_index else ""
        return (
            f"SELECT id FROM {TABLE}{hint} "
            f"ORDER BY vec_distance_{self._metric_fn}(v, %s) LIMIT {k}"
        )

    def _verify_plan(self, k: int = 10) -> bool:
        probe = to_binary_f32(numpy.zeros(self._dim_from_table(), dtype="<f4"))
        try:
            self._cur.execute("EXPLAIN " + self._query_sql(k), (probe,))
            rows = self._cur.fetchall()
        except Exception as exc:  # pragma: no cover - depends on server build
            print(f"[vb] WARNING: could not EXPLAIN the query plan: {exc}", file=sys.stderr)
            return False

        plan = " | ".join(str(col) for row in rows for col in row)
        used = self.dialect.index_name in plan
        if used:
            print(f"[vb] plan check OK: vector index in use ({plan})", file=sys.stderr)
        else:
            print(
                "[vb] WARNING: the vector index is NOT in the query plan. "
                "This measures a brute-force scan, not ANN search.\n"
                f"[vb] plan: {plan}",
                file=sys.stderr,
            )
        return used

    def _dim_from_table(self) -> int:
        self._cur.execute(
            "SELECT CHARACTER_OCTET_LENGTH FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME='v'",
            (DATABASE, TABLE),
        )
        row = self._cur.fetchone()
        # VECTOR(N) is stored as 4*N bytes.
        return int(row[0]) // 4 if row and row[0] else 1

    def query(self, v, n: int) -> List[int]:
        self._cur.execute(self._query_sql(n), (to_binary_f32(v),))
        return [row[0] for row in self._cur.fetchall()]

    def batch_query(self, X, n: int) -> None:
        threads = self._insert_threads

        def worker(chunk):
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"USE {DATABASE}")
            cur.execute(self.dialect.set_ef_search.format(ef_search=self._ef_search))
            sql = self._query_sql(n)
            out = []
            for v in chunk:
                cur.execute(sql, (to_binary_f32(v),))
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

        # Reassemble in the original order: chunk c holds indices c, c+t, c+2t…
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
        """Total on-disk size of the vector index structures.

        Both engines keep the HNSW graph in a companion table rather than in an
        index segment, so this is a file-glob measurement against the database
        directory rather than an information_schema lookup.
        """
        db_dir = os.path.join(self._data_dir, DATABASE)
        total = 0
        seen = set()
        for pattern in self.dialect.index_file_globs:
            for path in glob.glob(os.path.join(db_dir, pattern)):
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    total += os.stat(real).st_size
                except OSError:
                    pass
        if total == 0:
            print(
                f"[vb] WARNING: no index files matched "
                f"{self.dialect.index_file_globs} under {db_dir}; "
                "index size will be reported as 0",
                file=sys.stderr,
            )
        return total

    def get_memory_usage(self) -> float:
        # ann-benchmarks reports this in kilobytes and plots it as index size.
        return self._index_bytes / 1024.0

    def get_additional(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name(),
            "resource_pass": os.environ.get("VB_RESOURCE_PASS", "unknown"),
            "engine_version": self._server_version(),
            "storage_engine": self._storage_engine,
            "M": self._m,
            "ef_search": self._ef_search,
            "metric": self._metric,
            "index_bytes": self._index_bytes,
            "ingest_seconds": round(self._ingest_seconds, 3),
            "insert_threads": self._insert_threads,
            "vector_index_used": self._plan_verified,
            "march": self._image_march(),
            **self._warmup_attrs(),
        }

    def _server_version(self) -> str:
        try:
            self._cur.execute("SELECT VERSION()")
            return str(self._cur.fetchone()[0])
        except Exception:  # pragma: no cover
            return "unknown"

    @staticmethod
    def _image_march() -> str:
        """Which -march the server binary in this image was compiled with.

        Every MySQL-family image this base class serves is listed here, and a
        new one belongs in this tuple rather than behind a per-engine
        override. `percona/module.py` carried such an override for exactly one
        release of this file and it is gone: there is a SECOND reader on the
        ops path (`harness/main.py:_detect_march`) in a container that mounts
        no overlay, so an override could only ever fix half of it and the half
        it could not fix is the one a long run depends on. Two readers means
        two chances to miss an engine, and the next engine added should find
        both tuples rather than discover the second one from a blank field.

        A missing path returns "unknown", which is worse than a crash: it
        lands in a record field a reader takes for a measurement, and `march`
        is the field that would catch the mixed build "never mix -march
        between engines" exists to prevent. `/opt/percona/.march` is the one
        that was missing, and the one whose absence nothing noticed.
        """
        for path in ("/opt/mariadb/.march", "/opt/alisql/.march",
                     "/opt/percona/.march", "/opt/myvector/.march",
                     "/opt/villagesql/.march"):
            try:
                with open(path) as fh:
                    return fh.read().strip()
            except OSError:
                continue
        return "unknown"

    def __str__(self) -> str:
        return (
            f"{self.dialect.name.capitalize()}"
            f"(M={self._m}, engine={self._storage_engine}, ef_search={self._ef_search})"
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def done(self) -> None:
        try:
            if self._cur is not None:
                self._cur.execute("SHUTDOWN")
        except Exception:
            pass
        if self._server is not None:
            try:
                self._server.wait(SERVER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # pragma: no cover
                print("[vb] server did not shut down cleanly; terminating", file=sys.stderr)
                self._server.terminate()
                try:
                    self._server.wait(30)
                except subprocess.TimeoutExpired:
                    self._server.kill()
            self._server = None
