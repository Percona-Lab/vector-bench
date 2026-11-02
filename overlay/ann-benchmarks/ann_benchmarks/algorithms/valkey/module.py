"""Valkey + valkey-search HNSW, driven by ann-benchmarks.

The only in-memory engine in the overlay. Nothing is written to disk, so the
usual questions about cache hit ratios and index files do not apply and two
different ones take their place.

  Everything must fit. A disk-backed engine under-provisioned on memory gets
  slower; Valkey either refuses the write or, under any maxmemory-policy except
  noeviction, silently drops keys already stored. A vanished vector is
  indistinguishable from a bad index in the results: recall falls and nothing
  errors. The policy is asserted before the load and evicted_keys is checked
  after it.

  Index size is a memory figure. It is taken as the difference in used_memory
  across index creation, because there is no file to stat and reporting zero
  would read as a measurement we failed to take.

Otherwise it is the closest engine here to pgvector: M and EF_CONSTRUCTION are
set at FT.CREATE and EF_RUNTIME is overridable per query, so build_mode selects
the same two comparisons. Creating the index before the load indexes on write,
which is what MHNSW and VIDX are forced into; creating it after triggers a
backfill, which is pgvector's bulk build.
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
    import valkey as valkey_client
except ImportError:  # pragma: no cover
    try:
        import redis as valkey_client
    except ImportError:
        valkey_client = None

SERVER_START_TIMEOUT_S = int(os.environ.get("VB_SERVER_START_TIMEOUT", "180"))
SERVER_STOP_TIMEOUT_S = int(os.environ.get("VB_SERVER_STOP_TIMEOUT", "120"))
ENTRYPOINT = os.environ.get("VB_ENTRYPOINT", "/usr/local/bin/vb-entrypoint")

PORT = int(os.environ.get("VB_VALKEY_PORT", "6379"))
INDEX = "idx"
PREFIX = "v:"
VECTOR_FIELD = "embedding"
TAG_FIELD = "tag"
DIALECT = 2

WRITE_BATCH = int(os.environ.get("VB_VALKEY_WRITE_BATCH", "1000"))
BACKFILL_POLL_S = float(os.environ.get("VB_VALKEY_POLL_INTERVAL", "2"))
BACKFILL_TIMEOUT_S = float(os.environ.get("VB_VALKEY_BACKFILL_TIMEOUT", "43200"))

METRIC = {"angular": "COSINE", "euclidean": "L2"}

# Same reason as every other module here: without it the first configuration
# measured pays for a cold state and lands below the second, inverting the low
# end of the curve.
WARMUP_QUERIES = int(os.environ.get("VB_WARMUP_QUERIES", "30"))


def encode_vector(vector) -> bytes:
    """FLOAT32 little-endian, which is what the module expects in PARAMS."""
    return numpy.asarray(vector, dtype="<f4").tobytes()


def _client(**kwargs):
    factory = getattr(valkey_client, "Valkey", None) or valkey_client.Redis
    return factory(host="localhost", port=PORT, **kwargs)



def hdf5_safe(values):
    """Drop what ann-benchmarks cannot store as an HDF5 attribute.

    store_results writes every key of get_additional() straight into
    f.attrs[k]. A None has no HDF5 type, and the failure lands at the very end,
    after the entire sweep has run:

        TypeError: Object dtype dtype('O') has no native HDF5 equivalent

    ann-benchmarks catches it per worker and exits zero, so the phase reports
    completed and writes no results at all. Four runs of Percona Search failed
    that way, in smoke and at full scale alike, because ef_construction is
    genuinely not applicable to mongot and None was the honest thing to put
    there. Absent is the representable way to say the same thing.
    """
    return {k: v for k, v in values.items() if v is not None}


class ValkeySearch(IndexWarmupMixin, BaseANN):
    def __init__(self, metric: str, method_param: Dict[str, Any]):
        if valkey_client is None:
            raise RuntimeError(
                "no valkey/redis client is installed in this image; "
                "the valkey-bench image is required to run this module"
            )
        if metric not in METRIC:
            raise RuntimeError(f"unsupported metric for valkey-search: {metric}")

        self._metric = metric
        self._m = int(method_param["M"])
        self._ef_construction = int(method_param.get("efConstruction", 200))
        self._build_mode = method_param.get("build_mode", "post")
        if self._build_mode not in ("post", "incremental"):
            raise RuntimeError(f"unknown build_mode: {self._build_mode}")

        self._ef_runtime: Optional[int] = None
        self._index_bytes = 0
        self._table_bytes = 0
        self._load_seconds = 0.0
        self._build_seconds = 0.0
        self._backfill_seconds = 0.0
        self._query_verified: Optional[bool] = None
        self._dim: Optional[int] = None
        # A real vector from the corpus. valkey-search tolerates a zero vector
        # under COSINE where Percona Search does not, but a probe with no
        # direction is wrong on an angular corpus either way.
        self._probe: Optional[Any] = None
        self._batch_results: List[List[int]] = []

        self._server = None
        self._start_server()
        self._conn = _client(socket_timeout=600)
        self._assert_no_eviction()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        if not os.path.exists(ENTRYPOINT):
            raise RuntimeError(f"entrypoint not found at {ENTRYPOINT}")
        print(f"[vb] starting valkey: {ENTRYPOINT} server", file=sys.stderr)
        print(f"[vb] VB_SERVER_ARGS={os.environ.get('VB_SERVER_ARGS', '')}", file=sys.stderr)
        self._server = subprocess.Popen(
            [ENTRYPOINT, "server"], stdout=sys.stderr, stderr=sys.stderr
        )

        deadline = time.time() + SERVER_START_TIMEOUT_S
        last_error = None
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"valkey exited during startup with code {self._server.returncode}"
                )
            try:
                conn = _client(socket_connect_timeout=2)
                conn.ping()
                # PING passes before the module has finished loading, and a
                # Valkey without valkey-search takes every write and then fails
                # every query.
                modules = conn.execute_command("MODULE", "LIST")
                if any(b"search" in bytes(str(item), "utf-8").lower()
                       if not isinstance(item, (bytes, bytearray))
                       else b"search" in bytes(item).lower()
                       for entry in modules for item in
                       (entry if isinstance(entry, (list, tuple)) else [entry])):
                    conn.close()
                    print("[vb] valkey is up with the search module loaded",
                          file=sys.stderr)
                    return
                conn.close()
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)
        raise TimeoutError(
            f"valkey did not become ready within {SERVER_START_TIMEOUT_S}s: {last_error}"
        )

    def _assert_no_eviction(self) -> None:
        try:
            policy = self._conn.config_get("maxmemory-policy").get("maxmemory-policy")
            policy = policy.decode() if isinstance(policy, bytes) else policy
        except Exception:
            return
        if policy and policy != "noeviction":
            raise RuntimeError(
                f"maxmemory-policy is {policy!r}, not 'noeviction'. Under any "
                f"other policy Valkey drops keys when it reaches maxmemory, and "
                f"a missing vector reads as a bad index rather than as an error."
            )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: numpy.ndarray) -> None:
        self._keep_warmup_sample(X)
        dim = int(X.shape[1])
        self._dim = dim
        self._probe = X[0]
        self._conn.flushall()

        if self._build_mode == "incremental":
            print("[vb] creating index BEFORE load (incremental mode)", file=sys.stderr)
            before = self._used_memory()
            build_start = time.time()
            self._create_index(dim)
            self._load_seconds = self._write_rows(X)
            self._backfill_seconds = self._wait_for_backfill()
            self._build_seconds = time.time() - build_start
            self._table_bytes = 0
            self._index_bytes = max(max(0, self._used_memory() - before),
                                    self._indexed_memory())
        else:
            self._load_seconds = self._write_rows(X)
            self._table_bytes = self._used_memory()
            print("[vb] creating index AFTER load (post mode)", file=sys.stderr)
            build_start = time.time()
            self._create_index(dim)
            self._backfill_seconds = self._wait_for_backfill()
            self._build_seconds = time.time() - build_start
            self._index_bytes = max(
                max(0, self._used_memory() - self._table_bytes),
                self._indexed_memory())

        self._assert_nothing_evicted()
        print(
            f"[vb] load {self._load_seconds:.1f}s, build {self._build_seconds:.1f}s "
            f"(backfill {self._backfill_seconds:.1f}s), "
            f"index {self._index_bytes:,} bytes resident",
            file=sys.stderr,
        )

    def _write_rows(self, X: numpy.ndarray) -> float:
        print(f"[vb] writing {len(X):,} x {X.shape[1]} vectors", file=sys.stderr)
        start = time.time()
        pipe = self._conn.pipeline(transaction=False)
        for i, embedding in enumerate(X):
            pipe.hset(f"{PREFIX}{i}", mapping={
                TAG_FIELD: i % 100,
                VECTOR_FIELD: encode_vector(embedding),
            })
            if (i + 1) % WRITE_BATCH == 0:
                pipe.execute()
        pipe.execute()
        elapsed = time.time() - start
        print(
            f"[vb] write complete in {elapsed:.1f}s "
            f"({len(X) / max(elapsed, 1e-9):,.0f} rows/s)",
            file=sys.stderr,
        )
        return elapsed

    def _create_index(self, dim: int) -> None:
        args = [
            "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", PREFIX,
            "SCHEMA",
            TAG_FIELD, "NUMERIC",
            VECTOR_FIELD, "VECTOR", "HNSW", "10",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", METRIC[self._metric],
            "M", str(self._m),
            "EF_CONSTRUCTION", str(self._ef_construction),
        ]
        print(f"[vb] {' '.join(args)}", file=sys.stderr)
        self._conn.execute_command(*args)

    def _wait_for_backfill(self) -> float:
        """FT.CREATE returns before the index is complete.

        A query issued straight afterwards searches a partial graph and reports
        a recall that has nothing to do with the index parameters.
        """
        started = time.time()
        last_report = started
        while True:
            done, indexing = self._backfill_progress()
            if done >= 1.0 and not indexing:
                return time.time() - started
            waited = time.time() - started
            if waited > BACKFILL_TIMEOUT_S:
                raise TimeoutError(
                    f"backfill still at {done:.1%} after {waited / 3600:.1f} h")
            if time.time() - last_report >= 30:
                print(f"[vb] backfill {done:.1%}, {waited / 60:.1f} min elapsed",
                      file=sys.stderr)
                last_report = time.time()
            time.sleep(BACKFILL_POLL_S)

    def _backfill_progress(self):
        """Read valkey-search's own field names.

        Guessing these cost a whole smoke run. FT.INFO reports
        `backfill_complete_percent`, `backfill_in_progress` and `state`; there
        is no `percent_indexed` and no `indexing`. Looking for the names that
        do not exist made every default fire at once, so the wait returned
        immediately and the build, the row count, the index size and every
        query afterwards measured an index that was still empty.
        """
        info = self._ft_info()
        if not info:
            # No index yet, or FT.INFO failed. Not "finished".
            return 0.0, True
        try:
            done = float(info.get("backfill_complete_percent", 0.0))
        except (TypeError, ValueError):
            done = 0.0
        in_progress = str(info.get("backfill_in_progress", "1")) not in ("0", "false", "False")
        ready = str(info.get("state", "")).lower() in ("", "ready")
        return done, (in_progress or not ready)

    def _ft_info(self) -> Dict[str, Any]:
        try:
            raw = self._conn.execute_command("FT.INFO", INDEX)
        except Exception:
            return {}
        out: Dict[str, Any] = {}
        for i in range(0, len(raw) - 1, 2):
            key = raw[i].decode() if isinstance(raw[i], bytes) else str(raw[i])
            value = raw[i + 1]
            out[key] = value.decode() if isinstance(value, bytes) else value
        return out

    def _assert_nothing_failed_to_index(self) -> None:
        """A hash that fails to index is not an error anywhere else.

        The document still counts toward num_docs, the write returned OK and
        the query returns fewer neighbours than it should. It reads as poor
        recall rather than as a broken configuration, which is the failure this
        framework exists to catch rather than average into a curve.
        """
        info = self._ft_info()
        try:
            failures = int(info.get("hash_indexing_failures", 0))
        except (TypeError, ValueError):
            return
        if failures:
            raise RuntimeError(
                f"valkey-search failed to index {failures:,} hashes. The most "
                f"likely cause is a vector whose byte length does not match "
                f"DIM x 4. Recall computed against this index would be wrong "
                f"rather than merely low."
            )

    def _indexed_memory(self) -> int:
        """What the module says its own attributes cost, from FT.INFO.

        The used_memory delta across index creation is the obvious measure and
        it is not reliable: on the upstream build the index shows up there, and
        on Percona's it does not, which reported a 66 KB index beside an 82 MB
        table. FT.INFO carries user_indexed_memory per attribute, and it needs
        a nested walk because attributes is a list of lists.
        """
        total = 0

        def walk(node):
            nonlocal total
            if not isinstance(node, (list, tuple)):
                return
            for i in range(0, len(node) - 1, 2):
                key = node[i]
                key = key.decode() if isinstance(key, bytes) else key
                value = node[i + 1]
                if key == "user_indexed_memory":
                    try:
                        total += int(value)
                    except (TypeError, ValueError):
                        pass
                if isinstance(value, (list, tuple)):
                    walk(value)
                    for sub in value:
                        walk(sub)

        try:
            walk(self._conn.execute_command("FT.INFO", INDEX))
        except Exception:
            return 0
        return total

    def _used_memory(self) -> int:
        try:
            return int(self._conn.info("memory").get("used_memory", 0))
        except Exception:
            return 0

    def _assert_nothing_evicted(self) -> None:
        try:
            evicted = int(self._conn.info("stats").get("evicted_keys", 0))
        except Exception:
            return
        if evicted:
            raise RuntimeError(
                f"Valkey evicted {evicted:,} keys during the load. Every recall "
                f"figure from this run would be computed against a dataset that "
                f"is missing rows. Raise the container memory limit."
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_query_arguments(self, ef_runtime: int) -> None:
        self._ef_runtime = int(ef_runtime)
        if self._query_verified is None:
            self._query_verified = self._verify_query()
        self._warm_index_once()   # once per built index; see vb_warmup.py
        self._warm_cache()

    def _search(self, conn, vector, k: int) -> List[int]:
        query = (f"*=>[KNN {k} @{VECTOR_FIELD} $vec "
                 f"EF_RUNTIME {max(int(self._ef_runtime or k), k)}]")
        raw = conn.execute_command(
            "FT.SEARCH", INDEX, query,
            "PARAMS", "2", "vec", encode_vector(vector),
            "LIMIT", "0", str(k),
            # Keys only: returning the vectors would put 6 KB per hit on the
            # wire and measure the transport rather than the index.
            "NOCONTENT",
            "DIALECT", str(DIALECT),
        )
        out = []
        for item in raw[1:]:
            key = item.decode() if isinstance(item, bytes) else str(item)
            if key.startswith(PREFIX):
                out.append(int(key[len(PREFIX):]))
        return out

    def _warm_cache(self) -> None:
        if WARMUP_QUERIES <= 0 or not self._dim:
            return
        rng = numpy.random.RandomState(0)
        for _ in range(WARMUP_QUERIES):
            try:
                self.query(rng.normal(size=self._dim).astype(numpy.float32), 10)
            except Exception:
                return

    def _verify_query(self, k: int = 10) -> bool:
        """FT.SEARCH is only answerable through the index; without one it errors.

        What this separates is an erroring configuration from a genuinely slow
        one, so the two are not both recorded as low throughput.
        """
        try:
            probe = (self._probe if self._probe is not None
                     else numpy.random.RandomState(0).normal(
                         size=self._dim or 1).astype(numpy.float32))
            self._search(self._conn, probe, k)
        except Exception as exc:
            print(f"[vb] WARNING: FT.SEARCH failed: {exc}", file=sys.stderr)
            return False
        print("[vb] query check OK: FT.SEARCH served by the index", file=sys.stderr)
        return True

    def query(self, v, n: int) -> List[int]:
        return self._search(self._conn, v, n)

    def batch_query(self, X, n: int) -> None:
        from multiprocessing.pool import ThreadPool

        try:
            threads = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover
            threads = os.cpu_count() or 1
        threads = max(1, min(8, threads))

        def worker(chunk):
            conn = _client(socket_timeout=600)
            out = [self._search(conn, v, n) for v in chunk]
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

    def get_memory_usage(self) -> float:
        return self._index_bytes / 1024.0

    def get_additional(self) -> Dict[str, Any]:
        return hdf5_safe({
            "engine": "valkey",
            "resource_pass": os.environ.get("VB_RESOURCE_PASS", "unknown"),
            "engine_version": self._server_version(),
            "storage_engine": "memory",
            "M": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_runtime,
            "ef_runtime": self._ef_runtime,
            "metric": self._metric,
            "build_mode": self._build_mode,
            # Resident, not on disk. There is no file to stat.
            "index_bytes": self._index_bytes,
            "table_bytes": self._table_bytes,
            "in_memory_only": True,
            "load_seconds": round(self._load_seconds, 3),
            "build_seconds": round(self._build_seconds, 3),
            "backfill_seconds": round(self._backfill_seconds, 3),
            "vector_index_used": self._query_verified,
            # Percona ships prebuilt packages, so there is no -march of ours to
            # report and claiming one would be false. The installed package
            # versions are the provenance instead.
            "march": "none",
            "packages": self._packages(),
            **self._warmup_attrs(),
        })

    def _server_version(self) -> str:
        try:
            info = self._conn.info("server")
            version = info.get("valkey_version") or info.get("redis_version") or "?"
        except Exception:  # pragma: no cover
            return "unknown"
        return f"Valkey {version} / valkey-search"

    @staticmethod
    def _packages() -> str:
        """Stands in for the tag and commit every other engine records."""
        try:
            with open("/opt/valkey-artifacts/.packages") as fh:
                return " ".join(fh.read().split())
        except OSError:
            return "unknown"

    def __str__(self) -> str:
        return (f"ValkeySearch(m={self._m}, ef_construction={self._ef_construction}, "
                f"build={self._build_mode}, ef_runtime={self._ef_runtime})")

    def done(self) -> None:
        try:
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
