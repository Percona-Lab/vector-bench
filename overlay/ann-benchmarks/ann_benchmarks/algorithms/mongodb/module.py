"""Percona Search for MongoDB (mongot), driven by ann-benchmarks.

The other modules in this overlay talk to a server that owns its own index.
This one talks to two processes: `mongod` holds the documents, and a separate
Lucene process, `mongot`, holds the index and is fed from mongod's change
stream. Four things follow, and each is a place where a benchmark written for
the other engines would silently measure the wrong thing.

  Build is asynchronous. createSearchIndex returns in milliseconds and the
  index is unqueryable until mongot has finished its initial sync, so `fit`
  polls until the index reports READY and counts the wait. The index is created
  after the load, not before: created first, mongot queues an initial sync over
  an empty collection and the index stays PENDING while the writes stream past.
  `build_seconds` is therefore load plus wait, and the mode is a third kind --
  not incremental on INSERT like MHNSW and VIDX, and not a blocking bulk build
  like pgvector.

  Search width is a per-query argument. numCandidates goes inside the
  $vectorSearch stage rather than into a session variable, and it is clamped to
  at least k, because numCandidates below limit cannot return limit rows and
  the shared grid starts at ef_search 10.

  Vectors are BinData float32. As a BSON array of doubles the corpus would cost
  8 bytes a dimension in mongod, twice what the same vectors occupy in Lucene.

  There is no plan to inspect. $vectorSearch has no fallback to a collection
  scan the way a SQL optimiser does: without the index the aggregation errors.
  The check is kept anyway so that an erroring configuration and a genuinely
  slow one are told apart in the record rather than both appearing as low QPS.
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
    import pymongo
    from bson.binary import Binary
except ImportError:  # pragma: no cover
    pymongo = None
    Binary = None

SERVER_START_TIMEOUT_S = int(os.environ.get("VB_SERVER_START_TIMEOUT", "300"))
SERVER_STOP_TIMEOUT_S = int(os.environ.get("VB_SERVER_STOP_TIMEOUT", "300"))
ENTRYPOINT = os.environ.get("VB_ENTRYPOINT", "/usr/local/bin/vb-entrypoint")

DATABASE = os.environ.get("VB_DATABASE", "ann")
COLLECTION = "t1"
INDEX_NAME = "vector_index"
PORT = int(os.environ.get("VB_MONGOD_PORT", "27017"))
# mongot refuses to parse a config without SCRAM or x509, so mongod runs with
# auth on and this client authenticates too. Fixed credentials on an isolated
# container network: not a secret, a requirement mongot imposes.
USER = os.environ.get("VB_MONGO_USER", "bench")
PASSWORD = os.environ.get("VB_MONGO_PASSWORD", "bench")


def _uri() -> str:
    if not USER:
        return f"mongodb://localhost:{PORT}/?directConnection=true"
    return (f"mongodb://{USER}:{PASSWORD}@localhost:{PORT}/"
            f"?directConnection=true&authSource=admin")

READY_POLL_S = float(os.environ.get("VB_MONGOT_POLL_INTERVAL", "2"))
READY_TIMEOUT_S = float(os.environ.get("VB_MONGOT_READY_TIMEOUT", "43200"))
# createSearchIndex returns before the index is listable, so an index that is
# not there yet is normal for the first seconds and a fault only if it lasts.
MISSING_INDEX_GRACE_S = float(os.environ.get("VB_MONGOT_MISSING_GRACE", "60"))
INSERT_BATCH = int(os.environ.get("VB_MONGO_INSERT_BATCH", "1000"))

SIMILARITY = {"angular": "cosine", "euclidean": "euclidean"}

# Same reason as every other module here: the first measured configuration
# otherwise pays for a cold cache and lands below the second, which inverts the
# low end of the curve. It matters more for mongot than for the others, because
# its segments are served from the OS filesystem cache rather than from a cache
# the server manages itself.
WARMUP_QUERIES = int(os.environ.get("VB_WARMUP_QUERIES", "30"))


def encode_vector(vector) -> Any:
    """Pack a float vector as BSON BinData float32 (subtype 9)."""
    values = numpy.asarray(vector, dtype=numpy.float32)
    # Header is the float32 data-type marker followed by a zero padding count.
    return Binary(b"\x27\x00" + values.tobytes(), 9)



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


class PerconaSearch(IndexWarmupMixin, BaseANN):
    def __init__(self, metric: str, method_param: Dict[str, Any]):
        if pymongo is None:
            raise RuntimeError(
                "pymongo is not installed in this image; "
                "the mongodb-bench image is required to run this module"
            )
        if metric not in SIMILARITY:
            raise RuntimeError(f"unsupported metric for Percona Search: {metric}")

        self._metric = metric
        # Recorded rather than applied: mongot does not expose graph degree, so
        # M is carried into the results to say which sweep the point belongs to
        # and never sent to the server. Reporting it as though it were applied
        # would claim a comparison at matched M that is not being made.
        self._m = int(method_param.get("M", 16))
        self._quantization = str(method_param.get("quantization", "none"))
        if self._quantization not in ("none", "scalar", "binary"):
            raise RuntimeError(f"unknown quantization: {self._quantization}")

        self._num_candidates: Optional[int] = None
        self._index_bytes = 0
        self._load_seconds = 0.0
        self._build_seconds = 0.0
        self._ready_seconds = 0.0
        self._query_verified: Optional[bool] = None
        self._dim: Optional[int] = None
        # A real vector from the corpus, kept for the plan check. A zero vector
        # has no direction, so cosine similarity over it is undefined and the
        # server is entitled to reject the query.
        self._probe: Optional[Any] = None
        self._batch_results: List[List[int]] = []

        self._server = None
        self._start_server()
        self._client = self._connect()
        self._coll = self._client[DATABASE][COLLECTION]

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        if not os.path.exists(ENTRYPOINT):
            raise RuntimeError(f"entrypoint not found at {ENTRYPOINT}")
        print(f"[vb] starting mongod + mongot: {ENTRYPOINT} server", file=sys.stderr)
        print(f"[vb] VB_SERVER_ARGS={os.environ.get('VB_SERVER_ARGS', '')}", file=sys.stderr)
        self._server = subprocess.Popen(
            [ENTRYPOINT, "server"], stdout=sys.stderr, stderr=sys.stderr
        )

        deadline = time.time() + SERVER_START_TIMEOUT_S
        last_error = None
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"mongod exited during startup with code {self._server.returncode}"
                )
            try:
                client = pymongo.MongoClient(
                    _uri(), serverSelectionTimeoutMS=2000)
                # Writable primary, not merely reachable: mongod answers a ping
                # while still SECONDARY, and a load that starts before the
                # election completes fails on its first insert.
                if client.admin.command("hello").get("isWritablePrimary"):
                    client.close()
                    print("[vb] mongod is primary", file=sys.stderr)
                    self._wait_for_mongot()
                    return
                client.close()
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)
        raise TimeoutError(
            f"mongod did not become primary within {SERVER_START_TIMEOUT_S}s: {last_error}"
        )

    def _wait_for_mongot(self) -> None:
        """mongod is ready long before mongot is.

        The JVM needs fifteen to twenty seconds to reach its health check, and a
        createSearchIndexes issued inside that window is accepted, creates a
        Lucene index and then sits in PENDING with no initial sync queued. The
        ops path never saw it because the orchestrator spends that long starting
        a separate client container; this one runs in-process and reached the
        index three seconds after mongot's process started.
        """
        import socket

        port = int(os.environ.get("VB_MONGOT_HEALTH_PORT", "8080"))
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=2):
                    print("[vb] mongot health check is answering", file=sys.stderr)
                    return
            except OSError:
                time.sleep(1)
        print("[vb] WARNING: mongot did not answer its health check; "
              "index creation may hang in PENDING", file=sys.stderr)

    def _connect(self):
        return pymongo.MongoClient(_uri(), serverSelectionTimeoutMS=60000)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: numpy.ndarray) -> None:
        self._keep_warmup_sample(X)
        try:
            self._fit(X)
        except Exception:
            # Anything that goes wrong here is otherwise invisible: the phase
            # is reported completed and no result file is written.
            self._dump_mongot_log()
            raise

    def _fit(self, X: numpy.ndarray) -> None:
        dim = int(X.shape[1])
        self._dim = dim
        self._probe = X[0]
        db = self._client[DATABASE]

        db.drop_collection(COLLECTION)
        db.create_collection(COLLECTION)

        # The index is declared AFTER the load, which is the order that works.
        #
        # Declaring it first looks better on paper: mongot would index from the
        # change stream as rows arrive, overlapping the two. What it actually
        # does is queue an initial sync over an empty collection, and the index
        # then sits in PENDING indefinitely while the documents stream past --
        # observed for six minutes on 2,000 rows, with mongot logging
        # "Queued initial syncs, numQueued: 0" and never revisiting it. The ops
        # driver has always created the index after the load and reaches READY
        # in thirty seconds on the same data.
        self._load_seconds = self._insert_rows(X)

        print("[vb] creating search index and waiting for mongot", file=sys.stderr)
        self._create_index(dim)
        self._ready_seconds = self._wait_until_ready()
        # Sequential now rather than overlapping, but still two costs: the
        # write and the indexing of what was written. Neither alone is the cost
        # of having a queryable index.
        self._build_seconds = self._load_seconds + self._ready_seconds

        self._index_bytes = self._measure_index_bytes()
        print(
            f"[vb] load {self._load_seconds:.1f}s, "
            f"index ready {self._ready_seconds:.1f}s after load, "
            f"index {self._index_bytes:,} bytes",
            file=sys.stderr,
        )

    def _insert_rows(self, X: numpy.ndarray) -> float:
        print(f"[vb] inserting {len(X):,} x {X.shape[1]} vectors", file=sys.stderr)
        start = time.time()
        batch = []
        for i, embedding in enumerate(X):
            batch.append({"_id": i, "tag": i % 100,
                          "embedding": encode_vector(embedding)})
            if len(batch) >= INSERT_BATCH:
                self._coll.insert_many(batch, ordered=False)
                batch = []
        if batch:
            self._coll.insert_many(batch, ordered=False)
        elapsed = time.time() - start
        print(
            f"[vb] insert complete in {elapsed:.1f}s "
            f"({len(X) / max(elapsed, 1e-9):,.0f} rows/s)",
            file=sys.stderr,
        )
        return elapsed

    def _create_index(self, dim: int) -> None:
        fields: List[Dict[str, Any]] = [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": dim,
            "similarity": SIMILARITY[self._metric],
        }]
        if self._quantization != "none":
            fields[0]["quantization"] = self._quantization
        # Declared even when unused: a field can only be filtered on if it was
        # indexed as one, and adding it later would rebuild the index.
        fields.append({"type": "filter", "path": "tag"})

        print(f"[vb] createSearchIndex {INDEX_NAME} "
              f"(quantization={self._quantization})", file=sys.stderr)
        self._coll.create_search_index({
            "name": INDEX_NAME,
            "type": "vectorSearch",
            "definition": {"fields": fields},
        })

    def _wait_until_ready(self) -> float:
        started = time.time()
        last_report = started
        while True:
            status = self._index_status()
            if status == "READY":
                return time.time() - started
            # DOES_NOT_EXIST is not fatal on its own: createSearchIndex returns
            # before the index is listable, so the first polls legitimately see
            # nothing. It is fatal only if it persists, which means mongod
            # accepted the index and mongot never took it.
            fatal = status == "FAILED" or (
                status == "DOES_NOT_EXIST"
                and time.time() - started > MISSING_INDEX_GRACE_S)
            if fatal:
                self._dump_mongot_log()
                raise RuntimeError(
                    f"mongot index reported {status} after "
                    f"{time.time() - started:.0f}s. The mongot log tail is "
                    f"above; the usual causes are mongot not being reachable "
                    f"from mongod and the index never being queued for an "
                    f"initial sync.")
            waited = time.time() - started
            if waited > READY_TIMEOUT_S:
                self._dump_mongot_log()
                raise TimeoutError(
                    f"mongot index still {status} after {waited / 3600:.1f} h")
            if time.time() - last_report >= 30:
                print(f"[vb] mongot index {status}, {waited / 60:.1f} min elapsed",
                      file=sys.stderr)
                last_report = time.time()
            time.sleep(READY_POLL_S)

    def _dump_mongot_log(self, lines: int = 40) -> None:
        """Put mongot's own log where the failure is visible.

        ann-benchmarks catches a per-algorithm exception and exits zero, so a
        module that raises leaves a phase marked completed with no results and
        no explanation anywhere. mongot logs to a file inside the container that
        nothing collects, which is why three runs failed identically with
        nothing to diagnose from.
        """
        path = os.path.join(os.environ.get("VB_MONGOT_DATA",
                                           "/var/lib/vbench/mongot"),
                            "mongot.log")
        print(f"[vb] --- mongot log tail ({path}) ---", file=sys.stderr)
        try:
            with open(path) as fh:
                tail = fh.readlines()[-lines:]
            for line in tail:
                print(f"[mongot] {line.rstrip()}", file=sys.stderr)
        except OSError as exc:
            print(f"[vb] could not read the mongot log: {exc}", file=sys.stderr)
        print("[vb] --- end mongot log ---", file=sys.stderr)

    def _index_status(self) -> str:
        try:
            for info in self._coll.list_search_indexes():
                if info.get("name") == INDEX_NAME:
                    return str(info.get("status", "UNKNOWN"))
        except Exception:
            return "UNKNOWN"
        return "DOES_NOT_EXIST"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def set_query_arguments(self, num_candidates: int) -> None:
        self._num_candidates = int(num_candidates)
        if self._query_verified is None:
            self._query_verified = self._verify_query()
        self._warm_index_once()   # once per built index; see vb_warmup.py
        self._warm_cache()

    def _pipeline(self, vector, k: int) -> List[Dict[str, Any]]:
        return [
            {"$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": encode_vector(vector),
                # Never below k. The shared grid starts at 10 and k is 10, so
                # the low end of the sweep reaches the floor exactly.
                "numCandidates": max(int(self._num_candidates or k), k),
                "limit": k,
            }},
            {"$project": {"_id": 1}},
        ]

    def _warm_cache(self) -> None:
        """Throwaway queries so the timed ones do not pay for a cold cache.

        Random vectors rather than the benchmark's own query set, which would
        prime the exact graph regions about to be measured.
        """
        if WARMUP_QUERIES <= 0 or not self._dim:
            return
        rng = numpy.random.RandomState(0)
        for _ in range(WARMUP_QUERIES):
            try:
                self.query(rng.normal(size=self._dim).astype(numpy.float32), 10)
            except Exception:
                return

    def _verify_query(self, k: int = 10) -> bool:
        """Confirm a vector search actually runs.

        There is no seq-scan fallback to detect here, unlike the SQL engines: a
        missing index makes the aggregation raise. What this catches is that
        case, so a configuration that errored is not recorded as one that was
        merely slow.
        """
        try:
            # A real vector, not zeros. Under cosine similarity a zero vector
            # has no direction and the distance is undefined; the smoke dataset
            # is euclidean, where zeros are legal, so this only failed at full
            # scale on an angular corpus.
            probe = (self._probe if self._probe is not None
                     else numpy.random.RandomState(0).normal(
                         size=self._dim or 1).astype(numpy.float32))
            list(self._coll.aggregate(self._pipeline(probe, k)))
        except Exception as exc:
            print(f"[vb] WARNING: $vectorSearch failed: {exc}", file=sys.stderr)
            return False
        print("[vb] query check OK: $vectorSearch served by the index", file=sys.stderr)
        return True

    def query(self, v, n: int) -> List[int]:
        return [d["_id"] for d in self._coll.aggregate(self._pipeline(v, n))]

    def batch_query(self, X, n: int) -> None:
        from multiprocessing.pool import ThreadPool

        try:
            threads = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover
            threads = os.cpu_count() or 1
        threads = max(1, min(8, threads))

        def worker(chunk):
            client = self._connect()
            coll = client[DATABASE][COLLECTION]
            out = []
            for v in chunk:
                out.append([d["_id"] for d in coll.aggregate(self._pipeline(v, n))])
            client.close()
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
        """Size of the Lucene index.

        collStats cannot see it: the segments belong to mongot, not mongod. The
        index metadata is asked first, and the data directory is walked only as
        a fallback, because that directory holds mongot's logs and any other
        index it has been asked to keep.
        """
        try:
            for info in self._coll.list_search_indexes():
                if info.get("name") == INDEX_NAME:
                    size = int(info.get("indexSizeBytes") or 0)
                    if size:
                        return size
        except Exception:
            pass
        data_dir = os.environ.get("VB_MONGOT_DATA", "/var/lib/vbench/mongot")
        total = 0
        for root, _dirs, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".log"):
                    continue
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total

    def get_memory_usage(self) -> float:
        return self._index_bytes / 1024.0

    def get_additional(self) -> Dict[str, Any]:
        return hdf5_safe({
            "engine": "mongodb",
            "resource_pass": os.environ.get("VB_RESOURCE_PASS", "unknown"),
            "engine_version": self._server_version(),
            "storage_engine": "wiredTiger",
            # Carried, not applied: mongot exposes no graph degree. Present so
            # a point can be placed in the sweep, not as a claim that the
            # comparison is at matched M.
            "M": self._m,
            "m_applied": False,
            # ef_construction is deliberately absent, not None: mongot does
            # not expose it, and None is what store_results cannot write.
            "ef_search": self._num_candidates,
            "num_candidates": self._num_candidates,
            "quantization": self._quantization,
            "metric": self._metric,
            "build_mode": "async",
            "index_bytes": self._index_bytes,
            "load_seconds": round(self._load_seconds, 3),
            "build_seconds": round(self._build_seconds, 3),
            # Reported separately because it overlaps the load. Adding it to
            # load_seconds would double count; ignoring it would report a
            # build that finished before the index existed.
            "index_ready_seconds": round(self._ready_seconds, 3),
            "vector_index_used": self._query_verified,
            # Nothing here is compiled by us: mongod arrives as a published
            # binary and mongot runs on the JVM, so there is no -march to
            # report and claiming one would be false.
            "march": "none",
            "jvm_version": self._jvm_version(),
            **self._warmup_attrs(),
        })

    def _server_version(self) -> str:
        try:
            version = self._client.admin.command("buildInfo").get("version", "?")
        except Exception:  # pragma: no cover
            version = "unknown"
        return f"Percona Server for MongoDB {version} / mongot"

    @staticmethod
    def _jvm_version() -> str:
        """Stands in for the -march every other engine records.

        mongot's distance kernels come from the JVM's vector API, so the JVM
        build is the closest thing to a statement about which SIMD path ran.
        """
        try:
            out = subprocess.run(["java", "-version"], capture_output=True,
                                 text=True, timeout=15)
            line = (out.stderr or out.stdout).splitlines()
            return line[0].strip() if line else "unknown"
        except Exception:
            return "unknown"

    def __str__(self) -> str:
        return (f"PerconaSearch(quantization={self._quantization}, "
                f"numCandidates={self._num_candidates})")

    def done(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(SERVER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._server.kill()
            self._server = None
