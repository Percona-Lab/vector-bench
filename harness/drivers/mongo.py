"""Ops-harness driver for Percona Search for MongoDB (mongot).

A third client stack, and a third architecture. The MySQL-family engines and
pgvector each keep their vector index inside the query engine; here `mongod`
holds the documents and a separate Lucene process, `mongot`, holds the index and
is fed by a change stream. Three consequences run through this file:

  * The index build is asynchronous. createSearchIndex returns before the index
    exists, so `create_index` polls getSearchIndexes until status is READY and
    reports the waiting as build time. Neither `incremental_index = True`
    (MHNSW, VIDX) nor `False` (pgvector's separable bulk build) describes that,
    so this driver reports a third mode.

  * The filter is applied before vector comparison rather than after. Every
    other engine in the set post-filters, which is why their filtered results
    either collapse to a fraction of a query per second or come back short.

  * Sizing a query means numCandidates, not ef_search. It is the same knob in
    spirit, but the vendor's guidance starts at 20x the limit, so the low end of
    the shared grid is further outside its intended range than it is for the
    others.
"""

from __future__ import annotations

import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy

from .engine_waits import mongo_delta, mongo_snapshot
from .base import (DATABASE, FILTER_COLUMNS, FILTER_TAG, TABLE, ConnectionSpec,
                   EngineDriver, IndexSpec, LoadResult, RowAttrs,
                   checked_filter_column)

try:
    import pymongo
    from bson.binary import Binary
except ImportError:  # pragma: no cover - only the bench image has these
    pymongo = None
    Binary = None

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

INDEX_NAME = "vector_index"
# createSearchIndex is asynchronous with no completion callback, so the only
# way to know the index exists is to ask until it says so.
READY_POLL_S = float(os.environ.get("VB_MONGOT_POLL_INTERVAL", "2"))
READY_TIMEOUT_S = float(os.environ.get("VB_MONGOT_READY_TIMEOUT", "43200"))

#: How often `wait_index_synced` asks mongot whether it has the newest row,
#: and how long it asks for before giving up. Shorter than READY_TIMEOUT_S by
#: an order of magnitude on purpose: that one waits out a build of the whole
#: index, this one waits for one document to travel a change stream mongot is
#: already following. An hour of it means something is wrong, and a churn that
#: hangs for twelve is a lost night rather than a slow measurement.
SYNC_POLL_S = float(os.environ.get("VB_MONGOT_SYNC_INTERVAL", "2"))
SYNC_TIMEOUT_S = float(os.environ.get("VB_MONGOT_SYNC_TIMEOUT", "3600"))

#: How many hits the sync probe asks for. Not 1, and the reason is the churn
#: sequence itself: the re-inserted rows carry the SAME vectors as the rows
#: just deleted, so while mongot is still absorbing the deletes the old id
#: sits at distance 0 from the probe vector too and can take the only slot.
#: Ten is enough that a handful of un-absorbed duplicates cannot crowd the
#: row out, and small enough to be a cheap question to ask every two seconds.
SYNC_PROBE_LIMIT = 10

SIMILARITY = {"angular": "cosine", "euclidean": "euclidean"}

#: The index-size settle (ops measurement version 9). mongot's directory sum
#: moves for minutes after a build while Lucene merges segments --
#: `full-20260825` read 15.1 GiB after the load and 7.2 GiB at the churn
#: baseline -- so the `index_build` figure is read once two sums
#: `SETTLE_INTERVAL_S` apart differ by under `SETTLE_TOLERANCE`, or at the
#: cap with `settled: false` recorded.
SETTLE_INTERVAL_S = float(os.environ.get("VB_MONGOT_SETTLE_INTERVAL", "30"))
SETTLE_TOLERANCE = 0.01
SETTLE_CAP_S = float(os.environ.get("VB_MONGOT_SETTLE_CAP", "600"))

#: The settle's clock and sleep, module-level so a test can drive a shrinking
#: directory through minutes of simulated time without spending them.
_clock = time.time
_sleep = time.sleep


def encode_vector(vector) -> Any:
    """Pack a float vector as BSON BinData float32.

    A BSON array of doubles spends 8 bytes per dimension, so 990k x 1536 costs
    12.2 GB in mongod against the 5.7 GB the same vectors occupy in Lucene.
    BinData float32 halves the working set and is the encoding MongoDB
    recommends for vector search.
    """
    values = numpy.asarray(vector, dtype=numpy.float32)
    # Subtype 9 is BSON's vector type; the two header bytes are the float32
    # data-type marker and a zero padding count.
    payload = b"\x27\x00" + values.tobytes()
    return Binary(payload, 9)


def decode_vector(packed) -> numpy.ndarray:
    """The inverse of `encode_vector`, for reading a stored vector back.

    `wait_index_synced` needs it: the only way to ask mongot whether it holds
    a particular document is to search for that document's own vector, and the
    only copy of that vector the driver can still reach is the one in mongod.

    The two header bytes are dropped rather than checked. A subtype 9 payload
    this driver did not write cannot arise here -- the probe reads back a
    document this process inserted through `encode_vector` one call earlier --
    and a length that is not a multiple of four makes `frombuffer` raise,
    which is the loud failure a silent reshape would not be.
    """
    return numpy.frombuffer(bytes(packed)[2:], dtype="<f4")


class MongoDriver(EngineDriver):
    name = "mongodb"
    # Neither of the framework's two modes. The graph is not maintained on
    # INSERT and it is not built by a blocking statement after the load; it is
    # built in another process while and after the writes land.
    incremental_index = False
    async_index_build = True

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        self._db = None
        self._coll = None
        self._index: Optional[IndexSpec] = None
        self._num_candidates: int = 100
        self._ready_seconds: float = 0.0
        # The highest id `insert_rows` has written, which is what the sync
        # probe looks for. None until something has been written through that
        # path, which is a state the probe reports rather than guesses at.
        self._newest_inserted_id: Optional[int] = None
        # What the NEXT `wait_index_synced` must observe, set by the churn
        # workload through `expect_index_state` (ops version 9): a deleted
        # (id, vector) that must stop being returned, and an id that must
        # start being returned. Consumed by the wait, so a hint cannot leak
        # into a later one.
        self._sync_absent: Optional[Any] = None
        self._sync_present: Optional[int] = None
        # Why `index_sync_s` came out the way it did, for the churn record.
        # Read by `harness/workloads/churn.py` through `getattr`, the same way
        # it already reads Valkey's `write_batch_used`: an optional attribute
        # the engines with nothing to say simply do not have. A zero with no
        # reason beside it is what H1's acceptance criterion refuses.
        self.index_sync_note: Optional[str] = None

    # -- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        if pymongo is None:
            raise RuntimeError("pymongo is not installed in this image")
        uri = self._uri()
        self._conn = pymongo.MongoClient(uri, serverSelectionTimeoutMS=60000)
        # Fail here rather than inside the first measured operation.
        self._conn.admin.command("ping")
        self._db = self._conn[self.spec.database or DATABASE]
        self._coll = self._db[TABLE]

    def _uri(self) -> str:
        """Authenticated, unlike every other driver here.

        mongot will not parse a config without SCRAM or x509, so mongod runs
        with auth on and the benchmark client has to authenticate as well. The
        credentials are fixed and local to an isolated container network; they
        are not a secret, they are a requirement mongot imposes.
        """
        user, password = self.spec.user, self.spec.password
        if not user:
            return f"mongodb://{self.spec.host}:{self.spec.port}/?directConnection=true"
        return (f"mongodb://{user}:{password}@{self.spec.host}:{self.spec.port}/"
                f"?directConnection=true&authSource=admin")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- schema ---------------------------------------------------------

    def drop_schema(self) -> None:
        try:
            self._db.drop_collection(TABLE)
        except Exception:
            pass

    def create_schema(self, index: IndexSpec) -> None:
        """Create the collection only.

        Unlike MHNSW and VIDX there is nothing to declare here that builds a
        graph: the search index is a separate object owned by mongot, created
        in create_index.
        """
        self._index = index
        self._db.create_collection(TABLE)

    def create_index(self, index: IndexSpec) -> None:
        """Create the search index and wait for mongot to finish building it.

        The wait is the point. createSearchIndex returns in milliseconds and the
        index is unqueryable until mongot has consumed the change stream and
        written its segments, so treating the call as the build would report a
        build time of nearly zero.
        """
        self._index = index
        fields: List[Dict[str, Any]] = [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": index.dim,
            "similarity": SIMILARITY[index.metric],
        }]
        if index.quantization and index.quantization != "none":
            fields[0]["quantization"] = index.quantization
        # Declared even when unused: a field can only be filtered on if it was
        # indexed as one, and adding it later would rebuild the index.
        #
        # That sentence is also the reason PR 3 loads both filter columns in
        # one build rather than choosing one column's values per run. For this
        # engine "choose per run" is not a reload of the table, it is a rebuild
        # of the Lucene index by another process, waited out through
        # `_wait_until_ready` -- so the cheaper-looking option B is the more
        # expensive one here, and declaring both paths now costs one filter
        # field in the definition. Both are declared from FILTER_COLUMNS, the
        # same tuple `checked_filter_column` matches a query's path against,
        # so the set of paths this index carries and the set of paths a query
        # may ask for cannot drift apart -- and a drift here is only
        # discoverable after the build, which is the expensive moment.
        fields.extend({"type": "filter", "path": column}
                      for column in FILTER_COLUMNS)

        self._coll.create_search_index({
            "name": INDEX_NAME,
            "type": "vectorSearch",
            "definition": {"fields": fields},
        })
        self._ready_seconds = self._wait_until_ready()

    def _wait_until_ready(self) -> float:
        started = time.time()
        last_report = started
        while True:
            status = self._index_status()
            if status == "READY":
                return time.time() - started
            if status in ("FAILED", "DOES_NOT_EXIST"):
                raise RuntimeError(
                    f"mongot index build reported {status}; "
                    "check the mongot container logs")
            waited = time.time() - started
            if waited > READY_TIMEOUT_S:
                raise TimeoutError(
                    f"mongot index still {status} after {waited / 3600:.1f} h")
            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[mongot] index {status}, {waited / 60:.1f} min elapsed",
                      flush=True)
                last_report = time.time()
            time.sleep(READY_POLL_S)

    def _index_status(self) -> str:
        try:
            for info in self._coll.list_search_indexes():
                if info.get("name") == INDEX_NAME:
                    return str(info.get("status", "UNKNOWN"))
        except Exception:
            return "UNKNOWN"
        return "DOES_NOT_EXIST"

    # -- data -----------------------------------------------------------

    def load(self, vectors: numpy.ndarray,
             attrs: Union[RowAttrs, Sequence[int], numpy.ndarray],
             threads: int = 1, start_id: int = 0) -> LoadResult:
        rows = len(vectors)
        attrs = RowAttrs.coerce(attrs)
        started = time.time()
        if threads <= 1:
            self._insert_range(vectors, attrs, start_id, 0, rows)
        else:
            chunk = (rows + threads - 1) // threads
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = [
                    pool.submit(self._insert_range, vectors, attrs, start_id,
                                begin, min(begin + chunk, rows))
                    for begin in range(0, rows, chunk)
                ]
                for f in futures:
                    f.result()
        return LoadResult(rows=rows, wall_seconds=time.time() - started,
                          threads=threads)

    def _insert_range(self, vectors, attrs: RowAttrs, start_id: int,
                      begin: int, end: int, batch: int = 1000) -> None:
        # Each worker opens its own client: MongoClient is thread safe but
        # shares one connection pool, and sharing it would make the thread count
        # a setting on the pool rather than a real measure of write concurrency.
        client = pymongo.MongoClient(self._uri())
        try:
            coll = client[self.spec.database or DATABASE][TABLE]
            for chunk_start in range(begin, end, batch):
                chunk_end = min(chunk_start + batch, end)
                coll.insert_many([
                    {"_id": start_id + i,
                     **attrs.fields(i),
                     "embedding": encode_vector(vectors[i])}
                    for i in range(chunk_start, chunk_end)
                ], ordered=False)
        finally:
            client.close()

    def delete_ids(self, ids: Sequence[int]) -> None:
        self._coll.delete_many({"_id": {"$in": list(ids)}})

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    attrs: Union[RowAttrs, Sequence[int], numpy.ndarray]
                    ) -> None:
        attrs = RowAttrs.coerce(attrs)
        self._coll.insert_many([
            {"_id": int(ids[i]), **attrs.fields(i),
             "embedding": encode_vector(vectors[i])}
            for i in range(len(ids))
        ], ordered=False)
        # Recorded after the write returns, so the id the sync probe looks for
        # is one mongod has definitely accepted. Tracked here and not in
        # `load`: the probe exists for churn's re-insert, and `load` writes
        # from several threads through clients of their own, where "the newest
        # id" is whichever thread finished last rather than a fact about the
        # corpus.
        if len(ids):
            self._newest_inserted_id = max(int(i) for i in ids)

    def count_rows(self) -> int:
        return int(self._coll.count_documents({}))

    # -- churn ----------------------------------------------------------

    def expect_index_state(self, absent: Optional[Any] = None,
                           present: Optional[int] = None) -> None:
        """Tell the next `wait_index_synced` what to probe for (ops v9).

        `absent` is (id, vector) of a row just deleted -- its vector is gone
        from mongod, so the caller hands it over -- and `present` an id that
        must be searchable: the highest loaded id before the churn baseline,
        the newest re-inserted id after a re-insert. `full-20260825` measured
        the baseline and the post-delete state without either wait.
        """
        self._sync_absent = absent
        self._sync_present = None if present is None else int(present)

    def _wait_absent(self, row_id: int, vector, expected_rows: int) -> float:
        """Poll until mongot stops returning a deleted row for its own vector.

        A failed probe is NOT absence -- `_probe` returns None for it -- or an
        unreachable mongot would read as a delete it had absorbed.
        """
        started = time.time()
        last_report = started
        polls = 0
        while True:
            polls += 1
            hits = self._probe(vector)
            waited = time.time() - started
            if hits is not None and row_id not in hits:
                self.index_sync_note = (
                    f"mongot stopped returning deleted row {row_id:,} after "
                    f"{waited:.1f}s ({polls} poll(s))")
                print(f"[mongot] index sync: {self.index_sync_note}",
                      flush=True)
                return waited
            if waited > SYNC_TIMEOUT_S:
                self.index_sync_note = (
                    f"mongot still returned deleted row {row_id:,} after "
                    f"{waited:.0f}s; what follows measures an index that had "
                    f"not absorbed the deletes")
                print(f"[mongot] WARNING: {self.index_sync_note}", flush=True)
                return waited
            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[mongot] waiting for deleted row {row_id:,} to leave "
                      f"the index ({expected_rows:,} rows expected), "
                      f"{waited:.0f}s elapsed", flush=True)
                last_report = time.time()
            time.sleep(SYNC_POLL_S)

    def wait_index_synced(self, expected_rows: int) -> float:
        """Wait until mongot's index holds the row most recently written.

        **Since ops measurement version 9 it waits for what it was told**
        (`expect_index_state`): a deleted row until it stops being returned,
        then a present row until it is. With no hint it does what it always
        did -- the newest row `insert_rows` wrote.

        **The premise this exists for is not verified here.** `insert_rows`
        hands the documents to mongod and returns; mongot learns about them
        from a change stream, in another process, on its own schedule, and
        nothing in that path blocks. So a churn re-measurement could score a
        partially synced index -- high latency and low recall, reported as a
        property of the engine. Confirming the lag and its size needs a live
        server and is the human task H1; what is implemented here is the wait,
        against the only question mongot answers honestly, which is whether a
        search finds a given document.

        **The probe searches for the newest row's own vector, rather than
        filtering on its id.** `$vectorSearch.filter` only accepts paths the
        index declares as filter fields -- `tag` and `ctag`, not `_id` -- so
        "restricted to the newest inserted id" cannot be a predicate. A
        document is at distance 0 from itself, so once mongot holds it, it is
        in the top few hits for its own vector; until then it is in none.

        `expected_rows` is not what is waited on, and that is deliberate
        rather than an oversight: a document count is mongod's answer, and
        mongod was synchronous all along. It is printed in the heartbeat so
        the log says which churn step the wait belongs to.

        Returns seconds waited. A zero always comes with a reason, in
        `index_sync_note` and in the log: either nothing has been written
        through `insert_rows` yet, or mongot already had the row at the first
        poll. A timeout returns the time spent and warns -- the measurement
        that follows is of an index that was still catching up, which is worth
        publishing with that caveat and not worth hanging a run over.
        """
        absent, self._sync_absent = self._sync_absent, None
        hinted, self._sync_present = self._sync_present, None
        waited_absent = 0.0
        if absent is not None:
            waited_absent = self._wait_absent(int(absent[0]), absent[1],
                                              expected_rows)
            if hinted is None:
                return waited_absent
        newest = hinted if hinted is not None else self._newest_inserted_id
        if newest is None:
            self.index_sync_note = (
                "nothing has been written through insert_rows, so there was "
                "no document to wait for")
            print(f"[mongot] index sync: {self.index_sync_note}", flush=True)
            return 0.0

        probe = self._probe_vector(newest)
        started = time.time()
        last_report = started
        polls = 0
        while True:
            polls += 1
            if newest in self._probe_hits(probe):
                waited = time.time() - started
                self.index_sync_note = (
                    f"mongot returned row {newest:,} for its own vector "
                    f"after {waited:.1f}s ({polls} poll(s))")
                if polls == 1:
                    self.index_sync_note += (
                        "; the first poll already found it, so this is an "
                        "upper bound on the lag and not a measurement of it")
                print(f"[mongot] index sync: {self.index_sync_note}",
                      flush=True)
                return waited + waited_absent
            waited = time.time() - started
            if waited > SYNC_TIMEOUT_S:
                self.index_sync_note = (
                    f"mongot had not returned row {newest:,} after "
                    f"{waited:.0f}s; what follows measures an index that was "
                    f"still catching up")
                print(f"[mongot] WARNING: {self.index_sync_note}", flush=True)
                return waited + waited_absent
            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[mongot] waiting for row {newest:,} to appear in the "
                      f"index ({expected_rows:,} rows expected), "
                      f"{waited:.0f}s elapsed", flush=True)
                last_report = time.time()
            time.sleep(SYNC_POLL_S)

    def _probe_vector(self, row_id: int) -> numpy.ndarray:
        """The stored vector of `row_id`, read back out of mongod.

        Raises if the document is not there. That is the right failure: the
        caller has just written it, so its absence means the write did not
        land and every number taken after this one would describe a corpus
        nobody loaded.
        """
        doc = self._coll.find_one({"_id": int(row_id)}, {"embedding": 1})
        if doc is None or doc.get("embedding") is None:
            raise RuntimeError(
                f"mongod has no embedding for row {row_id}, which "
                f"insert_rows had just written")
        return decode_vector(doc["embedding"])

    def _probe(self, vector) -> Optional[List[int]]:
        """`_probe_hits`, but None -- not [] -- when the probe failed."""
        try:
            return [int(d["_id"]) for d in self._coll.aggregate(
                self._pipeline(vector, SYNC_PROBE_LIMIT))]
        except Exception as exc:
            print(f"[mongot] index sync probe failed, treating it as "
                  f"not-yet-synced: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _probe_hits(self, vector) -> List[int]:
        """Ids mongot returns for `vector`, or an empty list if it cannot say.

        The exception is swallowed here and nowhere else in this hook, for the
        reason the wait exists: mongot returns an error while the index is
        rebuilding or unreachable, and a poll that raised would turn "not
        caught up yet" -- the state being waited for -- into a dead run.
        """
        try:
            return [int(d["_id"]) for d in self._coll.aggregate(
                self._pipeline(vector, SYNC_PROBE_LIMIT))]
        except Exception as exc:
            print(f"[mongot] index sync probe failed, treating it as "
                  f"not-yet-synced: {type(exc).__name__}: {exc}", flush=True)
            return []

    # -- query ----------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        """numCandidates is the ef_search analogue.

        Not a session variable: it is a per-query argument, so this records the
        value the next query will use rather than sending anything.
        """
        self._num_candidates = int(ef_search)

    def effective_ef(self, ef: int, k: int) -> int:
        """`numCandidates` is clamped to k, so "ef 4" here is a search of k.

        The one driver in the set that overrides this. `_pipeline` sends
        `max(self._num_candidates, k)` because numCandidates below limit cannot
        return limit rows, and that clamp changes what a calibration point
        means. At the default k=10 the shared grid's floor of 10 is untouched,
        so nothing moves; at a k of 100 the six grid points below it -- 10, 16,
        24, 32, 48 and 64 -- all collapse onto 100 and measure the same search
        six times over. `tests/test_calibrate.py` counts those six against the
        grid itself rather than leaving the number here to rot.

        Written as the same expression `_pipeline` uses rather than as a
        condition on it, because the promise this makes is about the query the
        driver actually sends; `tests/test_calibrate.py` checks the two against
        each other over a range of (ef, k) pairs instead of trusting that they
        agree.
        """
        return max(int(ef), int(k))

    def _pipeline(self, vector, k: int,
                  tag_threshold: Optional[int] = None,
                  column: str = FILTER_TAG) -> List[Dict[str, Any]]:
        """The one aggregation both the query path and EXPLAIN build.

        `tag_threshold` keeps its name because `tests/test_config.py:1988`
        passes it by keyword and that file is not this task's to edit.
        """
        stage: Dict[str, Any] = {
            "index": INDEX_NAME,
            "path": "embedding",
            "queryVector": encode_vector(vector),
            # Never below k: numCandidates < limit cannot return limit rows,
            # and the low end of the shared grid reaches there at k=10.
            "numCandidates": max(self._num_candidates, k),
            "limit": k,
        }
        if tag_threshold is not None:
            # The path is whitelisted even though a BSON key cannot be
            # injected the way a SQL identifier can: what it protects against
            # here is not injection but silence. `$vectorSearch.filter` on a
            # path the index has no filter field for is rejected by mongot,
            # but a path that is merely *misspelled* and happens to be
            # declared is not -- and the whitelist and the index definition
            # above are now built from the same tuple, so they cannot differ.
            stage["filter"] = {
                checked_filter_column(column): {"$lt": int(tag_threshold)}
            }
        return [{"$vectorSearch": stage}, {"$project": {"_id": 1}}]

    def query(self, vector, k: int) -> List[int]:
        return [d["_id"] for d in
                self._coll.aggregate(self._pipeline(vector, k))]

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        return [d["_id"] for d in
                self._coll.aggregate(
                    self._pipeline(vector, k, threshold, column))]

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        """Whether the query ran as a vector search rather than a collection scan.

        $vectorSearch has no planner fallback the way a SQL optimiser does: if
        the index is missing the aggregation errors instead of quietly scanning.
        The check is kept anyway, because an error here and a slow exact scan
        are worth telling apart in the record.
        """
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
            plan = self._db.command({
                "explain": {
                    "aggregate": TABLE,
                    "pipeline": self._pipeline(vector, k, tag_threshold,
                                               column),
                    "cursor": {},
                },
                "verbosity": "queryPlanner",
            })
        except Exception:
            return False
        return "$vectorSearch" in str(plan)

    # -- sizing / metadata ----------------------------------------------

    def index_bytes(self) -> int:
        """Size of the Lucene index, which lives in mongot rather than mongod.

        collStats cannot see it: it is another process's files. The orchestrator
        shares mongot's data directory read-only for exactly this, and the
        fallback reports what the index metadata claims rather than zero.
        """
        total = sum((self.index_bytes_breakdown() or {}).values())
        if total:
            return total
        try:
            for info in self._coll.list_search_indexes():
                if info.get("name") == INDEX_NAME:
                    return int(info.get("indexSizeBytes") or 0)
        except Exception:
            pass
        return 0

    def index_bytes_breakdown(self) -> Optional[Dict[str, int]]:
        """mongot's shared data directory, summed per file extension.

        Lucene names its files by kind -- `.vec` raw vectors, `.vex`/`.vem`
        the HNSW graph and its metadata, `.veq`/`.vemq` quantized vectors,
        `.cfs` compound segments -- so the breakdown says what the bytes are
        without this repository asserting a format it has not verified.
        None when the directory is not shared.
        """
        if not (self.spec.data_dir and os.path.isdir(self.spec.data_dir)):
            return None
        kinds: Dict[str, int] = {}
        for root, _dirs, files in os.walk(self.spec.data_dir):
            for name in files:
                try:
                    size = os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
                kind = os.path.splitext(name)[1].lstrip(".") or "(none)"
                kinds[kind] = kinds.get(kind, 0) + size
        return kinds or None

    def settle_index_bytes(self):
        """(bytes, seconds, detail): the directory sum once it stops moving.

        See `SETTLE_INTERVAL_S`. Without a shared directory there is nothing
        to watch and the immediate figure is returned with that reason.
        """
        first = sum((self.index_bytes_breakdown() or {}).values())
        if not first:
            return self.index_bytes(), 0.0, {
                "method": "immediate",
                "reason": "mongot's data directory is not shared"}
        started = _clock()
        previous = first
        readings = [first]
        while _clock() - started < SETTLE_CAP_S:
            _sleep(SETTLE_INTERVAL_S)
            current = sum((self.index_bytes_breakdown() or {}).values())
            readings.append(current)
            if previous and abs(current - previous) / previous \
                    < SETTLE_TOLERANCE:
                return current, _clock() - started, {
                    "method": "directory_stable", "settled": True,
                    "readings": len(readings), "first_bytes": first}
            previous = current
        return previous, _clock() - started, {
            "method": "directory_stable", "settled": False,
            "readings": len(readings), "first_bytes": first}

    def table_bytes(self) -> int:
        try:
            stats = self._db.command("collStats", TABLE)
            return int(stats.get("storageSize") or 0)
        except Exception:
            return 0

    def server_version(self) -> str:
        try:
            build = self._conn.admin.command("buildInfo")
            version = build.get("version", "?")
        except Exception:
            version = "?"
        return f"Percona Server for MongoDB {version} / mongot"

    def wait_snapshot(self):
        """mongod's lock-acquisition waits and ticket gauges, plus mongot's
        Prometheus counters when reachable. The search itself runs in mongot,
        so mongod's locks describe only the half of a query that fetches
        documents; the reason recorded beside a missing mongot block says
        why the other half is not there."""
        if self._conn is None:
            return None, "no connection"
        return mongo_snapshot(self._conn, self.spec.host)

    def wait_delta(self, before, after):
        return mongo_delta(before, after)

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update({
            "async_index_build": True,
            "prefilter": True,
            "quantization": (self._index.quantization if self._index else None),
            # Reported separately from the load: the two overlap, so adding
            # them would double count and treating either alone would
            # understate the cost of having a queryable index.
            "index_ready_seconds": round(self._ready_seconds, 3),
            # `index_vector_format` stays the base's "unknown": mongot is a
            # Percona package and its Lucene codec is not in this checkout.
            # `index_bytes_by_kind` carries the per-extension evidence.
        })
        return caps
