"""Ops-harness driver for Valkey + valkey-search.

The only in-memory engine in the set. The MySQL family and pgvector keep a
disk-backed table with a cache in front of it, and Percona Search keeps its
index in a separate process; Valkey keeps everything resident, always. Four
consequences run through this file.

  There is no on-disk index to size. `index_bytes` is measured as the resident
  memory the index costs, taken as the difference in used_memory across index
  creation, and `table_bytes` is what the hashes themselves occupy. Reporting
  zero would read as a missing measurement, and reporting a file size would be
  reporting a file that does not exist.

  Eviction is the trap. If Valkey reaches maxmemory under any policy except
  noeviction it drops keys, and vectors vanishing mid-run looks exactly like a
  bad index: recall falls, nothing errors. The policy is pinned in the server
  config and checked here after the load, because a benchmark that silently
  measures a partial dataset is worse than one that fails.

  EF_RUNTIME is a per-query modifier inside the query string, not a session
  variable, so `set_ef_search` records rather than sends.

  The index can be created before or after the load, and both are legitimate.
  Creating it first indexes on write, which is what MHNSW and VIDX are forced
  into; creating it after triggers a backfill, which is pgvector's bulk build.
  `build_mode` selects which comparison is being made, exactly as it does for
  pgvector, and the backfill is waited out through FT.INFO's own
  backfill_complete_percent rather than assumed complete.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy

from .engine_waits import valkey_delta, valkey_snapshot
from .base import (DATABASE, FILTER_COLUMNS, FILTER_TAG, TABLE,
                   ConnectionSpec, EngineDriver, IndexSpec, LoadResult,
                   RowAttrs, checked_filter_column)

try:
    import valkey as valkey_client
except ImportError:  # pragma: no cover - only the bench image has this
    try:
        import redis as valkey_client        # protocol-compatible fallback
    except ImportError:
        valkey_client = None

PROGRESS_INTERVAL_S = float(os.environ.get("VB_PROGRESS_INTERVAL", "20"))

INDEX = "idx"
PREFIX = "v:"
VECTOR_FIELD = "embedding"
# Kept, and now redundant inside this file: the FT.CREATE schema and the
# filter expression both derive their field names from
# `datasets.FILTER_COLUMNS` (via base.py), and the HSET mappings come from
# `RowAttrs.fields()`, so nothing here spells a column name any more. The name
# survives because `scripts/probe-valkey-filter.py` and
# `scripts/verify-valkey-churn.py` import it, and those are lab tools outside
# this task's ownership. Bound to FILTER_TAG rather than to the literal "tag"
# so it cannot become the one place that disagrees with the dataset module.
# There is deliberately no CTAG_FIELD: a second unused alias is a second name
# for a reader to wonder which of the two the code actually uses.
TAG_FIELD = FILTER_TAG
DIALECT = 2

BACKFILL_POLL_S = float(os.environ.get("VB_VALKEY_POLL_INTERVAL", "2"))
BACKFILL_TIMEOUT_S = float(os.environ.get("VB_VALKEY_BACKFILL_TIMEOUT", "43200"))

METRIC = {"angular": "COSINE", "euclidean": "L2"}

# Churn touches a fraction of the whole corpus in one call, so the same
# batching the load path uses applies here. Unbatched, a 10% churn of a
# million rows is one DEL of 99,000 keys and one pipeline holding 600 MB.
CHURN_BATCH = int(os.environ.get("VB_VALKEY_CHURN_BATCH", "1000"))

#: The ceiling `wait_index_synced` gives the index to absorb a re-insert.
#: Numerically `_wait_for_mutations`' own default, and passed EXPLICITLY for a
#: reason that is not style: that function signals its timeout only by
#: printing to stderr -- the value it returns on giving up is a plain elapsed
#: time, indistinguishable from a successful long wait. Holding the ceiling
#: here is what lets the note below say "this wait timed out" instead of
#: publishing `index_sync_s: 900.0` beside a sentence implying the queue
#: drained. Measured before this constant existed: with the wait returning
#: 900.0, the note named none of timeout / timed out / still catching up /
#: WARNING / had not.
INDEX_SYNC_TIMEOUT_S = 900.0

#: What the caller expects FT.INFO's `num_docs` to DO while a wait runs. One
#: wait serves two callers that want opposite things from the same comparison,
#: and until now only the first of them got it. `delete_ids` watches the count
#: fall TO the target, which is exactly what `num_docs <= expected` says. A
#: re-insert raises the count towards the target, so it starts BELOW it and
#: that comparison is already true when the wait begins: measured against a
#: scripted FT.INFO with the queue drained, the shipped wait returned after
#: 1 poll with the index 200 documents short of the target, and after 1 poll
#: with the index holding nothing at all. `DOCS_AT_MOST` stays the default
#: because it is what every existing caller of this wait means; the insert
#: path names `DOCS_EQUAL` for itself.
DOCS_AT_MOST = "at_most"
DOCS_EQUAL = "equal"

#: `DOCS_EQUAL` is equality and not `>=`, and that is the other half of the
#: repair. `>=` fixes the direction and keeps the trap: the mutation queue has
#: been observed drained while this index reported 481 documents MORE than the
#: keyspace held, so "at least the expected count" can be satisfied by stale
#: entries while the new rows are still unindexed. Measured, that mutant ends
#: the wait at 1 poll holding 2,481 documents against an expected 2,000, where
#: equality waits the 3 polls it takes to actually reach 2,000. Only equality
#: catches both an undercount and an overcount.
#:
#: An index that converges a document or two off the target would then wait out
#: the whole of INDEX_SYNC_TIMEOUT_S and publish every Valkey churn as "timed
#: out" -- a worse lie than the one being fixed. So a count that has STOPPED
#: MOVING also ends the wait: off target, and said so in `index_sync_detail`
#: and in the published note rather than hidden. It is the rule the
#: `expected_docs is None` branch already used, now named -- this many
#: consecutive readings equal to the one before.
DOCS_SETTLED_POLLS = 2

#: And the escape has a floor, because `DOCS_SETTLED_POLLS` repeats is four
#: seconds at a two-second poll, which is not evidence that an index has
#: finished. Measured on this driver before the floor existed: readings
#: 1,800 / 1,800 / 1,800 / 1,900 / 2,000 with the queue drained ended the wait
#: at 3 polls reporting "(200 short)" -- an index that was still converging
#: and reached the target two polls later -- and a flat reading of zero ended
#: it after 4 seconds holding nothing at all. The only thing separating that
#: from a correct exit is `queued == 0`, and the queue is the signal this file
#: documents as unreliable: the 481-document incident IS "the queue drained
#: while the index disagreed". A GC or CPU-contention stall mid-re-insert
#: produces exactly that shape -- queue momentarily empty, count flat for a
#: few seconds -- so the escape waits out a real stall before concluding the
#: count has stopped. A small poll count beside a large gap is what a
#: premature exit looks like in the note, and the note says so.
DOCS_SETTLED_FLOOR_S = 30.0

# Long enough that no honest write hits it, short enough that a reply which is
# never coming does not hold the machine until someone notices. Never None:
# that is what turned a stalled churn into a run that hung overnight.
WRITE_TIMEOUT_S = float(os.environ.get("VB_VALKEY_WRITE_TIMEOUT", "300"))

# The stall diagnosis must not become a second stall. A healthy write of one
# hash is a millisecond; thirty seconds is the difference between "slow" and
# "not coming".
CANARY_TIMEOUT_S = float(os.environ.get("VB_VALKEY_CANARY_TIMEOUT", "30"))

# The churn write gets a much shorter leash than the load. A healthy batch
# of a thousand took a quarter of a second during the load, so a minute is
# already two orders of magnitude of headroom -- and the batch size is
# halved on each stall, so this is paid once per halving rather than once.
CHURN_WRITE_TIMEOUT_S = float(
    os.environ.get("VB_VALKEY_CHURN_WRITE_TIMEOUT", "60"))

# Where the diagnosis looks for the turnover. The first ladder put it
# between 1 and 10 -- one HSET returns in a millisecond, ten never come
# back, and ten HSETs is sixty kilobytes, so this was never about buffer
# size. The low rungs are there to say whether two is already too many,
# which is the difference between a threshold and pipelining itself.
_PROBE_SIZES = (1, 2, 3, 5, 10, 100, 1000)

# How long a write may be outstanding before the server is asked about it,
# rather than after. A healthy batch of a thousand took a quarter of a
# second during the load, so anything still running at twenty is already
# two orders of magnitude out and worth a look.
STALL_OBSERVE_S = float(os.environ.get("VB_VALKEY_STALL_OBSERVE", "20"))


class _WriteStalled(RuntimeError):
    """A batch that did not come back. Carries no diagnosis of its own:
    the caller decides whether this is one to retry smaller or the last
    one, and only the last one is worth a full report."""



def _optional_int(value) -> Optional[int]:
    """int(value), or None for a field FT.INFO did not report.

    The distinction is the whole point: a field that is absent means the
    build does not report it, and treating that as zero is how a wait that
    was supposed to guard the measurement became a no-op.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _docs_ready(docs: Optional[int], expected: Optional[int],
                expectation: str, settled_for: int,
                elapsed_s: float) -> Tuple[bool, str]:
    """Has the indexed document count reached what the caller expects of it?

    Lifted out of the polling loop because the two callers of that loop want
    opposite things from this one comparison (see `DOCS_AT_MOST`), and because
    a condition this easy to get backwards is worth testing without a clock in
    the way -- `elapsed_s` is passed in rather than read here for that reason.
    Returns the verdict and the phrase naming which condition produced it:
    "the queue drained" and "the count arrived" are different measurements,
    and the record could not previously tell them apart.
    """
    if expectation not in (DOCS_AT_MOST, DOCS_EQUAL):
        # Validated before anything else, so no earlier branch can let a
        # misspelling through. A typo would otherwise reinstate the delete
        # path's comparison on the insert path -- the exact defect this
        # parameter was added to fix -- and publish "num_docs fell to the
        # expected count" for a count that rose and never arrived. Measured
        # before this check existed: 'eqaul', 'EQUAL', 'exact', '' and None
        # all returned ready for an index 200 documents short.
        raise ValueError(f"unknown docs_expectation {expectation!r}; expected "
                         f"{DOCS_AT_MOST!r} or {DOCS_EQUAL!r}")
    if docs is None:
        # This build reports no document count. The queue is then the only
        # signal there is; waiting on a number nobody publishes would hang
        # every churn rather than guard one. `wait_index_synced` publishes
        # that this is what happened, because it is the same shape as the
        # defect this parameter exists to fix.
        return True, "this build publishes no num_docs"
    stopped_moving = settled_for >= DOCS_SETTLED_POLLS
    if expected is None:
        # Nothing to compare against: settle instead, which needs two
        # consecutive readings so a single sample cannot end the wait. No
        # floor here: this branch is what `scripts/verify-valkey-churn.py`
        # drives, it predates the churn hooks, and nothing measured asks for
        # its timing to change.
        return stopped_moving, "num_docs stopped moving"
    if expectation == DOCS_EQUAL:
        if docs == expected:
            return True, "num_docs reached the expected count"
        # Off target: only after `DOCS_SETTLED_FLOOR_S`, so a few seconds of
        # a stalled re-insert cannot be read as a finished one.
        return (stopped_moving and elapsed_s >= DOCS_SETTLED_FLOOR_S,
                "num_docs stopped moving while still off target")
    # `DOCS_AT_MOST` by elimination, the expectation having been checked at
    # the top: the count falls TO the target on the delete path, and may land
    # below it when some of the ids handed to `delete_ids` were not in the
    # keyspace, which is why this is not equality.
    return docs <= expected, "num_docs fell to the expected count"


def encode_vector(vector) -> bytes:
    """FLOAT32 little-endian, which is what the module expects in PARAMS."""
    return numpy.asarray(vector, dtype="<f4").tobytes()


class ValkeyDriver(EngineDriver):
    name = "valkey"
    incremental_index = False

    def __init__(self, spec: ConnectionSpec):
        super().__init__(spec)
        self._index: Optional[IndexSpec] = None
        self._ef_runtime: int = 100
        self._bytes_before_index = 0
        self._bytes_after_index = 0
        self._backfill_seconds = 0.0
        # Not a constant: see insert_rows. Starts where the load writes
        # and shrinks only if the engine will not take that size.
        self._write_batch = CHURN_BATCH
        self._write_conn = None
        self._probed = False
        self._largest_working = 0
        # Why `index_sync_s` came out the way it did, for the churn record.
        # Read by `harness/workloads/churn.py` through `getattr`, the same way
        # it already reads `write_batch_used` below: an engine with nothing to
        # say simply has no such attribute. See `wait_index_synced`.
        self.index_sync_note: Optional[str] = None
        # What the last wait actually OBSERVED, as opposed to what it rested
        # on: the final count, the count expected, how many polls, which
        # condition ended it and which signal was binding. The note is prose
        # rendered from this; this is the thing a live smoke run has to
        # produce, because no fake can settle whether a real index converges
        # to the expected count, sits a few short of it, or overcounts.
        self.index_sync_detail: Optional[Dict[str, Any]] = None

    # -- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        if valkey_client is None:
            raise RuntimeError("valkey (or redis) client is not installed in this image")
        self._conn = valkey_client.Valkey(
            host=self.spec.host, port=self.spec.port,
            socket_timeout=600, socket_connect_timeout=30,
        ) if hasattr(valkey_client, "Valkey") else valkey_client.Redis(
            host=self.spec.host, port=self.spec.port,
            socket_timeout=600, socket_connect_timeout=30,
        )
        self._conn.ping()
        self._assert_no_eviction()

    def _assert_no_eviction(self) -> None:
        """Refuse to measure a configuration that can silently drop vectors.

        Any policy except noeviction turns a full memory condition into missing
        keys rather than an error, and the result is a recall figure computed
        over a corpus that is no longer the corpus.
        """
        policy = self._config_get("maxmemory-policy")
        if policy and policy != "noeviction":
            raise RuntimeError(
                f"maxmemory-policy is {policy!r}, not 'noeviction'. Under any "
                f"other policy Valkey drops keys when it reaches maxmemory, and "
                f"a vanished vector is indistinguishable from a bad index in the "
                f"results. Fix the server configuration rather than this check."
            )

    def _config_get(self, key: str) -> Optional[str]:
        try:
            value = self._conn.config_get(key)
            raw = value.get(key) if isinstance(value, dict) else None
            return raw.decode() if isinstance(raw, bytes) else raw
        except Exception:
            return None

    def close(self) -> None:
        self._drop_write_connection()
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- schema ---------------------------------------------------------

    def drop_schema(self) -> None:
        try:
            self._conn.execute_command("FT.DROPINDEX", INDEX)
        except Exception:
            pass
        try:
            self._conn.flushall()
        except Exception:
            pass

    def create_schema(self, index: IndexSpec) -> None:
        """No schema to declare.

        Valkey has no table: the hashes are the data and the index is declared
        over a key prefix. In incremental mode the index is created here so
        writes are indexed as they arrive; in post mode create_index does it
        afterwards and the module backfills.
        """
        self._index = index
        if index.build_mode == "incremental":
            self.create_index(index)

    def create_index(self, index: IndexSpec) -> None:
        self._index = index
        if self._index_exists():
            return
        self._bytes_before_index = self._used_memory()
        # Both scalar columns are declared NUMERIC, which is this engine's
        # whole index on them -- there is no separate B-tree to create, so
        # "the same kind of index on both" is satisfied by the two lines
        # being the same line. Derived from FILTER_COLUMNS for that reason:
        # a column missing from the schema is not an error at FT.CREATE
        # time, it is an `@ctag:[-inf (t]` that matches nothing, which reads
        # as a filtered recall of zero rather than as a missing field.
        schema: List[str] = []
        for column in FILTER_COLUMNS:
            schema += [column, "NUMERIC"]
        args = [
            "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", PREFIX,
            "SCHEMA",
            *schema,
            VECTOR_FIELD, "VECTOR", "HNSW", "8",
            "TYPE", "FLOAT32",
            "DIM", str(index.dim),
            "DISTANCE_METRIC", METRIC[index.metric],
            "M", str(index.m),
        ]
        if index.ef_construction:
            args += ["EF_CONSTRUCTION", str(index.ef_construction)]
            # attr_count counts the key/value pairs after the algorithm name.
            args[args.index("8")] = "10"
        self._conn.execute_command(*args)
        self._backfill_seconds = self._wait_for_backfill()
        self._bytes_after_index = self._used_memory()

    def _index_exists(self) -> bool:
        try:
            self._conn.execute_command("FT.INFO", INDEX)
            return True
        except Exception:
            return False

    def _wait_for_backfill(self) -> float:
        """Wait until the module has finished indexing what already exists.

        FT.CREATE returns immediately and indexes in the background, so a query
        issued straight afterwards searches a partial graph and reports a recall
        that has nothing to do with the index parameters.
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
                    f"valkey-search backfill still at {done:.1%} after "
                    f"{waited / 3600:.1f} h")
            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[valkey] backfill {done:.1%}, {waited / 60:.1f} min elapsed",
                      flush=True)
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

    # -- data -----------------------------------------------------------

    def load(self, vectors: numpy.ndarray,
             attrs: Union[RowAttrs, Sequence[int], numpy.ndarray],
             threads: int = 1, start_id: int = 0) -> LoadResult:
        rows = len(vectors)
        attrs = RowAttrs.coerce(attrs)
        started = time.time()
        if threads <= 1:
            self._write_range(vectors, attrs, start_id, 0, rows)
        else:
            chunk = (rows + threads - 1) // threads
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = [
                    pool.submit(self._write_range, vectors, attrs, start_id,
                                begin, min(begin + chunk, rows))
                    for begin in range(0, rows, chunk)
                ]
                for f in futures:
                    f.result()
        elapsed = time.time() - started
        self._assert_nothing_evicted()
        return LoadResult(rows=rows, wall_seconds=elapsed, threads=threads)

    def _write_range(self, vectors, attrs: RowAttrs, start_id: int,
                     begin: int, end: int, batch: int = 1000) -> None:
        # One connection per worker: a shared client would make the thread
        # count a property of the connection pool rather than a measure of
        # write concurrency.
        conn = self._write_connection()
        try:
            pipe = conn.pipeline(transaction=False)
            for i in range(begin, end):
                # One of THREE places in this file that writes this
                # mapping -- the bulk load, churn's re-insert
                # (`_write_batch_once`) and the stall probe (`_probe_write`).
                # The scalar fields come from `attrs.fields()` at all three
                # rather than being listed here, because PR 3 adds a field to
                # each and the field that gets missed is the one in the path
                # nobody runs on purpose.
                #
                # Three IN THIS FILE, and the count is worth stating exactly
                # because there is a FOURTH outside it:
                # `scripts/verify-valkey-churn.py:92-95` builds this mapping
                # by hand with TAG_FIELD and VECTOR_FIELD only. That script
                # calls `driver.create_index()` at :202, so the index it
                # queries does declare `ctag NUMERIC`, and then it writes
                # ctag-less hashes into the same `v:` keyspace at :92 while
                # ALSO writing ctag-carrying ones through `driver.insert_rows`
                # at :245. No published number moves -- it issues no filtered
                # search anywhere, and `lab` tools diagnose rather than
                # measure -- and `scripts/` is not PR 3's to change, so the
                # boundary is recorded here instead of silently assumed.
                # `scripts/probe-valkey-filter.py` needs nothing: it writes
                # through `driver.load`, so its hashes carry both fields.
                pipe.hset(f"{PREFIX}{start_id + i}", mapping={
                    **attrs.fields(i),
                    VECTOR_FIELD: encode_vector(vectors[i]),
                })
                if (i - begin + 1) % batch == 0:
                    pipe.execute()
            pipe.execute()
        finally:
            conn.close()

    def _assert_nothing_evicted(self) -> None:
        try:
            evicted = int(self._conn.info("stats").get("evicted_keys", 0))
        except Exception:
            return
        if evicted:
            raise RuntimeError(
                f"Valkey evicted {evicted:,} keys during the load. The corpus "
                f"is no longer the corpus and every recall figure from this run "
                f"would be computed against a dataset that is missing rows. "
                f"Raise the container memory limit or lower maxmemory."
            )

    def _write_connection(self, timeout_s: Optional[float] = None):
        """A connection for bulk writes: generous timeout, keepalive, not none.

        This has been wrong in both directions. It first carried the driver's
        600 second query timeout, which a slow write legitimately exceeds. It
        then carried no timeout at all, which is worse: a reply that never
        arrives blocks forever, and that is exactly what happened. The client
        sat at 0% CPU, 6.4 GB of a 10.6 GB limit, every cgroup memory counter
        at zero, waiting on a socket, while the server answered a fresh HSET
        into the same indexed prefix in 104 milliseconds.

        A timeout that a real write cannot hit, plus keepalive so a dead peer
        is noticed rather than waited on, plus a health check so a connection
        that has gone stale is replaced instead of trusted.
        """
        factory = getattr(valkey_client, "Valkey", None) or valkey_client.Redis
        return factory(
            host=self.spec.host, port=self.spec.port,
            socket_timeout=WRITE_TIMEOUT_S if timeout_s is None else timeout_s,
            socket_connect_timeout=30,
            socket_keepalive=True,
            health_check_interval=30,
        )

    def delete_ids(self, ids: Sequence[int]) -> None:
        """Batched, because a churn deletes a tenth of the corpus at once.

        One DEL carrying 99,000 key names is a single multi-bulk command of a
        megabyte or so. The load path has always batched; this did not, and it
        worked at smoke scale and failed at a million rows.
        """
        keys = [f"{PREFIX}{int(i)}" for i in ids]
        before = self.count_rows()
        conn = self._write_connection()
        try:
            for start in range(0, len(keys), CHURN_BATCH):
                conn.delete(*keys[start:start + CHURN_BATCH])
        finally:
            conn.close()
        # `DOCS_AT_MOST` is named rather than defaulted: this is the call the
        # shared wait's comparison was written for -- the indexed count falls
        # TO the target here -- and the insert path that also uses that wait
        # needs the opposite. Saying which one this is at the call site is
        # what stops the next reader from "fixing" the comparison for the
        # other caller and silently unguarding this one.
        self._wait_for_mutations(expected_docs=max(0, before - len(keys)),
                                 docs_expectation=DOCS_AT_MOST)

    def _wait_for_mutations(self, expected_docs: Optional[int] = None,
                            timeout_s: float = 900.0,
                            docs_expectation: str = DOCS_AT_MOST) -> float:
        """Let the index absorb a batch of mutations before measuring it.

        DEL returns as soon as the key is gone; removing the vector from the
        HNSW graph is separate work, and FT.INFO reports a mutation queue
        precisely so a caller can see it.

        The queue alone is not enough to wait on, and this is the second time
        that lesson has been paid for here. `percent_indexed` does not exist,
        so a missing field read as a finished backfill and every measurement
        after it described an empty index. `mutation_queue_size` is the same
        shape of trap: absent, or present and zero while the index still
        reported 481 documents more than the keyspace held. So the document
        count is waited on too, against the number the caller knows to expect,
        and a missing field is reported rather than treated as agreement.

        `docs_expectation` is how the caller says what it expects that count
        to do -- see `DOCS_AT_MOST`. The comparison used to be the delete
        path's unconditionally, which left the document half of this wait
        satisfied before the insert path's first poll.

        Whatever ends the wait, what it saw lands in `index_sync_detail`. A
        wait that returns an elapsed time and nothing else cannot be audited:
        0.0s because the index was already there, and 0.0s because the only
        condition still in force was trivially true, are the same number.
        """
        started = time.time()
        last_report = started
        settled_for = 0
        previous: Optional[int] = None
        announced_missing = False
        polls = 0
        docs: Optional[int] = None
        queued: Optional[int] = None
        # Carried forward rather than computed at the end, so the reading that
        # ends the wait does not erase what had been holding it open. A wait
        # that ends on its first poll bound on nothing, and the record has to
        # be able to say that too.
        blocked_by = "neither: the first reading already agreed"

        def observed(ended: str, timed_out: bool) -> Dict[str, Any]:
            return {"expected_docs": expected_docs, "observed_docs": docs,
                    "queue_size": queued, "polls": polls,
                    "docs_expectation": docs_expectation, "ended_by": ended,
                    "binding_signal": blocked_by, "timed_out": timed_out}

        while time.time() - started < timeout_s:
            info = self._ft_info()
            polls += 1
            queued = _optional_int(info.get("mutation_queue_size"))
            if queued is None and not announced_missing:
                announced_missing = True
                print("[valkey] FT.INFO reports no mutation_queue_size in this "
                      "build; waiting on the document count instead",
                      flush=True)
            docs = _optional_int(info.get("num_docs"))

            queue_drained = queued in (None, 0)
            settled_for = settled_for + 1 if docs == previous else 0
            docs_ready, ended_by = _docs_ready(docs, expected_docs,
                                               docs_expectation, settled_for,
                                               time.time() - started)
            previous = docs

            if queue_drained and docs_ready:
                self.index_sync_detail = observed(ended_by, False)
                return time.time() - started
            blocked_by = ("the mutation queue" if docs_ready else
                          "the document count" if queue_drained else "both")

            if time.time() - last_report >= PROGRESS_INTERVAL_S:
                print(f"[valkey] index catching up: "
                      f"num_docs={docs if docs is not None else '?'}"
                      f"{f' (target {expected_docs:,})' if expected_docs is not None else ''}"
                      f", queued={queued if queued is not None else 'n/a'}, "
                      f"{time.time() - started:.0f}s elapsed", flush=True)
                last_report = time.time()
            time.sleep(2)

        waited = time.time() - started
        self.index_sync_detail = observed("the wait gave up", True)
        print(f"[valkey] WARNING: the index had not caught up after "
              f"{waited:.0f}s ({self.index_sync_detail}). Measuring it "
              f"anyway; treat what follows as a measurement of an index that "
              f"was still catching up.", file=sys.stderr, flush=True)
        return waited

    def wait_index_synced(self, expected_rows: int) -> float:
        """Let the index absorb the re-inserted rows before re-measuring.

        Delegates to `_wait_for_mutations`, which waits on FT.INFO's mutation
        queue and on the indexed document count, and which already knows that
        the queue alone has been caught reading drained while the index
        disagreed with the keyspace by 481 documents.

        **The document half of that wait used to be dead on this path**, and
        the first version of this hook published the fact instead of fixing
        it. The comparison was `docs <= expected_docs`, written for the delete
        path, where the count falls TO the target; a re-insert raises it, so
        the comparison was already satisfied when the wait began. Measured
        against a scripted FT.INFO with the queue drained, that form returned
        after **1 poll** with the index 200 documents short of the target, and
        after **1 poll** with the index holding nothing at all -- leaving the
        mutation queue, the weaker of the two signals by the measurement that
        put the count there, as the only thing guarding a published recall
        figure. `DOCS_EQUAL` is passed below so the count binds here, and
        `delete_ids` names `DOCS_AT_MOST` so its own behaviour is untouched:
        2,000 falling to 1,800 against an expected 1,800 is 3 polls either
        way, measured.

        **Equality, not `>=`.** `>=` repairs the direction and keeps the
        trap -- see `DOCS_EQUAL`, which also carries the stopped-moving escape
        that keeps an index converging a document or two off the target from
        waiting out INDEX_SYNC_TIMEOUT_S and publishing the churn as "timed
        out".

        **What the wait saw is published, not inferred.** `index_sync_note`
        renders `index_sync_detail` -- the final count, the count expected,
        the polls, which condition ended the wait and which signal was
        binding -- and `churn.run` carries the note into the record beside
        `index_sync_s`. Which is also how the one open question about this
        engine gets settled: no fake can say whether a real index converges to
        the expected count, stops a few short of it, or comes back over, and
        the first live run now answers that instead of another argument about
        it. Without any of this the engine would publish a bare
        `index_sync_s: 0.0`, indistinguishable from a synchronous engine and
        from a hook nobody implemented, in the one engine where that zero is
        most misleading: Valkey is the engine whose writes have actually
        stopped returning mid-churn and cost two runs.
        """
        # Cleared first: `delete_ids` earlier in the same churn step left its
        # own readings here, and a note that rendered those would describe the
        # delete's 1,800 documents as the re-insert's. The published sentence
        # has to be about the wait it is attached to.
        self.index_sync_detail = None
        waited = self._wait_for_mutations(expected_docs=int(expected_rows),
                                          timeout_s=INDEX_SYNC_TIMEOUT_S,
                                          docs_expectation=DOCS_EQUAL)
        seen = self._observed_sync_phrase()
        ended_by = (self.index_sync_detail or {}).get("ended_by")
        off_target = ended_by == "num_docs stopped moving while still off target"
        no_count = ended_by == "this build publishes no num_docs"
        if waited >= INDEX_SYNC_TIMEOUT_S:
            # `_wait_for_mutations` warns on stderr and returns anyway, so
            # without this branch the record would carry the timeout's elapsed
            # time under a sentence describing a wait that finished.
            self.index_sync_note = (
                f"the index had not absorbed the re-insert after "
                f"{waited:.0f}s and the wait gave up{seen}; what follows "
                f"measures an index that was still catching up")
        elif off_target:
            # Deliberately not "settled in 30.0s". The full sentence is
            # honest either way, but a skim takes away the first three words,
            # and "settled" beside a small number is what a reader carries
            # into a recall table. This wait did not reach what it was
            # waiting for.
            self.index_sync_note = (
                f"stopped off target after {waited:.1f}s{seen}. A small poll "
                f"count beside a large gap is what a premature exit looks "
                f"like: the count is only treated as stopped after "
                f"{DOCS_SETTLED_FLOOR_S:.0f}s, but the queue reading that "
                f"lets it end at all has been seen draining while this index "
                f"disagreed with the keyspace by 481 documents")
        elif no_count:
            # The one case where nothing about the count was checked at all,
            # so it must not lead with "settled" either -- same reason as the
            # branch above. "settled in 0.0s" here would be the most
            # misleading sentence this driver can write.
            self.index_sync_note = (
                f"the document count could not be checked{seen}; the "
                f"{waited:.1f}s this took means the queue was empty, not that "
                f"the index had absorbed the re-insert")
        else:
            self.index_sync_note = (
                f"settled in {waited:.1f}s{seen}. The insert path waits for "
                f"`num_docs == {int(expected_rows):,}`, not the delete path's "
                f"`num_docs <= {int(expected_rows):,}` that this shared wait "
                f"was written with: a re-insert starts BELOW the target, where "
                f"`<=` is satisfied already, which left FT.INFO's mutation "
                f"queue as the only signal guarding the measurement -- "
                f"measured, 1 poll with the index 200 docs short. Equality "
                f"rather than `>=` because the queue has been observed drained "
                f"while the index reported 481 documents more than the "
                f"keyspace held")
        print(f"[valkey] index sync: {self.index_sync_note}", flush=True)
        return waited

    def _observed_sync_phrase(self) -> str:
        """The last wait's own readings, for the published note.

        Empty when there are no readings to report, which is what a caller
        that stubs the wait out sees. A sentence a reader can act on needs the
        numbers and not just the condition: an index that lands exactly on the
        target, one that stops two short and one that comes back 481 over are
        three different engines, and only this phrase tells them apart in a
        record that otherwise holds one float.
        """
        detail = self.index_sync_detail or {}
        docs, expected = (detail.get("observed_docs"),
                          detail.get("expected_docs"))
        if not detail or expected is None:
            return ""
        if docs is None:
            # The twin of the original defect, and it has to be said out loud:
            # a build that publishes no `num_docs` leaves the document half
            # satisfied unconditionally and the queue as the only guard, which
            # is the state this call was changed to get out of. Without this
            # clause the note is byte-identical to one from a wait that was
            # stubbed out entirely.
            return (f": FT.INFO published no num_docs in this build, so after "
                    f"{detail['polls']} poll(s) the mutation queue was the "
                    f"only signal guarding this measurement and the document "
                    f"half was satisfied unconditionally")
        gap = docs - expected
        where = ("exactly on target" if gap == 0 else
                 f"{abs(gap):,} {'over' if gap > 0 else 'short'}")
        return (f": num_docs={docs:,} against an expected {expected:,} "
                f"({where}) after {detail['polls']} poll(s), ended because "
                f"{detail['ended_by']}; the binding signal was "
                f"{detail['binding_signal']}")

    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    attrs: Union[RowAttrs, Sequence[int], numpy.ndarray]
                    ) -> None:
        """Write back in batches, shrinking the batch if one does not return.

        Two facts fix the shape of this. A pipeline of a thousand HSETs is what
        the load uses and it moves 30,000 rows a second into an unindexed
        keyspace. The same pipeline into the populated index does not come back
        at all -- 302 seconds, zero rows -- while a single HSET into that same
        index, on a connection opened at the moment of the stall, returns in
        one millisecond.

        So the batch is not a constant any more. A write that does not return
        halves it and tries again on a fresh connection, down to one row at a
        time, and the size that worked is reported: at a millisecond a row even
        the floor finishes 99,000 rows inside the budget. That is a slower
        measurement than the load, which is the honest result -- writing into a
        live HNSW graph is slower than writing into a hash table -- but it is a
        measurement, where a timeout is not.
        """
        attrs = RowAttrs.coerce(attrs)
        cursor = 0
        while cursor < len(ids):
            size = min(self._write_batch, len(ids) - cursor)
            stop = cursor + size
            try:
                self._write_batch_once(ids[cursor:stop], vectors[cursor:stop],
                                       attrs[cursor:stop])
            except _WriteStalled:
                self._drop_write_connection()
                if self._probed:
                    # The ladder has already run once; a stall past that is a
                    # different failure from the one it measured, so back off
                    # rather than re-measuring the same thing.
                    if self._write_batch <= 1:
                        raise RuntimeError(
                            "a single-row write did not return either, and the "
                            "ladder had already found a size that worked. This "
                            "is not the batching limit.")
                    self._write_batch = max(1, self._write_batch // 2)
                    print(f"[valkey] write of {size:,} rows did not return; "
                          f"retrying at {self._write_batch:,}", flush=True)
                    continue

                # First stall. Ask the server about it while it is still in the
                # state that produced it, and let the ladder pick the batch:
                # halving from a thousand to one costs nine more timeouts to
                # learn what the ladder already measured.
                self._probed = True
                print(self._diagnose_stalled_write(
                    ids[cursor:stop], vectors[cursor:stop],
                    attrs[cursor:stop]), flush=True)
                if not self._largest_working:
                    raise RuntimeError(
                        "no write returned at any size, down to a single row. "
                        "The stall is not about batching.")
                self._write_batch = self._largest_working
                print(f"[valkey] largest write the server returns is "
                      f"{self._write_batch:,} row(s); re-inserting at that size",
                      flush=True)
                continue
            cursor = stop

    def _write_batch_once(self, ids, vectors, attrs) -> None:
        """One pipeline, one connection, a timeout a healthy write cannot hit.

        Fresh each time: redis-py disconnects a connection whose read timed
        out, and reusing one that has already failed would confuse a wedged
        socket with a wedged server -- which is the distinction this whole path
        exists to make.
        """
        attrs = RowAttrs.coerce(attrs)
        conn = self._churn_connection()
        pipe = conn.pipeline(transaction=False)
        for i, key_id in enumerate(ids):
            pipe.hset(f"{PREFIX}{int(key_id)}", mapping={
                **attrs.fields(i),
                VECTOR_FIELD: encode_vector(vectors[i]),
            })
        try:
            self._execute_watched(pipe, len(ids))
        except Exception as exc:
            raise _WriteStalled(str(exc)) from exc

    def _churn_connection(self):
        """One connection, held while it keeps working.

        Once the batch collapses to a single row, 99,000 rows means 99,000
        writes, and opening a connection for each would measure connection
        setup rather than the write. It is dropped the moment one fails --
        redis-py has disconnected it by then anyway, and reusing a socket that
        has already timed out would confuse a wedged connection with a wedged
        server.
        """
        if self._write_conn is None:
            self._write_conn = self._write_connection(
                timeout_s=CHURN_WRITE_TIMEOUT_S)
        return self._write_conn

    def _drop_write_connection(self) -> None:
        conn, self._write_conn = self._write_conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _execute_watched(self, pipe, rows: int) -> None:
        """Run the batch, and look at the server while it is still hanging.

        Every diagnosis so far has run after the fact, and that is why five of
        them were wrong. redis-py disconnects a connection whose read timed
        out, so by the time an exception is caught the socket the server would
        have had an opinion about is already gone: CLIENT LIST shows nothing,
        connected_clients is back to one, and the only honest reading of any of
        it is "the client is no longer connected", which was never in doubt.

        So the write goes on a thread and the server is questioned from here
        while it is still outstanding. CLIENT LIST settles it in one line. If
        the stalled connection is there with a full query buffer and cmd=hset,
        the server has the write and is not finishing it. If it is not there at
        all, the server never took the connection, and nothing about the index
        or the module is involved.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(pipe.execute)
            try:
                return future.result(timeout=STALL_OBSERVE_S)
            except FuturesTimeout:
                pass
            print(f"[valkey] a write of {rows:,} rows has not returned after "
                  f"{STALL_OBSERVE_S:.0f}s. The server, while it is still "
                  f"outstanding:", flush=True)
            print(self._live_server_state(), flush=True)
            return future.result(
                timeout=max(1.0, CHURN_WRITE_TIMEOUT_S - STALL_OBSERVE_S))
        except FuturesTimeout as exc:
            raise TimeoutError(
                f"the write did not return within "
                f"{CHURN_WRITE_TIMEOUT_S:.0f}s") from exc
        finally:
            # Not waited on: the thread is blocked on a socket that is about to
            # be closed under it, and holding the run for that is the failure
            # this path exists to end.
            pool.shutdown(wait=False)

    def _live_server_state(self) -> str:
        """CLIENT LIST and the input counter, over the monitoring connection.

        Deliberately best-effort: this runs while something is already wrong.
        """
        lines = []
        try:
            clients = self._conn.execute_command("CLIENT", "LIST")
            if isinstance(clients, bytes):
                clients = clients.decode(errors="replace")
            for row in str(clients).splitlines():
                lines.append(f"    {row}")
        except Exception as exc:
            lines.append(f"    CLIENT LIST failed: {type(exc).__name__}: {exc}")
        for field in ("connected_clients", "blocked_clients",
                      "total_net_input_bytes", "instantaneous_ops_per_sec",
                      "used_memory_human"):
            try:
                lines.append(f"    INFO {field:24} = "
                             f"{self._conn.info().get(field, 'n/a')}")
            except Exception:
                break
        try:
            info = self._ft_info()
            lines.append(f"    FT.INFO mutation_queue_size = "
                         f"{info.get('mutation_queue_size', 'n/a')}, "
                         f"num_docs = {info.get('num_docs', 'n/a')}")
        except Exception:
            pass
        return "\n".join(lines)

    # What the load writes in one pipeline, for the record to compare against.
    load_batch_rows = CHURN_BATCH

    @property
    def write_batch_used(self) -> int:
        """The batch size the re-insert settled on, for the record.

        A churn that completed at one row per write and one that completed at a
        thousand are not the same result, and nothing else in the record would
        say which happened.
        """
        return self._write_batch

    def _diagnose_stalled_write(self, ids, vectors, attrs) -> str:
        """What the server was doing when a write did not come back.

        Deliberately best-effort throughout: this runs while something is
        already wrong, and a diagnosis that raises tells nobody anything.
        """
        attrs = RowAttrs.coerce(attrs)
        lines = ["[valkey] the write did not return. Server state at that moment:"]

        info = self._ft_info()
        for field in ("num_docs", "num_records", "mutation_queue_size",
                      "backfill_in_progress", "backfill_complete_percent",
                      "hash_indexing_failures", "state"):
            lines.append(f"    FT.INFO {field:26} = {info.get(field, 'n/a')}")

        try:
            server = self._conn.info()
        except Exception as exc:
            lines.append(f"    INFO failed on the monitoring connection: {exc}")
            lines.append("    -- which means the server, not one connection, "
                         "stopped answering.")
            return "\n".join(lines)

        for field in ("connected_clients", "blocked_clients", "used_memory_human",
                      "maxmemory_human", "mem_fragmentation_ratio",
                      "instantaneous_ops_per_sec", "rdb_bgsave_in_progress",
                      "loading", "total_net_input_bytes"):
            lines.append(f"    INFO    {field:26} = {server.get(field, 'n/a')}")

        # num_docs disagreeing with the keyspace is the difference between an
        # index that is behind and one that is wrong, and only one of those is
        # something to wait out.
        try:
            lines.append(f"    DBSIZE  {'keys':26} = {self._conn.dbsize():,}")
        except Exception:
            pass

        # The decisive test, and then the size at which it stops working. One
        # HSET returning while a thousand did not says the engine was never
        # what stalled; where between one and a thousand it turns over says
        # what to do about it.
        self._largest_working = 0
        for size in _PROBE_SIZES:
            if size > len(ids):
                break
            outcome = self._probe_write(ids[:size], vectors[:size],
                                        attrs[:size])
            if outcome.startswith("OK"):
                self._largest_working = size
            lines.append(f"    write of {size:>5} row(s) on a fresh "
                         f"connection: {outcome}")
        return "\n".join(lines)

    def _probe_write(self, ids, vectors, attrs) -> str:
        """Try one sized write and describe what happened, without raising.

        Coerces like its two siblings: `tests/test_config.py` calls this and
        `_diagnose_stalled_write` directly with a plain list of tag values,
        and the diagnosis must not be the thing that raises.
        """
        attrs = RowAttrs.coerce(attrs)
        try:
            probe = self._write_connection(timeout_s=CANARY_TIMEOUT_S)
        except Exception as exc:
            return f"could not connect: {type(exc).__name__}: {exc}"
        try:
            started = time.time()
            pipe = probe.pipeline(transaction=False)
            for i, key_id in enumerate(ids):
                pipe.hset(f"{PREFIX}{int(key_id)}", mapping={
                    **attrs.fields(i),
                    VECTOR_FIELD: encode_vector(vectors[i]),
                })
            pipe.execute()
            return f"OK in {(time.time() - started) * 1000:.0f} ms"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            try:
                probe.close()
            except Exception:
                pass

    def count_rows(self) -> int:
        info = self._ft_info()
        try:
            return int(info.get("num_docs", 0))
        except (TypeError, ValueError):
            return 0

    # -- query ----------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        """EF_RUNTIME is a modifier inside the query string, not a setting."""
        self._ef_runtime = int(ef_search)

    def _search(self, vector, k: int, filter_expr: str = "*") -> List[int]:
        query = (f"{filter_expr}=>[KNN {k} @{VECTOR_FIELD} $vec "
                 f"EF_RUNTIME {max(self._ef_runtime, k)}]")
        raw = self._conn.execute_command(
            "FT.SEARCH", INDEX, query,
            "PARAMS", "2", "vec", encode_vector(vector),
            "LIMIT", "0", str(k),
            # Keys only. Returning the vectors would put 6 KB per hit on the
            # wire and measure the network rather than the index.
            "NOCONTENT",
            "DIALECT", str(DIALECT),
        )
        ids = []
        for item in raw[1:]:
            key = item.decode() if isinstance(item, bytes) else str(item)
            if key.startswith(PREFIX):
                ids.append(int(key[len(PREFIX):]))
        return ids

    def _filter_expr(self, column: str, threshold: int) -> str:
        """`@column:[-inf (threshold]` -- the one place a predicate is built.

        Exclusive upper bound, matching `WHERE tag < threshold` in the SQL
        engines and `{"tag": {"$lt": ...}}` in Percona Search. The `(` before
        the upper bound is what makes it exclusive; without it this engine
        would be answering a different question from the other five at every
        selectivity, and by exactly one bucket or one cluster.

        The column goes through the whitelist here, which is the only place it
        can: `query_filtered` and `explain_uses_vector_index` both built this
        string themselves before, so a check added to one of them would have
        left the other interpolating whatever it was handed into a query
        string.
        """
        return f"@{checked_filter_column(column)}:[-inf ({int(threshold)}]"

    def query(self, vector, k: int) -> List[int]:
        return self._search(vector, k)

    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        return self._search(vector, k, self._filter_expr(column, threshold))

    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        """There is no plan to inspect and no scan to fall back to.

        FT.SEARCH is only answerable through the index; without one the command
        errors. What this distinguishes is an erroring configuration from a
        genuinely slow one, so the two are not both recorded as low throughput.

        It does NOT reliably catch the new failure this column introduces,
        and that is worth saying rather than assuming: an index created
        without the `ctag` field would answer `@ctag:[-inf (t]` with either
        an error or an empty result set, and only the error reaches here.
        That is exactly why the FT.CREATE schema and `_filter_expr` are both
        derived from `FILTER_COLUMNS` instead of relying on this check -- the
        empty-result case is indistinguishable from a selective predicate.
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
            if tag_threshold is None:
                self._search(vector, k)
            else:
                self._search(vector, k,
                             self._filter_expr(column, tag_threshold))
            return True
        except Exception:
            return False

    # -- sizing / metadata ----------------------------------------------

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

    def index_bytes(self) -> int:
        """Resident cost of the index, not a file size.

        Two measurements, and the larger wins. The used_memory delta across
        index creation covers the graph as well as the vectors but only where
        the build accounts index allocations to the main allocator, which
        Percona's does not: it reported 66 KB beside an 82 MB table. FT.INFO's
        user_indexed_memory is always reported but counts the indexed vectors
        rather than the graph built over them. Taking the larger reports the
        more complete of the two rather than whichever the build happens to
        support.
        """
        delta = max(0, self._bytes_after_index - self._bytes_before_index)
        return max(delta, self._indexed_memory())

    def table_bytes(self) -> int:
        """What the hashes themselves cost, index excluded."""
        if self._bytes_before_index:
            return self._bytes_before_index
        return max(0, self._used_memory() - self.index_bytes())

    def server_version(self) -> str:
        try:
            info = self._conn.info("server")
            version = info.get("valkey_version") or info.get("redis_version") or "?"
        except Exception:
            version = "unknown"
        return f"Valkey {version} / valkey-search"

    def wait_snapshot(self):
        """INFO commandstats and LATENCY LATEST. Valkey's command thread is
        single, so per-command service time is the closest thing it has to a
        wait profile: a window dominated by one command's usec is that
        command serializing everything behind it."""
        if self._conn is None:
            return None, "no connection"
        return valkey_snapshot(self._conn)

    def wait_delta(self, before, after):
        return valkey_delta(before, after)

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update({
            "ef_construction_tunable": True,
            "in_memory_only": True,
            # Documented but not observed -- see config/engines/valkey.yml.
            # None rather than True: this lands in the records, and a True
            # there reads as something the run measured.
            "hybrid_filter_planner": None,
            "backfill_seconds": round(self._backfill_seconds, 3),
        })
        return caps
