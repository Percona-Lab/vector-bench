"""The churn workload's write side: parallel re-insert, index-sync waits,
and the index-size provenance a churn record carries.

Its own module for the reason `churn_readers.py` is one: `churn.py` sat at
780 lines of this repository's 800-line ceiling before ops measurement
version 9 gave the re-insert a writer pool and the sequence two more waits.
`churn.py` keeps `_reinsert`'s budgeted loop, which several tests read by
source; what moved here is the machinery around it.

WHAT VERSION 9 CHANGED, and the run that made each change necessary
(`full-20260825`):

* **The re-insert ran ONE writer against an EIGHT-thread load**, and the
  report divided one rate by the other as "re-insert vs load". That ratio
  compared a write path with a thread pool, not an index before and after
  churn. `parallel_reinsert` splits the slices over `writers` connections,
  defaulting to the run's ingest `load_threads`, and the record carries both
  counts so the report can withhold the ratio when they differ.
* **Percona Search (engine `mongodb`) was measured without waiting for its
  asynchronous mongot index** before the baseline and after the deletes, so
  the baseline and the post-delete state were whatever mongot had absorbed by
  then. `synced_wait` runs the driver's wait at all three points and records
  each wait's duration; the driver's own `expect_index_state` hint tells a
  mongot wait WHICH document to look for (a deleted id until it stops coming
  back, the newest id until it appears).
* **The accepted rate hid the indexed one.** `rows / insert_seconds` is when
  the server ACCEPTED the rows; for an asynchronous index the rows are not
  searchable until the wait returns. Both rates are recorded; for a
  synchronous engine the wait is ~0 and the two agree.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..datasets import RECALL_DISTANCE_THRESHOLD
from ..drivers.base import RowAttrs
from ..progress import Heartbeat
from .index_size import index_size_shape  # noqa: F401 - re-exported


def _slices(total: int, chunk: int) -> List[Tuple[int, int]]:
    return [(begin, min(begin + chunk, total))
            for begin in range(0, total, chunk)]


def parallel_reinsert(driver, ids: Sequence[int], vectors, attrs,
                      budget_s: float, writers: int,
                      factory: Callable[[], Any], index: Any = None
                      ) -> Tuple[int, float]:
    """Re-insert over `writers` connections, each taking whole slices.

    The same slicing and the same budget rule as `churn._reinsert` -- fifty
    slices, a deadline tested before each one -- so the budget's meaning does
    not change with the writer count. Each writer is a fresh driver from
    `factory`, connected and handed the index spec exactly as a concurrency
    client is (`client_process.Client`), because the MySQL family's own
    `load(threads=)` opens one connection per thread for the same reason: one
    connection cannot carry two statements at once.

    Returns (rows written, seconds). Rows written is the sum of the slices
    that COMPLETED, which with several writers need not be a prefix -- the
    caller only uses it to decide whether the re-insert finished, and records
    the count.

    Anything a writer driver learned that the main driver's later calls need
    is copied back -- see `INHERITED` for which attributes, merged how, and
    why never through a property.
    """
    attrs = RowAttrs.coerce(attrs)
    total = len(ids)
    chunk = max(1, total // 50)
    pending = _slices(total, chunk)
    started = time.time()
    done: List[int] = []
    failures: List[str] = []
    workers: List[Any] = []

    def write(worker) -> None:
        while True:
            if failures or time.time() - started > budget_s:
                return
            try:
                begin, end = pending.pop(0)
            except IndexError:
                return
            try:
                worker.insert_rows(ids[begin:end], vectors[begin:end],
                                   attrs[begin:end])
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                return
            done.append(end - begin)

    try:
        # Opened inside the `try`, so a connect that fails for the third
        # writer still closes the two already open.
        for _ in range(writers):
            worker = factory()
            workers.append(worker)
            worker.connect()
            if index is not None:
                worker._index = index
        with Heartbeat(f"re-inserting {total:,} rows over {writers} "
                       f"writer(s)", prefix=driver.name):
            with ThreadPoolExecutor(max_workers=writers) as pool:
                list(pool.map(write, workers))
    finally:
        # Each worker in its own try/finally: a merge that raised used to
        # skip every later worker's `close()` and replace the result the
        # rows had already been written for -- the step then wrote no record
        # at all.
        for worker in workers:
            try:
                _inherit(driver, worker)
            except Exception as exc:
                print(f"[churn] {driver.name}: could not copy a writer's "
                      f"state back: {type(exc).__name__}: {exc}")
            finally:
                try:
                    worker.close()
                except Exception:
                    pass
    if failures:
        print(f"[churn] {driver.name}: re-insert stopped after "
              f"{sum(done):,} rows: {failures[0]}")
    return sum(done), time.time() - started


#: The writer-side state the main driver's LATER calls read, and how several
#: writers' values merge. Audited against every real driver class
#: (`MySQLFamilyDriver` and its MariaDB, AliSQL, Percona and MyVector
#: subclasses, `PostgresDriver`, `MongoDriver`, `ValkeyDriver`): these are the
#: only two such attributes, each is a plain instance attribute of exactly one
#: class, and neither is a property or a slot -- which
#: `tests/test_ops_measurement_v9.py` re-checks over the registry.
#:
#: * `_newest_inserted_id` (mongot's sync probe): MAX -- the newest row any
#:   writer wrote is the one the wait must find.
#: * `_write_batch` (Valkey's re-insert batch, published through the
#:   READ-ONLY property `write_batch_used`): MIN -- the reduction is the fact
#:   `_write_shape`'s `write_batch_reduced` exists to report, and a max over
#:   the main driver's untouched 1,000 would erase a writer that fell back to
#:   one row per write. The first version of this merged through the
#:   property, which raised AttributeError after the rows were written.
#:
#: NOT inherited, deliberately: Valkey's `_probed` and `_largest_working`.
#: They are the stall ladder's own bookkeeping -- whether this connection
#: has already run `_diagnose_stalled_write`, and the largest size it found
#: returning -- and the main driver never reads them after a re-insert: its
#: next writes start from `_write_batch`, which IS inherited (as the min), so
#: the ladder's conclusion survives even though its scratch state does not.
#: Safe for as long as nothing outside `insert_rows` reads either one.
INHERITED = {"_newest_inserted_id": max, "_write_batch": min}


def _inherit(driver, worker) -> None:
    """Copy back the per-connection state the main driver's later calls read.

    Read and written through the instance dictionaries, never through
    `getattr`/`setattr` on a name that could be a property: an attribute the
    worker never set is skipped, and one the main driver does not hold as a
    plain attribute is refused rather than shadowed.
    """
    theirs_all = vars(worker)
    mine_all = vars(driver)
    for name, merge in INHERITED.items():
        theirs = theirs_all.get(name)
        if theirs is None:
            continue
        if isinstance(getattr(type(driver), name, None), property):
            raise TypeError(f"{type(driver).__name__}.{name} is a property; "
                            f"refusing to write through it")
        mine = mine_all.get(name)
        mine_all[name] = theirs if mine is None else merge(mine, theirs)


def synced_wait(driver, expected_rows: int, *,
                absent: Optional[Tuple[int, Any]] = None,
                present: Optional[int] = None) -> float:
    """`driver.wait_index_synced(expected_rows)`, told what to look for.

    `absent` is (id, vector) of a row just deleted, `present` the id of a row
    that must be searchable. Handed over through the optional
    `expect_index_state` hook -- only Percona Search's driver has one -- so
    every other driver's `wait_index_synced` keeps the one-argument
    signature it has always had.
    """
    hint = getattr(driver, "expect_index_state", None)
    if callable(hint):
        hint(absent=absent, present=present)
    return float(driver.wait_index_synced(expected_rows) or 0.0)


def indexed_rate(rows: int, insert_seconds: float,
                 sync_seconds: float) -> Optional[float]:
    """Rows per second until the rows were SEARCHABLE, not just accepted."""
    span = insert_seconds + max(0.0, sync_seconds)
    return round(rows / span, 3) if span > 0 else None


def _columns(measured: Dict[str, float]) -> Dict[str, float]:
    """`_measure`'s output minus the keys that belong in `extra`."""
    return {k: v for k, v in measured.items() if k != "recall_id_overlap"}


def _recall_shape(measured: Dict[str, float]) -> Dict[str, object]:
    """Which rule `recall_at_k` is and the other rule's figure (ops v9).

    Keyed off whether `_measure` scored by distance: a caller with no corpus
    (none in this file) would get the id overlap and must not be labelled as
    the threshold.
    """
    overlap = measured.get("recall_id_overlap")
    if overlap is None:
        return {}
    return {"recall_kind": RECALL_DISTANCE_THRESHOLD,
            "recall_id_overlap": overlap}


def _sync_shape(driver) -> Dict[str, object]:
    """Why `index_sync_s` came out the way it did, when the driver says.

    Optional, read through `getattr` exactly as `_write_shape` below reads
    Valkey's write batch: the engines that maintain their index on the write
    path have nothing to explain and carry no key.

    It exists because a seconds-valued hook returning 0.0 is ambiguous in the
    one direction that matters. `index_sync_s: 0.0` is what a synchronous
    engine records, what a hook nobody implemented records, and what a mongot
    that was already caught up records -- and H1's acceptance criterion is
    "non-zero or a reason documented", which a bare zero cannot satisfy.

    A driver that also publishes `index_sync_detail` gets that carried too,
    as a second key. The note is prose, so reading it back means matching a
    sentence; the detail is the same facts as fields -- the final indexed
    count, the count expected, the polls, which condition ended the wait and
    which signal was binding -- which is what makes "did this engine's index
    actually arrive?" a query over a run rather than a careful read.
    """
    note = getattr(driver, "index_sync_note", None)
    detail = getattr(driver, "index_sync_detail", None)
    out = {"index_sync_note": note} if note else {}
    if detail:
        out["index_sync_detail"] = detail
    return out


def _write_shape(driver) -> Dict[str, object]:
    """How the driver had to write, when that is not how it normally writes.

    Valkey re-inserted 99,000 rows at 86 rows/s against a load that ran at
    30,195, and almost none of that gap is the HNSW graph: pipelined writes
    into an indexed prefix do not return at all, so the re-insert had to fall
    back to one command per round trip while the load pipelined a thousand.
    Comparing those two rates as though they measured the same operation would
    attribute to the index a cost that belongs to the write shape, so the shape
    travels with the number.
    """
    used = getattr(driver, "write_batch_used", None)
    if used is None:
        return {}
    shape = {"write_batch_rows": used}
    if used < getattr(driver, "load_batch_rows", used):
        shape["write_batch_reduced"] = True
    return shape


__all__ = ["index_size_shape", "indexed_rate", "parallel_reinsert",
           "synced_wait"]
