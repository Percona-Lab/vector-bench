"""The reader pool that runs while the churn writer re-inserts (PR 4 §4.3).

Its own module because inlining it in `churn.py` breaches this repo's stated
800-line maximum badly;
`tests/test_t4b_readers.py::test_the_line_count_the_split_is_justified_by_is_the_real_one`
splices the two back together, computes by how much, and fails if that stops
being true. The figure is not repeated here: it was stated in this docstring
and in `churn.py`'s, went stale three times in two days, and the number was
never the argument -- "badly over the ceiling" is.

The other half of the reason is cohesion: one caller each, no state, and
everything here is about one question: what does a read cost while this engine
is taking writes?

The answer is published as a ratio, `p99 under writes / baseline p99`, and that
shape is the reason this module is as careful as it is. Two ratios have already
been published wrong in this repository. The filter-strategy ratio divided a
narrow point from one ops unit by a wide point from another, reporting 50.0x
"post-filter, exhaustive" where the two units separately measured 0.20x and
1.00x. `charts.churn_retention` divides a post-churn QPS by a pre-churn one
without distinguishing the maintenance state, so the un-maintained and the
repaired measurement land at the same x and cannot be told apart by position.
Both read as findings and both were bugs.

So the two percentiles in this ratio are measured by one pool configuration
(`clients.PoolSpec`, which carries the client count, the query pool, k, the ef
and the ops unit), divided by `clients.window_ratio`, which refuses outright if
the two windows disagree on any of those, and stamped into ONE record. Nothing
downstream has to pair two records, so nothing downstream can pair the wrong
two.

What the numbers here are NOT evidence about: `clients.CLIENT_BOUND_UTIL` is
sufficient and not necessary -- a CPU-utilisation threshold cannot see a
ceiling that costs no CPU, and the interpreter lock that used to be one is out
of the path now that the readers are processes. Measured on the RETIRED thread
pool against `tests/fake_driver.py`: its own CPU flattened at 2.06 cores while
throughput peaked at two clients and halved by thirty-two, with the flag
firing on none of it. A reader pool's QPS is this harness against this engine
under writes and is not evidence about where either one saturates.

**AND THE WRITE RATE DEPENDS ON THE READERS HAVING ACTUALLY RUN.**
`extra.reinsert_rows_per_s` is a version-5 quantity precisely because it is
now measured against `ops.churn_readers` concurrent reads. Until version 6
those reads were THREADS IN THE WRITER'S OWN INTERPRETER, so readers that
serialized on the interpreter lock starved the writer in the CLIENT and the
rate stopped being about the engine's ingest at all. The live `smoke` run
that exposed it recorded a writer at 12.0 rows/s -- 2,000 rows in 166.8 s --
with four readers at 2,819 q/s, against a modelled MariaDB ingest of about
150 rows/s.

**Each reader is a forked OS process now and the writer is the harness's
own, so that mechanism is out of the measurement path.** What is NOT out of
it is the dependency: the rate is still taken against whatever those readers
actually did, so a reader pool that served one read at a time still leaves
the published figure describing an unknown amount of read load. The two-arm
measurement below is kept because it is what established the size of the
effect, and both arms were taken on the retired THREAD pool -- it is
history, not a current reading of this pool. Measured on
`tests/fake_driver.py` with a 4 ms write slice and four readers whose query
cost is IDENTICAL in both arms, driving the real `churn.run`: readers that
release the lock put the published `reinsert_rows_per_s` at 948.2, and
readers that hold it put the same figure at 41.1 -- a factor of 23. Their
`served_concurrency` was 3.91 and 1.07 of four, while the exact
`outstanding_concurrency` was 3.96 and 3.95: blind to the whole difference,
which is why the status is keyed to the other one.

So every reader window here records `outstanding_concurrency`,
`served_concurrency` and `scaling_status` (`clients.PoolWindow`), the
during-writes record carries the during window's status unprefixed beside
`client_bound`, and the step record that publishes `reinsert_rows_per_s`
carries it as `extra.reader_scaling_status` -- see `reader_contention_shape`.
That last one is the point of the exercise: the published write rate cannot be
read without the status of the read load it was measured against.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy

from ..drivers.base import IndexSpec
from .clients import (CLIENT_BOUND_UTIL, SCALING_CONCURRENCY_NOT_ACHIEVED,
                      SERVED_CONCURRENCY_FLOOR, ClientPool, DriverFactory,
                      PoolWindow, window_ratio)

#: Readers running while the writer re-inserts, per PR 4 §4.3 item 2.
#: `ops.churn_readers: 0` disables them and restores exactly the write-only
#: re-insert rate the report showed before this existed -- which §4.3 item 7
#: asks be kept available, because a rate measured against four concurrent
#: readers and one measured against none are two different quantities.
DEFAULT_CHURN_READERS = 4

#: Seconds of reader traffic before each step's deletes, which is the
#: denominator of §4.3 item 5's headline ratio. Thirty seconds because that is
#: what the specification asks for and because it matches `main`'s concurrency
#: window, so a reader comparing the two is comparing windows of one length.
#:
#: BEFORE the deletes, not before the whole workload: for the second and later
#: fractions the corpus already carries the previous step's churn, and a
#: baseline taken once at the start would make every later ratio partly a
#: measurement of accumulated churn rather than of concurrent writes.
READER_BASELINE_WINDOW_S = 30.0

#: Queries each reader answers before the barrier releases. The same 20
#: `_measure` uses, for the same reason: cold-cache cost belongs to neither
#: window, and both halves of a ratio have to be equally warm.
READER_WARMUP_QUERIES = 20

#: The ops-unit axes, which both halves of the headline ratio must share.
#:
#: Declared here rather than imported from `report.generate.OPS_UNIT_FIELDS`,
#: and that is not duplication by accident: `harness/` runs in a container
#: that mounts `harness/` alone, so importing `report/` raises
#: ModuleNotFoundError seconds after the server comes up. The same reason
#: `orchestrator/ops_pass.py` re-declares the ef policy and the filtered
#: budget. `tests/test_t4b_readers.py` pins the two declarations together, as
#: `tests/test_calibrate.py` already does for the ef policy.
#:
#: Why a ratio needs them at all: this repository has published two wrong
#: ratios of exactly this shape. The filter-strategy ratio divided a narrow
#: point from one ops unit by a wide point from another -- 0.20x and 1.00x
#: separately against a pooled 50.0x that neither unit produced -- and
#: `charts.churn_retention` pairs records from two different maintenance
#: states. Both read as findings.
OPS_UNIT_KEYS = ("m", "build_mode", "storage_engine", "resource_pass")


def _reader_count(readers: int, queries: numpy.ndarray,
                  driver_factory: Optional[DriverFactory],
                  name: str) -> int:
    """How many readers this configuration can actually run, and why not more.

    Zero means "no reader pool", and there are three honest ways to get there.
    The two that were not asked for are printed rather than left to be
    inferred from a missing record; the one that was asked for is silent,
    because an operator who wrote `ops.churn_readers: 0` does not need to be
    told what they configured:

    * `ops.churn_readers: 0`, which §4.3 item 7 keeps available on purpose --
      it is the only way to measure the write-only re-insert rate the report
      showed before this existed. `_reader_shape` still records `readers: 0`
      on the step record, so the choice is in the JSONL even though nothing
      is printed.
    * No `driver_factory`. Readers need their own connections, so a caller
      that has only the writer's driver cannot have them. `harness/main.py`
      always passes one; the diagnostic callers in `tests/` do not.
    * Fewer queries than readers. `clients.shard_queries` hands client *c* the
      strided slice `queries[c::readers]`, which is EMPTY once *c* exceeds the
      pool size, and a client with an empty shard divides by zero on its first
      iteration. The concurrency workload skips such a point for the same
      reason.
    """
    if readers <= 0:
        return 0
    if driver_factory is None:
        print(f"[churn] {name}: no reader pool -- the caller passed no "
              f"driver_factory, so there is nothing to open reader "
              f"connections with")
        return 0
    if readers > len(queries):
        print(f"[churn] {name}: no reader pool -- {readers} reader(s) "
              f"requested but only {len(queries)} quer(ies) available, and a "
              f"reader with an empty query shard cannot run")
        return 0
    return readers


def _reader_shape(driver_factory: Optional[DriverFactory], requested: int,
                  resolved: int) -> Dict[str, Any]:
    """What read load the re-insert rate beside this was measured against.

    Read through the same convention `churn._write_shape` uses: a caller with
    nothing to explain carries no key. Here that caller is one with no
    `driver_factory` -- `tests/test_config.py` and `tests/test_t3b_drivers.py`
    drive this workload with a writer driver alone, so they could not have had
    readers under any setting, and their record is the pre-PR-4 record
    unchanged. `tests/test_t4a_churn.py` pins that exact key set, and a
    `readers: 0` written onto it would be this change claiming to have
    measured something about a run that had no reader pool to measure.

    A real run always has a factory, so a real record always states this --
    including `readers: 0`, which is what §4.3 item 7 needs: the write-only
    re-insert rate has to be identifiable as such rather than inferred from a
    missing key.

    `readers_requested` appears only when the request was reduced, which
    happens when the query pool is smaller than the reader count. Without it,
    `readers: 0` would mean both "the profile turned them off" and "the
    profile asked for four and the pool was too small", and only the first of
    those is a choice someone made.
    """
    if driver_factory is None:
        return {}
    shape: Dict[str, Any] = {"readers": resolved}
    if requested != resolved:
        shape["readers_requested"] = requested
    return shape


def reader_contention_shape(window: Optional[PoolWindow]) -> Dict[str, Any]:
    """What read load the step record's `reinsert_rows_per_s` was measured
    against, as a status rather than only as a reader count.

    `{}` for a step with no during-writes window, and that follows
    `_reader_shape`'s convention exactly: a caller with nothing to say carries
    no key. A `reader_scaling_status: null` written onto a record from a run
    that had no reader pool would be this change claiming to have looked at
    something that never existed, and `tests/test_t4a_churn.py` pins the key
    set those records have.

    Public, unlike its neighbours, because `churn.py` is a different module
    and this is the one key of the four that has to reach a record
    `churn_readers.py` does not itself write. The write rate is published on
    the step record; the status that decides whether it means anything is
    measured on the reader pool.
    """
    if window is None:
        return {}
    # Ops measurement version 9: the status is judged on the exact
    # mean-based figure (`PoolWindow.scaling_status` passes `outstanding`),
    # so that figure travels with it, and a heavy-tailed read window is the
    # `tail_dominated` note rather than a refusal of the writer rate.
    return {
        "reader_outstanding_concurrency": (
            round(window.outstanding_concurrency, 3)
            if window.outstanding_concurrency is not None else None),
        "reader_served_concurrency": (round(window.served_concurrency, 3)
                                      if window.served_concurrency is not None
                                      else None),
        "reader_scaling_status": window.scaling_status,
        "reader_scaling_note": window.scaling_note,
    }


def _reader_pool(driver_factory: DriverFactory, queries: numpy.ndarray,
                 readers: int, k: int, ef_search: int, index: IndexSpec,
                 unit: Tuple[Any, ...]) -> ClientPool:
    """A reader pool, built identically for both halves of the ratio.

    One function because `clients.PoolSpec` equality is what `window_ratio`
    checks before dividing, and two construction sites would be two chances
    for the halves to disagree about the client count, the query pool, k or
    the ef -- which is how a ratio of two things that never ran together gets
    published as a finding.

    `collect_results=False`: §4.3 item 6 says the readers compute no recall,
    because the corpus is mid-change and no ground truth describes it. The
    `recall_at_k` column on the during-writes record is therefore null by
    design, the way T3c made it null for a filtered point that managed no
    queries.
    """
    return ClientPool(driver_factory, queries, readers, k, ef_search, index,
                      READER_WARMUP_QUERIES, collect_results=False, unit=unit)


def _reader_window(driver_factory: DriverFactory, queries: numpy.ndarray,
                   readers: int, k: int, ef_search: int, index: IndexSpec,
                   unit: Tuple[Any, ...], label: str, prefix: str,
                   duration_s: float) -> PoolWindow:
    """One fixed-duration reader window, start to finish.

    Used for the baseline half of §4.3 item 5's ratio. The during-writes half
    cannot use it, because its duration is "until `_reinsert` returns" and the
    writer has to run between the pool's `start` and its `stop`.
    """
    pool = _reader_pool(driver_factory, queries, readers, k, ef_search, index,
                        unit)
    pool.start(label, prefix=prefix)
    time.sleep(duration_s)
    return pool.stop()


def _reader_extra(window: PoolWindow, prefix: str) -> Dict[str, Any]:
    """One reader window's figures, under a named prefix.

    Both halves of the ratio are described by the same keys with a different
    prefix, so a reader of the raw JSONL can see that the numerator and the
    denominator were measured the same way rather than having to trust it.
    """
    merged = window.merged
    return {
        f"{prefix}_qps": round(window.qps, 2),
        f"{prefix}_p50_ms": merged.get("latency_p50_ms"),
        f"{prefix}_p95_ms": merged.get("latency_p95_ms"),
        f"{prefix}_p99_ms": merged.get("latency_p99_ms"),
        f"{prefix}_queries_executed": window.queries_executed,
        f"{prefix}_window_s": round(window.wall_s, 3),
        f"{prefix}_inflight_discarded": window.inflight_discarded,
        f"{prefix}_prewindow_discarded": window.prewindow_discarded,
        # How many reader queries this window actually had outstanding and in
        # service. Under BOTH prefixes, because the ratio divides a window in
        # which the readers may have served four queries by one in which they
        # may have served one, and a reader of the JSONL cannot see that from
        # the percentiles: a pool that serialized measures a p99 of a query
        # that waited for a lock, not of a query the engine was slow to
        # answer. See `clients.SERVED_CONCURRENCY_FLOOR`.
        f"{prefix}_achieved_concurrency": (
            round(window.outstanding_concurrency, 3)
            if window.outstanding_concurrency is not None else None),
        f"{prefix}_served_concurrency": (
            round(window.served_concurrency, 3)
            if window.served_concurrency is not None else None),
        f"{prefix}_scaling_status": window.scaling_status,
        f"{prefix}_scaling_note": window.scaling_note,
    }


def _during_writes_extra(baseline: PoolWindow, during: PoolWindow,
                         rows_per_s: Optional[float],
                         rows_written: int) -> Dict[str, Any]:
    """The `during_writes` record's `extra`, including the headline ratio.

    **The ratio is computed here, from two windows of one pool, and stamped
    into one record.** That is the structural answer to the shape that has
    produced two published-wrong results in this repository: nothing
    downstream has to pair two records, so nothing downstream can pair the
    wrong two. `clients.window_ratio` refuses outright if the two windows
    disagree on client count, query pool, k, ef_search or the ops unit, and
    `extra.reader_pool` records that agreed identity so a reader of the JSONL
    can check it rather than take it on trust.

    The ratio is `None` rather than a number when either percentile is absent
    or zero. A zero denominator is not a measurement, and
    `tests/fake_driver.py` returns a latency of exactly 0.0 unless it is given
    one to spend -- so a ratio computed over it would be an arithmetic
    accident presented as a slowdown figure.

    `reader_p99_ratio` is deliberately NOT rounded the way most numbers here
    are. It is a ratio of two wall-clock percentiles, and on the fake the same
    configuration's churn QPS has been measured between 6,028 and 16,811 -- a
    factor of 2.79 -- so its decimals are not a claim this code can make. What
    it IS, and what the test suite pins, is that the two percentiles came from
    one pool, one ops unit and one configuration.
    """
    spec = during.spec
    extra: Dict[str, Any] = {
        "during_writes": True,
        "readers": spec.clients,
        "reader_queries": spec.queries,
        "reader_warmup_queries": READER_WARMUP_QUERIES,
        # The agreed pool identity, as a dict so the key names survive into
        # the JSONL. `window_ratio` has already refused if the two halves
        # differ on any of it. The ops unit is split out below rather than
        # left as a positional tuple, because a reader of the JSONL cannot
        # tell which axis a bare 4-tuple's third element is.
        "reader_pool": dict(zip(spec._fields[:-1], spec[:-1])),
        "reader_ops_unit": dict(zip(OPS_UNIT_KEYS, spec.unit)),
        "reader_p99_ratio": window_ratio(during, baseline, "latency_p99_ms"),
        # Unrounded, for the same reason `reader_p99_ratio` is: it is a ratio
        # of two wall-clock throughputs and its decimals are the box's. An
        # earlier revision rounded this one to 4 dp and left the other alone,
        # which implied the two were different kinds of number.
        "reader_qps_ratio": (during.qps / baseline.qps
                             if baseline.qps else None),
        # What the writer achieved while the readers were running. §4.3 item
        # 7: this is no longer the write-only rate, and the key says so in its
        # own name rather than leaving `reinsert_rows_per_s` to mean two
        # different things depending on `ops.churn_readers`.
        "writer_rows_per_s_with_readers": (round(rows_per_s, 3)
                                           if rows_per_s is not None
                                           else None),
        "writer_rows": rows_written,
        # Whose ceiling the reader figures show. The readers are forked OS
        # processes sharing one container's cpuset, and
        # `clients.CLIENT_BOUND_UTIL` is sufficient and not necessary: an
        # unmarked point is not evidence that the engine was the limit.
        "client_cpu_seconds": (round(during.client_cpu_seconds, 3)
                               if during.client_cpu_seconds is not None
                               else None),
        "client_cpus": during.client_cpus,
        "client_cpu_util": (round(during.client_cpu_util, 4)
                            if during.client_cpu_util is not None else None),
        "client_bound": during.client_bound,
        # Whether the readers ran the queries their count implies, over the
        # DURING-WRITES window -- the one every published figure on this
        # record is taken from. Unprefixed, beside `client_bound`, because
        # this is the record's own status the way `client_bound` is; the two
        # prefixed copies in `_reader_extra` are the per-window evidence.
        #
        # **This decides whether `writer_rows_per_s_with_readers` means
        # anything**, and it is the version-5 quantity PR 4 §4.3 item 7
        # introduced. The rate is measured against whatever these readers
        # actually did, so a pool that served one read at a time leaves the
        # figure describing an unknown amount of read load. Until version 6
        # there was a second and larger mechanism: the writer ran in the
        # harness's MAIN thread while the readers ran as threads in the SAME
        # interpreter, so readers serializing on the interpreter lock starved
        # the writer inside the client and the rate became a measurement of
        # Python. The live `smoke` run that exposed that recorded 12.0 rows/s
        # (2,000 rows in 166.8 s) with four readers at 2,819 q/s, against a
        # modelled MariaDB ingest of about 150 rows/s. Each reader is a
        # forked OS process now, so only the first mechanism is left.
        "outstanding_concurrency": (
            round(during.outstanding_concurrency, 3)
            if during.outstanding_concurrency is not None else None),
        "served_concurrency": (round(during.served_concurrency, 3)
                               if during.served_concurrency is not None
                               else None),
        "scaling_status": during.scaling_status,
        "scaling_note": during.scaling_note,
    }
    extra.update(_reader_extra(baseline, "reader_baseline"))
    extra.update(_reader_extra(during, "reader_under_writes"))
    return extra


def warn_empty_windows(name: str, fraction: float, baseline: PoolWindow,
                       during: PoolWindow) -> None:
    """Announce either window that completed nothing, before the record.

    Both halves, because either one empty makes the ratio `None` and the
    reason differs: an empty baseline means the 30 s window could not fit one
    query, an empty during-writes window means the re-insert outran the
    readers.
    """
    if not baseline.queries_executed:
        _warn_empty_reader_window(name, fraction, "30 s baseline",
                                  baseline.wall_s)
    if not during.queries_executed:
        _warn_empty_reader_window(name, fraction, "during-writes",
                                  during.wall_s)


def _print_during_writes(name: str, fraction: float,
                         extra: Dict[str, Any]) -> None:
    """The operator's line for one during-writes point.

    Takes the assembled `extra` rather than the two windows, so the line a
    reader sees on stderr during the run and the record they read a day later
    are the same numbers by construction -- not two formattings of one
    measurement that can drift apart.
    """
    ratio = extra.get("reader_p99_ratio")
    ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
    print(
        f"[churn] {name}: {extra['readers']} reader(s) during the "
        f"{fraction:.0%} re-insert: qps={extra['reader_under_writes_qps']:,.1f} "
        f"(baseline {extra['reader_baseline_qps']:,.1f}) "
        f"p99={_reader_ms(extra.get('reader_under_writes_p99_ms'))} "
        f"(baseline {_reader_ms(extra.get('reader_baseline_p99_ms'))}) "
        f"p99 ratio={ratio_text}; writer "
        f"{extra['writer_rows_per_s_with_readers'] or 0:,.1f} rows/s",
        flush=True)
    if extra.get("client_bound"):
        print(
            f"[churn] {name}: CLIENT CPU-BOUND: the {extra['readers']} "
            f"reader(s) used {extra['client_cpu_util']:.0%} of their "
            f"{extra['client_cpus']} CPU(s), above the "
            f"{CLIENT_BOUND_UTIL:.0%} limit. The reader throughput at this "
            f"point is bounded by the readers as well as by the engine, so "
            f"it is not evidence about what the writes cost the engine.",
            file=sys.stderr, flush=True)
    if extra.get("scaling_status") == SCALING_CONCURRENCY_NOT_ACHIEVED:
        _warn_readers_serialized(name, fraction, extra)


def _warn_readers_serialized(name: str, fraction: float,
                             extra: Dict[str, Any]) -> None:
    """Say so when the reader pool never ran the queries its count implies.

    Separate from `_print_during_writes` because it is two paragraphs about
    two different published figures and inlining it pushed that function past
    what this repository asks of one.

    The write rate is named as well as the read figures, and that is the point
    of this warning rather than an extra: it was measured against these reads,
    so a pool that served one read at a time leaves it describing an unknown
    amount of read load. The shape it is named for was measured when the
    readers were threads in the SAME interpreter as the writer and serializing
    readers starved the writer in the client -- the 12.0 rows/s the live
    `smoke` run recorded against four readers, where the modelled ingest is
    about 150 rows/s. Each reader is a forked OS process now, so the
    attribution is weaker than that run's was and the dependency is not.
    `extra.reinsert_rows_per_s` on this fraction's step record is the figure
    the report publishes, and it carries this status too
    (`extra.reader_scaling_status`) so it cannot be read without it.
    """
    served = extra.get("served_concurrency")
    served_text = f"{served:.2f}" if served is not None else "n/a"
    outstanding = extra.get("outstanding_concurrency")
    outstanding_text = (f"{outstanding:.2f}" if outstanding is not None
                        else "n/a")
    readers = extra.get("readers") or 0
    print(
        f"[churn] {name}: READERS DID NOT ACHIEVE CONCURRENCY at "
        f"{fraction:.0%} churn: {readers} reader(s) had {outstanding_text} "
        f"quer(ies) outstanding on average ({served_text} by the median "
        f"estimate), below the {SERVED_CONCURRENCY_FLOOR:.0%} of {readers} "
        f"these figures need: the readers did not keep their reads in "
        f"flight. "
        f"The p99 ratio and the reader throughput on this record are not "
        f"evidence about what the writes cost the engine, because the reads "
        f"were not running concurrently. The writer rate beside them "
        f"({extra.get('writer_rows_per_s_with_readers') or 0:,.1f} rows/s) "
        f"was measured against those same reads, so it is not a figure about "
        f"this engine under a known read load either. Until "
        f"`OPS_MEASUREMENT_VERSION` 6 the readers were threads in the "
        f"writer's own interpreter and reads that serialized on the "
        f"interpreter lock would starve the writer inside the client rather "
        f"than inside the engine; each reader is a separate OS process now, "
        f"so that mechanism is gone and the dependency is not.",
        file=sys.stderr, flush=True)


def _reader_ms(value: Optional[float]) -> str:
    """A reader percentile for the operator's line, or `n/a`.

    None arrives from a window that completed no query inside it, which is the
    one case with no percentile to print; formatting it with `:.3f` raises.
    """
    return f"{value:.3f}ms" if value is not None else "n/a"


def _warn_empty_reader_window(name: str, fraction: float, which: str,
                             window_s: float) -> None:
    """Say so when a reader window completed no query at all.

    The same check and the same voice as `concurrency._warn_nothing_finished`,
    and it exists for the reason `clients.client_cpu` gives for refusing to
    report a zero: "Zero is the claim 'the clients did nothing', which is the
    one thing that cannot be true of a window that executed queries." A
    during-writes window shorter than one reader query records a QPS of 0 and
    no percentiles, and without this line the table published that 0 beside a
    real baseline with nothing to say it meant "no window".

    Reachable on no shipped profile -- `main` and `smoke` re-insert tens of
    thousands of rows and the pool runs for all of it -- and reachable at once
    on a synthetic driver whose write path returns instantly, which is how it
    was found: three runs of three at a 1 ms re-insert gave
    `Reader QPS: 186 -> 0` with no diagnostic anywhere.
    """
    print(
        f"[churn] {name}: NOTHING FINISHED: no reader query completed inside "
        f"the {which} window at {fraction:.0%} churn; it was open for "
        f"{window_s * 1000:.1f} ms. The record carries a reader QPS of 0 and "
        f"no percentiles, because there is nothing to report one over, and "
        f"the report renders those cells blank rather than as a zero. A "
        f"during-writes window is as long as the re-insert takes, so this "
        f"means the write finished before a single read did.",
        file=sys.stderr, flush=True)


def _warn_reader_pool_failed(name: str, fraction: float,
                             exc: BaseException) -> None:
    """Say so when the reader pool failed, and say what survives it.

    §4.4 asks that a reader exception fail the record instead of hanging. It
    fails THAT record and not the workload, which is the same decision
    `_record_maintenance` documents below: the post-churn measurement is a
    finished, unaffected result, and a repair path -- or a reader pool -- that
    fails must not take it with it. `_reinsert` makes the same choice when a
    write path dies.

    The "before" and "after" records still carry `extra.readers`, so a reader
    of the JSONL can see that the re-insert rate beside them was measured
    against concurrent readers that then died, rather than against none.
    """
    print(
        f"[churn] {name}: READER POOL FAILED at {fraction:.0%} churn: {exc}. "
        f"No during-writes record is written for this fraction -- a reader "
        f"QPS from a pool whose clients died reads as the engine having been "
        f"slow under writes, which is the one thing it is not evidence of. "
        f"The post-churn measurement below is unaffected and still runs.",
        file=sys.stderr, flush=True)
