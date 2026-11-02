"""Filtered (hybrid) vector search: ANN combined with a scalar predicate.

This is the case integrated vector search is supposed to win — the reason to put
vectors in a relational database instead of a dedicated vector store — and it is
where the implementations diverge most visibly, because each has to decide
between pre-filtering, post-filtering, or iterating the graph until enough rows
qualify.

Five things make this measurement easy to get wrong, and all five are handled
here.

1. **Ground truth changes with the filter.** The correct top-k over the rows
   passing `tag < t` is not the unfiltered top-k. Scoring filtered results
   against unfiltered ground truth would report near-zero recall for every
   engine and say nothing. Filtered ground truth is computed exactly, by brute
   force over the qualifying rows, and cached per (dataset, metric, k,
   column, threshold, rows, tag_buckets, ctag_clusters, ctag_seed) — nine
   components, because every one of them changes the correct answer set and a
   key missing one silently scores a run against another configuration's
   targets. See `datasets.filtered_truth_cache_name`.

2. **One uncorrelated predicate measures half the problem.** `tag` is
   `row_id % 100`, so the qualifying rows are a uniform sample of the vector
   space and a query's true neighbours pass as often as anything else. `ctag`
   is a nearest-pivot cluster id, so `ctag < t` selects t whole clusters and
   for most queries the real nearest neighbours sit in a cluster that fails.
   Both columns are measured against the same graph, because `build.run` loads
   both in one pass; choosing one per run would cost a full reload, which is
   hours for MHNSW and VIDX at 1M rows.

3. **The recorded selectivity must be the one that was delivered.** See
   `_record_point`. This was wrong for the life of the project and the name
   hid it.

4. **Short result sets are not the same as wrong ones.** With post-filtering,
   an engine can exhaust its `ef_search` candidate list before finding k
   qualifying rows and return fewer than k. That is a real behaviour worth
   reporting, so the distribution of returned counts is recorded alongside
   recall rather than being silently padded.

5. **An unbounded query loop can outlive the run.** On MariaDB at 1%
   selectivity the measured rate was 0.9 QPS against 326 unfiltered, so 2,000
   queries cost 37 minutes for one point on one curve. With two columns and
   three selectivities that is a dimension that can quietly become the longest
   thing in the profile, which is why every point now runs against a budget.
"""

from __future__ import annotations

import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy

from ..datasets import (FILTER_COLUMNS, FILTER_TAG, Dataset, cached_ground_truth,
                        filtered_truth_for, recall_at_k, recall_per_query)
from ..datasets import (RECALL_DISTANCE_THRESHOLD,
                        recall_threshold_per_query)
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import LatencyCollector, LatencyStats
from ..metrics.records import PHASE_FILTERED, Record
from ..metrics.stats import ci_fields
from ..progress import Heartbeat, Progress
from .context import RunContext

DEFAULT_SELECTIVITIES = (0.01, 0.10, 0.50)

#: Both filter columns by default. PR 3 §3.3 item 1. `build.run` has loaded
#: both since T3b, so measuring only one would leave a column in the table
#: that nothing queries.
DEFAULT_COLUMNS: Tuple[str, ...] = FILTER_COLUMNS

#: Seconds per (column, selectivity, iterative-scan mode) point. Five minutes,
#: which at the 0.9 QPS measured on MariaDB at 1% selectivity buys ~270 queries
#: of the 500 `main.yml` asks for, and at the 326 QPS measured unfiltered buys
#: all of them many times over. The budget therefore binds exactly where the
#: cost is unpredictable and is invisible everywhere else.
DEFAULT_BUDGET_S = 300.0

#: The share of a point's budget the warm-up may spend before it is cut off.
#:
#: Without this the warm-up could eat the whole budget and the point would
#: record ZERO measured queries. That is not hypothetical on the measurement
#: PR 3 adds: the harness default is 20 warm-up queries, `main`'s budget is
#: 300 s, so anything slower than 15 s a query yields nothing at all -- and
#: this repo has already measured 1.1 s a query for `tag` at 1%, with `ctag`
#: at 1% expected to be far worse because that is the case the correlated
#: column exists to expose.
#:
#: One fifth, so four fifths of every budget is always available to the
#: measured loop and `insufficient_sample` means "few queries" rather than
#: "none". Measured at `main`'s shape -- 300 s, 20 warm-up queries, 15 s a
#: query -- this is 16 measured queries against 1 without it, and against 0
#: when the budget and the per-query cost are closer still.
#:
#: The ceiling is unaffected, because the warm-up's deadline is a FRACTION of
#: the same budget read from the same instant: warm-up plus loop still cannot
#: START work after one budget has elapsed. Both before and after this, the
#: true bound is one budget plus the single query already in flight when the
#: deadline passes -- a deadline checked before each query cannot abandon the
#: one it already began, and `churn.py:244` bounds its slices the same way.
#: Measured: 1.0 s budget at 0.1 s a query spends 1.1 s; 300 s at 15 s a
#: query spends 315 s.
WARMUP_BUDGET_SHARE = 0.2

#: Below this many executed queries a point is still recorded, but flagged.
#: PR 3 §3.4 item 3. Fifty because recall here is a mean over
#: `queries * k` Bernoulli-ish draws and the report greys the point out rather
#: than dropping it: a configuration too slow to finish fifty queries in five
#: minutes is itself the finding, and deleting the record would hide it.
MIN_SAMPLE_QUERIES = 50

#: The buckets of `extra.returned_count_hist`, in the order a report should
#: show them. See `_returned_count_bucket` for why there are six rather than
#: the four PR 3 §3.3 item 4 names.
RETURNED_COUNT_BUCKETS: Tuple[str, ...] = ("0", "1-4", "5-9", "short_other",
                                           "k", "over_k")


def _scan_modes(iterative_scan: Optional[Union[str, Sequence[str]]],
                driver: EngineDriver) -> List[Optional[str]]:
    """The iterative-scan modes to measure this index at, in order.

    pgvector 0.8 can iterate the index when a filter rejects candidates, and
    which mode is active changes the numbers substantially — so it is set
    explicitly and recorded rather than left at whatever the default is.

    Accepts one mode, a comma-separated list of them, or a sequence, because
    `harness/main.py` hands over `args.iterative_scan` verbatim and an
    argparse string is the only shape that survives a command line. Returns
    `[None]` for "measure once, set nothing", which keeps the no-mode case a
    single pass through the same loop rather than a branch around it.

    **A driver without the setter collapses to one pass.** Looping a list of
    modes against MariaDB would re-run the whole column x selectivity grid for
    byte-identical measurements and double the cost of the longest workload in
    the profile to do it. `_ops_iterative_scan` already returns None for
    anything but pgvector, so this is a second line of defence rather than the
    only one — but the first line lives in a file the harness cannot see.

    Order is preserved and duplicates are dropped: a profile declaring
    `[off, off]` asks for one measurement, not two.
    """
    if iterative_scan is None:
        return [None]
    if isinstance(iterative_scan, str):
        requested = [part.strip() for part in iterative_scan.split(",")]
    else:
        requested = [str(part).strip() for part in iterative_scan]
    requested = [mode for mode in requested if mode]
    if not requested:
        return [None]

    # The gate T3a's `_iterative_scan_gate_name()` reads the method name out
    # of, and the reason a renamed `PostgresDriver.set_iterative_scan` would
    # put every pgvector filtered record back to `null` with the suite green.
    if not hasattr(driver, "set_iterative_scan"):
        print(
            f"[filtered] {driver.name}: ignoring iterative-scan mode(s) "
            f"{', '.join(requested)} — this driver has no set_iterative_scan; "
            f"measuring once at the engine's own default",
            flush=True,
        )
        return [None]

    modes: List[Optional[str]] = []
    for mode in requested:
        if mode not in modes:
            modes.append(mode)
    return modes


def _returned_count_bucket(n: int, k: int) -> str:
    """Which `returned_count_hist` bucket a result set of `n` rows falls in.

    PR 3 §3.3 item 4 names four buckets — 0, 1-4, 5-9, k — and those four do
    not partition the integers for any k but 10, which is the only k every
    profile in this repo configures. Two more exist so that every query lands
    in exactly one bucket at every k, because a histogram whose counts do not
    sum to the queries executed is worse than no histogram: it reads as a
    complete distribution and is not one.

    * `k` is tested *before* the ranges, so a complete answer is reported as
      complete whatever k is. At k=5 a result of 5 rows is the whole answer,
      not a "5-9" short one, and the spec's literal reading would file it as
      short.
    * `short_other` catches 10..k-1, which exists only for k > 10. The spec's
      four buckets drop those queries silently.
    * `over_k` catches n > k. No correct driver returns more than k, which is
      exactly why it is counted: `query_filtered` slices `[:k]` in every
      driver, so a non-empty `over_k` means a driver stopped doing that and
      the recall denominator is no longer what the report says it is.

    Consequences at the k values this repo uses: at k=10 `short_other` is
    always 0 and the six buckets behave as the spec's four. At k=5 `5-9` and
    `short_other` are always 0. Both are reported rather than omitted so the
    shape of the dict does not vary with k.
    """
    if n == 0:
        return "0"
    if n == k:
        return "k"
    if n > k:
        return "over_k"
    if n <= 4:
        return "1-4"
    if n <= 9:
        return "5-9"
    return "short_other"


def _returned_count_hist(returned: Sequence[Sequence[int]], k: int) -> Dict[str, int]:
    """Counts per bucket, with every bucket present even at zero.

    Always all six keys: an absent key cannot be told from a zero by a reader
    or a chart, and `ops.pgvector_iterative_scan` already spent 69 records
    proving that absence is the one thing a flat schema cannot express.
    """
    hist = {bucket: 0 for bucket in RETURNED_COUNT_BUCKETS}
    for ids in returned:
        hist[_returned_count_bucket(len(ids), k)] += 1
    return hist


def _group_recall(returned: Sequence[Sequence[int]], truth: numpy.ndarray,
                  k: int, rows: Sequence[int],
                  per_query: Optional[Sequence[float]] = None
                  ) -> Optional[float]:
    """Recall over one subset of the queries, or None if the subset is empty.

    None rather than 0.0, and the difference is the whole point. `tag` at 1%
    selectivity on the 2,000-row fixture puts **zero** of fifty queries in the
    near_pass group — measured — because a uniform 1% of rows contains a
    query's true nearest neighbour about 1% of the time. A 0.0 there is a
    recall figure no query produced, and `selectivity` has already cost this
    repo one shipped defect of exactly that kind.
    """
    if len(rows) == 0:
        return None
    if per_query is not None:
        # Ops version 9: the group means of the SAME per-query vector the
        # point's recall is the mean of, so a group figure and the point
        # beside it are one rule.
        return sum(per_query[i] for i in rows) / len(rows)
    subset = [returned[i] for i in rows]
    return recall_at_k(subset, truth[list(rows)], k)


class _PointResult(NamedTuple):
    """Everything measured at one (column, selectivity, mode) point."""
    executed: int
    planned: int
    time_bounded: bool
    returned: List[List[int]]
    short_results: int
    stats: Optional[LatencyStats]
    qps: float
    #: Warm-up queries actually run, and asked for. Both recorded, because
    #: the warm-up exists to pay for cold graph caches and a share-truncated
    #: warm-up means the first MEASURED queries are partly cold -- on exactly
    #: the slow points the budget exists for. At `main`'s shape a 300 s budget
    #: at 15 s a query runs 5 of the 20 requested, 75% short, and without
    #: these two fields nothing in the record says so: `time_bounded` is
    #: deliberately not set by the warm-up, so a reader can see THAT a point
    #: was truncated but not WHICH loop was.
    warmup_executed: int
    warmup_planned: int
    #: Per-query latencies in seconds, for PR 5 §5.1 item 3's QPS interval.
    #: Defaulted so that a caller constructing this for a point that measured
    #: nothing gets an empty interval rather than a TypeError -- the same
    #: reason `stats` is Optional. Kept here rather than read out of
    #: `LatencyCollector._samples`, which is private and which the window
    #: trimming is free to filter. An empty TUPLE and not an empty list: a
    #: mutable default on a NamedTuple is shared by every instance that omits
    #: it, and nothing here needs it to be mutable.
    latencies: Sequence[float] = ()


def _measure_point(driver: EngineDriver, queries: numpy.ndarray, k: int,
                   column: str, threshold: int, warmup: int,
                   budget_s: float, label: str) -> _PointResult:
    """Warm up and then run the query loop, stopping when the budget is spent.

    One clock covers the warm-up and the measured loop, started together, so
    that `worst case = budget per point` is a real ceiling rather than a
    ceiling plus however long twenty cold queries took. At the 0.9 QPS
    measured on MariaDB at 1% selectivity those twenty queries are 22 s of a
    300 s budget — 7%, which is too much to leave outside a figure the
    estimate prints as a bound.

    The warm-up gets `WARMUP_BUDGET_SHARE` of that budget and no more. Sharing
    one deadline let the warm-up consume the whole thing and record zero
    measured queries, which turns a slow point from "measured badly" into "not
    measured at all" — and the point of recording a truncated point rather
    than dropping it is that the slowness IS the finding.

    Neither arrangement changes the ceiling, and the ceiling is not exactly
    the budget: a deadline tested before each query stops the point from
    STARTING work after the budget has elapsed, but cannot abandon the query
    already running. So the bound is one budget plus one query, measured at
    1.1 s for a 1.0 s budget at 0.1 s a query and 315 s for a 300 s budget at
    15 s a query. `churn.py` bounds its re-insert slices with the same
    comparison and carries the same overrun; the estimate's `points x budget`
    is short by at most one query per point for that reason.

    Ground truth is deliberately **outside** the clock: it is a NumPy brute
    force in the harness container, it is cached on disk, and charging the
    engine's budget for the yardstick's cost would make a cold cache look like
    a slow engine.

    The budget is checked before `perf_counter()` is read, so the check is
    never inside a measured span — and with one client that is the whole
    story. `collector.qps(clients=1)` divides by `measured_seconds`, the sum
    of the per-query spans, NOT by wall time, so work outside a span does not
    enter the throughput figure at all. (An earlier version of this comment
    said the check "counts against QPS" at "0.02%", which described the
    multi-client denominator this workload never uses: measured on ten
    samples with 10 ms of out-of-span overhead each, `qps(clients=1)` is
    1000.0 and `qps(clients=2)` is 99.3.)
    """
    collector = LatencyCollector(warmup=0)
    returned: List[List[int]] = []
    short_results = 0
    time_bounded = False
    planned = len(queries)
    warmup_planned = min(warmup, planned)
    warmup_executed = 0

    started = time.time()
    with Heartbeat(f"warm-up ({min(warmup, planned)} queries)",
                   prefix=driver.name):
        warmup_budget_s = budget_s * WARMUP_BUDGET_SHARE
        for i in range(warmup_planned):
            if time.time() - started > warmup_budget_s:
                # Not `time_bounded`: the warm-up being cut short says
                # nothing about whether the MEASURED loop was, and that flag
                # is what the report greys a point out for.
                # `extra.warmup_executed` is where this shows instead.
                #
                # SETTING `time_bounded` HERE WOULD BE INDISTINGUISHABLE on
                # every configured profile, and that is measured rather than
                # assumed. The two forms differ only when the warm-up
                # truncates AND the measured loop still completes -- the
                # warm-up needs its own deadline to bite
                # (`warmup_planned * step > WARMUP_BUDGET_SHARE * budget`)
                # while the whole point still fits
                # (`(warmup_executed + planned) * step <= budget`). That
                # needs `planned` small against `budget / step`, roughly
                # `planned < 4 * warmup`: measured, the forms DO differ at
                # `planned=8, warmup=20` (warm-up 3 of 8, loop 8 of 8) and do
                # NOT at `planned=50` (warm-up 3 of 20, loop 8 of 50, so both
                # forms set the flag from the loop anyway).
                #
                # No run produces the first shape. Every profile in
                # `config/profiles` sets `ops.max_queries` to 100 or more
                # (dev 100, smoke 200, main/mariadb-blog/tuned-complete 500,
                # quick 1,000, full 2,000) and both real test sets hold
                # 10,000, so `planned >= 100 > 4 * 20`. A mutant that sets
                # the flag here therefore leaves the whole suite green: it is
                # an equivalent form, not an unguarded one. Written down so
                # that nobody "fixes" it later and nobody counts it as a
                # guard.
                break
            driver.query_filtered(queries[i], k, threshold, column=column)
            warmup_executed += 1

    # Filtered queries can be far slower than unfiltered ones — a selective
    # predicate forces the engine to examine many more candidates to find k
    # that qualify — so this loop needs progress of its own.
    progress = Progress(planned, label, prefix=driver.name)
    latencies: List[float] = []
    collector.start()
    for q in queries:
        if time.time() - started > budget_s:
            time_bounded = True
            break
        query_started = time.perf_counter()
        ids = driver.query_filtered(q, k, threshold, column=column)
        elapsed = time.perf_counter() - query_started
        collector.add(elapsed)
        latencies.append(elapsed)
        returned.append(ids)
        if len(ids) < k:
            short_results += 1
        progress.step()
    collector.stop()
    progress.finish()

    # No executed query means no latency distribution. `summarize([])` returns
    # NaN for every percentile, and a NaN in a record is both invalid strict
    # JSON and indistinguishable downstream from a measured zero.
    stats = collector.stats() if returned else None
    return _PointResult(
        executed=len(returned), planned=planned, time_bounded=time_bounded,
        returned=returned, short_results=short_results, stats=stats,
        qps=collector.qps(clients=1) if returned else 0.0,
        warmup_executed=warmup_executed, warmup_planned=warmup_planned,
        latencies=latencies,
    )


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, ef_search: int,
        selectivities: Sequence[float] = DEFAULT_SELECTIVITIES,
        columns: Sequence[str] = DEFAULT_COLUMNS,
        max_queries: int = 1_000, warmup: int = 20,
        iterative_scan: Optional[Union[str, Sequence[str]]] = None,
        indexed_rows: Optional[int] = None,
        budget_s: float = DEFAULT_BUDGET_S) -> None:
    """Measure filtered recall and QPS at each column x selectivity.

    The loop is mode-major, then column, then selectivity. Mode outermost
    because `set_iterative_scan` is a session `SET`: one statement per mode
    rather than one per point, and every point under a mode is measured
    against the same server state. Column before selectivity so a reader of
    the raw JSONL sees each column's curve contiguously.
    """
    queries = dataset.test[:max_queries]
    driver.set_ef_search(ef_search)

    # Validated here, before the first `SET` and before any ground truth is
    # computed, rather than letting `filter_values` raise several minutes in.
    # The driver whitelist refuses an unknown column too, but it refuses it
    # per query — after this workload has already spent a brute force on a
    # yardstick for a column no engine can be asked about.
    if not columns:
        raise ValueError("no filter columns requested")
    unknown = [c for c in columns if c not in FILTER_COLUMNS]
    if unknown:
        raise ValueError(
            f"unknown filter column(s) {unknown}; expected one of "
            f"{list(FILTER_COLUMNS)}")

    # M-1. A non-positive budget means "no cap", which is what the estimate
    # has always assumed (`_filtered_search` treats `budget > 0` as the only
    # bounded case) and what `ops.filtered_budget_s: 0` is documented to do.
    # Before this, 0 meant the opposite: the first elapsed-time check fired
    # immediately and every point recorded ZERO queries -- a config a reader
    # was invited to write that silently measured nothing. Converting here, at
    # the boundary, rather than special-casing each comparison, keeps the two
    # enforcement sites a plain `elapsed > budget` and leaves the one budget
    # this repo already enforces (`churn.py:244`) the shape to match.
    #
    # The DECLARED value is what the record states; the effective one is only
    # the deadline. A record saying `filtered_budget_s: inf` would be both
    # invalid JSON and a claim no profile made.
    effective_budget = budget_s if budget_s > 0 else float("inf")

    modes = _scan_modes(iterative_scan, driver)
    common = ctx.record_defaults(driver, dataset, index)

    # The unfiltered nearest neighbour per query, which is what splits the
    # queries into near_pass and near_fail. Memoised on the row count rather
    # than recomputed per point: it does not depend on the column, the
    # threshold or the mode, and on a subset profile it is a brute force of
    # its own.
    nearest_cache: Dict[int, numpy.ndarray] = {}

    def nearest_unfiltered(rows: int) -> numpy.ndarray:
        if rows not in nearest_cache:
            with Heartbeat("unfiltered ground truth for the near_pass split",
                           prefix=f"filtered] {driver.name}"):
                truth = cached_ground_truth(ctx.cache_dir, dataset, ctx.k,
                                            indexed_rows=rows)
            nearest_cache[rows] = truth[:len(queries), 0]
        return nearest_cache[rows]

    for mode in modes:
        if mode is not None:
            driver.set_iterative_scan(mode)
        for column in columns:
            for selectivity in selectivities:
                _measure_and_record(
                    ctx, driver, dataset, common, queries, column,
                    selectivity, mode, ef_search, warmup,
                    effective_budget, budget_s,
                    indexed_rows, nearest_unfiltered,
                )


def _measure_and_record(ctx: RunContext, driver: EngineDriver,
                        dataset: Dataset,
                        common: Dict[str, object], queries: numpy.ndarray,
                        column: str, selectivity: float,
                        mode: Optional[str], ef_search: int, warmup: int,
                        effective_budget: float, declared_budget: float,
                        indexed_rows: Optional[int],
                        nearest_unfiltered) -> None:
    """One point: resolve the filter, measure it, write the record."""
    # `filtered_truth_for`, not `cached_filtered_ground_truth`. The pair it
    # returns is the whole mechanism: the truth array and the filter that
    # produced it come from one pass over one column, so the share this record
    # states cannot drift from the share the yardstick was built over. The
    # thin wrapper returns the array alone and leaves a caller nothing to
    # record but what it asked for, which is how `threshold / 100.0` survived.
    #
    # `indexed_rows` matters: a profile may have loaded only part of the
    # training set, and ground truth over the full set would score every
    # engine against rows it was never given.
    print(
        f"[filtered] {driver.name}: computing exact ground truth for "
        f"{column} at a nominal {selectivity:.0%} of rows"
        + (f" (iterative_scan={mode})" if mode else "") + "…",
        flush=True,
    )
    with Heartbeat(f"ground truth for {column} at {selectivity:.0%}",
                   prefix=f"filtered] {driver.name}"):
        truth, filt = filtered_truth_for(
            ctx.cache_dir, dataset, ctx.k, column, selectivity,
            indexed_rows=indexed_rows,
        )
    truth = truth[:len(queries)]

    # Per (column, selectivity), not once per run. PR 3 §3.3 item 2: the
    # MySQL-family optimizers may switch to `ctag_idx` or to a full scan at a
    # different threshold than they do for `tag`, and a plan check taken at
    # one point and stamped onto all of them would report the wrong one.
    index_used = driver.explain_uses_vector_index(
        queries[0], ctx.k, tag_threshold=filt.threshold, column=column
    )

    point = _measure_point(
        driver, queries, ctx.k, column, filt.threshold, warmup,
        effective_budget,
        f"filtered {column} {filt.actual_selectivity:.0%} queries",
    )

    # The near_pass / near_fail split, over the executed queries only.
    #
    # A query whose true unfiltered nearest neighbour passes the filter has
    # its best answer inside the candidate list the engine was going to walk
    # anyway; one whose nearest neighbour fails has to keep looking, and that
    # is where post-filtering gives up. Measured on a clustered 2,000-row
    # corpus at a 10% predicate: for `ctag` the two groups came back at 0.983
    # and 0.241 recall, and for `tag` at 0.625 and 0.641 — the same split
    # applied to both columns separates them by a factor of four on one and
    # not at all on the other, which is the difference the second column was
    # added to measure.
    nearest = nearest_unfiltered(filt.rows)
    passes = filt.mask[nearest]
    near_pass = [i for i in range(point.executed) if passes[i]]
    near_fail = [i for i in range(point.executed) if not passes[i]]

    # One call, two consumers: the published mean and the samples §5.1 item 4
    # draws its interval from. `recall_at_k` is itself `sum(per_query) / n`
    # (see `datasets.recall_per_query`), so taking the mean here keeps the
    # point and the bar on one arithmetic rather than two.
    #
    # Ops version 9: ann-benchmarks' distance threshold, scored against the
    # filtered ground truth -- still recomputed over the qualifying rows, so
    # the k-th distance is the k-th among the rows the predicate admits. The
    # id overlap rides beside it as `extra.recall_id_overlap`.
    executed_queries = queries[:len(point.returned)]
    per_query = recall_threshold_per_query(
        point.returned, truth, executed_queries, dataset.train,
        dataset.metric, ctx.k)
    overlap = recall_per_query(point.returned, truth, ctx.k)
    recall = (sum(per_query) / len(per_query)) if point.executed else None
    recall_id_overlap = ((sum(overlap) / len(overlap)) if point.executed
                         else None)
    recall_near_pass = _group_recall(point.returned, truth, ctx.k, near_pass,
                                     per_query)
    recall_near_fail = _group_recall(point.returned, truth, ctx.k, near_fail,
                                     per_query)
    insufficient = point.executed < MIN_SAMPLE_QUERIES

    print(
        f"[filtered] {driver.name}: {column} "
        f"selectivity={filt.actual_selectivity:.2%} "
        f"(nominal {filt.nominal_selectivity:.0%}) "
        f"recall@{ctx.k}={'—' if recall is None else f'{recall:.4f}'} "
        f"near_pass={'—' if recall_near_pass is None else f'{recall_near_pass:.4f}'}"
        f"/{len(near_pass)} "
        f"near_fail={'—' if recall_near_fail is None else f'{recall_near_fail:.4f}'}"
        f"/{len(near_fail)} "
        f"qps={point.qps:,.1f} "
        f"short_results={point.short_results}/{point.executed} "
        f"queries={point.executed}/{point.planned}"
        + (" BUDGET" if point.time_bounded else "")
        + (" INSUFFICIENT" if insufficient else "")
        + f" index_used={index_used}"
    )

    _record_point(ctx, common, filt, ef_search, index_used, point, recall,
                  recall_near_pass, recall_near_fail, len(near_pass),
                  len(near_fail), insufficient, mode, declared_budget,
                  column, per_query=per_query,
                  recall_id_overlap=recall_id_overlap)


def _record_point(ctx: RunContext, common: Dict[str, object], filt,
                  ef_search: int, index_used: bool, point: _PointResult,
                  recall: Optional[float], recall_near_pass: Optional[float],
                  recall_near_fail: Optional[float], near_pass_queries: int,
                  near_fail_queries: int, insufficient: bool,
                  mode: Optional[str], budget_s: float, column: str,
                  per_query: Optional[Sequence[float]] = None,
                  recall_id_overlap: Optional[float] = None) -> None:
    """Write one filtered record.

    `selectivity=filt.actual_selectivity` is the fix this file owed. What was
    here before computed `threshold / 100.0` and called it
    `actual_selectivity`, which is the NOMINAL share under a measured-sounding
    name. For `tag` it is exact only when the row count is a multiple of 100:
    glove's 1,183,514 training rows make a 1% request deliver 1.000073%
    (measured). For `ctag` it is not even the wrong row share — it is the
    fraction of *clusters*, and on the 2,000-row fixture a nominal 1% delivers
    1.90% of rows while `threshold / 100` would claim 2.00%: wrong in
    magnitude and wrong in direction at once. Across sixteen seeds on a
    100,000x100 gaussian corpus T3a measured a nominal 1% delivering 1.00% to
    4.96%, and 7.64% on a lognormal-norm one. The error moves with the seed,
    so a reader cannot correct for it.

    `selectivity` is written unrounded. It is a counted share — a ratio of two
    integers — and the chart's x axis is logarithmic, so rounding it to six
    places would blur exactly the gap between the two columns that this
    record exists to show.

    `tag_threshold` is emitted only for the `tag` column. It is kept because
    archived records and the readers of them speak in those terms, and it is
    withheld from a `ctag` record because a key named for one column carrying
    another column's threshold is the same class of defect as the selectivity
    above. `filter_threshold`, from `ResolvedFilter.as_extra()`, is the
    column-neutral name and is always present.
    """
    # `queries_executed` comes from `LatencyStats.as_dict()`, which reports the
    # collector's own sample count -- not from `point.executed` beside it. The
    # collector takes `warmup=0`, so it holds exactly one sample per executed
    # query and the two are the same number by construction; writing both
    # would be two sources for one field, and PR 3 §3.4 item 2 needs the one
    # that the latency percentiles were computed over. The empty case has no
    # percentiles to report but still owes a count, which is the whole point of
    # recording a budget-stopped point rather than dropping it.
    measured = (point.stats.as_dict() if point.stats is not None
                else {"queries_executed": 0})
    ctx.recorder.write(Record(
        **common,
        phase=PHASE_FILTERED,
        ef_search=ef_search,
        selectivity=filt.actual_selectivity,
        recall_at_k=None if recall is None else round(recall, 6),
        qps=round(point.qps, 2),
        clients=1,
        vector_index_used=index_used,
        **measured,
        extra={
            # column, both selectivities, threshold and the ctag parameters.
            **filt.as_extra(),
            "short_result_queries": point.short_results,
            # A filtered result set shorter than k is a legitimate engine
            # behaviour, not a harness bug — flagged so the report can say so.
            "returned_fewer_than_k": point.short_results > 0,
            "returned_count_hist": _returned_count_hist(point.returned, ctx.k),
            "recall_near_pass": (None if recall_near_pass is None
                                 else round(recall_near_pass, 6)),
            "recall_near_fail": (None if recall_near_fail is None
                                 else round(recall_near_fail, 6)),
            "near_pass_queries": near_pass_queries,
            "near_fail_queries": near_fail_queries,
            "queries_planned": point.planned,
            # Additive, and no measurement version bump: a field that was
            # absent is now populated and nothing a previous reader trusted
            # moves -- the same argument `records.py`'s version-3 paragraph
            # makes for the recall it added to concurrency records. It says
            # which loop the budget cut, which `time_bounded` alone cannot.
            "warmup_executed": point.warmup_executed,
            "warmup_planned": point.warmup_planned,
            # Written on every record, True or False, rather than only when
            # the budget bit. A key that appears only on truncated points
            # cannot distinguish "ran to completion" from "written by a
            # harness that did not have budgets yet".
            "time_bounded": point.time_bounded,
            "insufficient_sample": insufficient,
            "filtered_budget_s": budget_s,
            # PR 5 §5.1 item 4. Over the queries that actually RAN, which on a
            # budget-stopped point is fewer than `queries_planned` -- and that
            # is the case the interval matters most for: `insufficient_sample`
            # already says the recall came from under 50 queries, and this
            # says how wide that makes it.
            #
            # `per_query` is threaded down from `_point` rather than
            # recomputed here. Recomputing would be a second call to
            # `recall_per_query` on the same inputs, and two calls are the
            # only way for the bar and the point to disagree.
            #
            # (`recall` is NOT already rounded when it arrives -- an earlier
            # revision of this comment said it was. It is rounded in the
            # `recall_at_k=` argument near the top of this same `Record`,
            # which is why it cannot be reused for an interval anyway: the
            # interval needs the per-query samples, not the mean at any
            # precision.)
            **ci_fields(PHASE_FILTERED,
                        recall_per_query=per_query or None,
                        latencies_s=point.latencies or None),
            "iterative_scan": mode,
            "recall_kind": RECALL_DISTANCE_THRESHOLD,
            "recall_id_overlap": (None if recall_id_overlap is None
                                  else round(recall_id_overlap, 6)),
            **({"tag_threshold": filt.threshold}
               if column == FILTER_TAG else {}),
        },
    ))
