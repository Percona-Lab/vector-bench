"""Calibration: pick the ef_search that puts this engine at a target recall.

Every ops number downstream of this -- concurrency QPS, filtered recall, churn
degradation -- is read as a comparison between engines. That comparison is only
honest if the engines were doing a comparable amount of work per query, and
`ef_search` does not guarantee that. It is `mhnsw_ef_search` in MariaDB, VIDX's
own equivalent in AliSQL, `hnsw.ef_search` in pgvector, `numCandidates` in
MongoDB and `EF_RUNTIME` in Valkey. The same literal 100 buys five different
amounts of search effort, and two of these engines sit at recall 0.98 and 0.80
at that setting -- the ordinary spread of these implementations, not a
pathology. A throughput chart drawn across them at a shared ef compares one
engine answering the question with another approximating it.

So the ops pass measures the recall curve on the index it has just built, picks
the smallest ef that reaches a target recall, and runs everything after this at
that value. Engines are then compared at equal answer quality.

Why this cannot come from the recall/QPS path: in `main.yml` that path measures
M in {8, 32} while ops builds M=16, and the tuned pass sweeps quantization and
ef_construction that ops pins differently. A curve measured on a different
index is not a calibration of this one. Measuring it here costs minutes against
the hours the ops pass already spends ingesting, and couples to nothing.

Three details that decide whether the chosen value means anything:

1. **The next grid point has to agree.** Recall over a finite query set is a
   mean of Bernoulli draws, so a single point can clear the target by noise.
   The chosen ef is the smallest whose recall reaches the target *and* whose
   successor also reaches it. On a clean curve that is the same point; on a
   noisy one it is the difference between calibrating an engine and calibrating
   a fluctuation.
2. **A floor above the target is reported, not papered over.** For an engine
   whose recall at minimum effort is already above the target there may be no
   grid point that *slows* search to it -- MHNSW and VIDX at M=16 are the two
   this is expected of, and whether they are is what the sweep measures rather
   than something asserted here. The status says which case the run was in, so
   the report cannot imply an engine was held to the same quality as the
   others when it could not be.
3. **The sweep stops as soon as it is confirmed.** Larger ef values cost query
   time and change nothing about the answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..datasets import (RECALL_DISTANCE_THRESHOLD, RECALL_ID_OVERLAP,
                        Dataset, cached_ground_truth, recall_at_k,
                        recall_per_query, recall_threshold_per_query)
from ..drivers.base import EngineDriver, IndexSpec
from ..metrics.latency import LatencyCollector
from ..metrics.records import PHASE_CALIBRATION, Record
from ..metrics.stats import ci_fields
from ..progress import Heartbeat, Progress
from .context import RunContext

#: Grid swept by default: denser at the low end, where MHNSW and pgvector cross
#: 0.95, and reaching far enough up that an engine which needs 1,000 candidates
#: is measured rather than reported as unreachable. Its length is the sweep's
#: cost ceiling -- twelve points of (warm-up + queries) per engine-dataset is
#: the worst case, and early stop is expected to halve it.
#:
#: Declared here, on the harness side of the container boundary, because this
#: is the module that consumes it. `orchestrator/ops_pass.py` declares the same
#: list for the command line it builds: it runs on a host whose only Python
#: dependency is pyyaml and importing this module would put numpy on that list.
#: `tests/test_calibrate.py` pins the two together.
DEFAULT_EF_GRID: Tuple[int, ...] = (10, 16, 24, 32, 48, 64, 100, 160, 256,
                                    400, 640, 1000)

#: Recall the sweep aims for. 0.95 is the quality level ann-benchmarks' own
#: comparisons are usually read at, and it is reachable by every engine here at
#: M=16 on the datasets this framework measures.
DEFAULT_TARGET_RECALL = 0.95

#: Queries per grid point. The same default the harness's `--max-queries`
#: carries; a profile that lowers it to bound the filtered and churn workloads
#: lowers this too, which is the right trade -- a smaller query set is a
#: noisier recall figure, and the next-grid-point clause exists because of it.
DEFAULT_MAX_QUERIES = 1_000

#: Queries discarded before each point is measured. Cold caches belong to no
#: measurement, and a cold first point would read as an engine that needs a
#: higher ef than it does.
DEFAULT_WARMUP = 20

#: How far above the target the smallest grid ef has to sit before the sweep
#: reports that this engine cannot be slowed to the target. Below it, the floor
#: is simply the answer and the status is `ok`; a margin of zero would label
#: every engine whose first point is 0.951 as uncalibratable.
FLOOR_MARGIN = 0.02

#: Target met inside the grid.
STATUS_OK = "ok"
#: No grid point reached the target. The largest ef is used and the engine is
#: reported at its best achievable recall.
STATUS_TARGET_UNREACHED = "target_unreached"
#: The smallest grid ef already exceeds the target by more than FLOOR_MARGIN.
#: The smallest ef is used, and the report must say this engine cannot be
#: slowed down to the target.
STATUS_FLOOR_ABOVE_TARGET = "floor_above_target"

EF_MODE_FIXED = "fixed"
EF_MODE_CALIBRATED = "calibrated"
EF_MODES = (EF_MODE_FIXED, EF_MODE_CALIBRATED)

#: Workloads whose ef_search the sweep exists to choose. A run that measures
#: none of them would pay for the sweep and discard the answer, which at twelve
#: grid points is minutes per engine-dataset for nothing.
CALIBRATED_WORKLOADS = ("concurrency", "filtered", "churn")


@dataclass(frozen=True)
class CalibrationPoint:
    """One grid point: what was asked for, what ran, and what came back."""

    ef_search: int
    effective_ef: int
    recall_at_k: float
    qps: float
    #: Ops version 9: the id-overlap figure over the same queries, kept
    #: beside the distance-threshold `recall_at_k` the target is judged on.
    recall_id_overlap: Optional[float] = None
    recall_kind: str = RECALL_ID_OVERLAP

    def reached(self, target_recall: float) -> bool:
        """Whether this point meets the target.

        A method rather than a stored flag so there is one definition of
        "reached". Stored, it would be computed against the target in force
        when the point was measured and then compared against the target the
        selector was given, and the two would be free to disagree.
        """
        return self.recall_at_k >= target_recall


@dataclass(frozen=True)
class CalibrationResult:
    """The ef the run will measure at, and why.

    Positional construction is `CalibrationResult(ef, recall, status)`; every
    further field carries a default, so callers that only want those three keep
    working.

    `confirmed` is whether the next grid point also reached the target. It is
    not a fourth status: `ok` with `confirmed=False` is the honest description
    of a target first met at the last grid point, where the clause cannot be
    applied because the grid ran out rather than because it failed.
    """

    ef_search: int
    recall_at_k: float
    status: str
    effective_ef: int = 0
    confirmed: bool = False
    target_recall: float = DEFAULT_TARGET_RECALL
    points: Tuple[CalibrationPoint, ...] = field(default_factory=tuple)


def ascending_grid(ef_grid: Sequence[int]) -> List[int]:
    """The grid as the sweep walks it: ascending, de-duplicated, ints.

    Not a formality. Early stop reads consecutive points, so a descending grid
    swept in the order given stops at the two largest values and calibrates
    every engine to the top of its grid -- the exact opposite of what the sweep
    is for. A repeated value would pay for a point twice and satisfy the
    next-grid-point clause against itself.
    """
    return sorted({int(ef) for ef in ef_grid})


def select(points: Sequence[CalibrationPoint],
           target_recall: float = DEFAULT_TARGET_RECALL) -> CalibrationResult:
    """Choose the ef from a measured curve. Pure; no driver, no records.

    Separate from `run` so the selection rule can be tested over curves no
    error model produces -- non-monotone ones, and floors that sit exactly on
    the margin.
    """
    if not points:
        raise ValueError(
            "calibration measured no grid points, so there is nothing to "
            "choose from; an empty ef grid is a configuration error")
    index, status = _choose(points, target_recall)
    chosen = points[index]
    successor = points[index + 1] if index + 1 < len(points) else None
    return CalibrationResult(
        ef_search=chosen.ef_search,
        recall_at_k=chosen.recall_at_k,
        status=status,
        effective_ef=chosen.effective_ef,
        confirmed=bool(successor is not None
                       and successor.reached(target_recall)),
        target_recall=target_recall,
        points=tuple(points),
    )


def _choose(points: Sequence[CalibrationPoint],
            target_recall: float) -> Tuple[int, str]:
    """Which measured point to use, and under which status."""
    # The floor first. If minimum effort already overshoots the target by more
    # than the margin there is nothing to calibrate -- no grid point slows this
    # engine down -- so the answer is the smallest ef whatever the rest of the
    # curve does, and the status is what the report needs in order not to claim
    # this engine was held to the same recall as the others.
    if points[0].recall_at_k > target_recall + FLOOR_MARGIN:
        return 0, STATUS_FLOOR_ABOVE_TARGET

    # The smallest ef that reaches the target AND whose successor reaches it
    # too. The clause is the whole guard: without it a single point lifted over
    # the target by sampling noise becomes the ef every later workload is
    # measured at.
    for i in range(len(points) - 1):
        if points[i].reached(target_recall) and points[i + 1].reached(
                target_recall):
            return i, STATUS_OK

    # The target was met, but only at the last grid point, so nothing can
    # confirm it. Used anyway -- it did reach the target -- and the result's
    # `confirmed` flag is False so a reader is not told the clause held.
    for i, point in enumerate(points):
        if point.reached(target_recall):
            return i, STATUS_OK

    # Nothing reached it. The largest ef is the engine's best showing, which is
    # what the report has to present it at. This is also the largest point in
    # the grid rather than merely the largest measured: early stop needs two
    # consecutive points at the target, so a sweep that found none ran to the
    # end.
    return len(points) - 1, STATUS_TARGET_UNREACHED


def _measure_point(driver: EngineDriver, ctx: RunContext, queries,
                   truth, ef: int, warmup: int, corpus=None,
                   metric: Optional[str] = None):
    """One grid point: warm up, time the query set, score it.

    Returns the point, its latency summary and whether the planner used the
    vector index, which is everything the record needs. Split out of `run` so
    that function is the sweep's decisions -- order, early stop, selection --
    and this one is the measurement.
    """
    driver.set_ef_search(ef)
    # Asked of the driver rather than assumed: MongoDB cannot search fewer
    # candidates than k, so its "ef 10" at k=100 is a search of 100.
    effective = int(driver.effective_ef(ef, ctx.k))

    # Per point, after the ef is set, because the record carries this field and
    # a field describing a plan taken at a different ef would be worse than an
    # absent one. It is what tells "this engine needs ef 1000" apart from "this
    # engine fell back to a full scan", which recall alone cannot: a scan
    # reports 1.0 at the smallest ef and is indistinguishable from an engine
    # whose floor is above the target.
    index_used = driver.explain_uses_vector_index(queries[0], ctx.k)

    for i in range(min(warmup, len(queries))):
        driver.query(queries[i], ctx.k)

    collector = LatencyCollector(warmup=0)
    returned: List[List[int]] = []
    # Kept beside the collector rather than read back out of it. The collector
    # owns `_samples` privately and its public surface is summary statistics;
    # a bootstrap needs the samples themselves, and reaching into the private
    # list from a workload is the kind of coupling that breaks the next time
    # the collector grows a window-trimming rule that drops some of them.
    latencies: List[float] = []
    progress = Progress(len(queries), f"calibration ef={ef}",
                        prefix=driver.name)
    collector.start()
    for q in queries:
        started = time.perf_counter()
        ids = driver.query(q, ctx.k)
        elapsed = time.perf_counter() - started
        collector.add(elapsed)
        latencies.append(elapsed)
        returned.append(ids)
        progress.step()
    collector.stop()
    progress.finish()

    # Ops version 9: the target is judged on ann-benchmarks' distance
    # threshold, the rule `recall_qps` records use, so a calibration "reached
    # 0.95" and an ann point "at 0.95" are one claim. The id overlap rides
    # beside it. A caller with no corpus (a unit test of the selector) keeps
    # the id overlap, and the record says which it got.
    overlap = recall_at_k(returned, truth, ctx.k)
    if corpus is not None and metric is not None:
        per_query = recall_threshold_per_query(returned, truth, queries,
                                               corpus, metric, ctx.k)
        recall = sum(per_query) / len(returned) if returned else float("nan")
        kind = RECALL_DISTANCE_THRESHOLD
    else:
        per_query = recall_per_query(returned, truth, ctx.k)
        recall = overlap
        kind = RECALL_ID_OVERLAP
    point = CalibrationPoint(
        ef_search=int(ef), effective_ef=effective,
        recall_at_k=recall,
        qps=collector.qps(clients=1),
        recall_id_overlap=overlap,
        recall_kind=kind,
    )
    # PR 5 §5.1 item 4. Computed here, where the samples are, rather than in
    # `_record`: the per-query recalls and the per-query latencies are the two
    # biggest things this function holds and neither has any reason to outlive
    # it.
    return (point, collector.stats(), index_used,
            ci_fields(PHASE_CALIBRATION, recall_per_query=per_query,
                      latencies_s=latencies))


def _record(common: Dict[str, object], point: CalibrationPoint, stats,
            index_used: bool, target_recall: float, position: int,
            grid_size: int, warmup: int,
            ci: Optional[Dict[str, object]] = None) -> Record:
    """The record for one grid point.

    `ci` defaults to None rather than to `{}` so that a caller which has no
    interval to add writes a record with the keys absent, which is the shape
    every record written before PR 5 has.
    """
    return Record(
        **common,
        phase=PHASE_CALIBRATION,
        ef_search=point.ef_search,
        recall_at_k=round(point.recall_at_k, 6),
        qps=round(point.qps, 2),
        clients=1,
        vector_index_used=index_used,
        **stats.as_dict(),
        extra={
            # Requested vs searched. A reader comparing engines at "ef 10"
            # needs to know which of them actually looked at 10 candidates.
            "effective_ef": point.effective_ef,
            "target_recall": target_recall,
            "reached_target": point.reached(target_recall),
            # Where in the grid this point sat and how long the grid was, so a
            # sweep that stopped after three of twelve points says so in the
            # record rather than only in the log.
            "grid_position": position,
            "grid_size": grid_size,
            "warmup_queries": warmup,
            "recall_kind": point.recall_kind,
            "recall_id_overlap": (round(point.recall_id_overlap, 6)
                                  if point.recall_id_overlap is not None
                                  else None),
            # PR 5 §5.1 item 4's two keys, last so that a future named key
            # above cannot be silently overwritten by one of them.
            **(ci or {}),
        },
    )


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, ef_grid: Sequence[int] = DEFAULT_EF_GRID,
        target_recall: float = DEFAULT_TARGET_RECALL,
        max_queries: int = DEFAULT_MAX_QUERIES,
        indexed_rows: Optional[int] = None,
        warmup: int = DEFAULT_WARMUP) -> CalibrationResult:
    """Sweep the grid, record every point, and return the chosen ef.

    `indexed_rows` must be the number of rows the engine actually holds: a
    profile may have loaded a subset (smoke loads 20,000 of fashion-mnist's
    60,000), and ground truth over the full training set would score this
    engine against rows it was never given -- understating recall across the
    whole grid and calibrating every engine to the top of it.

    The query set overlaps the one the later workloads use, deliberately. This
    sweep's output is a recall figure and an ef value, not a speed, so a warm
    cache here is not a contaminated measurement there; and sharing the queries
    means the recall this picks an ef for is the recall those workloads will
    report.

    Leaves the driver's session at the chosen ef. Every later workload sets its
    own, but the plan check between them does not, and a session left at the
    top of the grid would explain a plan nobody chose.
    """
    grid = ascending_grid(ef_grid)
    if not grid:
        raise ValueError(
            "ef calibration needs at least one grid point; pass --ef-mode "
            "fixed to measure at a single ef_search instead")

    queries = dataset.test[:max_queries]
    if len(queries) == 0:
        raise ValueError("ef calibration needs at least one query vector")

    # Recomputed when only part of the training set was loaded; the shipped
    # neighbours would otherwise point at rows the engine never received.
    with Heartbeat("ground truth for calibration", prefix=driver.name):
        truth = cached_ground_truth(ctx.cache_dir, dataset, ctx.k,
                                    indexed_rows=indexed_rows)
    truth = truth[:len(queries)]

    common = ctx.record_defaults(driver, dataset, index)

    print(f"[calibrate] {driver.name}: sweeping ef_search {grid} for "
          f"recall@{ctx.k} >= {target_recall:.3f} over {len(queries):,} "
          f"queries", flush=True)

    points: List[CalibrationPoint] = []
    for position, ef in enumerate(grid):
        point, stats, index_used, ci = _measure_point(
            driver, ctx, queries, truth, ef, warmup,
            corpus=dataset.train, metric=dataset.metric)
        points.append(point)

        print(f"[calibrate] {driver.name}: ef_search={ef} "
              f"(effective {point.effective_ef}) "
              f"recall@{ctx.k}={point.recall_at_k:.4f} "
              f"qps={point.qps:,.1f} "
              f"{'>=' if point.reached(target_recall) else '< '} target")

        ctx.recorder.write(_record(
            common, point, stats, index_used, target_recall,
            position=position, grid_size=len(grid),
            warmup=min(warmup, len(queries)), ci=ci))

        # Early stop: two consecutive points at the target settle both the
        # selection rule and the question, and every larger ef is pure cost.
        if len(points) >= 2 and points[-1].reached(target_recall) \
                and points[-2].reached(target_recall):
            skipped = len(grid) - position - 1
            print(f"[calibrate] {driver.name}: target confirmed at "
                  f"ef_search={points[-2].ef_search}"
                  + (f"; skipping the {skipped} larger grid point(s)"
                     if skipped else "; the grid ends here"))
            break

    result = select(points, target_recall)
    driver.set_ef_search(result.ef_search)
    _announce(driver, ctx.k, result)
    return result


def _announce(driver: EngineDriver, k: int,
              result: CalibrationResult) -> None:
    """Say which ef the run will use and, for two of the statuses, why the
    comparison that follows is qualified.

    On stdout rather than only in the record because the phase log is what an
    operator reads while a run is going, and both of these change how every
    later number for this engine must be read.
    """
    target = result.target_recall
    print(f"[calibrate] {driver.name}: measuring at ef_search="
          f"{result.ef_search} (recall@{k}={result.recall_at_k:.4f}, "
          f"status={result.status}"
          f"{'' if result.confirmed else ', unconfirmed'})")
    if result.status == STATUS_TARGET_UNREACHED:
        print(f"[calibrate] WARNING: {driver.name} did not reach "
              f"recall@{k} {target:.3f} anywhere on the grid; it is reported "
              f"at its best achievable recall ({result.recall_at_k:.4f} at "
              f"ef_search={result.ef_search})")
    elif result.status == STATUS_FLOOR_ABOVE_TARGET:
        print(f"[calibrate] NOTE: {driver.name} is already at "
              f"recall@{k}={result.recall_at_k:.4f} at the smallest grid "
              f"ef_search ({result.ef_search}); it cannot be slowed to "
              f"{target:.3f} and is compared above the target")


def status_extra(ef_mode: str,
                 result: Optional[CalibrationResult]) -> Dict[str, object]:
    """The ef-policy keys every downstream record should carry.

    Assembled here rather than at the call site so the three keys cannot drift
    apart from the two modes. In `fixed` mode only the mode itself is true:
    nothing aimed at a recall and no calibration ran, so no target and no
    status are asserted. `Recorder` drops a None rather than writing a null.
    """
    if result is None:
        return {"ef_mode": ef_mode}
    return {
        "ef_mode": ef_mode,
        "target_recall": result.target_recall,
        "calibration_status": result.status,
    }
