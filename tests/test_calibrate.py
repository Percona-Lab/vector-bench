"""Tests for the ef-search calibration sweep: `harness/workloads/calibrate.py`.

The sweep exists so the ops workloads compare engines at the same *answer
quality* rather than at the same knob setting. `ef_search` is not one quantity
across these seven engines -- it is `mhnsw_ef_search` (both MariaDB versions),
VIDX's equivalent, `hnsw.ef_search`, MongoDB's `numCandidates`, Valkey's
`EF_RUNTIME` and Percona's `innodb_hnsw_ef_search` -- so a shared literal 100
buys six different amounts of search effort, and a concurrency chart drawn
from it compares engines doing materially different work per query.

Four things are pinned here, in this order of importance:

1. **The chosen ef is the one the recall curve justifies.** `FakeDriver` has an
   analytical error model (`p(ef) = ef / (ef + ef_half)` per neighbour), so the
   grid point a correct implementation must choose can be written down from the
   model rather than read back out of the implementation.
2. **The next-grid-point clause is tested on its own.** "The chosen ef matches
   the analytical one" is satisfied by a selection rule that ignores the clause
   entirely, because on a clean monotone curve the first point above target and
   the first *confirmed* point above target are the same point. Only a curve
   with a dip can tell the two apart, and `TestTheNextGridPointClause` is that
   curve.
3. **The assembly.** `orchestrator/cli.py::_run_unit` builds the harness
   command line for every ops measurement and had no test in this repo's
   history; that is exactly how `ops.pgvector_iterative_scan` came to be
   declared in three config files, plumbed to the SQL, rendered in a report
   column and never once executed -- 69 of 69 archived records read null. Three
   more arguments join that path here, so
   `TestTheRunUnitHandsTheArgumentsToTheHarness` drives `_run_unit` itself and
   reads both the call it makes to `harness_args` and the argv that comes out
   the far end.
4. **Every downstream record says which ef policy produced it**, including the
   records written by workloads this task never edited.

Recall figures in the driver-backed tests are means of Bernoulli draws, so the
fixtures below are chosen with the standard error written out: a grid whose
points straddle the target by several standard errors, not by one. The seeded
RNG makes them reproducible; the margins make them *right* rather than merely
reproducible.

Nothing here measures speed. `FakeDriver`'s query cost is numpy-bound and
in-process -- another task measured its concurrency curve peaking at two
clients with the client CPU flat -- so its QPS is not a real engine's and the
cost of the sweep cannot be calibrated against it. The QPS assertions below
check only that a figure was recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Sequence

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.datasets as ds  # noqa: E402
from harness.drivers.base import ConnectionSpec  # noqa: E402
from harness.metrics import records as records_mod  # noqa: E402
from harness.metrics.records import Record, Recorder  # noqa: E402
from harness.workloads import calibrate as calibrate_mod  # noqa: E402
from harness.workloads.context import RunContext  # noqa: E402
from tests.fake_driver import (K, N_QUERIES, FakeDriver,  # noqa: E402
                               index_spec, loaded_driver, tiny_dataset)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: §2.2.4's grid and §2.2.2's target, written out here rather than imported, so
#: the implementation cannot define the thing it is checked against.
SPEC_EF_GRID = (10, 16, 24, 32, 48, 64, 100, 160, 256, 400, 640, 1000)
SPEC_TARGET_RECALL = 0.95


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def analytic_recall(ef: int, ef_half: float) -> float:
    """`FakeDriver`'s documented per-neighbour survival probability.

    Written out from the fake's docstring (`p(ef) = min(1, ef / (ef +
    ef_half))`) instead of calling `driver.p_ef()`, so the curve the chosen ef
    is calibrated against is a statement about the model and not a second read
    of the same expression. `query()` draws once per true neighbour and
    replaces a missed one with a row from outside the true top-k, so measured
    recall@k is the mean of `n_queries * k` draws at this probability.
    """
    return min(1.0, ef / (ef + ef_half))


def standard_error(p: float, draws: int) -> float:
    """Standard error of a recall figure that is the mean of `draws` draws."""
    return math.sqrt(max(p * (1.0 - p), 0.0) / draws)


def read_records(path: str) -> List[dict]:
    return list(records_mod.read_records(path))


def phases(rows: Sequence[dict], phase: str) -> List[dict]:
    return [r for r in rows if r.get("phase") == phase]


class ScriptedDriver(FakeDriver):
    """A driver whose recall at each `ef_search` is dictated by the test.

    `FakeDriver`'s error model is the right fixture for "is the chosen ef the
    one the curve justifies", because the curve is a model rather than a table
    of numbers a test wrote. It is the wrong fixture for the status rules and
    the next-grid-point clause, which need a specific recall at a specific grid
    point -- including non-monotone curves no error model produces, and
    boundary values a mean of random draws cannot be asked to land on.

    So this driver answers each query with an exact number of true neighbours.
    A recall of 0.968 over 50 queries at k=10 is 484 correct answers out of
    500, distributed as 34 queries answered perfectly and 16 answered with 9 of
    10 -- an exact mean, not a draw near it. Filler ids come from far outside
    the id space, so they intersect no ground truth and cannot hand recall
    back.
    """

    def __init__(self, dataset, recall_by_ef: Dict[int, float],
                 k: int = K, **kwargs):
        super().__init__(dim=dataset.dim, metric=dataset.metric, **kwargs)
        self._script = {int(ef): float(r) for ef, r in recall_by_ef.items()}
        self._truth = dataset.neighbors[:, :k]
        self._k = int(k)
        self._row_of = {dataset.test[i].tobytes(): i
                        for i in range(len(dataset.test))}
        #: Every `set_ef_search` value, in the order the workload asked for it.
        self.ef_calls: List[int] = []
        #: Every query, warm-up included, as (ef_search, query row).
        self.queries: List[tuple] = []

    def _correct_for(self, recall: float, row: int, rows: int) -> int:
        """How many of the true top-k this query returns.

        Spread so the mean over `rows` queries is exactly `recall`: the first
        `extra` queries carry one more correct answer than the rest.
        """
        total = int(round(recall * rows * self._k))
        base, extra = divmod(total, rows)
        return min(self._k, base + (1 if row < extra else 0))

    def set_ef_search(self, ef_search: int) -> None:
        self.ef_calls.append(int(ef_search))
        super().set_ef_search(int(ef_search))

    def query(self, vector, k: int) -> List[int]:
        row = self._row_of[numpy.asarray(vector, dtype=numpy.float32).tobytes()]
        self.queries.append((self._ef_search, row))
        recall = self._script[int(self._ef_search)]
        correct = self._correct_for(recall, row, len(self._row_of))
        ids = [int(i) for i in self._truth[row][:correct]]
        # Ids no ground truth can contain, so they are wrong answers rather
        # than accidental right ones.
        ids += [10 ** 6 + j for j in range(k - correct)]
        return ids

    @property
    def measured_efs(self) -> List[int]:
        """The ef values the sweep actually ran queries at, in order."""
        out: List[int] = []
        for ef, _row in self.queries:
            if not out or out[-1] != ef:
                out.append(ef)
        return out


def scripted(dataset, recall_by_ef: Dict[int, float], **kwargs) -> ScriptedDriver:
    driver = ScriptedDriver(dataset, recall_by_ef, **kwargs)
    driver.connect()
    return driver


def context(tmp_path, recorder: Recorder, k: int = K) -> RunContext:
    return RunContext(run_id="run-0", resource_pass="normalized",
                      recorder=recorder, engine="fake", engine_tag="tag-0",
                      march="x86-64-v3", k=k, cache_dir=str(tmp_path / "cache"))


def sweep(tmp_path, driver, dataset, ef_grid, target_recall=SPEC_TARGET_RECALL,
          max_queries: int = N_QUERIES, **kwargs):
    """Run the sweep against `driver` and return (result, records)."""
    out = str(tmp_path / "ops.jsonl")
    recorder = Recorder(out)
    ctx = context(tmp_path, recorder)
    try:
        result = calibrate_mod.run(
            ctx, driver, dataset, index_spec(dataset),
            ef_grid=ef_grid, target_recall=target_recall,
            max_queries=max_queries, indexed_rows=len(dataset.train),
            **kwargs)
    finally:
        recorder.close()
    return result, read_records(out)


def curve_points(recall_by_ef: Dict[int, float], effective=None):
    """`CalibrationPoint`s straight from a recall curve, for the pure selector.

    The recalls are inputs here, not outputs of the code under test: `select`
    is a pure function of a measured curve, and giving it curves no engine
    would produce is how the status rules and the next-grid-point clause get
    tested at all.
    """
    return [
        calibrate_mod.CalibrationPoint(
            ef_search=int(ef),
            effective_ef=int(effective(ef) if effective else ef),
            recall_at_k=float(recall),
            qps=100.0,
        )
        for ef, recall in sorted(recall_by_ef.items())
    ]


# ---------------------------------------------------------------------------
# 1. The chosen ef against the model's own curve
# ---------------------------------------------------------------------------

class TestTheChosenEfMatchesTheAnalyticalCurve:
    """§2.4's first test. `FakeDriver` keeps each true neighbour with
    probability `ef / (ef + ef_half)`, so for a given grid and target the grid
    point a correct sweep must choose is arithmetic, done here by hand.

    `ef_half = 2` over the grid [8, 18, 98, 198] gives 0.800, 0.900, 0.980 and
    0.990. Against a 0.95 target the answer is 98: the first point at or above
    the target whose successor is also at or above it. The margins are what
    make the fixture honest -- at 500 draws the standard error is about 0.006,
    so 0.900 sits sixteen standard errors below the target and 0.980 nearly
    five above it. A fixture whose points sat at 0.951 and 0.949 would be
    measuring the RNG.
    """

    GRID = (8, 18, 98, 198)
    EF_HALF = 2.0

    def test_the_chosen_ef_is_where_the_curve_crosses_the_target(self, tmp_path):
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=self.EF_HALF)
        result, _rows = sweep(tmp_path, driver, dataset, self.GRID)
        assert result.ef_search == 98
        assert result.status == calibrate_mod.STATUS_OK
        assert result.confirmed is True

    def test_every_measured_point_sits_on_the_models_own_curve(self, tmp_path):
        """Across the parameter, not at one point. A sweep that queried at a
        stale ef -- set once and never updated -- produces a flat curve, and a
        check at the chosen point alone cannot see it."""
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=self.EF_HALF)
        _result, rows = sweep(tmp_path, driver, dataset, self.GRID)
        measured = {r["ef_search"]: r["recall_at_k"]
                    for r in phases(rows, records_mod.PHASE_CALIBRATION)}
        assert len(measured) >= 3, "the sweep recorded almost nothing"
        draws = N_QUERIES * K
        for ef, recall in measured.items():
            expected = analytic_recall(ef, self.EF_HALF)
            tolerance = 4.0 * standard_error(expected, draws)
            assert abs(recall - expected) <= tolerance, (
                f"ef={ef}: measured {recall:.4f}, the model says "
                f"{expected:.4f} +/- {tolerance:.4f}")

    def test_a_lower_target_picks_a_lower_point_on_the_same_curve(self, tmp_path):
        """0.85 is first cleared at ef=18 (0.900), and 98 confirms it."""
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=self.EF_HALF)
        result, _rows = sweep(tmp_path, driver, dataset, self.GRID,
                              target_recall=0.85)
        assert result.ef_search == 18
        assert result.status == calibrate_mod.STATUS_OK

    def test_a_blunter_engine_is_calibrated_to_a_higher_ef(self, tmp_path):
        """The whole point of the sweep: one target, two engines, two ef
        values. At `ef_half = 20` the curve reads 0.286, 0.474, 0.831, 0.908,
        0.980, 0.990 over the grid below, so 0.95 is first cleared at 998 where
        the sharper engine above cleared it at 98."""
        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=20.0)
        grid = (8, 18, 98, 198, 998, 1998)
        result, _rows = sweep(tmp_path, driver, dataset, grid)
        assert result.ef_search == 998
        assert result.status == calibrate_mod.STATUS_OK


# ---------------------------------------------------------------------------
# 2. The next-grid-point clause, on its own
# ---------------------------------------------------------------------------

class TestTheNextGridPointClause:
    """§2.1.5's guard against a single noisy point, and §2.4's third test.

    A monotone curve cannot test this: the first point above target and the
    first confirmed point above target coincide, so a rule that drops the
    clause passes every monotone fixture -- including the analytical one above.
    These curves dip.
    """

    # 0.960 at ef=16 is a lone point above the target; the curve falls back to
    # 0.930 and 0.940 before clearing it properly at 48 and 64.
    DIPPED = {10: 0.900, 16: 0.960, 24: 0.930, 32: 0.940,
              48: 0.960, 64: 0.970}

    def test_a_lone_point_above_target_is_not_the_answer(self, tmp_path):
        dataset = tiny_dataset()
        driver = scripted(dataset, self.DIPPED)
        result, _rows = sweep(tmp_path, driver, dataset, sorted(self.DIPPED))
        assert result.ef_search == 48, (
            "ef 16 cleared the target once and ef 24 did not; 48 is the "
            "smallest ef whose successor confirms it")
        assert result.recall_at_k == pytest.approx(0.960)
        assert result.confirmed is True

    def test_the_sweep_does_not_stop_at_the_lone_point(self, tmp_path):
        """§2.4: a dip must not stop the sweep early. If it did, ef 16 would be
        the last point measured and the grid above it never seen."""
        dataset = tiny_dataset()
        driver = scripted(dataset, self.DIPPED)
        _result, rows = sweep(tmp_path, driver, dataset, sorted(self.DIPPED))
        assert driver.measured_efs == [10, 16, 24, 32, 48, 64]
        assert [r["ef_search"]
                for r in phases(rows, records_mod.PHASE_CALIBRATION)] == [
                    10, 16, 24, 32, 48, 64]

    @pytest.mark.parametrize("dip_at,expected", [
        # The dip falls between 10 and 24, so the first confirmed pair is
        # (24, 48) and 10 -- which cleared the target on its own -- is passed
        # over.
        (16, 24),
        # The dip falls later, so (10, 16) is the first confirmed pair.
        (48, 10),
    ])
    def test_the_clause_is_what_separates_the_candidates(self, tmp_path,
                                                         dip_at, expected):
        """Measured across the curve rather than at one shape of it: the same
        rule is asked where the dip sits early and where it sits late."""
        curve = {10: 0.960, 16: 0.960, 24: 0.960, 48: 0.960, 64: 0.970}
        curve[dip_at] = 0.930
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        result, _rows = sweep(tmp_path, driver, dataset, sorted(curve))
        assert result.ef_search == expected

    def test_a_target_met_only_at_the_last_grid_point_is_unconfirmed(self):
        """The grid runs out before a successor exists. The target *was* met,
        so the status stays `ok` and the ef is used -- but nothing confirmed
        it, and the result says so rather than implying the clause held."""
        result = calibrate_mod.select(
            curve_points({10: 0.900, 16: 0.910, 24: 0.960}),
            SPEC_TARGET_RECALL)
        assert result.ef_search == 24
        assert result.status == calibrate_mod.STATUS_OK
        assert result.confirmed is False


# ---------------------------------------------------------------------------
# 3. The three statuses
# ---------------------------------------------------------------------------

class TestTheThreeStatuses:
    """§2.1.6. Three fixtures, one per status, plus both sides of the floor
    rule's 0.02 margin -- the margin is the whole content of that status, and a
    fixture only on the far side of it would pass a rule written `>= target`.
    """

    def test_ok_is_the_target_met_inside_the_grid(self):
        result = calibrate_mod.select(
            curve_points({10: 0.900, 16: 0.952, 24: 0.960}),
            SPEC_TARGET_RECALL)
        assert result.status == calibrate_mod.STATUS_OK
        assert result.ef_search == 16

    def test_target_unreached_uses_the_largest_ef_and_flags_it(self):
        """§2.1.6: the engine is reported at its best achievable recall."""
        curve = {10: 0.700, 16: 0.800, 24: 0.880, 32: 0.910}
        result = calibrate_mod.select(curve_points(curve), SPEC_TARGET_RECALL)
        assert result.status == calibrate_mod.STATUS_TARGET_UNREACHED
        assert result.ef_search == 32
        assert result.recall_at_k == pytest.approx(0.910)
        assert result.confirmed is False

    def test_the_whole_grid_is_measured_when_nothing_reaches_the_target(
            self, tmp_path):
        """What makes "the largest ef" and "the largest *measured* ef" the same
        ef: early stop needs two consecutive points at the target, so a sweep
        that found none cannot have stopped short."""
        curve = {10: 0.700, 16: 0.800, 24: 0.880, 32: 0.910}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        result, _rows = sweep(tmp_path, driver, dataset, sorted(curve))
        assert driver.measured_efs == [10, 16, 24, 32]
        assert result.ef_search == max(curve)

    def test_floor_above_target_uses_the_smallest_ef(self):
        """The case PR 2 expects of MHNSW and VIDX at M=16: an engine already
        above the target at minimum effort, which no grid point can slow down
        to it. Whether those two engines are in this case is a measurement the
        run makes, not a claim this test is in a position to check -- what is
        checked here is that the sweep reports the case instead of presenting
        the engine as calibrated."""
        curve = {10: 0.980, 16: 0.985, 24: 0.990}
        result = calibrate_mod.select(curve_points(curve), SPEC_TARGET_RECALL)
        assert result.status == calibrate_mod.STATUS_FLOOR_ABOVE_TARGET
        assert result.ef_search == 10
        assert result.recall_at_k == pytest.approx(0.980)

    @pytest.mark.parametrize("floor,status", [
        # Inside the margin, and at it: "exceeds the target by more than 0.02"
        # is strict, so these stay `ok`.
        (0.968, "ok"),
        (0.970, "ok"),
        # ... and past it, it does not.
        (0.972, "floor_above_target"),
        (0.990, "floor_above_target"),
    ])
    def test_the_floor_margin_is_a_margin_and_not_the_target(self, floor,
                                                             status):
        curve = {10: floor, 16: 0.991, 24: 0.992}
        result = calibrate_mod.select(curve_points(curve), SPEC_TARGET_RECALL)
        assert result.status == status
        # Either way the smallest ef is the one used: a floor inside the margin
        # is also the first point above target, and its successor confirms it.
        assert result.ef_search == 10

    def test_the_margin_is_the_specified_one(self):
        assert calibrate_mod.FLOOR_MARGIN == 0.02

    def test_the_statuses_are_the_three_the_spec_names(self):
        assert (calibrate_mod.STATUS_OK,
                calibrate_mod.STATUS_TARGET_UNREACHED,
                calibrate_mod.STATUS_FLOOR_ABOVE_TARGET) == (
                    "ok", "target_unreached", "floor_above_target")

    def test_a_floor_above_target_is_judged_against_the_runs_own_target(self):
        """The margin is measured against the target the run asked for, not
        against the default: the same 0.960 floor is inside the margin at a
        0.95 target and well outside it at 0.90."""
        curve = {10: 0.960, 16: 0.991, 24: 0.992}
        assert calibrate_mod.select(curve_points(curve), 0.95).status == "ok"
        assert calibrate_mod.select(curve_points(curve), 0.90).status == (
            calibrate_mod.STATUS_FLOOR_ABOVE_TARGET)

    def test_the_result_carries_the_target_it_was_judged_against(self):
        curve = {10: 0.900, 16: 0.960, 24: 0.970}
        assert calibrate_mod.select(curve_points(curve),
                                    0.95).target_recall == 0.95


# ---------------------------------------------------------------------------
# 4. Early stop
# ---------------------------------------------------------------------------

class TestTheSweepStopsEarly:
    """§2.1.4. Larger ef values cost time and add nothing once the target is
    confirmed, and the cost is real: §2.5 budgets up to twelve points of 1,020
    queries each per engine-dataset."""

    def test_two_consecutive_points_at_target_end_the_sweep(self, tmp_path):
        curve = {10: 0.900, 16: 0.960, 24: 0.970, 32: 0.980, 48: 0.985}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        result, rows = sweep(tmp_path, driver, dataset, sorted(curve))
        assert driver.measured_efs == [10, 16, 24]
        assert len(phases(rows, records_mod.PHASE_CALIBRATION)) == 3
        assert result.ef_search == 16

    def test_one_point_at_target_does_not_end_it(self, tmp_path):
        curve = {10: 0.960, 16: 0.930, 24: 0.930, 32: 0.930}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        _result, _rows = sweep(tmp_path, driver, dataset, sorted(curve))
        assert driver.measured_efs == [10, 16, 24, 32]

    def test_the_grid_is_swept_in_ascending_order_however_it_arrives(
            self, tmp_path):
        """A descending grid swept as given would stop at the top two points
        and calibrate every engine to its largest ef."""
        curve = {10: 0.900, 16: 0.960, 24: 0.970, 32: 0.980}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        result, _rows = sweep(tmp_path, driver, dataset, [32, 24, 16, 10])
        assert driver.measured_efs == [10, 16, 24]
        assert result.ef_search == 16

    def test_a_repeated_grid_value_is_measured_once(self, tmp_path):
        curve = {10: 0.900, 16: 0.960, 24: 0.970}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        _result, _rows = sweep(tmp_path, driver, dataset, [10, 10, 16, 24, 24])
        assert driver.measured_efs == [10, 16, 24]

    def test_an_empty_grid_is_refused_rather_than_silently_skipped(
            self, tmp_path):
        dataset = tiny_dataset()
        driver = scripted(dataset, {})
        with pytest.raises(ValueError):
            sweep(tmp_path, driver, dataset, [])

    def test_the_selector_refuses_a_curve_with_no_points(self):
        with pytest.raises(ValueError):
            calibrate_mod.select([], SPEC_TARGET_RECALL)


# ---------------------------------------------------------------------------
# 5. The records the sweep writes
# ---------------------------------------------------------------------------

class TestTheCalibrationRecords:
    """§2.1.3 and §2.2.1: one record per grid point, under a phase of its own
    so the report can route the sweep away from the measurement charts."""

    CURVE = {10: 0.900, 16: 0.960, 24: 0.970}

    def _rows(self, tmp_path):
        dataset = tiny_dataset()
        driver = scripted(dataset, self.CURVE)
        result, rows = sweep(tmp_path, driver, dataset, sorted(self.CURVE))
        return driver, result, phases(rows, records_mod.PHASE_CALIBRATION)

    def test_the_phase_is_its_own_name_and_the_recorder_knows_it(self):
        assert records_mod.PHASE_CALIBRATION == "ef_calibration"
        assert records_mod.PHASE_CALIBRATION in records_mod.ALL_PHASES

    def test_a_calibration_record_survives_the_recorders_phase_check(
            self, tmp_path):
        """`Recorder.write` refuses an unknown phase, which is how a new phase
        missing from ALL_PHASES kills a run at its first record."""
        out = str(tmp_path / "one.jsonl")
        with Recorder(out) as recorder:
            recorder.write(Record(
                run_id="r", engine="e", phase=records_mod.PHASE_CALIBRATION,
                dataset="d", timestamp="t"))
        assert len(read_records(out)) == 1

    def test_one_record_per_measured_point_at_the_requested_ef(self, tmp_path):
        _driver, _result, rows = self._rows(tmp_path)
        assert [r["ef_search"] for r in rows] == [10, 16, 24]
        assert [r["recall_at_k"] for r in rows] == [0.9, 0.96, 0.97]

    def test_the_warm_up_queries_are_not_in_the_measured_count(self, tmp_path):
        """20 warm-up queries per point, then the calibration queries, and only
        the latter are counted or timed."""
        driver, _result, rows = self._rows(tmp_path)
        assert [r["queries_executed"] for r in rows] == [N_QUERIES] * 3
        per_point = [sum(1 for ef, _row in driver.queries if ef == point)
                     for point in (10, 16, 24)]
        assert per_point == [calibrate_mod.DEFAULT_WARMUP + N_QUERIES] * 3
        assert [r["extra"]["warmup_queries"] for r in rows] == [
            calibrate_mod.DEFAULT_WARMUP] * 3

    def test_each_record_carries_the_effective_ef_and_the_target(self, tmp_path):
        _driver, _result, rows = self._rows(tmp_path)
        assert [r["extra"]["effective_ef"] for r in rows] == [10, 16, 24]
        assert [r["extra"]["target_recall"] for r in rows] == [0.95] * 3
        assert [r["extra"]["reached_target"] for r in rows] == [
            False, True, True]

    def test_the_plan_is_checked_at_the_ef_the_record_reports(self, tmp_path):
        """`vector_index_used` is checked after each `set_ef_search`, not once
        before the sweep: a full scan reports recall 1.0 at the smallest ef and
        is otherwise indistinguishable from an engine whose floor is above the
        target, so the field has to describe the plan at the ef beside it."""
        dataset = tiny_dataset()
        driver = scripted(dataset, self.CURVE)
        checks: List[int] = []
        real = driver.explain_uses_vector_index

        def record_ef(vector, k, tag_threshold=None):
            checks.append(driver._ef_search)
            return real(vector, k, tag_threshold)

        driver.explain_uses_vector_index = record_ef
        _result, rows = sweep(tmp_path, driver, dataset, sorted(self.CURVE))
        calib = phases(rows, records_mod.PHASE_CALIBRATION)
        assert checks == [10, 16, 24]
        assert [r["vector_index_used"] for r in calib] == [True] * 3

    def test_an_engine_that_fell_back_to_a_scan_says_so_on_every_point(
            self, tmp_path):
        dataset = tiny_dataset()
        driver = scripted(dataset, self.CURVE, uses_vector_index=False)
        _result, rows = sweep(tmp_path, driver, dataset, sorted(self.CURVE))
        calib = phases(rows, records_mod.PHASE_CALIBRATION)
        assert len(calib) == 3
        assert [r["vector_index_used"] for r in calib] == [False] * 3

    def test_each_record_says_where_in_the_grid_it_sat(self, tmp_path):
        """A sweep that stopped early recorded three of five points, and
        nothing else in the record says the grid was longer than the sweep."""
        dataset = tiny_dataset()
        curve = {**self.CURVE, 32: 0.98, 48: 0.99}
        driver = scripted(dataset, curve)
        _result, rows = sweep(tmp_path, driver, dataset, sorted(curve))
        calib = phases(rows, records_mod.PHASE_CALIBRATION)
        assert [r["extra"]["grid_position"] for r in calib] == [0, 1, 2]
        assert [r["extra"]["grid_size"] for r in calib] == [5, 5, 5]

    def test_the_records_carry_the_runs_provenance(self, tmp_path):
        _driver, _result, rows = self._rows(tmp_path)
        # Counted first: a loop over an empty list asserts nothing, and a
        # sweep that wrote no records at all would otherwise pass this.
        assert len(rows) == 3
        for r in rows:
            assert r["run_id"] == "run-0"
            assert r["engine"] == "fake"
            assert r["engine_tag"] == "tag-0"
            assert r["resource_pass"] == "normalized"
            assert r["m"] == 16
            assert r["k"] == K
            assert r["clients"] == 1
            assert r["qps"] > 0
            assert r["extra"]["ops_measurement_version"] == (
                records_mod.OPS_MEASUREMENT_VERSION)

    def test_the_result_carries_the_whole_curve_it_measured(self, tmp_path):
        """So a caller -- the report, a later task -- can say why this ef was
        chosen without re-reading the records file."""
        _driver, result, rows = self._rows(tmp_path)
        assert [p.ef_search for p in result.points] == [10, 16, 24]
        # `approx`, because the record rounds to six places on the way to the
        # file and the point in memory does not.
        assert [p.recall_at_k for p in result.points] == pytest.approx(
            [r["recall_at_k"] for r in rows], abs=1e-6)

    def test_the_result_is_constructible_the_way_the_spec_names_it(self):
        """§2.1.5: `CalibrationResult(ef, recall, status)`. Every field this
        adds carries a default, so that call stays valid."""
        result = calibrate_mod.CalibrationResult(48, 0.96, "ok")
        assert (result.ef_search, result.recall_at_k, result.status) == (
            48, 0.96, "ok")


# ---------------------------------------------------------------------------
# 6. Ground truth
# ---------------------------------------------------------------------------

class TestGroundTruthFollowsTheRowsTheEngineHolds:
    """§2.1.2. Calibration outputs recall, so the truth it scores against has
    to describe the rows the engine was actually given."""

    def test_only_the_first_max_queries_test_vectors_are_used(self, tmp_path):
        curve = {10: 0.900, 16: 0.960, 24: 0.970}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        _result, rows = sweep(tmp_path, driver, dataset, sorted(curve),
                              max_queries=30)
        calib = phases(rows, records_mod.PHASE_CALIBRATION)
        assert [r["queries_executed"] for r in calib] == [30, 30, 30]
        assert {row for _ef, row in driver.queries} == set(range(30))

    def test_the_truth_is_computed_over_the_rows_the_engine_holds(self,
                                                                  tmp_path):
        """A subset profile (smoke loads 20,000 of fashion-mnist's 60,000)
        would otherwise be scored against neighbours pointing at rows it never
        received -- recall near a half for an exact engine, which reads as a
        result rather than as a bug."""
        dataset = tiny_dataset()
        half = len(dataset.train) // 2
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, ef_half=0.0)
        driver.connect()
        driver.create_schema(index_spec(dataset))
        driver.load(dataset.train[:half], ds.assign_tags(half))

        out = str(tmp_path / "subset.jsonl")
        recorder = Recorder(out)
        try:
            result = calibrate_mod.run(
                context(tmp_path, recorder), driver, dataset,
                index_spec(dataset), ef_grid=(10, 16), target_recall=0.95,
                max_queries=N_QUERIES, indexed_rows=half)
        finally:
            recorder.close()
        # An exact engine (ef_half = 0) over the rows it holds: recall 1.0, and
        # therefore a floor above the target rather than a collapse to a half.
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.status == calibrate_mod.STATUS_FLOOR_ABOVE_TARGET


# ---------------------------------------------------------------------------
# 7. effective_ef
# ---------------------------------------------------------------------------

class TestEffectiveEf:
    """§2.1.7. A reader comparing two engines at "ef 10" needs to know that one
    of them searched 10 candidates and the other searched k."""

    def test_the_base_class_returns_what_it_was_asked_for(self):
        dataset = tiny_dataset()
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        assert driver.effective_ef(10, 10) == 10
        assert driver.effective_ef(4, 10) == 4
        assert driver.effective_ef(1000, 10) == 1000

    @staticmethod
    def _mongo(monkeypatch):
        """A MongoDriver whose vector encoding does not need pymongo, exactly
        as `tests/test_config.py` builds one: the host this suite runs on has
        no Mongo client, and the clamp is pure arithmetic."""
        from harness.drivers import mongo
        monkeypatch.setattr(mongo, "Binary", lambda payload, subtype: payload)
        return mongo.MongoDriver(ConnectionSpec(host="h", port=1))

    @pytest.mark.parametrize("ef,k,expected", [
        (10, 10, 10),    # the default grid floor at the default k: no clamp
        (4, 10, 10),     # below k: clamped up
        (10, 16, 16),    # a profile with k=16 clamps the grid floor
        (10, 100, 100),  # ... and k=100 clamps four of the twelve grid points
        (1000, 10, 1000),
    ])
    def test_mongo_cannot_search_fewer_candidates_than_k(self, monkeypatch,
                                                         ef, k, expected):
        assert self._mongo(monkeypatch).effective_ef(ef, k) == expected

    def test_mongos_answer_is_what_its_pipeline_actually_sends(self,
                                                               monkeypatch):
        """The claim measured against the query rather than against a second
        copy of the rule: `_pipeline` is what reaches the server, and
        `effective_ef` is a promise about it."""
        driver = self._mongo(monkeypatch)
        for ef, k in ((10, 10), (4, 10), (10, 16), (10, 100), (1000, 10)):
            driver.set_ef_search(ef)
            stage = driver._pipeline([0.0], k)[0]["$vectorSearch"]
            assert stage["numCandidates"] == driver.effective_ef(ef, k), (ef, k)

    def test_the_record_carries_the_drivers_answer_not_the_request(self,
                                                                   tmp_path):
        """What the workload is responsible for: asking the driver rather than
        echoing the grid value back into the record."""
        dataset = tiny_dataset()

        class ClampedDriver(ScriptedDriver):
            def effective_ef(self, ef: int, k: int) -> int:
                return max(int(ef), 40)

        curve = {10: 0.900, 16: 0.960, 24: 0.970}
        driver = ClampedDriver(dataset, curve)
        driver.connect()
        result, rows = sweep(tmp_path, driver, dataset, sorted(curve))
        calib = phases(rows, records_mod.PHASE_CALIBRATION)
        assert [r["ef_search"] for r in calib] == [10, 16, 24]
        assert [r["extra"]["effective_ef"] for r in calib] == [40, 40, 40]
        assert result.ef_search == 16 and result.effective_ef == 40

    def test_every_driver_in_the_run_answers_the_question(self):
        """The workload calls this on whatever driver the run gave it, so a
        driver that does not implement it drops the sweep on its face."""
        from harness.drivers.postgres import _driver_table
        table = _driver_table()
        assert len(table) >= 6, "the driver table is empty or unreadable"
        for engine, cls in table.items():
            assert callable(getattr(cls, "effective_ef", None)), engine

    def test_exactly_the_two_clamping_drivers_override_it(self):
        """`base.py` names the drivers that clamp, and this is that sentence.

        Measured, because a driver quietly acquiring or losing a clamp would
        make the base class's claim false with the suite green -- which is
        what it caught: this used to read `== ["MongoDriver"]` and fired the
        moment `PerconaDriver` landed, with `base.py` still saying "every
        engine here except one". Both were corrected together.

        A value assertion and not a membership one, deliberately. The whole
        point is to notice a THIRD.
        """
        from harness.drivers.base import EngineDriver
        from harness.drivers.postgres import _driver_table
        overriding = sorted(
            {cls.__name__ for cls in _driver_table().values()
             if cls.effective_ef is not EngineDriver.effective_ef})
        # MyVectorDriver is the third: hnswlib's `searchKnn` searches at
        # `std::max(ef_, k)`, and the driver reports it.
        # VillageSQLDriver is the fourth: the scan searches at
        # `std::max<uint32_t>(k, ef_search)`.
        assert overriding == ["MongoDriver", "MyVectorDriver", "PerconaDriver",
                              "VillageSQLDriver"]
        # What each clamp actually does is exercised where it belongs:
        # MongoDB's below, Percona's in tests/test_e2_percona_wiring.py.

    def test_the_clamp_bites_exactly_the_grid_points_below_k(self, monkeypatch):
        """The arithmetic `mongo.py`'s docstring states, counted against the
        grid rather than written down beside it: six of the twelve default grid
        points collapse onto a k of 100, and none of them onto a k of 10."""
        driver = self._mongo(monkeypatch)
        collapsed = [ef for ef in SPEC_EF_GRID
                     if driver.effective_ef(ef, 100) != ef]
        assert collapsed == [10, 16, 24, 32, 48, 64]
        assert [ef for ef in SPEC_EF_GRID
                if driver.effective_ef(ef, 10) != ef] == []


class TestTheDriverIsLeftAtTheChosenEf:
    def test_the_sweep_does_not_leave_the_session_at_the_last_grid_point(
            self, tmp_path):
        """The sweep's last measurement is at the largest ef it tried, which is
        not the ef the run goes on to measure at. Every later workload sets its
        own, but the plan check between them does not."""
        curve = {10: 0.900, 16: 0.960, 24: 0.970}
        dataset = tiny_dataset()
        driver = scripted(dataset, curve)
        result, _rows = sweep(tmp_path, driver, dataset, sorted(curve))
        assert driver.ef_calls[-1] == result.ef_search == 16


# ---------------------------------------------------------------------------
# 8. The harness wiring
# ---------------------------------------------------------------------------

class TestTheHarnessWiring:
    """§2.2.2 and §2.2.5, driven through `harness.main.main` with a fake
    driver: the dataset loader and the driver factory are replaced and nothing
    else is, so argument parsing, the calibration call, the substitution of the
    calibrated ef and the stamping of every downstream record are all the real
    code.

    `ef_half = 2` over the grid [8, 18, 98, 198] puts the answer at 98 (0.980
    measured against a 0.95 target, five standard errors clear), and
    `--ef-search 100` is deliberately NOT a grid value: any grid value in a
    downstream record proves the substitution happened.
    """

    GRID = "8,18,98,198"
    CALIBRATED = 98
    FIXED = 100

    def _run(self, tmp_path, monkeypatch, extra_argv, ef_half=2.0):
        from harness import main as main_mod
        dataset = tiny_dataset()
        fake = loaded_driver(dataset, ef_half=ef_half)
        monkeypatch.setattr(main_mod, "load_dataset",
                            lambda directory, name: dataset)
        monkeypatch.setattr(main_mod, "get_driver",
                            lambda *a, **kw: fake.clone())
        out = str(tmp_path / "ops.jsonl")
        argv = [
            "--engine", "fake", "--host", "h", "--port", "0",
            "--dataset", dataset.name, "--datasets-dir", str(tmp_path),
            "--m", "16", "--run-id", "run-0", "--output", out,
            "--cache-dir", str(tmp_path / "cache"),
            "--ef-search", str(self.FIXED),
            "--max-queries", str(N_QUERIES),
            "--client-counts", "1", "--concurrency-duration", "0.2",
            "--concurrency-queries", "20",
            "--churn-fractions", "0.10", "--selectivities", "0.50",
        ] + extra_argv
        assert main_mod.main(argv) == 0
        return read_records(out)

    def test_the_calibrated_ef_is_what_the_later_workloads_measure_at(
            self, tmp_path, monkeypatch):
        rows = self._run(tmp_path, monkeypatch,
                         ["--workloads", "concurrency,filtered,churn",
                          "--ef-grid", self.GRID])
        calib = phases(rows, records_mod.PHASE_CALIBRATION)
        # Four points, not three: confirming 98 means measuring 198, which is
        # what "two consecutive grid points both reach the target" costs. The
        # grid above the pair is what early stop saves, and this grid has none.
        assert [r["ef_search"] for r in calib] == [8, 18, 98, 198]
        downstream = [r for r in rows
                      if r["phase"] != records_mod.PHASE_CALIBRATION]
        assert downstream, "no measurement records were written"
        assert {r["phase"] for r in downstream} == {
            records_mod.PHASE_CONCURRENCY, records_mod.PHASE_FILTERED,
            records_mod.PHASE_CHURN}
        assert {r["ef_search"] for r in downstream} == {self.CALIBRATED}

    def test_fixed_mode_reproduces_todays_behaviour(self, tmp_path,
                                                     monkeypatch):
        """§2.4's fourth test: no calibration records, `--ef-search` passed
        through to every workload."""
        rows = self._run(tmp_path, monkeypatch,
                         ["--workloads", "concurrency,filtered,churn",
                          "--ef-grid", self.GRID, "--ef-mode", "fixed"])
        assert phases(rows, records_mod.PHASE_CALIBRATION) == []
        assert {r["ef_search"] for r in rows} == {self.FIXED}

    def test_every_downstream_record_says_which_policy_produced_it(
            self, tmp_path, monkeypatch):
        """§2.2.5. The three keys reach every workload's records, including the
        workloads this task never edited."""
        rows = self._run(tmp_path, monkeypatch,
                         ["--workloads", "concurrency,filtered,churn",
                          "--ef-grid", self.GRID, "--target-recall", "0.95"])
        downstream = [r for r in rows
                      if r["phase"] != records_mod.PHASE_CALIBRATION]
        assert len(downstream) >= 3
        for r in downstream:
            assert r["extra"]["ef_mode"] == "calibrated", r["phase"]
            assert r["extra"]["target_recall"] == 0.95, r["phase"]
            assert r["extra"]["calibration_status"] == "ok", r["phase"]

    def test_a_fixed_mode_record_does_not_claim_a_target_it_never_pursued(
            self, tmp_path, monkeypatch):
        """`ef_mode` says `fixed`, and no target or status is asserted: in
        fixed mode nothing aimed at a recall, and a record carrying
        `target_recall: 0.95` beside an ef nobody chose for it would be a claim
        about a calibration that never ran."""
        rows = self._run(tmp_path, monkeypatch,
                         ["--workloads", "concurrency", "--ef-mode", "fixed"])
        downstream = [r for r in rows
                      if r["phase"] != records_mod.PHASE_CALIBRATION]
        assert downstream
        for r in downstream:
            assert r["extra"]["ef_mode"] == "fixed"
            assert "target_recall" not in r["extra"]
            assert "calibration_status" not in r["extra"]

    def test_the_status_reaches_the_records_when_the_target_is_missed(
            self, tmp_path, monkeypatch):
        """A run reported at its best achievable recall has to say so on every
        record and not only in the log: `ef_half = 20` over this grid tops out
        at 0.908."""
        rows = self._run(tmp_path, monkeypatch,
                         ["--workloads", "concurrency",
                          "--ef-grid", self.GRID], ef_half=20.0)
        downstream = [r for r in rows
                      if r["phase"] != records_mod.PHASE_CALIBRATION]
        assert downstream
        assert {r["extra"]["calibration_status"] for r in downstream} == {
            "target_unreached"}
        assert {r["ef_search"] for r in downstream} == {198}

    def test_a_run_with_no_query_workload_does_not_pay_for_the_sweep(self):
        from harness import main as main_mod
        assert main_mod._should_calibrate("calibrated", ["concurrency"]) is True
        assert main_mod._should_calibrate("calibrated", ["filtered"]) is True
        assert main_mod._should_calibrate("calibrated", ["churn"]) is True
        assert main_mod._should_calibrate("calibrated", ["build"]) is False
        assert main_mod._should_calibrate("fixed", ["concurrency"]) is False

    def test_a_build_only_run_writes_no_calibration_records(self, tmp_path,
                                                            monkeypatch):
        rows = self._run(tmp_path, monkeypatch,
                         ["--workloads", "build", "--ef-grid", self.GRID,
                          "--subset-rows", "500"])
        assert phases(rows, records_mod.PHASE_CALIBRATION) == []
        assert rows, "the build workload recorded nothing"

    def test_the_parser_defaults_are_the_specified_ones(self):
        from harness.main import parse_args
        args = parse_args(["--engine", "e", "--host", "h", "--port", "1",
                           "--dataset", "d", "--m", "16", "--run-id", "r",
                           "--output", "o"])
        assert args.ef_mode == "calibrated"
        assert args.target_recall == SPEC_TARGET_RECALL
        assert tuple(int(x) for x in args.ef_grid.split(",")) == SPEC_EF_GRID

    def test_the_mode_is_constrained_to_the_two_that_exist(self):
        from harness.main import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--engine", "e", "--host", "h", "--port", "1",
                        "--dataset", "d", "--m", "16", "--run-id", "r",
                        "--output", "o", "--ef-mode", "sideways"])


# ---------------------------------------------------------------------------
# 9. The assembly point
# ---------------------------------------------------------------------------

class TestTheRunUnitHandsTheArgumentsToTheHarness:
    """`orchestrator/cli.py::_run_unit` is where the harness command line is
    built, and until recently nothing in this repo's history executed it. That
    is how `ops.pgvector_iterative_scan` reached three config files, the SQL
    and a report column while 69 of 69 records read null: every piece was
    tested and the assembly was not.

    So these drive `_run_unit`. The first reads the call it makes to
    `harness_args` -- the arguments the one untested line actually passes --
    and the rest read the argv that `run_harness` receives through the real
    `harness_args`, rather than a value the test computed for itself. No
    container and no network: `OpsRun` is a fake whose `run_harness` returns 0.
    """

    @staticmethod
    class _FakeRun:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run_harness(self, *a, **kw):
            return 0

    @staticmethod
    def _paths(tmp_path):
        return {"run_dir": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints")}

    def _call_unit(self, tmp_path, profile, engine, monkeypatch, fake_run):
        from orchestrator import cli as cli_mod
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        monkeypatch.setattr(ops_pass, "OpsRun", fake_run)
        rc = cli_mod._run_unit(
            "ops", engine, "glove-100-angular", profile, {},
            load_resources("normalized"), None, "normalized",
            self._paths(tmp_path), "run-0",
            argparse.Namespace(force=False), checkpoints=None)
        assert rc == 0

    def test_the_unit_hands_harness_args_the_profile_the_keys_live_in(
            self, tmp_path, monkeypatch):
        """The one line in between, which no other test in this suite covers:
        every other test here either calls `harness_args` itself or reads the
        argv it produced. This captures what `_run_unit` passes it.

        The three new flags are read from the profile inside `harness_args`, so
        what the call site has to get right is handing over the profile it was
        given -- a call site that reached for a module-level default or a
        re-loaded profile would be invisible to a test that checked the argv of
        a profile it had not modified."""
        from orchestrator import ops_pass
        from orchestrator.config import load_profile

        profile = load_profile("main")
        profile = dict(profile, ops=dict(profile["ops"], ef_mode="fixed",
                                         target_recall=0.87,
                                         ef_calibration=[7, 11, 13]))
        captured = []

        def capture(*a, **kw):
            captured.append((a, kw))
            return ["--fake"]

        monkeypatch.setattr(ops_pass, "harness_args", capture)
        self._call_unit(tmp_path, profile, "mariadb", monkeypatch,
                        self._FakeRun)
        assert captured, "harness_args was never called"
        for args, kwargs in captured:
            # Identity, not equality: the profile the unit was given is the one
            # the arguments are built from.
            assert args[0] is profile
            assert kwargs.get("build_mode") == "post"
            assert kwargs.get("storage_engine") == "InnoDB"
            assert kwargs.get("iterative_scan") is None

    def _argv(self, tmp_path, profile, engine="mariadb"):
        captured: List[List[str]] = []
        outer = self

        class CapturingRun(outer._FakeRun):
            def run_harness(self, args, output, **kw):
                captured.append(list(args))
                return 0

        with pytest.MonkeyPatch.context() as mp:
            self._call_unit(tmp_path, profile, engine, mp, CapturingRun)
        assert captured, "run_harness was never called"
        return captured

    @staticmethod
    def _value(argv: List[str], flag: str) -> str:
        assert flag in argv, f"{flag} is not on the harness command line"
        return argv[argv.index(flag) + 1]

    def test_the_three_arguments_reach_the_harness_command_line(self, tmp_path):
        from orchestrator.config import load_profile
        for argv in self._argv(tmp_path, load_profile("main")):
            assert self._value(argv, "--ef-mode") == "calibrated"
            assert float(self._value(argv, "--target-recall")) == (
                SPEC_TARGET_RECALL)
            assert tuple(int(x) for x in
                         self._value(argv, "--ef-grid").split(",")) == (
                             SPEC_EF_GRID)

    def test_a_profile_that_sets_them_is_what_arrives(self, tmp_path):
        """The guard the test above cannot give: three hardcoded strings in
        `harness_args` would satisfy it. These values exist in no profile and
        in no default."""
        from orchestrator.config import load_profile
        profile = load_profile("main")
        profile = dict(profile, ops=dict(profile["ops"], ef_mode="fixed",
                                         target_recall=0.87,
                                         ef_calibration=[7, 11, 13]))
        for argv in self._argv(tmp_path, profile):
            assert self._value(argv, "--ef-mode") == "fixed"
            assert float(self._value(argv, "--target-recall")) == 0.87
            assert self._value(argv, "--ef-grid") == "7,11,13"

    def test_the_arguments_survive_every_ops_unit_of_the_loop(self, tmp_path):
        """pgvector runs the ops workloads once per build mode, so the flags
        have to be on every command line the unit issues rather than on the
        first."""
        from orchestrator.config import load_profile
        argvs = self._argv(tmp_path, load_profile("main"), engine="pgvector")
        assert len(argvs) == 2, "main measures both pgvector build modes"
        for argv in argvs:
            assert self._value(argv, "--ef-mode") == "calibrated"

    def test_the_harness_parses_what_the_runner_emits(self, tmp_path):
        """Both halves of the boundary in one assertion. The argv is built on
        the host and consumed by a parser in another container, and the two
        have disagreed before: a run reached the point of starting the 12.3
        server and died on `argument --engine: invalid choice`."""
        from harness.main import parse_args
        from orchestrator.config import load_profile
        argv = self._argv(tmp_path, load_profile("main"))[0]
        args = parse_args(argv + ["--engine", "mariadb", "--host", "h",
                                  "--port", "1", "--dataset", "d",
                                  "--run-id", "r", "--output", "o"])
        assert args.ef_mode == "calibrated"
        assert args.target_recall == SPEC_TARGET_RECALL
        assert tuple(int(x) for x in args.ef_grid.split(",")) == SPEC_EF_GRID


# ---------------------------------------------------------------------------
# 10. The defaults, on both sides of the container boundary
# ---------------------------------------------------------------------------

class TestACalibratedNumberIsNotAFixedEfNumber:
    """Why PR 2 owes an `OPS_MEASUREMENT_VERSION` bump.

    A version 3 concurrency QPS is throughput at a *declared* ef_search -- 100
    for every engine -- and a version 4 one is throughput at whatever ef puts
    that engine at the target recall. One holds the knob fixed and lets quality
    vary; the other holds quality fixed and lets the knob vary. They are not
    two samples of one quantity.

    `extra.ef_mode` tells a reader which policy produced a number, and that is
    not the same thing as a guard: the version is what makes the report
    *refuse*, and without the bump two concurrency records -- one fixed at
    ef_search 100, one calibrated at 98 -- load together and land on one chart,
    which is the comparison this whole change exists to prevent. So the
    refusal is what is tested here, not the diagnostic.
    """

    def test_the_current_version_is_not_claimed_to_match_an_older_one(self):
        """Derived, not a literal: the equivalence set is built from
        IMPLICIT_ and FIRST_RECORDED_, so the current version cannot enter it
        without someone naming a third concept and saying why that version
        changed nothing. A bump that changed what a number means must not be
        in there, and this one did."""
        assert records_mod.OPS_MEASUREMENT_VERSION >= 4, (
            "PR 1 took this to 3 for the concurrency query pool; PR 2 chooses "
            "ef_search instead of declaring it, which is a second change to "
            "what the same number means")
        assert (records_mod.OPS_MEASUREMENT_VERSION
                not in records_mod.OPS_MEASUREMENT_EQUIVALENT_VERSIONS)
        assert not records_mod.ops_versions_measure_the_same_thing(
            [3, records_mod.OPS_MEASUREMENT_VERSION])
        assert set(records_mod.OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == {1, 2}

    def test_a_declared_ef_run_and_a_calibrated_one_cannot_share_a_chart(
            self, tmp_path):
        """The guard itself, end to end through the report's own loader: a run
        directory holding a record measured at the profile's literal ef and a
        record this harness wrote at a calibrated ef is refused before a chart
        is drawn. Without the bump both carry the same version and the two ef
        values go on one series."""
        from report.loaders import MixedMeasurementVersions, load_ops_records

        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps({
            "run_id": "r", "engines": {}, "phases": [], "warnings": [],
            "config": {"resource_pass": "normalized",
                       "ops_measurement_version": 3}}))
        path = str(run / "ops-mariadb-glove-100-angular-normalized-m16.jsonl")
        # What a version 3 run recorded: throughput at the declared ef.
        with open(path, "w") as fh:
            fh.write(json.dumps({
                "run_id": "r", "engine": "mariadb",
                "phase": records_mod.PHASE_CONCURRENCY,
                "dataset": "glove-100-angular", "m": 16, "clients": 1,
                "ef_search": 100, "qps": 326.0, "recall_at_k": 0.98,
                "resource_pass": "normalized",
                "timestamp": "2026-01-01T00:00:00Z",
                "extra": {"ops_measurement_version": 3,
                          "ef_mode": "fixed"}}) + "\n")
        # What this harness appends: throughput at the calibrated ef.
        with Recorder(path) as recorder:
            recorder.declare_run_extra(
                **calibrate_mod.status_extra("calibrated",
                                             calibrate_mod.select(
                                                 curve_points({10: 0.90,
                                                               98: 0.952,
                                                               160: 0.97}))))
            recorder.write(Record(
                run_id="r", engine="mariadb",
                phase=records_mod.PHASE_CONCURRENCY,
                dataset="glove-100-angular",
                timestamp="2026-01-01T01:00:00Z", m=16, clients=1,
                ef_search=98, qps=331.0, recall_at_k=0.952,
                resource_pass="normalized"))

        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(str(run))
        message = str(caught.value)
        assert "version 3" in message
        assert f"version {records_mod.OPS_MEASUREMENT_VERSION}" in message
        # The wording that must not be used for this pair: these are two
        # measurements, not one measurement under two labels.
        assert "two different measurements" in message
        assert "denote the same measurement" not in message


class TestTheDefaultsOnBothSidesOfTheContainerBoundary:
    """`harness/` cannot import `orchestrator`, and `orchestrator/` must not
    import `harness` at module scope -- the host's only Python dependencies are
    docker, python3, pyyaml, git and curl, and `harness.workloads.calibrate`
    reaches numpy. So the defaults are declared on each side and pinned
    together here, which is how `ops.selectivities` and `ops.churn_fractions`
    are already handled.
    """

    def test_the_grid_is_the_one_the_spec_specifies(self):
        """Denser at the low end, where MHNSW and pgvector cross 0.95."""
        assert tuple(calibrate_mod.DEFAULT_EF_GRID) == SPEC_EF_GRID

    def test_the_runner_and_the_workload_declare_the_same_defaults(self):
        from orchestrator import ops_pass
        assert tuple(ops_pass.DEFAULT_EF_CALIBRATION) == tuple(
            calibrate_mod.DEFAULT_EF_GRID)
        assert ops_pass.DEFAULT_TARGET_RECALL == (
            calibrate_mod.DEFAULT_TARGET_RECALL)
        assert ops_pass.DEFAULT_EF_MODE == calibrate_mod.EF_MODE_CALIBRATED

    def test_the_spec_default_target_is_the_declared_one(self):
        assert calibrate_mod.DEFAULT_TARGET_RECALL == SPEC_TARGET_RECALL

    def test_the_warm_up_is_the_twenty_queries_the_spec_names(self):
        assert calibrate_mod.DEFAULT_WARMUP == 20

    def test_the_query_cap_is_the_one_the_harness_already_carries(self):
        """`DEFAULT_MAX_QUERIES` says in its docstring that it is the same
        default as `--max-queries`, and main.py passes that flag through to the
        sweep. Two spellings of one number, so they are pinned together."""
        from harness.main import parse_args
        args = parse_args(["--engine", "e", "--host", "h", "--port", "1",
                           "--dataset", "d", "--m", "16", "--run-id", "r",
                           "--output", "o"])
        assert args.max_queries == calibrate_mod.DEFAULT_MAX_QUERIES == 1_000

    def test_todays_profiles_calibrate_without_carrying_the_keys(self):
        """No profile declares these keys yet, so the defaults are what every
        profile in the repo will run with."""
        from orchestrator.config import available_profiles, load_profile
        from orchestrator.ops_pass import harness_args
        seen = 0
        for name in available_profiles():
            profile = load_profile(name)
            ops = profile.get("ops", {}) or {}
            assert "ef_mode" not in ops, name
            assert "target_recall" not in ops, name
            assert "ef_calibration" not in ops, name
            argv = harness_args(profile, 16, "mariadb", None, "normalized", {})
            assert argv[argv.index("--ef-mode") + 1] == "calibrated", name
            seen += 1
        assert seen >= 6, "the profile directory is empty or unreadable"

    def test_the_orchestrator_does_not_import_the_workload_to_get_them(self):
        """The host runs on docker, python3, pyyaml, git and curl. Importing
        `harness.workloads.calibrate` from `ops_pass` -- at module scope or
        from inside `harness_args`, which also runs on the host -- would put
        numpy on that list, and a benchmark framework must not modify the
        machine it measures.

        The whole file, not just its head: a deferred import inside
        `harness_args` would pass a check that only read the imports at the
        top and would still fail on a bare host mid-run."""
        with open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")) as fh:
            source = fh.read()
        assert "import harness" not in source
        assert "from harness" not in source

    def test_the_argument_list_can_be_built_on_a_host_without_numpy(self):
        """The claim behind the duplicated defaults, measured over the whole
        import closure rather than over the import lines. `numpy` is poisoned
        in a subprocess, so any module on the path from `ops_pass` to the
        emitted argv that reaches for it fails loudly."""
        import subprocess
        code = (
            "import sys; sys.modules['numpy'] = None;\n"
            "from orchestrator.config import load_profile\n"
            "from orchestrator.ops_pass import harness_args\n"
            "argv = harness_args(load_profile('main'), 16, 'mariadb', None,"
            " 'normalized', {})\n"
            "print(argv[argv.index('--ef-mode') + 1],"
            " argv[argv.index('--ef-grid') + 1])\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=VB_ROOT,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == [
            "calibrated", ",".join(str(ef) for ef in SPEC_EF_GRID)]
