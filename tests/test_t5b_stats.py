"""T5b -- PR 5 §5.1 items 2-6, §5.2, and the build-cost chart's two axes.

Three things in this file are not the obvious tests, and each exists because
the obvious test passes on a defect this task could have shipped.

**1. A tie is a claim.** §5.1 item 6 asks the report to mark two engines as
tied when their intervals overlap. Writing that as a boolean overlap test
looks right and publishes two false statements. Measured on master's
`bootstrap_ci`: `bootstrap_ci([1.0] * 50)` and `bootstrap_ci([0.97])` both
return a width of exactly 0.0 -- 50 queries that all found the whole top-k,
and one query -- and the bounds alone cannot tell them apart. And a
`(nan, nan)` interval from a measurement that never happened is False under
BOTH `a.low <= b.high and b.low <= a.high` and its strict form, so a boolean
column reports "these two engines are ordered" for an engine with no
measurement. Hence `Interval.n`, hence three verdicts, and hence
`TestATieIsAClaim` below, which runs the mutant rather than describing it.

**2. The §5.2 match key had a second asymmetry nobody had found.** Item 1
omits `storage_engine` and names `build_mode`, and the two are the same
problem: the ops harness stamps a default on every record for both
(`--storage-engine InnoDB`, `--build-mode post`) while the ann modules for
MariaDB, MariaDB 12.3 and AliSQL declare neither. Taken literally the check
is empty for half the engines. `TestTheMatchKeyReconcilesWithTheOpsUnit` pins
the reconciliation, in both directions.

**3. A coverage test with a fixed seed is deterministic, so its band is a
property of the seed.** §5.4 asks for 200 trials and a 90-99% band. Stated
and proven rather than left implicit: the same seed gives the same number
twice in one session, and a different seed moves it.

**4. A guard keyed to the right answer cannot see a wrong one.** This is a
vacuity shape the catalogue in the brief does not list, and it was found here
by mutating a guard this task had just written. The guard was meant to stop a
stale line count returning to a comment, and it was written as:

    assert f"{together:,} lines" not in source

`together` is the CURRENT count. So it catches a restatement that happens to
be *correct* -- the harmless case -- and passes any stale one. Measured:
reintroducing "the two files are 1,151 lines" sailed through, while the
genuine defect it existed to prevent is exactly a figure that no longer
matches. The fix is to key the guard on the SHAPE of the thing being banned
(a thousands-separated number anywhere inside the justification
paragraph, matched as a pattern) rather than on its correct
value.

It generalises: any assertion of the form "the wrong value is absent" is only
a guard if it can enumerate the wrong values, and for a number it cannot.
Negative assertions about data have to match a pattern, not a reading. The
same reasoning is why `TestTheForbiddenVerdictIsAbsentEvenUnwrapped` searches
for a phrase after rejoining line wraps and string-literal splits instead of
for its one formatted spelling.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Dict, List

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds                                  # noqa: E402
from harness.datasets import (FILTER_CTAG, FILTER_TAG,              # noqa: E402
                              recall_at_k, recall_per_query)
from harness.drivers.base import RowAttrs                           # noqa: E402
from harness.metrics import records, stats                          # noqa: E402
from harness.workloads import calibrate as calw                     # noqa: E402
from harness.workloads import churn as cw                           # noqa: E402
from harness.workloads import filtered as fw                        # noqa: E402
from harness.workloads.context import RunContext                    # noqa: E402
from report import charts, crosscheck, generate, render             # noqa: E402

#: Every §5.2 mutation below patches `crosscheck`, NEVER `generate`.
#:
#: `report.generate` re-exports these names, so `generate.X = mutant`
#: rebinds a copy while `consistency_checks` keeps reading `X` out of
#: `crosscheck`'s own globals -- the module-level-container trap that
#: made three mutants against `orchestrator/estimate.py:_QUERY_TERMS`
#: report a fully green suite. It was caught here only because every
#: mutation below asserts a value that MOVED rather than a pass or a
#: fail: retargeting the split turned five of these tests red instead of
#: five of them vacuous.
#:
#: `TestTheMatchKeyReconcilesWithTheOpsUnit` asserts the two modules
#: still agree, so the facade cannot drift from the implementation.
MUTATION_TARGET = crosscheck
from tests.fake_driver import (FakeDriver, N_QUERIES, N_ROWS,       # noqa: E402
                               index_spec, tiny_dataset)
from tests.synthetic import (CROSSCHECK_PAIRS, make_manifest,       # noqa: E402
                             make_records)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K = 10

needs_matplotlib = pytest.mark.skipif(
    not charts.have_matplotlib(),
    reason="draws a chart; matplotlib lives in the bench images, not on the host")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="t5b-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


@pytest.fixture
def dataset():
    return tiny_dataset()


def written(ctx: RunContext) -> List[Dict[str, Any]]:
    return list(records.read_records(ctx.recorder.path))


def loaded(dataset, cache_dir, **kwargs) -> FakeDriver:
    """A driver holding the corpus with both filter columns genuinely apart."""
    rows = len(dataset.train)
    tags = ds.filter_values(cache_dir, dataset, FILTER_TAG, indexed_rows=rows)
    ctags = ds.filter_values(cache_dir, dataset, FILTER_CTAG,
                             indexed_rows=rows)
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, **kwargs)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, RowAttrs(tag=tags, ctag=ctags))
    return driver


def synthetic_summary():
    recs = make_records("synthetic-run")
    return recs, generate.summarize(recs, make_manifest("synthetic-run"))


def table_cells(markdown: str) -> List[List[str]]:
    """Header and body cells of the FIRST pipe table in a fragment.

    Rows are split on the delimiter, never searched with `in`: this repo has
    shipped ten substring defects, including `"tag"` satisfied by `"ctag"` and
    `"RESET (autovacuum_enabled)"` containing `"SET (autovacuum_enabled)"`.
    """
    out: List[List[str]] = []
    for line in markdown.splitlines():
        st = line.strip()
        if not (st.startswith("|") and st.endswith("|")):
            if out:
                break
            continue
        cells = [c.strip() for c in st.strip("|").split("|")]
        if all(set(c) <= set("-: ") and c for c in cells):
            continue
        out.append(cells)
    return out


# ---------------------------------------------------------------------------
# §5.4's first prescribed test, sealed
# ---------------------------------------------------------------------------

class TestBootstrapCoversTheTrueMean:
    """§5.4: "bootstrap_ci on a known normal sample covers the true mean in
    roughly 95% of seeded trials (run 200 trials, accept 90-99%)".

    Sealed, because an unseeded version of this would fail a few times a year
    in CI and teach everyone to re-run the suite. With the seeds fixed the
    number is deterministic, so the 90-99% band is a property of THESE seeds
    and not of the method -- which is fine and is said out loud here rather
    than being left for a reader to assume otherwise.
    """

    TRUE_MEAN = 10.0
    SIGMA = 2.0
    N = 40
    TRIALS = 200

    def _coverage(self, master_seed: int) -> float:
        """Fraction of trials whose interval contains the true mean.

        Each trial draws its own sample and gives `bootstrap_ci` its own
        resampling seed, so the trials are independent of each other and of
        the library default -- 200 trials that all resampled with
        `DEFAULT_SEED` would be 200 correlated readings of one PRNG stream.
        """
        rng = numpy.random.default_rng(master_seed)
        covered = 0
        for trial in range(self.TRIALS):
            sample = rng.normal(self.TRUE_MEAN, self.SIGMA, self.N)
            low, high = stats.bootstrap_ci(sample, numpy.mean,
                                           seed=master_seed + trial)
            if low <= self.TRUE_MEAN <= high:
                covered += 1
        return covered / self.TRIALS

    def test_coverage_is_within_the_band_the_spec_names(self):
        coverage = self._coverage(20261014)
        assert 0.90 <= coverage <= 0.99, coverage

    def test_the_figure_is_deterministic_in_one_session(self):
        """Twice in one session, same answer. Without this the band above
        could be passing on a coin flip and nobody would know which."""
        first = self._coverage(20261014)
        second = self._coverage(20261014)
        assert first == second, (first, second)

    def test_a_different_seed_moves_it(self):
        """The companion to determinism, and the thing that proves the test
        is measuring resampled intervals rather than a constant.

        A `bootstrap_ci` that ignored its input and returned
        `(-inf, +inf)` would cover the true mean in 200 of 200 trials under
        every seed -- which the band above accepts, since 1.00 is... not in
        it. It is not: the band's upper end is 0.99, so a 100% result fails.
        That is deliberate, and this test adds the other direction: two seeds
        give two different coverages, so the figure depends on the data.
        """
        seeds = [20261014, 991, 7]
        coverages = {s: self._coverage(s) for s in seeds}
        assert len(set(coverages.values())) > 1, coverages
        for seed, coverage in coverages.items():
            assert 0.90 <= coverage <= 0.99, (seed, coverage)

    def test_an_interval_that_ignored_the_data_would_fail_the_band(self):
        """The mutant for the band, run.

        Honest: coverage 0.9250 on seed 20261014 -- 185 of 200 trials. Not
            0.95: a percentile bootstrap of the mean of 40 samples is
            slightly anti-conservative, which is why §5.4 names a band
            rather than a point and why this test quotes the measured figure
            instead of the nominal one.
        Mutant (`bootstrap_ci` replaced by one returning (-1e9, 1e9)):
            coverage 1.0000, which is outside the 0.90-0.99 band the spec
            names -- so the band rejects a vacuously-wide interval rather
            than rewarding it.
        """
        honest = self._coverage(20261014)
        original = stats.bootstrap_ci
        try:
            stats.bootstrap_ci = lambda *a, **kw: (-1e9, 1e9)
            mutant = self._coverage(20261014)
        finally:
            stats.bootstrap_ci = original
        assert honest == 0.925, honest
        assert mutant == 1.0, mutant
        assert not (0.90 <= mutant <= 0.99), mutant
        # And the honest value is back, so the patch really was undone.
        assert self._coverage(20261014) == honest


# ---------------------------------------------------------------------------
# §5.1 item 2 -- per-query recall, and the mean that must not move
# ---------------------------------------------------------------------------

class TestPerQueryRecallReproducesThePublishedMean:
    """§5.1 item 2: "recall_at_k gains a sibling recall_per_query(); the mean
    stays the reported value"."""

    def _fixture(self):
        rng = numpy.random.default_rng(31337)
        truth = rng.integers(0, 10000, size=(37, K))
        returned = []
        for i, row in enumerate(truth):
            keep = int(rng.integers(0, K + 1))
            got = [int(x) for x in row[:keep]]
            got += [900000 + i * 100 + j for j in range(K - keep)]
            returned.append(got)
        return returned, truth

    def test_one_value_per_result_set(self):
        """`len(...) == len(returned)`, which is what makes the vector
        bootstrappable: an interval over fewer samples than the queries it
        claims to describe is an interval over a different measurement."""
        returned, truth = self._fixture()
        assert len(recall_per_query(returned, truth, K)) == len(returned)

    @staticmethod
    def _base_recall_at_k(returned, truth, k):
        """A VERBATIM copy of the implementation at 6ee0b40.

        `git show 6ee0b40:harness/datasets.py` -- the naive left-to-right
        accumulation PR 5 §5.1 item 2 replaced. Copied rather than imported
        because the point is to compare against what the report published
        BEFORE this task, and the module no longer contains it.
        """
        if len(returned) == 0:
            return float("nan")
        total = 0.0
        for got, want in zip(returned, truth):
            wanted = set(int(x) for x in want[:k])
            if not wanted:
                continue
            total += len(wanted.intersection(int(x) for x in got[:k])) \
                / len(wanted)
        return total / len(returned)

    def test_the_identity_form_of_this_test_proved_nothing(self):
        """Why the test below compares against a copy of the base loop.

        The first version asserted `sum(recall_per_query(...))/n ==
        recall_at_k(...)`. `recall_at_k`'s body IS that expression, so it
        asserted an identity and could not notice any error in either
        function.

        Honest: the identity holds, as it must.
        Mutant (`recall_per_query` patched to HALVE every value, so every
            number the report publishes is wrong by a factor of two): the
            identity STILL holds -- 0.2406... on both sides -- while the
            comparison against the base loop below moves by 0.2406, which is
            24 recall points.
        """
        returned, truth = self._fixture()
        original = ds.recall_per_query
        try:
            ds.recall_per_query = lambda g, w, k: [
                v / 2.0 for v in original(g, w, k)]
            per_query = ds.recall_per_query(returned, truth, K)
            identity = sum(per_query) / len(per_query) == ds.recall_at_k(
                returned, truth, K)
            halved = ds.recall_at_k(returned, truth, K)
        finally:
            ds.recall_per_query = original
        honest = ds.recall_at_k(returned, truth, K)
        base = self._base_recall_at_k(returned, truth, K)

        assert identity is True, "the tautology survives a 2x error"
        assert halved == pytest.approx(honest / 2.0), (halved, honest)
        assert abs(base - halved) > 0.2, (base, halved)
        # Where the real guard catches it: against the base loop.
        assert base == pytest.approx(honest, abs=1e-12)

    def test_the_mean_matches_the_pre_pr5_loop_to_the_stored_precision(self):
        """§5.1 item 2's "the mean stays the reported value", measured against
        the loop that reported it.

        Not bit-identity -- that claim was false. `sum()` is compensated in
        Python 3.12, so the raw means differ in the last bits. What has to
        hold is the figure anything downstream reads: `round(x, 6)`, which is
        what every caller stores, and the four decimals the report renders
        from it.

        Measured over 20,000 seeded fixtures: raw differs 6,132; stored
        `round(x, 6)` differs **1** (an exact rounding tie,
        0.57031250000000011 -> 0.570313 against 0.5703125 -> 0.570312); the
        rendered four decimals differ **0**. 2,000 fixtures here to keep the
        suite fast, and the 20,000 sweep is in the docstring of
        `harness/datasets.recall_at_k`.

        **`stored <= 1` below is a ceiling, not the guard**, and at 2,000
        fixtures it is satisfied with `stored == 0` -- so this test does not
        demonstrate the tie, it only refuses a world where the stored figure
        moves often. The tie itself is pinned by
        `test_the_rounding_tie_that_does_move_is_a_single_unit`, against its
        own numbers rather than against a count. The load-bearing assertions
        here are `rendered == 0` and `raw > 0`: together they say the change
        is real and invisible downstream, which is the claim.
        """
        rng = numpy.random.default_rng(99)
        raw = stored = rendered = 0
        worst = 0.0
        for _ in range(2000):
            n = int(rng.integers(1, 60))
            k = int(rng.integers(1, 12))
            truth = rng.integers(0, 5000, size=(n, k))
            returned = []
            for i, row in enumerate(truth):
                keep = int(rng.integers(0, k + 1))
                got = [int(x) for x in row[:keep]]
                got += [50000 + i * 50 + j for j in range(k - keep)]
                rng.shuffle(got)
                returned.append(got)
            old = self._base_recall_at_k(returned, truth, k)
            new = recall_at_k(returned, truth, k)
            if old != new:
                raw += 1
                worst = max(worst, abs(old - new))
            o6, n6 = round(old, 6), round(new, 6)
            if o6 != n6:
                stored += 1
            if f"{o6:.4f}" != f"{n6:.4f}":
                rendered += 1

        # The raw means DO differ -- asserting otherwise is the false claim
        # this test replaced, and a run of zero here would mean the sweep had
        # stopped exercising the compensated path at all.
        assert raw > 0, raw
        assert worst < 1e-14, worst
        # What the report reads does not move. THE assertion.
        assert rendered == 0, rendered
        # A ceiling on the stored figure, not a demonstration of the tie: at
        # this fixture count it passes with `stored == 0`, and the tie is
        # pinned by its own numbers in the next test. Kept because a world
        # where the six-decimal figure moved often would be a different
        # claim, and this is where that would show.
        assert stored <= 1, stored

    def test_the_rounding_tie_that_does_move_is_a_single_unit(self):
        """The one case in 20,000, pinned by its own numbers rather than by a
        count, so the paragraph in `recall_at_k` can be checked.

        A six-decimal figure that moved by more than one unit in the last
        place would be a different claim, and the tolerance floor in §5.2 is
        0.01 -- four orders of magnitude above this.
        """
        base, now = 0.57031250000000011, 0.5703125
        assert round(base, 6) == 0.570313
        assert round(now, 6) == 0.570312
        assert abs(round(base, 6) - round(now, 6)) == pytest.approx(1e-6)
        assert f"{round(base, 6):.4f}" == f"{round(now, 6):.4f}" == "0.5703"
        assert abs(base - now) < generate.CROSSCHECK_MIN_TOLERANCE / 1e12

    def test_the_compensated_sum_is_what_changed(self):
        """Names the mechanism, so the docstring's claim is executable.

        If a future interpreter reverted `sum` to naive accumulation this
        fails, and the paragraph in `recall_at_k` would need rewriting rather
        than silently becoming a description of nothing.
        """
        values = [1 / 3] * 3 + [1 / 7] * 7
        naive = 0.0
        for v in values:
            naive += v
        assert sum(values) == 2.0
        assert naive == 1.9999999999999996
        assert sum(values) != naive

    @pytest.mark.parametrize("name,returned,truth,expected_mean,expected_per", [
        # A query whose ground truth is empty. The old loop skipped the
        # numerator and still divided by every result set, so the published
        # mean already charged it a zero -- and the per-query vector has to
        # charge the same zero or the bar moves off the point.
        ("empty truth row", [[1, 2, 3]], numpy.zeros((1, 0), numpy.int64),
         0.0, [0.0]),
        # Two result sets against one truth row: `zip` dropped the second,
        # the denominator kept it. Measured on master: 0.5, not 1.0.
        ("more results than truth", [[1, 2, 3], [4, 5, 6]],
         numpy.array([[1, 2, 3]], numpy.int64), 0.5, [1.0, 0.0]),
    ])
    def test_the_two_edge_shapes_keep_their_published_values(
            self, name, returned, truth, expected_mean, expected_per):
        k = 3
        assert recall_at_k(returned, truth, k) == expected_mean, name
        assert recall_per_query(returned, truth, k) == expected_per, name

    def test_dropping_the_unscored_queries_would_move_the_published_mean(self):
        """The mutant for the paragraph above, run rather than described.

        The tempting "improvement" is to omit the queries that were never
        scored instead of charging them zero. It changes a number the report
        has published for its whole life, which §5.1 item 2 forbids.

        Honest: recall_at_k = 0.5 over two result sets and one truth row.
        Mutant (unscored queries omitted): the mean becomes 1.0 and the
            vector becomes one element long -- so the figure doubles AND the
            interval would be drawn from a vector shorter than the queries.
        """
        returned = [[1, 2, 3], [4, 5, 6]]
        truth = numpy.array([[1, 2, 3]], numpy.int64)
        honest_mean = recall_at_k(returned, truth, 3)
        honest_per = recall_per_query(returned, truth, 3)

        original = ds.recall_per_query
        try:
            ds.recall_per_query = lambda got, want, k: [
                v for i, v in enumerate(original(got, want, k))
                if i < len(want)]
            mutant_per = ds.recall_per_query(returned, truth, 3)
            mutant_mean = (sum(mutant_per) / len(mutant_per)) if mutant_per \
                else float("nan")
        finally:
            ds.recall_per_query = original

        assert honest_mean == 0.5 and honest_per == [1.0, 0.0]
        assert mutant_mean == 1.0 and mutant_per == [1.0]
        assert mutant_mean != honest_mean
        assert len(mutant_per) != len(returned)
        # Restored.
        assert ds.recall_per_query(returned, truth, 3) == honest_per


# ---------------------------------------------------------------------------
# §5.1 item 6 -- the tie, and the shapes that make it a false claim
# ---------------------------------------------------------------------------

class TestATieIsAClaim:
    """The collapse finding, measured end to end.

    Every number quoted in a docstring here is produced by the assertions in
    the same method, from `harness.metrics.stats` on master plus this task's
    additions. Nothing is supplied but the sample vectors, which are inline.
    """

    def test_the_axis_ignoring_statistic_collapses_the_interval(self):
        """Why `mean_interval` exists rather than each caller passing its own
        statistic to `bootstrap_ci`.

        `bootstrap_ci` calls `stat(resampled, axis=1)` on an
        `(n, len(values))` matrix and only falls back to `apply_along_axis` on
        TypeError. A statistic whose signature ACCEPTS `axis` and whose body
        ignores it therefore reduces the whole matrix to one scalar.

        Honest (numpy.mean, axis-aware): (0.00965000, 0.01095000), width
            1.300e-03.
        Mutant (a mean that accepts `axis` and ignores it): (0.01029375,
            0.01029375), width exactly 0.0 -- and 0.01029375 is not even the
            sample mean of 0.01030000, it is the mean of the whole resample
            matrix.
        """
        lat = [0.010, 0.011, 0.009, 0.0105, 0.0115, 0.0095, 0.012, 0.0085,
               0.010, 0.011]
        honest_low, honest_high = stats.bootstrap_ci(lat, numpy.mean)

        def ignores_axis(v, axis=None):
            return numpy.mean(v)

        mutant_low, mutant_high = stats.bootstrap_ci(lat, ignores_axis)

        assert (round(honest_low, 8), round(honest_high, 8)) == (0.00965,
                                                                 0.01095)
        assert honest_high - honest_low == pytest.approx(1.3e-3, rel=1e-6)
        assert mutant_low == mutant_high
        assert mutant_high - mutant_low == 0.0
        assert round(mutant_low, 8) == 0.01029375
        assert round(float(numpy.mean(lat)), 8) == 0.0103
        assert round(mutant_low, 8) != round(float(numpy.mean(lat)), 8)
        # `mean_interval` is the guard: it chooses the statistic, so the
        # collapse is not reachable through it.
        through_guard = stats.mean_interval(lat)
        assert (through_guard.low, through_guard.high) == (honest_low,
                                                           honest_high)

    def test_a_zero_width_interval_from_real_data_is_reachable(self):
        """The case that does not need a bad statistic at all, and is the
        common one: per-query recall of 1.0 on every query."""
        fifty = stats.mean_interval([1.0] * 50)
        one = stats.mean_interval([0.97])
        assert (fifty.low, fifty.high) == (1.0, 1.0)
        assert (one.low, one.high) == (0.97, 0.97)
        assert fifty.high - fifty.low == 0.0
        assert one.high - one.low == 0.0
        # Identical widths, different facts. `n` is what separates them, and
        # `usable` is what the report asks.
        assert fifty.n == 50 and one.n == 1
        assert fifty.usable is True
        assert one.usable is False

    def test_a_nan_interval_is_unknown_and_not_an_ordering(self):
        """The dangerous direction, measured.

        A boolean overlap test puts "no measurement" somewhere, and both
        placements publish something false.

        Honest (`tie_verdict`): "unknown".
        Mutant (closed boolean overlap): False -> the report says the two
            engines are distinguishable.
        Mutant (strict boolean overlap): False as well -- so the bug is NOT
            a matter of choosing the other comparison direction.
        """
        nan_iv = stats.mean_interval([])
        real = stats.mean_interval([0.95, 0.96, 0.94, 0.97, 0.93])
        assert nan_iv.low != nan_iv.low  # nan
        assert nan_iv.usable is False and real.usable is True
        assert stats.tie_verdict(nan_iv, real) == stats.TIE_UNKNOWN

        closed = nan_iv.low <= real.high and real.low <= nan_iv.high
        strict = nan_iv.low < real.high and real.low < nan_iv.high
        assert closed is False and strict is False

    def test_the_comparison_direction_changes_the_verdict(self):
        """Why the closed form is chosen and written down.

        `(1.0, 1.0)` against an interval that reaches 1.0 is TIED under the
        closed test and SEPARATED under the strict one, from the same two
        measurements -- so the choice is load-bearing and is not a detail.
        """
        a = stats.mean_interval([1.0] * 20)
        b = stats.mean_interval([1.0] * 18 + [0.98, 0.97])
        assert (a.low, a.high) == (1.0, 1.0)
        assert b.high == 1.0 and b.low < 1.0
        closed = a.low <= b.high and b.low <= a.high
        strict = a.low < b.high and b.low < a.high
        assert closed is True and strict is False
        # The shipped verdict is the closed one, which declines to separate
        # engines whose intervals touch.
        assert stats.tie_verdict(a, b) == stats.TIE_TIED

    def test_the_three_verdicts_are_all_reachable(self):
        wide_a = stats.mean_interval([0.90, 0.91, 0.89, 0.92, 0.88])
        wide_b = stats.mean_interval([0.95, 0.96, 0.94, 0.97, 0.93])
        overlapping = stats.mean_interval([0.91, 0.93, 0.95, 0.90, 0.96])
        assert stats.tie_verdict(wide_a, wide_b) == stats.TIE_SEPARATED
        assert stats.tie_verdict(wide_a, overlapping) == stats.TIE_TIED
        assert stats.tie_verdict(wide_a, None) == stats.TIE_UNKNOWN

    def test_a_two_element_interval_is_read_without_a_sample_count(self):
        """`Interval.from_field` tolerates the conventional `[low, high]` so a
        hand-written fixture is readable -- and reads it as unusable, because
        an interval whose provenance the record does not state cannot support
        a tie. The conservative direction."""
        pair = stats.Interval.from_field([0.9, 0.95])
        assert (pair.low, pair.high, pair.n) == (0.9, 0.95, 0)
        assert pair.usable is False
        full = stats.Interval.from_field({"low": 0.9, "high": 0.95, "n": 500})
        assert full.usable is True
        assert stats.Interval.from_field(None) is None
        assert stats.Interval.from_field({"low": None, "high": 0.9}) is None

    def test_the_field_is_null_rather_than_nan_for_a_missed_measurement(self):
        """`json.dumps` writes a bare `NaN`, which strict JSON readers
        reject, and a number is never unambiguously "not measured"."""
        assert stats.mean_interval([]).as_field() is None
        field = stats.mean_interval([0.9, 0.95, 0.92]).as_field()
        assert set(field) == {"low", "high", "n"}
        # Round trips through JSON, which is where it actually goes.
        assert json.loads(json.dumps(field)) == field


class TestTheQpsIntervalInvertsAndSwaps:
    """§5.1 item 3: "Single-client QPS: computed as n / sum latency today, so
    bootstrap the mean latency and invert the bounds"."""

    LAT = [0.010, 0.011, 0.009, 0.0105, 0.0115, 0.0095, 0.012, 0.0085,
           0.010, 0.011]

    def test_the_bounds_are_the_latency_bounds_inverted_and_swapped(self):
        latency = stats.mean_interval(self.LAT)
        qps = stats.qps_interval_from_latencies(self.LAT)
        assert qps.low == pytest.approx(1.0 / latency.high)
        assert qps.high == pytest.approx(1.0 / latency.low)
        assert qps.low < qps.high
        assert qps.n == latency.n == len(self.LAT)

    def test_the_interval_brackets_the_qps_the_collector_would_report(self):
        """The point the bar is drawn around is `len(s) / sum(s)`, which is
        what `LatencyCollector.qps(clients=1)` computes."""
        point = len(self.LAT) / sum(self.LAT)
        qps = stats.qps_interval_from_latencies(self.LAT)
        assert qps.low <= point <= qps.high

    def test_forgetting_to_swap_inverts_the_interval(self):
        """The mutant, run. An un-swapped interval has low > high, and every
        overlap test in the report then reads it as empty.

        Honest: (91.32, 103.63) -- low below high, bracketing 97.09.
        Mutant (bounds inverted without swapping): (103.63, 91.32), which is
            `low > high`, so `a.low <= b.high and b.low <= a.high` is False
            against ANY interval and the tie column reports every engine as
            distinguishable.
        """
        qps = stats.qps_interval_from_latencies(self.LAT)
        latency = stats.mean_interval(self.LAT)
        mutant = stats.Interval(1.0 / latency.low, 1.0 / latency.high,
                                latency.n)
        assert round(qps.low, 2) == 91.32 and round(qps.high, 2) == 103.63
        assert round(mutant.low, 2) == 103.63 and round(mutant.high, 2) == 91.32
        assert qps.low < qps.high
        assert mutant.low > mutant.high
        other = stats.Interval(95.0, 99.0, 500)
        assert stats.tie_verdict(qps, other) == stats.TIE_TIED
        assert stats.tie_verdict(mutant, other) == stats.TIE_SEPARATED

    def test_a_zero_latency_bound_does_not_become_a_trillion_qps(self):
        """`LatencyCollector.qps` clamps with `max(..., 1e-12)` because it must
        produce one number; an error bar reaching 1e12 QPS is not a figure."""
        zeroes = stats.qps_interval_from_latencies([0.0, 0.0, 0.0])
        assert zeroes.low != zeroes.low or zeroes.high != zeroes.high


class TestTheConcurrencyProhibitionIsEnforced:
    """§5.1 item 5: "Concurrency keeps its repeat-based spread (PR 1.5); do
    not bootstrap within one window, because samples inside a window are not
    independent."

    A prohibition, not a feature -- and a comment lasts exactly as long as the
    next person who reads it, so `ci_fields` refuses.
    """

    def test_the_phase_name_matches_the_records_module(self):
        """Spelled in `stats` rather than imported, following
        `churn_readers.OPS_UNIT_KEYS`, so this pins the two declarations
        together the way `test_t4b_readers` pins that one."""
        assert stats.NON_INDEPENDENT_PHASES == frozenset(
            {records.PHASE_CONCURRENCY})

    def test_asking_for_an_interval_on_concurrency_raises(self):
        with pytest.raises(ValueError) as excinfo:
            stats.ci_fields(records.PHASE_CONCURRENCY,
                            latencies_s=[0.01, 0.02, 0.03])
        message = str(excinfo.value)
        assert "5.1 item 5" in message
        assert "not independent" in message
        # And it names the measurement that IS the answer for that phase, so
        # the refusal is actionable rather than just a wall.
        assert "qps_samples" in message

    @pytest.mark.parametrize("phase", [records.PHASE_CALIBRATION,
                                       records.PHASE_FILTERED,
                                       records.PHASE_CHURN])
    def test_the_three_phases_item_4_names_are_allowed(self, phase):
        """The other direction. A guard that refused everything would pass the
        test above and write no intervals at all."""
        out = stats.ci_fields(phase, recall_per_query=[0.9, 0.95, 1.0, 0.85],
                              latencies_s=[0.01, 0.011, 0.009, 0.012])
        assert set(out) == {stats.RECALL_CI_KEY, stats.QPS_CI_KEY}
        assert out[stats.RECALL_CI_KEY]["n"] == 4

    def test_no_concurrency_record_carries_an_interval(self):
        """The prohibition read off the records a real sweep writes, not off
        the function that would have written them."""
        recs = make_records("synthetic-run")
        concurrency = [r for r in recs if r.get("phase") == "concurrency"]
        assert concurrency, "no concurrency records in the synthetic run"
        for r in concurrency:
            extra = r.get("extra") or {}
            assert stats.RECALL_CI_KEY not in extra, r["clients"]
            assert stats.QPS_CI_KEY not in extra, r["clients"]
            # And it still has the measurement of uncertainty that IS right
            # for it, so this is not "concurrency has no error bar".
            assert extra["repeats"] == 3
            assert extra["qps_samples"]


# ---------------------------------------------------------------------------
# §5.1 item 4 -- the records the real harness writes
# ---------------------------------------------------------------------------

class TestTheHarnessWritesTheIntervals:
    """§5.1 item 4: "Write extra.recall_ci95 and extra.qps_ci95 on
    calibration, filtered and churn records."

    Driven through the real workloads against the fake driver, so this is the
    records as written rather than the helper in isolation.
    """

    def _assert_interval(self, record, where):
        extra = record.get("extra") or {}
        for key in (stats.RECALL_CI_KEY, stats.QPS_CI_KEY):
            assert key in extra, (where, key, sorted(extra))
            field = extra[key]
            assert set(field) == {"low", "high", "n"}, (where, key, field)
            assert field["low"] <= field["high"], (where, key, field)
            assert field["n"] >= 1, (where, key, field)
        # The published point sits inside its own recall interval.
        recall = record.get("recall_at_k")
        if recall is not None:
            iv = extra[stats.RECALL_CI_KEY]
            assert iv["low"] - 1e-6 <= recall <= iv["high"] + 1e-6, (
                where, recall, iv)

    def test_calibration_records_carry_both(self, ctx, dataset):
        driver = loaded(dataset, ctx.cache_dir)
        calw.run(ctx, driver, dataset, index_spec(dataset),
                 ef_grid=(16, 64), max_queries=N_QUERIES,
                 indexed_rows=N_ROWS)
        rows = [r for r in written(ctx)
                if r["phase"] == records.PHASE_CALIBRATION]
        assert rows, "the calibration sweep wrote nothing"
        for r in rows:
            self._assert_interval(r, f"calibration ef={r['ef_search']}")

    def test_filtered_records_carry_both(self, ctx, dataset):
        driver = loaded(dataset, ctx.cache_dir)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
               selectivities=(0.10,), max_queries=N_QUERIES,
               indexed_rows=N_ROWS)
        rows = [r for r in written(ctx) if r["phase"] == records.PHASE_FILTERED]
        assert rows, "the filtered workload wrote nothing"
        for r in rows:
            self._assert_interval(r, f"filtered {r['selectivity']}")

    def test_churn_records_carry_both_including_the_baseline(self, ctx,
                                                             dataset):
        driver = loaded(dataset, ctx.cache_dir, tombstone_penalty=1.0)
        cw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
               fractions=(0.10,), max_queries=N_QUERIES,
               indexed_rows=N_ROWS)
        rows = [r for r in written(ctx)
                if r["phase"] == records.PHASE_CHURN
                and r.get("clients") == 1]
        assert len(rows) >= 2, [r.get("churn_fraction") for r in rows]
        # The baseline specifically: it is the x=0 point every fraction is
        # measured against, so a chart with bars on the churned points and
        # none here would show the comparison as exact where it is not.
        baseline = [r for r in rows if r["churn_fraction"] == 0.0]
        assert len(baseline) == 1
        for r in rows:
            self._assert_interval(r, f"churn {r['churn_fraction']}")

    def test_the_interval_narrows_as_the_sample_grows(self):
        """Not a property of the plumbing but of the statistic, and the thing
        that says the interval is drawn from the data rather than from a
        constant. Four samples against four hundred of the same distribution.
        """
        rng = numpy.random.default_rng(4242)
        population = rng.normal(0.95, 0.03, 400)
        small = stats.mean_interval(population[:4])
        large = stats.mean_interval(population)
        assert (small.high - small.low) > 3 * (large.high - large.low), (
            small, large)


# ---------------------------------------------------------------------------
# §5.2 -- the match key, and its reconciliation with the ops unit
# ---------------------------------------------------------------------------

class TestTheMatchKeyReconcilesWithTheOpsUnit:
    """Carried finding 2: `OPS_UNIT_FIELDS` is the convention already in the
    code and §5.2 item 1's key must reconcile with it, not start a second one.
    """

    def test_the_shared_axes_are_exactly_the_three_expected(self):
        shared = set(generate.CROSSCHECK_KEY_FIELDS) & set(
            generate.OPS_UNIT_FIELDS)
        assert shared == {"m", "build_mode", "resource_pass"}

    def test_the_additions_are_exactly_ef_search_and_quantization(self):
        added = (set(generate.CROSSCHECK_KEY_FIELDS)
                 - set(generate.OPS_UNIT_FIELDS) - {"engine", "dataset"})
        assert added == {"ef_search", "quantization"}

    def test_storage_engine_is_the_only_subtraction(self):
        dropped = set(generate.OPS_UNIT_FIELDS) - set(
            generate.CROSSCHECK_KEY_FIELDS)
        assert dropped == {"storage_engine"}

    def test_the_ast_pin_on_the_ops_loops_still_holds(self):
        """The pin `tests/test_t3c_filtered.py` owns, re-read here because
        this task added a key beside `OPS_UNIT_FIELDS` and the easiest way to
        "reconcile" the two would have been to change that tuple.

        A substring pin provably will not do this job: a genuine fourth looped
        axis was once added to `_run_unit` with all three original loop
        strings intact and the whole suite stayed green. So this walks the AST
        the same way the pin does.
        """
        import ast
        with open(os.path.join(VB_ROOT, "orchestrator", "cli.py")) as fh:
            tree = ast.parse(fh.read())
        run_unit = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "_run_unit")
        looped = set()
        for node in ast.walk(run_unit):
            if isinstance(node, ast.For) and any(
                    isinstance(i, ast.Call)
                    and getattr(i.func, "attr", None) in ("harness_args",
                                                          "run_harness")
                    for i in ast.walk(node)):
                looped.add(ast.unparse(node.target))
        assert looped == {"m", "build_mode", "storage"}, looped
        # And `OPS_UNIT_FIELDS` is untouched by this task.
        assert generate.OPS_UNIT_FIELDS == ("m", "build_mode",
                                            "storage_engine", "resource_pass")
        # Neither addition is a loop in `_run_unit`, which is why adding them
        # to the cross-check key cannot satisfy or break that pin.
        assert "ef_search" not in looped and "quantization" not in looped

    def test_the_generate_facade_is_the_crosscheck_implementation(self):
        """`report.generate` re-exports §5.2's names; this pins the two.

        The split that moved §5.2 into `report/crosscheck.py` left a
        re-export behind, and a re-export is a copy: patching
        `generate.CROSSCHECK_MIN_TOLERANCE` rebinds the copy while
        `_append_crosscheck_row` keeps reading its own module's global. Five
        mutation tests in this file went red on exactly that, which is the
        module-level-container trap in miniature.

        So the mutations target `crosscheck` and this asserts the facade has
        not drifted: same object for the functions, same value for the
        constants. `is` for the callables, because a copy that happened to
        compare equal is precisely what would hide the drift.
        """
        for name in ("consistency_checks", "crosscheck_key",
                     "build_mode_axis_engines", "_normalised_storage",
                     "_crosscheck_qps_inversions", "_append_crosscheck_row",
                     "_crosscheck_quantization", "_crosscheck_sort_key"):
            assert getattr(generate, name) is getattr(crosscheck, name), name
        for name in ("CROSSCHECK_KEY_FIELDS", "CROSSCHECK_MIN_TOLERANCE",
                     "CROSSCHECK_RECALL_BAND", "CROSSCHECK_BASIS_SLACK",
                     "COVERAGE_MATCHED", "COVERAGE_NO_ANN", "COVERAGE_NO_OPS",
                     "COVERAGE_NO_M_OVERLAP", "COVERAGE_NOTHING_MATCHED"):
            assert getattr(generate, name) == getattr(crosscheck, name), name
        # And the mutation target really is the defining module, so the
        # comment beside MUTATION_TARGET is not decoration.
        assert MUTATION_TARGET is crosscheck
        assert generate.consistency_checks.__module__ == "report.crosscheck"

    def test_requiring_storage_equality_would_empty_the_table(self):
        """The measurement behind the subtraction, run.

        `harness/main.py` defaults `--storage-engine` to `InnoDB` for every
        engine, so pgvector's ops records say "InnoDB" while its ann module
        reports "heap". A key that held raw storage constant matches nothing
        for pgvector.

        Honest: pgvector matches, paired on the normalised storage "heap".
        Mutant (storage compared raw): pgvector's ops "InnoDB" never meets
            its ann "heap" and the engine disappears from the table.
        """
        recs, _ = synthetic_summary()
        honest = generate.consistency_checks(recs)
        pg_honest = [r for r in honest["matches"] if r["engine"] == "pgvector"]
        assert pg_honest, "pgvector did not match at all"
        assert {r["storage_engine"] for r in pg_honest} == {"heap"}

        original = MUTATION_TARGET._normalised_storage
        try:
            MUTATION_TARGET._normalised_storage = lambda r: r.get("storage_engine")
            mutant = generate.consistency_checks(recs)
        finally:
            MUTATION_TARGET._normalised_storage = original
        pg_mutant = [r for r in mutant["matches"] if r["engine"] == "pgvector"]
        assert pg_mutant == [], [r["storage_engine"] for r in pg_mutant]
        # The MySQL family is unaffected, which is what makes this a naming
        # problem and not a matching problem.
        assert len([r for r in mutant["matches"]
                    if r["engine"] == "mariadb"]) == len(
                        [r for r in honest["matches"]
                         if r["engine"] == "mariadb"])

    def test_the_build_mode_axis_is_dropped_only_where_ann_is_silent(self):
        """The second asymmetry, which §5.2 item 1 does not anticipate.

        `algorithms/{mariadb,mariadb123,alisql}/module.py` write no
        `build_mode` attribute, so their ann records read None while every ops
        record carries one. The axis is therefore dropped for them and kept
        for pgvector, whose module declares it.
        """
        recs, _ = synthetic_summary()
        out = generate.consistency_checks(recs)
        by_engine = {}
        for row in out["matches"]:
            by_engine.setdefault(row["engine"], set()).add(
                row["build_mode_axis_dropped"])
        assert by_engine["pgvector"] == {False}, by_engine
        assert by_engine["mariadb"] == {True}, by_engine
        assert by_engine["alisql"] == {True}, by_engine
        # Both branches are present, so neither is untested.
        assert {True, False} == set().union(*by_engine.values())

    def test_the_ann_modules_really_do_differ_on_it(self):
        """The claim above read off the overlay rather than taken on trust.

        Anchored on the key with its quote, so a mention of `build_mode` in a
        comment cannot satisfy it -- the substring shape this repo has shipped
        ten times.
        """
        base = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms")
        for engine in ("mariadb", "mariadb123", "alisql"):
            with open(os.path.join(base, engine, "module.py")) as fh:
                assert '"build_mode"' not in fh.read(), engine
        for engine in ("pgvector", "valkey"):
            with open(os.path.join(base, engine, "module.py")) as fh:
                assert '"build_mode": self._build_mode' in fh.read(), engine

    def test_dropping_the_axis_everywhere_would_pool_two_pgvector_indexes(self):
        """The conservative direction, proven by its mutant.

        `post` and `incremental` are two different indexes. If the axis were
        dropped for every engine rather than only where the recall path is
        silent, a pgvector `incremental` ops unit would match a `post` ann
        point.

        Honest: with both pgvector build modes present, `post` matches
            `post` only -- 2 rows, both `ops_build_mode == "post"`, each with
            `ops_candidates == 1`.
        Mutant (`build_mode_axis_engines` returning the empty set): the two
            modes key identically, so the row count doubles to 4 and each row
            reports `ops_candidates == 2` -- the ops side's `incremental`
            index paired against the ann side's `post` one.

        That second number is here because of what the FIRST version of this
        test found. The matcher used to take `ops_rows[0]` inside a key, so
        the mutant published 2 rows rather than 4 and silently discarded the
        other match -- a quieter wrong answer than a duplicated one, and the
        reason `consistency_checks` now pairs every match.
        """
        recs, _ = synthetic_summary()
        extra_mode = []
        for r in recs:
            if r["phase"] == "ef_calibration" and r["engine"] == "pgvector":
                clone = copy.deepcopy(r)
                clone["build_mode"] = "incremental"
                clone["recall_at_k"] = round(r["recall_at_k"] - 0.05, 6)
                extra_mode.append(clone)
        assert extra_mode, "no pgvector calibration records to clone"
        recs = recs + extra_mode

        honest = generate.consistency_checks(recs)
        pg_honest = [r for r in honest["matches"] if r["engine"] == "pgvector"]

        original = MUTATION_TARGET.build_mode_axis_engines
        try:
            MUTATION_TARGET.build_mode_axis_engines = lambda ann: frozenset()
            mutant = generate.consistency_checks(recs)
        finally:
            MUTATION_TARGET.build_mode_axis_engines = original
        pg_mutant = [r for r in mutant["matches"] if r["engine"] == "pgvector"]

        assert len(pg_honest) == 2, [(r["ef_search"], r["ops_build_mode"])
                                     for r in pg_honest]
        assert {r["ops_build_mode"] for r in pg_honest} == {"post"}
        assert {r["ops_candidates"] for r in pg_honest} == {1}, pg_honest
        assert len(pg_mutant) == 4, [(r["ef_search"], r["ops_build_mode"])
                                     for r in pg_mutant]
        assert {r["ops_build_mode"] for r in pg_mutant} == {"post",
                                                            "incremental"}
        # Every match listed, and each row saying how many candidates its key
        # held -- so the pooling is visible rather than silently resolved.
        assert {r["ops_candidates"] for r in pg_mutant} == {2}, pg_mutant


# ---------------------------------------------------------------------------
# §5.4's second prescribed test
# ---------------------------------------------------------------------------

class TestTheCrossCheckListsAConsistentAndAnInconsistentPair:
    """§5.4: "Cross-check matcher: synthetic run directory (extend
    tests/synthetic.py) with one consistent and one inconsistent pair; the
    report lists both correctly."

    Read out of the RENDERED table, cell by cell. `tests/synthetic.py` declares
    the two pairs in `CROSSCHECK_PAIRS` by offsetting the ops recall from the
    ann recall of the record it matches, so "consistent" is a property of that
    declaration rather than of two formulas happening to agree.
    """

    def _rendered(self):
        _recs, summary = synthetic_summary()
        return table_cells(render._consistency_section(summary))

    def test_the_fixture_declares_one_of_each(self):
        """The fixture is not vacuous: if both pairs were consistent the test
        below would pass while only half the matcher ran."""
        kinds = {pair["consistent"] for pair in CROSSCHECK_PAIRS}
        assert kinds == {True, False}, CROSSCHECK_PAIRS

    def test_both_pairs_appear_with_the_right_mark(self):
        header, *body = self._rendered()
        engine = header.index("Engine")
        ef = header.index("ef_search")
        ann = header.index("Recall (ann)")
        ops = header.index("Recall (ops)")
        gap = header.index("Recall gap")
        verdict = header.index("Within tolerance?")

        labels = {"mariadb": render._label("mariadb"),
                  "pgvector": render._label("pgvector")}
        for pair in CROSSCHECK_PAIRS:
            rows = [r for r in body
                    if r[engine] == labels[pair["engine"]]
                    and r[ef] == str(pair["ef_search"])]
            assert len(rows) == 1, (pair, [r[engine] for r in body])
            row = rows[0]
            # The two recalls are both shown, as item 3 asks ("listing every
            # match with both values").
            assert float(row[ops]) == pytest.approx(
                float(row[ann]) + pair["recall_offset"], abs=1e-4), (pair, row)
            assert float(row[gap]) == pytest.approx(
                abs(pair["recall_offset"]), abs=1e-4), (pair, row)
            if pair["consistent"]:
                assert render.CROSSCHECK_PASS_MARK in row[verdict], row
                assert "agrees" in row[verdict], row
                assert render.CROSSCHECK_FAIL_MARK not in row[verdict], row
            else:
                assert render.CROSSCHECK_FAIL_MARK in row[verdict], row
                assert "DISAGREES" in row[verdict], row
                assert render.CROSSCHECK_PASS_MARK not in row[verdict], row

    def test_exactly_one_row_fails(self):
        """A count, so a matcher that marked everything as failing would not
        satisfy the per-pair assertions above by accident."""
        _recs, summary = synthetic_summary()
        matches = summary["consistency"]["matches"]
        failed = [r for r in matches if not r["passed"]]
        assert len(matches) == 6, [(r["engine"], r["ef_search"])
                                   for r in matches]
        assert len(failed) == 1, [(r["engine"], r["ef_search"])
                                  for r in failed]
        assert (failed[0]["engine"], failed[0]["ef_search"]) == ("pgvector",
                                                                 160)

    def test_both_branches_of_the_tolerance_are_exercised(self):
        """§5.2 item 2 is `max(0.01, 2 x half)`, and a fixture where every row
        took the same branch would leave half of it undrawn."""
        _recs, summary = synthetic_summary()
        bases = {r["tolerance_basis"] for r in summary["consistency"]["matches"]}
        assert bases == {"floor", "ci"}, bases

    def test_the_tolerance_is_the_maximum_and_not_either_term(self):
        """The mutant for the `max`, run on a real row.

        Honest: the pgvector ef=160 row has a CI half-width of 0.02, so the
            tolerance is 0.0400 and its |gap| of 0.1200 fails.
        Mutant (`max` dropped, the CI term used alone): the mariadb ef=160
            row's half-width is 0.0035, so its tolerance becomes 0.0070 and
            its |gap| of 0.0040 ... still passes. The row that MOVES is the
            ef=40 one at half-width 0.005: tolerance 0.0100 -> 0.0100. So the
            floor only bites below 0.005, and the honest demonstration is a
            record built at half-width 0.001: tolerance 0.0100 honest,
            0.0020 mutant, and a gap of 0.004 passes honest and fails mutant.
        """
        recs, _ = synthetic_summary()
        tightened = []
        for r in recs:
            if (r["phase"] == "ef_calibration" and r["engine"] == "alisql"
                    and r["ef_search"] == 40):
                clone = copy.deepcopy(r)
                extra = clone["extra"]
                centre = clone["recall_at_k"]
                extra["recall_ci95"] = {"low": round(centre - 0.001, 6),
                                        "high": round(centre + 0.001, 6),
                                        "n": 1000}
                tightened.append(clone)
            elif not (r["phase"] == "ef_calibration"
                      and r["engine"] == "alisql" and r["ef_search"] == 40):
                tightened.append(r)
        assert len(tightened) == len(recs)

        # Push the gap to 0.004: inside the 0.01 floor, outside 2 x 0.001.
        for r in tightened:
            if (r["phase"] == "ef_calibration" and r["engine"] == "alisql"
                    and r["ef_search"] == 40):
                ann = next(a for a in recs
                           if a["phase"] == "recall_qps"
                           and a["engine"] == "alisql" and a["m"] == r["m"]
                           and a["ef_search"] == 40)
                r["recall_at_k"] = round(ann["recall_at_k"] - 0.004, 6)
                centre = r["recall_at_k"]
                r["extra"]["recall_ci95"] = {"low": round(centre - 0.001, 6),
                                             "high": round(centre + 0.001, 6),
                                             "n": 1000}

        def row_for(result):
            return next(x for x in result["matches"]
                        if x["engine"] == "alisql" and x["ef_search"] == 40)

        honest = row_for(generate.consistency_checks(tightened))

        original = MUTATION_TARGET.CROSSCHECK_MIN_TOLERANCE
        try:
            # The mutant: the floor removed, so the CI term stands alone.
            MUTATION_TARGET.CROSSCHECK_MIN_TOLERANCE = 0.0
            mutant = row_for(generate.consistency_checks(tightened))
        finally:
            MUTATION_TARGET.CROSSCHECK_MIN_TOLERANCE = original

        assert round(honest["delta"], 4) == 0.004
        assert round(honest["tolerance"], 4) == 0.0100
        assert honest["passed"] is True
        assert round(mutant["delta"], 4) == 0.004
        assert round(mutant["tolerance"], 4) == 0.0020
        assert mutant["passed"] is False
        # Restored, and the real row is unchanged.
        assert generate.CROSSCHECK_MIN_TOLERANCE == 0.01

    def test_a_missing_interval_gets_the_floor_and_says_so(self):
        """A record with no usable interval must not widen the tolerance.
        `max(0.01, ...)` gives it exactly the floor, which is the stricter of
        the two branches."""
        recs, _ = synthetic_summary()
        stripped = []
        for r in recs:
            if (r["phase"] == "ef_calibration" and r["engine"] == "mariadb"):
                clone = copy.deepcopy(r)
                clone["extra"].pop("recall_ci95", None)
                stripped.append(clone)
            else:
                stripped.append(r)
        out = generate.consistency_checks(stripped)
        rows = [r for r in out["matches"] if r["engine"] == "mariadb"]
        assert rows
        for row in rows:
            assert row["ci_half_width"] is None, row
            assert row["tolerance"] == generate.CROSSCHECK_MIN_TOLERANCE, row
            assert row["tolerance_basis"] == "floor", row


class TestTheForbiddenVerdictIsAbsentEvenUnwrapped:
    """`tests/test_config.py:TestFilterStrategyIsMeasuredNotQuoted` asserts a
    sentence the report once published and never measured is absent from
    `report/render.py`. It searches the file as written.

    That leaves a hole this task fell into: a comment quoting the sentence
    across a line wrap passes the guard, because the `#` and the newline sit
    in the middle of it -- while the sentence is still there to anyone who
    reflows the paragraph, and the suite then breaks with a failure that
    names a file nobody edited. Measured on my own first fix: the phrase was
    absent verbatim and present after unwrapping, and a second comment ten
    lines below asserted it was absent in both.

    This closes the hole for the files this task owns, in the form the
    existing guard cannot: strip the comment prefixes, collapse the
    whitespace, then search.
    """

    PHRASE = "chooses its filtering strategy per query"

    #: `report/render.py`, which `test_config.py` already guards as written,
    #: plus every file this task puts prose into.
    #:
    #: **`report/generate.py` is deliberately NOT here**, and the reason is
    #: not that it would be inconvenient. Its `filter_strategies` comment has
    #: quoted that verdict since before this task -- `git show
    #: 6ee0b40:report/generate.py:497` -- as the historical record of what the
    #: report got wrong, which is why the existing guard names `render.py`
    #: alone: `render.py` is where the asymmetries section would say such a
    #: thing TO A READER, and `generate.py` is where the incident is written
    #: down so the next derivation does not repeat it. Adding it here would
    #: delete that record to satisfy a test, which is the wrong direction.
    FILES = ("report/render.py", "report/crosscheck.py", "report/charts.py",
             "harness/metrics/stats.py", "harness/datasets.py",
             "docs/05-methodology.md")

    def test_the_one_legitimate_occurrence_is_still_there(self):
        """The exclusion above, made executable rather than asserted.

        If `generate.py`'s record of the incident is ever deleted, the
        paragraph justifying the exclusion becomes false and this fails --
        so the list cannot quietly become "the files that happen to pass".
        """
        with open(os.path.join(VB_ROOT, "report", "generate.py")) as fh:
            text = fh.read()
        # The UNWRAPPED form: that record is itself split across a line wrap
        # (`report/generate.py:804-805`, in the comment `summarize` carries
        # above its `by_engine` filter-strategy grouping: `...Valkey "chooses
        # its filtering strategy` / `per query" on the strength of its
        # documentation.`), which is a neat demonstration of why the verbatim
        # search is the weaker one.
        #
        # **RE-DERIVED, AND IT WAS NOT A HISTORICAL QUOTATION.** This read
        # `c8de6a5:report/generate.py:607-608` in the present tense, and at
        # that commit the two lines were a `server_memory_bytes` lookup and a
        # `for name, rows` loop -- a memory ceiling, nothing to do with
        # filter strategies. The
        # citation is the EVIDENCE for the sentence around it: the argument is
        # that the verbatim search is the weaker one *because* the record
        # wraps, and these lines are offered as the demonstration. A reader
        # following the old number could not confirm the premise of the
        # `FILES` exclusion it justifies, which makes this worse than a stale
        # pointer. It was wrong before this branch existed and the branch does
        # not touch this file, so `tests/test_e16_citation_ratchet.py`
        # grandfathers it; the repair is a re-derivation with the symbol
        # named, not a revision prefix, because the claim is about the tree as
        # it is now.
        assert self.PHRASE in self._unwrapped(text), (
            "report/generate.py no longer records the filter-strategy "
            "verdict it published wrongly; if that was deliberate, remove "
            "the exclusion in FILES above and add the file to it.")
        # It is in the derivation's comment, not in anything rendered -- the
        # distinction the exclusion rests on.
        assert "filter_strategies" in text
        assert self.PHRASE not in text

    @staticmethod
    def _unwrapped(text: str) -> str:
        """The file with both kinds of line break rejoined.

        Two substitutions, and the second was missing in the first version of
        this guard -- which left the hole that matters most, because it is the
        one that reaches A READER rather than a maintainer:

        * `\\s*#\\s*` -> a comment wrap, the shape round 1 shipped.
        * `['"]\\s*[A-Za-z]{0,2}['"]` -> an ADJACENT STRING LITERAL JOIN,
          with the optional literal prefix (`f`, `r`, `rb`) that sits between
          the two quotes when the continuation is an f-string. Measured: a
          regex without the prefix misses the `" f"` join, and that is how
          `render.py` writes most of its rendered prose. This is how
          `report/render.py` writes every line of its rendered prose. A
          sentence split as `"... chooses its "` / `"filtering strategy ..."`
          is absent verbatim AND absent after the comment-strip, because the
          two quote characters sit in the middle of it. Measured on the five
          shapes in `test_every_way_a_sentence_can_hide_is_covered`: the
          comment forms were caught, both string forms evaded.

        Joining is the safe direction for a NEGATIVE assertion. Collapsing
        more than intended -- `["a", "b"]` becoming `["ab"]` -- can only
        create candidate matches, never conceal one, so the failure mode is a
        false alarm on a specific 40-character phrase rather than a miss.

        Still evading both forms, and left alone deliberately: a phrase split
        across a `"".join([...])` list, `+`-concatenation, or `%`-formatting.
        None is a natural way to write prose in this repository -- `render.py`
        uses adjacent literals throughout -- so they are recorded as the
        remaining surface rather than chased.
        """
        import re
        joined = re.sub(r"['\"]\s*[A-Za-z]{0,2}['\"]", "", text)
        return re.sub(r"\s+", " ", re.sub(r"\s*#\s*", " ", joined))

    @pytest.mark.parametrize("relative", FILES)
    def test_the_sentence_is_absent_in_both_forms(self, relative):
        with open(os.path.join(VB_ROOT, relative)) as fh:
            text = fh.read()
        assert self.PHRASE not in text, relative
        assert self.PHRASE not in self._unwrapped(text), (
            f"{relative} carries the verdict across a line wrap: absent "
            f"verbatim, present once the comment prefixes are stripped. "
            f"tests/test_config.py's guard reads the file as written and "
            f"cannot see it.")

    #: Every way that sentence can appear in a Python file, as
    #: `(name, source text, caught verbatim?)`. The last two are the ones
    #: round 1's guard missed, and they are the forms that would reach a
    #: reader: `report/render.py` builds its rendered prose out of adjacent
    #: string literals, so a sentence IN THE REPORT is split exactly that way.
    HIDING_PLACES = (
        ("one-line comment",
         "        # ... the report asserting that Valkey chooses its "
         "filtering strategy per query.\n", True),
        ("comment split across a wrap",
         "        # ... the report asserting that Valkey chooses its\n"
         "        # filtering strategy per query.\n", False),
        ("rendered one-line string",
         '        parts.append("Valkey chooses its filtering strategy per '
         'query")\n', True),
        ("rendered string split across literals",
         '        parts.append("Valkey chooses its "\n'
         '                     "filtering strategy per query")\n', False),
        ("f-string split across literals",
         '        note = (f"{label} chooses its "\n'
         '                f"filtering strategy per query")\n', False),
    )

    @pytest.mark.parametrize("name,source,verbatim",
                             HIDING_PLACES,
                             ids=[h[0] for h in HIDING_PLACES])
    def test_every_way_a_sentence_can_hide_is_covered(self, name, source,
                                                      verbatim):
        """The guard's own mutants, run rather than described.

        `verbatim` records which forms a plain substring search finds, so the
        parametrisation documents the hole as well as closing it: three of
        these five evade the verbatim check, and all five must be caught
        after unwrapping. A guard a reflow defeats is not a guard.
        """
        assert (self.PHRASE in source) is verbatim, name
        assert self.PHRASE in self._unwrapped(source), (
            f"{name} evades the unwrapped search too, so this guard would "
            f"pass a file that still carries the sentence.")

    def test_the_unwrapping_is_load_bearing_and_not_a_no_op(self):
        """The other direction: at least one shape must need the unwrapping,
        or the whole `_unwrapped` machinery could be deleted and every test
        above would still pass on the verbatim check alone."""
        needed = [name for name, source, verbatim in self.HIDING_PLACES
                  if not verbatim]
        assert len(needed) == 3, needed
        assert "comment split across a wrap" in needed
        assert "rendered string split across literals" in needed


class TestTheSectionDoesNotClaimAxesItDidNotMatchOn:
    """Two axes in §5.2 item 1's key do less than the sentence listing them
    implies, and the prose has to say which. R7: a claim in prose is part of
    the deliverable, and five defects here have been "code right, prose
    wrong".
    """

    def test_the_quantization_caveat_appears_when_nothing_declared_one(self):
        """Today every row on both sides reads no quantization:
        `context.record_defaults` does not put `IndexSpec.quantization` on an
        ops record and the ann loader does not copy the attribute the mongodb
        module writes. So the key carries the axis without discriminating on
        it, and the note says so rather than letting the match list read as
        though quantization had been compared."""
        _recs, summary = synthetic_summary()
        assert {r["quantization"] for r in
                summary["consistency"]["matches"]} == {None}
        text = render._consistency_section(summary)
        assert "told nothing apart here" in text
        assert "every row on both sides reads no quantization" in text

    def test_it_switches_to_the_other_branch_when_one_is_declared(self):
        """The mutant for the branch above, run.

        Honest (no quantization anywhere): the caveat.
        Mutant (`scalar` stamped on both sides): the caveat is replaced by a
            note naming the value, so the two branches are distinguishable and
            neither is dead prose.
        """
        recs, _ = synthetic_summary()
        stamped = []
        for r in recs:
            clone = copy.deepcopy(r)
            if clone["phase"] in ("recall_qps", "ef_calibration"):
                clone.setdefault("extra", {})["quantization"] = "scalar"
            stamped.append(clone)
        summary = generate.summarize(stamped, make_manifest("x"))
        assert {r["quantization"] for r in
                summary["consistency"]["matches"]} == {"scalar"}
        text = render._consistency_section(summary)
        assert "Quantization was matched on" in text
        assert "`scalar`" in text
        assert "told nothing apart here" not in text

    def test_an_unmatched_build_mode_is_marked_and_the_mark_explained(self):
        """The MySQL family's rows matched WITHOUT the build-mode axis, so the
        cell showing `post` must not read as a matched axis."""
        _recs, summary = synthetic_summary()
        text = render._consistency_section(summary)
        rows = table_cells(text)
        header, body = rows[0], rows[1:]
        engine = header.index("Engine")
        mode = header.index("Build mode")
        marked = {r[engine] for r in body
                  if render.CROSSCHECK_UNMATCHED_AXIS_MARK in r[mode]}
        unmarked = {r[engine] for r in body
                    if render.CROSSCHECK_UNMATCHED_AXIS_MARK not in r[mode]}
        assert marked == {render._label("mariadb"), render._label("alisql")}, \
            marked
        assert unmarked == {render._label("pgvector")}, unmarked
        # Every marked row still shows the value the record carries, so the
        # mark qualifies the cell rather than hiding it.
        for row in body:
            assert "post" in row[mode], row
        # And the mark is explained where a reader will meet it.
        assert render.CROSSCHECK_UNMATCHED_AXIS_MARK in text
        assert "was **not** matched on for that row" in text

    def test_the_mark_is_absent_when_every_engine_declares_a_build_mode(self):
        """The other direction: a run of pgvector alone has the axis for every
        row, so neither the mark nor its footnote should appear."""
        recs, _ = synthetic_summary()
        only_pg = [r for r in recs if r.get("engine") == "pgvector"]
        summary = generate.summarize(only_pg, make_manifest("x"))
        matches = summary["consistency"]["matches"]
        assert matches
        assert not any(r["build_mode_axis_dropped"] for r in matches)
        text = render._consistency_section(summary)
        assert "was **not** matched on for that row" not in text


class TestAnEmptyMatchListSaysWhichKindOfEmptyItIs:
    """Carried finding 4. §5.2 item 3 says an empty list is reported rather
    than hidden, and an empty list has two causes that must not print the
    same: `main` configures `ann.m_values: [8, 32]` against
    `ops.m_values: [16]`, so nothing CAN match and nothing is wrong, while a
    profile whose grids do overlap and still matches nothing has a fault.
    """

    def _only(self, recs, phase_filter):
        return [r for r in recs if phase_filter(r)]

    def test_no_m_overlap_is_reported_as_a_configuration(self):
        """main's own shape: the ann grid and the ops M never meet."""
        recs, _ = synthetic_summary()
        shifted = []
        for r in recs:
            clone = copy.deepcopy(r)
            if clone["phase"] == "ef_calibration":
                clone["m"] = 24          # a value the ann sweep never ran
            shifted.append(clone)
        out = generate.consistency_checks(shifted)
        assert out["matches"] == []
        statuses = {c["status"] for c in out["coverage"]}
        assert statuses == {generate.COVERAGE_NO_M_OVERLAP}, statuses
        for c in out["coverage"]:
            assert c["overlap_m"] == []
            assert 24 in c["ops_m"]
            assert 24 not in c["ann_m"]

    def test_an_overlap_that_matches_nothing_is_reported_as_a_fault(self):
        """The other cause, and the one a reader must not mistake for the
        configuration above. Produced by breaking an axis the key matches on
        while leaving M overlapping."""
        recs, _ = synthetic_summary()
        broken = []
        for r in recs:
            clone = copy.deepcopy(r)
            if clone["phase"] == "ef_calibration":
                clone["resource_pass"] = "tuned"   # the ann side is normalized
            broken.append(clone)
        out = generate.consistency_checks(broken)
        assert out["matches"] == []
        statuses = {c["status"] for c in out["coverage"]}
        assert statuses == {generate.COVERAGE_NOTHING_MATCHED}, statuses
        for c in out["coverage"]:
            assert c["overlap_m"] == [16], c

    def test_the_two_statuses_are_different_strings(self):
        """Obvious, and the whole point: a single "no matches" message is what
        conflates a choice with a bug."""
        assert (generate.COVERAGE_NO_M_OVERLAP
                != generate.COVERAGE_NOTHING_MATCHED)
        assert "configuration" in generate.COVERAGE_NO_M_OVERLAP

    @pytest.mark.parametrize("status,needle", [
        (generate.COVERAGE_NO_M_OVERLAP, "a profile choice and not a fault"),
        (generate.COVERAGE_NOTHING_MATCHED, "**fault**"),
    ])
    def test_the_rendered_section_explains_each_one(self, status, needle):
        """The prose a reader actually sees, not the constant."""
        summary = {"consistency": {
            "matches": [], "qps_inversions": [],
            "coverage": [{"engine": "mariadb", "dataset": "d",
                          "ann_m": [8, 32], "ops_m": [16], "overlap_m": [],
                          "status": status, "matches": 0}]}}
        text = render._consistency_section(summary)
        assert status in text
        assert needle in text
        # The empty list is reported, not hidden.
        assert "No point was measured by both paths" in text

    def test_a_run_with_no_calibration_at_all_says_that_instead(self):
        recs, _ = synthetic_summary()
        without = [r for r in recs if r["phase"] != "ef_calibration"]
        out = generate.consistency_checks(without)
        assert out["matches"] == []
        assert {c["status"] for c in out["coverage"]} == {
            generate.COVERAGE_NO_OPS}

    def test_the_summary_always_carries_the_key(self):
        """A key that is absent when empty makes the renderer's "no matches"
        branch indistinguishable from an older summary that never had it."""
        _recs, summary = synthetic_summary()
        assert "consistency" in summary
        empty = generate.summarize([], {})
        assert "consistency" in empty
        assert empty["consistency"]["matches"] == []
        assert empty["consistency"]["coverage"] == []


class TestMainsGridsActuallyMeet:
    """§5.2 item 4 and PR 5's acceptance criterion, as a property of the
    profile rather than of the fixture.

    The check matches an ops calibration point to an ann recall point at the
    same `M`. `main` swept `ann.m_values: [8, 32]` against
    `ops.m_values: [16]`, so nothing could ever match and the report's
    "Consistency checks" section was empty by construction on the one profile
    the human actually runs -- which the coverage table correctly reported as
    a configuration rather than a fault, and which is still not a check.

    The human chose §5.2 item 4's first option over the second: M=16 added to
    the ann grid for both datasets. The alternative, one dataset only, needs a
    per-dataset ann axis that `ann_pass.render_config` cannot express -- it
    takes no `dataset` argument at all -- and would have saved about half the
    ingest.
    """

    def _profile(self):
        from orchestrator.config import load_profile
        return load_profile("main")

    def test_the_ann_grid_contains_every_ops_m(self):
        profile = self._profile()
        ann = set(profile["ann"]["m_values"])
        ops = set(profile["ops"]["m_values"])
        assert ops, "main declares no ops M"
        assert ops <= ann, (
            f"main's ops M {sorted(ops)} is not inside its ann grid "
            f"{sorted(ann)}, so PR 5 §5.2's cross-check is empty by "
            f"configuration on the profile this repository recommends. "
            f"Either add the ops M values to `ann.m_values` or run "
            f"calibration at the ann M values in a cross-check profile.")
        # And the ann grid still spans the frontier it is there for: M=16 is
        # an addition, not a replacement.
        assert {8, 32} <= ann, sorted(ann)

    def test_a_main_shaped_run_reports_matches_not_a_configuration_gap(self):
        """The grids meeting is necessary; this is the sufficient half.

        Records shaped as `main` writes them -- the ann sweep at every M in
        the grid, calibration at the ops M -- must come out `matched`, not
        `no M is measured by both paths`.
        """
        profile = self._profile()
        ann_m = profile["ann"]["m_values"]
        ops_m = profile["ops"]["m_values"]
        dataset = profile["datasets"][0]
        records = []
        for m in ann_m:
            records.append({
                "phase": "recall_qps", "engine": "mariadb",
                "dataset": dataset, "resource_pass": "normalized", "m": m,
                "ef_search": 160, "recall_at_k": 0.97, "qps": 900.0,
                "storage_engine": "InnoDB",
            })
        for m in ops_m:
            records.append({
                "phase": "ef_calibration", "engine": "mariadb",
                "dataset": dataset, "resource_pass": "normalized", "m": m,
                "ef_search": 160, "recall_at_k": 0.968, "qps": 700.0,
                "storage_engine": "InnoDB",
                # A version-9 calibration record says which recall rule it
                # carries; without the key it is read as a pre-v9 ID overlap
                # and the cross-check declines to compare it.
                "extra": {"recall_ci95": {"low": 0.962, "high": 0.974,
                                          "n": 1000},
                          "recall_kind": "distance_threshold"},
            })
        out = generate.consistency_checks(records)
        assert len(out["matches"]) == 1, out["matches"]
        row = out["matches"][0]
        assert row["m"] == ops_m[0]
        assert row["passed"] is True
        assert [c["status"] for c in out["coverage"]] == [
            generate.COVERAGE_MATCHED]

        # The mutant is the configuration this change replaced: drop M=16
        # from the ann side and the same records report a configuration gap
        # with no matches, which is where `main` was before D1.
        without = [r for r in records
                   if not (r["phase"] == "recall_qps" and r["m"] in ops_m)]
        gap = generate.consistency_checks(without)
        assert gap["matches"] == []
        assert [c["status"] for c in gap["coverage"]] == [
            generate.COVERAGE_NO_M_OVERLAP]

    def test_the_cost_of_the_choice_is_the_figure_the_docs_quote(self):
        """R8: the ingest figure moved, and the three places that quote it
        have to agree with the estimator. `tests/test_estimate.py` pins the
        estimator's own numbers and docs/07; this pins the one claim that is
        about the DECISION rather than about a profile -- that adding M=16 to
        both datasets costs about twice what one dataset would have.
        """
        import copy

        from orchestrator.estimate import estimate_load_hours
        profile = self._profile()
        three = ["mariadb", "alisql", "pgvector"]

        def ingest(m_values, datasets):
            p = copy.deepcopy(profile)
            p["ann"]["m_values"] = list(m_values)
            return estimate_load_hours(p, three, list(datasets),
                                       ["normalized"], ["ann", "ops"]
                                       )["ingest_hours"]

        both = profile["datasets"]
        baseline = ingest([8, 32], both)
        chosen = ingest([8, 16, 32], both)
        # One dataset only: the alternative, costed the way the estimator
        # would have to cost it -- three M on sift, two on glove.
        sift = [d for d in both if d.startswith("sift")]
        glove = [d for d in both if d.startswith("glove")]
        assert sift and glove, both
        one_dataset = ingest([8, 16, 32], sift) + ingest([8, 32], glove)

        assert chosen == pytest.approx(68.11, abs=0.02)
        assert baseline == pytest.approx(51.07, abs=0.02)
        assert chosen - baseline == pytest.approx(17.04, abs=0.05)
        assert one_dataset - baseline == pytest.approx(8.12, abs=0.05)
        # The chosen option costs roughly twice the alternative, which is the
        # trade the human made for keeping the change inside one file.
        assert 1.9 < (chosen - baseline) / (one_dataset - baseline) < 2.2


class TestTheQpsOrderInversionIsFlaggedNotFailed:
    """§5.2 item 5: "at matched recall, the engine order from ann (in-process)
    and ops (TCP, 1 client) should agree. Flag inversions, do not fail on
    them: TCP overhead is a legitimate difference."
    """

    def test_the_fixture_contains_one_and_it_is_reported(self):
        _recs, summary = synthetic_summary()
        inversions = summary["consistency"]["qps_inversions"]
        assert len(inversions) == 1, inversions
        row = inversions[0]
        assert sorted(row["engines"]) == ["mariadb", "pgvector"]
        assert row["faster_in_ann"] == "pgvector"
        assert row["faster_in_ops"] == "mariadb"
        # The orders really are opposite, computed from the row's own numbers
        # rather than restated.
        ann_gap = row["qps_ann"][0] - row["qps_ann"][1]
        ops_gap = row["qps_ops"][0] - row["qps_ops"][1]
        assert (ann_gap > 0) != (ops_gap > 0), row

    def test_an_inversion_does_not_make_the_match_fail(self):
        """"Do not fail on them". The inverted pair's own cross-check rows
        still pass, so the inversion is reported beside them rather than
        turning them into disagreements."""
        _recs, summary = synthetic_summary()
        inversion = summary["consistency"]["qps_inversions"][0]
        rows = [r for r in summary["consistency"]["matches"]
                if r["ef_search"] == inversion["ef_search"]
                and r["engine"] in inversion["engines"]]
        assert len(rows) == 2
        assert all(r["passed"] for r in rows), rows

    def test_a_pair_at_mismatched_recall_is_not_flagged(self):
        """"At matched recall", enforced. The pgvector ef=160 row is the
        inconsistent pair -- its ops recall is 0.87 against mariadb's 0.996 --
        and its throughput order also differs between the paths. Flagging it
        would report an ordering between two engines doing different amounts
        of work.

        Honest: one inversion, at ef=40, where all three recalls agree to
            0.008.
        Mutant (the recall band widened to 1.0, i.e. no band): two
            inversions, the second at ef=160 where the two ops recalls are
            0.1245 apart.
        """
        recs, _ = synthetic_summary()
        honest = generate.consistency_checks(recs)["qps_inversions"]

        original = MUTATION_TARGET.CROSSCHECK_RECALL_BAND
        try:
            MUTATION_TARGET.CROSSCHECK_RECALL_BAND = 1.0
            mutant = generate.consistency_checks(recs)["qps_inversions"]
        finally:
            MUTATION_TARGET.CROSSCHECK_RECALL_BAND = original

        assert [r["ef_search"] for r in honest] == [40]
        assert sorted(r["ef_search"] for r in mutant) == [40, 160]
        extra = next(r for r in mutant if r["ef_search"] == 160)
        assert abs(extra["recall_ops"][0] - extra["recall_ops"][1]) > 0.1
        assert generate.CROSSCHECK_RECALL_BAND == 0.02

    def test_two_build_modes_of_one_engine_are_not_an_engine_inversion(self):
        """"The engine order". pgvector `post` against pgvector `incremental`
        is one engine's two indexes."""
        rows = [
            {"dataset": "d", "resource_pass": "normalized", "m": 16,
             "ef_search": 40, "quantization": None, "engine": "pgvector",
             "storage_engine": "heap", "ops_build_mode": "post",
             "recall_ann": 0.95, "recall_ops": 0.95,
             "qps_ann": 100.0, "qps_ops": 50.0},
            {"dataset": "d", "resource_pass": "normalized", "m": 16,
             "ef_search": 40, "quantization": None, "engine": "pgvector",
             "storage_engine": "heap", "ops_build_mode": "incremental",
             "recall_ann": 0.95, "recall_ops": 0.95,
             "qps_ann": 80.0, "qps_ops": 70.0},
        ]
        assert generate._crosscheck_qps_inversions(rows) == []
        # Rename one and the same numbers DO produce an inversion, which is
        # what says the exclusion is the engine identity and not the data.
        rows[1] = dict(rows[1], engine="mariadb")
        assert len(generate._crosscheck_qps_inversions(rows)) == 1

    def test_a_tie_on_either_side_is_not_an_inversion(self):
        rows = [
            {"dataset": "d", "resource_pass": "normalized", "m": 16,
             "ef_search": 40, "quantization": None, "engine": "a",
             "storage_engine": "x", "ops_build_mode": "post",
             "recall_ann": 0.95, "recall_ops": 0.95,
             "qps_ann": 100.0, "qps_ops": 50.0},
            {"dataset": "d", "resource_pass": "normalized", "m": 16,
             "ef_search": 40, "quantization": None, "engine": "b",
             "storage_engine": "y", "ops_build_mode": "post",
             "recall_ann": 0.95, "recall_ops": 0.95,
             "qps_ann": 100.0, "qps_ops": 70.0},
        ]
        assert generate._crosscheck_qps_inversions(rows) == []

    def test_the_rendered_section_says_it_is_not_a_failure(self):
        _recs, summary = synthetic_summary()
        text = render._consistency_section(summary)
        assert "Throughput order that differs between the two paths" in text
        assert "**Not a failure.**" in text


class TestTheReportContainerCanImportTheStatsModule:
    """Carried finding 3, verified by EXECUTION rather than by reasoning from
    a sibling module.

    `harness/` and `report/` mount separately and `CLAUDE.md` records an
    `orchestrator` import from `report/` crashing generation at the final step
    of a twenty-hour run. `loaders.py` and `measurement_versions.py` import
    `harness.metrics.records` function-locally in eight places for this
    reason; `generate.consistency_checks`, `render._tie_cells` and
    `charts.recall_whiskers` follow them for `harness.metrics.stats`.
    """

    def test_report_modules_do_not_import_harness_at_module_scope(self):
        """An `import harness...` at the top of a `report/` module is the
        failure mode, so this reads the AST rather than the text: a
        function-local import is an `ast.Import` too, and only its POSITION
        distinguishes the safe form from the fatal one.
        """
        import ast
        for name in ("generate", "render", "charts", "crosscheck",
                     "loaders", "measurement_versions", "churn_state"):
            path = os.path.join(VB_ROOT, "report", f"{name}.py")
            with open(path) as fh:
                tree = ast.parse(fh.read())
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("harness"), (
                        name, node.module, node.lineno)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("harness"), (
                            name, alias.name, node.lineno)

    def test_generation_runs_with_harness_removed_from_sys_modules(self):
        """Executed, in a subprocess, with `harness` made unimportable.

        This is the only form that proves anything: a test in this process has
        `harness` imported already, so a top-level import in `report/` would
        be satisfied from `sys.modules` and the container's failure would not
        reproduce. The subprocess blocks the package with a meta-path finder,
        then imports and RUNS the report pipeline -- and must fail, because
        the pipeline genuinely needs those function-local imports to succeed.

        So the test is two subprocesses: one where `harness` is blocked and
        the import of `report.generate` must still succeed (the container's
        module-load step), and one where it is importable and generation
        produces the consistency section (the step that needs it).
        """
        import subprocess
        blocked = r'''
import sys
class Block:
    def find_module(self, name, path=None):
        return self if name == "harness" or name.startswith("harness.") else None
    def find_spec(self, name, path=None, target=None):
        if name == "harness" or name.startswith("harness."):
            raise ImportError("harness is not mounted in this container")
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, %r)
import report.generate, report.render, report.charts   # must not need harness
print("IMPORTED-WITHOUT-HARNESS")
try:
    import harness.metrics.stats
except ImportError as exc:
    print("BLOCK-WORKS:", exc)
else:
    print("BLOCK-FAILED")
''' % (VB_ROOT,)
        out = subprocess.run([sys.executable, "-c", blocked],
                             capture_output=True, text=True, cwd=VB_ROOT)
        assert "IMPORTED-WITHOUT-HARNESS" in out.stdout, out
        # The block really was in force, so the line above is not vacuous.
        assert "BLOCK-WORKS:" in out.stdout, out
        assert "BLOCK-FAILED" not in out.stdout, out

        working = r'''
import sys
sys.path.insert(0, %r)
from report.generate import summarize
from report.render import render_markdown
from tests.synthetic import make_manifest, make_records
recs = make_records("r"); man = make_manifest("r")
md = render_markdown(man, summarize(recs, man), recs, {}, "t")
rows = md.count("agrees") + md.count("DISAGREES")
print("CONSISTENCY-ROWS:", rows)
''' % (VB_ROOT,)
        out = subprocess.run([sys.executable, "-c", working],
                             capture_output=True, text=True, cwd=VB_ROOT)
        assert out.returncode == 0, out.stderr
        # Six matched rows, which is the number the fixture produces -- so the
        # function-local imports ran rather than merely being present.
        assert "CONSISTENCY-ROWS: 6" in out.stdout, out.stdout


# ---------------------------------------------------------------------------
# §5.1 item 6 in the tables, and the whiskers on the charts
# ---------------------------------------------------------------------------

class TestTheTablesMarkTiesInsteadOfRanking:

    def _cells(self, table: str):
        rows = table_cells(table)
        return rows[0], rows[1:]

    def test_the_churn_table_has_a_tie_column_with_all_three_answers(self):
        _recs, summary = synthetic_summary()
        header, body = self._cells(render._churn_table(summary))
        column = header.index("Tied with")
        cells = [r[column] for r in body]
        tied = [c for c in cells if c.startswith("tied with ")]
        assert tied, cells
        assert render.TIE_SEPARATED_CELL in cells, cells
        assert render.TIE_ALONE_CELL in cells, cells

    def test_the_filtered_table_has_one_too(self):
        _recs, summary = synthetic_summary()
        header, body = self._cells(render._filtered_table(summary))
        column = header.index("Tied with")
        cells = [r[column] for r in body]
        assert [c for c in cells if c.startswith("tied with ")], cells

    def test_an_archived_record_reads_no_interval_rather_than_distinct(self):
        """The case every record written before PR 5 is in. "distinct" there
        would be an ordering claim with no measurement behind it."""
        recs, _ = synthetic_summary()
        stripped = []
        for r in recs:
            clone = copy.deepcopy(r)
            if clone["phase"] == "churn":
                (clone.get("extra") or {}).pop("recall_ci95", None)
            stripped.append(clone)
        summary = generate.summarize(stripped, make_manifest("x"))
        header, body = self._cells(render._churn_table(summary))
        column = header.index("Tied with")
        cells = {r[column] for r in body}
        assert render.TIE_NO_INTERVAL_CELL in cells, cells
        assert render.TIE_SEPARATED_CELL not in cells, cells
        assert not any(c.startswith("tied with ") for c in cells), cells

    def test_the_tie_group_holds_the_maintenance_state_constant(self):
        """A degraded index tied to a repaired one would be §4.2's defect
        wearing §5.1's clothes: PR 4 writes two records per fraction and
        pooling them made two readings of two different indexes look like one
        noisy measurement.

        Honest: pgvector's `after` row is alone at its point, so its cell is
            the dash -- no other engine has a repair path.
        Mutant (the state dropped from the group): the `after` row joins the
            `before` rows and comes back tied with MariaDB and AliSQL, which
            never ran a repair path at all.
        """
        _recs, summary = synthetic_summary()
        from report.churn_state import maintenance_state

        measurements = [r for r in summary["churn"]
                        if not (r.get("extra") or {}).get("during_writes")]
        after = [r for r in measurements
                 if maintenance_state(r) == "after"]
        assert after, "no repaired rows in the fixture"

        def group_with_state(r):
            return (r.get("dataset"), r.get("churn_fraction"),
                    maintenance_state(r), r.get("resource_pass"), r.get("m"),
                    r.get("build_mode"), r.get("ef_search"))

        def group_without_state(r):
            return (r.get("dataset"), r.get("churn_fraction"),
                    r.get("resource_pass"), r.get("m"),
                    r.get("build_mode"), r.get("ef_search"))

        honest = render._tie_cells(measurements, group_with_state)
        mutant = render._tie_cells(measurements, group_without_state)

        honest_cells = {honest[id(r)] for r in after}
        mutant_cells = {mutant[id(r)] for r in after}
        # Honest: every repaired row is alone at its point.
        assert honest_cells == {render.TIE_ALONE_CELL}, honest_cells
        # Mutant: none of them is alone any more, because the group now holds
        # the degraded rows of engines that never ran a repair path. One comes
        # back tied with them by name and one comes back "distinct" -- both
        # are claims about a comparison that did not happen, and neither is
        # the dash.
        assert render.TIE_ALONE_CELL not in mutant_cells, mutant_cells
        tied = [c for c in mutant_cells if c.startswith("tied with ")]
        assert tied, mutant_cells
        assert any("MariaDB" in c for c in tied), tied
        assert honest_cells != mutant_cells

    def test_a_tie_cell_names_the_storage_when_the_group_holds_two(self):
        """Engines differ on storage legitimately, so the group cannot hold it
        constant -- and the cell therefore has to name it, or a MariaDB
        InnoDB row tied with a MariaDB MyISAM row reads "tied with MariaDB"."""
        def rec(engine, storage, low, high):
            return {"engine": engine, "storage_engine": storage,
                    "dataset": "d", "churn_fraction": 0.1,
                    "extra": {"recall_ci95": {"low": low, "high": high,
                                              "n": 500}}}
        same_engine = [rec("mariadb", "InnoDB", 0.90, 0.95),
                       rec("mariadb", "MyISAM", 0.92, 0.97)]
        cells = render._tie_cells(same_engine, lambda r: r["dataset"])
        for value in cells.values():
            assert "MariaDB" in value and ("InnoDB" in value
                                           or "MyISAM" in value), value

        one_storage = [rec("mariadb", "InnoDB", 0.90, 0.95),
                       rec("alisql", "InnoDB", 0.92, 0.97)]
        cells = render._tie_cells(one_storage, lambda r: r["dataset"])
        for value in cells.values():
            assert "InnoDB" not in value, value

    def test_the_footnote_explains_all_three_cells(self):
        note = render.TIE_COLUMN_NOTE
        for needle in ("Tied with", "distinct", "no interval",
                       "5.1 item 6"):
            assert needle in note, needle


@needs_matplotlib
class TestTheChartsDrawTheIntervalsAsWhiskers:
    """§5.1 item 6: "show intervals as whiskers on churn and filtered
    charts"."""

    def test_the_helper_returns_asymmetric_clamped_error(self):
        rows = [{"extra": {"recall_ci95": {"low": 0.90, "high": 0.99,
                                           "n": 500}}}]
        out = charts.recall_whiskers(rows, [0.1], [0.95])
        px, py, (lower, upper) = out
        assert px == [0.1] and py == [0.95]
        assert lower == [pytest.approx(0.05)]
        assert upper == [pytest.approx(0.04)]

    def test_a_point_outside_its_own_rounded_interval_is_clamped(self):
        """`records` stores `round(recall, 6)` while the bounds are not
        rounded, so a point can sit just outside its interval. matplotlib
        raises `ValueError: 'yerr' must not contain negative values` on that,
        which would lose the whole figure to a rounding artifact.

        Honest: both error components are >= 0.
        Mutant (the clamp removed): the lower component is negative, which is
            exactly what matplotlib refuses.
        """
        rows = [{"extra": {"recall_ci95": {"low": 0.9500001, "high": 0.96,
                                           "n": 500}}}]
        _px, _py, (lower, upper) = charts.recall_whiskers(rows, [0.1], [0.95])
        assert lower[0] == 0.0 and upper[0] > 0
        unclamped = 0.95 - 0.9500001
        assert unclamped < 0
        assert min(lower + upper) >= 0.0

    def test_rows_without_a_usable_interval_are_skipped_entirely(self):
        assert charts.recall_whiskers([{"extra": {}}], [0.1], [0.95]) is None
        # A one-sample interval is not a measurement of spread.
        one = [{"extra": {"recall_ci95": {"low": 0.9, "high": 0.9, "n": 1}}}]
        assert charts.recall_whiskers(one, [0.1], [0.9]) is None

    @pytest.mark.parametrize("name,draw", [
        ("churn", lambda recs, d, o: charts.churn(recs, d, o, "churn-x")),
        ("filtered", lambda recs, d, o: charts.filtered(
            recs, d, o, "filtered-x", column="tag")),
    ])
    def test_both_charts_draw_error_bars_and_say_so(self, name, draw,
                                                    tmp_path, monkeypatch):
        """Counted off the AXIS, not off the selection predicate: the
        containers are what a reader sees."""
        recs = make_records("synthetic-run")
        dataset = recs[0]["dataset"]
        seen: Dict[str, int] = {"errorbar": 0}
        real = charts.plt.Axes.errorbar

        def counting(self, *a, **kw):
            seen["errorbar"] += 1
            return real(self, *a, **kw)

        monkeypatch.setattr(charts.plt.Axes, "errorbar", counting)
        titles: List[str] = []
        real_suptitle = charts.plt.Figure.suptitle

        def capture(self, text, *a, **kw):
            titles.append(str(text))
            return real_suptitle(self, text, *a, **kw)

        monkeypatch.setattr(charts.plt.Figure, "suptitle", capture)
        out = draw(recs, dataset, str(tmp_path))
        assert out is not None, name
        assert seen["errorbar"] > 0, name
        assert any(charts.WHISKER_TITLE_NOTE in t for t in titles), titles

    def test_a_chart_with_no_intervals_does_not_claim_any(self, tmp_path,
                                                          monkeypatch):
        """The other direction: an archived run has no intervals, and a legend
        explaining bars that are not on the figure is worse than none."""
        recs = []
        for r in make_records("synthetic-run"):
            clone = copy.deepcopy(r)
            (clone.get("extra") or {}).pop("recall_ci95", None)
            recs.append(clone)
        dataset = recs[0]["dataset"]
        seen = {"errorbar": 0}
        real = charts.plt.Axes.errorbar
        monkeypatch.setattr(
            charts.plt.Axes, "errorbar",
            lambda self, *a, **kw: (seen.update(
                errorbar=seen["errorbar"] + 1), real(self, *a, **kw))[1])
        titles: List[str] = []
        real_suptitle = charts.plt.Figure.suptitle
        monkeypatch.setattr(
            charts.plt.Figure, "suptitle",
            lambda self, text, *a, **kw: (titles.append(str(text)),
                                          real_suptitle(self, text, *a,
                                                        **kw))[1])
        assert charts.churn(recs, dataset, str(tmp_path), "churn-y") is not None
        assert seen["errorbar"] == 0
        assert not any(charts.WHISKER_TITLE_NOTE in t for t in titles), titles


# ---------------------------------------------------------------------------
# Added scope: the build-cost chart published two quantities on one axis
# ---------------------------------------------------------------------------

class TestATableCellCannotSilentlyShiftARow:
    """The table delimiter guard, and the two different authors it has.

    PR 5 §5.2 names its tolerance column `|delta|`, and writing that notation
    as a header cell put two extra pipes in the header row -- a ten-cell
    header over an eight-cell body, with the whole suite green. The only
    cell-count assertion in the repo covers the concurrency table, and a pipe
    added WITHOUT renaming a column leaves every `header.index(...)` lookup
    working, so nothing caught it.
    """

    def test_a_header_pipe_is_refused_and_names_the_column(self):
        """A header is a literal in `render.py`: a programming error, the
        same on every run, so failing fast is right."""
        with pytest.raises(ValueError) as excinfo:
            render._md_table(["Engine", "|delta|"], [["mariadb", "0.01"]])
        message = str(excinfo.value)
        assert "header cell" in message
        assert "'|delta|'" in message
        # The message says what to do, because the fix is a rename.
        assert "Recall gap" in message

    def test_a_body_pipe_is_substituted_so_the_report_survives(self):
        """A body cell can be driver-authored text -- `extra.maintenance` is
        whatever `maintain()` returned. Raising there would lose the whole
        report at the last step of a twenty-hour run, which is the failure
        `bootstrap_ci`'s docstring refuses to risk for one error bar.

        Honest: the row keeps its width and the pipe shows as a broken bar.
        Mutant (the body checked with `_check_no_delimiter`, which is what
            this file did for a round): ValueError, and `render_markdown`
            raises out of §9 with no report written.
        """
        table = render._md_table(["Engine", "Repair"],
                                 [["mariadb", "VACUUM | ANALYZE"]])
        header, _sep, body = table.strip().splitlines()
        assert len(body.strip("|").split("|")) == len(
            header.strip("|").split("|")) == 2
        assert render.MD_TABLE_DELIMITER_SUBSTITUTE in body
        assert "VACUUM | ANALYZE" not in body
        # The substitute is not the delimiter, to either renderer.
        assert render.MD_TABLE_DELIMITER_SUBSTITUTE != \
            render.MD_TABLE_DELIMITER
        # The mutant, run: the refusal form loses the table.
        with pytest.raises(ValueError):
            render._check_no_delimiter(["VACUUM | ANALYZE"], "body")

    def test_the_substitute_is_not_a_backslash_escape(self):
        """`_md_tables_to_html` splits a row on the raw delimiter, so an
        escaped pipe would render as two literal characters in the page while
        the Markdown looked right. Checked through the HTML path, not by
        reading the constant."""
        table = render._md_table(["Engine", "Repair"],
                                 [["mariadb", "VACUUM | ANALYZE"]])
        html = render._md_tables_to_html(table)
        assert "\\|" not in html
        assert render.MD_TABLE_DELIMITER_SUBSTITUTE in html
        # One row, two cells, in the HTML too -- which is the property the
        # whole guard exists for.
        assert html.count("<td>") == 2, html

    def test_every_table_in_the_consistency_section_is_rectangular(self):
        """The shape assertion the repo did not have, on the section this
        task added. Cell counts, not column names: the defect was a pipe
        added to a cell whose label never changed."""
        _recs, summary = synthetic_summary()
        text = render._consistency_section(summary)
        header = None
        tables = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("|") and stripped.endswith("|")):
                header = None
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            if header is None:
                header, tables = cells, tables + 1
                continue
            assert len(cells) == len(header), (
                f"a row has {len(cells)} cells against a {len(header)}-cell "
                f"header: {cells}")
        assert tables >= 3, tables


class TestBuildKindHasThreeValuesAndAbsenceIsOneOfThem:

    def test_the_three_kinds(self):
        assert generate.build_kind(
            {"extra": {"separable_build": True}}) == generate.BUILD_KIND_BULK
        assert generate.build_kind(
            {"extra": {"separable_build": False}}) == \
            generate.BUILD_KIND_INCREMENTAL
        assert generate.build_kind({"extra": {}}) == \
            generate.BUILD_KIND_UNRECORDED
        assert generate.build_kind({}) == generate.BUILD_KIND_UNRECORDED

    def test_absence_is_not_filed_as_incremental(self):
        """The defect in `render._index_build_kind`, which read the key with
        `.get` so a record that never had it came out "incremental" -- and
        every archived run contains pgvector, whose build is bulk.

        Honest: "not recorded".
        Mutant (`.get` truthiness, the previous form): "incremental", which
            is the opposite of the truth for the archived pgvector rows.
        """
        archived = {"engine": "pgvector", "extra": {"m_applied": True}}
        honest = render._index_build_kind(archived)
        mutant = ("bulk after load"
                  if (archived["extra"]).get("separable_build")
                  else "incremental")
        assert honest == "not recorded"
        assert mutant == "incremental"
        assert honest != mutant

    def test_the_comparable_figure_adds_the_load_only_for_a_bulk_build(self):
        bulk = {"build_wall_s": 37.0, "ingest_wall_s": 11.5,
                "extra": {"separable_build": True}}
        incremental = {"build_wall_s": 29.8, "ingest_wall_s": 29.8,
                       "extra": {"separable_build": False}}
        unknown = {"build_wall_s": 29.8, "ingest_wall_s": 29.8, "extra": {}}
        assert generate.time_to_queryable_index_s(bulk) == 48.5
        assert generate.time_to_queryable_index_s(incremental) == 29.8
        assert generate.time_to_queryable_index_s(unknown) is None

    def test_adding_the_load_to_an_incremental_build_would_double_it(self):
        """Why the incremental branch is `build_wall_s` alone.

        `build.py` writes `effective_build = load.wall_seconds` for an
        incremental build and `ingest_wall_s = load.wall_seconds` beside it,
        so the two are equal and summing them double counts the corpus load.

        Honest: 29.8 s. Mutant (ingest added unconditionally): 59.6 s --
        exactly double, which at 1M rows is hours.
        """
        incremental = {"build_wall_s": 29.8, "ingest_wall_s": 29.8,
                       "extra": {"separable_build": False}}
        honest = generate.time_to_queryable_index_s(incremental)
        mutant = incremental["build_wall_s"] + incremental["ingest_wall_s"]
        assert honest == 29.8
        assert mutant == pytest.approx(59.6)
        assert mutant == pytest.approx(2 * honest)

    def test_the_build_table_carries_the_comparable_column(self):
        _recs, summary = synthetic_summary()
        rows = table_cells(render._build_table(summary))
        header, body = rows[0], rows[1:]
        ttq = header.index("Time to queryable index (s)")
        kind = header.index("Index build")
        engine = header.index("Engine")
        assert {r[kind] for r in body} == {"bulk after load", "incremental"}
        # pgvector's comparable figure exceeds its raw build time, because the
        # load is added; the incremental engines' does not change.
        pg = [r for r in body if r[engine] == render._label("pgvector")]
        assert pg
        for row in pg:
            assert float(row[ttq].replace(",", "")) > 0


@needs_matplotlib
class TestTheBuildChartCannotPublishTheMisleadingComparison:
    """PR 5's added scope, proven the way the brief asks: with the fix a mixed
    set of bulk and incremental records must not produce a single-axis
    comparison, and with the fix reverted it must. Read off the RENDERED chart
    data, not off the selection predicate.
    """

    def _series(self, recs, dataset, tmp_path, monkeypatch, value_of=None):
        """Every (label, [(x, y)]) a `build_cost` panel actually plotted.

        Captured from `Axes.plot`, which is what a reader sees, and keyed by
        the axis's own y label so a panel can be identified by what it claims
        to show.
        """
        captured: List[Dict[str, Any]] = []
        real_plot = charts.plt.Axes.plot

        def capture(self, *args, **kw):
            if len(args) >= 2:
                captured.append({
                    "ylabel": self.get_ylabel(),
                    "label": kw.get("label"),
                    "points": list(zip(list(args[0]), list(args[1]))),
                })
            return real_plot(self, *args, **kw)

        monkeypatch.setattr(charts.plt.Axes, "plot", capture)
        out = charts.build_cost(recs, dataset, str(tmp_path), "build-x")
        assert out is not None
        return captured

    def _mixed(self):
        """Build records from both kinds, with the gap the real engines show:
        a bulk build whose index step is cheap beside a long load, and
        incremental engines whose build time IS their load."""
        recs = [r for r in make_records("synthetic-run")
                if r["phase"] == "index_build"]
        kinds = {generate.build_kind(r) for r in recs}
        assert kinds == {generate.BUILD_KIND_BULK,
                         generate.BUILD_KIND_INCREMENTAL}, kinds
        return recs, recs[0]["dataset"]

    def test_the_first_panel_is_one_quantity_for_every_engine(self, tmp_path,
                                                              monkeypatch):
        recs, dataset = self._mixed()
        series = self._series(recs, dataset, tmp_path, monkeypatch)
        comparable = [s for s in series
                      if s["ylabel"].startswith("Time to a queryable index")]
        assert comparable, [s["ylabel"] for s in series]
        # Every engine is on it, and each engine's value is its own
        # `time_to_queryable_index_s` -- so the axis holds one quantity.
        plotted = {}
        for s in comparable:
            for x, y in s["points"]:
                plotted[(s["label"], x)] = y
        expected = {}
        for r in recs:
            expected[(None, r["m"])] = generate.time_to_queryable_index_s(r)
        assert plotted, series
        for (label, m), y in plotted.items():
            matching = [r for r in recs if r["m"] == m
                        and generate.time_to_queryable_index_s(r)
                        == pytest.approx(y)]
            assert matching, (label, m, y)

    def test_the_bulk_step_panel_holds_only_bulk_records(self, tmp_path,
                                                         monkeypatch):
        recs, dataset = self._mixed()
        series = self._series(recs, dataset, tmp_path, monkeypatch)
        step = [s for s in series
                if s["ylabel"].startswith("Bulk index build step")]
        assert step, [s["ylabel"] for s in series]
        plotted = {y for s in step for _x, y in s["points"]}
        bulk_values = {r["build_wall_s"] for r in recs
                       if generate.build_kind(r) == generate.BUILD_KIND_BULK}
        incremental_values = {
            r["build_wall_s"] for r in recs
            if generate.build_kind(r) == generate.BUILD_KIND_INCREMENTAL}
        assert plotted <= bulk_values, (plotted, bulk_values)
        assert plotted
        # The crux: no incremental engine's load time is on that axis.
        assert plotted.isdisjoint(incremental_values), (plotted,
                                                        incremental_values)

    def test_reverting_the_fix_puts_both_quantities_on_one_axis(
            self, tmp_path, monkeypatch):
        """The mutation the brief asks for, with values read back off the
        rendered chart.

        Honest: the "Time to a queryable index" axis holds pgvector at
            ingest+build and the incremental engines at their single figure,
            and the "Bulk index build step" axis holds only pgvector.
        Mutant (`time_to_queryable_index_s` replaced by the raw
            `build_wall_s`, which is what the chart plotted before this task):
            one axis carries pgvector's `CREATE INDEX` time beside MariaDB's
            whole corpus load, and at M=16 that is 37.14 against 29.82 -- two
            different operations under a "lower is better" arrow.
        """
        recs, dataset = self._mixed()
        honest = self._series(recs, dataset, tmp_path, monkeypatch)
        honest_axis = {y for s in honest for _x, y in s["points"]
                       if s["ylabel"].startswith("Time to a queryable index")}

        original = generate.time_to_queryable_index_s
        try:
            generate.time_to_queryable_index_s = \
                lambda record: record.get("build_wall_s")
            mutant = self._series(recs, dataset, tmp_path, monkeypatch)
        finally:
            generate.time_to_queryable_index_s = original
        mutant_axis = {y for s in mutant for _x, y in s["points"]
                       if s["ylabel"].startswith("Time to a queryable index")}

        bulk = {r["build_wall_s"] for r in recs
                if generate.build_kind(r) == generate.BUILD_KIND_BULK}
        incremental = {
            r["build_wall_s"] for r in recs
            if generate.build_kind(r) == generate.BUILD_KIND_INCREMENTAL}

        # The mutant axis holds BOTH raw quantities -- the defect.
        assert bulk <= mutant_axis, (bulk, mutant_axis)
        assert incremental <= mutant_axis, (incremental, mutant_axis)
        # The honest axis holds the incremental figures (which legitimately
        # equal their load) but NOT the bulk build step on its own.
        assert bulk.isdisjoint(honest_axis), (bulk, honest_axis)
        assert honest_axis != mutant_axis
        # And the specific pair the defect published, read back as numbers.
        at_16 = {r["engine"]: r for r in recs if r["m"] == 16}
        assert round(at_16["pgvector"]["build_wall_s"], 2) == 37.14
        assert round(at_16["mariadb"]["build_wall_s"], 2) == 29.82
        assert round(generate.time_to_queryable_index_s(
            at_16["pgvector"]), 2) == 48.68
        assert round(generate.time_to_queryable_index_s(
            at_16["mariadb"]), 2) == 29.82

    def test_an_unrecorded_record_is_named_rather_than_filed(self, tmp_path,
                                                             monkeypatch):
        """A record with `separable_build` absent appears on neither of the
        first two panels, and the subtitle says how many were left off and
        why. Dropped silently is the one thing it must not be."""
        recs, dataset = self._mixed()
        stripped = []
        for r in recs:
            clone = copy.deepcopy(r)
            if clone["engine"] == "alisql":
                clone["extra"].pop("separable_build", None)
            stripped.append(clone)
        assert any(generate.build_kind(r) == generate.BUILD_KIND_UNRECORDED
                   for r in stripped)

        titles: List[str] = []
        real_suptitle = charts.plt.Figure.suptitle
        monkeypatch.setattr(
            charts.plt.Figure, "suptitle",
            lambda self, text, *a, **kw: (titles.append(str(text)),
                                          real_suptitle(self, text, *a,
                                                        **kw))[1])
        series = self._series(stripped, dataset, tmp_path, monkeypatch)

        # Counted by SERIES, not by value. The first version of this
        # assertion was `comparable.isdisjoint({v for v in alisql_values if v
        # not in mariadb_values})` -- and alisql and mariadb carry the same
        # three figures in this fixture, so that set is empty and
        # `isdisjoint(set())` is unconditionally True. A test that passes
        # whatever the chart does, which is the "tests passing against empty
        # input" shape on the vacuity list.
        #
        # The honest question is whether the unrecorded engine has a SERIES
        # on each of the first two panels, which is independent of whether
        # its numbers coincide with another engine's.
        def labels_on(prefix):
            return {s["label"] for s in series
                    if s["ylabel"].startswith(prefix) and s["label"]}

        # A series label is the engine label plus the series axes --
        # "MariaDB 11.8 (MHNSW) / M=16" -- so membership is tested on the " /"
        # delimiter rather than with a bare `in`. An unanchored substring is
        # the shape this repo has shipped ten times, `"tag"` inside `"ctag"`
        # among them.
        def holds(labels, engine):
            label = render._label(engine)
            return any(lab == label or lab.startswith(label + " /")
                       for lab in labels)

        comparable_labels = labels_on("Time to a queryable index")
        step_labels = labels_on("Bulk index build step")
        memory_labels = labels_on("Peak server memory")

        assert not holds(comparable_labels, "alisql"), comparable_labels
        assert not holds(step_labels, "alisql"), step_labels
        # Not vacuous in two directions: the panels do hold other engines,
        # and alisql is still on the panel that does not depend on the build
        # kind -- so it was excluded from these two specifically rather than
        # dropped from the chart.
        assert comparable_labels, series
        assert holds(comparable_labels, "mariadb"), comparable_labels
        assert holds(memory_labels, "alisql"), memory_labels
        # ...and the reader is told.
        assert any("left off the first two panels" in t for t in titles), \
            titles
        assert any("alisql" in t for t in titles), titles
