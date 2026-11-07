"""Unit tests for the dataset layer's filter columns and ground truth.

Two filter columns are measured, and the difference between them is the whole
point of the filtered-search comparison:

* `tag = row_id % 100` is **independent of the vector space**. The rows passing
  `tag < t` are a uniform random sample of the corpus, so a query's true
  neighbours pass the filter about `t`% of the time and an engine that
  post-filters only has to look a little further down its candidate list.
* `ctag` is a **nearest-pivot cluster id**, so its values follow the vector
  space. The rows passing `ctag < t` are t whole clusters, and for most queries
  the actual nearest neighbours are in a cluster that fails the filter. That is
  the case where pre-filtering, post-filtering and index iteration diverge.

The single highest-risk number in this file is the **actual** selectivity.
Cluster sizes are unequal by construction, so "1%" requested is never 1%
delivered, and this project has already published four claimed-as-measured
values. Every test here that touches selectivity checks the measured fraction
and checks that it differs from the nominal one — a test where the two coincide
proves nothing.

Run: python3 -m pytest tests/test_datasets.py -v     (from the vector-bench directory)
"""

from __future__ import annotations

import ast
import glob
import os
import subprocess
import sys
import textwrap

import numpy
import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from harness import datasets as ds  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures. None of these supply the thing under test: the corpora are plain
# vectors, and every expected value is derived independently of the
# implementation (explicit norms, hand-built label arrays, closed-form shares).
# ---------------------------------------------------------------------------

def _mixed_norm_corpus(n: int = 600, dim: int = 6, seed: int = 11) -> numpy.ndarray:
    """Random directions times lognormal norms spanning ~two decades.

    Euclidean and angular nearest-pivot assignments genuinely disagree here,
    which a uniform positive corpus does not deliver: scale is almost all of
    the Euclidean distance and none of the cosine distance. The fixture's own
    test (`test_the_fixture_really_separates_the_two_metrics`) measures that
    disagreement rather than assuming it.
    """
    rng = numpy.random.default_rng(seed)
    directions = rng.normal(size=(n, dim))
    directions /= numpy.linalg.norm(directions, axis=1, keepdims=True)
    norms = numpy.exp(rng.normal(0.0, 1.2, size=(n, 1)))
    return (directions * norms).astype("<f4")


def _line_of_blobs(n_blobs: int = 20, per_blob: int = 50, dim: int = 8,
                   seed: int = 3) -> numpy.ndarray:
    """Tight blobs whose centres march along one axis, rows ordered by blob.

    Row order is therefore spatial order, which is what makes this fixture able
    to see a missing renumbering permutation: pivots taken in row order would
    make `ctag < t` a contiguous left-hand chunk of the line.
    """
    rng = numpy.random.default_rng(seed)
    rows = []
    for j in range(n_blobs):
        centre = numpy.zeros(dim)
        centre[0] = j * 10.0
        rows.append(centre + rng.normal(0.0, 0.3, size=(per_blob, dim)))
    return numpy.concatenate(rows).astype("<f4")


def _pivot_rows(n: int, n_clusters: int, seed: int) -> numpy.ndarray:
    """The pivot draw the spec prescribes, reproduced for the reference labels.

    This is the one thing the tests take from the spec rather than deriving:
    §3.1 item 1.1 fixes the draw as
    `numpy.random.default_rng(seed).choice(..., replace=False)`. The mapping
    from pivots to labels — the part that can be wrong — is still computed
    independently below.
    """
    return numpy.random.default_rng(seed).choice(n, size=n_clusters, replace=False)


def _reference_labels(train: numpy.ndarray, pivot_rows: numpy.ndarray,
                      metric: str) -> numpy.ndarray:
    """Nearest-pivot labels by explicit distance, in float64.

    Deliberately *not* the implementation's expression: it forms the difference
    vectors and takes their norm, rather than expanding ||a-b||^2, and uses
    normalised dot products for angular. A reimplementation of the same algebra
    would hide an error in the algebra.
    """
    t64 = numpy.asarray(train, dtype=numpy.float64)
    p64 = t64[pivot_rows]
    if metric == ds.ANGULAR:
        a = t64 / numpy.linalg.norm(t64, axis=1, keepdims=True)
        b = p64 / numpy.linalg.norm(p64, axis=1, keepdims=True)
        return numpy.argmax(a @ b.T, axis=1)
    d = numpy.linalg.norm(t64[:, None, :] - p64[None, :, :], axis=2)
    return numpy.argmin(d, axis=1)


def _is_a_relabelling_of(got: numpy.ndarray, reference: numpy.ndarray) -> bool:
    """True when `got` partitions the rows exactly as `reference` does.

    Permutation-invariant, so it checks the partition the metric produced
    without needing to know which permutation the implementation applied.
    """
    pairs = set(zip(reference.tolist(), got.tolist()))
    return (len(pairs) == len(set(reference.tolist()))
            == len(set(got.tolist())))


def _min_margin_over_noise(train: numpy.ndarray, pivot_rows: numpy.ndarray,
                           metric: str) -> float:
    """Tightest row's winner-to-runner-up margin, in multiples of float32 noise.

    A fixture with a genuine near-tie would make the float32 implementation and
    the float64 reference legitimately disagree on one row, which would read as
    a defect. Tests that demand an exact partition match assert this first.

    The denominator is the quantity that decides whether a margin survives
    float32: the implementation sums `||a||^2 - 2 a.b + ||b||^2`, so the
    absolute error on one cost is about `eps * (||a||^2 + ||b||^2)` — set by
    the magnitudes of the terms being summed, which on a corpus whose norms
    span two decades is a *per-row* quantity and not a corpus-wide one.

    The first version of this helper divided instead by the largest cost
    anywhere in the matrix, and rejected a sound fixture. All four figures
    below are measured on the 600 x 6 fixture with 20 pivots at seed 2024.

    Under this helper, the tightest margin is 8,837.8x the noise for euclidean
    and 3,021.7x for angular, both far above the 100x floor. Under the old
    one, the same fixture gave 7.4535e-06 for euclidean and 7.2043e-04 for
    angular against a 1e-4 floor — in noise units, 62.5x and 6,043x against a
    floor worth 839x. That is the whole story of the failure: the old
    denominator deflated euclidean by ~140x and left angular untouched,
    because with unit vectors the corpus-wide and per-row scales are the same
    number — so euclidean dropped under the floor and angular never came near
    it. The ~140x is not a single ratio: the two helpers differ both in the
    quantity they measure (distance margin against squared-distance margin)
    and in the scale they divide by (the corpus's largest distance, 45.2594,
    against the magnitudes the tightest row's own comparison sums, whose
    deciding distances are 1.9082 and 1.9097).

    The code under test was right and this helper's reasoning was wrong, which
    is the shape this repo has now shipped five times.
    """
    t64 = numpy.asarray(train, dtype=numpy.float64)
    p64 = t64[pivot_rows]
    if metric == ds.ANGULAR:
        a = t64 / numpy.linalg.norm(t64, axis=1, keepdims=True)
        b = p64 / numpy.linalg.norm(p64, axis=1, keepdims=True)
        cost = -(a @ b.T)
        # Both operands are unit vectors, so no term summed exceeds 1.
        noise_scale = numpy.full(len(t64), 2.0)
    else:
        cost = ((t64[:, None, :] - p64[None, :, :]) ** 2).sum(axis=2)
        nearest_two = numpy.argsort(cost, axis=1)[:, :2]
        noise_scale = ((t64 ** 2).sum(axis=1)
                       + (p64 ** 2).sum(axis=1)[nearest_two].max(axis=1))
    best_two = numpy.sort(cost, axis=1)[:, :2]
    margin = best_two[:, 1] - best_two[:, 0]
    noise = noise_scale * numpy.finfo(numpy.float32).eps
    return float((margin / noise).min())


# A margin has to clear the rounding noise by two orders of magnitude before an
# exact-match assertion is fair. The error accumulated over a dot product of
# `dim` terms grows like sqrt(dim) * eps, so for these fixtures (dim 5-8) the
# realistic noise is a few eps and 100x is a ~40x safety factor on top. Derived
# from float32's epsilon, not from any measurement of this implementation.
SAFE_MARGIN_IN_EPS = 100.0


def _dataset(train: numpy.ndarray, queries: int = 4, name: str = "t",
             metric: str = ds.EUCLIDEAN) -> ds.Dataset:
    rng = numpy.random.default_rng(99)
    test = rng.normal(size=(queries, train.shape[1])).astype("<f4")
    neighbors = numpy.zeros((queries, 10), dtype=numpy.int64)
    return ds.Dataset(name, train, test, neighbors, None, metric)


def _brute_force(train: numpy.ndarray, query: numpy.ndarray,
                 mask: numpy.ndarray, metric: str, k: int) -> set:
    """Exact top-k over the masked rows, by the most literal method available."""
    qualifying = numpy.nonzero(mask)[0]
    t64 = numpy.asarray(train[qualifying], dtype=numpy.float64)
    q64 = numpy.asarray(query, dtype=numpy.float64)
    if metric == ds.ANGULAR:
        a = t64 / numpy.linalg.norm(t64, axis=1, keepdims=True)
        cost = -(a @ (q64 / numpy.linalg.norm(q64)))
    else:
        cost = numpy.linalg.norm(t64 - q64, axis=1)
    return set(qualifying[numpy.argsort(cost)[:k]].tolist())


# ---------------------------------------------------------------------------
# assign_ctags
# ---------------------------------------------------------------------------

class TestAssignCtagsDeterminism:
    """Three separate things have to be reproducible, not one.

    The pivot draw, the nearest-pivot assignment and the renumbering
    permutation all come off one seeded generator, and a run scored against a
    cached assignment from a different draw is scored against the wrong answer
    set. The spec gives a single seed for all three, so a single seed has to
    pin all three.
    """

    def test_same_seed_gives_the_same_labels(self):
        corpus = _mixed_norm_corpus(300, 5, seed=7)
        a = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=12, seed=1234)
        b = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=12, seed=1234)
        assert numpy.array_equal(a, b)

    def test_a_different_seed_gives_a_different_partition(self):
        """Not merely different labels: a different grouping of rows.

        Relabelling the same clusters would satisfy `not array_equal` while
        leaving every qualifying set identical, so the check is on the
        partition.
        """
        corpus = _mixed_norm_corpus(300, 5, seed=7)
        a = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=12, seed=1234)
        b = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=12, seed=5678)
        assert not numpy.array_equal(a, b)
        assert not _is_a_relabelling_of(b, a), (
            "a different seed only renamed the clusters; the pivots cannot "
            "have moved"
        )

    def test_determinism_holds_across_processes(self, tmp_path):
        """A fresh interpreter must reproduce the array byte for byte.

        `numpy.random.default_rng` is documented as reproducible across
        processes, but an earlier task in this series claimed byte-identical
        reruns while only ever testing two calls inside one interpreter — the
        property held by accident. The corpus is written to disk and loaded by
        the child so that only the assignment's own generator is under test,
        not the corpus generator's.
        """
        corpus = _mixed_norm_corpus(400, 6, seed=5)
        corpus_path = tmp_path / "corpus.npy"
        numpy.save(corpus_path, corpus)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(ds.__file__)))
        script = textwrap.dedent(
            """
            import sys
            import numpy
            sys.path.insert(0, sys.argv[1])
            from harness import datasets as ds
            corpus = numpy.load(sys.argv[2])
            out = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=17,
                                  seed=20261012)
            numpy.save(sys.argv[3], out)
            """
        )
        here = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=17,
                               seed=20261012)
        children = []
        for i in range(2):
            out_path = tmp_path / f"child{i}.npy"
            subprocess.run(
                [sys.executable, "-I", "-c", script, repo_root,
                 str(corpus_path), str(out_path)],
                check=True, capture_output=True,
            )
            children.append(numpy.load(out_path))
        assert numpy.array_equal(children[0], children[1])
        assert numpy.array_equal(here, children[0])
        assert here.tobytes() == children[0].tobytes()


class TestAssignCtagsFollowsTheMetric:
    """§3.5 asks for "every cluster non-empty on a 2,000-row fixture", which an
    assignment that ignores the vectors entirely also satisfies — 2,000 rows
    thrown at random into 100 buckets leaves none empty either. These tests ask
    the question that one cannot: are the clusters the *nearest-pivot* cells of
    the dataset's own metric?
    """

    def test_the_fixture_really_separates_the_two_metrics(self):
        """Guard on the fixture, not on the code.

        On a uniform positive corpus Euclidean and angular nearest-pivot
        labels agree almost everywhere, so a metric-ignoring implementation
        would pass both metric tests below. This measures that the fixture
        forces a real choice. The 10% floor is far below what random directions
        with lognormal norms produce and is not calibrated to any measurement.
        """
        corpus = _mixed_norm_corpus(600, 6, seed=11)
        pivots = _pivot_rows(len(corpus), 20, 2024)
        euclid = _reference_labels(corpus, pivots, ds.EUCLIDEAN)
        angular = _reference_labels(corpus, pivots, ds.ANGULAR)
        disagreement = float((euclid != angular).mean())
        assert disagreement > 0.10, (
            f"fixture is metric-blind: only {disagreement:.1%} of rows change "
            f"cluster between euclidean and angular"
        )

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    def test_clusters_are_the_nearest_pivot_cells_of_that_metric(self, metric):
        corpus = _mixed_norm_corpus(600, 6, seed=11)
        pivots = _pivot_rows(len(corpus), 20, 2024)
        margin = _min_margin_over_noise(corpus, pivots, metric)
        assert margin > SAFE_MARGIN_IN_EPS, (
            f"fixture's tightest margin is only {margin:.0f}x float32 noise; a "
            f"float32/float64 disagreement here would look like a defect"
        )
        got = ds.assign_ctags(corpus, metric, n_clusters=20, seed=2024)
        reference = _reference_labels(corpus, pivots, metric)
        assert _is_a_relabelling_of(got, reference)

    def test_the_angular_assignment_is_not_the_euclidean_one(self):
        """The two metrics must produce different partitions on this corpus.

        Without this, an implementation that normalised for both metrics (or
        for neither) would pass the two parametrised cases above only if the
        reference happened to agree — and this corpus is chosen so it does not.
        """
        corpus = _mixed_norm_corpus(600, 6, seed=11)
        euclid = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=20, seed=2024)
        angular = ds.assign_ctags(corpus, ds.ANGULAR, n_clusters=20, seed=2024)
        assert not _is_a_relabelling_of(angular, euclid)

    def test_every_pivot_lands_in_its_own_cluster(self):
        """A row at distance zero from its pivot cannot be anywhere else.

        Metric-sensitive in a way non-emptiness is not: a random assignment
        leaves no cluster empty on a 2,000-row fixture but scatters the pivots.
        It is also why non-emptiness is structural here rather than lucky — the
        pivots are rows of the corpus, so each cluster contains at least one.
        """
        corpus = _mixed_norm_corpus(800, 5, seed=13)
        pivots = _pivot_rows(len(corpus), 25, 77)
        got = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=25, seed=77)
        assert len(set(got[pivots].tolist())) == 25

    def test_every_cluster_is_non_empty_on_a_2000_row_fixture(self):
        # §3.5's own check, kept as the floor it is.
        corpus = _mixed_norm_corpus(2000, 8, seed=19)
        got = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=100,
                              seed=ds.CTAG_SEED)
        counts = numpy.bincount(got, minlength=100)
        assert len(counts) == 100
        assert (counts > 0).all()
        assert int(counts.sum()) == 2000


class TestAssignCtagsBatching:
    """§3.1 item 1.2 fixes the batch at 8,192 rows: the full row-by-pivot cost
    matrix at 1M rows and 100 pivots is 800 MB in float64, and the harness
    shares its container with the corpus. Batching is a memory decision and
    must not become a correctness one.
    """

    def test_the_default_batch_is_8192_rows(self):
        assert ds.CTAG_BATCH_ROWS == 8192

    @pytest.mark.parametrize("batch_rows", [1, 7, 64, 8192, 10_000])
    def test_the_batch_size_does_not_change_the_labels(self, batch_rows):
        corpus = _mixed_norm_corpus(301, 5, seed=23)
        whole = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=11, seed=31,
                                batch_rows=len(corpus))
        got = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=11, seed=31,
                              batch_rows=batch_rows)
        assert numpy.array_equal(got, whole)

    def test_a_ragged_final_batch_is_assigned_too(self):
        """301 rows in batches of 64 leaves a 45-row tail.

        A loop that wrote every batch to the same slice, or that skipped the
        remainder, would leave the tail holding whatever `numpy.empty`
        allocated — which looks like a valid label.
        """
        corpus = _mixed_norm_corpus(301, 5, seed=23)
        pivots = _pivot_rows(301, 11, 31)
        got = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=11, seed=31,
                              batch_rows=64)
        reference = _reference_labels(corpus, pivots, ds.EUCLIDEAN)
        assert _is_a_relabelling_of(got[256:], reference[256:])
        assert _is_a_relabelling_of(got, reference)


class TestCtagRenumbering:
    """§3.1 item 1.3: `ctag < t` must select t *random* clusters.

    Determinism and non-emptiness cannot see a missing permutation, so these
    look at what the permutation is for. The first pins that a permutation was
    applied at all; the second pins the property the permutation exists to
    protect — that the rows `ctag < t` selects are spread across the space
    rather than occupying one end of it.

    One honest limitation, measured rather than assumed. While the pivot draw
    is random, removing the permutation does *not* move the spatial statistic:
    over twelve seeds the median spread ratio is 1.05 with the permutation and
    0.98 without it, because a random draw already puts random clusters first.
    The permutation earns its place against an *ordered* draw — sorting the
    pivot indices for cache-friendly fancy indexing is the plausible edit, and
    on a corpus stored in class order that makes `ctag < t` one spatial corner.
    Holding the draw sorted, the median spread ratio is 0.95 with the
    permutation and 0.34 without it. So: the second test below fails whenever
    the selection becomes spatially ordered, which is the defect; the first
    fails if the permutation disappears, which is the cause.
    """

    def test_labels_are_not_the_raw_nearest_pivot_indices(self):
        corpus = _mixed_norm_corpus(600, 6, seed=11)
        pivots = _pivot_rows(len(corpus), 20, 2024)
        got = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=20, seed=2024)
        reference = _reference_labels(corpus, pivots, ds.EUCLIDEAN)
        assert _is_a_relabelling_of(got, reference)
        assert not numpy.array_equal(got, reference), (
            "cluster ids are the raw pivot indices; no permutation was applied"
        )

    # Twelve pivot draws, because one draw is not a measurement. Note which
    # parameter is varied: the draw depends on the row count and the ctag seed
    # and on nothing else, so parametrising the *corpus* seed instead — which
    # is what the first version of this test did — selects the same twelve
    # clusters twelve times and reports one point as twelve.
    SPATIAL_SEEDS = (ds.CTAG_SEED, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)

    def test_the_selected_clusters_are_spread_across_the_corpus(self):
        """The spatial check §3.1 item 1.3 asks for.

        Rows are ordered along a line, so a selection that followed row order
        would occupy one end of it. The statistic is the standard deviation of
        the selected rows' position on that axis over the whole corpus's —
        "comparable to the whole corpus rather than much smaller" — and the
        standard deviation rather than the range because a range is decided by
        two extreme points.

        Eight clusters sampled from a line of twenty cover it much as the whole
        corpus does, so the ratio is about 1; a contiguous prefix of eight
        covers the leftmost 40% and measures about 0.34. The floors below sit
        between the two: 0.6 on the median (honest 1.05, prefix 0.34) and 0.55
        on the tightest of the twelve draws, a 20% margin under the honest
        minimum of 0.69. Both distributions are in this class's docstring, so
        the margin can be checked rather than taken on trust.
        """
        corpus = _line_of_blobs(n_blobs=20, per_blob=50, dim=8, seed=3)
        axis = corpus[:, 0]
        ratios = []
        for seed in self.SPATIAL_SEEDS:
            ctags = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=20,
                                    seed=seed)
            selected = corpus[ctags < 8, 0]
            ratios.append(float(selected.std() / axis.std()))
        median = float(numpy.median(ratios))
        assert median > 0.6, (
            f"median spread of `ctag < 8` over {len(ratios)} pivot draws is "
            f"{median:.2f} of the corpus's: the selection is following "
            f"spatial order"
        )
        assert min(ratios) > 0.55, (
            f"tightest draw spreads only {min(ratios):.2f} of the corpus: "
            f"{['%.2f' % r for r in ratios]}"
        )


class TestAssignCtagsRejectsNonsense:
    def test_more_clusters_than_rows(self):
        corpus = _mixed_norm_corpus(10, 3, seed=1)
        with pytest.raises(ValueError):
            ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=11, seed=1)

    def test_zero_clusters(self):
        corpus = _mixed_norm_corpus(10, 3, seed=1)
        with pytest.raises(ValueError):
            ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=0, seed=1)

    def test_an_empty_corpus(self):
        # A test that passes on empty input proves nothing about the loop.
        with pytest.raises(ValueError):
            ds.assign_ctags(numpy.zeros((0, 3), dtype="<f4"), ds.EUCLIDEAN,
                            n_clusters=1, seed=1)

    def test_an_unknown_metric(self):
        corpus = _mixed_norm_corpus(10, 3, seed=1)
        with pytest.raises(ValueError):
            ds.assign_ctags(corpus, "manhattan", n_clusters=2, seed=1)


# ---------------------------------------------------------------------------
# Thresholds and the actual selectivity
# ---------------------------------------------------------------------------

def _ctags_with_sizes(sizes) -> numpy.ndarray:
    """A label array with exactly these cluster sizes, in label order."""
    return numpy.concatenate(
        [numpy.full(n, j, dtype=numpy.int64) for j, n in enumerate(sizes)]
    )


# Sizes chosen so every cumulative share below is exactly representable as a
# double, which lets the boundary cases be asserted as equalities rather than
# as tolerances: 125/1000 = 1/8 and 500/1000 = 1/2.
UNEQUAL_SIZES = (125, 375, 100, 400)


class TestCtagThresholdForRecordsTheActualShare:
    """The one sentence in §3.1 that matters most: cluster sizes are unequal, so
    the actual selectivity must be recorded and never the nominal one.

    Four claimed-as-measured values have already shipped in this project. A
    nominal 1% recorded where the measurement is 13.7% would be the fifth, and
    nothing downstream could tell the difference, because both are a float in
    the `selectivity` column.
    """

    def test_the_actual_share_is_returned_and_differs_from_the_nominal_one(self):
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        t, actual = ds.ctag_threshold_for(ctags, 0.01)
        assert t == 1
        assert actual == 0.125, "0.125 is cluster 0's share: 125 of 1000 rows"
        assert actual != 0.01

    def test_the_gap_is_not_a_rounding_artefact(self):
        """A nominal 12.6% request delivers 50% of the corpus.

        One cluster boundary away from the target is half the corpus here. A
        report that printed 12.6% for this run would be wrong by a factor of
        four, and would look exactly like a correct 12.6% run.
        """
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        t, actual = ds.ctag_threshold_for(ctags, 0.126)
        assert (t, actual) == (2, 0.5)

    def test_an_exactly_met_target_does_not_advance_the_threshold(self):
        # "cumulative share >= the target", not ">".
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        assert ds.ctag_threshold_for(ctags, 0.125) == (1, 0.125)
        assert ds.ctag_threshold_for(ctags, 0.5) == (2, 0.5)

    def test_the_threshold_is_the_smallest_one_that_meets_the_target(self):
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        for nominal in (0.001, 0.01, 0.125, 0.126, 0.4, 0.5, 0.55, 0.61, 0.99):
            t, actual = ds.ctag_threshold_for(ctags, nominal)
            assert actual >= nominal - 1e-12
            if t > 1:
                below = float((ctags < t - 1).mean())
                assert below < nominal, (
                    f"t={t} is not minimal for {nominal}: t-1 already "
                    f"delivers {below}"
                )

    def test_empty_clusters_do_not_advance_the_threshold(self):
        """Two empty clusters sit on the 50% boundary.

        `searchsorted(..., side="right")` walks past them and selects the whole
        corpus instead of half of it — a 2x error in the selectivity, in the
        direction that makes filtered search look easy.
        """
        ctags = _ctags_with_sizes((50, 0, 0, 50))
        assert ds.ctag_threshold_for(ctags, 0.5, n_clusters=4) == (1, 0.5)

    def test_a_tiny_target_still_selects_a_non_empty_set(self):
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        t, actual = ds.ctag_threshold_for(ctags, 0.0)
        assert t == 1
        assert actual == 0.125

    def test_a_target_above_one_is_clamped_to_every_cluster(self):
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        assert ds.ctag_threshold_for(ctags, 2.0) == (4, 1.0)

    def test_trailing_empty_clusters_are_counted_when_declared(self):
        """`n_clusters` cannot be inferred from the labels when the tail is
        empty, and a threshold clamped to `max(label)+1` would quietly cap a
        100%-selectivity request below 100% of the declared range."""
        ctags = _ctags_with_sizes((500, 500, 0, 0))
        assert ds.ctag_threshold_for(ctags, 2.0, n_clusters=4) == (4, 1.0)
        assert ds.ctag_threshold_for(ctags, 2.0) == (2, 1.0)

    def test_an_empty_label_array_raises(self):
        with pytest.raises(ValueError):
            ds.ctag_threshold_for(numpy.zeros(0, dtype=numpy.int64), 0.1)

    def test_the_result_names_its_fields(self):
        """`(t, actual)` by position, `.actual_selectivity` by name.

        The caller records `filt.actual_selectivity`; there is no attribute on
        this object that returns the nominal value, so recording the nominal
        one cannot happen by typo.
        """
        got = ds.ctag_threshold_for(_ctags_with_sizes(UNEQUAL_SIZES), 0.3)
        assert got._fields == ("threshold", "actual_selectivity")
        t, actual = got
        assert (t, actual) == (got.threshold, got.actual_selectivity)
        assert not [name for name in dir(got) if "nominal" in name]


class TestCtagThresholdOnARealisticDistribution:
    """Nearest-pivot cells of random pivots, which is what a run actually sees.

    Twenty clusters over 2,000 rows means the cumulative share moves in steps
    of roughly 5% of the corpus, so a request can only land on the target by
    coincidence; the typical overshoot is a couple of per cent per request. The
    floor below is set from that granularity, not from this fixture's
    measurement.
    """

    NOMINALS = (0.01, 0.10, 0.50)

    def _ctags(self):
        return ds.assign_ctags(_mixed_norm_corpus(2000, 8, seed=19),
                               ds.EUCLIDEAN, n_clusters=20, seed=ds.CTAG_SEED)

    def test_every_request_overshoots_rather_than_undershoots(self):
        ctags = self._ctags()
        for nominal in self.NOMINALS:
            _, actual = ds.ctag_threshold_for(ctags, nominal, n_clusters=20)
            assert actual >= nominal, (
                f"{nominal} requested, {actual} delivered: a filter that "
                f"selects fewer rows than asked for is not the same experiment"
            )

    def test_the_nominal_values_are_not_what_was_measured(self):
        ctags = self._ctags()
        gap = sum(
            abs(ds.ctag_threshold_for(ctags, nominal, n_clusters=20)[1] - nominal)
            for nominal in self.NOMINALS
        )
        assert gap > 0.01, (
            f"total |actual - nominal| over {self.NOMINALS} is only {gap}: "
            f"this fixture cannot tell a recorded nominal from a measured one"
        )


class TestTagThresholdIsMeasuredToo:
    """`tag = row_id % buckets` is uniform only when the row count is a
    multiple of the bucket count, and `threshold / buckets` is therefore not
    the selectivity — it is the selectivity the harness *intended*.

    The error is small for `tag` and unbounded for `ctag`, and the same code
    path reports both, so neither is allowed to assume. Measured: on 150 rows
    a 1% request delivers 1.3333%, and on glove's full 1,183,514 training rows
    it delivers 1.000073% rather than 1%. Every row count the profiles
    currently use is a multiple of 100, which is exactly why
    `threshold / 100.0` survived this long.
    """

    def test_a_row_count_that_is_not_a_multiple_of_the_buckets(self):
        tags = ds.assign_tags(150, buckets=100)
        t, actual = ds.tag_threshold_for(tags, 0.01, buckets=100)
        assert t == 1
        assert actual == pytest.approx(2 / 150)
        assert actual != pytest.approx(0.01), (
            "threshold/buckets would have claimed 1%; rows 0 and 100 both "
            "carry tag 0, so 1.33% of the corpus qualifies"
        )

    def test_glove_s_real_row_count_is_not_a_multiple_of_the_buckets(self):
        """The claim in this class's docstring, made executable.

        glove is a `main`-profile dataset, so this is not a hypothetical row
        count: it is what a full run loads.
        """
        rows = int(ds.KNOWN_DATASETS["glove-100-angular"]["train"])
        assert rows % ds.TAG_BUCKETS != 0, (
            f"glove's {rows:,} rows are a multiple of {ds.TAG_BUCKETS} after "
            f"all; this test no longer says anything"
        )
        t, actual = ds.tag_threshold_for(ds.assign_tags(rows), 0.01)
        assert t == 1
        assert actual == pytest.approx(0.01000073, abs=5e-9)
        assert actual > 0.01

    def test_a_round_row_count_makes_the_two_agree(self):
        """Included to be explicit that it proves nothing on its own.

        This is the case every existing test uses, and it is why
        `threshold / 100.0` survived: on 1,000 rows and 100 buckets the claimed
        and the measured value are the same number.
        """
        tags = ds.assign_tags(1000, buckets=100)
        assert ds.tag_threshold_for(tags, 0.01, buckets=100) == (1, 0.01)

    def test_the_threshold_still_comes_from_selectivity_to_threshold(self):
        tags = ds.assign_tags(1000, buckets=100)
        for nominal in (0.001, 0.01, 0.1, 0.5, 2.0):
            t, _ = ds.tag_threshold_for(tags, nominal, buckets=100)
            assert t == ds.selectivity_to_threshold(nominal, 100)


class TestTheFivePercentPointIsItsOwnPoint:
    """`main` and `full` measure 1%, 5% and 10%. On `tag` 5% is five exact
    buckets; on `ctag` it is however many whole clusters reach 5%, and a lumpy
    partition could in principle land 5% on the same cluster count as 1% or
    10%. Then two nominal points would share a threshold, a ground-truth cache
    file and a measured share, and the curve would plot one measurement
    twice under two labels. This pins that at the shipped bucket and cluster
    counts the three requests are three different filters."""

    NOMINALS = (0.01, 0.05, 0.10)

    def test_tag_gives_three_distinct_thresholds(self):
        tags = ds.assign_tags(20_000)
        got = [ds.tag_threshold_for(tags, n) for n in self.NOMINALS]
        assert [g.threshold for g in got] == [1, 5, 10]
        assert [g.actual_selectivity for g in got] == [0.01, 0.05, 0.10]

    def test_ctag_gives_three_distinct_increasing_shares(self):
        ctags = ds.assign_ctags(_mixed_norm_corpus(20_000, 8, seed=23),
                                ds.EUCLIDEAN, n_clusters=ds.CTAG_CLUSTERS,
                                seed=ds.CTAG_SEED)
        got = [ds.ctag_threshold_for(ctags, n) for n in self.NOMINALS]
        thresholds = [t for t, _ in got]
        shares = [a for _, a in got]
        assert thresholds == sorted(set(thresholds)), got
        assert shares == sorted(set(shares)), got
        for nominal, share in zip(self.NOMINALS, shares):
            assert share >= nominal, (nominal, share)


class TestThresholdForDispatch:
    def test_tag_and_ctag_route_to_their_own_rule(self):
        tags = ds.assign_tags(1000, buckets=100)
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        assert (ds.threshold_for(ds.FILTER_TAG, tags, 0.10)
                == ds.tag_threshold_for(tags, 0.10))
        assert (ds.threshold_for(ds.FILTER_CTAG, ctags, 0.10)
                == ds.ctag_threshold_for(ctags, 0.10))

    def test_the_two_columns_disagree_at_the_same_nominal_selectivity(self):
        """The reason `column` has to be in the cache key.

        Same dataset, same k, same nominal 10%: `tag` delivers 10% of the rows
        and `ctag` delivers 12.5% of them, over a different set of rows
        entirely.
        """
        tags = ds.assign_tags(1000, buckets=100)
        ctags = _ctags_with_sizes(UNEQUAL_SIZES)
        tag_result = ds.threshold_for(ds.FILTER_TAG, tags, 0.10)
        ctag_result = ds.threshold_for(ds.FILTER_CTAG, ctags, 0.10)
        assert tag_result.actual_selectivity == 0.10
        assert ctag_result.actual_selectivity == 0.125

    def test_an_unknown_column_raises(self):
        with pytest.raises(ValueError):
            ds.threshold_for("category", ds.assign_tags(100), 0.1)

    def test_the_known_columns_are_exactly_tag_and_ctag(self):
        assert ds.FILTER_COLUMNS == (ds.FILTER_TAG, ds.FILTER_CTAG)
        assert (ds.FILTER_TAG, ds.FILTER_CTAG) == ("tag", "ctag")


# ---------------------------------------------------------------------------
# Masked ground truth
# ---------------------------------------------------------------------------

class TestMaskedGroundTruth:
    """§3.1 item 4: the brute force takes a mask and no longer decides for
    itself which rows qualify. Two filter columns with one exact yardstick,
    rather than two yardsticks that can drift apart.
    """

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("column", [ds.FILTER_TAG, ds.FILTER_CTAG])
    def test_matches_brute_force(self, column, metric):
        corpus = _mixed_norm_corpus(400, 5, seed=29)
        rng = numpy.random.default_rng(41)
        test = rng.normal(size=(4, 5)).astype("<f4")
        if column == ds.FILTER_TAG:
            values = ds.assign_tags(len(corpus), buckets=20)
            t, _ = ds.tag_threshold_for(values, 0.30, buckets=20)
        else:
            values = ds.assign_ctags(corpus, metric, n_clusters=20, seed=43)
            t, _ = ds.ctag_threshold_for(values, 0.30, n_clusters=20)
        mask = values < t
        truth = ds.masked_ground_truth(corpus, test, metric, 5, mask)
        assert truth.shape == (4, 5)
        for i, q in enumerate(test):
            assert set(truth[i].tolist()) == _brute_force(corpus, q, mask,
                                                          metric, 5)

    def test_only_masked_rows_are_returned(self):
        corpus = _mixed_norm_corpus(400, 5, seed=29)
        test = _mixed_norm_corpus(3, 5, seed=30)
        ctags = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=20, seed=43)
        mask = ctags < 4
        truth = ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 5, mask)
        # Shape first: `mask[truth].all()` is True for an empty `truth`, so on
        # its own this assertion is satisfied by a function that returns
        # nothing at all.
        assert truth.shape == (3, 5)
        assert mask[truth].all(), "returned a row that fails the filter"

    def test_results_are_ordered_nearest_first(self):
        corpus = _mixed_norm_corpus(200, 4, seed=31)
        test = _mixed_norm_corpus(2, 4, seed=32)
        mask = ds.assign_tags(200, buckets=10) < 5
        truth = ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 6, mask)
        # `numpy.diff` of a short row is empty and `.all()` of empty is True,
        # so the ordering claim needs the shape pinned before it means
        # anything.
        assert truth.shape == (2, 6)
        for i, q in enumerate(test):
            d = numpy.linalg.norm(
                corpus[truth[i]].astype(numpy.float64) - q.astype(numpy.float64),
                axis=1)
            assert len(d) == 6
            assert (numpy.diff(d) >= 0).all()

    def test_an_all_false_mask_raises(self):
        corpus = _mixed_norm_corpus(20, 3, seed=33)
        test = _mixed_norm_corpus(1, 3, seed=34)
        with pytest.raises(ValueError):
            ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 1,
                                   numpy.zeros(20, dtype=bool))

    def test_a_mask_of_the_wrong_length_raises(self):
        corpus = _mixed_norm_corpus(20, 3, seed=33)
        test = _mixed_norm_corpus(1, 3, seed=34)
        with pytest.raises(ValueError):
            ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 1,
                                   numpy.ones(19, dtype=bool))

    def test_an_index_array_is_not_accepted_as_a_mask(self):
        """Qualifying row *indices* in place of a mask.

        `numpy.nonzero([5, 9, 13])` is `[0, 1, 2]`, so this would score every
        engine against the first three rows of the corpus and still return a
        full-shaped, plausible answer. Caught here by the length check, since
        the qualifying count is rarely the row count.
        """
        corpus = _mixed_norm_corpus(20, 3, seed=33)
        test = _mixed_norm_corpus(1, 3, seed=34)
        with pytest.raises((ValueError, TypeError)):
            ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 1,
                                   numpy.array([5, 9, 13]))

    def test_the_column_itself_is_not_accepted_as_a_mask(self):
        """`mask=ctags` instead of `mask=ctags < t` — the slip with no symptom.

        The column has exactly one value per row, so the length check cannot
        see it, and `numpy.nonzero` counts every non-zero *label* as
        qualifying. The predicate silently becomes "not in cluster 0": a 32%
        filter measured as a 96% one, with a full-shaped answer and a
        believable recall. This is why the dtype is checked and not just the
        length — and the mutation that removes the dtype check is caught by
        this test and by nothing else.
        """
        corpus = _mixed_norm_corpus(200, 3, seed=33)
        test = _mixed_norm_corpus(1, 3, seed=34)
        ctags = ds.assign_ctags(corpus, ds.EUCLIDEAN, n_clusters=20, seed=43)
        assert len(ctags) == len(corpus), "the length check cannot catch this"
        assert ctags.dtype != numpy.dtype(bool)
        with pytest.raises(TypeError):
            ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 5, ctags)

    def test_k_is_truncated_to_the_qualifying_set(self):
        corpus = _mixed_norm_corpus(20, 3, seed=33)
        test = _mixed_norm_corpus(1, 3, seed=34)
        mask = numpy.zeros(20, dtype=bool)
        mask[:3] = True
        truth = ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 10, mask)
        assert truth.shape == (1, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 256])
    def test_the_query_batch_size_does_not_change_the_result(self, batch_size):
        corpus = _mixed_norm_corpus(200, 4, seed=35)
        test = _mixed_norm_corpus(5, 4, seed=36)
        mask = ds.assign_tags(200, buckets=10) < 5
        whole = ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 4, mask,
                                       batch_size=1000)
        got = ds.masked_ground_truth(corpus, test, ds.EUCLIDEAN, 4, mask,
                                     batch_size=batch_size)
        assert numpy.array_equal(got, whole)

    def test_the_tag_wrapper_delegates_to_the_mask_form(self):
        """`filtered_ground_truth` is now a wrapper that builds the tag mask.

        Every caller and test that predates PR 3 goes through it, so it has to
        agree with the mask form exactly rather than approximately.
        """
        corpus = _mixed_norm_corpus(300, 4, seed=37)
        test = _mixed_norm_corpus(3, 4, seed=38)
        wrapper = ds.filtered_ground_truth(corpus, test, ds.EUCLIDEAN, 5,
                                           tag_threshold=7, buckets=20)
        direct = ds.masked_ground_truth(
            corpus, test, ds.EUCLIDEAN, 5,
            ds.assign_tags(300, buckets=20) < 7)
        assert numpy.array_equal(wrapper, direct)


# ---------------------------------------------------------------------------
# Caching: the fairness surface
# ---------------------------------------------------------------------------

class TestCachedCtags:
    """§3.1 item 2, with its cost claim corrected by measurement.

    The spec expected "a few minutes of NumPy" at 1M x 960. It is 1.8 s there
    and 0.62 s at 1M x 100 — one BLAS matrix product per batch, not a per-row
    loop. The assignment is cached anyway, because it is read once per
    selectivity per engine and because the cache is where the fairness
    argument lives: a key that omits something which changes the answer scores
    a later run against another run's clusters. The ann results tree is keyed
    by a resource fingerprint for exactly that reason.
    """

    def test_the_cached_file_is_reused(self, tmp_path):
        """Proved by poisoning the file, not by timing.

        A second call that recomputed would return the real assignment and the
        poison would go unnoticed.
        """
        dataset = _dataset(_mixed_norm_corpus(200, 4, seed=45))
        first = ds.cached_ctags(str(tmp_path), dataset, n_clusters=8, seed=5)
        name = ds.ctag_cache_name(dataset.name, dataset.metric, 200, 8, 5)
        poison = numpy.full(200, 3, dtype=numpy.int64)
        with open(os.path.join(str(tmp_path), name), "wb") as fh:
            numpy.save(fh, poison)
        second = ds.cached_ctags(str(tmp_path), dataset, n_clusters=8, seed=5)
        assert numpy.array_equal(second, poison), (
            "the second call recomputed; the file on disk is not the cache"
        )
        assert not numpy.array_equal(first, poison)

    def test_the_name_under_test_is_the_file_written(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(200, 4, seed=45))
        ds.cached_ctags(str(tmp_path), dataset, n_clusters=8, seed=5)
        assert os.listdir(str(tmp_path)) == [
            ds.ctag_cache_name(dataset.name, dataset.metric, 200, 8, 5)
        ]

    def test_every_component_of_the_key_changes_the_name(self):
        base = dict(dataset_name="glove-100-angular", metric=ds.ANGULAR,
                    rows=1_000_000, n_clusters=100, seed=20261012)
        variants = {
            "baseline": {},
            "dataset": {"dataset_name": "sift-128-euclidean"},
            # The metric decides which rows are in which cluster, not just the
            # arithmetic: on the mixed-norm fixture the two metrics disagree on
            # 58% of rows. Unreachable only because every KNOWN_DATASETS name
            # ends in its own metric, which nothing enforces.
            "metric": {"metric": ds.EUCLIDEAN},
            "rows": {"rows": 500_000},
            "n_clusters": {"n_clusters": 50},
            "seed": {"seed": 20261013},
        }
        names = {
            label: ds.ctag_cache_name(**dict(base, **override))
            for label, override in variants.items()
        }
        assert len(set(names.values())) == len(variants), (
            f"two of these share a cache file: {names}"
        )

    def test_a_subset_gets_its_own_pivots(self, tmp_path):
        """Why `rows` is in the key rather than a convenience.

        Pivots are drawn from the rows that were loaded, so the clusters over
        the smoke profile's 20,000 rows are not the first 20,000 labels of the
        full assignment. Serving one for the other would score the run against
        a different partition of a different corpus.
        """
        dataset = _dataset(_mixed_norm_corpus(500, 4, seed=47))
        subset = ds.cached_ctags(str(tmp_path), dataset, indexed_rows=100,
                                 n_clusters=8, seed=5)
        whole = ds.cached_ctags(str(tmp_path), dataset, n_clusters=8, seed=5)
        assert len(subset) == 100
        assert len(whole) == 500
        assert not _is_a_relabelling_of(subset, whole[:100])

    def test_more_clusters_than_loaded_rows_is_refused(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(500, 4, seed=47))
        with pytest.raises(ValueError):
            ds.cached_ctags(str(tmp_path), dataset, indexed_rows=5,
                            n_clusters=8, seed=5)


class TestFilteredTruthCacheKey:
    """Everything that changes the correct answer set is in the key.

    `ops.pgvector_iterative_scan` was declared in three config files, plumbed
    into the SQL and rendered in a report column while never executing once.
    The cache-key equivalent is quieter: the run finishes, the recall numbers
    look normal, and they were computed against another configuration's
    targets.
    """

    BASE = dict(dataset_name="glove-100-angular", metric=ds.ANGULAR, k=10,
                rows=1_000_000, column="ctag", threshold=7, buckets=100,
                n_clusters=100, seed=20261012)

    @pytest.mark.parametrize("component,override", [
        ("dataset", {"dataset_name": "sift-128-euclidean"}),
        ("metric", {"metric": ds.EUCLIDEAN}),
        ("k", {"k": 100}),
        ("rows", {"rows": 500_000}),
        ("column", {"column": "tag"}),
        ("threshold", {"threshold": 8}),
        ("buckets", {"buckets": 50}),
        ("n_clusters", {"n_clusters": 50}),
        ("seed", {"seed": 20261013}),
    ])
    def test_changing_it_changes_the_name(self, component, override):
        base = ds.filtered_truth_cache_name(**self.BASE)
        changed = ds.filtered_truth_cache_name(**dict(self.BASE, **override))
        assert base != changed, f"{component} is missing from the cache key"

    def test_all_the_components_give_distinct_names(self):
        overrides = [
            {}, {"dataset_name": "sift-128-euclidean"},
            {"metric": ds.EUCLIDEAN}, {"k": 100},
            {"rows": 500_000}, {"column": "tag"}, {"threshold": 8},
            {"buckets": 50}, {"n_clusters": 50}, {"seed": 20261013},
        ]
        names = [ds.filtered_truth_cache_name(**dict(self.BASE, **o))
                 for o in overrides]
        assert len(set(names)) == len(names)

    def test_the_name_under_test_is_the_file_written(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(300, 4, seed=51))
        truth, filt = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                            ds.FILTER_TAG, 0.30, buckets=20)
        written = os.listdir(str(tmp_path))
        assert written == [ds.filtered_truth_cache_name(
            dataset.name, dataset.metric, 5, 300, ds.FILTER_TAG,
            filt.threshold, 20, ds.CTAG_CLUSTERS, ds.CTAG_SEED)]
        assert truth.shape[0] == dataset.n_test


    def test_one_name_and_two_metrics_are_two_answer_sets(self, tmp_path):
        """The metric collision, end to end, over both cache keys.

        `Dataset.metric` is read from the HDF5 file's `distance` attribute, not
        parsed out of the name. Every `KNOWN_DATASETS` name happens to end in
        its own metric, so this is unreachable with today's seven datasets —
        but that is a naming accident, and a locally generated corpus (the
        `generate` command makes them) or a renamed file breaks it. Without
        `metric` in the key the second caller is handed the first caller's
        partition and its ground truth, and reports the same selectivity for
        both.
        """
        corpus = _mixed_norm_corpus(600, 6, seed=71)
        euclid = _dataset(corpus, name="same-name", metric=ds.EUCLIDEAN)
        angular = _dataset(corpus, name="same-name", metric=ds.ANGULAR)
        cache = str(tmp_path)

        e_ctags = ds.cached_ctags(cache, euclid, n_clusters=20)
        a_ctags = ds.cached_ctags(cache, angular, n_clusters=20)
        assert len(os.listdir(cache)) == 2, (
            "one file for two metrics: the angular caller was served the "
            "euclidean partition"
        )
        assert not _is_a_relabelling_of(a_ctags, e_ctags)
        disagreeing = int((a_ctags != e_ctags).sum())
        assert disagreeing > len(corpus) // 4, (
            f"only {disagreeing} of {len(corpus)} labels differ between the "
            f"metrics; this fixture cannot see the collision"
        )

        e_truth, e_filt = ds.filtered_truth_for(cache, euclid, 5,
                                                ds.FILTER_CTAG, 0.30,
                                                n_clusters=20)
        a_truth, a_filt = ds.filtered_truth_for(cache, angular, 5,
                                                ds.FILTER_CTAG, 0.30,
                                                n_clusters=20)
        assert not numpy.array_equal(e_truth, a_truth)
        assert not numpy.array_equal(e_filt.mask, a_filt.mask)

    def test_the_same_threshold_on_both_columns_is_still_two_answer_sets(
            self, tmp_path):
        """The collision, exercised end to end in the production configuration.

        `TAG_BUCKETS` and `CTAG_CLUSTERS` are both 100, so in a real run every
        component of the key except `column` can coincide: the thresholds are
        small integers drawn from the same sweep, and `tag < 30` and
        `ctag < 30` are the same dataset, k, row count, bucket count, cluster
        count and seed. The qualifying sets are completely different — one is a
        uniform sample, the other is thirty whole clusters.

        The name-level tests above catch a key that drops `column`, but only as
        a string comparison. This one catches it as what it would actually do:
        serve the uncorrelated filter's answer set to the correlated filter's
        measurement, and report the resulting recall as the correlated one's.
        """
        dataset = _dataset(_mixed_norm_corpus(1000, 6, seed=53))
        cache = str(tmp_path)
        tag_filt = ds.resolve_filter(cache, dataset, ds.FILTER_TAG, 0.30)
        # Find the ctag selectivity that lands on the same threshold. It
        # exists for most thresholds; the test says so rather than assuming.
        matching = [s / 200.0 for s in range(2, 200)
                    if ds.resolve_filter(cache, dataset, ds.FILTER_CTAG,
                                         s / 200.0).threshold
                    == tag_filt.threshold]
        assert matching, "no ctag selectivity shares the tag threshold"

        tag_truth, _ = ds.filtered_truth_for(cache, dataset, 5,
                                             ds.FILTER_TAG, 0.30)
        ctag_truth, ctag_filt = ds.filtered_truth_for(
            cache, dataset, 5, ds.FILTER_CTAG, matching[0])
        assert ctag_filt.threshold == tag_filt.threshold
        assert not numpy.array_equal(tag_truth, ctag_truth), (
            "the ctag measurement was handed the tag column's answer set"
        )
        assert ctag_filt.actual_selectivity != tag_filt.actual_selectivity

    def test_the_two_columns_do_not_share_a_cache_entry(self, tmp_path):
        """The collision that would be hardest to notice.

        Same dataset, same k, same rows, same nominal selectivity — and a
        completely different qualifying set. Without `column` in the key the
        second measurement reads the first's file and reports the correlated
        filter's recall as the independent filter's.
        """
        dataset = _dataset(_mixed_norm_corpus(300, 6, seed=53))
        tag_truth, tag_filt = ds.filtered_truth_for(
            str(tmp_path), dataset, 5, ds.FILTER_TAG, 0.30, buckets=20)
        ctag_truth, ctag_filt = ds.filtered_truth_for(
            str(tmp_path), dataset, 5, ds.FILTER_CTAG, 0.30, n_clusters=20)
        assert not numpy.array_equal(tag_truth, ctag_truth)
        # two ground-truth files plus the cached ctag assignment
        assert len(os.listdir(str(tmp_path))) == 3
        assert tag_filt.column == ds.FILTER_TAG
        assert ctag_filt.column == ds.FILTER_CTAG


class TestFilteredTruthFor:
    """The truth and the filter that produced it come back from one call.

    The caller cannot record the nominal selectivity by accident because it
    never has to reconstruct the actual one: `filt.actual_selectivity` arrives
    alongside the array it belongs to.
    """

    @pytest.mark.parametrize("metric", [ds.EUCLIDEAN, ds.ANGULAR])
    @pytest.mark.parametrize("column", [ds.FILTER_TAG, ds.FILTER_CTAG])
    def test_the_truth_matches_brute_force(self, tmp_path, column, metric):
        dataset = _dataset(_mixed_norm_corpus(400, 5, seed=55), metric=metric)
        truth, filt = ds.filtered_truth_for(
            str(tmp_path), dataset, 5, column, 0.30, buckets=20, n_clusters=20)
        for i, q in enumerate(dataset.test):
            assert set(truth[i].tolist()) == _brute_force(
                dataset.train, q, filt.mask, metric, 5)

    def test_the_mask_is_the_column_below_the_threshold(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(400, 5, seed=55))
        _, filt = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                        ds.FILTER_CTAG, 0.30, n_clusters=20)
        ctags = ds.cached_ctags(str(tmp_path), dataset, n_clusters=20,
                                seed=ds.CTAG_SEED)
        assert numpy.array_equal(filt.mask, ctags < filt.threshold)
        assert filt.mask.dtype == bool

    def test_the_actual_selectivity_is_the_measured_share(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(400, 5, seed=55))
        _, filt = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                        ds.FILTER_CTAG, 0.10, n_clusters=20)
        assert filt.actual_selectivity == pytest.approx(
            int(filt.mask.sum()) / 400)
        assert filt.nominal_selectivity == 0.10

    def test_the_record_fields_keep_the_two_apart(self, tmp_path):
        """`as_extra()` is what lands in the record's `extra` column.

        Both numbers are there and both are named, so a reader of an archived
        record can see what was asked for and what was measured. Four times in
        this project a claimed value arrived indistinguishable from a measured
        one; one of them is still null in 69 of 69 archived records.
        """
        dataset = _dataset(_mixed_norm_corpus(400, 5, seed=55))
        _, filt = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                        ds.FILTER_CTAG, 0.10, n_clusters=20)
        extra = filt.as_extra()
        assert extra["filter_column"] == "ctag"
        assert extra["nominal_selectivity"] == 0.10
        assert extra["actual_selectivity"] == filt.actual_selectivity
        assert extra["actual_selectivity"] != extra["nominal_selectivity"]
        assert extra["ctag_clusters"] == 20
        assert extra["ctag_seed"] == ds.CTAG_SEED
        assert extra["filter_threshold"] == filt.threshold

    def test_the_tag_filter_carries_no_cluster_provenance(self, tmp_path):
        """`tag` has no pivots, so its record must not claim any.

        Reporting `ctag_clusters: 100` on a `tag` measurement would be a
        plausible-looking value that describes nothing the run did.
        """
        dataset = _dataset(_mixed_norm_corpus(400, 5, seed=55))
        _, filt = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                        ds.FILTER_TAG, 0.10, buckets=20)
        extra = filt.as_extra()
        assert extra["filter_column"] == "tag"
        assert extra["ctag_clusters"] is None
        assert extra["ctag_seed"] is None
        assert extra["tag_buckets"] == 20

    def test_a_subset_corpus_is_never_scored_against_unloaded_rows(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(500, 4, seed=57))
        truth, filt = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                            ds.FILTER_CTAG, 0.50,
                                            indexed_rows=100, n_clusters=10)
        assert truth.max() < 100
        assert len(filt.mask) == 100

    def test_the_cached_second_call_returns_the_same_filter(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(400, 5, seed=55))
        first_truth, first = ds.filtered_truth_for(
            str(tmp_path), dataset, 5, ds.FILTER_CTAG, 0.10, n_clusters=20)
        second_truth, second = ds.filtered_truth_for(
            str(tmp_path), dataset, 5, ds.FILTER_CTAG, 0.10, n_clusters=20)
        assert numpy.array_equal(first_truth, second_truth)
        assert first.threshold == second.threshold
        assert first.actual_selectivity == second.actual_selectivity

    def test_an_unknown_column_raises(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(100, 4, seed=59))
        with pytest.raises(ValueError):
            ds.filtered_truth_for(str(tmp_path), dataset, 5, "category", 0.1)


class TestLegacyWrappersStillWork:
    """PR 3 must not cost the pre-PR-3 callers anything.

    `harness/workloads/build.py` and `harness/workloads/filtered.py` are owned
    by other tasks in this PR and still call the tag-only entry points.
    """

    def test_cached_filtered_ground_truth_returns_the_array_alone(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(300, 4, seed=61))
        truth = ds.cached_filtered_ground_truth(str(tmp_path), dataset, 5, 0.30,
                                                buckets=20)
        assert isinstance(truth, numpy.ndarray)
        # Same trap as above: an empty array passes `(... < 6).all()`.
        assert truth.shape == (dataset.n_test, 5)
        tags = ds.assign_tags(300, buckets=20)
        assert (tags[truth] < 6).all()

    def test_it_agrees_with_the_column_aware_form(self, tmp_path):
        dataset = _dataset(_mixed_norm_corpus(300, 4, seed=61))
        legacy = ds.cached_filtered_ground_truth(str(tmp_path), dataset, 5,
                                                 0.30, buckets=20)
        current, _ = ds.filtered_truth_for(str(tmp_path), dataset, 5,
                                           ds.FILTER_TAG, 0.30, buckets=20)
        assert numpy.array_equal(legacy, current)

    def test_cached_ground_truth_over_a_subset_is_unfiltered(self, tmp_path):
        """`cached_ground_truth` used to express "every row" as `tag < buckets`.

        Under the mask refactor it passes an all-true mask over the loaded
        rows instead. The answer must not move.
        """
        dataset = _dataset(_mixed_norm_corpus(300, 4, seed=63))
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5,
                                       indexed_rows=100)
        assert truth.max() < 100
        loaded = numpy.zeros(300, dtype=bool)
        loaded[:100] = True
        for i, q in enumerate(dataset.test):
            assert set(truth[i].tolist()) == _brute_force(
                dataset.train, q, loaded, ds.EUCLIDEAN, 5)


# ---------------------------------------------------------------------------
# The recorded selectivity, enforced at the source level
# ---------------------------------------------------------------------------

def _record_selectivity_sources():
    """Every `selectivity=` argument handed to `Record(...)` in a workload.

    Returns (path, lineno, source-of-the-expression) triples. AST, not a
    substring search: `"actual_selectivity" in src` is satisfied by the
    variable name `actual_selectivity` holding `threshold / 100.0`, which is
    precisely the defect this guard exists to find. Six substring assertions
    in this repo have been satisfied by a longer string.
    """
    found = []
    pattern = os.path.join(VB_ROOT, "harness", "workloads", "*.py")
    for path in sorted(glob.glob(pattern)):
        source = open(path).read()
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if name != "Record":
                continue
            for keyword in node.keywords:
                if keyword.arg == "selectivity":
                    found.append((os.path.relpath(path, VB_ROOT),
                                  keyword.value.lineno,
                                  ast.unparse(keyword.value)))
    return found


class TestTheRecordedSelectivityIsTheMeasuredOne:
    """A source-level gate, in the shape `test_harness_does_not_import_the
    orchestrator` already uses, because the API guard rails are not the only
    door into the record.

    `FilterThreshold` has no nominal field, `ResolvedFilter` names both, and
    `filtered_truth_for` hands back the measured share with the array it
    belongs to. All true, and none of it helps: `cached_filtered_ground_truth`
    returns an ndarray alone and now accepts `column=`, so the smallest edit
    that adds the correlated filter to the existing loop — one keyword on the
    existing call — keeps `threshold / 100.0` and records a nominal share.

    For `ctag` the nominal value is not even a wrong row share. It is the
    fraction of *clusters*, and the row share it stands in for was measured at
    1.00%-4.96% for a nominal 1% depending only on the seed. `selectivity` is
    a float in a flat schema, so nothing downstream can tell that apart from a
    measurement.

    §3.1 item 4 requires `column` on `cached_filtered_ground_truth`, so the
    answer is not to reshape that function's return. The answer is to make the
    careless call site fail a test.
    """

    def test_the_workloads_write_a_measured_share_or_none_at_all(self):
        sources = _record_selectivity_sources()
        assert sources, (
            "no workload passes selectivity= to Record(); this guard has "
            "stopped guarding anything"
        )
        offenders = [(path, line, expr) for path, line, expr in sources
                     if not expr.endswith(".actual_selectivity")]
        assert not offenders, "\n".join(
            f"{path}:{line} records selectivity from {expr!r}, which is not a "
            f"measured share — use filtered_truth_for() and record "
            f"filt.actual_selectivity"
            for path, line, expr in offenders
        )

    def test_the_guard_reads_the_expression_and_not_the_variable_name(self):
        """The guard must reject `selectivity=actual_selectivity`.

        A variable can be named anything. `harness/workloads/filtered.py`
        assigns `actual_selectivity = threshold / 100.0` at :83 and passes it
        to `Record` at :150, so a check that accepted a bare name whose
        spelling happened to contain "actual_selectivity" would pass on the
        defect itself. Line numbers go stale; the expression does not, which
        is why the guard reports `ast.unparse` of what it found.
        """
        tree = ast.parse("Record(selectivity=actual_selectivity)")
        call = tree.body[0].value
        expression = ast.unparse(call.keywords[0].value)
        assert expression == "actual_selectivity"
        assert not expression.endswith(".actual_selectivity"), (
            "the guard's test would accept a bare variable name"
        )
        # And the shape it must accept.
        ok = ast.unparse(
            ast.parse("Record(selectivity=filt.actual_selectivity)")
            .body[0].value.keywords[0].value)
        assert ok.endswith(".actual_selectivity")

    def test_filtered_truth_for_is_what_a_workload_should_call(self):
        """The fixed call site has somewhere to get the measured share from."""
        import inspect
        signature = inspect.signature(ds.filtered_truth_for)
        assert "column" in signature.parameters
        annotation = str(signature.return_annotation)
        assert "ResolvedFilter" in annotation and "ndarray" in annotation
        assert hasattr(ds.ResolvedFilter, "as_extra")
