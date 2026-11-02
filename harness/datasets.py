"""Dataset loading and ground-truth computation for the ops harness.

Reads the same HDF5 files ann-benchmarks uses, so both measurement paths see
identical data and identical ground truth. The one thing computed here that
ann-benchmarks does not provide is **filtered ground truth**: applying a WHERE
predicate changes the correct answer set, so reusing the unfiltered neighbours
would silently score every engine against the wrong targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy

# Distance metrics as ann-benchmarks names them in the HDF5 `distance` attribute.
ANGULAR = "angular"
EUCLIDEAN = "euclidean"

# Datasets this framework is set up for, with the shape facts needed to plan a
# run before the file is downloaded.
KNOWN_DATASETS: Dict[str, Dict[str, object]] = {
    "fashion-mnist-784-euclidean": {
        "dim": 784, "train": 60_000, "test": 10_000,
        "metric": EUCLIDEAN, "approx_bytes": 217 * 1024**2,
        "role": "smoke",
    },
    "glove-100-angular": {
        "dim": 100, "train": 1_183_514, "test": 10_000,
        "metric": ANGULAR, "approx_bytes": 463 * 1024**2,
        "role": "main",
    },
    "sift-128-euclidean": {
        "dim": 128, "train": 1_000_000, "test": 10_000,
        "metric": EUCLIDEAN, "approx_bytes": 501 * 1024**2,
        "role": "main",
    },
    "gist-960-euclidean": {
        "dim": 960, "train": 1_000_000, "test": 1_000,
        "metric": EUCLIDEAN, "approx_bytes": 3_600 * 1024**2,
        "role": "stress",
    },
    # Generated locally by scripts/generate-dataset.sh, not downloadable. The
    # corpus MariaDB used for its big-vector benchmark. Row counts are the
    # generator's actual split (10,000 of the million are held out as queries),
    # not the round number in the name.
    "dbpedia-openai-100k-angular": {
        "dim": 1536, "train": 90_000, "test": 10_000,
        "metric": ANGULAR, "approx_bytes": 590 * 1024**2,
        "role": "stress",
    },
    "dbpedia-openai-500k-angular": {
        "dim": 1536, "train": 490_000, "test": 10_000,
        "metric": ANGULAR, "approx_bytes": 3_070 * 1024**2,
        "role": "stress",
    },
    "dbpedia-openai-1000k-angular": {
        "dim": 1536, "train": 990_000, "test": 10_000,
        "metric": ANGULAR, "approx_bytes": 6_170 * 1024**2,
        "role": "stress",
    },
}


def resident_bytes_estimate(dataset: str, subset_rows: Optional[int] = None) -> int:
    """Roughly what an in-memory engine needs to hold this corpus.

    Vectors at four bytes a dimension, plus an allowance for the HNSW graph and
    the store's per-key overhead. Only in-memory engines care: a disk-backed
    engine given too little memory gets slower, while Valkey refuses writes
    partway through the load, so it is worth saying before the run rather than
    after.

    Deliberately generous. A false warning costs a sentence; a missed one costs
    the run. Returns 0 for an unknown dataset rather than guessing, because a
    warning derived from a number we made up is worse than none.
    """
    facts = KNOWN_DATASETS.get(dataset)
    if not facts:
        return 0
    rows = int(facts.get("train") or 0)
    if subset_rows:
        rows = min(rows, int(subset_rows))
    dim = int(facts.get("dim") or 0)
    if not rows or not dim:
        return 0
    vectors = rows * dim * 4
    # The graph and per-key overhead together are the same order as the
    # vectors at these dimensionalities, so doubling is the safe estimate.
    return int(vectors * 2.2)


@dataclass
class Dataset:
    name: str
    train: numpy.ndarray
    test: numpy.ndarray
    neighbors: numpy.ndarray     # ground truth ids, shape (n_test, 100)
    distances: Optional[numpy.ndarray]
    metric: str

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test.shape[0])


def dataset_path(datasets_dir: str, name: str) -> str:
    return os.path.join(datasets_dir, f"{name}.hdf5")


def load(datasets_dir: str, name: str, max_test: Optional[int] = None) -> Dataset:
    import h5py  # imported lazily: only the client container has h5py

    path = dataset_path(datasets_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            f"run: ./run-benchmark.sh fetch --datasets {name}"
        )

    with h5py.File(path, "r") as fh:
        train = numpy.asarray(fh["train"], dtype="<f4")
        test = numpy.asarray(fh["test"], dtype="<f4")
        neighbors = numpy.asarray(fh["neighbors"], dtype=numpy.int64)
        distances = numpy.asarray(fh["distances"]) if "distances" in fh else None
        metric = fh.attrs.get("distance", KNOWN_DATASETS.get(name, {}).get("metric", EUCLIDEAN))
        if isinstance(metric, bytes):
            metric = metric.decode()

    if max_test is not None and max_test < len(test):
        test = test[:max_test]
        neighbors = neighbors[:max_test]
        if distances is not None:
            distances = distances[:max_test]

    return Dataset(
        name=name, train=train, test=test, neighbors=neighbors,
        distances=distances, metric=str(metric),
    )


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def recall_per_query(returned: Sequence[Sequence[int]],
                     truth: numpy.ndarray, k: int) -> List[float]:
    """PR 5 §5.1 item 2: the per-query recalls, not only their mean.

    One entry per result set, so `len(...) == len(returned)` always and the
    list can be bootstrapped directly: an interval over a vector shorter than
    the queries it claims to describe is an interval over a different
    measurement.

    That invariant is why two entries here are `0.0` rather than absent, and
    both come straight out of the mean this has to keep reproducing:

    * a query whose ground truth is EMPTY. `recall_at_k` skipped its
      numerator (`if not wanted: continue`) and still divided by every result
      set, so the published mean already charges that query a zero. Measured
      on a one-query fixture with a zero-width truth row: 0.0, not nan.
    * a result set with NO ground truth row at all, which `zip` truncates
      away while the denominator keeps counting it. Measured on two result
      sets against one truth row, the first matching exactly: 0.5, not 1.0.

    Neither is a recall of zero in any honest sense -- they are both "this
    query was never scored" -- and an interval built from them is slightly
    pessimistic. Changing that would move a figure the report has published
    for its whole life, which §5.1 item 2 explicitly forbids ("the mean stays
    the reported value"), so the shape is preserved and named instead of
    quietly improved. A later task that wants to fix the mean has to fix it
    here, in one place, and both the point and the interval move together.
    """
    out: List[float] = []
    for index, got in enumerate(returned):
        if index >= len(truth):
            # `zip` in the original dropped these; the denominator did not.
            out.append(0.0)
            continue
        wanted = set(int(x) for x in truth[index][:k])
        if not wanted:
            out.append(0.0)
            continue
        out.append(
            len(wanted.intersection(int(x) for x in got[:k])) / len(wanted))
    return out


def recall_at_k(returned: Sequence[Sequence[int]],
                truth: numpy.ndarray, k: int) -> float:
    """Mean fraction of the true top-k IDS that each result set contains.

    **This is NOT ann-benchmarks' definition, and this docstring said it was
    until ops measurement version 9.** ann-benchmarks counts a returned
    neighbour as correct when its DISTANCE is within 1e-3 of the true k-th
    distance (`report/loaders.py:knn_recall`); this counts it only when its
    ID is one of the true k. The two agree on a corpus with no ties at the
    k-th distance and part where there are: a returned row at exactly the
    k-th distance under another id scores 1 there and 0 here. Since version 9
    the ops workloads publish `recall_threshold` below as `recall_at_k` and
    keep this one as `extra.recall_id_overlap`, so one name means one rule;
    every ops record says which in `extra.recall_kind`.

    Derived from `recall_per_query` rather than computed beside it. The two
    were a loop each when §5.1 item 2 added the sibling, and "two paths that
    agree today" is the shape this repository keeps paying for: the mean the
    report publishes and the samples its error bar is drawn from have to be
    the same arithmetic or the bar can sit beside a point it does not contain.

    **The published figure moves in the last bits, and an earlier revision of
    this docstring was wrong to claim otherwise.** It said `sum()` "adds left
    to right exactly as the previous loop did". It does not: Python 3.12 gave
    `sum()` Neumaier compensated summation, so `sum([1/3]*3 + [1/7]*7)` is
    2.0 where the naive loop gives 1.9999999999999996. Against a verbatim
    copy of the pre-PR-5 loop over 20,000 seeded fixtures, the raw mean
    differs in 6,132 of them, by at most 6 ULP.

    What does NOT move is the figure anything downstream reads. Every caller
    stores `round(recall, 6)` (`calibrate.py`, `filtered.py`, `churn.py`),
    and over those same 20,000 fixtures that six-decimal value differed in
    **1**, on an exact rounding tie: the base loop produced
    0.57031250000000011 and rounds to 0.570313, while this produces
    0.5703125 -- the tie itself -- and rounds half-to-even to 0.570312. The
    four decimals the report renders from the stored value differed in **0**
    of 20,000, and §5.2's tolerance floor is 0.01, thirteen orders of
    magnitude above the divergence.

    So no ops measurement version bump: what an ops number MEANS is
    unchanged, and the stored figure moves about once in twenty thousand
    points by one unit in a decimal the report does not print. The compensated
    sum is also the more accurate of the two, which is why the direction of
    the change is not the question -- its size is.
    `tests/test_t5b_stats.py` holds the base loop verbatim and asserts those
    three counts, so this paragraph cannot drift from the arithmetic.
    """
    if len(returned) == 0:
        return float("nan")
    return sum(recall_per_query(returned, truth, k)) / len(returned)


#: `extra.recall_kind` values. Every ops record that carries a recall says
#: which rule produced it; `report/` reads an ops record from before ops
#: version 9 as `RECALL_ID_OVERLAP`, which is what those workloads computed.
RECALL_DISTANCE_THRESHOLD = "distance_threshold"
RECALL_ID_OVERLAP = "id_overlap"

#: ann-benchmarks' `knn_threshold` epsilon (`plotting/metrics.py`), and
#: `report/loaders.py:knn_recall`'s default. One constant, not a second 1e-3.
RECALL_EPSILON = 1e-3


def _distances(query: numpy.ndarray, rows: numpy.ndarray,
               metric: str) -> numpy.ndarray:
    """ann-benchmarks' distance from `query` to each of `rows`.

    The two metrics this repository runs, spelled as
    `ann_benchmarks/distance.py` spells them -- euclidean is the L2 norm of
    the difference (not its square), angular is `1 - cos` -- because the
    threshold below is compared against distances in THAT unit, and 1e-3 of
    a squared distance is a different tolerance. In float64: the stored
    vectors are float32, and doing the subtraction at the higher precision
    cannot move a distance by anything near the epsilon.
    """
    q = numpy.asarray(query, dtype=numpy.float64)
    r = numpy.asarray(rows, dtype=numpy.float64)
    if len(r) == 0:
        return numpy.zeros(0)
    if metric == ANGULAR:
        norms = numpy.linalg.norm(r, axis=1) * numpy.linalg.norm(q)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            out = 1.0 - (r @ q) / norms
        return numpy.where(norms > 0, out, numpy.inf)
    return numpy.linalg.norm(r - q, axis=1)


def recall_threshold_per_query(returned: Sequence[Sequence[int]],
                               truth: numpy.ndarray,
                               queries: numpy.ndarray,
                               corpus: numpy.ndarray, metric: str, k: int,
                               row_of: Optional[Dict[int, int]] = None
                               ) -> List[float]:
    """Per-query recall by ann-benchmarks' distance-threshold rule.

    Exactly `report/loaders.py:knn_recall`'s rule, per query: a returned
    neighbour counts when its distance to the query is at most the k-th true
    distance plus `RECALL_EPSILON`, over the first k returned, divided by k.
    So a short answer scores what it found over k, a duplicated id counts
    twice (ann-benchmarks' loop does the same), and a row at exactly the k-th
    distance under a DIFFERENT id counts -- the one place this and
    `recall_at_k`'s id overlap disagree.

    WHY RECOMPUTED FROM THE CORPUS: no driver returns distances
    (`EngineDriver.query` returns ids), and ann-benchmarks itself recomputes
    the run distances from `X_train` (`runner.run_individual_query`) rather
    than trusting the engine's. The true k-th distance is recomputed from the
    truth ids the same way, because the subset and filtered ground truths
    this harness computes are ids only; for a full-corpus run it equals the
    HDF5 `distances` column to float precision.

    `row_of` maps an id the engine holds to the corpus row whose vector it
    has -- churn re-inserts rows under fresh ids. An id with no row (absent
    from the map and outside the corpus, or negative) is at infinite
    distance: an answer that names nothing in the corpus is not a neighbour.

    `k` is the denominator, as in `knn_recall`, except where a truth row is
    narrower than k: a filtered predicate that qualifies fewer than k rows
    has only that many correct answers, so the denominator is the truth
    width and the threshold is its last distance.
    """
    n_corpus = len(corpus)
    out: List[float] = []

    def row(ident: Any) -> int:
        ident = int(ident)
        if row_of is not None and ident in row_of:
            return int(row_of[ident])
        return ident

    for index, got in enumerate(returned):
        if index >= len(truth) or len(truth[index]) == 0:
            out.append(0.0)
            continue
        wanted = [row(t) for t in list(truth[index])[:k]]
        count = len(wanted)
        query = queries[index]
        kth = _distances(query, corpus[[wanted[-1]]], metric)[0]
        threshold = kth + RECALL_EPSILON
        rows = [row(g) for g in list(got)[:count]]
        valid = [r for r in rows if 0 <= r < n_corpus]
        hits = 0
        if valid:
            dist = _distances(query, corpus[valid], metric)
            hits = int(numpy.count_nonzero(dist <= threshold))
        out.append(hits / count)
    return out


def recall_threshold(returned: Sequence[Sequence[int]],
                     truth: numpy.ndarray, queries: numpy.ndarray,
                     corpus: numpy.ndarray, metric: str, k: int,
                     row_of: Optional[Dict[int, int]] = None) -> float:
    """Mean of `recall_threshold_per_query`; NaN for no result sets, as
    `recall_at_k` and `knn_recall` both return."""
    if len(returned) == 0:
        return float("nan")
    per = recall_threshold_per_query(returned, truth, queries, corpus,
                                     metric, k, row_of=row_of)
    return sum(per) / len(returned)


# ---------------------------------------------------------------------------
# Filter columns. The difference between them is the point of the measurement:
# `tag = row_id % TAG_BUCKETS` is independent of the vector space, so the rows
# passing `tag < t` are a uniform sample and a query's true neighbours pass as
# often as anything else; `ctag` is a nearest-pivot cluster id, so the rows
# passing `ctag < t` are t whole clusters and for most queries the real
# nearest neighbours sit in a cluster that fails. That second case is where
# pre-filtering, post-filtering and index iteration diverge, and what one
# uncorrelated column cannot measure. Both load in one build (PR 3 option A);
# choosing one column's values per run needs no driver change but a full
# reload per mode, which is hours for MHNSW and VIDX at 1M rows.
# ---------------------------------------------------------------------------

FILTER_TAG = "tag"
FILTER_CTAG = "ctag"
FILTER_COLUMNS = (FILTER_TAG, FILTER_CTAG)

TAG_BUCKETS = 100
CTAG_CLUSTERS = 100

# Three things in `assign_ctags` must come out the same on a rerun — the pivot
# draw, the assignment and the renumbering — and this one seed pins all three
# (two are draws from this generator; the assignment follows from the pivots).
# They share a seed because they are one choice from the measurement's point
# of view, "which clustering was this run scored against?", and a run that
# cannot name that in one number cannot be reproduced from its manifest. If a
# later task needs them to vary independently it needs a new constant and a
# new cache-key component, not a second seed derived from this one.
CTAG_SEED = 20261012

# Rows per batch when assigning clusters. The full row-by-pivot cost matrix at
# 1M rows and 100 pivots is 400 MB in float32 and 800 MB in float64, in a
# container that is already holding the corpus; 8,192 rows keeps it at 3 MB.
CTAG_BATCH_ROWS = 8192


def assign_tags(n_rows: int, buckets: int = TAG_BUCKETS) -> numpy.ndarray:
    """Deterministic scalar attribute used by the filtered-search workload.

    `tag = row_id % buckets` gives an almost exactly known selectivity for
    `tag < t`: t/buckets qualify, exactly when the row count is a multiple of
    `buckets` and off by at most one row per bucket otherwise — measured,
    1.3333% rather than 1% on a 150-row corpus and 1.000073% on glove's
    1,183,514. "Almost" is why `tag_threshold_for` counts instead of dividing.
    Deterministic, so the selectivity is a property of the query and not of a
    draw, and every engine sees the same qualifying set.

    Uniform, crucially, *with respect to the vector space*: the qualifying
    rows spread evenly through it, which makes this the control case and
    `ctag` the treatment.
    """
    return numpy.arange(n_rows, dtype=numpy.int64) % buckets


def assign_ctags(train: numpy.ndarray, metric: str,
                 n_clusters: int = CTAG_CLUSTERS, seed: int = CTAG_SEED,
                 batch_rows: int = CTAG_BATCH_ROWS) -> numpy.ndarray:
    """Cluster id per row, so that a `ctag < t` filter follows the vector space.

    `n_clusters` rows are drawn as pivots and every row is labelled with its
    nearest pivot under the dataset's own metric, giving a Voronoi partition:
    neighbouring rows share a label, which is the correlation `tag` lacks.
    Three things here are deliberate and each is load-bearing.

    **The metric is the dataset's.** Assigning by Euclidean distance on an
    angular dataset gives clusters following vector *length* rather than
    direction, which is not where that dataset's neighbours are; angular is
    normalised first, as `masked_ground_truth` does it. **The cost is
    batched** (see `CTAG_BATCH_ROWS`) — a memory decision that must not change
    the labels, which is why the batch size is a parameter tested across a
    ragged tail.

    **The labels are renumbered by a seeded permutation.** Without it the
    cluster ids are pivot-draw order, so `ctag < t` takes whichever clusters
    were drawn first. Measured, that is harmless *while the draw is random*:
    over twelve seeds the selected rows' spatial spread is 1.05 of the
    corpus's with the permutation and 0.98 without. It earns its place against
    the draw ceasing to be random — the plausible "optimisation" of sorting
    the pivot indices for cache-friendly indexing. Holding the draw sorted the
    spread is 0.95 with it and 0.34 without, because `ctag < t` then takes the
    clusters nearest the start of the file, which for any corpus whose storage
    order tracks position in the space is one spatial region. Filtered recall
    over one region is not filtered recall, and the fix is one call.
    """
    n_rows = int(len(train))
    if metric not in (ANGULAR, EUCLIDEAN):
        raise ValueError(f"unknown metric for cluster assignment: {metric!r}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
    if n_rows < n_clusters:
        raise ValueError(
            f"cannot draw {n_clusters} distinct pivots from {n_rows} rows"
        )
    if batch_rows < 1:
        raise ValueError(f"batch_rows must be at least 1, got {batch_rows}")

    rng = numpy.random.default_rng(seed)
    pivot_rows = rng.choice(n_rows, size=n_clusters, replace=False)
    pivots = numpy.asarray(train[pivot_rows], dtype="<f4")
    if metric == ANGULAR:
        pivots = _normalize(pivots)
    pivot_sq = (pivots ** 2).sum(axis=1)[None, :]

    raw = numpy.empty(n_rows, dtype=numpy.int64)
    for start in range(0, n_rows, batch_rows):
        batch = numpy.asarray(train[start:start + batch_rows], dtype="<f4")
        if metric == ANGULAR:
            # Cosine distance ordering is the reverse of dot-product ordering
            # once both sides are unit-normalized.
            #
            # Normalising the *batch* cannot change a label: it scales one
            # row's whole cost vector by 1/||a|| > 0 and argmin is invariant
            # under that — measured, 0 of 600, 0 of 2,000 and 0 of 5,000
            # labels move when it is removed. It stays for two reasons that
            # are not about the labels: it bounds `cost` to [-1, 1], which is
            # what lets the near-tie analysis in tests/test_datasets.py use a
            # constant noise scale of 2 for angular (unnormalised, |cost|
            # reaches 23.5 on the same fixture), and it keeps this path
            # identical to `masked_ground_truth`'s, whose cached output must
            # agree with these labels row for row. It costs the measured
            # 1.8 s -> 3.2 s at 1M x 960, so a later task may drop it —
            # knowingly, and fixing that noise scale with it.
            cost = -(_normalize(batch) @ pivots.T)
        else:
            # ||a-b||^2 expanded. The ||a||^2 term is constant across the
            # pivots of one row and so cannot change which pivot is nearest,
            # but it is kept because it makes `cost` an actual squared
            # distance and keeps the algebra the same as
            # `masked_ground_truth`'s, where the per-query term is likewise
            # redundant and likewise kept.
            cost = (
                (batch ** 2).sum(axis=1, keepdims=True)
                - 2.0 * (batch @ pivots.T)
                + pivot_sq
            )
        raw[start:start + len(batch)] = numpy.argmin(cost, axis=1)

    # Drawn from the same generator, so one seed still reproduces everything.
    renumber = rng.permutation(n_clusters)
    return renumber[raw]


def ctag_cache_name(dataset_name: str, metric: str, rows: int,
                    n_clusters: int, seed: int) -> str:
    """Cache file name for one cluster assignment.

    Everything that changes the answer is in the name: the corpus, the metric,
    how much was loaded (pivots come from the loaded rows, so the clusters
    over a 20,000-row subset are not the first 20,000 labels of the full
    assignment), the cluster count and the seed.

    `metric` changes the partition, not just the arithmetic — the two metrics
    disagree on 58% of rows on the mixed-norm fixture — and is unreachable
    today only because every `KNOWN_DATASETS` name ends in its own metric,
    which nothing enforces: `Dataset.metric` comes from the HDF5 attribute.
    """
    return f"{dataset_name}.{metric}.ctag.c{n_clusters}.s{seed}.n{rows}.npy"


def cached_ctags(cache_dir: str, dataset: Dataset,
                 indexed_rows: Optional[int] = None,
                 n_clusters: int = CTAG_CLUSTERS,
                 seed: int = CTAG_SEED) -> numpy.ndarray:
    """Compute the cluster assignment once per (dataset, rows, clusters, seed).

    Measured, not estimated: one BLAS GEMM of rows x clusters x dim per batch,
    at 20-120 GFLOP/s here. Euclidean 0.62 s at 1M rows x 100 dims (timed,
    best of three) and 1.8 s at 1M x 960 (extrapolated from timed 100k and
    200k runs at that width); angular 0.51 s and 3.2 s — *faster* at 100 dims
    and *slower* at 960, because angular adds a normalisation pass scaling
    with `dim` while euclidean adds two squared-norm reductions scaling with
    the row count.

    Worth stating plainly, because the PR 3 spec assumed "a few minutes of
    NumPy at 1M x 960" and a later task has to put a number in the run-time
    estimate. Against a 1M-row MHNSW ingest at ~150 rows/s (~1.9 h) this is
    rounding error; it is cached to avoid repeating it per selectivity and per
    engine, not because it is expensive. The same array is what the driver
    loads into the `ctag` column, so engine and ground truth cannot disagree
    about which cluster a row is in.
    """
    n_train = len(dataset.train)
    rows = n_train if indexed_rows is None else min(int(indexed_rows), n_train)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir,
                        ctag_cache_name(dataset.name, dataset.metric, rows,
                                        n_clusters, seed))
    if os.path.exists(path):
        return numpy.load(path)
    ctags = assign_ctags(dataset.train[:rows], dataset.metric,
                         n_clusters=n_clusters, seed=seed)
    # numpy.save() appends ".npy" to a *path* that lacks it; passing an open
    # file object suppresses that and keeps the atomic rename intact.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        numpy.save(fh, ctags)
    os.replace(tmp, path)
    return ctags


# ---------------------------------------------------------------------------
# Thresholds: the nominal selectivity asked for, and the one delivered
# ---------------------------------------------------------------------------

class FilterThreshold(NamedTuple):
    """`(threshold, actual_selectivity)` — the share measured, never the asked.

    Cluster sizes are unequal, so a 1% request is never 1% delivered. Measured
    on a corpus named so the figures can be re-derived rather than trusted —

        numpy.random.default_rng(11).normal(size=(100_000, 100)).astype("<f4")

    with `n_clusters=100`, euclidean, over `range(CTAG_SEED, CTAG_SEED + 16)`:
    a nominal 1% delivered 1.04%-3.47% of the rows (median 1.43%), with
    cluster shares from 1 row to 19,363 of the 100,000. An independent
    re-derivation from a different draw was *wider* — 1.00%-4.96%, cells of 1
    to 25,299 — so these are a lower bound on the gap, not its extent. The
    overshoot is always upwards (smallest threshold *covering* the target) but
    its size depends on which cluster the renumbering put first, so it is not
    even a fixed bias a reader could correct for.

    Recording the nominal value would put a number in `selectivity` that no
    part of the run measured, indistinguishable downstream from a measured one
    — which has happened four times here, most expensively with
    `ops.pgvector_iterative_scan`: declared in three config files, executed in
    none of 69 archived records. Hence no `nominal` field here; it travels on
    `ResolvedFilter` beside `actual_selectivity` so a record shows both.
    """
    threshold: int
    actual_selectivity: float


def selectivity_to_threshold(selectivity: float,
                             buckets: int = TAG_BUCKETS) -> int:
    """Convert a target selectivity into a `tag <` threshold.

    Rounded up and clamped to at least 1 so a 0.5% request on 100 buckets still
    selects a non-empty set rather than silently matching nothing.
    """
    return max(1, min(buckets, int(round(selectivity * buckets))))


def tag_threshold_for(tags: numpy.ndarray, selectivity: float,
                      buckets: int = TAG_BUCKETS) -> FilterThreshold:
    """`tag <` threshold for a target selectivity, with the share it delivers.

    The share is *counted*, not assumed. `threshold / buckets` is right only
    when the row count is a multiple of the bucket count: 150 rows over 100
    buckets gives rows 0 and 100 the same tag, so `tag < 1` selects 1.33% of
    the corpus and not 1%. glove's 1,183,514 training rows are not a multiple
    of 100 either. The error is small here and unbounded for `ctag`, and the
    same code path reports both, so neither is allowed to assume.
    """
    total = int(len(tags))
    if total == 0:
        raise ValueError("cannot pick a threshold over an empty tag array")
    threshold = selectivity_to_threshold(selectivity, buckets)
    actual = float(numpy.count_nonzero(tags < threshold)) / total
    return FilterThreshold(threshold, actual)


def ctag_threshold_for(ctags: numpy.ndarray, selectivity: float,
                       n_clusters: Optional[int] = None) -> FilterThreshold:
    """Smallest `ctag <` threshold whose clusters cover the target share.

    Returns that threshold and the share it actually covers. The clusters are
    nearest-pivot cells of real data, so their sizes span four orders of
    magnitude (1 row to 19,363 on the corpus named in `FilterThreshold`, 1 to
    25,299 on an independent re-derivation) and the delivered share steps
    rather than slides: the step is one whole cluster, so a request falling
    just inside a large one overshoots by its entire share. Worst at low
    selectivities, where one cluster is the whole qualifying set.

    `n_clusters` is accepted because an all-empty tail cannot be inferred from
    the labels; without it the range is `max(label) + 1`.
    """
    total = int(len(ctags))
    if total == 0:
        raise ValueError("cannot pick a threshold over an empty ctag array")
    labels = numpy.asarray(ctags, dtype=numpy.int64)
    if labels.min() < 0:
        raise ValueError("cluster ids must be non-negative")
    declared = (int(n_clusters) if n_clusters is not None
                else int(labels.max()) + 1)
    if declared < 1:
        raise ValueError(f"n_clusters must be at least 1, got {declared}")
    if int(labels.max()) >= declared:
        raise ValueError(
            f"cluster id {int(labels.max())} is outside the declared range of "
            f"{declared} clusters"
        )

    # Cumulative *row counts*, compared against the target expressed in rows,
    # so the comparison is integer-exact rather than a sum of shares whose
    # rounding would decide boundary cases.
    cumulative = numpy.cumsum(numpy.bincount(labels, minlength=declared))
    target_rows = float(selectivity) * total
    # side="left" gives the first index whose cumulative count is >= the
    # target, which is the "smallest t" the spec asks for. side="right" would
    # walk past any cluster that sits exactly on the boundary and past every
    # empty cluster behind it — selecting the whole corpus where half was
    # asked for, in the direction that makes filtered search look easy.
    threshold = int(numpy.searchsorted(cumulative, target_rows, side="left")) + 1
    threshold = max(1, min(declared, threshold))
    actual = float(cumulative[threshold - 1]) / total
    return FilterThreshold(threshold, actual)


def threshold_for(column: str, values: numpy.ndarray, selectivity: float,
                  buckets: int = TAG_BUCKETS,
                  n_clusters: Optional[int] = None) -> FilterThreshold:
    """Dispatch to the threshold rule for `column`."""
    if column == FILTER_TAG:
        return tag_threshold_for(values, selectivity, buckets)
    if column == FILTER_CTAG:
        return ctag_threshold_for(values, selectivity, n_clusters)
    raise ValueError(
        f"unknown filter column {column!r}; expected one of {FILTER_COLUMNS}"
    )


# `eq=False` on purpose: the generated __eq__ would compare the `mask` arrays
# inside a tuple comparison, which raises "truth value of an array is
# ambiguous" rather than answering. Nothing needs to compare two of these.
@dataclass(frozen=True, eq=False)
class ResolvedFilter:
    """Everything about one filter that a record has to be able to state.

    This exists so the number a workload records cannot drift from the number
    the ground truth was computed with: the mask, the threshold and the actual
    share are produced together from one pass over one column and handed over
    as one object beside the truth array. The caller never reconstructs the
    selectivity, so it cannot reconstruct it wrongly.
    """
    column: str
    threshold: int
    nominal_selectivity: float
    actual_selectivity: float
    rows: int
    buckets: Optional[int]
    n_clusters: Optional[int]
    seed: Optional[int]
    mask: numpy.ndarray

    def as_extra(self) -> Dict[str, object]:
        """The provenance block for a record's `extra` column.

        Both selectivities are present and named, so an archived record shows
        what was requested and what was measured — the distinction four
        shipped defects here turned on. Fields that do not apply to the column
        are `None` rather than their defaults: a `tag` measurement reporting
        `ctag_clusters: 100` would state a parameter the run never used.
        """
        return {
            "filter_column": self.column,
            "filter_threshold": self.threshold,
            "nominal_selectivity": self.nominal_selectivity,
            "actual_selectivity": self.actual_selectivity,
            "tag_buckets": self.buckets,
            "ctag_clusters": self.n_clusters,
            "ctag_seed": self.seed,
        }


def filter_values(cache_dir: str, dataset: Dataset, column: str,
                  indexed_rows: Optional[int] = None,
                  buckets: int = TAG_BUCKETS,
                  n_clusters: int = CTAG_CLUSTERS,
                  seed: int = CTAG_SEED) -> numpy.ndarray:
    """The scalar column itself, over the rows the engine was given."""
    n_train = len(dataset.train)
    rows = n_train if indexed_rows is None else min(int(indexed_rows), n_train)
    if column == FILTER_TAG:
        return assign_tags(rows, buckets)
    if column == FILTER_CTAG:
        return cached_ctags(cache_dir, dataset, indexed_rows=rows,
                            n_clusters=n_clusters, seed=seed)
    raise ValueError(
        f"unknown filter column {column!r}; expected one of {FILTER_COLUMNS}"
    )


def resolve_filter(cache_dir: str, dataset: Dataset, column: str,
                   selectivity: float, indexed_rows: Optional[int] = None,
                   buckets: int = TAG_BUCKETS,
                   n_clusters: int = CTAG_CLUSTERS,
                   seed: int = CTAG_SEED) -> ResolvedFilter:
    """Turn a requested selectivity into the predicate that will be run."""
    n_train = len(dataset.train)
    rows = n_train if indexed_rows is None else min(int(indexed_rows), n_train)
    values = filter_values(cache_dir, dataset, column, indexed_rows=rows,
                           buckets=buckets, n_clusters=n_clusters, seed=seed)
    found = threshold_for(column, values, selectivity, buckets=buckets,
                          n_clusters=n_clusters)
    return ResolvedFilter(
        column=column,
        threshold=found.threshold,
        nominal_selectivity=float(selectivity),
        actual_selectivity=found.actual_selectivity,
        rows=rows,
        buckets=buckets if column == FILTER_TAG else None,
        n_clusters=n_clusters if column == FILTER_CTAG else None,
        seed=seed if column == FILTER_CTAG else None,
        mask=values < found.threshold,
    )


# ---------------------------------------------------------------------------
# Filtered ground truth
# ---------------------------------------------------------------------------

#: Corpus rows gathered per block while the qualifying subset is materialised.
#: 8192 rows x 1536 dims x 4 B is 50 MB, so the block temporary stays small
#: even for the widest dataset in KNOWN_DATASETS. Expressed in rows rather
#: than bytes because it is a parameter a caller can pin, and a byte budget
#: would make the row count — and with it the gather pattern — vary with the
#: dim, which is the sort of thing that makes one dataset's numbers
#: irreproducible from another's settings.
GROUND_TRUTH_ROW_BLOCK = 8192


def masked_ground_truth(train: numpy.ndarray, test: numpy.ndarray,
                        metric: str, k: int, mask: numpy.ndarray,
                        batch_size: int = 256,
                        row_block: int = GROUND_TRUTH_ROW_BLOCK,
                        ) -> numpy.ndarray:
    """Exact top-k over the rows `mask` selects.

    Brute force, in batches, because this must be exact: it is the yardstick
    the engines' filtered results are scored against. Cost is O(n_test *
    n_qualify * dim) — minutes, not hours, at these sizes — and the caller
    caches it per (dataset, metric, k, column, threshold, rows).

    Takes a mask rather than a column and a threshold so one brute force
    serves both columns. The alternative, a second copy of this loop for
    `ctag`, is two yardsticks that can drift apart, and the whole comparison
    hangs off them agreeing.

    **This function once OOM-killed the client container.** Run
    `full-20261015-215006`, `gist-960-euclidean / tuned / M=32`: after 5 h 46 m
    of ingest and a complete concurrency grid, the filtered phase reached a
    nominal 50% selectivity and `percona-ops-m32-incremental-tuned.log` ends
    mid-sentence at "computing exact ground truth for tag at a nominal 50% of
    rows…" with no traceback, while the server container's memory samples run
    to the end of the phase. An abrupt end with no exception is a kill, not a
    failure, and it cost the 50% point and the entire churn step for the only
    gist unit that had ever survived its own load. The client limit is 8 GiB
    (`memory.client_limit_gb: 8`); the caller still holds `dataset.train`,
    3.84 GB at that shape, so only ~4.75 GB was ever available here. Four
    allocations, none of them necessary, spent more than that:

    * the one-step gather `train[qualifying]` (1.92 GB at 500,000 x 960),
      which on the ANGULAR path is held *beside* the normalized copy
      `_normalize` returns, so that path peaked at two full subsets;
    * `(subset ** 2).sum(axis=1)` inside the query loop — loop-invariant, and
      a third full-size copy rebuilt on every batch;
    * three more score-matrix-sized temporaries (512 MB each) from evaluating
      the expanded distance as one expression;
    * `numpy.argpartition(..., axis=1)`, which returns a full
      (batch, n_qualify) **int64** array — 1.02 GB, twice the score matrix —
      before `[:, :k]` discards all but ten columns of it.

    All four are gone. `row_block` bounds the gather and the norm pass;
    `batch_size` keeps its old meaning and batches *queries*.

    What is deliberately **not** done is scoring the corpus rows in blocks and
    merging a running top-k, which is the obvious way to make the peak
    independent of the selectivity. It cannot be done without changing the
    answer, measured two ways on numpy 1.26.4:

    * splitting `batch @ subset.T` along either axis **can** change the
      computed distances in their last bits, because the BLAS picks its kernel
      and its blocking from the shape it is handed. Splitting a
      256 x 5,000 x 960 product into 1,024-row column blocks moves values near
      31 by up to 7.6e-05, ~2.4e-6 relative. Small shapes can come out
      bit-identical — 4 x 256 x 16 in 64-row blocks does — so this is a hazard
      rather than a guarantee, which for a yardstick is the same thing. On a
      corpus seeded with rows one ULP apart it reordered the top-10 of 4 to 17
      queries in 20.
    * among *exactly* equal distances the surviving order is whatever
      `argpartition`'s introselect happens to leave behind. Measured, both
      metrics: in a 100-row corpus holding ten copies each of ten distinct
      4-dim vectors, queried with row 0, the ten rows tied at distance zero
      come back as `[6, 9, 0, 8, 7, 1, 5, 2, 4, 3]`
      (`test_argpartition_does_not_resolve_a_tie_by_row_id` pins it). It is
      not a rule to be reimplemented: ten tied rows *and nothing else* come
      back `[0..9]`, because introselect leaves an array that short alone.
      That is exactly the point — the order is an artefact of how the
      partition ran, not a property of the data, so no merge over blocks can
      reproduce it without the whole score row in front of it.

    So the qualifying axis is gathered in blocks but **scored whole**, and
    every engine is still scored against the array the previous runs were.
    `tests/test_e10_filtered_ground_truth_memory.py` holds the current
    algorithm as a reference and asserts bit-identical output.
    """
    qualifying_mask = numpy.asarray(mask)
    if qualifying_mask.dtype != numpy.dtype(bool):
        # An index array would pass silently: numpy.nonzero([5, 9, 13]) is
        # [0, 1, 2], so handing the qualifying row *ids* to this parameter
        # would score every engine against the first few rows of the corpus
        # and still return a full-shaped, entirely plausible answer.
        raise TypeError(
            f"mask must be a boolean array, got dtype {qualifying_mask.dtype}"
        )
    if qualifying_mask.ndim != 1 or len(qualifying_mask) != len(train):
        raise ValueError(
            f"mask must be one value per row: got {qualifying_mask.shape} for "
            f"{len(train)} rows"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if row_block < 1:
        raise ValueError(f"row_block must be at least 1, got {row_block}")

    qualifying = numpy.nonzero(qualifying_mask)[0]
    if len(qualifying) == 0:
        raise ValueError("no rows qualify for this filter")

    n_qualify = len(qualifying)
    dim = int(train.shape[1])

    # One full-size subset, written block by block into a buffer that is
    # allocated once. `subset = train[qualifying]` followed by
    # `subset = _normalize(subset)` reads as two statements and allocates two
    # arrays that are briefly live together; normalizing each block as it
    # lands keeps a single full-size array plus one `row_block`-sized
    # temporary. The dtype is pinned here rather than inferred later because
    # the ANGULAR path's buffer has to match what `_normalize` returns, or
    # every row would be rounded twice.
    subset = numpy.empty((n_qualify, dim),
                         dtype="<f4" if metric == ANGULAR else train.dtype)
    for start in range(0, n_qualify, row_block):
        rows = qualifying[start:start + row_block]
        # The gathered block is deliberately not given a name. A name holds two
        # blocks at once across each loop edge, and — the part that actually
        # costs — keeps the *last* one alive for the whole rest of the call.
        # Measured on the 30,000 x 512 probe, whose final block is 5,424 rows:
        # the euclidean peak goes 93.10 -> 104.21 MB, which is exactly the
        # 11.11 MB that trailing block occupies. The ANGULAR path reads
        # 112.05 MB either way and buys nothing here, because its peak is set
        # earlier by `_normalize`'s own temporaries, while the gathered block
        # is necessarily alive as its argument.
        subset[start:start + len(rows)] = (
            _normalize(train[rows]) if metric == ANGULAR else train[rows]
        )
    queries = _normalize(test) if metric == ANGULAR else test

    k = min(k, n_qualify)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    out = numpy.empty((len(queries), k), dtype=numpy.int64)

    # ||x||^2 per qualifying row, hoisted: it does not depend on the query
    # batch, and computing it inside the loop rebuilt a whole extra copy of
    # the subset every time round.
    #
    # Deliberately **not** `numpy.einsum("ij,ij->i", subset, subset)`, which
    # allocates only the n-element result and is the obvious way to write
    # this. Measured on numpy 1.26.4 it disagrees with
    # `(subset ** 2).sum(axis=1)` in the last bits for every shape tried,
    # because it does not reproduce the per-row pairwise summation `.sum` uses.
    # This term is per *column* of the score matrix, so unlike the per-query
    # term below it decides the ordering — swapping it would rescore every
    # filtered recall number in every archived report. Row-blocked `.sum` is
    # bit-identical instead, because each row's reduction reads only that row.
    row_sq = None
    if metric != ANGULAR:
        row_sq = numpy.empty(n_qualify, dtype=subset.dtype)
        for start in range(0, n_qualify, row_block):
            block = subset[start:start + row_block]
            row_sq[start:start + len(block)] = (block ** 2).sum(axis=1)

    # One score matrix for the whole run, reused batch by batch. Written as
    # `scores = batch @ subset.T` the matrix is reallocated per batch, and
    # because the name is only rebound once the new one exists, the old and the
    # new are briefly live together: at the gist shape the final short batch
    # allocated 464 MB beside the 512 MB it was replacing. `numpy.matmul` with
    # `out=` is the same gemm over the same shapes — the sub-view handed to it
    # is (len(batch), n_qualify), exactly what `batch @ subset.T` would be — so
    # the arithmetic is untouched.
    score_buf = numpy.empty(
        (min(batch_size, len(queries)), n_qualify),
        dtype=numpy.result_type(queries.dtype, subset.dtype),
    )

    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        # Scored across the whole qualifying axis in one call, on purpose: the
        # docstring has the measurement showing that splitting this gemm
        # changes the distances it returns.
        scores = score_buf[:len(batch)]
        numpy.matmul(batch, subset.T, out=scores)
        if metric == ANGULAR:
            # Cosine distance ordering is the reverse of dot-product ordering
            # once both sides are unit-normalized.
            numpy.negative(scores, out=scores)
        else:
            # ||a-b||^2 expanded; the ||a||^2 term is constant per query and so
            # does not affect ordering, but is kept for numerical clarity.
            # Accumulated in place rather than as one expression, which built
            # a fresh score matrix for each of the three operations. The
            # rearrangement from `a - 2*g + c` to `(-2*g) + a + c` is exact,
            # not approximate: IEEE negation and scaling by a power of two are
            # exact, and addition commutes.
            scores *= -2.0
            scores += (batch ** 2).sum(axis=1, keepdims=True)
            scores += row_sq[None, :]
        for i in range(len(batch)):
            # One query row at a time, because `numpy.argpartition(axis=1)`
            # returns a full (batch, n_qualify) int64 array — at the gist shape
            # 1.02 GB, twice the score matrix it indexes — and `[:, :k]` then
            # throws away all but k columns. Per row that allocation is 4 MB.
            # numpy runs the same 1-D introselect on each row either way, so
            # the indices and their order among equal distances are unchanged.
            row = scores[i]
            picked = numpy.argpartition(row, kth=k - 1)[:k]
            out[start + i] = qualifying[picked[numpy.argsort(row[picked])]]

    return out


def filtered_ground_truth(train: numpy.ndarray, test: numpy.ndarray,
                          metric: str, k: int, tag_threshold: int,
                          buckets: int = TAG_BUCKETS,
                          batch_size: int = 256,
                          row_block: int = GROUND_TRUTH_ROW_BLOCK,
                          ) -> numpy.ndarray:
    """Exact top-k over only the rows passing `tag < tag_threshold`.

    A thin wrapper: it builds the `tag` mask and hands it to
    `masked_ground_truth`, which does the work. Kept because every caller and
    test that predates the second filter column speaks in terms of a tag
    threshold, and because the tag column is the control case that stays.

    Both blocking parameters are passed straight through. A wrapper that
    accepted only one of them would quietly give its callers the default for
    the other, which is the drift this wrapper exists to prevent.
    """
    mask = assign_tags(len(train), buckets) < tag_threshold
    if not mask.any():
        raise ValueError(f"no rows qualify for tag < {tag_threshold}")
    return masked_ground_truth(train, test, metric, k, mask,
                               batch_size=batch_size, row_block=row_block)


def _normalize(a: numpy.ndarray) -> numpy.ndarray:
    norms = numpy.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (a / norms).astype("<f4")


def cached_ground_truth(cache_dir: str, dataset: Dataset, k: int,
                        indexed_rows: Optional[int] = None) -> numpy.ndarray:
    """Exact unfiltered top-k over the rows the engine actually holds.

    When the whole training set is loaded this is just the neighbours shipped in
    the HDF5 file. When a profile loads a subset, those neighbours point at rows
    the engine never received, so the correct answers have to be recomputed over
    the subset — otherwise recall is measured against unreachable targets.
    """
    n_train = len(dataset.train)
    rows = n_train if indexed_rows is None else min(indexed_rows, n_train)
    if rows >= n_train:
        return dataset.neighbors[:, :k]

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dataset.name}.k{k}.n{rows}.full.npy")
    if os.path.exists(path):
        return numpy.load(path)

    # An all-true mask over the loaded rows: the filtered brute force with
    # nothing filtered out is exactly the unfiltered ground truth for that
    # subset. (This used to say `tag < buckets`, which meant the same thing by
    # a longer route.)
    truth = masked_ground_truth(
        dataset.train[:rows], dataset.test, dataset.metric, k,
        numpy.ones(rows, dtype=bool),
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        numpy.save(fh, truth)
    os.replace(tmp, path)
    return truth


def filtered_truth_cache_name(dataset_name: str, metric: str, k: int,
                              rows: int, column: str, threshold: int,
                              buckets: int, n_clusters: int,
                              seed: int) -> str:
    """Cache file name for one filtered ground truth.

    Nine components, because all nine change the correct answer set and a key
    missing one silently scores a run against another configuration's targets.
    `column` is the easiest to leave out and the most damaging to: `tag` and
    `ctag` at the same threshold, over the same dataset, k and row count, have
    entirely different qualifying sets, so a shared key reports the correlated
    filter's recall as the independent filter's. `metric` decides the answer
    twice over — which rows are nearest, and through `assign_ctags` which rows
    share a cluster — and is reachable for the reason `ctag_cache_name`
    gives.

    `buckets`, `n_clusters` and `seed` are present for either column though
    each applies to one: that costs a recomputation if a default changes and
    guarantees no stale file is served under a changed parameter. A key that
    varies with something irrelevant only wastes work; one that misses
    something relevant publishes a wrong number.
    """
    return (f"{dataset_name}.{metric}.k{k}.{column}.t{threshold}.b{buckets}"
            f".c{n_clusters}.s{seed}.n{rows}.npy")


def filtered_truth_for(cache_dir: str, dataset: Dataset, k: int, column: str,
                       selectivity: float, indexed_rows: Optional[int] = None,
                       buckets: int = TAG_BUCKETS,
                       n_clusters: int = CTAG_CLUSTERS,
                       seed: int = CTAG_SEED,
                       ) -> Tuple[numpy.ndarray, ResolvedFilter]:
    """Filtered ground truth and the filter that produced it, together.

    The entry point workloads should use. It returns a pair on purpose: the
    actual selectivity comes from the data, not the arguments, so handing back
    the array alone leaves the caller nothing to record but what it asked for.
    `filt.actual_selectivity` belongs in the record's `selectivity` column and
    `filt.as_extra()` carries the rest;
    `tests/test_datasets.py::TestTheRecordedSelectivityIsTheMeasuredOne`
    enforces that at the source level.

    `indexed_rows` must be the rows actually loaded. Profiles can load a
    subset (smoke uses 20,000 of 60,000), and ground truth over the full set
    would include rows the engine never received, understating every engine at
    once. It is in the cache key for that reason.
    """
    filt = resolve_filter(cache_dir, dataset, column, selectivity,
                          indexed_rows=indexed_rows, buckets=buckets,
                          n_clusters=n_clusters, seed=seed)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filtered_truth_cache_name(
        dataset.name, dataset.metric, k, filt.rows, column, filt.threshold,
        buckets, n_clusters, seed,
    ))
    if os.path.exists(path):
        return numpy.load(path), filt
    truth = masked_ground_truth(dataset.train[:filt.rows], dataset.test,
                                dataset.metric, k, filt.mask)
    # numpy.save() appends ".npy" to a *path* that lacks it, which would turn
    # "<name>.npy.tmp" into "<name>.npy.tmp.npy" and break the atomic rename.
    # Passing an open file object suppresses that behaviour.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        numpy.save(fh, truth)
    os.replace(tmp, path)
    return truth, filt


def cached_filtered_ground_truth(cache_dir: str, dataset: Dataset, k: int,
                                 selectivity: float,
                                 buckets: int = TAG_BUCKETS,
                                 indexed_rows: Optional[int] = None,
                                 column: str = FILTER_TAG,
                                 n_clusters: int = CTAG_CLUSTERS,
                                 seed: int = CTAG_SEED) -> numpy.ndarray:
    """The truth array alone, discarding the resolved filter.

    A thin wrapper over `filtered_truth_for` for callers that record no
    selectivity. Anything that writes a record must call `filtered_truth_for`
    and record `filt.actual_selectivity`: the only selectivity available here
    is the one that was asked for.
    """
    return filtered_truth_for(cache_dir, dataset, k, column, selectivity,
                              indexed_rows=indexed_rows, buckets=buckets,
                              n_clusters=n_clusters, seed=seed)[0]
