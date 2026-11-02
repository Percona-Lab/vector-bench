"""T3c: the filtered workload's two columns, the recall split and the budget.

PR 3 §3.3 and §3.4, against `tests/fake_driver.py`. No engine is started.

**Why §3.5's prescribed tests are not the tests here.** Its third bullet asks
for "`ctag` `near_fail` recall is clearly below `tag` recall at the same
selectivity". Measured on the 2,000-row fixture that claim is false in two of
three places:

    column  nominal  actual  overall  near_pass        near_fail
    tag     0.01     0.0100  0.0740   none (0 queries) 0.0740 (50)
    tag     0.10     0.1000  0.6840   0.7667 (6)       0.6727 (44)
    tag     0.50     0.5000  1.0000   1.0000 (28)      1.0000 (22)
    ctag    0.01     0.0190  0.1460   0.1500 (2)       0.1458 (48)
    ctag    0.10     0.1085  0.6200   0.7750 (4)       0.6065 (46)
    ctag    0.50     0.5060  1.0000   1.0000 (24)      1.0000 (26)

At 50% both columns reach recall 1.0, so "clearly below" fails outright. At 1%
`ctag`'s near_fail recall (0.1458) is *higher* than `tag`'s overall recall
(0.0740) -- inverted -- because a nominal 1% on `ctag` delivers 1.90% of the
rows and more qualifying rows survive a post-filter. The two columns cannot be
at "the same selectivity" for one request, which is the whole point of the
measured-share fix.

The fixture is also the wrong corpus for the claim: `tiny_dataset` is uniform
random in 16 dimensions, where nearest-pivot cells barely predict nearest
neighbours. So the mechanism is built rather than hoped for -- `blob_dataset`
below has real clusters -- and the discriminating assertion is **two-sided**:
the near_pass/near_fail gap must be large for `ctag` AND small for `tag`. A
one-sided "ctag is worse" passes if the implementation reports garbage for
`ctag`, and passes vacuously wherever both numbers are near 0 or near 1.

**Mutation discipline.** Every guard below names the regression it catches and
was checked by injecting that regression at run time (monkeypatch / subclass /
`mock.patch`), never by editing a file on disk -- CPython invalidates a `.pyc`
on (whole-second mtime, size), so a same-size rewrite inside one second can
leave stale bytecode valid and measure the previous version. The honest and
mutant values are recorded in each docstring, read back out of the records the
workload wrote rather than compared against strings typed here.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List
from unittest import mock

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds                                  # noqa: E402
from harness.datasets import (FILTER_COLUMNS, FILTER_CTAG,           # noqa: E402
                              FILTER_TAG, Dataset)
from harness.drivers.base import RowAttrs                            # noqa: E402
from harness.metrics import records                                  # noqa: E402
from harness.workloads import filtered as fw                         # noqa: E402
from harness.workloads.context import RunContext                     # noqa: E402
from orchestrator import estimate as est_mod                         # noqa: E402
from orchestrator.estimate import estimate_load_hours                # noqa: E402
from report import charts, generate, render                          # noqa: E402
from tests.fake_driver import (DIM, FakeDriver, N_QUERIES, N_ROWS,    # noqa: E402
                               index_spec, tiny_dataset)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K = 10

#: Distinguishes "the profile omitted the key" from "the profile wrote null",
#: which are different inputs with the same `.get()` answer.
_ABSENT = object()

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
            run_id="t3c-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


def written(ctx: RunContext) -> List[Dict]:
    return list(records.read_records(ctx.recorder.path))


def two_column_driver(dataset: Dataset, cache_dir: str, **kwargs) -> FakeDriver:
    """A loaded FakeDriver whose `tag` and `ctag` genuinely differ.

    `fake_driver.loaded_driver` defaults to a bare array, which
    `RowAttrs.coerce` MIRRORS into `ctag` -- so a predicate on either column
    gives the same answer and every `ctag` assertion below would pass against
    a workload that read `tag` twice. That is the vacuity `loaded_driver`'s
    own docstring warns about; this helper is the "pass a RowAttrs whose two
    columns differ" it asks for.

    Sourced from `filter_values` for BOTH columns, keyed on the same
    (dataset, rows) pair the workload's ground truth uses, which is the
    discipline `build.run` exists to enforce.
    """
    rows = len(dataset.train)
    attrs = RowAttrs(
        tag=ds.filter_values(cache_dir, dataset, FILTER_TAG, indexed_rows=rows),
        ctag=ds.filter_values(cache_dir, dataset, FILTER_CTAG, indexed_rows=rows),
    )
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, **kwargs)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train, attrs)
    return driver


def blob_dataset(n=2_000, dim=DIM, centres=25, spread=0.06, seed=7) -> Dataset:
    """A clustered corpus, so `ctag` actually correlates with the neighbours.

    `tiny_dataset` is uniform random, where a nearest-pivot cell contains a
    query's true neighbours about as often as its share of the rows -- so
    `ctag` behaves almost like `tag` on it and the collapse PR 3 is about
    cannot be observed. Gaussian blobs put a query's neighbours in its own
    cluster, which is the structure a real embedding has and the structure the
    correlated filter is meant to exploit.

    Exact ground truth by argsort, like `tiny_dataset`, because the recall
    assertions below need the real top-k and not something close to it.
    """
    rng = numpy.random.default_rng(seed)
    centroids = rng.random((centres, dim), dtype=numpy.float32)
    labels = rng.integers(0, centres, size=n)
    train = (centroids[labels]
             + rng.normal(0, spread, size=(n, dim))).astype(numpy.float32)
    test_labels = rng.integers(0, centres, size=N_QUERIES)
    test = (centroids[test_labels]
            + rng.normal(0, spread, size=(N_QUERIES, dim))).astype(numpy.float32)
    scores = ((test ** 2).sum(1)[:, None] - 2.0 * (test @ train.T)
              + (train ** 2).sum(1)[None, :])
    neighbors = numpy.argsort(scores, axis=1)[:, :100].astype(numpy.int64)
    ordered = numpy.take_along_axis(scores, neighbors, axis=1)
    return Dataset(name=f"blob-{dim}-euclidean", train=train, test=test,
                   neighbors=neighbors,
                   distances=numpy.sqrt(numpy.maximum(ordered, 0.0)),
                   metric=ds.EUCLIDEAN)


class TickingClock:
    """A clock that advances only when a query runs.

    A wall-clock budget test is a timing assertion on a shared machine, which
    is the shape §1.3's "QPS within +-1% of 160" had when it passed the bug
    and failed the fix on 18 of 18 runs. This removes the machine from the
    measurement: `time.time` returns a value that moves by exactly `step`
    per `query_filtered` call and by nothing else, so the executed count is
    arithmetic rather than a race.

    Patching `time.time` globally is safe for this workload specifically:
    `LatencyCollector`, `Progress` and `Heartbeat` all read
    `time.perf_counter()`, and `RunContext.record_defaults` timestamps with
    `datetime`. Verified by `test_the_only_wall_clock_the_budget_reads_is_time_time`.
    """

    def __init__(self, driver, step: float, start: float = 0.0):
        self.now = start
        self.step = step
        self._inner = driver.query_filtered
        driver.query_filtered = self._query
        self.calls = 0

    def _query(self, *args, **kwargs):
        self.calls += 1
        self.now += self.step
        return self._inner(*args, **kwargs)

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# §3.3 item 1 -- columns x selectivity, and the share that gets recorded
# ---------------------------------------------------------------------------

class TestBothColumnsAreMeasuredAtEverySelectivity:

    def test_the_loop_is_column_times_selectivity(self, ctx, tmp_path):
        """Regression: measuring one column and leaving the other unqueried.

        `build.run` has loaded both since T3b, so a workload that measures one
        leaves a column in the table that nothing ever asks about -- and the
        correlated filter is the half of the problem PR 3 exists to measure.

        Honest: 4 records, (column, nominal) =
            [('tag', 0.01), ('tag', 0.5), ('ctag', 0.01), ('ctag', 0.5)].
        Mutant (`fw.DEFAULT_COLUMNS` patched to `('tag',)`): 2 records,
            [('tag', 0.01), ('tag', 0.5)] -- and every ctag assertion in this
            file disappears rather than failing, which is why the pairs are
            asserted as a list and not with `in`.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01, 0.50), max_queries=N_QUERIES, warmup=2,
               indexed_rows=N_ROWS)

        rows = written(ctx)
        assert [(r["extra"]["filter_column"], r["extra"]["nominal_selectivity"])
                for r in rows] == [("tag", 0.01), ("tag", 0.50),
                                   ("ctag", 0.01), ("ctag", 0.50)]
        assert all(r["phase"] == records.PHASE_FILTERED for r in rows)

    def test_the_default_is_every_column_the_dataset_module_declares(self):
        """Pinned against `FILTER_COLUMNS` rather than restated as a literal.

        A third column added to `harness/datasets.py` and not to the
        workload's default would load into the table and never be measured --
        exactly the state T3b left `ctag` in, and the state this test exists
        to make visible.
        """
        assert tuple(fw.DEFAULT_COLUMNS) == tuple(FILTER_COLUMNS)
        assert len(FILTER_COLUMNS) >= 2, "a one-column default proves nothing"

    def test_an_unknown_column_is_refused_before_anything_is_measured(
            self, ctx, tmp_path):
        """Regression: a typo costing a ground-truth brute force first.

        The driver whitelist refuses an unknown column too, but it refuses it
        per query -- after this workload has spent a brute force computing a
        yardstick for a column no engine can be asked about. On glove that is
        minutes.

        Honest: ValueError naming the column, and zero records written.
        Mutant (the `unknown` check deleted): `ValueError: unknown filter
        column 'ctag_idx'` still arrives, but out of `filter_values` after
        `_scan_modes` has already issued a SET -- and with `nothing` as the
        column it is the driver that raises, per query.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        with pytest.raises(ValueError, match="ctag_idx"):
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
                   selectivities=(0.50,), columns=("tag", "ctag_idx"),
                   max_queries=8, warmup=0, indexed_rows=N_ROWS)
        assert written(ctx) == []


class TestTheRecordedSelectivityIsCounted:
    """§3.3 item 1's "and the actual selectivity", which is F-1's defect.

    `tests/test_datasets.py::TestTheRecordedSelectivityIsTheMeasuredOne` is
    the source-level gate and it only proves the expression ends in
    `.actual_selectivity`. These are the value tests: that the number is the
    counted share, and that it differs from BOTH wrong answers.
    """

    def test_a_ctag_record_rejects_both_the_nominal_and_the_cluster_fraction(
            self, ctx):
        """The three-way discrimination the gate cannot make.

        A nominal 1% on `ctag` over the 2,000-row fixture selects clusters 0
        and 1, which hold 38 of 2,000 rows. So:

            counted share       0.019   <- what must be recorded
            nominal             0.01    <- what the request was
            threshold / 100     0.02    <- what the old code recorded

        All three are distinct here, which is what makes this fixture able to
        tell them apart; at 50% the counted share is 0.506 against a nominal
        0.5 and a threshold fraction of 0.54, so it discriminates there too.

        Honest: selectivity == 0.019, read back out of the record.
        Mutant A (`selectivity=filt.nominal_selectivity`): 0.01.
        Mutant B (`selectivity=filt.threshold / 100.0`, the shipped defect):
            0.02.
        Both mutants are caught, and each is caught by a different one of the
        two inequalities below -- which is why both are asserted rather than
        just the equality.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01,), columns=(FILTER_CTAG,), max_queries=8,
               warmup=0, indexed_rows=N_ROWS)

        row = written(ctx)[0]
        extra = row["extra"]
        threshold = extra["filter_threshold"]
        ctags = ds.filter_values(ctx.cache_dir, dataset, FILTER_CTAG,
                                 indexed_rows=N_ROWS)
        counted = float(numpy.count_nonzero(ctags < threshold)) / N_ROWS

        assert row["selectivity"] == counted == 0.019
        assert row["selectivity"] != extra["nominal_selectivity"] == 0.01
        assert row["selectivity"] != threshold / 100.0 == 0.02

    def test_the_nominal_is_kept_beside_it_rather_than_discarded(self, ctx):
        """Both numbers, named. A record carrying only the counted share
        cannot answer "what did the profile ask for?", and a reader comparing
        two engines needs to know they were asked the same question."""
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01,), max_queries=8, warmup=0,
               indexed_rows=N_ROWS)
        for row in written(ctx):
            assert row["extra"]["nominal_selectivity"] == 0.01
            assert row["extra"]["actual_selectivity"] == row["selectivity"]

    def test_tag_threshold_is_written_for_tag_and_withheld_from_ctag(self, ctx):
        """Regression: a key named for one column stating another's threshold.

        `extra.tag_threshold` predates the second column and archived readers
        speak in its terms, so it stays -- for `tag`. On a `ctag` record it
        would be the same class of defect as the selectivity above: a
        measured-sounding name over a number from somewhere else.

        Honest: present on the tag record (value 1), absent on the ctag one.
        Mutant (the `if column == FILTER_TAG` guard dropped): present on both,
        and the ctag record claims `tag_threshold: 2` while its predicate read
        `ctag`.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01,), max_queries=8, warmup=0,
               indexed_rows=N_ROWS)
        tag_row, ctag_row = written(ctx)
        assert tag_row["extra"]["tag_threshold"] == 1
        assert "tag_threshold" not in ctag_row["extra"]
        # The column-neutral name is on both.
        assert tag_row["extra"]["filter_threshold"] == 1
        assert ctag_row["extra"]["filter_threshold"] == 2


# ---------------------------------------------------------------------------
# §3.3 item 2 -- the plan check per (column, selectivity)
# ---------------------------------------------------------------------------

class TestThePlanCheckRunsPerColumnAndSelectivity:

    def _spy(self, driver):
        asked = []
        inner = driver.explain_uses_vector_index

        def explain(vector, k, tag_threshold=None, column=FILTER_TAG):
            asked.append((column, tag_threshold))
            return inner(vector, k, tag_threshold=tag_threshold, column=column)

        driver.explain_uses_vector_index = explain
        return asked

    def test_every_point_gets_its_own_plan_check(self, ctx):
        """Regression: one EXPLAIN stamped onto every point.

        The MySQL-family optimizers can switch to `ctag_idx` or to a full scan
        at a different threshold than they do for `tag`, so a plan check taken
        once and reused reports the wrong engine behaviour for five of six
        points -- and `vector_index_used` is the field that separates "exact
        results, slowly" from "accurate and slow".

        Honest: 4 calls, (column, threshold) =
            [('tag', 1), ('tag', 50), ('ctag', 2), ('ctag', 54)].
        Mutant (the call hoisted above the selectivity loop): 2 calls,
            [('tag', 1), ('ctag', 2)].
        Mutant (hoisted above both loops, as the pre-PR-3 shape would be if
        the column loop were added around it): 1 call, [('tag', 1)].
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        asked = self._spy(driver)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01, 0.50), max_queries=8, warmup=0,
               indexed_rows=N_ROWS)
        assert asked == [("tag", 1), ("tag", 50), ("ctag", 2), ("ctag", 54)]

    def test_the_column_asked_about_is_the_column_queried(self, ctx):
        """F-3's "the label and the predicate must agree".

        T3b pinned that a three-argument `query_filtered` still sends a bare
        `tag` predicate, so nothing was mislabelled until this workload began
        passing the parameter. The failure this catches is passing `column` to
        one call and not the other -- the plan check asks about `ctag_idx`
        while the queries read `tag`, and the record says the index was used
        for a predicate that never ran.

        Honest: the two call streams agree on (column, threshold) for all 4
        points.
        Mutant (`column=column` dropped from the explain call): the explain
        stream is [('tag', 1), ('tag', 50), ('tag', 2), ('tag', 54)] against a
        query stream naming ctag -- and the thresholds still match, which is
        why the column is compared and not just the threshold.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        asked = self._spy(driver)

        queried = []
        inner = driver.query_filtered

        def query(vector, k, threshold, column=FILTER_TAG):
            queried.append((column, threshold))
            return inner(vector, k, threshold, column=column)

        driver.query_filtered = query
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01, 0.50), max_queries=4, warmup=0,
               indexed_rows=N_ROWS)

        assert asked == sorted(set(queried), key=queried.index)
        assert len(set(queried)) == 4, set(queried)


# ---------------------------------------------------------------------------
# §3.3 item 3 -- near_pass / near_fail
# ---------------------------------------------------------------------------

class TestTheNearPassNearFailSplit:

    def _expected_groups(self, dataset, cache_dir, column, selectivity, n):
        """Group membership recomputed from the dataset, not from the code.

        `dataset.neighbors[:, 0]` is the true unfiltered nearest neighbour and
        the mask comes from `resolve_filter`, so this reproduces the rule from
        its definition rather than re-running the implementation's own
        expression -- which is one of the vacuity shapes this repo has been
        bitten by.
        """
        filt = ds.resolve_filter(cache_dir, dataset, column, selectivity,
                                 indexed_rows=len(dataset.train))
        nearest = dataset.neighbors[:n, 0]
        passes = filt.mask[nearest]
        return ([i for i in range(n) if passes[i]],
                [i for i in range(n) if not passes[i]], filt)

    def test_the_groups_are_the_mask_lookup_on_the_true_nearest_neighbour(
            self, ctx):
        """Regression: splitting on something other than the stated rule.

        Honest: at `ctag` 10% over 50 queries the record says
            near_pass_queries=4, near_fail_queries=46,
        which equals the independently recomputed membership.
        Mutant (`passes` inverted to `~passes`): 46 and 4 -- the counts swap,
        and both recalls swap with them, so a report would attribute the
        collapse to the wrong group. Caught by this test and by
        `test_a_correlated_column_separates_the_groups` below; the narrow form
        is this one, which asserts the counts alone.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.10,), columns=(FILTER_CTAG,),
               max_queries=N_QUERIES, warmup=0, indexed_rows=N_ROWS)

        row = written(ctx)[0]
        want_pass, want_fail, _ = self._expected_groups(
            dataset, ctx.cache_dir, FILTER_CTAG, 0.10, N_QUERIES)
        assert row["extra"]["near_pass_queries"] == len(want_pass) == 4
        assert row["extra"]["near_fail_queries"] == len(want_fail) == 46
        # Not a test over an empty group, and the two partition the queries.
        assert len(want_pass) > 0 and len(want_fail) > 0
        assert (row["extra"]["near_pass_queries"]
                + row["extra"]["near_fail_queries"]) == row["queries_executed"]

    def test_each_group_recall_is_that_group_s_own_queries(self, ctx):
        """The recalls, recomputed from the returned ids rather than trusted.

        Honest: the record's `recall_near_fail` equals `recall_at_k` over
        exactly the near_fail queries -- 0.6065 at ctag 10%.
        Mutant (both groups scored over all 50 queries): both figures become
        the overall recall, 0.62, and the gap this measurement exists to show
        vanishes while every count still looks right.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        want_pass, want_fail, filt = self._expected_groups(
            dataset, ctx.cache_dir, FILTER_CTAG, 0.10, N_QUERIES)
        truth, _ = ds.filtered_truth_for(ctx.cache_dir, dataset, K,
                                         FILTER_CTAG, 0.10,
                                         indexed_rows=N_ROWS)
        returned = [driver.query_filtered(q, K, filt.threshold,
                                          column=FILTER_CTAG)
                    for q in dataset.test[:N_QUERIES]]
        expect_fail = ds.recall_at_k([returned[i] for i in want_fail],
                                     truth[want_fail], K)
        expect_pass = ds.recall_at_k([returned[i] for i in want_pass],
                                     truth[want_pass], K)

        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.10,), columns=(FILTER_CTAG,),
               max_queries=N_QUERIES, warmup=0, indexed_rows=N_ROWS)
        extra = written(ctx)[0]["extra"]
        assert extra["recall_near_fail"] == round(expect_fail, 6) == 0.606522
        assert extra["recall_near_pass"] == round(expect_pass, 6) == 0.775
        # The two groups differ, so neither is the overall figure by accident.
        assert extra["recall_near_pass"] != extra["recall_near_fail"]

    def test_a_correlated_column_separates_the_groups_and_tag_does_not(
            self, ctx):
        """The mechanism, two-sided, on a corpus where it exists.

        This replaces §3.5's third bullet, which is false on this repo's
        fixture (see the module docstring). The claim being tested is the one
        PR 3 actually makes: the split separates a correlated predicate and
        does not separate an uncorrelated one.

        Measured on `blob_dataset` at a nominal 10%:
            ctag  near_pass 0.9833  near_fail 0.2409   gap  0.742
            tag   near_pass 0.6250  near_fail 0.6405   gap -0.016

        Honest: ctag's gap > 0.5 and tag's |gap| < 0.15.
        Mutant (`passes` inverted): ctag's gap becomes -0.742, failing the
            first assertion -- a one-sided "ctag near_fail is low" would
            still pass, since 0.98 is not low but 0.24 would be reported as
            near_pass and never checked.
        Mutant (the split computed from the FILTERED truth's nearest row
            instead of the unfiltered one, which is the plausible
            misreading): ctag gap 0.000 with 50 queries in near_pass, because
            the filtered truth's nearest row passes the filter by
            construction. That mutant also makes near_fail empty, which the
            emptiness assertion below catches independently.
        """
        dataset = blob_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.10,), max_queries=N_QUERIES, warmup=0,
               indexed_rows=len(dataset.train))

        by_column = {r["extra"]["filter_column"]: r["extra"]
                     for r in written(ctx)}
        assert set(by_column) == {"tag", "ctag"}
        for column, extra in by_column.items():
            assert extra["near_pass_queries"] > 0, column
            assert extra["near_fail_queries"] > 0, column

        ctag_gap = (by_column["ctag"]["recall_near_pass"]
                    - by_column["ctag"]["recall_near_fail"])
        tag_gap = (by_column["tag"]["recall_near_pass"]
                   - by_column["tag"]["recall_near_fail"])
        assert ctag_gap > 0.5, (ctag_gap, by_column["ctag"])
        assert abs(tag_gap) < 0.15, (tag_gap, by_column["tag"])
        # And the thing a single-column run could never show.
        assert ctag_gap > abs(tag_gap) * 4

    def test_an_empty_group_is_absent_rather_than_zero(self, ctx):
        """Regression: a recall figure no query produced.

        `tag` at 1% puts zero of fifty queries in near_pass, because a uniform
        1% of rows contains a given query's nearest neighbour about 1% of the
        time. 0.0 there is indistinguishable downstream from a measured total
        failure, which is how the nominal selectivity survived for years.

        Honest: near_pass_queries == 0 and recall_near_pass is None, and the
            key is present so a reader can tell "no queries" from "key not
            written by this harness version".
        Mutant (`_group_recall` returning 0.0 for an empty subset): 0.0, which
            the report renders as a recall of zero beside 50 near_fail queries
            at 0.074 -- a worse-looking number than any engine produced.
        Mutant (`recall_at_k` called unguarded): float('nan'), which
            `json.dumps` writes as bare `NaN` -- invalid strict JSON, caught
            by the round-trip below.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01,), columns=(FILTER_TAG,),
               max_queries=N_QUERIES, warmup=0, indexed_rows=N_ROWS)

        extra = written(ctx)[0]["extra"]
        assert extra["near_pass_queries"] == 0
        assert "recall_near_pass" in extra
        assert extra["recall_near_pass"] is None
        assert extra["near_fail_queries"] == N_QUERIES
        assert extra["recall_near_fail"] == 0.074
        # Strict JSON: NaN and Infinity are not valid and a reader outside
        # Python would refuse the file.
        with open(ctx.recorder.path) as fh:
            for line in fh:
                json.loads(line, parse_constant=_no_constants)


def _no_constants(name):
    raise AssertionError(f"record contains the non-JSON constant {name!r}")


# ---------------------------------------------------------------------------
# §3.3 item 4 -- the returned-count histogram
# ---------------------------------------------------------------------------

class TestTheReturnedCountHistogram:

    def test_every_executed_query_lands_in_exactly_one_bucket(self, ctx):
        """Regression: a histogram whose counts do not sum to the queries.

        A distribution that silently drops values reads as complete and is
        not. Checked at every selectivity, because the buckets that are
        exercised differ: at 1% the fake returns 0-4 rows, at 50% it returns
        all ten.

        Honest: each record's buckets sum to its own `queries_executed`.
        Mutant (`_returned_count_bucket` returning `short_other` only for
            `n <= 9`, i.e. dropping the `n > 9 and n < k` case): identical at
            k=10 -- which is why `test_a_complete_answer_is_complete_at_any_k`
            exercises k=5 and k=20 instead.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01, 0.10, 0.50), max_queries=N_QUERIES,
               warmup=0, indexed_rows=N_ROWS)

        rows = written(ctx)
        assert len(rows) == 6
        for row in rows:
            hist = row["extra"]["returned_count_hist"]
            assert sum(hist.values()) == row["queries_executed"] == N_QUERIES
            # Always the full set of keys, so absence never has to be read as
            # zero by a chart.
            assert set(hist) == set(fw.RETURNED_COUNT_BUCKETS)

    def test_it_says_more_than_the_short_result_count_it_replaces(self, ctx):
        """The reason item 4 exists. `short_result_queries` counts queries
        returning fewer than k; it cannot tell an engine returning three rows
        from one returning nine, and those are different behaviours.

        Honest: at ctag 1% the histogram is
            {'0': 24, '1-4': 21, '5-9': 5, 'short_other': 0, 'k': 0,
             'over_k': 0}
        -- 24 queries got nothing at all and 5 got most of the way, which the
        single figure 50 cannot express.
        Mutant: none needed; this asserts the strictly-greater information is
        present and consistent with the old field, which is the property a
        reader relies on when comparing an archived run to a new one.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.01,), columns=(FILTER_CTAG,),
               max_queries=N_QUERIES, warmup=0, indexed_rows=N_ROWS)

        extra = written(ctx)[0]["extra"]
        hist = extra["returned_count_hist"]
        assert hist == {"0": 24, "1-4": 21, "5-9": 5, "short_other": 0,
                        "k": 0, "over_k": 0}
        # The old field is exactly the buckets below k, so an archived
        # comparison still works.
        short = sum(v for key, v in hist.items()
                    if key not in ("k", "over_k"))
        assert short == extra["short_result_queries"] == 50
        assert hist["0"] > 0 and hist["5-9"] > 0, (
            "a histogram with one non-empty bucket proves nothing about the "
            "bucketing")

    @pytest.mark.parametrize("k, complete_bucket_should_hold", [(5, 50),
                                                                (10, 50),
                                                                (20, 50)])
    def test_a_complete_answer_is_complete_at_any_k(
            self, tmp_path, k, complete_bucket_should_hold):
        """The trap in item 4's literal four buckets: 0, 1-4, 5-9, k do not
        partition the integers for any k but 10.

        At k=5 a complete answer of 5 rows falls in "5-9" under the spec's
        reading and is filed as short. At k=20 a result of 12 rows falls in no
        bucket at all. So `k` is tested before the ranges and `short_other`
        catches 10..k-1.

        Honest, with a 50% predicate where the fake answers fully:
            k=5  -> {'k': 50}, '5-9' and 'short_other' both 0
            k=10 -> {'k': 50}
            k=20 -> {'k': 50}
        Mutant (ranges tested before `k`, the spec's literal order): at k=5
            the 50 complete answers land in '5-9' and 'k' is 0 -- the report
            would say no query was answered in full when every query was.
        """
        recorder = records.Recorder(str(tmp_path / f"ops-k{k}.jsonl"))
        cache = str(tmp_path / "cache")
        try:
            ctx = RunContext(run_id="t3c-k", resource_pass="normalized",
                             recorder=recorder, engine="fake",
                             engine_tag="none", march="x86-64-v3", k=k,
                             cache_dir=cache)
            dataset = tiny_dataset()
            driver = two_column_driver(dataset, cache, ef_half=0.0,
                                       ef_search=512)
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
                   selectivities=(0.50,), columns=(FILTER_TAG,),
                   max_queries=N_QUERIES, warmup=0, indexed_rows=N_ROWS)
        finally:
            recorder.close()

        hist = list(records.read_records(recorder.path))[0]["extra"][
            "returned_count_hist"]
        assert hist["k"] == complete_bucket_should_hold
        assert hist["5-9"] == 0
        assert hist["short_other"] == 0
        assert hist["over_k"] == 0

    @pytest.mark.parametrize("ef_search, expected", [
        # k=20 with a 32-candidate list: post-filtering a 50% predicate finds
        # 20 qualifying rows for 7 queries and 10..19 for the other 43.
        (32, {"0": 0, "1-4": 0, "5-9": 0, "short_other": 43, "k": 7,
              "over_k": 0}),
        # Tighter still, and three buckets move at once -- which is what the
        # single `short_result_queries` figure cannot express: 50 is 50 either
        # way, and these are two different engine behaviours.
        (16, {"0": 0, "1-4": 2, "5-9": 39, "short_other": 9, "k": 0,
              "over_k": 0}),
    ])
    def test_the_bucket_that_only_exists_above_k_ten_is_observed_moving(
            self, tmp_path, ef_search, expected):
        """`short_other` catches 10..k-1, so it can only be non-zero at k > 10
        -- and every other test here runs at a k or an ef where it is 0,
        which means it was asserted `== 0` and never seen to work.

        PR 3 §3.3 item 4 names four buckets. These 43 queries returned between
        10 and 19 rows of a requested 20, which is not 0, not 1-4, not 5-9 and
        not k: under the spec's literal reading they are dropped from the
        histogram entirely, and the counts would sum to 7 of 50.

        Honest: the distributions above, and they sum to 50.
        Mutant (`short_other` folded into `5-9`, i.e. `if n <= 9` widened to
            catch everything below k): at ef=32, `5-9` becomes 43 and
            `short_other` 0 -- the report would say 43 queries returned fewer
            than ten rows when every one of them returned at least ten.
        """
        recorder = records.Recorder(str(tmp_path / f"ops-ef{ef_search}.jsonl"))
        cache = str(tmp_path / "cache")
        try:
            ctx = RunContext(run_id="t3c-so", resource_pass="normalized",
                             recorder=recorder, engine="fake",
                             engine_tag="none", march="x86-64-v3", k=20,
                             cache_dir=cache)
            dataset = tiny_dataset()
            driver = two_column_driver(dataset, cache, ef_half=0.0,
                                       ef_search=ef_search)
            fw.run(ctx, driver, dataset, index_spec(dataset),
                   ef_search=ef_search, selectivities=(0.50,),
                   columns=(FILTER_TAG,), max_queries=N_QUERIES, warmup=0,
                   indexed_rows=N_ROWS)
        finally:
            recorder.close()

        row = list(records.read_records(recorder.path))[0]
        hist = row["extra"]["returned_count_hist"]
        assert hist == expected
        assert sum(hist.values()) == N_QUERIES
        assert hist["short_other"] > 0, (
            "this test exists to observe short_other non-zero")

    def test_a_driver_returning_more_than_k_is_counted_not_dropped(self, ctx):
        """Regression: a driver that stopped slicing `[:k]`.

        No correct driver returns more than k -- every one slices -- which is
        exactly why this bucket exists. If one stopped, every recall
        denominator would silently change and the four buckets item 4 names
        would drop those queries entirely.

        Honest: patching the driver to return k+1 ids puts all 8 queries in
            'over_k' and leaves 'k' at 0.
        Mutant (`over_k` removed and `n > k` folded into `k`): 'k' == 8, so
            the record claims every query was answered exactly and the
            histogram still sums correctly -- undetectable without this
            bucket.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=512)
        inner = driver.query_filtered

        def over_return(vector, k, threshold, column=FILTER_TAG):
            got = inner(vector, k + 4, threshold, column=column)
            return got[:k + 1]

        driver.query_filtered = over_return
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
               selectivities=(0.50,), columns=(FILTER_TAG,), max_queries=8,
               warmup=0, indexed_rows=N_ROWS)

        hist = written(ctx)[0]["extra"]["returned_count_hist"]
        assert hist["over_k"] == 8
        assert hist["k"] == 0
        assert sum(hist.values()) == 8


# ---------------------------------------------------------------------------
# §3.4 -- the time-bounded loop
# ---------------------------------------------------------------------------

class TestTheBudgetStopsTheLoop:

    def test_the_only_wall_clock_the_budget_reads_is_time_time(self):
        """What makes `TickingClock` a valid instrument rather than a trick.

        If `LatencyCollector` or `Progress` read `time.time()`, a fake
        `time.time` would move QPS and the percentiles too and the
        deterministic budget tests below would be measuring their own patch.
        Asserted at source level because the alternative is trusting a
        reading of three files.
        """
        for module in ("metrics/latency.py", "progress.py"):
            with open(os.path.join(VB_ROOT, "harness", module)) as fh:
                source = fh.read()
            assert "time.time()" not in source, module
            assert "perf_counter" in source, module
        # And the budget does read it, in the shape churn.py uses.
        with open(os.path.join(VB_ROOT, "harness", "workloads",
                               "filtered.py")) as fh:
            assert "time.time() - started > budget_s" in fh.read()

    def test_a_spent_budget_stops_the_loop_at_an_exact_query_count(self, ctx):
        """§3.4 items 1-2, without a wall-clock race.

        §3.5 asks for "about 10 queries run" with `latency_s=0.1` and
        `budget_s=1`, which is a timing assertion on a shared machine. With
        the clock advanced 0.1 s per query and by nothing else the count is
        arithmetic: the check is `elapsed > budget_s`, so after 10 queries
        elapsed is exactly 1.0, which is not greater than 1.0, and the 11th
        runs before the 12th check stops it.

        Honest: queries_executed == 11 of 50 planned, time_bounded True.
        (50, not 200: `queries` is `dataset.test[:max_queries]` and
        `tiny_dataset` ships 50 test vectors, so the corpus is the binding
        limit on `queries_planned` and the budget is the binding limit on
        `queries_executed`. Both are asserted, because a planned count equal
        to the executed one would mean the budget never bit.)
        Mutant (the budget check deleted from the query loop): **50
            executed** against the honest 11, and `time_bounded` **False**
            against True. No test count is given, deliberately: that figure
            was stated as 6, then 8, and measures 7, because an unbounded
            loop moves the executed count every budget test reads and a
            number that goes stale between rounds is worse than the values
            either side of it. Every other mutant note in this file names
            values for the same reason; the harness prints the counts.
        Mutant (the check left in the warm-up loop only, the realistic
            half-built state): the same 50, because the warm-up is 0 here and
            the loop that matters is unguarded.

        `>=` for `>` does NOT change this count, and an earlier version of
        this docstring claimed it did. At a step of 0.1 the elapsed time after
        ten queries is 0.9999999999999999 -- `0.1` is not representable in
        binary and ten additions of it fall short of 1.0 -- so neither
        comparison fires and both forms run 11 queries. The boundary needs a
        step that sums exactly, which is what the test below uses.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        clock = TickingClock(driver, step=0.1)
        with mock.patch.object(fw.time, "time", clock):
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
                   selectivities=(0.50,), columns=(FILTER_TAG,),
                   max_queries=200, warmup=0, indexed_rows=N_ROWS,
                   budget_s=1.0)

        row = written(ctx)[0]
        assert row["queries_executed"] == 11
        assert row["extra"]["queries_planned"] == N_QUERIES == 50
        assert row["extra"]["time_bounded"] is True
        assert row["extra"]["filtered_budget_s"] == 1.0

    def test_the_query_that_exactly_fills_the_budget_still_runs(self, ctx):
        """The `>` boundary, at a step the float arithmetic can land on.

        0.25 is a dyadic rational and exact in binary, so four queries
        accumulate to exactly 1.0 -- verified in the assertion below rather
        than asserted of the language -- and the comparison at the fifth check
        reads `1.0 > 1.0`, which is false. So the fifth query runs and the
        sixth check stops the loop.

        This matters beyond pedantry: `>` spends up to one query more than the
        budget and `>=` spends up to one less, and for a filtered point at the
        0.9 QPS measured on MariaDB one query is 1.1 seconds against a 300 s
        budget. The direction is the choice -- a budget that is a ceiling on
        *starting* work rather than on finishing it -- and `churn.py:244` uses
        `>`, so this matches the one budget this repo already enforces.

        Honest: 5 executed.
        Mutant (`>=` for `>`): 4 -- killed here and nowhere else, which is
            why this test exists rather than being folded into the one above.
        """
        step, budget = 0.25, 1.0
        accumulated = 0.0
        for _ in range(4):
            accumulated += step
        assert accumulated == budget, (
            "this test needs a step whose repeated addition is exact; "
            f"4 x {step} came to {accumulated!r}")

        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64)
        clock = TickingClock(driver, step=step)
        with mock.patch.object(fw.time, "time", clock):
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
                   selectivities=(0.50,), columns=(FILTER_TAG,),
                   max_queries=50, warmup=0, indexed_rows=N_ROWS,
                   budget_s=budget)

        assert written(ctx)[0]["queries_executed"] == 5

    def test_the_real_clock_honours_the_budget_too(self, ctx):
        """The same property against `time.time()` itself, in a form that
        cannot flake in the failing direction.

        Each query sleeps `latency_s=0.1`, so n queries cost at least 0.1n
        seconds and the loop cannot get past 11 before a 1.0 s budget is
        spent. A loaded machine makes every query slower, so it can only push
        the count DOWN -- the upper bound is arithmetic and the lower bound of
        1 holds because the first check reads an elapsed of roughly zero. The
        series precedent is replacing a QPS tolerance with Little's law.

        Honest: 10 or 11 on an idle box, and the assertion holds for anything
            from 1 to 11.
        Mutant (budget check deleted): 60 executed -- outside the bound by
            fivefold, and 6 seconds of sleeping makes it obvious.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=64, latency_s=0.1)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.50,), columns=(FILTER_TAG,), max_queries=60,
               warmup=0, indexed_rows=N_ROWS, budget_s=1.0)

        row = written(ctx)[0]
        assert 1 <= row["queries_executed"] <= 11, row["queries_executed"]
        assert row["extra"]["time_bounded"] is True
        assert row["queries_executed"] < row["extra"]["queries_planned"]

    def test_recall_and_latency_describe_the_executed_queries_only(self, ctx):
        """§3.4 item 2's "compute recall and latency over executed queries
        only".

        Honest: with 11 of 50 queries run, recall equals `recall_at_k` over
            exactly those 11, and the histogram and the two group counts each
            sum to 11.
        Mutant (recall scored over `truth` for all 50 while `returned` holds
            11): `recall_at_k` zips to the shorter list and returns the same
            number, so recall alone cannot catch it -- the histogram total and
            the group totals are what pin it, which is why all three are
            asserted.
        Mutant (`returned` padded to 50 with empty lists): recall 0.22
            instead of 1.0, and the histogram gains 39 queries in bucket '0'
            that never ran.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=512)
        truth, filt = ds.filtered_truth_for(ctx.cache_dir, dataset, K,
                                            FILTER_TAG, 0.50,
                                            indexed_rows=N_ROWS)
        clock = TickingClock(driver, step=0.1)
        with mock.patch.object(fw.time, "time", clock):
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
                   selectivities=(0.50,), columns=(FILTER_TAG,),
                   max_queries=200, warmup=0, indexed_rows=N_ROWS,
                   budget_s=1.0)

        row = written(ctx)[0]
        executed = row["queries_executed"]
        assert executed == 11
        expected = [driver.query_filtered(q, K, filt.threshold,
                                          column=FILTER_TAG)
                    for q in dataset.test[:executed]]
        assert row["recall_at_k"] == round(
            ds.recall_at_k(expected, truth[:executed], K), 6)
        assert sum(row["extra"]["returned_count_hist"].values()) == executed
        assert (row["extra"]["near_pass_queries"]
                + row["extra"]["near_fail_queries"]) == executed

    def test_a_point_that_finishes_says_so_rather_than_staying_silent(
            self, ctx):
        """`time_bounded` is written on every record, not only truncated ones.

        Honest: False, present.
        Mutant (the key written only when the budget bit): absent, and a
            reader cannot tell "ran to completion" from "written by a harness
            with no budgets" -- which is precisely the ambiguity
            `ops.pgvector_iterative_scan` spent 69 records demonstrating.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.50,), columns=(FILTER_TAG,), max_queries=8,
               warmup=0, indexed_rows=N_ROWS, budget_s=3_600.0)

        extra = written(ctx)[0]["extra"]
        assert extra["time_bounded"] is False
        assert extra["insufficient_sample"] is True      # 8 queries < 50
        assert extra["queries_planned"] == 8

    def test_the_budget_covers_the_warm_up_but_the_warm_up_cannot_eat_it(
            self, ctx):
        """M-2, and the two properties have to hold together.

        The clock covers the warm-up, because otherwise the printed ceiling is
        the budget plus however long twenty cold queries took and the estimate
        prints it as a bound. At the 0.9 QPS measured on MariaDB at 1%
        selectivity those twenty queries are 22 s of a 300 s budget.

        But a SHARED deadline let the warm-up consume the whole budget and
        record nothing. That is live risk on this measurement, not a corner:
        the harness default is 20 warm-up queries, `main`'s budget is 300 s,
        so anything slower than 15 s a query yielded no measured queries at
        all -- and 1.1 s a query is already measured for `tag` at 1%, with
        `ctag` at 1% expected to be worse because that is what the correlated
        column exists to expose.

        Honest (budget 1.0 s, warmup 20, 0.1 s a query, share 0.2):
            3 warm-up queries, **8 measured**, recall 1.0.
        Mutant (`WARMUP_BUDGET_SHARE = 1.0`, the shared deadline):
            11 warm-up queries, **0 measured**, recall None.
        Honest at `main`'s real shape (300 s, warmup 20, 15 s a query):
            **16 measured**; mutant: **1**.

        And the ceiling is unchanged by the share, which is the other half:
        both arrangements spend at most one budget plus the single query
        already in flight when the deadline passes.
        """
        dataset = tiny_dataset()

        def measure(share, step=0.1, budget=1.0, warmup=20):
            driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                       ef_search=512)
            recorder = records.Recorder(
                os.path.join(ctx.cache_dir, f"w{share}-{step}.jsonl"))
            local = RunContext(
                run_id="w", resource_pass="normalized", recorder=recorder,
                engine="fake", engine_tag="none", march="x", k=K,
                cache_dir=ctx.cache_dir)
            clock = TickingClock(driver, step=step)
            try:
                with mock.patch.object(fw.time, "time", clock), \
                     mock.patch.object(fw, "WARMUP_BUDGET_SHARE", share):
                    fw.run(local, driver, dataset, index_spec(dataset),
                           ef_search=512, selectivities=(0.50,),
                           columns=(FILTER_TAG,), max_queries=50,
                           warmup=warmup, indexed_rows=N_ROWS,
                           budget_s=budget)
            finally:
                recorder.close()
            row = list(records.read_records(recorder.path))[0]
            return row, clock

        honest, honest_clock = measure(fw.WARMUP_BUDGET_SHARE)
        mutant, mutant_clock = measure(1.0)

        assert honest["queries_executed"] == 8
        assert honest["recall_at_k"] == 1.0
        assert mutant["queries_executed"] == 0
        assert mutant.get("recall_at_k") is None

        # `main`'s real shape, where this is the difference between a
        # measurement and nothing.
        big_honest, _ = measure(fw.WARMUP_BUDGET_SHARE, step=15.0,
                                budget=300.0)
        big_mutant, _ = measure(1.0, step=15.0, budget=300.0)
        assert big_honest["queries_executed"] == 16
        assert big_mutant["queries_executed"] == 1

        # The ceiling, unchanged: a point cannot START work after its budget,
        # so it spends at most the budget plus the one query in flight.
        for clock, budget, step in ((honest_clock, 1.0, 0.1),
                                    (mutant_clock, 1.0, 0.1)):
            spent = clock.calls * step
            assert spent <= budget + step + 1e-9, (spent, budget, step)
        assert honest_clock.calls == mutant_clock.calls == 11


class TestTheFloorStillWritesARecord:
    """§3.4 item 3."""

    def test_under_fifty_queries_is_recorded_and_flagged(self, ctx):
        """Regression: dropping the point, which hides the finding.

        A configuration too slow to answer fifty queries inside its budget is
        the measurement, not a failure to measure. Deleting the record leaves
        a gap in the chart that reads as "not run".

        Honest: a record exists, queries_executed == 11,
            insufficient_sample True, and recall is still a number.
        Mutant (`MIN_SAMPLE_QUERIES = 0`): insufficient_sample False, so the
            report draws an 11-query recall as a full measurement.
        Mutant (the record skipped when the sample is short): zero records,
            caught by the length assertion.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                   ef_search=512)
        clock = TickingClock(driver, step=0.1)
        with mock.patch.object(fw.time, "time", clock):
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=512,
                   selectivities=(0.50,), columns=(FILTER_TAG,),
                   max_queries=200, warmup=0, indexed_rows=N_ROWS,
                   budget_s=1.0)

        rows = written(ctx)
        assert len(rows) == 1
        assert rows[0]["queries_executed"] == 11 < fw.MIN_SAMPLE_QUERIES
        assert rows[0]["extra"]["insufficient_sample"] is True
        assert rows[0]["recall_at_k"] is not None

    def test_zero_executed_queries_still_writes_a_usable_record(self, ctx):
        """The degenerate end, which is where NaN gets into a record.

        `latency.summarize([])` returns NaN for every percentile, and
        `json.dumps` writes NaN unquoted -- invalid strict JSON that a
        non-Python reader refuses and that a chart plots as a gap.

        Honest: one record, queries_executed 0, recall_at_k absent or None,
            no latency percentiles, and the line parses as strict JSON.
        Mutant (`**point.stats.as_dict()` unguarded): the line contains
            `NaN`, which the strict parse below rejects.

        Reached with a tiny POSITIVE budget, not with 0: since M-1 a
        non-positive budget means "no cap" and runs every query, which is the
        documented meaning of `ops.filtered_budget_s: 0` and what the
        estimate has always assumed. One warm-up query runs (its deadline is
        0.2 x 0.001 s and the first check sees an elapsed of 0.0), moves the
        clock a whole second, and the measured loop never starts.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        clock = TickingClock(driver, step=1.0)
        with mock.patch.object(fw.time, "time", clock):
            fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
                   selectivities=(0.50,), columns=(FILTER_TAG,),
                   max_queries=20, warmup=5, indexed_rows=N_ROWS,
                   budget_s=0.001)

        rows = written(ctx)
        assert len(rows) == 1
        assert rows[0]["queries_executed"] == 0
        assert rows[0].get("recall_at_k") is None
        assert rows[0].get("latency_p99_ms") is None
        assert rows[0]["extra"]["insufficient_sample"] is True
        assert rows[0]["extra"]["time_bounded"] is True
        with open(ctx.recorder.path) as fh:
            for line in fh:
                json.loads(line, parse_constant=_no_constants)

    def test_the_warm_up_records_how_much_of_itself_it_ran(self, ctx):
        """L-4, and this test exists because its mutant SURVIVED.

        The share means the warm-up can be cut short, and `time_bounded` is
        deliberately not set for it -- so a reader of a truncated point can
        see THAT the budget bit but not WHICH loop it bit. That matters
        because the warm-up exists to pay for cold graph caches: a warm-up cut
        to a quarter of its length means the first measured queries are partly
        cold, on exactly the slow points the budget exists for.

        Honest at `main`'s shape (300 s budget, warmup 20, 15 s a query):
            `warmup_executed` **5**, `warmup_planned` **20** -- 75% short,
            with the warm-up deadline at 0.2 x 300 = 60 s and checks landing
            at elapsed 0, 15, 30, 45 and 60, all of which pass `> 60`.
        Honest when nothing truncates: 2 and 2.
        Mutant (both keys not written, which is how they shipped before this
            test): the keys are absent, so a truncated warm-up is recorded
            nowhere -- 39 mutants and this was the only survivor.
        """
        dataset = tiny_dataset()

        def measure(budget, warmup, step):
            driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                       ef_search=512)
            recorder = records.Recorder(
                os.path.join(ctx.cache_dir, f"wu{budget}-{warmup}-{step}.jsonl"))
            local = RunContext(
                run_id="wu", resource_pass="normalized", recorder=recorder,
                engine="fake", engine_tag="none", march="x", k=K,
                cache_dir=ctx.cache_dir)
            clock = TickingClock(driver, step=step)
            try:
                with mock.patch.object(fw.time, "time", clock):
                    fw.run(local, driver, dataset, index_spec(dataset),
                           ef_search=512, selectivities=(0.50,),
                           columns=(FILTER_TAG,), max_queries=50,
                           warmup=warmup, indexed_rows=N_ROWS,
                           budget_s=budget)
            finally:
                recorder.close()
            return list(records.read_records(recorder.path))[0]["extra"]

        truncated = measure(300.0, 20, 15.0)
        assert truncated["warmup_executed"] == 5
        assert truncated["warmup_planned"] == 20
        # The point is greyed anyway, but the record now says which loop.
        assert truncated["time_bounded"] is True

        whole = measure(3_600.0, 2, 0.1)
        assert whole["warmup_executed"] == whole["warmup_planned"] == 2
        assert whole["time_bounded"] is False

        # And `warmup_planned` is the number the workload could run, not the
        # number asked for: a 50-query corpus cannot warm up 200 times.
        clamped = measure(3_600.0, 200, 0.1)
        assert clamped["warmup_planned"] == N_QUERIES == 50

    def test_a_non_positive_budget_means_no_cap_rather_than_no_queries(
            self, ctx):
        """M-1. `_filtered_search`'s docstring invited a reader to write
        `filtered_budget_s: 0` or `null` "to turn the cap off". Neither
        worked, and the docstring is what found it.

        `0` emitted `--filtered-budget 0`, whose first elapsed-time check
        fires immediately, so every point recorded ZERO queries -- recall
        null, `time_bounded` true, `insufficient_sample` true. `null` emitted
        the string `'None'` and argparse killed the whole ops unit with
        `SystemExit(2)`, taking concurrency and churn with it, because
        `harness/main.py` wraps all three in one `except`.

        The estimate was the honest half of the pair all along: it treats
        `budget > 0` as the only bounded case, so it never invented a ceiling
        for either spelling. The workload now agrees with it.

        Honest: budget 0 -> 50 of 50 executed, recall 1.0,
            `time_bounded` False, and the DECLARED 0.0 recorded (not `inf`,
            which is neither valid JSON nor a value any profile wrote).
        Mutant (`effective_budget = budget_s`, i.e. no conversion -- the
            shipped behaviour): 0 executed, recall None, `time_bounded` True.
        """
        dataset = tiny_dataset()
        for declared in (0.0, -1.0):
            driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                       ef_search=512)
            recorder = records.Recorder(
                os.path.join(ctx.cache_dir, f"nocap{declared}.jsonl"))
            local = RunContext(
                run_id="n", resource_pass="normalized", recorder=recorder,
                engine="fake", engine_tag="none", march="x", k=K,
                cache_dir=ctx.cache_dir)
            clock = TickingClock(driver, step=0.1)
            try:
                with mock.patch.object(fw.time, "time", clock):
                    fw.run(local, driver, dataset, index_spec(dataset),
                           ef_search=512, selectivities=(0.50,),
                           columns=(FILTER_TAG,), max_queries=50, warmup=20,
                           indexed_rows=N_ROWS, budget_s=declared)
            finally:
                recorder.close()
            row = list(records.read_records(recorder.path))[0]
            assert row["queries_executed"] == 50, declared
            assert row["recall_at_k"] == 1.0, declared
            assert row["extra"]["time_bounded"] is False, declared
            assert row["extra"]["filtered_budget_s"] == declared

    def test_both_documented_off_spellings_reach_the_workload_as_no_cap(self):
        """The other half of M-1: the orchestrator's end.

        Honest: `filtered_budget_s: 0` and `filtered_budget_s: null` both
            emit `--filtered-budget 0`, which `main.py` parses to 0.0 and the
            workload reads as no cap.
        Mutant (the `str(ops.get(...))` idiom restored): `null` emits the
            string `'None'` and `parse_args` raises SystemExit(2) -- asserted
            below, because the failure is a dead ops unit rather than a wrong
            number.

        `ops.churn_budget_s` shared the hole verbatim and T4a closed it, both
        halves at once: the emission here, through the same `_budget_arg`, and
        the enforcement in `churn.run`, which converts a non-positive budget
        into an infinite deadline. Both halves were needed -- fixing the
        emission alone would have moved the failure out of argparse and into a
        workload whose first deadline check fires immediately, which is the
        shape that made every filtered point record ZERO queries. The last
        assertion below is churn's end of that, kept in this test rather than
        moved because the two keys must not drift apart again: a reader adding
        a third budget key should find both of them pinned in one place.
        """
        from harness.main import parse_args
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        resources = load_resources("normalized")

        def emitted(value):
            ops = {"workloads": ["filtered"]}
            if value is not _ABSENT:
                ops["filtered_budget_s"] = value
            argv = ops_pass.harness_args({"name": "t", "k": 10, "ops": ops},
                                         16, "mariadb", None, "normalized",
                                         resources)
            return argv[argv.index("--filtered-budget") + 1]

        assert emitted(0) == "0"
        assert emitted(None) == "0"
        assert emitted(_ABSENT) == "300"
        assert emitted(45) == "45"

        # The value survives argparse and arrives as a float.
        for raw, want in (("0", 0.0), ("300", 300.0), ("45", 45.0)):
            args = parse_args([
                "--engine", "e", "--host", "h", "--port", "1",
                "--dataset", "d", "--m", "16", "--run-id", "r",
                "--output", "/tmp/o", "--filtered-budget", raw])
            assert args.filtered_budget == want

        # And the shape that used to kill the unit.
        with pytest.raises(SystemExit) as caught:
            parse_args([
                "--engine", "e", "--host", "h", "--port", "1",
                "--dataset", "d", "--m", "16", "--run-id", "r",
                "--output", "/tmp/o", "--filtered-budget", "None"])
        assert caught.value.code == 2

        # churn's hole, fixed by T4a: both halves, emission and enforcement.
        churn = ops_pass.harness_args(
            {"name": "t", "k": 10,
             "ops": {"workloads": ["churn"], "churn_budget_s": None}},
            16, "mariadb", None, "normalized", resources)
        assert churn[churn.index("--churn-budget") + 1] == "0", (
            "churn_budget_s: null must emit '0' (no cap), not the string "
            "'None' that argparse rejects with SystemExit(2); T4a fixed the "
            "emission here and the enforcement in churn.run")


# ---------------------------------------------------------------------------
# §3.4 item 4 / F-4 -- the iterative-scan loop
# ---------------------------------------------------------------------------

class TestTheIterativeScanModesAreLooped:

    def test_the_whole_grid_is_measured_once_per_declared_mode(self, ctx):
        """F-4: the loop belongs here rather than in `cli.py`, because looping
        there re-runs build, concurrency and churn per mode and doubles
        pgvector's ingest -- hours at 1M rows -- to vary a query-time setting.

        Honest: 2 modes x 1 column x 2 selectivities = 4 records, stamped
            ['off', 'off', 'relaxed_order', 'relaxed_order'], and the driver
            was asked to change mode twice, in order.
        Mutant (only the first mode applied, which is what the chain does
            today end to end): 2 records, both 'off'.
        Mutant (mode set per point rather than per mode-pass): 4 records with
            the same stamps but `iterative_scan_calls` of length 4 -- caught
            by the call list, which is why it is asserted and not just the
            stamps.
        """
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.10, 0.50), columns=(FILTER_TAG,),
               max_queries=8, warmup=0, indexed_rows=N_ROWS,
               iterative_scan="off,relaxed_order")

        rows = written(ctx)
        assert [r["extra"]["iterative_scan"] for r in rows] == [
            "off", "off", "relaxed_order", "relaxed_order"]
        assert driver.iterative_scan_calls == ["off", "relaxed_order"]

    def test_a_second_mode_costs_no_second_brute_force(self, tmp_path,
                                                       monkeypatch):
        """The cost claim the mode factor in `estimate._filtered_search` rests
        on: a second mode re-measures the grid and recomputes NOTHING.

        It holds because `datasets.filtered_truth_cache_name` keys the exact
        ground truth on (dataset, metric, k, column, threshold, rows, buckets,
        clusters, seed) and the mode is correctly absent -- the mode changes
        what the engine returns, not what the right answer is. That is a
        property of a key in another module, so nothing here failed when it was
        true and nothing would fail if it stopped being: adding `mode` to that
        key would silently turn one brute force per point into one per point
        per mode, on the workload whose ground truth is already named in
        `docs/07-planning-a-run.md` as an unmodelled cost outside
        `ops.filtered_budget_s`.

        Each arm gets its own cache directory, so both start cold and the
        comparison is between two whole runs rather than between a cold pass
        and a warm one.

        Honest: 1 mode writes 2 records and 2 modes write 4, off the SAME 2
            brute forces -- the second mode's two points are cache hits.
        Mutant (`mode` added to `filtered_truth_cache_name`'s key, or the
            cache bypassed): 4 computations for 2 modes, so the second mode
            costs a second exact search over the qualifying rows and the
            estimate's "+1 grid, no ingest" is wrong by a brute force per
            point.
        """
        def run_with(modes, name):
            calls = []
            real = ds.masked_ground_truth

            def counting(*a, **kw):
                calls.append(1)
                return real(*a, **kw)

            recorder = records.Recorder(str(tmp_path / f"ops-{name}.jsonl"))
            try:
                run_ctx = RunContext(
                    run_id=f"t3c-{name}", resource_pass="normalized",
                    recorder=recorder, engine="fake", engine_tag="none",
                    march="x86-64-v3", k=K,
                    cache_dir=str(tmp_path / f"cache-{name}"))
                monkeypatch.setattr(ds, "masked_ground_truth", counting)
                dataset = tiny_dataset()
                driver = two_column_driver(dataset, run_ctx.cache_dir,
                                           ef_half=0.0)
                fw.run(run_ctx, driver, dataset, index_spec(dataset),
                       ef_search=64, selectivities=(0.10, 0.50),
                       columns=(FILTER_TAG,), max_queries=8, warmup=0,
                       indexed_rows=N_ROWS, iterative_scan=modes)
                monkeypatch.undo()
            finally:
                recorder.close()
            return len(calls), len(written(run_ctx))

        one_bf, one_rows = run_with("off", "one")
        two_bf, two_rows = run_with("off,relaxed_order", "two")

        # The records double...
        assert (one_rows, two_rows) == (2, 4)
        # ...and the brute forces do not move at all.
        assert one_bf == two_bf == 2, (one_bf, two_bf)

    def test_a_comma_string_and_a_sequence_are_the_same_request(self, ctx):
        """`harness/main.py` hands over `args.iterative_scan` verbatim, so the
        workload has to accept the only shape a command line can carry. A
        sequence is what a direct caller passes.

        Honest: both forms give ['off', 'relaxed_order'].
        Mutant (the `isinstance(x, str)` branch removed): a string is iterated
            character by character and the modes become
            ['o', 'f', 'f', ',', 'r', ...] -- 15 SETs of nonsense, each of
            which a real PostgreSQL would reject.
        """
        dataset = tiny_dataset()
        for form in ("off,relaxed_order", ["off", "relaxed_order"],
                     ("off", "relaxed_order")):
            driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
            assert fw._scan_modes(form, driver) == ["off", "relaxed_order"], form

    def test_a_repeated_mode_is_measured_once(self, ctx):
        """A profile declaring `[off, off]` asks for one measurement."""
        dataset = tiny_dataset()
        driver = two_column_driver(dataset, ctx.cache_dir, ef_half=0.0)
        assert fw._scan_modes("off,off,relaxed_order,off", driver) == [
            "off", "relaxed_order"]

    def test_a_driver_without_the_knob_measures_the_grid_once(self, ctx):
        """Regression: doubling MariaDB's filtered cost for free.

        Only pgvector has the knob. A driver without it must not run the grid
        once per mode, because every pass would produce byte-identical
        measurements -- and filtered search is the longest workload in the
        profile.

        Honest: 1 record, `iterative_scan` None, on a driver with the method
            removed.
        Mutant (the `hasattr` gate dropped from `_scan_modes`): 2 records and
            an AttributeError on `set_iterative_scan` -- or, with the gate
            kept but the modes returned anyway, 2 identical records labelled
            'off' and 'relaxed_order' by an engine that has neither.
        """
        dataset = tiny_dataset()

        class NoKnob:
            """Every driver method except the one knob.

            A delegating proxy rather than a FakeDriver subclass, because an
            inherited method cannot be deleted from a subclass and patching
            it to None leaves `hasattr` True -- which is the one thing this
            test needs to be False. MariaDB, AliSQL, Percona Search and
            Valkey are all in this shape: no `set_iterative_scan` at all.
            """

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "set_iterative_scan":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        driver = NoKnob(two_column_driver(dataset, ctx.cache_dir, ef_half=0.0,
                                          ef_search=64))
        assert not hasattr(driver, "set_iterative_scan")
        assert driver.name == "fake"          # everything else delegates

        fw.run(ctx, driver, dataset, index_spec(dataset), ef_search=64,
               selectivities=(0.50,), columns=(FILTER_TAG,), max_queries=8,
               warmup=0, indexed_rows=N_ROWS,
               iterative_scan="off,relaxed_order")
        rows = written(ctx)
        assert len(rows) == 1
        assert rows[0]["extra"]["iterative_scan"] is None

    def test_the_real_driver_still_implements_what_the_gate_asks_for(self):
        """The fake now has `set_iterative_scan`, so it can no longer prove
        that `PostgresDriver` does. Asserted directly, and the name is read
        out of the gate so a rename has to happen on both sides."""
        import re

        from harness.drivers.postgres import PostgresDriver
        with open(os.path.join(VB_ROOT, "harness", "workloads",
                               "filtered.py")) as fh:
            gate = re.search(r'hasattr\(driver,\s*"([A-Za-z_]\w*)"\)', fh.read())
        assert gate, "filtered.py no longer gates on hasattr(driver, ...)"
        assert callable(getattr(PostgresDriver, gate.group(1), None))
        assert callable(getattr(FakeDriver, gate.group(1), None))


# ---------------------------------------------------------------------------
# F-10 -- the config keys have to reach the command line
# ---------------------------------------------------------------------------

class TestTheProfileKeysReachTheHarness:
    """F-10. `ops.pgvector_iterative_scan` was declared in three config files,
    plumbed the whole way to a `SET`, and passed by nothing -- 69 of 69
    archived records read `iterative_scan: null` -- because `_run_unit` had no
    test anywhere in the repo. Every key added here gets one.
    """

    def _argv(self, profile_ops, engine="pgvector"):
        from orchestrator import ops_pass
        from orchestrator.config import load_resources
        return ops_pass.harness_args(
            {"name": "t", "k": 10, "ops": profile_ops}, 16, engine, None,
            "normalized", load_resources("normalized"))

    def test_the_budget_and_the_columns_are_emitted(self):
        """Honest: `--filtered-budget 300` and `--filter-columns tag,ctag`.
        Mutant (either flag not emitted): the workload falls back to its own
        default and a profile that set the key is silently ignored, which is
        the state `ops.pgvector_iterative_scan` was in."""
        argv = self._argv({"workloads": ["filtered"]})
        assert argv[argv.index("--filtered-budget") + 1] == "300"
        assert argv[argv.index("--filter-columns") + 1] == "tag,ctag"

    def test_a_profile_overrides_both(self):
        """Proves the value is read from the profile and not pinned."""
        argv = self._argv({"workloads": ["filtered"], "filtered_budget_s": 45,
                           "filter_columns": ["ctag"]})
        assert argv[argv.index("--filtered-budget") + 1] == "45"
        assert argv[argv.index("--filter-columns") + 1] == "ctag"

    def test_an_empty_column_list_means_both_not_none(self):
        """`filter_columns: []` is falsy in YAML. Treated as "none" it would
        measure nothing while the phase still reported success."""
        argv = self._argv({"workloads": ["filtered"], "filter_columns": []})
        assert argv[argv.index("--filter-columns") + 1] == "tag,ctag"

    def test_several_scan_modes_are_comma_joined_and_one_is_verbatim(self):
        """The carriage §3.4 item 4 needs. A single string must pass through
        unchanged, because `cli.py:_ops_iterative_scan` comma-joins the modes
        it resolves and hands the result over as one string -- so this
        function sees `"off,relaxed_order"` from a two-mode profile and
        `"off"` from a one-mode profile, and must not re-split or re-join
        either. The list branch is still exercised by hand-built callers and
        by the tests below.

        Mutant (`",".join(iterative_scan)` without the str branch): "off"
            becomes "o,f,f".
        """
        argv = self._argv({"workloads": ["filtered"],
                           "pgvector_iterative_scan": ["off"]})
        single = self._argv({"workloads": ["filtered"]})
        assert "--iterative-scan" not in single
        from orchestrator import ops_pass
        assert ops_pass._iterative_scan_modes("off") == "off"
        assert ops_pass._iterative_scan_modes(
            ["off", "relaxed_order"]) == "off,relaxed_order"
        assert ops_pass._iterative_scan_modes([]) is None
        assert argv is not None

    def test_a_raw_yaml_sequence_is_named_not_strd(self):
        """The YAML 1.1 trap on the path `cli.py` does not take.

        `iterative_scan: [off, relaxed_order]` parses as
        `[False, "relaxed_order"]`, and the sequence branch used plain `str`,
        so `str(False)` made it `"False,relaxed_order"` and the workload
        issued `SET hnsw.iterative_scan = False` -- a value pgvector does not
        have, which the server rejects, which costs the whole ops unit because
        `harness/main.py` wraps concurrency, filtered and churn in one
        `except`. That is the exact defect `_yaml_mode_name` exists to
        prevent, and it survived on this branch because the one production
        caller names and comma-joins its modes first and so reaches the string
        branch. Nothing guarded it.

        Honest: a raw YAML sequence comes back `"off,relaxed_order"`, and a
            bare `on` -- the other keyword YAML 1.1 steals -- comes back
            `"on"` rather than `"True"`.
        Mutant (`str(mode)` restored in the sequence branch):
            `"False,relaxed_order"` and `"True"`.
        """
        from orchestrator import ops_pass
        assert ops_pass._iterative_scan_modes(
            [False, "relaxed_order"]) == "off,relaxed_order"
        assert ops_pass._iterative_scan_modes([True]) == "on"
        # The empty-entry filter survives the change: a blank is still dropped
        # rather than emitted as an empty mode between two commas.
        assert ops_pass._iterative_scan_modes(["off", "", "strict_order"]) == (
            "off,strict_order")

    def test_run_unit_puts_the_flags_on_the_real_command_line(self, tmp_path):
        """The link F-10 names: not `harness_args` in isolation, and not the
        resolver in isolation, but `_run_unit` calling the real
        `harness_args` and handing the result to the runner.

        No container and no network: `OpsRun` is a fake whose `run_harness`
        records the argv and returns 0.

        Honest: the argv `_run_unit` would have executed carries
            `--filtered-budget 300` and `--filter-columns tag,ctag`.
        Mutant (the flags added to `harness_args` but `_run_unit` passing a
            hand-built list, which is how a plumbed-but-unused key happens):
            the flags are absent from what the runner receives while every
            other test in this class still passes.
        """
        import argparse

        from orchestrator import cli as cli_mod
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        seen = []

        class FakeRun:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_harness(self, args, output, **kw):
                seen.append(list(args))
                return 0

        import yaml
        profile = yaml.safe_load(
            open(os.path.join(VB_ROOT, "config", "profiles", "main.yml")))
        paths = {"run_dir": str(tmp_path),
                 "checkpoints": str(tmp_path / "checkpoints")}
        with mock.patch.object(ops_pass, "OpsRun", FakeRun):
            rc = cli_mod._run_unit(
                "ops", "pgvector", "glove-100-angular", profile, {},
                load_resources("normalized"), None, "normalized", paths,
                "run-0", argparse.Namespace(force=False), checkpoints=None)
        assert rc == 0
        assert seen, "run_harness was never reached"
        for argv in seen:
            assert argv[argv.index("--filtered-budget") + 1] == "300"
            assert argv[argv.index("--filter-columns") + 1] == "tag,ctag"
            # main.yml restored the 1% point; §3.4 item 4.
            assert argv[argv.index("--selectivities") + 1] == "0.01,0.1,0.5"

    def test_main_hands_the_budget_and_the_columns_to_the_workload(self):
        """The middle link, and two mutants survived until this existed.

        `tests/test_estimate.py` checks this link for `iterative_scan` by
        reading the source, "because running it needs a server". It does not:
        patching `load_dataset` and `get_driver` is enough to execute
        `harness.main.main` end to end against the in-process fake, which
        tests the wiring rather than its spelling. A source-level check passes
        on `budget_s=args.filtered_budget` appearing anywhere in the call; this
        one fails unless the value arrives.

        Honest: the workload is called with columns=['ctag'] and budget_s=7.0,
            read off the captured kwargs.
        Mutant (`budget_s=args.filtered_budget` deleted from the call in
            main.py): budget_s is the workload's own 300.0 default, so a
            profile that lowered it is silently ignored -- the exact shape of
            the `ops.pgvector_iterative_scan` defect, and it SURVIVED every
            other test in this file.
        Mutant (`columns=_strings(args.filter_columns)` deleted): columns is
            ('tag', 'ctag'), so `--filter-columns ctag` is ignored and the run
            costs twice what the profile asked for. Also survived everything
            else.
        """
        from harness import main as main_mod

        dataset = tiny_dataset()
        captured = {}

        def spy(ctx, driver, ds_, index, **kwargs):
            captured.update(kwargs)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            driver = two_column_driver(dataset, os.path.join(tmp, "cache"),
                                       ef_half=0.0, ef_search=64)
            with mock.patch.object(main_mod, "load_dataset",
                                   lambda *a, **kw: dataset), \
                 mock.patch.object(main_mod, "get_driver",
                                   lambda *a, **kw: driver), \
                 mock.patch.object(main_mod.filtered_workload, "run", spy):
                rc = main_mod.main([
                    "--engine", "fake", "--host", "h", "--port", "1",
                    "--dataset", "tiny", "--m", "16", "--run-id", "r",
                    "--output", os.path.join(tmp, "ops.jsonl"),
                    "--cache-dir", os.path.join(tmp, "cache"),
                    "--workloads", "filtered", "--ef-mode", "fixed",
                    "--max-queries", "8",
                    "--filtered-budget", "7", "--filter-columns", "ctag",
                ])

        assert rc == 0, captured
        assert captured["columns"] == ["ctag"]
        assert captured["budget_s"] == 7.0
        # And the argument the chain already had, so this test also covers the
        # link `test_estimate.py` reads for.
        assert captured["iterative_scan"] is None

    def test_the_two_declarations_of_each_default_agree(self):
        """`orchestrator/` runs on the host, whose only Python dependency is
        pyyaml, so it cannot import the workload to read its defaults -- the
        same reason `DEFAULT_EF_MODE` is declared twice and
        `tests/test_calibrate.py` pins the pair. This is that pin for the two
        keys T3c adds.

        Mutant (either default changed on one side only): the run would be
        bounded at one number and the estimate would print another.
        """
        from orchestrator import ops_pass
        assert ops_pass.DEFAULT_FILTERED_BUDGET_S == fw.DEFAULT_BUDGET_S
        assert tuple(ops_pass.DEFAULT_FILTER_COLUMNS) == tuple(FILTER_COLUMNS)
        assert tuple(ops_pass.DEFAULT_FILTER_COLUMNS) == tuple(
            fw.DEFAULT_COLUMNS)

    def test_main_yml_declares_what_the_spec_asks_for(self):
        """PR 3 §3.4 item 4, as configuration."""
        import yaml
        ops = yaml.safe_load(
            open(os.path.join(VB_ROOT, "config", "profiles",
                              "main.yml")))["ops"]
        assert ops["selectivities"] == [0.01, 0.10, 0.50]
        assert ops["filtered_budget_s"] == 300


# ---------------------------------------------------------------------------
# The churn term, folded in from T4a: the same two defects, two functions over
# ---------------------------------------------------------------------------

class TestTheChurnTermHandlesAnUncappedBudgetToo:
    """`ops.churn_budget_s: null` and `: 0` in `estimate._churn_reinsert`.

    Found by T4a, which fixed the emission in `ops_pass` and the enforcement
    in `churn.run` and could not fix this third site because `orchestrator/`
    is not in its ownership. Both are the defect class closed for
    `_filtered_search` in the previous round, so the two functions now agree.

    **No configured profile is affected.** Only `tuned-complete.yml` sets
    `churn_budget_s`, to 7200. Every published figure and every
    `query_worst_case_is_a_ceiling` was compared across all nine profiles at
    three engines and six, one pass and two -- 36 configurations, byte
    identical before and after.
    """

    PROFILE = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                    "m_values": [16],
                                    "churn_fractions": [0.10, 0.25]}}

    def _estimate(self, budget=_ABSENT, terms=None):
        ops = dict(self.PROFILE["ops"])
        if budget is not _ABSENT:
            ops["churn_budget_s"] = budget
        profile = {"name": "t", "ops": ops}
        import contextlib
        ctx = (mock.patch.object(est_mod, "_QUERY_TERMS", terms) if terms
               else contextlib.nullcontext())
        with ctx:
            return estimate_load_hours(profile, ["alisql"],
                                       ["glove-100-angular"], ["normalized"],
                                       ["ops"])

    def _term(self, estimate):
        found = [t for t in estimate["query_terms"]
                 if t["name"] == "churn re-insert"]
        return found[0] if found else None

    @staticmethod
    def _old_form(plan):
        """`_churn_reinsert` exactly as it stood before this fix.

        Two differences, both of them the defect: `float(...)` without the
        `or 0.0`, and the `worst <= 0` guard that drops the term.
        """
        if "churn" not in plan.workloads or plan.rows <= 0:
            return None
        fractions = [float(f) for f in plan.ops.get("churn_fractions", [0.1])]
        budget = float(plan.ops.get("churn_budget_s", 1800))
        rate = plan.write_rows_per_s
        typical = worst = 0.0
        rows = 0
        churned_so_far = 0.0
        for fraction in fractions:
            step = max(0.0, fraction - churned_so_far)
            if step <= 0:
                continue
            count = int(round(step * plan.rows))
            if count == 0:
                continue
            churned_so_far = fraction
            rows += count
            typical += min(count / rate, budget) if rate > 0 else budget
            worst += budget
        if worst <= 0:
            return None
        return est_mod._QueryTerm(name="churn re-insert", typical_s=typical,
                                  worst_s=worst, basis="old form",
                                  bounded=True)

    def _with_old_form(self):
        return (est_mod._concurrency_windows, est_mod._filtered_search,
                self._old_form)

    def test_rebinding_the_module_attribute_does_not_reach_the_term(self):
        """The injection hazard, measured before it is relied on.

        `_QUERY_TERMS` is a module-level tuple built at import time holding
        the ORIGINAL function objects, so `mock.patch.object(est_mod,
        "_churn_reinsert", ...)` changes a name nothing reads. A probe built
        that way reports a green suite while measuring unmutated code, which
        is how three estimate mutants elsewhere came back clean.

        Honest: rebinding the attribute leaves the estimate at
            `typ=1.64 worst=1.64 bounded=False` -- byte identical to no
            mutation at all, which is the proof the rebinding reached
            nothing.
        The tuple rewrite below is therefore the only valid injection, and
        every mutant in this class uses it.
        """
        unmutated = self._term(self._estimate(0))
        with mock.patch.object(est_mod, "_churn_reinsert", self._old_form):
            rebound = self._term(self._estimate(0))
        assert rebound is not None
        assert (rebound["typical_hours"], rebound["worst_hours"],
                rebound["bounded"]) == (unmutated["typical_hours"],
                                        unmutated["worst_hours"],
                                        unmutated["bounded"])
        assert rebound["typical_hours"] == 1.64

        # And the tuple rewrite DOES reach it: same input, different answer.
        injected = self._estimate(0, terms=self._with_old_form())
        assert self._term(injected) is None

    def test_a_null_budget_no_longer_kills_the_estimate_on_the_host(self):
        """Defect 1. This ran on the host before any container started, so a
        profile with that one key printed a traceback instead of an estimate.

        Honest: `typ=1.64 worst=1.64 bounded=False`, the term present.
        Mutant (`float(...)` without `or 0.0`, injected through the tuple):
            `TypeError: float() argument must be a string or a real number,
            not 'NoneType'`.
        """
        term = self._term(self._estimate(None))
        assert term is not None
        assert (term["typical_hours"], term["worst_hours"]) == (1.64, 1.64)
        assert term["bounded"] is False

        with pytest.raises(TypeError, match="NoneType"):
            self._estimate(None, terms=self._with_old_form())

    def test_an_uncapped_budget_is_reported_unbounded_not_dropped(self):
        """Defect 2, and the dropped term took the ceiling claim with it.

        Honest: the term is present at `typ=1.64 worst=1.64`, `bounded`
            False, and `query_worst_case_is_a_ceiling` is **False** -- the
            estimate says it cannot bound the run, which is true.
        Mutant (the `worst <= 0` guard, injected through the tuple): the term
            is **absent**, `query_terms` is `[]`, and
            `query_worst_case_is_a_ceiling` is **True** -- because `all()`
            over an empty list is True. So the old form did not merely hide a
            cost: it claimed a ceiling with nothing behind it, on a profile
            whose only measured cost it had just discarded.

        That is the same shape `_filtered_search` reports for the same input,
        which is the point of the change.
        """
        honest = self._estimate(0)
        term = self._term(honest)
        assert term is not None
        assert (term["typical_hours"], term["worst_hours"]) == (1.64, 1.64)
        assert term["bounded"] is False
        assert honest["query_worst_case_is_a_ceiling"] is False
        assert "not a ceiling" in term["basis"]

        mutant = self._estimate(0, terms=self._with_old_form())
        assert self._term(mutant) is None
        assert [t["name"] for t in mutant["query_terms"]] == []
        assert mutant["query_worst_case_is_a_ceiling"] is True

    def test_the_two_budget_terms_agree_on_what_off_means(self):
        """The consistency the fix is for: one input, two terms, one answer.

        Both read a budget key, both default it, and both now treat `0` and
        `null` as "no cap" and report an unbounded term rather than raising
        or vanishing.
        """
        profile = {"name": "t", "k": 10,
                   "ops": {"enabled": True,
                           "workloads": ["filtered", "churn"],
                           "m_values": [16], "max_queries": 500,
                           "selectivities": [0.10, 0.50],
                           "churn_fractions": [0.10, 0.25],
                           "filtered_budget_s": 0, "churn_budget_s": 0}}
        e = estimate_load_hours(profile, ["alisql"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        by_name = {t["name"]: t for t in e["query_terms"]}
        # Both present: neither raised and neither vanished, which were the
        # two old behaviours.
        assert {"filtered search", "churn re-insert"} <= set(by_name)
        for name in ("filtered search", "churn re-insert"):
            assert by_name[name]["bounded"] is False, name
            # Asserted on the cap CLAUSE rather than on a phrase, because the
            # two unbounded branches word themselves differently and
            # legitimately -- filtered says "no budget anywhere in the run to
            # cap it", churn says "turns the cap off". What must not appear
            # is either bounded branch's concrete ceiling.
            basis = by_name[name]["basis"]
            assert "cap/point" not in basis, name
            assert "cap/step" not in basis, name
        assert e["query_worst_case_is_a_ceiling"] is False

    def test_the_capped_cases_are_untouched(self):
        """The regression guard for the two spellings that DO ship.

        `absent` (the 1800 default) and `tuned-complete`'s 7200 must read
        exactly as before, which is what the 36-configuration comparison
        showed and what this pins in the suite.
        """
        # Ops measurement version 9 moved the default from 1800 to 7200, so
        # `absent` now reads exactly as `tuned-complete`'s explicit 7200 --
        # and the old form, which hard-codes 1800, still reads 1.0 h for it:
        # that difference is the bump, deliberately.
        default = self._term(self._estimate())
        assert (default["typical_hours"], default["worst_hours"]) == (1.64, 4.0)
        assert default["bounded"] is True

        tuned = self._term(self._estimate(7200))
        assert (tuned["typical_hours"], tuned["worst_hours"]) == (1.64, 4.0)
        assert tuned["bounded"] is True
        assert "7200s cap/step" in tuned["basis"]

        old = self._term(self._estimate(7200, terms=self._with_old_form()))
        assert (old["typical_hours"], old["worst_hours"]) == (1.64, 4.0)
        old_absent = self._term(self._estimate(terms=self._with_old_form()))
        assert (old_absent["typical_hours"],
                old_absent["worst_hours"]) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _filtered_record(engine="mariadb", column="tag", selectivity=0.10,
                     p50=10.0, recall=0.9, scan=None, **extra):
    base = {
        "run_id": "r", "engine": engine, "phase": "filtered",
        "dataset": "glove-100-angular", "timestamp": "2026-01-01T00:00:00Z",
        "resource_pass": "normalized", "build_mode": "post",
        "storage_engine": "InnoDB", "m": 16, "ef_search": 100, "k": 10,
        "selectivity": selectivity, "recall_at_k": recall, "qps": 100.0,
        "clients": 1, "latency_p50_ms": p50, "latency_p99_ms": p50 * 2,
        "vector_index_used": True, "queries_executed": 500,
        "extra": {"filter_column": column, "iterative_scan": scan,
                  "short_result_queries": 0, "queries_planned": 500,
                  "time_bounded": False, "insufficient_sample": False,
                  **extra},
    }
    return base


class TestTheReportKeepsTheTwoColumnsApart:

    def test_the_strategy_ratio_is_never_taken_across_two_columns(self):
        """A defect found while reading, not one the brief handed over.

        `summarize` derived each engine's filtering strategy from the
        narrowest and widest point of a series keyed on (engine, dataset). Two
        columns in one series means the narrow end can be a `ctag` point and
        the wide end a `tag` one, so the latency ratio divides two different
        predicates over two different qualifying sets -- and the report
        publishes the result as what that engine did. The report has already
        been wrong about exactly this once, asserting for weeks that Valkey
        chose its strategy per query.

        The fixture makes the mixing visible: `tag` gets slower as it narrows
        (post-filter, ratio 4.0) and `ctag` gets faster (pre-filter, ratio
        0.25). Keyed correctly that is two rows with opposite verdicts.

        Honest: 2 strategies, ratios {'tag': 4.0, 'ctag': 0.25}, verdicts
            'filters during the graph walk' (4.0 is slower-when-narrower but
            under the 10x exhaustive threshold) and 'pre-filter'.
        Mutant (the key reverted to (engine, dataset)): 1 strategy, ratio 0.5,
            verdict 'pre-filter' -- a single confident claim about an engine,
            derived from a pairing that never ran, and the `tag` behaviour
            gone entirely.
        """
        rows = [
            _filtered_record(column="tag", selectivity=0.01, p50=40.0),
            _filtered_record(column="tag", selectivity=0.50, p50=10.0),
            _filtered_record(column="ctag", selectivity=0.02, p50=5.0),
            _filtered_record(column="ctag", selectivity=0.51, p50=20.0),
        ]
        summary = generate.summarize(rows)
        got = {s["filter_column"]: s for s in summary["filter_strategies"]}
        assert set(got) == {"tag", "ctag"}
        assert got["tag"]["ratio"] == 4.0
        assert got["ctag"]["ratio"] == 0.25
        assert got["tag"]["observed"] == "filters during the graph walk"
        assert got["ctag"]["observed"] == "pre-filter"
        # Opposite verdicts, which is the point: one pooled series could only
        # ever have produced one of them.
        assert got["tag"]["observed"] != got["ctag"]["observed"]

    #: Two units differing in exactly one axis, and the latencies that make
    #: pooling them publish a THIRD verdict. Unit B is emitted first so that
    #: the stable sort by selectivity puts B's narrow point and A's wide point
    #: at the two ends of a pooled series -- which is what makes the pooled
    #: ratio a number neither unit produced rather than a copy of one of them.
    OPS_UNIT_AXES = {
        "m": (16, 32),
        "build_mode": ("post", "incremental"),
        "storage_engine": ("InnoDB", "MyISAM"),
        "resource_pass": ("normalized", "tuned"),
    }

    def _two_units(self, axis):
        a, b = self.OPS_UNIT_AXES[axis]
        rows = []
        for unit, narrow_p50, wide_p50 in ((b, 100.0, 50.0), (a, 2.0, 10.0)):
            for sel, p50 in ((0.01, narrow_p50), (0.50, wide_p50)):
                r = _filtered_record(column="tag", selectivity=sel, p50=p50,
                                     recall=0.95)
                r[axis] = unit
                rows.append(r)
        return [rows[0], rows[2], rows[1], rows[3]]

    def _published_strategies(self, summary):
        """The (Change, Observed strategy) cells of the rendered table.

        Read out of the Markdown the report publishes, not off the summary
        dict: the requirement is that a *published* ratio and a *published*
        verdict move, and a dict key that never reached a table would satisfy
        a weaker test.
        """
        rendered = []
        for line in render._filter_strategy_table(summary).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 12 and cells[-1] not in ("Observed strategy",
                                                      "---"):
                rendered.append((cells[-4], cells[-1]))
        return rendered

    @pytest.mark.parametrize("axis", list(OPS_UNIT_AXES))
    def test_the_ratio_is_never_taken_across_two_ops_units(self, axis):
        """H-1. The ops phase loops M x build mode x storage engine inside each
        resource pass, and every pass through `_run_unit` reloads the corpus,
        rebuilds the index and writes its own full column x selectivity grid.
        Two points from two units never ran together, so dividing their
        latencies and publishing the result as an engine's filtering strategy
        is the same defect as pooling the two filter columns -- on four more
        axes, all of them reachable: `main` gives pgvector 2 units per (pass,
        dataset), `full` gives mariadb 3 and pgvector 6, and
        `--resource-pass both` adds the pass axis for every engine.

        Honest, for every one of the four axes -- two rows:
            [('0.2x', 'pre-filter'),
             ('2.0x', 'filters during the graph walk')]
        Mutant (that axis dropped from `OPS_UNIT_FIELDS` at run time) -- one
        row:
            [('10.0x', 'post-filter, exhaustive')]
        a ratio and a verdict NEITHER unit produced, obtained by dividing one
        unit's 1% latency (100.0 ms) by the other's 50% latency (10.0 ms).

        The mutation rewrites the module-level tuple `ops_unit()` reads on
        every call, and is proven by the published value moving rather than by
        the patch having been applied -- a rebinding that the code never
        re-reads changes nothing, which is how three estimate mutants came
        back green while measuring nothing.
        """
        rows = self._two_units(axis)
        honest = self._published_strategies(generate.summarize(rows))
        assert honest == [("0.2x", "pre-filter"),
                          ("2.0x", "filters during the graph walk")], honest

        original = generate.OPS_UNIT_FIELDS
        try:
            generate.OPS_UNIT_FIELDS = tuple(
                f for f in original if f != axis)
            mutant = self._published_strategies(generate.summarize(rows))
        finally:
            generate.OPS_UNIT_FIELDS = original

        assert mutant == [("10.0x", "post-filter, exhaustive")], mutant
        # The values moved, which is the proof the axis is load-bearing.
        assert {r for r, _ in mutant}.isdisjoint({r for r, _ in honest})
        assert {v for _, v in mutant}.isdisjoint({v for _, v in honest})
        # And the honest run is not vacuous: the two units really do disagree.
        assert honest[0][1] != honest[1][1]

    #: Loop variable in `cli._run_unit` -> the record field that names the
    #: axis it varies. The two spellings differ for storage, which is why this
    #: map exists rather than a set comparison.
    LOOPED_AXIS_FIELDS = {"m": "m", "build_mode": "build_mode",
                          "storage": "storage_engine"}

    #: The one ops-unit axis that is NOT a loop in `_run_unit`, because the
    #: pass axis sits above it: `cmd_run`'s `for resource_pass in passes:`
    #: loop calls `_run_unit` from inside its body, so a unit is already
    #: single-pass by the time that function runs. The mapping from loops to
    #: fields is therefore deliberately not one-to-one, and this names the
    #: exception instead of the pin pretending otherwise.
    #:
    #: **BY SYMBOL BECAUSE THE LINE NUMBER WAS WRONG.** This carried a line
    #: number until #47 moved the loop out from under it. At `c8de6a5` the
    #: loop is `c8de6a5:orchestrator/cli.py:508`, and the line the comment
    #: went on naming, `c8de6a5:orchestrator/cli.py:454`, holds
    #: `print(f"engines   : ...")` -- a line that exists, is well inside the
    #: file and says nothing about resource passes. Both numbers carry the
    #: commit in front so that they stay true and `git show` can settle
    #: them; the live reference is the name.
    #:
    #: That is the shape of this defect and the reason a range check cannot
    #: find it: every line citation in this tree's comments resolves to a
    #: real line, and the wrong ones are wrong about the *content*, not the
    #: arithmetic. Checking content would need a table saying what each
    #: citation claims is there -- which is what `CITED_IN_REPO` gave
    #: `docs/09` and `docs/10` until both were converted to symbol citations,
    #: and what a comment has nothing of either way. Those two documents cite
    #: symbols now and their guards key on `(path, symbol)`; a comment still
    #: has nowhere to put the claim, which is why this one is a prose note and
    #: not a row.
    #:
    #: #47's commit body lists the citations it re-derived and names three
    #: more it checked and found unmoved. This one it moved and listed in
    #: neither, which is how a change whose subject was citation rot shipped
    #: a new instance of it. The name survives the next such edit;
    #: `tests/test_e16_citation_ratchet.py` refuses the next new number.
    PASS_LEVEL_AXIS = "resource_pass"

    def _looped_axes(self):
        """The axes `_run_unit` iterates, by AST rather than by substring.

        A loop counts when the call that launches the unit is INSIDE its body.
        That is what "the ops phase iterates this" means, and it is what tells
        the three real axes apart from `for stale in (output, memory_ts):`,
        which is a file cleanup two lines above the launch and matches any
        pattern that just looks for `ast.For`.
        """
        import ast
        with open(os.path.join(VB_ROOT, "orchestrator", "cli.py")) as fh:
            tree = ast.parse(fh.read())
        run_unit = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "_run_unit")
        axes = set()
        for node in ast.walk(run_unit):
            if not isinstance(node, ast.For):
                continue
            launches = any(
                isinstance(inner, ast.Call)
                and getattr(inner.func, "attr", None) in ("harness_args",
                                                          "run_harness")
                for inner in ast.walk(node))
            if launches:
                axes.add(ast.unparse(node.target))
        return axes

    def test_every_axis_the_ops_phase_loops_is_in_the_key(self):
        """The direction the previous version of this test was NAMED for and
        did not guard.

        It asserted a literal set equality against `OPS_UNIT_FIELDS` plus the
        presence of three fixed strings in `cli.py`. Neither can notice a
        FOURTH loop: adding a genuine new looped axis to `_run_unit` --

            quantizations = list(ops_cfg.get('quantizations', [None]))
            for quantization in quantizations:

        -- leaves all three strings present and `OPS_UNIT_FIELDS` unchanged,
        so the test passed and the whole suite stayed green while every
        filtered comparison silently pooled two quantizations. That is the
        same shape as H-1 itself, unguarded in the one place that would catch
        the next instance of it.

        Honest: the AST finds exactly {'m', 'build_mode', 'storage'} enclosing
            the unit launch, which maps onto three of the four
            `OPS_UNIT_FIELDS`, the fourth being the pass-level axis.
        Mutant (a fourth looped axis injected around the launch): the AST
            finds {'m', 'build_mode', 'storage', 'quantization'} and the set
            equality below fails naming `quantization` -- proven by on-disk
            injection into a copied tree, since this reads a file rather than
            an object and no runtime patch can move it.

        The opposite direction -- a field REMOVED from `OPS_UNIT_FIELDS` --
        is caught by the mapping assertions here and, better, by the four
        parameterised behaviour tests above, which fire on the published ratio
        rather than on a literal.
        """
        looped = self._looped_axes()
        assert looped == set(self.LOOPED_AXIS_FIELDS), (
            f"`cli._run_unit` iterates {sorted(looped)} around the unit "
            f"launch, but this test knows about "
            f"{sorted(self.LOOPED_AXIS_FIELDS)}. A new looped axis is a new "
            f"way for two filtered points that never ran together to land in "
            f"one series: add it to LOOPED_AXIS_FIELDS and to "
            f"report.generate.OPS_UNIT_FIELDS.")

        fields = {self.LOOPED_AXIS_FIELDS[name] for name in looped}
        assert fields <= set(generate.OPS_UNIT_FIELDS), (
            f"looped but not in the key: "
            f"{sorted(fields - set(generate.OPS_UNIT_FIELDS))}")
        assert set(generate.OPS_UNIT_FIELDS) - fields == {
            self.PASS_LEVEL_AXIS}, (
            f"the key holds {sorted(set(generate.OPS_UNIT_FIELDS) - fields)} "
            f"beyond the looped axes; only {self.PASS_LEVEL_AXIS!r} is "
            f"expected, because the pass axis is iterated above _run_unit.")

        # The pass axis really is above `_run_unit`, so the asymmetry above is
        # a fact about the code and not an excuse.
        with open(os.path.join(VB_ROOT, "orchestrator", "cli.py")) as fh:
            assert "for resource_pass in passes:" in fh.read()

    def test_the_axis_pin_notices_a_fourth_loop(self, tmp_path):
        """The mutant for the test above, run rather than described.

        `_looped_axes` reads `cli.py` off disk, so the injection has to be on
        disk too -- into a COPY, in a directory that has never been imported.
        A same-size rewrite of the real file inside one second can leave stale
        bytecode valid, which is why the real tree is never touched.

        The fourth loop is inserted by AST transformation: the storage loop's
        body becomes the body of a new `for quantization in quantizations:`,
        which is exactly what adding a real axis does and is guaranteed to
        parse. A text splice leaves the new loop's body empty and raises
        IndentationError, which would "kill" the mutant for the wrong reason.

        Honest: the pin sees {'m', 'build_mode', 'storage'} and passes.
        Mutant: it sees those plus 'quantization' and fails naming it --
            while `cli.py` still parses and all three original loop
            statements are still present, which is precisely why the
            substring form this replaced could not tell the two files apart.
        """
        import ast
        import copy

        source = os.path.join(VB_ROOT, "orchestrator", "cli.py")
        with open(source) as fh:
            tree = ast.parse(fh.read())

        run_unit = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "_run_unit")
        storage_loop = next(
            n for n in ast.walk(run_unit)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "storage")

        mutant_tree = copy.deepcopy(tree)
        mutant_run_unit = next(n for n in ast.walk(mutant_tree)
                               if isinstance(n, ast.FunctionDef)
                               and n.name == "_run_unit")
        mutant_storage = next(
            n for n in ast.walk(mutant_run_unit)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "storage")
        mutant_storage.body = [ast.For(
            target=ast.Name(id="quantization", ctx=ast.Store()),
            iter=ast.Name(id="quantizations", ctx=ast.Load()),
            body=mutant_storage.body, orelse=[], type_comment=None)]
        ast.fix_missing_locations(mutant_tree)

        mutant_path = tmp_path / "cli_mutant.py"
        mutant_path.write_text(ast.unparse(mutant_tree))

        # Read back off disk, and confirm the mutant is the *subtle* kind: it
        # parses, and every string the old pin looked for survives.
        back = mutant_path.read_text()
        ast.parse(back)
        for loop in ("for m in m_values:", "for build_mode in build_modes:",
                     "for storage in storage_engines:"):
            assert loop in back, loop
        assert "for quantization in quantizations:" in back

        def looped_in(text):
            parsed = ast.parse(text)
            fn = next(n for n in ast.walk(parsed)
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "_run_unit")
            axes = set()
            for node in ast.walk(fn):
                if isinstance(node, ast.For) and any(
                        isinstance(i, ast.Call)
                        and getattr(i.func, "attr", None) in ("harness_args",
                                                              "run_harness")
                        for i in ast.walk(node)):
                    axes.add(ast.unparse(node.target))
            return axes

        with open(source) as fh:
            honest = looped_in(fh.read())
        mutant = looped_in(back)

        assert honest == {"m", "build_mode", "storage"}, honest
        assert mutant == {"m", "build_mode", "storage", "quantization"}, mutant
        assert mutant - honest == {"quantization"}

        # What the pin does with each: passes on one, fails on the other.
        assert honest == set(self.LOOPED_AXIS_FIELDS)
        assert mutant != set(self.LOOPED_AXIS_FIELDS)
        # The live pin agrees with the honest reading, so the two halves of
        # this guard cannot drift apart.
        assert self._looped_axes() == honest
        # And the loop really does enclose the launch, which is what makes it
        # an axis rather than the `for stale in ...` cleanup beside it.
        assert "stale" not in honest

    def test_the_filtered_table_can_tell_two_ops_units_apart(self):
        """H-1's other direction: the rendered table, not the derivation.

        Two units produce two rows with identical engine, dataset, column and
        selectivity. Without the unit columns `full` renders mariadb three
        indistinguishable rows per (column, selectivity) and pgvector six --
        verbatim the "four real measurements rendered as the same row four
        times and read as a rendering fault" that the short-results block
        beside it cites as its own reason for carrying pass and build mode.

        **The fixture is this test's own, and that is the fix for a claim that
        used to be false.** It previously reused `_two_units`, which gives the
        two units DIFFERENT latencies because the ratio test needs them to --
        so the rendered rows stayed distinct whatever the unit columns did,
        `len(set(body))` was 4 under honest and mutant alike, and the kill
        actually came from the `"| post |"` assertion below rather than from
        the row-string comparison the docstring credited. The two purposes
        genuinely conflict in one helper, so this one uses records identical
        in every rendered cell except the ops unit.

        Honest: 4 rows, all 4 distinct, differing only in `Build mode`.
        Mutant (the `Build mode` cell dropped from the row -- the pre-H-1
            shape): the post and incremental rows become byte-identical at
            each selectivity, so `len(set(body))` falls from **4 to 2**,
            which is now what the row-string comparison catches.
        """
        rows = []
        for build_mode in ("post", "incremental"):
            for sel in (0.01, 0.50):
                r = _filtered_record(column="tag", selectivity=sel, p50=10.0,
                                     recall=0.95)
                r["build_mode"] = build_mode
                rows.append(r)
        table = render._filtered_table(generate.summarize(rows))
        body = [line for line in table.splitlines()
                if line.startswith("| MariaDB")]
        assert len(body) == 4                      # 2 units x 2 selectivities
        assert len(set(body)) == 4, "two ops units render identically"
        assert "| post |" in table
        assert "| incremental |" in table
        # Every ops-unit axis is a column, so no two units can collide.
        for header in ("| Pass |", "| M |", "| Build mode |", "| Storage |"):
            assert header in table, header

    def test_the_m_column_sorts_numerically_not_as_text(self):
        """L-1. `full` sweeps `ops.m_values: [8, 16, 32]`, and as text that
        sorts 16, 32, 8 -- in both new tables, because both keyed on
        `str(m)`. `_build_table` in the same file already used `m or 0`.

        Presentation only, no number wrong. Honest: rows ordered 8, 8, 16,
        16, 32, 32 and strategies [8, 16, 32]. Mutant (`str(s["m"])` and
        `str(r.get("m"))` restored): rows 16, 16, 32, 32, 8, 8 and strategies
        [16, 32, 8].
        """
        rows = []
        for m in (32, 8, 16):                  # emitted out of order on purpose
            for sel, p50 in ((0.01, 2.0), (0.50, 10.0)):
                r = _filtered_record(column="tag", selectivity=sel, p50=p50,
                                     recall=0.95)
                r["m"] = m
                rows.append(r)
        summary = generate.summarize(rows)

        table = render._filtered_table(summary)
        rendered = [c.strip() for line in table.splitlines()
                    if line.startswith("| MariaDB")
                    for c in [line.strip().strip("|").split("|")[3]]]
        assert rendered == ["8", "8", "16", "16", "32", "32"], rendered

        assert [s["m"] for s in summary["filter_strategies"]] == [8, 16, 32]

    def test_two_scan_modes_are_not_pooled_either(self):
        """The same defect one axis over: pgvector with index iteration off
        and on walks different amounts of graph, which is the whole reason the
        mode is recorded.

        Honest: 2 strategies for one engine and one column.
        Mutant (mode dropped from the key): 1, mixing the two.
        """
        rows = [
            _filtered_record(engine="pgvector", selectivity=0.01, p50=40.0,
                             scan="off"),
            _filtered_record(engine="pgvector", selectivity=0.50, p50=10.0,
                             scan="off"),
            _filtered_record(engine="pgvector", selectivity=0.01, p50=5.0,
                             scan="relaxed_order"),
            _filtered_record(engine="pgvector", selectivity=0.50, p50=20.0,
                             scan="relaxed_order"),
        ]
        summary = generate.summarize(rows)
        modes = {s["iterative_scan"] for s in summary["filter_strategies"]}
        assert modes == {"off", "relaxed_order"}
        assert len(summary["filter_strategies"]) == 2

    def test_an_archived_record_without_a_column_reads_as_tag(self):
        """All 69 archived pgvector filtered records, and every record
        `tests/synthetic.py` builds, carry no `filter_column`: those runs had
        one column and it was `tag`.

        Honest: `filter_column` returns 'tag' and the strategy row says 'tag'.
        Mutant (returning None or 'unknown'): an archived run's series is
            keyed on None, so it still renders -- but its chart is named
            `filtered-None-<dataset>` and the strategy table's Column cell is
            blank, and a reader comparing it against a new run sees two
            different columns where there is one.
        """
        archived = [
            {**_filtered_record(selectivity=0.01, p50=40.0), "extra":
             {"tag_threshold": 1, "short_result_queries": 0}},
            {**_filtered_record(selectivity=0.50, p50=10.0), "extra":
             {"tag_threshold": 50, "short_result_queries": 0}},
        ]
        assert generate.filter_column(archived[0]) == "tag"
        summary = generate.summarize(archived)
        assert [s["filter_column"] for s in summary["filter_strategies"]] == [
            "tag"]
        assert generate._filter_columns_present(
            archived, "glove-100-angular") == ["tag"]

    def test_the_table_shows_two_decimals_so_the_shares_stay_distinct(self):
        """A vacuity shape with teeth: at `{:.0%}` a `tag` point at 0.0100 and
        a `ctag` point at 0.0190 both render as "2%", so the one difference
        the measured-share fix exists to expose is rounded away in the only
        place a reader sees it.

        Honest: the rendered table contains '1.00%' and '1.90%'.
        Mutant (`{:.0%}` kept): both cells read '2%' and the assertion on
            '1.90%' fails. Note the guard is written against the cell text
            with its delimiters, because '1.00%' is a substring of '11.00%'
            and this repo has lost four tests to exactly that.
        """
        rows = [_filtered_record(column="tag", selectivity=0.0100,
                                 nominal_selectivity=0.01),
                _filtered_record(column="ctag", selectivity=0.0190,
                                 nominal_selectivity=0.01)]
        table = render._filtered_table(generate.summarize(rows))
        assert "| 1.00% |" in table
        assert "| 1.90% |" in table
        assert "| 2% |" not in table
        # Both columns named, and the nominal kept beside the counted share.
        assert "| tag |" in table and "| ctag |" in table

    def test_a_truncated_point_is_listed_and_flagged(self):
        """PR 3's "greyed time-bounded points", in the Markdown half of the
        report. A recall over 93 of 500 queries carries that sample's
        variance, and nothing in the old table said which points those were.

        Honest: the point appears in `time_bounded_points` with 93/500 and
            `insufficient_sample` False, and the validity section names it.
        Mutant (`time_bounded` not read into the summary): the list is empty
            and the section is absent, while the record itself still says so
            -- the information reaches the JSONL and not the reader.
        """
        rows = [_filtered_record(column="ctag", selectivity=0.019,
                                 time_bounded=True, queries_planned=500),
                _filtered_record(column="tag", selectivity=0.01)]
        rows[0]["queries_executed"] = 93
        rows[0]["extra"]["filtered_budget_s"] = 300
        summary = generate.summarize(rows)
        assert len(summary["time_bounded_points"]) == 1
        point = summary["time_bounded_points"][0]
        assert (point["queries_executed"], point["queries_planned"]) == (93, 500)
        assert point["filter_column"] == "ctag"
        assert point["insufficient_sample"] is False

        section = render._validity_section({"warnings": []}, summary)
        assert "Filtered points the time budget stopped" in section
        assert "| 93/500 |" in section

    @needs_matplotlib
    def test_one_chart_per_column_is_drawn(self, tmp_path):
        """PR 3's "a column selector (or two charts)". Two charts, because the
        report is also rendered to Markdown.

        Honest: `filtered-tag-*` and `filtered-ctag-*` are both produced, and
            each holds only its own column's points.
        Mutant (`_column=column` default dropped from the lambda): both files
            are produced and both plot `ctag`, because every lambda closes
            over the same loop variable -- two filenames, one chart, and the
            `tag` curve silently gone. Caught below by drawing the charts with
            different point counts per column and checking the titles.
        """
        rows = [_filtered_record(column="tag", selectivity=0.01, p50=40.0),
                _filtered_record(column="tag", selectivity=0.50, p50=10.0),
                _filtered_record(column="ctag", selectivity=0.019, p50=5.0)]
        tag_only = charts.filtered(rows, "glove-100-angular", str(tmp_path),
                                   "filtered-tag-x", column="tag")
        ctag_only = charts.filtered(rows, "glove-100-angular", str(tmp_path),
                                    "filtered-ctag-x", column="ctag")
        assert tag_only and ctag_only
        for paths in (tag_only, ctag_only):
            assert os.path.exists(paths["svg"])
        with open(tag_only["svg"]) as fh:
            tag_svg = fh.read()
        with open(ctag_only["svg"]) as fh:
            ctag_svg = fh.read()
        # The titles name the column, so the two files are not the same chart.
        assert "independent of the vector space" in tag_svg
        assert "correlated with the vector space" in ctag_svg
        assert "independent of the vector space" not in ctag_svg

    def test_each_chart_entry_is_bound_to_its_own_column(self):
        """Regression: two filenames and one chart.

        Written as a lambda closing over the loop variable -- the obvious
        spelling -- every entry plots whichever column the comprehension
        finished on, so the `tag` curve vanishes from a report that still has
        the right number of figures with the right names. `functools.partial`
        binds the value at construction, so the mistake cannot be written;
        this asserts the binding rather than the spelling.

        Honest: entry names ['filtered-tag', 'filtered-ctag'] and the columns
            their callables actually pass are ['tag', 'ctag'].
        Mutant (`partial` replaced by `lambda *a, **kw: draw(*a,
            column=column, **kw)`): the names are unchanged and the columns
            become ['ctag', 'ctag'] -- which is why the columns are read back
            out of the calls and not inferred from the names.
        """
        rows = [_filtered_record(column="tag"),
                _filtered_record(column="ctag", selectivity=0.019)]
        seen = []

        def draw(records, dataset, out_dir=None, stem=None, column=None):
            seen.append(column)
            return None

        entries = generate.filtered_chart_entries(
            rows, "glove-100-angular", draw=draw)
        assert [name for name, _, _ in entries] == ["filtered-tag",
                                                    "filtered-ctag"]
        for _, fn, payload in entries:
            fn(*payload, out_dir="/tmp", stem="x")
        assert seen == ["tag", "ctag"]

    def test_a_run_with_no_filtered_records_keeps_the_skippable_entry(self):
        """The fallback. With no filtered records the registry keeps the
        single `filtered` name so the `needs` table matches it and the shared
        "not in this profile's workloads" skip line prints -- otherwise a
        profile that ran no filtered workload generates no chart and says
        nothing about why.

        Honest: one entry named exactly `filtered`, which `needs` resolves.
        Mutant (the `or [...]` fallback dropped): no entries, no log line, and
            a reader of the console output cannot tell "not configured" from
            "crashed".
        """
        entries = generate.filtered_chart_entries([], "glove-100-angular")
        assert [name for name, _, _ in entries] == ["filtered"]
        assert generate._filter_columns_present([], "glove-100-angular") == []
