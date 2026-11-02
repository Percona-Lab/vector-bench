"""End-to-end tests for the four ops workloads, run against FakeDriver.

The ops harness is the half of this framework that no test could reach. Build,
concurrency, filtered and churn all need a populated server, so the cheapest
check that any of them still works was a benchmark run — which meant they were
in practice checked by reading the code and by noticing, hours later, that a
chart was empty. `tests/fake_driver.py` removes that: the four workloads below
run end to end, against a 2,000-row in-process corpus, in well under a second.

What these tests are for is the plumbing between a driver and a record: that
each workload writes the phases it claims to write, that the numbers land in
the fields the report reads, and that the quantities which are supposed to move
together actually do. They do not measure any engine and the numbers in them
mean nothing outside this file.

The fake's own error model — the thing that makes a recall assertion here mean
anything — is calibrated in tests/test_fake_driver.py.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import sys
import threading
import time
import types

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds  # noqa: E402
from harness.datasets import Dataset  # noqa: E402
from harness.metrics import latency, records  # noqa: E402
from harness.workloads import build as build_workload  # noqa: E402
from harness.workloads import churn as churn_workload  # noqa: E402
from harness.workloads import churn_readers  # noqa: E402
from harness.workloads import concurrency as concurrency_workload  # noqa: E402
from harness.workloads import filtered as filtered_workload  # noqa: E402
from harness.workloads.context import RunContext  # noqa: E402

from .fake_driver import (DIM, K, N_QUERIES, N_ROWS,  # noqa: E402
                          FakeDriver, index_spec, loaded_driver,
                          measure_recall, tiny_dataset)

#: Queries in `wide_dataset`. 1,000 because that is the number §1.2's
#: sharding claim is about, and because the recall comparison needs it — see
#: the docstring there.
N_WIDE_QUERIES = 1_000


def wide_dataset(seed: int = 20261013) -> Dataset:
    """`tiny_dataset`'s corpus with 1,000 queries instead of fifty.

    Two things in this file need a query set far wider than `N_QUERIES`, and
    both would be misleading without it.

    **Sharding.** The claim is about 1,000 queries across 32 clients. At fifty
    queries each shard is one or two elements long, every client finishes its
    first pass almost immediately, and the disjointness that matters at 32
    clients is not exercised at all.

    **The recall comparison's error bar.** The fake keeps each true neighbour
    independently with probability p(ef), so recall over `q` queries is the
    mean of `q * k` Bernoulli draws. At p = 0.98 and q = 50 the standard error
    is 0.0198: an assertion that two independent measurements agree within
    0.01 would fail roughly half the time, and tuning the tolerance until it
    passed would be tuning away the only discrimination the test has. At
    q = 1,000 the standard error is 0.0014 and 0.01 is seven sigma. Measured
    over twenty runs the two sides differ by at most 0.0032.

    Query count is the only thing that moves: the corpus, the metric and the
    exact-top-k ground truth are `tiny_dataset`'s, computed the same way, so
    nothing about the fake's error model changes between the two.
    """
    base = tiny_dataset()
    rng = numpy.random.default_rng(seed)
    test = rng.random((N_WIDE_QUERIES, DIM), dtype=numpy.float32)
    train = base.train
    scores = ((test ** 2).sum(1)[:, None]
              - 2.0 * (test @ train.T)
              + (train ** 2).sum(1)[None, :])
    # 100 columns, the width ann-benchmarks' HDF5 files ship.
    neighbors = numpy.argsort(scores, axis=1)[:, :100].astype(numpy.int64)
    ordered = numpy.take_along_axis(scores, neighbors, axis=1)
    return Dataset(name=f"wide-{DIM}-euclidean", train=train, test=test,
                   neighbors=neighbors,
                   distances=numpy.sqrt(numpy.maximum(ordered, 0.0)),
                   metric=ds.EUCLIDEAN)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return tiny_dataset()


@pytest.fixture(scope="module")
def angular_dataset() -> Dataset:
    return tiny_dataset(metric=ds.ANGULAR)


@pytest.fixture(scope="module")
def wide() -> Dataset:
    return wide_dataset()


@pytest.fixture
def ctx(tmp_path):
    """A RunContext whose records and ground-truth cache land in tmp_path."""
    recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
    try:
        yield RunContext(
            run_id="t0-fake", resource_pass="normalized", recorder=recorder,
            engine="fake", engine_tag="none", march="x86-64-v3", k=K,
            cache_dir=str(tmp_path / "cache"),
        )
    finally:
        recorder.close()


def written(ctx: RunContext):
    """Every record the workload under test wrote, in order."""
    return list(records.read_records(ctx.recorder.path))


class TestBuildWorkload:
    def test_records_ingest_and_build_for_an_incremental_engine(self, ctx,
                                                                dataset):
        # `load_latency_s` buys a measurable ingest cost, so the equality
        # below is between two real numbers. The fake loads 2,000 x 16 rows in
        # about 0.3 ms and build.py rounds to milliseconds, so without it both
        # figures are literally 0.0 — measured 20 times out of 20 — and
        # `build_wall_s == ingest_wall_s` is 0.0 == 0.0.
        #
        # It has to be a knob inside the driver rather than a wrapper around
        # `driver.load`, and the reason is an asymmetry in build.py worth
        # knowing before it costs someone a red test: build.py times
        # `create_index` with a timer of its own, but takes the load cost from
        # the driver's self-reported `LoadResult.wall_seconds`. So the sibling
        # test below can wrap `create_index` and have the number move, while a
        # wrapper around `load` sleeps outside the measurement entirely and
        # the record still reads 0.0.
        #
        # The direction being covered matters: the invariant is that an
        # incrementally built engine's build cost IS its load cost, and the
        # fault to catch is that cost being recorded as the bulk-build timer,
        # which for such an engine never runs and is zero. MHNSW and VIDX
        # reported as having no build cost at all is precisely the
        # plausible-looking wrong number this repo opens by warning about. The
        # opposite direction — a bulk build misfiled as load time — is covered
        # by the sibling test below.
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            load_latency_s=0.02)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        ingest, built = written(ctx)
        assert ingest["phase"] == records.PHASE_INGEST
        assert built["phase"] == records.PHASE_BUILD
        assert ingest["rows"] == built["rows"] == N_ROWS
        assert ingest["ingest_rows_per_s"] > 0
        assert ingest["ingest_threads"] == 1
        # An engine that maintains its graph on INSERT has no separable build,
        # so the build cost recorded is the load cost.
        assert built["extra"]["separable_build"] is False
        assert built["extra"]["incremental_index"] is True
        assert ingest["ingest_wall_s"] >= 0.02, "the load cost was not recorded"
        assert built["build_wall_s"] == ingest["ingest_wall_s"]
        assert built["index_bytes"] > 0 and built["table_bytes"] > 0
        assert driver.create_index_calls == 0
        assert driver.count_rows() == N_ROWS
        # The provenance every record shares, so a new workload cannot drop it.
        assert ingest["engine"] == "fake"
        assert ingest["engine_version"] == "fake-engine 0.1"
        assert ingest["resource_pass"] == "normalized"
        assert ingest["schema_version"] == records.SCHEMA_VERSION

    def test_separates_a_bulk_build_when_the_engine_can(self, ctx, dataset):
        # A load and a bulk build that each cost measurable time, and
        # deliberately different amounts. The fake does both in microseconds
        # and build.py rounds to milliseconds, so without this "separable"
        # could not mean two different numbers rather than one number written
        # twice — and the load equality at the end would be 0.0 == 0.0. The
        # two costs differ so the figures stay apart after rounding.
        #
        # Note the two mechanisms, for the reason given in the test above:
        # the load cost is a driver knob because build.py believes the driver
        # about it, while the bulk build is wrapped because build.py times
        # that one itself.
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            capabilities={"incremental_index": False},
                            load_latency_s=0.03)
        driver.connect()

        instant_build = driver.create_index

        def deliberate_build(index):
            time.sleep(0.02)
            instant_build(index)

        driver.create_index = deliberate_build
        build_workload.run(ctx, driver, dataset, index_spec(dataset))

        ingest, built = written(ctx)
        assert driver.create_index_calls == 1
        assert built["extra"]["separable_build"] is True
        # The build cost is the bulk build, not the load; for the incremental
        # engine above it is the load cost and the two are equal by
        # construction. Getting these the wrong way round would report
        # pgvector's cheap bulk build as its load time and vice versa.
        assert built["build_wall_s"] >= 0.02
        assert ingest["ingest_wall_s"] >= 0.03
        assert built["build_wall_s"] != built["ingest_wall_s"]
        assert built["ingest_wall_s"] == ingest["ingest_wall_s"]

    def test_subset_rows_loads_only_part_of_the_corpus(self, ctx, dataset):
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
        driver.connect()
        build_workload.run(ctx, driver, dataset, index_spec(dataset),
                           subset_rows=500)
        assert driver.count_rows() == 500
        assert written(ctx)[0]["rows"] == 500


class TestConcurrencyWorkload:
    def test_records_one_point_per_client_count(self, ctx, dataset):
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1, 2), duration_s=0.1, warmup=2,
            max_queries=N_QUERIES,
        )

        rows = written(ctx)
        assert [r["clients"] for r in rows] == [1, 2]
        assert all(r["phase"] == records.PHASE_CONCURRENCY for r in rows)
        for row in rows:
            assert row["ef_search"] == 32
            assert row["qps"] > 0
            assert row["queries_executed"] > 0
            # The wall-clock basis of QPS, asserted rather than described.
            # With N busy clients the wall clock is shared, so throughput is
            # queries over elapsed time; summing per-query latencies instead
            # counts concurrent work N times and divides QPS by roughly N.
            #
            # `queries_executed / qps` reconstructs the window the workload
            # measured, because both figures come out of the same merge. On
            # the wall-clock basis that quotient IS the measured wall, which
            # is a `time.sleep(duration_s)` and therefore never shorter than
            # `duration_s`, only longer by whatever the scheduler adds. On a
            # summed-latency basis it is `clients` times longer.
            #
            # The bound is one-sided on purpose. All the noise lives above
            # `duration_s` and none below it, so a two-sided tolerance tests
            # the machine rather than the code: a 5% one failed at a 5.7%
            # sleep overshoot on a box running three pytest processes at once,
            # while the fault being guarded against is a whole factor of N.
            #
            # Deliberately not asserted here: scaling_efficiency. The honest
            # value for two clients on this fake is 0.56-0.72 and varies run
            # to run, because the query path is Python and holds the GIL — the
            # same range a summed-latency QPS would report. An assertion on it
            # would be flaky and would not discriminate.
            implied_window = row["queries_executed"] / row["qps"]
            # The floor is structural, not measured: the quotient IS the wall
            # handed to merge_collectors, and that wall is taken around the
            # sleep, so it cannot come in under `duration_s` while the basis
            # stays the wall clock. Over 80 loaded windows the closest it came
            # was 1.00048x. It is kept as a statement of the relation — if the
            # basis ever changes to something unrelated to the window, this is
            # the line that says so — but all the discrimination lives in the
            # upper bound below and in the cross-client-count ratio after the
            # loop. Making the floor live would mean recording the measured
            # wall in the record, which is a production change.
            assert implied_window >= row["extra"]["duration_s"] * 0.99
            assert implied_window < row["extra"]["duration_s"] * 1.5
            # Separately, and unrelated to the above: percentiles are ordered.
            assert row["latency_p99_ms"] >= row["latency_p50_ms"]
            assert row["extra"]["duration_s"] == 0.1
            assert row["extra"]["repeats"] == 1
        assert rows[0]["extra"]["scaling_efficiency"] == 1.0
        # The same invariant in the form that does not care how loaded the
        # machine is: both windows are the same requested length, so the
        # reconstructed window must not grow with the client count. A
        # summed-latency QPS would double it from one client to two.
        windows = [r["queries_executed"] / r["qps"] for r in rows]
        assert windows[1] < windows[0] * 1.5

    def test_repeats_record_every_sample(self, ctx, dataset):
        # The spread is the point: a single sample has no error bar, which is
        # how two engines differing by less than the noise came to be presented
        # as ordered.
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1,), duration_s=0.05, warmup=1,
            max_queries=N_QUERIES, repeats=3,
        )
        row = written(ctx)[0]
        assert row["extra"]["repeats"] == 3
        assert len(row["extra"]["qps_samples"]) == 3
        assert row["extra"]["qps_spread_pct"] >= 0

    def test_a_discarded_warm_up_window_precedes_each_client_count(
            self, ctx, dataset, monkeypatch):
        """Ops version 9 (F7): one window, same client count, same length,
        before the first measured repeat -- recorded, never selected.

        `full-20260825` spread +/-22-24% across repeats at one client from a
        cold first repeat, and the one-client point is the baseline every
        efficiency divides by. Staged by making the FIRST window of each
        client count ten times slower: the warm-up must absorb it, so no
        published sample and no median can carry it.

        Mutant (the warm-up window appended to `samples`): `repeats` reads 4,
        `qps_samples` holds the slow window and the spread explodes -- each
        asserted below.
        """
        real = concurrency_workload._one_window
        calls = []

        scored = []

        def staged(*args, **kwargs):
            window = real(*args, **kwargs)
            calls.append(args[2])
            scored.append(kwargs.get("score", True))
            if calls.count(args[2]) == 1:
                # The warm-up window: never scored for recall (#78 review),
                # and its recall fields say so.
                assert window.recall is None and window.recall_queries == 0
                assert window.recall_id_overlap is None
                window = window._replace(qps=window.qps / 10.0)
            return window

        monkeypatch.setattr(concurrency_workload, "_one_window", staged)
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1, 2), duration_s=0.05, warmup=1,
            max_queries=N_QUERIES, repeats=3)
        rows = written(ctx)
        # One extra window per client count, and it came first.
        assert calls == [1, 1, 1, 1, 2, 2, 2, 2]
        assert scored == [False, True, True, True] * 2
        for row in rows:
            extra = row["extra"]
            assert extra["repeats"] == 3
            assert len(extra["qps_samples"]) == 3
            assert extra["warmup_window_s"] == 0.05
            slow = extra["warmup_window_qps"]
            # Recorded, and never a candidate: every published sample beats
            # the slowed warm-up window by a wide margin.
            assert slow > 0
            assert min(extra["qps_samples"]) > slow * 3, extra
            assert row["qps"] > slow * 3

    def test_the_baseline_median_check_withholds_a_cold_efficiency(
            self, ctx, dataset, monkeypatch):
        """F2 item 3: `baseline_is_credible` wired in.

        Staged by inflating the one-client window's median 3x -- a baseline
        whose median got FASTER as clients were added, which a warm one
        cannot do. Honest: the two-client record carries no
        `scaling_efficiency`, says `baseline_not_credible`, and still carries
        the ratio that decided it. Mutant (the verdict ignored): an efficiency
        about 3x the truth is published.
        """
        real = concurrency_workload._one_window

        def cold(*args, **kwargs):
            window = real(*args, **kwargs)
            if args[2] == 1:
                merged = dict(window.merged)
                merged["latency_p50_ms"] = merged["latency_p50_ms"] * 3.0
                window = window._replace(merged=merged, qps=window.qps / 3.0)
            return window

        monkeypatch.setattr(concurrency_workload, "_one_window", cold)
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1, 2), duration_s=0.1, warmup=2,
            max_queries=N_QUERIES)
        one, two = written(ctx)
        assert one["extra"]["scaling_efficiency"] == 1.0
        assert two["extra"]["scaling_efficiency"] is None
        assert (two["extra"]["efficiency_reliability"]
                == concurrency_workload.EFFICIENCY_BASELINE_NOT_CREDIBLE)
        assert two["extra"]["baseline_p50_ratio"] > 1.4

    def test_every_point_says_which_recall_rule_it_carries(self, ctx,
                                                           dataset):
        """F8: `recall_at_k` is the distance threshold; the id overlap rides
        beside it under its own name."""
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1,), duration_s=0.1, warmup=2,
            max_queries=N_QUERIES)
        row = written(ctx)[0]
        assert row["extra"]["recall_kind"] == ds.RECALL_DISTANCE_THRESHOLD
        assert row["extra"]["recall_id_overlap"] is not None
        # No ties in a random float corpus, so the two rules agree here; the
        # tie that separates them is `test_harness.py`'s.
        assert row["recall_at_k"] == pytest.approx(
            row["extra"]["recall_id_overlap"], abs=1e-6)
        # And the window's wall-clock ends, for the server-sample join.
        assert (row["extra"]["window_ended_at"]
                > row["extra"]["window_started_at"] > 1e9)

    def test_a_client_count_above_the_query_count_is_skipped(self, ctx,
                                                             dataset):
        driver = loaded_driver(dataset, ef_half=64.0)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1, 8), duration_s=0.05, warmup=1, max_queries=4,
        )
        assert [r["clients"] for r in written(ctx)] == [1]


#: Per-query cost for the window-trimming tests, and the whole reason they can
#: say anything. The effect being measured is one in-flight query per client,
#: so it has to be a visible fraction of the window: at the fake's native
#: ~0.09 ms a two-second window holds twenty thousand queries per client and
#: one of them is 0.005% -- below every rounding in the record. At 50 ms it is
#: one in forty.
TRIM_LATENCY_S = 0.05

#: The spec's configuration for §1.3: eight clients, 50 ms queries, a
#: two-second window. Kept because that is the configuration the change was
#: specified against; what is asserted over it is not.
TRIM_CLIENTS = 8
TRIM_WINDOW_S = 2.0

#: A query slower than the whole window, which is the only way to reach a
#: window in which nothing finished. Six times `EMPTY_WINDOW_S` so the outcome
#: is a property of the arithmetic rather than of how loaded the host is: the
#: one query a client issues cannot complete before the cutoff however the
#: scheduler behaves, because `time.sleep` is a floor.
EMPTY_WINDOW_LATENCY_S = 0.3
EMPTY_WINDOW_S = 0.05


def strict_json_lines(path: str) -> int:
    """Parse every record in `path` the way a strict JSON parser would.

    `json.dumps` writes a NaN as the bare literal `NaN`, which is not JSON.
    Python's own `json.loads` accepts it as a documented extension, so a
    records file full of NaNs reads back perfectly here and in
    `report/loaders.py` while `JSON.parse` in the browser throws — one
    consumer silently fine and another broken, which is this repository's
    signature failure mode. `parse_constant` is the hook CPython calls for
    exactly the three non-standard literals, so this is the browser's verdict
    rather than an imitation of it.

    Returns the number of records parsed, so a caller cannot be reassured by
    an empty file.
    """
    def reject(literal: str):
        raise AssertionError(
            f"records file contains the non-standard JSON literal "
            f"{literal!r}, which JSON.parse rejects")

    seen = 0
    with open(path) as handle:
        for line in handle:
            if line.strip():
                json.loads(line, parse_constant=reject)
                seen += 1
    return seen


class TestConcurrencyCountsOnlyTheWindow:
    """After `stop` is set every client finishes the query it is holding and
    records it, but the wall clock stopped at `stop`.

    So QPS counted one query per client that its own denominator did not
    cover. The overstatement is that query over the ones the client did
    finish: `1 / queries-completed-per-client`, which is
    `inflight_discarded / queries_executed` in the record and about
    `latency / window`. THE CLIENT COUNT CANCELS -- each client contributes one
    spurious query and roughly `window / latency` real ones -- so it is the
    same at 1 client as at 32, measured flat at 2.564% across all six counts
    of a full sweep. What makes it bite is a slow engine over a short window:
    0.005% for a 1.4 ms query in a 30-second window, 2.5% for a 50 ms query in
    two seconds, whatever the concurrency.

    WHAT IS NOT ASSERTED HERE, and why. The specification for this change
    proposed "QPS within +/-1% of 8 / 0.05 = 160". That assertion is worse
    than flaky: it passes the bug and fails the fix. 160 q/s assumes a query
    costs exactly `latency_s`, but `time.sleep` only guarantees a floor, and on
    this machine a query costs 50.40 ms. Thirty-nine of them fit in two
    seconds, not forty, so the honest trimmed figure is ~156 q/s -- 2.5% below
    160 and outside the tolerance. The untrimmed code reports 159.98, because
    the fortieth query it wrongly counts is precisely what closes the gap.
    Measured here: trimmed, 155.94-155.98 q/s over nine idle runs and
    151.86-155.86 over nine at 3x oversubscription; untrimmed, 159.98-159.99
    over six idle runs. Every trimmed run is outside +/-1% and every untrimmed
    one is inside it. See
    `test_queries_in_flight_cannot_exceed_the_client_count` for what is
    pinned instead.
    """

    def _run(self, ctx, dataset, driver, clients, duration_s, **kw):
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(clients,), duration_s=duration_s, warmup=2,
            max_queries=concurrency_workload.WHOLE_TEST_SET,
            indexed_rows=N_ROWS, **kw)
        return written(ctx)[-1]

    def _driver(self, dataset):
        return loaded_driver(dataset, ef_half=64.0, ef_search=32,
                             latency_s=TRIM_LATENCY_S)

    def test_queries_in_flight_cannot_exceed_the_client_count(self, ctx,
                                                              dataset):
        """Little's law, which is the invariant the overstatement broke.

        Mean latency x QPS is the average number of queries in flight over the
        window, and a client issues one query at a time and waits for it, so
        that average cannot exceed the client count.

        It is an ALGEBRAIC IDENTITY on trimmed data, not a tendency of this
        machine: every counted span lies inside the window, and one client's
        spans cannot overlap because it waits for each query, so each client
        contributes at most `wall` seconds of query time to a window `wall`
        seconds long. No absolute rate appears in it and nothing in it assumes
        the machine finished a particular amount of work, which is why it can
        be asserted where a QPS figure cannot.

        What load does is narrow the observed margin, not create a failure:
        the largest value seen was 7.888 of 8 idle and 7.954 of 8 under
        oversubscription, because a slower query raises the mean faster than it
        lowers the count. The identity is what makes that safe -- the honest
        value approaches the bound and provably cannot cross it. An earlier
        version of this docstring claimed load pushes the value further inside
        the bound; it does the opposite, and the identity rather than that
        claim is the reason this assertion cannot flake.

        Counting the in-flight query breaks the identity from the other side:
        that query is the longest one in the sample (it has not finished yet),
        and it charges the window with time the window did not contain.
        Untrimmed this reads 8.072-8.085 queries in flight across eight
        clients, which eight clients cannot have had.

        The 0.1% slack is for the record's own rounding -- `qps` to two
        decimals and `latency_mean_ms` to four, 6e-5 relative at these
        magnitudes -- and never for the measurement, which cannot reach the
        bound. The untrimmed violation is about 1%, an order of magnitude
        clear of it.
        """
        row = self._run(ctx, dataset, self._driver(dataset),
                        TRIM_CLIENTS, TRIM_WINDOW_S)
        # The latency floor really was in force, so "one query per client" is
        # a visible share of this window rather than a rounding.
        assert row["latency_p50_ms"] >= TRIM_LATENCY_S * 1000.0
        assert row["queries_executed"] > 0
        in_flight = row["qps"] * row["latency_mean_ms"] / 1000.0
        assert in_flight <= TRIM_CLIENTS * 1.001, (
            f"{in_flight:.4f} queries in flight across {TRIM_CLIENTS} "
            f"clients: the window is being credited with work that finished "
            f"outside it")
        # And the bound is live rather than vacuously satisfied: eight busy
        # clients really were nearly always inside a query, so a merge that
        # dropped far too much would fail here instead.
        assert in_flight > TRIM_CLIENTS * 0.8, in_flight

    def test_the_record_says_how_many_queries_the_window_cut_off(self, ctx,
                                                                 dataset):
        """`extra.inflight_discarded` is exactly one per client, which is a
        property of the client loop rather than a measurement.

        At most one: a client runs its queries one after another, so only the
        last one it started can still be running when the window ends. At
        least one: for a client to contribute none, it would have to complete
        a query before the cutoff and then read the flag after `set()` landed
        — a window of a microsecond or two inside a 50.4 ms cycle, and all
        eight clients would have to manage it at once.

        Asserted as an EQUALITY, not as `1 <= n <= clients`. The range form is
        satisfied by a merge that trims only the first client's in-flight
        query, which is a plausible off-by-one over the collector loop and
        exactly the kind of mutant a range hides. Measured `== clients` in
        every run of six different client counts, idle and under
        oversubscription; Little's law and the scored-versus-counted test both
        kill that mutant too, so this is defence in depth rather than the only
        line.

        `prewindow_discarded` is the other end of the same window and is zero
        by construction, because the window opens at the barrier release and no
        measured query can precede it. It is recorded so that "a client began
        before the window" stops being an assumption; the unit tests in
        test_harness.py drive the non-zero case, which no real window produces.
        """
        row = self._run(ctx, dataset, self._driver(dataset),
                        TRIM_CLIENTS, TRIM_WINDOW_S)
        extra = row["extra"]
        assert extra["inflight_discarded"] == TRIM_CLIENTS, extra
        assert extra["prewindow_discarded"] == 0, extra

    def test_trimming_removed_queries_the_untrimmed_merge_counted(
            self, ctx, dataset, monkeypatch):
        """The same spans, merged both ways, which is the only form of this
        claim that does not depend on how fast the machine is.

        An absolute rate cannot be asserted here (see the class docstring) and
        a before/after comparison across two runs would compare two different
        windows. Merging one window's real spans with and without the cutoff
        holds everything else fixed, so the difference is the change and
        nothing else.

        A short window on purpose: one unfinished query out of three is a
        third of the count, where over two seconds it is 2.5%. The exact
        relation below holds at any length; this one just costs 0.3 s.
        """
        real_merge = concurrency_workload.merge_collectors
        both = []

        def merge_twice(collectors, wall_seconds, **kw):
            trimmed = real_merge(collectors, wall_seconds, **kw)
            # No cutoff: what the code did before §1.3, over the very same
            # collectors. `merge_collectors` only reads them.
            untrimmed = real_merge(collectors, wall_seconds)
            # A copy, because the workload pops keys out of what it is handed.
            both.append((dict(trimmed), untrimmed))
            return trimmed

        monkeypatch.setattr(concurrency_workload, "merge_collectors",
                            merge_twice)
        clients = 4
        row = self._run(ctx, dataset, self._driver(dataset), clients, 0.3)
        # Two since ops version 9: the discarded warm-up window is merged by
        # the same code and then thrown away, so the record's window is the
        # LAST merge.
        assert len(both) == 2, "one warm-up and one measured window"
        trimmed, untrimmed = both[-1]

        dropped = (trimmed[latency.INFLIGHT_DISCARDED]
                   + trimmed[latency.PREWINDOW_DISCARDED])
        assert dropped >= 1
        # Exact, not approximate: the two merges differ by the spans one of
        # them dropped and by nothing else.
        assert (untrimmed["queries_executed"] - trimmed["queries_executed"]
                == dropped)
        # The overstatement was real and is gone. Strict inequality: the
        # untrimmed merge divides a larger count by the same wall.
        assert untrimmed["qps"] > trimmed["qps"]
        # And it is the record that carries the trimmed figure, not the
        # discarded one -- the whole point is which number reaches the chart.
        assert row["qps"] == pytest.approx(trimmed["qps"], rel=1e-3)
        assert row["queries_executed"] == trimmed["queries_executed"]

    def test_a_window_that_finished_nothing_records_nulls_not_nan(
            self, ctx, dataset, capsys):
        """The path this change newly created, and why it is not cosmetic.

        Before the window was trimmed, every client completed at least one
        query, so the sample set was never empty. Trim it and a configuration
        whose queries outlast the window keeps nothing — and `summarize([])`
        returns NaN for every percentile, which `json.dumps` writes as the
        bare literal `NaN`. That is not JSON: Python's `json.loads` accepts it
        as an extension and the browser's `JSON.parse` does not, so the report
        would read the run happily while the web UI threw on it.

        None, not NaN. It serialises as a value every parser accepts, and
        `report/render.py::_fmt` already renders a missing figure as an
        em-dash — the convention T1b used for `recall_spread` at one window.
        Note which shape it takes in the record: `Record.to_dict` drops its
        own None columns, as `test_none_fields_are_omitted` pins, so an
        unmeasured percentile is an absent key rather than a null one. Both
        parse; the assertion below says which it is so the schema is not left
        to inference.

        And loud as well as empty: a window too short for one query is a
        profile fault that should not be discovered later from a chart with a
        gap in it.
        """
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32,
                               latency_s=EMPTY_WINDOW_LATENCY_S)
        # warmup=0 deliberately: at 0.3 s a query, warming up would cost more
        # than the rest of this file's concurrency tests put together.
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=(1,), duration_s=EMPTY_WINDOW_S, warmup=0,
            max_queries=concurrency_workload.WHOLE_TEST_SET,
            indexed_rows=N_ROWS)
        row = written(ctx)[-1]

        # The window really did finish nothing, and for the stated reason.
        assert row["queries_executed"] == 0
        assert row["extra"]["inflight_discarded"] == 1, row["extra"]
        assert row["extra"]["recall_queries"] == 0

        for field in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                      "latency_p99_ms"):
            assert row.get(field) is None, (field, row.get(field))
            assert field not in row, (
                f"{field} was written as a value; an unmeasured column is "
                f"omitted, like every other None column in a Record")

        # The file a strict parser has to read, over every record in it.
        assert strict_json_lines(ctx.recorder.path) == 1

        # Zero completions per second is the honest throughput for a window
        # that completed nothing, and unlike NaN it serialises and plots.
        assert row["qps"] == 0.0

        err = capsys.readouterr().err
        assert "NOTHING FINISHED" in err, err
        # Naming the window length, because the remedy is to raise it.
        assert str(EMPTY_WINDOW_S) in err, err

    def test_the_window_closes_before_the_stop_flag_is_set(self, ctx, dataset,
                                                           monkeypatch):
        """The cutoff is read BEFORE `stop.set()`, not after.

        Unamplified this is unguardable and should not be guarded: setting an
        `Event` nobody is waiting on costs well under a microsecond against a
        2-60 second window, and any tolerance able to see that would be
        asserting on the implementation. Saying "no test can distinguish it"
        was the wrong conclusion, though — the ORDERING is observable once the
        cost of setting the flag is made large, and then the evidence is in the
        record rather than in the source.

        So the stop flag is given a `set()` that takes `SLOW_STOP_S`. During
        that time the clients are still looping, because none of them has seen
        the flag yet. Read before `set()`, the cutoff excludes everything they
        do in that interval and the window stays `duration_s` long. Read after,
        the window silently absorbs `SLOW_STOP_S` of extra querying — which is
        §1.3's fault in macroscopic form, a denominator that ends later than
        the numerator believes.

        Note `inflight_discarded` here is far more than one per client, and
        that is the amplifier rather than a counter-example: the clients get a
        whole `SLOW_STOP_S` of querying past the cutoff instead of the
        microsecond a real `Event` gives them. `test_the_record_says_how_many_
        queries_the_window_cut_off` pins the real case.

        The shim is scoped to this module's `threading` attribute on purpose.
        Patching `threading.Event` globally would also slow the `Heartbeat`
        around the barrier — whose `__exit__` now runs inside the measured
        window — and inflate the very figure under test.
        """
        slow_stop_s = 0.2
        clients = 4
        window_s = 0.2

        class _SlowStop(threading.Event):
            """A stop flag that takes a visible amount of time to set."""

            def set(self) -> None:
                time.sleep(slow_stop_s)
                super().set()

        monkeypatch.setattr(
            concurrency_workload, "threading",
            types.SimpleNamespace(
                Event=_SlowStop, Barrier=threading.Barrier,
                BrokenBarrierError=threading.BrokenBarrierError))

        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32,
                               latency_s=0.02)
        row = self._run(ctx, dataset, driver, clients, window_s)

        # The window the record implies. `queries_executed / qps` is the wall
        # handed to the merge, because both come out of it. The tolerance is
        # half the injected delay: 140x the honest excess (the barrier-to-
        # sleep gap, well under a millisecond) and half the thing being
        # detected.
        implied_window = row["queries_executed"] / row["qps"]
        assert implied_window < window_s + slow_stop_s / 2, (
            f"the measured window is {implied_window:.4f}s against a "
            f"requested {window_s}s: it has absorbed the {slow_stop_s}s the "
            f"stop flag took to set, so its end is being read after the flag "
            f"rather than before it")

        # And the amplifier worked, so the assertion above was not vacuous:
        # the clients really did keep querying past the cutoff while the flag
        # was being set. One query per client would mean the flag went up
        # instantly and there was no ordering to observe.
        assert row["extra"]["inflight_discarded"] > clients, row["extra"]


#: p(ef) = 98 / (98 + 2) = 0.98 exactly, so the recall these tests compare is
#: a real measurement of a real error model rather than the 1.0 a perfect fake
#: returns. An assertion that two 1.0s agree is satisfied by a hardcoded 1.0;
#: this one is not. Near the top of the range on purpose, because the sampling
#: error of a mid-range p swamps the 0.01 tolerance — see `wide_dataset`.
EF_SEARCH_98 = 98
EF_HALF_98 = 2.0

#: Long enough, on an idle machine, for every client to finish one pass over
#: its shard at 1, 4 and 16 clients.
#:
#: The margin here is SMALL, and an earlier version of this comment claimed it
#: was five times larger than it is. Measured by bisection, full coverage at
#: 16 clients needs about 0.20s on this machine and about 0.30s on the
#: reviewer's, so the real headroom is 1.7x-2.5x. Aggregate throughput is not
#: what binds: 1,000 queries at ~11,000 q/s is 0.09s of work however it is
#: divided, and 1 and 4 clients cover the pool inside 0.10s. What binds is the
#: SLOWEST thread finishing its own shard, and with seventeen threads on one
#: GIL a thread can receive far less than its 1/16 share.
#:
#: That is why `run_until_pool_covered` escalates rather than trusting this
#: number. An overstated margin is worse than none, because the next person
#: reads it instead of measuring: at 0.5s flat, the coverage test failed 7
#: times in 18 runs under deliberate oversubscription.
WINDOW_S = 0.5

#: Windows the coverage runs escalate through, each 4x the last.
#:
#: Escalation rather than a bigger constant, and rather than a looser
#: assertion. "Every query was covered" is the only thing that discriminates a
#: truncated or overlapping shard shape -- a wrong shape never reaches full
#: coverage at ANY duration, because the queries it never sends stay unsent --
#: so the assertion has to stay an equality. But it is also a claim that the
#: machine finished a fixed amount of work inside a fixed wall-clock window,
#: and under load a starved thread simply does not. Retrying longer separates
#: the two: a correct implementation pays nothing on an idle machine and a
#: retry on a busy one, while a broken one fails at 0.5s, at 2s and at 8s.
COVERAGE_WINDOWS_S = (WINDOW_S, WINDOW_S * 4, WINDOW_S * 16)


def run_until_pool_covered(ctx: RunContext, dataset: Dataset, driver,
                           clients, max_queries: int, expected: int) -> None:
    """The concurrency workload, retried longer until the pool is covered.

    **Returns nothing on purpose.** The caller reads the records back from the
    recorder itself, with `last_points`, and asserts on those. An earlier
    version returned the rows it had just inspected, which made the helper and
    the caller's assertion one code path: a mutant that overwrote
    `recall_queries` with `expected` on the way out passed all 621 tests. A
    helper that both decides when to stop and supplies the evidence for the
    assertion cannot be checked by it.

    See `COVERAGE_WINDOWS_S` for why this escalates rather than using one
    longer window or one weaker assertion.
    """
    for duration in COVERAGE_WINDOWS_S:
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset),
            ef_search=EF_SEARCH_98, client_counts=clients,
            duration_s=duration, warmup=2, max_queries=max_queries,
            indexed_rows=N_ROWS)
        if all(r["extra"]["recall_queries"] == expected
               for r in last_points(ctx, len(clients))):
            return


def last_points(ctx: RunContext, count: int):
    """The `count` most recent records, read back from the recorder.

    A tail rather than the whole file because each escalation attempt appends,
    so the earlier attempts' records sit above the ones that matter.
    """
    return written(ctx)[-count:]

#: Per-query latency for the one test that needs a window too short to finish
#: a pass. Without it that test is calibrated on this machine's speed in the
#: dangerous direction: at zero latency a 0.05s window covered 679 of 1,000
#: queries, so a machine only 1.5x faster would have completed the pass and
#: failed an assertion that nothing was left over. With a 1ms floor per query
#: one pass costs at least a second however fast the host is, and 0.05s covers
#: about forty queries.
SHORT_WINDOW_LATENCY_S = 0.001


class TestConcurrencyRecordsRecall:
    """A concurrency point without recall invites the comparison it cannot
    support.

    QPS and latency were recorded; the ids every query returned were thrown
    away. Two engines at `ef_search=100` can sit at recall 0.98 and 0.80 —
    which is the ordinary spread of these implementations at a shared ef, not
    a pathology — and a throughput chart with no quality figure beside it
    presents them as comparable when one is simply doing less work per query.
    """

    def _run(self, ctx, wide, driver, clients, **kw):
        kw.setdefault("duration_s", WINDOW_S)
        concurrency_workload.run(
            ctx, driver.clone, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=clients, warmup=2,
            max_queries=concurrency_workload.WHOLE_TEST_SET,
            indexed_rows=N_ROWS, **kw)
        return written(ctx)

    def test_recall_equals_single_client_recall_at_every_client_count(
            self, ctx, wide):
        """The number has to come from somewhere independent of the workload.

        `measure_recall` drives the driver directly, one query at a time, in
        test-set order, and zips the results against the dataset's own
        neighbours. The workload shards the query set across threads, records
        only each shard's first pass, maps every result back through
        `_shard_queries`'s offsets and scores it against
        `cached_ground_truth`. Both estimate the same quantity — the fake's
        p(ef) — by routes that share no code past `recall_at_k` itself, which
        is what makes the agreement evidence rather than a tautology.

        Four and sixteen clients are load-bearing, not thoroughness. At one
        client the shard IS the query set and the offsets are `arange(n)`, so
        every plausible way of getting the mapping wrong still produces the
        right answer: the mutation below scores 0.9825 at one client and
        0.0084 at four.
        """
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        # Measured before the workload runs, and never through it. The clones
        # the workload uses are seeded independently of this driver's stream,
        # so neither measurement perturbs the other.
        single = measure_recall(driver, wide, K)
        # Not 1.0 and not 0.0: a tolerance between two saturated numbers is
        # satisfied by anything that also saturates.
        assert 0.9 < single < 1.0, single

        rows = self._run(ctx, wide, driver, (1, 4, 16))
        assert [r["clients"] for r in rows] == [1, 4, 16]
        for row in rows:
            assert row["phase"] == records.PHASE_CONCURRENCY
            assert abs(row["recall_at_k"] - single) < 0.01, (
                f"{row['clients']} clients: {row['recall_at_k']} vs {single}")
            # recall_at_k is an existing Record column, not an extra key, so
            # the report reads it without a per-phase special case.
            assert row["recall_at_k"] is not None

    def test_recall_covers_every_query_in_the_pool(self, ctx, wide):
        """How many queries the recall is over is part of the claim.

        A window too short for one pass leaves most of the pool unmeasured,
        and recall over 200 of 10,000 queries is a different statement from
        recall over all of them. This also catches every way of mis-shaping a
        shard that the recall assertion above cannot, and it is the only thing
        that catches two of them. Shards truncated to five queries each still
        score 0.98, because the queries they do send are answered correctly.
        Worse, giving every client the SAME slice scores 0.979-0.982 and
        passes the recall comparison deterministically — and a truncation to
        100 or 300 queries passes it too, while coverage is out by 10x. Only
        the count discriminates either one.

        Which is exactly why the window escalates instead of the assertion
        loosening. A wrong shard shape never reaches full coverage at any
        duration; a starved thread on a loaded machine does, given longer.
        """
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        clients = (1, 4, 16)
        run_until_pool_covered(
            ctx, wide, driver, clients,
            max_queries=concurrency_workload.WHOLE_TEST_SET,
            expected=N_WIDE_QUERIES)
        # Read back from the recorder, not handed over by the helper above.
        rows = last_points(ctx, len(clients))
        assert [r["clients"] for r in rows] == list(clients)
        for row in rows:
            assert row["extra"]["recall_queries"] == N_WIDE_QUERIES, (
                f"{row['clients']} clients covered "
                f"{row['extra']['recall_queries']} of {N_WIDE_QUERIES} "
                f"even at {COVERAGE_WINDOWS_S[-1]}s")

    def test_a_short_window_records_the_coverage_it_managed(self, ctx, wide):
        """The honest answer to a window too short for a pass is a smaller
        count, not a recall collapse. Padding the queries nobody ran with
        empty result sets would report the window's length as the engine's
        quality.

        `latency_s` rather than a very short window, and the reason is the
        direction of the risk. "Fewer than the whole pool" is an assertion
        about the machine being slow enough, so calibrating it on this
        machine's speed makes it fail on a faster one: at zero latency a
        0.05s window covered 679 of 1,000 and a host 1.5x quicker would have
        covered all of them. A 1ms floor per query puts one pass at a second
        whatever the host does.
        """
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98,
                              latency_s=SHORT_WINDOW_LATENCY_S)
        rows = self._run(ctx, wide, driver, (1,), duration_s=0.05)
        covered = rows[0]["extra"]["recall_queries"]
        assert 0 < covered < N_WIDE_QUERIES, covered
        # Recall over the part that ran is still the engine's recall, not a
        # fraction of it scaled by how much of the pool was reached.
        assert rows[0]["recall_at_k"] > 0.9, rows[0]["recall_at_k"]
        # And the latency floor really was in force, so the window above is
        # short for the reason this test says it is.
        assert rows[0]["latency_p50_ms"] >= SHORT_WINDOW_LATENCY_S * 1000.0

        # `query_reuse` is executed over DISTINCT-executed, and this is the
        # only regime in which that distinction is observable. Inside one
        # incomplete pass nothing is repeated, so the honest answer is exactly
        # 1.0 — every query sent was a query not sent before.
        #
        # The sibling test asserting `pool x reuse == executed` cannot see
        # this: it also asserts the window covered the pool, which makes
        # distinct and pool the same number and the cross-check a restatement
        # of itself. Dividing by `query_pool` instead passes there and reads
        # 0.04 here.
        assert rows[0]["extra"]["query_reuse"] == 1.0, (
            rows[0]["extra"]["query_reuse"], covered)
        assert rows[0]["queries_executed"] == covered

    def test_the_scored_queries_are_the_counted_queries(self, ctx, wide):
        """Recall and QPS must be about the same set of queries.

        The window is trimmed to the queries that finished inside it, and the
        ids a query returned are kept in the same breath as its latency. Trim
        one without the other and the record reports recall over queries whose
        timing it refused to count: `queries_executed` comes in BELOW
        `recall_queries`, and `query_reuse` -- executed over distinct -- drops
        under 1.0, which reads as a pool reused less than once.

        Four clients inside one incomplete pass, which is the only regime
        where it is visible: every client is still on its first pass when the
        window ends, so every client's unfinished query is one whose result
        was recorded. One client would make the discrepancy a single query;
        four make it four, and a real sweep reaches thirty-two.

        The sibling test above pins the same equality at one client. This one
        is here because the per-client factor is what a reader would notice,
        and because `SHORT_WINDOW_LATENCY_S` is what makes an incomplete pass
        a property of the code rather than of this machine's speed.
        """
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98,
                              latency_s=SHORT_WINDOW_LATENCY_S)
        rows = self._run(ctx, wide, driver, (4,), duration_s=0.05)
        row = rows[0]
        covered = row["extra"]["recall_queries"]
        # Inside one pass: nothing was repeated, so "covered" and "executed"
        # are two counts of the same queries and must agree exactly.
        assert 0 < covered < N_WIDE_QUERIES, covered
        assert row["queries_executed"] == covered, (
            f"{row['queries_executed']} counted against {covered} scored: "
            f"recall and QPS are describing different query sets")
        assert row["extra"]["query_reuse"] == 1.0
        # The queries that were cut off are the reason this could go wrong,
        # so the record has to show there were some.
        assert row["extra"]["inflight_discarded"] >= 1, row["extra"]
        # Recall over the part that ran is still the engine's recall.
        assert row["recall_at_k"] > 0.9, row["recall_at_k"]

    def test_a_recall_that_moves_between_windows_is_reported_as_a_bug(
            self, ctx, wide, capsys):
        """Recall is a property of the index and the ef, never of the window.

        Two windows of one configuration disagreeing means the
        query-to-truth mapping is wrong or the index moved underneath the
        measurement, and in both cases the QPS recorded beside it is not the
        quantity the record names. Staged by degrading the driver the second
        window is handed, which is the only way to make the fake's recall
        depend on which window it is in.
        """
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        # The healthy window's recall, measured independently of the workload.
        healthy = measure_recall(driver, wide, K)
        # Counted in a `multiprocessing.Value`, not a list, because G3 made
        # the clients OS processes and this factory is called IN THE CHILD.
        # With a list the child saw only the metadata driver's entry, the
        # second window was never degraded, and `extra.recall_spread` came
        # back **0.0** against the 0.005 limit this test exists to trip -- a
        # fixture that staged nothing and an assertion that could not fail
        # for the right reason.
        handed_out = MP_FORK.Value("i", 0)

        def degrading_factory():
            clone = driver.clone()
            with handed_out.get_lock():
                handed_out.value += 1
                mine = handed_out.value
            # One for the metadata driver, one for the discarded warm-up
            # window's client (ops version 9), one for the first measured
            # window's; everything after that is the second window.
            if mine > 3:
                clone.ef_half = 400.0
            return clone

        concurrency_workload.run(
            ctx, degrading_factory, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=(1,),
            duration_s=0.3, warmup=1, repeats=2,
            max_queries=concurrency_workload.WHOLE_TEST_SET,
            indexed_rows=N_ROWS)

        row = written(ctx)[0]
        assert row["extra"]["repeats"] == 2
        spread = row["extra"]["recall_spread"]
        assert spread > concurrency_workload.RECALL_SPREAD_LIMIT, spread
        # The degradation crossed the fork: three drivers were handed out as
        # the PARENT counts them, so the second window really got the bad one.
        assert handed_out.value >= 4, (
            f"only {handed_out.value} drivers were counted in the parent, so "
            f"the second window was never degraded and the spread above is "
            f"not this fixture's doing")

        # One window's recall, not an average of the two. The two windows
        # measure `healthy` (about 0.98) and `healthy - spread` (about 0.21),
        # and the record must carry one of those endpoints; their mean, about
        # 0.59, is a third of the way from either and excluded by a margin of
        # 0.38. Which endpoint is recorded is whichever window came out median
        # by QPS, and that is genuinely not fixed -- the degraded driver draws
        # far more filler rows per query, so it is usually the slower of the
        # two but not reliably so. Asserting the healthy value specifically
        # passed on this machine and failed under pytest on the same machine
        # minutes later.
        endpoints = (healthy, healthy - spread)
        assert min(abs(row["recall_at_k"] - e) for e in endpoints) < 0.01, (
            row["recall_at_k"], endpoints)
        assert abs(row["recall_at_k"] - sum(endpoints) / 2) > 0.1, (
            "the record carries the mean of the windows, not one of them")

        err = capsys.readouterr().err
        assert "BUG: recall varied" in err, err
        # Actionable, not merely alarmed: it has to say which configuration
        # and what the operator should doubt.
        #
        # Anchored on the whole number, because `"1 client" in err` against a
        # message reading "at {clients} client(s)" is satisfied by 11, 21 and
        # 31 clients too: reporting clients + 20 left every test in this file
        # passing. The assertion claimed to check WHICH configuration and did
        # not. The fourth instance of this species in this series, and the
        # third in this file alone.
        assert re.search(r"\bat 1 client\(s\)", err), err
        assert "QPS" in err, err

    def test_one_window_has_no_recall_spread_to_report(self, ctx, wide):
        """None, not 0.0.

        A single window has no spread in the same sense that it has no error
        bar. Writing 0.0 claims the stronger thing — that two windows were
        compared and agreed exactly — which is the entire evidence this field
        exists to carry, and a reader cannot tell the two apart without
        cross-checking `repeats`. `report/render.py` already draws that
        distinction by hand for `qps_spread_pct`; this field does not make
        anyone do it again.
        """
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        rows = self._run(ctx, wide, driver, (1,), duration_s=0.05)
        extra = rows[0]["extra"]
        assert extra["repeats"] == 1
        # Present and null, which is how `scaling_efficiency` and
        # `query_reuse` already express "not measurable here" in this same
        # dict. `Record.to_dict` drops a None only among the Record's own
        # columns; it does not reach inside `extra`, so a key omitted here
        # would be omitted inconsistently with its neighbours.
        assert "recall_spread" in extra
        assert extra["recall_spread"] is None, extra["recall_spread"]
        # And the recall itself is still there, so this is not a record that
        # simply failed to measure anything.
        assert rows[0]["recall_at_k"] > 0.9

    def test_a_stable_recall_says_nothing(self, ctx, wide, capsys):
        """A warning that always fires is noise, and the first thing an
        operator learns to ignore. With p(ef) = 1 the fake is deterministic,
        so two windows measure exactly the same recall and the spread is
        exactly zero -- there is no tolerance being leaned on here."""
        driver = loaded_driver(wide, ef_half=0.0, ef_search=EF_SEARCH_98)
        concurrency_workload.run(
            ctx, driver.clone, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=(1,),
            duration_s=0.3, warmup=1, repeats=2,
            max_queries=concurrency_workload.WHOLE_TEST_SET,
            indexed_rows=N_ROWS)
        row = written(ctx)[0]
        assert row["recall_at_k"] == 1.0
        assert row["extra"]["recall_spread"] == 0.0
        assert "BUG" not in capsys.readouterr().err

    def test_a_subset_corpus_is_scored_against_the_subset(self, ctx, wide):
        """`indexed_rows`, for the same reason filtered and churn take it.

        A profile that loaded part of the training set would otherwise be
        scored against neighbours pointing at rows the engine never received,
        understating every engine at once — which reads as a result rather
        than as a measurement fault.
        """
        rows_loaded = 500
        driver = FakeDriver(dim=wide.dim, metric=wide.metric, ef_half=0.0,
                            ef_search=EF_SEARCH_98)
        driver.connect()
        driver.create_schema(index_spec(wide))
        driver.load(wide.train[:rows_loaded],
                    ds.assign_tags(N_ROWS)[:rows_loaded])

        concurrency_workload.run(
            ctx, driver.clone, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=(1,), duration_s=WINDOW_S,
            warmup=2, max_queries=concurrency_workload.WHOLE_TEST_SET,
            indexed_rows=rows_loaded)

        # A perfect driver over the rows it holds. Against the full corpus's
        # neighbours this would score far below 1.0, because most of the true
        # top ten are rows 500..1999 that were never loaded.
        assert written(ctx)[0]["recall_at_k"] == 1.0


class TestConcurrencyQueryPoolIsItsOwn:
    """`ops.max_queries` caps the filtered and churn workloads, which answer
    each query once; main.yml holds it at 500 to bound their cost. The
    concurrency workload cycles its shard for a fixed duration instead, so
    sharing the key gave 32 clients about fifteen queries each for thirty
    seconds and showed every graph cache in the comparison — MHNSW's
    per-TABLE_SHARE cache, VIDX's shared cache, pgvector's shared_buffers — a
    hot set small enough to hold entirely. That is the one thing the
    concurrency chart exists to tell apart.
    """

    def test_shards_are_disjoint_and_cover_the_whole_pool(self, wide):
        """At 32 clients, which is where it matters and where fifty queries
        would have made every shard one or two elements long."""
        pairs = concurrency_workload._shard_queries(wide.test, 32)
        assert len(pairs) == 32
        positions = [int(o) for _shard, offsets in pairs for o in offsets]
        # Disjoint: no position handed to two clients, so no client can warm
        # a cache for another and no result can overwrite another's.
        assert len(positions) == len(set(positions))
        # Complete: every query in the pool is somebody's.
        assert set(positions) == set(range(N_WIDE_QUERIES))

    def test_each_offset_names_the_query_beside_it(self, wide):
        """The pairing itself, checked against the array rather than against
        the expression that produced it. Recomputing `arange(c, n, clients)`
        here would assert the implementation equals itself; comparing the
        vectors proves the offset points at the query the client will send.
        """
        seen = 0
        for shard, offsets in concurrency_workload._shard_queries(wide.test, 7):
            # Non-empty first. `len(shard) == len(offsets)` holds as 0 == 0
            # and the zip body never runs, so seven empty pairs passed this
            # test while failing eleven of the other twelve in the file —
            # which is the signature of a guard that is along for the ride.
            assert len(shard) > 0
            assert len(shard) == len(offsets)
            for query, offset in zip(shard, offsets):
                assert numpy.array_equal(query, wide.test[int(offset)])
                seen += 1
        # And every query was actually compared, not just the ones that
        # happened to survive whatever shaping produced the shards.
        assert seen == N_WIDE_QUERIES

    def test_the_whole_test_set_is_the_default(self, ctx, wide):
        """`WHOLE_TEST_SET` is a sentinel, not a cap. Read as a cap it leaves
        the pool empty, every client count skipped by the too-few-queries
        rule, and a silently empty concurrency chart — and the default is what
        today's profiles get, since none of them set the key."""
        assert concurrency_workload.WHOLE_TEST_SET == 0
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        concurrency_workload.run(
            ctx, driver.clone, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=(2,), duration_s=0.05,
            warmup=1, indexed_rows=N_ROWS)
        rows = written(ctx)
        assert len(rows) == 1
        assert rows[0]["extra"]["query_pool"] == N_WIDE_QUERIES

    def test_the_record_states_the_pool_and_how_hard_it_was_reused(
            self, ctx, wide):
        """So a reader can judge a cache-sensitive number. The same QPS over a
        pool of 10,000 and a pool of fifteen are measurements of two different
        things, and nothing in the record used to say which.

        Over a deliberately small pool, which is both the case `query_reuse`
        exists to expose and the one that leaves the assertion no timing
        margin to get wrong: 200 queries across four clients is fifty each, so
        the window completes about 38 passes and the "reuse is real" bound
        would survive a machine thirty times slower.
        """
        pool = 200
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        run_until_pool_covered(ctx, wide, driver, (4,), max_queries=pool,
                               expected=pool)
        row = last_points(ctx, 1)[0]
        extra = row["extra"]
        assert extra["query_pool"] == pool
        assert extra["recall_queries"] == pool
        # Reuse is measured, not a rounded 1.0 standing in for "not recorded".
        assert extra["query_reuse"] > 5.0, extra["query_reuse"]
        # The three published numbers have to agree with each other, which is
        # a property of the record rather than a restatement of the expression
        # that produced any one of them. When the window covered the whole
        # pool, pool x reuse is the executed count — and that fails loudly for
        # any shard shape that does not cover the pool: truncated or
        # overlapping shards put it out by five orders of magnitude.
        implied = extra["query_reuse"] * extra["query_pool"]
        # The only slack is `query_reuse`'s own two-decimal rounding: half a
        # hundredth of a reuse, multiplied back up by the pool.
        assert abs(implied - row["queries_executed"]) <= 0.01 * pool, (
            implied, row["queries_executed"])

    def test_a_cap_still_caps(self, ctx, wide):
        """The way back. A profile that wants the old small hot set — to
        reproduce an earlier run, say — sets the key and gets exactly it."""
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        concurrency_workload.run(
            ctx, driver.clone, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=(2,), duration_s=0.05,
            warmup=1, max_queries=120, indexed_rows=N_ROWS)
        assert written(ctx)[0]["extra"]["query_pool"] == 120

    def test_the_skip_rule_survives(self, ctx, wide):
        """Kept, not removed: it cannot trigger on a real dataset any more,
        but a profile that caps the pool by hand can still ask for more
        clients than there are queries."""
        driver = loaded_driver(wide, ef_half=EF_HALF_98,
                              ef_search=EF_SEARCH_98)
        concurrency_workload.run(
            ctx, driver.clone, wide, index_spec(wide),
            ef_search=EF_SEARCH_98, client_counts=(2, 9), duration_s=0.05,
            warmup=1, max_queries=4, indexed_rows=N_ROWS)
        assert [r["clients"] for r in written(ctx)] == [2]


class TestFilteredWorkload:
    def test_records_recall_and_short_results_per_selectivity(self, ctx,
                                                              dataset):
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64)
        filtered_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=64,
            selectivities=(0.01, 0.50), max_queries=N_QUERIES, warmup=2,
            indexed_rows=N_ROWS, columns=(ds.FILTER_TAG,),  # T3c: this test is about `tag`
        )

        rows = written(ctx)
        assert [r["selectivity"] for r in rows] == [0.01, 0.50]
        assert all(r["phase"] == records.PHASE_FILTERED for r in rows)
        assert all(r["vector_index_used"] is True for r in rows)
        assert all(r["clients"] == 1 and r["qps"] > 0 for r in rows)
        assert [r["extra"]["tag_threshold"] for r in rows] == [1, 50]

        selective, permissive = rows
        # Post-filtering a 64-candidate list when 1% of rows qualify cannot
        # find ten of them, so the result sets are short and recall collapses.
        assert selective["extra"]["returned_fewer_than_k"] is True
        assert selective["extra"]["short_result_queries"] == N_QUERIES
        assert permissive["extra"]["short_result_queries"] == 0
        assert permissive["extra"]["returned_fewer_than_k"] is False
        assert permissive["recall_at_k"] > selective["recall_at_k"] + 0.5
        # Scored against filtered ground truth, which is what makes the 50%
        # number near-perfect rather than near-zero.
        assert permissive["recall_at_k"] > 0.95

    def test_runs_against_an_angular_corpus(self, ctx, angular_dataset):
        """The angular distance branch, through a real workload.

        `filtered.run` scores against `filtered_ground_truth`, which has its
        own angular branch and normalizes before comparing. A near-perfect
        recall here means the fake's cosine ordering and the harness's agree;
        if either drifted, this is where it would show rather than in a
        six-hour run on glove-100-angular.
        """
        driver = loaded_driver(angular_dataset, ef_half=0.0, ef_search=64)
        filtered_workload.run(
            ctx, driver, angular_dataset, index_spec(angular_dataset),
            ef_search=64, selectivities=(0.50,), max_queries=N_QUERIES,
            warmup=2, indexed_rows=N_ROWS, columns=(ds.FILTER_TAG,),  # T3c: this test is about `tag`
        )
        row = written(ctx)[0]
        assert row["metric_space"] == ds.ANGULAR
        assert row["recall_at_k"] > 0.95
        assert row["extra"]["short_result_queries"] == 0

    def test_a_plan_failure_is_carried_into_the_record(self, ctx, dataset):
        # "Exact results, slowly" is indistinguishable from "accurate and slow"
        # unless the flag survives all the way to the record.
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64,
                               uses_vector_index=False)
        filtered_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=64,
            selectivities=(0.50,), max_queries=N_QUERIES, warmup=2,
            indexed_rows=N_ROWS, columns=(ds.FILTER_TAG,),  # T3c: this test is about `tag`
        )
        assert written(ctx)[0]["vector_index_used"] is False


class TestChurnWorkload:
    def test_records_a_baseline_and_each_churn_step(self, ctx, dataset):
        # ef high enough that the baseline is near-exact, so the drop measured
        # after churn is the tombstones and not the ANN error.
        driver = loaded_driver(dataset, ef_half=1.0, ef_search=512,
                               tombstone_penalty=1.0)
        churn_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=512,
            fractions=(0.10, 0.25), max_queries=N_QUERIES,
        )

        rows = written(ctx)
        assert [r["churn_fraction"] for r in rows] == [0.0, 0.10, 0.25]
        assert all(r["phase"] == records.PHASE_CHURN for r in rows)
        assert rows[0]["notes"] == "baseline before any churn"

        for row in rows[1:]:
            assert row["extra"]["reinsert_completed"] is True
            assert row["extra"]["rows_churned"] > 0
            assert row["extra"]["insert_seconds"] >= 0
        # Each step churns the additional fraction, so 10% then 25% is 200 then
        # 300 rows, not 200 then 500.
        assert [r["extra"]["rows_churned"] for r in rows[1:]] == [200, 300]

        # The whole reason this workload exists: recall after churn, scored
        # against ground truth remapped to the re-inserted ids.
        recalls = [r["recall_at_k"] for r in rows]
        assert recalls[0] > recalls[1] > recalls[2], (
            f"recall did not degrade under churn: {recalls}")
        assert rows[2]["extra"]["recall_drop_vs_baseline"] > 0
        # The corpus is whole again after each step, and the index is not.
        assert driver.count_rows() == N_ROWS
        assert rows[2]["index_bytes"] > rows[0]["index_bytes"]

    def test_an_unfinished_reinsert_is_recorded_without_a_recall_figure(
            self, ctx, dataset):
        # A corpus missing rows is neither the baseline nor a churned corpus,
        # so what was achieved is recorded and recall deliberately is not.
        driver = loaded_driver(dataset, ef_half=1.0, ef_search=512,
                               latency_s=0.0)

        slow = driver.insert_rows

        def crawl(ids, vectors, tags):
            time.sleep(0.02)
            slow(ids, vectors, tags)

        driver.insert_rows = crawl
        churn_workload.run(
            ctx, driver, dataset, index_spec(dataset), ef_search=512,
            fractions=(0.10,), max_queries=N_QUERIES, budget_s=0.05,
        )

        rows = written(ctx)
        assert len(rows) == 2
        stopped = rows[1]
        assert stopped["extra"]["reinsert_completed"] is False
        assert stopped["extra"]["rows_reinserted"] < stopped["extra"]["rows_expected"]
        assert "recall_at_k" not in stopped
        assert stopped["notes"] == (
            "re-insert did not complete within the churn budget")


class TestWorkloadsShareOneSchema:
    def test_all_four_write_accepted_phases_into_one_file(self, ctx, dataset):
        """Every workload in one run, as a real invocation orders them.

        Churn runs last because it mutates the table: everything before it sees
        a pristine index and nothing after it inherits a degraded one. The
        ordering is part of what this asserts.
        """
        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            ef_half=16.0, ef_search=64)
        driver.connect()
        index = index_spec(dataset)

        build_workload.run(ctx, driver, dataset, index)
        indexed_rows = driver.count_rows()
        concurrency_workload.run(
            ctx, driver.clone, dataset, index, ef_search=64,
            client_counts=(1, 2), duration_s=0.05, warmup=1,
            max_queries=N_QUERIES)
        filtered_workload.run(
            ctx, driver, dataset, index, ef_search=64, selectivities=(0.50,),
            max_queries=N_QUERIES, warmup=2, indexed_rows=indexed_rows,
            # T3c: this test is about `tag`
            columns=(ds.FILTER_TAG,))
        churn_workload.run(
            ctx, driver, dataset, index, ef_search=64, fractions=(0.10,),
            max_queries=N_QUERIES, indexed_rows=indexed_rows)

        phases = [r["phase"] for r in written(ctx)]
        assert phases == [
            records.PHASE_INGEST, records.PHASE_BUILD,
            records.PHASE_CONCURRENCY, records.PHASE_CONCURRENCY,
            records.PHASE_FILTERED,
            records.PHASE_CHURN, records.PHASE_CHURN,
        ]
        # Recorder.write refuses an unknown phase, so getting this far proves
        # every phase these workloads emit is in ALL_PHASES.
        assert set(phases) <= set(records.ALL_PHASES)

    def test_no_phase_claims_a_storage_engine_the_engine_does_not_have(
            self, ctx, dataset, monkeypatch):
        """The defect, reproduced over every phase one unit writes.

        `FakeDriver` is not a MySQL-family driver and therefore has no
        `ENGINE=` clause, exactly like `PostgresDriver`, `ValkeyDriver` and
        `MongoDriver`. The `IndexSpec` it is handed still carries `InnoDB`,
        because `harness/main.py` defaults `--storage-engine` to it and
        `orchestrator/cli.py:_ops_storage_engines` passes it for every engine
        that is not MariaDB -- so the value is present, unused, and was being
        stamped on the record anyway. In `full-20261015-215006` that published
        268 pgvector, 128 Valkey and 120 Percona Search records claiming
        InnoDB, beside the same run's `heap`, `memory` and `wiredTiger` from
        the ann path: one engine, two storage engines, one run.
        * Honest: not one record carries the key.
        * Mutant (`record_defaults` reading `index.storage_engine` again):
          every record in the file carries `InnoDB`.

        All five phases, not a sample, because the defect was in the shared
        provenance dict and a per-phase check could pass against a workload
        that happens to overwrite it.
        """
        # Imported here rather than in this module's header: the file is about
        # the four measurement workloads, and calibration is in
        # tests/test_calibrate.py. It is in this list because the archived run
        # carries 139 `ef_calibration` records for Percona Server alone, and
        # they were wrong in the same way.
        from harness.workloads import calibrate as calibrate_workload

        driver = FakeDriver(dim=dataset.dim, metric=dataset.metric,
                            ef_half=16.0, ef_search=64)
        driver.connect()
        index = index_spec(dataset)
        # The precondition that makes the rest of this non-vacuous: the spec
        # really does carry the value the bug published. A fixture whose spec
        # held None would pass this test against the unfixed code.
        assert index.storage_engine == "InnoDB"

        build_workload.run(ctx, driver, dataset, index)
        indexed_rows = driver.count_rows()
        calibrate_workload.run(ctx, driver, dataset, index, ef_grid=(64,),
                               max_queries=N_QUERIES, warmup=1,
                               indexed_rows=indexed_rows)
        concurrency_workload.run(
            ctx, driver.clone, dataset, index, ef_search=64,
            client_counts=(1,), duration_s=0.05, warmup=1,
            max_queries=N_QUERIES)
        filtered_workload.run(
            ctx, driver, dataset, index, ef_search=64, selectivities=(0.50,),
            max_queries=N_QUERIES, warmup=2, indexed_rows=indexed_rows,
            columns=(ds.FILTER_TAG,))
        # WITH A READER POOL, which is not decoration. `churn_readers.py`
        # stamps `extra.reader_ops_unit` -- the ops unit both halves of the
        # reader ratio must share -- and `OPS_UNIT_KEYS` names
        # `storage_engine` as one of its four axes. `Record.to_dict` drops
        # None at the TOP LEVEL only, so that nested axis is a literal `null`
        # on the wire for an engine with no storage engine. Without readers
        # the key is absent and the raw-text assertion below passes for the
        # wrong reason; it did, and this is what that cost.
        #
        # The baseline window is shortened the way `tests/test_t4b_readers.py`
        # shortens it, and patched on `churn` rather than on `churn_readers`
        # because `churn.run` reads the name out of its own module globals --
        # patching the defining module would silently run the real 30 s window
        # and cost this test thirty seconds per fraction. It did, before this
        # line existed.
        monkeypatch.setattr(churn_workload, "READER_BASELINE_WINDOW_S", 0.4)
        churn_workload.run(
            ctx, driver, dataset, index, ef_search=64, fractions=(0.10,),
            max_queries=N_QUERIES, indexed_rows=indexed_rows,
            driver_factory=driver.clone, readers=2)

        rows = written(ctx)
        # Every phase a unit writes is represented, so "no record claims one"
        # below is a statement about all of them.
        assert {r["phase"] for r in rows} == {
            records.PHASE_INGEST, records.PHASE_BUILD,
            records.PHASE_CALIBRATION, records.PHASE_CONCURRENCY,
            records.PHASE_FILTERED, records.PHASE_CHURN,
        }, sorted({r["phase"] for r in rows})
        claimed = [(r["phase"], r["storage_engine"]) for r in rows
                   if "storage_engine" in r]
        assert claimed == [], claimed
        # No record claims InnoDB in any field, nested ones included, which is
        # the whole point and is what a raw-text check is good for.
        raw = open(ctx.recorder.path).read()
        assert "InnoDB" not in raw

        # The ONE place the name still appears, deliberately, and asserted
        # rather than excluded: `extra.reader_ops_unit` is an identity tuple,
        # not a claim about the engine. `None` there says "this unit has no
        # storage engine", which is exactly what both halves of the ratio must
        # agree about -- and dropping the axis to make the text assertion
        # above simpler would weaken the guard for the one engine where the
        # axis is real, since MariaDB's InnoDB and MyISAM units must never be
        # paired with each other.
        units = [(r.get("extra") or {}).get("reader_ops_unit") for r in rows]
        units = [u for u in units if u is not None]
        assert units, "the reader pool wrote no ops unit to key the ratio on"
        for unit in units:
            assert set(unit) == set(churn_readers.OPS_UNIT_KEYS), unit
            assert unit["storage_engine"] is None, unit
            # The other three axes are populated, so the None above is this
            # field's own answer and not a unit that failed to resolve.
            assert unit["m"] == index.m
            assert unit["build_mode"] == index.build_mode
            assert unit["resource_pass"] == "normalized"
        # Which is a null on the wire, under `extra` and nowhere else.
        assert '"storage_engine": null' in raw
        assert raw.count('"storage_engine"') == len(units)


#: Warm-up queries per client for the bracketing test below, and the client
#: count and window it uses. Large enough that the warm-up is a visible
#: fraction of the window: at `wide_dataset`'s 1,000 queries each of two
#: clients gets a 500-query shard, so 300 warm-up queries each is 600 queries
#: of CPU spent before the window opens, against roughly a thousand inside it.
#: The fork context every cross-client counter below is built from. G3 made
#: the concurrency clients OS PROCESSES, and that broke a technique three
#: fixtures in this file were built on: put a dict or a `threading.Event` in
#: the test, have the client's `query` touch it, read it back in the test.
#: Under fork the client gets a SNAPSHOT -- its writes never come back and the
#: test's later writes never reach it -- so the plain-object form does not
#: fail loudly, it silently measures nothing. Each converted fixture carries
#: the honest and the plain-object reading in its docstring, and asserts the
#: shared value moved, so the conversion cannot rot back.
MP_FORK = multiprocessing.get_context("fork")

BRACKET_WARMUP = 300
BRACKET_CLIENTS = 2
BRACKET_WINDOW_S = 0.15

#: Microseconds of virtual CPU the bracketing tests charge per completed
#: query. One millisecond, so `extra.client_cpu_seconds` -- rounded to three
#: decimals -- carries exactly one query per unit and the delta reads back out
#: of the record without a tolerance of its own.
USEC_PER_QUERY = 1000


class TestTheClientsOwnCpuIsRecorded:
    """§1.4. The clients are forked OS processes sharing one container's
    cpuset; for a fast engine at 16-32 clients they can run out of CPU before
    the server does, and a curve that flattens for that reason is
    indistinguishable from one that flattens because the engine stopped
    scaling -- except that the first is a property of this harness rather than
    a finding about the database. `fork` never changes cgroup membership, so
    the one cgroup counter this class calibrates still covers all of them.
    """

    def _run(self, ctx, dataset, factory, **kw):
        options = dict(client_counts=(2,), duration_s=0.15, warmup=2,
                       max_queries=concurrency_workload.WHOLE_TEST_SET)
        options.update(kw)
        concurrency_workload.run(
            ctx, factory, dataset, index_spec(dataset), ef_search=32,
            **options)
        return written(ctx)[-1]

    def test_the_record_carries_the_ratio_and_both_of_its_inputs(self, ctx,
                                                                 dataset):
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        extra = self._run(ctx, dataset, driver.clone)["extra"]
        # The denominator is the cpuset the process may actually run on rather
        # than the host's CPU count, which in the normalized pass differ by
        # five times.
        assert extra["client_cpus"] == len(os.sched_getaffinity(0))
        assert extra["client_cpus"] > 0
        # A window that executed queries spent CPU doing it. Zero here would
        # be the measurement silently absent.
        assert extra["client_cpu_seconds"] > 0.0
        assert extra["client_cpu_util"] > 0.0
        assert isinstance(extra["client_bound"], bool)

    def test_an_unreadable_counter_is_unknown_rather_than_not_bound(
            self, ctx, dataset, monkeypatch):
        """None, not False.

        False is the claim "the clients were measured and were not the limit".
        On a container where the counter cannot be read -- which is what
        routing the reading through `docker exec` produces on every run -- that
        claim is wrong and nothing says so. A None in the record and an em dash
        in the report are the honest shape, and the report's footnote counts
        them.
        """
        monkeypatch.setattr(concurrency_workload, "read_own_cpu_usec",
                            lambda: None)
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        extra = self._run(ctx, dataset, driver.clone)["extra"]
        assert extra["client_cpu_seconds"] is None
        assert extra["client_cpu_util"] is None
        assert extra["client_bound"] is None
        # The cpuset is still known; only the counter was missing.
        assert extra["client_cpus"] == len(os.sched_getaffinity(0))

    def test_a_saturated_client_container_is_flagged_and_says_so(
            self, ctx, dataset, monkeypatch, capsys):
        """A counter that advances by one CPU-second per CPU per wall second.

        That is a container using every CPU it has, whatever the window turned
        out to be, so the expected utilisation is 1.0 by construction rather
        than by measurement -- the calibration of the real counter against a
        known number of saturated cores lives in tests/test_harness.py.
        """
        cpus = len(os.sched_getaffinity(0))
        monkeypatch.setattr(
            concurrency_workload, "read_own_cpu_usec",
            lambda: int(time.perf_counter() * 1e6) * cpus)
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        extra = self._run(ctx, dataset, driver.clone,
                          client_counts=(4,))["extra"]
        assert extra["client_cpu_util"] == pytest.approx(1.0, abs=0.02)
        assert extra["client_bound"] is True
        err = capsys.readouterr().err
        assert "CLIENT CPU-BOUND" in err, err
        # Naming the point, because the remedy is more client CPUs for that
        # point rather than a change anywhere in the code.
        assert "4 client(s)" in err, err

    def test_an_idle_client_container_is_not_flagged(self, ctx, dataset,
                                                     monkeypatch, capsys):
        """The other branch, and the reason False above is a measurement.

        A tenth of one CPU, against a cpuset of at least one: below the limit
        however many CPUs the box has.
        """
        monkeypatch.setattr(
            concurrency_workload, "read_own_cpu_usec",
            lambda: int(time.perf_counter() * 1e5))
        driver = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        extra = self._run(ctx, dataset, driver.clone)["extra"]
        assert extra["client_cpu_util"] < concurrency_workload.CLIENT_BOUND_UTIL
        assert extra["client_bound"] is False
        assert "CLIENT CPU-BOUND" not in capsys.readouterr().err

    def _counting(self, base, counter):
        """A driver factory that charges one virtual query per query answered.

        The virtual counter below returns `counter.value * USEC_PER_QUERY`,
        so the CPU delta the workload computes has a unit the record also
        reports independently -- queries -- and the assertion can be against
        `queries_executed` rather than against a tolerance of its own.

        **`counter` is a `multiprocessing.Value` and was a dict behind a
        `threading.Lock`.** G3 made the clients OS processes, so the increment
        now happens in a child: a dict would have stayed at 0 in the parent
        and the "virtual CPU counter" would have returned 0 for every reading,
        making the whole calibration vacuous rather than failing. Measured
        with the plain dict, two clients at 300 warm-up queries each:
        `state["queries"]` read back **0** against the 3,922 the clients
        answered. The `Value`'s own lock replaces the `threading.Lock`,
        because an increment that crosses processes needs a lock that does
        too.
        """
        def clone():
            client = base.clone()
            answer = client.query

            def query(vector, k):
                result = answer(vector, k)
                with counter.get_lock():
                    counter.value += 1
                return result

            client.query = query
            return client

        return clone

    def test_the_counter_brackets_the_window_and_not_the_warm_up(
            self, ctx, wide, monkeypatch):
        """The CPU delta covers the measured window only.

        Both readings are taken where the wall's own endpoints are taken: the
        first inside the barrier action, which runs after every client has
        finished warming up and before any is released, and the second
        immediately after the stop instant and before the stop flag goes up.
        Read anywhere earlier, the delta would include the warm-up -- hundreds
        of queries per client, charged to a window that did not run them --
        and the utilisation would be high for a reason the record could not
        show.
        """
        counter = MP_FORK.Value("q", 0)
        base = loaded_driver(wide, ef_half=64.0, ef_search=32)
        # Read in the PARENT, in the barrier action and at the stop instant,
        # against a counter the CHILDREN advance -- which is the whole point
        # of the shared `Value`. See `_counting`.
        monkeypatch.setattr(concurrency_workload, "read_own_cpu_usec",
                            lambda: counter.value * USEC_PER_QUERY)

        concurrency_workload.run(
            ctx, self._counting(base, counter), wide, index_spec(wide),
            ef_search=32, client_counts=(BRACKET_CLIENTS,),
            duration_s=BRACKET_WINDOW_S, warmup=BRACKET_WARMUP,
            max_queries=concurrency_workload.WHOLE_TEST_SET)

        row = written(ctx)[-1]
        charged = round(row["extra"]["client_cpu_seconds"]
                        * 1e6 / USEC_PER_QUERY)
        executed = row["queries_executed"]
        warmed = BRACKET_CLIENTS * BRACKET_WARMUP
        # The warm-up really happened, so there was something to exclude --
        # and the counter crossed the fork at all, which is what a plain dict
        # would have failed to do while reading a silent 0.
        assert counter.value >= executed + warmed, (
            f"the shared counter read {counter.value} against "
            f"{executed} + {warmed}; a value of 0 means the clients' "
            f"increments never reached the parent and this calibration is "
            f"measuring nothing")
        # And the charge is the window's own queries, within the handful each
        # client can complete between the stop instant and the second reading.
        assert executed <= charged <= executed + BRACKET_CLIENTS + 2, (
            f"charged {charged} queries of CPU against {executed} executed; "
            f"the warm-up was {warmed} queries, so a charge near "
            f"{executed + warmed} means a reading taken before the clients "
            f"warmed up")

    def test_the_figure_is_the_median_windows_and_not_another_windows(
            self, ctx, dataset, monkeypatch):
        """Three windows of deliberately different speed, so the median is
        known in advance rather than whichever one the scheduler produced.

        One client per window, each slower than the last, so sorting the three
        by throughput puts the second one in the middle every time. The CPU
        charged per window is then a label: 100 CPU-seconds is the median
        window's, 1 is the fastest window's and 10,000 is the slowest's. A
        figure taken from any window but the one whose QPS was reported names
        itself.
        """
        latencies = (0.0005, 0.001, 0.002)
        charges = (1.0, 100.0, 10_000.0)
        base = loaded_driver(dataset, ef_half=64.0, ef_search=32)
        # A `multiprocessing.Value` and not a dict, because G3 made the
        # clients OS processes and this factory is now called IN THE CHILD.
        # With a dict the parent's counter never advanced past call 0 and
        # every window's client ran at the same latency: measured, `made["n"]`
        # read back **0** after a three-window run and `qps_samples` came out
        # [1348.7, 1388.1, 1401.3] -- a 1.04x spread against the 1.5x this
        # test needs, so the median had no middle to pick and the fixture was
        # silently staging nothing.
        made = MP_FORK.Value("i", -1)

        def clone():
            with made.get_lock():
                made.value += 1
                mine = made.value
            sibling = base.clone()
            # Call 0 is `run`'s metadata driver, which issues no queries;
            # call 1 is the discarded warm-up window's client (ops version
            # 9) and calls 2..4 the single client of each measured window.
            if 2 <= mine <= len(latencies) + 1:
                sibling.latency_s = latencies[mine - 2]
            return sibling

        reads = {"n": -1}

        def window_cpu():
            # Twice per window: the open reads zero, the close reads that
            # window's charge. The first pair is the warm-up window's.
            reads["n"] += 1
            n = reads["n"] - 2
            if n < 0 or n % 2 == 0:
                return 0
            return int(charges[n // 2] * 1e6)

        monkeypatch.setattr(concurrency_workload, "read_own_cpu_usec",
                            window_cpu)
        # Long enough that even the slowest window covers the whole
        # 50-query pool. Shorter, and the three windows measure recall over
        # different numbers of queries, which trips the workload's own
        # recall-spread check -- a loud and correct complaint about a fixture
        # artefact, printed by a test that is not about recall at all.
        row = self._run(ctx, dataset, clone, client_counts=(1,),
                        duration_s=0.3, repeats=3)
        # The three windows really did differ in throughput, or there was no
        # median to pick: a flat set of samples would make the assertion below
        # pass for any selection rule.
        samples = sorted(row["extra"]["qps_samples"])
        assert samples[-1] > samples[0] * 1.5, samples
        # And the staging crossed the fork: four calls, so each window's
        # client really was handed its own latency. A plain dict reads 0 here.
        assert made.value == len(latencies) + 1, (
            f"the factory was called {made.value + 1} times as the PARENT "
            f"counts them; the three windows were therefore not staged and "
            f"the spread above is this box's noise")
        assert row["extra"]["client_cpu_seconds"] == pytest.approx(100.0), (
            f"the reported CPU figure is "
            f"{row['extra']['client_cpu_seconds']}, which is the "
            f"{'fastest' if row['extra']['client_cpu_seconds'] < 50 else 'slowest'} "
            f"window's rather than the median window's")
