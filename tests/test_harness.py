"""Unit tests for the parts of the harness that can be wrong silently.

Priority is on the code where a bug would corrupt results rather than crash:
recall math, percentiles, the Pareto frontier, filtered ground truth, CPU
topology detection, and resource resolution. A crash gets noticed; a subtly
wrong recall number gets published.

Run: python3 -m pytest tests/ -v      (from the vector-bench directory)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional  # noqa: E402

from harness import datasets as ds  # noqa: E402
from harness.metrics import latency, records, resources, sysinfo  # noqa: E402
from harness.workloads import concurrency as concurrency_workload  # noqa: E402


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_perfect_recall(self):
        truth = numpy.array([[1, 2, 3], [4, 5, 6]])
        got = [[1, 2, 3], [4, 5, 6]]
        assert ds.recall_at_k(got, truth, 3) == 1.0

    def test_zero_recall(self):
        truth = numpy.array([[1, 2, 3]])
        assert ds.recall_at_k([[7, 8, 9]], truth, 3) == 0.0

    def test_partial_recall_ignores_order(self):
        # Recall is set intersection: order within the result must not matter.
        truth = numpy.array([[1, 2, 3, 4]])
        assert ds.recall_at_k([[4, 3, 99, 98]], truth, 4) == pytest.approx(0.5)

    def test_truncates_to_k(self):
        # An engine returning more than k must not be credited for the extras.
        truth = numpy.array([[1, 2]])
        assert ds.recall_at_k([[1, 2, 3, 4]], truth, 2) == 1.0
        assert ds.recall_at_k([[9, 9, 1, 2]], truth, 2) == 0.0

    def test_short_result_set_scores_proportionally(self):
        # Filtered search can legitimately return fewer than k.
        truth = numpy.array([[1, 2, 3, 4]])
        assert ds.recall_at_k([[1, 2]], truth, 4) == pytest.approx(0.5)

    def test_empty_input_is_nan_not_zero(self):
        # Zero would silently look like a real, terrible result.
        assert numpy.isnan(ds.recall_at_k([], numpy.array([[1]]), 1))


# ---------------------------------------------------------------------------
# Filtered ground truth
# ---------------------------------------------------------------------------

class TestFilteredGroundTruth:
    def test_tags_are_deterministic_and_uniform(self):
        tags = ds.assign_tags(1000, buckets=100)
        assert len(tags) == 1000
        assert set(tags.tolist()) == set(range(100))
        assert numpy.array_equal(tags, ds.assign_tags(1000, buckets=100))

    def test_selectivity_conversion(self):
        assert ds.selectivity_to_threshold(0.01) == 1
        assert ds.selectivity_to_threshold(0.10) == 10
        assert ds.selectivity_to_threshold(0.50) == 50
        # Never zero: a 0.1% request must still select a non-empty set.
        assert ds.selectivity_to_threshold(0.001) == 1
        assert ds.selectivity_to_threshold(2.0) == 100

    def test_only_qualifying_rows_are_returned(self):
        rng = numpy.random.default_rng(0)
        train = rng.random((500, 8), dtype=numpy.float32)
        test = rng.random((5, 8), dtype=numpy.float32)
        truth = ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, k=5,
                                         tag_threshold=10, buckets=100)
        tags = ds.assign_tags(len(train))
        assert truth.shape == (5, 5)
        assert (tags[truth] < 10).all(), "returned a row that fails the filter"

    def test_matches_brute_force_euclidean(self):
        rng = numpy.random.default_rng(1)
        train = rng.random((200, 4), dtype=numpy.float32)
        test = rng.random((3, 4), dtype=numpy.float32)
        truth = ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, k=3,
                                         tag_threshold=50, buckets=100)
        tags = ds.assign_tags(len(train))
        qualifying = numpy.nonzero(tags < 50)[0]
        for i, q in enumerate(test):
            distances = numpy.linalg.norm(train[qualifying] - q, axis=1)
            expected = qualifying[numpy.argsort(distances)[:3]]
            assert set(truth[i].tolist()) == set(expected.tolist())

    def test_matches_brute_force_angular(self):
        rng = numpy.random.default_rng(2)
        train = rng.random((200, 4), dtype=numpy.float32)
        test = rng.random((3, 4), dtype=numpy.float32)
        truth = ds.filtered_ground_truth(train, test, ds.ANGULAR, k=3,
                                         tag_threshold=50, buckets=100)
        tags = ds.assign_tags(len(train))
        qualifying = numpy.nonzero(tags < 50)[0]
        norm = lambda a: a / numpy.linalg.norm(a, axis=-1, keepdims=True)
        subset = norm(train[qualifying])
        for i, q in enumerate(test):
            similarity = subset @ norm(q)
            expected = qualifying[numpy.argsort(-similarity)[:3]]
            assert set(truth[i].tolist()) == set(expected.tolist())

    def test_empty_filter_raises(self):
        train = numpy.zeros((10, 2), dtype=numpy.float32)
        test = numpy.zeros((1, 2), dtype=numpy.float32)
        with pytest.raises(ValueError):
            ds.filtered_ground_truth(train, test, ds.EUCLIDEAN, k=1,
                                     tag_threshold=0, buckets=100)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestLatency:
    def test_nearest_rank_percentile_returns_a_real_sample(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert latency.percentile(samples, q) in samples

    def test_percentile_bounds(self):
        samples = list(range(1, 101))
        assert latency.percentile(samples, 0.50) == 50
        assert latency.percentile(samples, 0.95) == 95
        assert latency.percentile(samples, 0.99) == 99
        assert latency.percentile(samples, 1.0) == 100

    def test_empty_percentile_is_nan(self):
        assert numpy.isnan(latency.percentile([], 0.5))

    def test_summarize_converts_seconds_to_ms(self):
        stats = latency.summarize([0.001, 0.002, 0.003])
        assert stats.count == 3
        assert stats.mean_ms == pytest.approx(2.0)
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.max_ms == pytest.approx(3.0)

    def test_warmup_samples_are_excluded(self):
        collector = latency.LatencyCollector(warmup=3)
        collector.start()
        for value in (10.0, 10.0, 10.0, 1.0, 1.0):
            collector.add(value)
        collector.stop()
        stats = collector.stats()
        assert stats.count == 2, "warm-up queries leaked into the measurement"
        assert stats.mean_ms == pytest.approx(1000.0)

    def test_multi_client_qps_uses_wall_clock(self):
        # With N clients, summing per-query latencies would count concurrent
        # work N times over and inflate QPS by roughly N.
        a = latency.LatencyCollector()
        b = latency.LatencyCollector()
        for c in (a, b):
            c.start()
            for _ in range(100):
                c.add(0.01)
            c.stop()
        merged = latency.merge_collectors([a, b], wall_seconds=1.0)
        assert merged["qps"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Trimming a measured window to the queries that finished inside it
# ---------------------------------------------------------------------------

class TestWindowTrimming:
    """QPS is counted queries over wall time, and the two used to disagree
    about where the window ended.

    Every client finishes the query it holds when `stop` is set and records
    it, but the wall clock stops at `stop`. So the numerator included one query
    per client that the denominator did not cover, and the overstatement is
    that query over the ones the client did finish:
    `1 / queries-completed-per-client`, about `latency / window`. The client
    count cancels, so it is 2.5% over a two-second window of 50 ms queries at
    any client count from 1 to 32 -- what makes it bite is a slow engine over
    a short window, not a wide client sweep.

    These tests use synthetic spans with exact instants, so every number below
    is arithmetic rather than a measurement. The window is [0, 1] throughout.
    """

    #: A span wholly inside the window, 100 ms long.
    INSIDE = (0.10, 0.20)
    #: A span that began inside the window and finished after the cutoff:
    #: the in-flight query §1.3 is about. Deliberately a full second long, so
    #: a merge that keeps it cannot help but move the percentiles too.
    OVERRUNS = (0.90, 1.90)
    #: A span that began before the window opened, which is what a client that
    #: passed the barrier late leaves behind.
    STARTED_EARLY = (-0.30, 0.40)

    def _collector(self, *spans) -> latency.LatencyCollector:
        collector = latency.LatencyCollector()
        collector.start()
        for started, finished in spans:
            collector.add_span(started, finished)
        collector.stop()
        return collector

    def test_the_record_keys_are_the_names_the_report_will_read(self):
        # Pinned as literals because they are a published part of the record,
        # not an internal name: a rename here silently renames a field in
        # every concurrency record. The workload and this test both spell the
        # key through the constant, so only this assertion notices.
        assert latency.INFLIGHT_DISCARDED == "inflight_discarded"
        assert latency.PREWINDOW_DISCARDED == "prewindow_discarded"

    def test_a_query_finishing_after_the_cutoff_is_not_counted(self):
        collector = self._collector(self.INSIDE, (0.3, 0.4), self.OVERRUNS)
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 2
        assert merged["qps"] == pytest.approx(2.0)
        assert merged[latency.INFLIGHT_DISCARDED] == 1
        assert merged[latency.PREWINDOW_DISCARDED] == 0

    def test_a_query_finishing_exactly_at_the_cutoff_is_counted(self):
        # The boundary belongs to the window: `finished > cutoff` drops, not
        # `>=`. A query that completed at the instant the clock stopped was
        # paid for inside the window the denominator measures.
        merged = latency.merge_collectors(
            [self._collector((0.5, 1.0))],
            wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 1
        assert merged[latency.INFLIGHT_DISCARDED] == 0

    def test_the_discarded_query_is_gone_from_the_percentiles_too(self):
        # Not only from the count. The in-flight query is the one that ran
        # longest -- it is still running -- so leaving its latency in while
        # dropping it from the count would move p99 by more than the count
        # error it was fixing.
        collector = self._collector(self.INSIDE, (0.3, 0.4), self.OVERRUNS)
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        # Both kept spans are exactly 100 ms, so every percentile is 100.0 and
        # the 1,000 ms overrun cannot hide inside a tolerance.
        assert merged["latency_p99_ms"] == pytest.approx(100.0)
        assert merged["latency_mean_ms"] == pytest.approx(100.0)

    def test_a_query_that_began_before_the_window_is_not_counted(self):
        # Should never happen: every client waits on the barrier and the
        # window opens after it releases. Recording it is what turns "should
        # never happen" into something the record can be checked against.
        collector = self._collector(self.STARTED_EARLY, self.INSIDE)
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 1
        assert merged[latency.PREWINDOW_DISCARDED] == 1
        assert merged[latency.INFLIGHT_DISCARDED] == 0
        # 700 ms long against the survivor's 100 ms, so a merge that counted
        # it would be visible here even if the count above were right.
        assert merged["latency_mean_ms"] == pytest.approx(100.0)

    def test_every_span_lands_in_exactly_one_of_the_three_buckets(self):
        # Kept + discarded == recorded, so neither counter can double-count a
        # span that both started early and overran, and no span can vanish
        # without being accounted for in one of the two fields the record
        # publishes.
        spans = (self.INSIDE, self.OVERRUNS, self.STARTED_EARLY,
                 (-0.5, 1.5), (0.0, 1.0))
        merged = latency.merge_collectors(
            [self._collector(*spans)],
            wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        accounted = (merged["queries_executed"]
                     + merged[latency.INFLIGHT_DISCARDED]
                     + merged[latency.PREWINDOW_DISCARDED])
        assert accounted == len(spans)
        # And the two kept are the two that fit: [0.1, 0.2] and [0.0, 1.0].
        assert merged["queries_executed"] == 2

        # WHICH bucket, not just how many buckets. The accounting assertion
        # above is satisfied by either ordering of the two tests inside
        # `span_disposition`, and `(-0.5, 1.5)` offends at both ends: it
        # started before the window opened AND outlived the cutoff. The
        # cutoff is tested first, so it counts as in-flight — 2 in-flight
        # here, not 1.
        #
        # This is diagnostic rather than numeric: the kept set, the QPS and
        # the percentiles are identical either way. What changes is the one
        # field whose entire value is being zero. `prewindow_discarded` is
        # documented to mean "the window's two ends disagree about when it
        # began", and under the other ordering an ordinary in-flight query
        # that merely happened to start early raises that alarm. Swapping the
        # two tests passed every other test in this repository.
        assert merged[latency.INFLIGHT_DISCARDED] == 2
        assert merged[latency.PREWINDOW_DISCARDED] == 1

    def test_without_a_cutoff_nothing_is_trimmed(self):
        # The way back, and what the other three workloads get: they measure
        # to completion rather than to a deadline, so there is no cutoff to
        # pass and no span of theirs may be dropped.
        collector = self._collector(self.INSIDE, self.OVERRUNS,
                                    self.STARTED_EARLY)
        merged = latency.merge_collectors([collector], wall_seconds=1.0)
        assert merged["queries_executed"] == 3
        assert merged[latency.INFLIGHT_DISCARDED] == 0
        assert merged[latency.PREWINDOW_DISCARDED] == 0

    def test_a_window_with_nothing_to_trim_is_untouched(self):
        # The trimmed and untrimmed merges of spans that all fit must agree
        # exactly, or the cutoff is doing something besides trimming.
        spans = ((0.0, 0.1), (0.2, 0.35), (0.4, 0.9))
        trimmed = latency.merge_collectors(
            [self._collector(*spans)],
            wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        untrimmed = latency.merge_collectors(
            [self._collector(*spans)], wall_seconds=1.0)
        for key in ("queries_executed", "qps", "latency_mean_ms",
                    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
            assert trimmed[key] == pytest.approx(untrimmed[key]), key

    def test_durations_recorded_without_a_span_survive_a_cutoff(self):
        """`add(seconds)` is what the filtered and churn workloads call.

        A duration carries no instants, so there is nothing to compare
        against a cutoff and the only honest thing to do is keep it. If a
        cutoff silently dropped spanless samples, passing one would empty
        those workloads' percentiles rather than trim them.
        """
        collector = latency.LatencyCollector()
        collector.start()
        for _ in range(4):
            collector.add(0.25)
        collector.stop()
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 4
        assert merged["latency_mean_ms"] == pytest.approx(250.0)
        assert merged[latency.INFLIGHT_DISCARDED] == 0

    def test_add_span_and_add_agree_about_the_duration(self):
        # One quantity, two spellings. `add_span` must not be an independent
        # path to a different latency.
        spanned = self._collector((10.5, 10.75))
        duration = latency.LatencyCollector()
        duration.start()
        duration.add(0.25)
        duration.stop()
        assert (spanned.stats().mean_ms
                == pytest.approx(duration.stats().mean_ms))
        assert spanned.measured_seconds == pytest.approx(0.25)

    def test_warmup_still_counts_spans(self):
        # The prefix discard is about which queries are measured, not about
        # how they were recorded, so `add_span` has to honour it exactly as
        # `add` does -- the concurrency workload passes warmup=0 today, and a
        # span path that ignored it would make that silently load-bearing.
        collector = latency.LatencyCollector(warmup=2)
        collector.start()
        for started in (0.0, 0.1, 0.2, 0.3):
            collector.add_span(started, started + 0.05)
        collector.stop()
        assert collector.stats().count == 2

    def test_a_window_that_kept_nothing_reports_no_latency_at_all(self):
        """NaN is not JSON, and this is the path that reaches it.

        `summarize([])` returns NaN for every percentile and `json.dumps`
        writes that as the bare literal `NaN`: Python's `json.loads` takes it
        as an extension, the browser's `JSON.parse` throws. Before the window
        was trimmed the sample set was never empty, because every client
        completed at least one query; a configuration whose queries outlast
        the window now keeps nothing.
        """
        collector = self._collector((0.5, 1.5), (1.5, 2.5))
        merged = latency.merge_collectors(
            [collector], wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        assert merged["queries_executed"] == 0
        assert merged[latency.INFLIGHT_DISCARDED] == 2
        for field in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                      "latency_p99_ms"):
            assert merged[field] is None, (field, merged[field])
        # Zero completions over the window is what was observed, and it both
        # serialises and plots.
        assert merged["qps"] == 0.0

    def test_the_null_is_scoped_to_the_merged_window(self):
        """`summarize` itself is deliberately untouched.

        It predates this window and the build, filtered and churn workloads
        all reach their percentiles through it. Those loops run to completion,
        so an empty sample set there means nothing was asked of the engine at
        all — a different fact from "nothing finished in time", and not one
        this change has any business relabelling. Pinned because the obvious
        tidy-up is to move the null down into `summarize` and make all four
        workloads share it.
        """
        assert numpy.isnan(latency.summarize([]).p99_ms)
        empty = latency.LatencyCollector()
        empty.start()
        empty.stop()
        assert numpy.isnan(empty.stats().p99_ms)
        assert numpy.isnan(empty.stats().mean_ms)

    def test_trimming_cannot_let_more_work_through_than_the_clients_could_do(
            self):
        """Little's law, which is the invariant §1.3 restores.

        Mean latency x QPS is the average number of queries in flight, and a
        client runs one query at a time, so it cannot exceed the client count.
        Counting a query that outlived the window breaks that: here two
        clients each hold a 0.6 s query across the cutoff of a 1 s window, so
        untrimmed the four queries average 0.55 s at 4 q/s = 2.2 queries in
        flight across two clients, which no two clients can have done.

        This is the property the workload test pins, where the spans are real
        and no absolute rate can be asserted. Stated here on exact numbers so
        that what it means is not left to a measurement.
        """
        clients = [self._collector((0.1, 0.6), (0.6, 1.2)),
                   self._collector((0.2, 0.7), (0.7, 1.3))]
        trimmed = latency.merge_collectors(
            clients, wall_seconds=1.0, window_start=0.0, cutoff=1.0)
        untrimmed = latency.merge_collectors(clients, wall_seconds=1.0)
        in_flight = trimmed["qps"] * trimmed["latency_mean_ms"] / 1000.0
        assert in_flight <= len(clients)
        assert in_flight == pytest.approx(1.0)
        over = untrimmed["qps"] * untrimmed["latency_mean_ms"] / 1000.0
        assert over > len(clients), over


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class TestRecords:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "r.jsonl")
        with records.Recorder(path) as recorder:
            recorder.write(records.Record(
                run_id="r1", engine="mariadb", phase=records.PHASE_RECALL,
                dataset="d", timestamp="t", recall_at_k=0.9, qps=100.0,
            ))
        loaded = list(records.read_records(path))
        assert len(loaded) == 1
        assert loaded[0]["recall_at_k"] == 0.9

    def test_none_fields_are_omitted(self):
        record = records.Record(run_id="r", engine="e", phase=records.PHASE_INGEST,
                                dataset="d", timestamp="t")
        as_dict = record.to_dict()
        assert "recall_at_k" not in as_dict
        assert as_dict["engine"] == "e"

    def test_unknown_phase_rejected(self, tmp_path):
        with records.Recorder(str(tmp_path / "r.jsonl")) as recorder:
            with pytest.raises(ValueError):
                recorder.write(records.Record(
                    run_id="r", engine="e", phase="nonsense",
                    dataset="d", timestamp="t"))

    def test_truncated_tail_is_tolerated(self, tmp_path):
        # A killed run leaves one partial line; everything before it is valid.
        path = tmp_path / "r.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
        assert list(records.read_records(str(path))) == [{"a": 1}, {"b": 2}]

    def test_corruption_before_valid_data_raises(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text('{"a": 1}\n{broken\n{"c": 3}\n')
        with pytest.raises(ValueError):
            list(records.read_records(str(path)))


# ---------------------------------------------------------------------------
# Storage engine
# ---------------------------------------------------------------------------

class TestOnlyTheDriverDecidesTheStorageEngine:
    """A record may name a storage engine only where there is one to name.

    `IndexSpec.storage_engine` is a MySQL-family `ENGINE=` clause.
    `harness/main.py` defaults `--storage-engine` to `InnoDB` and
    `orchestrator/cli.py:_ops_storage_engines` passes `InnoDB` for every
    engine that is not MariaDB, so the field arrives populated for all seven
    engines and only three of them put it in a CREATE TABLE.
    `RunContext.record_defaults` stamped it regardless, which published this,
    counted over the seven-engine run `full-20261015-215006`
    (`report/records.jsonl`, by engine and storage_engine):

        alisql     InnoDB 161                 percona  InnoDB 433
        mariadb    InnoDB 163                 pgvector InnoDB 268
        mariadb123 InnoDB 163                 pgvector heap   200
        mongodb    InnoDB 120                 valkey   InnoDB 128
        mongodb    wiredTiger 37              valkey   memory 100

    The honest rows are the `recall_qps` ones, written by the ann overlay
    modules; every other phase is the ops harness. So pgvector, Valkey and
    Percona Search each reported two different storage engines inside one run,
    and the false value was the one on the build-cost, concurrency, filtered
    and churn records. PostgreSQL has no InnoDB, and neither does Valkey or
    MongoDB.

    MariaDB's InnoDB-versus-MyISAM sweep is the one real within-engine
    comparison here (`orchestrator/cli.py:_ops_storage_engines`,
    `docs/05-methodology.md`), so the fix cannot be a blanket None: the first
    test below is what stops it becoming one.
    """

    #: Every engine the harness drives, and the storage engine its records are
    #: entitled to claim. Spelled per ENGINE rather than per driver class
    #: because that is the axis the defect was measured on, and because
    #: `mariadb`/`mariadb123` sharing one class is exactly the kind of mapping
    #: a per-class table hides.
    EXPECTED = {
        "mariadb": "InnoDB", "mariadb123": "InnoDB", "alisql": "InnoDB",
        "percona": "InnoDB", "pgvector": None, "mongodb": None,
        "valkey": None,
        # MyVector's table is an ordinary InnoDB table; the plugin's graph is
        # beside it in files, but the rows the record describes are InnoDB's.
        "myvector": "InnoDB",
        # VillageSQL's table is InnoDB too; the SVECTOR storage and the HNSW
        # graph are extension segments inside the same tablespace.
        "villagesql": "InnoDB",
    }

    @staticmethod
    def _spec(storage_engine: str = "InnoDB"):
        from harness.drivers.base import IndexSpec
        return IndexSpec(dim=8, m=16, metric="euclidean",
                         storage_engine=storage_engine)

    def _drivers(self):
        """One offline instance per engine, keyed by engine name.

        Constructed, not connected: none of the four client libraries is
        installed on the host, and `__init__` needs none of them.

        The registry check lives here rather than in a test of its own, so
        that it is carried by tests a mutation of the fix can fail. A table
        that fell behind `_driver_table` would leave a new engine's answer
        untested, which is how an InnoDB default survived three engines with
        no storage engine being added beneath it -- but a standalone
        assertion about two dictionaries passes against the defect and
        against the repair alike, and a probe that cannot fail is not
        coverage.
        """
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.postgres import _driver_table

        table = _driver_table()
        assert set(self.EXPECTED) == set(table)
        # Both branches are populated, so neither test below runs over an
        # empty collection and passes by doing nothing.
        assert sum(1 for v in self.EXPECTED.values() if v is not None) == 6
        assert sum(1 for v in self.EXPECTED.values() if v is None) == 3
        spec = ConnectionSpec(host="h", port=1)
        return {engine: cls(spec) for engine, cls in table.items()}

    def test_the_mysql_family_reports_the_engine_its_ddl_used(self):
        """And reports MyISAM when MyISAM is what was asked for.

        Asserted against the DDL the driver builds rather than against the
        `IndexSpec` it was handed, so this cannot pass for a driver that
        reports a storage engine it did not create the table with -- which is
        the whole failure being repaired, one engine over.
        """
        drivers = self._drivers()
        family = [name for name, want in self.EXPECTED.items()
                  if want is not None]
        assert family, "no MySQL-family engine to check"
        for engine in family:
            driver = drivers[engine]
            for storage in ("InnoDB", "MyISAM"):
                index = self._spec(storage)
                ddl = driver._create_table_ddl(index, with_index=True)
                assert ddl.rstrip().endswith(f") ENGINE={storage}"), ddl
                assert driver.storage_engine_in_use(index) == storage, engine

    def test_no_other_driver_claims_one(self):
        """None, whatever the spec says, because there is nothing to report.

        Both values are tried: a driver that returned the spec's value
        unconditionally passes a single-value check as long as that value is
        the one the test expected to see suppressed.
        """
        drivers = self._drivers()
        others = [name for name, want in self.EXPECTED.items() if want is None]
        assert others, "no non-MySQL engine to check"
        for engine in others:
            for storage in ("InnoDB", "MyISAM"):
                driver = drivers[engine]
                got = driver.storage_engine_in_use(self._spec(storage))
                assert got is None, f"{engine} claimed {got!r}"

    def test_record_defaults_asks_the_driver_and_not_the_spec(self, tmp_path):
        """The one line the defect lived on, both ways round.

        A real `MariaDBDriver` and a real `PostgresDriver`, with only
        `server_version` stubbed -- that call needs a live server and has
        nothing to do with this field, while `storage_engine_in_use` is the
        drivers' own.

        * Honest: MariaDB's record says MyISAM, pgvector's says nothing.
        * Mutant (`index.storage_engine` back in `record_defaults`): both say
          MyISAM, and the pgvector one is a claim about an engine that has no
          storage engines.
        """
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.mysql_family import MariaDBDriver
        from harness.drivers.postgres import PostgresDriver
        from harness.workloads.context import RunContext

        class _OfflineMariaDB(MariaDBDriver):
            def server_version(self) -> str:
                return "11.8.8-offline"

        class _OfflinePostgres(PostgresDriver):
            def server_version(self) -> str:
                return "17.0-offline"

        class _Dataset:
            name = "d"
            metric = "euclidean"

        recorder = records.Recorder(str(tmp_path / "ops.jsonl"))
        try:
            ctx = RunContext(run_id="r", resource_pass="normalized",
                             recorder=recorder, engine="e")
            conn = ConnectionSpec(host="h", port=1)
            index = self._spec("MyISAM")
            mariadb = ctx.record_defaults(_OfflineMariaDB(conn), _Dataset(),
                                          index)
            pgvector = ctx.record_defaults(_OfflinePostgres(conn), _Dataset(),
                                           index)
            assert mariadb["storage_engine"] == "MyISAM"
            assert pgvector["storage_engine"] is None

            # And None leaves the written record with no such key at all,
            # rather than a null. That is what `report/loaders.py:memory_stem`
            # needs: it suffixes the `mem-*.jsonl` name it reconstructs only
            # for a storage engine that is not InnoDB, and the orchestrator
            # named that file from its own `InnoDB`, so an absent field takes
            # the same unsuffixed branch and the sampled peak RSS still joins.
            for common in (mariadb, pgvector):
                recorder.write(records.Record(
                    **common, phase=records.PHASE_BUILD))
        finally:
            recorder.close()

        written = list(records.read_records(recorder.path))
        assert written[0]["storage_engine"] == "MyISAM"
        assert "storage_engine" not in written[1], written[1]

    def test_the_writer_reads_the_driver_in_source(self):
        """No workload may stamp the field from an `IndexSpec` again.

        `record_defaults` is the choke point every ops record's provenance
        passes through, and the field now has exactly one producer. A second
        one would be invisible to the tests above, which drive the choke
        point: a workload that overwrote `storage_engine` in its own
        `Record(**common, ...)` call would restore the defect for its phase
        alone.
        """
        import glob
        import re

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_src = open(os.path.join(root, "harness", "workloads",
                                        "context.py")).read()
        writer = [line for line in context_src.splitlines()
                  if re.search(r'^\s*"storage_engine":', line)]
        assert len(writer) == 1, writer
        assert "driver.storage_engine_in_use(index)" in writer[0], writer[0]

        workloads = glob.glob(os.path.join(root, "harness", "workloads",
                                           "*.py"))
        assert len(workloads) > 5, workloads
        for path in workloads:
            src = open(path).read()
            assert "index.storage_engine" not in src, path
            if os.path.basename(path) != "context.py":
                assert '"storage_engine":' not in src, path


# ---------------------------------------------------------------------------
# CPU topology
# ---------------------------------------------------------------------------

class TestSysinfo:
    def test_collect_returns_plausible_values(self):
        info = sysinfo.collect()
        assert info.cpu.logical_cpus >= 1
        assert info.total_ram_bytes > 0
        assert isinstance(info.cpu.simd_flags, list)

    def test_cpuset_formatting_collapses_runs(self):
        assert sysinfo.format_cpuset([0, 1, 2, 3]) == "0-3"
        assert sysinfo.format_cpuset([0, 2, 4]) == "0,2,4"
        assert sysinfo.format_cpuset([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
        assert sysinfo.format_cpuset([3]) == "3"
        assert sysinfo.format_cpuset([]) == ""

    def test_cpulist_parsing(self):
        assert sysinfo._parse_cpulist("0-3") == [0, 1, 2, 3]
        assert sysinfo._parse_cpulist("0,2,4") == [0, 2, 4]
        assert sysinfo._parse_cpulist("0-2,8,10-11") == [0, 1, 2, 8, 10, 11]
        assert sysinfo._parse_cpulist("") == []
        assert sysinfo._parse_cpulist(None) == []

    def test_hybrid_classes_are_disjoint(self):
        topology = sysinfo.detect_hybrid_topology()
        overlap = set(topology["performance"]) & set(topology["efficiency"])
        assert not overlap, f"a CPU was classified as both P and E: {overlap}"

    def test_recommended_cpuset_excludes_smt_siblings(self):
        chosen = sysinfo.recommended_cpuset(1000, allow_smt=False)
        siblings = sysinfo.hyperthread_siblings()
        if not siblings:
            pytest.skip("no topology information available")
        seen = set()
        for cpu in chosen:
            group = frozenset(siblings.get(cpu, [cpu]))
            assert group not in seen, f"two SMT siblings selected: {group}"
            seen.add(group)

    def test_recommended_cpuset_respects_the_requested_size(self):
        assert len(sysinfo.recommended_cpuset(2, allow_smt=True)) <= 2
        assert len(sysinfo.recommended_cpuset(0)) == 0


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

class TestPareto:
    def _frontier(self, points):
        from report.loaders import pareto_frontier
        return pareto_frontier(points)

    def test_dominated_points_are_dropped(self):
        points = [
            {"recall_at_k": 0.90, "qps": 1000},   # on the frontier
            {"recall_at_k": 0.90, "qps": 500},    # dominated: same recall, slower
            {"recall_at_k": 0.95, "qps": 800},    # on the frontier
            {"recall_at_k": 0.80, "qps": 900},    # dominated by the 0.90/1000 point
        ]
        frontier = self._frontier(points)
        assert {(p["recall_at_k"], p["qps"]) for p in frontier} == {(0.90, 1000), (0.95, 800)}

    def test_frontier_is_ordered_by_recall(self):
        points = [{"recall_at_k": r, "qps": 1000 - r * 500}
                  for r in (0.5, 0.7, 0.9, 0.99)]
        frontier = self._frontier(points)
        recalls = [p["recall_at_k"] for p in frontier]
        assert recalls == sorted(recalls)

    def test_incomplete_points_are_ignored(self):
        points = [{"recall_at_k": 0.9, "qps": None}, {"recall_at_k": None, "qps": 10},
                  {"recall_at_k": 0.8, "qps": 50}]
        assert len(self._frontier(points)) == 1

    def test_empty_input(self):
        assert self._frontier([]) == []


class TestSubsetGroundTruth:
    """Ground truth must describe the rows the engine actually received.

    Profiles can load part of the training set (smoke loads 20,000 of 60,000).
    Scoring against ground truth computed over the full set would point at rows
    the engine never got — understating recall for every engine at once, which
    looks like a real result rather than a harness fault.
    """

    def _dataset(self, n=500, dim=8, queries=4):
        rng = numpy.random.default_rng(3)
        train = rng.random((n, dim), dtype=numpy.float32)
        test = rng.random((queries, dim), dtype=numpy.float32)
        d2 = ((test ** 2).sum(1)[:, None] - 2 * test @ train.T
              + (train ** 2).sum(1)[None, :])
        neighbors = numpy.argsort(d2, axis=1)[:, :10]
        return ds.Dataset("t", train, test, neighbors, None, ds.EUCLIDEAN)

    def test_never_references_unloaded_rows(self, tmp_path):
        dataset = self._dataset()
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5, indexed_rows=100)
        assert truth.max() < 100

    def test_matches_brute_force_over_the_subset(self, tmp_path):
        dataset = self._dataset()
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5, indexed_rows=100)
        for i, q in enumerate(dataset.test):
            distances = numpy.linalg.norm(dataset.train[:100] - q, axis=1)
            expected = numpy.argsort(distances)[:5]
            assert set(truth[i].tolist()) == set(expected.tolist())

    def test_full_load_reuses_shipped_neighbours(self, tmp_path):
        # Recomputing a 1M-row ground truth that already ships in the HDF5 file
        # would add hours to every run for no benefit.
        dataset = self._dataset()
        truth = ds.cached_ground_truth(str(tmp_path), dataset, 5, indexed_rows=None)
        assert numpy.array_equal(truth, dataset.neighbors[:, :5])

    def test_filtered_truth_also_respects_the_subset(self, tmp_path):
        dataset = self._dataset()
        truth = ds.cached_filtered_ground_truth(
            str(tmp_path), dataset, 5, 0.10, indexed_rows=100)
        tags = ds.assign_tags(100)
        assert truth.max() < 100
        assert (tags[truth] < 10).all()

    def test_row_count_is_part_of_the_cache_key(self, tmp_path):
        # Otherwise a subset run would poison the cache for a full run.
        dataset = self._dataset()
        a = ds.cached_filtered_ground_truth(
            str(tmp_path), dataset, 5, 0.50, indexed_rows=100)
        b = ds.cached_filtered_ground_truth(
            str(tmp_path), dataset, 5, 0.50, indexed_rows=300)
        assert a.max() < 100
        assert b.max() >= 100, "the 300-row truth was served from the 100-row cache"


class TestProgress:
    """Progress reporting must be time-driven and must not spam.

    Every long phase in this harness independently shipped with no output, so a
    working run was indistinguishable from a hung one. These guard the shared
    helper that replaced four near-identical loops.
    """

    def _capture(self):
        import io
        return io.StringIO()

    def test_reports_rate_and_eta_on_a_time_interval(self):
        import time as _t
        from harness.progress import Progress
        buf = self._capture()
        p = Progress(100, "widgets", prefix="eng", interval_s=0.05, stream=buf)
        for _ in range(100):
            p.step()
            _t.sleep(0.002)
        p.finish()
        out = buf.getvalue()
        assert "widgets" in out and "ETA" in out and "/s" in out
        assert out.count("\n") >= 2

    def test_silent_when_work_finishes_inside_one_interval(self):
        # A count-based trigger would print here; a time-based one must not.
        from harness.progress import Progress
        buf = self._capture()
        p = Progress(10, "fast", interval_s=60, stream=buf)
        for _ in range(10):
            p.step()
        p.finish()
        assert buf.getvalue() == ""

    def test_heartbeat_reports_liveness_then_completion(self):
        import time as _t
        from harness.progress import Heartbeat
        buf = self._capture()
        with Heartbeat("long thing", prefix="eng", interval_s=0.05, stream=buf):
            _t.sleep(0.22)
        out = buf.getvalue()
        assert "still running" in out
        assert "done in" in out

    def test_heartbeat_silent_for_short_operations(self):
        from harness.progress import Heartbeat
        buf = self._capture()
        with Heartbeat("quick", interval_s=60, stream=buf):
            pass
        assert buf.getvalue() == ""

    def test_interval_is_configurable_from_the_environment(self):
        # Operators on very slow hardware need to widen it without a code change.
        import importlib, os
        import harness.progress as prog
        old = os.environ.get("VB_PROGRESS_INTERVAL")
        try:
            os.environ["VB_PROGRESS_INTERVAL"] = "7"
            importlib.reload(prog)
            assert prog.DEFAULT_INTERVAL_S == 7.0
        finally:
            if old is None:
                os.environ.pop("VB_PROGRESS_INTERVAL", None)
            else:
                os.environ["VB_PROGRESS_INTERVAL"] = old
            importlib.reload(prog)


# ---------------------------------------------------------------------------
# Ops measurement version
# ---------------------------------------------------------------------------

class TestOpsMeasurementVersion:
    """Every ops record must say which measurement path produced it.

    `ANN_MEASUREMENT_VERSION` already does this for the recall/QPS path, where
    it feeds the fingerprint that keeps a 16 GB curve out of a 64 GB report.
    The ops path had no equivalent, so a harness change that altered what a
    number means left no trace in the number: a concurrency figure measured
    before the change and one measured after were indistinguishable in the
    record, and the report drew them as one series.

    The stamp is applied by `Recorder.write` rather than by
    `RunContext.record_defaults` -- the choke point every workload's provenance
    already passes through -- because four of the workloads pass `extra=` to
    `Record()` explicitly and an `extra` key in the defaults dict collides with
    that keyword. `Recorder.write` is the only point every ops record provably
    passes through, which is why `ALL_PHASES` is enforced there too.
    """

    def _written(self, tmp_path, record):
        path = str(tmp_path / "ops.jsonl")
        with records.Recorder(path) as recorder:
            recorder.write(record)
        return list(records.read_records(path))[0]

    def test_the_constant_starts_at_two(self):
        # Today's behaviour is implicitly 1: a record written before the
        # constant existed carries no version at all, so 1 is the value every
        # reader must infer for it and the first explicit value is 2.
        assert records.OPS_MEASUREMENT_VERSION >= 2
        assert records.IMPLICIT_OPS_MEASUREMENT_VERSION == 1
        assert records.OPS_MEASUREMENT_VERSION_KEY == "ops_measurement_version"

    def test_every_phase_carries_the_version(self, tmp_path):
        """ALL_PHASES, not a sample: a new phase must not be able to opt out."""
        for phase in records.ALL_PHASES:
            got = self._written(tmp_path, records.Record(
                run_id="r", engine="e", phase=phase, dataset="d", timestamp="t"))
            assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
                records.OPS_MEASUREMENT_VERSION, phase

    def test_a_workload_supplied_extra_is_kept_beside_the_version(self, tmp_path):
        # build.py writes extra={**driver.capabilities(), 'separable_build': ...}.
        # The stamp must neither replace that dict nor be shadowed by it.
        got = self._written(tmp_path, records.Record(
            run_id="r", engine="e", phase=records.PHASE_BUILD, dataset="d",
            timestamp="t", extra={"separable_build": False,
                                  "incremental_index": True}))
        assert got["extra"]["separable_build"] is False
        assert got["extra"]["incremental_index"] is True
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION

    def test_the_version_is_not_a_default_a_record_can_preset(self, tmp_path):
        """The stamp is unconditional, not setdefault.

        The version says which code produced the number. A record that arrives
        already claiming one is making a claim about a measurement this
        Recorder did not take -- a merge or replay tool feeding old records
        through a new Recorder, say -- and honouring it would relabel this
        run's own output. `setdefault` here passes every other test in this
        class, so this is the only one that pins it.
        """
        got = self._written(tmp_path, records.Record(
            run_id="r", engine="e", phase=records.PHASE_BUILD, dataset="d",
            timestamp="t",
            extra={records.OPS_MEASUREMENT_VERSION_KEY: 999}))
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] != 999

    def test_the_stamp_leaves_the_payload_it_was_given_alone(self):
        """A direct unit test of the function, which is where the copy is.

        The previous version of this guard asserted that `Recorder.write` left
        the caller's `Record` untouched and could not fail: `to_dict()` goes
        through `dataclasses.asdict`, which deep-copies `extra`, so the stamp
        never sees the caller's dict however it is written. asdict was doing
        the work the test credited to this code -- a stamp written straight
        into `payload` passed the whole suite.

        What is genuinely this function's contract is that it copies rather
        than writes, which is what makes it safe for a caller that has not
        been through asdict. The sibling test below pins the deep copy itself.
        """
        payload = {"phase": records.PHASE_BUILD,
                   "extra": {"separable_build": False}}
        stamped = records.stamp_measurement_version(payload)
        assert payload == {"phase": records.PHASE_BUILD,
                           "extra": {"separable_build": False}}
        assert stamped["extra"] is not payload["extra"]
        assert stamped["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION

    def test_to_dict_hands_back_a_private_extra(self):
        """The deep copy `Recorder.write` relies on for the caller's Record.

        Not decoration: `to_dict` is a hot path that builds a dict per record
        and `asdict`'s recursive copy is the obvious thing to replace with a
        comprehension over `__dataclass_fields__`. Doing so would make the
        stamp -- and `attach_peak_rss`, and anything else that edits a record
        dict -- reach back into the live `Record`.
        """
        shared = {"separable_build": False}
        record = records.Record(
            run_id="r", engine="e", phase=records.PHASE_BUILD, dataset="d",
            timestamp="t", extra=shared)
        as_dict = record.to_dict()
        assert as_dict["extra"] is not shared
        as_dict["extra"]["scribbled"] = True
        assert shared == {"separable_build": False}
        assert record.extra == {"separable_build": False}

    def test_an_empty_extra_still_produces_the_key(self, tmp_path):
        # Record.to_dict() drops `extra` when it is empty, so the stamp has to
        # survive that filter rather than be applied before it. Asserted on the
        # value, not just membership -- a key present with the wrong number is
        # the failure that matters.
        got = self._written(tmp_path, records.Record(
            run_id="r", engine="e", phase=records.PHASE_INGEST, dataset="d",
            timestamp="t"))
        assert got["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
            records.OPS_MEASUREMENT_VERSION

    def test_write_many_stamps_every_record(self, tmp_path):
        path = str(tmp_path / "ops.jsonl")
        with records.Recorder(path) as recorder:
            recorder.write_many([
                records.Record(run_id="r", engine="e", phase=p, dataset="d",
                               timestamp="t")
                for p in (records.PHASE_INGEST, records.PHASE_CHURN)
            ])
        loaded = list(records.read_records(path))
        assert len(loaded) == 2
        assert all(r["extra"][records.OPS_MEASUREMENT_VERSION_KEY]
                   == records.OPS_MEASUREMENT_VERSION for r in loaded)

    def test_a_real_workloads_records_carry_it(self, tmp_path):
        """Through the actual path, including the files this change cannot edit.

        build.py writes two records: an ingest record with no `extra` at all
        and a build record with an explicit `extra`. Both must come out
        stamped, which is the whole claim -- the version reaches records
        written by workload modules that know nothing about it.
        """
        from harness.workloads import build as build_workload
        from harness.workloads.context import RunContext

        from .fake_driver import K, FakeDriver, index_spec, tiny_dataset

        dataset = tiny_dataset()
        recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
        try:
            ctx = RunContext(run_id="t1a", resource_pass="normalized",
                             recorder=recorder, engine="fake", engine_tag="none",
                             march="x86-64-v3", k=K,
                             cache_dir=str(tmp_path / "cache"))
            driver = FakeDriver(dim=dataset.dim, metric=dataset.metric)
            driver.connect()
            build_workload.run(ctx, driver, dataset, index_spec(dataset))
        finally:
            recorder.close()

        written = list(records.read_records(str(tmp_path / "ops-fake.jsonl")))
        assert [r["phase"] for r in written] == [records.PHASE_INGEST,
                                                 records.PHASE_BUILD]
        for r in written:
            assert r["extra"][records.OPS_MEASUREMENT_VERSION_KEY] == \
                records.OPS_MEASUREMENT_VERSION, r["phase"]
        # The build record's own extra is still there, so the stamp did not
        # displace what build.py wrote.
        assert written[1]["extra"]["separable_build"] is False


# ---------------------------------------------------------------------------
# The client container's own CPU (§1.4)
# ---------------------------------------------------------------------------

#: Saturated cores for the live calibration below, and the window over which
#: they are counted.
#:
#: Child PROCESSES rather than threads, and this is the whole reason the
#: calibration can say anything. Python threads share one GIL, so four threads
#: spinning in a `while True: pass` loop consume about ONE core between them --
#: measured here at 0.449 CPU-seconds over a 0.405 s window against 0.491 for
#: a single thread, i.e. indistinguishable. A thread-based "four busy threads"
#: fixture would therefore calibrate against a truth of 1.0 while claiming 4.0,
#: and would pass for a reader off by a factor of four. Four subprocesses
#: really do occupy four cores, which makes the known utilisation 1.0 of a
#: four-CPU cpuset and 0.125 of a thirty-two-CPU one.
CALIBRATION_CORES = 4
CALIBRATION_WINDOW_S = 0.5

#: Attempts allowed before the calibration is called a fault in the reader.
#: Escalation in attempts rather than in window length, and rather than in a
#: looser bound: see the test's docstring. Three, because the interference is a
#: neighbouring process rather than a starved thread of our own -- a sibling
#: worktree's suite run is seconds long, so a second or third try usually falls
#: outside it, while a reader off by a factor of a thousand is off on all three.
CALIBRATION_ATTEMPTS = 3

#: Window over which the test samples how busy the whole box is, to decide
#: whether a missed calibration is the reader's fault or the machine's.
CALIBRATION_NOISE_WINDOW_S = 0.2

#: Foreign load, in cores, above which a missed calibration is attributed to
#: the machine rather than to the reader, and the test skips instead of
#: failing. The binding bound below tolerates about 1.6 cores of someone
#: else's work in the numerator, and under five competing spinners this
#: instrument reads 5.2-5.6, so 1.0 separates the two cases.
#:
#: **"Idle" is doing a lot of work in that sentence, and a developer desktop is
#: not idle.** A quiet moment on this box reads 0.13-0.43 cores, which is where
#: 1.0 came from, but across one working session the same instrument returned
#: anything from 0.15 to 19.99 and crossed 1.0 routinely -- a sibling worktree
#: running its own suite is enough. So the skip below is not a rare event on a
#: shared machine: it is the common case, and over roughly a dozen unmutated
#: attempts in one session it fired more often than it passed. That is why the
#: skip is a DEFAULT and not a verdict. `VB_REQUIRE_CALIBRATION=1` turns it
#: into a failure, so a quiet runner can demand the calibration actually ran;
#: an allowlisted skip with no such switch would legitimise never executing
#: the one test that calibrates the figure this whole task exists to produce.
CALIBRATION_MAX_FOREIGN_CORES = 1.0

#: Environment flag that turns the busy-box skip below into a failure.
#:
#: The skip is honest about a shared desktop and useless as a gate: a test that
#: skips more often than it runs is a test nobody notices has stopped working.
#: Set this on a runner that can be kept quiet -- CI, or a developer about to
#: trust the number -- and a box too busy to calibrate becomes a red test
#: naming the foreign load rather than a green run that measured nothing.
#:
#: It changes only WHICH VERDICT a missed calibration gets. It does not change
#: the measurement, the bounds, the retries, or any other test; and it cannot
#: turn a failing reader green, because every reader fault reaches `pytest.fail`
#: on both settings.
REQUIRE_CALIBRATION_ENV = "VB_REQUIRE_CALIBRATION"


class TestTheClientsOwnCpuCounter:
    """`read_own_cpu_usec` reads THIS container's cgroup, by opening the file.

    The transport is the whole point. Everything else in `resources.py` reaches
    a cgroup by `docker exec`-ing into the server container, because the
    harness runs in a second container and measures the server from outside.
    The harness's own cgroup is not reachable that way: the client image
    carries no Docker CLI, so a reading routed through `_cat` returns None on
    every call, `extra.client_cpu_util` is None on every record and the
    client-bound flag never fires. Nothing crashes and nothing is measured.
    """

    def _skip_without_a_cgroup(self):
        if not (os.path.exists(resources.OWN_CPU_STAT)
                or os.path.exists(resources.OWN_CPUACCT_USAGE)):
            pytest.skip("no cgroup CPU accounting on this kernel")

    def test_it_returns_microseconds_from_the_live_cgroup(self):
        self._skip_without_a_cgroup()
        value = resources.read_own_cpu_usec()
        assert isinstance(value, int)
        # A booted machine has burned CPU-seconds, so the counter is in the
        # millions of microseconds. Zero would mean the file parsed to nothing.
        assert value > 0

    def test_it_does_not_shell_out_to_another_container(self, monkeypatch):
        """Poisoning the docker-exec transport must not affect it."""
        self._skip_without_a_cgroup()

        def refuse(*args, **kwargs):
            raise AssertionError(
                "read_own_cpu_usec reached for another container's cgroup via "
                "docker exec; the client image has no Docker CLI, so that "
                "returns None on every real run")

        monkeypatch.setattr(resources, "_exec", refuse)
        assert resources.read_own_cpu_usec() > 0

    def test_the_cgroup_v1_fallback_converts_nanoseconds(self, tmp_path,
                                                         monkeypatch):
        """`cpuacct.usage` is NANOseconds; the function promises microseconds.

        Unreachable on a cgroup v2 host, which is why the two paths are module
        constants: an unreachable branch that converts units is exactly where a
        factor of a thousand survives review, and a thousand-fold utilisation
        clamps every point of every run to client-bound.
        """
        v1 = tmp_path / "cpuacct.usage"
        v1.write_text("1500000000\n")          # 1.5 seconds, in nanoseconds
        monkeypatch.setattr(resources, "OWN_CPU_STAT", str(tmp_path / "absent"))
        monkeypatch.setattr(resources, "OWN_CPUACCT_USAGE", str(v1))
        assert resources.read_own_cpu_usec() == 1_500_000

    def test_the_v2_counter_wins_where_both_exist(self, tmp_path, monkeypatch):
        v2 = tmp_path / "cpu.stat"
        v2.write_text("nr_periods 0\nusage_usec 777\nuser_usec 500\n")
        v1 = tmp_path / "cpuacct.usage"
        v1.write_text("999000000\n")
        monkeypatch.setattr(resources, "OWN_CPU_STAT", str(v2))
        monkeypatch.setattr(resources, "OWN_CPUACCT_USAGE", str(v1))
        assert resources.read_own_cpu_usec() == 777

    def test_a_malformed_v2_body_falls_through_to_v1(self, tmp_path,
                                                     monkeypatch):
        v2 = tmp_path / "cpu.stat"
        v2.write_text("usage_usec not-a-number\n")
        v1 = tmp_path / "cpuacct.usage"
        v1.write_text("2000000\n")             # 2 ms, in nanoseconds
        monkeypatch.setattr(resources, "OWN_CPU_STAT", str(v2))
        monkeypatch.setattr(resources, "OWN_CPUACCT_USAGE", str(v1))
        assert resources.read_own_cpu_usec() == 2_000

    def test_no_counter_anywhere_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resources, "OWN_CPU_STAT", str(tmp_path / "a"))
        monkeypatch.setattr(resources, "OWN_CPUACCT_USAGE", str(tmp_path / "b"))
        assert resources.read_own_cpu_usec() is None

    def test_the_cpu_count_follows_the_cpuset_not_the_host(self, monkeypatch):
        """`--cpuset-cpus` is how the harness container is started.

        `os.cpu_count()` would report the host's twenty against the client's
        four, which understates utilisation by exactly that ratio -- 20% where
        the truth is 100% -- and is the one error in this calculation that
        still produces a number between 0 and 1.
        """
        monkeypatch.setattr(resources.os, "sched_getaffinity",
                            lambda pid: {12, 13, 14})
        assert resources.own_cpu_count() == 3


class TestClientCpuUtilisationIsCalibrated:
    """The ratio, checked against utilisations known without reference to it.

    `client_cpu_util` is a cgroup counter over a wall clock over a CPU count,
    and each of those three can be wrong while the quotient stays plausibly
    between 0 and 1: the counter in nanoseconds, the wall taken around the
    wrong interval, the CPU count taken from the host. A threshold test over a
    quantity nobody has calibrated would pass against all three.

    So the arithmetic is pinned across the parameter rather than at one point,
    and then the live counter is checked against a known number of saturated
    cores.
    """

    # (CPU-seconds spent, window length, CPUs available, true utilisation).
    # Every expectation is a literal a reader can verify by hand, not the
    # implementation's own expression evaluated again.
    @pytest.mark.parametrize("cpu_seconds,wall,cpus,expected", [
        (10.0, 10.0, 1, 1.00),     # one CPU, saturated
        (10.0, 10.0, 4, 0.25),     # the same work spread over four
        (20.0, 10.0, 4, 0.50),
        (40.0, 10.0, 4, 1.00),     # four CPUs, saturated
        (0.0, 10.0, 4, 0.00),      # idle clients over a real window
        (3.0, 30.0, 4, 0.025),     # a 30 s window, as `main` runs
        (1.0, 0.5, 2, 1.00),       # a short window does not change the ratio
        (96.0, 30.0, 4, 0.80),     # exactly on the limit, which must not trip
    ])
    def test_the_formula_over_known_utilisations(self, cpu_seconds, wall,
                                                 cpus, expected):
        seconds, util = concurrency_workload._client_cpu(
            0, int(round(cpu_seconds * 1e6)), wall, cpus)
        assert seconds == pytest.approx(cpu_seconds)
        assert util == pytest.approx(expected)

    def test_the_limit_is_a_strict_inequality(self):
        """0.80 exactly is not client-bound; a hair above it is.

        Stated here rather than inferred from the workload, because the record
        writes `util > CLIENT_BOUND_UTIL` and a `>=` would flag the one case
        the constant names as the boundary.
        """
        limit = concurrency_workload.CLIENT_BOUND_UTIL
        assert limit == 0.80
        _, exactly = concurrency_workload._client_cpu(
            0, int(0.80 * 4 * 30 * 1e6), 30.0, 4)
        assert exactly == pytest.approx(0.80)
        assert not exactly > limit

    @pytest.mark.parametrize("args", [
        (None, 1_000_000, 10.0, 4),      # no reading at the open
        (0, None, 10.0, 4),              # none at the close
        (0, 1_000_000, 0.0, 4),          # a zero-length window
        (0, 1_000_000, -1.0, 4),         # a negative one
        (0, 1_000_000, 10.0, 0),         # no CPUs
    ])
    def test_an_unmeasurable_window_is_none_not_zero(self, args):
        """Zero is the claim "the clients did nothing", which cannot be true of
        a window that executed queries, and it makes the client-bound flag read
        false on a run where nothing was ever measured."""
        assert concurrency_workload._client_cpu(*args) == (None, None)

    def test_a_backwards_counter_does_not_become_a_negative_ratio(self):
        seconds, util = concurrency_workload._client_cpu(
            5_000_000, 1_000_000, 10.0, 4)
        assert (seconds, util) == (0.0, 0.0)

    @staticmethod
    def _busy_jiffies() -> Optional[int]:
        """Non-idle time across every CPU, from /proc/stat."""
        try:
            with open("/proc/stat") as fh:
                fields = fh.readline().split()
        except OSError:
            return None
        if not fields or fields[0] != "cpu":
            return None
        try:
            values = [int(v) for v in fields[1:]]
        except ValueError:
            return None
        # user nice system idle iowait irq softirq steal guest guest_nice
        return sum(values) - (values[3] + values[4])

    def _machine_cores_busy(self) -> Optional[float]:
        """Cores the WHOLE machine is burning, measured independently.

        Through /proc/stat and the clock tick, deliberately not through
        `read_own_cpu_usec`: this figure decides whether a missed calibration
        is the reader's fault or the box's, and an instrument that shared the
        suspected fault could not answer that. A reader off by a thousand
        would otherwise report a thousandfold baseline, be called "busy" and
        skip itself.
        """
        before = self._busy_jiffies()
        opened = time.perf_counter()
        time.sleep(CALIBRATION_NOISE_WINDOW_S)
        wall = time.perf_counter() - opened
        after = self._busy_jiffies()
        if before is None or after is None:
            return None
        return (after - before) / os.sysconf("SC_CLK_TCK") / wall

    def _four_busy_cores(self):
        """One measurement with `CALIBRATION_CORES` cores pinned busy.

        Returns (CPU-seconds, util against that many CPUs, util against eight
        times as many, wall), or None where there is no counter to read.
        """
        kids = [subprocess.Popen([sys.executable, "-c", "while True: pass"])
                for _ in range(CALIBRATION_CORES)]
        try:
            time.sleep(0.2)            # let every child reach its loop
            start = resources.read_own_cpu_usec()
            opened = time.perf_counter()
            time.sleep(CALIBRATION_WINDOW_S)
            closed = time.perf_counter()
            end = resources.read_own_cpu_usec()
        finally:
            for kid in kids:
                kid.kill()
            for kid in kids:
                kid.wait()
        if start is None or end is None:
            return None
        wall = closed - opened
        seconds, util = concurrency_workload._client_cpu(
            start, end, wall, CALIBRATION_CORES)
        _, wide = concurrency_workload._client_cpu(
            start, end, wall, CALIBRATION_CORES * 8)
        return seconds, util, wide, wall

    @staticmethod
    def _explainable_by_other_work(excess: float,
                                   foreign: Optional[float]) -> bool:
        """Whether a missed calibration is the box's fault rather than the
        reader's, from how far it missed and in which direction.

        "The machine was busy" is only an excuse for the miss it could cause,
        and getting that wrong turns the skip into a way for a broken reader to
        pass: a counter reading nanoseconds as microseconds missed by 5,431
        cores and was skipped on the strength of 1.1 cores of unrelated
        background work, because the first version of this check looked only at
        whether the box was busy at all.

        So three conditions, each ruling out a different non-explanation:

        * The excess must be POSITIVE. Neighbours add to a shared cgroup's
          counter; they cannot subtract. Anything reading LOW -- microseconds
          taken for seconds, a host-wide CPU count -- is the reader.
        * The reading must be physically possible: no cgroup can report more
          busy cores than the machine has CPUs.
        * The excess must be of the ORDER of the observed foreign load. Twice
          it plus half a core, because the sample is taken just before the
          window rather than during it and bursts move; 5,431 against 1.1 is
          not within any such factor.

        **The guarantee is an inequality, not those three lines read as a
        list**, and it is stronger than "of the same order". Write a reader
        that over-reads by a factor F: with `foreign` cores of other work in a
        root cgroup it reports `F * (CALIBRATION_CORES + foreign)`, so

            excess = F * (4 + foreign) - 4   must satisfy   <= 2*foreign + 0.5

        which rearranges to `foreign * (F - 2) <= 4.5 - 4F`. For **F >= 2** the
        left side is non-negative and the right is at most -3.5, so the
        condition is UNSATISFIABLE at any foreign load whatever: an over-read
        inflates the foreign term it would be excused by at least as fast as
        the allowance grows. No amount of neighbour noise can excuse a 2x or
        worse reader, which is why the two unit mutants fail on a busy box as
        well as on a quiet one.

        The residual exposure is therefore a band, F in (1.4, ~1.75) with
        roughly 3-8 cores of sustained foreign work -- above the noise this
        instrument sees on a quiet box and below the structurally inexcusable
        2x. It contains no named defect class: a wrong unit either way is
        1000x, unconverted nanoseconds likewise, `user_usec` taken for
        `usage_usec` reads LOW, a host CPU count in the denominator reads LOW,
        and summing `cpu.stat`'s fields is about 2x. A later reader tempted to
        widen the allowance should redo this arithmetic first: raising the
        coefficient from 2.0 moves the unsatisfiable threshold up with it.
        """
        if excess <= 0 or foreign is None:
            return False
        if CALIBRATION_CORES + excess > len(os.sched_getaffinity(0)):
            return False
        return excess <= 2.0 * foreign + 0.5

    @pytest.mark.skipif(
        len(os.sched_getaffinity(0)) < CALIBRATION_CORES + 2,
        reason="needs spare CPUs: oversubscribed spinners do not each get one")
    def test_a_known_number_of_saturated_cores_reads_back_as_that_many(self):
        """Four busy cores for half a second, measured through the real reader.

        The truth is known without the code under test: four processes in a
        `while True: pass` loop, on a box with at least two CPUs to spare,
        occupy four cores for the duration. So the CPU-seconds must come back
        at four times the wall, the utilisation against a four-CPU cpuset must
        come back at 1.0, and the same reading against a thirty-two-CPU cpuset
        must come back eight times smaller.

        **Retried rather than loosened**, which is the same decision
        `COVERAGE_WINDOWS_S` records one class up and for the mirror-image
        reason. On a host, `/sys/fs/cgroup` is the ROOT cgroup, so the counter
        also sees the rest of the machine: a tight window is a claim about what
        the whole box did, and a neighbour -- in this repo, a sibling worktree
        running its own suite -- lands in the numerator. This assertion was
        written with a flat 1.4 ceiling and failed at 1.65 on the third
        consecutive full-suite run, with 2.6 cores of someone else's work in
        it. Widening the ceiling instead would have been the wrong repair:
        every fault worth catching here is a factor of a thousand or a factor
        of the CPU count, so a loose ceiling keeps the discrimination and
        throws away the statement that the figure is 1.0. Retrying keeps both:
        a correct reader lands in range on an idle attempt and a broken one is
        out by 1,000x or 5x on every attempt.

        The arithmetic that cannot be noisy is checked on EVERY attempt, before
        any of it is retried -- the eight-times-the-CPUs ratio is exact, and a
        dropped CPU divisor must not be retried away.
        """
        attempts, foreign = [], []
        for _ in range(CALIBRATION_ATTEMPTS):
            # Sampled before the spinners start, so it is somebody else's
            # work and not ours.
            foreign.append(self._machine_cores_busy())
            measured = self._four_busy_cores()
            if measured is None:
                pytest.skip("no cgroup CPU accounting on this kernel")
            seconds, util, wide, wall = measured
            # Exact, and therefore never retried and never skipped: the same
            # reading against eight times the CPUs is an eighth of the
            # utilisation. A formula that dropped the CPU count reports the
            # two identically and calls a comfortably idle container bound.
            assert wide == pytest.approx(util / 8)
            attempts.append(measured)
            # Four cores for `wall` seconds is 4 x wall CPU-seconds, and four
            # saturated cores on a four-CPU cpuset is a utilisation of 1.0.
            if (CALIBRATION_CORES * wall * 0.9 <= seconds
                    <= CALIBRATION_CORES * wall + 2.0
                    and 0.9 <= util <= 1.4):
                break
        else:
            readings = "; ".join(
                f"{s:.3f} CPU-s over {w:.3f}s = {s / w:.2f} cores "
                f"(util {u:.3f})" for s, u, _, w in attempts)
            quietest = min((f for f in foreign if f is not None), default=None)
            excess = min(s / w - CALIBRATION_CORES for s, _, _, w in attempts)
            if self._explainable_by_other_work(excess, quietest):
                busy = (
                    f"cannot calibrate on a box this busy: /proc/stat reports "
                    f"{quietest:.2f} cores of other work at the quietest of "
                    f"{CALIBRATION_ATTEMPTS} attempts, the root cgroup counts "
                    f"it in the numerator, and the smallest excess over "
                    f"{CALIBRATION_CORES} cores was {excess:.2f}. "
                    f"Readings: {readings}")
                # A skip is the honest verdict on a shared desktop and a
                # useless gate: this test skipped more often than it passed
                # across a working session, and a check that usually does not
                # run is one nobody notices has stopped working. The flag lets
                # a runner that can be kept quiet demand that it ran, without
                # turning every sibling worktree's suite into a red build
                # here. It moves the verdict only -- nothing above it changes.
                if os.environ.get(REQUIRE_CALIBRATION_ENV):
                    pytest.fail(
                        f"{REQUIRE_CALIBRATION_ENV} is set, so a box too busy "
                        f"to calibrate is a failure rather than a skip: "
                        f"{busy}. Quiet the machine and re-run, or unset "
                        f"{REQUIRE_CALIBRATION_ENV} to let a busy box skip.")
                pytest.skip(busy)
            pytest.fail(
                f"{CALIBRATION_ATTEMPTS} attempts at {CALIBRATION_CORES} "
                f"saturated cores never read back as {CALIBRATION_CORES}: "
                f"{readings}. Other work on the box was "
                f"{'unknown' if quietest is None else f'{quietest:.2f} cores'} "
                f"by /proc/stat — which does not go through the reader under "
                f"test — and cannot account for an excess of {excess:.2f} "
                f"cores, so the fault is in the reader: check its unit and "
                f"which CPU count it divides by.")

        seconds, util, wide, wall = attempts[-1]
        # The threshold, over the figure just calibrated. Not implied by the
        # range above: these are what a moved CLIENT_BOUND_UTIL breaks.
        assert util > concurrency_workload.CLIENT_BOUND_UTIL
        assert not wide > concurrency_workload.CLIENT_BOUND_UTIL
