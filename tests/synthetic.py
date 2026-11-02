"""Generate a synthetic run directory so the report pipeline can be exercised
without spending hours on a real benchmark.

Used by tests/test_report.py and by `tests/make-demo-report.sh`, which produces a
sample report for reviewing layout and wording before any real data exists.

The numbers are fabricated and shaped to exercise every branch of the report:
one engine that wins on throughput, one that wins on recall, one plan failure,
one short-result filtered case, and a churn series that degrades.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List

ENGINES = ("mariadb", "alisql", "pgvector")
DATASET = "fashion-mnist-784-euclidean"

#: The ops unit every single-client ops record here was measured in. Spelled
#: once: `_run_unit` loops M x build mode x storage engine, and a fixture
#: whose records disagreed about which unit they came from would exercise the
#: report's pooling guards against a confound it invented rather than against
#: the one they are for.
OPS_M = 16
OPS_EF_SEARCH = 100
#: pgvector is the only engine here with two build modes, and the ops records
#: below all describe its `post` build. `build_mode` is on EVERY ops record,
#: including the MySQL family's, because `context.record_defaults` puts it
#: there -- a fixture that left it absent would make `None` look like a legal
#: value of the §5.2 match key.
OPS_BUILD_MODE = "post"

#: The two filter columns PR 3 loads in one build. Both, because
#: `generate._filter_columns_present` draws one chart per column present and a
#: fixture with one column never exercises the second chart -- and because
#: `FILTER_COLUMN_WHEN_ABSENT` makes an absent column read as `tag`, so a
#: fixture that omitted the key would silently test the archived-record path
#: instead of the one the harness writes today.
FILTER_COLUMNS = ("tag", "ctag")

#: PR 5 §5.4's second test: "synthetic run directory (extend
#: tests/synthetic.py) with one consistent and one inconsistent pair; the
#: report lists both correctly."
#:
#: Each entry offsets the ops calibration recall from the ann recall of the
#: record it matches, so the pair's consistency is a property of this
#: declaration rather than of the formula two sections above. `ci_half` is
#: the half-width of the ops record's `recall_ci95`, which sets the tolerance
#: `max(0.01, 2 x half)`:
#:
#: * mariadb: half 0.0035, so 2 x half = 0.007 is BELOW the 0.01 floor and the
#:   tolerance is the floor. Offset 0.004 < 0.01, so the pair passes and its
#:   tolerance basis renders as "floor".
#: * pgvector: half 0.02, so the tolerance is 0.04 -- four times the floor, so
#:   the row renders basis "ci" and a reader can see the CI branch working.
#:   Offset -0.12 is three times that, so the pair fails on a tolerance that
#:   was computed from the interval and not from the floor.
#:
#: Both branches of the `max`, and both marks, from two rows. A fixture where
#: every row took the same branch would leave half of §5.2 item 2 undrawn.
#: Engines whose ANN module declares a build mode, mirroring the overlay:
#: `algorithms/pgvector/module.py` and `algorithms/valkey/module.py` put
#: `build_mode` in `get_additional()` and the MariaDB, MariaDB 12.3 and AliSQL
#: modules write no such attribute at all. That asymmetry is load-bearing for
#: PR 5 §5.2 -- it is what `generate.build_mode_axis_engines` reads -- so a
#: fixture that gave every engine a build mode, or none of them one, would
#: exercise only one branch of the matcher.
ANN_BUILD_MODE = {"pgvector": "post"}

#: What one client over TCP costs each engine, as a share of its in-process
#: throughput. Not uniform, and that is the point: §5.2 item 5 asks for
#: engine-order inversions between the two paths to be flagged, and an
#: identical factor across engines preserves every ordering, so the branch
#: would never be reached. pgvector is the fastest engine in-process here
#: (speed 12,000 against MariaDB's 9,000) and the most expensive per round
#: trip, so the two paths disagree about which of the pair is faster -- which
#: is exactly the real effect the item describes as "a legitimate
#: difference".
OPS_TCP_SHARE = {"mariadb": 0.80, "alisql": 0.75, "pgvector": 0.50}

CROSSCHECK_PAIRS = (
    {"engine": "mariadb", "m": OPS_M, "ef_search": 160,
     "recall_offset": 0.004, "ci_half": 0.0035, "consistent": True},
    {"engine": "pgvector", "m": OPS_M, "ef_search": 160,
     "recall_offset": -0.12, "ci_half": 0.02, "consistent": False},
)


def _timestamp(i: int) -> str:
    return f"2026-08-03T12:{i % 60:02d}:00Z"


def _ci(value: float, half: float, n: int = 500) -> Dict[str, float]:
    """A `recall_ci95`-shaped interval centred on `value`.

    The shape `harness.metrics.stats.Interval.as_field` writes, including `n`
    -- which is the field that stops a one-sample interval being published as
    a tie, so a fixture that omitted it would make every tie column in the
    report read "no interval" and nothing would notice.

    Clamped at 1.0 on the high side: recall is a bounded fraction, and an
    interval reaching to 1.004 would be a fixture asserting something the
    measurement cannot produce.
    """
    return {"low": round(max(0.0, value - half), 6),
            "high": round(min(1.0, value + half), 6), "n": n}


def _qps_ci(qps: float, fraction: float = 0.03,
            n: int = 500) -> Dict[str, float]:
    """A `qps_ci95`-shaped interval, `fraction` wide either side.

    Unbounded above, unlike recall: throughput has no ceiling of 1.
    """
    return {"low": round(qps * (1 - fraction), 4),
            "high": round(qps * (1 + fraction), 4), "n": n}


def make_manifest(run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": "2026-08-03T12:00:00Z",
        "finished_at": "2026-08-03T14:30:00Z",
        "status": "completed",
        "framework": {"name": "vector-bench", "commit": "synthetic"},
        "host": {
            "cpu": {
                "model": "Synthetic CPU @ 3.5GHz",
                "arch": "x86_64",
                "logical_cpus": 20,
                "physical_cores": 14,
                "sockets": 1,
                "numa_nodes": 1,
                "threads_per_core": 2,
                "hybrid": True,
                "performance_cpus": list(range(12)),
                "efficiency_cpus": list(range(12, 20)),
                "simd_flags": ["avx", "avx2", "fma", "f16c"],
                "has_avx512": False,
            },
            "total_ram_bytes": 64 * 1024 ** 3,
            "available_ram_bytes": 40 * 1024 ** 3,
            "kernel": "6.8.0-synthetic",
            "docker_version": "29.1.3",
            "cgroup_version": "v2",
            "hostname": "synthetic",
            "python_version": "3.12.3",
        },
        "engines": {
            "mariadb": {
                "source": {"tag": "mariadb-11.8.8", "commit": "a" * 40},
                "build": {"march": "x86-64-v3", "build_type": "RelWithDebInfo"},
                "images": {"runtime": {"ref": "vector-bench/mariadb-runtime", "id": "sha256:1"},
                           "bench": {"ref": "vector-bench/mariadb-bench", "id": "sha256:2"}},
            },
            "alisql": {
                "source": {"tag": "AliSQL-8.0.44-2", "commit": "b" * 40},
                "build": {"march": "x86-64-v3", "build_type": "RelWithDebInfo"},
                "images": {"runtime": {"ref": "vector-bench/alisql-runtime", "id": "sha256:3"},
                           "bench": {"ref": "vector-bench/alisql-bench", "id": "sha256:4"}},
            },
            "pgvector": {
                "source": {"tag": "v0.8.6", "commit": "c" * 40},
                "build": {"march": "x86-64-v3", "build_type": "Release"},
                "images": {"runtime": {"ref": "vector-bench/pgvector-runtime", "id": "sha256:5"},
                           "bench": {"ref": "vector-bench/pgvector-bench", "id": "sha256:6"}},
            },
        },
        "config": {
            "profile": {"name": "synthetic",
                        "description": "fabricated data for pipeline testing"},
            "resource_pass": "normalized",
            "resolved_resources": {
                "name": "normalized",
                "server_cpuset": "0,2,4,6,8,10",
                "client_cpuset": "12,13",
                "server_cpu_count": 6,
                "server_memory_bytes": 16 * 1024 ** 3,
                "buffer_bytes": int(16 * 1024 ** 3 * 0.35),
                "graph_cache_bytes": int(16 * 1024 ** 3 * 0.35),
                "maintenance_bytes": int(16 * 1024 ** 3 * 0.15),
                "build_threads": 1,
                "shm_size": "2g",
                "hybrid_cpu": True,
                "core_class_used": "performance",
                "warnings": [],
            },
        },
        "phases": [],
        "warnings": [
            "This CPU has no AVX-512. MariaDB MHNSW and AliSQL VIDX both document "
            "AVX-512 distance kernels, so both are running narrower SIMD paths here.",
            "Hybrid CPU detected (12 performance / 8 efficiency logical CPUs). Runs "
            "are pinned to one core class.",
        ],
    }


def make_records(run_id: str) -> List[Dict[str, Any]]:
    rng = random.Random(20260803)
    records: List[Dict[str, Any]] = []

    # Distinct shapes so the charts are visibly different per engine.
    shape = {
        "mariadb": {"speed": 9000.0, "ceiling": 0.995},
        "alisql": {"speed": 7200.0, "ceiling": 0.998},
        "pgvector": {"speed": 12000.0, "ceiling": 0.988},
    }

    base = {
        "run_id": run_id, "dataset": DATASET, "metric_space": "euclidean",
        "resource_pass": "normalized", "march": "x86-64-v3", "k": 10,
    }

    # --- recall / QPS ------------------------------------------------
    n = 0
    for engine in ENGINES:
        for m in (8, 16, 32):
            for ef in (10, 20, 40, 80, 160, 320):
                n += 1
                cfg = shape[engine]
                recall = cfg["ceiling"] * (1 - math.exp(-ef / (14.0 + 40.0 / m)))
                recall = min(cfg["ceiling"], recall * (0.94 + 0.02 * math.log2(m)))
                qps = cfg["speed"] / (1 + ef / 22.0) / (1 + m / 48.0)
                records.append({
                    **base, "phase": "recall_qps", "engine": engine,
                    "engine_version": f"{engine}-synthetic",
                    "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                    # Present for pgvector, absent for the MySQL family, as
                    # the overlay modules are. See ANN_BUILD_MODE.
                    **({"build_mode": ANN_BUILD_MODE[engine]}
                       if engine in ANN_BUILD_MODE else {}),
                    "m": m, "ef_search": ef,
                    "recall_at_k": round(min(1.0, recall + rng.uniform(-0.004, 0.004)), 6),
                    "qps": round(qps * rng.uniform(0.97, 1.03), 2),
                    "clients": 1,
                    "latency_p50_ms": round(1000.0 / qps, 4),
                    "latency_p95_ms": round(1600.0 / qps, 4),
                    "latency_p99_ms": round(2400.0 / qps, 4),
                    "index_bytes": int(60e6 + m * 3.1e6),
                    "vector_index_used": True,
                    "timestamp": _timestamp(n),
                })

    # One plan failure, so the Validity section has something to report.
    records.append({
        **base, "phase": "recall_qps", "engine": "alisql",
        "engine_version": "alisql-synthetic", "storage_engine": "InnoDB",
        "m": 8, "ef_search": 800, "recall_at_k": 1.0, "qps": 41.3, "clients": 1,
        "vector_index_used": False, "timestamp": _timestamp(99),
        "notes": "optimizer chose a full scan at this LIMIT",
    })

    # --- build cost ---------------------------------------------------
    for engine in ENGINES:
        incremental = engine != "pgvector"
        for m in (8, 16, 32):
            ingest = 5200.0 if not incremental else 2300.0 - m * 18
            wall = 60000 / ingest
            records.append({
                **base, "phase": "ingest", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": m,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "rows": 60000, "ingest_wall_s": round(wall, 2),
                "ingest_rows_per_s": round(ingest, 1), "ingest_threads": 1,
                "timestamp": _timestamp(m),
            })
            build_wall = wall if incremental else wall + m * 1.6
            records.append({
                **base, "phase": "index_build", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": m,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "build_mode": "incremental" if incremental else "post",
                "rows": 60000,
                "build_wall_s": round(build_wall, 2),
                "ingest_wall_s": round(wall, 2), "ingest_threads": 1,
                "peak_rss_bytes": int(1.1e9 + m * 4.2e7),
                "index_bytes": int(60e6 + m * 3.1e6),
                "table_bytes": int(1.9e8),
                "extra": {"separable_build": not incremental},
                "timestamp": _timestamp(m),
            })

    # --- concurrency ---------------------------------------------------
    #
    # Every field the concurrency surfaces read, because a synthetic record
    # that omits one renders an em dash and a reviewer reads that as a report
    # bug rather than as a fixture gap. The three fields the report grew after
    # this fixture was written -- recall, the query pool and the client's own
    # CPU -- were all missing, so the synthetic report showed no recall column
    # at all and nothing noticed.
    #
    # Recall at a fixed ef_search is a property of the index, not of the
    # window, so it does NOT vary with the client count here: a fixture whose
    # recall moved across a sweep would make the workload's own recall-spread
    # check look like it was firing on nothing.
    #
    # 0.98 against 0.80 at one ef_search is the spread these implementations
    # really show, and it is the reason the column exists.
    recall_at_ef = {"mariadb": 0.9812, "alisql": 0.9975, "pgvector": 0.8043}
    #: Share of the client container's CPU each client costs. Chosen so that
    #: exactly one engine crosses the 0.80 limit and only at its top two
    #: client counts: pgvector at 16 and 32. That renders both branches of
    #: every surface the flag reaches — the hollow marker against the plain
    #: one, the marked cell against the bare one, and the "client-bound points
    #: in this run" note against its absence — which a fixture where
    #: everything or nothing was bound could not.
    cpu_per_client = {"mariadb": 0.012, "alisql": 0.014, "pgvector": 0.055}
    CLIENT_CPUS = 4
    DURATION_S = 20
    QUERY_POOL = 10_000
    for engine in ENGINES:
        # Different saturation points, which is the whole point of the chart.
        knee = {"mariadb": 8, "alisql": 16, "pgvector": 4}[engine]
        base_qps = shape[engine]["speed"] / 6.0
        for clients in (1, 2, 4, 8, 16, 32):
            scale = clients / (1 + (clients / knee) ** 1.6)
            qps = base_qps * scale
            executed = int(qps * DURATION_S)
            # A window inside its first pass over the pool covers only what it
            # executed; reuse is then below 1.0, which is legitimate and which
            # the report has to be able to print.
            distinct = min(QUERY_POOL, executed)
            util = min(0.99, cpu_per_client[engine] * clients)
            records.append({
                **base, "phase": "concurrency", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": 16, "ef_search": 100,
                "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
                "clients": clients, "qps": round(qps, 2),
                "recall_at_k": recall_at_ef[engine],
                "latency_mean_ms": round(clients * 1000.0 / qps, 4),
                "latency_p50_ms": round(clients * 900.0 / qps, 4),
                "latency_p95_ms": round(clients * 1700.0 / qps, 4),
                "latency_p99_ms": round(clients * 2900.0 / qps, 4),
                "queries_executed": executed,
                "extra": {
                    "scaling_efficiency": round(scale / clients, 4),
                    "duration_s": DURATION_S,
                    "warmup_queries_per_client": 50,
                    # Three windows per point, so the Spread column renders a
                    # real error bar rather than the "measured once" dash.
                    "repeats": 3,
                    "qps_samples": [round(qps * f, 1)
                                    for f in (0.94, 1.0, 1.05)],
                    "qps_spread_pct": 11.0,
                    "recall_queries": distinct,
                    "recall_spread": 0.0004,
                    "query_pool": QUERY_POOL,
                    "query_reuse": round(executed / distinct, 2),
                    # One in-flight query per client, which is the most the
                    # trimming can ever drop; zero before the window, which is
                    # the only honest value for a correct implementation.
                    "inflight_discarded": clients,
                    "prewindow_discarded": 0,
                    "client_cpu_seconds": round(
                        util * DURATION_S * CLIENT_CPUS, 3),
                    "client_cpus": CLIENT_CPUS,
                    "client_cpu_util": round(util, 4),
                    "client_bound": util > 0.80,
                },
                "timestamp": _timestamp(clients),
            })

    # --- filtered -------------------------------------------------------
    #
    # Both filter columns, which is what the harness has written since PR 3.
    # `ctag` follows the vector space, so its recall collapses where `tag`'s
    # does not -- that difference is the whole reason the second column
    # exists, and a fixture that gave both columns the same curve would make
    # the two charts look like a duplicated figure.
    #
    # The counted selectivity differs from the nominal share, and differs
    # BETWEEN the columns for the same request: `ctag` passes whole clusters,
    # so a nominal 1% delivers about 1.9%. Modelled here rather than set equal
    # to the request, because `_filtered_table` prints both and the report's
    # own prose says the two are not the same number.
    for engine in ENGINES:
        for column in FILTER_COLUMNS:
            for nominal in (0.01, 0.10, 0.50):
                counted = nominal if column == "tag" else min(
                    0.95, nominal * 1.9)
                degradation = {"mariadb": 0.55, "alisql": 0.40,
                               "pgvector": 0.80}[engine]
                recall = min(0.99, 0.35 + degradation * nominal
                             + 0.45 * nominal)
                if column == "ctag":
                    # The collapse the column was added to measure.
                    recall = round(recall * 0.55, 6)
                short = (int(180 * (1 - nominal))
                         if engine == "alisql" else 0)
                qps = round(shape[engine]["speed"] / 8 * (0.4 + nominal), 2)
                records.append({
                    **base, "phase": "filtered", "engine": engine,
                    "engine_version": f"{engine}-synthetic", "m": OPS_M,
                    "ef_search": OPS_EF_SEARCH,
                    "build_mode": OPS_BUILD_MODE,
                    "storage_engine": ("heap" if engine == "pgvector"
                                       else "InnoDB"),
                    "selectivity": counted,
                    "recall_at_k": round(recall, 6),
                    "qps": qps,
                    "clients": 1,
                    "queries_executed": 500,
                    "latency_p50_ms": 0.8, "latency_p95_ms": 1.6,
                    "latency_p99_ms": 2.9,
                    "vector_index_used": True,
                    "extra": {
                        # The column-neutral name the harness always writes,
                        # and `tag_threshold` only for the column it names --
                        # a `ctag` record carrying a key named for `tag` is
                        # the defect `_record_point`'s docstring describes.
                        "filter_column": column,
                        "filter_threshold": int(nominal * 100),
                        **({"tag_threshold": int(nominal * 100)}
                           if column == "tag" else {}),
                        "nominal_selectivity": nominal,
                        "short_result_queries": short,
                        "returned_fewer_than_k": short > 0,
                        "queries_planned": 500,
                        "time_bounded": False,
                        "insufficient_sample": False,
                        # PR 5 §5.1 item 4. Widths differ per engine so the
                        # tie column has something to say: at the nominal 10%
                        # point mariadb and alisql overlap and pgvector does
                        # not -- which is what makes the "tied with" and
                        # "distinct" cells both appear in one table.
                        "recall_ci95": _ci(
                            round(recall, 6),
                            0.05 if engine != "pgvector" else 0.004),
                        "qps_ci95": _qps_ci(qps),
                        "iterative_scan": ("relaxed_order"
                                           if engine == "pgvector" else None),
                    },
                    "timestamp": _timestamp(int(nominal * 100)),
                })

    # --- churn -----------------------------------------------------------
    #
    # Three kinds of churn record, because the harness now writes three and
    # the report has a surface for each:
    #
    # * the x=0 baseline, which names NO maintenance state -- claiming one
    #   would invite a report to plot a pre-churn point on a maintenance
    #   series;
    # * one `maintenance_state: "before"` record per fraction, the index as
    #   the churn left it;
    # * for the engine with a repair path, a `"after"` record per fraction.
    #   pgvector is that engine here (VACUUM), so the Repair column and the
    #   churn chart's second series both have input. MariaDB and AliSQL have
    #   none, and their rows reading "—" is what the report means by a record
    #   that measures one state without saying which.
    #
    # Plus PR 4 §4.3's during-writes records, which are a DIFFERENT kind of
    # churn record: aggregate reader throughput over several clients against a
    # corpus that is mid-change, with no recall at all. They carry
    # `during_writes` so `churn_state.is_during_writes` keeps them off the
    # single-client table and the retention chart.
    REPAIR_PATHS = {"pgvector": "VACUUM", "mariadb": None, "alisql": None}
    READERS = 4
    for engine in ENGINES:
        decay = {"mariadb": 0.030, "alisql": 0.018, "pgvector": 0.055}[engine]
        baseline_recall = 0.972
        repair = REPAIR_PATHS[engine]
        common = {
            **base, "phase": "churn", "engine": engine,
            "engine_version": f"{engine}-synthetic", "m": OPS_M,
            "ef_search": OPS_EF_SEARCH, "build_mode": OPS_BUILD_MODE,
            "storage_engine": "heap" if engine == "pgvector" else "InnoDB",
            "rows": 60000,
        }
        # The baseline. Its `extra` carries the intervals and nothing else:
        # §5.1 item 4 puts them on every churn record, and the baseline is the
        # point every fraction is measured against, so a chart with bars on
        # the churned points and none here would show the comparison as exact
        # at the one place it is not.
        base_qps = round(shape[engine]["speed"] / 7, 2)
        records.append({
            **common, "churn_fraction": 0.0, "clients": 1,
            "recall_at_k": round(baseline_recall, 6), "qps": base_qps,
            "index_bytes": int(1.1e8),
            "notes": "baseline before any churn",
            "extra": {"recall_ci95": _ci(baseline_recall, 0.006),
                      "qps_ci95": _qps_ci(base_qps)},
            "timestamp": _timestamp(0),
        })
        for fraction in (0.10, 0.25):
            degraded = baseline_recall - decay * (fraction / 0.10)
            qps = round(shape[engine]["speed"] / 7 * (1 - fraction * 0.3), 2)
            # Widths chosen so that at 10% churn mariadb and alisql overlap
            # and pgvector does not: the tie column and the chart's whiskers
            # both need a figure where some engines are indistinguishable and
            # some are not, or neither branch is ever drawn.
            half = 0.02 if engine != "pgvector" else 0.004
            records.append({
                **common, "churn_fraction": fraction, "clients": 1,
                "recall_at_k": round(degraded, 6), "qps": qps,
                "index_bytes": int(1.1e8 * (1 + fraction * 0.2)),
                "extra": {
                    "rows_churned": int(60000 * fraction),
                    "delete_seconds": round(fraction * 12.0, 3),
                    "insert_seconds": round(fraction * 48.0, 3),
                    "reinsert_completed": True,
                    "reinsert_rows_per_s": 1250.0,
                    "recall_drop_vs_baseline": round(
                        baseline_recall - degraded, 6),
                    # §4.2 items 4 and 5 and §4.1 item 4, on every record
                    # whatever the engine. `index_sync_s` is 0.0 for an engine
                    # with no asynchronous index to wait for, which is a
                    # measurement and not a missing value.
                    "maintenance_state": "before",
                    "repair_path": repair or "none",
                    "index_sync_s": 0.0,
                    "index_sync_note": ("no asynchronous index to wait for"
                                        if repair is None else
                                        "index reported in sync immediately"),
                    "recall_ci95": _ci(round(degraded, 6), half),
                    "qps_ci95": _qps_ci(qps),
                },
                "timestamp": _timestamp(int(fraction * 100)),
            })
            if repair:
                # The repaired index: better recall and a smaller index than
                # the degraded reading beside it. Two rows per fraction whose
                # engine, dataset, storage and churn columns are identical is
                # exactly what the Repair column exists to tell apart.
                repaired = min(baseline_recall, degraded + decay * 0.6)
                repaired_qps = round(qps * 1.04, 2)
                records.append({
                    **common, "churn_fraction": fraction, "clients": 1,
                    "recall_at_k": round(repaired, 6), "qps": repaired_qps,
                    "index_bytes": int(1.1e8 * (1 + fraction * 0.08)),
                    "extra": {
                        "rows_churned": int(60000 * fraction),
                        "delete_seconds": round(fraction * 12.0, 3),
                        "insert_seconds": round(fraction * 48.0, 3),
                        "reinsert_completed": True,
                        "reinsert_rows_per_s": 1250.0,
                        "maintenance_state": "after",
                        "maintenance": repair,
                        "maintenance_s": round(fraction * 30.0, 3),
                        "repair_path": repair,
                        "index_sync_s": 0.0,
                        "index_sync_note": (
                            "index reported in sync immediately"),
                        "recall_drop_vs_baseline": round(
                            baseline_recall - repaired, 6),
                        "recall_ci95": _ci(round(repaired, 6), half),
                        "qps_ci95": _qps_ci(repaired_qps),
                    },
                    "timestamp": _timestamp(int(fraction * 100) + 1),
                })

            # PR 4 §4.3's reader pool. `latency_p99_ms` is the window under
            # writes and `extra.reader_baseline_p99_ms` is the quiet one, so
            # `during_writes_rows` recomputes the headline ratio from this one
            # record's own two numbers -- which is the point of writing both
            # into one record. `reader_p99_ratio` is stamped to match, because
            # the report cross-checks the two and says so when they disagree.
            baseline_p99 = 2.4
            under_p99 = round(baseline_p99 * (1.8 + fraction), 4)
            reader_qps = round(base_qps * READERS * 0.55, 2)
            reader_baseline_qps = round(base_qps * READERS * 0.92, 2)
            records.append({
                **common, "churn_fraction": fraction, "clients": READERS,
                "qps": reader_qps,
                "index_bytes": int(1.1e8 * (1 + fraction * 0.2)),
                "latency_p50_ms": round(under_p99 * 0.3, 4),
                "latency_p95_ms": round(under_p99 * 0.8, 4),
                "latency_p99_ms": under_p99,
                "queries_executed": 4000,
                "notes": "reader pool measured during the re-insert",
                "extra": {
                    "during_writes": True,
                    "readers": READERS,
                    "reader_queries": 10000,
                    "reader_warmup_queries": 20,
                    "reader_pool": {"clients": READERS, "queries": 10000,
                                    "k": 10,
                                    "ef_search": OPS_EF_SEARCH},
                    "reader_ops_unit": {
                        "m": OPS_M, "build_mode": OPS_BUILD_MODE,
                        "storage_engine": common["storage_engine"],
                        "resource_pass": "normalized"},
                    "reader_p99_ratio": under_p99 / baseline_p99,
                    "reader_qps_ratio": reader_qps / reader_baseline_qps,
                    "writer_rows_per_s_with_readers": 980.0,
                    "writer_rows": int(60000 * fraction),
                    "reader_baseline_p99_ms": baseline_p99,
                    "reader_baseline_qps": reader_baseline_qps,
                    "reader_baseline_queries_executed": 5200,
                    "reader_baseline_window_s": 30.0,
                    "reader_under_writes_p99_ms": under_p99,
                    "reader_under_writes_qps": reader_qps,
                    "reader_under_writes_queries_executed": 4000,
                    "reader_under_writes_window_s": 30.0,
                    "client_cpu_seconds": 18.0,
                    "client_cpus": 4,
                    "client_cpu_util": 0.15,
                    "client_bound": False,
                },
                "timestamp": _timestamp(int(fraction * 100) + 2),
            })

    # --- ef calibration, and PR 5 §5.2's cross-check pair ----------------
    #
    # The fixture had no calibration records at all, so §6's operating-point
    # table, Appendix A's curves and §10's consistency checks all rendered
    # empty against it -- three surfaces that a reviewer reads as broken.
    #
    # The recalls here are taken FROM the recall/QPS records above and offset
    # by a declared amount, so each pair's consistency is a property of
    # `CROSSCHECK_PAIRS` rather than of the two formulas agreeing by luck.
    ann_recall = {
        (r["engine"], r["m"], r["ef_search"]): r["recall_at_k"]
        for r in records if r["phase"] == "recall_qps"
    }
    offsets = {(pair["engine"], pair["m"], pair["ef_search"]): pair
               for pair in CROSSCHECK_PAIRS}
    GRID = (40, 160)
    for engine in ENGINES:
        for position, ef in enumerate(GRID):
            pair = offsets.get((engine, OPS_M, ef))
            reference = ann_recall.get((engine, OPS_M, ef))
            if reference is None:
                continue
            if pair is not None:
                recall = reference + pair["recall_offset"]
                half = pair["ci_half"]
            else:
                # Not a declared pair, so it still matches on the §5.2 key and
                # still has to render -- a fixture whose only matches were the
                # two interesting ones would leave the "agrees" branch tested
                # by exactly one row.
                recall = reference - 0.002
                half = 0.005
            recall = round(min(1.0, max(0.0, recall)), 6)
            qps = round(shape[engine]["speed"] / (1 + ef / 22.0)
                        * OPS_TCP_SHARE[engine], 2)
            records.append({
                **base, "phase": "ef_calibration", "engine": engine,
                "engine_version": f"{engine}-synthetic", "m": OPS_M,
                "ef_search": ef, "build_mode": OPS_BUILD_MODE,
                # `InnoDB` for EVERY engine, pgvector included, because that
                # is what the harness writes: `harness/main.py` defaults
                # `--storage-engine` to InnoDB and `record_defaults` stamps
                # `index.storage_engine` on every record, whatever the engine
                # actually stores in. The ann side reports `heap` for the same
                # index, and that disagreement is exactly why §5.2 item 1
                # omits storage from the match key -- so a fixture that
                # quietly wrote `heap` on both sides would make the matcher
                # look correct for a reason the real records do not supply.
                #
                # The filtered, churn and build records above still say `heap`
                # for pgvector. That is a FIXTURE INACCURACY of the same kind,
                # left alone deliberately: `render._storage` overrides the
                # cell to `heap` for display anyway, so correcting them moves
                # no published value, while `ops_unit` keys off the raw field
                # and changing it would regroup every filtered and churn
                # series in the demo report. That is a separate change from
                # this one and is noted here rather than half-made.
                "storage_engine": "InnoDB",
                "recall_at_k": recall, "qps": qps, "clients": 1,
                "vector_index_used": True,
                "queries_executed": 1000,
                "latency_p50_ms": round(1000.0 / qps, 4),
                "latency_p95_ms": round(1600.0 / qps, 4),
                "latency_p99_ms": round(2400.0 / qps, 4),
                "extra": {
                    "effective_ef": ef,
                    "target_recall": 0.95,
                    "reached_target": recall >= 0.95,
                    "grid_position": position,
                    "grid_size": len(GRID),
                    "warmup_queries": 20,
                    # What an ops version-9 calibration record says; without
                    # it the record is read as a pre-v9 ID overlap and the
                    # cross-check declines to compare it.
                    "recall_kind": "distance_threshold",
                    "recall_ci95": _ci(recall, half, n=1000),
                    "qps_ci95": _qps_ci(qps, n=1000),
                },
                "timestamp": _timestamp(50 + position),
            })

    return records


def write_run(run_dir: str, run_id: str = "synthetic-run") -> str:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run-manifest.json"), "w") as fh:
        json.dump(make_manifest(run_id), fh, indent=2, sort_keys=True)
    path = os.path.join(run_dir, f"ops-synthetic-{DATASET}-normalized-m16-post.jsonl")
    with open(path, "w") as fh:
        for record in make_records(run_id):
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    # A short memory timeseries so the timeline chart has input.
    mem_path = os.path.join(run_dir, "mem-mariadb-{}-normalized-m16-post.jsonl".format(DATASET))
    with open(mem_path, "w") as fh:
        for i in range(120):
            fh.write(json.dumps({
                "t": 1000.0 + i * 0.25,
                "container": "synthetic-srv",
                "rss_bytes": int(4e8 + 6e8 * (1 - math.exp(-i / 25.0))),
                "peak_bytes": int(1.1e9),
                "cpu_seconds": i * 0.2,
            }) + "\n")
    return run_dir
