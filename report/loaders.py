"""Load both measurement paths into the one flat record schema.

ann-benchmarks writes HDF5 result files; the ops harness writes JSON lines. The
report generator should not care about that difference, so everything is
normalized here.

Recall is computed exactly the way ann-benchmarks computes it — a distance
threshold rather than an id intersection — so numbers produced by this framework
are directly comparable to published ann-benchmarks results. Using an id
intersection would disagree whenever a dataset contains ties at the k-th
distance, which is common in quantised embeddings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

import numpy

# The measurement-version machinery moved to its own module when this file
# reached 793 of its 800 allowed lines. Imported as a module and called through
# it, so there is exactly one place to patch when a test needs to stand the
# guard down -- binding the functions here as well would give every reader its
# own copy of the reference and a patch would reach only some of them.
from report import disclosures as _disclosures, measurement_versions as _versions

# Re-exported, not re-implemented: `from report.loaders import ...` is how the
# existing callers and tests reach these. New code should import
# `report.measurement_versions` directly.
from report.measurement_versions import (  # noqa: F401
    MixedMeasurementVersions,
    declared_ops_version,
    ops_measurement_version,
    ops_measurement_versions,
    refuse_mixed_measurement_versions,
)

# Extracted for the same reason. Aliased rather than renamed at the call sites,
# which read them a dozen times; `report/coercers.py` has the argument for why
# the conversion happens at all. New code should import it directly.
from report.coercers import (  # noqa: F401
    maybe_bool as _maybe_bool,
    maybe_float as _maybe_float,
    maybe_int as _maybe_int,
    maybe_str as _maybe_str,
)

# Moved to `report/builds.py`, which also says which build a summary may use.
from report.builds import build_params, build_signature  # noqa: E402,F401


def knn_recall(dataset_distances: numpy.ndarray, run_distances: numpy.ndarray,
               count: int, epsilon: float = 1e-3) -> float:
    """ann-benchmarks' knn recall: fraction within the true k-th distance."""
    total = 0.0
    n = len(run_distances)
    if n == 0:
        return float("nan")
    for i in range(n):
        threshold = dataset_distances[i][count - 1] + epsilon
        actual = int(numpy.count_nonzero(numpy.asarray(run_distances[i])[:count] <= threshold))
        total += actual / count
    return total / n


#: The pass label a recall record carries when neither the result file's
#: `resource_pass` attribute nor the directory layout says which pass produced
#: it. Named rather than spelled twice: `report/generate.py` has to hand it in
#: explicitly for the whole-tree fallback, and two copies of a sentinel drift.
UNKNOWN_PASS = "unknown"


def pass_or_unknown(value: Any) -> str:
    """One spelling for a record that names no resource pass.

    A record this pipeline wrote always names one -- `load_annb_results` sets
    the field unconditionally and the value is always truthy -- so this is
    reached only by a hand-assembled or third-party `records.jsonl`. It
    existed anyway, three times over: §4 keyed its tables on the raw value and
    printed `#### d — None`, the floor chart's stem became `qpsatrecall-None`,
    and the chart legend said `unknown`. Three spellings of one thing in one
    report, one of them a Python repr in a published artefact -- and §4 printed
    two tables for one engine under an IDENTICAL heading when a falsy pass sat
    beside a named one, because the count filtered it out and the table key
    did not.
    """
    return UNKNOWN_PASS if not value else str(value)


def load_annb_results(annb_dir: str, datasets_dir: str,
                      run_id: str,
                      resource_pass_default: str = UNKNOWN_PASS
                      ) -> List[Dict[str, Any]]:
    """Walk ann-benchmarks HDF5 results and emit recall/QPS records."""
    records: List[Dict[str, Any]] = []
    if not os.path.isdir(annb_dir):
        return records

    # Imported only once there is something to read, so an ops-only run can be
    # reported on a machine without h5py.
    import h5py

    from harness.metrics.records import PHASE_RECALL

    truth_cache: Dict[str, numpy.ndarray] = {}

    for root, _dirs, files in os.walk(annb_dir, followlinks=True):
        for filename in files:
            if not filename.endswith(".hdf5"):
                continue
            path = os.path.join(root, filename)
            try:
                with h5py.File(path, "r") as fh:
                    attrs = dict(fh.attrs)
                    dataset_name = str(attrs.get("dataset", ""))
                    count = int(attrs.get("count", 10))

                    if dataset_name not in truth_cache:
                        truth_path = os.path.join(datasets_dir, f"{dataset_name}.hdf5")
                        if not os.path.exists(truth_path):
                            print(f"[report] skipping {path}: dataset "
                                  f"{dataset_name} not available for ground truth")
                            continue
                        with h5py.File(truth_path, "r") as tf:
                            truth_cache[dataset_name] = numpy.asarray(tf["distances"])

                    truth = truth_cache[dataset_name]
                    run_distances = numpy.asarray(fh["distances"])
                    times = numpy.asarray(fh["times"]) if "times" in fh else None

                    recall = knn_recall(truth, run_distances, count)
                    best = float(attrs.get("best_search_time", float("nan")))
                    qps = 1.0 / best if best and best > 0 else float("nan")

                    record: Dict[str, Any] = {
                        "run_id": run_id,
                        "phase": PHASE_RECALL,
                        # The directory ann-benchmarks wrote the file into is
                        # named for the algorithm and is authoritative. The
                        # `engine` attribute is written by the module, which
                        # reported its dialect name -- shared between MariaDB
                        # 11.8 and 12.3 -- so 12.3 results arrived labelled
                        # "mariadb" despite sitting in a mariadb123 directory.
                        "engine": (os.path.basename(os.path.dirname(path))
                                   or str(attrs.get("engine",
                                                    attrs.get("algo", "unknown")))),
                        "engine_version": _maybe_str(attrs.get("engine_version")),
                        "dataset": dataset_name,
                        "metric_space": str(attrs.get("distance", "")),
                        # The pass is recorded by the module when it can read
                        # VB_RESOURCE_PASS; otherwise it is recovered from the
                        # directory the results were written into.
                        "resource_pass": str(
                            attrs.get("resource_pass")
                            or _pass_from_path(path, annb_dir)
                            or resource_pass_default),
                        "storage_engine": _maybe_str(attrs.get("storage_engine")),
                        "k": count,
                        "recall_at_k": round(float(recall), 6),
                        "qps": round(float(qps), 3) if qps == qps else None,
                        "clients": 1,
                        "timestamp": _maybe_str(attrs.get("timestamp")) or "",
                        "m": _maybe_int(attrs.get("M")),
                        "ef_search": _maybe_int(attrs.get("ef_search")),
                        "ef_construction": _maybe_int(attrs.get("ef_construction")),
                        "build_mode": _maybe_str(attrs.get("build_mode")),
                        # The one swept build axis no column carried, so the
                        # floor table could not tell Percona Search's
                        # scalar and unquantized curves apart (25 Aug run).
                        "quantization": _maybe_str(attrs.get("quantization")),
                        "index_bytes": _maybe_int(attrs.get("index_bytes")),
                        "march": _maybe_str(attrs.get("march")),
                        "vector_index_used": _maybe_bool(attrs.get("vector_index_used")),
                        "extra": {
                            "parameters": _maybe_str(attrs.get("name")),
                            "batch_mode": bool(attrs.get("batch_mode", False)),
                            "run_count": _maybe_int(attrs.get("run_count")),
                            "source_file": os.path.relpath(path, annb_dir),
                            # Used to detect a result file that predates the
                            # run it is being reported under. ann-benchmarks
                            # skips configurations that already have results,
                            # so an unchanged tree silently reappears in a new
                            # report.
                            "source_mtime": os.path.getmtime(path),
                            **_disclosures.query_path_extra(attrs),
                            **_disclosures.warmup_extra(attrs),
                        },
                    }
                    if attrs.get("index_size") is not None:
                        # ann-benchmarks records index size in kilobytes.
                        record.setdefault("index_bytes",
                                          int(float(attrs["index_size"]) * 1024))
                    # The build cost: measured since this framework existed
                    # and dropped by this reader for just as long, so the
                    # published claim was the ratio of the KNOB ("16 times the
                    # build effort") against a measured 2.17x. `build_time` is
                    # the only timing EVERY engine writes and is the sole
                    # source of `build_wall_s`. The load is spelled
                    # `ingest_seconds` by `vb_mysql` and `load_seconds` by the
                    # pgvector, valkey and mongodb modules, never both at once,
                    # and is one quantity, so one field reads both;
                    # `build_seconds` beside them is NOT read, because
                    # mongodb's includes the load and the other two's do not.
                    # No rate is derived -- the file holds no row count.
                    # Absent stays absent rather than 0.0, and a NaN is dropped
                    # because `json` writes it as the bare token `NaN`, which
                    # is not JSON and which `allow_nan=False` refuses to emit
                    # -- the reason the `qps` line above drops one. NOT because
                    # `generate.py`'s `default=str` would stringify it: by this
                    # point `maybe_float` has already called `float()`, so
                    # `default` is never reached. Why each
                    # of those, with the measured figures:
                    # tests/test_e18_ann_build_timing_reaches_the_records.py
                    for field, attributes in (
                            ("build_wall_s", ("build_time",)),
                            ("ingest_wall_s", ("ingest_seconds", "load_seconds")),
                    ):
                        for attribute in attributes:
                            value = _maybe_float(attrs.get(attribute))
                            if value is not None and value == value:
                                record[field] = value
                                break
                    if times is not None and len(times):
                        record["latency_p50_ms"] = float(numpy.percentile(times, 50) * 1000)
                        record["latency_p95_ms"] = float(numpy.percentile(times, 95) * 1000)
                        record["latency_p99_ms"] = float(numpy.percentile(times, 99) * 1000)
                        record["queries_executed"] = int(len(times))
                    records.append(record)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the report
                print(f"[report] could not read {path}: {exc}")

    return records


def _pass_from_path(path: str, annb_dir: str) -> Optional[str]:
    """Recover the resource pass from the results tree layout."""
    try:
        relative = os.path.relpath(path, annb_dir)
    except ValueError:
        return None
    head = relative.split(os.sep)[0]
    return head if head in ("normalized", "tuned") else None


def load_ops_records(run_dir: str) -> List[Dict[str, Any]]:
    """Read every ops-harness JSONL file in a run directory.

    Raises MixedMeasurementVersions if the directory holds records from more
    than one ops measurement version, or records a version behind the manifest
    sitting above them. See refuse_mixed_measurement_versions.
    """
    from harness.metrics.records import read_records

    out: List[Dict[str, Any]] = []
    if not os.path.isdir(run_dir):
        return out
    for filename in sorted(os.listdir(run_dir)):
        if not (filename.startswith("ops-") and filename.endswith(".jsonl")):
            continue
        try:
            # Materialised: read_records is a generator, and the relabelling
            # below walks it before out.extend does.
            records = list(read_records(os.path.join(run_dir, filename)))
        except Exception as exc:  # noqa: BLE001
            print(f"[report] could not read {filename}: {exc}")
            continue
        # The orchestrator writes ops-<engine>-<dataset>-<pass>-<tag>.jsonl, so
        # the filename is authoritative about which engine produced the file.
        # The records themselves were not: the harness recorded the driver's
        # own name, and mariadb123 shares MariaDBDriver, so every 12.3
        # measurement arrived labelled "mariadb". Trusting the filename repairs
        # runs recorded before that was fixed, without re-measuring them.
        engine = _engine_from_ops_filename(filename)
        if engine:
            for r in records:
                r["engine"] = engine
        out.extend(records)
    # After the relabelling above, so the records the refusal names carry the
    # engine the operator will see in the filename rather than the driver's own
    # name. Before anything else in the report has run.
    _versions.refuse_mixed_measurement_versions(
        out, source=run_dir,
        declared=_versions.declared_ops_version(run_dir))
    return out


def _engine_from_ops_filename(filename: str) -> Optional[str]:
    """Engine name out of ops-<engine>-<dataset>-<pass>-<tag>.jsonl.

    Datasets contain hyphens, so match the longest known engine prefix rather
    than splitting on the separator.
    """
    stem = filename[len("ops-"):-len(".jsonl")]
    known = ("mariadb123", "mariadb", "alisql", "pgvector")
    for name in known:
        if stem.startswith(name + "-"):
            return name
    return None


def load_memory_series(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Read the per-container memory timeseries files.

    Keyed by the filename stem so a phase's peak can be derived by intersecting
    with that phase's time window, rather than trusting a single scalar that may
    have been sampled at the wrong moment.
    """
    import json

    series: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(run_dir):
        return series
    for filename in sorted(os.listdir(run_dir)):
        if not (filename.startswith("mem-") and filename.endswith(".jsonl")):
            continue
        rows: List[Dict[str, Any]] = []
        with open(os.path.join(run_dir, filename)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # truncated tail from an interrupted run
        if rows:
            series[filename[4:-6]] = rows
    return series


def peak_from_series(rows: List[Dict[str, Any]]) -> Optional[int]:
    values = [r.get("rss_bytes") for r in rows if r.get("rss_bytes")]
    return max(values) if values else None


def memory_stem(record: Dict[str, Any]) -> str:
    """Name of the memory series belonging to one ops unit.

    Mirrors the stem `orchestrator/cli.py` gives the `mem-*.jsonl` file it
    writes, including the rule that InnoDB stays unsuffixed so checkpoints from
    runs that predate the storage-engine sweep are still honoured. Kept here
    rather than imported from the orchestrator because this package runs inside
    a bench image that mounts only report/ and harness/.

    Matching on a prefix instead matched both storage engines and took
    whichever the dict happened to yield first, which published MyISAM's peak
    memory as InnoDB's.
    """
    stem = (f"{record.get('engine')}-{record.get('dataset')}"
            f"-{record.get('resource_pass')}-m{record.get('m')}"
            f"-{record.get('build_mode')}")
    storage = (record.get("storage_engine") or "").lower()
    if storage and storage != "innodb":
        stem += f"-{storage}"
    return stem


# Within this of the container limit counts as "at the ceiling". cgroup peak
# accounting cannot exceed memory.max, so an engine under sustained pressure
# reports a value just below the limit rather than above it, and an exact
# comparison would never fire.
_CEILING_TOLERANCE = 0.98


def ceiling_pressure(rows: List[Dict[str, Any]],
                     limit_bytes: Optional[int]) -> Optional[Dict[str, Any]]:
    """Detect a phase that spent significant time against its memory limit.

    An engine pinned at its cgroup limit is reclaiming continuously, so its
    throughput and latency describe the budget rather than the implementation.
    That is invisible in the records themselves: the phase either succeeds with
    depressed numbers or is OOM-killed and shows up only as a non-zero exit.

    Returns None when there is nothing to report, so callers can filter falsy.
    """
    if not limit_bytes:
        return None
    values = [r.get("rss_bytes") for r in rows if r.get("rss_bytes")]
    if not values:
        return None
    threshold = limit_bytes * _CEILING_TOLERANCE
    at_ceiling = sum(1 for v in values if v >= threshold)
    if not at_ceiling:
        return None
    first = next(i for i, v in enumerate(values) if v >= threshold)
    return {
        "peak_bytes": max(values),
        "limit_bytes": limit_bytes,
        "samples": len(values),
        "samples_at_ceiling": at_ceiling,
        "fraction_at_ceiling": at_ceiling / len(values),
        "first_hit_fraction": first / len(values),
    }


def pareto_frontier(points: List[Dict[str, Any]], x: str = "recall_at_k",
                    y: str = "qps") -> List[Dict[str, Any]]:
    """Upper-left envelope: the points not beaten on both recall and QPS.

    Comparing raw scatter points is misleading — an engine with a badly chosen
    parameter looks slow at a recall where a different parameter of the same
    engine is fast. The frontier is what an operator could actually achieve.
    """
    usable = [p for p in points if p.get(x) is not None and p.get(y) is not None]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda p: (p[x], p[y]))
    frontier: List[Dict[str, Any]] = []
    best_qps = float("-inf")
    for point in reversed(ordered):        # highest recall first
        if point[y] > best_qps:
            frontier.append(point)
            best_qps = point[y]
    return list(reversed(frontier))

# A phase is "contended" when the machine burned meaningfully more CPU than the
# container we were measuring did. Some gap is always there -- the kernel, the
# sampler itself, the other cpuset -- so the threshold is a share of a core,
# not zero.
FOREIGN_CPU_CORES = 2.0


def foreign_cpu(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """CPU the host spent that the measured container did not.

    Three AliSQL recall points were taken while something outside the harness
    was on the machine. Nothing recorded it, so it surfaced five days later as
    a latency ratio somebody happened to compare by hand. Both counters are in
    every sample now, and the difference between them over the phase is the
    answer without anybody comparing anything.

    Returns None when the series cannot answer -- an older run without the
    host fields, or too few samples to difference.
    """
    worst = None
    for segment in split_sessions(rows):
        found = _foreign_cpu_one(segment)
        if found and (worst is None or found["foreign_cores"] > worst["foreign_cores"]):
            worst = found
    return worst


def _foreign_cpu_one(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """foreign_cpu for a single measurement. See split_sessions."""
    usable = [r for r in rows
              if r.get("host_cpu_seconds") is not None
              and r.get("cpu_seconds") is not None]
    if len(usable) < 2:
        return None
    first, last = usable[0], usable[-1]
    elapsed = (last.get("t") or 0) - (first.get("t") or 0)
    if elapsed <= 0:
        return None
    host = last["host_cpu_seconds"] - first["host_cpu_seconds"]
    ours = last["cpu_seconds"] - first["cpu_seconds"]
    foreign = host - ours
    cores = foreign / elapsed
    if cores < FOREIGN_CPU_CORES:
        return None
    loads = [r["host_load1"] for r in usable if r.get("host_load1") is not None]
    return {
        "elapsed_s": round(elapsed, 1),
        "container_cores": round(ours / elapsed, 2),
        "foreign_cores": round(cores, 2),
        "peak_load1": round(max(loads), 1) if loads else None,
    }

# A gap this long between consecutive samples cannot be sampling jitter -- the
# sampler runs four times a second -- so it separates two measurements that
# were appended to one file. Used only for series written before sessions were
# stamped.
SESSION_GAP_S = 300.0


def split_sessions(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """One list per measurement, from a file that may hold several.

    The sampler appends, so a phase that runs twice for the same engine,
    dataset and resource pass writes both into one file. That is deliberate --
    the earlier measurement is not wrong and should not be discarded -- but
    treating the result as a single continuous series draws a line across the
    gap and differences the two ends as though they bracketed one run. When
    MHNSW and VIDX were given a second graph degree five days after the first
    run, that gap was 123 hours.

    Splits on the recorded session where there is one, and on a long gap
    between samples otherwise, so files written before the marker existed are
    still read correctly.
    """
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r.get("t") or 0)
    if any(r.get("session") is not None for r in ordered):
        by_session: Dict[Any, List[Dict[str, Any]]] = {}
        for r in ordered:
            by_session.setdefault(r.get("session"), []).append(r)
        return [v for _k, v in sorted(by_session.items(), key=lambda kv: str(kv[0]))]

    segments: List[List[Dict[str, Any]]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if (current.get("t") or 0) - (previous.get("t") or 0) > SESSION_GAP_S:
            segments.append([])
        segments[-1].append(current)
    return segments


# ---------------------------------------------------------------------------
# ef_search calibration
# ---------------------------------------------------------------------------

def calibration_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ef_search sweep's own records, selected by explicit phase equality.

    A named selector rather than a comprehension at each call site, because
    nothing in this package's aggregations picks a new phase up on its own.
    Each one names the phase it wants: the concurrency chart's
    `_is_concurrency_point`, `summarize`'s build / concurrency / filtered /
    churn / ingest group-by, `attach_peak_rss`, `storage_breakdown` and
    `churn_retention` all compare `phase` against a literal, so
    `ef_calibration` records were read out of the JSONL by `load_ops_records`
    and then dropped on the floor by every one of them, including code that
    looks generic.

    That default is right for all of those and must stay: a calibration point
    carries `clients=1` and a `qps`, so admitting it to the concurrency
    aggregation would add up to twelve single-client points per engine to a
    sweep measured over a shared query pool and move the published QPS.

    Three consumers SHOULD see these records, and all three already do -- by
    being written as inequalities or as no test at all, which is what makes
    "every consumer filters by equality" the wrong generalisation to reason
    from:

    * `summarize`'s plan-failure loop has no phase filter, so a calibration
      probe that ran without the vector index in the plan is reported like any
      other full scan. Correct: it is a full scan.
    * `measurement_versions._is_ops_record` is `phase != PHASE_RECALL`, so
      `_is_ops_record({"phase": "ef_calibration"})` is True (measured) and the
      sweep's records are grouped and version-checked with the rest of the ops
      path. Also correct -- they are ops records, written by the same harness
      under the same `OPS_MEASUREMENT_VERSION`.
    * `webui/records.py`'s `facets`, `filter_records` and `series` do not
      mention `phase` at all (measured), so `ef_calibration` shows up as a
      facet value in the Explore tab's Raw view. The web UI has no calibration
      VIEW -- `webui/static/fields.js` declares one per *reportable workload*
      rather than one per phase, five of them against seven entries in
      `ALL_PHASES`, with `ingest` unviewed as well as `ef_calibration`
      (measured) -- so the Explore tab is blind to the sweep rather than wrong
      about it, and `explore.js`'s per-view `phaseFilter` is what keeps these
      records out of its concurrency view. Left for a follow-up; `webui/` is
      not this change's to edit.

    This function's callers are the fourth consumer, and the only one that
    needed writing.
    """
    from harness.metrics.records import PHASE_CALIBRATION

    return [r for r in records if r.get("phase") == PHASE_CALIBRATION]


def operating_point_key(record: Dict[str, Any]) -> Any:
    """Which calibrated measurement unit a record belongs to.

    PR 2 §2.3.1 asks for a summary "keyed by (engine, dataset, pass)". That
    triple is not unique: `orchestrator/cli.py` invokes the harness once per
    (M, build_mode, storage_engine) and each invocation runs its own sweep, so
    `config/profiles/m-sweep.yml` -- `m_values: [6, 8, 16, 24, 32]` --
    calibrates five times per engine, dataset and pass. A literal three-part
    key would collapse five independent operating points into one row and
    publish whichever the iteration order reached last.

    The key is the resource pass plus `build_signature`, which is this repo's
    existing definition of "everything about a measurement except the query
    effort" -- and `ef_search` is exactly the query effort the sweep chooses,
    so the signature is already the right grain. On a profile that measures one
    configuration it reduces to the spec's triple.

    `resource_pass` is added because `build_signature` omits it, and for the
    MySQL-family engines nothing else in the signature separates the two
    passes. `config/resources/tuned.yml` sweeps `pgvector_ef_construction`
    and `valkey_ef_construction` over [64, 200, 400], which IS in the
    signature, and pins `mariadb_storage_engines: [InnoDB]` -- so MariaDB's
    normalized and tuned sweeps carry the same engine, dataset, M, build mode,
    storage engine and (absent) ef_construction, and collide on a key without
    the pass. Those are two separate sweeps on two different memory and CPU
    budgets, and merging them would report one ef for both.
    """
    return (record.get("resource_pass"), build_signature(record))


def _operating_point_sort_key(row: Dict[str, Any]) -> Any:
    """Stable order for the table, None-safe on every axis a profile may skip.

    `m` sorts NUMERICALLY. Stringifying it alongside the text axes ordered
    `m-sweep`'s five values as 16, 24, 32, 6, 8 -- in the very profile
    `operating_point_key` cites as the reason this key has the grain it does,
    and in a column a reader scans for a trend. The text axes still sort as
    text, and every axis keeps its None-last tiebreak because a profile may
    leave `build_mode` or `storage_engine` unset.
    """
    def text(value: Any) -> Any:
        return (value is None, str(value))

    def number(value: Any) -> Any:
        try:
            return (value is None, float(value), "")
        except (TypeError, ValueError):
            # A hand-edited record can hold anything here; sort it after the
            # real numbers rather than raising inside a sort.
            return (value is None, float("inf"), str(value))

    return (text(row.get("engine")), text(row.get("dataset")),
            text(row.get("resource_pass")), number(row.get("m")),
            text(row.get("build_mode")), text(row.get("storage_engine")))


def operating_points(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ef each calibrated unit settled on, at what recall, and why.

    A join, not a filter, and the two sides carry different halves of the
    answer:

    * the *chosen* ef and the status come from the DOWNSTREAM records. The
      sweep writes one record per grid point and marks none of them as the
      choice -- `CalibrationResult` is returned to `harness/main.py`, which
      substitutes `ef_search` and then calls `Recorder.declare_run_extra`, so
      what the run actually measured at is the `ef_search` on the concurrency,
      filtered and churn records and the status is their
      `extra.calibration_status`. Reading the choice back off the measurement
      that used it cannot disagree with it. Re-deriving it here by reapplying
      `calibrate.select` to the curve could: it would be a second copy of the
      selection rule, and the first symptom of the two drifting apart would be
      a report whose "chosen ef" column names an ef no workload ran at.
    * the *recall* comes from the CALIBRATION record at that ef. The sweep
      measured it over `--max-queries` single-client queries against freshly
      warmed caches, which is the measurement the choice was made on. The
      concurrency records carry a `recall_at_k` too and it is NOT this number:
      it is the median window's, over a shared query pool, under contention
      from every other client. Using it here would put a figure in the
      "Recall@k" column that the ef was not chosen by, which is how this table
      would come to disagree with the sweep that produced it.

    A unit with only one half is kept rather than dropped, with the missing
    half as None. A sweep whose downstream workloads never ran cost minutes per
    engine and produced no operating point; a report that simply omits it says
    the sweep never happened.
    """
    from harness.metrics.records import (CALIBRATION_STATUS_KEY,
                                         PHASE_CALIBRATION,
                                         TARGET_RECALL_KEY)

    curves: Dict[Any, List[Dict[str, Any]]] = {}
    chosen: Dict[Any, List[Dict[str, Any]]] = {}
    for record in records:
        extra = record.get("extra") or {}
        if record.get("phase") == PHASE_CALIBRATION:
            curves.setdefault(operating_point_key(record), []).append(record)
        elif extra.get(CALIBRATION_STATUS_KEY):
            # Only records carrying a status, which is what makes this the
            # downstream half of a CALIBRATED run rather than every ops record
            # in the directory. `stamp_run_extra` drops a None, so in `fixed`
            # mode no record has the key and no operating point is claimed --
            # which is correct: nothing aimed at a recall.
            chosen.setdefault(operating_point_key(record), []).append(record)

    rows: List[Dict[str, Any]] = []
    for key in set(curves) | set(chosen):
        curve = sorted(curves.get(key, []),
                       key=lambda r: (r.get("ef_search") is None,
                                      r.get("ef_search")))
        users = chosen.get(key, [])
        sample = (users or curve)[0]

        # The set, never the first element: two disagreeing efs under one key
        # are two measurements presented as one, and printing either as "the"
        # operating point is the failure this column exists to expose. A run
        # directory cannot produce it -- one JSONL file per unit, truncated per
        # attempt -- but `--from-records` over a hand-merged file can.
        efs = sorted({r.get("ef_search") for r in users
                      if r.get("ef_search") is not None})
        statuses = sorted({(r.get("extra") or {}).get(CALIBRATION_STATUS_KEY)
                           for r in users
                           if (r.get("extra") or {}).get(
                               CALIBRATION_STATUS_KEY)})
        targets = sorted({(r.get("extra") or {}).get(TARGET_RECALL_KEY)
                          for r in users
                          if (r.get("extra") or {}).get(TARGET_RECALL_KEY)
                          is not None})

        ef = efs[0] if len(efs) == 1 else None
        at_ef = next((r for r in curve if r.get("ef_search") == ef), None)
        rows.append({
            # The key itself, so a consumer that has to match a row back to the
            # records it came from uses this function's definition of the unit
            # rather than rebuilding it from the row's own columns. Rebuilding
            # it does not work: `build_signature`'s column fallback includes
            # `ef_construction` and `extra.quantization`, neither of which is
            # a column of this row, so a reconstructed key silently fails to
            # equal the one the records hash to -- and the symptom is a chart
            # that marks no chosen point at all.
            "key": key,
            "engine": sample.get("engine"),
            "dataset": sample.get("dataset"),
            "resource_pass": sample.get("resource_pass"),
            "m": sample.get("m"),
            "build_mode": sample.get("build_mode"),
            "storage_engine": sample.get("storage_engine"),
            "ef_search": ef,
            "ef_conflict": efs if len(efs) > 1 else None,
            "effective_ef": ((at_ef.get("extra") or {}).get("effective_ef")
                             if at_ef else None),
            "recall_at_k": at_ef.get("recall_at_k") if at_ef else None,
            "status": statuses[0] if len(statuses) == 1 else None,
            "status_conflict": statuses if len(statuses) > 1 else None,
            "target_recall": targets[0] if len(targets) == 1 else None,
            # How much of the grid ran, straight from the sweep's own record: a
            # sweep that early-stopped after three of twelve points says so
            # here rather than only in the log, and a reader comparing two
            # engines' chosen ef needs to know neither was capped by a grid
            # that stopped short.
            "grid_size": ((curve[0].get("extra") or {}).get("grid_size")
                          if curve else None),
            "points_measured": len(curve) or None,
            "ef_grid": [r.get("ef_search") for r in curve] or None,
        })
    rows.sort(key=_operating_point_sort_key)
    return rows


def calibration_targets(records: List[Dict[str, Any]]) -> List[float]:
    """Every distinct recall target the given records were measured against.

    A list rather than one value because a report can honestly hold more than
    one: `--target-recall` is a flag, and `--from-records` over records merged
    from two runs can hand this package both. A caller naming the target in a
    chart title or a footnote has to decide what to say for zero and for two,
    and it cannot do that from a single float.

    Read off the records rather than from `DEFAULT_TARGET_RECALL`, because the
    default is what the flag falls back to and not what the run used. An empty
    list means `ef_mode: fixed` -- no target was pursued, and `stamp_run_extra`
    leaves the key out rather than writing the default as though one had been.
    """
    from harness.metrics.records import TARGET_RECALL_KEY

    found = {(r.get("extra") or {}).get(TARGET_RECALL_KEY) for r in records}
    return sorted(float(t) for t in found
                  if isinstance(t, (int, float)) and not isinstance(t, bool))

