"""E18 -- the recall harness measures the build and then throws the number away.

THE DEFECT
----------
Every ann-benchmarks result file this framework writes carries a build timing
and a load timing, and the report read neither. The build timing is spelled
once; the load timing is spelled twice, and which spelling a file uses depends
on the engine:

* `build_time`, written by upstream `ann_benchmarks/runner.py:build_index`
  around its `algo.fit` call. **Every** engine has it -- it is upstream's, not
  ours.
* `ingest_seconds`, written by
  `overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py:get_additional`, which
  rounds its own wall clock to 3 decimals. That base class serves the MySQL
  family only: **mariadb, mariadb123, alisql and percona**.
* `load_seconds`, written by the **pgvector, valkey and mongodb** modules,
  which do not derive from `vb_mysql` and write no `ingest_seconds` at all.
  Confirmed in the three modules' own `get_additional` and in the archived
  artifacts: mongodb `load_seconds` 1.899 / `build_seconds` 30.228, valkey
  1.184 / 4.022, pgvector 10.280 / 71.544, and `ingest_seconds` absent from
  all three.

`ingest_seconds` and `load_seconds` are one quantity under two names -- wall
time of the corpus load -- so `ingest_wall_s` reads whichever is present. The
three engines that write neither would otherwise have been the three engines
left with a build time and no load time, and they are the three the second
clock exists for.

`build_seconds`, which those same three write, is deliberately **not** read.
It is not one quantity: mongodb's is `load + index-ready`, pgvector's and
valkey's exclude the load in `post` mode and include it in incremental mode,
and an ann record carries no `separable_build` for a reader to tell which. One
column holding three definitions of "build" is the defect
`charts.build_cost`'s own docstring describes on the ops path.

`report/loaders.py:load_annb_results` copies a dozen other attributes into the
record and, out of the whole result file, takes exactly one derived value off
the file itself -- `os.path.getmtime(path)` into `extra.source_mtime`. So the
cost half of "what does a higher `ef_construction` buy" is in the artifacts, in
no record, and therefore in no report.

What that cost. Read straight out of the archived `fashion-mnist-784-euclidean`
sweep at `annb/tuned/a9349676/.../percona/`, M=16, `ef_search` 10:

    ef_construction    build_time    ingest_seconds
                 50     130.790         130.372
                200     161.559         161.545
                800     283.275         283.260

50 -> 800 is **2.17x** the build time. The published report told the server
team it was "16 times the build effort" -- which is the ratio of the
`ef_construction` values, not of anything measured, and is the figure a reader
reaches for when the measured one is not in the report. The numbers were in the
files the whole time.

WHY BOTH FIELDS, which is not "the gap is build work"
-----------------------------------------------------
That was the first answer here and the evidence refuses it. The gap between
`build_time` and the load is 0.418 s at `ef_construction` 50, 0.014 at 200 and
0.015 at 800 -- **largest at the cheapest configuration**, which graph
construction would not do. For an incremental engine there is no separate
build to measure at all: `vb_mysql.fit` sets `_build_seconds =
_ingest_seconds` in so many words, because the graph is maintained on every
INSERT. The gap is `fit()`'s fixture overhead -- `DROP`/`CREATE DATABASE`, the
`CREATE TABLE` carrying the index, and `_measure_index_bytes` walking the
datadir -- and pgvector's incremental artifact shows the same ~0.42 s gap with
`build_seconds == load_seconds`.

The two fields earn their place for a different reason: they are two clocks of
different SCOPE, and the scope matters most where the index goes live after the
load rather than during it. mongodb loads in 1.899 s and upstream's whole-fit()
clock reads 30.450 s -- **16.0x the load**, a difference of 28.551 s. "Time to
load the corpus" and "time until the index is queryable" are both findings,
they differ by more than an order of magnitude on that engine, and one field
cannot carry both. For the MySQL family they nearly coincide, and that
coincidence is itself the statement that those engines have no separable build.

**THE WARNING THAT REPLACES THE WITHDRAWN CLAIM.** Dropping "the gap is build
work" leaves a reader needing to know what the gap IS, because
`build_wall_s - ingest_wall_s` is now computable for every engine and will be
computed. It carries fit()'s fixture overhead -- `DROP`/`CREATE DATABASE`, the
`CREATE TABLE` carrying the index, `_measure_index_bytes` walking the datadir
-- measured at 0.014-0.418 s across the three percona configurations above and
0.325-0.333 s across pgvector's four incremental ones. Against mongodb's
28.551 s that is noise. For the MySQL family it is the ENTIRE quantity: there
the difference is overhead and nothing else, because the graph is built inside
the load. So the difference is a real measurement only where it is large, and
a small one is not a small amount of build work -- it is no build work and
some fixture.

THE COLLATERAL HAZARD, which is why this unit also edits `report/charts.py`
----------------------------------------------------------------------------
`charts.build_cost` selected its records on dataset and `build_wall_s is not
None` and did **not** filter on phase. Nothing leaked while no `recall_qps`
record carried a build time -- which is precisely what the loader change
above ends. Populating the field without a phase filter would have put the ann
path's builds onto the ops path's build-cost chart, where the `peak_rss_bytes`
and `index_bytes` panels sit: four panels of ops-harness quantities with
ann-container figures drawn through them, each measured in a different
container under a different memory budget. The fix is a named predicate,
`charts._is_build_cost_point`, the sibling of `charts._is_concurrency_point`,
which exists for the same reason -- selecting a chart's records on "has the
field" rather than "is the measurement" has now produced this class of defect
twice.

The two neighbours were already safe and stay untouched: `render.py`'s §5
build table reads `summary["build"]`, which `generate._PHASE_GROUPS` filters to
`index_build`, and `charts.pass_comparison` names its phase in its dimension
table.

WHAT IS PINNED HERE
-------------------
* `TestTheTwoTimingsReachTheRecord` -- the headline. Both fields land, at the
  artifacts' own precision.
* `TestAnAbsentAttributeIsAbsentAndNotZero` -- a result file carrying neither
  attribute yields a record carrying neither key. A `0.0` would read as a
  build that cost nothing, and `build_cost`'s own selection is
  `is not None`, so a zero would also be a plotted point.
* `TestTheLoadIsSpelledTwiceAcrossTheEngines` -- the `load_seconds`
  fallback, the precedence between the two spellings, and `build_seconds`
  staying unread.
* `TestNoRateIsDerivedFromAFileThatCannotSupportOne` -- the result file holds
  no training-row count, so `ingest_rows_per_s` stays `None` rather than being
  computed from a number the file does not have. There is a second,
  independent reason, and it is worth stating because it bounds what
  `ingest_wall_s` on a recall record MEANS: the ann path does not measure
  ingest under control. `vb_mysql._resolve_insert_threads`' own docstring says
  so -- "Ingest speed is not what this pass measures -- the ops harness
  measures it properly and under controlled conditions. Here we only need the
  load to finish in reasonable time, so parallelism is allowed". All four
  load paths resolve it with the identical `max(1, min(8, affinity))` rule
  (`vb_mysql._resolve_insert_threads` and the three modules' own), so the
  field is at least not engine-BIASED -- but only the MySQL family records
  `insert_threads`, so for pgvector, valkey and mongodb the thread count is
  not on the record at all. So this field is the build cost's
  companion, not an ingest-rate measurement, and a rate derived from it would
  invite exactly the comparison against the ops path's `ingest_rows_per_s`
  that it cannot support. The guard is the record
  assertion; `ARCHIVED_RESULT_ATTRIBUTES` below is the evidence behind the
  decision and is checked against the fixture, not asserted as a tautology.
* `TestAnnBuildsStayOffTheOpsBuildCostChart` -- the phase filter, asserted on
  what `build_cost` RETURNS rather than on how it is spelled or on the drawn
  PNG, so one assertion holds it on a host with no matplotlib. An earlier
  version of this file asserted on `inspect.getsource`; a reviewer defeated it
  with a mutant that re-spelled the leak around the predicate call while
  leaving the call in place, and was green on a bare host. A test that pins a
  spelling is not a test of behaviour.
* `TestTheCoercersMovedWithoutLoss` -- the `report/coercers.py` extraction,
  after `report/measurement_versions.py`'s own precedent test.

HOW THE SYNTHETIC RESULT FILES ARE BUILT, and why not with h5py
---------------------------------------------------------------
h5py lives in the bench images and not on a host built from the documented
prerequisites (`CLAUDE.md`, "Host requirements"), so these tests cannot write a
real HDF5 container -- and a test that skips on the measuring host's own
toolchain is not a test of this loader. `load_annb_results` imports `h5py`
inside its body, once there is something to read, so a stub in `sys.modules`
reaches it. The stub is an HDF5 *reader* only: the files are real files, on
disk, with the real `.hdf5` extension, in the real `<dataset>/<k>/<engine>/`
layout, reached by the real `os.walk`, gated by the real ground-truth
existence check and stamped with the real `os.path.getmtime`. Only the binary
container is JSON. Attribute values are handed back as the numpy scalars h5py
hands back, because that is what the coercers have to survive.

WHY THE COERCION MATTERS, which is not "json.dump would fail"
-------------------------------------------------------------
That was the second wrong answer here. `numpy.float64` is a subclass of
`float` and serialises as a JSON number with nothing wrong. The hazard is the
opposite of a loud failure: `report/generate.py` writes records with
`json.dumps(..., default=str)`, so a value `json` cannot serialise is not an
error -- it is silently stringified. `numpy.float32` is NOT a `float`
subclass, so it lands in `report/records.jsonl` as the JSON **string**
`"283.27"` and `report --from-records` reads it back as a `str`. A measurement
whose Python type depends on which numpy width a result file happened to use
is one a chart sorts as text or drops on a comparison, and nothing anywhere
says so. `float()` at the boundary is what makes the record's type the
record's own.

A NaN is a separate matter, is dropped by `load_annb_results` rather than
here, and is NOT dropped for the reason one revision of the loader comment
gave. It claimed `default=str` would stringify it; `maybe_float` calls
`float()` before that check, so the value is a plain `float` by then and
`default` is never reached -- measured with a `default` that would have
returned a sentinel and which recorded zero calls. What happens instead is
that `json` writes the bare token `NaN`, which `allow_nan=False` refuses to
emit and a strict reader rejects. Measured in `jq`, the tool the schema
charter in `harness/metrics/records.py` has in mind when it promises that
"the raw data can be queried with anything that reads JSON lines":

    .build_wall_s renders as        null
    .build_wall_s | type says       "number"
    .build_wall_s == null is        false
    select(.build_wall_s > 100)     silently drops the row
    jq's exit status                0

The row is lost with no diagnostic and cannot be found again by testing for
null -- only `isnan` sees it. That is worse than a parse error, and it is why
the `qps` line above drops a NaN too.

`tests/test_e14_per_pass_manifest.py` and `tests/test_e17_report_scope.py`
monkeypatch `load_annb_results` away wholesale for the same reason. That is
the right call where the loader is not the subject; here it is the subject.
"""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Any, Dict, List, Optional, Sequence

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import records as records_mod  # noqa: E402
from report import charts, loaders  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Skips the two tests that read a drawn figure. The phase filter itself is
#: held by `test_a_recall_record_alone_draws_no_build_chart`, which asserts on
#: `build_cost`'s return value and runs everywhere; these two add what the
#: reader would have SEEN -- the plotted series and the published subtitle.
needs_matplotlib = pytest.mark.skipif(
    not charts.have_matplotlib(),
    reason="reads a drawn figure; matplotlib lives in the bench images")


# ---------------------------------------------------------------------------
# The measured figures this unit exists for
# ---------------------------------------------------------------------------

#: `(ef_construction, build_time, ingest_seconds)` from
#: `results/annb/tuned/a9349676/fashion-mnist-784-euclidean/10/percona/`,
#: M=16, `ef_search` 10, read out of the archived artifacts. Rounded to the
#: three decimals `ingest_seconds` itself carries.
ARCHIVED_EF_CONSTRUCTION_SWEEP = (
    (50, 130.790, 130.372),
    (200, 161.559, 161.545),
    (800, 283.275, 283.260),
)

#: `build_time` at `ef_construction` 800, at the full float64 precision the
#: artifact holds, so that a loader which rounds on the way in is caught.
#: `ingest_seconds` has no such value: the overlay rounds it before writing.
ARCHIVED_BUILD_TIME_FULL_PRECISION = 283.274920463562

#: Every attribute name in that artifact, read out of its object header by
#: parsing the HDF5 attribute-message layout (there is no h5py on the
#: measuring host). This is the evidence behind
#: `TestNoRateIsDerivedFromAFileThatCannotSupportOne`: no row count, no
#: training-set size, no `rows`, so no rate is derivable from the file.
#:
#: **It was wrong once and the error is instructive.** The first version
#: listed 23 of these 25, missing `candidates` (float64, 10.0) and
#: `expect_extra` (bool) -- both written by upstream
#: `ann_benchmarks/runner.py` rather than by one of our modules, which is
#: exactly the half a census assembled by reading our own code misses.
#: Neither is a row count, so the decision survived its own evidence being
#: incomplete; a list asserted complete and not complete is still a false
#: statement in shipped text. `test_the_synthetic_attributes_are_all_real`
#: now checks this against the fixture, so a name here that is not in a real
#: file, or a fixture key that is not in this list, fails.
ARCHIVED_RESULT_ATTRIBUTES = (
    "M", "algo", "batch_mode", "best_search_time", "build_time", "candidates",
    "count", "dataset", "distance", "ef_construction", "ef_search", "engine",
    "engine_version", "expect_extra", "index_bytes", "index_size",
    "ingest_seconds", "insert_threads", "march", "metric", "name",
    "resource_pass", "run_count", "storage_engine", "vector_index_used",
)

#: `(engine, load_seconds, build_seconds, build_time)` from the archived
#: artifacts of the three engines whose modules do not derive from `vb_mysql`
#: and write `load_seconds` instead of `ingest_seconds`. Trees named, because
#: they are not all one tree:
#:
#:   mongodb   normalized/84129bf3/fashion-mnist-784-euclidean/10/mongodb/
#:                 euclidean_M_16_quantization_none_10.hdf5
#:   valkey    tuned/a9349676/fashion-mnist-784-euclidean/10/valkey/
#:                 euclidean_M_8_build_mode_post_efConstruction_64_10.hdf5
#:   pgvector  normalized/84129bf3/fashion-mnist-784-euclidean/10/pgvector/
#:                 euclidean_M_16_build_mode_post_efConstruction_200_200.hdf5
#:
#: mongodb is the row that makes the two clocks worth keeping apart, and each
#: of its figures belongs to a different pair -- an earlier version of this
#: note crossed them:
#:
#:   * 28.329 s is `build_seconds - load_seconds`, and is exactly the
#:     `index_ready_seconds` the module also writes: how long after the load
#:     ended before mongot could answer.
#:   * 16.0x is `build_time / load_seconds` (30.449686 / 1.899 = 16.035).
#:   * 28.551 s is `build_time - load_seconds`, which is what
#:     `build_wall_s - ingest_wall_s` equals on the record and therefore the
#:     only one of the three a report can compute.
ARCHIVED_LOAD_SECONDS_ENGINES = (
    ("mongodb", 1.899, 30.228, 30.44968605041504),
    ("valkey", 1.184, 4.022, 5.206523418426514),
    ("pgvector", 10.280, 71.544, 82.10410451889038),
)


# ---------------------------------------------------------------------------
# The HDF5 reader stub and the synthetic tree
# ---------------------------------------------------------------------------

def _as_h5py_value(value: Any) -> Any:
    """What h5py hands back for a scalar attribute of this Python type.

    Numbers come back as numpy scalars rather than Python ones, and that is
    the whole reason this function exists: `float(numpy.float64(...))` is the
    conversion the loader's coercer has to perform, and a test that handed it
    a Python float would not exercise it.
    """
    if isinstance(value, bool):
        return numpy.bool_(value)
    if isinstance(value, int):
        return numpy.int64(value)
    if isinstance(value, float):
        return numpy.float64(value)
    return value


class _StubH5File:
    """An HDF5 file whose container happens to be JSON. See the module docstring."""

    def __init__(self, path: str, mode: str = "r") -> None:
        if mode != "r":
            raise AssertionError("the loader only ever opens for reading")
        with open(path) as fh:
            blob = json.load(fh)
        self.attrs = {key: _as_h5py_value(value)
                      for key, value in blob.get("attrs", {}).items()}
        self._datasets = {key: numpy.asarray(value, dtype="float64")
                          for key, value in blob.get("datasets", {}).items()}

    def __enter__(self) -> "_StubH5File":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def __contains__(self, key: str) -> bool:
        return key in self._datasets

    def __getitem__(self, key: str) -> numpy.ndarray:
        return self._datasets[key]


@pytest.fixture
def stub_h5py(monkeypatch):
    """Put the reader where `load_annb_results`' own `import h5py` finds it."""
    monkeypatch.setitem(sys.modules, "h5py",
                        types.SimpleNamespace(File=_StubH5File))
    return None


#: k, and the width of every distance row below. Matches the `10` in the
#: archived tree's `<dataset>/10/<engine>/` path.
COUNT = 10

DATASET = "fashion-mnist-784-euclidean"


def _write_hdf5(path: str, attrs: Dict[str, Any],
                datasets: Dict[str, List[List[float]]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"attrs": attrs, "datasets": datasets}, fh)
    return path


def _truth_rows(rows: int = 4) -> List[List[float]]:
    """Ground-truth distances. Ascending within a row, as a kNN result is."""
    return [[0.1 * (i + 1) for i in range(COUNT)] for _ in range(rows)]


def _write_truth(datasets_dir: str, dataset: str = DATASET) -> str:
    return _write_hdf5(os.path.join(datasets_dir, f"{dataset}.hdf5"),
                       {"dataset": dataset}, {"distances": _truth_rows()})


def _result_attrs(ef_construction: int, **overrides: Any) -> Dict[str, Any]:
    """The attribute set the overlay and upstream actually write, trimmed to
    what this loader reads, with the archived values where there are any."""
    attrs: Dict[str, Any] = {
        "dataset": DATASET,
        "count": COUNT,
        "distance": "euclidean",
        "algo": "percona",
        "engine": "percona",
        "engine_version": "8.0.44-PerconaServer",
        "storage_engine": "InnoDB",
        "resource_pass": "tuned",
        "M": 16,
        "ef_search": 10,
        "ef_construction": ef_construction,
        "index_bytes": 249856 * 1024,
        "best_search_time": 0.0004973109960556031,
        "insert_threads": 8,
        "march": "x86-64-v3",
        "vector_index_used": True,
        "batch_mode": False,
        "run_count": 1,
        "name": f"PerconaServer(M=16, efConstruction={ef_construction})",
    }
    for key, value in overrides.items():
        if value is _ABSENT:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    return attrs


def _non_mysql_attrs(engine: str, load_seconds: Any, build_seconds: Any,
                     build_time: Any, **overrides: Any) -> Dict[str, Any]:
    """A result file from a module that does not derive from `vb_mysql`.

    The attribute set the pgvector, valkey and mongodb modules' own
    `get_additional` returns, trimmed to what this loader reads. No
    `ingest_seconds`, which is the whole point: that key exists only in
    `vb_mysql.get_additional`.
    """
    attrs: Dict[str, Any] = {
        "dataset": DATASET,
        "count": COUNT,
        "distance": "euclidean",
        "algo": engine,
        "engine": engine,
        "engine_version": "x",
        "resource_pass": "tuned",
        "ef_search": 10,
        "best_search_time": 0.0005,
        "vector_index_used": True,
        "batch_mode": False,
        "run_count": 1,
    }
    for key, value in (("build_time", build_time),
                       ("load_seconds", load_seconds),
                       ("build_seconds", build_seconds)):
        if value is not _ABSENT:
            attrs[key] = value
    for key, value in overrides.items():
        if value is _ABSENT:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    return attrs


class _Absent:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


#: Sentinel for "this attribute is not in the file at all", which is a
#: different thing from "it is in the file holding zero".
_ABSENT = _Absent()


def _write_result(annb_dir: str, attrs: Dict[str, Any],
                  filename: Optional[str] = None,
                  rows: int = 4) -> str:
    """One result file, in the layout ann-benchmarks writes.

    The stem is built from the attributes rather than hard-coded, because
    ann-benchmarks names the file after the build arguments and
    `charts.series_key` reads identity back out of it. A fixture that wrote
    `engine_InnoDB` into a pgvector filename would be describing a file that
    cannot exist.
    """
    engine = str(attrs.get("engine", "percona"))
    parts = ["euclidean"]
    for key, label in (("M", "M"), ("ef_construction", "efConstruction"),
                       ("storage_engine", "engine")):
        if attrs.get(key) is not None:
            parts += [label, str(attrs[key])]
    parts.append(str(attrs.get("ef_search", 10)))
    stem = filename or ("_".join(parts) + ".hdf5")
    path = os.path.join(annb_dir, "tuned", str(attrs.get("dataset", DATASET)),
                        str(COUNT), engine, stem)
    # Run distances inside the ground-truth threshold, so recall is 1.0 and no
    # assertion below is reading past a NaN.
    return _write_hdf5(path, attrs, {
        "distances": [[0.01 * (i + 1) for i in range(COUNT)]
                      for _ in range(rows)],
        "times": [0.0005] * rows,
    })


def _load(tmp_path, results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annb_dir = str(tmp_path / "annb")
    datasets_dir = str(tmp_path / "datasets")
    os.makedirs(annb_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    _write_truth(datasets_dir)
    for attrs in results:
        _write_result(annb_dir, attrs)
    return loaders.load_annb_results(annb_dir, datasets_dir, "e18-run")


# ---------------------------------------------------------------------------
# 1. Both timings reach the record
# ---------------------------------------------------------------------------

class TestTheTwoTimingsReachTheRecord:

    def test_the_archived_sweep_lands_as_two_fields(self, tmp_path, stub_h5py):
        """The headline, over the three archived configurations.

        `build_wall_s` from `build_time` and `ingest_wall_s` from
        `ingest_seconds`, on the one record each file produces.
        """
        records = _load(tmp_path, [
            _result_attrs(ef, build_time=build, ingest_seconds=ingest)
            for ef, build, ingest in ARCHIVED_EF_CONSTRUCTION_SWEEP])
        assert len(records) == len(ARCHIVED_EF_CONSTRUCTION_SWEEP)
        by_ef = {r["ef_construction"]: r for r in records}
        for ef, build, ingest in ARCHIVED_EF_CONSTRUCTION_SWEEP:
            assert by_ef[ef]["build_wall_s"] == build, ef
            assert by_ef[ef]["ingest_wall_s"] == ingest, ef

    def test_the_single_configuration_the_spec_names(self, tmp_path, stub_h5py):
        """`ef_construction` 800 on its own, the slowest of the three."""
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert len(records) == 1
        assert records[0]["build_wall_s"] == 283.275
        assert records[0]["ingest_wall_s"] == 283.26

    def test_the_build_cost_ratio_is_answerable_from_the_records_alone(
            self, tmp_path, stub_h5py):
        """The question the report could not answer, answered off the records.

        2.17x, not the 16x the ratio of the ef_construction values suggests
        and the published report claimed.
        """
        records = _load(tmp_path, [
            _result_attrs(ef, build_time=build, ingest_seconds=ingest)
            for ef, build, ingest in ARCHIVED_EF_CONSTRUCTION_SWEEP])
        by_ef = {r["ef_construction"]: r["build_wall_s"] for r in records}
        ratio = by_ef[800] / by_ef[50]
        assert round(ratio, 2) == 2.17
        # Where "16x" came from: it is the ratio of the knob, not of any
        # measurement. Both numbers are now in the records, so a reader does
        # not have to reach for the knob.
        assert 800 / 50 == 16.0
        assert round(ratio, 2) != 16.0

    def test_the_artifacts_precision_is_not_rounded_away(self, tmp_path,
                                                         stub_h5py):
        """`build_time` arrives as the file holds it.

        A reviewer checking a record against the artifact compares against the
        artifact's number, so the loader must not re-round it. `qps` and
        `recall_at_k` above it are rounded because they are DERIVED here; this
        is copied.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=ARCHIVED_BUILD_TIME_FULL_PRECISION,
            ingest_seconds=283.260)])
        assert records[0]["build_wall_s"] == ARCHIVED_BUILD_TIME_FULL_PRECISION

    def test_a_numpy_scalar_is_coerced_to_a_plain_float(self, tmp_path,
                                                        stub_h5py):
        """h5py hands back `numpy.float64`, which `json.dump` cannot write.

        A record field that cannot be serialised would fail at the end of a
        long run, writing `report/records.jsonl` -- the same shape of failure
        as importing the orchestrator from `report/`.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert type(records[0]["build_wall_s"]) is float
        assert type(records[0]["ingest_wall_s"]) is float
        json.dumps(records[0])

    def test_the_record_is_still_a_recall_record(self, tmp_path, stub_h5py):
        """Nothing about the phase or the recall half moves.

        The timings are additive: a `recall_qps` record that now also says what
        the build cost. If this started arriving under another phase the ops
        charts would pick it up by phase and the collateral fix below would be
        pointless.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert records[0]["phase"] == records_mod.PHASE_RECALL
        assert records[0]["recall_at_k"] == 1.0
        assert records[0]["engine"] == "percona"
        assert records[0]["extra"]["source_mtime"] > 0


# ---------------------------------------------------------------------------
# 2. Absent is absent
# ---------------------------------------------------------------------------

class TestAnAbsentAttributeIsAbsentAndNotZero:

    def test_neither_attribute_yields_neither_key(self, tmp_path, stub_h5py):
        """Not `0.0`, and not `None` under a key that was not there before.

        A zero reads as a build that cost nothing, and since
        `charts._is_build_cost_point` and the webui both select on `is not
        None`, a zero is a plotted point at the origin rather than an absent
        one. Every result file written before the overlay's `ingest_seconds`
        existed is in this case, and so is every engine whose module does not
        write it.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=_ABSENT, ingest_seconds=_ABSENT)])
        assert len(records) == 1
        assert "build_wall_s" not in records[0]
        assert "ingest_wall_s" not in records[0]

    def test_one_present_and_one_absent_is_one_key(self, tmp_path, stub_h5py):
        """Upstream's `build_time` with neither of our two load spellings.

        Described here first as "what a third-party algorithm module's result
        file looks like", which was wrong in the direction that mattered: with
        `ingest_seconds` alone it was what **pgvector, valkey and mongodb**
        look like -- three of our own seven engines, and the three the second
        clock exists for. The `load_seconds` fallback is why this is now the
        hypothetical it was described as: a module writing `build_time` and
        neither `ingest_seconds` nor `load_seconds`.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=_ABSENT)])
        assert records[0]["build_wall_s"] == 283.275
        assert "ingest_wall_s" not in records[0]

    def test_a_recorded_zero_is_kept(self, tmp_path, stub_h5py):
        """The converse, so that "absent" is not implemented as "falsy".

        A zero that the file actually holds is a measurement -- of something
        wrong, which is exactly when a reader needs to see it rather than have
        it silently dropped.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=0.0, ingest_seconds=0.0)])
        assert records[0]["build_wall_s"] == 0.0
        assert records[0]["ingest_wall_s"] == 0.0

    def test_an_unparseable_attribute_is_absent_rather_than_fatal(
            self, tmp_path, stub_h5py):
        """One unreadable attribute must not cost the whole record.

        The loader's `except Exception` is per FILE, so a raise here would
        drop a recall point as well as a build time.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time="n/a", ingest_seconds=float("nan"))])
        assert len(records) == 1
        assert records[0]["recall_at_k"] == 1.0
        assert "build_wall_s" not in records[0]
        # NaN too: `json.dump` writes a bare `NaN`, which is not JSON, and the
        # line above it already drops a NaN `qps` for this reason.
        assert "ingest_wall_s" not in records[0]


# ---------------------------------------------------------------------------
# 2b. The load is spelled twice across the seven engines
# ---------------------------------------------------------------------------

class TestTheLoadIsSpelledTwiceAcrossTheEngines:
    """`ingest_seconds` is `vb_mysql`'s; `load_seconds` is the other three's.

    The first version of this change read `ingest_seconds` only, which meant
    pgvector, valkey and mongodb got a `build_wall_s` and no `ingest_wall_s`
    at all -- and they are the three engines where the two differ by more than
    rounding, so they are the three the second field exists for.
    """

    def test_load_seconds_fills_ingest_wall_s(self, tmp_path, stub_h5py):
        """All three archived engines, from their own recorded figures."""
        records = _load(tmp_path, [
            _non_mysql_attrs(engine, load_s, build_s, build_t)
            for engine, load_s, build_s, build_t
            in ARCHIVED_LOAD_SECONDS_ENGINES])
        assert len(records) == len(ARCHIVED_LOAD_SECONDS_ENGINES)
        by_engine = {r["engine"]: r for r in records}
        for engine, load_s, _build_s, build_t in ARCHIVED_LOAD_SECONDS_ENGINES:
            assert by_engine[engine]["ingest_wall_s"] == load_s, engine
            assert by_engine[engine]["build_wall_s"] == build_t, engine

    def test_the_mongodb_row_is_why_both_fields_exist(self, tmp_path,
                                                      stub_h5py):
        """A 1.899 s load; `build_time` 30.450 s, which is 16.0x it.

        (The 28.329 s mongot wait is `build_seconds - load_seconds`, a
        different pair -- see `ARCHIVED_LOAD_SECONDS_ENGINES`. What this test
        asserts is `build_wall_s / ingest_wall_s` and their difference,
        28.551 s, which is `build_time - load_seconds`.)

        The one relationship in the archive that a single field cannot
        express, and the reason "the gap is small so one field would do" is
        wrong for this corpus of engines rather than merely unsupported.
        """
        engine, load_s, _build_s, build_t = ARCHIVED_LOAD_SECONDS_ENGINES[0]
        assert engine == "mongodb"
        records = _load(tmp_path,
                        [_non_mysql_attrs(engine, load_s, _build_s, build_t)])
        record = records[0]
        assert round(record["build_wall_s"] / record["ingest_wall_s"], 1) == 16.0
        assert round(record["build_wall_s"] - record["ingest_wall_s"], 3) == 28.551

    def test_ingest_seconds_wins_where_both_somehow_appear(self, tmp_path,
                                                           stub_h5py):
        """No module writes both; the precedence is pinned anyway.

        A fallback whose order is undefined is a fallback that changes
        behaviour the first time a module grows the other key. `ingest_seconds`
        first, because it is the one a `vb_mysql` engine writes and the one the
        archived percona figures in this file were read from.
        """
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260,
            load_seconds=99.9)])
        assert records[0]["ingest_wall_s"] == 283.26

    def test_build_seconds_is_not_read(self, tmp_path, stub_h5py):
        """`build_wall_s` comes from `build_time` even when both are present.

        mongodb's `build_seconds` is load + index-ready, pgvector's and
        valkey's exclude the load in `post` mode, and an ann record has no
        `separable_build` to tell a reader which -- so one column holding all
        three would be the ops path's own published defect, re-created on the
        ann path. `build_time` is upstream's and means the same thing for all
        seven engines.
        """
        engine, load_s, build_s, build_t = ARCHIVED_LOAD_SECONDS_ENGINES[0]
        records = _load(tmp_path,
                        [_non_mysql_attrs(engine, load_s, build_s, build_t)])
        assert records[0]["build_wall_s"] == build_t
        assert records[0]["build_wall_s"] != build_s
        assert records[0]["ingest_wall_s"] == load_s
        # Asserted as "no field is SOURCED from it", which is the invariant.
        # This test first asserted `"build_seconds" not in records[0]` -- a key
        # nothing ever writes under that name, so it held in every mutant
        # including the one that fed the attribute into a field.
        assert build_s not in records[0].values(), records[0]

    def test_build_seconds_does_not_fill_the_load_column(self, tmp_path,
                                                         stub_h5py):
        """A build with no load is where the wrong edit actually bites.

        Adding `build_seconds` to the `ingest_wall_s` tuple survives every
        other case here, because every fixture carrying `build_seconds` also
        carries `load_seconds`, which wins first -- and the tuple's shape
        invites exactly that edit. On a file with a build and no load it is a
        real defect: pgvector `post`'s load-EXCLUDED 71.544 s index build would
        land in `ingest_wall_s`, which is the three-definitions-in-one-column
        failure the declination exists to prevent, relocated to the load
        column. Declining `build_seconds` is also what KEEPS
        `build_wall_s - ingest_wall_s` well defined for every engine: both
        sources mean exactly one thing each.
        """
        records = _load(tmp_path, [_non_mysql_attrs(
            "pgvector", _ABSENT, 71.544, 82.10410451889038)])
        assert len(records) == 1
        assert records[0]["build_wall_s"] == 82.10410451889038
        assert "ingest_wall_s" not in records[0]

    def test_build_seconds_alone_populates_nothing(self, tmp_path, stub_h5py):
        """And it is not a fallback for `build_time` either."""
        records = _load(tmp_path, [_non_mysql_attrs(
            "pgvector", 10.280, 71.544, _ABSENT)])
        assert len(records) == 1
        assert "build_wall_s" not in records[0]
        assert records[0]["ingest_wall_s"] == 10.280


# ---------------------------------------------------------------------------
# 3. No rate is invented
# ---------------------------------------------------------------------------

class TestNoRateIsDerivedFromAFileThatCannotSupportOne:

    def test_the_synthetic_attributes_are_all_real(self):
        """The census is checked against the fixture, both ways.

        The version of this test that shipped first asserted only that no name
        in `ARCHIVED_RESULT_ATTRIBUTES` contained the substring "row" -- a
        statement about a literal in this module, which no change to any
        source file or any artifact could falsify. It was not a test. The
        no-rate invariant is held by
        `test_no_rate_and_no_row_count_reach_the_record` below, which drives
        the loader.

        What IS worth checking is the thing the census is for: these tests
        stand in for real artifacts, so a fixture key that no real file
        carries would make every assertion above a statement about a file that
        does not exist. Both directions, because the census being short is how
        it was wrong before.
        """
        fixture = set(_result_attrs(800, build_time=1.0,
                                   ingest_seconds=1.0))
        census = set(ARCHIVED_RESULT_ATTRIBUTES)
        assert fixture <= census, fixture - census
        # The four the fixture has no reason to carry, each for a stated
        # reason rather than as a tolerated gap:
        #   candidates, expect_extra -- upstream's, read by nothing here;
        #   index_size -- upstream's kilobyte figure, which the loader uses
        #     only as a fallback where `index_bytes` is absent, and the
        #     fixture sets `index_bytes` directly;
        #   metric -- written by our modules, but the loader takes
        #     `metric_space` from `distance`, which the fixture does set.
        assert census - fixture == {"candidates", "expect_extra",
                                    "index_size", "metric"}

    def test_no_rate_and_no_row_count_reach_the_record(self, tmp_path,
                                                       stub_h5py):
        records = _load(tmp_path, [_result_attrs(
            800, build_time=283.275, ingest_seconds=283.260)])
        assert records[0].get("ingest_rows_per_s") is None
        assert records[0].get("rows") is None


# ---------------------------------------------------------------------------
# 4. The collateral fix: the ops build-cost chart stays an ops chart
# ---------------------------------------------------------------------------

def _ops_build_record(engine: str = "pgvector", m: int = 16
                      ) -> Dict[str, Any]:
    return {"run_id": "e18-run", "phase": records_mod.PHASE_BUILD,
            "engine": engine, "dataset": DATASET, "m": m,
            "resource_pass": "tuned", "storage_engine": None,
            "build_wall_s": 37.14, "ingest_wall_s": 900.0,
            "peak_rss_bytes": 10 ** 9, "index_bytes": 10 ** 8,
            "table_bytes": 10 ** 8, "extra": {"separable_build": True}}


def _ann_recall_record(engine: str = "percona", m: int = 16
                       ) -> Dict[str, Any]:
    """What this unit's loader change now produces: a recall record with a
    build time on it."""
    return {"run_id": "e18-run", "phase": records_mod.PHASE_RECALL,
            "engine": engine, "dataset": DATASET, "m": m,
            "resource_pass": "tuned", "storage_engine": "InnoDB",
            "ef_construction": 800, "ef_search": 10,
            "recall_at_k": 0.98, "qps": 2010.8,
            "build_wall_s": 283.275, "ingest_wall_s": 283.26,
            "index_bytes": 249856 * 1024, "extra": {}}


class TestAnnBuildsStayOffTheOpsBuildCostChart:

    def test_the_predicate_takes_the_ops_build_phase_alone(self):
        """Assertion 3, through the predicate rather than the PNG.

        Deliberately not `@needs_matplotlib`: this is the invariant, and an
        invariant that can only be checked where matplotlib is installed is
        unchecked on the host the documented prerequisites produce.
        """
        ops = _ops_build_record()
        ann = _ann_recall_record()
        selected = [r for r in (ann, ops)
                    if charts._is_build_cost_point(r, DATASET)]
        assert selected == [ops]

    def test_the_phase_it_selects_is_the_schemas_build_phase(self):
        """The predicate spells the phase as a literal, because
        `report/charts.py` must not import `harness` at module scope -- the
        comment above `RECALL_CI_EXTRA_KEY` records the long run that died of
        it. This is what stops the literal drifting from the constant.
        """
        assert records_mod.PHASE_BUILD == "index_build"
        kept = _ops_build_record()
        moved = dict(kept, phase="index_build_v2")
        assert charts._is_build_cost_point(kept, DATASET)
        assert not charts._is_build_cost_point(moved, DATASET)

    def test_the_other_phases_that_carry_a_build_time_are_also_excluded(self):
        """Not just `recall_qps`.

        `ef_calibration` runs on the index the ops pass has just built and
        nothing stops a future record from carrying the build time forward.
        The predicate is an allow-list of one phase, not a deny-list of the
        phases that have gone wrong so far.
        """
        for phase in (records_mod.PHASE_RECALL, records_mod.PHASE_CALIBRATION,
                      records_mod.PHASE_INGEST, records_mod.PHASE_CONCURRENCY,
                      records_mod.PHASE_FILTERED, records_mod.PHASE_CHURN):
            record = dict(_ops_build_record(), phase=phase)
            assert not charts._is_build_cost_point(record, DATASET), phase

    def test_the_dataset_and_the_field_still_gate_it(self):
        """The two terms that were already there are still there."""
        ops = _ops_build_record()
        assert not charts._is_build_cost_point(ops, "glove-100-angular")
        assert not charts._is_build_cost_point(
            dict(ops, build_wall_s=None), DATASET)

    def test_the_selection_reaching_the_chart_is_the_build_phase_alone(
            self, tmp_path, monkeypatch):
        """The phase filter over a MIXED set, without matplotlib.

        `test_a_recall_record_alone_draws_no_build_chart` below only ever
        probes the EMPTY-selection path: an ann-only set returns at `if not
        plotted_records: return None` before the rest of the function runs. So
        a reviewer wrote the leak it cannot see -- keep the predicate's
        selection and the early return, then append the unfiltered records
        after it -- and the whole module was green on a bare host with the leak
        fully restored. The mirror case was uncovered too: a `return None` at
        the top of `build_cost`, so the chart silently never renders, was also
        green.

        `_grouped` is the one hook after the selection and before `plt`
        (`plotted_records` -> early return -> `_grouped` -> `plt.subplots`), so
        spying on it and raising captures exactly what the chart was about to
        draw and stops before matplotlib is touched. That makes this the
        positive direction as well: the selection must be non-empty AND
        correct.

        It couples to a private name in its own package, which is weaker
        coupling than the source string this replaces and stronger than none.
        """
        class Reached(Exception):
            """Carries the selection out before `plt.subplots`."""

        seen = {}

        def spy(records):
            seen["phases"] = [r.get("phase") for r in records]
            raise Reached

        monkeypatch.setattr(charts, "_grouped", spy)
        try:
            charts.build_cost([_ann_recall_record(), _ops_build_record()],
                              DATASET, str(tmp_path), "e18-grouped")
        except Reached:
            pass
        assert "phases" in seen, (
            "`_grouped` was never reached, so `build_cost` returned before the "
            "selection: with an ops build record present the chart must render")
        assert seen["phases"] == [records_mod.PHASE_BUILD], seen["phases"]

    def test_a_recall_record_alone_draws_no_build_chart(self, tmp_path):
        """The invariant, asserted on what `build_cost` RETURNS.

        This is the whole phase filter in one line, and it is deliberately not
        `@needs_matplotlib`. `build_cost` reaches `if not plotted_records:
        return None` BEFORE it touches `plt`, so on a host that has matplotlib
        a leaking selection draws a figure and returns a dict, and on a host
        that does not it raises on `plt.subplots`. Either way this assertion
        fails and the fixed tree returns `None`, so one line covers both host
        configurations.

        WHY NOT `inspect.getsource`, which this test used to do: a reviewer
        wrote a mutant that kept the predicate call and re-spelled the leak
        beside it --

            plotted_records = [r for r in records
                               if _is_build_cost_point(r, dataset)
                               or (r.get("dataset") == dataset
                                   and not r.get("build_wall_s") is None)]

        -- which satisfied every string the source assertions looked for, put
        the leak fully back on the drawn figure, and was green on a bare host.
        A spelling is not the invariant; what the function selects is.
        """
        assert charts.build_cost([_ann_recall_record()], DATASET,
                                 str(tmp_path), "e18-leak") is None

    @needs_matplotlib
    def test_the_published_subtitle_does_not_name_an_unmeasured_engine(
            self, tmp_path, monkeypatch):
        """The most quotable harm the leak would have done.

        A `recall_qps` record carries no `extra.separable_build`, so
        `generate.build_kind` returns `BUILD_KIND_UNRECORDED` for it and
        `build_cost` names it in the subtitle: "1 engine(s) left off the first
        two panels (percona): their records do not say whether the build was
        separable, so the build time cannot be placed". That is a published
        sentence about an engine that was never measured on the ops path,
        explaining its absence from panels it never belonged on. Asserted on
        the figure's own suptitle rather than on the record set, because the
        sentence is the artefact.
        """
        captured = []
        real_suptitle = charts.plt.Figure.suptitle

        def capture(self, text, *args, **kw):
            captured.append(text)
            return real_suptitle(self, text, *args, **kw)

        monkeypatch.setattr(charts.plt.Figure, "suptitle", capture)
        recs = ([_ops_build_record(m=m) for m in (8, 16, 32)]
                + [_ann_recall_record(m=m) for m in (8, 16, 32)])
        assert charts.build_cost(recs, DATASET, str(tmp_path),
                                 "e18-sub") is not None
        assert captured, "no suptitle was drawn"
        subtitle = captured[0]
        assert "percona" not in subtitle, subtitle
        assert "left off the first two panels" not in subtitle, subtitle

    @needs_matplotlib
    def test_the_drawn_chart_plots_the_ops_record_alone(self, tmp_path,
                                                        monkeypatch):
        """And `build_cost` actually uses the predicate.

        Captured off `Axes.plot`, the way
        `tests/test_t5b_stats.py::TestTheBuildChartCannotPublishTheMisleadingComparison`
        does, because a correct predicate that the chart does not call is the
        other way to fail this.
        """
        captured: List[Dict[str, Any]] = []
        real_plot = charts.plt.Axes.plot

        def capture(self, *args, **kw):
            if len(args) >= 2:
                captured.append({"ylabel": self.get_ylabel(),
                                 "label": kw.get("label"),
                                 "points": list(zip(list(args[0]),
                                                    list(args[1])))})
            return real_plot(self, *args, **kw)

        monkeypatch.setattr(charts.plt.Axes, "plot", capture)
        ops = [_ops_build_record(m=m) for m in (8, 16, 32)]
        ann = [_ann_recall_record(m=m) for m in (8, 16, 32)]
        out = charts.build_cost(ann + ops, DATASET, str(tmp_path), "e18-build")
        assert out is not None
        labels = {s["label"] for s in captured}
        assert labels and all("pgvector" in str(label) for label in labels), \
            labels
        # Both are asserted; only ONE of them is a guard, and an earlier
        # revision of this file deleted the wrong one. Measured under a
        # verified-applied unfiltered selection:
        #
        #   283.275 (the ann build time) reaches NO panel -- a recall record
        #       has no `extra.separable_build`, so `generate.build_kind`
        #       returns BUILD_KIND_UNRECORDED and both TIME panels return
        #       None for it even under the leak. This assertion is therefore
        #       vacuous today. It stays as a forward guard, for the day a
        #       record arrives carrying `separable_build` onto the ann path,
        #       and is labelled so nobody mistakes it for what catches the
        #       leak.
        #   249856 * 1024 (the ann container's index size) DOES reach the ops
        #       index-size panel, as three `percona / M=8|16|32` series. That
        #       is the live assertion: a figure measured inside the bench
        #       container, plotted on an axis of ops-harness quantities.
        plotted = {y for s in captured for _x, y in s["points"]}
        assert 283.275 not in plotted, (
            "vacuous today, kept as a forward guard in case a future record "
            "carries separable_build onto the ann path")
        assert 249856 * 1024 not in plotted, (
            "the ann container's index size reached the ops index-size panel")


# ---------------------------------------------------------------------------
# 5. The report/coercers.py extraction
# ---------------------------------------------------------------------------

class TestTheCoercersMovedWithoutLoss:
    """`report/loaders.py` was at 798 of this repo's 800-line ceiling.

    After `report/measurement_versions.py`'s precedent, whose own test says
    the extraction was taken at 793 lines "so the next addition would not
    breach it, and so the change would be reviewed here rather than inherited
    by unrelated work in the file". This change arrived with less margin than
    the condition that triggered that one, and two lines do not hold one
    comment, so the four coercers left as well.

    The ceiling tests in `tests/test_config.py` and `tests/test_t2b_report.py`
    glob `loaders.py` and `measurement_versions.py` by name and neither file
    belongs to this unit, so `report/coercers.py`'s own ceiling is asserted
    here.
    """

    def test_the_private_names_still_resolve_from_loaders(self):
        """`loaders._maybe_*` is how every call site in that file reaches
        them, and aliasing on import is what keeps those sites unchanged."""
        from report import coercers, loaders
        for private, public in (("_maybe_int", "maybe_int"),
                                ("_maybe_float", "maybe_float"),
                                ("_maybe_str", "maybe_str"),
                                ("_maybe_bool", "maybe_bool")):
            assert getattr(loaders, private) is getattr(coercers, public), \
                private

    def test_all_three_files_are_within_the_line_ceiling(self):
        for parts in (("report", "loaders.py"),
                      ("report", "measurement_versions.py"),
                      ("report", "coercers.py")):
            path = os.path.join(VB_ROOT, *parts)
            lines = len(open(path).read().splitlines())
            assert lines <= 800, f"{parts}: {lines} lines"

    def test_the_extraction_actually_bought_headroom(self):
        """The point of the move, stated as a number.

        Not `<= 800`, which the file already satisfied: the precedent was
        taken at 793 because 7 lines is not room to work in, and landing this
        one at 798 and calling it done would repeat that mistake rather than
        learn from it.

        **The file is at 779 against this 780, so there is ONE line of
        margin** -- and the change set that set this tripwire spent four of
        the five it had, on the corrected NaN comment below. That is the
        original ceiling finding one level down, recorded rather than smoothed
        over. There are still 21 lines to the repo's real 800 ceiling, so
        tripping this is a prompt and not a breach: raise it by a line or two
        with a reason, or take the next extraction, and do not quietly widen
        it to 800.

        **Raised to 781, by one line, deliberately.** The index warm-up
        (ANN_MEASUREMENT_VERSION 3) needs its three result attributes on the
        ann record so `summarize` can stop calling a first-step inversion on a
        warmed index "warm-up". The reading went to `disclosures.warmup_extra`
        beside `query_path_extra`, so what `loaders.py` pays is the one call
        line -- the file stood at 780 when that change began, not 779.
        """
        path = os.path.join(VB_ROOT, "report", "loaders.py")
        lines = len(open(path).read().splitlines())
        assert lines <= 781, (
            f"report/loaders.py is at {lines} lines against this unit's 781 "
            f"tripwire; the repo ceiling is 800 and is enforced elsewhere. "
            f"This is a prompt, not a breach: raise it deliberately with a "
            f"reason, or take the next extraction the way coercers.py was.")

    def test_the_new_module_does_not_import_the_orchestrator(self):
        """`report/` runs in a bench image that mounts `report/` and
        `harness/` only; this once crashed generation at the end of a
        twenty-hour run."""
        source = open(os.path.join(VB_ROOT, "report", "coercers.py")).read()
        # Line-wise and skipping comments, after the precedent's own form in
        # `tests/test_config.py`: this module's docstring explains the rule,
        # so a whole-file substring search fails on the explanation.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import orchestrator" not in stripped, stripped
            assert "from orchestrator" not in stripped, stripped

    def test_nan_is_the_callers_decision_and_not_the_coercers(self):
        """`maybe_float` converts a NaN; `load_annb_results` drops it.

        Split that way on purpose: a NaN is a value the file genuinely holds,
        and the two readers of this module want opposite things from it.
        Folding the filter into the coercer would make
        `test_an_unparseable_attribute_is_absent_rather_than_fatal` pass for
        the wrong reason.
        """
        from report import coercers
        value = coercers.maybe_float(float("nan"))
        assert value is not None and value != value
        assert coercers.maybe_float(None) is None
        assert coercers.maybe_float("n/a") is None
