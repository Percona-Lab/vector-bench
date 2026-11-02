"""Every summary figure says which build it came from, and the headline uses
the build the rest of the report measured.

THE INCIDENT. The published post "Benchmarking vector search in six database
engines" was written from run `full-20260825` (tuned pass, dbpedia-openai-
1000k-angular). Its QPS-at-recall-floor table gave Percona Search 138 / 124 /
88 at recall 0.90 / 0.95 / 0.99. Those three cells came from the
`quantization=none` sweep, while every other Percona Search number in the same
report -- load, build, concurrency, filtered, churn -- came from the
scalar-quantized index the ops pass builds in the tuned pass, and that index
peaks at recall 0.892: it has no figure at 0.90 at all. pgvector's and
Valkey's floor cells came from `ef_construction=400` builds while their ops
numbers were taken at 200 and at the server default. Nothing in the table said
so, because `summarize` took the frontier over every build the grid swept.

What these tests hold:

* F6: each engine declares a reference build (`runtime.reference_build`),
  resolved on the host against the profile and pass, refused when the ann grid
  does not contain it, and carried to the report through the manifest's
  `engines.<name>.presentation`. The "at the build the ops pass measured"
  table uses only those points and names the build under every number. (It
  was the headline until the whole-sweep frontier replaced it --
  `tests/test_e28_frontier_headline.py` -- which also names each cell's
  build; the assertions here now read the `reference_` fields.)
* F1: the calibration point (chosen ef, achieved recall, status) is printed
  beside every concurrency and churn throughput figure, and an engine that did
  not reach the recall target is not ranked beside those that did.
* F3: `client_cpus: auto`, refusal when the topology cannot supply it, and
  client-bound points excluded from the peak-throughput headline.
"""

from __future__ import annotations

import copy
import io
import os
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from report import frontier_render, generate, loaders, render  # noqa: E402

DATASET = "dbpedia-openai-1000k-angular"


# ---------------------------------------------------------------------------
# Fixtures: records shaped like the ones the 25 Aug run wrote
# ---------------------------------------------------------------------------

def _recall(engine: str, recall: float, qps: float, *, ef_search: int,
            m: int = 16, ef_construction: Optional[int] = None,
            quantization: Optional[str] = None, build_mode: str = "post",
            resource_pass: str = "tuned", top_level_quant: bool = True,
            ) -> Dict[str, Any]:
    """One `recall_qps` record with the source file ann-benchmarks names.

    The stem is the real naming scheme (`angular_M_16_quantization_none_120`),
    because `build_signature` reads the build out of it.
    """
    parts = ["angular", "M", str(m)]
    if quantization is not None:
        parts += ["quantization", quantization]
    if ef_construction is not None:
        parts += ["build_mode", build_mode, "efConstruction",
                  str(ef_construction)]
    stem = "_".join(parts + [str(ef_search)])
    record = {
        "phase": "recall_qps", "engine": engine, "dataset": DATASET,
        "resource_pass": resource_pass, "recall_at_k": recall, "qps": qps,
        "m": m, "ef_search": ef_search, "ef_construction": ef_construction,
        "build_mode": build_mode, "clients": 1, "k": 10,
        "extra": {"source_file":
                  f"{DATASET}/10/{engine}/{stem}.hdf5",
                  "parameters": (f"PerconaSearch(quantization={quantization}, "
                                 f"numCandidates={ef_search})"
                                 if quantization else None)},
    }
    if quantization is not None and top_level_quant:
        record["quantization"] = quantization
    return record


def _mongodb_records(top_level_quant: bool = True) -> List[Dict[str, Any]]:
    """Percona Search on the 25 Aug run, abridged to the points that matter.

    The scalar index peaks at 0.892; the unquantized one reaches 0.997. 138,
    124 and 88 are the three figures the post published.
    """
    scalar = [(0.71, 260.0, 10), (0.83, 220.0, 40), (0.871, 190.0, 120),
              (0.892, 150.0, 400)]
    none = [(0.939, 138.0, 40), (0.962, 124.0, 120), (0.997, 88.0, 800)]
    return ([_recall("mongodb", r, q, ef_search=ef, quantization="scalar",
                     top_level_quant=top_level_quant)
             for r, q, ef in scalar]
            + [_recall("mongodb", r, q, ef_search=ef, quantization="none",
                       top_level_quant=top_level_quant)
               for r, q, ef in none])


def _pgvector_records() -> List[Dict[str, Any]]:
    """efC=200 is what the ops pass builds; efC=400 is faster at 0.95."""
    rows = []
    for efc, points in ((200, [(0.90, 700.0, 40), (0.951, 520.0, 80),
                               (0.985, 300.0, 200)]),
                        (400, [(0.93, 690.0, 40), (0.968, 610.0, 80),
                               (0.993, 350.0, 200)])):
        rows += [_recall("pgvector", r, q, ef_search=ef, ef_construction=efc)
                 for r, q, ef in points]
    return rows


def _valkey_records() -> List[Dict[str, Any]]:
    rows = []
    for efc, points in ((64, [(0.80, 900.0, 10)]),
                        (200, [(0.91, 800.0, 40), (0.97, 600.0, 120)]),
                        (400, [(0.95, 850.0, 40), (0.99, 640.0, 120)])):
        rows += [_recall("valkey", r, q, ef_search=ef, ef_construction=efc)
                 for r, q, ef in points]
    return rows


def _manifest(reference_builds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A manifest carrying each engine's per-pass reference build, the way
    `cmd_run` records it in `engines.<name>.presentation`."""
    return {
        "run_id": "full-20260825",
        "config": {"resource_pass": "tuned", "profile": {"name": "t"}},
        "engines": {
            engine: {"presentation": {"reference_builds": {"tuned": build}}}
            for engine, build in reference_builds.items()},
        "phases": [],
    }


REFERENCE = {"mongodb": {"quantization": "scalar"},
             "pgvector": {"m": 16, "ef_construction": 200},
             "valkey": {"m": 16, "ef_construction": 200}}


def _entry(summary, engine, resource_pass="tuned"):
    return summary["per_dataset"][DATASET][resource_pass][engine]


# ---------------------------------------------------------------------------
# F6.2 / F6.5: the ops-build view uses the reference build only.
#
# This was the headline until the whole-sweep frontier replaced it
# (tests/test_e28_frontier_headline.py). The reference figures moved to the
# `reference_` fields and the secondary table, and every assertion below is
# the same claim about them that it was about the headline.
# ---------------------------------------------------------------------------

class TestTheHeadlineUsesTheReferenceBuild:

    def _summary(self, records=None, manifest=None):
        records = records if records is not None else (
            _mongodb_records() + _pgvector_records() + _valkey_records())
        return generate.summarize(records, manifest or _manifest(REFERENCE))

    def test_percona_search_has_no_headline_figure_at_any_floor(self):
        """Its scalar index peaks at 0.892, so 0.90, 0.95 and 0.99 are all
        unreached -- never 138 / 124 / 88 from the unquantized build."""
        entry = _entry(self._summary(), "mongodb")
        for floor in (90, 95, 99):
            assert entry[f"reference_qps_at_recall_{floor}"] is None, floor
        assert entry["reference_max_recall"] == pytest.approx(0.892)

    def test_the_best_of_the_sweep_keeps_the_published_figures(self):
        """They are real measurements -- of another build, and labelled so."""
        entry = _entry(self._summary(), "mongodb")
        assert entry["best_qps_at_recall_90"] == 138.0
        assert entry["best_qps_at_recall_95"] == 124.0
        assert entry["best_qps_at_recall_99"] == 88.0
        assert entry["best_qps_at_recall_90_build_label"] == "quantization=none"

    def test_pgvector_headline_comes_from_ef_construction_200(self):
        entry = _entry(self._summary(), "pgvector")
        assert entry["reference_qps_at_recall_95"] == 520.0
        assert entry["reference_qps_at_recall_95_build_label"] == "M=16 efC=200"
        assert entry["best_qps_at_recall_95"] == 610.0
        assert entry["best_qps_at_recall_95_build_label"] == "M=16 efC=400"

    def test_the_build_signature_is_stored_beside_the_storage(self):
        """F6.3: `qps_at_recall_N_build` is `build_signature` of the point."""
        records = _pgvector_records()
        entry = _entry(self._summary(records, _manifest(REFERENCE)),
                       "pgvector")
        winner = next(r for r in records if r["qps"] == 520.0)
        assert entry["reference_qps_at_recall_95_build"] == loaders.build_signature(winner)
        assert "qps_at_recall_95_storage" in entry

    def test_points_and_range_describe_the_reference_build(self):
        entry = _entry(self._summary(), "mongodb")
        assert entry["reference_points"] == 4
        assert entry["reference_max_recall"] == pytest.approx(0.892)
        assert entry["sweep_points"] == 7
        assert entry["sweep_max_recall"] == pytest.approx(0.997)

    def test_an_archived_record_without_the_top_level_key_still_matches(self):
        """`records.jsonl` written before the loader copied `quantization`
        carries it only in `extra.parameters` and the source file name."""
        entry = _entry(self._summary(_mongodb_records(top_level_quant=False)),
                       "mongodb")
        assert entry["reference_qps_at_recall_90"] is None
        assert entry["reference_max_recall"] == pytest.approx(0.892)


class TestAnArchivedManifestFallsBackToTheOpsBuild:
    """The 25 Aug manifest predates `reference_builds`. The ops pass's own
    `index_build` record says what it built, so that is the reference -- and
    an axis it left to the server default names no ann point at all."""

    def _ops_build(self, engine, ef_construction=None, quantization=None,
                   tunable=False):
        extra = {"ef_construction_tunable": tunable}
        if quantization:
            extra["quantization"] = quantization
        return {"phase": "index_build", "engine": engine, "dataset": DATASET,
                "resource_pass": "tuned", "m": 16, "build_mode": "post",
                "ef_construction": ef_construction, "extra": extra}

    def _summary(self):
        records = (_mongodb_records(top_level_quant=False)
                   + _pgvector_records() + _valkey_records()
                   + [self._ops_build("mongodb", quantization="scalar"),
                      self._ops_build("pgvector", ef_construction=200,
                                      tunable=True),
                      self._ops_build("valkey", tunable=True)])
        manifest = {"run_id": "full-20260825", "engines": {}, "phases": [],
                    "config": {"resource_pass": "tuned"}}
        return generate.summarize(records, manifest)

    def test_mongodb_falls_back_to_the_scalar_index(self):
        entry = _entry(self._summary(), "mongodb")
        assert entry["reference_qps_at_recall_90"] is None
        assert entry["reference_source"] == "ops index_build record"

    def test_pgvector_falls_back_to_ef_construction_200(self):
        assert _entry(self._summary(), "pgvector")["reference_qps_at_recall_95"] == 520.0

    def test_valkey_at_the_server_default_matches_no_ann_build(self):
        """The ops pass passed no EF_CONSTRUCTION, and the record cannot say
        what the server defaulted to. A number from 64, 200 or 400 would be a
        guess presented as the reference."""
        entry = _entry(self._summary(), "valkey")
        assert entry["reference_qps_at_recall_90"] is None
        assert entry["best_qps_at_recall_90"] == 850.0


class TestOneBuildNeedsNoDeclaration:
    """A profile that sweeps one build per engine has nothing to choose
    between, so a report without a reference keeps its numbers."""

    def test_a_single_build_is_its_own_reference(self):
        records = [_recall("mariadb", 0.95, 400.0, ef_search=10),
                   _recall("mariadb", 0.99, 200.0, ef_search=40)]
        entry = _entry(generate.summarize(records, {}), "mariadb")
        assert entry["reference_qps_at_recall_90"] == 400.0
        assert entry["reference_source"] == "only build measured"

    def test_two_builds_and_no_declaration_are_not_guessed_between(self):
        records = [_recall("mariadb", 0.95, 400.0, ef_search=10, m=8),
                   _recall("mariadb", 0.97, 300.0, ef_search=10, m=16)]
        entry = _entry(generate.summarize(records, {}), "mariadb")
        assert entry["reference_qps_at_recall_90"] is None
        assert entry["reference_build"] is None
        assert entry["best_qps_at_recall_90"] == 400.0


# ---------------------------------------------------------------------------
# F6.2 / F6.3 / F6.5: what the page says
# ---------------------------------------------------------------------------

class TestTheRenderedTables:

    def _page(self):
        summary = generate.summarize(
            _mongodb_records() + _pgvector_records(), _manifest(REFERENCE))
        return render._headline_tables(summary)

    def _row(self, table: str, label: str, after: str = "") -> List[str]:
        text = table.split(after, 1)[1] if after else table
        line = next(l for l in text.splitlines()
                    if l.startswith(f"| {label} "))
        return [c.strip() for c in line.strip("|").split("|")]

    def test_the_ops_build_row_is_dashes_for_percona_search(self):
        row = self._row(self._page(), render._label("mongodb"),
                        after=frontier_render.OPS_TABLE_HEADING)
        assert row[1:4] == ["—", "—", "—"]

    def test_the_footnote_gives_the_reference_builds_max_recall(self):
        table = self._page()
        ops_view = table.split(frontier_render.OPS_TABLE_HEADING)[1]
        assert "0.892" in ops_view.split("| ---")[-1]
        assert "quantization=scalar" in ops_view

    def test_each_ops_build_number_names_its_build(self):
        row = self._row(self._page(), render._label("pgvector"),
                        after=frontier_render.OPS_TABLE_HEADING)
        assert "520 @0.951" in row[2] and "M=16 efC=200" in row[2]

    def test_the_headline_names_the_build_each_cell_came_from(self):
        table = self._page()
        row = self._row(table, render._label("mongodb"))
        assert "138" in row[1] and "quantization=none" in row[1]
        row = self._row(table, render._label("pgvector"))
        assert "610" in row[2] and "M=16 efC=400" in row[2]

    def test_the_build_line_survives_into_html(self):
        html = render._md_tables_to_html(self._page())
        assert "<br>" in html
        assert "&lt;br&gt;" not in html


# ---------------------------------------------------------------------------
# F6.4: the loader copies the swept attribute
# ---------------------------------------------------------------------------

class TestTheLoaderCopiesQuantization:

    def test_quantization_reaches_the_record(self, tmp_path, monkeypatch):
        from tests import test_e18_ann_build_timing_reaches_the_records as e18
        monkeypatch.setitem(sys.modules, "h5py",
                            __import__("types").SimpleNamespace(
                                File=e18._StubH5File))
        annb = tmp_path / "annb"
        datasets = tmp_path / "datasets"
        datasets.mkdir()
        e18._write_truth(str(datasets))
        attrs = e18._non_mysql_attrs("mongodb", 1.0, 2.0, 3.0,
                                     M=16, quantization="scalar",
                                     m_applied=False)
        e18._write_result(str(annb), attrs,
                          filename="euclidean_M_16_quantization_scalar_10.hdf5")
        [record] = loaders.load_annb_results(str(annb), str(datasets), "r")
        assert record["quantization"] == "scalar"
        assert loaders.build_params(record)["quantization"] == "scalar"

    def test_build_params_reads_an_archived_record(self):
        record = _recall("mongodb", 0.9, 1.0, ef_search=10,
                         quantization="none", top_level_quant=False)
        assert loaders.build_params(record) == {
            "m": 16, "ef_construction": None, "quantization": "none"}


# ---------------------------------------------------------------------------
# F1: the calibration point travels with every throughput figure
# ---------------------------------------------------------------------------

def _unit(engine: str, **kw) -> Dict[str, Any]:
    base = {"engine": engine, "dataset": DATASET, "resource_pass": "tuned",
            "m": 16, "build_mode": "post", "storage_engine": "InnoDB",
            "ef_construction": None}
    base.update(kw)
    return base


def _calibration(engine: str, ef: int, recall: float) -> Dict[str, Any]:
    return {**_unit(engine), "phase": "ef_calibration", "ef_search": ef,
            "recall_at_k": recall, "clients": 1, "qps": 100.0,
            "extra": {"grid_size": 3}}


def _concurrency(engine: str, clients: int, qps: float, *, ef: int,
                 status: Optional[str] = "ok", client_bound: bool = False,
                 ef_mode: str = "calibrated") -> Dict[str, Any]:
    extra: Dict[str, Any] = {"ef_mode": ef_mode, "client_bound": client_bound,
                             "scaling_efficiency": 0.9}
    if status:
        extra.update({"calibration_status": status, "target_recall": 0.95})
    return {**_unit(engine), "phase": "concurrency", "clients": clients,
            "qps": qps, "ef_search": ef, "recall_at_k": 0.9,
            "extra": extra}


def _churn(engine: str, *, ef: int, status: str = "ok") -> Dict[str, Any]:
    return {**_unit(engine), "phase": "churn", "churn_fraction": 0.1,
            "clients": 1, "qps": 500.0, "ef_search": ef, "recall_at_k": 0.94,
            "extra": {"ef_mode": "calibrated", "calibration_status": status,
                      "target_recall": 0.95}}


def _calibrated_run() -> List[Dict[str, Any]]:
    return [
        _calibration("pgvector", 40, 0.93), _calibration("pgvector", 80, 0.9561),
        _concurrency("pgvector", 1, 900.0, ef=80),
        _concurrency("pgvector", 16, 9000.0, ef=80),
        # Faster, and client-bound: the clients were the ceiling.
        _concurrency("pgvector", 32, 12000.0, ef=80, client_bound=True),
        _churn("pgvector", ef=80),
        # Never got to 0.95: its throughput is at a lower recall.
        _calibration("mariadb", 800, 0.9012),
        _concurrency("mariadb", 1, 5000.0, ef=800, status="target_unreached"),
        _concurrency("mariadb", 16, 40000.0, ef=800,
                     status="target_unreached"),
    ]


class TestTheConcurrencyTableCarriesTheOperatingPoint:

    def _table(self, records=None):
        summary = generate.summarize(records or _calibrated_run(), {})
        return render._concurrency_table(summary)

    def test_headers_name_the_ef_and_the_recall_it_bought(self):
        header = next(l for l in self._table().splitlines()
                      if l.startswith("| Engine | Dataset | Pass"))
        assert "Calibrated ef" in header
        assert "Calibration recall" in header

    def test_each_row_carries_its_units_calibration_point(self):
        rows = [l for l in self._table().splitlines()
                if l.startswith(f"| {render._label('pgvector')} | {DATASET}")]
        assert rows and all("| 80 |" in r and "0.9561" in r for r in rows)

    def test_an_unreached_target_is_shown_in_the_row(self):
        rows = [l for l in self._table().splitlines()
                if l.startswith(f"| {render._label('mariadb')} | {DATASET}")]
        assert rows and all("target_unreached" in r for r in rows)


class TestThePeakThroughputHeadline:

    def _peaks(self):
        return generate.summarize(_calibrated_run(), {})["concurrency_peaks"]

    def test_a_client_bound_point_is_not_the_peak(self):
        """12,000 at 32 clients was the clients' ceiling, not the engine's."""
        [pg] = [p for p in self._peaks() if p["engine"] == "pgvector"]
        assert pg["peak_qps"] == 9000.0
        assert pg["peak_clients"] == 16
        assert pg["client_bound_excluded"] == 1

    def test_an_engine_that_missed_the_target_is_not_ranked(self):
        """40,000 QPS at recall 0.90 is not a figure to rank beside 9,000 at
        0.956 -- it is a cheaper query."""
        [maria] = [p for p in self._peaks() if p["engine"] == "mariadb"]
        assert maria["ranked"] is False
        assert maria["peak_qps"] is None
        assert maria["status"] == "target_unreached"

    def test_the_rendered_headline_shows_the_status_in_place_of_a_number(self):
        table = self._table_text()
        line = next(l for l in table.splitlines()
                    if l.startswith(f"| {render._label('mariadb')} |"))
        assert "target_unreached" in line
        assert "40,000" not in line

    def test_ranked_engines_come_first(self):
        table = self._table_text()
        names = [l.split("|")[1].strip() for l in table.splitlines()
                 if l.startswith("| ") and "---" not in l
                 and not l.startswith("| Rank")]
        assert names.index(render._label("pgvector")) < \
            names.index(render._label("mariadb"))

    def _table_text(self):
        summary = generate.summarize(_calibrated_run(), {})
        return render._peak_throughput_table(summary)


class TestTheChurnTableCarriesTheOperatingPoint:

    def test_churn_rows_name_the_ef_and_recall(self):
        summary = generate.summarize(_calibrated_run(), {})
        table = render._churn_table(summary)
        header = next(l for l in table.splitlines()
                      if l.startswith("| Engine | Dataset | Pass"))
        assert "Calibrated ef" in header and "Calibration recall" in header
        row = next(l for l in table.splitlines()
                   if l.startswith(f"| {render._label('pgvector')} |"))
        assert "0.9561" in row


class TestFilteredCaption:

    def test_the_caption_says_the_ef_is_the_unfiltered_one(self):
        record = {**_unit("pgvector"), "phase": "filtered", "selectivity": 0.1,
                  "qps": 10.0, "recall_at_k": 0.9, "ef_search": 80,
                  "extra": {}}
        table = render._filtered_table({"filtered": [record]})
        assert "reuses the unfiltered calibrated ef" in table


class TestAFixedEfRunSaysSoInValidity:

    def test_the_validity_entry_names_the_ef(self):
        records = [_concurrency("pgvector", 1, 900.0, ef=100, status=None,
                                ef_mode="fixed"),
                   _concurrency("mariadb", 1, 5000.0, ef=100, status=None,
                                ef_mode="fixed")]
        summary = generate.summarize(records, {})
        text = render._validity_section({}, summary, records)
        assert "ef_search = 100" in text
        assert "not compared at equal recall" in text

    def test_a_calibrated_run_has_no_such_entry(self):
        records = _calibrated_run()
        summary = generate.summarize(records, {})
        text = render._validity_section({}, summary, records)
        assert "not compared at equal recall" not in text


# ---------------------------------------------------------------------------
# Review round 1
# ---------------------------------------------------------------------------

def _pre_v4(engine: str, clients: int, qps: float, *, phase="concurrency",
            version: Optional[int] = None) -> Dict[str, Any]:
    """An ops record from before `ef_mode` existed: fixed ef 100, no key.

    The 25 Aug run's records are this shape -- no `ef_mode`, no
    `ops_measurement_version`, so implicit version 1.
    """
    extra: Dict[str, Any] = {}
    if version is not None:
        extra["ops_measurement_version"] = version
    record = {**_unit(engine), "phase": phase, "clients": clients,
              "qps": qps, "ef_search": 100, "recall_at_k": 0.9,
              "extra": extra}
    if phase == "churn":
        record["churn_fraction"] = 0.1
    return record


class TestARecordFromBeforeEfModeWasFixedEf:
    """Version 4 is the harness that started calibrating and stamping
    `extra.ef_mode` (`harness/metrics/records.py`, "VERSION 4 chose ef_search
    instead of declaring it"). Everything before it ran at the profile's
    declared ef_search. A record with no `ef_mode` and a version below 4 is
    therefore fixed-ef, and the report must say so -- the 25 Aug run is one."""

    def test_the_predicate(self):
        from report.operating import is_fixed_ef
        assert is_fixed_ef({"phase": "concurrency", "extra": {}})
        assert is_fixed_ef({"phase": "concurrency",
                            "extra": {"ops_measurement_version": 3}})
        assert is_fixed_ef({"phase": "concurrency",
                            "extra": {"ef_mode": "fixed",
                                      "ops_measurement_version": 8}})
        assert not is_fixed_ef({"phase": "concurrency",
                                "extra": {"ops_measurement_version": 4}})
        assert not is_fixed_ef({"phase": "concurrency",
                                "extra": {"ef_mode": "calibrated"}})

    def test_validity_names_the_ef_for_records_with_no_key(self):
        records = [_pre_v4("pgvector", 1, 900.0), _pre_v4("mariadb", 1, 50.0)]
        text = render._fixed_ef_section(records)
        assert "ef_search = 100" in text
        assert "not compared at equal recall" in text

    def test_validity_names_the_ef_for_an_explicit_old_version(self):
        records = [_pre_v4("pgvector", 1, 900.0, version=3)]
        assert "ef_search = 100" in render._fixed_ef_section(records)

    def test_a_version_4_record_without_the_key_is_not_assumed_fixed(self):
        records = [_pre_v4("pgvector", 1, 900.0, version=4)]
        assert render._fixed_ef_section(records) == ""

    def test_the_concurrency_row_says_fixed_not_calibrated(self):
        summary = generate.summarize([_pre_v4("pgvector", 1, 900.0)], {})
        row = next(l for l in render._concurrency_table(summary).splitlines()
                   if l.startswith(f"| {render._label('pgvector')} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[-2] == "100 (fixed)"
        assert cells[-1] == "not calibrated"

    def test_the_churn_row_says_fixed_not_calibrated(self):
        summary = generate.summarize(
            [_pre_v4("pgvector", 1, 900.0, phase="churn")], {})
        row = next(l for l in render._churn_table(summary).splitlines()
                   if l.startswith(f"| {render._label('pgvector')} |"))
        assert "100 (fixed)" in row and "not calibrated" in row

    def test_the_peak_table_does_not_call_a_fixed_ef_calibrated(self):
        summary = generate.summarize([_pre_v4("pgvector", 1, 900.0)], {})
        line = next(l for l in render._peak_throughput_table(summary)
                    .splitlines()
                    if l.startswith(f"| {render._label('pgvector')} |"))
        assert "100 (fixed)" in line and "not calibrated" in line


class TestAFallbackThatPoolsBuildsIsAmbiguous:
    """An archived Percona Search `index_build` record with no quantization
    falls back to `{m: 16}`, which matches the scalar AND the unquantized
    sweep -- the 25 Aug defect again, under a label that says "ops record".
    A fallback that matches more than one build is refused."""

    def _ops_build(self, engine, **extra):
        return {"phase": "index_build", "engine": engine, "dataset": DATASET,
                "resource_pass": "tuned", "m": 16, "build_mode": "post",
                "ef_construction": None, "extra": dict(extra)}

    def test_mongodb_without_quantization_reads_as_a_dash(self):
        records = _mongodb_records() + [self._ops_build("mongodb")]
        entry = _entry(generate.summarize(records, {}), "mongodb")
        for floor in (90, 95, 99):
            assert entry[f"reference_qps_at_recall_{floor}"] is None
        assert entry["reference_build"] is None
        assert entry["reference_source"].startswith("ambiguous")
        assert entry["best_qps_at_recall_90"] == 138.0

    def test_an_m_sweep_with_no_m_on_the_ops_record_is_ambiguous(self):
        records = [_recall("mariadb", 0.95, 400.0, ef_search=10, m=8),
                   _recall("mariadb", 0.97, 300.0, ef_search=10, m=16)]
        ops = self._ops_build("mariadb")
        ops["m"] = None
        entry = _entry(generate.summarize(records + [ops], {}), "mariadb")
        assert entry["reference_qps_at_recall_90"] is None
        assert entry["reference_source"].startswith("ambiguous")

    def test_the_footnote_says_the_build_is_ambiguous(self):
        records = _mongodb_records() + [self._ops_build("mongodb")]
        table = render._headline_tables(generate.summarize(records, {}))
        ops_view = table.split(frontier_render.OPS_TABLE_HEADING)[1]
        assert "ambiguous" in ops_view

    def test_storage_engines_alone_do_not_make_it_ambiguous(self):
        """InnoDB and MyISAM at one M are one reference build measured in two
        storage engines; §4 names the storage that won each cell."""
        a = _recall("mariadb", 0.95, 400.0, ef_search=10)
        b = _recall("mariadb", 0.97, 300.0, ef_search=10)
        a["storage_engine"], b["storage_engine"] = "InnoDB", "MyISAM"
        b["extra"]["source_file"] = b["extra"]["source_file"].replace(
            "_M_16_", "_M_16_engine_MyISAM_")
        records = [a, b, self._ops_build("mariadb")]
        entry = _entry(generate.summarize(records, {}), "mariadb")
        assert entry["reference_qps_at_recall_90"] == 400.0

    def test_the_ops_build_mode_narrows_a_pgvector_fallback(self):
        """#77 review: `_build_shape` counted build_mode and `matches()`
        ignored it, so an archived pgvector run with post AND incremental
        curves at one M was "ambiguous" although its ops pass built one.

        Honest: the ops record's `post` picks the post curve (400 q/s, not
        the incremental 300). Mutant (the ops pass names both modes): still
        ambiguous, because then there really are two builds."""
        post = _recall("pgvector", 0.95, 400.0, ef_search=40,
                       ef_construction=200, build_mode="post")
        inc = _recall("pgvector", 0.95, 300.0, ef_search=40,
                      ef_construction=200, build_mode="incremental")
        ops = self._ops_build("pgvector", ef_construction_tunable=True)
        ops["ef_construction"] = 200
        entry = _entry(generate.summarize([post, inc, ops], {}), "pgvector")
        assert entry["reference_qps_at_recall_90"] == 400.0, entry
        assert not str(entry["reference_source"]).startswith("ambiguous")

        both = dict(ops, build_mode="incremental")
        entry = _entry(generate.summarize([post, inc, ops, both], {}),
                       "pgvector")
        assert entry["reference_qps_at_recall_90"] is None
        assert entry["reference_source"].startswith("ambiguous")

    def test_an_ann_point_without_a_build_mode_still_matches(self):
        """The ann overlay declares build_mode for pgvector only; a mariadb
        point without one must not be excluded by an ops record that says
        `post`."""
        from report import builds
        point = _recall("mariadb", 0.95, 400.0, ef_search=10)
        point.pop("build_mode")
        assert builds.matches(point, {"m": 16, "build_mode": "post"})
        assert not builds.matches(dict(point, build_mode="incremental"),
                                  {"m": 16, "build_mode": "post"})


class TestTheManifestCarriesTheReferenceBuilds:
    """`cmd_run` hands `presentation.reference_builds` to `Manifest.set_engine`
    and the report reads it back from `run-manifest.json`. Written and read
    through the real API, so a later change to the manifest's engine entry
    that drops the key fails here instead of every headline going dark."""

    def test_round_trip(self, tmp_path):
        import json
        from orchestrator import engines as engines_mod
        from orchestrator.manifest import MANIFEST_NAME, Manifest
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        builds = {"tuned": {"quantization": "scalar"},
                  "normalized": {"quantization": "none"}}
        manifest = Manifest(str(run_dir), "rt-1")
        manifest.set_engine(
            "mongodb", str(tmp_path / "sources"), "rt", "bench",
            {"runtime": "x", "bench": "y"},
            presentation={**engines_mod.presentation("mongodb"),
                          "reference_builds": builds})
        manifest.save()
        loaded = json.load(open(run_dir / MANIFEST_NAME))
        assert loaded["engines"]["mongodb"]["presentation"][
            "reference_builds"] == builds
        summary = generate.summarize(_mongodb_records(), loaded)
        entry = _entry(summary, "mongodb")
        assert entry["reference_source"] == "manifest"
        assert entry["reference_build"] == {"quantization": "scalar"}
