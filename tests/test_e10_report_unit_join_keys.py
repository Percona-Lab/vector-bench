"""E10 -- three tables in `report/render.py` printed a number from a different
ops unit, because their join key and their header named fewer axes than the
ops phase loops.

WHAT WAS OBSERVED, and the whole reason this file exists: the run bundle
`full-20261015-215006` -- profile `full`, engine `percona`, 4 datasets x
M 8/16/32 x both resource passes. None of these three defects can show on a
narrower grid than that, which is why they survived to be published: every
one of them needs two values of one axis inside a single ops unit. Its
`report/report.md` contained, in section 5 -- four of its eleven rows, with
the engine, dataset and storage columns elided:

    | ... | M | ... | Build time  | Ingest rate |
    | ... | 16 | ... |     167.1 s |   598 rows/s |   <- 60,000 rows / 167.1 s = 359
    | ... | 32 | ... |     370.3 s |   545 rows/s |   <- 60,000 rows / 370.3 s = 162
    | ... | 32 | ... |     356.9 s |   598 rows/s |   <- 60,000 rows / 356.9 s = 168
    | ... | 16 | ... |   2,344.5 s |   206 rows/s |   <- 1,000,000 / 2,344.5 = 427

Four of eleven rows. Each one is arithmetically self-refuting: for an
incremental build the build wall clock IS the corpus load, so the row's own
two numbers imply a third rate that neither of them is. The honest rates are
in `report/records.jsonl` and are 359.12, 162.01, 168.13 and 426.53 -- the
rates those rows' own arithmetic implies, to the digit the table prints.

THE CAUSE was `_build_table`'s `ingest_by_key`, keyed on
(engine, dataset, resource_pass, build_mode, storage_engine) with `m` absent.
Three ingest records differing only in M collapsed into one dict entry and
the last one written won. The comment above that dict already recorded
adding `storage_engine` for the identical reason one axis over -- MariaDB's
InnoDB and MyISAM builds collided and one rate was printed for both rows --
so this is the same defect, found again because a run finally varied the axis.

THE SECOND HALF of the same defect is a header rather than a key. Section 5
had no `Pass` column, section 7 (concurrency) had neither `Pass` nor `M`, and
section 9's first churn table had neither, while section 8 (filtered) and the
churn-under-writes table below section 9 already carried both. In the same
run that cost:

  * section 5: fashion-mnist M=8 printed twice, 110.1 s at 6.5 GiB peak RSS
    and 100.3 s at 44.4 GiB, with nothing saying which machine either ran on;
  * section 7: five fashion-mnist rows at one client reading 1,677 / 2,037 /
    1,952 / 1,673 / 1,789 QPS -- normalized M=32, normalized M=8, tuned M=16,
    tuned M=32, tuned M=8, in that order -- rendered as five identical
    identities;
  * section 9: every (dataset, fraction, repair state) five times over, e.g.
    at 10% before repair recalls 0.9865 / 0.9594 / 0.9672 / 0.9860 / 0.9598
    beside QPS 1,503 / 1,898 / 1,561 / 1,331 / 1,829, which reads as one
    noisy measurement rather than five readings of five indexes.

The pass is the damaging half of that. `config/resources/normalized.yml`
grants the server 8 CPUs and a 16 GB limit; the tuned pass in that run gave
it 76 cores and 112.5 GiB, and section 1 names a single pass for the whole
document -- so an 8-core row read as a 76-core one.

PROBE STYLE, per this suite's convention. Every decision test names its
HONEST and its MUTANT value in its docstring, and these are repairs rather
than additions, so each one also exercises the PRE-CHANGE form over the very
same records. There are two of those, because the defect has two halves:

  * the pre-change JOIN is copied verbatim into `_old_ingest_by_key` and run
    against the same ingest records, so "the old key published 598.41 three
    times" is a value this file measures and not a claim it makes;
  * the pre-change HEADERS are column-name lists (`OLD_IDENTITY`) read back
    out of the table the FIXED code renders, so "those columns carry no
    information that tells these rows apart" is measured on the page rather
    than on a tuple hand-copied out of the old source, which could drift.

A repair whose old form also passes is not a repair, and an identical pair is
a dead probe. Two tests here are deliberately non-regression guards rather
than probes and say so in their own docstrings; the rest fail against the
pre-change renderer. Nothing on disk is patched.

Pure logic. No engine, no container, no benchmark run; the records below are
dicts carrying the figures the bundle recorded.
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report import render as render_mod                               # noqa: E402

FASHION = "fashion-mnist-784-euclidean"
SIFT = "sift-128-euclidean"

#: Rows in the two datasets the four wrong rows came from. The divisor in
#: every "rate implied by the same row" figure in this file's header.
ROWS = {FASHION: 60_000, SIFT: 1_000_000}


def _cells(table: str) -> List[List[str]]:
    """A rendered Markdown table as rows of cells, separator row dropped.

    Cells and not a substring search, for the reason `tests/test_config.py`'s
    copy of this helper gives: "598" turns up in several columns of these
    tables and in the raw records, so a membership test against the whole
    fragment is satisfied by the number appearing somewhere else entirely --
    which is exactly the failure mode this file is about.
    """
    rows = []
    for line in table.strip().splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c and set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _column(table: str, name: str) -> List[str]:
    """One column's body cells, by header name.

    By name rather than by index so these guards survive another column being
    inserted, and so a table that does not have the column fails loudly here
    rather than silently reading its neighbour.
    """
    rows = _cells(table)
    header, body = rows[0], rows[1:]
    assert name in header, (name, header)
    return [row[header.index(name)] for row in body]


def _identities(table: str, names: List[str]) -> List[tuple]:
    """Each body row reduced to the named columns.

    Used to read the PRE-CHANGE header's identity columns out of the table
    the fixed code renders. Reading them from the rendered text rather than
    from a tuple hand-copied out of the old source is the stronger form: a
    hand-copied tuple can drift from what the renderer actually did, and the
    claim being made here is about the page.
    """
    rows = _cells(table)
    header, body = rows[0], rows[1:]
    for name in names:
        assert name in header, (name, header)
    return [tuple(row[header.index(name)] for name in names) for row in body]


def _md_table_headers() -> List[tuple]:
    """Every `_md_table` call in `report/render.py` whose header is literal.

    Returns (lineno, [column names]). An AST scan rather than a regex, and
    over the file rather than over rendered output, because the question is
    "which tables exist" -- a table this test file forgot to render is
    exactly the one a hand-written list omits, and that has happened twice in
    this series. A header built at run time rather than spelled as a literal
    list is skipped and would need its own guard; there are none today, which
    the count pinned in `OPS_UNIT_TABLE_COUNT` is what notices.
    """
    source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "report", "render.py")
    out = []
    for node in ast.walk(ast.parse(open(source).read())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr",
                                                         None)
        if name != "_md_table" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, (ast.List, ast.Tuple)):
            continue
        if not all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                   for e in first.elts):
            continue
        out.append((first.lineno, [e.value for e in first.elts]))
    return out


def _strategy(dataset: str, m: int, column: str = "tag",
              wide_p50: float = 10.1, narrow_p50: float = 0.7,
              ratio: float = 0.07, engine: str = "percona",
              pass_: str = "tuned", build_mode: str = "incremental",
              storage: str = "InnoDB") -> Dict[str, Any]:
    """One `filter_strategies` entry, as `generate.summarize` builds it."""
    return {"engine": engine, "dataset": dataset, "filter_column": column,
            "iterative_scan": None, "m": m, "build_mode": build_mode,
            "storage_engine": storage, "resource_pass": pass_,
            "narrow": 0.01, "wide": 0.50, "narrow_p50": narrow_p50,
            "wide_p50": wide_p50, "narrow_recall": 0.98, "ratio": ratio,
            "span": 49.0, "observed": "pre-filter"}


def _operating_point(dataset: str, m: int, engine: str = "percona",
                     pass_: str = "tuned",
                     build_mode: str = "incremental",
                     storage: str = "InnoDB",
                     ef: int = 64) -> Dict[str, Any]:
    """One `operating_points` entry, as `loaders.operating_points` builds it."""
    return {"key": (engine, dataset, pass_, m, build_mode, storage),
            "engine": engine, "dataset": dataset, "resource_pass": pass_,
            "m": m, "build_mode": build_mode, "storage_engine": storage,
            "ef_search": ef, "ef_conflict": None, "effective_ef": None,
            "recall_at_k": 0.9712, "status": "ok", "status_conflict": None,
            "target_recall": 0.95, "grid_size": 12, "points_measured": 12,
            "ef_grid": [ef]}


def _CHURN_STATE_ORDER_FOR_MUTANT(record: Dict[str, Any]) -> int:
    """`_churn_table`'s own state rank, for the mutant sort below.

    Read off `render_mod` rather than re-spelled, so the mutant differs from
    the real sort key in ONE respect -- where the unit axes sit relative to
    this rank -- and not in how a state is ranked.
    """
    from report.churn_state import maintenance_state
    return render_mod._CHURN_STATE_ORDER.get(maintenance_state(record), 99)


# ---------------------------------------------------------------------------
# The run's own records, for the units that produced the four wrong rows.
# ---------------------------------------------------------------------------

def _ingest(dataset: str, pass_: str, m: int, rate: float) -> Dict[str, Any]:
    return {"phase": "ingest", "engine": "percona", "dataset": dataset,
            "resource_pass": pass_, "m": m, "build_mode": "incremental",
            "storage_engine": "InnoDB", "rows": ROWS[dataset],
            "ingest_rows_per_s": rate}


def _build(dataset: str, pass_: str, m: int, wall: float,
           rss: int) -> Dict[str, Any]:
    return {"phase": "index_build", "engine": "percona", "dataset": dataset,
            "resource_pass": pass_, "m": m, "build_mode": "incremental",
            "storage_engine": "InnoDB", "rows": ROWS[dataset],
            "build_wall_s": wall, "peak_rss_bytes": rss,
            "extra": {"separable_build": False}}


#: The tuned fashion-mnist sweep, verbatim from
#: `full-20261015-215006/report/records.jsonl`. Three M values, one pass, one
#: storage engine, one build mode: the grid that collapsed.
#:
#: **In the file's own order, which is M 16, 32, 8 and not 8, 16, 32.** A
#: dict comprehension keeps the LAST entry for a colliding key, so the order
#: these were written in is what decided which rate the run published three
#: times -- M=8's 598.41. Sorting them here would change the mutant value and
#: stop this file reproducing the published defect.
TUNED_FASHION_INGEST = [
    _ingest(FASHION, "tuned", 16, 359.12),
    _ingest(FASHION, "tuned", 32, 168.13),
    _ingest(FASHION, "tuned", 8, 598.41),
]
TUNED_FASHION_BUILD = [
    _build(FASHION, "tuned", 16, 167.074, 47_670_747_136),
    _build(FASHION, "tuned", 32, 356.870, 47_703_887_872),
    _build(FASHION, "tuned", 8, 100.265, 47_674_806_272),
]


def _old_ingest_by_key(ingest: List[Dict[str, Any]]) -> Dict[Any, Any]:
    """`_build_table`'s pre-change join, verbatim in shape: no `m`.

    Copied rather than described, so "the old form published the wrong rate"
    is a value this file measures and not a claim it makes.
    """
    return {
        (i.get("engine"), i.get("dataset"), i.get("resource_pass"),
         i.get("build_mode"), i.get("storage_engine")):
            i.get("ingest_rows_per_s")
        for i in ingest
    }


class TestTheIngestRateBelongsToTheRowsOwnM:
    """D1. One rate per (unit, M), not one rate per unit."""

    def _table(self) -> str:
        return render_mod._build_table({"build": TUNED_FASHION_BUILD,
                                        "ingest": TUNED_FASHION_INGEST})

    def test_each_m_row_carries_its_own_measured_rate(self):
        """Honest: 598 / 359 / 168 rows/s for M 8 / 16 / 32.

        Mutant (the key with `m` removed again): 598 / 598 / 598 -- the last
        ingest record written, three times. Two of the three cells move, and
        they are two of the four rows the run published wrongly.
        """
        table = self._table()
        assert _column(table, "M") == ["8", "16", "32"]
        assert _column(table, "Ingest rate") == ["598 rows/s", "359 rows/s",
                                                 "168 rows/s"]

    def test_the_pre_change_key_published_one_rate_for_all_three(self):
        """R2: the old form, over these very records, is wrong where the new
        form is right.

        Old form: the five-tuple key collapses three ingest records into ONE
        entry, so all three builds read 598.41. New form: 598.41, 359.12,
        168.13. A repair whose old form also distinguishes them is not a
        repair, and this is the line that says which it is.
        """
        old = _old_ingest_by_key(TUNED_FASHION_INGEST)
        assert len(old) == 1, old
        old_rates = [
            old[(b["engine"], b["dataset"], b["resource_pass"],
                 b["build_mode"], b["storage_engine"])]
            for b in TUNED_FASHION_BUILD]
        assert old_rates == [598.41, 598.41, 598.41], old_rates
        assert len(set(old_rates)) == 1, "the old rates must collide"

        new = _column(self._table(), "Ingest rate")
        assert len(set(new)) == 3, new
        assert [f"{r:,.0f} rows/s" for r in old_rates] != new

    def test_the_rate_and_the_build_time_in_one_row_no_longer_disagree(self):
        """The arithmetic that exposed the defect, run as a guard.

        For an incremental build the build wall clock is the corpus load, so
        `rows / build_wall_s` must equal the rate printed beside it. Honest:
        implied 598.4 / 359.1 / 168.1 against printed 598 / 359 / 168 -- every
        row inside one row/s, and the M=8 figure lands on the recorded 598.41
        to the second decimal, which is the evidence that the two quantities
        really are one measurement. Mutant: the M=16 row implies 359.1 and
        prints 598, the M=32 row implies 168.1 and prints 598 -- the
        published contradiction, in a unit test.
        """
        rows = _cells(self._table())
        header, body = rows[0], rows[1:]
        # Keyed on the M cell rather than zipped against the input list: the
        # table sorts and the fixture is in the records' own order, and a
        # positional pairing of two differently ordered sequences is the
        # defect this file is about wearing a test's clothes.
        printed = {row[header.index("M")]: row[header.index("Ingest rate")]
                   for row in body}
        assert len(printed) == len(TUNED_FASHION_BUILD)
        for record in TUNED_FASHION_BUILD:
            cell = printed[str(record["m"])]
            implied = record["rows"] / record["build_wall_s"]
            shown = float(cell.replace(" rows/s", "").replace(",", ""))
            assert abs(implied - shown) <= 1.0, (record["m"], implied, cell)

    def test_the_sift_row_the_run_published_as_206_reads_427(self):
        """The fourth wrong row, which came from a different dataset and a
        different collision order -- two ingest records, M=16 then M=32, so
        the loser was the FIRST one rather than the middle one.

        Honest: 427 rows/s beside 2,344.5 s over 1,000,000 rows (implied
        426.5). Mutant: 206 rows/s, which is M=32's rate, beside M=16's build.
        """
        ingest = [_ingest(SIFT, "tuned", 16, 426.53),
                  _ingest(SIFT, "tuned", 32, 206.15)]
        build = [_build(SIFT, "tuned", 16, 2_344.488, 48_482_803_712),
                 _build(SIFT, "tuned", 32, 4_850.863, 48_808_448_000)]
        table = render_mod._build_table({"build": build, "ingest": ingest})
        assert _column(table, "Ingest rate") == ["427 rows/s", "206 rows/s"]

        old = _old_ingest_by_key(ingest)
        assert old[("percona", SIFT, "tuned", "incremental", "InnoDB")] \
            == 206.15, "the old form must publish M=32's rate for M=16"

    def test_a_rate_measured_under_the_other_pass_is_not_borrowed(self):
        """The pass stayed in the key when `m` joined it.

        A non-regression guard rather than a probe of the published defect:
        `resource_pass` was in the key before this change and the pre-change
        code passes this test too. Its mutant arm is therefore executed here
        rather than left to a revert -- a key without the pass, over these
        same records, returns 598.41 and would print a rate measured on a
        76-core server against a build that ran on 8 cores.

        Honest: an em dash, because this run measured no normalized M=8 rate.
        Mutant: 598 rows/s.
        """
        build = _build(FASHION, "normalized", 8, 110.140, 6_948_192_256)
        ingest = [_ingest(FASHION, "tuned", 8, 598.41)]
        table = render_mod._build_table({"build": [build], "ingest": ingest})
        assert _column(table, "Ingest rate") == ["—"]

        passless = {(i["engine"], i["dataset"], i["m"], i["build_mode"],
                     i["storage_engine"]): i["ingest_rows_per_s"]
                    for i in ingest}
        mutant = passless[(build["engine"], build["dataset"], build["m"],
                           build["build_mode"], build["storage_engine"])]
        assert mutant == 598.41
        assert render_mod._fmt(mutant, 0, " rows/s") == "598 rows/s"


class TestTheBuildTableNamesItsResourcePass:
    """D1, second half: section 5 had no `Pass` column.

    The run printed fashion-mnist M=8 twice, 110.1 s at 6.5 GiB peak RSS and
    100.3 s at 44.4 GiB. Those are the normalized and the tuned pass -- 8 CPUs
    and 16 GB against 76 and 112.5 GiB -- and nothing on the page said so.
    """

    #: The two rows as the run wrote them. Identical but for the pass.
    PAIR = [_build(FASHION, "normalized", 8, 110.140, 6_948_192_256),
            _build(FASHION, "tuned", 8, 100.265, 47_674_806_272)]

    #: The identity columns section 5 carried before this change.
    OLD_IDENTITY = ["Engine", "Dataset", "M", "Storage", "Build mode"]

    def test_the_two_passes_are_distinguishable_rows(self):
        """Honest: the `Pass` column reads `normalized` and `tuned`.

        Old form, read back out of the rendered table: both rows carry the
        identity ('percona', 'fashion-mnist-784-euclidean', '8', 'InnoDB',
        'incremental'), byte-identical, with 6.5 GiB and 44.4 GiB of peak RSS
        beside them. Distinguishable: False.
        """
        table = render_mod._build_table({"build": self.PAIR, "ingest": []})
        old = _identities(table, self.OLD_IDENTITY)
        assert len(old) == 2 and old[0] == old[1], old

        assert _column(table, "Pass") == ["normalized", "tuned"]
        body = _cells(table)[1:]
        assert body[0] != body[1]

    def test_the_peak_rss_stays_with_its_own_pass(self):
        """The figure the missing column made unreadable, now beside the pass
        that produced it. Honest: 6.5 GiB on normalized, 44.4 GiB on tuned.
        """
        table = render_mod._build_table({"build": self.PAIR, "ingest": []})
        rows = _cells(table)
        header, body = rows[0], rows[1:]
        pairs = {row[header.index("Pass")]: row[header.index("Peak server RSS")]
                 for row in body}
        assert pairs == {"normalized": "6.5 GiB", "tuned": "44.4 GiB"}

    def test_a_record_with_no_pass_renders_the_em_dash(self):
        """Archived records predate the field. The dash is this file's
        convention for "nobody recorded this" everywhere else, and a blank
        would read as a column that failed to render.
        """
        bare = dict(_build(FASHION, "tuned", 8, 100.265, 1))
        bare.pop("resource_pass")
        table = render_mod._build_table({"build": [bare], "ingest": []})
        assert _column(table, "Pass") == ["—"]

    def test_the_m_cell_renders_a_float_m_the_way_every_other_table_does(self):
        """`_fmt(m, 0)`, not `str(r.get("m", "—"))`.

        **Named after the live half.** This test has two arms and only one of
        them discriminates:

        * the FLOAT arm is the probe. `str(16.0)` is "16.0" where the other
          six ops-unit tables print "16", so honest `16` / mutant `16.0`.
        * the `None` arm is an identical pair under both spellings and is
          asserted as documentation, not as a probe. `str(None)` is the
          four-character string "None" -- `.get`'s default never fires,
          because a record carrying `m: None` has the key present -- and it
          reached the page as an em dash only because `_md_table`'s
          `_sanitise_cell` lists "None" among the three empties it maps. That
          dependency is unguarded either way: remove the arm and the old
          spelling prints `None` on the page with the whole suite green. It
          is recorded here so the next person to touch `_sanitise_cell` can
          find out who was relying on it.

        `_fmt(m, 0)` also rounds and comma-groups -- `16.4` renders `16` and
        `1024.0` renders `1,024`. Neither is a measurement being altered:
        a non-integral M is nonsense input rather than a reading, so this
        hides a configuration error instead of publishing a wrong number, and
        the other six ops-unit tables have behaved this way for longer. This
        is convergence on their behaviour, not a new one.
        """
        none_m = dict(_build(FASHION, "tuned", 8, 100.265, 1))
        none_m["m"] = None
        table = render_mod._build_table({"build": [none_m], "ingest": []})
        assert _column(table, "M") == ["—"]
        assert str(none_m.get("m", "—")) == "None"

        float_m = dict(_build(FASHION, "tuned", 8, 100.265, 1))
        float_m["m"] = 16.0
        table = render_mod._build_table({"build": [float_m], "ingest": []})
        assert _column(table, "M") == ["16"]
        assert str(float_m.get("m", "—")) == "16.0"

    def test_every_body_cell_is_filled(self):
        """A column added to the header and not to the row shifts every cell
        after it, which renders as a plausible table of wrong numbers."""
        rows = _cells(render_mod._build_table(
            {"build": self.PAIR, "ingest": []}))
        header, body = rows[0], rows[1:]
        assert len(body) == 2
        for row in body:
            assert len(row) == len(header), list(zip(header, row))


def _concurrency(pass_: str, m: int, qps: float, recall: float,
                 spread: float, clients: int = 1,
                 engine: str = "percona", dataset: str = FASHION,
                 build_mode: str = "incremental",
                 storage: str = "InnoDB") -> Dict[str, Any]:
    """One concurrency record, as `full-20261015-215006` wrote it."""
    return {"phase": "concurrency", "engine": engine, "dataset": dataset,
            "resource_pass": pass_, "m": m, "build_mode": build_mode,
            "storage_engine": storage, "clients": clients, "qps": qps,
            "recall_at_k": recall, "latency_p50_ms": 0.5,
            "latency_p95_ms": 0.6, "latency_p99_ms": 0.7,
            "extra": {"repeats": 3, "qps_spread_pct": spread,
                      "query_pool": 10_000, "query_reuse": 5.0,
                      "scaling_efficiency": 1.0, "served_concurrency": 0.98,
                      "client_cpu_util": 0.03, "client_cpus": 4}}


#: The five one-client fashion-mnist points the run published as five
#: indistinguishable rows, in the order section 7 printed them.
FIVE_UNITS = [
    _concurrency("normalized", 32, 1_677.0, 0.9871, 0.6),
    _concurrency("normalized", 8, 2_037.0, 0.9629, 0.1),
    _concurrency("tuned", 16, 1_952.0, 0.97138, 0.9),
    _concurrency("tuned", 32, 1_673.0, 0.98813, 6.4),
    _concurrency("tuned", 8, 1_789.0, 0.96223, 0.9),
]


class TestTheConcurrencyTableNamesItsOpsUnit:
    """D2. Section 7 carried `Storage` and neither `Pass` nor `M`.

    `_run_unit` loops `m_values x build_modes x storage_engines` inside each
    resource pass and every pass through it rebuilds the index, so a
    concurrency row is a statement about one of those units and not about the
    engine. Section 8's filtered table has carried Pass, M and Build mode
    since PR 3 for exactly this reason; this is that fix, one section over.
    """

    #: The identity columns section 7 carried before this change.
    OLD_IDENTITY = ["Engine", "Dataset", "Storage", "Clients"]

    def _table(self) -> str:
        return render_mod._concurrency_table({"concurrency": FIVE_UNITS})

    def test_five_units_render_as_five_distinguishable_rows(self):
        """Honest: five distinct identities, because `Pass` and `M` are in
        them -- (normalized, 8), (normalized, 32), (tuned, 8), (tuned, 16),
        (tuned, 32).

        Old form, read back out of the rendered table: one identity,
        ('percona', 'fashion-mnist-784-euclidean', 'InnoDB', '1'), five times
        over, with 2,037 and 1,673 QPS beside it -- a 22% spread presented as
        five readings of one configuration. Distinct identities: 1 of 5.
        """
        old = _identities(self._table(), self.OLD_IDENTITY)
        assert len(old) == 5 and len(set(old)) == 1, old

        rows = _cells(self._table())
        header, body = rows[0], rows[1:]
        keys = {(row[header.index("Pass")], row[header.index("M")])
                for row in body}
        assert keys == {("normalized", "8"), ("normalized", "32"),
                        ("tuned", "8"), ("tuned", "16"), ("tuned", "32")}
        assert len({tuple(row) for row in body}) == 5

    def test_each_qps_is_rendered_beside_the_machine_that_produced_it(self):
        """The pass is the damaging half: normalized gives the server 8 CPUs
        and 16 GB, tuned gave it 76 and 112.5 GiB in this run, and section 1
        names one pass for the whole document. Honest: 2,037 is labelled
        `normalized`. Mutant: it is labelled nothing, and reads as tuned.
        """
        rows = _cells(self._table())
        header, body = rows[0], rows[1:]
        by_qps = {row[header.index("QPS")]:
                  (row[header.index("Pass")], row[header.index("M")])
                  for row in body}
        assert by_qps["2,037"] == ("normalized", "8")
        assert by_qps["1,677"] == ("normalized", "32")
        assert by_qps["1,952"] == ("tuned", "16")
        assert by_qps["1,673"] == ("tuned", "32")
        assert by_qps["1,789"] == ("tuned", "8")

    def test_the_m_column_sorts_as_a_number(self):
        """`full` sweeps 8/16/32, which as text sorts 16, 32, 8. The filtered
        table's comment records the same trap; this is the same `m or 0`.
        """
        rows = _cells(render_mod._concurrency_table(
            {"concurrency": [r for r in FIVE_UNITS
                             if r["resource_pass"] == "tuned"]}))
        header, body = rows[0], rows[1:]
        assert [row[header.index("M")] for row in body] == ["8", "16", "32"]

    def test_a_record_with_no_unit_axes_renders_dashes(self):
        """A point written before the fields existed gets the em dash, not a
        zero: `_fmt(None, 0)` is "nobody measured this" and `0` would be a
        graph degree no engine was ever given.
        """
        bare = {"phase": "concurrency", "engine": "valkey", "dataset": "d",
                "clients": 32, "qps": 5_820.0, "extra": {}}
        table = render_mod._concurrency_table({"concurrency": [bare]})
        assert _column(table, "Pass") == ["—"]
        assert _column(table, "M") == ["—"]


def _churn(pass_: str, m: int, fraction: float, qps: float, recall: float,
           state: str | None, index_bytes: int,
           engine: str = "percona", dataset: str = FASHION,
           build_mode: str = "incremental",
           storage: str = "InnoDB") -> Dict[str, Any]:
    """One churn measurement, as `full-20261015-215006` wrote it."""
    extra: Dict[str, Any] = {}
    if state is not None:
        extra["maintenance_state"] = state
        if state == "after":
            extra["maintenance"] = "ALTER TABLE (index replace)"
            extra["maintenance_s"] = 0.0
    return {"phase": "churn", "engine": engine, "dataset": dataset,
            "resource_pass": pass_, "m": m, "build_mode": build_mode,
            "storage_engine": storage, "churn_fraction": fraction,
            "qps": qps, "recall_at_k": recall, "index_bytes": index_bytes,
            "extra": extra}


#: The five 10%-before-repair rows the run published with byte-identical
#: identity cells, in the order section 9 printed them.
FIVE_CHURN_UNITS = [
    _churn("normalized", 32, 0.10, 1_503.0, 0.9865, "before", 285_212_672),
    _churn("normalized", 8, 0.10, 1_898.0, 0.9594, "before", 281_018_368),
    _churn("tuned", 16, 0.10, 1_561.0, 0.9672, "before", 281_018_368),
    _churn("tuned", 32, 0.10, 1_331.0, 0.9860, "before", 281_018_368),
    _churn("tuned", 8, 0.10, 1_829.0, 0.9598, "before", 281_018_368),
]


class TestTheChurnTableNamesItsOpsUnit:
    """D3. Section 9's first churn table carried neither `Pass` nor `M`.

    `churn_group` in the same function has always held both constant when
    deciding what a row is tied with, so the renderer knew they were part of
    a row's identity while the header named neither -- and the
    churn-under-writes table immediately below prints both. This is the same
    shape as the `Repair` column, which was also a fix: two real measurements
    rendered as one row twice, read as a noisy measurement.
    """

    #: The identity columns section 9's first table carried before this
    #: change. `Repair` is in the list because it is itself a fix of this
    #: shape and the five units collide through it too.
    OLD_IDENTITY = ["Engine", "Dataset", "Storage", "Churn", "Repair"]

    def test_five_units_render_as_five_distinguishable_rows(self):
        """Honest: five distinct identities.

        Old form, read back out of the rendered table: one identity,
        ('percona', 'fashion-mnist-784-euclidean', 'InnoDB', '10%',
        'before'), five times, with recalls 0.9865 / 0.9594 / 0.9672 /
        0.9860 / 0.9598 and QPS 1,503 / 1,898 / 1,561 / 1,331 / 1,829 beside
        it. Distinct identities: 1 of 5.
        """
        table = render_mod._churn_table({"churn": FIVE_CHURN_UNITS})
        old = _identities(table, self.OLD_IDENTITY)
        assert len(old) == 5 and len(set(old)) == 1, old

        rows = _cells(table)
        header, body = rows[0], rows[1:]
        keys = {(row[header.index("Pass")], row[header.index("M")])
                for row in body}
        assert keys == {("normalized", "8"), ("normalized", "32"),
                        ("tuned", "8"), ("tuned", "16"), ("tuned", "32")}
        assert len({tuple(row) for row in body}) == 5

    #: Two units, each with its own before/after pair at 10%. Two is the
    #: minimum that can see the defect this guards: with ONE unit the four
    #: unit axes are constant across the pair, so their position in the sort
    #: tuple cannot affect the order and the test is decorative.
    TWO_PAIRS = [
        _churn("tuned", 16, 0.10, 1_561.0, 0.9672, "before", 281_018_368),
        _churn("tuned", 16, 0.10, 1_561.0, 0.9672, "after", 281_018_368),
        _churn("tuned", 32, 0.10, 1_331.0, 0.9860, "before", 281_018_368),
        _churn("tuned", 32, 0.10, 1_374.0, 0.9860, "after", 281_018_368),
    ]

    def test_each_units_repair_pair_is_adjacent(self):
        """The precedence claim in `_churn_table`'s sort key, as a value.

        The four unit axes sort ahead of `_CHURN_STATE_ORDER`, which stays
        last, so a unit's degraded and repaired rows are adjacent and in the
        order they were measured.

        Honest: `before, after, before, after`.
        Mutant (the four unit axes moved BEHIND the state rank, computed
        inline below over these same records): `before, before, after,
        after` -- every unit's pair separated by as many rows as the grid is
        wide, which on `full-20261015-215006` is five. Nothing about the
        numbers changes under that mutation and the whole suite stays green,
        which is exactly why this is pinned here.
        """
        rows = _cells(render_mod._churn_table({"churn": self.TWO_PAIRS}))
        header, body = rows[0], rows[1:]
        states = [row[header.index("Repair")] for row in body]
        assert len(states) == 4
        assert states[0] == "before"
        assert states[1].startswith("after ALTER TABLE (index replace)")
        assert states[2] == "before"
        assert states[3].startswith("after ALTER TABLE (index replace)")

        # And the pairs really are two different units, not one rendered
        # twice: without this the sequence above could be satisfied by a
        # fixture that collapsed.
        units = [(row[header.index("Pass")], row[header.index("M")])
                 for row in body]
        assert units == [("tuned", "16"), ("tuned", "16"),
                         ("tuned", "32"), ("tuned", "32")]

        from report.churn_state import maintenance_state
        mutant = [maintenance_state(r) or "—" for r in sorted(
            self.TWO_PAIRS,
            key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                           # The state rank FIRST, the unit axes behind it:
                           # the one reordering of this tuple that keeps every
                           # number right and makes the page unreadable.
                           _CHURN_STATE_ORDER_FOR_MUTANT(r),
                           r.get("resource_pass") or "",
                           r.get("m") or 0,
                           r.get("build_mode") or "",
                           r.get("storage_engine") or "",
                           r.get("churn_fraction") or 0))]
        assert mutant == ["before", "before", "after", "after"], mutant
        assert [s.split()[0] for s in states] != mutant

    def test_every_body_cell_is_filled(self):
        """A column added to the header and not to the row shifts every cell
        after it, which renders as a plausible table of wrong numbers."""
        rows = _cells(render_mod._churn_table({"churn": self.TWO_PAIRS}))
        header, body = rows[0], rows[1:]
        assert len(body) == 4
        for row in body:
            assert len(row) == len(header), list(zip(header, row))

    def test_the_m_column_sorts_as_a_number(self):
        rows = _cells(render_mod._churn_table(
            {"churn": [r for r in FIVE_CHURN_UNITS
                       if r["resource_pass"] == "tuned"]}))
        header, body = rows[0], rows[1:]
        assert [row[header.index("M")] for row in body] == ["8", "16", "32"]

    def test_a_record_with_no_unit_axes_renders_dashes(self):
        bare = {"phase": "churn", "engine": "mariadb", "dataset": "d",
                "storage_engine": "InnoDB", "churn_fraction": 0.25,
                "qps": 23.0, "recall_at_k": 0.99, "extra": {}}
        table = render_mod._churn_table({"churn": [bare]})
        assert _column(table, "Pass") == ["—"]
        assert _column(table, "M") == ["—"]


class TestTheBuildModeAxisIsNamedToo:
    """The fourth unit axis, which section 7 and section 9 still omitted.

    `orchestrator/cli.py`'s ops loop is `for m in m_values: for build_mode in
    build_modes: for storage in storage_engines:` inside each resource pass --
    four axes, and after Pass and M landed these two tables named three of
    them. The omission is reachable rather than theoretical: `full.yml`,
    `main.yml`, `smoke.yml` and `quick.yml` all set `pgvector_build_modes:
    [post, incremental]`, so a pgvector run writes two concurrency points and
    two churn rows per (pass, M, storage) that are a bulk-built index and an
    incrementally maintained one. It did not show in
    `full-20261015-215006` only because `config/profiles/percona.yml` pins
    `percona_build_modes: [incremental]`.

    The figures below are the shape the reviewer measured for that case -- a
    22% throughput spread between two different indexes -- not numbers read
    off a bundle, because no bundle here has run that grid yet.
    """

    #: Two pgvector units differing in NOTHING but the build mode.
    TWO_MODES = [
        _concurrency("tuned", 16, 4_120.0, 0.9810, 0.4, engine="pgvector",
                     build_mode="post", storage="heap"),
        _concurrency("tuned", 16, 3_380.0, 0.9655, 0.5, engine="pgvector",
                     build_mode="incremental", storage="heap"),
    ]

    TWO_MODES_CHURN = [
        _churn("tuned", 16, 0.10, 4_120.0, 0.9810, "before", 281_018_368,
               engine="pgvector", build_mode="post", storage="heap"),
        _churn("tuned", 16, 0.10, 3_380.0, 0.9655, "before", 281_018_368,
               engine="pgvector", build_mode="incremental", storage="heap"),
    ]

    #: Section 7's identity columns after Pass and M landed and before
    #: `Build mode` did.
    OLD_IDENTITY = ["Engine", "Dataset", "Pass", "M", "Storage", "Clients"]

    def test_section_7_tells_the_two_build_modes_apart(self):
        """Honest: `Build mode` reads `post` and `incremental`, so 4,120 and
        3,380 QPS are visibly two indexes.

        Old form, read back out of the rendered table: both rows carry
        ('pgvector', 'fashion-mnist-784-euclidean', 'tuned', '16', 'heap',
        '1') -- byte-identical across a 22% throughput spread and a recall
        difference of 0.0155. Distinct identities: 1 of 2.
        """
        table = render_mod._concurrency_table({"concurrency": self.TWO_MODES})
        old = _identities(table, self.OLD_IDENTITY)
        assert len(old) == 2 and old[0] == old[1], old

        assert _column(table, "Build mode") == ["incremental", "post"]
        body = _cells(table)[1:]
        assert body[0] != body[1]

    def test_section_9_tells_the_two_build_modes_apart(self):
        """The same omission in the churn table, which `churn_group` has
        always held `build_mode` constant in."""
        table = render_mod._churn_table({"churn": self.TWO_MODES_CHURN})
        old = _identities(table, ["Engine", "Dataset", "Pass", "M", "Storage",
                                  "Churn", "Repair"])
        assert len(old) == 2 and old[0] == old[1], old

        assert _column(table, "Build mode") == ["incremental", "post"]
        body = _cells(table)[1:]
        assert body[0] != body[1]

    def test_a_record_with_no_build_mode_renders_the_em_dash(self):
        bare = {"phase": "concurrency", "engine": "valkey", "dataset": "d",
                "clients": 32, "qps": 5_820.0, "extra": {}}
        table = render_mod._concurrency_table({"concurrency": [bare]})
        assert _column(table, "Build mode") == ["—"]


class TestSectionSevenAndNineSayWhatTheBuildModeCellIs:
    """`BUILD_MODE_COLUMN_NOTE`, and that both sections actually carry it.

    This is the one thing on the page stopping a reader from taking a
    MariaDB `post` cell as a description of how its index was built --
    `cli.py` defaults the mode to `post` for every engine that does not
    declare `capabilities.bulk_vector_index_build: false`, and for MHNSW and
    VIDX `post` is a no-op. It shipped with no test at all: deleting the
    constant from `_concurrency_table` produced 0 failures in 1,636. Its own
    stated precedent, `TIE_COLUMN_NOTE`, has a content test, so this is the
    matching one.
    """

    #: Phrases the note cannot lose without ceasing to do its job: what the
    #: column IS, the mechanism that makes the cell potentially misleading,
    #: and where the honest answer lives. A substring list rather than the
    #: whole string, so rewording the prose is allowed and gutting it is not.
    MUST_SAY = ("ops-unit axis", "post", "no-op", "section 5",
                "`Index build`")

    def test_the_note_names_the_axis_the_mechanism_and_the_alternative(self):
        note = render_mod.BUILD_MODE_COLUMN_NOTE
        for phrase in self.MUST_SAY:
            assert phrase in note, (phrase, note)

    def test_section_7_carries_it(self):
        """Honest: the note is in `_concurrency_table`'s output. Mutant (the
        constant dropped from that return, which was the shipped state of the
        previous commit): it is absent and nothing else notices."""
        table = render_mod._concurrency_table({"concurrency": FIVE_UNITS})
        assert render_mod.BUILD_MODE_COLUMN_NOTE in table

    def test_section_9_carries_it(self):
        table = render_mod._churn_table({"churn": FIVE_CHURN_UNITS})
        assert render_mod.BUILD_MODE_COLUMN_NOTE in table

    def test_both_sections_carry_the_same_string(self):
        """One constant, two readers, for the reason `TIE_COLUMN_NOTE` is
        one: two sections wording the same caveat separately is how one of
        them comes to be edited into contradicting the other."""
        concurrency = render_mod._concurrency_table(
            {"concurrency": FIVE_UNITS})
        churn = render_mod._churn_table({"churn": FIVE_CHURN_UNITS})
        assert render_mod.BUILD_MODE_COLUMN_NOTE in concurrency
        assert render_mod.BUILD_MODE_COLUMN_NOTE in churn


class TestTheOperatingPointMCellMatchesTheOtherTables:
    """Section 6's M cell, which shipped unguarded.

    Reverting it from `_fmt(m, 0)` to `str(point.get("m") or "—")` produced
    0 failures in 1,636 tests, so the fix had no probe. A float M is the live
    half for the same reason it is in the build table's test: `str(16.0)` is
    "16.0" where the other seven ops-unit tables print "16".
    """

    def test_a_float_m_renders_without_its_decimal(self):
        """Honest: `16`. Mutant (the `str()` spelling): `16.0`."""
        point = dict(_operating_point(FASHION, 16))
        point["m"] = 16.0
        table = render_mod._operating_point_table(
            {"operating_points": [point]})
        assert _column(table, "M") == ["16"]
        assert str(point.get("m") or "—") == "16.0"

    def test_a_point_with_no_m_renders_the_em_dash(self):
        point = dict(_operating_point(FASHION, 16))
        point["m"] = None
        table = render_mod._operating_point_table(
            {"operating_points": [point]})
        assert _column(table, "M") == ["—"]


class TestTheFilterStrategyTableNamesItsCorpus:
    """`_filter_strategy_table` named every ops-unit axis and not the dataset.

    Pre-existing, and not caused or worsened by the join-key fix: it reads
    the same way in the archived `report.md`. Closed here because it is the
    same defect as the rest of this file in the same file -- a table that
    collapses two measurements into repeated rows -- and because
    `generate.summarize` has always put `dataset` in the grouping key and
    carried it on every entry, so the renderer was the only thing dropping
    it.

    On `full-20261015-215006` the table published 21 rows carrying 12
    distinct identities, 8 of them two or three times over.
    """

    #: Two datasets, same ops unit and same filter column, with the two
    #: latency pairs the published table printed as repeated rows.
    TWO_DATASETS = [
        _strategy(FASHION, 16, wide_p50=10.1, narrow_p50=0.7),
        _strategy(SIFT, 16, wide_p50=15.8, narrow_p50=0.9),
    ]

    #: The identity columns this table carried before `Dataset` landed.
    OLD_IDENTITY = ["Engine", "Pass", "M", "Build mode", "Storage", "Column"]

    def test_two_datasets_are_distinguishable_rows(self):
        """Honest: the `Dataset` column separates them.

        Old form, read back out of the rendered table: both rows carry
        ('Percona Server (InnoDB HNSW)', 'tuned', '16', 'incremental',
        'InnoDB', 'tag') -- byte-identical -- beside `0.7 → 10.1 ms` and
        `0.9 → 15.8 ms`. Distinct identities: 1 of 2.
        """
        table = render_mod._filter_strategy_table(
            {"filter_strategies": self.TWO_DATASETS})
        old = _identities(table, self.OLD_IDENTITY)
        assert len(old) == 2 and old[0] == old[1], old

        assert _column(table, "Dataset") == [FASHION, SIFT]
        body = _cells(table)[1:]
        assert body[0] != body[1]

    def test_the_latency_pair_stays_with_its_own_corpus(self):
        table = render_mod._filter_strategy_table(
            {"filter_strategies": self.TWO_DATASETS})
        rows = _cells(table)
        header, body = rows[0], rows[1:]
        pairs = {row[header.index("Dataset")]:
                 row[header.index("Median latency")] for row in body}
        assert pairs == {FASHION: "10.1 → 0.7 ms", SIFT: "15.8 → 0.9 ms"}

    def test_an_entry_with_no_dataset_renders_the_em_dash(self):
        bare = dict(_strategy(FASHION, 16))
        bare.pop("dataset")
        table = render_mod._filter_strategy_table(
            {"filter_strategies": [bare]})
        assert _column(table, "Dataset") == ["—"]


class TestEveryOpsUnitTableSpellsTheUnitColumnsTheSameWay:
    """One reader convention across the seven tables that render an ops unit.

    A column named `Pass` in one section and `Resource pass` in the next is a
    second thing for a reader to notice, and this report is read by scanning
    down a column.

    **The axis list is read from `report.generate.OPS_UNIT_FIELDS`, not
    restated here.** Restating it is the shape that let `build_mode` slip:
    a guard that names a subset of the convention cannot see the part it left
    out, and the first version of this class asserted the two literals
    `"Pass"` and `"M"` while two of these tables named three of the four
    axes. The key side of this is already self-guarding --
    `tests/test_t3c_filtered.py` AST-parses `cli._run_unit` and fails if a
    fifth looped axis appears without joining `OPS_UNIT_FIELDS` -- so reading
    the same tuple here closes the header side against the same mutation.
    Add `quantization` to the loop and to `OPS_UNIT_FIELDS` and
    `test_the_column_map_covers_every_ops_unit_field` fails naming it,
    instead of seven tables silently collapsing two quantizations.
    """

    #: `OPS_UNIT_FIELDS` member -> the column header that prints it. The map
    #: is asserted to cover the tuple exactly, which is what makes a new axis
    #: fail here rather than pass.
    COLUMN_FOR_FIELD = {
        "resource_pass": "Pass",
        "m": "M",
        "build_mode": "Build mode",
        "storage_engine": "Storage",
    }

    #: Every table in `_rendered_tables` opens with these, in this order.
    #: There is no exemption list and this must not acquire one by loosening:
    #: a table that genuinely cannot carry `Dataset` belongs out of that dict
    #: and named in `SEQUENCE_NOT_CHECKED` with its reason, not inside it
    #: under a weakened assertion.
    UNIT_PREFIX = ["Engine", "Dataset", "Pass", "M"]

    #: Then these two, and **the build table alone has them the other way
    #: round** -- `Storage | Build mode` where the others read
    #: `Build mode | Storage`. Reordering it was considered and declined
    #: twice.
    #:
    #: The narrow fact the decision rests on: **no parser anywhere reads
    #: `Storage` or `Build mode` by index**, so swapping those two
    #: specifically is invisible to every consumer. That is NOT the same as
    #: "nothing reads these columns positionally", which an earlier version
    #: of this comment claimed and which is false --
    #: `tests/test_t3c_filtered.py:2382` --
    #: `test_the_m_column_sorts_numerically_not_as_text` -- reads the filtered
    #: table's cell `[3]`, and `tests/test_t3c_filtered.py:2053-2055` --
    #: `_published_strategies`, in the same class -- reads the strategy table
    #: as `cells[-4]` and `cells[-1]` behind a `len(cells) >= 12` gate that a
    #: 13-column header would also pass if a column went missing. **An
    #: INSERTION is therefore not safe**, and believing otherwise is what
    #: cost one commit in this series 24 re-pointed assertion sites -- 25
    #: integer subscripts -- in `tests/test_t2b_report.py`.
    #:
    #: **BOTH NUMBERS WERE WRONG AND BOTH NOW CARRY THE SYMBOL BESIDE THEM.**
    #: They read `c8de6a5:tests/test_t3c_filtered.py:2263` and
    #: `c8de6a5:tests/test_t3c_filtered.py:1955-1958`, and at that commit 2263
    #: held the opening of a nested `ast.walk` helper while 1955-1958 held a
    #: section banner and the signature of `_filtered_record` -- none of which
    #: reads a cell at all. Nothing had moved them recently; they were already
    #: wrong, which is how most of the rotten citations found so far have
    #: been wrong, and the reason `tests/test_e16_citation_ratchet.py` now
    #: refuses new ones. The two live numbers above are re-derived rather than
    #: converted to symbols, because repairing this tree's existing citations
    #: was explicitly out of scope for the change that added that ratchet; the
    #: symbol names are written beside them so the next reader can recover the
    #: lines the next time they move.
    #:
    #: With nine tables carrying these axes across four column vocabularies
    #: there is no single achievable ordering for the whole file, so the set
    #: that genuinely is a convention is pinned and the deviation is recorded
    #: rather than hidden. A MISSING column is the defect class that
    #: mattered, and a set still catches that.
    UNIT_PAIR = {"Build mode", "Storage"}

    #: Tables the sequence assertion deliberately skips, with the reason.
    #: Checked for the AXES by `test_every_ops_unit_header_names_every_axis`,
    #: which reads the file rather than rendering it, so skipping the
    #: sequence here does not drop them from the covered set.
    SEQUENCE_NOT_CHECKED = {
        "consistency_match": "section 10 interposes `ef_search` between M "
                             "and Storage, and is rendered as part of "
                             "`_consistency_section` rather than by a table "
                             "function this test can call directly",
    }

    def _rendered_tables(self):
        """The ops-unit tables this test can call and render directly.

        **Hand-written, and therefore not the covered set.** This list has
        been wrong twice -- five when it was seven, seven when it was eight --
        and each time the omission was measurable: renaming
        `_filter_strategy_table`'s `Build mode` to `Build` left every test in
        this file green, and dropping section 10's `Storage` column produced
        no failure in 3,143 tests but a citation line-shift. So the AXIS
        dimension is derived from the file by
        `test_every_ops_unit_header_names_every_axis` and this dict is used
        only for the checks that need rendered output: the column sequence
        and rectangularity.
        """
        return {
            "build": render_mod._build_table(
                {"build": TUNED_FASHION_BUILD,
                 "ingest": TUNED_FASHION_INGEST}),
            "concurrency": render_mod._concurrency_table(
                {"concurrency": FIVE_UNITS}),
            "churn": render_mod._churn_table({"churn": FIVE_CHURN_UNITS}),
            "filtered": render_mod._filtered_table({"filtered": [
                {"phase": "filtered", "engine": "percona", "dataset": FASHION,
                 "resource_pass": "tuned", "m": 16,
                 "build_mode": "incremental", "storage_engine": "InnoDB",
                 "selectivity": 0.01, "recall_at_k": 0.99, "qps": 100.0,
                 "extra": {}}]}),
            "churn_during_writes": render_mod._churn_during_writes_table(
                {"churn_during_writes": [
                    {"engine": "percona", "dataset": FASHION,
                     "resource_pass": "tuned", "m": 16,
                     "build_mode": "incremental", "storage_engine": "InnoDB",
                     "churn_fraction": 0.10, "readers": 4,
                     "baseline_p99_ms": 10.0, "under_writes_p99_ms": 20.0,
                     "p99_ratio": 2.0, "baseline_qps": 100.0,
                     "under_writes_qps": 50.0,
                     "writer_rows_per_s": 10.0}]}),
            "filter_strategy": render_mod._filter_strategy_table(
                {"filter_strategies": [_strategy(FASHION, 16)]}),
            "operating_point": render_mod._operating_point_table(
                {"operating_points": [_operating_point(FASHION, 16)]}),
        }

    def test_the_column_map_covers_every_ops_unit_field(self):
        """Non-vacuity for the test below, and the guard against a fifth axis.

        Honest: `OPS_UNIT_FIELDS` is exactly the four keys of
        `COLUMN_FOR_FIELD`. Mutant (a `quantization` axis added to the loop
        and to `OPS_UNIT_FIELDS`, which `tests/test_t3c_filtered.py` requires
        of anyone adding one): this fails naming `quantization`, and the
        author has to give it a column before the header check below can
        pass.
        """
        from report.generate import OPS_UNIT_FIELDS
        assert set(OPS_UNIT_FIELDS) == set(self.COLUMN_FOR_FIELD), (
            sorted(set(OPS_UNIT_FIELDS) ^ set(self.COLUMN_FOR_FIELD)))

    #: How many `_md_table` literal headers in `report/render.py` name both
    #: `Pass` and `M`. Pinned so the test below cannot pass by selecting
    #: nothing: a predicate that stops matching is a guard that stopped
    #: working, and it would look identical to a clean run.
    #:
    #: 8 -> 9: §2's "same configuration was measured more than once" table
    #: gained the unit. It lists recall configurations rather than ops units,
    #: but it is keyed on `build_signature`, which takes the configuration
    #: from the result filename and so carries the storage engine and the
    #: build mode inside it -- two rows can differ on either, and a table
    #: printing neither made two configurations' duplicates read as one
    #: configuration listed twice. The reasoning this guard records for
    #: section 10's `Storage` applies unchanged.
    #:
    #: 9 -> 10: §7's peak-throughput headline (`_peak_throughput_table`). It
    #: ranks concurrency UNITS -- one row per (engine, pass, M, build mode,
    #: storage) -- so it names all four axes like every other unit table.
    #: 10 -> 11: §7's "What limited throughput" table (`_limiter_table`), one
    #: row per concurrency point, so the same four axes and the client count.
    OPS_UNIT_TABLE_COUNT = 11

    def test_every_ops_unit_header_names_every_axis(self):
        """The header side of `ops_unit`, over the FILE rather than a list.

        **Both dimensions are derived.** The axes come from
        `report.generate.OPS_UNIT_FIELDS`, and the tables come from an AST
        scan of `report/render.py` for every `_md_table` call whose header is
        a literal list of strings. The predicate is "names both `Pass` and
        `M`", which selects every table that renders a unit.

        It is NOT true that the validity-section partials are excluded by
        construction. That clause stood here until §2's "same configuration
        was measured more than once" table gained the full unit -- it lists
        recall configurations rather than ops units, but it is keyed on
        `build_signature`, which takes the configuration from the result
        filename and so carries the storage engine and the build mode inside
        it, and a table printing neither made two configurations' duplicates
        read as one configuration listed twice. Any validity table that names
        a unit is selected and must spell it in full, which is the rule this
        guard enforces and not an exemption from it.

        Deriving the table dimension is the point. A hand-written list was
        wrong twice in this series, and the second time the missing table was
        section 10's cross-check between the two measurement paths, where
        dropping `Storage` makes one engine's cross-check read as another's
        and produced no test failure at all.

        Honest: 9 tables selected, none missing an axis.
        Mutant (section 10's `Storage` removed, or any axis renamed in any of
        the nine): this fails naming the header's line number and the
        missing column.
        """
        from report.generate import OPS_UNIT_FIELDS
        wanted = {self.COLUMN_FOR_FIELD[f] for f in OPS_UNIT_FIELDS}
        selected = [(lineno, h) for lineno, h in _md_table_headers()
                    if {"Pass", "M"} <= set(h)]
        assert len(selected) == self.OPS_UNIT_TABLE_COUNT, (
            f"{len(selected)} headers in report/render.py name both Pass and "
            f"M, expected {self.OPS_UNIT_TABLE_COUNT}. If a table was added "
            f"or removed on purpose, update the count; if the predicate "
            f"stopped matching, this guard had stopped working.",
            [lineno for lineno, _ in selected])
        for lineno, header in selected:
            missing = sorted(wanted - set(header))
            assert not missing, (f"render.py:{lineno}", missing, header)

    def test_the_rendered_tables_open_with_the_same_unit_columns(self):
        """The column ORDER, on the tables this test can render.

        Honest: each opens `Engine, Dataset, Pass, M` and then carries both
        `Build mode` and `Storage`.

        Mutant (this test as it was first written -- `"Pass" in header and
        "M" in header`): passes while two tables name three of the four axes
        and three more are not checked at all, which is the state that let
        `build_mode` stay unnamed under a test called "one reader convention".
        """
        assert set(self.SEQUENCE_NOT_CHECKED) & set(self._rendered_tables()) \
            == set(), "a skipped table must not also be in the rendered dict"
        for name, table in self._rendered_tables().items():
            header = _cells(table)[0]
            assert header[:len(self.UNIT_PREFIX)] == self.UNIT_PREFIX, (
                name, header)
            pair = header[len(self.UNIT_PREFIX):len(self.UNIT_PREFIX) + 2]
            assert set(pair) == self.UNIT_PAIR, (name, header)

    def test_every_table_is_rectangular(self):
        """A column added to the header and not to the row shifts every cell
        after it, which renders as a plausible table of wrong numbers."""
        for name, table in self._rendered_tables().items():
            rows = _cells(table)
            header, body = rows[0], rows[1:]
            assert body, name
            for row in body:
                assert len(row) == len(header), (name, list(zip(header, row)))
