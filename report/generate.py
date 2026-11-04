#!/usr/bin/env python3
"""Generate the vector-bench report: charts, Markdown and self-contained HTML.

Runs inside an engine bench image (which already carries numpy, h5py,
matplotlib and jinja2), so producing a report does not require installing a
scientific Python stack on the machine under test.

Refuses to emit a report for a run directory with no manifest. Every conclusion
this framework can support depends on facts the manifest records — CPU model,
SIMD flags, source commits, resource limits — and a chart without them invites
exactly the over-generalisation docs/05-methodology.md warns against.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for candidate in (VB_ROOT, "/opt", os.path.dirname(os.path.abspath(__file__))):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from report import charts, loaders, scope  # noqa: E402
from report import provenance as _provenance  # noqa: E402
from report import measurement_versions as versions  # noqa: E402
from report.render import (adopt_presentation as render_adopt_presentation,  # noqa: E402
                           render_html, render_markdown)

from report.render import ENGINE_LABEL  # noqa: E402  (one table, not two)
# One reader of "which passes did this run execute", not two: the narrowing of
# the ann tree, the two sections that describe the run and §2's Validity
# notice all have to give the same answer.
from report.render import (described_passes,  # noqa: E402
                           passes_without_a_tree, recorded_passes)
# The sentinel the loader labels an unattributable recall record with. Imported
# rather than respelled -- the literal lives beside the code that applies it.
from report.loaders import UNKNOWN_PASS, pass_or_unknown  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="generate the vector-bench report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--annb-results", default=None,
                   help="ann-benchmarks results tree (default: <run-dir>/../annb)")
    p.add_argument("--datasets-dir", default=None,
                   help="dataset HDF5 directory, needed to recompute recall")
    p.add_argument("--output-dir", default=None,
                   help="where to write the report (default: <run-dir>/report)")
    p.add_argument("--title", default="Vector search benchmark")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names to report on. The ann "
                        "results tree is keyed by resource configuration, not "
                        "by corpus, so a machine that has measured two corpora "
                        "under one configuration reports both. Both are "
                        "correct and separated, but a report meant to describe "
                        "one of them should say which.")
    p.add_argument("--from-records", default=None,
                   help="regenerate from a previously merged records.jsonl "
                        "instead of reading the ops and ann-benchmarks trees. "
                        "Use when a run directory has been archived or copied "
                        "away from the machine that produced it.")
    return p.parse_args(argv)


def load_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run-manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no run-manifest.json in {run_dir}.\n"
            "A results directory without a manifest cannot be reported: the "
            "hardware, engine versions and resource limits behind the numbers "
            "are unknown, and every conclusion depends on them."
        )
    with open(path) as fh:
        return json.load(fh)


def _parse_ts(value: Any) -> Optional[float]:
    """Manifest timestamps are ISO-8601 Z; return epoch seconds or None."""
    if not value:
        return None
    try:
        import datetime
        return datetime.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


#: Which phase lands in which `summary` bucket, as a named constant rather
#: than a literal inside `summarize`, for two reasons.
#:
#: The first is that `ef_calibration` has to be ABSENT from it and absence is
#: invisible in an inline tuple. A calibration point carries `clients=1` and a
#: `qps`, so a sixth entry mapping it to "concurrency" would put up to twelve
#: single-client probes per engine into `summary["concurrency"]` and from there
#: into both the concurrency table and the concurrency chart, beside the
#: numbers this report publishes as throughput. The two are not the same
#: quantity: a calibration point is one client over `--max-queries` fresh
#: queries, a concurrency point is N clients over the shared query pool, and
#: `extra.query_pool` and `extra.query_reuse` are recorded precisely because
#: throughput over a small hot set and throughput over the whole corpus are
#: measurements of different things. The sweep's records reach the report
#: through `operating_points` instead, which is a join over them rather than a
#: phase bucket.
#:
#: The second is that the absence is then testable: a test can add that entry
#: at run time and watch the published QPS move, which is the only way to show
#: the exclusion is doing work rather than being a comment.
_PHASE_GROUPS = (
    ("index_build", "build"),
    ("concurrency", "concurrency"),
    ("filtered", "filtered"),
    ("churn", "churn"),
    ("ingest", "ingest"),
)


#: What `extra.filter_column` means when it is absent, which it is on every
#: filtered record written before PR 3: all 69 archived pgvector filtered
#: records, every record `tests/synthetic.py` builds, and anything a
#: pre-T3c harness produced. Those runs had one filter column and it was
#: `tag`, so reading absence as `tag` is a statement of fact about them
#: rather than a default -- and the alternative, bucketing them as "unknown",
#: would split one engine's `tag` curve across two series on every chart that
#: reads an archived run.
FILTER_COLUMN_WHEN_ABSENT = "tag"


def filter_column(record: Dict[str, Any]) -> str:
    """The filter column a filtered record measured."""
    return ((record.get("extra") or {}).get("filter_column")
            or FILTER_COLUMN_WHEN_ABSENT)


def _why_absent(records: List[Dict[str, Any]], dataset: str,
                workload: str) -> str:
    """Why a workload's chart has no records: not asked for, or declined.

    The harness stamps `extra.declined_workloads` on every record of a run
    whose driver declined a workload (MyVector's filtered search). Without
    this the skip line said "not in this profile's workloads" about a profile
    that DID ask, and the one place a reader would look for the reason gave
    the wrong one.
    """
    decliners = sorted({
        str(r.get("engine")) for r in records
        if r.get("dataset") == dataset
        and workload in ((r.get("extra") or {}).get("declined_workloads") or {})})
    if decliners:
        return f"declined by the driver for {', '.join(decliners)}"
    return "not in this profile's workloads"


def filtered_chart_entries(records: List[Dict[str, Any]], dataset: str,
                           draw=None) -> List[Any]:
    """The `(name, fn, payload)` chart entries for the filtered section.

    One per filter column present, or -- when there are no filtered records
    at all -- the single `filtered` entry this always had, so that the `needs`
    table still matches it and the shared "not in this profile's workloads"
    skip line still prints. Without that fallback a profile running no
    filtered workload would generate no chart and say nothing about why.

    `functools.partial`, not a lambda. A lambda written
    `lambda *a, **kw: charts.filtered(*a, column=column, **kw)` closes over
    the NAME `column`, not its value, so every entry in the list would plot
    whichever column the comprehension finished on: two filenames, one chart,
    and the `tag` curve silently gone from a report that still looks
    complete. `partial` binds the value at construction and the mistake
    cannot be written. A module-level function rather than an expression
    inside `main()` so the binding is testable without a run directory --
    `draw` is the chart function, injectable for exactly that.
    """
    from functools import partial

    from report import charts as charts_mod
    draw = draw or charts_mod.filtered
    columns = _filter_columns_present(records, dataset)
    if not columns:
        return [("filtered", draw, (records, dataset))]
    return [(f"filtered-{column}", partial(draw, column=column),
             (records, dataset))
            for column in columns]


def qps_at_recall_entries(summary: Dict[str, Any], dataset: str,
                          draw=None) -> List[Any]:
    """The `(name, fn, payload)` chart entries for §4's floor bars.

    One per resource pass once `per_dataset` is keyed by pass, because those
    bars compare engines at ONE budget: two passes on one axis is fourteen
    bars per floor on a nine-inch figure, paired only by a legend. A single
    pass yields the one `qpsatrecall` entry this always had, so its stem --
    and every published report's filename -- is unchanged.

    `functools.partial` rather than a lambda, for the reason
    `filtered_chart_entries` records above: a lambda closes over the NAME, so
    every entry would draw whichever pass the comprehension finished on, and
    the result is two filenames holding one chart.
    """
    from functools import partial

    from report import charts as charts_mod
    draw = draw or charts_mod.qps_at_recall
    # Split on the run's pass count, not on this dataset's. One dataset's ANN
    # phase failing in one pass is the normal shape -- the reference bundle has
    # exactly that, sift measured in tuned only -- and splitting per dataset
    # gave `qpsatrecall-normalized-glove…`, `qpsatrecall-tuned-glove…` and a
    # bare `qpsatrecall-sift-128-euclidean`, whose figcaption then discriminates
    # nothing. The on-figure title does not name the pass, so the stem is the
    # only discriminator there is.
    if len(described_passes(summary)) < 2:
        return [("qpsatrecall", draw, (summary, dataset))]
    present = sorted((summary.get("per_dataset", {}) or {}).get(dataset, {}),
                     key=str)
    return [(f"qpsatrecall-{resource_pass}",
             partial(draw, resource_pass=resource_pass),
             (summary, dataset))
            for resource_pass in present]


#: The axes `cli.py:_run_unit` loops over. One ops unit is one
#: (M, build_mode, storage_engine) combination inside one resource pass, and
#: each unit reloads the corpus, rebuilds the index and writes its OWN full
#: column x selectivity grid of filtered records. `main` gives pgvector two
#: units per (pass, dataset); `full` gives mariadb three and pgvector six;
#: `--resource-pass both` doubles all of them.
OPS_UNIT_FIELDS = ("m", "build_mode", "storage_engine", "resource_pass")


def ops_unit(record: Dict[str, Any]) -> tuple:
    """Which ops unit a record was measured in.

    Any derivation that compares two filtered points -- a latency ratio
    across a selectivity curve, most of all -- has to hold these constant or
    it is comparing two indexes built differently on two different machine
    allocations. Measured on a two-build-mode fixture: `post` alone gives a
    0.20x ratio and the verdict "pre-filter", `incremental` alone gives 1.00x
    and "post-filter, abandons the query", and pooling them publishes ONE row
    at 50.0x and "post-filter, exhaustive" -- a verdict neither unit
    produced, obtained by dividing one unit's 1% latency by the other's 50%.
    """
    return tuple(record.get(field) for field in OPS_UNIT_FIELDS)


#: What `extra.separable_build` means when it is absent, in the voice
#: `FILTER_COLUMN_WHEN_ABSENT` above uses and for the opposite reason.
#:
#: There, absence had ONE cause and so one honest reading. Here it has two at
#: once: every record written before the key existed is implicit ops
#: measurement version 1, and those runs contain pgvector (which bulk-builds)
#: AND MHNSW and VIDX (which do not). So there is no fact to read absence as,
#: and inventing one is the defect: `build_wall_s` is the `CREATE INDEX` call
#: for a separable build and the ENTIRE million-row load for an incremental
#: one, so filing an unmarked record under either kind puts two different
#: quantities on one axis under a lower-is-better arrow.
#:
#: `render._index_build_kind` read a falsy `separable_build` as
#: "incremental", which silently files every archived pgvector build -- a
#: bulk one -- as incremental. Hence a third value rather than a default.
#:
#: There IS a diagnostic, and it is deliberately not acted on here:
#: `build.py` writes `effective_build = load.wall_seconds` for an incremental
#: build and `ingest_wall_s = load.wall_seconds` beside it, both rounded to
#: three places, so `build_wall_s == ingest_wall_s` exactly on an incremental
#: record and essentially never on a bulk one. That is evidence, not a
#: record, and a chart that reclassified on it would be publishing an
#: inference as a measurement -- which is the thing this constant exists to
#: stop. It is written down so the next reader knows the signal is there.
BUILD_KIND_BULK = "bulk"
BUILD_KIND_INCREMENTAL = "incremental"
BUILD_KIND_UNRECORDED = "unrecorded"


def build_kind(record: Dict[str, Any]) -> str:
    """Which of the three kinds of build cost `build_wall_s` holds.

    One classifier for the chart and the table both, so the figure and the
    column beside it cannot disagree about what a row measured.
    """
    extra = record.get("extra") or {}
    if "separable_build" not in extra:
        return BUILD_KIND_UNRECORDED
    return (BUILD_KIND_BULK if extra.get("separable_build")
            else BUILD_KIND_INCREMENTAL)


def time_to_queryable_index_s(record: Dict[str, Any]) -> Optional[float]:
    """The one quantity every engine's build cost can be compared on.

    Wall clock from an empty table to an index that answers queries:

    * a BULK build paid for a plain load and then a `CREATE INDEX`, so it is
      `ingest_wall_s + build_wall_s`;
    * an INCREMENTAL build did the graph work inside the INSERTs, so it is
      `build_wall_s` alone -- which `build.py` set to `load.wall_seconds`,
      making it equal to `ingest_wall_s`. Adding them would double count the
      load, and that is not a rounding error: it would double the figure.
    * an UNRECORDED record cannot be placed, because the two formulas differ
      by a whole corpus load and nothing in the record says which applies.
      None, and the chart says how many it dropped and why.

    This is what makes both of `build.py`'s promised claims answerable from
    one axis: "pgvector's bulk build is N x cheaper than its own incremental
    build" is its two build modes on this axis, and "pgvector's incremental
    build costs X versus MHNSW's Y" is two engines on it. The bulk engine's
    `CREATE INDEX` time on its own stays available -- it is `build_wall_s` on
    a bulk record -- and is drawn in its own panel where only bulk records
    appear, so it is never beside a number that includes a load.
    """
    kind = build_kind(record)
    build = record.get("build_wall_s")
    if build is None:
        return None
    if kind == BUILD_KIND_INCREMENTAL:
        return float(build)
    if kind == BUILD_KIND_BULK:
        ingest = record.get("ingest_wall_s")
        if ingest is None:
            # A bulk record with no load time recorded is the one case where
            # the sum cannot be formed. Dropped rather than reported as the
            # build step alone, which is precisely the comparison this
            # function exists to prevent.
            return None
        return float(ingest) + float(build)
    return None


# ---------------------------------------------------------------------------
# PR 5 §5.2 lives in `report/crosscheck.py`.
#
# Re-exported here rather than left to callers to import, because the §5.2
# names were part of this module's surface for a round and `report/render.py`,
# `summarize` below and `tests/test_t5b_stats.py` all reach them through it.
# The implementations moved; where they are imported from did not.
#
# A module-scope import is safe in this direction: `crosscheck` imports
# nothing at module scope but `typing`, so this cannot pull `harness` into
# `report/`'s import graph -- which is the property `test_t5b_stats.py`
# asserts by walking both files' ASTs.
# ---------------------------------------------------------------------------
from report.crosscheck import (COVERAGE_MATCHED,  # noqa: E402,F401
                               COVERAGE_NO_ANN, COVERAGE_NO_M_OVERLAP,
                               COVERAGE_NO_OPS, COVERAGE_NOTHING_MATCHED,
                               CROSSCHECK_BASIS_SLACK,
                               CROSSCHECK_KEY_FIELDS,
                               CROSSCHECK_MIN_TOLERANCE,
                               CROSSCHECK_RECALL_BAND,
                               _append_crosscheck_row,
                               _crosscheck_qps_inversions,
                               _crosscheck_quantization, _crosscheck_sort_key,
                               _normalised_storage, build_mode_axis_engines,
                               consistency_checks, crosscheck_key)


def _filter_columns_present(records: List[Dict[str, Any]],
                            dataset: str) -> List[str]:
    """The filter columns this dataset has filtered records for, in order.

    `FILTER_COLUMNS`' order rather than first-seen order, so the list does
    not depend on whatever order the JSONL happens to be in -- which is what
    makes the generated file set reproducible for a given run.

    It does NOT decide the order they appear in the report: `render.figures()`
    iterates `sorted(chart_paths.items())`, so `filtered-ctag-<dataset>`
    renders before `filtered-tag-<dataset>` alphabetically. Changing that
    means changing how every figure in every section is ordered, which is a
    larger decision than this function should make; both charts carry the
    column in their title and caption, so a reader is not misled.
    """
    present = {filter_column(r) for r in records
               if r.get("phase") == "filtered" and r.get("dataset") == dataset}
    ordered = [c for c in ("tag", "ctag") if c in present]
    # Anything a future column adds is appended rather than dropped: a chart
    # missing entirely is harder to notice than one with an unexpected name.
    return ordered + sorted(present - set(ordered))


#: The fewest warm-up queries that can have converged: `vb_warmup` compares
#: two consecutive batches of `WARMUP_BATCH` (200), so a count under two
#: batches ended early -- a query raised, or the sample was that small --
#: and the first configuration may still have met a cold cache. Not imported
#: from the overlay, which is not on this package's path in the report image.
MIN_CONVERGED_WARMUP_QUERIES = 400


def index_warmup_flags(recall_records: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
    """One row per (engine, dataset, pass, build) whose ann index warm-up
    did not converge or ran fewer than `MIN_CONVERGED_WARMUP_QUERIES`.

    For the Validity section (#75's review): ANN_MEASUREMENT_VERSION 3 warms
    the index until two batch means agree, and a sweep whose warm-up gave up
    at the cap, or stopped before it could compare, has a first point that
    may be the cold-cache artefact the warm-up exists to remove. Records that
    predate the warm-up carry neither key and are not flagged here; the QPS
    inversion table already labels their first steps.
    """
    from report.builds import build_label
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in recall_records:
        extra = r.get("extra") or {}
        queries = extra.get("index_warmup_queries")
        converged = extra.get("index_warmup_converged")
        if queries is None and converged is None:
            continue
        short = queries is not None and queries < MIN_CONVERGED_WARMUP_QUERIES
        if converged is not False and not short:
            continue
        key = (r.get("engine"), r.get("dataset"), r.get("resource_pass"),
               build_label(r))
        seen.setdefault(key, {
            "engine": r.get("engine"), "dataset": r.get("dataset"),
            "resource_pass": r.get("resource_pass"), "build": key[3],
            "queries": queries, "converged": converged,
            "seconds": extra.get("index_warmup_seconds")})
    return [seen[k] for k in sorted(seen, key=str)]


def summarize(records: List[Dict[str, Any]],
              manifest: Optional[Dict[str, Any]] = None,
              memory: Optional[Dict[str, List[Dict[str, Any]]]] = None
              ) -> Dict[str, Any]:
    """Headline figures the narrative sections are written from."""
    from report import builds as builds_mod
    from report.loaders import pareto_frontier

    recall_records = [r for r in records if r.get("phase") == "recall_qps"]
    datasets = sorted({r["dataset"] for r in records if r.get("dataset")})
    engines = sorted({r["engine"] for r in records if r.get("engine")})

    summary: Dict[str, Any] = {
        "datasets": datasets,
        "engines": engines,
        # Which engines have a recall curve at all, for a §3 row that is only
        # true when one exists (`render._build_mode_rows`'s Percona row).
        "ann_engines": sorted({r["engine"] for r in recall_records
                               if r.get("engine")}),
        "per_dataset": {},
        "plan_failures": [],
        "short_result_cases": [],
        "failed_phases": [],
        "memory_pressure": [],
        "stale_ann": [],
        # Passes this run executed whose recall this report does not carry,
        # so §2 Validity can say so inside the document.
        #
        # An INTERSECTION, and the manifest half alone was a bug. The manifest
        # says which passes have no locatable tree; it cannot say what the
        # report ended up reading. Both whole-tree fallbacks in
        # `_narrow_to_this_run` hand the loader the un-narrowed root, where
        # `_pass_from_path` recovers the real pass from the directory layout --
        # so §4 published a pass's recall under §2's bold claim that the same
        # pass's "recall and throughput measurements are absent from every
        # chart and table below". A reader who believed §2 then read a
        # normalized table as tuned's, which is the misattribution this whole
        # change exists to end, with Validity vouching for it.
        #
        # Not circular, which is what the comment here used to get wrong. A
        # pure-records derivation would be: "absent from the records", proved
        # by the records. The intersection is not, because the two halves are
        # independent -- `phases[]` supplies "this pass ran" from the
        # orchestrator's own log, and the records supply "and its recall is not
        # here". A pass that ran and contributed no recall record is still
        # named; a pass whose recall this report prints no longer is.
        #
        # `recall_records`, not `records`: the notice's claim is specifically
        # about recall and throughput, and it says in the next breath that the
        # ops measurements are unaffected. An ops record for the pass must not
        # silence it.
        "passes_without_a_tree": [
            p for p in passes_without_a_tree(manifest or {})
            if p not in {r.get("resource_pass") for r in recall_records}],
        "duplicate_ann": [],
        "recall_floor_gaps": [],
        "silent_ann_failures": [],
        "qps_inversions": [],
        "contended_phases": [],
        "filter_strategies": [],
        # Points the filtered budget stopped early. A record that measured 93
        # of 500 planned queries is a real measurement of a slow
        # configuration, not a failure -- but its recall carries the variance
        # of 93 queries and the report must not draw it as though it carried
        # 500. `insufficient_sample` is the stricter flag of the two: fewer
        # than 50 queries finished.
        "time_bounded_points": [],
        "passes": sorted({r.get("resource_pass") for r in records
                          if r.get("resource_pass")}),
        # Every pass this report describes, computed ONCE and carried, because
        # §1, §2, §4's headings and the floor chart's stem all have to agree
        # about how many there are. They did not: §4 derived its own count
        # from `per_dataset`, which holds only the passes that produced recall
        # for one dataset, so a report whose §1 named two passes printed §4
        # under a bare single-pass heading -- and the floor chart qualified
        # glove's stem and not sift's, where only one pass had an ANN phase.
        # Deriving one number in two files is the defect class this whole
        # change is about.
        #
        # TWO readers share it, not four: §4's headings
        # (`render._headline_tables`) and the floor chart's stem
        # (`qps_at_recall_entries`). An earlier version of this comment
        # claimed §1's row and §2's notice did too, and they do not -- §1
        # derives from `_pass_resources(manifest)` and the single-valued
        # `resource_pass`, and §2 reads `summary["passes_without_a_tree"]`.
        # Those two are correct as they are; they are simply not this number's
        # readers, and saying they were turned a narrow true claim into a
        # broad false one.
        #
        # `render._pass_note` and `main`'s `passes_present` derive their own
        # again, and disagree with these in grain: a dataset measured in one
        # pass inside a two-pass run loses its `passcompare` figure with
        # nothing said. Pre-existing on both sides and left alone
        # deliberately; folding them in means deciding what a per-dataset pass
        # set should mean for a run-level chart, which is a behaviour question
        # and not a de-duplication.
        #
        # The UNION of the manifest's passes and the records', not the
        # manifest's alone: `--from-records` over an archive carries no
        # manifest at all, and records spanning two passes must still qualify
        # the headings. `recorded_passes` already unions `config["passes"]`,
        # the single-valued pair and `phases[]`, so a single-pass manifest and
        # an archived one with no `passes` key both still come out at one --
        # which the byte-identical single-pass path depends on.
        # TWO record terms, and dropping either one regresses something.
        #
        # The first is any record naming a pass, whatever its phase. The
        # reference bundle measures ops in both passes and recall in one, so
        # on a manifest that names no passes -- a legacy single-valued pair
        # with no `phases[]`, no manifest passes at all, or an empty
        # `phases[]` -- the ops records are the only evidence the run had two,
        # and without this term §4's recall heading and the floor-chart stem
        # went bare on exactly those three shapes. That is the defect this
        # change is named for, reintroduced by a narrowing meant only to
        # normalise. It also kept `main`'s `passes_present` and this count
        # agreeing on those shapes: without it `passcompare` drew a
        # normalized-vs-tuned chart above a bare single-pass heading.
        #
        # The second runs recall records through `pass_or_unknown`, so this
        # count and the keys `per_dataset` is built on cannot disagree about
        # whether an unnamed pass exists -- a falsy name dropped from the
        # count while the table kept it as a key is how §4 came to print two
        # tables for one engine under one heading. A manifest never carries an
        # unnamed pass, so only records need normalising.
        "described_passes": sorted(
            {name for name, _fp in recorded_passes(manifest or {})}
            | {r.get("resource_pass") for r in records
               if r.get("resource_pass")}
            | {pass_or_unknown(r.get("resource_pass"))
               for r in records if r.get("phase") == "recall_qps"}, key=str),
    }

    # Two results for one configuration means the report is reading more than
    # one measurement tree. Timestamps can be lost by copying a run between
    # machines, so this catches the same problem without depending on them: a
    # 16 GB curve and a 64 GB curve were merged into one chart, and the charts
    # silently showed whichever was faster at each point.
    from report.loaders import build_signature
    by_config: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recall_records:
        # The key must name every axis a profile is allowed to sweep, or a
        # legitimate curve looks like a duplicate. Listing them by hand failed
        # twice: once when the tuned pass began sweeping storage engine and
        # ef_construction, flagging 16 real measurements as repeats, and again
        # when Percona Search gained a quantization axis that is not a column
        # at all, flagging 8 more. build_signature takes the configuration from
        # the result filename instead, which covers axes this code has never
        # heard of.
        #
        # The RESOURCE PASS is named explicitly, because `build_signature`
        # cannot see it. It reads `extra.source_file`, which is the result
        # file's path relative to the tree it was read from -- and the trees
        # are per pass, so one build configuration measured in both passes
        # produces the SAME relative path and therefore the same signature.
        # Once the report began reading both trees that stopped being an error
        # path and became the normal one: a legitimate `--resource-pass both`
        # run reported 123 of its own measurements as duplicates of each
        # other, under prose telling the reader to delete one of the trees.
        # Two passes are two budgets, which is the same reasoning the filtered
        # ratio above records for its own key.
        by_config.setdefault(
            (build_signature(r), r.get("resource_pass"),
             r.get("ef_search")), []).append(r)
    for key, group in sorted(by_config.items(), key=lambda kv: str(kv[0])):
        if len(group) > 1:
            first = group[0]
            summary["duplicate_ann"].append({
                "engine": first.get("engine"), "dataset": first.get("dataset"),
                "m": first.get("m"), "ef_search": key[2],
                # The pass, and the other two axes `build_signature` carries
                # inside the filename it keys on: two rows can differ on any
                # of them, and a table that prints neither reads as the same
                # configuration listed twice -- which is the exact misreading
                # this section exists to report.
                "resource_pass": first.get("resource_pass"),
                "build_mode": first.get("build_mode"),
                "storage_engine": first.get("storage_engine"),
                "ef_construction": first.get("ef_construction"),
                "count": len(group),
                "qps": sorted(round(g.get("qps") or 0, 1) for g in group),
            })

    # Throughput that rises as query effort rises. Examining more candidates
    # cannot be faster on the same index, so an inversion is never the engine
    # -- it is the machine, and the measurement describes whatever else was on
    # it. AliSQL measured 2.6 QPS at ef_search 400 and 23.3 at 800 in one
    # process against one index, and nobody noticed for five days because
    # nothing looked at the series as a series.
    #
    # The warm-up artifact lands here too, and should: the first configuration
    # of a sweep pays for a cold cache and reads as an inversion of about
    # 1.8x. Both are reported, with the ratio, because the ratio is what
    # separates a cold cache from a contaminated window.
    by_series: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recall_records:
        if r.get("qps") is None or r.get("ef_search") is None:
            continue
        # Per (configuration, pass), for the reason `by_config` above records
        # at length: `build_signature` is blind to the pass, so two passes'
        # curves merged into one series and every normalized point paired
        # against a tuned point at higher effort read as throughput rising
        # with effort. On a legitimate two-pass run that invented 97
        # inversions, 71 of them past the first step, and presented them as
        # "measurements of something other than the engine".
        by_series.setdefault(
            (build_signature(r), r.get("resource_pass")), []).append(r)
    for key, group in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(group, key=lambda r: r["ef_search"])
        # An index the overlay warmed before its first configuration (see
        # overlay/.../vb_warmup.py, ANN_MEASUREMENT_VERSION 3) has no cold
        # first point left to blame, so a first-step inversion on it is not
        # the warm-up artifact and must not be labelled as one.
        warmed = any((r.get("extra") or {}).get("index_warmup_queries")
                     for r in group)
        for previous, current in zip(ordered, ordered[1:]):
            if current["qps"] <= previous["qps"]:
                continue
            first = previous["ef_search"] == ordered[0]["ef_search"]
            summary["qps_inversions"].append({
                "engine": previous.get("engine"),
                "dataset": previous.get("dataset"),
                "storage_engine": previous.get("storage_engine"),
                "ef_construction": previous.get("ef_construction"),
                "from_ef": previous["ef_search"], "to_ef": current["ef_search"],
                "from_qps": round(previous["qps"], 1),
                "to_qps": round(current["qps"], 1),
                "ratio": round(current["qps"] / previous["qps"], 2),
                "first_in_sweep": first,
                "index_warmed": warmed,
                "likely": ("warm-up" if first and not warmed
                           else "unexplained"),
            })
    summary["qps_inversions"].sort(key=lambda i: -i["ratio"])

    # ann-benchmarks skips configurations that already have result files, and
    # reports that as success. A re-run after a config change therefore returns
    # instantly with the previous numbers, and nothing in the records says so.
    started = _parse_ts((manifest or {}).get("started_at"))
    if started:
        for r in recall_records:
            # The loader stores this under `extra`, alongside source_file.
            # Reading it from the top level meant the check never fired once.
            mtime = (r.get("extra") or {}).get("source_mtime") or r.get("source_mtime")
            if mtime and mtime < started:
                summary["stale_ann"].append({
                    "engine": r.get("engine"),
                    "dataset": r.get("dataset"),
                    "measured_at": mtime,
                    "run_started": started,
                    "source_file": (r.get("extra") or {}).get("source_file"),
                })

    # An engine that spent the phase against its cgroup limit was reclaiming
    # continuously, so its numbers describe the memory budget and not the
    # implementation. Nothing in the records shows this.
    #
    # The limit is the series' own engine's, from `engines_resolved`, and not
    # `config.resolved_resources` -- that key is whichever engine
    # `set_config` saw last (mongodb's in one `full-20260825` snapshot,
    # AliSQL's in another). Older manifests fall back to it, and the row says
    # so for §2 to state.
    for name, rows in sorted((memory or {}).items()):
        from report.loaders import ceiling_pressure
        limit, limit_source = _provenance.series_memory_limit(manifest, name)
        pressure = ceiling_pressure(rows, limit)
        if not pressure:
            continue
        pressure["limit_source"] = limit_source
        engine = name.split("-", 1)[0]
        pressure["engine"] = engine
        pressure["series"] = name
        summary["memory_pressure"].append(pressure)

    # CPU the machine spent that the container being measured did not. A phase
    # that ran under someone else's job is not a measurement of the engine, and
    # until both counters were sampled there was no way to say so.
    for name, rows in sorted((memory or {}).items()):
        from report.loaders import foreign_cpu
        contention = foreign_cpu(rows)
        if not contention:
            continue
        contention["series"] = name
        contention["engine"] = name.split("-", 1)[0]
        summary["contended_phases"].append(contention)
    summary["contended_phases"].sort(key=lambda c: -c["foreign_cores"])

    # Any measurement taken while the vector index was NOT in the plan is a
    # full scan, and must be surfaced rather than averaged into a curve.
    for r in records:
        if r.get("vector_index_used") is False:
            summary["plan_failures"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "phase": r.get("phase"), "m": r.get("m"),
                "ef_search": r.get("ef_search"),
                "selectivity": r.get("selectivity"),
                # `explain_uses_vector_index` runs per (column, selectivity)
                # since PR 3 §3.3 item 2, because the MySQL-family optimizers
                # can switch to `ctag_idx` or to a full scan at a different
                # threshold than they do for `tag`. A plan failure that does
                # not say which column it was asked about names the wrong
                # index to go and look at.
                "filter_column": (filter_column(r)
                                  if r.get("phase") == "filtered" else None),
            })
        if (r.get("extra") or {}).get("time_bounded"):
            extra = r.get("extra") or {}
            summary["time_bounded_points"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "filter_column": filter_column(r),
                "selectivity": r.get("selectivity"),
                "resource_pass": r.get("resource_pass"),
                "build_mode": r.get("build_mode"),
                "queries_executed": r.get("queries_executed"),
                "queries_planned": extra.get("queries_planned"),
                "budget_s": extra.get("filtered_budget_s"),
                "insufficient_sample": bool(extra.get("insufficient_sample")),
            })
        if (r.get("extra") or {}).get("returned_fewer_than_k"):
            summary["short_result_cases"].append({
                "engine": r.get("engine"), "dataset": r.get("dataset"),
                "selectivity": r.get("selectivity"),
                # Without the column, `tag` at 10% and `ctag` at 10.85% are
                # two rows differing only in a selectivity a reader cannot
                # attribute to a predicate.
                "filter_column": filter_column(r),
                # Without these, four measurements of one engine are one row
                # printed four times and the table reads as broken.
                "resource_pass": r.get("resource_pass"),
                "build_mode": r.get("build_mode"),
                "queries": (r.get("extra") or {}).get("short_result_queries"),
            })

    # Per (dataset, pass), and within that per engine. A frontier is a claim
    # about one grid on one machine: `min_recall`, `max_recall`, `points` and
    # `storage_engines` are all taken across the whole point set, so pooling
    # two resource passes produced a row whose sample count was the sum of two
    # machines' and whose recall range spanned both. Measured on a two-pass
    # run: Percona 24 points became 48, pgvector 18 became 36, and §4 carried
    # no column to say so.
    # Normalised, so a record naming no pass keys one table called `unknown`
    # rather than one called `None` -- the same word the count, the chart stem
    # and the legend use. See `loaders.pass_or_unknown`.
    passes = sorted({pass_or_unknown(r.get("resource_pass"))
                     for r in recall_records}, key=str)
    for dataset in datasets:
        for resource_pass in passes:
            per_engine: Dict[str, Any] = {}
            for engine in engines:
                points = [
                    r for r in recall_records
                    if r["dataset"] == dataset and r["engine"] == engine
                    and pass_or_unknown(
                        r.get("resource_pass")) == resource_pass]
                # Ordered so an exact tie names the smaller build, whatever
                # order the result files were walked in.
                points = builds_mod.smaller_builds_last(points)
                frontier = pareto_frontier(points)
                if not frontier:
                    continue
                per_engine[engine] = _floor_entry(
                    points, frontier, records, manifest or {}, engine,
                    dataset, resource_pass, summary["recall_floor_gaps"])
            if per_engine:
                # Keyed by the pair, and `render._headline_tables` emits one table
                # per pass with the pass in its heading. The pass is a property of
                # the whole table rather than of a row, because every figure in a
                # row is derived across that table's point set -- which is also
                # why §4 differs from §7-§10, whose rows are individual
                # measurement windows and so carry a Pass cell instead. A scope
                # that belongs to the table belongs in its heading, where a reader
                # cannot drop it; as a cell, an engine measured twice becomes two
                # rows with the same label and the qualifier has to be carried by
                # eye. The prose under that table is also written in the singular
                # about "the swept grid" and closes "this run pins M=16 for every
                # engine": true per table, a per-row read under a column.
                #
                # The cross-pass comparison this does NOT provide is already
                # built: `charts.pass_comparison` exists because that question
                # "had to be reconstructed by hand from two tables", and its first
                # dimension is recall throughput. A Pass column here would put
                # that reconstruction back, one section earlier.
                summary["per_dataset"].setdefault(dataset, {})[resource_pass] = \
                    per_engine

    # Which filtering strategy each engine actually used, derived from how its
    # latency moves when the predicate narrows.
    #
    # A pre-filter compares only the qualifying vectors, so a ten-times
    # narrower predicate is roughly ten times less work. Anything that walks
    # the graph and applies the predicate during or after the walk pays more,
    # because it has to travel further to find k survivors. The direction of
    # the change separates them, and it is the one thing here that does not
    # depend on trusting a vendor's description.
    #
    # This exists because the report asserted for weeks that Valkey "chooses
    # its filtering strategy per query", taken from its documentation. The run
    # showed 18.2 ms at 10% selectivity and 120.6 ms at 1% -- six times more
    # work for a tenth of the candidates, which is inline filtering and not a
    # planner choosing to pre-filter.
    # The key has to hold constant everything that changes the answer, because
    # the derivation takes the narrowest and the widest point of a series and
    # divides their latencies. Anything pooled into one series is a pair of
    # points that may never have run together, and the result is published as
    # a verdict about an engine.
    #
    # Four kinds of confound, all reachable in configured profiles:
    #
    # * The filter COLUMN. A series holding both can pair a `ctag` point
    #   against a `tag` one -- two different predicates over two different
    #   qualifying sets. `tag` at 1% and `ctag` at 1.9% (the share a nominal
    #   1% actually delivers on the 2,000-row fixture) are not two points on
    #   one curve.
    # * The iterative-scan MODE. pgvector with index iteration off and with it
    #   on walk different amounts of graph, which is the entire reason the
    #   mode is recorded.
    # * The OPS UNIT -- M, build mode, storage engine. `_run_unit` loops
    #   `m_values x build_modes x storage_engines` and each pass through it
    #   reloads the corpus, rebuilds the index and writes its own full grid.
    #   pgvector's `post` and `incremental` builds are different indexes;
    #   MariaDB's InnoDB and MyISAM are different storage. Measured, one
    #   engine and one column over two build modes: 0.20x "pre-filter" and
    #   1.00x "post-filter, abandons the query" separately, against a pooled
    #   50.0x "post-filter, exhaustive" that neither produced.
    # * The RESOURCE PASS. normalized and tuned give the engine different
    #   memory and cpusets; pairing a normalized 1% point against a tuned 50%
    #   one published a ratio of 100.0x on the same fixture.
    #
    # The report has already been wrong about exactly this class of thing
    # once, asserting for weeks that Valkey "chooses its filtering strategy
    # per query" on the strength of its documentation.
    by_engine: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        if (r.get("phase") == "filtered" and r.get("selectivity")
                and r.get("latency_p50_ms")):
            key = ((r.get("engine"), r.get("dataset"), filter_column(r),
                    (r.get("extra") or {}).get("iterative_scan"))
                   + ops_unit(r))
            by_engine.setdefault(key, []).append(r)
    for key, points in sorted(by_engine.items(), key=lambda kv: str(kv[0])):
        engine, dataset, column, scan_mode = key[:4]
        # Zipped rather than unpacked positionally, so the key can carry
        # however many axes `OPS_UNIT_FIELDS` names. A fixed-length unpack
        # here made the obvious regression -- dropping an axis from the key --
        # raise a ValueError instead of publishing a wrong ratio, which is a
        # mutant that cannot be injected and therefore a guard that cannot be
        # proven.
        unit = dict(zip(OPS_UNIT_FIELDS, key[4:]))
        if len(points) < 2:
            continue
        points.sort(key=lambda r: r["selectivity"])
        narrow, wide = points[0], points[-1]
        ratio = narrow["latency_p50_ms"] / wide["latency_p50_ms"]
        span = wide["selectivity"] / narrow["selectivity"]
        # Recall is what separates "gives up" from "keeps looking": an engine
        # that returns quickly because it stopped early shows it here.
        collapsed = (narrow.get("recall_at_k") or 1) < 0.5
        if ratio < 0.9:
            observed = "pre-filter"
        elif collapsed:
            observed = "post-filter, abandons the query"
        elif ratio < 10:
            observed = "filters during the graph walk"
        else:
            observed = "post-filter, exhaustive"
        summary["filter_strategies"].append({
            "engine": engine, "dataset": dataset,
            "filter_column": column, "iterative_scan": scan_mode,
            # Carried so the rendered row can say which unit it describes.
            # A table that cannot tell two units apart prints the same row
            # twice and reads as a rendering fault -- which is the reason the
            # `short_result_cases` block beside this one carries them.
            # `.get`, not `[...]`: an axis absent from OPS_UNIT_FIELDS is
            # reported as None rather than crashing, so the rendered table
            # shows a blank cell where the pooling happened.
            "m": unit.get("m"), "build_mode": unit.get("build_mode"),
            "storage_engine": unit.get("storage_engine"),
            "resource_pass": unit.get("resource_pass"),
            "narrow": narrow["selectivity"], "wide": wide["selectivity"],
            "narrow_p50": narrow["latency_p50_ms"],
            "wide_p50": wide["latency_p50_ms"],
            "narrow_recall": narrow.get("recall_at_k"),
            "ratio": round(ratio, 2), "span": round(span, 1),
            "observed": observed,
        })
    summary["filter_strategies"].sort(
        key=lambda s: (s["ratio"], s["engine"] or "", s["filter_column"],
                       str(s["resource_pass"]),
                       # Numeric, so M 8/16/32 does not come out 16, 32, 8 --
                       # which `full`'s `m_values: [8, 16, 32]` would produce.
                       s["m"] or 0,
                       str(s["build_mode"]), str(s["storage_engine"])))

    # PR 4 §4.3: one row per during-writes record, with the p99-under-writes
    # ratio recomputed from that record's own two percentiles. Built in
    # `report/churn_state.py` rather than in `render.py` for the reason
    # `filter_strategies` above is: the grouping that keeps two ops units
    # apart is the correctness property, and it belongs in one place that a
    # test can mutate.
    #
    # Imported in the function body, not at module scope: `churn_state` reads
    # `OPS_UNIT_FIELDS` and `ops_unit` from here, so a top-level import either
    # way round is a cycle. The same shape `filtered_chart_entries` uses for
    # `report.charts`.
    from report.churn_state import during_writes_rows
    summary["churn_during_writes"] = during_writes_rows(records)

    # Build-cost and concurrency headlines.
    for phase, key in _PHASE_GROUPS:
        summary[key] = [r for r in records if r.get("phase") == phase]

    # PR 2 §2.3.1: the ef each calibrated unit settled on, with the recall and
    # the status that justify it. Built here rather than in `render.py` so the
    # concurrency chart and the operating-point table are drawn from one
    # derivation -- a chart marking an engine `floor_above_target` beside a
    # table that does not is worse than neither.
    from report.loaders import calibration_targets, operating_points
    summary["operating_points"] = operating_points(records)
    summary["calibration_targets"] = calibration_targets(records)
    # The cross-engine concurrency headline, built from the same operating
    # points so a unit that missed its recall target is unranked here and
    # marked in the table by one derivation. Client-bound points never set a
    # peak; see `report/operating.py`.
    from report.operating import concurrency_peaks
    summary["concurrency_peaks"] = concurrency_peaks(
        records, summary["operating_points"])
    # Ops version 9: the server's CPU and pressure over each concurrency
    # window, joined by timestamp from the `mem-*.jsonl` samples. Beside the
    # Client CPU column, so a flattening curve says whose ceiling it is.
    from report.server_windows import server_windows
    summary["server_windows"] = server_windows(records, memory, manifest)
    # And the ann warm-up that did not converge or ran short (#75).
    summary["index_warmup_flags"] = index_warmup_flags(recall_records)

    # PR 5 §5.2. Built here for the reason `filter_strategies` and
    # `churn_during_writes` are: the matching rule is the correctness
    # property and it belongs in one place a test can mutate, not spread
    # between the table that prints it and the chart beside it.
    #
    # Always set, never conditional on there being matches. §5.2 item 3 says
    # "An empty match list is itself reported, not hidden", and a key that is
    # absent when empty makes the renderer's "no matches" branch
    # indistinguishable from an older summary that never had the key.
    summary["consistency"] = consistency_checks(records)

    # An engine whose ann phase ran and wrote nothing.
    #
    # ann-benchmarks catches a per-algorithm exception and exits zero, so a
    # module that raises leaves the phase marked completed and the engine
    # simply absent from the recall comparison. Three runs of Percona Search
    # failed that way before anyone noticed the engine was missing rather than
    # slow, because nothing distinguished "did not run" from "ran and produced
    # no measurement".
    measured = {r.get("engine") for r in recall_records}
    for phase in _provenance.latest_attempts((manifest or {}).get("phases")):
        engine = phase.get("engine")
        if (phase.get("phase") == "ann" and phase.get("status") == "completed"
                and engine and engine not in measured):
            summary["silent_ann_failures"].append({
                "engine": engine,
                "dataset": phase.get("dataset"),
                "resource_pass": phase.get("resource_pass"),
                "duration_s": phase.get("duration_s"),
            })

    # A phase that failed leaves a hole in the results. Without this the report
    # simply omits that engine/pass and a reader cannot distinguish "not
    # measured" from "measured and unremarkable".
    # A unit the orchestrator refused up front (`cli._memory_refusal`: the
    # build would not fit the server limit) is neither: it carries its reason
    # and is reported as a refusal, not as a crash.
    summary.setdefault("refused_phases", [])
    # The last attempt per unit: a failure a later attempt fixed is history,
    # not a hole in the results.
    for phase in _provenance.latest_attempts((manifest or {}).get("phases")):
        if phase.get("status") == "refused":
            summary["refused_phases"].append(phase)
        elif phase.get("status") != "completed":
            summary["failed_phases"].append(phase)

    return summary



def _floor_entry(points: List[Dict[str, Any]],
                 frontier: List[Dict[str, Any]],
                 records: List[Dict[str, Any]], manifest: Dict[str, Any],
                 engine: str, dataset: str, resource_pass: Any,
                 gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """§4's figures for one engine: the swept frontier's, and the ops build's.

    QPS at a recall floor is the comparison an operator actually makes: "how
    fast is it, at accuracy I can accept?" -- over every configuration they
    could choose, because identically named parameters are not identical
    work across engines (`report/frontier.py` has the reviewer's argument).
    So the unprefixed `qps_at_recall_N` fields -- the headline, the floor
    chart, `recall_floor_gaps` -- are the frontier over EVERY build swept in
    this (dataset, pass), and each names the build it came from (`_build`,
    the `build_signature`; `_build_label`, the words printed under it) and
    whether that is the build the ops pass measured (`_is_ops_build`).

    The `reference_` fields are the same floors at the reference build --
    the index the ops pass builds (`report/builds.py` says where it comes
    from) -- for the secondary table. Before this change those were the
    unprefixed fields and the sweep's were `best_`; `best_` is kept as an
    alias of the headline so a reader of an older summary layout finds the
    same numbers under the name it knew. `full-20260825`'s lesson is kept by
    the labels and the marker, not by narrowing the headline: its table
    printed 138 / 124 / 88 for Percona Search from the unquantized index
    with nothing saying so.
    """
    from report import builds, frontier as frontier_mod
    from report.loaders import pareto_frontier

    build, source = builds.reference_build(points, records, manifest, engine,
                                           dataset, resource_pass)
    reference = ([p for p in points if builds.matches(p, build)]
                 if build is not None else [])
    reference_frontier = pareto_frontier(reference)
    ops_builds = [r for r in records
                  if r.get("phase") == "index_build"
                  and r.get("engine") == engine
                  and r.get("dataset") == dataset
                  and pass_or_unknown(r.get("resource_pass")) == resource_pass]
    space = frontier_mod.swept_space(points)
    # The ops build as far as a point can be compared with it: None when it
    # is unknown, names an unrecorded server default, or matches no swept
    # point -- "cannot be said", never "different" (`usable_reference`).
    comparable = frontier_mod.usable_reference(build, points)

    entry: Dict[str, Any] = {
        "reference_build": build,
        "reference_build_label": builds.spell_build(build),
        "reference_source": source,
        "reference_points": len(reference),
        "points": len(points),
        "sweep_points": len(points),
        "sweep_builds": space["builds"],
        # What was actually swept, so a frontier from one build is
        # called a curve and a reader can see which axes were explored.
        "swept_space": space,
        # Which storage engine reached each floor, not only how fast. The
        # frontier spans every configuration swept, so on the tuned pass
        # "MariaDB 12.3: 1,176 QPS" was a MyISAM result -- printing it
        # unattributed credits it to the default build.
        "storage_engines": sorted({p.get("storage_engine") for p in points
                                   if p.get("storage_engine")}),
    }
    entry.update(builds.floor_figures(frontier, ""))
    frontier_mod.ops_build_flags(entry, frontier, comparable)
    entry.update(builds.floor_figures(frontier, "best_"))
    entry.update(builds.floor_figures(reference_frontier, "reference_"))
    entry.update(frontier_mod.smallest_figures(points, ops_builds,
                                               comparable))
    entry["frontier_builds"] = frontier_mod.frontier_builds(
        points, frontier, ops_builds, comparable)
    # The reference build's ceiling, which is what the secondary table's "—"
    # footnote prints: "its reference build peaks at 0.892" is the sentence
    # the 25 Aug post needed and did not have.
    entry["reference_max_recall"] = (
        max((p.get("recall_at_k") or 0) for p in reference)
        if reference else None)
    entry["reference_min_recall"] = (
        min((p.get("recall_at_k") or 0) for p in reference)
        if reference else None)
    # The floor columns compare engines at an accuracy the operator accepts,
    # which is only a comparison if both engines have measurements near it.
    # ef_search cannot go below k and MHNSW exposes no ef_construction, so
    # with M pinned there is no MariaDB configuration returning recall below
    # about 0.975: its 0.90 and 0.95 figures are one measurement printed
    # twice. Recording the lowest recall reached is what lets the report say
    # so -- over every build, because that is what the headline cells span.
    entry["min_recall"] = min((p.get("recall_at_k") or 0) for p in points)
    entry["max_recall"] = max((p.get("recall_at_k") or 0) for p in frontier)
    entry["sweep_min_recall"] = entry["min_recall"]
    entry["sweep_max_recall"] = entry["max_recall"]
    for floor in builds.FLOORS:
        key = f"qps_at_recall_{int(floor * 100)}"
        if entry[key] is None:
            continue
        if entry["min_recall"] > floor:
            gaps.append({
                "engine": engine, "dataset": dataset, "floor": floor,
                "lowest_recall": entry["min_recall"],
                "measured_at": entry[f"{key}_recall"],
                "qps": entry[key],
            })
    return entry


def attach_peak_rss(records: List[Dict[str, Any]],
                    peaks: Dict[str, Optional[int]]) -> None:
    """Fill in peak server memory from the sampled timeseries, in place.

    Matched on the exact series name rather than a prefix. A prefix of
    engine-dataset-pass matches both `...-m16-post` and `...-m16-post-myisam`,
    so MariaDB's InnoDB and MyISAM builds were both given whichever series came
    first: the InnoDB row was published at 13.9 GiB against a true 28.7.

    A value derived here is tagged as such, so regenerating a report from a
    merged records.jsonl recomputes it rather than trusting it. A value the
    harness read from the kernel's high-water mark carries no tag and is left
    alone: polling a timeseries can miss a peak that memory.peak recorded.
    """
    from report.loaders import memory_stem

    for r in records:
        if r.get("phase") != "index_build":
            continue
        derived = (r.get("extra") or {}).get("peak_rss_source") == "sampled"
        if r.get("peak_rss_bytes") and not derived:
            continue
        peak = peaks.get(memory_stem(r))
        if peak:
            r["peak_rss_bytes"] = peak
            r.setdefault("extra", {})["peak_rss_source"] = "sampled"


def _annb_root(annb_dir: str,
               recorded: List[Tuple[str, Optional[str]]]) -> str:
    """The shared annb root, even when the caller already narrowed to one tree.

    `orchestrator/cli.py` narrows `--annb-results` to
    `/results/annb/<pass>/<fingerprint>` on the host, because that is where
    `ann_fingerprint` is importable. It can only ever name one tree -- it
    reads the single-valued keys -- so a container handed that path cannot see
    the other pass's results at all, however many passes this manifest
    records. The whole results directory is mounted either way, so the other
    tree is right there, two levels up.

    Two conditions, and the second exists because the first is only about
    **how the path is spelled**. A directory genuinely rooted at
    `.../<pass>/<fingerprint>/` -- reachable by hand-passing
    `--annb-results`, never by `cli.py` -- matches the spelling test, and
    climbing out of it made the report declare a pass whose tree was right
    there absent. So the contents are consulted too: a real annb root has a
    child directory named for a pass, and a narrowed tree's children are
    datasets. A root that is spelled like a narrowed tree *and* holds no pass
    directory can still be mis-climbed, and has nothing to lose by it.
    """
    trimmed = annb_dir.rstrip(os.sep)
    parent, base = os.path.split(trimmed)
    spelled_like_a_tree = bool(base) and any(
        base == fingerprint and os.path.basename(parent) == name
        for name, fingerprint in recorded if fingerprint)
    holds_a_pass_directory = any(
        os.path.isdir(os.path.join(trimmed, name)) for name, _fp in recorded)
    if spelled_like_a_tree and not holds_a_pass_directory:
        return os.path.dirname(parent) or os.sep
    return annb_dir


def _narrow_to_this_run(annb_dir: str,
                        manifest: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Every tree this run wrote, as (directory, resource pass) pairs.

    Results are stored under <pass>/<config fingerprint>. Walking the parent
    would pull in every configuration ever measured on this machine, which is
    how a 16 GB curve ended up in a report whose manifest said 64 GB. Falls
    back to the whole tree for older runs that predate the fingerprint, where
    the staleness check below is the only guard.

    One pair per recorded pass, because a run of both passes writes two trees
    and narrowing to one of them silently dropped the other: of the 42 hours
    `full-20261015-215006` spent measuring, the report omitted the 13.1 hours
    of normalized recall phases entirely, and printed the tuned machine's
    limits above the normalized rows the ops path had still collected.

    A tree the manifest names but disk does not have is reported and skipped;
    the trees that are present are still read, because one pass's results
    being gone is not a reason to lose the other's.

    The pass travels beside the directory so the loader can label records that
    carry no `resource_pass` attribute. `_pass_from_path` recovers it from the
    layout, which only works relative to the root -- narrowing to a tree is
    precisely what blinds it.
    """
    recorded = recorded_passes(manifest)
    root = _annb_root(annb_dir, recorded)
    named = [(name, fp) for name, fp in recorded if fp]
    if not named:
        # Warned, not silent. Every pass this run executed is unnarrowable
        # here -- the legacy class, a manifest predating `ann_fingerprint` --
        # so the whole shared tree is read, which is the cross-configuration
        # risk. This branch returned without a word, which left stdout silent
        # while §2 spoke, and the two were not even describing the same thing.
        _warn_about_unrecorded_passes(manifest)
        return [(annb_dir, UNKNOWN_PASS)]

    trees: List[Tuple[str, str]] = []
    missing: List[str] = []
    for name, fingerprint in named:
        candidate = os.path.join(root, name, fingerprint)
        if os.path.isdir(candidate):
            print(f"[report] ann-bench tree: {name}/{fingerprint}")
            trees.append((candidate, name))
        else:
            missing.append(f"{name}/{fingerprint}")

    if not trees:
        # Unchanged for a single pass, which is the only case that could reach
        # this before: every tree the manifest names is gone, so there is
        # nothing to narrow to and the staleness check in Validity takes over.
        print(f"[report] no tree at {', '.join(missing)}; reading the whole "
              f"ann-benchmarks directory. Results older than this run will be "
              f"flagged in Validity.")
        # Before the return, not after it. This branch is exactly the state an
        # archived bundle is in -- a manifest naming a pass whose tree is not
        # on this machine -- and the warning used to be unreachable there,
        # because it sat below. No test saw it: `_archived_report` builds both
        # trees, so the archived fixture never took this branch.
        _warn_about_unrecorded_passes(manifest)
        return [(root, UNKNOWN_PASS)]
    if missing:
        print(f"[report] no ann-bench tree at {', '.join(missing)}; those "
              f"recall measurements are absent from this report. The "
              f"{len(trees)} tree(s) that are present are still read, because "
              f"one pass's results being gone is not a reason to lose "
              f"another's. Look under {root} on the machine that measured "
              f"them.")
    _warn_about_unrecorded_passes(manifest)
    return trees


def _warn_about_unrecorded_passes(manifest: Dict[str, Any]) -> None:
    """Name, on stdout, any pass whose results tree could not be narrowed to.

    An archived manifest written before `config["passes"]` existed records one
    pass in its config block and both of them in `phases[]`. The fingerprint
    of the pass it does not name was never written down, and nothing here can
    recompute it -- that needs `ann_fingerprint`, which lives in the
    orchestrator. Guessing by listing `annb/<pass>/*` would be worse than
    saying nothing: the tree is shared across runs, so any directory there
    might belong to another configuration entirely, which is the 16 GB curve
    under a 64 GB manifest all over again.

    **This says less than §2 Validity does, and deliberately.** It runs before
    a single record has been read, so the only thing it can know is that the
    tree could not be narrowed to. Whether the pass's recall ends up in the
    report is a different question with a different answer: both whole-tree
    fallbacks in `_narrow_to_this_run` hand the loader the un-narrowed root,
    and
    `_pass_from_path` then recovers the real pass from the directory layout,
    so a pass named here can still be published in §4. An earlier version of
    this docstring claimed the two notices were one statement made twice "so
    the two cannot disagree" -- they are two statements about two different
    moments, and conflating them is exactly how §2 came to assert that recall
    was absent from a table that was printing it. §2's notice is the
    intersection; see `summarize`.

    Called from every branch that fails to narrow, including the one that
    finds no fingerprint at all: that branch reads the whole shared tree,
    which is the cross-configuration risk worth hearing about, and it used to
    return in silence.

    stdout is not bundled by `./run-benchmark.sh export` and is not what
    anybody reads six months later, which is why §2 exists as well as this.
    """
    orphans = passes_without_a_tree(manifest)
    if not orphans:
        return
    one = len(orphans) == 1
    # Hoisted, not inlined. Written inline this read
    # `f"{"'s" if one else "'"}"` -- a replacement field reusing the enclosing
    # f-string's own quote, which is PEP 701 and therefore Python 3.12+. On
    # 3.11 it is a SyntaxError at module parse time, unconditional and
    # independent of this warning ever firing, and `report/` is parsed by
    # whatever `python3` the bench image has: both `postgres:17-bookworm` and
    # `debian:bookworm-slim` are Debian 12, which ships 3.11. The failure mode
    # is the one CLAUDE.md already records for this package -- a report that
    # dies at the end of a twenty-hour run -- plus a documented-prerequisites
    # host where `pytest` cannot even collect, because `tests/test_config.py`
    # imports `report.generate` dozens of times. The whole repository was
    # 3.11-compatible grammar until this line.
    possessive = "'s" if one else "'"
    print(f"[report] WARNING: this run also executed "
          f"{', '.join(orphans)}, and the manifest records no ann "
          f"fingerprint for {'it' if one else 'them'}, so this report cannot "
          f"narrow to {'that pass' if one else 'those passes'}"
          f"{possessive} results tree. "
          f"Whatever recall it does read is attributed by the directory it "
          f"came from, and §2 Validity names any pass whose recall is "
          f"genuinely missing. The ops measurements are unaffected either "
          f"way. A manifest predating per-pass recording can be completed "
          f"without re-measuring anything: `run --resume --run-id "
          f"{manifest.get('run_id', '<run>')}` records every pass's "
          f"fingerprint and skips every finished unit.")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    manifest = load_manifest(run_dir)
    # Engine colours and labels come from the run that produced the numbers, so
    # an engine added after this code was written still names and draws itself.
    charts.adopt_presentation(manifest)
    render_adopt_presentation(manifest)
    run_id = manifest.get("run_id", os.path.basename(run_dir))

    annb_dir = args.annb_results or os.path.join(os.path.dirname(run_dir), "annb")
    annb_trees = _narrow_to_this_run(annb_dir, manifest)
    datasets_dir = args.datasets_dir or os.path.join(VB_ROOT, "datasets")
    if not os.path.isdir(datasets_dir):
        datasets_dir = "/datasets"
    out_dir = args.output_dir or os.path.join(run_dir, "report")
    chart_dir = os.path.join(out_dir, "charts")

    print(f"[report] run       : {run_id}")
    print(f"[report] ann-bench : "
          f"{', '.join(path for path, _ in annb_trees)}")
    print(f"[report] datasets  : {datasets_dir}")
    print(f"[report] output    : {out_dir}")

    # Loaded before anything is created or deleted under out_dir. The chart
    # cleanup below used to run first, so a regeneration that then refused --
    # or simply found no records -- left the previous report as a page of
    # broken images with index.html still pointing at the files it had just
    # removed. A report that cannot be rebuilt should leave the one that
    # already works alone.
    records: List[Dict[str, Any]] = []
    try:
        if args.from_records:
            # A run directory is routinely copied off the machine that produced
            # it, and the ann-benchmarks HDF5 tree is not copied with it. The
            # merged records file is self-contained, so a report can be rebuilt
            # from an archived run without re-running anything.
            with open(args.from_records) as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            print(f"[report] records from {args.from_records}: {len(records)}")
            # The same guard load_ops_records applies, because this path does
            # not go through it. records.jsonl is written by this generator and
            # read back by the web UI, so a mixed set admitted here does not
            # just draw one bad chart -- it is persisted below as the file
            # webui/records.py prefers over the raw ops records, and every
            # later reader inherits it.
            # `declared` as well, so this path gets the whole guard and not
            # half of it. The manifest that will be printed above these numbers
            # is run_dir's, whatever file the records came out of -- so it is
            # run_dir's declared version they have to agree with.
            versions.refuse_mixed_measurement_versions(
                records, source=args.from_records,
                declared=versions.declared_ops_version(run_dir))
        else:
            records += loaders.load_ops_records(run_dir)
            print(f"[report] ops records: {len(records)}")

            # Once per tree this run wrote. The pass name goes in as the
            # loader's default label: inside a narrowed tree
            # `_pass_from_path` has nothing left to read, so without it every
            # result file that carries no `resource_pass` attribute would
            # arrive labelled "unknown" and two passes would merge into one
            # nameless column.
            annb: List[Dict[str, Any]] = []
            for tree, pass_name in annb_trees:
                found = loaders.load_annb_results(tree, datasets_dir, run_id,
                                                  pass_name)
                if len(annb_trees) > 1:
                    print(f"[report] ann-benchmarks records from "
                          f"{pass_name}: {len(found)}")
                annb += found
            records += annb
            print(f"[report] ann-benchmarks records: {len(annb)}")
    except versions.MixedMeasurementVersions as exc:
        # Reported the way this module reports every other refusal to produce a
        # report -- a message and a non-zero exit, not a traceback. The message
        # is multi-line and names what to do; a stack trace above it would bury
        # that.
        print(f"[report] {exc}", file=sys.stderr)
        return 2

    # The shared ann results tree is not a record of any one run, so narrow
    # what came out of it to the engines and corpora the manifest says this
    # run measured. See `report/scope.py` for the whole argument, including
    # why the manifest is the only source and why the ops records are never
    # narrowed. Before `--datasets`, so that flag's refusal message is about
    # what this run measured rather than about what the machine has ever
    # measured, and before `records.jsonl` is written, because that file is
    # what the web UI prefers over the raw ops records and a mixed set
    # persisted there is inherited by every later reader.
    measured = scope.scope_of(manifest)
    records, out_of_scope = scope.narrow_to_measured_scope(records, manifest)
    # Then per file: a pair the phase log names is not enough, because the
    # attempt it names may not be the one that wrote the file -- see
    # `report/provenance.py` for `full-20260825`, whose M=16 recall table
    # stood under a manifest naming M=6 attempts.
    records, unprovenanced, ann_provenance = \
        _provenance.split_by_provenance(records, manifest)
    if unprovenanced:
        print(f"[report] excluded "
              f"{sum(r['records'] for r in unprovenanced)} recall record(s) "
              f"from {sum(r['files'] for r in unprovenanced)} result file(s) "
              f"that no ann attempt of this run wrote or reused; §2 Validity "
              f"names them.")
    if ann_provenance["counts"][_provenance.UNVERIFIED]:
        print(f"[report] {ann_provenance['counts'][_provenance.UNVERIFIED]} "
              f"recall record(s) come from ann attempts that recorded no file "
              f"list; they are labelled unverified.")
    # The PAIRS first, because that is the unit the narrowing works in. The
    # two axis lists are printed under it as the sets the pairs are drawn
    # from and as what each unrecorded-axis warning is about -- printing only
    # the axes described a product this no longer applies, and a reader
    # comparing stdout against §2 would have found two different claims.
    if measured.pairs:
        print(f"[report] this run measured {len(measured.pairs)} "
              f"engine/corpus combination(s): "
              + "; ".join(f"{engine} on {dataset}"
                          for engine, dataset in sorted(measured.pairs)))
    if measured.unpaired:
        print(f"[report] this run configured but recorded no unit for: "
              f"{', '.join(sorted(measured.unpaired))}; "
              f"those are read on whichever of its corpora the tree holds")
    for axis, named in (("engines", sorted(measured.engines)),
                        ("datasets", sorted(measured.datasets))):
        if named:
            print(f"[report] this run's {axis}: {', '.join(named)}")
            continue
        # The legacy class, and the same bargain `_warn_about_unrecorded_passes`
        # strikes: stdout, not a Validity notice. An archived manifest can
        # record neither the engines it configured nor the corpora it asked
        # for, and a scope derived from it would be empty -- which, applied,
        # deletes the entire recall half of the report. Reading too much is
        # the defect; reading nothing is worse, so that axis is left alone and
        # §2's staleness notice stays the guard, exactly as it was before this
        # change.
        print(f"[report] WARNING: this manifest records no {axis} for the "
              f"run, so this report cannot narrow the shared ann results "
              f"tree by {axis[:-1]}. Whatever that tree holds for another "
              f"run at this resource pass is read as though this run had "
              f"measured it, and §2 Validity flags any of it that predates "
              f"this run. A run directory written by a current "
              f"./run-benchmark.sh run records both.")
    if out_of_scope:
        print(f"[report] excluded {scope.excluded_record_count(out_of_scope)} "
              f"recall record(s) from "
              f"{scope.excluded_file_count(out_of_scope)} result file(s) that "
              f"another run wrote into the shared tree; §2 Validity names "
              f"them.")

    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        present = {r.get("dataset") for r in records if r.get("dataset")}
        unknown = wanted - present
        if unknown:
            print(f"[report] --datasets names nothing measured here: "
                  f"{sorted(unknown)}. Present: {sorted(present)}",
                  file=sys.stderr)
            return 2
        before = len(records)
        records = [r for r in records
                   if not r.get("dataset") or r["dataset"] in wanted]
        print(f"[report] narrowed to {sorted(wanted)}: "
              f"{len(records)} of {before} records")

    if not records:
        print("[report] no records found — nothing to report", file=sys.stderr)
        # The one state in which the exclusion census has nowhere else to go.
        # No records means no report, so §2 is never rendered and the only
        # account of what was dropped is this. It is reachable: an ann-only
        # profile whose own phase wrote nothing leaves a run directory with no
        # ops records and a shared tree full of other runs' results, and
        # before the narrowing that published seven engines under a one-engine
        # manifest. Exiting 1 is right -- there is nothing of THIS run to
        # report -- but exiting 1 without saying why reads as a broken tree.
        if out_of_scope:
            print(f"[report] every recall result under the shared ann tree "
                  f"belonged to another run "
                  f"({scope.excluded_record_count(out_of_scope)} record(s) "
                  f"across "
                  f"{len(out_of_scope)} engine/corpus combination(s)), and "
                  f"this run produced none of its own. Report the run that "
                  f"measured them, or re-run this one's ann phase.",
                  file=sys.stderr)
        return 1

    # Only now, with records in hand and every refusal behind us, is the output
    # directory touched. Every chart is redrawn from the records on each run, so
    # anything already here is from a previous one. Left in place they
    # accumulate: narrowing a report to one corpus with --datasets leaves the
    # other corpus's charts sitting beside the new ones, named after a dataset
    # the report no longer covers. Nothing references them, which is exactly
    # what makes them dangerous -- a stale file that nothing points at is one
    # somebody opens by hand and believes.
    os.makedirs(chart_dir, exist_ok=True)
    removed = 0
    for name in os.listdir(chart_dir):
        if name.endswith((".png", ".svg")):
            try:
                os.remove(os.path.join(chart_dir, name))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[report] cleared {removed} chart(s) from the previous report")

    memory = loaders.load_memory_series(run_dir)

    # Fill in peak memory from the timeseries where the harness could not read
    # a kernel high-water mark itself.
    peaks = {name: loaders.peak_from_series(rows) for name, rows in memory.items()}
    attach_peak_rss(records, peaks)

    # Persisted AFTER the enrichment above, not before: writing it earlier left
    # records.jsonl missing the peak-RSS values the rendered report displayed,
    # so the raw data and the report disagreed.
    merged_path = os.path.join(out_dir, "records.jsonl")
    with open(merged_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    print(f"[report] merged records -> {merged_path}")

    summary = summarize(records, manifest, memory)
    # The perf profiles `ops.profile_window_clients` asked for, keyed like the
    # memory series plus the client count, for the limiter table's links.
    # From the run directory, which `summarize` is not given.
    from report.limiters import profile_index
    summary["profiles"] = profile_index(run_dir, out_dir)
    # Set here rather than built inside `summarize`, because `summarize`'s
    # inputs cannot produce it: by the time it is called the excluded records
    # are gone, and a census of them is exactly the thing the remaining
    # records cannot describe. Passing it through as an argument would make
    # `summarize` a courier for a value it neither derives nor uses. §2 reads
    # it with `.get`, like every other optional notice, so a summary built by
    # hand -- several tests build one -- simply has nothing to say.
    summary["out_of_scope_ann"] = out_of_scope
    # Same reasoning: the census is of records `summarize` never sees.
    summary["unprovenanced_ann"] = unprovenanced
    summary["ann_provenance"] = ann_provenance
    chart_paths: Dict[str, Dict[str, str]] = {}

    for dataset in summary["datasets"]:
        # Keyed by (engine, pass), not by engine. A Pareto frontier drawn
        # through two resource passes is a frontier of nothing: it takes the
        # faster of two machines at every recall and draws a curve no
        # configuration ever produced. `charts.pareto` styles off the engine
        # half and names the pass in the legend only when more than one is
        # present, so a single-pass run draws and labels exactly as before.
        by_engine: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            if r.get("phase") == "recall_qps" and r.get("dataset") == dataset:
                by_engine[(r["engine"], r.get("resource_pass"))].append(r)
        k = next((r.get("k") for r in records
                  if r.get("dataset") == dataset and r.get("k")), 10)

        safe = dataset.replace(".", "_")

        # A profile chooses which workloads to run. Drawing an empty axis for
        # one that was never enabled has confused every reader of these
        # reports so far -- an absent chart reads as a broken report, when it
        # only means `workloads:` did not include it. Charts whose phase
        # produced no records are skipped entirely, and the section says so.
        phases_present = {r.get("phase") for r in records
                          if r.get("dataset") == dataset}
        needs = {
            "latency": "concurrency",
            "concurrency": "concurrency",
            "filtered": "filtered",
            "churn": "churn",
            "churnimpact": "churn",
            # PR 2 §2.3.4 calls the calibration chart optional. This is what
            # makes it so, and through the existing mechanism rather than a
            # second one: `ef_mode: fixed` runs no sweep and writes no
            # `ef_calibration` records, so the chart is skipped.
            #
            # Two caveats on the shared skip line below, both of which are
            # about this entry and not about the mechanism. It says "not in
            # this profile's workloads", which is wrong for this one: the
            # cause is `ops.ef_mode`, not `ops.workloads`. And it is §6 that
            # explains the absence to a reader, not the appendix -- there IS
            # no appendix on a fixed-ef run, which `render_markdown` enforces
            # and `test_there_is_no_appendix_without_a_chart` pins. Fixing the
            # console wording means giving this entry its own skip branch; it
            # is a log line, the report itself is unambiguous, and that branch
            # belongs with whatever next needs a second reason for a skip.
            "calibration": "ef_calibration",
        }

        # The pass comparison exists to put normalized next to tuned. With one
        # pass it draws pale bars against nothing, and its per-workload panels
        # are blank for whatever the profile did not run -- four panels, two of
        # them empty, on a chart that could not say anything either way.
        passes_present = {r.get("resource_pass") for r in records
                          if r.get("dataset") == dataset and r.get("resource_pass")}

        # One filtered chart per column, or -- when there are no filtered
        # records at all -- the single `filtered` entry this always had, so
        # that `needs` below still matches it and still prints the shared
        # "not in this profile's workloads" skip line. Without the fallback a
        # profile that runs no filtered workload would generate no chart and
        # say nothing about why, which is the one case the log line exists
        # for.
        filtered_charts = filtered_chart_entries(records, dataset)

        for name, fn, payload in (
            ("pareto", charts.pareto, (dict(by_engine), dataset, k)),
            # 0.95, not 0.85. With the main chart on a logit axis the whole
            # range is already legible, so the companion's job is the region
            # people actually deploy at -- and at 0.85 every engine's curve was
            # still mostly below the floor anyone would ship.
            ("paretozoom", lambda *a, **kw: charts.pareto(*a, recall_floor=0.95, **kw),
             (dict(by_engine), dataset, k)),
            *qps_at_recall_entries(summary, dataset),
            ("latency", charts.latency_percentiles, (records, dataset)),
            ("build", charts.build_cost, (records, dataset)),
            ("storage", charts.storage_breakdown, (records, dataset)),
            ("concurrency", charts.concurrency, (records, dataset)),
            # One per filter column present. PR 3's acceptance asks for "a
            # column selector (or two charts)"; two charts is the option that
            # survives being rendered to Markdown, which this report also is.
            # Derived from the records rather than from FILTER_COLUMNS: an
            # archived run carries no `filter_column` and must keep producing
            # the one chart it always produced, and a profile narrowed to
            # `filter_columns: [tag]` must not emit an empty `ctag` panel.
            *filtered_charts,
            ("churn", charts.churn, (records, dataset)),
            ("churnimpact", charts.churn_impact, (records, dataset)),
            ("passcompare", charts.pass_comparison, (records, dataset)),
            # The whole record list, not the calibration records alone: the
            # chart rings the chosen point on each curve, and which point that
            # is lives on the downstream records (see
            # `loaders.operating_points`). Handed only the sweep's own records
            # it would draw twelve points per engine and mark none of them.
            ("calibration", charts.calibration_curves, (records, dataset)),
        ):
            if name == "passcompare" and len(passes_present) < 2:
                print(f"[report] skipping passcompare: only the "
                      f"{', '.join(sorted(passes_present)) or 'one'} pass ran "
                      f"(needs normalized and tuned)")
                continue
            # Split on "-" so `filtered-tag` and `filtered-ctag` find the
            # `filtered` entry. No other chart name contains a hyphen, and a
            # future one that does would want the same prefix behaviour.
            required = needs.get(name) or needs.get(name.split("-")[0])
            if required and required not in phases_present:
                print(f"[report] skipping {name}: no {required} records "
                      f"({_why_absent(records, dataset, name.split('-')[0])})")
                continue
            stem = f"{name}-{safe}"
            try:
                paths = fn(*payload, chart_dir, stem)
            except Exception as exc:  # noqa: BLE001 - one bad chart must not kill the report
                print(f"[report] chart {stem} failed: {exc}")
                paths = None
            if paths:
                chart_paths[stem] = paths
                print(f"[report] chart -> {os.path.basename(paths['svg'])}")

    if memory:
        # One chart per phase. The ann phase loads a corpus and sweeps a grid;
        # the ops phase loads it again and runs four workloads. Different
        # durations, different questions, and twelve series on one axis was
        # crowded before any of them held two measurements.
        for stem, phase in (("memory-timeline", None),
                            ("memory-timeline-ann", "ann")):
            paths = charts.memory_timeline(memory, chart_dir, stem, phase=phase)
            if paths:
                chart_paths[stem] = paths

    md_path = os.path.join(out_dir, "report.md")
    html_path = os.path.join(out_dir, "report.html")

    markdown = render_markdown(manifest, summary, records, chart_paths, args.title)
    with open(md_path, "w") as fh:
        fh.write(markdown)
    print(f"[report] markdown -> {md_path}")

    html = render_html(manifest, summary, records, chart_paths, args.title)
    with open(html_path, "w") as fh:
        fh.write(html)
    print(f"[report] html -> {html_path}")

    if summary["plan_failures"]:
        print(f"\n[report] WARNING: {len(summary['plan_failures'])} measurement(s) ran "
              f"WITHOUT the vector index in the query plan. Those are full scans, "
              f"not ANN search — see the 'Validity' section of the report.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
