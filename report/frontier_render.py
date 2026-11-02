"""§4's frontier disclosures: what was swept, what it cost, and which build.

Split from `report/render.py` (already the largest file in the package) and
imported by it lazily, inside `_headline_tables`, because this module uses
render's cell helpers at import time and render is the module `generate`
imports first. `report/frontier.py` computes every figure printed here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from report import frontier as _frontier
from report.render import (_fmt, _fmt_bytes, _label, _md_table, _recall_span,
                           _with_build)

#: Beside a headline figure taken from a build other than the one the ops
#: pass measured. Not `†` or `‡`: those already mean "measured above the
#: floor" and "never reached the target" in §4 and §6, and one glyph for one
#: meaning is the rule `render.FLOOR_MARK` records. The chart imports this
#: rather than spelling it again.
OPS_MISMATCH_MARK = "◇"

#: The heading of §4's secondary table. A constant because the tests split
#: the section on it: the headline and the ops-build view must never be read
#: as one table.
OPS_TABLE_HEADING = "**At the build the ops pass measured**"

FLOORS = (90, 95, 99)


def is_curve(entry: Dict[str, Any]) -> bool:
    """One build swept: the line is a curve at that build, not a frontier.

    `sweep_builds` is absent on an entry built by hand or by an older
    summarize; such an entry makes no claim, so it is not relabelled.
    """
    return entry.get("sweep_builds") == 1


def line_kind(entry: Dict[str, Any]) -> str:
    """The words for one engine's recall-throughput line.

    The reviewer's point, applied everywhere the report used to say
    "frontier": varying ef_search at one build traces that build's curve.
    Calling it a Pareto frontier claims the build space was searched.
    """
    if is_curve(entry):
        label = ((entry.get("swept_space") or {}).get("build_label")
                 or entry.get("reference_build_label") or "one build")
        return f"recall–throughput curve at `{label}`"
    builds = entry.get("sweep_builds")
    return (f"Pareto frontier over {builds} builds" if builds
            else "Pareto frontier")


def section_intro(summary: Dict[str, Any]) -> str:
    """§4's opening sentence, worded by what the run actually swept."""
    entries = [entry for by_pass in (summary.get("per_dataset") or {}).values()
               for per_engine in by_pass.values()
               for entry in per_engine.values()]
    curves = [e for e in entries if is_curve(e)]
    lead = ("The headline comparison: the throughput each engine reaches at "
            "a recall, taken from its best configuration across every build "
            "this run swept (M, ef_construction, quantization, storage "
            "engine and ef_search), with the build that produced each figure "
            "named beside it. Identically named parameters are not identical "
            "work across engines, so engines are compared at a recall, not "
            "at a setting. The faint scatter behind each line is every "
            "measured point. The recall–throughput charts draw one line per "
            "engine and do not name builds on it; which build supplies each "
            "stretch of a line is the per-build recall range in the "
            "\"Build cost on the frontier\" table below.")
    if entries and len(curves) == len(entries):
        return (lead + " **In this run only ef_search was varied for every "
                "engine**, so each line is a recall–throughput curve at one "
                "build, not a Pareto frontier: the build space was not "
                "explored, and the swept-space table below says which build "
                "each engine was measured at.\n")
    if curves:
        return (lead + " Where an engine was measured at a single build the "
                "line is a recall–throughput curve at that build, not a "
                "frontier, and the swept-space table below says so.\n")
    return lead + "\n"


def headline_mark(entry: Dict[str, Any], floor: int) -> str:
    """The mismatch marker for one headline cell, or nothing."""
    if entry.get(f"qps_at_recall_{floor}_is_ops_build") is False:
        return f" {OPS_MISMATCH_MARK}"
    return ""


def _unanswerable_reason(entry: Dict[str, Any]) -> str:
    """Why "is this the ops build?" has no answer for one engine."""
    build = entry.get("reference_build")
    if not build:
        return "the ops build is not recorded"
    spelled = entry.get("reference_build_label") or "not recorded"
    if any(value is None for value in build.values()):
        return (f"the ops build (`{spelled}`) left a setting to the server "
                f"default, which no record names")
    return f"the ops build (`{spelled}`) matches none of the swept builds"


def mismatch_notes(per_engine: Dict[str, Any]) -> str:
    """Name the ops build wherever a headline figure is from another one.

    The concurrency, churn and filtered sections measure the ops build, so a
    reader who takes the headline QPS and the concurrency curve as one index
    is comparing two. Until the ops pass is pointed at the frontier build,
    the page has to say which figures that is -- and has to say when it
    cannot tell, rather than mark a figure "different" from a build nobody
    can compare it with.
    """
    lines = []
    unknown = []
    for engine, entry in sorted(per_engine.items()):
        flags = [entry.get(f"qps_at_recall_{f}_is_ops_build") for f in FLOORS
                 if entry.get(f"qps_at_recall_{f}") is not None]
        off = [f for f in FLOORS
               if entry.get(f"qps_at_recall_{f}_is_ops_build") is False]
        if off:
            floors = ", ".join(f"0.{f}" for f in off)
            lines.append(
                f"- **{_label(engine)}** at {floors}: the figure is from "
                f"another build than the ops pass measured "
                f"(`{entry.get('reference_build_label') or 'not recorded'}`); "
                f"its concurrency, filtered and churn figures describe that "
                f"ops build, not this one.\n")
        elif flags and all(flag is None for flag in flags) and (
                "reference_build" in entry or "swept_space" in entry):
            unknown.append(f"- **{_label(engine)}**: "
                           f"{_unanswerable_reason(entry)}.\n")
    out = ""
    if lines:
        out += (f"\n_A **{OPS_MISMATCH_MARK}** marks a headline figure taken "
                f"from a different build than the one the ops pass "
                f"measured:_\n\n" + "".join(lines))
    if unknown:
        out += ("\n_Whether the headline build is the one §5 onward measured "
                "cannot be said for these engines, so their figures carry no "
                "mark either way; the build under each figure is still the "
                "one that produced it:_\n\n" + "".join(unknown))
    return out


def swept_space_table(per_engine: Dict[str, Any]) -> str:
    """Per engine, which axes were swept and with which values."""
    rows = []
    for engine, entry in sorted(per_engine.items()):
        space = entry.get("swept_space")
        if not space:
            continue
        if space.get("single_build"):
            kind = (f"curve at `{space.get('build_label')}` -- build space "
                    f"not explored")
        else:
            kind = f"frontier over {space.get('builds')} builds"
        rows.append([_label(engine), _frontier.spell_swept(space), kind])
    if not rows:
        return ""
    return ("\n**Swept space**\n\n"
            "_What each engine's line above is drawn from. A frontier is "
            "only as wide as the space it searched: an engine measured at "
            "one build has a curve, and its figures say nothing about the "
            "builds that were not tried._\n"
            + _md_table(["Engine", "Swept", "Line"], rows))


def _time_cell(row: Dict[str, Any]) -> str:
    seconds = row.get("time_to_queryable_s")
    if seconds is None:
        return "not measured"
    source = {"ann": "ann fit", "ops": "ops"}.get(row.get("time_source"), "")
    text = _fmt(float(seconds), 0, " s")
    return f"{text} ({source})" if source else text


def _size_cell(row: Dict[str, Any]) -> str:
    size = row.get("index_bytes")
    if size is None:
        return "not measured"
    text = _fmt_bytes(size)
    if row.get("index_resident"):
        text += " resident"
    if row.get("index_bytes_source") == "ops":
        text += " (ops)"
    if row.get("same_size_as_another"):
        text += " ="
    return text


def build_cost_table(per_engine: Dict[str, Any]) -> str:
    """What each build on the frontier cost to build."""
    rows = []
    same_size = False
    for engine, entry in sorted(per_engine.items()):
        for row in entry.get("frontier_builds") or []:
            ops = {True: "yes", False: "no", None: "cannot be said"}[
                row.get("is_ops_build")]
            label = row.get("build_label") or "—"
            if len(entry.get("storage_engines") or []) > 1 and row.get(
                    "storage_engine"):
                label += f" ({row['storage_engine']})"
            same_size = same_size or bool(row.get("same_size_as_another"))
            rows.append([
                _label(engine), f"`{label}`",
                _recall_span(row.get("recall_low"), row.get("recall_high")),
                str(row.get("frontier_points", 0)),
                _time_cell(row), _size_cell(row), ops])
    if not rows:
        return ""
    same = (" A trailing **=** marks a size another swept build of the same "
            "engine reports to the byte. Different graphs rarely do, so read "
            "it as one reading per run rather than a per-build measurement "
            "until the module that took it says otherwise."
            if same_size else "")
    return ("\n**Build cost on the frontier**\n\n"
            "_Every build that contributes a point to an engine's line. "
            "`Time to queryable index` is wall clock from an empty table to "
            "an index that answers queries, load included. **(ann fit)** is "
            "ann-benchmarks' `build_time` for that build's `fit()`; **(ops)** "
            "means the ann run recorded none and the figure is the ops "
            "pass's own build of the same index (load plus `CREATE INDEX` "
            "for a bulk build, the load alone where the graph is built on "
            "insert). The two are separate measurements of the same "
            "quantity and are labelled so. `Index size` is measured on the "
            "server (index files, `pg_relation_size`, mongot's index, or "
            "Valkey's resident growth), with **(ops)** meaning the same "
            "fallback; ann-benchmarks' own `index_size` is the benchmark "
            "client's memory and is never printed. `Ops build` says whether "
            "this is the index §5 onward measured." + same + "_\n"
            + _md_table(["Engine", "Build", "Frontier recall range",
                         "Frontier points", "Time to queryable index",
                         "Index size", "Ops build"], rows))


def _smallest_cell(entry: Dict[str, Any], floor: int) -> str:
    key = f"smallest_qps_at_recall_{floor}"
    value = _fmt(entry.get(key), 0)
    if value == "—":
        return value
    at = entry.get(f"{key}_recall")
    if at is not None:
        value += f" @{at:.3f}"
    size = entry.get(f"{key}_index_bytes")
    label = entry.get(f"{key}_build_label") or ""
    if size is not None:
        label += f", {_fmt_bytes(size)}"
    ranked_by = entry.get(f"{key}_ranked_by")
    if ranked_by == _frontier.TIED:
        value += " (tie)"
    elif ranked_by == _frontier.BY_SIZE_THEN_M:
        value += " (=size)"
    return _with_build(value, label)


def smallest_table(per_engine: Dict[str, Any]) -> str:
    """Beside "fastest build", the smallest build reaching each floor.

    The fastest build at a floor is usually the biggest graph swept. This
    table answers "what is the cheapest index that gets there" instead, so
    the headline cannot be read as an argument for the largest M. The caption
    names only the rules that actually decided a cell: it said "ranked by
    measured index size" over pgvector cells whose three builds all read the
    same byte count, where size had decided nothing.
    """
    rows, rules = [], set()
    for engine, entry in sorted(per_engine.items()):
        if not any(f"smallest_qps_at_recall_{f}" in entry for f in FLOORS):
            continue
        rows.append([_label(engine)]
                    + [_smallest_cell(entry, f) for f in FLOORS])
        rules |= {entry.get(f"smallest_qps_at_recall_{f}_ranked_by")
                  for f in FLOORS} - {None}
    if not rows:
        return ""
    notes = []
    if _frontier.ONLY in rules:
        notes.append("trivially where only one build reaches the floor")
    if _frontier.BY_SIZE in rules:
        notes.append("by measured index size where every build reaching the "
                     "floor has one and the smallest is unique")
    if _frontier.BY_SIZE_THEN_M in rules:
        notes.append("**(=size)** where the smallest size is shared by "
                     "several builds, so M and then efC decided among them")
    if rules & {_frontier.BY_M, _frontier.TIED}:
        notes.append("by lowest M and then lowest efC where a size is "
                     "missing")
    if _frontier.TIED in rules:
        notes.append("**(tie)** where no size was measured and several "
                     "builds share the lowest M and efC, so the smallest "
                     "cannot be told apart and the fastest of them is shown")
    return ("\n**Smallest build that reaches each floor**\n\n"
            f"_Chosen {'; '.join(notes)}. The QPS is that build's best at or "
            f"above the floor. Compare with the headline above, which is the "
            f"fastest build._\n"
            + _md_table(["Engine", "Smallest @ recall≥0.90",
                         "Smallest @ recall≥0.95", "Smallest @ recall≥0.99"],
                        rows))


def ops_build_table(per_engine: Dict[str, Any],
                    floor_mark: Optional[str] = None) -> str:
    """The same floors at the build the ops pass measured.

    Formerly the headline (PR #77). Kept because it is the one view in which
    §4 and §5 onward describe the same index; every cell names the build.
    """
    rows = []
    for engine, entry in sorted(per_engine.items()):
        lowest = entry.get("reference_min_recall")

        def cell(floor: int) -> str:
            key = f"reference_qps_at_recall_{floor}"
            value = _fmt(entry.get(key), 0)
            at = entry.get(f"{key}_recall")
            if value != "—" and at is not None:
                value += f" @{at:.3f}"
                if floor_mark and lowest is not None and lowest > floor / 100:
                    value += f" {floor_mark}"
            return _with_build(value, entry.get(f"{key}_build_label"))

        rows.append([_label(engine), cell(90), cell(95), cell(99),
                     _recall_span(entry.get("reference_min_recall"),
                                  entry.get("reference_max_recall")),
                     str(entry.get("reference_points", 0))])
    return (f"\n{OPS_TABLE_HEADING}\n\n"
            "_The same floors restricted to the engine's reference build -- "
            "the index the ops pass built, so these are the figures that "
            "describe the same index as the concurrency, filtered and churn "
            "sections. A dash means that build has no point at or above the "
            "floor, whatever another build reached._\n"
            + _md_table(["Engine", "Ops build QPS @ recall≥0.90",
                         "Ops build QPS @ recall≥0.95",
                         "Ops build QPS @ recall≥0.99",
                         "Ops build recall range", "Ops build points"], rows))
