"""Markdown for `report/provenance.py`: §1's per-engine budget table, §2's
provenance notices and §4's label line.

Its own module so `render.py` -- which another change is editing in the floor
tables at the same time -- gains three call sites and nothing else. The
helpers it shares with `render.py` (`_md_table`, `_label`, `_fmt_bytes`) are
imported at call time, because `render.py` imports this module and a
module-level import back would be a cycle.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

from report import provenance

#: §1's table header. A constant so the test reads the same cells this writes.
BUDGET_HEADER = ["Engine", "Pass", "Main cache", "Graph cache",
                 "Build memory", "Build threads", "shm", "cpuset", "-march"]
BUDGET_HEADING = "### What each engine was given"
UNPROVENANCED_HEADING = ("### Recall results no ann attempt of this run "
                         "accounts for")
UNPROVENANCED_HEADER = ["Engine", "Dataset", "Pass", "M", "Result files",
                        "Why it is not this run's measurement"]
REUSED_HEADING = "### Recall results reused from an earlier attempt"
REUSED_HEADER = ["Engine", "Dataset", "Pass", "Result files",
                 "Oldest written"]
UNVERIFIED_HEADING = "### Recall provenance that cannot be verified"

_REASON = {
    provenance.NO_ANN_PHASE:
        "no ann phase of this run ran this engine on this corpus at this "
        "pass",
    provenance.NOT_LISTED:
        "an ann phase ran here, but neither wrote nor reused these files",
}


def _r():
    from report import render
    return render


def _cell_bytes(value: Any) -> str:
    return _r()._fmt_bytes(value) if value else "not set"


def _main_cache(kind_value) -> str:
    kind, value = kind_value
    if kind == "maxmemory":
        return f"{_r()._fmt_bytes(value)} (maxmemory)"
    if kind == "buffer":
        return _r()._fmt_bytes(value)
    # Resolved and never applied -- MongoDB's WiredTiger cache in
    # `full-20260825`, which ran at its own default (57-64 GB observed) while
    # every other engine's budget was set. Printing the resolved share here
    # would be the claim that post made.
    return "engine default (not set)"


def engine_budget_section(manifest: Dict[str, Any]) -> str:
    """§1's per-engine table, or the statement that this manifest cannot
    give one.

    The single-valued rows above it are one engine's split, and on
    `full-20260825` they were a different engine's from one snapshot to the
    next. For a manifest without `engines_resolved` this says so instead of
    leaving those rows to be read as everyone's budget.
    """
    r = _r()
    rows = provenance.engine_budget_rows(manifest)
    head = f"\n{BUDGET_HEADING}\n\n"
    if not rows:
        return (head +
                "_This manifest predates per-engine resolution records "
                "(`engines_resolved`), so it cannot show what each engine "
                "was given. The buffer, graph-cache and build figures above "
                "are ONE engine's split -- whichever was configured last -- "
                "and resolution differs by engine: pgvector's buffer absorbs "
                "the graph cache's share, an in-memory engine gets "
                "`maxmemory` instead, and an engine whose cache flag is not "
                "set runs at its own default. Do not read them as every "
                "engine's budget; the server logs saved beside the "
                "measurements are the only record of the rest._\n")
    recorded = {row["engine"] for row in rows}
    missing = sorted(set((manifest or {}).get("engines") or {}) - recorded)
    table = r._md_table(BUDGET_HEADER, [[
        r._label(row["engine"]), row["resource_pass"],
        _main_cache(row["main_cache"]),
        _cell_bytes(row["graph_cache"]), _cell_bytes(row["build_memory"]),
        str(row["build_threads"] if row["build_threads"] is not None
            else "?"),
        str(row["shm_size"] or "?"), str(row["cpuset"] or "?"),
        str(row["march"] or "?"),
    ] for row in rows])
    note = ("Read from `engines_resolved` in the manifest: the resolution "
            "each engine's containers were started with, per pass. A figure "
            "reads `not set` when it was resolved but no server flag carries "
            "it, so the engine ran at its own default.\n")
    if missing:
        note += (f"\n_No resolution is recorded for "
                 f"{', '.join(r._label(e) for e in missing)}: measured "
                 f"before this record existed, and not re-stamped since._\n")
    return head + note + table


def _date(epoch: Any) -> str:
    try:
        return _dt.datetime.fromtimestamp(
            float(epoch), _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def validity_notices(manifest: Dict[str, Any], summary: Dict[str, Any]) -> str:
    r = _r()
    parts: List[str] = []
    missing = summary.get("unprovenanced_ann") or []
    if missing:
        files = sum(row.get("files") or 0 for row in missing)
        parts.append(
            f"\n{UNPROVENANCED_HEADING}\n\n"
            f"Every recall number in §4 has to have been written -- or "
            f"knowingly reused -- by an ann attempt this run recorded for the "
            f"same engine, corpus and pass. **{files} result file(s) below "
            f"were not**, and are excluded from every chart and table in this "
            f"report and from `records.jsonl`. They are in the shared tree, "
            f"so they are some run's measurement; nothing here says which, "
            f"under what budget or at which grid.\n\n"
            f"_The case this exists for: `full-20260825` listed MySQL-family "
            f"ann attempts that wrote M=6 files, while its recall table was "
            f"built from M=16 files written by earlier attempts the manifest "
            f"no longer named._\n")
        parts.append(r._md_table(UNPROVENANCED_HEADER, [[
            r._label(row["engine"]), row["dataset"], row["resource_pass"],
            ", ".join(str(m) for m in row.get("m") or []) or "?",
            str(row.get("files") or row.get("records") or 0),
            _REASON.get(row.get("reason"), "not accounted for"),
        ] for row in missing]))
    info = summary.get("ann_provenance") or {}
    reused = info.get("reused") or []
    if reused:
        parts.append(
            f"\n{REUSED_HEADING}\n\n"
            f"ann-benchmarks skips a configuration whose result file exists. "
            f"The attempts below found these files and measured nothing for "
            f"them, so their §4 points are **reused**: a measurement of the "
            f"same configuration, made earlier, under the budget recorded "
            f"then.\n")
        parts.append(r._md_table(REUSED_HEADER, [[
            r._label(row["engine"]), row["dataset"], row["resource_pass"],
            str(row.get("files") or 0), _date(row.get("oldest")),
        ] for row in reused]))
    unverified = info.get("unverified") or []
    if unverified:
        count = sum(row.get("records") or 0 for row in unverified)
        parts.append(
            f"\n{UNVERIFIED_HEADING}\n\n"
            f"The ann attempts this manifest records for "
            f"{len(unverified)} engine/corpus/pass combination(s) carry no "
            f"list of the result files they wrote or reused -- the manifest "
            f"predates that record -- so this report cannot check that the "
            f"{count} recall point(s) it read for them were produced by those "
            f"attempts. They are shown, and labelled **unverified** in §4: a "
            f"file another attempt or another run left in the shared tree is "
            f"indistinguishable from this run's own.\n")
    if any(row.get("limit_source") == "run-level"
           for row in summary.get("memory_pressure") or []):
        parts.append(
            "\n_The memory limit the pressure figures below are measured "
            "against is the run-level value, not the engine's own: this "
            "manifest records no per-engine resolution for those series._\n")
    return "".join(parts)


def section4_note(summary: Dict[str, Any]) -> str:
    info = summary.get("ann_provenance") or {}
    counts = info.get("counts") or {}
    total = sum(counts.values())
    lines: List[str] = []
    if counts.get(provenance.UNVERIFIED):
        lines.append(
            f"**Unverified:** {counts[provenance.UNVERIFIED]} of {total} "
            f"recall point(s) below come from ann attempts that recorded no "
            f"file list, so nothing ties them to this run's measurement -- "
            f"see §2.")
    if counts.get(provenance.REUSED):
        lines.append(
            f"**Reused:** {counts[provenance.REUSED]} of {total} recall "
            f"point(s) below were served from result files an earlier "
            f"attempt wrote -- see §2.")
    excluded = sum(row.get("records") or 0
                   for row in summary.get("unprovenanced_ann") or [])
    if excluded:
        lines.append(
            f"{excluded} recall record(s) no ann attempt of this run "
            f"accounts for are excluded -- see §2.")
    return "".join(f"\n_{line}_\n" for line in lines)
