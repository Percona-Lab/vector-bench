"""Stage 2 of the two-stage frontier: choose the builds worth a full-size run.

`./run-benchmark.sh select-builds --from results/<stage-1 run> --out <file>`
reads a `frontier-explore` run (90,000 rows, every build knob), keeps the
builds that make each engine's frontier at or near the recall floors, and
writes an overlay `run --profile tuned-complete --overlay <file>` measures at
990,000 rows. See `config/profiles/frontier-explore.yml` for why there are two
stages and `orchestrator/profile_overlay.py` for how the overlay is applied.

THE RULE, per (engine, tuned pass), over every stage-1 point that used the
vector index:

1. The frontier is the upper-left envelope of (recall, QPS) over ALL of the
   engine's builds and query efforts -- the same envelope
   `report/loaders.py:pareto_frontier` draws, which
   `tests/test_two_stage_m_sweep.py` checks this one against point for point.
2. At each floor f in 0.90 / 0.95 / 0.99, the WINNER is the build owning the
   frontier point with the smallest recall >= f: the fastest configuration
   that reaches the floor. It is kept.
3. NEAR is defined in recall, as |recall - f| <= 0.015: every build owning a
   frontier point that close to a floor is kept as well. 0.015 because recall
   at fixed M moved by at most 0.014 between 90,000 and 990,000 rows (M=16,
   the m-sweep of 2026-08-28 against full-20260825) -- a build whose frontier
   point sits that close to a floor at 90k can be on the other side of it at
   990k, and pruning it would let stage 2 miss the build that wins there.
4. A floor no point reaches keeps the build with the highest recall, and the
   floor is recorded as unreached: that build is the engine's closest
   approach, and the stage-2 report has to be able to say so.
5. NEIGHBOURS: for every build kept by 2-4, the next explored M below and
   above it at the same ef_construction / quantization are kept too, so stage
   2 can confirm the choice transfers -- if the winner moves one step at 990k,
   the step it moved to was measured. A kept M at the edge of the explored
   grid (M=4 or M=48) has no neighbour on that side; that is recorded as an
   edge note, because the true optimum may lie outside the grid.
6. The REFERENCE build of the target profile (what its ops pass builds;
   `reference_build.resolve_reference_build`) is always kept. `run` refuses a
   grid that does not contain it, and the headline figures are taken from it.
7. NO CLOSURE. The overlay names the kept builds exactly
   (`resources.extras.ann_builds_by_engine`, applied by
   `ann_knobs.restrict_to_builds`), beside the per-axis lists that make the
   values legal for the render. Lists alone are multiplied out by upstream,
   and on a synthetic stage 1 three chosen pgvector builds came back as eight
   -- five full 990,000-row loads nobody chose.
8. A NEIGHBOUR WITH NO STAGE-1 POINTS is dropped, and recorded among the
   pruned builds with that reason, rather than kept. Rule 5 keeps a neighbour
   so stage 2 can compare it with stage 1; with no stage-1 point there is
   nothing to compare it with, and a build that failed or fell back to a scan
   at 90,000 rows is an hours-long failure waiting at 990,000.

Everything else explored at stage 1 is pruned, with the reason. The overlay's
`build_selection` block -- stage-1 run id and path, the grid explored, kept
and pruned builds, and every stage-1 (build, ef_search, recall, QPS) of the
kept builds for the transfer check -- is what `Manifest.set_build_selection`
records in the stage-2 manifest.

WHY RECORDS AND NOT THE ANN RESULT FILES. Reading the HDF5 results needs
h5py, which the host does not have (CLAUDE.md: python3 and pyyaml only); the
report generator runs inside a bench image for that reason and writes what it
read to `report/records.jsonl`. That file is the contract between the two, so
this reads it, and asks for `report --run-dir` when it is missing.

WHY THE FRONTIER IS NOT IMPORTED FROM report/. No orchestrator module imports
report/ today, and `report/loaders.py` imports numpy at module level, which
the host does not guarantee either. The envelope is eight lines; it is
re-stated here and pinned against the report's by test rather than bought
with a new dependency direction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .ann_knobs import KNOB_BOUNDS, excluded_builds

FLOORS = (0.90, 0.95, 0.99)
#: See rule 3 above.
NEAR_RECALL = 0.015
#: A later report flags a (build, ef_search) whose recall moved by more than
#: this between stage 1 and stage 2.
TRANSFER_TOLERANCE = 0.02
#: The build axes the overlay can express, as `render_config` spells them.
AXES = ("M", "efConstruction", "quantization")
SCHEMA = 1

REASON_REFERENCE = ("reference build of the target profile (what its ops "
                    "pass builds; `run` refuses a grid without it)")
REASON_NEIGHBOUR_UNMEASURED = (
    "M neighbour of kept build {kept}, dropped: no stage-1 point used the "
    "vector index, so there is nothing to confirm a transfer against, and a "
    "build that failed at 90k is hours lost at 990k")
REASON_NO_POINTS = ("no stage-1 point used the vector index (not measured, "
                    "failed, or every point fell back to a scan)")
REASON_PRUNED = ("no frontier point at or within +/-{near} of any floor "
                 "{floors}, not an M neighbour of a kept build, not the "
                 "reference build")

_FILE_AXES = {"M": re.compile(r"_M_(\d+)(?:_|$)"),
              "efConstruction": re.compile(r"_efConstruction_(\d+)(?:_|$)"),
              "quantization": re.compile(r"_quantization_([A-Za-z0-9]+)_")}


# --- builds -------------------------------------------------------------------

def build_of_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """A record's build in `render_config`'s spelling: M, efConstruction,
    quantization, whichever it has. Fields first, the result file name second
    -- the same fallbacks `report/builds.py:build_params` uses, for records
    merged before a field was copied."""
    extra = record.get("extra") or {}
    stem = os.path.basename(str(extra.get("source_file") or ""))
    found = {"M": record.get("m"),
             "efConstruction": record.get("ef_construction"),
             "quantization": record.get("quantization")
             or extra.get("quantization")}
    for axis, pattern in _FILE_AXES.items():
        if found[axis] is None:
            match = pattern.search(stem)
            found[axis] = match.group(1) if match else None
    build: Dict[str, Any] = {}
    for axis in AXES:
        if found[axis] is not None:
            build[axis] = (str(found[axis]) if axis == "quantization"
                           else int(found[axis]))
    return build


def build_of_grid(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """A rendered ann definition's build, cut to the axes in `AXES`."""
    return {axis: arguments[axis] for axis in AXES if axis in arguments}


def normalize(engine: str, build: Dict[str, Any]) -> Dict[str, Any]:
    """The build without axes the engine does not apply. mongot's M is a
    label `render_config` carries and never sends (`ann_knobs`), so two
    mongodb builds differing only in it are the same index."""
    if "M" in (KNOB_BOUNDS.get(engine) or {}):
        return dict(build)
    return {a: v for a, v in build.items() if a != "M"}


def key(build: Dict[str, Any]) -> str:
    return json.dumps(build, sort_keys=True)


def sorted_builds(builds: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(builds, key=lambda b: (str(b.get("quantization", "")),
                                         b.get("efConstruction", 0),
                                         b.get("M", 0)))


# --- points -------------------------------------------------------------------

def load_points(run_dir: str, resource_pass: str = "tuned",
                dataset: Optional[str] = None
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Stage-1 ann points from `<run_dir>/report/records.jsonl`, and notes.

    Points whose `vector_index_used` is False are dropped and counted: a scan
    returns exact results, slowly, and would sit on the frontier at recall
    1.0 looking like the best build there is (CLAUDE.md, the full-table-scan
    invariant).
    """
    path = os.path.join(run_dir, "report", "records.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. It is written by the report generator, which "
            f"reads the ann result files inside a bench image; run "
            f"`./run-benchmark.sh report --run-dir {run_dir}` first.")
    points: List[Dict[str, Any]] = []
    scans = 0
    datasets = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (record.get("phase") != "recall_qps"
                    or record.get("resource_pass") != resource_pass
                    or record.get("recall_at_k") is None
                    or record.get("qps") is None):
                continue
            if dataset and record.get("dataset") != dataset:
                continue
            if record.get("vector_index_used") is False:
                scans += 1
                continue
            datasets.add(record.get("dataset"))
            points.append({"engine": record.get("engine"),
                           "dataset": record.get("dataset"),
                           "build": build_of_record(record),
                           "ef_search": record.get("ef_search"),
                           "recall": float(record["recall_at_k"]),
                           "qps": float(record["qps"])})
    if len(datasets) > 1:
        raise ValueError(
            f"stage-1 records cover {sorted(datasets)}; a selection is made on "
            f"one corpus. Pass --dataset.")
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return points, {"records": path, "records_sha256": digest,
                    "dataset": next(iter(datasets), dataset),
                    "points_dropped_scan": scans}


def frontier(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Upper-left envelope of (recall, qps): `report.loaders.pareto_frontier`
    restated -- see the module docstring for why it is not imported."""
    ordered = sorted(points, key=lambda p: (p["recall"], p["qps"]))
    out: List[Dict[str, Any]] = []
    best = float("-inf")
    for point in reversed(ordered):
        if point["qps"] > best:
            out.append(point)
            best = point["qps"]
    return list(reversed(out))


# --- the rule -----------------------------------------------------------------

def _frontier_reasons(points: Sequence[Dict[str, Any]],
                      floors: Sequence[float], near: float
                      ) -> Tuple[Dict[str, List[str]], List[float]]:
    """Rules 2-4: the builds the frontier keeps, with reasons, and the floors
    no point reached."""
    kept: Dict[str, List[str]] = {}
    unreached: List[float] = []
    envelope = frontier(points)

    def keep(point: Dict[str, Any], reason: str) -> None:
        reasons = kept.setdefault(key(point["build"]), [])
        if reason not in reasons:
            reasons.append(reason)

    for floor in floors:
        above = [p for p in envelope if p["recall"] >= floor]
        if above:
            p = min(above, key=lambda q: q["recall"])
            keep(p, f"frontier winner at floor {floor:.2f}: recall "
                    f"{p['recall']:.4f}, {p['qps']:.1f} QPS at ef_search "
                    f"{p['ef_search']}")
        elif points:
            unreached.append(floor)
            p = max(points, key=lambda q: (q["recall"], q["qps"]))
            keep(p, f"floor {floor:.2f} unreached; highest recall measured "
                    f"({p['recall']:.4f} at ef_search {p['ef_search']})")
        for p in envelope:
            if abs(p["recall"] - floor) <= near + 1e-12:
                keep(p, f"frontier point near floor {floor:.2f}: recall "
                        f"{p['recall']:.4f} at ef_search {p['ef_search']} "
                        f"(within +/-{near})")
    return kept, unreached


def _neighbours(engine: str, build: Dict[str, Any],
                explored: Sequence[Dict[str, Any]]
                ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Rule 5: the explored builds one M step either side, same other axes,
    and an edge note per side that has none.

    The note says WHY there is none. Past the grid's own edge the frontier
    may continue; but pgvector's M=32 at ef_construction 64 has M=48 above it
    in the grid and not at 64, because pgvector cannot build it there -- a
    different statement, and "largest M explored" would have hidden it.
    """
    if "M" not in build:
        return [], []
    rest = {a: v for a, v in build.items() if a != "M"}
    ms = sorted({b["M"] for b in explored if "M" in b and
                 {a: v for a, v in b.items() if a != "M"} == rest})
    every_m = sorted({b["M"] for b in explored if "M" in b})
    m = build["M"]
    found, edges = [], []
    for side, pick in (("below", max), ("above", min)):
        here = [v for v in ms if (v < m if side == "below" else v > m)]
        if here:
            found.append({**rest, "M": pick(here)})
            continue
        elsewhere = [v for v in every_m
                     if (v < m if side == "below" else v > m)]
        illegal = (excluded_builds(engine, [pick(elsewhere)],
                                   [rest["efConstruction"]])
                   if elsewhere and "efConstruction" in rest else [])
        if illegal:
            edges.append(f"M={m} at {key(rest)} has no neighbour {side} it: "
                         f"M={illegal[0][0]} is in the grid but cannot be "
                         f"built here -- {illegal[0][2]}")
        else:
            edges.append(f"M={m} is the "
                         f"{'smallest' if side == 'below' else 'largest'} M "
                         f"explored at {key(rest) if rest else 'this engine'}"
                         f"; the frontier may continue {side} it")
    return found, edges


def select_engine(engine: str, points: Sequence[Dict[str, Any]],
                  explored: Sequence[Dict[str, Any]],
                  reference: Optional[Dict[str, Any]],
                  floors: Sequence[float] = FLOORS,
                  near: float = NEAR_RECALL) -> Dict[str, Any]:
    """Apply rules 2-8 to one engine. Pure: points and grids in, lists out."""
    points = [{**p, "build": normalize(engine, p["build"])} for p in points]
    explored = [normalize(engine, b) for b in explored]
    reference = normalize(engine, reference) if reference else None
    kept, unreached = _frontier_reasons(points, floors, near)
    measured = {key(p["build"]) for p in points}
    edges: List[str] = []
    dropped: Dict[str, str] = {}
    by_key = {key(b): b for b in explored}
    for point in points:
        by_key.setdefault(key(point["build"]), point["build"])
    for chosen in [json.loads(k) for k in list(kept)]:
        found, notes = _neighbours(engine, chosen, list(by_key.values()))
        edges += [n for n in notes if n not in edges]
        for neighbour in found:
            if key(neighbour) not in measured:
                dropped.setdefault(key(neighbour),
                                   REASON_NEIGHBOUR_UNMEASURED.format(
                                       kept=key(chosen)))
                continue
            reasons = kept.setdefault(key(neighbour), [])
            reason = f"M neighbour of kept build {key(chosen)}"
            if reason not in reasons:
                reasons.append(reason)
    if reference:
        kept.setdefault(key(reference), []).append(REASON_REFERENCE)
        by_key.setdefault(key(reference), reference)
    builds = [json.loads(k) for k in sorted(kept)]
    generic = REASON_PRUNED.format(
        near=near, floors="/".join(f"{f:.2f}" for f in floors))
    pruned = [{"build": b, "reason": dropped.get(k) or (
                   generic if k in measured else REASON_NO_POINTS)}
              for k, b in by_key.items() if k not in kept]
    return {
        "kept": [{"build": json.loads(k), "reasons": r}
                 for k, r in sorted(kept.items())],
        "pruned": sorted(pruned, key=lambda e: key(e["build"])),
        "edges": edges,
        "floors_unreached": unreached,
        "lists": overlay_lists(engine, builds, reference),
        "builds": sorted_builds(builds),
        "explored_count": len(by_key),
    }


def overlay_lists(engine: str, builds: Sequence[Dict[str, Any]],
                  reference: Optional[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """The per-axis value lists the overlay writes for one engine.

    They make every kept value legal for the render; which COMBINATIONS are
    built is `ann_builds_by_engine` (rule 7). M only for an engine with a
    graph-degree knob (`ann_knobs.KNOB_BOUNDS`): mongot's M is a label.
    Ascending, except quantization, whose reference value goes first: the ops
    pass takes the head of `mongodb_quantization` (`ops_pass._quantization`)
    and the headline build must not move. No other list is read by position.
    """
    lists: Dict[str, List[Any]] = {}
    for axis in AXES:
        if axis == "M" and "M" not in (KNOB_BOUNDS.get(engine) or {}):
            continue
        values = sorted({b[axis] for b in builds if axis in b}, key=str
                        if axis == "quantization" else int)
        if not values:
            continue
        head = (reference or {}).get(axis) if axis == "quantization" else None
        if head is not None and head in values:
            values = [head] + [v for v in values if v != head]
        lists[axis] = values
    return lists
