"""`select-builds`: stage-1 run directory in, stage-2 overlay out.

The rule lives in `build_selection` and is pure; this module does the I/O
around it -- the stage-1 manifest and records, the grid that run rendered,
the target profile's reference builds -- and checks the overlay it writes the
way `run` will read it, before writing it. An overlay `run` would refuse, or
one whose rendered grid is not the set of builds this module says it kept, is
refused here instead, where the cause is still on screen.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import yaml

from . import build_selection as sel
from .ann_pass import excluded_ann_builds
from .config import load_profile, load_resources, merge_resource_overrides
from .estimate_grid import ann_builds
from .manifest import Manifest
from .profile_overlay import OVERLAY_EXTRAS, apply_overlay, overlay_digest
from .reference_build import (GRID_ARGUMENT, reference_build_problem,
                              resolve_reference_build)

#: The only pass whose extras `render_config` reads; see `profile_overlay`.
PASS = "tuned"


def _resources(profile: Dict[str, Any]) -> Dict[str, Any]:
    return merge_resource_overrides(load_resources(PASS), profile)


def _explored(engine: str, profile: Dict[str, Any],
              points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The grid stage 1 rendered for this engine, from its recorded profile.

    Rendered from the manifest's copy of the profile, so it is the grid that
    run asked for, against today's resource pass file. If that no longer
    renders (an engine bound tightened, a key retired) the builds seen in the
    records stand in, which is what was measured even if not what was asked.
    """
    try:
        return [sel.build_of_grid(b) for b in ann_builds(
            engine, profile, _resources(profile), PASS)]
    except ValueError:
        return [p["build"] for p in points]


def _reference(engine: str, target: Dict[str, Any]) -> Dict[str, Any]:
    build = resolve_reference_build(engine, target, _resources(target), PASS)
    return {GRID_ARGUMENT[axis]: value for axis, value in build.items()}


def _extras(chosen: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The overlay's `resources.extras`, from each engine's lists."""
    extras: Dict[str, Any] = {}
    for engine, result in sorted(chosen.items()):
        lists = result["lists"]
        if "M" in lists:
            extras.setdefault("m_values_by_engine", {})[engine] = lists["M"]
        for axis, name in (("efConstruction", f"{engine}_ef_construction"),
                           ("quantization", f"{engine}_quantization")):
            if axis not in lists:
                continue
            if name not in OVERLAY_EXTRAS:
                # Never written to a key nothing reads: the selection would
                # vanish and the run would measure the target's default grid.
                raise ValueError(f"{engine}: no resources.extras key carries "
                                 f"its {axis} axis ({name})")
            extras[name] = lists[axis]
        # The builds themselves; the lists above only make their values
        # legal for the render (build_selection rule 7: no closure).
        extras.setdefault("ann_builds_by_engine", {})[engine] = result["builds"]
    return extras


def select(stage1_dir: str, target_name: str = "tuned-complete",
           dataset: Optional[str] = None) -> Dict[str, Any]:
    """The overlay document for `run --profile <target_name> --overlay`."""
    manifest = Manifest.load(stage1_dir)
    config = manifest.get("config") or {}
    stage1_profile = config.get("profile") or {}
    points, notes = sel.load_points(stage1_dir, PASS, dataset)
    if not points:
        raise ValueError(f"{stage1_dir}: no {PASS}-pass recall points that "
                         f"used the vector index; nothing to select from")
    # build_mode is not an axis the overlay can spell, so points from two
    # modes would merge under one build and the winner could belong to one.
    ann = stage1_profile.get("ann") or {}
    for name in ("pgvector_build_modes", "valkey_build_modes"):
        if len(ann.get(name) or ["post"]) > 1:
            raise ValueError(f"stage 1 swept ann.{name} {ann[name]}; a "
                             f"selection cannot tell the modes apart")
    # The same for MariaDB's storage engine: points from InnoDB and MyISAM
    # would merge under one M.
    extras = _resources(stage1_profile).get("extras") or {}
    storages = (extras.get("mariadb_storage_engines")
                or ann.get("mariadb_storage_engines") or ["InnoDB"])
    if len(storages) > 1:
        raise ValueError(f"stage 1 swept mariadb_storage_engines {storages}; "
                         f"a selection cannot tell the storage engines apart")
    target = load_profile(target_name)
    chosen: Dict[str, Dict[str, Any]] = {}
    grid: Dict[str, Any] = {}
    stage1_excluded: Dict[str, Any] = {}
    for engine in sorted({p["engine"] for p in points}):
        mine = [p for p in points if p["engine"] == engine]
        explored = _explored(engine, stage1_profile, mine)
        chosen[engine] = sel.select_engine(engine, mine, explored,
                                           _reference(engine, target))
        grid[engine] = sel.sorted_builds(
            sel.normalize(engine, b) for b in explored)
        try:
            dropped = excluded_ann_builds(engine, stage1_profile,
                                          _resources(stage1_profile), PASS)
        except ValueError:
            dropped = []
        stage1_excluded[engine] = [
            {"build": {"M": m, "efConstruction": e}, "reason": r}
            for m, e, r in dropped]
    document = _document(stage1_dir, manifest, notes, target_name, chosen,
                         grid, stage1_excluded, points)
    # Engines stage 1 set out to measure that left no usable point. The
    # overlay sets nothing for them, so stage 2 measures the target profile's
    # own grid there -- stated, never left for a reader to infer from an
    # absence.
    attempted = set(manifest.get("engines") or {}) | {
        p.get("engine") for p in manifest.get("phases") or []
        if isinstance(p, dict) and p.get("engine")}
    document["build_selection"]["unselected"] = {
        engine: (f"no stage-1 point used the vector index; {target_name}'s "
                 f"own grid is measured for it")
        for engine in sorted(attempted - set(chosen))}
    document["build_selection"]["overlay_sha256"] = overlay_digest(
        document["resources"]["extras"], document["build_selection"])
    return document


def _transfer_points(chosen: Dict[str, Dict[str, Any]],
                     points: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for engine, result in chosen.items():
        kept = {sel.key(k["build"]) for k in result["kept"]}
        rows = [{"build": sel.normalize(engine, p["build"]),
                 "ef_search": p["ef_search"],
                 "recall": round(p["recall"], 6), "qps": round(p["qps"], 3)}
                for p in points if p["engine"] == engine
                and sel.key(sel.normalize(engine, p["build"])) in kept]
        out[engine] = sorted(rows, key=lambda r: (sel.key(r["build"]),
                                                  r["ef_search"] or 0))
    return out


def _document(stage1_dir: str, manifest: Dict[str, Any], notes: Dict[str, Any],
              target_name: str, chosen: Dict[str, Dict[str, Any]],
              grid: Dict[str, Any], stage1_excluded: Dict[str, Any],
              points: List[Dict[str, Any]]) -> Dict[str, Any]:
    config = manifest.get("config") or {}
    floors = list(sel.FLOORS)
    selection = {
        "schema": sel.SCHEMA,
        "stage1": {
            "run_id": manifest.get("run_id"),
            "run_dir": os.path.abspath(stage1_dir),
            "profile": (config.get("profile") or {}).get("name"),
            "dataset": notes["dataset"],
            "resource_pass": PASS,
            "ann_fingerprint": config.get("ann_fingerprint"),
            "records": notes["records"],
            "records_sha256": notes["records_sha256"],
            "points_dropped_scan": notes["points_dropped_scan"],
        },
        "target": {"profile": target_name, "resource_pass": PASS},
        "rule": {
            "floors": floors,
            "near_recall": sel.NEAR_RECALL,
            "near": "a frontier point with |recall - floor| <= near_recall",
            "winner": "the frontier point with the smallest recall >= floor",
            "neighbours": "the next explored M below and above each build "
                          "the frontier kept, at the same ef_construction "
                          "and quantization",
            "reference_build": "always kept",
            "closure": "every build in the product of the kept lists",
        },
        "grid_explored": grid,
        "stage1_excluded": stage1_excluded,
        "kept": {e: r["kept"] for e, r in chosen.items()},
        "pruned": {e: r["pruned"] for e, r in chosen.items()},
        # Always zero since the overlay names its builds; recorded so a reader
        # of the manifest does not have to know that to trust the kept list.
        "closure_builds": 0,
        "edges": {e: r["edges"] for e, r in chosen.items() if r["edges"]},
        "floors_unreached": {e: r["floors_unreached"]
                             for e, r in chosen.items()
                             if r["floors_unreached"]},
        "transfer_check": {
            "tolerance": sel.TRANSFER_TOLERANCE,
            "compare": "recall_at_k at equal (engine, build, ef_search) "
                       "between these stage-1 points and the stage-2 run; "
                       "flag a difference above the tolerance",
            "stage1_points": _transfer_points(chosen, points),
        },
    }
    return {"overlay_of": target_name,
            "resources": {"extras": _extras(chosen)},
            "build_selection": selection}


def check(document: Dict[str, Any]) -> List[str]:
    """Problems `run` would hit with this overlay, or a grid that is not the
    kept set. Empty when the overlay does exactly what it says."""
    target = load_profile(document["overlay_of"])
    merged, selection = apply_overlay(target, document)
    resources = _resources(merged)
    problems: List[str] = []
    for engine, kept in (selection or {}).get("kept", {}).items():
        problem = reference_build_problem(engine, merged, resources, PASS)
        if problem:
            problems.append(problem)
        rendered = {sel.key(sel.normalize(engine, sel.build_of_grid(b)))
                    for b in ann_builds(engine, merged, resources, PASS)}
        wanted = {sel.key(k["build"]) for k in kept}
        if rendered != wanted:
            problems.append(
                f"{engine}: the overlay renders {sorted(rendered - wanted)} "
                f"beyond the kept builds and misses {sorted(wanted - rendered)}")
    return problems


HEADER = """\
# GENERATED by `./run-benchmark.sh select-builds`. Do not edit by hand: the
# build_selection block below is recorded in the stage-2 run manifest as the
# provenance of this grid, and a hand edit would make it describe a grid the
# run did not measure. Regenerate instead.
#
#   ./run-benchmark.sh run --profile {target} --resource-pass tuned \\
#       --overlay {out}
#
"""


def write(document: Dict[str, Any], out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(HEADER.format(target=document["overlay_of"], out=out))
        yaml.safe_dump(document, fh, sort_keys=False, default_flow_style=False,
                       width=100)
    os.replace(tmp, out)


def summary(document: Dict[str, Any]) -> List[str]:
    """The lines `select-builds` prints: per engine, kept / pruned / lists."""
    block = document["build_selection"]
    lines = [f"stage 1: {block['stage1']['run_id']} "
             f"({block['stage1']['dataset']}), target "
             f"{document['overlay_of']}"]
    extras = document["resources"]["extras"]
    for engine in sorted(block["kept"]):
        kept, pruned = block["kept"][engine], block["pruned"][engine]
        lists = {k: v for k, v in (
            ("M", (extras.get("m_values_by_engine") or {}).get(engine)),
            ("efC", extras.get(f"{engine}_ef_construction")),
            ("quantization", extras.get(f"{engine}_quantization"))) if v}
        explored = len(block["grid_explored"].get(engine) or [])
        lines.append(f"  {engine:<11} keep {len(kept):>2} of {explored:>2} "
                     f"explored (exact builds, 0 closure), prune "
                     f"{len(pruned):>2}  {json.dumps(lists)}")
        for edge in (block.get("edges") or {}).get(engine, []):
            lines.append(f"    ! {edge}")
        for floor in (block.get("floors_unreached") or {}).get(engine, []):
            lines.append(f"    ! floor {floor:.2f} not reached at stage 1")
    for engine, reason in (block.get("unselected") or {}).items():
        lines.append(f"  {engine:<11} ! {reason}")
    return lines
