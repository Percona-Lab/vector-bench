"""Which swept build the ops pass should measure, chosen off the ann frontier.

WHY THIS EXISTS. A reviewer of the published benchmark pointed out that
comparing engines at one M is meaningless: an identically named parameter is
a different amount of work in a different algorithm, so "M=16 for everyone"
hands each engine a different budget under one label. The headline moved to
the Pareto frontier over each engine's whole swept parameter space. The ops
pass -- build cost, concurrency, filtered, churn -- still built the fixed
reference build declared in `config/engines/<e>.yml`, so a report could put
a frontier figure from M=32 beside concurrency and churn measured on M=16
and present them as one engine's numbers.

With `ops.build_selection: frontier` the host runs this module after the ann
phase of an (engine, dataset, pass) and before its ops phase, and the ops
pass builds what it picks.

THE RULE. Group the ann phase's points by build (`builds.build_signature`:
everything about a measurement except the query effort). Take each build's
own Pareto frontier (`loaders.pareto_frontier`) and, on it, the fastest
point at or above the target recall -- the same target the ops pass's
ef_search calibration then chases on the index it builds, read from one
place on the host and passed in here. The build whose qualifying point is
fastest wins. A build with no point at the target does not compete.

TIE-BREAK, deterministic and in this order: higher QPS; then higher recall
at the qualifying point; then the cheaper build -- smaller M, then smaller
ef_construction (a missing value sorts as 0, the engine having no knob) --
because between two indexes that serve the target equally fast the one that
costs less to build and hold is what an operator would deploy; then the
build signature as a string, which is stable across runs and machines. QPS
is a float measured to three decimals, so an exact tie is rare, but "rare"
is not "never" and the choice has to be the same on every re-read.

NOTHING REACHES THE TARGET. Then the build with the highest maximum recall
wins (ties: higher QPS at that point, then the same cheaper-build and
signature order), and the status says `below_target`. That is the build
closest to the target the calibration is about to chase, and the record
says it never got there.

WHY HERE AND NOT IN THE ORCHESTRATOR. The points come out of
ann-benchmarks' HDF5 files, and recall is computed against the corpus's
ground truth (`loaders.knn_recall`). Both need h5py and numpy, which the
host deliberately lacks, and `report/` must not import the orchestrator.
So the whole choice runs inside a bench image, beside the functions it
reuses, and the host reads back one JSON line. A second copy of the
frontier rule on the host is how the 25 Aug defect happened the first time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Prefix of the one stdout line the host parses. The bench image prints
#: other things (h5py warnings, the loader's per-file complaints), so the
#: answer is marked rather than assumed to be the last line.
RESULT_MARKER = "VB_OPS_BUILD_CHOICE "

STATUS_SELECTED = "selected"
STATUS_BELOW_TARGET = "below_target"


def _cost_key(point: Dict[str, Any]) -> Tuple[int, int]:
    return (int(point.get("m") or 0), int(point.get("ef_construction") or 0))


def _usable(point: Dict[str, Any]) -> bool:
    """A point that measured the index, with both coordinates.

    `vector_index_used is False` is a full table scan: exact and slow,
    indistinguishable from "very accurate" by recall alone. It never wins on
    QPS at a real target, but a below-target fallback ranks by recall, and a
    scan would win that ranking for the wrong reason.
    """
    return (point.get("recall_at_k") is not None
            and point.get("qps") is not None
            and point.get("vector_index_used") is not False)


def _describe(signature: Any, point: Dict[str, Any],
              frontier_size: int) -> Dict[str, Any]:
    from report.builds import build_params

    params = build_params(point)
    extra = point.get("extra") or {}
    return {
        "signature": str(signature),
        "build": {"m": params["m"],
                  "ef_construction": params["ef_construction"],
                  "quantization": params["quantization"],
                  "storage_engine": point.get("storage_engine"),
                  "build_mode": point.get("build_mode")},
        "point": {"recall": point.get("recall_at_k"), "qps": point.get("qps"),
                  "ef_search": point.get("ef_search"),
                  "source_file": extra.get("source_file")},
        "frontier_points": frontier_size,
    }


def choose_ops_build(points: Sequence[Dict[str, Any]],
                     target_recall: float) -> Optional[Dict[str, Any]]:
    """The build the ops pass should measure, or None with nothing usable.

    `points` are recall/QPS records of ONE engine on ONE dataset in one
    pass, as `loaders.load_annb_results` emits them. See the module
    docstring for the rule, the tie-break and the below-target fallback.
    """
    from report.builds import build_signature
    from report.loaders import pareto_frontier

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for point in points:
        if _usable(point):
            groups.setdefault(build_signature(point), []).append(point)
    if not groups:
        return None

    reaching: List[Tuple[Any, Dict[str, Any], int]] = []
    ceilings: List[Tuple[Any, Dict[str, Any], int]] = []
    for signature, group in groups.items():
        frontier = pareto_frontier(group)
        qualifying = [p for p in frontier
                      if p["recall_at_k"] >= target_recall]
        if qualifying:
            best = max(qualifying, key=lambda p: (p["qps"], p["recall_at_k"]))
            reaching.append((signature, best, len(frontier)))
        # The frontier's last point is its highest recall, and the fastest
        # point at that recall by construction.
        ceilings.append((signature, frontier[-1], len(frontier)))

    if reaching:
        signature, point, size = min(
            reaching, key=lambda c: (-c[1]["qps"], -c[1]["recall_at_k"],
                                     _cost_key(c[1]), str(c[0])))
        status = STATUS_SELECTED
    else:
        signature, point, size = min(
            ceilings, key=lambda c: (-c[1]["recall_at_k"], -c[1]["qps"],
                                     _cost_key(c[1]), str(c[0])))
        status = STATUS_BELOW_TARGET
    return {"status": status, "target_recall": target_recall,
            "builds_considered": len(groups),
            "builds_reaching_target": len(reaching),
            **_describe(signature, point, size)}


def _link_tree(annb_dir: str, files: Sequence[str]) -> str:
    """A directory holding exactly `files`, at their paths relative to the tree.

    `load_annb_results` walks a directory, and the ann tree for this pass
    also holds every other engine, every other corpus and -- the 25 Aug case
    -- files of this engine that the phase never asked for. Linking only the
    files the ann phase wrote or stood on keeps the loader unmodified and the
    choice confined to THAT phase. The relative layout is kept because the
    loader names the engine after the directory a file sits in.
    """
    root = tempfile.mkdtemp(prefix="vb-ops-build-")
    for relative in files:
        relative = os.path.normpath(relative)
        if relative.startswith("..") or os.path.isabs(relative):
            continue
        source = os.path.join(annb_dir, relative)
        if not os.path.isfile(source):
            continue
        target = os.path.join(root, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.symlink(source, target)
    return root


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annb-dir", required=True)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-recall", type=float, required=True)
    parser.add_argument("--files", required=True,
                        help="JSON list of result files, relative to "
                             "--annb-dir, that the ann phase wrote or reused")
    args = parser.parse_args(argv)

    from report.loaders import load_annb_results

    files = json.loads(args.files)
    tree = _link_tree(args.annb_dir, files)
    points = [p for p in load_annb_results(tree, args.datasets_dir, "")
              if p.get("engine") == args.engine
              and p.get("dataset") == args.dataset]
    choice = choose_ops_build(points, args.target_recall)
    payload = {"points": len(points), "files": len(files), "choice": choice}
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sys.exit(main())
