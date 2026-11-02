"""`ops.build_selection`: which build the ops pass measures.

THE GAP THIS CLOSES. The report's headline is the Pareto frontier over each
engine's whole swept parameter space, because a reviewer of the published
benchmark pointed out -- correctly -- that "M=16 for every engine" is not
equal work: an identically named parameter is a different amount of effort
in a different algorithm. The ops pass (build cost, concurrency, filtered,
churn) kept building one fixed build per engine, the `runtime.reference_build`
declared in `config/engines/<e>.yml`. Once the headline left that build, a
report could print a frontier figure from M=32 beside concurrency and churn
measured on M=16 and call both "the engine" -- the 25 Aug defect
(`reference_build.py`) again, in a new direction.

TWO MODES.

* `reference` -- the default, and what every profile did before this key
  existed: the ops pass builds `ops.m_values` with the profile-derived
  quantization and ef_construction, and the manifest names the declared
  reference build. Nothing here runs.
* `frontier` -- after the ann phase of an (engine, dataset, pass) and before
  its ops phase, `report/ops_build_choice.py` reads THAT ann phase's result
  files (the ones `ann_provenance` recorded it wrote or reused, so a stale
  file from another grid cannot be chosen), and picks the build whose
  frontier point at the ops calibration target is fastest. The ops pass then
  builds exactly that M, ef_construction and quantization, and the manifest
  presentation names it, with its source, per dataset.

NEVER SILENTLY. Three outcomes, each recorded on the manifest and on every
ops record (`extra.ops_build_selection.status`):

* `selected` -- a build reached the target; source `frontier@<target>`.
* `below_target` -- none did; the build with the highest recall is used, and
  a warning says so. The calibration that follows will report it missed.
* `fallback_reference` -- the ann phase did not run in this run directory,
  failed, or produced no usable point; the declared reference build is
  measured, as `reference` mode would, and a warning says why.

RESUME. The choice is persisted in the manifest under
`ops_build_selection["<pass>/<engine>/<dataset>"]` the moment it is made,
and a resumed run reuses it rather than choosing again. Choosing again could
pick a different build -- a re-run ann phase can move a QPS by more than the
gap between two builds -- and the ops units checkpointed under the first
choice would then sit beside units measured on another index, under one
engine name.

A module of its own because `cli.py` and `ops_pass.py` are far past the
repository's line ceiling and a sibling change is editing `cli.py` too; the
call sites there are a few lines each.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from . import docker_ctl
from . import engines as engines_mod
from .ann_pass import annb_results_dir
from .manifest import utcnow
from .ops_pass import ops_target_recall
from .reference_build import resolve_reference_build

MODE_REFERENCE = "reference"
MODE_FRONTIER = "frontier"
MODES = (MODE_REFERENCE, MODE_FRONTIER)

#: The two statuses `report/ops_build_choice.py` returns, spelled again
#: because a module on the host cannot import one that needs numpy to load.
#: `tests/test_ops_build_selection.py` pins the two spellings together.
STATUS_SELECTED = "selected"
STATUS_BELOW_TARGET = "below_target"
STATUS_FALLBACK = "fallback_reference"

#: Top-level manifest key the choices persist under.
MANIFEST_KEY = "ops_build_selection"
#: Key under `engines.<name>.presentation` the report reads per dataset.
PRESENTATION_KEY = "reference_builds_by_dataset"
#: Mirrors `report/ops_build_choice.py`'s `RESULT_MARKER`; pinned by a test.
RESULT_MARKER = "VB_OPS_BUILD_CHOICE "

Chooser = Callable[..., Optional[Dict[str, Any]]]


def build_selection_mode(profile: Dict[str, Any]) -> str:
    """`ops.build_selection`, validated. `reference` when absent.

    An unknown value raises rather than meaning `reference`: a typo in a
    profile that asked for the frontier would otherwise measure the fixed
    build and label nothing, which is the outcome this key exists to end.
    """
    value = ((profile.get("ops") or {}).get("build_selection")
             or MODE_REFERENCE)
    if value not in MODES:
        raise ValueError(f"ops.build_selection must be one of "
                         f"{', '.join(MODES)}, got {value!r}")
    return str(value)


def selection_key(resource_pass: str, engine: str, dataset: str) -> str:
    return f"{resource_pass}/{engine}/{dataset}"


def stored_selection(manifest_data: Dict[str, Any], resource_pass: str,
                     engine: str, dataset: str) -> Optional[Dict[str, Any]]:
    block = manifest_data.get(MANIFEST_KEY) or {}
    entry = block.get(selection_key(resource_pass, engine, dataset))
    return dict(entry) if isinstance(entry, dict) else None


def ann_phase_files(manifest_data: Dict[str, Any], engine: str, dataset: str,
                    resource_pass: str) -> Optional[List[str]]:
    """The result files the latest completed ann attempt wrote or reused.

    None when no ann attempt of this unit completed in this run directory --
    the phase was not asked for (`--phases ops`), failed, or ran under code
    that recorded no files. The latest completed attempt, because it is the
    one whose grid the ops pass follows; its `skipped` list already names the
    earlier attempts' files it stood on.
    """
    attempts = [p for p in manifest_data.get("phases") or []
                if isinstance(p, dict) and p.get("phase") == "ann"
                and p.get("engine") == engine and p.get("dataset") == dataset
                and p.get("resource_pass") == resource_pass
                and p.get("status") == "completed"
                and isinstance(p.get("files"), dict)]
    if not attempts:
        return None
    files = attempts[-1]["files"]
    paths = [entry.get("path") for key in ("written", "skipped")
             for entry in files.get(key) or [] if isinstance(entry, dict)]
    return sorted({p for p in paths if p})


def run_choice_container(*, image: str, ann_tree: str, datasets_dir: str,
                         vb_root: str, engine: str, dataset: str,
                         files: List[str],
                         target_recall: float) -> Optional[Dict[str, Any]]:
    """Run `report/ops_build_choice.py` in a bench image; its payload or None.

    In a container for the reason `cli.generate_report` is: the choice needs
    h5py and numpy to read the result files and score recall, and the host
    has neither by design.
    """
    output: List[str] = []
    spec = docker_ctl.ContainerSpec(
        name=f"vb-ops-build-{os.getpid()}",
        image=image,
        network="none",
        entrypoint="python3",
        workdir="/vb",
        env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/vb"},
        volumes=[
            f"{os.path.join(vb_root, 'report')}:/vb/report:ro",
            f"{os.path.join(vb_root, 'harness')}:/vb/harness:ro",
            f"{ann_tree}:/annb:ro",
            f"{datasets_dir}:/datasets:ro",
        ],
        command=["/vb/report/ops_build_choice.py",
                 "--annb-dir", "/annb", "--datasets-dir", "/datasets",
                 "--engine", engine, "--dataset", dataset,
                 "--target-recall", str(target_recall),
                 "--files", json.dumps(files)],
        detach=False,
    )
    try:
        rc = docker_ctl.run_foreground(spec, timeout=1800, stream=False,
                                       sink=output)
    except docker_ctl.DockerError as exc:
        print(f"[ops] build selection container failed: {exc}",
              file=sys.stderr)
        return None
    if rc != 0:
        return None
    for line in reversed(output):
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER):])
            except ValueError:
                return None
    return None


def _declared_axes(engine: str) -> List[str]:
    return [axis for axis, _ in engines_mod.get(engine).reference_build]


def _presentation_build(engine: str, build: Dict[str, Any]) -> Dict[str, Any]:
    """The chosen build on the axes the engine's reference build names.

    The report matches ann points to a reference build axis by axis
    (`report/builds.matches`), so the frontier choice is published on the
    same axes the declared one was: M for MariaDB, quantization for Percona
    Search, M and ef_construction where both are knobs. Storage engine and
    build mode stay out, as they do for a declared build -- the ops pass
    still measures every declared storage engine and build mode, and the
    report names the one that won each cell.
    """
    return {axis: build.get(axis) for axis in _declared_axes(engine)}


def _fallback(engine: str, profile: Dict[str, Any], resources: Dict[str, Any],
              resource_pass: str, target: float, reason: str
              ) -> Dict[str, Any]:
    reference = resolve_reference_build(engine, profile, resources,
                                        resource_pass)
    return {"mode": MODE_FRONTIER, "status": STATUS_FALLBACK,
            "source": STATUS_FALLBACK, "target_recall": target,
            "build": dict(reference), "reference_build": dict(reference),
            "ann_point": None, "reason": reason}


def _from_choice(engine: str, choice: Dict[str, Any], target: float
                 ) -> Dict[str, Any]:
    status = choice["status"]
    source = (f"frontier@{target:g}" if status == STATUS_SELECTED
              else f"{STATUS_BELOW_TARGET}@{target:g}")
    build = dict(choice.get("build") or {})
    reason = None
    if status == STATUS_BELOW_TARGET:
        reason = (f"none of the {choice.get('builds_considered')} swept "
                  f"build(s) reached recall {target:g}; measuring the one "
                  f"with the highest recall "
                  f"({(choice.get('point') or {}).get('recall')})")
    return {"mode": MODE_FRONTIER, "status": status, "source": source,
            "target_recall": target, "build": build,
            "reference_build": _presentation_build(engine, build),
            "ann_point": choice.get("point"),
            "signature": choice.get("signature"),
            "builds_considered": choice.get("builds_considered"),
            "builds_reaching_target": choice.get("builds_reaching_target"),
            "reason": reason}


def choose(manifest_data: Dict[str, Any], *, engine: str, dataset: str,
           resource_pass: str, profile: Dict[str, Any],
           resources: Dict[str, Any], chooser: Callable[..., Any]
           ) -> Dict[str, Any]:
    """A fresh selection for one unit. `chooser(files=..., target_recall=...)`
    returns the container's payload, or None when it could not run."""
    target = ops_target_recall(profile)
    files = ann_phase_files(manifest_data, engine, dataset, resource_pass)
    if not files:
        return _fallback(engine, profile, resources, resource_pass, target,
                         "no completed ann phase of this engine and dataset "
                         "in this run directory recorded any result file")
    payload = chooser(files=files, target_recall=target)
    choice = (payload or {}).get("choice")
    if not choice:
        why = ("the selection could not read the ann results"
               if payload is None else
               f"the ann phase's {len(files)} result file(s) held no usable "
               f"recall/QPS point")
        return _fallback(engine, profile, resources, resource_pass, target,
                         why)
    return _from_choice(engine, choice, target)


def _record(manifest: Any, engine: str, dataset: str, resource_pass: str,
            selection: Dict[str, Any]) -> None:
    """Persist the choice and fold it into the engine's presentation."""
    data = manifest.data
    data[MANIFEST_KEY] = {**(data.get(MANIFEST_KEY) or {}),
                          selection_key(resource_pass, engine, dataset):
                              selection}
    info = (data.get("engines") or {}).get(engine)
    if isinstance(info, dict):
        info["presentation"] = {**(info.get("presentation") or {}),
                                **presentation_entries(data, engine)}
    manifest.save()


def presentation_entries(manifest_data: Dict[str, Any], engine: str
                         ) -> Dict[str, Any]:
    """`{PRESENTATION_KEY: {pass: {dataset: {build, source, status}}}}`.

    Empty when nothing was chosen for this engine, so a `reference` run's
    presentation is exactly what it was. `cmd_run` merges this into the
    presentation it rewrites on every pass, or a resumed run would erase the
    choices an earlier invocation recorded.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for key, selection in sorted((manifest_data.get(MANIFEST_KEY)
                                  or {}).items()):
        resource_pass, chosen_engine, dataset = (key.split("/", 2) + ["", ""])[:3]
        if chosen_engine != engine or not isinstance(selection, dict):
            continue
        out.setdefault(resource_pass, {})[dataset] = {
            "build": selection.get("reference_build") or {},
            "source": selection.get("source"),
            "status": selection.get("status")}
    return {PRESENTATION_KEY: out} if out else {}


def ensure_selection(manifest: Any, *, engine: str, dataset: str,
                     resource_pass: str, profile: Dict[str, Any],
                     resources: Dict[str, Any], chooser: Callable[..., Any]
                     ) -> Dict[str, Any]:
    """The stored choice for this unit, or a new one, recorded and announced."""
    stored = stored_selection(manifest.data, resource_pass, engine, dataset)
    if stored is not None:
        print(f"[ops] reusing the build chosen earlier in this run: "
              f"{_spell(stored)} ({stored.get('source')})")
        return stored
    selection = {**choose(manifest.data, engine=engine, dataset=dataset,
                          resource_pass=resource_pass, profile=profile,
                          resources=resources, chooser=chooser),
                 "selected_at": utcnow()}
    _record(manifest, engine, dataset, resource_pass, selection)
    line = (f"{selection_key(resource_pass, engine, dataset)}: ops build "
            f"{_spell(selection)} ({selection['source']})")
    if selection["status"] == STATUS_SELECTED:
        print(f"[ops] {line}")
    else:
        warning = f"{line}: {selection['reason']}"
        print(f"  ! {warning}", file=sys.stderr)
        manifest.add_warning(warning)
    return selection


def _spell(selection: Dict[str, Any]) -> str:
    build = selection.get("build") or {}
    parts = [f"{axis}={build[axis]}" for axis in
             ("m", "ef_construction", "quantization", "storage_engine",
              "build_mode") if build.get(axis) is not None]
    return " ".join(parts) or "engine defaults"


def ops_m_values(selection: Optional[Dict[str, Any]],
                 declared: List[int]) -> List[int]:
    """The M values the ops phase builds: the chosen one, else `declared`.

    `declared` (`ops.m_values`) is what `reference` mode builds and is kept
    when the chosen build names no M -- Percona Search, whose M in a result
    file name is a label rather than a knob.
    """
    m = ((selection or {}).get("build") or {}).get("m")
    return [int(m)] if m is not None else list(declared)


def ops_build(selection: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The `harness_args(build=...)` overrides for a selection."""
    if not selection:
        return None
    build = selection.get("build") or {}
    return {"ef_construction": build.get("ef_construction"),
            "quantization": build.get("quantization")}


def disclosure(selection: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What every ops record carries as `extra.ops_build_selection`."""
    if not selection:
        return None
    return {key: selection.get(key) for key in
            ("mode", "status", "source", "target_recall", "build",
             "ann_point", "reason")}


def default_chooser(*, engine: str, dataset: str, resource_pass: str,
                    resolved: Any, paths: Dict[str, str],
                    engine_cfg: Dict[str, Any],
                    vb_root: str) -> Callable[..., Any]:
    """The production chooser: the container, on this engine's bench image."""
    image = (engine_cfg.get("image") or {}).get("bench", "")

    def chooser(*, files: List[str], target_recall: float
                ) -> Optional[Dict[str, Any]]:
        if not image or not docker_ctl.image_exists(image):
            print(f"[ops] no bench image {image!r} to read the ann results "
                  f"with", file=sys.stderr)
            return None
        return run_choice_container(
            image=image,
            ann_tree=annb_results_dir(paths, resource_pass, resolved),
            datasets_dir=paths["datasets"], vb_root=vb_root, engine=engine,
            dataset=dataset, files=files, target_recall=target_recall)
    return chooser
