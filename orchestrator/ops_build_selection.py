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
  file from another grid cannot be chosen), keeps the points of the ops
  pass's primary storage engine and build mode, and picks the
  (M, ef_construction, quantization) whose frontier point at the ops
  calibration target is fastest. The ops pass then builds exactly that, for
  every declared storage engine and build mode, and the manifest
  presentation names it, with its source, per dataset.

NEVER SILENTLY. Three outcomes, each recorded on the manifest and on every
ops record (`extra.ops_build_selection.status`):

* `selected` -- a build reached the target; source `frontier@<target>`.
* `below_target` -- none did; the build with the highest recall is used, and
  a warning says so. The calibration that follows will report it missed.
* `fallback_reference` -- the ann phase did not run in this run directory,
  failed, produced no usable point, or the selection container could not
  read it (image missing, non-zero exit, unparsable output, timeout, a
  docker CLI that would not start); the declared reference build is
  measured, as `reference` mode would, and a warning says why.

RESUME, AND WHEN A STORED CHOICE IS NOT REUSED. The choice is persisted in
the manifest under `ops_build_selection["<pass>/<engine>/<dataset>"]` the
moment it is made, together with the identity of the ann attempt it was
computed from (attempt number, start time, and a digest of its file list).
A `selected` or `below_target` choice is reused only while that attempt is
still the latest completed ann attempt of the unit: a `--resume` after an
ann re-run (`--force`, a changed grid, a retried failure) chooses again from
the new attempt, because the old choice may name a build the new grid does
not even contain. A `fallback_reference` is never reused once a completed
ann attempt exists -- the fallback was for its absence. Any choice, stored
or fresh, that names a build outside the grid this profile renders now is
refused rather than measured. And a re-selection that changes the build
after ops units of this (engine, dataset, pass) were already measured on the
old one is refused too: those units' records stay in the run directory, and
measuring the rest on another index would put two builds under one name.

A module of its own because `cli.py` and `ops_pass.py` are far past the
repository's line ceiling and a sibling change is editing `cli.py` too; the
call sites there are a few lines each.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

from . import docker_ctl
from . import engines as engines_mod
from .ann_pass import annb_results_dir, render_config
from .manifest import utcnow
from .ops_pass import ops_target_recall
from .reference_build import _grid_contains, resolve_reference_build

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
#: How long the selection container may take. It reads one engine's result
#: files for one corpus; minutes at the outside. A stuck container must not
#: hold the run, and the fallback it produces is visible.
CHOICE_TIMEOUT_S = 1800


class SelectionRefused(RuntimeError):
    """A frontier choice that cannot be measured honestly. Fails the unit."""


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


def latest_ann_attempt(manifest_data: Dict[str, Any], engine: str,
                       dataset: str, resource_pass: str
                       ) -> Optional[Dict[str, Any]]:
    """The latest completed ann attempt of this unit that recorded files.

    None when there is none in this run directory -- the phase was not asked
    for (`--phases ops`), failed, or ran under code that recorded no files.
    The latest, because it is the one whose grid the ops pass follows; its
    `skipped` list already names the earlier attempts' files it stood on.
    """
    attempts = [p for p in manifest_data.get("phases") or []
                if isinstance(p, dict) and p.get("phase") == "ann"
                and p.get("engine") == engine and p.get("dataset") == dataset
                and p.get("resource_pass") == resource_pass
                and p.get("status") == "completed"
                and isinstance(p.get("files"), dict)]
    return attempts[-1] if attempts else None


def _attempt_files(attempt: Optional[Dict[str, Any]]) -> List[str]:
    files = (attempt or {}).get("files") or {}
    paths = [entry.get("path") for key in ("written", "skipped")
             for entry in files.get(key) or [] if isinstance(entry, dict)]
    return sorted({p for p in paths if p})


def ann_phase_files(manifest_data: Dict[str, Any], engine: str, dataset: str,
                    resource_pass: str) -> Optional[List[str]]:
    """The files the latest completed ann attempt wrote or reused, or None."""
    attempt = latest_ann_attempt(manifest_data, engine, dataset,
                                 resource_pass)
    return _attempt_files(attempt) if attempt is not None else None


def attempt_identity(attempt: Optional[Dict[str, Any]]
                     ) -> Optional[Dict[str, Any]]:
    """What a stored choice records about the ann attempt it came from.

    The attempt number and start time name the attempt; the digest of its
    file list catches a hand-edited or rewritten phase entry that kept both.
    """
    if attempt is None:
        return None
    files = _attempt_files(attempt)
    digest = hashlib.sha256(json.dumps(files).encode()).hexdigest()
    return {"attempt": attempt.get("attempt"),
            "started_at": attempt.get("started_at"),
            "files_digest": digest}


def _failure(message: str) -> Dict[str, Any]:
    print(f"[ops] build selection could not read the ann results: {message}",
          file=sys.stderr)
    return {"error": message}


def run_choice_container(*, image: str, ann_tree: str, datasets_dir: str,
                         vb_root: str, engine: str, dataset: str,
                         files: List[str], target_recall: float,
                         storage_engine: Optional[str] = None,
                         build_mode: Optional[str] = None,
                         timeout_s: int = CHOICE_TIMEOUT_S
                         ) -> Dict[str, Any]:
    """Run `report/ops_build_choice.py` in a bench image.

    Returns its payload, or `{"error": why}` -- never raises for a failure of
    the container, the docker CLI or the output, because every one of those
    must end in a visible `fallback_reference` and not in a traceback that
    fails the ops unit with nothing recorded. A container that outlives the
    timeout (the CLI is killed, the container is not) is removed here.

    In a container for the reason `cli.generate_report` is: the choice needs
    h5py and numpy to read the result files and score recall, and the host
    has neither by design.
    """
    output: List[str] = []
    command = ["/vb/report/ops_build_choice.py",
               "--annb-dir", "/annb", "--datasets-dir", "/datasets",
               "--engine", engine, "--dataset", dataset,
               "--target-recall", str(target_recall),
               "--files", json.dumps(files)]
    if storage_engine:
        command += ["--storage-engine", storage_engine]
    if build_mode:
        command += ["--build-mode", build_mode]
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
        command=command,
        detach=False,
    )
    try:
        rc = docker_ctl.run_foreground(spec, timeout=timeout_s, stream=False,
                                       sink=output)
    except (docker_ctl.DockerError, subprocess.TimeoutExpired,
            OSError) as exc:
        _remove_quietly(spec.name)
        kind = ("timed out after " f"{timeout_s} s"
                if isinstance(exc, subprocess.TimeoutExpired)
                else f"{type(exc).__name__}: {exc}")
        return _failure(f"the selection container {kind}")
    if rc != 0:
        _remove_quietly(spec.name)
        return _failure(f"the selection container exited {rc}")
    for line in reversed(output):
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER):])
            except ValueError as exc:
                return _failure(f"its result line is not JSON ({exc})")
            if not isinstance(payload, dict):
                return _failure("its result line is not a JSON object")
            return payload
    return _failure("the selection container printed no result line")


def _remove_quietly(name: str) -> None:
    try:
        docker_ctl.remove(name)
    except (docker_ctl.DockerError, subprocess.TimeoutExpired, OSError):
        print(f"[ops] could not remove container {name}; "
              f"`docker rm -f {name}` by hand", file=sys.stderr)


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


def grid_problem(engine: str, profile: Dict[str, Any],
                 resources: Dict[str, Any], resource_pass: str,
                 build: Dict[str, Any]) -> Optional[str]:
    """Why `build` is not in the ann grid this profile renders now, or None.

    Checked on the engine's reference axes, as `reference_build_problem` is,
    because those are the axes the report matches on. A stored choice from an
    attempt the grid has since moved away from is the case: measuring it would
    publish ops figures for a build with no recall point in the report.
    """
    wanted = {axis: value
              for axis, value in _presentation_build(engine, build).items()
              if value is not None}
    if not wanted:
        return None
    try:
        body = render_config(engine, profile, resources, resource_pass)
    except ValueError as exc:
        return f"the ann grid cannot be rendered: {exc}"
    groups = [group
              for algorithm in body["float"]["any"]
              for run_group in algorithm["run_groups"].values()
              for group in run_group["arg_groups"]]
    if _grid_contains(groups, wanted):
        return None
    spelled = ", ".join(f"{k}={v}" for k, v in sorted(wanted.items()))
    return (f"the chosen build ({spelled}) is not in the {resource_pass} ann "
            f"grid this profile renders now")


def _fallback(engine: str, profile: Dict[str, Any], resources: Dict[str, Any],
              resource_pass: str, target: float, reason: str
              ) -> Dict[str, Any]:
    reference = resolve_reference_build(engine, profile, resources,
                                        resource_pass)
    return {"mode": MODE_FRONTIER, "status": STATUS_FALLBACK,
            "source": STATUS_FALLBACK, "target_recall": target,
            "build": dict(reference), "reference_build": dict(reference),
            "selected_on": None, "ann_point": None, "reason": reason}


def _from_choice(engine: str, choice: Dict[str, Any], target: float
                 ) -> Dict[str, Any]:
    status = choice["status"]
    source = (f"frontier@{target:g}" if status == STATUS_SELECTED
              else f"{STATUS_BELOW_TARGET}@{target:g}")
    raw = choice.get("build") or {}
    build = {axis: raw.get(axis)
             for axis in ("m", "ef_construction", "quantization")}
    reason = None
    if status == STATUS_BELOW_TARGET:
        reason = (f"none of the {choice.get('builds_considered')} swept "
                  f"build(s) reached recall {target:g}; measuring the one "
                  f"with the highest recall "
                  f"({(choice.get('point') or {}).get('recall')})")
    return {"mode": MODE_FRONTIER, "status": status, "source": source,
            "target_recall": target, "build": build,
            "reference_build": _presentation_build(engine, build),
            "selected_on": choice.get("selected_on"),
            "ann_point": choice.get("point"),
            "signature": choice.get("signature"),
            "builds_considered": choice.get("builds_considered"),
            "builds_reaching_target": choice.get("builds_reaching_target"),
            "reason": reason}


def choose(manifest_data: Dict[str, Any], *, engine: str, dataset: str,
           resource_pass: str, profile: Dict[str, Any],
           resources: Dict[str, Any], chooser: Callable[..., Any]
           ) -> Dict[str, Any]:
    """A fresh selection for one unit.

    `chooser(files=..., target_recall=...)` returns the container's payload,
    or `{"error": why}` (or None) when it could not produce one.
    """
    target = ops_target_recall(profile)
    attempt = latest_ann_attempt(manifest_data, engine, dataset,
                                 resource_pass)
    files = _attempt_files(attempt)
    if not files:
        return {**_fallback(engine, profile, resources, resource_pass, target,
                            "no completed ann phase of this engine and "
                            "dataset in this run directory recorded any "
                            "result file"),
                "ann_attempt": attempt_identity(attempt)}
    payload = chooser(files=files, target_recall=target) or {}
    if payload.get("error"):
        # Once more before a fallback is stored. A stored fallback is
        # replaced on the next resume (a completed ann attempt exists), and
        # if ops units were measured on it in between, that replacement is
        # refused -- so one transient docker hiccup would otherwise lock the
        # unit out until someone moved its ops files aside.
        print(f"[ops] retrying the build selection once: "
              f"{payload['error']}", file=sys.stderr)
        payload = chooser(files=files, target_recall=target) or {}
    choice = payload.get("choice")
    if not choice:
        why = (f"the selection could not read the ann results: "
               f"{payload['error']}" if payload.get("error") else
               f"the ann phase's {len(files)} result file(s) held no usable "
               f"recall/QPS point on the primary storage engine and build "
               f"mode")
        selection = _fallback(engine, profile, resources, resource_pass,
                              target, why)
    else:
        selection = _from_choice(engine, choice, target)
    return {**selection, "ann_attempt": attempt_identity(attempt)}


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
        resource_pass, chosen_engine, dataset = (key.split("/", 2)
                                                 + ["", ""])[:3]
        if chosen_engine != engine or not isinstance(selection, dict):
            continue
        out.setdefault(resource_pass, {})[dataset] = {
            "build": selection.get("reference_build") or {},
            "source": selection.get("source"),
            "status": selection.get("status")}
    return {PRESENTATION_KEY: out} if out else {}


def _reusable(stored: Dict[str, Any], current: Optional[Dict[str, Any]]
              ) -> Optional[str]:
    """None if `stored` may be reused, else why it is chosen again."""
    if stored.get("status") == STATUS_FALLBACK:
        if current is not None and _attempt_files(current):
            return ("the stored choice was a fallback for a missing ann "
                    "phase, and a completed ann attempt now exists")
        return None
    if current is None:
        return ("the ann attempt the stored choice was computed from is no "
                "longer the latest completed one, and none is")
    if stored.get("ann_attempt") != attempt_identity(current):
        return ("the ann phase has run again since the stored choice was "
                "made (attempt " f"{(stored.get('ann_attempt') or {}).get('attempt')}"
                f" -> {current.get('attempt')})")
    return None


def ensure_selection(manifest: Any, *, engine: str, dataset: str,
                     resource_pass: str, profile: Dict[str, Any],
                     resources: Dict[str, Any], chooser: Callable[..., Any],
                     ops_units_done: bool = False) -> Dict[str, Any]:
    """The choice for this unit -- reused, or made, recorded and announced.

    Raises `SelectionRefused` for a choice outside the current grid and for a
    re-selection that would change the build under already-measured units.
    `ops_units_done` says whether any ops unit of this (engine, dataset,
    pass) is already checkpointed.
    """
    key = selection_key(resource_pass, engine, dataset)
    stored = stored_selection(manifest.data, resource_pass, engine, dataset)
    current = latest_ann_attempt(manifest.data, engine, dataset,
                                 resource_pass)
    if stored is not None:
        why_not = _reusable(stored, current)
        if why_not is None:
            _check_grid(engine, profile, resources, resource_pass, key,
                        stored)
            print(f"[ops] reusing the build chosen earlier in this run: "
                  f"{_spell(stored)} ({stored.get('source')})")
            return stored
        print(f"[ops] {key}: choosing the ops build again -- {why_not}")
    selection = {**choose(manifest.data, engine=engine, dataset=dataset,
                          resource_pass=resource_pass, profile=profile,
                          resources=resources, chooser=chooser),
                 "selected_at": utcnow()}
    _check_grid(engine, profile, resources, resource_pass, key, selection)
    if (stored is not None and ops_units_done
            and stored.get("build") != selection.get("build")):
        raise SelectionRefused(
            f"{key}: the ops build would change from {_spell(stored)} to "
            f"{_spell(selection)}, but ops units of this engine and dataset "
            f"were already measured on the first; their records stay in the "
            f"run directory and the rest would be measured on another index. "
            f"Start a fresh --run-id, or move this unit's ops-*.jsonl aside "
            f"and drop its ops checkpoints.")
    _record(manifest, engine, dataset, resource_pass, selection)
    line = (f"{key}: ops build {_spell(selection)} "
            f"({selection['source']})")
    if selection["status"] == STATUS_SELECTED:
        print(f"[ops] {line}")
    else:
        warning = f"{line}: {selection['reason']}"
        print(f"  ! {warning}", file=sys.stderr)
        manifest.add_warning(warning)
    return selection


def _check_grid(engine: str, profile: Dict[str, Any],
                resources: Dict[str, Any], resource_pass: str, key: str,
                selection: Dict[str, Any]) -> None:
    # A fallback's grid membership is already checked before the run, by
    # `reference_build_problem`.
    if selection.get("status") == STATUS_FALLBACK:
        return
    problem = grid_problem(engine, profile, resources, resource_pass,
                           selection.get("build") or {})
    if problem:
        raise SelectionRefused(
            f"{key}: {problem}; it was chosen from ann attempt "
            f"{(selection.get('ann_attempt') or {}).get('attempt')}. Re-run "
            f"the ann phase under this grid (--phases ann) or restore the "
            f"grid it was chosen under.")


def _spell(selection: Dict[str, Any]) -> str:
    build = selection.get("build") or {}
    parts = [f"{axis}={build[axis]}" for axis in
             ("m", "ef_construction", "quantization")
             if build.get(axis) is not None]
    text = " ".join(parts) or "engine defaults"
    on = selection.get("selected_on") or {}
    where = "/".join(str(v) for v in (on.get("storage_engine"),
                                      on.get("build_mode")) if v)
    return f"{text}, selected on {where}" if where else text


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
             "selected_on", "ann_point", "reason")}


def default_chooser(*, engine: str, dataset: str, resource_pass: str,
                    resolved: Any, paths: Dict[str, str],
                    engine_cfg: Dict[str, Any], vb_root: str,
                    storage_engine: Optional[str] = None,
                    build_mode: Optional[str] = None) -> Callable[..., Any]:
    """The production chooser: the container, on this engine's bench image.

    `storage_engine` and `build_mode` are the ops pass's primary ones (the
    heads of the lists it measures), or None for an axis the engine's ann
    grid does not sweep; the choice ranks only points measured on them.
    """
    image = (engine_cfg.get("image") or {}).get("bench", "")

    def chooser(*, files: List[str], target_recall: float) -> Dict[str, Any]:
        try:
            present = bool(image) and docker_ctl.image_exists(image)
        except (docker_ctl.DockerError, subprocess.TimeoutExpired,
                OSError) as exc:
            return _failure(f"cannot ask docker for image {image!r}: {exc}")
        if not present:
            return _failure(f"no bench image {image!r} to read them with")
        return run_choice_container(
            image=image,
            ann_tree=annb_results_dir(paths, resource_pass, resolved),
            datasets_dir=paths["datasets"], vb_root=vb_root, engine=engine,
            dataset=dataset, files=files, target_recall=target_recall,
            storage_engine=storage_engine, build_mode=build_mode)
    return chooser
