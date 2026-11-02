"""Resolve every (pass, engine) once, record it, and refuse a changed resume.

Split out of `cli.cmd_run` so the objects that start the containers and the
objects the manifest records are provably the same ones: `cmd_run` takes
`resources`, `engine_cfg` and `resolved` out of the `EnginePlan` this returns
and hands them to `_run_unit` unchanged, and `engines_resolved` was written
from them here. See `config.engine_resolution` for the incident.

Everything is resolved and checked BEFORE the first unit runs. The loop in
`cmd_run` is pass -> engine -> dataset -> phase, so a check made inside it
would refuse at engine five of a resume after engines one to four had already
been re-measured under the run's original budget -- an hour, or a day, of
work that the refusal then strands in a directory it has declared unusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .config import (ResolvedResources, engine_resolution, load_engine,
                     load_resources, merge_resource_overrides,
                     ops_wait_instruments, ops_wait_sampler,
                     profile_max_clients,
                     profile_window_clients, resolve_resources)
from .manifest import Manifest, ResolutionChanged


@dataclass
class EnginePlan:
    #: Per pass: the merged resource block `_run_unit` is handed.
    resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Per engine: its loaded `config/engines/<e>.yml`.
    engine_cfg: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Per (pass, engine): the resolution that starts its containers.
    resolved: Dict[Tuple[str, str], ResolvedResources] = field(
        default_factory=dict)
    #: One message per (pass, engine) whose recorded resolution differs from
    #: this run's while units of it are still to be measured.
    refusals: List[str] = field(default_factory=list)


def _pending(resource_pass: str, engine: str, datasets: Sequence[str],
             phases: Sequence[str], checkpoints: set) -> bool:
    return any(f"{resource_pass}/{engine}/{dataset}/{phase}" not in checkpoints
               for dataset in datasets for phase in phases)


def _refusal(exc: ResolutionChanged) -> str:
    moved = exc.differences()
    return (
        f"{exc.engine} / {exc.resource_pass}: this run directory measured "
        f"{exc.engine} under a different resolution than this invocation "
        f"derives, and units of it are still to be measured. Resuming would "
        f"put two budgets under one manifest entry.\n"
        + "\n".join(f"    {line}" for line in moved[:20])
        + (f"\n    ...and {len(moved) - 20} more" if len(moved) > 20 else "")
        + f"\n  recorded: {json.dumps(exc.recorded.get('resolved'), sort_keys=True)}"
        + f"\n  now:      {json.dumps(exc.now.get('resolved'), sort_keys=True)}"
    )


def resolve_engines(manifest: Manifest, passes: Sequence[str],
                    engines: Sequence[str], profile: Dict[str, Any],
                    info: Any, expected_corpus: int, checkpoints: set,
                    datasets: Sequence[str],
                    phases: Sequence[str]) -> EnginePlan:
    """Resolve, record first-writer-wins, and collect refusals.

    **A differing resolution is refused only when something is left to
    measure.** A resume whose every unit of that (pass, engine) is already
    checkpointed measures nothing under the new resolution, so nothing can be
    mislabelled -- and that resume is the recovery `_accumulate_passes`
    exists for (record a pass an archived manifest lacks, measure nothing).
    It is told, not refused. `checkpoints` is empty without `--resume`, so a
    plain re-run into a used directory counts every unit as pending.
    """
    plan = EnginePlan()
    # Both validated here, before anything is recorded or measured, rather
    # than when the first ops unit builds its harness argv hours later.
    wait_instruments = ops_wait_instruments(profile)
    wait_sampler = ops_wait_sampler(profile)
    profile_window_clients(profile)
    for resource_pass in passes:
        resources = merge_resource_overrides(
            load_resources(resource_pass), profile)
        if expected_corpus:
            resources.setdefault("memory", {})["expected_corpus_bytes"] = \
                expected_corpus
        plan.resources[resource_pass] = resources
        for engine in engines:
            engine_cfg = plan.engine_cfg.get(engine) or load_engine(engine)
            plan.engine_cfg[engine] = engine_cfg
            # The profile's largest client count, which `client_cpus: auto`
            # is sized from. Derived here rather than passed in so every
            # caller of this plan gets it; `cmd_run` has already refused a
            # topology that cannot supply it, before anything was recorded.
            resolved = resolve_resources(
                resources, engine, info,
                max_clients=profile_max_clients(profile))
            plan.resolved[(resource_pass, engine)] = resolved
            entry = engine_resolution(engine_cfg, resource_pass, resolved,
                                      wait_instruments=wait_instruments,
                                      wait_sampler=wait_sampler)
            try:
                outcome = manifest.record_engine_resolution(
                    engine, resource_pass, entry)
            except ResolutionChanged as exc:
                if _pending(resource_pass, engine, datasets, phases,
                            checkpoints):
                    plan.refusals.append(_refusal(exc))
                else:
                    print(f"[manifest] {engine} / {resource_pass} resolves "
                          f"differently now than when it was measured; every "
                          f"unit of it is complete, so nothing is re-measured "
                          f"and the recorded resolution is kept")
                continue
            if outcome == "archived":
                print(f"[manifest] {engine} / {resource_pass} was measured "
                      f"before per-engine resolutions were recorded; its "
                      f"budget is left unrecorded rather than re-stamped "
                      f"with this invocation's")
    return plan
