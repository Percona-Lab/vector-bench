"""A build-selection overlay on a profile: `run --profile <p> --overlay <f>`.

WHAT IT IS FOR. Stage 2 of the two-stage frontier (see
`config/profiles/frontier-explore.yml`): `select-builds` reads a stage-1 run
and writes the builds worth measuring at full size, and `run --profile
tuned-complete` measures them. The selection has to reach the run without
anyone editing `tuned-complete.yml` by hand -- a hand-copied list is how a
grid and its provenance drift apart -- and without a second, generated copy of
the profile beside the original that the next change to tuned-complete would
leave behind.

WHY NOT A NEW MECHANISM. A profile can already change the resource pass it
runs under: its `resources:` block deep-merges over `config/resources/
<pass>.yml` (`config.merge_resource_overrides`), and every per-engine build
axis the tuned pass sweeps lives in `resources.extras` --
`m_values_by_engine`, `<engine>_ef_construction`, `mongodb_quantization`.
So the overlay is a profile fragment with exactly that shape, merged onto the
profile by the same deep-merge rule, and resolution downstream is unchanged:
`render_config`, the reference-build check, the ops pass and the estimate all
read the merged profile the way they read any other.

WHAT IT MAY TOUCH, AND WHY ONLY THAT. The whitelist below is the build-axis
keys of `resources.extras` and nothing else. An overlay that could also set
`datasets`, `runs`, `ops` or a memory fraction would be a second profile
under a name that says it is the first one, and the manifest would record
`tuned-complete` above a run that was not tuned-complete. Anything else is
refused before the run, naming the key.

Two more refusals, both for silent failures:

* `overlay_of` must name the profile it is applied to. A selection made for
  tuned-complete's grid applied to `main` would merge cleanly and measure
  something nobody chose.
* The extras it sets are read in the tuned pass only (`render_config` ignores
  them in the normalized pass, which pins every engine-specific axis by
  design), so `cmd_run` refuses an overlay on any run that includes the
  normalized pass -- it would apply to half the run and silently not to the
  other half.

The provenance block (`build_selection`) is split off and returned separately:
`cmd_run` records it in the manifest through `Manifest.set_build_selection`,
and it stays out of `config.profile` so the stage-1 points it carries are
written once rather than once per (pass, engine).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

import yaml

from . import ann_knobs
from . import engines as engines_mod

#: The `resources.extras` keys an overlay may set: every per-engine build axis
#: `ann_pass.render_config` reads in the tuned pass, and the exact build list
#: (`ann_knobs.restrict_to_builds`) that keeps the lists from multiplying.
OVERLAY_EXTRAS = frozenset({
    "ann_builds_by_engine",
    "m_values_by_engine",
    "pgvector_ef_construction",
    "valkey_ef_construction",
    "percona_ef_construction",
    "myvector_ef_construction",
    "villagesql_ef_construction",
    "mongodb_quantization",
})

OVERLAY_TOP_LEVEL = frozenset({"overlay_of", "resources", "build_selection"})


def deep_merge(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
    """`config.merge_resource_overrides`'s rule: dict into dict, anything else
    replaces. A new dict; neither input is modified."""
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_overlay(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"overlay not found: {path}")
    with open(path) as fh:
        body = yaml.safe_load(fh) or {}
    if not isinstance(body, dict):
        raise ValueError(f"{path}: an overlay is a mapping, got "
                         f"{type(body).__name__}")
    return body


def overlay_problems(overlay: Dict[str, Any], profile_name: str) -> list:
    """Why this overlay cannot be applied to this profile, one line each."""
    problems = []
    unknown = sorted(set(overlay) - OVERLAY_TOP_LEVEL)
    if unknown:
        problems.append(
            f"keys an overlay may not set: {unknown}. It may set only "
            f"resources.extras build axes ({sorted(OVERLAY_EXTRAS)}); "
            f"anything else would make the run a different profile under "
            f"{profile_name}'s name.")
    target = overlay.get("overlay_of")
    if target != profile_name:
        problems.append(
            f"the overlay was written for profile {target!r} and is being "
            f"applied to {profile_name!r}. Regenerate it with "
            f"`select-builds --profile {profile_name}`.")
    resources = overlay.get("resources") or {}
    if not isinstance(resources, dict) or set(resources) - {"extras"}:
        problems.append(
            "resources may contain only `extras`; got "
            f"{sorted(resources) if isinstance(resources, dict) else resources!r}")
    else:
        extras = resources.get("extras") or {}
        bad = sorted(set(extras) - OVERLAY_EXTRAS) if isinstance(
            extras, dict) else ["<not a mapping>"]
        if bad:
            problems.append(
                f"resources.extras keys an overlay may not set: {bad}; "
                f"allowed: {sorted(OVERLAY_EXTRAS)}")
        else:
            problems += _value_problems(extras)
            problems += _digest_problems(overlay, extras)
    return problems


def _is_list_of(value: Any, kind: type) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(v, kind) and not isinstance(v, bool)
                    for v in value))


def _engine_of(name: str) -> str:
    return name.rsplit("_ef_construction", 1)[0].rsplit("_quantization", 1)[0]


def _value_problems(extras: Dict[str, Any]) -> list:
    """Shapes, engines and ranges, checked when the overlay is loaded.

    Shapes so a scalar where a list belongs is a refusal and not a TypeError
    out of `render_config` after the run banner; engine names so a typo is
    not a selection nothing reads; ranges and enumerations
    (`ann_knobs.KNOB_BOUNDS`, `ann_knobs.QUANTIZATIONS`) so a value the
    engine cannot take is refused here, naming the overlay, rather than later
    as a render error that names only the engine.
    """
    known = set(engines_mod.known_engines())
    problems = []
    for name, value in extras.items():
        if name == "m_values_by_engine":
            if not isinstance(value, dict) or not all(
                    _is_list_of(v, int) for v in value.values()):
                problems.append(f"{name} must map engines to non-empty "
                                f"lists of integers; got {value!r}")
                continue
            for engine, ms in value.items():
                problems += ([f"{name}: unknown engine {engine!r}"]
                             if engine not in known
                             else ann_knobs.knob_problems(engine, ms))
        elif name == "ann_builds_by_engine":
            problems += _builds_problems(value, known)
        elif name.endswith("_quantization"):
            if not _is_list_of(value, str) or set(value) - set(
                    ann_knobs.QUANTIZATIONS):
                problems.append(f"{name} must be a non-empty list drawn from "
                                f"{list(ann_knobs.QUANTIZATIONS)}; got "
                                f"{value!r}")
        elif not _is_list_of(value, int):
            problems.append(f"{name} must be a non-empty list of integers; "
                            f"got {value!r}")
        else:
            problems += ann_knobs.knob_problems(_engine_of(name), (), value)
    return problems


def _builds_problems(value: Any, known: set) -> list:
    if not isinstance(value, dict):
        return [f"ann_builds_by_engine must map engines to lists of builds; "
                f"got {value!r}"]
    problems = []
    for engine, builds in value.items():
        if engine not in known:
            problems.append(f"ann_builds_by_engine: unknown engine {engine!r}")
            continue
        if not isinstance(builds, list) or not builds or not all(
                isinstance(b, dict) and b
                and set(b) <= set(ann_knobs.BUILD_AXES) for b in builds):
            problems.append(f"ann_builds_by_engine.{engine} must be a "
                            f"non-empty list of builds naming only "
                            f"{list(ann_knobs.BUILD_AXES)}; got {builds!r}")
    return problems


def overlay_digest(extras: Dict[str, Any], selection: Dict[str, Any]) -> str:
    """sha256 over the extras AND the whole provenance block (bar the digest
    itself and the manifest's `recorded_at`), as canonical JSON.

    `select-builds` stores it in `build_selection.overlay_sha256`. The
    manifest's resume check compares the selection block, so without this a
    hand edit to the lists under an unchanged block resumed as 'unchanged'
    with two grids under one manifest -- and an edit to the kept/pruned lists
    or the transfer-check points would record provenance that is not what
    `select-builds` derived.
    """
    trimmed = {k: v for k, v in (selection or {}).items()
               if k not in ("overlay_sha256", "recorded_at")}
    blob = json.dumps({"extras": extras, "build_selection": trimmed},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _digest_problems(overlay: Dict[str, Any], extras: Dict[str, Any]) -> list:
    """Required, not optional: an overlay without its provenance block would
    run and record nothing about where its grid came from."""
    selection = overlay.get("build_selection")
    if not isinstance(selection, dict) or not selection:
        return ["the overlay has no build_selection block, so the run would "
                "record no provenance for its grid. Overlays are written by "
                "`select-builds`; regenerate it."]
    if selection.get("overlay_sha256") != overlay_digest(extras, selection):
        return ["the overlay does not match the digest select-builds recorded "
                "in build_selection.overlay_sha256: its grid or its "
                "provenance was edited. Regenerate the overlay."]
    return []


def apply_overlay(profile: Dict[str, Any], overlay: Dict[str, Any]
                  ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(merged profile, provenance block). Raises ValueError, listing every
    problem, when the overlay cannot be applied as written."""
    problems = overlay_problems(overlay, str(profile.get("name")))
    if problems:
        raise ValueError("; ".join(problems))
    merged = deep_merge(profile, {"resources": overlay.get("resources") or {}})
    return merged, dict(overlay["build_selection"])
