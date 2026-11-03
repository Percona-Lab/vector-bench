"""The build each engine's summary figures are taken from.

THE INCIDENT. Run `full-20260825` (tuned pass, dbpedia-openai-1000k-angular)
published Percona Search's QPS-at-recall-floor figures -- 138 / 124 / 88 at
0.90 / 0.95 / 0.99 -- from the `quantization=none` sweep, while every other
Percona Search number in the same report came from the scalar-quantized index
the ops pass builds in that pass, which peaks at recall 0.892 and so has no
figure at any of those floors. pgvector's and Valkey's floor cells came from
`ef_construction=400` builds beside ops figures taken at 200 and at the server
default. `summarize` took the frontier over every build the grid swept and
nothing on the page said which one won.

The fix has three halves. Each engine declares a reference build in
`config/engines/<e>.yml` (`runtime.reference_build`, read through the
registry in `engines.py`); this module resolves it per profile and pass and
refuses one the ann grid does not contain; `cmd_run` records the resolved
builds in the manifest's `engines.<name>.presentation`, which is how the
report -- which must not import this package -- learns them.

A module of its own rather than more of `config.py`, which these functions
would have pushed past the repository's 800-line ceiling, and because they
need `ops_pass` and `ann_pass`, both of which import `config`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import engines as engines_mod
from .ann_knobs import excluded_builds, knob_problems
from .ann_pass import render_config
from .ops_pass import _quantization, ops_ef_construction

#: The ann-benchmarks argument each reference-build axis is swept under in
#: `ann_pass.render_config`'s arg groups.
GRID_ARGUMENT = {"m": "M", "ef_construction": "efConstruction",
                 "quantization": "quantization"}


def resolve_reference_build(engine: str, profile: Dict[str, Any],
                            resources: Dict[str, Any],
                            resource_pass: str) -> Dict[str, Any]:
    """The build this engine's summary figures come from, under one pass.

    An axis declared `ops` is resolved from the same function the ops pass
    builds its index with, so the two cannot drift: `ops_pass._quantization`
    for Percona Search (scalar in the tuned pass, none in the normalized one
    and in `m-sweep`), `ops_pass.ops_ef_construction` for pgvector and
    MyVector. `resources` is the pass with the profile's overrides merged in,
    as `cmd_run` resolves it.
    """
    declared = dict(engines_mod.get(engine).reference_build)
    resolved: Dict[str, Any] = {}
    for axis, value in declared.items():
        if value == "ops":
            if axis == "quantization":
                value = _quantization(profile, resources, resource_pass)
            elif axis == "ef_construction":
                value = ops_ef_construction(engine, profile)
                if value is None:
                    raise ValueError(
                        f"{engine}: runtime.reference_build declares "
                        f"ef_construction: ops, but the ops pass passes no "
                        f"--ef-construction for this engine; declare the "
                        f"value its driver or server defaults to instead")
            else:
                raise ValueError(
                    f"{engine}: runtime.reference_build declares {axis}: ops, "
                    f"which has no single ops value -- the ops pass builds "
                    f"every ops.m_values entry; name the one to report")
        resolved[axis] = value
    return resolved


def _grid_contains(groups: List[Dict[str, Any]], build: Dict[str, Any]) -> bool:
    def values(group: Dict[str, Any], axis: str) -> List[Any]:
        value = group.get(GRID_ARGUMENT[axis])
        return list(value) if isinstance(value, (list, tuple)) else [value]

    return any(
        all(GRID_ARGUMENT[axis] in group and value in values(group, axis)
            for axis, value in build.items())
        for group in groups)


def reference_build_problem(engine: str, profile: Dict[str, Any],
                            resources: Dict[str, Any],
                            resource_pass: str) -> Optional[str]:
    """Why this engine's reference build cannot be reported, or None.

    A refusal before the run, because the failure it prevents is silent and
    arrives at the end: a reference build the ann grid never measures turns
    the engine's whole headline row into "—", and one the ops pass does not
    build is the 25 Aug defect -- floor figures from one index beside load
    and concurrency figures from another, unlabelled.
    """
    try:
        build = resolve_reference_build(engine, profile, resources,
                                        resource_pass)
    except ValueError as exc:
        return str(exc)
    if not build:
        return (f"{engine}: config/engines/{engine}.yml declares no "
                f"runtime.reference_build, so the report cannot say which of "
                f"its swept builds its headline figures describe")
    spelled = ", ".join(f"{axis}={value}"
                        for axis, value in sorted(build.items()))

    try:
        body = render_config(engine, profile, resources, resource_pass)
    except ValueError as exc:
        return f"{engine}: the ann grid cannot be rendered: {exc}"
    groups = [group
              for algorithm in body["float"]["any"]
              for run_group in algorithm["run_groups"].values()
              for group in run_group["arg_groups"]]
    if not _grid_contains(groups, build):
        return (f"{engine}: its reference build ({spelled}, what the ops pass "
                f"builds) is not in the {resource_pass} ann grid this profile "
                f"renders, so no recall point would describe the index every "
                f"other section measures. Add it to the grid "
                f"(ann.* or resources.extras.*) or change the ops build.")

    ops = profile.get("ops", {}) or {}
    # Imported here: `ops_build_selection` resolves its fallback through
    # `resolve_reference_build` above, so a module-level import would cycle.
    from .ops_build_selection import MODE_FRONTIER, build_selection_mode
    try:
        mode = build_selection_mode(profile)
    except ValueError as exc:
        return f"{engine}: {exc}"
    if mode == MODE_FRONTIER and ops.get("enabled", True) is not False:
        # The ops pass builds what the ann phase's frontier picks, so the
        # `ops.m_values` check below no longer describes what it builds and
        # is not applied. The grid check above still is: the reference build
        # is the declared fallback when the ann phase leaves nothing to
        # choose from, and a fallback with no recall point is the same blank
        # headline row. What remains impossible is a frontier with no ann
        # phase to take it from.
        if (profile.get("ann") or {}).get("enabled", True) is False:
            return (f"{engine}: ops.build_selection is frontier, which picks "
                    f"the ops build from the ann phase's results, and this "
                    f"profile disables the ann phase -- every unit would "
                    f"fall back to the reference build. Enable ann or set "
                    f"ops.build_selection: reference.")
        return None
    if "m" in build and ops.get("enabled", True) is not False:
        m_values = [int(m) for m in (ops.get("m_values") or [16])]
        if int(build["m"]) not in m_values:
            return (f"{engine}: its reference build ({spelled}) names "
                    f"M={build['m']}, which the ops pass does not build "
                    f"(ops.m_values {m_values})")
    return None


def ops_build_problems(engine: str, profile: Dict[str, Any]) -> List[str]:
    """Each ops build (`ops.m_values` x the ops ef_construction) the engine
    cannot take, as one sentence each; [] when every one is buildable.

    The ann grid has had this check since `ann_knobs`; the ops grid is a
    second, independent list (`ops.m_values` with `ops_ef_construction`) and
    had none, so `ops.m_values: [6]` with VillageSQL's ef_construction 200 --
    a build vsql_vector refuses -- would start the server, load the corpus
    and only then fail inside the harness. Refused before any container
    starts instead, by the same `knob_problems` / `excluded_builds` rules the
    render uses, naming the combination.

    Not applied under `ops.build_selection: frontier`, where the ops pass
    builds a build the ann render already admitted (or the reference build,
    which `reference_build_problem` holds to that grid), nor when ops is off
    or the engine has no ops ef_construction (the driver's default then, and
    `ann_knobs` bounds M alone).
    """
    ops = profile.get("ops", {}) or {}
    if ops.get("enabled", True) is False:
        return []
    from .ops_build_selection import MODE_FRONTIER, build_selection_mode
    try:
        if build_selection_mode(profile) == MODE_FRONTIER:
            return []
    except ValueError:
        return []   # reported by `reference_build_problem`
    m_values = [int(m) for m in (ops.get("m_values") or [16])]
    efc = ops_ef_construction(engine, profile)
    problems = [f"ops.m_values: {p}" for p in knob_problems(
        engine, m_values, [efc] if efc is not None else [])]
    if efc is not None:
        problems += [f"{engine}: the ops pass would build M={m} at "
                     f"ef_construction={e} (ops.m_values x the ops "
                     f"ef_construction), which the engine cannot build: "
                     f"{reason}"
                     for m, e, reason in excluded_builds(engine, m_values,
                                                         [efc])]
    return problems
