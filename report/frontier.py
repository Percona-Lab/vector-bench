"""The headline frontier: every build swept, each figure naming its build.

WHY THE HEADLINE MOVED. PR #77 restricted §4's QPS-at-recall-floor figures to
each engine's REFERENCE build -- the index the ops pass builds -- because the
published `full-20260825` table took Percona Search's 138 / 124 / 88 from the
unquantized index while every other section measured the scalar one. A
reviewer of that report then made the opposite point, and it is the right
one for the headline: "M=16" does not mean the same work in MHNSW, VIDX,
pgvector and hnswlib, so comparing engines at identically named parameters
compares parameter names. What a user gets is the QPS an engine reaches at a
recall, over every configuration they could choose -- the frontier over the
whole swept build space (M x ef_construction x quantization x storage engine
x ef_search). And a "frontier" drawn from one build with only ef_search varied
is not one: it is a recall-throughput curve at that build.

So the headline is the whole-sweep frontier again, but the 25 Aug lesson is
kept rather than undone: every figure names the build it came from, a figure
from a build other than the one the ops pass measured is marked (concurrency,
churn and filtered figures describe the ops build, not this one), the swept
space is disclosed per engine, and the reference-build figures stay as the
secondary table. This module computes the parts of that which are not a
floor figure: what was swept, what each frontier build cost to build, and the
smallest build that reaches each floor.

`report/` must not import `orchestrator/`; nothing here does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from report import builds

#: The build axes disclosed per engine, as (record field, printed name). The
#: named columns, because that is what a reader can act on; `build_signature`
#: (which also reads the result file name) still decides how many builds there
#: were, so an axis missing from this list cannot merge two builds into one.
SWEPT_AXES = (("m", "M"), ("ef_construction", "efC"),
              ("quantization", "quantization"),
              ("storage_engine", "storage"), ("build_mode", "build mode"))


def axis_value(record: Dict[str, Any], axis: str) -> Any:
    """One build axis of a record, reading archived spellings too.

    `quantization` goes through `builds.build_params`, which recovers it from
    `extra.parameters` and the file name on a records.jsonl merged before the
    loader copied the attribute -- the 25 Aug run's is one.
    """
    if axis in builds.REFERENCE_AXES:
        return builds.build_params(record).get(axis)
    return record.get(axis)


def swept_space(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which build axes actually varied, with their values, for one engine.

    `builds` is the number of distinct build signatures, and it alone decides
    whether the engine's line may be called a frontier: an archived record
    with no named columns still has a result file name, and two files are two
    builds whatever the columns say.
    """
    axes: Dict[str, List[Any]] = {}
    for field, _name in SWEPT_AXES:
        values = {axis_value(p, field) for p in points} - {None}
        if values:
            axes[field] = sorted(values, key=str)
    if "quantization" in axes:
        # mongot exposes no graph degree; the M in its file name is a label
        # `render_config` carries, not a setting the index read -- the rule
        # `builds.build_label` applies, so "fixed: M=16" would be a claim
        # about a knob that does not exist.
        axes.pop("m", None)
    efs =sorted({p.get("ef_search") for p in points} - {None})
    signatures = {builds.build_signature(p) for p in points}
    return {"axes": axes, "ef_search": efs, "builds": len(signatures),
            "single_build": len(signatures) == 1,
            "build_label": (builds.build_label(points[0])
                            if len(signatures) == 1 and points else None)}


def spell_swept(space: Dict[str, Any]) -> str:
    """The swept space as the page prints it: varied axes first.

    An axis with one value is printed as fixed, not omitted: "M=16 fixed"
    is the sentence the reviewer's second point needed to be able to read.
    """
    varied, fixed = [], []
    names = dict(SWEPT_AXES)
    for field, values in (space.get("axes") or {}).items():
        name = names.get(field, field)
        if len(values) > 1:
            varied.append(f"{name} ∈ {{{', '.join(str(v) for v in values)}}}")
        else:
            fixed.append(f"{name}={values[0]}")
    efs = space.get("ef_search") or []
    if efs:
        ef = (f"ef_search {efs[0]}–{efs[-1]} ({len(efs)} values)"
              if len(efs) > 1 else f"ef_search={efs[0]}")
        (varied if len(efs) > 1 else fixed).append(ef)
    text = "; ".join(varied) or "nothing varied"
    if fixed:
        text += f" · fixed: {', '.join(fixed)}"
    return text


def _positive(value: Any) -> Optional[float]:
    """A size or a time, or None when it is absent, zero or not a number.

    Zero is "not measured", not "free": `vb_mysql._measure_index_bytes`
    returns 0 and prints a WARNING when its file glob matched nothing, and
    every module initialises `_index_bytes` to 0 before it measures.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and number == number else None


def _ops_index_bytes(build_point: Dict[str, Any],
                     ops_builds: List[Dict[str, Any]]) -> Optional[float]:
    """The ops pass's index size for the same build, when exactly one matches.

    Only a single match is used: two matching `index_build` records (InnoDB
    and MyISAM at one M, or two build modes) are two indexes, and choosing
    between them would print one index's size against another's curve.
    """
    params = builds.build_params(build_point)
    wanted = {axis: value for axis, value in params.items()
              if value is not None}
    storage = build_point.get("storage_engine")
    matched = [r for r in ops_builds
               if builds.matches(r, wanted)
               and (storage is None or r.get("storage_engine") in (None, storage))
               and _positive(r.get("index_bytes")) is not None]
    if len(matched) != 1:
        return None
    return _positive(matched[0].get("index_bytes"))


def build_cost(build_points: List[Dict[str, Any]],
               ops_builds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build time and index size for one build, and where each came from.

    WHAT THE ANN FIGURES ARE. Every point of one build is one ann-benchmarks
    result file from one `fit()`, so they share these values; the first
    positive one is taken.

    * `build_wall_s` is ann-benchmarks' `build_time`: the wall clock of
      `fit()`, which loads the corpus AND builds the index -- for MHNSW and
      VIDX the two are one operation. It is a time-to-queryable-index, and
      the page says so rather than calling it an index build time.
    * `index_bytes` is NOT ann-benchmarks' `index_size`. That attribute is
      the change in `get_memory_usage()` across `fit()`, and upstream's
      default for that method is the CLIENT process's RSS -- for a
      client/server engine, the size of the Python benchmark, not the index.
      Every overlay module here overrides it, but the loader does not rely
      on that: it reads the module's own `index_bytes` attribute, which each
      module measures on the server side (`vb_mysql`: the index's files under
      the datadir; pgvector: `pg_relation_size`; Percona Search: mongot's
      index; Valkey: resident `used_memory` growth, no file; MyVector: the
      hnswlib index files). Its `index_size` fallback is a `setdefault` on a
      key the record already holds, so a file without `index_bytes` loads as
      None, never as the client figure.

    Where the ann file has no size, the ops pass's `index_build` record for
    the same build is used and labelled as such; otherwise "not measured".
    """
    build_s = next((v for v in (_positive(p.get("build_wall_s"))
                                for p in build_points) if v is not None), None)
    size = next((v for v in (_positive(p.get("index_bytes"))
                             for p in build_points) if v is not None), None)
    source = "ann" if size is not None else None
    if size is None and build_points:
        size = _ops_index_bytes(build_points[0], ops_builds)
        source = "ops index_build" if size is not None else None
    resident = any(p.get("storage_engine") == "memory" for p in build_points)
    return {"build_wall_s": build_s, "index_bytes": size,
            "index_bytes_source": source, "index_resident": resident}


def _group_by_build(points: List[Dict[str, Any]]
                    ) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(builds.build_signature(point), []).append(point)
    return grouped


def frontier_builds(points: List[Dict[str, Any]],
                    frontier: List[Dict[str, Any]],
                    ops_builds: List[Dict[str, Any]],
                    reference: Optional[Dict[str, Any]]
                    ) -> List[Dict[str, Any]]:
    """Every build that contributes a point to the frontier, with its cost.

    F14b: a frontier point bought by a graph twice the size is a different
    offer from one that was not, and the frontier alone hides which is which.
    """
    by_build = _group_by_build(points)
    on_frontier = _group_by_build(frontier)
    rows = []
    for signature, chosen in on_frontier.items():
        all_points = by_build.get(signature, chosen)
        recalls = [p.get("recall_at_k") or 0 for p in chosen]
        rows.append({
            "build": signature,
            "build_label": builds.build_label(chosen[0]),
            "storage_engine": chosen[0].get("storage_engine"),
            "frontier_points": len(chosen),
            "recall_low": min(recalls), "recall_high": max(recalls),
            "is_ops_build": is_ops_build(chosen[0], reference),
            **build_cost(all_points, ops_builds),
        })
    rows.sort(key=lambda row: row["recall_low"])
    return rows


def is_ops_build(point: Optional[Dict[str, Any]],
                 reference: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Whether `point` was measured on the build the ops pass built.

    None when the ops build is not known for this engine (no declaration, no
    usable ops record, or an ambiguous one): "different" would be a guess.
    """
    if point is None or reference is None:
        return None
    return builds.matches(point, reference)


#: `smallest_..._ranked_by` when no size was measured and two or more builds
#: reaching the floor share the lowest M and efC.
TIED = "tied, no size measured"


def _smallness_key(row: Dict[str, Any], by_size: bool) -> Tuple[Any, ...]:
    """Lower is smaller. By index size when every candidate has one;
    otherwise by graph degree, then build effort, which is what "fewest edges"
    means on an HNSW graph when nothing measured its bytes."""
    if by_size:
        return (row["index_bytes"], -row["qps"])
    inf = float("inf")
    m = row["m"] if row["m"] is not None else inf
    efc = row["ef_construction"] if row["ef_construction"] is not None else inf
    return (m, efc, -row["qps"])


def smallest_figures(points: List[Dict[str, Any]],
                     ops_builds: List[Dict[str, Any]],
                     reference: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per floor, the smallest build that reaches it, and its best QPS there.

    The answer to "what does this recall cost" that does not favour the
    biggest graph: the fastest build at a floor is very often the largest one
    swept, and a user sizing a machine wants the cheapest index that still
    gets there. Ranked by index size only when every build that reaches the
    floor has one -- comparing a measured size against "not measured" would
    rank by the gap in the instrument -- and otherwise by M, then
    ef_construction, saying which rule was used.
    """
    out: Dict[str, Any] = {}
    grouped = _group_by_build(points)
    for floor in builds.FLOORS:
        key = f"smallest_qps_at_recall_{int(floor * 100)}"
        candidates = []
        for signature, members in grouped.items():
            reaching = [p for p in members
                        if (p.get("recall_at_k") or 0) >= floor
                        and p.get("qps") is not None]
            if not reaching:
                continue
            best = max(reaching, key=lambda p: p["qps"])
            params = builds.build_params(best)
            candidates.append({
                "point": best, "qps": best["qps"],
                "m": params["m"], "ef_construction": params["ef_construction"],
                "index_bytes": build_cost(members, ops_builds)["index_bytes"],
            })
        if not candidates:
            for suffix in ("", "_recall", "_build", "_build_label",
                           "_index_bytes", "_ranked_by", "_is_ops_build"):
                out[key + suffix] = None
            continue
        by_size = all(c["index_bytes"] is not None for c in candidates)
        chosen = min(candidates, key=lambda c: _smallness_key(c, by_size))
        point = chosen["point"]
        ranked_by = "index size" if by_size else "M, then efC"
        if not by_size and sum(
                1 for c in candidates
                if _smallness_key(c, False)[:2]
                == _smallness_key(chosen, False)[:2]) > 1:
            # Two builds at one M and efC (Percona Search's quantizations,
            # or InnoDB and MyISAM) with no size to separate them: "smallest"
            # cannot be decided, and the fastest of the tie is shown as such.
            ranked_by = TIED
        out[key] = chosen["qps"]
        out[f"{key}_recall"] = point.get("recall_at_k")
        out[f"{key}_build"] = builds.build_signature(point)
        out[f"{key}_build_label"] = builds.build_label(point)
        out[f"{key}_index_bytes"] = chosen["index_bytes"]
        out[f"{key}_ranked_by"] = ranked_by
        out[f"{key}_is_ops_build"] = is_ops_build(point, reference)
    return out


def ops_build_flags(entry: Dict[str, Any], frontier: List[Dict[str, Any]],
                    reference: Optional[Dict[str, Any]]) -> None:
    """Set `qps_at_recall_N_is_ops_build` beside each headline figure.

    Looked up by signature on the frontier rather than recomputed from the
    label, because two storage engines at one M share a label.
    """
    by_signature = {builds.build_signature(p): p for p in frontier}
    for floor in builds.FLOORS:
        key = f"qps_at_recall_{int(floor * 100)}"
        point = by_signature.get(entry.get(f"{key}_build"))
        entry[f"{key}_is_ops_build"] = is_ops_build(point, reference)
