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
            # Numbers numerically ("M ∈ {6, 16}", not "{16, 6}"), words
            # alphabetically, and never a comparison between the two.
            axes[field] = sorted(values, key=lambda v: (
                isinstance(v, str), v if not isinstance(v, str) else 0,
                str(v)))
    if "quantization" in axes:
        # mongot exposes no graph degree; the M in its file name is a label
        # `render_config` carries, not a setting the index read -- the rule
        # `builds.build_label` applies, so "fixed: M=16" would be a claim
        # about a knob that does not exist.
        axes.pop("m", None)
    efs = sorted({p.get("ef_search") for p in points} - {None})
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


def swept_flags(points: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Whether storage engine and build mode were swept for this engine.

    The ops harness stamps `storage_engine: InnoDB` and `build_mode: post` on
    every unit, including pgvector, mongot and Valkey, which have neither
    knob: `full-20260825`'s mongot `index_build` record says `post`/`InnoDB`
    beside ann points that say `async`/`wiredTiger`. Comparing those axes
    matched no ann build at all. So they are compared only where the ann grid
    varied them -- MariaDB's InnoDB and MyISAM, pgvector's post and
    incremental -- which is the only place they tell two builds apart.
    """
    return {
        "storage_engine": len({p.get("storage_engine") for p in points}
                              - {None}) > 1,
        "build_mode": len({p.get("build_mode") for p in points}
                          - {None}) > 1,
    }


def _ops_record(build_point: Dict[str, Any], ops_builds: List[Dict[str, Any]],
                flags: Dict[str, bool]) -> Optional[Dict[str, Any]]:
    """The ops pass's `index_build` record for this build, if exactly one.

    Every named build axis must agree, and an axis the ops record leaves to
    the server default (Valkey's `ef_construction` on the 25 Aug run) agrees
    with nothing: the default is not recorded, and guessing it would print
    one index's cost against another's curve. Two matching records (two
    storage engines or two build modes that the ann grid did not split) are
    two indexes, and neither is chosen.
    """
    params = builds.build_params(build_point)
    matched = []
    for record in ops_builds:
        theirs = builds.build_params(record)
        if any(value is not None and theirs.get(axis) != value
               for axis, value in params.items()):
            continue
        if any(flags.get(axis) and record.get(axis) is not None
               and record.get(axis) != build_point.get(axis)
               for axis in ("storage_engine", "build_mode")):
            continue
        matched.append(record)
    return matched[0] if len(matched) == 1 else None


def _ops_time_to_queryable(record: Dict[str, Any]) -> Optional[float]:
    """`generate.time_to_queryable_index_s`, imported in the body: `generate`
    imports this module lazily from `_floor_entry`, and a module-level import
    back would be a cycle. It is the ops figure with the same scope as the
    ann `build_time` -- empty table to queryable index, load included."""
    from report.generate import time_to_queryable_index_s
    return _positive(time_to_queryable_index_s(record))


def build_cost(build_points: List[Dict[str, Any]],
               ops_builds: List[Dict[str, Any]],
               flags: Dict[str, bool]) -> Dict[str, Any]:
    """Time to a queryable index and index size for one build, and whence.

    WHAT THE ANN FIGURES ARE. Every point of one build is one ann-benchmarks
    result file from one `fit()`, so they share these values; the first
    positive one is taken.

    * `build_wall_s` on an ann record is ann-benchmarks' `build_time`: the
      wall clock of `fit()`, which loads the corpus AND builds the index --
      for MHNSW and VIDX the two are one operation. The ops fallback is
      `time_to_queryable_index_s` of the matching `index_build` record, the
      same scope (ingest + `CREATE INDEX` for a bulk build, the load alone
      for an incremental one), and never its bare `build_wall_s`, which for a
      bulk build excludes the load. `full-20260825`'s ann records carry no
      build time at all, so without the fallback all 14 rows said "not
      measured" beside ops records holding AliSQL's 24,130 s.
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

    Each value says where it came from ("ann" or "ops"), because the two are
    different measurements of the same quantity and the page labels them.
    """
    ops = _ops_record(build_points[0], ops_builds, flags) if build_points \
        else None
    seconds = next((v for v in (_positive(p.get("build_wall_s"))
                                for p in build_points) if v is not None), None)
    time_source = "ann" if seconds is not None else None
    if seconds is None and ops is not None:
        seconds = _ops_time_to_queryable(ops)
        time_source = "ops" if seconds is not None else None
    size = next((v for v in (_positive(p.get("index_bytes"))
                             for p in build_points) if v is not None), None)
    size_source = "ann" if size is not None else None
    if size is None and ops is not None:
        size = _positive(ops.get("index_bytes"))
        size_source = "ops" if size is not None else None
    resident = any(p.get("storage_engine") == "memory" for p in build_points)
    return {"time_to_queryable_s": seconds, "time_source": time_source,
            "index_bytes": size, "index_bytes_source": size_source,
            "index_resident": resident}


def _group_by_build(points: List[Dict[str, Any]]
                    ) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(builds.build_signature(point), []).append(point)
    return grouped


def usable_reference(reference: Optional[Dict[str, Any]],
                     points: List[Dict[str, Any]]
                     ) -> Optional[Dict[str, Any]]:
    """The ops build, only when it can be compared against a point.

    None when it is unknown, when it leaves an axis to an unrecorded server
    default (Valkey's `{m: 16, ef_construction: None}` -- `matches` would call
    every swept efC "different" from a value nobody knows), or when it
    matches none of the swept points. In each case "is this the ops build?"
    cannot be answered, and answering "no" put a mismatch mark on every
    Valkey floor of the 25 Aug report.
    """
    if reference is None or any(v is None for v in reference.values()):
        return None
    if not any(builds.matches(p, reference) for p in points):
        return None
    return reference


def is_ops_build(point: Optional[Dict[str, Any]],
                 reference: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Whether `point` was measured on the build the ops pass built.

    `reference` must already have been through `usable_reference`; None
    means the question cannot be answered, and is never read as "no".
    """
    if point is None or reference is None:
        return None
    return builds.matches(point, reference)


def frontier_builds(points: List[Dict[str, Any]],
                    frontier: List[Dict[str, Any]],
                    ops_builds: List[Dict[str, Any]],
                    reference: Optional[Dict[str, Any]]
                    ) -> List[Dict[str, Any]]:
    """Every build that contributes a point to the frontier, with its cost.

    A frontier point bought by a graph twice the size is a different offer
    from one that was not, and the frontier alone hides which is which.
    `same_size_as_another` flags a size another swept build of this engine
    also reports to the byte: on the 25 Aug run pgvector's three efC builds
    all read 8,110,088,192 and AliSQL's M=6 and M=16 both 4,156,555,264,
    which is more likely one reading per run than three per build.
    """
    flags = swept_flags(points)
    by_build = _group_by_build(points)
    sizes: Dict[Any, Optional[float]] = {
        signature: build_cost(members, ops_builds, flags)["index_bytes"]
        for signature, members in by_build.items()}
    rows = []
    for signature, chosen in _group_by_build(frontier).items():
        all_points = by_build.get(signature, chosen)
        recalls = [p.get("recall_at_k") or 0 for p in chosen]
        cost = build_cost(all_points, ops_builds, flags)
        rows.append({
            "build": signature,
            "build_label": builds.build_label(chosen[0]),
            "storage_engine": chosen[0].get("storage_engine"),
            "frontier_points": len(chosen),
            "recall_low": min(recalls), "recall_high": max(recalls),
            "is_ops_build": is_ops_build(chosen[0], reference),
            "same_size_as_another": cost["index_bytes"] is not None and any(
                other == cost["index_bytes"]
                for other_sig, other in sizes.items() if other_sig != signature),
            **cost,
        })
    rows.sort(key=lambda row: row["recall_low"])
    return rows


#: `smallest_..._ranked_by`: how the smallest build at a floor was chosen.
#: One build reached the floor, so nothing was ranked.
ONLY = "only build reaching the floor"
BY_SIZE = "index size"
#: Sizes were all measured, but the smallest is shared by several builds --
#: on the 25 Aug run all three pgvector efC builds read the same byte count.
#: Ranking by size there would be ranking by walk order, so M and efC decide
#: among the tied builds and the page says so.
BY_SIZE_THEN_M = "equal index sizes, then M, then efC"
BY_M = "M, then efC"
#: No size measured and two or more builds reaching the floor share the
#: lowest M and efC.
TIED = "tied, no size measured"


def _m_key(row: Dict[str, Any]) -> Tuple[float, float]:
    inf = float("inf")
    return (row["m"] if row["m"] is not None else inf,
            row["ef_construction"] if row["ef_construction"] is not None
            else inf)


def _choose_smallest(candidates: List[Dict[str, Any]]
                     ) -> Tuple[Dict[str, Any], str]:
    """The smallest candidate, and the rule that chose it.

    By index size only when every candidate has one -- comparing a measured
    size against "not measured" would rank by the gap in the instrument --
    and only as far as sizes differ; then by M and ef_construction ("fewest
    edges" on an HNSW graph); then, if still tied, the fastest of the tie,
    called a tie. Faster wins every remaining tie, so the result does not
    depend on the order the result files were walked in.
    """
    def fastest(rows):
        return min(rows, key=lambda c: (-c["qps"], str(c["signature"])))

    if len(candidates) == 1:
        return candidates[0], ONLY
    pool, rule = candidates, BY_M
    if all(c["index_bytes"] is not None for c in candidates):
        least = min(c["index_bytes"] for c in candidates)
        pool = [c for c in candidates if c["index_bytes"] == least]
        rule = BY_SIZE if len(pool) == 1 else BY_SIZE_THEN_M
        if len(pool) == 1:
            return pool[0], rule
    least_m = min(_m_key(c) for c in pool)
    tied = [c for c in pool if _m_key(c) == least_m]
    if len(tied) > 1:
        return fastest(tied), (TIED if rule == BY_M else rule)
    return tied[0], rule


def smallest_figures(points: List[Dict[str, Any]],
                     ops_builds: List[Dict[str, Any]],
                     reference: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per floor, the smallest build that reaches it, and its best QPS there.

    The answer to "what does this recall cost" that does not favour the
    biggest graph: the fastest build at a floor is very often the largest one
    swept, and a user sizing a machine wants the cheapest index that still
    gets there. `_ranked_by` says which rule decided (`_choose_smallest`).
    """
    out: Dict[str, Any] = {}
    flags = swept_flags(points)
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
                "point": best, "qps": best["qps"], "signature": signature,
                "m": params["m"], "ef_construction": params["ef_construction"],
                "index_bytes": build_cost(members, ops_builds,
                                          flags)["index_bytes"],
            })
        if not candidates:
            for suffix in ("", "_recall", "_build", "_build_label",
                           "_index_bytes", "_ranked_by", "_is_ops_build"):
                out[key + suffix] = None
            continue
        chosen, ranked_by = _choose_smallest(candidates)
        point = chosen["point"]
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
