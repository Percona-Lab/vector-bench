"""Which build a measurement came from, and which build a summary must use.

THE INCIDENT. The post "Benchmarking vector search in six database engines"
was written from run `full-20260825`. Its QPS-at-recall-floor table gave
Percona Search 138 / 124 / 88 at 0.90 / 0.95 / 0.99, taken from the
`quantization=none` sweep; every other Percona Search section of the same
report measured the scalar-quantized index the ops pass builds in the tuned
pass, which peaks at recall 0.892 and so has no figure at any of those floors.
pgvector's and Valkey's cells came from efC=400 builds beside ops figures at
200 and at the server default. `summarize` took one frontier over every build
the grid swept, and the table had no way to say which build won a cell.

So every floor figure now carries its build, and the headline is restricted to
the engine's REFERENCE build -- the one the ops pass builds, declared in
`config/engines/<e>.yml`, resolved on the host and carried here through the
manifest's `engines.<name>.presentation.reference_builds`, because `report/`
must not import `orchestrator/`. A second table keeps the best of the whole
sweep, with each cell naming its build.

`build_signature` moved here from `report/loaders.py`, which sits at its
line ceiling, and is re-exported from there for its existing callers.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

#: The axes a reference build may name, as `orchestrator/engines.py`'s
#: `REFERENCE_BUILD_AXES` declares them. Spelled again rather than imported
#: because this package cannot import the orchestrator.
REFERENCE_AXES = ("m", "ef_construction", "quantization")

#: Where the reference build came from, in the order they are consulted.
SOURCE_MANIFEST = "manifest"
SOURCE_OPS = "ops index_build record"
SOURCE_SINGLE = "only build measured"
#: Prefix of the source when a fallback matched more than one build.
SOURCE_AMBIGUOUS = "ambiguous"

FLOORS = (0.90, 0.95, 0.99)


def build_signature(record: Dict[str, Any]) -> Any:
    """Everything about a measurement except the query effort.

    Two points belong to the same curve when they differ only in ef_search.
    Deciding that from named columns needs the key to list every axis a
    profile may sweep, and it silently fails the moment an engine gains one:
    Percona Search's quantization is not a column, so its unquantized and
    scalar curves collapsed into a single series -- which made eight legitimate
    measurements look like duplicates and invented six inversions comparing
    ef_search 10 against ef_search 10.

    ann-benchmarks already solves this. It names each result file after the
    build arguments with the query argument appended:

        angular_M_16_quantization_none_400.hdf5
        angular_M_16_build_mode_post_efConstruction_200_400.hdf5

    Dropping the trailing query argument leaves the build configuration, for
    any engine, including axes this code has never heard of. The named columns
    stay as the fallback for records loaded without a source file.
    """
    source = (record.get("extra") or {}).get("source_file")
    if source:
        stem = os.path.basename(str(source))
        if stem.endswith(".hdf5"):
            stem = stem[:-len(".hdf5")]
        head, _, tail = stem.rpartition("_")
        # Only strip a trailing query argument, never part of the build name.
        # A stem that does not end in one is still a configuration identifier
        # and is kept whole: falling back to the columns here would discard
        # the very information the filename was consulted for.
        build = head if (head and tail.isdigit()) else stem
        return ("file", record.get("engine"), record.get("dataset"), build)
    return ("cols", record.get("engine"), record.get("dataset"), record.get("m"),
            record.get("build_mode"), record.get("storage_engine"),
            record.get("ef_construction"),
            (record.get("extra") or {}).get("quantization"))


_PARAM_QUANTIZATION = re.compile(r"quantization=([A-Za-z0-9]+)")
_FILE_QUANTIZATION = re.compile(r"_quantization_([A-Za-z0-9]+)_")


def build_params(record: Dict[str, Any]) -> Dict[str, Any]:
    """The reference-build axes of one record, however old the record is.

    `quantization` is a top-level field only on records the loader read since
    it started copying the attribute. A `records.jsonl` merged before that --
    the 25 Aug run's is one -- carries it only inside `extra.parameters`
    (`PerconaSearch(quantization=none, numCandidates=120)`) and the source
    file name (`angular_M_16_quantization_none_120.hdf5`), so both are read
    as fallbacks. Without them every archived Percona Search point would
    match no quantization and the regenerated headline would be all dashes
    for a reason that has nothing to do with the measurement.
    """
    extra = record.get("extra") or {}
    quantization = record.get("quantization") or extra.get("quantization")
    if quantization is None:
        found = _PARAM_QUANTIZATION.search(str(extra.get("parameters") or ""))
        if not found:
            found = _FILE_QUANTIZATION.search(
                os.path.basename(str(extra.get("source_file") or "")))
        quantization = found.group(1) if found else None
    return {"m": record.get("m"),
            "ef_construction": record.get("ef_construction"),
            "quantization": quantization}


def build_label(record: Dict[str, Any]) -> str:
    """What a reader needs to tell this record's build from its siblings."""
    params = build_params(record)
    parts = []
    if params["quantization"] is not None:
        # mongot exposes no graph degree; the M in its file name is a label
        # `render_config` carries, not a setting the index read.
        parts.append(f"quantization={params['quantization']}")
    else:
        if params["m"] is not None:
            parts.append(f"M={params['m']}")
        if params["ef_construction"] is not None:
            parts.append(f"efC={params['ef_construction']}")
    if record.get("build_mode") == "incremental":
        parts.append("incremental")
    return " ".join(parts) or "default build"


def spell_build(build: Optional[Dict[str, Any]]) -> str:
    """A reference build as the page prints it: the same words as a label."""
    if not build:
        return "not recorded"
    if build.get("quantization") is not None:
        return f"quantization={build['quantization']}"
    parts = []
    if build.get("m") is not None:
        parts.append(f"M={build['m']}")
    if "ef_construction" in build:
        value = build["ef_construction"]
        parts.append(f"efC={value if value is not None else 'server default'}")
    # The same word `build_label` prints, and only the non-default one.
    if build.get("build_mode") == "incremental":
        parts.append("incremental")
    return " ".join(parts) or "not recorded"


def matches(record: Dict[str, Any], build: Dict[str, Any]) -> bool:
    """Whether `record` is a point of `build`.

    `build_mode` is matched against the record's own field, and only when
    the record HAS one: the ann overlay declares the axis for pgvector alone,
    so an ann point with no `build_mode` is not excluded by an ops record
    that names one. Before this, `_build_shape` counted build_mode and this
    ignored it, so an archived pgvector run with post and incremental curves
    at one M was "ambiguous" even when its ops pass had built one of them.
    """
    params = build_params(record)
    for axis, value in build.items():
        if axis == "build_mode":
            mode = record.get("build_mode")
            if mode is not None and mode != value:
                return False
            continue
        if params.get(axis) != value:
            return False
    return True


def _from_manifest(manifest: Dict[str, Any], engine: str,
                   resource_pass: Any, dataset: Any = None
                   ) -> Tuple[Optional[Dict[str, Any]], str]:
    """The manifest's build for this cell, and the source to print with it.

    A per-dataset entry first: `ops.build_selection: frontier` chooses the
    ops build per (engine, dataset, pass) from the ann phase's frontier, and
    records it under `presentation.reference_builds_by_dataset` with where it
    came from -- `frontier@0.95`, `below_target@0.95` or
    `fallback_reference` -- which the source carries so the page says which.
    Then the per-pass declared build, as before.
    """
    info = ((manifest or {}).get("engines") or {}).get(engine) or {}
    presentation = info.get("presentation") or {}
    chosen = ((presentation.get("reference_builds_by_dataset") or {})
              .get(resource_pass) or {}).get(dataset)
    if isinstance(chosen, dict) and chosen.get("build"):
        return (dict(chosen["build"]),
                f"{SOURCE_MANIFEST}: {chosen.get('source') or 'chosen'}")
    build = (presentation.get("reference_builds") or {}).get(resource_pass)
    return (dict(build) if build else None), SOURCE_MANIFEST


def _from_ops(records: List[Dict[str, Any]], engine: str, dataset: str,
              resource_pass: Any) -> Optional[Dict[str, Any]]:
    """What the ops pass's own `index_build` record says it built.

    The fallback for a manifest written before reference builds were
    recorded. An axis the ops pass left to the server is kept as None when
    the engine HAS the knob (`extra.ef_construction_tunable`): Valkey's ops
    index on the 25 Aug run was built at the server default, the record does
    not name a value, and choosing among the 64/200/400 ann builds would be a
    guess printed as a reference. None matches no ann point, so the cell is a
    dash and the footnote says why.
    """
    builds = [r for r in records
              if r.get("phase") == "index_build" and r.get("engine") == engine
              and r.get("dataset") == dataset
              and r.get("resource_pass") == resource_pass]
    if not builds:
        return None
    sample = builds[0]
    extra = sample.get("extra") or {}
    params = build_params(sample)
    build: Dict[str, Any] = {}
    if params["quantization"] is not None:
        build["quantization"] = params["quantization"]
    else:
        ms = sorted({r.get("m") for r in builds if r.get("m") is not None})
        if len(ms) == 1:
            build["m"] = ms[0]
        if params["ef_construction"] is not None or extra.get(
                "ef_construction_tunable"):
            build["ef_construction"] = params["ef_construction"]
    # The build mode the ops pass built, when it built exactly one: the
    # ambiguity check below counts build modes, so the fallback has to be
    # able to name one or a pgvector run with both curves can never resolve.
    modes = {r.get("build_mode") for r in builds} - {None}
    if len(modes) == 1:
        build["build_mode"] = modes.pop()
    # Possibly empty: a record that names no axis still says the ops pass
    # built ONE index, and `reference_build` refuses it if that matches more
    # than one swept build rather than falling through to a guess.
    return build


def reference_build(points: List[Dict[str, Any]], records: List[Dict[str, Any]],
                    manifest: Dict[str, Any], engine: str, dataset: str,
                    resource_pass: Any) -> Tuple[Optional[Dict[str, Any]],
                                                 Optional[str]]:
    """The build one engine's headline figures are taken from, and whence.

    The manifest first, which is what `cmd_run` resolved on the host. Then the
    ops pass's own record of what it built, for an archived run. Then, when
    every point shares one value on each reference axis, that build -- there
    is nothing to choose between. Otherwise None: two builds and no
    declaration is not a reason to pick the faster one, which is the exact
    defect this replaces.

    The reference axes only, not `build_signature`: a MariaDB grid swept over
    InnoDB and MyISAM at one M is one reference build in two storage engines,
    which the ops pass also measures both of, and §4 has always named the
    storage engine that won a cell.
    """
    build, source = _from_manifest(manifest, engine, resource_pass, dataset)
    if build:
        return build, source
    build, source = _from_ops(records, engine, dataset, resource_pass), \
        SOURCE_OPS
    if build is None:
        shapes = {tuple(sorted(build_params(p).items())) for p in points}
        if len(shapes) != 1:
            return None, None
        build = {axis: value for axis, value in shapes.pop()
                 if value is not None}
        source = SOURCE_SINGLE
    # The ops harness stamps `build_mode: post` (and `storage_engine:
    # InnoDB`) on every unit, including engines with no such knob: the
    # archived mongot `index_build` record of `full-20260825` says `post`
    # while its ann points say `async`, so a fallback naming the mode matched
    # 0 of the 8 scalar points and the report said "no recall point was
    # measured at its ops build" over eight that were. The mode is kept only
    # where the ann grid actually swept more than one, which is the only case
    # it is needed to tell builds apart (pgvector's post and incremental).
    swept_modes = {p.get("build_mode") for p in points} - {None}
    if "build_mode" in build and len(swept_modes) <= 1:
        build = {axis: value for axis, value in build.items()
                 if axis != "build_mode"}
    # A FALLBACK must name exactly one build. An archived Percona Search
    # `index_build` record with no quantization falls back to {m: 16}, which
    # matches the scalar AND the unquantized sweep -- the 25 Aug defect again,
    # now labelled "ops index_build record"; an m-sweep whose ops record
    # names no M is the same. The manifest's declaration is exempt: it is
    # the build the ops pass was told to make, resolved on the host.
    builds = {_build_shape(p) for p in points if matches(p, build)}
    if len(builds) > 1:
        return None, (f"{SOURCE_AMBIGUOUS}: the {source} names "
                      f"{spell_build(build)}, which matches {len(builds)} "
                      f"swept builds")
    return build, source


def _build_shape(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """A build's identity for the ambiguity check, storage engine excluded.

    Storage is excluded on purpose: InnoDB and MyISAM at one M are one
    reference build measured in two storage engines, both of which the ops
    pass also measures, and §4 names the storage that won each cell.
    """
    return (tuple(sorted(build_params(record).items())),
            record.get("build_mode"))


def _fastest_then_smallest(point: Dict[str, Any]) -> Tuple[Any, ...]:
    """Sort key: fastest first; on an exact QPS tie, the smaller build.

    `max` returned whichever tied point came first in the frontier's order,
    which is the order the result files were walked in -- so two identical
    runs could print different builds under the same number. Smaller wins
    the tie (index size, then M, then ef_construction) because a reader
    offered equal throughput wants the cheaper index; the signature makes
    the last resort deterministic.
    """
    inf = float("inf")
    params = build_params(point)
    size = point.get("index_bytes")
    return (-(point.get("qps") or 0),
            size if isinstance(size, (int, float)) and size > 0 else inf,
            params["m"] if params["m"] is not None else inf,
            params["ef_construction"]
            if params["ef_construction"] is not None else inf,
            str(build_signature(point)))


def smaller_builds_last(points: List[Dict[str, Any]]
                        ) -> List[Dict[str, Any]]:
    """`points` ordered so `pareto_frontier` keeps the smaller of two ties.

    `pareto_frontier` keeps ONE of two points with identical recall and QPS
    -- the one later in its input, through a stable sort walked backwards --
    so which build a tied floor figure named depended on the order the
    result files were walked in. Putting smaller builds last makes the
    smaller one win, deterministically.
    """
    return sorted(points, key=lambda p: _fastest_then_smallest(p)[1:],
                  reverse=True)


def floor_figures(frontier: List[Dict[str, Any]], prefix: str
                  ) -> Dict[str, Any]:
    """The fastest frontier point at or above each floor, with its build."""
    out: Dict[str, Any] = {}
    for floor in FLOORS:
        key = f"{prefix}qps_at_recall_{int(floor * 100)}"
        qualifying = [p for p in frontier
                      if (p.get("recall_at_k") or 0) >= floor]
        best = (min(qualifying, key=_fastest_then_smallest)
                if qualifying else None)
        out[key] = best["qps"] if best else None
        out[f"{key}_storage"] = best.get("storage_engine") if best else None
        out[f"{key}_recall"] = best.get("recall_at_k") if best else None
        out[f"{key}_build"] = build_signature(best) if best else None
        out[f"{key}_build_label"] = build_label(best) if best else None
    return out
