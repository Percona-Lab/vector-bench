"""Per-engine disclosures the records carry and the report must state.

Two kinds, both declared by a driver and stamped onto every record of its ops
run (`harness/main.py` puts them in the recorder's run extra), and for the
query-path facts also onto the ann records (`report/loaders.py` copies the
two attributes an overlay module may set):

* **Declined workloads** -- `extra.declined_workloads`, `{workload: reason}`.
  A workload the profile asked for and the driver refused to measure because
  it could not do so honestly (MyVector v1.26.9 has no filtered search). In a
  mixed run every OTHER engine populates the section, so a decline that is not
  rendered makes the engine simply vanish from it -- which reads as "it was
  not run" or "it was dropped", both wrong.
* **Structural query-path facts** -- `extra.round_trips_per_query` and
  `extra.distance_impl`. Costs the engine's own design puts inside every timed
  query that no other engine pays, which CLAUDE.md requires the report to
  state rather than leave in the code.

Pure functions over the record list; the label function is passed in so this
module imports nothing from `render.py`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

#: `extra.distance_impl` values that mean "no SIMD on this path".
SCALAR_DISTANCE = "scalar"

#: `extra.vector_binding` values that mean the vector crosses the wire as text
#: and is parsed by the server -- every other MySQL-family engine binds packed
#: float32 (VillageSQL sends its SVECTOR values as text).
TEXT_BINDING = "text"

#: `extra.extension_status` values that mean the vector support is a preview
#: feature the vendor does not call stable (VillageSQL's custom indexes).
PREVIEW_STATUS = "preview"

#: The free-text query-path facts copied off ann HDF5 attributes as well as
#: `distance_impl`. Absent for every engine that does not declare them, so
#: their records are unchanged.
_TEXT_FACTS = ("vector_binding", "extension_status")


def query_path_extra(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """The query-path facts an ann overlay module set, for a record's `extra`.

    `round_trips_per_query` and `distance_impl`, read off the HDF5 attributes
    by `report/loaders.py`. Only keys that are present come back, so every
    other engine's record stays byte-identical to what the loader produced
    before. HDF5 hands numbers back as numpy scalars and strings as bytes.
    """
    out: Dict[str, Any] = {}
    trips = attrs.get("round_trips_per_query")
    if trips is not None:
        try:
            out["round_trips_per_query"] = int(trips)
        except (TypeError, ValueError):
            pass
    for key in ("distance_impl",) + _TEXT_FACTS:
        value = attrs.get(key)
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str) and value:
            out[key] = value
    return out


def warmup_extra(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """The one-time index warm-up the overlay module ran, for `extra`.

    Only keys that are present come back, so a result file that predates the
    warm-up (ANN_MEASUREMENT_VERSION 2 and older) yields a record identical to
    the one it always did -- and `summarize` can tell the two apart, which is
    the point: a first-step inversion is the cold-cache artifact only on an
    index nobody warmed.
    """
    # Imported here: coercers needs numpy, and nothing else in this module does.
    from report.coercers import maybe_bool, maybe_float, maybe_int

    out: Dict[str, Any] = {}
    queries = maybe_int(attrs.get("index_warmup_queries"))
    if queries is not None:
        out["index_warmup_queries"] = queries
    seconds = maybe_float(attrs.get("index_warmup_seconds"))
    if seconds is not None and seconds == seconds:
        out["index_warmup_seconds"] = seconds
    converged = maybe_bool(attrs.get("index_warmup_converged"))
    if converged is not None:
        out["index_warmup_converged"] = converged
    return out


def _size(nbytes: int) -> str:
    """GiB for a real corpus, MiB below one -- `0.00 GiB` reads as nothing."""
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.2f} GiB"
    return f"{nbytes / 1024 ** 2:.1f} MiB"


def _extra(record: Dict[str, Any]) -> Dict[str, Any]:
    extra = record.get("extra")
    return extra if isinstance(extra, dict) else {}


def declined_by_engine(records: Iterable[Dict[str, Any]]
                       ) -> Dict[str, Dict[str, str]]:
    """`{engine: {workload: reason}}` from every record that carries a decline."""
    out: Dict[str, Dict[str, str]] = {}
    for record in records:
        declined = _extra(record).get("declined_workloads")
        if not isinstance(declined, dict):
            continue
        engine = str(record.get("engine") or "?")
        for workload, reason in declined.items():
            out.setdefault(engine, {})[str(workload)] = str(reason)
    return out


def declined_lines(records: Iterable[Dict[str, Any]], workload: str,
                   label: Callable[[str], str]) -> str:
    """A markdown note naming each engine that declined `workload`, or ""."""
    declined = declined_by_engine(records)
    rows = [(engine, reasons[workload])
            for engine, reasons in sorted(declined.items())
            if workload in reasons]
    if not rows:
        return ""
    lines = [f"- **{label(engine)}** — declined: `{workload}` ({reason})"
             for engine, reason in rows]
    return ("\n**Declined by the engine's driver, not measured.** These "
            "engines are absent from this section because their driver "
            "refused the workload rather than report a number it could not "
            "stand behind:\n\n" + "\n".join(lines) + "\n")


def all_declined_lines(records: Iterable[Dict[str, Any]],
                       label: Callable[[str], str]) -> str:
    """Every decline in the run, for the Validity section."""
    declined = declined_by_engine(records)
    lines = [f"- **{label(engine)}** — declined: `{workload}` ({reason})"
             for engine, reasons in sorted(declined.items())
             for workload, reason in sorted(reasons.items())]
    if not lines:
        return ""
    return ("\n### Workloads a driver declined\n\n"
            "Requested by the profile and NOT measured for these engines, by "
            "the driver's own declaration (`extra.declined_workloads`). Their "
            "absence from those sections is a statement about the engine, "
            "not a failure of the run.\n\n" + "\n".join(lines) + "\n")


#: The query-path facts an engine's design fixes, for a record that predates
#: their being written. Keyed by engine and, for the distance, by the record's
#: metric. Must equal `MyVectorDriver.DISTANCE_IMPL` and the overlay module's
#: table; `tests/test_e25_myvector_memory_fit.py` pins all three together.
ENGINE_QUERY_PATH_FACTS: Dict[str, Dict[str, Any]] = {
    "myvector": {
        "round_trips_per_query": 2,
        "distance_impl": {"angular": "scalar (computeCosineDistance)",
                          "euclidean": "simd (hnswlib L2Space)"},
    },
}

#: The ann-benchmarks record phase (`harness.metrics.records.PHASE_RECALL`),
#: spelled here so this module stays import-free.
_ANN_PHASE = "recall_qps"


def _derived(record: Dict[str, Any]) -> Dict[str, Any]:
    """The facts for an ann record of a known engine that carries none.

    An ann result cached before the overlay module wrote these attributes
    (MyVector before commit 970a55a) has neither key. ann-benchmarks reuses
    cached results whose parameters match (CLAUDE.md, "The ann results tree
    ... is shared across runs"), so such a record can sit in a new run's
    report. The facts are properties of the engine's design, not of the
    measurement, so they are derived from the engine and the record's metric
    rather than left out -- and the record is counted as stale so the report
    says so.
    """
    known = ENGINE_QUERY_PATH_FACTS.get(str(record.get("engine") or ""))
    if not known or record.get("phase") != _ANN_PHASE:
        return {}
    extra = _extra(record)
    if "round_trips_per_query" in extra and "distance_impl" in extra:
        return {}
    out: Dict[str, Any] = {}
    if "round_trips_per_query" not in extra:
        out["round_trips_per_query"] = known["round_trips_per_query"]
    if "distance_impl" not in extra:
        impl = known["distance_impl"].get(str(record.get("metric_space") or ""))
        if impl:
            out["distance_impl"] = impl
    return out


def structural_facts(records: Iterable[Dict[str, Any]]
                     ) -> Dict[str, Dict[str, Any]]:
    """`{engine: {"round_trips_per_query": n, "distance_impl": {..},
    "stale_ann": count}}`.

    Only engines that declare something different from the default: more than
    one round trip per query, or a scalar distance path. An ann record of a
    known engine that lacks the attributes contributes the engine's facts
    (`_derived`) and is counted in `stale_ann`.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        derived = _derived(record)
        extra = {**_extra(record), **derived}
        engine = str(record.get("engine") or "?")
        if derived:
            entry = out.setdefault(engine, {})
            entry["stale_ann"] = entry.get("stale_ann", 0) + 1
        trips = extra.get("round_trips_per_query")
        if isinstance(trips, (int, float)) and not isinstance(trips, bool) \
                and trips > 1:
            out.setdefault(engine, {})["round_trips_per_query"] = int(trips)
        impl = extra.get("distance_impl")
        if isinstance(impl, str) and impl.startswith(SCALAR_DISTANCE):
            entry = out.setdefault(engine, {})
            entry.setdefault("distance_impl", set()).add(impl)
        binding = extra.get("vector_binding")
        if isinstance(binding, str) and binding.startswith(TEXT_BINDING):
            out.setdefault(engine, {})["vector_binding"] = binding
        status = extra.get("extension_status")
        if isinstance(status, str) and status.startswith(PREVIEW_STATUS):
            out.setdefault(engine, {})["extension_status"] = status
    return out


def index_sync_lines(records: Iterable[Dict[str, Any]],
                     label: Callable[[str], str]) -> str:
    """A note for engines whose `index_sync_s` is not an incremental sync.

    `extra.index_sync_semantics` is declared by the driver. MyVector's refresh
    rewrites the whole graph file on every call, so its `index_sync_s` grows
    with the corpus, not with the rows re-inserted, and is not the same
    quantity as another engine's wait for an asynchronous index to catch up.
    """
    notes: Dict[str, str] = {}
    for record in records:
        note = _extra(record).get("index_sync_semantics")
        if isinstance(note, str) and note:
            notes[str(record.get("engine") or "?")] = note
    if not notes:
        return ""
    lines = [f"- **{label(engine)}** — `index_sync_s` {note}"
             for engine, note in sorted(notes.items())]
    return ("\n**`index_sync_s` is not one quantity across engines.**\n\n"
            + "\n".join(lines) + "\n")


def memory_lines(records: Iterable[Dict[str, Any]],
                 label: Callable[[str], str]) -> str:
    """Where an engine's graph lives, when its driver says it is unusual.

    `extra.memory_note` from the driver, plus the largest
    `extra.myvector_graph_bytes_floor` its build records carry.
    """
    notes: Dict[str, str] = {}
    floors: Dict[str, int] = {}
    for record in records:
        extra = _extra(record)
        engine = str(record.get("engine") or "?")
        note = extra.get("memory_note")
        if isinstance(note, str) and note:
            notes[engine] = note
        floor = extra.get("myvector_graph_bytes_floor")
        if isinstance(floor, int) and not isinstance(floor, bool):
            floors[engine] = max(floor, floors.get(engine, 0))
    if not notes:
        return ""
    lines = []
    for engine, note in sorted(notes.items()):
        line = f"- **{label(engine)}** {note}"
        if engine in floors:
            line += (f" Largest graph built in this run: at least "
                     f"{_size(floors[engine])} "
                     f"(`extra.myvector_graph_bytes_floor`, level 0 only).")
        lines.append(line)
    return ("\n**Memory placement differs.** Read this engine's memory "
            "figures and any out-of-memory failure with the following in "
            "mind:\n\n" + "\n".join(lines) + "\n")


def structural_lines(records: Iterable[Dict[str, Any]],
                     label: Callable[[str], str]) -> str:
    """Footnote text for the QPS/latency, concurrency and Validity sections."""
    facts = structural_facts(records)
    lines: List[str] = []
    for engine, fact in sorted(facts.items()):
        name = label(engine)
        trips = fact.get("round_trips_per_query")
        if trips:
            lines.append(
                f"- **{name}** issues {trips} statements per query, all inside "
                f"the timed region (`extra.round_trips_per_query`): the query "
                f"vector is set into a session variable and the search reads "
                f"it, because the engine's query rewrite cannot carry a long "
                f"vector literal. Every other engine here is one round trip, "
                f"so part of this engine's latency is that extra round trip.")
        for impl in sorted(fact.get("distance_impl") or ()):
            lines.append(
                f"- **{name}** computes this corpus's distance with a "
                f"{impl} implementation (`extra.distance_impl`); its other "
                f"metric uses SIMD kernels, so its throughput on this metric "
                f"is not a statement about its SIMD path.")
        binding = fact.get("vector_binding")
        if binding:
            lines.append(
                f"- **{name}** receives every vector as a text literal "
                f"(`extra.vector_binding`: {binding}): the client formats "
                f"each element and the server parses it on every INSERT and "
                f"inside every timed query. The other MySQL-family engines "
                f"bind packed float32, so part of this engine's load time and "
                f"latency is that conversion.")
        status = fact.get("extension_status")
        if status:
            lines.append(
                f"- **{name}**'s vector type and index are a preview feature "
                f"of the vendor's extension framework "
                f"(`extra.extension_status`: {status}), not a release the "
                f"vendor calls stable; its figures describe a build under "
                f"active development.")
        stale = fact.get("stale_ann")
        if stale:
            lines.append(
                f"- **{name}**: {stale} recall/QPS result(s) predate these "
                f"attributes (an ann result cached before they were written "
                f"and reused by ann-benchmarks); the facts above are derived "
                f"from the engine for them. Move that engine's ann subtree "
                f"aside to re-measure.")
    if not lines:
        return ""
    return ("\n**Structural asymmetries in the query path.** Read these "
            "engines' throughput and latency with the following in mind:\n\n"
            + "\n".join(lines) + "\n")
