"""Which ann attempt produced each recall number, and what each engine was given.

Two facts the manifest now records and this module reads. Neither may be
derived here from the orchestrator -- the report container mounts report/ and
harness/ only, and importing the orchestrator from report/ once crashed
generation at the end of a twenty-hour run -- so both arrive as manifest data
written on the host.

ANN PROVENANCE
--------------
`full-20260825`'s recall table used M=16 result files. The manifest's phase
list named only the 08-31 MySQL-family ann attempts, which rendered M=6 and
wrote M=6 files; the attempts that wrote the M=16 files had been overwritten
by `Manifest.add_phase`. `report/scope.py` narrows the shared tree to the
(engine, corpus) pairs the phase log names, and that check passed: an ann
phase for that pair existed. It just was not the one that wrote those files.

So a recall record is now admitted to §4 only if an ann attempt of this run,
for the same (pass, engine, corpus), lists its `extra.source_file` -- as
written, or as skipped (a cache hit, admitted and labelled `reused`). Records
no attempt accounts for go to `unprovenanced_ann` and are named in §2,
never drawn. An ops-only phase accounts for no ann file at all: it wrote
none.

**Archived manifests are labelled, not refused.** An attempt recorded before
file lists existed cannot vouch for or against a file, so its pair's records
are kept and labelled `unverified`, and §2 and §4 say so. Refusing them would
delete the recall half of every report written before this change, which is
the over-narrowing `report/scope.py` already declines for the same reason.
Where an ann attempt of the pair HAS a file list, the list is authoritative
for that pair -- a resumed archive's new attempts are checked even though its
old ones cannot be.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: `extra.ann_provenance`, and what §4 and §2 call each.
MEASURED = "measured"
REUSED = "reused"
UNVERIFIED = "unverified"

#: Why a record has no provenance.
NO_ANN_PHASE = "no_ann_phase"
NOT_LISTED = "not_listed"

def _norm(path: Any) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _ann_attempts(manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in ((manifest or {}).get("phases") or [])
            if isinstance(p, dict) and p.get("phase") == "ann"]


def _attempt_index(manifest: Optional[Dict[str, Any]]
                   ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """(pass, engine, corpus) -> what its ann attempts listed.

    Every attempt, not the last: an earlier attempt's files are still on disk
    and a later attempt may have skipped onto them, which is precisely when
    the later one lists them as `skipped`.
    """
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for attempt in _ann_attempts(manifest):
        key = (str(attempt.get("resource_pass") or ""),
               str(attempt.get("engine") or ""),
               str(attempt.get("dataset") or ""))
        cell = index.setdefault(key, {"listed": False, "written": {},
                                      "skipped": {}})
        files = attempt.get("files")
        if not isinstance(files, dict):
            continue
        cell["listed"] = True
        for kind in ("written", "skipped"):
            for entry in files.get(kind) or ():
                if isinstance(entry, dict) and entry.get("path"):
                    cell[kind][_norm(entry["path"])] = entry
    return index


def _cells_for(index, record) -> List[Dict[str, Any]]:
    engine = str(record.get("engine") or "")
    dataset = str(record.get("dataset") or "")
    # Deferred: `report.loaders` imports numpy, and this module is read by
    # the pure-logic half of the generator. One spelling of the sentinel.
    from report.loaders import UNKNOWN_PASS

    resource_pass = record.get("resource_pass")
    if not resource_pass or resource_pass == UNKNOWN_PASS:
        return [cell for (p, e, d), cell in index.items()
                if e == engine and d == dataset]
    cell = index.get((str(resource_pass), engine, dataset))
    return [cell] if cell else []


def _lookup(cell: Dict[str, Any], source: str) -> Tuple[Optional[str], Any]:
    """Whether `source` is one of the cell's files, and as which kind.

    `source_file` is relative to whichever tree the loader read: the per-pass
    tree in the normal case, which is exactly what the attempt lists, or the
    annb root in the whole-tree fallback, where it carries
    `<pass>/<fingerprint>/` in front. A suffix match on a path boundary
    covers both without guessing at the fingerprint.
    """
    for kind in ("written", "skipped"):
        for path, entry in cell[kind].items():
            if source == path or source.endswith("/" + path):
                return kind, entry
    return None, None


def _labelled(record: Dict[str, Any], label: str) -> Dict[str, Any]:
    return {**record,
            "extra": {**(record.get("extra") or {}), "ann_provenance": label}}


def split_by_provenance(records: Sequence[Dict[str, Any]],
                        manifest: Optional[Dict[str, Any]]
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                   Dict[str, Any]]:
    """`(kept, unprovenanced census, provenance summary)`.

    Non-recall records pass through untouched. Every kept recall record is a
    new dict carrying `extra.ann_provenance`. A manifest with no phase log at
    all -- a run killed before its first unit returned, or a hand-built
    bundle -- can vouch for nothing and refute nothing, so its recall is kept
    and labelled unverified, the same bargain `scope.narrow_to_measured_scope`
    strikes for an unnarrowable manifest.
    """
    from harness.metrics.records import PHASE_RECALL

    index = _attempt_index(manifest)
    has_log = bool((manifest or {}).get("phases"))
    kept: List[Dict[str, Any]] = []
    census: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    reused: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    unverified: Dict[Tuple[str, str, str], int] = {}
    counts = {MEASURED: 0, REUSED: 0, UNVERIFIED: 0}

    def keep(record: Dict[str, Any], label: str) -> None:
        counts[label] += 1
        kept.append(_labelled(record, label))

    for record in records:
        if record.get("phase") != PHASE_RECALL:
            kept.append(record)
            continue
        key3 = (str(record.get("resource_pass") or ""),
                str(record.get("engine") or ""),
                str(record.get("dataset") or ""))
        cells = _cells_for(index, record)
        source = _norm((record.get("extra") or {}).get("source_file"))
        listed = [c for c in cells if c["listed"]]
        # No source file means the record did not come out of the tree -- a
        # hand-assembled or third-party records file -- and no file list can
        # speak to it. Kept and labelled, as `scope.py` keeps a record it
        # cannot attribute.
        if (not has_log or not source or not key3[1] or not key3[2]
                or (cells and not listed)):
            unverified[key3] = unverified.get(key3, 0) + 1
            keep(record, UNVERIFIED)
            continue
        if not cells:
            reason = NO_ANN_PHASE
        else:
            reason = NOT_LISTED
            found = [hit for hit in (_lookup(c, source) for c in listed)
                     if hit[0]]
            if found:
                kind, entry = found[0]
                if kind == "written":
                    keep(record, MEASURED)
                    continue
                row = reused.setdefault(key3, {
                    "resource_pass": key3[0], "engine": key3[1],
                    "dataset": key3[2], "files": set(), "oldest": None})
                row["files"].add(source)
                mtime = entry.get("mtime")
                if mtime and (row["oldest"] is None or mtime < row["oldest"]):
                    row["oldest"] = mtime
                keep(record, REUSED)
                continue
        row = census.setdefault(key3 + (reason,), {
            "resource_pass": key3[0], "engine": key3[1], "dataset": key3[2],
            "reason": reason, "records": 0, "files": set(), "m": set()})
        row["records"] += 1
        if source:
            row["files"].add(source)
        if record.get("m") is not None:
            row["m"].add(record["m"])

    unprovenanced = sorted(
        ({**row, "files": len(row["files"]), "m": sorted(row["m"])}
         for row in census.values()),
        key=lambda r: (r["engine"], r["dataset"], r["resource_pass"]))
    summary = {
        "counts": counts,
        "reused": sorted(
            ({**row, "files": len(row["files"])} for row in reused.values()),
            key=lambda r: (r["engine"], r["dataset"], r["resource_pass"])),
        "unverified": [
            {"resource_pass": p, "engine": e, "dataset": d, "records": n}
            for (p, e, d), n in sorted(unverified.items())],
    }
    return kept, unprovenanced, summary


# ---------------------------------------------------------------------------
# What each engine was given
# ---------------------------------------------------------------------------

def engines_resolved(manifest: Optional[Dict[str, Any]]
                     ) -> Dict[str, Dict[str, Any]]:
    block = (manifest or {}).get("engines_resolved")
    return block if isinstance(block, dict) else {}


def _pass_of_series(name: str) -> Optional[str]:
    """`<engine>-<dataset>-<pass>-<tag>`: the pass is a whole token."""
    tokens = name.split("-")
    for candidate in ("normalized", "tuned"):
        if candidate in tokens:
            return candidate
    return None


def series_memory_limit(manifest: Optional[Dict[str, Any]],
                        series: str) -> Tuple[Optional[int], str]:
    """`(server memory limit, where it came from)` for a memory series.

    `"engine"` when `engines_resolved` has that (engine, pass). Otherwise the
    pass's own entry in `config.passes`, then `config.resolved_resources` --
    the latter being whichever engine `set_config` saw last, which is the
    value this used to read for every engine. The memory LIMIT happens to be
    pass-level today, so the fallback is usually the right number; the
    caveat is stated anyway, because "usually" is how the buffer figures in
    `full-20260825` came to be published as everyone's.
    """
    engine = series.split("-", 1)[0]
    resource_pass = _pass_of_series(series)
    entry = (engines_resolved(manifest).get(engine) or {}).get(
        resource_pass or "")
    if isinstance(entry, dict):
        limit = (entry.get("resolved") or {}).get("server_memory_bytes")
        if limit:
            return limit, "engine"
    config = (manifest or {}).get("config") or {}
    for pass_entry in config.get("passes") or ():
        if (isinstance(pass_entry, dict)
                and pass_entry.get("resource_pass") == resource_pass):
            limit = (pass_entry.get("resolved_resources") or {}).get(
                "server_memory_bytes")
            if limit:
                return limit, "run-level"
    return ((config.get("resolved_resources") or {}).get(
        "server_memory_bytes"), "run-level")


#: The resolved key §1's build-memory column reads -- spelled once.
_BUILD_MEMORY = "maintenance_bytes"


def _applied(entry: Dict[str, Any], key: str) -> bool:
    """Whether a resolved figure reached a server flag.

    From the host's own `applied` list where the entry has one: it is read
    off the engine's flag templates, so it cannot be fooled by two figures
    resolving to the same byte count -- MyVector's buffer and graph cache do,
    and matching values said it had a graph cache it has no flag for. For an
    entry without the list, the value is looked for in the rendered args,
    which is right whenever the figures differ.

    Either way resolving a figure and applying it are different facts:
    MongoDB resolves a buffer share and starts on WiredTiger's own default,
    and the table must not print the first as if it were the second.
    """
    applied = entry.get("applied")
    value = (entry.get("resolved") or {}).get(key)
    if not value:
        return False
    if isinstance(applied, list):
        return key in applied
    text = str(value)
    return any(text in str(arg) for arg in entry.get("server_args") or ())


def engine_budget_rows(manifest: Optional[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
    """One row per (engine, pass) recorded, in engine then pass order."""
    engines_block = (manifest or {}).get("engines") or {}
    rows: List[Dict[str, Any]] = []
    for engine, by_pass in sorted(engines_resolved(manifest).items()):
        for resource_pass, entry in sorted((by_pass or {}).items()):
            entry = entry if isinstance(entry, dict) else {}
            resolved = entry.get("resolved") or {}
            container = entry.get("container") or {}
            if _applied(entry, "maxmemory_bytes"):
                main = ("maxmemory", resolved["maxmemory_bytes"])
            elif _applied(entry, "buffer_bytes"):
                main = ("buffer", resolved["buffer_bytes"])
            else:
                main = ("default", None)
            graph = resolved.get("graph_cache_bytes")
            build = resolved.get(_BUILD_MEMORY)
            rows.append({
                "engine": engine,
                "resource_pass": resource_pass,
                "main_cache": main,
                "graph_cache": (graph if _applied(entry, "graph_cache_bytes")
                                else None),
                "build_memory": (build if _applied(entry, _BUILD_MEMORY)
                                 else None),
                "build_threads": resolved.get("build_threads"),
                "shm_size": container.get("shm_size",
                                          resolved.get("shm_size")),
                "cpuset": container.get("cpuset",
                                        resolved.get("server_cpuset")),
                "march": ((engines_block.get(engine) or {}).get("build")
                          or {}).get("march"),
            })
    return rows


def latest_attempts(phases: Any) -> List[Dict[str, Any]]:
    """The last attempt of each unit, in phase-log order.

    `Manifest.add_phase` appends every attempt now; it used to replace, so
    that a failure fixed by a re-run would stop being reported. The readers
    that ask "how did this unit end" -- §2's failed and refused lists, the
    silent-ann-failure check, the web UI's tally -- read this; the provenance
    check above reads every attempt, because an earlier attempt's files are
    still on disk.
    """
    last: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for phase in phases or ():
        if isinstance(phase, dict):
            unit = (phase.get("phase"), phase.get("engine"),
                    phase.get("dataset"), phase.get("resource_pass"))
            last.pop(unit, None)
            last[unit] = phase
    return list(last.values())
