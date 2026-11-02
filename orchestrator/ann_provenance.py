"""Which result files one ann phase attempt wrote, and which it stood on.

THE DEFECT THIS EXISTS FOR
--------------------------
`full-20260825`'s manifest listed MySQL-family ann phases from 08-31, and
those phases rendered an M=6 grid and wrote M=6 files. The recall table its
post was written from used M=16 files -- written by 08-26/27 phases whose
entries `Manifest.add_phase` had replaced, because a repeated unit used to
overwrite its earlier attempt. Nothing in the run directory connected a
number in §4 to the phase that measured it, so nothing noticed that the
phase the manifest vouched for had not produced it.

A phase entry now carries the files, so the report can require that every
recall record it publishes was written -- or knowingly reused -- by an ann
phase of this run. Paths are relative to the per-pass results tree
(`results/annb/<pass>/<fingerprint>/`), which is exactly what
`report/loaders.py` stores in `extra.source_file` when it reads a narrowed
tree.

WRITTEN AND SKIPPED, AND WHY "SKIPPED" IS NOT "EVERYTHING ELSE"
---------------------------------------------------------------
ann-benchmarks skips any configuration whose result file already exists and
reports that as success. `run_engine` already says so before the run ("N
existing result file(s) ... will be skipped"); this module records which
ones. A file counts as skipped only if it was there before, is unchanged
after, AND its name is one the rendered grid asks for. The last condition is
the 25 Aug case: the M=16 files sat in the same directory as the M=6 ones the
08-31 phase wrote, untouched -- not skipped, never looked at. Listing every
unchanged file as a cache hit would have vouched for precisely the records
the report has to refuse. With no readable grid nothing is listed as skipped,
which makes the report refuse rather than vouch.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

from .ann_config import result_definitions, result_stem

#: Upstream substitutes these from the dataset and the run, so the stem on
#: disk carries the resolved value ("angular", 10, 960) where the rendered
#: config carries the token. Matched as one run of word characters.
_PLACEHOLDERS = ("@metric", "@count", "@dimension")
_SENTINEL = "VBPLACEHOLDERVB"


def snapshot(results_dir: str, engine: str, dataset: str) -> Dict[str, float]:
    """`{path relative to results_dir: mtime}` for this engine and corpus.

    The same walk `ann_pass._count_results` counts, so the two can never
    disagree about which files belong to the cell.
    """
    root = os.path.join(results_dir, dataset)
    found: Dict[str, float] = {}
    if not os.path.isdir(root):
        return found
    for base, _dirs, files in os.walk(root):
        if os.path.basename(base) != engine:
            continue
        for name in files:
            if not name.endswith(".hdf5"):
                continue
            path = os.path.join(base, name)
            try:
                found[os.path.relpath(path, results_dir)] = \
                    os.path.getmtime(path)
            except OSError:
                continue
    return found


def _substitute(value: Any) -> Any:
    if isinstance(value, str):
        return _SENTINEL if value in _PLACEHOLDERS else value
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    return value


def _grid(body: Any) -> Optional[List[Tuple[Pattern[str], List[Any]]]]:
    """One (file-name pattern, index arguments) per file the body asks for."""
    if not isinstance(body, dict):
        return None
    try:
        definitions = result_definitions(body)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    grid = []
    for _name, index_args, query in definitions:
        stem = result_stem(_substitute(index_args + query))
        pattern = re.escape(stem).replace(_SENTINEL, r"\w+?")
        grid.append((re.compile(f"^{pattern}\\.hdf5$"), index_args))
    return grid


def _m_of(index_args: List[Any]) -> Optional[int]:
    """M, where the arguments name it -- the axis the 25 Aug mix-up was on."""
    for arg in index_args:
        if isinstance(arg, dict):
            for key in ("M", "m"):
                if isinstance(arg.get(key), int):
                    return arg[key]
    return None


def _entry(path: str, mtime: float,
           grid: Optional[List[Tuple[Pattern[str], List[Any]]]]
           ) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": path, "mtime": mtime}
    name = os.path.basename(path)
    for pattern, index_args in grid or ():
        if pattern.match(name):
            entry["build_args"] = index_args
            entry["M"] = _m_of(index_args)
            break
    return entry


def phase_files(before: Dict[str, float], after: Dict[str, float],
                body: Any) -> Dict[str, List[Dict[str, Any]]]:
    """`{"written": [...], "skipped": [...]}` for one attempt.

    Written: absent before, or rewritten (a changed mtime -- `--force`
    rewrites in place). Skipped: present and unchanged, and named by the grid
    -- see the module docstring for why the grid is required.
    """
    grid = _grid(body)
    written = [_entry(path, mtime, grid)
               for path, mtime in sorted(after.items())
               if before.get(path) != mtime]
    skipped: List[Dict[str, Any]] = []
    if grid:
        for path, mtime in sorted(after.items()):
            if before.get(path) != mtime:
                continue
            entry = _entry(path, mtime, grid)
            if "build_args" in entry:
                skipped.append(entry)
    return {"written": written, "skipped": skipped}
