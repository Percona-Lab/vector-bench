"""The ann phase's cost counted from the grid it will actually build.

WHY A SECOND FIGURE. `estimate.estimate_load_hours` charges every engine one
corpus reload per entry of `ann.m_values`. That was right while every engine
swept M and nothing else. It stopped being right when the tuned pass gave
engines their own axes -- pgvector, Valkey, Percona and MyVector sweep
ef_construction, Percona Search sweeps quantization, the MySQL-family engines
take `m_values_by_engine` -- and ann-benchmarks reloads the corpus for EVERY
build definition, not every M. Under `tuned-complete` the published figure
charges pgvector one reload where the render asks for three; under the
frontier-explore profile it charges Percona seven where the render asks for
twenty-one.

Correcting the published figure in place is not done here, deliberately. It is
quoted by README's front-page table, docs/07, docs/10 and CLAUDE.md, and
`tests/test_estimate.py` pins every one of those against it; moving it is a
change to every published planning figure and belongs in a change of its own.
What this module adds is the per-engine count of builds the rendered grid
asks for, the ingest hours that implies, and the difference from what the
published line charged -- printed beside it before the run, so a profile
whose grid the old arithmetic cannot see is planned against the right number.

M IS PRICED, where it changes the load. The rate tables in `estimate.py` are
mid-grid rates at M=16, and for an engine that builds its graph inside every
INSERT the cost of a load rises with M: MariaDB measured ~320 rows/s at M=8
and ~70 at M=32 on glove-100 (the comment above `estimate._INGEST_ROWS_PER_S`),
a factor of 4.6 over a fourfold change in M, so cost ~ M^1.1. The factor is
applied to those engines only (`INSERT_BUILD_ENGINES`). For the engines whose
rate is the load alone -- pgvector, Valkey, Percona Search, MyVector build
after it -- the build itself is not in either figure, as it never was, and the
printed block says so. An M of 48 is therefore charged ~3.3x an M=16 load on
MariaDB, AliSQL and Percona Server, and the same load on the others.

Nothing here starts a container or reads a run directory. It renders each
engine's ann config, which is pure, and counts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from . import ann_pass
from .ann_config import result_definitions
from .config import load_resources, merge_resource_overrides
# The rate tables and the rate rule, borrowed rather than copied: a second
# copy is how two estimates of one plan come to disagree.
from .estimate import _DATASET_ROWS, _effective_rate

#: Engines whose ann load builds the HNSW graph row by row, so a larger M is a
#: slower load. Percona Server is here because its ann module inherits
#: `vb_mysql.VBMySQLBase.fit`, which declares the index before the INSERTs.
INSERT_BUILD_ENGINES = ("mariadb", "mariadb123", "alisql", "percona")

#: Load cost ~ M^exponent relative to the M=16 rate the tables carry. From
#: MariaDB's glove-100 rates at M=8 and M=32: log(320/70) / log(4) = 1.10.
M_COST_EXPONENT = 1.1
M_RATE_REFERENCE = 16


def m_cost_factor(engine: str, m: Any) -> float:
    """How many M=16 loads one load at this M costs, for this engine."""
    if engine not in INSERT_BUILD_ENGINES or m is None:
        return 1.0
    return (float(m) / M_RATE_REFERENCE) ** M_COST_EXPONENT


def ann_builds(engine: str, profile: Dict[str, Any],
               resources: Dict[str, Any], resource_pass: str
               ) -> List[Dict[str, Any]]:
    """Every distinct build the rendered config asks for, as its index args.

    Distinct by value: ann-benchmarks names a result file from the arguments,
    so two definitions with equal arguments are one build on disk and one
    load (see `ann_config`'s module docstring).
    """
    body = ann_pass.render_config(engine, profile, resources, resource_pass)
    seen: Dict[str, Dict[str, Any]] = {}
    for _name, arguments, _query in result_definitions(body):
        build = next((a for a in arguments if isinstance(a, dict)), {})
        seen.setdefault(json.dumps(build, sort_keys=True), build)
    return list(seen.values())


def ann_grid_estimate(profile: Dict[str, Any], engines: Sequence[str],
                      datasets: Sequence[str], passes: Sequence[str],
                      phases: Sequence[str]) -> Dict[str, Any]:
    """Per engine: builds, the largest M, and the ann ingest they imply.

    `charged_hours` is what `estimate_load_hours` put on the ann side of the
    published ingest figure for the same engine, so `extra_hours` is the
    amount the plan costs beyond that line. Engines without an ingest rate are
    left out of the hours exactly as they are there, and listed.
    """
    out: Dict[str, Dict[str, Any]] = {}
    ann = profile.get("ann", {}) or {}
    if "ann" not in phases or not ann.get("enabled", True):
        return {"engines": out, "extra_hours": 0.0}
    m_count = max(1, len(ann.get("m_values", [16])))
    for engine in engines:
        builds_total = 0
        largest_m = None
        hours = charged = 0.0
        rated = False
        for resource_pass in passes:
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            builds = ann_builds(engine, profile, resources, resource_pass)
            builds_total += len(builds)
            for build in builds:
                m = build.get("M")
                if m is not None:
                    largest_m = max(int(m), largest_m or 0)
            for dataset in datasets:
                rows = _DATASET_ROWS.get(dataset)
                rate = _effective_rate(engine, dataset)
                if not rows or rate is None:
                    continue
                rated = True
                load_h = rows / rate / 3600
                hours += sum(load_h * m_cost_factor(engine, b.get("M"))
                             for b in builds)
                charged += load_h * m_count
        out[engine] = {
            "builds": builds_total,
            "largest_m": largest_m,
            "hours": round(hours, 2) if rated else None,
            "charged_hours": round(charged, 2) if rated else None,
        }
    extra = sum((v["hours"] or 0.0) - (v["charged_hours"] or 0.0)
                for v in out.values())
    return {"engines": out, "extra_hours": round(extra, 2)}


def print_ann_grid(profile: Dict[str, Any], engines: Sequence[str],
                   datasets: Sequence[str], passes: Sequence[str],
                   phases: Sequence[str]) -> Dict[str, Any]:
    """Print the per-engine build count and its cost, before the run.

    Silent when every engine builds exactly what the published line charged
    for and no M moves a load's price -- the common case, which must not grow
    a block that says nothing.
    """
    grid = ann_grid_estimate(profile, engines, datasets, passes, phases)
    rows = grid["engines"]
    if not rows or abs(grid["extra_hours"]) < 0.05:
        return grid
    print("\nann builds the rendered grid asks for (each one reloads the "
          "corpus):")
    for engine, row in rows.items():
        cost = ("no ingest rate" if row["hours"] is None
                else f"~{row['hours']:.1f} h (line above: "
                     f"~{row['charged_hours']:.1f} h)")
        largest = (f", largest M {row['largest_m']}"
                   if row["largest_m"] is not None else "")
        print(f"  {engine:<11} {row['builds']:>3} build(s){largest}  {cost}")
    sign = "+" if grid["extra_hours"] >= 0 else "-"
    print(f"  {sign}{abs(grid['extra_hours']):.1f} h against the ingest "
          f"total above, which charges every engine "
          f"{max(1, len((profile.get('ann') or {}).get('m_values', [16])))} "
          f"reload(s) per pass. Plan against this block.")
    print(f"  Loads on {', '.join(INSERT_BUILD_ENGINES)} are priced at "
          f"(M/{M_RATE_REFERENCE})^{M_COST_EXPONENT} of the M=16 rate; the "
          f"bulk index builds of the\n  others come after the load and are "
          f"in neither figure.")
    return grid
