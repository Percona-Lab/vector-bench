"""Which build-knob values each engine can actually take.

WHY THIS EXISTS. Until the frontier-explore profile, every grid this repo
rendered sat comfortably inside every engine's legal range -- M from 6 to 32,
ef_construction from 64 to 400 -- so nothing here needed to know where the
ranges ended. A reviewer of `full-20260825` asked for M = 4, 6, 8, 12, 16, 32
and 48 on every engine, and two of those values meet a vendor limit:

* pgvector refuses an index whose `ef_construction` is below `2 * m`
  (`hnswbuild.c`, "ef_construction must be greater than or equal to 2 * m").
  M=48 at ef_construction 64 is therefore not a slow build or a bad build; it
  is a `CREATE INDEX` error, and ann-benchmarks would record it as a missing
  definition after the rest of the sweep had run.
* hnswlib -- MyVector's index, and the HNSW under valkey-search -- does
  `ef_construction_ = max(ef_construction, M_)`. An ef_construction below M
  is not refused: it is raised to M without a word, and the result file would
  then carry an ef_construction label the graph was never built with. No
  grid here reaches that today (the smallest ef_construction swept is 64, the
  largest M 48), and the rule is enforced so the first grid that would is
  told rather than mislabelled.

So there are two kinds of answer, and they are handled differently:

* A value outside an engine's range on its own -- M=2 on MariaDB, M=150 on
  pgvector -- is REFUSED (`knob_problems`). `render_config` raises, and `run`
  refuses before loading anything, naming the engine, the value and the
  bound. The remedy is a per-engine list (`m_values_by_engine`), which the
  profile can spell; dropping the value here would silently shrink the
  engine's grid below what the profile asked for.
* A combination that is illegal while each value is fine -- pgvector M=48 at
  ef_construction 64 -- is DROPPED from the render (`excluded_builds`) and
  named. The profile has no way to spell "M=48 except at ef_construction 64"
  (the grid is a cartesian product), and refusing would make the whole
  M=48 column unmeasurable for an engine that can build it at 200 and 400.
  Every dropped build is printed before the run, by `cmd_run` and
  `cmd_render`, and `select-builds` records it among the pruned builds with
  this module's reason.

Never clamped. Rounding M=2 up to MariaDB's 3 would produce a curve labelled
M=2 that is M=3, which is the shape of failure this repository exists to
prevent.

WHERE EACH BOUND COMES FROM. Pinned source, read in the trees this repo builds
from (`sources/` holds the exported tags; the vendor checkouts beside the repo
are read-only):

* MariaDB 11.8.8 and 12.3.2: `sql/vector_mhnsw.cc`, the `default_m` sysvar,
  `MYSQL_THDVAR_UINT(default_m, ..., 6, 3, 200, 1)` -- default 6, range
  3..200. The table option `M=` is `HA_IOPTION_SYSVAR("m", M, default_m)`, so
  it inherits that range.
* AliSQL 8.0.44-2: `include/vidx/vidx_common.h`, `M_MIN = 3`, `M_MAX = 200`
  (`M_DEF = 6`), read by the `hnsw_default_m` sysvar in
  `sql/vidx/vidx_index.cc`.
* pgvector v0.8.6: `src/hnsw.h`, `HNSW_MIN_M 2`, `HNSW_MAX_M 100`,
  `HNSW_MIN_EF_CONSTRUCTION 4`, `HNSW_MAX_EF_CONSTRUCTION 1000`; the 2*m rule
  in `src/hnswbuild.c`.
* Percona Server: `MIN_M`, `MAX_M`, `MIN_EF_CONSTRUCTION` and
  `MAX_EF_CONSTRUCTION` in `overlay/.../algorithms/percona/module.py`, which
  cites `storage/innobase/include/vec0vec.h` and refuses out-of-range values
  itself. Repeated here, because the overlay module is not importable from the
  host; `tests/test_two_stage_m_sweep.py` reads the module's source and fails
  if the two disagree.
* MyVector v1.26.9 (hnswlib): no explicit bound in the plugin. hnswlib computes
  `mult_ = 1 / log(M_)`, which is a division by zero at M=1, and caps M above
  10000 with a printed warning that goes to mysqld's stderr -- a silent clamp
  from the benchmark's point of view. Hence 2..10000.
* valkey-search: installed from a package, so its FT.CREATE validation is not
  in this checkout and nothing here claims to know it. Its HNSW is hnswlib's,
  so hnswlib's own mechanics -- the 1/log(M) floor and the ef_construction
  raise -- are enforced and nothing else. A value valkey-search refuses fails
  FT.CREATE loudly and the phase reports the missing definitions.
* VillageSQL + vsql_vector, built from vsql-vector-pk at `93ed0087` (the
  extension, not the server, owns the index options; the constants and
  `Options::validate` are the same as plain vsql-vector's at `bac885dd`):
  `src/index/hnsw/storage.h`, `Options::MAX_M = 128` and
  `MAX_EF_CONSTRUCTION_FACTOR = 32`; `Options::validate` in
  `src/index/hnsw/storage.cc` refuses `M < 2`, `M > MAX_M`,
  `ef_construction < M` and `ef_construction > M * 32`. The last two are
  combination rules (`excluded_builds`); on their own the values run from 2
  (the smallest M) to 4096 (32 x the largest). The source is not in this
  checkout -- `scripts/prepare-sources.sh` fetches it into `sources/` -- so
  these constants are pinned by `tests/test_e26_villagesql_engine.py` against
  the overlay module, which refuses the same combinations itself, and against
  the exported tree where it exists. The PR that added the engine CLAMPED
  ef_construction to 32 x M (200 became 192 at M=6) and recorded both values;
  the result file was still labelled 200, which is the mislabelling this module
  exists to prevent, so the combination is dropped instead.
* Percona Search (mongot): no knob. `render_config` carries M as a label and
  never sends it, so there is nothing to bound; the profiles pin it to one
  value per run (`m_values[:1]`).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

#: (lowest, highest) legal value per knob, per engine. `None` on either side
#: means no bound is known on that side. An engine or knob absent from the
#: table is not checked -- see the module docstring for which and why.
KNOB_BOUNDS: Dict[str, Dict[str, Tuple[Optional[int], Optional[int]]]] = {
    "mariadb": {"M": (3, 200)},
    "mariadb123": {"M": (3, 200)},
    "alisql": {"M": (3, 200)},
    "pgvector": {"M": (2, 100), "efConstruction": (4, 1000)},
    "percona": {"M": (2, 200), "efConstruction": (1, 4096)},
    "myvector": {"M": (2, 10000)},
    "valkey": {"M": (2, 10000)},
    "villagesql": {"M": (2, 128), "efConstruction": (2, 4096)},
    # The same vsql-vector-pk commit on the MySQL 8.4 server line: the bounds
    # are the extension's, not the server's.
    "villagesql84": {"M": (2, 128), "efConstruction": (2, 4096)},
}

#: The engines whose index is vsql_vector's HNSW: one per VillageSQL server
#: line, the same extension commit in each. Every engine-keyed branch that
#: is about the EXTENSION (its ef_construction rule, its ann grid, the ops
#: ef_construction) takes this tuple rather than one name, as
#: `ann_pass.MARIADB_ENGINES` does for the two MariaDB versions, so a branch
#: written for `villagesql` alone cannot silently skip `villagesql84`.
VILLAGESQL_ENGINES = ("villagesql", "villagesql84")

#: Engines whose HNSW is hnswlib's, which raises ef_construction to M silently.
HNSWLIB_ENGINES = ("myvector", "valkey")

REASON_PGVECTOR_2M = ("pgvector refuses ef_construction < 2 * m "
                      "(hnswbuild.c: 'ef_construction must be greater than or "
                      "equal to 2 * m')")
#: vsql_vector's `Options::validate`: M <= ef_construction <= 32 * M.
VILLAGESQL_EFC_FACTOR = 32
REASON_VILLAGESQL_EFC = ("vsql_vector refuses ef_construction < M or > 32 * M "
                         "(storage.cc Options::validate: 'ef_construction "
                         "(%u) must be >= M' / 'exceeds maximum allowed "
                         "value (%u) for M=%u')")
REASON_HNSWLIB_EFC = ("hnswlib raises ef_construction to M without a word "
                      "(ef_construction_ = max(ef_construction, M_)), so the "
                      "build would not be the one its label names")


def knob_problems(engine: str, m_values: Iterable[int],
                  ef_constructions: Iterable[int] = ()) -> List[str]:
    """One sentence per value this engine cannot take on its own, or []."""
    bounds = KNOB_BOUNDS.get(engine) or {}
    problems: List[str] = []
    for knob, values in (("M", m_values), ("efConstruction", ef_constructions)):
        if knob not in bounds:
            continue
        low, high = bounds[knob]
        for value in values:
            value = int(value)
            if (low is not None and value < low) or (
                    high is not None and value > high):
                problems.append(
                    f"{engine}: {knob}={value} is outside the engine's legal "
                    f"range [{low}, {high}] (orchestrator/ann_knobs.py "
                    f"KNOB_BOUNDS). Give {engine} its own list with "
                    f"m_values_by_engine / <engine>_ef_construction instead "
                    f"of a value it would refuse or silently clamp.")
    return problems


def excluded_builds(engine: str, m_values: Iterable[int],
                    ef_constructions: Iterable[int]
                    ) -> List[Tuple[int, int, str]]:
    """(M, ef_construction, reason) for every combination the engine cannot
    build although each value is legal alone. Order follows the inputs."""
    out: List[Tuple[int, int, str]] = []
    efcs = [int(e) for e in ef_constructions]
    for m in (int(v) for v in m_values):
        for efc in efcs:
            if engine == "pgvector" and efc < 2 * m:
                out.append((m, efc, REASON_PGVECTOR_2M))
            elif engine in HNSWLIB_ENGINES and efc < m:
                out.append((m, efc, REASON_HNSWLIB_EFC))
            elif engine in VILLAGESQL_ENGINES and not (
                    m <= efc <= VILLAGESQL_EFC_FACTOR * m):
                out.append((m, efc, REASON_VILLAGESQL_EFC))
    return out


def efc_run_groups(base_name: str, arg_group: Dict[str, Any], engine: str,
                   query_args: List[List[int]]
                   ) -> Dict[str, Dict[str, object]]:
    """The run groups for an engine with both M and ef_construction.

    `arg_group` is the single `arg_groups` entry the branch in
    `render_config` would emit -- `{"M": ..., "efConstruction": ..., ...}` --
    so each branch still spells its own grid where docs/09 §4.10 says it does.

    One group, exactly as before, when every combination is legal -- so every
    profile that rendered before this module renders byte-identically. Only
    when `excluded_builds` names something is the grid split into one group per
    ef_construction, each with the M values legal at it. Splitting by run group
    rather than by `arg_groups` entry matters: upstream MULTIPLIES the entries
    of one `arg_groups` list (see `ann_config._combinations`), so two entries
    would have produced the product again. The run-group name is not part of
    the result file name, so the split moves no file.
    """
    m_values = list(arg_group["M"])
    ef_constructions = list(arg_group["efConstruction"])
    args = {k: v for k, v in arg_group.items()
            if k not in ("M", "efConstruction")}
    excluded = {(m, e) for m, e, _ in excluded_builds(
        engine, m_values, ef_constructions)}
    if not excluded:
        return {base_name: {
            "arg_groups": [dict(arg_group)],
            "args": {},
            "query_args": query_args,
        }}
    # A whole ef_construction, or a whole M, with nothing buildable is a
    # profile asking for an axis value the engine can never take: refused,
    # because dropping it would remove the value from the grid without a
    # combination left to name it (and `excluded_ann_builds`, which reads the
    # render back, would then not see it either).
    for m in m_values:
        if all((int(m), int(e)) in excluded for e in ef_constructions):
            raise ValueError(f"{engine}: M={m} is buildable at none of "
                             f"ef_construction {list(ef_constructions)}")
    groups: Dict[str, Dict[str, object]] = {}
    for efc in ef_constructions:
        legal = [m for m in m_values if (int(m), int(efc)) not in excluded]
        if not legal:
            raise ValueError(f"{engine}: ef_construction={efc} is buildable "
                             f"at none of M {list(m_values)}")
        groups[f"{base_name}_efc{efc}"] = {
            "arg_groups": [{"M": legal, "efConstruction": [efc], **args}],
            "args": {},
            "query_args": query_args,
        }
    return groups


#: What Percona Search's ann module accepts (`mongodb/module.py` refuses
#: anything else at construction); pinned to that source by test.
QUANTIZATIONS = ("none", "scalar", "binary")

#: The build axes a selected build may name, as `render_config` spells them.
BUILD_AXES = ("M", "efConstruction", "quantization")


def restrict_to_builds(engine: str, run_groups: Dict[str, Dict[str, Any]],
                       selected: List[Dict[str, Any]]
                       ) -> Dict[str, Dict[str, Any]]:
    """Keep only the builds `selected` names, one run group per build.

    WHY. A profile's grid is per-axis lists, which upstream multiplies, so a
    stage-2 selection written as lists alone re-admits every product of the
    kept values: three chosen pgvector builds came back as eight, five of them
    builds nobody chose, each a full 990,000-row load. The tuned pass's
    `resources.extras.ann_builds_by_engine` names the builds themselves, and
    this cuts the rendered grid down to exactly those.

    A selected build matches a rendered combination when every axis it names
    is equal -- so a mongodb entry naming only `quantization` matches the
    combination carrying mongot's M label. Each match becomes its own run
    group whose list values are one-element lists (scalars stay scalars), so
    the expansion, and therefore the result file name, is the one the full
    grid would have produced. A selected build the grid does not contain is
    refused: measuring fewer builds than the selection lists, silently, is
    the failure this exists to prevent.
    """
    from .ann_config import _expand

    wanted = [dict(b) for b in selected]
    matched = set()
    out: Dict[str, Dict[str, Any]] = {}
    for name, group in run_groups.items():
        for entry in group["arg_groups"]:
            for combo in _expand(entry):
                hits = [i for i, b in enumerate(wanted)
                        if all(combo.get(a) == v for a, v in b.items())]
                if not hits:
                    continue
                matched.update(hits)
                out[f"{name}_{len(out)}"] = {
                    "arg_groups": [{k: ([combo[k]] if isinstance(v, list)
                                        else v) for k, v in entry.items()}],
                    "args": {},
                    "query_args": group["query_args"],
                }
    missing = [b for i, b in enumerate(wanted) if i not in matched]
    if missing:
        raise ValueError(f"{engine}: ann_builds_by_engine names build(s) the "
                         f"grid does not contain: {missing}. Add their values "
                         f"to the per-axis lists, or regenerate the overlay.")
    return out
