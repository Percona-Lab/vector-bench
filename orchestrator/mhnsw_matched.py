"""Percona's ef_construction set per M to MariaDB MHNSW's build width.

WHY. MHNSW has no ef_construction knob. Its insert path widens every build
search to a fixed constant or to the layer's neighbour cap, whichever is
larger, so its build effort is a function of M alone -- 32 candidates at M=16.
Percona's InnoDB HNSW is built at ef_construction 200 everywhere else in this
repo (the server's default, `PERCONA_DIALECT.ef_construction`), six times that
at M=16. A Percona-vs-MariaDB recall/QPS curve at those settings compares two
implementations AND two build budgets, and nothing on the chart separates the
two. `config/profiles/percona-efc-matched.yml` builds Percona at MHNSW's width
instead, so what remains between the curves is the implementation.

THE FORMULA, read at both MariaDB pins this repo builds (tags mariadb-11.8.8
and mariadb-12.3.2, `sql/vector_mhnsw.cc`, identical in both):

* `static constexpr uint ef_construction= 10;` -- a file-scope constant, not
  a sysvar and not a table option.
* `mhnsw_insert` searches each layer at or below the new node's level with
  `search_layer(..., max_neighbors, ..., construction=true)`, where
  `MHNSW_Share::max_neighbors(layer)` is `(layer ? 1 : 2) * M`.
* `search_layer`, when `construction` is true, sets
  `ef= std::max(ef_construction, ef)` for any `ef > 1`.

So the build candidate list is max(10, 2*M) on layer 0 -- the layer that holds
every node and that a query's final search runs on -- and max(10, M) above it.
`mhnsw_build_width` returns the layer-0 figure, because Percona's
ef_construction is one number and layer 0 is where recall is decided. The
upper-layer difference (MHNSW at M, Percona at the same 2*M) is stated in
docs/05-methodology.md rather than hidden; it makes Percona's matched build
slightly MORE generous than MHNSW's on the sparse layers, never less.

ONE PLACE FOR THE NUMBER. The ann render (`ann_pass.render_config`) and the
ops build (`ops_pass.ops_ef_construction`) both read the profile's
`ann.percona_ef_construction_by_m` through `efc_for_m` below, and
`tests/test_config.py` pins the shipped profile's map to
`mhnsw_build_width` for every M it sweeps -- so a hand-edited map that drifts
from the formula fails a test rather than publishing a "matched" curve that
is not.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

#: `sql/vector_mhnsw.cc`, `static constexpr uint ef_construction= 10;`, at
#: mariadb-11.8.8 and mariadb-12.3.2.
MHNSW_EF_CONSTRUCTION = 10

#: The profile key: `{M: ef_construction}` for Percona, one entry per M.
BY_M_KEY = "percona_ef_construction_by_m"


def mhnsw_build_width(m: int) -> int:
    """MHNSW's layer-0 build candidate width at graph degree `m`.

    `max(ef_construction, max_neighbors(0))` = max(10, 2*M); see the module
    docstring for the three symbols this is read from.
    """
    return max(MHNSW_EF_CONSTRUCTION, 2 * int(m))


def efc_by_m(profile: Dict[str, Any]) -> Optional[Dict[int, int]]:
    """The profile's per-M Percona ef_construction map, or None if unset.

    Keys are coerced to int: YAML gives ints, but a profile that has been
    through JSON (the web UI) gives strings, and `{"16": 32}` must not miss
    M=16 and fall through to the 200 default.
    """
    ann = profile.get("ann", {}) or {}
    raw = ann.get(BY_M_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"ann.{BY_M_KEY} must be a non-empty mapping of "
                         f"M to ef_construction, got {raw!r}")
    if ann.get("percona_ef_construction") is not None:
        # Two answers to one question. Which one wins would be a rule nobody
        # reads, and the loser's value would still sit in the profile looking
        # like what was measured.
        raise ValueError(f"ann.{BY_M_KEY} and ann.percona_ef_construction "
                         f"are both set; keep one")
    return {int(m): int(e) for m, e in raw.items()}


def efc_for_m(profile: Dict[str, Any], m: int) -> Optional[int]:
    """Percona's ef_construction at `m` from the per-M map, or None if the
    profile has no map. A map that lacks `m` is refused, never defaulted:
    falling back to 200 for one M would put a 200 build inside a profile
    whose every curve claims to be matched."""
    by_m = efc_by_m(profile)
    if by_m is None:
        return None
    if int(m) not in by_m:
        raise ValueError(f"ann.{BY_M_KEY} has no entry for M={int(m)} "
                         f"(it names M {sorted(by_m)}); every M the profile "
                         f"builds Percona at needs one")
    return by_m[int(m)]


def missing_ms(profile: Dict[str, Any], m_values: Iterable[int]) -> list:
    """The M values in `m_values` the per-M map does not cover."""
    by_m = efc_by_m(profile) or {}
    return [int(m) for m in m_values if int(m) not in by_m]
