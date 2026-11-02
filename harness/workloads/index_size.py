"""What an `index_bytes` figure is a measurement OF (ops measurement v9).

`full-20260825` read Percona Search's index at 15.1 GiB after the load and
7.2 GiB at the churn baseline -- the same index, before and after Lucene
merged its segments -- and both were published as "index size" with nothing
to say when each was read or what the bytes contained. MHNSW's companion
table holds int16 copies of the vectors, pgvector's index tuples hold the
float32 ones, and a size comparison that does not say so compares formats.

So every record that carries `index_bytes` also carries:

* `extra.size_read_at` -- `after_settle` on the `index_build` record,
  `churn_baseline`, `after_churn_step`, `after_maintenance` on churn records;
* `extra.index_vector_format` / `extra.index_includes_vectors` from the
  driver's `capabilities()`, `"unknown"` / None where this repository has not
  verified the engine's on-disk format from its source;
* `extra.index_bytes_by_kind` where the driver can break the figure down
  (mongot: bytes per Lucene file extension).

`index_build` additionally carries the immediate post-build reading
(`extra.index_bytes_after_build`) beside the settled one, and how long the
settle took (`extra.index_size_settle_s`).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def index_size_shape(driver, read_at: str) -> Dict[str, Any]:
    """The provenance keys above, for one record's `extra`."""
    try:
        caps = dict(driver.capabilities() or {})
    except Exception:
        caps = {}
    out: Dict[str, Any] = {"size_read_at": read_at}
    for key in ("index_vector_format", "index_includes_vectors"):
        if key in caps:
            out[key] = caps[key]
    breakdown = getattr(driver, "index_bytes_breakdown", None)
    if callable(breakdown):
        try:
            kinds = breakdown()
        except Exception:
            kinds = None
        if kinds:
            out["index_bytes_by_kind"] = kinds
    return out


def settled_index_bytes(driver, immediate: int
                        ) -> Tuple[int, float, Dict[str, Any]]:
    """`driver.settle_index_bytes()`, defended: a settle that raises must not
    cost the build it follows, so it falls back to the immediate reading and
    says why."""
    settle = getattr(driver, "settle_index_bytes", None)
    if not callable(settle):
        return immediate, 0.0, {"method": "immediate"}
    try:
        size, seconds, detail = settle()
    except Exception as exc:
        return immediate, 0.0, {"method": "immediate",
                                "settle_failed": f"{type(exc).__name__}: "
                                                 f"{exc}"}
    return int(size), float(seconds), dict(detail or {})


__all__ = ["index_size_shape", "settled_index_bytes"]
