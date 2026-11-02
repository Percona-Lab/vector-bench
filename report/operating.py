"""Joining throughput figures to the calibration point that produced them.

A concurrency or churn QPS is a statement about an engine at one ef_search,
and the ef_search is a statement about recall: the calibrated policy chose it
because the sweep measured the target recall there. Printed without that
recall, two throughput columns look comparable when one engine is answering
at 0.95 and another, which never reached the target, at 0.90 -- a cheaper
query, and a higher number for that reason alone. `loaders.operating_points`
already holds the chosen ef, the recall it bought and the status; this module
puts them beside every throughput figure and keeps engines that missed the
target out of any ranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: The two sweep outcomes that leave an engine off the recall target. Spelled
#: here rather than imported from `harness.workloads.calibrate`, which pulls
#: numpy and the drivers in for two strings -- the same reason `render.py`
#: declares `SCALING_CONCURRENCY_NOT_ACHIEVED` itself.
STATUS_TARGET_UNREACHED = "target_unreached"
STATUS_FLOOR_ABOVE_TARGET = "floor_above_target"
UNRANKED_STATUSES = (STATUS_TARGET_UNREACHED, STATUS_FLOOR_ABOVE_TARGET)

_UNIT_FIELDS = ("engine", "dataset", "resource_pass", "m", "build_mode",
                "storage_engine")


def _unit(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(row.get(field) for field in _UNIT_FIELDS)


class OperatingPointIndex:
    """`operating_points` rows, findable from a record or a derived row.

    By `operating_point_key` first, which is the unit's own definition. By
    the six named unit columns second, for rows that are not records -- the
    churn-under-writes rows are built from records but carry only those
    columns -- and only when exactly one point has them: two points on one
    loose key are two sweeps, and naming either would be a guess.
    """

    def __init__(self, points: List[Dict[str, Any]]) -> None:
        self._by_key = {p.get("key"): p for p in points}
        self._by_unit: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for point in points:
            self._by_unit.setdefault(_unit(point), []).append(point)

    def find(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row.get("phase"):
            from report.loaders import operating_point_key
            point = self._by_key.get(operating_point_key(row))
            if point is not None:
                return point
        candidates = self._by_unit.get(_unit(row), [])
        return candidates[0] if len(candidates) == 1 else None


def status_of(row: Dict[str, Any],
              point: Optional[Dict[str, Any]]) -> Optional[str]:
    """The calibration status a throughput figure was measured under."""
    if point is not None and point.get("status"):
        return point["status"]
    return (row.get("extra") or {}).get("calibration_status")


#: The first ops measurement version that calibrates ef_search and stamps
#: `extra.ef_mode` (`harness/metrics/records.py`: "VERSION 4 chose ef_search
#: instead of declaring it"). Every earlier record ran at the profile's
#: declared `ops.ef_search` -- 100 on every engine -- and says nothing.
FIRST_EF_MODE_VERSION = 4


def is_fixed_ef(record: Dict[str, Any]) -> bool:
    """Whether a throughput record was measured at a declared, fixed ef.

    `extra.ef_mode == "fixed"`, or no `ef_mode` at all on a record from
    before version 4. The second clause is the 25 Aug run: its records carry
    neither key, so they are implicit version 1, they ran at ef_search 100,
    and a report that read "no ef_mode" as "nothing to say" presented
    engines at recall 0.98 and 0.80 as comparable throughput. A version that
    is present and not an integer is NOT assumed fixed: the mixed-version
    guard refuses such a record anyway, and guessing here would not help.
    """
    from report.measurement_versions import ops_measurement_version

    extra = record.get("extra") or {}
    mode = extra.get("ef_mode")
    if mode is not None:
        return mode == "fixed"
    version = ops_measurement_version(record)
    return (isinstance(version, int) and not isinstance(version, bool)
            and version < FIRST_EF_MODE_VERSION)


def concurrency_peaks(records: List[Dict[str, Any]],
                      points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The peak throughput of each concurrency unit, and the client count it
    scales to -- the cross-engine headline, with two kinds of point kept out.

    A **client-bound** point (`extra.client_bound`, the client container's
    CPUs above 80%) is the clients' ceiling, not the engine's, and stays in
    its own table and hollow in the chart; it is never the peak.

    An engine whose calibration ended **target_unreached** or
    **floor_above_target** is not ranked: its throughput was taken at a
    recall the others were not held to, so the row carries the status where
    the number would be. Sorting it among the rest is the same mislabelling
    as the 25 Aug floor table, one section later.
    """
    index = OperatingPointIndex(points)
    units: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for record in records:
        if record.get("phase") == "concurrency" and record.get("qps"):
            units.setdefault(_unit(record), []).append(record)

    peaks: List[Dict[str, Any]] = []
    for unit, rows in units.items():
        sample = rows[0]
        point = index.find(sample)
        status = status_of(sample, point)
        fixed = point is None and is_fixed_ef(sample)
        eligible = [r for r in rows
                    if not (r.get("extra") or {}).get("client_bound")]
        best = max(eligible, key=lambda r: r["qps"]) if eligible else None
        ranked = status not in UNRANKED_STATUSES and best is not None
        peaks.append({
            **dict(zip(_UNIT_FIELDS, unit)),
            "ranked": ranked,
            "status": status,
            "peak_qps": best["qps"] if ranked else None,
            "peak_clients": best.get("clients") if ranked else None,
            "scaling_efficiency": ((best.get("extra") or {}).get(
                "scaling_efficiency") if ranked else None),
            "client_bound_excluded": len(rows) - len(eligible),
            # The CALIBRATED ef, or the declared one flagged as fixed: printing
            # a fixed ef under a calibration heading would claim equal recall.
            "ef_search": ((point or {}).get("ef_search") if not fixed
                          else sample.get("ef_search")),
            "fixed_ef": fixed,
            "calibration_recall": (point or {}).get("recall_at_k"),
        })
    peaks.sort(key=lambda p: (not p["ranked"], -(p["peak_qps"] or 0),
                              str(p["engine"]), str(p["dataset"]),
                              str(p["resource_pass"]), p["m"] or 0))
    return peaks
