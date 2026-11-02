"""The engine's own account of what its threads waited on, per window.

WHY THIS EXISTS. The published six-engine post attributed the MySQL-family
concurrency ceiling first to the server and then to the client, and had no
server-side evidence either way. Ops measurement version 9 added the server's
CPU and cgroup pressure per window (`report/server_windows.py`), and its own
guard says outright that it cannot tell a server that serializes from one that
is saturated: both read "server CPU below its cpuset, client not bound". What
CAN tell them apart is the engine's wait accounting -- a mutex or rwlock that
dominates the window's wait time is serialization, a window with no dominant
wait and the CPU pinned is saturation. This module turns each engine's
cumulative counters into one small, comparable dict per window.

THE SHAPE, the same for every engine so the report has one reader:

    {"source": <what was read>, "ranked_by": "wait_s" | "count",
     "events": [{"event", "count", "wait_s"}, ...],   # top TOP_EVENTS
     "events_total": <events with a non-zero delta>,
     "total_wait_s": <sum over ALL of them, not only the top ten>, ...}

plus engine-specific evidence beside it (InnoDB mutexes, Valkey's latency
monitor, mongod's ticket gauges). `wait_s` is None where the engine counts
occurrences without timing them (pgvector's sampled `pg_stat_activity`), and
the events are then ranked by count, which `ranked_by` says.

ABSENT IS NOT ZERO. A driver returns `(None, reason)` when it cannot measure --
performance_schema off, no timed wait instrument enabled, the side connection
refused -- and the record carries `server_waits: null` beside
`server_waits_reason`. A `{"events": []}` is reserved for "measured, and
nothing waited", which is a finding; the two must never be one value, for the
reason `EngineDriver.status_counters` omits rather than zeroes.

Pure functions only. No connector is imported here, so the arithmetic is
testable on a bare host.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: How many events a record keeps. The spec's "small dict": ten is enough to
#: show a dominant wait and its nearest rivals, and few enough that a
#: concurrency record does not grow by a performance_schema dump per window.
TOP_EVENTS = 10

#: The reason a driver that overrides nothing gives. Spelled once so the
#: report and the tests compare against the same text.
NOT_IMPLEMENTED = "not implemented for this engine's driver"

#: (data, reason): exactly one of the two is None.
WaitResult = Tuple[Optional[Dict[str, Any]], Optional[str]]


def event(count: Optional[float], wait_s: Optional[float] = None
          ) -> Dict[str, Optional[float]]:
    """One cumulative counter pair, as every snapshot stores it."""
    return {"count": count, "wait_s": wait_s}


def counter_delta(before: Dict[str, Any], after: Dict[str, Any],
                  source: str, top: int = TOP_EVENTS) -> WaitResult:
    """The window's events from two cumulative snapshots' `events` maps.

    An event present only in `after` started counting inside the window and
    its whole value is the delta. A counter that went BACKWARDS is not a
    negative wait: it is a server that restarted, or a table someone
    truncated, between the two reads, and the whole window is refused with
    that reason rather than published with the event silently dropped.
    """
    old = (before or {}).get("events") or {}
    new = (after or {}).get("events") or {}
    rows: List[Dict[str, Any]] = []
    for name, now in new.items():
        then = old.get(name) or {}
        count = _minus(now.get("count"), then.get("count"))
        wait = _minus(now.get("wait_s"), then.get("wait_s"))
        if (count is not None and count < 0) or (wait is not None
                                                 and wait < 0):
            return None, (f"counter for {name!r} went backwards between the "
                          f"two snapshots; the server restarted or its "
                          f"counters were reset inside the window")
        if not count and not wait:
            continue
        rows.append({"event": name,
                     "count": None if count is None else int(round(count)),
                     "wait_s": None if wait is None else round(wait, 6)})
    return summarise(rows, source, top), None


def summarise(rows: List[Dict[str, Any]], source: str,
              top: int = TOP_EVENTS) -> Dict[str, Any]:
    """Rank, total and truncate a list of per-event deltas."""
    timed = any(r.get("wait_s") is not None for r in rows)
    key = "wait_s" if timed else "count"
    ranked = sorted(rows, key=lambda r: (-(r.get(key) or 0), r["event"]))
    total = (round(sum(r.get("wait_s") or 0 for r in rows), 6)
             if timed else None)
    return {
        "source": source,
        "ranked_by": key,
        "events": ranked[:top],
        "events_total": len(rows),
        "total_wait_s": total,
    }


def top_event(waits: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The largest event of a summarised window, or None."""
    events = (waits or {}).get("events") or []
    return events[0] if events else None


def _minus(a: Any, b: Any) -> Optional[float]:
    if a is None:
        return None
    try:
        return float(a) - float(b or 0)
    except (TypeError, ValueError):
        return None


__all__ = ["NOT_IMPLEMENTED", "TOP_EVENTS", "WaitResult", "counter_delta",
           "event", "summarise", "top_event"]
