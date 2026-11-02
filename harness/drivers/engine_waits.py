"""Per-engine wait snapshots, written against a cursor or client handle.

One function per engine family, called by that driver's `wait_snapshot` /
`wait_delta`. Kept out of the driver files because those are already past the
repository's size ceiling, and because every function here takes its handle as
an argument, so the suite can drive each against a recording fake without a
connector installed.

Each snapshot function returns `(snapshot, None)` or `(None, reason)` and
never raises: it runs on a side connection after a window has been measured,
and a diagnostic that cost a finished measurement would be the wrong trade.
"""

from __future__ import annotations

import time
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

from .server_waits import WaitResult, counter_delta, event

# -- MySQL family ------------------------------------------------------------

PS_SOURCE = "performance_schema.events_waits_summary_global_by_event_name"

#: Timer columns are picoseconds in every MySQL-family server here.
PICOSECONDS = 1e12

#: The instruments `ops.server_wait_instruments` turns on, and whose state a
#: snapshot reports so a reader can tell "no mutex wait" from "mutex waits were
#: not instrumented". `sxlock` is InnoDB's shared-exclusive latch, which MySQL
#: 8 instruments under its own prefix rather than under rwlock.
SYNCH_PREFIXES = ("wait/synch/mutex/", "wait/synch/rwlock/",
                  "wait/synch/sxlock/")


def mysql_snapshot(cur: Any) -> WaitResult:
    """Cumulative wait counters from performance_schema, plus InnoDB mutexes.

    The `idle` event is excluded: it is the time sessions spent waiting for
    the CLIENT to send the next statement, and at 32 clients it would top
    every window while saying nothing about the server.
    """
    try:
        cur.execute("SELECT @@performance_schema")
        row = cur.fetchone()
        if not row or not _truthy(row[0]):
            return None, ("performance_schema is OFF on this server, so it "
                          "keeps no wait counters (MariaDB's default; set "
                          "ops.server_wait_instruments to turn it on)")
        cur.execute(
            "SELECT NAME FROM performance_schema.setup_instruments "
            "WHERE NAME LIKE 'wait/%' AND ENABLED = 'YES' AND TIMED = 'YES'")
        timed = [str(r[0]) for r in (cur.fetchall() or [])]
        if not timed:
            return None, ("performance_schema is ON but no wait/% instrument "
                          "is both enabled and timed")
        cur.execute(
            f"SELECT EVENT_NAME, COUNT_STAR, SUM_TIMER_WAIT FROM {PS_SOURCE} "
            f"WHERE COUNT_STAR > 0 AND EVENT_NAME <> 'idle'")
        events = {str(name): event(int(count or 0),
                                   float(timer or 0) / PICOSECONDS)
                  for name, count, timer in (cur.fetchall() or [])}
    except Exception as exc:
        return None, f"performance_schema read failed: {exc}"
    synch = sum(1 for name in timed if name.startswith(SYNCH_PREFIXES))
    return {"events": events, "synch_instruments_timed": synch,
            "innodb_mutex": _innodb_mutex(cur)}, None


def _innodb_mutex(cur: Any) -> Optional[Dict[str, int]]:
    """`SHOW ENGINE INNODB MUTEX` as {name: waits}, or None if it says nothing.

    It reports only instrumented mutexes and rwlocks, so with the default
    instrument set it usually returns no rows -- None then, not {}.
    """
    try:
        cur.execute("SHOW ENGINE INNODB MUTEX")
        rows = cur.fetchall() or []
    except Exception:
        return None
    out: Dict[str, int] = {}
    for row in rows:
        if len(row) < 3:
            continue
        status = str(row[2])
        if status.startswith("waits="):
            try:
                out[str(row[1])] = int(status.split("=", 1)[1])
            except ValueError:
                continue
    return out or None


def mysql_delta(before: Dict[str, Any], after: Dict[str, Any]) -> WaitResult:
    waits, reason = counter_delta(before, after, PS_SOURCE)
    if waits is None:
        return None, reason
    waits["synch_instruments_timed"] = after.get("synch_instruments_timed")
    old = before.get("innodb_mutex") or {}
    new = after.get("innodb_mutex") or {}
    mutex = {name: count - old.get(name, 0) for name, count in new.items()
             if count - old.get(name, 0) > 0}
    if mutex:
        waits["innodb_mutex_waits"] = dict(
            sorted(mutex.items(), key=lambda kv: -kv[1])[:10])
    return waits, None


# -- PostgreSQL --------------------------------------------------------------

PG_SOURCE = "pg_stat_activity (sampled)"

#: Active client backends other than the sampler's own. A NULL wait event on
#: an active backend means it was running on CPU, which is a finding and gets
#: its own name rather than being dropped.
PG_SAMPLE_SQL = (
    "SELECT wait_event_type, wait_event FROM pg_stat_activity "
    "WHERE pid <> pg_backend_pid() AND state = 'active' "
    "AND backend_type = 'client backend'")

ON_CPU = "CPU (no wait event)"


def postgres_sample(cur: Any) -> WaitResult:
    """One point-in-time sample, as counts per `type:event`."""
    try:
        cur.execute(PG_SAMPLE_SQL)
        rows = cur.fetchall() or []
    except Exception as exc:
        return None, f"pg_stat_activity read failed: {exc}"
    events: Dict[str, Dict[str, Any]] = {}
    for kind, name in rows:
        label = ON_CPU if kind is None else f"{kind}:{name}"
        events.setdefault(label, event(0))["count"] += 1
    return {"events": events, "samples": 1}, None


def postgres_delta(before: Dict[str, Any], after: Dict[str, Any],
                   interval_s: float) -> WaitResult:
    samples = int(after.get("samples") or 0) - int(before.get("samples") or 0)
    if samples <= 0:
        return None, ("no pg_stat_activity sample was taken: the window was "
                      f"shorter than the {interval_s:g} s sampling interval")
    waits, reason = counter_delta(before, after, PG_SOURCE)
    if waits is None:
        return None, reason
    waits["samples"] = samples
    waits["interval_s"] = interval_s
    return waits, None


# -- Valkey ------------------------------------------------------------------

VALKEY_SOURCE = ("INFO commandstats (usec is time spent executing each "
                 "command, not time spent waiting)")


def valkey_snapshot(conn: Any) -> WaitResult:
    try:
        stats = conn.info("commandstats") or {}
    except Exception as exc:
        return None, f"INFO commandstats failed: {exc}"
    events = {}
    for key, value in stats.items():
        if not isinstance(value, dict):
            continue
        name = str(key)[len("cmdstat_"):] if str(key).startswith(
            "cmdstat_") else str(key)
        events[f"cmd:{name}"] = event(int(value.get("calls") or 0),
                                      float(value.get("usec") or 0) / 1e6)
    return {"events": events, "taken_at": time.time(),
            "latency_latest": _latency_latest(conn),
            "latency_threshold_ms": _latency_threshold(conn)}, None


def _latency_latest(conn: Any) -> Optional[List[List[Any]]]:
    try:
        rows = conn.execute_command("LATENCY", "LATEST") or []
    except Exception:
        return None
    return [[_text(r[0]), int(r[1]), int(r[2]), int(r[3])]
            for r in rows if len(r) >= 4]


def _latency_threshold(conn: Any) -> Optional[int]:
    try:
        value = conn.config_get("latency-monitor-threshold") or {}
        return int(value.get("latency-monitor-threshold"))
    except Exception:
        return None


def valkey_delta(before: Dict[str, Any], after: Dict[str, Any]) -> WaitResult:
    waits, reason = counter_delta(before, after, VALKEY_SOURCE)
    if waits is None:
        return None, reason
    # LATENCY LATEST is not cumulative: each row is an event's most recent
    # spike and when it happened. Only spikes stamped inside the window
    # belong to it.
    since = int(before.get("taken_at") or 0)
    latest = after.get("latency_latest")
    if latest is None:
        waits["latency_latest_reason"] = "LATENCY LATEST was not readable"
    elif not after.get("latency_threshold_ms"):
        waits["latency_latest_reason"] = (
            "latency-monitor-threshold is 0, so the latency monitor records "
            "nothing")
    else:
        waits["latency_latest"] = [
            {"event": name, "at": at, "latest_ms": last, "max_ms": peak}
            for name, at, last, peak in latest if at >= since]
    return waits, None


# -- MongoDB / mongot --------------------------------------------------------

MONGO_SOURCE = "serverStatus locks (acquireWaitCount, timeAcquiringMicros)"

#: mongot's Prometheus endpoint, when it is enabled. The generated config
#: (`docker/mongodb/entrypoint-mongodb.sh`) writes `metrics: enabled: false`,
#: and when on it binds the server container's loopback, so from the client
#: container this is expected to be unreachable and to say so.
MONGOT_METRICS_PORT = 9946
MONGOT_METRICS_TIMEOUT_S = 1.0


def mongo_snapshot(client: Any, host: str) -> WaitResult:
    try:
        status = client.admin.command("serverStatus")
    except Exception as exc:
        return None, f"serverStatus failed: {exc}"
    events: Dict[str, Dict[str, Any]] = {}
    for resource, block in (status.get("locks") or {}).items():
        waits = (block or {}).get("acquireWaitCount") or {}
        micros = (block or {}).get("timeAcquiringMicros") or {}
        for mode in set(waits) | set(micros):
            events[f"locks.{resource}.{mode}"] = event(
                int(waits.get(mode) or 0), float(micros.get(mode) or 0) / 1e6)
    gauges = {
        "global_lock_queue": (status.get("globalLock") or {}).get(
            "currentQueue"),
        "tickets": ((status.get("queues") or {}).get("execution")
                    or (status.get("wiredTiger") or {}).get(
                        "concurrentTransactions")),
    }
    snapshot = {"events": events, "gauges": _plain(gauges)}
    snapshot.update(_mongot_metrics(host))
    return snapshot, None


def _mongot_metrics(host: str) -> Dict[str, Any]:
    url = f"http://{host}:{MONGOT_METRICS_PORT}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=MONGOT_METRICS_TIMEOUT_S) as r:
            body = r.read(1 << 20).decode("utf-8", "replace")
    except Exception as exc:
        return {"mongot_metrics_reason": (
            f"mongot metrics endpoint {url} not reachable ({exc}); the "
            f"generated mongot config disables metrics and binds loopback")}
    return {"mongot_metrics": _prometheus_counters(body)}


def _prometheus_counters(body: str) -> Dict[str, float]:
    """The `*_total` / `*_seconds_sum` / `*_count` series, which difference."""
    out: Dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name.split("{")[0].endswith(("_total", "_sum", "_count")):
            continue
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def mongo_delta(before: Dict[str, Any], after: Dict[str, Any]) -> WaitResult:
    waits, reason = counter_delta(before, after, MONGO_SOURCE)
    if waits is None:
        return None, reason
    waits["gauges_after"] = after.get("gauges")
    if "mongot_metrics" in after and "mongot_metrics" in before:
        old, new = before["mongot_metrics"], after["mongot_metrics"]
        moved = {k: v - old.get(k, 0.0) for k, v in new.items()
                 if v - old.get(k, 0.0) > 0}
        waits["mongot_metrics"] = dict(
            sorted(moved.items(), key=lambda kv: -kv[1])[:10])
    else:
        waits["mongot_metrics_reason"] = (
            after.get("mongot_metrics_reason")
            or before.get("mongot_metrics_reason")
            or "mongot metrics were not read at both ends of the window")
    return waits, None


# -- helpers -----------------------------------------------------------------

def _truthy(value: Any) -> bool:
    return str(value).strip().upper() in ("1", "ON", "TRUE", "YES")


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _plain(value: Any) -> Any:
    """BSON scalars (Int64 and friends) as JSON-serialisable Python."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    try:
        return int(value) if float(value).is_integer() else float(value)
    except (TypeError, ValueError):
        return str(value)


__all__: Iterable[str] = (
    "MONGO_SOURCE", "ON_CPU", "PG_SOURCE", "PS_SOURCE", "VALKEY_SOURCE",
    "mongo_delta", "mongo_snapshot", "mysql_delta", "mysql_snapshot",
    "postgres_delta", "postgres_sample", "valkey_delta", "valkey_snapshot")
