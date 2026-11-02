"""What the SERVER was doing during each concurrency window.

The concurrency table has carried the client container's CPU since the
client-bound flag existed, and nothing about the server's: a curve that
flattens at 16 clients could be an engine out of cores, an engine stalled on
IO, or an engine with cores to spare and a lock inside it, and the record
could not say which. Ops measurement version 9 gave the server sampler
(`orchestrator/docker_ctl.py:MemorySampler`) the cgroup's `cpu.pressure` and
`io.pressure` and gave every concurrency record its median window's
wall-clock ends (`extra.window_started_at` / `window_ended_at`). This module
joins the two by timestamp.

Every figure is a DIFFERENCE of two cumulative counters over the samples
inside the window, never a mean of the `avg10` moving averages: avg10 is a
ten-second exponential average and a 20 s window would be described half by
the window before it. `cpu_seconds` and the pressure `total` (microseconds
stalled) are cumulative, so the first and last sample inside the window
bound exactly the work and the stall that happened between them.

`None` wherever the inputs are missing -- a pre-version-9 record has no
window ends, a pre-version-9 sample has no pressure, a kernel without PSI
records None per sample (`MemorySampler` never writes 0 for "absent") -- and
the report prints a dash. A zero here would be the claim "the server never
stalled", which nobody measured.

Imports nothing from `orchestrator`; the container that runs `report/` does
not mount it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: The pressure fields this module differences, as the sampler writes them.
CPU_SOME_TOTAL = "cpu_pressure_some_total"
IO_SOME_TOTAL = "io_pressure_some_total"


def window_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """One concurrency record's identity for the join.

    The window's start instant is in it, so two records of one unit at one
    client count (a resumed run that appended) cannot share an entry.
    """
    extra = record.get("extra") or {}
    return (record.get("engine"), record.get("dataset"),
            record.get("resource_pass"), record.get("m"),
            record.get("build_mode"), record.get("storage_engine"),
            record.get("clients"), extra.get("window_started_at"))


def _delta(first: Dict[str, Any], last: Dict[str, Any],
           key: str) -> Optional[float]:
    a, b = first.get(key), last.get(key)
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def window_figures(samples: List[Dict[str, Any]], started: Optional[float],
                   ended: Optional[float],
                   server_cpus: Optional[int] = None
                   ) -> Optional[Dict[str, Optional[float]]]:
    """Server CPU and pressure over `[started, ended]`, or None.

    `cpu_cores` is CPU-seconds per wall second; `cpu_util` divides it by the
    server's cpuset size when the manifest says what that was.
    `cpu_pressure` and `io_pressure` are the share of the span in which at
    least one runnable task was stalled ("some"), from the cumulative totals.
    Fewer than two samples inside the window is None: one sample bounds no
    interval.
    """
    if started is None or ended is None or ended <= started:
        return None
    inside = sorted((s for s in samples
                     if s.get("t") is not None
                     and started <= float(s["t"]) <= ended),
                    key=lambda s: float(s["t"]))
    if len(inside) < 2:
        return None
    first, last = inside[0], inside[-1]
    span = float(last["t"]) - float(first["t"])
    if span <= 0:
        return None
    cpu = _delta(first, last, "cpu_seconds")
    cores = cpu / span if cpu is not None else None
    util = (cores / server_cpus if cores is not None and server_cpus
            else None)
    stall_cpu = _delta(first, last, CPU_SOME_TOTAL)
    stall_io = _delta(first, last, IO_SOME_TOTAL)
    return {
        "cpu_cores": cores,
        "cpu_util": util,
        "cpu_pressure": (stall_cpu / 1e6 / span
                         if stall_cpu is not None else None),
        "io_pressure": (stall_io / 1e6 / span
                        if stall_io is not None else None),
        "samples": len(inside),
    }


def server_cpu_count(manifest: Optional[Dict[str, Any]], engine: Any,
                     resource_pass: Any) -> Optional[int]:
    """The server cpuset size `engines_resolved` recorded, or None."""
    block = (manifest or {}).get("engines_resolved")
    if not isinstance(block, dict):
        return None
    entry = (block.get(engine) or {}).get(resource_pass or "") or {}
    count = (entry.get("resolved") or {}).get("server_cpu_count")
    return int(count) if count else None


def server_windows(records: List[Dict[str, Any]],
                   memory: Optional[Dict[str, List[Dict[str, Any]]]],
                   manifest: Optional[Dict[str, Any]] = None
                   ) -> Dict[Tuple[Any, ...], Dict[str, Optional[float]]]:
    """`window_key(record) -> window_figures(...)` for every concurrency
    record whose unit has a memory series and whose window has ends."""
    from report.loaders import memory_stem

    out: Dict[Tuple[Any, ...], Dict[str, Optional[float]]] = {}
    if not memory:
        return out
    for record in records:
        if record.get("phase") != "concurrency":
            continue
        extra = record.get("extra") or {}
        samples = memory.get(memory_stem(record))
        if not samples:
            continue
        figures = window_figures(
            samples, extra.get("window_started_at"),
            extra.get("window_ended_at"),
            server_cpu_count(manifest, record.get("engine"),
                             record.get("resource_pass")))
        if figures is not None:
            out[window_key(record)] = figures
    return out


__all__ = ["CPU_SOME_TOTAL", "IO_SOME_TOTAL", "server_cpu_count",
           "server_windows", "window_figures", "window_key"]
