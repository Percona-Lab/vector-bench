"""What limited throughput, per concurrency point: the measured components.

One row per concurrency record, beside the scaling table: the client
container's CPU, the server's CPU and CPU pressure over the same window
(`report/server_windows.py`, ops version 9), the server's largest wait from
its own counters (`extra.server_waits`, `harness/workloads/server_waits.py`),
and a link to a perf profile when one was taken for that point.

**No verdict.** The published six-engine post named the MySQL-family ceiling
first as the server's and then as the client's, from evidence that could not
tell the two apart. This table therefore names only the LARGEST MEASURED
component among the three that share a unit -- the share of a CPU budget used
by the client, by the server, and the share of the window the server spent
stalled waiting for CPU -- and leaves the wait column to be read beside it.
A wait profile is seconds summed across threads and has no common unit with
those three, so it is shown, not ranked.

Imports nothing from `orchestrator`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

#: Profile directories are `<memory stem>-c<clients>` under `profiles/`.
PROFILES_DIR = "profiles"


def profile_index(run_dir: str, out_dir: Optional[str] = None
                  ) -> Dict[str, Dict[str, Any]]:
    """`{<stem>-c<clients>: profile.json + href}` for every profile written.

    `href` is the folded stacks (or the SVG when one was rendered) relative
    to the report's own directory, so the link works wherever the report is
    written. A profile that could not be taken is indexed too: its reason is
    what the report prints.
    """
    root = os.path.join(run_dir, PROFILES_DIR)
    out: Dict[str, Dict[str, Any]] = {}
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        path = os.path.join(root, name, "profile.json")
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        files = meta.get("files") or {}
        target = files.get("svg") or files.get("folded") or files.get(
            "perf_data")
        if target:
            absolute = os.path.join(root, name, target)
            meta["href"] = (os.path.relpath(absolute, out_dir) if out_dir
                            else absolute)
        out[name] = meta
    return out


def profile_key(record: Dict[str, Any]) -> str:
    from report.loaders import memory_stem
    return f"{memory_stem(record)}-c{record.get('clients')}"


def top_wait_cell(extra: Dict[str, Any]) -> str:
    """The largest server wait, or why there is none."""
    waits = extra.get("server_waits")
    if waits is None:
        reason = extra.get("server_waits_reason")
        return f"— ({reason})" if reason else "— (not recorded)"
    events = waits.get("events") or []
    if not events:
        return "none measured"
    top = events[0]
    name = str(top.get("event"))
    if waits.get("ranked_by") == "wait_s" and top.get("wait_s") is not None:
        total = waits.get("total_wait_s") or 0
        share = (f", {top['wait_s'] / total:.0%} of waited time"
                 if total else "")
        return f"{name}: {top['wait_s']:,.3f} s ({top.get('count', 0):,}x{share})"
    samples = waits.get("samples")
    per = f" of {samples} samples" if samples else ""
    return f"{name}: {top.get('count', 0):,}{per}"


def profile_cell(record: Dict[str, Any],
                 profiles: Optional[Dict[str, Dict[str, Any]]]) -> str:
    meta = (profiles or {}).get(profile_key(record))
    if not meta:
        return "—"
    if meta.get("href"):
        return f"[profile]({meta['href']})"
    return f"— ({meta.get('reason') or meta.get('status') or 'no file'})"


def largest_component(client_util: Optional[float],
                      server_util: Optional[float],
                      cpu_pressure: Optional[float]) -> str:
    """The name and value of the largest measured share, or a dash.

    Only components that were measured compete; a missing one is not a zero.
    """
    shares: List[Tuple[str, float]] = [
        (label, float(value)) for label, value in (
            ("client CPU", client_util), ("server CPU", server_util),
            ("server CPU pressure", cpu_pressure))
        if value is not None]
    if not shares:
        return "—"
    label, value = max(shares, key=lambda kv: kv[1])
    return f"{label} ({value:.0%})"


__all__ = ["PROFILES_DIR", "largest_component", "profile_cell",
           "profile_index", "profile_key", "top_wait_cell"]
