"""`perf record -g` of the server over one designated concurrency window.

`ops.profile_window_clients` (default null, off) names a client count; for
each ops unit the harness runs one EXTRA window at that count after the
measured repeats -- never a candidate for the published median -- and the
orchestrator profiles the server over it. Wait counters say WHAT the server waited on; a CPU profile
says where the time went when it was not waiting -- the two halves of telling
an engine that serializes from one that is saturated.

**Host-side, against the server container's cgroup.** perf runs on the host
because that is where the kernel's sampling interface is; it targets the
container's cgroup (`perf record -a -G <cgroup>`) so every server process is
covered -- pgvector's backends are separate processes, which `-p <pid>` of the
postmaster would miss -- and falls back to `-p <pid>` only when the cgroup
cannot be read, saying so in `method`.

**No new host dependency.** perf is used if the host already has it and
permits it; otherwise the reason (`perf` not installed, `perf_event_paranoid`,
missing CAP_PERFMON, whatever perf printed) is written to `profile.json` and
the run continues. Folding the stacks is done here in Python. A flame-graph
SVG is rendered only if a renderer exists inside the engine's bench image,
never by installing one.

**Timing comes from the harness.** It prints `BEGIN_MARK` before the pool
starts and waits for this module's ack file, and prints `END_MARK` after the
pool stops -- see `harness/workloads/profile_window.py` for why that keeps
every handshake step outside the timed span.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional

from harness.workloads.profile_window import BEGIN_MARK, END_MARK

from . import docker_ctl

#: perf's sampling rate. 99 Hz rather than a round number so the sampler does
#: not run in lockstep with a periodic timer in the server.
PERF_FREQUENCY_HZ = 99

#: How long perf may take to exit after SIGINT before it is killed. It writes
#: its buffers and build-ids on the way out, which on a large data file takes
#: seconds.
PERF_STOP_TIMEOUT_S = 60

#: How long a freshly started perf is watched for an immediate exit, which is
#: how a permission refusal shows up.
PERF_START_GRACE_S = 0.5

#: Renderers looked for INSIDE the bench image, in order.
SVG_RENDERERS = ("inferno-flamegraph", "flamegraph.pl")

_MARK_RE = re.compile(r"clients=(\d+)\s+token=([0-9a-f]+)")


def perf_unavailable_reason(which: Callable[[str], Optional[str]] = shutil.which
                            ) -> Optional[str]:
    """Why perf certainly cannot run here, or None if it is worth trying.

    Only the certain refusal is decided in advance: no binary. A
    `perf_event_paranoid` above 0 is NOT decided here, because a perf binary
    carrying CAP_PERFMON (or a root orchestrator) works regardless; it is
    reported beside whatever perf itself says if it then refuses.
    """
    if which("perf") is None:
        return "perf is not installed on the host"
    return None


def paranoid_level(path: str = "/proc/sys/kernel/perf_event_paranoid"
                   ) -> Optional[int]:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def container_target(container: str) -> Dict[str, Any]:
    """`{"method", "args", "target"}` for perf, or `{"reason"}`."""
    try:
        proc = docker_ctl._run(["docker", "inspect", "-f", "{{.State.Pid}}",
                                container], check=False, timeout=30)
        pid = int((proc.stdout or "").strip() or 0)
    except Exception as exc:
        return {"reason": f"docker inspect of {container} failed: {exc}"}
    if pid <= 0:
        return {"reason": f"{container} has no running process to profile"}
    cgroup = _cgroup_of(pid)
    if cgroup:
        return {"method": "cgroup", "target": cgroup,
                "args": ["-a", "-e", "cpu-clock", "-G", cgroup]}
    return {"method": "pid", "target": str(pid),
            "args": ["-e", "cpu-clock", "-p", str(pid)]}


def _cgroup_of(pid: int) -> Optional[str]:
    """The unified (v2) cgroup path of `pid`, relative to the mount."""
    try:
        with open(f"/proc/{pid}/cgroup") as fh:
            for line in fh:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0" and parts[2] != "/":
                    return parts[2].lstrip("/")
    except OSError:
        return None
    return None


def fold_perf_script(lines: Iterable[str]) -> Counter:
    """`perf script` output as folded stacks (`comm;outer;...;leaf` -> n).

    The stackcollapse-perf format, in a dozen lines, so producing it needs
    nothing on the host but perf itself.
    """
    folded: Counter = Counter()
    comm: Optional[str] = None
    frames: List[str] = []

    def flush() -> None:
        if comm is not None:
            folded[";".join([comm] + list(reversed(frames)))] += 1

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            flush()
            comm, frames = None, []
        elif line[0] in " \t":
            if comm is None:
                continue
            parts = line.strip().split(None, 1)
            symbol = parts[1] if len(parts) > 1 else parts[0]
            symbol = re.sub(r"\s+\([^)]*\)$", "", symbol)
            symbol = re.sub(r"\+0x[0-9a-f]+$", "", symbol)
            frames.append(symbol.replace(";", ":") or "[unknown]")
        else:
            flush()
            comm, frames = line.split()[0], []
    flush()
    return folded


class PerfWindow:
    """One perf process over one window, and the files it leaves."""

    def __init__(self, container: str, out_dir: str, meta: Dict[str, Any],
                 bench_image: Optional[str] = None):
        self.container = container
        self.out_dir = out_dir
        self.bench_image = bench_image
        self.meta: Dict[str, Any] = dict(meta)
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_path = os.path.join(out_dir, "perf.stderr")

    @property
    def perf_data(self) -> str:
        return os.path.join(self.out_dir, "perf.data")

    def start(self) -> bool:
        os.makedirs(self.out_dir, exist_ok=True)
        self.meta["perf_started_at"] = round(time.time(), 3)
        reason = perf_unavailable_reason()
        if reason is None:
            target = container_target(self.container)
            reason = target.get("reason")
        if reason is not None:
            return self._unavailable(reason)
        self.meta.update(method=target["method"], target=target["target"])
        argv = ["perf", "record", "-g", "-F", str(PERF_FREQUENCY_HZ),
                *target["args"], "-o", self.perf_data]
        try:
            with open(self._stderr_path, "w") as err:
                self._proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                              stderr=err)
        except OSError as exc:
            return self._unavailable(f"perf could not be started: {exc}")
        time.sleep(PERF_START_GRACE_S)
        code = self._proc.poll()
        if code is not None:
            return self._unavailable(self._refusal(code))
        self.meta["status"] = "recording"
        return True

    def stop(self) -> Dict[str, Any]:
        """Stop perf, fold its stacks, write `profile.json`. Never raises."""
        self.meta["perf_ended_at"] = round(time.time(), 3)
        if self._proc is None:
            return self._write()
        try:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=PERF_STOP_TIMEOUT_S)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        if not os.path.exists(self.perf_data):
            self.meta.update(status="failed",
                             reason="perf exited without writing perf.data")
            return self._write()
        self.meta["status"] = "recorded"
        self.meta["files"] = {"perf_data": "perf.data"}
        self._fold()
        return self._write()

    # -- internals ----------------------------------------------------------

    def _fold(self) -> None:
        try:
            proc = subprocess.run(["perf", "script", "-i", self.perf_data],
                                  capture_output=True, text=True, timeout=600,
                                  check=False)
        except Exception as exc:
            self.meta["folded_reason"] = f"perf script failed: {exc}"
            return
        folded = fold_perf_script((proc.stdout or "").splitlines())
        if not folded:
            self.meta["folded_reason"] = ("perf script produced no stacks: "
                                          + (proc.stderr or "").strip()[:300])
            return
        path = os.path.join(self.out_dir, "folded.txt")
        with open(path, "w") as fh:
            for stack, count in folded.most_common():
                fh.write(f"{stack} {count}\n")
        self.meta["files"]["folded"] = "folded.txt"
        self.meta["samples"] = sum(folded.values())
        self._render_svg(path)

    def _render_svg(self, folded_path: str) -> None:
        renderer = _image_renderer(self.bench_image)
        if renderer is None:
            self.meta["svg_reason"] = (
                "no flame-graph renderer (" + ", ".join(SVG_RENDERERS)
                + ") in the bench image; render folded.txt elsewhere")
            return
        try:
            with open(folded_path) as src:
                proc = subprocess.run(
                    ["docker", "run", "--rm", "-i", "--network", "none",
                     "--entrypoint", renderer, str(self.bench_image)],
                    stdin=src, capture_output=True, text=True, timeout=600,
                    check=False)
        except Exception as exc:
            self.meta["svg_reason"] = f"{renderer} failed: {exc}"
            return
        if proc.returncode != 0 or not proc.stdout:
            self.meta["svg_reason"] = f"{renderer} exited {proc.returncode}"
            return
        with open(os.path.join(self.out_dir, "flamegraph.svg"), "w") as fh:
            fh.write(proc.stdout)
        self.meta["files"]["svg"] = "flamegraph.svg"

    def _refusal(self, code: int) -> str:
        try:
            with open(self._stderr_path) as fh:
                said = fh.read().strip().splitlines()
        except OSError:
            said = []
        level = paranoid_level()
        text = f"perf record exited {code} at start"
        if said:
            text += ": " + " ".join(said[:3])[:400]
        if level is not None and level > 0:
            text += (f" (perf_event_paranoid is {level}; system-wide cgroup "
                     f"profiling needs <= 0, root, or CAP_PERFMON on perf)")
        return text

    def _unavailable(self, reason: str) -> bool:
        print(f"[ops] perf profile unavailable: {reason}; continuing "
              f"unprofiled", flush=True)
        self.meta.update(status="unavailable", reason=reason)
        self._proc = None
        return False

    def _write(self) -> Dict[str, Any]:
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "profile.json"), "w") as fh:
                json.dump(self.meta, fh, indent=2, sort_keys=True)
        except OSError as exc:
            print(f"[ops] could not write profile.json: {exc}", flush=True)
        return self.meta


def _image_renderer(image: Optional[str]) -> Optional[str]:
    if not image:
        return None
    script = " || ".join(f"command -v {name}" for name in SVG_RENDERERS)
    try:
        proc = docker_ctl._run(["docker", "run", "--rm", "--network", "none",
                                "--entrypoint", "sh", image, "-c", script],
                               check=False, timeout=120)
    except Exception:
        return None
    found = (proc.stdout or "").strip().splitlines()
    return os.path.basename(found[0]) if proc.returncode == 0 and found \
        else None


class ProfileWatcher:
    """The `line_filter` that turns the harness's marks into a perf window.

    Passes every line through unchanged. On a begin mark it starts perf (or
    records why it cannot) and writes the ack the harness is waiting for --
    in both cases, so a host without perf costs the harness no timeout. On
    the end mark, or at end of stream, it stops perf and writes the files.
    """

    def __init__(self, container: str, profiles_dir: str, stem: str,
                 ack_dir: str, meta: Dict[str, Any],
                 bench_image: Optional[str] = None,
                 window_factory: Callable[..., PerfWindow] = PerfWindow):
        self.container = container
        self.profiles_dir = profiles_dir
        self.stem = stem
        self.ack_dir = ack_dir
        self.meta = dict(meta)
        self.bench_image = bench_image
        self._factory = window_factory
        self._active: Optional[PerfWindow] = None
        self.results: List[Dict[str, Any]] = []

    def __call__(self, line: Optional[str]) -> List[str]:
        if line is None:
            self._finish()
            return []
        if line.startswith(BEGIN_MARK):
            self._begin(line)
        elif line.startswith(END_MARK):
            self._finish()
        return [line]

    def _begin(self, line: str) -> None:
        self._finish()
        match = _MARK_RE.search(line)
        if match is None:
            return
        clients, token = int(match.group(1)), match.group(2)
        out_dir = os.path.join(self.profiles_dir, f"{self.stem}-c{clients}")
        window = self._factory(self.container, out_dir,
                               {**self.meta, "clients": clients,
                                "token": token}, self.bench_image)
        try:
            started = window.start()
        except Exception as exc:
            window.meta.update(status="unavailable", reason=str(exc))
            started = False
        self._active = window
        status = "recording"
        if not started:
            status = (f"unavailable: "
                      f"{window.meta.get('reason') or 'perf did not start'}")
            self.results.append(window.stop())
            self._active = None
        self._ack(token, status)

    def _finish(self) -> None:
        if self._active is not None:
            self.results.append(self._active.stop())
            self._active = None

    def _ack(self, token: str, status: str) -> None:
        """Release the harness, telling it whether perf is recording, so the
        record can say when its profile window has no profile."""
        try:
            os.makedirs(self.ack_dir, exist_ok=True)
            with open(os.path.join(self.ack_dir, f"ack-{token}"), "w") as fh:
                fh.write(status + "\n")
        except OSError as exc:
            print(f"[ops] could not acknowledge the profiled window: {exc}",
                  flush=True)


__all__ = ["PerfWindow", "ProfileWatcher", "container_target",
           "fold_perf_script", "perf_unavailable_reason"]
