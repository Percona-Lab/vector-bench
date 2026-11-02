"""Docker container, network and volume management for the orchestrator.

Uses the docker CLI rather than the Python SDK so the orchestrator's only
dependencies are python3 and pyyaml. That matters because the orchestrator runs
on the host, and a benchmark framework that requires pip installs on the machine
under test is a framework that changes the machine under test.

Everything created here is tagged with the run id and torn down by
`cleanup_run`, so an interrupted run leaves nothing behind that would perturb
the next one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

LABEL = "vector-bench"


class DockerError(RuntimeError):
    pass


def _run(args: Sequence[str], check: bool = True, timeout: int = 120,
         capture: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(args),
        capture_output=capture, text=True, timeout=timeout, check=False,
    )
    if check and proc.returncode != 0:
        raise DockerError(
            f"command failed ({proc.returncode}): {' '.join(shlex.quote(a) for a in args)}\n"
            f"stdout: {(proc.stdout or '').strip()}\n"
            f"stderr: {(proc.stderr or '').strip()}"
        )
    return proc


def docker_available() -> bool:
    try:
        _run(["docker", "info"], timeout=30)
        return True
    except (DockerError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(image: str) -> bool:
    try:
        _run(["docker", "image", "inspect", image], timeout=30)
        return True
    except DockerError:
        return False


def image_id(image: str) -> str:
    try:
        proc = _run(["docker", "image", "inspect", "--format", "{{.Id}}", image], timeout=30)
        return proc.stdout.strip()
    except DockerError:
        return "unknown"


# ---------------------------------------------------------------------------
# Networks and volumes
# ---------------------------------------------------------------------------

def create_network(name: str, internal: bool = True) -> str:
    if _network_exists(name):
        return name
    args = ["docker", "network", "create", "--label", f"{LABEL}=1"]
    if internal:
        # No route off the host. The engines never need outbound access, and
        # removing it eliminates a source of variance (and of surprise).
        args.append("--internal")
    args.append(name)
    _run(args)
    return name


def _network_exists(name: str) -> bool:
    proc = _run(["docker", "network", "ls", "--format", "{{.Name}}"], check=False)
    return name in (proc.stdout or "").split()


def remove_network(name: str) -> None:
    _run(["docker", "network", "rm", name], check=False)


def create_volume(name: str, device: Optional[str] = None) -> str:
    """Create a named volume, optionally backed by a specific host directory.

    Without `device`, Docker puts the volume under its own data-root, which on
    a default install is /var/lib/docker on the root filesystem. That is rarely
    where the space is: a benchmark box typically has a small root volume and a
    large data mount, and a million-vector corpus lands on the wrong one. The
    failure is not subtle but it is misleading — initdb reports "No space left
    on device" from inside the container and it reads as an engine problem.
    """
    cmd = ["docker", "volume", "create", "--label", f"{LABEL}=1"]
    if device:
        os.makedirs(device, exist_ok=True)
        cmd += ["--driver", "local",
                "--opt", "type=none", "--opt", "o=bind", "--opt", f"device={device}"]
    cmd.append(name)
    _run(cmd)
    return name


def root_dir() -> Optional[str]:
    """Where the daemon stores images, containers and volumes."""
    try:
        proc = _run(["docker", "info", "-f", "{{.DockerRootDir}}"], check=False)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def remove_volume(name: str) -> None:
    _run(["docker", "volume", "rm", "-f", name], check=False, timeout=180)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class ContainerSpec:
    name: str
    image: str
    network: Optional[str] = None
    cpuset: Optional[str] = None
    memory_bytes: Optional[int] = None
    shm_size: Optional[str] = None
    env: Dict[str, str] = None
    volumes: List[str] = None            # "src:dst[:mode]"
    command: List[str] = None
    entrypoint: Optional[str] = None
    user: Optional[str] = None
    workdir: Optional[str] = None
    detach: bool = True
    # Set only when the container must reach the internet (dataset fetch).
    allow_network: bool = False
    # Allocate a TTY and keep stdin open. Only the lab shell wants this: a
    # measurement container must never depend on a terminal, or the same
    # command behaves differently under nohup than it does by hand.
    interactive: bool = False
    #: `--ulimit name=value` per entry, or None for the daemon's defaults.
    #:
    #: Added for RLIMIT_CORE, and the two halves of a core dump are separate
    #: switches that both have to be thrown: mysqld's `--core-file` asks the
    #: process to dump, and this decides how much of it the kernel is willing
    #: to write. A container inherits the daemon's own limit, which on a stock
    #: install is whatever started dockerd, so asking for the core without
    #: raising the limit produces a TRUNCATED core or none -- and a truncated
    #: core is worse than no core, because gdb opens it and prints a backtrace
    #: that stops where the file does.
    #:
    #: Per-engine rather than global (`runtime.ulimits` in the engine config),
    #: so the six engines not under investigation keep the argv they had.
    #:
    #: NOT RECORDED ANYWHERE A RUN LEAVES BEHIND, and that gap is open rather
    #: than closed here. `Manifest.set_engine` keeps an engine's `source`,
    #: `build`, `images` and `presentation` and NOT its `runtime` block, and
    #: the `exec mysqld ...` banner that `_save_server_log` archives carries
    #: the SERVER flags only -- `--ulimit` is a docker flag and never reaches
    #: mysqld's argv. So the value in force is visible in the engine config
    #: and in nothing a run produces.
    #:
    #: LAST IN THE DATACLASS AND DEFAULTED, for the reason
    #: `engines.EngineRuntime.error_log` records: Python forbids a field
    #: without a default after one that has it, so a defaulted field goes at
    #: the end rather than beside `memory_bytes` where it reads best. Every
    #: caller constructs this class by keyword, so the position is invisible
    #: to them.
    ulimits: Optional[Dict[str, str]] = None


def _spec_args(spec: ContainerSpec) -> List[str]:
    args = ["docker", "run", "--name", spec.name, "--label", f"{LABEL}=1"]
    if spec.detach:
        args.append("-d")
    else:
        args.append("--rm")
    if spec.interactive:
        args.append("-it")
    if spec.network:
        args += ["--network", spec.network]
    if spec.cpuset:
        args += ["--cpuset-cpus", spec.cpuset]
    if spec.memory_bytes:
        # Memory and swap set to the same value: without this the container can
        # swap past its limit and the measurement silently becomes a disk
        # benchmark instead of an out-of-memory failure.
        args += ["--memory", str(spec.memory_bytes),
                 "--memory-swap", str(spec.memory_bytes)]
    if spec.shm_size:
        args += ["--shm-size", spec.shm_size]
    # One `--ulimit` per entry, not one flag carrying several: docker reads a
    # second value as an argument to the image's command, so a loop that
    # emitted the flag once would start the server with a stray argv entry.
    # Empty or None adds nothing at all, which is what keeps every engine that
    # declares no ulimits on byte-identical argv.
    for name, value in (spec.ulimits or {}).items():
        args += ["--ulimit", f"{name}={value}"]
    if spec.user:
        args += ["--user", spec.user]
    if spec.workdir:
        args += ["--workdir", spec.workdir]
    for key, value in (spec.env or {}).items():
        args += ["-e", f"{key}={value}"]
    for volume in (spec.volumes or []):
        args += ["-v", volume]
    if spec.entrypoint:
        args += ["--entrypoint", spec.entrypoint]
    args.append(spec.image)
    args += list(spec.command or [])
    return args


def start(spec: ContainerSpec) -> str:
    remove(spec.name)
    proc = _run(_spec_args(spec), timeout=300)
    return proc.stdout.strip()


def save_phase_log(run_dir: str, engine: str, dataset: str, phase: str,
                   resource_pass: str, lines: List[str]) -> Optional[str]:
    """Keep a phase's console output next to its measurements.

    The output of a phase is where a failure explains itself, and until now it
    existed only on the operator's terminal. A run that wrote no recall results
    and a churn that stopped after nothing both had their reason printed and
    then lost, and by the time anyone asked, the terminal was gone. The
    measurements are archived and copied between machines; their explanation
    should travel with them.

    `dataset` IS IN THE NAME AND IS NOT OPTIONAL, because it was missing and
    the name collided. One engine, one phase and one resource pass is not a
    unit of work -- a profile runs that triple once per dataset -- and this
    file is opened "w". `full` has four datasets, so the ann phase wrote this
    name four times per pass and three of the four were overwritten by the
    next dataset in the loop. Run full-20261015-215006 kept 14 log files for
    16 phases, and the one surviving `percona-ann-normalized.log` was gist's:
    the sift-128 ANN phase had failed in six seconds in both passes and the
    only record of why was gone. A keyword with a default would let the next
    caller reintroduce exactly that, silently, so it is positional and
    required and mirrors the order of the name it builds. `mem-` and `ops-`
    files in the same run directory already carry the dataset here.
    """
    if not lines:
        return None
    directory = os.path.join(run_dir, "logs")
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory, f"{engine}-{dataset}-{phase}-{resource_pass}.log")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return path
    except OSError:
        return None


def run_foreground(spec: ContainerSpec, timeout: int = 24 * 3600,
                   stream: bool = True,
                   sink: Optional[List[str]] = None,
                   line_filter: Optional[Any] = None) -> int:
    """Run a container in the foreground, streaming its output.

    Used for the measurement containers, whose logs are the primary record of
    what happened and must reach the operator live rather than after the fact.

    `sink` collects the streamed lines. An exit code alone cannot distinguish
    "this tool failed" from "this tool considers having nothing to do an
    error", and at least one dependency here takes the latter view.

    `line_filter(line) -> list[str]` decides what actually reaches the operator.
    It returns the lines to print, so it can hold some back and release them
    later — which is what suppressing a known-benign multi-line traceback
    requires. Everything still reaches `sink` regardless.
    """
    spec.detach = False
    remove(spec.name)
    args = _spec_args(spec)
    if not stream:
        proc = _run(args, check=False, timeout=timeout)
        for chunk in (proc.stdout, proc.stderr):
            if chunk:
                print(chunk)
                if sink is not None:
                    sink.extend(chunk.splitlines())
        return proc.returncode

    process = subprocess.Popen(args, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if sink is not None:
                sink.append(line)
            for shown in (line_filter(line) if line_filter else [line]):
                print(shown)
        if line_filter is not None:
            for shown in line_filter(None):      # flush anything held back
                print(shown)
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[docker] timeout after {timeout}s; killing {spec.name}")
        process.kill()
        remove(spec.name)
        return 124
    finally:
        if process.poll() is None:
            process.kill()


def run_interactive(spec: ContainerSpec) -> int:
    """Hand the terminal to a container and wait.

    Deliberately separate from run_foreground, which pipes stdout so it can
    stream, filter and archive every line. That piping is what a measurement
    needs and exactly what an interactive prompt cannot have: with stdout on a
    pipe there is no TTY, so no readline, no job control and no way to answer a
    password prompt. Nothing here is captured, because nothing typed at a
    prompt is a measurement.
    """
    spec.detach = False
    spec.interactive = True
    remove(spec.name)
    try:
        return subprocess.call(_spec_args(spec))
    finally:
        remove(spec.name)


def remove(name: str) -> None:
    _run(["docker", "rm", "-f", name], check=False, timeout=120)


def stop(name: str, timeout_s: int = 60) -> None:
    _run(["docker", "stop", "-t", str(timeout_s), name], check=False,
         timeout=timeout_s + 30)


def is_running(name: str) -> bool:
    proc = _run(["docker", "inspect", "--format", "{{.State.Running}}", name],
                check=False, timeout=30)
    return (proc.stdout or "").strip() == "true"


def container_exists(name: str) -> bool:
    """Whether the container is there at all, running or exited.

    Distinct from `is_running`, and the distinction is the one that matters
    when a phase failed during startup: a container that exited still HAS the
    logs, so `is_running` is the wrong question to ask before collecting them.
    What this rules out is the container never having been created -- a
    missing image, a failed network -- where `logs` would otherwise return the
    daemon's "No such container" on stderr and that text would be archived as
    though the server had said it.
    """
    proc = _run(["docker", "inspect", "--type", "container", name],
                check=False, timeout=30)
    return proc.returncode == 0


#: The fields of `docker inspect`'s State that say how a container ended, in
#: one template. Tab-separated with `State.Error` LAST deliberately: Go's
#: text/template copies the separator through verbatim, and the only one whose
#: value can itself contain a tab is the daemon's error string, so putting it
#: last means an embedded tab lands inside the field it belongs to rather than
#: shifting every field after it.
#:
#: `State.Running` is read and NOT reported. It is one of two independent
#: discriminators that stop this reporting an exit the container has not had;
#: see `exit_state`.
_EXIT_STATE_FORMAT = ("{{.State.Running}}\t{{.State.ExitCode}}\t"
                      "{{.State.OOMKilled}}\t{{.State.FinishedAt}}\t"
                      "{{.State.Error}}")

#: What `FinishedAt` reads as for a container that has never finished -- Go's
#: zero time, which the daemon returns for one created and never started.
_NEVER_FINISHED = "0001-01-01T00:00:00Z"


def exit_state(name: str) -> Optional[Dict[str, Any]]:
    """How a container ended: exit code, OOM verdict, finish time, error.

    `is_running` and `container_exists` both run `docker inspect` and throw
    the rest away, so until now nothing in the orchestrator recorded WHY a
    server container stopped. In run full-20261015-215006 twelve ops units
    died inside their first INSERT with `Lost connection to server during
    query` and the run directory could not say whether the kernel had killed
    the server or the server had killed itself.

    `OOMKilled` IS THE FIELD THIS EXISTS FOR. It settles a cgroup OOM against
    an in-process crash with no cooperation from the server at all -- a server
    killed by its cgroup writes nothing to its own error log, which is exactly
    the case where the error-log capture comes back empty and the absence of
    evidence reads as evidence of absence. That run's peak server RSS ACROSS
    ITS OPS PHASES was 48.5 GiB against a 112.5 GiB container cap, so the
    hypothesis was live and unfalsifiable from the artefacts. (Scoped to the
    ops phases deliberately, and the scope is the whole point of stating it:
    the run's percona peak is 54.7 GiB, in `gist-960-euclidean-tuned-ann`,
    and the ann path is not where the units under investigation died. Both
    figures are from this run's `mem-percona-*.jsonl` timeseries.)

    ONE INSPECT, NOT FOUR. The capture chain a failed phase runs already sits
    on 510 s of worst-case timeouts (`copy_file_out` documents the
    arithmetic), and three more inspects would be three more chances for a
    wedged daemon to hold a teardown open.

    RETURNS None FOR EVERY FAILURE AND RAISES FOR NONE, the contract
    `read_file_tail` and `copy_file_out` already document and for the same
    reason: the only caller runs in a `finally` on the path a crashed phase
    takes, so a container that has already gone, a daemon that is not
    answering and an output shape this does not understand must all come back
    as "no exit state" rather than as an exception that converts a lost
    measurement into a lost run.

    `OOMKilled` is COMPARED, not coerced. Everything arrives as text, and
    `bool("false")` is True: an engine that died on its own would be reported
    as having been killed by the kernel, which is the one error this field
    exists to prevent.

    None FOR A CONTAINER THAT HAS NOT ENDED, which is not a failure case, and
    TWO CHECKS RATHER THAN ONE because they catch different containers:

    * `Running` true. The usual caller is `run_harness`'s `finally` AFTER THE
      CLIENT FAILED, where the server is very often still up, and the daemon
      answers `ExitCode` 0 for it. "exit_code: 0" there would say the server
      exited cleanly when it had not exited at all, and a reader would close
      the question on it. It also catches a container STARTED AGAIN after a
      stop, which keeps the previous life's `FinishedAt` -- a real timestamp
      belonging to an exit that is not the one being asked about.
    * `FinishedAt` at Go's zero time. A container created and never started is
      not running either, and reports `ExitCode` 0 with no exit behind it;
      this is the only field that tells that apart from a clean exit 0.

    A server that was alive at capture time and is then stopped by `teardown`
    has no exit state worth recording, because teardown is what stopped it.
    """
    try:
        proc = _run(["docker", "inspect", "--type", "container",
                     "--format", _EXIT_STATE_FORMAT, name],
                    check=False, timeout=30)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    # maxsplit, so a tab inside the daemon's error string stays in it.
    #
    # rstrip of the LINE ENDING ONLY, not of whitespace: `State.Error` is
    # empty for every container that merely exited, so a `.strip()` here eats
    # the trailing tab and leaves three fields, and the normal case -- a
    # server that crashed without the daemon having anything to add -- parses
    # as "no exit state". Found by the test, not by reading.
    fields = (proc.stdout or "").rstrip("\r\n").split("\t", 4)
    if len(fields) != 5:
        return None
    running, code, oom, finished, error = fields
    if running.strip().lower() == "true":
        return None
    if finished.strip() == _NEVER_FINISHED:
        return None
    try:
        exit_code = int(code.strip())
    except ValueError:
        return None
    return {
        "exit_code": exit_code,
        "oom_killed": oom.strip().lower() == "true",
        "finished_at": finished.strip(),
        "error": error.strip(),
    }


def logs(name: str, tail: int = 100) -> str:
    proc = _run(["docker", "logs", "--tail", str(tail), name], check=False, timeout=60)
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def exec_in(name: str, args: Sequence[str], timeout: int = 60,
            check: bool = False) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", name, *args], check=check, timeout=timeout)


def read_file_tail(name: str, path: str, lines: int = 100) -> Optional[str]:
    """Last `lines` of a file *inside* a running container, or None.

    `logs` reads the container's stdout, which for an engine that was started
    with `--log-error=<file>` is the entrypoint banner and nothing else. The
    MySQL-family engines here are all started that way, so the stream that
    carries the signal, the assertion or the refused statement is a file in a
    volume that teardown deletes. This is how that file gets out while the
    container is still alive.

    `tail` rather than `cat` because an error log has no bound: mongod writes a
    few thousand lines an hour, and the caller wants the end of it anyway.

    RETURNS None FOR EVERY FAILURE and raises for none of them. The only
    caller runs in a `finally` on the path a crashed phase takes, so a file
    that was never created, a container that has already gone, a path the
    engine could not open and a daemon that is not answering must all come
    back as "no log" rather than as an exception that converts a lost
    measurement into a lost run.
    """
    try:
        proc = exec_in(name, ["tail", "-n", str(lines), path], timeout=120)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return text or None


def copy_file_out(name: str, path: str, dest_dir: str) -> Optional[str]:
    """Copy one file out of a container that may already have STOPPED.

    `read_file_tail` needs a running container, because `docker exec` does.
    `docker cp` does not -- it reads the container's filesystem through the
    daemon, which is why it is the documented way to retrieve a file from a
    container that has exited. (READ, from Docker's own documentation of the
    two commands, NOT measured here: verifying it would mean running a
    container, which this change is not allowed to do.)

    That difference is the whole point. A server that *failed its readiness
    probe* is still running and `read_file_tail` reaches it; a server that
    *died during startup* is not, and that is the one whose last words are
    worth most. It also does not care who owns the file: the engine writes its
    error log as root inside the container, so the same file on the host side
    of a bind-backed volume is typically root:root 0640 and unreadable to the
    orchestrator, which is why reading the bind directory directly is not the
    fallback here.

    Unbounded in SIZE, unlike `read_file_tail` -- `docker cp` has no `--tail`
    -- so the caller bounds it afterwards by taking the last lines of the copy,
    and chooses where the copy lands. Bounded in TIME at 300 s, which is the
    largest single wait in the capture chain and the figure an operator needs:
    `container_exists` 30 + `logs` 60 + `read_file_tail` 120 + this 300 is
    510 s of worst case before a failed setup gives up and tears down.

    Returns the local path, or None for every failure.
    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path) or "copied")
        proc = _run(["docker", "cp", f"{name}:{path}", dest],
                    check=False, timeout=300)
    except Exception:
        return None
    if proc.returncode != 0 or not os.path.isfile(dest):
        return None
    return dest


def wait_healthy(name: str, probe: Sequence[str], timeout_s: int = 180,
                 interval_s: float = 1.0) -> None:
    """Wait until `probe` succeeds inside the container.

    Fails with the container's own log tail attached, because "the server did
    not come up" is useless without the reason, and the reason is always in
    that log.
    """
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if not is_running(name):
            raise DockerError(
                f"container {name} exited during startup.\n--- logs ---\n{logs(name, 200)}"
            )
        proc = exec_in(name, probe, timeout=20)
        if proc.returncode == 0:
            return
        last = ((proc.stdout or "") + (proc.stderr or "")).strip()
        time.sleep(interval_s)
    raise DockerError(
        f"container {name} did not become ready within {timeout_s}s.\n"
        f"last probe output: {last}\n--- logs ---\n{logs(name, 200)}"
    )


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------

#: The cgroup v2 pressure files `MemorySampler` reads, by the prefix their
#: fields carry in a sample row.
PRESSURE_FILES = (("cpu", "/sys/fs/cgroup/cpu.pressure"),
                  ("io", "/sys/fs/cgroup/io.pressure"))

#: The fields kept from each file's `some` and `full` lines. `avg10` is the
#: kernel's ten-second moving average, kept for a reader looking at one row;
#: `total` is cumulative stall MICROSECONDS, which is what a window's figure
#: is computed from (`report/server_windows.py`), because a difference of
#: totals covers exactly the window and a moving average does not.
PRESSURE_FIELDS = ("avg10", "total")


def pressure_fields(prefix: str, text: Optional[str]
                    ) -> Dict[str, Optional[float]]:
    """`{<prefix>_pressure_<some|full>_<avg10|total>: value}` from one PSI
    file's text.

    EVERY key is always present, and a missing file, an unreadable one or a
    line the kernel did not write is None -- never 0. Zero is the claim "no
    task stalled", which is a measurement; a kernel built without PSI, a
    cgroup v1 host or a container that has exited says nothing at all, and a
    sampler that wrote 0 for those would publish an idle server where nobody
    looked. (cgroup v2 writes no `full` line for cpu at the root on older
    kernels; that key is None for the same reason.)
    """
    out: Dict[str, Optional[float]] = {
        f"{prefix}_pressure_{kind}_{field}": None
        for kind in ("some", "full") for field in PRESSURE_FIELDS}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        for item in parts[1:]:
            name, _, value = item.partition("=")
            if name not in PRESSURE_FIELDS:
                continue
            try:
                out[f"{prefix}_pressure_{parts[0]}_{name}"] = float(value)
            except ValueError:
                continue
    return out


class MemorySampler(threading.Thread):
    """Sample a container's memory and CPU into a JSONL timeseries.

    A timeseries rather than a single peak, for two reasons: the peak of each
    phase can be derived from it after the fact by intersecting with that
    phase's time window, and a memory curve shows things a scalar cannot — a
    graph cache filling up, a build spiking, an engine steadily leaking.
    """

    def __init__(self, container: str, output_path: str, interval_s: float = 0.25,
                 wait_for_start_s: float = 0.0):
        super().__init__(daemon=True)
        self.container = container
        self.output_path = output_path
        self.interval_s = interval_s
        # The ops path starts its server and then samples it. The ann path runs
        # its container in the foreground, so the sampler has to be started
        # first and wait for the container to appear -- without this it checked
        # once, found nothing running, and exited before the phase began.
        self.wait_for_start_s = wait_for_start_s
        self._stop_event = threading.Event()
        self.samples = 0
        # Which measurement these samples belong to. The file is opened for
        # append, so a phase that runs twice for one engine, dataset and
        # resource pass lands both in it -- which is what happened when MHNSW
        # and VIDX were given a second graph degree five days after the first
        # run. Without a marker the two are one series with a 123-hour gap in
        # the middle, drawn as a continuous line and differenced end to end.
        self.session = int(time.time())

    def _read(self, path: str) -> Optional[int]:
        proc = exec_in(self.container, ["cat", path], timeout=10)
        value = (proc.stdout or "").strip()
        return int(value) if value.isdigit() else None

    def _cpu_seconds(self) -> Optional[float]:
        proc = exec_in(self.container, ["cat", "/sys/fs/cgroup/cpu.stat"], timeout=10)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("usage_usec"):
                try:
                    return int(line.split()[1]) / 1e6
                except (IndexError, ValueError):
                    return None
        return None

    def _cgroup_pressure(self) -> Dict[str, Optional[float]]:
        """The container's own CPU and IO pressure (ops version 9).

        One `docker exec` for both files, not one each: the sampler already
        spends three per row at a 0.25 s interval, and every exec is time a
        row is late. A file that does not exist prints nothing between its
        markers, and `pressure_fields` turns that into None.
        """
        script = "; ".join(f"echo @@{prefix}; cat {path} 2>/dev/null"
                           for prefix, path in PRESSURE_FILES)
        try:
            proc = exec_in(self.container, ["sh", "-c", script], timeout=10)
            text = proc.stdout or ""
        except Exception:
            text = ""
        sections: Dict[str, str] = {}
        current = None
        for line in text.splitlines():
            if line.startswith("@@"):
                current = line[2:].strip()
                sections[current] = ""
            elif current is not None:
                sections[current] += line + "\n"
        out: Dict[str, Optional[float]] = {}
        for prefix, _path in PRESSURE_FILES:
            out.update(pressure_fields(prefix, sections.get(prefix)))
        return out

    def _host_pressure(self) -> Dict[str, Optional[float]]:
        """What the whole machine is doing, not only this container.

        The container's own cgroup cannot see load the harness did not create,
        and that is exactly the load worth catching: three AliSQL recall points
        were measured while something else was on the box, and the only trace
        of it was a latency ratio that took five days and a hand comparison to
        notice. Subtracting the container's CPU from the host's over the same
        window names it directly.

        Read from /proc rather than through the container, because the
        orchestrator runs on the host and the container's /proc is namespaced.
        """
        out: Dict[str, Optional[float]] = {"host_load1": None,
                                           "host_cpu_seconds": None}
        try:
            with open("/proc/loadavg") as fh:
                out["host_load1"] = float(fh.read().split()[0])
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/stat") as fh:
                fields = fh.readline().split()
            if fields and fields[0] == "cpu":
                values = [int(v) for v in fields[1:8]]
                # user+nice+system+irq+softirq+steal, i.e. everything but idle
                # and iowait. Jiffies at the kernel's tick rate.
                busy = values[0] + values[1] + values[2] + sum(values[5:])
                out["host_cpu_seconds"] = busy / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError, AttributeError):
            pass
        return out

    def _await_container(self) -> bool:
        deadline = time.time() + self.wait_for_start_s
        while not self._stop_event.is_set():
            if is_running(self.container):
                return True
            if time.time() >= deadline:
                return False
            self._stop_event.wait(0.5)
        return False

    def run(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        if self.wait_for_start_s and not self._await_container():
            return
        with open(self.output_path, "a", buffering=1) as fh:
            while not self._stop_event.is_set():
                if not is_running(self.container):
                    break
                current = (self._read("/sys/fs/cgroup/memory.current")
                           or self._read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
                peak = (self._read("/sys/fs/cgroup/memory.peak")
                        or self._read("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"))
                if current is not None:
                    fh.write(json.dumps({
                        "t": round(time.time(), 3),
                        "container": self.container,
                        "rss_bytes": current,
                        "peak_bytes": peak,
                        "cpu_seconds": self._cpu_seconds(),
                        "session": self.session,
                        **self._host_pressure(),
                        **self._cgroup_pressure(),
                    }) + "\n")
                    self.samples += 1
                self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=10)


def running_containers() -> List[str]:
    """Names of vector-bench containers currently running."""
    proc = _run(["docker", "ps", "--format", "{{.Names}}",
                 "--filter", f"label={LABEL}=1"], check=False, timeout=60)
    return [n for n in (proc.stdout or "").split() if n]


def containers_publishing(port: int) -> List[str]:
    """Names of containers publishing this host port, vector-bench or not.

    Docker reports a taken port as an endpoint id and a network driver, which
    does not tell an operator which container to stop.
    """
    proc = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
                check=False, timeout=30)
    names: List[str] = []
    for line in (proc.stdout or "").splitlines():
        name, _, ports = line.partition("\t")
        if f":{port}->" in ports:
            names.append(name)
    return names


def remove_tree_as_root(path: str, image: str) -> bool:
    """Delete a host directory that a container filled as root.

    Engines run as root inside their containers, so everything they write to a
    bind mount is root-owned. shutil.rmtree cannot remove it, and with
    ignore_errors=True it fails silently, which is how a teardown that looked
    like it worked left tens of GB behind.

    Returns True if the path is gone afterwards.
    """
    if not os.path.exists(path):
        return True
    if os.getuid() == 0:
        shutil.rmtree(path, ignore_errors=True)
        return not os.path.exists(path)

    # Try it directly first. The directory is only root-owned once a server has
    # actually written into it; when setup failed before that -- a missing
    # image, a server that never came up -- it is an empty directory this
    # process made, and spawning a container to delete it needs the very image
    # whose absence caused the failure. That path then fails and warns that a
    # corpus is still on disk, which is both alarming and untrue.
    #
    # Not ignore_errors: a silent failure here is what leaves a full corpus
    # behind. If it does not work, fall through to the container.
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        pass

    parent, name = os.path.dirname(path), os.path.basename(path)
    spec = ContainerSpec(
        name=f"vb-rm-{os.getpid()}",
        image=image,
        network="none",
        entrypoint="rm",
        command=["-rf", f"/target/{name}"],
        volumes=[f"{parent}:/target:rw"],
        detach=False,
    )
    try:
        run_foreground(spec, timeout=600, stream=False)
    except Exception as exc:  # pragma: no cover - depends on the daemon
        print(f"[clean] could not remove {path}: {exc}", file=sys.stderr)
    return not os.path.exists(path)


#: How deep under the bind directory a match is looked for.
#:
#: PUBLIC, because it is part of this function's contract rather than an
#: implementation detail: `ops_pass.core_capture_verdict` has to know it to
#: tell an operator whether a core written at a given `core_pattern` will
#: actually be found. The volume is
#: mounted at the engine's `data_mount` and mysqld chdirs into its datadir one
#: level below it, so depth 2 covers both the volume root and the datadir while
#: refusing to walk an InnoDB tree of several hundred thousand files.
COPY_GLOB_MAX_DEPTH = 2

#: Bounded in TIME, like `copy_file_out`, and the figure is the one an operator
#: needs. A core here is bounded by the server's resident set minus whatever
#: `innodb_buffer_pool_in_core_file=OFF` keeps out of it: the reference run's
#: peak RSS across its OPS phases was 48.5 GiB against a ~39 GiB buffer pool,
#: so ~9 GiB, and 900 s is a slow disk's worth of that. (54.7 GiB is that
#: run's percona peak overall, in an ann phase, which this capture does not
#: serve.)
_COPY_GLOB_TIMEOUT_S = 900


def copy_glob_as_root(bind_dir: str, pattern: str, dest_dir: str,
                      image: str) -> List[str]:
    """Copy files matching `pattern` out of a host directory a container
    filled as root, and hand the copies back to the invoking user.

    WHY A CONTAINER, which is the whole reason this is not four lines of
    `glob` and `shutil`. The engine writes its data directory as root inside
    its container, so the host side of a bind-backed volume is typically
    root:root 0640 and a non-root orchestrator cannot even list it -- the same
    fact `remove_tree_as_root` exists for and branches on `os.getuid() == 0`
    for.

    WHY NOT THE TWO MECHANISMS THAT ALREADY EXIST. `read_file_tail` needs a
    RUNNING container, and the case this serves is a server that died.
    `copy_file_out` works on a stopped one but needs an EXACT path -- and a
    core's filename is decided by the host's `/proc/sys/kernel/core_pattern`,
    which this framework must not change (see `ops_pass.host_core_pattern`).
    So the ENUMERATION, not just the copy, has to happen through a container.

    `pattern` is a FILENAME glob, matched at a depth of at most
    `COPY_GLOB_MAX_DEPTH` below `bind_dir`, not a path glob: the caller knows
    what a core is called, not which of the volume's two plausible directories
    the kernel put it in.

    The source is mounted READ-ONLY. The directory is about to be deleted by
    teardown, so a `mv` -- which on one filesystem would cost no space at all
    -- would be defensible; it is not done because a copy that mutates what it
    is reading is the wrong default for an artefact somebody may want to
    re-examine, and a half-failed `mv` leaves it in neither place. The cost is
    one core's worth of transient disk, on the filesystem the checkout is on,
    released seconds later when `remove_tree_as_root` takes the volume.

    The copies are CHOWNED BACK inside the same container, the way
    `ann_pass.fix_ownership` does it and for the reason it gives: without that
    the artefact is unreadable to the operator and to `export`. In the same
    container rather than a second one because a second `docker run` is a
    second chance for a wedged daemon to hold a teardown open.

    No `os.getuid() == 0` short cut, unlike `remove_tree_as_root`. The case
    that branch exists for -- a setup that failed before the image existed --
    is already ruled out by the caller, which does not come here unless the
    bind directory is there, and it is only there with contents once a server
    has run from that image.

    Returns the local paths copied, and `[]` for every failure: no such
    directory, no match, no daemon, a timeout, an unwritable destination. The
    only caller runs before teardown on the path a crashed phase takes.
    """
    if not os.path.isdir(bind_dir):
        return []
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError:
        return []
    uid, gid = os.getuid(), os.getgid()
    # -exec rather than a `for` over $(find ...), which word-splits on a
    # filename containing a space. `core_pattern` can be set to one.
    script = (
        f"find /src -maxdepth {COPY_GLOB_MAX_DEPTH} "
        f"-name {shlex.quote(pattern)} -type f "
        f"-exec cp -p {{}} /dst/ \\; ; "
        f"chown -R {uid}:{gid} /dst"
    )
    spec = ContainerSpec(
        name=f"vb-cp-{os.getpid()}",
        image=image,
        network="none",
        entrypoint="sh",
        command=["-c", script],
        volumes=[f"{bind_dir}:/src:ro", f"{dest_dir}:/dst:rw"],
        detach=False,
    )
    try:
        run_foreground(spec, timeout=_COPY_GLOB_TIMEOUT_S, stream=False)
    except Exception as exc:  # pragma: no cover - depends on the daemon
        print(f"[docker] could not copy {pattern} out of {bind_dir}: {exc}",
              file=sys.stderr)
        return []
    try:
        names = sorted(os.listdir(dest_dir))
    except OSError:
        return []
    return [os.path.join(dest_dir, n) for n in names]


def cleanup_run(run_id: str = "") -> Dict[str, int]:
    """Remove every container, network and volume belonging to a run.

    With no run_id this removes everything the harness has ever created. Always
    scoped by the vector-bench label, so nothing else on a shared box is
    touched. Returns counts per kind.
    """
    removed: Dict[str, int] = {}
    for kind in ("container", "network", "volume"):
        filters = ["--filter", f"label={LABEL}=1"]
        # An empty --filter name= is not "match everything", it is a filter on
        # the empty string, and whether that matches is version-dependent.
        if run_id:
            filters += ["--filter", f"name={run_id}"]
        proc = _run(["docker", kind, "ls", "-q", *filters], check=False, timeout=60)
        ids = [i for i in (proc.stdout or "").split() if i]
        removed[kind] = len(ids)
        if not ids:
            continue
        force = ["-f"] if kind in ("container", "volume") else []
        _run(["docker", kind, "rm", *force, *ids], check=False, timeout=300)
    return removed
