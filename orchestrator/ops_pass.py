"""Ops measurement path: build cost, concurrency, filtered search, churn.

Two containers, not one. The server runs alone in its own container so that its
cgroup accounting measures the server and nothing else — if the harness shared
that container, the several hundred megabytes of NumPy holding the dataset would
be charged to the engine and every peak-memory number would be wrong.

The harness runs in a second container on the same private network and connects
over TCP. That adds loopback network cost to every query, identically for all
three engines, which is why concurrency numbers here are compared against each
other rather than against the ann-benchmarks in-process numbers.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import docker_ctl
from . import engines as engines_mod
from .ann_knobs import VILLAGESQL_ENGINES
from .config import (ResolvedResources, engine_wait_instruments,
                     engine_wait_sampler_interval, ops_wait_instruments,
                     ops_wait_sampler, profile_window_clients,
                     server_args, server_env)
from .perf_window import ProfileWatcher

#: Where, inside the client container, the harness waits for the profiled
#: window's acknowledgement. Under the results mount, which is the one
#: directory both sides can write; see `perf_window.ProfileWatcher`.
PROFILE_ACK_SUBDIR = ".vb-profile"
from .manifest import utcnow


# How much of the server's own log to keep beside the measurements. Generous,
# because the lines that matter are the ones written while something was going
# wrong and there is no way to know in advance how much noise follows them.
# mongod is the verbose extreme at a few thousand lines an hour; the others
# write almost nothing.
SERVER_LOG_TAIL = 20000

# Where the kernel decides what a core dump is called and whether one is
# written at all. Read and reported, NEVER written: a benchmark framework must
# not modify the machine it measures, and this is a machine-wide sysctl that
# governs every other process on the box. The one-line opt-in is the
# operator's to make, in the rebuild procedure, not the framework's.
HOST_CORE_PATTERN_PATH = "/proc/sys/kernel/core_pattern"

# What a core is called. The kernel's own default `core_pattern` is the bare
# string `core`, and the common customisations append the pid, the executable
# or both (`core.%p`, `core.%e.%p`), so a prefix match is what covers them.
# Matched as a FILENAME at a bounded depth under the volume -- see
# `docker_ctl.copy_glob_as_root` -- because mysqld chdirs into its datadir, one
# level below the mount point, and a relative `core_pattern` lands in the
# crashing process's working directory.
#
# WHAT THIS CANNOT REACH, said plainly rather than discovered later: a
# `core_pattern` that is an ABSOLUTE path puts the core outside the engine's
# volume, where neither this nor anything else in the run directory will find
# it, and a pattern beginning with `|` writes no file anywhere. Both cases are
# reported at server start instead.
CORE_FILE_GLOB = "core*"

# What to tell an operator to set, and it is not an arbitrary example. This
# capture collects `core*`; `scripts/gdb-backtrace.py`, which reads the result,
# searches for `core`, `core.*`, `*.core` and `core-*`. The two sets are not
# the same, so only their INTERSECTION works end to end, and the two edges are
# both real:
#
#   core%p   -> `core12345`   collected here, invisible to the backtrace script
#   %e.core  -> `mysqld.core` sought by that script, never collected here
#
# `core.%p` is in both and is what every message here recommends, so an
# operator who follows the console lands in the agreed set rather than on one
# of those edges. `tests/test_e21_crash_capture_plumbing.py` asserts the
# agreement against that script's own constants rather than against this
# comment.
RECOMMENDED_CORE_PATTERN = "core.%p"


#: How far mysqld's working directory sits below the VOLUME's mount point, and
#: therefore how much of `COPY_GLOB_MAX_DEPTH` a relative `core_pattern` has
#: already spent before its own directory components are counted.
#:
#: One, because `docker/percona/entrypoint-percona.sh` defaults
#: `VB_DATA_DIR` to `/var/lib/vbench/data` against a `data_mount` of
#: `/var/lib/vbench`, and mysqld chdirs into its datadir. An engine that put
#: its datadir somewhere else would need this passed rather than assumed, and
#: `core_capture_verdict` takes it as an argument for that reason.
DATADIR_DEPTH_BELOW_MOUNT = 1


def core_capture_verdict(
        pattern: Optional[str], data_mount: str,
        datadir_depth: int = DATADIR_DEPTH_BELOW_MOUNT) -> Tuple[bool, str]:
    """Whether a core written under `pattern` would reach the run directory.

    Returns `(reachable, reason)`. A pure function of the two inputs so every
    case can be probed, which matters because this is the function the whole
    crash-capture finding rests on: if it says "a core would land in the
    server's data directory" and no core ever appears, nothing downstream can
    tell that from "the server did not crash".

    THE ABSOLUTE CASE HAS THREE OUTCOMES, NOT ONE, and an earlier version of
    this said only the middle one. `core(5)` says paths "are interpreted
    according to the settings that are active for the crashing process", so an
    absolute pattern resolves in the CONTAINER's mount namespace -- and that
    namespace contains the bind-backed volume at the engine's data mount. So:

      * under the data mount and within `COPY_GLOB_MAX_DEPTH` -- the core
        lands in the volume, survives `docker rm`, and the capture's
        `find /src -maxdepth 2` reaches it. This is a WORKING configuration
        and the earlier text called it broken.
      * under the data mount but deeper -- written, survives, and missed by
        the capture's depth bound.
      * anywhere else -- either it lands in the container's own writable
        layer and goes with `docker rm`, or, if that directory does not exist
        in the image, NO FILE IS WRITTEN AT ALL. Neither can be told apart
        from here, and both mean the run directory receives nothing, so the
        verdict is the same and the reason names both.

    The half of the earlier claim derived from `core(5)` was right and the
    half assumed was wrong, which is in miniature the fault the change this
    belongs to exists to remove.

    AND THE RELATIVE CASE HAS TWO OUTCOMES FOR THE SAME REASON. A relative
    pattern lands in mysqld's working directory, which is the datadir -- but
    it may carry directory components of its own, and `cores/core.%p` is then
    three levels below the mount point and outside the capture's reach. The
    depth arithmetic therefore runs on BOTH arms; running it on only the
    absolute one is how this function came to promise a copy-out for a core
    it could not collect.
    """
    if pattern is None:
        return (False, f"unreadable ({HOST_CORE_PATTERN_PATH}); whether a "
                       f"core can be written is unknown")
    if pattern.startswith("|"):
        return (False,
                "that is a pipe, so the kernel hands the dump to that "
                "program and writes NO FILE to any filesystem -- no core "
                "will reach this run directory however the server dies. The "
                "helper itself runs fine; piping is what leaves no file, so "
                "putting it in the image would change nothing")
    directory, _, name = pattern.rpartition("/")
    if pattern.startswith("/"):
        mount = data_mount.rstrip("/")
        if directory == mount or directory.startswith(mount + "/"):
            depth = len([part for part in
                         directory[len(mount):].split("/") if part]) + 1
            if depth > docker_ctl.COPY_GLOB_MAX_DEPTH:
                return (False,
                        f"that is an absolute path {depth} levels under "
                        f"{mount}, and the capture searches only "
                        f"{docker_ctl.COPY_GLOB_MAX_DEPTH}, so the core is written and "
                        f"the copy-out does not find it")
            if not name.startswith("core"):
                return (False,
                        f"that is an absolute path inside the volume, so the "
                        f"core is written and survives teardown -- but under "
                        f"a name the capture looks for as {CORE_FILE_GLOB!r} "
                        f"and will not find")
            return (True,
                    f"that is an absolute path inside the volume at {mount}, "
                    f"so the core lands where the copy-out reaches it")
        return (False,
                "that is an absolute path outside the engine's volume. It "
                "resolves in the container's mount namespace, so either it "
                "lands in the container's own writable layer and goes with "
                "`docker rm`, or -- if that directory does not exist in the "
                "image -- no file is written at all. Either way this run "
                "directory receives nothing")
    # A RELATIVE PATTERN CAN CARRY A DIRECTORY TOO, and the first version of
    # this function computed depth on the absolute arm and not here -- so
    # `cores/core.%p`, which lands three levels under the mount and outside
    # the capture's reach, was reported as "copied out before teardown". That
    # is the same false-positive direction as the bug this function was
    # written to remove, left un-applied to the other arm, and the branch
    # table could not catch it because there was no branch for it to have a
    # row against.
    depth = datadir_depth + len([part for part in directory.split("/")
                                 if part]) + 1
    if depth > docker_ctl.COPY_GLOB_MAX_DEPTH:
        return (False,
                f"that is relative with a directory in it, so the core lands "
                f"{depth} levels under the volume's mount point and the "
                f"capture searches only {docker_ctl.COPY_GLOB_MAX_DEPTH}; it "
                f"is written and survives teardown, and the copy-out does "
                f"not find it")
    if not name.startswith("core"):
        return (False,
                f"that is relative, so the core does land in the server's "
                f"data directory, but under a name the capture looks for as "
                f"{CORE_FILE_GLOB!r} and will not find")
    return (True, "a core would land in the server's data directory and be "
                  "copied out before teardown")


def host_core_pattern() -> Optional[str]:
    """The host's `kernel.core_pattern`, or None if it cannot be read.

    None rather than an exception: this is read on the server-start path, and a
    kernel that does not publish the file, a locked-down /proc or a non-Linux
    host must not stop a measurement that has nothing to do with core dumps.
    """
    try:
        with open(HOST_CORE_PATTERN_PATH) as fh:
            return fh.read().strip() or None
    except OSError:
        return None

# Readiness probes. Each performs a real query, not just a port check: all three
# servers accept connections before they are able to serve, and a premature
# start would charge initialisation time to the first measurement.

# Database account per engine. PostgreSQL's bootstrap superuser is `postgres`;
# the MySQL-family images create a `bench` account from their --init-file
# (--skip-grant-tables is unusable because on MySQL 8 it disables networking).



# The ops client loads the corpus once (unlike the ann client, which loads it
# twice) and then needs working space for the brute-force ground truth it
# computes over the qualifying subset. 1.5x the file plus a flat 2 GB covers
# both, and the caller takes the max against the configured client limit so
# small corpora keep the profile's value.
_OPS_CLIENT_COPIES = 1.5
_OPS_CLIENT_BASE_BYTES = 2 * 1024**3


def ops_client_memory_bytes(datasets_dir: str, dataset: str) -> int:
    """Floor for the ops client's container memory, sized to the corpus."""
    try:
        size = os.path.getsize(os.path.join(datasets_dir, f"{dataset}.hdf5"))
    except OSError:
        return _OPS_CLIENT_BASE_BYTES
    return int(size * _OPS_CLIENT_COPIES) + _OPS_CLIENT_BASE_BYTES


class OpsRun:
    """Manages the server/client container pair for one ops measurement."""

    def __init__(self, engine: str, engine_cfg: Dict[str, Any],
                 resolved: ResolvedResources, resource_pass: str,
                 paths: Dict[str, str], run_id: str, dataset: str, tag: str,
                 wait_instruments: bool = False,
                 profile_window_clients: Optional[int] = None):
        self.engine = engine
        # Both from the profile, through `config.ops_wait_instruments` and
        # `config.profile_window_clients`; off by default, which keeps the
        # server's argv and the harness's byte-identical to before.
        self.wait_instruments = bool(wait_instruments)
        self.profile_window_clients = profile_window_clients
        self.engine_cfg = engine_cfg
        self.resolved = resolved
        self.resource_pass = resource_pass
        self.paths = paths
        self.run_id = run_id
        self.dataset = dataset
        self.tag = tag

        safe = f"{run_id}-{engine}-{tag}".replace("_", "-").replace(".", "-")[:55]
        self.network = f"{safe}-net"
        self.volume = f"{safe}-data"
        self.server_name = f"{safe}-srv"
        self.client_name = f"{safe}-cli"
        self.port = int(engine_cfg.get("port", engines_mod.get(engine).port))

    # ------------------------------------------------------------------

    def __enter__(self) -> "OpsRun":
        try:
            docker_ctl.create_network(self.network, internal=True)
            # Backed by a directory under VB_ROOT rather than Docker's
            # data-root, so the corpus lands on the filesystem the checkout is
            # on. See docker_ctl.create_volume.
            docker_ctl.create_volume(
                self.volume,
                device=os.path.join(self.paths["engine_state"], "ops",
                                    self.volume),
            )
            self._start_server()
        except BaseException:
            # `with` calls __exit__ only if __enter__ *returned*. Setup happens
            # in three steps and the last two can fail -- a missing image, or a
            # server that never becomes healthy inside five minutes -- and
            # every failure after the first step leaves behind a network, a
            # volume, a root-owned directory under state/ops and sometimes a
            # server container, with nothing saying so. Reproduced by asking
            # for an engine whose image is not built.
            #
            # BaseException, not Exception: the widest window here is
            # wait_healthy's five-minute poll, and the likeliest thing to
            # arrive during it is the operator's Ctrl-C.
            #
            # THE LOGS COME OUT BEFORE TEARDOWN, because this is the one path
            # where the server's own log is most wanted and was least likely
            # to exist. `run_harness` is never entered when setup fails, so
            # neither capture ran, and `teardown` then removed the volume: a
            # server that refused to start under a tuned
            # `innodb_buffer_pool_size` left "did not become ready within
            # 300s" plus the three-line banner, and the assertion that
            # actually caused it was deleted. Cannot raise -- see
            # `_save_startup_log`.
            #
            # TEARDOWN IS IN A `finally`, not merely sequenced after the
            # capture. Against a wedged daemon the capture can sit on
            # 30 + 60 + 120 + 300 s of timeouts -- inspect, logs, the exec
            # tail, the `docker cp` fallback -- 510 s in all. Its handlers
            # are
            # `except Exception`, which by design does not catch
            # KeyboardInterrupt -- so a second Ctrl-C aborts it at once. Merely
            # sequenced, that interrupt escapes past `teardown()` and leaks
            # every one of the things listed above. The `finally` is what makes
            # the fast abort safe.
            try:
                self._save_startup_log()
            finally:
                self.teardown()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.teardown()

    def _start_server(self) -> None:
        image = self.engine_cfg.get("image", {}).get(
            "runtime", f"vector-bench/{self.engine}-runtime"
        )
        if not docker_ctl.image_exists(image):
            raise docker_ctl.DockerError(
                f"image {image} not found. Build it first:\n"
                f"  ./run-benchmark.sh build --engines {self.engine}"
            )

        flags = server_args(self.engine_cfg, self.resource_pass, self.resolved,
                            wait_instruments=self.wait_instruments)
        data_mount = ("/var/lib/postgresql" if self.engine == "pgvector"
                      else "/var/lib/vbench")

        spec = docker_ctl.ContainerSpec(
            name=self.server_name,
            image=image,
            network=self.network,
            cpuset=self.resolved.server_cpuset,
            memory_bytes=self.resolved.server_memory_bytes,
            shm_size=self.resolved.shm_size,
            env={
                "VB_SERVER_ARGS": " ".join(flags),
                "VB_RUN_ID": self.run_id,
                # See `config.server_env`: one builder for both paths and
                # for the manifest's `engines_resolved` record.
                **server_env(self.resolved),
            },
            volumes=[f"{self.volume}:{data_mount}:rw"],
            command=["server"],
            detach=True,
            # Empty for six of the seven engines, which keeps their `docker
            # run` argv byte-identical -- see `ContainerSpec.ulimits`.
            ulimits=self._ulimits() or None,
        )
        print(f"[ops] starting {self.engine} server: cpuset={self.resolved.server_cpuset} "
              f"mem={self.resolved.server_memory_bytes / 1024**3:.1f}GB")
        print(f"[ops] server flags: {' '.join(flags)}")
        self._report_core_pattern(data_mount)
        docker_ctl.start(spec)
        docker_ctl.wait_healthy(self.server_name, list(engines_mod.get(self.engine).probe), timeout_s=300)
        print(f"[ops] {self.engine} server ready")

    def _ulimits(self) -> Dict[str, str]:
        """The engine's declared `--ulimit` entries, as a mapping.

        Empty for every engine whose config declares none, which is six of the
        seven. The registry holds a tuple of pairs rather than a dict so an
        `EngineRuntime` stays hashable; this is where it becomes the mapping
        `ContainerSpec` wants.
        """
        try:
            return dict(engines_mod.get(self.engine).ulimits)
        except Exception:                        # pragma: no cover - defensive
            return {}

    def _report_core_pattern(self, data_mount: str) -> None:
        """Print the host's `kernel.core_pattern` beside the server flags, and
        say plainly when it means no core will be produced.

        THE FLAGS AND THE ULIMIT ARE NOT SUFFICIENT and that is the point of
        printing this. There are three ways for the host to make the whole
        chain inert, and `core(5)` is the source for each:

        * A pattern beginning with `|` makes the kernel PIPE THE DUMP TO THAT
          PROGRAM'S STDIN AND WRITE NO FILE AT ALL -- not into the volume, not
          anywhere else on any filesystem. The helper then does whatever its
          own policy says; on Ubuntu that is apport, which may file a crash
          report or discard it, and either way nothing lands where this capture
          could read it. AN EARLIER VERSION OF THIS COMMENT SAID THE HELPER
          "does not exist in the container's mount namespace", AND THAT WAS
          WRONG in a way that would send an operator down a dead end --
          `/usr/share/apport/apport` is present on this host, and `core(5)` is
          explicit that "the program pathname is interpreted with respect to
          the initial mount namespace as it is always executed there. It is not
          affected by the settings (e.g., root directory, mount namespace,
          current working directory) of the crashing process", which is why
          `%P` and `%I` exist at all. The helper runs. The point is that a pipe
          produces no file, so bind-mounting apport into the image or baking it
          in would change nothing.
        * An ABSOLUTE pattern writes a file, but `core(5)` says paths "are
          interpreted according to the settings that are active for the
          crashing process", so it resolves INSIDE THE CONTAINER -- in the
          container's own writable layer rather than in the bind-backed volume,
          and that layer goes with `docker rm`.
        * A RELATIVE pattern that does not begin with `core` -- `%e.core`,
          `dump-%p` -- writes into mysqld's working directory, which IS the
          volume, under a name `CORE_FILE_GLOB` does not match.

        All three are ordinary settings on a Linux desktop or a managed server,
        so an operator who is told only "core-file requested" will conclude the
        server did not crash.

        IT IS READ AND NOT WRITTEN. Changing it would change the behaviour of
        every process on the machine under measurement; CLAUDE.md's rule about
        host dependencies is the mild version of the same principle. The
        one-line opt-in belongs in the rebuild procedure, as something the
        operator does knowingly.

        Only for an engine that asked for a core, so the other six engines'
        console output is unchanged.
        """
        if "core" not in self._ulimits():
            return
        pattern = host_core_pattern()
        reachable, reason = core_capture_verdict(pattern, data_mount)
        # The value only when there IS one: an unreadable sysctl printed as
        # `core_pattern: None` reads as a setting rather than as a failure to
        # look, and the reason already says which.
        if pattern is not None:
            print(f"[ops] core_pattern: {pattern}")
        print(f"[ops] {reason}")
        if pattern == RECOMMENDED_CORE_PATTERN:
            return
        # The note goes on EVERY other value, including the ones the capture
        # does reach. `core%p` is the case that needs it: this capture
        # collects it and `scripts/gdb-backtrace.py` cannot read it, so
        # "copied out before teardown" is true and still not enough. This
        # function deliberately does not know that script's globs -- the
        # agreement between the two is asserted in
        # tests/test_e21_crash_capture_plumbing.py -- so the note is
        # unconditional rather than clever.
        note = (f"[ops] {RECOMMENDED_CORE_PATTERN!r} is the value this chain "
                f"handles end to end")
        if not reachable:
            note += (", and setting it is the operator's to do, not this "
                     "framework's")
        print(note + ".")

    # ------------------------------------------------------------------

    def run_harness(self, args: List[str], output_path: str,
                    memory_timeseries: Optional[str] = None,
                    timeout_s: int = 12 * 3600) -> int:
        """Run the ops harness against the running server."""
        image = self.engine_cfg.get("image", {}).get(
            "bench", f"vector-bench/{self.engine}-bench"
        )

        volumes = [
            f"{self.paths['harness']}:/opt/harness:ro",
            f"{self.paths['datasets']}:/datasets:ro",
            f"{self.paths['ops_results']}:/results:rw",
        ]
        data_dir_arg: List[str] = []
        mount_point = engines_mod.get(self.engine).server_data_mount
        if mount_point:
            # Read-only view of the server's data directory so index files can
            # be sized exactly, rather than inferred from a catalog that does
            # not track companion tables.
            volumes.append(f"{self.volume}:/server-data:ro")
            data_dir_arg = ["--server-data-dir", mount_point]

        db_user, db_password = engines_mod.get(self.engine).credentials

        # The run directory is mounted at /results inside the client container,
        # so the harness must be given a container path. Passing the host path
        # would make Recorder create that directory inside the container and
        # write the records into a filesystem that disappears on exit.
        container_output = "/results/" + os.path.basename(output_path)

        command = [
            "/opt/harness/main.py",
            "--engine", self.engine,
            # Read from config/engines/*.yml here, because the harness container
            # mounts harness/ only and cannot read it for itself.
            "--driver", engines_mod.get(self.engine).driver,
            "--user", db_user,
            "--password", db_password,
            "--host", self.server_name,
            "--port", str(self.port),
            "--dataset", self.dataset,
            "--datasets-dir", "/datasets",
            "--run-id", self.run_id,
            "--resource-pass", self.resource_pass,
            "--output", container_output,
            "--cache-dir", "/results/.cache",
            *data_dir_arg,
            *args,
        ]

        spec = docker_ctl.ContainerSpec(
            name=self.client_name,
            image=image,
            network=self.network,
            cpuset=self.resolved.client_cpuset,
            # The ops client loads the corpus once and then computes ground
            # truth over it by brute force, which needs working space on top.
            # A fixed client_limit_gb is fine at 100 dimensions and far too
            # small at 1536, where the corpus alone is 6 GB — see the same
            # failure mode in ann_pass.client_memory_bytes().
            memory_bytes=max(self.resolved.client_memory_bytes,
                             ops_client_memory_bytes(self.paths['datasets'], self.dataset)),
            entrypoint="python3",
            workdir="/opt",
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/opt",
                "VB_DB_USER": db_user,
                "VB_DB_PASSWORD": db_password,
                "VB_ENGINE_TAG": str(self.engine_cfg.get("source", {}).get("tag", "")),
            },
            volumes=volumes,
            command=command,
            detach=False,
        )

        watcher = self._profile_watcher(image, spec)
        sampler = None
        if memory_timeseries:
            sampler = docker_ctl.MemorySampler(self.server_name, memory_timeseries)
            sampler.start()
        try:
            captured: List[str] = []
            rc = docker_ctl.run_foreground(spec, timeout=timeout_s,
                                           sink=captured, line_filter=watcher)
            saved = docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, self.dataset,
                f"ops-{self.tag}", self.resource_pass, captured)
            if saved:
                print(f"[ops] log -> "
                      f"{os.path.relpath(saved, self.paths['run_dir'])}")
            return rc
        finally:
            # In the finally, not on the success path: a phase that timed out
            # or crashed is exactly the one whose server log is worth having.
            self._save_server_log()
            if sampler is not None:
                sampler.stop()
                print(f"[ops] captured {sampler.samples} memory samples "
                      f"-> {os.path.basename(memory_timeseries)}")

    def _profile_watcher(self, bench_image: str,
                         spec: "docker_ctl.ContainerSpec"
                         ) -> Optional[ProfileWatcher]:
        """The perf `line_filter` for this unit, or None when profiling is off.

        Also tells the harness where to wait for the ack, by appending to the
        spec's command -- the only place the container path is known.
        """
        if self.profile_window_clients is None:
            return None
        host_ack = os.path.join(self.paths["ops_results"], PROFILE_ACK_SUBDIR)
        os.makedirs(host_ack, exist_ok=True)
        spec.command = [*spec.command, "--profile-ack-dir",
                        f"/results/{PROFILE_ACK_SUBDIR}"]
        stem = f"{self.engine}-{self.dataset}-{self.resource_pass}-{self.tag}"
        return ProfileWatcher(
            self.server_name, os.path.join(self.paths["run_dir"], "profiles"),
            stem, host_ack,
            meta={"engine": self.engine, "dataset": self.dataset,
                  "resource_pass": self.resource_pass, "unit": self.tag,
                  "run_id": self.run_id},
            bench_image=bench_image)

    def run_script(self, script: str, args: List[str],
                   interactive: bool = False,
                   client_memory_bytes: Optional[int] = None,
                   timeout_s: int = 4 * 3600,
                   artifacts_dir: Optional[str] = None) -> int:
        """Run one script from scripts/ against the running server.

        The diagnostic counterpart to run_harness. Same two-container shape,
        same server flags, same cpuset and memory -- because a probe that
        answers a question about a run has to run against the configuration
        that run used. The first version of this was a standalone shell script
        that started its own Valkey with hand-written flags, and it was already
        drifting: a 32 GB default where the run gives 101 GB, and
        maxmemory-policy re-specified by hand next to the copy in
        config/engines/valkey.yml. Two places to state one fact is how a probe
        ends up disproving something the run never did.

        What is deliberately *not* here is anywhere to write a measurement.
        run_harness mounts the run directory at /results; this mounts no
        writable path at all. A lab script generates its own vectors and its
        own ground truth, and those numbers must never be able to reach a
        report and sit in a table beside corpus numbers looking like a
        measurement.

        `artifacts_dir` IS THE ONE EXCEPTION AND IT IS READ-ONLY, which is
        what keeps the paragraph above true. A lab script that has to open a
        core dump cannot generate one, so it needs to be shown something a run
        already produced; mounted `:ro` that is a read of an existing artefact
        and not a route for a made-up number to reach a report. `:rw` here is
        exactly the defect the paragraph above describes.

        IT IS AN EXPLICIT PATH AND NOT THE SESSION'S RUN DIRECTORY, which is
        the version that does not work. A lab session's `run_dir` is
        `state/lab/<session>`, which `cmd_lab` has just created and which is
        empty by construction; a core lives in `results/<run-id>/cores/...`,
        under a run id the lab session cannot name. So the caller supplies the
        path, and validating it is the caller's job -- `cmd_lab` is where
        `VB_ROOT` and the operator's argv are both in scope.
        """
        image = self.engine_cfg.get("image", {}).get(
            "bench", f"vector-bench/{self.engine}-bench"
        )
        if not docker_ctl.image_exists(image):
            raise docker_ctl.DockerError(
                f"image {image} not found. Build it first:\n"
                f"  ./run-benchmark.sh build --engines {self.engine}"
            )

        db_user, db_password = engines_mod.get(self.engine).credentials
        volumes = [
            # Mounted rather than baked into the image, so editing a probe and
            # re-running it costs nothing. The whole point of the lab is a
            # turnaround measured in seconds.
            f"{self.paths['harness']}:/opt/harness:ro",
            f"{self.paths['scripts']}:/opt/scripts:ro",
            # Read-only, and only so a script *can* use the real corpus. Most
            # generate their own; the ones that need dbpedia should not have to
            # be run from inside a benchmark to reach it.
            f"{self.paths['datasets']}:/datasets:ro",
        ]
        if artifacts_dir:
            volumes.append(f"{artifacts_dir}:/artifacts:ro")

        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/opt",
            "VB_ENGINE": self.engine,
            "VB_DB_USER": db_user,
            "VB_DB_PASSWORD": db_password,
            "VB_HOST": self.server_name,
            "VB_PORT": str(self.port),
        }

        if interactive:
            entrypoint, command = "bash", []
        else:
            # --host and --port are passed for every script, so the contract a
            # lab script signs is those two flags. Credentials go through the
            # environment instead: an engine that needs them differs per
            # engine, and a script that does not take --user must still be
            # runnable.
            entrypoint = "python3"
            command = [f"/opt/scripts/{script}",
                       "--host", self.server_name,
                       "--port", str(self.port), *args]

        spec = docker_ctl.ContainerSpec(
            name=self.client_name,
            image=image,
            network=self.network,
            cpuset=self.resolved.client_cpuset,
            memory_bytes=client_memory_bytes or self.resolved.client_memory_bytes,
            entrypoint=entrypoint,
            workdir="/opt",
            env=env,
            volumes=volumes,
            command=command,
            detach=False,
        )

        try:
            if interactive:
                return docker_ctl.run_interactive(spec)
            return docker_ctl.run_foreground(spec, timeout=timeout_s)
        finally:
            # Same reasoning as run_harness: the server's half of the story is
            # worth most precisely when the client's half ended badly.
            self._save_server_log()

    def _save_server_log(self) -> None:
        """Archive what the server said, not only what the client saw.

        Only the client's half was ever kept. A Valkey churn then stalled with
        the client blocked on a socket and the server at its idle CPU
        baseline, and the question of which one was wrong could not be answered
        from the run directory at all -- the half that would have said was
        discarded when the container was removed.

        THE ANN PATH DOES NOT GET THIS FOR FREE, and that gap is still open.
        This docstring used to say it did, because the engine and the
        benchmark share one container there. They do, but
        `overlay/.../vb_mysql.py` starts the engine with
        `Popen([ENTRYPOINT, "server"])` -- the same `vb-entrypoint`, since
        every bench image is `FROM runtime AS bench` -- so mysqld gets the
        same `--log-error=<file>` and that stream carries no more of the
        server than this one did. What the ann path has instead is
        `run_engine`'s read-write bind of `state_dir` at the engine's
        `data_mount`, so the file lands on the host: root-owned, keyed by
        (pass, engine) and NOT by dataset, wiped by the next dataset's
        `rmtree`. Nothing copies it into the run directory yet.

        TWO CAPTURES, because for five of the seven engines `docker logs` is
        not all of the server's voice, and for four of those it is almost none
        of it. `mariadb`, `mariadb123`, `alisql` and `percona` are started
        with `--log-error=<file>`, so their stdout is the entrypoint banner --
        three lines -- and everything the server says goes into a file in the
        data volume that teardown deletes. That is the half this method was
        added to keep, and for those engines it was capturing the wrong one;
        `config/engines/percona.yml`'s `error_log` records what that cost in
        run full-20261015-215006. `mongodb` is the fifth and a partial case:
        `mongod` does log to stdout, but `mongot`, where vector search
        actually happens, is redirected into the volume -- so both captures
        carry something. `runtime.error_log` names the file;
        `_save_server_error_log` copies it out before teardown -- through
        `docker exec` while the container is up, and `docker cp` once it is
        not, which is the case that matters when the server died rather than
        merely failed its probe.

        FOUR CAPTURES NOW, NOT TWO, and the order is deliberate. The container
        exit facts go first of the three that need the daemon, because they are
        the cheapest -- one `docker inspect`, bounded at 30 s -- and the only
        one that cannot be reconstructed from anything else; a second Ctrl-C
        during the 510 s the error-log chain can take should not cost them. The
        core copy goes last because it is the most expensive and the least
        likely to find anything.

        ONE INSPECT, SHARED. The exit state is read here and passed to both
        callees rather than fetched twice. It is also what gates the core
        lookup, and that gate is load-bearing rather than tidy: this method
        runs from `run_harness`'s UNCONDITIONAL `finally`, so without it every
        SUCCESSFUL phase would spawn a container to search a live datadir and
        print a note about a core that was never going to exist --
        `--profile full` is 24 percona ops units, so 24 needless container
        starts and 24 misleading notes on a healthy run. `exit_state` is None
        while the server is still up, which is exactly the "nothing crashed
        here" signal.

        Independent of each other, and none of them can end the run: a server
        whose stdout is empty is exactly the one whose file log matters, so an
        exception from any of them must not skip the rest.
        """
        try:
            text = docker_ctl.logs(self.server_name, tail=SERVER_LOG_TAIL)
        except Exception:
            text = ""
        if text:
            docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, self.dataset,
                f"server-{self.tag}", self.resource_pass, text.splitlines())
        try:
            exit_state = docker_ctl.exit_state(self.server_name)
        except Exception:                        # pragma: no cover - defensive
            exit_state = None
        self._save_container_exit_facts(exit_state)
        self._save_server_error_log()
        self._save_server_core(exit_state)

    def _save_container_exit_facts(
            self, state: Optional[Dict[str, Any]]) -> None:
        """Record how the server container ended, before it is removed.

        NOT AS RECORD FIELDS, and that is a decision rather than a shortcut.
        The case these facts exist for is a phase that wrote NO RECORD AT ALL
        -- twelve of run full-20261015-215006's twenty-four percona ops units
        are empty files -- so an `extra` key would be absent in exactly the
        case it was added for. They go into the run directory beside the two
        log captures, through the same `save_phase_log`, whose filename carries
        the dataset for the reason that function's docstring gives.

        It also keeps the measurement version where it is: a run-directory
        artefact is not an ops number, so nothing a reader already trusts
        changes meaning and `OPS_MEASUREMENT_VERSION` does not move.

        MUST RUN BEFORE TEARDOWN and does, for the same reason
        `_save_server_error_log` does: every caller reaches it from a `finally`
        inside a `with OpsRun(...)` block, and `teardown` -- whose second call
        is `docker rm` -- is `__exit__`. `docker inspect` needs the container to
        EXIST; after `remove` there is nothing to ask.

        NOTHING HERE MAY RAISE, the standard `_save_server_error_log` sets.
        `state` is whatever `_save_server_log`'s single inspect produced, and
        None for every daemon-side failure, so the `try` is for the write.
        """
        try:
            if not state:
                # No file rather than a file saying nothing: `export` ships
                # the run directory and a reader spends time on whatever is
                # in it. A container that never existed -- the missing-image
                # path -- is the usual way to get here.
                return
            lines = [
                f"container: {self.server_name}",
                f"engine: {self.engine}",
                f"dataset: {self.dataset}",
                f"phase: {self.tag}",
                f"resource_pass: {self.resource_pass}",
                f"exit_code: {state['exit_code']}",
                # The field this artefact exists for. Printed with its value
                # on the same line and in words, because "OOMKilled" alone
                # reads as an assertion that it was.
                f"oom_killed: {state['oom_killed']} "
                f"({'the kernel killed it' if state['oom_killed'] else 'not a cgroup OOM'})",
                f"finished_at: {state['finished_at']}",
                f"daemon_error: {state['error'] or '(none)'}",
            ]
            saved = docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, self.dataset,
                f"server-exit-{self.tag}", self.resource_pass, lines)
            if saved:
                print(f"[ops] server exit facts -> "
                      f"{os.path.relpath(saved, self.paths['run_dir'])}"
                      f" (exit {state['exit_code']}, "
                      f"OOMKilled={state['oom_killed']})")
        except Exception as exc:                 # pragma: no cover - defensive
            print(f"[ops] note: could not record {self.engine}'s container "
                  f"exit state: {exc}", file=sys.stderr)

    def _save_server_core(self, state: Optional[Dict[str, Any]]) -> None:
        """Copy a core dump out of the data volume before teardown deletes it.

        THE DEFECT THIS EXISTS FOR IS ORDERING, not absence. `teardown` stops
        the container, removes it, removes the network, removes the volume and
        then `remove_tree_as_root`s the host directory the volume was backed
        by -- and `__exit__` is `teardown`. A core written into the engine's
        data mount is deleted with the run that produced it, so there is no
        second chance after this point and no amount of later tooling can
        recover one.

        ONLY FOR AN ENGINE THAT ASKED FOR A CORE. `_save_server_error_log`
        sets the precedent and its reason transfers exactly: looking costs a
        container, and an engine whose config declares no core ulimit was never
        asked to dump, so for the other six this is a no-op and their
        behaviour is unchanged.

        AND ONLY IF THE CONTAINER HAS ACTUALLY ENDED, which is what `state`
        being non-None means. The caller reaches this from `run_harness`'s
        UNCONDITIONAL `finally`, so the alternative is one container start and
        one "no core found" note per SUCCESSFUL phase as well -- 24 of each on
        `--profile full`, every one of them noise, over a datadir the server is
        still writing to. A server that is still running has not dumped.

        AND ONLY IF THE BIND DIRECTORY IS THERE. `remove_tree_as_root`'s first
        line is the same check and gives the reason: when setup failed before
        the server wrote anything, the directory is one this process made and
        spawning a container to look inside it would need the very image whose
        absence may have caused the failure.

        WHERE IT LANDS: `<run_dir>/cores/<engine>-<dataset>-<phase>-<pass>/`,
        the same quadruple `save_phase_log` keys its filenames on and for the
        same reason -- one engine, one phase and one resource pass is not a
        unit of work, and a four-dataset profile would otherwise overwrite
        three of four.

        AND WHAT THAT COSTS `export`, which is not handled here and should be
        known before it surprises someone. `orchestrator/export.py`'s
        `write_bundle` tars the WHOLE run directory, so a run holding a core
        produces a bundle carrying it -- gigabytes, in a thing whose stated
        job is to be "sent to someone who does not have this checkout".
        `cmd_export` prints the size, so it is visible, but only afterwards. A
        core is also the wrong artefact to send: it is specific to this
        machine and this build, and what the server team can act on is the
        BACKTRACE taken from it. Excluding `cores/` from the bundle is the
        right fix and belongs to whoever owns `export.py`.

        NOTHING HERE MAY RAISE. `copy_glob_as_root` returns `[]` for every
        failure; the `try` is for the registry lookup and the path arithmetic.
        """
        try:
            if "core" not in self._ulimits():
                return
            if not state:
                return
            bind_dir = os.path.join(self.paths["engine_state"], "ops",
                                    self.volume)
            if not os.path.isdir(bind_dir):
                return
            dest = os.path.join(
                self.paths["run_dir"], "cores",
                f"{self.engine}-{self.dataset}-{self.tag}-"
                f"{self.resource_pass}")
            image = self.engine_cfg.get("image", {}).get(
                "runtime", f"vector-bench/{self.engine}-runtime")
            copied = docker_ctl.copy_glob_as_root(
                bind_dir, CORE_FILE_GLOB, dest, image)
            if not copied:
                # A note rather than silence, and it names the two reasons a
                # core is absent that are not "the server did not crash", so
                # an operator does not read the empty directory as evidence.
                print(f"[ops] note: no {CORE_FILE_GLOB} found in "
                      f"{self.engine}'s data volume. Either the server did "
                      f"not dump, or kernel.core_pattern sent the core "
                      f"somewhere this cannot reach -- the value is printed "
                      f"at server start.", file=sys.stderr)
                return
            for path in copied:
                print(f"[ops] server core -> "
                      f"{os.path.relpath(path, self.paths['run_dir'])} "
                      f"({os.path.getsize(path) / 1024**3:.2f} GiB)")
        except Exception as exc:                 # pragma: no cover - defensive
            print(f"[ops] note: could not archive {self.engine}'s core: "
                  f"{exc}", file=sys.stderr)

    def _save_startup_log(self) -> None:
        """`_save_server_log`, but only once there is a container to ask.

        Setup fails in three places and only the last has started a server: a
        missing image raises before `docker run`, and `create_network` /
        `create_volume` before that. `container_exists` is what rules those
        out, and its own docstring says why asking is necessary.

        Cannot raise: that inspect is `check=False`, and `_save_server_log`
        swallows everything underneath it.
        """
        try:
            if not docker_ctl.container_exists(self.server_name):
                return
        except Exception:                        # pragma: no cover - defensive
            return
        self._save_server_log()

    def _read_error_log(self, error_log: str) -> Optional[str]:
        """The tail of the engine's error log, however the container is doing.

        TWO MECHANISMS, because `wait_healthy` fails two ways and only one
        leaves a container `docker exec` can enter: its probe timeout, where
        the server is still RUNNING and merely never answered, and
        `is_running()` false -- "container ... exited during startup" -- where
        it DIED, which is the case whose last words are worth most.
        `docker_ctl.copy_file_out` covers the second and says how.

        Reading the host side of the bind-backed volume needs no docker at all
        and is deliberately NOT the fallback: the engine writes that file as
        root inside the container, so it is typically root:root 0640 and
        unreadable to a non-root orchestrator -- the thing
        `docker_ctl.remove_tree_as_root` exists for and branches on
        `os.getuid() == 0` for.

        RETURNS None FOR EVERYTHING, like the two functions it calls, which is
        why the `mkdtemp` is inside a try: an unwritable scratch location is
        one more way to have no log.

        The copy lands under `engine_state`, not TMPDIR, for the reason
        `docker_ctl.create_volume` gives about where the space is -- it is
        unbounded by construction, and mongodb's log grows all phase.
        """
        text = docker_ctl.read_file_tail(self.server_name, error_log,
                                         lines=SERVER_LOG_TAIL)
        if text:
            return text
        try:
            # Beside the corpus, on the filesystem sized for it.
            parent = self.paths["engine_state"]
            os.makedirs(parent, exist_ok=True)
            scratch = tempfile.mkdtemp(prefix="vb-errlog-", dir=parent)
        except (OSError, KeyError):
            return None
        try:
            copied = docker_ctl.copy_file_out(self.server_name, error_log,
                                              scratch)
            if not copied:
                return None
            with open(copied, errors="replace") as fh:
                lines = fh.read().splitlines()
            print(f"[ops] {self.engine} is no longer running; its error log "
                  f"was copied out rather than read in place")
            # Bounded HERE, because the copy could not be: every other
            # capture is capped at this figure and `export` ships the lot.
            return "\n".join(lines[-SERVER_LOG_TAIL:]).strip() or None
        except OSError:
            return None
        finally:
            # Once per failed phase, holding a FULL copy of the log: nine
            # failures in the reference run would be nine of these.
            shutil.rmtree(scratch, ignore_errors=True)

    def _save_server_error_log(self) -> None:
        """Copy the engine's own `--log-error` file out before teardown.

        Only for an engine whose config declares `runtime.error_log`; the
        engines that log to stderr are already covered by `docker logs` above
        and this is a no-op for them, so their behaviour is unchanged.

        Must run BEFORE teardown and does, because every caller reaches it from
        a `finally` inside a `with OpsRun(...)` block, and `teardown` -- which
        stops the container and then removes the volume the file lives in -- is
        `__exit__`. There is no second chance after that: the file is not on
        the host, and the volume is bind-backed under state/ops, which
        `remove_tree_as_root` then deletes as well.

        NOTHING HERE MAY RAISE. This is on the path a crashed phase takes, and
        a phase that has already lost its measurement must not also lose the
        run because its log could not be fetched. Every failure -- no such
        engine, no such file, an unreadable path, a container that has gone, no
        daemon -- degrades to a printed note and no file. `_read_error_log`
        returns None for all of them; the try is for the registry lookup and
        the write.
        """
        try:
            error_log = engines_mod.get(self.engine).error_log
            if not error_log:
                return
            text = self._read_error_log(error_log)
            if not text:
                print(f"[ops] note: {self.engine} declares "
                      f"runtime.error_log={error_log} but nothing could be "
                      f"read from it; the server's own log is not in this run "
                      f"directory", file=sys.stderr)
                return
            saved = docker_ctl.save_phase_log(
                self.paths["run_dir"], self.engine, self.dataset,
                f"server-error-{self.tag}", self.resource_pass,
                text.splitlines())
            if saved:
                print(f"[ops] server error log -> "
                      f"{os.path.relpath(saved, self.paths['run_dir'])}")
        except Exception as exc:                 # pragma: no cover - defensive
            print(f"[ops] note: could not archive {self.engine}'s error log: "
                  f"{exc}", file=sys.stderr)

    # ------------------------------------------------------------------

    def teardown(self) -> None:
        # Stop the server before removing the volume, or the removal races the
        # engine's own shutdown flush and leaves a dangling volume behind.
        docker_ctl.stop(self.server_name, timeout_s=120)
        docker_ctl.remove(self.server_name)
        docker_ctl.remove(self.client_name)
        docker_ctl.remove_network(self.network)
        docker_ctl.remove_volume(self.volume)
        # Removing a bind-backed volume leaves the host directory behind, so
        # without this every configuration would leak a full copy of the corpus
        # and the index onto disk.
        # Root-owned: the engine wrote it from inside the container, so this
        # has to go through a container too. shutil.rmtree cannot touch it and
        # fails silently, leaving a full corpus and index on disk.
        bind_dir = os.path.join(self.paths["engine_state"], "ops", self.volume)
        image = self.engine_cfg.get("image", {}).get(
            "runtime", f"vector-bench/{self.engine}-runtime")
        if not docker_ctl.remove_tree_as_root(bind_dir, image):
            print(f"[ops] WARNING: {bind_dir} survived teardown and is still "
                  f"using disk", file=sys.stderr)


def _quantization(profile: Dict[str, Any], resources: Dict[str, Any],
                  resource_pass: str) -> str:
    """Which quantization the ops build should use.

    The same rule render_config applies to the recall path: pinned off in the
    normalized pass so no engine gets an axis the others lack, and the vendor's
    recommendation in the tuned pass. Reading it in only one of the two paths
    meant a tuned run measured a quantized index for recall and an unquantized
    one for build cost and every ops workload, then reported them side by side
    as one configuration.
    """
    if resource_pass == "tuned":
        values = (resources.get("extras", {}) or {}).get("mongodb_quantization")
        if values:
            return str(list(values)[0])
    return str((profile.get("ann", {}) or {}).get("mongodb_quantization", "none"))


#: What `ops.concurrency_queries` means when a profile does not set it: every
#: query the dataset ships (10,000 for glove-100 and sift-128). The harness
#: reads 0 as a sentinel, not as a cap — see `_query_pool` in
#: harness/workloads/concurrency.py. The default is the new behaviour and the
#: way back is to set the key, which is the convention every switch here
#: follows.
DEFAULT_CONCURRENCY_QUERIES = 0


#: The ef policy, as `harness/workloads/calibrate.py` defines it. Declared
#: again here rather than imported, and that is deliberate: this module runs on
#: the host, whose only Python dependency is pyyaml, and
#: `harness.workloads.calibrate` reaches numpy through `harness.datasets`. The
#: same reason `cli.py` imports `harness.datasets` inside a function body
#: instead of at module scope. `tests/test_calibrate.py` pins the two
#: declarations together, which is how `ops.selectivities` and
#: `ops.churn_fractions` are already handled.
DEFAULT_EF_MODE = "calibrated"
DEFAULT_TARGET_RECALL = 0.95
DEFAULT_EF_CALIBRATION = (10, 16, 24, 32, 48, 64, 100, 160, 256, 400, 640,
                          1000)


def ops_target_recall(profile: Dict[str, Any]) -> float:
    """The recall the ops pass calibrates ef_search to reach.

    One reader for two consumers that must agree: `harness_args`, which hands
    it to `harness/workloads/calibrate.py` as `--target-recall`, and
    `ops_build_selection`, which picks the frontier build that serves this
    recall fastest. If they read the profile separately, a profile that set
    `ops.target_recall: 0.99` would pick the build fastest at 0.95 and then
    calibrate it to 0.99 -- a build chosen for a quality the ops pass never
    measures at.
    """
    ops = profile.get("ops", {}) or {}
    return float(ops.get("target_recall", DEFAULT_TARGET_RECALL))


def _concurrency_queries(ops: Dict[str, Any]) -> int:
    """How many queries the concurrency clients may draw from.

    Its own function rather than an inline `.get` because the value does not
    come from `ops.max_queries` and the reason is not obvious: that key caps
    the filtered and churn workloads, which answer each query once, and
    main.yml holds it at 500 to bound their cost. The concurrency workload
    cycles its shard for a fixed duration, so sharing the key gave 32 clients
    fifteen queries each and measured every engine's cache against a hot set
    that fitted in it.
    """
    return int(ops.get("concurrency_queries", DEFAULT_CONCURRENCY_QUERIES))


#: Seconds per (column, selectivity, iterative-scan mode) point in the
#: filtered workload. Declared here as well as in
#: `harness.workloads.filtered` for the reason DEFAULT_EF_MODE is: this module
#: runs on the host, whose only Python dependency is pyyaml, and the workload
#: reaches numpy through `harness.datasets`. `tests/test_t3c_filtered.py` pins
#: the two declarations together, as `tests/test_calibrate.py` already does
#: for the ef policy.
DEFAULT_FILTERED_BUDGET_S = 300

#: Seconds one churn re-insertion may spend. Matches
#: `harness/workloads/churn.py:DEFAULT_CHURN_BUDGET_S`, which is not
#: importable here -- the orchestrator runs on the host and the harness runs
#: in a container -- and was a bare literal in the argument list until the
#: `null` fix needed to pass it to `_budget_arg` as a default.
DEFAULT_CHURN_BUDGET_S = 7200

#: The filter columns the filtered workload measures when a profile says
#: nothing. Both, matching `harness.datasets.FILTER_COLUMNS`, and pinned
#: against it by the same test.
DEFAULT_FILTER_COLUMNS = ("tag", "ctag")

#: Readers querying the index while the churn writer re-inserts, per PR 4
#: §4.3 item 2. Matches `harness.workloads.churn_readers`'
#: `DEFAULT_CHURN_READERS`, which is not importable here -- the orchestrator
#: runs on the host and the harness runs in a container -- and is pinned
#: against it by `tests/test_t4b_readers.py`, the way `tests/test_calibrate.py`
#: already pins the ef policy and `tests/test_t3c_filtered.py` the filtered
#: budget.
#:
#: Emitted unconditionally, including the default, for the reason `--ef-mode`
#: and `--filtered-budget` are: the harness command line in a run's phase log
#: should say what read load the churn write rate was measured against rather
#: than leaving a reader to infer it from whatever the workload's default was
#: at the time. `ops.churn_readers: 0` is the write-only rate and is a real
#: choice, not an absence.
#:
#: NOT run through `_budget_arg`. That function turns a `null` into "0", which
#: is right for a budget -- null and 0 both mean "no cap" -- and wrong here: a
#: reader count of 0 means "no readers", so `null` would silently become
#: "turn the pool off" rather than "use the default". A profile that wants no
#: readers writes 0.
DEFAULT_CHURN_READERS = 4


def _budget_arg(ops: Dict[str, Any], key: str, default: Any) -> str:
    """A budget key as a command-line value, with YAML `null` meaning no cap.

    `str(ops.get(key, default))` -- the idiom every other numeric key here
    uses -- turns a key PRESENT with value null into the string "None", which
    argparse rejects with `invalid float value: 'None'` and `SystemExit(2)`.
    That kills the whole ops unit, not just the workload: `harness/main.py`
    wraps concurrency, filtered and churn in one `except`, so a reader who
    wrote `filtered_budget_s: null` to turn the cap off lost two other
    workloads with it.

    `null` and `0` both mean no cap, which is what `_filtered_search` in the
    estimate has always assumed (it treats `budget > 0` as the only bounded
    case) and what the workload now implements.

    Applied to `ops.churn_budget_s` as well, as of T4a. T3c left that one
    alone on purpose: the enforcement lives in `harness/workloads/churn.py`
    and fixing the emission without it would have moved the failure from
    argparse into a workload whose first deadline check fires immediately,
    which is the shape that made every filtered point record ZERO queries.
    `churn.run` now converts a non-positive budget into an infinite deadline,
    so both halves are in place and both spellings mean no cap.

    One place still raises on `null` and it is not reachable from here:
    `orchestrator/estimate.py:_churn_reinsert` reads
    `float(plan.ops.get("churn_budget_s", 1800))` with no `or 0.0`, so the
    estimate -- which runs on the host, before the run starts -- dies with
    `TypeError: float() argument must be a string or a real number, not
    'NoneType'`. `_filtered_search` two functions above it has the `or 0.0`.
    Fixing that is a one-line change in a file T4a does not own.
    """
    value = ops.get(key, default)
    return "0" if value is None else str(value)


def _churn_readers(ops: Dict[str, Any]) -> int:
    """How many readers run while the churn writer re-inserts.

    Its own function rather than an inline `.get` for the same reason
    `_concurrency_queries` is one: a `null` here must not mean what a `null`
    in a budget key means. `ops.churn_budget_s: null` means "no cap" and
    `_budget_arg` turns it into "0"; `ops.churn_readers: null` is a profile
    that cleared the key and wants whatever the default is, and emitting "0"
    for it would quietly turn the reader pool off -- which changes what
    `extra.reinsert_rows_per_s` measures while the record still looks the
    same. A profile that wants the write-only rate writes 0 explicitly.
    """
    value = ops.get("churn_readers", DEFAULT_CHURN_READERS)
    return int(DEFAULT_CHURN_READERS if value is None else value)


def _filter_columns(ops: Dict[str, Any]) -> List[str]:
    """The filter columns to measure, as the harness spells them.

    Its own function because a profile may legitimately narrow this -- a
    reproduction of a pre-PR-3 run wants `[tag]` alone -- and because an empty
    or absent key must mean "both" rather than "none". `[]` in YAML is falsy,
    and a profile that wrote it would otherwise silently measure no filtered
    point at all while the phase still reported success.
    """
    declared = ops.get("filter_columns")
    if not declared:
        return list(DEFAULT_FILTER_COLUMNS)
    return [str(column) for column in declared]


def _yaml_mode_name(value: Any) -> str:
    """The name a YAML scalar was written as, for keywords YAML 1.1 steals.

    `iterative_scan: [off, relaxed_order]` does not parse to the strings it
    looks like: YAML 1.1 reads a bare `off` as boolean False, so the list comes
    back as `[False, "relaxed_order"]`. `str(False)` is "False", which is not a
    pgvector mode and would reach the server as
    `SET hnsw.iterative_scan = False`.

    It lives here rather than in `cli.py`, where it was written, because two
    modules now have to name the same modes the same way -- see
    `_declared_scan_modes` directly below for which two and why a drift
    between them would matter.
    """
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


def _declared_scan_modes(declared: Any) -> List[str]:
    """The iterative-scan modes a profile or extras block declares, named.

    Two jobs, and both have to happen identically wherever the count matters:
    YAML 1.1 name recovery (`_yaml_mode_name` above) and an order-preserving
    de-duplication, because `[off, off]` asks for one measurement and not two
    -- the same rule `harness/workloads/filtered.py:_scan_modes` applies on
    the far side of the command line, so agreeing with it here keeps the
    orchestrator's count and the workload's loop the same length.

    Shared rather than written twice because two readers have to agree:
    `cli.py:_ops_iterative_scan` decides what a run measures and
    `estimate.py:_filtered_search` charges for it. A drift between those two
    is an estimate that is wrong about the run it describes, which is the one
    thing that estimator exists to avoid -- and the precedent is in this file
    already, where `DEFAULT_FILTERED_BUDGET_S` is pinned against the
    workload's own default by a test for exactly this reason.

    Validation is deliberately NOT here. It needs the engine's own
    `capabilities.iterative_scan_modes`, and only one of the two callers can
    sensibly load an engine config; see `cli.py:_ops_iterative_scan`.
    """
    modes: List[str] = []
    for value in declared or []:
        mode = _yaml_mode_name(value)
        if mode and mode not in modes:
            modes.append(mode)
    return modes


def _iterative_scan_modes(iterative_scan) -> Optional[str]:
    """The `--iterative-scan` value, from one mode or several.

    A plain string passes through verbatim, which is what keeps a single-mode
    call site emitting exactly what it asked for; a sequence is joined with
    commas and the workload splits it again. `str` is checked first on purpose
    -- a string IS a sequence, and joining one gives "o,f,f".

    **The sequence branch names its modes through `_yaml_mode_name` rather than
    `str`,** which it did not until now. `str(False)` is "False", so a sequence
    straight out of YAML -- `[off, relaxed_order]` parses as
    `[False, "relaxed_order"]` -- would have emitted `--iterative-scan
    False,relaxed_order` and put `SET hnsw.iterative_scan = False` on the wire.
    "Would have", and not "did": that is the exact defect `_yaml_mode_name` was
    written for, but it never fired here, because the only production caller --
    `cli.py:_ops_iterative_scan` -- names and comma-joins its modes first and so
    reaches the string branch above. Nothing guarded it;
    `tests/test_t3c_filtered.py` now does. The function is cheap to call
    correctly and `_yaml_mode_name` lives in this module, so the branch no
    longer depends on its callers having been careful.
    """
    if iterative_scan is None:
        return None
    if isinstance(iterative_scan, str):
        return iterative_scan or None
    modes = [_yaml_mode_name(mode) for mode in iterative_scan]
    return ",".join(mode for mode in modes if mode) or None


def harness_args(profile: Dict[str, Any], m: int, engine: str,
                 resolved: ResolvedResources,
                 resource_pass: str, resources: Dict[str, Any],
                 build_mode: str = "post",
                 storage_engine: str = "InnoDB",
                 iterative_scan: Optional[Union[str, Sequence[str]]] = None,
                 bulk_vector_index_build: bool = True,
                 vector_index_replace: bool = True,
                 build: Optional[Dict[str, Any]] = None,
                 build_selection: Optional[Dict[str, Any]] = None
                 ) -> List[str]:
    """Translate a profile into ops-harness command-line arguments.

    `bulk_vector_index_build` is the engine's own declaration, read by
    `cli.py:_bulk_vector_index_build`, and its default here is True for the
    same reason the default is True there: absence of the declaration has to
    mean today's behaviour. The only call site derives it from the engine
    config, so the default is reached by hand-built calls and by tests, both
    of which are about engines whose bulk build works.

    `vector_index_replace` is the churn repair's own reading,
    `cli.py:_vector_index_replace`, which falls back to the bulk-build
    reading when an engine declares nothing separate. Its default is True for
    the same reason. It is a SEPARATE parameter, and a separate flag, because
    the two statements stopped sharing a verdict: at Percona's current pin
    the bulk build is verified and the index replace is not.

    `iterative_scan` takes one mode or several. One is emitted verbatim and
    several are comma-joined for the workload to loop, which is the shape
    PR 3 §3.4 item 4 costs out. Both branches are now live: the only call site
    in `cli.py` passes every declared mode that validates, so `main` and
    `full` both reach this with `off,relaxed_order` and a profile naming one
    mode still reaches it with that mode alone. This end was wired a round
    before its caller was, which is why turning the caller on needed no change
    here.

    `build` is the build `ops.build_selection: frontier` chose off the ann
    phase (`ops_build_selection.ops_build`): its `ef_construction` and
    `quantization` replace the profile-derived values below when set, so the
    ops pass builds the index the frontier figure came from rather than the
    declared reference build. `build_selection` is the disclosure of how it
    was chosen, carried to every ops record as `extra.ops_build_selection`.
    Both are None in `reference` mode, which leaves this command line exactly
    what it was.
    """
    ops = profile.get("ops", {}) or {}
    build = build or {}
    quantization = (build.get("quantization")
                    or _quantization(profile, resources, resource_pass))

    args = [
        "--m", str(m),
        "--k", str(profile.get("k", 10)),
        "--ef-search", str(ops.get("ef_search", 100)),
        # The ef policy. Emitted unconditionally, including the default, so the
        # harness command line in a run's phase log says which policy produced
        # the numbers rather than leaving a reader to infer it from the
        # harness's own default at the time.
        "--ef-mode", str(ops.get("ef_mode", DEFAULT_EF_MODE)),
        "--target-recall", str(ops_target_recall(profile)),
        "--ef-grid", ",".join(
            str(int(ef)) for ef in
            (ops.get("ef_calibration") or DEFAULT_EF_CALIBRATION)),
        "--storage-engine", storage_engine,
        "--build-mode", build_mode,
        # Emitted only when the engine declares its bulk vector index build
        # BROKEN -- VillageSQL alone now, whose driver uses it only to name
        # the known crash before an explicitly requested `post` build. Like
        # `--no-vector-index-replace` just below, it is a conditional flag
        # that carries no value; `--quantization` further down is the
        # conditional one that does. The "emit unconditionally" rule
        # `--ef-mode`, `--churn-readers` and `--filtered-budget` follow is
        # about keys that carry a VALUE a profile chose, where an omitted flag
        # would leave a reader inferring the workload's default at the time.
        # This one is a declaration that few engine configs make, so
        # mirroring the config -- most engines say nothing, and absence there
        # reads as "the bulk build works" -- keeps one fact with one default
        # on both sides of the container boundary rather than two spellings
        # of it.
        #
        # **Neither flag is the readable artefact, and it would be wrong to
        # claim it is: NOTHING in this repo writes the harness argv to disk
        # or to the phase log.** What a reader actually sees is the driver's
        # own print -- `PerconaDriver.maintain()` for the repair,
        # `VillageSQLDriver.create_index` for the build -- which does reach
        # the phase log. The flags' job is only to carry the host's decision
        # across the container boundary.
        *([] if bulk_vector_index_build
          else ["--no-bulk-vector-index-build"]),
        # The churn repair's withdrawal, on its own reading. It used to ride
        # on `--no-bulk-vector-index-build`, which was right while both
        # statements failed at Percona's old pin and became wrong the moment
        # the bulk build was re-verified on its own: flipping one key would
        # have restored a repair that at the old pin was a 1-4 ms no-op the
        # report published as `after ALTER TABLE (index replace)`, and that
        # nobody has re-tested since. Same polarity and the same absence
        # rule as the flag above, for the same reason.
        *([] if vector_index_replace
          else ["--no-vector-index-replace"]),
        "--churn-budget", _budget_arg(ops, "churn_budget_s",
                                      DEFAULT_CHURN_BUDGET_S),
        # Read from the same extras render_config uses for the recall path, so
        # both measurement paths in one run build the same index. Only Percona
        # Search has the knob; the flag is omitted for everything else.
        *(["--quantization", quantization]
          if engine == "mongodb" else []),
        "--load-threads", str(ops.get("load_threads", 1)),
        "--max-queries", str(ops.get("max_queries", 1000)),
        "--workloads", ",".join(ops.get("workloads", ["build"])),
        "--client-counts", ",".join(str(c) for c in ops.get("client_counts", [1])),
        "--concurrency-duration", str(ops.get("concurrency_duration_s", 20)),
        "--concurrency-repeats", str(ops.get("concurrency_repeats", 1)),
        "--concurrency-queries", str(_concurrency_queries(ops)),
        "--selectivities", ",".join(str(s) for s in ops.get("selectivities", [0.1])),
        "--filter-columns", ",".join(_filter_columns(ops)),
        # Emitted unconditionally, including the default, for the reason
        # --ef-mode is: the harness command line in a run's phase log should
        # say what bounded the filtered loop rather than leaving a reader to
        # infer it from whatever the workload's default was at the time.
        "--filtered-budget", _budget_arg(ops, "filtered_budget_s",
                                         DEFAULT_FILTERED_BUDGET_S),
        "--churn-fractions", ",".join(str(c) for c in ops.get("churn_fractions", [0.1])),
        "--churn-readers", str(_churn_readers(ops)),
    ]
    if ops.get("subset_rows"):
        args += ["--subset-rows", str(ops["subset_rows"])]
    ef_construction = build.get("ef_construction")
    if ef_construction is None:
        ef_construction = ops_ef_construction(engine, profile)
    if ef_construction is not None:
        args += ["--ef-construction", str(int(ef_construction))]
    if build_selection:
        # One JSON argument rather than a flag per field: the harness only
        # stamps it, never interprets it, so it travels as the host wrote it.
        args += ["--ops-build-selection",
                 json.dumps(build_selection, sort_keys=True, default=str)]
    if engine == "pgvector":
        modes = _iterative_scan_modes(iterative_scan)
        if modes:
            args += ["--iterative-scan", modes]
    # Conditional, unlike `--ef-mode`: both are server-side evidence, off by
    # default, and an absent flag is exactly "off". The instruments flag is
    # emitted only for an engine that declares instruments, because only
    # there did the server change (`config.engine_wait_instruments`).
    if (ops_wait_instruments(profile)
            and engine_wait_instruments(_engine_cfg_or_empty(engine))):
        args += ["--server-wait-instruments"]
    if (ops_wait_sampler(profile)
            and engine_wait_sampler_interval(_engine_cfg_or_empty(engine))):
        args += ["--server-wait-sampler"]
    window = profile_window_clients(profile)
    if window is not None:
        args += ["--profile-window-clients", str(window)]
    return args


def _engine_cfg_or_empty(engine: str) -> Dict[str, Any]:
    from .config import load_engine
    try:
        return load_engine(engine)
    except Exception:
        return {}


def ops_ef_construction(engine: str, profile: Dict[str, Any]) -> Optional[int]:
    """The `--ef-construction` the ops pass hands the harness, or None.

    One function because two readers need the same answer: `harness_args`,
    which builds the index, and `config.resolve_reference_build`, which names
    that index as the build every summary figure is taken from. Two copies of
    the rule are how the 25 Aug report came to print pgvector's floor figures
    from an efC=400 build beside ops numbers measured at 200.

    None for every engine the ops pass leaves to its driver -- Valkey (the
    server default) and Percona (`PERCONA_DIALECT.ef_construction`) -- whose
    config declares the literal those defaults resolve to instead.
    """
    ann = profile.get("ann", {}) or {}
    if engine == "pgvector":
        return int(ann.get("pgvector_ef_construction", 200))
    if engine == "myvector":
        # Explicit, and the same key and default the ann render reads
        # (`ann_pass.render_config`'s myvector branch), so the two paths build
        # the graph at one ef_construction. Left to the driver it would be the
        # plugin's own default of 100, half what every other engine with the
        # knob is built at.
        return int(ann.get("myvector_ef_construction", 200))
    if engine in VILLAGESQL_ENGINES:
        # The same key and default `ann_pass.render_config`'s villagesql
        # branch reads (one key for both server lines), so both paths build
        # the graph at one ef_construction.
        # Never clamped: a combination outside [M, 32 x M] is refused before
        # the run (`reference_build.ops_build_problems`) and by the driver.
        return int(ann.get("villagesql_ef_construction", 200))
    return None
