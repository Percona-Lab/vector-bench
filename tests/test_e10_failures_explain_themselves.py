"""E10 -- a 42-hour run failed 9 of its 16 phases and the run directory could
not say why, for three separate reasons.

THE RUN: `full-20261015-215006`, percona only, `--profile full
--resource-pass both`, four datasets, 16 phases. What it left behind, all of
it checked by hand before this file was written:

  * 13 of its 24 `ops-percona-*.jsonl` files are zero bytes. Each of those
    units died inside the first INSERT with `mariadb.InterfaceError: Lost
    connection to server during query` from
    `harness/drivers/mysql_family.py`'s `_load_range`, within 0.5-5.5 s, with
    server RSS flat at the buffer-pool baseline and CPU at 0.16 cores.
  * All six `logs/percona-server-*.log` files are THREE LINES: the entrypoint
    banner. Whether the server took a signal, hit an assertion or refused the
    statement is not recoverable from that run, and cannot be.
  * `logs/` holds 14 files for 16 phases, and only ONE `percona-ann-*.log`
    per resource pass. The surviving `percona-ann-normalized.log` is gist's
    (`Got a train set of size (1000000 * 960)`); the sift-128 ANN phase had
    failed in six seconds in both passes and its output had been overwritten.
  * `percona-ann-normalized.log` completes 16 `Running query argument group`
    blocks where the rendered config asks for 24, ends at line 4171 with
    `mariadb.InterfaceError: Server has gone away` raised from the forked
    worker's `_insert_parallel`, then `Terminating 1 workers` -- and
    `checkpoints.json` nevertheless contains
    `normalized/percona/gist-960-euclidean/ann`. The phase ran 7 h 46 m
    (06:36:18 -> 14:22:58), exited 0, and `--resume` will never return to it.

THREE DEFECTS, one per heading below:

  D1  the engine's own error log never leaves the container. Four engines are
      started with `--log-error=<file>`, so `docker logs` is the banner and
      the file is in the volume that teardown deletes. mongodb redirects
      mongot's log into that volume too, with a shell `>` rather than a flag,
      which is why the guard here looks for every spelling and not just one.
  D2  the phase-log filename had no dataset in it and is opened "w", so a
      four-dataset profile wrote one name four times.
  D3  `after > 0` was taken as a complete sweep. ann-benchmarks runs each
      definition in a forked worker and never checks its exit code, so a
      sweep that lost one curve exits 0.

None of the three changes a measured number. All three change what is
recorded about a failure.

PROBE STYLE, per this suite's convention (see
`tests/test_e9_percona_bulk_build_crash.py`): every decision test reports its
HONEST and its MUTANT value and asserts they differ, because an identical
pair means the probe is dead and is measuring nothing. The mutant arm is
always an INPUT -- a different engine, a repeated dataset, a different result
count -- never a patched file, so nothing here can leave residue.

NO LIVE ENGINES. Nothing in this file builds an image, starts a container or
runs a profile. `docker exec` is faked at `docker_ctl.exec_in` and the
measurement container at `docker_ctl.run_foreground`; both fakes record their
argv so the tests can assert what WOULD have been run.
"""

import glob
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The run this file was written from. Quoted so the evidence above can be
#: traced back to a directory rather than to a memory.
RUN_ID = "full-20261015-215006"

#: Engines whose entrypoint sends a log to a FILE, and the file each one
#: names. `mariadb123` is `alias_of: mariadb`, so it is started by the same
#: entrypoint and logs to the same path.
#:
#: `mongodb` is here and it is the odd one: `mongod` itself runs in the
#: foreground unredirected, so `docker logs` has it, but `start_mongot`
#: redirects MONGOT's log into the data volume -- and vector search on this
#: engine happens in mongot. The key is still `error_log` because that is the
#: mechanism it needs; the config comment says what it actually is.
FILE_LOGGING = {
    "percona": ("percona", "/var/lib/vbench/percona.err"),
    "mariadb": ("mariadb", "/var/lib/vbench/mariadb.err"),
    "mariadb123": ("mariadb", "/var/lib/vbench/mariadb.err"),
    "alisql": ("alisql", "/var/lib/vbench/alisql.err"),
    "mongodb": ("mongodb", "/var/lib/vbench/mongot/mongot.log"),
    "myvector": ("myvector", "/var/lib/vbench/myvector.err"),
    "villagesql": ("villagesql", "/var/lib/vbench/villagesql.err"),
}

#: Engines whose every log reaches the container's stdout, so `docker logs`
#: already carries all of it and a second copy would be noise.
STDERR_LOGGING = ("pgvector", "valkey")

#: EVERY WAY A SERVER HERE COULD BE TOLD TO LOG TO A FILE, not just the
#: MySQL-family flag. The first version of the guard below grepped
#: `--log-error` alone, and its failure message -- "{engine}'s entrypoint now
#: redirects the server's log to a file" -- was ALREADY UNTRUE OF MONGODB when
#: it was written: `start_mongot` redirects with a plain shell `>`. The test
#: passed only because it was looking for a flag mongod does not take. A guard
#: that can only see one vendor's spelling is a guard that reports what it
#: happens to understand.
LOG_TO_FILE_MARKERS = ("--log-error", "--logpath", "--logfile",
                       "logging_collector", "log_directory", ".log\"",
                       ".log'", ".log ")


def _entrypoint(name):
    path = os.path.join(REPO, "docker", name, f"entrypoint-{name}.sh")
    with open(path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# D1 -- the engine's own error log never leaves the container
# ---------------------------------------------------------------------------

class TestTheFileLoggingEnginesDeclareTheirErrorLog:
    """`runtime.error_log` is the only thing that knows the file exists.

    It is not derivable: the path is chosen by the entrypoint, the orchestrator
    never sees that shell script, and `docker logs` gives no hint that the
    interesting stream went elsewhere. Five engines need it and two must not
    have it, because for those two the existing capture is already complete
    and a second copy of the same lines would be noise.
    """

    @pytest.mark.parametrize("engine", sorted(FILE_LOGGING))
    def test_it_is_declared(self, engine):
        from orchestrator import engines as engines_mod
        want = FILE_LOGGING[engine][1]
        got = engines_mod.registry(refresh=True)[engine].error_log
        assert got == want, (
            f"{engine}'s entrypoint sends a log to a file, so without "
            f"runtime.error_log those diagnostics stay in the container and "
            f"go with the volume; expected {want!r}, got {got!r}")

    @pytest.mark.parametrize("engine", sorted(set(FILE_LOGGING) - {"mongodb"}))
    def test_it_matches_the_entrypoint_that_starts_the_server(self, engine):
        """The config and the shell script must name the same file.

        A LINE CITATION WOULD ROT HERE, so this reads the value instead: the
        entrypoint's `LOG_FILE` default is the single place the path is chosen
        and the only value mysqld is given, so comparing against it is a
        check that survives the file being reorganised.
        """
        image, declared = FILE_LOGGING[engine]
        source = _entrypoint(image)
        match = re.search(r'LOG_FILE="\$\{VB_LOG_FILE:-([^}]+)\}"', source)
        assert match, (
            f"docker/{image}/entrypoint-{image}.sh no longer sets LOG_FILE "
            f"from VB_LOG_FILE; config/engines/{engine}.yml's error_log is "
            f"derived from that default and can no longer be checked")
        assert match.group(1) == declared, (
            f"config/engines/{engine}.yml declares error_log={declared!r} but "
            f"the entrypoint logs to {match.group(1)!r}; the orchestrator "
            f"would copy out a file the server never wrote")
        assert '--log-error="$LOG_FILE"' in source, (
            f"docker/{image}/entrypoint-{image}.sh no longer passes LOG_FILE "
            f"to --log-error, so the declared error_log is not where the "
            f"server logs")

    def test_the_mongot_path_is_composed_from_the_entrypoint(self):
        """mongodb spells it differently, so it is checked differently.

        No flag and no LOG_FILE: `start_mongot` redirects with a shell `>` to
        `${MONGOT_DATA}/mongot.log`. Both halves are read here, so a change to
        either the variable's default or the redirection fails rather than
        leaving the config pointing at a file nothing writes.
        """
        source = _entrypoint("mongodb")
        match = re.search(r'MONGOT_DATA="\$\{VB_MONGOT_DATA:-([^}]+)\}"', source)
        assert match, "entrypoint-mongodb.sh no longer sets MONGOT_DATA"
        assert '>"${MONGOT_DATA}/mongot.log"' in source, (
            "entrypoint-mongodb.sh no longer redirects mongot into "
            "MONGOT_DATA/mongot.log")
        assert f"{match.group(1)}/mongot.log" == FILE_LOGGING["mongodb"][1]

    def test_mongod_itself_still_reaches_stdout(self):
        """Which is why the declared path is mongot's and not mongod's, and why
        the stdout capture is still worth keeping for this engine.

        CHECKED AT THE INVOCATION, not at the argv literal. The first version
        of this matched `mongod --dbpath` -- a line INSIDE the `args=(` array
        -- and `--logpath` anywhere in the file, which between them would have
        let a `>"$DBPATH/mongod.log"` appended to the real invocation pass
        unnoticed. That is the same narrow-guard shape this file criticises in
        the `--log-error`-only version of the test below it, reproduced two
        tests later.
        """
        source = _entrypoint("mongodb")
        assert re.search(r'^\s*"\$\{args\[@\]\}" &\s*$', source, re.MULTILINE), (
            "entrypoint-mongodb.sh no longer launches mongod as "
            '`"${args[@]}" &`; this guard reads that line and can no longer '
            "find it")
        launch = [line for line in source.splitlines()
                  if re.match(r'^\s*"\$\{args\[@\]\}" &\s*$', line)]
        assert len(launch) == 1, launch
        assert ">" not in launch[0], (
            f"mongod's output is now redirected at the invocation ({launch[0]!r}), "
            f"so `docker logs` no longer carries it and mongodb's error_log "
            f"-- which is mongot's -- is not the whole story any more")

    def test_the_mongodb_entrypoint_redirects_nothing_but_mongot(self):
        """`LOG_TO_FILE_MARKERS` applied to the engine it was written for.

        The tuple exists because the old guard knew only `--log-error`, and it
        is applied to the stdout-only engines below -- which meant it never
        looked at mongodb, the engine whose redirect it was widened to catch.
        Here it does, minus the one redirect that is declared.
        """
        source = _entrypoint("mongodb")
        declared = '>"${MONGOT_DATA}/mongot.log" 2>&1 &'
        assert declared in source, declared
        rest = source.replace(declared, "")
        found = [m for m in LOG_TO_FILE_MARKERS if m in rest]
        assert not found, (
            f"entrypoint-mongodb.sh sends a second log to a file ({found}); "
            f"runtime.error_log carries only mongot's, so that one would be "
            f"lost with the volume")

    @pytest.mark.parametrize("engine", STDERR_LOGGING)
    def test_the_stdout_only_engines_declare_nothing(self, engine):
        """Absence is the statement that `docker logs` is already complete.

        Checked against every spelling in `LOG_TO_FILE_MARKERS`, not just the
        MySQL flag -- see the note on that tuple for why a narrower guard
        passed while being wrong.
        """
        from orchestrator import engines as engines_mod
        assert engines_mod.registry(refresh=True)[engine].error_log is None
        source = _entrypoint(engine)
        found = [m for m in LOG_TO_FILE_MARKERS if m in source]
        assert not found, (
            f"{engine}'s entrypoint now sends a log to a file ({found}), so it "
            f"needs a runtime.error_log like the other five")

    def test_every_engine_is_accounted_for_one_way_or_the_other(self):
        """A new engine must land in one of the two tables, not neither.

        Discovery is by directory, so an engine added later appears in the
        registry without appearing here, and would silently inherit the
        stdout-only treatment whether or not that is right for it.
        """
        from orchestrator import engines as engines_mod
        known = set(engines_mod.known_engines(refresh=True))
        classified = set(FILE_LOGGING) | set(STDERR_LOGGING)
        assert known == classified, (
            f"unclassified: {sorted(known - classified)}; "
            f"stale: {sorted(classified - known)}")


class _FakeExec:
    """A stand-in for `docker_ctl.exec_in` holding a fake container filesystem.

    Records every argv, so a test can assert what `docker exec` WOULD have
    been asked to run without running it.

    `gate` models the one thing `docker exec` insists on: a RUNNING container.
    When it returns False every call fails the way the daemon fails, which is
    what lets a test tell "captured before the container was stopped" from
    "captured afterwards" -- the distinction a check against teardown's last
    call could not see.

    `order` is a shared list the fakes append to, so a test can assert the
    sequence rather than only the end state.
    """

    NOT_RUNNING = "is not running"

    def __init__(self, files=None, error="No such file or directory",
                 gate=None, order=None):
        self.files = dict(files or {})
        self.error = error
        self.gate = gate
        self.order = order
        self.calls = []

    def __call__(self, name, args, timeout=60, check=False):
        args = list(args)
        self.calls.append((name, args))
        if self.order is not None:
            self.order.append("exec")
        if self.gate is not None and not self.gate():
            return subprocess.CompletedProcess(
                args, 1, "", f"Error response from daemon: container {name} "
                             f"{self.NOT_RUNNING}")
        path = args[-1]
        if args[:2] == ["tail", "-n"] and path in self.files:
            return subprocess.CompletedProcess(args, 0, self.files[path], "")
        return subprocess.CompletedProcess(args, 1, "", f"tail: {path}: {self.error}")


class TestReadingTheErrorLogOutOfTheContainerCannotFail:
    """`read_file_tail` runs in a `finally` on the path a crashed phase takes.

    A phase that has already lost its measurement must not also lose the run
    because its log could not be fetched, so every failure has to come back as
    "no log" rather than as an exception.
    """

    def test_it_returns_the_tail_of_the_file(self, monkeypatch):
        from orchestrator import docker_ctl
        fake = _FakeExec({"/var/lib/vbench/percona.err": "[ERROR] assertion\n"})
        monkeypatch.setattr(docker_ctl, "exec_in", fake)
        got = docker_ctl.read_file_tail("srv", "/var/lib/vbench/percona.err",
                                        lines=20000)
        assert got == "[ERROR] assertion"
        assert fake.calls == [("srv", ["tail", "-n", "20000",
                                       "/var/lib/vbench/percona.err"])], (
            "the file has no bound -- mongod writes thousands of lines an "
            "hour -- so this must be a bounded tail and not a cat")

    def test_an_absent_file_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec({}))
        assert docker_ctl.read_file_tail("srv", "/nope.err") is None

    def test_an_unreadable_file_is_not_an_error(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in",
                            _FakeExec({}, error="Permission denied"))
        assert docker_ctl.read_file_tail("srv", "/var/lib/vbench/x.err") is None

    def test_an_empty_file_reads_as_no_log(self, monkeypatch):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec({"/x.err": "  \n"}))
        assert docker_ctl.read_file_tail("srv", "/x.err") is None

    def test_a_daemon_that_is_not_answering_is_not_an_error(self, monkeypatch):
        """`docker exec` against a container that has gone, or with no docker
        on PATH at all, must not propagate."""
        from orchestrator import docker_ctl

        def explode(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(docker_ctl, "exec_in", explode)
        assert docker_ctl.read_file_tail("gone", "/x.err") is None

    def test_a_stopped_container_is_what_the_copy_is_for(self, monkeypatch):
        """`read_file_tail` cannot reach a stopped container, so it returns
        None there and `copy_file_out` is the mechanism that works.

        READ, NOT MEASURED: `docker exec` requiring a running container and
        `docker cp` not requiring one are Docker's documented behaviours, and
        confirming them here would mean running a container, which this change
        may not do. What IS measured is that the code takes the second path
        when the first returns nothing.
        """
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/x.err": "late words\n"}, gate=lambda: False))
        assert docker_ctl.read_file_tail("stopped", "/x.err") is None

    def test_the_copy_builds_the_argv_docker_expects(self, monkeypatch,
                                                      tmp_path):
        from orchestrator import docker_ctl
        seen = {}

        def fake_run(args, check=True, timeout=120, capture=True):
            seen["args"] = list(args)
            with open(args[3], "w") as fh:
                fh.write("[ERROR] signal 11\n")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(docker_ctl, "_run", fake_run)
        dest = docker_ctl.copy_file_out("srv", "/var/lib/vbench/percona.err",
                                        str(tmp_path / "scratch"))
        assert seen["args"][:3] == ["docker", "cp",
                                    "srv:/var/lib/vbench/percona.err"]
        assert dest and open(dest).read().startswith("[ERROR]")

    def test_a_copy_that_fails_is_not_an_error(self, monkeypatch, tmp_path):
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", lambda *a, **k:
                            subprocess.CompletedProcess([], 1, "", "no such"))
        assert docker_ctl.copy_file_out("gone", "/x.err", str(tmp_path)) is None

    def test_a_copy_that_cannot_even_start_is_not_an_error(self, monkeypatch,
                                                            tmp_path):
        from orchestrator import docker_ctl

        def explode(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(docker_ctl, "_run", explode)
        assert docker_ctl.copy_file_out("gone", "/x.err", str(tmp_path)) is None

    def test_a_copy_reporting_success_without_a_file_is_not_trusted(
            self, monkeypatch, tmp_path):
        """rc 0 and nothing on disk would otherwise be read as an empty log."""
        from orchestrator import docker_ctl
        monkeypatch.setattr(docker_ctl, "_run", lambda *a, **k:
                            subprocess.CompletedProcess([], 0, "", ""))
        assert docker_ctl.copy_file_out("srv", "/x.err", str(tmp_path)) is None

    def test_the_copied_log_is_bounded_before_it_is_archived(self, tmp_path,
                                                             monkeypatch):
        """`docker cp` has no `--tail`, so the BOUND IS THE CALLER'S JOB.

        `copy_file_out`'s docstring states it as a contract and nothing
        checked it: replacing `lines[-SERVER_LOG_TAIL:]` with `lines` passed
        the entire suite. An unbounded error log would then be archived whole
        into the run directory, where every other capture is capped at the
        same figure -- and `export` ships whatever the run directory holds.
        """
        from orchestrator import ops_pass
        overshoot = 25
        body = "".join(f"line {i}\n"
                       for i in range(ops_pass.SERVER_LOG_TAIL + overshoot))

        def copy_file_out(name, path, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, "percona.err")
            with open(dest, "w") as fh:
                fh.write(body)
            return dest

        monkeypatch.setattr(ops_pass.docker_ctl, "read_file_tail",
                            lambda *a, **k: None)
        monkeypatch.setattr(ops_pass.docker_ctl, "copy_file_out",
                            copy_file_out)
        run = _ops_run(tmp_path, "percona")
        text = run._read_error_log("/var/lib/vbench/percona.err")
        kept = text.splitlines()
        assert len(kept) == ops_pass.SERVER_LOG_TAIL, len(kept)
        assert kept[-1] == f"line {ops_pass.SERVER_LOG_TAIL + overshoot - 1}", (
            "the TAIL is what matters -- the lines written while something was "
            "going wrong are the last ones")
        assert kept[0] == f"line {overshoot}"

    def test_the_scratch_copy_is_always_cleaned_up(self, tmp_path,
                                                   monkeypatch):
        """It holds a FULL copy of the error log, once per failed phase.

        Deleting the `finally` passed the entire suite, and nine failed phases
        in the reference run would have been nine leaked copies sitting beside
        the corpus.
        """
        from orchestrator import ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "read_file_tail",
                            lambda *a, **k: None)

        def copy_file_out(name, path, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, "percona.err")
            with open(dest, "w") as fh:
                fh.write("[ERROR] signal 11\n")
            return dest

        monkeypatch.setattr(ops_pass.docker_ctl, "copy_file_out",
                            copy_file_out)
        run = _ops_run(tmp_path, "percona")
        assert run._read_error_log("/var/lib/vbench/percona.err")
        leaked = glob.glob(os.path.join(str(tmp_path / "state"),
                                        "vb-errlog-*"))
        assert leaked == [], leaked

    def test_the_scratch_copy_is_cleaned_up_on_an_interrupt_too(
            self, tmp_path, monkeypatch):
        """The `finally` covers the operator's second Ctrl-C, which is the one
        case a success-path cleanup would miss."""
        from orchestrator import ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "read_file_tail",
                            lambda *a, **k: None)

        def interrupted(name, path, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, "percona.err"), "w") as fh:
                fh.write("half a log\n")
            raise KeyboardInterrupt()

        monkeypatch.setattr(ops_pass.docker_ctl, "copy_file_out", interrupted)
        run = _ops_run(tmp_path, "percona")
        with pytest.raises(KeyboardInterrupt):
            run._read_error_log("/var/lib/vbench/percona.err")
        leaked = glob.glob(os.path.join(str(tmp_path / "state"),
                                        "vb-errlog-*"))
        assert leaked == [], leaked

    def test_the_copy_lands_beside_the_corpus_and_not_in_tmp(self, tmp_path,
                                                             monkeypatch):
        """The copy is unbounded, and /tmp is on the small filesystem.

        `docker_ctl.create_volume`'s docstring makes the same argument for the
        corpus: a benchmark box typically has a small root volume and a large
        data mount. mongodb's declared log grows for the length of a phase.
        """
        from orchestrator import ops_pass
        seen = {}
        real = ops_pass.tempfile.mkdtemp

        def mkdtemp(*a, **k):
            seen["dir"] = k.get("dir")
            return real(*a, **k)

        monkeypatch.setattr(ops_pass.tempfile, "mkdtemp", mkdtemp)
        monkeypatch.setattr(ops_pass.docker_ctl, "read_file_tail",
                            lambda *a, **k: None)
        monkeypatch.setattr(ops_pass.docker_ctl, "copy_file_out",
                            lambda *a, **k: None)
        run = _ops_run(tmp_path, "percona")
        assert run._read_error_log("/var/lib/vbench/percona.err") is None
        assert seen["dir"] == run.paths["engine_state"], seen

    def test_an_unwritable_scratch_location_returns_none_like_the_rest(
            self, tmp_path, monkeypatch):
        """`read_file_tail` and `copy_file_out` return None for everything;
        this must not be the one link in the chain that raises."""
        from orchestrator import ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "read_file_tail",
                            lambda *a, **k: None)

        def refuse(*a, **k):
            raise PermissionError("read-only")

        monkeypatch.setattr(ops_pass.tempfile, "mkdtemp", refuse)
        run = _ops_run(tmp_path, "percona")
        assert run._read_error_log("/var/lib/vbench/percona.err") is None


def _ops_run(tmp_path, engine, dataset="gist-960-euclidean", tag="m8-incremental"):
    """An OpsRun with no containers behind it.

    `__init__` only computes names and reads the registry, so constructing one
    touches neither docker nor an engine.
    """
    from orchestrator import ops_pass
    from orchestrator.config import load_engine
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return ops_pass.OpsRun(
        engine, load_engine(engine), _Resolved(), "normalized",
        {"run_dir": str(run_dir), "engine_state": str(tmp_path / "state")},
        "r1", dataset, tag)


def _log_names(tmp_path):
    directory = tmp_path / "run" / "logs"
    return sorted(os.listdir(directory)) if directory.is_dir() else []


class TestTheServerLogThatMattersIsTheOneInTheContainer:
    """D1's probe, and the before-teardown requirement.

    `_save_server_log`'s own docstring already made the right argument -- "the
    half that would have said was discarded when the container was removed" --
    and then captured the container's stdout, which for the four MySQL-family
    engines is the entrypoint banner. Six such files in run
    full-20261015-215006 are three lines each.
    """

    def test_a_file_logging_engine_gets_its_error_log_archived(
            self, tmp_path, monkeypatch):
        """HONEST arm: percona, which declares `runtime.error_log`.
        MUTANT arm: valkey, which does not and must keep today's behaviour.
        """
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs",
                            lambda *a, **k: "[vb-percona] exec mysqld ...")

        honest_dir = tmp_path / "honest"
        honest_dir.mkdir()
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err":
             "[ERROR] [MY-013183] InnoDB: Assertion failure\n"}))
        _ops_run(honest_dir, "percona")._save_server_log()
        honest = [n for n in _log_names(honest_dir) if "server-error" in n]

        mutant_dir = tmp_path / "mutant"
        mutant_dir.mkdir()
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err": "would not be read\n"}))
        _ops_run(mutant_dir, "valkey")._save_server_log()
        mutant = [n for n in _log_names(mutant_dir) if "server-error" in n]

        print(f"HONEST (percona, error_log declared) = {honest}")
        print(f"MUTANT (valkey, no error_log)        = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: declaring runtime.error_log made no difference to "
            f"what was archived, so this test is measuring nothing. Both arms "
            f"produced {honest}")
        assert len(honest) == 1, honest
        assert mutant == [], (
            "an engine without runtime.error_log must keep today's behaviour "
            f"exactly; valkey produced {mutant}")
        text = (honest_dir / "run" / "logs" / honest[0]).read_text()
        assert "Assertion failure" in text

    def test_it_asks_for_the_path_the_engine_config_names(self, tmp_path,
                                                          monkeypatch):
        """The whole fix is worthless if it fetches the wrong file."""
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        fake = _FakeExec({"/var/lib/vbench/alisql.err": "[Note] shutdown\n"})
        monkeypatch.setattr(docker_ctl, "exec_in", fake)
        _ops_run(tmp_path, "alisql")._save_server_log()
        assert any(argv[-1] == "/var/lib/vbench/alisql.err"
                   for _name, argv in fake.calls), fake.calls

    def test_the_two_server_streams_land_in_two_files(self, tmp_path,
                                                       monkeypatch):
        """The stdout capture STAYS, and does not become the error log.

        The banner it holds records the exact argv mysqld was started with,
        which is how run full-20261015-215006's buffer-pool size and its
        `--innodb-hnsw-max-memory=0` were confirmed after the fact. Two
        streams, two names.
        """
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs",
                            lambda *a, **k: "[vb-percona] exec mysqld ...")
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err": "[ERROR] gone\n"}))
        _ops_run(tmp_path, "percona")._save_server_log()
        names = _log_names(tmp_path)
        assert len(names) == 2, names
        assert any("-server-error-m8-incremental-" in n for n in names), names
        assert any("-server-m8-incremental-" in n for n in names), names

    def test_an_empty_stdout_does_not_skip_the_error_log(self, tmp_path,
                                                         monkeypatch):
        """A server that said nothing on stdout is exactly the one whose file
        log matters, so the two captures cannot be chained."""
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            {"/var/lib/vbench/percona.err": "[ERROR] signal 11\n"}))
        _ops_run(tmp_path, "percona")._save_server_log()
        assert [n for n in _log_names(tmp_path) if "server-error" in n]

    def test_a_missing_error_log_is_reported_and_not_fatal(self, tmp_path,
                                                            monkeypatch, capsys):
        """Losing the log is bad. Losing the run because the log could not be
        fetched would be worse -- this is called from a `finally` on the path a
        crashed phase takes."""
        from orchestrator import docker_ctl, ops_pass
        monkeypatch.setattr(ops_pass.docker_ctl, "logs", lambda *a, **k: "")
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec({}))
        _ops_run(tmp_path, "percona")._save_server_log()
        assert [n for n in _log_names(tmp_path) if "server-error" in n] == []
        assert "error_log" in capsys.readouterr().err, (
            "an engine that declares an error log and then has none is worth "
            "a line; silence here reads as 'the server had nothing to say'")

    def test_neither_capture_can_end_the_run(self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl, ops_pass

        def explode(*a, **k):
            raise RuntimeError("no such container")

        monkeypatch.setattr(ops_pass.docker_ctl, "logs", explode)
        monkeypatch.setattr(docker_ctl, "exec_in", explode)
        _ops_run(tmp_path, "percona")._save_server_log()      # must not raise

    def test_it_runs_before_teardown_on_both_container_paths(self):
        """`teardown` removes the volume the file lives in, so there is no
        second chance. Both `run_harness` and `run_script` reach the capture
        from a `finally`, inside the `with` whose `__exit__` is `teardown`.
        """
        with open(os.path.join(REPO, "orchestrator", "ops_pass.py")) as fh:
            source = fh.read()
        for method in ("run_harness", "run_script"):
            block = source.split(f"def {method}")[1].split("\n    def ")[0]
            assert "_save_server_log()" in block.split("finally:")[1], method
        teardown = source.split("def teardown")[1].split("\n    def ")[0]
        assert "remove_volume" in teardown, (
            "if teardown no longer removes the volume, the before-teardown "
            "requirement this test guards has changed shape")



class TestSetupThatNeverReachedTheHarnessStillExplainsItself:
    """The one path where the server's log was most wanted and least likely.

    `__enter__` runs three steps and the last two can fail -- a missing image,
    or a server that never answers inside `wait_healthy`'s five minutes. On
    failure it tears down and re-raises, and `run_harness` is never entered,
    so neither capture ran and `teardown` removed the volume the engine's own
    log lives in. A percona that refuses to start under a tuned
    `innodb_buffer_pool_size` left "did not become ready within 300s" and the
    three-line banner, and the assertion that caused it was deleted.

    THE ORDERING IS ASSERTED AGAINST `stop`, NOT AGAINST `remove_volume`, and
    the reason is NOT the one an earlier revision of this docstring gave.

    `remove_volume` is teardown's fifth call, so a capture moved inside
    `teardown` before it satisfied "the log existed before the volume went"
    and passed while being wrong. That much stands. But the mechanism stated
    here was "`docker exec` needs a running container, so after `stop` the
    error log reads back as nothing" -- and once `_read_error_log` gained its
    `docker cp` fallback, that stopped being true. `docker cp` goes through
    the daemon, so between `stop` and `remove` the log IS still retrievable;
    `test_a_server_that_died_is_copied_out_instead` is the proof of it.

    The real boundaries, as the fakes below now model them separately:

    * before `stop` -- `docker exec` works, so the BOUNDED tail is used.
    * after `stop`, before `remove` -- exec fails, `docker cp` still works.
      The log survives, but only through the unbounded copy.
    * after `remove` -- both fail. This is where the fix is nullified, and it
      is `remove`, not `stop`, that draws that line.

    So precede-`stop` is a stricter requirement than strictly necessary, kept
    deliberately: it is the only placement that does not force the capture
    onto a copy of an unbounded file. `test_the_capture_is_not_inside_teardown`
    forbids the bad placement unconditionally, so no guard depends on which
    of the two calls is the operative one.
    """

    def _wire(self, monkeypatch, tmp_path, exists=True, error_log=True,
              copyable=False):
        """Every docker call `__enter__` and `teardown` make, replaced.

        Returns the shared order list and the state dict, so a test can assert
        the sequence and not merely the end state.
        """
        from orchestrator import docker_ctl, ops_pass
        order = []
        # TWO FLAGS, NOT ONE, because the two mechanisms lose access at
        # different calls and collapsing them is what let a wrong reason stand:
        # `stop` ends `docker exec`, `remove` ends `docker cp` as well.
        state = {"running": True, "exists": True}
        server = f"{_ops_run(tmp_path, 'percona').server_name}"

        def record(label, fn=None):
            def call(*a, **k):
                order.append(label)
                if label == "stop":
                    state["running"] = False
                if label == "remove" and a and a[0] == server:
                    state["running"] = False
                    state["exists"] = False
                return fn(*a, **k) if fn else None
            return call

        for name, value in (
            ("create_network", lambda *a, **k: "net"),
            ("create_volume", lambda *a, **k: "vol"),
            ("image_exists", lambda *a, **k: True),
            ("start", lambda *a, **k: "cid"),
            ("stop", record("stop")),
            ("remove", record("remove")),
            ("remove_network", record("remove_network")),
            ("remove_volume", record("remove_volume")),
            ("remove_tree_as_root", record("remove_tree", lambda *a, **k: True)),
            ("container_exists", record("inspect", lambda *a, **k: exists)),
            # `docker logs` works on a stopped container, so it is NOT the
            # discriminator here -- only `docker exec` is.
            ("logs", record("logs",
                            lambda *a, **k: "[vb-percona] exec mysqld ...")),
        ):
            monkeypatch.setattr(ops_pass.docker_ctl, name, value)

        def never_ready(*a, **k):
            raise docker_ctl.DockerError(
                "container ...-srv did not become ready within 300s")

        monkeypatch.setattr(ops_pass.docker_ctl, "wait_healthy", never_ready)

        files = {"/var/lib/vbench/percona.err":
                 "[ERROR] [MY-012271] InnoDB: Cannot allocate memory for the "
                 "buffer pool\n"} if error_log else {}
        monkeypatch.setattr(docker_ctl, "exec_in", _FakeExec(
            files, gate=lambda: state["running"], order=order))

        def copy_file_out(name, path, dest_dir):
            order.append("cp")
            # `docker cp` survives `stop` and dies with `remove`.
            if not state["exists"] or not copyable or path not in files:
                return None
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(path))
            with open(dest, "w") as fh:
                fh.write(files[path])
            return dest

        monkeypatch.setattr(ops_pass.docker_ctl, "copy_file_out", copy_file_out)
        return order, state

    def _error_log_text(self, tmp_path):
        names = [n for n in _log_names(tmp_path) if "server-error" in n]
        if not names:
            return None
        return (tmp_path / "run" / "logs" / names[0]).read_text()

    def test_the_error_log_comes_out_before_the_container_is_stopped(
            self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl
        order, _state = self._wire(monkeypatch, tmp_path)
        with pytest.raises(docker_ctl.DockerError):
            _ops_run(tmp_path, "percona").__enter__()

        text = self._error_log_text(tmp_path)
        assert text and "Cannot allocate memory" in text, (
            "the reason the server refused to start is the whole point of "
            f"this path; got {text!r}")
        assert "exec" in order and "stop" in order, order
        assert order.index("exec") < order.index("stop"), (
            f"the capture must precede `stop`: that is the only placement "
            f"where the BOUNDED `docker exec` tail still works, so the "
            f"capture is not forced onto an unbounded `docker cp` of a file "
            f"with no size limit. Order was {order}")
        assert "cp" not in order, (
            f"nothing should have fallen back to the unbounded copy while the "
            f"container was still running: {order}")

    def test_after_stop_the_log_survives_only_through_the_unbounded_copy(
            self, tmp_path, monkeypatch):
        """Why precede-`stop` is the requirement even though `stop` is not the
        cliff edge.

        `docker cp` goes through the daemon, so a capture after `stop` still
        RETRIEVES the log -- the earlier claim that it "reads back as nothing"
        was wrong once the fallback existed. What it loses is the bound: the
        bounded `docker exec` tail is gone and the whole file gets copied.
        """
        order, state = self._wire(monkeypatch, tmp_path, copyable=True)
        run = _ops_run(tmp_path, "percona")
        state["running"] = False          # `stop` has happened
        run._save_startup_log()
        assert "Cannot allocate memory" in (self._error_log_text(tmp_path) or "")
        assert order.index("exec") < order.index("cp"), order

    def test_after_remove_the_log_is_gone_for_good(self, tmp_path,
                                                   monkeypatch):
        """`remove`, not `stop`, is where the fix is nullified.

        Both mechanisms need the container to EXIST; only one needs it to be
        running. So this is the boundary a capture inside `teardown` falls the
        wrong side of -- `remove` is teardown's second call.
        """
        order, state = self._wire(monkeypatch, tmp_path, copyable=True)
        run = _ops_run(tmp_path, "percona")
        state["running"] = False
        state["exists"] = False           # `remove` has happened too
        run._save_startup_log()
        assert self._error_log_text(tmp_path) is None
        assert "cp" in order, (
            f"the copy should have been attempted and failed: {order}")
        assert [n for n in _log_names(tmp_path) if "server-error" not in n], (
            "the stdout half should still be archived -- `docker logs` works "
            "on a stopped container, and `container_exists` is faked True "
            "here because this models the placement, not a vanished container")

    def test_a_server_that_died_is_copied_out_instead(self, tmp_path,
                                                      monkeypatch):
        """`wait_healthy`'s OTHER failure, and the one that matters most.

        It raises two ways: the 300 s probe timeout, where the container is
        still running and `docker exec` reaches it, and
        `if not is_running(name)` -- "container ... exited during startup" --
        where the server DIED. On the second, `docker exec` cannot get in at
        all, and that is precisely the server whose last words are wanted. So
        `docker cp`, which reads through the daemon rather than through the
        container, is the fallback.
        """
        from orchestrator import docker_ctl
        order, state = self._wire(monkeypatch, tmp_path, copyable=True)
        run = _ops_run(tmp_path, "percona")
        state["running"] = False          # exited during startup
        run._save_startup_log()
        text = self._error_log_text(tmp_path)
        assert text and "Cannot allocate memory" in text, text
        assert order.index("exec") < order.index("cp"), (
            f"the bounded `docker exec` tail must be tried first; `docker cp` "
            f"has no --tail and copies the whole file. Order was {order}")

    def test_a_file_that_cannot_be_copied_either_is_still_not_fatal(
            self, tmp_path, monkeypatch, capsys):
        """Both mechanisms exhausted: the container is stopped so `docker exec`
        cannot get in, AND the copy fails. One honest note, no file, and the
        run goes on. The copy failing is what the `copyable=False` arm models
        -- without it the fallback succeeds and there is no such case."""
        _order, state = self._wire(monkeypatch, tmp_path, copyable=False)
        run = _ops_run(tmp_path, "percona")
        state["running"] = False
        run._save_startup_log()
        assert self._error_log_text(tmp_path) is None
        assert "error_log" in capsys.readouterr().err

    def test_a_setup_that_never_made_a_container_writes_no_log(
            self, tmp_path, monkeypatch):
        """A missing image raises before `docker run`, and `docker logs` then
        answers with the daemon's "No such container" on stderr. Archiving
        that under a `-server-` name would put a Docker error in the run
        directory as though the server had said it."""
        from orchestrator import docker_ctl
        order, _state = self._wire(monkeypatch, tmp_path, exists=False)
        with pytest.raises(docker_ctl.DockerError):
            _ops_run(tmp_path, "percona").__enter__()
        assert _log_names(tmp_path) == []
        assert "logs" not in order, (
            f"nothing should have asked for logs at all: {order}")

    def test_it_still_tears_down(self, tmp_path, monkeypatch):
        """The capture is added in front of teardown, not instead of it: a
        failed setup must still not leak a volume, a network and a root-owned
        corpus directory."""
        from orchestrator import docker_ctl
        order, _state = self._wire(monkeypatch, tmp_path)
        with pytest.raises(docker_ctl.DockerError):
            _ops_run(tmp_path, "percona").__enter__()
        for call in ("stop", "remove", "remove_network", "remove_volume",
                     "remove_tree"):
            assert call in order, (call, order)

    def test_an_interrupt_during_the_capture_still_tears_down(
            self, tmp_path, monkeypatch):
        """A SECOND Ctrl-C must not turn a failed setup into a leak.

        The capture can sit on three timeouts (30 s inspect, 60 s logs, 120 s
        exec) against a wedged daemon, and its inner handlers are
        `except Exception`, which deliberately does not catch
        KeyboardInterrupt -- so the interrupt aborts the capture at once. With
        `teardown()` merely sequenced after it, that interrupt escaped and
        leaked the network, the bind-backed volume and the root-owned corpus
        directory: everything the handler exists to prevent. `teardown` is in
        a `finally`, which is what makes the fast abort safe.
        """
        from orchestrator import ops_pass
        order, _state = self._wire(monkeypatch, tmp_path)

        def interrupted(*a, **k):
            order.append("inspect")
            raise KeyboardInterrupt()

        monkeypatch.setattr(ops_pass.docker_ctl, "container_exists",
                            interrupted)
        with pytest.raises(KeyboardInterrupt):
            _ops_run(tmp_path, "percona").__enter__()
        for call in ("stop", "remove_volume", "remove_tree"):
            assert call in order, (
                f"a Ctrl-C during the capture skipped teardown's {call}: "
                f"{order}")

    def test_a_capture_that_fails_does_not_mask_the_real_error(
            self, tmp_path, monkeypatch):
        """The setup failure is the diagnosis; losing the log must not replace
        it with an exception about logging."""
        from orchestrator import docker_ctl, ops_pass
        self._wire(monkeypatch, tmp_path, error_log=False)

        def explode(*a, **k):
            raise RuntimeError("daemon went away")

        monkeypatch.setattr(ops_pass.docker_ctl, "container_exists", explode)
        with pytest.raises(docker_ctl.DockerError) as caught:
            _ops_run(tmp_path, "percona").__enter__()
        assert "did not become ready" in str(caught.value)

    def test_the_capture_is_not_inside_teardown(self):
        """Where it must NOT be, stated directly.

        The source grep that guards the two container paths looks for the
        capture inside a `finally:`, which a capture moved into `teardown`
        also satisfies. Teardown's second call is `remove`, after which
        neither `docker exec` nor `docker cp` can reach the container at all,
        so that placement loses the log outright. Named here because this
        assertion is unconditional: it does not depend on which teardown call
        is the operative one, which is the thing a previous revision got
        wrong.
        """
        with open(os.path.join(REPO, "orchestrator", "ops_pass.py")) as fh:
            source = fh.read()
        teardown = source.split("def teardown")[1].split("\n    def ")[0]
        assert "docker_ctl.remove(self.server_name)" in teardown, (
            "teardown no longer removes the server container, so the reason "
            "this location is forbidden has changed shape")
        for forbidden in ("_save_server_log", "_save_server_error_log",
                          "_save_startup_log", "read_file_tail",
                          "copy_file_out"):
            assert forbidden not in teardown, (
                f"`{forbidden}` is inside `teardown`, which removes the "
                f"container before it removes the volume -- after that "
                f"neither mechanism can reach the error log")


# ---------------------------------------------------------------------------
# D2 -- each dataset overwrote the previous dataset's log
# ---------------------------------------------------------------------------

class TestEachDatasetKeepsItsOwnPhaseLog:

    def test_two_datasets_do_not_collide(self, tmp_path):
        """HONEST arm: two datasets, as a four-dataset profile runs them.
        MUTANT arm: the same dataset twice, which is what the name used to
        express for every dataset in the profile.
        """
        from orchestrator.docker_ctl import save_phase_log
        honest_dir, mutant_dir = tmp_path / "h", tmp_path / "m"
        for directory, second in ((honest_dir, "sift-128-euclidean"),
                                  (mutant_dir, "gist-960-euclidean")):
            save_phase_log(str(directory), "percona", "gist-960-euclidean",
                           "ann", "normalized", ["gist output"])
            save_phase_log(str(directory), "percona", second,
                           "ann", "normalized", ["second output"])
        honest = sorted(os.listdir(honest_dir / "logs"))
        mutant = sorted(os.listdir(mutant_dir / "logs"))
        print(f"HONEST (two datasets)  = {honest}")
        print(f"MUTANT (one, repeated) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the dataset made no difference to the filename, so "
            f"the second phase still overwrote the first. Both arms left "
            f"{honest}")
        assert len(honest) == 2, honest
        assert len(mutant) == 1, mutant
        kept = (honest_dir / "logs" / [n for n in honest if "gist" in n][0])
        assert kept.read_text().strip() == "gist output", (
            "in run full-20261015-215006 this is the file that was lost: the "
            "surviving percona-ann-normalized.log was gist's and sift's had "
            "been overwritten, so the six-second sift failure has no record")

    def test_the_dataset_is_required_and_cannot_be_defaulted(self):
        """A keyword with a default is how this comes back.

        Both call sites have the dataset in scope -- `ann_pass` as a local and
        `ops_pass` as `self.dataset` -- so there is nothing to default to, and
        a default would let the next caller silently reintroduce the collision.
        """
        import inspect
        from orchestrator.docker_ctl import save_phase_log
        parameter = inspect.signature(save_phase_log).parameters["dataset"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    def test_every_other_field_still_separates(self, tmp_path):
        """The dataset is added to the name, not substituted for anything."""
        from orchestrator.docker_ctl import save_phase_log
        written = {
            save_phase_log(str(tmp_path), *args, ["x"]) for args in (
                ("percona", "gist-960-euclidean", "ann", "normalized"),
                ("percona", "gist-960-euclidean", "ann", "tuned"),
                ("percona", "gist-960-euclidean", "ops-m8-incremental",
                 "normalized"),
                ("percona", "gist-960-euclidean", "server-m8-incremental",
                 "normalized"),
                ("mariadb", "gist-960-euclidean", "ann", "normalized"),
            )
        }
        assert len(written) == 5, written

    def test_the_lab_shape_stays_sane(self, tmp_path):
        """`cmd_lab` drives OpsRun with dataset="lab", which is not a corpus.

        It still has to produce a legible, collision-free name, because a lab
        probe and a measurement can share a run directory.
        """
        from orchestrator.docker_ctl import save_phase_log
        path = save_phase_log(str(tmp_path), "valkey", "lab", "server-lab",
                              "tuned", ["probe output"])
        assert os.path.basename(path) == "valkey-lab-server-lab-tuned.log"

    def test_both_measurement_paths_pass_their_dataset(self):
        """A grep, because the defect was a caller that had the value in scope
        and did not pass it."""
        for module, expression in (("ann_pass", "engine, dataset,"),
                                   ("ops_pass", "self.engine, self.dataset,")):
            with open(os.path.join(REPO, "orchestrator",
                                   f"{module}.py")) as fh:
                source = fh.read()
            for call in source.split("save_phase_log(")[1:]:
                head = call.split(")")[0]
                assert expression in head, (
                    f"orchestrator/{module}.py has a save_phase_log call that "
                    f"does not pass its dataset: {head!r}")


# ---------------------------------------------------------------------------
# D3 -- a partially crashed ANN sweep was checkpointed as complete
# ---------------------------------------------------------------------------

#: What each engine's `full` grid asks for, per resource pass. MEASURED, by
#: rendering the config and counting the file names it leads to.
#:
#: THE WHOLE TABLE EXISTS BECAUSE ONE ROW IS NOT ENOUGH. Every `run_engine`
#: test below drives percona under `full`, and percona is 24 in BOTH passes --
#: so four separate defects were invisible: reading the count for a hardcoded
#: engine, returning a hardcoded 24 whenever the config parses, and getting
#: either aggregation wrong in a way that happens to land on 24 for this one
#: grid. mariadb is where it bites, at 24 normalized and 16 tuned, because the
#: tuned pass gives it its own `m_values_by_engine`.
FULL_PROFILE_COUNTS = {
    ("percona", "normalized"): 24, ("percona", "tuned"): 24,
    ("mariadb", "normalized"): 24, ("mariadb", "tuned"): 16,
    ("mariadb123", "normalized"): 24, ("mariadb123", "tuned"): 16,
    ("alisql", "normalized"): 24, ("alisql", "tuned"): 16,
    ("pgvector", "normalized"): 48, ("pgvector", "tuned"): 144,
    ("valkey", "normalized"): 24, ("valkey", "tuned"): 72,
    ("mongodb", "normalized"): 8, ("mongodb", "tuned"): 16,
}

#: Result file names an ACTUAL ann-benchmarks run left in
#: `results/annb/normalized/fashion-mnist-784-euclidean/10/`, with the
#: arguments that produced them.
#:
#: This is the one part of the reproduction in `ann_config` that cannot be
#: settled by reading upstream: it has to agree with `json.dumps`' spacing and
#: key order and with `re`'s `\W` class, exactly. These literals are the only
#: external ground truth available -- `results/` is gitignored, so they are
#: copied in rather than read, and the listing they came from is
#: `euclidean_M_16_engine_InnoDB_{10,40,160}.hdf5` under `mariadb/` and
#: `alisql/` and `euclidean_M_16_build_mode_post_efConstruction_200_{...}.hdf5`
#: under `pgvector/`.
REAL_RESULT_STEMS = [
    (["euclidean", {"M": 16, "engine": "InnoDB"}, 10],
     "euclidean_M_16_engine_InnoDB_10"),
    (["euclidean", {"M": 16, "engine": "InnoDB"}, 40],
     "euclidean_M_16_engine_InnoDB_40"),
    (["euclidean", {"M": 16, "engine": "InnoDB"}, 160],
     "euclidean_M_16_engine_InnoDB_160"),
    (["euclidean", {"M": 16, "build_mode": "post", "efConstruction": 200}, 10],
     "euclidean_M_16_build_mode_post_efConstruction_200_10"),
    (["euclidean", {"M": 16, "build_mode": "post", "efConstruction": 200}, 40],
     "euclidean_M_16_build_mode_post_efConstruction_200_40"),
    (["euclidean", {"M": 16, "build_mode": "post", "efConstruction": 200}, 160],
     "euclidean_M_16_build_mode_post_efConstruction_200_160"),
]


def _resources(profile, resource_pass):
    """The resources production actually renders against.

    `cli.py` never hands `render_config` a bare `load_resources()`: it merges
    the profile's own `resources:` block over it first. Five profiles carry
    one, and `m-sweep.yml` clears `extras.m_values_by_engine` with `null` --
    the deep-merge idiom CLAUDE.md warns about, where `{}` would recurse and
    change nothing. Skipping the merge here evaluated the count against grids
    those five profiles do not run.
    """
    from orchestrator.config import load_resources, merge_resource_overrides
    return merge_resource_overrides(load_resources(resource_pass), profile)


def _render(engine, profile_name, resource_pass):
    from orchestrator import ann_pass
    from orchestrator.config import load_profile
    profile = load_profile(profile_name)
    return ann_pass.render_config(engine, profile,
                                  _resources(profile, resource_pass),
                                  resource_pass)


def _body(run_groups, name="e"):
    """A minimal rendered body, for the cases no profile produces."""
    return {"float": {"any": [{"name": name, "base_args": ["@metric"],
                               "run_groups": run_groups}]}}


class TestTheResultFileNameIsWhatGetsCounted:
    """ann-benchmarks names the file from the ARGUMENTS and nothing else.

    `results.build_result_filepath` is handed
    `definition.arguments + query_arguments`; the run group's name is not in
    the path. So two definitions with equal argument values write ONE file,
    and multiplying grid sizes over-counts -- which would make a healthy phase
    fail the shortfall check and blame a worker that never died. Counting the
    set of file names cannot make that mistake.
    """

    @pytest.mark.parametrize("data,stem", REAL_RESULT_STEMS,
                             ids=[s for _d, s in REAL_RESULT_STEMS])
    def test_it_reproduces_a_name_a_real_run_wrote(self, data, stem):
        from orchestrator import ann_config
        assert ann_config.result_stem(data) == stem

    def test_the_key_order_is_sorted_and_not_the_build_order(self):
        """`sort_keys=True` is LOAD-BEARING, and nothing could see it.

        Every other literal in `REAL_RESULT_STEMS` happens to be written in
        sorted key order, so sorted and insertion order coincide and deleting
        the flag left all of them passing. `render_config` does not build them
        that way: its pgvector branch is `{"M": …, "efConstruction": …,
        "build_mode": …}`, which unsorted dumps as
        `euclidean_M_16_efConstruction_200_build_mode_post_10` where the file
        on disk is `euclidean_M_16_build_mode_post_efConstruction_200_10`.

        So this literal is written in `render_config`'s own order, and the
        first assertion is the non-vacuity guard for that: if someone
        "tidies" it into alphabetical order the test stops measuring the flag
        and says so.
        """
        from orchestrator import ann_config
        as_built = {"M": 16, "efConstruction": 200, "build_mode": "post"}
        assert list(as_built) == ["M", "efConstruction", "build_mode"], (
            "this dict has to be in render_config's build order, not sorted "
            "order, or it cannot detect sort_keys going missing")
        assert (ann_config.result_stem(["euclidean", as_built, 10])
                == "euclidean_M_16_build_mode_post_efConstruction_200_10")

    def test_the_expansion_preserves_the_build_order(self):
        """Which is what makes the test above true of the LIVE path too.

        `_expand` assembles its dicts from the input's key order, so
        `render_config`'s order survives all the way to `json.dumps` and an
        unsorted dump would carry it into the file name -- it is not only the
        hand-written literal that depends on the flag.
        """
        from orchestrator import ann_config
        expanded = ann_config._expand({"M": [16], "efConstruction": [200],
                                       "build_mode": "post"})
        assert [list(d) for d in expanded] == [["M", "efConstruction",
                                                "build_mode"]]

    def test_a_repeated_index_value_is_one_file(self):
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16, 16]}],
                            "query_args": [[10, 20]]}})
        assert ann_config.expected_result_count(body) == 2, (
            "M: [16, 16] is two definitions writing the same two file names; "
            "the arithmetic this replaces said 4")

    def test_a_repeated_query_value_is_one_file(self):
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16]}],
                            "query_args": [[10, 10, 20]]}})
        assert ann_config.expected_result_count(body) == 2

    def test_values_the_sanitiser_cannot_tell_apart_are_one_file(self):
        """`\\W+` -> `_` collapses the quotes, so 10 and "10" name one file."""
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16]}],
                            "query_args": [[10, "10"]]}})
        assert ann_config.expected_result_count(body) == 1

    def test_two_run_groups_that_differ_only_by_name_share_their_files(self):
        """The case that would have broken a future engine branch.

        Every branch of `render_config` today happens to put its
        distinguishing value inside the arg dict -- `build_mode` for pgvector
        and valkey, `quantization` for mongodb, `engine` for the MySQL family.
        A branch distinguishing its run groups by NAME alone would write one
        set of files and be charged for two, and every complete phase of that
        engine would fail after hours with a message about a dead worker.
        """
        from orchestrator import ann_config
        body = _body({"a": {"arg_groups": [{"M": [8, 16, 32]}],
                            "query_args": [[10, 20]]},
                      "b": {"arg_groups": [{"M": [16, 32]}],
                            "query_args": [[10, 20]]}})
        assert ann_config.expected_result_count(body) == 6, (
            "six distinct (M, ef) pairs across the two groups; the arithmetic "
            "this replaces said 10")

    def test_a_disabled_algorithm_writes_nothing(self):
        """`main.filter_disabled_algorithms` drops it before anything runs.

        Counting it would charge the phase for files nothing writes, which
        fails a healthy sweep -- the same shape as every other over-count
        here. `render_config` always emits `disabled: False`, so this is
        unreachable today, which is exactly the standard applied to the
        `arg_groups` product and the run-group collision.
        """
        from orchestrator import ann_config
        run_groups = {"g": {"arg_groups": [{"M": [8, 16]}],
                            "query_args": [[10, 20]]}}
        live = {"name": "e", "base_args": ["@metric"], "run_groups": run_groups}
        assert ann_config.expected_result_count(
            {"float": {"any": [dict(live, disabled=False)]}}) == 4
        assert ann_config.expected_result_count(
            {"float": {"any": [dict(live, disabled=True)]}}) == 0

    def test_two_entries_sharing_a_name_are_one_algorithm(self):
        """`_get_algorithm_definitions` does `definitions[cc.pop("name")] = cc`.

        The later entry REPLACES the earlier one outright -- it does not merge
        with it -- so a body carrying two entries called `e` writes only the
        second's files. Unioning both over-counted.
        """
        from orchestrator import ann_config
        body = {"float": {"any": [
            {"name": "e", "base_args": ["@metric"],
             "run_groups": {"a": {"arg_groups": [{"M": [8]}],
                                  "query_args": [[10]]}}},
            {"name": "e", "base_args": ["@metric"],
             "run_groups": {"b": {"arg_groups": [{"M": [16, 32]}],
                                  "query_args": [[10]]}}},
        ]}}
        assert ann_config.expected_result_count(body) == 2, (
            "only the second entry survives upstream, and it asks for two "
            "files; unioning the two entries said 3")

    def test_equal_stems_under_two_algorithms_are_two_files(self):
        """Upstream writes under `results/<dataset>/<count>/<algorithm>/`, so
        the algorithm IS in the path even though the run group is not. Merging
        them would under-count, and under-counting hides a lost curve."""
        from orchestrator import ann_config
        body = {"float": {"any": [
            {"name": "one", "base_args": ["@metric"],
             "run_groups": {"g": {"arg_groups": [{"M": [16]}],
                                  "query_args": [[10]]}}},
            {"name": "two", "base_args": ["@metric"],
             "run_groups": {"g": {"arg_groups": [{"M": [16]}],
                                  "query_args": [[10]]}}},
        ]}}
        assert ann_config.expected_result_count(body) == 2

    def test_arg_group_entries_are_multiplied_not_added(self):
        """W10. `generate_arg_combinations` expands each `arg_groups` entry and
        then takes the cartesian PRODUCT of the expansions. Summing them --
        which is what this helper did first, and what its docstring wrongly
        claimed upstream does -- gives 5 where 6 files are written.
        """
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [8, 16, 32]},
                                           {"ef": [100, 200]}],
                            "query_args": [[10]]}})
        assert ann_config.expected_result_count(body) == 6

    def test_query_arg_lists_are_multiplied_not_replaced(self):
        """W11. `query_args` is expanded by the same cartesian product, so two
        lists are every pair of them, not the last list."""
        from orchestrator import ann_config
        body = _body({"g": {"arg_groups": [{"M": [16]}],
                            "query_args": [[10, 20], [1, 2, 3]]}})
        assert ann_config.expected_result_count(body) == 6

    def test_a_definition_with_no_query_args_still_writes_one_file(self):
        """`definition.query_argument_groups or [[]]` -- one iteration and one
        file, not zero."""
        from orchestrator import ann_config
        assert ann_config.expected_result_count(
            _body({"g": {"arg_groups": [{"M": [8, 16]}]}})) == 2

    def test_a_scalar_index_parameter_is_not_a_sweep(self):
        """`engine: "InnoDB"` is one value carried, not a list to multiply by."""
        from orchestrator import ann_config
        assert ann_config.expected_result_count(
            _body({"g": {"arg_groups": [{"M": [8, 16, 32],
                                         "efConstruction": [200],
                                         "engine": "InnoDB"}],
                         "query_args": [[10]]}})) == 3


class TestTheExpectedResultCountComesFromTheRenderedConfig:
    """The rendered config is the only source of truth for the grid.

    The profile, the resource pass and `render_config`'s per-engine branch all
    contribute to it and none of them reaches the result tree, so there is
    nothing else to compare `after` against.
    """

    def test_the_full_profile_asks_percona_for_twenty_four(self):
        """3 M x 1 efConstruction x 8 ef_search, in both passes.

        Which is the number run full-20261015-215006 is measured against: its
        normalized gist phase completed 16 `Running query argument group`
        blocks and its tuned gist phase completed 24.
        """
        from orchestrator import ann_config
        for resource_pass in ("normalized", "tuned"):
            body = _render("percona", "full", resource_pass)
            group = body["float"]["any"][0]["run_groups"]["innodb"]
            assert len(group["arg_groups"][0]["M"]) == 3
            assert len(group["arg_groups"][0]["efConstruction"]) == 1
            assert len(group["query_args"][0]) == 8
            assert ann_config.expected_result_count(body) == 24, resource_pass

    @pytest.mark.parametrize("engine,resource_pass",
                             sorted(FULL_PROFILE_COUNTS),
                             ids=[f"{e}-{p}"
                                  for e, p in sorted(FULL_PROFILE_COUNTS)])
    def test_every_engine_and_pass_gets_its_own_number(self, engine,
                                                       resource_pass,
                                                       tmp_path):
        """W6 and W7, through `write_config` -- the path `run_engine` uses.

        Reading the count for a fixed engine, or returning a fixed 24 whenever
        the config parses, passes every percona case and fails here.
        """
        from orchestrator import ann_config
        body = _render(engine, "full", resource_pass)
        ann_config.write_config(str(tmp_path), engine, body)
        assert (ann_config.expected_results(str(tmp_path), engine)
                == FULL_PROFILE_COUNTS[(engine, resource_pass)])

    def test_the_profiles_resource_block_reaches_the_count(self):
        """The merge is load-bearing, not a formality.

        MEASURED across every profile x pass x engine: the merge changes the
        count for 7 of the 140 combinations. The sharpest is
        `m-sweep`/tuned/mariadb -- 16 unmerged, 40 merged -- because
        `m-sweep.yml` clears the tuned pass's inherited
        `extras.m_values_by_engine` with `null`, restoring its own five M
        values where the inherited pin leaves two. That is the deep-merge
        idiom CLAUDE.md singles out (`{}` recurses and changes nothing; `null`
        clears), and it is the key whose silent inheritance halved this very
        profile once before.

        Without the merge these tests judged five profiles against grids they
        do not run.
        """
        from orchestrator import ann_config, ann_pass
        from orchestrator.config import load_profile, load_resources
        profile = load_profile("m-sweep")
        assert profile.get("resources"), "m-sweep lost its resources block"
        merged = ann_config.expected_result_count(ann_pass.render_config(
            "mariadb", profile, _resources(profile, "tuned"), "tuned"))
        unmerged = ann_config.expected_result_count(ann_pass.render_config(
            "mariadb", profile, load_resources("tuned"), "tuned"))
        assert (merged, unmerged) == (40, 16), (merged, unmerged)

    def test_the_passes_disagree_for_at_least_one_engine(self):
        """Non-vacuity for the table above. If every engine were 24 in both
        passes the parametrised case could not see the pass being ignored --
        which is exactly how it was invisible while only percona was driven."""
        differing = {engine for (engine, _p) in FULL_PROFILE_COUNTS
                     if FULL_PROFILE_COUNTS[(engine, "normalized")]
                     != FULL_PROFILE_COUNTS[(engine, "tuned")]}
        assert "mariadb" in differing, differing
        assert len(differing) >= 4, differing

    def test_it_moves_with_the_grid(self):
        """HONEST arm: the `full` grid. MUTANT arm: one M value removed.

        A count that does not move with the config is a constant, and a
        constant cannot tell a short sweep from a complete one.
        """
        from orchestrator import ann_config
        honest_body = _render("percona", "full", "normalized")
        mutant_body = _render("percona", "full", "normalized")
        mutant_body["float"]["any"][0]["run_groups"]["innodb"][
            "arg_groups"][0]["M"] = [16]
        honest = ann_config.expected_result_count(honest_body)
        mutant = ann_config.expected_result_count(mutant_body)
        print(f"HONEST (M=[8,16,32]) = {honest}")
        print(f"MUTANT (M=[16])      = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the expected count did not move when the grid did, "
            f"so it is not derived from the rendered config. Both arms "
            f"returned {honest}")
        assert (honest, mutant) == (24, 8)

    def test_every_profile_in_the_repo_renders_and_counts(self):
        """No profile may produce a body the counter reads as zero.

        Zero switches the shortfall check off silently, so a profile whose
        body this helper chokes on would lose the guard without saying so.
        `render_config` refusing outright is the only acceptable way for a
        profile and an engine not to go together.
        """
        from orchestrator import ann_config, ann_pass
        from orchestrator import engines as engines_mod
        from orchestrator.config import CONFIG_DIR, load_profile
        names = sorted(f[:-4] for f in
                       os.listdir(os.path.join(CONFIG_DIR, "profiles"))
                       if f.endswith(".yml"))
        assert len(names) >= 8, names
        checked = 0
        for profile_name in names:
            profile = load_profile(profile_name)
            for resource_pass in ("normalized", "tuned"):
                resources = _resources(profile, resource_pass)
                for engine in engines_mod.known_engines(refresh=True):
                    try:
                        body = ann_pass.render_config(engine, profile,
                                                      resources, resource_pass)
                    except ValueError:
                        continue          # a refusal is a legitimate answer
                    count = ann_config.expected_result_count(body)
                    assert count > 0, (profile_name, resource_pass, engine)
                    checked += 1
        assert checked >= 100, checked

    def test_an_unreadable_config_switches_the_check_off(self, tmp_path):
        """Zero, not a guess. A rendered config that cannot be read is a
        reason to fall back to today's behaviour, not a reason to fail a phase
        whose grid is unknown.
        """
        from orchestrator import ann_config
        assert ann_config.expected_results(str(tmp_path), "percona") == 0
        path = ann_config.config_path(str(tmp_path), "percona")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("float: [this: is\n  not: valid\n")
        assert ann_config.expected_results(str(tmp_path), "percona") == 0

    def test_a_body_with_an_unrunnable_run_group_counts_zero(self):
        """`prepare_args` raises on a run group carrying neither `args` nor
        `arg_groups`, so the run would not start at all. Counting only the
        groups that DO parse would under-count and hide a real shortfall."""
        from orchestrator import ann_config
        assert ann_config.expected_result_count(
            {"float": {"any": [{"name": "e", "base_args": [],
                                "run_groups": {
                                    "ok": {"arg_groups": [{"M": [8]}],
                                           "query_args": [[10]]},
                                    "bad": {"query_args": [[10]]}}}]}}) == 0

    def test_it_reads_the_file_the_container_will_read(self, tmp_path):
        from orchestrator import ann_config, ann_pass
        body = _render("percona", "full", "normalized")
        written = ann_pass.write_config(str(tmp_path), "percona", body)
        assert written == ann_config.config_path(str(tmp_path), "percona")
        assert ann_config.expected_results(str(tmp_path), "percona") == 24

    def test_the_names_are_still_reachable_through_ann_pass(self):
        """`cli.py` and a dozen existing tests call these on `ann_pass`. The
        split moved where they live, not who may call them."""
        from orchestrator import ann_config, ann_pass
        for name in ("config_path", "write_config", "expected_result_count",
                     "expected_results", "result_stem", "result_stems"):
            assert getattr(ann_pass, name) is getattr(ann_config, name), name


class _Resolved:
    """The resource figures `server_args` and `ann_fingerprint` substitute.

    A plain object rather than a real `ResolvedResources`, because nothing
    here depends on how they were derived -- only that they are stable, so
    the fingerprint and therefore the results tree is.
    """

    server_cpuset = "0-7"
    client_cpuset = "8-15"
    server_memory_bytes = 8 * 1024 ** 3
    client_memory_bytes = 2 * 1024 ** 3
    shm_size = "1g"
    buffer_bytes = 4 * 1024 ** 3
    graph_cache_bytes = 0
    maintenance_bytes = 512 * 1024 ** 2
    maxmemory_bytes = 0
    mongot_heap_bytes = 1 * 1024 ** 3
    build_threads = 8
    server_cpu_count = 8
    io_threads = 4
    pass_signature = "e10-fixed-signature"


class _Sampler:
    """`MemorySampler` without the thread or the container."""

    samples = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class _AnnHarness:
    """Everything `run_engine` needs, with the container replaced by a count.

    The fake `run_foreground` writes `files` result files where
    ann-benchmarks would and returns `rc`, which is the whole of the shape
    this defect lives in: ann-benchmarks forks a worker per definition, never
    checks its exit code, and returns 0 whether every definition finished or
    one of them died.
    """

    def __init__(self, tmp_path, monkeypatch, engine="percona",
                 dataset="gist-960-euclidean", resource_pass="normalized",
                 profile_name="full"):
        from orchestrator import ann_pass
        from orchestrator.config import load_engine, load_profile

        self.ann_pass = ann_pass
        self.engine = engine
        self.dataset = dataset
        self.resource_pass = resource_pass
        self.engine_cfg = load_engine(engine)
        self.profile = load_profile(profile_name)
        self.resolved = _Resolved()
        self.specs = []

        self.paths = {}
        for key in ("run_dir", "datasets", "annb_results", "engine_state",
                    "work_annb"):
            path = tmp_path / key
            path.mkdir(parents=True, exist_ok=True)
            self.paths[key] = str(path)

        # Merged, like production: `cli.py` never renders from a bare
        # `load_resources()`. Inert while `full.yml` carries no `resources:`
        # block, and wrong in both directions the moment it gains one.
        self.body = ann_pass.render_config(
            engine, self.profile, _resources(self.profile, resource_pass),
            resource_pass)
        ann_pass.write_config(self.paths["work_annb"], engine, self.body)

        self.results_dir = ann_pass.annb_results_dir(
            self.paths, resource_pass, self.resolved)

        monkeypatch.setattr(ann_pass.docker_ctl, "image_exists",
                            lambda image: True)
        monkeypatch.setattr(ann_pass.docker_ctl, "MemorySampler", _Sampler)
        monkeypatch.setattr(ann_pass, "prune_empty_results",
                            lambda directory, image: 0)
        self.monkeypatch = monkeypatch

    def seed(self, count):
        """Result files already on disk, i.e. `before`."""
        self._write(count)
        return self

    def _write(self, count):
        target = os.path.join(self.results_dir, self.dataset, "10", self.engine)
        os.makedirs(target, exist_ok=True)
        for index in range(count):
            open(os.path.join(target, f"r{index:02d}.hdf5"), "w").close()

    def drop_rendered_config(self):
        os.remove(self.ann_pass.config_path(self.paths["work_annb"],
                                            self.engine))
        return self

    def run(self, files, rc, lines=(), force=False):
        """Run the phase with a container that writes `files` and exits `rc`."""
        def run_foreground(spec, timeout=None, sink=None, line_filter=None,
                           stream=True):
            self.specs.append(spec)
            self._write(files)
            if sink is not None:
                sink.extend(lines)
            return rc

        self.monkeypatch.setattr(self.ann_pass.docker_ctl, "run_foreground",
                                 run_foreground)
        return self.ann_pass.run_engine(
            self.engine, self.dataset, self.profile, self.engine_cfg,
            self.resolved, self.resource_pass, self.paths, "r1", force=force)

    def kept_results(self):
        return self.ann_pass._count_results(self.results_dir, self.engine,
                                            self.dataset)


class TestAPartialSweepIsNotACompleteOne:
    """D3's probe and the three resume paths that already worked.

    What happened in run full-20261015-215006: the normalized gist-960 phase
    ran 7 h 46 m, lost its M=8 curve to `mariadb.InterfaceError: Server has
    gone away` during the parallel load, wrote 16 of 24 files, exited 0, and
    was checkpointed. `cli.py` saves a checkpoint only `if rc == 0`, so
    returning non-zero is what brings `--resume` back to the missing curve.
    """

    @pytest.mark.parametrize("engine,resource_pass",
                             sorted(FULL_PROFILE_COUNTS),
                             ids=[f"{e}-{p}"
                                  for e, p in sorted(FULL_PROFILE_COUNTS)])
    def test_every_engine_and_pass_is_judged_against_its_own_grid(
            self, engine, resource_pass, tmp_path, monkeypatch):
        """W6, and the reason every other case here cannot see it.

        `_AnnHarness` defaults to percona under `full`, and percona asks for 24
        in BOTH passes -- so `run_engine` reading the expectation for a
        hardcoded engine, or for a hardcoded pass, passed every test in this
        file. It is caught here and only here, because this drives the real
        `run_engine` for an engine whose number is different: mariadb is 24
        normalized and 16 tuned, so a complete 16-file tuned mariadb sweep
        judged against percona's 24 would be reported as a lost curve and
        checkpointed out of the run.

        Both directions, for all fourteen: the exact count passes, one file
        short fails.
        """
        want = FULL_PROFILE_COUNTS[(engine, resource_pass)]
        complete = _AnnHarness(tmp_path / "c", monkeypatch, engine=engine,
                               resource_pass=resource_pass)
        assert complete.run(want, 0) == 0, (
            f"{engine}/{resource_pass} wrote all {want} files and was still "
            f"reported as short")
        short = _AnnHarness(tmp_path / "s", monkeypatch, engine=engine,
                            resource_pass=resource_pass)
        assert short.run(want - 1, 0) == 1, (
            f"{engine}/{resource_pass} wrote {want - 1} of {want} files and "
            f"was reported as complete")

    def test_a_profile_with_a_resources_block_is_judged_against_the_merge(
            self, tmp_path, monkeypatch):
        """The only case in this class that can see the merge at all.

        Every other one drives `full`, which carries no `resources:` block, so
        merged and unmerged resources render identically and `_AnnHarness`
        reading a bare `load_resources()` was invisible -- correct by accident.
        `m-sweep` tuned/mariadb is 40 merged and 16 unmerged, because the
        profile clears the inherited `extras.m_values_by_engine` with `null`.

        So a phase that wrote 16 of those 40 curves must FAIL. Read unmerged,
        16 is the whole grid and it would be checkpointed complete -- which is
        D3's defect arriving through the resource layer instead of through the
        counter.
        """
        short = _AnnHarness(tmp_path / "s", monkeypatch, engine="mariadb",
                            resource_pass="tuned", profile_name="m-sweep")
        assert short.run(16, 0) == 1, (
            "16 of m-sweep's 40 tuned mariadb curves is a lost sweep; read "
            "against the unmerged grid it looks complete")
        whole = _AnnHarness(tmp_path / "w", monkeypatch, engine="mariadb",
                            resource_pass="tuned", profile_name="m-sweep")
        assert whole.run(40, 0) == 0

    def test_a_sweep_that_lost_a_curve_is_reported(self, tmp_path,
                                                    monkeypatch, capsys):
        """HONEST arm: 16 of the 24 the config asks for, exit 0 -- this run.
        MUTANT arm: all 24, exit 0 -- the same phase, complete.
        """
        honest = _AnnHarness(tmp_path / "h", monkeypatch).run(
            16, 0, lines=["mariadb.InterfaceError: Server has gone away",
                          "Terminating 1 workers"])
        message = capsys.readouterr().err
        mutant = _AnnHarness(tmp_path / "m", monkeypatch).run(24, 0)
        print(f"HONEST (16 of 24 files, exit 0) = {honest}")
        print(f"MUTANT (24 of 24 files, exit 0) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: a sweep missing a third of its result files returned "
            f"the same code as a complete one, so --resume will never come "
            f"back for the missing curve. Both arms returned {honest}")
        assert honest == 1
        assert mutant == 0
        assert "16 of 24" in message, (
            f"the message has to name how many of how many, or the operator "
            f"cannot tell a lost curve from a narrower grid: {message!r}")

    def test_nothing_to_do_stays_a_success(self, tmp_path, monkeypatch):
        """`after == expected == before`: a complete tree, re-run. The whole
        point of leaving the results tree in place between runs."""
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 0) == 0
        assert harness.kept_results() == 24

    def test_nothing_to_run_stays_a_success(self, tmp_path, monkeypatch):
        """ann-benchmarks signals "every configuration is done" by raising, so
        the process exits non-zero for what is successful resumption here."""
        from orchestrator import ann_pass
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(0, 1, lines=[
            "Traceback (most recent call last):",
            f"Exception: {ann_pass.NOTHING_TO_RUN}"]) == 0

    def test_nothing_to_run_over_a_short_tree_is_not_a_success(
            self, tmp_path, monkeypatch, capsys):
        """HONEST arm: 16 of 24 on disk. MUTANT arm: 24 of 24. Same exit, same
        output from ann-benchmarks; only the tree differs.

        "Nothing to run" does NOT mean "everything is done". `main()` raises
        it whenever the definition list is empty when it looks, and the resume
        filter is only one of the things that empties it: in --local mode
        `check_module_import_and_constructor` drops a definition whose module
        will not import with a logged WARNING and no raise, and it runs AFTER
        the resume filter. So a bench image whose algorithm module lost a
        dependency in a rebuild, over a tree holding 16 of 24, left 8
        definitions to run, dropped all 8, raised -- and this returned 0, the
        phase was checkpointed, and the report carried 16 of 24 curves. That
        is the defect the shortfall check exists to prevent, arriving through
        the one branch that was not gated on it.
        """
        from orchestrator import ann_pass
        said = [
            "WARNING ann_benchmarks.algorithms.percona.module could not be "
            "loaded; skipping",
            "Traceback (most recent call last):",
            f"Exception: {ann_pass.NOTHING_TO_RUN}",
        ]
        honest = _AnnHarness(tmp_path / "h", monkeypatch).seed(16).run(0, 1,
                                                                      lines=said)
        message = capsys.readouterr().err
        mutant = _AnnHarness(tmp_path / "m", monkeypatch).seed(24).run(0, 1,
                                                                      lines=said)
        print(f"HONEST (16 of 24 on disk, Nothing to run) = {honest}")
        print(f"MUTANT (24 of 24 on disk, Nothing to run) = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: a 'Nothing to run' over a tree missing a third of "
            f"its files returned the same code as one over a complete tree, "
            f"so --resume will never come back for it. Both arms returned "
            f"{honest}")
        assert honest == 1
        assert mutant == 0
        assert "16 result file(s)" in message and "24" in message, message
        assert "could not be loaded" in message, (
            "the message has to name the import failure as a cause, because "
            f"that is the one an operator would not guess: {message!r}")

    def test_nothing_to_run_with_an_unknown_grid_stays_a_success(
            self, tmp_path, monkeypatch):
        """`expected == 0` switches the gate off here too. Without a grid to
        compare against there is no evidence of a shortfall, and failing a
        resumption on no evidence would break the path that works."""
        from orchestrator import ann_pass
        harness = _AnnHarness(tmp_path, monkeypatch).seed(16)
        harness.drop_rendered_config()
        assert harness.run(0, 1, lines=[
            f"Exception: {ann_pass.NOTHING_TO_RUN}"]) == 0

    def test_a_tree_holding_more_than_this_profile_asks_for_is_not_a_failure(
            self, tmp_path, monkeypatch):
        """The results tree is shared across runs and keyed only by the
        resource configuration, so a wider grid measured earlier at the same
        budget is legitimately still in it."""
        harness = _AnnHarness(tmp_path, monkeypatch).seed(40)
        assert harness.run(40, 0) == 0
        assert harness.kept_results() == 40

    def test_force_still_recomputes_and_still_passes(self, tmp_path,
                                                      monkeypatch):
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 0, force=True) == 0
        assert "--force" in harness.specs[0].command

    def test_a_sweep_that_produced_nothing_is_still_reported(self, tmp_path,
                                                              monkeypatch):
        """The total loss was already caught; adding the partial case must not
        move it."""
        harness = _AnnHarness(tmp_path, monkeypatch)
        assert harness.run(0, 0, lines=["Terminating 1 workers"]) == 1

    def test_an_unknown_grid_falls_back_to_todays_behaviour(self, tmp_path,
                                                             monkeypatch):
        """With no rendered config to read, `expected` is 0 and the check is
        off -- a phase is not failed over a grid nothing could establish."""
        harness = _AnnHarness(tmp_path, monkeypatch).drop_rendered_config()
        assert harness.run(16, 0) == 0

    def test_a_non_zero_exit_is_still_a_non_zero_exit(self, tmp_path,
                                                       monkeypatch):
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 7) == 7

    def test_the_phase_log_lands_under_this_dataset(self, tmp_path,
                                                     monkeypatch):
        """D2 and D3 meet here: the log naming the dead worker is the one that
        used to be overwritten by the next dataset."""
        harness = _AnnHarness(tmp_path, monkeypatch)
        harness.run(16, 0, lines=["mariadb.InterfaceError: Server has gone away"])
        logs = os.listdir(os.path.join(harness.paths["run_dir"], "logs"))
        assert logs == ["percona-gist-960-euclidean-ann-normalized.log"], logs

    def test_a_short_sweep_is_not_also_reported_as_done(self, tmp_path,
                                                        monkeypatch, capsys):
        """The shortfall test runs before anything is called complete.

        It used to run after, so a seven-hour phase that lost a curve ended
        with "nothing to do: all 16 configuration(s) ... already have results"
        on stdout and "FAILED: 16 of 24" on stderr -- a log arguing with
        itself about what had just happened, in the place an operator looks
        first.
        """
        harness = _AnnHarness(tmp_path, monkeypatch).seed(16)
        assert harness.run(16, 0) == 1
        captured = capsys.readouterr()
        assert "nothing to do" not in captured.out, captured.out
        assert "new result file" not in captured.out, captured.out
        assert "16 of 24" in captured.err

    def test_a_complete_sweep_is_still_reported_as_done(self, tmp_path,
                                                        monkeypatch, capsys):
        """Non-vacuity for the test above: the reporting still happens, it is
        only gated behind the shortfall test."""
        harness = _AnnHarness(tmp_path, monkeypatch).seed(24)
        assert harness.run(24, 0) == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_new_results_are_still_counted_out(self, tmp_path, monkeypatch,
                                               capsys):
        harness = _AnnHarness(tmp_path, monkeypatch).seed(8)
        assert harness.run(24, 0) == 0
        assert "16 new result file(s), 24 total" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The citations this change moved, and could not see moving
# ---------------------------------------------------------------------------

#: What docs/09 §4.10 says about `render_config`, as the strings the section
#: quotes rather than as positions in a file.
#:
#: WHY THIS IS HERE AT ALL, AND WHAT CHANGED. §4.10 used to write nine
#: citations as a BARE `:N`, taking the file from its heading, and
#: `tests/test_e1_percona_design.py`'s citation guard needed a path in the
#: backticks -- so the one section devoted to `ann_pass.py` was the one
#: section whose pointers into it were unchecked, and all nine went stale the
#: moment a change put an import block above `render_config`. They were found
#: by hand, which is what this table was written to stop happening again.
#:
#: The section cites symbols now. Three consequences, and the third is the
#: reason this class is shorter than it was:
#:
#: * There are no numbers left to go stale, so the "re-derive it with grep -n"
#:   failure message has nothing to ask for.
#: * The parenthetical ENGINE LABELS are gone, because the citation is the
#:   branch condition itself. §4.10 said "`:206` (mongodb)" against Percona's
#:   own branch for two revisions, directly under a sentence saying Percona
#:   had no branch; a label that is the thing it labels cannot disagree with
#:   it. That whole failure mode is now unreachable rather than guarded.
#: * What is LEFT to check is membership, order and placement: that each
#:   quoted condition is a branch of `render_config`, that the section lists
#:   them in the order the function tests them, and that the three
#:   `efConstruction` lines sit one per branch. The permutation this class was
#:   extended to catch -- swapping two engines -- is an order claim, so it is
#:   still checked as one.
#:
#: The anchors live in `CITED_SYMBOLS` in `tests/test_e1_percona_design.py`,
#: which asserts each of these strings is inside `render_config`. This class
#: asserts the three things a per-symbol anchor cannot: order, uniqueness and
#: which branch a repeated line belongs to.
SECTION_4_10_BRANCHES = (
    'if engine in MARIADB_ENGINES + ("alisql",)',
    'elif engine == "pgvector":',
    'elif engine == "valkey":',
    'elif engine == "percona":',
    'elif engine == "mongodb":',
)

#: The two build-mode loops §4.10 quotes, and the engine each belongs to.
SECTION_4_10_BUILD_MODE_LOOPS = {
    "pgvector": 'for build_mode in ann.get("pgvector_build_modes", ["post"]):',
    "valkey": 'for build_mode in ann.get("valkey_build_modes", ["post"]):',
}


def _section_4_10():
    with open(os.path.join(REPO, "docs", "09-adding-percona-server.md")) as fh:
        doc = fh.read()
    return doc.split("### 4.10")[1].split("\n## ")[0]


def _render_config_body():
    with open(os.path.join(REPO, "orchestrator", "ann_pass.py")) as fh:
        source = fh.read()
    return source.split("def render_config", 1)[1].split("\ndef ", 1)[0]


class TestSection410StillDescribesRenderConfig:
    """Nine pointers into a file this change reorganised, now by name.

    The section is `### 4.10 orchestrator/ann_pass.py`, so everything it
    quotes is a line of that file. Nothing here reads a number.
    """

    def test_the_section_still_quotes_the_branches(self):
        """Non-vacuity: if the quotations were ever rewritten into another
        form the tests below would pass by finding nothing."""
        section = _section_4_10()
        for branch in SECTION_4_10_BRANCHES:
            assert branch in section, branch
        for loop in SECTION_4_10_BUILD_MODE_LOOPS.values():
            assert loop in section, loop
        assert '"efConstruction": ef_constructions' in section
        assert 'raise ValueError(f"unknown engine: {engine}")' in section
        # And it must carry no line citation of its own any more -- the form
        # it used for all nine. Checked here as well as in the document-wide
        # ratchet, because this section is where the form was invisible.
        assert not re.search(r"`:\d+", section), (
            "§4.10 is back to citing bare line numbers, which is the form "
            "that left all nine of its pointers unchecked")

    @pytest.mark.parametrize("branch", SECTION_4_10_BRANCHES)
    def test_each_quoted_branch_is_a_branch_of_render_config(self, branch):
        """Every condition the section quotes must occur exactly once in the
        function -- that is what makes "the section cites it" checkable."""
        body = _render_config_body()
        assert body.count(branch) == 1, (
            f"{branch!r} occurs {body.count(branch)} times in "
            f"render_config; §4.10's quotation of it is only checkable while "
            f"it is unique")

    def test_the_section_lists_the_branches_in_the_order_they_are_tested(self):
        """The permutation this class was extended to catch.

        Set membership cannot see a swap: exchanging the pgvector and valkey
        citations left the old version passing. The section's list is an
        ordered claim -- "its branches, in the order the function tests them"
        -- so it is checked as one, in both the document and the function.
        """
        section = _section_4_10()
        body = _render_config_body()
        in_doc = [section.index(b) for b in SECTION_4_10_BRANCHES]
        in_code = [body.index(b) for b in SECTION_4_10_BRANCHES]
        assert in_doc == sorted(in_doc), (
            "§4.10 lists render_config's branches out of order; it claims to "
            "list them in the order the function tests them")
        assert in_code == sorted(in_code), (
            "render_config's branches are no longer in the order §4.10 lists "
            "them; the section has to be re-read against the function")

    def test_the_raise_is_the_fallback_every_branch_misses(self):
        """§4.10's whole point: Percona matched none of them and landed here.

        An ORDER claim, which is what "landed on the `raise`" means
        mechanically -- the raise comes after the last engine branch. It is
        not the last statement in the function, and an earlier draft of this
        test asserted that it was and failed on the `return` beneath it.
        """
        body = _render_config_body()
        statement = 'raise ValueError(f"unknown engine: {engine}")'
        assert statement in body
        assert body.rindex("elif engine ==") < body.index(statement)

    @pytest.mark.parametrize("engine", sorted(SECTION_4_10_BUILD_MODE_LOOPS))
    def test_each_build_mode_loop_sits_in_its_own_branch(self, engine):
        """The loop §4.10 attributes to an engine has to be inside that
        engine's branch and not in a sibling's."""
        body = _render_config_body()
        loop = SECTION_4_10_BUILD_MODE_LOOPS[engine]
        assert body.count(loop) == 1, (engine, body.count(loop))
        start, end = _branch_span(body, engine)
        assert start < body.index(loop) < end, (
            f"§4.10 attributes {loop!r} to {engine}, and it is outside "
            f"{engine}'s branch")

    def test_all_three_ef_construction_lines_are_one_per_branch(self):
        """`"efConstruction": ef_constructions` occurs THREE times now.

        §4.10's bullet said "the two that do are ..." in the present tense
        long after Percona's branch had added a third. The bullet says three
        and names the percona one as the third, so each occurrence must fall
        inside one of the three branches that emit it, one each.
        """
        section = _section_4_10()
        body = _render_config_body()
        at = [m.start() for m in
              re.finditer(r'"efConstruction": ef_constructions', body)]
        # Five since VillageSQL's branch, which has the same two knobs.
        assert len(at) == 5, at
        assert re.search(r"occurs five times", section, re.I), (
            "§4.10 must state how many there are; the count is the thing "
            "that went stale when Percona's branch added the third")
        for engine in ("pgvector", "valkey", "percona", "myvector",
                       "villagesql"):
            start, end = _branch_span(body, engine)
            inside = [n for n in at if start < n < end]
            assert len(inside) == 1, (engine, inside)

    def test_the_branch_span_helper_can_fail(self):
        """The helper every placement assertion above rests on, shown able to
        come back empty rather than silently spanning the whole function."""
        body = _render_config_body()
        with pytest.raises(AssertionError):
            _branch_span(body, "no_such_engine_zq")
        start, end = _branch_span(body, "percona")
        assert start < end
        assert 'elif engine == "percona":' in body[start:end]
        assert 'elif engine == "mongodb":' not in body[start:end]


def _branch_span(body: str, engine: str):
    """`(start, end)` offsets of one `elif engine == "<name>":` branch.

    A branch runs to the next branch condition in file order, or to the
    function's trailing `return`. Offsets rather than line numbers, because a
    line number is the thing this whole section stopped using.
    """
    head = f'elif engine == "{engine}":'
    assert head in body, f"render_config has no {engine} branch"
    start = body.index(head)
    later = [m.start() for m in re.finditer(r'elif engine == "\w+":', body)
             if m.start() > start]
    return start, (min(later) if later else len(body))
