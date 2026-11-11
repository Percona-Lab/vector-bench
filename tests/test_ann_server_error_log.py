"""The ann pass keeps the in-container server's own error log.

Run percona-mariadb-20261110 lost Percona's ann load to "connection reset by
peer" on two datasets, and whether the server had crashed could not be
answered: the ann container runs the server in-process and is gone by the
time anyone looks, its `--log-error` file sat root-owned in the phase's
state directory, and the next phase deleted that directory. The ops pass has
archived the same file all along (`OpsRun._save_server_error_log`).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from orchestrator import ann_pass, docker_ctl  # noqa: E402


def _completed(rc, out=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr="")


class TestReadHostFileTail:
    def test_a_readable_file_is_read_directly(self, tmp_path, monkeypatch):
        f = tmp_path / "percona.err"
        f.write_text("\n".join(f"line {i}" for i in range(10)))
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda *a, **k: pytest.fail("no docker needed"))
        assert docker_ctl.read_host_file_tail(str(f), "img", lines=3) == \
            "line 7\nline 8\nline 9"

    def test_a_missing_file_is_none_without_docker(self, tmp_path,
                                                   monkeypatch):
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda *a, **k: pytest.fail("no docker needed"))
        assert docker_ctl.read_host_file_tail(
            str(tmp_path / "absent.err"), "img") is None

    def test_an_unreadable_file_is_read_through_the_image(self, tmp_path,
                                                          monkeypatch):
        f = tmp_path / "percona.err"
        f.write_text("x")
        calls = []

        def fake_open(*a, **k):
            raise PermissionError("root:root 0640")

        def fake_run(args, **k):
            calls.append(args)
            return _completed(0, "mysqld got signal 11\n")

        monkeypatch.setattr("builtins.open", fake_open)
        monkeypatch.setattr(docker_ctl, "_run", fake_run)
        assert docker_ctl.read_host_file_tail(str(f), "vb/percona", 50) == \
            "mysqld got signal 11"
        (args,) = calls
        assert args[:3] == ["docker", "run", "--rm"]
        assert f"{tmp_path}:/target:ro" in args
        assert args[-4:] == ["vb/percona", "-n", "50", "/target/percona.err"]

    @pytest.mark.parametrize("outcome", ("rc", "raise"))
    def test_every_docker_failure_is_none(self, tmp_path, monkeypatch,
                                          outcome):
        f = tmp_path / "percona.err"
        f.write_text("x")

        def fake_run(args, **k):
            if outcome == "raise":
                raise subprocess.TimeoutExpired(args, 120)
            return _completed(1)

        monkeypatch.setattr("builtins.open",
                            lambda *a, **k: (_ for _ in ()).throw(
                                PermissionError()))
        monkeypatch.setattr(docker_ctl, "_run", fake_run)
        assert docker_ctl.read_host_file_tail(str(f), "img") is None


class TestSaveServerErrorLog:
    def _paths(self, tmp_path):
        return {"run_dir": str(tmp_path / "run")}

    def test_percona_s_log_is_saved_beside_the_ann_log(self, tmp_path,
                                                       monkeypatch):
        state = tmp_path / "state"
        state.mkdir()
        (state / "percona.err").write_text("x")
        seen = []

        def fake_read(path, image, lines):
            seen.append((path, image, lines))
            return "mysqld got signal 11\nbacktrace"

        monkeypatch.setattr(docker_ctl, "read_host_file_tail", fake_read)
        saved = ann_pass.save_server_error_log(
            self._paths(tmp_path), "percona", "sift-128-euclidean",
            "normalized", str(state), "vb/percona")
        # /var/lib/vbench/percona.err under data_mount /var/lib/vbench.
        assert seen == [(str(state / "percona.err"), "vb/percona",
                         ann_pass.SERVER_LOG_TAIL)]
        assert os.path.basename(saved) == \
            "percona-sift-128-euclidean-ann-server-error-normalized.log"
        assert open(saved).read().startswith("mysqld got signal 11")

    @pytest.mark.parametrize("engine", ("mariadb", "alisql", "myvector",
                                        "villagesql"))
    def test_every_mysql_family_engine_is_covered(self, tmp_path,
                                                  monkeypatch, engine):
        name = "mariadb" if engine == "mariadb" else engine
        (tmp_path / f"{name}.err").write_text("x")
        monkeypatch.setattr(docker_ctl, "read_host_file_tail",
                            lambda *a, **k: "log")
        assert ann_pass.save_server_error_log(
            self._paths(tmp_path), engine, "d", "normalized",
            str(tmp_path), "img")

    def test_an_engine_without_an_error_log_is_skipped(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(docker_ctl, "read_host_file_tail",
                            lambda *a, **k: pytest.fail("pgvector logs to "
                                                        "the console"))
        assert ann_pass.save_server_error_log(
            self._paths(tmp_path), "pgvector", "d", "normalized",
            str(tmp_path), "img") is None

    def test_no_file_is_silent(self, tmp_path, monkeypatch, capsys):
        """A resume with nothing to run never starts the server."""
        monkeypatch.setattr(docker_ctl, "read_host_file_tail",
                            lambda *a, **k: pytest.fail("nothing to read"))
        assert ann_pass.save_server_error_log(
            self._paths(tmp_path), "percona", "d", "normalized",
            str(tmp_path), "img") is None
        assert capsys.readouterr().err == ""

    def test_mongodb_s_nested_log(self, tmp_path, monkeypatch):
        (tmp_path / "mongot").mkdir()
        (tmp_path / "mongot" / "mongot.log").write_text("x")
        seen = []
        monkeypatch.setattr(docker_ctl, "read_host_file_tail",
                            lambda path, *a, **k: seen.append(path) or "log")
        assert ann_pass.save_server_error_log(
            self._paths(tmp_path), "mongodb", "d", "normalized",
            str(tmp_path), "img")
        assert seen == [str(tmp_path / "mongot" / "mongot.log")]

    def test_nothing_readable_is_a_note_not_a_failure(self, tmp_path,
                                                      monkeypatch, capsys):
        (tmp_path / "percona.err").write_text("")
        monkeypatch.setattr(docker_ctl, "read_host_file_tail",
                            lambda *a, **k: None)
        assert ann_pass.save_server_error_log(
            self._paths(tmp_path), "percona", "d", "normalized",
            str(tmp_path), "img") is None
        assert "nothing could be read" in capsys.readouterr().err

    def test_run_engine_saves_it_after_every_phase(self):
        """After the ann log and before any return: a failed phase and a
        resumed one both reach it."""
        src = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        body = src[src.index("\ndef run_engine("):]
        log = body.index('"ann", resource_pass, output)')
        call = body.index("save_server_error_log(paths, engine, dataset")
        first_return = body.index("\n            return 0", log)
        assert log < call < first_return
