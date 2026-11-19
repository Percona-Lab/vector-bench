"""The ann load's writer count follows the resource pass, for every engine.

THE DEFECT. `ann_pass.run_engine` passed `ann.insert_threads` through as
`VB_INSERT_THREADS` with a default of 0, and `vb_mysql._resolve_insert_threads`
turns 0 into min(8, cpuset). No normalized profile sets the key, so under the
pass that declares `build.threads: 1` for every engine, Percona, AliSQL,
MyVector and VillageSQL loaded their ann corpus with up to eight writers while
MariaDB loaded with one (`MARIADB.max_insert_threads`, PR #13) and pgvector,
Valkey and Percona Search load over one connection regardless. The normalized
ann `build_time` of the first four was an 8-writer figure beside 1-writer
figures. These tests hold the count to the pass's resolved `build_threads`.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import sysinfo  # noqa: E402
from orchestrator import ann_pass  # noqa: E402
from orchestrator.config import (available_profiles,  # noqa: E402
                                 load_profile, load_resources,
                                 merge_resource_overrides, resolve_resources)

from .test_e10_failures_explain_themselves import _AnnHarness  # noqa: E402

#: Every MySQL-family engine whose ann module resolves VB_INSERT_THREADS.
MYSQL_FAMILY = ("percona", "alisql", "mariadb", "mariadb123", "myvector",
                "villagesql", "villagesql84")


def _resolved(build_threads):
    return SimpleNamespace(build_threads=build_threads)


class TestTheRule:
    def test_unset_takes_the_passs_build_threads(self):
        assert ann_pass.ann_insert_threads({"ann": {}}, _resolved(1)) == (
            1, None)
        assert ann_pass.ann_insert_threads({"ann": {}}, _resolved(8)) == (
            8, None)

    def test_a_profile_may_lower_it(self):
        profile = {"ann": {"insert_threads": 4}}
        assert ann_pass.ann_insert_threads(profile, _resolved(8)) == (4, None)

    def test_a_profile_cannot_raise_it_and_is_told(self):
        profile = {"ann": {"insert_threads": 8}}
        threads, note = ann_pass.ann_insert_threads(profile, _resolved(1))
        assert threads == 1
        assert note and "insert_threads=8" in note and "build_threads=1" in note

    def test_never_zero_which_the_overlay_reads_as_min_8_cpus(self):
        assert ann_pass.ann_insert_threads({}, _resolved(0))[0] == 1


class TestEveryShippedProfile:
    @pytest.mark.parametrize("name", sorted(available_profiles()))
    def test_normalized_loads_with_one_writer(self, name):
        """The normalized pass merged with the profile's own `resources:`
        block the way `cmd_run` merges it, so no profile can raise the pass's
        one build thread -- including the four tuned profiles that pin
        `insert_threads: 8`. The declared figure is read rather than resolved
        on this host: resolution also sizes cpusets, which refuses some
        profiles on a small machine and says nothing about threads."""
        profile = load_profile(name)
        resources = merge_resource_overrides(load_resources("normalized"),
                                             profile)
        declared = int((resources.get("build") or {}).get("threads", 1))
        assert declared == 1
        assert ann_pass.ann_insert_threads(profile,
                                           _resolved(declared))[0] == 1

    def test_the_resolver_turns_normalized_into_one_build_thread(self):
        resolved = resolve_resources(load_resources("normalized"), "percona",
                                     sysinfo.collect(), max_clients=1)
        assert resolved.build_threads == 1

    @pytest.mark.parametrize("name", sorted(available_profiles()))
    def test_tuned_is_what_it_was_on_a_host_with_eight_cores(self, name):
        """min(8, cpuset) before, the cpuset capped at `build.max_threads`
        (8) now: the same 8 for every tuned profile on an 8-core cpuset."""
        profile = load_profile(name)
        declared = int(profile.get("ann", {}).get("insert_threads") or 0)
        assert ann_pass.ann_insert_threads(profile, _resolved(8))[0] == (
            declared or 8)


class TestTheContainerIsHandedIt:
    @pytest.mark.parametrize("engine", MYSQL_FAMILY)
    @pytest.mark.parametrize("resource_pass,build_threads,expected",
                             [("normalized", 1, "1"), ("tuned", 8, "8")])
    def test_vb_insert_threads(self, tmp_path, monkeypatch, capsys, engine,
                               resource_pass, build_threads, expected):
        h = _AnnHarness(tmp_path, monkeypatch, engine=engine,
                        dataset="fashion-mnist-784-euclidean",
                        resource_pass=resource_pass, profile_name="main")
        h.resolved.build_threads = build_threads
        h.run(files=0, rc=1)
        assert h.specs, "the container was never launched"
        assert h.specs[0].env["VB_INSERT_THREADS"] == expected
        assert f"up to {expected} thread(s)" in capsys.readouterr().out

    def test_the_tuned_profiles_8_is_lowered_under_normalized(
            self, tmp_path, monkeypatch, capsys):
        h = _AnnHarness(tmp_path, monkeypatch, engine="percona",
                        dataset="fashion-mnist-784-euclidean",
                        resource_pass="normalized",
                        profile_name="tuned-complete")
        h.resolved.build_threads = 1
        h.run(files=0, rc=1)
        assert h.specs[0].env["VB_INSERT_THREADS"] == "1"
        assert "insert_threads=8 exceeds" in capsys.readouterr().err
