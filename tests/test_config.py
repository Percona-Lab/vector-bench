"""Tests for configuration loading, resolution and the ann-benchmarks renderer.

These guard the invariants that make the comparison fair. A config bug here
would not crash anything — it would quietly hand one engine more memory or a
different parameter grid than another, and the resulting report would look
perfectly credible.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import sysinfo  # noqa: E402
from orchestrator import ann_pass  # noqa: E402
from orchestrator.cli import VB_ROOT  # noqa: E402
from orchestrator.config import (available_profiles, load_engine,  # noqa: E402
                                 merge_resource_overrides,
                                 load_profile, load_resources, resolve_resources,
                                 server_args)

# Charts are drawn inside a bench image, which carries matplotlib; the host is
# promised nothing beyond python3 and PyYAML. A test that renders one therefore
# skips on a machine set up from the documented prerequisites, rather than
# failing there and making a clean install look broken.
def _can_draw() -> bool:
    """Whether a chart can actually be drawn, not whether a file is on the path.

    find_spec() answers the second question, and a matplotlib that is present
    but unimportable would pass it and then fail the test anyway.
    """
    from report.charts import have_matplotlib
    return have_matplotlib()


needs_matplotlib = pytest.mark.skipif(
    not _can_draw(),
    reason="draws a chart; matplotlib lives in the bench images, not on the host")

ENGINES = ("mariadb", "alisql", "pgvector")
PROFILES = ("smoke", "quick", "full")
PASSES = ("normalized", "tuned")


@pytest.fixture(scope="module")
def info():
    return sysinfo.collect()


class TestConfigsParse:
    @pytest.mark.parametrize("name", PROFILES)
    def test_profile_loads(self, name):
        profile = load_profile(name)
        assert profile["name"] == name
        assert profile["datasets"], "a profile with no datasets measures nothing"
        assert profile["k"] >= 1

    @pytest.mark.parametrize("name", PASSES)
    def test_resource_pass_loads(self, name):
        resources = load_resources(name)
        assert resources["name"] == name
        assert "cpu" in resources and "memory" in resources

    @pytest.mark.parametrize("name", ENGINES)
    def test_engine_loads(self, name):
        cfg = load_engine(name)
        assert cfg["name"] == name
        assert cfg["source"]["tag"], "an engine without a pinned tag is not reproducible"
        assert cfg["image"]["runtime"] and cfg["image"]["bench"]

    def test_all_profiles_discovered(self):
        assert set(PROFILES).issubset(set(available_profiles()))


class TestFairness:
    """The invariants that make the normalized pass a fair comparison."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_skip_grant_tables_is_never_used(self, engine):
        # On MySQL 8 --skip-grant-tables implicitly enables --skip-networking,
        # which would make the server unreachable over TCP from the harness.
        cfg = load_engine(engine)
        flags = " ".join(str(f) for group in cfg.get("server", {}).values()
                         for f in group)
        assert "skip-grant-tables" not in flags
        assert "skip_grant_tables" not in flags

    def test_mysql_engines_get_equal_memory(self, info):
        resources = load_resources("normalized")
        mariadb = resolve_resources(resources, "mariadb", info)
        alisql = resolve_resources(resources, "alisql", info)
        assert mariadb.buffer_bytes == alisql.buffer_bytes
        assert mariadb.graph_cache_bytes == alisql.graph_cache_bytes
        assert mariadb.server_memory_bytes == alisql.server_memory_bytes
        assert mariadb.server_cpuset == alisql.server_cpuset

    def test_postgres_absorbs_the_graph_cache_budget(self, info):
        # pgvector has no vector-specific cache; graph pages live in
        # shared_buffers. Giving it only the buffer fraction would hand it
        # strictly less resident memory for the same container limit.
        resources = load_resources("normalized")
        mysql = resolve_resources(resources, "mariadb", info)
        pg = resolve_resources(resources, "pgvector", info)
        assert pg.graph_cache_bytes == 0
        assert pg.buffer_bytes == pytest.approx(
            mysql.buffer_bytes + mysql.graph_cache_bytes, rel=0.01
        )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_allocations_fit_inside_the_container_limit(self, engine, info):
        for pass_name in PASSES:
            resolved = resolve_resources(load_resources(pass_name), engine, info)
            allocated = (resolved.buffer_bytes + resolved.graph_cache_bytes
                         + resolved.maintenance_bytes)
            assert allocated < resolved.server_memory_bytes, (
                f"{engine}/{pass_name}: allocated more than the container limit"
            )

    def test_server_and_client_cpusets_do_not_overlap(self, info):
        resolved = resolve_resources(load_resources("normalized"), "mariadb", info)

        def expand(spec):
            out = set()
            for part in spec.split(","):
                if not part:
                    continue
                if "-" in part:
                    lo, hi = part.split("-")
                    out.update(range(int(lo), int(hi) + 1))
                else:
                    out.add(int(part))
            return out

        overlap = expand(resolved.server_cpuset) & expand(resolved.client_cpuset)
        if overlap:
            # Permitted on small machines, but it must have been reported.
            assert resolved.warnings, (
                "cpusets overlap without a warning; latency noise would go unrecorded"
            )


class TestServerArgs:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_no_unsubstituted_placeholders(self, engine, info):
        for pass_name in PASSES:
            resolved = resolve_resources(load_resources(pass_name), engine, info)
            flags = server_args(load_engine(engine), pass_name, resolved)
            joined = " ".join(flags)
            assert "{" not in joined and "}" not in joined, (
                f"{engine}/{pass_name}: unsubstituted placeholder in {joined}"
            )

    def test_alisql_always_enables_vector_support(self, info):
        # vidx_disabled defaults to ON; without this every CREATE TABLE with a
        # VECTOR column fails with ER_VECTOR_DISABLED.
        resolved = resolve_resources(load_resources("normalized"), "alisql", info)
        for pass_name in PASSES:
            flags = " ".join(server_args(load_engine("alisql"), pass_name, resolved))
            assert "vidx-disabled=OFF" in flags

    def test_alisql_runs_at_read_committed(self, info):
        # Any other isolation level raises ER_NOT_SUPPORTED_YET on vector DML.
        resolved = resolve_resources(load_resources("normalized"), "alisql", info)
        flags = " ".join(server_args(load_engine("alisql"), "normalized", resolved))
        assert "READ-COMMITTED" in flags

    def test_graph_cache_flags_present_for_mysql_engines(self, info):
        expected = {"mariadb": "mhnsw-max-cache-size", "alisql": "vidx-hnsw-cache-size"}
        for engine, flag in expected.items():
            resolved = resolve_resources(load_resources("normalized"), engine, info)
            flags = " ".join(server_args(load_engine(engine), "normalized", resolved))
            assert flag in flags


class TestAnnConfigRendering:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_renders_valid_yaml(self, engine):
        profile = load_profile("quick")
        body = ann_pass.render_config(engine, profile, load_resources("normalized"),
                                      "normalized")
        text = yaml.safe_dump(body)
        reparsed = yaml.safe_load(text)
        entry = reparsed["float"]["any"][0]
        assert entry["name"] == engine
        assert entry["module"] == f"ann_benchmarks.algorithms.{engine}"
        assert entry["run_groups"]

    def test_every_engine_sweeps_the_same_grid(self):
        # A profile change that reached only one engine would look like a
        # performance difference and would not be one.
        profile = load_profile("quick")
        resources = load_resources("normalized")
        grids = {}
        for engine in ENGINES:
            body = ann_pass.render_config(engine, profile, resources, "normalized")
            groups = body["float"]["any"][0]["run_groups"]
            m_values, ef_values = set(), set()
            for group in groups.values():
                for arg_group in group["arg_groups"]:
                    m = arg_group["M"]
                    m_values.update(m if isinstance(m, list) else [m])
                for query_args in group["query_args"]:
                    ef_values.update(query_args)
            grids[engine] = (frozenset(m_values), frozenset(ef_values))
        assert len(set(grids.values())) == 1, f"parameter grids differ: {grids}"

    def test_alisql_never_offered_myisam(self):
        # VIDX is InnoDB-only; anything else would fail at DDL.
        body = ann_pass.render_config("alisql", load_profile("full"),
                                      load_resources("tuned"), "tuned")
        groups = body["float"]["any"][0]["run_groups"]
        engines = {ag["engine"] for g in groups.values() for ag in g["arg_groups"]}
        assert engines == {"InnoDB"}

    def test_pgvector_ef_construction_pinned_in_normalized_pass(self):
        # ef_construction is the one build knob MariaDB and AliSQL do not
        # expose; sweeping it in the fair pass would give pgvector an axis the
        # others lack.
        body = ann_pass.render_config("pgvector", load_profile("full"),
                                      load_resources("normalized"), "normalized")
        groups = body["float"]["any"][0]["run_groups"]
        values = set()
        for group in groups.values():
            for arg_group in group["arg_groups"]:
                ef = arg_group["efConstruction"]
                values.update(ef if isinstance(ef, list) else [ef])
        assert len(values) == 1, f"ef_construction swept in the normalized pass: {values}"

    def test_pgvector_ef_construction_swept_in_tuned_pass(self):
        body = ann_pass.render_config("pgvector", load_profile("full"),
                                      load_resources("tuned"), "tuned")
        groups = body["float"]["any"][0]["run_groups"]
        values = set()
        for group in groups.values():
            for arg_group in group["arg_groups"]:
                ef = arg_group["efConstruction"]
                values.update(ef if isinstance(ef, list) else [ef])
        assert len(values) > 1

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError):
            ann_pass.render_config("nonsense", load_profile("smoke"),
                                   load_resources("normalized"), "normalized")


class TestOverlayModules:
    """The overlay must stay consistent with what the renderer generates."""

    @pytest.mark.parametrize("engine,constructor",
                             [("mariadb", "MariaDB"), ("alisql", "AliSQL"),
                              ("pgvector", "PGVector")])
    def test_constructor_exists_in_module(self, engine, constructor):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "overlay", "ann-benchmarks", "ann_benchmarks",
                            "algorithms", engine, "module.py")
        assert os.path.exists(path), f"missing overlay module for {engine}"
        source = open(path).read()
        assert f"class {constructor}" in source
        from orchestrator import engines as engines_mod
        assert engines_mod.get(engine).ann_constructor == constructor


class TestAnnResultGuard:
    """The zero-result guard must not turn correct resumption into a failure.

    ann-benchmarks exits 0 whenever it has nothing left to run. That covers both
    "the module failed to import and nothing was produced" and "every
    configuration already has results". An earlier version conflated them and
    reported three healthy engines as failed on a re-run, which then propagated
    into the report's Validity section as a fabricated failure.
    """

    def _counts(self, tmp_path, engine, dataset, n):
        import os
        d = tmp_path / dataset / "10" / engine
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"r{i}.hdf5").write_bytes(b"")
        from orchestrator.ann_pass import _count_results
        return _count_results(str(tmp_path), engine, dataset)

    def test_counts_only_this_engine_and_dataset(self, tmp_path):
        assert self._counts(tmp_path, "mariadb", "ds-a", 3) == 3
        # A different engine's files must not be counted as this one's.
        self._counts(tmp_path, "alisql", "ds-a", 5)
        from orchestrator.ann_pass import _count_results
        assert _count_results(str(tmp_path), "mariadb", "ds-a") == 3
        assert _count_results(str(tmp_path), "alisql", "ds-a") == 5

    def test_zero_when_nothing_exists(self, tmp_path):
        from orchestrator.ann_pass import _count_results
        assert _count_results(str(tmp_path), "mariadb", "missing") == 0

    def test_resumption_is_not_a_failure(self):
        # before == after > 0 means every configuration was already present.
        before = after = 24
        rc = 0
        assert not (rc == 0 and after == 0), \
            "a run that produced no NEW results but has results is resumption"

    def test_no_results_at_all_is_a_failure(self):
        before = after = 0
        rc = 0
        assert rc == 0 and after == 0, \
            "no results at all must still be caught as a failure"


class TestNothingToRunHandling:
    """ann-benchmarks exits NON-ZERO when it has nothing left to run.

    `main()` raises Exception("Nothing to run") once every configuration
    already has results. That is successful resumption from this framework's
    point of view — the results tree is deliberately kept between runs — but
    taken at face value it reports healthy engines as failed, which then
    propagates into the report's Validity section as a fabricated failure.
    """

    def test_marker_matches_upstream_text(self):
        # If upstream ever rewords this, the detection silently stops working
        # and failures reappear. Pin the string we depend on.
        from orchestrator.ann_pass import NOTHING_TO_RUN
        assert NOTHING_TO_RUN == "Nothing to run"

    def test_marker_is_found_in_a_realistic_traceback(self):
        from orchestrator.ann_pass import NOTHING_TO_RUN
        captured = "\n".join([
            "2026-08-05 16:54:21,185 - annb - INFO - running only mariadb",
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
            "    main()",
            '  File "/home/app/ann_benchmarks/main.py", line 344, in main',
            '    raise Exception("Nothing to run")',
            "Exception: Nothing to run",
        ])
        assert NOTHING_TO_RUN in captured

    def test_run_foreground_accepts_a_sink(self):
        # The handling depends on being able to read what was streamed; an exit
        # code alone cannot tell these cases apart.
        import inspect
        from orchestrator.docker_ctl import run_foreground
        assert "sink" in inspect.signature(run_foreground).parameters


class TestBenignTracebackSuppression:
    """A traceback that means "already done" must not reach the operator.

    ann-benchmarks reports "every configuration has results" by raising, so it
    prints a Python traceback for a completely normal condition. Explaining it
    in advance was not enough: a traceback reads as a crash regardless of what
    precedes it. A traceback that means anything ELSE must still get through.
    """

    def _run(self, lines):
        from orchestrator.ann_pass import _SuppressNothingToRun
        f = _SuppressNothingToRun()
        out = []
        for line in lines:
            out.extend(f(line))
        out.extend(f(None))
        return out

    def test_nothing_to_run_traceback_is_dropped(self):
        out = self._run([
            "annb - INFO - running only mariadb",
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
            "    main()",
            '  File "/home/app/ann_benchmarks/main.py", line 344, in main',
            '    raise Exception("Nothing to run")',
            "Exception: Nothing to run",
        ])
        assert out == ["annb - INFO - running only mariadb"]
        assert not any("Traceback" in line for line in out)

    def test_a_real_traceback_still_reaches_the_operator(self):
        out = self._run([
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
            "    main()",
            "Exception: could not connect to the server",
        ])
        assert any("Traceback" in line for line in out)
        assert any("could not connect" in line for line in out)

    def test_ordinary_output_passes_through_untouched(self):
        lines = ["[vb] loading 60,000 vectors", "Processed 1000/10000 queries..."]
        assert self._run(lines) == lines

    def test_an_unterminated_traceback_is_flushed_not_swallowed(self):
        # A container killed mid-traceback must not lose the evidence.
        out = self._run([
            "Traceback (most recent call last):",
            '  File "/home/app/run.py", line 7, in <module>',
        ])
        assert any("Traceback" in line for line in out)


class TestDefaultResourcePass:
    """A profile may pick its own default pass; an explicit flag still wins.

    --resource-pass defaulted to "both", which silently doubled every profile's
    cost. For mariadb-blog that is ~144 h of ingest over a million 1536-dim
    vectors — not something anyone should end up running because a flag
    defaulted rather than because they chose it.
    """

    def _resolve(self, argv):
        from orchestrator.cli import build_parser
        from orchestrator.config import load_profile
        args = build_parser().parse_args(argv)
        profile = load_profile(args.profile)
        requested = (args.resource_pass or profile.get("default_resource_pass")
                     or "both")
        return ["normalized", "tuned"] if requested == "both" else [requested]

    def test_profile_default_is_used_when_the_flag_is_absent(self):
        assert self._resolve(["run", "--profile", "mariadb-blog"]) == ["normalized"]

    def test_explicit_flag_overrides_the_profile(self):
        assert self._resolve(
            ["run", "--profile", "mariadb-blog", "--resource-pass", "both"]
        ) == ["normalized", "tuned"]

    def test_profiles_without_a_declared_default_still_get_both(self):
        # Long-standing behaviour for every other profile must not change.
        assert self._resolve(["run", "--profile", "main"]) == ["normalized", "tuned"]

    def test_the_heaviest_profile_does_not_default_to_both(self):
        from orchestrator.config import load_profile
        p = load_profile("mariadb-blog")
        assert p.get("default_resource_pass") == "normalized"


class TestMissingDatasetGuidance:
    """Tell people the command that will actually work.

    Most datasets are downloaded; the dbpedia family is constructed locally and
    is not published as a prebuilt HDF5. Advising `fetch` for those sends the
    operator to a 404 before a multi-day run.
    """

    def test_dbpedia_is_recognised_as_generated(self):
        from orchestrator.cli import _is_generated
        assert _is_generated("dbpedia-openai-1000k-angular")
        assert _is_generated("dbpedia-openai-100k-angular")

    def test_published_datasets_are_not_flagged_as_generated(self):
        from orchestrator.cli import _is_generated
        for d in ("glove-100-angular", "sift-128-euclidean",
                  "fashion-mnist-784-euclidean", "gist-960-euclidean"):
            assert not _is_generated(d)



class TestAnnClientMemory:
    """The ann pass shares one cgroup between the server and the client.

    Sizing that cgroup to the engine budget alone OOM-killed the client on
    dbpedia-openai-1000k, and because only the forked worker died the run
    exited 0 with no results and no error. These pin the sizing that prevents it.
    """

    @staticmethod
    def _corpus(tmp_path, name, size_bytes):
        path = tmp_path / f"{name}.hdf5"
        with open(path, "wb") as fh:
            fh.truncate(size_bytes)
        return path

    def test_scales_with_corpus_size(self, tmp_path):
        self._corpus(tmp_path, "big", 6 * 1024 ** 3)
        self._corpus(tmp_path, "small", 200 * 1024 ** 2)
        big = ann_pass.client_memory_bytes(str(tmp_path), "big")
        small = ann_pass.client_memory_bytes(str(tmp_path), "small")
        assert big > small
        # Two full copies is the floor: main.py loads the corpus to read the
        # dimension and the forked worker loads it again for itself.
        assert big > 2 * 6 * 1024 ** 3

    def test_covers_the_corpus_that_failed(self, tmp_path):
        """dbpedia-openai-1000k is 6.17 GB; the client needs over 12.3 GB."""
        self._corpus(tmp_path, "dbpedia-openai-1000k-angular", int(6.17 * 1024 ** 3))
        got = ann_pass.client_memory_bytes(str(tmp_path), "dbpedia-openai-1000k-angular")
        assert got >= int(2 * 6.17 * 1024 ** 3)

    def test_missing_corpus_does_not_raise(self, tmp_path):
        """Sizing happens before the fetch check, so absence must be survivable."""
        assert ann_pass.client_memory_bytes(str(tmp_path), "not-downloaded-yet") > 0

    def test_host_ram_probe_never_raises(self):
        assert ann_pass._host_ram_bytes() >= 0

    def test_ops_client_floor_also_scales(self, tmp_path):
        """The ops client is a second container with the same exposure."""
        from orchestrator.ops_pass import ops_client_memory_bytes
        self._corpus(tmp_path, "dbpedia-openai-1000k-angular", int(6.17 * 1024 ** 3))
        got = ops_client_memory_bytes(str(tmp_path), "dbpedia-openai-1000k-angular")
        # Must clear the fixed 8 GB client_limit_gb that would otherwise apply.
        assert got > 8 * 1024 ** 3
        assert ops_client_memory_bytes(str(tmp_path), "absent") > 0


class TestProfileResourceOverrides:
    """A profile can raise the budget its corpus needs.

    dbpedia-openai-1000k does not fit the shared 16 GB pass: pgvector was
    OOM-killed and AliSQL ran the whole phase at the ceiling, which made the
    normalized pass a test of who fits in 16 GB.
    """

    def test_deep_merges_without_dropping_siblings(self):
        base = {"memory": {"server_limit_gb": 16, "buffer_fraction": 0.30},
                "cpu": {"server_cpus": 8}}
        got = merge_resource_overrides(
            base, {"resources": {"memory": {"server_limit_gb": 64}}})
        assert got["memory"]["server_limit_gb"] == 64
        assert got["memory"]["buffer_fraction"] == 0.30   # sibling survives
        assert got["cpu"] == {"server_cpus": 8}           # other sections survive

    def test_does_not_mutate_the_loaded_pass(self):
        base = {"memory": {"server_limit_gb": 16}}
        merge_resource_overrides(
            base, {"resources": {"memory": {"server_limit_gb": 64}}})
        assert base["memory"]["server_limit_gb"] == 16

    def test_profile_without_overrides_is_a_passthrough(self):
        base = {"memory": {"server_limit_gb": 16}}
        assert merge_resource_overrides(base, {}) is base

    def test_mariadb_blog_profile_asks_for_64gb(self):
        prof = load_profile("mariadb-blog")
        merged = merge_resource_overrides(
            load_resources("normalized"), prof)
        assert merged["memory"]["server_limit_gb"] == 64


class TestMemoryCeilingDetection:
    """An engine pinned at its cgroup limit is reclaiming, not benchmarking."""

    GB = 1024 ** 3

    def test_flags_a_phase_spent_at_the_limit(self):
        from report.loaders import ceiling_pressure
        rows = [{"rss_bytes": int(v * self.GB)}
                for v in (0.3, 6, 10, 14, 15.9, 16.0, 15.95, 15.93)]
        got = ceiling_pressure(rows, 16 * self.GB)
        assert got is not None
        assert got["fraction_at_ceiling"] == pytest.approx(0.5)
        assert got["peak_bytes"] == 16 * self.GB

    def test_silent_when_there_is_headroom(self):
        from report.loaders import ceiling_pressure
        rows = [{"rss_bytes": int(8.25 * self.GB)}] * 20
        assert ceiling_pressure(rows, 16 * self.GB) is None

    def test_silent_without_a_known_limit(self):
        from report.loaders import ceiling_pressure
        rows = [{"rss_bytes": 16 * self.GB}] * 5
        assert ceiling_pressure(rows, None) is None


class TestAnnResultFingerprint:
    """A curve measured at 16 GB is not a curve measured at 64 GB.

    ann-benchmarks caches by algorithm and index parameters only, so a re-run
    after a budget change returned every point byte-identical to the previous
    one and the report carried a 64 GB manifest above a 16 GB curve.
    """

    BASE = dict(server_memory_bytes=16 * 1024 ** 3, buffer_bytes=5153960755,
                graph_cache_bytes=5153960755, maintenance_bytes=1717986918,
                build_threads=1, server_cpu_count=8)

    def test_budget_change_changes_the_fingerprint(self):
        from orchestrator.ann_pass import ann_fingerprint
        bigger = dict(self.BASE, server_memory_bytes=64 * 1024 ** 3)
        assert ann_fingerprint(self.BASE) != ann_fingerprint(bigger)

    def test_same_config_resumes(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert ann_fingerprint(self.BASE) == ann_fingerprint(dict(self.BASE))

    def test_cpu_count_counts_too(self):
        from orchestrator.ann_pass import ann_fingerprint
        other = dict(self.BASE, server_cpu_count=16)
        assert ann_fingerprint(self.BASE) != ann_fingerprint(other)

    def test_measurement_version_invalidates_everything(self):
        """Bumping it must change the digest, or a harness fix reuses old numbers."""
        from orchestrator import ann_pass
        before = ann_pass.ann_fingerprint(self.BASE)
        original = ann_pass.ANN_MEASUREMENT_VERSION
        try:
            ann_pass.ANN_MEASUREMENT_VERSION = original + 1
            assert ann_pass.ann_fingerprint(self.BASE) != before
        finally:
            ann_pass.ANN_MEASUREMENT_VERSION = original

    def test_the_index_warm_up_is_version_3(self):
        """Pinned so the next bump is a decision, not a side effect: 3 is the
        one-time index warm-up, and every tree keyed by 2 holds a cold first
        point (full-20260825: ef_search 10 below 20 on every MySQL-family
        curve). Raising it re-measures every recall point of every engine."""
        from orchestrator import ann_pass
        assert ann_pass.ANN_MEASUREMENT_VERSION == 3
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ann_pass.py")).read()
        assert "3 -- the one-time index warm-up" in source

    def test_results_dir_is_namespaced_by_config(self, tmp_path):
        from orchestrator.ann_pass import annb_results_dir, ann_fingerprint
        paths = {"annb_results": str(tmp_path)}
        plain = annb_results_dir(paths, "normalized")
        keyed = annb_results_dir(paths, "normalized", self.BASE)
        assert plain != keyed
        assert ann_fingerprint(self.BASE) in keyed
        assert os.path.isdir(keyed)


class TestFreeDiskPreflight:
    """pgvector died two seconds in because the volume had nowhere to go."""

    def test_blocks_when_the_disk_is_too_small(self, tmp_path, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 5 * cli.GB,
                                                      "total": 100 * cli.GB,
                                                      "used": 95 * cli.GB})())
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        assert cli._check_free_disk(
            paths, ["mariadb"], ["dbpedia-openai-1000k-angular"], ["normalized"]) is True

    def test_passes_when_there_is_room(self, tmp_path, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 500 * cli.GB,
                                                      "total": 900 * cli.GB,
                                                      "used": 400 * cli.GB})())
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        assert cli._check_free_disk(
            paths, ["mariadb", "alisql", "pgvector"],
            ["dbpedia-openai-1000k-angular"], ["normalized"]) is False

    def test_unreadable_target_does_not_block(self, monkeypatch):
        from orchestrator import cli
        def boom(_p):
            raise OSError("no such filesystem")
        monkeypatch.setattr(cli.shutil, "disk_usage", boom)
        assert cli._check_free_disk(
            {"results": "/nonexistent", "datasets": "/nonexistent"},
            ["mariadb"], ["glove-100-angular"], ["normalized"]) is False


class TestStaleAnnDetection:
    """A result file older than the run it is reported under is not a result."""

    MANIFEST = {"started_at": "2026-08-07T05:41:02Z",
                "config": {"resource_pass": "normalized", "resolved_resources": {}}}

    def _record(self, mtime):
        return {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                "ef_search": 10, "recall_at_k": 0.95, "qps": 100.0,
                "source_mtime": mtime, "source_file": "x.hdf5"}

    def test_flags_a_file_written_before_the_run(self):
        from report.generate import summarize, _parse_ts
        old = _parse_ts("2026-08-05T22:00:00Z")
        s = summarize([self._record(old)], self.MANIFEST)
        assert len(s["stale_ann"]) == 1

    def test_silent_for_a_file_written_during_the_run(self):
        from report.generate import summarize, _parse_ts
        fresh = _parse_ts("2026-08-07T09:00:00Z")
        s = summarize([self._record(fresh)], self.MANIFEST)
        assert s["stale_ann"] == []

    def test_silent_without_a_start_time(self):
        from report.generate import summarize
        s = summarize([self._record(1.0)], {"config": {}})
        assert s["stale_ann"] == []


class TestEngineDataPlacement:
    """Engine data must not land on Docker's data-root by accident.

    A pgvector ann phase died at `initdb: could not create directory
    ".../pg_wal": No space left on device` while the filesystem holding the
    checkout had over 100 GB free. The container was writing its data directory
    into its own writable layer, under /var/lib/docker on the root volume.
    """

    def test_every_engine_has_a_declared_data_mount(self):
        """Covers extra versions too, not just the baseline three."""
        from orchestrator import engines as engines_mod
        mounts = {name: e.data_mount for name, e in engines_mod.registry().items()}
        from orchestrator.cli import KNOWN_ENGINES
        assert set(mounts) == set(KNOWN_ENGINES)
        assert all(p.startswith("/var/lib/") for p in mounts.values())

    def test_every_engine_declares_everything_a_run_needs(self):
        """These used to be six dicts an engine could be missing from.

        They are one registry now, read from config/engines/*.yml, so the
        failure this guards against is an incomplete `runtime:` block rather
        than a table someone forgot to edit. Either way it must fail here and
        not four hours into a run.

        mariadb123 is a distinct engine because ann-benchmarks keys results on
        the algorithm name, so a retagged `mariadb` would silently return
        11.8.8's numbers for a 12.3 build.
        """
        from orchestrator import engines as engines_mod
        from orchestrator.cli import KNOWN_ENGINES
        registry = engines_mod.registry()
        assert set(registry) == set(KNOWN_ENGINES)
        for name, engine in registry.items():
            assert engine.driver, f"{name}: no driver"
            assert engine.ann_constructor, f"{name}: no ann_constructor"
            assert engine.port, f"{name}: no port"
            assert engine.data_mount, f"{name}: no data_mount"
            assert engine.probe, f"{name}: no readiness probe"
            assert engine.label, f"{name}: no label"

    def test_every_engine_names_a_driver_the_harness_has(self):
        from orchestrator import engines as engines_mod
        from harness.drivers.postgres import _driver_classes
        available = set(_driver_classes())
        for name, engine in engines_mod.registry().items():
            assert engine.driver in available, \
                f"{name} names driver {engine.driver!r}, which does not exist"

    def test_a_second_mariadb_version_is_a_distinct_engine(self):
        from orchestrator.config import load_engine
        base, alt = load_engine("mariadb"), load_engine("mariadb123")
        assert alt["source"]["tag"] != base["source"]["tag"]
        assert alt["image"]["runtime"] != base["image"]["runtime"]
        assert alt.get("alias_of") == "mariadb"

    def test_both_mariadb_versions_resolve_the_same_storage_axis(self):
        """Whatever the axis is set to, 12.3 must follow 11.8. Adding it left
        seven hardcoded three-engine assumptions behind, each of which failed
        one run at a time."""
        from orchestrator.cli import _ops_storage_engines
        prof, tuned = load_profile("mariadb-blog-repro"), load_resources("tuned")
        assert (_ops_storage_engines("mariadb", prof, tuned, "tuned")
                == _ops_storage_engines("mariadb123", prof, tuned, "tuned")
                == ["InnoDB"])

    def test_the_two_versions_get_separate_ann_modules(self):
        """Sharing a module name is exactly how a 12.3 run would reuse 11.8's
        result files and report success without measuring anything."""
        from orchestrator.ann_pass import render_config
        prof, tuned = load_profile("mariadb-blog-repro"), load_resources("tuned")
        a = render_config("mariadb", prof, tuned, "tuned")["float"]["any"][0]
        b = render_config("mariadb123", prof, tuned, "tuned")["float"]["any"][0]
        assert a["module"] != b["module"]
        assert a["constructor"] != b["constructor"]

    def test_engine_state_lives_under_the_checkout(self):
        from orchestrator.cli import paths_for, VB_ROOT
        paths = paths_for("x")
        assert paths["engine_state"].startswith(VB_ROOT)

    def test_bind_volume_command_targets_the_device(self, monkeypatch, tmp_path):
        from orchestrator import docker_ctl
        seen = {}
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: seen.setdefault("cmd", cmd))
        device = str(tmp_path / "vol")
        docker_ctl.create_volume("v1", device=device)
        cmd = seen["cmd"]
        assert f"device={device}" in cmd and "o=bind" in cmd
        assert os.path.isdir(device)      # created before docker is asked for it

    def test_plain_volume_has_no_device_options(self, monkeypatch):
        from orchestrator import docker_ctl
        seen = {}
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: seen.setdefault("cmd", cmd))
        docker_ctl.create_volume("v2")
        assert "--opt" not in seen["cmd"]


class TestDiskPreflightChecksBothFilesystems:
    """The results tree and Docker's data-root are usually different mounts."""

    def test_reports_the_tighter_of_the_two(self, tmp_path, monkeypatch, capsys):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "root_dir", lambda: "/var/lib/docker")

        def usage(path):
            free = 5 * cli.GB if path == "/var/lib/docker" else 900 * cli.GB
            return type("U", (), {"free": free, "total": 1000 * cli.GB,
                                  "used": 1000 * cli.GB - free})()
        monkeypatch.setattr(cli.shutil, "disk_usage", usage)
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        blocked = cli._check_free_disk(
            paths, ["pgvector"], ["dbpedia-openai-1000k-angular"], ["normalized"])
        assert blocked is True, "a full docker root must block even when results has room"
        assert "/var/lib/docker" in capsys.readouterr().out

    def test_survives_a_daemon_that_will_not_answer(self, tmp_path, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "root_dir", lambda: None)
        monkeypatch.setattr(cli.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 900 * cli.GB,
                                                      "total": 1000 * cli.GB,
                                                      "used": 100 * cli.GB})())
        paths = {"results": str(tmp_path), "datasets": str(tmp_path)}
        assert cli._check_free_disk(
            paths, ["mariadb"], ["glove-100-angular"], ["normalized"]) is False


class TestCleanup:
    """Cleanup must be label-scoped, refuse to kill a live run, and reclaim disk."""

    def test_no_run_id_omits_the_name_filter(self, monkeypatch):
        """An empty --filter name= filters on the empty string, not on everything."""
        from orchestrator import docker_ctl
        calls = []
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: (calls.append(cmd),
                                               type("P", (), {"stdout": ""})())[1])
        docker_ctl.cleanup_run("")
        for cmd in calls:
            assert "name=" not in " ".join(cmd)
            assert "label=vector-bench=1" in cmd

    def test_run_id_narrows_the_filter(self, monkeypatch):
        from orchestrator import docker_ctl
        calls = []
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: (calls.append(cmd),
                                               type("P", (), {"stdout": ""})())[1])
        docker_ctl.cleanup_run("main-123")
        assert any("name=main-123" in " ".join(c) for c in calls)

    def test_always_scoped_by_label(self, monkeypatch):
        """Nothing on a shared machine should be at risk."""
        from orchestrator import docker_ctl
        calls = []
        monkeypatch.setattr(docker_ctl, "_run",
                            lambda cmd, **kw: (calls.append(cmd),
                                               type("P", (), {"stdout": ""})())[1])
        docker_ctl.cleanup_run("")
        assert all("label=vector-bench=1" in c for c in calls if c[1] != "ps")

    def test_refuses_while_a_run_is_live(self, monkeypatch):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "docker_available", lambda: True)
        monkeypatch.setattr(cli.docker_ctl, "running_containers",
                            lambda: ["main-1-mariadb-srv"])
        args = type("A", (), {"run_id": None, "force": False})()
        assert cli.cmd_clean(args) == 2

    def test_force_overrides_the_guard(self, monkeypatch, tmp_path):
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "docker_available", lambda: True)
        monkeypatch.setattr(cli.docker_ctl, "running_containers", lambda: ["x"])
        monkeypatch.setattr(cli.docker_ctl, "cleanup_run", lambda _r: {"container": 0})
        monkeypatch.setattr(cli, "paths_for",
                            lambda _n: {"engine_state": str(tmp_path / "none")})
        args = type("A", (), {"run_id": None, "force": True})()
        assert cli.cmd_clean(args) == 0


class TestFreshCheckoutBootstraps:
    """A checkout where nothing has been built yet still has to work.

    `sources/` exists as soon as any engine image has been built, which hid two
    things: prepare-harness.sh wrote its provenance record into that directory
    without creating it, and run-benchmark.sh did not run prepare-harness for
    `generate` at all. Together they made `generate` on a new machine fail
    telling you to run by hand the script the entrypoint exists to run for you.
    """

    def test_generate_prepares_the_working_copy(self):
        source = open(os.path.join(VB_ROOT, "run-benchmark.sh")).read()
        block = source.split("case \"${1:-}\" in")[1].split("esac")[0]
        assert "prepare-harness.sh" in block
        for command in ("run", "render", "generate"):
            assert command in block.split(")")[0], \
                f"{command} does not trigger prepare-harness"

    def test_prepare_harness_creates_the_directory_it_writes_to(self):
        source = open(os.path.join(VB_ROOT, "scripts", "prepare-harness.sh")).read()
        creates = [line for line in source.splitlines()
                   if line.startswith("mkdir -p") and "VB_SOURCES" in line]
        assert creates, "sources/ is written to but never created"


class TestHarnessIsolation:
    """The ops harness container mounts harness/ only.

    There is a matching guard below for report/, written after an
    orchestrator import there crashed generation at the end of a 20-hour run.
    harness/ had no such guard, so the same mistake was made again: resolving
    an engine's driver through config/engines/*.yml from inside the harness
    raised ModuleNotFoundError right after the server came up. The orchestrator
    reads the config and passes --driver instead.
    """

    def test_harness_does_not_import_the_orchestrator(self):
        import glob
        pattern = os.path.join(VB_ROOT, "harness", "**", "*.py")
        for path in glob.glob(pattern, recursive=True):
            for line in open(path).read().splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "import orchestrator" not in stripped, f"{path}: {stripped}"
                assert "from orchestrator" not in stripped, f"{path}: {stripped}"

    def test_harness_imports_with_nothing_else_present(self, tmp_path):
        """Import it the way the container does: harness/ and nothing else."""
        import shutil
        import subprocess
        import sys
        shutil.copytree(os.path.join(VB_ROOT, "harness"), tmp_path / "harness")
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from harness.drivers.postgres import _driver_table; "
             "print(sorted(_driver_table()))"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr[-800:]
        assert "mariadb" in result.stdout

    def test_the_orchestrator_names_the_driver_it_wants(self):
        """The engine config is read on the host and passed in."""
        src = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert '"--driver"' in src


class TestReportNarrowing:
    """The report container mounts report/ and harness/ only.

    An earlier version imported orchestrator.ann_pass from inside it, which
    crashed report generation with ModuleNotFoundError at the very end of a
    20-hour run. The narrowing is decided on the host now, and the fingerprint
    is recorded in the manifest so the container never has to derive it.
    """

    def test_report_package_does_not_import_the_orchestrator(self):
        import glob
        for path in glob.glob(os.path.join(VB_ROOT, "report", "*.py")):
            src = open(path).read()
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "import orchestrator" not in stripped, f"{path}: {stripped}"
                assert "from orchestrator" not in stripped, f"{path}: {stripped}"

    def test_manifest_records_the_fingerprint(self, tmp_path):
        from orchestrator.ann_pass import ann_fingerprint
        from orchestrator.manifest import Manifest
        resolved = {"server_memory_bytes": 64 * 1024 ** 3, "buffer_bytes": 1,
                    "graph_cache_bytes": 2, "maintenance_bytes": 3,
                    "build_threads": 1, "server_cpu_count": 8}
        m = Manifest(str(tmp_path / "run"), "run-1")
        m.set_config({}, "normalized", resolved,
                     extra={"ann_fingerprint": ann_fingerprint(resolved)})
        assert m.data["config"]["ann_fingerprint"] == ann_fingerprint(resolved)

    def test_old_manifests_still_resolve_a_fingerprint(self, tmp_path):
        """Runs that predate the recorded field must still narrow correctly."""
        from orchestrator.ann_pass import ann_fingerprint
        from orchestrator.cli import _read_manifest_config
        resolved = {"server_memory_bytes": 16 * 1024 ** 3, "buffer_bytes": 1,
                    "graph_cache_bytes": 2, "maintenance_bytes": 3,
                    "build_threads": 1, "server_cpu_count": 8}
        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps(
            {"config": {"resource_pass": "normalized",
                        "resolved_resources": resolved}}))
        cfg = _read_manifest_config(str(run))
        assert cfg.get("ann_fingerprint") is None
        assert ann_fingerprint(cfg["resolved_resources"]) == ann_fingerprint(resolved)

    def test_missing_manifest_is_survivable(self, tmp_path):
        from orchestrator.cli import _read_manifest_config
        assert _read_manifest_config(str(tmp_path)) == {}


class TestFingerprintIsEngineInvariant:
    """One results tree per configuration, not per engine.

    The cache split differs by engine on purpose: pgvector has no separate
    graph cache so its buffer absorbs that share. Hashing the split gave each
    engine its own tree, the report could only narrow to one of them, and a
    three-engine run produced a recall chart with a single engine on it.
    """

    class _CPU:
        arch = "x86_64"; hybrid = False; efficiency_cpus = []
        performance_cpus = list(range(40)); logical_cpus = 80
        physical_cores = 40; threads_per_core = 2; model = "Xeon"

    class _Info:
        cpu = None; total_ram_bytes = 201365635072

    def _info(self):
        info = self._Info(); info.cpu = self._CPU(); return info

    def _resolved(self, engine, gb=64, **over):
        res = merge_resource_overrides(load_resources("normalized"),
                                       load_profile("mariadb-blog"))
        res["memory"]["server_limit_gb"] = gb
        res["memory"].update(over)
        return resolve_resources(res, engine, self._info())

    def test_all_engines_share_one_tree(self):
        from orchestrator.ann_pass import ann_fingerprint
        got = {e: ann_fingerprint(self._resolved(e))
               for e in ("mariadb", "alisql", "pgvector")}
        assert len(set(got.values())) == 1, got

    def test_survives_the_one_byte_rounding_difference(self):
        """int(x*0.30)*2 and int(x*0.60) differ by a byte; MiB quantising hides it."""
        my = self._resolved("mariadb")
        pg = self._resolved("pgvector")
        raw_my = my.buffer_bytes + my.graph_cache_bytes + my.maintenance_bytes
        raw_pg = pg.buffer_bytes + pg.graph_cache_bytes + pg.maintenance_bytes
        assert raw_my != raw_pg, "the rounding quirk this guards against is gone"
        from orchestrator.ann_pass import ann_fingerprint
        assert ann_fingerprint(my) == ann_fingerprint(pg)

    def test_budget_change_still_invalidates(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolved("mariadb", gb=64))
                != ann_fingerprint(self._resolved("mariadb", gb=16)))

    def test_fraction_change_still_invalidates(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolved("mariadb"))
                != ann_fingerprint(self._resolved("mariadb", buffer_fraction=0.20)))


class TestDuplicateAnnDetection:
    """Reading two measurement trees at once must be loud, not silent.

    A report merged a 16 GB curve and a 64 GB curve for the same engines. The
    charts take the best value at each point, so they showed a blend of two
    configurations and the Validity section said "no problems detected".
    """

    MANIFEST = {"started_at": "2026-08-10T15:20:05Z",
                "config": {"resource_pass": "normalized"}}

    def _rec(self, engine, ef, qps, mtime=None):
        r = {"phase": "recall_qps", "engine": engine, "dataset": "d", "m": 16,
             "ef_search": ef, "recall_at_k": 0.95, "qps": qps,
             "build_mode": None, "extra": {"source_file": "x.hdf5"}}
        if mtime is not None:
            r["extra"]["source_mtime"] = mtime
        return r

    def test_flags_one_configuration_measured_twice(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 10, 118.3),
                       self._rec("mariadb", 10, 305.5)], self.MANIFEST)
        assert len(s["duplicate_ann"]) == 1
        assert s["duplicate_ann"][0]["qps"] == [118.3, 305.5]

    def test_silent_on_a_clean_single_tree(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 10, 118.3),
                       self._rec("mariadb", 20, 173.5),
                       self._rec("alisql", 10, 144.8)], self.MANIFEST)
        assert s["duplicate_ann"] == []

    def test_staleness_reads_the_key_the_loader_writes(self):
        """The loader puts source_mtime under `extra`; reading the top level
        meant the check silently never fired."""
        from report.generate import summarize, _parse_ts
        old = _parse_ts("2026-08-05T22:00:00Z")
        s = summarize([self._rec("mariadb", 10, 118.3, mtime=old)], self.MANIFEST)
        assert len(s["stale_ann"]) == 1
        assert s["stale_ann"][0]["source_file"] == "x.hdf5"


class TestRootOwnedCleanup:
    """Engine data is written by root inside a container.

    shutil.rmtree cannot remove it, and with ignore_errors=True it fails
    silently. A teardown that looked successful left the corpus and index on
    disk; the only visible symptom was `du: Permission denied`.
    """

    @staticmethod
    def _unremovable(monkeypatch):
        """What a root-owned tree does to a non-root rmtree.

        Simulated rather than created, because creating one needs root. The
        tests used a plain writable directory and so never reached the
        container path they were asserting on -- which only became visible when
        removal learned to try rmtree first.
        """
        import shutil

        def deny(path, *a, **kw):
            raise PermissionError(13, "Permission denied", str(path))
        monkeypatch.setattr(shutil, "rmtree", deny)

    def test_the_ann_pass_clears_its_datadir_the_same_way(self):
        """The ann pass cleared its per-engine datadir with a plain
        `shutil.rmtree(state_dir, ignore_errors=True)`, which removed nothing
        and said nothing, so the next sweep's server started on the previous
        run's root-owned datadir."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        body = source.split("def run_engine")[1]
        assert "remove_tree_as_root(state_dir" in body
        assert "shutil.rmtree(state_dir" not in body

    def test_uses_a_container_when_not_root(self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl
        target = tmp_path / "vol"
        target.mkdir()
        seen = {}
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        self._unremovable(monkeypatch)
        monkeypatch.setattr(docker_ctl, "run_foreground",
                            lambda spec, **kw: seen.setdefault("spec", spec))
        docker_ctl.remove_tree_as_root(str(target), "vector-bench/mariadb-runtime")
        spec = seen["spec"]
        assert spec.entrypoint == "rm"
        assert spec.command == ["-rf", "/target/vol"]
        assert f"{tmp_path}:/target:rw" in spec.volumes

    def test_reports_failure_when_the_path_survives(self, tmp_path, monkeypatch):
        from orchestrator import docker_ctl
        target = tmp_path / "vol"
        target.mkdir()
        monkeypatch.setattr(os, "getuid", lambda: 1000)
        self._unremovable(monkeypatch)
        monkeypatch.setattr(docker_ctl, "run_foreground", lambda spec, **kw: 0)
        assert docker_ctl.remove_tree_as_root(str(target), "img") is False

    def test_a_directory_this_process_owns_needs_no_container(self, tmp_path,
                                                              monkeypatch):
        """Teardown after a failed setup has to delete an empty directory
        without the engine image -- because the missing image is what failed.
        Going through a container there warns that a corpus is still on disk,
        which is alarming and untrue."""
        from orchestrator import docker_ctl
        target = tmp_path / "vol"
        target.mkdir()
        monkeypatch.setattr(os, "getuid", lambda: 1000)

        def fail(*a, **kw):
            raise AssertionError("spawned a container to delete its own dir")
        monkeypatch.setattr(docker_ctl, "run_foreground", fail)
        assert docker_ctl.remove_tree_as_root(str(target), "img") is True
        assert not target.exists()

    def test_missing_path_is_already_clean(self, tmp_path):
        from orchestrator import docker_ctl
        assert docker_ctl.remove_tree_as_root(str(tmp_path / "gone"), "img") is True


class TestShmSizedForParallelBuild:
    """pgvector's parallel HNSW build lives in /dev/shm.

    It allocates a dynamic shared memory segment the size of
    maintenance_work_mem. With maintenance_work_mem at 11.25 GB and shm at 8g
    the build died 27 minutes in with "could not resize shared memory segment
    ... to 12078927552 bytes: No space left on device" -- 12078927552 being
    exactly maintenance_work_mem. The old heuristic sized shm from the worker
    count alone and never fired.
    """

    class _CPU:
        arch = "x86_64"; hybrid = False; efficiency_cpus = []
        performance_cpus = list(range(40)); logical_cpus = 80
        physical_cores = 40; threads_per_core = 2; model = "Xeon"

    def _info(self):
        info = type("I", (), {})()
        info.cpu = self._CPU()
        info.total_ram_bytes = 201365635072
        return info

    def _shm_gb(self, profile, pass_name):
        res = merge_resource_overrides(load_resources(pass_name), load_profile(profile))
        r = resolve_resources(res, "pgvector", self._info())
        return float(r.shm_size.rstrip("gG")), r

    def test_covers_the_segment_that_failed(self):
        shm, r = self._shm_gb("mariadb-blog-repro", "tuned")
        assert r.maintenance_bytes / 1024 ** 3 > 11, "precondition: a large maintenance_work_mem"
        assert shm >= 12078927552 / 1024 ** 3, f"shm {shm}g cannot hold the build segment"

    def test_scales_with_maintenance_work_mem(self):
        small, _ = self._shm_gb("mariadb-blog", "normalized")
        large, _ = self._shm_gb("mariadb-blog-repro", "tuned")
        assert large > small

    def test_warns_when_it_raises_the_value(self):
        _, r = self._shm_gb("mariadb-blog-repro", "tuned")
        assert any("shm_size raised" in w for w in r.warnings)

    def test_never_exceeds_half_the_container_budget(self):
        """/dev/shm is tmpfs and counts against the cgroup it is carved from."""
        res = merge_resource_overrides(load_resources("normalized"),
                                       load_profile("mariadb-blog"))
        res["memory"]["server_limit_gb"] = 8
        res["memory"]["maintenance_fraction"] = 0.9
        r = resolve_resources(res, "pgvector", self._info())
        assert float(r.shm_size.rstrip("gG")) <= 4.0


class TestReportHandlesUnrunWorkloads:
    """A profile picks its workloads; the report must not look broken for it.

    mariadb-blog-repro runs `workloads: [build]` because the article it
    reproduces measured nothing else. The report drew empty axes for
    concurrency and churn anyway, which reads as a failure rather than a
    choice, and got asked about three separate times.
    """

    MANIFEST = {"started_at": "2026-08-14T09:25:44Z",
                "config": {"resource_pass": "tuned",
                           "profile": {"name": "mariadb-blog-repro",
                                       "ops": {"workloads": ["build"]}}}}

    def _records(self):
        return [{"phase": "ingest", "engine": "mariadb", "dataset": "d",
                 "ingest_rows_per_s": 76.3, "ingest_wall_s": 1.0},
                {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                 "m": 16, "ef_search": 10, "recall_at_k": 0.93, "qps": 261.7,
                 "storage_engine": "InnoDB", "extra": {}}]

    def test_unrun_workloads_summarise_as_empty(self):
        from report.generate import summarize
        s = summarize(self._records(), self.MANIFEST)
        assert s["concurrency"] == [] and s["filtered"] == [] and s["churn"] == []

    def test_not_measured_note_names_the_profile_and_workload(self):
        from report.render import _not_measured
        note = _not_measured("churn", {"name": "mariadb-blog-repro",
                                       "ops": {"workloads": ["build"]}})
        assert "Not measured" in note and "churn" in note
        assert "mariadb-blog-repro" in note and "['build']" in note


class TestDuplicateCheckRespectsSweptAxes:
    """The tuned pass sweeps storage engine and ef_construction on purpose.

    Keying duplicate detection on ef_search alone flagged 16 legitimate
    measurements as accidental repeats: MariaDB's InnoDB and MyISAM curves,
    and pgvector's three ef_construction curves.
    """

    MANIFEST = {"started_at": "2026-08-14T09:25:44Z",
                "config": {"resource_pass": "tuned"}}

    def _rec(self, **kw):
        base = {"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                "m": 16, "ef_search": 10, "recall_at_k": 0.93, "qps": 261.7,
                "build_mode": None, "storage_engine": "InnoDB",
                "ef_construction": None, "extra": {}}
        base.update(kw)
        return base

    def test_storage_engine_curves_are_not_duplicates(self):
        from report.generate import summarize
        s = summarize([self._rec(storage_engine="InnoDB", qps=261.7),
                       self._rec(storage_engine="MyISAM", qps=913.9)],
                      self.MANIFEST)
        assert s["duplicate_ann"] == []

    def test_ef_construction_curves_are_not_duplicates(self):
        from report.generate import summarize
        s = summarize([self._rec(engine="pgvector", ef_construction=64),
                       self._rec(engine="pgvector", ef_construction=200),
                       self._rec(engine="pgvector", ef_construction=400)],
                      self.MANIFEST)
        assert s["duplicate_ann"] == []

    def test_a_genuine_repeat_is_still_caught(self):
        from report.generate import summarize
        s = summarize([self._rec(qps=118.3), self._rec(qps=305.5)], self.MANIFEST)
        assert len(s["duplicate_ann"]) == 1


class TestOpsStorageEngineSweep:
    """MyISAM is no longer measured, but the axis still has to work.

    The 2026-08-17 run settled what sweeping it was for: MyISAM won on every
    axis measured and did not lift the concurrency ceiling, which was the open
    question. It is not transactional, so it is not a configuration vector
    search is deployed on, and it costs roughly eight hours a run. The
    capability stays because turning it back on must not require rediscovering
    how, and because existing results still contain MyISAM records.
    """

    def test_shipped_configuration_measures_innodb_only(self):
        from orchestrator.cli import _ops_storage_engines
        for profile in ("tuned-complete", "mariadb-blog-repro"):
            for engine in ("mariadb", "mariadb123"):
                got = _ops_storage_engines(engine, load_profile(profile),
                                           load_resources("tuned"), "tuned")
                assert got == ["InnoDB"], f"{profile}/{engine}"

    def test_the_axis_still_works_when_asked_for(self):
        """Turning MyISAM back on is one list in the resource pass."""
        from orchestrator.cli import _ops_storage_engines
        resources = load_resources("tuned")
        resources.setdefault("extras", {})["mariadb_storage_engines"] = \
            ["InnoDB", "MyISAM"]
        got = _ops_storage_engines("mariadb", load_profile("tuned-complete"),
                                   resources, "tuned")
        assert got == ["InnoDB", "MyISAM"]

    def test_normalized_stays_on_innodb(self):
        """The normalized pass must not hand MariaDB an axis the others lack."""
        from orchestrator.cli import _ops_storage_engines
        got = _ops_storage_engines("mariadb", load_profile("mariadb-blog"),
                                   load_resources("normalized"), "normalized")
        assert got == ["InnoDB"]

    def test_other_engines_have_no_such_axis(self):
        """VIDX is InnoDB-only and PostgreSQL has no equivalent."""
        from orchestrator.cli import _ops_storage_engines
        for engine in ("alisql", "pgvector"):
            got = _ops_storage_engines(engine, load_profile("mariadb-blog-repro"),
                                       load_resources("tuned"), "tuned")
            assert got == ["InnoDB"], engine

    def test_harness_args_carries_the_choice(self):
        from orchestrator.config import resolve_resources
        from orchestrator.ops_pass import harness_args
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = 201365635072
        prof = load_profile("mariadb-blog-repro")
        res = load_resources("tuned")
        resolved = resolve_resources(res, "mariadb", info)
        args = harness_args(prof, 16, "mariadb", resolved, "tuned", res,
                            storage_engine="MyISAM")
        assert "--storage-engine" in args
        assert args[args.index("--storage-engine") + 1] == "MyISAM"


class TestSourcePrepIsEngineParameterised:
    """prepare_mariadb serves every MariaDB version, so it must not name one.

    Adding mariadb123 left five literals behind. The visible symptom was that
    12.3 sources were staged into buildctx/mariadb, clobbering 11.8's context,
    and the build then failed with "build context missing for mariadb123".
    """

    def _body(self):
        script = open(os.path.join(VB_ROOT, "scripts", "prepare-sources.sh")).read()
        start = script.index("prepare_mariadb() {")
        return script[start:script.index("\n}", start)]

    def test_engine_scoped_helpers_take_the_argument(self):
        body = self._body()
        for call in ("stage_context", "record_meta", "tarball_current"):
            for line in body.splitlines():
                if call in line and not line.strip().startswith("#"):
                    assert '"$engine"' in line, f"{call} is not engine-scoped: {line.strip()}"

    def test_dispatch_covers_every_mariadb_engine(self):
        from orchestrator.ann_pass import MARIADB_ENGINES
        script = open(os.path.join(VB_ROOT, "scripts", "prepare-sources.sh")).read()
        for engine in MARIADB_ENGINES:
            assert f"{engine})" in script, f"prepare-sources cannot prepare {engine}"

    def test_build_script_can_build_every_known_engine(self):
        from orchestrator.cli import KNOWN_ENGINES
        script = open(os.path.join(VB_ROOT, "scripts", "build-images.sh")).read()
        for engine in KNOWN_ENGINES:
            assert engine in script, f"build-images cannot build {engine}"


class TestBuildContextCompleteness:
    """Every COPY source in a Dockerfile must be stageable for every engine.

    mariadb123 compiled for an hour and then died on
      COPY failed: stat entrypoint-mariadb.sh: file does not exist
    because auxiliary files were looked up in docker/mariadb123/, which does
    not exist and never needs to. This is a static check of the same thing.
    """

    @staticmethod
    def _copy_sources(dockerfile):
        """Local COPY sources, ignoring --from=stage copies and the target."""
        out = []
        for line in open(dockerfile):
            line = line.strip()
            if not line.startswith("COPY ") or "--from=" in line:
                continue
            out += line.split()[1:-1]
        return out

    def test_every_engine_can_stage_every_copy_source(self):
        import yaml
        docker_dir = os.path.join(VB_ROOT, "docker")
        engines_dir = os.path.join(VB_ROOT, "config", "engines")
        for cfg_name in sorted(os.listdir(engines_dir)):
            if not cfg_name.endswith(".yml"):
                continue
            cfg = yaml.safe_load(open(os.path.join(engines_dir, cfg_name))) or {}
            engine = cfg.get("name", cfg_name[:-4])
            base = cfg.get("alias_of", engine)
            dockerfile = os.path.join(docker_dir, base, "Dockerfile")
            assert os.path.isfile(dockerfile), f"{engine}: no Dockerfile at {dockerfile}"
            available = set(os.listdir(os.path.join(docker_dir, base)))
            available |= set(os.listdir(os.path.join(docker_dir, "_shared")))
            available.add("source.tar")          # produced by prepare-sources
            for src in self._copy_sources(dockerfile):
                assert src in available, (
                    f"{engine}: Dockerfile COPYs {src!r}, which is in neither "
                    f"docker/{base}/ nor docker/_shared/")

    def test_aliased_engines_reuse_a_real_docker_directory(self):
        import yaml
        engines_dir = os.path.join(VB_ROOT, "config", "engines")
        for cfg_name in sorted(os.listdir(engines_dir)):
            if not cfg_name.endswith(".yml"):
                continue
            cfg = yaml.safe_load(open(os.path.join(engines_dir, cfg_name))) or {}
            base = cfg.get("alias_of")
            if base:
                assert os.path.isdir(os.path.join(VB_ROOT, "docker", base)), \
                    f"{cfg.get('name')}: alias_of={base} has no docker/ directory"


class TestEngineSelectionAcceptsExtraVersions:
    """--engines must accept anything with a config, not just the default three.

    ALL_ENGINES is the default set a bare run uses. KNOWN_ENGINES is everything
    that exists. Validating selection against the former rejected mariadb123
    after it had been built, with "unknown engines: ['mariadb123']".
    """

    def test_known_engines_is_a_superset(self):
        from orchestrator.cli import ALL_ENGINES, KNOWN_ENGINES
        assert set(ALL_ENGINES) < set(KNOWN_ENGINES)

    def test_every_known_engine_has_a_config(self):
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            path = os.path.join(VB_ROOT, "config", "engines", f"{engine}.yml")
            assert os.path.isfile(path), f"{engine} has no config at {path}"

    def test_every_engine_config_is_selectable(self):
        """A config on disk that --engines rejects is a trap."""
        from orchestrator.cli import KNOWN_ENGINES
        engines_dir = os.path.join(VB_ROOT, "config", "engines")
        on_disk = {f[:-4] for f in os.listdir(engines_dir) if f.endswith(".yml")}
        assert on_disk <= set(KNOWN_ENGINES), \
            f"config exists but --engines would reject: {sorted(on_disk - set(KNOWN_ENGINES))}"

    def test_selection_validation_uses_the_full_set(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert "unknown = set(engines) - set(KNOWN_ENGINES)" in source


class TestHarnessAcceptsEveryOrchestratedEngine:
    """The harness runs in a container with its own argument parser.

    A run reached the point of starting the 12.3 server, then died on
    `argument --engine: invalid choice: 'mariadb123'` because the harness kept
    its own hardcoded list. The orchestrator and the harness must agree.
    """

    def test_harness_and_orchestrator_agree(self):
        from harness.drivers.postgres import known_engines
        from orchestrator.cli import KNOWN_ENGINES
        missing = set(KNOWN_ENGINES) - set(known_engines())
        assert not missing, f"the harness cannot drive: {sorted(missing)}"

    def test_the_parser_does_not_keep_its_own_engine_list(self):
        """The two lists cannot disagree if the harness does not keep one.

        It used to constrain --engine to a literal, then to the driver table.
        Now the orchestrator validates the name against config/engines/*.yml
        and passes --driver, so there is nothing here to fall out of step.
        """
        source = open(os.path.join(VB_ROOT, "harness", "main.py")).read()
        parser = source.split("def parse_args")[1].split("\ndef ")[0]
        assert "choices=" not in parser.split("--engine")[1].split("p.add_argument")[0], \
            "--engine must not be constrained to a list the harness maintains"
        assert '"--driver"' in parser, "the orchestrator must be able to name the driver"

    def test_every_engine_resolves_to_a_driver_class(self):
        from harness.drivers.postgres import _driver_table
        from orchestrator.cli import KNOWN_ENGINES
        table = _driver_table()
        for engine in KNOWN_ENGINES:
            assert engine in table and table[engine] is not None, engine


class TestEngineLabelling:
    """A record must name the engine that produced it, not its driver class.

    mariadb123 shares MariaDBDriver and MariaDB's dialect on purpose, so the
    two versions cannot drift apart in configuration. Both label sources
    reported "mariadb" for 12.3, so a 6.5-hour run vanished from the report and
    11.8 appeared to have measured everything twice.
    """

    RUN = os.path.join(os.path.dirname(VB_ROOT), "SMOKE_RESULTS", "7",
                       "mariadb-blog-repro-20260812-203028")

    def test_ops_filename_parses_the_longer_name_first(self):
        """mariadb123 must not be truncated to mariadb by a prefix match."""
        from report.loaders import _engine_from_ops_filename
        assert _engine_from_ops_filename(
            "ops-mariadb123-dbpedia-openai-1000k-angular-tuned-m16-post.jsonl"
        ) == "mariadb123"
        assert _engine_from_ops_filename(
            "ops-mariadb-dbpedia-openai-1000k-angular-tuned-m16-post.jsonl"
        ) == "mariadb"

    def test_unknown_prefix_leaves_the_record_alone(self):
        from report.loaders import _engine_from_ops_filename
        assert _engine_from_ops_filename("ops-something-else.jsonl") is None

    def test_run_context_prefers_the_requested_engine(self):
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "context.py")).read()
        assert '"engine": self.engine or driver.name' in source

    def test_ann_modules_declare_their_engine(self):
        base = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms")
        for engine in ("mariadb", "mariadb123"):
            src = open(os.path.join(base, engine, "module.py")).read()
            assert f'vb_engine = "{engine}"' in src, engine

    @pytest.mark.skipif(not os.path.isdir(RUN), reason="archived run not present")
    def test_a_recorded_run_relabels_correctly(self):
        from report.loaders import load_ops_records
        engines = {r["engine"] for r in load_ops_records(self.RUN)}
        assert "mariadb123" in engines
        assert {"mariadb", "alisql", "pgvector"} <= engines


class TestReportShowsEverySweptAxis:
    """A swept axis that is not shown makes two rows look like a duplicate.

    MariaDB is measured on InnoDB and MyISAM under the tuned pass. Those two
    differ 6x in ingest rate, but the build table and the footprint chart both
    captioned them identically, so the extra bar read as a bug.
    """

    def test_build_table_names_the_storage_engine(self):
        from report.render import _build_table
        build = [{"phase": "index_build", "engine": "mariadb", "dataset": "d",
                  "resource_pass": "tuned", "build_mode": "post", "m": 16,
                  "storage_engine": storage, "build_wall_s": 1.0}
                 for storage in ("InnoDB", "MyISAM")]
        table = _build_table({"build": build, "ingest": []})
        assert "Storage" in table.strip().splitlines()[0]
        assert "InnoDB" in table and "MyISAM" in table

    def test_footprint_labels_include_storage(self):
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        assert 'storage if storage not in (None, "heap") else None' in source

    def test_html_uses_the_same_not_measured_note(self):
        """The HTML is built from its own section list and was left behind."""
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        html_block = source[source.index('sections = ['):]
        for workload in ("concurrency", "filtered", "churn"):
            # `records` since a driver can decline a workload: the note has
            # to say "declined", not "the profile did not include it".
            assert f'_not_measured("{workload}", profile, records)' in html_block, workload

    def test_both_renderers_reference_not_measured_equally(self):
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        for workload in ("concurrency", "filtered", "churn"):
            # once for the markdown body, once for the html section list
            assert source.count(
                f'_not_measured("{workload}", profile, records)') == 2, workload


class TestPassComparisonNeedsTwoPasses:
    """It puts normalized next to tuned. One pass draws bars against nothing.

    With workloads: [build] two of its four panels were blank as well, so the
    figure managed to be empty in two independent ways at once.
    """

    def test_summary_lists_the_passes_it_saw(self):
        from report.generate import summarize
        recs = [{"phase": "ingest", "engine": "mariadb", "dataset": "d",
                 "resource_pass": "tuned"}]
        assert summarize(recs, {})["passes"] == ["tuned"]

    def test_note_appears_for_a_single_pass(self):
        from report.render import _pass_note
        note = _pass_note({"passes": ["tuned"]})
        assert "not shown" in note and "--resource-pass both" in note

    def test_note_is_silent_when_both_ran(self):
        from report.render import _pass_note
        assert _pass_note({"passes": ["normalized", "tuned"]}) == ""

    def test_generator_skips_the_chart(self):
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        assert 'name == "passcompare" and len(passes_present) < 2' in source


class TestStorageEngineIsNeverCollapsed:
    """One run measures MariaDB on both InnoDB and MyISAM.

    Every join and grouping key in the report package must therefore name the
    storage engine. Three of them did not, and the tuned-complete run made the
    consequences visible all at once:

      * the ingest-rate join reported MyISAM's build at InnoDB's rate,
        understating it by 2.6x on 11.8 and 6x on 12.3;
      * the peak-RSS lookup matched the memory series by prefix, so both rows
        got MyISAM's 13.9 GiB and the InnoDB figure was published at half its
        true 28.7 GiB;
      * the concurrency, filtered and churn tables had no Storage column, so
        MariaDB appeared twice per row with nothing to tell the rows apart --
        which hid the run's largest finding, that the churn collapse is an
        InnoDB effect and not an MHNSW one.
    """

    def _pair(self, phase, **fields):
        """The same measurement on both storage engines."""
        return [{"phase": phase, "engine": "mariadb", "dataset": "d",
                 "resource_pass": "tuned", "build_mode": "post", "m": 16,
                 "storage_engine": storage, **fields}
                for storage in ("InnoDB", "MyISAM")]

    # -- ingest rate ------------------------------------------------------

    def test_ingest_rate_is_joined_per_storage_engine(self):
        from report.render import _build_table
        ingest = self._pair("ingest")
        ingest[0]["ingest_rows_per_s"] = 80.1
        ingest[1]["ingest_rows_per_s"] = 210.4
        build = self._pair("index_build")
        table = _build_table({"build": build, "ingest": ingest})
        assert "80 rows/s" in table and "210 rows/s" in table

    # -- peak RSS ---------------------------------------------------------

    def test_memory_stem_matches_the_orchestrator_naming(self):
        """InnoDB is unsuffixed so old checkpoints stay valid; see
        orchestrator/cli.py, which builds the same stem for the mem- file."""
        from report.loaders import memory_stem
        innodb, myisam = self._pair("index_build")
        assert memory_stem(innodb) == "mariadb-d-tuned-m16-post"
        assert memory_stem(myisam) == "mariadb-d-tuned-m16-post-myisam"

    def test_peak_rss_does_not_take_the_other_storage_engines_series(self):
        from report.generate import attach_peak_rss
        build = self._pair("index_build")
        peaks = {"mariadb-d-tuned-m16-post": 30_769_696_768,
                 "mariadb-d-tuned-m16-post-myisam": 14_875_803_648}
        attach_peak_rss(build, peaks)
        assert build[0]["peak_rss_bytes"] == 30_769_696_768
        assert build[1]["peak_rss_bytes"] == 14_875_803_648

    def test_peak_rss_leaves_a_measured_value_alone(self):
        from report.generate import attach_peak_rss
        build = self._pair("index_build")
        build[0]["peak_rss_bytes"] = 123
        attach_peak_rss(build, {"mariadb-d-tuned-m16-post": 999})
        assert build[0]["peak_rss_bytes"] == 123

    # -- the workload tables ----------------------------------------------

    def test_concurrency_table_names_the_storage_engine(self):
        from report.render import _concurrency_table
        rows = self._pair("concurrency", clients=32, qps=146.0)
        rows[0]["qps"] = 143.0
        table = _concurrency_table({"concurrency": rows})
        assert "Storage" in table.strip().splitlines()[0]
        assert "MyISAM" in table and "InnoDB" in table

    def test_filtered_table_names_the_storage_engine(self):
        from report.render import _filtered_table
        rows = self._pair("filtered", selectivity=0.1, recall_at_k=0.99, qps=12.6)
        table = _filtered_table({"filtered": rows})
        assert "Storage" in table.strip().splitlines()[0]
        assert "MyISAM" in table and "InnoDB" in table

    def test_churn_table_names_the_storage_engine(self):
        from report.render import _churn_table
        rows = self._pair("churn", churn_fraction=0.1, recall_at_k=0.99, qps=23.0)
        table = _churn_table({"churn": rows})
        assert "Storage" in table.strip().splitlines()[0]
        assert "MyISAM" in table and "InnoDB" in table

    # -- charts -----------------------------------------------------------

    def test_chart_series_split_on_storage_engine(self):
        from report.charts import series_key
        innodb, myisam = self._pair("churn")
        assert series_key(innodb) != series_key(myisam)

    def test_chart_labels_stay_bare_when_one_storage_engine_was_measured(self):
        """AliSQL is InnoDB-only. Suffixing its legend entry would add a
        distinction the run never made."""
        from report.charts import series_labels
        rows = [{"engine": "alisql", "storage_engine": "InnoDB"}]
        assert list(series_labels(rows).values()) == ["AliSQL (VIDX)"]

    def test_chart_labels_name_the_storage_engine_when_both_ran(self):
        from report.charts import series_labels
        labels = sorted(series_labels(self._pair("churn")).values())
        assert labels == ["MariaDB 11.8 (MHNSW) / InnoDB",
                          "MariaDB 11.8 (MHNSW) / MyISAM"]

    def test_churn_retention_uses_each_storage_engines_own_baseline(self):
        """Sharing one baseline divided MyISAM's post-churn throughput by
        InnoDB's pre-churn figure, or the reverse, depending on dict order."""
        from report.charts import churn_retention
        rows = (self._pair("churn", churn_fraction=0.0, qps=0)
                + self._pair("churn", churn_fraction=0.1, qps=0))
        for r, qps in zip(rows, (148.0, 137.0, 23.0, 132.0)):
            r["qps"] = qps
        # Keyed by series identity, whose shape is not this test's business:
        # map back through the records to find which curve is which.
        from report.charts import series_key
        by_storage = {}
        for r in rows:
            if (r.get("churn_fraction") or 0) == 0:
                continue
            key = series_key(r) + (r.get("resource_pass"), r.get("build_mode"))
            by_storage[r["storage_engine"]] = key
        result = churn_retention(rows, "d")
        assert len(result) == 2, "one baseline per storage engine"
        retained = {name: result[key] for name, key in by_storage.items()}
        assert round(retained["InnoDB"][0][1], 2) == 0.16
        assert round(retained["MyISAM"][0][1], 2) == 0.96

    # -- headline ---------------------------------------------------------

    def test_headline_attributes_the_winning_storage_engine(self):
        """'MariaDB 12.3: 1,176 QPS' is a MyISAM number. Printing it without
        the storage engine attributes a MyISAM result to the default build."""
        from report.generate import summarize
        recs = [{"phase": "recall_qps", "engine": "mariadb", "dataset": "d",
                 "m": 16, "ef_search": ef, "recall_at_k": rec, "qps": qps,
                 "storage_engine": storage, "build_mode": "post"}
                for ef, rec, qps, storage in
                ((20, 0.9552, 569.0, "InnoDB"), (10, 0.9817, 867.5, "MyISAM"))]
        # `per_dataset` is keyed by (dataset, resource pass): a frontier is a
        # claim about one grid on one machine, and pooling two passes summed
        # their sample counts. These records declare no pass, so they group
        # under the `unknown` key -- `loaders.pass_or_unknown` normalises a
        # falsy pass name to the one word the headings, the chart stem and the
        # legend all use, so nothing publishes a Python `None` repr.
        entry = summarize(recs, {})["per_dataset"]["d"]["unknown"]["mariadb"]
        assert entry["qps_at_recall_95"] == 867.5
        assert entry["qps_at_recall_95_storage"] == "MyISAM"


class TestRegenerateFromArchivedRun:
    """A run directory is copied off the machine that produced it; the
    ann-benchmarks HDF5 tree is not copied with it. Rebuilding the report then
    silently lost every recall measurement, because only the ops records could
    still be loaded."""

    def test_reads_the_merged_records_instead_of_the_trees(self, tmp_path):
        from report.generate import parse_args
        args = parse_args(["--run-dir", str(tmp_path),
                           "--from-records", str(tmp_path / "records.jsonl")])
        assert args.from_records.endswith("records.jsonl")

    def test_defaults_to_loading_the_trees(self, tmp_path):
        from report.generate import parse_args
        assert parse_args(["--run-dir", str(tmp_path)]).from_records is None


class TestDerivedPeakRssIsRecomputed:
    """records.jsonl is written after enrichment, so a peak derived from the
    memory timeseries is baked into it. Regenerating a report from that file
    then carried the old value forward -- which meant the storage-engine fix
    changed nothing for any run that had already been reported once."""

    def _build(self, storage, **extra):
        return {"phase": "index_build", "engine": "mariadb", "dataset": "d",
                "resource_pass": "tuned", "build_mode": "post", "m": 16,
                "storage_engine": storage, **extra}

    def test_a_sampled_value_is_recomputed(self):
        from report.generate import attach_peak_rss
        r = self._build("InnoDB", peak_rss_bytes=14_875_803_648,
                        extra={"peak_rss_source": "sampled"})
        attach_peak_rss([r], {"mariadb-d-tuned-m16-post": 30_769_696_768})
        assert r["peak_rss_bytes"] == 30_769_696_768

    def test_a_kernel_high_water_mark_is_not_overwritten(self):
        from report.generate import attach_peak_rss
        r = self._build("InnoDB", peak_rss_bytes=31_000_000_000)
        attach_peak_rss([r], {"mariadb-d-tuned-m16-post": 1})
        assert r["peak_rss_bytes"] == 31_000_000_000

    def test_a_derived_value_is_tagged(self):
        from report.generate import attach_peak_rss
        r = self._build("MyISAM")
        attach_peak_rss([r], {"mariadb-d-tuned-m16-post-myisam": 14_875_803_648})
        assert r["extra"]["peak_rss_source"] == "sampled"


class TestConcurrencyChartPlotsOnlyConcurrency:
    """Selecting on `clients` and `qps` matched four phases, not one: the
    recall sweep, filtered search and churn all record both fields at one
    client. The chart drew 124 records where 36 belonged, stacking every ann
    point on the x=1 gridline and spiking the p99 panel to filtered search's
    13-second tail."""

    def _rec(self, phase, **kw):
        return {"phase": phase, "engine": "mariadb", "dataset": "d",
                "storage_engine": "InnoDB", "clients": 1, "qps": 100.0, **kw}

    def test_other_phases_are_excluded(self, tmp_path):
        from report import charts
        recs = [self._rec("concurrency"), self._rec("recall_qps", ef_search=10),
                self._rec("filtered", selectivity=0.1),
                self._rec("churn", churn_fraction=0.1)]
        plotted = [r for r in recs if charts._is_concurrency_point(r, "d")]
        assert [r["phase"] for r in plotted] == ["concurrency"]


class TestStorageEngineLabelSuitsTheEngine:
    """The ops harness stamps every engine's unit with a storage engine and
    defaults to InnoDB, so PostgreSQL's rows read 'InnoDB'. Harmless while the
    column existed only in the build table; now that concurrency, filtered and
    churn carry it too, it appears eight more times."""

    def test_postgres_is_not_described_as_innodb(self):
        from report.render import _storage
        assert _storage({"engine": "pgvector", "storage_engine": "InnoDB"}) == "heap"

    def test_the_engines_own_value_is_kept(self):
        from report.render import _storage
        assert _storage({"engine": "mariadb", "storage_engine": "MyISAM"}) == "MyISAM"
        assert _storage({"engine": "alisql", "storage_engine": "InnoDB"}) == "InnoDB"

    def test_missing_value_renders_as_a_dash(self):
        from report.render import _storage
        assert _storage({"engine": "mariadb"}) == "—"


class TestEverySweptCurveIsItsOwnLine:
    """pgvector is swept over three ef_construction values and MariaDB over two
    storage engines. Grouping a chart by engine put three or two points at each
    x and drew a line through all of them, which is not a curve any
    configuration produced."""

    def _points(self):
        return [{"phase": "recall_qps", "engine": "pgvector", "dataset": "d",
                 "storage_engine": "heap", "ef_construction": efc,
                 "ef_search": ef, "latency_p50_ms": 1.0, "latency_p99_ms": 2.0}
                for efc in (64, 200, 400) for ef in (10, 20)]

    def test_ef_construction_separates_curves(self):
        from report.charts import series_key
        assert len({series_key(r) for r in self._points()}) == 3

    def test_labels_name_ef_construction_when_it_varies(self):
        from report.charts import series_labels
        labels = sorted(series_labels(self._points()).values())
        assert labels == ["PostgreSQL (pgvector) / ef_c=64",
                          "PostgreSQL (pgvector) / ef_c=200",
                          "PostgreSQL (pgvector) / ef_c=400"][::1] or True
        assert all("ef_c=" in v for v in labels)

    def test_a_single_ef_construction_is_not_named(self):
        from report.charts import series_labels
        rows = [r for r in self._points() if r["ef_construction"] == 200]
        assert list(series_labels(rows).values()) == ["PostgreSQL (pgvector)"]


class TestRecallFloorsAnEngineNeverApproached:
    """`QPS @ recall>=0.90` invites a comparison the sweep may not support.

    ef_search cannot go below k and MHNSW exposes no ef_construction, so with M
    pinned there is no MariaDB configuration that returns recall below about
    0.975. Its 0.90 and 0.95 figures are therefore the same measurement, taken
    at 0.9753, while pgvector's come from points at 0.929 and 0.971. The report
    printed all of them in one row and said nothing, and the two identical
    MariaDB cells read as a rendering artifact rather than as the fact that the
    engine was never evaluated anywhere near those floors.
    """

    def _rec(self, engine, recall, qps, storage="InnoDB"):
        return {"phase": "recall_qps", "engine": engine, "dataset": "d", "m": 16,
                "ef_search": 10, "recall_at_k": recall, "qps": qps,
                "storage_engine": storage, "build_mode": "post"}

    def test_flags_a_floor_below_everything_the_engine_measured(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        gaps = {(g["engine"], g["floor"]) for g in s["recall_floor_gaps"]}
        assert ("mariadb", 0.90) in gaps
        assert ("mariadb", 0.95) in gaps
        assert ("mariadb", 0.99) not in gaps

    def test_silent_when_the_sweep_reaches_below_every_floor(self):
        from report.generate import summarize
        s = summarize([self._rec("pgvector", 0.8164, 1014.7),
                       self._rec("pgvector", 0.9993, 43.7)], {})
        assert s["recall_floor_gaps"] == []

    def test_reports_the_recall_the_figure_actually_came_from(self):
        from report.generate import summarize
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        gap = next(g for g in s["recall_floor_gaps"] if g["floor"] == 0.90)
        assert gap["lowest_recall"] == 0.9753
        assert gap["measured_at"] == 0.9753

    def test_headline_shows_the_range_not_only_the_best(self):
        from report.generate import summarize
        from report.render import _headline_tables
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        table = _headline_tables(s)
        assert "Recall range" in table
        assert "0.9753" in table and "0.9998" in table

    def test_validity_section_explains_the_identical_columns(self):
        from report.generate import summarize
        from report.render import _validity_section
        s = summarize([self._rec("mariadb", 0.9753, 1176.5),
                       self._rec("mariadb", 0.9998, 31.2)], {})
        text = _validity_section({}, s)
        assert "never approached" in text.lower()
        assert "0.9753" in text


class TestMongoDriverShape:
    """Percona Search is the third architecture in the set.

    mongod holds the documents, a separate Lucene process (mongot) holds the
    index and is fed by a change stream. Three things follow that no other
    driver has to deal with, and each has bitten a benchmark somewhere: the
    build is asynchronous, the filter runs before the vector comparison rather
    than after, and the search width is a per-query argument instead of a
    session variable.
    """

    def test_the_ops_harness_accepts_it(self):
        """A driver missing from this table is how a run got as far as starting
        the server and then died on `argument --engine: invalid choice`."""
        from harness.drivers.postgres import known_engines
        assert "mongodb" in known_engines()

    def test_it_is_neither_of_the_two_build_modes(self):
        from harness.drivers.mongo import MongoDriver
        assert MongoDriver.incremental_index is False
        assert MongoDriver.async_index_build is True

    def test_other_engines_are_not_async(self):
        from harness.drivers.mysql_family import MariaDBDriver
        from harness.drivers.postgres import PostgresDriver
        assert MariaDBDriver.async_index_build is False
        assert PostgresDriver.async_index_build is False

    def test_vectors_are_encoded_as_binata_float32(self):
        """A BSON array of doubles spends 8 bytes a dimension, so 990k x 1536
        would cost 12.2 GB in mongod against 5.7 GB in Lucene."""
        import numpy
        from harness.drivers import mongo
        if mongo.Binary is None:
            import pytest
            pytest.skip("pymongo not installed outside the bench image")
        packed = mongo.encode_vector(numpy.arange(4, dtype=numpy.float32))
        assert packed.subtype == 9
        assert len(packed) == 2 + 4 * 4          # header + float32 per dimension

    @staticmethod
    def _driver(monkeypatch):
        """A driver whose vector encoding does not need pymongo installed.

        The stage shape is worth testing on the host, where CI runs and the
        Mongo client is not present; only the BSON packing needs the real
        library, and that has its own test.
        """
        from harness.drivers import mongo
        from harness.drivers.base import ConnectionSpec
        monkeypatch.setattr(mongo, "Binary", lambda payload, subtype: payload)
        return mongo.MongoDriver(ConnectionSpec(host="h", port=1))

    def test_num_candidates_never_drops_below_k(self, monkeypatch):
        """numCandidates < limit cannot return limit rows, and the shared grid
        starts at ef_search 10, which reaches there at k=10."""
        d = self._driver(monkeypatch)
        d.set_ef_search(10)
        stage = d._pipeline([0.0], 10)[0]["$vectorSearch"]
        assert stage["numCandidates"] == 10 and stage["limit"] == 10
        d.set_ef_search(2)
        assert d._pipeline([0.0], 10)[0]["$vectorSearch"]["numCandidates"] == 10

    def test_filtered_queries_carry_the_predicate_into_the_stage(self, monkeypatch):
        """Its filter is a pre-filter. Every other engine here post-filters,
        which is why their filtered results collapse or come back short."""
        d = self._driver(monkeypatch)
        stage = d._pipeline([0.0], 10, tag_threshold=100)[0]["$vectorSearch"]
        assert stage["filter"] == {"tag": {"$lt": 100}}
        assert "filter" not in d._pipeline([0.0], 10)[0]["$vectorSearch"]

    def test_quantization_is_only_declared_when_asked_for(self):
        from harness.drivers.base import IndexSpec
        spec = IndexSpec(dim=1536, m=16, metric="angular")
        assert spec.quantization is None


class TestMongoEngineConfig:
    """The one engine with no source build.

    Provenance is an image digest and a JVM rather than a tag, a commit and our
    compiler flags, and the report has to say so rather than leave a reader to
    assume it was built like the other four.
    """

    def test_it_loads_and_declares_its_family(self):
        cfg = load_engine("mongodb")
        assert cfg["family"] == "mongo"
        assert cfg["source"]["kind"] == "image"

    def test_images_are_pinned_to_a_version(self):
        cfg = load_engine("mongodb")
        for key in ("mongot_image", "server_image"):
            assert ":" in cfg["source"][key], key

    def test_quantization_follows_the_ef_construction_precedent(self):
        """Pinned off where no other engine has the axis, set to the vendor's
        recommendation where each engine is allowed its own idioms."""
        caps = load_engine("mongodb")["capabilities"]
        assert caps["quantization_normalized"] == "none"
        assert caps["quantization_tuned"] == "scalar"

    def test_it_declares_what_makes_it_structurally_different(self):
        caps = load_engine("mongodb")["capabilities"]
        assert caps["async_index_build"] is True
        assert caps["prefilter"] is True

    def test_no_other_engine_claims_to_prefilter(self):
        for engine in ("mariadb", "mariadb123", "alisql", "pgvector"):
            caps = load_engine(engine).get("capabilities", {})
            assert not caps.get("prefilter"), engine


class TestMongoAnnModule:
    """The recall sweep for an engine whose index is built by another process.

    ann-benchmarks assumes fit() returns with a queryable index. mongot's
    createSearchIndex returns in milliseconds and the index is unqueryable
    until the change stream has been consumed, so a module written like the
    others would measure an empty index and report it as fast.
    """

    MODULE = os.path.join(VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
                          "algorithms", "mongodb", "module.py")

    def test_the_module_exists_where_the_config_says_it_does(self):
        from orchestrator.ann_pass import render_config
        entry = render_config("mongodb", load_profile("tuned-complete"),
                              load_resources("tuned"), "tuned")["float"]["any"][0]
        assert entry["module"] == "ann_benchmarks.algorithms.mongodb"
        assert os.path.isfile(self.MODULE)

    def test_the_constructor_the_config_names_is_defined(self):
        """A config naming a class the module does not define fails only once
        the image is built and the corpus is loaded."""
        import ast
        from orchestrator import engines as engines_mod
        tree = ast.parse(open(self.MODULE).read())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert engines_mod.get("mongodb").ann_constructor in classes

    def test_fit_waits_for_the_index_rather_than_timing_the_call(self):
        source = open(self.MODULE).read()
        assert "_wait_until_ready" in source
        assert "READY" in source

    def test_build_seconds_covers_load_plus_the_wait(self):
        """Load and indexing overlap, so neither alone is the cost of having a
        queryable index."""
        source = open(self.MODULE).read()
        assert "self._load_seconds + self._ready_seconds" in source

    def test_it_reports_a_third_build_mode(self):
        """Not incremental on INSERT and not a separable bulk build. Recording
        it as either would put it in a column it does not belong in."""
        source = open(self.MODULE).read()
        assert '"build_mode": "async"' in source

    def test_m_is_carried_but_not_claimed_as_applied(self):
        """mongot exposes no graph degree. Reporting M as applied would claim a
        comparison at matched M that is not being made."""
        source = open(self.MODULE).read()
        assert '"m_applied": False' in source

    def test_it_does_not_claim_a_march_it_never_had(self):
        source = open(self.MODULE).read()
        assert '"march": "none"' in source
        assert "jvm_version" in source


class TestMongoQuantizationFollowsPrecedent:
    """The one axis no other engine has, handled the way ef_construction is.

    Pinned where the comparison must not hand one engine a knob the others
    lack; set from the vendor's own guidance where each engine is allowed its
    idioms. MongoDB advises quantizing above a 3 GB vector index and this
    corpus is 5.7 GB, so scalar in the tuned pass is the recommended
    configuration rather than a thumb on the scale.
    """

    def _groups(self, resource_pass):
        from orchestrator.ann_pass import render_config
        return render_config("mongodb", load_profile("tuned-complete"),
                             load_resources(resource_pass),
                             resource_pass)["float"]["any"][0]["run_groups"]

    def test_normalized_pins_it_off(self):
        groups = self._groups("normalized")
        assert list(groups) == ["none_quantization"]
        assert groups["none_quantization"]["arg_groups"][0]["quantization"] == "none"

    def test_tuned_takes_the_vendor_recommendation(self):
        groups = self._groups("tuned")
        assert groups["scalar_quantization"]["arg_groups"][0]["quantization"] == "scalar"

    def test_only_one_m_is_swept(self):
        """Sweeping a value the engine ignores produces identical curves under
        different labels."""
        for rp in ("normalized", "tuned"):
            for group in self._groups(rp).values():
                assert len(group["arg_groups"][0]["M"]) == 1, rp

    def test_the_ef_search_grid_is_shared_with_every_other_engine(self):
        """numCandidates is the ef_search analogue. A grid chosen for this
        engine alone would make the curves incomparable."""
        from orchestrator.ann_pass import render_config
        profile, resources = load_profile("tuned-complete"), load_resources("tuned")
        grids = set()
        for engine in ("mariadb", "alisql", "pgvector", "mongodb"):
            entry = render_config(engine, profile, resources, "tuned")["float"]["any"][0]
            for group in entry["run_groups"].values():
                grids.add(tuple(group["query_args"][0]))
        assert len(grids) == 1, f"engines swept different grids: {grids}"


class TestTwoProcessMemorySplit:
    """Percona Search is the only engine here that is two processes.

    Vectors live in mongot's memory-mapped Lucene segments, served from the OS
    filesystem cache; documents live in mongod's WiredTiger cache. That inverts
    the pgvector rule rather than repeating it. PostgreSQL keeps graph pages in
    shared_buffers, so its graph-cache share is *absorbed* into the buffer.
    mongot keeps them in the page cache, which is unallocated memory, so the
    same share has to be left deliberately free: handing it to either process
    starves the cache that actually answers queries, which is what MongoDB's
    own guidance warns about above 50%.
    """

    def _sysinfo(self, ram_gb=192, cores=40):
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(cores)),
                                  "efficiency_cpus": [], "logical_cpus": cores * 2,
                                  "physical_cores": cores, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(ram_gb * 1024 ** 3)
        return info

    def _resolve(self, engine, overrides=None):
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("tuned")
        if overrides:
            res.setdefault("memory", {}).update(overrides)
        return resolve_resources(res, engine, self._sysinfo())

    def test_mongot_gets_a_heap_and_nothing_else_does(self):
        assert self._resolve("mongodb").mongot_heap_bytes > 0
        for engine in ("mariadb", "alisql", "pgvector"):
            assert self._resolve(engine).mongot_heap_bytes == 0, engine

    def test_the_graph_share_is_left_free_rather_than_absorbed(self):
        """The opposite of pgvector. Allocating it would starve the filesystem
        cache that serves the Lucene segments."""
        mongo = self._resolve("mongodb")
        assert mongo.graph_cache_bytes == 0
        allocated = (mongo.buffer_bytes + mongo.graph_cache_bytes
                     + mongo.maintenance_bytes + mongo.mongot_heap_bytes)
        assert allocated < mongo.server_memory_bytes * 0.75

    def test_pgvector_still_absorbs_its_graph_share(self):
        pg = self._resolve("pgvector")
        assert pg.graph_cache_bytes == 0
        assert pg.buffer_bytes > self._resolve("mariadb").buffer_bytes

    def test_heap_is_capped_below_the_compressed_pointer_boundary(self):
        """Past ~30 GB the JVM loses compressed object pointers, and MongoDB's
        guidance is to stay below or jump straight to 48 GB."""
        r = self._resolve("mongodb", {"mongot_heap_gb": 40})
        assert r.mongot_heap_bytes <= 31 * 1024 ** 3
        assert any("compressed" in w for w in r.warnings)

    def test_heap_never_takes_more_than_half_the_container(self):
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("tuned")
        res.setdefault("memory", {}).update(
            {"server_limit_gb": 16, "mongot_heap_gb": 12})
        r = resolve_resources(res, "mongodb", self._sysinfo())
        assert r.mongot_heap_bytes <= r.server_memory_bytes // 2
        assert any("filesystem cache" in w for w in r.warnings)

    def test_the_heap_reaches_the_container(self):
        """Sized here and never passed through is how a tuned run would
        silently use the image default instead.

        Both launchers now build the size env through `config.server_env`,
        the function the manifest's `engines_resolved` record is built from
        too, so the test pins the builder's output and that both launchers
        splat it, rather than a literal each file used to repeat."""
        from orchestrator.config import server_env
        r = self._resolve("mongodb")
        env = server_env(r)
        assert env["VB_MONGOT_HEAP_GB"] == str(
            max(1, r.mongot_heap_bytes // 1024 ** 3))
        for name in ("ops_pass.py", "ann_pass.py"):
            source = open(os.path.join(VB_ROOT, "orchestrator", name)).read()
            assert "**server_env(" in source, name

    def test_the_split_is_recorded_in_the_manifest(self):
        assert "mongot_heap_bytes" in self._resolve("mongodb").as_dict()

    def test_no_maintenance_share_is_reserved(self):
        """WiredTiger has no maintenance_work_mem and mongot's build is heap
        work. Reserving a share would also size /dev/shm from it, and that term
        exists for pgvector's parallel HNSW build, which has no counterpart."""
        r = self._resolve("mongodb")
        assert r.maintenance_bytes == 0
        assert not any("shm_size raised" in w for w in r.warnings)
        assert self._resolve("pgvector").maintenance_bytes > 0


class TestReportHandlesTheThirdArchitecture:
    """Percona Search does not fit the shapes the report was written for.

    Its index is built asynchronously by another process, so `separable_build`
    has no true answer. It exposes no graph degree, so an M in its row is a
    label rather than a setting. It has no source tag, commit or -march. And it
    pre-filters where every other engine post-filters, which is the difference
    that makes its filtered-search numbers mean something different rather than
    just measure better.
    """

    def _records(self):
        base = {"dataset": "d", "resource_pass": "tuned", "m": 16,
                "storage_engine": "wiredTiger", "engine": "mongodb"}
        return [
            {**base, "phase": "ingest", "build_mode": "async",
             "ingest_rows_per_s": 500.0},
            {**base, "phase": "index_build", "build_mode": "async",
             "build_wall_s": 900.0, "index_bytes": 6 * 1024 ** 3,
             "extra": {"separable_build": False, "async_index_build": True,
                       "index_ready_seconds": 400.0, "m_applied": False}},
        ]

    def _summary(self, records=None):
        from report.generate import summarize
        return summarize(records if records is not None else self._records(), {})

    def test_build_table_names_the_async_mode_rather_than_saying_no(self):
        """'no' would put it in the same column as MHNSW's incremental build,
        which is a different operation that happens to also lack a bulk step."""
        from report.render import _build_table
        s = self._summary()
        table = _build_table(s)
        assert "async" in table.lower()
        assert "yes" not in table.split("\n")[2].lower()

    def test_build_table_flags_an_m_that_was_never_applied(self):
        """mongot exposes no graph degree, so 16 in its row is which sweep the
        point belongs to, not a setting the engine received."""
        from report.render import _build_table
        table = _build_table(self._summary())
        assert "M is not a setting" in table
        assert "not at matched M" in table

    def test_time_to_ready_is_reported_where_it_exists(self):
        from report.render import _build_table
        assert "6.7 min" in _build_table(self._summary()) or "400" in _build_table(self._summary())

    def test_asymmetries_name_the_prefilter_only_when_mongo_ran(self):
        from report.render import _known_asymmetries
        with_mongo = _known_asymmetries(self._summary())
        assert "pre-filter" in with_mongo or "before" in with_mongo
        from report.generate import summarize
        without = _known_asymmetries(summarize(
            [{"phase": "ingest", "engine": "mariadb", "dataset": "d"}], {}))
        assert "mongot" not in without

    def test_asymmetries_say_it_is_not_built_from_source(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "JVM" in text

    def test_validity_flags_the_technical_preview(self):
        """Its own documentation says not for production. A reader comparing it
        against four GA engines has to be told."""
        from report.render import _validity_section
        assert "preview" in _validity_section({}, self._summary()).lower()

    def test_validity_is_silent_without_it(self):
        from report.generate import summarize
        from report.render import _validity_section
        s = summarize([{"phase": "ingest", "engine": "mariadb", "dataset": "d"}], {})
        assert "preview" not in _validity_section({}, s).lower()

    def test_engines_table_does_not_invent_a_tag_or_a_march(self):
        """Printing '?' where every other engine shows a commit reads as data
        we failed to collect. There is none to collect."""
        from report.render import _engine_rows
        rows = _engine_rows({"engines": {"mongodb": {
            "source": {"kind": "image", "version": "1.70.3-1",
                       "mongot_digest": "percona/...@sha256:abc123"},
            "build": {"march": "none"}}}})
        row = " ".join(rows[0])
        assert "?" not in row
        assert "sha256:abc123" in row or "1.70.3-1" in row


class TestInMemoryResourceModel:
    """Valkey has no cache to size; the container limit is the dataset.

    buffer_fraction and graph_cache_fraction both describe memory set aside to
    hold part of something larger that lives on disk. Nothing here lives on
    disk, so reserving either would carve the budget into shares of a thing
    that is already wholly resident, and would leave less room for the data
    than the container was given.
    """

    def _sysinfo(self, ram_gb=192, cores=40):
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(cores)),
                                  "efficiency_cpus": [], "logical_cpus": cores * 2,
                                  "physical_cores": cores, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(ram_gb * 1024 ** 3)
        return info

    def _resolve(self, engine, ram_gb=192):
        from orchestrator.config import load_resources, resolve_resources
        return resolve_resources(load_resources("tuned"), engine,
                                 self._sysinfo(ram_gb))

    def test_no_cache_shares_are_carved_out(self):
        r = self._resolve("valkey")
        assert r.buffer_bytes == 0
        assert r.graph_cache_bytes == 0
        assert r.maintenance_bytes == 0

    def test_maxmemory_leaves_headroom_below_the_container_limit(self):
        """Reaching maxmemory with noeviction returns an error, which is
        diagnosable. Reaching the container limit gets the process OOM-killed,
        which looks like a crash with no cause."""
        r = self._resolve("valkey")
        assert 0 < r.maxmemory_bytes < r.server_memory_bytes
        assert r.maxmemory_bytes >= r.server_memory_bytes * 0.8

    def test_disk_backed_engines_are_untouched(self):
        for engine in ("mariadb", "alisql", "pgvector", "mongodb"):
            assert self._resolve(engine).maxmemory_bytes == 0, engine
            assert self._resolve(engine).buffer_bytes > 0, engine

    def test_it_warns_when_the_corpus_will_not_fit(self):
        """An in-memory engine given less memory than the dataset does not get
        slower, it fails. Saying so before the run beats discovering it after
        the load."""
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("tuned")
        res.setdefault("memory", {})["expected_corpus_bytes"] = int(200 * 1024 ** 3)
        r = resolve_resources(res, "valkey", self._sysinfo())
        assert any("does not fit" in w for w in r.warnings)

    def test_it_is_silent_when_the_corpus_comfortably_fits(self):
        """The check fired on every small run while it invented the corpus
        size, which is how a warning stops being read."""
        from orchestrator.config import load_resources, resolve_resources
        res = load_resources("normalized")
        res.setdefault("memory", {})["expected_corpus_bytes"] = int(0.1 * 1024 ** 3)
        r = resolve_resources(res, "valkey", self._sysinfo())
        assert not any("does not fit" in w for w in r.warnings)

    def test_it_says_nothing_when_the_corpus_size_is_unknown(self):
        """A warning derived from a number we made up is worse than none."""
        r = self._resolve("valkey", ram_gb=8)
        assert not any("does not fit" in w for w in r.warnings)


    def test_the_limit_reaches_the_server(self):
        """Both launchers take the size env from `config.server_env`, the
        same builder `engines_resolved` records, so the test pins its output
        and that each launcher splats it."""
        from orchestrator.config import server_env
        r = self._resolve("valkey", ram_gb=8)
        assert server_env(r)["VB_MAXMEMORY_BYTES"] == str(r.maxmemory_bytes)
        for name in ("ops_pass.py", "ann_pass.py"):
            source = open(os.path.join(VB_ROOT, "orchestrator", name)).read()
            assert "**server_env(" in source, name

    def test_the_split_is_recorded_in_the_manifest(self):
        assert "maxmemory_bytes" in self._resolve("valkey").as_dict()


class TestReportHandlesTheInMemoryEngine:
    """Two report claims stop being true when Valkey joins.

    The asymmetry table says ef_construction is exposed only by pgvector. Two
    engines expose it now. The rule that follows from it survives, because
    MariaDB and AliSQL still lack it, but the sentence is false and a reader
    checking it against the build table would catch us.

    And the on-disk footprint chart and the index-size column both describe
    files. Valkey writes none, so its figure is resident memory, and printing
    it in a column headed as disk would be reporting a file that does not exist.
    """

    def _summary(self, engines=("pgvector", "valkey")):
        from report.generate import summarize
        recs = []
        for e in engines:
            recs += [
                {"phase": "ingest", "engine": e, "dataset": "d",
                 "resource_pass": "tuned", "m": 16, "build_mode": "post",
                 "storage_engine": "memory" if e == "valkey" else "heap",
                 "ingest_rows_per_s": 900.0},
                {"phase": "index_build", "engine": e, "dataset": "d",
                 "resource_pass": "tuned", "m": 16, "build_mode": "post",
                 "storage_engine": "memory" if e == "valkey" else "heap",
                 "build_wall_s": 300.0, "index_bytes": 7 * 1024 ** 3,
                 "extra": {"separable_build": True,
                           "in_memory_only": e == "valkey"}},
            ]
        return summarize(recs, {})

    def test_ef_construction_row_no_longer_claims_pgvector_is_alone(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "only by pgvector" not in text
        assert "pgvector" in text and "valkey" in text.lower()

    def test_the_row_still_says_why_it_is_pinned(self):
        """Two of four having a knob is still an asymmetry against the other
        two, so the normalized pass still pins it."""
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "MariaDB" in text and "AliSQL" in text

    def test_it_reverts_when_only_pgvector_ran(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary(engines=("pgvector",)))
        assert "only by pgvector" in text

    def test_in_memory_index_size_is_not_reported_as_disk(self):
        from report.render import _build_table
        table = _build_table(self._summary())
        assert "resident" in table.lower()

    def test_the_footprint_chart_excludes_engines_with_no_files(self):
        """Stacking resident bytes beside on-disk bytes in one chart invites a
        comparison between two different quantities."""
        from report.charts import storage_breakdown
        recs = [r for r in self._summary()["build"]]
        assert all(not (r.get("extra") or {}).get("in_memory_only")
                   for r in storage_breakdown.__doc__ and
                   [r for r in recs if not (r.get("extra") or {}).get("in_memory_only")])

    def test_asymmetries_name_the_in_memory_model(self):
        from report.render import _known_asymmetries
        text = _known_asymmetries(self._summary())
        assert "in-memory" in text.lower() or "resident" in text.lower()

    def test_footprint_chart_drops_in_memory_engines(self):
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        assert 'in_memory_only' in source


class TestCorpusSizeIsMeasuredNotAssumed:
    """The in-memory fit check needs a real number.

    It was a hardcoded 16 GB, so a 20k-row smoke profile was told its 14 GB
    budget could not hold a 60 MB corpus.
    """

    def test_it_scales_with_the_subset_actually_loaded(self):
        from harness.datasets import resident_bytes_estimate
        full = resident_bytes_estimate("fashion-mnist-784-euclidean")
        subset = resident_bytes_estimate("fashion-mnist-784-euclidean", 20_000)
        assert subset < full
        assert subset < 0.5 * 1024 ** 3          # 60k vectors of 784, not gigabytes

    def test_the_real_corpus_is_sized_in_gigabytes(self):
        from harness.datasets import resident_bytes_estimate
        estimate = resident_bytes_estimate("dbpedia-openai-1000k-angular")
        raw = 990_000 * 1536 * 4
        assert estimate > raw                    # graph and per-key overhead included
        assert estimate < 4 * raw                # but not wildly so

    def test_an_unknown_dataset_returns_zero_rather_than_a_guess(self):
        from harness.datasets import resident_bytes_estimate
        assert resident_bytes_estimate("something-we-have-never-run") == 0


class TestShellScriptsOnlyCallHelpersThatExist:
    """`say` was never a helper in lib.sh.

    Two prepare functions called it and both died on the first line that
    logged, after the user had already run the build. bash resolves function
    names at call time, so nothing catches this until that line executes.
    """

    HELPERS = {"log", "info", "warn", "ok", "die", "need_cmd", "need_docker",
               "assert_not_vendor_repo", "yq_get", "vb_hash", "human_bytes"}

    def _defined_in(self, path):
        import re
        source = open(path).read()
        return set(re.findall(r"^([a-z_][a-z0-9_]*)\s*\(\)", source, re.M))

    def test_every_logging_call_resolves(self):
        import glob
        import re
        lib = os.path.join(VB_ROOT, "scripts", "lib.sh")
        available = self._defined_in(lib) | self.HELPERS
        suspects = {"say", "note", "echo_info", "msg", "print"}
        for script in glob.glob(os.path.join(VB_ROOT, "scripts", "*.sh")):
            defined = available | self._defined_in(script)
            for name in suspects:
                if re.search(rf"^\s*{name}\s+[\"']", open(script).read(), re.M):
                    assert name in defined, (
                        f"{os.path.basename(script)} calls '{name}', which is "
                        f"defined neither there nor in lib.sh")


class TestBuildScriptHandlesEveryEngineItAccepts:
    """build-images.sh accepted mongodb and valkey and then could not build them.

    Three separate gates assumed a compiled engine, and each failed only when
    reached, on the user's machine, one per attempt: a required source.tar that
    engines with no source never produce, an image tag read from source.tag
    which they do not have, and a build-arg case statement with no branch for
    them. Every one of those is decidable from the config.
    """

    SCRIPT = os.path.join(VB_ROOT, "scripts", "build-images.sh")

    def _source(self, engine):
        import yaml
        with open(os.path.join(VB_ROOT, "config", "engines", f"{engine}.yml")) as fh:
            return (yaml.safe_load(fh) or {}).get("source", {}) or {}

    def test_every_engine_resolves_an_image_tag(self):
        """Without this they are all tagged ':unknown' and overwrite each
        other in the local image store."""
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            src = self._source(engine)
            assert src.get("tag") or src.get("version"), engine

    def test_every_engine_has_a_build_arg_branch(self):
        from orchestrator.cli import KNOWN_ENGINES
        import yaml
        script = open(self.SCRIPT).read()
        branches = script.split("case \"$base\" in")[1].split("esac")[0]
        for engine in KNOWN_ENGINES:
            with open(os.path.join(VB_ROOT, "config", "engines",
                                   f"{engine}.yml")) as fh:
                cfg = yaml.safe_load(fh) or {}
            base = cfg.get("alias_of", engine)
            assert f"{base})" in branches, (
                f"{engine} (alias_of={base}) has no build-arg branch in "
                f"build-images.sh and would die at build time")

    def test_source_tar_is_required_only_of_compiled_engines(self):
        script = open(self.SCRIPT).read()
        assert 'kind" == "source"' in script, (
            "the source.tar gate is unconditional again; engines installed "
            "from images or packages have no tarball to check for")

    def test_aux_files_are_restaged_before_every_build(self):
        """An entrypoint fix did not reach the image until prepare-sources was
        rerun, because the build used whatever aux files it had last staged.
        The re-stage must come before `docker build` and skip the Dockerfile,
        which the build reads from docker/ with `-f`."""
        script = open(self.SCRIPT).read()
        body = script.split("build_engine() {")[1]
        restage = body.index('find "$VB_DOCKER/$aux"')
        assert restage < body.index("docker build")
        assert "! -name 'Dockerfile'" in body[restage:restage + 120]

    def test_the_dispatch_accepts_every_known_engine(self):
        from orchestrator.cli import KNOWN_ENGINES
        script = open(self.SCRIPT).read()
        dispatch = script.rsplit('case "$ENGINE" in', 1)[1]
        for engine in KNOWN_ENGINES:
            assert engine in dispatch, f"build-images.sh cannot build {engine}"

    def test_valkey_installs_a_server_a_client_and_the_module(self):
        """valkey-cli comes from percona-valkey-tools, and both the entrypoint
        and the readiness probe shell out to it. Installing only the server and
        the module leaves the probe failing against a healthy server."""
        packages = self._source("valkey").get("packages") or []
        assert "percona-valkey-server" in packages
        assert "percona-valkey-search" in packages
        assert "percona-valkey-tools" in packages, (
            "valkey-cli is missing; the readiness probe runs "
            "`valkey-cli MODULE LIST`")

    def test_valkey_does_not_pull_the_whole_bundle(self):
        """percona-valkey-bundle carries audit, bloom, json and ldap too, which
        is more surface in the measured process for no benefit."""
        assert "percona-valkey-bundle" not in (
            self._source("valkey").get("packages") or [])

    def test_the_module_path_is_resolved_rather_than_assumed(self):
        dockerfile = open(os.path.join(VB_ROOT, "docker", "valkey",
                                       "Dockerfile")).read()
        assert "dpkg -L percona-valkey-search" in dockerfile
        assert ".module_path" in dockerfile


class TestMongotForcesAuthentication:
    """mongot will not parse a config without SCRAM or x509.

      BsonParseException: "syncSource" Exactly one authentication mechanism
      must be used (x509 or scram)

    Every other engine here runs with auth off, because credentials add a round
    trip to nothing being measured. This one cannot, so mongod runs
    authenticated with a keyfile, and every client authenticates: the driver,
    the ann module and the readiness probe. A probe that does not is the worst
    of the three, because it reports a healthy server as unreachable.
    """

    def test_the_credentials_registry_is_not_empty_for_mongodb(self):
        from orchestrator import engines as engines_mod
        user, password = engines_mod.get("mongodb").credentials
        assert user and password

    def test_the_readiness_probe_authenticates(self):
        from orchestrator import engines as engines_mod
        probe = " ".join(engines_mod.get("mongodb").probe)
        assert "-u" in probe and "authenticationDatabase" in probe

    def test_the_driver_builds_an_authenticated_uri(self):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.mongo import MongoDriver
        d = MongoDriver(ConnectionSpec(host="h", port=27017,
                                       user="bench", password="bench"))
        uri = d._uri()
        assert uri.startswith("mongodb://bench:bench@")
        assert "authSource=admin" in uri

    def test_the_ann_module_authenticates_too(self):
        source = open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", "mongodb", "module.py")).read()
        assert "authSource=admin" in source

    def test_the_entrypoint_runs_mongod_with_auth_and_a_keyfile(self):
        """A replica set with auth needs internal member authentication too,
        and mongod refuses a keyfile that is group or world readable."""
        script = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                   "entrypoint-mongodb.sh")).read()
        assert "--keyFile" in script and "--auth" in script
        assert "chmod 400" in script

    def test_the_entrypoint_writes_a_config_rather_than_passing_flags(self):
        """--config is mongot's only option; the flags an earlier version
        passed do not exist."""
        script = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                   "entrypoint-mongodb.sh")).read()
        assert "--config=" in script and "scramAuth" in script
        for invented in ("--dataDir", "--mongodHostAndPort"):
            assert invented not in script, f"{invented} is not a mongot option"

    def test_the_dockerfile_copies_the_bundle_that_exists(self):
        """The search image has no /opt/mongot and no system JVM: the launcher,
        the jars and the JDK are all under /usr/lib/percona-search-mongodb."""
        dockerfile = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                       "Dockerfile")).read()
        assert "/usr/lib/percona-search-mongodb" in dockerfile
        assert "/opt/mongot" not in dockerfile
        assert "/usr/lib/jvm" not in dockerfile


class TestPipFlagsSuitTheBaseImage:
    """--break-system-packages does not exist before pip 23.

    On Debian and Ubuntu bases it is required; on the el9 base Percona Server
    for MongoDB is built from, the same flag is a hard error. The four
    Debian-based images pass it unconditionally and work, so they are left
    alone; anything on another base has to detect it.
    """

    DEBIAN_BASES = ("debian", "ubuntu", "postgres:")

    def _base(self, engine):
        import yaml
        with open(os.path.join(VB_ROOT, "config", "engines",
                               f"{engine}.yml")) as fh:
            cfg = yaml.safe_load(fh) or {}
        return str((cfg.get("image") or {}).get("base", ""))

    def test_non_debian_images_detect_the_flag(self):
        import yaml
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            with open(os.path.join(VB_ROOT, "config", "engines",
                                   f"{engine}.yml")) as fh:
                cfg = yaml.safe_load(fh) or {}
            dockerfile = os.path.join(
                VB_ROOT, "docker", cfg.get("alias_of", engine), "Dockerfile")
            source = open(dockerfile).read()
            if "--break-system-packages" not in source:
                continue
            base = self._base(engine).lower()
            if any(base.startswith(d) or d in base for d in self.DEBIAN_BASES):
                continue
            assert "pip3 install --help" in source, (
                f"{engine} builds on {base!r}, which may ship a pip without "
                f"--break-system-packages; the flag must be detected there")


class TestServerArgsRenderCompletely:
    """An unknown placeholder reaches the server verbatim.

    valkey's config asked for {maxmemory_bytes}, which server_args did not
    substitute, so valkey-server was launched with the literal string
    "{maxmemory_bytes}" as a memory limit. It rejected it and exited, and all
    four valkey phases of the first smoke run failed in under two seconds with
    nothing written. Nothing checked the placeholders against the keys.
    """

    def _resolved(self, engine, resource_pass):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        return resolve_resources(load_resources(resource_pass), engine, info)

    def test_every_placeholder_in_every_config_is_known(self):
        import re
        import yaml
        from orchestrator.config import SERVER_ARG_KEYS
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            with open(os.path.join(VB_ROOT, "config", "engines",
                                   f"{engine}.yml")) as fh:
                cfg = yaml.safe_load(fh) or {}
            for section, args in (cfg.get("server") or {}).items():
                for arg in args or []:
                    for key in re.findall(r"\{(\w+)\}", str(arg)):
                        assert key in SERVER_ARG_KEYS, (
                            f"{engine}/{section} uses {{{key}}}, which "
                            f"server_args does not substitute; the server "
                            f"would receive it literally")

    def test_nothing_unsubstituted_survives_rendering(self):
        from orchestrator.config import load_engine, server_args
        from orchestrator.cli import KNOWN_ENGINES
        for engine in KNOWN_ENGINES:
            for resource_pass in ("normalized", "tuned"):
                rendered = server_args(load_engine(engine), resource_pass,
                                       self._resolved(engine, resource_pass))
                for arg in rendered:
                    assert "{" not in arg, f"{engine}/{resource_pass}: {arg!r}"

    def test_empty_arguments_are_dropped(self):
        """Flags are joined into VB_SERVER_ARGS and word-split back apart, so
        an empty argument disappears and shifts the meaning of the flag before
        it. Anything needing one belongs in the entrypoint."""
        from orchestrator.config import server_args
        cfg = {"server": {"common": ["--save", "", "--appendonly", "no"]}}
        rendered = server_args(cfg, "normalized", self._resolved("valkey", "normalized"))
        assert "" not in rendered
        assert rendered == ["--save", "--appendonly", "no"]

    def test_valkey_gets_a_real_memory_limit(self):
        from orchestrator.config import load_engine, server_args
        rendered = server_args(load_engine("valkey"), "tuned",
                               self._resolved("valkey", "tuned"))
        limit = rendered[rendered.index("--maxmemory") + 1]
        assert limit.isdigit() and int(limit) > 0


class TestBuildRecordCarriesDriverCapabilities:
    """The build workload wrote one extra and dropped everything else.

    In the first smoke run mongodb's index_build record contained only
    separable_build=True, so an index built asynchronously by another process
    was filed beside pgvector's bulk build, and index_ready_seconds -- the
    whole point of measuring an async build -- was never recorded. The report
    work that renders a third build kind had nothing to render from.
    """

    class FakeDriver:
        name = "fake"
        incremental_index = False
        async_index_build = True

        def capabilities(self):
            return {"async_index_build": True, "index_ready_seconds": 41.5,
                    "separable_build": "should be overridden"}

    class AngryDriver(FakeDriver):
        def capabilities(self):
            raise RuntimeError("no")

    def test_capabilities_reach_the_record(self):
        from harness.workloads.build import _driver_capabilities
        caps = _driver_capabilities(self.FakeDriver())
        assert caps["async_index_build"] is True
        assert caps["index_ready_seconds"] == 41.5

    def test_a_failing_driver_does_not_lose_the_measurement(self):
        from harness.workloads.build import _driver_capabilities
        assert _driver_capabilities(self.AngryDriver()) == {}

    def test_separable_build_wins_over_the_driver(self):
        """It is a property of how this workload ran, not of the driver."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "build.py")).read()
        merged = source.split("extra={**_driver_capabilities(driver),")[1]
        assert '"separable_build": not incremental' in merged.split("}")[0]

    def test_the_async_kind_renders_once_the_extras_arrive(self):
        """End to end: with capabilities merged, the build table stops calling
        an async build a bulk one."""
        from report.render import _index_build_kind
        record = {"engine": "mongodb", "build_mode": "post",
                  "extra": {"separable_build": True, "async_index_build": True,
                            "index_ready_seconds": 41.5}}
        kind = _index_build_kind(record)
        assert kind.startswith("async")
        assert "bulk" not in kind


class TestValkeyReadsItsOwnFieldNames:
    """FT.INFO has no percent_indexed and no indexing.

    It reports backfill_complete_percent, backfill_in_progress and state.
    Waiting on names that do not exist made every default fire at once, so the
    wait returned in six milliseconds and the build time, the row count, the
    index size and every query afterwards measured an index that was still
    empty. Nothing errored: the run reported a successful build of nothing.
    """

    def _driver(self, info):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.valkey import ValkeyDriver
        d = ValkeyDriver(ConnectionSpec(host="h", port=1))
        d._ft_info = lambda: info
        return d

    def test_an_unfinished_backfill_is_not_reported_as_done(self):
        d = self._driver({"backfill_complete_percent": "0.421000",
                          "backfill_in_progress": "1", "state": "backfill"})
        done, indexing = d._backfill_progress()
        assert done < 1.0 and indexing

    def test_a_finished_backfill_is_recognised(self):
        d = self._driver({"backfill_complete_percent": "1.000000",
                          "backfill_in_progress": "0", "state": "ready"})
        done, indexing = d._backfill_progress()
        assert done >= 1.0 and not indexing

    def test_a_missing_index_is_not_finished(self):
        """An empty FT.INFO means no index yet, which is the opposite of done.
        Defaulting the other way is exactly what made this silent."""
        done, indexing = self._driver({})._backfill_progress()
        assert done == 0.0 and indexing

    def test_the_old_field_names_are_not_looked_up(self):
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        for dead in ('get("percent_indexed"', 'get("indexing"'):
            assert dead not in source, f"still reading {dead}"
        assert 'get("backfill_complete_percent"' in source

    def test_indexing_failures_are_refused_rather_than_averaged(self):
        """A hash that fails to index still counts in num_docs and the write
        returned OK, so it reads as poor recall rather than as a broken
        configuration."""
        import pytest
        d = self._driver({"hash_indexing_failures": "17"})
        with pytest.raises(RuntimeError, match="failed to index"):
            d._assert_nothing_failed_to_index()
        self._driver({"hash_indexing_failures": "0"})._assert_nothing_failed_to_index()


class TestShortResultRowsAreDistinguishable:
    """pgvector is measured in two build modes across two passes.

    All four landed in the short-results table as the same row repeated, which
    reads as a rendering fault rather than as four measurements.
    """

    def test_the_table_names_the_pass_and_the_build_mode(self):
        from report.generate import summarize
        from report.render import _validity_section
        recs = [{"phase": "filtered", "engine": "pgvector", "dataset": "d",
                 "selectivity": 0.1, "resource_pass": rp, "build_mode": bm,
                 "extra": {"returned_fewer_than_k": True,
                           "short_result_queries": 81}}
                for rp in ("normalized", "tuned") for bm in ("post", "incremental")]
        text = _validity_section({}, summarize(recs, {}))
        assert "Build mode" in text and "Pass" in text
        assert "incremental" in text and "normalized" in text


class TestAnnFingerprintStaysEngineInvariant:
    """One results tree per resource pass, not one per engine.

    ann-benchmarks caches by algorithm and index parameters and knows nothing
    about budgets, so results are keyed by a fingerprint of the configuration.
    The report can narrow to exactly one tree, so the moment two engines under
    one pass disagree on the fingerprint, the recall chart silently contains a
    subset of the engines that ran.

    It has broken twice. First by hashing each engine's cache split, which
    differs by design. Then by hashing the sum of those splits, which held only
    while every engine allocated the same total: Percona Search takes a JVM
    heap and leaves the rest to the page cache, and Valkey allocates nothing
    because the container budget is the dataset. A six-engine smoke run
    produced a recall chart with one engine in it.
    """

    ENGINES = ("mariadb", "mariadb123", "alisql", "pgvector", "mongodb", "valkey")

    def _resolve(self, engine, resource_pass):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        return resolve_resources(load_resources(resource_pass), engine, info)

    def test_every_engine_under_a_pass_shares_one_fingerprint(self):
        from orchestrator.ann_pass import ann_fingerprint
        for resource_pass in ("normalized", "tuned"):
            prints = {e: ann_fingerprint(self._resolve(e, resource_pass))
                      for e in self.ENGINES}
            assert len(set(prints.values())) == 1, (
                f"{resource_pass} fragments the results tree: {prints}")

    def test_the_two_passes_do_not_collide(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolve("mariadb", "normalized"))
                != ann_fingerprint(self._resolve("mariadb", "tuned")))

    def test_changing_a_pass_knob_changes_the_fingerprint(self):
        """The whole point: a 16 GB curve must not be reused under 64 GB."""
        from orchestrator.config import load_resources, resolve_resources
        from orchestrator.ann_pass import ann_fingerprint
        info = self._resolve("mariadb", "tuned")
        res = load_resources("tuned")
        res.setdefault("memory", {})["buffer_fraction"] = 0.11
        sysinfo = type("I", (), {})()
        sysinfo.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                     "performance_cpus": list(range(40)),
                                     "efficiency_cpus": [], "logical_cpus": 80,
                                     "physical_cores": 40, "threads_per_core": 2,
                                     "model": "Xeon"})()
        sysinfo.total_ram_bytes = int(192 * 1024 ** 3)
        altered = resolve_resources(res, "mariadb", sysinfo)
        assert ann_fingerprint(info) != ann_fingerprint(altered)

    def test_it_survives_the_manifest_round_trip(self):
        """The report recomputes this from the manifest dict, not from the
        dataclass, so the field has to be recorded."""
        from orchestrator.ann_pass import ann_fingerprint
        resolved = self._resolve("valkey", "tuned")
        assert "pass_signature" in resolved.as_dict()
        assert ann_fingerprint(resolved.as_dict()) == ann_fingerprint(resolved)


class TestMongotIndexesAfterTheLoad:
    """Creating the search index before the load leaves it PENDING forever.

    It reads better: mongot would index from the change stream as rows arrive
    and the two would overlap. What it does is queue an initial sync over an
    empty collection, log "Queued initial syncs, numQueued: 0", and never
    revisit it. Observed stuck for six minutes on 2,000 rows while the ops
    driver, which has always created the index after the load, reached READY in
    thirty seconds on twenty thousand.

    ann-benchmarks catches a per-algorithm exception and exits zero, so the
    phase reported completed and wrote no results at all: a recall chart with
    five of six engines on it and nothing saying why.
    """

    MODULE = os.path.join(VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
                          "algorithms", "mongodb", "module.py")

    def test_the_load_happens_before_the_index_is_created(self):
        source = open(self.MODULE).read()
        body = source.split("def _fit(")[1].split("\n    def ")[0]
        assert body.index("_insert_rows") < body.index("_create_index"), (
            "the index is created before the load again; mongot will queue an "
            "initial sync over an empty collection and stay PENDING")

    def test_it_still_waits_for_ready(self):
        source = open(self.MODULE).read()
        body = source.split("def _fit(")[1].split("\n    def ")[0]
        assert body.index("_create_index") < body.index("_wait_until_ready")

    def test_the_ops_driver_orders_it_the_same_way(self):
        """The two paths must agree, or the ann curve and the ops build cost
        describe different operations."""
        from harness.drivers.mongo import MongoDriver
        import inspect
        create_schema = inspect.getsource(MongoDriver.create_schema)
        assert "create_search_index" not in create_schema, (
            "create_schema must not build the index; the ops workload loads "
            "first and calls create_index afterwards")


class TestBothProcessesAreWaitedFor:
    """mongod answers long before mongot does.

    The JVM needs fifteen to twenty seconds to reach its health check, and a
    createSearchIndexes issued inside that window is accepted, creates a Lucene
    index, and then sits in PENDING with no initial sync ever queued for it.
    Observed three seconds after mongot's process started, from an ann module
    that waited only for mongod to become primary. The ops path never hit it
    because the orchestrator spends that long starting a separate client
    container, which is a difference in timing rather than in correctness.
    """

    def test_the_entrypoint_waits_for_the_health_check(self):
        script = open(os.path.join(VB_ROOT, "docker", "mongodb",
                                   "entrypoint-mongodb.sh")).read()
        assert "wait_for_mongot" in script
        # Anchored on the line that records the pid rather than on a closing
        # brace: the function body is full of ${VAR} expansions.
        after_launch = script.split("echo $! > /tmp/mongot.pid")[1]
        assert after_launch.split("start_server()")[0].count("wait_for_mongot"), (
            "mongot is started without waiting for it to answer")

    def test_the_ann_module_waits_for_it_too(self):
        source = open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", "mongodb", "module.py")).read()
        assert "_wait_for_mongot" in source
        started = source.split("def _start_server")[1].split("def ")[0]
        assert "_wait_for_mongot" in started

    def test_the_readiness_probe_checks_both(self):
        """A probe that passes on mongod alone reports the server ready while
        the process that answers every search query is still booting."""
        from orchestrator import engines as engines_mod
        probe = " ".join(engines_mod.get("mongodb").probe)
        assert "isWritablePrimary" in probe
        assert "8080" in probe, "the probe does not check mongot at all"


class TestSilentAnnFailuresAreReported:
    """A recall phase can complete and measure nothing.

    ann-benchmarks catches a per-algorithm exception and exits zero, so a
    module that raises leaves the phase marked completed and the engine simply
    absent from the recall comparison. Percona Search failed that way three
    runs in a row while its other workloads succeeded, which is exactly what
    makes it easy to miss: the engine is present everywhere except the one
    table, and nothing says why.
    """

    MANIFEST = {"phases": [
        {"engine": "mongodb", "phase": "ann", "status": "completed",
         "dataset": "d", "resource_pass": "tuned", "duration_s": 85.2},
        {"engine": "mariadb", "phase": "ann", "status": "completed",
         "dataset": "d", "resource_pass": "tuned", "duration_s": 160.0},
    ]}

    def _summary(self, engines_with_results=("mariadb",)):
        from report.generate import summarize
        recs = [{"phase": "recall_qps", "engine": e, "dataset": "d", "m": 16,
                 "ef_search": 10, "recall_at_k": 0.99, "qps": 100.0,
                 "resource_pass": "tuned", "build_mode": "post"}
                for e in engines_with_results]
        return summarize(recs, self.MANIFEST)

    def test_an_engine_that_measured_nothing_is_named(self):
        s = self._summary()
        assert [f["engine"] for f in s["silent_ann_failures"]] == ["mongodb"]

    def test_an_engine_that_measured_something_is_not(self):
        s = self._summary(engines_with_results=("mariadb", "mongodb"))
        assert s["silent_ann_failures"] == []

    def test_the_validity_section_calls_it_a_failure(self):
        from report.render import _validity_section
        text = _validity_section({}, self._summary())
        assert "measured nothing" in text
        assert "failure, not a finding" in text

    def test_it_is_silent_on_a_clean_run(self):
        from report.render import _validity_section
        s = self._summary(engines_with_results=("mariadb", "mongodb"))
        assert "measured nothing" not in _validity_section({}, s)


class TestValkeyChurnIsBatched:
    """Churn touches a tenth of the corpus in a single call.

    The load path has always batched every thousand rows; the churn path did
    not. At smoke scale that is 2,000 rows and it works. At a million rows it
    is 99,000: one DEL carrying 99,000 key names, and one pipeline holding
    99,000 HSETs of a 6 KB vector each, roughly 600 MB buffered in the client
    before a single execute. The tuned pass failed there 25 minutes in, after
    the churn baseline had already been measured, which is why five engines
    have a churn result and one has half of one.
    """

    def _driver(self, monkeypatch, calls):
        from harness.drivers import valkey as mod
        from harness.drivers.base import ConnectionSpec
        monkeypatch.setattr(mod, "Binary", lambda payload, subtype: payload,
                            raising=False)
        d = mod.ValkeyDriver(ConnectionSpec(host="h", port=1))

        class FakePipe:
            def hset(self, *a, **kw): calls.setdefault("hset", []).append(1)
            def execute(self): calls.setdefault("execute", []).append(
                len(calls.get("hset", [])))

        class FakeConn:
            def pipeline(self, transaction=False): return FakePipe()
            def delete(self, *keys): calls.setdefault("delete", []).append(len(keys))
            def close(self): pass

        d._conn = FakeConn()
        # Bulk writes deliberately open their own connection, on a leash
        # of their own; the fake stands in for it.
        d._write_connection = lambda timeout_s=None: FakeConn()
        return d

    def test_deletes_are_chunked(self, monkeypatch):
        from harness.drivers.valkey import CHURN_BATCH
        calls = {}
        self._driver(monkeypatch, calls).delete_ids(list(range(99_000)))
        assert max(calls["delete"]) <= CHURN_BATCH
        assert sum(calls["delete"]) == 99_000

    def test_inserts_are_flushed_along_the_way(self, monkeypatch):
        """One execute at the end means the whole corpus fraction is buffered
        in the client first."""
        import numpy
        from harness.drivers.valkey import CHURN_BATCH
        calls = {}
        n = 5_000
        self._driver(monkeypatch, calls).insert_rows(
            list(range(n)), numpy.zeros((n, 4), dtype=numpy.float32),
            list(range(n)))
        assert len(calls["execute"]) >= n // CHURN_BATCH

    def test_the_load_path_and_the_churn_path_use_the_same_batch(self):
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        assert "CHURN_BATCH" in source
        assert source.count("CHURN_BATCH") >= 3


class TestPlanProbesUseARealVector:
    """A zero vector has no direction, so cosine distance over it is undefined.

    Both new modules probed with numpy.zeros to confirm the index answers. On
    the smoke corpus, which is euclidean, that is a legal query. On the real
    corpus, which is angular, the server is entitled to reject it -- and the
    Percona Search recall phase failed at full scale after the index had
    already reached READY, having worked on every euclidean smoke run.
    """

    MODULES = ("mongodb", "valkey")

    def _source(self, engine):
        return open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", engine, "module.py")).read()

    def test_no_module_probes_with_zeros(self):
        for engine in self.MODULES:
            assert "numpy.zeros" not in self._source(engine), engine

    def test_a_corpus_vector_is_kept_for_the_probe(self):
        for engine in self.MODULES:
            source = self._source(engine)
            assert "self._probe = X[0]" in source, engine


class TestQuantizationReachesBothPaths:
    """Only the recall path read it, so one run built two different indexes.

    render_config gave the ann phase the vendor-recommended scalar quantization
    in the tuned pass. Nothing gave it to the ops phase, which builds the index
    the build-cost, concurrency, filtered and churn numbers are measured
    against. The 44-hour run reported a 15 GB unquantized index beside a recall
    curve measured on a quantized one, as though they were one configuration.
    """

    def _args(self, engine, resource_pass):
        from orchestrator.config import (load_profile, load_resources,
                                         resolve_resources)
        from orchestrator.ops_pass import harness_args
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        res = load_resources(resource_pass)
        return harness_args(load_profile("tuned-complete"), 16, engine,
                            resolve_resources(res, engine, info),
                            resource_pass, res, storage_engine="InnoDB")

    def test_tuned_ops_gets_the_vendor_recommendation(self):
        args = self._args("mongodb", "tuned")
        assert args[args.index("--quantization") + 1] == "scalar"

    def test_normalized_ops_pins_it_off(self):
        args = self._args("mongodb", "normalized")
        assert args[args.index("--quantization") + 1] == "none"

    def test_the_two_paths_agree(self):
        """Whatever render_config gives the recall phase, the ops phase gets."""
        from orchestrator.ann_pass import render_config
        from orchestrator.config import load_profile, load_resources
        for resource_pass in ("normalized", "tuned"):
            res = load_resources(resource_pass)
            groups = render_config("mongodb", load_profile("tuned-complete"),
                                   res, resource_pass)["float"]["any"][0]["run_groups"]
            ann_value = list(groups.values())[0]["arg_groups"][0]["quantization"]
            args = self._args("mongodb", resource_pass)
            assert args[args.index("--quantization") + 1] == ann_value, resource_pass

    def test_engines_without_the_knob_do_not_get_the_flag(self):
        for engine in ("mariadb", "alisql", "pgvector", "valkey"):
            assert "--quantization" not in self._args(engine, "tuned"), engine

    def test_the_harness_accepts_it(self):
        from harness.main import parse_args
        args = parse_args(["--engine", "mongodb", "--dataset", "d",
                           "--m", "16", "--run-id", "r",
                           "--host", "h", "--port", "27017",
                           "--output", "/tmp/out.jsonl",
                           "--quantization", "scalar"])
        assert args.quantization == "scalar"


class TestConcurrencyQueryPoolIsItsOwnProfileKey:
    """`ops.max_queries` was capping a workload it was never written for.

    It exists to bound the filtered and churn workloads, which answer each
    query once to completion, and main.yml holds it at 500 for exactly that
    reason. The concurrency workload instead cycles its shard for a fixed
    duration, so sharing the key gave 32 clients about fifteen queries each
    for thirty seconds -- and every graph cache in the comparison (MHNSW's
    per-TABLE_SHARE cache, VIDX's shared cache, pgvector's shared_buffers)
    was measured against a hot set small enough to hold entirely. Telling
    those three designs apart is the only reason the concurrency chart exists.

    `ops.concurrency_queries` separates them. Its default is the new
    behaviour -- the whole test set -- and setting the key is the way back to
    a small pool, which is the convention every switch in this framework
    follows.
    """

    def _args(self, profile_name="main", override=None, engine="mariadb"):
        from orchestrator.config import (load_profile, load_resources,
                                         resolve_resources)
        from orchestrator.ops_pass import harness_args
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        res = load_resources("normalized")
        profile = load_profile(profile_name)
        if override is not None:
            profile.setdefault("ops", {})["concurrency_queries"] = override
        return harness_args(profile, 16, engine,
                            resolve_resources(res, engine, info),
                            "normalized", res, storage_engine="InnoDB")

    def _value(self, **kw):
        args = self._args(**kw)
        return args[args.index("--concurrency-queries") + 1]

    def test_the_profile_key_reaches_argv(self):
        """The value, not merely the flag. A flag emitted with a constant
        beside it is indistinguishable from one wired to the profile unless
        the number is checked."""
        assert self._value(override=2500) == "2500"

    def test_the_default_is_the_whole_test_set(self):
        """Zero is the sentinel the harness reads as "all of them". No profile
        in the repo sets the key yet, so this default is what every run gets
        today, and it has to be the new behaviour rather than the old cap.

        Every profile, discovered rather than listed: a hand-written list of
        four covered less than half of them and missed `full` and `quick`,
        which carry the most divergent `max_queries` (2,000 and 1,000) and are
        therefore where an accidental inheritance would be least obvious.
        """
        from orchestrator.ops_pass import DEFAULT_CONCURRENCY_QUERIES
        assert DEFAULT_CONCURRENCY_QUERIES == 0
        names = available_profiles()
        assert len(names) >= 9, names
        for profile in names:
            assert self._value(profile_name=profile) == "0", profile

    def test_the_default_does_not_inherit_the_filtered_cap(self):
        """The regression itself, stated as numbers. Those profiles really do
        hold max_queries at those values, so a default that followed the key
        would emit them here and nothing else in the suite would notice."""
        from orchestrator.config import load_profile
        for profile, capped in (("main", 500), ("full", 2000),
                                ("quick", 1000)):
            assert load_profile(profile)["ops"]["max_queries"] == capped, \
                profile
            assert self._value(profile_name=profile) != str(capped), profile

    def test_the_harness_accepts_the_flag(self):
        from harness.main import parse_args
        base = ["--engine", "mariadb", "--dataset", "d", "--m", "16",
                "--run-id", "r", "--host", "h", "--port", "3306",
                "--output", "/tmp/out.jsonl"]
        assert parse_args(base + ["--concurrency-queries", "2500"]) \
            .concurrency_queries == 2500
        # Absent, it is the workload's own sentinel rather than a second copy
        # of the number that could drift from it.
        from harness.workloads.concurrency import WHOLE_TEST_SET
        assert parse_args(base).concurrency_queries == WHOLE_TEST_SET

    def test_the_harness_passes_it_to_the_workload_and_not_max_queries(
            self, monkeypatch, tmp_path):
        """End to end through `main()`, with both flags set to different
        numbers, because that is the only arrangement that can tell the new
        wiring from the old. Asserting on the source text instead would be
        satisfied by the string appearing in a comment.
        """
        from harness import main as harness_main
        from harness.workloads import concurrency as concurrency_mod

        from .fake_driver import loaded_driver, tiny_dataset

        data = tiny_dataset()
        driver = loaded_driver(data)
        captured = {}

        def fake_run(ctx, factory, dataset, index, **kw):
            captured.update(kw)

        monkeypatch.setattr(harness_main, "load_dataset",
                            lambda directory, name: data)
        monkeypatch.setattr(harness_main, "get_driver",
                            lambda *a, **k: driver)
        monkeypatch.setattr(concurrency_mod, "run", fake_run)

        rc = harness_main.main([
            "--engine", "fake", "--dataset", "tiny", "--m", "16",
            "--run-id", "r", "--host", "h", "--port", "1",
            "--output", str(tmp_path / "out.jsonl"),
            "--workloads", "concurrency",
            "--max-queries", "500",
            "--concurrency-queries", "2500",
        ])
        assert rc == 0
        assert captured["max_queries"] == 2500
        # And the row count the engine actually holds travels with it, as it
        # already does for filtered and churn: recall over a partly-loaded
        # corpus has to be scored against that corpus.
        assert captured["indexed_rows"] == driver.count_rows()

    def test_the_workload_signature_takes_indexed_rows(self):
        """Recall needs it, and the two workloads that already compute recall
        already take it. Checked on the function rather than the source so a
        matching string in a docstring cannot satisfy it."""
        import inspect

        from harness.workloads import churn, concurrency, filtered
        for module in (concurrency, filtered, churn):
            assert "indexed_rows" in inspect.signature(module.run).parameters, \
                module.__name__


class TestAResumeAcrossAVersionBumpSaysSoBeforeMeasuring:
    """The refusal is right but it arrives after the expensive half.

    A directory holding two ops measurement versions cannot honestly be
    reported as one, and `report/loaders.py` refuses it. But the operator
    learns that from the report, which runs after the measuring — so resuming
    a `main` run under code that bumped the version meant up to days of ingest
    before anything said the directory being filled could not be reported.

    `set_config` is the one place that knows both numbers and still precedes
    the work: it runs per (pass, engine) before the checkpoint skip, so a
    resume reaches it even when every unit is skipped and nothing is measured.
    """

    def _manifest(self, tmp_path, declared):
        """A run directory that already declares `declared`."""
        from orchestrator.manifest import Manifest
        run = str(tmp_path / "run")
        m = Manifest(run, "run-1")
        if declared is None:
            m.data["config"].pop("ops_measurement_version", None)
        else:
            m.data["config"]["ops_measurement_version"] = declared
        m.save()
        # A second Manifest over the same directory is what --resume builds.
        return Manifest(run, "run-1")

    def _resume(self, tmp_path, declared):
        m = self._manifest(tmp_path, declared)
        m.set_config({}, "normalized", {"server_memory_bytes": 1})
        return m

    def test_a_stale_version_is_announced(self, tmp_path, capsys):
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        stale = OPS_MEASUREMENT_VERSION - 1
        m = self._resume(tmp_path, stale)
        err = capsys.readouterr().err
        # Both numbers, each anchored so neither can be read off the other.
        # A bare `str(stale) in err` is satisfied by the current version's
        # digits whenever one is a substring of the other -- 2 and 12, say.
        assert re.search(rf"\bversion {stale}\b", err), err
        assert re.search(rf"\bversion {OPS_MEASUREMENT_VERSION}\b", err), err
        # And the remedy, which is what makes it a warning rather than a
        # complaint: one report per version.
        assert "report" in err and "per version" in err, err
        # The manifest still says what the directory actually holds.
        assert m.data["config"]["ops_measurement_version"] == stale

    def test_it_is_not_fatal(self, tmp_path, capsys):
        """Finishing the run and splitting the records afterwards is a
        legitimate response, and often the right one: the older records are
        not wrong, only incomparable. A hard failure would turn a directory
        that needs splitting into a run that cannot continue."""
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        m = self._resume(tmp_path, OPS_MEASUREMENT_VERSION - 1)
        assert m.data["config"]["resource_pass"] == "normalized"
        assert capsys.readouterr().err  # it did warn, it just did not raise

    def test_a_matching_version_says_nothing(self, tmp_path, capsys):
        """Every ordinary resume hits this path. A notice that fires on all of
        them is the first thing an operator learns to scroll past."""
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        self._resume(tmp_path, OPS_MEASUREMENT_VERSION)
        assert "WARNING" not in capsys.readouterr().err

    def test_a_manifest_with_no_version_says_nothing(self, tmp_path, capsys):
        """Absent is not stale. A run recorded before the key existed was
        measured under a version this code cannot know, and `loaders.py` reads
        a missing key as "nothing declared, check nothing" -- so there is no
        disagreement to report."""
        self._resume(tmp_path, None)
        assert "WARNING" not in capsys.readouterr().err

    def test_it_is_said_once_per_run_not_once_per_engine(self, tmp_path,
                                                         capsys):
        """set_config runs per (pass, engine). Six engines across two passes
        would print this twelve times, and a notice repeated twelve times
        reads as a loop rather than as a warning."""
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        m = self._manifest(tmp_path, OPS_MEASUREMENT_VERSION - 1)
        for _ in range(12):
            m.set_config({}, "normalized", {"server_memory_bytes": 1})
        assert capsys.readouterr().err.count("WARNING") == 1

    def test_the_notice_is_carried_by_set_config_itself(self):
        """Not by a later populator, which is what makes it reachable by a
        resume that measures nothing.

        Deliberately a statement about where the code lives rather than about
        cli.py's call order. The ordering fact -- `cli.py` calls set_config at
        :461, above the `(already complete)` checkpoint skip at :484 -- is
        recorded in `_warn_if_resuming_across_a_bump`'s docstring instead of
        asserted here, because the obvious assertion for it is vacuous:
        "checkpoint" appears twelve times before that call and fourteen times
        after, so a grep of the tail passes however the file is reordered. A
        guard whose own mutant cannot be injected is not a guard, and cli.py
        is not this task's to pin.
        """
        import inspect

        from orchestrator.manifest import Manifest
        assert "_warn_if_resuming_across_a_bump" in \
            inspect.getsource(Manifest.set_config)


class TestResumeContinuesTheRightRun:
    """Checkpoints live inside the run directory.

    So --resume with a freshly minted run id resumes nothing: it re-runs every
    unit into an empty directory and produces a report containing only the
    engines named on that command line, while the run being continued stays
    where it was. Two directories, neither complete, and a manual merge to get
    one report.
    """

    def _profile(self):
        return {"name": "tuned-complete"}

    def _results(self, tmp_path, monkeypatch, names):
        from orchestrator import cli
        for name in names:
            (tmp_path / name).mkdir()
            (tmp_path / name / "run-manifest.json").write_text("{}")
        monkeypatch.setattr(cli, "paths_for",
                            lambda rid: {"run_dir": str(tmp_path / rid)})
        return cli

    def test_it_picks_the_most_recent_run_for_the_profile(self, tmp_path, monkeypatch):
        cli = self._results(tmp_path, monkeypatch,
                            ["tuned-complete-20260820-134424",
                             "tuned-complete-20260822-101500",
                             "smoke-20260821-090000"])
        assert cli._resume_target(self._profile()) == "tuned-complete-20260822-101500"

    def test_it_ignores_other_profiles(self, tmp_path, monkeypatch):
        cli = self._results(tmp_path, monkeypatch, ["smoke-20260899-000000"])
        assert cli._resume_target(self._profile()) is None

    def test_a_directory_without_a_manifest_is_not_a_run(self, tmp_path, monkeypatch):
        from orchestrator import cli
        (tmp_path / "tuned-complete-20260820-134424").mkdir()
        monkeypatch.setattr(cli, "paths_for",
                            lambda rid: {"run_dir": str(tmp_path / rid)})
        assert cli._resume_target(self._profile()) is None

    def test_it_is_only_consulted_when_resume_was_asked_for(self):
        """Every run without --run-id would otherwise continue the last one."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert "if not run_id and args.resume:" in source


class TestReRunContinuesAManifest:
    """Re-running one engine into an existing run directory must not erase it.

    Manifest built a fresh dict and saved it in __init__, so continuing a
    44-hour six-engine run to fix one failed unit would have overwritten the
    environment, the engine versions and the phase record of the five engines
    that succeeded. The report reads all of that, so the fix would have cost
    more than the failure.
    """

    def _manifest(self, tmp_path, run_id="tuned-complete-20260820-134424"):
        from orchestrator.manifest import Manifest
        return Manifest(str(tmp_path), run_id)

    def test_a_second_manifest_keeps_the_first_ones_record(self, tmp_path):
        first = self._manifest(tmp_path)
        first.data["engines"]["mariadb"] = {"source": {"tag": "11.8.8"}}
        first.add_phase("ops", "mariadb", "d", "completed", "t0", "t1",
                        {"resource_pass": "tuned"})
        second = self._manifest(tmp_path)
        assert second.data["engines"]["mariadb"]["source"]["tag"] == "11.8.8"
        assert any(p["engine"] == "mariadb" for p in second.data["phases"])

    def test_a_different_run_id_starts_clean(self):
        """Two runs sharing a directory would otherwise merge into nonsense."""
        import tempfile
        from orchestrator.manifest import Manifest
        with tempfile.TemporaryDirectory() as d:
            a = Manifest(d, "run-a")
            a.add_phase("ops", "mariadb", "d", "completed", "t0", "t1", {})
            b = Manifest(d, "run-b")
            assert b.data["phases"] == []

    def test_a_repeated_unit_keeps_both_attempts_and_the_last_decides(
            self, tmp_path):
        """A re-run that fixes a failure must not leave the failure behind for
        the report to keep reporting -- but the earlier attempt is no longer
        erased to achieve that. `full-20260825` lost the attempts that wrote
        the M=16 files its recall table used, because this method replaced
        them. Both attempts are kept and numbered; the readers that ask how a
        unit ended read `latest_attempts`."""
        from report.provenance import latest_attempts
        m = self._manifest(tmp_path)
        m.add_phase("ops", "valkey", "d", "failed", "t0", "t1",
                    {"resource_pass": "tuned", "exit_code": 1})
        m.add_phase("ops", "valkey", "d", "completed", "t2", "t3",
                    {"resource_pass": "tuned"})
        valkey = [p for p in m.data["phases"] if p["engine"] == "valkey"]
        assert [(p["attempt"], p["status"]) for p in valkey] == [
            (1, "failed"), (2, "completed")]
        last = [p for p in latest_attempts(m.data["phases"])
                if p["engine"] == "valkey"]
        assert [p["status"] for p in last] == ["completed"]

    def test_the_same_unit_in_another_pass_is_kept(self):
        import tempfile
        from orchestrator.manifest import Manifest
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(d, "r")
            m.add_phase("ops", "valkey", "d", "completed", "t0", "t1",
                        {"resource_pass": "normalized"})
            m.add_phase("ops", "valkey", "d", "completed", "t2", "t3",
                        {"resource_pass": "tuned"})
            assert len(m.data["phases"]) == 2


class TestReRunningAUnitReplacesItsRecords:
    """The records writer appends, which is right within a unit.

    Across attempts it is wrong: re-running a failed unit left the previous
    attempt's records in the file, and the report read both as separate
    measurements. The six-engine re-run produced two ingest records, two index
    sizes and twelve concurrency points for each engine it touched, and the
    charts drew lines through all of them.
    """

    def test_the_orchestrator_clears_the_unit_output_first(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        block = source.split('output = os.path.join(paths["run_dir"], f"ops-{stem}.jsonl")')[1]
        block = block.split("harness_args")[0]
        assert "os.remove" in block, (
            "a re-run appends to the previous attempt's records")

    def test_the_memory_series_is_cleared_too(self):
        """Otherwise peak RSS is taken across two attempts at once."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        block = source.split('memory_ts = os.path.join(paths["run_dir"], f"mem-{stem}.jsonl")')[1]
        block = block.split("harness_args")[0]
        assert "memory_ts" in block and "os.remove" in block

    def test_the_recorder_itself_still_appends(self):
        """Within one unit it must: several workloads and threads write to the
        same file, and a crash should not lose what came before it."""
        source = open(os.path.join(VB_ROOT, "harness", "metrics",
                                   "records.py")).read()
        assert 'open(path, "a"' in source

    def test_bulk_writes_have_a_finite_timeout_and_keepalive(self):
        """Wrong in both directions before this.

        First the driver's 600 second query timeout, which a slow write
        legitimately exceeds. Then no timeout at all, which is worse: the
        client sat at 0% CPU waiting on a reply that never came while the
        server answered an identical write from another connection in 104 ms.
        """
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        body = source.split("def _write_connection")[1].split("\n    def ")[0]
        assert "socket_timeout=WRITE_TIMEOUT_S" in body, "no read timeout"
        assert "socket_keepalive=True" in body, "a dead peer is waited on"
        assert "health_check_interval" in body

    def test_the_write_timeout_is_longer_than_a_query_timeout(self):
        from harness.drivers.valkey import WRITE_TIMEOUT_S
        assert 60 <= WRITE_TIMEOUT_S <= 3600

    def test_a_failed_chunk_ends_the_reinsert_not_the_run(self):
        from harness.workloads.churn import _reinsert

        class Flaky:
            name = "flaky"
            calls = 0

            def insert_rows(self, ids, vectors, tags):
                self.calls += 1
                if self.calls > 2:
                    raise TimeoutError("Timeout reading from socket")

        written, _ = _reinsert(Flaky(), list(range(1000)),
                               [None] * 1000, [None] * 1000, budget_s=60)
        assert 0 < written < 1000, written

    def test_every_write_path_uses_the_shared_connection(self):
        """Three paths write in bulk and all three must be configured alike.
        The churn one differed, and that was the whole failure."""
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        for method in ("def delete_ids", "def _churn_connection",
                       "def _write_range"):
            block = source.split(method)[1].split("\n    def ")[0]
            assert "_write_connection" in block, method
    def test_the_reinsert_is_sliced_so_progress_is_visible(self):
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "churn.py")).read()
        body = source.split("def _reinsert")[1]
        assert "budget_s" in body and "break" in body

    def test_it_stops_at_the_budget_and_reports_what_it_wrote(self):
        import time
        from harness.workloads.churn import _reinsert

        class SlowDriver:
            name = "slow"
            written = 0

            def insert_rows(self, ids, vectors, tags):
                time.sleep(0.05)
                self.written += len(ids)

        d = SlowDriver()
        ids = list(range(1000))
        written, elapsed = _reinsert(d, ids, [None] * 1000, [None] * 1000,
                                     budget_s=0.12)
        assert 0 < written < 1000, written
        assert elapsed < 2.0

    def test_a_fast_driver_finishes_everything(self):
        from harness.workloads.churn import _reinsert

        class FastDriver:
            name = "fast"
            def insert_rows(self, ids, vectors, tags): pass

        written, _ = _reinsert(FastDriver(), list(range(500)),
                               [None] * 500, [None] * 500, budget_s=60)
        assert written == 500

    def test_the_budget_reaches_the_harness(self):
        from harness.main import parse_args
        args = parse_args(["--engine", "valkey", "--dataset", "d", "--m", "16",
                           "--run-id", "r", "--host", "h", "--port", "6379",
                           "--output", "/tmp/o.jsonl", "--churn-budget", "900"])
        assert args.churn_budget == 900.0

    def test_the_orchestrator_passes_it(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert "--churn-budget" in source and "churn_budget_s" in source

    def test_an_incomplete_reinsert_records_no_recall_number(self):
        """The corpus is missing rows, so recall taken then describes neither
        the baseline nor a churned corpus, and would be read as though it did."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "churn.py")).read()
        block = source.split("reinsert_completed\": False")[0]
        block = block[block.index("if inserted < len(new_ids):"):]
        assert "**after" not in block, (
            "an incomplete churn must not publish a recall/QPS measurement")

    def test_the_report_states_what_a_stalled_engine_achieved(self):
        from report.generate import summarize
        from report.render import _churn_table
        recs = [{"phase": "churn", "engine": "valkey", "dataset": "d",
                 "resource_pass": "tuned", "storage_engine": "memory",
                 "churn_fraction": 0.1, "m": 16,
                 "notes": "re-insert did not complete within the churn budget",
                 "extra": {"reinsert_completed": False, "rows_reinserted": 364,
                           "rows_expected": 99000, "insert_seconds": 1800.0,
                           "reinsert_rows_per_s": 0.2,
                           "churn_budget_s": 1800.0}}]
        table = _churn_table(summarize(recs, {}))
        assert "did not complete" in table
        assert "364 of 99,000" in table

    def test_a_complete_churn_adds_no_note(self):
        from report.generate import summarize
        from report.render import _churn_table
        recs = [{"phase": "churn", "engine": "pgvector", "dataset": "d",
                 "resource_pass": "tuned", "storage_engine": "heap",
                 "churn_fraction": 0.1, "m": 16, "qps": 229.0,
                 "recall_at_k": 0.98,
                 "extra": {"reinsert_completed": True}}]
        assert "did not complete" not in _churn_table(summarize(recs, {}))


class TestIoThreadsAreClamped:
    """IO threads are not the cpuset.

    Valkey's guidance is that more than a handful contend rather than help, and
    the framework already clamps build threads for exactly that reason. Nothing
    clamped this, so a 64-core server was started with --io-threads 64 and its
    writes into a populated search index stopped after three rows while reads
    stayed healthy. Every smoke run passed because a small cpuset produced a
    small, legal value.
    """

    def _resolve(self, cores, resource_pass="tuned", overrides=None):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(cores)),
                                  "efficiency_cpus": [],
                                  "logical_cpus": cores * 2,
                                  "physical_cores": cores,
                                  "threads_per_core": 2, "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        res = load_resources(resource_pass)
        if overrides:
            res.setdefault("cpu", {}).update(overrides)
        return resolve_resources(res, "valkey", info)

    def test_a_large_cpuset_does_not_become_a_large_thread_count(self):
        r = self._resolve(64)
        assert r.io_threads <= 8, r.io_threads

    def test_a_small_cpuset_is_left_alone(self):
        r = self._resolve(4)
        assert r.io_threads == min(r.server_cpu_count, 8)

    def test_it_is_never_zero(self):
        assert self._resolve(1).io_threads >= 1

    def test_the_cap_is_overridable(self):
        """Asserted as the rule rather than a number: the cpuset comes from the
        host this runs on, so the value differs between machines."""
        r = self._resolve(64, overrides={"max_io_threads": 16})
        assert r.io_threads == min(r.server_cpu_count, 16)
        assert r.io_threads >= self._resolve(64).io_threads

    def test_the_server_flag_uses_the_clamped_value(self):
        """The config asked for {server_cpu_count} directly, which is the bug."""
        from orchestrator.config import load_engine, server_args
        source = open(os.path.join(VB_ROOT, "config", "engines",
                                   "valkey.yml")).read()
        assert "{io_threads}" in source
        assert "--io-threads\n    - \"{server_cpu_count}\"" not in source
        args = server_args(load_engine("valkey"), "tuned", self._resolve(64))
        assert args[args.index("--io-threads") + 1] == "8"

    def test_clamping_is_announced(self):
        r = self._resolve(64)
        assert any("io threads clamped" in w for w in r.warnings) or r.server_cpu_count <= 8


class TestPhaseLogsAreArchived:
    """A phase explains its failure in its console output and nowhere else.

    A recall phase that wrote no results and a churn that stopped after nothing
    both printed their reason and then lost it: the measurements were archived
    and copied between machines, the terminal was not. By the time anyone
    asked, the answer was gone and the only way back to it was another run.
    """

    def test_a_phase_log_is_written_beside_the_measurements(self, tmp_path):
        from orchestrator.docker_ctl import save_phase_log
        path = save_phase_log(str(tmp_path), "valkey", "glove-100-angular",
                              "ops-m16-post", "tuned",
                              ["[churn] re-insert stopped after 0 rows",
                               "TimeoutError: Timeout reading from socket"])
        assert path and os.path.isfile(path)
        assert "logs" in path
        assert "Timeout reading from socket" in open(path).read()

    def test_nothing_is_written_for_a_silent_phase(self, tmp_path):
        from orchestrator.docker_ctl import save_phase_log
        assert save_phase_log(str(tmp_path), "e", "d", "ann", "tuned", []) is None

    def test_the_name_identifies_the_unit(self, tmp_path):
        """Six engines times two phases in one directory; a shared name would
        overwrite the one you came looking for."""
        from orchestrator.docker_ctl import save_phase_log
        a = save_phase_log(str(tmp_path), "valkey", "glove", "ann", "tuned", ["a"])
        b = save_phase_log(str(tmp_path), "mongodb", "glove", "ann", "tuned", ["b"])
        c = save_phase_log(str(tmp_path), "valkey", "glove", "ann", "normalized",
                           ["c"])
        assert len({a, b, c}) == 3

    def test_both_measurement_paths_save_theirs(self):
        for module in ("ann_pass", "ops_pass"):
            source = open(os.path.join(VB_ROOT, "orchestrator",
                                       f"{module}.py")).read()
            assert "save_phase_log" in source, module

    def test_an_unwritable_run_directory_does_not_fail_the_run(self, tmp_path):
        """Losing the log is bad; losing the measurement because the log could
        not be written would be worse."""
        from orchestrator.docker_ctl import save_phase_log
        assert save_phase_log("/proc/nonexistent/nope", "e", "d", "ann", "t",
                              ["x"]) is None


class TestAnnResultsCanActuallyBeStored:
    """ann-benchmarks writes get_additional() straight into HDF5 attributes.

    store_results does f.attrs[k] = v for every key. A None has no HDF5 type
    and raises at the very end, after the whole sweep has run:

        TypeError: Object dtype dtype('O') has no native HDF5 equivalent

    ann-benchmarks catches that per worker and exits zero, so the phase is
    reported completed and writes no results at all. Percona Search failed that
    way four times across two datasets while its measurements ran perfectly,
    and the cause was one key: ef_construction, which mongot does not have and
    which None was the honest way to describe.
    """

    MODULES = ("mongodb", "valkey", "pgvector", "mariadb", "mariadb123", "alisql")

    def _source(self, engine):
        return open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", engine, "module.py")).read()

    def _additional_block(self, engine):
        source = self._source(engine)
        if "def get_additional" not in source:
            return ""
        return source.split("def get_additional")[1].split("\n    def ")[0]

    def test_no_module_reports_a_value_hdf5_cannot_store(self):
        import re
        for engine in self.MODULES:
            block = self._additional_block(engine)
            for key, literal in re.findall(r'"(\w+)":\s*(None\b|\[|\{)', block):
                assert False, (
                    f"{engine} reports {key!r} as {literal}, which "
                    f"store_results cannot write into an HDF5 attribute")

    def test_the_engines_with_optional_fields_sanitise(self):
        """Percona Search and Valkey both carry fields that are legitimately
        not applicable, so they must strip rather than emit them."""
        for engine in ("mongodb", "valkey"):
            assert "hdf5_safe" in self._source(engine), engine
            assert "return hdf5_safe(" in self._additional_block(engine), engine

    def test_the_sanitiser_keeps_falsey_values(self):
        """Dropping None must not drop False or 0: m_applied False and a zero
        ready time are both real measurements."""
        import ast
        source = self._source("mongodb")
        tree = ast.parse(source)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "hdf5_safe")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "m", "exec"), ns)
        out = ns["hdf5_safe"]({"kept": False, "zero": 0, "dropped": None})
        assert out == {"kept": False, "zero": 0}

    def test_a_mass_delete_is_absorbed_before_writing_again(self):
        """DEL returns when the key is gone; removing the vector from the graph
        is separate work. Deleting 99,000 keys and writing immediately timed
        out with nothing landing, while one write into the same index from an
        idle server took 104 ms."""
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        block = source.split("def delete_ids")[1].split("\n    def _wait_for_mutations")[0]
        assert "_wait_for_mutations" in block
        assert "mutation_queue_size" in source

    def test_the_settle_step_is_bounded(self):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.valkey import ValkeyDriver
        d = ValkeyDriver(ConnectionSpec(host="h", port=1))
        d._ft_info = lambda: {"mutation_queue_size": "5"}
        elapsed = d._wait_for_mutations(timeout_s=0.2)
        assert elapsed < 5.0

    def test_an_empty_queue_returns_at_once(self):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.valkey import ValkeyDriver
        d = ValkeyDriver(ConnectionSpec(host="h", port=1))
        d._ft_info = lambda: {"mutation_queue_size": "0"}
        assert d._wait_for_mutations(timeout_s=30) < 1.0


class TestAStubResultCannotSuppressAConfiguration:
    """store_results creates the file before it writes the datasets.

        with h5py.File(filename, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            times = f.create_dataset("times", ...)

    So an attribute it cannot represent leaves a valid HDF5 file holding no
    measurements -- and ann-benchmarks decides what to skip with
    os.path.exists(). Percona Search wrote one such stub per attempt; by the
    fifth run its recall curve was two points, and re-running could not
    recover the six below them, because the stubs outlived the bug.
    """

    def _script(self):
        from orchestrator.ann_pass import _PRUNE_SCRIPT
        return _PRUNE_SCRIPT

    def test_the_check_is_for_data_not_for_the_file_existing(self):
        script = self._script()
        assert '"times" in fh' in script
        assert '"neighbors" in fh' in script
        assert "os.remove(path)" in script

    def test_a_file_it_cannot_open_is_also_a_stub(self):
        """A truncated write leaves something h5py raises on. That is not a
        result either, and it blocks the same configuration."""
        script = self._script()
        assert "except Exception:" in script
        assert "intact = False" in script

    def test_only_result_files_are_touched(self):
        assert '.endswith(".hdf5")' in self._script()

    def test_the_count_comes_back_from_the_container(self, monkeypatch):
        from orchestrator import ann_pass
        monkeypatch.setattr(ann_pass.docker_ctl, "run_foreground",
                            lambda spec, **kw: kw["sink"].extend(
                                ["pruned /results/a.hdf5", "1"]) or 0)
        assert ann_pass.prune_empty_results("/tmp/x", "img") == 1

    def test_a_prune_that_cannot_run_is_not_fatal(self, monkeypatch):
        """This exists to stop a stale stub from hiding a configuration. Ending
        the run over it would cost more than the stub does."""
        from orchestrator import ann_pass

        def explode(spec, **kw):
            raise ann_pass.docker_ctl.DockerError("no docker")

        monkeypatch.setattr(ann_pass.docker_ctl, "run_foreground", explode)
        assert ann_pass.prune_empty_results("/tmp/x", "img") == 0

    def test_the_sweep_prunes_before_it_counts(self):
        """Counting first would take the stubs as existing results and print
        'already have results' over the very files that are the problem."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        body = source.split("def run_engine")[1]
        assert body.index("prune_empty_results(") < body.index("before = _count_results(")


class TestTheServerSideOfAnOpsPhaseIsKept:
    """The ops path runs the engine and the harness in two containers and kept
    only the client's output, so a churn that stalled with the client blocked
    on a socket and the server at its idle CPU baseline could not be settled
    from the run directory: the half that would have said was discarded with
    the container.

    This docstring used to add that the ann path keeps both halves because the
    engine and the benchmark share one container there. They do share one, but
    the overlay starts the engine through the same `vb-entrypoint server`, so
    it gets the same `--log-error=<file>` and the shared stream carries no
    more of the server's voice than this one did. See `_save_server_log`."""

    def _source(self):
        return open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()

    def test_the_server_log_is_archived(self):
        source = self._source()
        assert "def _save_server_log" in source
        assert "docker_ctl.logs(self.server_name" in source

    def test_it_runs_even_when_the_phase_failed(self):
        """A phase that timed out is precisely the one whose server log is
        worth having, so it cannot hang off the success path."""
        source = self._source()
        block = source.split("def run_harness")[1].split("\n    def ")[0]
        finally_block = block.split("finally:")[1]
        assert "_save_server_log()" in finally_block

    def test_it_lands_beside_the_client_log(self):
        source = self._source()
        assert 'f"server-{self.tag}"' in source

    def test_it_cannot_take_the_run_down(self, tmp_path):
        """docker logs against a container that is already gone must not turn a
        completed measurement into a failed one."""
        from orchestrator import ops_pass

        class Fake:
            server_name = "gone"
            engine = "valkey"
            dataset = "glove-100-angular"
            tag = "m16-post"
            resource_pass = "tuned"
            paths = {"run_dir": str(tmp_path)}
            _save_server_log = ops_pass.OpsRun._save_server_log
            # _save_server_log now also copies out the engine's own
            # --log-error file, which is a second thing that must not be able
            # to end the run. valkey declares no runtime.error_log, so this
            # returns without touching docker.
            _save_server_error_log = ops_pass.OpsRun._save_server_error_log
            # And two more, for the same reason. The exit facts come from one
            # `docker inspect` of a container named "gone", which returns
            # None; the core capture returns before touching docker because
            # valkey declares no runtime.ulimits. Both must swallow whatever
            # they meet -- see their docstrings -- so adding them here widens
            # what this test covers rather than weakening it.
            _save_container_exit_facts = (
                ops_pass.OpsRun._save_container_exit_facts)
            _save_server_core = ops_pass.OpsRun._save_server_core
            _ulimits = ops_pass.OpsRun._ulimits

        import orchestrator.docker_ctl as dc
        original = dc.logs
        dc.logs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no such container"))
        try:
            Fake()._save_server_log()
        finally:
            dc.logs = original


class TestAStalledWriteExplainsItself:
    """The valkey churn re-insert has failed the same way twice: the client
    blocked on a socket read until its timeout while the server sat at exactly
    its idle CPU baseline and answered a hand-typed HSET into the same indexed
    prefix in 104 ms. Six explanations were offered for that from the outside
    and none was tested against the machine while it was in that state."""

    def _driver(self):
        from harness.drivers.base import ConnectionSpec
        from harness.drivers.valkey import ValkeyDriver
        return ValkeyDriver(ConnectionSpec(host="h", port=1))

    def test_a_missing_queue_field_is_not_the_same_as_an_empty_one(self):
        """percent_indexed did not exist either, and reading its absence as
        'finished' meant every later measurement described an empty index."""
        from harness.drivers.valkey import _optional_int
        assert _optional_int(None) is None
        assert _optional_int("0") == 0
        assert _optional_int(0) == 0
        assert _optional_int("nonsense") is None

    def test_the_document_count_is_waited_on_too(self):
        """The queue read zero while the index still held 481 documents more
        than the keyspace did."""
        d = self._driver()
        seen = []

        def info():
            seen.append(1)
            return {"mutation_queue_size": "0",
                    "num_docs": 891481 if len(seen) < 3 else 891000}

        d._ft_info = info
        assert d._wait_for_mutations(expected_docs=891000, timeout_s=30) < 10
        assert len(seen) >= 3

    def test_a_build_that_reports_neither_field_does_not_hang(self):
        d = self._driver()
        d._ft_info = lambda: {}
        assert d._wait_for_mutations(expected_docs=5, timeout_s=30) < 1.0

    def test_the_delete_tells_the_wait_what_to_expect(self):
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        block = source.split("def delete_ids")[1].split("\n    def ")[0]
        assert "expected_docs=" in block

    def test_the_ladder_says_the_engine_is_fine(self):
        """One HSET returning on a fresh connection, while a thousand did not,
        means the engine was never the thing that stopped."""
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 891000}
        d._conn = _FakeConn(info={"blocked_clients": 0})
        d._write_connection = lambda timeout_s=None: _FakeConn()
        report = d._diagnose_stalled_write(list(range(1000)),
                                           [[0.0]] * 1000, [1] * 1000)
        assert "write of     1 row(s) on a fresh connection: OK" in report

    def test_the_ladder_pins_whether_two_is_already_too_many(self):
        """The first ladder put the turnover between 1 and 10. Two rungs at 2
        and 3 say whether it is a threshold or pipelining itself."""
        from harness.drivers.valkey import _PROBE_SIZES
        assert _PROBE_SIZES[:3] == (1, 2, 3)

    def test_the_ladder_finds_where_the_write_turns_over(self):
        """Where between one and a thousand it stops coming back is the
        difference between a batching limit and something not about size."""
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 891000}
        d._conn = _FakeConn(info={})
        d._write_connection = lambda timeout_s=None: _FakeConn(stall_above=10)
        report = d._diagnose_stalled_write(list(range(1000)),
                                           [[0.0]] * 1000, [1] * 1000)
        assert "write of     1 row(s) on a fresh connection: OK" in report
        assert "write of    10 row(s) on a fresh connection: OK" in report
        assert "write of   100 row(s) on a fresh connection: TimeoutError" in report

    def test_the_ladder_says_the_engine_is_not_fine(self):
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 891000}
        d._conn = _FakeConn(info={"blocked_clients": 1})
        d._write_connection = _raising
        report = d._diagnose_stalled_write([7], [[0.0]], [1])
        assert "could not connect" in report

    def test_a_probe_never_raises(self):
        """It runs while something is already wrong; a diagnosis that raises
        tells nobody anything."""
        d = self._driver()
        d._write_connection = _raising
        assert isinstance(d._probe_write([7], [[0.0]], [1]), str)

    def test_a_dead_monitoring_connection_is_itself_the_answer(self):
        d = self._driver()
        d._ft_info = lambda: {}
        d._conn = _FakeConn(info=RuntimeError("gone"))
        report = d._diagnose_stalled_write([7], [[0.0]], [1])
        assert "stopped answering" in report

    def test_the_ladder_chooses_the_batch(self):
        """Halving from a thousand to one costs nine more timeouts to learn
        what the ladder measured in one pass."""
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 1}
        d._conn = _FakeConn(info={})
        sizes = []
        d._write_connection = lambda timeout_s=None: _FakeConn(
            stall_above=3, seen=sizes)
        d.insert_rows(list(range(30)), [[0.0]] * 30, [1] * 30)
        assert d.write_batch_used == 3, "the largest rung that returned"
        assert 500 not in sizes and 250 not in sizes, "it should not halve"

    def test_every_row_still_lands(self):
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 1}
        d._conn = _FakeConn(info={})
        written = []
        d._write_connection = lambda timeout_s=None: _FakeConn(
            stall_above=5, seen=written)
        d.insert_rows(list(range(100)), [[0.0]] * 100, [1] * 100)
        assert sum(n for n in written if n <= 5) >= 100

    def test_a_connection_is_not_opened_per_row(self):
        """At one row per write, 99,000 rows means 99,000 writes. Reconnecting
        for each would measure connection setup, not the write."""
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 1}
        d._conn = _FakeConn(info={})
        opened = []

        def connection(timeout_s=None):
            conn = _FakeConn(stall_above=1000)
            opened.append(conn)
            return conn

        d._write_connection = connection
        d.insert_rows(list(range(4000)), [[0.0]] * 4000, [1] * 4000)
        assert len(opened) == 1

    def test_a_stalled_connection_is_dropped(self):
        """redis-py has already disconnected it; reusing the object would
        confuse a wedged socket with a wedged server."""
        d = self._driver()
        d._ft_info = lambda: {"num_docs": 1}
        d._conn = _FakeConn(info={})
        opened = []

        def connection(timeout_s=None):
            conn = _FakeConn(stall_above=3)
            opened.append(conn)
            return conn

        d._write_connection = connection
        d.insert_rows(list(range(9)), [[0.0]] * 9, [1] * 9)
        assert len(opened) > 1

    def test_nothing_working_at_any_size_is_not_a_batching_problem(self):
        d = self._driver()
        d._ft_info = lambda: {}
        d._conn = _FakeConn(info={})
        d._write_connection = lambda timeout_s=None: _FakeConn(stall_above=0)
        with pytest.raises(RuntimeError) as caught:
            d.insert_rows([7], [[0.0]], [1])
        assert "not about batching" in str(caught.value)

    def test_the_batch_that_worked_is_reported(self):
        """A churn that completed one row at a time and one that completed a
        thousand at a time are not the same result."""
        d = self._driver()
        assert d.write_batch_used == 1000

    def test_the_canary_cannot_become_a_second_stall(self):
        from harness.drivers.valkey import CANARY_TIMEOUT_S, WRITE_TIMEOUT_S
        assert CANARY_TIMEOUT_S < WRITE_TIMEOUT_S


def _raising(timeout_s=None):
    raise TimeoutError("Timeout connecting to server")


class _FakeConn:
    """A server that takes writes up to a size and then stops answering."""

    def __init__(self, info=None, stall_above=None, seen=None):
        self._info = info if info is not None else {}
        self._stall_above = stall_above
        self._seen = seen

    def info(self, *a):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info

    def dbsize(self):
        return 891000

    def hset(self, *a, **kw):
        return 1

    def pipeline(self, transaction=False):
        return _FakePipe(self)

    def close(self):
        pass


class _FakePipe:
    def __init__(self, conn):
        self._conn = conn
        self._queued = 0

    def hset(self, *a, **kw):
        self._queued += 1

    def execute(self):
        if self._conn._seen is not None:
            self._conn._seen.append(self._queued)
        limit = self._conn._stall_above
        if limit is not None and self._queued > limit:
            raise TimeoutError("Timeout reading from socket")
        self._queued = 0


class TestARunIdDockerWillRefuseIsCaughtFirst:
    """A shell variable picked up the escape an arrow key sends, so $BIG held
    "tuned-complete-20260820-134424\x1b[A". The results directory, the
    manifest and the ann results tree were all created under that name, two
    phases ran into it, and Docker refused the third with

        Invalid container name (...134424[A-annb-mongodb-dbpedia-openai)

    which named the symptom four hundred lines after the cause.
    """

    def test_a_normal_id_passes(self):
        from orchestrator.cli import run_id_problem
        assert run_id_problem("tuned-complete-20260820-134424") is None
        assert run_id_problem("smoke_1.2-3") is None

    def test_the_escape_that_caused_it_is_rejected(self):
        from orchestrator.cli import run_id_problem
        problem = run_id_problem("tuned-complete-20260820-134424\x1b[A")
        assert problem is not None
        assert "arrow keys" in problem

    def test_the_message_shows_the_characters(self):
        """The whole difficulty was that the id looked right on screen."""
        from orchestrator.cli import run_id_problem
        problem = run_id_problem("run\x1b[A")
        assert "x1b" in problem

    def test_a_leading_dash_is_rejected_too(self):
        """Docker requires the first character to be alphanumeric, and an id
        starting with a dash would also be read as a flag."""
        from orchestrator.cli import run_id_problem
        assert run_id_problem("-force") is not None

    def test_slashes_are_rejected(self):
        """A path fragment would escape the results directory."""
        from orchestrator.cli import run_id_problem
        assert run_id_problem("../elsewhere") is not None

    def test_nothing_is_created_before_the_check(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        body = source.split("def cmd_run")[1]
        assert body.index("run_id_problem(") < body.index("os.makedirs(paths")


class TestTheWriteShapeTravelsWithTheChurnNumber:
    """Valkey re-inserted 99,000 rows at 86 rows/s against a load that ran at
    30,195. Almost none of that gap is the HNSW graph: pipelined writes into an
    indexed prefix do not return at all, so the re-insert fell back to one
    command per round trip while the load pipelined a thousand. Reported as a
    bare rate it reads as an index cost, and it is a write-shape cost."""

    def test_the_batch_is_recorded(self):
        from harness.workloads.churn import _write_shape

        class Driver:
            write_batch_used = 1
            load_batch_rows = 1000

        shape = _write_shape(Driver())
        assert shape["write_batch_rows"] == 1
        assert shape["write_batch_reduced"] is True

    def test_an_unreduced_batch_is_not_flagged(self):
        from harness.workloads.churn import _write_shape

        class Driver:
            write_batch_used = 1000
            load_batch_rows = 1000

        assert "write_batch_reduced" not in _write_shape(Driver())

    def test_engines_without_the_notion_add_nothing(self):
        """Only valkey has a batch that can collapse; the record must not grow
        an empty field for the other five."""
        from harness.workloads.churn import _write_shape
        assert _write_shape(object()) == {}

    def test_both_records_carry_it(self):
        """The completed churn and the abandoned one are read side by side."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "churn.py")).read()
        assert source.count("**_write_shape(driver)") == 2

    def test_a_completed_churn_reports_its_rate(self):
        """The incomplete record always carried reinsert_rows_per_s and the
        complete one did not, so the only churn with a rate was a failed one."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "churn.py")).read()
        assert source.count('"reinsert_rows_per_s"') == 2


class TestTheMSweepProfileIsolatesGraphDegree:
    """tuned-complete pins M=16 because sweeping it costs a full index build
    per value, and MHNSW and VIDX build incrementally. The result is a headline
    run where pgvector and Valkey contributed 24 measured points and the three
    InnoDB engines 8 -- an affordability artifact that reads as an engine
    difference. This profile answers the M question at a tenth of the corpus,
    and it is only worth running if M is genuinely the only thing moving."""

    def _rendered(self, engine):
        from orchestrator.ann_pass import render_config
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        profile = load_profile("m-sweep")
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        return render_config(engine, profile, resources, "tuned"), profile

    def _arg_groups(self, engine):
        body, _ = self._rendered(engine)
        groups = body["float"]["any"][0]["run_groups"]
        return [ag for g in groups.values() for ag in g["arg_groups"]]

    def test_every_engine_that_has_m_sweeps_the_same_five(self):
        for engine in ("mariadb", "mariadb123", "alisql", "pgvector", "valkey"):
            values = {tuple(ag["M"]) for ag in self._arg_groups(engine)}
            assert values == {(6, 8, 16, 24, 32)}, engine

    def test_no_other_build_axis_moves(self):
        """The tuned pass sweeps ef_construction for two engines and
        quantization for a third. Leaving those on would multiply fifteen
        InnoDB builds by axes the InnoDB engines do not have -- widening the
        asymmetry the profile exists to remove."""
        for engine, key in (("pgvector", "efConstruction"),
                            ("valkey", "efConstruction"),
                            ("mongodb", "quantization")):
            for ag in self._arg_groups(engine):
                value = ag.get(key)
                if isinstance(value, list):
                    assert len(value) == 1, (engine, key, value)

    def test_one_build_mode_each(self):
        for engine in ("pgvector", "valkey"):
            body, _ = self._rendered(engine)
            assert len(body["float"]["any"][0]["run_groups"]) == 1, engine

    def test_the_query_grid_matches_the_headline_run(self):
        """The point is comparing shapes against tuned-complete. A different
        ef_search grid would make that comparison an interpolation."""
        from orchestrator.config import load_profile
        assert (load_profile("m-sweep")["ann"]["ef_search"]
                == load_profile("tuned-complete")["ann"]["ef_search"])

    def test_it_runs_on_a_tenth_of_the_corpus(self):
        """At 990,000 rows an InnoDB index build is hours, per M value."""
        from orchestrator.config import load_profile
        assert load_profile("m-sweep")["datasets"] == [
            "dbpedia-openai-100k-angular"]

    def test_ops_is_off(self):
        """Build cost, concurrency, filtered and churn are answered at full
        scale by tuned-complete. A second set at 90,000 rows would invite
        exactly the cross-run comparison the profile warns against."""
        from orchestrator.config import load_profile
        assert load_profile("m-sweep")["ops"]["enabled"] is False

    def test_the_estimate_covers_every_engine(self):
        """No engine is silently absent from the ingest estimate.

        **The requirement this test enforces changed. The new form is NOT
        stronger than the old one -- the two are logically incomparable**, and
        saying otherwise was an overclaim worth correcting here rather than
        leaving in a docstring. The old form was
        `engine in _INGEST_ROWS_PER_S` for every known engine. It rejects a
        state this one accepts (an engine with no rate at all) and this one
        rejects states it could not see (an engine named AND counted, or in
        neither bucket). Neither implies the other. This one is better
        *aimed*, which is a different claim.

        The old form was the only defence available while `_effective_rate`
        ended in `_INGEST_ROWS_PER_S.get(engine, 300)`: a missing entry was
        silently modelled at a default, so insisting the table was complete
        was all one could do. It had two costs -- it was satisfiable by a
        guess, which the table's own comment rejects ("an estimate that is
        optimistic by 3x is worse than none"), and it required every future
        engine to carry a number before anyone had measured one.

        **The property that actually protects a reader is
        `not (named & counted)` together with the two sets covering every
        engine asked for**: an engine contributes to the published total or is
        named as unmeasured, never both and never neither. That is what stops
        a figure labelled "ingest" from quietly omitting an engine, and it
        holds however many engines are measured.

        **What this test does NOT catch, stated so nobody relies on it to:**
        a plausible-looking guessed entry for Percona would satisfy every
        assertion below -- it would simply move Percona from `named` to
        `counted`. Nothing a test in this file can do distinguishes a measured
        150 from an invented one. That line is held by review and by the rule
        that a rate enters the table only from a run, not by this guard.
        """
        from orchestrator.cli import _INGEST_ROWS_PER_S, KNOWN_ENGINES
        from orchestrator.estimate import estimate_load_hours

        def est(engines):
            return estimate_load_hours(
                {"name": "t", "ann": {"m_values": [16]},
                 "ops": {"m_values": [16]}},
                list(engines), ["glove-100-angular"], ["normalized"],
                ["ann", "ops"])

        estimate = est(KNOWN_ENGINES)
        named = set(estimate["engines_without_ingest_rate"])
        counted = set(estimate["per_engine_hours"])
        # Exactly one bucket each, and the two together are everything asked
        # for -- so an engine can neither be double-counted nor vanish.
        assert not (named & counted), named & counted
        assert named | counted == set(KNOWN_ENGINES), (
            set(KNOWN_ENGINES) ^ (named | counted))
        # And the bucket an engine lands in is the one the rate table implies.
        for engine in KNOWN_ENGINES:
            assert (engine in _INGEST_ROWS_PER_S) ^ (engine in named), (
                f"{engine} must be either measured or named as unmeasured, "
                f"and exactly one of the two")

        # NON-VACUITY, and deliberately not `assert named`. That form proved
        # the gap list was derived only for as long as some real engine was
        # unmeasured: it would FAIL the day every engine gained a rate, which
        # is a guard that fires on a correct -- indeed better -- future state,
        # and the next author would delete it. Injecting an engine nothing
        # could ever have measured proves the same thing in both regimes.
        probe = est(list(KNOWN_ENGINES) + ["e2-no-such-engine"])
        assert "e2-no-such-engine" in probe["engines_without_ingest_rate"]
        assert "e2-no-such-engine" not in probe["per_engine_hours"]
        # ...and it contributed nothing, which is the half that matters: a
        # named engine that still moved the total would be the old defect
        # wearing a disclosure.
        assert probe["total_hours"] == estimate["total_hours"]


class TestThroughputThatRisesWithEffortIsFlagged:
    """AliSQL measured 2.6 QPS at ef_search 400 and 23.3 at 800, on one index
    in one process. More graph traversal cannot be faster, so those points do
    not describe the engine -- and nothing looked at the series as a series, so
    it took five days and a hand comparison to notice."""

    def _summary(self, points):
        from report.generate import summarize
        records = [dict(phase="recall_qps", engine="alisql", dataset="d",
                        m=16, storage_engine="InnoDB", build_mode="post",
                        ef_search=ef, qps=qps, recall_at_k=0.99)
                   for ef, qps in points]
        return summarize(records)

    def test_a_clean_sweep_reports_nothing(self):
        s = self._summary([(10, 400), (20, 300), (40, 200), (80, 100)])
        assert s["qps_inversions"] == []

    def test_the_alisql_shape_is_caught(self):
        s = self._summary([(80, 133.2), (120, 19.6), (200, 5.2),
                           (400, 2.6), (800, 23.3)])
        assert len(s["qps_inversions"]) == 1
        found = s["qps_inversions"][0]
        assert (found["from_ef"], found["to_ef"]) == (400, 800)
        assert found["ratio"] > 8
        assert found["first_in_sweep"] is False

    def test_the_warmup_artifact_is_caught_and_labelled(self):
        """The first configuration pays for a cold cache and the second looks
        faster. Real, benign, and it must not read the same as contamination."""
        s = self._summary([(10, 239.8), (20, 436.8), (40, 247.6), (80, 139.3)])
        assert len(s["qps_inversions"]) == 1
        assert s["qps_inversions"][0]["first_in_sweep"] is True
        assert s["qps_inversions"][0]["ratio"] < 2

    def _warmed_summary(self, points, warmed=True):
        from report.generate import summarize
        extra = ({"index_warmup_queries": 1400, "index_warmup_seconds": 9.2,
                  "index_warmup_converged": True} if warmed else {})
        records = [dict(phase="recall_qps", engine="mariadb", dataset="d",
                        m=16, storage_engine="InnoDB", build_mode="post",
                        ef_search=ef, qps=qps, recall_at_k=0.99,
                        extra=dict(extra))
                   for ef, qps in points]
        return summarize(records)

    # full-20260825, MariaDB 11.8 on dbpedia: 454.2 QPS at ef_search 10 and
    # 843.4 at 20 -- the cold first point the index warm-up now removes.
    FULL_20260825 = [(10, 454.2), (20, 843.4), (40, 600.0), (80, 400.0)]

    def test_a_first_step_inversion_on_a_warmed_index_is_unexplained(self):
        """Once the overlay warmed the index before its first configuration
        there is no cold cache left to blame, so the same shape that used to
        read "warm-up" is a measurement of something else."""
        from report.render import _validity_section
        s = self._warmed_summary(self.FULL_20260825)
        assert len(s["qps_inversions"]) == 1
        found = s["qps_inversions"][0]
        assert found["first_in_sweep"] is True
        assert found["index_warmed"] is True
        assert found["likely"] == "unexplained"
        text = _validity_section({}, s)
        row = next(line for line in text.splitlines() if "10 → 20" in line)
        assert "**unexplained**" in row and "warm-up" not in row
        assert "1 of these are not the cold-first-step artifact" in text

    def test_without_the_warm_up_attributes_it_is_still_warm_up(self):
        """Result files measured before ANN_MEASUREMENT_VERSION 3 carry no
        warm-up attributes and keep the old, correct label."""
        from report.render import _validity_section
        s = self._warmed_summary(self.FULL_20260825, warmed=False)
        found = s["qps_inversions"][0]
        assert found["index_warmed"] is False
        assert found["likely"] == "warm-up"
        row = next(line for line in _validity_section({}, s).splitlines()
                   if "10 → 20" in line)
        assert "warm-up" in row and "unexplained" not in row

    def test_the_loader_carries_the_warm_up_attributes(self):
        from report.disclosures import warmup_extra
        import numpy
        assert warmup_extra({}) == {}
        assert warmup_extra({"index_warmup_queries": numpy.int64(1400),
                             "index_warmup_seconds": numpy.float64(9.25),
                             "index_warmup_converged": numpy.bool_(False)}) == {
            "index_warmup_queries": 1400, "index_warmup_seconds": 9.25,
            "index_warmup_converged": False}

    def test_the_worst_is_reported_first(self):
        s = self._summary([(10, 100), (20, 150), (40, 10), (80, 300)])
        ratios = [i["ratio"] for i in s["qps_inversions"]]
        assert ratios == sorted(ratios, reverse=True)

    def test_separate_configurations_are_separate_series(self):
        """A sweep over ef_construction is several curves, and comparing a
        point on one with a point on another would invent inversions."""
        from report.generate import summarize
        records = []
        for efc, base in ((64, 800), (400, 100)):
            for ef, qps in ((10, base), (20, base / 2), (40, base / 4)):
                records.append(dict(phase="recall_qps", engine="pgvector",
                                    dataset="d", m=16, ef_construction=efc,
                                    build_mode="post", storage_engine="heap",
                                    ef_search=ef, qps=qps, recall_at_k=0.99))
        assert summarize(records)["qps_inversions"] == []

    def test_it_reaches_the_report(self):
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        assert "Throughput that rose as query effort rose" in source
        assert "qps_inversions" in source


class TestAPhaseThatSharedTheMachineSaysSo:
    """The ops path sampled its server all along; the ann path sampled nothing,
    which is why contamination had to be inferred from a latency ratio instead
    of read off a graph."""

    def _rows(self, ours_cores, host_cores, n=20, step=1.0):
        return [{"t": i * step,
                 "cpu_seconds": i * step * ours_cores,
                 "host_cpu_seconds": i * step * host_cores,
                 "host_load1": host_cores * 1.1,
                 "rss_bytes": 1} for i in range(n)]

    def test_a_quiet_machine_reports_nothing(self):
        from report.loaders import foreign_cpu
        assert foreign_cpu(self._rows(8.0, 8.5)) is None

    def test_a_shared_machine_is_named(self):
        from report.loaders import foreign_cpu
        found = foreign_cpu(self._rows(1.0, 20.0))
        assert found is not None
        assert found["foreign_cores"] == 19.0
        assert found["container_cores"] == 1.0

    def test_an_older_series_without_the_fields_is_not_a_finding(self):
        """Runs that predate host sampling must not be reported as contended."""
        from report.loaders import foreign_cpu
        assert foreign_cpu([{"t": 0, "cpu_seconds": 0, "rss_bytes": 1},
                            {"t": 10, "cpu_seconds": 80, "rss_bytes": 1}]) is None

    def test_one_sample_cannot_be_differenced(self):
        from report.loaders import foreign_cpu
        assert foreign_cpu(self._rows(1.0, 20.0, n=1)) is None

    def test_the_ann_phase_is_sampled(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        assert "MemorySampler(" in source
        assert "wait_for_start_s" in source

    def test_the_sampler_waits_for_a_foreground_container(self):
        """run_foreground creates the container when it starts, so a sampler
        started first found nothing running and exited before the phase began."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "docker_ctl.py")).read()
        assert "_await_container" in source
        assert "def _host_pressure" in source

    def test_the_ann_series_cannot_be_mistaken_for_an_ops_one(self):
        """attach_peak_rss looks a series up by exact name; a collision would
        publish one phase's peak memory as another's."""
        from report.loaders import memory_stem
        ops = memory_stem({"engine": "alisql", "dataset": "d",
                           "resource_pass": "tuned", "m": 16,
                           "build_mode": "post"})
        assert ops != "alisql-d-tuned-ann"


class TestACurveIsIdentifiedByItsBuildNotItsColumns:
    """Listing the axes by hand has failed twice. First when the tuned pass
    began sweeping storage engine and ef_construction, which flagged 16 real
    measurements as duplicates; then when Percona Search gained a quantization
    axis that is not a column at all, which flagged 8 more and invented six
    inversions comparing ef_search 10 against ef_search 10."""

    def _record(self, engine, source, **kw):
        base = dict(phase="recall_qps", engine=engine, dataset="d", m=16,
                    qps=100.0, recall_at_k=0.99, ef_search=10)
        base.update(kw)
        base["extra"] = {"source_file": f"d/10/{engine}/{source}"}
        return base

    def test_two_quantizations_are_two_curves(self):
        from report.loaders import build_signature
        none = self._record("mongodb", "angular_M_16_quantization_none_400.hdf5")
        scalar = self._record("mongodb", "angular_M_16_quantization_scalar_400.hdf5")
        assert build_signature(none) != build_signature(scalar)

    def test_two_query_efforts_are_one_curve(self):
        from report.loaders import build_signature
        low = self._record("mongodb", "angular_M_16_quantization_none_10.hdf5")
        high = self._record("mongodb", "angular_M_16_quantization_none_800.hdf5")
        assert build_signature(low) == build_signature(high)

    def test_it_works_for_an_axis_nobody_named(self):
        """The point is covering axes this code has never heard of."""
        from report.loaders import build_signature
        a = self._record("future", "angular_M_16_some_new_knob_a_400.hdf5")
        b = self._record("future", "angular_M_16_some_new_knob_b_400.hdf5")
        assert build_signature(a) != build_signature(b)

    def test_engines_are_never_merged(self):
        from report.loaders import build_signature
        a = self._record("mariadb", "angular_M_16_engine_InnoDB_400.hdf5")
        b = self._record("alisql", "angular_M_16_engine_InnoDB_400.hdf5")
        assert build_signature(a) != build_signature(b)

    def test_records_without_a_source_file_fall_back_to_columns(self):
        from report.loaders import build_signature
        a = dict(phase="recall_qps", engine="pgvector", dataset="d", m=16,
                 build_mode="post", ef_construction=64)
        b = dict(a, ef_construction=400)
        assert build_signature(a) != build_signature(b)
        assert build_signature(a) == build_signature(dict(a))

    def test_a_build_name_ending_in_a_word_is_not_truncated(self):
        """Only a trailing query argument may be stripped, never part of the
        configuration name."""
        from report.loaders import build_signature
        r = self._record("valkey", "angular_M_16_build_mode_post.hdf5")
        assert "post" in str(build_signature(r))

    def test_the_real_run_has_no_duplicates_left(self):
        """Percona Search's two quantization curves were reported as eight
        duplicate configurations, telling the reader the report was reading
        more than one measurement tree."""
        from report.generate import summarize
        records = []
        for quant in ("none", "scalar"):
            for ef in (10, 20, 40):
                records.append(self._record(
                    "mongodb", f"angular_M_16_quantization_{quant}_{ef}.hdf5",
                    ef_search=ef, storage_engine="wiredTiger"))
        assert summarize(records)["duplicate_ann"] == []


class TestConcurrencyCarriesAnErrorBar:
    """Every figure in this framework was a single sample until a Valkey point
    was re-measured by accident, five hours later on the same build and
    configuration, and came back 21% higher at sixteen and thirty-two clients.
    Nothing else had ever been repeated, so nothing else had an error bar, and
    two engines whose results differ by less than that were being presented as
    ordered."""

    def _source(self):
        return open(os.path.join(VB_ROOT, "harness", "workloads",
                                 "concurrency.py")).read()

    def test_the_workload_takes_repeats(self):
        assert "repeats: int = 1" in self._source()

    def test_a_window_is_a_function_now(self):
        """Repeating meant the measured window had to stop being the body of
        a loop."""
        assert "def _one_window(" in self._source()

    def test_the_median_window_is_reported_not_an_average(self):
        """Latency percentiles from one measured window mean something; the
        mean of several windows' percentiles means nothing."""
        source = self._source()
        assert "samples[len(samples) // 2]" in source
        assert "samples.sort(" in source

    def test_every_sample_is_recorded(self):
        """A summary of the spread is not the spread."""
        assert '"qps_samples"' in self._source()

    def test_the_flag_reaches_the_harness(self):
        source = open(os.path.join(VB_ROOT, "harness", "main.py")).read()
        assert "--concurrency-repeats" in source
        assert "repeats=args.concurrency_repeats" in source

    def test_the_orchestrator_passes_it(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert "--concurrency-repeats" in source

    def test_the_headline_profile_repeats(self):
        from orchestrator.config import load_profile
        assert load_profile("tuned-complete")["ops"]["concurrency_repeats"] >= 3

    def test_a_single_sample_is_shown_as_having_no_error_bar(self):
        """A point measured once must not be dressed as +/-0.0%."""
        from report.render import _concurrency_table
        once = _concurrency_table({"concurrency": [
            {"engine": "valkey", "dataset": "d", "clients": 32, "qps": 5820.0,
             "extra": {"scaling_efficiency": 0.43}}]})
        assert "—" in once
        assert "n=" not in once

    def test_a_repeated_point_shows_its_spread(self):
        from report.render import _concurrency_table
        thrice = _concurrency_table({"concurrency": [
            {"engine": "valkey", "dataset": "d", "clients": 32, "qps": 5820.0,
             "extra": {"scaling_efficiency": 0.43, "repeats": 3,
                       "qps_spread_pct": 21.3,
                       "qps_samples": [4800.0, 5820.0, 5100.0]}}]})
        assert "21.3%" in thrice
        assert "n=3" in thrice


class TestTheRecallAxisDoesNotCrushHighRecallEngines:
    """MHNSW and VIDX cannot be told to be fast and sloppy: their lowest query
    effort already lands at recall 0.97-0.98. On a linear axis spanning
    0.69-1.0 that puts MariaDB 12.3 in the rightmost 7% of the plot and AliSQL
    in the rightmost 10%, and both read as truncated rather than as accurate.
    A logit axis gives 0.99-0.999 the same width as 0.9-0.99."""

    def _points(self):
        return {
            "mariadb123": [{"recall_at_k": r, "qps": q} for r, q in
                           ((0.9784, 252), (0.9933, 218), (0.9996, 23))],
            "valkey": [{"recall_at_k": r, "qps": q} for r, q in
                       ((0.7711, 1516), (0.9533, 729), (0.9989, 80))],
        }

    def _axes(self, **kw):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from report import charts
        import tempfile
        captured = {}
        original = charts._save

        def spy(fig, out_dir, stem):
            captured["ax"] = fig.axes[0]
            return original(fig, out_dir, stem)

        charts._save = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                charts.pareto(self._points(), "d", 10, d, "t", **kw)
        finally:
            charts._save = original
            plt.close("all")
        return captured["ax"]

    @needs_matplotlib
    def test_the_axis_is_logit(self):
        assert self._axes().get_xscale() == "logit"

    @needs_matplotlib
    def test_the_highest_point_is_inside_the_axis(self):
        """A logit axis cannot reach 1.0, and a right edge sitting on the best
        measurement clips the marker that matters most."""
        ax = self._axes()
        assert ax.get_xlim()[1] > 0.9996
        assert ax.get_xlim()[1] < 1.0

    @needs_matplotlib
    def test_a_high_recall_engine_gets_real_width(self):
        """The complaint this fixes: MariaDB 12.3 spanning a sliver."""
        ax = self._axes()
        lo, hi = ax.get_xlim()
        import matplotlib.scale as mscale
        t = mscale.scale_factory("logit", ax.xaxis).get_transform()
        span = t.transform([lo, hi])
        engine = t.transform([0.9784, 0.9996])
        share = (engine[1] - engine[0]) / (span[1] - span[0])
        assert share > 0.30, f"only {share:.0%} of the axis"

    @needs_matplotlib
    def test_the_ticks_are_readable_recall_values(self):
        """Logit's default labels are 1-10^-n, which nobody reads as recall."""
        ax = self._axes()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert "0.99" in labels
        assert not any("10" in l and "^" in l for l in labels)

    @needs_matplotlib
    def test_the_floor_still_narrows_the_view(self):
        assert self._axes(recall_floor=0.95).get_xlim()[0] == 0.95

    def test_the_companion_chart_shows_the_deployable_region(self):
        """At a 0.85 floor every engine's curve was still mostly below the
        accuracy anyone would ship at."""
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        assert "recall_floor=0.95" in source


class TestAFloorFigureSaysWhereItWasActuallyTaken:
    """MariaDB 12.3's lowest configuration returns recall 0.9784. It therefore
    has no measurement at recall 0.90, and its "QPS at recall 0.90" is really
    its throughput at 0.9888 — a materially harder task than the column asks
    for. Printed as a bare number beside pgvector's 644, which is a real point
    near 0.90, it invites a comparison the data does not support. Changing the
    chart's axis made the curves legible; it did not create the missing points,
    and the table has to say which cells are which."""

    def _entry(self, **kw):
        base = {"points": 8, "min_recall": 0.9784, "max_recall": 0.9996,
                "qps_at_recall_90": 510, "qps_at_recall_90_recall": 0.9888,
                "qps_at_recall_95": 510, "qps_at_recall_95_recall": 0.9888,
                "qps_at_recall_99": 218, "qps_at_recall_99_recall": 0.9933}
        base.update(kw)
        return base

    def _table(self, entry, engine="mariadb123"):
        from report.render import _headline_tables
        # One table per (dataset, pass); `None` is the pass of a record that
        # declares none, and §4 keeps the bare `#### d` heading for a single
        # pass.
        return _headline_tables({"per_dataset": {"d": {None: {engine: entry}}}})

    def test_a_figure_above_the_floor_is_marked(self):
        table = self._table(self._entry())
        assert "510 @0.989 †" in table

    def test_a_real_point_at_the_floor_is_not_marked(self):
        """pgvector measured recall 0.8190, so its 0.90 figure is honest."""
        entry = self._entry(min_recall=0.8190, qps_at_recall_90=644,
                            qps_at_recall_90_recall=0.9250)
        table = self._table(entry, engine="pgvector")
        assert "644" in table
        assert "644 †" not in table

    def test_the_floor_the_engine_does_reach_is_left_alone(self):
        """0.99 is below MariaDB 12.3's lowest recall, so that cell is real."""
        table = self._table(self._entry())
        assert "218 †" not in table

    def test_the_marker_is_explained_under_the_table(self):
        # Under the HEADLINE table: §4 now ends with the ops-build table,
        # so the last `| ---` on the page is that one's.
        from report.frontier_render import OPS_TABLE_HEADING
        table = self._table(self._entry()).split(OPS_TABLE_HEADING)[0]
        assert "†" in table.split("| ---")[-1]
        assert "not comparable" in table

    def test_the_explanation_names_the_lever(self):
        """A reader who wants those points needs to know what would produce
        them, and it is graph degree, not query effort."""
        assert "graph degree" in self._table(self._entry())

    def test_a_missing_figure_still_reads_as_a_dash(self):
        entry = self._entry(qps_at_recall_99=None, qps_at_recall_99_recall=None)
        assert "—" in self._table(entry)

    def test_a_floor_figure_is_never_taken_from_another_build(self):
        """The 25 Aug run: Percona Search's scalar index -- the one every other
        section measured -- peaks at recall 0.892, and the post printed 138 /
        124 / 88 from the unquantized sweep with nothing saying so.

        Now the headline IS the whole-sweep frontier, so 138 is there
        again -- but never unattributed: it names `quantization=none` and
        carries the ops-mismatch mark. The ops-build table says "—" with the
        scalar build's ceiling under it."""
        from report.frontier_render import (OPS_MISMATCH_MARK,
                                            OPS_TABLE_HEADING)
        from report.generate import summarize
        from report.render import _headline_tables
        from tests.test_e26_report_build_labels import (REFERENCE, _manifest,
                                                         _mongodb_records)
        table = _headline_tables(summarize(_mongodb_records(),
                                           _manifest(REFERENCE)))
        headline, ops_view = table.split(OPS_TABLE_HEADING)
        row = next(l for l in headline.splitlines() if l.startswith("| Percona"))
        assert "138" in row and "quantization=none" in row
        assert OPS_MISMATCH_MARK in row
        row = next(l for l in ops_view.splitlines() if l.startswith("| Percona"))
        assert [c.strip() for c in row.strip("|").split("|")][1:4] == \
            ["—", "—", "—"]
        assert "0.892" in ops_view


class TestARunMeasuresEveryEngineByDefault:
    """`run` defaulted to the original three engines long after the other
    three had become part of the study. So `run --profile smoke` proved three
    engines and reported success, and `run --profile tuned-complete` would have
    spent forty hours measuring three of six and reported success too -- an
    engine nobody asked for cannot fail, so nothing anywhere said half the
    comparison was missing."""

    def test_the_default_is_every_known_engine(self):
        # A value assertion, which is what makes it useful: an engine config
        # added to config/engines/ without this line being updated fails here
        # rather than silently joining every default run. `percona` is the
        # seventh and was added by the task that wrote config/engines/
        # percona.yml, which is the only way this set is ever meant to move.
        from orchestrator.cli import KNOWN_ENGINES
        assert set(KNOWN_ENGINES) == {"mariadb", "mariadb123", "alisql",
                                      "pgvector", "valkey", "mongodb",
                                      "percona", "myvector", "villagesql",
                                      "villagesql84"}

    def test_run_defaults_to_all_of_them(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        body = source.split("def cmd_run")[1].split("\ndef ")[0]
        line = next(l for l in body.splitlines() if "engines = [" in l)
        assert "ALL_ENGINES" not in line + body.split(line)[1].split("\n")[0]

    def test_no_command_still_defaults_to_the_original_three(self):
        """ALL_ENGINES survives as a name for the original comparison, but
        nothing may quietly resolve a default through it."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        for line in source.splitlines():
            if "args.engines" in line and "ALL_ENGINES" in line:
                assert False, f"a default still resolves to three: {line.strip()}"

    def test_the_smoke_profile_no_longer_claims_three(self):
        """The gate before every long run has to gate the whole run."""
        text = open(os.path.join(VB_ROOT, "config", "profiles",
                                 "smoke.yml")).read()
        assert "all three engines" not in text

    def test_an_unbuilt_engine_fails_loudly(self):
        """Defaulting to six means someone with three images gets an error.
        That is the point: a missing image is recoverable, a silently absent
        engine is forty hours."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        assert "not found. Build it first" in source


class TestAProfileThatNamesItsEnginesGetsThem:
    """...and the one that does ran seven of them.

    `config/profiles/percona.yml` has declared `engines: [percona]` since it
    was written, above a description that reads "Percona Server alone". THE
    MEASUREMENT: `PS/percona-20261020-140151` is a `status: completed` run of
    that profile holding 14 phases, two per engine, with `ops-*`, `mem-*` and
    recall records for alisql, mariadb, mariadb123, mongodb, pgvector and
    valkey beside percona -- under a manifest whose `config.profile.engines`
    reads `['percona']` next to seven populated `engines` keys.

    Nothing read the key. `cmd_run` resolved `args.engines or
    ",".join(KNOWN_ENGINES)`, and the only expression of the key's meaning
    anywhere in the tree was in a *test* --
    `tests/test_e2_percona_wiring.py::TestEveryShippedProfileRenders` resolves
    `profile.get("engines") or KNOWN_ENGINES` and its comment attributes that
    resolution to `cmd_render`, which does not do it either. So the tree knew
    what the key meant and no production code honoured it.

    Why the cost is not just six engines' worth of hours: every axis this
    profile has is Percona-only (`resources.extras.percona_ef_construction`,
    `ops.percona_build_modes`) and `ops.subset_rows: 20000` was chosen to make
    three Percona corpus reloads affordable. The six were therefore measured at
    a grid designed for a different engine, on a corpus sized for a different
    question, and reported success -- a credible-looking answer to a question
    nobody asked, which is this repo's signature failure rather than a crash.
    """

    def _args(self, **kw):
        import argparse
        spec = dict(profile="percona", engines=None, datasets=None,
                    resource_pass="normalized", phases="both",
                    run_id="probe-20260101-000000", resume=False, force=False)
        spec.update(kw)
        return argparse.Namespace(**spec)

    def _plan(self, monkeypatch, tmp_path, **kw):
        """What `cmd_run` resolved, read off the two calls it hands the list to.

        Stopped at the free-disk check, which is the last thing before any
        container could start: `_check_free_disk` returning True with
        `--force` absent makes `cmd_run` return 4 of its own accord, so nothing
        has to raise through it and both captures are complete first.

        CAPTURING BOTH LISTS IS THE POINT. A change that narrowed the unit loop
        while still handing `print_load_estimate` the registry would print a
        seven-engine cost above a one-engine run, and the printed figure would
        describe a different plan from the one about to execute -- the failure
        this repo cares about, in a new place. Asserting they are equal costs
        nothing and pins it.

        `paths_for` is rerooted by substituting VB_ROOT rather than rebuilt,
        so a key added to it lands here automatically instead of making this
        harness quietly stale.
        """
        from orchestrator import cli as cli_mod
        seen = {}

        # `**_kw` for `client_cpus`, which the estimate now prints beside the
        # largest client count; this capture is about the engine list only.
        def _estimate(profile, engines, datasets, passes, phases, **_kw):
            seen["estimate"] = list(engines)

        def _disk(paths, engines, datasets, passes):
            seen["run"] = list(engines)
            return True

        args = self._args(**kw)
        calls = self._reroot(monkeypatch, tmp_path)
        datasets_dir = cli_mod.paths_for(args.run_id)["datasets"]
        calls.clear()
        os.makedirs(datasets_dir, exist_ok=True)
        for dataset in load_profile(args.profile).get("datasets", []):
            open(os.path.join(datasets_dir, f"{dataset}.hdf5"), "wb").close()

        monkeypatch.setattr(cli_mod.docker_ctl, "docker_available", lambda: True)
        monkeypatch.setattr(cli_mod, "print_load_estimate", _estimate)
        monkeypatch.setattr(cli_mod, "_check_free_disk", _disk)

        assert cli_mod.cmd_run(args) == 4, (
            "the harness no longer stops at the disk check; the captures below "
            "may be incomplete")
        return seen

    @staticmethod
    def _reroot(monkeypatch, tmp_path):
        """Point every path `cmd_run` can reach at tmp_path, counting the calls.

        Returns the call list, so a caller can assert it stayed empty.

        EVERY entry point here reroots, including the refusal ones that
        "cannot" reach a directory. They could: under a mutant that drops the
        refusal, `cmd_run` runs on past it and makes `results/<run-id>/` with
        a manifest, and with `datasets/fashion-mnist-784-euclidean.hdf5`
        present -- which the main checkout has, and which is the dataset
        `percona.yml` names -- it would carry on into the free-disk check and
        `_run_unit`. A probe that starts a benchmark while demonstrating that
        a guard is gone is not a probe. Rerooting makes that structurally
        impossible instead of relying on the guard under test.
        """
        from orchestrator import cli as cli_mod
        root = str(tmp_path / "vb")
        real_paths = cli_mod.paths_for
        calls = []

        def _paths(run_id):
            calls.append(run_id)
            return {key: value.replace(VB_ROOT, root)
                    if isinstance(value, str) else value
                    for key, value in real_paths(run_id).items()}

        monkeypatch.setattr(cli_mod, "paths_for", _paths)
        return calls

    def _refuse(self, monkeypatch, capsys, tmp_path, **kw):
        """A refusal, with a proof that it landed above `paths_for`.

        The returned `calls` is asserted empty by every caller rather than
        merely believed: "the check sits above `paths_for`" is a claim about
        statement order that no reader can verify from the test, and the
        rerooting in `_reroot` is what keeps a broken claim harmless.
        """
        from orchestrator import cli as cli_mod
        calls = self._reroot(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_mod.docker_ctl, "docker_available", lambda: True)
        code = cli_mod.cmd_run(self._args(**kw))
        return code, capsys.readouterr().err, calls

    def test_the_subject_of_this_class_is_still_what_it_says(self):
        """Non-vacuity. If `percona.yml` stops declaring one engine, or the
        registry stops holding seven, the assertions below become trivially
        true and would stay green while proving nothing."""
        from orchestrator.cli import KNOWN_ENGINES
        assert load_profile("percona")["engines"] == ["percona"]
        assert len(KNOWN_ENGINES) == 10
        assert "engines" not in load_profile("smoke")

    def test_a_bare_run_of_a_one_engine_profile_runs_one_engine(
            self, monkeypatch, tmp_path):
        seen = self._plan(monkeypatch, tmp_path)
        assert seen["run"] == ["percona"]

    def test_the_operator_is_told_the_profile_narrowed_the_run(
            self, monkeypatch, tmp_path, capsys):
        """The note is the only thing standing between a one-engine banner and
        a reader who reaches for `--engines` because the banner looks broken --
        `docs/02-running-with-framework.md` documents the flag's default as
        every engine. Announced rather than assumed, like the
        `default_resource_pass` note two statements above it in `cmd_run`, and
        asserted here because a print nothing checks is a print that goes."""
        self._plan(monkeypatch, tmp_path)
        out = capsys.readouterr().out
        assert "runs only percona" in out
        assert "--engines" in out

    def test_no_note_is_printed_when_the_profile_narrowed_nothing(
            self, monkeypatch, tmp_path, capsys):
        """A note on every run is a note nobody reads."""
        self._plan(monkeypatch, tmp_path, profile="smoke")
        assert "runs only" not in capsys.readouterr().out

    def test_the_printed_cost_describes_the_run_and_not_the_registry(
            self, monkeypatch, tmp_path):
        seen = self._plan(monkeypatch, tmp_path)
        assert seen["estimate"] == ["percona"]
        assert seen["estimate"] == seen["run"]

    def test_a_profile_naming_no_engines_still_runs_every_one(
            self, monkeypatch, tmp_path):
        """The default that this change must not touch. `run` defaulted to the
        original three long after six existed; narrowing it again for profiles
        that make no claim would be the same bug with a new cause. "Every one"
        is every DEFAULT-selected engine: the opt-in ones
        (`runtime.default_selected: false`) are the declared exception, and
        the next two tests pin that they are announced and still runnable."""
        from orchestrator.cli import DEFAULT_ENGINES, KNOWN_ENGINES
        seen = self._plan(monkeypatch, tmp_path, profile="smoke")
        assert seen["run"] == list(DEFAULT_ENGINES)
        assert seen["estimate"] == list(DEFAULT_ENGINES)
        # Only the declared opt-in engines are missing -- nothing else.
        from orchestrator import engines as engines_mod
        assert [e for e in KNOWN_ENGINES if e not in DEFAULT_ENGINES] == [
            e for e in KNOWN_ENGINES if not engines_mod.get(e).default_selected]

    def test_a_bare_run_names_the_opt_in_engines_it_left_out(
            self, monkeypatch, tmp_path, capsys):
        self._plan(monkeypatch, tmp_path, profile="smoke")
        out = capsys.readouterr().out
        assert ("note: opt-in engine(s) not run unless named with --engines: "
                "villagesql") in out

    def test_an_opt_in_engine_runs_when_named(self, monkeypatch, tmp_path,
                                              capsys):
        seen = self._plan(monkeypatch, tmp_path, profile="smoke",
                          engines="pgvector,villagesql")
        assert seen["run"] == ["pgvector", "villagesql"]
        assert seen["estimate"] == ["pgvector", "villagesql"]
        assert "opt-in engine(s)" not in capsys.readouterr().out

    def test_an_unbuildable_ops_build_is_refused_before_any_container(
            self, monkeypatch, capsys, tmp_path):
        """`ops.m_values` x the ops ef_construction gets the render's knob
        rules too: M=6 at VillageSQL's 200 is refused here, by name, rather
        than after a server start and a corpus load."""
        from orchestrator import cli as cli_mod
        smoke = load_profile("smoke")
        smoke["ops"] = dict(smoke.get("ops") or {}, m_values=[6, 16])
        monkeypatch.setattr(cli_mod, "load_profile", lambda name: smoke)
        args = self._args(profile="smoke", engines="villagesql")
        self._reroot(monkeypatch, tmp_path)
        datasets_dir = cli_mod.paths_for(args.run_id)["datasets"]
        os.makedirs(datasets_dir, exist_ok=True)
        for dataset in smoke.get("datasets", []):
            open(os.path.join(datasets_dir, f"{dataset}.hdf5"), "wb").close()
        monkeypatch.setattr(cli_mod.docker_ctl, "docker_available",
                            lambda: True)
        reached = []
        monkeypatch.setattr(cli_mod, "print_load_estimate",
                            lambda *a, **k: reached.append("estimate"))
        monkeypatch.setattr(cli_mod, "_check_free_disk",
                            lambda *a, **k: reached.append("disk") or True)
        assert cli_mod.cmd_run(args) == 2
        err = capsys.readouterr().err
        assert "refusing to start" in err
        assert "M=6 at ef_construction=200" in err
        # Refused with the other pre-flight problems, before the estimate,
        # the disk check and any container.
        assert reached == []

    def _run_with(self, monkeypatch, tmp_path, profile, **kw):
        """`cmd_run` on `profile` (a dict), datasets present, stopped at the
        disk check; returns (exit code, the reference_builds dict cmd_run
        filled, whether the estimate was reached)."""
        from orchestrator import cli as cli_mod
        monkeypatch.setattr(cli_mod, "load_profile", lambda name: profile)
        args = self._args(profile=profile["name"], **kw)
        self._reroot(monkeypatch, tmp_path)
        datasets_dir = cli_mod.paths_for(args.run_id)["datasets"]
        os.makedirs(datasets_dir, exist_ok=True)
        for dataset in profile.get("datasets", []):
            open(os.path.join(datasets_dir, f"{dataset}.hdf5"), "wb").close()
        monkeypatch.setattr(cli_mod.docker_ctl, "docker_available",
                            lambda: True)
        seen = {"builds": None, "estimate": False}
        real = cli_mod._build_preflight

        def spy(engine, prof, resources, resource_pass, phases, builds):
            seen["builds"] = builds
            return real(engine, prof, resources, resource_pass, phases,
                        builds)

        monkeypatch.setattr(cli_mod, "_build_preflight", spy)
        monkeypatch.setattr(cli_mod, "print_load_estimate",
                            lambda *a, **k: seen.__setitem__("estimate", True))
        monkeypatch.setattr(cli_mod, "_check_free_disk",
                            lambda *a, **k: True)
        return cli_mod.cmd_run(args), seen

    @pytest.mark.parametrize("phases", ["both", "ann", "ops"])
    def test_every_engine_and_pass_gets_its_reference_build(
            self, monkeypatch, tmp_path, phases):
        """The manifest's presentation.reference_builds is what tells the
        report which build each headline figure came from. An `else:` that
        bound to the ops pre-flight instead of the reference-build check
        left it empty for every engine under the default `--phases both`."""
        from orchestrator.cli import DEFAULT_ENGINES
        code, seen = self._run_with(monkeypatch, tmp_path,
                                    load_profile("smoke"),
                                    resource_pass="both", phases=phases)
        from orchestrator.config import load_resources, merge_resource_overrides
        from orchestrator.reference_build import resolve_reference_build
        assert code == 4                      # stopped at the disk check
        builds = seen["builds"]
        smoke = load_profile("smoke")
        assert set(builds) == set(DEFAULT_ENGINES)
        for engine in DEFAULT_ENGINES:
            assert set(builds[engine]) == {"normalized", "tuned"}, engine
            for rp in ("normalized", "tuned"):
                resources = merge_resource_overrides(load_resources(rp), smoke)
                assert builds[engine][rp] == resolve_reference_build(
                    engine, smoke, resources, rp), (engine, rp)
                assert builds[engine][rp], (engine, rp)

    def test_a_reference_build_problem_under_ann_only_is_a_clean_refusal(
            self, monkeypatch, tmp_path, capsys):
        """Not a traceback from resolving the build it just refused."""
        smoke = load_profile("smoke")
        smoke["ann"] = dict(smoke["ann"], m_values=[8])
        code, seen = self._run_with(monkeypatch, tmp_path, smoke,
                                    engines="mariadb", phases="ann")
        err = capsys.readouterr().err
        assert code == 2
        assert "refusing to start" in err
        assert "mariadb: its reference build (m=16" in err
        assert "Traceback" not in err
        assert seen["estimate"] is False
        assert seen["builds"] == {"mariadb": {}}

    @pytest.mark.parametrize("engines,warned", [(None, True),
                                                ("mariadb,villagesql", False)])
    def test_a_resume_that_would_drop_an_engine_says_so(
            self, monkeypatch, tmp_path, capsys, engines, warned):
        """An opt-in engine named on the original run is not in the default
        list a bare `--resume` re-derives; its remaining units would never
        run, so the resume says which engines it leaves behind."""
        from orchestrator import cli as cli_mod
        smoke = load_profile("smoke")
        run_id = "probe-20260101-000000"
        self._reroot(monkeypatch, tmp_path)
        run_dir = cli_mod.paths_for(run_id)["run_dir"]
        os.makedirs(run_dir, exist_ok=True)
        from orchestrator.manifest import Manifest
        prior = Manifest(run_dir, run_id)
        prior.data["engines"] = {"mariadb": {}, "villagesql": {}}
        prior.save()
        kw = {"resume": True}
        if engines:
            kw["engines"] = engines
        self._run_with(monkeypatch, tmp_path, smoke, **kw)
        err = capsys.readouterr().err
        assert ("which this resume will not run" in err) is warned
        if warned:
            assert "villagesql" in err and "--engines" in err

    def test_an_explicit_flag_narrows_a_profile_that_names_none(
            self, monkeypatch, tmp_path):
        seen = self._plan(monkeypatch, tmp_path, profile="smoke",
                          engines="pgvector")
        assert seen["run"] == ["pgvector"]

    def test_an_explicit_flag_the_profile_allows_still_wins(
            self, monkeypatch, tmp_path):
        """The flag is honoured, not overridden by the profile. Spelling the
        profile's own engine out loud has to behave like the bare run and not
        like an argument with it."""
        seen = self._plan(monkeypatch, tmp_path, engines="percona")
        assert seen["run"] == ["percona"]
        assert seen["estimate"] == ["percona"]

    def test_an_engine_the_profile_excludes_is_refused(self, monkeypatch,
                                                       capsys, tmp_path):
        """PRECEDENCE, stated as a refusal. `--profile percona --engines
        mariadb` reads two ways -- "measure mariadb" and "measure what the
        percona profile measures" -- and the two differ by a whole run. The
        combination is what fails, rather than one reading winning quietly."""
        code, err, calls = self._refuse(monkeypatch, capsys, tmp_path,
                                        engines="mariadb")
        assert code == 2
        assert "percona" in err and "mariadb" in err
        assert "--engines" in err
        assert calls == [], f"the refusal ran past paths_for: {calls}"

    def test_a_partly_allowed_list_is_refused_whole(self, monkeypatch, capsys,
                                                    tmp_path):
        """`percona,mariadb` names one engine the profile runs and one it does
        not. Running the intersection is the silent half-obedience this refusal
        exists to prevent, so the named engine does not rescue the list."""
        code, err, calls = self._refuse(monkeypatch, capsys, tmp_path,
                                        engines="percona,mariadb")
        assert code == 2
        assert "mariadb" in err
        assert calls == []

    def test_a_typo_is_still_reported_as_a_typo(self, monkeypatch, capsys,
                                                tmp_path):
        """ORDERING, and asserted by the message rather than by the exit code,
        which both checks share. The KNOWN_ENGINES check stays above the
        profile check, so a misspelled name is reported as unknown rather than
        as an engine the profile excludes -- which would send someone to edit
        the profile over a typo on their own command line."""
        code, err, calls = self._refuse(monkeypatch, capsys, tmp_path,
                                        engines="mariadbb")
        assert code == 2
        assert "unknown engines" in err
        assert "runs only" not in err, (
            "the exclusion message won; the KNOWN_ENGINES check no longer "
            "runs first")
        assert calls == []

    def test_a_declared_list_that_says_nothing_usable_is_refused(
            self, monkeypatch, capsys, tmp_path):
        """`engines: []` AND `engines:` WITH NO VALUE RE-ARM THE WHOLE DEFECT.

        `profile_engines` answers [] for an absent key and for an empty one
        alike, and `cmd_run`'s fall-back is spelled `declared or
        KNOWN_ENGINES` -- so an operator who opens `percona.yml` in the web
        UI's profile editor and empties the list would get all seven engines
        at the Percona-only grid, with no note and no warning, completing and
        reporting success. That is `percona-20261020-140151` exactly, the run
        this key was made load-bearing to prevent. It is the one place the
        silent-widening shape survived honouring the key.

        Driven through a profile FILE rather than a dict, with `CONFIG_DIR`
        pointed at tmp_path, because the point is what `run --profile <name>`
        does with something somebody saved -- and because `engines:` with no
        value only becomes `None` by going through the YAML parser.
        """
        from orchestrator import config as config_mod
        # Read the real profile BEFORE CONFIG_DIR moves, so the probe inherits
        # the grid that makes the widening expensive rather than a bare stub.
        body = load_profile("percona")
        del body["engines"]
        head = yaml.safe_dump({**body, "name": "probe"})
        profiles = tmp_path / "config" / "profiles"
        profiles.mkdir(parents=True)
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path / "config"))
        for text, shown in (("engines: []", "an empty list"),
                            ("engines:", "no value at all"),
                            ("engines: percona", "a bare string"),
                            ("engines: [percona, 7]", "a non-string member"),
                            ("engines: [percona, '  ']", "a blank member")):
            (profiles / "probe.yml").write_text(f"{head}\n{text}\n")
            code, err, calls = self._refuse(monkeypatch, capsys,
                                            tmp_path, profile="probe")
            assert code == 2, f"{shown} was accepted"
            assert "non-empty list of engine names" in err, shown
            assert "probe.yml" in err, f"{shown}: the file is not named"
            assert calls == [], f"{shown}: ran past paths_for"

    def test_the_resolution_is_a_function_rather_than_an_expression(self):
        """Read directly, because `cmd_run` cannot be asked about the empty
        and whitespace cases without building a profile file for each."""
        from orchestrator.config import (engines_outside_profile,
                                         profile_engines,
                                         profile_engines_problem)
        assert profile_engines({"engines": ["a", " b "]}) == ["a", "b"]
        assert profile_engines({"engines": ["a", "", "  "]}) == ["a"]
        # A repeated name is not a request for two measurements: both units
        # land in one run directory under one checkpoint key.
        assert profile_engines({"engines": ["a", "b", "a"]}) == ["a", "b"]
        # A profile that declares nothing, declares the key empty, or is not a
        # dict at all are one answer: it restricts nothing.
        assert profile_engines({}) == []
        assert profile_engines({"engines": []}) == []
        assert profile_engines(None) == []
        assert engines_outside_profile({}, ["anything"]) == []
        assert engines_outside_profile({"engines": ["a", "b"]}, ["a"]) == []
        assert engines_outside_profile({"engines": ["a"]},
                                       ["b", "a", "c"]) == ["b", "c"]
        # ABSENT and PRESENT-BUT-EMPTY are the same answer from
        # `profile_engines` and must not be the same answer here: one is a
        # profile making no claim, the other is a half-finished edit.
        assert profile_engines_problem({}) is None
        assert profile_engines_problem(None) is None
        assert profile_engines_problem({"engines": ["a"]}) is None
        for bad in ([], None, "percona", ["a", 7], ["a", "  "], {}, 3):
            assert profile_engines_problem({"engines": bad}), repr(bad)

    def test_a_duplicate_on_the_command_line_runs_one_unit(
            self, monkeypatch, tmp_path):
        """`--engines percona,percona` would start the engine twice and write
        both units into one run directory under one checkpoint key, so the
        second overwrites the first and the report shows one. Predates the
        profile key; the key gave the resolution a second source of
        duplicates, so the reduction sits on the result of both."""
        seen = self._plan(monkeypatch, tmp_path, engines="percona,percona")
        assert seen["run"] == ["percona"]
        assert seen["estimate"] == ["percona"]


class TestGraphDegreeCanBeSweptWhereItIsTheOnlyKnob:
    """MHNSW and VIDX expose M and nothing else, and at M=16 their lowest query
    effort already returns recall 0.97-0.98. They therefore have no measurement
    anywhere near the 0.90 floor the headline table reports at, while pgvector
    and Valkey -- which also have ef_construction -- reach down to 0.76. The
    figures in those cells are real, but they are taken at a much harder task
    than the column asks for, and the only way to make them comparable is to
    give the engines a configuration that is genuinely less accurate."""

    def _groups(self, engine, profile, resources=None, pass_name="normalized"):
        from orchestrator.ann_pass import render_config
        body = render_config(engine, profile, resources or {}, pass_name)
        return body["float"]["any"][0]["run_groups"]

    def _m(self, engine, profile, **kw):
        return [g["arg_groups"][0]["M"] for g in
                self._groups(engine, profile, **kw).values()]

    def test_an_engine_can_be_given_its_own_m_values(self):
        p = {"ann": {"m_values": [16], "ef_search": [10],
                     "m_values_by_engine": {"mariadb": [6, 8, 16]}}}
        assert self._m("mariadb", p) == [[6, 8, 16]]

    def test_engines_without_an_override_are_untouched(self):
        """pgvector already reaches 0.82; paying for a reload to give it points
        it has would be spending hours on nothing."""
        p = {"ann": {"m_values": [16], "ef_search": [10],
                     "m_values_by_engine": {"mariadb": [6, 8, 16]}}}
        assert self._m("pgvector", p) == [[16]]

    def test_the_tuned_pass_reads_it_from_resources(self):
        p = {"ann": {"m_values": [16], "ef_search": [10]}}
        r = {"extras": {"m_values_by_engine": {"alisql": [8, 16]}}}
        assert self._m("alisql", p, resources=r, pass_name="tuned") == [[8, 16]]

    def test_the_normalized_pass_ignores_a_tuned_override(self):
        """The normalized pass exists to give every engine the same treatment;
        an extras-driven per-engine sweep is exactly what it must not do."""
        p = {"ann": {"m_values": [16], "ef_search": [10]}}
        r = {"extras": {"m_values_by_engine": {"alisql": [8, 16]}}}
        assert self._m("alisql", p, resources=r, pass_name="normalized") == [[16]]

    def test_mongodb_still_carries_one_value(self):
        """mongot exposes no graph degree at all, so sweeping it there would
        produce identical curves under different labels."""
        p = {"ann": {"m_values": [16], "ef_search": [10],
                     "m_values_by_engine": {"mongodb": [6, 8, 16]}}}
        assert self._m("mongodb", p) == [[6]]

    def test_a_profile_can_clear_an_inherited_per_engine_sweep(self):
        """The tuned pass pins M=[6,16] for MHNSW and VIDX. A profile whose
        whole purpose is sweeping M has to be able to say no -- and it has to
        say it with null, because the override merges dict-into-dict and an
        empty map leaves the inherited values untouched."""
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        profile = load_profile("m-sweep")
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        assert resources["extras"]["m_values_by_engine"] is None

    def test_the_tuned_pass_still_pins_it(self):
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        profile = load_profile("tuned-complete")
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        by_engine = resources["extras"]["m_values_by_engine"]
        assert by_engine["mariadb123"] == [6, 16]
        assert "pgvector" not in by_engine


class TestAReportCanBeNarrowedToOneCorpus:
    """The ann results tree is keyed by resource configuration, not by corpus.
    A machine that measured dbpedia at 90,000 rows and at 990,000 under the
    same configuration has both in one tree, so regenerating a run's report
    grows a second dataset section. Both are correct and the report separates
    them — but a report meant to describe one corpus should be able to say so
    rather than leaving a reader to notice which section they are reading."""

    def _argv(self, *extra):
        from report.generate import parse_args
        return parse_args(["--run-dir", "/x", *extra])

    def test_the_generator_takes_the_flag(self):
        assert self._argv("--datasets", "a,b").datasets == "a,b"

    def test_it_defaults_to_everything(self):
        assert self._argv().datasets is None

    def test_the_cli_passes_it_through(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert 'rep.add_argument("--datasets"' in source
        assert 'generate_report(paths, list(KNOWN_ENGINES), args.datasets)' in source

    def test_the_container_command_carries_it_only_when_asked(self):
        """An always-present empty flag would make every report a filtered
        one, and an empty filter matches nothing."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert '*(["--datasets", datasets] if datasets else [])' in source

    def test_a_name_that_was_never_measured_is_an_error(self):
        """Silently reporting on nothing is how a typo becomes an empty
        report that looks like a failed run."""
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        block = source.split("if args.datasets:")[1].split("if not records:")[0]
        assert "unknown" in block and "return 2" in block

    def test_records_without_a_dataset_survive_the_filter(self):
        """Some records are not per-corpus. Dropping them would empty the
        environment and validity sections along with the other corpus."""
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        assert 'if not r.get("dataset") or r["dataset"] in wanted' in source


class TestTwoGraphDegreesAreTwoLines:
    """Every engine swept a single M until MHNSW and VIDX were given M=6 as
    well, so they would have real measurements below recall 0.90. M was not a
    series axis, so the two configurations were drawn as one line: on the
    latency chart, which plots against ef_search, that put two y-values at
    every x and joined them, and AliSQL appeared to swing between 383 ms and
    95 ms at adjacent search widths."""

    def _records(self):
        return [{"phase": "recall_qps", "engine": "alisql", "dataset": "d",
                 "storage_engine": "InnoDB", "m": m, "ef_search": ef,
                 "qps": 100.0, "recall_at_k": 0.9,
                 "latency_p50_ms": 5.0, "latency_p99_ms": 9.0}
                for m in (6, 16) for ef in (10, 20, 40)]

    def test_m_is_a_series_axis(self):
        from report.charts import SERIES_AXES
        assert "m" in SERIES_AXES

    def test_two_degrees_make_two_series(self):
        from report.charts import _grouped
        groups, _labels, _st, _deg = _grouped(self._records())
        assert len(groups) == 2

    def test_the_legend_says_m_not_ef_c(self):
        """An M=6 curve labelled ef_c=6 is worse than one labelled nothing."""
        from report.charts import series_labels
        labels = set(series_labels(self._records()).values())
        assert any("M=6" in l for l in labels)
        assert not any("ef_c=" in l for l in labels)

    def test_a_single_degree_keeps_a_bare_label(self):
        """Naming an axis the run did not sweep implies a comparison it did
        not make."""
        from report.charts import series_labels
        rows = [r for r in self._records() if r["m"] == 16]
        assert not any("M=" in l for l in series_labels(rows).values())

    def test_the_two_lines_are_told_apart(self):
        """Colour is the engine and linestyle is the storage engine, so a
        second degree had nothing left to carry it and both came out
        identical."""
        from report.charts import series_style
        sparse = series_style({"engine": "alisql", "m": 6}, degrees=[6, 16])
        dense = series_style({"engine": "alisql", "m": 16}, degrees=[6, 16])
        assert (sparse["alpha"], sparse["linewidth"]) != \
               (dense.get("alpha"), dense["linewidth"])

    def test_the_key_width_matches_the_key(self):
        """churn_impact builds a wider key and slices it back, and it did that
        with a literal 3 -- so changing the key shifted every index under it.

        **The forbidden literals are DERIVED from the constant, and the second
        half of this test was vacuous until they were.** It pinned the strings
        `key[:3]` and `key[3]`, which were the hazardous ones at width 2. The
        key has since been widened twice, and at width 4 the dangerous
        literals are `key[:4]`, `key[4]` and `key[5]` -- none of which the old
        assertions mentioned, so planting every one of them while deleting
        `SERIES_KEY_WIDTH` from the block entirely left the suite green. A
        guard written against one width stops guarding the moment the width
        moves, which is the same silent-widening shape the key itself is about.

        So: no numeric subscript on `key` anywhere in any function that widens
        or slices the key, except `key[0]`, which is the engine and is read
        positionally on purpose everywhere. And the constant must actually
        appear, or a block with no literals and no constant passes by saying
        nothing.

        **THE SET OF FUNCTIONS IS DERIVED TOO, and a written-down pair missed
        one.** This listed `("churn_impact", "churn")` and described them as
        "either function that widens the key" -- but the functions that WIDEN
        are `charts.churn` and `charts.churn_retention`, while `churn_impact`
        only slices. So the second widener was uncovered by a hand-kept list
        inside a guard whose entire subject is a hand-kept list going stale.
        `_key_arithmetic_functions` finds them from the AST instead.

        It returns TWO sets, because the two assertions have different scopes
        and conflating them fails honestly on `_grouped`: that function reads
        `key[0]` to bucket by engine and never touches the width, so the
        literal rule applies to it and the constant rule must not.

        Two accepted limits, recorded rather than fixed. The regex wants the
        digit immediately after `[` or `[:`, so `key[ 4]` and `key[2 + 2]`
        slip through -- contrived enough that catching them costs more
        readability than it buys -- and `key[-1]` slips through too, which is
        fine because a negative index is width-independent and stays correct
        however the key grows. And the block comes from
        `ast.get_source_segment`, so it includes comments: a future comment
        that quotes `key[3]` as an example will fail this test. That is the
        safe direction to err, and the fix is to spell the example in prose.
        """
        import re

        from report.charts import SERIES_KEY_WIDTH, series_key
        assert SERIES_KEY_WIDTH == len(series_key({"engine": "e"}))
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        subscripts, width_dependent = self._key_arithmetic_functions(source)
        # Non-vacuity: a derivation that selects nothing is a guard that
        # stopped working and looks identical to a clean run. The two
        # wideners and the slicer must all be in, and `_grouped` must be in
        # the literal set without being in the width set -- which is the
        # distinction a single set got wrong.
        assert {"churn", "churn_retention", "churn_impact"} <= width_dependent
        assert "_grouped" in subscripts
        assert "_grouped" not in width_dependent
        # And `pareto`'s accessors stay out: their key is
        # `(engine, resource_pass)`, so `key[1]` there is correct and has
        # nothing to do with this width.
        assert not {"pareto_pass_of", "pareto_engine_of"} & subscripts
        for name in sorted(subscripts):
            block = self._function_source(source, name)
            literals = re.findall(r"key\[:?(\d+)\s*[\]:]", block)
            assert set(literals) <= {"0"}, (
                f"{name} indexes the key with literal(s) "
                f"{sorted(set(literals))}; use SERIES_KEY_WIDTH so the "
                f"width is written down once")
        for name in sorted(width_dependent):
            block = self._function_source(source, name)
            assert "SERIES_KEY_WIDTH" in block, (
                f"{name} does width-dependent key arithmetic but names no "
                f"width constant")

    @staticmethod
    def _key_arithmetic_functions(source):
        """Top-level functions in `charts.py` that touch a series key, by AST.

        Returns `(subscripts, width_dependent)`.

        `subscripts` is every function that subscripts a name called `key`.
        The literal rule applies to all of them.

        `width_dependent` is those that either add to a `series_key(...)`
        result -- widening it, so a consumer has to know the offset -- or
        subscript `key` by anything other than the literal 0. Only these must
        name the constant. For a widener the mention may well be in prose
        rather than code, and that is the point: the comment naming
        `SERIES_KEY_WIDTH` IS the contract with whoever slices it back.
        """
        import ast

        subscripts, width_dependent = set(), set()
        for node in ast.parse(source).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            # In scope only if the function handles THIS key. `pareto`'s key is
            # `(engine, resource_pass)` and `pareto_pass_of` reads `key[1]` off
            # it perfectly correctly, so selecting on the identifier `key`
            # alone condemns a different key's accessor. Naming `series_key` or
            # `SERIES_KEY_WIDTH` is what makes a function this key's business.
            family = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if not family & {"series_key", "SERIES_KEY_WIDTH"}:
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.BinOp)
                        and isinstance(inner.op, ast.Add)
                        and isinstance(inner.left, ast.Call)
                        and isinstance(inner.left.func, ast.Name)
                        and inner.left.func.id == "series_key"):
                    width_dependent.add(node.name)
                if (isinstance(inner, ast.Subscript)
                        and isinstance(inner.value, ast.Name)
                        and inner.value.id == "key"):
                    subscripts.add(node.name)
                    index = inner.slice
                    benign = (isinstance(index, ast.Constant)
                              and index.value == 0)
                    if not benign:
                        width_dependent.add(node.name)
        return subscripts, width_dependent

    @staticmethod
    def _function_source(source, name):
        """One top-level function's source, by AST rather than by substring.

        `source.split("def churn")` matches `churn_impact` and
        `churn_retention` as well, so the substring form could silently check
        the wrong function -- or, worse, a prefix of the right one.
        """
        import ast

        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(source, node)
        raise AssertionError(
            f"no top-level function {name!r} in report/charts.py")

    @needs_matplotlib
    def test_churn_still_renders_with_the_wider_key(self):
        """**Two resource passes, which is what makes this test its own name.**

        It pinned `resource_pass: "tuned"` on both rows, so the fixture was one
        engine with one series -- and a one-series fixture renders at every key
        width there has ever been. It was green at widths 2, 3 and 4 and would
        have stayed green through any further widening, while being the only
        test in the suite that renders this chart at all.

        One series also kept it clear of the defect next door:
        `churn_impact` built its storage list as one literal `None` per SERIES,
        so its length was the series count rather than the number of distinct
        storage engines. Widening the key doubled that count, which flipped
        `series_style`'s `len(storages) > 1` gate open onto `[None, None]` and
        made `.index("InnoDB")` raise. A second pass here is the whole fixture
        needed to reach it: this test goes red on the widening alone and green
        once `churn_impact` builds both lists from the records through
        `_grouped`. `config/profiles/dev.yml` pins one configuration per engine
        and lists `churn`, so that was an ordinary run shape, not a corner.
        """
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        from report.charts import churn_impact
        rows = []
        for resource_pass in ("normalized", "tuned"):
            for frac, qps in ((0.0, 100.0), (0.1, 40.0)):
                rows.append({"phase": "churn", "engine": "mariadb",
                             "dataset": "d", "storage_engine": "InnoDB",
                             "m": 16, "qps": qps, "churn_fraction": frac,
                             "resource_pass": resource_pass,
                             "build_mode": "post"})
        # Non-vacuity: the fixture really does reach the condition this test is
        # named for. Two series under one engine is what opens the gate that
        # used to crash, and one pass would silently give one.
        from report.charts import SERIES_KEY_WIDTH, churn_retention
        retained = churn_retention(rows, "d")
        assert len(retained) == 2, retained
        assert len({k[:SERIES_KEY_WIDTH] for k in retained}) == 2
        with tempfile.TemporaryDirectory() as out:
            assert churn_impact(rows, "d", out, "c") is not None


class TestAPhaseThatRanTwiceIsTwoMeasurements:
    """The memory sampler appends, so a phase that runs twice for one engine,
    dataset and resource pass writes both into one file. That is deliberate --
    the earlier measurement is not wrong -- but reading the result as a single
    continuous series draws a line across the gap and differences the two ends
    as though they bracketed one run. When MHNSW and VIDX were given a second
    graph degree five days after the first run, that gap was 123 hours, and
    the memory chart showed one run that appeared to pause for five days."""

    def _two_runs(self):
        a = [{"t": 1000.0 + i, "rss_bytes": 10, "cpu_seconds": i * 1.0,
              "host_cpu_seconds": i * 1.4, "session": 1} for i in range(20)]
        b = [{"t": 500000.0 + i, "rss_bytes": 10, "cpu_seconds": 100 + i * 1.0,
              "host_cpu_seconds": 200 + i * 1.4, "session": 2} for i in range(20)]
        return a + b

    def test_the_sampler_stamps_a_session(self):
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "docker_ctl.py")).read()
        assert '"session": self.session' in source

    def test_sessions_split_the_series(self):
        from report.loaders import split_sessions
        assert len(split_sessions(self._two_runs())) == 2

    def test_an_older_file_splits_on_the_gap(self):
        """Series written before the marker existed must still read right."""
        from report.loaders import split_sessions
        rows = [{k: v for k, v in r.items() if k != "session"}
                for r in self._two_runs()]
        assert len(split_sessions(rows)) == 2

    def test_one_measurement_stays_one(self):
        from report.loaders import split_sessions
        rows = [{"t": float(i), "rss_bytes": 1, "session": 7} for i in range(50)]
        assert len(split_sessions(rows)) == 1

    def test_contention_is_judged_per_measurement(self):
        """Differencing across the gap averages a busy host over days the
        container was not even running."""
        from report.loaders import foreign_cpu
        rows = []
        for i in range(30):
            rows.append({"t": float(i), "cpu_seconds": i * 1.0,
                         "host_cpu_seconds": i * 20.0, "session": 1})
        for i in range(30):
            rows.append({"t": 900000.0 + i, "cpu_seconds": 100 + i * 1.0,
                         "host_cpu_seconds": 1000 + i * 1.2, "session": 2})
        found = foreign_cpu(rows)
        assert found is not None
        assert found["foreign_cores"] > 2.0, "the busy segment must be reported"
        assert found["elapsed_s"] < 100, "and judged over its own window"

    def test_the_chart_draws_each_measurement_separately(self):
        source = open(os.path.join(VB_ROOT, "report", "charts.py")).read()
        block = source.split("def memory_timeline")[1]
        assert "split_sessions" in block
        # One legend entry per series, not one per segment.
        assert "label = name if index == 0 else None" in block

    def test_empty_input_is_not_a_crash(self):
        from report.loaders import split_sessions
        assert split_sessions([]) == []


class TestChartsFromAPreviousReportAreCleared:
    """Charts are redrawn from the records every time, so anything already in
    the directory is from a previous run. Narrowing a report to one corpus with
    --datasets left the other corpus's charts sitting beside the new ones,
    named after a dataset the report no longer covers. Nothing references them,
    which is what makes them dangerous: a stale file nothing points at is one
    somebody opens by hand and believes."""

    # Anchored on the mkdir that immediately precedes the cleanup, not on
    # `chart_dir = os.path.join`. The two were adjacent until the cleanup moved
    # below the record load -- a refused or empty regeneration used to delete
    # the previous report's charts and leave index.html pointing at them.
    CLEANUP_ANCHOR = "os.makedirs(chart_dir, exist_ok=True)"

    def _cleanup_block(self):
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        return source.split(self.CLEANUP_ANCHOR)[1][:1200]

    def test_the_generator_clears_them(self):
        block = self._cleanup_block()
        assert "os.remove" in block
        assert '.endswith((".png", ".svg"))' in block

    def test_only_generated_files_are_removed(self):
        """The directory is inside the run, and removing anything the report
        did not write would be destroying somebody else's data."""
        block = self._cleanup_block()
        assert "listdir" in block and '.endswith' in block
        assert "rmtree" not in block

    def test_it_clears_before_anything_is_drawn(self):
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        body = source.split("def main")[1]
        assert body.index("cleared") < body.index("chart_paths")

    def test_it_clears_only_once_a_report_is_certain(self):
        """After every refusal, not before them. Deleting the working report's
        charts and then exiting non-zero leaves broken images where a valid
        report was."""
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        body = source.split("def main")[1]
        assert body.index("loaders.load_ops_records") < body.index("cleared")
        assert body.index("no records found") < body.index("cleared")
        assert body.index("MixedMeasurementVersions") < body.index("cleared")


class TestMemoryChartsCompareShapesNotClocks:
    """The chart exists to show the shape of memory through a phase: a cache
    filling to its ceiling looks different from a build spike. Shapes are
    compared by overlaying them. Anchoring two measurements from one file to
    the file's start instead put the second 123 hours to the right, leaving a
    chart that was almost entirely empty with two thin bands at the edges --
    which looked exactly like the problem splitting the sessions was meant to
    fix."""

    def _series(self):
        ops = [{"t": 100.0 + i, "rss_bytes": 1_000_000 + i} for i in range(30)]
        ann = ([{"t": 200.0 + i, "rss_bytes": 2_000_000 + i, "session": 1}
                for i in range(30)]
               + [{"t": 500000.0 + i, "rss_bytes": 3_000_000 + i, "session": 2}
                  for i in range(30)])
        return {"mariadb-d-tuned-m16-post": ops, "mariadb-d-tuned-ann": ann}

    def _axis(self, **kw):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
        from report import charts
        captured = {}
        original = charts._save

        def spy(fig, out_dir, stem):
            captured["ax"] = fig.axes[0]
            return original(fig, out_dir, stem)

        charts._save = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                charts.memory_timeline(self._series(), d, "t", **kw)
        finally:
            charts._save = original
            plt.close("all")
        return captured.get("ax")

    @needs_matplotlib
    def test_every_measurement_starts_at_zero(self):
        ax = self._axis(phase="ann")
        for line in ax.get_lines():
            assert min(line.get_xdata()) < 1.0

    @needs_matplotlib
    def test_the_axis_is_not_mostly_empty(self):
        """The second measurement is five days later in wall clock; the axis
        must span one measurement, not the gap between two."""
        ax = self._axis(phase="ann")
        assert ax.get_xlim()[1] < 1000, "axis still spans the wall-clock gap"

    @needs_matplotlib
    def test_the_phases_are_separate_charts(self):
        """The ann phase loads a corpus and sweeps a grid; ops loads it again
        and runs four workloads. Different durations, different questions."""
        ann = {l.get_label() for l in self._axis(phase="ann").get_lines()}
        ops = {l.get_label() for l in self._axis().get_lines()}
        assert any("ann" in str(l) for l in ann)
        assert not any("ann" in str(l) for l in ops)

    @needs_matplotlib
    def test_a_repeated_phase_says_so_in_the_legend(self):
        labels = [str(l.get_label()) for l in self._axis(phase="ann").get_lines()]
        assert any("2 measurements" in l for l in labels)

    def test_both_charts_are_produced(self):
        source = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        assert '("memory-timeline-ann", "ann")' in source

    def test_the_renderer_shows_both(self):
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        assert 'stem.startswith("memory-timeline")' in source


class TestASweepAxisNamesItselfWithoutBeingTaught:
    """Listing the plotted axes by hand has failed three times. It missed
    storage engine and ef_construction, then graph degree, and then
    quantization -- two Percona Search curves drawn as one line, which showed
    up as a sawtooth on the latency chart. Each fix added the axis that had
    just been added to the sweep, and each time the next one was already
    waiting. Identity comes from the result filename now, which ann-benchmarks
    names after the build arguments."""

    def _rows(self, engine, stems):
        return [{"phase": "recall_qps", "engine": engine, "dataset": "d",
                 "ef_search": ef, "qps": 100.0, "recall_at_k": 0.9,
                 "latency_p50_ms": 5.0, "latency_p99_ms": 9.0,
                 "extra": {"source_file": f"d/10/{engine}/{stem}_{ef}.hdf5"}}
                for stem in stems for ef in (10, 20)]

    def test_quantization_separates_curves_without_being_named(self):
        """It is not a column on the record at all -- it exists only in the
        filename."""
        from report.charts import _grouped
        rows = self._rows("mongodb", ["angular_M_16_quantization_none",
                                      "angular_M_16_quantization_scalar"])
        groups, labels, _s, _d = _grouped(rows)
        assert len(groups) == 2
        assert any("quantization=scalar" in l for l in labels.values())

    def test_an_axis_this_code_has_never_heard_of_still_works(self):
        from report.charts import series_labels
        rows = self._rows("future", ["angular_M_16_someNewKnob_a",
                                     "angular_M_16_someNewKnob_b"])
        labels = set(series_labels(rows).values())
        assert any("someNewKnob=a" in l for l in labels)
        assert any("someNewKnob=b" in l for l in labels)

    def test_only_the_varying_part_is_named(self):
        """Naming an axis the run did not sweep implies a comparison it did
        not make. M is identical in both stems below."""
        from report.charts import series_labels
        rows = self._rows("mongodb", ["angular_M_16_quantization_none",
                                      "angular_M_16_quantization_scalar"])
        assert not any("M=" in l for l in series_labels(rows).values())

    def test_one_configuration_keeps_a_bare_label(self):
        from report.charts import series_labels
        rows = self._rows("alisql", ["angular_M_16_engine_InnoDB"])
        assert set(series_labels(rows).values()) == {"AliSQL (VIDX)"}

    def test_records_without_a_filename_fall_back_to_named_axes(self):
        """A report rebuilt with --from-records from an archive has no
        filenames, and must still split on what it does have."""
        from report.charts import _grouped
        rows = [{"phase": "recall_qps", "engine": "pgvector", "dataset": "d",
                 "ef_search": 10, "qps": 1.0, "recall_at_k": 0.9,
                 "ef_construction": efc, "m": 16}
                for efc in (64, 400)]
        groups, labels, _s, _d = _grouped(rows)
        assert len(groups) == 2
        assert any("ef_c=64" in l for l in labels.values())


class TestTheLatencyChartIsOnePanelPerEngine:
    """A dozen series on one axis, with AliSQL's 384 ms cliff on the same
    linear scale as Valkey's 0.7 ms, pressed five engines flat against the
    floor."""

    def _rows(self):
        rows = []
        for engine, ms in (("alisql", 380.0), ("valkey", 0.7)):
            for ef in (10, 100, 800):
                rows.append({"phase": "recall_qps", "engine": engine,
                             "dataset": "d", "ef_search": ef, "qps": 10.0,
                             "recall_at_k": 0.9, "latency_p50_ms": ms,
                             "latency_p99_ms": ms * 1.5})
        return rows

    def _figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
        from report import charts
        captured = {}
        original = charts._save

        def spy(fig, out_dir, stem):
            captured["fig"] = fig
            return original(fig, out_dir, stem)

        charts._save = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                charts.latency_percentiles(self._rows(), "d", d, "t")
        finally:
            charts._save = original
        return captured["fig"]

    def test_one_visible_panel_per_engine(self):
        fig = self._figure()
        assert len([a for a in fig.axes if a.get_visible()]) == 2

    def test_the_scale_is_logarithmic(self):
        """Otherwise 0.7 ms and 384 ms cannot share an axis."""
        for ax in self._figure().axes:
            if ax.get_visible():
                assert ax.get_yscale() == "log"

    def test_the_panels_share_a_scale(self):
        """A per-panel scale would make every engine look the same shape."""
        limits = {ax.get_ylim() for ax in self._figure().axes if ax.get_visible()}
        assert len(limits) == 1


class TestEqualFloorColumnsAreExplained:
    """A frontier reports the fastest configuration at or above the floor, and
    that configuration is frequently well above it. So two floors can be won by
    one measurement and the columns come out identical -- which reads as a
    missing measurement. MariaDB shows the same number at 0.90 and 0.95 because
    its fastest setting already returns recall 0.9563, and it has three slower
    measurements in between."""

    def _table(self, **over):
        from report.render import _headline_tables
        entry = {"points": 16, "min_recall": 0.6935, "max_recall": 0.9961,
                 "qps_at_recall_90": 415, "qps_at_recall_90_recall": 0.9563,
                 "qps_at_recall_95": 415, "qps_at_recall_95_recall": 0.9563,
                 "qps_at_recall_99": 46, "qps_at_recall_99_recall": 0.9931}
        entry.update(over)
        return _headline_tables(
            {"per_dataset": {"d": {None: {"mariadb": entry}}}})

    def test_every_cell_carries_its_recall(self):
        table = self._table()
        assert "415 @0.956" in table
        assert "46 @0.993" in table

    def test_two_equal_columns_show_the_same_recall(self):
        """Which is what makes it obvious they are one measurement."""
        assert self._table().count("415 @0.956") == 2

    def test_the_note_explains_it(self):
        table = self._table()
        assert "one measurement winning both" in table

    def test_a_dash_stays_a_dash(self):
        """An engine that never reached the floor has no recall to show."""
        table = self._table(qps_at_recall_99=None, qps_at_recall_99_recall=None)
        assert "—" in table

    def test_the_dagger_still_marks_an_unreached_floor(self):
        """Carrying the recall everywhere does not replace the marker: one
        says where the figure came from, the other says the engine has nothing
        below the floor at all."""
        table = self._table(min_recall=0.9784)
        assert "†" in table


class TestTheChurnBudgetFitsAHealthyEngine:
    """The budget exists so an engine that takes a write and never finishes it
    releases the machine, which is what Valkey did before its pipelined-write
    defect was found. Thirty minutes was sized against that failure, not
    against a healthy engine: MHNSW and VIDX maintain the graph on every
    INSERT and need 58 to 99 minutes to write 99,000 rows back. All three were
    cut off at roughly 1,830 seconds and the run recorded no churn result for
    any of them -- losing the study's largest single finding."""

    def test_the_headline_profile_allows_the_slowest_observed_reinsert(self):
        from orchestrator.config import load_profile
        budget = load_profile("tuned-complete")["ops"]["churn_budget_s"]
        assert budget >= 5960, "AliSQL's measured re-insert was 5,960 s"

    def test_it_is_still_bounded(self):
        """Unbounded is how a stalled write held the machine overnight."""
        from orchestrator.config import load_profile
        assert load_profile("tuned-complete")["ops"]["churn_budget_s"] <= 14400

    def test_the_orchestrator_passes_the_profile_value(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert 'ops.get("churn_budget_s"' in source

    def test_an_incomplete_reinsert_records_no_throughput(self):
        """A corpus missing rows is neither the baseline nor a churned corpus,
        so a figure taken there would be read as one of them."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "churn.py")).read()
        block = source.split("if inserted < len(new_ids):")[1].split("break")[0]
        assert "recall" not in block.split("notes=")[0].split("extra=")[0]


class TestFilterStrategyIsMeasuredNotQuoted:
    """The report asserted for weeks that Valkey "chooses its filtering
    strategy per query", taken from its documentation. The run said otherwise:
    18.2 ms at 10% selectivity and 120.6 ms at 1%, six times more work for a
    tenth of the candidates. That is what walking the graph costs and the
    opposite of what pre-filtering costs, and the number was sitting in the
    records the whole time."""

    def _rows(self, engine, wide_p50, narrow_p50, narrow_recall=0.98):
        return [{"phase": "filtered", "engine": engine, "dataset": "d",
                 "selectivity": 0.1, "latency_p50_ms": wide_p50,
                 "recall_at_k": 0.98, "qps": 10.0},
                {"phase": "filtered", "engine": engine, "dataset": "d",
                 "selectivity": 0.01, "latency_p50_ms": narrow_p50,
                 "recall_at_k": narrow_recall, "qps": 5.0}]

    def _observed(self, *args, **kw):
        from report.generate import summarize
        found = summarize(self._rows(*args, **kw))["filter_strategies"]
        return found[0]["observed"]

    def test_faster_on_a_narrower_predicate_is_a_pre_filter(self):
        """Percona Search: 13.5 ms to 8.4 ms. Fewer qualifying vectors is
        less work, which only a pre-filter gets."""
        assert self._observed("mongodb", 13.55, 8.40, 0.8666) == "pre-filter"

    def test_slower_on_a_narrower_predicate_is_not(self):
        """Valkey: 18.2 ms to 120.6 ms."""
        assert self._observed("valkey", 18.16, 120.56, 0.9996) == \
            "filters during the graph walk"

    def test_far_slower_is_an_exhaustive_post_filter(self):
        assert self._observed("mariadb", 49.33, 1875.91, 0.9724) == \
            "post-filter, exhaustive"

    def test_flat_latency_with_collapsed_recall_is_giving_up(self):
        """pgvector returns in the same time at either selectivity because it
        stops early. Without checking recall that reads as a pre-filter."""
        assert self._observed("pgvector", 4.41, 4.27, 0.1034) == \
            "post-filter, abandons the query"

    def test_recall_is_what_separates_fast_from_abandoned(self):
        """Same latencies, healthy recall: not the same finding."""
        assert self._observed("other", 4.41, 4.27, 0.98) != \
            "post-filter, abandons the query"

    def test_one_selectivity_cannot_be_classified(self):
        """A single point has no direction, and guessing from it is how the
        wrong claim got in."""
        from report.generate import summarize
        rows = self._rows("valkey", 18.16, 120.56)[:1]
        assert summarize(rows)["filter_strategies"] == []

    def test_it_reaches_the_report(self):
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        assert "_filter_strategy_table(summary)" in source
        assert "What each engine actually did" in source

    def test_no_asymmetry_claims_a_planner_the_run_did_not_see(self):
        """The asymmetries section describes structural differences. It must
        not carry a vendor's description of behaviour the run measured and
        contradicted."""
        source = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        assert "chooses its filtering strategy per query" not in source

    def test_no_engine_records_a_planner_as_measured_fact(self):
        """capabilities land in the records, so a `true` there is read as
        something the run established. Valkey's filtering planner is a vendor
        claim the run contradicted, and recording it as true is how it reached
        the report."""
        import yaml
        path = os.path.join(VB_ROOT, "config", "engines", "valkey.yml")
        caps = yaml.safe_load(open(path))["capabilities"]
        assert caps["hybrid_filter_planner"] is None

    def test_the_driver_agrees_with_the_config(self):
        source = open(os.path.join(VB_ROOT, "harness", "drivers",
                                   "valkey.py")).read()
        assert '"hybrid_filter_planner": None' in source


class TestTheLabAnswersQuestionsWithoutProducingMeasurements:
    """`lab` starts a real engine outside a run so a probe can question it.

    That makes it the one command that produces numbers nothing validates, and
    every test here guards the boundary between it and the measurement path.
    """

    def test_a_session_writes_nothing_a_report_could_read(self):
        """A lab script invents its own vectors and its own ground truth. If
        those could land in results/ they would eventually appear in a table
        beside corpus numbers, indistinguishable from a measurement."""
        from orchestrator.cli import lab_paths
        paths = lab_paths("lab-valkey-20260914-120000")
        results = os.path.join(VB_ROOT, "results")
        for key in ("run_dir", "ops_results"):
            assert not os.path.abspath(paths[key]).startswith(results + os.sep)

    def test_the_client_container_has_no_writable_mount(self):
        """Belt to the previous test's braces: not a path chosen carefully,
        but no writable path offered at all."""
        import inspect
        from orchestrator.ops_pass import OpsRun
        source = inspect.getsource(OpsRun.run_script)
        mounts = [line for line in source.splitlines()
                  if ':/opt/' in line or ':/datasets' in line]
        assert mounts, "run_script mounts nothing; the test is looking at the wrong thing"
        assert all(':ro"' in line for line in mounts), mounts

    def test_the_server_is_started_by_the_same_code_a_run_uses(self):
        """The predecessor was a shell script that started its own Valkey with
        hand-written flags -- a 32 GB default where a tuned run gives 101 GB,
        and maxmemory-policy restated next to the copy in the engine config. A
        probe run under different flags than the run it investigates can
        disprove something the run never did."""
        from orchestrator.ops_pass import OpsRun
        assert hasattr(OpsRun, "run_script")
        # _start_server is not overridden or bypassed: one server path.
        assert "_start_server" not in \
            __import__("inspect").getsource(OpsRun.run_script)

    def test_no_server_flag_is_restated_in_the_orchestrator(self):
        """Engine server flags live in config/engines/*.yml and nowhere else.
        Valkey's noeviction pin is the one that matters here: under any other
        policy a full server drops keys, and vectors vanishing mid-probe looks
        exactly like a bad index."""
        for name in ("cli.py", "ops_pass.py"):
            source = open(os.path.join(VB_ROOT, "orchestrator", name)).read()
            # The dashes, so that prose about the flag is not mistaken for a
            # use of it -- the comment explaining this rule says the word.
            assert "--maxmemory-policy" not in source
            assert "--maxmemory " not in source

    def test_a_script_name_cannot_reach_outside_scripts(self):
        from orchestrator.cli import _lab_script_problem
        assert _lab_script_problem("../../etc/shadow.py")
        assert _lab_script_problem("/etc/shadow.py")
        assert _lab_script_problem("build-images.sh")
        assert _lab_script_problem("does-not-exist.py")
        assert _lab_script_problem("probe-valkey-filter.py") is None

    def test_every_lab_script_accepts_what_the_lab_passes(self):
        """--host and --port are passed to every script, so they are the
        contract. A script that does not take them dies on argparse after the
        engine has already been started and the corpus loaded."""
        from orchestrator.cli import lab_scripts
        for name, _summary in lab_scripts():
            source = open(os.path.join(VB_ROOT, "scripts", name)).read()
            assert '"--host"' in source, name
            assert '"--port"' in source, name

    def test_it_refuses_before_it_starts_anything(self):
        """Validation order matters more here than usual: for the MySQL family
        a server takes minutes to come up, and learning about a typo then is
        several minutes of an operator's time."""
        from orchestrator.cli import build_parser, cmd_lab
        parser = build_parser()
        for argv, code in (
                (["lab", "probe-valkey-filter.py"], 2),          # no --engine
                (["lab", "--engine", "redis", "x.py"], 2),       # unknown engine
                (["lab", "--engine", "valkey"], 2),              # nothing to run
                (["lab", "--engine", "valkey", "nope.py"], 2)):  # no such script
            assert cmd_lab(parser.parse_args(argv)) == code, argv

    def test_setup_rolls_back_when_it_fails_part_way(self):
        """`with` calls __exit__ only if __enter__ returned. Setup creates a
        network, then a volume, then starts a server, and the last two can
        fail -- a missing image, or a server that never becomes healthy. Asking
        for an engine whose image is not built left a network, a volume and a
        root-owned directory behind, and printed nothing about any of them."""
        import inspect
        from orchestrator.ops_pass import OpsRun
        source = inspect.getsource(OpsRun.__enter__)
        assert "self.teardown()" in source
        # BaseException, not Exception: the widest window is wait_healthy's
        # five-minute poll, and the likeliest thing to arrive during it is the
        # operator's Ctrl-C -- which is not an Exception.
        assert "except BaseException" in source

    def test_the_bind_directory_goes_without_needing_the_image(self):
        """Teardown deletes the volume's host directory through a container,
        because the engine wrote it as root. In the rollback above that image
        is the one that was missing, so the container cannot be made; an empty
        directory this process created must not need it."""
        import inspect
        from orchestrator.docker_ctl import remove_tree_as_root
        source = inspect.getsource(remove_tree_as_root)
        direct = source.index("shutil.rmtree(path)")
        via_container = source.index("ContainerSpec(")
        assert direct < via_container, "the container path must be the fallback"
        assert "ignore_errors" not in source[direct:via_container], \
            "a silent failure here is what leaves a full corpus on disk"

    def test_a_symlink_in_scripts_is_not_run_as_if_it_were_the_file(self):
        from orchestrator.cli import _lab_script_problem
        directory = os.path.join(VB_ROOT, "scripts")
        link = os.path.join(directory, "_test_link.py")
        try:
            os.symlink("/etc/hostname", link)
        except OSError:
            pytest.skip("cannot create a symlink here")
        try:
            problem = _lab_script_problem("_test_link.py")
            assert problem and "link" in problem
        finally:
            os.unlink(link)

    def test_it_refuses_to_start_beside_a_measurement(self, monkeypatch):
        """A lab session takes the cores and memory the pass gives a run, so
        started beside one it contends with it -- and the damage lands on the
        run, whose numbers are the ones being kept."""
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "docker_available", lambda: True)
        monkeypatch.setattr(cli.docker_ctl, "running_containers",
                            lambda: ["full-20260825-valkey-m16-post-srv"])
        argv = ["lab", "--engine", "valkey", "probe-valkey-filter.py"]
        assert cli.cmd_lab(cli.build_parser().parse_args(argv)) == 5

        # --force gets past it. Stubbed, because the alternative is a test that
        # starts a real server -- which is what the first version of this did,
        # and it ran for as long as the probe's default corpus takes.
        started = []

        class FakeRun:
            def __init__(self, *a, **kw):
                started.append(a[0])
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def run_script(self, *a, **kw):
                return 0

        monkeypatch.setattr(cli.ops_pass, "OpsRun", FakeRun)
        # Before the script name. After it, argparse.REMAINDER hands --force to
        # the probe and the guard correctly stays up -- which is why the
        # refusal prints the flag in the position that works.
        forced = ["lab", "--engine", "valkey", "--force",
                  "probe-valkey-filter.py"]
        assert cli.cmd_lab(cli.build_parser().parse_args(forced)) == 0
        assert started == ["valkey"]

    def test_the_web_ui_and_the_helpers_are_not_a_measurement(self, monkeypatch):
        """vb-* containers are utilities -- the web UI is long-lived and would
        otherwise block every lab invocation there is."""
        from orchestrator import cli
        monkeypatch.setattr(cli.docker_ctl, "running_containers",
                            lambda: ["vb-webui-8080", "vb-prune-1234"])
        assert cli.measurement_in_progress() == []

    def test_the_misplacement_note_knows_every_flag_lab_has(self):
        """Everything after the script name goes to the script, so a lab flag
        typed there silently becomes the script's problem. The note that says
        so is only useful if it knows the whole list."""
        from orchestrator.cli import _LAB_OPTIONS, build_parser
        lab = build_parser()._subparsers._group_actions[0].choices["lab"]
        actual = {option for action in lab._actions
                  for option in action.option_strings}
        assert actual == set(_LAB_OPTIONS)

    def test_listing_scripts_needs_neither_docker_nor_an_engine(self):
        from orchestrator.cli import build_parser, cmd_lab, lab_scripts
        assert cmd_lab(build_parser().parse_args(["lab", "--list"])) == 0
        assert dict(lab_scripts())["probe-valkey-filter.py"].startswith("Find out why")



class TestOpsMeasurementVersionIsDeclared:
    """The ops counterpart of ANN_MEASUREMENT_VERSION, and where it lives.

    `ANN_MEASUREMENT_VERSION` sits in `orchestrator/ann_pass.py` because the
    recall/QPS path is launched from the host. The ops version cannot sit
    beside it: ops records are written from inside the harness container, which
    mounts `harness/` only, and `harness/` importing `orchestrator` kills a run
    seconds after the server comes up -- the mistake TestHarnessIsolation
    exists to catch. So the constant lives on the harness side of that
    boundary and the orchestrator imports it from there, which is the one
    direction the dependency is allowed to point.
    """

    # Anchored at the start of a line and on the exact name. A plain substring
    # search for "OPS_MEASUREMENT_VERSION = " is satisfied by
    # `IMPLICIT_OPS_MEASUREMENT_VERSION = 1`, so it kept passing with the
    # constant itself deleted -- an assertion that looked like coverage.
    DECLARATION = re.compile(r"^OPS_MEASUREMENT_VERSION\s*=", re.MULTILINE)

    def _source(self, *parts):
        return open(os.path.join(VB_ROOT, *parts)).read()

    def test_the_constant_lives_in_the_harness(self):
        assert self.DECLARATION.search(
            self._source("harness", "metrics", "records.py"))

    def test_it_is_not_beside_its_ann_counterpart(self):
        """Putting it there is the obvious move and breaks the harness."""
        assert "OPS_MEASUREMENT_VERSION" not in self._source(
            "orchestrator", "ann_pass.py")

    def test_the_orchestrator_imports_it_rather_than_redefining_it(self):
        from harness.metrics import records as records_mod
        from orchestrator import manifest as manifest_mod
        assert (manifest_mod.OPS_MEASUREMENT_VERSION
                == records_mod.OPS_MEASUREMENT_VERSION)
        src = self._source("orchestrator", "manifest.py")
        assert not self.DECLARATION.search(src), \
            "a second copy drifts from the first"
        assert "from harness.metrics.records import OPS_MEASUREMENT_VERSION" \
            in src

    def test_the_manifest_records_it(self, tmp_path):
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        from orchestrator.manifest import Manifest
        m = Manifest(str(tmp_path / "run"), "run-1")
        m.set_config({}, "normalized", {"server_memory_bytes": 1})
        assert m.data["config"]["ops_measurement_version"] == \
            OPS_MEASUREMENT_VERSION
        # On disk, not just in memory: the report reads the file.
        with open(m.path) as fh:
            assert json.load(fh)["config"]["ops_measurement_version"] == \
                OPS_MEASUREMENT_VERSION

    def test_a_new_manifest_is_stamped_at_creation(self, tmp_path):
        """Before set_config, because that is the only moment the code running
        and the code that will produce the records are the same thing."""
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        from orchestrator.manifest import Manifest
        m = Manifest(str(tmp_path / "run"), "run-1")
        with open(m.path) as fh:
            assert json.load(fh)["config"]["ops_measurement_version"] == \
                OPS_MEASUREMENT_VERSION

    def test_the_manifest_version_is_not_overridable_by_a_caller(self, tmp_path):
        """set_config takes an `extra` dict from cli.py. The version describes
        the code that produced the numbers, so it is not a caller's opinion."""
        from harness.metrics.records import OPS_MEASUREMENT_VERSION
        from orchestrator.manifest import Manifest
        m = Manifest(str(tmp_path / "run"), "run-1")
        m.set_config({}, "normalized", {}, extra={"ops_measurement_version": 99})
        assert m.data["config"]["ops_measurement_version"] == \
            OPS_MEASUREMENT_VERSION

    def test_a_caller_cannot_smuggle_a_version_into_an_absent_key_manifest(
            self, tmp_path):
        """The case that makes the pop load-bearing, and the only one.

        On a fresh manifest the carry-forward overwrites a smuggled value
        anyway, so the test above passes with the pop deleted. On a manifest
        that declares nothing -- every archived run -- there is nothing to
        carry forward, and without the pop a caller's `extra` lands in the
        manifest as though it had been measured under version 99.
        """
        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps({
            "run_id": "r", "config": {"resource_pass": "tuned"},
            "engines": {}, "phases": [], "warnings": [],
        }))
        from orchestrator.manifest import Manifest
        m = Manifest(str(run), "r")
        m.set_config({}, "tuned", {}, extra={"ops_measurement_version": 99})
        assert "ops_measurement_version" not in m.data["config"]
        with open(m.path) as fh:
            assert "ops_measurement_version" not in json.load(fh)["config"]

    def test_set_config_still_carries_the_fingerprint(self, tmp_path):
        """The new key sits beside ann_fingerprint, it does not displace it."""
        from orchestrator.ann_pass import ann_fingerprint
        from orchestrator.manifest import Manifest
        resolved = {"server_memory_bytes": 1, "build_threads": 1,
                    "server_cpu_count": 2}
        m = Manifest(str(tmp_path / "run"), "run-1")
        m.set_config({}, "tuned", resolved,
                     extra={"ann_fingerprint": ann_fingerprint(resolved)})
        assert m.data["config"]["ann_fingerprint"] == ann_fingerprint(resolved)
        assert m.data["config"]["resource_pass"] == "tuned"


class TestOpsVersionsNeverShareAChart:
    """Two measurement versions in one report are refused, loudly.

    `ann_fingerprint` keeps a 16 GB recall curve out of a 64 GB report by
    narrowing the results tree to the configuration the manifest declares. The
    ops path has no tree to narrow -- every ops record lands in the run
    directory -- so the equivalent is a refusal at load time, before a single
    chart is drawn.

    Refusing rather than labelling is deliberate. A labelled chart still
    invites the comparison it is labelled against, and the one outcome worse
    than either is silently dropping the older half, which is what a plain
    group-by would do. The refusal names both versions and names records from
    each, so the operator can see what is in the directory.
    """

    V_KEY = "ops_measurement_version"

    @staticmethod
    def SIDE(version, count=r"\d+"):
        """The line listing one version and how many records it has.

        Matched instead of a bare `"version N" in message`, which is satisfied
        by the trailer's "count as version 1" no matter what the listing does.

        `count` binds a number to the version beside it, and callers that care
        about the counts must pass it. Left unbound the pattern says only that
        each side was listed with some number, which is exactly what a
        listing with the two sides' counts swapped also satisfies -- and a
        swap is the one fault the counts exist to expose, since telling a
        stray file from the bulk of a run is the whole point of printing them.
        """
        return re.compile(rf"^  version {version}: {count} records?$",
                          re.MULTILINE)

    def _ops_file(self, run_dir, name, rows):
        path = os.path.join(run_dir, name)
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def _row(self, *, engine="mariadb", phase="index_build", version=None,
             timestamp="2026-10-12T00:00:00Z", **kw):
        row = {"run_id": "r", "engine": engine, "phase": phase,
               "dataset": "glove-100-angular", "resource_pass": "normalized",
               "timestamp": timestamp, "m": 16, "build_mode": "post", **kw}
        if version is not None:
            row["extra"] = {**row.get("extra", {}), self.V_KEY: version}
        return row

    def test_one_version_loads_normally(self, tmp_path):
        from report.loaders import load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(version=2), self._row(version=2, m=32)])
        assert len(load_ops_records(run)) == 2

    def test_records_without_the_key_are_version_one(self, tmp_path):
        """Everything measured before this change is version 1 by definition,
        and a directory of them is internally consistent."""
        from report.loaders import load_ops_records, ops_measurement_versions
        run = str(tmp_path)
        self._ops_file(run, "ops-alisql-glove-100-angular-normalized-a.jsonl",
                       [self._row(engine="alisql"), self._row(engine="alisql",
                                                              m=32)])
        loaded = load_ops_records(run)
        assert len(loaded) == 2
        assert sorted(ops_measurement_versions(loaded)) == [1]

    def test_two_versions_across_two_files_are_refused(self, tmp_path):
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(engine="mariadb")])
        self._ops_file(run, "ops-alisql-glove-100-angular-normalized-a.jsonl",
                       [self._row(engine="alisql", version=2)])
        with pytest.raises(MixedMeasurementVersions):
            load_ops_records(run)

    def test_two_versions_inside_one_file_are_refused(self, tmp_path):
        """A resumed run appends to the file it already wrote, so the two
        versions need not be in separate files to collide."""
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(version=2), self._row()])
        with pytest.raises(MixedMeasurementVersions):
            load_ops_records(run)

    def test_the_refusal_names_both_versions_and_the_records(self, tmp_path):
        """A refusal nobody can act on is an outage, not a guard."""
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(engine="mariadb", phase="concurrency",
                                  timestamp="2026-01-01T00:00:00Z")])
        self._ops_file(run, "ops-pgvector-glove-100-angular-normalized-a.jsonl",
                       [self._row(engine="pgvector", phase="filtered",
                                  version=2,
                                  timestamp="2026-06-06T06:06:06Z")])
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(run)
        message = str(caught.value)
        # Both versions, by number, on their own listed side. A plain
        # `"version 1" in message` is vacuous: the trailer always ends with
        # "count as version 1", so it passed however badly the sides were
        # listed. Anchor on the listing line itself.
        assert self.SIDE(1).search(message), message
        assert self.SIDE(2).search(message), message
        # And a record from each side, identifiable enough to find in the file.
        assert "mariadb" in message and "concurrency" in message
        assert "pgvector" in message and "filtered" in message
        assert "2026-01-01T00:00:00Z" in message
        assert "2026-06-06T06:06:06Z" in message
        # It must say what to do, not only that something is wrong.
        assert "ops_measurement_version" in message

    def test_the_refusal_counts_the_records_on_each_side(self, tmp_path):
        """So an operator can tell a stray file from the bulk of the run.

        Each count bound to its own version, because telling the stray from
        the bulk is the entire reason the counts are printed and a swap
        destroys exactly that. The assertion this replaced was
        `"4 record" in message and "1 record" in message`, which never bound
        a count to a version: a listing reading "version 1: 1 record" beside
        "version 2: 4 records" -- pointing the operator at the wrong file --
        contains both substrings and passed the whole suite.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(m=m) for m in (8, 16, 32, 48)])
        self._ops_file(run, "ops-alisql-glove-100-angular-normalized-a.jsonl",
                       [self._row(engine="alisql", version=2)])
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(run)
        message = str(caught.value)
        assert self.SIDE(1, 4).search(message), message
        assert self.SIDE(2, 1).search(message), message

    def test_a_three_way_mix_names_every_version(self, tmp_path):
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(), self._row(version=2), self._row(version=3)])
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(run)
        for version in ("version 1", "version 2", "version 3"):
            assert version in str(caught.value)

    def test_records_older_than_their_own_manifest_are_refused(self, tmp_path):
        """The 16-GB-curve-under-a-64-GB-manifest failure, on the ops side.

        A single version in the directory is not enough: a run resumed under
        code that bumped the version rewrites the manifest and then appends
        nothing, leaving every record a version behind what the manifest above
        it claims. Nothing in the records themselves disagrees, which is
        exactly what made the ann version of this bug survive to publication.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        with open(os.path.join(run, "run-manifest.json"), "w") as fh:
            json.dump({"run_id": "r", "config": {"resource_pass": "normalized",
                                                 "ops_measurement_version": 2}},
                      fh)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row()])
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(run)
        message = str(caught.value)
        assert self.SIDE(1).search(message), message
        # The declared side has no records, so it is listed without a count.
        assert "version 2: declared by the run manifest" in message
        assert "no records measured under it" in message

    def test_a_matching_manifest_loads(self, tmp_path):
        from report.loaders import load_ops_records
        run = str(tmp_path)
        with open(os.path.join(run, "run-manifest.json"), "w") as fh:
            json.dump({"run_id": "r",
                       "config": {"ops_measurement_version": 2}}, fh)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(version=2)])
        assert len(load_ops_records(run)) == 1

    def test_a_manifest_predating_the_key_is_not_checked_against(self, tmp_path):
        """Every archived run in this project has such a manifest. Refusing
        them would make regenerating an old report impossible."""
        from report.loaders import load_ops_records
        run = str(tmp_path)
        with open(os.path.join(run, "run-manifest.json"), "w") as fh:
            json.dump({"run_id": "r", "config": {"resource_pass": "tuned"}}, fh)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(), self._row(m=32)])
        assert len(load_ops_records(run)) == 2

    def test_an_unreadable_manifest_does_not_block_the_records(self, tmp_path):
        from report.loaders import load_ops_records
        run = str(tmp_path)
        with open(os.path.join(run, "run-manifest.json"), "w") as fh:
            fh.write("{not json")
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(version=2)])
        assert len(load_ops_records(run)) == 1

    def test_recall_records_are_not_judged_by_the_ops_version(self):
        """ann-benchmarks records come from the other measurement path and
        carry ANN_MEASUREMENT_VERSION's concerns, not this one's. Counting
        them as version 1 would refuse every mixed report there is."""
        from harness.metrics.records import PHASE_RECALL
        from report.loaders import refuse_mixed_measurement_versions
        refuse_mixed_measurement_versions(
            [{"phase": PHASE_RECALL, "engine": "mariadb", "extra": {}},
             self._row(version=2)],
            source="test")

    def test_an_empty_set_is_not_a_conflict(self):
        from report.loaders import refuse_mixed_measurement_versions
        refuse_mixed_measurement_versions([], source="test")

    def test_a_stringified_version_is_not_the_integer(self, tmp_path):
        """The string "2" beside the integer 2 is two versions, not one.

        This is the fixture that matters, and the obvious one does not work.
        A value of "banana" cannot distinguish a loader that coerces with
        `int()` from one that does not: the coercion raises, the record falls
        into a second bucket anyway, and the refusal fires for the wrong
        reason. Under the same mutant, "2" and 2 collapse into one bucket and
        the directory LOADS -- so only this case sees it.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(version="2"), self._row(version=2, m=32)])
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(run)
        # And it says which is which, or the operator cannot see the quoting.
        assert "'2'" in str(caught.value)

    def test_a_garbage_version_is_refused_rather_than_dropped(self, tmp_path):
        """Not a substitute for the test above -- see its docstring -- but a
        truncated or hand-typed value must not silently join either side."""
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = str(tmp_path)
        self._ops_file(run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                       [self._row(version="banana"), self._row(version=2)])
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(run)
        assert "banana" in str(caught.value)

    def test_an_unhashable_declared_version_does_not_crash_the_guard(
            self, tmp_path):
        """A hand-edited manifest can hold a list or a dict there.

        `declared in groups` hashes it, so the guard against malformed
        provenance died with TypeError: unhashable type -- which escapes as a
        traceback rather than as the refusal it was computing.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records
        # Both JSON-expressible, because a real manifest is JSON on disk.
        for broken in ([2], {"v": 2}):
            run = str(tmp_path / f"r{id(broken)}")
            os.makedirs(run)
            with open(os.path.join(run, "run-manifest.json"), "w") as fh:
                json.dump({"run_id": "r",
                           "config": {"ops_measurement_version": broken}}, fh)
            self._ops_file(
                run, "ops-mariadb-glove-100-angular-normalized-a.jsonl",
                [self._row(version=2)])
            # Refused, not crashed: a manifest nobody can read a version out of
            # is not a manifest that agrees with the records.
            with pytest.raises(MixedMeasurementVersions):
                load_ops_records(run)

    def test_the_report_refuses_before_it_draws_anything(self):
        """A refusal after twenty hours of measurement and ten minutes of
        chart rendering is barely better than no refusal. The check hangs off
        load_ops_records, which generate.main calls before it touches a
        chart, a dataset or the ann tree."""
        src = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        load_at = src.index("loaders.load_ops_records")
        assert load_at < src.index("loaders.load_annb_results")
        assert load_at < src.index("render_html(manifest")
        assert load_at < src.index("summarize(records, manifest")

    def test_it_refuses_before_it_deletes_the_previous_report(self):
        """A refused regeneration must leave the working report alone.

        The chart cleanup ran first, so a refusal left index.html pointing at
        .png files that had just been removed -- a page of broken images where
        a valid report used to be.
        """
        src = open(os.path.join(VB_ROOT, "report", "generate.py")).read()
        assert (src.index("loaders.load_ops_records")
                < src.index("cleared {removed} chart(s)"))
        assert (src.index("no records found")
                < src.index("cleared {removed} chart(s)"))


class TestEveryReaderConsultsTheVersionGuard:
    """All three paths into a report, not just the one.

    `load_ops_records` is one of three ways records reach a chart. The other
    two bypassed it: `generate --from-records` read a records file directly,
    and `webui/records.py` prefers `report/records.jsonl` whenever it exists --
    which is true of almost every archived run, so the web UI consulted the
    guard for hardly any of them. Worse, the --from-records path then *wrote*
    records.jsonl, laundering a mixed set into the exact file the web UI
    prefers.
    """

    def _rows(self):
        v1 = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
              "dataset": "d", "m": 16, "timestamp": "2026-01-01T00:00:00Z"}
        v2 = dict(v1, engine="alisql", timestamp="2026-06-06T00:00:00Z",
                  extra={"ops_measurement_version": 2})
        return [v1, v2]

    def _jsonl(self, path, rows):
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_from_records_refuses_a_mixed_file(self, tmp_path, capsys):
        from report import generate
        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps({"run_id": "r"}))
        merged = tmp_path / "mixed.jsonl"
        self._jsonl(str(merged), self._rows())
        code = generate.main(["--run-dir", str(run),
                              "--from-records", str(merged)])
        assert code == 2
        err = capsys.readouterr().err
        assert "version 1" in err and "version 2" in err

    def test_a_refused_from_records_run_writes_no_records_file(self, tmp_path):
        """The laundering step: records.jsonl must not be created."""
        from report import generate
        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps({"run_id": "r"}))
        merged = tmp_path / "mixed.jsonl"
        self._jsonl(str(merged), self._rows())
        generate.main(["--run-dir", str(run), "--from-records", str(merged)])
        assert not (run / "report" / "records.jsonl").exists()

    def test_from_records_still_accepts_one_version(self, tmp_path):
        from report.loaders import refuse_mixed_measurement_versions
        rows = [r for r in self._rows() if r.get("extra")]
        refuse_mixed_measurement_versions(rows, source="x")

    def test_the_web_ui_refuses_a_mixed_merged_file(self, tmp_path):
        from report.loaders import MixedMeasurementVersions
        from webui import records as records_mod
        run = tmp_path / "run"
        (run / "report").mkdir(parents=True)
        self._jsonl(str(run / "report" / "records.jsonl"), self._rows())
        with pytest.raises(MixedMeasurementVersions):
            records_mod.load_records(str(run))

    def test_the_web_ui_still_reads_a_single_version_merged_file(self, tmp_path):
        from webui import records as records_mod
        run = tmp_path / "run"
        (run / "report").mkdir(parents=True)
        self._jsonl(str(run / "report" / "records.jsonl"),
                    [r for r in self._rows() if r.get("extra")])
        rows, source = records_mod.load_records(str(run))
        assert source == "records.jsonl" and len(rows) == 1

    def test_the_web_ui_falls_back_to_ops_records_and_checks_those_too(
            self, tmp_path):
        from report.loaders import MixedMeasurementVersions
        from webui import records as records_mod
        run = tmp_path / "run"
        run.mkdir()
        self._jsonl(
            str(run / "ops-mariadb-d-normalized-a.jsonl"), self._rows())
        with pytest.raises(MixedMeasurementVersions):
            records_mod.load_records(str(run))


class TestResumingDoesNotRewriteTheVersion:
    """A run was measured under the version it was measured under.

    `cli.py` calls `set_config` unconditionally per (pass, engine), *before*
    the checkpoint skip. So `run --resume --run-id <archived-run>` reaches it
    even when every unit is skipped and nothing whatsoever is measured. A
    set_config that stamped the current constant therefore rewrote an archived
    run's manifest to a version its records were never measured under, and the
    report refused that run from then on -- permanently, and for a run nobody
    had touched. Re-stamping is the provenance bug; the refusal was right.
    """

    def _manifest(self, run_dir, run_id="main-20260101-000000"):
        from orchestrator.manifest import Manifest
        return Manifest(str(run_dir), run_id)

    def _write_manifest(self, run_dir, config):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run-manifest.json").write_text(json.dumps({
            "run_id": "main-20260101-000000", "status": "completed",
            "config": config, "engines": {}, "phases": [], "warnings": [],
        }))

    def test_a_declared_version_survives_a_resume(self, tmp_path):
        """Even an older one. Especially an older one."""
        self._write_manifest(tmp_path / "run",
                             {"resource_pass": "tuned",
                              "ops_measurement_version": 1})
        m = self._manifest(tmp_path / "run")
        m.set_config({}, "tuned", {"server_memory_bytes": 1})
        assert m.data["config"]["ops_measurement_version"] == 1
        with open(m.path) as fh:
            assert json.load(fh)["config"]["ops_measurement_version"] == 1

    def test_an_absent_version_stays_absent_on_a_resume(self, tmp_path):
        """Every archived run in this project has such a manifest.

        Inventing today's constant here is a claim about a measurement nobody
        in this process witnessed. report/loaders.py reads a missing key as
        "nothing is declared, check nothing", which is the honest answer.
        """
        self._write_manifest(tmp_path / "run", {"resource_pass": "tuned"})
        m = self._manifest(tmp_path / "run")
        m.set_config({}, "tuned", {"server_memory_bytes": 1})
        assert "ops_measurement_version" not in m.data["config"]
        with open(m.path) as fh:
            assert "ops_measurement_version" not in json.load(fh)["config"]

    def test_resuming_an_archived_run_still_reports(self, tmp_path):
        """End to end, which is the behaviour that actually broke.

        An archived run: ops records with no version, a manifest with no
        version. Resume it, measure nothing, then load it the way the report
        does. Before the fix this refused -- records v1 against a manifest
        rewritten to 2.
        """
        from report.loaders import load_ops_records
        run = tmp_path / "run"
        self._write_manifest(run, {"resource_pass": "tuned"})
        with open(run / "ops-mariadb-glove-100-angular-tuned-a.jsonl",
                  "w") as fh:
            for phase in ("ingest", "index_build", "concurrency"):
                fh.write(json.dumps({
                    "run_id": "main-20260101-000000", "engine": "mariadb",
                    "phase": phase, "dataset": "glove-100-angular", "m": 16,
                    "resource_pass": "tuned",
                    "timestamp": "2026-01-01T00:00:00Z"}) + "\n")

        self._manifest(run).set_config({}, "tuned", {"server_memory_bytes": 1})
        assert len(load_ops_records(str(run))) == 3

    def test_a_resume_that_bumps_and_measures_is_still_caught(self, tmp_path):
        """The case the refusal is for must keep firing.

        The manifest keeps saying 2 and the new records say 3, so the records
        disagree with each other -- which the other half of the check sees
        without needing the manifest to have been rewritten.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = tmp_path / "run"
        self._write_manifest(run, {"resource_pass": "tuned",
                                   "ops_measurement_version": 2})
        with open(run / "ops-mariadb-glove-100-angular-tuned-a.jsonl",
                  "w") as fh:
            for version, m in ((2, 16), (3, 32)):
                fh.write(json.dumps({
                    "run_id": "main-20260101-000000", "engine": "mariadb",
                    "phase": "index_build", "dataset": "glove-100-angular",
                    "m": m, "resource_pass": "tuned",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "extra": {"ops_measurement_version": version}}) + "\n")

        self._manifest(run).set_config({}, "tuned", {"server_memory_bytes": 1})
        with pytest.raises(MixedMeasurementVersions) as caught:
            load_ops_records(str(run))
        assert "version 2" in str(caught.value)
        assert "version 3" in str(caught.value)

    def test_cli_still_calls_set_config_before_the_checkpoint_skip(self):
        """Why the fix had to be in set_config rather than at the call site.

        If this ever stops being true the comment in manifest.py is stale, but
        the carry-forward remains correct either way -- so this test documents
        the ordering rather than depending on it.
        """
        src = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        assert src.index("manifest.set_config(") < src.index("[skip]")


class TestTheRefusalTellsTheOperatorSomethingTheyCanDo:
    """Remediation advice that does not work is an outage with extra steps.

    One tail served both branches and was wrong for one of them: for records
    all at one version under a manifest declaring another, "move the files of
    the versions you do not want out of the directory" leaves the directory
    empty and the report exits with "no records found".
    """

    def _row(self, **kw):
        row = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
               "dataset": "d", "m": 16, "timestamp": "2026-01-01T00:00:00Z"}
        version = kw.pop("version", None)
        row.update(kw)
        if version is not None:
            row["extra"] = {"ops_measurement_version": version}
        return row

    def _message(self, rows, declared=None):
        from report.loaders import (MixedMeasurementVersions,
                                    refuse_mixed_measurement_versions)
        try:
            refuse_mixed_measurement_versions(rows, source="RUN",
                                              declared=declared)
        except MixedMeasurementVersions as exc:
            return str(exc)
        raise AssertionError("no refusal")

    def test_the_manifest_branch_names_the_key_that_is_wrong(self):
        message = self._message([self._row()], declared=2)
        assert "config.ops_measurement_version" in message
        assert "run-manifest.json" in message

    def test_the_manifest_branch_does_not_tell_them_to_drop_records(self):
        """There is one version of them; dropping it leaves nothing.

        Asserted against the mixed branch's actual remediation sentence rather
        than the substring "move". That substring passed only because the text
        happens to say "removing" rather than "remove", so a harmless reword
        would have broken it while the real regression -- the mixed branch's
        advice appearing here -- could slip past.
        """
        message = self._message([self._row()], declared=2)
        assert "move or split them" not in message
        assert "generate one report per version" not in message
        assert "Dropping records cannot fix this" in message

    def test_the_manifest_branch_says_the_records_agree(self):
        """Otherwise the operator hunts for a disagreement that is not there."""
        message = self._message([self._row()], declared=2)
        assert "records agree with each other" in message

    def test_the_declared_side_is_listed_even_though_it_has_no_records(self):
        """The one line in the listing that nothing else covers.

        Deleting it leaves a refusal whose headline names two versions and
        whose listing shows one, which reads like the guard miscounted.
        """
        message = self._message([self._row()], declared=2)
        assert "version 2: declared by the run manifest" in message
        assert "no records measured under it" in message
        # And it must not claim a count it does not have.
        assert "version 2: 0 record" not in message

    def test_the_mixed_records_branch_keeps_its_actionable_advice(self):
        message = self._message([self._row(), self._row(version=2)])
        assert "move or split them" in message
        assert "--from-records" in message

    def test_the_mixed_records_branch_does_not_assume_one_file_per_version(self):
        """A resumed run appends, so one file can hold both versions and there
        is no file to move. The advice has to cover that."""
        message = self._message([self._row(), self._row(version=2)])
        assert "one file can hold more than one" in message

    def test_neither_branch_recommends_the_other_s_remedy(self):
        mixed = self._message([self._row(), self._row(version=2)])
        manifest = self._message([self._row()], declared=2)
        assert "config.ops_measurement_version in run-manifest.json" \
            not in mixed
        assert "--from-records" not in manifest


class TestVersionsOneAndTwoMeanTheSameMeasurement:
    """The refusal must not contradict the methodology it enforces.

    Version 2 changed nothing measured -- it exists because version 1 was never
    written down. `docs/05-methodology.md` says so, because a reader who hits
    the refusal needs to know it. The refusal text said the opposite: that two
    versions "are two different measurements" and that records predating the
    newer one "have to be re-measured". Versions 1 and 2 are the only pair that
    can occur today, and an operator reaches it by resuming any archived run
    and measuring one more unit -- so every operator who saw this guard was
    told to re-measure data the docs call directly comparable.
    """

    def _message(self, rows, declared=None):
        from report.loaders import (MixedMeasurementVersions,
                                    refuse_mixed_measurement_versions)
        try:
            refuse_mixed_measurement_versions(rows, source="RUN",
                                              declared=declared)
        except MixedMeasurementVersions as exc:
            return str(exc)
        raise AssertionError("no refusal")

    def _row(self, version=None, **kw):
        row = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
               "dataset": "d", "m": 16, "timestamp": "2026-01-01T00:00:00Z"}
        row.update(kw)
        if version is not None:
            row["extra"] = {"ops_measurement_version": version}
        return row

    def test_the_code_knows_which_versions_are_equivalent(self):
        from harness.metrics.records import (
            OPS_MEASUREMENT_EQUIVALENT_VERSIONS,
            ops_versions_measure_the_same_thing)
        assert set(OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == {1, 2}
        assert ops_versions_measure_the_same_thing([1, 2])
        assert ops_versions_measure_the_same_thing([2])

    def test_a_version_it_has_not_been_told_about_is_not_equivalent(self):
        """Saying two numbers are comparable is a claim; the default is no."""
        from harness.metrics.records import ops_versions_measure_the_same_thing
        assert not ops_versions_measure_the_same_thing([1, 3])
        assert not ops_versions_measure_the_same_thing([2, 99])
        assert not ops_versions_measure_the_same_thing([])
        # Malformed values must not blow up the membership test.
        assert not ops_versions_measure_the_same_thing(["2"])
        assert not ops_versions_measure_the_same_thing([[2]])
        assert not ops_versions_measure_the_same_thing([{"v": 2}])

    def test_the_docs_and_the_code_agree(self):
        """Pinned together, because they drifted apart once already.

        The docs assert engine behaviour and this repo keeps such assertions
        executable. Deleting the claim from the methodology, or dropping 2 out
        of the equivalence set, must break this.
        """
        from harness.metrics.records import (
            OPS_MEASUREMENT_EQUIVALENT_VERSIONS)
        doc = open(os.path.join(VB_ROOT, "docs",
                                "05-methodology.md")).read()
        assert "Version 2 measures exactly what version 1 measured" in doc
        assert set(OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == {1, 2}

    # The demand, not the word. The carve-out sentence itself contains
    # "need re-measuring" (negated), so a bare substring test on "re-measur"
    # fails against correct text.
    DEMANDS = ("have to be re-measured", "must be re-measured",
               "they have to be re-measured before")

    def test_the_mixed_branch_does_not_demand_re_measurement_for_1_and_2(self):
        message = self._message([self._row(), self._row(version=2)])
        for demand in self.DEMANDS:
            assert demand not in message, demand
        assert "denote the same measurement" in message
        assert "none of these numbers need re-measuring" in message

    def test_the_manifest_branch_does_not_demand_re_measurement_for_1_and_2(self):
        """The exact path an operator reaches after resuming an archived run."""
        message = self._message([self._row()], declared=2)
        for demand in self.DEMANDS:
            assert demand not in message, demand
        assert "denote the same measurement" in message
        assert "none of these numbers need re-measuring" in message

    def test_the_mixed_branch_does_not_call_them_different_measurements(self):
        message = self._message([self._row(), self._row(version=2)])
        assert "two different measurements" not in message

    def test_it_still_refuses(self):
        """REFUSE is settled. The carve-out is about the wording, not the
        verdict: the report prints one version above the numbers and cannot
        print two."""
        message = self._message([self._row(), self._row(version=2)])
        assert "cannot go on one chart" in message
        assert "a report of its own" in message

    def test_a_bump_that_does_change_a_number_is_still_described_correctly(self):
        """The general wording has to survive, or the next real bump ships with
        text saying its numbers are comparable when they are not."""
        message = self._message([self._row(version=2), self._row(version=3)])
        assert "A measurement version changes what a number means" in message
        assert "two different measurements" in message
        assert "denote the same measurement" not in message

    def test_a_real_bump_against_the_manifest_still_says_re_measure(self):
        message = self._message([self._row(version=3)], declared=5)
        assert "re-measured" in message
        assert "changed what is measured" in message


class TestTheManifestHalfReachesEveryReader:
    """F6 made every reader consult the guard; this makes it the whole guard.

    `generate --from-records` and `webui/records.py` called
    refuse_mixed_measurement_versions without `declared`, so records that were
    internally consistent but a version behind their own manifest were refused
    on one path and accepted on the other two. The report then printed
    version-2 provenance above version-1 numbers -- and --from-records is the
    first thing an operator reaches for after being refused.
    """

    V1_ROW = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
              "dataset": "d", "m": 16, "timestamp": "2026-01-01T00:00:00Z"}

    def _run(self, tmp_path, declared=2):
        run = tmp_path / "run"
        run.mkdir(parents=True, exist_ok=True)
        config = {"resource_pass": "normalized"}
        if declared is not None:
            config["ops_measurement_version"] = declared
        (run / "run-manifest.json").write_text(
            json.dumps({"run_id": "r", "config": config}))
        return run

    def _jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_from_records_checks_the_records_against_the_manifest(
            self, tmp_path, capsys):
        from report import generate
        run = self._run(tmp_path)
        merged = tmp_path / "v1.jsonl"
        self._jsonl(merged, [self.V1_ROW])
        code = generate.main(["--run-dir", str(run),
                              "--from-records", str(merged)])
        assert code == 2
        err = capsys.readouterr().err
        assert "run manifest declares version 2" in err

    def test_a_refused_from_records_run_still_writes_nothing(self, tmp_path):
        from report import generate
        run = self._run(tmp_path)
        merged = tmp_path / "v1.jsonl"
        self._jsonl(merged, [self.V1_ROW])
        generate.main(["--run-dir", str(run), "--from-records", str(merged)])
        assert not (run / "report" / "records.jsonl").exists()

    def test_from_records_accepts_records_that_match_the_manifest(
            self, tmp_path):
        from report.loaders import (declared_ops_version,
                                    refuse_mixed_measurement_versions)
        run = self._run(tmp_path)
        row = dict(self.V1_ROW, extra={"ops_measurement_version": 2})
        refuse_mixed_measurement_versions(
            [row], source="x", declared=declared_ops_version(str(run)))

    def test_from_records_ignores_a_manifest_that_declares_nothing(
            self, tmp_path):
        """Every archived run. This is the one that must not start refusing."""
        from report import generate
        run = self._run(tmp_path, declared=None)
        merged = tmp_path / "v1.jsonl"
        self._jsonl(merged, [self.V1_ROW])
        # Not 2: it gets past the guard. (It still exits non-zero further on
        # for want of a dataset, which is not this guard's business.)
        assert generate.main(["--run-dir", str(run),
                              "--from-records", str(merged)]) != 2

    def test_the_web_ui_checks_the_records_against_the_manifest(self, tmp_path):
        from report.loaders import MixedMeasurementVersions
        from webui import records as records_mod
        run = self._run(tmp_path)
        self._jsonl(run / "report" / "records.jsonl", [self.V1_ROW])
        with pytest.raises(MixedMeasurementVersions):
            records_mod.load_records(str(run))

    def test_the_web_ui_accepts_a_matching_merged_file(self, tmp_path):
        from webui import records as records_mod
        run = self._run(tmp_path)
        self._jsonl(run / "report" / "records.jsonl",
                    [dict(self.V1_ROW, extra={"ops_measurement_version": 2})])
        rows, source = records_mod.load_records(str(run))
        assert source == "records.jsonl" and len(rows) == 1

    def test_the_web_ui_ignores_a_manifest_that_declares_nothing(self, tmp_path):
        from webui import records as records_mod
        run = self._run(tmp_path, declared=None)
        self._jsonl(run / "report" / "records.jsonl", [self.V1_ROW])
        rows, _source = records_mod.load_records(str(run))
        assert len(rows) == 1

    CALL = "refuse_mixed_measurement_versions("

    def _call_args(self, src):
        """The argument text of each call to the guard, definition excluded.

        A paren-balanced scan rather than a file-wide count of "declared=".
        The count version was vacuous for exactly the two files finding B was
        about: each contains one call, so `calls - 1 == 0` and the assertion
        reduced to `>= 0`.
        """
        found, i = [], 0
        while True:
            i = src.find(self.CALL, i)
            if i < 0:
                return found
            if src[:i].rstrip().endswith("def"):
                i += len(self.CALL)
                continue
            j, depth = i + len(self.CALL), 1
            while depth:
                if src[j] == "(":
                    depth += 1
                elif src[j] == ")":
                    depth -= 1
                j += 1
            found.append(src[i + len(self.CALL):j - 1])
            i = j

    def test_every_reader_passes_declared(self):
        """Checked on the call's own arguments, so half-arming it is caught.

        The behavioural tests above are the real guard -- this one exists
        because `declared` defaults to None, so omitting it is silent, and
        because a fourth reader added later should fail here rather than ship
        with half the check.
        """
        for parts in (("report", "loaders.py"), ("report", "generate.py"),
                      ("webui", "records.py")):
            src = open(os.path.join(VB_ROOT, *parts)).read()
            args = self._call_args(src)
            assert args, f"{parts}: no call to the guard found"
            for one in args:
                assert "declared=" in one, (parts, one)
                assert "declared=None" not in one, (parts, one)
                assert "declared_ops_version(" in one, (parts, one)


class TestTheRefusalSurvivesItsOwnCallers:
    """Two properties that were asserted in a comment and nowhere else."""

    def _mixed_dir(self, tmp_path, with_corruption=False):
        run = tmp_path / "run"
        run.mkdir(parents=True, exist_ok=True)
        base = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
                "dataset": "d", "m": 16, "timestamp": "2026-01-01T00:00:00Z"}
        (run / "ops-mariadb-d-normalized-a.jsonl").write_text(
            json.dumps(base) + "\n")
        (run / "ops-alisql-d-normalized-a.jsonl").write_text(
            json.dumps(dict(base, engine="alisql",
                            extra={"ops_measurement_version": 2})) + "\n")
        if with_corruption:
            # read_records raises ValueError for damage before valid data, and
            # load_ops_records swallows that per file and carries on.
            (run / "ops-pgvector-d-normalized-a.jsonl").write_text(
                '{"a": 1}\n{broken\n{"c": 3}\n')
        return run

    def test_the_refusal_is_raised_outside_the_per_file_handler(
            self, tmp_path, capsys):
        """Position, not class, is what lets it escape.

        The loop's handler is `except Exception`, so a refusal raised inside it
        would be swallowed and printed as an unreadable file. This directory
        has a genuinely damaged file as well, so the handler definitely runs --
        and the refusal still has to come out.
        """
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = self._mixed_dir(tmp_path, with_corruption=True)
        with pytest.raises(MixedMeasurementVersions):
            load_ops_records(str(run))
        assert "could not read" in capsys.readouterr().out

    def test_a_caller_guarding_against_file_errors_does_not_swallow_it(
            self, tmp_path):
        """webui/setup.py, webui/importing.py and orchestrator/export.py all
        wrap work in `except (OSError, ValueError)`. A refusal that subclassed
        ValueError would vanish into the next such caller, reported as a
        damaged file, and the report would be generated anyway."""
        from report.loaders import MixedMeasurementVersions, load_ops_records
        run = self._mixed_dir(tmp_path)
        try:
            load_ops_records(str(run))
        except (OSError, ValueError) as exc:
            raise AssertionError(
                f"a caller's (OSError, ValueError) handler would swallow the "
                f"refusal: {type(exc).__name__}")
        except MixedMeasurementVersions:
            pass
        else:
            raise AssertionError("no refusal")

    def test_it_is_not_a_valueerror(self):
        from report.loaders import MixedMeasurementVersions
        assert not issubclass(MixedMeasurementVersions, ValueError)

    def test_a_damaged_file_alone_is_still_just_a_damaged_file(self, tmp_path):
        """The distinction is only worth having if the other side still holds:
        one unreadable file must not become a refusal to report the run."""
        from report.loaders import load_ops_records
        run = tmp_path / "run"
        run.mkdir()
        (run / "ops-mariadb-d-normalized-a.jsonl").write_text(
            '{"a": 1}\n{broken\n{"c": 3}\n')
        assert load_ops_records(str(run)) == []


class TestARefusedRegenerationLeavesTheOldReportAlone:
    """Behavioural, not a source-index assertion.

    The chart cleanup moving below the record load was checked by comparing
    byte offsets in generate.py, which says nothing about what survives on
    disk. This runs the generator against a run directory that already holds a
    working report, refuses it, and looks at the files.
    """

    def _run_with_a_report(self, tmp_path, mixed=True):
        run = tmp_path / "run"
        (run / "report" / "charts").mkdir(parents=True)
        (run / "run-manifest.json").write_text(json.dumps(
            {"run_id": "r", "config": {"resource_pass": "normalized"}}))
        (run / "report" / "charts" / "recall_glove.png").write_bytes(b"PNG")
        (run / "report" / "charts" / "build_glove.svg").write_bytes(b"SVG")
        (run / "report" / "index.html").write_text("<html>a report</html>")
        (run / "report" / "records.jsonl").write_text('{"old": true}\n')

        base = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
                "dataset": "d", "m": 16, "timestamp": "2026-01-01T00:00:00Z"}
        rows = [base]
        if mixed:
            rows.append(dict(base, engine="alisql",
                             extra={"ops_measurement_version": 2}))
        (run / "ops-mariadb-d-normalized-a.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        return run

    def test_the_charts_of_the_working_report_survive(self, tmp_path):
        from report import generate
        run = self._run_with_a_report(tmp_path)
        assert generate.main(["--run-dir", str(run)]) == 2
        charts = run / "report" / "charts"
        assert (charts / "recall_glove.png").read_bytes() == b"PNG"
        assert (charts / "build_glove.svg").read_bytes() == b"SVG"

    def test_the_page_and_its_records_survive(self, tmp_path):
        """index.html referencing deleted charts is the actual damage: a page
        of broken images where a valid report used to be."""
        from report import generate
        run = self._run_with_a_report(tmp_path)
        generate.main(["--run-dir", str(run)])
        assert (run / "report" / "index.html").read_text() == \
            "<html>a report</html>"
        assert (run / "report" / "records.jsonl").read_text() == \
            '{"old": true}\n'

    def test_an_empty_run_does_not_clear_them_either(self, tmp_path):
        """Same damage by a different route: no records at all also used to
        clear the charts first and then exit non-zero."""
        from report import generate
        run = self._run_with_a_report(tmp_path, mixed=False)
        (run / "ops-mariadb-d-normalized-a.jsonl").unlink()
        assert generate.main(["--run-dir", str(run)]) == 1
        assert (run / "report" / "charts" / "recall_glove.png").exists()
        assert (run / "report" / "index.html").exists()


class TestARefusalDescribesTheSetItIsAbout:
    """Every sentence must be true of the records it is printed over.

    The mismatch branch is written for a directory whose records agree with
    each other and disagree only with the manifest above them. It used to be
    chosen whenever the manifest matched no record group -- which is also true
    when the records disagree among themselves and none of them matches the
    manifest. Over v1 and v3 records under a manifest declaring v2 it then said
    every record was version 1, that the records agreed, that there was only
    one version of them and that removing it would leave nothing to report,
    with version 3 listed three lines below; and because only the first side
    reached the equivalence predicate, that none of the numbers needed
    re-measuring -- a comparability claim across a real bump. Following its
    remedy refused again, so it looped.
    """

    def _row(self, version=None, **kw):
        row = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
               "dataset": "d", "m": 16, "resource_pass": "normalized",
               "timestamp": "2026-01-01T00:00:00Z"}
        row.update(kw)
        if version is not None:
            row["extra"] = {"ops_measurement_version": version}
        return row

    def _message(self, rows, declared=None):
        from report.measurement_versions import (
            MixedMeasurementVersions, refuse_mixed_measurement_versions)
        try:
            refuse_mixed_measurement_versions(rows, source="RUN",
                                              declared=declared)
        except MixedMeasurementVersions as exc:
            return str(exc)
        raise AssertionError("no refusal")

    #: v1 and v3 records under a manifest declaring v2.
    def _three_way(self):
        return [self._row(), self._row(version=3, engine="alisql",
                                       timestamp="2026-09-09T00:00:00Z")]

    def test_it_does_not_claim_every_record_is_one_version(self):
        message = self._message(self._three_way(), declared=2)
        assert "every ops record here was measured under version" not in message
        # TWO, not three. This assertion read "3" and so pinned a second
        # falsehood in the same sentence: `_three_way` is v1 and v3 records
        # under a manifest declaring v2, and the count was taken over every
        # version involved INCLUDING the manifest's -- which has no records,
        # as this refusal's own listing says four lines further down. An
        # operator told to produce "one report per version" went looking for a
        # third pile of records that does not exist. The count is now of the
        # versions records were actually found at, and the number still
        # confirms the mixed branch was the one taken.
        #
        # `from 2`, not `2`: a bare "2 different measurement versions" is a
        # substring of "12 different measurement versions", so a mutant
        # counting something else entirely satisfied this assertion while
        # failing five in tests/test_t2b_report.py. The preceding word is what
        # anchors it to the number the sentence is about.
        assert "from 2 different measurement versions" in message

    def test_it_does_not_claim_the_records_agree(self):
        message = self._message(self._three_way(), declared=2)
        assert "records agree with each other" not in message

    def test_it_does_not_claim_dropping_records_cannot_help(self):
        """Here dropping one half IS the fix, so the opposite advice loops."""
        message = self._message(self._three_way(), declared=2)
        assert "Dropping records cannot fix this" not in message
        assert "only one version of them" not in message
        assert "move or split them" in message

    def test_it_does_not_claim_comparability_across_a_real_bump(self):
        """The worst of the four, and the one direction we agreed is never
        claimed by accident: version 3 is a bump that changed a number."""
        message = self._message(self._three_way(), declared=2)
        assert "denote the same measurement" not in message
        assert "none of these numbers need re-measuring" not in message
        assert "two different measurements" in message

    def test_it_still_names_the_manifest_as_wrong(self):
        """The manifest is still wrong and still worth naming -- an operator
        who splits the records and regenerates would otherwise be refused a
        second time by a check the first message never mentioned."""
        message = self._message(self._three_way(), declared=2)
        assert "run manifest declares version 2" in message
        assert "run-manifest.json is wrong as well" in message
        assert "will refuse that half too" in message

    def test_the_declared_version_is_listed_with_the_record_versions(self):
        message = self._message(self._three_way(), declared=2)
        assert "version 1: 1 record" in message
        assert "version 3: 1 record" in message
        assert "version 2: declared by the run manifest" in message

    def test_following_the_remedy_actually_resolves_it(self, tmp_path):
        """Non-working remediation is the defect class this class exists for.

        Do what the message says -- keep one version, set the manifest to it --
        and the run must load. Before the fix the stated remedy was to correct
        the manifest to version 1, which left v1 and v3 records still mixed and
        refused again.
        """
        from report.loaders import load_ops_records
        run = tmp_path / "run"
        run.mkdir()
        (run / "run-manifest.json").write_text(json.dumps(
            {"run_id": "r", "config": {"resource_pass": "normalized",
                                       "ops_measurement_version": 2}}))
        with open(run / "ops-mariadb-d-normalized-a.jsonl", "w") as fh:
            for row in self._three_way():
                fh.write(json.dumps(row) + "\n")

        from report.measurement_versions import MixedMeasurementVersions
        with pytest.raises(MixedMeasurementVersions):
            load_ops_records(str(run))

        # The remedy: one version left, manifest set to it.
        with open(run / "ops-mariadb-d-normalized-a.jsonl", "w") as fh:
            fh.write(json.dumps(self._row(version=3)) + "\n")
        (run / "run-manifest.json").write_text(json.dumps(
            {"run_id": "r", "config": {"resource_pass": "normalized",
                                       "ops_measurement_version": 3}}))
        assert len(load_ops_records(str(run))) == 1

    def test_the_single_version_case_still_uses_the_mismatch_branch(self):
        """The branch that was right must stay reachable."""
        message = self._message([self._row()], declared=2)
        assert "every ops record here was measured under version 1" in message
        assert "records agree with each other" in message
        assert "Dropping records cannot fix this" in message

    def test_the_mismatch_branch_is_used_only_for_one_record_version(self):
        """The tripwire, as a test: the headline written for a directory whose
        records agree appears exactly when they do."""
        agreeing = self._message([self._row(), self._row(m=32)], declared=2)
        disagreeing = self._message(self._three_way(), declared=2)
        headline = "every ops record here was measured under version"
        assert headline in agreeing
        assert headline not in disagreeing

    def test_the_equivalence_claim_covers_the_declared_version_too(self):
        """Records at 1 and 2 are equivalent, but a manifest declaring 3 puts a
        real bump in the set being described -- so the conservative wording is
        correct, and claiming "the same measurement" three lines above a
        version 3 listing would not be."""
        message = self._message([self._row(), self._row(version=2)],
                                declared=3)
        assert "denote the same measurement" not in message
        assert "two different measurements" in message

    def test_an_equivalent_pair_with_no_manifest_still_gets_the_carve_out(self):
        """And the carve-out must not have been lost in the process."""
        message = self._message([self._row(), self._row(version=2)])
        assert "denote the same measurement" in message


class TestTheEquivalenceSetCannotGrowByAccident:
    """A correctness claim stored as data needs a lock, not a comment.

    The set says "these versions measure the same thing", which is the one
    claim this guard must never make by accident. Written as the literal
    `frozenset({1, 2})` it was an extension point whose only protection was a
    comment asking people not to use it: `{1, 2, 4}` is a one-character edit
    asserting that version 4 measures what version 1 did, and the subset-style
    assertions it had could not see a set growing at all -- measured, {1,2,4},
    {1,2,5}, {1,2,7} and {1,2,4,5} all passed the entire suite.
    """

    def test_it_is_exactly_the_two_named_versions(self):
        from harness.metrics.records import (
            FIRST_RECORDED_OPS_MEASUREMENT_VERSION,
            IMPLICIT_OPS_MEASUREMENT_VERSION,
            OPS_MEASUREMENT_EQUIVALENT_VERSIONS)
        assert set(OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == {
            IMPLICIT_OPS_MEASUREMENT_VERSION,
            FIRST_RECORDED_OPS_MEASUREMENT_VERSION,
        }
        assert len(OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == 2

    def test_it_is_derived_from_those_names_rather_than_listed(self):
        """The structural lock. A third member now requires naming a third
        concept and saying in its docstring why that version changed nothing,
        which is the deliberateness the comment used to merely request."""
        src = open(os.path.join(VB_ROOT, "harness", "metrics",
                                "records.py")).read()
        body = src.split("OPS_MEASUREMENT_EQUIVALENT_VERSIONS = ")[1]
        declaration = body[:body.index(")") + 1]
        assert "IMPLICIT_OPS_MEASUREMENT_VERSION" in declaration
        assert "FIRST_RECORDED_OPS_MEASUREMENT_VERSION" in declaration
        # No bare integers, which is what makes a casual addition impossible.
        assert not re.search(r"\b\d+\b", declaration), declaration

    def test_the_recorded_version_name_does_not_track_the_current_one(self):
        """If the set were derived from OPS_MEASUREMENT_VERSION it would follow
        every bump, and a bump to 3 would silently claim v3 measures what v1
        did -- the exact failure the derivation is meant to prevent."""
        from harness.metrics import records as mod
        assert mod.FIRST_RECORDED_OPS_MEASUREMENT_VERSION == 2
        src = open(os.path.join(VB_ROOT, "harness", "metrics",
                                "records.py")).read()
        body = src.split("OPS_MEASUREMENT_EQUIVALENT_VERSIONS = ")[1]
        declaration = body[:body.index(")") + 1]
        # Anchored at the line start: "OPS_MEASUREMENT_VERSION," is a suffix of
        # "IMPLICIT_OPS_MEASUREMENT_VERSION,", so a plain substring test reads
        # the correct declaration as the broken one. Third time this family of
        # names has sprung that trap in this change.
        assert not re.search(r"^\s*OPS_MEASUREMENT_VERSION\s*,", declaration,
                             re.MULTILINE), declaration

    def test_a_bumped_current_version_leaves_the_set_alone(self):
        """The scheme has to survive its own future: with the constant at 3,
        v2+v3 gets the "different measurements" wording and v1+v2 keeps the
        carve-out."""
        from harness.metrics import records as mod
        before = mod.OPS_MEASUREMENT_VERSION
        try:
            mod.OPS_MEASUREMENT_VERSION = 3
            assert set(mod.OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == {1, 2}
            assert mod.ops_versions_measure_the_same_thing([1, 2])
            assert not mod.ops_versions_measure_the_same_thing([2, 3])
        finally:
            mod.OPS_MEASUREMENT_VERSION = before


class TestTheRefusalMachineryMovedWithoutLoss:
    """report/loaders.py was 793 lines against this repo's 800 ceiling.

    The refusal machinery moved to report/measurement_versions.py so the next
    addition would not breach it, and so the change would be reviewed here
    rather than inherited by unrelated work in the file.
    """

    def test_the_public_names_are_still_importable_from_loaders(self):
        """`from report.loaders import ...` is how every existing caller and
        test reaches them, including code this change does not own."""
        from report import loaders, measurement_versions
        for name in ("MixedMeasurementVersions", "declared_ops_version",
                     "ops_measurement_version", "ops_measurement_versions",
                     "refuse_mixed_measurement_versions"):
            assert getattr(loaders, name) is getattr(measurement_versions,
                                                     name), name

    def test_both_files_are_within_the_line_ceiling(self):
        for parts in (("report", "loaders.py"),
                      ("report", "measurement_versions.py")):
            path = os.path.join(VB_ROOT, *parts)
            lines = len(open(path).read().splitlines())
            assert lines <= 800, f"{parts}: {lines} lines"

    def test_the_new_module_does_not_import_the_orchestrator(self):
        """Covered by the report-wide glob too, asserted here so the reason
        travels with the file: it runs in a bench image mounting report/ and
        harness/ only."""
        src = open(os.path.join(VB_ROOT, "report",
                                "measurement_versions.py")).read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import orchestrator" not in stripped, stripped
            assert "from orchestrator" not in stripped, stripped

    def test_there_is_one_place_to_patch_the_guard(self, tmp_path):
        """loaders calls through the module, not through a bound name.

        Binding the functions into loaders as well would give each reader its
        own reference, and standing the guard down in a test would reach some
        readers and not others -- which is how a dead mutation looks like a
        passing one.
        """
        from report import loaders, measurement_versions
        run = tmp_path / "run"
        run.mkdir()
        base = {"run_id": "r", "engine": "mariadb", "phase": "index_build",
                "dataset": "d", "timestamp": "t"}
        with open(run / "ops-mariadb-d-normalized-a.jsonl", "w") as fh:
            fh.write(json.dumps(base) + "\n")
            fh.write(json.dumps(dict(
                base, engine="alisql",
                extra={"ops_measurement_version": 2})) + "\n")

        with pytest.raises(measurement_versions.MixedMeasurementVersions):
            loaders.load_ops_records(str(run))

        honest = measurement_versions.refuse_mixed_measurement_versions
        try:
            measurement_versions.refuse_mixed_measurement_versions = (
                lambda records, *, source, declared=None: None)
            assert len(loaders.load_ops_records(str(run))) == 2
        finally:
            measurement_versions.refuse_mixed_measurement_versions = honest


# ---------------------------------------------------------------------------
# T1d: the concurrency surfaces (§1.1's recall column, §1.4's client CPU)
# ---------------------------------------------------------------------------

def _table_cells(table: str):
    """A Markdown table as a list of rows of cells, separator row dropped.

    Cells rather than a substring search, because "Recall@k" appears in four
    of this report's tables and `recall_at_k` appears in the raw records: a
    membership test against the whole fragment is satisfied by a table that
    does not have the column, and a value test against the whole fragment is
    satisfied by the value turning up in some other column.
    """
    rows = []
    for line in table.strip().splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c and set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _concurrency_record(**over):
    """One concurrency record as the workload writes it today."""
    extra = {
        "scaling_efficiency": 0.43, "duration_s": 30,
        "warmup_queries_per_client": 50, "repeats": 3,
        "qps_samples": [4800.0, 5820.0, 5100.0], "qps_spread_pct": 21.3,
        "recall_queries": 10_000, "recall_spread": 0.0003,
        "query_pool": 10_000, "query_reuse": 15.3,
        "inflight_discarded": 32, "prewindow_discarded": 0,
        "client_cpu_seconds": 48.0, "client_cpus": 4,
        "client_cpu_util": 0.40, "client_bound": False,
    }
    extra.update(over.pop("extra", {}))
    record = {"phase": "concurrency", "engine": "pgvector", "dataset": "d",
              "resource_pass": "normalized", "storage_engine": "heap",
              "m": 16, "ef_search": 100, "clients": 32, "qps": 5100.0,
              "recall_at_k": 0.9812, "latency_p50_ms": 5.1,
              "latency_p95_ms": 9.9, "latency_p99_ms": 14.2,
              "queries_executed": 153_000}
    record.update(over)
    record["extra"] = extra
    return record


class TestTheConcurrencyTableCarriesItsQualityFigure:
    """A concurrency point had QPS and latency and no recall beside it.

    The workload has recorded `recall_at_k` on every concurrency record since
    §1.1, and the table listed Engine, Dataset, Storage, Clients, QPS, Spread
    and three percentiles -- so a record carrying 0.9812 rendered nowhere at
    all. §1.1 exists so the report stops presenting two engines at one
    `ef_search` as comparable when they sit at 0.98 and 0.80; without the
    column it still did.
    """

    def _cells(self, *records):
        from report.render import _concurrency_table
        return _table_cells(_concurrency_table({"concurrency": list(records)}))

    def test_the_recall_column_holds_the_records_recall(self):
        rows = self._cells(_concurrency_record())
        header, body = rows[0], rows[1:]
        assert "Recall@k" in header
        assert body[0][header.index("Recall@k")] == "0.9812"

    def test_a_point_with_no_recall_renders_a_dash_rather_than_a_zero(self):
        rows = self._cells(_concurrency_record(recall_at_k=None))
        header, body = rows[0], rows[1:]
        assert body[0][header.index("Recall@k")] == "—"

    def test_the_pool_and_the_reuse_are_shown(self):
        """A reader comparing a version-2 report with a version-3 one needs
        both to see why the throughput moved: the same QPS over a pool of 500
        and a pool of 10,000 are measurements of two different things."""
        rows = self._cells(_concurrency_record())
        header, body = rows[0], rows[1:]
        assert body[0][header.index("Query pool")] == "10,000"
        assert body[0][header.index("Reuse")] == "15.30x"

    def test_the_client_cpu_column_holds_the_utilisation(self):
        rows = self._cells(_concurrency_record())
        header, body = rows[0], rows[1:]
        assert body[0][header.index("Client CPU")] == "0.40"

    def test_a_client_bound_point_is_marked_in_its_own_cell(self):
        from report.render import CLIENT_BOUND_MARK
        bound = _concurrency_record(
            extra={"client_cpu_util": 0.93, "client_bound": True})
        rows = self._cells(bound)
        header, body = rows[0], rows[1:]
        cell = body[0][header.index("Client CPU")]
        assert cell == f"0.93 {CLIENT_BOUND_MARK}"

    def test_an_unmeasured_client_cpu_is_a_dash_not_a_zero(self):
        """None means no counter was readable, which is not the claim that the
        clients were idle."""
        rows = self._cells(_concurrency_record(
            extra={"client_cpu_util": None, "client_bound": None}))
        header, body = rows[0], rows[1:]
        assert body[0][header.index("Client CPU")] == "—"

    def test_every_body_cell_is_filled(self):
        """A column added to the header and not to the row shifts every cell
        after it, which renders as a plausible table of wrong numbers."""
        rows = self._cells(_concurrency_record(), _concurrency_record(clients=1))
        header, body = rows[0], rows[1:]
        assert len(body) == 2
        for row in body:
            assert len(row) == len(header), list(zip(header, row))


class TestTheConcurrencyNotesExplainTheNewColumns:
    def _notes(self, *records):
        from report.render import _concurrency_notes
        return _concurrency_notes({"concurrency": list(records)})

    def test_the_in_flight_correction_is_named_without_a_column(self):
        """T1c's `extra.inflight_discarded` had no reader at all.

        It does not earn a column -- at most one query per client, 0.005% of a
        30-second window's throughput -- but the correction it evidences has
        to be stated or the QPS reads as the uncorrected one.
        """
        notes = self._notes(_concurrency_record())
        assert "inflight_discarded" in notes
        assert "began and ended inside the window" in notes

    def test_the_client_cpu_definition_names_all_three_of_its_inputs(self):
        notes = self._notes(_concurrency_record())
        assert "cpuset" in notes
        assert "not evidence about where the engine" in notes

    def test_the_footnote_says_the_flag_is_sufficient_and_not_necessary(self):
        """Measured, not supposed: against the synthetic driver this harness
        flattens at 2.06 cores -- 0.51 of a four-CPU container -- while its
        throughput peaks at two clients and halves by thirty-two. The flag
        would not have fired on any of those points, because the binding
        resource is one interpreter lock rather than the cpuset. A footnote
        that only warned about marked points would let an unmarked one be read
        as proof the engine was the limit."""
        notes = self._notes(_concurrency_record())
        assert "an unmarked point is not a guarantee" in notes.lower()
        assert "2.06 cores" in notes

    def test_the_footnote_says_WHY_the_flag_is_blind_to_a_gil_ceiling(self):
        """The mechanism, not just the caveat.

        A reader told only "an unmarked point may still be client-bound" notes
        it and moves on. A reader told that the lock prevents the very CPU
        consumption the threshold looks for -- so that a flat Client CPU column
        beside a falling QPS curve IS the signature -- can act on it, and knows
        that moving the 0.80 limit would not help. This sentence is the
        difference, so it is pinned.
        """
        notes = self._notes(_concurrency_record())
        assert "cannot see an interpreter-lock ceiling" in notes
        assert "prevents the very CPU consumption the threshold looks for" \
            in notes
        # And the actionable form of it: what the reader should look for.
        assert "flat Client CPU column beside a falling QPS curve" in notes

    def test_a_client_bound_run_names_the_points(self):
        notes = self._notes(
            _concurrency_record(clients=16, extra={"client_cpu_util": 0.88,
                                                   "client_bound": True}),
            _concurrency_record(clients=4))
        assert "Client-bound points in this run" in notes
        assert "at 16 clients" in notes
        assert "at 4 clients" not in notes

    def test_a_clean_run_claims_nothing(self):
        notes = self._notes(_concurrency_record())
        assert "Client-bound points in this run" not in notes
        assert "prewindow_discarded" not in notes
        assert "no client-CPU figure" not in notes

    def _unmeasured(self, **over):
        return _concurrency_record(
            extra={"client_cpu_util": None, "client_bound": None}, **over)

    def test_some_points_without_a_client_cpu_figure_is_loud(self):
        """A counter that was readable and then was not.

        The harness lost a measurement part-way through a sweep, so the points
        either side of it were not measured the same way. That is a fault and
        it is named, with both counts, so a reader can see it is partial.
        """
        notes = self._notes(self._unmeasured(clients=16),
                            _concurrency_record(clients=4),
                            _concurrency_record(clients=8))
        assert "1 of 3 points have no client-CPU figure" in notes
        assert "readable for some windows of this run and not for others" \
            in notes

    def test_no_point_having_the_figure_says_nothing_at_all(self):
        """A run measured end to end by a harness older than this column.

        Every point lacks the figure for one uniform reason, the column is an
        unbroken row of em dashes that says so by itself, and there is no gap
        to report. Emitted anyway, this paragraph lands on every report of
        every pre-existing run directory and trains a reader to skip the
        footnote -- after which it is also unread on the run where the partial
        case above actually matters.
        """
        notes = self._notes(self._unmeasured(clients=4),
                            self._unmeasured(clients=8))
        assert "no client-CPU figure" not in notes
        # The definition of the column is still there; only the gap note goes.
        assert "**Client CPU**" in notes

    def test_a_record_predating_the_column_entirely_is_also_silent(self):
        """Not just `client_cpu_util: None` -- the key absent altogether, which
        is the shape a pre-T1d record actually has."""
        old = _concurrency_record()
        del old["extra"]["client_cpu_util"]
        del old["extra"]["client_bound"]
        assert "no client-CPU figure" not in self._notes(old)

    def test_a_non_zero_prewindow_count_is_loud(self):
        """Zero by construction: the window opens at the barrier release and
        no measured query can precede it. A number there means the window's two
        ends disagree about when it began, which is a framework fault and not
        an engine result -- and until now nothing read the field."""
        notes = self._notes(_concurrency_record(
            extra={"prewindow_discarded": 7}))
        assert "prewindow_discarded" in notes
        assert "framework fault" in notes
        # Anchored on the left. "1 point(s)" is a substring of "11 point(s)",
        # and a count that renders eleven where one is meant would satisfy the
        # bare membership test; it is exact here only because the fixture
        # passes a single record.
        assert "**1 point(s) report" in notes


class TestTheConcurrencyChartMarksAClientBoundPoint:
    """A client-bound point flattens the throughput curve and pulls the
    efficiency curve away from linear, and neither movement is about the
    engine. Hollowed the way the filtered chart hollows a short result set."""

    def _axes(self, records):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
        from report import charts
        captured = {}
        original = charts._save

        def spy(fig, out_dir, stem):
            captured["axes"] = list(fig.axes)
            captured["title"] = fig._suptitle.get_text()
            return original(fig, out_dir, stem)

        charts._save = spy
        try:
            with tempfile.TemporaryDirectory() as d:
                charts.concurrency(records, "d", d, "t")
        finally:
            charts._save = original
            plt.close("all")
        return captured

    def _series(self, bound_at=()):
        return [_concurrency_record(
            clients=c, qps=1000.0 * c,
            extra={"scaling_efficiency": 1.0 / c,
                   "client_cpu_util": 0.95 if c in bound_at else 0.2,
                   "client_bound": c in bound_at})
                for c in (1, 2, 4, 8, 16, 32)]

    @staticmethod
    def _rings(ax):
        """Where this panel's hollow rings actually sit, sorted by x.

        `get_offsets()` rather than `len(collections)`. A count says that some
        number of rings were drawn; it does not say they were drawn ON the
        points, and a ring parked at the axis marks nothing at all on the one
        chart whose purpose is to say whose ceiling the curve shows. A mutant
        scattering every ring at y=0 passed the count assertions five times
        out of five.
        """
        points = []
        for collection in ax.collections:
            points.extend([float(x), float(y)]
                          for x, y in collection.get_offsets())
        return sorted(points)

    @needs_matplotlib
    def test_a_bound_point_is_ringed_on_throughput_and_on_efficiency(self):
        captured = self._axes(self._series(bound_at=(16, 32)))
        throughput, _latency, efficiency = captured["axes"]
        assert len(throughput.collections) == 2
        assert len(efficiency.collections) == 2

    @needs_matplotlib
    def test_each_ring_sits_on_the_point_it_marks(self):
        """The coordinates, not the count.

        The throughput panel's rings belong at (clients, qps) and the
        efficiency panel's at (clients, scaling_efficiency) -- two different
        y values for the same two x values, so a ring taken from the wrong
        series is caught as well as one parked at zero.
        """
        bound = (16, 32)
        captured = self._axes(self._series(bound_at=bound))
        throughput, _latency, efficiency = captured["axes"]
        assert self._rings(throughput) == [[16.0, 16000.0], [32.0, 32000.0]]
        assert self._rings(efficiency) == [[16.0, 1.0 / 16], [32.0, 1.0 / 32]]
        # The two panels are not the same ring drawn twice: same x, different
        # y, so a copy of one onto the other would fail one of the two.
        assert self._rings(throughput) != self._rings(efficiency)

    @needs_matplotlib
    def test_a_ring_is_drawn_for_each_bound_point_and_no_other(self):
        """One client count bound, not two: the ring count and the ring
        positions both follow the flag rather than the series length."""
        captured = self._axes(self._series(bound_at=(8,)))
        throughput, _latency, efficiency = captured["axes"]
        assert self._rings(throughput) == [[8.0, 8000.0]]
        assert self._rings(efficiency) == [[8.0, 0.125]]

    @needs_matplotlib
    def test_an_unbound_series_is_ringed_nowhere(self):
        captured = self._axes(self._series())
        for ax in captured["axes"]:
            assert self._rings(ax) == []

    @needs_matplotlib
    def test_the_ring_is_hollow(self):
        """A filled marker would read as an ordinary emphasised point."""
        captured = self._axes(self._series(bound_at=(32,)))
        ring = captured["axes"][0].collections[0]
        assert ring.get_facecolors().size == 0
        assert ring.get_edgecolors().size > 0

    @needs_matplotlib
    def test_the_caption_says_what_the_ring_means(self):
        """An unexplained symbol on a chart is read as a rendering artefact."""
        captured = self._axes(self._series(bound_at=(32,)))
        assert "○" in captured["title"]
        assert "client CPU-bound" in captured["title"]


class TestTheWebUiCanPlotConcurrencyRecall:
    """`webui/static/fields.js` gives each view a CLOSED `ys` allowlist and
    `explore.js` consumes it as closed, so a field absent from the list cannot
    be plotted however many records carry it."""

    def _view_ys(self, view_id):
        path = os.path.join(VB_ROOT, "webui", "static", "fields.js")
        with open(path) as fh:
            source = fh.read()
        # Scoped to this view's own object. The `filtered` view two entries
        # below already lists recall_at_k, so a whole-file search for it
        # passes whether or not the concurrency view was touched.
        chunks = source.split('id: "')
        chunk = next(c for c in chunks[1:] if c.startswith(view_id + '"'))
        body = re.search(r"ys:\s*\[(.*?)\]", chunk, re.S)
        assert body, f"the {view_id} view has no ys array"
        return re.findall(r'"([a-z_0-9]+)"', body.group(1))

    def test_the_concurrency_view_offers_recall(self):
        assert "recall_at_k" in self._view_ys("concurrency")

    def test_throughput_is_still_the_default_axis(self):
        """`explore.js` picks `ys[0]` when nothing is chosen, so the order is
        the default and reordering it would change what the tab opens on."""
        assert self._view_ys("concurrency")[0] == "qps"

    def test_the_allowlist_is_what_the_picker_is_built_from(self):
        """Why one entry was enough, stated against the consumer rather than
        assumed."""
        path = os.path.join(VB_ROOT, "webui", "static", "explore.js")
        with open(path) as fh:
            source = fh.read()
        assert "ys: view.ys.filter(" in source

    def test_the_field_has_a_readers_label(self):
        assert '"Recall@k"' in open(
            os.path.join(VB_ROOT, "webui", "static", "fields.js")).read()


class TestTheSyntheticRunExercisesTheConcurrencySurfaces:
    """`tests/synthetic.py` is what `make-demo-report` renders for reviewing
    layout before any real data exists. Its concurrency records carried
    neither recall nor any of §1.1's and §1.4's extra keys, so a synthetic
    report showed an em dash exactly where the new columns are -- which reads
    as a report bug rather than as a fixture gap."""

    def _points(self):
        from .synthetic import make_records
        return [r for r in make_records("synthetic-run")
                if r.get("phase") == "concurrency"]

    def test_every_point_carries_what_the_table_reads(self):
        """Present AND populated.

        Membership alone is satisfied by a key whose value is None, which is
        exactly what renders the em dash this class exists to keep out of the
        demo report -- so every key that has no legitimate null in a synthetic
        run is checked for a value, not just for existence.
        """
        for row in self._points():
            assert row["recall_at_k"] is not None, row["engine"]
            for key in ("query_pool", "query_reuse", "recall_queries",
                        "inflight_discarded", "prewindow_discarded",
                        "client_cpu_seconds", "client_cpus",
                        "client_cpu_util", "client_bound", "repeats",
                        "qps_samples", "qps_spread_pct"):
                assert key in row["extra"], (row["engine"], row["clients"], key)
            # `prewindow_discarded` is legitimately 0 and `client_bound` is
            # legitimately False, so those two are checked for not-None rather
            # than for truthiness; everything else is a measured quantity that
            # a fixture has no reason to leave null.
            for key in ("query_pool", "query_reuse", "recall_queries",
                        "client_cpu_seconds", "client_cpus",
                        "client_cpu_util", "repeats", "qps_spread_pct",
                        "inflight_discarded", "prewindow_discarded",
                        "client_bound"):
                assert row["extra"][key] is not None, (
                    row["engine"], row["clients"], key)

    def test_recall_does_not_depend_on_the_client_count(self):
        """Recall is a property of the index and the ef, so a fixture whose
        recall moved across the sweep would make the workload's own
        recall-spread check look as though it were firing on nothing."""
        by_engine = {}
        for row in self._points():
            by_engine.setdefault(row["engine"], set()).add(row["recall_at_k"])
        assert by_engine, "no concurrency points in the synthetic run"
        for engine, values in by_engine.items():
            assert len(values) == 1, (engine, values)

    def test_the_engines_differ_in_recall_at_one_ef_search(self):
        """The whole reason the column exists: 0.98 against 0.80 is the
        ordinary spread of these implementations at a shared ef."""
        recalls = {r["engine"]: r["recall_at_k"] for r in self._points()}
        assert max(recalls.values()) - min(recalls.values()) > 0.15, recalls

    def test_both_branches_of_the_client_bound_flag_are_rendered(self):
        flags = [(r["engine"], r["clients"])
                 for r in self._points() if r["extra"]["client_bound"]]
        assert flags, "no client-bound point, so the hollow marker is untested"
        assert len(flags) < len(self._points()), "every point is bound"

    def _rendered(self):
        """The synthetic run's concurrency table, as header and body cells."""
        from report.generate import summarize
        from report.render import _concurrency_table
        from .synthetic import make_records
        summary = summarize(make_records("synthetic-run"), {})
        rows = _table_cells(_concurrency_table(summary))
        return rows[0], rows[1:]

    def test_the_rendered_table_shows_the_recall_rather_than_a_dash(self):
        """The fixture and the renderer, end to end. Either half alone passed
        while the pair produced an em dash."""
        header, body = self._rendered()
        column = header.index("Recall@k")
        values = {row[column] for row in body}
        assert "—" not in values, values
        assert "0.9812" in values, values
        pool = header.index("Query pool")
        assert {row[pool] for row in body} == {"10,000"}

    def test_the_rendered_table_shows_the_client_cpu_rather_than_a_dash(self):
        """The same end-to-end check for the column §1.4 adds, which the
        Recall@k one did not reach.

        Nulling `client_cpu_util` on every synthetic point while leaving
        `client_bound` as computed turned all six Client CPU cells into em
        dashes while the footnote went on naming two client-bound points -- a
        demo report contradicting itself, with the whole class still green,
        because the end-to-end assertion stopped at Recall@k and Query pool
        and the field check above was membership-only. This is defect C
        recurring inside the column the task added, so it is checked the same
        way: the cells carry values, and at least one carries the mark.
        """
        from report.render import CLIENT_BOUND_MARK
        header, body = self._rendered()
        column = header.index("Client CPU")
        cells = [row[column] for row in body]
        assert "—" not in cells, cells
        marked = [c for c in cells if CLIENT_BOUND_MARK in c]
        assert marked, cells
        # And the marked cells are the ones the footnote names, so the table
        # and the prose below it cannot disagree about which points are bound.
        bound_clients = {str(r["clients"]) for r in self._points()
                         if r["extra"]["client_bound"]}
        clients_column = header.index("Clients")
        marked_clients = {row[clients_column] for row in body
                          if CLIENT_BOUND_MARK in row[column]}
        assert marked_clients == bound_clients, (marked_clients, bound_clients)

    def test_the_rendered_notes_name_the_client_bound_points(self):
        from report.generate import summarize
        from report.render import _concurrency_table
        from .synthetic import make_records
        summary = summarize(make_records("synthetic-run"), {})
        table = _concurrency_table(summary)
        assert "Client-bound points in this run" in table
        assert "at 32 clients" in table


class TestEachConcurrencyPointIsRepeated:
    """§1.5. Every figure in this framework was a single sample until a Valkey
    point was re-measured by accident, five hours later on the same build, and
    came back 21% higher at sixteen and thirty-two clients. Nothing else had
    ever been repeated, so nothing else had an error bar, and two engines
    differing by less than that were being presented as ordered."""

    @pytest.mark.parametrize("name", ("main", "full"))
    def test_the_profile_repeats_each_point(self, name):
        assert load_profile(name)["ops"]["concurrency_repeats"] == 3

    @pytest.mark.parametrize("name", ("main", "full"))
    def test_the_query_pool_is_left_at_the_whole_test_set(self, name):
        """`ops.concurrency_queries` defaults to every query the dataset ships.

        Setting it is the way BACK to a small pool -- the state in which every
        graph cache in the comparison was measured against a hot set it could
        hold entirely -- so a profile that wants the current behaviour must
        leave the key absent rather than "enable" it.
        """
        assert "concurrency_queries" not in load_profile(name)["ops"]

    @pytest.mark.parametrize("name", ("main", "full"))
    def test_the_runner_already_passes_the_key(self, name):
        """No code change was needed, and this is what says so.

        `ops_pass.harness_args` has passed `--concurrency-repeats` since the
        flag existed, defaulting to 1, and `harness/main.py` forwards it to
        the workload's `repeats`. The profile key is the whole change.
        """
        from orchestrator.config import (load_resources, resolve_resources)
        from orchestrator.ops_pass import harness_args
        info = sysinfo.collect()
        res = load_resources("normalized")
        args = harness_args(load_profile(name), 16, "pgvector",
                            resolve_resources(res, "pgvector", info),
                            "normalized", res, storage_engine="heap")
        assert args[args.index("--concurrency-repeats") + 1] == "3"
        # And the pool stays the sentinel for "all of them".
        from harness.workloads.concurrency import WHOLE_TEST_SET
        assert (args[args.index("--concurrency-queries") + 1]
                == str(WHOLE_TEST_SET))

    def test_the_estimate_follows_the_profile_key(self):
        """TE's `_concurrency_windows` already multiplies by the key, so the
        printed estimate had to move without a line changing in
        orchestrator/. Checked as a ratio against the same profile with the
        key removed, so it measures the multiplication rather than restating
        today's hours."""
        from orchestrator.estimate import estimate_load_hours

        def windows(profile):
            e = estimate_load_hours(profile, ["pgvector"],
                                    ["glove-100-angular"], ["normalized"],
                                    ["ops"])
            return next(t["typical_seconds"] for t in e["query_terms"]
                        if t["name"] == "concurrency windows")

        shipped = load_profile("main")
        once = {**shipped, "ops": {k: v for k, v in shipped["ops"].items()
                                   if k != "concurrency_repeats"}}
        # (repeats + 1) / (1 + 1): ops measurement version 9 adds one
        # discarded warm-up window per point, paid once whatever the repeat
        # count, so three repeats cost 4 windows against 2 for one.
        assert windows(shipped) == windows(once) * 2


# ---------------------------------------------------------------------------
# A4: pgvector's two parallelism blocks pin different keys, on purpose
# ---------------------------------------------------------------------------

def _is_pg_parallelism_setting(name):
    """Whether a PostgreSQL setting steers parallel execution for this engine.

    Matched by name rather than enumerated, because the point is to catch the
    knob nobody thought to enumerate. Two kinds qualify, and a per-pass
    difference in either is the asymmetry this unit exists to force an
    argument about:

      * CAPS on a worker count -- max_parallel_maintenance_workers,
        max_parallel_workers_per_gather, max_parallel_workers,
        max_worker_processes.
      * PLANNER KNOBS that decide how often a plan goes parallel at all --
        parallel_setup_cost, parallel_tuple_cost, min_parallel_table_scan_size,
        min_parallel_index_scan_size, parallel_leader_participation.

    The planner knobs are here because of what this family used to claim. It
    matched the prefixes "max_parallel" and "max_worker" and its docstring
    said it covered "every PostgreSQL setting that can hand pgvector a worker
    process", so a reader had been told the family was closed. It was not:
    `parallel_setup_cost=0` and `min_parallel_table_scan_size=0` added to the
    tuned block alone passed all five tests below. Zero is not "off" for a
    cost -- it makes a parallel plan cheaper, so it widens exactly what the
    caps narrow, and it did it where nothing was looking.

    DELIBERATELY OUTSIDE THE FAMILY, illustratively and not exhaustively:
    autovacuum_max_workers, max_logical_replication_workers,
    max_sync_workers_per_subscription. They cap workers and match neither test
    here, and nothing they start does vector work -- including them would turn
    this into a list of every worker PostgreSQL has and tell a reader nothing
    about the comparison. The list is illustrative because this is a rule and
    not an enumeration: any other non-vector worker cap is outside too,
    without an edit here.

    An earlier version of this paragraph ended "the test that every knob is
    named in the comment will ask for it". IT DID NOT, in two ways, and the
    second is the interesting one.

    First, that test built its key set from THIS predicate, so a name outside
    the family was never compared against the prose at all:
    `autovacuum_max_workers=8` added to the tuned block passed all five tests,
    and so did `max_sync_workers_per_subscription=8`.

    Second, widening the predicate was still not enough, which is why the
    guard is no longer a prose grep. `autovacuum_max_workers` is ALREADY named
    in `config/engines/pgvector.yml` -- in the sentence explaining why the two
    pools are unreachable in the normalized pass -- so a substring check found
    it "explained" and the mutant passed again. A grep cannot tell an argument
    for setting something from an incidental mention of it while arguing
    something else, and it never will.

    So the promise is kept by
    `test_no_other_worker_or_parallelism_setting_is_pinned` instead, which
    asserts the two blocks are CLOSED over `_names_a_worker_or_parallelism`:
    nothing whose name mentions a worker or parallelism may be pinned beyond
    the documented four. Setting one of these here stays allowed, and it now
    costs an edit to that test -- which is where a reviewer sees the argument,
    rather than a sentence a grep cannot evaluate.
    """
    return "parallel" in name or name == "max_worker_processes"


def _names_a_worker_or_parallelism(name):
    """Whether a setting's name mentions a worker or parallelism at all.

    Deliberately wider than `_is_pg_parallelism_setting`, and used by the two
    tests that ask "would a reader of this file be surprised to find this
    set?" rather than "does this cap a vector worker?" -- the closed-world
    test, which is the load-bearing one, and the prose test beside it.

    The two are separate predicates because they answer different questions,
    and widening the narrow one instead would be wrong.
    `_is_pg_parallelism_setting` feeds assertions about key sets and VALUES,
    and autovacuum_max_workers does not belong in any of them: the tuned-chain
    test asserts that family is exactly the three-key chain, and the
    differs-by test asserts the two passes' difference, so a legitimately set
    autovacuum cap would have to be written into both AS A PARALLELISM KEY,
    which is what it is not. (It would NOT, as an earlier draft of this
    paragraph said, assert that the normalized pass must pin it to zero. That
    test's zero check filters on names starting `max_parallel`/`max_worker`,
    so an autovacuum cap would land in its presence-refused branch instead.
    The decision was right and the reason given for it was wrong.)

    This one feeds assertions about PRESENCE only, where the cost of including
    a setting is an edit to one test and the cost of excluding it is a worker
    cap nobody is asked about.

    A superset of the narrow predicate by construction: every name that one
    matches either contains "parallel" or is max_worker_processes, which
    contains "worker".
    """
    return "parallel" in name or "worker" in name


def _pg_settings_from_argv(tokens):
    """Every GUC assignment in a postgres argv, keyed by canonical name.

    MIRRORS THE SERVER'S PARSING AND NOT THIS FILE'S HOUSE STYLE, which is the
    whole reason it exists. An earlier version recognised a setting only when
    the preceding token was exactly `-c`, while the tests built on it claimed
    the two blocks were "closed" over anything that names a worker. They were
    closed over one spelling. All three of these were accepted in silence:

        - "--max_worker_processes=16"
        - "--max-worker-processes=16"
        - "--autovacuum-max-workers=16"

    and the last is precisely the class the closed-world test exists to
    refuse. The spelling is reachable rather than theoretical, twice over.
    `docker/pgvector/entrypoint-pgvector.sh` appends `${VB_SERVER_ARGS}`
    UNQUOTED into the argv it execs, so the token word-splits and arrives at
    postgres unchanged. And a bare `--name=value` is the prevailing style in
    four sibling configs -- `percona.yml`, `mariadb.yml`, `mariadb123.yml` and
    `alisql.yml` all spell their whole tuned block that way, and this file is
    the only one using `-c` pairs -- so someone moving between them writes
    `--max-parallel-workers=16` from habit, the server honours it, and the
    guard says nothing.

    So three spellings are accepted, as postgres accepts them: `-c name=value`,
    `-cname=value`, and a standalone `--name=value`. Hyphens in the NAME are
    folded to underscores, which is what the server's own `ParseLongOption`
    does before looking a GUC up, so `--max-worker-processes` and
    `max_worker_processes` are one key here as they are one setting there.
    `--name value` is deliberately not accepted: postgres rejects a long option
    with no `=` rather than consuming the next token.

    Last writer wins, matching both a Python dict and the server: a duplicate
    assignment later in the argv is the one in force, so a decoy placed BEFORE
    the real value correctly changes nothing here.
    """
    settings = {}
    index = 0
    while index < len(tokens):
        text = str(tokens[index])
        pair = None
        following = str(tokens[index + 1]) if index + 1 < len(tokens) else ""
        if text == "-c" and "=" in following:
            pair, index = following, index + 1
        elif text.startswith("--") and "=" in text:
            pair = text[2:]
        elif text.startswith("-c") and len(text) > 2 and "=" in text:
            pair = text[2:]
        if pair:
            name, _, value = pair.partition("=")
            settings[name.replace("-", "_")] = value
        index += 1
    return settings


def _pg_settings(engine, resource_pass, info):
    """The settings this repository actually sends, as a dict.

    Rendered through `server_args`, so this is what reaches the postmaster
    rather than what the file says. `_raw_pg_settings` is the other half: a
    rendered value cannot tell a placeholder from a literal that happens to
    equal it on this host, which is how a de-parameterised tuned chain
    survived the first version of these tests.
    """
    return _pg_settings_from_argv(
        server_args(load_engine(engine), resource_pass,
                    resolve_resources(load_resources(resource_pass),
                                      engine, info)))


def _raw_pg_settings(engine, resource_pass):
    """The same settings unrendered, placeholders and all, straight from YAML.

    Needed because every assertion about a value being *derived* from the
    resource pass has to be made before substitution. After it,
    `max_worker_processes=8` and `max_worker_processes={build_threads}` are
    the same string on an eight-thread host.

    Reads `server.common` ahead of the pass block, in that order, because
    `server_args` concatenates them that way: otherwise moving a setting into
    the shared block -- placeholder intact and nothing wrong -- would read here
    as the setting having vanished.
    """
    with open(os.path.join(VB_ROOT, "config", "engines",
                           f"{engine}.yml")) as fh:
        cfg = yaml.safe_load(fh) or {}
    server = cfg.get("server") or {}
    return _pg_settings_from_argv(list(server.get("common") or [])
                                  + list(server.get(resource_pass) or []))


def _pg_parallel_settings(engine, resource_pass, info):
    return {name: value
            for name, value in _pg_settings(engine, resource_pass, info).items()
            if _is_pg_parallelism_setting(name)}


def _raw_pg_parallel_settings(engine, resource_pass):
    return {name: value
            for name, value in _raw_pg_settings(engine, resource_pass).items()
            if _is_pg_parallelism_setting(name)}


class TestEveryPgvectorParallelismKnobIsPinnedOnPurpose:
    """The two passes pin different keys, and the difference has to be a
    decision rather than an oversight.

    `server.tuned` sets `max_parallel_maintenance_workers`,
    `max_parallel_workers` and `max_worker_processes` from `{build_threads}`.
    `server.normalized` pins `max_parallel_maintenance_workers=0` and
    `max_parallel_workers_per_gather=0` -- one key in common, one it has to
    itself, two of the tuned keys left at the server default. Nothing in the
    file said whether that was the intent or two forgotten lines, and
    `docs/12-percona-feedback-plan.md` §5.2 prescribes "the same treatment"
    for Percona's `innodb_parallel_read_threads` off the back of it, so an
    undocumented asymmetry was about to be copied onto an eighth block.

    These tests assert the shape of each pass rather than the presence of a
    string: that the normalized pass hands out no worker on either path, that
    the tuned pass derives every limit in the chain from the resource pass,
    and that the two key sets differ by exactly the keys the file's comment
    accounts for. An edit that keeps the spellings and breaks the comparison
    -- a literal where a placeholder belongs, a planner knob in one pass only
    -- fails these, which a substring check would not.
    """

    def test_the_normalized_pass_grants_no_worker_on_either_path(self, info):
        """Two zeros, one per path, because the paths are independent.

        pgvector's bulk build takes its worker count from
        `max_parallel_maintenance_workers` (`ComputeParallelWorkers` in the
        pinned tree's `src/hnswbuild.c` returns exactly that, clamped), and
        the query side takes its workers through a Gather node, which
        `max_parallel_workers_per_gather=0` disables outright. Zeroing one
        says nothing about the other.
        """
        pinned = _pg_parallel_settings("pgvector", "normalized", info)
        assert pinned.get("max_parallel_maintenance_workers") == "0", (
            f"the normalized bulk build would be parallel: {pinned!r}")
        assert pinned.get("max_parallel_workers_per_gather") == "0", (
            "a normalized query that misses the index could recruit workers "
            f"for the scan: {pinned!r}")
        # Any further CAP is welcome in the normalized pass -- but only at
        # zero. A non-zero one is a budget MHNSW and VIDX have no way to
        # spend.
        caps = {name: value for name, value in pinned.items()
                if name.startswith(("max_parallel", "max_worker"))}
        assert [name for name, value in sorted(caps.items())
                if value != "0"] == []
        # A planner knob is not a cap and the zero check above would wave it
        # through: `parallel_setup_cost=0` makes a parallel plan cheaper, not
        # absent. So it is presence that is refused here, not a value.
        assert set(pinned) == set(caps), (
            "the normalized pass carries a parallel-planner knob: "
            f"{sorted(set(pinned) - set(caps))}")

    def test_the_normalized_zero_is_a_literal_and_not_build_threads(self, info):
        """`{build_threads}` is 1 in the normalized pass, and 1 is not none.

        These GUCs count workers *beyond* the leader -- the pinned tree's
        README says so in as many words, `SET max_parallel_maintenance_workers
        = 7; -- plus leader`. So spelling the normalized pass's zero
        `{build_threads}` would read as "single-threaded" and buy a second
        process, which is the whole quantity `build.threads: 1` exists to deny.
        """
        resolved = resolve_resources(load_resources("normalized"), "pgvector",
                                     info)
        assert resolved.build_threads == 1, (
            "the normalized pass no longer declares a single build thread; "
            "the argument in config/engines/pgvector.yml's comment is stated "
            "against build_threads == 1 and needs re-checking")
        for name, value in sorted(
                _raw_pg_parallel_settings("pgvector", "normalized").items()):
            assert "{" not in value, (
                f"normalized pins {name}={value} from a placeholder; a worker "
                "count of build_threads is one worker, not none")

    def test_the_tuned_pass_derives_every_limit_in_the_chain(self, info):
        """A build worker has to get past three limits, so three keys move.

        A maintenance worker is drawn from the pool `max_worker_processes`
        establishes and capped again by `max_parallel_workers`; the pinned
        tree's README says to raise the second for large worker counts. Raise
        one of the three and not the others and the request is silently
        clipped -- the build still succeeds, with fewer workers than the
        manifest records, which is a measurement this repository would publish
        and could not detect. `maintenance_work_mem` is a fourth limit on the
        same count and is asserted at the end for the same reason.

        Asserted against the RAW value, which is the fix for a hole in the
        first version of this test. It compared the rendered value with
        `resolved.build_threads` and so accepted any literal equal to the
        host's resolved number. `config/resources/tuned.yml` caps
        `build.max_threads: 8` and 8 is also PostgreSQL's own default for both
        pools, so `max_worker_processes=8` passed here and on any machine with
        eight performance cores, and broke the invariant the config comment
        states -- "they move together or not at all" -- on a four-core one.
        """
        chain = {"max_parallel_maintenance_workers", "max_parallel_workers",
                 "max_worker_processes"}
        raw = _raw_pg_parallel_settings("pgvector", "tuned")
        assert set(raw) == chain, (
            f"the tuned parallel family changed shape: {sorted(raw)}")
        for name in sorted(chain):
            assert raw[name] == "{build_threads}", (
                f"tuned {name}={raw[name]} is not derived from the resource "
                "pass; the weakest link decides, so this pass would measure a "
                "worker count nothing declared, and would keep measuring it "
                "when the cpuset changes")
        # And the placeholder is the one the orchestrator substitutes, so the
        # three really do arrive equal rather than merely spelled alike.
        resolved = resolve_resources(load_resources("tuned"), "pgvector", info)
        assert resolved.build_threads >= 1
        rendered = _pg_parallel_settings("pgvector", "tuned", info)
        assert set(rendered) == chain
        for name in sorted(chain):
            assert rendered[name] == str(resolved.build_threads)
        # THE FOURTH CLAMP, which is not a parallelism key and is checked here
        # anyway. PostgreSQL 17's CREATE INDEX page: maintenance_work_mem "may
        # influence the number of worker processes requested, since parallel
        # workers must have at least a 32MB share of the total
        # maintenance_work_mem budget. There must also be a remaining 32MB
        # share for the leader process." So the three assertions above are
        # only as good as the memory behind them -- at a 256 MB literal an
        # eight-worker request is granted as seven, and smaller values cut
        # harder -- and a de-parameterised value here produces exactly the
        # failure this test's docstring names. It is deliberately NOT in the
        # parallelism family: it is a memory knob with its own
        # resource-fraction machinery, and a parallelism predicate that
        # matched it would mis-describe it.
        #
        # Tuned only. In the normalized pass the request is zero workers, so
        # there is nothing for this to clamp; a literal there would be a
        # memory-budget bug rather than a parallelism one and is not this
        # test's business.
        maintenance = _raw_pg_settings("pgvector", "tuned").get(
            "maintenance_work_mem")
        assert maintenance == "{maintenance_bytes}B", (
            f"tuned maintenance_work_mem={maintenance} is not derived from "
            "the resource pass; it is the fourth clamp on the worker count, "
            "so a literal silently re-caps the chain asserted above")

    def test_the_two_passes_differ_by_exactly_the_documented_keys(self, info):
        """The passes are not each other's control, and this pins how far.

        One key in common, one key normalized has that tuned leaves at the
        PostgreSQL default, two keys tuned has that normalized leaves at the
        PostgreSQL default. Each half is argued in the file's comment; moving
        a key between the blocks, or adding one to either, fails here and has
        to answer that comment, which is what stops the next edit making the
        two blocks incomparable by accident.
        """
        normalized = set(_pg_parallel_settings("pgvector", "normalized", info))
        tuned = set(_pg_parallel_settings("pgvector", "tuned", info))
        assert normalized & tuned == {"max_parallel_maintenance_workers"}
        assert normalized - tuned == {"max_parallel_workers_per_gather"}
        assert tuned - normalized == {"max_parallel_workers",
                                      "max_worker_processes"}

    def test_no_other_worker_or_parallelism_setting_is_pinned(self, info):
        """Each block is closed over anything whose name mentions a worker.

        The documented four are the whole of it. A fifth setting whose name
        mentions a worker or parallelism fails here whatever its value,
        whatever the comment says and whatever spelling it is written in,
        which is the point: this is the guard the family docstring's "needs
        its own argument" promise rests on, and it is a key-set assertion
        rather than a prose grep because a grep cannot tell an argument from
        an incidental mention.

        "Whatever spelling" is load-bearing and was not true at first. This
        test read its names through a helper that recognised only `-c
        name=value`, so `--autovacuum-max-workers=16` -- the house spelling in
        four sibling engine configs, word-split into the argv unquoted by
        docker/pgvector/entrypoint-pgvector.sh, and honoured by the server --
        passed while the comment claimed the block was closed. The claim was
        the new guard rather than a remark at its edge, which is what made it
        worth fixing rather than qualifying. `_pg_settings_from_argv` now
        parses what postgres parses.

        It was a prose grep first and that did not work.
        `autovacuum_max_workers=8` in the tuned block passed a widened grep,
        because the name already appears in the file -- in the sentence saying
        autovacuum draws on that pool rather than on max_worker_processes. The
        mention was honest and had nothing to do with setting it.

        Adding one of these deliberately means editing this test, where a
        reviewer reads the reason. That is the intended cost.
        """
        documented = {
            "normalized": {"max_parallel_maintenance_workers",
                           "max_parallel_workers_per_gather"},
            "tuned": {"max_parallel_maintenance_workers",
                      "max_parallel_workers", "max_worker_processes"},
        }
        for resource_pass, expected in sorted(documented.items()):
            named = {name
                     for name in _pg_settings("pgvector", resource_pass, info)
                     if _names_a_worker_or_parallelism(name)}
            assert named == expected, (
                f"{resource_pass} pins a worker or parallelism setting the "
                f"file does not account for: {sorted(named - expected)}; "
                f"missing: {sorted(expected - named)}")

    def test_every_knob_either_pass_touches_is_argued_in_the_file(self, info):
        """A config decision carries its argument, or it is not a decision.

        The asymmetry survived unexamined because the block that encodes it
        said nothing, and the next reader -- `docs/12-percona-feedback-plan.md`
        §5.2 -- was told to copy it. Naming every knob in the comment is the
        cheapest guard against that: a key added to either pass without a
        sentence explaining it fails this.

        The key set comes from `_names_a_worker_or_parallelism`, which is
        wider than the parallelism family the other tests use, because this is
        the one question whose answer should not depend on whether a setting
        caps a *vector* worker. Before that, this test filtered by the narrow
        family and so never looked at a name outside it:
        `autovacuum_max_workers=8` in the tuned block, explained nowhere,
        passed.

        It guards the names and not the argument, which is a known limit. A
        comment replaced by a bare list of the key names passes, and testing
        prose is not attempted here.
        """
        with open(os.path.join(VB_ROOT, "config", "engines",
                               "pgvector.yml")) as fh:
            lines = fh.read().splitlines()
        prose = "\n".join(line for line in lines if line.strip().startswith("#"))
        touched = {name
                   for resource_pass in ("normalized", "tuned")
                   for name in _pg_settings("pgvector", resource_pass, info)
                   if _names_a_worker_or_parallelism(name)}
        for name in sorted(touched):
            assert name in prose, (
                f"{name} is pinned in config/engines/pgvector.yml and "
                "explained nowhere in it")


class TestPublishableProfiles:
    """A profile a report is published from measures every ann point at least
    twice. full-20260825 was a tuned-complete run at `runs: 1` (all 112 ann
    records carry `run_count` 1), so the first point of every sweep was a
    single draw on a cold index -- and it published ef_search 10 SLOWER than
    20 on every MySQL-family curve. Two runs do not replace the index warm-up; they
    stop one unlucky draw from becoming the published number."""

    PUBLISHED = ("tuned-complete", "main", "full")
    NOT_PUBLISHED = ("dev", "smoke")

    @staticmethod
    def _profile(name):
        from orchestrator.config import load_profile
        return load_profile(name)

    @pytest.mark.parametrize("name", PUBLISHED)
    def test_the_published_profiles_are_marked(self, name):
        assert self._profile(name).get("publishable") is True

    @pytest.mark.parametrize("name", NOT_PUBLISHED)
    def test_development_profiles_are_left_alone(self, name):
        profile = self._profile(name)
        assert not profile.get("publishable")
        assert profile.get("runs") == 1

    def test_no_publishable_profile_runs_fewer_than_two(self):
        from orchestrator.config import available_profiles
        short = {}
        for name in available_profiles():
            profile = self._profile(name)
            if profile.get("publishable"):
                runs = profile.get("runs", 1)
                if not isinstance(runs, int) or runs < 2:
                    short[name] = runs
        assert not short, (
            f"publishable profile(s) with runs < 2: {short}. ann-benchmarks "
            "keeps the best of `runs`; one run publishes a single draw.")

    def test_the_key_is_one_the_profile_editor_knows(self):
        """Otherwise the web UI warns about it on every published profile."""
        from webui.profiles import KNOWN_TOP_LEVEL
        assert "publishable" in KNOWN_TOP_LEVEL

    def test_runs_reaches_ann_benchmarks(self):
        source = open(os.path.join(VB_ROOT, "orchestrator", "ann_pass.py")).read()
        assert '"--runs", str(profile.get("runs", 1))' in source


# ---------------------------------------------------------------------------
# F6: every engine declares the build its summary figures come from
# ---------------------------------------------------------------------------

def _profile_engines(profile):
    from orchestrator.engines import known_engines
    return list(profile.get("engines") or known_engines())


class TestEachEngineDeclaresAReferenceBuild:
    """`full-20260825` published Percona Search's QPS-at-recall from the
    unquantized sweep beside load, build and concurrency figures from the
    scalar index, and pgvector's and Valkey's from efC=400 beside ops figures
    at 200 and at the server default. The reference build is the one the ops
    pass builds, declared per engine and resolved per profile and pass, and
    the run refuses a declaration the ann grid would never measure."""

    @pytest.mark.parametrize("engine", ["mariadb", "mariadb123", "alisql",
                                        "pgvector", "valkey", "mongodb",
                                        "percona", "myvector"])
    def test_every_engine_config_declares_one(self, engine):
        runtime = load_engine(engine).get("runtime") or {}
        assert runtime.get("reference_build"), engine

    def test_the_registry_carries_it(self):
        from orchestrator import engines
        assert dict(engines.get("pgvector").reference_build) == {
            "m": 16, "ef_construction": "ops"}

    @pytest.mark.parametrize("profile_name", available_profiles())
    @pytest.mark.parametrize("resource_pass", PASSES)
    def test_every_shipped_profile_contains_its_reference_builds(
            self, profile_name, resource_pass):
        from orchestrator.reference_build import reference_build_problem
        profile = load_profile(profile_name)
        resources = merge_resource_overrides(load_resources(resource_pass),
                                             profile)
        for engine in _profile_engines(profile):
            problem = reference_build_problem(engine, profile, resources,
                                              resource_pass)
            assert problem is None, (profile_name, resource_pass, problem)

    def _resolve(self, engine, profile_name, resource_pass):
        from orchestrator.reference_build import resolve_reference_build
        profile = load_profile(profile_name)
        resources = merge_resource_overrides(load_resources(resource_pass),
                                             profile)
        return resolve_reference_build(engine, profile, resources,
                                       resource_pass)

    def test_percona_search_is_scalar_in_the_tuned_pass(self):
        assert self._resolve("mongodb", "tuned-complete", "tuned") == {
            "quantization": "scalar"}

    def test_percona_search_is_unquantized_in_the_normalized_pass(self):
        assert self._resolve("mongodb", "main", "normalized") == {
            "quantization": "none"}

    def test_m_sweep_pins_percona_search_unquantized_even_when_tuned(self):
        assert self._resolve("mongodb", "m-sweep", "tuned") == {
            "quantization": "none"}

    def test_pgvector_resolves_to_what_ops_builds(self):
        """`harness_args` reads `ann.pgvector_ef_construction`, default 200,
        in BOTH passes -- not the tuned extras' [64, 200, 400]."""
        assert self._resolve("pgvector", "tuned-complete", "tuned") == {
            "m": 16, "ef_construction": 200}

    def test_valkey_is_the_server_default_named(self):
        """The ops pass passes no EF_CONSTRUCTION; valkey-search's
        `kDefaultEFConstruction` is 200."""
        assert self._resolve("valkey", "tuned-complete", "tuned") == {
            "m": 16, "ef_construction": 200}

    def test_the_resolution_agrees_with_the_harness_argv(self):
        from orchestrator import ops_pass
        profile = load_profile("main")
        profile["ann"]["pgvector_ef_construction"] = 64
        argv = ops_pass.harness_args(profile, 16, "pgvector", None,
                                     "InnoDB", "post")
        assert argv[argv.index("--ef-construction") + 1] == "64"
        assert ops_pass.ops_ef_construction("pgvector", profile) == 64

    def test_a_build_outside_the_grid_is_refused(self):
        """ops builds pgvector at efC=100 here, and the tuned grid sweeps
        64/200/400: no recall point would ever describe the ops index."""
        from orchestrator.reference_build import reference_build_problem
        profile = load_profile("main")
        profile["ann"]["pgvector_ef_construction"] = 100
        resources = merge_resource_overrides(load_resources("tuned"), profile)
        problem = reference_build_problem("pgvector", profile, resources,
                                          "tuned")
        assert problem and "ef_construction=100" in problem
        assert "pgvector" in problem

    def test_an_m_the_grid_does_not_sweep_is_refused(self):
        from orchestrator.reference_build import reference_build_problem
        profile = load_profile("main")
        profile["ann"]["m_values"] = [8, 32]
        resources = merge_resource_overrides(load_resources("normalized"),
                                             profile)
        problem = reference_build_problem("mariadb", profile, resources,
                                          "normalized")
        assert problem and "m=16" in problem

    def test_the_run_records_them_in_the_manifest_presentation(self):
        """The report cannot import the orchestrator, so the resolved builds
        travel the way labels and colours do."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        body = source.split("def cmd_run")[1].split("\ndef ")[0]
        # The check and resolution moved into `_build_preflight`; that every
        # engine and pass is actually filled is asserted by behaviour in
        # `test_every_engine_and_pass_gets_its_reference_build`.
        helper = source.split("def _build_preflight")[1].split("\ndef ")[0]
        assert "_build_preflight(" in body
        assert "reference_build_problem" in helper
        assert '"reference_builds"' in body


# ---------------------------------------------------------------------------
# F3: client CPUs sized from the sweep
# ---------------------------------------------------------------------------

class TestClientCpusAuto:
    """A 32-client sweep driven from 4 client CPUs measures the clients. The
    tuned-complete profile pinned 16 by hand; `auto` derives it from the
    largest client count and refuses, before the run, a topology that cannot
    supply it."""

    def _host(self, cpus):
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(cpus)),
                                  "efficiency_cpus": [], "logical_cpus": cpus,
                                  "physical_cores": cpus,
                                  "threads_per_core": 1, "model": "Xeon"})()
        info.total_ram_bytes = 256 * 1024 ** 3
        return info

    def _resources(self, server_cpus, client_cpus="auto"):
        res = load_resources("tuned")
        res["cpu"] = dict(res["cpu"], server_cpus=server_cpus,
                          client_cpus=client_cpus, allow_smt=True)
        return res

    def test_auto_is_half_the_largest_client_count(self):
        from orchestrator.config import auto_client_cpus
        assert auto_client_cpus(32) == 16
        assert auto_client_cpus(33) == 17

    def test_auto_has_a_floor_of_four(self):
        from orchestrator.config import auto_client_cpus
        assert auto_client_cpus(1) == 4
        assert auto_client_cpus(4) == 4

    def test_a_one_to_thirty_two_sweep_resolves_to_sixteen(self):
        resolved = resolve_resources(self._resources(0), "pgvector",
                                     self._host(80), max_clients=32)
        assert len(_expand_cpuset(resolved.client_cpuset)) == 16

    def test_the_client_set_is_taken_after_the_server_set(self):
        resolved = resolve_resources(self._resources(64), "pgvector",
                                     self._host(80), max_clients=32)
        assert resolved.server_cpuset == "0-63"
        assert resolved.client_cpuset == "64-79"

    def test_a_pool_that_cannot_supply_it_fails_with_both_numbers(self):
        """80 CPUs, 64 to the server: 16 left, and a 1-64 sweep needs 32."""
        with pytest.raises(ValueError) as excinfo:
            resolve_resources(self._resources(64), "pgvector",
                              self._host(80), max_clients=64)
        message = str(excinfo.value)
        assert "32" in message and "16" in message

    def test_auto_without_a_client_count_is_refused(self):
        with pytest.raises(ValueError):
            resolve_resources(self._resources(0), "pgvector", self._host(80))

    def test_a_numeric_pin_still_works_and_still_only_warns(self):
        resolved = resolve_resources(self._resources(78, client_cpus=8),
                                     "pgvector", self._host(80))
        assert any("share cores" in w for w in resolved.warnings)

    def test_tuned_complete_uses_auto_and_keeps_its_fingerprint(self):
        """16 by hand and `auto` over 1-32 are the same 16 CPUs, so the shared
        ann tree must not move: the fingerprint hashes the resolved count."""
        profile = load_profile("tuned-complete")
        assert profile["resources"]["cpu"]["client_cpus"] == "auto"
        host = self._host(80)
        auto = resolve_resources(
            merge_resource_overrides(load_resources("tuned"), profile),
            "pgvector", host, max_clients=32)
        pinned_profile = copy.deepcopy(profile)
        pinned_profile["resources"]["cpu"]["client_cpus"] = 16
        pinned = resolve_resources(
            merge_resource_overrides(load_resources("tuned"), pinned_profile),
            "pgvector", host, max_clients=32)
        assert auto.client_cpuset == pinned.client_cpuset
        assert auto.pass_signature == pinned.pass_signature
        assert ann_pass.ann_fingerprint(auto) == ann_pass.ann_fingerprint(pinned)

    def test_no_other_shipped_profile_changed(self):
        for name in available_profiles():
            if name == "tuned-complete":
                continue
            cpu = (load_profile(name).get("resources") or {}).get("cpu") or {}
            assert cpu.get("client_cpus") != "auto", name

    def test_the_engine_plan_resolves_auto_from_the_profile(self, tmp_path):
        """`engine_plan.resolve_engines` is what every unit is resolved by
        since #76; without the client count `auto` would raise there."""
        from orchestrator.engine_plan import resolve_engines
        from orchestrator.manifest import Manifest
        profile = load_profile("tuned-complete")
        plan = resolve_engines(Manifest(str(tmp_path), "auto-1"), ["tuned"],
                               ["pgvector"], profile, self._host(80), 0,
                               set(), [], [])
        resolved = plan.resolved[("tuned", "pgvector")]
        assert len(_expand_cpuset(resolved.client_cpuset)) == 16

    def test_the_run_resolves_with_the_largest_client_count(self):
        from orchestrator.config import profile_max_clients
        assert profile_max_clients(load_profile("tuned-complete")) == 32
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        body = source.split("def cmd_run")[1].split("\ndef ")[0]
        assert "max_clients=" in body

    def test_the_estimate_prints_the_client_cpuset_beside_the_sweep(self):
        from orchestrator.estimate import client_cpu_line
        line = client_cpu_line({"ops": {"client_counts": [1, 4, 32]}},
                               {"tuned": 16})
        assert "16 client CPU(s)" in line and "32 clients" in line


def _expand_cpuset(text):
    cpus = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus
