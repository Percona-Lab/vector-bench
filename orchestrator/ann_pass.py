"""Recall/QPS measurement path: ann-benchmarks, run inside each engine image.

ann-benchmarks is used unmodified. Rather than letting its own Docker runner
launch containers (which would give us no control over env, cpuset or memory),
the orchestrator launches the container itself and runs `run.py --local` inside
it. Local mode executes the algorithm in-process, so ann-benchmarks still owns
definitions, ground truth, recall computation and result files, while the
orchestrator owns the container's resource limits. No patching of the upstream
project is required.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import docker_ctl
from . import engines as engines_mod
from .ann_knobs import (VILLAGESQL_ENGINES, efc_run_groups, excluded_builds,
                        knob_problems, restrict_to_builds)
from .config import ResolvedResources, server_args, server_env

# Re-exported, not just imported: `cli.py` and a dozen tests reach these
# through `ann_pass`, and the split that moved them into their own module is
# about where the code lives, not about who may call it. `render_config`
# deliberately stayed here -- docs/09 §4.10 is a historical record of the
# Percona addition carrying ten line citations into it, and moving the
# function would have falsified the section rather than updated it.
from .ann_config import (  # noqa: F401
    config_path,
    expected_result_count,
    expected_results,
    read_config,
    result_stem,
    result_stems,
    write_config,
)


# ann-benchmarks raises this when a run has no definitions left. NOT only when
# everything is already done -- see the handling in `run_engine`.
# See ann_benchmarks/main.py: `raise Exception("Nothing to run")`.
NOTHING_TO_RUN = "Nothing to run"

# Engines that are MariaDB at a different tag. They share the driver, the SQL,
# the Dockerfile and the storage-engine axis; only the source tag differs.
MARIADB_ENGINES = ("mariadb", "mariadb123")


# Bump when the measurement path itself changes in a way that makes older
# numbers non-comparable, even at an identical resource configuration. It is
# part of `ann_fingerprint`, so a bump moves EVERY pass's results tree and
# forces every recall point of every engine to be measured again -- which is
# the intent when it is bumped, and the reason it is not the remedy for a
# single engine's re-pin (CLAUDE.md, the ann results tree invariant).
#
# History:
#   2 -- the per-configuration `_warm_cache` (30 random-vector queries): without
#        it the first ef_search point paid for a cold cache and landed below
#        the second.
#   3 -- the one-time index warm-up (overlay/.../vb_warmup.py): training-set
#        queries before the first configuration on each built index until two
#        consecutive 200-query batches agree within 5%, capped at 5,000. Thirty
#        queries had not been enough: full-20260825 (tuned-complete, dbpedia
#        1M x 1536) still measured ef_search 10 slower than 20 on all six
#        MariaDB/AliSQL curves (1.69x-2.13x) and on both Percona Search curves
#        (1.13x). A curve measured at 2 has that cold first point in it.
ANN_MEASUREMENT_VERSION = 3


def ann_fingerprint(resolved: Any) -> str:
    """Short digest of everything that makes two ann runs incomparable.

    ann-benchmarks caches by algorithm and index parameters. It has no idea how
    much memory the engine was given or how many cores it had, so a result
    measured under one budget is silently reused under another.

    That is not hypothetical. A dbpedia run was re-launched at 64 GB after the
    first attempt had run at 16 GB, and every recall point came back
    byte-identical to the old ones because the files were already on disk. The
    report then carried a manifest saying 64 GB above a curve measured at 16.
    """
    # Accepts a ResolvedResources or the dict a manifest stores, so the report
    # can recompute the fingerprint without rebuilding the resource objects.
    get = (resolved.get if isinstance(resolved, dict)
           else lambda k, d=None: getattr(resolved, k, d))

    # Deliberately engine-invariant. The individual cache figures differ by
    # engine by design -- pgvector has no separate graph cache, so its buffer
    # absorbs that share -- and hashing them gave every engine its own results
    # tree. The report can only narrow to one tree, so a three-engine run
    # produced a recall chart containing a single engine.
    #
    # Summing the allocations fixed that only while every engine allocated the
    # same total. Percona Search takes a JVM heap and leaves the rest to the
    # page cache, and Valkey allocates nothing at all because the container
    # budget is the dataset, so the sum fragmented the tree into three and a
    # six-engine smoke run again produced a recall chart with one engine in it.
    #
    # The pass signature is the pass's declared knobs, taken before any
    # engine-specific adjustment. It is identical across engines by
    # construction, and it still changes when the fractions do.
    allocated = str(get("pass_signature") or "")
    fields = (
        ANN_MEASUREMENT_VERSION,
        get("server_memory_bytes"),
        allocated,
        get("build_threads"),
        get("server_cpu_count"),
    )
    blob = "|".join(str(f) for f in fields).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def annb_results_dir(paths: Dict[str, str], resource_pass: str,
                     resolved: Optional[ResolvedResources] = None) -> str:
    """Result tree for one resource pass and one resource configuration.

    ann-benchmarks derives its result filenames from the algorithm and its
    parameters, with no notion of a resource pass. Sharing one tree would make
    the tuned pass skip every point the normalized pass already computed — and
    the report would then present normalized numbers as tuned ones. Separate
    trees keep the two passes independent and keep resumption working within
    each.

    The configuration fingerprint extends the same reasoning to the budget: a
    curve measured at 16 GB is not a curve measured at 64 GB, and resumption
    should not treat them as interchangeable. Runs at an unchanged
    configuration still resume normally, because the fingerprint is unchanged.
    """
    parts = [paths["annb_results"], resource_pass]
    if resolved is not None:
        parts.append(ann_fingerprint(resolved))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def ann_m_values(engine: str, profile: Dict[str, Any],
                 resources: Dict[str, Any], resource_pass: str) -> List[int]:
    """The M values the ann phase builds for `engine`, in this pass.

    The tuned pass's `resources.extras.m_values_by_engine` first, then the
    profile's `ann.m_values_by_engine`, then `ann.m_values`. One function so
    the ann render and `cli._memory_refusal` -- which must estimate the same
    graphs the ann phase will build -- cannot read the grid two ways.
    """
    ann = profile.get("ann", {}) or {}
    by_engine = ((resources.get("extras", {}) or {}).get("m_values_by_engine")
                 if resource_pass == "tuned" else None) or \
                ann.get("m_values_by_engine") or {}
    return list(by_engine.get(engine) or ann.get("m_values", [16]))


def render_config(engine: str, profile: Dict[str, Any],
                  resources: Dict[str, Any], resource_pass: str) -> Dict[str, Any]:
    """Build the ann-benchmarks config.yml body for one engine and profile.

    Generated rather than hand-maintained so a profile change cannot leave one
    engine sweeping a different grid from the others — which would look like a
    performance difference and would not be one.
    """
    ann = profile.get("ann", {}) or {}
    # Graph degree is the only build knob MHNSW and VIDX expose, and at M=16
    # their lowest query effort already returns recall 0.97-0.98. So they have
    # no measurement anywhere near the 0.90 floor the headline table reports at,
    # while pgvector and Valkey -- which have ef_construction as well -- reach
    # down to 0.76. Sweeping M only where it is the sole knob costs a reload
    # per value on the engines that need it and nothing on the engines that do
    # not, and it turns three daggered cells into real measurements.
    m_values = ann_m_values(engine, profile, resources, resource_pass)
    query_args = [list(ann.get("ef_search", [10, 40, 160]))]
    extras = (resources.get("extras", {}) or {}) if resource_pass == "tuned" else {}

    run_groups: Dict[str, Any] = {}

    # Refused, never clamped -- see orchestrator/ann_knobs.py. M is checked
    # here for every engine; ef_construction in each branch that reads it.
    problems = knob_problems(engine, m_values)
    if problems:
        raise ValueError("; ".join(problems))

    if engine in MARIADB_ENGINES + ("alisql",):
        if engine in MARIADB_ENGINES:
            storage_engines = list(
                extras.get("mariadb_storage_engines")
                or ann.get("mariadb_storage_engines", ["InnoDB"])
            )
        else:
            # VIDX is InnoDB-only; offering anything else would fail at DDL.
            storage_engines = ["InnoDB"]

        for storage in storage_engines:
            run_groups[storage.lower()] = {
                "arg_groups": [{"M": m_values, "engine": storage}],
                "args": {},
                "query_args": query_args,
            }

    elif engine == "pgvector":
        if resource_pass == "tuned" and extras.get("pgvector_ef_construction"):
            ef_constructions = list(extras["pgvector_ef_construction"])
        else:
            # Pinned in the normalized pass: ef_construction is the one build
            # knob MariaDB and AliSQL do not expose, so sweeping it there would
            # hand pgvector a tuning axis the others lack.
            ef_constructions = [int(ann.get("pgvector_ef_construction", 200))]

        _refuse_efc(engine, m_values, ef_constructions)
        for build_mode in ann.get("pgvector_build_modes", ["post"]):
            run_groups.update(efc_run_groups(f"{build_mode}_build", {
                "M": m_values,
                "efConstruction": ef_constructions,
                "build_mode": build_mode,
            }, engine, query_args))
    elif engine == "valkey":
        # The same two knobs pgvector has, so the same rule: pinned in the
        # normalized pass because MariaDB and AliSQL still lack ef_construction,
        # swept in the tuned pass where each engine gets its own idioms.
        if resource_pass == "tuned" and extras.get("valkey_ef_construction"):
            ef_constructions = list(extras["valkey_ef_construction"])
        else:
            ef_constructions = [int(ann.get("valkey_ef_construction", 200))]

        _refuse_efc(engine, m_values, ef_constructions)
        for build_mode in ann.get("valkey_build_modes", ["post"]):
            run_groups.update(efc_run_groups(f"{build_mode}_build", {
                "M": m_values,
                "efConstruction": ef_constructions,
                "build_mode": build_mode,
            }, engine, query_args))

    elif engine == "percona":
        # A branch of its own and NOT an addition to the MySQL-family
        # condition above, because the thing this engine needs that branch
        # cannot express is `efConstruction`. No MySQL-family branch emits it,
        # because no other MySQL-family engine here exposes it: Percona has a
        # build-time knob AND a query-time one (`innodb_hnsw_ef_search`),
        # which makes it the only engine in this study with both.
        #
        # Pinned in the normalized pass for the same reason pgvector's and
        # Valkey's are, and `percona/config.yml` says the same thing at
        # length: MariaDB and AliSQL still lack the knob, and sweeping it
        # where only one engine has it hands that engine a tuning axis the
        # others cannot answer. Swept in the tuned pass, where following each
        # vendor's own idioms is the point.
        if resource_pass == "tuned" and extras.get("percona_ef_construction"):
            ef_constructions = list(extras["percona_ef_construction"])
        else:
            ef_constructions = [int(ann.get("percona_ef_construction", 200))]

        # `percona_build_modes` IS REFUSED RATHER THAN HONOURED, and this is
        # the one place in the Percona addition where a documented intention
        # could not be delivered.
        #
        # docs/09 §2.5 and §4.10 propose it: Percona maintains the graph on
        # every INSERT *and* has a separable bulk path, so build mode is a
        # within-engine comparison on a MySQL-family server -- which would
        # separate "bulk is cheaper than incremental" from "Postgres differs
        # from MySQL here", two claims that cannot be told apart while
        # pgvector is the only engine with both modes and also the only
        # Postgres. The ops harness can do it: `PerconaDriver.capabilities()`
        # derives `incremental_index` from `IndexSpec.build_mode` and
        # `create_schema`/`create_index` split accordingly.
        #
        # THE ANN PATH CANNOT, because the overlay module has no `build_mode`.
        # `pgvector/module.py:76-78` and `valkey/module.py:110-112` each read
        # it, validate it, branch on it in `fit` and report it back in
        # `get_additional`; `percona/module.py` contains the string zero
        # times, and so does the `VBMySQLBase` it inherits from. Emitting the
        # key here anyway would produce two run groups, `post_build` and
        # `incremental_build`, differing only in a `method_param` nothing
        # reads -- two IDENTICAL curves under two labels, with no field in the
        # record to say they were the same measurement. That is the shape this
        # repo guards against everywhere else, and a loud refusal is the
        # alternative it prefers.
        modes = list(ann.get("percona_build_modes", ["post"]))
        if modes != ["post"]:
            raise ValueError(
                "ann.percona_build_modes is not implemented: "
                "overlay/.../algorithms/percona/module.py has no `build_mode`, "
                "so every mode would build the index the same way and report "
                f"identical curves under different labels. Got {modes!r}. The "
                "ops harness does support it (PerconaDriver.capabilities), so "
                "a build-mode comparison is available there today; the ann "
                "half needs the overlay module to read `build_mode` the way "
                "pgvector's and valkey's do.")

        # One run group, matching `percona/config.yml`'s own `innodb` group
        # and alisql's branch above rather than pgvector's: the vector index
        # is InnoDB's, implemented in storage/innobase, so there is no second
        # storage engine to curve against. `engine` is still passed rather
        # than omitted, so a profile asking for something else gets a server
        # error naming the engine instead of silently getting InnoDB.
        _refuse_efc(engine, m_values, ef_constructions)
        run_groups.update(efc_run_groups("innodb", {
            "M": m_values,
            "efConstruction": ef_constructions,
            "engine": "InnoDB",
        }, engine, query_args))

    elif engine == "myvector":
        # Percona's shape for the same reason: MyVector exposes a build-time
        # effort knob (`ef=` in the column options, hnswlib's ef_construction)
        # and a query-time one (`ef_search` per MYVECTOR_IS_ANN call), so it is
        # pinned in the normalized pass and swept only where the tuned pass
        # asks. 200 rather than the plugin's own default of 100, so that the
        # normalized pass builds MyVector at the same ef_construction as
        # pgvector, Valkey and Percona -- every engine that has the knob.
        if resource_pass == "tuned" and extras.get("myvector_ef_construction"):
            ef_constructions = list(extras["myvector_ef_construction"])
        else:
            ef_constructions = [int(ann.get("myvector_ef_construction", 200))]
        # No build-mode axis anywhere: the overlay module and the ops driver
        # both build after the load, which is the only build the engine has
        # on this server (see `INCREMENTAL_DECLINED` in
        # harness/drivers/myvector.py).
        _refuse_efc(engine, m_values, ef_constructions)
        run_groups.update(efc_run_groups("innodb", {
            "M": m_values,
            "efConstruction": ef_constructions,
        }, engine, query_args))

    elif engine in VILLAGESQL_ENGINES:
        # Both VillageSQL server lines, one extension: the same grid and the
        # same `villagesql_ef_construction` key, so a profile cannot build the
        # 8.4 graph at a different ef_construction from the 9.7 one.
        # MyVector's shape: a build-time knob (`ef_construction` in the index
        # options) and a query-time one (`vsql_vector.ef_search`). Pinned in
        # the normalized pass at the value every engine with the knob is built
        # at, swept only where the tuned pass asks. The extension refuses
        # ef_construction < M and > 32 x M, so those combinations are dropped
        # by name (`ann_knobs.excluded_builds`), never clamped. One build
        # mode: the module declares the index in CREATE TABLE and the load
        # builds it (`capabilities.bulk_vector_index_build: false`).
        if resource_pass == "tuned" and extras.get("villagesql_ef_construction"):
            ef_constructions = list(extras["villagesql_ef_construction"])
        else:
            ef_constructions = [int(ann.get("villagesql_ef_construction", 200))]
        _refuse_efc(engine, m_values, ef_constructions)
        run_groups.update(efc_run_groups("innodb", {
            "M": m_values,
            "efConstruction": ef_constructions,
        }, engine, query_args))

    elif engine == "mongodb":
        # Quantization is the one axis no other engine has, so it follows the
        # ef_construction precedent exactly: pinned in the normalized pass, set
        # from the vendor's own guidance in the tuned pass. MongoDB advises
        # quantizing above a 3 GB vector index and 990k x 1536 float32 is
        # 5.7 GB, so `scalar` there is the vendor-recommended configuration
        # rather than a thumb on the scale.
        if resource_pass == "tuned" and extras.get("mongodb_quantization"):
            quantizations = list(extras["mongodb_quantization"])
        else:
            quantizations = [str(ann.get("mongodb_quantization", "none"))]

        for quantization in quantizations:
            # M is carried, not applied: mongot exposes no graph degree. One
            # value only, because sweeping it would produce identical curves
            # under different labels.
            run_groups[f"{quantization}_quantization"] = {
                "arg_groups": [{"M": m_values[:1], "quantization": quantization}],
                "args": {},
                "query_args": query_args,
            }

    else:
        raise ValueError(f"unknown engine: {engine}")

    # A stage-2 selection names its builds exactly (`select-builds`); see
    # `ann_knobs.restrict_to_builds` for why the lists alone were not enough.
    selected = (extras.get("ann_builds_by_engine") or {}).get(engine)
    if selected:
        run_groups = restrict_to_builds(engine, run_groups, selected)

    return {
        "float": {
            "any": [{
                "base_args": ["@metric"],
                "constructor": engines_mod.get(engine).ann_constructor,
                "disabled": False,
                "docker_tag": f"vector-bench/{engine}-bench",
                "module": f"ann_benchmarks.algorithms.{engine}",
                "name": engine,
                "run_groups": run_groups,
            }]
        }
    }


def excluded_ann_builds(engine: str, profile: Dict[str, Any],
                        resources: Dict[str, Any], resource_pass: str
                        ) -> List[Tuple[int, int, str]]:
    """(M, ef_construction, reason) for each build the profile's grid names
    and `render_config` left out because the engine cannot build it.

    Read back from the render rather than re-derived, so the list printed
    before a run is the list the container will not be asked for.
    """
    body = render_config(engine, profile, resources, resource_pass)
    m_values: List[int] = []
    efcs: List[int] = []
    for algorithm in body["float"]["any"]:
        for run_group in algorithm["run_groups"].values():
            for group in run_group["arg_groups"]:
                if "efConstruction" not in group:
                    return []
                for m in group["M"]:
                    if int(m) not in m_values:
                        m_values.append(int(m))
                for efc in group["efConstruction"]:
                    if int(efc) not in efcs:
                        efcs.append(int(efc))
    return excluded_builds(engine, m_values, efcs)


def _refuse_efc(engine: str, m_values: List[int],
                ef_constructions: List[int]) -> None:
    """`knob_problems` for ef_construction, raised as `render_config` does
    for M. Combinations legal value-by-value but not together are dropped by
    `efc_run_groups` instead, and named by `ann_knobs.excluded_builds`."""
    problems = knob_problems(engine, (), ef_constructions)
    if problems:
        raise ValueError("; ".join(problems))


# In the ann pass the database server and the ann-benchmarks client share one
# container, so they also share one cgroup memory limit. The client's share is
# not small and is not constant: ann-benchmarks loads the corpus into RAM twice
# over — once in the parent process (main.py calls get_dataset() to learn the
# dimension) and again inside the forked worker (runner.run() calls it again on
# its own). Neither copy is shared; the second is a fresh h5py read.
#
# 2.0x for those two copies, 0.2x for the transient float32 conversion numpy
# makes while reading, and a flat 1 GB for the interpreter, the connector and
# the query set.
_CLIENT_COPIES = 2.2
_CLIENT_BASE_BYTES = 1 * 1024**3


def _host_ram_bytes() -> int:
    """Total host RAM, or 0 if it cannot be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def client_memory_bytes(datasets_dir: str, dataset: str) -> int:
    """Estimate what the ann-benchmarks client needs resident for this dataset.

    Sized from the HDF5 file on disk rather than a lookup table so that
    generated corpora (the dbpedia family) are covered without anyone having to
    remember to add them. The train matrix is >98% of these files, which makes
    file size a good proxy and, unlike h5py, one the host orchestrator can read
    without any dependency at all.
    """
    path = os.path.join(datasets_dir, f"{dataset}.hdf5")
    try:
        size = os.path.getsize(path)
    except OSError:
        return _CLIENT_BASE_BYTES
    return int(size * _CLIENT_COPIES) + _CLIENT_BASE_BYTES


def run_engine(engine: str, dataset: str, profile: Dict[str, Any],
               engine_cfg: Dict[str, Any], resolved: ResolvedResources,
               resource_pass: str, paths: Dict[str, str], run_id: str,
               force: bool = False, timeout_s: int = 24 * 3600) -> int:
    """Run the ann-benchmarks sweep for one engine on one dataset."""
    image = engine_cfg.get("image", {}).get("bench", f"vector-bench/{engine}-bench")
    if not docker_ctl.image_exists(image):
        raise docker_ctl.DockerError(
            f"image {image} not found. Build it first:\n"
            f"  ./run-benchmark.sh build --engines {engine}"
        )

    flags = server_args(engine_cfg, resource_pass, resolved)
    container = f"{run_id}-annb-{engine}-{dataset}".replace("_", "-")[:60]

    # The engine's own budget stays exactly as the resource profile specifies —
    # that is what makes the normalized pass comparable across engines. The
    # client's copies of the corpus are not an engine resource, so they are
    # added on top of the container limit rather than taken out of it.
    #
    # Sizing the container to the engine budget alone silently capped the client
    # too: on dbpedia-openai-1000k (6.1 GB corpus) the worker was OOM-killed the
    # moment it finished loading, and because only the forked child died, the
    # parent exited 0 with no results and no error.
    client_bytes = client_memory_bytes(paths["datasets"], dataset)
    container_memory_bytes = resolved.server_memory_bytes + client_bytes

    # Without this the engine writes its data directory into the container's
    # writable layer, which lives under Docker's data-root — usually the root
    # filesystem, and usually not where the space is. A pgvector run died at
    # `initdb: could not create directory ".../pg_wal": No space left on
    # device` while the filesystem holding the checkout had 100+ GB free.
    #
    # Each configuration builds a fresh index, so the directory is cleared
    # rather than reused; leaving it would also make the ingest measurement
    # depend on whatever the previous run left behind.
    #
    # Cleared through `remove_tree_as_root`, as the ops pass clears its own: the
    # engine writes this directory as root from inside the container, so a
    # plain `shutil.rmtree(..., ignore_errors=True)` running as the host user
    # removed nothing and said nothing, and the next run started on the
    # previous run's datadir. With a server binary newer than the one that
    # wrote it, a data-dictionary upgrade guard then refuses to start
    # ("Upgrading from a development version is not allowed") and every
    # configuration of the sweep dies at startup.
    state_dir = os.path.join(paths["engine_state"], "annb", f"{resource_pass}-{engine}")
    if not docker_ctl.remove_tree_as_root(state_dir, image):
        print(f"[ann] WARNING: {state_dir} survived teardown (still root-owned); "
              f"the next server starts on the previous run's datadir",
              file=sys.stderr)
    os.makedirs(state_dir, exist_ok=True)

    command = [
        "run.py", "--local",
        "--algorithm", engine,
        "--dataset", dataset,
        "--count", str(profile.get("k", 10)),
        "--runs", str(profile.get("runs", 1)),
        "--parallelism", "1",
    ]
    if profile.get("ann", {}).get("batch"):
        command.append("--batch")
    if force:
        command.append("--force")

    spec = docker_ctl.ContainerSpec(
        name=container,
        image=image,
        # No network needed: the dataset is mounted and the server is in-process.
        network="none",
        cpuset=resolved.server_cpuset,
        memory_bytes=container_memory_bytes,
        shm_size=resolved.shm_size,
        entrypoint="python3",
        workdir="/home/app",
        env={
            "VB_SERVER_ARGS": " ".join(flags),
            "VB_INSERT_THREADS": str(profile.get("ann", {}).get("insert_threads", 0)),
            "VB_RUN_ID": run_id,
            "VB_RESOURCE_PASS": resource_pass,
            # Sized centrally so both measurement paths give mongot the same
            # heap, and so `engines_resolved` records exactly this env.
            **server_env(resolved),
            "PYTHONUNBUFFERED": "1",
        },
        volumes=[
            f"{paths['work_annb']}:/home/app:rw",
            f"{paths['datasets']}:/home/app/data:rw",
            f"{annb_results_dir(paths, resource_pass, resolved)}:/home/app/results:rw",
            f"{state_dir}:{engines_mod.get(engine).data_mount}:rw",
        ],
        command=command,
        detach=False,
    )

    print(f"[ann] {engine} / {dataset}: cpuset={resolved.server_cpuset} "
          f"mem={container_memory_bytes / 1024**3:.1f}GB "
          f"(engine {resolved.server_memory_bytes / 1024**3:.1f}GB "
          f"+ client {client_bytes / 1024**3:.1f}GB for the in-RAM corpus)")
    print(f"[ann] server flags: {' '.join(flags)}")

    # Better to say this now than to have the OOM killer say it later, when it
    # will arrive as a silent zero-result exit rather than an error.
    host_ram = _host_ram_bytes()
    if host_ram and container_memory_bytes > host_ram * 0.9:
        print(
            f"[ann] WARNING: this container needs "
            f"{container_memory_bytes / 1024**3:.1f} GB but the host has only "
            f"{host_ram / 1024**3:.1f} GB. Expect the kernel to kill the client "
            f"mid-load. Lower memory.server_limit_gb in the resource profile, or "
            f"use a smaller corpus.",
            file=sys.stderr,
        )

    results_dir = annb_results_dir(paths, resource_pass, resolved)
    pruned = prune_empty_results(results_dir, image)
    if pruned:
        print(f"[ann] discarded {pruned} result file(s) that contain no "
              f"measurements; those configurations will be recomputed")
    before = _count_results(results_dir, engine, dataset)
    # What the rendered config asks for, so a sweep that stopped half way can
    # be told from one that finished. See the shortfall check at the end.
    expected = expected_results(paths["work_annb"], engine)

    # Warn before the fact, not after. ann-benchmarks signals "everything is
    # already done" by raising and printing a full traceback, which is
    # indistinguishable from a crash unless you already know to expect it.
    if before and not force:
        print(f"[ann] {before} existing result file(s) for {engine} / "
              f"{dataset}; already-computed configurations will be skipped")
    # The ops path has sampled its server all along; the ann path sampled
    # nothing, which is why three contaminated recall points had to be
    # identified five days later from a latency ratio rather than read off a
    # graph. The series carries host load alongside the container's own usage,
    # so a phase measured under someone else's job says so.
    memory_timeseries = os.path.join(
        paths["run_dir"],
        f"mem-{engine}-{dataset}-{resource_pass}-ann.jsonl")
    sampler = docker_ctl.MemorySampler(
        container, memory_timeseries, wait_for_start_s=120.0)
    sampler.start()

    output: List[str] = []
    try:
        rc = docker_ctl.run_foreground(
            spec, timeout=timeout_s, sink=output,
            # Only filter when a benign traceback is actually possible.
            line_filter=_SuppressNothingToRun() if (before and not force) else None,
        )
    finally:
        sampler.stop()
        if sampler.samples:
            print(f"[ann] captured {sampler.samples} resource samples -> "
                  f"{os.path.basename(memory_timeseries)}")
    after = _count_results(results_dir, engine, dataset)
    text = "\n".join(output)

    # Kept beside the measurements. A phase that writes no results explains
    # itself here and nowhere else, and a terminal is not an archive.
    saved = docker_ctl.save_phase_log(
        paths["run_dir"], engine, dataset, "ann", resource_pass, output)
    if saved:
        print(f"[ann] log -> {os.path.relpath(saved, paths['run_dir'])}")

    # ann-benchmarks treats "every configuration already has results" as an
    # error: main() raises Exception("Nothing to run") and the process exits
    # non-zero. For us that is successful resumption, not failure — the whole
    # point of leaving the results tree in place between runs. Recognise it
    # rather than reporting three healthy engines as broken.
    if rc != 0 and NOTHING_TO_RUN in text:
        # GATED ON THE SAME EXPECTATION AS THE CLEAN-EXIT PATH BELOW, because
        # "Nothing to run" does NOT mean "every configuration is done". It
        # means the definition list was empty by the time main() looked, and
        # already-done is only one of the ways that happens. The other one
        # matters here: in --local mode `check_module_import_and_constructor`
        # drops a definition whose module will not import with a logged
        # warning and no raise (NO_MODULE; a missing *constructor* does raise),
        # and the already-run filter runs BEFORE it. So a shared tree holding
        # 16 of 24 plus a bench image whose algorithm module lost a dependency
        # in a rebuild gives: 8 definitions survive the resume filter, all 8
        # are dropped for the import, the list is empty, main() raises.
        # Reading that as success checkpointed the phase, --resume never came
        # back, and the report carried 16 of 24 curves -- which is exactly the
        # defect the shortfall check below exists to prevent, one branch away
        # from it.
        if after > 0 and (not expected or after >= expected):
            print(
                f"[ann] nothing to do: all {after} configuration(s) for "
                f"{engine} / {dataset} already have results in {results_dir}. "
                f"Pass --force to recompute them."
            )
            return 0
        print(
            f"[ann] FAILED: ann-benchmarks had nothing to run, and "
            f"{results_dir} holds {after} result file(s)"
            + (f" where the rendered config asks for {expected}."
               if expected else " and the rendered config could not be read.")
            + f"\n[ann] Either the config matched no definitions, or the "
            f"algorithm module failed to import and every remaining "
            f"definition was silently dropped -- ann-benchmarks logs that as a "
            f"warning and then reports having nothing to run. Look for "
            f"'could not be loaded; skipping' in the log saved beside the "
            f"measurements.",
            file=sys.stderr,
        )
        return 1

    # A clean exit that produced nothing at all, with nothing already present,
    # means every configuration errored inside the run loop — ann-benchmarks
    # logs those per-definition and still exits 0. Without this check a whole
    # engine disappears from the report as an apparent success.
    if rc == 0 and after == 0:
        # ann-benchmarks runs each definition in a forked worker and never
        # checks its exit code: main() joins, logs "Terminating N workers" from
        # a finally block, and returns 0. So a worker killed by the OOM killer
        # is indistinguishable from a clean run except that nothing was written.
        # That is by far the most common cause here, because the client holds
        # the whole corpus in RAM twice, so name it first and show the numbers.
        killed_silently = "Terminating" in text and "Traceback" not in text
        print(
            f"[ann] FAILED: {engine} / {dataset} exited 0 but produced no result "
            f"files at all in {results_dir}.",
            file=sys.stderr,
        )
        if killed_silently:
            print(
                f"[ann] The worker died without raising, which almost always means "
                f"the kernel OOM-killed it: ann-benchmarks does not check worker "
                f"exit codes, so a kill shows up as a silent zero-result success.\n"
                f"[ann]   container limit : {container_memory_bytes / 1024**3:.1f} GB\n"
                f"[ann]   engine budget   : {resolved.server_memory_bytes / 1024**3:.1f} GB "
                f"(buffer pool is allocated up front)\n"
                f"[ann]   client estimate : {client_bytes / 1024**3:.1f} GB "
                f"(corpus is held in RAM twice)\n"
                f"[ann] Confirm with: dmesg -T | tail  — look for Killed process ... python3",
                file=sys.stderr,
            )
        else:
            print(
                f"[ann] The algorithm module most likely failed to import, or every "
                f"configuration errored — check the container output above.",
                file=sys.stderr,
            )
        return 1
    if rc == 0:
        # THE SHORTFALL TEST COMES FIRST, before anything is reported as done.
        # Reporting "nothing to do: all 16 configuration(s) already have
        # results" and then "FAILED: 16 of 24" on the next line, at the end of
        # a seven-hour phase, is a log that argues with itself.
        #
        # A PARTIAL SWEEP IS NOT A COMPLETE ONE, and until this check it was
        # recorded as one: the case above catches the total loss
        # (`after == 0`) and nothing caught the partial, because `after`
        # against `before` cannot see it.
        #
        # Measured, in run full-20261015-215006: the gist-960 normalized ANN
        # phase ran 7 h 46 m, lost its M=8 curve to
        # `mariadb.InterfaceError: Server has gone away` during the parallel
        # load, wrote 16 of the 24 files the config asks for, exited 0, and was
        # checkpointed as `normalized/percona/gist-960-euclidean/ann` -- so
        # `--resume` will never retry it and the lost curve is
        # indistinguishable in the report from a curve nobody asked for.
        #
        # `after > expected` is deliberately NOT a failure: the results tree
        # is shared across runs and keyed only by the resource configuration,
        # so a wider grid measured earlier at the same budget is legitimately
        # still in it.
        if expected and after < expected:
            print(
                f"[ann] FAILED: {engine} / {dataset} exited 0 with {after} of "
                f"{expected} result file(s) in {results_dir} -- "
                f"{expected - after} short.\n"
                f"[ann] ann-benchmarks runs each definition in a forked worker "
                f"and does not check its exit code, so a configuration that "
                f"died mid-sweep exits 0 and looks complete. Failing the phase "
                f"is what keeps it out of the checkpoints and brings --resume "
                f"back for it; the worker's traceback is in the log saved "
                f"beside the measurements.",
                file=sys.stderr,
            )
            return 1

        if after == before:
            print(f"[ann] nothing to do: all {after} configuration(s) for "
                  f"{engine} / {dataset} already have results "
                  f"(pass --force to recompute)")
        else:
            print(f"[ann] {after - before} new result file(s), {after} total")
    return rc


class _SuppressNothingToRun:
    """Hide ann-benchmarks' traceback when it only means "already done".

    ann-benchmarks reports "every configuration has results" by raising, so it
    prints a full Python traceback for a condition that is completely normal
    here. Explaining that in advance was not enough — a traceback in the output
    reads as a crash no matter what precedes it.

    Lines are held from "Traceback (most recent call last):" onwards and
    released only if the traceback turns out to be something else. A real
    failure is therefore never swallowed, merely delayed by a few lines.
    """

    def __init__(self) -> None:
        self._held: Optional[List[str]] = None

    def __call__(self, line: Optional[str]) -> List[str]:
        if line is None:                              # end of stream
            held, self._held = self._held, None
            return held or []

        if self._held is None:
            if line.startswith("Traceback (most recent call last)"):
                self._held = [line]
                return []
            return [line]

        self._held.append(line)
        if line.startswith("Exception: ") or line.startswith("  File "):
            if NOTHING_TO_RUN in line:
                self._held = None                     # benign: drop it entirely
                return []
            if line.startswith("Exception: "):
                held, self._held = self._held, None   # some other error: show it
                return held
        return []


# store_results() opens the file, writes the attributes, and only then creates
# the datasets:
#
#     with h5py.File(filename, "w") as f:
#         for k, v in attrs.items():
#             f.attrs[k] = v
#         times = f.create_dataset("times", ...)
#
# An attribute h5py cannot represent therefore leaves a syntactically valid
# HDF5 file holding no measurements at all -- and ann-benchmarks decides what
# to skip with os.path.exists(). One `"ef_construction": None` in a module's
# get_additional() was enough to make Percona Search write a stub for every
# ef_search it reached, and each subsequent run then skipped that point as
# already done. Four attempts later its recall curve was two points at the top
# of the sweep, and the six below them could not be recomputed by re-running:
# the stubs outlived the bug that made them.
#
# The modules no longer emit unwritable attributes, but a stub is permanent
# once written, so the ones already on disk have to go. h5py is not installed
# on the host -- deliberately, see client_memory_bytes -- so the check runs in
# the image that has it.
_PRUNE_SCRIPT = """
import os, sys
import h5py

removed = 0
for base, _dirs, files in os.walk("/results"):
    for name in files:
        if not name.endswith(".hdf5"):
            continue
        path = os.path.join(base, name)
        try:
            with h5py.File(path, "r") as fh:
                intact = "times" in fh and "neighbors" in fh
        except Exception:
            intact = False
        if not intact:
            try:
                os.remove(path)
                removed += 1
                print("pruned " + path, file=sys.stderr)
            except OSError:
                pass
print(removed)
"""


def prune_empty_results(results_dir: str, image: str) -> int:
    """Delete result files that hold no measurements, and count them.

    Returns 0 rather than raising if the check itself cannot run: this exists
    to stop a stale stub from silently suppressing a configuration, and failing
    the run over it would be a worse trade than leaving it in place, where at
    least the report's own empty-curve warning still fires.
    """
    output: List[str] = []
    spec = docker_ctl.ContainerSpec(
        name=f"vb-prune-{os.getpid()}",
        image=image,
        entrypoint="python3",
        command=["-c", _PRUNE_SCRIPT],
        volumes=[f"{results_dir}:/results:rw"],
        detach=False,
        network="none",
    )
    try:
        rc = docker_ctl.run_foreground(spec, timeout=300, stream=False,
                                       sink=output)
    except docker_ctl.DockerError:
        return 0
    if rc != 0:
        return 0
    for line in reversed(output):
        try:
            return int(line.strip())
        except ValueError:
            continue
    return 0


def _count_results(results_dir: str, engine: str, dataset: str) -> int:
    """Number of ann-benchmarks result files for this engine and dataset."""
    root = os.path.join(results_dir, dataset)
    if not os.path.isdir(root):
        return 0
    return sum(
        len([f for f in files if f.endswith(".hdf5")])
        for base, _dirs, files in os.walk(root)
        if os.path.basename(base) == engine
    )


def fix_ownership(path: str, image: str) -> None:
    """Return root-created result files to the invoking user.

    The engines run as root inside their containers, so anything they write to a
    bind mount lands root-owned on the host. Left alone that makes the results
    directory unreadable to the user who started the run.
    """
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        return
    spec = docker_ctl.ContainerSpec(
        name=f"vb-chown-{os.getpid()}",
        image=image,
        entrypoint="chown",
        command=["-R", f"{uid}:{gid}", "/target"],
        volumes=[f"{path}:/target:rw"],
        detach=False,
        network="none",
    )
    docker_ctl.run_foreground(spec, timeout=600, stream=False)
