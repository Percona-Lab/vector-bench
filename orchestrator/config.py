"""Configuration loading and resolution for the orchestrator.

Three config layers combine into one resolved plan:

  profiles/<name>.yml    what to measure (datasets, grids, workloads)
  resources/<pass>.yml   how much machine each engine gets
  engines/<engine>.yml   how each engine is built, started and spoken to

Resolution turns fractions and "0 means auto" placeholders into concrete
numbers — cpusets, byte counts, server flags — because a manifest that records
"buffer_fraction: 0.35" tells a reader nothing, while one that records
"innodb_buffer_pool_size=6012954214" is reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(VB_ROOT, "config")

GB = 1024 ** 3


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_profile(name: str) -> Dict[str, Any]:
    return _load(os.path.join(CONFIG_DIR, "profiles", f"{name}.yml"))


# A profile's `engines:` key is a RESTRICTION on the run, not a decoration, and
# until these functions existed nothing HONOURED it. Two tests read it --
# `test_e2_percona_wiring.py`'s `TestTheProfiles::
# test_the_percona_profile_runs_only_percona` asserts the declared value, and
# `TestEveryShippedProfileRenders` resolves it the way this file now does, under
# a comment that attributed the resolution to `cmd_render`, which never did it.
# So the tree knew what the key meant, in a test, and no production code acted
# on it.
#
# `config/profiles/percona.yml` has declared `engines: [percona]` since it was
# written, under an opening paragraph that says "NOT a comparison profile. It
# runs one engine". `cli.cmd_run` resolved the list as
# `args.engines or ",".join(KNOWN_ENGINES)` and never looked at the profile, so
# the declaration was inert prose beside live code. The run at
# `percona-20261020-140151`
# measured ALL SEVEN engines under it -- 14 phases, two per engine, with
# `ops-*`, `mem-*` and recall records for alisql, mariadb, mariadb123, mongodb,
# pgvector and valkey beside percona -- and its manifest records
# `config.profile.engines: ['percona']` next to seven populated `engines` keys.
# It finished `status: completed`, because an engine nobody asked for cannot
# fail.
#
# What it cost is worse than six engines' worth of time. Every axis this
# profile has is Percona-only -- `resources.extras.percona_ef_construction`,
# `ops.percona_build_modes` -- and `ops.subset_rows: 20000` was chosen to make
# three Percona corpus reloads affordable, not to measure anything about
# MariaDB. So the other six were measured at a grid designed for a different
# engine, on a corpus sized for a different question, in a run directory whose
# profile says one engine ran. That is this repo's signature failure: not a
# crash, a credible-looking result that answers a question nobody asked.
#
# A RESTRICTION rather than a default, and the difference is the whole
# precedence rule. `--datasets` overrides `profile.datasets` outright two lines
# below the engine resolution in `cmd_run`, and copying that shape here would
# have been wrong: a dataset list is a choice of corpus, while a profile that
# names engines has a GRID that is only meaningful on them. `--profile percona
# --engines mariadb` therefore has two readings -- "measure mariadb" and
# "measure what the percona profile measures" -- which differ by a whole run,
# and a stale command line is a likelier source of it than an intention. So an
# explicit flag may NARROW the profile's list and `cmd_run` refuses to widen
# it, rather than silently picking a winner. See the refusal there.


def profile_engines(profile: Dict[str, Any]) -> List[str]:
    """The engines a profile restricts itself to, or [] when it names none.

    [] rather than every known engine, deliberately. This module cannot see the
    registry -- `orchestrator/engines.py` imports *this* file, so that
    dependency runs one way only -- and the caller holds `KNOWN_ENGINES`
    anyway. Returning [] keeps the fall-back visible at the call site, as
    `declared or KNOWN_ENGINES`, instead of burying a seven-engine default
    inside a function named after one profile's key.

    Every shipped profile but `percona` declares none, which is why honouring
    the key changes exactly one profile's behaviour and leaves the hard-won
    "a bare run measures everything" default (`cli.KNOWN_ENGINES`) intact.

    De-duplicated, order preserved. A repeated name is not a request for two
    measurements: the run loop would start the engine twice and write both
    units into one run directory under one checkpoint key, so the second
    silently overwrites the first and the report shows one. `cmd_run` applies
    the same reduction to the resolved list, which is where `--engines
    percona,percona` arrives.
    """
    declared = (profile or {}).get("engines")
    if not declared:
        return []
    return dedupe(str(engine).strip() for engine in declared
                  if str(engine).strip())


def dedupe(names: Any) -> List[str]:
    """First occurrence of each name, in order."""
    return list(dict.fromkeys(names))


def profile_engines_problem(profile: Dict[str, Any]) -> Optional[str]:
    """Why a profile's `engines:` key cannot be honoured, or None.

    **AN EMPTY OR NULL LIST RE-ARMS THE DEFECT THIS KEY EXISTS TO PREVENT, SO
    IT IS REFUSED RATHER THAN READ AS "NO RESTRICTION".** `profile_engines`
    answers [] for an absent key and for `engines: []` alike, and the call site
    spells its fall-back `declared or KNOWN_ENGINES` -- so without this check
    an operator who opens `percona.yml` in the web UI's profile editor and
    empties the list gets all seven engines at the Percona-only grid, with no
    note and no warning, completing and reporting success. That is
    `percona-20261020-140151` verbatim, which is the run this key was made
    load-bearing to prevent.

    An ABSENT key is a different statement and stays silent: a profile that
    never mentions engines is making no claim about which ones its grid suits,
    and nine of the ten shipped profiles are in that case. A key that is
    PRESENT and says nothing usable is a half-finished edit, and this repo
    prefers failing loudly to producing a plausible wrong result.

    A bare string is refused for a quieter reason: `engines: percona` iterates
    as seven one-character names, so without this the refusal below fired
    anyway and the note read "runs only p, e, r, c, o, n, a".
    """
    if "engines" not in (profile or {}):
        return None
    declared = (profile or {}).get("engines")
    if (not isinstance(declared, list) or not declared
            or not all(isinstance(e, str) and e.strip() for e in declared)):
        return (f"engines must be a non-empty list of engine names; "
                f"got {declared!r}. Remove the key entirely to run every "
                f"engine -- an empty list reads as 'no restriction' and would "
                f"measure all of them at this profile's grid.")
    return None


def engines_outside_profile(profile: Dict[str, Any],
                            requested: List[str]) -> List[str]:
    """Requested engines the profile's own list excludes, in request order.

    Empty for a profile that declares no list: there is nothing to be outside
    of, and a profile naming no engines is making no claim about which ones its
    grid suits.
    """
    declared = set(profile_engines(profile))
    if not declared:
        return []
    return [engine for engine in requested if engine not in declared]


def load_resources(name: str) -> Dict[str, Any]:
    return _load(os.path.join(CONFIG_DIR, "resources", f"{name}.yml"))


def merge_resource_overrides(resources: Dict[str, Any],
                             profile: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a profile's `resources:` block on top of the resource pass.

    Resource passes are sized for the common case, and the common case is a
    corpus that fits comfortably. A profile that loads a million 1536-dimension
    vectors needs a bigger budget than one that loads 60,000 at 784, and without
    this the only options were to edit the shared pass (changing every other
    profile) or to run under a budget that cannot hold the data.

    Returns a new dict; the inputs are not modified, so a caller can resolve
    several profiles against one loaded pass.
    """
    override = (profile or {}).get("resources") or {}
    if not override:
        return resources

    def deep_merge(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for key, value in top.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    merged = deep_merge(resources, override)
    merged["_overridden_by_profile"] = sorted(override)
    return merged


def load_engine(name: str) -> Dict[str, Any]:
    return _load(os.path.join(CONFIG_DIR, "engines", f"{name}.yml"))


def available_profiles() -> List[str]:
    d = os.path.join(CONFIG_DIR, "profiles")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".yml"))


@dataclass
class ResolvedResources:
    """Concrete resource settings, ready to hand to Docker and to the servers."""

    name: str
    server_cpuset: str
    client_cpuset: str
    server_cpu_count: int
    server_memory_bytes: int
    client_memory_bytes: int
    buffer_bytes: int
    graph_cache_bytes: int
    maintenance_bytes: int
    #: JVM heap for mongot. Zero for every engine that is a single process.
    mongot_heap_bytes: int
    #: Valkey's maxmemory. Zero for every engine that is not in-memory.
    maxmemory_bytes: int
    #: Network IO threads, clamped. Not the same as the server's cpuset.
    io_threads: int
    build_threads: int
    shm_size: str
    transaction_isolation: str
    hybrid_cpu: bool
    core_class_used: str
    #: The pass's declared knobs, before any engine-specific adjustment.
    #: Identical for every engine under one pass by construction, which is what
    #: lets the ann fingerprint notice a changed pass without fragmenting the
    #: results tree by engine.
    pass_signature: str = ""
    warnings: List[str] = field(default_factory=list)
    #: mongod's WiredTiger cache. Zero for every engine that is not mongodb.
    #: Set explicitly since ops measurement version 9 -- see the mongodb
    #: branch of `resolve_resources` for the run that made it necessary.
    wiredtiger_cache_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "server_cpuset": self.server_cpuset,
            "client_cpuset": self.client_cpuset,
            "server_cpu_count": self.server_cpu_count,
            "server_memory_bytes": self.server_memory_bytes,
            "client_memory_bytes": self.client_memory_bytes,
            "buffer_bytes": self.buffer_bytes,
            "graph_cache_bytes": self.graph_cache_bytes,
            "maintenance_bytes": self.maintenance_bytes,
            "mongot_heap_bytes": self.mongot_heap_bytes,
            "maxmemory_bytes": self.maxmemory_bytes,
            "wiredtiger_cache_bytes": self.wiredtiger_cache_bytes,
            "io_threads": self.io_threads,
            "pass_signature": self.pass_signature,
            "build_threads": self.build_threads,
            "shm_size": self.shm_size,
            "transaction_isolation": self.transaction_isolation,
            "hybrid_cpu": self.hybrid_cpu,
            "core_class_used": self.core_class_used,
            "warnings": self.warnings,
        }


#: The fewest client CPUs `client_cpus: auto` resolves to. The pass default is
#: 4, and a one-client sweep needs no fewer than it already gets.
AUTO_CLIENT_CPUS_FLOOR = 4


def auto_client_cpus(max_clients: int) -> int:
    """What `client_cpus: auto` resolves to for a sweep up to `max_clients`.

    One client CPU per two clients, never fewer than four. Each client is a
    forked OS process that spends most of a query waiting on the server, so
    two share a CPU without becoming the ceiling; one per client would take
    from the server the cores the comparison is about. `tuned-complete`
    pinned 16 by hand for its 32-client sweep, which is this rule.
    """
    return max(AUTO_CLIENT_CPUS_FLOOR, -(-int(max_clients) // 2))


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def profile_max_clients(profile: Dict[str, Any]) -> Optional[int]:
    """The largest client count the profile's concurrency sweep asks for."""
    counts = ((profile or {}).get("ops", {}) or {}).get("client_counts") or [1]
    try:
        return max(int(c) for c in counts)
    except (TypeError, ValueError):
        return None


def resolve_resources(resources: Dict[str, Any], engine: str,
                      sysinfo: Any,
                      max_clients: Optional[int] = None) -> ResolvedResources:
    """Turn a resource profile plus the detected hardware into concrete limits.

    `max_clients` is the largest client count of the run's concurrency sweep,
    which `client_cpus: auto` sizes the client cpuset from. Only a caller that
    knows the profile can supply it, so auto without it is refused rather
    than guessed.
    """
    from harness.metrics.sysinfo import format_cpuset, recommended_cpuset

    cpu_cfg = resources.get("cpu", {}) or {}
    mem_cfg = resources.get("memory", {}) or {}
    build_cfg = resources.get("build", {}) or {}
    docker_cfg = resources.get("docker", {}) or {}
    iso_cfg = resources.get("isolation", {}) or {}

    warnings: List[str] = []

    prefer = cpu_cfg.get("prefer_cores", "performance")
    allow_smt = bool(cpu_cfg.get("allow_smt", False))

    want_server = int(cpu_cfg.get("server_cpus", 0) or 0)
    # `auto` sizes the client set from the sweep instead of a hand-kept pin.
    # A 32-client sweep from the pass default of 4 client CPUs measures the
    # clients' ceiling and publishes it as the engine's, which is why
    # `tuned-complete` pinned 16 by hand -- and nothing kept that number in
    # step with its `client_counts`.
    auto_client = _is_auto(cpu_cfg.get("client_cpus"))
    if auto_client:
        if not max_clients:
            raise ValueError(
                "resources.cpu.client_cpus is `auto`, which is sized from the "
                "largest concurrency client count, and none was given to "
                "resolve it against")
        want_client = auto_client_cpus(max_clients)
    else:
        want_client = int(cpu_cfg.get("client_cpus", 2) or 2)

    # Resolve against the topology we were handed, not the live machine. In a
    # run they are the same host; in anything that passes a topology in, they
    # were not, and the pool silently came from wherever the code happened to
    # be running.
    pool = recommended_cpuset(10 ** 6, prefer=prefer, allow_smt=allow_smt,
                              cpu=sysinfo.cpu)
    if not pool:
        pool = list(range(sysinfo.cpu.logical_cpus))
        warnings.append("could not determine core topology; using all logical CPUs")

    if want_server <= 0:
        # "all available", minus what the client needs.
        want_server = max(1, len(pool) - want_client)

    if auto_client and want_server + want_client > len(pool):
        # A FAILURE, not the warning a numeric pin gets below. A pin is an
        # operator's explicit choice and sharing cores is a visible trade they
        # made; `auto` is a promise that the clients will not be the ceiling,
        # and quietly shrinking the set breaks it in exactly the runs -- the
        # big sweeps -- where it matters. Taken after the server set, so the
        # server keeps the cores the pass gave it.
        left = max(0, len(pool) - want_server)
        raise ValueError(
            f"resources.cpu.client_cpus: auto needs {want_client} client "
            f"CPU(s) for a sweep up to {max_clients} clients, but the "
            f"{len(pool)}-CPU pool (prefer_cores={prefer}, "
            f"allow_smt={allow_smt}) has only {left} left after the "
            f"{want_server}-CPU server set. Lower the largest client count, "
            f"give the server fewer CPUs, or pin client_cpus to a number to "
            f"accept shared cores.")

    if want_server + want_client > len(pool):
        warnings.append(
            f"requested {want_server} server + {want_client} client CPUs but only "
            f"{len(pool)} homogeneous CPUs are available "
            f"(prefer_cores={prefer}, allow_smt={allow_smt}); "
            f"server and client will share cores, which adds latency noise"
        )
        want_server = max(1, len(pool) - 1)
        want_client = max(1, len(pool) - want_server)

    server_cpus = pool[:want_server]
    client_cpus = pool[want_server:want_server + want_client] or pool[-want_client:]

    if sysinfo.cpu.hybrid and prefer == "performance" and not sysinfo.cpu.performance_cpus:
        warnings.append(
            "hybrid CPU detected but performance cores could not be identified; "
            "results will carry P-core/E-core scheduling noise"
        )

    # --- memory -------------------------------------------------------
    limit_gb = float(mem_cfg.get("server_limit_gb", 0) or 0)
    if limit_gb > 0:
        server_bytes = int(limit_gb * GB)
    else:
        fraction = float(mem_cfg.get("host_fraction", 0.6))
        server_bytes = int(sysinfo.total_ram_bytes * fraction)

    if server_bytes > sysinfo.total_ram_bytes:
        warnings.append(
            f"requested server memory ({server_bytes / GB:.1f} GB) exceeds host RAM "
            f"({sysinfo.total_ram_bytes / GB:.1f} GB); clamping"
        )
        server_bytes = int(sysinfo.total_ram_bytes * 0.8)

    client_bytes = int(float(mem_cfg.get("client_limit_gb", 8) or 8) * GB)

    buffer_fraction = float(mem_cfg.get("buffer_fraction", 0.35))
    graph_fraction = float(mem_cfg.get("graph_cache_fraction", 0.35))
    maint_fraction = float(mem_cfg.get("maintenance_fraction", 0.15))

    if engine == "pgvector" and mem_cfg.get("postgres_absorbs_graph_cache", True):
        # PostgreSQL has no vector-specific cache; graph pages live in
        # shared_buffers. Handing pgvector only the buffer fraction would give
        # it strictly less resident memory than the other two for the same
        # container limit, which would not be a fair normalization.
        buffer_fraction = buffer_fraction + graph_fraction
        graph_fraction = 0.0

    # Taken here, before any engine-specific adjustment below. Every engine
    # under one pass sees the same declared knobs; what each does with them
    # differs by design, so hashing the resolved split fragments the results
    # tree by engine and hashing this does not.
    pass_signature = "|".join(str(x) for x in (
        resources.get("name", ""),
        mem_cfg.get("server_limit_gb", ""), mem_cfg.get("host_fraction", ""),
        mem_cfg.get("buffer_fraction", ""), mem_cfg.get("graph_cache_fraction", ""),
        mem_cfg.get("maintenance_fraction", ""),
        # The RESOLVED count for `auto`, the declared one otherwise. The
        # signature keys the shared ann tree, and `auto` over a 1-32 sweep is
        # the same 16 CPUs `tuned-complete` used to pin -- hashing the word
        # would move that profile's tree and re-measure every recall point
        # for nothing.
        cpu_cfg.get("server_cpus", ""),
        want_client if auto_client else cpu_cfg.get("client_cpus", ""),
        build_cfg.get("threads", ""), build_cfg.get("max_threads", ""),
    ))

    # Network IO threads are not the cpuset. Valkey's own guidance is that
    # more than a handful is counterproductive, and the framework already
    # clamps build threads for the same reason. Handing it the whole cpuset
    # gave a 64-core server 64 IO threads, and its writes into a populated
    # search index stopped after three rows while reads stayed healthy. The
    # same engine on a small cpuset, which is what every smoke run used,
    # completed the identical workload.
    io_cap = int(cpu_cfg.get("max_io_threads", 8) or 8)
    io_threads = max(1, min(len(server_cpus), io_cap))
    if len(server_cpus) > io_cap:
        warnings.append(
            f"io threads clamped from {len(server_cpus)} to {io_threads}; "
            f"more than a handful contend rather than help, and a large value "
            f"stalled writes into a populated index entirely (raise "
            f"cpu.max_io_threads to override)"
        )

    maxmemory_bytes = 0
    if engine == "valkey":
        # Nothing here lives on disk, so there is no share of a larger thing to
        # reserve. buffer_fraction and graph_cache_fraction both describe memory
        # set aside to hold part of something bigger; carving them out of an
        # in-memory store would leave the data less room than the container was
        # given, for no benefit.
        buffer_fraction = 0.0
        graph_fraction = 0.0
        maint_fraction = 0.0

        # maxmemory sits below the container limit on purpose. Reaching
        # maxmemory under noeviction returns an error the driver can report;
        # reaching the container limit gets the process OOM-killed, which
        # arrives as a crash with no cause attached to it.
        headroom = float((resources.get("memory", {}) or {}).get(
            "valkey_maxmemory_fraction", 0.9))
        maxmemory_bytes = int(server_bytes * headroom)

        # An in-memory engine given less memory than the dataset does not get
        # slower, it fails partway through the load. Saying so now beats
        # discovering it after an hour of writing.
        corpus_bytes = int((resources.get("memory", {}) or {}).get(
            "expected_corpus_bytes", 0) or 0)
        # No default. The caller knows the dataset and passes its size in; a
        # warning derived from a number invented here would fire on every
        # small run and stop being read.
        if corpus_bytes and maxmemory_bytes < corpus_bytes:
            warnings.append(
                f"the corpus does not fit: maxmemory resolves to "
                f"{maxmemory_bytes / GB:.1f} GB and an in-memory engine needs "
                f"roughly {corpus_bytes / GB:.0f} GB for 990k x 1536 vectors "
                f"plus the graph. Valkey will refuse writes partway through the "
                f"load rather than running slowly."
            )

    mongot_heap_bytes = 0
    if engine == "mongodb":
        # The opposite of the pgvector rule above, for the opposite reason.
        #
        # Percona Search is two processes: mongod holds the documents in its
        # WiredTiger cache, and mongot holds the index as memory-mapped Lucene
        # segments served from the OS filesystem cache. The filesystem cache is
        # unallocated memory by definition, so the graph share cannot be
        # absorbed into anything -- it has to be left free. Giving it to either
        # process starves the cache that actually answers queries, which is
        # what MongoDB's guidance means by warning against a heap above 50%.
        graph_fraction = 0.0

        # There is no maintenance_work_mem equivalent either. WiredTiger has no
        # such knob, and mongot's index build is heap work, already accounted
        # above. Reserving a maintenance share would take memory from the page
        # cache and, worse, size /dev/shm from it: that term exists for
        # pgvector's parallel HNSW build, which has no counterpart here.
        maint_fraction = 0.0

        # Heap scales with the number of indexed fields, not with the number of
        # vectors. This index has one vector field and one filter field, so a
        # large heap buys nothing and costs page cache.
        heap_gb = float(mem_cfg.get("mongot_heap_gb", 8) or 8)
        mongot_heap_bytes = int(heap_gb * GB)

        # Past roughly 30 GB the JVM drops compressed object pointers and every
        # reference doubles in width. MongoDB's advice is to stay below the
        # boundary or jump straight to 48 GB; nothing here needs 48.
        compressed_oops_ceiling = int(30 * GB)
        if mongot_heap_bytes > compressed_oops_ceiling:
            warnings.append(
                f"mongot heap of {heap_gb:.0f} GB is above the ~30 GB "
                f"compressed object pointer boundary, where every reference "
                f"doubles in width; capping at 30 GB. Heap scales with indexed "
                f"field count, not vector count, so this is very unlikely to "
                f"be the limit that matters."
            )
            mongot_heap_bytes = compressed_oops_ceiling

        # And never more than half the container, or there is nothing left for
        # the segments themselves.
        half = server_bytes // 2
        if mongot_heap_bytes > half:
            warnings.append(
                f"mongot heap of {mongot_heap_bytes / GB:.1f} GB is over half "
                f"the {server_bytes / GB:.1f} GB container budget, leaving "
                f"insufficient filesystem cache for the memory-mapped index "
                f"segments that serve queries; capping at {half / GB:.1f} GB"
            )
            mongot_heap_bytes = half

    buffer_bytes = int(server_bytes * buffer_fraction)
    graph_cache_bytes = int(server_bytes * graph_fraction)
    maintenance_bytes = int(server_bytes * maint_fraction)

    # mongod's WiredTiger cache IS this engine's buffer pool, so it is given
    # the buffer share every other engine's main cache gets, rather than a
    # number of its own. Until ops measurement version 9 the framework never
    # set it: `config/engines/mongodb.yml` declared a
    # `wiredtiger_cache_fraction: 0.6` that nothing read, and on
    # `full-20260825` MongoDB ran on the server's own default (57-64 GB, from
    # the server log) beside a 42.3 GB InnoDB buffer pool -- the one engine
    # whose main cache was sized by its vendor's heuristic rather than by the
    # resource pass. The flag takes GB; `server_args` renders this in GB.
    wiredtiger_cache_bytes = buffer_bytes if engine == "mongodb" else 0

    total_allocated = (buffer_bytes + graph_cache_bytes + maintenance_bytes
                       + mongot_heap_bytes)
    if total_allocated > server_bytes * 0.9:
        warnings.append(
            "buffer + graph cache + maintenance exceeds 90% of the container "
            "memory limit, leaving little for connections and sorting; "
            "the engine may be OOM-killed"
        )

    build_threads = int(build_cfg.get("threads", 1) or 0)
    if build_threads <= 0:
        build_threads = len(server_cpus)

    # Hard cap. "threads: 0" means "match the cpuset", which on a 40-core server
    # resolved to 76 parallel maintenance workers -- absurd for an index build,
    # and enough to exhaust the container's /dev/shm with dynamic shared memory
    # segments, which is exactly how pgvector's tuned pass died. Parallel index
    # build stops paying past a handful of workers regardless.
    max_threads = int(build_cfg.get("max_threads", 8) or 8)
    if build_threads > max_threads:
        warnings.append(
            f"build threads clamped from {build_threads} to {max_threads}; "
            f"parallel index build does not benefit beyond a handful of workers "
            f"and each one consumes shared memory (raise build.max_threads to override)"
        )
        build_threads = max_threads

    # Dynamic shared memory for parallel workers lives in /dev/shm. Size it from
    # the parallelism actually granted rather than trusting a fixed value that
    # silently becomes too small when build.threads is raised.
    shm_size = str(docker_cfg.get("shm_size", "2g"))
    try:
        shm_gb = float(shm_size.rstrip("gG"))

        # Two independent demands on /dev/shm, and the larger one wins.
        #
        # Per-worker segments scale with the worker count. That was the only
        # term here originally, and it is the smaller of the two by a wide
        # margin once maintenance_work_mem is generous.
        per_worker_gb = 1.0 + 0.5 * build_threads

        # pgvector's parallel HNSW build puts the graph in a dynamic shared
        # memory segment sized from maintenance_work_mem, so /dev/shm has to
        # hold the whole thing. With maintenance_work_mem at 11.25 GB and
        # shm at 8g the build died with
        #   could not resize shared memory segment ... to 12078927552 bytes:
        #   No space left on device
        # after 27 minutes. 12078927552 is exactly maintenance_work_mem, which
        # is what identifies the term that was missing.
        build_segment_gb = (maintenance_bytes / GB) * 1.1 + 1.0

        needed_gb = max(per_worker_gb, build_segment_gb)

        # /dev/shm is tmpfs and counts against the container's memory limit, so
        # it cannot be sized past the budget it is being carved out of.
        ceiling_gb = (server_bytes / GB) * 0.5
        if needed_gb > ceiling_gb:
            warnings.append(
                f"a parallel index build wants {needed_gb:.0f}g of /dev/shm but "
                f"that is over half the {server_bytes / GB:.0f}g container "
                f"budget; capping at {ceiling_gb:.0f}g. Lower "
                f"memory.maintenance_fraction or build.max_threads if the build "
                f"fails with 'No space left on device'."
            )
            needed_gb = ceiling_gb

        if shm_gb < needed_gb:
            warnings.append(
                f"shm_size raised from {shm_size} to {needed_gb:.0f}g: a "
                f"parallel index build needs /dev/shm to hold a segment the "
                f"size of maintenance_work_mem "
                f"({maintenance_bytes / GB:.1f}g), plus room for "
                f"{build_threads} worker(s)"
            )
            shm_size = f"{needed_gb:.0f}g"
    except ValueError:
        pass

    return ResolvedResources(
        name=resources.get("name", "unnamed"),
        server_cpuset=format_cpuset(server_cpus),
        client_cpuset=format_cpuset(client_cpus),
        server_cpu_count=len(server_cpus),
        server_memory_bytes=server_bytes,
        client_memory_bytes=client_bytes,
        buffer_bytes=buffer_bytes,
        graph_cache_bytes=graph_cache_bytes,
        maintenance_bytes=maintenance_bytes,
        mongot_heap_bytes=mongot_heap_bytes,
        maxmemory_bytes=maxmemory_bytes,
        wiredtiger_cache_bytes=wiredtiger_cache_bytes,
        io_threads=io_threads,
        pass_signature=pass_signature,
        build_threads=build_threads,
        shm_size=shm_size,
        transaction_isolation=str(iso_cfg.get("transaction_isolation", "engine_default")),
        hybrid_cpu=bool(sysinfo.cpu.hybrid),
        core_class_used=prefer if sysinfo.cpu.hybrid else "uniform",
        warnings=warnings,
    )


#: Placeholders an engine's server args may use. Anything else is rendered
#: literally and reaches the server as a nonsense value.
SERVER_ARG_KEYS = (
    "buffer_bytes", "graph_cache_bytes", "maintenance_bytes", "build_threads",
    "server_cpu_count", "maxmemory_bytes", "mongot_heap_bytes",
    "server_memory_bytes", "io_threads", "wiredtiger_cache_gb",
)


def wiredtiger_cache_gb(resolved: "ResolvedResources") -> str:
    """`--wiredTigerCacheSizeGB`'s value: GB with two decimals.

    The flag takes a float in GB (mongod's floor is 0.25), and the resolved
    figure is bytes like every other one here. Two decimals is 10 MB of
    resolution against a cache measured in tens of GB, so nothing is lost
    that a reader of `engines_resolved` would want back.
    """
    # `getattr`: several callers hand `server_args` a hand-built stand-in for
    # `ResolvedResources` that predates the field, and only mongodb renders it.
    cache = getattr(resolved, "wiredtiger_cache_bytes", 0) or 0
    return f"{max(0.25, cache / GB):.2f}"


#: The profile key that turns the MySQL-family wait instruments on.
WAIT_INSTRUMENTS_KEY = "server_wait_instruments"


def ops_wait_instruments(profile: Dict[str, Any]) -> bool:
    """`ops.server_wait_instruments`, default False, refused unless boolean.

    **Default off because turning it on changes the server being measured.**
    performance_schema's mutex and rwlock instruments put a timer read on
    every latch acquisition, which is the hot path of exactly the
    concurrency windows they are meant to explain, and on MariaDB they also
    switch performance_schema itself on (its default is OFF) and with it the
    memory that allocates. Measure the overhead in a `smoke` run -- the same
    unit with the key off and on -- before turning it on for a profile whose
    numbers are published; while it is on, `engines_resolved` and every
    record say so (`extra.server_wait_instruments`).

    Refused rather than coerced: YAML's `"false"` is a non-empty string and
    would be truthy, which is a server flag nobody asked for.
    """
    value = (profile.get("ops") or {}).get(WAIT_INSTRUMENTS_KEY, False)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"ops.{WAIT_INSTRUMENTS_KEY} must be true or false, "
                         f"not {value!r}")
    return value


#: The profile key naming the concurrency client count to profile with perf.
PROFILE_WINDOW_KEY = "profile_window_clients"


def profile_window_clients(profile: Dict[str, Any]) -> Optional[int]:
    """`ops.profile_window_clients`: None (off, the default) or a client count
    the profile actually measures.

    Refused when it names a count not in `ops.client_counts`: the harness
    would never reach that window, perf would never start, and the run
    would end with no profile and nothing saying why.
    """
    ops = profile.get("ops") or {}
    value = ops.get(PROFILE_WINDOW_KEY)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"ops.{PROFILE_WINDOW_KEY} must be a client count "
                         f"or null, not {value!r}")
    counts = [int(c) for c in ops.get("client_counts", [1])]
    if value not in counts:
        raise ValueError(f"ops.{PROFILE_WINDOW_KEY} is {value}, which is not "
                         f"one of ops.client_counts {counts}")
    return value


#: The profile key that lets a sampling engine sample during the window.
WAIT_SAMPLER_KEY = "server_wait_sampler"


def ops_wait_sampler(profile: Dict[str, Any]) -> bool:
    """`ops.server_wait_sampler`, default False, refused unless boolean.

    **Default off because it is the one wait collector that runs INSIDE the
    measured window.** Every other engine's wait profile is two snapshots at
    the window's edges; pgvector has no cumulative wait-event counters, so its
    only profile is `pg_stat_activity` sampled once a second while the clients
    run -- an extra backend and a query whose cost grows with the client
    count. On by default it would have measured pgvector under load no other
    engine's window carries. When on, `engines_resolved` and every pgvector
    record say so (`server_wait_sampler: {interval_s, concurrent: true}`).
    """
    value = (profile.get("ops") or {}).get(WAIT_SAMPLER_KEY, False)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"ops.{WAIT_SAMPLER_KEY} must be true or false, "
                         f"not {value!r}")
    return value


def engine_wait_sampler_interval(engine_cfg: Dict[str, Any]
                                 ) -> Optional[float]:
    """`capabilities.wait_sampler_interval_s`, or None for a snapshot engine."""
    caps = engine_cfg.get("capabilities") or {}
    value = caps.get("wait_sampler_interval_s")
    return float(value) if value else None


def engine_wait_instruments(engine_cfg: Dict[str, Any]) -> List[str]:
    """The flags an engine declares under `server.wait_instruments`."""
    server = engine_cfg.get("server", {}) or {}
    return [str(a) for a in (server.get("wait_instruments") or [])]


def server_args(engine_cfg: Dict[str, Any], resource_pass: str,
                resolved: ResolvedResources,
                wait_instruments: bool = False) -> List[str]:
    """Render an engine's server flags with the resolved values substituted.

    `wait_instruments` appends the engine's `server.wait_instruments` flags
    (`ops_wait_instruments`). Only the ops path passes it: the waits are read
    around ops windows, and the ann results tree is shared across runs and
    keyed without it, so an instrumented recall curve would be reused later
    under a manifest that says the server was not instrumented.
    """
    server = engine_cfg.get("server", {}) or {}
    args: List[str] = list(server.get("common", []) or [])
    args += list(server.get(resource_pass, []) or [])
    if wait_instruments:
        args += engine_wait_instruments(engine_cfg)

    substitutions = {
        "buffer_bytes": resolved.buffer_bytes,
        "graph_cache_bytes": resolved.graph_cache_bytes,
        "maintenance_bytes": resolved.maintenance_bytes,
        "build_threads": resolved.build_threads,
        "server_cpu_count": resolved.server_cpu_count,
        # Added with the in-memory and two-process engines. A placeholder with
        # no entry here is passed through literally, and the server rejects it
        # and exits in under a second, which is how every valkey phase of the
        # first smoke run failed. See the test that scans for unknown keys.
        "maxmemory_bytes": resolved.maxmemory_bytes,
        "io_threads": resolved.io_threads,
        "mongot_heap_bytes": resolved.mongot_heap_bytes,
        "server_memory_bytes": resolved.server_memory_bytes,
        "wiredtiger_cache_gb": wiredtiger_cache_gb(resolved),
    }

    rendered: List[str] = []
    for arg in args:
        text = str(arg)
        for key, value in substitutions.items():
            text = text.replace("{" + key + "}", str(value))
        # A graph cache of zero is meaningless; drop the flag entirely rather
        # than passing 0 and having the engine reject it or disable caching.
        if resolved.graph_cache_bytes == 0 and "cache-size=0" in text.replace("_", "-"):
            continue
        # An empty argument cannot survive the trip: flags are joined into
        # VB_SERVER_ARGS and word-split back apart in the entrypoint, which
        # discards it and silently shifts the meaning of the flag before it.
        # Anything needing an empty value belongs in the entrypoint, where the
        # quoting holds.
        if text == "":
            continue
        rendered.append(text)
    return rendered


def server_env(resolved: ResolvedResources) -> Dict[str, str]:
    """The size-carrying env vars every server container is started with.

    One function for both measurement paths and for the manifest, so what
    `engines_resolved` records is the env the container got rather than a
    second spelling of it. `ann_pass.run_engine` and `ops_pass._start_server`
    used to build these two entries inline, each with its own copy of the
    `max(1, ...)` floor; a third copy here, for the manifest, would be the
    one nobody noticed drifting.
    """
    return {
        # Sized by resolve_resources and passed through, so a tuned run
        # cannot silently fall back to the image default. Harmless for the
        # single-process engines, which never read it.
        "VB_MONGOT_HEAP_GB": str(max(1, resolved.mongot_heap_bytes // GB)),
        "VB_MAXMEMORY_BYTES": str(resolved.maxmemory_bytes),
    }


def engine_resolution(engine_cfg: Dict[str, Any], resource_pass: str,
                      resolved: ResolvedResources,
                      wait_instruments: bool = False,
                      wait_sampler: bool = False) -> Dict[str, Any]:
    """What one engine was given under one pass, for `engines_resolved`.

    WHY THIS IS PER ENGINE. `full-20260825` published "all six got the same
    budget", and the budgets were not the same: InnoDB got a 42.3 GB buffer
    pool, pgvector 72.5 GB of shared_buffers, Valkey 108.7 GB of maxmemory,
    and MongoDB ran on its default WiredTiger cache (57-64 GB). That is
    `resolve_resources` working as designed -- pgvector's buffer absorbs the
    graph cache's share, an in-memory engine gets maxmemory instead of a
    buffer -- but the manifest held one `config.resolved_resources`, written by
    whichever engine `set_config` saw LAST. Snapshot 1 of that run, written
    after mongodb, shows graph_cache 0 and shm 8g; snapshot 9, written after an
    AliSQL rerun, shows 30.2 GB and 13g. The per-engine figures could only be
    recovered from server logs.

    Built from the same `engine_cfg` and `resolved` objects `cmd_run` hands to
    `_run_unit`, through the same `server_args` and `server_env` the two
    container launchers call, so the record and the container cannot disagree
    without one of those functions changing for both.

    `mem_limit_bytes` is the SERVER's limit. The ann container adds the
    client's in-RAM corpus on top (`ann_pass.client_memory_bytes`), because
    that is the benchmark's memory and not the engine's; it is per dataset,
    and this record is per (engine, pass).
    """
    # Which resolved figures a server flag actually carries, read off the
    # templates rather than inferred from the rendered values: MyVector's
    # graph cache and buffer resolve to the same byte count, so "is the
    # number in the args" said MyVector had a graph cache it has no flag for.
    server = engine_cfg.get("server", {}) or {}
    templates = " ".join(str(a) for a in list(server.get("common", []) or [])
                         + list(server.get(resource_pass, []) or []))
    applied = [key for key in SERVER_ARG_KEYS if "{" + key + "}" in templates]
    if resolved.graph_cache_bytes == 0 and "graph_cache_bytes" in applied:
        # `server_args` drops a zero cache-size flag; so does this record.
        applied.remove("graph_cache_bytes")
    entry = {
        "resolved": resolved.as_dict(),
        "server_args": server_args(engine_cfg, resource_pass, resolved),
        "applied": applied,
        "container": {
            "cpuset": resolved.server_cpuset,
            "mem_limit_bytes": resolved.server_memory_bytes,
            "shm_size": resolved.shm_size,
            "env": server_env(resolved),
        },
    }
    # Only when ON and only for an engine that declares instruments, so a
    # default run's entry is byte-identical to one written before the key
    # existed -- `record_engine_resolution` refuses a resume whose entry
    # differs, and an always-present `false` would refuse every resume of
    # every run recorded before it.
    flags = engine_wait_instruments(engine_cfg) if wait_instruments else []
    if flags:
        entry["applied"] = applied + [WAIT_INSTRUMENTS_KEY]
        entry[WAIT_INSTRUMENTS_KEY] = {
            "enabled": True,
            "flags": flags,
            # `server_args` above is what the ann path starts; the ops server
            # gets these appended (see `server_args`).
            "applies_to": ["ops"],
        }
    # Same rule for the sampler: present only when on, and only for an engine
    # that samples, so every default entry stays byte-identical.
    interval = engine_wait_sampler_interval(engine_cfg) if wait_sampler \
        else None
    if interval:
        entry["applied"] = entry["applied"] + [WAIT_SAMPLER_KEY]
        entry[WAIT_SAMPLER_KEY] = {"interval_s": interval, "concurrent": True,
                                   "applies_to": ["ops"]}
    return entry
