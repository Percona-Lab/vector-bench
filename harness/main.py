"""Ops-harness entrypoint. Runs inside the client container.

The orchestrator has already started the engine's server container with the
right resource limits and server flags; this process only connects to it, runs
the requested workloads, and appends records.

Deliberately does NOT start or configure any server: keeping that in the
orchestrator is what guarantees every engine gets identical treatment and that
the treatment is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow running as `python3 /opt/vb-harness/main.py` as well as `-m harness.main`.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "harness"

from .datasets import load as load_dataset  # noqa: E402
from .drivers.base import ConnectionSpec, IndexSpec  # noqa: E402
from .drivers.postgres import get_driver  # noqa: E402
from .metrics.records import Recorder  # noqa: E402
from .workloads import build as build_workload  # noqa: E402
from .workloads import calibrate as calibrate_workload  # noqa: E402
from .workloads import churn as churn_workload  # noqa: E402
from .workloads import concurrency as concurrency_workload  # noqa: E402
from .workloads import filtered as filtered_workload  # noqa: E402
from .workloads import graph_warmup  # noqa: E402
from .workloads.context import RunContext  # noqa: E402
from .workloads.server_waits import sampler_disclosure  # noqa: E402

ALL_WORKLOADS = ("build", "concurrency", "filtered", "churn")


def _should_calibrate(ef_mode: str, workloads: Sequence[str]) -> bool:
    """Whether to sweep for an ef_search before measuring.

    Two conditions, and the second is the one worth stating: the sweep exists
    to choose the ef that concurrency, filtered and churn are measured at, so a
    run that measures none of them would pay up to twelve grid points of
    queries and then discard the answer. `--workloads build` is a real
    invocation: this parser accepts any subset of ALL_WORKLOADS, and a run that
    only wants ingest and index-build cost asks for exactly that one.

    A narrowing of PR 2 §2.2.2, which says to calibrate whenever the mode is
    calibrated. No profile in the repo selects a build-only ops phase today, so
    nothing changes for any configured run; what it buys is that the cheapest
    possible ops invocation does not silently acquire the most expensive new
    cost in this change.
    """
    return (ef_mode == calibrate_workload.EF_MODE_CALIBRATED
            and any(w in workloads
                    for w in calibrate_workload.CALIBRATED_WORKLOADS))


def _apply_declined(driver: Any, workloads: Sequence[str]
                    ) -> Tuple[List[str], Dict[str, str]]:
    """The workloads to run, and the ones this driver declines with why.

    `EngineDriver.declined_workloads` is a declaration on the driver class --
    MyVector's pinned release has no filtered search, and answering filtered
    queries by post-filtering the k nearest would publish a number under a
    label it does not deserve. Applied here, before the first record, so the
    decline is stamped onto every record of the run
    (`extra.declined_workloads`) and the measurement workloads that CAN run
    still do. A driver that declares nothing gets its workload list back
    unchanged and an empty mapping, which writes no key at all.
    """
    declared = dict(getattr(driver, "declined_workloads", ()) or ())
    declined = {w: declared[w] for w in workloads if w in declared}
    return [w for w in workloads if w not in declined], declined


def _query_path_facts(driver: Any, index: IndexSpec) -> Dict[str, Any]:
    """`EngineDriver.query_path_facts`, or `{}` for a driver without it.

    Through `getattr` because test fakes and lab scripts build drivers that
    are not `EngineDriver` subclasses; an engine that declares nothing writes
    no key, so every other engine's records are unchanged.
    """
    hook = getattr(driver, "query_path_facts", None)
    return dict(hook(index) or {}) if hook is not None else {}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vb-harness",
        description="vector-bench ops harness (build cost, concurrency, filtered, churn)",
    )
    # Not constrained to a fixed list: the orchestrator validates the name
    # against config/engines/*.yml before it starts anything, and an engine
    # added there must not need this file edited too.
    p.add_argument("--engine", required=True)
    p.add_argument("--driver", default=None,
                   help="driver class to use; the orchestrator reads it from "
                        "the engine config. Defaults to the engine's own name.")
    p.add_argument("--host", required=True, help="server container hostname")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--user", default=os.environ.get("VB_DB_USER", "bench"))
    p.add_argument("--password", default=os.environ.get("VB_DB_PASSWORD", "bench"))
    p.add_argument("--database", default="ann")
    p.add_argument(
        "--server-data-dir", default=None,
        help="path where this process can read the server's data directory "
             "read-only; enables exact on-disk index sizing",
    )

    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-dir", default="/datasets")
    p.add_argument("--subset-rows", type=int, default=None,
                   help="load only the first N training vectors (smoke profiles)")
    p.add_argument("--max-queries", type=int, default=1000)

    p.add_argument("--m", type=int, required=True)
    p.add_argument("--ef-construction", type=int, default=None)
    p.add_argument("--ef-search", type=int, default=100)
    # The ef policy. `calibrated` is the default because a shared literal
    # ef_search is not a shared amount of work: the knob is mhnsw_ef_search,
    # VIDX's equivalent, hnsw.ef_search, numCandidates and EF_RUNTIME, and two
    # of these engines sit at recall 0.98 and 0.80 at 100. `fixed` keeps
    # --ef-search verbatim, which is what every run before this did.
    p.add_argument("--ef-mode", default=calibrate_workload.EF_MODE_CALIBRATED,
                   choices=calibrate_workload.EF_MODES,
                   help="calibrated (default): sweep --ef-grid and measure at "
                        "the smallest ef reaching --target-recall; fixed: "
                        "measure at --ef-search")
    p.add_argument("--target-recall", type=float,
                   default=calibrate_workload.DEFAULT_TARGET_RECALL,
                   help="recall@k the calibration sweep aims for")
    p.add_argument("--ef-grid",
                   default=",".join(str(ef) for ef in
                                    calibrate_workload.DEFAULT_EF_GRID),
                   help="comma-separated ef_search values the calibration "
                        "sweep measures, in any order")
    p.add_argument("--k", type=int, default=10)
    # The MySQL family's `ENGINE=` clause, and nothing else. The orchestrator
    # emits it unconditionally -- `_ops_storage_engines` returns `["InnoDB"]`
    # for every engine that is not MariaDB -- so the default stays `InnoDB`
    # for the DDL that needs a value. It does NOT reach the record any more:
    # `EngineDriver.storage_engine_in_use` decides what a record claims, so an
    # engine with no storage engine claims none. See that docstring for the
    # 516 records of InnoDB this flag used to put on PostgreSQL, Valkey and
    # Percona Search.
    p.add_argument("--storage-engine", default="InnoDB",
                   help="the MySQL family's ENGINE= clause (InnoDB, MyISAM); "
                        "ignored by every other engine, which records no "
                        "storage engine at all")
    p.add_argument("--build-mode", default="post", choices=("post", "incremental"))
    # The engine's BULK vector index build, as its own config declares it.
    # Decided on the host and passed in, the way `--driver` above is, and for
    # the same reason: `harness/` must not import `orchestrator`, so it cannot
    # read `capabilities.bulk_vector_index_build` out of
    # `config/engines/<e>.yml` itself. `orchestrator/cli.py` reads the
    # capability and `ops_pass.harness_args` emits this flag for the engines
    # that declare it broken.
    #
    # NEGATIVE POLARITY, and `store_true`, so the absence of the flag is
    # today's behaviour exactly -- which is also how the capability reads when
    # an engine config omits it (`cli.py:_bulk_vector_index_build`). One fact,
    # one default, on both sides of the container boundary. A positive flag
    # that had to be spelled out for all seven engines would be a second
    # place for that default to live, and the two would drift.
    #
    # **It does NOT gate `--build-mode post`, and no longer gates the churn
    # repair either.** An explicit `post` is deliberately honoured with a
    # warning so a newer server can be tested for the fix without editing the
    # engine config (`cli.py:_ops_build_modes`); the only driver that reads
    # this flag now is `VillageSQLDriver`, to name the known crash in the
    # phase log before it runs a bulk build a profile asked for. The repair
    # withdrawal it used to carry is `--no-vector-index-replace` below.
    p.add_argument("--no-bulk-vector-index-build", action="store_true",
                   help="the engine's bulk vector index build is unusable at "
                        "its pinned version; a driver that knows the symptom "
                        "names it before running an explicitly requested "
                        "post build")
    # The churn REPAIR's withdrawal, as `cli.py:_vector_index_replace` reads
    # it -- `capabilities.vector_index_replace`, falling back to the bulk-build
    # key when an engine declares nothing separate. Same negative polarity and
    # `store_true` as the flag above, for the same reason: absent means
    # today's behaviour, which is a driver running whatever repair it has.
    #
    # **A SEPARATE FLAG BECAUSE IT IS A SEPARATE STATEMENT.** `maintain()`'s
    # `ALTER TABLE ... DROP KEY vi, ADD VECTOR KEY ..., ALGORITHM=INPLACE`
    # replaces an EXISTING index; `--build-mode post` adds one to a table
    # that has none. At Percona's old pin the first returned in 1-4 ms as a
    # no-op the report published as a repair, while the second killed the
    # server; at the current pin the second builds and the first has not been
    # re-tested. Riding on one flag would have restored the unverified one
    # the moment the verified one was enabled. See
    # `PerconaDriver.declare_vector_index_replace`.
    p.add_argument("--no-vector-index-replace", action="store_true",
                   help="replacing the vector index on an indexed table is "
                        "not a verified repair at the engine's pinned "
                        "version; a driver whose repair path is that replace "
                        "declares no repair path instead of timing it")
    # Percona Search is the only engine that compresses indexed vectors. Without
    # this the ops path always built unquantized while the recall path built
    # whatever the profile asked for, so one run measured two different indexes.
    p.add_argument("--quantization", default=None,
                   choices=(None, "none", "scalar", "binary"))
    p.add_argument("--load-threads", type=int, default=1)
    # How the orchestrator chose the build this run measures, when the profile
    # asked it to choose (`ops.build_selection: frontier`): JSON written on the
    # host, stamped verbatim on every record as `extra.ops_build_selection`.
    # The build itself is already on every record (m, ef_construction,
    # quantization); this says WHY it is that build and whether it reached the
    # target -- a `fallback_reference` or `below_target` build measured
    # without that note would read as the frontier's choice.
    p.add_argument("--ops-build-selection", default=None, type=_json_object)

    p.add_argument("--workloads", default=",".join(ALL_WORKLOADS),
                   help=f"comma-separated subset of: {','.join(ALL_WORKLOADS)}")
    p.add_argument("--client-counts", default="1,2,4,8,16,32")
    p.add_argument("--concurrency-duration", type=float, default=20.0)
    # Deliberately NOT --max-queries. That key exists to cap the cost of the
    # filtered and churn workloads, which run one query at a time to
    # completion, and main.yml sets it to 500 for exactly that reason. The
    # concurrency workload cycles its shard for a fixed duration instead, so
    # inheriting the cap gave 32 clients about fifteen queries each for thirty
    # seconds and showed every graph cache in the comparison a hot set small
    # enough to hold entirely. 0 means the whole test set.
    p.add_argument("--concurrency-queries", type=int,
                   default=concurrency_workload.WHOLE_TEST_SET,
                   help="queries the concurrency clients draw from; "
                        "0 (the default) means the whole test set")
    # Every figure here was a single sample until a Valkey point was
    # re-measured by accident and came back 21% higher. Concurrency is the
    # headline finding, the noisiest measurement, and the cheapest to repeat.
    p.add_argument("--concurrency-repeats", type=int, default=1,
                   help="measure each client count N times and record the "
                        "median window with the spread beside it")
    p.add_argument("--selectivities", default="0.01,0.10,0.50")
    # Both filter columns by default, because `build` loads both: measuring
    # one would leave a column in the table that nothing ever queries, and
    # choosing between them per run would cost a full reload -- hours for
    # MHNSW and VIDX at 1M rows.
    p.add_argument("--filter-columns",
                   default=",".join(filtered_workload.DEFAULT_COLUMNS),
                   help="comma-separated filter columns the filtered "
                        f"workload measures; any of "
                        f"{','.join(filtered_workload.DEFAULT_COLUMNS)}")
    # The same reason --churn-budget exists, for the workload that turned out
    # to need it more. Filtered search answers one query at a time and a
    # selective predicate can take an engine from 326 QPS to 0.9, so the
    # profile's query count is a cost nothing bounded: 500 queries at 0.9 QPS
    # is 9 minutes, and it was 37 minutes at the 2,000 that `full` asks for --
    # per point, per column, per ops unit.
    p.add_argument("--filtered-budget", type=float,
                   default=filtered_workload.DEFAULT_BUDGET_S,
                   help="seconds per (column, selectivity, iterative-scan "
                        "mode) point in the filtered workload")
    p.add_argument("--churn-fractions", default="0.10,0.25")
    # An engine that stalls re-inserting must release the machine and record
    # what it managed, rather than holding the run until someone notices.
    p.add_argument("--churn-budget", type=float,
                   default=churn_workload.DEFAULT_CHURN_BUDGET_S)
    # PR 4 §4.3 item 2. A churn number measured with nothing else touching the
    # index describes a maintenance window, not a database under load, and the
    # two can differ by more than the engines differ from each other. 0 turns
    # the pool off, which is what `ops.churn_readers: 0` is for: the
    # write-only re-insert rate the report showed before this existed is still
    # a quantity someone may want, and it is not the same quantity as a rate
    # measured against four concurrent readers.
    p.add_argument("--churn-readers", type=int,
                   default=churn_workload.DEFAULT_CHURN_READERS,
                   help="readers querying the index while the churn writer "
                        "re-inserts; 0 disables the pool and restores the "
                        "write-only re-insert rate")
    # Comma-separated, and the workload loops whatever it is given: PR 3 §3.4
    # item 4 measures the filtered grid once per declared mode, against ONE
    # loaded index. Looping in the orchestrator instead would re-run build,
    # concurrency and churn per mode and double pgvector's ingest to vary a
    # query-time setting.
    p.add_argument("--iterative-scan", default=None,
                   help="pgvector only: one or more of off | relaxed_order | "
                        "strict_order, comma-separated")

    # Server-side evidence of what limited throughput. Both decided on the
    # host from the profile (`ops.server_wait_instruments`,
    # `ops.profile_window_clients`); the harness only records the first and
    # signals the window for the second.
    p.add_argument("--server-wait-instruments", action="store_true",
                   help="the server was started with performance_schema "
                        "mutex/rwlock wait instruments ON; stamped on every "
                        "record because it changes server behaviour")
    p.add_argument("--server-wait-sampler", action="store_true",
                   help="let a sampling engine (pgvector) sample its wait "
                        "events DURING each window; off by default because "
                        "it is load the other engines' windows do not carry")
    p.add_argument("--profile-window-clients", type=int, default=None,
                   help="client count whose first measured concurrency "
                        "window the orchestrator profiles with perf")
    p.add_argument("--profile-ack-dir", default=None,
                   help="directory the orchestrator acknowledges a profiled "
                        "window's start in")
    p.add_argument("--run-id", required=True)
    p.add_argument("--resource-pass", default="normalized",
                   choices=("normalized", "tuned"))
    p.add_argument("--engine-tag", default=os.environ.get("VB_ENGINE_TAG"))
    p.add_argument("--march", default=None)
    p.add_argument("--output", required=True, help="JSONL file to append records to")
    p.add_argument("--cache-dir", default="/results/.cache")
    return p.parse_args(argv)


def _json_object(text: str) -> Dict[str, Any]:
    """An argparse type: a JSON object, refused loudly if it is anything else."""
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def _floats(csv: str) -> List[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def _ints(csv: str) -> List[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def _strings(csv: str) -> List[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _detect_march() -> str:
    """Which -march the server image was compiled with.

    One of two independent readers, in two different containers. This one runs
    in the HARNESS container, which mounts `harness/` and not `overlay/`, so no
    override in an ann-benchmarks module can reach it -- and
    `orchestrator/ops_pass.py` passes no `--march`, so `args.march` is empty
    and this is what runs for every ops record.

    A missing path here does not crash: it records `march: "unknown"` in a
    field a reader takes for a measurement. That matters more than it looks,
    because "never mix -march between engines" is a stated invariant -- the
    five engines built from source compile SIMD distance kernels, and the
    other two are installed from packages -- and `march` is the field that
    would catch a mixed build. An engine added to the image set and not to
    this tuple loses exactly the check that would have caught it:
    `/opt/percona/.march` was absent from both readers when Percona's image
    was written, and nothing anywhere would have said so.
    """
    for path in ("/opt/mariadb/.march", "/opt/alisql/.march",
                 "/opt/pgvector-artifacts/.march",
                 "/opt/percona/.march", "/opt/myvector/.march",
                 "/opt/villagesql/.march"):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            continue
    return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    unknown = set(workloads) - set(ALL_WORKLOADS)
    if unknown:
        print(f"unknown workloads: {sorted(unknown)}", file=sys.stderr)
        return 2

    spec = ConnectionSpec(
        host=args.host, port=args.port, user=args.user, password=args.password,
        database=args.database, data_dir=args.server_data_dir,
    )
    index = IndexSpec(
        dim=0,  # filled from the dataset below
        m=args.m,
        metric="",  # filled from the dataset below
        ef_construction=args.ef_construction,
        storage_engine=args.storage_engine,
        build_mode=args.build_mode,
        quantization=args.quantization,
    )

    print(f"[vb-harness] loading dataset {args.dataset} from {args.datasets_dir}")
    dataset = load_dataset(args.datasets_dir, args.dataset)
    index.dim = dataset.dim
    index.metric = dataset.metric
    print(
        f"[vb-harness] {dataset.name}: {dataset.n_train:,} train x {dataset.dim} dims, "
        f"{dataset.n_test:,} test, metric={dataset.metric}"
    )

    def driver_factory():
        driver = get_driver(args.engine, spec, args.driver)
        # The two host-side capabilities applied AFTER construction rather
        # than handed to `get_driver`: that function takes the connection spec and
        # the driver class name, and widening its signature is a change to
        # `harness/drivers/postgres.py`, which every engine's driver is built
        # through. Read through `getattr`, the way the churn workload reads
        # Valkey's optional sync hooks -- a driver with nothing to withdraw
        # has no such method, and most do not. Today `VillageSQLDriver` alone
        # has the first and `PerconaDriver` alone the second.
        #
        # No runtime warning for the "flag given, nobody listened" case, on
        # purpose: that would fire once per client connection (up to 32 in the
        # concurrency pool) and it is a WIRING fault, which belongs in the
        # suite rather than in a run's log.
        # `tests/test_e13_percona_churn_repair_is_a_noop.py` pins that every
        # engine whose host reading withdraws the replace and whose driver
        # class HAS a repair path can act on it, so a declaration cannot
        # silently reach a driver that would ignore it.
        declare = getattr(driver, "declare_bulk_vector_index_build", None)
        if declare is not None:
            declare(not args.no_bulk_vector_index_build)
        withdraw = getattr(driver, "declare_vector_index_replace", None)
        if withdraw is not None:
            withdraw(not args.no_vector_index_replace)
        return driver

    # The mode is known now and is true of every record this run writes; the
    # target and the status are not known until the sweep has run, and are
    # declared afterwards. See `Recorder.declare_run_extra`.
    recorder = Recorder(args.output, run_extra={"ef_mode": args.ef_mode})
    if args.ops_build_selection:
        recorder.declare_run_extra(ops_build_selection=args.ops_build_selection)
    if args.server_wait_instruments:
        # On every record, not only on the ones that read waits: the
        # instruments are a server setting, and a build or filtered figure
        # measured with them on was measured on a different server. Absent
        # (not False) when off, so a default run's records are unchanged.
        recorder.declare_run_extra(server_wait_instruments=True)
    ctx = RunContext(
        run_id=args.run_id,
        resource_pass=args.resource_pass,
        recorder=recorder,
        engine=args.engine,
        engine_tag=args.engine_tag,
        march=args.march or _detect_march(),
        k=args.k,
        cache_dir=args.cache_dir,
    )

    driver = driver_factory()
    workloads, declined = _apply_declined(driver, workloads)
    for workload, reason in declined.items():
        print(f"[vb-harness] {workload}: DECLINED by the {args.engine} "
              f"driver, not run. {reason}", file=sys.stderr)
    if declined:
        recorder.declare_run_extra(declined_workloads=declined)
    # Stamped on every record when the sampler ran, because then every
    # window of this engine carried a sampling backend the other engines'
    # windows did not. Absent otherwise, so a default record is unchanged.
    sampled = sampler_disclosure(driver, args.server_wait_sampler)
    if sampled is not None:
        recorder.declare_run_extra(server_wait_sampler=sampled)
    facts = _query_path_facts(driver, index)
    if facts:
        recorder.declare_run_extra(**facts)
    failures: List[str] = []
    try:
        driver.connect()
        print(f"[vb-harness] connected to {args.engine}: {driver.server_version()}")

        if "build" in workloads:
            # The warm-up plan is built for every engine and used only by a
            # driver that declares `warm_graph_after_build`; `build.run`
            # decides. Calibration's query slice (`dataset.test[:max_queries]`)
            # at the top ef any later phase can ask for -- see
            # `graph_warmup.top_ef`.
            warmed = build_workload.run(
                ctx, driver, dataset, index,
                load_threads=args.load_threads, subset_rows=args.subset_rows,
                warmup=graph_warmup.plan(
                    dataset.test, max_queries=args.max_queries,
                    ef_grid=_ints(args.ef_grid), ef_search=args.ef_search),
            )
            if warmed is not None:
                # From here on, every record -- calibration, concurrency,
                # filtered, churn -- says it was measured on a graph this
                # harness warmed, and whether the warm-up completed. Declared
                # AFTER `build.run` returns, so the ingest and index_build
                # records do not carry it: they were measured before it ran,
                # and the index_build one has the full detail instead.
                recorder.declare_run_extra(
                    **{graph_warmup.RUN_EXTRA_KEY: warmed["status"]})
            # The authoritative row count comes from the engine, not from the
            # profile: ground truth must describe what was actually indexed.
            # A profile that loads a subset (smoke uses 20,000 of 60,000) would
            # otherwise be scored against rows the engine never received.
            indexed_rows = driver.count_rows()
        else:
            # Later workloads need a populated table; without the build workload
            # the caller must have populated it in an earlier invocation.
            rows = driver.count_rows() if _table_exists(driver) else 0
            if rows == 0:
                print("[vb-harness] ERROR: no data present and 'build' was not "
                      "requested; nothing to measure", file=sys.stderr)
                return 3
            driver._index = index
            indexed_rows = rows

        # A plan check before the measurement workloads. Every one of these
        # engines can silently fall back to a full scan, which yields exact
        # results slowly and is indistinguishable from "accurate but slow"
        # unless it is checked.
        driver.set_ef_search(args.ef_search)
        if not driver.explain_uses_vector_index(dataset.test[0], args.k):
            print(
                "[vb-harness] WARNING: the vector index is not in the query plan. "
                "Results below measure a full scan, not ANN search.",
                file=sys.stderr,
            )

        # Which ef_search everything below is measured at. In fixed mode it is
        # the profile's literal, exactly as before this existed; in calibrated
        # mode the sweep replaces it with the smallest ef reaching the target,
        # so the engines downstream are compared at equal answer quality
        # instead of at equal knob setting.
        ef_search = args.ef_search
        if _should_calibrate(args.ef_mode, workloads):
            calibration = calibrate_workload.run(
                ctx, driver, dataset, index,
                ef_grid=_ints(args.ef_grid),
                target_recall=args.target_recall,
                max_queries=args.max_queries,
                indexed_rows=indexed_rows,
            )
            ef_search = calibration.ef_search
            # Stamped onto every record written from here on, including the
            # three workloads' own. A concurrency figure at a calibrated ef and
            # one at a literal are different quantities and the record has to
            # say which it is.
            recorder.declare_run_extra(
                **calibrate_workload.status_extra(args.ef_mode, calibration))

        if "concurrency" in workloads:
            concurrency_workload.run(
                ctx, driver_factory, dataset, index,
                ef_search=ef_search,
                client_counts=_ints(args.client_counts),
                duration_s=args.concurrency_duration,
                max_queries=args.concurrency_queries,
                repeats=args.concurrency_repeats,
                indexed_rows=indexed_rows,
                profile_window_clients=args.profile_window_clients,
                profile_ack_dir=args.profile_ack_dir,
                wait_sampler=args.server_wait_sampler,
            )

        # The calibrated *unfiltered* ef, not one calibrated per selectivity.
        # Re-calibrating here would hide exactly what this workload exists to
        # show: that a scalar predicate costs recall or speed, which it cannot
        # do if the ef is raised until it does not. See docs/05-methodology.md.
        if "filtered" in workloads:
            filtered_workload.run(
                ctx, driver, dataset, index,
                ef_search=ef_search,
                selectivities=_floats(args.selectivities),
                columns=_strings(args.filter_columns),
                max_queries=args.max_queries,
                iterative_scan=args.iterative_scan,
                indexed_rows=indexed_rows,
                budget_s=args.filtered_budget,
            )

        # Churn mutates the table, so it runs last: everything before it sees a
        # pristine index, and nothing after it inherits a degraded one.
        if "churn" in workloads:
            churn_workload.run(
                ctx, driver, dataset, index,
                ef_search=ef_search,
                fractions=_floats(args.churn_fractions),
                max_queries=args.max_queries,
                indexed_rows=indexed_rows,
                budget_s=args.churn_budget,
                # PR 4 §4.3 item 1. The same factory the concurrency workload
                # already gets: readers need their own connections, and the
                # writer above keeps `driver`. Churn deliberately runs last
                # and mutates the table, so nothing after it inherits what the
                # readers saw.
                driver_factory=driver_factory,
                readers=args.churn_readers,
                # Ops version 9: the re-insert writes with the load's thread
                # count, and both are recorded beside its rate.
                ingest_threads=args.load_threads,
                wait_sampler=args.server_wait_sampler,
            )

    except Exception:
        failures.append(traceback.format_exc())
        print("[vb-harness] FAILED:\n" + failures[-1], file=sys.stderr)
    finally:
        try:
            driver.close()
        except Exception:
            pass
        recorder.close()

    if failures:
        return 1
    print(f"[vb-harness] done; records appended to {args.output}")
    return 0


def _table_exists(driver) -> bool:
    try:
        driver.count_rows()
        return True
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
