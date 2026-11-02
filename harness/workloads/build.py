"""Index build cost: ingest throughput, build time, index size on disk.

This is the dimension ann-benchmarks does not measure at all, and the one where
the three engines differ most structurally:

* MariaDB MHNSW and AliSQL VIDX maintain the graph on every INSERT. There is no
  separable build step — load cost and build cost are the same number.
* pgvector builds the graph as one bulk operation after the load, which is
  substantially cheaper per vector.

Reporting a single "build time" for all three would compare different
operations. This workload therefore records `ingest_wall_s` and `build_wall_s`
separately and marks which engine could separate them, so the report can say
"pgvector's bulk build is N× cheaper than its own incremental build" and
"pgvector's incremental build costs X versus MHNSW's Y" — two honest claims
instead of one misleading one.
"""

from __future__ import annotations

from typing import Optional

import numpy

from ..datasets import FILTER_CTAG, FILTER_TAG, Dataset, filter_values
from ..drivers.base import EngineDriver, IndexSpec, RowAttrs
from ..metrics.latency import Timer
from ..metrics.records import PHASE_BUILD, PHASE_INGEST, Record, Recorder
from ..progress import Heartbeat
from .index_size import index_size_shape, settled_index_bytes
from .context import RunContext


def run(ctx: RunContext, driver: EngineDriver, dataset: Dataset,
        index: IndexSpec, load_threads: int = 1,
        subset_rows: Optional[int] = None) -> None:
    """Load the dataset and build the index, recording the cost of each."""
    vectors = dataset.train
    if subset_rows is not None and subset_rows < len(vectors):
        vectors = vectors[:subset_rows]

    # Both filter columns, loaded in the one build -- PR 3's option A, argued
    # in `drivers/base.py:RowAttrs` and `datasets.py:FILTER_COLUMNS`. Three
    # things about these two lines are load-bearing.
    #
    # `filter_values` for BOTH columns, including `tag`, whose values it just
    # computes with `assign_tags`. The filtered workload asks the same
    # function for the values it builds exact ground truth over, keyed on the
    # same (dataset, rows) pair, so the column in the engine and the column
    # the yardstick was computed from cannot be two different arrays. Calling
    # `assign_tags` here and `filter_values` there would be two paths that
    # agree today.
    #
    # `len(vectors)`, not `subset_rows`: the pivots are drawn from the rows
    # given, so clustering 2,000 rows and clustering the first 500 of them
    # produce different labels for those 500 -- measured, 497 of 500 differ.
    # `main.py` takes `indexed_rows` from `driver.count_rows()` after this,
    # which is this number and not the profile's request.
    #
    # Before `drop_schema` and before either timer below, so the cluster
    # assignment is charged to neither ingest nor build. It is cached and
    # measured at 0.5-3.2 s for 1M rows against a ~1.9 h MHNSW ingest, but
    # "small" is not "outside the measurement".
    #
    # It is also a new SIDE EFFECT in a phase that previously had none:
    # `cached_ctags` does `makedirs(cache_dir, exist_ok=True)` and saves an
    # .npy, so a build-only profile -- `config/profiles/mariadb-blog-repro.yml`
    # declares `workloads: [build]` -- now writes a cache entry for labels
    # that run will never query. ~0.6 s and a few megabytes for glove, and
    # the alternative is a flag threading "will anything filter later?" from
    # the profile down to here, which is more coupling than the cost buys.
    #
    # One new failure mode, worth knowing before it costs someone an hour:
    # `assign_ctags` refuses a corpus smaller than its cluster count (100 by
    # default), so `build.run` now needs at least that many rows where it
    # previously took any. Clamping the count to the row count is the obvious
    # alternative and the dangerous one -- the count is part of the ctag cache
    # key, so a build that clamped and a ground truth that did not would score
    # the engine against a different clustering, which is the one failure the
    # shared `filter_values` call above exists to prevent.
    attrs = RowAttrs(
        tag=filter_values(ctx.cache_dir, dataset, FILTER_TAG,
                          indexed_rows=len(vectors)),
        ctag=filter_values(ctx.cache_dir, dataset, FILTER_CTAG,
                           indexed_rows=len(vectors)),
    )

    driver.drop_schema()

    with Timer() as schema_timer:
        driver.create_schema(index)
    print(f"[{driver.name}] schema created in {schema_timer.elapsed:.2f}s")

    incremental = driver.capabilities().get("incremental_index", True)
    if index.build_mode == "incremental":
        incremental = True

    print(
        f"[{driver.name}] loading {len(vectors):,} x {index.dim} vectors "
        f"with {load_threads} thread(s); "
        f"{'graph maintained during load' if incremental else 'bulk build after load'}"
    )
    load = driver.load(vectors, attrs, threads=load_threads)
    print(
        f"[{driver.name}] load: {load.wall_seconds:.1f}s "
        f"({load.rows_per_second:,.0f} rows/s)"
    )

    build_seconds = 0.0
    if not incremental:
        # A bulk CREATE INDEX is a single call that can run for many minutes
        # with no way to report fractional progress.
        with Heartbeat("bulk index build", prefix=driver.name):
            with Timer() as build_timer:
                driver.create_index(index)
        build_seconds = build_timer.elapsed
        print(f"[{driver.name}] bulk index build: {build_seconds:.1f}s")

    # For incrementally-built graphs the honest build cost IS the load cost:
    # the graph work happened inside those INSERTs and cannot be separated.
    effective_build = build_seconds if not incremental else load.wall_seconds

    # Whatever status variables the driver declares, read HERE: after the
    # build has finished and been timed, and before the sizing and counting
    # queries below. Both halves of that matter.
    #
    # After the build, because the quantity of interest is what the completed
    # graph occupies -- Percona's `Innodb_hnsw_memory_used` reads 0 after a
    # restart and rises as nodes are faulted in, so a reading taken before the
    # load would describe an empty index.
    #
    # Before the sizing queries, because a status read is a statement and an
    # honest record says which statements preceded it. Nothing below touches
    # the HNSW graph today -- `SELECT COUNT(*)` and the catalog lookups do not
    # -- so the ordering changes no number; it is fixed so that it cannot
    # start to, and `tests/test_e20_server_status_capture.py` asserts it.
    #
    # Six of seven engines declare no gauges and this is one attribute read
    # and an early return for them.
    server_status = _server_status(driver)
    if server_status:
        # A FLOOR on graph memory, not a measurement of it: the figure omits
        # the node map (~30-50 B/node) and search scratch. The caveat is
        # printed with the number for the same reason it is in the hook's
        # docstring and in `PerconaDriver.status_gauges` -- whoever reads this
        # line in a run log is the first person to see the figure.
        print(f"[{driver.name}] server status (floor, omits the node map): "
              + ", ".join(f"{name}={value:,}"
                          for name, value in sorted(server_status.items())))

    index_bytes_after_build = driver.index_bytes()
    # Ops measurement version 9: the published size is read after a settle
    # step, and the immediate reading travels beside it. See
    # `harness/workloads/index_size.py` for the 15.1 -> 7.2 GiB run.
    index_bytes, settle_s, settle = settled_index_bytes(
        driver, index_bytes_after_build)
    table_bytes = driver.table_bytes()
    rows = driver.count_rows()
    print(
        f"[{driver.name}] index {index_bytes:,} B"
        + (f" (settled in {settle_s:.0f}s from {index_bytes_after_build:,})"
           if index_bytes != index_bytes_after_build else "")
        + f", table {table_bytes:,} B, {rows:,} rows"
    )

    common = ctx.record_defaults(driver, dataset, index)

    ctx.recorder.write(Record(
        **common,
        phase=PHASE_INGEST,
        rows=rows,
        ingest_wall_s=round(load.wall_seconds, 3),
        ingest_rows_per_s=round(load.rows_per_second, 2),
        ingest_threads=load.threads,
        notes=("graph maintained during INSERT" if incremental
               else "plain load, index built separately"),
    ))

    ctx.recorder.write(Record(
        **common,
        phase=PHASE_BUILD,
        rows=rows,
        build_wall_s=round(effective_build, 3),
        ingest_wall_s=round(load.wall_seconds, 3),
        ingest_threads=load.threads,
        index_bytes=index_bytes,
        table_bytes=table_bytes,
        notes=("incremental: build cost is the load cost" if incremental
               else "bulk build after load"),
        # The driver's own capabilities go in first. Without them an engine
        # whose index is built asynchronously by another process reported only
        # separable_build=True and was filed beside pgvector's bulk build,
        # which is a different operation, and its time-to-ready was dropped
        # entirely. separable_build is set last because it is a property of
        # how this workload ran, not of the driver.
        #
        # `server_status` is appended after it and collides with neither: it
        # is a reading taken off the server rather than a property of the
        # driver or of this workload. ONE key holding a {name: int} dict and
        # not one key per variable, so a second gauge needs no schema
        # decision and a report looks a name up inside a key it already
        # knows.
        #
        # **Conditional, and that is the point of the unit rather than a
        # tidiness.** Written unconditionally -- even as an empty dict -- this
        # would change the `extra` key set of every index_build record of all
        # seven engines, and `extra` key sets are what a reader diffs when
        # asking whether two runs measured the same thing. Absent means "this
        # engine reports no such variable", which is the honest claim for six
        # of the seven and for the seventh on a server that has lost the
        # variable.
        extra={**_driver_capabilities(driver),
               "separable_build": not incremental,
               **index_size_shape(driver, "after_settle"),
               "index_bytes_after_build": index_bytes_after_build,
               "index_size_settle_s": round(settle_s, 3),
               "index_size_settle": settle,
               **({"server_status": server_status} if server_status else {})},
    ))


def _driver_capabilities(driver) -> dict:
    """Whatever the driver knows about itself that the record should carry.

    Defensive: a driver whose capabilities() raises should not lose the
    measurement that has already been taken.
    """
    try:
        return dict(driver.capabilities() or {})
    except Exception:
        return {}


def _server_status(driver) -> dict:
    """The status variables this driver declares, as `{name: int}`, or `{}`.

    `_driver_capabilities`' discipline above, for the same reason and with one
    addition. The reason: this runs after a build that may have taken hours,
    so a driver or a server that misbehaves here must not cost the
    measurement already in hand -- hence the bare `except`.

    The addition is the `names` check. An engine that declares no gauges --
    `EngineDriver.status_gauges` is `()`, which is six of the seven here --
    is never ASKED, so the hook is free for everyone who does not use it and
    a future implementation that opens a connection or logs cannot be
    triggered by a driver that wants none of it.

    **It is not what keeps the change additive, and an earlier version of
    this comment claimed it was.** Three independent things do that, and
    deleting this return changes no record and sends no statement, because
    `MySQLFamilyDriver.status_counters` would iterate an empty tuple and the
    base hook touches no cursor at all -- measured, by deleting it. What this
    return buys is that the driver is not CALLED, which
    `tests/test_e20_server_status_capture.py` pins directly, since nothing
    else would notice.

    Coerced per name rather than in one comprehension: a single value that
    will not become an `int` loses its own name and not the gauges beside it.
    The comprehension this replaced was inside the `try`, so one bad value
    discarded the whole reading.

    **Every way this comes back empty prints something, and they are
    different lines.** The run log is this unit's only field diagnostic -- no
    test here can run against a server -- so "no gauge declared", "the driver
    declares one but implements nothing", "the statement was refused", "the
    server has no such variable" and "the value is not a number" must not
    share a silence. The last two are printed by
    `MySQLFamilyDriver.status_counters`; the middle one here; the first is
    visible in `status_gauges` itself.
    """
    names = tuple(getattr(driver, "status_gauges", ()) or ())
    if not names:
        return {}
    if type(driver).status_counters is EngineDriver.status_counters:
        # Declared a gauge and inherited the no-op. A silent `{}` here is
        # indistinguishable from a server that has no such variable, and the
        # two call for opposite responses: one is a missing override in this
        # repository, the other is a fact about the server.
        print(f"[{driver.name}] declares status gauges {list(names)} but no "
              f"driver implementation of status_counters, so none is "
              f"collected")
        return {}
    try:
        reported = driver.status_counters(names) or {}
    except Exception as exc:
        # Includes the `ValueError` a malformed `status_gauges` raises, which
        # is a bug in this repository and has to be findable: without this
        # line it reads exactly like a server with no such variable. The
        # first review of this change demonstrated that it did.
        print(f"[{driver.name}] status read failed: {exc}")
        return {}
    gauges: dict = {}
    for name, value in reported.items():
        try:
            gauges[str(name)] = int(value)
        except (TypeError, ValueError) as exc:
            print(f"[{driver.name}] status value for {str(name)!r} is not an "
                  f"int ({value!r}): {exc}")
    return gauges
