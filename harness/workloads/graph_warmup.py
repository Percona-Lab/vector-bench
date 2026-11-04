"""One discarded pass over the query set, after a build, for an engine whose
graph is cold when the build returns.

WHY THIS EXISTS, AND FOR WHICH ENGINE. Percona Server loads its HNSW graph
lazily. At the pin (`vec_runtime_load` in *storage/innobase/vec/vec0hnsw.cc*,
commit 56124b03) "Only the entry point is read here. Every other node is
faulted in on demand when traversal reaches it" -- there is no preload. And
the server team stated on 2026-11-04 that `ALTER TABLE ... ADD VECTOR KEY`,
which is what a `post` build issues, evicts the table as part of the ALTER.
So a post build hands calibration a graph holding one node:
`Innodb_hnsw_memory_used` read on the `index_build` record was 0 in 95 of 95
post builds (20k, 60k and 1M rows), while an `incremental` build of the same
60k corpus read 207,407,472 B. Calibration's first grid point after a post
build was visibly slower than the same point after an incremental one, so
"post vs incremental" latency was a comparison of a cold graph against a warm
one -- of the server's page-in path, not of the two builds.

The engine is chosen by a DRIVER DECLARATION, `EngineDriver.
warm_graph_after_build`, which is True on `PerconaDriver` alone. No engine
name appears in harness logic. The other engines build their graph in memory
as they go (MHNSW's and VIDX's caches are filled by the inserts that built
them, pgvector's bulk build leaves its pages in shared_buffers, Valkey's
index is memory-resident) and nobody has measured a cold start for them after
a build, so they are left exactly as they were: no warm-up, no statement, no
record key.

**Both of Percona's build modes get it**, including `incremental`, whose
graph is already warm. The flag is per driver and not per build mode on
purpose: the within-engine post-vs-incremental comparison is the one this
exists to repair, and it is only symmetric if both sides were treated alike.
For an incremental build the pass is close to free -- it faults in nothing --
and its reading is the control the post reading is compared against.

WHICH QUERIES AND WHICH ef, and why that is deterministic. The pass runs the
query set calibration is about to run -- `dataset.test[:max_queries]`, the
same slice, capped the same way -- once, at the highest ef_search any later
phase of this ops unit can use: the top of the calibration grid or the
profile's `ef_search`, whichever is larger. HNSW search at a larger ef keeps
a wider candidate list and expands every node a smaller ef would have
expanded on the way to the same answer, plus more; the node sets are not a
strict superset query by query (the layer-0 descent order can differ once
the candidate lists differ), but they are close to one, and the top ef's set
is the largest a later phase will ask for. So one pass at that ef faults in
what CALIBRATION will touch, with no plateau heuristic deciding when the
graph "looks warm" -- which, unlike this, would stop at a different point on
every run.

**WHAT IT DOES NOT COVER, stated rather than implied.** Only calibration's
query set is warmed. Concurrency draws from its own `--concurrency-queries`
budget, filtered search adds a predicate that walks the graph differently,
and churn queries a graph its own writes are changing; nodes only those
reach are still faulted in inside their windows. Concurrency's discarded
warm-up window absorbs some of it, and the gauge is not re-read after any
later phase, so how much remains is not measured. And an ops invocation
whose workloads omit `build` -- one run against a table an earlier
invocation built -- gets no warm-up and carries no `graph_warmup_status`
at all: no shipped profile does that, and a record without the key must not
be read as warm.

Results are DISCARDED and nothing is scored. Its wall time is NOT build time:
`build.run` has already computed `build_wall_s` from the load and the
`create_index` timer before this runs, and this only adds a key beside it.

WHAT IT RECORDS -- `extra.graph_warmup` on the `index_build` record, see
`warm` -- and why there rather than as a record of its own: the records file
is append-only, so "the after-build reading next to the after-warm-up
reading" is only one record if the warm-up runs before that record is
written. It is a property of the build the record describes (how the graph
was left for every later phase), and there is no phase in `ALL_PHASES` for
"queries nobody scored". The existing `extra.server_status` keeps its exact
meaning -- the reading taken straight after the build -- and the second
reading lives under `extra.graph_warmup.server_status`, a different path, so
a reader grepping either finds one meaning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

#: `extra` key on the `index_build` record holding what the warm-up did.
EXTRA_KEY = "graph_warmup"

#: Run-wide `extra` key stamped on every record written after the build, so a
#: calibration or concurrency figure says whether it was measured on a graph
#: this harness warmed. Its value is the warm-up's `status`.
RUN_EXTRA_KEY = "graph_warmup_status"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class WarmupPlan:
    """What a warm-up would run: the queries and the ef. Built in `main.py`,
    where the grid, `ef_search` and `max_queries` are known, and handed to
    `build.run`, which decides whether the driver wants it.

    `queries` is the slice calibration will take, not a copy of it.
    """

    queries: Any
    ef_search: int


def top_ef(ef_grid: Sequence[int], ef_search: int) -> int:
    """The highest ef_search a later phase of this ops unit can use.

    Calibration chooses from the grid; fixed mode measures at `ef_search`;
    the grid's maximum is included even in fixed mode, so a build-only or a
    fixed-mode run warms at the same ef as a calibrated one and its
    after-warm-up memory reading is the same quantity.
    """
    return max([int(ef) for ef in ef_grid] + [int(ef_search)])


def plan(test_vectors, max_queries: int, ef_grid: Sequence[int],
         ef_search: int) -> WarmupPlan:
    """The plan for this run: calibration's query slice at `top_ef`."""
    return WarmupPlan(queries=test_vectors[:max_queries],
                      ef_search=top_ef(ef_grid, ef_search))


def wanted(driver) -> bool:
    """Whether this driver declared that its graph is cold after a build.

    Through `getattr`, as `_query_path_facts` reads its hook in `main.py`:
    test fakes and lab scripts build drivers that are not `EngineDriver`
    subclasses, and those must behave as every undeclared engine does.
    """
    return bool(getattr(driver, "warm_graph_after_build", False))


def warm(driver, warmup: WarmupPlan, k: int,
         read_status: Callable[[Any], Dict[str, int]]) -> Dict[str, Any]:
    """Run the pass and return what `extra.graph_warmup` should hold.

    Never raises. A warm-up runs after a build that may have taken hours, and
    the build's record has not been written yet, so a query that fails here
    must not cost it -- the discipline of `build._driver_capabilities` and
    `build._server_status`. A failure is recorded as `status: failed` with
    how far it got, and the status gauges are still re-read, because "the
    pass died at query 312" is exactly when the partial memory figure is
    worth having.

    `read_status` is `build._server_status`, passed in rather than imported
    so this module does not import the workload that calls it.
    """
    queries = warmup.queries
    if len(queries) == 0:
        # No query set at all -- a dataset with an empty test split. Recorded
        # rather than silently skipped: the absence of a warm reading has to
        # be told apart from a warm-up that was never asked for.
        print(f"[{driver.name}] graph warm-up: skipped, no query vectors")
        return {"status": STATUS_SKIPPED, "reason": "no query vectors",
                "queries": 0, "ef_search": warmup.ef_search}

    # Printed BEFORE the pass: on a cold 1M-row graph it is minutes of
    # queries at the top ef with nothing else in the log, which reads as a hang.
    print(f"[{driver.name}] graph warm-up: {len(queries):,} discarded queries "
          f"at ef={warmup.ef_search} (the graph is faulted in lazily)")
    completed = 0
    error: Optional[str] = None
    started = time.perf_counter()
    try:
        driver.set_ef_search(warmup.ef_search)
        for q in queries:
            driver.query(q, k)  # discarded: nothing here is scored
            completed += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - started

    # Re-read AFTER the pass, so a bulk-built Percona graph gets a real
    # `Innodb_hnsw_memory_used` beside the 0 the after-build reading carries.
    # Same floor caveat as that reading: the gauge omits the node map.
    status = read_status(driver)
    try:
        effective = int(driver.effective_ef(warmup.ef_search, k))
    except Exception:
        effective = None

    out: Dict[str, Any] = {
        "status": STATUS_OK if error is None else STATUS_FAILED,
        "queries": completed,
        "ef_search": warmup.ef_search,
        "effective_ef": effective,
        "wall_s": round(wall_s, 3),
    }
    if error is not None:
        out["queries_planned"] = len(queries)
        out["error"] = error
    if status:
        # Conditional for the reason `extra.server_status` is: absent means
        # "nothing was read", which an empty dict would blur.
        out["server_status"] = status

    gauges = ", ".join(f"{name}={value:,}"
                       for name, value in sorted(status.items()))
    if error is None:
        print(f"[{driver.name}] graph warm-up: {completed:,} queries at "
              f"ef={warmup.ef_search}, {wall_s:.1f}s"
              + (f"; {gauges} (floor, omits the node map)" if gauges else ""))
    else:
        print(f"[{driver.name}] graph warm-up FAILED after {completed:,} of "
              f"{len(queries):,} queries at ef={warmup.ef_search} "
              f"({wall_s:.1f}s): {error}; the build record is kept"
              + (f"; {gauges}" if gauges else ""))
    return out
