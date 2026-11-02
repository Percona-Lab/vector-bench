"""One-time index warm-up, shared by every ann overlay module.

Why it exists. In full-20260825 every MySQL-family sweep measured ef_search 10
SLOWER than ef_search 20 -- 1.69x to 2.13x on all six MariaDB and AliSQL
curves of a 990k x 1536 corpus -- and both Percona Search curves did the same
at 1.13x. Examining more candidates cannot be faster on the same index, so the
first point was measuring a cold graph, not the engine, and it still set
numbers in the recall-floor table. The 30 random-vector queries each module's
`_warm_cache` runs before every configuration were never sized to warm a graph
that large; they stay, but they are not this.

What it does. Before the FIRST configuration measured on a freshly built index,
query that index with training-set vectors in batches of `WARMUP_BATCH` until
two consecutive batches agree on mean latency to within `WARMUP_TOLERANCE`,
or `WARMUP_CAP` queries have run. A stop rule rather than a fixed count,
because the amount of warming a graph needs scales with the graph and the
cache budget, and a fixed count tuned on one corpus under-warms the next.

Why TRAINING vectors. The test set is what ann-benchmarks is about to time.
Warming with it would pre-cache exactly the graph paths those queries walk and
flatter every engine by however much of its graph that is -- a different
amount per engine, so not even a fair flattery. Training rows are drawn from
the same distribution, so they warm the same regions of the graph in general
without priming any measured query in particular. The module never receives
the test set except through `query`/`batch_query`, and this code is not on
that path.

Why one helper. Every engine must get the identical treatment -- the same
rule, the same cap, the same sample -- or the warm-up becomes one more knob
that differs between the curves being compared. Each module only decides
WHEN (end of `fit` keeps the sample; first `set_query_arguments` spends it)
and HOW one query is issued (its own `query`).

The count and duration land in the result attributes
(`index_warmup_queries`, `index_warmup_seconds`, `index_warmup_converged`) so
the report can tell a sweep that was warmed from one that was not: a
first-step throughput inversion on a warmed index is no longer the cold-cache
artifact and is reported as unexplained.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, Optional

import numpy

# Queries per batch; the stop rule compares consecutive batch means.
WARMUP_BATCH = 200
# Two consecutive batch means within this fraction of each other = warm.
WARMUP_TOLERANCE = 0.05
# Hard ceiling. A graph still settling after 5,000 queries is reported as not
# converged rather than warmed for ever -- the attribute says which happened.
WARMUP_CAP = 5000
# Rows kept from the training set. One per query at the cap, so no vector is
# repeated unless the corpus itself is smaller; ~30 MB at 1536 dimensions,
# against a corpus of several GB that is not kept.
WARMUP_SAMPLE_ROWS = WARMUP_CAP
# Fixed, so every engine is warmed with the same vectors in the same order.
WARMUP_SEED = 0
# The k each warm-up query asks for; the benchmark's own k is 10 throughout.
WARMUP_K = 10


def training_sample(X: Any, rows: int = WARMUP_SAMPLE_ROWS,
                    seed: int = WARMUP_SEED) -> numpy.ndarray:
    """A fixed, seeded sample of training rows, as a copy.

    A copy and not a view: a view keeps the whole training set alive for as
    long as the algorithm object lives, which is the whole sweep.
    """
    total = int(len(X))
    take = min(int(rows), total)
    order = numpy.random.RandomState(seed).choice(total, size=take,
                                                  replace=False)
    return numpy.array(X[order], copy=True)


def warm_until_stable(query_one: Callable[[Any], Any],
                      vectors: numpy.ndarray, *,
                      batch: int = WARMUP_BATCH,
                      tolerance: float = WARMUP_TOLERANCE,
                      cap: int = WARMUP_CAP,
                      clock: Callable[[], float] = time.perf_counter
                      ) -> Dict[str, Any]:
    """Query until two consecutive batches agree, or the cap is reached.

    `query_one(vector)` issues one query. The sample is cycled when it is
    shorter than the cap. A query that raises ends the warm-up rather than the
    run: the measurement that follows will meet the same fault and report it
    where it belongs, and a warm-up crash would otherwise lose the whole
    configuration with no result file.
    """
    done = 0
    seconds = 0.0
    previous: Optional[float] = None
    converged = False
    n = len(vectors)
    while done < cap and n:
        size = min(batch, cap - done)
        started = clock()
        try:
            for i in range(size):
                query_one(vectors[(done + i) % n])
        except Exception as exc:  # noqa: BLE001 - see docstring
            seconds += clock() - started
            done += i
            print(f"[vb] index warm-up stopped after {done} queries: {exc}",
                  file=sys.stderr)
            break
        elapsed = clock() - started
        seconds += elapsed
        done += size
        mean = elapsed / size
        if previous is not None:
            if previous > 0:
                converged = abs(mean - previous) / previous < tolerance
            else:
                converged = mean <= 0
            if converged:
                break
        previous = mean
    return {"queries": done, "seconds": seconds, "converged": converged}


class IndexWarmupMixin:
    """Wiring for the overlay modules. Class-level defaults, so no module's
    `__init__` has to know about it.

    A module calls `self._keep_warmup_sample(X)` in `fit` and
    `self._warm_index_once()` in `set_query_arguments`, after its own knob is
    set (the warm-up queries run at the first configuration's search width),
    and merges `self._warmup_attrs()` into `get_additional`.
    """

    _vb_warmup_sample: Optional[numpy.ndarray] = None
    _vb_warmup_stats: Optional[Dict[str, Any]] = None

    def _keep_warmup_sample(self, X: Any) -> None:
        # A new fit is a new, cold index: forget the previous warm-up.
        self._vb_warmup_sample = training_sample(X)
        self._vb_warmup_stats = None

    def _warm_index_once(self, clock: Callable[[], float] = time.perf_counter
                         ) -> None:
        sample = self._vb_warmup_sample
        if self._vb_warmup_stats is not None or sample is None:
            return
        # Released before querying, not after: the sample is only needed by
        # the call below, which holds its own reference.
        self._vb_warmup_sample = None
        stats = warm_until_stable(lambda v: self.query(v, WARMUP_K), sample,
                                  clock=clock)
        self._vb_warmup_stats = stats
        print(f"[vb] index warm-up: {stats['queries']} training-set queries "
              f"in {stats['seconds']:.1f}s, "
              f"{'converged' if stats['converged'] else 'did NOT converge'}",
              file=sys.stderr)

    def _warmup_attrs(self) -> Dict[str, Any]:
        """Result attributes. Empty until the warm-up has run -- never None
        values, which ann-benchmarks cannot write into HDF5."""
        stats = self._vb_warmup_stats
        if stats is None:
            return {}
        return {
            "index_warmup_queries": int(stats["queries"]),
            "index_warmup_seconds": round(float(stats["seconds"]), 3),
            "index_warmup_converged": bool(stats["converged"]),
        }
