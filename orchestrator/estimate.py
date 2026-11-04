"""What a run will cost, before it starts.

Two estimates, accumulated separately and printed together:

* **Ingest** -- rows / measured rate, per engine, per dataset, multiplied by
  the M grid and the resource passes. This is the original estimate and its
  numbers are published: `docs/07-planning-a-run.md` quotes them, and the web
  UI renders `total_hours` as "estimated ingest". It does not move.
* **Query time** -- a registry of named terms (`_QUERY_TERMS`), each derived
  from keys the profiles already carry, with a typical figure and a ceiling
  carried separately wherever a budget caps the work.

Lifted out of `orchestrator/cli.py` once the second estimate arrived: `cli.py`
was 1,604 lines before this work and 2,056 with query time added inline,
against this repo's 800-line guideline. Those two are git facts and cannot
rot; a figure for what `cli.py` measures today would, and the first version of
this sentence went stale inside the same commit that wrote it. Most of what is
here is new rather than moved -- the ingest tables and `estimate_load_hours`
came across, the query half was written for it. `cli.py` re-exports everything
that was importable from it before, so `webui/control.py` and
`tests/test_config.py` resolve unchanged.

Nothing here starts a container, reads a run directory or touches the network.
It is arithmetic over a profile, which is what makes it testable
(`tests/test_estimate.py`) and cheap enough to call from the web UI's request
path on every form change.
"""

from __future__ import annotations

import sys
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence,
                    Tuple)

import yaml

# Imported for `MARIADB_ENGINES`, a two-element tuple, and nothing else. It is
# declared in ann_pass.py and belongs in engines.py with the rest of the
# registry; moving it needs both of those files, which this task does not own.
# It costs this module nothing in practice: the web UI's estimate handler
# already imports orchestrator.cli for KNOWN_ENGINES, so ann_pass and
# docker_ctl are on that path either way.
from . import ann_pass
from .ops_pass import (DEFAULT_FILTERED_BUDGET_S, _declared_scan_modes,
                       _filter_columns)

# Measured ingest rates, rows/s, on a Xeon Gold 6230.
#
# These were originally guessed at 400/150/5600 from a 60k-row smoke run. A
# real 1.18M-row load measured MariaDB at 147 rows/s average — 2.7x slower —
# because HNSW insert cost rises as the graph grows and rises again with M
# (glove-100: ~320 rows/s at M=8, ~70 at M=32). An estimate that is optimistic
# by 3x is worse than none, so these are the observed mid-grid numbers.
_INGEST_ROWS_PER_S = {"mariadb": 150, "mariadb123": 110, "alisql": 55,
                      "pgvector": 3000, "valkey": 40000, "mongodb": 60000,
                      # The M=16 `incremental` rate on glove-100-angular,
                      # which is the corpus `_REFERENCE_DIMS` (100) and
                      # `_REFERENCE_ROWS` (1,000,000) describe, so it needs no
                      # unscaling to sit here. Exactly MariaDB's precedent:
                      # its 150 is its own glove-100 measurement of 147.
                      # Provenance for this and the three entries below is in
                      # the comment above `_MEASURED_ROWS_PER_S`.
                      "percona": 275}
_DATASET_ROWS = {
    "fashion-mnist-784-euclidean": 60_000,
    "glove-100-angular": 1_183_514,
    "sift-128-euclidean": 1_000_000,
    "gist-960-euclidean": 1_000_000,
    "glove-25-angular": 1_183_514,
    "deep-image-96-angular": 9_990_000,
    # MariaDB's big-vector benchmark corpus. At 1536 dimensions the per-row
    # cost is far above the others, so the row count alone understates it —
    # see _DIM_PENALTY below.
    "dbpedia-openai-1000k-angular": 1_000_000,
    "dbpedia-openai-500k-angular": 500_000,
    "dbpedia-openai-100k-angular": 100_000,
}

# Ingest rate falls with dimensionality: every graph traversal during insert
# costs distance computations proportional to the vector width. The reference
# rates above were measured on 100-dim data, so wider datasets are scaled down.
# Sublinear because SIMD amortises part of it.
_REFERENCE_DIMS = 100
# The reference rates are for a ~1M-row corpus; smaller ones ingest faster.
_REFERENCE_ROWS = 1_000_000
_DATASET_DIMS = {
    "fashion-mnist-784-euclidean": 784,
    "glove-100-angular": 100,
    "sift-128-euclidean": 128,
    "gist-960-euclidean": 960,
    "glove-25-angular": 25,
    "deep-image-96-angular": 96,
    "dbpedia-openai-1000k-angular": 1536,
    "dbpedia-openai-500k-angular": 1536,
    "dbpedia-openai-100k-angular": 1536,
}


# Rates measured directly, per (engine, dataset), on a Xeon Gold 6230. These
# beat any model and are used whenever available.
_MEASURED_ROWS_PER_S = {
    ("mariadb", "fashion-mnist-784-euclidean"): 745,
    ("alisql", "fashion-mnist-784-euclidean"): 150,
    ("pgvector", "fashion-mnist-784-euclidean"): 5655,
    ("mariadb", "glove-100-angular"): 147,
    # Measured on dbpedia-openai-1000k (990k x 1536, M=16). The formula-based
    # fallback predicted ~19 h for MariaDB and ~52 h for AliSQL against actuals
    # of 3.3 h and 6.6 h, so it was 6-8x pessimistic at this dimensionality:
    # _dim_penalty overcharges for width once the per-row cost is dominated by
    # graph traversal rather than by the distance computation itself.
    ("mariadb", "dbpedia-openai-1000k-angular"): 74,
    ("mariadb123", "dbpedia-openai-1000k-angular"): 51,
    ("alisql", "dbpedia-openai-1000k-angular"): 43,
    # The three that do not maintain the graph on the write path. Their rates
    # are the load alone; the index build is separate and is not what this
    # estimate is for. Two to three orders of magnitude above the InnoDB
    # engines, which is the point.
    ("pgvector", "dbpedia-openai-1000k-angular"): 617,
    ("valkey", "dbpedia-openai-1000k-angular"): 30122,
    ("mongodb", "dbpedia-openai-1000k-angular"): 71182,
    # Percona Server, InnoDB HNSW. The `incremental` build mode at M=16, which
    # is this table's stated mid-grid convention, read off the archived ops
    # records of the run `full-20261015-215006`. The census behind them: 17
    # distinct `phase: ingest` measurements across six runs, deduped on
    # `(run_id, dataset, M, resource_pass, build_mode, rows)` -- `run_id`
    # included, because without it two pairs of distinct runs collapse and the
    # count comes out at fifteen.
    #
    # THE PASS IS `normalized` WHEREVER BOTH PASSES MEASURED THE CELL. This
    # key has no pass axis, and normalized is both the fairness baseline and
    # the slower of the two in every Percona pair ever measured -- tuned is
    # 9.8% / 6.0% / 3.8% faster at M=8/16/32 on fashion-mnist -- so the entry
    # errs pessimistic, which is the direction the comment above this table
    # requires. The whole pass spread is under 10%, against a factor of ~5
    # for M.
    #
    # ROWS MATTER AS MUCH AS M. These are the 60,000-row and 1M-row
    # full-corpus loads, not the 20,000-row ones from
    # `config/profiles/percona.yml`'s `ops.subset_rows`, which ran 13% faster
    # because HNSW insert cost grows with the node count already indexed.
    # `estimate_load_hours` multiplies by `_DATASET_ROWS`, so the rate and the
    # row count describe the same load.
    ("percona", "fashion-mnist-784-euclidean"): 339,   # 338.94, normalized
    ("percona", "glove-100-angular"): 275,             # 275.04, normalized
    # The one entry the pass convention could not be applied to: only the
    # TUNED pass has an M=16 incremental record for sift-128 (normalized ran
    # at M=8 and nothing else), so this is a tuned measurement standing in for
    # a key with no pass axis. By the fashion-mnist M=16 ratio the normalized
    # figure would be ~6% lower, so this entry is optimistic by about that
    # much and by nothing else. Taken as measured rather than scaled, because
    # a scaled figure is interpolation.
    ("percona", "sift-128-euclidean"): 427,            # 426.53, tuned
    #
    # gist-960-euclidean IS DELIBERATELY ABSENT AND MUST STAY ABSENT. It was
    # measured at exactly one M -- 32, at 48.17 rows/s, 20,760 s for one
    # reload -- and it is the slowest corpus here, so its single figure in a
    # table whose convention is "mid-grid" would be a mislabelled number and
    # not merely a conservative one. The formula fallback gives 70.8 rows/s
    # where the M=16 rate is ~100 (from the 2.07-2.09 M16/M32 ratio measured
    # on fashion-mnist and sift-128): 1.41x pessimistic and honestly a model,
    # against 2.08x pessimistic and dressed as a measurement. Interpolating an
    # M=16 figure from that ratio is not done; one measurement at one M is not
    # a grid.
    #
    # AND THIS TABLE CANNOT EXPRESS THE BUILD MODE. Every figure above is
    # `incremental`, which was Percona's ops default while
    # `capabilities.bulk_vector_index_build` was false and is still what its
    # ann phase builds (the ann module declares the index in CREATE TABLE).
    # The `post` plain load measured 2653.0 rows/s -- 7.8x the fashion-mnist
    # entry -- and is in neither table because no graph is built during it.
    # pgvector, Valkey and Percona Search are here with `post` figures for
    # the opposite reason: they never maintain the graph on the write path,
    # so `post` IS their load.
    #
    # **THE CAPABILITY HAS NOW BEEN FLIPPED and `post` is Percona's ops
    # default**, so the right ops rate is mode-dependent and this key still
    # cannot say so -- `_ops_units` reads `ops.percona_build_modes` and this
    # table does not. No `post` rate is added, because there is no
    # load-plus-build rate measured on the grid this table prices: the one
    # comparison at the current pin (`percona-build-modes`, fashion-mnist,
    # 60,000 rows, M=16, one load thread) took 195 s for `post` against 288 s
    # for `incremental`, one dataset at one M, which is a direction and not a
    # rate. `print_load_estimate` says the ops share is charged at the
    # incremental rate whenever the plan runs Percona `post` units.
    #
    # TWO PROVENANCE FACTS, RECORDED RATHER THAN CORRECTED FOR.
    #   * All twelve load-bearing records come from a binary built
    #     `march: native`, not the `x86-64-v3` the cross-engine invariant
    #     mandates. On this Xeon Gold 6230 `native` admits AVX-512 where
    #     `x86-64-v3` stops at AVX2 and the HNSW insert path is the
    #     SIMD-sensitive code, so the direction of concern is that these rates
    #     are optimistic. The archive offers no clean paired comparison, but
    #     correcting for the pass with the measured fashion-mnist ratio puts
    #     an implied `native` tuned-20k figure at ~407 rows/s against two
    #     `x86-64-v3` measurements of 413-418: a few percent, and if anything
    #     in the safe direction. Not adjusted for.
    #   * All seventeen records carry `engine_version 9.7.1-1-vector-bench` at
    #     `engine_tag 02a8b80c`. These are therefore rates for the OLD pin,
    #     and once the re-pin and its rebuild land they describe a binary that
    #     no longer exists. They are kept because a planning estimate from the
    #     old binary beats the `None` that was here before, and they are dated
    #     here so the first run at the new pin has something explicit to
    #     supersede: re-read the fashion-mnist M=16 incremental rate then and
    #     replace these if it has moved by more than ~10%.
    #
    # NO MEASUREMENT-VERSION BUMP, and the reason is not merely that this
    # module writes no records. It reads none either, at runtime: nothing here
    # reaches `build_signature`, `memory_stem` or `charts.series_key`, so no
    # recorded number changes value, meaning or series, which is the whole of
    # `OPS_MEASUREMENT_VERSION`'s charter.
    #
    # The version question that DOES apply points the other way: all three
    # entries above and the reference rate were READ OFF records stamped
    # `ops_measurement_version` **7**, while the constant is 8. It resolves,
    # and the argument is about this field and not about the versions in
    # general. `ingest_rows_per_s` is SINGLE-WRITER load throughput -- every
    # one of the seventeen census records carries `ingest_threads: 1` and the
    # ingest phase has no client pool -- and nothing since version 5, which
    # widened every row with the `ctag` column, has moved it:
    #
    #   * version 6 made the clients OS processes and moved every
    #     MULTI-CLIENT figure: `qps` and all four latency columns wherever
    #     `clients > 1`, the four `extra.*concurrency`/`scaling_*` keys, and
    #     the churn during-writes record's aggregate reader throughput,
    #     percentiles, p99 ratio and `extra.reinsert_rows_per_s`. TWO OF
    #     THOSE ARE THROUGHPUT FIGURES and neither is this one: a pool's
    #     `qps` and a re-insert rate against readers are not a one-writer
    #     load rate.
    #   * version 7 moved five one-client `extra.*scaling_status` fields and
    #     explicitly nothing else -- "No `qps`, no percentile, no efficiency".
    #   * version 8 moved `storage_engine`, an identity label and not a
    #     measurement. For Percona it stays populated; InnoDB is real here.
    #
    # So these three rates are commensurable with each other and with a load
    # measured at HEAD. (6 and 7 are the span of the CENSUS, not the
    # provenance of the entries: its one version-6 record is the `post` plain
    # load noted above, which is in neither table.) A future bump that does
    # move `ingest_rows_per_s` -- another column, a different load path, more
    # than one writer -- makes these a different quantity, and they must then
    # be re-read rather than carried forward.
}


def _dim_penalty(dataset: str) -> float:
    dims = _DATASET_DIMS.get(dataset, _REFERENCE_DIMS)
    return max(1.0, (dims / _REFERENCE_DIMS) ** 0.6)


def _size_penalty(dataset: str) -> float:
    """How much slower ingest is on this corpus because the graph is bigger.

    HNSW insert cost grows with the number of nodes already indexed, and on
    these engines that dominates dimensionality outright. Measured on one
    machine with MariaDB: 60k rows at 784 dims ran at 745 rows/s while 1.18M
    rows at 100 dims ran at 147 — twenty times the rows and nearly eight times
    *fewer* dimensions, and still five times slower.

    An earlier version scaled only by dimensionality, using a rate measured at
    1.18M rows, and was therefore ~17x too pessimistic on a 60k dataset. The
    reference rates below correspond to a 1M-row corpus.
    """
    rows = _DATASET_ROWS.get(dataset, _REFERENCE_ROWS)
    return max(0.05, (rows / _REFERENCE_ROWS) ** 0.55)


def _effective_rate(engine: str, dataset: str) -> Optional[float]:
    """Rows/s for this engine on this corpus, or None -- never a guess.

    Three outcomes, in order: a rate measured for this exact (engine, dataset)
    pair; the engine's reference rate scaled for width and corpus size; and
    **None**, when this repo has measured neither.

    The third used to be `_INGEST_ROWS_PER_S.get(engine, 300)`, and that
    default contradicted the comment above the table it read from. The table's
    own history is the argument: rates guessed at 400/150/5600 off a 60,000-row
    smoke run were optimistic by 2.7x against the first real 1.18M-row load,
    and "an estimate that is optimistic by 3x is worse than none". 300 rows/s
    is 2x the measured MariaDB figure and 5.5x the measured AliSQL one, so a
    new MySQL-family engine -- one building an HNSW graph inside every INSERT,
    which is the slowest shape here -- was modelled as the fastest of the three
    with nothing printed to say the number was invented.

    `None` is the same answer `_query_qps` already gives for a missing query
    rate, and it is handled the same way: the engine is NAMED in
    `engines_without_ingest_rate` and left out of the total, rather than folded
    into it behind a number a reader cannot tell from a measurement. An engine
    enters the published figure by being measured, which is how every rate in
    the two tables above got there.
    """
    measured = _MEASURED_ROWS_PER_S.get((engine, dataset))
    if measured:
        return float(measured)
    base = _INGEST_ROWS_PER_S.get(engine)
    if base is None:
        return None
    return base / (_dim_penalty(dataset) * _size_penalty(dataset))


# --- query time -------------------------------------------------------------
#
# The ingest estimate is ingest only, and stays that way: it is what its label
# says, docs/07-planning-a-run.md quotes its numbers, and two web-UI renderers
# publish `total_hours` as "estimated ingest". What follows is a *second*,
# separately accumulated estimate of time spent querying, so the printed output
# can show ingest, query time and a total -- and so "a run longer than its
# printed estimate is a bug" can be a true statement about something. Against
# the ingest figure alone it never was: that figure explicitly excludes every
# query, so every run that ever ran exceeded it.
#
# Two rules hold this together, and both exist because an over-confident
# estimate is worse than none:
#
#   * A cost that is a *cap* is reported as a cap. A churn budget of 1800 s is
#     a ceiling the workload may not reach, so the typical figure and the
#     ceiling are carried separately rather than collapsed into one number.
#   * A cost this repo has no measurement for is *named*, not guessed and not
#     scored zero -- the same treatment `datasets_without_estimate` already
#     gives a dataset with no row count.

# Single-client query rate per (engine, dataset), for the workloads whose cost
# is a number of queries rather than a span of time.
#
# Deliberately empty. No run in this repo has recorded a per-(engine, dataset)
# QPS anywhere this estimator can read, and the ingest table has already paid
# for the alternative: rates guessed off a 60,000-row smoke run were optimistic
# by 2.7x against the first real 1.18M-row load, which docs/07 calls worse than
# having no estimate at all. So every engine missing from here is reported in
# `engines_without_query_rate`, and the query-count costs are reported as not
# modelled. Add a measured figure and that engine drops out of the gap list;
# a term that needs a rate reads it from `plan.query_qps`.
_MEASURED_QUERY_QPS: Dict[Tuple[str, str], float] = {}

# What costs a number of queries rather than a span of time, keyed by the ops
# workload that pays it. Each is named in the output while no rate exists.
#
# The filtered QUERY LOOP is deliberately not here: it has a term of its own,
# which carries both the shape of the cost and the fact that nothing has
# measured it. Naming it in both places would have the printed output charge
# for it and disclaim it in the same breath.
#
# Its GROUND TRUTH is a different cost and is here, because neither figure
# includes it. It is harness CPU rather than engine time, it runs over the
# whole test set rather than `ops.max_queries`, and it sits outside
# `ops.filtered_budget_s` on purpose -- charging the engine's budget for the
# yardstick would make a cold cache look like a slow engine.
# `tests/test_estimate.py` pins which filtered-related disclosures exist, so
# a second one cannot be added without that being a decision.
_PER_QUERY_OPS_COSTS: Dict[str, Tuple[str, ...]] = {
    "churn": (
        "churn re-measurement: ops.max_queries before churn and again after "
        "each churn fraction",
        # churn_budget_s bounds _reinsert and nothing else: churn.py wraps
        # driver.delete_ids in a Heartbeat and a Timer with no budget, so a
        # quarter of a million tombstones is an open-ended cost. The term above
        # is named "churn re-insert" for exactly this reason, and saying so
        # here stops the printed caveat reading as "churn is fully bounded".
        "churn delete: ops.churn_fractions of the corpus removed per step, "
        "with no budget bounding the delete itself",
        # PR 4 §4.2's repair path, and the item `_QUERY_TERMS` below records
        # as owed-and-disclosed. Two costs, neither modellable: a THIRD
        # ops.max_queries measurement per fraction on the engines that have a
        # repair path (pgvector and Percona Server, two of the seven), and the
        # repair itself -- pgvector's `VACUUM (ANALYZE)` over a table a
        # quarter of whose rows are tombstones, or Percona's index replace,
        # which rebuilds the entire HNSW graph and so costs on the order of a
        # bulk build per fraction.
        #
        # **"because no live engine has been run" was true when this was
        # written and is not any more, and the correction cuts the other way
        # than you would expect.** `full-20261015-215006` ran Percona's churn
        # phase: the index replace returned in 1-4 ms and changed nothing --
        # identical `index_bytes` to the byte and identical recall to five
        # decimals in all 20 before/after pairs. That was the old pin's
        # index replace, and no one has re-tested it since. So
        # pgvector's VACUUM still has no measured rate, and Percona's repair
        # now has one of zero while `capabilities.vector_index_replace` is
        # false, because the driver withdraws the path and runs nothing. That
        # key used to be `bulk_vector_index_build`; it was split off when the
        # bulk build was re-verified and the replace was not. The disclosure string below therefore qualifies Percona
        # rather than dropping it: this module reads no engine configs, so it
        # cannot know which pin a given run is on, and a reader planning a run
        # on a FIXED server still needs the cost disclosed.
        # Named separately from
        # the re-measurement line above because that line is about the two
        # measurements every engine pays and this one is about the engines
        # that repair.
        # The engine list is named rather than counted, and the count it used
        # to carry ("pgvector of the six") was wrong the moment a seventh
        # engine with a repair path landed. Percona's is an index replace,
        # which rebuilds the whole HNSW graph -- on a 1M-row corpus that is on
        # the order of a bulk build per churn fraction, so it is the larger of
        # the two unmeasured repairs disclosed here, not a footnote to
        # pgvector's VACUUM.
        "churn repair path: on an engine that has one it can execute at its "
        "pinned version (pgvector's VACUUM (ANALYZE); Percona Server's index "
        "replace only while capabilities.vector_index_replace is true -- "
        "it is false at the pinned version, where the driver declares no "
        "repair path, runs nothing and pays neither cost), driver.maintain() "
        "per churn fraction plus a third ops.max_queries measurement of the "
        "repaired index -- the repair has no measured rate at any corpus "
        "size and nothing caps either",
    ),
    "concurrency": (
        "concurrency warm-up: 50 queries per client before each measured "
        "window",
    ),
    "filtered": (
        # Measured rather than modelled, and the figure ranges over the
        # WHOLE PROFILE rather than one dataset -- `main` has two.
        #
        # Rate: 2,000 test queries scaled to the 10,000 both datasets ship,
        # against 40,000 qualifying rows, five interleaved repeats per
        # dimensionality. Medians 158.7 s/M at 100 dims, 155.0 at 128, 164.7
        # at 256, 240.2 at 960, with a 5-8% spread within each and ~10% drift
        # between sessions on a shared box. So ~150 s/M is the working figure
        # and its uncertainty is the box, not the dimensionality: 128 dims
        # came out 0.98x of 100 dims -- marginally FASTER, inside the spread.
        #
        # That is a structural fact rather than luck. The dominant term
        # `batch @ subset.T` is linear in dim, so 128/100 would be 1.28x and
        # 960/100 would be 9.6x if the dot products ruled; the measured 0.98x
        # and 1.51x say the dim-INDEPENDENT work dominates instead -- the
        # n_test_batch x n_qualify score matrix and the argpartition over it.
        # Hence one rate for both of main's datasets, flat to a few hundred
        # dims and rising sub-linearly beyond.
        #
        # Total: glove is 11,835 + 118,351 + 591,757 = 721,943 qualifying
        # rows per column and sift 610,000, so both columns of both datasets
        # are ~217 + ~183 = ~400 s (6.7 min) against main's printed 4.0 h
        # filtered ceiling -- 2.8%. Small, cached and shared, and in neither
        # printed figure, which is the only reason it is named at all.
        "filtered ground truth: exact top-k recomputed by brute force over "
        "the qualifying rows of every (column, selectivity), over the whole "
        "test set rather than ops.max_queries and outside "
        "ops.filtered_budget_s; cached per (dataset, metric, k, column, "
        "threshold, rows) and shared across engines, ops units and passes, "
        "so paid once each -- measured at ~150 s per million qualifying rows "
        "(5-8% spread, near dimension-independent below ~256), which is "
        "~6.7 min for main's two datasets x two columns",
    ),
}
# Named, not quantified: there is no measured single-client ann QPS to
# multiply by, so `runs: 2` on the publishable profiles doubles a term this
# estimator never put a number on. The index warm-up is in the same position
# -- bounded (5,000 queries per built index) but with no rate to price it.
_ANN_QUERY_COST = ("recall/QPS sweep: every ann.ef_search point, `runs` times, "
                   "over each dataset's query set, plus a one-time index "
                   "warm-up of up to 5,000 training-set queries per built index")


def _query_qps(engine: str, dataset: str) -> Optional[float]:
    """Measured single-client QPS, or None -- never a guess."""
    return _MEASURED_QUERY_QPS.get((engine, dataset))


def _ops_rows(ops: Dict[str, Any], dataset: str) -> int:
    """Rows the ops phase actually loads, which is what churn works on.

    `ops.subset_rows` caps it (smoke loads 20,000 of fashion-mnist's 60,000).
    A dataset with no row count and a subset is bounded by the subset, which is
    an upper bound rather than a guess; one with neither is 0, and is already
    reported in `datasets_without_estimate`.

    Note the asymmetry with the ingest estimate, which does **not** read
    `subset_rows` and charges smoke the full 60,000 rows for both phases. That
    is pre-existing and deliberately frozen -- the ingest numbers were not
    allowed to move when query time was added -- so smoke's ingest figure is
    about three times its churn figure for the same corpus, and the difference
    is this key, not an error in either.
    """
    rows = int(_DATASET_ROWS.get(dataset, 0))
    subset = ops.get("subset_rows")
    if subset:
        return min(rows, int(subset)) if rows else int(subset)
    return rows


def _frontier(profile: Dict[str, Any]) -> bool:
    # Imported here: the selection module pulls in the manifest and docker
    # plumbing, which nothing else in this module needs.
    from .ops_build_selection import MODE_FRONTIER, build_selection_mode
    return build_selection_mode(profile) == MODE_FRONTIER


def frontier_ops_bound(profile: Dict[str, Any], engines: Sequence[str],
                       passes: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The ops build to charge when `ops.build_selection: frontier`, or None.

    Under `frontier` the ops build is not known until the ann phase of each
    (engine, dataset, pass) has run, so there is no build to estimate -- and
    printing the M=16 figure as though it were the plan would be the same
    unlabelled substitution the key exists to remove. What IS known is the
    grid it will be chosen from, so each engine's ops ingest is charged at
    the largest M its RENDERED grid asks for -- `estimate_grid.ann_builds`,
    which sees `m_values_by_engine`, the tuned extras and a select-builds
    overlay exactly as the run will -- at ONE build per unit (the selection
    picks exactly one, so `ops.m_values` no longer multiplies it).

    The M price is `estimate_grid.m_cost_factor`, the rule the per-build ann
    block beside this line uses, so the two blocks of one estimate cannot
    disagree about what an M costs: ~M^1.1 from the M=16 rate for the engines
    that build the graph inside every INSERT, and nothing for the others,
    whose rate is the load alone -- which also covers Percona Search, whose M
    is a label in a result file name and not a knob. ef_construction is not
    priced, because no ingest rate here was measured at more than one value
    of it -- so where the grid sweeps it above the rates' 200 (the tuned pass
    sweeps pgvector and Valkey to 400) this is NOT an upper bound, and the
    printed line says so.
    """
    if not _frontier(profile):
        return None
    from .config import load_resources, merge_resource_overrides
    from .estimate_grid import ann_builds, m_cost_factor

    per_engine: Dict[str, Dict[str, Any]] = {}
    for engine in engines:
        largest: Optional[int] = None
        for resource_pass in passes:
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            try:
                builds = ann_builds(engine, profile, resources, resource_pass)
            except ValueError:
                # `run` refuses an unrenderable grid before it starts; the
                # estimate has nothing to charge from and charges M=16.
                builds = []
            for build in builds:
                if build.get("M") is not None:
                    largest = max(int(build["M"]), largest or 0)
        per_engine[engine] = {
            "m": largest,
            "ingest_factor": round(m_cost_factor(engine, largest), 3)}
    return {"engines": per_engine,
            "m": max((r["m"] for r in per_engine.values()
                      if r["m"] is not None), default=None)}


def ops_ingest_factor(bound: Optional[Dict[str, Any]], engine: str) -> float:
    """The M factor one engine's ops ingest is charged at: 1.0 in `reference`
    mode, else what `frontier_ops_bound` priced for that engine."""
    if not bound:
        return 1.0
    return ((bound.get("engines") or {}).get(engine) or {}).get(
        "ingest_factor", 1.0)


def _ops_units(profile: Dict[str, Any], engine: str) -> int:
    """How many times the ops phase runs for one (engine, dataset, pass).

    `_run_unit` loops ops M values x build modes x storage engines, and every
    combination pays the query workloads again. pgvector is the one that bites
    today: `main` measures both `pgvector_build_modes`, so pgvector runs every
    ops workload twice per dataset.

    The storage-engine axis is read from the profile only. The tuned pass can
    widen it through `resources.extras.mariadb_storage_engines`, which this
    function is not given -- `_ops_storage_engines` takes `resources` and the
    estimate does not. Today every configured value is `[InnoDB]`, so nothing
    is lost; a profile that widens it would under-report here.
    """
    ops = profile.get("ops", {}) or {}
    ann = profile.get("ann", {}) or {}
    # One M under `frontier`: the selection builds the chosen build only.
    m_values = (1 if _frontier(profile)
                else max(1, len(ops.get("m_values", [16]))))
    # Mirrors `cli.py`'s `_OPS_BUILD_MODE_KEYS`, and must: this is how many
    # times the ops workloads actually run, so an engine whose second build
    # mode the runner executes and this does not count is an estimate short by
    # a whole unit of every query term. pgvector and Percona are the two
    # engines with a separable bulk build; for the rest `post` is a no-op and
    # a second mode would be the same measurement twice.
    #
    # The BLOCK is part of the key and not an implementation detail:
    # pgvector's list drives the ann render as well as the ops loop and lives
    # with the ann configuration, while Percona's drives the ops loop alone
    # and lives under `ops:` precisely so `ann_pass.render_config` cannot see
    # it. `cli.py` carries the argument at length.
    mode_ref = {"pgvector": ("ann", "pgvector_build_modes"),
                "percona": ("ops", "percona_build_modes")}.get(engine)
    if mode_ref:
        block, key = mode_ref
        source = ann if block == "ann" else ops
        build_modes = max(1, len(source.get(key, ["post"])))
    else:
        build_modes = 1
    if engine in ann_pass.MARIADB_ENGINES:
        storage = max(1, len(ops.get("storage_engines") or ["InnoDB"]))
    else:
        storage = 1
    return m_values * build_modes * storage


class _QueryPlan(NamedTuple):
    """Everything a query-time term is given, for one (engine, dataset).

    `units` is `_ops_units` for this engine -- ops M values x build modes x
    storage engines, i.e. how many times the ops workloads run per (dataset,
    pass). `passes` is the number of resource passes. A term with
    scope="unit" is multiplied by both; see `_QueryTerm.scope`.
    """
    profile: Dict[str, Any]
    ops: Dict[str, Any]
    ann: Dict[str, Any]
    workloads: List[str]
    phases: List[str]
    k: int
    engine: str
    dataset: str
    rows: int
    # Optional for the same reason `query_qps` is, and now with the same
    # meaning: None is "nothing here has measured an ingest rate for this
    # engine", not zero. A term that needs it must say so rather than
    # multiply by a default -- see `_churn_reinsert`.
    write_rows_per_s: Optional[float]
    query_qps: Optional[float]
    units: int
    passes: int


class _QueryTerm(NamedTuple):
    """One named contribution to query time.

    `typical_s` is what the configuration should cost; `worst_s` is what it
    cannot exceed. They differ whenever a budget caps the work, and the gap is
    the honest part of the estimate -- see `_churn_reinsert` for why collapsing
    them would have hidden two lost runs.

    `bounded` says whether `worst_s` is a real ceiling. A term with nothing
    capping it must say so, because the printed worst case is the number the
    "a run longer than its estimate is a bug" criterion refers to.

    `scope` says what the seconds are per. A term is always asked once per
    (engine, dataset) -- that is what a `_QueryPlan` describes -- so the scope
    only says how many times the cost recurs *within* that:

        "unit"  (default) once per ops unit, per pass: x units x passes.
                Everything the ops harness runs is per unit, because
                `_run_unit` reloads and re-measures for every combination of
                ops M value, build mode and storage engine.
        "pass"  once per resource pass: x passes. For work done once against
                an (engine, dataset) however many ops units follow.

    There is deliberately no whole-run scope. The accumulator cannot offer one
    honestly -- it would have to decide which (engine, dataset) iteration owned
    it -- and a cost that is genuinely independent of engine and dataset does
    not belong in a per-engine estimate.
    """
    name: str
    typical_s: float
    worst_s: float
    basis: str
    bounded: bool
    scope: str = "unit"


def _scope_factor(plan: _QueryPlan, scope: str) -> int:
    if scope == "unit":
        return plan.units * plan.passes
    if scope == "pass":
        return plan.passes
    raise ValueError(
        f"unknown query-term scope: {scope!r}. A term is asked once per "
        f"(engine, dataset); scope says how often the cost recurs inside "
        f"that -- 'unit' (x ops units x passes) or 'pass' (x passes).")


def _concurrency_windows(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """The measured windows of the concurrency workload.

    Exact, and the only term here that is: each point runs for a fixed span
    whatever the engine manages, which is why the workload is written that way
    ("a slow configuration cannot stretch the run indefinitely and every point
    is measured over an identical window").

    `ops.concurrency_repeats` is already honoured by `ops_pass.harness_args`
    and defaults to 1, so a profile that sets it today gets that many windows
    per point and this has to count them.

    The per-client warm-up is not here: it is 50 queries, and a query count
    needs a rate this repo does not have. See `_PER_QUERY_OPS_COSTS`.
    """
    if "concurrency" not in plan.workloads:
        return None
    counts = [int(c) for c in plan.ops.get("client_counts", [1])]
    # The workload skips any point with more clients than it has queries.
    max_queries = int(plan.ops.get("max_queries", 1000))
    points = [c for c in counts if c <= max_queries]
    duration = float(plan.ops.get("concurrency_duration_s", 20))
    repeats = max(1, int(plan.ops.get("concurrency_repeats", 1)))
    if not points or duration <= 0:
        return None
    # `+ 1`: ops measurement version 9 runs one discarded warm-up window,
    # same length, before each client count's first repeat
    # (`harness/workloads/concurrency.py`). The run pays it, so this counts it.
    seconds = len(points) * duration * (repeats + 1)
    # `ops.profile_window_clients`: one more window of the same length at
    # that count, outside the measured repeats, for the perf profile.
    profiled = plan.ops.get("profile_window_clients")
    if profiled is not None and int(profiled) in points:
        seconds += duration
    return _QueryTerm(
        name="concurrency windows",
        typical_s=seconds,
        worst_s=seconds,
        basis=(f"{len(points)} client count(s) x {duration:.0f}s"
               + (f" x {repeats} repeat(s)" if repeats > 1 else "")
               + " + one warm-up window each"
               + " per ops unit, fixed by the profile not by the engine"),
        bounded=True,
    )


def _charged_scan_modes(plan: _QueryPlan) -> List[str]:
    """The iterative-scan modes the filtered grid is charged for, per ops unit.

    Engine-gated, and the gate has to be here rather than left to the profile
    key being absent: `ops.pgvector_iterative_scan` is declared once in a
    profile and `_run_unit` runs it for pgvector only, so a profile-only
    reading would charge the other SIX engines -- `mariadb`, `mariadb123`,
    `alisql`, `mongodb`, `valkey` and `percona` -- for a grid they measure
    once. Six and not five: `mariadb123` is a second MariaDB and is as much an
    engine here as the first, which is the kind of omission a count in the
    sentence catches and a list alone does not. That is the mirror of
    `cli.py:_ops_iterative_scan`'s own first line, and the two must keep
    agreeing: the whole value of this term is that it describes the run the
    orchestrator will actually start.

    The engine name is spelled out rather than read from a capability, because
    this function cannot load an engine config -- see `_filtered_search` on the
    two approximations this term makes and why both fall on the safe side.
    """
    if plan.engine != "pgvector":
        return []
    return _declared_scan_modes(plan.ops.get("pgvector_iterative_scan"))


def _runs_filtered_search(engine: str) -> bool:
    """False only for an engine whose config declares `filtered_search: false`.

    Read from the engine config and not from the driver, which lives in
    `harness/` where the orchestrator must not reach. The two are held in
    agreement by `tests/test_e23_myvector_engine.py`, for the one engine that
    declines (MyVector v1.26.9 has no filtered search). Absence keeps the
    charge -- over-charging leaves a ceiling a ceiling.

    Narrow on purpose. An engine with no config at all (the estimator's own
    probe engines in the suite) is charged quietly. A config that exists and
    cannot be read or parsed is charged and SAID, on stderr, because the
    printed estimate is then built on an assumption about that engine.
    Anything else is a bug here and propagates.
    """
    return _runs_declared(engine, "filtered_search", "filtered-search")


def _runs_churn(engine: str) -> bool:
    """False only for an engine whose config declares `churn: false`.

    `_runs_filtered_search`'s rule for the churn terms: VillageSQL
    rejects DELETE on a table with a custom index, so its driver declines the
    workload and nothing of it runs. Every other engine declares nothing and
    is charged exactly as before.
    """
    return _runs_declared(engine, "churn", "churn")


def _runs_declared(engine: str, key: str, what: str) -> bool:
    """`capabilities.<key>` is not false in the engine config (see above)."""
    from . import config as config_mod
    try:
        caps = (config_mod.load_engine(engine) or {}).get("capabilities") or {}
    except FileNotFoundError:
        return True
    except (OSError, yaml.YAMLError) as exc:
        print(f"[estimate] WARNING: could not read config/engines/{engine}.yml "
              f"({exc}); charging its {what} term as if the engine "
              f"ran it", file=sys.stderr)
        return True
    return caps.get(key, True) is not False


def _filtered_search(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """Filtered search: a fixed number of queries per point, capped per point.

    What runs is `len(ops.filter_columns)` columns x `len(ops.selectivities)`
    selectivities x one iterative-scan mode x `ops.max_queries` queries per
    ops unit. What a query costs depends on a rate nothing here has measured
    -- on MariaDB at 1% selectivity the measured rate was 0.9 QPS against 326
    unfiltered, a 360x spread that no default could stand in for -- so the
    typical figure stays unquantified while a rate is missing.

    **What changed in T3c is the worst case.** `ops.filtered_budget_s` caps
    every point, `harness/workloads/filtered.py` stops the loop when it is
    spent, and the record says so (`extra.time_bounded`), so this term can
    report a real ceiling for the first time: points x budget. That matters
    beyond this term, because `query_worst_case_is_a_ceiling` is an `all()`
    over the terms -- filtered search was the one term holding the whole run
    figure open, and the printed block had to say "not yet a ceiling" about
    the number the "a run longer than its estimate is a bug" convention
    refers to.

    A profile that turns the cap off still gets no ceiling. The flag has a
    default in the workload and in `ops_pass`, so in practice every configured
    run is bounded; the unbounded branch is for `filtered_budget_s: 0` or
    `null`, both of which mean "no cap" the whole way down -- `ops_pass`
    emits `0` for either spelling and `filtered.run` converts a non-positive
    budget into an infinite deadline. An estimate that printed a ceiling for
    those would be inventing one.

    Both spellings used to be traps rather than options, and the docstring
    that invited them is what found it: `0` emitted `--filtered-budget 0`,
    whose first elapsed-time check fired immediately, so every point recorded
    ZERO queries; `null` emitted the string `'None'`, which argparse rejected
    with `SystemExit(2)` and took concurrency and churn down with the unit,
    because `harness/main.py` wraps all three in one `except`.

    The ceiling this reports is `points x budget`, and the real bound is one
    query per point above it: a deadline tested before each query cannot
    abandon the query already running.

    **Both figures below are per ops unit, matched to a per-unit
    denominator.** `main` runs SIX filtered points per ops unit on an engine
    with one mode -- two columns x three selectivities -- and TWELVE on
    pgvector, which declares two. With three engines that is 8 ops units
    (mariadb 1 + alisql 1 + pgvector 2 build modes, over 2 datasets), so
    4 x 6 + 4 x 12 = 72 points per profile. Pairing the per-unit overrun
    against the per-profile ceiling is the error that has cost this repo the
    most: docs once said 46 h where the code said 51.1, and CLAUDE.md was out
    by 270 h.

    At the 1.1 s a query measured for `tag` at 1%, the overrun is 6 x 1.1 =
    **6.6 s against the 0.50 h a one-mode unit's six points can spend** --
    0.37%, and the same 0.37% on pgvector, where both halves of the ratio
    double. The same ratio per profile is 72 x 1.1 = 79.2 s against the
    printed 6.0 h. Either pairing is honest; mixing them understates it
    twelvefold -- 72 points over 6, where before the mode factor it was
    eightfold over 48.

    **The iterative-scan modes ARE in the arithmetic now, and the factor is
    pgvector's alone.** `cli.py:_ops_iterative_scan` used to resolve the FIRST
    declared `ops.pgvector_iterative_scan` value and pin it, so the grid ran
    once per ops unit whatever a profile declared and a `* modes` here would
    have charged for runs nobody could start. It now passes every declared
    mode that validates, the workload loops them against one loaded index, and
    this term doubles for pgvector on both `main` and `full` -- which is why
    the factor had to arrive in the same change as the call site. An estimate
    short by a whole mode is not a conservative estimate: `worst_s` is the
    number the "a run longer than its estimate is a bug" convention refers to,
    so understating it converts a correct run into a reported bug and a false
    ceiling into a published one.

    **Two approximations, both in the safe direction, both of which could
    silently become unsafe.** This function is given a profile and an engine
    name, not resources and not an engine config, so:

      * The tuned pass's `resources.extras.pgvector_iterative_scan` is not
        read -- the same limitation `_ops_units` documents for
        `mariadb_storage_engines`, and for the same reason. Today `main.yml`,
        `full.yml` and `tuned.yml` all declare `[off, relaxed_order]`, so
        nothing is lost; extras that declared MORE modes than the profile
        would under-report here, which is the direction that matters.
      * Modes are counted as declared, without the engine-capability
        validation the call site applies, so a profile naming a mode pgvector
        does not have is charged for a grid the run will skip. That
        over-charges, and over-charging leaves a ceiling a ceiling.

    `_declared_scan_modes` is shared with that call site rather than
    re-derived here, so the two cannot disagree about what `[off, off]` means.
    """
    if "filtered" not in plan.workloads:
        return None
    if not _runs_filtered_search(plan.engine):
        # The harness skips the workload for an engine whose driver declines
        # it (`EngineDriver.declined_workloads`), so charging it here would
        # put hours into the ceiling for points nobody will run.
        return None
    # Defaulted exactly as ops_pass.harness_args defaults them. A profile that
    # omits either key still runs the workload, so charging nothing here would
    # drop a real cost out of the terms list *and* out of the disclosure list.
    selectivities = list(plan.ops.get("selectivities") or [0.1])
    columns = _filter_columns(plan.ops)
    max_queries = int(plan.ops.get("max_queries", 1000))
    budget = float(plan.ops.get("filtered_budget_s",
                                DEFAULT_FILTERED_BUDGET_S) or 0.0)
    # `max(1, ...)` and not `len(...)`: an engine with no iterative-scan knob,
    # and a pgvector profile that declares no mode, both measure the grid once
    # at whatever the engine's own default is. Zero modes is one pass, which
    # is the same convention `filtered._scan_modes` keeps by returning
    # `[None]` -- and a `len()` here would charge nothing at all for six of
    # the seven engines.
    mode_passes = max(1, len(_charged_scan_modes(plan)))
    points = len(columns) * len(selectivities) * mode_passes
    queries = points * max_queries
    shape = (f"{len(columns)} filter column(s) x {len(selectivities)} "
             f"{'selectivity' if len(selectivities) == 1 else 'selectivities'}"
             f" x {mode_passes} iterative-scan "
             f"{'mode' if mode_passes == 1 else 'modes'}"
             f" x {max_queries:,} queries = {queries:,} queries per ops unit")

    # The cap, when there is one. `typical` still needs a rate: a budget says
    # what the work cannot exceed, not what it will take, and collapsing the
    # two is what made the churn estimate wrong by the whole budget before
    # `_churn_reinsert` separated them.
    if budget > 0:
        worst = points * budget
        if plan.query_qps:
            typical = min(queries / plan.query_qps, worst)
            return _QueryTerm(
                name="filtered search", typical_s=typical, worst_s=worst,
                basis=(f"{shape} at a measured {plan.query_qps:,.1f} QPS, "
                       f"{budget:.0f}s cap/point"),
                bounded=True)
        return _QueryTerm(
            name="filtered search", typical_s=0.0, worst_s=worst,
            basis=(f"{shape}, at a rate nothing here has measured; "
                   f"{budget:.0f}s cap/point bounds it"),
            bounded=True)

    if plan.query_qps:
        seconds = queries / plan.query_qps
        return _QueryTerm(
            name="filtered search", typical_s=seconds, worst_s=seconds,
            basis=(f"{shape} at a measured {plan.query_qps:,.1f} QPS; nothing "
                   f"caps it, so this is a figure and not a ceiling"),
            bounded=False)
    return _QueryTerm(
        name="filtered search", typical_s=0.0, worst_s=0.0,
        basis=(f"{shape}, at a rate nothing here has measured and with no "
               f"budget anywhere in the run to cap it"),
        bounded=False)


def _churn_reinsert(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """Writing the churned rows back, which is the churn workload's long pole.

    `ops.churn_fractions` are cumulative: [0.10, 0.25] deletes and re-inserts
    10% and then a further 15%, for 25% in total rather than 35%.

    The typical figure uses the engine's *load* rate, the only write rate this
    repo has measured. The worst case is `ops.churn_budget_s` per step, and the
    distance between the two is the whole reason the budget exists: Valkey
    loaded this corpus at 30,195 rows/s and then wrote the same rows back at
    86, because pipelined writes into a populated index do not return. Two runs
    were lost waiting on it. An estimate that printed only the modelled figure
    would have been wrong by the entire budget, which is exactly the failure
    the run-time convention is about.

    The re-insert only. `churn_budget_s` bounds `_reinsert` and nothing else,
    so the delete that precedes each step is open-ended and is disclosed in
    `_PER_QUERY_OPS_COSTS` rather than counted here.

    **`churn_budget_s: null` and `: 0` are handled the way
    `_filtered_search` handles them, and both used to be defects here.**
    Found by T4a, which could not fix them because `orchestrator/` is not in
    its ownership:

    * `null` raised `TypeError: float() argument must be a string or a real
      number, not 'NoneType'` -- on the HOST, before any container starts, so
      a profile with that one key printed a traceback instead of an estimate.
      `or 0.0` is what `_filtered_search` already had, and the defect turned
      out to have three sites rather than two: the emission in `ops_pass`, the
      enforcement in `churn.run`, and this.
    * `0`, meaning "no cap", made `worst` stay 0 and the `worst <= 0` guard
      drop the term entirely -- so the estimate stopped disclosing a cost it
      could not bound, where the same input makes `_filtered_search` report
      one unbounded. Dropping a term is the one thing worse than printing it
      without a ceiling: a reader sees no churn cost at all.

    The guard is now on `rows`, which is what "this profile does not churn"
    actually means -- every fraction zero, or duplicated so that no step adds
    anything.

    No configured profile is affected: only `tuned-complete.yml` sets
    `churn_budget_s` and it sets 7200. Verified across all nine profiles at
    three engines and six, one pass and two -- every published figure and
    every `query_worst_case_is_a_ceiling` is unchanged.
    """
    if "churn" not in plan.workloads or plan.rows <= 0:
        return None
    if not _runs_churn(plan.engine):
        return None
    fractions = [float(f) for f in plan.ops.get("churn_fractions", [0.1])]
    # `or 0.0` so a key present with value null means "no cap" rather than a
    # TypeError, matching `_filtered_search` and `ops_pass`'s `_budget_arg`.
    # The same default `ops_pass` passes the harness (7200 s since ops
    # measurement version 9), so the estimate and the run cannot disagree.
    from orchestrator.ops_pass import DEFAULT_CHURN_BUDGET_S
    budget = float(plan.ops.get("churn_budget_s", DEFAULT_CHURN_BUDGET_S)
                   or 0.0)
    uncapped = budget <= 0
    # None means no ingest rate has ever been measured for this engine, which
    # is a different state from a rate of zero and is reported differently:
    # there is no duration to model, so `typical` stays at 0.0 and the basis
    # says why. That is exactly what `_filtered_search` above does when
    # `plan.query_qps` is None -- `typical_s=0.0` beside a real ceiling, with
    # "at a rate nothing here has measured" in the basis -- and the pair is
    # the honest reading rather than a claim that the work is free.
    #
    # It replaces `rate > 0 else budget`, which charged the whole budget as
    # the TYPICAL figure for an engine nothing had measured. No engine could
    # reach that branch when it was written, so no published figure moves.
    rate = plan.write_rows_per_s
    modelled = rate is not None and rate > 0
    typical = 0.0
    worst = 0.0
    rows = 0
    churned_so_far = 0.0
    for fraction in fractions:
        step = max(0.0, fraction - churned_so_far)
        if step <= 0:
            continue
        count = int(round(step * plan.rows))
        if count == 0:
            continue
        churned_so_far = fraction
        rows += count
        if uncapped:
            # No ceiling to collapse onto, so the modelled duration is all
            # there is -- and it is a figure, not a bound.
            typical += count / rate if modelled else 0.0
        else:
            typical += min(count / rate, budget) if modelled else 0.0
            worst += budget
    if rows <= 0:
        return None
    at_rate = (f"at ~{rate:,.0f} rows/s (the load rate; re-insert can be far "
               f"slower)" if modelled
               else "at a rate nothing here has measured")
    if uncapped:
        return _QueryTerm(
            name="churn re-insert",
            typical_s=typical,
            worst_s=typical,
            basis=(f"{rows:,} rows written back {at_rate}; "
                   f"ops.churn_budget_s turns the cap off, so this is a "
                   f"figure and not a ceiling"),
            bounded=False,
        )
    return _QueryTerm(
        name="churn re-insert",
        typical_s=typical,
        worst_s=worst,
        basis=(f"{rows:,} rows written back {at_rate}, "
               f"{budget:.0f}s cap/step"),
        bounded=True,
    )


def _churn_reader_baseline(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """The reader pool's baseline window, which is PR 4 §4.3's only new wall.

    Exact, and one of only two terms here that is: the window is a fixed
    `READER_BASELINE_WINDOW_S` per churn fraction whatever the engine manages,
    for the same reason the concurrency windows are fixed -- a slow
    configuration cannot stretch the run indefinitely and every measurement
    covers an identical span.

    The readers that run DURING the re-insert cost no wall clock of their own:
    they start after the delete, stop when `_reinsert` returns, and the
    re-insert was going to happen anyway (it is `_churn_reinsert` above). What
    they can do is slow the writer down, which makes the re-insert's own
    modelled duration optimistic -- but that duration already uses the LOAD
    rate against a re-insert measured 350x slower on Valkey, so a read-load
    correction on top of it would be a refinement of a figure whose error is
    dominated by something else entirely. Said here rather than modelled.

    None when the pool is off, which is `ops.churn_readers: 0` and is what
    `dev` runs: no readers, no baseline window, no cost.
    """
    if "churn" not in plan.workloads or plan.rows <= 0:
        return None
    if not _runs_churn(plan.engine):
        return None
    readers = _churn_readers_declared(plan.ops)
    if readers <= 0:
        return None
    fractions = _churn_steps(plan)
    if not fractions:
        return None
    seconds = fractions * CHURN_READER_BASELINE_WINDOW_S
    return _QueryTerm(
        name="churn reader baseline", typical_s=seconds, worst_s=seconds,
        basis=(f"{fractions} churn fraction(s) x "
               f"{CHURN_READER_BASELINE_WINDOW_S:.0f}s of {readers} reader(s) "
               f"before each step's deletes, fixed by the profile not by the "
               f"engine"),
        bounded=True)


def _mongot_index_settle(plan: _QueryPlan) -> Optional[_QueryTerm]:
    """mongot's index-size settle after each build (ops version 9).

    `harness/drivers/mongo.py:MongoDriver.settle_index_bytes` re-reads
    mongot's directory sum every `SETTLE_INTERVAL_S` until two readings
    agree within 1%, or until `SETTLE_CAP_S` -- because Lucene keeps merging
    segments for minutes after a build (`full-20260825` read 15.1 GiB after
    the load and 7.2 GiB at the churn baseline). Version 9 added that wait
    and the estimate did not count it, so the printed worst case was short by
    up to ten minutes per mongodb unit. Typical is one interval (the fewest
    readings that can agree); worst is the cap. mongodb only: every other
    engine's settle is immediate.
    """
    if plan.engine != "mongodb" or "build" not in plan.workloads:
        return None
    return _QueryTerm(
        name="mongot index-size settle",
        typical_s=MONGOT_SETTLE_INTERVAL_S, worst_s=MONGOT_SETTLE_CAP_S,
        basis=(f"one {MONGOT_SETTLE_INTERVAL_S:.0f}s reading interval typical, "
               f"the {MONGOT_SETTLE_CAP_S:.0f}s cap worst, after each mongodb "
               f"build"),
        bounded=True)


#: `harness.drivers.mongo.SETTLE_INTERVAL_S` / `SETTLE_CAP_S` at their
#: defaults (both env-overridable inside the container, which the host cannot
#: see). Pinned against the harness by `tests/test_estimate.py`.
MONGOT_SETTLE_INTERVAL_S = 30.0
MONGOT_SETTLE_CAP_S = 600.0


#: Seconds of reader traffic before each churn step's deletes. Matches
#: `harness.workloads.churn_readers.READER_BASELINE_WINDOW_S`, which is not
#: importable here -- the orchestrator runs on the host and the harness runs in
#: a container -- and is pinned against it by `tests/test_t4b_readers.py`.
CHURN_READER_BASELINE_WINDOW_S = 30.0


def _churn_readers_declared(ops: Dict[str, Any]) -> int:
    """`ops.churn_readers` as the orchestrator will emit it.

    Mirrors `ops_pass._churn_readers`, including its treatment of `null` as
    "use the default" rather than as 0: a cleared key means the profile wants
    whatever the harness default is, and reading it as "no readers" would make
    the estimate omit a cost the run then pays.
    """
    value = ops.get("churn_readers", DEFAULT_CHURN_READERS_ESTIMATE)
    return int(DEFAULT_CHURN_READERS_ESTIMATE if value is None else value)


#: Matches `harness.workloads.churn_readers.DEFAULT_CHURN_READERS` and
#: `ops_pass.DEFAULT_CHURN_READERS`; all three are pinned together by
#: `tests/test_t4b_readers.py`.
DEFAULT_CHURN_READERS_ESTIMATE = 4


def _churn_steps(plan: _QueryPlan) -> int:
    """How many churn fractions actually churn something.

    The same walk `_churn_reinsert` does and `churn.run` does: fractions are
    cumulative, so [0.10, 0.25] is two steps and [0.10, 0.10] is one. A
    separate function because three terms now need the count and a fourth
    reader of this file should not have to re-derive "cumulative" from the
    loop.
    """
    fractions = [float(f) for f in plan.ops.get("churn_fractions", [0.1])]
    steps = 0
    churned_so_far = 0.0
    for fraction in fractions:
        step = max(0.0, fraction - churned_so_far)
        if step <= 0:
            continue
        if int(round(step * plan.rows)) == 0:
            continue
        churned_so_far = fraction
        steps += 1
    return steps


# The extension point. Each entry takes a `_QueryPlan` and returns a
# `_QueryTerm` or None ("this profile does not run it"). Everything else --
# both totals, the per-engine split, the printed block and /api/estimate --
# follows from the registry, so adding a cost is one function and one line
# here, and nothing reshapes.
#
#   T1d  concurrency repeats  -- already multiplied inside _concurrency_windows
#   T2a  ef-calibration sweep -- gate on plan.phases/plan.ann, scope "unit"
#   T3c  filtered budget      -- DONE, inside _filtered_search: the budget
#                                became a real ceiling and the column loop
#                                doubled the points. The mode factor is still
#                                outstanding and the docstring says why
#   T4a  churn maintenance    -- DONE, but as a DISCLOSURE rather than a
#                                term, in _PER_QUERY_OPS_COSTS["churn"].
#                                Written as a term first and reverted, with
#                                the measurement: pgvector's repaired
#                                re-measurement is ops.max_queries queries
#                                per fraction that nothing caps, and
#                                maintain() itself has no measured rate at
#                                any corpus size, so the term had to report
#                                bounded=False. Since
#                                query_worst_case_is_a_ceiling is an all()
#                                over the terms, that ONE term turned the
#                                printed worst case from a ceiling into a
#                                figure for main, smoke, quick, full and
#                                tuned-complete at once -- and it did so
#                                while charging a churn query loop, which is
#                                the exact cost this file has always
#                                disclosed rather than charged. The two
#                                churn measurements that predate PR 4 are in
#                                that same disclosure for that same reason.
#                                A ceiling nobody measured is worse than a
#                                named cost outside the figure.
#   T4b  churn reader baseline -- DONE, in _churn_reader_baseline: 30 s per
#                                fraction, exact and genuinely bounded --
#                                fixed by the profile, not by the engine, the
#                                way the concurrency windows are. The readers
#                                during the re-insert add no wall of their
#                                own
#
# Ordered as the profiles order `ops.workloads`, so the printed block reads in
# the order the run pays the costs. The churn reader baseline follows the
# re-insert because the two belong to the same workload, not because it runs
# later -- it is measured before each step's deletes, and the deletes
# themselves are not a term here. (An earlier revision said "the two churn
# terms"; the maintenance one came back as a disclosure, so there is one.)
_QUERY_TERMS: Tuple[Callable[[_QueryPlan], Optional[_QueryTerm]], ...] = (
    _concurrency_windows,
    _filtered_search,
    _churn_reinsert,
    _churn_reader_baseline,
    _mongot_index_settle,
)


class _TermTally:
    """Sums query-time terms by name, keeping the order the registry produced.

    A term is asked once per (engine, dataset), so the same name arrives many
    times with a different basis each time (different write rates, different
    row counts). The distinct bases are kept so the printed line can quote one
    as an example rather than implying it covers every engine.
    """

    def __init__(self) -> None:
        self._order: List[str] = []
        self._typical: Dict[str, float] = {}
        self._worst: Dict[str, float] = {}
        self._bases: Dict[str, List[str]] = {}
        self._bounded: Dict[str, bool] = {}

    def add(self, term: _QueryTerm, factor: int) -> None:
        name = term.name
        if name not in self._typical:
            self._order.append(name)
            self._typical[name] = 0.0
            self._worst[name] = 0.0
            self._bases[name] = []
            self._bounded[name] = True
        self._typical[name] += term.typical_s * factor
        self._worst[name] += term.worst_s * factor
        # One uncapped contribution makes the whole term uncapped.
        self._bounded[name] = self._bounded[name] and term.bounded
        if term.basis not in self._bases[name]:
            self._bases[name].append(term.basis)

    @property
    def typical_s(self) -> float:
        return sum(self._typical.values())

    @property
    def worst_s(self) -> float:
        return sum(self._worst.values())

    def rows(self) -> List[Dict[str, Any]]:
        return [{
            "name": name,
            "typical_seconds": round(self._typical[name], 3),
            "worst_seconds": round(self._worst[name], 3),
            "typical_hours": round(self._typical[name] / 3600, 2),
            "worst_hours": round(self._worst[name] / 3600, 2),
            "basis": (self._bases[name][0] if len(self._bases[name]) == 1
                      else f"e.g. {self._bases[name][0]}"),
            "bounded": self._bounded[name],
            # A term that knows its shape but not its duration contributes
            # nothing to either figure. Printing "~0.0 h" for it would read as
            # "free" rather than "unmeasured", so it is flagged instead.
            "quantified": self._worst[name] > 0,
        } for name in self._order]


def _query_costs_not_modelled(profile: Dict[str, Any], workloads: Sequence[str],
                              phases: Sequence[str]) -> List[str]:
    """The query costs that exist, are paid, and cannot be put a number on."""
    ann = profile.get("ann", {}) or {}
    costs: List[str] = []
    if "ann" in phases and ann.get("enabled", True):
        costs.append(_ANN_QUERY_COST)
    for workload, texts in _PER_QUERY_OPS_COSTS.items():
        if workload in workloads:
            costs += list(texts)
    return costs


def _estimate_query_hours(profile: Dict[str, Any], engines: List[str],
                          datasets: List[str], passes: List[str],
                          phases: List[str]) -> Dict[str, Any]:
    """Accumulate every registered query-time term over the plan."""
    ops = profile.get("ops", {}) or {}
    ann = profile.get("ann", {}) or {}
    ops_runs = "ops" in phases and ops.get("enabled", True)
    workloads = ([str(w) for w in (ops.get("workloads") or [])]
                 if ops_runs else [])

    tally = _TermTally()
    per_engine_s: Dict[str, float] = {e: 0.0 for e in engines}

    for engine in engines:
        for dataset in datasets:
            plan = _QueryPlan(
                profile=profile, ops=ops, ann=ann, workloads=list(workloads),
                phases=list(phases), k=int(profile.get("k", 10)),
                engine=engine, dataset=dataset,
                rows=_ops_rows(ops, dataset),
                write_rows_per_s=_effective_rate(engine, dataset),
                query_qps=_query_qps(engine, dataset),
                units=_ops_units(profile, engine), passes=len(passes),
            )
            for make_term in _QUERY_TERMS:
                term = make_term(plan)
                if term is None:
                    continue
                factor = _scope_factor(plan, term.scope)
                tally.add(term, factor)
                per_engine_s[engine] += term.typical_s * factor

    terms = tally.rows()
    not_modelled = _query_costs_not_modelled(profile, workloads, phases)
    # Only an engine whose rate is missing *and* which is about to be asked to
    # run queries is a gap worth printing.
    no_rate = ([e for e in engines
                if any(_query_qps(e, d) is None for d in datasets)]
               if not_modelled and datasets else [])

    return {
        "query_hours": round(tally.typical_s / 3600, 2),
        "query_hours_worst": round(tally.worst_s / 3600, 2),
        "query_per_engine_hours": {e: round(s / 3600, 2)
                                   for e, s in per_engine_s.items()},
        "query_terms": terms,
        "query_worst_case_is_a_ceiling": all(t["bounded"] for t in terms),
        "engines_without_query_rate": no_rate,
        "query_costs_not_modelled": not_modelled,
    }


def estimate_load_hours(profile: Dict[str, Any], engines: List[str],
                        datasets: List[str], passes: List[str],
                        phases: List[str]) -> Dict[str, Any]:
    """Estimate ingest time before the run starts, and query time beside it.

    ann-benchmarks reloads the whole dataset for every M value, and the
    incrementally-building engines ingest at a few hundred rows per second. That
    multiplies quietly: a seven-value M grid over four datasets is days of pure
    loading. Showing the number up front is the difference between choosing that
    and discovering it six hours in.

    Returns the breakdown rather than printing it, so the CLI and the web UI
    cannot report different numbers for the same plan.

    `total_hours` still means *ingest*, and must keep meaning it:
    `webui/static/control.js` and `webui/static/setup.js` both render it as
    "estimated ingest ~N h", and docs/07-planning-a-run.md quotes its figures.
    Redefining it as the run total would have changed what both pages have been
    telling people. `ingest_hours` is the same number under a name that cannot
    be misread, and `worst_case_total_hours` is the figure to plan against --
    the one the "a run longer than its printed estimate is a bug" criterion
    refers to. `long_run` stays keyed on ingest because the warning it triggers
    is about M values reloading the corpus.

    Query time is accumulated separately by `_estimate_query_hours`; see the
    comment block above it for what is modelled, what is only a ceiling, and
    what is named as unknown instead of being guessed at.
    """
    m_count = max(1, len(profile.get("ann", {}).get("m_values", [16])))
    ops_bound = frontier_ops_bound(profile, engines, passes)
    ops_m = (1 if ops_bound
             else max(1, len(profile.get("ops", {}).get("m_values", [16]))))
    per_engine: Dict[str, float] = {}
    unknown: List[str] = [d for d in datasets if d not in _DATASET_ROWS]
    # An engine this repo has no ingest rate for, on any dataset the plan
    # loads. Keyed on the RATE and not on the row count, which is the care
    # `_estimate_query_hours`'s `engines_without_query_rate` takes for the
    # same reason: `dev`'s only dataset has no row count, and every engine
    # would land in this list if the test were "contributed no hours" rather
    # than "has no rate". A missing row count is already named separately, in
    # `datasets_without_estimate`, and belongs to the dataset.
    no_ingest_rate: List[str] = [
        e for e in engines
        if datasets and all(_effective_rate(e, d) is None for d in datasets)]

    for engine in engines:
        if engine in no_ingest_rate:
            # Named above and left out, exactly as a dataset with no row count
            # is skipped below. A key here would be `0.0`, and `0.0` in a
            # column of hours reads as "costs nothing" -- the opposite of what
            # is known about an engine that maintains an HNSW graph per INSERT.
            continue
        engine_h = 0.0
        ops_factor = ops_ingest_factor(ops_bound, engine)
        for dataset in datasets:
            rows = _DATASET_ROWS.get(dataset)
            if not rows:
                continue
            effective = _effective_rate(engine, dataset)
            if effective is None:
                # Reachable only for an engine measured on SOME datasets and
                # absent from `_INGEST_ROWS_PER_S`, which no engine is today:
                # the per-engine table is keyed by engine alone, so an engine
                # in it has a rate everywhere. Such an engine's figure would
                # be a floor over the datasets it has a rate for, and the gap
                # list above cannot say so because it requires ALL of them to
                # be missing. Named here rather than left as a silent skip.
                continue
            if "ann" in phases:
                engine_h += (rows / effective) * m_count / 3600
            if "ops" in phases:
                engine_h += (rows / effective) * ops_m * ops_factor / 3600
        per_engine[engine] = round(engine_h * len(passes), 2)

    total_h = round(sum(per_engine.values()), 2)
    query = _estimate_query_hours(profile, engines, datasets, passes, phases)
    # Summed from the rounded parts so the printed lines add up for a reader
    # checking the arithmetic by eye.
    return {
        "total_hours": total_h,
        "ingest_hours": total_h,
        "per_engine_hours": per_engine,
        "m_values": m_count,
        "ops_m_values": ops_m,
        # None in `reference` mode. Otherwise the ops build is chosen after
        # the ann phase and this is what the ops ingest was charged at.
        "ops_build_bound": ops_bound,
        "passes": len(passes),
        "phases": list(phases),
        "datasets_without_estimate": unknown,
        # The ingest half of the disclosure `engines_without_query_rate` makes
        # for the query half. An engine here contributed nothing to
        # `total_hours`, so that figure is a total over the engines it could
        # be computed for and not over the engines asked for -- a different
        # quantity from the same plan with this list empty, which is why
        # `print_load_estimate` says so rather than leaving a reader to
        # notice a name missing from the per-engine line.
        "engines_without_ingest_rate": no_ingest_rate,
        "long_run": total_h > 12,
        **query,
        "typical_total_hours": round(total_h + query["query_hours"], 2),
        "worst_case_total_hours": round(
            total_h + query["query_hours_worst"], 2),
    }


def client_cpu_line(profile: Dict[str, Any],
                    client_cpus: Dict[str, int]) -> Optional[str]:
    """The resolved client cpuset size beside the largest client count.

    Printed before the run because the two numbers are only meaningful
    together: a 32-client sweep from 4 client CPUs measures the clients, and
    the report can mark such a point hollow only after the hours are spent.
    """
    if not client_cpus:
        return None
    counts = ((profile or {}).get("ops", {}) or {}).get("client_counts") or [1]
    largest = max(int(c) for c in counts)
    sizes = ", ".join(f"{n} client CPU(s) ({name})"
                      for name, n in client_cpus.items())
    return f"clients   : {sizes} for up to {largest} clients"


def print_load_estimate(profile: Dict[str, Any], engines: List[str],
                         datasets: List[str], passes: List[str],
                         phases: List[str],
                         client_cpus: Optional[Dict[str, int]] = None,
                         axis_hint: bool = True) -> None:
    # `axis_hint` False when `estimate_grid.print_run_estimate` will print
    # the per-build block beneath: there the multiplier may be ef_construction
    # or quantization rather than M, and that block names the axis per engine
    # instead of the "Reduce ann.m_values" advice below.
    # Before the early return below: a short run still drives its largest
    # client count from whatever the client cpuset is.
    line = client_cpu_line(profile, client_cpus or {})
    if line:
        print(line)
    estimate = estimate_load_hours(profile, engines, datasets, passes, phases)
    # The gate is the whole run, not ingest alone. It used to be ingest, which
    # meant `smoke` -- 0.5 h of loading against a 3.5 h churn ceiling across
    # six engines -- printed nothing at all.
    if estimate["worst_case_total_hours"] < 1:
        return
    total_h = estimate["ingest_hours"]
    rows_per_pass = [f"{engine} ~{hours:.1f} h"
                     for engine, hours in estimate["per_engine_hours"].items()]
    print(f"\nestimated ingest time (loading only, before any queries):")
    print("  " + "  |  ".join(rows_per_pass))
    print(f"  total ~{total_h:.1f} h across {estimate['passes']} pass(es), "
          f"{estimate['m_values']} M value(s)")
    # Printed immediately under the total, because it changes what the total
    # is rather than adding a footnote to it: the figure above covers the
    # engines on the per-engine line and no others. The sentence names the
    # engines and both counts, so a reader can see at a glance that a
    # seven-engine plan produced a six-engine figure -- and never has to
    # reconcile a total against a per-engine line by subtraction.
    #
    # Deliberately not a modelled number for them. The ingest table's own
    # history is that a guessed rate reached docs/07 and two web-UI renderers
    # and was 2.7x optimistic; naming the gap costs a line and cannot be
    # mistaken for a measurement.
    missing = estimate["engines_without_ingest_rate"]
    if missing:
        counted = len(estimate["per_engine_hours"])
        print(f"  ! No measured ingest rate for {', '.join(missing)}, so the "
              f"total above is\n    {counted} of "
              f"{counted + len(missing)} engine(s) and excludes "
              f"{'them' if len(missing) > 1 else 'it'} entirely rather than "
              f"modelling\n    "
              f"{'them' if len(missing) > 1 else 'it'} at a guessed rate. A "
              f"smoke run supplies the figure; until then the ingest\n    "
              f"cost of this plan is higher than the number above, by an "
              f"amount nothing here knows.")
    # Percona's ingest rate is its `incremental` rate (see
    # `_INGEST_ROWS_PER_S`), which is right for its ann phase and for an
    # `incremental` ops unit. Its ops DEFAULT is `post` now that
    # `capabilities.bulk_vector_index_build` is true, and `post` loaded and
    # built faster in the one comparison at this pin -- so the ops share is a
    # likely overestimate, which is said rather than corrected with a rate
    # nothing measured. A profile that names no modes gets the ENGINE's
    # default, read off the same capability `cli._ops_build_modes` reads, so
    # flipping it back to false cannot leave this note claiming `post` units
    # the run will not build (found in review: a literal `["post"]` here did).
    from . import config as config_mod
    percona_caps = ((config_mod.load_engine("percona") or {})
                    .get("capabilities") or {}) \
        if "percona" in estimate["per_engine_hours"] else {}
    percona_default = (["post"]
                       if percona_caps.get("bulk_vector_index_build", True)
                       else ["incremental"])
    if ("percona" in estimate["per_engine_hours"]
            and "ops" in estimate["phases"]
            and "post" in ((profile.get("ops") or {})
                           .get("percona_build_modes", percona_default))):
        print("  percona's ops ingest is charged at its measured incremental "
              "rate; its post\n    units (the ops default) have no rate here "
              "and loaded faster in the one\n    comparison at this pin "
              "(195 s vs 288 s, 60,000 rows), so that share is likely high.")
    bound = estimate.get("ops_build_bound")
    if bound:
        priced = ", ".join(
            f"{engine} M={row['m']} x{row['ingest_factor']:g}"
            for engine, row in bound["engines"].items()
            if row["ingest_factor"] != 1.0)
        print(f"  ops build: unknown until each ann phase has run "
              f"(ops.build_selection: frontier); the ops ingest above is "
              f"charged\n    one build per unit at each engine's largest "
              f"swept M"
              + (f" ({priced})" if priced else "")
              + f", priced as the\n    per-build block prices M. "
              f"ef_construction is not priced, so where the grid sweeps it\n"
              f"    above 200 this is not an upper bound.")
    if estimate["long_run"] and axis_hint:
        print(f"  ! This is a long run. Each M value costs a full reload of every "
              f"dataset,\n    and MHNSW/VIDX build incrementally at a few hundred "
              f"rows/s. Reduce\n    ann.m_values or the dataset list to cut it "
              f"roughly proportionally.")
    _print_query_estimate(estimate)
    print()


def _print_query_estimate(estimate: Dict[str, Any]) -> None:
    """The second half: query time, its ceiling, the total, and the gaps.

    Printed as two figures per term wherever they differ. One confident number
    would be the wrong shape for these costs -- a budget is a ceiling, not a
    duration -- and the criterion that a run must not outlast its estimate is
    about the ceiling, so the ceiling is what is labelled as the number to plan
    against.
    """
    terms = estimate["query_terms"]
    if terms:
        print("\nestimated query time (the parts the profile puts a clock on):")
        width = max(len(t["name"]) for t in terms)
        for t in terms:
            if not t["quantified"]:
                figure = "cost not quantified"
            elif abs(t["worst_hours"] - t["typical_hours"]) < 0.05:
                figure = f"~{t['typical_hours']:.1f} h"
            else:
                figure = (f"~{t['typical_hours']:.1f} h typical, "
                          f"~{t['worst_hours']:.1f} h worst case")
            print(f"  {t['name']:<{width}}  {figure}")
            print(f"  {'':<{width}}    {t['basis']}")
        print(f"  subtotal ~{estimate['query_hours']:.1f} h typical, "
              f"~{estimate['query_hours_worst']:.1f} h worst case")

    print(f"\nestimated run time ~{estimate['typical_total_hours']:.1f} h "
          f"typical, ~{estimate['worst_case_total_hours']:.1f} h worst case")
    # The caveat comes before the claim, and the claim is conditional on it.
    # Printed the other way round, two adjacent lines said the worst case was a
    # number a run cannot exceed and then that a run can exceed it.
    ceiling = estimate["query_worst_case_is_a_ceiling"]
    if terms and not ceiling:
        loose = ", ".join(t["name"] for t in terms if not t["bounded"])
        print(f"  ! Not yet a ceiling: nothing caps {loose}, so a run can "
              f"exceed the worst case until that is fixed.")
        print("  Plan against it all the same: it is the largest figure this "
              "configuration justifies.")
    else:
        # "Materially", and the clause after it, because the claim was
        # absolute and is false by a small structural amount. Every budgeted
        # term -- filtered search per point, churn re-insert per step --
        # tests its deadline BEFORE starting work and cannot abandon the
        # query or slice already running, so each can overrun by one of
        # those. Measured at 0.37% of the filtered ceiling at 1.1 s a query;
        # churn carries the identical overshoot from the same `>` comparison
        # in `churn.py`. One clause rather than an entry per term in the
        # disclosure list below, which is for costs measured in whole hours.
        print("  Plan against the worst case: a run materially longer than "
              "it is a bug.")
        print("  Each budgeted term tests its deadline before starting work, "
              "so it can overrun\n    by the one query or slice already "
              "running when the budget expires.")

    gaps = estimate["query_costs_not_modelled"]
    if gaps:
        who = ", ".join(estimate["engines_without_query_rate"])
        print(f"  ! Neither figure includes the costs below. They are a number "
              f"of queries, and\n    there is no measured query rate for "
              f"{who or 'these engines'} to multiply them by:")
        for gap in gaps:
            print(f"      {gap}")
