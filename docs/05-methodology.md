# Methodology

This document states exactly what is measured, how, and what the numbers do *not*
mean. Read it before quoting any result.

## 1. What is being compared

Three relational databases with **built-in** vector search, all using an
HNSW-family index:

| Engine | Version | Index implementation | Source |
| --- | --- | --- | --- |
| MariaDB | 11.8.8 (LTS) | MHNSW (`sql/vector_mhnsw.cc`) | `mariadb-11.8.8` tag |
| AliSQL | 8.0.44-2 | VIDX (`sql/vidx/vidx_hnsw.cc`) | `AliSQL-8.0.44-2` tag |
| PostgreSQL | 17 + pgvector 0.8.x | pgvector HNSW | `pgvector` release tag |

All three run HNSW. The comparison therefore isolates **implementation quality**,
not algorithm choice. That is deliberate: comparing HNSW against IVFFlat or
DiskANN would measure a different thing.

This is **not** a comparison against dedicated vector databases (Milvus, Qdrant,
Weaviate). Those live in the same ann-benchmarks harness and can be added later,
but the question here is "MariaDB vs AliSQL vs PostgreSQL".

## 2. Measurement dimensions

### 2.0 The operating point: the one `ef_search` three sections share

§2.3, §2.4 and §2.5 are each measured at **one** `ef_search` per engine, and by
default that value is chosen by a measurement rather than declared in a
profile. So this section is the premise of those three rather than a note about
them, which is why it comes before §2.1 — and why the report puts the same
table before the same three sections instead of after them
(`report/render.py:_operating_point_table`). A reader who meets those
throughput numbers first has already read them as equal-effort comparisons by
the time any caveat arrives.

#### A fixed `ef_search` is not one quantity across these engines

It is a different knob in each of them, and this repository names every one:

| Engine | What the harness sets | Where |
| --- | --- | --- |
| MariaDB 11.8 and 12.3 | `SET mhnsw_ef_search` | `harness/drivers/mysql_family.py:MARIADB_DIALECT` |
| AliSQL | `SET vidx_hnsw_ef_search` | `harness/drivers/mysql_family.py:ALISQL_DIALECT` |
| pgvector | `SET hnsw.ef_search` | `harness/drivers/postgres.py:set_ef_search` |
| Valkey | `EF_RUNTIME`, a modifier inside the query string | `harness/drivers/valkey.py:set_ef_search` |
| Percona Search | `numCandidates`, a per-query argument | `harness/drivers/mongo.py:set_ef_search` |
| MyVector | `ef_search`, an option inside each `MYVECTOR_IS_ANN` call | `harness/drivers/myvector.py:set_ef_search` |

Three of those six do not search the number they are handed. Valkey sends
`EF_RUNTIME max(ef, k)` and Percona Search sends `max(numCandidates, k)`,
because neither can return `k` rows out of fewer than `k` candidates, and
MyVector's hnswlib searches at `std::max(ef_, k)` inside `searchKnn`. At any
`k` above the grid's floor the low end of a shared grid therefore collapses: at
`k = 100` the six points 10, 16, 24, 32, 48 and 64 are one search measured six
times (`harness/drivers/mongo.py:effective_ef`). That much is read out of the
drivers and needs no server to check, and on its own it settles the weaker half
of the claim — a shared literal is demonstrably not a shared amount of work.
The harness asks each driver what it will really search and records the answer
as `extra.effective_ef`, so a reader comparing two engines at "ef 10" can see
which of them looked at ten candidates. Only Percona Search and MyVector override the
default (`harness/drivers/base.py:effective_ef`), so **Valkey's clamp is
applied to its queries and is not reflected in its `effective_ef`** — that is a
known gap in the record rather than a claim about Valkey.

The stronger half of the claim — *how far apart* the recalls at one `ef_search`
actually are — is **supplied**, and nothing here measures it. The figure quoted
throughout the code and the report is two engines at recall 0.98 and 0.80 at
`ef_search = 100`, up to 0.18 of recall apart
(`harness/workloads/calibrate.py`, module docstring). It comes from the
tuned-profile notes the project specification cites. **No run this repository
holds contains an `ef_calibration` record at all** — the phase is newer than
every archive (`tests/test_c2_webui_phases.py`) — so nothing here can confirm
or refute that spread, and no test tries to.

#### What the sweep does

`harness/workloads/calibrate.py` runs once per measurement unit, on the index
the build phase has just finished, before any of the three workloads below.

- The grid is twelve requested `ef_search` values — `10, 16, 24, 32, 48, 64,
  100, 160, 256, 400, 640, 1000`, declared as
  `harness/workloads/calibrate.py:DEFAULT_EF_GRID` — walked ascending and
  de-duplicated. A descending grid swept in the order given would stop at its
  two largest values and calibrate every engine to the top of it, which is the
  opposite of what the sweep is for. The profile key that replaces the grid is
  `ops.ef_calibration` (`orchestrator/ops_pass.py:DEFAULT_EF_CALIBRATION`); the
  report's own footnote calls it `ops.ef_grid`, which is not a key the
  orchestrator reads.
- At each point: set the knob, ask the driver what it will really search, check
  once that the vector index is in the query plan, discard 20 warm-up queries,
  then time and score `ops.max_queries` queries — 1,000 by default — at a
  single client. The plan check is per point and after the knob is set, because
  a full-table scan returns recall 1.0 at the smallest `ef_search` and is
  otherwise indistinguishable from an engine whose floor is above the target.
- Each point becomes one record of its own, `phase: ef_calibration`
  (`harness/metrics/records.py:PHASE_CALIBRATION`), carrying
  `extra.effective_ef`, `extra.target_recall`, `extra.reached_target`,
  `extra.grid_position` and `extra.grid_size` — so a sweep that stopped after
  three of twelve points says so in the records and not only in the log. Each
  also carries the bootstrap intervals of §2.6.
- Ground truth is recomputed over the row count the engine was actually given,
  because a profile may have loaded a subset of the training set.
- The sweep stops as soon as two consecutive points reach the target. Every
  larger `ef_search` costs query time and changes nothing about the choice.

The query set is deliberately the one the later workloads use. This sweep's
output is an `ef_search` and a recall rather than a speed, so a warm cache here
is not a contaminated measurement there — and sharing the queries means the
recall the choice was made on is the recall those workloads will report.

#### The selection rule

The target is `recall@k >= 0.95` by default
(`harness/workloads/calibrate.py:DEFAULT_TARGET_RECALL`), settable per profile
as `ops.target_recall`. **No profile shipped in this repository sets
`ops.ef_mode`, `ops.target_recall` or `ops.ef_calibration`**, so every profile
here runs the default grid against the default target in `calibrated` mode.

Given the measured curve, `harness/workloads/calibrate.py:_choose` returns the
first case that applies:

1. The smallest grid point's recall is more than 0.02 above the target
   (`FLOOR_MARGIN`) — the smallest `ef_search` is used and the status is
   `floor_above_target`. The margin is not zero on purpose: an engine whose
   first point reads 0.951 against a 0.95 target is calibrated, not
   uncalibratable.
2. The smallest `ef_search` that reaches the target **and whose successor also
   reaches it** — status `ok`. The successor clause is the whole guard. Recall
   over a finite query set is a mean of Bernoulli draws, so a single point can
   clear the target by sampling noise, and without the clause that point
   becomes the `ef_search` every later workload is measured at.
3. The target is first met at the last grid point, where nothing can confirm it
   — status `ok`, with the result's `confirmed` flag false. **`confirmed` is
   not a fourth status.** The clause could not be applied because the grid ran
   out, not because the check failed, and the run log prints `unconfirmed`
   beside the status rather than qualifying the comparison.
4. Nothing reached the target — the largest grid point is used and the status
   is `target_unreached`.

"Reaches" is `recall_at_k >= target`, inclusive; the floor test in case 1 is a
strict `>` against `target + 0.02`.

The chosen value then replaces `ef_search` for everything below, and `ef_mode`,
`target_recall` and `calibration_status` are stamped on every record written
afterwards (`harness/workloads/calibrate.py:status_extra`). That stamp is what
the report joins on: the chosen `ef_search` and the status are read back off the
records of the workloads that *used* them and never re-derived by applying the
selection rule a second time (`report/loaders.py:operating_points`). A second
copy of the rule could drift from the first, and the symptom would be a report
whose "chosen ef" column names an `ef_search` no workload ran at.

#### The two statuses that qualify everything below them

`ok` is the only status under which an engine was held to the target. The other
two say it was not, in opposite directions, and each has its own footnote mark.

**`floor_above_target` — the engine cannot be slowed down to the target.** Its
recall at minimum search effort is already more than 0.02 above it, so no grid
point brings it down, and its numbers in §2.3, §2.4 and §2.5 were bought at a
**higher** recall than the engines marked `ok`. That direction understates its
throughput against theirs rather than flattering it. What an operator should
conclude depends on the engine. For one with an `ef_construction` knob it means
`ops.ef_calibration` does not start low enough to reach the target from below,
and the grid is the thing to change. For MHNSW and VIDX at `M = 16` the
expectation is that this is a property of the implementation: `M` is the only
build knob either exposes (§4), so there is nothing left to turn down. **That
expectation is supplied, not established here.** It reaches this repository as
a line of the project specification's rollout step 2, quoted in
`docs/10-rollout.md` §2 — the operating-point table "should show
`floor_above_target` or a low chosen `ef`" for those two. The status names in
that sentence are real and checked against the code; which engines land on them
is what the sweep measures, and anything else is a finding rather than a
configuration slip.

**`target_unreached` — the engine reached the target at no grid point.** It is
reported at the best recall it achieved, at the largest `ef_search` in the
grid, which the sweep is guaranteed to have measured because early stop needs
two consecutive points at the target and this curve had none. Its numbers below
were bought at a **lower** recall than the `ok` engines', so they will tend to
look better than they are. The remedy is to extend `ops.ef_calibration`
upwards and re-run, at the cost of query time.

A third mark exists and is not a sweep outcome: an engine with concurrency,
filtered or churn records and **no** `calibration_status` was never calibrated
at all. Runs are checkpointed per `(pass, engine, dataset, phase)`, so flipping
`ops.ef_mode` to `calibrated` and resuming keeps the fixed-`ef` records of every
engine already measured, and nothing refuses the mixture because the ops
measurement version (§5) did not change. Such engines have throughput in the
sections below, no row in the operating-point table, and no recall at which
they are comparable with anything.

#### What a dagger means

In the operating-point table's Status column and in the concurrency chart's
legend:

| Mark | Status | How to read the throughput below |
| --- | --- | --- |
| `†` | `floor_above_target` | measured at a **higher** recall than the `ok` engines |
| `‡` | `target_unreached` | measured at a **lower** recall than the `ok` engines |
| `*` | no status at all | never held to any target; no recall at which it is comparable |

One glyph for one meaning. The chart imports all three marks from the renderer
rather than spelling them again (`report/charts.py:FLOOR_MARK`), so a mark on a
chart cannot be explained by a footnote under a different glyph in a table.
Each footnote is printed only when the run earned it, and each names the
engines, the `M`, the resource pass, the recall and the `ef_search` concerned.
Further notes cover rows that carry no status at all: two different chosen
`ef_search` values or two different statuses under one measurement unit, which
only a hand-merged records file can produce; a unit whose downstream workloads
never ran, so the sweep's cost was paid and no comparison came of it; and a
unit whose sweep records are absent, so the ef and the status are trustworthy
and the evidence for the choice cannot be shown.

**The same dagger appears in §2.1's headline table for a different object.**
There it marks a QPS-at-a-recall-floor cell for which every swept configuration
returned recall above that floor, so the figure is the engine's throughput at
the higher recall printed beside it. The two uses are the same kind of caveat —
a figure taken above the recall it is being compared at — about two different
sweeps, and neither footnote covers the other's table.

#### What §2.3, §2.4 and §2.5 inherit from this

All three are measured at the single `ef_search` chosen here
(`harness/workloads/calibrate.py:CALIBRATED_WORKLOADS`), so **every number in
them is conditional on this section's Status column.**

- **§2.3 Concurrency.** Its QPS curves and latency percentiles are at this
  `ef_search` at every client count. Its `recall_at_k` is **not** the recall in
  the operating-point table: that one is the sweep's, single-client against warm
  caches over `ops.max_queries` queries, and §2.3 defines the other. They are
  not expected to be equal and neither is a check on the other. §2.3 also
  describes what the concurrency figures can and cannot be read as on their
  own, including the one limit `extra.client_bound` does not catch; that is a
  separate question from this one and is not restated here.
- **§2.4 Filtered search.** Measured at the *unfiltered* calibrated
  `ef_search`, not at one re-calibrated per selectivity. Re-calibrating per
  point would hide exactly what that section exists to show — that a scalar
  predicate costs recall or speed — because the `ef_search` would be raised
  until it did not.
- **§2.5 Churn.** Baseline and post-churn recall and QPS are both at this
  `ef_search`, which is what makes the degradation a property of the churned
  graph rather than of two different search efforts.

With `ops.ef_mode: fixed` none of this happens: the profile's literal
`ef_search` is used, no record carries a status, and in place of the table the
report says that the engines below are compared at equal *search effort* rather
than at equal *answer quality*. Nothing is wrong with such a run — it answers a
different question, and the report declines to imply otherwise.

#### What this section does not establish

No claim here is a measurement of an engine. No server can be started from this
checkout, no run the repository holds contains an `ef_calibration` record, and
every statement above about the shape of a real curve — the 0.98/0.80 spread,
the 0.18, MHNSW's and VIDX's floors at `M = 16` — is supplied by the project
specification and the tuned-profile notes it cites. What *is* established from
this repository is the mechanism: five distinct knobs, two clamps of which one
is recorded, the grid, the selection rule, the three statuses and the marks the
report renders them with. The first run of this sweep against live servers is
what would turn the supplied half into a measurement; until then the Status
column is the only honest way into §2.3, §2.4 and §2.5.

### 2.1 Recall vs QPS (ann-benchmarks)

The standard approximate-nearest-neighbour quality/speed tradeoff.

- **Recall@k** — for each of the dataset's held-out test vectors, the fraction of
  the true top-`k` neighbours that appear in the engine's returned top-`k`.
  Ground truth ships with the HDF5 dataset; it is exact, computed by brute force.
- **QPS** — queries per second, single client, best of `--runs` repetitions.
  Every profile a report is published from (`publishable: true` —
  `tuned-complete`, `main`, `full`) sets `runs` to 2 or more, and
  `tests/test_config.py` refuses one that does not.
- **Warm-up** — before the first configuration on each built index, the
  overlay module queries that index with a fixed, seeded sample of
  **training** vectors (never the test set, which would pre-cache the paths
  about to be timed) in batches of 200, until two consecutive batches agree on
  mean latency within 5%, capped at 5,000 queries
  (`overlay/ann-benchmarks/ann_benchmarks/vb_warmup.py`, identical for every
  engine). The count, duration and whether it converged are written into each
  result file as `index_warmup_queries`, `index_warmup_seconds` and
  `index_warmup_converged`. Thirty further random-vector queries still run
  before every configuration. Without the one-time warm-up the first
  `ef_search` point measured a cold graph: full-20260825 published ef_search
  10 slower than 20 on every MySQL-family curve. On a warmed index the report
  labels a first-step throughput inversion **unexplained**, not "warm-up".
  Introduced as `ANN_MEASUREMENT_VERSION` 3, which moved every results tree.
- Sweep: each point is one build (graph degree `M`, and wherever the engine
  exposes them `ef_construction`, quantization, storage engine and build mode)
  queried at one `ef_search`. The upper-left envelope of every point an engine
  produced in one dataset and resource pass is its **Pareto frontier** — that
  is what gets plotted and compared.

Higher and further right is better. An engine is faster than another *only* if
its frontier is above the other's at the same recall.

**A frontier is only as wide as the space it searched.** Where a profile swept
one build for an engine and varied only `ef_search`, the line is that build's
recall–throughput curve, not a Pareto frontier, and the report says so: the
§4 opening sentence, the swept-space table ("curve at `M=16` — build space not
explored") and the chart legend ("— curve at M=16") all switch wording for
that engine. The swept-space table lists, per engine, every build axis that
varied with its values, the axes that were fixed, and the `ef_search` range.

`k = 10` by default (`--count`). Recall at k=100 is also recorded.

**The headline QPS-at-recall table is the frontier over every build swept.**
Engines are compared at a recall, not at a setting: `M=16` names different work
in MHNSW, VIDX, pgvector and hnswlib, so a table at matched parameter names
compares the names. Each floor figure is the fastest point at or above the
floor across every build swept in that dataset and resource pass (two passes
are never pooled into one frontier), and the build that produced it is printed
under the number. Where two builds give exactly the same throughput, the
smaller build (by index size, then `M`, then `ef_construction`) is named, so
the result does not depend on the order the result files were read in. The
floor chart draws the same figures, with value, mark and build printed upright
above each bar. The recall–throughput charts draw one line per engine and do
not name builds on it; the build-cost table's per-build recall ranges say which
build supplies which stretch of a line.

**Every headline figure says whether it is the build the ops pass measured.**
Each engine declares its reference build in `config/engines/<e>.yml`
(`runtime.reference_build`) as the index the ops pass builds — M=16 for the
MySQL family, `ef_construction` as the ops pass sets it for pgvector, Valkey,
Percona Server, MyVector and VillageSQL, and the pass's quantization for
Percona Search.
`orchestrator/reference_build.py` resolves it per profile and pass, `run`
refuses one the ann grid does not contain, and the manifest carries it to the
report in `engines.<name>.presentation.reference_builds`. An archived run whose
manifest predates that falls back to the ops pass's own `index_build` record.
A headline figure from any other build carries a **◇** and a note naming the
ops build, because §2.2–§2.5 (build cost, concurrency, filtered, churn)
describe the ops build, not the one that won the floor.

The answer is "cannot be said", never "different", in three cases, and the
report lists the engine with the reason instead of marking anything: no ops
build is recorded; the ops build left a setting to the server default, which
no record names (Valkey's `ef_construction` on 2026-08-25); or the ops build
matches none of the swept builds. The ops harness stamps `build_mode: post` and
`storage_engine: InnoDB` on every unit, including engines with neither knob,
so those two axes are compared only where the ann grid actually varied them
(MariaDB's storage engines, pgvector's build modes). Comparing them everywhere
made the 2026-08-25 mongot ops record (`post`, `InnoDB`) match none of its
scalar points (`async`, `wiredTiger`).

A secondary table, "at the build the ops pass measured", keeps the same floors
restricted to that reference build. It is the one view in which §4 and the ops
sections describe the same index. A floor the reference build never reaches
reads "—" there, with that build's highest recall in a note. The run of
2026-08-25 is why every figure names its build: its published table gave
Percona Search 138 / 124 / 88 QPS at 0.90 / 0.95 / 0.99 from the unquantized
sweep with nothing saying so, while every other Percona Search figure came from
the scalar index, which peaks at recall 0.892. The headline now shows those
figures again, labelled `quantization=none` and marked ◇.

**Build cost on the frontier.** For every build that contributes a frontier
point the report prints its time to a queryable index and its index size, each
labelled with where it came from. The time is ann-benchmarks' `build_time`
("ann fit"), the wall clock of `fit()`, which loads the corpus and builds the
index (for MHNSW and VIDX those are one operation). Where the ann file has none
— true of every build on 2026-08-25 — it is the ops pass's time to a queryable
index for the same build, marked "(ops)": ingest plus `CREATE INDEX` for a bulk
build, the load alone for a graph built on insert, never the bare
`CREATE INDEX`, which leaves the load out. The size is the server-side figure
each overlay module measures and writes as `index_bytes`: the index's files
under the datadir for the MySQL family, `pg_relation_size` for pgvector,
mongot's index for Percona Search, the hnswlib index files for MyVector, and
resident `used_memory` growth for Valkey (labelled "resident"). ann-benchmarks'
own `index_size` attribute is never used: upstream computes it from the
benchmark client's RSS, which for a client/server engine is the size of the
Python process, not the index. Where the ann file has no size (or a size of 0,
which is what the MySQL-family module writes when its file glob matched
nothing) the ops record for the same build is used if exactly one matches,
marked "(ops)"; otherwise the cell says "not measured". An ops axis left to the
server default matches no ann build, so it supplies nothing.

**What an ann index size can and cannot tell apart.** On 2026-08-25 pgvector's
three `ef_construction` builds all report 8,110,088,192 bytes, AliSQL's M=6 and
M=16 both 4,156,555,264, and Valkey's efC 64 and 200 both 6,086,322,000. Some of
that may be real (pgvector's page count depends little on `ef_construction`),
but a byte-identical figure across different graphs is at least as likely to be
one reading per run as one per build, and nothing in the result file says
which. The build-cost table marks such a size with a trailing **=**, and the
smallest-build view below never treats equal sizes as a ranking.

**Smallest build that reaches each floor.** Beside the fastest build, the
report names, per floor, the smallest build whose points reach it and that
build's best QPS there. Each cell's rule is stated: nothing is ranked when one
build reaches the floor; otherwise by measured index size when every build
reaching it has one and the smallest is unique; "(=size)" when the smallest
size is shared, and `M` then `ef_construction` decided among those; by lowest
`M` then lowest `ef_construction` when a size is missing; and "(tie)" when no
size was measured and several builds share the lowest `M` and
`ef_construction` (Percona Search's quantizations, for one), where the fastest
of the tie is shown rather than a pick presented as a ranking. The fastest
build at a floor is very often the largest graph swept, and this view keeps the
headline from reading as an argument for it.

#### Which build the ops pass measures: `ops.build_selection`

Equal M is not equal work. M names the graph degree in every engine here, but
what one unit of it costs and buys differs by implementation, so "every engine
at M=16" hands each a different budget under one label. The comparison that
means something is each engine's Pareto frontier over its whole swept grid, and
once the headline is that frontier the ops sections — build cost, concurrency,
filtered, churn — have to describe the same index, or the report puts a
frontier figure from one build beside throughput measured on another.

`ops.build_selection` decides which build the ops pass measures:

- **`reference`** (the default, and what every profile did before the key
  existed): the declared reference build above, built at every `ops.m_values`
  entry.
- **`frontier`** (`tuned-complete`): after the ann phase of each
  (engine, dataset, pass), and from **that phase's own result files** — the
  ones it wrote or reused, as recorded in the manifest, so a stale file from
  another grid cannot be chosen — the ops pass picks the
  (M, `ef_construction`, quantization) whose frontier point at the ops
  calibration target (`ops.target_recall`, 0.95 by default, the same value
  §2.0's calibration then chases) has the highest QPS. Only points of the ops
  pass's **primary** storage engine and build mode compete — the head of the
  lists it measures (InnoDB for MariaDB unless the profile reorders them; the
  first of `pgvector_build_modes`) — because the chosen M is then built for
  every declared storage engine and build mode, and a MyISAM or incremental
  curve must not choose the M InnoDB or `post` is measured at (an axis the
  grid did not sweep is not filtered; see below). Points that fell back to a
  table scan are excluded. Ties go to the smaller index, then the smaller M,
  then the smaller `ef_construction`, then the build's name, so the choice is
  the same on every re-read. The ops
  pass then builds exactly that M, `ef_construction` and quantization, in
  every declared storage engine and build mode; the record says which curve
  the ranking was done on (`selected_on`).

Nothing substitutes silently. If no build reaches the target, the build with
the highest recall is measured and the status is **`below_target`**; if the ann
phase did not run in this run directory, failed, or left no usable point —
or the selection container could not read it (no bench image, a non-zero
exit, unparsable output, a timeout, a docker CLI that would not start; the
container is removed) — the declared reference build is measured and the
status is **`fallback_reference`**. Both print a warning and land in the
manifest's `warnings`.

The choice is stored in the manifest
(`ops_build_selection["<pass>/<engine>/<dataset>"]`) when it is made, with
the identity of the ann attempt it came from (attempt number, start time, a
digest of its file list). A resumed run reuses a `selected` or `below_target`
choice only while that attempt is still the latest completed ann attempt of
the unit; after an ann re-run (`--force`, a changed grid, a retried failure)
it chooses again. A `fallback_reference` is never reused once a completed ann
attempt exists. Two cases are refused rather than measured: a choice, stored
or fresh, naming a build outside the grid the profile renders now, and a
re-selection that would change the build after ops units of that engine and
dataset were already measured on the old one — their records stay in the run
directory, and the rest would be measured on another index under one name.

The manifest's `engines.<name>.presentation.reference_builds_by_dataset`
carries the chosen build and its source (`frontier@0.95`, `below_target@0.95`,
`fallback_reference`) to the report. It becomes the ops build the headline's ◇
is judged against and the build of the "at the build the ops pass measured"
table, and a line under that table says how each engine's build was chosen.
The choice breaks an exact QPS tie with the headline's own key (index size,
then M, then `ef_construction`, then the build's name), so where both see the
same points they name the same build, and the headline's 0.95 figure is
normally the ops build and carries no ◇. Under `frontier` a ◇ can still appear
at 0.95 in exactly these cases: the fastest point at 0.95 is on a storage
engine or build mode other than the ops pass's primary one (only possible
where the grid swept that axis); `ops.target_recall` is not 0.95, so the build
was chosen at another recall; the status is `below_target` or
`fallback_reference`; or the ann points the report reads are not the ones the
selection read (a report over a wider ann tree than the phase's own files).
A storage engine or build mode is filtered at all only where the ann grid
swept more than one value of it — every ann module stamps both axes, and
Percona Search's `async`/`wiredTiger` are not the ops pass's words. Every ops
record carries the same facts in
`extra.ops_build_selection`. With a `select-builds` overlay the grid the
choice must lie in is the overlay's restricted one. Ops
measurement version 10 marks the change: a version 9 and a version 10 figure
for one engine may describe different indexes, and the report refuses to chart
them together.

### 2.2 Index build cost (ops harness)

Measured per `(engine, dataset, M)`:

- **Ingest throughput** — rows/sec loading the training vectors, excluding index
  build where the engine allows separating them.
- **Index build wall time** and **CPU time**.
- **Peak RSS during build** — read from the container's cgroup v2
  `memory.peak` (falls back to sampling `memory.current` on cgroup v1).
- **Index size on disk** — the on-disk size of index structures only, measured
  per engine:
  - MariaDB: the `#i#` auxiliary index files / tablespace for the vector index.
  - AliSQL: the InnoDB auxiliary table holding the HNSW graph.
  - pgvector: `pg_relation_size()` of the HNSW index relation.

These are not directly comparable to the byte, because each engine stores graph
nodes differently. They are comparable in *order of magnitude* and in how they
scale with `M`, which is the useful signal.

**`build_wall_s` is two different quantities and must never be compared across
engines on its own.** For an engine that bulk-builds — pgvector, Valkey — it is
the `CREATE INDEX` call alone, after a plain load. For MariaDB MHNSW and AliSQL
VIDX there is no separable build step at all: the graph is maintained on every
INSERT, so the honest build cost *is* the load cost and that is what the field
holds. `extra.separable_build` says which, and the report's `Index build`
column renders it.

So the figure that compares like with like is **time to a queryable index**:
ingest + build for a bulk engine, the single figure for an incremental one.
Adding ingest to an incremental record would double-count the load, which at
1M rows is hours rather than a rounding error. The build table and the build
chart both carry that column, and the chart keeps the raw bulk build step on
an axis of its own where no number includes a corpus load. That preserves both
of the claims the two fields exist to support — "this engine's bulk build is
N× cheaper than its own incremental build", and "this engine's incremental
build costs X against that engine's Y" — instead of collapsing them into one
misleading one.

A record written before `extra.separable_build` existed appears on **neither**
of those axes, and the chart says how many it left off. Absence has no honest
reading here: archived runs contain both kinds of engine, so filing an unmarked
record as either would be inventing the fact the field exists to record. (There
is a diagnostic — an incremental record's `build_wall_s` equals its
`ingest_wall_s` exactly, because both are the load time — but that is evidence,
not a record, and the report does not reclassify on it.)

### 2.3 Concurrency scaling (ops harness)

QPS and latency percentiles (p50/p95/p99) at 1, 4, 8, 16, 32 concurrent
clients, at a fixed `ef_search` chosen to put all engines at comparable recall.

This is the dimension that separates a database from an ANN library: it exercises
the engines' graph caches, buffer pools and locking. MariaDB caches the MHNSW
graph per `TABLE_SHARE`; AliSQL keeps a shared cache plus a per-transaction
cache; pgvector reads graph pages through `shared_buffers`. Those designs behave
differently under concurrency and that difference is the point.

Reported as: absolute QPS curve, and **scaling efficiency** = `QPS(n) / (n × QPS(1))`.

**Only the queries that finished inside the window are counted.** The window is
a closed interval: it opens at the instant the clients' barrier releases and
closes at the instant recorded immediately *before* the stop flag is set — not
when a client later notices the flag, which is the whole point. QPS is the
queries completing inside that interval divided by its length. Every client
finishes the query it is holding when the window closes, so counting those put
one query per client in the numerator that the denominator never covered.

The relative error is that one query over the ones a client did finish:
`1 ÷ queries-completed-per-client`, which in record fields is
`extra.inflight_discarded ÷ queries_executed` and is about **per-query latency ÷
window length**. The client count cancels — each client contributes one spurious
query and roughly `window ÷ latency` real ones — so it is the same at 1 client as
at 32, measured flat at 2.564% across 1, 2, 4, 8, 16 and 32 clients over a
two-second window of 50 ms queries. What it depends on is a **slow engine over a
short window**: 0.005% for a 1.4 ms query in a 30-second window, 2.5% for a
50 ms query in two seconds, whatever the concurrency. The uncorrected figure also
reported more queries in flight on average than there were clients to hold them,
which no client count can do. `extra.inflight_discarded` counts the queries
dropped at the close; `extra.prewindow_discarded` counts those dropped at the
open, and is zero by construction because the window opens at the barrier release
and no measured query can precede it, so **a non-zero value there means the
window's two ends disagree about when it began**. Recall is scored over the same
trimmed set, so `extra.recall_queries` and `queries_executed` always describe the
same queries. A window in which nothing finished at all — a configuration whose
queries outlast `ops.concurrency_duration_s` — records a QPS of 0, no latency
figures, and says so on stderr while it runs.

**Recall is recorded beside the throughput**, as `recall_at_k` on every
concurrency record. The sentence above — "a fixed `ef_search` chosen to put all
engines at comparable recall" — is an intention, and until the ids were kept
nothing in the run checked it. Two engines at `ef_search=100` can sit at recall
0.98 and 0.80, which is the ordinary spread of these implementations at a shared
ef rather than a pathology, and a throughput curve with no quality figure beside
it presents them as comparable while one is doing materially less work per
query. Each client records the ids from the first pass over its own shard — one
result list per query however long the window runs — and those are scored against
the same ground truth every other phase uses, recomputed over the loaded row
count when a profile loads a subset. `extra.recall_queries` says how many
distinct queries the figure covers; a window too short for one pass reports the
coverage it managed rather than padding the rest into a recall collapse. With
`concurrency_repeats > 1` the recall is the median window's, and
`extra.recall_spread` carries the difference between windows: recall is a
property of the index and the ef, so a spread above 0.005 means the mapping or
the index moved and the run says so on stderr.

**The query pool is its own profile key.** `ops.concurrency_queries` defaults to
`0`, meaning the whole test set — 10,000 queries for glove-100 and sift-128.
It used to be `ops.max_queries`, which exists to bound the *filtered* and *churn*
workloads: those answer each query once to completion, and `main.yml` holds the
key at 500 for exactly that reason. The concurrency workload instead cycles its
shard for a fixed duration, so at 32 clients each client was repeating about
fifteen queries for thirty seconds and every graph cache in the comparison was
measured against a hot set small enough to hold entirely — erasing the
difference the chart exists to show. `extra.query_pool` and
`extra.query_reuse` (executed ÷ distinct) record what the caches were actually
shown. Setting the key is the way back to a small pool for reproducing an
earlier run.

Each client's shard is `queries[c::clients]`, a strided slice rather than a
contiguous block, so no two clients hold adjacent queries and part of the window
is not one client warming a cache for another. The shards are disjoint and
together cover the pool exactly.

**Whose ceiling a point shows, and the flag that tries to say.** The clients are
forked OS processes sharing one container's cpuset, so a concurrency figure
can be limited by the client rather than by the engine — and a chart that does
not distinguish those publishes the harness as a database. Every concurrency
record therefore carries the three numbers the judgement rests on rather than
only the quotient: `extra.client_cpu_seconds` from the client container's own
cgroup counter, `extra.client_cpus` for the CPUs it was given, and
`extra.client_cpu_util` for the share those imply. All three, because each can
be wrong while the quotient stays plausibly between 0 and 1.
`extra.client_bound` is set when the utilisation exceeds
`clients.CLIENT_BOUND_UTIL`, which is **0.80**, and the report draws those
points hollow and names them in a footnote. That threshold is not a measured
constant and is not offered as one: it is the level at which this harness stops
being able to say whose ceiling a figure shows. A client-bound point never sets
an engine's peak in the report's peak-throughput table, and neither does a unit
whose ef calibration ended `target_unreached` or `floor_above_target`: those
are listed with their status and not ranked.

**Sizing the client set.** `resources.cpu.client_cpus: auto` gives the clients
one CPU per two clients of the profile's largest `ops.client_counts`, never
fewer than four (16 for a 1–32 sweep), taken from the topology pool after the
server cpuset. When the pool cannot supply it, `run` refuses before anything
loads and names both numbers; a numeric pin keeps its old behaviour of warning
and sharing cores. The run's estimate prints the resolved size beside the
largest client count.

**The flag is blind to a ceiling that costs no CPU, and that is measured rather
than suspected. The case below is the one it was measured on, and §7 has since
removed that case from the harness — read the two together.** A client can
limit throughput *without* burning CPU. When the clients were threads in one
interpreter, a driver that held the interpreter lock while it waited serialised
them, so the pool delivered one query at a time while every thread sat idle. On
a `smoke` run MariaDB measured 2,920.7 QPS at one client and 2,927.5 at four
with every latency percentile unchanged, `scaling_efficiency` 0.25, and
`qps × p50` ≈ 0.95 — about one query in flight with four clients running. Run
through the then-current `ClientPool` with two synthetic drivers of identical
per-query cost differing only in whether they release the lock, the releasing
one scaled 1.02 (2,487 → 10,186 QPS at four clients) and the holding one 0.26,
and the holding one's `client_cpu_util` was **0.177** against the 0.80
threshold. The pool delivered a quarter of its throughput and `client_bound`
stayed `False`. (Those run figures are measurements of a live `smoke` run and
of that reproduction, not of this document.) The clients are processes now and
Python's own lock can no longer do this; the flag's blindness is structural and
is kept for the cases that remain, which §7 lists.

**What that 0.177 is a measurement OF, which this section stated too
loosely.** `read_own_cpu_usec` opens `/sys/fs/cgroup/cpu.stat`. In a real run
that is the harness container's own cgroup, which is the quantity
`client_cpu_util` claims. On a bare host it is the **root** cgroup — the whole
machine: measured on a developer box, it reported **2.421 cores** burned while
the reading process slept. So the quotient describes the clients only when the
harness is in its own container, and the 0.177 is a figure from such a run
rather than something a checkout can reproduce. **This is documented one file
away and is easy to walk past**, which is how it was walked past here: the
docstring of `tests/test_harness.py`'s
`test_a_known_number_of_saturated_cores_reads_back_as_that_many` already says
`/sys/fs/cgroup` is the root cgroup on a host, "so the counter also sees the
rest of the machine: a sibling worktree running its own suite lands in the
numerator" — which is why that test **retries rather than loosens** its
ceiling. Read that docstring before reaching for this counter; the helper
beside it is the one everybody reaches for first.

**None of this weakens the point above.** A CPU-utilisation threshold cannot
see a lock stall whatever the counter's scope, because the lock prevents the
very consumption the threshold looks for. A wider scope makes the reading too
HIGH, not too low, so it cannot explain a 0.177 away.

So `client_bound` is **sufficient and not necessary**: a point it marks was
client-limited, and a point it does not mark is not thereby evidence that the
engine was the limit. The signature of the case it misses is **flat QPS with
flat latency** — throughput that stops rising while the percentiles do not
move, which no saturating engine produces, because an engine at its limit
makes queries wait and that shows in p95 and p99. `scaling_efficiency` beside
it is the other half of the reading. **That instrument has landed** — §7 has
it: the harness now measures from the query spans how many queries a window
had in service, refuses to publish a scaling efficiency where that is far
below the client count, and carries the measurement showing the obvious
formulation of it, summing the spans, is blind to this very case.

### 2.4 Filtered / hybrid search (ops harness)

Vector search with a scalar `WHERE` predicate, at three selectivities (~1%, ~10%,
~50% of rows passing) on **two different predicates**. Six points per engine,
dataset and index configuration.

#### The two filter modes

The two columns exist because one predicate measures half the problem. Both are
loaded in the same build, so both are measured against the same graph.

| Column | Definition | Relationship to the vector space |
| --- | --- | --- |
| `tag` | `row_id % 100` | **Independent.** The qualifying rows are a uniform sample, so a query's true neighbours pass the filter about as often as anything else does. |
| `ctag` | nearest-pivot cluster id (100 pivots, fixed seed) | **Correlated.** `ctag < t` selects `t` whole clusters, so for most queries the real nearest neighbours sit in a cluster that fails. |

`tag` is the control case and `ctag` is the treatment. An engine that
pre-filters sees little difference between them: it compares only the
qualifying vectors either way. An engine that post-filters — walk the graph,
then discard what fails — sees a large one, because under `ctag` the
candidates it walks are exactly the rows the predicate rejects. This is the
case a single uncorrelated column cannot distinguish, and it is the realistic
one: a filter in production usually correlates with the embedding (a category,
a tenant, a language), it rarely correlates with a row id.

**The selectivity recorded is the one that was delivered, not the one that was
asked for.** Cluster sizes are unequal, so a nominal 1% on `ctag` is never 1%:
measured, it delivers 1.90% on the 2,000-row test fixture and between 1.00%
and 4.96% on a 100,000-row corpus depending only on the clustering seed. The
`selectivity` field is a counted share of rows and `extra.nominal_selectivity`
is the request beside it. Comparing the two columns means comparing them at
their measured shares; comparing at the nominal share compares two different
predicates and calls the difference an engine property.

#### The near_pass / near_fail split

Within one column and one selectivity, queries divide into two groups by
whether **the query's true unfiltered nearest neighbour passes the filter**:

* `near_pass` — it passes. The best answer is inside the candidate list the
  engine was going to walk anyway, so the filter costs little.
* `near_fail` — it fails. The engine has to keep looking, and this is where
  post-filtering collapses: it exhausts `ef_search` candidates without finding
  `k` that qualify, and returns a short result set or a poor one.

Recall is reported for each group (`extra.recall_near_pass`,
`extra.recall_near_fail`) with the query counts beside them. For `tag` the two
groups track the selectivity and their recalls stay close. For `ctag` they
separate: on a clustered 2,000-row corpus at a 10% predicate the two groups
measured 0.983 and 0.241 recall for `ctag`, against 0.625 and 0.641 for `tag`
— the same split applied to both columns, separating them by a factor of four
on one and not at all on the other.

A dash in the report means the group was empty, which is a real state rather
than a missing measurement: `tag` at 1% puts zero of fifty queries in
`near_pass`, because a uniform 1% of rows contains a given query's nearest
neighbour about 1% of the time. The group's recall is recorded as absent, not
as zero.

#### Short and truncated results

The distribution of returned result counts is recorded in
`extra.returned_count_hist`, not only a count of the short ones: an engine
returning three rows for most queries and an engine returning nine are both
"short" and are not the same behaviour.

**Six buckets, and every query lands in exactly one at every `k`:**

| Bucket | Meaning |
| --- | --- |
| `0` | nothing qualified in the candidate list at all |
| `1-4` | a token result |
| `5-9` | most of the way |
| `short_other` | 10 to `k-1`. Only reachable for `k > 10` |
| `k` | the complete answer |
| `over_k` | more than `k` rows, which no correct driver returns |

The last three are why there are six rather than the four an earlier draft of
this section listed. `k` is tested **before** the ranges, so at `k = 5` a
complete answer of five rows is reported as complete rather than filed under
`5-9`. `short_other` exists because at `k = 20` a result of twelve rows
belongs to no other bucket — measured on the test fixture at `k = 20`,
`ef_search = 32` and a 50% predicate, 43 of 50 queries returned between ten
and nineteen rows, and the four-bucket reading would have dropped all 43 and
left a histogram summing to 7 of 50. `over_k` is counted precisely because it
should always be zero: every driver slices its result to `k`, so a non-empty
`over_k` means one stopped and the recall denominator is no longer what the
report says it is.

All six keys are always present, including the zeros. An absent key cannot be
told from a zero by a reader or a chart.

Each point is also **time-bounded**: `ops.filtered_budget_s` caps it (300 s in
`main`), and the query loop stops when the budget is spent. Recall and latency
are then computed over the queries that finished, and the record says so —
`extra.time_bounded`, `queries_executed` against `extra.queries_planned`, and
`extra.insufficient_sample` when fewer than 50 queries completed. The report
greys those points out. The budget exists because this dimension is the one
whose cost cannot be predicted: MariaDB measured 0.9 QPS at 1% selectivity
against 326 unfiltered, a 360x spread, which made 2,000 queries at one point
cost 37 minutes. A truncated point is a measurement of a slow configuration,
not a harness fault, so it is recorded and flagged rather than dropped.

#### Ground truth

Recall here is computed against ground truth **recomputed for the filtered
subset** — filtering changes the correct answer set, so reusing the unfiltered
ground truth would be wrong. The harness computes filtered ground truth by
brute force over the qualifying rows and caches it on nine components —
dataset, metric, `k`, column, threshold, row count, `tag_buckets`,
`ctag_clusters` and `ctag_seed`. Every one of them changes the correct answer
set: `tag` and `ctag` at the same threshold have entirely different qualifying
sets, so a key missing the column would score one filter against the other's
answers, and the last three are present for either column though each applies
to one, because a key that varies with something irrelevant only wastes work
while one that misses something relevant publishes a wrong number.

That brute force is **not** inside `ops.filtered_budget_s`, and it runs over
the whole test set rather than `ops.max_queries`: it is harness CPU spent on
the yardstick, and charging the engine's budget for it would make a cold cache
look like a slow engine.

Measured at **~150 s per million qualifying rows** over 10,000 test queries,
with a 5–8% spread within a run and ~10% drift between sessions on a shared
box. The figure is close to **dimension-independent** at these widths, which
is worth stating because it looks like it should not be: the dominant
`queries @ subset.T` term is linear in the vector width, so 128 dimensions
against 100 would be 1.28× and 960 would be 9.6× if the dot products ruled.
The measured medians are 158.7 s/M at 100 dimensions, 155.0 at 128 (0.98×,
marginally *faster*, inside the spread), 164.7 at 256 and 240.2 at 960 — so
what dominates is the dimension-independent work instead: materialising the
`n_test × n_qualifying` score matrix and partially sorting it. One rate
therefore covers both of `main`'s datasets.

For `main` that is ~3.6 minutes for glove's two-column grid and ~3.1 for
sift's — **~6.7 minutes for the profile**, against the 4.0 h filtered ceiling
it prints, so ~2.8%. Paid once per (dataset, metric, k, column, threshold,
rows) and then shared across every engine, ops unit and resource pass. The run
estimate names it under the costs neither of its figures includes.

Ground truth is also recomputed whenever a profile loads only **part** of the
training set (the smoke profile loads 20,000 of fashion-MNIST's 60,000 rows).
The neighbours shipped in the HDF5 file describe the full corpus, so scoring a
partially-loaded engine against them would point at rows the engine never
received — understating recall for every engine simultaneously, which looks like
a legitimate result rather than a measurement fault. The row count is part of
the ground-truth cache key so a subset run cannot poison a full run's cache.

This is where integrated vector search is supposed to beat standalone vector
stores, and where the engines' query planners diverge most: the SQL engines
post-filter, Percona Search pre-filters, pgvector can iterate its index scan,
and Valkey's planner chooses per query. The report does not take any of those
descriptions on trust — §5's strategy table derives what each engine did from
how its own median latency moves as the predicate narrows, per column, because
the report once asserted for weeks that Valkey chose its strategy per query on
the strength of its documentation while its own numbers said otherwise.

pgvector's `hnsw.iterative_scan` is set explicitly rather than left at its
default and is recorded on every filtered record. `ops.pgvector_iterative_scan`
declares the modes, the orchestrator passes every one of them that the engine
config recognises, and the workload measures the grid once per mode against one
loaded index — so a run measures each declared mode and every record names
which one produced it. `main` and `full` declare `off` and `relaxed_order`;
`strict_order` is recognised but not declared by any profile.

**The archive predates that, and it is not uniform — do not join it as though it
were.** Of the 153 archived pgvector filtered records, 77 read
`iterative_scan: null` and 76 read `off`. The `null` ones are every run from
before the mode reached the harness at all, plus the normalized half of the
2026-10-14 `smoke` run, whose profile declares no mode; the `off` ones are the
72 of the 2026-10-15 `full` run plus 4 from that same smoke run's *tuned* pass,
which picked the mode up from `config/resources/tuned.yml`'s extras. Until
2026-10-20 only the first declared mode was ever passed, so no archived record
reads `relaxed_order` at all. `null` and `off` are **different groups** to §5's
tables, and should stay so: a `null` record was measured at whatever the
server's default happened to be, which is not a value the run chose or recorded.

The 2026-10-15 run put 49 of its 72 `off` points under 0.6 recall, each with
recall close to the predicate's own selectivity, which is what a post-filter
that does not go back for more candidates looks like.

### 2.5 Churn (ops harness)

Recall and QPS re-measured after deleting and re-inserting 10% and 25% of rows.
HNSW graphs degrade under deletion; how much, and whether the engine offers a
repair path, differs.

Three things about the post-churn state are decided here rather than left to
the engine's own timing, because each of them was previously deciding what got
measured.

**Autovacuum is off for the whole churn phase on pgvector.** `prepare_churn()`
runs `ALTER TABLE ... SET (autovacuum_enabled = false)` once, before the
baseline, and `maintain()` runs `VACUUM (ANALYZE)` and then `RESET`s the
option. Without this, whether a churned index still held its tombstones when
recall was re-measured depended on when the autovacuum daemon last looked at
the table, and nothing in the record said which it had been — two runs of the
same profile could differ by the whole size of the effect being measured, with
the difference read as a property of the engine.

**An asynchronous index is confirmed caught up before re-measuring.**
`wait_index_synced(rows)` runs after the re-insert, and its cost is recorded as
`extra.index_sync_s`. For Valkey it waits on FT.INFO's mutation queue and on
its indexed document count, both of them; for Percona Search it polls
`$vectorSearch` for the newest written document until the index returns it.
The other four engines maintain their index on the write
path and record `0.0`. "The engine is slow and inaccurate" and "we asked before
the index was ready" are the same numbers in a record otherwise.

**`index_sync_s` for Valkey is a wait on both signals.** The shared wait tests
FT.INFO's mutation queue and the indexed document count against the row count
expected, and each caller says which way it expects that count to move: the
delete path waits for it to fall *to* the target, the re-insert for
`num_docs == rows`. Equality rather than "at least", because the queue has been
observed drained while the index reported 481 documents more than the keyspace
held, so a count that is merely large enough can be stale entries. An index
that converges near but not on the target ends the wait when the count stops
moving, reported as off target rather than as a timeout. Every record carries
`extra.index_sync_note` with what the wait observed: the final count, the
expected count, the polls, which condition ended it and which of the two
signals was binding.

**When that note says the count ended off target, the direction names the
mechanism — and both mechanisms push the published figure the same way,
down.** A count *short* of the target means rows that were written are not in
the graph yet, so the queries that follow cannot return them. A count *over*
means the graph still holds entries the keyspace does not; the churn path
scores with `harness/datasets.py:recall_at_k` against truth translated through
the re-insert's id map, which is an id intersection, and a stale id is absent
from that translated set by construction. Every stale id returned therefore
consumes a top-k slot and contributes nothing. Measured on that function with
two of five true neighbours re-inserted, a result of stale ids and a result
missing the two new rows both score **0.6000** where a correct one scores
**1.0000**. So an off-target count understates recall either way and cannot
inflate it, and the direction tells you which repair to look for rather than
which way to read the number. The note reports the gap as `(n short)` or
`(n over)`; `n` is the figure any tolerance would have to be set from, and
until a live run produces one, none is assumed.

**Reading that note after a run.** It is on every churn record:

```bash
jq -r 'select(.phase=="churn" and .extra.index_sync_note)
       | [.extra.index_sync_s, .extra.index_sync_note] | @tsv' \
   results/<run-id>/ops-valkey-*.jsonl
```

The same sentence appears once per churn step in the run log as
`[valkey] index sync: …`, and a wait that gives up also prints what it saw to
stderr. The clause to read is the one after the leading phrase —
`num_docs=N against an expected M (…) after K poll(s), ended because …; the
binding signal was …` — and there are seven things it can say:

| The note says | What it means |
| --- | --- |
| `settled in …` · `(exactly on target)` · binding signal the document count, with K > 1 | The index converges exactly, and the count is what the wait rested on. Roughly `2·(K−1)` seconds of indexing lag now sits inside `index_sync_s`; while this wait compared with `<=` that lag sat inside the recall measurement instead. |
| the same with K = 1 and binding signal `neither: the first reading already agreed` | The index was already at the target when the wait began. Nothing is wrong and nothing is proven: this is what a corpus too small to lag looks like. |
| `settled in …` with binding signal the mutation queue | The queue outlasted the count on that step. Both signals agree; the count cost nothing and added nothing there. |
| `stopped off target after …` with `(n short)` or `(n over)` and a large K | The count stopped `n` documents from the keyspace after being given at least 30 s to continue. `n` is the figure a tolerance would be set from, and the direction names the mechanism (above). |
| `stopped off target after 30.0s` with K at the floor (16 polls) | The count was flat from the first readings. A small poll count beside a large gap is what a stall during the re-insert looks like, not a converged index — check the queue reading in the same sentence, because an empty queue is what makes this exit possible and an empty queue has been wrong before. |
| `the wait gave up …` with `index_sync_s: 900.0` | In fifteen minutes the count neither arrived nor stopped moving. That step's post-churn recall measures an index that was still indexing; it should not be published unqualified. |
| `FT.INFO published no num_docs in this build` | The document half was satisfied unconditionally and the queue was the only guard — the state this wait was changed to get out of. `index_sync_s` then means "the queue was empty", nothing more. |

**The same facts are on the record as fields, not only as prose.** Beside the
note, each churn record carries `extra.index_sync_detail`: `observed_docs`,
`expected_docs`, `queue_size`, `polls`, `docs_expectation`, `ended_by`,
`binding_signal` and `timed_out`. So the table above is a query rather than a
careful read — for instance, every churn step across a run whose index did not
arrive exactly:

```bash
jq -r 'select(.phase=="churn" and .extra.index_sync_detail)
       | select(.extra.index_sync_detail.observed_docs
                != .extra.index_sync_detail.expected_docs)
       | [.engine, .churn_fraction, .extra.index_sync_s,
          .extra.index_sync_detail.observed_docs,
          .extra.index_sync_detail.expected_docs,
          .extra.index_sync_detail.polls,
          .extra.index_sync_detail.ended_by] | @tsv' \
   results/<run-id>/ops-*.jsonl
```

An engine whose driver publishes neither key carries neither, exactly as it
carried no `index_sync_note` before.

**The engines with a repair path are measured twice per churn step, and the
second fraction starts from the post-maintenance state.** Each step writes a
record stamped `extra.maintenance_state = "before"`, and then, if the engine
has a repair path, runs it, times it, and writes a second record stamped
`"after"` carrying `extra.maintenance` and `extra.maintenance_s`. Engines with
no repair path (MariaDB, AliSQL, Valkey, Percona Search) write one record per
step and carry `extra.repair_path = "none"`; pgvector carries
`"vacuum_analyze"`.

**Percona Server belongs in the second list at its pinned version, and a
reader looking for it in either list would until now have found it in
neither.** Its driver declares the `"index-replace"` repair mechanism, an
`ALTER TABLE … DROP KEY vi, ADD VECTOR KEY … ALGORITHM=INPLACE` on the
already-indexed table. Measured on `full-20261015-215006`, at the old pin
(9.7.1-1), it returned in 1–4 ms and changed nothing: identical
`index_bytes` to the byte and identical recall to five decimals in all 20
before/after pairs. It has not been re-tested at the current pin, so
`config/engines/percona.yml` declares `capabilities.vector_index_replace:
false`, the host tells the harness to withdraw the repair path, Percona Server
writes **one** record per step carrying `extra.repair_path = "none"`, and no
`"after"` record exists to be read as a repair. That key is separate from
`capabilities.bulk_vector_index_build`, which is `true` at the current pin and
decides only the ops build mode: re-verifying the bulk build does not restore
the repair. If `vector_index_replace` is ever flipped to `true`, Percona moves
back to the first list with no other change. `OPTIMIZE TABLE` is not used for MHNSW or VIDX: at 1M rows
it rebuilds the whole vector index, which is a separate experiment and not a
churn step.

That the 25% step therefore starts from a *repaired* index on pgvector and from
a degraded one everywhere else is a choice, and the alternative is defensible:
never maintaining between steps would measure cumulative degradation under a
realistic write pattern, which is arguably the more interesting question. It is
written down here because it is not derivable from the numbers. The reason for
this choice is comparability in the other direction — the sequence is "churn,
measure, repair, measure" per step, so every record states which of the two
states it describes, and a reader who wants the un-repaired cumulative curve
reads the `"before"` records across fractions and gets exactly it. Under the
alternative, the repaired state would exist only at the end and pgvector's
repair cost could not be attributed to a fraction at all.

The rendered report now names which of the two states each row describes. It
did not until PR 4 §4.3: nothing in `report/` read `extra.maintenance_state`,
so the churn table printed two rows per fraction whose engine, dataset,
storage and churn columns were identical and whose recall and index size were
not, the charts drew a line through both, and the retention ratio returned
each fraction's pair in sorted order with nothing to say which was which. The
table's `Repair` column, a separate series per state on the degradation chart,
and the exclusion of the repaired measurement from the retention ratio are
that fix. The retention ratio is against the *pre-churn* baseline — a state
before any maintenance — so a repaired measurement over it answers a
different question from the one its axis asks, and `charts.churn` is where
that comparison belongs.

**Reads are measured while the writer re-inserts.** `ops.churn_readers`
readers (4 by default) query the index, unfiltered and at the calibrated
`ef_search`, for the whole of each step's re-insertion. The same pool is first
measured for 30 s before that step's deletes, and the headline figure is the
ratio of the two p99 latencies: what a read costs while the engine is taking
writes. A churn number measured with nothing else touching the index describes
a maintenance window rather than a database under load, and the two can differ
by more than the engines differ from each other.

Three things about that ratio, because a ratio of two noisy quantities is the
shape this report has published wrong twice before:

- **Both percentiles come from one pool.** The client count, the query pool,
  k, the `ef_search` and the ops unit — M, build mode, storage engine,
  resource pass — are identical across the two windows, the harness refuses to
  divide them otherwise, and both land in a single record so nothing
  downstream has to pair two records. The table prints both percentiles beside
  the ratio so a reader can do the division.
- **Readers compute no recall.** The corpus is mid-change while they run, so
  no ground truth describes it; `recall_at_k` is null on those records rather
  than zero.
- **It is a ratio of wall clocks, and should be read as a direction and a
  magnitude.** Treat its decimals as the box's rather than the engine's.

`extra.reinsert_rows_per_s` on a churn step record is therefore now a write
rate measured *with* those readers running, and `extra.readers` beside it says
how many. `ops.churn_readers: 0` measures the write-only rate, which is the
quantity this report showed before PR 4 and is not the same quantity;
`config/profiles/dev.yml` runs that pass.

**The reader pool writes a third kind of churn record, and nothing may pool it
with the other two.** It carries `phase: "churn"` and a `churn_fraction` like
the rest, so every existing churn artifact selects it by default — but its
`qps` is aggregate throughput over `ops.churn_readers` clients against a corpus
that is mid-change, and it carries no recall at all, because no ground truth
describes a corpus being rewritten (`recall_at_k` is null there rather than
zero, for the same reason a budget-stopped filtered point is). It is marked
`extra.during_writes: true`, and `report/churn_state.py:is_during_writes`
is the one place that reads the mark, and `churn_measurements` beside it is
the filter every other surface goes through: the single-client churn table and
all three churn charts — degradation, retention and impact — select their
records with it rather than on `phase` alone. Handed one of
these records, the retention chart would divide an aggregate multi-client
figure by a single-client pre-churn baseline and publish the quotient on an
axis labelled "QPS as a fraction of pre-churn" — a new wrong result of exactly
the kind PR 4 exists to remove, and one that the change which fixed the others
would have introduced.

### 2.6 Uncertainty intervals, and when two engines are not ordered

Every figure here is one sample of a noisy process. Without an interval beside
it a 3% gap between two engines reads as a result, and the report has no way to
say "these two measurements cannot be told apart".

**Single-client recall and QPS carry a 95% bootstrap interval.** Calibration,
filtered and churn records each carry `extra.recall_ci95` and
`extra.qps_ci95`, as `{low, high, n}`. The method is a percentile bootstrap:
1,000 resamples of the per-query measurements with replacement, the statistic
taken on each, and the 2.5th and 97.5th percentiles of those reported. No
closed form, because almost nothing measured here is normally distributed —
latencies are right-skewed with a long tail and per-query recall is a bounded
fraction that piles up at 1.0. The resampling seed is fixed, so a report
regenerated from the same records is byte-identical.

- **Recall** is resampled from the per-query recalls, not from repeated runs.
  So the interval says how well *this measurement* pins the engine's recall
  over its query set; it does **not** describe run-to-run variation.
- **QPS** at one client is `queries / sum(latency)`, which is `1 / mean
  latency`. The mean latency is what gets resampled and the bounds are
  inverted, which also swaps them: the slow end of the latency interval is the
  low end of the throughput interval.
- **`n` travels with the bounds**, and it is not decoration. A zero-width
  interval has two causes that the bounds alone cannot separate: 500 queries
  that all found the whole top-k, which is a real measurement of no spread,
  and a single query, which is the absence of one. Only the first supports a
  claim.

**Concurrency deliberately has no bootstrap interval.** Its spread comes from
repeating each point — `ops.concurrency_repeats`, reported as
`extra.qps_samples` and `extra.qps_spread_pct`. Samples inside one concurrency
window are not independent: sixteen clients queuing on one server produce
latencies that rise and fall together, so resampling them would estimate the
spread of a quantity nobody measured and report it narrower than the
window-to-window spread already recorded. The harness refuses to compute one
rather than relying on this paragraph.

**Where two engines' intervals overlap, the tables say "tied with" instead of
letting the numbers imply an order.** The churn and filtered tables carry a
`Tied with` column and the corresponding charts draw the interval as a
whisker. The comparison is closed-interval overlap — two engines whose
intervals merely touch are reported as tied, which is the conservative
direction — and it is made only within a group that holds the dataset, the
resource pass, `M`, the build mode, `ef_search` and the point being measured
constant. Storage engine is **not** held constant, because engines
legitimately differ on it; it is named inside the cell instead.

Three answers, not two. `distinct` means the intervals do not overlap and the
ordering the numbers suggest is supported. `no interval` means at least one row
at that point has no usable interval — an archived record, or too few queries
for a spread to mean anything — so the comparison is **not available** rather
than negative. A dash means no other engine was measured at that exact point.
The distinction matters: a two-valued version of this column reports
"distinguishable" for a measurement that never happened, because a comparison
against a `nan` interval is false whichever way it is written.

### 2.7 Cross-checking the two measurement paths

The recall/throughput sweep and the ops calibration sweep measure **the same
index** with different code: ann-benchmarks in-process against the whole test
set, and the ops harness over TCP against the first 1,000 queries. Where a
configuration was measured by both, the two recalls should agree. Section 10 of
the report lists every such pair with both values and a pass/fail mark, and an
empty list is reported rather than hidden.

**The tolerance is `max(0.01, 2 × the ops 95% CI half-width)`.** One recall
point is allowed unconditionally, because the two paths score different query
subsets; a noisier ops measurement is allowed proportionally more. A record
with no usable interval gets exactly the floor, which is the stricter of the
two branches — a missing interval cannot widen the tolerance.

**What the pair is matched on, and the two fields it cannot be matched on.**
Engine, dataset, resource pass, `M`, `ef_search`, quantization and build mode.
Two fields are special, and both for the same measurable reason: the ops
harness stamps a default on every record that the ann path either names
differently or does not name at all.

- **Storage engine** is not in the key. `--storage-engine` defaults to
  `InnoDB` for every engine, so pgvector's ops records say `InnoDB` while its
  ann module reports `heap`; Valkey's says `memory` and Percona Search's
  `wiredTiger`. Requiring equality would empty the check for three of seven
  engines on a naming difference. (Percona Server is not a fourth: its ann
  module reports `InnoDB`, like the rest of the MySQL family.) It is used to
  pair rows *inside* a key, under the same normalisation the rest of the
  report uses, and is shown on every row.
- **Build mode** is in the key, but only for the engines that have one. The
  MariaDB, MariaDB 12.3, AliSQL and Percona Server ann modules write no
  `build_mode` at all — four of seven — while the ops harness stamps `post`
  on every record regardless of what the engine did. For MHNSW and VIDX the
  graph is maintained on every INSERT whatever that field says. So the axis is
  dropped wherever the recall sweep is silent about it and kept wherever it is
  not, and a row whose build mode was not matched on is marked.

  **Percona Server is the fourth for a different reason from the other three,
  and it is the one worth knowing about.** MHNSW and VIDX have no separable
  bulk build, so a build mode would be meaningless for them. Percona Server
  has one — `PerconaDriver` splits `create_schema` and `create_index` on
  `IndexSpec.build_mode`, and `config/profiles/percona.yml` runs both modes
  through the ops path — but its *ann* module does not read `build_mode`,
  which is why `orchestrator/ann_pass.py` refuses `ann.percona_build_modes`
  rather than emitting two identical recall curves. So for this engine the
  axis is real on the ops path and silent on the recall path, and dropping it
  here is still right: the ann records genuinely say nothing about it. The
  build-mode comparison for Percona therefore comes from the ops records and
  not from a pair of recall curves: the recall path is silent about
  `build_mode`. `index_bytes` is not ops-only; every recall record carries it.

**An empty list has two causes and the report distinguishes them.** "No `M` is
measured by both paths" is a profile choice: the recall sweep varies `M`
because it is the only build knob MHNSW and VIDX expose, while the ops phase
pins one `M` because every extra value reloads the whole corpus. "`M` overlaps
but no point matched" is a **fault**: both paths measured a shared `M` and no
pair formed anyway. The coverage table under section 10 says which case each
engine and dataset is in.

**On `main` the grids meet, so an empty section there is the fault case.** That
profile runs `ann.m_values: [8, 16, 32]` against `ops.m_values: [16]`, and the
16 is in the ann grid for no other reason than to give this check something to
compare — so every engine should produce matched rows and an empty
consistency section on a `main` run is a reason to investigate rather than a
reassurance. An earlier revision of this paragraph described the `[8, 32]` grid
`main` carried before that change and called the check "empty by design"
there; the profile and the sentence moved in the same pull request, and only
one of them was updated.

**Throughput order is flagged, never failed.** At matched recall the engine
order should agree between the two paths; where it does not, the report lists
the pair. The ops path pays a TCP round trip per query that the in-process
path does not, so two engines close together in-process can legitimately swap
places once that overhead is paid. "At matched recall" is enforced with a band
rather than assumed from a shared `ef_search`: a fixed `ef_search` is not one
quantity across these engines and buys recalls up to 0.18 apart, and a
throughput order between two engines doing different amounts of work is not a
measurement of speed.

## 3. Fairness

Two passes are run. Both are reported; neither alone is the answer.

### Normalized pass

Every engine gets identical constraints:

- Same `--cpuset-cpus`, restricted to a homogeneous core set. On hybrid CPUs
  (Intel P-core/E-core) the harness detects core types and uses performance cores
  only; hyperthread siblings are excluded unless explicitly enabled. Mixing core
  types produces run-to-run variance that swamps the effect being measured.
- Same `--memory` container limit.
- Equalized graph-cache budget: `mhnsw_max_cache_size` ≈ `vidx_hnsw_cache_size` ≈
  the pgvector `shared_buffers` allocation.
- Identical `M` grid and identical `ef_search` grid.
- Single client (for the recall/QPS pass).

### Tuned pass

Each engine configured per its own vendor documentation, full machine available.
This answers "what can this engine actually do", at the cost of confounding
implementation quality with documentation quality.

## 4. Known asymmetries

These are real differences between the engines that no amount of configuration
removes. They must accompany any published number.

| Asymmetry | Consequence |
| --- | --- |
| **`ef_construction` is exposed** by pgvector, Valkey, Percona Server, MyVector and VillageSQL (on MySQL 9.7 and on MySQL 8.4 — one extension, so one knob), and by no other MySQL-family engine here — not by MariaDB MHNSW and not by AliSQL VIDX. Percona Server, MyVector and VillageSQL are the three MySQL-family engines with both a build-time and a query-time effort knob (Percona's `ef_construction` and `innodb_hnsw_ef_search`; MyVector's `ef=` column option and the `ef_search` option of each `MYVECTOR_IS_ANN` call; VillageSQL's `WITH (ef_construction = ..)` index option, capped by the extension at 32 x M, and the `vsql_vector.ef_search` session variable). | The build-quality sweep varies `ef_construction` only for the five that have it, and `M` for every engine but Percona Search, whose index exposes no graph degree (`orchestrator/ann_pass.py` gives it one `M` value and carries it as a label) — so neither claim is about the MySQL family, which has engines on each side of this line. It is pinned to each engine's default in the normalized pass (pgvector's is `ef_construction=200`) so no engine is given a tuning axis the others lack, and swept only in the tuned pass. §3 of a generated report states this row from the records of the engines that actually ran, so a single-engine report names one engine where this table names five. |
| **Percona Server's graph is cold after a build.** It is faulted in lazily from the entry point, and the server team states that the ALTER a `post` build issues evicts the table; `Innodb_hnsw_memory_used` read 0 after 95 of 95 `post` builds. No other engine here has been shown to start its query phases cold after a build. | Percona alone (`EngineDriver.warm_graph_after_build`) gets one discarded pass over calibration's query set at the top ef, inside `build.run` and in both of its build modes, before calibration; every other engine gets only the per-point discarded queries every engine gets. The pass is not scored and not in `build_wall_s`; the `index_build` record carries it as `extra.graph_warmup` and later records as `extra.graph_warmup_status`. Ops measurement version 11. See `docs/09-adding-percona-server.md` §2.6. |
| **AliSQL requires `READ-COMMITTED`** and `vidx_disabled=OFF`; vector indexes are InnoDB-only. | AliSQL runs at RC. MariaDB and PostgreSQL are run at their defaults in the tuned pass and at RC in the normalized pass where the engine allows it. |
| **MariaDB supports MyISAM** for vector tables (and its own published benchmark uses MyISAM); AliSQL cannot. | Headline comparison is **InnoDB for both**. MariaDB MyISAM is reported as an additional MariaDB-only curve in the tuned pass, clearly labelled. |
| **`M` is set differently** — MariaDB via the `mhnsw_default_m` server variable at startup, AliSQL via a per-index DDL option, pgvector via `WITH (m=…)`. | Handled per-driver; the effective `M` is recorded in the manifest, not assumed. |
| **SIMD paths differ.** Both AliSQL and MariaDB document AVX-512 distance kernels. | The harness records CPU flags. On a machine without AVX-512 both fall back to narrower paths, and the result is only valid for that class of hardware. Any report states the CPU and its flags. |
| **Default distance metric differs** — AliSQL defaults to `EUCLIDEAN`, and metric is fixed at index creation for all three. | Metric is always set explicitly per dataset; never left to a default. |

## 5. Reproducibility

Every run writes `results/<run-id>/run-manifest.json` containing:

- Docker image digests for each engine image actually used.
- Engine source commit SHAs and tags.
- Full resolved configuration (profile + engine + resource pass), plus its hash.
- CPU model, flags (`avx512f`, `avx2`, `avx_vnni`, …), core topology, hybrid-core
  detection result, total RAM, kernel version, Docker version.
- Start/end timestamps per phase.
- The ops measurement version (`config.ops_measurement_version`), which is also
  written into every ops record as `extra.ops_measurement_version`.

A result without its manifest is not a result. The report generator refuses to
emit a report for a run directory with no manifest.

The ops measurement version is bumped whenever a change to `harness/` alters what
an ops number means, as distinct from `SCHEMA_VERSION`, which describes only the
record's shape; records carrying no version at all were measured under version 1.
**Version 2 measures exactly what version 1 measured** — it is simply the first
value the scheme writes down, so a version 1 number and a version 2 number are
directly comparable and nothing needs re-measuring to get one. That pair is what
`OPS_MEASUREMENT_EQUIVALENT_VERSIONS` (`harness/metrics/records.py`) holds — not
as a literal set but *derived* from the two constants that name the versions,
`IMPLICIT_OPS_MEASUREMENT_VERSION` and `FIRST_RECORDED_OPS_MEASUREMENT_VERSION`,
so a third member cannot be added without first naming a third concept and
saying why that version changed nothing measured. That set is what the refusal
below reads, so that it never tells an operator to re-measure numbers this page
calls comparable.

**Version 3 is not comparable with either**, and is deliberately absent from that
set. It separated the concurrency query pool from `ops.max_queries` (§2.3), so a
version 2 concurrency figure is throughput over a 500-query working set and a
version 3 figure is throughput over the whole 10,000, measured against caches
that behave differently at the two sizes. That is the same workload over two
different amounts of the corpus, not two measurements of one quantity, and the
engines whose caches differ most are exactly the ones the difference moves.
Concurrency recall also arrives in version 3, but a field that was absent
becoming populated is new information rather than a changed number and would not
on its own have earned a bump. Version 3 also counts only the queries that
finish inside each concurrency window (§2.3), and that correction is part of what
version 3 means rather than a fourth version — no run has ever been measured at
version 3, because the value was introduced and then extended inside the same
change set, so there is no version 3 figure anywhere for a trimmed one to be
incomparable with. The magnitude says the same thing from the other side: the
correction moves a figure by roughly `per-query latency ÷ window length`, which
at `main`'s 30-second windows and the p50s these engines actually record is
about 0.005% — well inside the window-to-window spread
`ops.concurrency_repeats` exists to expose.

The bump invalidates comparison across the other
ops phases too — build cost, filtered, churn — none of which version 3 changes;
the version names the harness that wrote the record rather than the phase, which
is the cost of having one constant instead of four that can each be forgotten.

The report refuses
to draw records from two versions on one chart, naming both versions and the
records involved, exactly as `ann_fingerprint` keeps a 16 GB recall curve out of a
report whose manifest says 64 GB. A version already recorded in a manifest is
carried forward untouched rather than re-stamped, so resuming an old run does not
relabel measurements it did not take.

**A resume across a bump says so before it measures anything.** The refusal
above is correct but it arrives at report time, which is after the expensive
half: resuming a `main` run under newer code meant up to days of ingest before
anything mentioned that the directory could not be reported as one. `set_config`
now prints a warning naming both versions and the remedy — one report per
version — at the start of the resume, once per run. It is not fatal: finishing
the run and splitting the `ops-*.jsonl` files by
`extra.ops_measurement_version` afterwards is legitimate, and the older records
are not wrong, only incomparable.

## 6. What would change the conclusions

Stated up front so readers can judge relevance:

- **Different CPU.** AVX-512 availability plausibly reorders the engines.
- **Dataset scale.** These datasets are 60k–1M vectors. Behaviour at 100M is not
  extrapolable from them — cache-resident graphs become disk-resident.
- **Version drift.** All three projects are actively developing vector search.
  Results are pinned to the tags in the manifest and go stale quickly.
- **Client-side overhead.** ann-benchmarks drives queries from Python. At high
  QPS the client can become the bottleneck; the harness records client CPU
  utilisation so this is detectable rather than silent. CPU utilisation is not
  sufficient on its own — see §7, which is about the case it cannot see.

## 7. Multi-client throughput and whether the clients achieved it

The ops harness runs its concurrency clients, and the churn workload's readers,
as **forked OS processes**, each with its own interpreter and its own
connection. It ran them as Python threads in one interpreter until
`OPS_MEASUREMENT_VERSION` 6, and most of this section is about why that
changed — read it as the history it is, because the figures in it were taken in
both regimes and each says which.

Either way, every multi-client figure is conditional on something the figure
itself does not state: that the pool really had *N* queries in flight.
`Scaling efficiency` divides an *N*-client throughput by *N* times a
one-client throughput, so if the clients took turns it reports a property of
the harness in a column a reader will take for a property of the database.

**This was found by a live run, not by reading the code.** §2.3 has that run
and what it shows — flat QPS with flat latency at four clients, about one query
in flight, `client_bound` false at a `client_cpu_util` of 0.177 against its
0.80 threshold — and this section does not repeat it. Read §2.3 first; it is
the reason this one exists. One figure from the same run belongs here rather
than there because it is a *write* rate: the churn writer managed 12.0 rows/s,
2,000 rows in 166.8 s, against four concurrent readers, where the modelled
MariaDB ingest is about 150 rows/s.

**What the harness now records, and why it is not a CPU threshold.** Every
measured window reports two figures derived from the spans it already keeps:

| Field | What it is |
| --- | --- |
| `extra.outstanding_concurrency` | The exact time-average number of queries **outstanding**: the sum of every in-window query's duration, divided by the window. No estimator, no distributional assumption. |
| `extra.served_concurrency` | The queries the window counted times their **median** latency, divided by the window — a median **estimate** of how many were in service; `served / outstanding` is exactly `p50 / mean`. |
| `extra.scaling_status` | `ok`, or `concurrency_not_achieved`. Since `ops_measurement_version` 9: when `outstanding_concurrency` fell below half the client count, or when both the median estimate and a credible scaling efficiency did. Before version 9: when `served_concurrency` fell below half the client count. |
| `extra.scaling_note` | Version 9: `tail_dominated` when `served / outstanding < 0.5` — the latency had a heavy tail. A note, never a refusal. |

**What the version-9 guard detects, and what it does not.** It detects
client-side starvation — the clients kept fewer than half their queries
outstanding — and the residue where both the median estimate and a credible
efficiency are under half. It does **not** detect a server that answers one
query at a time: from the clients' clocks that server is a saturated one
(outstanding ≈ N, the median ≈ N because the queue raises it, efficiency ≈
1/N), and it is published `ok` with its low efficiency. Serialization inside
the server and saturation of it are not distinguished by these figures; the
report's Server CPU and Server CPU pressure columns, taken from the server's
own cgroup over the same window, are the evidence for that.
| `extra.efficiency_reliability` | Version 9: `baseline_not_credible` when the one-client median was more than 1.4x this point's, in which case `scaling_efficiency` is null. |

Where the status is `concurrency_not_achieved` the report prints no scaling
efficiency for that point at all, the chart's efficiency panel omits it, and
the Validity section names the point and shows the figure it withheld. The QPS
and the percentiles stay: the pool did answer that many queries at those
latencies, and it is the division by the client count that has no basis.

The threshold is half the client count, and it sits in the middle of a measured
gap. Five synthetic drivers of identical per-query cost, differing only in what
they do with the interpreter lock, were run through the real client pool at one
and four clients — CPython 3.12.3. **Both arms are measured, and the second is
the point of this section**: the left column is the retired thread pool and the
right is the process pool the harness runs today. One driver built once per
row, both arms in one fixture, so a row is one experiment rather than two runs.
All of it is reproducible from the tree —
`tests/test_g3_process_clients.py` builds the regimes and reports the table,
and `tests/retired_thread_pool.py` is the thread arm, kept for the reason given
below.

| Driver | Efficiency at 4 (threads / processes) | `served / clients` | `outstanding / clients` |
| --- | --- | --- | --- |
| releases the lock (`time.sleep`) | 1.006 / 1.009 | 0.975 / 0.983 | 0.994 / 0.980 |
| **saturated server** (latency rises with the client count) | 0.276 / 0.272 | **0.977 / 1.004** | 0.994 / 0.991 |
| C call holding the lock (`ctypes.PyDLL` `usleep`) | 0.249 / **0.995** | 0.243 / **0.982** | 0.944 / 0.979 |
| Python spin holding the lock | 0.265 / **0.999** | 0.264 / **0.995** | 0.835 / 0.995 |
| half released, half held | 0.518 / 0.980 | 0.979 / 0.980 | 0.997 / 0.991 |

Four things to read off it. **Only the two lock-holding rows moved**, which is
the whole claim of the change below: the clients no longer share an
interpreter, so a driver that holds the lock no longer makes them take turns.
**A saturated server with *deterministic* service time is not flagged in
either arm**, which is the case the threshold was chosen to protect: a real
queue raises the whole latency distribution, median included, so that row's
`served` stays at the client count while its efficiency falls to 0.27.
**That is the whole of what this table measures, and the claim is narrowed to
it on purpose.** Every driver in it has a fixed per-query cost, so a server
whose service time is skewed far enough to put its median well below its mean
can still be refused — a known false positive, handled separately, and this
section does not claim the rule is sound. What the table does establish is
that the rule does not refuse the *deterministic* ceiling, which is the case
a reader of a seven-engine study meets most often. **The lock-releasing row
did not move either**, so the pipes, the barrier and the span transport cost
the pool nothing. And **the exact figure is the blind one** — `outstanding`
is 0.84–1.00 of the client count on all ten readings, across efficiencies
from 0.249 to 1.009. The mechanism holds in both arms and is not about the
lock: a span brackets `driver.query`, so it covers whatever the query spent
waiting as well as whatever it spent being served, and the stretched spans
tile the window either way. Under threads the wait was the interpreter lock;
under processes it is a queue in the engine or in its socket. It is recorded
anyway, because the pair is the diagnostic — outstanding at the client count
while in service sits near one is the signature of waiting rather than of
serving. It says the clients were not being served and not what they were
waiting for; see §2.3 and the report's own footnote for the limits of that.

**The lock-holding row is a C call that WAITS with the lock held, and the
earlier version of this table burned CPU with it held instead.** That matters
for two reasons. A real client library blocked on a socket with the lock held
burns nothing, so a CPU-burning driver was never a faithful model of it. And
four CPU-bound client *processes* deliver four queries at once only if the box
has four cores spare, so an assertion on that row would have been a joint
measurement of the lock and of the machine — measured, it failed at 8–18 busy
cores of 20, correctly and for a reason that was not the lock.
`ctypes.PyDLL` does not release the interpreter lock around a call and
`ctypes.CDLL` does, so the same libc `usleep` gives both halves of a controlled
pair: four threads × 200 waits of 500 µs took **0.466 s** held against a
fully-serial 0.400 s, and **0.116 s** released against a fully-parallel
0.100 s, and the held form costs a measured **0.034 cores**. §7's Python-spin
row is kept because it is the regime this document published, and only the part
of it that is about the lock is asserted.

**`client_bound` is kept, and §2.3 says why it cannot see this.** Nothing here
replaces it: a client container that genuinely runs out of CPU is a real case
and `scaling_status` cannot name it as a CPU limit, while a lock ceiling
prevents the very CPU consumption `client_cpu_util` looks for. The two
quantities are complementary and both are recorded.

One correction to how §2.3's 0.177 should be read, because it bears on every
use of that counter: `read_own_cpu_usec` opens `/sys/fs/cgroup/cpu.stat`, which
**outside a container is the root cgroup — the whole machine**. Measured on a
bare host, it reported 2.421 cores burned while the reading process slept. So
`client_cpu_util` is only a measurement of the clients when the harness runs
in its own container, which is how a real run runs it; the 0.177 is carried
through this repository as a figure from such a run and not as something a
test on a developer box can reproduce. `PoolWindow.client_bound`'s docstring
says so beside the number.

**The churn workload's write rate depends on this too, and the dependency
outlived the mechanism.** `extra.reinsert_rows_per_s` is measured against
`ops.churn_readers` concurrent readers, so a reader pool that served one read
at a time leaves the published rate describing an unknown amount of read load
— whatever produced it. Until version 6 there was a second and much larger
mechanism: those readers were threads in the *writer's own interpreter*, so
readers that took turns starved the writer inside the client rather than inside
the engine. Measured on the retired thread pool against the synthetic driver,
with a 4 ms write slice and four readers whose per-query cost is identical in
both arms: readers that release the lock put that published rate at
948.2 rows/s, and readers that hold it put the same figure at 41.1 — a factor
of 23, from nothing but what the client library does with a lock. **Each reader
is a forked OS process now, so that factor is history rather than a current
property of this harness**, and the figures above are kept because they are
what established the size of the effect. The reader pool's status still travels
onto the step record that publishes the rate, as
`extra.reader_scaling_status`, and the during-writes table still marks the
writer cell as well as the latency ratio.

**What was read rather than measured, and the run that measured it.** Whether a
particular client library releases the interpreter lock around its socket wait
is a property of that library's C extension, and no measurement in this
repository settles it: these tests start no server. The engines do not share a
library — pgvector goes through **psycopg3** and both MySQL-family engines
through **MariaDB Connector/Python** over Connector/C, which
`docker/_shared/requirements-mysql.txt` explains is deliberate so the two are
not compared through different clients. This section used to say the next real
run would settle it for free, and name the shape: pgvector near 1.0 at four
clients while MariaDB sits near 0.25.

**That run happened on 2026-10-14 and the shape was exactly that** (SUPPLIED —
a six-engine `smoke` pass on fashion-mnist-784 at M=16, a Xeon Gold 6230,
`SMOKE/results/smoke-20261014-075119/`, 158 ops records; not reproduced by any
test here). `served = qps × p50` per client, at one client and at four:

| Engine | Client library | `served / clients` at 1 → 4 |
| --- | --- | --- |
| `mariadb` | MariaDB Connector/Python | 0.95 → **0.238** |
| `mariadb123` | MariaDB Connector/Python | 0.95 → **0.237** |
| `alisql` | MariaDB Connector/Python | 0.97 → **0.242** |
| `pgvector` | psycopg3 | 0.98 → 0.974 |
| `valkey` | valkey-py | 0.97 → 0.969 |
| `mongodb` | pymongo | 0.99 → 0.984 |

Exactly the three engines sharing one library are pinned near one query in
service however many clients ask, and the three that do not are at their client
count. It is not the pool: the same pool took pgvector from 2,466.9 to
8,754.7 q/s over the same four-fold rise. The same run's churn writer managed
**11.991 rows/s** against four readers where the same run's single-threaded
load of the same corpus recorded `ingest_rows_per_s` **769.9** at
`ingest_threads` 1 — a factor of 64 inside one run, with nothing changed but
whether four readers were querying.

**The clients are processes, and this is what it cost.** Running them that way
is the only way to remove a shared interpreter lock from the measurement path,
and it is an architectural change rather than a correction. It was taken before
the `main` run because taking it afterwards would mean re-running the
measurement. Each client is now an OS process with its own connection and its
own interpreter; `harness/workloads/client_process.py` is the client and
`harness/workloads/clients.py` is the pool. What crosses the fork is three
things and no more: the barrier stays a *thread* barrier in the parent, with
one attendant thread standing in for each client, because its action reads the
window's opening instant into parent memory; two `multiprocessing.Event`s carry
the barrier's decision and the stop flag's to the children; and each client's
spans come back over its own pipe and are rebuilt into the same
`LatencyCollector` objects the merge has always been handed, so
`merge_collectors` still works over one shared wall under one
`span_disposition` rule, and `extra.prewindow_discarded` is still zero by
construction.

The cost list this section used to publish as a reason to decide carefully, now
measured item by item rather than estimated:

- **fork and not spawn, and it is forced rather than preferred** (MEASURED).
  `driver_factory` is a callable the caller supplies, and every real one is
  unpicklable: `harness/main.py` defines it as a `def` nested inside `main()`,
  whose `__qualname__` is `main.<locals>.driver_factory`, and pickle resolves a
  function by qualified name — `AttributeError: Can't pickle local object`. The
  test suite's are a `lambda` and a bound method. Under spawn or forkserver the
  pool would have to be handed something picklable, which is a change to every
  caller's contract. **The start method is pinned and not inherited**, because
  CPython's default on Linux is `fork` today and becomes `forkserver` in 3.14:
  an inherited default would turn a working pool into `Can't pickle local
  object` at the start of a measured window on a release upgrade.
- **The memory cost this section used to state was wrong** (MEASURED for the
  direction, SUPPLIED for the figures). It said each client gets its own copy
  of the query set, so the harness container's memory grows with the client
  count. It does not. `shard_queries` hands out strided *views* of one array,
  a client never *writes* to one, and fork's copy-on-write never copies a page
  nobody writes. It does not follow that a reader child leaves the pages
  alone: a shard is `queries[c::clients]`, strided across the whole array, so
  every child *reads* pages spread over the entire query set — what is absent
  is the write that would make the kernel copy one.
  The figures below are **SUPPLIED**: no test here reproduces them, because
  `tests/test_g3_process_clients.py` runs the same probe at four children over
  a 33.6 MB array and asserts the ratio between its two arms rather than any
  absolute. They are a **sum over eight children against one 104.9 MB float32
  array**, not a per-client figure and not a figure against a baseline: summed
  `Private_Dirty` **13.8 MB — 0.13×** the array, against 838.9 MB if each
  child had copied it. The control arm is what makes the direction a
  measurement rather than a blind probe: eight children that *write* the array
  sum to **851.7 MB, 8.12×**, so the probe can see a copy when there is one.
  What a client does cost is its own interpreter's private pages, about
  1.7 MB.
- **The clock needed no replacing** (MEASURED, and the asserted bound is
  looser than the observation — deliberately). `time.perf_counter()` is
  `clock_gettime(CLOCK_MONOTONIC)`, which is the machine's clock rather than
  the process's — unlike `time.process_time`, which is the control. Over 200
  forks the child's offset from `CLOCK_MONOTONIC` matched the parent's to
  within a few microseconds — **4.2 µs** on the box this change was made on
  and **8.05 µs** re-measured on a box at 20 busy cores of 20 — which is the
  cost of reading two clocks back to back rather than an offset, and the
  child's reading fell inside the parent's bracketing pair **200 times out of
  200** in both. `tests/test_g3_process_clients.py` runs both of those at 200
  forks, and asserts **10 ms** rather than the observed microseconds: the
  quantity is the difference of two offsets, each taken across two consecutive
  clock reads, so a thread descheduled between them on a shared box widens it
  by milliseconds for a reason that is not the clock. 10 ms is still four
  orders of magnitude below what a per-process clock would give — the
  control's child reads near zero against a parent's whole process age — so
  the loose bound is what distinguishes the two clocks, and the microsecond
  figures are reported rather than asserted. A platform where this is false
  fails loudly rather than producing spans that cannot be compared with the
  window they are inside.
- **Client CPU accounting still covers the clients** (VERIFIED STRUCTURALLY).
  `fork` never changes cgroup membership — only an explicit write to
  `cgroup.procs` does — so every client is in the harness container's cgroup
  and the one counter `read_own_cpu_usec` reads covers all of them. Checked by
  comparing `/proc/self/cgroup` between parent and children, which is exact.
  The *magnitude* cannot be checked outside a container, for the reason §2.3
  gives: there the same file is the root cgroup and reads the whole machine.
- **`/dev/shm` needs no raising** (MEASURED). `multiprocessing` semaphores are
  POSIX semaphores, and it calls `sem_unlink` immediately after `sem_open` —
  so they never appear in a `/dev/shm` directory listing while their pages stay
  charged to the tmpfs. A probe that counts files reports zero for the wrong
  reason; the filesystem's used bytes are what to measure. One pool costs
  **40,960 bytes** and the figure is *independent of the client count*, because
  the go and halt flags belong to the pool rather than to each client — the
  rejected per-client design measured 1,310,720 bytes at 32 clients. Pipes cost
  zero. The harness container is started with `shm_size` from the resource pass
  (2 GB normalized, 8 GB tuned), so the pool's footprint is five orders of
  magnitude inside it.
- **`PoolError`'s discipline is re-established against exit codes and
  `SIGKILL`** rather than thread exceptions, which is a wider set of failures.
  A client that dies *before* the barrier closes its pipe, so the parent's
  `recv` raises `EOFError` and the attendant aborts the barrier — measured, the
  pool fails in 0.01 s against a 2 s timeout instead of waiting it out. One
  that dies *mid-window* is caught the same way. One that is killed *after*
  reporting leaves no error and a window that looks complete, so `stop` checks
  `exitcode` as well: a client reaped by a container's memory limit at the end
  of a long window would otherwise have its partial window published as a
  measurement. A window in which nothing finished is still a legitimate
  measurement and still does not raise.
- **The cost this section never listed, and in practice the largest one**: the
  process boundary cuts the *test suite's* instrumentation channel. Five
  existing tests injected a plain Python object into a driver factory — a dict,
  a counter, a `threading.Event` — had the client touch it, and read it back in
  the parent. Under fork the child gets a snapshot: its writes never come back
  and the parent's later writes never reach it, so the technique fails
  *silently* rather than loudly. Measured before conversion: a query counter
  read 0 against 3,922 queries answered; a three-window latency staging left
  every window at one speed and a 1.04× throughput spread where the test needed
  1.5×; a reader-failure `Event` set by the writer never reached the readers.
  All five now use `multiprocessing` shared state and each says why. Anything
  new that needs to observe or steer a client has to do the same.

**One consequence for this whole section, and it changes how
`concurrency_not_achieved` should be read.** With the clients in one
interpreter, that status was an expected finding — the live `smoke` run
produced it on three engines. With the clients in separate processes it is
close to unreachable: a serializing *server* raises the median along with
everything else, so `served` stays near the client count, and it was the
client-side lock that produced the low reading. The guard has therefore become
a **tripwire** rather than a finding — for a revert to threads, for a driver
that serializes inside one process at the C level, for a future pool bug. A
tripwire nothing can trip cannot be told apart from a broken one, which is why
`tests/retired_thread_pool.py` keeps the retired pool alive as a test helper:
it is the only thing in the tree that can still produce the shape, and the
refusal, the operator warning and the report's withheld cell are all still
exercised end to end against a window it measured.

It also changes what every published multi-client number means, so it
invalidates comparison with every run measured before it — including the
2026-10-14 run above, which is the first that ever stamped a version 5. That
is an `OPS_MEASUREMENT_VERSION` bump by the same rule version 3's query-pool
change earned one, and it is version **6**.

### 7.1 Server-side evidence: wait counters and CPU profiles

The version-9 guard above cannot tell a server that serializes from one that
is saturated, and neither can the Server CPU and pressure columns on their
own: both readings are "server CPU below its cpuset, clients not bound". What
can is the engine's own account of what its threads waited on, so every
concurrency record (and the churn `during_writes` record, for both reader
windows) carries `extra.server_waits`: the top ten events by wait time, their
counts and the total, differenced from the engine's counters.

| Engine | Source | Unit |
| --- | --- | --- |
| MySQL family | `performance_schema.events_waits_summary_global_by_event_name` (excluding `idle`), plus `SHOW ENGINE INNODB MUTEX` when it returns rows | seconds waited |
| pgvector | **opt-in only** (`ops.server_wait_sampler`): `pg_stat_activity` sampled once a second *during* the window, active client backends only | samples per wait event (`CPU (no wait event)` when none) |
| Valkey | `INFO commandstats`, plus `LATENCY LATEST` spikes stamped inside the window | seconds of command *service* time, not waiting |
| MongoDB | `serverStatus` lock acquisition waits and ticket gauges; mongot's metrics endpoint when reachable | seconds waited |

The snapshots are read on a **separate connection**, immediately before and
after the window and never inside it, so the deltas also span the pool's
warm-up and teardown (`server_waits.covers` says so). When an engine cannot
say — MariaDB's `performance_schema` is off by default — the record carries
`server_waits: null` and `server_waits_reason`; an empty event list is
reserved for "measured, and nothing waited".

**pgvector is the asymmetry, and by default it has no wait profile.**
PostgreSQL keeps no cumulative wait-event counters, so two snapshots at the
window's edges would be two point-in-time views of mostly idle backends, not
a delta of anything — and they are not substituted. The only source is
sampling `pg_stat_activity` while the clients run, which adds a backend and a
query whose cost grows with the client count to the very window being
measured. That perturbs pgvector's window and no other engine's, so it is off
by default: pgvector's records carry `server_waits: null` with the reason.
`ops.server_wait_sampler: true` turns it on; then `engines_resolved` lists it
and every pgvector record carries `extra.server_wait_sampler: {interval_s,
concurrent: true}`, and its throughput is not comparable with a run that had
it off.

**Latch waits are opt-in.** With the shipped settings the MySQL-family
profiles contain file IO, table IO and table-lock waits but no mutex or
rwlock, because those instruments are off by default. `ops.server_wait_instruments:
true` turns them on (and `performance_schema` itself on MariaDB) for the ops
servers only. **It changes the server being measured**: every latch
acquisition then reads a timer. Measure the overhead first — the same `smoke`
unit with the key off and on — and do not publish throughput from a profile
that has it on without saying so; while it is on, `engines_resolved` lists it
under `applied` and every record carries `extra.server_wait_instruments`.

**CPU profiles are optional.** `ops.profile_window_clients: <N>` adds one
EXTRA window at N clients per ops unit — same length, after the measured
repeats, never a candidate for the published median, like the version-9
warm-up window — and has the orchestrator run `perf record -g` against the
server container's cgroup over it. Its throughput is recorded as
`extra.profile_window_qps` beside whether perf was recording
(`extra.profile_window`), writing `perf.data`,
`folded.txt` and `profile.json` under `<run>/profiles/<unit>-c<N>/`. perf is
used only if the host already has it and permits it; otherwise the reason
(not installed, `perf_event_paranoid`, missing CAP_PERFMON) is written and the
run continues.

The report's **What limited throughput** table puts client CPU, server CPU,
server CPU pressure, the top server wait and the profile link side by side and
names only the largest of the three CPU shares. It draws no conclusion beyond
that.

None of this needed an `OPS_MEASUREMENT_VERSION` bump: by default every
engine's measured windows are exactly what version 9 measured — the snapshots
are outside the timed span, and the instruments, the pgvector sampler and the
profile window are all off — and no identity or published figure reads the
new keys.
