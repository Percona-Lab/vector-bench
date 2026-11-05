# Planning a run

Benchmarking these engines is expensive in a way that is not obvious until you
are eight hours into it. This document exists so you can choose the size of a
run before starting it rather than discovering the size afterwards.

Every number here was measured, not estimated. Where a figure is extrapolated
it says so.

---

## 1. Why runs are expensive

Three multiplying factors, none of them avoidable:

### Incremental graph building

MariaDB MHNSW and AliSQL VIDX maintain the HNSW graph on **every INSERT**.
There is no bulk build. Measured on a Xeon Gold 6230, glove-100-angular
(1,183,514 × 100), single-threaded, batched:

| Engine | Ingest rate | Time to load glove-100 once |
| --- | ---: | ---: |
| pgvector (bulk build after load) | ~3,000 rows/s | ~7 min |
| MariaDB (incremental) | ~150 rows/s | ~2.2 h |
| AliSQL (incremental) | ~55 rows/s | ~6 h |

That gap is itself one of the benchmark's findings. It is also why a run that
is trivial for pgvector takes days for the other two.

### The cost rises steeply with M

More neighbours per node means more distance evaluations and more link updates
per insert. Measured on glove-100 with MariaDB:

| M | Ingest rate | Load time |
| ---: | ---: | ---: |
| 8 | ~320 rows/s | ~1.0 h |
| 16 | ~200 rows/s | ~1.6 h |
| 32 | ~70 rows/s | ~4.7 h |

**M=32 alone is about 65% of the cost of the `[8, 16, 32]` grid.** Sweeping M
is not a linear expense.

### Every M value reloads the entire dataset

ann-benchmarks constructs a fresh index per configuration, and the ops harness
does the same. Three M values means three full loads — per engine, per dataset,
per phase, per resource pass. The multiplication is quiet and it dominates
everything else.

---

## 2. What each profile actually costs

Ingest time only, before any query runs. Three engines, from the measured rates
above — hand arithmetic, kept because it shows where the cost comes from. **The
live figures are in the next table**, and two rows here no longer agree with any
method († and ‡). A third marker, §, flags a row whose figure is current but
whose profile changed for a reason worth knowing before you plan around it:

| Profile | Datasets | Passes | Ingest |
| --- | --- | --- | ---: |
| `dev` | tiny synthetic | 1 | ~1 min |
| `smoke` | fashion-mnist | 1 | ~30 min |
| `quick` | fashion-mnist, glove-100 | 1 | ~35 h † |
| `main` | glove-100, sift-128 | 1 | **~61 h (~2.5 days)** § |
| `full` | all four | 2 | **~272 h (~11 days)** |
| `mariadb-blog` | dbpedia-openai-1000k | 1 | **~10.5 h** (measured) ‡ |

† **Corrected.** This row said ~14 h, which was not a measurement of anything:
it was the same hand arithmetic as the rows around it against an M grid
`quick` no longer has. It sweeps `ann.m_values: [8, 16, 32]` and every M value
reloads the whole corpus, so the hand method now gives ~35 h and the estimator
says ~36 h.

‡ **Measured, but for one load per engine, not for the run.** The 10.5 h is the
real figure from §6 — 3.3 h MariaDB, 6.6 h AliSQL, ~0.5 h pgvector — and it is
one load each. The profile runs the recall phase and the ops phase, and each
loads the corpus for its own M value, so the run pays it about twice: the
estimator says ~21 h.

§ **`main` grew an M value for a cross-check, not for the frontier.**
PR 5 §5.2 matches an ops calibration point to an ann recall point at the same
`M`, and `main` swept `ann.m_values: [8, 32]` against `ops.m_values: [16]` —
grids that never meet, so the report's "Consistency checks" section was empty
by construction. `ann.m_values` is now `[8, 16, 32]`, which is a third full
reload of both corpora per engine: this row went from ~46 h to ~61 h by the
hand method and the estimator's figure from ~51.1 h to ~68.1 h. Restricting
M=16 to sift-128 alone would have saved about half of the increase, and needs
a per-dataset ann axis the config does not have.

**That increase is an upper bound, and by a knowable amount.** The estimator
charges every M value the same corpus reload at one rate per engine — 147
rows/s for MariaDB on glove-100, which `orchestrator/estimate.py` states is
the observed *mid-grid* number. M=16 is the cheap middle of this grid: the
table in §1 measures it at ~200 rows/s, so the extra M really costs ~1.6 h
there against the ~2.2 h the estimate charges, and the estimate is high on
the marginal M by about a third. The same flat rate makes the *total* low:
§1's measured per-M times sum to ~7.4 h for the three-value grid where three
flat loads charge ~6.7 h, about 9% under. So plan against the printed figure
— it is the conservative one for the decision this footnote is about — and
do not read either number as a measurement of a per-M cost.

`full` describes the complete measurement space. It is not a recommendation.
Running it end to end is a deliberate decision to spend a fortnight, and
`gist-960-euclidean` accounts for a large share of that on its own — at 960
dimensions it is ~19 h of loading per phase.

**`main` is the profile to run.** It was sized from these measurements to
answer the questions the smoke run raised without spending a fortnight.

### Ingest is not the run

The table above is loading only, and so was the only estimate this framework
printed for most of its life. **Querying is not free**, and for a profile like
`smoke` it is most of the cost: half an hour of loading against a churn budget
that can run for eight (two hours per step since ops measurement version 9,
up from half an hour). What the estimator prints now is both, the same three
engines:

| Profile | Passes | Ingest | Query, typical | Query, worst case | Run, worst case |
| --- | --- | ---: | ---: | ---: | ---: |
| `dev` | 1 | ~0.0 h | ~0.01 h | ~0.2 h | **~0.21 h** |
| `smoke` | 1 | ~0.3 h | ~0.07 h | ~8.7 h | **~9.01 h** |
| `quick` | 1 | ~36.2 h | ~2.69 h | ~36.4 h | **~72.61 h** |
| `main` | 1 | ~68.1 h | ~5.78 h | ~39.5 h | **~107.58 h** |
| `full` | 2 | ~532.0 h | ~79.33 h | ~473.6 h | **~1005.63 h** |
| `mariadb-blog` | 1 | ~21.3 h | ~1.69 h | ~7.6 h | **~28.95 h** |

The ingest column differs from the table above it because that one was computed
by hand from §1's three headline rates, while the estimator uses the measured
per-`(engine, dataset)` figures in `_MEASURED_ROWS_PER_S` and scales for corpus
size and dimensionality. The estimator is the live number; §8 explains why it
changes when someone measures a rate.

**Plan against the worst case.** A run longer than the printed worst case is a
bug, and that is the point of printing it. Two things to understand about what
it is:

* It is a bound over **the configuration, not the machine.** Where a cost is a
  budget — `ops.churn_budget_s`, two hours per churn step by default — the worst case
  is that budget, whatever the engine does inside it. Where it is a modelled
  duration, it is modelled from rates measured on one machine and yours will
  differ. The typical figure is what the configuration should cost; the worst
  case is what it cannot be made to exceed by an engine behaving badly.
* It is honest about its holes, and prints them. Filtered search runs a fixed
  number of queries at a rate nothing here has measured, so it is listed with
  a ceiling and **no typical duration** — `~0.0 h typical, ~7.0 h worst case`
  rather than one confident figure. The ceiling is real:
  `ops.filtered_budget_s` caps every (column, selectivity, iterative-scan
  mode) point at five minutes, the workload stops the query loop when it is
  spent, and the record says so. Until that budget existed this was the one
  term with no bound at all, and the run figure had to say in as many words
  that it was not yet a ceiling — so the "a run longer than its printed
  estimate is a bug" convention was false for every profile that measured
  filtered search, which is all of them but `m-sweep`. The recall sweep, the
  churn re-measurements and the concurrency warm-ups are still named under
  the total, because those are counted in queries and no rate exists to price
  them. A dataset with no measured ingest rate has always been reported this
  way; this is the same treatment for query time.

A run prints all of it before starting:

```
estimated ingest time (loading only, before any queries):
  mariadb ~17.5 h  |  alisql ~49.7 h  |  pgvector ~0.9 h
  total ~68.1 h across 1 pass(es), 3 M value(s)
  ! This is a long run. Each M value costs a full reload of every dataset,
    and MHNSW/VIDX build incrementally at a few hundred rows/s. Reduce
    ann.m_values or the dataset list to cut it roughly proportionally.

estimated query time (the parts the profile puts a clock on):
  concurrency windows    ~1.3 h
                           5 client count(s) x 30s x 3 repeat(s) + one warm-up window each per ops unit, fixed by the profile not by the engine
  filtered search        ~0.0 h typical, ~6.0 h worst case
                           e.g. 2 filter column(s) x 3 selectivities x 1 iterative-scan mode x 500 queries = 3,000 queries per ops unit, at a rate nothing here has measured; 300s cap/point bounds it
  churn re-insert        ~4.3 h typical, ~32.0 h worst case
                           e.g. 295,878 rows written back at ~147 rows/s (the load rate; re-insert can be far slower), 7200s cap/step
  churn reader baseline  ~0.1 h
                           2 churn fraction(s) x 30s of 4 reader(s) before each step's deletes, fixed by the profile not by the engine
  subtotal ~5.8 h typical, ~39.5 h worst case

estimated run time ~73.9 h typical, ~107.6 h worst case
  Plan against the worst case: a run materially longer than it is a bug.
  Each budgeted term tests its deadline before starting work, so it can overrun
    by the one query or slice already running when the budget expires.
  ! Neither figure includes the costs below. They are a number of queries, and
    there is no measured query rate for mariadb, alisql, pgvector to multiply them by:
      recall/QPS sweep: every ann.ef_search point, `runs` times, over each dataset's query set, plus a one-time index warm-up of up to 5,000 training-set queries per built index
      churn re-measurement: ops.max_queries before churn and again after each churn fraction
      churn delete: ops.churn_fractions of the corpus removed per step, with no budget bounding the delete itself
      churn repair path: on an engine that has one it can execute at its pinned version (pgvector's VACUUM (ANALYZE); Percona Server's index replace only while capabilities.vector_index_replace is true -- it is false at the pinned version, where the driver declares no repair path, runs nothing and pays neither cost), driver.maintain() per churn fraction plus a third ops.max_queries measurement of the repaired index -- the repair has no measured rate at any corpus size and nothing caps either
      concurrency warm-up: 50 queries per client before each measured window
      filtered ground truth: exact top-k recomputed by brute force over the qualifying rows of every (column, selectivity), over the whole test set rather than ops.max_queries and outside ops.filtered_budget_s; cached per (dataset, metric, k, column, threshold, rows) and shared across engines, ops units and passes, so paid once each -- measured at ~150 s per million qualifying rows (5-8% spread, near dimension-independent below ~256), which is ~6.7 min for main's two datasets x two columns
```

Read it. It is the cheapest part of the whole exercise.

The web UI shows the same breakdown on its Profiles & launch page — ingest, each
query term, and the worst-case total — updating as you change the engine list,
the dataset list or the resource pass, so you can find an affordable scope before
committing rather than after. It is the same calculation, not a second one: both
call `estimate_load_hours()` in `orchestrator/estimate.py`, so they cannot
disagree about the same plan. See [08-web-ui.md](08-web-ui.md).

### The ann grid, counted build by build

The ingest line above charges every engine one corpus reload per entry of
`ann.m_values`. ann-benchmarks reloads for every *build* -- every
(M, ef_construction, quantization) the rendered config asks for -- so an
engine with a second axis costs more than that line says: under
`tuned-complete` pgvector and Valkey build three indexes each, not one, and
the tuned pass's `m_values_by_engine` gives the MySQL-family engines two. The
published figures in this document and README are left on the old basis so
they stay comparable with each other; `run` now prints a second block beneath
them, from `orchestrator/estimate_grid.py`, that counts what the render asks
for and prices each load by its M where the load builds the graph
(MariaDB, AliSQL, Percona Server: `(M/16)^1.1` of the M=16 rate, from MariaDB's
glove-100 rates at M=8 and M=32). When it appears it also says by how much
the run time above is low and what to plan against instead, names the axis
that multiplies each engine's builds (the headline's "Reduce ann.m_values"
advice is withheld then, because the multiplier may be ef_construction or
quantization), and lists what neither figure prices: the bulk index builds of
the engines that build after the load, the recall/QPS sweep, and any engine
with no measured ingest rate.

## 2a. Exploring the whole parameter space: the two-stage frontier

A reviewer of `full-20260825` asked for the frontier over M = 4, 6, 8, 12, 16,
32, 48 on every engine, with ef_construction and quantization swept where the
engine has them. At 990,000 rows that grid is not affordable: counted by the
block above it is **~267 h of ann loads** -- 7 MariaDB / 12.3 / AliSQL builds
at 31, 45 and 53 h, and 21 Percona Server builds at ~129 h -- before a single
query, and before tuned-complete's ops phase. So it is done in two stages.

**Stage 1 -- explore at 90,000 rows** (`config/profiles/frontier-explore.yml`).
Every engine with a graph-degree knob sweeps the reviewer's seven M values;
pgvector, Valkey, Percona Server, MyVector and VillageSQL also sweep
ef_construction 64/200/400; Percona Search sweeps quantization scalar/none at
one M label (mongot exposes no graph degree here -- the profile header says
why `maxEdges` is not used). Seven builds are left out by name, each printed
before the run (`orchestrator/ann_knobs.py`) and never clamped: pgvector
refuses ef_construction < 2·M, so M=48 at 64 is dropped, and vsql_vector
refuses ef_construction > 32·M, so VillageSQL loses 200 and 400 at M=4 and 6
and 400 at M=8 and 12. VillageSQL is opt-in
(`runtime.default_selected: false`): a bare run leaves it out and says so, and
only `--engines ...,villagesql` measures it. `runs: 2`, because the
selection is made on QPS too and a single cold draw can push a build off the
frontier.

| Engine | Builds | Ann loads, estimated |
| --- | ---: | ---: |
| `alisql` | 7 | ~6.0 h |
| `percona` | 21 | ~3.6 h |
| `mariadb123` | 7 | ~3.0 h |
| `mariadb` | 7 | ~2.2 h |
| `pgvector` | 20 | ~0.3 h |
| `valkey`, `mongodb` | 21, 2 | minutes |
| `myvector` | 21 | no measured rate |
| `villagesql` (opt-in) | 15 | no measured rate |

About **15 h of loads** plus the queries, which no rate prices (the recall
sweep is named under the total as always); the bulk index builds of the
post-load engines are in neither figure.

```bash
./scripts/generate-dataset.sh dbpedia-openai-100k-angular
./run-benchmark.sh run --profile frontier-explore
```

**Selection** (`select-builds`, `orchestrator/build_selection.py`). Reads the
stage-1 run's `report/records.jsonl` -- run `report --run-dir` first if it
was skipped -- and per engine, over every point that used the vector index:

* the **winner** at each floor 0.90 / 0.95 / 0.99: the build owning the
  frontier point with the smallest recall at or above it;
* every build owning a frontier point **near** a floor, defined as
  |recall − floor| ≤ 0.015 -- the most recall at fixed M moved between 90,000
  and 990,000 rows (M=16, 0.014), so a build that close can be on the other
  side of the floor at full size;
* for a floor nothing reaches, the build with the highest recall, recorded as
  an unreached floor;
* the **neighbouring M** below and above each of those, at the same
  ef_construction and quantization, so stage 2 can confirm the choice
  transfers. A kept M=4 or M=48 has one neighbour and an edge note: the
  optimum may lie outside the explored grid. Where the next M exists but the
  engine cannot build it at that ef_construction (pgvector M=48 at 64), the
  note says so instead. A neighbour stage 1 has no point for -- a failed
  build, or one that fell back to a scan -- is **dropped**, with that reason:
  there is nothing to confirm a transfer against, and a build that failed at
  90,000 rows is hours lost at 990,000;
* the target profile's **reference build**, always (`run` refuses a grid
  without it).

Everything else is pruned, with its reason. **The overlay names the kept
builds exactly** (`ann_builds_by_engine`), so there is no closure: per-axis
lists alone would be multiplied out by `render_config`, and on Percona Server
three chosen builds -- M=12 at 64, 16 at 200, 32 at 400 -- would come back as
nine, ~60 h of loads at 990,000 rows instead of ~20 h.

```bash
./run-benchmark.sh select-builds --from results/<stage-1 run id> \
    --out state/frontier-stage2.yml
```

It refuses to write an overlay whose rendered grid is not exactly the kept
set, or one `run` would refuse, and refuses a stage 1 that swept more than one
build mode or MariaDB storage engine (the selection cannot tell them apart).

**Stage 2 -- re-measure at 990,000 rows.** The overlay is applied to
`tuned-complete` (`orchestrator/profile_overlay.py`): it may set only the
build-axis keys of `resources.extras`, each checked at load against the
engine's legal range and enumeration; it merges the way a profile's
`resources:` block always has; it must carry its `build_selection` block,
whose `overlay_sha256` covers both the grid and the provenance, so neither can
be edited without the other; and it is refused on the normalized pass, which
ignores those keys by design.

```bash
./run-benchmark.sh run --profile tuned-complete --resource-pass tuned \
    --overlay state/frontier-stage2.yml
```

What it costs depends on what stage 1 keeps, and the run prints it. For
scale: three M values per engine at one ef_construction (say 12/16/32) is
**~82 h** of ann loads, against ~22 h for tuned-complete's own grid and
~267 h for the whole space -- most of it MariaDB 12.3 and AliSQL at M=32.
The stage-2 manifest records `build_selection`: the stage-1 run id and path,
the grid explored, every kept and pruned build with its reasons, and every
stage-1 (build, ef_search, recall, QPS) of the kept builds, so a later report
can flag any pair whose recall moved by more than 0.02 between the stages.
A resume must pass the same overlay; a different one, or none, is refused.

---

## 3. Choosing a scope

Measured options, all three engines. **Ingest only** — add query time from the
second table in §2, which the estimator prints for whatever scope you actually
ask for.

| Scope | Ingest |
| --- | ---: |
| `--profile full --resource-pass both` | 272 h |
| `--profile main` | 61 h |
| `--profile main --datasets glove-100-angular` | **33 h** |
| `--profile main --datasets glove-100-angular --phases ops` | 8 h |

If you run one thing, run this:

```bash
./run-benchmark.sh run --profile main \
  --datasets glove-100-angular --resource-pass normalized
```

Roughly a day, and it produces a complete three-way comparison at 1M scale
across every measured dimension.

### What each cut costs you

Cuts are not equivalent. What you lose matters as much as what you save.

| Cut | Saves | Costs you |
| --- | --- | --- |
| Drop the `tuned` pass | ~50% | Whether tuning changes the ranking. On the smoke run it changed little except AliSQL's concurrency (0.27 → 0.65 efficiency) — measurable separately at one M in about an hour. **Usually the best first cut.** |
| Drop `gist-960` | ~40% of `full` | The high-dimensionality data point. 960 dims is where SIMD and cache behaviour diverge most, so this is a real loss if that is your workload. |
| Drop M=32 | ~65% of the M grid | The high-recall end of the Pareto frontier — the part that matters for production accuracy targets. **Cut the middle value instead.** |
| Drop 1% selectivity | ~37 min per configuration | The hardest filtered case. MariaDB measured 0.9 QPS there against 326 unfiltered — a 360× penalty that is itself a finding. Keep it for a targeted run, drop it from broad sweeps. |
| Fewer `ops.m_values` | proportional | Little. Build cost, concurrency and churn describe engine behaviour, not index-parameter sensitivity. One M value is usually enough. |
| Fewer `ann.ef_search` points | small | Little — ef_search is swept without reloading, so extra points are cheap. This is the wrong place to economise. |

The last two lines are the important ones: **sweeping `ef_search` is nearly
free, sweeping `M` is not.** Extra ef_search points reuse the built index;
every extra M rebuilds it.

---

## 4. Watching a run

Every long phase reports rate and ETA every 20 seconds
(`VB_PROGRESS_INTERVAL` overrides it):

```
[mariadb]   412,500/1,183,514 rows, 148 rows/s, ETA 86.8 min
[mariadb]   filtered 10% queries 340/500, 34.1/s, ETA 0.1 min
[pgvector]  bulk index build — still running, 120s elapsed
```

Operations that cannot report a fraction — a bulk `CREATE INDEX`, a NumPy
brute-force ground-truth pass — emit a heartbeat instead, so silence always
means something is wrong.

If you are running an older checkout without this, poll the server directly:

```bash
C=$(docker ps --format '{{.Names}}' | grep srv)
M="/opt/mariadb/bin/mariadb -ubench -pbench --socket=/var/run/vbench/mariadb.sock -N -e"
prev=$(docker exec $C $M "SELECT COUNT(*) FROM ann.t1")
while sleep 60; do
  n=$(docker exec $C $M "SELECT COUNT(*) FROM ann.t1") || break
  r=$(( (n - prev) / 60 )); prev=$n
  echo "$(date +%T)  $n rows  ${r} rows/s"
done
```

A count that stops advancing for several minutes is stuck. A count that keeps
moving is not, however slow it looks.

---

## 4a. Before you start: disk

A run needs roughly `corpus size x 2.2 + 10 GB`. The table and the index each
come out close to the size of the raw vectors, and the MySQL family also writes
a redo log and a doublewrite buffer. On dbpedia-openai-1000k that is a 6.2 GB
corpus producing a 7.7 GB table and a 3.9 GB index per engine.

Engines run one at a time and their volumes are removed after each, so the peak
is one engine's footprint rather than the sum of all three.

### Which filesystem

Two of them matter, and on a benchmark box they are usually different mounts.

Engine data directories and ops volumes go under `$VB_ROOT/state`, so they land
on whatever filesystem the checkout is on. Images, and anything else Docker
stores, stay under the daemon's data-root (`docker info -f '{{.DockerRootDir}}'`,
normally `/var/lib/docker`).

This used to be worse. The ann containers ran the engine with no data mount at
all, so the database wrote into the container's own writable layer — under
Docker's data-root, on the root volume. A pgvector phase died at

```
initdb: error: could not create directory "/var/lib/postgresql/data/pg_wal": No space left on device
```

while the filesystem holding the checkout had over 100 GB free. If your root
volume is small, that is now handled: both paths write under `$VB_ROOT`. If you
would rather move Docker itself, set `data-root` in `/etc/docker/daemon.json`
and restart the daemon.

The run checks the tighter of the two filesystems before starting and refuses if
it looks too tight. It is
worth having: a pgvector phase once died two seconds in because the volume had
nowhere to put the cluster, and it read as an engine failure for a while before
anyone thought to check `df`.

## 5. Interruption and resumption

Work is checkpointed at two levels:

- per `(resource pass, engine, dataset, phase)`
- and within an ops phase, per `(M, build_mode)`

The second matters: an ops phase can run a dozen hours across several M values,
and checkpointing only the whole phase meant an interruption discarded every M
that had already finished.

```bash
./run-benchmark.sh run --profile main --resume --run-id main-<timestamp>
```

Completed units are skipped. Failed units are not checkpointed, so `--resume`
retries exactly those.

ann-benchmarks additionally skips result files it has already produced, so an
interrupted recall sweep resumes at configuration granularity without any help
from the checkpoint file.

That skip is keyed on the algorithm and its index parameters, and knows nothing
about the resource budget. So recall results are stored under
`results/annb/<pass>/<config fingerprint>/`, where the fingerprint covers the
memory budget, the cache split, the build threads, the core count and a manual
measurement version.

Change the budget and you get a new tree and a fresh measurement. Change
nothing and resumption works exactly as before. Without this, a dbpedia run
re-launched at 64 GB returned every recall point byte-identical to the earlier
16 GB attempt, because ann-benchmarks found the files already on disk and
reported success in under a second. The report then carried a manifest saying
64 GB above a curve measured at 16.

If you need to recompute a tree that is genuinely current, `--force` does it.
Any result file older than the run reporting it is now flagged in the report's
Validity section.

---

## 6. Reproducing MariaDB's published benchmark

MariaDB's [big vector search benchmark](https://mariadb.org/big-vector-search-benchmark-10-databases-comparison/)
used **dbpedia-openai-1000k** — one million DBpedia texts as 1536-dimensional
OpenAI embeddings, angular distance — run through their fork of ann-benchmarks,
which is what this framework is built on.

That dataset is **not** published as a prebuilt HDF5, so it has to be built
locally from HuggingFace:

```bash
./scripts/generate-dataset.sh dbpedia-openai-1000k-angular   # multi-GB download,
                                                             # hours, ~20 GB working space
./run-benchmark.sh run --profile mariadb-blog
```

Ground truth is computed by brute force inside the generator, which is the slow
part.

**The download is the same size whichever variant you pick.** ann-benchmarks
fetches the full 1M-row HuggingFace dataset and then selects the first N rows,
so `dbpedia-openai-100k-angular` downloads exactly as much as the 1000k one — it
only shrinks the ground-truth computation and every subsequent engine load.
There is no saving in starting small except in run time.

### What will not match, and why

| Difference | Effect |
| --- | --- |
| Their CPU (Xeon E5-2660 v4) has **no AVX-512** | If yours does, MHNSW and VIDX take wider SIMD paths and will look faster than published. The largest single source of divergence, and not configurable away — building `--march x86-64-v3` gets closer at the cost of not measuring your own hardware. |
| They compared ten systems | This framework covers MariaDB, AliSQL and pgvector. The other seven are out of scope. |
| Versions | MariaDB 11.8.8 here vs their 11.8 **and** 12.3; pgvector 0.8.6 vs their master of 2026-02-08. |
| Index parameters were not published | The M and ef_search grids in the profile are ann-benchmarks' conventional ranges, not a claim to match theirs. |

### Measured: what a full 1M x 1536 run actually cost

The first complete `mariadb-blog` run, on a Xeon Gold 6230 (40 physical cores,
187 GB RAM, AVX-512, `--march=native`), 990,000 x 1536, M=16, normalized pass:

| Engine | Ingest | Index | Table | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| MariaDB | 83.8 rows/s, 3.3 h | 3.88 GB | 7.69 GB | 8.25 GB |
| AliSQL | 41.4 rows/s, 6.6 h | 3.87 GB | 7.66 GB | 16.0 GB (at the limit) |
| pgvector | — | — | — | 16.0 GB, OOM-killed |

Total wall clock for the run was 28.4 hours, about 6 of them wasted on the
pgvector phases before they died.

**The 16 GB normalized budget was the problem.** The table alone is 7.7 GB and
the index 3.9 GB. MariaDB fitted with headroom, AliSQL spent 55% of its phase
pinned at the ceiling reclaiming continuously, and pgvector was OOM-killed 42%
of the way in. A pass where only one engine fits measures which engine fits,
not which is faster.

The profile now asks for 64 GB through a `resources:` block, and the report
flags any engine that ran against its limit in the Validity section. If you add
a corpus of this size, set its budget deliberately:

```yaml
resources:
  memory:
    server_limit_gb: 64
```

### Memory, which is the part that surprises people

In the recall/QPS pass the database server and the ann-benchmarks client run in
the **same container**, so they share one cgroup limit. The client is not a
thin driver: ann-benchmarks holds the entire corpus in RAM **twice** — `main.py`
loads it once to read the dimension, and the forked worker loads it again for
itself. Neither copy is shared.

For dbpedia-openai-1000k that is 6.1 GB of corpus, so ~13 GB of client, on top
of whatever the engine is given:

| Corpus | File | Client needs | Container (with a 16 GB engine budget) |
| --- | ---: | ---: | ---: |
| fashion-mnist-784 | 0.2 GB | 1.5 GB | 17.5 GB |
| glove-100 | 0.5 GB | 2.0 GB | 18.0 GB |
| dbpedia-openai-1000k | 6.2 GB | 14.6 GB | **30.6 GB** |

The harness sizes the container as *engine budget + client estimate* and prints
both, so the engine's own budget stays identical across engines — which is what
makes the normalized pass mean anything — while the client's copies of the
corpus are added on top rather than taken out of the engine's share.

Plan for **~32 GB of free RAM** for a 1536-dimensional run, or lower
`memory.server_limit_gb` in the resource profile.

> Earlier versions sized the container to the engine budget alone. On
> dbpedia-openai-1000k the client was OOM-killed seconds after loading, and
> because only the forked worker died, ann-benchmarks exited 0 having written
> nothing — it does not check worker exit codes. If you see `Terminating 1
> workers` immediately after `Got a train set of size ...` and no results,
> that is what happened. Check `dmesg -T | tail`.

At 1536 dimensions this is the heaviest run the framework supports — roughly 5×
the per-row cost of glove-100. Estimated ingest:

| Engine | Ingest |
| --- | ---: |
| pgvector | ~0.5 h |
| MariaDB | ~3.3 h (measured) |
| AliSQL | ~6.7 h (measured) |
| **all three** | **~10.5 h** |

An earlier version of this table said 19 h and 52 h for MariaDB and AliSQL. Those
came from the formula, which overcharges for dimensionality once per-row cost is
dominated by graph traversal rather than by the distance computation. The
measured rates are now in `_MEASURED_ROWS_PER_S`.

AliSQL is 72% of that, which makes it the obvious thing to drop. Resist it:
MariaDB versus AliSQL is the comparison this framework was built for, and the
article's framing — MariaDB against everything else — is a different question.
If three days is too long, **cut the corpus rather than the engine list**:
`dbpedia-openai-500k-angular` and `-100k-angular` are the same embeddings at
smaller scale and keep all three engines in the picture.

---

## 7. Sizing a new dataset

Before adding a dataset, estimate its load:

```
hours ≈ rows / rate / 3600 × (M values) × (phases) × (passes) × (engines)
```

with `rate` from §1 — and remember it falls as the graph grows and falls again
with M. For anything near 10M vectors (`deep-image-96`), assume days per engine
for the incremental builders and plan a dedicated run rather than folding it
into a sweep.

---

## 8. A note on the numbers in this document

The ingest rates here come from real runs on a dual-socket Xeon Gold 6230 with
AVX-512, engines built `--march=native`. They will differ on your hardware,
possibly a lot: a machine without AVX-512 runs the distance kernels on narrower
paths, and MHNSW and VIDX both document AVX-512 support.

An earlier version of the estimator used rates guessed from a 60,000-row smoke
run and was optimistic by 2.7× against the first real 1.18M-row load. That is
worse than having no estimate at all, because it invites you to start something
you would not have started. If your measured rates diverge from these, update
`_INGEST_ROWS_PER_S` in `orchestrator/estimate.py` — the estimate is only as honest
as the numbers behind it.
