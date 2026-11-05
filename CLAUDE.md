# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`vector-bench` benchmarks **built-in vector search across ten database engines**, all
running HNSW, so the comparison isolates implementation quality rather than algorithm
choice. It is a measurement instrument: the most damaging bugs here do not crash — they
quietly hand one engine a different budget, grid or dataset and produce a credible-looking
report that is wrong.

The repo root is `vector-bench/` (git repo, branch `master`). All commands below run from there.

| Engine key | What it is | Source pin |
| --- | --- | --- |
| `mariadb` | MariaDB MHNSW | tag `mariadb-11.8.8` |
| `mariadb123` | MariaDB MHNSW, second version | tag `mariadb-12.3.2` |
| `alisql` | AliSQL VIDX | tag `AliSQL-8.0.44-2` |
| `pgvector` | PostgreSQL + pgvector | tag `v0.8.6` |
| `mongodb` | Percona Search for MongoDB (mongot) | Percona package |
| `valkey` | Valkey + valkey-search | Percona repo `valkey-91` |
| `percona` | Percona Server, InnoDB HNSW | `9.7` series, commit `56124b03` |
| `myvector` | MyVector plugin (hnswlib) on MySQL 9.7.2 | release v1.26.9, commit `384d9df7` |
| `villagesql` | VillageSQL 0.0.7 + vsql_vector extension (HNSW custom index, built from vsql-vector-pk), MySQL 9.7.2 base | server `publish/mysql-9.7_0.0.7` (`533ab4b5`), vsql-vector-pk release `0.0.3` (`93ed0087`) |
| `villagesql84` | The same extension commit on VillageSQL's MySQL 8.4 base (`alias_of: villagesql`) | server `publish/mysql-8.4_0.0.7` (`689c74a9`), vsql-vector-pk release `0.0.3` (`93ed0087`) |

`KNOWN_ENGINES` is whatever `config/engines/*.yml` holds — there is no list in Python to keep
in step with it. `orchestrator/engines.py` is the registry everything derives from.

## Commands

```bash
python3 -m pytest tests/ -q                 # thousands of tests, ~6 min. The fast inner loop.
./tests/make-tiny-dataset.sh                # synthetic corpus, no download
./run-benchmark.sh run --profile dev        # ~1 min/engine end-to-end cycle
./run-benchmark.sh run --profile smoke      # ~45 min/pass gate before any long run
./run-benchmark.sh report --run-dir results/<run-id> [--datasets glove-100-angular]
./run-benchmark.sh render --profile <p>     # regenerate ann-benchmarks configs only
./run-benchmark.sh build --engines alisql --march native
./run-benchmark.sh clean --run-id <run-id>
./run-benchmark.sh web [--allow-control]    # web UI; --no-container for host mode
./run-benchmark.sh generate <dataset>       # datasets fetch cannot download
./run-benchmark.sh lab --engine valkey probe-valkey-filter.py   # diagnose, don't measure
./run-benchmark.sh export --run-dir <id>    # bundle a run to send to someone
```

Never pipe pytest through `tail`/`head` in a command whose exit status matters — a broken
commit has already gone out that way, because the pipeline swallowed the non-zero exit.

**That first figure is deliberately vague.** It read "~1,700 tests, ~70 s" for a long time
after the suite had passed 3,600 tests and six minutes, because an exact count is wrong again
the next time anyone adds a test, and a count nobody maintains is worse than no count: it
tells a reader the run they are watching has hung. Run `--collect-only` for the real number
when you need it, and if you change this line, keep it to a magnitude.

Host requirements are deliberately only `docker`, `python3`, `pyyaml`, `git`, `curl`. Do not
add a host-side Python dependency: a benchmark framework must not modify the machine it
measures. Everything heavy (numpy, h5py, matplotlib, DB clients) lives inside the images.

**That constraint binds the test suite too.** `report/charts.py` guards its matplotlib import
and `_new_axes()` refuses loudly if it is missing, so `report.generate`'s pure-logic helpers
stay importable on a bare host; the dozen tests that actually render are marked
`@needs_matplotlib` and skip. A test that fails on a machine set up from the documented
prerequisites makes a correct install look broken — which is exactly what happened.

## Architecture

Two measurement paths converge on **one flat record schema** (`harness/metrics/records.py`,
`SCHEMA_VERSION`, `PHASE_*`). Every chart routes off `phase`; unused fields stay `None`.

- **Path A — recall/QPS.** ann-benchmarks, used *unmodified*, run as `run.py --local` inside
  each engine's `*-bench` image (`orchestrator/ann_pass.py`). The orchestrator launches the
  container so it owns cpuset/memory/env; ann-benchmarks keeps ownership of definitions,
  ground truth and recall computation. Our algorithm modules live in
  `overlay/ann-benchmarks/` and are copied over a disposable clone at `work/ann-benchmarks/`.
- **Path B — ops harness.** `harness/` runs in a *second* container talking to the server
  container over a private network (`orchestrator/ops_pass.py`). Two containers, not one, so
  the server's cgroup accounting measures the server and not the several hundred MB of NumPy
  holding the dataset. Produces build cost, concurrency, filtered search, churn.

The harness never starts or configures a server — the orchestrator does, so every engine gets
identical treatment and that treatment lands in the manifest.

**Web UI** (`webui/`, `docs/08-web-ui.md`). Covers every long command — `run`, `fetch`,
`build`, `generate`, `report`, `render` — through one job supervisor (`webui/jobs.py`), with
live log streaming. Nothing may run alongside a benchmark; setup jobs may overlap each other. Stdlib-only HTTP server + vanilla-JS front end,
served from `vector-bench/webui` (ubuntu + python3/pyyaml/numpy/git + the Docker CLI). Read-only
by default; `--allow-control` adds profile editing and run launching and mounts `docker.sock`.
It reads `report/records.jsonl` when a report exists and falls back to `load_ops_records()`.
The repo is bind-mounted **at its own absolute host path** — the orchestrator hands host paths
to the Docker daemon, so a container-only path would mount nothing on the host.

**Three config layers** resolve into one plan (`orchestrator/config.py`):
`config/profiles/<p>.yml` (what to measure) × `config/resources/{normalized,tuned}.yml`
(how much machine) × `config/engines/<e>.yml` (how to build, start and speak to it).
Resolution turns fractions into concrete byte counts and cpusets before they are recorded.

**`run-manifest.json` is mandatory.** The report generator refuses a run directory without
one — a result without its environment is not a result.

## Editing without rebuilding

`orchestrator/` and `config/` run on the host; `harness/` and `report/` are **bind-mounted
read-only into the containers at run time** (`/opt/harness`, `/vb/report`). Images carry the
third-party Python stack but none of this repo's own code, so edits take effect on the next
command with no rebuild:

| Changed | What is needed |
| --- | --- |
| `harness/`, `report/`, `orchestrator/`, `config/` | nothing — just re-run |
| `overlay/ann-benchmarks/` | nothing — `run` and `render` re-apply the overlay onto `work/` every time |
| `docker/*/Dockerfile`, entrypoints, `docker/_shared/requirements-*.txt` | `./run-benchmark.sh build --engines <e>` |
| engine `source.tag` | `./run-benchmark.sh build` (AliSQL is 1.5–3 h), then move that engine's ann subtree aside — see the ann results tree invariant below |
| `webui/` (server or static) | nothing in `--no-container` mode; the repo is mounted, so container mode also just needs a restart |
| `docker/webui/Dockerfile` | `./scripts/build-images.sh --engine webui` (~1 min) |

`sources/`, `work/`, `datasets/`, `results/`, `state/` are generated and gitignored. The
vendor repositories (`$VB_REPO_MARIADB`, `$VB_REPO_ALISQL`, `$VB_REPO_ANNB`) are **read-only
inputs** — never check out, fetch into or otherwise modify them.

## Adding or changing an engine

An engine is not one file. Touch all of these, or a run gets as far as starting a server and
then dies on `argument --engine: invalid choice`:

**A variant of a family the harness already drives is config only** — a second MariaDB, a
Percona Server, another Postgres build. Write `config/engines/<e>.yml` (the `runtime:` block
carries driver, ann_constructor, port, mounts, credentials, probe, chart colour, label and the
`reference_build` — the index the ops pass builds, which §4's secondary "at the build the ops
pass measured" table is taken from and which a headline frontier figure from any other build is
marked against; under `ops.build_selection: frontier` it is only the fallback, and the ops pass
builds the swept build the ann phase's frontier is fastest with instead
(`orchestrator/ops_build_selection.py`); `run` refuses one the ann grid does not contain),
build the image, done. The Engines tab in the web UI does exactly this.

**A new architecture needs code:**

```
config/engines/<e>.yml                                  runtime: block + build + server flags
harness/drivers/<e>.py                                  + register in _driver_classes()
                                                        (harness/drivers/postgres.py)
overlay/ann-benchmarks/ann_benchmarks/algorithms/<e>/   module.py + config.yml
docker/<e>/Dockerfile + entrypoint-<e>.sh               runtime + bench targets
scripts/build-images.sh, scripts/prepare-sources.sh
tests/test_engine_registry.py                           add to GOLDEN
```

`orchestrator/engines.py` is the registry; `KNOWN_ENGINES`, chart styles and the driver
lookup all derive from it. `_driver_classes()` maps driver *class name* to class — one entry
per architecture, not per engine, which is what makes a variant configuration rather than code.

**Two collisions the validator refuses, both of which produce plausible-but-wrong configs:**
a duplicate `ann_constructor` (ann-benchmarks keys result files on it, so the second engine
reports the first's recall — this is why `mariadb123` exists) and a duplicate image tag.

## Invariants that fail silently

These are the reasons the code looks over-defensive. Do not relax them.

- **Never mix `-march` between engines.** The eight built from source compile SIMD distance
  kernels — `mariadb`, `mariadb123`, `alisql`, `pgvector`, `percona`, `myvector` (whose
  kernels are hnswlib's, inside the plugin), `villagesql` and `villagesql84` (whose kernels
  are in the `vsql_vector.veb` extension, compiled Release and checked for -O3), all at
  `x86-64-v3`;
  `mongodb` and `valkey` are installed from packages and declare `march: none`. A different
  flag on one turns the benchmark into a comparison of compiler flags.
- **`harness/` and `report/` must not import `orchestrator`.** Each runs in a container that
  mounts only itself. Importing the orchestrator from `report/` once crashed generation at the
  end of a 20-hour run; the same mistake in `harness/` kills a run right after the server comes
  up. Anything they need is decided on the host and passed in — `--driver` for the harness, the
  manifest's `engines.<name>.presentation` for the report. Both are enforced by tests.
- **Every engine can fall back to a full table scan** — exact results, slowly, indistinguishable
  in output from "very accurate but slow". Every driver implements
  `explain_uses_vector_index()` and it is called per configuration; anything that missed the
  index goes into the report's Validity section. The marker lives in the driver
  (`Dialect.index_name` for the MySQL family) and nowhere else. `config/engines/*.yml` once
  carried an `sql:` block naming it too, alongside a `create_table` and a `query_filtered`,
  and this bullet used to describe that key as "documentation only and read by nothing".
  Read by nothing it was — and **also wrong**: `mariadb.yml` and `mariadb123.yml` declared
  `index_plan_marker: "v"` where the driver names the index `vi`, and the real check is a
  substring test against the joined EXPLAIN text, so `"v"` would have accepted almost any
  plan. The rest of the block had drifted the same way, for the same reason: a `create_table`
  with no `tag` column beside its own `tag` filter, and then no mention of `ctag` while every
  live driver created it. The blocks are gone and `tests/test_c1_engine_sql.py` fails if one
  returns.
- **The ann results tree at `results/annb/<pass>/<fingerprint>/` is shared across runs.**
  ann-benchmarks caches by algorithm and index parameters only — it has no idea how much
  memory or how many cores the engine had. `ann_fingerprint()` keys the tree by the resource
  pass's *declared* knobs so a 16 GB curve is never silently reused under a 64 GB manifest.
  The fingerprint is deliberately **engine-invariant**; making it engine-specific fragments the
  tree and produces a seven-engine recall chart containing one engine. It also holds **no
  source tag**, so a re-pin does not move the tree: once an engine's `source.tag` changes,
  ann-benchmarks skips every configuration whose result file is already on disk and the old
  binary's recall cells are reused under the new pin's `engine_version`. `ann_pass.run_engine`
  does print `[ann] N existing result file(s) … will be skipped` before the run — "Warn before
  the fact, not after" — and `report/generate.py`'s `stale_ann` bucket adds the per-cell
  attribution in the Validity section afterwards. A fresh `--run-id` does **not** fix it: the
  run directory holds `ops_results`, while `annb_results` is `results/annb` beside it, so a new
  run directory leaves every recall cell where it was. **Move that engine's subtree of the ann
  tree aside before the first run at a new pin** — `results/annb/<pass>/<fingerprint>/<dataset>/<k>/<engine>/`
  — which makes `_count_results` return 0 and the curve get measured again. **Do not reach for
  `--force` instead**, even though its help offers exactly that: the same flag is the escape
  hatch on `_check_free_disk`, so it silently disables the free-disk refusal, which is the one
  a core-dump campaign most needs. Bumping `ANN_MEASUREMENT_VERSION` is not the remedy either:
  it moves every pass's tree and re-measures every recall point for all seven engines.
- **Filtered ground truth is recomputed** (`harness/datasets.py`) — a `WHERE` predicate changes
  the correct answer set, and reusing unfiltered neighbours scores every engine against the
  wrong targets.
- **Recall is ann-benchmarks' distance-threshold definition**, not an id intersection
  (`report/loaders.py:knn_recall`), so numbers stay comparable to published results. The
  ops path uses the same rule since ops measurement version 9 (`harness/datasets.py`
  `recall_threshold`); before it, ops recall was an id overlap under the same name. Every ops
  record says which in `extra.recall_kind`, and `report/crosscheck.py` compares only matching
  kinds.
- **MariaDB and AliSQL ship a 16 MiB graph cache by default** (`mhnsw_max_cache_size`,
  `vidx_hnsw_cache_size`). The framework sets both from the resource pass; leaving the default
  judges a vendor on a value it plainly intended you to change.
- **Percona's `innodb_hnsw_max_memory` is NOT a third member of that family** and must never be
  fed from `graph_cache_bytes`. Those two are caches — a graph larger than the cache still
  works and performance degrades. Percona's is a hard allocation ceiling with no per-node
  eviction: exceeding it fails the statement with `ER_CAPACITY_EXCEEDED`, and not only during
  ingest, because the limit is re-checked as a cold search faults the graph back in after a
  restart. Equal byte counts would therefore compare a cache budget against a wall, and one
  engine would be unable to finish the load where two merely slowed down. It is set to `0`
  (unlimited) so the container's memory limit bounds it like every other engine, and
  `Innodb_hnsw_memory_used` is recorded **on the `index_build` record only** — not on
  `ingest`, `concurrency`, `filtered`, `churn` or `ef_calibration`, so the audit trail for
  that gauge covers one of the six ops phases. Server-side *wait* evidence is a separate
  collector and reaches further: `concurrency` records and the churn `during_writes` record
  carry `extra.server_waits` (the engine's own wait counters, read on a side connection
  before and after each window — see `harness/workloads/server_waits.py`), so the phases
  with no server status of any kind are `ingest`, `filtered` and `ef_calibration`.
  **pgvector is the exception**: it has no cumulative wait counters, so its profile exists
  only with `ops.server_wait_sampler: true`, which samples `pg_stat_activity` *during* the
  window and so perturbs pgvector's window and no other engine's; off by default, stated on
  every record when on — and
  the gauge is a figure that
  **omits the node map
  (~30–50 B/node) and so is a floor on graph memory, not a measurement of it**. See
  `docs/09-adding-percona-server.md` §3.
  **The graph is cold after a build** — it is faulted in lazily, and the `post` build's ALTER
  evicts the table (server team) — so the gauge read 0 after 95 of 95 `post` builds. Percona
  alone (`warm_graph_after_build`) gets one discarded pass over calibration's queries at the
  top ef inside `build.run`, in both build modes, before anything is timed; the re-read gauge
  is `extra.graph_warmup.server_status` on that same `index_build` record (ops version 11).
  Per the server team, `OPTIMIZE TABLE` holds two graphs at its peak and the gauge shows
  neither afterwards, so a finite limit would need ~2× the graph; whether the churn index
  replace has the same peak is an open question (§3).
- **A profile's `resources:` block deep-merges dict-into-dict.** To *clear* an inherited key
  write `null`, not `{}` — `{}` recurses into the inherited map and changes nothing. This
  silently halved the `m-sweep` profile once.
- Structural asymmetries no config removes, and which the report always states:
  `ef_construction` is exposed by pgvector, Valkey, Percona Server and MyVector, and by no
  other MySQL-family engine here — Percona and MyVector are the MySQL-family engines with both
  a build-time and a query-time effort knob; pgvector bulk-builds while MHNSW and VIDX build
  incrementally on every INSERT, and Percona Server does either, so its `build_mode` is a
  comparison within one engine (ops path only — the ann overlay module has no `build_mode`,
  and `render_config` refuses the key rather than emitting two identical curves). Its ops
  default is `post` (`capabilities.bulk_vector_index_build: true` since the 9.7.2-2 re-test;
  it killed the server at 9.7.1-1) while every ann curve is an incremental build, and the
  churn index replace stays withdrawn on its own key, `capabilities.vector_index_replace:
  false` — a different ALTER, a 1–4 ms no-op at the old pin, not re-tested; MyVector
  only bulk-builds after the load, because its one write-path mechanism (a binlog listener)
  exits after its first idle second on this server; AliSQL VIDX is InnoDB-only and requires
  READ COMMITTED. MyVector has **no filtered search** at its pin and its driver declines that
  workload (`EngineDriver.declined_workloads`); churn re-inserts reach its graph through an
  explicit `myvector_index_refresh`, and deleted rows never leave it until a full rebuild —
  see `docs/04-engine-notes.md`.
- **VillageSQL's index is used only because the extension is vsql-vector-pk, and the code
  checks.** The server drives its KNN distance scan only from a custom index that stores row
  references (HAS_ROW_REF). Plain vsql-vector's HNSW stores column references, so on the
  engine's first pin every `ORDER BY L2_DISTANCE(v, ..) LIMIT k` was a filesort over a full
  scan with recall 1.000; vsql-vector-pk's declares HAS_ROW_REF and plans as "Custom index
  distance scan on vi". `VillageSQLDriver` and the ann module check EXPLAIN FORMAT=TREE for
  that and raise rather than time a query without it. DELETE is rejected on a table with a
  custom index (churn declined), a WHERE beside the KNN ORDER BY is a post-filter (filtered
  declined), vectors are text literals (the query vector inlined, a constant the KNN
  recognition needs), and the bulk `CREATE INDEX` can crash the server, so
  `capabilities.bulk_vector_index_build: false` makes `incremental` its build mode. vsql_vector
  refuses `ef_construction` outside `[M, 32 x M]`, so 200 is unbuildable at M=6: the
  combination is dropped from the render by name (`ann_knobs.excluded_builds`), never clamped
  under a label it was not built with, and `m-sweep` gives it M 8..32. It is **opt-in**
  (`runtime.default_selected: false`, read through the registry as `cli.DEFAULT_ENGINES`): a
  run that names no engines leaves it out and prints so. **`villagesql84` is the same extension
  commit on the MySQL 8.4 server line and behaves the same**: 8.4's `FindCustomKnnIndexOnField`
  carries the identical HAS_ROW_REF gate, so everything in this bullet applies to it, and it is
  opt-in too. Its own `ann_constructor` (`VillageSQL84`) and image names keep it from reading
  the 9.7 engine's result files or overwriting its image, whose tag is the same extension
  commit. See `docs/04-engine-notes.md`.

## Cost awareness

Ingest dominates everything. MHNSW and VIDX build their graph on every INSERT (~150 and ~55
rows/s on 1M rows), and **every M value reloads the whole corpus**. `--profile full
--resource-pass both` is ~796 h of ingest before a single query runs, and ~1,824 h of run time
in the worst case once querying is counted — all eight default engines, which is what that
command measures when `--engines` is omitted. VillageSQL, the ninth, is **opt-in**
(`runtime.default_selected: false`): a run that names no engines leaves it out and says so.
Named with `--engines` it adds ~4 h (its concurrency windows; it declines filtered search and
churn) and no ingest figure. `villagesql84`, the tenth, is the same on the MySQL 8.4 server
line: opt-in, and another ~4 h when named. **The two figures do not cover the same engines.**
The run-time figure covers all eight; the ingest figure covers the seven with a measured rate.
MyVector, the eighth, has none: the estimator names it in `engines_without_ingest_rate` and
prints that the ingest total is a floor, and its query terms (concurrency windows, churn) are
in the 1,824 h — its filtered-search term is not, because its driver declines that workload.
That figure was ~1,163 h until ops measurement version 9 raised the default churn budget from
1800 s to 7200 s per step and added one discarded warm-up window per concurrency client count:
the worst case charges the whole cap to every churn step of every unit, so the cap, not any
engine, moved it. It was ~1,820 h until the estimate counted mongot's index-size settle
(`SETTLE_CAP_S`, 600 s after every mongodb build), which version 9 added to the run and not
to the estimate.
The ingest half
used to cover six: Percona's rate was measured and unmodelled, the estimator named it in
`engines_without_ingest_rate` rather than folding it in, and the published 726 h was a floor
by an amount nothing here knew. The amount is **69.83 h**, and 726 h became 796 h when it was
added. Seventeen `ingest` records across six runs measure the engine: sixteen `incremental`,
from 48 rows/s (gist-960, M=32, 1M rows) to 638 (sift-128, M=8, 1M rows), plus one `post`
plain load at 2,653 rows/s, which is a different quantity because no graph is built during it
and is in neither rate table. What `orchestrator/estimate.py` now carries is the M=16
`incremental` mid-grid rate on the three corpora that have one — 339, 275 and 427 rows/s —
with gist-960 left to the formula because it was measured at M=32 alone. They are rates from
the OLD pin (`engine_version 9.7.1-1-vector-bench`), which the table's comment says, so the
first run at the new pin has something explicit to supersede. Before proposing a run, check
`docs/07-planning-a-run.md` and the estimate the run prints. Prefer `dev` → `smoke` when
validating framework changes; long profiles are the user's call, not a default.

Runs are resumable, checkpointed per `(pass, engine, dataset, phase)` and, inside an ops
phase, per `(M, build_mode)`:

```bash
./run-benchmark.sh run --profile main --resume --run-id main-<timestamp>
```

## Conventions

- Commit subjects are lowercase conventional-commit types (`fix:`, `feat:`, `docs:`, `test:`,
  `config:`, `perf:`, `chore:`) and name **the problem in plain language**, not the file
  touched — "fix: the tuned M pin silently halved the m-sweep profile". Bodies explain why,
  with the concrete evidence that produced the change (a run date, a measured number, the
  symptom it was mistaken for). Attribution is disabled globally.
- Comments in this codebase carry the *rationale and the incident* behind a decision, often at
  length. That density is intentional — match it when the code encodes a fairness or
  correctness decision, and keep it terse where the code is ordinary.
- Documentation that asserts engine behaviour should be executable: see
  `tests/verify-alisql-traps.sh`, which checks `docs/04-engine-notes.md` against a live server.
- Prefer adding a test in `tests/test_config.py` (fairness invariants), `tests/test_harness.py`
  (recall math, percentiles, Pareto, topology) or `tests/test_webui*.py` (API, guards, job
  validation) over a manual check — a crash gets noticed, a subtly wrong recall number gets
  published. Web UI tests build synthetic run dirs rather than reading `results/`, which is
  gitignored.

## Documentation map

`docs/01-architecture.md` (diagrams, container model) · `02-running-with-framework.md` (every
flag) · `03-running-manually.md` (standalone: each engine by hand with `docker run` and raw
SQL) · `04-engine-notes.md` (per-implementation traps) · `05-methodology.md` (what is measured,
fairness policy, known asymmetries) · `06-new-machine.md` · `07-planning-a-run.md` (read before
any long run) · `08-web-ui.md` (the browser front end, its access model and its limits) ·
`09-adding-percona-server.md` (how the seventh engine was added, and the vendor behaviour that
decided each choice) · `10-rollout.md` (what to run after this series, in what order, and the
two checks that cannot be done as specified) · `11-methodology-roadmap.md` (what the
measurement methodology cannot yet observe, what it already handles, and in what order to
address the rest — a roadmap, not a description of the system. **One** of its items is now
implemented, §4.1's capture of `Innodb_hnsw_memory_used`, and only on the `index_build`
record; nothing else in it is, so read a section as a proposal unless it says otherwise) ·
`12-percona-feedback-plan.md` (the four things the server team asked for after reading the
report, the re-pin that gates all of them, and what each costs — also a plan, also
unimplemented).

09, 10 and 11 are guarded: `tests/test_e1_percona_design.py`, `tests/test_e3_rollout_doc.py`
and `tests/test_e15_methodology_roadmap_doc.py` fail when the document drifts from the tree.
12 is not guarded yet, and its own phase 0 says so. **All three cite symbols only — never a
line number — and each guard ratchets its document to zero, over one shared pattern.** That
pattern and the frozen fixture that proves it still sees every form live in
`tests/test_e15_methodology_roadmap_doc.py`, and the other two guards import them: there were
three near-copies before, they were not identical, and the narrow ones could not see MTR
`.result`/`.test` paths, `.yy` grammars, extensionless files or bare `:N` continuations. A
gutter number reproduced inside a fenced block is the one form no pattern over code-font spans
can see, so it has a separate check. If you are adding a document, copy 11;
`docs/11-methodology-roadmap.md` §0 is where the policy and its reasoning live. 09 and 10 cited `file:line` until they were converted, and those citations rotted twice
in two days and put `master` red both times (#41, #43), both times because a later round of
edits in the same change moved lines after the citations were derived. A line number dies to
any edit above it; a symbol name dies only to a rename. The vendor half was worse and nothing
reported it: 09's guard deny-listed every Percona path by name, so on the order of 164 line
references into a tree that is not in this checkout were checked by nothing at all.
