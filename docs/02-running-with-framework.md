# Running the benchmark with the framework

Everything goes through one entrypoint: `./run-benchmark.sh`.

## 0. Requirements

On the host: **docker**, **python3 with PyYAML**, **git**. Nothing else. The
database clients and the scientific Python stack live inside the images, so
running a benchmark does not install anything onto the machine being measured.

```bash
# Debian / Ubuntu
sudo apt-get install -y docker.io python3-yaml git
sudo usermod -aG docker "$USER"   # then log out and back in
```

Disk: budget roughly **40 GB** for images and sources, plus 2–4× your dataset
size for index data. The `gist-960-euclidean` dataset alone is 3.6 GB.

## 1. First run, end to end

```bash
cd vector-bench

# 1. Export engine sources at their pinned tags and build all six images.
#    pgvector ~10 min, MariaDB ~40 min, AliSQL 1.5-3 h. See the note below.
./run-benchmark.sh build

# 2. Download the datasets.
./run-benchmark.sh fetch --datasets fashion-mnist-784-euclidean

# 3. Prove the whole pipeline works before spending real time on it.
./run-benchmark.sh run --profile smoke
```

> **On build times.** AliSQL dominates. Its CMake calls `MYSQL_CHECK_DUCKDB()`
> unconditionally on Linux, so the bundled DuckDB in `extra/duckdb` is compiled
> on every build — `WITH_DUCKDB_STORAGE_ENGINE=OFF` only makes the plugin a
> separate `.so` instead of linking it into `mysqld`; it does not skip the
> compile. DuckDB's ExternalProject also redirects its output to `/dev/null`, so
> the build looks stalled for a long stretch. It is not. Build engines
> separately (`--engines alisql`) if you want to start measuring the other two
> sooner.

The smoke profile takes about 30 minutes per resource pass and exercises every
stage for every engine. **Do not skip it.** It is far cheaper to discover a broken image
or a missing dataset here than eight hours into a full run.

When it finishes:

```
results/
├── annb/                                ann-benchmarks recall/QPS results
│   ├── normalized/<dataset>/<k>/<engine>/*.hdf5
│   └── tuned/<dataset>/<k>/<engine>/*.hdf5
└── smoke-20260803-141500/
    ├── run-manifest.json                provenance: CPU, versions, limits
    ├── checkpoints.json                 what completed, for --resume
    ├── ops-*.jsonl                      ops-harness records
    ├── mem-*.jsonl                      server memory timeseries
    └── report/
        ├── report.html                  self-contained, shareable
        ├── report.md
        ├── records.jsonl                every measurement, one flat schema
        └── charts/*.svg, *.png
```

The `annb/` tree is split by resource pass on purpose. ann-benchmarks names its
result files from the algorithm and its parameters only, with no notion of a
resource pass — a shared tree would make the tuned pass skip every point the
normalized pass had already computed, and the report would then present
normalized numbers as tuned ones. It also sits outside the per-run directory so
an interrupted run can resume without recomputing points it already has.

## 2. A real run

```bash
# Download the rest of the datasets first (~5 GB).
./run-benchmark.sh fetch

# The profile sized to actually be run: two datasets at 1M scale, ~2 days.
./run-benchmark.sh run --profile main

# One dataset, ~1 day — a complete three-way comparison at 1M scale.
./run-benchmark.sh run --profile main --datasets glove-100-angular
```

> `--profile full --resource-pass both` is **~272 h of ingest — about 11 days**
> — before a single query runs. It describes the complete measurement space; it
> is not a recommendation. Every run prints its own estimate before starting.
> [07-planning-a-run.md](07-planning-a-run.md) has the measured rates and what
> each way of narrowing the scope costs you.

## 3. Commands

| Command | Purpose | In the web UI |
| --- | --- | --- |
| `build` | Export sources at pinned tags, build runtime + bench images | Engines |
| `fetch` | Download datasets into `datasets/` | Datasets |
| `generate` | Build a dataset that is not published for download | Datasets |
| `run` | Execute a benchmark run | Profiles & launch |
| `report` | Regenerate the report from an existing run directory | Runs → Report |
| `export` | Package a run as a `.tar.gz` to send to someone | Runs → Report |
| `render` | Regenerate the ann-benchmarks configs for a profile | — |
| `select-builds` | Choose the full-size builds from a `frontier-explore` run (stage 2 of the two-stage frontier) | — |
| `sources` | Export sources only, without building | — |
| `web` | Serve the browser interface | — |
| `clean` | Remove containers, networks and volumes left by a run | — |
| `lab` | Run one diagnostic script against one engine, recording nothing | — |

Everything in the third column can be driven from
[the web UI](08-web-ui.md) instead, which shows the command it is about to run
before it runs it. Its **Setup** page walks a fresh machine through the first
three in order, and checks that every image agrees on `-march` — a mixed build
compares compiler flags rather than implementations, and looks perfectly
plausible in the report. The CLI remains the whole interface on a machine you have
only SSH'd into.

### `run` options

| Option | Default | Meaning |
| --- | --- | --- |
| `--profile` | `quick` | `smoke`, `quick`, `full`, or your own in `config/profiles/` |
| `--engines` | the profile's `engines:` list, else every engine except the opt-in ones | e.g. `mariadb,alisql`; may only **narrow** a profile's own list; the only way to run an opt-in engine |
| `--datasets` | from profile | Override the profile's dataset list |
| `--resource-pass` | `both` | `normalized`, `tuned`, or `both` |
| `--phases` | `both` | `ann` (recall/QPS), `ops` (build/concurrency/filtered/churn) |
| `--resume` | off | Skip units already recorded complete |
| `--force` | off | Re-run ann-benchmarks points that already have results |
| `--fail-fast` | off | Stop at the first failure instead of continuing |
| `--no-report` | off | Skip report generation |
| `--run-id` | timestamped | Reuse an id to add to an existing run directory |
| `--overlay` | none | A `select-builds` overlay applied to `--profile`; tuned pass only, and a resume must pass the same one |

**`--engines` and `--datasets` are not symmetrical, deliberately.** A profile's
dataset list is a choice of corpus and `--datasets` replaces it outright. A
profile that declares `engines:` has a *grid* that is only meaningful on those
engines — `config/profiles/percona.yml` is the one that does, and every axis it
has is Percona-only — so `--engines` may select a subset of that list and is
**refused** if it names anything outside it, rather than silently measuring one
engine at another's grid. A profile that declares no `engines:` key is making no
such claim, and `--engines` is free there. Omitting the key is also the only way
to say "every engine": `engines: []` is refused as a half-finished edit, because
reading it as "no restriction" is how a one-engine profile measured seven.

**Opt-in engines.** An engine whose `config/engines/<e>.yml` sets
`runtime.default_selected: false` is left out of every run that names no
engines, and `run` prints `note: opt-in engine(s) not run unless named with
--engines: ...` so the shorter list is not mistaken for a bug. Naming it runs
it: `--engines villagesql`, or `--engines mariadb,villagesql`. VillageSQL is
the one opt-in engine today. It was made opt-in when its first pin's index
was never used (every query was a full scan the module refused to time); its
current pin's index is used, and whether it joins every default run is a cost
decision the flag leaves open. The web UI's estimate follows the same
default.

### `select-builds` options — the two-stage frontier

Stage 1 explores every build knob at 90,000 rows; this command keeps the
builds worth measuring at 990,000; stage 2 measures them. The rule, its
definition of "near" and the cost of each stage are in
[07-planning-a-run.md §2a](07-planning-a-run.md#2a-exploring-the-whole-parameter-space-the-two-stage-frontier).

```bash
./run-benchmark.sh run --profile frontier-explore                    # stage 1
./run-benchmark.sh select-builds --from results/<stage-1 run id> \
    --out state/frontier-stage2.yml                                  # select
./run-benchmark.sh run --profile tuned-complete --resource-pass tuned \
    --overlay state/frontier-stage2.yml                              # stage 2
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--from` | required | The stage-1 run directory. Reads its `run-manifest.json` and `report/records.jsonl` (run `report --run-dir` first if the run skipped it) |
| `--out` | required | The overlay to write. Generated — regenerate rather than edit it |
| `--profile` | `tuned-complete` | The profile the overlay will be applied to; `run` refuses it on any other |
| `--dataset` | the stage-1 corpus | Needed only when the stage-1 records hold more than one |

It prints, per engine, how many builds it kept and pruned, the lists it wrote,
any kept M at the edge of the explored grid and any floor stage 1 never
reached. The overlay names the kept builds exactly, so none is added by
multiplying the per-axis lists. It refuses to write an overlay whose rendered grid is not exactly the
kept set. `render --profile tuned-complete --resource-pass tuned --overlay
<file>` shows the ann configs it produces without running anything.

### `build` options

| Option | Meaning |
| --- | --- |
| `--engines` | Build one engine only |
| `--target` | `runtime` (server only), `bench` (server + Python), or `all` |
| `--march` | **SIMD baseline for every engine.** See below. |
| `--jobs` | Compile parallelism (default: all cores) |
| `--no-cache` | Force a clean rebuild |

### `generate` options

`fetch` downloads datasets published as prebuilt HDF5. The `dbpedia-openai-*`
family is not published that way: ann-benchmarks assembles it from a source
corpus and then computes exact ground truth by brute force.

```bash
./run-benchmark.sh generate --list
./run-benchmark.sh generate dbpedia-openai-1000k-angular
```

> **This is not a download.** The full 1M-row source is fetched whichever size
> you pick (~6–10 GB); choosing a smaller variant only shrinks the ground-truth
> computation and every later engine load. Budget hours and about 20 GB of
> working space for the 1000k variant, and build a bench image first — the
> generation runs inside one.

### `export` options

| Option | Meaning |
| --- | --- |
| `--run-dir` | `results/<run-id>`, or just the run id |
| `--output` | Path for the archive (default `./vector-bench-<run-id>.tar.gz`) |

Produces the run directory plus a README naming the machine the numbers belong
to, its SIMD flags, what was measured and how to view it. A recipient with
vector-bench extracts it into their `results/`; one without opens
`report/report.html`, which is self-contained. Refuses a run with no readable
manifest, because a result without the environment that produced it is not a
result.

### `lab` options

Everything above produces measurements. `lab` is the one that does not: it
starts a single engine, runs one script from `scripts/` against it, and removes
both containers on the way out — including on Ctrl-C.

```bash
./run-benchmark.sh lab --list
./run-benchmark.sh lab --engine valkey probe-valkey-filter.py --rows 200000
./run-benchmark.sh lab --engine valkey --shell
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--engine` | required | Which engine to start |
| `--resource-pass` | `tuned` | Server flags and limits, exactly as a run of that pass |
| `--shell` | off | A prompt in the bench container instead of a script |
| `--list` | — | The scripts `lab` can run, with what each one answers |
| `--client-memory-gb` | from the pass | Raise it for a script holding a million vectors |
| `--force` | off | Start even though a measurement is running |
| `--timeout` | 4 h | Before the script's container is killed |

Put lab's own flags **before** the script name. Everything after it goes to the
script, so that a script can have any flags it likes — including ones lab also
uses.

It exists because the questions a run raises cannot be answered by another run.
Valkey's filtered search came back six times *slower* at 1% selectivity than at
10%; settling whether that was the engine or our query meant four query shapes
against a populated index — five minutes of work that otherwise costs a
forty-hour profile or a pair of hand-started containers.

Two properties make the answers worth having:

- **The server is started by the same code a run uses**, so it gets the pass's
  flags, cpuset and memory from `config/engines/<e>.yml`. The predecessor was a
  standalone shell script that started its own Valkey with hand-written flags,
  and it had already drifted — 32 GB where a tuned run gives 101 GB. A probe run
  under a different configuration can disprove something the run never did.
- **Nothing is recorded.** No writable path is mounted, so a lab session cannot
  write a record. A script here generates its own vectors and its own ground
  truth; those numbers are for comparing shapes against each other and must
  never reach a report and sit beside corpus numbers.
- **It refuses to start beside a measurement.** A session takes the same cores
  and memory the pass hands a run, so run concurrently it contends with one —
  and the damage lands on the run, whose numbers are the ones being kept. This
  is the command most likely to be typed into a second terminal while the first
  is measuring, which is why the invariant is checked rather than trusted.
  `--force` overrides it.

A script is any `.py` file in `scripts/`; it is passed `--host` and `--port`, and
receives credentials in `VB_DB_USER` / `VB_DB_PASSWORD`. The server's log is kept
under `state/lab/<session>/`; the data volume goes at teardown. If a session is
killed hard enough to strand its containers, `./run-benchmark.sh clean` removes
them.

### `web` options

| Option | Default | Meaning |
| --- | --- | --- |
| `--port` | `8080` | |
| `--host` | `127.0.0.1` | Interface to publish on. Non-loopback implies `--auth` |
| `--allow-control` | off | Allow editing profiles, adding engines, and starting work |
| `--auth` | off on loopback | Require a password |
| `--no-auth` | — | Publish without one. Only on a network you already trust |
| `--behind-proxy` | off | TLS is terminated in front; marks cookies `Secure` |
| `--no-container` | off | Run the server on the host instead of in a container |

See [08-web-ui.md](08-web-ui.md).

## 4. `-march` — the one build flag that changes results

Every engine compiles SIMD distance kernels. `-march` decides which
instructions those kernels get, and it is applied identically to all of them:

```bash
./run-benchmark.sh build --march x86-64-v3   # AVX2 + FMA. Portable. Default.
./run-benchmark.sh build --march native      # Whatever this CPU has, incl. AVX-512.
```

Use `native` when the build host and the benchmark host are the same machine and
you want each engine's best case. Use `x86-64-v3` when the images must move
between machines. **Never mix**: rebuilding one engine with a different `-march`
turns the benchmark into a comparison of compiler flags.

The value used is baked into each image and reported in the manifest.

## 5. Resumption and failures

Work is checkpointed on success at two levels: per
`(resource pass, engine, dataset, phase)`, and within an ops phase per
`(M, build_mode)`. The second matters — an ops phase can run a dozen hours
across several M values, and checkpointing only the whole phase would discard
every M that had already finished. After an interruption:

```bash
./run-benchmark.sh run --profile main --resume --run-id main-20260804-185624
```

Completed units are skipped, so an interruption costs one unit rather than the
run. Failed units are not checkpointed, so `--resume` retries exactly those.

If a run leaves containers behind (SIGKILL, host reboot):

```bash
./run-benchmark.sh clean --run-id full-20260803-120000
```

## 6. Choosing a resource pass

`normalized` gives every engine identical CPU, memory and cache budgets, so any
difference belongs to the implementations. `tuned` gives each engine its
vendor-recommended configuration on the full machine, so the numbers reflect
what each can actually do.

Run both. They answer different questions, and either one alone is easy to
argue with. See [05-methodology.md](05-methodology.md) §3.

## 7. Customising

Profiles and resource passes are plain YAML; copy and edit.

```bash
cp config/profiles/quick.yml config/profiles/mine.yml
$EDITOR config/profiles/mine.yml
./run-benchmark.sh run --profile mine
```

The knobs worth changing first:

- `ann.m_values` and `ann.ef_search` — sweep density. More points make a
  smoother Pareto frontier and cost linearly more time.
- `ops.client_counts` — how far the concurrency sweep goes. Push past your core
  count to find the saturation point.
- `ops.subset_rows` — load only part of the training set, for faster iteration.
- `ops.build_selection` — `reference` (default) or `frontier`. Which index the
  ops pass (build cost, concurrency, filtered, churn) measures. `reference`
  builds each engine's declared `runtime.reference_build` at every
  `ops.m_values` entry. `frontier` waits for the ann phase of each engine and
  dataset and builds the one swept build that is fastest at the ops calibration
  target (`ops.target_recall`, 0.95 by default), so the ops numbers describe
  the index the frontier headline comes from; `ops.m_values` is then used only
  where the chosen build names no M (Percona Search). It needs the ann phase:
  `run` refuses `frontier` with `ann.enabled: false`, and a unit whose ann
  phase did not run in this run directory (`--phases ops`, a failed ann phase)
  measures the reference build and says `fallback_reference` in a warning, the
  manifest and every record. A resumed run reuses the build chosen before the
  interruption while the ann attempt it came from is still the latest one,
  and chooses again after an ann re-run. The cost estimate cannot know the
  build ahead of time and charges the ops ingest at the largest M in the grid
  instead (not for Percona Search, and not a bound on `ef_construction`). See
  [05-methodology.md](05-methodology.md) §2.1.
- `memory.server_limit_gb` in `config/resources/normalized.yml` — the single
  most consequential value, because an engine whose graph does not fit in cache
  behaves completely differently from one whose graph does.

### Adding an engine

`config/engines/*.yml` is the whole definition of an engine — how it is built,
how it is started, the SQL it speaks, and a `runtime:` block with the port, data
mounts, credentials, readiness probe, chart colour and label. `KNOWN_ENGINES` is
whatever that directory holds; there is no list to keep in step with it.

**A variant of a family already driven** — another MariaDB or MySQL fork,
another Postgres build — is therefore configuration and a build, with no Python:

```bash
cp config/engines/mariadb.yml config/engines/perconaserver.yml
$EDITOR config/engines/perconaserver.yml     # name, source tag, image names,
                                             # runtime.ann_constructor
./run-benchmark.sh build --engines perconaserver
```

The web UI's Engines page does exactly this, and validates it. Two collisions
are worth knowing about because neither crashes:

- **`runtime.ann_constructor` must be unique.** ann-benchmarks keys its result
  files on that name, so two engines sharing one share a results tree and the
  second reports the first's recall numbers. This is precisely why `mariadb123`
  exists as its own constructor rather than a retagged `mariadb`.
- **`image.runtime` and `image.bench` must be unique**, or two builds overwrite
  each other.

**A new architecture** — something none of the existing drivers speak — needs
code as well: a driver in `harness/drivers/` registered in `_driver_classes()`,
an ann-benchmarks module under `overlay/`, and a Dockerfile. The engine config
names its driver by class name, and a name nothing implements is refused at load
rather than four hours into a run.

## 8. Reading the output

Start with `report/report.html`. Its order is deliberate:

1. **Environment** — the hardware and versions the numbers are scoped to.
2. **Validity** — anything that invalidates or narrows a result: measurements
   that ran without the vector index, filtered queries that returned short,
   environment warnings. Read this before the charts.
3. **Known asymmetries** — structural differences no configuration removes.
4. **Recall vs throughput**, then build cost, concurrency, filtered, churn.

`report/records.jsonl` holds every measurement in one flat schema, so you can
query it directly:

```bash
jq -r 'select(.phase=="recall_qps" and .recall_at_k>0.95)
       | [.engine, .m, .ef_search, .recall_at_k, .qps] | @tsv' \
  results/<run-id>/report/records.jsonl | sort -k5 -rn | head
```

## 9. Troubleshooting

**"image not found"** — build it: `./run-benchmark.sh build --engines <name>`.

**"dataset not found"** — fetch it: `./run-benchmark.sh fetch --datasets <name>`.

**A server container exits during startup.** The orchestrator prints the
container's log tail with the error. The usual causes are a memory limit too
small for the configured buffer pool, or a `shm_size` too small for PostgreSQL.

**"the vector index is NOT in the query plan".** Not a framework failure — the
engine's optimizer chose a full scan. The run continues and the affected
measurements are listed in the report's Validity section. For AliSQL this is
expected at large `LIMIT` values; see [04-engine-notes.md](04-engine-notes.md).

**Warnings about CPU topology.** On a machine with fewer homogeneous cores than
the profile requests, the server and client containers are forced to share
cores. That adds latency noise. Lower `cpu.server_cpus` in the resource config,
or accept the noise — it is recorded in the manifest either way.

**Results owned by root.** The engines run as root inside their containers. The
orchestrator chowns the results back at the end of a run; if it was killed
first, `sudo chown -R "$USER" results/` fixes it.
