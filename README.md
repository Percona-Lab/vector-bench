# vector-bench

Benchmark framework comparing **built-in vector search** across seven databases,
all using an HNSW index:

| Engine key | Engine | Version | Implementation |
| --- | --- | --- | --- |
| `mariadb` | MariaDB | 11.8.8 LTS | MHNSW (`sql/vector_mhnsw.cc`) |
| `alisql` | AliSQL | 8.0.44-2 | VIDX (`sql/vidx/vidx_hnsw.cc`) |
| `pgvector` | PostgreSQL | 17 + pgvector 0.8.6 | pgvector HNSW |
| `mariadb123` | MariaDB | 12.3.2 | MHNSW, a second version |
| `mongodb` | Percona Search for MongoDB | mongot 1.70.3 | Lucene HNSW in a sidecar process |
| `valkey` | Valkey + valkey-search | 9.1 | HNSW, entirely in memory |
| `percona` | Percona Server | 9.7.2-2 (MySQL 9.7.2 LTS base) | InnoDB HNSW (`storage/innobase`) |

The first three are the original comparison and what `build` produces by
default; the rest are opt-in, because each is another image to compile or
another process to stand up. `run` measures all seven unless told otherwise.

Every one runs HNSW, so the comparison isolates **implementation quality**
rather than algorithm choice. They are not otherwise alike — one keeps the
index in a separate JVM process, one keeps everything in RAM — and
[docs/05-methodology.md](docs/05-methodology.md) records the asymmetries no
configuration removes.

---

## Contents

- [Quick start](#quick-start) — zero to a report
- [What you get](#what-you-get)
- [Step by step](#step-by-step)
- [Commands](#commands)
- [The web UI](docs/08-web-ui.md)
- [Running the engines by hand](#running-the-engines-by-hand)
- [Things that will ruin your results](#things-that-will-ruin-your-results)
- [Documentation](#documentation)
- [Design](#design)

---

## Quick start

```bash
# prerequisites (Debian/Ubuntu) — this is the complete list
sudo apt-get update
sudo apt-get install -y docker.io python3-yaml git curl
sudo usermod -aG docker "$USER"        # then log out and back in

git clone https://github.com/Percona-Lab/vector-bench.git
cd vector-bench
chmod +x run-benchmark.sh scripts/*.sh tests/*.sh

./run-benchmark.sh build --march native    # ~2-4 h, AliSQL dominates
./run-benchmark.sh fetch                   # ~5 GB of datasets
./run-benchmark.sh run --profile smoke     # ~30 min gate — do not skip
./run-benchmark.sh run --profile main      # ~2 days: the real measurement
```

> **Check the runtime before you start.** MHNSW and VIDX build their graphs on
> every INSERT — ~150 rows/s for MariaDB and ~55 for AliSQL on a 1M-row dataset
> — and every M value reloads the whole dataset. `--profile full
> --resource-pass both` is **~11 days of ingest** before a single query runs.
> `main` is the profile sized to be run. Every run prints its own estimate
> first; see [docs/07-planning-a-run.md](docs/07-planning-a-run.md).

The report lands in `results/<run-id>/report/report.html` — self-contained,
charts inlined, no network needed to view it.

**Host requirements:** docker, python3 with PyYAML, git, curl. Nothing else —
the database clients and the scientific Python stack live inside the images, so
running a benchmark does not install anything onto the machine being measured.

**Resources:** ~80 GB disk, 32 GB+ RAM, 8+ physical cores.

---

## What you get

| Dimension | Measured by | Why it is here |
| --- | --- | --- |
| Recall vs QPS | ann-benchmarks | The standard ANN quality/speed frontier |
| Index build cost | ops harness | Build time, peak server RSS, index size on disk |
| Concurrency scaling | ops harness | QPS and p50/p95/p99 at 1→32 clients |
| Filtered (hybrid) search | ops harness | Vector search + `WHERE` at 1% / 10% / 50% selectivity |
| Churn | ops harness | Recall drift after 10% / 25% delete+reinsert |

Only the first is what ann-benchmarks provides. The rest are what separate a
database from an ANN library, and are where these three diverge most.

Each run produces:

```
results/<run-id>/
├── run-manifest.json        CPU model, SIMD flags, versions, resolved limits
├── checkpoints.json         what completed, for --resume
├── ops-*.jsonl              ops-harness records
├── mem-*.jsonl              server memory timeseries
└── report/
    ├── report.html          self-contained, shareable
    ├── report.md
    ├── records.jsonl        every measurement, one flat schema
    └── charts/*.svg, *.png
```

---

## Step by step

### 1. Check the host

```bash
df -h /                                    # want 80 GB+ free
nproc; free -g
docker info >/dev/null && echo "docker ok"
python3 -c 'import yaml' && echo "pyyaml ok"
```

### 2. Build the images

Each engine produces two images: `<engine>-runtime` (the server alone, usable by
hand) and `<engine>-bench` (runtime plus the Python stack the harness needs).

```bash
./run-benchmark.sh build --march native          # the original three
# or one at a time — start AliSQL first, it is the long pole:
./run-benchmark.sh build --engines alisql   --march native
./run-benchmark.sh build --engines mariadb  --march native
./run-benchmark.sh build --engines pgvector --march native
```

| Engine | Cold build |
| --- | ---: |
| pgvector | ~10 min |
| MariaDB | ~40 min |
| AliSQL | **1.5–3 h** |

AliSQL is slow because its CMake compiles the bundled DuckDB unconditionally on
Linux and sends that output to `/dev/null` — the build looks stalled for a long
stretch. It is not.

Sources are exported from pinned tags. If you have local MariaDB / AliSQL /
ann-benchmarks checkouts, point at them to skip the clones — they are read only:

```bash
export VB_REPO_MARIADB=/path/to/server
export VB_REPO_ALISQL=/path/to/AliSQL
export VB_REPO_ANNB=/path/to/ann-benchmarks
```

### 3. Fetch datasets

```bash
./run-benchmark.sh fetch --list      # names, sizes, roles
./run-benchmark.sh fetch             # the four profile datasets, ~5 GB
```

| Dataset | Shape | Role |
| --- | --- | --- |
| fashion-mnist-784-euclidean | 60k × 784 | smoke |
| glove-100-angular | 1.18M × 100 | main |
| sift-128-euclidean | 1M × 128 | main |
| gist-960-euclidean | 1M × 960 | stress |

Downloads are resumable and size-verified, so an interrupted transfer is never
mistaken for a complete dataset.

### 4. Verify before measuring

```bash
python3 -m pytest tests/ -q          # 777 unit tests, ~30 s
./tests/verify-alisql-traps.sh       # 8 engine-behaviour checks against a live server
./run-benchmark.sh run --profile smoke
```

The smoke profile takes ~30 minutes per resource pass and exercises every stage
for every engine. **Do not skip it** — far cheaper than discovering a broken image eight
hours into a full run.

There is also a one-minute synthetic cycle needing no dataset download:

```bash
./tests/make-tiny-dataset.sh
./run-benchmark.sh run --profile dev
```

### 5. Measure

```bash
./run-benchmark.sh run --profile main                              # ~2 days
./run-benchmark.sh run --profile main --datasets glove-100-angular # ~1 day
```

Resumable. Work is checkpointed per `(pass, engine, dataset, phase)` and, inside
an ops phase, per `(M, build_mode)` — an ops phase can run a dozen hours across
several M values, so an interruption does not discard the ones already done:

```bash
./run-benchmark.sh run --profile main --resume --run-id main-<timestamp>
```

### 6. Read the report

Open `results/<run-id>/report/report.html`. Its order is deliberate:

1. **Environment** — the hardware and versions the numbers are scoped to
2. **Validity** — measurements that did not use the vector index, filtered
   queries that returned short, environment warnings. **Read this before the charts.**
3. **Known asymmetries** — structural differences no configuration removes
4. Recall vs throughput, build cost, concurrency, filtered search, churn

Query the raw numbers directly:

```bash
jq -r 'select(.phase=="recall_qps" and .recall_at_k>0.95)
       | [.engine, .m, .ef_search, .recall_at_k, .qps] | @tsv' \
  results/<run-id>/report/records.jsonl | sort -k5 -rn | head
```

### 7. Or use the browser

```bash
./run-benchmark.sh build --engine webui     # once, ~1 min
./run-benchmark.sh web                      # http://127.0.0.1:8080
```

A Status page saying whether the machine can measure yet, an index of every run,
the manifest as a readable page, and an explorer over `records.jsonl` that leads
with the five things this framework measures. The generated `report.html` is
still there, unchanged, in its own tab.

`--allow-control` additionally allows downloading datasets, building images,
adding an engine variant, editing `config/profiles/*.yml` with validation and an
ingest estimate, launching a run, and watching any of it live. Every button shows
the command it runs. Nothing may run alongside a benchmark. It is off by default, and the server binds loopback only — reach a remote
rig with `ssh -N -L 8080:127.0.0.1:8080 you@bench-host`. See
[docs/08-web-ui.md](docs/08-web-ui.md).

---

## Commands

| Command | Purpose |
| --- | --- |
| `build` | Export sources at pinned tags, build runtime + bench images |
| `fetch` | Download datasets |
| `generate` | Build a dataset that is not published for download (dbpedia-openai-*) |
| `run` | Execute a benchmark run |
| `report` | Regenerate the report from an existing run directory |
| `web` | Serve the web UI: run index, interactive explorer, profile editor, launcher |
| `render` | Regenerate the ann-benchmarks configs for a profile |
| `sources` | Export sources only, without building |
| `export` | Package a run as a .tar.gz to send to someone |
| `clean` | Remove containers, networks and volumes left by a run |

### `run` options

| Option | Default | Meaning |
| --- | --- | --- |
| `--profile` | `quick` | `dev`, `smoke`, `quick`, **`main`**, `full`, or your own |
| `--engines` | the profile's `engines:` list, else every engine | e.g. `mariadb,alisql`; may only narrow a profile's own list — see [docs/02](docs/02-running-with-framework.md#run-options) |
| `--datasets` | from profile | Override the profile's dataset list |
| `--resource-pass` | `both` | `normalized`, `tuned`, or `both` |
| `--phases` | `both` | `ann` (recall/QPS), `ops` (build/concurrency/filtered/churn) |
| `--resume` | off | Skip units already recorded complete |
| `--force` | off | Re-run ann-benchmarks points that already have results |
| `--fail-fast` | off | Stop at the first failure |
| `--no-report` | off | Skip report generation |

### Profiles

| Profile | Datasets | Ingest time | Use for |
| --- | --- | ---: | --- |
| `dev` | tiny synthetic | ~1 min | validating framework changes |
| `smoke` | fashion-mnist | ~30 min | the gate before any long run |
| `quick` | fashion-mnist + glove-100 | ~35 h | coarse but real numbers |
| **`main`** | glove-100 + sift-128 | **~61 h** | **the report** |
| `full` | all four, both passes | ~272 h | the complete space, rarely run whole |
| `mariadb-blog` | dbpedia-openai-1000k, 1536 dims | ~21 h | reproducing MariaDB's published big-vector benchmark |

Times are measured ingest only, the original three engines, before any query
runs, and they are what the whole run loads: every `M` value reloads the
corpus on both paths, so a profile with three `M` values pays for the corpus
four times. `smoke` is lower than its dataset suggests because
`ops.subset_rows` caps its ops phase at 20,000 of fashion-mnist's 60,000.
`mariadb-blog` is the one row not derived from the per-engine rates above:
dbpedia-openai-1000k is 1,536 dims and loads far slower than they imply, so
its figure comes from the estimator the run prints. `full`
describes the measurement space rather than recommending it. Narrow with
`--datasets` and `--resource-pass`; see
[docs/07-planning-a-run.md](docs/07-planning-a-run.md) for what each cut costs
you scientifically — and note that every figure here is pinned against the
estimator by `tests/test_estimate.py`, so a profile change moves them or fails
the suite.

Profiles are plain YAML — copy and edit:

```bash
cp config/profiles/quick.yml config/profiles/mine.yml
./run-benchmark.sh run --profile mine
```

---

## Running the engines by hand

You do not need the framework to try these engines. Build only the runtime
images and drive them yourself:

```bash
./scripts/prepare-sources.sh
./scripts/build-images.sh --target runtime

docker run -d --name mdb vector-bench/mariadb-runtime:mariadb-11.8.8
docker exec -it mdb /usr/local/bin/vb-entrypoint client
```

```sql
CREATE DATABASE demo; USE demo;
CREATE TABLE items (
  id INT PRIMARY KEY, tag INT NOT NULL, v VECTOR(3) NOT NULL,
  VECTOR INDEX vi (v) M=6 DISTANCE=cosine
) ENGINE=InnoDB;
INSERT INTO items VALUES (1, 0, VEC_FromText('[0.1,0.2,0.3]'));
SET mhnsw_ef_search = 100;
SELECT id FROM items ORDER BY VEC_DISTANCE_COSINE(v, VEC_FromText('[0.1,0.2,0.3]')) LIMIT 10;
```

[docs/03-running-manually.md](docs/03-running-manually.md) covers the original three
engines this way — start, connect, create, insert, search, confirm the index was
used, and measure recall by hand. It is standalone and needs nothing else here.

---

## Things that will ruin your results

**Check the runtime before starting.** `--profile full --resource-pass both` is
~272 h of ingest — about 11 days — before a single query runs, because MHNSW and
VIDX build incrementally at 55–150 rows/s and every M value reloads the whole
dataset. Use `main`, read the estimate the run prints, and see
[docs/07-planning-a-run.md](docs/07-planning-a-run.md).

**Give the machine to the benchmark.** A competing build distorted MariaDB's
numbers by 2× during development. The harness reports CPU, SIMD and cpuset
problems in its Validity section, but a concurrent workload is invisible to it.

**Never mix `-march` between engines.** All five that are compiled from source
compile SIMD distance kernels (mongodb and valkey are installed from packages).
Rebuilding one with a different value turns the benchmark into a comparison of
compiler flags. Use `native` when the build host is the benchmark host,
`x86-64-v3` when images must move between machines.

**AVX-512 changes the answer.** MariaDB MHNSW and AliSQL VIDX both document
AVX-512 distance kernels. Results from a host without it do not transfer to one
with it. Check after your first run:

```bash
jq '.host.cpu | {model, has_avx512, hybrid}' results/<run-id>/run-manifest.json
```

**Every engine can silently fall back to a full table scan**, which returns
exact results slowly — indistinguishable in the output from "very accurate but
slow". Every driver runs `EXPLAIN` per configuration and records whether the
index was used; anything that was not appears in the report's Validity section.

**MariaDB and AliSQL ship a 16 MiB graph cache by default**
(`mhnsw_max_cache_size`, `vidx_hnsw_cache_size`) — far too small for any real
corpus. The framework sets both from the resource profile so neither is judged
on a value its vendor plainly intended you to change.

**Percona Server's `innodb_hnsw_max_memory` is not a third one of those.** It is
a hard allocation ceiling rather than a cache: exceeding it fails the statement
instead of evicting anything, and the check is re-applied as a cold search
faults the graph back in after a restart. It is therefore set to `0`
(unlimited), so the container's memory limit bounds it as it does every other
engine, and `Innodb_hnsw_memory_used` is recorded on the `index_build` record
only — one of the six ops phases, with `ingest`, `concurrency`, `filtered`,
`churn` and `ef_calibration` uncollected — and even there it excludes the node
map, so it is a floor on graph memory rather than a measurement of it. See
[docs/09](docs/09-adding-percona-server.md) §3.

---

## Documentation

| Document | Contents |
| --- | --- |
| [01-architecture.md](docs/01-architecture.md) | System architecture, container model, per-engine internals, data flow — with diagrams |
| [02-running-with-framework.md](docs/02-running-with-framework.md) | Every command and flag, resumption, customising profiles, troubleshooting |
| [03-running-manually.md](docs/03-running-manually.md) | **Standalone.** Each engine by hand with `docker run` and raw SQL |
| [04-engine-notes.md](docs/04-engine-notes.md) | What each implementation does, and the traps in each |
| [05-methodology.md](docs/05-methodology.md) | What is measured, how, fairness policy, known asymmetries |
| [06-new-machine.md](docs/06-new-machine.md) | Moving the framework to another machine |
| [07-planning-a-run.md](docs/07-planning-a-run.md) | **Read before a long run.** Measured ingest rates, what each profile costs, how to scope, resumption |
| [08-web-ui.md](docs/08-web-ui.md) | The browser front end: run index, record explorer, profile editing, launching and live monitoring |

---

## Design

Two measurement paths write into one flat record schema:

- **ann-benchmarks**, run unmodified inside each engine image (`run.py --local`),
  produces recall/QPS. The orchestrator launches the container so it controls
  cpuset, memory and environment; ann-benchmarks keeps ownership of definitions,
  ground truth and recall computation. No patching of the upstream project.
- **The ops harness**, a separate client container talking to a server container
  over a private network, produces everything else. Two containers rather than
  one so the server's cgroup accounting measures the server and not the several
  hundred megabytes of NumPy holding the dataset.

### Fairness

Two passes, both reported. **normalized** gives every engine identical CPU set
(pinned to one core class on hybrid CPUs), memory limit, graph-cache budget,
parameter grid and `-march`, so differences belong to the implementations.
**tuned** gives each engine its vendor-recommended configuration on the full
machine. Neither alone is the answer, which is why both are run.

Three asymmetries no configuration removes, recorded in every report:

- `ef_construction` is exposed **only** by pgvector — verified: MariaDB rejects
  it with `ERROR 1911 Unknown option 'EF_CONSTRUCTION'`. It is therefore pinned
  to pgvector's default in the normalized pass.
- pgvector **bulk-builds** its index after load; MHNSW and VIDX build
  incrementally on every INSERT. Both modes are measured, because "pgvector
  builds faster" and "pgvector builds faster when allowed to bulk-build" are
  different claims.
- AliSQL VIDX is **InnoDB-only** and requires **READ COMMITTED**.

### Provenance

Every run writes `run-manifest.json`: CPU model and SIMD flags, core topology,
RAM, kernel, Docker version, engine source commits and tags, image ids, and the
fully resolved resource limits. The report generator refuses to run without it —
a result without its environment is not a result.

### Layout

```
vector-bench/
├── run-benchmark.sh          single entrypoint
├── config/profiles/          dev · smoke · quick · full — what to measure
├── config/resources/         normalized · tuned — how much machine
├── config/engines/           per-engine build, server flags, runtime
├── docker/                   multi-stage Dockerfiles + entrypoints
├── overlay/ann-benchmarks/   our algorithm modules (alisql is new)
├── harness/                  ops harness: drivers, workloads, metrics
├── orchestrator/             host-side: containers, limits, manifest
├── report/                   charts, Markdown and self-contained HTML
├── webui/                    web UI: HTTP server, JSON API, static front end
├── docs/                     the documents above
├── tests/                    unit tests + live engine-behaviour checks
└── sources/ work/ datasets/ results/     generated, gitignored
```

The vendor repositories are **read-only inputs** and are never modified.

---

## Licence

The framework is provided as-is for benchmarking purposes. MariaDB, AliSQL,
PostgreSQL, pgvector and ann-benchmarks are each under their own licences.
