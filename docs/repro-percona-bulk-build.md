# Reproducing Percona Server's bulk vector-index build crash

Percona Server's HNSW index can be built two ways: maintained on every INSERT
(`incremental`), or added after the load with

```sql
ALTER TABLE t1 ADD VECTOR KEY vi (v) TYPE hnsw (M = 16, ef_construction = 200, metric = euclidean), ALGORITHM=INPLACE
```

(`post`). At the old pin the second one killed the server. The client saw
`mariadb.InterfaceError: Lost connection to server during query` while the
ALTER was running, after the load had finished. That is why
`config/engines/percona.yml` declared `capabilities.bulk_vector_index_build:
false` and every benchmark built Percona's index incrementally until the
re-test at the current pin, below. It now declares `true`, and the ops pass
builds Percona with `post` unless a profile names a mode. The ann phase is
unchanged: its module declares the index in `CREATE TABLE`, so it always
builds incrementally.

## What was seen

On 2026-10-15, Percona 9.7.1-1 (the pin before the current one), built with `-march=native`.
The test was fashion-mnist-784-euclidean, 20,000 rows, one load thread, M=16,
ef_construction=200, normalized pass (8 CPUs, 16 GiB).

| Run | Result of the ALTER |
| --- | --- |
| `percona-20261015-183321` | built in 7.6 s, 100 MB index, recall@10 0.99 |
| `percona-20261015-203911` | lost connection: the server died |

The two runs used the same profile, the same image and the same data, so the
crash is intermittent. In the same runs, the same corpus built and served
correctly in `incremental` mode. mysqld's error log and any core were lost
with the container. Capturing both is what this procedure adds.

## At the current pin

Percona 9.7.2-2 (`source.tag` in `config/engines/percona.yml`), built
`-march=x86-64-v3`, on 2026-11-03, on a laptop (i9-12900HK), with this
procedure and the same profile: **25 of 25 attempts built.** The ALTER took
3.4-5.8 s (median 3.8 s). mysqld's error log held no error, signal or
assertion, and no core was written with `kernel.core_pattern` set to
`core.%p`.

25 clean builds on one machine do not prove the old crash gone, and nothing
here says what fixed it. They are enough to stop routing every run around it,
and the profile and script stay so the next pin can be tested the same way.

## On a clean machine

Ubuntu 24.04 or Debian 12, x86-64 with AVX2 (the image is built for
x86-64-v3), 8+ cores, 16 GB+ RAM. Building Percona Server from source takes
most of the time; each attempt after that takes a minute or two.

```bash
# 1. Prerequisites. This is the complete list.
sudo apt-get update
sudo apt-get install -y docker.io python3-yaml git curl
sudo usermod -aG docker "$USER"
newgrp docker                       # or log out and back in

# 2. The framework.
git clone https://github.com/Percona-Lab/vector-bench.git
cd vector-bench
chmod +x run-benchmark.sh scripts/*.sh

# 3. Build, download the 20k-row corpus's source dataset, and run 5 attempts.
#    It asks for sudo once, to set kernel.core_pattern (see below).
./scripts/repro-percona-bulk-build.sh

# More attempts, a bigger load, or an image that is already built:
ATTEMPTS=20 SKIP_BUILD=1 ./scripts/repro-percona-bulk-build.sh
ROWS=60000  SKIP_BUILD=1 ./scripts/repro-percona-bulk-build.sh

# The same ALTER at 1M rows (all of sift-128, config/profiles/percona-bulk-1m.yml):
PROFILE=percona-bulk-1m ATTEMPTS=1 SKIP_BUILD=1 ./scripts/repro-percona-bulk-build.sh
```

`PROFILE` names any profile in `config/profiles/`; the script reads the dataset
and row count from it, and `ROWS` still overrides the row count through a
temporary copy. `percona-bulk-1m` is the scale a full run asks of the bulk
build: sift-128's 1M rows loaded with four threads, then the one ALTER. Its
memory fits the normalized pass's 16 GB server limit with room to spare, so it
runs on a laptop; budget the load in minutes and the ALTER as unmeasured at
this size until the first run records it.

The script does four things:

1. Builds `vector-bench/percona-runtime` and `-bench` at the pinned commit with
   `-march=x86-64-v3`.
2. Fetches `fashion-mnist-784-euclidean`.
3. Runs `config/profiles/percona-bulk-repro.yml` `ATTEMPTS` times. Each attempt
   uses a fresh container and data directory, loads the rows with no vector
   index, and then runs the ALTER. (While the capability was `false` the
   framework warned that this mode was declared broken and ran it anyway; it
   no longer warns.)
4. For every crash, gets a backtrace from the core with gdb inside the bench
   image (`scripts/gdb-backtrace.py`). If any attempt crashed, it also copies
   out the `mysqld` binary that wrote the core.

## Cores

The framework already does three of the four things a core needs:

- mysqld runs with `--core-file` and `--innodb-buffer-pool-in-core-file=OFF`;
- the server container runs with `--ulimit core=-1`;
- after a crash, the core is copied out of the data volume before the
  container is removed.

The fourth is the host's `kernel.core_pattern`, which the framework never
changes. On stock Ubuntu it pipes cores to apport, which writes no file. The
script sets it to `core.%p` with sudo after the image build, right before the
attempts, and puts the old value back as soon as they finish (or on Ctrl-C).
It applies to the whole host while it is set. If sudo asks for a password
again at the restore and does not get it, the script prints the command that
puts the old value back. `NO_CORE=1` leaves it alone, and you get the logs without a core.

The image strips debug sections (`strip --strip-debug`) and keeps the symbol
table. So the backtrace has function names on every thread, but no line
numbers, arguments or locals. For line numbers, rebuild with the strip step in
`docker/percona/Dockerfile` removed.

## What comes out

Everything goes in `results/percona-bulk-repro-<timestamp>/`:

| File | What it is |
| --- | --- |
| `summary.txt` | one line per attempt: built (with the ALTER's time), CRASHED, or other failure |
| `versions.txt` | framework commit, Percona `source.tag`, image IDs, host |
| `<attempt>/logs/percona-ops-m16-post-normalized.log` | the client side: statements, timings, traceback |
| `<attempt>/logs/*server-error*` | mysqld's own error log, including the signal and its stack dump |
| `<attempt>/cores/` | the core file |
| `<attempt>/backtrace.txt` | `thread apply all bt full` from the core |
| `mysqld` | the binary the cores came from; a core can only be read against it |

To send it:

```bash
tar czf percona-bulk-repro.tar.gz -C results percona-bulk-repro-<timestamp>
```

## By hand, without the framework

The framework's `post` path runs these statements against a server started
by `docker/percona/entrypoint-percona.sh` with the flags in
`config/engines/percona.yml`. To try it on any Percona build:

```sql
CREATE TABLE t1 (
  id BIGINT UNSIGNED PRIMARY KEY,
  tag INT NOT NULL,
  ctag INT NOT NULL,
  v VECTOR(784) NOT NULL,
  KEY tag_idx (tag),
  KEY ctag_idx (ctag)
) ENGINE=InnoDB;
-- 20,000 rows: INSERT INTO t1 (id, tag, ctag, v) VALUES (?, ?, ?, ?), in
-- batches of 500, one connection, with v from fashion-mnist's training set.
ALTER TABLE t1 ADD VECTOR KEY vi (v) TYPE hnsw (M = 16, ef_construction = 200, metric = euclidean), ALGORITHM=INPLACE;
```

## Comparing the two build modes

Surviving the ALTER is not the same as building a good index.
`config/profiles/percona-build-modes.yml` runs both modes on all 60,000
fashion-mnist rows with queries: build cost, recall at the calibrated
operating point, 1 and 4 clients, and filtered search at 10% and 1%.

```bash
./run-benchmark.sh run --profile percona-build-modes --engines percona
```

The report's tables carry one row per build mode.

At the current pin, on 2026-11-03:

| | `post` | `incremental` |
| --- | --- | --- |
| load + build | 180.9 s + 14.1 s ALTER = 195 s | 288 s |
| index size | 289 MB | 256 MB |
| recall@10 at ef_search 10 | 0.9714 | 0.9709 |
| QPS, 1 / 4 clients | 5,895 / 19,233 | 5,599 / 18,103 |
| filtered recall, `tag` 10% / 1% | 0.988 / 0.996 | 0.989 / 0.997 |
| filtered recall, `ctag` 11% / 1.5% | 0.987 / 0.924 | 0.988 / 0.928 |

The same index quality, built in two thirds of the time.

## What was changed, and what was not

`capabilities.bulk_vector_index_build` is now `true` in
`config/engines/percona.yml`. That makes `post` the ops default for Percona
again, and `config/profiles/percona.yml` is back to
`percona_build_modes: [post, incremental]`.

**The churn repair is still withdrawn**, by a separate key,
`capabilities.vector_index_replace: false`. The repair is a different ALTER:

```sql
ALTER TABLE t1 DROP KEY vi, ADD VECTOR KEY vi (...), ALGORITHM=INPLACE
```

on a table that already has the index. At the old pin it returned in 1-4 ms
with the index byte-identical and recall identical in all 20 before/after
pairs of run `full-20261015-215006`. It was a no-op, and the report
published it as a repair. Nothing above tests it. It used to be withdrawn by
`bulk_vector_index_build: false`, on the grounds that both statements add
the index with `ADD VECTOR KEY ... ALGORITHM=INPLACE`, so flipping that one
key alone would have brought the no-op back.

Set `vector_index_replace: true` only after running the replace by hand at
the pin and seeing a real rebuild: `index_bytes` that changes, a time on the
order of the bulk build (seconds at 20,000 rows, not milliseconds), and recall
that moves on a churned table.
