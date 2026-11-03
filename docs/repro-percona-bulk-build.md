# Reproducing Percona Server's bulk vector-index build crash

Percona Server's HNSW index can be built two ways: maintained on every INSERT
(`incremental`), or added after the load with

```sql
ALTER TABLE t1 ADD VECTOR KEY vi (v) TYPE hnsw (M = 16, ef_construction = 200, metric = euclidean), ALGORITHM=INPLACE
```

(`post`). The second one has killed the server. The client sees
`mariadb.InterfaceError: Lost connection to server during query` while the
ALTER is running, after the load has finished. That is why
`config/engines/percona.yml` declares `capabilities.bulk_vector_index_build:
false` and every benchmark builds Percona's index incrementally.

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

It has not been tried at the current pin (`source.tag` in
`config/engines/percona.yml`). If no attempt crashes there, that is a result
too: the capability can be set back to `true`.

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

# 2. The framework, on this branch.
git clone https://github.com/EvgeniyPatlan/vector-bench.git
cd vector-bench
git checkout repro/percona-bulk-build-crash
chmod +x run-benchmark.sh scripts/*.sh

# 3. Build, download the 20k-row corpus's source dataset, and run 5 attempts.
#    It asks for sudo once, to set kernel.core_pattern (see below).
./scripts/repro-percona-bulk-build.sh

# More attempts, a bigger load, or an image that is already built:
ATTEMPTS=20 SKIP_BUILD=1 ./scripts/repro-percona-bulk-build.sh
ROWS=60000  SKIP_BUILD=1 ./scripts/repro-percona-bulk-build.sh
```

The script does four things:

1. Builds `vector-bench/percona-runtime` and `-bench` at the pinned commit with
   `-march=x86-64-v3`.
2. Fetches `fashion-mnist-784-euclidean`.
3. Runs `config/profiles/percona-bulk-repro.yml` `ATTEMPTS` times. Each attempt
   uses a fresh container and data directory, loads the rows with no vector
   index, and then runs the ALTER. The framework warns that this mode is
   declared broken and runs it anyway.
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
script sets it to `core.%p` with sudo and restores the old value when it
exits, including on Ctrl-C. While the script runs, the setting applies to the
whole host. `NO_CORE=1` leaves it alone, and you get the logs without a core.

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

## When it is fixed

Set `capabilities.bulk_vector_index_build: true` in
`config/engines/percona.yml`. That makes `post` the default again, restores
the churn repair path (the same ALTER), and lets `config/profiles/percona.yml`
go back to `percona_build_modes: [post, incremental]`.
