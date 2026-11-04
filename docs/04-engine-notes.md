# Engine notes

What each implementation actually does, and the traps in each. Everything here
was read out of the source trees or the engines' own test suites, not out of
marketing material.

## Quick reference

| | MariaDB 11.8 MHNSW | AliSQL 8.0.44-2 VIDX | pgvector 0.8 |
| --- | --- | --- | --- |
| Enable | on by default | `SET GLOBAL vidx_disabled=OFF` **and** session `READ-COMMITTED` | `CREATE EXTENSION vector` |
| Column type | `VECTOR(N)` | `VECTOR(N)`, N ≤ 16383 | `vector(N)` |
| Binary literal | raw LE float32 | raw LE float32 | `'[…]'::vector` |
| Text helper | `VEC_FromText` / `VEC_ToText` | `VEC_FROMTEXT` / `VEC_TOTEXT` | cast from text |
| Index DDL | `VECTOR INDEX vi (v) M=n DISTANCE=cosine` | `VECTOR INDEX vi (v) M=n DISTANCE=COSINE` | `USING hnsw (v vector_cosine_ops) WITH (m, ef_construction)` |
| **M** | index option (or `mhnsw_default_m`) | index option (or `vidx_hnsw_default_m`) | `WITH (m = …)` |
| **ef_construction** | **not exposed** | **not exposed** | `WITH (ef_construction = …)` |
| Search width | `mhnsw_ef_search` | `vidx_hnsw_ef_search` (default 20) | `hnsw.ef_search` |
| Graph cache | `mhnsw_max_cache_size` (**default 16 MiB**) | `vidx_hnsw_cache_size` (**default 16 MiB**) | none — `shared_buffers` |
| Distance fn | `VEC_DISTANCE_COSINE` / `_EUCLIDEAN` | same names | `<=>` / `<->` |
| Storage engine | InnoDB **or MyISAM** | **InnoDB only** | heap |
| Isolation | any | **READ COMMITTED required** | any |
| Graph location | companion table `<t>#i#<nn>` | InnoDB aux table `vidx_%016lx_%02x` | index relation |
| Build timing | incremental, on INSERT | incremental, on INSERT | **bulk, after load** |

---

## MariaDB — MHNSW

Source: `sql/vector_mhnsw.cc`, `sql/vector_mhnsw.h`. GA in the 11.7/11.8 series.

**Configuration surface.** Four plugin variables, declared at the bottom of
`vector_mhnsw.cc`:

```c
MYSQL_SYSVAR_ULONGLONG(max_cache_size, …)   // global
MYSQL_THDVAR_UINT(ef_search, …)             // session
MYSQL_THDVAR_UINT(default_m, …)             // session
MYSQL_THDVAR_ENUM(default_distance, …)      // session
```

plus two index options that default from the session variables:

```c
HA_IOPTION_SYSVAR("m",        M,      default_m),
HA_IOPTION_SYSVAR("distance", metric, default_distance),
```

So `M` can be given per index in DDL, which is what this framework does — it
matches how AliSQL takes `M` and keeps the two engines configured alike.

**Graph storage.** The HNSW graph lives in a hidden "high-level index"
companion table named `<table>#i#<nn>`, one row per node. Consequences: index
size is a filesystem measurement rather than a catalog lookup, and the graph
inherits the storage engine's page cache behaviour.

**Caching.** One cache per `TABLE_SHARE`, bounded by `mhnsw_max_cache_size`,
shared across sessions. Under concurrency all clients contend for the same
structure, which is what the concurrency workload is designed to expose.

**MyISAM.** MariaDB supports MyISAM for vector tables, and MariaDB's own
published benchmark uses it. AliSQL cannot, so this framework's headline
comparison is InnoDB for both, with MyISAM reported as a MariaDB-only extra
curve in the tuned pass.

**Not exposed:** `ef_construction`. Build quality is tunable only through `M`.
Verified on 11.8.8 — the only `mhnsw*` variables the server has are
`mhnsw_default_distance`, `mhnsw_default_m`, `mhnsw_ef_search` and
`mhnsw_max_cache_size`, and supplying it as an index option fails outright:

```
CREATE TABLE t (..., VECTOR INDEX vi (v) M=6 EF_CONSTRUCTION=200);
ERROR 1911 (HY000): Unknown option 'EF_CONSTRUCTION'
```

This is the most load-bearing claim in the methodology — it is why the
normalized pass pins pgvector's `ef_construction` — so it is stated with
evidence rather than asserted.

---

## AliSQL — VIDX

Source: `sql/vidx/vidx_hnsw.cc`, `sql/vidx/vidx_index.cc`, `sql/vidx/vidx_field.cc`,
headers under `include/vidx/`. Shipped in AliSQL-8.0.44-2, based on MySQL 8.0.44.

### Trap 1 — it is off by default

`vidx_disabled` defaults to `ON`. Any `VECTOR` column or `VECTOR INDEX` fails
with `ER_VECTOR_DISABLED` until you set it `OFF`. The runtime image passes
`--vidx-disabled=OFF` at startup.

### Trap 2 — READ COMMITTED is mandatory

Every vector operation raises `ER_NOT_SUPPORTED_YET` at any other isolation
level. This is a session setting and must be applied on every connection,
including every connection in a connection pool.

### Trap 3 — the optimizer will silently choose a full scan

AliSQL costs the vector index against a table scan and picks the scan when the
`LIMIT` is large relative to the row count. A full scan returns **exact**
results, so the symptom is recall ≈ 1.0 with collapsed throughput —
indistinguishable from "very accurate but slow" unless you check the plan.

**The threshold is a fraction of the table, not a fixed LIMIT**, and it does not
depend on `ef_search`. Measured on AliSQL-8.0.44-2 (`tests/verify-alisql-traps.sh`):

| Table size | index used | falls back to a scan |
| --- | --- | --- |
| 100 rows | `LIMIT` ≤ 25 | `LIMIT` ≥ 40 |
| 1000 rows | `LIMIT` ≤ 250 | `LIMIT` ≥ 290 |

Identical results at `vidx_hnsw_ef_search` = 20, 40 and 80 — the search width
plays no part in the decision. The crossover sits around **25–28% of rows**.

AliSQL's own `mysql-test/suite/rds/t/vidx_dml.test` appears to show a fixed
`LIMIT 16` / `LIMIT 17` boundary, and an earlier version of this document
repeated that reading. It is the same rule: that test deletes a third of its 100
rows first, so 17 of the ~67 remaining is ~25%.

**What this means in practice.** At benchmark scale the fallback does not
trigger — `k=10` against 60,000 rows is 0.017% of the table, three orders of
magnitude below the crossover. The plan guard is therefore cheap insurance
rather than a live hazard for these workloads. It stays on because the failure
is silent and would invalidate a whole run: every driver runs `EXPLAIN` per
configuration and records `vector_index_used`, and the report lists any
measurement where it was false in its Validity section, above the charts.

Mitigation when you need the index regardless of cost: `FORCE INDEX (vi)`.

### Trap 4 — the default graph cache is 16 MiB (and not only in AliSQL)

`vidx_hnsw_cache_size` defaults to 16777216 bytes — far below the graph size of
any realistic corpus, so an out-of-the-box AliSQL will thrash.

This is **not** an AliSQL-specific failing, and an earlier draft of this document
was wrong to imply it was. MariaDB ships the identical default: `SHOW VARIABLES
LIKE 'mhnsw%'` on 11.8.8 reports `mhnsw_max_cache_size = 16777216`. Both engines
degrade badly on their defaults. The framework sets both from the resource
profile so neither is judged on a value its vendor plainly intended you to
change.

### Values

`Field_vector::store(const char*, size_t, cs)` accepts a **binary string of
exactly 4 × dim bytes** — it validates the length and rejects NaN and Inf. So
binary binding works, and `VEC_FROMTEXT()` text parsing is avoidable. This
matters for fairness: charging AliSQL for text parsing that MariaDB does not
pay would be a client-side artefact, not an engine difference.

`VECTOR(N)` is stored as `varbinary(4N)` and presented that way in
`INFORMATION_SCHEMA.COLUMNS` — the data dictionary rewrites the type into a
versioned comment so tools that do not know `VECTOR` still see something valid.

### Caching

Two caches with different lifetimes (`vidx_hnsw.cc`):

- **MHNSW Share** — attached to the auxiliary table's `TABLE_SHARE`, shared by
  read-only transactions.
- **MHNSW Trx** — attached to the session via `thd_set_ha_data`; a read-write
  transaction keeps accessed and modified nodes in its own cache and updates
  the shared cache at commit.

This split is the most interesting structural difference from MariaDB's single
shared cache, and it should show up under mixed read/write concurrency.

### Other constraints

- InnoDB only.
- Vector index DDL cannot use `ALGORITHM=INPLACE`.
- Vector indexes cannot be `INVISIBLE`.
- Rows with `NULL` vectors are excluded from the index and sort last.
- HNSW layer assignment is randomised: replicas built from identical rows are
  not guaranteed to have identical graph topology.
- `M` range is 3–200; `ef_search` range is 1–10000.
- **Not exposed:** `ef_construction`.

---

## PostgreSQL + pgvector

Source: the pgvector extension, built against PostgreSQL 17.

**Bulk build.** The defining difference: pgvector builds its HNSW graph as one
operation after the data is loaded, parallelisable across
`max_parallel_maintenance_workers` and sized by `maintenance_work_mem`. MHNSW
and VIDX maintain their graphs on every INSERT and have no equivalent.

Comparing pgvector's bulk build against the others' incremental build compares
two different operations. This framework measures both, via `build_mode`:

- `post` — load, then `CREATE INDEX`. pgvector's idiomatic path.
- `incremental` — `CREATE INDEX` on the empty table, then load. Mirrors what
  MHNSW and VIDX are forced to do.

**No vector-specific cache.** Graph pages come through `shared_buffers` like
any other index pages. Mature and well-understood, but not vector-aware — there
is no equivalent of pinning the graph in memory. In the normalized pass
pgvector's buffer budget therefore absorbs the graph-cache fraction the other
two get separately; otherwise it would be handed strictly less resident memory
for the same container limit.

**`ef_construction`.** The one build-quality knob only pgvector has. Pinned to
its default of 200 in the normalized pass so pgvector is not given a tuning axis
the others lack; swept in the tuned pass and reported separately.

**Iterative index scan (0.8+).** `hnsw.iterative_scan` = `off` /
`relaxed_order` / `strict_order`. Without it, a selective `WHERE` can exhaust
the `ef_search` candidate list before k qualifying rows are found, and the query
returns fewer than k results. That is legitimate behaviour, but it makes
filtered recall ambiguous unless the mode is stated — so the framework sets it
explicitly and records it with every filtered measurement.

**Operator/opclass must match.** `<=>` requires `vector_cosine_ops`; `<->`
requires `vector_l2_ops`. A mismatch produces a sequential scan **with no error
and no warning** — pgvector's equivalent of AliSQL's silent fallback, and just
as capable of turning a benchmark into a brute-force measurement. Verified on
pgvector 0.8.6 against a cosine index over 5,000 rows:

```
-- matching operator: index used
EXPLAIN SELECT id FROM m ORDER BY embedding <=> '[…]' LIMIT 10;
  Limit
    ->  Index Scan using m_cos on m
          Order By: (embedding <=> '[…]'::vector)

-- mismatched operator against the same index: silently brute force
EXPLAIN SELECT id FROM m ORDER BY embedding <-> '[…]' LIMIT 10;
  Limit
    ->  Sort
          Sort Key: ((embedding <-> '[…]'::vector))
          ->  Seq Scan on m
```

The driver's plan check catches this: it requires the index name to appear in
the plan, and records `vector_index_used=false` otherwise.

**Storage.** `SET STORAGE PLAIN` on the vector column keeps vectors inline;
TOASTed vectors add a detoast to every distance comparison.

---

## Valkey — valkey-search

### Filtered search gets slower as the predicate narrows

valkey-search describes two strategies for a hybrid query: **pre-filtering**
(resolve the filter from the secondary index, then brute-force the survivors by
distance) and **inline filtering** (walk the HNSW graph, testing the predicate
as you go). Its documentation says the planner picks between them, and Google's
Memorystore documentation — which runs this module — says pre-filtering is
chosen "when the filtered search space is much smaller than the original".

Measured, it does not. Latency rises as the predicate tightens, which is the
signature of the graph walk: with one node in a hundred qualifying, the walk
travels much further to find ten of them.

| Selectivity | p50 | QPS |
| --- | --- | --- |
| unfiltered | — | 479 |
| 10% | 18.16 ms | 53.5 |
| 1% | 120.56 ms | 8.2 |

9,900 vectors out of 990,000 is a search space where an exact scan takes a
couple of milliseconds. It took 120.

**There is no way to ask for the other strategy.** RediSearch exposes exactly
this choice as `HYBRID_POLICY ADHOC_BF | BATCHES`; valkey-search's parser
rejects the keyword outright:

```
Error parsing vector similarity parameters:
`[KNN 10 @embedding $vec EF_RUNTIME 100 HYBRID_POLICY ADHOC_BF]`.
Unexpected argument `HYBRID_POLICY`
```

Two things about our own query were ruled out before concluding that. Dropping
`EF_RUNTIME` does not reach the pre-filter — it lowers search effort, and recall
falls with it. Replacing the open `[-inf (threshold]` bound with a bounded
`[0 max]` range, in case an unbounded side defeated cardinality estimation,
changes nothing: 136.72 ms against 136.72 ms.

Reproduce all of it in about five minutes:

```bash
./run-benchmark.sh lab --engine valkey probe-valkey-filter.py --rows 200000
```

The probe generates random vectors, so its *recall* is not comparable to a run's
— random high-dimensional vectors are nearly equidistant and any HNSW graph over
them is poor. Only the comparison between shapes, and the direction across
selectivities, carries over. Both reproduce the run: 5.2x slower at 1% than at
10% in the probe, 6.6x in the run.

---

## MyVector — hnswlib as a MySQL plugin

Source: `askdba/myvector` release v1.26.9, commit `384d9df7`, built in-tree
into MySQL `mysql-9.7.2` (the server line Percona Server's `9.7` series is
based on). Symbols below are in that tree's `src/myvector.cc`,
`src/myvector_binlog.cc`, `src/myvector_plugin.cc`, `include/hnswdisk.h` and
`sql/myvectorplugin.sql`. Driver: `harness/drivers/myvector.py`.

The only engine here whose index is not part of the server's storage layer.
The graph is an hnswlib graph held in the plugin's memory and saved to files
in `myvector_index_dir`; the table is an ordinary InnoDB table with a native
`VECTOR(n)` column. How each piece works, and the traps in it:

| | MyVector v1.26.9 |
| --- | --- |
| Declared by | column `COMMENT 'MYVECTOR Column \|type=HNSW,dim=…,size=…,M=…,ef=…,dist=…,track=updatets'` |
| Built by | `CALL mysql.myvector_index_build('db.t.col','pk')`, after the load |
| Searched by | `WHERE MYVECTOR_IS_ANN('db.t.col','pk', vec, 'nn=k,ef_search=e')`, rewritten before parsing |
| **M** / **ef_construction** | `M=` / `ef=` column options (exact case) |
| Search width | `ef_search=` inside each call; hnswlib searches at `max(ef, k)` |
| Cosine | `dist=Cosine` → `AngularDistanceSpace`, a **scalar** loop (`computeCosineDistance`); L2 uses hnswlib's SIMD `L2Space` |
| Capacity | `size=` is hnswlib's fixed `max_elements`; `addPoint` throws past it, inside mysqld |
| Writes after the build | `myvector_index_refresh` over a `track=` timestamp — **adds only** (below) |
| Online (binlog) maintenance | documented; **exits after its first idle second** on this server (below) |
| Filtered search | **none** |

### Trap 1 — options are case-sensitive and unknown spellings are silent

`HNSWMemoryIndex::HNSWMemoryIndex` reads `getIntOption("M", 16)` and compares
`getOption("dist") == "Cosine"`. `m=32` builds M=16; `dist=cosine` builds L2.
Nothing warns. `MYVECTOR_INDEX_STATUS` prints `M` as the effective value but
`Distance` **as written**, and does not print `ef` at all. The driver writes
the exact spellings, then reads status back and refuses a build whose `Type`,
`M`, `Dimension`, `Distance`, capacity or row count is not what it declared.

### Trap 2 — a failed build is a result row, not an error

`myvector_index_build` opens its own connection back to the server using
`myvector.cnf` (`BuildMyVectorIndexSQL`); the file must be mode 0600/0400 or it
is refused. Every failure — no file, refused connection, failed save — comes
back as text in the procedure's result set and the `CALL` succeeds. The driver
requires the text to start `SUCCESS` and to report the loaded row count.

### Trap 3 — the binlog listener does not survive idleness, and never deletes

`docs/ONLINE_INDEX_UPDATES.md` describes `online=Y` as real-time maintenance
from the binary log, applying inserts, updates and deletes. Two things are
true instead, the first observed in this image and the second read from the
source:

- **It dies.** `myvector_binlog_loop` opens the stream with a 1-second
  `MYSQL_OPT_READ_TIMEOUT` and treats any fetch error whose text lacks
  `timed out` as fatal. Against MySQL 9.7.2 the idle-timeout error text is
  empty, so the server log shows `Binlog fetch failed: ` and `Exiting binlog
  func` about a second after startup, no `Binlog Dump` thread remains, and a
  row inserted after that never reaches the graph. Nothing restarts it.
- **It only inserts.** The loop dispatches `TABLE_MAP_EVENT` and
  `WRITE_ROWS_EVENT` and nothing else, and `myvector_table_op` only ever calls
  `insertVector`. (Its row parser also reads the key from a `MYSQL_TYPE_LONG`
  column only, so a BIGINT key would be indexed as 0.)

So the framework does not use `online=Y` and the engine gets no binlog
requirement: its server flags log exactly as Percona Server's do. Writes after
the build reach the index through the release's other documented incremental
path, a `track=` timestamp column and `myvector_index_refresh`, which the
churn workload's `wait_index_synced` hook runs after each re-insert and times
(`index_sync_s`). The refresh selects rows changed since the last build or
refresh at one-second resolution, so the driver waits out the watermark
second after each build and refresh — at most one second, inside the timed
maintenance — or a row written in that second would never be selected.

**`index_sync_s` is therefore not an incremental sync, and is not comparable
to another engine's.** A refresh is `myvector_open_index_impl` (action
`refresh`) → `BuildMyVectorIndexSQL`, which inserts the changed rows and then
calls `saveIndex(myvector_index_dir, "build")` — a full rewrite of the graph
file — after which `myvector_open_index_impl` calls `saveIndex(…, "refresh")`,
an incremental checkpoint, on top. So the figure is the re-inserted rows plus
one whole-graph write plus a checkpoint, and it grows with the corpus rather
than with the churn. It is one procedure call, so the parts cannot be timed
apart from SQL. The driver stamps this on every record
(`extra.index_sync_semantics`) and the report prints it under §9 Churn and in
Validity.

**Neither path removes a deleted row.** It stays in the graph; the search
still returns its id; the rewritten `id IN (…)` drops it, and the query
returns fewer than k rows. That is the degradation churn measures. Every
re-insert also consumes a graph slot for good, so the driver sizes the graph
at 1.5× the loaded rows and refuses a re-insert that would overflow it rather
than let hnswlib throw inside mysqld. The repair is a full
`myvector_index_build`, which takes `LOCK TABLES … READ` for its duration.

### Trap 4 — the rewrite is textual

The audit-plugin rewrite (`myvector_query_rewrite`) fires only on a statement
whose first character is `S`/`E`/`C`/`A` and that matches `^SELECT\s+` or
`^EXPLAIN\s+` — a leading comment or space disables it. It finds the end of
`MYVECTOR_IS_ANN(…)` by counting parentheses over the raw text, quotes
included, so an escaped binary vector can derail it with the bytes 0x28/0x29.
Worse, **a long statement kills the server**: with the query vector inline as a
hex literal, a 100-dim search (a 0.9 KB statement) works and a 784-dim one
(6.4 KB) makes mysqld exit with nothing in its error log — reproduced on this
image, and the cause of the first smoke run's failure. The same search with
the vector in a user variable works. So every query is two statements, `SET
@vb_q = X'…'` (no `MYVECTOR` in it, so the rewrite returns at its first
`strstr`) and the search over `@vb_q` — a second round trip per query that no
other engine here pays, charged by the plugin's design. `ef_search` mutates the shared index
(`setSearchEffort`), so it is sent on every query; omitting it searches at
whatever another session last asked for.

Both query-path asymmetries — the second round trip, and Cosine's scalar
distance loop (`AngularDistanceSpace` → `computeCosineDistance`; L2 runs
hnswlib's SIMD `L2Space`) — are recorded on every record
(`extra.round_trips_per_query`, `extra.distance_impl`, set by
`MyVectorDriver.query_path_facts` on the ops path and by the overlay module's
`get_additional` on the ann path) and footnoted by the report under §4, §7 and
in Validity.

### Trap 5 — dropping the table does not drop the index

`DROP DATABASE` leaves the plugin's in-memory index object alive under the
same `db.table.col` name, and the next build reuses that object — with the
dimension, M and ef it was constructed with. The driver calls
`myvector_index_drop` first, and the status check above would catch a stale
object anyway.

### No filtered search

`myvector_ann_set_init` accepts three or four arguments and the options string
carries `nn` and `ef_search` only. A predicate beside `MYVECTOR_IS_ANN`
filters the k results after the search (the release's own
`docs/LIMITATIONS.md` says so). The driver declares the `filtered` workload
declined (`EngineDriver.declined_workloads`); the harness skips it, prints the
reason and stamps `extra.declined_workloads` on every record of the run.

### Build mode

`post` only: load, then one `myvector_index_build`. There is no incremental
mode to compare it with, because the only write-path maintenance is the dead
binlog listener; the driver refuses `build_mode: incremental` with that
reason. `myvector_index_bg_threads` is the build's parallelism — 1 in the
normalized pass, `{build_threads}` in the tuned one; the plugin only builds in
parallel at 2 or more (`HNSWMemoryIndex::startParallelBuild`).

### Memory: the graph is outside the buffer pool, and the build holds the table twice

The graph is hnswlib's, in mysqld's own heap: it is not InnoDB pages, it is not
bounded by `innodb_buffer_pool_size` or by any MyVector variable, and it cannot
be paged out — the whole graph is resident or the search does not run. It IS
charged the same way as every other engine's memory: it is mysqld's address
space, inside the server container, so the container memory limit bounds it
and the run's `mem-*.jsonl` sampler and peak-RSS figures include it. What
differs is how the same budget is split. The normalized pass gives every
MySQL-family server a buffer pool of 30% of 16 GiB (4.8 GiB) and MariaDB/AliSQL
a graph cache of another 30%; MyVector's buffer pool serves only the table and
the k primary-key lookups per query, and its graph takes whatever is left.

Size, per element, is `MYVECTOR_INDEX_STATUS`'s `Element Data Size` — the
vector, the level-0 links (`2·M` 4-byte ids plus a count) and the 8-byte label;
3,276 B at 784 dims and M=16, measured — plus an allowance of 100 B for the
per-element lock, level and pointer arrays and the label map (an allowance,
not a measurement). The build additionally keeps the WHOLE `(id, vector)`
result set client-side inside mysqld while it inserts (`BuildMyVectorIndexSQL`:
`mysql_store_result`), counted as `4·dim·rows`. So the build peak is buffer pool
+ graph + that copy, with no further margin. `harness/memory_fit.py` computes
it; these rows are its output at M=16 and the normalized pass's 16 GiB limit
(30% buffer pool), and `tests/test_e25_myvector_memory_fit.py` fails if they
drift from it:

| Corpus | Graph | Build transient | Buffer pool | Build peak | Against 16 GiB |
| --- | --- | --- | --- | --- | --- |
| gist-960 | 3.8 GiB | 3.6 GiB | 4.8 GiB | **12.2 GiB** | fits |
| dbpedia-openai-1000k | 5.9 GiB | 5.7 GiB | 4.8 GiB | **16.4 GiB** | does **not** fit |

(dbpedia-openai-1000k is 1536-d and 990,000 training rows; gist-960 is 960-d
and 1,000,000.) Nothing beyond those parts is added — mysqld's own baseline,
the ann-benchmarks client sharing the container on the ann path, and
page cache are not in the figure — so a peak near the limit is a peak that
will not survive; a refusal at 16.4 GiB against 16.0 is not a close call.

**The orchestrator refuses a unit whose estimated peak is over the run's
resolved server limit** (`cli._memory_refusal`, gated on
`capabilities.memory_fit: myvector`), before the load, for both the ann and the
ops phase, with the rows that phase loads (`ops.subset_rows` for ops) and the
largest M it builds. The manifest records the phase as `refused` with the
reason — `does not fit: estimated peak X GiB > limit Y GiB at M=32 (largest of
[8, 16, 32])` — and the report lists it in Validity under "Units refused before
they ran", separately from failures, and the run's finish banner counts them.
The ann grid is `ann_pass.ann_m_values`, the function the ann render itself
uses, so a tuned `m_values_by_engine` override is honoured; the ops grid is
`ops.m_values`. **A unit is refused as a whole when its largest M does not
fit**, even if its smaller M values would: one unit is one phase over the
whole grid, and splitting it would change what a checkpoint means. That is
conservative, and the reason text names the M and the grid so the reader can
see it. A phase the profile disables (`ann.enabled: false`, `ops.enabled:
false`) is never checked and never shows a refusal. **The check compares
against the server container's limit only**, not the client's and not the
host's: the graph and the build's copy of the vectors live in mysqld, so the
server's memory cgroup is the one that OOM-kills it, and the ann-benchmarks
client that shares the container on the ann path is sized separately by
`ann_pass.client_memory_bytes`. No default budget changed: the normalized pass stays at 16 GiB.
dbpedia under `mariadb-blog`'s own 64 GiB, or under the tuned pass's host
fraction, fits and runs. Other engines are never refused: their graphs are
caches or InnoDB pages that degrade instead of killing the server.

The build record carries `extra.myvector_graph_bytes_floor` (level-0 bytes,
measured from the server's status) and every record carries
`extra.memory_note`; the report states both under §5 and in Validity. One
budget question remains the user's to decide: whether MyVector's normalized
buffer pool should give up the graph-cache share MariaDB and AliSQL get, so
its graph has the same headroom.

---

## VillageSQL — vsql_vector built from vsql-vector-pk (SVECTOR + HNSW custom index)

Server `villagesql/villagesql-server` tag `publish/mysql-9.7_0.0.7` (`533ab4b5`,
VillageSQL 0.0.7 on MySQL 9.7.2) with the `villagesql/vsql-vector-pk` extension
at release `0.0.3` (`93ed0087`). vsql-vector-pk is the build of vsql-vector
whose HNSW index stores each row's primary key; it installs as `vsql_vector`,
and a server runs one or the other. It builds against the server's build tree
(`VillageSQL_BUILD_DIR`), -O3, at `x86-64-v3`. `villagesql84` (below) builds the
same extension commit against the MySQL 8.4 line.

"This pin" in the rows below is vsql-vector-pk `0.0.3`. Its measurements were
taken on the 8.4 build (`villagesql84`'s server, `689c74a9`), whose KNN
recognition, custom-index DML layer and storage ABI are the same files as 9.7's
(see the 8.4 section). On 9.7 this framework built vsql-vector-pk `d23b87c4`
(`0.0.3` without its version bump) against `533ab4b5` and ran the gate profile
through both phases with the plan check passing, and the 8.4-built
`vsql_vector.veb` installed into the 9.7 server and planned the distance scan.

The engine's first pin was server `publish/mysql-9.7_0.0.7` (`533ab4b5`) with
plain vsql-vector `bac885dd`, measured on 2026-10-30. Where a row below says
"first pin", that is the build it was measured on.

| | VillageSQL + vsql_vector (vsql-vector-pk) |
| --- | --- |
| Enable | `INSTALL EXTENSION vsql_vector` once (the entrypoint's bootstrap), and `--vsql-allow-preview-extensions=ON` on every start: on the first pin, once the extension was installed a server started without it **refused to start** (`Failed to load VEF extension 'vsql_vector': extension requires preview capabilities but vsql_allow_preview_extensions is OFF`, then `Aborting`), and `SET GLOBAL` of it was refused (`must be set with SET PERSIST`) |
| Column type | `SVECTOR(N)`, stored by the extension in its own InnoDB pages |
| Value input | the text literal `'[a,b,…]'`, parsed with `strtof` per element; that is what the driver and the ann module send, the query vector inlined. vsql-vector-pk also accepts a binary form (a tagged big-endian dimension, then little-endian float32; `svector_from_binary`, src/vector.cc), which this engine does not use |
| Index DDL | `INDEX vi (v hnsw_l2) USING EXTENDED(hnsw) WITH (M = 16, ef_construction = 200)` inline, or `CREATE INDEX vi ON t1 (…)` |
| Metric | chosen by the key part's profile: `hnsw_l2`, `hnsw_cosine`, `hnsw_l1`, `hnsw_inner_product` |
| **ef_construction** | `WITH (ef_construction = …)`, default 200, rejected above 32 × M and below M (`Options::validate`, src/index/hnsw/storage.cc; M itself is 2..128; the same constants as plain vsql-vector's). On the first pin a refusal said only `InnoDB: Custom type operation failed. See server error log for details.` (M=4: 128 accepted, 129 refused). So 200 is not buildable at M=6 or below. The engine's first version clamped to 192 and recorded both values, but the result file and every label downstream still said 200; the combination is now dropped from the render by name (`orchestrator/ann_knobs.py`), and the driver and module refuse it if anything bypasses that. m-sweep gives VillageSQL M 8..32 for this reason |
| Search width | `SET SESSION vsql_vector.ef_search = n` (default 40, 1..65536); the scan searches at `max(k, ef)` |
| Distance fn | `L2_DISTANCE`, `COSINE_DISTANCE`, `L1_DISTANCE`, `INNER_PRODUCT` |
| Graph location | extension segments inside `t1.ibd`; on the first pin the catalog reported one page for `vi` |
| Build timing | incremental on INSERT, or bulk `CREATE INDEX` — see the third trap |

### The traps

1. **The index is used only if it stores row references.** The server
   drives a distance scan ("Custom index distance scan on vi" in EXPLAIN
   FORMAT=TREE) only from an index whose storage props include
   `HAS_ROW_REF` (`FindCustomKnnIndexOnField` in
   `villagesql/sql/custom_index_knn_recognition.cc`, gate from server PR
   #1269). Plain vsql-vector's HNSW declares `HAS_COLUMN_REF | REF_LOOKUP`:
   on the first pin `SELECT id FROM t1 ORDER BY L2_DISTANCE(v, '[…]') LIMIT
   10` planned as `Sort` over `Table scan on t1` (TRADITIONAL: `type ALL`,
   `Using temporary; Using filesort`), with the vector inlined or bound,
   under `hypergraph_optimizer=off` and `=on`, and recall was 1.000 at
   ef_search 10, 40, 160 and 400 alike — exact brute force. vsql-vector-pk
   declares `HAS_ROW_REF`, and on this pin the plan is `Custom index distance
   scan on vi` under the default (classic) optimizer; on a 300 × 8 probe its
   top 5 matched exact brute force. `VillageSQLDriver` and
   the ann module still check the plan and **raise instead of timing the
   query** without it, so a pin that loses the scan fails loudly into the
   report's failed phases rather than publishing brute force as ANN search.
   The query vector must be a constant (`BindFieldAndQuery`): a bound
   parameter is not one at optimisation time, so it is inlined.
2. **DELETE is rejected.** `DELETE FROM t WHERE id = 1` → `DELETE on a table
   with a custom index (USING EXTENDED) is not supported yet.` Churn is
   declined (`declined_workloads`, `capabilities.churn: false`). An UPDATE of
   `v`, and of `tag` (which carries its own B-tree index), is rejected too:
   `UPDATE that changes an indexed column on a table with a custom index
   (USING EXTENDED) is not supported yet.` — any indexed column, not only the
   vector. Nothing here issues an UPDATE. All three measured on this pin, as
   on the first.
3. **The bulk build can kill the server.** On the first pin, `CREATE INDEX vi
   ON t1 (v hnsw_…) USING EXTENDED(hnsw)` over a loaded table worked and was
   fast, and in about one build in seven it deadlocked: two threads X-latched
   extension pages at the storage ABI's page get
   (`storage/innobase/villagesql/storage_abi.cc`) and waited on each other,
   InnoDB's monitor printed `DEADLOCK of threads detected!` and aborted
   (`[FATAL]`, `sync0arr.cc`, signal 6); never in 12 incremental builds.
   vsql-vector-pk's README says building the index over a populated table
   "is not safe yet and can crash the server". So
   `capabilities.bulk_vector_index_build: false` and the build mode is
   `incremental`; a profile may still ask for `post` to re-test a new pin.
4. **A WHERE beside the KNN ORDER BY is a post-filter.** The scan returns its
   top `max(ef_search, LIMIT)` candidates and the predicate filters those
   (vsql-vector-pk README, "Filtering with a WHERE clause"), so a selective
   predicate returns fewer than k rows even when enough matching rows exist.
   Filtered search is declined: measuring it would publish post-filtered
   recall as filtered search.
5. **Text binding is a cost only this engine pays**, on the load and inside
   every timed query; `extra.vector_binding` carries it and the report states
   it. The shortest float32 repr per element round-trips exactly through
   `strtof`, so the stored vectors are bit-identical to the dataset's.
6. **Ingest slowed within one server lifetime on the first pin.** Repeated
   drop-and-reload cycles of a 3,000 × 16 incremental build on one server ran
   at ~290, ~330, ~330, ~150, ~120, ~125 rows/s; a restarted server started
   fast again. Unexplained, and not re-measured on this pin; the framework
   restarts the server per configuration, so it does not reach a
   measurement, but a hand-run loop may see it.

### The MySQL 8.4 line — `villagesql84`

VillageSQL's team asked for their 8.4 build to be benchmarked rather than 9.7, so
`villagesql84` compiles the **same** vsql-vector-pk commit (`93ed0087`) against server tag
`publish/mysql-8.4_0.0.7` (`689c74a9f88c91ae035f57e0ae8915a168274112`, the same commit as
`release/0.0.7`; MySQL 8.4.11 base — not the Percona Server 8.4 based line). Its config is
`alias_of: villagesql`: the same Dockerfile, entrypoint, init.sql, driver, cmake flags,
server flags and capabilities, so the server line is the only difference between the two
curves. It has its own `ann_constructor` (`VillageSQL84`) and its own image names
(`vector-bench/villagesql84-*`), because both engines' images are tagged with the extension
commit and would otherwise overwrite each other.

What was checked in the 8.4 tree, and found the same as 9.7:

- **The build.** Every cmake flag in the 9.7 config exists in 8.4, Boost is bundled in
  `extra/boost` (1.84) as it is in 9.7, and `villagesql/sdk` — the headers the extension
  includes, dev ABI and all — is byte-identical between the two tags (all 69 blobs). The
  extension therefore needs no change, and both engines pin the same commit.
- **The server surface the driver uses.** `vsql_allow_preview_extensions`, the DELETE /
  UPDATE refusal text, `EXPLAIN FORMAT=TREE`, `information_schema.INNODB_TABLESPACES`
  and the performance_schema wait instruments all exist. `L2_DISTANCE` /
  `COSINE_DISTANCE` are the extension's functions, not the server's, so 8.4's lack of a
  native vector type does not matter. 8.4 still ships `mysql_native_password`, disabled;
  the bench account stays on `caching_sha2_password` as on 9.7.
- **Trap 1 holds on 8.4.** `FindCustomKnnIndexOnField` in
  `villagesql/sql/custom_index_knn_recognition.cc` is the same file on both tags and skips
  any index without `VEF_INDEX_STORAGE_HAS_ROW_REF`. The only differences in `villagesql/`
  between the tags are build scripts and the 8.4 iterator API (`Init`/`Read` where 9.7 has
  `DoInit`/`DoRead` in `custom_index_knn_scan.cc`). vsql-vector-pk declares `HAS_ROW_REF`,
  and the measurements in the rows above were taken on this 8.4 build: the KNN query plans
  as `Custom index distance scan on vi`. At the previous pin, plain vsql_vector `bac885d`
  (`HAS_COLUMN_REF | REF_LOOKUP`, `src/vector.cc`), the KNN query was a filesort over a full
  scan on 8.4 as on 9.7 and the plan check refused it. **Measured** at that pin on the built
  image (2026-11-04, `8.4.11-villagesql-0.0.7-dev-vector-bench`, 2,000 × 16 rows, index
  declared inline at M=16, ef_construction=200):
  `EXPLAIN FORMAT=TREE SELECT id FROM t1 ORDER BY L2_DISTANCE(v, '[…]') LIMIT 10` is `Limit`
  → `Sort: vsql_vector.l2_distance(…)` → `Stream results` → `Table scan on t1`, TRADITIONAL
  is `type ALL … Using temporary; Using filesort`, the same at `vsql_vector.ef_search = 10`,
  and `DELETE` fails with the 9.7 text. One difference from 9.7: the hypergraph optimizer
  cannot be switched on at all in a non-debug 8.4 build
  (`ERROR 3999 … does not yet support 'use in non-debug builds'`), so "both optimizers" is
  one optimizer here. The `dev` profile on this engine produced its `ingest` and
  `index_build` records (incremental, 595 and 1,130 rows/s on the 4,000 × 16 tiny corpus in
  the normalized and tuned passes) and the concurrency workload raised `BruteForceRefused`
  in both passes, as designed.
- **Trap 3** (the bulk-build deadlock) was measured on 9.7 only. Every file under
  `storage/innobase/villagesql/` — `storage_abi.cc`, where the two latches were taken,
  among them — is byte-identical between the tags, so `bulk_vector_index_build: false` is
  kept rather than assumed fixed.

---

## Things that are true of all of them

- **The optimizer can always decline the index.** Every one of them will fall
  back to exact brute force under some conditions, and every one makes that
  look like high recall at low throughput. Check the plan, every time.
- **Metric is fixed at index creation.** You cannot query a cosine index with a
  Euclidean distance function and get index acceleration.
- **Recall is not deterministic across rebuilds.** HNSW construction involves
  randomised layer assignment; two indexes built from identical data will not
  have identical graphs, and recall will differ slightly.
- **Warm-up matters.** The first queries after startup pay for cold graph
  caches. Measuring them as steady state understates every engine, and
  understates the ones with larger caches most.
