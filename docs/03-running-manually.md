# Running the engines by hand, without the framework

This document is deliberately standalone. Nothing here depends on
`run-benchmark.sh`, on the ops harness, or on ann-benchmarks. If you only want
to try MariaDB, AliSQL and pgvector vector search yourself — or to verify a
number the framework produced — this is enough.

Each engine has a `*-runtime` image containing only the server. Build those
first (see §1), then jump to whichever engine you care about.

---

## 1. Build just the runtime images

```bash
cd vector-bench
./scripts/prepare-sources.sh              # export sources at their pinned tags
./scripts/build-images.sh --target runtime
```

That produces:

| Image | Contains |
| --- | --- |
| `vector-bench/mariadb-runtime:mariadb-11.8.8` | MariaDB server, `/opt/mariadb` |
| `vector-bench/alisql-runtime:AliSQL-8.0.44-2` | AliSQL server, `/opt/alisql` |
| `vector-bench/pgvector-runtime:v0.8.6` | PostgreSQL 17 + pgvector |

Build one at a time with `--engine mariadb`. Rough cold-build times: pgvector
~10 minutes, MariaDB ~40 minutes, **AliSQL 1.5–3 hours**. AliSQL is slow because
its CMake compiles the bundled DuckDB unconditionally on Linux, with its output
redirected to `/dev/null` — the build looks stalled but is not.

> Every image is built with the same `-march` so the distance kernels get the
> same SIMD instructions. If you rebuild one engine with a different `-march`,
> you are no longer comparing implementations. Pass `--march` explicitly to be
> sure: `./scripts/build-images.sh --target runtime --march x86-64-v3`.

---

## 2. MariaDB — MHNSW

### Start it

```bash
docker run -d --name mdb \
  -p 3306:3306 \
  vector-bench/mariadb-runtime:mariadb-11.8.8

# wait for readiness
until docker exec mdb /opt/mariadb/bin/mariadb -ubench -pbench \
      --socket=/var/run/vbench/mariadb.sock -e 'SELECT 1' >/dev/null 2>&1; do
  sleep 1
done
```

Server flags can be added through `VB_SERVER_ARGS`, e.g.

```bash
docker run -d --name mdb \
  -e VB_SERVER_ARGS="--innodb-buffer-pool-size=4G --mhnsw-max-cache-size=4G" \
  -p 3306:3306 vector-bench/mariadb-runtime:mariadb-11.8.8
```

### Open a client

```bash
docker exec -it mdb /usr/local/bin/vb-entrypoint client
```

### Create a table and index

```sql
CREATE DATABASE demo;
USE demo;

CREATE TABLE items (
  id  INT PRIMARY KEY,
  tag INT NOT NULL,
  v   VECTOR(3) NOT NULL,
  VECTOR INDEX vi (v) M=6 DISTANCE=cosine
) ENGINE=InnoDB;
```

`M` and `DISTANCE` are index options. They can also be defaulted per session
with `mhnsw_default_m` and `mhnsw_default_distance`.

### Insert

MariaDB stores `VECTOR` as packed little-endian float32, and accepts a binary
string directly:

```sql
INSERT INTO items VALUES (1, 0, 0xCDCC8C3FCDCCCC3F9A99993F);   -- [1.1, 1.6, 1.2]
INSERT INTO items VALUES (2, 1, VEC_FromText('[0.2, 0.1, 0.4]'));
```

From an application, bind a 4×dim byte string. In Python:

```python
import numpy, mariadb   # Connector/Python 2.0, as the harness uses it
conn = mariadb.connect(host="127.0.0.1", port=3306, user="bench", password="bench",
                       binary=True)
cur = conn.cursor()
cur.execute("USE demo")
vec = numpy.array([0.1, 0.2, 0.3], dtype="<f4").tobytes()
cur.execute("INSERT INTO items (id, tag, v) VALUES (?, ?, ?)", (3, 0, vec))
conn.commit()
```

### Search

```sql
SET mhnsw_ef_search = 100;

SELECT id, VEC_ToText(v) AS vector,
       VEC_DISTANCE_COSINE(v, VEC_FromText('[0.1,0.2,0.3]')) AS distance
FROM items
ORDER BY VEC_DISTANCE_COSINE(v, VEC_FromText('[0.1,0.2,0.3]'))
LIMIT 10;
```

### Confirm the index is actually used

**Do this every time.** If the optimizer picks a full scan you get exact
results at brute-force speed, which looks like excellent recall and terrible
throughput rather than like a mistake.

```sql
EXPLAIN SELECT id FROM items
ORDER BY VEC_DISTANCE_COSINE(v, VEC_FromText('[0.1,0.2,0.3]')) LIMIT 10;
```

The `key` column must name your vector index (`vi`). If it does not, force it:

```sql
SELECT id FROM items FORCE INDEX (vi)
ORDER BY VEC_DISTANCE_COSINE(v, VEC_FromText('[0.1,0.2,0.3]')) LIMIT 10;
```

### Knobs

| Variable | Scope | Meaning |
| --- | --- | --- |
| `mhnsw_ef_search` | session | Search width. The recall/speed dial. |
| `mhnsw_default_m` | session | Default graph degree for new indexes. |
| `mhnsw_default_distance` | session | Default metric for new indexes. |
| `mhnsw_max_cache_size` | global | Graph cache ceiling, in bytes. |

MariaDB does **not** expose `ef_construction`.

---

## 3. AliSQL — VIDX

### Start it

```bash
docker run -d --name ali \
  -p 3307:3306 \
  vector-bench/alisql-runtime:AliSQL-8.0.44-2

until docker exec ali /opt/alisql/bin/mysql -ubench -pbench \
      --socket=/var/run/vbench/alisql.sock -e 'SELECT 1' >/dev/null 2>&1; do
  sleep 1
done
```

### Two things that will bite you

AliSQL ships vector support **disabled**, and every vector operation requires
**READ COMMITTED**. Without both, DDL fails with `ER_VECTOR_DISABLED` and DML
with `ER_NOT_SUPPORTED_YET`.

```sql
SET GLOBAL vidx_disabled = OFF;                          -- once per server
SET SESSION transaction_isolation = 'READ-COMMITTED';    -- every session
```

The runtime image already passes `--vidx-disabled=OFF` at startup, but the
session-level isolation is yours to set.

### Create a table and index

```sql
CREATE DATABASE demo;
USE demo;

CREATE TABLE items (
  id  INT PRIMARY KEY,
  tag INT NOT NULL,
  v   VECTOR(3) NOT NULL,
  VECTOR INDEX vi (v) M=6 DISTANCE=COSINE
) ENGINE=InnoDB;         -- InnoDB is the only supported engine
```

Or after the fact:

```sql
CREATE VECTOR INDEX vi ON items (v);
```

Vector indexes cannot use `ALGORITHM=INPLACE` and cannot be `INVISIBLE`.
`VECTOR(N)` allows N up to 16383, subject to the InnoDB row-size limit.

### Insert

Like MariaDB, AliSQL's `Field_vector` accepts a binary string of exactly
4 × dim bytes, so binary binding works and avoids text parsing:

```sql
INSERT INTO items VALUES (1, 0, VEC_FROMTEXT('[0.1,0.2,0.3]'));
```

```python
import numpy, mariadb   # Connector/Python 2.0 speaks the MySQL protocol
conn = mariadb.connect(host="127.0.0.1", port=3307, user="bench", password="bench",
                       binary=True)   # what the harness uses; see docs/05 §3
cur = conn.cursor()
cur.execute("SET SESSION transaction_isolation = 'READ-COMMITTED'")
cur.execute("USE demo")
vec = numpy.array([0.1, 0.2, 0.3], dtype="<f4").tobytes()
cur.execute("INSERT INTO items (id, tag, v) VALUES (?, ?, ?)", (2, 0, vec))
conn.commit()
```

### Search

```sql
SET vidx_hnsw_ef_search = 100;

SELECT id, VEC_TOTEXT(v) AS vector,
       VEC_DISTANCE_COSINE(v, VEC_FROMTEXT('[0.1,0.2,0.3]')) AS distance
FROM items
ORDER BY distance
LIMIT 10;
```

### Confirm the index is actually used — this matters most here

AliSQL costs the vector index against a full table scan and **will choose the
scan when the `LIMIT` is large relative to the table**. Its own test suite
(`mysql-test/suite/rds/t/vidx_dml.test`) shows a 100-row table using the index
at `LIMIT 16` and falling back to a full scan at `LIMIT 17`.

```sql
EXPLAIN SELECT id FROM items
ORDER BY VEC_DISTANCE_COSINE(v, VEC_FROMTEXT('[0.1,0.2,0.3]')) LIMIT 10;
```

If the plan does not name `vi`, force it:

```sql
SELECT id FROM items FORCE INDEX (vi)
ORDER BY VEC_DISTANCE_COSINE(v, VEC_FROMTEXT('[0.1,0.2,0.3]')) LIMIT 10;
```

### Knobs

| Variable | Scope | Default | Meaning |
| --- | --- | --- | --- |
| `vidx_disabled` | global | `ON` | Must be `OFF` to use vectors at all. |
| `vidx_hnsw_ef_search` | session | 20 | Search width. Range 1–10000. |
| `vidx_hnsw_default_m` | session | 6 | Default graph degree. Range 3–200. |
| `vidx_default_distance` | session | `EUCLIDEAN` | Default metric. |
| `vidx_hnsw_cache_size` | global | 16 MiB | Graph cache ceiling, in bytes. |

Note the default `vidx_hnsw_cache_size` of **16 MiB** — far too small for any
real corpus. Raise it before drawing conclusions about AliSQL's speed.

AliSQL does **not** expose `ef_construction`.

---

## 4. PostgreSQL + pgvector

### Start it

```bash
docker run -d --name pg \
  -p 5432:5432 \
  -e POSTGRES_HOST_AUTH_METHOD=trust \
  vector-bench/pgvector-runtime:v0.8.6

until docker exec pg pg_isready -U postgres -d ann >/dev/null 2>&1; do sleep 1; done
docker exec -it pg psql -U postgres -d ann
```

### Create a table and index

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items (
  id        int PRIMARY KEY,
  tag       int NOT NULL,
  embedding vector(3)
);

-- Keep vectors inline; TOAST would add a detoast to every comparison.
ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN;

INSERT INTO items VALUES (1, 0, '[0.1,0.2,0.3]'), (2, 1, '[0.2,0.1,0.4]');

-- Build AFTER loading: this is pgvector's idiomatic and much faster path.
CREATE INDEX items_embedding_idx ON items
USING hnsw (embedding vector_cosine_ops) WITH (m = 6, ef_construction = 200);

ANALYZE items;
```

Use `vector_l2_ops` with the `<->` operator for Euclidean distance.

### Search

```sql
SET hnsw.ef_search = 100;
SET jit = off;   -- JIT costs more than it saves at these row counts

SELECT id, embedding <=> '[0.1,0.2,0.3]' AS distance
FROM items
ORDER BY embedding <=> '[0.1,0.2,0.3]'
LIMIT 10;
```

The operator in `ORDER BY` must match the operator class of the index
(`<=>` ↔ `vector_cosine_ops`, `<->` ↔ `vector_l2_ops`), or the planner will not
use the index.

### Confirm the index is actually used

```sql
EXPLAIN SELECT id FROM items ORDER BY embedding <=> '[0.1,0.2,0.3]' LIMIT 10;
```

Look for `Index Scan using items_embedding_idx`. A `Seq Scan` means brute force.

### Filtered search

pgvector 0.8 can iterate the graph when a filter rejects candidates. Without
it, a selective filter can exhaust `ef_search` before finding k qualifying rows
and simply return fewer results:

```sql
SET hnsw.iterative_scan = relaxed_order;   -- or strict_order, or off
SELECT id FROM items WHERE tag < 10
ORDER BY embedding <=> '[0.1,0.2,0.3]' LIMIT 10;
```

### Knobs

| Setting | Scope | Meaning |
| --- | --- | --- |
| `hnsw.ef_search` | session | Search width. |
| `hnsw.iterative_scan` | session | `off` / `relaxed_order` / `strict_order`. |
| `m`, `ef_construction` | index | Build parameters, set in `WITH (…)`. |
| `shared_buffers` | server | Where graph pages live; pgvector has no separate cache. |
| `maintenance_work_mem` | server | Governs whether the index build stays in memory. |

pgvector is the **only** one of the three that exposes `ef_construction`.

---

## 5. Measuring recall by hand

If you want a recall number without the harness, the method is:

1. Take a query vector `q` and ask the engine for its top-k with the index on.
2. Ask the same question with the index bypassed, to get the exact answer:
   - MariaDB / AliSQL: `SELECT id FROM items IGNORE INDEX (vi) ORDER BY … LIMIT k`
   - PostgreSQL: `SET enable_indexscan = off; SET enable_bitmapscan = off;`
3. Recall for that query is `|approx ∩ exact| / k`. Average over many queries.

Two cautions:

- **Ties.** If several rows sit at exactly the k-th distance, an id-intersection
  under-reports recall. The framework instead counts results within the true
  k-th distance plus a small epsilon, which is what ann-benchmarks does.
- **One query is not a measurement.** Recall varies substantially per query;
  a few hundred queries is the minimum for a stable figure.

---

## 6. Cleaning up

```bash
docker rm -f mdb ali pg
```

Data lives inside the containers unless you mounted a volume, so removing the
container discards it.

---

## 7. When to use the framework instead

Doing the above by hand is fine for exploring behaviour and for verifying a
specific claim. It is not sufficient for producing comparable numbers, because
the things that make results comparable — identical CPU pinning, identical
memory budgets, identical `-march`, identical client library, recomputed
filtered ground truth, plan verification on every configuration, and a manifest
recording all of it — are exactly what the framework exists to enforce.

See [02-running-with-framework.md](02-running-with-framework.md).
