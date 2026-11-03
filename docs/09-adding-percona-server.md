# Adding Percona Server as a seventh engine

Percona Server's `9.7` series carries a native InnoDB HNSW vector index.
This document is the investigation that has to happen before any of it is wired
in: what the dialect actually is, what changes in which file, and — kept
separate, in [§7](#7-what-only-a-live-server-can-settle) — what no amount of
reading can answer.

**Nothing in this repository drives Percona yet.** No driver is registered, no
`config/engines/percona.yml` exists, and no profile names it. The two long-lead
artefacts are written and unbuilt: `docker/percona/` and
`overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/`. They exist now
because the image compiles a server from source, and that is the item with the
longest lead time in the whole addition.

## 0. Provenance of every claim here

| Kind of claim | How it is sourced | Example |
|---|---|---|
| Percona behaviour | a named symbol or MTR test on the pinned SHA, in prose | *field.cc*'s *Field_vector::store* |
| This repo's behaviour | a symbol in this checkout, value quoted | `vb_mysql.py`'s `Dialect.create_table` |
| A number | its inputs listed, each marked measured or supplied | [§3](#3-innodb_hnsw_max_memory-is-not-a-cache) |

MTR result files are preferred over the vendor design document
*percona-hnsw.md* wherever both say the same thing: a recorded result is the
server's own output, a design document is an intention. Where only the design
document says something, it is cited as such and marked.

### How this document cites code, and why it cites no line numbers

**Every citation here names a symbol or a file — a function, a class, a
constant, a config key, an MTR test, a path — and never a line number.**
`to_binary_f32`, `_driver_classes`, `harness/workloads/build.py`: never
*some/file.py:123*.

This document used to cite `path`-colon-line, and
`tests/test_e1_percona_design.py` re-opened every cited line in this
repository and compared it against what the document claimed was there. **That
mechanism worked, and it is how the rot below was found at all.** But the rot
kept happening: line citations in this document and in `docs/10-rollout.md`
went stale twice in two days and put `master` red both times, repaired in pull
requests #41 and #43. Neither failure was somebody editing a document
carelessly — in both cases a later round of edits *inside the same change*
moved lines below the point where the citations had been derived, and the
guard, correctly, fired on the document rather than on the code.

A line number is invalidated by **any** edit above it, including an edit that
changes nothing the citation is about. A symbol name is invalidated **only by a
rename**, which is a deliberate act a reader can follow.
`docs/11-methodology-roadmap.md` §0 is where this policy is set out and
`CLAUDE.md` says to copy it; this document now does.

Two consequences that follow from the policy rather than from taste, and they
are the reason the conversion is not merely cosmetic here:

* **A citation here is coarser.** `_detect_march` names a function, not the
  three lines of its path tuple that §4.8 quotes. Where the exact passage
  matters, this document quotes it and says which symbol it sits in, so the
  quotation is checkable by search rather than by line arithmetic.
* **Vendor material is named in prose, never in code font.** *ha_innodb.cc*,
  the *percona_vector* suite's *index_build.test*, *percona-hnsw.md*: a
  backticked path in this document means a path in **this** repository, and a
  guard enforces that. The pinned Percona tree is not in this checkout —
  `sources/` is generated and gitignored, and it holds no Percona tree — so a
  backticked path into it would be a citation nobody reviewing this document
  could check. **That was already true of the line numbers**, and it is the
  half of the rot nothing reported: the guard's deny-list skipped every vendor
  path by name, so the vendor line references in this document's previous
  revision — 70 vendor path citations plus the 93 bare continuations the old
  guard's own note attributed to Percona sources, **164** line references
  between them — were checked by nothing at all and
  would have rotted silently at the re-pin, in a document whose provenance
  table above commits it to having been read against a specific tree. A vendor
  symbol citation still cannot be verified by any test in this repository.
  What changes is that it now **names** something a reader holding the tree can
  find, instead of a line number that is wrong.

One piece of typography that is not typography: the illustration of the banned
form at the top of this subsection is in *italics* rather than code font, for the reason
`docs/11` §0 gives — the guard bans that form as a pattern over code-font
spans, and an allowlist carving out one citation would make this the one
document permitted to make the citation it forbids.

### The pin

| | |
|---|---|
| Repository | `https://github.com/percona/percona-server` |
| Base branch | `9.7` — the vector MVP was merged out of `vector-mvp` into this series by PR #6278 |
| **Pinned SHA** | **`56124b03fdc5954e2401ab4d2c7bf7cb3a767158`** |
| Pin **author** date | 2026-10-14T22:27:46Z |
| Pin **committer** date | 2026-10-26T12:41:43Z |
| Pin subject | `PS-11751 : [9.7] VIRTUAL generated column returns a stale, identical value for every row when read through an HNSW vector index (ANN query)` |
| Server version | MySQL 9.7.2 LTS base (`MYSQL_VERSION`: `MAJOR=9 MINOR=7 PATCH=2 EXTRA=-2 MATURITY="LTS"`, i.e. `9.7.2-2`) |
| MTR suite | `mysql-test/suite/percona_vector/` — **97** `.test` files and 97 `.result` files, plus two in `mysql-test/suite/sys_vars/` |

That SHA is PR #6278's `merge_commit_sha` (`gh api 'repos/percona/percona-server/pulls/6278'`).
The merge was a REBASE, so that field names the last of the 30 rebased commits and not a squash.

**Dates on this pin need their basis stated, and the table above gives both.**
The series was rebased onto `9.7` at merge, so the pin's **author** date sits
eleven days under its **committer** date and its parent carries the same
committer second — which is the fingerprint of a rebase, not of a backdate. An
earlier revision of this table printed one date without saying which, and then
built the warning on it; the warning's conclusion survives either way, for a
simpler reason than the one it gave: **a date never tells you whether you have
the tip, so compare SHAs.** Every date in the table below is an author date,
because that is what `git log` shows by default.

### What the pre-merge `vector-mvp` commit log said about stability

A snapshot of the **branch as it stood before the merge**, kept because its
reading still holds. Of the twelve newest commits *on that branch*, **four**
were crash or assertion fixes and two more were the features that brought the
vector work to the state it was merged in. All six carry author dates between
2026-10-02 and 2026-10-13, every one of them earlier than the pin's own:

| SHA | Date | What it fixed |
|---|---|---|
| `1d62ab22bc75` | 2026-10-13 | PS-11612: store resolved HNSW parameters in the DD |
| `5c30ac32877a` | 2026-10-12 | PS-11673: debug assertion in `Dictionary_client::acquire` on cross-schema `RENAME TABLE` after restart |
| `a86503214416` | 2026-10-12 | PS-11615: assertion `!other->is_vector()` in *vec_index_of* when a vector search runs during an in-place index replace |
| `af72e2c0fa40` | 2026-10-12 | PS-11591: assertion `inserted.second` in `HNSW::insert()` — crash on INSERT after an **unclean restart** |
| `eebc427aab7b` | 2026-10-02 | PS-11623: assertion in *ddl0loader.cc* when an index build aborts on `innodb_hnsw_max_memory` exhaustion |
| `d436bb1c77d7` | 2026-10-09 | PS-11618: COSINE distance support for vector indexes |

Read as a group, those were the failures of a branch then under daily
development, and they clustered in three places: restart recovery, concurrent
DDL against a live search, and the memory ceiling. Two of those three are
things a four-day benchmark run does — it restarts servers between phases and
it loads until something stops it — which is the argument for the smoke run in
[§7](#7-what-only-a-live-server-can-settle), not against the engine.

Three qualifications on reading that table as a risk assessment, all of which
cut against over-reading it:

- Each row is a **fix** the pin contains, under a *different* SHA because the
  series was rebased. It says where the bugs were, not where they are.
- Two are explicitly debug-build assertions (`a86503214416`, `5c30ac32877a` say
  "debug"). `config/engines/percona.yml` will build `RelWithDebInfo` with
  *NDEBUG* defined, as every other engine here does, so those two cannot fire in a
  benchmark build. They still matter, because an assertion marks a state the
  author did not expect, and in a release build that state is reached silently.
- Nothing in a commit log shows what is still open. This is a survey of fixed
  things and must not be reported as a clean bill of health.

## 1. The question that had to be answered first: is there an `ef_search`?

**Yes.** `innodb_hnsw_ef_search` is a real, documented, tested system variable
with both SESSION and GLOBAL scope, and it is read per query.

The primary source is the declaration itself, the *hnsw_ef_search*
*MYSQL_THDVAR_ULONG* block in *ha_innodb.cc*, which carries the scope, the
default and both bounds in five lines:

```c
static MYSQL_THDVAR_ULONG(
    hnsw_ef_search, PLUGIN_VAR_RQCMDARG,
    "Minimum candidate-list width for HNSW vector index searches (ANN"
    " recall/latency knob; the effective width is max(ef, LIMIT))",
    nullptr, nullptr, 40, 1, 100000, 0);
```

*MYSQL_THDVAR_ULONG* is the session-variable macro, and the trailing arguments
are default `40`, minimum `1`, maximum `100000`. It is registered in
*ha_innodb.cc*'s *innobase_system_variables* array, as
*MYSQL_SYSVAR(hnsw_ef_search)*. The engine reads it through *THDVAR* on the
query path, in the *vec_ann_open* call the scan's open makes:

```c
  const dberr_t err = vec_ann_open(
      vindex, reinterpret_cast<const float *>(m_vec_query.data()),
      std::max<size_t>(limit, 1), THDVAR(m_user_thd, hnsw_ef_search),
      m_user_thd, &m_vec_search);
```

*THDVAR* is a per-session read taken inside the scan's open, which is what
"per query" means mechanically rather than by assertion. Note also that the
help text states the `max(ef, LIMIT)` clamp as part of the variable's own
contract — see the riders below.

The behaviour at the edges is in the server's own recorded output, the
*sys_vars* suite's *innodb_hnsw_ef_search_basic.result*:

```
SET @start_global_value = @@global.innodb_hnsw_ef_search;
# Default: 40, in both scopes.
SELECT @@global.innodb_hnsw_ef_search, @@session.innodb_hnsw_ef_search;
@@global.innodb_hnsw_ef_search  @@session.innodb_hnsw_ef_search
40  40
...
# A session value does not change the global one.
SET SESSION innodb_hnsw_ef_search = 200;
SELECT @@global.innodb_hnsw_ef_search, @@session.innodb_hnsw_ef_search;
@@global.innodb_hnsw_ef_search  @@session.innodb_hnsw_ef_search
40  200
...
# The bounds, and values outside them clamped with a warning.
SET SESSION innodb_hnsw_ef_search = 1;
...
SET SESSION innodb_hnsw_ef_search = 100000;
...
SET SESSION innodb_hnsw_ef_search = 0;
Warnings:
Warning 1292  Truncated incorrect innodb_hnsw_ef_search value: '0'
```

(The gutter numbers this excerpt used to carry are gone, and the `...` lines
are what is left of them: they mark where the recording was elided, which is
the only thing the numbers were doing that the reader needed. **A gutter number
is a line citation that the ratchet cannot see**, because the ratchet is a
pattern over code-font spans and a fenced block is not one, so it has a check
of its own in `tests/test_e1_percona_design.py`, which walks every fence in
this document and refuses a line that opens with a bare number. That check
exists because leaving the class to the author failed on the first attempt —
eight of this document's eleven gutter-bearing blocks were cleared, a commit
message said all of them were, and four vendor line numbers survived in the
excerpt below this paragraph, which asserts they are gone. The hand scan missed
them because it anchored the fence at the start of a line and that block is
indented inside a bullet. Three lines here still open with a number inside a
fence and are **not** gutters — two recorded `SELECT` result rows and one
EXPLAIN `id` column — and the guard names each one with what its number is,
asserting both that it is exempt and that it is still present.

The check is pointed at by file and not by the names of its own tests, and that
is the self-poisoning rule rather than vagueness: the three document guards
exclude all three of themselves from the symbol index they resolve citations
against, because a guard whose index contains the guard can certify a name that
exists nowhere else. So a symbol living only in a test file is deliberately
uncitable from here, and the file is the finest pointer this document is
allowed.)

So: **default 40, range 1..100000, clamped with a warning outside it, integers
only** (`1.1`, `1e1` and `'AUTO'` are all *ER_WRONG_TYPE_FOR_VAR*, in the same
file's type-rejection block).

It is genuinely **per query and not per index**. Two independent sources:

- the *percona_vector* suite's *search.test*, inside a test of the search path
  itself, in the block it heads `# ef_search is a session knob.`:
  ```
  --echo # ef_search is a session knob.
  SET SESSION innodb_hnsw_ef_search = 200;
  SELECT id FROM t1 ORDER BY DISTANCE(v, STRING_TO_VECTOR('[0,0]'), 'EUCLIDEAN') LIMIT 3;
  SET SESSION innodb_hnsw_ef_search = default;
  ```
- *percona-hnsw.md*, in its *ef_search* paragraph: "`ef_search` comes from the session variable
  `innodb_hnsw_ef_search` (default 40); the effective width is
  `max(ef_search, batch)`. It is per-query rather than per-index because the
  same index serves a dashboard wanting speed and a batch job wanting recall."

**Consequences for PR 2's calibration phase.** It is a normal engine: it has a
recall/QPS frontier, the calibration sweep has something to do, and
`set_ef_search` is one `SET SESSION`. Two details change what the sweep means:

- **The clamp is `max(ef, LIMIT)`**, stated in the variable's own help text,
  quoted above, and visible in the *vec_ann_open* call also quoted above, where
  the *hnsw_ef_search* read is passed beside `std::max<size_t>(limit, 1)`;
  *percona-hnsw.md* calls the same thing `max(ef_search, batch)`. So the effective width is never
  below `LIMIT k`. At `k = 10` every grid
  value here is above 10, so the grid is taken verbatim and
  `effective_ef()`'s default pass-through is correct. **But** the harness
  already has a hook for exactly this shape — `EngineDriver.effective_ef`
  in `harness/drivers/base.py`, which exists because MongoDB clamps
  `numCandidates` up to `limit`. If a profile ever sweeps `ef_search` below `k`,
  Percona clamps the same way MongoDB does and the driver must say so. The
  follow-on task should implement `effective_ef` as `max(ef, k)` and not leave
  the inherited identity, because the inherited one would be wrong at the one
  grid point where it mattered and silently right everywhere else.
- `ef_construction` is *also* exposed, as an index build option — see the table
  in §2. So Percona has **both** a query-time and a build-time effort knob,
  which no other MySQL-family engine here has. That is a finding for the report:
  `CLAUDE.md`'s "Structural asymmetries no config removes" bullet currently
  reads "`ef_construction` is exposed only by pgvector and Valkey", and adding
  Percona makes that sentence false.

## 2. The dialect surface

Throughout: **PS** = Percona Server at the pinned SHA; **MDB** = MariaDB MHNSW
as this repo already drives it.

### 2.1 Column type and value binding

| | Percona Server | MariaDB MHNSW (for comparison) |
|---|---|---|
| Column | `v VECTOR(n) NOT NULL` | `v VECTOR(n) NOT NULL` |
| Max `n` | 16383 — *vector_constants.h*'s *max_dimensions*, `constexpr unsigned int max_dimensions = 16383` | n/a here |
| Text constructor | `STRING_TO_VECTOR('[1,2]')`, alias *TO_VECTOR* | *VEC_FromText* |
| Read-back | `VECTOR_TO_STRING(v)`; a bare `SELECT v` returns raw bytes | *VEC_ToText* |
| Width helper | `VECTOR_DIM(v)` | n/a |
| **Raw float32 bytes accepted?** | **Yes, if the parameter carries the binary charset** | Yes |

*STRING_TO_VECTOR* and *TO_VECTOR* are the *same function* —
*item_create.cc*'s function registry registers both names against
*Item_func_to_vector*. *VECTOR_TO_STRING* and *VECTOR_DIM* are the next two
entries in the same registry.

The binding question is the one that decides whether this repo's existing batched
loader works unchanged, and it is settled by *field.cc*'s
*Field_vector::store*, whose signature is
`store(const char *from, size_t length, const CHARSET_INFO *cs)`:

```c
  uint32 dimensions = get_dimensions(length, Field_vector::precision);
  if (dimensions == UINT32_MAX || cs != &my_charset_bin) {
    my_error(ER_DATA_INCOMPATIBLE_WITH_VECTOR, MYF(0), "string", length);
    return TYPE_ERR_BAD_VALUE;
  }
...
  /* Check for NAN or INF value in the vector. */
...
  return Field_blob::store(from, length, cs);
```

with *Field_vector::precision* declared `sizeof(float)` in *field.h*, and
*get_dimensions* in *vector_constants.h* returning *UINT32_MAX* unless the
length is a non-zero multiple of 4. On a little-endian host the bytes are then
stored verbatim; the *WORDS_BIGENDIAN* branch of the same *store* byte-swaps,
so the wire format is **little-endian float32** either way.

So **`harness/drivers/mysql_family.py`'s `to_binary_f32()` produces exactly
the right bytes**, and `vb_mysql.py`'s copy of `to_binary_f32` does too. Three
caveats, in descending order of how likely they are to bite:

1. **`cs != &my_charset_bin` is a hard refusal, not a conversion.** The
   parameter must reach the server as binary, not as utf8mb4. The repo passes
   Python `bytes`, which MariaDB Connector/C sends as *MYSQL_TYPE_BLOB*, and
   that is what makes this work today for AliSQL's *Field_vector*. It is
   *expected* to work identically here, and it is on the §7 list because
   "expected" is not "measured" and the failure mode is every INSERT erroring.
2. NaN and Inf are refused per element, by the check *store* makes between
   the two quoted hunks. The datasets are real
   embeddings and contain neither, but a synthetic corpus could.
3. A length that is not `4 × n` for the declared `n` is refused. The repo always
   sends `4 × dim`.

**Percona does not need the text path, so it must not use it.** That is the same
argument the module docstring of
`overlay/ann-benchmarks/ann_benchmarks/algorithms/alisql/module.py` makes for
AliSQL:
routing one engine through *STRING_TO_VECTOR* while the others take binary
would charge Percona for text parsing that belongs to the client, and would show
up as a Percona ingest penalty. Everything in this repo binds binary; Percona
keeps doing so.

### 2.2 Index DDL

| | Percona Server | MariaDB MHNSW |
|---|---|---|
| Inline in `CREATE TABLE` | **Yes** | Yes |
| By `ALTER TABLE` | **Yes** — `ALTER TABLE t ADD VECTOR KEY vk (v) TYPE hnsw (…)` | n/a (inline only in this repo's usage) |
| By `CREATE INDEX` | **Yes** — `CREATE VECTOR INDEX vk ON t (v) TYPE hnsw (…)` | no |
| Spelling | `VECTOR KEY` / `VECTOR INDEX`, then `TYPE hnsw` or `USING hnsw`, then `(opts)` | `VECTOR INDEX (v) M=n DISTANCE=m` |
| Canonical form | `SHOW CREATE TABLE` emits `USING \`hnsw\`` | — |
| Indexes per table | **exactly one** | more than one allowed |
| **Primary key** | **`BIGINT UNSIGNED`, single column** — explicit, or one MySQL promotes from a `UNIQUE NOT NULL` key | any |

All four creation paths are exercised in the *percona_vector* suite's
*cosine.test*, which uses the inline form, `CREATE VECTOR INDEX`,
`ALTER TABLE … ADD VECTOR KEY` and `ADD VECTOR INDEX` in that order.
`USING hnsw` is accepted as a synonym for `TYPE hnsw`, recorded in the same
suite's *index_syntax.test*. A second vector index on one table is
*ER_NOT_SUPPORTED_YET* there too, and so is a second one on the same column.

**The primary-key requirement is the single biggest difference, because it
changes the schema rather than the index clause.** *index_syntax.result*
records four refusals with one message:

```
CREATE TABLE t1 ( id BIGINT PRIMARY KEY, …              -> refused
CREATE TABLE t1 ( id INT UNSIGNED PRIMARY KEY, …        -> refused
CREATE TABLE t1 ( id1 …, id2 …, PRIMARY KEY (id1,id2)…  -> refused
ALTER TABLE t1 CHANGE id id INT UNSIGNED;               -> refused

ERROR 42000: This version of MySQL doesn't yet support 'vector indexes on
tables lacking a BIGINT UNSIGNED single column primary key'
```

and *pk_combinations.result* records the same message for a table with no
primary key at all, and calls it a "current temporary restriction".

**Both of this repo's table definitions declared `id INT PRIMARY KEY`** —
`vb_mysql.py`'s `Dialect.create_table`, which still does, and the ops driver,
which no longer does. Percona refuses that column outright, so the Percona DDL
must say `id BIGINT UNSIGNED PRIMARY KEY`, and §4.1's answer was to make the
type a `Dialect` field: `MySQLFamilyDriver._create_table_ddl` in
`harness/drivers/mysql_family.py` now emits
`id {self.dialect.pk_type} PRIMARY KEY` with `pk_type` defaulting to `"INT"`,
so MariaDB and AliSQL emit exactly what they emitted before. The ann overlay
needs no such field because `percona/module.py` overrides `create_table`
outright.

That is a real, if small, asymmetry the report should state rather than bury:
Percona's clustered index and its aux table's `base_pk` are both 4 bytes per row
wider than MariaDB's and AliSQL's, by the server's requirement and not by our
choice. At 1.18M rows that is about 4.7 MB in the base table — far below the
resolution of any figure here, but `index_bytes` and `table_bytes` are reported
and someone will notice.

### 2.3 Index options

| Option | Range | Default | Evidence |
|---|---|---|---|
| `M` | 2..200 | **25** | *vec0vec.h*'s option name and bound constants |
| `ef_construction` | 1..4096 | 200 | the same constants in *vec0vec.h* |
| `metric` | `euclidean` \| `cosine` | `euclidean` | *vec0vec.h*, with the refusals recorded in *index_syntax.result* |

"`M`, `metric` and `ef_construction` are settable, and they are all the
parameters there are" — *percona-hnsw.md*, in its index-options section.

Syntax rules, each with a recorded refusal:

- An option value is **a bare identifier or an integer, never a quoted string**.
  `metric = cosine` is right; `metric = 'cosine'` is a parse error, per
  *percona-hnsw.md*, and so are `M = 6.0` and `M = "6"`, both recorded in
  *index_syntax.test*.
- An unknown option name is *ER_ILLEGAL_INDEX_CONSTRUCTION_PARAMETER* and a
  bad value is *ER_ILLEGAL_INDEX_CONSTRUCTION_PARAMETER_VALUE*, both recorded
  in *index_syntax.test*. The second prints only the value and not which
  parameter it belonged to — *dd_params_resolved.result* has
  `ERROR HY000: Illegal index construction parameter value: 0.` That is why the
  overlay module range-checks `M` and
  `efConstruction` itself before the server sees them.
- **Defaults are resolved at create time and stored.** An index created with no
  option list comes back from `SHOW CREATE TABLE` with the full canonical set:
  *dd_params_resolved.result* creates `VECTOR KEY vk (v) TYPE hnsw` and
  gets back `` VECTOR KEY `vk` (`v`) USING `hnsw` (`M`=25,
  `ef_construction`=200, `metric`=`euclidean`) ``. So the manifest can read the
  parameters a graph was actually built with out of `SHOW CREATE TABLE`, which
  is better than trusting the DDL we sent.

### 2.4 The distance expression

| | Percona Server | MariaDB MHNSW |
|---|---|---|
| Form | `DISTANCE(v, q, 'METRIC')` — **arity exactly 3** | `vec_distance_<metric>(v, q)` — one function per metric |
| L2 | `DISTANCE(v, q, 'EUCLIDEAN')` | `vec_distance_euclidean(v, q)` |
| L2 squared | `DISTANCE(v, q, 'EUCLIDEAN_SQUARED')` — also served by an L2 index | — |
| Cosine | `DISTANCE(v, q, 'COSINE')` | `vec_distance_cosine(v, q)` |
| Inner product | `DISTANCE(v, q, 'DOT')` — **exact scan only, cannot be indexed** | — |
| Manhattan | `DISTANCE(v, q, 'MANHATTAN')` — exact scan only | — |

*item_create.cc*'s registry entry for it is
`{"DISTANCE", SQL_FN(Item_func_vector_distance, 3)}` — the metric argument is
mandatory. The five names are a table in *vector_constants.h*, resolved
case-insensitively through *my_strnncoll* on *my_charset_latin1*, which is why
the suite's *search.test* uses `'EUCLIDEAN'` and its *index_build.test* uses
`'euclidean'` interchangeably.

**A correction to the brief this task was given.** The brief said "L2 and
inner-product kernels with AVX-512 paths", which is true of the *kernels* and
not of the *index*. An HNSW index can only be built on `euclidean` or `cosine`:

```
index_syntax.result
  CREATE TABLE t ( … VECTOR KEY ( v1 ) TYPE hnsw ( metric = dot ) );
  ERROR 42000: This version of MySQL doesn't yet support 'HNSW indexes on
  anything but the EUCLIDEAN or COSINE metric'
```

with the same message recorded there for `euclidean_squared` and `manhattan`.
`DISTANCE(…, 'DOT')` still works, as an exact full scan.

This costs this benchmark nothing — the harness needs `angular` → `cosine` and
`euclidean` → `euclidean`, both supported — but **the published report cannot
show a Percona inner-product frontier**, and if an inner-product dataset is ever
added, Percona has no index for it while pgvector and Valkey do.

Two more rules the index enforces, both relevant to `query_filtered`:

- **A cosine index serves only `'COSINE'`; an L2 index serves only
  `'EUCLIDEAN'` and `'EUCLIDEAN_SQUARED'` — never the cross product.**
  *cosine.test* asserts both directions after a restart, and
  *percona-hnsw.md* explains it: the optimizer reads
  *KEY::vector_index_metric* from the stored option. So a dialect that built the
  index with one metric and queried with another gets a silent full scan, not an
  error.
- `ORDER BY` must be **ascending** and there must be a `LIMIT`. Both missing
  cases, plus `DESC`, plus `DISTANCE()` only in `WHERE`, plus both arguments
  non-constant, are listed as "shapes that must NOT use the index" in
  *search.test*.

### 2.5 How the index is maintained: both ways, which is new here

| | Percona Server | MariaDB MHNSW | pgvector |
|---|---|---|---|
| On `INSERT` | **incremental** | incremental | no |
| Separable bulk build | **yes** | no | yes |

This is the second place Percona breaks a sentence in `CLAUDE.md`. It maintains
the graph on every `INSERT` — the *percona_vector* suite's *insert.test* loads rows
and its *insert_graph.test* checks the graph grew — *and* it has a real bulk
path: `ALTER TABLE … ADD VECTOR KEY` builds the graph from one clustered scan.
*index_build.test* names it, in its first `--echo` block:

```
--echo # First ADD, explicit INPLACE: rebuilds the table with the column and
--echo # builds a node per row. An in-place rebuild that wrote the column and
--echo # no graph is the case that used to give nodes=0.
ALTER TABLE t1 ADD VECTOR KEY k1 (v) TYPE hnsw (M = 4), ALGORITHM=INPLACE;
```

and *percona-hnsw.md* contrasts the two build algorithms: "`INPLACE` builds
the graph in memory and bulk-loads the aux (§15), so a COPY-built aux is correct
but slower to build and larger."

So Percona is the **second** engine in this benchmark that can run both
`build_mode`s, and the **first in the MySQL family**. An earlier revision of
this document called it the first outright and the only one where "incremental
versus bulk" is a within-engine comparison; both were wrong. **pgvector already
does it**: `pgvector_build_modes` appears in eight profiles including `smoke`,
`main`, `full` and `quick`, and `harness/workloads/build.py`'s module
docstring was written for exactly that comparison —

> …so the report can say "pgvector's bulk build is N× cheaper than its own
> incremental build" and "pgvector's incremental build costs X versus MHNSW's
> Y" — two honest claims instead of one misleading one.

What Percona adds is still worth having, but it is narrower than "a new
capability": it **decouples build mode from MySQL-versus-Postgres**. Today the
only within-engine build-mode curve belongs to the one engine that is also the
only Postgres, so "bulk is cheaper than incremental" and "Postgres differs from
MySQL here" cannot be separated. A second such curve, on a MySQL-family server,
separates them. That is a
genuinely valuable result for the report and it is worth a profile.

**There is a defect in `harness/workloads/build.py` directly in the way**, in
its `run`, which decides the build mode like this:

```python
    incremental = driver.capabilities().get("incremental_index", True)
    if index.build_mode == "incremental":
        incremental = True
```

`build_mode == "incremental"` forces the flag true; **`build_mode == "post"` has
no counterpart that forces it false.** The flag then decides everything
downstream in the same function: whether `driver.create_index(index)` is called
at all, and what `build_wall_s` means —
`effective_build = build_seconds if not incremental else load.wall_seconds`.

For every engine that exists today this is harmless, because `incremental_index`
is a fixed property of the engine and the two never disagree. For Percona they
do: a class-level `incremental_index = True` would make a `build_mode: post`
Percona run load an already-indexed table, never call `create_index`, and record
`build_wall_s` as the load time — a number labelled "bulk build after load" in
neither the notes nor the `extra` dict, since both branch off the same flag. The
record would read as incremental and nothing would say the profile asked for
something else.

The fix belongs to whoever owns the Percona driver, and the smaller of the two
options is the right one: **make the driver's `incremental_index` and
`capabilities()` depend on `IndexSpec.build_mode`**, so the existing
`build.py` logic reaches the right branch without `build.py` changing. The
driver already receives the spec in `create_schema`. The alternative — adding
`elif index.build_mode == "post": incremental = False` to `build.py` — is one
line and arguably more honest, but it changes a file that governs every engine's
build accounting, and it is not this task's to change.

### 2.6 Server variables

| Variable | Scope | Default | Harness must | Evidence |
|---|---|---|---|---|
| `innodb_hnsw_ef_search` | SESSION + GLOBAL | 40 | **set** per session (calibration) | *sys_vars*' *innodb_hnsw_ef_search_basic.result* |
| `innodb_hnsw_max_memory` | **GLOBAL only** | 1073741824 (1 GiB), `0` = unlimited | **set** from the resource pass — see §3 | *sys_vars*' *innodb_hnsw_max_memory_basic.result*, whose default, scope and bound blocks all record it |
| `Innodb_hnsw_memory_used` | GLOBAL status | — | **record** | *percona_vector*'s *hnsw_memory_used.result* |
| *optimizer_switch* → *hypergraph_optimizer* | SESSION + GLOBAL | **unreachable in this build** | record only | three gates in *sys_vars.cc*, below |

**That table is complete, and it is now established from the registration array
rather than from filenames.** InnoDB exposes a variable by listing it in
*innobase_system_variables*, so that array is the authoritative enumeration.
At the pin it contains exactly two HNSW entries —

```
MYSQL_SYSVAR(hnsw_ef_search),
MYSQL_SYSVAR(hnsw_max_memory),
```

declared earlier in the same file, the first as the *MYSQL_THDVAR_ULONG* block
§1 quotes — and the status-variable array in the same file contains exactly
one:

```
{"hnsw_memory_used", (char *)&export_vars.innodb_hnsw_memory_used,
```

A grep of the whole file for `hnsw` returns no other variable of either kind.
**There is no third.**

An earlier revision of this section reached the same conclusion by a method that
could not support it: a recursive tree listing matching *innodb_hnsw*, which
returned the four test and result paths of the two *sys_vars* tests. That was
wrong twice over — it searched **path names, not file contents**, so a variable
with no *sys_vars* test would not have appeared; and the GitHub recursive tree
API returns `truncated: true` at this SHA, so the listing was not even complete
on its own terms. The same truncation is why that revision said the
*percona_vector* suite had 65 tests when it has 91. A truncated listing is not
an enumeration, and it does not announce itself in the part you read.

`Innodb_hnsw_memory_used` is a status variable worth recording for its own sake —
it is the only engine here that reports its graph's live memory, and
*hnsw_memory_used.result* shows it reading `0` after a restart (the graph
is loaded lazily) and rising on the first search.

The **hypergraph optimizer** row is the one trap in this table that is a
behaviour and not a knob. *percona-hnsw.md*, on the read path: "The read path is the
old optimizer's, with no cost model. A session using the hypergraph optimizer
plans scan-plus-sort, with correct results and no index." A session with
*hypergraph_optimizer=ON* therefore gets **exact results at full-scan speed**,
which is this repo's canonical silent failure — "indistinguishable in output from
'very accurate but slow'". `explain_uses_vector_index()` catches it, which is
why that check must stay on for Percona even though, unlike AliSQL, Percona
never declines the index for *cost* reasons (vector index access is priced at
`prohibitive_cost = 1e100` specifically "so that it always loses against any
other access method, since a vector index is only ever activated for ORDER BY
... LIMIT" — *vector_constants.h*'s *prohibitive_cost*).

**In this build the hypergraph risk cannot occur, and that is now established
rather than assumed.** Three gates in *sys_vars.cc*, all keyed on CMake
symbols `docker/percona/Dockerfile` does not define:

- the default *optimizer_switch* string includes the hypergraph bit
  only `#if defined(ENABLE_HYPERGRAPH_OPTIMIZER)`, so it is **off by default**.
- *update_optimizer_switch* masks the bit out of the global
  value `#ifndef WITH_HYPERGRAPH_OPTIMIZER`, so it cannot survive startup.
- *check_optimizer_switch* rejects an attempt to turn it on
  with *ER_HYPERGRAPH_NOT_SUPPORTED_YET*, naming "this build configuration; to
  enable it, build with CMake option `WITH_HYPERGRAPH_OPTIMIZER=ON`".

So a session cannot reach the hypergraph planner on this image even
deliberately. The table's "record, and verify OFF" is therefore cheap insurance
against a future build flag rather than a live hazard, and the
`explain_uses_vector_index()` check still earns its place on the *shape*
failures in §2.4 — no `LIMIT`, `DESC`, a mismatched metric — which are reachable
and silent.

### 2.7 Maintenance, and which churn hook applies

| Operation | Percona Server | Evidence |
|---|---|---|
| `OPTIMIZE TABLE` | accepted, and **does compact** — it rebuilds the table in place and recreates the vector index from the copied rows | *handler0alter.cc*'s in-place-rebuild comment, quoted below; *insert.test* |
| Index replace in one `ALTER` | also compacts, **without rewriting the base table** | *index_build.test*'s index-replace block, quoted below |
| `ALGORITHM=INSTANT`, `LOCK=NONE` | refused while a vector index exists | *percona-hnsw.md*, in its ALTER-support and lock sections |
| Lock level of **any** ALTER leaving a vector index | **`LOCK=SHARED`** — never ONLINE, so **readers are not blocked, writers are** | *handler0alter.cc*'s *innobase_vector_exist* gate and the *HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE* return, both quoted below |
| `DELETE` | drops the row from results immediately; leaves a graph tombstone | *search.test*; *index_build.test* |
| Vacuum-like background process | **none** | no such test or variable on the tree |

**`OPTIMIZE TABLE` does compact, and an earlier revision of this document said
it did not.** The source is explicit — *handler0alter.cc* carries this comment
above the branch that clears *online* for an in-place rebuild:

```
/* A table with a vector index can be rebuilt in place. The rebuild
recreates every index on the new table, the vector index among them, so
the graph and a new aux are built from the copied rows, with their labels
intact and base_pk following the new primary key.

Such a rebuild is not ONLINE - the branch below clears `online` for it,
because the row log cannot maintain a graph while DML runs against it.
LOCK=SHARED it is.

percona_vector.alter_rebuild covers the shapes: FORCE, OPTIMIZE,
ENGINE=InnoDB, ROW_FORMAT, a primary key swap, ADD and DROP COLUMN, and
the AUTO_INCREMENT ones that need the label counter to survive. */
```

`OPTIMIZE` is named in that list, the graph is rebuilt "from the copied rows",
and a DELETEd row is not among the copied rows — so the tombstones go.

The error is worth recording because of how it happened, not just that it did.
*index_build.test* writes `the first working "OPTIMIZE" for a vector index`
**with scare quotes**, hedging to mean "the first thing that does what
you would want of OPTIMIZE, at index level". The block quote below preserves
them faithfully; the prose then re-rendered it as `` `OPTIMIZE` `` and read a
hedge as a claim about the statement. A quoted hedge is not evidence for the
unhedged version of itself.

```
--echo # A DELETEd row is not collected by the build, so replacing the index
--echo # compacts tombstones — the first working "OPTIMIZE" for a vector
--echo # index. The labels are kept, so row 2's leaves a gap.
DELETE FROM t1 WHERE id = 2;
ALTER TABLE t1 DROP KEY k2, ADD VECTOR KEY k3 (v) TYPE hnsw (M = 4),
  ALGORITHM=INPLACE;
```

**`maintain()` is still the index replace, for a different reason.** Both
statements compact; they differ in what else they do. `OPTIMIZE TABLE` rewrites
the whole base table — every row, every secondary index — to reclaim graph
tombstones, so its cost is dominated by work that has nothing to do with the
vector index, and the churn phase would be measuring a table rebuild. The
index replace touches only the aux and reuses each row's existing label, which
*index_build.test* proves by asserting `TABLE_ID` is unchanged across
it. That is the narrower repair and the one whose cost means something.

Anyone sizing the churn phase should know `OPTIMIZE TABLE` is available, which
the previous wording would have wrongly ruled out.

**The index replace takes a SHARED lock, so it blocks writers and not readers.**
This was an open question in a previous revision's §7, on the reasoning that
the "LOCK=SHARED it is" in the rebuild comment above was scoped to a table
*rebuild* and the index replace is not one (`TABLE_ID` is unchanged, per
*index_build.test*). That scoping was right but the question was answerable
anyway, further down the same *handler0alter.cc*, from a rule that is not about
rebuilds at all — the *innobase_vector_exist* gate in
*ha_innobase::check_if_supported_inplace_alter* and the return beneath it:

```
/* No ALTER that leaves a vector index on the table runs with LOCK=NONE,
whatever it changes. A rebuild applies concurrent DML through the row
log, and row_log_apply maintains neither the HNSW graph nor the aux
table, so a row written during the rebuild would reach the new table and
not the index. Same reason ADD VECTOR INDEX is offline above.
...
if (online && innobase_vector_exist(altered_table)) {
  ha_alter_info->unsupported_reason = innobase_get_err_msg(
      ER_ALTER_OPERATION_NOT_SUPPORTED_REASON_VECTOR_NOLOCK);
  online = false;
}
...
return online ? HA_ALTER_INPLACE_NO_LOCK_AFTER_PREPARE
              : HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE;
```

`DROP KEY vi, ADD VECTOR KEY vi` leaves a vector index, so
*innobase_vector_exist* holds, *online* is forced false, and the
handler returns *HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE*. A shared metadata
lock **permits concurrent reads and blocks writes** for the duration.

Two consequences, and they pull in opposite directions:

- **§5's search-during-DDL test stays meaningful, and is meaningful *because*
  of this.** `a86503214416` fixed an assertion when a vector search runs during
  an in-place index replace; if the replace excluded readers there would be
  nothing to test. It does not, so a reader can be mid-scan while the graph is
  replaced under it, which is exactly the state that assertion marked.
- **The churn phase's writers stop for the whole rebuild.** On a 1M-row corpus
  that is on the order of a bulk build (§2.5), and `harness/workloads/churn.py`'s
  `run` times the `driver.maintain()` call, so the cost is recorded — but any
  concurrent write workload is paused,
  not slowed, and that belongs in `docs/07-planning-a-run.md` before anyone
  schedules it.

A comment a few lines below that gate notes one exception that does not apply
here: dropping the **last** vector index may run online, because the new table
has no graph to maintain. The replace keeps one.

So, hook by hook, over the churn hooks `EngineDriver` declares in
`harness/drivers/base.py` — `prepare_churn`, `wait_index_synced`, `maintain`
and `repair_path`:

| Hook | Percona | Why |
|---|---|---|
| `prepare_churn()` | **no-op — inherit** | There is no autovacuum-equivalent to pin down. This hook exists for pgvector, whose autovacuum decides on its own schedule whether tombstones are still there when recall is re-measured; Percona has no background graph maintenance at all, so there is no knob whose state could differ between the baseline and the post-churn measurement. |
| `wait_index_synced()` | **no-op — inherit `0.0`** | The graph is maintained on the write path, inside the `INSERT`. There is nothing to wait for and the honest value is zero, which is what the inherited method returns. Returning a small positive number "to be safe" would be a fabricated measurement in a field documented as a measurement. |
| `maintain()` | **implement** — `ALTER TABLE t1 DROP KEY vi, ADD VECTOR KEY vi (v) TYPE hnsw (…), ALGORITHM=INPLACE`, returning `"ALTER TABLE (index replace)"` — **but it returns `None` at the pinned version; see the correction below** | This is the one hook Percona changes relative to MariaDB and AliSQL, both of which inherit `None`. Percona has a repair path, it is the only MySQL-family engine here that does, and `maintain()`'s contract is to run it and name it. |
| `repair_path` | **`"index-replace"` on the class; `"none"` on the instance at the pinned version** | The comment over `EngineDriver.repair_path` is explicit that this is declared and not inferred, because the churn workload stamps it onto the record *before* maintenance runs. The class attribute still declares `"index-replace"` because that **is** this engine's repair mechanism; `PerconaDriver.declare_bulk_vector_index_build` shadows it with `"none"` per instance when the host says the bulk build is broken, which is what the records then carry. Note that the same comment says "MariaDB, AliSQL, Valkey and Percona Search inherit `"none"`" — "Percona Search" there is the MongoDB engine, not this one, and the sentence stays true. |

> **CORRECTION, and it reverses the argument in the `maintain()` row above.**
> That row said returning `None` "would lose the measurement this engine is
> uniquely able to supply". **At the pinned `source.tag` there is no such
> measurement to lose**, and the row was written before anyone had run the
> repair. `maintain()`'s `ALTER … ADD VECTOR KEY … ALGORITHM=INPLACE` is the
> same statement `capabilities.bulk_vector_index_build: false` records as
> killing the server. Measured on `full-20261015-215006`: it returned in
> **1–4 ms** and changed nothing — `index_bytes` identical to the byte either
> side of it and recall identical to five decimals in **all 20** before/after
> pairs (10 ops units × 2 fractions, three datasets). What the before/after
> pair supplied was the degraded index measured twice and published as
> `after ALTER TABLE (index replace) (0.0 s)`. So the driver now declares no
> repair path when the host says the bulk build is broken, and the churn
> workload writes one record per fraction — the shape the five engines with no
> repair path already use. Nothing is lost; a false row is removed. Flip
> `capabilities.bulk_vector_index_build` in `config/engines/percona.yml` when
> the server is fixed and both the bulk build and this repair come back.

Two warnings about that `maintain()`, both of which make it a *measurement* and
not a tidy-up:

- The replace must re-state the full option list. The DDL that rebuilds the
  index decides the new graph's `M`, `ef_construction` and `metric`, and
  omitting them silently rebuilds at Percona's defaults — `M=25` instead of the
  profile's. The post-maintenance recall would then be a different index's
  recall. Read the options back out of `SHOW CREATE TABLE` (§2.3) and re-send
  them rather than re-deriving them from the profile.
- It rebuilds the whole graph, so on a 1M-row corpus it costs on the order of a
  whole bulk build. `harness/workloads/churn.py`'s `run` times the call, so the
  cost is recorded rather than hidden, but a churn phase that previously ran in
  minutes for the
  MySQL family will not for Percona. That belongs in `docs/07-planning-a-run.md`
  before anyone schedules it.

## 3. `innodb_hnsw_max_memory` is not a cache, and that is a fairness problem

This is the finding most likely to change what gets run, so it has a section.

`CLAUDE.md` already treats graph memory as a fairness parameter: "**Both
MySQL-family engines ship a 16 MiB graph cache by default**
(`mhnsw_max_cache_size`, `vidx_hnsw_cache_size`). The framework sets both from
the resource pass; leaving the default judges a vendor on a value it plainly
intended you to change." The obvious move is to add `innodb_hnsw_max_memory` as
the third member of that family, driven by `graph_cache_bytes`.

**That would be wrong, and wrong in a way that produces a plausible number.**
The two things are not the same kind of limit.

- MariaDB's and AliSQL's are **caches**. A graph larger than the cache still
  works; it is read from the companion table and performance degrades.
- Percona's is a **hard allocation ceiling**. *percona-hnsw.md*, in its
  memory-limit section: "`innodb_hnsw_max_memory` bounds, in bytes, the memory
  held by HNSW graphs **across all tables and all indexes**." Exceeding it does
  not evict anything — it **fails the statement**. The same section says so, and
  the *percona_vector* suite's *hnsw_memory_used.result* records it, in the
  block it heads "The limit is compared with this number":

```
# The limit is compared with this number: at it, an insert is refused,
# one byte above it, the insert goes through.
SET GLOBAL innodb_hnsw_max_memory = @t1_loaded;
INSERT INTO t1 VALUES (4, STRING_TO_VECTOR('[0,0,0,1]'));
ERROR HY000: Memory capacity of N bytes for 'innodb_hnsw_max_memory' exceeded.
The vector index graph was left unchanged. Raise innodb_hnsw_max_memory and retry.
```

There is **no per-node eviction**. The only way the counter goes down is a whole
graph being freed: *percona-hnsw.md* says the counter goes "down by exactly a
graph's bytes when that graph is freed (DROP, a rebuild, eviction of its
table)", and the only eviction discussed anywhere on the tree is
dictionary-cache eviction of a whole *dict_table_t*, in its §22. So **the
entire graph must fit** in `innodb_hnsw_max_memory`.

**And "the load stops partway through" is not the only failure — it is not even
the worst one.** *percona-hnsw.md*, in the same section: "The limit is checked before
an insert or a load starts **and again as a cold search faults nodes in**." A
*query* can therefore fail with `ER_CAPACITY_EXCEEDED`, not only an INSERT.

Combine that with the lazy load §2.6 establishes — `Innodb_hnsw_memory_used`
reads `0` after a restart and the graph is faulted in node by node
(*hnsw_memory_used.result*, and *percona-hnsw.md* §11) — and the
exposure is not a one-off at ingest. It **recurs at the start of every
post-restart query phase**, because each one re-faults the graph from nothing
and re-crosses the limit on the way up. A run that survived ingest can still
die in the first minute of querying, after the expensive part is already paid
for, and the orchestrator restarts servers between phases.

That shape also makes the failure harder to read than an ingest failure: an
ingest that stops has an obvious cause, whereas a query phase that errors
partway through a recall sweep looks like an engine defect.

Setting it equal to the other two engines' `graph_cache_bytes` would therefore
compare a cache budget against a wall. Under `normalized.yml`'s
`graph_cache_fraction: 0.30` the three engines would get the same number of
bytes and one of them would be unable to finish the load.

### What the default costs, with the inputs named

**These figures are projections, not measurements.** Nothing here was run
against a server. The chain is:

| Input | Status |
|---|---|
| per-row floor formula `dims × 4 + 2 × M × 8` | **measured on Percona**, in the sense that it is the floor an MTR test asserts the server stays above — *hnsw_memory_used.result* sets `SET @floor = 1000 * (64 * 4 + 2 * 16 * 8);` and then asserts `actual >= floor` and `actual <= 2 × floor + 2 × 65536` |
| row counts | **supplied** — from this repo (`harness/datasets.py`'s `tag_threshold_for` quotes "glove's 1,183,514 training rows"; `docs/07-planning-a-run.md` §8 calls fashion-mnist "a 60,000-row smoke run"; `config/profiles/smoke.yml` sets `subset_rows: 20000`) |
| dimensionality | **supplied** — from the dataset names |
| `M` | **supplied** — from the profiles |
| 1 GiB default | **measured** — *innodb_hnsw_max_memory_basic.result*'s default block |

Because row counts, dims and `M` are all supplied, **none of the numbers below
is a measurement.** They are what the server's own asserted floor implies for
these configurations, and the regime is "graph fully resident, one vector index,
nothing else on the server" — which is the regime a benchmark run is in, and is
also the most favourable one, since the bound is server-wide.

| Dataset / phase | rows | dims | M | floor | floor ÷ 1 GiB | over the default? |
|---|---|---|---|---|---|---|
| fashion-mnist-784, ops smoke subset | 20,000 | 784 | 16 | 67,840,000 B | 0.063 | no |
| fashion-mnist-784, ann training set | 60,000 | 784 | 16 | 203,520,000 B | 0.190 | no |
| glove-100-angular (`main`) | 1,183,514 | 100 | 16 | 776,385,184 B | 0.723 | floor no, **ceiling yes (1.446)** |
| glove-100-angular (`main`) | 1,183,514 | 100 | 32 | 1,079,364,768 B | **1.005** | **yes** |
| gist-960-euclidean (`full`) | per 1M | 960 | 16 | 4,096,000,000 B | **3.815** | **yes, ~4×** |

Read three ways:

- **The smoke profile fits the default comfortably** (0.19 GiB at worst). So the
  smoke run will *not* surface this, and must not be taken as evidence that the
  default is fine.
- **The `main` profile does not.** At `M=32` on glove-100 the *floor* alone
  exceeds 1 GiB, so the load is guaranteed to stop. At `M=16` the floor fits but
  the MTR's own ceiling (2× floor) does not, so whether it completes is a
  coin-flip on the layer draws — the worst possible failure shape for a two-day
  run.
- Expressed as a row limit at the default, same basis: 316,551 rows at 784 dims
  and `M=16`; 1,177,348 rows at 100 dims and `M=32`; 262,144 rows at 960 dims
  and `M=16`.

### The policy: `innodb_hnsw_max_memory = 0`

**This is decided, not proposed.** The choice was the user's to make, because it
is a fairness judgement rather than a fact, and it has been made:
`config/engines/percona.yml` sets `--innodb-hnsw-max-memory=0` (unlimited),
with `Innodb_hnsw_memory_used` collected onto the `index_build` record. The two policies below are
kept as the justification for it, not as an open question — a reader asking
"why is this engine's graph budget unlimited when the other two have a cache
size" needs the whole argument, and a decision whose reasoning has been deleted
gets relitigated.

What the policy does **not** change: `--innodb-hnsw-max-memory` must not be fed
from `graph_cache_bytes`, and the report must state the asymmetry rather than
paper over it.

1. **`0` (unlimited), with `Innodb_hnsw_memory_used` collected onto the
   `index_build` record — CHOSEN.** It is collected there and nowhere else:
   `ingest`, `concurrency`, `filtered`, `churn` and `ef_calibration` carry no
   reading, so this audit trail covers one of the six ops phases and the
   earlier wording "per phase" over-claimed it. Widening it is unlanded work.
   The graph is then bounded only by the container's memory limit,
   which the orchestrator already sets and already records — so Percona is held
   to the same *total* memory budget as every other engine, measured where the
   other engines' budgets are measured. This is the honest like-for-like, and
   the recorded status variable is what makes it auditable — **with one caveat
   that has to travel with every figure taken from it**, below.
2. **`graph_cache_bytes`, accepting that some configurations will fail.** This
   is "same number of bytes for the same purpose", which is defensible on its
   face, but it produces an engine that cannot complete the load at `M=32` while
   the other two complete it slowly. A benchmark in which one engine errors and
   two degrade is reporting on a difference in limit semantics, not on index
   quality, and the chart would not say so. And under this policy the error is
   not confined to ingest: per the note above, a post-restart query phase can
   raise `ER_CAPACITY_EXCEEDED` as it faults the graph back in.

**`Innodb_hnsw_memory_used` understates the graph, and policy (1) leans on it.**
*percona-hnsw.md*, on what the gauge omits: "It does not count the node map
(*HNSW::m_nodes*), which uses the normal heap, roughly 30 to 50 bytes per node,
or search scratch space."

So the gauge is a floor on true graph memory, not a measurement of it. At
glove-100's 1,183,514 rows that is **35.5 to 59.2 MB unreported** — inputs: the
row count is **supplied** (`harness/datasets.py`'s `tag_threshold_for`) and the 30-to-50-bytes
range is **Percona's own stated estimate**, so the figure is a projection from a
range, not a measurement, and it is one node map per vector index. Small against
a 16 GB container and not small against a 1 GiB ceiling: under policy (2) it is
about 3 to 6 per cent of the entire budget, invisible to the variable that is
supposed to be auditing it. A reader reconciling `Innodb_hnsw_memory_used`
against the container's RSS will find the difference and should not have to
rediscover why.

**So: any figure quoted from `Innodb_hnsw_memory_used` is a lower bound on the
graph's memory and must be labelled as one.** That applies to the chosen
policy's audit trail specifically — the whole point of recording the variable
under policy (1) is to show Percona stayed inside the container budget, and the
number that shows it is short by one node map per vector index.

The value is a declared knob either way, so it lands in `run-manifest.json` and
in `ann_fingerprint()`'s keying, which is what stops a 1 GiB curve being
silently reused under an unlimited manifest. Under the chosen policy that
keying is what keeps any curve measured during the smoke gate — where the
default was never reached, because the gate's worst case is 0.19 GiB and the
ceiling is 1 GiB, so **the gate cannot surface this at all** — from being
reused under a manifest that set the knob.

## 4. File-by-file change list for the follow-on task

Nothing in this list is done. Items marked **done** are the two long-lead
artefacts this task wrote.

### 4.1 `harness/drivers/mysql_family.py` — the `Dialect`

`MySQLDialect` is a flat dataclass of values, and three of Percona's
differences are not values today because `MySQLFamilyDriver` spells the
statements out. Add these fields, each with its default set to what MariaDB and
AliSQL already do so neither engine changes:

| New field | Default | Percona |
|---|---|---|
| `pk_type: str` | `"INT"` | `"BIGINT UNSIGNED"` |
| `index_clause: str` | `"VECTOR INDEX {name} (v) M={m} DISTANCE={metric}"` | `"VECTOR KEY {name} (v) TYPE hnsw (M = {m}, ef_construction = {ef_construction}, metric = {metric})"` |
| `distance_expr: str` | `"vec_distance_{metric}(v, %s)"` | `"DISTANCE(v, %s, '{metric_upper}')"` |
| `plan_marker: str` | `""` (meaning "use `index_name`") | `"vector_ann"` |
| `ef_construction: Optional[int]` | `None` | from `IndexSpec` |

Then:

- `MySQLFamilyDriver._create_table_ddl` interpolates `pk_type` and
  `index_clause` instead of hard-coding `id INT PRIMARY KEY` and the MariaDB
  index clause, and `_index_clause` builds the clause itself.
- `_select` interpolates `distance_expr` instead of hard-coding
  `vec_distance_{...}(v, %s)`. Note the **parameter order changes**: MariaDB's
  filtered form binds `(threshold, vector)` because the predicate precedes the
  `ORDER BY`; Percona's is the same order, because the metric is a literal and
  not a parameter. Keep it that way — making the metric a bound parameter would
  add a third placeholder and reorder every call site.
- `explain_uses_vector_index` must test `plan_marker` when set.
  **This is not cosmetic, and it is not a hypothetical.** That method's verdict
  starts as `used = self.dialect.index_name in plan`, a substring test against
  the joined EXPLAIN text. Percona's own recorded output contains a plan that
  defeats it — the *percona_vector* suite's *optimizer.result*, in its
  filesort case:

  ```
  id select_type table partitions type possible_keys key  key_len ref  rows filtered Extra
  1  SIMPLE      t1    NULL       ALL  v1            NULL NULL    NULL X    100.00   Using filesort
  ```

  The only index on that table is a `VECTOR KEY`. The access type is `ALL`,
  `key` is `NULL`, `Extra` is `Using filesort` — **a full table scan** — and the
  index name `v1` is right there in `possible_keys`, so
  `index_name in plan` returns True. That is the brute-force scan this repo
  guards against, published as an ANN search, demonstrated rather than argued.
  The same file's `eq_ref` case is the second shape
  (`type=eq_ref, possible_keys=u,v, key=u` — the vector index offered, a
  different one taken), and the optimizer trace it records gives the mechanism:
  `{"cause": "vector", "index": "v", "usable": false}` in
  *potential_range_indexes*. A filtered query — `WHERE tag < ? ORDER BY
  DISTANCE(…) LIMIT k`, which is what the ops harness runs — is exactly that
  `eq_ref` shape.

  `vector_ann` is the access type and appears only when the vector path was
  taken, which *optimizer.result*'s own first case shows. This is the same vacuity shape this repo
  has already been bitten by ten times — a substring satisfied by a longer
  string — and it is worth noticing that `CLAUDE.md`'s own history records
  `index_plan_marker: "v"` having been wrong in `mariadb.yml` for exactly this
  reason.
- **Issue `EXPLAIN FORMAT=TRADITIONAL`, not bare `EXPLAIN`** — the `EXPLAIN`
  that `explain_uses_vector_index` executes. One free
  hardening while that method is open. `vector_ann` is a value in the `type`
  column, which only the tabular format has. `explain_format` defaults to
  TRADITIONAL so a bare `EXPLAIN` works today — but the variable is settable and
  reachable through `VB_SERVER_ARGS`, and under `TREE` or `JSON` the marker
  vanishes and the check records a **false negative about a server that did use
  the index**. That lands in the report's Validity section and sends someone
  hunting an engine bug that is not there. No MTR test could catch this, because
  MTR output is always TRADITIONAL; the suite spells the keyword out anyway in
  *optimizer.result*. `overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/module.py`
  already does this.

  **The risk this creates for the OTHER two engines is settled, and it was a
  real one rather than a theoretical one.** `explain_uses_vector_index` is
  shared: hardening it sends `EXPLAIN FORMAT=TRADITIONAL` to MariaDB and AliSQL
  as well, and a server that rejected the keyword would have turned the ops
  plan check into a **permanent false negative on three engines to protect
  one** — every configuration reported as not using its index, the whole
  Validity section full, and the one engine the change was for the only one
  working. Established by reading the three parsers rather than by assuming
  MySQL-family compatibility: MariaDB 11.8.8's *sql_yacc.yy*, MariaDB 12.3.2's
  and AliSQL 8.0.44's all accept `FORMAT = TRADITIONAL` in their *describe*
  rule. Recorded here because "it is
  the default, so the keyword must be accepted" is the kind of inference this
  document is not allowed to make.

  The ann overlay's shared `_verify_plan` is deliberately NOT hardened, and
  that is a different judgement rather than an inconsistency: its check is
  name-only, and an index name appears in `TREE` and `JSON` output too, so the
  keyword would buy it nothing. Only a check that reads the `type` column
  needs the format pinned.
- `MySQLFamilyDriver.capabilities` currently returns a constant
  `{"incremental_index": True, "ef_construction_tunable": False}`. For Percona
  both values change, and `incremental_index` must depend on
  `IndexSpec.build_mode` — see §2.5 for why, and for what `build.py` does if it
  does not.
- `effective_ef(ef, k)` should become `max(ef, k)` for Percona — see §1.
- `maintain()` and `repair_path` — see §2.7.
- A `PerconaDriver(MySQLFamilyDriver)` class and a `PERCONA_DIALECT` beside
  `MARIADB_DIALECT` and `ALISQL_DIALECT`.

### 4.2 `harness/drivers/postgres.py` — `_driver_classes()`

`_driver_classes` in `harness/drivers/postgres.py` is the driver-class-name
registry. Add one entry, `"PerconaDriver": PerconaDriver`, to the dict it
returns, and import it from `.mysql_family` at the top of the same function.
Its docstring is explicit that this is
"one entry per architecture, not per engine" — Percona is a new architecture
only because of the dialect gaps in §4.1; once those are `Dialect` fields it is
arguably a variant, but it needs its own class for `capabilities()`,
`maintain()`, `repair_path` and `effective_ef`.

`_driver_table`, beside it in the same module, maps engine name to class for
"callers that still think in engines". Add `"percona"` there too, for symmetry — the
orchestrator passes `--driver` by name so it is not load-bearing.

### 4.3 `config/engines/percona.yml`

**Do not write an `sql:` block.** At the time of writing, task C1 is in flight
(uncommitted in the main checkout, with `tests/test_c1_engine_sql.py` present
and `config/engines/mariadb.yml` already stripped) and removes the `sql:` block
from every engine config, because its contents had drifted — `index_plan_marker:
"v"` against a driver that names the index `vi`, a `create_table` with no `tag`
column beside its own `tag` filter, and no mention of `ctag`. A new config
carrying one would fail that test and reintroduce exactly what C1 removes. The
dialect lives in the driver; this file carries build, server flags and `runtime:`.

Required keys, modelled on `alisql.yml`:

```yaml
name: percona
display_name: Percona Server
family: mysql
source:
  # The key is `tag` even though the value is a SHA, and that is not a
  # cosmetic choice. `_source_tag` in `orchestrator/engines.py` is
  # `return str(source.get("tag") or source.get("version") or "")` — there is
  # no `sha` key, so a `source: sha:` block would make `_source_tag` return
  # the empty string and `prepare_percona` die — it passes `""` as the
  # default, so there is no hard-coded fallback for THIS engine; the three
  # other `prepare_*` functions that read the key do pass one.
  # `OpsRun.run_harness` in
  # `orchestrator/ops_pass.py` reads the same key
  # into `VB_ENGINE_TAG`, which is what reaches the manifest. `git archive`
  # takes a SHA wherever it takes a tag, so the mechanism is unchanged.
  tag: 56124b03fdc5954e2401ab4d2c7bf7cb3a767158   # tip of PR #6278 on base 9.7
  upstream: https://github.com/percona/percona-server.git
  # No `submodules:` key, deliberately: all four of the pin's submodules are
  # switched off in the build instead, which takes four cmake flags, one of
  # them non-obvious. See docker/percona/Dockerfile note 2.
image:
  runtime: vector-bench/percona-runtime
  bench: vector-bench/percona-bench
build:
  march: x86-64-v3        # MUST equal every other engine's — see CLAUDE.md
  cmake_flags: [...]      # as in docker/percona/Dockerfile
server:
  common: [--skip-name-resolve]
  normalized:
    - --innodb-buffer-pool-size={buffer_bytes}
    - --innodb-hnsw-max-memory=0          # §3 POLICY, decided; NOT {graph_cache_bytes}
    - --innodb-flush-log-at-trx-commit=2
    - --innodb-doublewrite=0
capabilities:
  ef_construction_tunable: true           # unlike mariadb and alisql
  m_per_index: true
  requires_read_committed: false          # unlike alisql
  filtered_search: true
runtime:
  order: 70
  driver: PerconaDriver
  ann_constructor: PerconaServer          # must be unique — see below
  port: 3306
  data_mount: /var/lib/vbench
  server_data_mount: /server-data/data
  credentials: {user: bench, password: bench}
  group: extra
  probe: ["sh", "-c", "/opt/percona/bin/mysql -ubench -pbench --socket=/var/run/vbench/percona.sock -e 'SELECT 1' >/dev/null 2>&1"]
  chart: {color: "#ff7f0e", marker: "P", linestyle: "-", label: "Percona Server (InnoDB HNSW)"}
  label: "Percona Server (InnoDB HNSW)"
```

**Why the key is `tag` and not `sha`, in prose rather than only in that
comment.** `_source_tag` in `orchestrator/engines.py` is
`return str(source.get("tag") or source.get("version") or "")`. There is no
`sha` key, so a `source: sha:` block would make it return the empty string and
`prepare_percona` in `scripts/prepare-sources.sh` die: it reads the key with
`""` as the default and `die`s on an empty value. There is no hard-coded
fallback for this engine. `prepare_alisql`, `prepare_mariadb` and
`prepare_pgvector` do pass one (`AliSQL-8.0.44-2`, `mariadb-11.8.8`, `v0.8.6`),
which is why "it falls back to a default tag" reads as true and is not.
`OpsRun.run_harness` in `orchestrator/ops_pass.py` reads the same key
into `VB_ENGINE_TAG`, which is what reaches the manifest. `git archive` takes a
SHA wherever it takes a tag, so the mechanism is unchanged.

Two collisions the validator refuses, both named in `CLAUDE.md`: a duplicate
`ann_constructor` and a duplicate image tag. `PerconaServer` collides with
nothing — the six existing constructors are `MariaDB`, `AliSQL`, `PGVector`,
`MariaDB123`, `PerconaSearch`, `ValkeySearch`. Note `PerconaSearch` is the
MongoDB engine; the two names are close enough to swap by accident, and swapping
them means one engine reports the other's recall, which is the defect
`mariadb123` exists to prevent.

The `#ff7f0e` colour is unused by the six in `GOLDEN` (`#1f77b4`, `#d62728`,
`#2ca02c`, `#5fa8d3`, `#9467bd`, `#e377c2`) and the `P` marker is unused
(`o`, `s`, `^`, `s`, `v`, `D`).

**The guard that currently blocks this file.** A hard-coded engine-family list
asserts the known set, so adding a seventh config fails it until that assertion
is relaxed. That is another task's; this one does not write the file.

### 4.4 `tests/test_engine_registry.py` — `GOLDEN` and `GOLDEN_ORDER`

`GOLDEN` is a name → 12-tuple of `(driver, ann_constructor, port,
data_mount, server_data_mount, (user, password), group, colour, marker,
linestyle, chart label, report label)`, read by the parametrized
`test_runtime_matches_the_old_constants`,
`test_presentation_matches_the_old_constants` and
`test_every_engine_has_a_readiness_probe`, plus
`test_every_engine_is_found`. Add a
`"percona"` row matching the `runtime:` block above exactly.

`GOLDEN_ORDER` is `("mariadb", "alisql", "pgvector", "mariadb123",
"mongodb", "valkey")` — "the original three, then the engines added later".
Percona is appended: `(..., "valkey", "percona")`, with `order: 70` in the
config to put it there.

`test_groups` asserts the group membership tuples literally:
`engines_in_group("extra") == ("mariadb123", "mongodb", "valkey")`. With
`group: extra` that assertion must gain `"percona"`. It is a value assertion and
not a membership one, so it will fail rather than silently pass — which is the
right behaviour and worth not "fixing" by loosening it.

### 4.5 `webui/static/fields.js` — colour and label

Two literal maps at the top of the file, each listing every engine:

- `ENGINE_COLOR` gained `percona: "#ff7f0e",`.
- `ENGINE_LABEL` gained `percona: "Percona Server (InnoDB HNSW)",`.

(Written without line numbers on purpose. These two maps grow by a line per
engine, so a citation here rots on every addition while pointing at nothing a
reader could not find by name in a twelve-line file.)

Both must match `config/engines/percona.yml`'s `chart.color` and `label`
character for character. They are a separate copy of the same facts — the
front end is static and cannot read the engine registry — so this is the one
place in the addition where two files have to be kept in step by hand.

The index-build chart's hint read "What the index costs to make. pgvector
bulk-builds; MHNSW and VIDX build on every INSERT." That sentence is
incomplete with Percona in the chart (§2.5) and now names the third
behaviour. Scoped to the ops path, which is the only path where Percona's
build mode is a real axis — see §4.10.

### 4.6 Profiles

- `config/profiles/smoke.yml` — the gate before every long run. It names no
  engines, so Percona joins automatically once the registry knows it; the
  `description` said "all six engines" and now says seven. Its
  20,000-row ops subset and 60,000-row ann set both fit the default
  `innodb_hnsw_max_memory` (§3), so **the smoke run will not exercise the memory
  ceiling** — that has to be checked deliberately.
- `config/profiles/main.yml` and `full.yml` — no engine list to edit, but §3
  says the memory setting must be right before either is started, and §2.7 says
  the churn phase now costs a full graph rebuild per fraction for this engine.
- `config/profiles/percona.yml` is the new Percona-only profile and is the home
  for the two things only this engine can show: an `ef_construction` sweep, and
  `build_mode: [post, incremental]` as a within-engine comparison. Neither
  belongs in `main`, because neither has a counterpart on MariaDB or AliSQL.
  **The build-mode half runs on the ops path only**, and the ann half could not
  be delivered — see §4.10.
  **The build-mode comparison is unmeasurable at the pinned version** and the
  profile asks for `[incremental]` alone: `ALTER TABLE … ADD VECTOR KEY …
  ALGORITHM=INPLACE` kills the server, so `post` cannot complete a unit. The
  reproduction is recorded at `capabilities.bulk_vector_index_build` in
  `config/engines/percona.yml`, which is also what the ops loop reads to keep
  the default off that mode for every profile that names none. Restore the
  pair when the server is fixed.
- `docs/07-planning-a-run.md` carries the cost model (`_INGEST_ROWS_PER_S` in
  `orchestrator/estimate.py`). Percona's ingest rate is unmeasured, so it has no
  entry, and an estimate for it would be a guess presented as a plan. It stays
  out until the smoke run supplies one.

  **Leaving it out was not enough on its own, and that was a defect rather than
  a gap.** `_effective_rate` ended in `_INGEST_ROWS_PER_S.get(engine, 300)`, so
  an engine with no entry was silently modelled at 300 rows/s — 2x the measured
  MariaDB figure and 5.5x the measured AliSQL one, for an engine that builds its
  graph inside every INSERT. On `main` that put 9.10 h of invented ingest into a
  published total (102.16 h against the honest 93.06 h) with nothing printed to
  say the number was a guess, and it contradicted the comment four lines above
  the table it read from: "an estimate that is optimistic by 3x is worse than
  none". The fallback now returns `None`, and `estimate_load_hours` reports
  `engines_without_ingest_rate` on the same terms as the query half's
  `engines_without_query_rate` — named, excluded from the total, and said so in
  the printed block.

### 4.7 Already written by this task

| Path | State |
|---|---|
| `docker/percona/Dockerfile` | **done**, never built |
| `docker/percona/entrypoint-percona.sh` | **done**, never run |
| `docker/percona/init.sql` | **done**, never run |
| `algorithms/percona/module.py` | **done**, never run |
| `algorithms/percona/config.yml` | **done**, never rendered |
| `scripts/build-images.sh`, `scripts/prepare-sources.sh`, `scripts/lib.sh` | **not done** — see below |

`scripts/lib.sh` needs one line too, and it is a safety line rather than a
functional one. It exports `VB_REPO_MARIADB`, `VB_REPO_ALISQL` and
`VB_REPO_ANNB`, and `assert_not_vendor_repo` loops over exactly
those three to refuse writing inside a read-only vendor checkout, with
`die "refusing to write inside read-only vendor repo: $canon"`. A Percona
vendor checkout gets **no protection at all** until `VB_REPO_PERCONA` is
exported beside them and added to the loop. `CLAUDE.md` names the vendor
repositories as read-only inputs that must never be checked out, fetched into or
modified; right now that rule is enforced for three of them and would not be for
the fourth.

Both build scripts need a `percona` case, and `build-images.sh` will say so
loudly: the `case` in its `build_engine` ends with
`*) die "engine '$engine' has no known build-arg (alias_of=$base)"`, so a
missing entry is an immediate, clear failure rather than a silent one. The
entry follows the convention of the `mariadb` and `alisql` cases beside it —
`percona) bargs+=(--build-arg "PERCONA_TAG=${tag}") ;;` — and
`docker/percona/Dockerfile` declares `PERCONA_TAG` as a build `ARG` to
receive it.
`prepare-sources.sh` must ship a `git archive` of the pinned SHA, which is what
its `prepare_percona` does, on the pattern `prepare_alisql` sets, and must
*not* try to carry submodules.

### 4.8 The `march` field goes blank in **two** places, not one

There are two independent `-march` readers, in two different containers, and
only one of them is reachable from the overlay module.

**The ann path.** `VBMySQLBase._image_march` in
`overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py`
read a fixed pair of paths, `/opt/mariadb/.march` and `/opt/alisql/.march`, and
returned `"unknown"` otherwise. `percona/module.py` overrode it to try
`/opt/percona/.march` first and then fall back, so **the ann path was already
correct** in the file the design task wrote.

**The ops path, which the override cannot reach.** `_detect_march` in
`harness/main.py` has its own, and these three lines are the whole of it:

```python
    for path in ("/opt/mariadb/.march", "/opt/alisql/.march",
                 "/opt/pgvector-artifacts/.march",
                 "/opt/percona/.march"):
```

consumed in the same module's `main` as `march=args.march or _detect_march()`. That runs in the
**harness container**, which mounts `harness/` and not `overlay/`, so no
override in an ann-benchmarks module can affect it. `orchestrator/ops_pass.py`
contains the string `march` zero times, so nothing passes `--march` in and the
`or` falls through to the detector. **Percona's ops records would therefore
have carried `march: "unknown"` too**, with its ann records looking fine —
half a run measured and nothing saying which half.

**Both tuples now carry the path and the override is gone**, which is what the
last paragraph of this section asks for. `percona/module.py` keeps a comment
where the override was, naming the second reader, so that the absence reads as
a decision rather than as an omission someone should repair.

Either blank field is the failure shape this project has already paid for once:
a probe that hard-codes a fixed set and whose figure reaches a published
artefact looking measured. It matters more than it looks, because "never mix
`-march` between engines" is a stated invariant — all seven engines compile SIMD
distance kernels — and `march` is the field that would catch a mixed build. Two
readers means two chances to miss it, and the one the module can fix is not the
one a long run depends on most.

The follow-on task should add `/opt/percona/.march` to **both** tuples and then
drop the override, since the next engine added will hit the same thing twice.

### 4.9 A family-wide latent defect, recorded and deliberately not fixed

This is not Percona's and not this task's, but it was found while matching the
family's entrypoint and it should not be lost.

`docker/percona/entrypoint-percona.sh` copies a two-line pattern from its
siblings. The **call site** is byte-identical in all three — it is in
`start_server` in each of `docker/alisql/entrypoint-alisql.sh`,
`docker/mariadb/entrypoint-mariadb.sh` and `entrypoint-percona.sh` — and
`user_args` behaves identically in all three though it is not byte-identical:
alisql's and percona's are one-liners, mariadb's spreads the same
test over four lines with a comment. The behaviour that matters is the last
command in the body, which is the same everywhere:

```bash
user_args() { [[ "$(id -u)" -eq 0 ]] && printf '%s' "--user=root"; }
...
  local u; u="$(user_args)"; [[ -n "$u" ]] && args+=( "$u" )
```

Under the `set -euo pipefail` each of the three sets at the top of its own
file, when
the container does **not** run as root, `user_args` returns 1 — the status of
its failed `[[ ]]` — and because the assignment is written as the separate
command `u="$(user_args)"` rather than `local u="$(user_args)"`, that non-zero
status is the command's status and `set -e` aborts the script. The server is
never exec'd and nothing is logged.

Reproduced with the pattern extracted verbatim and `id -u` stubbed:

```
FAKE_UID=0     -> REACHED_EXEC args=--datadir=/x --user=root   rc=0
FAKE_UID=1000  -> (no output)                                  rc=1
```

It is **latent**: no image in this repo sets `USER`, so every container runs as
root and takes the `rc=0` path. It would fire the moment one does, or under a
runtime that drops to a non-root user — and it would present as a server that
never starts and never says why, which is the expensive kind.

Matching the family was the right call for a new engine; diverging would have
left one of three entrypoints behaving differently for a reason unrelated to
Percona. The fix, when someone takes it, is one character per file (`local
u="$(user_args)"`, which returns `local`'s status) applied to all three
together.

### 4.10 `orchestrator/ann_pass.py` — Percona hits its `raise`

This file is the one that *renders* `algorithms/percona/config.yml`, which
this task wrote,
and the previous revision of this document never mentioned it.

`render_config` dispatches on engine name and ends:

```python
        raise ValueError(f"unknown engine: {engine}")
```

Its branches, in the order the function tests them, are
`if engine in MARIADB_ENGINES + ("alisql",)`, then
`elif engine == "pgvector":`, `elif engine == "valkey":`,
`elif engine == "percona":` and `elif engine == "mongodb":`. **Percona matched
none of them and landed on the `raise`.** Either add it to the MySQL-family
condition or give it a branch of its own, and a branch of its own is what the
two things §4.6 proposes actually need.

The branch this section asked for is the `elif engine == "percona":` one, and
it was inserted at the position mongodb's branch used to occupy — which is why
this list labelled that position `(mongodb)` for two revisions, pointing at
Percona's own branch while the sentence above it said Percona had none. **That
is the error this section's own citation form made possible**: the branch was
named by a line number, the line number was right, and the label on it was
wrong. Naming the branch condition instead makes the label and the thing it
labels the same string. The branch conditions below are checked against
`render_config`'s body by `tests/test_e10_failures_explain_themselves.py`,
which is what caught it.

- **`efConstruction`.** No existing MySQL-family branch emits it. The two that
  did were pgvector's and valkey's, and both sat inside a build-mode loop —
  `for build_mode in ann.get("pgvector_build_modes", ["post"]):` and
  `for build_mode in ann.get("valkey_build_modes", ["post"]):`. The
  MySQL-family branch has no equivalent, so a Percona `arg_groups` carrying
  `efConstruction` had nowhere to come from. `"efConstruction": ef_constructions`
  occurs five times in the function now; the third, in the percona branch, is
  the one this section asked for, the fourth is the myvector branch, added
  with the eighth engine on the same reasoning, and the fifth is the
  villagesql branch, added with the ninth.
- **`build_modes`.** Those same two loops read `pgvector_build_modes` and
  `valkey_build_modes` out of the profile's `ann` block. A
  `percona_build_modes` key needs the same treatment, and it is the key that
  makes §2.5's within-engine build comparison runnable.

So §4.6's Percona-only profile is not a profile edit alone; it is a profile
edit plus a `render_config` branch. The failure without it is loud — a
`ValueError` at render time, before anything is built — which is the good kind.

#### The `build_modes` half of that could not be delivered, and this is why

**`ann.percona_build_modes` is REFUSED by the new branch, not honoured.** This
is the one item in §4 that the follow-on task stopped on rather than
implemented, and the reason is a file §4.7 marks **done**.

`pgvector/module.py` and `valkey/module.py` each read `build_mode` out of
`method_param`, validate it against `("post", "incremental")`, branch on it in
`fit`, and report it back in `get_additional`. **`algorithms/percona/module.py`
contains the string zero times, and so does the `VBMySQLBase` it inherits
from.** Emitting the key anyway would produce two run groups, `post_build` and
`incremental_build`, differing only in a `method_param` nothing reads — two
*identical* recall curves under two labels, with no field in the record to say
they were the same measurement. That is the shape this repo guards against
everywhere else, so `render_config` raises with a message naming the missing
support instead.

Three consequences worth carrying:

- **The ops half is real and is where the comparison comes from.**
  `PerconaDriver.capabilities()` derives `incremental_index` from
  `IndexSpec.build_mode`, and `create_schema`/`create_index` split the index
  out of the table definition in `post` mode, so `build_wall_s` times a genuine
  bulk build. `config/profiles/percona.yml` runs both modes through the ops
  path. Since `build_wall_s` and `index_bytes` are ops-only fields anyway, the
  measurement §2.5 argues for is available — it is the *recall* curve per build
  mode that is not.
- **§4 never mentions `orchestrator/cli.py`, and it had to change.** The ops
  path hard-coded `build_modes = ["post"] if engine != "pgvector" else …`, so
  `PerconaDriver`'s build-mode support was unreachable from a profile until
  that lookup became per-engine. `estimate._ops_units` carries the same list
  and had to match, or the printed estimate is short by a whole ops unit of
  every query term. Both now read a shared two-engine mapping.
- **It makes Percona a fourth engine that cannot supply the build-mode
  cross-check.** `report/crosscheck.py:build_mode_axis_engines` drops the axis
  for any engine whose recall path is silent about it, which was MariaDB,
  MariaDB 12.3 and AliSQL — engines for which a build mode is *meaningless*.
  Percona joins them for a different reason: the axis is real for it and the
  recall path just does not say so.

Giving the overlay module a `build_mode` the way pgvector's and valkey's have
one is a small, well-precedented change that would close all three at once. It
is not this task's file.

## 5. The smoke run, and what it has to survive

The user's decision is to pin a SHA and prove it on a smoke run before
committing to a four-day run. Given §0's commit log, that run has to survive
more than "it produced numbers". In rough order of what the log says is most
likely to break:

1. **A restart with a live vector-indexed table.** `af72e2c0fa40` is an
   assertion in `HNSW::insert()` after an *unclean* restart, and
   `5c30ac32877a` is a dictionary assertion after a restart. The orchestrator
   restarts servers between phases. The smoke run must include at least one
   restart followed by an `INSERT` and a search, and one of them should be
   unclean (`docker kill`, not `SHUTDOWN`) because that is the case the fix was
   for.
2. **An index build that hits `innodb_hnsw_max_memory`.** `eebc427aab7b` is an
   assertion when a build aborts on capacity exhaustion. §3 shows the smoke
   profile does not reach the ceiling on its own, so this has to be provoked:
   set `innodb_hnsw_max_memory` low deliberately on a throwaway run and check
   the failure is a clean `ER_CAPACITY_EXCEEDED` that leaves a usable server.
   Doing this *before* the long run is the difference between a known limit and
   a two-day run that dies at hour thirty.
3. **A search concurrent with DDL.** `a86503214416` is an assertion when a
   vector search runs during an in-place index replace. The churn phase's
   `maintain()` (§2.7) does exactly that if any reader is still running, so the
   smoke run's churn phase is the test.
4. Every phase writing records, the plan check passing on both filter columns,
   and the report rendering with seven series.
5. **The run needs a fresh `--run-id` and this engine's recall subtree moved
   aside — and must *not* be given `--force`.** None of it is about the
   server, and all of it is something a green run would hide.

   *The recall path.* `ann_fingerprint` in
   `orchestrator/ann_pass.py` is deliberately engine-invariant and holds no
   source tag, so the re-pin did not move the ann results tree;
   ann-benchmarks skips any configuration whose result file is already on
   disk and reports that as success. Without `--force` the first run at a new
   pin reuses the old binary's recall cells and publishes them under the new
   pin's `engine_version`. `run_engine` in the same module does say so before
   the run — it prints the count of existing result files and that
   already-computed configurations will be skipped, under a comment whose
   first five words are "Warn before the fact, not after" — and what the
   `stale_ann` bucket built in `summarize` in `report/generate.py` adds is
   per-cell attribution in the Validity section afterwards.

   **A fresh run directory does not fix this**, which is the trap worth
   stating plainly: `paths` in `orchestrator/cli.py` puts the ops results in
   the run directory and the ann results in a sibling of it, shared across
   runs by design, so a new run id leaves every recall cell where it was. The
   remedy is to **move this engine's subtree of the ann results tree aside**
   before the run — `_count_results` counts result files under directories
   named for the engine, so moving them makes it return zero and the curve
   gets measured again.

   **Not `--force`**, even though its help text offers exactly that. The same
   flag is the escape hatch on the free-disk refusal in
   `orchestrator/cli.py`, so passing it silently disables the one refusal a
   core-dump campaign most needs — a Percona core is about its peak RSS, and
   that guard exists because a pgvector phase once died two seconds in with
   nowhere to put the cluster and nothing in the run said so. And not a bump
   of `ANN_MEASUREMENT_VERSION`, which would move every pass's tree and
   re-measure every recall point for all seven engines and both resource
   passes — weeks of ingest to fix one engine's cache.

   *A fresh run directory, for the ops path.* This is the same hazard one
   layer down and it has no `--force`. The report's provenance table is
   `_engine_rows` in `report/render.py`, which prints the source tag and the
   first twelve characters of the source commit from the manifest — **one
   value per engine per run directory** — and `set_engine` in
   `orchestrator/manifest.py` replaces that whole entry from the per-engine
   source JSON the build writes, without comparing it to what was there.
   `engine_version` is on every ops record but no reader prints it. So a
   directory resumed across the rebuild pools `9.7.1-1` and `9.7.2-2` numbers
   into one `build_signature` and attributes all of them to the new commit,
   with nothing in the report to tell them apart. So the `smoke` command above
   is run with **neither `--resume` nor `--run-id`**, which is what makes the
   orchestrator start a fresh directory; the failure mode is reaching for
   `--resume` after an interruption and landing the validating run in the
   directory it was meant to replace. A `smoke` result from a stale directory
   would stand in for the one that validates the re-pin.

**What the re-pin leaves unverified in this document, stated here because
nothing else will say it.** The pin table, §4.3's skeleton and every claim this
document makes about *our* repository were re-derived at the new pin. §0's
citation policy already covers the vendor **paths and symbols**: it cites no
line numbers, so nothing there rots to a rebase. What the policy does not
cover is the vendor **commit abbreviations** — every `af72e2c0fa40`-shaped SHA
in §0's pre-merge table, and the four reused in the list above. The fixes they
name are in the pinned series, but under *different* SHAs, because a rebase
gives every commit a new one, so `git show` on one of them against the pinned
tree finds nothing. §0 says so beside the table; this says so beside the four.

## 6. Executable guards

`tests/test_e1_percona_design.py` holds the checks that keep this document and
the new artefacts from drifting apart. It asserts, against the files rather
than against prose: that the dialect claims in §2 match what
`percona/module.py` actually emits; that `docker/percona/` has the same shape as
the other engines' directories; that every one of the pin's four submodules has
a cmake flag and that each flag is really passed; that `init.sql` neither copies
AliSQL's `mysql_native_password` line nor sets a measurement parameter the
orchestrator owns; that the graph-memory fallback is conditional; that the
module's plan marker is not satisfiable by the index name alone and pins
`FORMAT=TRADITIONAL`; that §1's stated default, maximum and scope match the
declaration §1 itself quotes; that **this document makes no line citation at
all**; that **every symbol it cites into this repository resolves, and every
claim it makes about a named symbol's contents is checked by opening that
symbol**; and that the §3 arithmetic reproduces from the MTR formula. It reads
this document as a file where a claim here is the thing under test, so a figure
edited in one place and not the other fails.

What those guards deliberately do **not** do is assert anything about Percona's
behaviour. Nothing in a test file in this repository can establish what a server
does, and a test that asserted "Percona requires a BIGINT UNSIGNED primary key"
would be asserting that this document contains a sentence.

### Why this document stopped citing line numbers

This document used to make about two hundred line citations — roughly a hundred
naming a path and ninety-seven written as a bare `:N` continuation of a path
named earlier in the sentence. **Thirty-one occurrences, twenty-seven of them
distinct, pointed into this repository and were checked. Everything else was
checked by nothing**, because the guard's classifier was a deny-list of vendor
filenames and the vendor tree is not in this checkout.

The checked ones still rotted. Merging three pull requests under this work
invalidated several in a day:

| file | was → is | cause |
|---|---|---|
| `harness/drivers/postgres.py` | 481 → 689 | C4 (#13) added 230 lines; the old line became a sentence of prose |
| `harness/main.py` | 197-199 → 209-211 | T4b (#12), uniform +12 |
| `harness/main.py` | 252 → 264 | same |
| `harness/workloads/churn.py` | 310-311 → 484-485 | T4b moved `maintain()`'s timing |
| `docs/07-planning-a-run.md` | 478 → 481 | unrelated edit |

(The superseded numbers above are written as plain digits, not in code font, on
purpose. A dead pointer typeset like a live one invites a reader to follow it —
and the ratchet, correctly, counts a code-font one as a live claim and fails on
it. It caught its own document's prose within a minute of being written, which
is the behaviour being paid for.)

The first was caught by the one guard that existed — which was the whole
argument for having the rest. **The last two were caught by nothing**, and were
found only by walking every citation in the document against the files. Each
repair then moved other citations in the same commit, which is how line
citations in this document and in `docs/10-rollout.md` put `master` red twice
in two days (#41, #43).

**The unchecked ones were the worse half, and nothing would ever have said so.**
At a re-pin of the vendor SHA, on the order of 164 vendor line references go
stale at once, in a document whose §0 provenance table commits it to having been
read against a specific tree — and not one test in this repository would fail,
because the deny-list said in its own docstring that it checked none of them.
A citation that cannot go red is not a citation; it is a decoration that reads
like one.

So the citations are symbols now, and the guard has three layers instead of one:

- **A ratchet.** The document must contain **zero** line citations of any form.
  That is one assertion in place of twenty-seven parameterised ones and it is
  strictly stronger about the thing that was rotting. The pattern is widened
  past the old one at four points, three of which the old one was blind to: it
  takes any extension, so it sees MTR result and test paths and the one `.yy`
  grammar (32 occurrences); any extensionless file rather than `Dockerfile`
  alone, which was the one special case the old pattern carried; a bare `:N`
  continuation (97 occurrences, the single largest class); and a span carrying
  several ranges. It is proved against a frozen fixture carrying one of each
  form, because a ratchet whose pattern matched nothing would pass while the
  document was full of citations. That is the same vacuity the old
  parser's own note recorded being caught by.
- **A symbol table.** Every claim this document makes about the *contents* of
  something in this repository is a row keyed by `(path, symbol)` and carrying
  the string the document is pointing at. The test opens the file, locates that
  symbol's declaration, and asserts the string is inside it — the same strength
  as the old line check, keyed by a name instead of a number, so it survives an
  edit above it. Completeness runs both ways: a repository code file this
  document cites with no row fails, and a row naming a symbol the document
  stopped citing fails.
- **An existence check over everything else.** Every code-font path resolves to
  a file in this tree, and every code-font symbol appears somewhere in it.
  `tests/test_e15_methodology_roadmap_doc.py`'s classifier is the one used, so
  the three guarded documents agree about what a path is and what a symbol is.

One thing it still cannot do, stated rather than implied: **a vendor symbol
citation is checked by nothing.** It was checked by nothing before, too — what
changed is that it now names a function, a class or an MTR test that a reader
holding the pinned tree can find by searching for it, instead of a line number
that is wrong. That is the whole of the improvement on the vendor side, and it
is worth being precise about, because a guard that *appeared* to cover the
vendor half would be worse than the line numbers it replaced.

### What a prose guard can and cannot do

§1 was attacked four times, each attack conceding the previous defence and
moving one layer down. The progression is the clearest statement of the limit
this series has produced, and it is worth keeping because the temptation at
every step was to add one more substring:

| | attack | status |
|---|---|---|
| **A** | delete the content, keep three tokens | caught — polarity and placement |
| **B** | keep every token, invert the **conclusions** | caught — conclusions pinned to their values |
| **C** | keep tokens **and** correct conclusions, invert the **facts** | caught — prose cross-checked against the quoted declaration |
| **D** | prose that contradicts no quote it carries | **not substring-checkable** |

C is the last layer that is cheaply checkable, and only because **the section
already quotes the primary source for every number in it**. The attack that
got through all 104 guards stated the variable was GLOBAL-only and established
once per connection, and that the default was 100000 with a floor of 40 — three
lines below a verbatim quote of `nullptr, nullptr, 40, 1, 100000, 0)` that says
otherwise. The guards required that quote; nothing compared the prose to it.
Now five checks do, value against role: which number is the default, which is
the maximum, whether the default is miscalled a floor, whether the scope
contradicts `MYSQL_THDVAR_*`, and whether the per-query read is cited.

**Pinning the numbers to the quote is the right place to stop.** What remains
after C is prose that contradicts no quote it carries, and no amount of string
matching reaches that. Reviewing the claim against the source is the only
defence left, and a fifth layer of substrings would buy nothing while adding
more ways to fire on correct writing — which is the other failure mode, below.

**Three of these guards exist because the first versions of them were vacuous,
and the review found all three.** They are worth naming, because the shapes
recur:

- A guard on `_image_march`'s path sliced the function's *text*, and the
  method's **docstring** names the path — so deleting it from the real tuple
  left the guard green. It now reads the tuple out of the AST. Measured: with
  the path removed, the text assertion returned True and the AST assertion
  False.
- The §1 guard required only that the section contain `innodb_hnsw_ef_search`,
  `**Yes.**` and `SESSION`. Replacing the whole section with hedged prose that
  kept those three tokens — no citations, no bounds, neither rider — passed.
  It now asserts **polarity and placement**: the affirmative answer must
  precede the first citation, the negating phrasings must be absent, and the
  section must carry at least four distinct source citations and the quoted
  sysvar declaration. *A guard that asserts prose contains certain strings does not
  guard the prose.*
- The submodule-count guard compared the mapping's length against the **set** of
  number words anywhere in the Dockerfile note. The note's prose contains two,
  three, four and one, so `len(mapping) in {1,2,3,4}` held for any plausible
  mapping; with libkmip deleted the mapping was 3 and the guard passed. It now
  parses one deterministic phrase, `TAKES <n> FLAGS`, from the note's heading.
  That is membership-not-value, which is already on this repo's list of
  measured vacuity shapes.

**And one guard failed the opposite way, which is the hazard a tightening pass
creates.** The guard on §1's first rider demanded the literal `should
implement`, and it fired on rewordings that preserve every fact and both
conclusions — `should **override**` (if anything the more precise verb, since
`effective_ef` is an inherited concrete method with a default in
`EngineDriver`, so overriding is literally what happens) and `**must**
implement` (a strictly stronger instruction). A second ban on `needs no edit`
was subject-blind: it fired on a true sentence about a different subject, and
that phrase recurs in a change-list document by nature.

A guard that fires on correct prose teaches the next author to weaken or delete
it, which is how the vacuity shapes reached this repo in the first place. **A
false positive on a correct document is a slower-acting version of the same
failure.** Both were loosened to the shape that survives rewording: require
the instruction *paired with its value* rather than one verb in one modal
voice, and scope a ban to the sentence carrying its subject. Both fixes were
measured against the rewordings they used to reject and against the attack
they still have to catch.

## 7. What only a live server can settle

Everything above is reading. These are the open questions, in the order the
smoke run will hit them. None is a reason not to proceed; all of them are
reasons the smoke run is not optional.

**Two questions that were on this list have been struck**, because they turned
out to be answerable by reading after all — see the note at the end of this
section. Keeping a settled question on an open list is its own kind of error:
it spends a smoke run's attention on something already known and makes the real
unknowns look less urgent.

1. **Does the image build?** Nothing here has been compiled, but this question
   was **not entirely open** and treating it as one cost a real bug. The
   submodule half was readable, and the answer was **no**: the first revision
   of `docker/percona/Dockerfile` enumerated all four submodules, miscounted
   them as two, and left `extra/libkmip` unmitigated. The top-level
   *CMakeLists.txt*
   descends into it when **neither** *WITH_COMPONENT_KEYRING_KMIP* nor
   *WITHOUT_COMPONENT_KEYRING_KMIP* is defined, and `prepare-sources.sh` ships
   a `git archive`, which omits submodule paths — so cmake configure would have
   failed on a directory that is not in the tarball. Fixed with
   `-DWITHOUT_COMPONENT_KEYRING_KMIP=1`; the Dockerfile now carries one flag per
   submodule and explains why `-DWITH_...=OFF` would not have worked.

   What genuinely remains: whether a 9.7 tree builds clean with Ubuntu 24.04's
   GCC 13 (the branch recently made clang-23 its CI default — `ccb8bbd9f156`),
   and whether the apt list is complete. Out-of-source is required, not
   optional: the same *CMakeLists.txt* defaults *FORCE_INSOURCE_BUILD* OFF.
2. **Can MariaDB Connector/C authenticate over TCP?** This base has no
   `mysql_native_password` — the main MTR suite's
   *mysql_native_password_removed.result*
   records `CREATE USER … IDENTIFIED WITH mysql_native_password BY 'abcd'`
   returning `ERROR HY000: Plugin 'mysql_native_password' is not loaded` — so
   copying `docker/alisql/init.sql`'s line would make the container fail to
   boot, and `docker/percona/init.sql` uses the server default
   `caching_sha2_password` instead. The **ann pass** connects over a unix socket
   (`VBMySQLBase._conn_kwargs`, which passes `unix_socket`), where the cleartext
   exchange is allowed.
   The **ops harness** connects over TCP (`MySQLFamilyDriver.connect`, which
   passes `host=self.spec.host` and a port),
   where `caching_sha2_password` needs TLS or the server's RSA public key.
   `caching_sha2_password_auto_generate_rsa_keys` defaults ON so the key exists;
   whether Connector/C 3.3.x asks for it without an explicit option is untested.
   **This is the first thing to check, because it blocks every ops measurement.**
3. **Does a `bytes` parameter arrive with the binary charset?** §2.1: `cs !=
   &my_charset_bin` is a hard refusal in *field.cc*'s *Field_vector::store*.
   Expected to work,
   since AliSQL's *Field_vector* takes the same path today, but the failure mode
   is every INSERT erroring and it costs one statement to check.
4. **Does `information_schema.COLUMNS.CHARACTER_OCTET_LENGTH` populate for a
   `VECTOR` column?** `VBMySQLBase._dim_from_table` divides it by 4 and returns
   1 if it is NULL, which would make `_warm_cache` warm nothing and
   `_verify_plan` report a false negative. `percona/module.py` prefers
   `VECTOR_DIM()`, which *item_create.cc*'s registry carries beside
   *Item_func_to_vector*, and falls back, so it is covered either way — but
   the shared path is still wrong for the next engine.
5. **Actual graph memory against the §3 floor.** The projections use an MTR
   assertion's floor and its 2× ceiling, and §3 now also records that
   `Innodb_hnsw_memory_used` omits the node map, so the gauge itself is a floor.
   One reading after a 20,000-row load replaces the whole table with a
   measurement, and the `main` profile should not start until it has.
6. **Ingest rate.** Unknown. MHNSW and VIDX manage ~150 and ~55 rows/s on 1M
   rows (`CLAUDE.md`), and Percona's figure decides whether a Percona column in
   the `main` profile is affordable at all. It is also the number
   `orchestrator/estimate.py` needs and must not be guessed.
7. **Is `FORCE INDEX (vi)` accepted on a `VECTOR KEY`?** `Dialect.force_index_hint`
   is set for all three MySQL-family engines and is dead repo-wide today —
   `VBMySQLBase.hint` only uses it when `force_index=True` and no call site
   passes that — so this costs nothing right now. It is listed because Percona
   is the engine where it might not *parse*: the pinned series carries
   PS-11626, folded into PS-11212's optimizer commit, which fixed
   `IGNORE INDEX … FOR ORDER BY (vk)` being ignored for vector (ANN)
   queries — so this is settled code inside the pin rather than its tip. If a
   future caller switches the hint on, a `FORCE INDEX` the parser rejects turns
   every plan check into an exception instead of a plan.
8. **The ops path's `march` field.** §4.8: `_detect_march` in
   `harness/main.py` is a second
   `-march` reader in a different container, which no overlay override can
   reach. Both tuples now carry `/opt/percona/.march`, so this should read the
   image's real value — but *nothing in this repository can prove that*, since
   proving it means a built image with a `/opt/percona/.march` in it. It stays
   on this list for that reason, and because it is a thing to **look at in the
   first smoke run's records**: it will not fail anything, it will just
   quietly say `unknown` if the file is not where the Dockerfile puts it.
### Three items struck from this list, and why

Each was on it in a previous revision. All three are now settled from the
pinned tree, and all three are **reassuring** — which is exactly why they
should not sit on an open-questions list implying doubt. Keeping a settled
question open is not a conservative choice; it spends a smoke run's attention
on something already known and makes the real unknowns look less urgent.

- ***hypergraph_optimizer* cannot be enabled in this build.** Three gates in
  *sys_vars.cc* — the default *optimizer_switch* string,
  *update_optimizer_switch* and *check_optimizer_switch* — all key on CMake
  symbols the Dockerfile does not define, and the last raises
  *ER_HYPERGRAPH_NOT_SUPPORTED_YET* on an attempt. The silent-full-scan risk
  §2.6 described cannot occur here. Moved into §2.6.
- **There is no third `innodb_hnsw_*` variable.** *innobase_system_variables*
  in *ha_innodb.cc* registers exactly two — *MYSQL_SYSVAR(hnsw_ef_search)* and
  *MYSQL_SYSVAR(hnsw_max_memory)* — and the status array in the same file
  exactly one, *hnsw_memory_used*. §2.6's table is complete, established from the
  registration arrays rather than from the truncated tree listing the previous
  revision used. Moved into §2.6.
- **The index replace does not block readers.** It takes a SHARED lock, which
  permits concurrent reads and blocks writes: any ALTER leaving a vector index
  is forced offline by *handler0alter.cc*'s *innobase_vector_exist* gate, and the
  handler returns *HA_ALTER_INPLACE_SHARED_LOCK_AFTER_PREPARE*.
  The previous revision kept this open because the "LOCK=SHARED it is" comment
  it had found was scoped to a table rebuild and the replace is not one — a
  correct refusal to generalise, but the answer was a hundred lines further on
  under a rule that is not about rebuilds. Moved into **§2.7**, not §2.6: it is
  a lock level, and the maintenance table is where the other lock facts live.
