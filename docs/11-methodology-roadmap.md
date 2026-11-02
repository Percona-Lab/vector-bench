# What this instrument cannot see yet, and in what order to fix it

This page is a design document, not a runbook and not a change. Its purpose is
to record, once and in one place, what this framework's measurement methodology
**cannot currently observe**, what it already handles better than a reading of
its gaps suggests, and in what order the rest is worth addressing.

**One item has since been implemented, and it is marked where it sits:** §4.1's
first option was taken, so `Innodb_hnsw_memory_used` is now collected at the
build phase. Everything else on this page is still unbuilt. The blanket "nothing
in it has been implemented" this paragraph used to carry was true when the page
was written and is the kind of sentence that goes stale without anything
failing, so `tests/test_e15_methodology_roadmap_doc.py` now keys a guard on the
tree rather than on this claim.

It exists because four independent read-only studies of this tree were run,
against five hypotheses, and **four of the five did not survive contact with
the code** — the safeguard was already built, or the defect had already been
found and repaired under a measurement-version bump. A reader who starts from
the gap list in §4 and after will misjudge the instrument, so §2 comes first
and is longer than any single tier. The one hypothesis that was confirmed is in
§6, and it is narrower and better evidenced than the guess that found it.

Read `docs/05-methodology.md` for what each figure currently means and
`docs/10-rollout.md` for what the first live run will and will not settle.
This page is about what the figures cannot mean yet.

## 0. How this document cites code, and why it cites no line numbers

**Every citation here names a symbol or a file — a function, a class, a
constant, a config key, a path — and never a line number.** `wait_index_synced`,
`OPS_MEASUREMENT_VERSION`, `harness/workloads/churn_readers.py`: never
*some/file.py:123*.

When this page was written this was a deliberate departure from
`docs/09-adding-percona-server.md` and `docs/10-rollout.md`, which cited
`path:line` and were each guarded by a test that re-opened the cited line and
compared it against what the document claimed was there —
`tests/test_e1_percona_design.py` for the first,
`tests/test_e3_rollout_doc.py` for the second. **That mechanism worked, and it
is how the rot below was found at all.** **Both documents have since been
converted**, so this is no longer a departure from anything: symbol citations
are the repository's only citation form for a document, and those two guards
now check symbols with the same mechanism this page's does. But the rot
kept happening while they did not: line citations in those two documents
went stale twice in two days and put `master` red both times, repaired in
pull requests #41 and #43.
Neither failure was somebody editing a document carelessly. In both cases a
later round of edits *inside the same change* moved lines below the point where
the citations had been derived, and the guard — correctly — fired on the
document rather than on the code.

A line number is invalidated by **any** edit above it, including an edit that
changes nothing the citation is about. A symbol name is invalidated only by a
rename, which is a deliberate act that a reader can follow. This document will
be read long after the lines have moved, and it is about work that has not been
done yet, so it will be read against a tree that has moved a great deal. Symbols
are the only citation form that survives that.

Two consequences that follow from the policy rather than from taste:

* **A citation here is coarser.** `baseline_is_credible` names a function, not
  the eight-row table inside its docstring that this page quotes from. Where
  the exact passage matters, this page quotes it and says which symbol it sits
  in, so the quotation is checkable by search rather than by line arithmetic.
* **Vendor material is named in prose, never in code font.** The pgvector
  *README*'s *Half-Precision Indexing* and *Binary Quantization* sections, the
  AliSQL *vidx* headers, MariaDB's *vector_mhnsw.cc*: a backticked path in this
  document means a path in *this* repository, and a guard enforces that. A
  vendor file is not in this repository on a fresh checkout — `sources/` is
  generated and gitignored — so a backticked path into it would be a citation
  that cannot be checked anywhere except on the machine that happened to fetch
  the tree.

One piece of typography that is not typography: the illustration of the banned
form in the first paragraph is in *italics* rather than code font. The guard
below bans the `path`-colon-line form **as a pattern over code-font spans**,
and the first run of that guard failed on this very page. The alternative was
an allowlist carving out one citation — which would have made this document the
one file in the repository permitted to make the citation it forbids. Italics
cost nothing and leave the ban unconditional.

`tests/test_e15_methodology_roadmap_doc.py` makes the policy real rather than
aspirational. It extracts every code-font span from this page, classifies each
as a path or a symbol, and asserts that every path exists (or is one of the
small set declared deliberately absent) and that every symbol still appears
somewhere in the tree. It also asserts that **this document contains no
`path:line` citation at all**, which is the policy itself as an executable
statement, and it runs the same pattern against a frozen fixture made of
them to prove the detector can see one. That fixture used to be
`docs/10-rollout.md`, which was full of them; it is not any more, and a
non-vacuity fixture must not be something another document can stop being.
Its parsers each carry a floor on what they found, so
a parser that silently matched nothing fails instead of passing over an empty
set — one guard in this series shipped green because it scanned zero files, and
that failure mode is worth more care than the thing being guarded.

**Two limits of this kind of guard, one of which it had before it was probed.**

The first is structural and remains: the guard cannot establish that a symbol
still **means** what this page says it means. A renamed-away symbol fails; a
symbol that kept its name and changed its behaviour does not. That is the
honest limit of a symbol citation, and it is the price paid for one that does
not rot weekly. It is why the passages this page leans on are quoted here
rather than only pointed at.

The second was a real defect, found by planting a failure rather than by
reading, and it is worth recording because it is the characteristic bug of
every guard shaped like this one. **A guard whose index of the tree includes
the guard itself can certify a symbol that exists nowhere but in the guard.**
The index covers `tests/`, deliberately — several things this page has to name
live only there — and the guard is a file in `tests/`, so the names it spells
while describing what it would reject were resolvable *as citations*. The first
run of its planted arm therefore passed the live document and **failed the
plant**: the bad symbol resolved, against the test asserting that it must not.
Left in, that is worse than a guard that checks nothing, because it reads as
coverage and would have accepted a citation into thin air. The file now
excludes itself from its own index and asserts that it does.

A third limit, smaller, and a scope boundary rather than a bug. The guard
recognises a path by its **file extension**, so a citation in directory form —
a trailing-slash path into a tree, however deep — is classified as prose and
escapes both the existence check and the generated-tree check. That is
deliberate and this page depends on it: it names `sources/`, `work/`,
`results/` and `tests/` in code font as the things they are, and a classifier
strict enough to reject those would reject every sentence about them. The
consequence to be aware of is that **a deep directory citation into a generated
tree is exactly as uncheckable as a file citation into one, and is not
flagged** — so the discipline §0 asks for is the author's there, not the
guard's.

The general form is worth carrying to the next guard of this shape: **a check
that builds a corpus and then searches it must say what is in the corpus, and
a check that lives inside its own corpus is checking itself.** The companion
failure arrived from the same probe and is of the same family — one assertion
here passed in isolation and failed under the full suite, because another test
creates `state/` while it runs, so the assertion was about execution order and
not about this document at all. Both were invisible to reading and both
surfaced the moment the guard was asked to fail on purpose.

## 1. Provenance of every claim here

| Kind of claim | How it was established | Marked |
|---|---|---|
| A constant, a function, a config key, a file's presence or absence in this tree | re-read at the commit this page ships on, by this page's author | **checked** |
| A finding about this tree that this page reports but did not re-derive | one of four read-only studies of the codebase | **studied** |
| A consequence of a mechanism — queueing, bias of an estimator, arithmetic on sample counts | argued from the mechanism; **no measurement** | **reasoned** |
| A number quoted from a comment, a config file or a past run | recorded in this repository by an earlier change; not re-takeable here | **supplied** |
| Behaviour of MariaDB, AliSQL, pgvector, Valkey, Percona Search or Percona Server | a vendor tree under `sources/` or a vendor document | **supplied** |
| A claim with no source reachable from this checkout | named, and left open | **unconfirmed** |

Three things this table is load-bearing for:

* **A reasoned claim is not a measurement, and several of the strongest claims
  below are reasoned.** That closed-loop p99 goes optimistic exactly in the
  regime §6 identifies is a queueing argument; what is *checked* is that the
  regime exists in this harness and is not flagged. That a minimum over N
  draws is more optimistic the larger N is, is a fact about estimators; what is
  *checked* is that N is a profile key, that it differs between shipped
  profiles, and that it is absent from the cache key. The distinction is kept
  in every tier below, in the sentence that makes the claim.
* **A supplied number is not re-derivable here and must not read as though it
  were.** "Scalar took the recall and returned none of the 16.5 GB it was
  supposed to save" is a result from a past run, written into a comment in
  `config/resources/tuned.yml`. This page quotes it as a quotation. It does not
  become a measurement by being repeated in a document, and the run that
  produced it is not re-runnable from this tree without the corpus.
* **Most of the findings here were established by reading, not by measuring,
  and two kinds of reading are not the same.** Where a study's finding was
  re-derived for this page it is marked **checked**; where it was not, it is
  marked **studied** and the reader should treat it as a pointer to a symbol
  rather than as a settled fact. §4.1 and §4.4 are the two that were re-derived
  most carefully, because both are claims that this repository says something
  false.

### What is explicitly unconfirmed

**Whether Percona Server's InnoDB HNSW or valkey-search exposes any
compression or quantization option is unconfirmed, and cannot be confirmed from
this checkout.** `sources/` holds the MariaDB 11.8.8 and 12.3.2 trees, the
pgvector v0.8.6 tree and an AliSQL tarball; it holds no Percona Server tree
(the pin is commit `56124b03`, the tip of Percona PR #6278 on base `9.7`) and
no valkey-search tree, because neither has been fetched here. §4.4 is written so that its claim does not depend on those two: it
asserts something about Percona Search that is checked, and withholds the
"nothing else here can" that would depend on all six others.

Also unconfirmed, and listed so that nobody treats them as settled:

* **Whether any results tree on disk mixes two values of the ann `runs` knob**
  (§7). `results/` is gitignored, so an absence of evidence here is not
  evidence of absence — the same caveat `docs/10-rollout.md` makes about runs.
* **The magnitude of mongot's index lag** (§5, §9). It is currently waited out
  by the harness and never published, so the figure §9 wants it for does not
  exist yet in any form, not even as an order of magnitude.
* **Whether the ranking changes under the new axes.** §9's contribution
  depends on it and nothing here establishes it. It is the thing the work is
  for, not a result of the work being planned.

## 2. What the instrument already does, which is most of what was assumed missing

The scoreboard first, because the counting matters and is easy to garble.
**Five** hypotheses, of which four guessed at a missing safeguard:

| Hypothesis | Verdict | What was actually there |
|---|---|---|
| Every client is closed-loop and nothing compensates for it | **confirmed** | closed-loop everywhere, no correction, the open-loop workload specified and deferred (§6) |
| No dispersion is published; everything is a point estimate | **refuted** | a 350-line-plus statistics module, bootstrap intervals, a three-valued tie verdict, chart whiskers, and a *raised exception* rather than a comment (§2.1) |
| Cold start is uncontrolled | **half refuted** | warm-up is a property of the workload, not of the engine; what survives is a documented measurement that does not exist and a measured guard left unwired (§2.4) |
| Quantization is pinned off, so the comparison is of full-fidelity vectors | **refuted** | it is a swept axis in the tuned pass, pinned in the normalized one, and guarded by two test classes (§2.2) |
| The concurrency query pool is small enough to sit in every cache | **refuted** | the pool is the whole test set; the small-pool failure was found and repaired under a measurement-version bump (§2.3) |

The shape of the defects that **did** survive is worth stating before any of
them: they are almost all **a claim in prose that the code does not
implement**, not an aspect of measurement nobody thought about. That is a
different and much cheaper class of problem than "nobody considered this", and
it is also the class a reader of the documentation cannot detect, which is why
§4 exists at all.

### 2.1 Dispersion is handled, and the refusals are in code rather than in comments

The hypothesis was that the framework publishes point estimates with no
interval. It does not. **Checked:** `harness/metrics/stats.py` is over 350
lines of it — a size rather than an exact line count, deliberately, for the
reason §0 gives about figures that any edit above them invalidates; the guard
recomputes the floor —

* `bootstrap_ci` with a fixed seed, so a report regenerated from the same
  records is byte-identical; `mean_interval` and
  `qps_interval_from_latencies`, the latter inverting the latency bounds and
  therefore swapping them, because the slow end of a latency interval is the
  low end of a throughput interval;
* `Interval` carrying `n` beside the bounds, so that a zero-width interval's
  two causes — a real measurement of no spread, and too few samples for a
  spread to mean anything — stay distinguishable;
* `tie_verdict` returning **three** values and not two, with `TIE_UNKNOWN` for
  "nobody could look". A two-valued overlap test has to place the unmeasured
  cases somewhere, and both placements publish something false: under either
  the closed or the strict overlap form, an unusable interval against a real
  one reads as *these two engines are ordered*;
* whiskers on the charts, via `recall_whiskers` and `draw_recall_whiskers` in
  `report/charts.py`, with `WHISKER_TITLE_NOTE` added to a title **only** when
  a series actually drew any.

And the part that makes this a different kind of safeguard from a documented
intention: `ci_fields` takes the phase and **raises** `ValueError` for any
phase in `NON_INDEPENDENT_PHASES`, which today is the concurrency phase. A
future author who reaches for interval keys on the concurrency sweep gets an
exception naming the reason — samples inside one window are not independent, so
resampling them understates the spread, and the window-to-window spread from
`concurrency_repeats` is the measurement of uncertainty for that phase. A
comment saying the same thing lasts exactly as long as the next person who
reads it.

**The real gap, and it is in the other measurement path.** The ann path
publishes a best-of-N throughput, which has no interval and is not a point
estimate of the kind the above is about. §7 is that gap.

A second, smaller one, **reasoned**: the refusal above is correct and a
reviewer will want it argued rather than found. `docs/05-methodology.md`
records the reason in prose and `harness/metrics/stats.py` raises on it, but
neither states the alternative — a window-level bootstrap over the repeats,
which is what `concurrency_repeats` already collects — or says why it is not
taken. §9 depends on that argument being made explicitly.

### 2.2 Quantization is a swept axis, not something pinned off

The hypothesis was that quantization is disabled everywhere so that the
comparison is of full-fidelity vectors. **Checked:** it is an axis.
`config/resources/tuned.yml` sweeps `mongodb_quantization` over two values,
scalar first because the ops path builds one index and takes the list's head;
`config/engines/mongodb.yml` declares `quantization_tunable`,
`quantization_normalized` and `quantization_tuned`; the normalized pass pins it
off, so that no engine is handed an axis the others lack. `TestMongoQuantizationFollowsPrecedent`
and `TestQuantizationReachesBothPaths` in `tests/test_config.py` hold both ends
of that, the second because of a real incident: the recall path once got the
vendor-recommended setting and the ops path did not, and one 44-hour run
reported an unquantized index's build cost beside a quantized index's recall
curve as though they were one configuration.

Why both values are swept rather than just the vendor's recommendation is
**supplied** from the comment in `config/resources/tuned.yml`, and is a good
piece of methodology worth not losing: the first complete curve stopped at
recall 0.892 and did not move between two query-effort settings, and *an index
that will not reach 0.90 at any query effort is a different claim from an
engine that will not*. Without the unquantized curve beside it there is no way
to say which of the two was measured.

**The real gaps are two, and both are §4 items:** quantization is in the
cross-check key and in no structured field (§4.5), and it is absent from the
list of known asymmetries where `ef_construction` is present (§4.6).

### 2.3 The concurrency query pool is the whole test set, and the small-pool failure is already repaired

The hypothesis was that the concurrency workload draws from a pool of a few
hundred queries, small enough to sit entirely in every engine's graph cache.
**Checked:** it draws from the whole test set. `_query_pool` in
`harness/workloads/concurrency.py` treats `WHOLE_TEST_SET` as a sentinel
meaning *every query the dataset ships*, and both `config/profiles/main.yml`
and `config/profiles/full.yml` carry a comment saying `concurrency_queries` is
deliberately left unset for exactly that reason.

It was not always so, and the repair is the part worth knowing. **Supplied**
from the version notes in `harness/metrics/records.py`: the pool used to be
`max_queries`, a key that exists to bound the filtered and churn workloads'
cost and that `config/profiles/main.yml` holds at 500 — so 32 clients were
getting about fifteen queries each for thirty seconds, and every graph cache in
the comparison was being measured against a hot set small enough to hold
entirely, *which is the one thing the concurrency chart exists to tell apart*.
That earned measurement version 3, and 3 is deliberately absent from the set of
versions that denote the same measurement.

Note what did **not** earn the bump, because §3 turns on it: version 3 also
added recall to concurrency records, and the notes say in as many words that a
field which was `None` and is now populated would not on its own have earned
one. The pool is what earned it.

**The real gap is not pool size but pool shape**, and it is a limitation to
name rather than a bug to fix. §8.

### 2.4 Warm-up is engine-invariant; the cold-start hypothesis is half refuted

The hypothesis was that cold start is uncontrolled. **Studied, and partly
checked**, and the distinction matters here more than elsewhere because the
claim is a negative one. The finding from the studies is that every ops
workload warms up before it measures and that the warm-up count is a property
of the workload rather than of the engine, so no engine gets a different number
of warmed queries than another at the same point. What was **checked** for this
page is the half that leaves a trace: three warm-up counts reach a record —
`warmup_queries` from the calibration workload,
`warmup_queries_per_client` from concurrency, `reader_warmup_queries` from the
churn reader pool — and none of the three is engine-conditional. *No engine
anywhere gets a different count* is a statement about every call site and is
not re-derived here.

Two things survive, and they are the two §4 items of the "claim in prose" shape:
a documented measurement that was collected nowhere until §4.1 was built, and a
measured,
tested, deliberately unwired guard against the one quantified per-engine cold
bias this harness has (§4.3). The churn workload's own warm-up count is the
third, and the smallest (§4.7).

## 3. The version-bump rule, which governs almost every item below

Nearly every item in §4 through §8 is scoped by one rule, so it is stated once
here rather than in each.

**`OPS_MEASUREMENT_VERSION` is what makes the report refuse to mix, and the
refusal is total.** When a records set holds two distinct values,
`refuse_mixed_measurement_versions` raises `MixedMeasurementVersions`,
`report/generate.py` prints the message and returns 2, and **there is no report
at all** — not a report with a caveat, not a chart with two series, nothing.
Resuming a long run under newer code therefore costs the run's reportability,
which is why `set_config` warns about it at the start of a resume rather than
at report time, after the expensive half. The remedy is one report per version,
and splitting the record files by `extra.ops_measurement_version` afterwards is
legitimate: the older records are not wrong, only incomparable.

The rule that decides whether a change owes a bump, as the version notes in
`harness/metrics/records.py` state it:

* **A field that was absent or `None` becoming populated does not earn a
  bump.** It is new information. Nothing a reader of the previous version
  already trusted moves. Version 3's addition of recall to concurrency records
  is the worked example, and it is stated there explicitly.
* **A published value changing does earn one** — including when the change is
  a correction, and including when only *some* published values move. Version
  3's pool change, version 4's move from a declared `ef_search` to a
  calibrated one, version 5's second filter column (which moved
  `ingest_rows_per_s`, `table_bytes` and `index_bytes`, none of which the
  change was for): all three are value changes.
* **A bump is per run, not per phase.** A record names the harness that wrote
  it. So a bump invalidates comparison across every ops phase, including phases
  the change did not touch, and the notes accept that cost explicitly rather
  than splitting the constant four ways and creating four ways to forget one.
* **An unnecessary refusal is the safe direction.** Its cost is one report per
  version; a missing refusal's cost is a published chart that is wrong.

Everything in §4 is therefore additive and owes no bump: a new `extra` key, a
new validity check, a reworded sentence. §5 and §6 are additive *by design* —
a new phase constant with its own `extra` keys, defaulted off, with the
existing closed-loop records untouched — and §6 names the one way to get them
wrong: writing an open-loop response time into the existing `latency_p99_ms`
column on existing concurrency records would be a value change, and would cost
the report entirely.

## 4. Tier 1 — what this repository asserts and does not do

Ordered by severity. Every item is a thing this repo says and does not do, and
every fix is additive.

### 4.1 `Innodb_hnsw_memory_used` was documented as recorded and collected nowhere — BUILT

**BUILT — and the record of what was wrong is kept because it is the most
embarrassing item on this page and the reason the collector exists.** For as
long as this section described the present tense, the audit trail for a
deliberate fairness decision did not exist. It does now: the first of the two
options below was taken.

`CLAUDE.md` says `Innodb_hnsw_memory_used` is recorded per phase. So does
`README.md`. So does `config/engines/percona.yml`, which calls it "the audit
trail". `docs/09-adding-percona-server.md` returns to it repeatedly — no count
is given here, because a count of mentions is the kind of figure this page's
own §0 is about — and one of those mentions is the justification for setting
`innodb_hnsw_max_memory` to 0 at all. The argument for leaving Percona's hard
allocation ceiling unlimited, so that the container's memory limit bounds it
like every other engine's rather than comparing a wall against the graph caches
`mhnsw_max_cache_size` and `vidx_hnsw_cache_size` are, ends in *and we record
the gauge instead*.

**There is a collection site now, and naming it is the whole of the fix.**
`EngineDriver.status_gauges` declares which status variables a driver is worth
asking for and is `()` for six of the seven engines; `EngineDriver.status_counters`
is the hook, returning `{}` by default; `MySQLFamilyDriver.status_counters`
implements it as one `SHOW GLOBAL STATUS LIKE` per name, with
`_matching_status_row` rejecting a row the widened `LIKE` pattern dragged in;
`PerconaDriver.status_gauges` is the only declaration in the set; and
`harness/workloads/build.py`'s `_server_status` merges the reading into the
`index_build` record as `extra.server_status`, a `{name: int}` dict written only
when a gauge was actually read.

**What the search used to return, kept because the search is how the gap was
found.** Grepping every `.py` in the tree for that status variable, in any
casing, returned **no hit at all** — not one, none — and a case-sensitive
search for the capitalised status-variable prefix *Innodb_* anywhere under
`harness/`, `report/` or `orchestrator/` returned nothing, which is what
established that the gauge was not merely being read under another name. It is
in italics because it is a fragment of a name and not a name, which this page's
§0 rule and its guard both insist on. **Both searches now resolve**, to
`harness/drivers/base.py`, `harness/drivers/mysql_family.py` and
`harness/workloads/build.py`. The lowercase `innodb_hnsw_ef_search` and
`innodb_buffer_pool_size` that also appear are server variables this framework
*sets*, which is the opposite direction and always was.

**The fix was a choice of two, and the first was taken.** Collect it — a
`SHOW GLOBAL STATUS` read in `harness/drivers/mysql_family.py`, stamped into
`extra`, which is additive and owes no bump — or strike the claim from all four
places. **Collecting it is what shipped, at the build phase only**: the
concurrency and churn phases are still uncollected, and the filtered phase is
the same hook at a different call site and is not built either, so this section
is closed for `index_build` and open everywhere else. The verdict that it owes
no measurement-version bump held on inspection — every identity component reads
`extra` by literal key name and none enumerates a record's key set. Collecting
it carries a mandatory rider that
`docs/09-adding-percona-server.md` already supplies and that must travel with
every figure quoted from it: the gauge **omits the node map**, at a supplied
30–50 bytes per node, so it is a floor on graph memory and not a measurement of
it. A reader reconciling it against a container's resident set will find it
short, and the document that publishes it has to say so first.

### 4.2 The churn reader ratio divides two incommensurable windows

**Checked.** `PoolSpec` in `harness/workloads/pool_window.py` carries
`clients`, `queries`, `k`, `ef_search` and `unit` — and **no duration and no
sample count.** The baseline window is a fixed `READER_BASELINE_WINDOW_S`, 30
seconds, set in `harness/workloads/churn_readers.py` and pinned against the
estimator's `CHURN_READER_BASELINE_WINDOW_S` by a test. The during-writes
window is as long as the re-insert takes, which is a property of the engine and
the churn fraction.

So the headline ratio that `window_ratio` computes — a p99 under writes over a
baseline p99 — divides two nearest-rank percentiles taken over sample counts
that are **unequal by an unknown factor in an unknown direction**. Both halves
run the same reader pool at whatever rate the server allows, and both
accumulate one sample per completed query, so the longer window carries the
more samples: on a slow engine or a large churn fraction the re-insert outlasts
30 seconds and the *during-writes* window is the larger, and on a fast one it
is the smaller. Neither the factor nor the direction is recorded, and an
earlier draft of this page asserted a direction — a few hundred samples under
writes against hundreds of thousands at baseline — which nothing in the code
supports and which this paragraph used to state as though it were measured.

Why unequal counts matter at all is **reasoned**, and it is arithmetic:
`percentile` is nearest rank, `ceil(q × n)` into an ascending sort, so a p99
over 300 samples is the **fourth** slowest of the 300 and a p99 over 300,000 is
the **3,001st** slowest. Those two estimators have different bias and wildly
different variance, so a ratio of them is not a ratio of one quantity measured
twice — and **the guard whose entire stated purpose is refusing ratios of
different things does not check the axis that differs.**

`PoolSpec`'s own docstring lists two ratios of exactly that shape already
published wrong in this repository — the filter-strategy ratio, which divided a
narrow point from one ops unit by a wide point from another, and the churn
retention chart, which paired records from two different maintenance states.
This is the third, and it is the one the tuple was built for.

**The fix is one field.** `PoolSpec` is a `NamedTuple` whose equality *is* the
safety property, so adding a duration (or a sample count) to it makes
`window_ratio` refuse on its own, with no new logic. It is additive: the
refusal withholds a ratio that should never have been published, which is a
field going absent rather than a published value changing, and no comparable
figure moves.

**Do this one first.** §5 builds a new ratio on the same tuple and would
inherit the same bug.

### 4.3 `baseline_is_credible` is measured, tested, and deliberately unwired

**Checked.** `baseline_is_credible` in `harness/metrics/scaling.py` exists,
carries its own calibration, is exercised by
`tests/test_skewed_latency_is_not_serialization.py`, and its docstring opens by
saying that **nothing calls it yet, deliberately**, naming
`harness/workloads/concurrency.py` as the caller it is for and noting that
wiring it up needs no new recorded field, because that workload already
computes the efficiency and already holds both medians.

What it defends is the one quantified per-engine cold bias in this harness: the
concurrency warm-up is **per client**, so the one-client baseline gets the
fewest warmed queries of any point, reaches cache steady state slowest in wall
clock, and is measured first. A baseline depressed by a factor of *k* inflates
every efficiency above it by about *k*, and `CREDIBLE_EFFICIENCY_CEILING`
cannot see it, because a 2× depression turns an honest 0.25 into 0.50, which is
under every ceiling and over the floor.

The docstring's own measured table — **supplied**, taken against a synthetic
driver through the real client pool with the four-client window held fixed and
only the baseline's cost scaled — is the whole argument:

> a genuinely serialized four-client pool at honest efficiency 0.248 publishes
> **0.579 with status `ok`** once its baseline is 2.5× cold.

The threshold `CREDIBLE_BASELINE_MEDIAN_RATIO` is the midpoint of a measured
gap in the same style as the other two constants in that module: legitimate
readings ran 0.97–1.02 across every regime and client count the test rebuilds,
against a lowest corrupt reading of 1.889.

**The fix is to record it in `extra` from the concurrency workload.** And a
constraint on the fix that is easy to get wrong: it must **not** gate
`scaling_efficiency` or `scaling_status`. Letting it decide either changes
published values and owes a bump under §3 — and the §3 bump is total, so the
cheap additive version of this fix and the expensive one differ by the entire
reportability of every archived run.

### 4.4 "The only engine here that compresses indexed vectors" is not supportable as written

**Checked, against the vendor trees that are present.** The sentence appears in
`config/resources/tuned.yml`, in `harness/drivers/base.py` beside
`IndexSpec`'s `quantization` field, in `harness/main.py` beside the
`--quantization` flag, and in report text in `report/render.py`. In the report
it reaches a reader.

Where "checked" was checked, precisely, because it is not this worktree: the
four vendor reads below were done in the `sources/` tree of the clone this work
was driven from. `sources/` is generated and gitignored, so it is in **no**
fresh checkout and not in the worktree this page was written in. That is the
reason each is cited by vendor section or variable name rather than by path: a
path citation into it would be checkable on exactly one machine. It is also the
reason the vendor trees were read without being altered — AliSQL is held there
as an unextracted tarball, and the two files needed were streamed out of it
rather than unpacked, because the vendor repositories are read-only inputs.

**pgvector v0.8.6**, the pinned tag, documents half-precision vectors,
half-precision indexing, binary vectors and binary quantization, each with its
own section in its *README*, and ships the corresponding types and HNSW
operator classes. So pgvector plainly can compress an indexed vector.

**The two MariaDB trees are the other end of the same check and they support
the claim**: in both 11.8.8 and 12.3.2 the MHNSW plugin registers four system
variables — a cache size, a default graph degree, a default distance and an
`ef_search` — and nothing resembling a stored-representation option.

**AliSQL is the interesting one, because a naive search finds a false
positive.** Its *vidx* code copies an InnoDB table option named *compress* from
the base table onto the hidden index table it creates, alongside the encryption
type and the key block size. That is InnoDB **page** compression, not a change
of vector representation, and AliSQL's five *vidx* system variables are a
feature switch, a default distance, a default graph degree, an `ef_search` and
a cache size. So AliSQL supports the claim too — but only once that hit is read
rather than counted, and a reader who greps for the word and stops has the
opposite answer.

**What is left unsupportable is the scope, not the subject.** Percona Server's
InnoDB HNSW and valkey-search are **unconfirmed**: neither tree is in
`sources/` here. So "nothing else here can" is a claim about six engines of
which four are checked and two cannot be.

**The fix is a reword, and the honest reword is narrower than a hedge.** The
distinction that survives every check above is not *who can compress* but *what
kind of knob it is*: Percona Search takes quantization as a field in the index
definition, which is why this framework can sweep it as an axis. pgvector's
routes are a different column type or an expression index — a schema decision
taken before any index exists, which is why this framework does not sweep it
and pins a full-fidelity column instead, as `harness/drivers/postgres.py`
shows. That is a defensible reason to pin pgvector where it is pinned. *Nothing
else here can* is not that reason, and is the part that must go.

### 4.5 Quantization is in the cross-check key and in no structured field

**Checked, and the code says it about itself.** `CROSSCHECK_KEY_FIELDS` in
`report/crosscheck.py` includes `quantization`, and the comment above it states
that today the value is `None` on both sides for every engine in every shipped
profile — and that saying so is the point. `report/render.py` carries the
reader-facing version of the same sentence in the report's cross-check section.

The two halves of why: `harness/main.py` takes `--quantization` into
`IndexSpec`, so the index really is built with it, but `record_defaults` does
not put it on a record; and the mongodb ann module under
`overlay/ann-benchmarks/` **does** write it into the result file's attributes,
but the ann loader in `report/loaders.py` does not copy it onto the record.
Both paths know the value. Neither puts it where the other can see it.

**The fix is two lines and owes no bump.** Declare it as run-level extra from
`harness/main.py`, where `declare_run_extra` and `stamp_run_extra` already
exist for exactly this and already drop a `None` — so no engine without the
knob gains a null field — and copy the attribute onto the record in the ann
loader. Both the matched and the caveat branches of the cross-check section
already exist and both are already tested, so this flips which branch renders
and adds no untested path.

### 4.6 Quantization is absent from the list of known asymmetries

**Checked, and the three lists do not agree with each other, which is a
finding of its own.** There are three statements of the known asymmetries: the
structural-asymmetries bullet in `CLAUDE.md`, the known-asymmetries section of
`docs/05-methodology.md`, and `_known_asymmetries` in `report/render.py`, which
is the one a reader of a report sees. All three name `ef_construction` and
AliSQL's InnoDB-and-READ-COMMITTED requirement. **None of the three names
quantization.**

`build_mode` is named by two of the three — `CLAUDE.md` and
`report/render.py`, the latter as *pgvector bulk-builds after load; MHNSW and
VIDX build incrementally* — and **not** by `docs/05-methodology.md`, whose
table has no bulk-versus-incremental row at all. That list is the stalest of
the three in general: it still says `ef_construction` is exposed by pgvector
alone, where `report/render.py` builds its row from the engines that actually
ran and knows valkey-search exposes it too, and it carries no row for Percona
Server or Percona Search. Bringing it level is not this item, but anyone
editing one of the three should know there are three.

It is the larger of the two asymmetries. `ef_construction` is a tuning axis: an engine that
has it can trade build time for graph quality, and an engine that lacks it
cannot. Quantization changes **what is stored**, which is a different kind of
difference, and the report's own fairness argument is about comparing engines
on a corpus rather than on representations of it.

And the measured result that belongs in that row currently lives **only** in a
config comment — **supplied** from `config/resources/tuned.yml`: scalar
quantization took the recall and returned none of the 16.5 GB it was supposed
to save. A row in each of the three places, carrying that figure and its
provenance, is the whole fix.

### 4.7 Three smaller ones

* **An even repeat count takes the upper middle, and for two windows that is
  the maximum.** The concurrency workload sorts its windows by throughput,
  ascending, and takes the element at `len(samples) // 2` — which for two
  windows is index 1, the faster of the two, and for four is the
  second-fastest of four. A maximum is not a median, and a best-of-two is the
  same biased estimator §7 is about, arrived at by accident instead of by
  design. **Checked as latent, not live:** `concurrency_repeats` defaults to
  1, and the three
  profiles that set it — `config/profiles/main.yml`,
  `config/profiles/full.yml`, `config/profiles/tuned-complete.yml` — all set 3.
  So every shipped value is odd and no published number is affected today. Fix
  by taking the lower middle, or by refusing an even count in validation; the
  second is better, because "the median of an even number of windows" has no
  single right answer and a silent choice between two is what this is.
* **Churn's warm-up count never reaches a record.** The churn measurement
  helper in `harness/workloads/churn.py` warms up with a default count, and it
  is the one ops warm-up with no counterpart to `warmup_queries`,
  `warmup_queries_per_client` or `reader_warmup_queries` on the record. Purely
  additive.
* **`docs/05-methodology.md` says "the first 1,000 queries"** where the value
  is `max_queries`, which is 500 in `config/profiles/main.yml` and 2000 in
  `config/profiles/full.yml` — so the sentence is true of exactly one shipped
  profile, `config/profiles/quick.yml`, and of no profile anyone runs. It sits
  in the section that explains how the two measurement paths are cross-checked,
  which is the section a sceptical reader goes to first.

## 5. Tier 2 — the write/freshness workload

This is the external reviewer's request, and most of it is already built.

**Checked:** reads-during-writes exists. `harness/workloads/churn_readers.py`
runs a process-based reader pool against the server while a re-insert
proceeds, reports a p99 under writes over a baseline p99, and records the
achieved write rate as `reinsert_rows_per_s` in `extra`.

Two things are missing, and the second is the interesting one.

**A rate-controlled, bursty arrival process.** The reader pool is closed-loop
like everything else (§6), and the writer runs as fast as it can. Neither is a
model of a production write stream. Target rate and achieved rate must both be
recorded, beside the arrival generator's name and its parameters: *N
updates/sec* is a request, not a fact, and a workload that records only the
request cannot be falsified.

**Freshness sampled during load rather than waited out — and the barrier is
the obstacle.** `wait_index_synced` exists precisely to erase index lag before
querying, so that recall is fair across engines that index synchronously and
engines that do not. A freshness workload that calls it before querying
therefore measures nothing: the quantity it wants is the thing the barrier was
built to remove. The design has to **sample rather than barrier** — probe rows
with known vectors, polled while the read pool runs, recorded as a
distribution (p50, p95, max) and never as a mean, because the mean of a lag
distribution is the one summary that hides the behaviour being measured.

**Recall under writes is scoped out, and that is a measurement decision rather
than a shortcut.** Ground truth is computed against a fixed corpus —
`harness/datasets.py` recomputes it for filtered predicates for the same
reason — so a recall taken while rows are being inserted and deleted is a lower
bound on the engine's recall by an amount nothing in the run knows. The
workload should deliver throughput, latency and freshness, and make **no**
recall claim. An engine that defers indexing would otherwise be rewarded twice:
once in throughput and once in a recall number measured against a corpus it has
not finished absorbing.

**Fix §4.2 first.** The freshness and under-load ratios are `PoolSpec` ratios.

## 6. Tier 3 — open-loop latency, the one confirmed structural blindness

This is the hypothesis that was confirmed, and it is the only one.

**Checked:** every client in this harness is closed-loop. The per-client loop
in `harness/workloads/client_process.py` is a bare `while` over a halt flag
with exactly one request outstanding — issue, wait for the answer, record both
instants, issue the next — and the parent in `harness/workloads/concurrency.py`
sleeps for the window's duration while the children spin. There is no scheduled
issuing, no intended start time, and no correction for a request that should
have been sent while a client was blocked. `harness/workloads/rate.py`, the
open-loop latency workload, was specified and deferred; `docs/10-rollout.md`
records it as optional and absent, and the test suite asserts its absence
rather than leaving it ambiguous.

**What makes this blindness demonstrable rather than arguable is a table this
repository already carries.** `harness/metrics/scaling.py`, in the comment
calibrating `SERVED_CONCURRENCY_FLOOR`, records a saturated server scoring a
served-to-clients fraction of 0.989 at a scaling efficiency of 0.277 — and
**not** being flagged, correctly, because a real queue raises the whole
distribution and that pool's median rose with the client count. That is the
honest reading of that row. It is also, **reasoned**, precisely the regime in
which a closed-loop p99 goes optimistic: every client that is waiting is a
client not issuing, so the requests that would have measured the queue are the
requests the queue prevented.

One docs claim to revisit while this is being built, and it is **reasoned**:
`docs/05-methodology.md` says the signature of the case `client_bound` misses
is flat QPS with flat latency, "which no saturating engine produces, because an
engine at its limit makes queries wait and that shows in p95 and p99". True in
steady-state saturation. False for a transient stall, which is the
coordinated-omission case: sixteen clients each pausing two seconds contribute
at most sixteen late samples, and against the order of hundreds of thousands of
completions in a thirty-second window, sixteen samples cannot reach p99. The
sentence is right about the regime it was written for and wrong as a general
claim, and the general claim is the one a reader will take from it.

### The design, and the trap that governs it

* **Additive, therefore no bump.** A new phase constant, its own `extra` keys,
  off by default, every existing closed-loop record untouched. Writing an
  open-loop response time into `latency_p99_ms` on existing concurrency
  records is the one move that is not additive, and under §3 it costs the
  entire report rather than earning a caveat.
* **`Sample` needs a fourth field, and both series must be published.**
  `Sample` in `harness/metrics/latency.py` carries a duration and the two
  instants bracketing it; an open loop also needs the instant at which the
  request was *due*. Then publish both `finished - started`, the service time,
  and `finished - intended`, the response time. Only the second alone loses the
  ability to separate *the engine got slow* from *the harness fell behind*, and
  those two have the same shape in one series and opposite remedies.
* **Record schedule lag, backlog depth, and target against achieved arrival
  rate.** This is what makes the mode falsifiable, and §5 wants the same three
  for the same reason.
* **`PoolSpec` must carry the target rate**, or `window_ratio` divides a
  500 q/s window by a 5,000 q/s one. Same field, same reason as §4.2, which is
  why §4.2 is first in the sequence.
* **The capacity trap, which is the reason this is not a small change.**
  **Reasoned**, and it is the constraint the whole design has to be built
  around: one request outstanding per process caps each process's issue rate
  at one over its service time, and so caps the pool's achievable arrival rate
  at the client count over the service time. Ask for more than that and an
  open loop **silently degenerates into a closed one** — which is the worst
  available outcome, a mode that reports open-loop numbers that are in fact
  closed-loop numbers, under a label asserting they are not. Two routes out, and both have
  a cost. Over-provision processes — which costs RAM and cpuset, and collides
  with `CLIENT_BOUND_UTIL`, the flag that exists to catch a client container
  short of CPU. Or issue concurrently inside each child — which re-imports the
  interpreter-lock problem that version 6 was a bump to escape, and **must not
  be taken for the MySQL-family drivers** for that reason. Whichever route is
  chosen, the mode needs a guard that **refuses to publish** when the schedule
  was not met, in the style of `SERVED_CONCURRENCY_FLOOR` rather than in the
  style of a warning.

## 7. Tier 4 — the ann path's headline estimator is a minimum over N runs

**Checked**, on both sides of the repository boundary — and where the far side
was checked, since it is not in this tree. `report/loaders.py` reads the best
search time out of each result file's attributes and publishes its reciprocal
as the QPS, so the headline throughput on this framework's recall chart is one
over a *minimum*. That the minimum is taken over the N runs was read in
ann-benchmarks' own runner, in the **read-only vendor input** `CLAUDE.md` names
as `$VB_REPO_ANNB` — the same kind of source, and the same kind of caveat, as
§4.4's vendor reads: the runner initialises a best search time to infinity,
loops `run_count` times, and keeps `min(best_search_time, search_time)` on each
pass, writing both that minimum and `run_count` into the result attributes.

Two things make this reproducible rather than one machine's reading.
`scripts/prepare-harness.sh` clones the ann-benchmarks working copy **from**
that read-only input, so the tree under `work/` is a clone of it rather than an
independent download; and `overlay/ann-benchmarks/` contains nothing but
per-engine `algorithms/<engine>/` directories — no runner and no *run.py* — so
the overlay cannot have changed the file the claim is about. (Both of those
vendor filenames are in italics, for the reason §0 gives: the guard rejected
them in code font on the first run of this very correction, because neither is
a path in this repository. The rule caught its own author twice.) The runner is
therefore named here rather than cited as a path because it is not in *this*
repository, not because it was unreadable.

A minimum over N draws from a noisy positive variable is a biased estimator of
the quantity, and — **reasoned**, as a fact about estimators rather than a
measurement here — it is *more* optimistic the larger N is. N is the `runs` key:
3 in `config/profiles/full.yml`, 2 in `config/profiles/main.yml`,
`config/profiles/tuned-complete.yml` and `config/profiles/quick.yml`, 1 in
the rest — and every profile marked `publishable: true` must carry at least 2. So the degree of optimism differs
per profile while the records look identical. Within one run it is fair
engine-to-engine, because every engine gets the same N. **Across profiles it is
not comparable**, and nothing in a record or a chart says so. The flag is
passed through by `orchestrator/ann_pass.py`, which reads the profile key and
hands it to ann-benchmarks unchanged.

Three things compound it:

* **QPS and latency come from different runs** — **checked**, in the same
  runner and by the same route as above. The per-query results list is
  **reassigned inside** the `run_count` loop and returned after it, so the
  times dataset the published percentiles derive from holds the **last**
  iteration's latencies while the throughput beside them is the **minimum**
  over all N. A chart therefore carries a throughput from one iteration beside
  percentiles from another, and the ann loader in `report/loaders.py` takes
  both out of the same file without reconciling them. The ops path refuses
  exactly
  this mismatch, deliberately and with a comment explaining it: the recall and
  the status of its median window ride inside the same sample tuple
  specifically so that "the recall of the median window" is true by
  construction and cannot drift into being some other window's.
* **`runs` is not in `ann_fingerprint`.** **Checked:** the fingerprint in
  `orchestrator/ann_pass.py` hashes
  `ANN_MEASUREMENT_VERSION`, the server memory, the pass signature, the build
  thread count and the server CPU count. So a `runs: 1` results tree is
  silently reused under a `runs: 3` manifest, and two different estimators are
  drawn as one curve. That is the same class of defect the fingerprint exists
  to prevent — it was built because a 16 GB recall curve came back
  byte-identical under a 64 GB manifest.
* **The detector is already in every record and is read by nothing.**
  `run_count` is copied onto each ann record by `report/loaders.py` and no
  other file in the tree mentions it.

**The cheap fix and the expensive one are very far apart.** Cheap, and no bump:
compare the manifest's `runs` against each record's `run_count` in
`report/generate.py` and report any disagreement in the report's validity
section, and state best-of-N on the recall/QPS chart. Expensive and correct:
add `runs` to `ann_fingerprint`, which **discards every cached recall curve**
in every tree keyed by the old fingerprint. Do the cheap one unless a mixed
tree actually turns up — and note that whether one exists is **unconfirmed**,
because `results/` is gitignored.

## 8. Tier 5 — query distribution, a limitation to name rather than a bug to fix

**Checked:** the concurrency pool is the whole test set, partitioned across
clients by stride and then **cycled in order**, and nothing in the tree
contains the word *zipf* or any other skew. **Supplied**, and worth separating
because it is a property of the datasets rather than of this framework: that
whole test set is ten thousand queries for most of the corpora here and one
thousand for gist-960-euclidean. Those are ann-benchmarks' dataset sizes and
nothing in this repository records them, so a guard here cannot check them.

A fixed pool cycled round-robin is the most cache-favourable non-trivial
distribution there is: every query is drawn exactly as often as every other,
and the hot set is the whole set. Production traffic is not like that, and the
direction of the error is predictable (**reasoned**): a skewed distribution
concentrates the working set, which favours a small cache, so the engines whose
graph caches this framework sets from the resource pass —
`mhnsw_max_cache_size` and `vidx_hnsw_cache_size` — are the ones whose ranking
a skew is most likely to move.

And the list in `docs/05-methodology.md` of what would change the conclusions
— different CPU, dataset scale, version drift, client-side overhead — **does
not name query distribution.** That list is the document's statement of its own
limits, and this is a limit missing from it.

**The cheap step is documentation and owes no bump:** name it in that list, and
in the report, and fix the query-count sentence in §4.7 while in the
neighbourhood.

**The measurement is a new phase, not a change to the existing pool**, and the
reason is in `_query_pool`'s docstring. That function must return a **prefix**
of the test set, because the concurrency workload scores results by indexing
ground truth with positions into the pool, and those positions agree with
positions into the full test set only while the pool is a prefix. Narrow or
reorder the pool any other way — a random sample, a stratified draw, a Zipf
draw — and **recall is silently scored against the wrong rows**, with nothing
in the record to show it happened. A skewed draw must therefore translate its
own positions back to test-set positions before sharding. Changing the existing
pool's draw would also change published throughput and owe a bump under §3; a
new phase changes nothing existing.

## 9. Tier 6 — what these measurements would support in a paper

Kept because it is the reason the tiers are ordered as they are, and framed as
what the measurements **would** support rather than as a commitment to write
anything. Nothing below is a result.

**The thesis.** Standard ANN benchmarking — ann-benchmarks and its descendants
— ranks engines on recall and throughput over a static corpus in a closed loop.
That methodology is structurally unable to observe four things that decide
production behaviour, and the claim is that each is demonstrable rather than
arguable:

1. **The headline metric is optimistically biased by an unrecorded amount.**
   Best-of-N over a noisy variable, with N a configuration knob absent from the
   artefact's cache key (§7). This is the strongest of the four, because it
   applies to every published ann-benchmarks result, and it needs no new
   measurement to state — only the arithmetic and the cache key.
2. **Tail latency is unobservable by construction.** Closed-loop arrival means
   a stall suppresses the very requests that would have measured it (§6).
3. **Index freshness is invisible, and batching converts it into apparent
   speed.** An engine that defers indexing returns higher throughput at the
   same measured recall (§5). **Reasoned**, and the magnitude is
   **unconfirmed**.
4. **Resource-budget and representation fairness are not checked at all**, so
   one engine can be measured under a different budget, grid or stored
   representation than another. The evidence for this one is this repository's
   own defect catalogue, which is unusual and is the point: these are the bugs
   that a framework built specifically to avoid them still shipped.

**The figure it would turn on.** Two engines indistinguishable on the
recall/throughput Pareto frontier, separated by orders of magnitude on
freshness lag or on open-loop p99. mongot is the candidate, because its build
mode is asynchronous and its lag is currently waited out by
`wait_index_synced` and never published — but whether the separation is orders
of magnitude, or anything at all, is **unconfirmed** and is exactly what §5
would measure.

**The second contribution is the one that is genuinely under-published:
testing the benchmark.** Guards derived from the tree rather than written as
lists; probes checked by planting the failure they are supposed to catch; a
citation guard that fails when a reference rots; refusals in code rather than
in comments. With this week's catalogue as the instance set — a report
publishing one machine's numbers under another's limits, four tables joining on
incomplete keys, ratios dividing incommensurable windows, a documented
measurement that was never collected. §0 of this page is an instance of the
same idea applied to itself.

**What is missing before any of this is writable:** §5 and §6 implemented; a
run matrix across enough engines to show that the ranking **changes** under the
new axes (not a complete matrix — the contribution is the method); and the
statistical argument of §2.1 made explicitly rather than left as a raise and a
comment.

**The conflict of interest is real and has to be led with, not buried.**
Percona engines benchmarked by a Percona employee. The answer is the paper's
own subject matter: the fairness invariants are the defence, they are in
`config/` and `tests/` rather than in prose, and they are auditable by anyone
who clones the repository. A paper that argues benchmarks should be testable
and then asks to be trusted has argued against itself.

## 10. Sequencing

1. **§4.1 through §4.3.** They are defects, and §4.2 blocks §5.
2. **§7's cheap fix** — one validity check, one chart note.
3. **§4.4 through §4.7** — the rewords and the small additive records.
4. **§5**: the freshness probe first, then the rate-controlled writer.
5. **§6**: open-loop as an additive phase, with its refusal guard.
6. **§8**: the documentation now, the new phase later.
7. **§9** once §5 and §6 produce a figure, and not before.
