# Implementing the Percona vector team's feedback

Four requests came back from the server team on the benchmark we published, and a
fifth message pointed at a new commit. This document is the plan for all five. It
is a plan and nothing in it is implemented.

The short version: the commit we were pointed at is not a commit on the branch we
measure. It is what a pull request landed when it **merged the whole vector search
MVP into Percona Server's `9.7` branch**, on a base roughly four months newer than
our pin. Seven of the vector tickets that merge carries are already reachable from
the commit we build, so they are not news; what it adds that we do not have is
**one** named INSERT-crash fix, PS-11748, and that one is a robustness fix layered
on PS-11591, which our pin already contains.

So the re-pin still comes first, but for weaker and different reasons than the
first version of this document gave: one robustness fix, a four-month-newer base,
and the fact that we cannot attribute a 2026-10-15 failure to any of them without
re-measuring. The re-pin gates the other four requests because it changes the tree
every number would be measured on, not because it closes the bugs we hit.

And the thing that is now true is better news for the team than the thing we first
wrote. PS-11748's own commit body names PS-11591 as "the most likely reason for
corrupted HNSW graph" and says it "has been fixed separately". PS-11591 is in our
pin. **The crashes we measured on 2026-10-15 therefore happened with the fix for
the most likely cause of graph corruption already in the build**, which makes
those traces worth collecting rather than worth discounting.

---

## 0. How this document cites code, and why it cites no line numbers

Two rules, both taken from `docs/11-methodology-roadmap.md` §0, where they are
written down and where `tests/test_e15_methodology_roadmap_doc.py` enforces them.

**First: symbols only** — a file and a function, class or key name, never
*file*-colon-*line*. The reason applies here with extra force: this document's
whole subject is a vendor tree that moved 721 commits under a pin, and §3 below is
partly about what line citations into a tree that moves have already cost us. A
line number dies to any edit above it; a symbol name dies only to a rename.

**Second: vendor material is named in prose, never in code font.** A backticked
path or identifier in this document means a path or identifier in *this*
repository, and the guard enforces that by resolving every code-font span against
this tree. So the vendor's design document, vendor C++ types and functions,
server and kernel variables this framework does not itself set, and commit SHAs
are all in *italics*. A vendor file is not in this repository on a fresh
checkout — `sources/` is generated and gitignored — so a backticked path into it
would be a citation checkable only on the machine that happened to fetch the tree.
A server variable that *is* in our configs stays in code font, because it resolves:
`innodb_hnsw_ef_search` and `innodb_hnsw_max_memory` are ours to set, while
*innodb_parallel_read_threads* is not set anywhere here yet and *Handler_read_key*
is a counter we read rather than a key we write.

Phase 0's own deliverable is a guard for this document, so the document has to be
guardable. Running `classify()` and `unresolved()` from
`tests/test_e15_methodology_roadmap_doc.py` over this file is the check, and it
reports no unresolved path and no unresolved symbol.

---

## 1. Provenance of every claim here

**The four requests.** Relayed to this session on 2026-10-27, from the server
team's review of the benchmark report and from a standup comment. Paraphrased
below with their substance kept:

1. Get stack traces for the crashes, to get an idea of the cause.
2. `ef_construction` having so little effect is surprising, and worth digging into.
3. Filtering going that badly against pgvector is interesting, since the two
   implementations are more or less the same thing for iterative scan.
4. Benchmark the scenario where loading data and building the index are
   **separated**, as was done for pgvector in an earlier document. The reasons
   given: a bulk build parallelises the distance calculations across cores, and it
   avoids rewriting neighbour lists at every insert, so the writes become fewer and
   sequential instead of random. The ask includes a recommendation to users to
   bulk-load with a separate index build where possible.
5. A new commit, *56124b03*.

The report's own counterpart to request 2 is that `ef_construction` 50 → 800 bought
0.9 points of recall "for 16× the build cost", and to request 3 that filtered
throughput falls to 0.13-0.22 QPS on sift at 3.3% selectivity while pgvector holds
flat. Both are in the published findings. §6.1 below is about the first of those
two sentences: the recall figure is real and the cost figure is not measured.

**The upstream state.** Measured on 2026-10-27 through the GitHub API against
*percona/percona-server*, not read from a page or inferred:

| What | Value |
| --- | --- |
| Our pin (`config/engines/percona.yml`, `source.tag`) | *02a8b80c*, 2026-10-13, PS-11626 |
| *vector-mvp* head now | *c0f76a5e*, 2026-10-20, PS-11751 — 7 commits past our pin |
| The commit we were sent | *56124b03*, 2026-10-26 |
| What it is | the squash *merge_commit_sha* of PR #6278, *vector-mvp-97-squashed* → base `9.7`, **merged 2026-10-26 12:41 UTC** |
| That PR | 30 commits, 331 files, +37,587 / −322 |
| `9.7` head **on 2026-10-27** | *f52fe4b0* — 4 commits past the merge, none of them vector work |
| Our pin against the merged series | **diverged**: our pin is 43 ahead and 721 behind |
| Server version | `9.7.1-1` at our pin, `9.7.2-2` at `9.7` |

Two details about *56124b03* that the first version of this document got wrong,
and that decide how to describe it. It is PR #6278's *merge_commit_sha* and a
squash: it has the single parent *e2a8d316*, and the PR's own *head.sha*
(*1fc5dadc7265e0080e70320edcf238c3743d2b7a*) is **not** an ancestor of it. So it
is "what PR #6278 landed", not "the last commit of" that PR — the branch's own
commits are not in the history we would be pinning to. And the `9.7`-head row was
exact when it was derived and was already stale within a day; it is timestamped
above for that reason, and a reader who needs the current value has to re-derive
it rather than read it here.

**What is unconfirmed.** Nothing in §4 onward has been measured at the new pin,
because the image has not been built from it. Every estimate of what a phase costs
comes from the 2026-10-15 run, from the two *percona* profile runs on 2026-10-20
and 2026-10-21, or from the estimator — all of which describe the old pin. The
vendor's design document *percona-hnsw.md*, which now ships in the tree, is read
as a statement of intent by its authors: where this plan quotes it, the quote is
evidence about what the server is *meant* to do and the measurement is what settles
whether it does.

---

## 2. The merge, and what it changes about the four requests

The series was not fast-forwarded onto a newer base. It was re-landed: our pin is
43 ahead and 721 behind the merged series, so our 43 vector commits were replaced
by a re-sequenced set on top of roughly four months of upstream MySQL and Percona
work. Pulling the new pin therefore changes the InnoDB underneath the graph as well
as the graph code. That base move is the real content of the re-pin.

Commits in the merge that bear on a request — **with the column that the first
version of this document was missing.** Each "yes" was verified by asking the API
whether the ticket's commit is an ancestor of our pin; all seven come back
*status: ahead, behind_by: 0*, which means our pin already contains them.

| Ticket | What it is | In our pin? | Bears on |
| --- | --- | --- | --- |
| PS-11591 | Assertion *inserted.second* in *HNSW::insert()* — crash on INSERT into a vector-indexed table after an unclean restart | **yes** (*af72e2c0fa40*, 2026-10-12) | request 1 |
| PS-11623 | Assertion in *ddl0loader* when a vector index build aborts on `innodb_hnsw_max_memory` exhaustion | **yes** (*eebc427aab7b*, 2026-10-09) | request 1, request 4 |
| PS-11615 | Assertion in *vec_index_of()* when a vector search runs during an in-place vector index replace | **yes** (*a86503214416*, 2026-10-12) | request 1, our churn phase |
| PS-11612 | Resolved HNSW parameters stored in the data dictionary | **yes** (*1d62ab22bc75*, 2026-10-13) | request 2 |
| PS-11618 | COSINE distance support for vector indexes | **yes** (*d436bb1c77d7*, 2026-10-12) | puts the angular corpora on the index path |
| PS-11212 | Vector index optimizer changes | **yes** (*3146c007c323*, 2026-10-05, plus a post-fix *23dcf68055f1*, 2026-10-07) | request 3 |
| PS-11750 | Status variable for total vector index memory usage | **yes** (*63e175565572*, 2026-10-08) | the gauge `docs/11` §4.1 says we collect nowhere |
| PS-11748 | SIGSEGV in *HNSW::Node::state()* on ANN SELECT or INSERT/REPLACE after an unclean restart, release and debug | **no** (*1e08863e66f7*) | request 1 |
| PS-11588 | HNSW index options silently accept duplicate keys | **no** (*18aef90c0214*) | request 2 |
| PS-11616 | The remaining vector index optimizer improvements | **no** (*4186d36fee28*) | request 3 |
| PS-11614 | *SELECT … FOR UPDATE/FOR SHARE* locked rows outside the result set via a stale candidate | **no** (*f62d3be879fe*) | correctness of the read path |
| PS-11617 | A same-name DROP + ADD of a vector index with a different *M* is a silent no-op | **no** (*8437132cc95a*) | our churn repair path, §5.3 |

Those five "no" rows are **exactly** the five of *vector-mvp*'s seven commits past
our pin that are ticketed vector fixes. The other two are *32604201d783*, a
test-flakiness fix, and *c0f76a5ecf73*, which is PS-11751. There is no sixth
surprise hiding in the gap.

**So the merge carries one named INSERT-crash fix our tree lacks, not two.**
PS-11748 is that one, and it is the **second** commit past the SHA we build, not
the third: *compare/02a8b80c...1e08863e66f7* returns *ahead_by: 2*. PS-11591, the
other crash-on-INSERT fix, is already ours.

Beneath them, the newer base carries InnoDB buffer-pool fixes that a crash
investigation would otherwise have to rule out by hand: PS-11850
(*buf_page_init_for_read()* latching a stale page-hash cell during a buffer pool
resize) and the *buf_page_optimistic_get* fast-path change filed under PS-11495.
**That second ticket has two commits and only one of them is in the merge** —
see §3.1, which excludes the other. Crediting one ticket to both sides of the pin
is how a changelog becomes an argument.

**The consequence for request 1, restated.** The 2026-10-15 failures were measured
on a tree that is missing PS-11748 and nothing else named in the crash family. That
is not a reason to withhold a backtrace; it is a reason to read PS-11748 before
taking one. Its commit body says the crash came from a neighbour list referencing a
node absent from that layer, that such references "should not normally occur in
properly constructed HNSW graph, but might be caused by bugs or data corruption",
that PS-11591 "was the most likely reason for corrupted HNSW graph" and "has been
fixed separately", and that the fix makes the HNSW class robust against misplaced
nodes — detecting the corruption and reporting it rather than crashing on it.

Read against our pin, that says something useful. **We already had PS-11591**, so
the graphs that died on 2026-10-15 were built by a server with the fix for the most
likely corruption source in place, and PS-11748 would have converted the crash into
a reported index corruption rather than prevented the corruption. Re-measure at the
new pin first, because the base moved and we cannot attribute a failure to a
four-month-old InnoDB without it. But go in expecting the failures to survive, and
plan the capture chain in §4 as work that will be needed rather than work that
probably will not. Re-measuring is not free, and §4.1 gives the figures: a
configuration that dies dies 0.5-5.5 s into its first INSERT, while one that
survives costs a whole corpus load.

---

## 3. Phase 0 — re-pin, which gates everything else

1. **`source.tag` → *56124b03*.** Not `9.7`'s head: that branch moves daily, and
   the four commits past the merge as of 2026-10-27 are an OIDC install-script
   option (PS-11568), a *buf_block_t* padding computation (PS-11495, commit
   *22e64c5b3beb*) and two keyring-KMIP changes (PS-11925). None is vector work.
   **The padding commit is PS-11495's other half**, and it is why §2 credits that
   ticket to the merge only for the *buf_page_optimistic_get* change: the fast-path
   commit is an ancestor of *56124b03* and the padding commit is two commits past
   it. *56124b03* is an ancestor of `9.7`, so the tree is exactly what PR #6278
   landed and it does not move under us. The key stays spelled `tag` for the reason
   that file already records: `orchestrator/engines.py`'s source-tag helper reads
   `tag` or `version` and nothing else, and a renamed key would publish a run whose
   engine version is blank. The surrounding comment names branch *vector-mvp* and
   must be rewritten — the engine is no longer built from a feature branch.

2. **Re-verify the cmake flags against the new base.** The flag list in
   `config/engines/percona.yml` and its deliberate duplicate in
   `docker/percona/Dockerfile` were established by five failed configures, and the
   lesson recorded there is that on this tree switching a dependency off breaks a
   Percona component that wanted the target. A four-month base move is exactly the
   event that invalidates that work. `march: x86-64-v3` does not change, for the
   reason CLAUDE.md gives.

3. **Version strings.** `9.7.1-1` → `9.7.2-2` wherever the figure is asserted
   rather than read. **The probe banner is not one of those places and needs no
   edit**: `harness/main.py` prints `driver.server_version()`, which for this family
   runs *SELECT VERSION()* and reports whatever the server says. What does need
   editing is the literal string "MySQL 9.7.1", which appears six times in six
   files — `README.md`, `docs/09-adding-percona-server.md`,
   `config/engines/percona.yml`, `docker/percona/init.sql`,
   `docker/percona/entrypoint-percona.sh` and `docker/percona/Dockerfile`. All six
   are prose or comments; none is a runtime check. **Three of them are inside
   `docker/percona/` and are baked into the image** — the Dockerfile itself, the
   entrypoint it copies to `/usr/local/bin/vb-entrypoint`, and `init.sql`, which it
   copies to `/opt/vbench/init.sql` — so editing those three costs a rebuild and
   should land in the same change as the `source.tag` bump rather than after it.

4. **`docs/09-adding-percona-server.md` citations will be wrong, and the suite will
   say so.** That document cites *file*-colon-*line* into the pinned tree and into
   recorded MTR `.result` output, and `tests/test_e1_percona_design.py` checks them:
   70 citations, 61 of them distinct, 27 carrying a path-and-anchor pair in
   `CITED_IN_REPO`. The base moved and the vector commits were re-sequenced, so most
   of those citations now point at unrelated lines. Two ways forward: re-derive
   every one at the new pin, or convert the document to symbol-only citations as
   `docs/11` §0 prescribes. **Recommend the conversion**, with its true size stated
   in §9 rather than implied — it is not a document edit, it is a rewrite of
   `CITED_IN_REPO` and the guard built on it, and it is the largest single item in
   this plan. Line citations into a moving vendor tree have now cost a red `master`
   twice and a full re-derivation once; this is the third event of the same kind, and
   the conversion is the only one of the two that does not have a fourth.

5. ***percona-hnsw.md* now ships in the vendor tree.** It is the authoritative
   design document for the feature, written by its authors, and several of
   `docs/09`'s §2.x subsections — the index options, how the index is maintained,
   the server variables — exist because no such document was available when they
   were written. Cite it by section and delete what it supersedes rather than
   keeping a second account that can drift from it.

6. **Build, then `dev`, then `smoke`.** No long profile runs until smoke is green at
   the new pin. **What that supplies is the ingest rate at the new pin, not the
   first one ever measured.** Percona's ingest rate has been measured fifteen
   times across four runs — *full-20261015-215006*, *percona-20261015-213836* and
   the two *percona* runs of 2026-10-20 and 2026-10-21 all record
   `ingest_rows_per_s` — from 48.17 on gist-960 at `M=32` to 638.11 on sift-128 at
   `M=8`. None of it was ever fed back into `orchestrator/estimate.py`, whose
   `_MEASURED_ROWS_PER_S` and `_INGEST_ROWS_PER_S` both omit this engine, so
   `_effective_rate` still returns `None` for it and the estimator still names it
   in `engines_without_ingest_rate`. **So the rate is measured but unmodelled**,
   which makes the fix a table entry rather than a run; CLAUDE.md said the rate was
   unmeasured and is corrected in the same change as this section. Feeding the
   table is separate work from running smoke, and the smoke run is what makes the
   figure describe the tree we would be publishing.

7. **A guard for this document.** `docs/09`, `docs/10` and `docs/11` are each
   guarded by a test that fails when the document drifts from the tree. This one
   should be too, symbol-only, in the shape of
   `tests/test_e15_methodology_roadmap_doc.py`. §0 above states the two rules it has
   to enforce, and this document already satisfies both.

---

## 4. Phase 1 — stack traces for the crashes (request 1)

### 4.1 Re-measure before building any diagnostics

What the 2026-10-15 run recorded, as the evidence a re-run is compared against:
13 of 24 ops units died inside the first INSERT with *Lost connection to server
during query* within 0.5-5.5 s, server RSS flat at the buffer-pool baseline and CPU
at 0.16 cores; across the million-row load configurations, 12 of 18 killed the
server. The failure was non-monotonic in `M`, which rules out a simple memory
ceiling, and peak RSS was 48.5 GiB against a 112.5 GiB cap. One case went further:
**gist-960 euclidean in the tuned pass at `M=32`, 1,000,000 rows**, loaded
successfully in 20,760 s — 48.17 rows/s — and then died during churn, which puts
the fault on the write path generally rather than on bulk load alone.

Re-run those configurations at the new pin. If they survive, the answer to the
request is the ticket list in §2 and no backtrace is needed. If any still dies, the
capture chain below is what turns it into something the server team can act on.

**The failures are cheap and the survivals are not**, and the plan should say so
before anyone schedules this. A unit that dies dies 0.5-5.5 s into its first
INSERT. A unit that survives costs a full corpus re-load: at the rates this engine
actually recorded, that is minutes on fashion-mnist and 5.8 h for the one gist
`M=32` configuration alone. **The good outcome is the expensive one**, and phase 1's
budget has to be the all-survive budget rather than the all-die one.

### 4.2 Container exit facts — the one artefact that cannot be reconstructed

Nothing in the orchestrator records the server container's exit state. *docker
inspect* carries *ExitCode*, *OOMKilled* and *FinishedAt*, and *OOMKilled* settles
cgroup-OOM against an in-process crash without any cooperation from the server.
Record all three per failed phase. This is the cheapest item in the whole plan and
the only one whose data is gone forever once teardown runs.

### 4.3 The server's own backtrace

mysqld's fatal-signal handler prints an *Attempting backtrace* block to
*--log-error*, and the build is *RelWithDebInfo*, so those frames carry names. The
error log is already copied out before teardown — `OpsRun._save_server_error_log`
exists for exactly this, and `config/engines/percona.yml`'s `error_log` key records
what its absence cost in the 2026-10-15 run. What is missing is the check that the
block actually arrives, plus *--core-file* in the engine's server flags.

### 4.4 Core dumps

Two gaps in the engine's configuration, and **two missing pieces of plumbing that
the first version of this document left unnamed.**

The configuration gaps: the server container runs with no *--ulimit*, and
`orchestrator/docker_ctl` has no ulimit plumbing at all, so the core size limit is
whatever the daemon defaults to; add *--ulimit core=-1* for this engine. And the
kernel's *core_pattern* is host-wide: a benchmark framework must not modify the
machine it measures, so **read it, print it, and say plainly that no core will be
produced when it is a pipe to apport or systemd-coredump.** Document the one-line
opt-in for the build host instead of performing it.

The plumbing: a core that lands at an absolute path lands inside the container's
namespace, so pointing the pattern into the data mount puts it where the error log
already goes — **and under the same teardown trap.** `OpsRun.__exit__` calls
`teardown`, which removes the volume and then deletes the bind-backed directory
under `state/ops`, so the core dies with it. Two things have to be written for a
core to survive:

* **A copy-out beside `OpsRun._save_server_error_log`.** That method is the pattern
  to follow and the reason it exists: it runs from a `finally` inside the
  `with OpsRun(...)` block, before `__exit__`, and its docstring records that there
  is no second chance afterwards. It must not raise, for the reason its own docstring
  gives, and neither may the core copy-out.
* **A readable run-directory mount in `OpsRun.run_script`.** That method deliberately
  mounts no writable path — its docstring says so, and the reason is good: a lab
  script must not be able to put a number in a report. But it also means the `lab`
  route in §4.5 cannot *see* a saved core, let alone write beside it. A read-only
  mount of the run directory at a distinct path is the narrow change that keeps the
  no-writable-path rule intact and still lets gdb open the file.

Without both, the sequence is: the server crashes, a core lands in the volume, and
`__exit__` deletes it before anything has read it.

### 4.5 Turning a core into text

**gdb is installed in no stage of any image, and that is a recorded decision rather
than an oversight.** `docker/percona/Dockerfile` lists it among "Vendor packages
deliberately NOT taken, with the reason, so the audit does not have to be redone",
alongside valgrind and the rest of Percona's packaging and test pipeline. It appears
in no *apt-get install* anywhere, and nowhere else under `docker/`, `scripts/`,
`config/`, `orchestrator/` or `harness/`. So "gdb is in the builder stage" was
simply wrong, and the route this section proposes needs a Dockerfile change and a
rebuild before it can run at all.

The change is to add gdb to the **bench** image rather than to runtime, because the
bench image is what `OpsRun.run_script` starts and because keeping it out of runtime
keeps it out of the container being measured. **The one point in the plan's favour
survives intact**: every bench image is built `FROM runtime AS bench`, Percona's
included, so the *RelWithDebInfo* mysqld and its symbols are already in the image
the debugger would run in. Nothing has to be copied between stages.

With that in place, add a `lab` route —
`./run-benchmark.sh lab --engine percona percona-backtrace.py` — that runs
*gdb --batch -ex 'thread apply all bt full'* against the saved core and the matching
mysqld. `lab`'s documented job is "diagnose, don't measure", which is exactly this.
The deliverable to the server team is the backtrace text and the statement in
flight, not our reading of them.

### 4.6 Reproducer context

With the backtrace, send the statement in flight, the row offset it died at, the
resolved server flags, and *SHOW ENGINE INNODB STATUS* where the server still
answers. The team asked for an idea of the cause; a reproducer they can run is worth
more than a trace they cannot.

---

## 5. Phase 2 — load and index build separated (request 4)

This is the highest-value request, and the framework already has the machinery: it
is `build_mode: post`, implemented, tested, and switched off because
*ALTER TABLE … ADD VECTOR KEY … ALGORITHM=INPLACE* killed the server at the old pin
(9.7.1-1). `config/engines/percona.yml`'s `capabilities.bulk_vector_index_build` is
the switch. **It is now on**: at the current pin (9.7.2-2) the bulk build completed
25 times out of 25 on 2026-11-03 and matched `incremental`'s index quality at 60,000
rows, so the ops pass builds Percona `post` by default. Items 1 and 2 of §5.2 are
done; the rest of §5.2 is not.

### 5.1 What the vendor document now says the bulk path does

*percona-hnsw.md* §15 states, as the authors' design: the first
*ADD VECTOR KEY* is an in-place **rebuild**, because the hidden label column has to
reach every row; the rebuild takes the sort rather than skipping it, so the
clustered scan and the graph inserts run on *innodb_parallel_read_threads* threads;
and the aux table is written **once at the end** by walking the finished graph,
instead of rewriting a node's row every time a later insert rewires it, which the
document costs as O(N·M·log N) row updates avoided. Those are the two reasons the
server team gave, in the server team's own words. §18 adds the third variant:
*ALGORITHM=COPY* writes the aux row by row with undo and redo, and is both slower to
build and larger.

### 5.2 The work

1. **DONE, with the hazard below removed by splitting the key.**
   `capabilities.bulk_vector_index_build` → `true` once `post` survived at the new
   pin. The churn repair is now withdrawn by its own key,
   `capabilities.vector_index_replace: false`, read by `_vector_index_replace` in
   `orchestrator/cli.py` (falling back to the bulk-build key for engines that
   declare no replace key), carried as `--no-vector-index-replace` and applied by
   `PerconaDriver.declare_vector_index_replace`. So the flip restored `post` and did
   NOT re-arm the repair; §5.3's switch can land on its own. What follows is the
   hazard as it stood before the split. **That flip did two things, not one, and
   the second one was a hazard.** `orchestrator/cli.py`'s `_bulk_vector_index_build` was the single reader
   of that key, and it fed both the ops loop's build-mode default and the
   `--no-bulk-vector-index-build` flag that then reached
   `PerconaDriver.declare_bulk_vector_index_build`. So flipping it would have
   restored the `post` build mode **and** re-armed the churn repair path — which
   §5.3 establishes is still a no-op at the new pin, because PS-11617 fixes only the
   differing-*M* case. Seven profiles set `workloads: [..., churn]` (`full`, `main`, `smoke`, `dev`,
   `quick`, `mariadb-blog`, `tuned-complete`). **So the flip and the
   *ALTER TABLE … FORCE* switch in §5.3 had to land in the same change**, or no
   churn-running profile could run between them — otherwise we publish the degraded
   index a second time and label it repaired, which is the exact outcome the
   capability was set to `false` to prevent.
2. **DONE.** `config/profiles/percona.yml`: `percona_build_modes: [incremental]` →
   `[post, incremental]`.
3. ***innodb_parallel_read_threads* is set in none of our configs, and pgvector's
   counterpart is set in ours.** pgvector's tuned pass sets
   `max_parallel_maintenance_workers`, `max_parallel_workers` and
   `max_worker_processes`, all three from `{build_threads}`. **Its normalized pass
   pins two keys, and they are not the same two**: `max_parallel_maintenance_workers=0`
   and `max_parallel_workers_per_gather=0`. The other two tuned keys do not appear in
   the normalized block at all and stay at the server default there — which matters,
   because "the same treatment" is what this item prescribes and the treatment is
   asymmetric already. Enabling Percona's bulk build without any of it would compare
   pgvector at `build_threads` workers against Percona at the server default: a
   comparison of defaults wearing the label of a comparison of engines, which is the
   failure mode CLAUDE.md opens with. Set *innodb_parallel_read_threads* from
   `build_threads` in tuned and to 1 in normalized, record it in the manifest, and
   while doing so decide whether pgvector's normalized block is missing two keys or
   deliberately leaving them alone — the answer belongs in that file's comment either
   way.
4. `PerconaDriver.create_index` should spell *LOCK=SHARED*. *LOCK=NONE* is refused
   while a vector index is on the table, and an unqualified ALTER leaves the choice
   to the server.
5. **Report time-to-queryable as its two components, and name the asymmetry.**
   Percona's first *ADD VECTOR KEY* rebuilds the table; pgvector's *CREATE INDEX*
   does not. A single "bulk build is faster" number hides a table rebuild inside one
   engine's figure. This belongs in the known-asymmetries list that the report always
   prints, beside the existing `ef_construction` and bulk-versus-incremental entries —
   and see §9, because both of those entries are currently wrong about this engine.
6. **Add the load-parallelism axis**, so the shape matches the pgvector document the
   request points at: one thread with the graph maintained per INSERT, against N
   threads of pure load followed by one bulk build. `load_threads: 1` is what every
   profile but two sets today, and `load_threads: 8` is already live in
   `config/profiles/tuned-complete.yml`, so this is a profile change and not a code
   change.
7. Optionally a third point, *ALGORITHM=COPY*, to put a number on what §18 claims
   about it.

### 5.3 The churn repair path is our measurement bug, not theirs

**In the 2026-10-15 run**, `PerconaDriver.maintain()` issued a same-name
*ALTER TABLE … DROP KEY vi, ADD VECTOR KEY vi …, ALGORITHM=INPLACE*. The vendor
document states that a same-name DROP + ADD with an identical definition is
cancelled by the server before InnoDB sees it, and PS-11617 fixes only the case
where *M* differs. That is a complete explanation for what we published 20 times:
1-4 ms, `index_bytes` identical to the byte, recall identical to five decimals. We
read it as a repair that does nothing; it is a statement the server discards.

The past tense is deliberate. **At HEAD that statement is no longer issued at all**:
`maintain()` returns `None` before touching the cursor whenever
`capabilities.vector_index_replace` is false, printing the withdrawal into the
phase log instead. So the no-op is a property of the published run, not of the
current code. Flipping that key is what would bring it back — the bulk-build key no
longer can, since §5.2 item 1 split them.

The repair that does rebuild is *ALTER TABLE … FORCE* or *OPTIMIZE TABLE*, which the
vendor document names as the rebuild that repairs a corrupt index. Switching to it
changes what a churn number means, so it takes an `OPS_MEASUREMENT_VERSION` bump —
the charter in `harness/metrics/records.py` is explicit that a bump belongs to any
change that alters what an ops number means.

**That switch has an objection already written down, in the docstring of the method
it would change, and this plan has to answer it rather than walk past it.**
`maintain()`'s docstring rejects *OPTIMIZE TABLE* on exactly these grounds: it
"rewrites the whole base table — every row, every secondary index — so its cost is
dominated by work that has nothing to do with the vector index and the churn phase
would be timing a table rebuild." The objection is correct, it applies to
*ALTER TABLE … FORCE* at least as strongly, and it is the reason the narrow index
replace was chosen in the first place.

The answer is that the objection is about the *cost* number and the switch is about
the *recall* number, and the two can be separated:

* **What the churn phase is for is whether the graph recovers**, and at our pin it
  cannot answer that at all — a statement the server discards leaves recall
  identical to five decimals whether the graph was degraded or not. A rebuild whose
  cost is a composite still answers the question the phase exists to ask. An exact
  cost for an operation that did nothing answers none.
* **The cost figure then has to be labelled for what it is** and must not be
  compared across engines: a table-rebuild-inclusive repair cost is not pgvector's
  *REINDEX*. That is the same treatment §5.2.5 prescribes for time-to-queryable, for
  the same reason, and it belongs in the same asymmetry list.
* **The composite can be decomposed** if the number is wanted: time an
  *ALTER TABLE … FORCE* on the same corpus with no vector index present and subtract.
  That is one extra configuration, it is cheap relative to a churn phase, and it
  turns "dominated by work that has nothing to do with the vector index" from an
  argument into a measured term.
* **The narrow repair stays available** and should keep being issued once PS-11617's
  differing-*M* path is the one we drive — the two are not exclusive, and the
  docstring's preference for the narrower operation is right whenever the narrower
  operation does something.

One supporting claim in that docstring does not check out and should not be weighed
while this is decided: it says the suite "proves" the index replace leaves
*TABLE_ID* unchanged "by asserting" it, and no test in this repository mentions
*TABLE_ID* at all. That is a separate repair, in a separate file, and it is noted
here only so the objection is weighed on the parts of it that are evidenced.

---

## 6. Phase 3 — ef_construction (request 2)

### 6.1 What our current answer actually is

**The sweep has been run, twice, and the first version of this document described a
run that does not exist.** It is in the *percona* profile runs
*percona-20261020-140151* and *percona-20261021-134142*, in the `tuned` pass, phase
`recall_qps`, at `M=16` on fashion-mnist only, with `ef_construction` in {50, 200,
800} — five `ef_search` points at 50 and 800 and nine at 200. Both runs carry the
same grid and identical figures. The profile ran a third time, on 2026-10-15
(*percona-20261015-213836*), but that one ran the ops phase alone in the
`normalized` pass, where the knob is coerced to the single vendor default, so it
contributes a baseline and not a sweep point. The `percona` profile's own header
comment says nothing here has ever been run, which is false of all three, is what
misled a review of this document, and is corrected in the same change as this
section.

The five `ef_search` points the three settings share, from `recall_at_k`:

| `ef_search` | efC 50 | efC 200 | efC 800 | spread | QPS at efC 50 / 200 / 800 |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.96474 | 0.97136 | 0.97339 | 0.87 pt | 1971 / 2154 / 2011 |
| 20 | 0.98724 | 0.99061 | 0.99258 | 0.53 pt | 1810 / 1882 / 1862 |
| 40 | 0.99530 | 0.99663 | 0.99812 | 0.28 pt | 1527 / 1547 / 1523 |
| 80 | 0.99824 | 0.99910 | 0.99946 | 0.12 pt | 1191 / 1190 / 1170 |
| 160 | 0.99919 | 0.99951 | 0.99982 | 0.06 pt | 882 / 827 / 824 |

Three things follow, and all three make the document's own argument correct instead
of merely true-sounding.

**The corpus is the full 60,000 rows, not 20,000.** `subset_rows` is an `ops:` key.
It is read by `orchestrator/ops_pass.py`, `orchestrator/cli.py` and
`orchestrator/estimate.py`, and reaches the harness as `--subset-rows`;
`orchestrator/ann_pass.py` never reads it, so it cannot cap an ann sweep. The 20,000
figure is the ops path's, where the same two runs recorded 20,000 rows — and the ann
sweep above is a different pass on the whole corpus. Reporting the ops subset as the
sweep's corpus understated the sweep by a factor of three.

**The ceiling argument belongs at the top of the grid, not the bottom.** Recall at
the smallest grid `ef_search` is 0.9647-0.9734, not 0.989, so the point the report's
headline figure comes from is *not* the saturated one. The saturation is above it:
by `ef_search = 40` all three `ef_construction` values exceed 0.995, and by 160 all
three exceed 0.999. The spread falls monotonically from 0.87 points to 0.06, so the
only point on this grid where the three settings separate by anything like the
published figure is `ef_search = 10` — one point out of five, at the lowest search
effort anyone would run. That is still a real weakness in the evidence, and it is a
different weakness from the one we claimed.

**"16× the build cost" is not measured anywhere.** These ann records carry no
build-time field at all. Their keys are `build_mode`, `clients`, `dataset`,
`ef_construction`, `ef_search`, `engine`, `engine_version`, `extra`, `index_bytes`,
`k`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`, `m`, `march`,
`metric_space`, `phase`, `qps`, `queries_executed`, `recall_at_k`, `resource_pass`,
`run_id`, `storage_engine`, `timestamp` and `vector_index_used` — and the ops path,
which does carry `build_wall_s` and `ingest_rows_per_s`, never varied the knob: it
ran one build mode at one `ef_construction` and does not record which value.
**The "16×" in the shipped report is the knob ratio 800/50 presented as a cost.**
It should be struck from the report and from any message to the team, because the
team's surprise is partly a response to a cost figure we invented.

One measured fact about cost that we do have, and that belongs in the answer:
`index_bytes` is **255,852,544 bytes in all fifteen rows above** — identical to the
byte across a 16× range of the knob. Build effort buys no index size here. That is
a real finding, it is the kind of thing the team asked for, and it has been sitting
in the records unreported.

The team's surprise is the right instinct. Our measurement is not strong enough to
justify it either way, and one half of what we published in support of it was not a
measurement at all.

### 6.2 What to measure instead

1. **Measure build time per `ef_construction`, so the cost side exists.** This is
   the item that was missing and the reason the "16×" could be invented: nothing in
   the ann path times a build, and the ops path never varied the knob. Vary
   `ef_construction` on the ops path, where `build_wall_s` and `ingest_rows_per_s`
   are already recorded, and populate `ef_construction` on those records. The field
   is already in the shared schema in `harness/metrics/records.py`; no ops writer
   fills it, so an ops record today says `ef_construction_tunable: True` in `extra`
   and never says what the value was. Until both exist, the honest statement about
   build cost is that we have not measured it.
2. **Read the resolved parameters back.** PS-11612 stores them in the data
   dictionary — and it is already in our pin, so this is available now and not after
   the re-pin. The value the graph was actually built with becomes observable rather
   than inferred. Record it per build. The same read catches PS-11588's silently
   accepted duplicate option keys and PS-11617's silent no-op, both of which would
   otherwise look like a knob that does nothing; those two *are* new at the merge.
3. **Sweep where there is headroom.** gist-960 euclidean, and glove-100 on cosine
   now that PS-11618 puts the angular corpora on the index path. At `M=8` as well as
   `M=16`: a smaller out-degree is where edge quality should matter most, and `M=16`
   alone cannot separate "the knob does nothing" from "the graph is already good
   enough at this width".
4. **Report the `ef_search` needed to reach recall 0.95 per `ef_construction`, not
   peak recall.** If build effort buys nothing, that number does not move. Peak
   recall at a saturated operating point cannot show the difference, which is how we
   got here. On the grid above, every setting is already past 0.95 at the smallest
   `ef_search`, which is itself the argument for a corpus with headroom.
5. **Keep it inside the `percona` profile.** One engine, nothing added to the shared
   seven-way grid, which is the reason that profile exists.
6. **State the two bounds the vendor document gives.** `ef_construction` is the
   candidate-search width inside *insert()* and graph quality is fixed at insert
   time, so no query-time setting can recover edges that were never created; and dead
   nodes accumulate per mutation, so a sweep measured after a churn phase is not
   measuring the same graph.

Cost: one full corpus reload per `(ef_construction, M)` pair. Three values at two
`M` on gist is six reloads. **The rate to plan against is the one measured on that
configuration and no other**: gist-960 in the tuned pass at `M=32` loaded 1,000,000
rows at 48.17 rows/s, 20,760 s for one reload. That figure is this engine's slowest
recorded configuration, not its general rate — the same run recorded 598.41 rows/s
on fashion-mnist at `M=8`, 206.15 on sift-128 at `M=32` and 626.47 on glove-100 at
`M=8`, so the cost is dominated by the 960 dimensions and by `M`. At `M=8` and
`M=16` gist will be faster than 48 rows/s and nothing here knows by how much, which
is itself a reason to put the estimate in front of the estimator before anyone
starts it. It may be the item that argues for a smaller corpus than gist.

---

## 7. Phase 4 — filtered search against pgvector (request 3)

### 7.1 Where the server team is right

The mechanism is the same shape. *percona-hnsw.md* §10 describes one
*nn_search_start* followed by *nn_search_next* per candidate, with the traversal's
visited set and unexplored frontier kept in the scan, so continuing costs only the
nodes not yet seen, a candidate is never offered twice, and the scan ends when the
frontier empties rather than at a guessed bound. *LIMIT* sizes the first batch and
nothing more. That is iterative scan, and the colleague's reading of it is correct.

### 7.2 Two differences the same document states, and how to measure each

**No cost model.** Vector index access carries a *prohibitive_cost* of *1e100* so
that it always loses against any other access method, and activation is therefore a
*shape* match rather than a cost decision. **That constant is not in §18 of the
vendor document** — it is in the vendor header *vector-common/vector_constants.h*,
which `config/engines/percona.yml` already quotes verbatim, with the vendor's own
rationale: "Chosen so that it always loses against any other access method, since a
vector index is only ever activated for ORDER BY … LIMIT". What §18 adds is that the
read path is the old optimizer's, with no cost model, and that nothing refuses a
*LIMIT* that asks an approximate search to scan most of the index. pgvector's planner
costs the alternatives. Measure it: run every filtered point twice, once as published
and once with the index suppressed for the *ORDER BY*, and publish the pair. PS-11626
— the last commit in our own pin — is what makes *IGNORE INDEX … FOR ORDER BY* work
for a vector index at all, so the experiment is available from the current pin
forward. If the suppressed plan wins at 1% selectivity, the finding is "the index is
chosen where a scan is cheaper", which is a planner result and not a graph-quality
one. That is a different conversation with the team than the one the current report
starts.

**One clustered primary-key read per candidate.** §10 is explicit that the only disk
access per candidate is a primary-key read, performed under the session's own read
view, before the filter above can reject the row. At 1% selectivity that is on the
order of a hundred row reads per returned row. Instrument it rather than inferring
it — but **the first thing phase 4 has to verify is that the instrument measures
anything at all**, and the first version of this document asserted that it did.

The document describes a **new handler family** for this access path — *vec_init*,
*vec_read_first* and *vec_read_next* — explicitly "the vector analog of the *ft_init*
family", existing as a separate family beside the ordinary *index_* and *rnd_*
families "rather than as a mode of them". It says *index_flags()* returns 0 for a
vector index, because the index "can serve no ordinary read". And it puts the
per-candidate primary-key read *inside* the handler: *vec_read_next* calls
*row_search_for_mysql* with *ROW_SEL_EXACT* under the session's read view. The
server-layer counters *Handler_read_key* and *Handler_read_next* are incremented by
the ordinary *index_* entry points. **Whether the new entry points increment them is
unestablished**, and the design as described gives at least as much reason to expect
zero as to expect a count. So:

* **Step one of phase 4 is a one-query check**: difference the counters across a
  single filtered ANN query at a known selectivity and see whether they move at all.
  That is a `lab` script, it needs no profile run, and it decides whether the rest of
  this instrument exists.
* **The fallback, if they do not move, is engine-side.** *Innodb_rows_read* counts
  rows the storage engine read regardless of which handler entry point asked, and
  *SHOW ENGINE INNODB STATUS* — already named in §4.6 — carries per-second row-read
  rates. Both are coarser than a per-query difference and both are enough to turn
  "filtering is bad" into candidates consumed per result row.

pgvector's symmetric instrument is *EXPLAIN (ANALYZE, BUFFERS)* with rows removed by
filter and heap fetches, and it is unaffected by any of this. Then "filtering is bad"
becomes a number per selectivity, on both engines, in the same units — once we know
which of our two Percona-side instruments is the one that reads.

### 7.3 What that needs from the harness

Per-phase engine status variables, which the harness collects for no engine today.
`docs/11` §4.1 already names the same gap from the other end:
`Innodb_hnsw_memory_used` is documented as recorded per phase in CLAUDE.md, the
README and `docs/09`, and is collected nowhere. Build the capture once and both the
memory figure and the candidate counters land.

**The gauge itself is not what we are waiting for.** PS-11750 is an ancestor of our
pin and *mysql-test/suite/percona_vector/r/hnsw_memory_used.result* exists at
*02a8b80c*, so `Innodb_hnsw_memory_used` is already there to read on the engine we
build today. Only the collector is ours to write. **That half of phase 5 is
therefore not gated by phase 0 at all** and can be built, tested and landed while
the re-pin is in progress — §8's dependency column says so. What does need the new
pin is the filtered instrumentation, because the numbers it produces have to
describe the tree we publish.

Whether that bumps `OPS_MEASUREMENT_VERSION` is a question for the charter in
`harness/metrics/records.py` and not an assumption to make here: new fields beside
unchanged numbers are not the same as a changed meaning, and the charter is the
thing that decides.

### 7.4 One asymmetry to state rather than fix

pgvector has two iterative-scan modes and we sweep both. Percona has no mode to
sweep, so its comparable axis is `innodb_hnsw_ef_search` per selectivity. Say that
in the report instead of leaving a column that looks empty for one engine.

---

## 8. Order, gates and what each phase produces

| # | Phase | Gated by | Produces |
| --- | --- | --- | --- |
| 0 | Re-pin, rebuild, `dev`, `smoke` | nothing | a buildable engine at *56124b03*, and an ingest rate that describes it |
| 1 | Re-run the known crash configurations | phase 0 green | either "fixed upstream, here are the tickets" or a list of what still dies |
| 2 | Crash capture chain | a surviving failure from phase 1; gdb in the bench image; the §4.4 plumbing | exit facts, error-log backtrace, core, *thread apply all bt full* |
| 3 | Bulk build restored, parallel load axis | phase 0; `post` surviving once; **lands with phase 6** | the separated load-and-build comparison the team asked for, with time-to-queryable split in two |
| 4 | `ef_construction` with build time measured | phase 0, estimator check | the `ef_search`-to-reach-0.95 curve per `ef_construction` at two `M`, and the first measured build cost for the knob |
| 5a | Status-variable capture | **nothing** — the gauge is already at our pin | `Innodb_hnsw_memory_used` per phase, for the first time |
| 5b | Filtered instrumentation | phase 0; 5a; the §7.2 counter check | candidates consumed per result row, both engines; index-suppressed comparison |
| 6 | Churn repair switched to a real rebuild | version bump; **lands with phase 3** | a churn before/after pair that measures something |
| 7 | Report and docs | all of the above | `docs/05` and `report/render.py` asymmetries, `docs/09` re-pinned and re-cited, the team-facing write-up |

**Two dependencies the first version of this document had wrong.** Phase 6 was
listed as gated by phase 3; it is not downstream of phase 3, it is the other half of
it. One config key arms both the `post` build mode and the churn repair, so phase 3's
flip re-arms a repair that is still a no-op at the new pin. Either both land in one
change, or no churn-running profile runs between them — and seven profiles run churn.
And phase 5 was listed as gated by phase 0; its status-variable half is not, because
the gauge exists at the pin we already build.

Phases 3+6, 4 and 5b are independent of each other and depend only on phase 0, so
they can run in any order or in parallel on separate machines. 5a depends on nothing
and is the one item that can start today. Phase 2 is the only one that might never
run — but §2 now expects it to, so plan it as work rather than as insurance.

---

## 9. What this plan deliberately does not do, and what it has to do that it first omitted

Deliberately not doing:

- **It does not re-run `full`.** The `percona` profile answers requests 2 and 4 and
  most of 3, one engine at a time, at corpus sizes measured in minutes to hours
  rather than the 726 h of ingest a full seven-engine pass costs before a single
  query runs.
- **It does not change the `innodb_hnsw_max_memory = 0` policy.** The argument in
  `docs/09` §3 is about limit semantics, not about a value, and nothing in the merge
  changes that the knob is a hard allocation ceiling rather than a cache.
- **It does not touch `-march`.** Five engines compile SIMD distance kernels at
  `x86-64-v3` and a different flag on one turns the benchmark into a comparison of
  compiler flags.
- **It does not pin to `9.7`'s head.** A moving pin makes every result
  unreproducible, and the four commits past the merge are not vector work.
- **It does not add a host dependency.** gdb runs inside a container, like
  everything heavy here — which is why §4.5 puts it in the bench image and not on the
  machine.
- **It does not take the team's statements as measurements, or ours as settled.**
  The vendor design document says what the server is meant to do; §5.3 is an example
  of our own published number having a server-side explanation we did not have, and
  §6.1 is an example of our evidence being weaker than the conclusion drawn from
  it — in one case because the cost half of the conclusion was never measured.

Seven items the first version of this document left out, each of which is work
somebody has to do, or a cost somebody has to budget:

- **Phase 1 costs a full corpus re-load per surviving configuration**, and the first
  version of this document called the re-measurement "cheap" on the strength of how
  fast the failures came. A failure is cheap; a survival is a whole load. §4.1 carries
  the figures. **The good outcome is the expensive one**, so phase 1 has to be
  scheduled against the all-survive budget — which for the gist `M=32` configuration
  alone is 5.8 h — and not against the 0.5-5.5 s it takes to reproduce a crash.
- **Converting `docs/09` to symbol-only citations is the largest single item in this
  plan**, and §3.4 recommends it without sizing it. It is 70 citations, 61 distinct,
  across 1,624 lines; 27 of them resolve through `CITED_IN_REPO` in
  `tests/test_e1_percona_design.py`, which maps a document spelling to a path and an
  anchor string, and the rest go through the same guard's other arms. Symbol-only
  citations have no line to re-open, so the mechanism is not re-pointed, it is
  replaced — by the kind of resolver
  `tests/test_e15_methodology_roadmap_doc.py` builds. That is a test rewrite, not a
  document edit.
- **gdb has to be added to the bench image**, which is a `docker/percona/Dockerfile`
  change and a rebuild, and which reverses a decision that file records
  deliberately. The reversal should be recorded there in the same terms.
- **The core copy-out and the lab run-directory mount** in §4.4. Without both,
  `OpsRun.__exit__` deletes every core before anything reads it, and the `lab` route
  cannot see the file it is meant to open.
- **`churn` has to go back into the `percona` profile.** It sets
  `workloads: [build, concurrency, filtered]` today, and the exclusion is
  deliberate — the profile's own comment says a churn phase would measure the no-op.
  Phase 6 needs it, so the comment and the list change together with the §5.3 switch.
- **`report/render.py`'s `_known_asymmetries` prints a false row for a Percona
  run.** Its `ef_construction` row is computed from `engines & {"pgvector", "valkey"}`
  only. Percona is in neither set, so a Percona-only run takes the `else` branch and
  prints "`ef_construction` is exposed only by pgvector" — in a report about the one
  engine in this study that has both a build-time and a query-time effort knob. Its
  bulk-versus-incremental row is unconditional and says pgvector bulk-builds while
  MHNSW and VIDX build incrementally, with no mention of the engine that does either.
  Both rows need Percona, and the set that drives the first one needs to stop being a
  literal.
- **`docs/05-methodology.md` §4 and that generated list are two different artefacts
  and both need the new row.** docs/05's static table is the staler of the two: its
  `ef_construction` row says the knob is not exposed by MariaDB MHNSW or AliSQL VIDX
  and that pgvector exposes it, which omits Valkey as well as Percona. Its
  consequence column then says the build-quality sweep "can only vary `M` for the
  MySQL-family engines" and describes a pgvector-only pin in the normalized pass —
  and Percona Server *is* a MySQL-family engine with the knob, so that sentence is
  not merely incomplete, it is false. Editing one artefact and not the other leaves
  the report and the methodology document disagreeing about the same fact, which is
  how a reader learns to trust neither.
