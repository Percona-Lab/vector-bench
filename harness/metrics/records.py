"""The single flat record schema every measurement lands in.

Both measurement paths — ann-benchmarks (recall/QPS) and this ops harness
(build cost, concurrency, filtered search, churn) — emit the same record shape.
Keeping one flat schema means the report generator has no per-source special
cases and the raw data can be queried with anything that reads JSON lines.

Unused fields stay None. A record is never partially written: `Recorder` writes
one complete JSON object per line and flushes, so a killed run leaves valid
data for everything that completed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA_VERSION = 1

# Bump this in any change that alters what an ops number *means* — the shape of
# the record is `SCHEMA_VERSION`'s business, this is the measurement's. A
# concurrency figure measured with a per-client warmup is not the same quantity
# as one measured without; until this constant existed nothing in the record
# said which it was, so two such figures were indistinguishable and the report
# had no way to keep them off one series.
#
# It is the ops counterpart of ANN_MEASUREMENT_VERSION
# (`orchestrator/ann_pass.py`), which does the same job for the recall/QPS path
# by feeding the fingerprint that keeps a 16 GB curve out of a 64 GB report.
# The two are separate numbers on purpose: a fix to the churn workload must not
# invalidate every recall curve ever measured.
#
# WHY IT IS NOT BESIDE ITS ANN COUNTERPART, which is the obvious place and the
# first thing a reader will want to "fix": ops records are written from inside
# the harness container, which bind-mounts `harness/` and nothing else.
# `harness/` importing `orchestrator` raises ModuleNotFoundError seconds after
# the server comes up and kills the run — it has happened, and
# tests/test_config.py::TestHarnessIsolation exists to stop it happening again.
# So the constant the ops writer needs lives on the harness side of that
# boundary, and `orchestrator/manifest.py` imports it from here. That is the one
# direction this dependency is allowed to point.
#
# Version 1 is implicit. Records written before this constant existed carry no
# `extra.ops_measurement_version` at all, so every reader must treat an absent
# value as 1 — see IMPLICIT_OPS_MEASUREMENT_VERSION — rather than as "unknown"
# and therefore comparable with anything.
#
# VERSION 3 separated the concurrency query pool from `ops.max_queries`. That
# key caps the filtered and churn workloads, which answer each query once, and
# main.yml holds it at 500 to bound their cost; the concurrency workload
# cycles its shard for a fixed duration instead, so it was giving 32 clients
# about fifteen queries each for thirty seconds. Every graph cache in the
# comparison — MHNSW's per-TABLE_SHARE cache, VIDX's shared cache, pgvector's
# shared_buffers — was being measured against a hot set small enough to hold
# entirely, which is the one thing the concurrency chart exists to tell apart.
# The pool now defaults to the whole test set, so a version 2 concurrency QPS
# and a version 3 one are throughput over a 500-query and a 10,000-query
# working set: the same workload over two different amounts of the corpus, and
# not the same quantity. Hence a bump, and hence 3 is deliberately absent from
# OPS_MEASUREMENT_EQUIVALENT_VERSIONS below.
#
# The recall that version 3 also added to concurrency records is new
# information rather than a changed number and would not on its own have
# earned a bump — a field that was None is now populated, and nothing a
# version 2 reader already trusted moves. The pool is what earned it.
#
# Version 3 ALSO counts only the queries that finish inside each concurrency
# window, which sits inside this version rather than earning a fourth one
# because no run has ever been measured at version 3 — the value was
# introduced and then extended inside one change set, so there is no version 3
# figure anywhere for a trimmed one to be incomparable with.
#
# VERSION 4 chose ef_search instead of declaring it. The ops pass now
# calibrates: it sweeps a grid on the index it has just built, picks the
# smallest ef_search reaching a target recall (0.95 by default), and runs
# concurrency, filtered and churn at that value instead of at the profile's
# `ops.ef_search`. A version 3 concurrency QPS is throughput at a declared ef
# — 100 for every engine — and a version 4 one is throughput at whatever ef
# puts THAT engine at the target. Two engines at ef_search 100 sit at recall
# 0.98 and 0.80, which is the ordinary spread of these implementations, so the
# version 3 figure and the version 4 figure are not two samples of one
# quantity: one holds the knob fixed and lets quality vary, the other holds
# quality fixed and lets the knob vary. That is the whole point of PR 2 and it
# is exactly what this constant exists to keep off one series.
#
# WHY A BUMP AND NOT JUST extra.ef_mode, which the records also now carry and
# which is the obvious reason to think one is unnecessary: this constant is
# what makes the REPORT REFUSE TO MIX, not what lets a reader tell two numbers
# apart by hand. Two concurrency records both stamped version 3, one `fixed` at
# ef_search 100 and one `calibrated` at 98, load together with no refusal and
# land on one chart — the comparison PR 2 exists to prevent. `extra.ef_mode`
# is the right diagnostic for which policy produced a number; it is not a
# guard, and `report/measurement_versions.py` reads this.
#
# It covers `--ef-mode fixed` too, which measures what version 3 measured. The
# version names the harness that wrote a record, not the flags it was given:
# per-run is the granularity the whole scheme is built on (see the per-phase
# argument below), and a constant that varied with a flag would have to be
# computed per record and could no longer be stamped from one place. A fixed
# run under version 4 is therefore labelled conservatively — reported as a
# different measurement from a version 3 run that it in fact matches. That is
# the safe direction of the two: this set is about refusing to merge, and the
# cost of an unnecessary refusal is one report per version, while the cost of a
# missing one is a published chart that is wrong.
#
# The cost this scheme knowingly pays: the bump invalidates comparison across
# every ops phase, including build cost and churn, which this change does not
# touch. The version is per-run, not per-phase, because a record names the
# harness that wrote it and splitting it per phase would mean four constants
# and four ways to forget one.
#
# VERSION 5 added a second filter column to the table. Every row now carries
# `ctag` -- a nearest-pivot cluster id -- beside `tag`, so that filtered
# search can be measured on a predicate that follows the vector space and on
# one that does not, both against the same graph. Three recorded quantities
# move, and none of them is the one the change is for:
#
#   * Ingest. Every engine writes one more scalar per row: an INT column in
#     the MySQL family, an int4 in PostgreSQL, a BSON field in Percona Search,
#     a hash field in Valkey. A version 4 `ingest_rows_per_s` and a version 5
#     one are throughput over rows of two different widths.
#   * `table_bytes`, and by more than the column. Wider rows first, and not
#     reliably four bytes per row at that: every engine here rounds a record
#     up to an alignment boundary, so four more bytes of payload can cost
#     nothing or cost eight. But for MariaDB and AliSQL `table_bytes` is
#     measured from a `t1.ibd` glob, and in InnoDB file-per-table the .ibd
#     holds the table's SECONDARY INDEXES too -- so for those two it now
#     absorbs an entire new B-tree on `ctag` as well, which at scale is of
#     the order of fifteen bytes per row rather than four. The direction is
#     the same and the magnitude is not, which is the whole reason this is a
#     version bump and not a correction a reader could apply by hand.
#   * `index_bytes`, but only where the scalar column lives inside the thing
#     being sized -- Valkey, where it is a resident-memory delta across
#     FT.CREATE and the index now carries a second NUMERIC attribute, and
#     Percona Search, where it is the size of mongot's Lucene directory and
#     the index now carries a second filter field. The MySQL family and
#     pgvector size only the vector structure (MariaDB's `t1#i#*` glob,
#     AliSQL's `vidx_*.ibd` glob, `pg_relation_size` of pgvector's HNSW
#     index), which this change does not touch, so their `index_bytes` is
#     unmoved -- and still not comparable across the bump, because the
#     version is per-run, as argued above.
#
# One thing that moved and is recorded NOWHERE, worth writing down in the
# block that enumerates what moved: pgvector's new `t1_ctag_idx` is counted
# by neither figure. `index_bytes` is `pg_relation_size('t1_embedding_idx')`,
# the HNSW index alone, and `table_bytes` is `pg_table_size('t1')`, which
# excludes indexes. So the same new structure lands inside `table_bytes` for
# MariaDB and AliSQL and in no number at all for pgvector. Pre-existing --
# `t1_tag_idx` was already invisible the same way -- and therefore not a
# regression this change introduced; it is an asymmetry in what the two
# sizing methods count, and it is larger now that there are two such indexes
# rather than one.
#
# The filtered numbers move too, and for a reason easy to miss when reading
# the diff: a version 5 run measured at `column="tag"` is not a re-run of a
# version 4 filtered measurement. The predicate is the same and the
# qualifying rows are the same, but the table is wider, pgvector carries a
# second B-tree and mongot's index carries a second filter path, so the
# per-query cost is being paid on a different object.
#
# WHY A BUMP AND NOT just `extra.filter_column`, which the records will also
# carry once PR 3's filtered workload lands: the same argument as for
# `extra.ef_mode` above. A diagnostic lets a reader tell two numbers apart by
# hand; this constant is what makes the report refuse to put them on one
# series. Two ingest figures both stamped version 4 -- one from a three-column
# table and one from a four-column table -- load together with no refusal and
# land on the same bar chart of rows per second.
#
# VERSION 5 ALSO COVERS PR 3's FILTERED WORKLOAD, and this paragraph exists
# because the enumeration above lists only the `ctag` COLUMN's effects, which
# is not all of what moved. Two of the changes are to `Record` fields rather
# than to `extra`, so a reader who took "the rest is additive" from the list
# above would be wrong:
#
#   * `selectivity` moves, materially, for the SAME request. It used to be
#     `threshold / 100.0` -- the nominal share -- and is now the counted share
#     of rows the predicate actually selects. For `tag` the correction is
#     small (glove's 1,183,514 rows make a 1% request deliver 1.000073%); for
#     `ctag` it is not, because the clusters are unequal: on a 2,000-row
#     corpus a nominal 1% records 0.0190 where the old code recorded 0.0200,
#     and a nominal 10% records 0.1085 against 0.1200. The PREDICATE is
#     unchanged -- same threshold, same qualifying rows -- so recall, QPS and
#     the percentiles beside it are the same quantity measured the same way.
#     Only the x axis stops lying. That is why it did not earn a bump of its
#     own, and why it has to be written down somewhere: a reader diffing two
#     archived `selectivity` values could otherwise read a label correction as
#     a change of measurement.
#   * `recall_at_k` can now be NULL, and can be a mean over fewer queries than
#     the profile asked for. `ops.filtered_budget_s` stops the query loop when
#     it is spent, so a filtered record may carry recall over 93 of 500
#     planned queries -- `queries_executed`, `extra.queries_planned`,
#     `extra.time_bounded` and `extra.insufficient_sample` say which, and the
#     report greys those points out. A point that managed no queries at all
#     records no recall rather than a zero. The precedent is version 3, which
#     likewise counts only the queries finishing inside each concurrency
#     window and sits inside that version rather than earning a fourth.
#
# Neither needs a bump, and the reason is the same in both cases: the quantity
# is unchanged and the record says what it is. Version 5 was introduced by
# T3b inside this same change set, so nothing has ever been MEASURED at a
# version 5 that lacked these, and the cross-version guard already refuses to
# chart a version 1 record beside a version 5 one.
#
# VERSION 5 ALSO COVERS PR 4, and this paragraph exists because PR 4 is the
# first change here whose effect is a NEW KIND OF RECORD rather than a new
# key, so the enumerations above do not describe it.
#
# §4.1 and §4.2 (T4a) added three driver hooks and a second record per churn
# fraction stamped `extra.maintenance_state`. Every field of that is an
# `extra` key, and the hooks' no-op defaults mean MariaDB, AliSQL and Valkey
# record exactly what they recorded before.
#
# §4.3 (T4b) is the one worth writing down, and it moves a recorded quantity:
#
#   * A THIRD KIND of churn record, carrying `extra.during_writes = true`. It
#     is aggregate reader throughput and percentiles measured by
#     `ops.churn_readers` clients while the writer re-inserts, and it uses no
#     new COLUMN -- checked field by field against the `Record` dataclass
#     below: `clients`, `qps`, the four latency fields and
#     `queries_executed` are the columns it fills, and everything else about
#     it is an `extra` key. `recall_at_k` is deliberately NULL on it, not
#     0.0: the corpus is mid-change while the readers run, so no ground truth
#     describes it. T3c made a null recall legitimate for a filtered point
#     that managed no queries and the report renders it; this is the same
#     null for a different reason.
#   * `extra.reinsert_rows_per_s` CHANGES WHAT IT MEASURES, and this is the
#     part a reader has to be told. The re-insert now runs against four
#     concurrent readers by default, so the rate on a churn step record is no
#     longer a write-only rate. That is the same class of move as version 3's
#     query pool -- the same workload against a different amount of load, not
#     two samples of one quantity.
#
# WHY THAT DOES NOT EARN A SIXTH VERSION, by version 3's own precedent: "no
# run has ever been measured at version 3 -- the value was introduced and
# then extended inside one change set, so there is no version 3 figure
# anywhere for a trimmed one to be incomparable with." The same is true of 5,
# and it is measured rather than assumed. The archived corpus the test suite
# reads (`SMOKE_RESULTS/`, the tree `tests/test_config.py:1499` points at)
# holds 239 ops record files and 2,134 ops records, and
# `extra.ops_measurement_version` is ABSENT from every one of them -- so all
# 2,134 are implicit version 1. THE LAST CLAUSE EXPIRED ON 2026-10-14: that
# day's `smoke` run stamps 158 records at version 5, so version 5 figures now
# exist and the version 6 block below is written against them, not around them.
#
# WHAT DOES KEEP THE TWO RATES APART, since the version cannot once a version
# 5 run exists: `extra.readers`, which every churn step record from a caller
# with a driver factory now carries, INCLUDING the value 0. §4.3 item 7 keeps
# `ops.churn_readers: 0` available precisely so the write-only rate stays
# measurable, and `config/profiles/dev.yml` runs it, so the distinction is a
# number in the record rather than a setting nobody uses. Note the limit of
# that: `extra.readers` is a diagnostic, not a guard. Two
# `reinsert_rows_per_s` figures both stamped version 5, one at 4 readers and
# one at 0, load together with no refusal -- exactly the gap the version-4
# block argues `extra.ef_mode` leaves. If a profile ever sweeps
# `ops.churn_readers` within one run, that gap becomes a real chart and the
# fix is a key the report groups on, not a bump that would invalidate every
# recall curve beside it.
#
# VERSION 6 -- THE CLIENTS ARE OS PROCESSES, AND THAT CHANGES WHAT EVERY
# MULTI-CLIENT NUMBER MEASURES. Until this version the concurrency sweep's
# clients and the churn workload's readers were Python THREADS in one
# interpreter, each with its own connection; from it they are separate
# processes. `docs/05-methodology.md` §7 set out the change and its cost and
# ended "none of that is attempted here"; `harness/workloads/clients.py` now
# does it and carries the mechanism.
#
# WHAT MOVES, by record field: `qps` and all four latency columns on every
# concurrency record with `clients > 1`; `extra.scaling_efficiency`,
# `extra.outstanding_concurrency`, `extra.served_concurrency` and
# `extra.scaling_status` with them; the churn during-writes record's
# aggregate reader throughput and percentiles, its p99 ratio, and
# `extra.reinsert_rows_per_s` -- which moves AGAIN, having already moved in
# version 5 for being measured against readers at all. A one-client
# concurrency point is the only multi-client-pool figure that does not move,
# and it moves in meaning anyway because it is the denominator of every
# efficiency beside it.
#
# WHY A BUMP RATHER THAN A FOOTNOTE, which is the test every version here has
# to pass. It is version 3's query-pool precedent exactly: the same workload
# against a different amount of load is not two samples of one quantity. And
# unlike version 5, THERE ARE NOW RECORDS TO BE INCOMPARABLE WITH. The
# 2026-10-14 `smoke` run (`SMOKE/results/smoke-20261014-075119/`, six
# engines on fashion-mnist-784 at M=16, a Xeon Gold 6230) wrote 158 ops
# records and every one of them is stamped 5. Those are the first version 5
# records that have ever existed, and they are exactly the ones a reader
# would most want to compare against the first version 6 run.
#
# THE EVIDENCE, because this bump is not a tidy-up. In that run
# `served = qps x p50` -- Little's law, the quantity `served_concurrency`
# computes -- per client, going from 1 client to 4:
#
#     mariadb      0.95 -> 0.238      pgvector   0.98 -> 0.974
#     mariadb123   0.95 -> 0.237      valkey     0.97 -> 0.969
#     alisql       0.97 -> 0.242      mongodb    0.99 -> 0.984
#
# The three engines pinned near one query in service are precisely the three
# that share MariaDB Connector/Python, and the three at their client count
# are the three that do not. The same run's churn writer managed 11.991
# rows/s against four readers where the same run's single-threaded load of
# the same corpus recorded `ingest_rows_per_s` 769.9 -- a factor of 64 inside
# one run, with nothing changed but whether four readers ran. So a version 5
# multi-client figure for mariadb, mariadb123 or alisql is a measurement of a
# client library's lock and a version 6 one is not, and putting them on one
# axis would read as those engines having become four times faster.
#
# AND THE INSTRUMENT STILL REFUSES WHAT IT REFUSED, which is the half a bump
# could quietly break. Five synthetic drivers of identical per-query cost
# through the real pool at 1 and 4 clients, `served / clients` at 4, threads
# then processes -- CITED from the table `docs/05-methodology.md` §7 publishes
# and `tests/test_g3_process_clients.py` builds, not re-measured here:
# lock-releasing 0.975 -> 0.983; **saturated server 0.977 -> 1.004 at
# efficiency 0.272, still not flagged**; a `ctypes.PyDLL` `usleep` holding the
# lock 0.243 -> 0.982; Python spin holding the lock 0.264 -> 0.995. Only the
# lock-holding rows moved, which is this version's whole claim.
#
# VERSION 7 -- THE SCALING STATUS AT ONE CLIENT WAS A STATEMENT ABOUT SKEW.
# `metrics/scaling.scaling_status` refuses when `max(served_fraction,
# throughput_fraction)` falls below `SERVED_CONCURRENCY_FLOOR` -- which at ONE
# client is `served_concurrency` below a half on its own, because
# `throughput_fraction` withholds the efficiency wherever `1 / clients` is not
# itself under the floor. So at one client the condition is `p50 < mean / 2`
# and nothing else: a one-client pool has NO CONCURRENCY TO FAIL TO ACHIEVE.
# One client cannot take turns with itself, so "the clients serialized on the
# interpreter lock" is impossible, and "the engine answers one query at a
# time" is not a defect at n=1 but the definition of n=1. From this version
# `clients == 1` is `ok` whenever the window measured anything, and `None`
# when it measured nothing.
#
# WHAT MOVES, by record field. FIVE status fields, measured by rebinding
# `pool_window.scaling_status` to a verbatim copy of the version-6 rule over a
# window at `served / clients` 0.18 -- every one `ok` after, every one
# `concurrency_not_achieved` before:
#
#     concurrency record     extra.scaling_status
#     during-writes record   extra.scaling_status
#     during-writes record   extra.reader_baseline_scaling_status
#     during-writes record   extra.reader_under_writes_scaling_status
#     churn STEP record      extra.reader_scaling_status
#
# all of them at ONE client or one reader, and none of them at four (the
# control moved nothing). The two `reader_baseline`/`reader_under_writes` keys
# are `PoolWindow.scaling_status` read at the same reader count
# (`churn_readers.py:_reader_extra`, written under both prefixes), and an
# earlier draft of this note omitted them. `extra.reader_scaling_status` is
# NOT a key of the during-writes record: `reader_contention_shape` is merged
# into the churn STEP record's extra by `churn.py`, which
# `churn_readers.py`'s own docstring already says is the one key of the four
# that reaches a record it does not write.
#
# NOTHING ELSE. No `qps`, no percentile, no efficiency, no multi-client
# status.
#
# WHY A BUMP RATHER THAN A FOOTNOTE. The value of a published field changes
# for records that already exist, with no change in any engine, so a reader
# putting a version-6 Validity section beside a version-7 one would see
# one-client points move from refused to published and have nothing to
# attribute it to.
#
# AND WHY IT IS LATE, which is the part a reader needs most. The predicate
# landed on master WITHOUT this bump, so version 6 has already labelled two
# different behaviours: a one-client concurrency record written before that
# merge and one written after it disagree about the verdict, and nothing in
# the data distinguishes them. Records stamped 6 are therefore ambiguous at
# `clients == 1` and only at `clients == 1`; everything else a 6 stamps is
# unaffected. There is no repair for the records already written -- the
# recomputation below is what a reader has instead.
#
# NO NEW FIELD IS NEEDED, which is unusual for a bump here and worth saying.
# On a VERSION 6 record -- the genuinely ambiguous ones, written by the code
# that has the one-client predicate -- `extra.served_concurrency` and
# `extra.scaling_efficiency` are both published unconditionally, so either
# verdict is recomputable and a reader can tell which one a 6 meant.
#
# NOT ON THE VERSION 5 RUN, and an earlier draft of this note said otherwise.
# Measured on `SMOKE/results/smoke-20261014-075119/`: 14 ops files, 158
# records, every one stamped 5, 28 of them concurrency records -- and
# `extra.served_concurrency`, `extra.outstanding_concurrency` and
# `extra.scaling_status` are absent from all 28. `extra.scaling_efficiency` is
# present on all 28 but cannot stand in at one client, where it is 1.0 by
# construction and `throughput_fraction` withholds it anyway:
# `achieved_fraction(None, 1, 1.0)` is None, so neither verdict is derivable
# that way. The route that DOES work there is Little's law off the published
# columns, `qps x latency_p50_ms / 1000`, which the version-6 block above and
# `tests/test_g3_process_clients.py` already use. Measured over that run's 14
# one-client points it gives 0.941-0.995, so on that run both rules say `ok`
# and no one-client verdict moves.
#
# THE EVIDENCE. A synthetic driver with a 30% cache-miss rate at 20x penalty,
# through the real `ClientPool` at a SINGLE client, measured
# `served / clients` of 0.1428, 0.1716, 0.1764, 0.1850 and 0.2050 over five
# builds, and version 6 returned `concurrency_not_achieved` for all five. So
# every run of a skewed workload printed "the clients serialized" about one
# client while withholding an efficiency that is 1.0 by definition, and
# `clients == 1` is in every profile -- `smoke`, `percona` and `dev` among
# them. AND NOTHING IN THE TREE EVER ASSERTED THAT REFUSAL AT ONE CLIENT.
# One-client call sites, in the keyword and the positional spelling -- the
# `\b` matters, or the first branch also matches `clients=16` --
#     grep -rnE 'scaling_status\([^)]*,[[:space:]]*1[,)]|clients=1\b' tests/
# -- assert `ok`, or `None` for a window that counted nothing. A site can
# also assert the verdict with neither spelling, by sweeping the client count
# (`test_skewed_latency_is_not_serialization.py` loops `(1, 2, 3, 4)` and
# asserts `{1: SCALING_OK, ...}`), so that grep is a starting point and not
# an enumeration; it too says `ok`. The only one-client refusals anywhere are
# the MUTANT arms, where a verbatim copy of the pre-change rule is run to
# show the two disagree. No count is given
# here on purpose: an earlier draft said "68 places", which does not
# reproduce under any counting, and a figure nobody can re-derive is worse
# than no figure. The old behaviour at n=1 was never intended by anything
# that wrote a test.
#
# VERSION 8 -- `storage_engine` STOPPED NAMING AN ENGINE THAT HAS NONE.
# `RunContext.record_defaults` stamped `IndexSpec.storage_engine` on every ops
# record. That field is a MySQL-family `ENGINE=` clause:
# `harness/main.py` defaults `--storage-engine` to `InnoDB` and
# `orchestrator/cli.py:_ops_storage_engines` passes `InnoDB` for every engine
# that is not MariaDB, so it arrived populated for all seven while only three
# of them put it in a CREATE TABLE. `EngineDriver.storage_engine_in_use` now
# decides, `MySQLFamilyDriver` overrides it with the clause its DDL emitted,
# and `Record.to_dict` drops the None -- so for pgvector, Valkey and Percona
# Search the field is ABSENT rather than false.
#
# WHAT MOVES, by record field: one, and it is not a measurement.
#
#     every ops phase     storage_engine   "InnoDB" -> absent
#                                          (pgvector, valkey, mongodb only)
#
# NOTHING ELSE. No `qps`, no recall, no percentile, no byte count, no ingest
# rate, no status. The workloads, the grids, the predicates, the ef policy and
# every number any of them records are unchanged, and the three MySQL-family
# engines' records are unchanged too -- `InnoDB`, or `MyISAM` where MariaDB's
# extra curve was swept.
#
# Measured on `full-20261015-215006`: 516 of its 1,168 ops records carried the
# false value, against the same run's honest `heap` (200), `memory` (100) and
# `wiredTiger` (37) on the `recall_qps` records the ann overlay modules write.
# So one engine reported two different storage engines inside one run.
#
# WHY A BUMP IS OWED ANYWAY, which is the whole difficulty: no measured
# quantity moves, and that is true and insufficient. `storage_engine` is not a
# label, it is an IDENTITY COMPONENT -- of `OPS_UNIT_FIELDS` in
# `report/generate.py`, `OPS_UNIT_KEYS` in `workloads/churn_readers.py`,
# `report/loaders.py`'s
# `build_signature` and `memory_stem`, and through the first of those
# `charts.series_key`. Changing it re-partitions the record set, and the
# charter at the top of this block is "any change that alters what an ops
# number MEANS", which a re-partitioning does: the number keeps its value and
# changes which series it belongs to.
#
# WHAT THAT PRODUCES WITHOUT A BUMP, measured on two records of ONE pgvector
# configuration differing only in the pre/post spelling of this field:
#
#     refuse_mixed_measurement_versions -> no refusal; both stamp version 7
#     build_signature pre  (..., 16, 'post', 'InnoDB', 200, None)
#     build_signature post (..., 16, 'post',  None,    200, None)
#     charts._grouped      -> 2 series, legend
#                             ['PostgreSQL (pgvector)',
#                              'PostgreSQL (pgvector) / InnoDB']
#
# The published legend reads `PostgreSQL (pgvector) / InnoDB`: the exact claim
# this change exists to withdraw, reproduced across the version boundary, with
# nothing in the data to attribute it to. And a reader cannot discount it,
# because for MariaDB a second storage-engine series IS a real within-engine
# comparison.
#
# THE PATH IS ORDINARY, NOT CONTRIVED. `--profile full --resource-pass both`
# is ~1,135 h -- the figure `orchestrator/estimate.py` computes and CLAUDE.md's
# "Cost awareness" publishes, worst case once querying is counted -- so
# resumption is the norm, and `report/measurement_versions.py` names this
# hazard in its own words -- "a resumed run that appended to a file written by
# older code". That figure read ~1,066 h here for as long as the estimator
# excluded Percona's ingest: its rate was measured and unmodelled, so
# `engines_without_ingest_rate` named the engine instead of folding it in, and
# both halves of the pair were a floor by 69.83 h. Nothing asserts the number
# in this comment, which is exactly why it rotted while CLAUDE.md's copy
# moved; the point it supports does not depend on the value, only on it being
# long enough that nobody runs it in one sitting.
#
# A run whose pgvector M=8 unit completed before this change and
# whose M=16 unit runs after it puts both spellings in one run directory;
# `Manifest.set_config` carries the existing version forward, so without a
# bump no refusal fires and `series_key` splits pgvector on every chart that
# groups through it.
#
# THE PRECEDENT IS VERSION 7, directly above: "The value of a published field
# changes for records that already exist, with no change in any engine, so a
# reader putting a version-6 Validity section beside a version-7 one would see
# one-client points move ... and have nothing to attribute it to." That is
# this shape exactly, and the block beside it records what shipping such a
# change late cost -- "There is no repair for the records already written."
#
# WHY VERSION 5's `selectivity` DECLINATION DOES NOT APPLY, since it is the
# closest-looking escape and it was used to argue for one. That note declines
# a bump OF ITS OWN, and rests on a second leg this change does not have:
# version 5 was introduced by T3b inside that same change set, so nothing had
# ever been measured at a version 5 lacking the correction. There was a bump
# in that change; the label correction simply rode it. This change set
# proposed none, so there is nothing for this correction to ride.
#
# AND WHY `_ops_iterative_scan`'s "lands beside rather than on top of" ARGUES
# THE OTHER WAY, which is the second escape. There, two values of
# `extra.iterative_scan` are two genuinely different measurements, so landing
# beside each other is the SAFETY property. Here the two values are one
# measurement of one configuration, so landing beside each other is precisely
# the harm.
#
# THE COST, stated rather than discovered later: `report/generate.py` prints
# and returns 2 for a directory holding both versions, and
# `orchestrator/manifest.py` warns on a resume of an archived run. An
# archived single-version run still reports -- the refusal compares the
# records against their OWN manifest, not against this constant -- so
# `full-20261015-215006` regenerates unchanged. What needs a report per
# version is a directory this change's harness appended to.
#
# VERSION 9 -- WHAT `full-20260825` PUBLISHED THAT ITS RECORDS COULD NOT
# BACK. One bump for all of it, because every item below changes what an
# existing field means or how a published figure was taken, and a run that
# straddled any one of them would put two measurements under one label.
#
#     F2  concurrency  `scaling_status` is judged on `outstanding_concurrency`
#                      (exact, mean-based) instead of `served_concurrency`
#                      (queries x p50 / window). The median was chosen while
#                      the clients were threads, when a lock wait inside a
#                      span made the mean blind; with process clients (v6)
#                      that reason is gone and the median only undercounted
#                      heavy tails -- MariaDB 12.3 at 32 clients, p50 34 ms,
#                      p99 1,043 ms. A low median is now `extra.scaling_note:
#                      tail_dominated`, a note and never a refusal; the
#                      residue (BOTH in-service estimators under the floor)
#                      still refuses. `extra.scaling_efficiency` is null with
#                      `extra.efficiency_reliability: baseline_not_credible`
#                      where `scaling.baseline_is_credible` says the one-client
#                      median was cold (p50_1 / p50_n > 1.4); new keys
#                      `baseline_p50_ratio`, `window_started_at`,
#                      `window_ended_at`. The report's "in service" cell
#                      prints the mean first and the median as an estimate.
#     F7  concurrency  one discarded warm-up window, same clients and length,
#                      before each client count's first repeat
#                      (`extra.warmup_window_qps`, never selected) -- the
#                      first repeat at 1 client ran cold, +/-22-24% spread.
#     F8  calibration, `recall_at_k` is ann-benchmarks' distance threshold
#         concurrency, (`datasets.recall_threshold`, the rule of
#         filtered,    `report/loaders.py:knn_recall`), where it was an ID
#         churn        overlap under the same name; the overlap is kept as
#                      `extra.recall_id_overlap`, and every ops recall says
#                      which in `extra.recall_kind`. A pre-v9 ops record is
#                      read as `id_overlap`; `report/crosscheck.py` compares
#                      only matching kinds. Equal on a corpus with no ties at
#                      the k-th distance, different where there are.
#     F9  churn        re-insert over `extra.reinsert_writers` connections,
#                      default the run's ingest `load_threads` (was 1 against
#                      an 8-thread load), with `ingest_threads` beside it;
#                      `reinsert_rows_per_s_accepted` and
#                      `reinsert_rows_per_s_indexed` (rows / time until
#                      `wait_index_synced` returned); the sync wait
#                      also runs before the baseline and after the deletes
#                      (`index_sync_before_baseline_s`,
#                      `index_sync_after_delete_s`) -- Percona Search's
#                      mongot was measured before it had caught up; the
#                      budget default is 7200 s (was 1800).
#     F10 build, churn `index_bytes` on `index_build` is read after a settle
#                      step (mongot: two directory sums 30 s apart within
#                      1%), with `extra.index_bytes_after_build`,
#                      `index_size_settle_s`, `size_read_at`,
#                      `index_vector_format`, `index_includes_vectors` and,
#                      for mongot, `index_bytes_by_kind` -- 15.1 GiB after
#                      the load and 7.2 GiB at the churn baseline were one
#                      index before and after a merge.
#     F10 every phase  top-level `quantization` from `record_defaults`.
#     F4  mongodb      WiredTiger's cache is set from the resource pass's
#                      buffer share (`--wiredTigerCacheSizeGB`) instead of
#                      the server default, 57-64 GB on that run beside a
#                      42.3 GB InnoDB buffer pool. Every MongoDB ops number
#                      moves with it. ANN_MEASUREMENT_VERSION is NOT bumped
#                      for the ann side of this: PR #75 moved it to 3 and no
#                      run has been made since, so no version-3 ann figure
#                      exists to be confused with.
#     F11 (sampler)    `mem-*.jsonl` rows carry cgroup cpu/io pressure (None
#                      when unreadable, never 0); not an ops record field,
#                      listed because the concurrency table's new server
#                      columns join on it.
#
# WHAT MOVES FOR AN ARCHIVED RUN: nothing -- the report reads a v8 record's
# verdict, recall and size the way they were written, and labels the recall
# `id_overlap`. What a v9 run cannot be is a continuation of a v8 one, which
# `refuse_mixed_measurement_versions` now enforces for 8 against 9.
#
# VERSION 10 -- THE OPS PASS CAN MEASURE THE FRONTIER'S BUILD. A reviewer of
# the published benchmark wrote that comparing engines at one M is
# meaningless, because an identically named parameter is a different amount
# of work in a different algorithm, and the headline moved to each engine's
# Pareto frontier over its whole swept grid. With `ops.build_selection:
# frontier` the orchestrator now picks the ops build per (engine, dataset,
# pass) off that ann phase's frontier -- the build fastest at the ops
# calibration target -- instead of building the declared reference build
# (`orchestrator/ops_build_selection.py`, `report/ops_build_choice.py`).
# What moves:
#
#     build cost, concurrency, filtered, churn, ef_calibration: measured on
#                      the chosen M / ef_construction / quantization, which
#                      under a frontier profile is in general NOT the M=16
#                      every v9 tuned-complete record describes. A v9 and a
#                      v10 concurrency figure for one engine can be two
#                      different indexes under one engine name.
#     every record     `extra.ops_build_selection` -- mode, status
#                      (`selected`, `below_target`, `fallback_reference`),
#                      source (`frontier@0.95`), target_recall, the chosen
#                      build and the ann point it was chosen from -- when the
#                      profile asked for a selection. Absent in `reference`
#                      mode, whose build is unchanged.
#
# WHY A BUMP WHEN `reference` MODE MEASURES WHAT v9 DID: the version names
# the harness that wrote a record, not the flags it was given -- the
# argument version 4 makes for `--ef-mode fixed`. A run that straddled the
# change could hold v9 records of M=16 and v10 records of the frontier's
# build for one engine and dataset, and only the version keeps the report
# from charting them as one series (`refuse_mixed_measurement_versions`
# refuses 9 against 10, as it refuses 8 against 9).
#
# VERSION 11 -- PERCONA SERVER'S GRAPH IS WARMED BEFORE ANYTHING IS TIMED.
# Percona loads its HNSW graph lazily (`vec_runtime_load` reads only the
# entry point at the pin) and, per the server team, the ALTER a `post` build
# issues evicts the table, so every post build handed calibration a cold
# graph: `Innodb_hnsw_memory_used` read 0 on the `index_build` record in 95
# of 95 post builds, against 207,407,472 B after an incremental build of the
# same 60k corpus. `build.run` now runs one discarded pass over
# calibration's query set at the top ef for a driver declaring
# `EngineDriver.warm_graph_after_build` -- `PerconaDriver` alone -- in both
# build modes (`harness/workloads/graph_warmup.py`). What moves:
#
#     ef_calibration  Percona's per-point latency and QPS, most at the first
#     (percona)       grid points after a post build, which were measured
#                     while the graph was being faulted in. Recall and the
#                     chosen ef are not expected to move: a cold node is
#                     read, not skipped.
#     concurrency,    Percona's, to the extent any of their windows still
#     filtered, churn saw a partly cold graph -- small, since calibration
#     (percona)       ran first, but not zero and not measured.
#     index_build     `extra.graph_warmup` (queries, ef_search, effective_ef,
#     (percona)       wall_s, status, the after-warm-up `server_status`);
#                     `extra.server_status` keeps its meaning. New
#                     information, not a changed number.
#     every later     `extra.graph_warmup_status` on Percona's records after
#     record          the build. Absent for every other engine.
#
# WHY A BUMP WHEN EIGHT OF NINE ENGINES DO NOT MOVE: the charter at the top of
# this block is the example this change is -- "a concurrency figure measured
# with a per-client warmup is not the same quantity as one measured
# without" -- and version 9's F7 (one discarded concurrency warm-up window)
# rode a bump for the same reason. A resumed run straddling this change
# would hold a cold-graph and a warm-graph Percona calibration curve for one
# engine and dataset under one version, and
# `refuse_mixed_measurement_versions` is the only thing that keeps them off
# one series. The version
# names the harness, not the engine (version 10's argument): the refusal
# costs the other engines a re-run of a straddling directory, and buys that
# no Percona figure is silently mixed. ANN_MEASUREMENT_VERSION is NOT bumped:
# the ann path has its own warm-up (`vb_warmup.py`) and nothing in it moved.
OPS_MEASUREMENT_VERSION = 11

#: Key under `Record.extra` that `Recorder.write` stamps the version into.
#: Lives in `extra` rather than as a column because SCHEMA_VERSION stays 1.
OPS_MEASUREMENT_VERSION_KEY = "ops_measurement_version"

#: What an ops record with no recorded version was measured by.
IMPLICIT_OPS_MEASUREMENT_VERSION = 1

#: The first version that was ever written into a record. It changed nothing
#: measured -- it exists because version 1 was never written down, and a
#: version nobody records cannot be compared against a later one.
#:
#: Distinct from OPS_MEASUREMENT_VERSION, which is whatever the harness is
#: stamping today and moves on every bump. This one never moves: it names a
#: historical fact about version 2, not the current state of the code.
FIRST_RECORDED_OPS_MEASUREMENT_VERSION = 2

# Versions that denote the SAME measurement, despite being different numbers.
# A version 1 number and a version 2 number are the same quantity, so a run
# holding both needs a report per version, not a re-measurement.
# `docs/05-methodology.md` says exactly this and `tests/test_config.py` pins
# the two together -- the refusal text used to tell operators to re-measure
# data the docs call comparable, on the only version pair that occurs today.
#
# DERIVED from the two constants above rather than written as a literal, and
# that is the whole safety property. A set spelled `frozenset({1, 2})` is an
# extension point whose only lock is a comment asking people not to use it:
# `{1, 2, 4}` would be a one-character edit asserting that version 4 measures
# what version 1 did, and a subset-style test cannot see a set growing. Built
# this way, a third member cannot be added without first naming a third
# concept and saying in its own docstring why that version changed nothing --
# which is the deliberateness the comment used to merely request.
#
# So: a bump that changes what a number means -- the per-client warmup, a
# different filtered predicate -- simply cannot appear here, and the refusal
# gives every pair this set does not cover the "two different measurements"
# wording.
OPS_MEASUREMENT_EQUIVALENT_VERSIONS = frozenset({
    IMPLICIT_OPS_MEASUREMENT_VERSION,
    FIRST_RECORDED_OPS_MEASUREMENT_VERSION,
})


def ops_versions_measure_the_same_thing(versions: Iterable[Any]) -> bool:
    """Whether every version given denotes the same measurement.

    False for anything unrecognised, including a malformed value, so the
    general "these are different measurements" wording is what an unknown
    version gets. Saying two numbers are comparable is a claim; saying they
    may not be is the safe default.
    """
    found = list(versions)
    if not found:
        return False
    for version in found:
        # A hand-edited manifest can hold a list or a dict here, and `in`
        # against a frozenset hashes what it is given.
        if not isinstance(version, int) or isinstance(version, bool):
            return False
        if version not in OPS_MEASUREMENT_EQUIVALENT_VERSIONS:
            return False
    return True

# Phases a record can belong to. Used by the report generator to route records
# to the right chart without inspecting which fields happen to be populated.
PHASE_RECALL = "recall_qps"       # ann-benchmarks Pareto point
PHASE_INGEST = "ingest"           # rows loaded, no index or index maintained
PHASE_BUILD = "index_build"       # index construction cost
PHASE_CONCURRENCY = "concurrency"  # multi-client QPS / latency
PHASE_FILTERED = "filtered"       # vector search with a scalar predicate
PHASE_CHURN = "churn"             # recall after update/delete cycles
# The ef_search calibration sweep: recall and QPS at each grid point, written
# so the choice of ef is auditable. Its own phase and not PHASE_RECALL, because
# these are not Pareto points: they are single-client probes against the index
# the ops pass just built, and a chart that drew them as recall/QPS results
# would put twelve points per engine on a curve measured with a different
# method, a different container and a different query count.
PHASE_CALIBRATION = "ef_calibration"

ALL_PHASES = (
    PHASE_RECALL, PHASE_INGEST, PHASE_BUILD,
    PHASE_CALIBRATION, PHASE_CONCURRENCY, PHASE_FILTERED, PHASE_CHURN,
)

#: Keys under `extra` that name the ef policy a record was measured under.
#: Stamped by `Recorder`, not written by the workloads -- see
#: `stamp_run_extra`.
EF_MODE_KEY = "ef_mode"
TARGET_RECALL_KEY = "target_recall"
CALIBRATION_STATUS_KEY = "calibration_status"


@dataclass
class Record:
    # --- identity -----------------------------------------------------
    run_id: str
    engine: str
    phase: str
    dataset: str
    timestamp: str

    engine_version: Optional[str] = None
    engine_tag: Optional[str] = None
    resource_pass: Optional[str] = None      # "normalized" | "tuned"
    storage_engine: Optional[str] = None
    metric_space: Optional[str] = None       # "angular" | "euclidean"
    march: Optional[str] = None

    # --- index configuration -----------------------------------------
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None
    build_mode: Optional[str] = None         # pgvector: "post" | "incremental"
    #: Percona Search's build axis ("none" | "scalar" | "binary"), stamped on
    #: every ops record since ops measurement version 9 -- the ann records
    #: already carried it, and an ops record without it could not be told
    #: which of the two tuned/normalized indexes it measured. None (and so
    #: absent, `to_dict` drops it) for every engine without the knob.
    quantization: Optional[str] = None
    k: Optional[int] = None

    # --- quality ------------------------------------------------------
    recall_at_k: Optional[float] = None
    vector_index_used: Optional[bool] = None

    # --- throughput / latency ----------------------------------------
    clients: Optional[int] = None
    qps: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    queries_executed: Optional[int] = None

    # --- build cost ---------------------------------------------------
    ingest_rows_per_s: Optional[float] = None
    ingest_wall_s: Optional[float] = None
    ingest_threads: Optional[int] = None
    build_wall_s: Optional[float] = None
    build_cpu_s: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    index_bytes: Optional[int] = None
    table_bytes: Optional[int] = None
    rows: Optional[int] = None

    # --- workload shape ----------------------------------------------
    selectivity: Optional[float] = None      # fraction of rows passing the filter
    churn_fraction: Optional[float] = None

    # --- escape hatch -------------------------------------------------
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {}}


def stamp_measurement_version(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a record dict carrying the version that measured it.

    Applied in `Recorder.write`, not in `RunContext.record_defaults`. The
    defaults dict is the obvious home — it is the choke point every workload's
    provenance already travels through, and it exists precisely so a new
    workload cannot drop a field. But four of the workloads build their record
    as `Record(**common, ..., extra={...})`, so an `extra` key in `common`
    raises `TypeError: got multiple values for keyword argument 'extra'`.
    Merging the two would mean editing every workload, and then a fifth
    workload added later would still be free to forget.

    `Recorder.write` is the only point every ops record provably passes
    through, which is why the ALL_PHASES check lives there too. Stamping here
    makes "every ops record carries the version" true by construction rather
    than by convention.

    Takes and returns a plain dict, and copies both it and its `extra` rather
    than writing into either. Note what that does and does not protect: the
    `Record` the caller still holds is protected by `to_dict()`, which goes
    through `dataclasses.asdict` and therefore deep-copies `extra` before this
    function ever sees it — not by anything here. What the copy here buys is
    that this function is safe for any caller, including a future `to_dict`
    that stops deep-copying to save the allocation. `TestOpsMeasurementVersion`
    guards both halves separately, because an earlier version of that test
    asserted the record was untouched and could not fail: asdict was doing the
    work and a stamp written straight into `payload` passed it.
    """
    extra = dict(payload.get("extra") or {})
    # Unconditional, not setdefault. A record reaching this Recorder was
    # produced by this code, so this code's version is the honest answer; a
    # caller-supplied value would be a claim about someone else's measurement,
    # and a merge tool replaying old records through a new Recorder would
    # relabel them as new ones if it were allowed to carry its own version in.
    extra[OPS_MEASUREMENT_VERSION_KEY] = OPS_MEASUREMENT_VERSION
    return {**payload, "extra": extra}


def stamp_run_extra(payload: Dict[str, Any],
                    run_extra: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a record dict carrying facts that are true of the whole run.

    The second stamp applied in `Recorder.write`, and it exists for exactly the
    reason `stamp_measurement_version` does: `RunContext.record_defaults` is
    the obvious home for run-wide provenance and cannot carry an `extra` key,
    because four workloads splat that dict into `Record(**common, ...,
    extra={...})` and a duplicate keyword argument is a TypeError.

    What travels this way is the ef policy -- `ef_mode`, `target_recall`,
    `calibration_status`. A concurrency figure measured at a calibrated
    ef_search and one measured at the profile's literal are not the same
    quantity, and the record has to say which it is without the reader having
    to find the calibration records that preceded it. Stamping here makes that
    true for every workload, including the three this change does not touch.

    Unconditional, like the version stamp and for the same reason: a record
    reaching this Recorder was produced by this run, so this run's policy is
    the honest answer. Keys whose value is None are left out rather than
    written as nulls -- in `fixed` mode no target was pursued and no
    calibration ran, and a record carrying `target_recall: null` beside an ef
    nobody chose would be a claim about a calibration that never happened.
    `ops.pgvector_iterative_scan` spent 69 records saying null; absence is the
    honest encoding of "there was none".
    """
    if not run_extra:
        return payload
    extra = dict(payload.get("extra") or {})
    extra.update({k: v for k, v in run_extra.items() if v is not None})
    return {**payload, "extra": extra}


class Recorder:
    """Append-only JSON-lines writer, safe for use from multiple threads."""

    def __init__(self, path: str, run_extra: Optional[Dict[str, Any]] = None):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1, encoding="utf-8")
        self._run_extra: Dict[str, Any] = dict(run_extra or {})

    def declare_run_extra(self, **values: Any) -> None:
        """Add facts every record written from now on should carry.

        Deliberately not a constructor-only argument: `calibration_status` is
        not known until the calibration sweep has finished, and the sweep's own
        records are written before it exists. A per-point record therefore
        carries no status, which is correct -- at the time it was written there
        was not one.
        """
        with self._lock:
            self._run_extra.update(values)

    def write(self, record: Record) -> None:
        if record.phase not in ALL_PHASES:
            raise ValueError(f"unknown phase: {record.phase}")
        # After to_dict(), which drops an empty `extra`, so the stamps survive
        # that filter instead of being removed by it.
        with self._lock:
            run_extra = dict(self._run_extra)
        line = json.dumps(
            stamp_run_extra(stamp_measurement_version(record.to_dict()),
                            run_extra),
            sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def write_many(self, records: Iterable[Record]) -> None:
        for record in records:
            self.write(record)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_records(path: str) -> Iterator[Dict[str, Any]]:
    """Read a JSONL file, skipping any truncated trailing line.

    A run killed mid-write can leave one partial line. Everything before it is
    still valid data and must remain readable.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Only tolerate damage at the very end of the file.
                remainder = fh.read().strip()
                if remainder:
                    raise ValueError(
                        f"{path}: malformed JSON at line {lineno} with data following it"
                    )
                return


def load_all(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        if os.path.exists(path):
            out.extend(read_records(path))
    return out
