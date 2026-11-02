"""PR 5 §5.2: do the two measurement paths agree about the same index?

The recall/throughput sweep and the ops calibration sweep measure one index
with different code -- ann-benchmarks in-process against the whole test set,
the ops harness over TCP against the first 1,000 queries. Where a
configuration was measured by both, the two recalls should agree to within
the ops measurement's own uncertainty, and where they do not the disagreement
is a fact about this framework rather than about the engine.

**Its own module for the reason `report/churn_state.py` is.** What this file
owns is a *matching rule*, and a matching rule is the thing this repository
keeps getting wrong: the filter-strategy ratio divided a narrow point from
one ops unit by a wide point from another and published 50.0x, a figure
neither unit produced, and `charts.churn_retention` paired records from two
different maintenance states. Both read as findings. A rule that lives in one
module is a rule a test can mutate in one place, which is what
`tests/test_t5b_stats.py` does to every axis of the key below.

It also keeps `report/generate.py` from growing another 565 lines; that file
was already past the repo's stated 800-line maximum before this task.

**Imports `harness.metrics.stats` in function bodies, never at module
scope.** `harness/` and `report/` are bind-mounted into different containers
at different paths (`/opt/harness`, `/vb/report`), and `CLAUDE.md` records an
`orchestrator` import from `report/` crashing report generation at the final
step of a twenty-hour run. `loaders.py` and `measurement_versions.py` already
import `harness.metrics.records` function-locally in eight places for this
reason. `tests/test_t5b_stats.py` walks this module's AST to check no
top-level `harness` import creeps back in, and runs the report under the
container's own layout with `harness` blocked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# PR 5 §5.2: the ann path and the ops path measured the same index. Do they
# agree?
# ---------------------------------------------------------------------------

#: §5.2 item 1's match key, verbatim: "the same engine, dataset, resource
#: pass, M, ef_search, quantization and build mode".
#:
#: **How this reconciles with `OPS_UNIT_FIELDS` above**, which is the other
#: key in this file and must not become a second convention for the same job.
#: They answer different questions. `ops_unit` names the axes `cli._run_unit`
#: LOOPS, so holding them constant is what makes two ops points comparable
#: with each other; this names the axes the two measurement PATHS have to
#: share before one path's recall can be set against the other's. So:
#:
#: * `m`, `build_mode` and `resource_pass` are in both. They are the axes a
#:   profile sweeps on both paths.
#: * `ef_search` is added, and it is the one that would make this check
#:   nonsense if it were missing. It is not an ops-unit axis -- the
#:   calibration sweep varies it INSIDE one unit, writing a record per grid
#:   point -- and recall is exactly what varies along it. `ann_pass.py`'s own
#:   grid comment is the measurement: MHNSW and VIDX at M=16 "already
#:   returns recall 0.97-0.98" at their LOWEST query effort, so the ef axis
#:   is where their recall moves at all. A key without `ef_search` would set
#:   one path's recall at one effort against the other's at a different
#:   effort, and the |delta| flag would fire on the sweep rather than on a
#:   disagreement between the two paths.
#: * `quantization` is added because it is an index property with no column
#:   of its own, and `loaders.build_signature`'s docstring records what that
#:   costs: Percona Search's unquantized and scalar curves collapsed into one
#:   series, which made eight real measurements look like duplicates and
#:   invented six inversions. TODAY IT IS None ON BOTH SIDES for every engine
#:   in every shipped profile, and saying so is the point: `harness/main.py`
#:   takes `--quantization` into `IndexSpec` but `context.record_defaults`
#:   does not put it on a record, and the mongodb ann module DOES write it
#:   into the HDF5 attrs (`module.py:524`) but `report/loaders.py`'s own ann
#:   loader does not copy it onto the record. So this is a guard against
#:   the day either path starts carrying it, not a live discriminator, and a
#:   reader must not be told the check matched on quantization when both
#:   sides read None.
#: * `storage_engine` is OMITTED, which is the only subtraction, and the
#:   reason is measurable rather than stylistic: the two paths do not name
#:   the same storage comparably. The ops path records one only where there
#:   is one. `harness/main.py:120` still defaults `--storage-engine` to
#:   `InnoDB`, because the MySQL family's DDL needs a value, but
#:   `EngineDriver.storage_engine_in_use` decides what reaches a record and
#:   returns None for the three engines that have no such concept -- so
#:   pgvector's, Valkey's and Percona Search's ops records carry the field
#:   not at all, while their ann modules do name something: "heap"
#:   (`algorithms/pgvector/module.py:324`), "memory" and "wiredTiger". The
#:   mismatch is therefore an absent field against a name, and it is the same
#:   three engines it always was. Put it in the key and they never match --
#:   the check would be empty for three of seven (Percona SERVER reports
#:   `InnoDB` like the rest of the MySQL family, so it is not a fourth here).
#:   Before `OPS_MEASUREMENT_VERSION` 8 those three ops records said
#:   `InnoDB`, a value the harness stamped on engines that have no storage
#:   engine, so the mismatch was two different names; the omission was
#:   equally necessary then and is unchanged by the repair. It is
#:   instead used to PAIR inside a
#:   key (see `_normalised_storage`) and is carried on every row, so two ops
#:   units differing only in storage engine stay two visible rows instead of
#:   one pooled one. That is H-1's lesson applied to the axis the key drops.
#:
#: `cli._run_unit`'s AST pin in `tests/test_t3c_filtered.py` is unaffected and
#: must stay so: it compares the function's loop targets against
#: `OPS_UNIT_FIELDS`, and neither `ef_search` nor `quantization` is a loop in
#: `_run_unit`. Adding either to THIS tuple cannot satisfy that pin, and
#: `tests/test_t5b_stats.py` asserts the two keys' relationship directly so
#: the reconciliation above is checked rather than merely written down.
CROSSCHECK_KEY_FIELDS = ("engine", "dataset", "resource_pass", "m",
                         "ef_search", "quantization", "build_mode")

#: §5.2 item 2's floor. "max(0.01, 2 x the ops CI half-width)" -- so one
#: recall point of disagreement is tolerated however tight the interval is,
#: because "The two paths use different query subsets (all test vectors vs
#: the first 1,000)".
CROSSCHECK_MIN_TOLERANCE = 0.01

#: Relative slack for deciding WHICH branch of that `max` won, and only for
#: the label. The interval is stored in a record as two rounded bounds, so a
#: half-width that was exactly 0.005 comes back as 0.0050000000000000044 and
#: `2 x half > 0.01` is then True by one part in 1e18 -- which labelled a row
#: "CI" whose tolerance was the floor to every digit shown. The tolerance
#: itself is unaffected either way; the label is a claim about how the row
#: was judged and it was wrong.
CROSSCHECK_BASIS_SLACK = 1e-9


def _crosscheck_quantization(record: Dict[str, Any]) -> Any:
    """The quantization a record declares, or None where it declares none.

    `extra.quantization` and nothing else. The value is reachable from a
    result FILENAME on the ann side (`..._quantization_scalar`) and
    `charts._config_tokens` already parses that shape, but a cross-check that
    read one side from a parsed filename and the other from a field would be
    comparing two different derivations -- and when the two disagreed, the
    disagreement would be in this function rather than in the engines.
    """
    return (record.get("extra") or {}).get("quantization")


def build_mode_axis_engines(ann_records: List[Dict[str, Any]]) -> frozenset:
    """Engines whose RECALL path declares a build mode, so the axis is real.

    **The second asymmetry between the two paths, measured, and the one §5.2
    item 1 does not anticipate.** The item asks for a match on build mode,
    and four of seven engines cannot supply one:
    `overlay/.../algorithms/{mariadb,mariadb123,alisql,percona}/module.py`
    write no
    `build_mode` attribute at all, so `report/loaders.py`'s ann loader reads
    None for them -- while `harness/main.py` defaults `--build-mode` to "post"
    and
    `context.record_defaults` stamps it on EVERY ops record. Taken literally,
    the key is then `(..., None)` on one side and `(..., "post")` on the
    other and the check is empty for MariaDB, MariaDB 12.3, AliSQL and
    Percona Server.

    **Percona Server is the fourth for a different reason from the other
    three, and it is the one worth knowing about.** MHNSW and VIDX have no
    separable bulk build, so a build mode would be meaningless for them.
    Percona Server HAS one -- `PerconaDriver` splits `create_schema` and
    `create_index` on `IndexSpec.build_mode`, and `config/profiles/
    percona.yml` runs both modes through the ops path -- but its ann module
    does not read `build_mode`, which is why `orchestrator/ann_pass.py`
    refuses `ann.percona_build_modes` rather than emitting two identical
    curves. So the axis is real for this engine on the OPS path and silent on
    the recall path, and dropping it here is still the right call: the ann
    records genuinely say nothing about it. Giving the overlay module a
    `build_mode` the way pgvector's and valkey's have one would make this
    engine supply the axis, and is the change that would close both gaps at
    once.

    That ops value is not a measurement of how the index was built, and the
    mechanism runs the opposite way round from how an earlier revision of
    this docstring described it. `IndexSpec.build_mode` is never written
    after `harness/main.py` sets it from `--build-mode`;
    `context.py:record_defaults` copies it onto every record verbatim. What
    decides how the index is ACTUALLY built is a local in
    `workloads/build.py`: `incremental =
    driver.capabilities().get("incremental_index", True)`, which the field
    can only RAISE (`if index.build_mode == "incremental": incremental =
    True`) and never lower. So for MHNSW and VIDX `incremental` is True from
    the driver's own capabilities while the record still says `post`, and the
    field describes what was requested rather than what happened.

    It is the same non-functional default as the `InnoDB` that
    `render._STORAGE_OVERRIDES` exists to paper over -- and `storage_engine`
    is the one field §5.2 item 1 DOES omit, for exactly this reason. So the
    two asymmetries are one asymmetry, and the item names only the half of it
    that was known.

    Rather than hard-code which engines have two build modes -- `CLAUDE.md`
    is explicit that there is no engine list in Python and that `report/` may
    not import `orchestrator` -- this reads it off the records: an engine
    whose recall sweep declares a build mode anywhere has the axis, and one
    whose sweep never mentions it does not. So pgvector and Valkey keep a
    real `post`/`incremental` discrimination, Percona Search keeps `async`,
    and the MySQL family's meaningless default stops blocking a match.

    The conservative direction is the one taken: the axis is dropped only
    where the recall path is SILENT about it, never where the two paths
    disagree. A pgvector ops `incremental` unit still fails to match a
    pgvector ann `post` point, which is correct -- those are two different
    indexes.
    """
    return frozenset(r.get("engine") for r in ann_records
                     if r.get("build_mode") is not None)


def crosscheck_key(record: Dict[str, Any],
                   build_mode_engines: Optional[frozenset] = None) -> tuple:
    """The §5.2 item 1 key for one record, from either measurement path.

    `build_mode_engines` is `build_mode_axis_engines(...)`. Passing None
    reads the record's `build_mode` verbatim, which is the literal reading of
    the item and the form a test can compare the normalised one against.
    """
    def value(field):
        if field == "quantization":
            return _crosscheck_quantization(record)
        if (field == "build_mode" and build_mode_engines is not None
                and record.get("engine") not in build_mode_engines):
            return None
        return record.get(field)

    return tuple(value(field) for field in CROSSCHECK_KEY_FIELDS)


def _normalised_storage(record: Dict[str, Any]) -> Any:
    """The storage a record measured, under one name across both paths.

    `render._STORAGE_OVERRIDES` is the existing table for exactly this -- its
    own comment says the ops harness "stamps a storage engine on every unit
    and defaults to InnoDB, which is meaningless for PostgreSQL. Its own ann
    records call the same thing `heap`; this keeps the two paths from
    disagreeing on the page." Reused rather than re-tabulated, so a new
    engine is added in one place.

    With it, a MariaDB MyISAM ann point and a MariaDB InnoDB ops point still
    come out different and are correctly never paired; without it, pgvector
    would never pair with itself.
    """
    from report.render import _STORAGE_OVERRIDES
    return (_STORAGE_OVERRIDES.get(record.get("engine"))
            or record.get("storage_engine"))


def _crosscheck_sort_key(record: Dict[str, Any]) -> Any:
    """Deterministic order for picking among several records on one side.

    A report regenerated from the same records has to come out the same, and
    the JSONL's own order is whatever the run happened to write.
    """
    extra = record.get("extra") or {}
    return (str(record.get("timestamp") or ""),
            str(extra.get("source_file") or ""))


#: The three states §5.2 item 3's "an empty match list is itself reported, not
#: hidden" collapses into if they are not told apart. Two of them are choices
#: and one is a bug, and the output has to say which:
#:
#: * NO_ANN / NO_OPS -- that path did not run in this profile. A choice.
#: * NO_M_OVERLAP -- both ran, and their M grids do not intersect. Also a
#:   choice, and the one `main` makes today: `ann.m_values: [8, 32]` against
#:   `ops.m_values: [16]`, which is §5.2 item 4's whole subject.
#: * NOTHING_MATCHED -- both ran, their M grids DO intersect, and no pair
#:   formed anyway. That is a bug in this matcher or in one path's fields,
#:   and it is the one a reader must not mistake for the configuration above.
COVERAGE_NO_ANN = "no recall sweep for this engine and dataset"
COVERAGE_NO_OPS = "no calibration sweep for this engine and dataset"
COVERAGE_NO_M_OVERLAP = "no M is measured by both paths (by configuration)"
COVERAGE_NOTHING_MATCHED = "M overlaps but no point matched"
COVERAGE_MATCHED = "matched"
#: Ops version 9: the calibration sweep exists but its recall is the ID
#: overlap (every ops record before version 9), which is not the rule the
#: ann path's `recall_qps` records use. Not compared, and said rather than
#: folded into "no calibration sweep", which would hide that it ran.
COVERAGE_RECALL_KIND = ("calibration recall is ID overlap (ops version < 9), "
                        "not the ann path's distance threshold")

#: `extra.recall_kind` values, spelled as `harness.datasets` spells them.
#: Literals rather than an import for the reason `render.py` declares its
#: status strings: `harness.datasets` pulls numpy for two strings.
RECALL_DISTANCE_THRESHOLD = "distance_threshold"
RECALL_ID_OVERLAP = "id_overlap"


def recall_kind(record: Dict[str, Any]) -> str:
    """Which rule a record's `recall_at_k` was computed with.

    ann-benchmarks' `recall_qps` records are the distance threshold by
    construction (`report/loaders.py:knn_recall`). An ops record says so in
    `extra.recall_kind` since version 9; one without the key predates it and
    was the ID overlap, which is what `datasets.recall_at_k` computed.
    """
    if record.get("phase") == "recall_qps":
        return RECALL_DISTANCE_THRESHOLD
    return str((record.get("extra") or {}).get("recall_kind")
               or RECALL_ID_OVERLAP)


def _append_crosscheck_row(matches: List[Dict[str, Any]],
                           named: Dict[str, Any], storage: Any,
                           ann_rec: Dict[str, Any], ops_rec: Dict[str, Any],
                           ann_candidates: int, ops_candidates: int) -> None:
    """One §5.2 row for one (ann point, ops point) pair.

    Split out of `consistency_checks` so the pairing loop above reads as the
    pairing rule and this reads as the comparison -- and so a test can build
    one row from two records without assembling a whole run.
    """
    from harness.metrics.stats import Interval

    # The ops record's own build mode, beside the key's normalised one. Where
    # the two differ the row describes an engine whose recall path never
    # declared the axis, and a reader comparing the row against the raw JSONL
    # would otherwise find a "post" the table does not show.
    declared_build_mode = ops_rec.get("build_mode")

    # One rule on both sides, or no row: a delta between a distance-threshold
    # recall and an ID overlap is a delta between definitions, and on a
    # corpus with ties at the k-th distance it is not zero for any engine.
    if recall_kind(ann_rec) != recall_kind(ops_rec):
        raise ValueError(
            f"refusing to compare a {recall_kind(ann_rec)} recall with a "
            f"{recall_kind(ops_rec)} one for {named}")
    recall_ann = float(ann_rec["recall_at_k"])
    recall_ops = float(ops_rec["recall_at_k"])
    interval = Interval.from_field(
        (ops_rec.get("extra") or {}).get("recall_ci95"))
    half = ((interval.high - interval.low) / 2.0
            if interval is not None and interval.usable else None)
    # `max`, as item 2 writes it. A record with no usable interval therefore
    # gets exactly the floor, which is the STRICTER of the two branches -- a
    # missing interval cannot widen the tolerance and quietly turn a failing
    # pair into a passing one.
    tolerance = (max(CROSSCHECK_MIN_TOLERANCE, 2.0 * half)
                 if half is not None else CROSSCHECK_MIN_TOLERANCE)
    delta = abs(recall_ann - recall_ops)
    matches.append({
        **named,
        "storage_engine": storage,
        "recall_ann": recall_ann,
        "recall_ops": recall_ops,
        "delta": delta,
        "tolerance": tolerance,
        "ci_half_width": half,
        # Which branch of the `max` produced the tolerance, so a row cannot
        # be read as CI-based when no interval was available.
        "tolerance_basis": (
            "ci" if half is not None
            and 2.0 * half > CROSSCHECK_MIN_TOLERANCE
            * (1.0 + CROSSCHECK_BASIS_SLACK) else "floor"),
        "passed": delta <= tolerance,
        "qps_ann": ann_rec.get("qps"),
        "qps_ops": ops_rec.get("qps"),
        # How many candidates each side offered for this key. More than one
        # means several records keyed identically: every one of them gets its
        # own row, and this says how many that was.
        "ann_candidates": ann_candidates,
        "ops_candidates": ops_candidates,
        "ops_build_mode": declared_build_mode,
        # True where the build-mode axis was dropped for this engine because
        # its recall sweep never declared one. The row is a real match; this
        # says which axis it did NOT match on.
        "build_mode_axis_dropped": (named["build_mode"] is None
                                    and declared_build_mode is not None),
    })


def consistency_checks(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PR 5 §5.2. Match ops calibration points to ann recall points.

    Returns the matched rows, the per-(engine, dataset) coverage that explains
    an empty list, and §5.2 item 5's QPS-order inversions.

    `harness.metrics.stats` is imported in the body, not at module scope.
    `report/` is bind-mounted into a container that does NOT mount `harness/`
    at the same path as the host does, and `CLAUDE.md` records an
    `orchestrator` import from `report/` crashing report generation at the end
    of a twenty-hour run. `loaders.py` and `measurement_versions.py` already
    import `harness.metrics.records` function-locally in eight places for this
    reason; this follows them.
    """
    from harness.metrics.stats import Interval

    ann = [r for r in records if r.get("phase") == "recall_qps"
           and r.get("recall_at_k") is not None]
    calibrated = [r for r in records if r.get("phase") == "ef_calibration"
                  and r.get("recall_at_k") is not None]
    # Ops version 9: only calibration records scored by the ann path's rule.
    ops = [r for r in calibrated
           if recall_kind(r) == RECALL_DISTANCE_THRESHOLD]
    other_kind = {(r.get("engine"), r.get("dataset")) for r in calibrated
                  if recall_kind(r) != RECALL_DISTANCE_THRESHOLD}

    # Which engines have a real build-mode axis, decided from the RECALL
    # records alone -- the ops side always has a value, so letting it vote
    # would make the axis unanimous and restore the mismatch.
    build_mode_engines = build_mode_axis_engines(ann)

    def grouped(rows):
        out: Dict[tuple, Dict[Any, List[Dict[str, Any]]]] = {}
        for r in rows:
            out.setdefault(crosscheck_key(r, build_mode_engines), {}
                           ).setdefault(_normalised_storage(r), []).append(r)
        return out

    ann_by_key, ops_by_key = grouped(ann), grouped(ops)

    matches: List[Dict[str, Any]] = []
    for key in sorted(set(ann_by_key) & set(ops_by_key), key=str):
        named = dict(zip(CROSSCHECK_KEY_FIELDS, key))
        ann_storages, ops_storages = ann_by_key[key], ops_by_key[key]
        for storage in sorted(set(ann_storages) & set(ops_storages), key=str):
            ann_rows = sorted(ann_storages[storage], key=_crosscheck_sort_key)
            ops_rows = sorted(ops_storages[storage], key=_crosscheck_sort_key)
            # Every PAIR, not the first of each side. §5.2 item 3 asks for
            # "every match with both values", and an earlier form of this
            # loop took `ann_rows[0], ops_rows[0]` -- which was invisible
            # while each side had one record and lost a real match as soon as
            # one had two. It does have two: where the build-mode axis is
            # dropped for an engine (see `build_mode_axis_engines`), that
            # engine's `post` and `incremental` calibration records key
            # identically, and the discarded one was a different index. A
            # dropped match is the quieter failure of the two and therefore
            # the worse one.
            #
            # The product is bounded in practice by `summary["duplicate_ann"]`
            # being empty: a second ann record for one configuration is
            # already reported as reading more than one measurement tree.
            for ann_rec, ops_rec in [(a, o) for a in ann_rows
                                     for o in ops_rows]:
                _append_crosscheck_row(matches, named, storage, ann_rec,
                                       ops_rec, len(ann_rows), len(ops_rows))

    # Coverage, per (engine, dataset), so an empty `matches` says WHY.
    def m_values(rows):
        out: Dict[tuple, set] = {}
        for r in rows:
            if r.get("m") is not None:
                out.setdefault((r.get("engine"), r.get("dataset")),
                               set()).add(r["m"])
        return out

    ann_m, ops_m = m_values(ann), m_values(ops)
    matched_pairs = {(row["engine"], row["dataset"]) for row in matches}
    coverage: List[Dict[str, Any]] = []
    for pair in sorted(set(ann_m) | set(ops_m), key=str):
        engine, dataset = pair
        mine_ann = ann_m.get(pair, set())
        mine_ops = ops_m.get(pair, set())
        overlap = mine_ann & mine_ops
        if not mine_ann:
            status = COVERAGE_NO_ANN
        elif not mine_ops and pair in other_kind:
            status = COVERAGE_RECALL_KIND
        elif not mine_ops:
            status = COVERAGE_NO_OPS
        elif not overlap:
            status = COVERAGE_NO_M_OVERLAP
        elif pair in matched_pairs:
            status = COVERAGE_MATCHED
        else:
            status = COVERAGE_NOTHING_MATCHED
        coverage.append({
            "engine": engine, "dataset": dataset,
            "ann_m": sorted(mine_ann), "ops_m": sorted(mine_ops),
            "overlap_m": sorted(overlap), "status": status,
            "matches": sum(1 for row in matches
                           if (row["engine"], row["dataset"]) == pair),
        })

    return {
        "matches": sorted(matches, key=lambda row: (
            str(row["dataset"]), str(row["engine"]), row["m"] or 0,
            row["ef_search"] or 0, str(row["storage_engine"]))),
        "coverage": coverage,
        "qps_inversions": _crosscheck_qps_inversions(matches),
    }


#: How close two engines' recalls must be for a throughput comparison between
#: them to be "at matched recall", which is the condition §5.2 item 5 attaches
#: to its inversion check.
#:
#: It is a real condition and not a formality. A fixed `ef_search` is not one
#: quantity across these engines -- `render.py`'s section 6 prose spells this
#: out: the same literal is `mhnsw_ef_search`, VIDX's equivalent,
#: `hnsw.ef_search`, MongoDB's `numCandidates` and Valkey's `EF_RUNTIME`, and
#: it buys recalls up to 0.18 apart. Two engines at one ef but 0.18 of recall
#: apart are not doing comparable work, so a throughput order that differs
#: between the two paths says nothing about either path: the faster engine is
#: simply the less accurate one, and which way that lands is arithmetic.
#:
#: 0.02 rather than the interval overlap test §5.1 item 6 uses, deliberately.
#: Overlap would make this check depend on how tightly each engine's recall
#: happened to be pinned -- a noisy engine would be "matched" with everyone --
#: whereas the question here is whether the two did a comparable amount of
#: work, which is a fixed band.
CROSSCHECK_RECALL_BAND = 0.02


def _crosscheck_qps_inversions(matches: List[Dict[str, Any]]
                               ) -> List[Dict[str, Any]]:
    """§5.2 item 5: "at matched recall, the engine order from ann (in-process)
    and ops (TCP, 1 client) should agree. Flag inversions, do not fail on
    them: TCP overhead is a legitimate difference."

    Grouped by `(dataset, resource_pass, m, ef_search, quantization)` -- the
    §5.2 key WITHOUT `engine`, and without `build_mode` or the normalised
    storage either. That subtraction is the whole difficulty of this item and
    the first version of this function got it wrong: it grouped on every
    non-engine field plus the storage, and since storage differs per engine by
    design (pgvector's `heap` against MariaDB's `InnoDB`) and the build-mode
    axis is dropped for the MySQL family but not for pgvector, no two engines
    ever landed in one group. Zero inversions on a fixture built to contain
    one, and nothing said so.

    So the group holds constant what the engines were all GIVEN -- the corpus,
    the machine allocation, the graph degree and the query effort -- and the
    things that are structural properties of an implementation rather than
    settings are carried on the row instead. That is the same reasoning §5.2
    item 1 applies when it omits `storage_engine` from the match key.

    Flagged per PAIR rather than as a rank correlation over the group: a
    correlation says the orders differ without saying which two engines
    swapped, and the pair is the actionable fact.

    Three things are NOT inversions:

    * a pair of the same engine in two build modes. pgvector `post` against
      pgvector `incremental` is one engine's two indexes, and item 5 is about
      the ENGINE order.
    * a tie on either side. Two engines with identical ann QPS have no ann
      order for the ops order to contradict, and `>` on both sides would
      report a disagreement with nothing.
    * a pair whose recalls are further apart than `CROSSCHECK_RECALL_BAND`.
      Those two engines were not doing comparable work, so their throughput
      order is not a measurement of speed. The constant's comment has the
      reason.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in matches:
        if row.get("qps_ann") is None or row.get("qps_ops") is None:
            continue
        groups.setdefault(
            (row["dataset"], row["resource_pass"], row["m"],
             row["ef_search"], row["quantization"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=str):
        rows = sorted(groups[key],
                      key=lambda r: (str(r["engine"]),
                                     str(r["storage_engine"]),
                                     str(r["ops_build_mode"])))
        for i, first in enumerate(rows):
            for second in rows[i + 1:]:
                if first["engine"] == second["engine"]:
                    continue
                # "At matched recall". Both paths' recalls are checked, not
                # just one: the pair has to be comparable in the measurement
                # whose ordering is being questioned, and §5.2 item 2 has
                # already established that the two paths can disagree.
                if (abs(first["recall_ann"] - second["recall_ann"])
                        > CROSSCHECK_RECALL_BAND
                        or abs(first["recall_ops"] - second["recall_ops"])
                        > CROSSCHECK_RECALL_BAND):
                    continue
                ann_gap = first["qps_ann"] - second["qps_ann"]
                ops_gap = first["qps_ops"] - second["qps_ops"]
                if ann_gap == 0 or ops_gap == 0:
                    continue
                if (ann_gap > 0) == (ops_gap > 0):
                    continue
                out.append({
                    "dataset": first["dataset"],
                    "resource_pass": first["resource_pass"],
                    "m": first["m"], "ef_search": first["ef_search"],
                    "quantization": first["quantization"],
                    "engines": [first["engine"], second["engine"]],
                    "faster_in_ann": (first if ann_gap > 0
                                      else second)["engine"],
                    "faster_in_ops": (first if ops_gap > 0
                                      else second)["engine"],
                    "qps_ann": [first["qps_ann"], second["qps_ann"]],
                    "qps_ops": [first["qps_ops"], second["qps_ops"]],
                    # The recalls the comparison was judged matched on, so a
                    # reader can see the band was applied rather than take it
                    # on trust -- and the structural axes the group dropped.
                    "recall_ann": [first["recall_ann"], second["recall_ann"]],
                    "recall_ops": [first["recall_ops"], second["recall_ops"]],
                    "storage_engines": [first["storage_engine"],
                                        second["storage_engine"]],
                    "build_modes": [first["ops_build_mode"],
                                    second["ops_build_mode"]],
                })
    return out
