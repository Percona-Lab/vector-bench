"""Churn's two report-side distinctions, both of which were published wrong.

Its own module rather than another 150 lines in `report/generate.py`, which is
928 lines against this repository's stated 800 maximum before anything is added
to it. The precedent is `report/measurement_versions.py`, moved out of
`loaders.py` for the same reason and with the same note: so the change is
reviewed here rather than inherited by unrelated work in a file nobody wanted
to open. `generate.py` does **not** re-export these names -- it reaches
`during_writes_rows` through a function-body import inside `summarize`, and
`render.py` and `charts.py` import from here directly. That is the opposite of
how `measurement_versions.py` was split out, where `loaders.py` kept every
public name bound and a test asserts the identity; the difference is that
those names had existing callers and these have none, so a re-export here
would be an interface nothing uses. Said explicitly because an earlier
revision of this docstring claimed the re-exports existed and cited that
precedent as if it applied.

The two distinctions:

1. **Which side of `driver.maintain()` a churn record was measured on.** PR 4
   §4.2 writes a `"before"` and an `"after"` record per churn fraction, and
   nothing in `report/` read `extra.maintenance_state` -- it had zero
   occurrences in this directory. So `render._churn_table` emitted two rows
   per fraction whose engine, dataset, storage and fraction columns were
   byte-identical, carrying recalls 0.8960 and 0.9980 and index sizes
   275.0 KiB and 250.0 KiB. Both pairs reproduced exactly in 9 of 9 runs of
   one configuration on `tests/fake_driver.py`, because the fake's recall
   comes off a seeded generator and its `index_bytes` is arithmetic on the
   row count. The only two readings a reader had were "a noisy measurement"
   and "a duplicate row", and both were false.

2. **Whether a churn record measures the corpus or the reader pool.** PR 4
   §4.3 writes a third kind: aggregate reader throughput and percentiles
   measured while the writer re-inserts. It carries `phase == "churn"` and a
   `churn_fraction`, which is what all three existing churn artifacts select
   on, so it has to be excluded from each of them explicitly. See
   `is_during_writes` for what including it would have published.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from report import generate

#: What `extra.maintenance_state` means when it is absent, which it is on the
#: pre-churn baseline record, on every churn record written before PR 4 §4.2,
#: and on anything `tests/synthetic.py` builds. Those measurements describe one
#: state and do not name it, so a rendered dash is the honest cell: reading
#: absence as `"before"` would claim a repair path was about to run, and
#: reading it as `"after"` would claim one had.
MAINTENANCE_STATE_WHEN_ABSENT = None

#: How far the ratio a `during_writes` record STAMPS may differ from the one
#: recomputed off that record's own two percentiles before the report stops
#: trusting them to describe each other. One per cent, because both numbers
#: are rounded on the way into the record and nothing else should move them.
DURING_WRITES_RATIO_TOLERANCE = 0.01


def maintenance_state(record: Dict[str, Any]) -> Optional[str]:
    """Which side of `driver.maintain()` a churn record was measured on."""
    return ((record.get("extra") or {}).get("maintenance_state")
            or MAINTENANCE_STATE_WHEN_ABSENT)


def is_during_writes(record: Dict[str, Any]) -> bool:
    """Whether this churn record is the reader pool's rather than the corpus's.

    PR 4 §4.3 writes a second kind of churn record: reader QPS and percentiles
    measured by `ops.churn_readers` clients WHILE the writer re-inserts. It
    carries `phase == "churn"` and a `churn_fraction`, which is what every
    existing churn artifact selects on, and its `qps` is aggregate throughput
    over several clients against a corpus that is mid-change.

    So it has to be excluded from all three of them, and the reason is not
    tidiness. `charts.churn_retention` divides a post-churn QPS by the
    series' single-client pre-churn baseline; handed an aggregate reader
    figure it publishes a spurious retention on an axis labelled "QPS as a
    fraction of pre-churn (higher is better)". That is a new wrong result of
    exactly the kind PR 4 exists to remove, and it would have been introduced
    by the change that fixed the others.

    **The magnitude depends on whether the reader's query releases the
    interpreter lock, and an earlier revision of this docstring quoted a
    single number that was its own test fixture's arithmetic.** Measured over
    9 real churn runs each, on `tests/fake_driver.py`, **both arms on the
    RETIRED THREAD POOL** -- these are the figures that established the size
    of the hazard and they are not a current reading of this harness. Since
    `OPS_MEASUREMENT_VERSION` 6 each reader is a forked OS process, so the
    two rows below no longer differ by what a lock does; what survives is the
    hazard itself, because an aggregate reader QPS over a single-client
    baseline is a spurious retention whatever produced the readers:

      * readers whose query releases the GIL (a `latency_s` sleep, which is
        what a real driver blocked on a socket does): spurious retention
        **3.74 to 4.26**, 9 of 9 above 1.0 -- near the reader count, which is
        the naive reading and here the right one.
      * readers that are numpy-bound in-process (`latency_s = 0`): **0.89 to
        2.62**, 5 of 9 above 1.0, and most of that spread is the
        single-client baseline itself, which ranged 4,336 to 12,076 QPS
        across the same runs.

    So the hazard is "a retention that is not a retention" rather than any
    particular factor. The GIL-versus-socket distinction behind the two rows
    is `clients.py`'s module docstring, which carries it because
    `CLIENT_BOUND_UTIL` is blind in the same direction -- the flag's own
    2.06-core measurement was taken in the numpy-bound regime, the bottom row
    here, and a real driver runs in the top one. (An earlier revision of this
    paragraph cited `CLIENT_BOUND_UTIL` itself as documenting the
    distinction. It does not: it documents the GIL-blindness and the
    numpy-bound measurement, and the word "socket" does not appear beside it.
    The sentence that did make the distinction was deleted by the §4.3 lift
    and has been restored to `clients.py`.)
    """
    return bool((record.get("extra") or {}).get("during_writes"))


def churn_measurements(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Churn records that measure the corpus, excluding the reader pool's.

    The one place the exclusion is spelled, so a fourth churn artifact cannot
    be added without it. See `is_during_writes`.
    """
    return [r for r in records
            if r.get("phase") == "churn" and not is_during_writes(r)]


def during_writes_rows(records: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
    """One published row per `during_writes` record: PR 4's headline ratio.

    **The ratio is recomputed from ONE record's own two percentiles, never by
    pairing two records**, and that is the whole design. This repository has
    published two wrong ratios of exactly this shape. The filter-strategy
    ratio divided a narrow point from one ops unit by a wide point from
    another -- 0.20x "pre-filter" and 1.00x "post-filter, abandons the query"
    separately, against a pooled 50.0x "post-filter, exhaustive" that neither
    unit produced. `charts.churn_retention` pairs records from two different
    maintenance states. Both read as findings and both were bugs.

    `harness/workloads/churn_readers.py` measures both percentiles with one
    pool whose identity `clients.window_ratio` checks before dividing -- same
    client count, same query pool, same k, same ef_search, same ops unit --
    and writes both into one record. Here, `latency_p99_ms` (the window under
    writes) over `extra.reader_baseline_p99_ms` is therefore a ratio of two
    numbers that provably ran together, and the table can show all three so a
    reader can do the division themselves.

    The row is keyed and SORTED by `(engine, dataset, fraction) +
    ops_unit(record)` for the reason `filter_strategies` above is: a table
    whose rows cannot be told apart prints the same identity twice with
    different numbers, which reads as a rendering fault rather than as the
    two different measurements it is. `ops_unit` and `OPS_UNIT_FIELDS` are
    T3c's, reused rather than re-invented, so the AST pin against
    `cli._run_unit`'s loops keeps covering this too.

    `ratio_disagrees` is the cross-check. The harness also stamps its own
    `extra.reader_p99_ratio`; if that and the recomputed value differ by more
    than `DURING_WRITES_RATIO_TOLERANCE`, the record's two halves do not
    describe each other and the row says so rather than publishing whichever
    of them a reader happens to look at.

    No value here is rounded to a published precision, and that is deliberate.
    A `p99` ratio is a ratio of two wall-clock percentiles: measured on
    `tests/fake_driver.py`, the same churn configuration's QPS ranged from
    6,028 to 16,811 across nine runs, a factor of 2.79, while its seeded
    recall did not move in the last digit. The structure of the pairing is
    what this function guarantees; the decimals are the box's.
    """
    rows: List[Dict[str, Any]] = []
    for r in records:
        if r.get("phase") != "churn" or not is_during_writes(r):
            continue
        extra = r.get("extra") or {}
        # Reached through the MODULE, never through a `from ... import`
        # of the tuple. `ops_unit` reads `OPS_UNIT_FIELDS` out of
        # `generate`'s globals on every call, so a from-imported copy here
        # would pair stale names against fresh values the moment anything
        # changed that tuple -- and something does: `tests/test_t3c_filtered.py`
        # drops an axis from it at run time to prove the grouping is load
        # bearing. Measured with the from-import form, that mutation reported
        # `m: "post"` (build_mode's value under m's name) where this form
        # reports `m: None`.
        unit = dict(zip(generate.OPS_UNIT_FIELDS, generate.ops_unit(r)))
        baseline_p99 = extra.get("reader_baseline_p99_ms")
        under_p99 = r.get("latency_p99_ms")
        # Falsy and not just None: a percentile of exactly 0.0 is what
        # `tests/fake_driver.py` returns unless it is given a latency to
        # spend, and a ratio over a zero denominator is an arithmetic
        # accident, not a slowdown figure.
        ratio = (under_p99 / baseline_p99
                 if baseline_p99 and under_p99 else None)
        stamped = extra.get("reader_p99_ratio")
        rows.append({
            "engine": r.get("engine"), "dataset": r.get("dataset"),
            "churn_fraction": r.get("churn_fraction"),
            # Carried so the rendered row can say which unit it describes.
            # `.get`, not `[...]`: an axis absent from OPS_UNIT_FIELDS is
            # reported as None rather than crashing, so the rendered table
            # shows a blank cell where the pooling happened.
            "m": unit.get("m"), "build_mode": unit.get("build_mode"),
            "storage_engine": unit.get("storage_engine"),
            "resource_pass": unit.get("resource_pass"),
            "readers": extra.get("readers"),
            "baseline_qps": extra.get("reader_baseline_qps"),
            "under_writes_qps": r.get("qps"),
            # How many reader queries each window actually counted, and how
            # long each was open. Carried because a QPS of 0.0 from a window
            # that closed before one read finished is not a measurement of
            # anything, and the renderer cannot tell it from a genuine zero
            # without these. See `churn_readers._warn_empty_reader_window`.
            "baseline_queries": extra.get("reader_baseline_queries_executed"),
            "under_writes_queries": extra.get(
                "reader_under_writes_queries_executed"),
            "baseline_window_s": extra.get("reader_baseline_window_s"),
            "under_writes_window_s": extra.get(
                "reader_under_writes_window_s"),
            "baseline_p99_ms": baseline_p99,
            "under_writes_p99_ms": under_p99,
            "p99_ratio": ratio,
            "writer_rows_per_s": extra.get("writer_rows_per_s_with_readers"),
            "client_bound": extra.get("client_bound"),
            # Whether the readers ran the queries their count implies. Carried
            # because EVERY figure on this row depends on it -- the p99 ratio,
            # both throughputs and the writer rate beside them. A pool that
            # served one read at a time measures a p99 of a read that spent
            # its time queued somewhere rather than of one the engine was slow
            # to answer, and the writer rate beside it was taken against those
            # same reads. Until `ops_measurement_version` 6 the somewhere was
            # Python's interpreter lock, which the readers shared with the
            # writer; each reader is a forked OS process now, so a low reading
            # here is no longer attributable to it.
            # `extra.scaling_status` on this record is the DURING-WRITES
            # window's, which is the window every published figure here comes
            # from; the baseline's is carried separately so a reader can see
            # whether the two halves of the ratio were measured in the same
            # regime. See `harness/workloads/clients.py`.
            "served_concurrency": extra.get("served_concurrency"),
            "outstanding_concurrency": extra.get("outstanding_concurrency"),
            "scaling_status": extra.get("scaling_status"),
            # Ops measurement version 9: the heavy-tail note, beside the
            # status rather than instead of it.
            "scaling_note": extra.get("scaling_note"),
            "baseline_scaling_status": extra.get(
                "reader_baseline_scaling_status"),
            "baseline_served_concurrency": extra.get(
                "reader_baseline_served_concurrency"),
            "ratio_disagrees": (
                ratio is not None and stamped is not None
                and abs(ratio - stamped) > DURING_WRITES_RATIO_TOLERANCE),
            "stamped_p99_ratio": stamped,
            # True when a window counted no query, which makes every figure
            # derived from it absent rather than zero. `== 0`, not falsy: a
            # missing key is a record from a harness that did not record the
            # count and is not a claim that the window was empty.
            "baseline_empty": extra.get(
                "reader_baseline_queries_executed") == 0,
            "under_writes_empty": extra.get(
                "reader_under_writes_queries_executed") == 0,
        })
    rows.sort(key=lambda s: (s["engine"] or "", s["dataset"] or "",
                             s["churn_fraction"] or 0.0,
                             str(s["resource_pass"]), s["m"] or 0,
                             str(s["build_mode"]), str(s["storage_engine"])))
    return rows
