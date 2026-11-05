"""Tests for the run-time estimate: `orchestrator/estimate.py`.

This function had no tests at all, which is how it came to be load-bearing for a
convention it could not satisfy. The convention says the printed estimate must
include every cost and that *a run longer than its printed estimate is a bug* —
but the function modelled ingest only, said so in its label, and every run had
therefore always exceeded it.

Two things are pinned here, in this order of importance:

1. **The ingest numbers did not move.** They are the published figures: the
   printed block, `docs/07-planning-a-run.md`, and two web-UI renderers all
   carry them. Every real profile's ingest total is pinned to the value it
   produced before query time existed, so a reader can see it did not change.
2. **`total_hours` still means ingest.** `webui/static/control.js` and
   `webui/static/setup.js` both render it as `estimated ingest ~N h`. Silently
   redefining it would have changed what the UI has been telling people without
   touching the UI.

The arithmetic tests state their own numbers. An estimate is arithmetic over a
config, so a test that recomputes the implementation's expression passes
whatever that expression is; the hand-computed cases here are written out from
the profile and the measured rate tables so a wrong formula cannot satisfy them.
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest.mock as mock
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import cli as cli_mod  # noqa: E402
from orchestrator import estimate as est_mod  # noqa: E402
from orchestrator import ops_pass as ops_pass_mod  # noqa: E402
from orchestrator.cli import KNOWN_ENGINES  # noqa: E402
from orchestrator.config import load_profile  # noqa: E402
from orchestrator.estimate import (print_load_estimate,  # noqa: E402
                                   estimate_load_hours)

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL = list(KNOWN_ENGINES)
BOTH_PHASES = ["ann", "ops"]


def est(profile_name, engines=None, datasets=None, passes=("normalized",),
        phases=BOTH_PHASES):
    profile = load_profile(profile_name)
    return estimate_load_hours(
        profile, list(engines or ALL),
        list(datasets if datasets is not None else profile.get("datasets", [])),
        list(passes), list(phases))


def printed(profile_name, engines=None, datasets=None, passes=("normalized",),
            phases=BOTH_PHASES):
    profile = load_profile(profile_name)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_load_estimate(
            profile, list(engines or ALL),
            list(datasets if datasets is not None else profile.get("datasets", [])),
            list(passes), list(phases))
    return buf.getvalue()


def term(estimate, name):
    found = [t for t in estimate["query_terms"] if t["name"] == name]
    assert len(found) == 1, (
        f"{name!r} not in {[t['name'] for t in estimate['query_terms']]}")
    return found[0]


# A budget is enforced by comparing elapsed time against it, which is what
# `churn.py:244` does: `if time.time() - started > budget_s`. Accepting a
# mention of the word instead passed two states in which nothing capped
# anything: the realistic one, where the flag is plumbed from ops_pass through
# the argument parser into the workload's signature and never looked at again,
# and the trivial one, where a comment in the workload says no budget is
# applied -- and in this codebase, prose about a missing bound is likely.
_ENFORCES_A_BUDGET = re.compile(
    r"(?:time\.(?:time|monotonic)|perf_counter)\s*\(\s*\)[^\n]*[<>][^\n]*budget"
    r"|budget[^\n]*[<>][^\n]*(?:time\.(?:time|monotonic)|perf_counter)\s*\(\s*\)")


def _filtered_budget_is_enforced() -> bool:
    """Whether the filtered workload stops when its budget runs out.

    Only `harness/workloads/filtered.py` is consulted. `harness/main.py` was in
    an earlier version of this check and is not enforcement: parsing a flag and
    handing it on is exactly the state that passed while the query loop ran
    unbounded.

    If T3c enforces the budget in a shape this pattern does not recognise -- a
    precomputed deadline, say -- widen the pattern deliberately and say in the
    commit how the loop stops. Do not relax it to a mention.
    """
    # Calibrated against the one budget this repo does enforce, so a pattern
    # that has stopped matching real enforcement fails here rather than
    # silently reporting that nothing is enforced anywhere.
    with open(os.path.join(VB_ROOT, "harness", "workloads", "churn.py")) as fh:
        assert _ENFORCES_A_BUDGET.search(fh.read()), (
            "the enforcement pattern no longer matches churn.py, which does "
            "enforce its budget; fix the pattern, not the conclusion")
    with open(os.path.join(VB_ROOT, "harness", "workloads", "filtered.py")) as fh:
        source = fh.read()
    return bool(re.search(r"\bbudget", source)
                and _ENFORCES_A_BUDGET.search(source))


def _iterative_scan_gate_name() -> str:
    """The driver attribute `harness/workloads/filtered.py` gates on.

    Read from the gate rather than written here, so that a rename has to happen
    on both sides of it: the workload records `iterative_scan: None` and says
    nothing when the driver lacks the method.
    """
    path = os.path.join(VB_ROOT, "harness", "workloads", "filtered.py")
    with open(path) as fh:
        gate = re.search(r'hasattr\(driver,\s*"([A-Za-z_][A-Za-z0-9_]*)"\)',
                         fh.read())
    assert gate, "filtered.py no longer gates on hasattr(driver, ...)"
    return gate.group(1)


def _one_hour_per_pass_term(plan):
    """A probe term worth one hour per (engine, dataset, pass).

    `main` with one pass is SEVEN engines x two datasets, so it adds exactly
    14 h — a number the extension tests can state rather than derive. It was
    12 h over six engines, before `config/engines/percona.yml` existed; the
    quantity is "one hour per engine per dataset", so this figure moves with
    the engine list by construction and a reader should expect it to.
    """
    return est_mod._QueryTerm(
        name="test term", typical_s=3600.0, worst_s=3600.0,
        basis="one hour, flat", bounded=True, scope="pass")


# ---------------------------------------------------------------------------
# 1. The ingest estimate, pinned as it was before query time was added
# ---------------------------------------------------------------------------

class TestTheIngestEstimateDidNotMove:
    """These were the numbers on origin/master. The task that added query time
    was explicitly not allowed to change them, and the published docs quote
    them, so they are pinned rather than recomputed.

    **`main` moved once, deliberately, and this records why.** PR 5 §5.2 item
    4 needed the ann M grid and the ops M to meet, or the report's
    "Consistency checks" section is empty by construction on the profile the
    human actually runs. `config/profiles/main.yml` therefore went from
    `ann.m_values: [8, 32]` to `[8, 16, 32]`, which is a third full reload of
    both corpora per engine: all six engines with a rate went from 69.77 h to
    93.06 h of ingest. Every other profile is untouched, `full` included.

    **And once more, for the opposite reason: a rate was MEASURED.** Percona
    was the seventh engine with no entry in either rate table, so it was named
    in `engines_without_ingest_rate` and contributed nothing to any figure
    here. B1's census of the archived ops records supplied the M=16
    incremental rate -- 339 / 275 / 427 rows/s on fashion-mnist / glove-100 /
    sift-128, from `full-20261015-215006` -- and every total below therefore
    ROSE: `main` 93.06 -> 100.44, `full` at both passes 726.45 -> 796.28.
    `CLAUDE.md`'s standing figure moves with that one, because it is the
    seven-engine figure and the seventh engine is now in it.

    Nothing else may move them. A diff that changes a number here without
    changing a profile, or without a new measurement in the rate tables, is
    the drift this class exists to catch.
    """

    def test_main_one_pass(self):
        e = est("main")
        assert e["total_hours"] == 100.44
        # Seven keys, not six. Percona's 7.38 h is 60,000 rows at the measured
        # 339 rows/s plus 1.18M at 275, over three ann M values and one ops M,
        # and it is the whole of the 7.38 h this profile's total rose by.
        assert e["per_engine_hours"] == {
            "mariadb": 17.54, "alisql": 49.66, "pgvector": 0.91,
            "mariadb123": 24.83, "mongodb": 0.05, "valkey": 0.07,
            "percona": 7.38,
        }
        # MyVector, the eighth engine, has no measured rate: named, and
        # contributing nothing to the ingest total, exactly as Percona was
        # before its rate was measured. That is why no figure above moved.
        # VillageSQL, the ninth, is the same: no measured rate yet; and so is
        # villagesql84, the same extension on the MySQL 8.4 server line.
        assert e["engines_without_ingest_rate"] == ["myvector", "villagesql",
                                                    "villagesql84"]
        # Three ann M values now, one ops M. The pair is the whole point of
        # the change: `ops_m_values` is still 1 and `ann.m_values` contains
        # it, which is what makes §5.2's match key able to meet.
        assert (e["m_values"], e["ops_m_values"], e["passes"]) == (3, 1, 1)
        assert 16 in load_profile("main")["ann"]["m_values"]
        assert load_profile("main")["ops"]["m_values"] == [16]
        assert e["long_run"] is True

    def test_main_both_passes_is_exactly_double(self):
        assert est("main", passes=("normalized", "tuned"))["total_hours"] == 200.87

    def test_main_ops_only(self):
        """Dropping the recall phase drops the ann M grid from the estimate."""
        assert est("main", phases=["ops"])["total_hours"] == 25.11

    def test_smoke(self):
        assert est("smoke")["total_hours"] == 0.59

    def test_full_two_passes(self):
        e = est("full", passes=("normalized", "tuned"))
        assert e["total_hours"] == 796.28
        # The published 726.45 h was six engines of seven. Percona's own
        # 69.83 h is the amount CLAUDE.md used to describe as the one "nothing
        # here knows", and it is now a figure rather than a floor.
        assert e["per_engine_hours"]["percona"] == 69.83
        assert round(796.28 - 69.83, 2) == 726.45

    def test_a_dataset_with_no_measured_rate_is_still_named_not_guessed(self):
        e = est("dev")
        assert e["datasets_without_estimate"] == ["tiny-16-euclidean"]
        assert e["total_hours"] == 0.0

    def test_ingest_hours_is_the_same_number_under_its_own_name(self):
        """`total_hours` keeps its published meaning; `ingest_hours` is the
        unambiguous name for the same figure, for code written from now on."""
        e = est("main")
        assert e["ingest_hours"] == e["total_hours"] == 100.44


class TestPerconasMeasuredIngestRateReachesTheEstimator:
    """The seventh engine's rate, and the three decisions behind each figure.

    Percona was measured seventeen times across six runs and modelled nowhere:
    both rate tables omitted it, `_effective_rate` returned `None` for every
    dataset, and `--profile percona` printed an estimate with no hour figure
    for the only engine it runs. The entries below close that, and every one of
    them is a record from `full-20261015-215006` rather than a model.

    **The three axes the key cannot express, resolved once.** The table is
    keyed `(engine, dataset)` and the archive is indexed by
    `(run_id, dataset, M, resource_pass, build_mode, rows)`:

    * *M* is 16, the mid-grid value this table's own comment sets as its
      convention, and a dataset with no M=16 incremental record is OMITTED
      rather than interpolated -- which is why gist-960-euclidean, measured at
      M=32 alone, has no entry.
    * *rows* is the 60,000-row full-corpus load, not the 20,000-row
      `ops.subset_rows` one, because `estimate_load_hours` multiplies by
      `_DATASET_ROWS`. The 20k loads ran 13% faster.
    * *build_mode* is `incremental`, Percona's default at this pin. The `post`
      record's 2653.0 rows/s is in neither table.

    The figures are asserted TWICE, and the pair is deliberate: once by value,
    because each is a specific record and a neighbouring integer is not that
    record, and once against the census SPAN, which is the shape check that no
    `post`-mode figure can pass -- 2653.0 is 4.2x the top of the incremental
    range. The span alone was not enough. fashion-mnist is 60,000 rows, so
    337 through 341 all satisfied every OTHER assertion in this repository
    while plus or minus one on the other two moves nine published figures;
    `test_the_three_entries_are_the_records_they_came_from` is what closes
    that, and `--profile main` cannot close it because `main` loads
    glove-100 and sift-128 and not fashion-mnist.
    """

    #: The incremental min and max of B1's census: gist-960 tuned M=32 at
    #: 48.17 rows/s and sift-128 normalized M=8 at 638.11.
    #: `config/profiles/percona.yml` prints both.
    CENSUS_SPAN = (48.17, 638.11)
    #: The seventeenth record -- a `post` plain load with no graph being built.
    #: Here so the span assertion below has something it demonstrably rejects.
    POST_RATE = 2653.0
    #: The datasets with an M=16 incremental record, and the one without.
    MEASURED = ("fashion-mnist-784-euclidean", "glove-100-angular",
                "sift-128-euclidean")
    NO_MID_GRID = "gist-960-euclidean"

    def test_the_profile_gets_an_hour_figure_and_names_no_gap(self):
        """A10's acceptance criterion, on the returned dict.

        NOT on the printed block: `--profile percona` is one engine on one
        60,000-row corpus at one M, so its worst case is 0.27 h and
        `print_load_estimate` returns before printing anything at all under
        its 1 h gate. The hour figure reaching the dict is what the CLI and
        the web UI both read, and `main` below is where the printed form is
        checked.
        """
        profile = load_profile("percona")
        e = estimate_load_hours(profile, list(profile["engines"]),
                                list(profile["datasets"]), ["normalized"],
                                BOTH_PHASES)
        assert e["engines_without_ingest_rate"] == []
        assert e["per_engine_hours"]["percona"] > 0
        assert e["total_hours"] > 0

    def test_the_printed_block_no_longer_announces_the_gap(self):
        """The other half of the acceptance criterion, on a profile above the
        gate. `main` runs all seven engines, so the `! No measured ingest
        rate` line fired there before this change and must not now."""
        out = printed("main")
        assert "percona ~" in out
        # The disclosure is back, for MyVector alone -- the eighth engine,
        # whose rate nothing has measured -- and Percona is not in it.
        # VillageSQL, the ninth, joins it on the same line, and villagesql84.
        assert ("No measured ingest rate for myvector, villagesql, "
                "villagesql84, so"
                in " ".join(out.split()))
        assert "No measured ingest rate for percona" not in out

    def test_the_three_entries_are_the_records_they_came_from(self):
        """By value, per dataset. The span below is a shape check and this is
        the identity one: these are three named records -- 338.94, 275.04 and
        426.53 rows/s, M=16, `incremental`, from `full-20261015-215006` -- and
        a figure one row/s away from any of them is not a measurement this
        repository has taken."""
        assert {k[1]: v for k, v in est_mod._MEASURED_ROWS_PER_S.items()
                if k[0] == "percona"} == {
            "fashion-mnist-784-euclidean": 339,
            "glove-100-angular": 275,
            "sift-128-euclidean": 427,
        }

    def test_every_entry_is_inside_the_census_span(self):
        lo, hi = self.CENSUS_SPAN
        for dataset in self.MEASURED:
            rate = est_mod._MEASURED_ROWS_PER_S[("percona", dataset)]
            assert lo <= rate <= hi, (dataset, rate)
        assert lo <= est_mod._INGEST_ROWS_PER_S["percona"] <= hi
        # Non-vacuity: the span is what rejects the `post` figure, which is
        # the mutant this assertion exists for. Without this line a span of
        # (0, inf) would pass the loop above.
        assert not lo <= self.POST_RATE <= hi

    def test_the_reference_rate_is_the_reference_corpus(self):
        """`_INGEST_ROWS_PER_S` is scaled from `_REFERENCE_DIMS` 100 and
        `_REFERENCE_ROWS` 1,000,000, so the entry has to be the rate measured
        on a ~100-dim ~1M-row corpus. That is glove-100-angular, at 275.04
        rows/s. This is MariaDB's precedent exactly: its 150 is its glove-100
        measurement of 147, nudged to a round number."""
        assert (est_mod._INGEST_ROWS_PER_S["percona"]
                == est_mod._MEASURED_ROWS_PER_S[("percona",
                                                 "glove-100-angular")])
        assert est_mod._DATASET_DIMS["glove-100-angular"] == est_mod._REFERENCE_DIMS

    def test_the_dataset_with_no_mid_grid_record_is_omitted_not_interpolated(self):
        """gist-960 was measured at one M -- 32, at 48.17 rows/s, the
        pessimistic extreme of the whole census -- so an entry for it would be
        a figure mislabelled as mid-grid. The formula fallback serves it
        instead and is honest about being a model: 70.79 rows/s against an
        implied ~100 at M=16 (the M16/M32 ratio is 2.07-2.09 where it was
        measured), i.e. 1.41x pessimistic rather than 2.08x and dressed as a
        measurement."""
        assert ("percona", self.NO_MID_GRID) not in est_mod._MEASURED_ROWS_PER_S
        rate = est_mod._effective_rate("percona", self.NO_MID_GRID)
        # Served, and by the model: below the measured M=32 figure's 48.17
        # would be worse than the formula, above the M=16 entries would mean
        # the penalty had stopped applying.
        assert rate is not None
        assert 48.17 < rate < min(est_mod._MEASURED_ROWS_PER_S[("percona", d)]
                                  for d in self.MEASURED)

    def test_no_dataset_is_left_without_a_rate_for_this_engine(self):
        """`_effective_rate` returns `None` only for an engine absent from
        `_INGEST_ROWS_PER_S`, so one entry there empties the gap list for
        every profile in the repository -- including the datasets Percona has
        never loaded, which the formula serves."""
        for dataset in sorted(est_mod._DATASET_ROWS):
            assert est_mod._effective_rate("percona", dataset) is not None, dataset
        # Non-vacuity: the None path is still reachable, and
        # `tests/test_config.py` keeps it covered with the same probe.
        assert est_mod._effective_rate("e2-no-such-engine",
                                       "glove-100-angular") is None


# ---------------------------------------------------------------------------
# 2. total_hours still means ingest
# ---------------------------------------------------------------------------

class TestTotalHoursStillMeansIngest:
    """webui/static/setup.js:237 and webui/static/control.js:41 both print this
    field as `estimated ingest ~N h`. setup.js is not owned by this change, so
    a redefinition here would have made a file nobody touched start lying."""

    def test_total_hours_excludes_query_time(self):
        e = est("main")
        assert e["query_hours"] > 0
        assert e["total_hours"] == e["ingest_hours"]
        assert e["worst_case_total_hours"] > e["total_hours"]

    def test_a_new_query_term_does_not_move_total_hours(self, monkeypatch):
        """The guard that matters for the four tasks that will add terms: an
        added term must reach the worst case and must not reach the ingest
        figure the UI labels 'ingest'."""
        before = est("main")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (_one_hour_per_pass_term,))
        after = est("main")
        assert after["total_hours"] == before["total_hours"]
        assert after["per_engine_hours"] == before["per_engine_hours"]
        # 20.0: one hour per (engine, dataset, pass), ten engines.
        assert after["worst_case_total_hours"] == round(
            before["worst_case_total_hours"] + 20.0, 2)

    def test_long_run_is_still_the_ingest_warning(self):
        """The warning text is about M values reloading the corpus, so the flag
        that triggers it stays keyed on ingest. pgvector alone is 0.91 h of
        loading and 7.58 h of worst-case run time, and the M-reload warning
        would be the wrong thing to print about it.

        The worst case grew in T3c -- the filtered budget turned a term with
        no ceiling into one with a real one -- and again in T4b, by the churn
        reader baseline: 30 s per churn fraction, and pgvector pays it twice
        per (dataset, pass) because its two build modes are two ops units.
        The ingest figure grew in T5b, when PR 5 §5.2 added M=16 to
        `main`'s ann grid. It grew a fourth time when the iterative-scan modes
        started reaching the run: `main` declares two, the filtered grid runs
        once per mode, and 4 pgvector ops units x 6 points x 300 s is 2.0 h
        more -- the only engine that moved, which is the point of charging it
        per engine. `long_run` did not move any of the four times, which is
        the property being pinned: pgvector bulk-builds, so a third reload
        costs it a quarter of an hour and the M-reload warning would still be
        the wrong thing to print."""
        e = est("main", engines=["pgvector"])
        assert e["total_hours"] == 0.91
        # Ops measurement version 9: the churn budget default is 7200 s per
        # step (was 1800) and every client count pays one discarded
        # warm-up window, so the figures below moved from the ones the
        # docstring narrates; the arithmetic is the same with those two
        # inputs.
        assert e["worst_case_total_hours"] == 21.64
        assert e["long_run"] is False


# ---------------------------------------------------------------------------
# 3. Query terms, pinned against hand arithmetic
# ---------------------------------------------------------------------------

class TestTheConcurrencyTermIsTimeNotQueries:
    """`main` runs client_counts [1, 4, 8, 16, 32] for 30 s each, three
    times per point. Each point runs for a fixed span whatever the engine
    manages, so this term is exact.

    The grid lost its 2-client point on 2026-10-14, which is why the figure
    below is 450 s and not 540 s: at two clients BOTH estimators of the
    achieved concurrency equal exactly 1/n = 0.5 for a perfectly serialized
    pool, the floor is a strict `<`, and the verdict therefore lands on
    whichever side of it the box's noise puts it -- measured 0.499 (refused)
    in one synthetic driver and 0.537 (published) in another. See
    tests/test_two_client_points_are_not_measured.py.

    By hand, for `main`, one pass, SEVEN engines, two datasets:
        5 points x 30 s x 3 repeats               =   450 s per ops unit
        ops units per (engine, dataset): 1 each,
            except pgvector, which runs both
            pgvector_build_modes               -> 8 units per dataset
        8 units x 2 datasets x 1 pass             =    16 unit-instances
        16 x 450 s = 7,200 s                      =  2.00 h

    Over the six engines that preceded Percona the same arithmetic gave 7
    units per dataset, 14 unit-instances and 1.75 h. Percona contributes one
    ops unit per dataset like every engine except pgvector: `percona.yml` is
    the only profile that sets `percona_build_modes`, and `main` does not.
    """

    def test_main(self):
        # 2.25 and not 2.00: MyVector, the eighth engine, adds one ops unit per
        # dataset -- 9 units x 2 datasets = 18 unit-instances x 450 s.
        # 3.00 since ops version 9: (3 repeats + 1 warm-up window) per point.
        # 3.33 with VillageSQL, the ninth: 10 units x 2 datasets.
        # 3.67 with villagesql84, the tenth: 11 units x 2 datasets.
        t = term(est("main"), "concurrency windows")
        assert t["typical_hours"] == 3.67
        assert t["worst_hours"] == 3.67
        assert t["bounded"] is True

    def test_the_client_count_list_is_counted_not_summed(self):
        """[1, 2, 4, 8, 16, 32] is six windows, not 63 of anything."""
        profile = {"name": "t", "ops": {"enabled": True,
                                        "workloads": ["concurrency"],
                                        "m_values": [16],
                                        "client_counts": [1, 2, 4, 8, 16, 32],
                                        "concurrency_duration_s": 30,
                                        "max_queries": 500}}
        e = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        # 6 x 30 s x (1 + 1 warm-up window, ops v9) = 360 s = 0.10 h
        assert term(e, "concurrency windows")["typical_hours"] == 0.10

    def test_pgvector_runs_every_build_mode_so_it_costs_twice(self):
        base = {"enabled": True, "workloads": ["concurrency"], "m_values": [16],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 500}
        two = {"name": "t", "ops": base,
               "ann": {"pgvector_build_modes": ["post", "incremental"]}}
        one = {"name": "t", "ops": base, "ann": {"pgvector_build_modes": ["post"]}}
        args = (["pgvector"], ["glove-100-angular"], ["normalized"], ["ops"])
        # Each window count doubled by ops v9's one warm-up window per point.
        assert term(estimate_load_hours(two, *args),
                    "concurrency windows")["typical_hours"] == 0.20
        assert term(estimate_load_hours(one, *args),
                    "concurrency windows")["typical_hours"] == 0.10

    def test_more_ops_m_values_cost_proportionally(self):
        base = {"enabled": True, "workloads": ["concurrency"],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 500}
        three = {"name": "t", "ops": {**base, "m_values": [8, 16, 32]}}
        e = estimate_load_hours(three, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        # x 2 for ops v9's warm-up window at each point.
        assert term(e, "concurrency windows")["typical_hours"] == 0.30

    def test_repeats_multiply_because_the_runner_already_honours_them(self):
        """ops_pass.harness_args passes --concurrency-repeats today, defaulting
        to 1. A profile that sets it gets that many windows per point, so the
        estimate has to count them or it under-reports a cost the run pays."""
        base = {"enabled": True, "workloads": ["concurrency"], "m_values": [16],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 500}
        args = (["mariadb"], ["glove-100-angular"], ["normalized"], ["ops"])
        plain = estimate_load_hours({"name": "t", "ops": base}, *args)
        thrice = estimate_load_hours(
            {"name": "t", "ops": {**base, "concurrency_repeats": 3}}, *args)
        # (repeats + 1): ops v9's warm-up window is not a repeat, and it is
        # paid once per point however many repeats follow it.
        assert term(plain, "concurrency windows")["typical_hours"] == 0.10
        assert term(thrice, "concurrency windows")["typical_hours"] == 0.20

    def test_a_point_with_more_clients_than_queries_is_skipped(self):
        """The workload skips it, so the estimate must not charge for it."""
        base = {"enabled": True, "workloads": ["concurrency"], "m_values": [16],
                "client_counts": [1, 2, 4, 8, 16, 32],
                "concurrency_duration_s": 30, "max_queries": 4}
        e = estimate_load_hours({"name": "t", "ops": base}, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        # Only 1, 2 and 4 clients can run against four queries: 3 x 30 s,
        # twice each with ops v9's warm-up window = 180 s.
        assert term(e, "concurrency windows")["typical_hours"] == 0.05


class TestTheChurnFractionsAreCumulative:
    """`churn_fractions: [0.10, 0.25]` re-inserts 10% and then a further 15% —
    a cumulative 25%, not 35%. The workload says so and the estimate must agree.

    By hand, mariadb on glove-100-angular (1,183,514 rows, measured 147 rows/s),
    one ops unit, one pass:
        step 1:  10% of 1,183,514 = 118,351 rows / 147 =   805.11 s
        step 2:  15% of 1,183,514 = 177,527 rows / 147 = 1,207.67 s
        total                                          = 2,012.78 s = 0.56 h
    and the worst case is the budget, twice: 2 x 1,800 s = 3,600 s = 1.00 h.
    """

    PROFILE = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                    "m_values": [16],
                                    "churn_fractions": [0.10, 0.25]}}

    def test_typical_is_the_incremental_rows_at_the_measured_rate(self):
        e = estimate_load_hours(self.PROFILE, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        assert term(e, "churn re-insert")["typical_hours"] == 0.56

    def test_worst_case_is_the_budget_per_step(self):
        e = estimate_load_hours(self.PROFILE, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        # 2 x 7,200 s (the ops v9 default budget) = 4.00 h.
        assert term(e, "churn re-insert")["worst_hours"] == 4.00

    def test_a_single_fraction_is_one_step(self):
        """10% alone: 118,351 rows / 147 = 805.11 s = 0.22 h, budget once."""
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "churn_fractions": [0.10]}}
        t = term(estimate_load_hours(profile, ["mariadb"],
                                     ["glove-100-angular"], ["normalized"],
                                     ["ops"]), "churn re-insert")
        # Budget once: 7,200 s since ops v9.
        assert (t["typical_hours"], t["worst_hours"]) == (0.22, 2.00)

    def test_the_subset_a_profile_loads_is_what_gets_churned(self):
        """A profile that loads 20,000 rows of fashion-mnist churns 10% of
        those 2,000 rows at the measured 745 rows/s — 2.7 s, not 10% of
        60,000 rows."""
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "churn_fractions": [0.10],
                                        "subset_rows": 20000}}
        e = estimate_load_hours(profile, ["mariadb"],
                                ["fashion-mnist-784-euclidean"],
                                ["normalized"], ["ops"])
        t = term(e, "churn re-insert")
        assert t["typical_seconds"] == pytest.approx(2000 / 745.0, abs=0.01)
        assert t["worst_hours"] == 2.00


class TestTheFilteredTermChargesEveryModeItWillMeasure:
    """Filtered search runs `columns x selectivities x modes x max_queries`
    queries per ops unit. The shape is exact and so, since T3c, is the
    ceiling: `ops.filtered_budget_s` caps every point and the workload stops
    when it is spent. The *typical* duration is still unknown, because no QPS
    has been measured, and the term carries both facts so the output does not
    charge for the cost and disclaim it at the same time.

    Two columns, not one: `harness/workloads/filtered.py` measures `tag` and
    `ctag` by default because `build` loads both, so the point count doubled
    in T3c and this term has to double with it.

    The mode factor arrived with the call site that produces it.
    `cli.py:_ops_iterative_scan` used to pin the first declared
    `ops.pgvector_iterative_scan` value, so `* modes` could only ever have
    been `* 1`; it now passes every mode that validates, so the factor is real
    and pgvector's filtered term doubles. It is charged PER ENGINE -- the
    tests in this class run on `mariadb`, which has no such knob and pays one
    pass whatever a profile declares. See `_charged_scan_modes`.

    `main`: 2 columns x 3 selectivities x 1 mode x 500 queries = 3,000 queries
    per ops unit on six of the seven engines, 6,000 on pgvector, capped at
    300 s a point either way.
    """

    PROFILE = {"name": "t", "ops": {"enabled": True, "workloads": ["filtered"],
                                    "m_values": [16], "max_queries": 500,
                                    "selectivities": [0.10, 0.50]}}

    def _term(self, profile=None, engine="mariadb"):
        return term(estimate_load_hours(profile or self.PROFILE, [engine],
                                        ["glove-100-angular"], ["normalized"],
                                        ["ops"]), "filtered search")

    def test_the_shape_is_stated_even_though_the_duration_is_not(self):
        t = self._term()
        assert ("2 filter column(s) x 2 selectivities x 1 iterative-scan mode "
                "x 500 queries = 2,000 queries") in t["basis"]
        # Anchored: "2,000 queries per ops unit" is a substring of
        # "12,000 queries per ops unit".
        assert "= 2,000 queries per ops unit" in t["basis"]
        # The typical figure is still the thing no cap can supply.
        assert "at a rate nothing here has measured" in t["basis"]

    def test_the_mode_factor_is_in_the_product_and_is_pgvectors_alone(self):
        """The arithmetic the call-site change made owing, and the two mutants
        either side of it.

        `PROFILE` declares two modes here. On pgvector that is 2 x 2 x 2 = 8
        points at 300 s = 0.67 h per ops unit; on mariadb it is 4 points and
        0.33 h, because `_charged_scan_modes` gates on the engine.

        Honest: 0.67 h for pgvector, 0.33 h for mariadb, from one profile.
        Mutant (`mode_passes` dropped from `points`): pgvector reads 0.33 h,
            so a run that will take an hour per unit is planned as half of
            one and `worst_case_total_hours` stops being a ceiling.
        Mutant (`_charged_scan_modes` returning the declared list for every
            engine): mariadb reads 0.67 h, and the estimate charges six
            engines for a dimension only pgvector has.
        """
        ops = dict(self.PROFILE["ops"])
        ops["pgvector_iterative_scan"] = [False, "relaxed_order"]
        profile = {"name": "t", "ops": ops}
        pg = self._term(profile, engine="pgvector")
        md = self._term(profile, engine="mariadb")
        assert pg["worst_hours"] == 0.67
        assert md["worst_hours"] == 0.33
        assert ("2 filter column(s) x 2 selectivities x 2 iterative-scan "
                "modes x 500 queries = 4,000 queries") in pg["basis"]
        assert ("2 filter column(s) x 2 selectivities x 1 iterative-scan "
                "mode x 500 queries = 2,000 queries") in md["basis"]

    def test_a_mode_declared_twice_is_charged_once(self):
        """`[off, off]` asks for one measurement, which is the rule
        `filtered._scan_modes` applies to the same list on the far side of the
        command line. Charged through the shared `_declared_scan_modes`, so
        the estimate and the run cannot disagree about it.

        Honest: 0.33 h, the same as one mode.
        Mutant (`len(declared)` in place of `_declared_scan_modes`): 0.67 h,
            and the estimate charges for a grid the workload de-duplicates
            away.
        """
        ops = dict(self.PROFILE["ops"])
        ops["pgvector_iterative_scan"] = [False, False, "off"]
        t = self._term({"name": "t", "ops": ops}, engine="pgvector")
        assert t["worst_hours"] == 0.33

    def test_an_unmeasurable_typical_is_flagged_not_scored_zero(self):
        """The typical figure, which a cap cannot supply.

        `bounded` is True and `typical_hours` is 0.0 at once, and that pair is
        the honest state rather than a contradiction: the budget says what the
        work cannot exceed, and nothing here says what it will actually take.
        The same split `_churn_reinsert` keeps, for the same reason --
        collapsing them there would have hidden two lost runs.
        """
        t = self._term()
        assert t["typical_hours"] == 0.0
        # 4 points x 300 s = 1,200 s = 0.33 h, which the budget guarantees.
        assert t["worst_hours"] == 0.33
        assert t["bounded"] is True
        # `quantified` reads off the worst case, so a real ceiling makes it
        # True even while the typical figure is unknown.
        assert t["quantified"] is True

    def test_it_no_longer_stops_the_run_worst_case_being_a_ceiling(self):
        """The inverse of what this asserted before T3c, and the reason the
        budget was worth building.

        `query_worst_case_is_a_ceiling` is an `all()` over the terms, and
        filtered search was the single term holding it open: every configured
        profile printed "not yet a ceiling" about the one number the "a run
        longer than its printed estimate is a bug" convention refers to. The
        budget closes it.
        """
        e = estimate_load_hours(self.PROFILE, ["mariadb"],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        assert e["query_worst_case_is_a_ceiling"] is True

    def test_the_budget_key_now_buys_a_real_ceiling(self):
        """`ops.filtered_budget_s` now exists in all three places the
        cross-check below names: `ops_pass.harness_args` emits it,
        `harness/main.py` parses it and hands it on, and `filtered.py` stops
        the query loop on it. So honouring it here prints a ceiling something
        enforces, which is the state this test used to say T3c had to reach.

        Lowering it lowers the ceiling proportionally, which is what proves the
        figure is read from the key rather than from the default; and turning
        it off restores the unbounded term, so the branch that refuses to
        invent a ceiling is still reachable.
        """
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "filtered_budget_s": 300}}
        t = self._term(profile)
        assert (t["typical_hours"], t["worst_hours"]) == (0.0, 0.33)
        assert t["bounded"] is True
        assert "300s cap/point" in t["basis"]

        # Half the budget, half the ceiling: 4 points x 150 s = 600 s = 0.17 h.
        halved = {"name": "t", "ops": {**self.PROFILE["ops"],
                                       "filtered_budget_s": 150}}
        assert self._term(halved)["worst_hours"] == 0.17
        assert "150s cap/point" in self._term(halved)["basis"]

        off = {"name": "t", "ops": {**self.PROFILE["ops"],
                                    "filtered_budget_s": 0}}
        assert self._term(off)["bounded"] is False
        assert self._term(off)["worst_hours"] == 0.0

    def test_the_estimate_and_the_runner_honour_the_budget_together(self):
        """Whichever part of T3c lands first, this fails until the rest does.

        The chain has three links, like `churn_budget_s` before it:
        `ops_pass.harness_args` emits the flag (ops_pass.py:432 for churn),
        `harness/main.py` passes it to the workload (main.py:242), and the
        workload stops when it runs out (churn.py:244). Only the first and the
        third are checked, and the third has to be an elapsed-time comparison
        rather than a reference to the word.

        Two states have already passed a weaker version of this check while
        nothing capped the loop. Emission alone, with the flag never plumbed.
        And then the realistic one: the flag emitted, parsed, and accepted as
        `budget_s: float = 1800.0` by the workload, which never looks at it --
        every link in place and no bound anywhere. The second is how a feature
        like this is actually built, one layer at a time, which is why the test
        has to care about the layer that stops rather than the layers that
        carry.

        Since T3c all three exist, which is consistent, so this passes. It
        passed before T3c too, when none of them did; the states it rejects
        are the partial ones in between."""
        from orchestrator import ops_pass
        from orchestrator.config import load_resources
        profile = {"name": "t", "k": 10,
                   "ops": {**self.PROFILE["ops"], "filtered_budget_s": 300}}
        estimate_honours = self._term(profile)["bounded"]
        argv = ops_pass.harness_args(profile, 16, "pgvector", None,
                                     "normalized", load_resources("normalized"))
        runner_emits = any(str(a).startswith("--filtered-budget") for a in argv)

        # The workload has to be the thing that stops.
        enforced = _filtered_budget_is_enforced()
        wired = runner_emits and enforced
        assert estimate_honours == wired, (
            "ops.filtered_budget_s is honoured by some of the run and not all "
            f"of it (estimate={estimate_honours}, harness_args={runner_emits}, "
            f"workload={enforced}); the estimate must not claim a ceiling "
            "nothing enforces, nor ignore one that is enforced")

    def test_a_measured_rate_turns_it_into_a_duration(self, monkeypatch):
        """The other half: 2,000 queries at 0.9 QPS = 2,222.2 s, capped at
        4 points x 300 s = 1,200 s = 0.33 h. The measured MariaDB filtered
        rate -- which docs/07 records as 37 minutes for 2,000 queries at 1%
        selectivity -- is slow enough that the budget binds, which is the
        budget doing its job: the modelled cost exceeds the cap, so the cap is
        what the run will spend."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 0.9)
        t = self._term()
        assert t["typical_hours"] == 0.33
        assert t["quantified"] is True
        # Bounded by the cap, not by the rate: a rate is still not a cap, and
        # here the two meet only because the modelled cost is the larger.
        assert t["bounded"] is True
        assert t["worst_hours"] == 0.33

    def test_more_selectivities_cost_proportionally(self, monkeypatch):
        """2 columns x 3 selectivities x 500 = 3,000 queries at 0.9 QPS =
        3,333.3 s, capped at 6 points x 300 s = 1,800 s = 0.50 h."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 0.9)
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "selectivities": [0.01, 0.10, 0.50]}}
        assert self._term(profile)["typical_hours"] == 0.50

    def test_it_scales_with_ops_units_like_everything_else(self, monkeypatch):
        """pgvector runs both build modes: 2 units x 2,000 queries at 2 QPS =
        2,000 s = 0.56 h, under a 4 x 300 = 1,200 s per-unit cap that the
        1,000 s a unit costs does not reach."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("pgvector", "glove-100-angular"), 2.0)
        two = {"name": "t",
               "ann": {"pgvector_build_modes": ["post", "incremental"]},
               "ops": self.PROFILE["ops"]}
        one = {"name": "t", "ann": {"pgvector_build_modes": ["post"]},
               "ops": self.PROFILE["ops"]}
        assert self._term(two, engine="pgvector")["typical_hours"] == 0.56
        assert self._term(one, engine="pgvector")["typical_hours"] == 0.28

    def test_a_profile_without_the_workload_pays_nothing(self):
        profile = {"name": "t", "ops": {**self.PROFILE["ops"],
                                        "workloads": ["build"]}}
        e = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        assert [t["name"] for t in e["query_terms"]] == []

    def test_a_profile_without_selectivities_still_runs_one(self, monkeypatch):
        """ops_pass.harness_args defaults --selectivities to 0.1, so such a
        profile pays one point. Charging nothing dropped that cost out of the
        terms list and out of the disclosure list both — a third state, neither
        counted nor named."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 0.9)
        ops = {k: v for k, v in self.PROFILE["ops"].items()
               if k != "selectivities"}
        t = self._term({"name": "t", "ops": ops})
        # 2 columns x 1 selectivity x 500 = 1,000 queries at 0.9 QPS =
        # 1,111.1 s, capped at 2 points x 300 s = 600 s = 0.17 h.
        assert t["typical_hours"] == 0.17
        assert ("2 filter column(s) x 1 selectivity x 1 iterative-scan mode "
                "x 500 queries = 1,000 queries") in t["basis"]

    def test_the_default_matches_the_one_the_runner_uses(self):
        """Pinned against ops_pass rather than restated, because the defect was
        the two diverging."""
        from orchestrator import ops_pass
        from orchestrator.config import load_resources
        ops = {k: v for k, v in self.PROFILE["ops"].items()
               if k != "selectivities"}
        argv = ops_pass.harness_args({"name": "t", "k": 10, "ops": ops}, 16,
                                     "mariadb", None, "normalized",
                                     load_resources("normalized"))
        emitted = argv[argv.index("--selectivities") + 1].split(",")
        assert len(emitted) == 1
        assert f"{len(emitted)} selectivity" in self._term(
            {"name": "t", "ops": ops})["basis"]


class TestTheIterativeScanModeNowReachesTheRun:
    """`ops.pgvector_iterative_scan` was declared by `main`, `full` and the
    tuned extras, the plumbing to `SET hnsw.iterative_scan` was complete, and
    the report rendered a column for it — but `_run_unit` never passed it, so
    every archived pgvector filtered record carries `iterative_scan: null`.

    There was a second fault behind the first: YAML 1.1 reads a bare `off` as
    boolean False, so the declared list parses as `[False, "relaxed_order"]`.
    Passing the value through naively would have emitted nothing at all
    (`if iterative_scan:` is false for False) or `SET hnsw.iterative_scan =
    False`, which is not a mode pgvector has.

    **And a third behind that**, which is what the later tests here are for:
    once a mode did reach the run, only the FIRST declared one did.
    `list(declared)[0]` discarded the rest, so `full`'s `[off, relaxed_order]`
    measured `off` -- and the 2026-10-15 seven-engine run published 72
    pgvector filtered records, all at `off`, 49 of them under 0.6 recall, each
    one's recall sitting on the predicate's own selectivity. pgvector ships
    iterative scan to fix exactly that collapse; the profile asked for it and
    the request was dropped at the one call site that chooses.
    """

    def _resources(self, name):
        from orchestrator.config import load_resources
        return load_resources(name)

    def test_the_profile_key_really_does_parse_as_a_boolean(self):
        """The premise, pinned: if this ever changes the normaliser is dead
        weight rather than load-bearing, and a reader should be told."""
        assert load_profile("main")["ops"]["pgvector_iterative_scan"][0] is False

    def test_every_declared_mode_is_resolved_to_its_yaml_name(self):
        """Honest: `main` declares `[off, relaxed_order]` and both come back,
        in order, under the names they were written as.

        Mutant (`list(declared)[0]`, which is what shipped): "off", so the
            second mode never reaches a `SET` and the run measures the vendor
            default while the profile says otherwise. This is the defect.
        Mutant (`str()` in place of the YAML-name recovery): "False", which is
            not a pgvector mode and would reach the server verbatim.
        """
        mode = cli_mod._ops_iterative_scan(
            "pgvector", load_profile("main"),
            self._resources("normalized"), "normalized")
        assert mode == "off,relaxed_order"

    def test_a_mode_the_engine_does_not_declare_is_dropped_on_its_own(self):
        """Per-mode degradation, which the all-or-nothing form did not do.

        Honest: the two good modes still run and the bad one is dropped with
            the warning.
        Mutant (`return None` on the first unknown mode, which is what
            shipped): None for the whole engine, so one typo in a profile
            silently returns pgvector to its vendor default -- the same
            published-wrong-number outcome this change exists to close, from
            the opposite direction.
        """
        profile = {"ops": {"pgvector_iterative_scan":
                           [False, "sideways", "relaxed_order"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, {}, "normalized") == "off,relaxed_order"

    def test_an_unvalidated_mode_never_reaches_the_joined_value(self):
        """The guard that matters most, because
        `PostgresDriver.set_iterative_scan` interpolates each mode into a
        `SET` and an unrecognised string would reach the server verbatim.

        Honest: "sideways" is absent from the result in both shapes -- alone
            it yields None, beside a good mode it yields the good mode only.
        Mutant (the `valid` check deleted): "off,sideways", so
            `SET hnsw.iterative_scan = sideways` is issued and takes the whole
            ops unit down with concurrency and churn.
        """
        one = {"ops": {"pgvector_iterative_scan": ["sideways"]}}
        both = {"ops": {"pgvector_iterative_scan": [False, "sideways"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", one, {}, "normalized") is None
        joined = cli_mod._ops_iterative_scan(
            "pgvector", both, {}, "normalized")
        assert joined == "off"
        assert "sideways" not in joined

    def test_the_whitelist_is_the_engine_configs_own_declaration(self):
        """Not a literal in `cli.py`. The pin is that every mode
        `config/engines/pgvector.yml` declares resolves, so the day pgvector
        adds or removes one, editing that file is the whole change.

        Mutant (a hard-coded `{"off", "relaxed_order"}`): `strict_order`,
            which the engine config declares and no profile uses, is refused
            -- and the refusal would look like a profile typo.
        """
        from orchestrator.config import load_engine
        declared = (load_engine("pgvector").get("capabilities") or {}
                    ).get("iterative_scan_modes")
        assert declared, "pgvector.yml no longer declares its modes"
        profile = {"ops": {"pgvector_iterative_scan": list(declared)}}
        resolved = cli_mod._ops_iterative_scan(
            "pgvector", profile, {}, "normalized")
        assert resolved == ",".join(
            ops_pass_mod._yaml_mode_name(m) for m in declared)

    def test_a_mode_declared_twice_is_passed_once(self):
        """`[off, off]` asks for one measurement, which is also what
        `filtered._scan_modes` would do with it on the far side of the command
        line. De-duplicated here as well so the orchestrator's count and the
        estimate's agree -- both read `ops_pass._declared_scan_modes`."""
        profile = {"ops": {"pgvector_iterative_scan": [False, False, "off"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, {}, "normalized") == "off"

    def test_the_tuned_pass_reads_its_own_extras(self):
        """The two sources have to disagree for this to mean anything:
        `main.yml:201` and `tuned.yml:68` both declare `[off, relaxed_order]`,
        so a fixture built from the real files returns "off,relaxed_order"
        whether the extras branch is read or falls through to the profile --
        which is why this one is synthetic. It also pins the precedence, which
        nothing else did.

        The two numbers above were `main.yml:64` and `:68`; `:64` was wrong on
        master already, pointing at a comment about which profiles have an ann
        phase. The return value was "off" until the resolver stopped dropping
        every mode after the first."""
        profile = {"ops": {"pgvector_iterative_scan": ["off"]}}
        extras = {"extras": {"pgvector_iterative_scan": ["relaxed_order"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, extras, "tuned") == "relaxed_order"
        # and the extras are a tuned-pass thing only
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, extras, "normalized") == "off"

    def test_the_real_tuned_pass_resolves(self):
        """Both real sources declare the same two modes in the same order, so
        the pass makes no difference today — pinned so that stops being
        invisible if one of them changes."""
        for pass_name in ("normalized", "tuned"):
            assert cli_mod._ops_iterative_scan(
                "pgvector", load_profile("main"),
                self._resources(pass_name),
                pass_name) == "off,relaxed_order"

    def test_the_profiles_that_declare_two_modes_get_two(self):
        """Both headline profiles, against the real files rather than a
        fixture, because the defect was invisible precisely in the files a run
        actually uses."""
        for name in ("main", "full"):
            assert cli_mod._ops_iterative_scan(
                "pgvector", load_profile(name),
                self._resources("normalized"),
                "normalized") == "off,relaxed_order"

    def test_nothing_is_pinned_for_an_engine_without_the_knob(self):
        assert cli_mod._ops_iterative_scan(
            "mariadb", load_profile("main"),
            self._resources("tuned"), "tuned") is None

    def test_a_profile_that_declares_nothing_still_passes_nothing(self):
        """So the engine keeps its own default and the record keeps saying so,
        rather than this inventing a mode on the profile's behalf."""
        assert cli_mod._ops_iterative_scan(
            "pgvector", {"ops": {}}, {}, "normalized") is None

    def test_an_unknown_mode_is_refused_rather_than_interpolated(self):
        """The driver builds `SET hnsw.iterative_scan = {mode}` by string
        interpolation, so an unrecognised value would reach the server."""
        profile = {"ops": {"pgvector_iterative_scan": ["sideways"]}}
        assert cli_mod._ops_iterative_scan(
            "pgvector", profile, {}, "normalized") is None

    def test_the_run_unit_passes_the_mode_to_the_harness(self, tmp_path):
        """The guard the rest of this class does not give: every other test
        here wires `iterative_scan=` itself, which proves the resolver works
        and that `harness_args` consumes it, and proves nothing about the one
        line in between. A call site that forgets the argument is the exact
        defect that produced 58 ops files and 69 null records, and `_run_unit`
        had no test anywhere in the repo.

        No container and no network: `harness_args` is captured and `OpsRun` is
        a fake whose `run_harness` returns 0."""
        import argparse
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        captured = []

        def capture(*a, **kw):
            captured.append(kw)
            return ["--fake"]

        class FakeRun:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_harness(self, *a, **kw):
                return 0

        paths = {"run_dir": str(tmp_path),
                 "checkpoints": str(tmp_path / "checkpoints")}
        with mock.patch.object(ops_pass, "harness_args", capture), \
             mock.patch.object(ops_pass, "OpsRun", FakeRun):
            for engine, expected in (("pgvector", "off,relaxed_order"),
                                     ("mariadb", None), ("alisql", None)):
                captured.clear()
                rc = cli_mod._run_unit(
                    "ops", engine, "glove-100-angular", load_profile("main"),
                    {}, load_resources("normalized"), None, "normalized",
                    paths, "run-0", argparse.Namespace(force=False),
                    checkpoints=None)
                assert rc == 0
                assert captured, f"{engine}: harness_args was never called"
                for kwargs in captured:
                    assert kwargs.get("iterative_scan") == expected, engine

    def test_two_modes_add_no_ops_unit_and_no_corpus_reload(self, tmp_path):
        """The objection the old `_ops_iterative_scan` docstring raised against
        passing a list, settled by measurement rather than by argument.

        `_run_unit`'s `unit_count` is `m_values x build_modes x
        storage_engines` and every unit reloads the corpus, so a mode that
        became a fourth axis there would re-run build, concurrency and churn
        per mode and double pgvector's ingest. It does not: the modes are one
        argument to one `harness_args` call per unit, and the workload loops
        them against the index that unit already built.

        Honest: pgvector launches the same number of units under a profile
            declaring two modes as under one -- 2 for `main` (1 M value x 2
            build modes) -- and the only difference is the argument's value.
        Mutant (the mode made an axis of the `for m / for build_mode /
            for storage` loops): 4 units, and the second mode costs a full
            reload of the corpus.
        """
        import argparse
        from orchestrator import ops_pass
        from orchestrator.config import load_resources

        launched = []

        class FakeRun:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run_harness(self, args, output, **kw):
                launched.append(list(args))
                return 0

        paths = {"run_dir": str(tmp_path),
                 "checkpoints": str(tmp_path / "checkpoints")}
        counts, flags = {}, {}
        for label, modes in (("one", [False]),
                             ("two", [False, "relaxed_order"])):
            loaded = load_profile("main")
            profile = dict(loaded, ops=dict(loaded["ops"],
                                            pgvector_iterative_scan=modes))
            launched.clear()
            with mock.patch.object(ops_pass, "OpsRun", FakeRun):
                rc = cli_mod._run_unit(
                    "ops", "pgvector", "glove-100-angular", profile, {},
                    load_resources("normalized"), None, "normalized", paths,
                    "run-0", argparse.Namespace(force=False),
                    checkpoints=None)
            assert rc == 0
            counts[label] = len(launched)
            flags[label] = {
                argv[argv.index("--iterative-scan") + 1] for argv in launched}

        assert counts["one"] == counts["two"] == 2, counts
        assert flags["one"] == {"off"}
        assert flags["two"] == {"off,relaxed_order"}

    def test_the_harness_passes_it_on_to_the_workload(self):
        """The other unguarded link in the same chain, asserted at source
        level: `harness/main.py` hands `args.iterative_scan` to
        `filtered.run`. Read rather than executed because running it needs a
        server; this suite already reads source this way to check that the web
        UI's field names exist."""
        path = os.path.join(VB_ROOT, "harness", "main.py")
        with open(path) as fh:
            source = fh.read()
        call = re.search(r"filtered_workload\.run\((?P<args>.*?)\n\s*\)",
                         source, re.S)
        assert call, "filtered_workload.run( is no longer where this expects"
        assert "iterative_scan=args.iterative_scan" in call.group("args")

    def test_the_harness_command_line_carries_it(self):
        """`harness_args` ignores its `resolved` argument, so None keeps this
        test off the hardware-detection path."""
        from orchestrator import ops_pass
        resources = self._resources("normalized")
        profile = load_profile("main")
        argv = ops_pass.harness_args(
            profile, 16, "pgvector", None, "normalized", resources,
            iterative_scan=cli_mod._ops_iterative_scan(
                "pgvector", profile, resources, "normalized"),
        )
        assert "--iterative-scan" in argv
        assert argv[argv.index("--iterative-scan") + 1] == "off,relaxed_order"

    def test_the_old_call_site_emitted_nothing(self):
        """What the run has always done: no mode argument at all, which is how
        69 filtered records came to say null."""
        from orchestrator import ops_pass
        argv = ops_pass.harness_args(
            load_profile("main"), 16, "pgvector", None, "normalized",
            self._resources("normalized"))
        assert "--iterative-scan" not in argv

    def test_the_driver_implements_the_method_the_workload_gates_on(self):
        """The last link, and the one the test below cannot prove because it
        attaches the method itself. `filtered.py:56` gates on
        `hasattr(driver, ...)` and records None when it is absent, so renaming
        the method on `PostgresDriver` puts every pgvector filtered record back
        to null with the whole suite green. Nothing asserted it existed: the
        only references outside the driver are that gate and the fake below.

        The name is read out of the gate rather than written here, so a rename
        has to happen on both sides or this fails."""
        from harness.drivers.postgres import PostgresDriver
        name = _iterative_scan_gate_name()
        assert callable(getattr(PostgresDriver, name, None)), (
            f"filtered.py asks for driver.{name}() and PostgresDriver does not "
            f"implement it, so the mode is silently dropped")

    def test_a_filtered_record_now_carries_the_mode(self, tmp_path):
        """End of the chain, against the fake driver rather than a server: the
        record's `extra["iterative_scan"]` was None for every run ever made."""
        from harness.drivers.postgres import PostgresDriver
        from harness.metrics import records
        from harness.workloads import filtered as filtered_workload
        from harness.workloads.context import RunContext
        from harness.datasets import FILTER_TAG
        from tests.fake_driver import index_spec, loaded_driver, tiny_dataset

        dataset = tiny_dataset()
        driver = loaded_driver(dataset, ef_half=0.0, ef_search=64)
        # Attached under the name the workload actually gates on, and only if
        # the real driver implements it, so this fake cannot stand in for a
        # method PostgresDriver does not have.
        applied = []
        name = _iterative_scan_gate_name()
        assert hasattr(PostgresDriver, name)
        setattr(driver, name, applied.append)

        recorder = records.Recorder(str(tmp_path / "ops-fake.jsonl"))
        try:
            ctx = RunContext(
                run_id="t0-fake", resource_pass="normalized", recorder=recorder,
                engine="fake", engine_tag="none", march="x86-64-v3", k=5,
                cache_dir=str(tmp_path / "cache"))
            filtered_workload.run(
                ctx, driver, dataset, index_spec(dataset), ef_search=64,
                selectivities=(0.50,), max_queries=8, warmup=1,
                indexed_rows=len(dataset.train),
                # T3c: one column, so this stays a test about the MODE
                # reaching the record rather than about the column loop.
                columns=(FILTER_TAG,),
                iterative_scan=cli_mod._ops_iterative_scan(
                    "pgvector", load_profile("main"), {}, "normalized"),
            )
        finally:
            recorder.close()

        written = list(records.read_records(recorder.path))
        # Both modes, in the order declared, each SET once and each producing
        # its own record -- the end of the chain the run of 2026-10-15 got
        # halfway along. One record per mode and not one record overwritten:
        # the report keys filtered records on `extra.iterative_scan`, so these
        # two land beside each other rather than on top of one another, which
        # is why this change owes no OPS_MEASUREMENT_VERSION bump.
        assert applied == ["off", "relaxed_order"]
        assert [r["extra"]["iterative_scan"] for r in written] == [
            "off", "relaxed_order"]


class TestACapIsPrintedAsACapNotADuration:
    """A budget is a ceiling, not a duration, and the gap is not academic:
    Valkey loaded this corpus at 30,195 rows/s and then wrote the same rows
    back at 86, because pipelined writes into a populated index do not return.
    Modelling churn from the load rate alone would have under-reported by the
    whole budget, so both figures are carried."""

    def test_alisql_on_glove_is_capped_at_the_budget(self):
        """50.1 rows/s modelled: step 1 is 118,351 rows = 2,361 s and step 2 is
        177,527 rows = 3,541 s. Both exceed the 1,800 s budget, so typical
        collapses onto the ceiling: 2 x 1,800 s = 1.00 h."""
        profile = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                        "m_values": [16],
                                        "churn_fractions": [0.10, 0.25]}}
        t = term(estimate_load_hours(profile, ["alisql"], ["glove-100-angular"],
                                     ["normalized"], ["ops"]), "churn re-insert")
        # Under the ops v9 default of 7,200 s neither step reaches the cap:
        # typical is the modelled 2,361 + 3,541 s = 1.64 h, worst 2 x 7,200 s.
        # The cap-collapse this test is named for is pinned at 1,800 below.
        assert t["typical_hours"] == 1.64
        assert t["worst_hours"] == 4.00
        capped = {"name": "t", "ops": {**profile["ops"],
                                       "churn_budget_s": 1800}}
        t = term(estimate_load_hours(capped, ["alisql"], ["glove-100-angular"],
                                     ["normalized"], ["ops"]), "churn re-insert")
        assert (t["typical_hours"], t["worst_hours"]) == (1.00, 1.00)

    def test_a_profile_may_lower_the_budget(self):
        """ops.churn_budget_s is honoured by ops_pass.harness_args today."""
        profile = {"name": "t", "ops": {"enabled": True, "workloads": ["churn"],
                                        "m_values": [16], "churn_budget_s": 300,
                                        "churn_fractions": [0.10, 0.25]}}
        t = term(estimate_load_hours(profile, ["alisql"], ["glove-100-angular"],
                                     ["normalized"], ["ops"]), "churn re-insert")
        # 2 x 300 s = 600 s = 0.17 h, and both steps still exceed it.
        assert (t["typical_hours"], t["worst_hours"]) == (0.17, 0.17)

    def test_typical_never_exceeds_the_worst_case(self):
        for name in ("main", "full", "smoke", "quick", "dev"):
            e = est(name)
            assert e["query_hours"] <= e["query_hours_worst"], name
            assert e["typical_total_hours"] <= e["worst_case_total_hours"], name
            for t in e["query_terms"]:
                assert t["typical_hours"] <= t["worst_hours"], (name, t["name"])


class TestTheWholeQueryEstimateForMain:
    """The aggregate, so a reader can see the number the convention now refers
    to and check it against the parts: 1.75 h of concurrency windows, 4.77 h
    of modelled churn re-insert and 0.23 h of churn reader baseline, against a
    14.00 h churn ceiling and a 7.00 h filtered one.

    T3c moved the worst case and left the typical figure alone, which is what
    a budget does: 6 filtered points per ops unit x 300 s is a ceiling nothing
    had before, and no rate exists to turn any of it into a duration. So
    `query_hours` was unchanged at 6.87 and `query_hours_worst` went 16.10 ->
    23.10.

    T4b moved BOTH by the same 0.23 h, which is what an exact term does:
    `ops.churn_readers` readers are measured for 30 s before each churn step's
    deletes, main has 2 churn fractions, and the window is fixed by the
    profile rather than by the engine -- so it is a duration and a ceiling at
    once, like the concurrency windows. 7.10 and 23.33. The readers that run
    DURING the re-insert add no term: they stop when `_reinsert` returns, and
    that is `churn re-insert` above.

    The repaired re-measurement PR 4 §4.2 added is deliberately NOT a term
    here -- see `_QUERY_TERMS` for the measurement that reverted it, and
    `test_the_unmodelled_costs_are_named_one_by_one` for where it is named
    instead."""

    def test_the_parts_and_the_totals(self):
        e = est("main")
        assert [t["name"] for t in e["query_terms"]] == [
            "concurrency windows", "filtered search", "churn re-insert",
            "churn reader baseline", "mongot index-size settle"]
        # `churn re-insert`'s TYPICAL moved with the seventh engine's RATE,
        # and this is the one place the ingest table reaches the query
        # estimate: the typical figure is rows divided by the engine's load
        # rate, so Percona went from contributing no modelled duration at all
        # -- the honest reading while nothing had measured it -- to 0.46 h,
        # which is 10%+15% of 1,183,514 rows at 275 rows/s plus 10%+15% of
        # 1,000,000 at 427, one ops unit each. Its WORST did not move: the
        # budget is a ceiling the profile sets per step and has always applied
        # to every engine whether or not anyone had measured it.
        # Ops measurement version 9: the churn budget default is 7200 s per
        # step (was 1800) and every client count pays one discarded
        # warm-up window, so the figures below moved from the ones the
        # docstring narrates; the arithmetic is the same with those two
        # inputs.
        assert term(e, "churn re-insert")["typical_hours"] == 6.33
        # 18.00, not 16.00: MyVector's two ops units are budgeted like every
        # engine's, while its TYPICAL adds nothing -- no measured rate.
        assert term(e, "churn re-insert")["worst_hours"] == 72.00
        assert term(e, "filtered search")["typical_hours"] == 0.0
        # 10.0 and not 8.00 because the iterative-scan modes now reach the
        # run. `main` declares two and only pgvector has the knob, so of the
        # 16 ops units across seven engines the 4 pgvector ones pay 12 points
        # x 300 s and the other 12 pay 6 -- 2.0 h more, charged to one engine.
        # A `* modes` applied to the whole term instead would have charged the
        # other six for a dimension they do not have, which is the mistake
        # `_charged_scan_modes` exists to refuse.
        # Unmoved by the eighth engine: MyVector's driver declines filtered
        # search, and `_runs_filtered_search` keeps it out of this term.
        assert term(e, "filtered search")["worst_hours"] == 10.00
        # 2 fractions x 30 s x 16 ops units across main's SEVEN engines and
        # two datasets = 960 s = 0.267 h, and it is both the typical figure
        # and the ceiling because the window is a fixed span. Over six
        # engines it was 14 units, 840 s and 0.23 h.
        # 18 ops units over eight engines: 1,080 s = 0.30 h.
        assert term(e, "churn reader baseline")["typical_hours"] == 0.30
        assert term(e, "churn reader baseline")["worst_hours"] == 0.30
        # Neither query figure moves with the ann M grid: every term here
        # is charged per OPS unit, and `ops.m_values` is untouched by T5b.
        # That is the half of the D1 change worth pinning -- if a query term
        # ever starts scaling with the ann grid, these two lines catch it.
        # +0.02 / +0.33: mongot's index-size settle, one 30 s interval
        # typical and the 600 s cap worst, after each of mongodb's two
        # builds. Version 9 added the wait to the run and not to the
        # estimate until the server-wait-evidence change counted it.
        # +0.33 / +0.34 with VillageSQL's concurrency windows (it declines
        # filtered search and churn, so those terms charge it nothing).
        # +0.34 / +0.33 again with villagesql84's, for the same reason.
        assert e["query_hours"] == 10.32
        assert e["query_hours_worst"] == 86.30
        # The totals move with the ENGINE LIST as well as with the ann M
        # grid. Over six engines these were 100.16 and 116.39, before the
        # iterative-scan modes added 2.0 h to pgvector alone. The INGEST half
        # of both used to be 93.06 over six engines, because Percona was named
        # in `engines_without_ingest_rate` and contributed nothing to it; it
        # is 100.44 over seven now that its rate is measured, and the 7.38 h
        # of the difference is ingest and not query time.
        # 110.42 / 186.41 with VillageSQL's concurrency windows; 110.76 /
        # 186.74 with villagesql84's.
        assert e["typical_total_hours"] == 110.76
        assert e["worst_case_total_hours"] == 186.74

    def test_the_totals_are_the_sums_a_reader_can_do_on_the_printed_lines(self):
        e = est("main")
        assert e["typical_total_hours"] == round(
            e["ingest_hours"] + e["query_hours"], 2)
        assert e["worst_case_total_hours"] == round(
            e["ingest_hours"] + e["query_hours_worst"], 2)

    def test_the_per_engine_split_mirrors_the_ingest_one(self):
        e = est("main")
        # The asymmetry this test was named for is gone, and that is worth
        # recording rather than quietly deleting: Percona used to appear here
        # with a query figure while being absent from `per_engine_hours`,
        # because query terms are spans the profile puts a clock on and need
        # no engine rate, while the ingest figure is rows divided by a rate
        # and needs one. It now has a rate, so both splits name all seven.
        # Ops measurement version 9: the churn budget default is 7200 s per
        # step (was 1800) and every client count pays one discarded
        # warm-up window, so the figures below moved from the ones the
        # docstring narrates; the arithmetic is the same with those two
        # inputs.
        assert e["query_per_engine_hours"] == {
            "mariadb": 1.46, "alisql": 3.47, "pgvector": 0.85,
            "mariadb123": 1.92, "mongodb": 0.39, "valkey": 0.37,
            "percona": 0.83, "myvector": 0.37, "villagesql": 0.33,
            "villagesql84": 0.33,
        }
        # The asymmetry is back, for the eighth engine: MyVector has query
        # figures (spans the profile clocks) and no ingest one (no rate).
        # And for the ninth, VillageSQL, for the same reason.
        # And villagesql84, the same extension on the 8.4 server line.
        assert (set(e["query_per_engine_hours"]) - set(e["per_engine_hours"])
                == {"myvector", "villagesql", "villagesql84"})
        assert e["engines_without_ingest_rate"] == ["myvector", "villagesql",
                                                    "villagesql84"]
        # The mechanism that made the asymmetry possible is still there, and
        # an engine nothing has measured still falls out of the ingest split
        # alone. This is the non-vacuous half: `tests/test_config.py` injects
        # the same probe engine for the same reason -- a guard that needs some
        # real engine to be unmeasured fires on the day one stops being.
        probe = est("main", engines=ALL + ["e2-no-such-engine"])
        assert probe["engines_without_ingest_rate"] == [
            "myvector", "villagesql", "villagesql84", "e2-no-such-engine"]
        assert "e2-no-such-engine" not in probe["per_engine_hours"]
        assert "e2-no-such-engine" in probe["query_per_engine_hours"]

    def test_ops_disabled_means_no_ops_terms(self):
        e = est("m-sweep")
        assert e["query_terms"] == []
        assert e["query_hours"] == 0.0


# ---------------------------------------------------------------------------
# 4. What it cannot know, reported rather than guessed
# ---------------------------------------------------------------------------

class TestAMissingQueryRateIsReportedNotInvented:
    """The ingest path names datasets it has no rate for instead of scoring
    them zero. Query time needs the same honesty about engines: the per-query
    workloads (filtered, the churn re-measurements, the warm-ups, the whole
    recall sweep) cost a number of queries, and nothing in this repo records a
    per-(engine, dataset) QPS to multiply them by."""

    def test_every_engine_is_named_while_the_table_is_empty(self):
        assert est("main")["engines_without_query_rate"] == ALL

    def test_a_measured_rate_closes_the_gap_for_that_engine(self, monkeypatch):
        """Proves the list is derived from the table rather than hardcoded."""
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "glove-100-angular"), 326.0)
        monkeypatch.setitem(est_mod._MEASURED_QUERY_QPS,
                            ("mariadb", "sift-128-euclidean"), 326.0)
        assert "mariadb" not in est("main")["engines_without_query_rate"]
        assert "alisql" in est("main")["engines_without_query_rate"]

    def test_the_unmodelled_costs_are_named_one_by_one(self):
        names = est("main")["query_costs_not_modelled"]
        assert any("ef_search" in n for n in names)          # the recall sweep
        assert any("churn" in n for n in names)
        assert any("warm-up" in n for n in names)

    def test_the_churn_repair_path_is_disclosed_and_not_charged(self):
        """PR 4 §4.2's repair path, in one list and never both.

        The same rule `test_the_filtered_query_loop_is_charged_and_not_also_
        disclaimed` enforces in the other direction: output that charges for a
        cost and calls it unmodelled in the same breath tells a reader
        nothing. Anchored on the entry's own prefix rather than on the
        substring "maintain", which `churn re-measurement` could grow at any
        time without anyone noticing this assertion had stopped binding.
        """
        e = est("main")
        assert "churn maintenance" not in [t["name"] for t in e["query_terms"]]
        repair = [n for n in e["query_costs_not_modelled"]
                  if n.startswith("churn repair path:")]
        assert len(repair) == 1, repair
        assert "driver.maintain() per churn fraction" in repair[0]
        assert "no measured rate" in repair[0]
        # The reader baseline IS charged, so it must not also be disclaimed.
        assert "churn reader baseline" in [t["name"] for t in e["query_terms"]]
        assert not [n for n in e["query_costs_not_modelled"]
                    if "reader" in n], e["query_costs_not_modelled"]

    def test_the_filtered_query_loop_is_charged_and_not_also_disclaimed(self):
        """The query loop may appear in one list or the other, never both:
        output that charges for a cost and calls it unmodelled in the same
        breath tells a reader nothing.

        Its GROUND TRUTH is a different cost in neither figure -- harness CPU,
        over the whole test set rather than `ops.max_queries`, and outside the
        budget -- so it is disclosed. This pins exactly which filtered-related
        disclosures exist rather than asserting none do, which is stricter
        than the bare substring check it replaces: a second disclosure, or one
        about the query loop, fails here instead of passing.
        """
        e = est("main")
        assert "filtered search" in [t["name"] for t in e["query_terms"]]
        mentions = [n for n in e["query_costs_not_modelled"] if "filtered" in n]
        assert len(mentions) == 1, mentions
        assert mentions[0].startswith("filtered ground truth:")
        # The thing the term already charges for must not be in here.
        assert "ops.max_queries queries per point" not in mentions[0]
        assert "~150 s per million qualifying rows" in mentions[0]
        # M-2: the figure has to range over the PROFILE, not over one of its
        # two datasets. Pairing a per-dataset number with main's two-dataset
        # ceiling understated this by nearly half, and the same class of error
        # (a per-unit number against a per-profile denominator) has cost this
        # repo a 46-vs-51.1 h docs drift and a 270 h one in CLAUDE.md.
        assert "main's two datasets" in mentions[0]
        assert "glove" not in mentions[0], (
            "a per-dataset figure cannot be quoted against a per-profile "
            "ceiling")

    def test_an_ann_only_run_still_declares_its_query_cost(self):
        e = est("m-sweep")
        assert e["query_terms"] == []
        assert any("ef_search" in n for n in e["query_costs_not_modelled"])
        assert e["engines_without_query_rate"] == ALL

    def test_nothing_is_claimed_when_nothing_queries(self):
        profile = {"name": "t", "ann": {"enabled": False},
                   "ops": {"enabled": False}}
        e = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ann", "ops"])
        assert e["query_costs_not_modelled"] == []
        assert e["engines_without_query_rate"] == []


# ---------------------------------------------------------------------------
# 5. The extension point the four later tasks will use
# ---------------------------------------------------------------------------

class TestTheExtensionPoint:
    """T1d (concurrency repeats), T2a (the ef-calibration sweep), T3c (the
    filtered budget) and T4b (the churn reader baseline) each owe this
    function one named term. Adding one is: write a function taking a
    `_QueryPlan` and returning a `_QueryTerm` or None, and append it to
    `_QUERY_TERMS`. Nothing else may need to change, and this test is what
    says so.

    T4a's churn maintenance was the fifth, and it is the one that came back as
    a disclosure instead. It was written as a term and reverted on a
    measurement: the repaired re-measurement is a churn query loop that
    nothing caps, so the term had to report `bounded=False`, and
    `query_worst_case_is_a_ceiling` is an `all()` over the terms -- so that
    one term turned the printed worst case from a ceiling into a figure for
    main, smoke, quick, full and tuned-complete at once. The two churn
    measurements that predate PR 4 are disclosed rather than charged for
    exactly that reason, and this one now joins them. A ceiling nobody
    measured is worse than a named cost outside the figure, and
    `test_the_churn_repair_path_is_disclosed_and_not_charged` below is what
    keeps it in one list and not both."""

    def test_an_added_term_reaches_every_total_and_the_print(self, monkeypatch):
        before = est("main")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (_one_hour_per_pass_term,))
        after = est("main")
        # In the list, not necessarily last: the tally orders terms by first
        # appearance, and `mongot index-size settle` first appears at the
        # mongodb engine, after the test term's first (mariadb) appearance.
        assert "test term" in [t["name"] for t in after["query_terms"]]
        # 14.0 and not 12.0: `_one_hour_per_pass_term` is one hour per
        # (engine, dataset, pass) and `main` is seven engines x two datasets
        # x one pass. Six engines gave 12.0.
        # 16.0 with MyVector, the eighth engine; 18.0 with VillageSQL; 20.0
        # with villagesql84.
        assert after["query_hours"] == round(before["query_hours"] + 20.0, 2)
        assert after["query_hours_worst"] == round(
            before["query_hours_worst"] + 20.0, 2)
        assert after["worst_case_total_hours"] == round(
            before["worst_case_total_hours"] + 20.0, 2)
        out = printed("main")
        assert "test term" in out
        assert "one hour, flat" in out

    def test_a_term_may_decline_this_profile(self, monkeypatch):
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (lambda plan: None,))
        # Unchanged by a term that declines: `main`'s query subtotal over all
        # seven engines. It was 6.76 over six; Percona adds one ops unit per
        # dataset, 0.28 h of it while its ingest rate was unmeasured and
        # 0.74 h now that `churn re-insert` can model a duration for it.
        # 7.78 with MyVector, whose 0.28 h is its concurrency windows and
        # reader baseline -- no churn duration, since it has no rate.
        # 9.98 with VillageSQL, whose 0.33 h is its concurrency windows alone:
        # it declines churn (`capabilities.churn: false`). 10.32 with
        # villagesql84's 0.33 h (+0.01 rounding), for the same reason.
        assert est("main")["query_hours"] == 10.32  # + mongot settle

    def test_scope_unit_is_charged_per_ops_unit_and_pass(self, monkeypatch):
        """main has 8 ops units per dataset x 2 datasets = 16 per pass.

        7 and 14 over the six engines that preceded Percona. Every engine is
        one unit except pgvector, which runs both of its build modes on
        `main`; Percona runs both only under `config/profiles/percona.yml`.
        """
        def one_second_per_unit(plan):
            return est_mod._QueryTerm(
                name="unit term", typical_s=1.0, worst_s=1.0,
                basis="one second", bounded=True, scope="unit")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS", (one_second_per_unit,))
        one = est("main")
        two = est("main", passes=("normalized", "tuned"))
        # 18 with MyVector: one more unit per dataset; 20 with VillageSQL;
        # 22 with villagesql84.
        assert term(one, "unit term")["typical_seconds"] == 22.0
        assert term(two, "unit term")["typical_seconds"] == 44.0

    def test_scope_pass_skips_the_ops_unit_multiplier(self, monkeypatch):
        """14, not 16: SEVEN engines x two datasets, where unit scope would
        have charged pgvector twice for its two build modes.

        The pair was 12 against 14 over six engines. What is being pinned is
        the GAP between the two scopes -- one engine's worth of double
        charging -- and that gap is still exactly one unit per dataset however
        many engines there are."""
        def one_second_per_pass(plan):
            return est_mod._QueryTerm(
                name="pass term", typical_s=1.0, worst_s=1.0,
                basis="one second", bounded=True, scope="pass")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS", (one_second_per_pass,))
        # 16 with MyVector, the eighth engine; the gap to unit scope is
        # still pgvector's one extra unit per dataset.
        # 18 with VillageSQL, the ninth; 20 with villagesql84.
        assert term(est("main"), "pass term")["typical_seconds"] == 20.0
        assert term(est("main", passes=("normalized", "tuned")),
                    "pass term")["typical_seconds"] == 40.0

    def test_an_unknown_scope_is_refused_loudly(self, monkeypatch):
        """Silently charging it once per (engine, dataset) is how a term meant
        to be counted once became twelve."""
        def nonsense(plan):
            return est_mod._QueryTerm(
                name="bad", typical_s=1.0, worst_s=1.0, basis="-",
                bounded=True, scope="run")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS", (nonsense,))
        with pytest.raises(ValueError, match="scope"):
            est("main")

    def test_a_term_declares_whether_its_worst_case_is_a_ceiling(self):
        """An unbounded term must not be read as a worst case. Both of today's
        terms are bounded by the configuration; a future one may not be."""
        def unbounded(plan):
            return est_mod._QueryTerm(
                name="open ended", typical_s=60.0, worst_s=60.0,
                basis="no cap exists", bounded=False, scope="pass")
        with mock.patch.object(est_mod, "_QUERY_TERMS", (unbounded,)):
            e = est("main")
            assert term(e, "open ended")["bounded"] is False
            assert e["query_worst_case_is_a_ceiling"] is False
            assert "open ended" in printed("main")
        # And with only the two bounded terms, it is a ceiling again.
        with mock.patch.object(est_mod, "_QUERY_TERMS",
                               (est_mod._concurrency_windows,
                                est_mod._churn_reinsert)):
            assert est("main")["query_worst_case_is_a_ceiling"] is True

    def test_the_plan_a_term_receives_carries_what_it_needs(self):
        """Documented so an implementer does not have to read the accumulator."""
        seen = []

        def spy(plan):
            seen.append(plan)
            return None

        with mock.patch.object(est_mod, "_QUERY_TERMS", (spy,)):
            est("main", engines=["mariadb"], datasets=["glove-100-angular"])
        assert len(seen) == 1
        plan = seen[0]
        assert plan.engine == "mariadb"
        assert plan.dataset == "glove-100-angular"
        assert plan.rows == 1_183_514
        assert plan.write_rows_per_s == 147.0
        assert plan.query_qps is None
        assert plan.units == 1
        assert plan.passes == 1
        assert plan.workloads == ["build", "concurrency", "filtered", "churn"]
        assert plan.phases == ["ann", "ops"]
        assert plan.ops["max_queries"] == 500
        assert plan.k == 10


# ---------------------------------------------------------------------------
# 6. The printed output
# ---------------------------------------------------------------------------

class TestThePrintedOutput:
    def test_the_ingest_block_is_unchanged_word_for_word(self):
        out = printed("main")
        assert ("\nestimated ingest time (loading only, before any queries):\n"
                "  mariadb ~17.5 h  |  alisql ~49.7 h  |  pgvector ~0.9 h  |  "
                "mariadb123 ~24.8 h  |  mongodb ~0.1 h  |  valkey ~0.1 h  |  "
                "percona ~7.4 h\n"
                "  total ~100.4 h across 1 pass(es), 3 M value(s)\n") in out
        assert "! This is a long run." in out

    def test_query_time_is_a_separate_block_with_both_figures(self):
        out = printed("main")
        assert "estimated query time" in out
        assert "concurrency windows" in out
        assert "churn re-insert" in out
        # pgvector's ingest line. Not a query figure -- it is here because
        # this block is asserted against the whole printed output, and it
        # moved with the ann grid where the two query figures below did not.
        assert "pgvector ~0.9 h" in out
        # The churn line, at ops v9's 7,200 s budget.
        assert "~6.3 h typical" in out
        assert "~72.0 h worst case" in out

    def test_a_term_with_a_ceiling_but_no_duration_prints_both(self):
        """Filtered search used to print "cost not quantified", because it had
        neither a rate nor a cap. T3c gave it a cap, so it now prints a
        typical figure of ~0.0 h beside a real worst case -- and the pair is
        the honest reading: nothing has measured the rate, and the budget says
        what it cannot exceed regardless.

        `~0.0 h typical` alone would read as free. It is only not misleading
        because the worst case sits on the same line."""
        out = printed("main")
        # Padded to the longest term name, `mongot index-size settle`.
        assert ("filtered search           ~0.0 h typical, ~10.0 h worst case"
                in out)
        # Anchored on the `=`, because "3,000 queries per ops unit" is a
        # substring of "13,000 queries per ops unit" -- the shape that has
        # cost this series four tests.
        assert "= 3,000 queries per ops unit" in out
        assert "at a rate nothing here has measured" in out

    def test_the_total_is_printed(self):
        out = printed("main")
        # + mongot settle, + VillageSQL, + villagesql84
        assert "~186.7 h worst case" in out
        assert "~110.8 h typical" in out

    def test_the_bug_criterion_is_only_claimed_when_it_holds(self, monkeypatch):
        """Two adjacent lines used to say the worst case was a figure a run
        cannot exceed and then that a run can exceed it. The caveat comes
        first and the claim is conditional on it.

        Since T3c every configured profile makes the claim, because filtered
        search was the only unbounded term and the budget closed it. So the
        case worth pinning flipped: the claim is asserted on the real profile
        and the caveat is reached by injecting an unbounded term, which is the
        direction that cannot pass vacuously -- an `all()` over an empty or
        all-bounded list is True, and the caveat branch would otherwise never
        be executed by any test again."""
        out = printed("main")
        assert "a run materially longer than it is a bug" in out
        assert "Not yet a ceiling" not in out
        # The qualifier that makes "materially" mean something, and it has to
        # cover churn as well as filtered search: both print a ceiling and
        # both overrun by one unit of work from the same `>` comparison.
        assert "can overrun" in out
        assert "the one query or slice already running" in out

        # Inject one unbounded term and the caveat comes back, after the run
        # line and naming the term responsible.
        def unbounded(plan):
            return est_mod._QueryTerm(
                name="unbounded probe", typical_s=60.0, worst_s=60.0,
                basis="no cap exists", bounded=False, scope="pass")
        monkeypatch.setattr(est_mod, "_QUERY_TERMS",
                            tuple(est_mod._QUERY_TERMS) + (unbounded,))
        loose = printed("main")
        run_line = loose.index("estimated run time")
        caveat = loose.index("Not yet a ceiling")
        assert caveat > run_line
        assert "nothing caps unbounded probe" in loose
        assert "a run materially longer than it is a bug" not in loose
        assert "can overrun" not in loose
        assert "largest figure this configuration justifies" in loose

    def test_the_printed_total_is_the_dict_total(self):
        """The two cannot drift: the printed figure is the dict's own."""
        e = est("main")
        assert f"~{e['worst_case_total_hours']:.1f} h worst case" in printed("main")

    def test_what_it_cannot_know_is_printed_too(self):
        """Asserted against the whole engine list in the disclosure sentence,
        not engine names loose in the output: the ingest block above prints
        every engine, so `"mariadb" in out` passed even with the gap list
        emptied — and `"mariadb"` is a substring of `"mariadb123"`, which has
        now cost this series four tests."""
        out = printed("main")
        assert ("there is no measured query rate for "
                f"{', '.join(ALL)} to multiply them by") in out
        assert "ef_search" in out
        assert "churn delete" in out

    def test_a_trivial_run_still_prints_nothing(self):
        """`dev` is a one-minute framework exercise; an estimate for it is
        noise. The gate is the worst-case total, not ingest alone.

        This is what `dev.yml`'s own `filtered_budget_s: 60` is for. Four
        filtered points at the 300 s default would be a 2.0 h worst case for
        a run of about six minutes across six engines -- 20x -- which clears
        the 1 h threshold this function gates on and makes the shortest
        profile in the repo print an estimate at all.

        Not to be confused with `long_run`, which is keyed on ingest above
        12 h and is False for `dev` under either budget: that flag decides
        whether the block carries the M-reload caveat, not whether the block
        appears."""
        assert printed("dev") == ""

    def test_a_profile_whose_cost_is_query_time_is_no_longer_silent(self):
        """smoke is 0.59 h of ingest — under the old 1 h gate, so it printed
        nothing at all — against a 4.0 h churn ceiling across seven engines.

        It was 0.49 h while that was six engines' ingest out of the seven the
        gate runs: Percona had no measured rate, was named rather than
        modelled, and the printed block carried a line saying the figure was a
        floor. The rate is measured now, so the figure covers all seven and
        that line is correctly absent — asserted below, because its
        disappearance is the published consequence of this change on the
        profile a reader is most likely to run. The worst case was ~5.2 h over
        six engines and ~5.9 h before Percona's ingest joined it."""
        out = printed("smoke")
        assert "estimated ingest time" in out
        assert "churn re-insert" in out
        # ~20.1 h since ops v9's 7,200 s budget: four 2 h churn ceilings
        # per engine step where there were four half hours.
        # ~20.2 h with mongot's 600 s settle cap counted; ~20.3 h with
        # villagesql84's concurrency windows.
        assert "~20.3 h worst case" in out
        # MyVector, VillageSQL and villagesql84 are named as unmeasured;
        # nothing else is.
        assert ("No measured ingest rate for myvector, villagesql, "
                "villagesql84, so" in " ".join(out.split()))
        assert est("smoke")["engines_without_ingest_rate"] == [
            "myvector", "villagesql", "villagesql84"]


# ---------------------------------------------------------------------------
# 7. The web UI reads the same dict
# ---------------------------------------------------------------------------

class TestTheDocumentedFiguresAreTheOnesTheCodeProduces:
    """docs/07-planning-a-run.md is what a reader consults before committing
    days of machine time, and its figures had already drifted: the sample
    output block said 45.8 h for `main` while the estimator said 51.1, because
    measured rates were added to the table afterwards and nobody re-ran it.

    The planning table is therefore executable, like `verify-alisql-traps.sh`
    is for the engine notes. It is written for the three original engines, one
    or two passes, which is what the surrounding prose says.
    """

    THREE = ["mariadb", "alisql", "pgvector"]
    ROW = re.compile(
        r"^\| `(?P<profile>[a-z-]+)` \| (?P<passes>\d) \| ~(?P<ingest>[\d.]+) h"
        r" \| ~(?P<typical>[\d.]+) h \| ~(?P<worst>[\d.]+) h"
        r" \| \*\*~(?P<run>[\d.]+) h\*\* \|$", re.M)

    def _rows(self):
        path = os.path.join(VB_ROOT, "docs", "07-planning-a-run.md")
        with open(path) as fh:
            found = list(self.ROW.finditer(fh.read()))
        assert len(found) >= 6, "the planning table is gone or was reshaped"
        return found

    def test_every_documented_profile_matches_the_estimator(self):
        for row in self._rows():
            name = row.group("profile")
            passes = ["normalized", "tuned"][:int(row.group("passes"))]
            e = est(name, engines=self.THREE, passes=passes)
            for column, key in (("ingest", "ingest_hours"),
                                ("typical", "query_hours"),
                                ("worst", "query_hours_worst"),
                                ("run", "worst_case_total_hours")):
                assert float(row.group(column)) == pytest.approx(e[key], abs=0.05), (
                    f"docs/07 {name} {column}: says {row.group(column)}, "
                    f"estimator says {e[key]:.2f}")

    def test_the_sample_output_block_is_what_the_tool_prints(self):
        """The whole fenced block, not a few lines from it. Pinning only the
        numeric lines covered the 45.8-vs-51.1 drift and left sixteen lines of
        basis and caveat prose free to rot: changing a basis string left the
        suite green and docs/07 misquoting the tool."""
        path = os.path.join(VB_ROOT, "docs", "07-planning-a-run.md")
        with open(path) as fh:
            doc = fh.read()
        block = re.search(
            r"A run prints all of it before starting:\n\n```\n(?P<body>.*?)```",
            doc, re.S)
        assert block, "the sample output block moved or lost its heading"
        assert block.group("body").strip() == printed(
            "main", engines=self.THREE).strip()

    def test_claude_mds_standing_figure_is_the_one_the_command_produces(self):
        """CLAUDE.md restates the cost convention as standing guidance, and the
        sentence names a command rather than an engine list: with `--engines`
        omitted, `cli.py` resolves to all seven KNOWN_ENGINES and `full.yml`
        declares no engines key. Quoting docs/07's three-engine figures there
        left an operator planning ~270 h short of the worst case.

        It fired again when Percona became the seventh engine: the worst case
        over `full` at both passes went from ~1,006 h to ~1,046 h, because an
        engine in the registry is an engine that command runs. The INGEST
        figure did not move then -- it was 726 h over the six engines with a
        measured rate, with Percona named rather than modelled -- so the
        sentence had to say which of its two numbers covered which engines.

        It moved when the seventh rate was measured. Both numbers now cover
        all seven, the scope split is gone, and the sentence says ~796 h and
        ~1,135 h. The assertions below pin both against the estimator AND
        against the six-engine figure, so the old 726 h cannot satisfy them:
        a revert of the rate table reddens this test rather than agreeing
        with a document nobody re-ran."""
        path = os.path.join(VB_ROOT, "CLAUDE.md")
        with open(path) as fh:
            claude = fh.read()
        sentence = re.search(
            r"`--profile full\n--resource-pass both` is ~(?P<ingest>[\d,]+) h of "
            r"ingest[^~]*~(?P<worst>[\d,]+) h of run time\s*\nin the worst case",
            claude)
        assert sentence, "the standing cost sentence moved or was reworded"
        # What the command measures when `--engines` is omitted: the
        # default-selected engines, which leave out the opt-in VillageSQL.
        e = est("full", engines=list(cli_mod.DEFAULT_ENGINES),
                passes=("normalized", "tuned"))
        assert float(sentence.group("ingest").replace(",", "")) == pytest.approx(
            e["ingest_hours"], abs=0.5)
        assert float(sentence.group("worst").replace(",", "")) == pytest.approx(
            e["worst_case_total_hours"], abs=0.5)
        # Pinned as VOCABULARIES over whitespace-normalised text, not as exact
        # phrases. The old `"all six engines" in claude` was neither: a reflow
        # could put a newline inside a phrase that sits mid-sentence in
        # hand-wrapped prose, and an exact phrase rejects correct rewrites --
        # `"all seven engines"` fires on "all seven of the engines", "every one
        # of the seven engines" and "all 7 engines", all of which say the same
        # thing. The vocabularies live in `tests/test_e2_percona_wiring.py`
        # with their own accepts and rejects asserted beside them.
        from tests.test_e2_percona_wiring import _prose
        _prose("the engine count is eight", claude)
        # The scope split is BACK, for MyVector: the worst case covers all
        # eight engines and the ingest figure the seven with a measured rate,
        # and the sentence says so. VillageSQL, the ninth, is opt-in and not
        # in a bare run, so not in either figure.
        _prose("the ingest figure covers seven of the eight", claude)
        assert e["engines_without_ingest_rate"] == ["myvector"]
        assert "villagesql" not in e["per_engine_hours"]
        assert "percona" in e["per_engine_hours"]
        # Non-vacuity, and the guard that replaces the deleted claim: the
        # document's figure must be the SEVEN-engine one. `abs=0.5` above is
        # loose enough that the published six-engine 726 h would satisfy it
        # if the rate table were reverted and the document were not, so the
        # two figures are asserted to be different quantities.
        six = est("full", engines=[x for x in cli_mod.DEFAULT_ENGINES
                                   if x != "percona"],
                  passes=("normalized", "tuned"))
        assert round(e["ingest_hours"] - six["ingest_hours"], 2) == 69.83
        assert float(sentence.group("ingest").replace(",", "")) != pytest.approx(
            six["ingest_hours"], abs=0.5)

    #: README's profile table claims "measured ingest only, the original
    #: three engines", which is docs/07 §1's hand method: the flat per-engine
    #: rates times the number of full corpus loads. Spelled here because the
    #: rates live in prose in two documents and nothing executable held them.
    HAND_RATES = {"mariadb": 150.0, "alisql": 55.0, "pgvector": 3000.0}

    #: Every row of README's profile table: `(basis, passes, tolerance)`.
    #:
    #: The basis is per row because the table's is not uniform, and pretending
    #: otherwise is how two of these went wrong. Four rows come from the flat
    #: rates above. `mariadb-blog` does NOT: dbpedia-openai-1000k is 1,536
    #: dims and loads far slower than those rates imply -- docs/07 measures
    #: MariaDB at 3.3 h for its million rows, about 84 rows/s against the flat
    #: 150 -- so that row comes from the estimator, which carries a dimension
    #: penalty. Using the flat method there gives 14 h against the estimator's
    #: 21 h, and README carried 72 h.
    #:
    #: Tolerances are the row's own rounding, not slack for drift: a row
    #: printed as "~35 h" cannot be pinned tighter than its last digit. The
    #: `smoke` row is in MINUTES and rounded to the nearest ten, so it gets
    #: ten minutes.
    README_ROWS = {
        "smoke": ("hand", 1, 0.17),
        "quick": ("hand", 1, 1.0),
        "main": ("hand", 1, 1.0),
        "full": ("hand", 2, 2.0),
        "mariadb-blog": ("estimator", 1, 1.0),
    }

    #: The one row with no pin, and why. `_DATASET_ROWS` has no count for
    #: `tiny-16-euclidean` -- it is generated by `make-tiny-dataset.sh`, not
    #: downloaded -- so neither basis can produce a figure, and "~1 min" is a
    #: statement about a synthetic corpus rather than a derived number.
    #: Named here so `test_every_readme_row_is_pinned_or_named` can insist the
    #: exclusion list is exactly this and a new row cannot slip in unpinned.
    README_UNPINNED = {"dev"}

    def _hand_ingest_hours(self, profile_name, passes=1,
                           honour_subset=True):
        """The hand method, over whichever datasets the profile declares.

        `len(ann.m_values) + len(ops.m_values)` full reloads per
        (engine, dataset), because every M value reloads the whole corpus on
        both paths. Verified against the figures docs/07 carried before T5b:
        45.8 h for `main` at `[8, 32]`, 24.8 h for glove alone, 8.3 h for
        glove ops-only.

        `honour_subset` charges the OPS phase `ops.subset_rows` where the
        profile sets one, and it is not optional for correctness: `smoke`
        caps its ops phase at 20,000 of fashion-mnist's 60,000, and ignoring
        that overstates the row by 16 minutes -- which is how a round of this
        task came to report README's `smoke` figure as stale when it was
        right. The parameter exists so
        `test_the_subset_cap_is_what_makes_smoke_agree` can show the
        difference rather than assert it.
        """
        profile = load_profile(profile_name)
        ann = profile.get("ann") or {}
        ops = profile.get("ops") or {}
        ann_loads = (len(ann.get("m_values", [16]))
                     if ann.get("enabled", True) else 0)
        ops_loads = (len(ops.get("m_values", [16]))
                     if ops.get("enabled", True) else 0)
        subset = ops.get("subset_rows") if honour_subset else None
        total = 0.0
        for dataset in profile.get("datasets", []):
            rows = est_mod._DATASET_ROWS.get(dataset)
            assert rows, f"no row count for {dataset}"
            ops_rows = min(subset, rows) if subset else rows
            for rate in self.HAND_RATES.values():
                total += rows / rate * ann_loads
                total += ops_rows / rate * ops_loads
        return total / 3600.0 * passes

    def _expected_readme_hours(self, profile_name):
        basis, passes, _tol = self.README_ROWS[profile_name]
        if basis == "hand":
            return self._hand_ingest_hours(profile_name, passes=passes)
        return est(profile_name, engines=self.THREE,
                   passes=("normalized", "tuned")[:passes])["ingest_hours"]

    #: A row of README's profile table. The figure is optionally bold, and
    #: optionally in minutes, so both are captured rather than assumed.
    README_ROW = re.compile(
        r"^\| \*{0,2}`(?P<profile>[a-z-]+)`\*{0,2} \|[^|]*\| "
        r"\*{0,2}~(?P<value>[\d.]+) (?P<unit>h|min)\*{0,2} \|", re.M)

    def _readme_table(self):
        path = os.path.join(VB_ROOT, "README.md")
        with open(path) as fh:
            readme = fh.read()
        rows = {m.group("profile"): m for m in self.README_ROW.finditer(readme)}
        assert len(rows) >= 6, (
            f"README's profile table is gone or was reshaped; found "
            f"{sorted(rows)}")
        return readme, rows

    def test_the_hand_method_reproduces_the_pre_t5b_published_figures(self):
        """The arithmetic, pinned against three figures docs/07 published.

        **This leg does NOT protect the helper from common-mode error, and an
        earlier revision of this docstring claimed it did.** It defines its
        own `hand()` closure below and never calls `_hand_ingest_hours`; the
        two formulas differ in the ops term, this one hard-coding `+ 1` where
        the helper takes `len(ops.m_values)`. So it pins a SIBLING
        implementation: it establishes that the flat-rate arithmetic
        reproduces 45.8 / 24.8 / 8.3 h, which is worth having because those
        are the numbers docs/07 carried before T5b, but a helper wrong in the
        same direction as the README would survive it.

        What does protect against that is
        `test_every_readme_ingest_figure_matches_its_basis`'s `full` row,
        which runs `_hand_ingest_hours` itself at `len(ops.m_values) == 3` --
        a branch the closure here cannot reach -- and agrees with README's
        ~272 h. That leg is the control; this one is the reference.
        """
        rows = est_mod._DATASET_ROWS
        glove, sift = "glove-100-angular", "sift-128-euclidean"

        def hand(m_values, datasets, phases=("ann", "ops")):
            loads = (len(m_values) if "ann" in phases else 0) + (
                1 if "ops" in phases else 0)
            return sum(rows[d] / rate * loads
                       for d in datasets
                       for rate in self.HAND_RATES.values()) / 3600.0

        assert hand([8, 32], [glove, sift]) == pytest.approx(45.8, abs=0.1)
        assert hand([8, 32], [glove]) == pytest.approx(24.8, abs=0.1)
        assert hand([8, 32], [glove], phases=("ops",)) == pytest.approx(
            8.3, abs=0.1)
        assert load_profile("main")["ann"]["m_values"] == [8, 16, 32]

    def test_the_subset_cap_is_what_makes_smoke_agree(self):
        """`ops.subset_rows`, and the wrong flag it produced when ignored.

        Honest (the cap honoured): 0.56 h, 34 minutes, against README's
            "~30 min".
        Mutant (the cap ignored, which is how this helper was first written):
            0.84 h, 50 minutes -- which reads as a stale row and was reported
            as one. The row was right and the method was wrong.
        """
        honest = self._hand_ingest_hours("smoke")
        mutant = self._hand_ingest_hours("smoke", honour_subset=False)
        assert load_profile("smoke")["ops"]["subset_rows"] == 20000
        assert est_mod._DATASET_ROWS["fashion-mnist-784-euclidean"] == 60000
        assert honest == pytest.approx(0.56, abs=0.01)
        assert mutant == pytest.approx(0.84, abs=0.01)
        assert round(honest * 60) == 34 and round(mutant * 60) == 50
        # The honest figure agrees with the published row; the mutant does
        # not, which is the whole content of the mistake.
        _readme, rows = self._readme_table()
        shown = float(rows["smoke"].group("value")) / 60.0
        assert honest == pytest.approx(shown, abs=self.README_ROWS["smoke"][2])
        assert mutant != pytest.approx(shown,
                                       abs=self.README_ROWS["smoke"][2])

    @pytest.mark.parametrize("profile", sorted(README_ROWS))
    def test_every_readme_ingest_figure_matches_its_basis(self, profile):
        """README's front-page table, row by row.

        It had no pin at all until T5b, and that is how three of its six rows
        came to be wrong at once: `main` 33% low because this task added an M
        value and moved every other published figure but not this one,
        `quick` 2.5x low because docs/07's own † footnote corrected it and
        README never followed, and `mariadb-blog` 3.4x high from a figure no
        method produces. A reader uses this table to decide whether to start
        a run that takes days.
        """
        _readme, rows = self._readme_table()
        basis, _passes, tolerance = self.README_ROWS[profile]
        match = rows.get(profile)
        assert match, f"README has no `{profile}` row"
        shown = float(match.group("value"))
        if match.group("unit") == "min":
            shown /= 60.0
        expected = self._expected_readme_hours(profile)
        assert shown == pytest.approx(expected, abs=tolerance), (
            f"README says ~{match.group('value')} {match.group('unit')} for "
            f"`{profile}`; the {basis} basis gives {expected:.2f} h. A "
            f"profile changed and this row did not move with it.")

    def test_every_readme_row_is_pinned_or_named(self):
        """A new profile row cannot arrive unpinned.

        The failure this table has had is not a wrong number -- it is a
        number nothing checked. So the set of rows is asserted, not just the
        ones that happen to be listed.
        """
        _readme, rows = self._readme_table()
        assert set(rows) == set(self.README_ROWS) | self.README_UNPINNED, (
            f"README's profile table lists {sorted(rows)}; this test pins "
            f"{sorted(self.README_ROWS)} and names {sorted(self.README_UNPINNED)} "
            f"as unpinnable. Add the new row to one of the two.")
        # The exclusion is real, not a convenience: the only unpinned row is
        # unpinned because no basis can produce a figure for it.
        for profile in self.README_UNPINNED:
            datasets = load_profile(profile).get("datasets", [])
            assert any(est_mod._DATASET_ROWS.get(d) is None
                       for d in datasets), (
                f"`{profile}` is in README_UNPINNED but every dataset it "
                f"declares has a row count, so it can be pinned like the "
                f"rest.")

    def test_the_caption_still_describes_the_basis_the_pin_uses(self):
        """R7: the table's caption is what tells a reader what the numbers
        mean, and the pin is only honest if the two agree. Both the flat-rate
        basis and the two rows that depart from it are named in the prose."""
        readme, _rows = self._readme_table()
        assert "measured ingest only, the original three engines" in readme
        # The two departures the pin encodes.
        assert "ops.subset_rows" in readme
        assert "not derived from the per-engine rates above" in readme

    def test_the_stale_rows_stay_marked(self):
        """The first table's `quick` and `mariadb-blog` rows match no method,
        and a later editor dropping the markers would leave two misleading
        figures looking endorsed."""
        path = os.path.join(VB_ROOT, "docs", "07-planning-a-run.md")
        with open(path) as fh:
            doc = fh.read()
        quick = re.search(r"^\| `quick` \|.*$", doc, re.M).group(0)
        blog = re.search(r"^\| `mariadb-blog` \|.*$", doc, re.M).group(0)
        assert quick.rstrip().endswith("† |"), quick
        assert blog.rstrip().endswith("‡ |"), blog
        assert "14 h" not in quick, (
            "the hand figure for an M grid the profile no longer has")
        # Scoped to the footnotes themselves: whole-document membership would
        # have tolerated the figures appearing anywhere else.
        dagger = re.search(r"^† \*\*Corrected\.\*\*(?P<body>.*?)\n\n", doc,
                           re.M | re.S)
        double = re.search(r"^‡ \*\*Measured, but(?P<body>.*?)\n\n", doc,
                           re.M | re.S)
        assert dagger and double, "a footnote heading was removed or reworded"
        assert "~36 h" in dagger.group("body")
        assert "~21 h" in double.group("body")


class TestTheWebUiAndTheCliCannotDisagree:
    """The docstring's reason for returning the breakdown rather than printing
    it. /api/estimate serves this function's return value verbatim, and two
    static front-ends read fields off it by name."""

    def _api(self, body):
        from webui import control as control_mod
        status, payload = control_mod.estimate(None, None, None, body)
        assert status == 200
        return payload

    def test_the_endpoint_returns_the_function_unchanged(self):
        payload = self._api({"profile": "main", "resource_pass": "normalized"})
        # A request naming no engines previews what `cmd_run` would run:
        # the default-selected ones, without the opt-in VillageSQL.
        assert payload == est("main", engines=cli_mod.DEFAULT_ENGINES)
        assert "villagesql" not in payload["query_per_engine_hours"]

    def test_a_profile_that_restricts_the_run_restricts_the_preview(self):
        """The disagreement this class is named for, in the one place it could
        still happen. `control.js` opens with `plan.engines: []`, so selecting
        the percona profile and pressing nothing sent no engine list -- and the
        endpoint fell back to the registry while `cmd_run` falls back to the
        profile's `engines: [percona]`. The preview would have quoted a
        seven-engine cost for a job that measures one."""
        payload = self._api({"profile": "percona",
                             "resource_pass": "normalized"})
        named = (set(payload["per_engine_hours"])
                 | set(payload["engines_without_ingest_rate"]))
        assert named == {"percona"}, named
        assert payload == est("percona", engines=["percona"])

    def test_an_explicit_selection_still_overrides_the_profile(self):
        """The request wins over the profile's list here, as the flag does over
        it in `cmd_run`. The endpoint deliberately does NOT reproduce
        `cmd_run`'s refusal of an engine the profile excludes: a preview is not
        a run, and the launch it precedes is where that combination fails,
        loudly and before anything starts."""
        payload = self._api({"profile": "percona", "engines": ["mariadb"],
                             "resource_pass": "normalized"})
        assert list(payload["per_engine_hours"]) == ["mariadb"]

    def test_the_old_import_path_still_resolves(self):
        """`webui.control.estimate` and `test_config.py`'s
        `TestTheMSweepProfileIsolatesGraphDegree::
        test_the_estimate_covers_every_engine` import these from
        orchestrator.cli, and the move out to orchestrator/estimate.py must not
        be visible to either.

        BY NAME BECAUSE THE LINES WERE WRONG: this read `webui/control.py:268`
        and `tests/test_config.py:4171`. The first was right; the second was
        297 lines out and pointed at a `_FakeConn` lambda, identically on
        `master` at ffcb3a7, so it had never been re-derived since the split.
        The real import is the one named above."""
        from orchestrator import cli
        assert cli.estimate_load_hours is est_mod.estimate_load_hours
        assert cli.print_load_estimate is est_mod.print_load_estimate
        assert cli._INGEST_ROWS_PER_S is est_mod._INGEST_ROWS_PER_S

    @pytest.mark.parametrize("script", ["control.js", "setup.js"])
    def test_every_field_the_front_end_reads_exists(self, script):
        """A renamed or dropped key shows up here rather than as `undefined` in
        a browser. setup.js is included although this change does not own it:
        it renders total_hours as 'estimated ingest', which is exactly the
        field a redefinition would have broken."""
        path = os.path.join(VB_ROOT, "webui", "static", script)
        with open(path) as fh:
            source = fh.read()
        read = set(re.findall(r"\best\.([A-Za-z_][A-Za-z0-9_]*)", source))
        assert read, f"no est.<field> reads found in {script}"
        payload = self._api({"profile": "main", "resource_pass": "normalized"})
        missing = sorted(read - set(payload))
        assert missing == [], f"{script} reads fields the estimate does not return"

    def test_the_front_end_still_labels_total_hours_as_ingest(self):
        """If total_hours ever stops being ingest, these two lines have to
        change in the same commit. The test is here so they cannot not."""
        for script in ("control.js", "setup.js"):
            with open(os.path.join(VB_ROOT, "webui", "static", script)) as fh:
                source = fh.read()
            assert "estimated ingest ~${est.total_hours.toFixed(1)} h" in source

    def test_every_per_term_field_the_front_end_reads_exists(self):
        """`est.<field>` was pinned and `t.<field>` was not, so a renamed term
        key would still have reached a browser as `undefined`."""
        path = os.path.join(VB_ROOT, "webui", "static", "control.js")
        with open(path) as fh:
            source = fh.read()
        read = set(re.findall(r"\bt\.([A-Za-z_][A-Za-z0-9_]*)", source))
        assert read, "no t.<field> reads found in control.js"
        row = est("main")["query_terms"][0]
        assert sorted(read - set(row)) == []

    def test_the_front_end_shows_the_worst_case_total_as_well(self):
        with open(os.path.join(VB_ROOT, "webui", "static", "control.js")) as fh:
            source = fh.read()
        assert "worst_case_total_hours" in source
        assert "query_terms" in source
