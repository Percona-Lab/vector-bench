"""E19 -- §3's asymmetry rows must be about the engines that ran.

THE DEFECT
----------
`report/render.py:_known_asymmetries` says it is "conditional on what actually
ran", and two of its rows are not.

* The `ef_construction` row computed its subject from the literal set
  `{"pgvector", "valkey"}`. `percona` is not in it, so a run whose engine set
  is Percona Server alone intersected to nothing, fell through to the
  single-engine arm and published

      `ef_construction` is exposed only by pgvector

  in a report about the one engine in this study that has BOTH a build-time
  and a query-time effort knob. `config/engines/percona.yml` declares
  `capabilities.ef_construction_tunable: true`, `PerconaDriver.capabilities`
  returns it, and `CLAUDE.md`'s invariants bullet names pgvector, Valkey
  **and** Percona Server -- so the report contradicted the repository, the
  engine config and the driver at once, and the sentence was false in the one
  report where the knob was the subject of the investigation.

* The next row, "pgvector bulk-builds after load; MHNSW and VIDX build
  incrementally", was unconditional. In a Percona-only report it named three
  engines, none of which ran, and said nothing about the one that did --
  whose `build_mode` is the only within-engine bulk/incremental comparison in
  the MySQL family (`docs/09-adding-percona-server.md` §2, "Separable bulk
  build: **yes**").

`docs/05-methodology.md` §4's static table carried the same claim and a second
one beside it: its consequence column said the build-quality sweep "can only
vary `M` for the MySQL-family engines", which Percona Server -- a MySQL-family
engine with the knob -- makes false rather than merely incomplete.

WHAT IS PINNED HERE
-------------------
* `TestAPerconaRunIsNotToldTheKnobBelongsToPgvector` -- the headline case.
  The two-engine set passed before and after the fix, so the only test that
  can fail on the defect is one whose engine set contains Percona.
* `TestEveryEngineThatExposesTheKnobIsNamed` -- the general shape: the row is
  a list derived from the records, not a sentence with a length.
* `TestARunWithNoTunableEngineSaysSo` -- the arm for a run in which nothing
  exposes the knob, which the old two-branch code could not say.
* `TestTheFallbackIsPinnedInBothDirections` -- the constant, and the two
  mutants that nothing else in the repository can see.
* `TestTheBuildModeRowNamesTheEnginesThatBuilt` -- the second row, including
  the asynchronous case, which `separable_build` alone cannot tell from a
  bulk build.
* `TestTheMethodologyTableSaysTheSameThing` -- the static table in
  `docs/05-methodology.md` §4, and the one cross-check that can hold it to
  the generated list.

Nothing here reads `results/`: every summary below is built from record dicts
shaped the way `harness/workloads/build.py` writes them.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report import generate, render  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODOLOGY_DOC = os.path.join(VB_ROOT, "docs", "05-methodology.md")
ENGINES_DIR = os.path.join(VB_ROOT, "config", "engines")

#: "ops-only", for the separators in this class. Hyphen, the Unicode dashes,
#: a plain space, the fullwidth and small forms, and the invisible characters
#: a paste from a word processor leaves behind -- soft hyphen, zero-width
#: space and joiners, byte-order mark -- in any mix and any case. Two of my
#: own attempts got past an earlier version by writing "are ops only anyway"
#: with a space and "are ops\u2013only anyway" with an en dash, and the
#: reviewer then got past that with a soft hyphen and a zero-width space.
#:
#: The class is enumerable, which is why it is widened rather than disclosed:
#: a separator is one of a finite set of characters, unlike the semantic
#: walk-arounds the guard below discloses instead. It still carries NO
#: claim of closure -- a separator outside this set reads as the term to a
#: human and not to this pattern, and the docstring below says so.
OPS_ONLY = re.compile(
    r"ops[\s\-\u00ad\u2010-\u2015\u2043\u2212\u200b-\u200d"
    r"\ufeff\ufe63\uff0d]*only", re.I)

#: The knob's name as §3 and §4 both spell it, and the needle every row
#: lookup below uses. Backticked, because `ef_construction` unbackticked also
#: appears in prose about the sweep.
EF = "`ef_construction`"


def _records(spec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`ingest` + `index_build` pairs, in the shape `build.py` writes them.

    Per engine, `spec` takes `tunable` (what the driver's `capabilities()`
    reported, omitted entirely to model a record written before
    `_driver_capabilities` existed), `separable` (one `index_build` record per
    value, which is how an engine swept over both build modes appears) and
    `async_build`.

    `Record.ef_construction` is populated on every synthetic record here, and
    that models a tree state the orchestrator cannot produce: `ops_pass.py`
    appends `--ef-construction` only inside `if engine == "pgvector"`, and
    across every reachable archive a non-null `ef_construction` appears on
    pgvector's `index_build` records (182 of 182) and on no other engine's.
    It is kept deliberately, and for a different reason from the one an
    earlier revision of this file gave.

    The earlier reason was wrong and is recorded here because it inverted the
    failure: it said a `tunable` derived from `ef_construction is not None`
    would over-name MariaDB. Against a real run it does the opposite -- the
    field is null on every Percona record, so that discriminator yields an
    empty set and the row reads "exposed by none of the engines in this run",
    denying Percona the knob, which is this unit's defect arriving by another
    route. `TestAPerconaRunIsNotToldTheKnobBelongsToPgvector` is what refutes
    it against reality.

    Populating the field anyway closes the other direction: it is what makes
    `TestARunWithNoTunableEngineSaysSo` refuse that discriminator even in the
    hypothetical where `ops_pass.py` is widened to pass the flag for every
    engine, which is a one-line change to a gate nothing here guards.
    """
    out: List[Dict[str, Any]] = []
    for engine, how in sorted(spec.items()):
        for separable in (how.get("separable") or (False,)):
            extra: Dict[str, Any] = {}
            if "tunable" in how:
                extra["ef_construction_tunable"] = how["tunable"]
            if "tunable_raw" in how:
                # Whatever a hand-edited or round-tripped file can put there,
                # which is not restricted to a boolean.
                extra["ef_construction_tunable"] = how["tunable_raw"]
            if how.get("async_build"):
                extra["async_index_build"] = True
            if how.get("separable") is not None:
                extra["separable_build"] = separable
            base = {
                "engine": engine, "dataset": "d", "resource_pass": "tuned",
                "m": 16, "storage_engine": "heap", "ef_construction": 200,
                "build_mode": ("async" if how.get("async_build")
                               else "post" if separable else "incremental"),
            }
            out.append({**base, "phase": "ingest",
                        "ingest_rows_per_s": 900.0})
            out.append({**base, "phase": "index_build", "build_wall_s": 300.0,
                        "index_bytes": 7 * 1024 ** 3, "extra": extra})
    return out


def _text(spec: Dict[str, Dict[str, Any]]) -> str:
    return render._known_asymmetries(generate.summarize(_records(spec), {}))


def _table_rows(text: str) -> List[str]:
    """The body rows of the one Markdown table in the section."""
    return [line for line in text.splitlines()
            if line.startswith("|") and not re.match(r"^\|\s*-{3}", line)
            and not line.startswith("| Asymmetry")]


def _row(text: str, needle: str) -> str:
    """The single row carrying `needle`, both cells of it.

    Both cells, because a table row is both cells: the subject of the claim
    is in the first and what it costs these results is in the second, and a
    guard that reads only one of them can be satisfied by a row that
    contradicts itself across the delimiter.
    """
    found = [line for line in _table_rows(text) if needle in line]
    assert len(found) == 1, (
        f"{needle!r} matched {len(found)} rows, expected 1:\n"
        + "\n".join(found))
    return found[0]


def _exposed_by(text: str) -> str:
    """The clause naming the engines the row says DO expose the knob.

    Scoped, and the scope is load-bearing. "`Percona` is somewhere in the
    row" is satisfied by a row reading "can be tuned for pgvector and Valkey,
    and not for Percona Server" -- which is the false claim this unit exists
    to withdraw, and an unscoped assertion accepts it. Both of the mutants
    below were measured against the unscoped form first and one of them
    survived it.
    """
    row = _row(text, EF)
    clause = re.search(r"exposed (?:only )?by ([^|.]*)", row)
    assert clause, f"the row makes no 'exposed by' claim at all:\n{row}"
    return clause.group(1)


def _rows_about_build_mode(text: str) -> str:
    """Every row about how the index came to exist, joined.

    Two rows rather than one: an engine that did both modes earns its own
    row, so an assertion about "the bulk/incremental row" has to look at the
    pair or it passes by reading the wrong half.
    """
    return "\n".join(line for line in _table_rows(text)
                     if "bulk" in line.lower() or "incrementally" in line)


def _methodology_section_4() -> str:
    text = open(METHODOLOGY_DOC).read()
    section = text.split("## 4. Known asymmetries")[1].split("## 5.")[0]
    assert section.strip(), "§4 is empty"
    return section


def _configured_tunable_engines() -> List[str]:
    """Engines whose config declares the knob, straight from `config/engines`.

    `KNOWN_ENGINES` is whatever that directory holds and there is no list in
    Python to keep in step with it (`CLAUDE.md`), so the set of engines §4 has
    to name is read from the same place the drivers' own tests read it.
    """
    found = []
    for name in sorted(os.listdir(ENGINES_DIR)):
        if not name.endswith(".yml"):
            continue
        cfg = yaml.safe_load(open(os.path.join(ENGINES_DIR, name))) or {}
        if (cfg.get("capabilities") or {}).get("ef_construction_tunable"):
            found.append(os.path.splitext(name)[0])
    assert found, "no engine config declares ef_construction_tunable"
    return found


#: How §4 names, in prose, each engine that exposes the knob. Three entries
#: and a completeness assertion rather than a derivation: the document writes
#: "pgvector" where the chart legend writes "PostgreSQL (pgvector)", so there
#: is nothing to derive it from -- and the assertion is what makes a fourth
#: tunable engine fail this file instead of being silently unchecked.
DOC_NAME = {"pgvector": "pgvector", "valkey": "Valkey",
            "percona": "Percona", "myvector": "MyVector",
            "villagesql": "VillageSQL"}


class TestAPerconaRunIsNotToldTheKnobBelongsToPgvector:
    """The run the old literal got wrong, and the only shape that can fail.

    A Percona-only run at the current pin: one `index_build` record,
    incremental, `ef_construction_tunable: True` -- which is exactly what
    `percona-20261021-134142`'s ops record carries.
    """

    SPEC = {"percona": {"tunable": True, "separable": [False]}}

    def test_the_knob_is_not_attributed_to_an_engine_that_did_not_run(self):
        assert "only by pgvector" not in _text(self.SPEC)

    def test_the_row_names_percona(self):
        assert "Percona" in _exposed_by(_text(self.SPEC))

    def test_the_row_does_not_name_pgvector_at_all(self):
        """Not merely "not alone": pgvector is absent from the run, so the
        row has nothing to say about it in either cell."""
        assert "pgvector" not in _row(_text(self.SPEC), EF)

    def test_it_says_percona_has_both_knobs(self):
        """`CLAUDE.md`'s invariant: the build-time knob is only half of what
        makes this engine the interesting one, and a reader of a report about
        `ef_construction` has to be told the query-time knob exists too."""
        row = _row(_text(self.SPEC), EF)
        assert "build-time" in row and "query-time" in row

    def test_the_pin_policy_survives(self):
        """The rule the false sentence was justifying is unchanged: the knob
        is pinned to its default in the normalized pass."""
        row = _row(_text(self.SPEC), EF)
        assert "normalized" in row and "pinned" in row


class TestEveryEngineThatExposesTheKnobIsNamed:

    SPEC = {
        "percona": {"tunable": True, "separable": [False]},
        "pgvector": {"tunable": True, "separable": [True]},
        "valkey": {"tunable": True, "separable": [True]},
    }

    def test_all_three_are_named_in_the_row(self):
        exposed = _exposed_by(_text(self.SPEC)).lower()
        for engine in ("percona", "pgvector", "valkey"):
            assert engine in exposed, engine

    def test_it_no_longer_claims_one_engine_has_it_alone(self):
        assert "only by" not in _row(_text(self.SPEC), EF)

    def test_a_two_engine_run_still_reads_as_it_did(self):
        """The set that was right before this change stays right after it:
        the fix is a widening, not a rewrite."""
        spec = {k: v for k, v in self.SPEC.items() if k != "percona"}
        exposed = _exposed_by(_text(spec))
        assert "pgvector" in exposed and "Valkey" in exposed
        assert "percona" not in exposed.lower()

    def test_a_single_engine_run_says_only(self):
        """One engine with the knob is still an "only", and the sentence is
        parameterised by the engine found rather than by pgvector -- which is
        what keeps `tests/test_config.py`'s pgvector-only assertion true."""
        spec = {"pgvector": {"tunable": True, "separable": [True]}}
        assert "only by pgvector" in _row(_text(spec), EF)

    def test_an_engine_whose_records_predate_the_capability_key_is_placed(
            self):
        """A record written before `_driver_capabilities` existed states
        nothing, and the three engines that expose the knob did so before the
        key was recorded. Reading that silence as "no" would withdraw a true
        claim from every archived run."""
        spec = {"pgvector": {"separable": [True]},
                "valkey": {"separable": [True]}}
        exposed = _exposed_by(_text(spec))
        assert "pgvector" in exposed and "Valkey" in exposed

    def test_a_recorded_false_beats_the_fallback(self):
        """The records are the authority, not the fallback: an engine whose
        driver reports False is not tunable however it is spelled here.

        Naming it as an engine that LACKS the knob is correct and is what the
        row then does, so the assertion is on the claim and not on the token.
        """
        exposed = _exposed_by(_text({"pgvector": {"tunable": False,
                                                  "separable": [True]}}))
        assert "none" in exposed.lower()
        assert "pgvector" not in exposed


class TestARunWithNoTunableEngineSaysSo:
    """MariaDB and AliSQL: the arm the old two-branch code could not say.

    Its `else` covered both "one engine has it" and "none of them does", so
    this run was told the knob was "exposed only by pgvector" -- an engine
    that had not run.
    """

    SPEC = {"mariadb": {"tunable": False, "separable": [False]},
            "alisql": {"tunable": False, "separable": [False]}}

    def test_the_row_says_no_engine_here_exposes_it(self):
        assert "none" in _exposed_by(_text(self.SPEC)).lower()

    def test_it_does_not_name_pgvector_as_present(self):
        assert "pgvector" not in _exposed_by(_text(self.SPEC))
        assert "pgvector" not in _row(_text(self.SPEC), EF)
        assert "only by pgvector" not in _text(self.SPEC)

    def test_the_consequence_is_stated_without_a_second_knob(self):
        """A row about an absence still has to say what the absence costs --
        and must say it without the obvious sentence, "so the sweep varies
        `M` alone", which is false for Percona Search.

        That engine gets `m_values[:1]` from `orchestrator/ann_pass.py` under
        its own comment, "M is carried, not applied: mongot exposes no graph
        degree", so a report narrowed to it would have carried this row three
        rows above "M is pinned at one value for every other engine and
        cannot be set here at all" and contradicted it. The consequence is
        therefore about the pass, which is true of every engine.
        """
        row = _row(_text(self.SPEC), EF)
        assert "normalized" in row and "no knob to pin" in row
        assert "`M`" not in row

    @pytest.mark.parametrize("spec", [
        {"percona": {"tunable": True, "separable": [False]}},
        {"pgvector": {"tunable": True, "separable": [True]},
         "valkey": {"tunable": True, "separable": [True]}},
        {"mariadb": {"tunable": False, "separable": [False]}},
    ])
    def test_no_arm_claims_what_the_m_sweep_varies(self, spec):
        """All three arms, because the row is about `ef_construction` and a
        claim about a second knob in it is a claim nothing checked."""
        assert "`M`" not in _row(_text(spec), EF)

    def test_a_report_narrowed_to_percona_search_does_not_contradict_itself(
            self):
        """The reproduction: `--profile full --engines mongodb` is an
        ordinary invocation that `orchestrator/cli.py` permits, and in that
        report the two rows sit in one table."""
        text = _text({"mongodb": {"separable": [True], "async_build": True}})
        assert "cannot be set here at all" in text
        assert not re.search(r"(varies|varying|sweeps|swept over)\s+`M`",
                             text), "a row claims the sweep varies M"


class TestTheFallbackIsPinnedInBothDirections:
    """`EF_CONSTRUCTION_TUNABLE_WHEN_UNRECORDED`, and why one assertion is
    not enough for it.

    Absence-as-DENIAL was already guarded, by
    `test_an_engine_whose_records_predate_the_capability_key_is_placed` here
    and by `tests/test_config.py`'s pgvector-only assertion. The other two
    directions were guarded by nothing in the repository, measured: with
    `else True` in place of the set, or with `mariadb` and `alisql` added to
    it, this file and the whole of `tests/test_config.py` stayed green -- 677
    tests blind to a one-token edit.

    What that edit costs is not theoretical. 132 archived `index_build`
    records carry no `ef_construction_tunable` at all, 94 of them from
    engines that do not have the knob (mariadb 44, alisql 34, mariadb123 12,
    mongodb 4), so under either mutant a report regenerated over a legacy
    mariadb-only directory reads "exposed only by MariaDB MHNSW" -- the
    original defect with its sign flipped.
    """

    def test_the_set_is_exactly_what_the_engine_configs_declare(self):
        """Closes the widening direction. A set that may be extended by a
        one-word edit is the literal this unit removed, wearing a name."""
        assert (set(render.EF_CONSTRUCTION_TUNABLE_WHEN_UNRECORDED)
                == set(_configured_tunable_engines()))

    def test_a_silent_non_tunable_engine_is_not_named(self):
        """Closes the affirmation direction.

        The spec records NO `ef_construction_tunable` at all, which is the
        only shape that reaches the fallback: every other non-tunable spec in
        this file records `False`, a recorded value beats the fallback, and
        that is precisely why the hole was open.
        """
        exposed = _exposed_by(_text({"mariadb": {"separable": [False]}}))
        assert "none" in exposed.lower()
        assert "MariaDB" not in exposed

    def test_a_string_is_not_an_answer_in_either_direction(self):
        """`bool("false")` is `True`, so reading the key without a type check
        hands the knob to an engine that does not have it off a string -- the
        published sentence being "exposed only by MariaDB MHNSW". A value
        that is not a boolean is not an answer: the record is silent and the
        fallback decides, which for this engine is a denial."""
        exposed = _exposed_by(
            _text({"mariadb": {"tunable_raw": "false", "separable": [False]}}))
        assert "none" in exposed.lower()
        assert "MariaDB" not in exposed

    def test_a_recorded_null_falls_to_the_fallback_rather_than_denying(self):
        """This repo records `null` to mean "documented, not observed" --
        `config/engines/valkey.yml` does it for `hybrid_filter_planner`, with
        a comment about a claim that reached the report for weeks. Read as a
        `False` it would withdraw the knob from an engine that has it
        WITHOUT consulting the fallback, which is the whole point of keeping
        silence a third state."""
        exposed = _exposed_by(
            _text({"pgvector": {"tunable_raw": None, "separable": [True]}}))
        assert "pgvector" in exposed

    def test_a_real_boolean_still_beats_the_fallback(self):
        """The type check must not quietly turn every record silent."""
        exposed = _exposed_by(
            _text({"pgvector": {"tunable": False, "separable": [True]}}))
        assert "pgvector" not in exposed
        assert "none" in exposed.lower()

    def test_a_legacy_single_engine_directory_is_not_handed_the_knob(self):
        """The regeneration above, as the sentence a reader would see."""
        text = _text({"mariadb": {"separable": [False]}})
        assert "exposed only by" not in text

    def test_silence_from_several_non_tunable_engines_is_still_silence(self):
        spec = {"mariadb": {"separable": [False]},
                "alisql": {"separable": [False]},
                "mongodb": {"separable": [True], "async_build": True}}
        exposed = _exposed_by(_text(spec))
        assert "none" in exposed.lower()
        for engine in ("MariaDB", "AliSQL", "Percona Search"):
            assert engine not in exposed, engine


class TestTheBuildModeRowNamesTheEnginesThatBuilt:

    def test_it_names_percona_when_percona_bulk_built(self):
        text = _text({"percona": {"tunable": True, "separable": [True]}})
        assert "Percona" in _rows_about_build_mode(text)

    def test_it_does_not_name_percona_when_only_mariadb_ran(self):
        text = _text({"mariadb": {"tunable": False, "separable": [False]}})
        rows = _rows_about_build_mode(text)
        assert rows, "nothing was said about how MariaDB's index was built"
        assert "Percona" not in rows
        assert "pgvector" not in rows

    def test_both_sides_are_named_from_the_records(self):
        text = _text({"pgvector": {"tunable": True, "separable": [True]},
                      "mariadb": {"tunable": False, "separable": [False]}})
        rows = _rows_about_build_mode(text)
        assert "pgvector" in rows and "MariaDB" in rows

    def test_an_engine_that_did_both_modes_gets_its_own_row(self):
        """Percona swept over both modes is a comparison within one engine,
        and the two halves are not the same operation: its first *ADD VECTOR
        KEY* rebuilds the base table where pgvector's *CREATE INDEX* does
        not, so one "the bulk build is faster" figure hides a table rebuild
        inside one engine's number."""
        text = _text({"percona": {"tunable": True,
                                  "separable": [True, False]}})
        row = _row(text, "either")
        assert "within one engine" in row
        assert "ADD VECTOR KEY" in row
        assert "rebuild" in row.lower()

    def test_one_mode_earns_no_within_engine_row(self):
        """At this pin `capabilities.bulk_vector_index_build` is false and
        `config/profiles/percona.yml` runs incremental alone, so a report of
        such a run must not claim the engine did either: the row is about
        what ran."""
        text = _text({"percona": {"tunable": True, "separable": [False]}})
        assert "does either" not in text

    def test_the_consequence_does_not_claim_two_modes_when_one_ran(self):
        """The subject was derived and master's consequence was left alone,
        so `percona-20261021-134142` -- one engine, one mode -- rendered
        "reported separately for both modes" beside a subject naming one.
        A row whose two cells disagree about how many modes ran is worse
        than the unconditional row it replaced, because the derivation makes
        it look checked."""
        one = _rows_about_build_mode(
            _text({"percona": {"tunable": True, "separable": [False]}}))
        assert "both modes" not in one
        assert "the operation that ran" in one
        both = _rows_about_build_mode(
            _text({"pgvector": {"tunable": True, "separable": [True]},
                   "mariadb": {"tunable": False, "separable": [False]}}))
        assert "both modes" in both

    def test_an_async_build_is_not_filed_as_a_bulk_build(self):
        """mongot's `separable_build` is True and its build is neither bulk
        nor incremental -- `_index_build_kind` already splits the four kinds
        for the build table, and this row has to split them the same way or
        it contradicts the asynchronous row below it."""
        text = _text({"mongodb": {"separable": [True],
                                  "async_build": True}})
        assert not _rows_about_build_mode(text)
        assert "asynchronously" in text

    def test_an_unrecorded_build_is_not_guessed(self):
        """A record with no `separable_build` at all cannot be placed on
        either side -- the same three-way split `generate.build_kind` makes
        for the chart and the build table."""
        text = _text({"mariadb": {"tunable": False, "separable": None}})
        assert not _rows_about_build_mode(text)


class TestTheMethodologyTableSaysTheSameThing:
    """§4's static table, and how far a test can hold it to §3's generated one.

    The generated list is per run and the document's table is about the study,
    so the two cannot be asserted equal: a Percona-only report legitimately
    names one engine where the document names three. What a test can pin is
    the study-level claim they both make -- which engines expose the knob at
    all -- and that is the claim that went stale in both places at once.
    """

    def test_the_row_names_percona_and_valkey(self):
        row = [line for line in _methodology_section_4().splitlines()
               if EF in line]
        assert len(row) == 1, row
        assert "Percona" in row[0] and "Valkey" in row[0]

    def test_it_no_longer_says_only_m_can_vary_for_the_mysql_family(self):
        """The second false claim in the same cell. Percona Server is a
        MySQL-family engine with the knob, so this is wrong rather than
        incomplete."""
        doc = open(METHODOLOGY_DOC).read()
        assert "can only vary `M` for the MySQL-family engines" not in doc

    def test_the_cell_names_the_engine_the_sweep_does_not_reach(self):
        """§4's cell must name Percona Search. Unconditionally.

        Everything this cell can say about the build-quality sweep across the
        engine set has that engine as a counter-example:
        `orchestrator/ann_pass.py` gives it `m_values[:1]` under its own
        comment, "M is carried, not applied: mongot exposes no graph degree",
        and `report/render.py` says "Percona Search exposes no graph degree"
        in a row of its own. So the requirement is not conditional on how the
        cell is worded, and three successive conditional versions of this
        assertion were each walked past with the false claim restored and
        every test green:

        * a trigger on the verb adjacent to the knob missed "varies
          `ef_construction` only for the three that have it, and `M` for
          every engine";
        * a determiner-plus-"engine" universal missed "`M` for all seven
          engines", where a numeral sits between the determiner and the noun;
        * the same universal missed "`M` is applied right across the engine
          set", which has no adjacency at all;
        * and keying on the token `` `M` `` missed "the graph degree", which
          is this repository's own term for it -- the wording this very cell
          used until the round that introduced the guard.

        WHAT THIS CHECK DOES NOT COVER, stated because the last three
        docstrings here each claimed more than they delivered: it is a
        necessary condition on the cell, not a reading of the claim. A cell
        that names the engine and then asserts the sweep reaches it anyway --
        "for every engine, including Percona Search" -- satisfies this and is
        false. Nothing mechanical catches that; what catches it is a reviewer
        reading this cell beside `test_the_cells_claim_about_that_engine_is_
        executable` below, which pins the code the cell is about.

        That one sentence is the WHOLE residue, and the reason it can be is
        the shape of the check: a necessary condition has exactly one failure
        mode, being satisfied while the claim is false, so naming that mode
        names everything it misses.

        And it is not what does the real work here: that is the three
        positive assertions beside it, none of which depends on how the cell
        is worded -- `test_every_engine_whose_config_declares_the_knob_is_
        named` reads the capability out of `config/engines/*.yml`,
        `test_the_generated_row_names_the_same_engine_set` holds §3's
        derived row to the same set, and
        `test_the_cells_claim_about_that_engine_is_executable` pins the code
        the cell is about. This check only stops the cell from dropping the
        exception.

        Either of this repository's two names for the engine satisfies it --
        "Percona Search" or "mongot" -- because the requirement is that the
        exception be named, not that it be named one way, and refusing a cell
        that said "mongot" would be a guard firing on correct prose.

        The cost is that a true, narrowly-scoped sentence about `M` in this
        cell must also name that engine. That is accepted: the subject of
        the cell is a different knob, and the cell carried a false claim
        about this one in two of three review rounds.
        """
        row = [line for line in _methodology_section_4().splitlines()
               if EF in line][0]
        assert "Percona Search" in row or "mongot" in row, (
            "§4's cell does not name the one engine the build-quality sweep "
            "does not reach; every claim it makes about the sweep across the "
            "engine set has that engine as a counter-example")

    def test_2_7_does_not_call_the_cost_fields_ops_only(self):
        """§2.7's stated reason for reading build mode off the ops records
        was that `build_wall_s`, `ingest_wall_s` and `index_bytes` "are
        ops-only fields anyway". `index_bytes` is on every one of the 6,035
        archived `recall_qps` records, so that was false before A1's loader
        work and stays false after it -- and the real reason is the one the
        two sentences above it already give.

        Three things about the check, each of them a walk-around that
        happened:

        ONE PHRASE WAS NOT ENOUGH. The first version refused the exact string
        "are ops-only fields", and deleting the single word "fields" walked
        through it with the claim intact. So the rule is about the term.

        THE NEGATION MUST BE IN THE SAME CLAUSE AS THE TERM. A plain window
        accepted "are not recall fields; all three are ops-only", where the
        `not` inside the window negates something else and the ops-only claim
        is an unqualified affirmation. `[^.;]` in the pattern is what refuses
        that -- and the example's own semicolon is what it trips on.

        THE TERM IS MATCHED, NOT ITS TYPOGRAPHY. Two of my own attempts got
        past an earlier version of this check by writing "ops only" with a
        space and "ops-only" with an en dash -- one character each, claim
        intact, every test green. `OPS_ONLY` collapses the separator.

        THE WHOLE DOCUMENT IS SCANNED, not the one paragraph, because moving
        the sentence out of the paragraph was otherwise a walk-around on its
        own. SUBJECT-SCOPED to sentences naming `index_bytes`, which is the
        shape `tests/test_e1_percona_design.py` argues for at length: a ban
        that does not name its subject fires on a true sentence about
        something else, and "`build_wall_s` and `ingest_wall_s` are ops-only"
        is such a sentence -- 0 of 6,035 archived recall records carry
        either.

        WHAT IT COVERS, which is the only honest way to say what it does
        not. This fires only when both operands appear as matched -- the
        subject as the exact token `index_bytes`, the predicate as
        `OPS_ONLY`. A sentence that states either operand any other way is
        outside it: a pronoun or a prose name for the field ("the two wall
        clocks and the index size are ops-only"), a different spelling or
        escaping of either -- a Markdown-escaped underscore renders
        identically and `INDEX_BYTES` differs only in case -- or the claim
        made without the term. Separately, a negation in the right clause whose
        scope is not the predicate asserts what it appears to deny ("not just
        ops-only but ops-first"; "not merely" and "not only" the same way).
        Each needs a reader, and a reader is an acceptable terminus here --
        it is this file's own idiom, as `_PROSE_NAME`'s "NOT A GATE" and
        `generate.BUILD_KIND_UNRECORDED` are -- on the one condition that
        what is written down is true.

        That paragraph replaced a list of five enumerated escapes, and the
        replacement is the lesson of five review rounds rather than tidying.
        A way past this guard was found in every one of them, four of those
        past a list introduced as a class. Two sentences describing the check
        cannot go stale the same way, because THEY ARE THE IMPLEMENTATION --
        two matched operands and a clause window -- so there is nothing left
        to enumerate around. Anaphora and the absent term, which that list
        named as separate escapes, are both instances of one thing: an
        operand not matched as written. Neither operand's matching is
        widened to chase this; widening the predicate's was defensible
        because a separator is a finite set of characters, and doing the same
        to the subject would start the sequence again on a third surface.

        Whitespace-normalised, because the document is hard-wrapped and the
        sentence spans a line break.
        """
        doc = " ".join(open(METHODOLOGY_DOC).read().split())
        para = doc.split(
            "Percona Server is the fourth for a different reason")[1].split(
            "An empty list has two causes")[0]
        assert re.search(r"silent about `build_mode`|says nothing about "
                         r"`build_mode`", para), (
            "§2.7 no longer gives the reason that is true of either tree")
        for sentence in re.split(r"(?<=[.!?])\s+", doc):
            if "index_bytes" not in sentence or not OPS_ONLY.search(sentence):
                continue
            for mention in OPS_ONLY.finditer(sentence):
                clause = sentence[max(0, mention.start() - 20):mention.start()]
                assert re.search(
                    r"\b(not|never|no longer|nor)\b[^.;]*$", clause), (
                    "a sentence naming `index_bytes` calls it ops-only "
                    f"without negating it in the same clause: {sentence!r} "
                    "-- `index_bytes` is on every one of the 6,035 archived "
                    "recall records")

    def test_the_cells_claim_about_that_engine_is_executable(self):
        """`CLAUDE.md`: documentation asserting engine behaviour should be
        checkable. The cell's parenthetical is about one expression."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ann_pass.py")).read()
        assert "m_values[:1]" in source
        assert "exposes no graph degree" in source

    def test_every_engine_whose_config_declares_the_knob_is_named(self):
        """The ratchet. Both stale sentences were stale because an engine
        landed and the claim about the engine set was not revisited."""
        row = [line for line in _methodology_section_4().splitlines()
               if EF in line][0]
        for engine in _configured_tunable_engines():
            assert engine in DOC_NAME, (
                f"{engine} declares ef_construction_tunable and this file "
                f"does not know how §4 names it")
            assert DOC_NAME[engine] in row, engine

    def test_the_generated_row_names_the_same_engine_set(self):
        """The cross-check, in the one direction that is a claim about the
        study: a run that contains every engine the document names must
        generate a row naming every engine the document names."""
        spec = {engine: {"tunable": True, "separable": [True]}
                for engine in _configured_tunable_engines()}
        exposed = _exposed_by(_text(spec)).lower()
        for engine in _configured_tunable_engines():
            assert DOC_NAME[engine].lower() in exposed, engine

    def test_neither_side_claims_the_knob_for_mhnsw_or_vidx(self):
        """The half of the old sentence that was true, in both places."""
        doc_row = [line for line in _methodology_section_4().splitlines()
                   if EF in line][0]
        assert "MHNSW" in doc_row and "VIDX" in doc_row
        text = _text({"mariadb": {"tunable": False, "separable": [False]},
                      "alisql": {"tunable": False, "separable": [False]},
                      "percona": {"tunable": True, "separable": [False]}})
        generated, exposed = _row(text, EF), _exposed_by(text)
        assert "MariaDB" in generated and "AliSQL" in generated
        assert "MariaDB" not in exposed and "AliSQL" not in exposed
