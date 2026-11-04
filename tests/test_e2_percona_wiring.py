"""Guards for wiring Percona Server in as the seventh engine (docs/09 §4).

`tests/test_e1_percona_design.py` guards the DESIGN -- the document, the
long-lead artefacts it wrote, and every `file:line` it cites. This file guards
the WIRING: the dialect fields, the driver, the registry entries, the two
scripts, the render branch and the estimator change that make the engine
reachable from a command line.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
Nothing here establishes anything about Percona Server. No image has been built
and no server has been started; every claim about the engine's behaviour is
sourced to a file:line on commit 56124b03fdc5954e2401ab4d2c7bf7cb3a767158 in
docs/09 and is settled by reading that tree. What these establish is that the
statements this repo would SEND are the ones the document says it sends, and
that the seven-engine wiring agrees with itself across the files that have to.

Two shapes recur below and both are deliberate:

* **Driver behaviour is exercised, not grepped.** The drivers are instantiated
  against a recording fake cursor and the SQL they emit is read back out of it.
  A substring test for `"BIGINT UNSIGNED"` passes against a file that also
  still says `id INT PRIMARY KEY` somewhere, which is exactly the DDL bug it
  would be looking for.
* **Every assertion about Percona is paired with the same assertion about
  MariaDB or AliSQL.** The dialect fields all default to what those two
  already did, so "Percona emits X" is only half the claim; "and MariaDB still
  emits what it emitted" is the half that says no existing engine moved.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import (ConnectionSpec, EngineDriver,  # noqa: E402
                                  IndexSpec)
from harness.drivers.mysql_family import (ALISQL_DIALECT,  # noqa: E402
                                          MARIADB_DIALECT, PERCONA_DIALECT,
                                          AliSQLDriver, MariaDBDriver,
                                          MySQLDialect, PerconaDriver)
from harness.drivers.postgres import _driver_classes, _driver_table  # noqa: E402

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED_SHA = "56124b03fdc5954e2401ab4d2c7bf7cb3a767158"


# ---------------------------------------------------------------------------
# Prose guards: a vocabulary, not a sentence
# ---------------------------------------------------------------------------
#
# Every prose claim below is pinned as a REGEX over whitespace-normalised text,
# fixing subject, mechanism and polarity, with the rewordings it must accept
# and the attacks it must reject written beside it and asserted. This is the
# shape `tests/test_e3_rollout_doc.py` uses, and it is here because the first
# version of these guards matched exact phrases and **fired on correct prose**:
# `"all seven engines"` rejected "all seven of the engines", "every one of the
# seven engines" and "all 7 engines"; `"six of those seven"` rejected "covers
# six engines of the seven". A guard that fires on a correct rewrite teaches
# the next author to delete it, which is how the vacuity shapes this repo keeps
# finding got here in the first place.
#
# Normalising whitespace is necessary and not sufficient: a reflow was the
# first failure mode found, an exact phrase is the second.

PROSE_CLAIMS = {
    "the engine count is seven": dict(
        pattern=r"\b(all|every one of|each of)\b[^.]{0,24}?"
                r"\b(seven|7)\b[^.]{0,24}?\bengines?\b",
        accepts=[
            "all seven engines, which is what that command measures",
            "all seven of the engines",
            "every one of the seven engines",
            "all 7 engines",
            "each of the seven engines is measured",
        ],
        rejects=[
            "all six engines, which is what that command measures",
            "all six of the engines",
            "every one of the six engines",
        ]),
    "the engine count is eight": dict(
        pattern=r"\b(all|every one of|each of)\b[^.]{0,24}?"
                r"\b(eight|8)\b[^.]{0,24}?\bengines?\b",
        accepts=[
            "all eight engines, which is what that command measures",
            "all eight of the engines",
            "all 8 engines",
        ],
        rejects=[
            "all seven engines, which is what that command measures",
            "every one of the seven engines",
        ]),
    "the engine count is nine": dict(
        pattern=r"\b(all|every one of|each of)\b[^.]{0,24}?"
                r"\b(nine|9)\b[^.]{0,24}?\bengines?\b",
        accepts=[
            "all nine engines, which is what that command measures",
            "all nine of the engines",
            "all 9 engines",
        ],
        rejects=[
            "all eight engines, which is what that command measures",
            "every one of the eight engines",
        ]),
    "the ingest figure covers seven of the nine": dict(
        pattern=r"\bingest\b[^.]{0,80}?\bseven\b[^.]{0,40}?"
                r"\b(measured rate|rate)\b",
        accepts=[
            "the ingest figure covers the seven with a measured rate",
            "the ingest total is over the seven engines that have a rate",
        ],
        rejects=[
            "the ingest figure covers all nine",
            "Both figures cover all nine engines.",
        ]),
    "the ingest figure covers seven of the eight": dict(
        pattern=r"\bingest\b[^.]{0,80}?\bseven\b[^.]{0,40}?"
                r"\b(measured rate|rate)\b",
        accepts=[
            "the ingest figure covers the seven with a measured rate",
            "the ingest total is over the seven engines that have a rate",
        ],
        rejects=[
            "the ingest figure covers all eight",
            "Both figures cover all eight engines.",
        ]),
    # "the ingest figure covers six of the seven" WAS HERE, and was deleted
    # rather than reworded: the claim it guarded has stopped being true.
    # Percona's M=16 incremental rate is in `_MEASURED_ROWS_PER_S` and
    # `_INGEST_ROWS_PER_S` now, so the published ingest figure covers all
    # seven engines and CLAUDE.md no longer splits its two numbers by scope.
    # What replaces it is a computation and not a vocabulary:
    # `tests/test_estimate.py`'s CLAUDE.md guard asserts the document's
    # ingest figure is 69.83 h above the six-engine one, so the old 726 h
    # cannot satisfy it. A vocabulary cannot do that -- it would accept any
    # sentence with the right words in it, including a stale one.
    "percona does either build mode": dict(
        pattern=r"Percona(\s+Server)?\b[^.]{0,80}?"
                r"\b(does either|either mode|both|can do both|does both)\b",
        accepts=[
            "Percona Server does either, so its two build modes are a "
            "comparison within one engine",
            "Percona Server can do both, so build mode is an axis within "
            "one engine",
            "uniquely, Percona Server does both",
            "Percona Server builds either mode on request",
        ],
        rejects=[
            "pgvector bulk-builds; MHNSW and VIDX build on every INSERT.",
            "Percona Server builds incrementally like MHNSW and VIDX.",
        ]),
    "the total covers only the engines with a rate": dict(
        pattern=r"\b(\d+) of (\d+) engine",
        accepts=[
            "the total above is 6 of 7 engine(s) and excludes it",
            "that total covers 6 of 7 engines",
        ],
        rejects=[
            "the total above excludes percona",
            "no measured ingest rate for percona.",
        ]),
    "the real cost is higher than the printed total": dict(
        pattern=r"\b(higher|greater|larger|more)\b[^.]{0,40}?"
                r"\bthan\b[^.]{0,40}?\b(number|figure|total)\b"
                r"|\b(number|figure|total)\b[^.]{0,40}?"
                r"\bunderstates?\b",
        accepts=[
            "the ingest cost of this plan is higher than the number above, "
            "by an amount nothing here knows",
            "the real ingest cost is greater than the figure above",
            "the total above understates this plan by an unknown amount",
        ],
        rejects=[
            "No measured ingest rate for percona.",
            "percona is excluded from the total.",
        ]),
    "a re-pin leaves the recall cells and the subtree must be moved aside": dict(
        # FOUR REQUIREMENTS, ORDER-INDEPENDENT. A paragraph that drops any one
        # of them stops being a warning: the TRIGGER (the source pin moved),
        # the HAZARD (the cells already on disk are reused rather than
        # re-measured, because `ann_fingerprint` is engine-invariant and holds
        # no source tag), and the REMEDY in two halves -- the ACT (move,
        # rename, delete, re-measure) and the OBJECT it acts on (the engine's
        # subtree of the ann results tree). The act and the object are
        # separate legs because "move the pin" and "the results tree is
        # shared" are each true on their own and neither is the instruction.
        #
        # THE REMEDY IS NOT `--force`, AND IT USED TO SAY IT WAS. That flag
        # does re-run points that already have results -- its help text says
        # exactly that -- but it is also the escape hatch on the free-disk
        # refusal (`_check_free_disk(...) and not args.force` in
        # `orchestrator/cli.py`), so passing it silently disables the one
        # refusal a core-dump campaign most needs. Moving the engine's subtree
        # aside buys the re-measurement without buying that, and
        # `_count_results` counts result files under directories named for the
        # engine, so moving them is what makes it return zero.
        #
        # A fresh `--run-id` is NOT a substitute and is not part of this
        # claim: `paths` in `orchestrator/cli.py` puts `ops_results` at the
        # run directory and `annb_results` at a sibling of it, so a new run
        # directory leaves every recall cell where it was. That is a separate
        # requirement, for the ops path, and it is stated beside this one.
        #
        # ANCHORED AT \A for the reason the entry below documents.
        #
        # BOTH HALVES ARE SENTENCE-LEVEL, AND THE FIRST VERSION WAS NOT. It
        # asked for four tokens anywhere in the block, and two of this file's
        # own mutants then SURVIVED: deleting the remedy sentence from
        # CLAUDE.md left "Re-measure" and "recall" elsewhere in the same
        # twenty-line bullet, and deleting the hazard sentence left
        # "ann-benchmarks caches by algorithm" -- the bullet's original
        # opening, which is a true statement about caching and not about this
        # pin. A block larger than the claim can satisfy a conjunction out of
        # sentences that have nothing to do with each other, which is the
        # same defect `_paragraphs` exists to stop, one level further in.
        #
        # So the trigger must meet the hazard INSIDE ONE SENTENCE -- it is
        # the pin moving that causes the reuse -- and the act must meet the
        # object inside one sentence, because "move the pin" and "the results
        # tree is shared" are each true alone and neither is the instruction.
        # `(?:(?!\.)[^.])*?` is what keeps a match inside a sentence; either
        # order is accepted, because English puts them both ways.
        pattern=r"\A"
                r"(?=.*\b(?:re-?pin(?:ned|ning)?|new pin"
                r"|pin (?:move|change)[sd]?|moves? the pin|moved the pin"
                r"|source\.tag)\b(?:(?!\.)[^.])*?"
                r"\b(?:reuse[sd]?|reusing|cached?|caches|stale"
                r"|skip(?:s|ped)?|already on disk|where it was"
                r"|left behind)\b"
                r"|.*\b(?:reuse[sd]?|reusing|cached?|caches|stale"
                r"|skip(?:s|ped)?|already on disk|where it was"
                r"|left behind)\b(?:(?!\.)[^.])*?"
                r"\b(?:re-?pin(?:ned|ning)?|new pin"
                r"|pin (?:move|change)[sd]?|moves? the pin|moved the pin"
                r"|source\.tag)\b)"
                r"(?=.*\b(?:mov(?:e|es|ed|ing)|aside|renam\w+|delet\w+"
                r"|re-?measur\w+|re-?comput\w+)\b(?:(?!\.)[^.])*?"
                r"\b(?:subtree|results tree|ann tree|result files"
                r"|recall cells?|curve)\b"
                r"|.*\b(?:subtree|results tree|ann tree|result files"
                r"|recall cells?|curve)\b(?:(?!\.)[^.])*?"
                r"\b(?:mov(?:e|es|ed|ing)|aside|renam\w+|delet\w+"
                r"|re-?measur\w+|re-?comput\w+)\b)",
        accepts=[
            "After a re-pin the recall cells on disk are reused, so move "
            "that engine's subtree of the ann results tree aside before the "
            "run.",
            "Because source.tag changed the cached result files are "
            "skipped; moving the engine's subtree aside makes the curve get "
            "measured again.",
            "A new pin leaves every recall cell where it was, so move the "
            "subtree aside and the curve is re-measured.",
        ],
        rejects=[
            # Each drops one leg.
            "After a re-pin the recall cells are reused. Nothing to do.",
            "Move the subtree aside before the run.",
            "The cached result files are skipped; move the subtree aside.",
            "A re-pin changes the engine_version on every record.",
            # The two shapes that survived the block-level version: the
            # vocabulary present, scattered over sentences that do not make
            # the claim between them.
            "A re-pin does not move the tree. ann-benchmarks caches by "
            "algorithm. Re-measure the engine if you want its real recall.",
            "ann-benchmarks caches by algorithm and index parameters only. "
            "Move that engine's subtree aside. The tree is keyed by the "
            "resource pass.",
        ]),
    "a sentence that tells the operator to pass --force": dict(
        # NOT A CLAIM TO BE MADE -- a shape to be ABSENT. The test below
        # asserts no paragraph at any of the three sites matches this, which
        # is the assertion that would have caught the error this entry exists
        # because of: for two rounds these documents told the operator to pass
        # `--force`, and the guard over them only ever grepped for the string,
        # so it could not tell an instruction from a warning. It passed on
        # prose saying the opposite of what it was guarding.
        #
        # A DIRECTIVE IS AN INSTRUCTION WITH NO NEGATION IN FRONT OF THE FLAG.
        # That is the whole definition and it is why both halves scan with a
        # TEMPERED character class -- `(?:(?!\b(?:no|not|never|...)\b)[^.])*?`
        # -- rather than `[^.]{0,60}`: a plain window matched "the same hazard
        # WITH NO `--force` FOR IT" and "must *not* be given `--force`", both
        # of which are prohibitions, and both of which are live in this
        # repository's own prose. The tempered class cannot cross a negation,
        # so a sentence that denies the flag can never read as a directive.
        #
        # THE SECOND BRANCH IS A COMMAND LINE, and it is the one that matters
        # most: a copy-pasteable `./run-benchmark.sh ... --force` is the form
        # an operator will actually act on, and the first branch cannot see it
        # because `run-benchmark.sh` contains a period and the sentence scan
        # stops at one. It found a real instance -- CLAUDE.md's rebuild table
        # still said "then `--force` on the first ann run" after the prose
        # above it had been corrected.
        #
        # `must` is in the verb list and `give[ns]?` is too, which is why the
        # negation guard is not optional: "must *not* be given `--force`" has
        # two matching verbs and is the opposite of a directive.
        pattern=r"(?:\A|\.)(?:(?!\b(?:no|not|never|cannot|nor)\b)[^.])*?"
                r"\b(?:pass(?:es|ing)?|launch(?:ed|ing)?|re-?runs?|add|use"
                r"|needs?|requires?|must|give[ns]?|supply|with)\b"
                r"(?:(?!\b(?:no|not|never|cannot|nor)\b)[^.])*?--force\b"
                r"|run-benchmark\.sh"
                r"(?:(?!\b(?:no|not|never|cannot|nor)\b)[^\n])*?--force\b",
        accepts=[
            # "accepts" here means "is correctly identified AS a directive".
            "Pass --force on the first ann run after a re-pin.",
            "Re-run with --force to recompute the cached points.",
            "The first run at a new pin needs --force.",
            "Launch the smoke run with --force.",
            "The run must be launched with --force.",
            "So: ./run-benchmark.sh run --profile smoke --engines percona "
            "--force",
        ],
        rejects=[
            # Prohibitions and bare mentions, every one of them live prose in
            # this repository at the time this entry was written.
            "Do not reach for --force instead, even though its help text "
            "offers exactly that.",
            "Not --force, even though its help text offers exactly that.",
            "--force is the escape hatch on the free-disk refusal, so it "
            "disables the warning.",
            "Move the subtree aside before the run; not --force, which also "
            "disables the refusal.",
            "--force is deprecated and has no effect on the ann path.",
            "the ops path has the same hazard WITH NO --force FOR IT",
            "The run needs a fresh --run-id and must *not* be given --force.",
            "Run ./run-benchmark.sh run --profile smoke --engines percona, "
            "and never --force.",
        ]),
}


def _flat(text: str) -> str:
    """Whitespace-normalised, so a reflow cannot defeat a prose guard."""
    return " ".join(text.split())


def _prose(name: str, haystack: str):
    """Assert a named claim is still made, however it is worded."""
    entry = PROSE_CLAIMS[name]
    found = re.search(entry["pattern"], _flat(haystack), re.I)
    assert found, (
        f"this claim is no longer made: {name}\n"
        f"  pattern: {entry['pattern']}\n"
        f"  the pattern is a vocabulary, not a sentence -- if the claim is "
        f"still true and only the wording moved, widen PROSE_CLAIMS rather "
        f"than restoring the old phrasing")
    return found


class TestTheProseGuardsAreNeitherBrittleNorVacuous:
    """Each vocabulary against its own accepts and rejects.

    Without this the patterns are untested assertions about assertions. It is
    the check that caught the exact-phrase versions: three legitimate
    rewordings of the engine count and one of the ingest scope were rejected
    by guards that had looked fine in review.
    """

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_every_accepted_wording_matches(self, name):
        entry = PROSE_CLAIMS[name]
        assert entry["accepts"], name
        for wording in entry["accepts"]:
            assert re.search(entry["pattern"], _flat(wording), re.I), (
                f"{name}: fires on correct prose: {wording!r}")

    @pytest.mark.parametrize("name", sorted(PROSE_CLAIMS))
    def test_every_rejected_wording_fails(self, name):
        entry = PROSE_CLAIMS[name]
        assert entry["rejects"], name
        for wording in entry["rejects"]:
            assert not re.search(entry["pattern"], _flat(wording), re.I), (
                f"{name}: accepts the attack: {wording!r}")


# ---------------------------------------------------------------------------
# A recording cursor, so the SQL a driver sends is the thing under test
# ---------------------------------------------------------------------------

class FakeCursor:
    """Records every statement and replays canned answers.

    `fetchone` returns whatever was queued, so `SHOW CREATE TABLE` can be made
    to answer the way a server would without one being reachable.
    """

    def __init__(self, answers=None):
        self.statements = []
        self.params = []
        self._answers = list(answers or [])
        self._last = None

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self.params.append(params)
        self._last = self._answers.pop(0) if self._answers else None

    def executemany(self, sql, rows):
        self.statements.append(sql)

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last or []

    def close(self):
        pass

    @property
    def sql(self):
        return "\n".join(self.statements)


def driver_with(cls, answers=None, index=None):
    driver = cls(ConnectionSpec(host="h", port=3306))
    driver._cur = FakeCursor(answers)
    if index is not None:
        driver._index = index
    return driver


def spec(**kw):
    base = dict(dim=8, m=16, metric="euclidean", build_mode="post")
    base.update(kw)
    return IndexSpec(**base)


# ---------------------------------------------------------------------------
# §4.1 -- the five new Dialect fields, each defaulted so nothing else moves
# ---------------------------------------------------------------------------

class TestTheNewDialectFieldsDefaultToTodaysBehaviour:
    """Five fields were added to a dataclass two live engines share.

    The risk is not that Percona gets them wrong; it is that MariaDB or AliSQL
    silently gets a different statement because a default was chosen to suit
    the new engine. Each test below therefore pins the OLD engines' value.
    """

    NEW_FIELDS = ("pk_type", "index_clause", "distance_expr", "plan_marker",
                  "ef_construction")

    @pytest.mark.parametrize("field", NEW_FIELDS)
    def test_the_field_exists_and_a_bare_dialect_gets_the_default(self, field):
        bare = MySQLDialect(name="probe")
        assert hasattr(bare, field), field

    def test_a_bare_dialect_is_mariadbs_spelling(self):
        """The defaults ARE MariaDB's statements, which is what makes adding
        the fields a no-op for the engines that predate them."""
        bare = MySQLDialect(name="probe")
        assert bare.pk_type == "INT"
        assert bare.index_clause == (
            "VECTOR INDEX {name} (v) M={m} DISTANCE={metric}")
        assert bare.distance_expr == "vec_distance_{metric}(v, %s)"
        assert bare.plan_marker == ""
        assert bare.ef_construction is None

    @pytest.mark.parametrize("dialect", [MARIADB_DIALECT, ALISQL_DIALECT])
    def test_neither_existing_engine_sets_any_of_them(self, dialect):
        """Not "the defaults are right" but "neither dialect overrides them",
        which is the stronger statement: a dialect that set `pk_type="INT"`
        explicitly would pass a value check and would still be a file someone
        has to keep in step."""
        bare = MySQLDialect(name="probe")
        for field in self.NEW_FIELDS:
            assert getattr(dialect, field) == getattr(bare, field), field

    def test_percona_sets_every_one_of_them(self):
        """The mirror image, and the reason the fields exist at all."""
        bare = MySQLDialect(name="probe")
        differs = [f for f in self.NEW_FIELDS
                   if getattr(PERCONA_DIALECT, f) != getattr(bare, f)]
        assert sorted(differs) == sorted(self.NEW_FIELDS), differs


class TestTheDDLEachEngineActuallySends:
    """`create_schema` is exercised and its statement read back."""

    @pytest.mark.parametrize("cls,expected_pk", [(MariaDBDriver, "id INT"),
                                                 (AliSQLDriver, "id INT")])
    def test_the_existing_engines_still_declare_an_int_key(self, cls,
                                                           expected_pk):
        driver = driver_with(cls)
        driver.create_schema(spec())
        assert expected_pk in driver._cur.sql
        assert "BIGINT UNSIGNED" not in driver._cur.sql

    def test_percona_declares_the_key_the_server_requires(self):
        """`BIGINT UNSIGNED`, single column. The server refuses `BIGINT`,
        `INT UNSIGNED` and composite keys on any table with a vector index."""
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        sql = driver._cur.sql
        assert "id BIGINT UNSIGNED PRIMARY KEY" in sql
        # The negative that matters: the old literal must be gone, not merely
        # joined by a new one.
        assert "id INT PRIMARY KEY" not in sql

    def test_the_existing_engines_still_send_the_mariadb_index_clause(self):
        driver = driver_with(MariaDBDriver)
        driver.create_schema(spec(m=24, metric="angular"))
        assert "VECTOR INDEX vi (v) M=24 DISTANCE=cosine" in driver._cur.sql

    def test_percona_sends_a_parenthesised_option_list(self):
        """`TYPE hnsw (M = n, ef_construction = n, metric = name)`.

        An option value is a bare identifier or an integer and never a quoted
        string -- `metric = 'cosine'`, `M = 6.0` and `M = "6"` are all parse
        errors on this server -- so the absence of quotes is asserted too.
        """
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(m=24, metric="angular",
                                  ef_construction=320,
                                  build_mode="incremental"))
        sql = driver._cur.sql
        assert ("VECTOR KEY vi (v) TYPE hnsw "
                "(M = 24, ef_construction = 320, metric = cosine)") in sql
        assert "metric = 'cosine'" not in sql
        assert 'metric = "cosine"' not in sql

    def test_percona_takes_ef_construction_from_the_spec_not_the_dialect(self):
        """The dialect's 200 is what a by-hand reproduction gets; a profile
        that asks for something else must reach the DDL, or an
        ef_construction sweep would silently run one value seven times."""
        assert PERCONA_DIALECT.ef_construction == 200
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(ef_construction=64,
                                  build_mode="incremental"))
        assert "ef_construction = 64" in driver._cur.sql
        assert "ef_construction = 200" not in driver._cur.sql

    def test_percona_falls_back_to_the_dialect_when_the_spec_says_nothing(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(ef_construction=None,
                                  build_mode="incremental"))
        assert "ef_construction = 200" in driver._cur.sql

    @pytest.mark.parametrize("cls", [MariaDBDriver, AliSQLDriver,
                                     PerconaDriver])
    def test_every_engine_indexes_both_filter_columns_the_same_way(self, cls):
        """The parity `tag` and `ctag` have is the measurement, not tidiness:
        the two columns exist to be compared, and an optimiser offered an
        index on one and not the other is choosing between plans rather than
        between filter correlations."""
        driver = driver_with(cls)
        driver.create_schema(spec(build_mode="incremental"))
        sql = driver._cur.sql
        assert "KEY tag_idx (tag)" in sql
        assert "KEY ctag_idx (ctag)" in sql


class TestTheSelectEachEngineActuallySends:

    def test_the_existing_engines_still_use_one_function_per_metric(self):
        for cls, want in ((MariaDBDriver, "vec_distance_cosine(v, %s)"),
                          (AliSQLDriver, "vec_distance_cosine(v, %s)")):
            driver = driver_with(cls, index=spec(metric="angular"))
            assert want in driver._select(10, False)

    def test_percona_passes_the_metric_as_a_third_argument(self):
        """`DISTANCE()` is registered with arity 3 and the metric is
        mandatory. Uppercase here is for the reader -- the server matches it
        case-insensitively."""
        driver = driver_with(PerconaDriver, index=spec(metric="angular"))
        assert "DISTANCE(v, %s, 'COSINE')" in driver._select(10, False)

    def test_the_metric_stays_a_literal_and_not_a_bound_parameter(self):
        """One placeholder in the unfiltered form and two in the filtered one,
        for every engine. Making the metric a parameter would add a third and
        reorder every call site -- `query_filtered` binds
        `(threshold, vector)` because the predicate precedes the ORDER BY."""
        for cls in (MariaDBDriver, AliSQLDriver, PerconaDriver):
            driver = driver_with(cls, index=spec(metric="euclidean"))
            assert driver._select(10, False).count("%s") == 1, cls
            assert driver._select(10, True).count("%s") == 2, cls

    def test_percona_builds_the_index_and_queries_it_with_one_metric_name(self):
        """A cosine index serves only 'COSINE' and an L2 index only
        'EUCLIDEAN' -- never the cross product, because the optimizer reads
        the metric back out of the stored index option. A mismatch is a
        SILENT FULL SCAN rather than an error, so both statements must come
        from the same `metric_names` entry.
        """
        for metric, want in (("angular", "cosine"), ("euclidean", "euclidean")):
            driver = driver_with(PerconaDriver)
            driver.create_schema(spec(metric=metric, build_mode="incremental"))
            assert f"metric = {want}" in driver._cur.sql
            assert f"'{want.upper()}'" in driver._select(10, False)

    def test_only_the_two_indexable_metrics_are_offered(self):
        """`euclidean_squared`, `dot` and `manhattan` are accepted by
        DISTANCE() and cannot be indexed, so this benchmark can never show a
        Percona inner-product frontier. The map is the place that decides."""
        assert set(PERCONA_DIALECT.metric_names) == {"angular", "euclidean"}
        assert set(PERCONA_DIALECT.metric_names.values()) == {"cosine",
                                                              "euclidean"}


# ---------------------------------------------------------------------------
# §4.1 -- the plan check, which is vacuous for this engine without a marker
# ---------------------------------------------------------------------------

class TestThePlanCheckIsNotSatisfiedByTheIndexNameAlone:
    """The defect this field exists for, reproduced against the driver.

    Percona's own recorded output (`…/percona_vector/r/optimizer.result:
    491-492`) contains a plan with access type `ALL`, `key=NULL` and
    `Extra=Using filesort` -- a full table scan -- on a table whose only index
    is a `VECTOR KEY`, with the index name sitting in `possible_keys`. A
    substring test against the joined EXPLAIN text returns True for it.

    The rows below are that plan and its passing counterpart (`:95-96`),
    transcribed, and fed to the real `explain_uses_vector_index`.
    """

    #: A brute-force scan whose plan text still contains the index name.
    SCAN = [(1, "SIMPLE", "t1", None, "ALL", "vi", None, None, None,
             100, 100.00, "Using filesort")]
    #: The vector path actually taken.
    ANN = [(1, "SIMPLE", "t1", None, "vector_ann", "vi", "vi", None, None,
            10, 100.00, "Using where")]

    def _ask(self, cls, rows):
        driver = driver_with(cls, answers=[rows], index=spec())
        return driver.explain_uses_vector_index([0.0] * 8, 10)

    def test_percona_rejects_the_full_scan_that_names_the_index(self):
        assert self._ask(PerconaDriver, self.SCAN) is False

    def test_percona_accepts_the_plan_that_took_the_vector_path(self):
        assert self._ask(PerconaDriver, self.ANN) is True

    def test_the_index_name_alone_would_have_accepted_the_scan(self):
        """NON-VACUITY, and the reason `plan_marker` is not cosmetic.

        Measured against the engines that have no marker: given the very same
        brute-force plan, MariaDB's and AliSQL's check -- which is the check
        Percona would have inherited -- returns True. That is brute force
        published as ANN search, and it is the difference the marker makes.
        """
        assert self._ask(MariaDBDriver, self.SCAN) is True
        assert self._ask(AliSQLDriver, self.SCAN) is True
        assert self._ask(PerconaDriver, self.SCAN) is False

    def test_the_marker_alone_is_not_enough_either(self):
        """Both conjuncts. The access type is the load-bearing half; the name
        is kept so a second vector index on the table could not satisfy the
        check for the wrong one."""
        other_index = [(1, "SIMPLE", "t1", None, "vector_ann", "other",
                        "other", None, None, 10, 100.00, "Using where")]
        assert self._ask(PerconaDriver, other_index) is False

    @pytest.mark.parametrize("cls", [MariaDBDriver, AliSQLDriver,
                                     PerconaDriver])
    def test_every_engine_pins_the_traditional_explain_format(self, cls):
        """`plan_marker` is a value in the `type` column, which only the
        tabular format has. `explain_format` defaults to TRADITIONAL so a bare
        EXPLAIN worked -- but it is settable and reachable through
        `VB_SERVER_ARGS`, and under TREE or JSON the marker vanishes and the
        check records a FALSE NEGATIVE about a server that did use the index.
        That lands in the report's Validity section.
        """
        driver = driver_with(cls, answers=[self.ANN], index=spec())
        driver.explain_uses_vector_index([0.0] * 8, 10)
        assert driver._cur.statements[0].startswith(
            "EXPLAIN FORMAT=TRADITIONAL ")

    def test_the_marker_is_read_from_the_type_column_and_not_the_text(self):
        """The last substring in the chain, closed.

        `plan_marker in plan` over the joined row text is satisfied by the
        marker appearing ANYWHERE -- a table named `vector_ann`, the string in
        `Extra`, an index called `vector_ann_idx`. None of those is reachable
        for the statements this driver sends (`t1`, fixed columns), so this is
        a class being closed rather than a live defect being fixed. It is
        worth closing because the argument for `plan_marker` existing at all
        is that a substring check accepted a brute-force scan.

        Each row below is a full table scan -- type `ALL`, `key` NULL -- with
        the marker smuggled into a different column.
        """
        smuggled = {
            "table name": (1, "SIMPLE", "vector_ann", None, "ALL", "vi",
                            None, None, None, 100, 100.00, "Using filesort"),
            "possible_keys": (1, "SIMPLE", "t1", None, "ALL", "vi,vector_ann",
                               None, None, None, 100, 100.00, "Using filesort"),
            "Extra": (1, "SIMPLE", "t1", None, "ALL", "vi", None, None, None,
                       100, 100.00, "Using filesort; vector_ann unusable"),
            "ref": (1, "SIMPLE", "t1", None, "ALL", "vi", None, None,
                     "vector_ann", 100, 100.00, "Using filesort"),
        }
        for where, row in smuggled.items():
            driver = driver_with(PerconaDriver, answers=[[row]], index=spec())
            assert driver.explain_uses_vector_index([0.0] * 8, 10) is False, \
                where
            # Non-vacuity for each: the text form WOULD have accepted it, so
            # these rows really do exercise the difference.
            plan = " | ".join(str(c) for c in row)
            assert "vector_ann" in plan, where

    def test_the_access_type_is_matched_case_insensitively(self):
        """A server that capitalised an access type would otherwise turn a
        used index into a reported full scan."""
        row = (1, "SIMPLE", "t1", None, " VECTOR_ANN ", "vi", "vi", None,
               None, 10, 100.00, "Using where")
        driver = driver_with(PerconaDriver, answers=[[row]], index=spec())
        assert driver.explain_uses_vector_index([0.0] * 8, 10) is True

    def test_an_unexpected_row_shape_falls_back_rather_than_reporting_false(
            self, capsys):
        """A `False` here is published as "this engine did not use its index".

        An EXPLAIN shape with fewer than twelve columns is a statement about
        the FORMAT, not about the engine, so the method falls back to the text
        match it used before -- no plan that passed can start failing -- and
        says on stderr that it did.
        """
        short = (1, "SIMPLE", "t1", "vector_ann", "vi")
        driver = driver_with(PerconaDriver, answers=[[short]], index=spec())
        assert driver.explain_uses_vector_index([0.0] * 8, 10) is True
        out = " ".join(capsys.readouterr().out.split())
        assert "falling back to a text match" in out
        # And a short row that does NOT carry the marker is still False, so
        # the fallback is a fallback and not an unconditional pass.
        driver = driver_with(PerconaDriver,
                             answers=[[(1, "SIMPLE", "t1", "ALL", "vi")]],
                             index=spec())
        assert driver.explain_uses_vector_index([0.0] * 8, 10) is False

    def test_the_markerless_engines_are_untouched_by_any_of_this(self):
        """MariaDB and AliSQL declare no marker, so the access-type path is
        never entered and their check is exactly what it was."""
        for cls in (MariaDBDriver, AliSQLDriver):
            assert cls.dialect.plan_marker == ""
            short = (1, "SIMPLE", "t1", "vi")
            driver = driver_with(cls, answers=[[short]], index=spec())
            assert driver.explain_uses_vector_index([0.0] * 8, 10) is True

    def test_a_failed_check_names_what_it_looked_for(self, capsys):
        """"Not used" over two conjuncts does not say which failed, and the
        two mean different things: a missing index name is the optimiser
        taking another index, a missing access type with the name present is
        the brute-force scan."""
        self._ask(PerconaDriver, self.SCAN)
        # Whitespace-normalised: the message is built from several adjacent
        # f-string fragments, and re-wrapping those fragments is an edit that
        # must not be able to defeat this guard.
        out = " ".join(capsys.readouterr().out.split())
        assert "vector_ann" in out and "'vi'" in out


# ---------------------------------------------------------------------------
# §2.5 / §4.1 -- capabilities must follow the build mode, not the class
# ---------------------------------------------------------------------------

class TestCapabilitiesFollowTheBuildMode:
    """`harness/workloads/build.py` is the reason this is not a constant.

    Its flag is `capabilities().get("incremental_index", True)` with an
    override forcing it True for `build_mode == "incremental"` -- and no
    counterpart forcing it False for `post`. The flag then decides whether
    `create_index` is called at all and what `build_wall_s` means.
    """

    def test_build_py_still_has_no_post_counterpart(self):
        """The premise, checked against the file rather than assumed. If
        `build.py` ever grows an `elif build_mode == "post"`, this driver's
        derivation becomes belt-and-braces and someone should know."""
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "build.py")).read()
        assert 'index.build_mode == "incremental"' in source   # findable
        assert 'build_mode == "post"' not in source

    def test_post_mode_reports_a_separable_build(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="post"))
        assert driver.capabilities()["incremental_index"] is False

    def test_incremental_mode_reports_no_separable_build(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        assert driver.capabilities()["incremental_index"] is True

    def test_before_a_schema_exists_the_answer_is_the_engines_own_default(self):
        """Percona maintains the graph on the write path unless a profile asks
        otherwise, so `True` is the honest answer with no spec to read."""
        assert driver_with(PerconaDriver).capabilities()[
            "incremental_index"] is True

    def test_the_two_modes_do_not_produce_the_same_capabilities(self):
        """An identical pair here would mean the derivation is dead."""
        post = driver_with(PerconaDriver)
        post.create_schema(spec(build_mode="post"))
        inc = driver_with(PerconaDriver)
        inc.create_schema(spec(build_mode="incremental"))
        assert post.capabilities() != inc.capabilities()

    def test_the_existing_engines_are_unaffected_by_build_mode(self):
        """MariaDB and AliSQL have no separable build, so both modes must
        still report incremental -- a `post` profile on them calls a
        documented no-op and the load IS the build."""
        for cls in (MariaDBDriver, AliSQLDriver):
            for mode in ("post", "incremental"):
                driver = driver_with(cls)
                driver.create_schema(spec(build_mode=mode))
                assert driver.capabilities()["incremental_index"] is True

    def test_percona_is_the_only_mysql_family_engine_with_the_build_knob(self):
        for cls, want in ((MariaDBDriver, False), (AliSQLDriver, False),
                          (PerconaDriver, True)):
            driver = driver_with(cls)
            assert driver.capabilities()["ef_construction_tunable"] is want


class TestTheIndexLeavesTheTableDefinitionInBulkMode:
    """What `build_mode` actually changes, which is the DDL and not a label."""

    def test_post_mode_creates_the_table_without_the_index(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="post"))
        assert "VECTOR KEY" not in driver._cur.sql
        assert "CREATE TABLE" in driver._cur.sql

    def test_post_mode_adds_the_index_in_create_index(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="post"))
        before = len(driver._cur.statements)
        driver.create_index(spec(build_mode="post"))
        added = driver._cur.statements[before:]
        assert len(added) == 1
        assert added[0].startswith(f"ALTER TABLE ")
        assert "ADD VECTOR KEY vi (v) TYPE hnsw" in added[0]
        # INPLACE builds the graph in memory and bulk-loads the aux table; a
        # COPY-built aux is correct but slower to build and larger. Leaving
        # the choice to the server would let "bulk build" be either one.
        assert "ALGORITHM=INPLACE" in added[0]

    def test_incremental_mode_declares_the_index_inline(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        assert "VECTOR KEY vi (v) TYPE hnsw" in driver._cur.sql

    def test_incremental_mode_makes_create_index_a_no_op(self):
        driver = driver_with(PerconaDriver)
        driver.create_schema(spec(build_mode="incremental"))
        before = len(driver._cur.statements)
        driver.create_index(spec(build_mode="incremental"))
        assert driver._cur.statements[before:] == []


# ---------------------------------------------------------------------------
# §2.7 -- the repair path, and the options it must preserve
# ---------------------------------------------------------------------------

class TestTheRepairPath:

    SHOW_CREATE = (
        "t1",
        "CREATE TABLE `t1` (\n"
        "  `id` bigint unsigned NOT NULL,\n"
        "  `v` vector(8) NOT NULL,\n"
        "  PRIMARY KEY (`id`),\n"
        "  VECTOR KEY `vi` (`v`) USING `hnsw` "
        "(`M`=24, `ef_construction`=320, `metric`=`cosine`)\n"
        ") ENGINE=InnoDB",
    )

    def test_percona_declares_its_repair_path(self):
        assert PerconaDriver.repair_path == "index-replace"

    def test_the_existing_engines_still_declare_none(self):
        assert MariaDBDriver.repair_path == "none"
        assert AliSQLDriver.repair_path == "none"

    def test_maintain_replaces_the_index_in_one_alter(self):
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec())
        what = driver.maintain()
        alter = [s for s in driver._cur.statements if s.startswith("ALTER")]
        assert len(alter) == 1
        assert "DROP KEY vi" in alter[0] and "ADD VECTOR KEY" in alter[0]
        assert what == "ALTER TABLE (index replace)"

    def test_maintain_is_not_optimize_table(self):
        """`OPTIMIZE TABLE` also compacts and is deliberately not used: it
        rewrites the whole base table, so its cost is dominated by work that
        has nothing to do with the vector index and the churn phase would be
        timing a table rebuild."""
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec())
        driver.maintain()
        assert "OPTIMIZE" not in driver._cur.sql.upper()

    def test_the_rebuild_reuses_the_servers_own_option_list(self):
        """THE CORRECTNESS ARGUMENT FOR READING IT BACK.

        The DDL that rebuilds the index decides the new graph's M,
        ef_construction and metric. Omitting them rebuilds at the server's
        defaults -- M=25, not the profile's -- and the post-maintenance recall
        would be a different index's recall reported as the repaired one's.

        The spec here deliberately carries DIFFERENT values (M=16,
        ef_construction unset) from the live index the fake server reports
        (M=24, ef_construction=320), so a rebuild derived from the profile
        instead of from the server fails this.
        """
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec(m=16, ef_construction=None))
        driver.maintain()
        alter = [s for s in driver._cur.statements if s.startswith("ALTER")][0]
        assert "`M`=24" in alter
        assert "`ef_construction`=320" in alter
        assert "`metric`=`cosine`" in alter
        # And not the profile's values, which is the half that bites.
        assert "M = 16" not in alter

    def test_it_asks_the_server_before_it_rebuilds(self):
        driver = driver_with(PerconaDriver, answers=[self.SHOW_CREATE],
                             index=spec())
        driver.maintain()
        assert driver._cur.statements[0].startswith("SHOW CREATE TABLE")

    def test_a_server_that_cannot_answer_is_warned_about_not_silently_guessed(
            self, capsys):
        """A silent fall-back here is the "rebuilt at the defaults" failure
        this method exists to avoid, except with our numbers instead of the
        server's -- so it happens loudly or not at all."""
        driver = driver_with(PerconaDriver, answers=[("t1", "CREATE TABLE t1")],
                             index=spec(m=16))
        driver.maintain()
        out = " ".join(capsys.readouterr().out.split())
        assert "WARNING" in out
        assert "post-maintenance recall is a different index's" in out
        alter = [s for s in driver._cur.statements if s.startswith("ALTER")][0]
        assert "M = 16" in alter

    def test_the_churn_hooks_percona_does_not_need_are_inherited(self):
        """`prepare_churn` and `wait_index_synced` stay the base class's.

        Percona has no background graph maintenance at all, so there is no
        knob whose state could differ between the baseline and the post-churn
        measurement, and nothing to wait for: the graph is maintained inside
        the INSERT and the honest wait is zero. Returning a small positive
        number "to be safe" would be a fabricated measurement in a field
        documented as a measurement.
        """
        assert PerconaDriver.prepare_churn is EngineDriver.prepare_churn
        assert PerconaDriver.wait_index_synced is EngineDriver.wait_index_synced
        assert driver_with(PerconaDriver).wait_index_synced(1000) == 0.0


class TestTheEfSearchClamp:

    def test_percona_clamps_up_to_k(self):
        driver = driver_with(PerconaDriver)
        assert [driver.effective_ef(ef, 100) for ef in (10, 40, 160)] == \
            [100, 100, 160]

    def test_it_changes_nothing_at_the_k_every_profile_uses(self):
        """Honest about its own reach: at k=10 every grid value this repo
        sweeps is above 10, so this override moves no figure today. It is
        implemented because the inherited identity would be wrong at the one
        grid point where it mattered and silently right everywhere else."""
        driver = driver_with(PerconaDriver)
        assert [driver.effective_ef(ef, 10) for ef in (10, 40, 160)] == \
            [10, 40, 160]

    def test_the_existing_mysql_engines_do_not_clamp(self):
        for cls in (MariaDBDriver, AliSQLDriver):
            driver = driver_with(cls)
            assert driver.effective_ef(10, 100) == 10

    def test_ef_search_is_set_per_session_and_not_globally(self):
        """The variable has both scopes; a GLOBAL write would outlive the
        calibration sweep that set it and reach every later phase."""
        driver = driver_with(PerconaDriver)
        driver.set_ef_search(200)
        stmt = driver._cur.statements[0]
        assert stmt == "SET SESSION innodb_hnsw_ef_search = 200"
        assert "GLOBAL" not in stmt


# ---------------------------------------------------------------------------
# §4.2 -- the driver registries
# ---------------------------------------------------------------------------

class TestTheDriverIsReachable:

    def test_the_class_registry_carries_it(self):
        assert _driver_classes()["PerconaDriver"] is PerconaDriver

    def test_the_engine_table_carries_it(self):
        assert _driver_table()["percona"] is PerconaDriver

    def test_the_config_names_the_class_the_registry_knows(self):
        """The two halves of `--driver`: the orchestrator reads the name out of
        config/engines/percona.yml and the harness looks it up here. A typo in
        either is `argument --engine: invalid choice` an hour into a run."""
        from orchestrator import engines as engines_mod
        declared = engines_mod.registry(refresh=True)["percona"].driver
        assert declared in _driver_classes()

    def test_every_registered_engine_has_a_driver_class(self):
        from orchestrator import engines as engines_mod
        classes = _driver_classes()
        for name, runtime in engines_mod.registry(refresh=True).items():
            assert runtime.driver in classes, name


# ---------------------------------------------------------------------------
# §4.3 / §4.5 -- the config, and the hand-kept copy of two of its values
# ---------------------------------------------------------------------------

class TestTheEngineConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        import yaml
        with open(os.path.join(VB_ROOT, "config", "engines",
                               "percona.yml")) as fh:
            return yaml.safe_load(fh)

    def test_the_source_pin_is_the_sha_under_the_tag_key(self, cfg):
        """`_source_tag` reads `tag` or `version` and there is no `sha` key, so
        a `source: sha:` block would make it return "" and `prepare_percona`
        die -- it passes `""` as the default and `die`s on an empty value.
        There is no hard-coded fallback for THIS engine, unlike the three
        other `prepare_*` functions that read the key; this docstring was the
        third copy of that error in the tree.
        """
        assert cfg["source"]["tag"] == PINNED_SHA
        assert "sha" not in cfg["source"]

    def test_the_registry_carries_that_pin_into_the_manifest(self):
        from orchestrator import engines as engines_mod
        assert engines_mod.registry(refresh=True)["percona"].tag == PINNED_SHA

    def test_the_pin_is_a_commit_and_could_not_be_a_branch_name(self, cfg):
        """The two tests above compare this value against this file's own
        constant, so `9.7` in both places passes both of them -- and `git
        archive` takes a branch name as happily as a SHA, resolving it afresh
        on every build, so two runs carrying the same manifest would not be
        the same server. The branch this pin sits in moved four commits in the
        day after the pin was taken. So the SHAPE is asserted separately from
        the value, on the config and on the constant.
        """
        assert re.fullmatch(r"[0-9a-f]{40}", str(cfg["source"]["tag"])), \
            cfg["source"]["tag"]
        assert re.fullmatch(r"[0-9a-f]{40}", PINNED_SHA), PINNED_SHA

    def test_no_submodules_key(self, cfg):
        """All four of the pin's submodules are switched off in the build
        instead. Listing them here would make it fetch from three more hosts
        for code that is never compiled."""
        assert "submodules" not in cfg["source"]

    def test_no_sql_block(self, cfg):
        assert "sql" not in cfg

    def test_the_graph_memory_knob_is_unlimited_in_both_passes(self, cfg):
        """Decided policy, not a default. §3: Percona's is a hard allocation
        ceiling and not a cache, so feeding it `graph_cache_bytes` would
        compare a cache budget against a wall."""
        for pass_name in ("normalized", "tuned"):
            flags = cfg["server"][pass_name]
            assert "--innodb-hnsw-max-memory=0" in flags, pass_name
            assert not any("graph_cache_bytes" in f for f in flags), pass_name

    def test_it_is_not_given_the_other_engines_cache_knobs(self, cfg):
        """A different variable name with the same intent would reintroduce
        the same error under another spelling.

        Comments are stripped first, because the file EXPLAINS the absence by
        naming both knobs -- and the first version of this test matched its
        own explanation. That is the inverse of the usual vacuity: a negative
        assertion failing on prose that agrees with it.
        """
        text = open(os.path.join(VB_ROOT, "config", "engines",
                                 "percona.yml")).read()
        keys = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
        for knob in ("mhnsw_max_cache_size", "vidx_hnsw_cache_size",
                     "mhnsw-max-cache-size", "vidx-hnsw-cache-size"):
            assert knob not in keys, knob
        # ...and the explanation really is still there, so this is not
        # passing because the file went silent on the question.
        assert "mhnsw_max_cache_size" in text

    def test_the_march_matches_every_engine_that_compiles_one(self, cfg):
        """"Never mix -march between engines" is a stated invariant.

        Compared against the other configs rather than against a literal, so
        a repo-wide change of the flag does not need this test edited to stay
        true.

        `march: none` is excluded and that is not a loophole: mongodb and
        valkey compile nothing -- they are installed from published packages
        -- so they have no SIMD kernels of ours to mismatch. The engines that
        DO compile are the ones the invariant is about, and `none` is checked
        to be exactly the non-compiling set rather than waved through.
        """
        import yaml
        marches = {}
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "config",
                                                   "engines"))):
            if not name.endswith(".yml"):
                continue
            with open(os.path.join(VB_ROOT, "config", "engines", name)) as fh:
                build = yaml.safe_load(fh).get("build") or {}
            if build.get("march"):
                marches[name[:-4]] = build["march"]
        compiling = {k: v for k, v in marches.items() if v != "none"}
        assert len(compiling) >= 4, marches             # non-vacuity
        assert "percona" in compiling
        assert len(set(compiling.values())) == 1, compiling
        # The ones that opt out are exactly the ones that build nothing.
        assert {k for k, v in marches.items() if v == "none"} <= {"mongodb",
                                                                  "valkey"}

    def test_the_ann_constructor_is_unique_and_is_not_the_mongodb_one(self):
        """ann-benchmarks keys result files on this name, so a duplicate makes
        the second engine report the first's recall -- the defect `mariadb123`
        exists to prevent. `PerconaSearch` is the MongoDB engine."""
        from orchestrator import engines as engines_mod
        registry = engines_mod.registry(refresh=True)
        names = [e.ann_constructor for e in registry.values()]
        assert len(names) == len(set(names)), names
        assert registry["percona"].ann_constructor == "PerconaServer"
        assert registry["mongodb"].ann_constructor == "PerconaSearch"

    def test_the_image_tags_are_unique_too(self):
        import yaml
        tags = []
        for name in sorted(os.listdir(os.path.join(VB_ROOT, "config",
                                                   "engines"))):
            if not name.endswith(".yml"):
                continue
            with open(os.path.join(VB_ROOT, "config", "engines", name)) as fh:
                image = yaml.safe_load(fh).get("image") or {}
            tags.extend(v for k, v in image.items() if k in ("runtime", "bench"))
        assert len(tags) == len(set(tags)), sorted(tags)

    def test_percona_took_an_unused_colour_and_an_unused_marker(self):
        """Scoped to this engine, because markers are NOT globally unique and
        pretending otherwise would be a false claim: mariadb and mariadb123
        deliberately share `s`, being the same software at two versions, and
        are told apart by colour. Colours ARE unique and that is asserted
        across the whole registry."""
        from orchestrator import engines as engines_mod
        registry = engines_mod.registry(refresh=True)
        colours = [e.color for e in registry.values()]
        assert len(colours) == len(set(colours)), colours
        others = [e for name, e in registry.items() if name != "percona"]
        assert others                                   # non-vacuity
        mine = registry["percona"]
        assert mine.color == "#ff7f0e"
        assert mine.marker == "P"
        assert mine.color not in {e.color for e in others}
        assert mine.marker not in {e.marker for e in others}


class TestTheWebUiCopyMatchesTheRegistryCharacterForCharacter:
    """The front end is static and cannot read the engine registry, so
    `webui/static/fields.js` holds a second copy of every engine's colour and
    label. This is the one place in the addition where two files are kept in
    step by hand, which is why it is checked for EVERY engine and not only for
    the new one."""

    @pytest.fixture(scope="class")
    def maps(self):
        source = open(os.path.join(VB_ROOT, "webui", "static",
                                   "fields.js")).read()
        out = {}
        for const in ("ENGINE_COLOR", "ENGINE_LABEL"):
            body = source.split(f"const {const} = {{", 1)[1].split("};", 1)[0]
            out[const] = dict(re.findall(r'(\w+)\s*:\s*"([^"]*)"', body))
        return out

    def test_the_parser_found_both_maps(self, maps):
        """Non-vacuity: an empty dict would make every loop below pass."""
        assert len(maps["ENGINE_COLOR"]) >= 7, maps["ENGINE_COLOR"]
        assert len(maps["ENGINE_LABEL"]) >= 7, maps["ENGINE_LABEL"]

    def test_every_engine_appears_in_both_maps(self, maps):
        from orchestrator import engines as engines_mod
        registry = engines_mod.registry(refresh=True)
        assert set(maps["ENGINE_COLOR"]) == set(registry)
        assert set(maps["ENGINE_LABEL"]) == set(registry)

    def test_every_colour_and_label_matches_the_yaml_exactly(self, maps):
        """`chart_label` and not `label`: these maps feed charts, and the two
        differ for mongodb, whose chart says "Percona Search (mongot)" while
        its report label spells out "for MongoDB". For percona the two are the
        same string, so getting this wrong would have passed for the new
        engine and failed for an old one -- which is how it was found."""
        from orchestrator import engines as engines_mod
        for name, runtime in engines_mod.registry(refresh=True).items():
            assert maps["ENGINE_COLOR"][name] == runtime.color, name
            assert maps["ENGINE_LABEL"][name] == runtime.chart_label, name
        # Percona's two labels are identical, which is what makes the single
        # fields.js entry sufficient for it.
        percona = engines_mod.registry(refresh=True)["percona"]
        assert percona.chart_label == percona.label
        assert percona.label == "Percona Server (InnoDB HNSW)"

    def test_the_build_chart_hint_mentions_the_engine_that_does_both(self):
        """The hint read "pgvector bulk-builds; MHNSW and VIDX build on every
        INSERT", which became incomplete with a third behaviour in the chart.

        Matched on whitespace-normalised text, because a hint long enough to
        describe three behaviours is long enough for an editor to wrap.
        """
        source = open(os.path.join(VB_ROOT, "webui", "static",
                                   "fields.js")).read()
        assert "pgvector bulk-builds" in _flat(source)   # needle findable
        _prose("percona does either build mode", source)


# ---------------------------------------------------------------------------
# §4.10 -- the render branch, and the key it refuses
# ---------------------------------------------------------------------------

class TestTheAnnRenderBranch:

    @staticmethod
    def _render(ann=None, resource_pass="normalized", resources=None):
        from orchestrator import ann_pass
        profile = {"name": "t", "ann": ann or {}}
        return ann_pass.render_config("percona", profile, resources or {},
                                      resource_pass)

    def test_percona_no_longer_lands_on_the_raise(self):
        body = self._render()
        assert body["float"]["any"][0]["constructor"] == "PerconaServer"

    def test_an_unknown_engine_still_does(self):
        """Non-vacuity: the `raise` is still reachable, so the test above is
        about the branch and not about the raise having been deleted."""
        from orchestrator import ann_pass
        with pytest.raises(ValueError, match="unknown engine"):
            ann_pass.render_config("nope", {"name": "t"}, {}, "normalized")

    def test_it_emits_ef_construction_which_no_mysql_branch_does(self):
        groups = self._render()["float"]["any"][0]["run_groups"]
        args = list(groups.values())[0]["arg_groups"][0]
        assert args["efConstruction"] == [200]

    def test_the_mysql_family_branch_still_does_not(self):
        """The reason Percona needed a branch of its own rather than a line in
        the condition at the top."""
        from orchestrator import ann_pass
        body = ann_pass.render_config(
            "mariadb", {"name": "t", "ann": {}}, {}, "normalized")
        for group in body["float"]["any"][0]["run_groups"].values():
            for args in group["arg_groups"]:
                assert "efConstruction" not in args

    def test_the_tuned_pass_can_sweep_it(self):
        body = self._render(
            resource_pass="tuned",
            resources={"extras": {"percona_ef_construction": [64, 200, 512]}})
        args = list(body["float"]["any"][0]["run_groups"].values())[0]
        assert args["arg_groups"][0]["efConstruction"] == [64, 200, 512]

    def test_the_normalized_pass_ignores_the_tuned_extra(self):
        """Pinned in the normalized pass for the same reason pgvector's and
        Valkey's are: MariaDB and AliSQL lack the knob, and sweeping it where
        only one engine has it hands that engine a tuning axis the others
        cannot answer."""
        body = self._render(
            resources={"extras": {"percona_ef_construction": [64, 512]}})
        args = list(body["float"]["any"][0]["run_groups"].values())[0]
        assert args["arg_groups"][0]["efConstruction"] == [200]

    def test_one_run_group_and_innodb_only(self):
        """The vector index is InnoDB's, implemented in storage/innobase, so
        there is no second storage engine to curve against -- the same shape
        as alisql's branch and not mariadb's."""
        groups = self._render()["float"]["any"][0]["run_groups"]
        assert list(groups) == ["innodb"]
        assert groups["innodb"]["arg_groups"][0]["engine"] == "InnoDB"

    def test_a_build_mode_sweep_is_refused_loudly_not_emitted(self):
        """THE ONE DOCUMENTED INTENTION THAT COULD NOT BE DELIVERED.

        docs/09 §2.5 and §4.10 propose `percona_build_modes` as a within-engine
        ann comparison. The ann overlay module has no `build_mode` --
        pgvector's and valkey's each read it, validate it and branch on it in
        `fit`; `algorithms/percona/module.py` contains the string zero times --
        so emitting the key would produce two run groups differing only in a
        `method_param` nothing reads: two IDENTICAL recall curves under two
        labels, with no field in the record to say they were the same
        measurement.

        A `ValueError` at render time, before anything is built, is the good
        kind of failure. This pins that it is refused rather than honoured,
        and the message has to name where the gap is.
        """
        with pytest.raises(ValueError) as excinfo:
            self._render(ann={"percona_build_modes": ["post", "incremental"]})
        message = str(excinfo.value)
        assert "percona_build_modes" in message
        assert "build_mode" in message
        assert "module.py" in message

    def test_the_default_single_mode_is_accepted(self):
        """`["post"]` is what every profile without the key resolves to, so
        refusing it would refuse every run."""
        assert self._render(ann={"percona_build_modes": ["post"]})

    def test_the_premise_of_that_refusal_is_checked_not_assumed(self):
        """If `percona/module.py` ever learns `build_mode`, this fails and the
        refusal above should be replaced by a build-mode loop rather than
        left in place."""
        overlay = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                               "ann_benchmarks")
        percona = open(os.path.join(overlay, "algorithms", "percona",
                                    "module.py")).read()
        shared = open(os.path.join(overlay, "vb_mysql.py")).read()
        assert "build_mode" not in percona
        assert "build_mode" not in shared
        # Non-vacuity: the two engines that DO support it really do say so,
        # so "the string is absent" is a fact about these files and not about
        # the spelling being wrong.
        for engine in ("pgvector", "valkey"):
            other = open(os.path.join(overlay, "algorithms", engine,
                                      "module.py")).read()
            assert "build_mode" in other, engine


class TestTheOpsPathDoesSupportBothBuildModes:
    """The half that works, and where the comparison actually comes from.

    `orchestrator/cli.py` hard-coded `["post"]` for every engine but pgvector,
    which docs/09 §4 never mentions -- so without this the Percona-only
    profile's headline axis would have been silently inert on both paths.
    """

    def test_the_profile_declares_both_modes_and_the_capability_agrees(self):
        """It declared `[post, incremental]`, was cut to `[incremental]` when
        `post` was found to kill the server at the old pin (9.7.1-1), and is
        back to both since the current pin built it 25 times out of 25 -- see
        `capabilities.bulk_vector_index_build` in
        `config/engines/percona.yml`. The profile and the capability have to
        move together: `post` in the profile with the capability false would
        be the warned-and-honoured crash path in a shipped profile.
        """
        from orchestrator.config import load_profile
        from orchestrator.config import load_engine
        modes = load_profile("percona")["ops"]["percona_build_modes"]
        assert modes == ["post", "incremental"]
        caps = load_engine("percona").get("capabilities", {}) or {}
        assert caps.get("bulk_vector_index_build") is True

    def test_the_key_lives_under_ops_where_the_ann_renderer_cannot_see_it(self):
        """**This assertion was inverted, and the inversion was a defect that
        cost this profile both of its axes.**

        It first read `"percona_build_modes" in profile["ann"]`, following
        `pgvector_build_modes`. That precedent is only safe because pgvector's
        overlay module reads `build_mode`; Percona's does not, which is why
        `render_config` POLICES `ann.percona_build_modes` and raises on any
        value but `["post"]`. With the key under `ann:` the profile tripped
        its own refusal: `run --profile percona` failed its ann phase, and
        `render --profile percona` raised uncaught out of `cmd_render` -- so
        `percona_ef_construction`, which only `render_config` reads, was never
        swept either.

        Under `ops:` the ann renderer structurally cannot see it. The readers
        carry the block in the key, so this is where they look.
        """
        from orchestrator.config import load_profile
        profile = load_profile("percona")
        assert "percona_build_modes" in profile["ops"]
        assert "percona_build_modes" not in profile.get("ann", {})
        # And the `ann:` block still explains the absence, so the next author
        # does not "fix" it by moving the key back.
        text = open(os.path.join(VB_ROOT, "config", "profiles",
                                 "percona.yml")).read()
        ann_block = text.split("ann:", 1)[1].split("\nops:", 1)[0]
        assert "percona_build_modes" in ann_block

    def test_pgvectors_key_still_lives_under_ann_because_it_drives_both(self):
        """The contrast that makes the split meaningful rather than untidy:
        pgvector's list really does drive the ann render as well as the ops
        loop, so it belongs with the ann configuration."""
        from orchestrator.config import load_profile
        for name in ("main", "smoke"):
            profile = load_profile(name)
            assert "pgvector_build_modes" in profile["ann"], name
            assert "pgvector_build_modes" not in profile.get("ops", {}), name

    def test_the_estimate_counts_both_modes_as_two_ops_units(self):
        """If the runner executes a second unit the estimator does not count,
        the printed figure is short by a whole unit of every query term."""
        from orchestrator import estimate as est_mod
        one = {"name": "t",
               "ops": {"m_values": [16], "percona_build_modes": ["post"]}}
        two = {"name": "t",
               "ops": {"m_values": [16],
                       "percona_build_modes": ["post", "incremental"]}}
        assert est_mod._ops_units(one, "percona") == 1
        assert est_mod._ops_units(two, "percona") == 2
        # And no other engine is affected by the key.
        assert est_mod._ops_units(two, "mariadb") == 1
        assert est_mod._ops_units(two, "alisql") == 1
        # The BLOCK is load-bearing, not cosmetic: the same list under `ann:`
        # is not read, which is what keeps it out of `render_config`'s way.
        under_ann = {"name": "t", "ops": {"m_values": [16]},
                     "ann": {"percona_build_modes": ["post", "incremental"]}}
        assert est_mod._ops_units(under_ann, "percona") == 1
        # ...and pgvector's really is read from `ann:`, which is the contrast
        # that makes the split a decision rather than an inconsistency.
        pg = {"name": "t", "ops": {"m_values": [16]},
              "ann": {"pgvector_build_modes": ["post", "incremental"]}}
        assert est_mod._ops_units(pg, "pgvector") == 2

    def test_the_cli_and_the_estimate_use_the_same_engine_list(self):
        """Two copies of "which engines have a separable build, and where the
        list lives", in two files that must agree.

        Compared against each other rather than each pinned to a literal, so a
        third engine added to one and not the other fails here -- and the
        BLOCK is part of what is compared, because a key read from `ann:` in
        one file and `ops:` in the other would silently give the runner two
        units and the estimate one.
        """
        from orchestrator import estimate as est_mod
        pattern = r'"(\w+)":\s*\(\s*"(\w+)",\s*"(\w+)"\s*\)'
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        block = source.split("_OPS_BUILD_MODE_KEYS = {", 1)[1].split("}", 1)[0]
        in_cli = {m[0]: (m[1], m[2]) for m in re.findall(pattern, block)}
        assert in_cli, "the cli lookup did not parse"
        est_source = open(os.path.join(VB_ROOT, "orchestrator",
                                       "estimate.py")).read()
        est_block = est_source.split("mode_ref = {", 1)[1].split("}", 1)[0]
        in_est = {m[0]: (m[1], m[2]) for m in re.findall(pattern, est_block)}
        assert in_est == in_cli, (in_cli, in_est)
        assert in_cli == {"pgvector": ("ann", "pgvector_build_modes"),
                          "percona": ("ops", "percona_build_modes")}

    def test_the_ef_construction_sweep_is_real_and_not_only_claimed(self):
        """The profile's FIRST axis, read out of `render_config`.

        `render_config`'s normalized branch coerces the profile key to a
        single int by construction, so a scalar under `ann:` with
        sweep-shaped prose beside it would have claimed a measurement the
        profile never takes. The sweep is a `resources.extras` override that
        the TUNED branch reads, and both ends are asserted here -- one value
        normalized, three tuned -- because asserting only the tuned one would
        pass against a profile that swept in both passes and lost the pin.
        """
        from orchestrator import ann_pass
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        profile = load_profile("percona")
        got = {}
        for resource_pass in ("normalized", "tuned"):
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            body = ann_pass.render_config("percona", profile, resources,
                                          resource_pass)
            group, = body["float"]["any"][0]["run_groups"].values()
            got[resource_pass] = group["arg_groups"][0]["efConstruction"]
        assert got["normalized"] == [200], got
        assert len(got["tuned"]) == 3, got
        assert got["tuned"] == [50, 200, 800], got
        # Every value inside the range the server accepts, checked against the
        # overlay module's own bounds rather than against numbers repeated
        # here: out-of-range arrives as "Illegal index construction parameter
        # value: 0" with no mention of which parameter.
        source = open(os.path.join(
            VB_ROOT, "overlay", "ann-benchmarks", "ann_benchmarks",
            "algorithms", "percona", "module.py")).read()
        lo = int(re.search(r"MIN_EF_CONSTRUCTION = (\d+)", source).group(1))
        hi = int(re.search(r"MAX_EF_CONSTRUCTION = (\d+)", source).group(1))
        assert (lo, hi) == (1, 4096)
        for value in got["tuned"]:
            assert lo <= value <= hi, value


# ---------------------------------------------------------------------------
# The estimator: an engine with no measured rate is named, not guessed
# ---------------------------------------------------------------------------

class TestAnUnmeasuredIngestRateIsNamedAndNotModelled:
    """`_effective_rate` ended in `.get(engine, 300)`, which contradicted the
    comment four lines above the table it read from: "an estimate that is
    optimistic by 3x is worse than none". 300 rows/s is 2x the measured
    MariaDB figure and 5.5x the measured AliSQL one, and it reached published
    output -- `webui/control.py` passes `KNOWN_ENGINES` straight in.

    **Percona is no longer the example, and the reason is the point of this
    class.** It moved from *named* to *counted* because a measurement was
    written down, not because a guess was: B1's census of the archived ops
    records gave the M=16 `incremental` rate on three corpora -- 339, 275 and
    427 rows/s from `full-20261015-215006` -- and those went into
    `_MEASURED_ROWS_PER_S` with 275 as the reference rate. The `None` path
    they left is still the one this class guards, so every assertion below
    that needed an unmeasured engine now uses `e2-no-such-engine`, the probe
    `tests/test_config.py` injects for the same reason and with the same
    argument: a guard that requires some real engine to be unmeasured fires
    on the day the last one is measured, which is a guard that fires on a
    better state and teaches the next author to delete it.
    """

    #: An engine nothing could ever have measured, so the `None` branch stays
    #: covered whatever the rate tables gain. Spelled as `tests/test_config.py`
    #: spells it.
    UNMEASURED = "e2-no-such-engine"

    @staticmethod
    def _estimate(engines):
        from orchestrator.estimate import estimate_load_hours
        return estimate_load_hours(
            {"name": "t", "ann": {"m_values": [16]}, "ops": {"m_values": [16]}},
            list(engines), ["glove-100-angular"], ["normalized"],
            ["ann", "ops"])

    def test_an_engine_with_no_rate_returns_none_rather_than_a_number(self):
        from orchestrator.estimate import _effective_rate
        assert _effective_rate(self.UNMEASURED, "glove-100-angular") is None
        # Non-vacuity: a measured engine still gets its number.
        assert _effective_rate("mariadb", "glove-100-angular") == 147.0
        # And Percona is now one of those, at the figure its glove-100 M=16
        # `incremental` record carries: 275.04 rows/s, normalized pass,
        # 1,183,514 rows, `full-20261015-215006`.
        assert _effective_rate("percona", "glove-100-angular") == 275.0

    def test_it_is_named(self):
        e = self._estimate(["mariadb", self.UNMEASURED])
        assert e["engines_without_ingest_rate"] == [self.UNMEASURED]

    def test_it_contributes_nothing_to_the_total(self):
        """The alternative -- a key worth 0.0 -- reads as "costs nothing",
        which is the opposite of what is known about an engine building an
        HNSW graph inside every INSERT. That argument is why the `None` path
        must keep existing, and it is why Percona's entry had to come from a
        record rather than from a plausible number."""
        one = self._estimate(["mariadb"])
        two = self._estimate(["mariadb", self.UNMEASURED])
        assert two["total_hours"] == one["total_hours"]
        assert self.UNMEASURED not in two["per_engine_hours"]

    def test_a_measured_engine_is_counted_and_percona_is_one(self):
        """The inversion. Percona used to be this file's unmeasured engine in
        a test asserting the opposite of every line here."""
        e = self._estimate(["mariadb", "alisql", "pgvector"])
        assert e["engines_without_ingest_rate"] == []
        two = self._estimate(["mariadb", "percona"])
        one = self._estimate(["mariadb"])
        assert two["engines_without_ingest_rate"] == []
        assert two["per_engine_hours"]["percona"] > 0
        assert two["total_hours"] > one["total_hours"]

    def test_the_six_measured_rates_did_not_move(self):
        """The constraint on the repair: another task re-baselined these
        figures, and if the six moved its numbers and this change would
        fight. Percona is the seventh and is asserted beside them rather than
        in a second dict, so ALL OF THIS table is pinned by one literal.

        Only this one. `_MEASURED_ROWS_PER_S` is per `(engine, dataset)` and
        is pinned for percona by the assertion below plus
        `tests/test_estimate.py`'s
        `test_the_three_entries_are_the_records_they_came_from`; its other
        engines' entries are not pinned anywhere by value, which is the state
        this change found and did not widen."""
        from orchestrator import estimate as est_mod
        assert est_mod._INGEST_ROWS_PER_S == {
            "mariadb": 150, "mariadb123": 110, "alisql": 55,
            "pgvector": 3000, "valkey": 40000, "mongodb": 60000,
            "percona": 275}

    def test_the_rate_was_measured_for_percona_and_not_invented(self):
        """It was "no rate was invented for percona", and the distinction it
        drew is the one that still has to hold: an estimate for an unmeasured
        engine is a guess presented as a plan. The three keys below are
        `(dataset, M=16, incremental, 60k/1M-row)` records from one named run,
        and gist-960-euclidean is deliberately absent because it was measured
        at M=32 alone. `tests/test_estimate.py` holds the span assertion that
        a `post`-mode figure fails."""
        from orchestrator import estimate as est_mod
        assert est_mod._INGEST_ROWS_PER_S["percona"] == 275
        assert {k[1]: v for k, v in est_mod._MEASURED_ROWS_PER_S.items()
                if k[0] == "percona"} == {
            "fashion-mnist-784-euclidean": 339,
            "glove-100-angular": 275,
            "sift-128-euclidean": 427,
        }
        assert ("percona", "gist-960-euclidean") not in est_mod._MEASURED_ROWS_PER_S

    def test_a_dataset_with_no_row_count_does_not_make_every_engine_a_gap(self):
        """The care `engines_without_query_rate` takes, taken here too: the
        gap list is keyed on the RATE, not on "contributed no hours". `dev`'s
        only dataset has no row count, and every engine would be listed if
        the test were the latter.

        Percona now demonstrates the point from the other side, which is
        better evidence than it used to be: it has a rate and no row count, so
        it is absent from the gap list while contributing 0.0 hours -- the
        exact pair the list would conflate if it were keyed on hours."""
        from orchestrator.estimate import estimate_load_hours
        e = estimate_load_hours(
            {"name": "t", "ann": {}, "ops": {}},
            ["mariadb", "percona", self.UNMEASURED], ["tiny-16-euclidean"],
            ["normalized"], ["ann", "ops"])
        assert e["engines_without_ingest_rate"] == [self.UNMEASURED]
        assert e["datasets_without_estimate"] == ["tiny-16-euclidean"]
        assert e["per_engine_hours"] == {"mariadb": 0.0, "percona": 0.0}

    def test_the_printed_output_says_which_engines_the_total_covers(self):
        """A reader must not have to reconcile an eight-engine plan against a
        seven-engine total by noticing a name missing from a list.

        It was a seven-engine plan and a six-engine total, and Percona was the
        missing name. Its rate is measured now, so `main` over
        `KNOWN_ENGINES` prints no disclosure at all -- which is the correct
        output and would make this test vacuous. The probe engine restores the
        state the disclosure exists for, and the assertion that `main` alone
        prints nothing is in `tests/test_estimate.py` beside the figures it
        moved."""
        import io
        from contextlib import redirect_stdout
        from orchestrator.config import load_profile
        from orchestrator.cli import KNOWN_ENGINES
        from orchestrator.estimate import print_load_estimate
        profile = load_profile("main")
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_load_estimate(profile,
                                list(KNOWN_ENGINES) + [self.UNMEASURED],
                                list(profile["datasets"]), ["normalized"],
                                ["ann", "ops"])
        out = buf.getvalue()
        # MyVector is the registry's own unmeasured engine now, so the probe is
        # named beside it, in registry order, on the one disclosure line.
        assert (f"No measured ingest rate for myvector, villagesql, "
                f"villagesql84, {self.UNMEASURED}" in _flat(out))
        # The engine that gained a rate is on the per-engine line instead of
        # in the disclosure, which is the whole difference this change makes
        # to what an operator reads.
        assert "percona ~" in out
        _prose("the total covers only the engines with a rate", out)
        # And it says what a reader should CONCLUDE, not merely that a figure
        # is missing -- the attack this catches is a line that announces the
        # gap and drops the consequence.
        _prose("the real cost is higher than the printed total", out)

    def test_the_post_rate_note_follows_the_engines_own_default(
            self, monkeypatch):
        """The note that Percona's `post` ops units are priced at its
        incremental rate fires when the plan runs `post` units. A profile
        that names no modes gets the ENGINE's default, so it must follow
        `capabilities.bulk_vector_index_build`: a literal `["post"]` default
        (found in review) would keep the note after the key was flipped back
        to false, describing `post` units the run no longer builds."""
        import io
        from contextlib import redirect_stdout
        from orchestrator import config as config_mod
        from orchestrator.config import load_profile
        from orchestrator.estimate import print_load_estimate
        profile = load_profile("smoke")
        real = config_mod.load_engine

        def printed(bulk):
            def load(name):
                cfg = real(name)
                if name == "percona":
                    cfg = dict(cfg)
                    cfg["capabilities"] = dict(cfg.get("capabilities") or {},
                                               bulk_vector_index_build=bulk)
                return cfg
            monkeypatch.setattr(config_mod, "load_engine", load)
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_load_estimate(profile, ["percona"],
                                    list(profile["datasets"]),
                                    ["normalized"], ["ann", "ops"])
            return "ops ingest is charged at its measured incremental" in \
                _flat(buf.getvalue())

        honest, mutant = printed(True), printed(False)
        print(f"HONEST (bulk true)  note={honest}")
        print(f"MUTANT (bulk false) note={mutant}")
        assert honest, "the current default is post; the note is owed"
        assert not mutant, ("with the key false the default is incremental "
                            "and no post unit runs")

    def test_the_churn_term_does_not_charge_an_unmeasured_engine_a_duration(
            self):
        """`typical` stays 0.0 and the basis says why, which is what
        `_filtered_search` already does for a missing query rate. A budget
        says what work cannot exceed, not what it will take."""
        from orchestrator.estimate import estimate_load_hours
        profile = {"name": "t", "ops": {"enabled": True,
                                        "workloads": ["churn"],
                                        "m_values": [16],
                                        "subset_rows": 20000,
                                        "churn_fractions": [0.1],
                                        "churn_budget_s": 1800,
                                        "max_queries": 200}}
        e = estimate_load_hours(profile, [self.UNMEASURED],
                                ["glove-100-angular"], ["normalized"], ["ops"])
        churn = [t for t in e["query_terms"] if t["name"] == "churn re-insert"]
        assert len(churn) == 1
        # Seconds, not hours: 2,000 rows is a few seconds of work even at
        # MariaDB's measured 147 rows/s, so `typical_hours` rounds to 0.0 for
        # BOTH engines and a comparison on it would be the identical
        # honest/mutant pair that signals a dead probe.
        assert churn[0]["typical_seconds"] == 0.0
        assert churn[0]["worst_seconds"] > 0
        assert "at a rate nothing here has measured" in churn[0]["basis"]
        # Percona was the engine this test used, and it now takes the other
        # branch: a measured load rate, so the re-insert gets a modelled
        # duration with the rate printed in its basis. The caveat beside that
        # rate is still the load rate's -- re-insert can be far slower.
        p = estimate_load_hours(profile, ["percona"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        pchurn = [t for t in p["query_terms"]
                  if t["name"] == "churn re-insert"][0]
        assert pchurn["typical_seconds"] > 0
        assert "at ~275 rows/s" in pchurn["basis"]
        # Non-vacuity: a measured engine gets a duration and a rate.
        m = estimate_load_hours(profile, ["mariadb"], ["glove-100-angular"],
                                ["normalized"], ["ops"])
        mchurn = [t for t in m["query_terms"]
                  if t["name"] == "churn re-insert"][0]
        assert mchurn["typical_seconds"] > 0
        assert "rows/s" in mchurn["basis"]
        assert "at a rate nothing here has measured" not in mchurn["basis"]

    def test_the_repair_path_disclosure_names_both_engines_that_have_one(self):
        """It said "pgvector of the six", which was a count and an engine list
        at once and became wrong in both halves. Percona's repair rebuilds the
        whole graph, so it is the larger of the two unmeasured repairs, not a
        footnote to pgvector's VACUUM."""
        from orchestrator.estimate import estimate_load_hours
        from orchestrator.config import load_profile
        from orchestrator.cli import KNOWN_ENGINES
        profile = load_profile("main")
        e = estimate_load_hours(profile, list(KNOWN_ENGINES),
                                list(profile["datasets"]), ["normalized"],
                                ["ann", "ops"])
        repair = [n for n in e["query_costs_not_modelled"]
                  if n.startswith("churn repair path:")]
        assert len(repair) == 1, repair
        flat = " ".join(repair[0].split())
        assert "pgvector" in flat and "Percona Server" in flat
        assert "of the six" not in flat


# ---------------------------------------------------------------------------
# §4.7 -- the scripts
# ---------------------------------------------------------------------------

class TestTheBuildScripts:

    @pytest.fixture(scope="class")
    def build_images(self):
        return open(os.path.join(VB_ROOT, "scripts", "build-images.sh")).read()

    @pytest.fixture(scope="class")
    def prepare(self):
        return open(os.path.join(VB_ROOT, "scripts",
                                 "prepare-sources.sh")).read()

    @pytest.fixture(scope="class")
    def lib(self):
        return open(os.path.join(VB_ROOT, "scripts", "lib.sh")).read()

    def test_build_images_has_both_cases_and_not_only_one(self, build_images):
        """There are TWO dispatches: the outer one on `$ENGINE` that decides
        whether the engine can be built at all, and the inner one on the
        alias base that supplies the tag build-arg. Adding only the inner one
        fails with "unknown engine" before the build-arg is ever read."""
        outer = build_images.rsplit('case "$ENGINE" in', 1)[1]
        assert "percona)" in outer
        inner = build_images.split('case "$base" in', 1)[1].split("esac", 1)[0]
        assert 'percona)' in inner
        assert "PERCONA_TAG" in inner

    def test_the_dockerfile_declares_the_arg_the_script_passes(self):
        dockerfile = open(os.path.join(VB_ROOT, "docker", "percona",
                                       "Dockerfile")).read()
        assert "ARG PERCONA_TAG" in dockerfile

    def test_prepare_sources_can_build_a_percona_context(self, prepare):
        assert "prepare_percona()" in prepare
        dispatch = prepare.rsplit('case "$ENGINE" in', 1)[1]
        assert "percona)" in dispatch

    def test_it_archives_rather_than_clones_with_submodules(self, prepare):
        """`git archive` of the pinned commit, and no submodule init -- all
        four are switched off by cmake flags instead."""
        body = prepare.split("prepare_percona()", 1)[1].split("\n}", 1)[0]
        assert "git archive" in body
        assert "submodule" not in body

    def test_it_verifies_the_commit_it_resolved_is_the_pinned_one(self, prepare):
        """A SHA that is passed along without being checked is a string, not a
        pin."""
        body = prepare.split("prepare_percona()", 1)[1].split("\n}", 1)[0]
        assert "rev-parse" in body
        assert 'resolved" == "$sha' in body

    def test_the_vendor_repo_guard_covers_the_percona_checkout(self, lib):
        """`assert_not_vendor_repo` loops over an explicit list. A repo absent
        from it gets NO protection, while the ones beside it make the rule
        look enforced. CLAUDE.md names these as read-only inputs."""
        assert "VB_REPO_PERCONA=" in lib
        loop = lib.split("assert_not_vendor_repo()", 1)[1].split("\n}", 1)[0]
        for var in ("VB_REPO_MARIADB", "VB_REPO_ALISQL", "VB_REPO_ANNB",
                    "VB_REPO_PERCONA"):
            assert var in loop, var

    def test_every_exported_vendor_repo_is_in_the_guard_loop(self, lib):
        """The general form, so the eighth engine fails here rather than
        shipping an unprotected checkout."""
        exported = set(re.findall(r"^export (VB_REPO_\w+)=", lib, re.M))
        loop = lib.split("assert_not_vendor_repo()", 1)[1].split("\n}", 1)[0]
        in_loop = set(re.findall(r"\$(VB_REPO_\w+)", loop))
        assert exported, "no vendor repos are exported; this test is stale"
        assert exported == in_loop, exported ^ in_loop


# ---------------------------------------------------------------------------
# §4.6 -- the profiles
# ---------------------------------------------------------------------------

class TestTheProfiles:

    def test_smoke_no_longer_claims_six(self):
        text = open(os.path.join(VB_ROOT, "config", "profiles",
                                 "smoke.yml")).read()
        _prose("the engine count is eight", text)

    def test_smoke_names_no_engines_so_percona_joins_automatically(self):
        from orchestrator.config import load_profile
        assert "engines" not in load_profile("smoke")

    def test_the_percona_profile_runs_only_percona(self):
        from orchestrator.config import load_profile
        assert load_profile("percona")["engines"] == ["percona"]

    def test_every_profile_still_loads(self):
        """A new YAML file in config/profiles/ is picked up by name, so a
        malformed one fails only when someone runs it."""
        from orchestrator.config import load_profile
        directory = os.path.join(VB_ROOT, "config", "profiles")
        names = sorted(f[:-4] for f in os.listdir(directory)
                       if f.endswith(".yml"))
        assert "percona" in names
        for name in names:
            assert load_profile(name)["name"] == name


class TestEveryShippedProfileRenders:
    """`render_config` called against every profile in the repo, for every
    engine that profile would run, in both resource passes.

    **This class exists because its absence let a shipped profile trip a
    refusal added in the same change.** `config/profiles/percona.yml` set
    `ann.percona_build_modes: [post, incremental]`, which
    `ann_pass.render_config` policed -- so the Percona-only profile's ann
    phase raised, and nothing noticed: every other `render_config` test in
    this repo either uses the `quick` profile or builds a synthetic dict, and
    `test_every_profile_still_loads` only parses the YAML.

    Both call sites are covered by this, and the second is the louder one:
    `cli._run_unit` catches and marks a unit failed with a traceback, while
    `cli.cmd_render` has NO handler and loops over KNOWN_ENGINES -- so a
    render of a one-engine profile tracebacks out of the process.
    """

    @staticmethod
    def _profiles():
        directory = os.path.join(VB_ROOT, "config", "profiles")
        return sorted(f[:-4] for f in os.listdir(directory)
                      if f.endswith(".yml"))

    def test_the_profile_list_is_not_empty(self):
        """Non-vacuity: an empty listing would make the parametrized tests
        below collect nothing and the suite report success."""
        names = self._profiles()
        assert len(names) >= 9, names
        assert "percona" in names

    @pytest.mark.parametrize("resource_pass", ["normalized", "tuned"])
    def test_every_profile_renders_for_every_engine_it_runs(self,
                                                             resource_pass):
        from orchestrator import ann_pass
        from orchestrator.cli import KNOWN_ENGINES
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        checked = 0
        for name in self._profiles():
            profile = load_profile(name)
            # `engines:` when the profile names them, every known engine when
            # it does not -- which is what `config.profile_engines` and
            # `cli.cmd_run` resolve.
            #
            # **NOT what `cmd_render` does, though this comment said so until
            # the key became load-bearing.** `cmd_render` loops `KNOWN_ENGINES`
            # whatever the profile declares, deliberately, and
            # `test_cmd_render_renders_every_known_engine_when_told_none` below
            # asserts the literal expression that makes it so -- which is why a
            # profile naming one engine can still be asked to render seven, and
            # why a render failure for any of them is a whole-command failure.
            # One file holding a comment and a test that contradicted each
            # other about one function left a reader to conclude the pin was
            # the stale half.
            engines = profile.get("engines") or list(KNOWN_ENGINES)
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            for engine in engines:
                body = ann_pass.render_config(engine, profile, resources,
                                              resource_pass)
                groups = body["float"]["any"][0]["run_groups"]
                assert groups, f"{name}/{engine}: no run groups"
                for group in groups.values():
                    assert group["arg_groups"], f"{name}/{engine}"
                checked += 1
        assert checked >= 40, checked

    def test_cmd_render_renders_every_known_engine_when_told_none(self):
        """The property that turns a one-engine profile's render error into a
        whole-command failure, read out of `cli.py` rather than assumed."""
        source = open(os.path.join(VB_ROOT, "orchestrator", "cli.py")).read()
        body = source.split("def cmd_render", 1)[1].split("\ndef ", 1)[0]
        assert "args.engines.split(\",\") if args.engines else KNOWN_ENGINES" \
            in body
        # ...and it does not guard the call, which is why the profile had to
        # be the thing that changed. Stated so that a future `try` here is a
        # deliberate decision rather than a silent one.
        assert "try:" not in body

    def test_the_percona_profile_renders_both_of_its_axes(self):
        """The regression itself, named: this is the call that raised."""
        from orchestrator import ann_pass
        from orchestrator.config import (load_profile, load_resources,
                                         merge_resource_overrides)
        from orchestrator import estimate as est_mod
        profile = load_profile("percona")
        for resource_pass in ("normalized", "tuned"):
            resources = merge_resource_overrides(
                load_resources(resource_pass), profile)
            body = ann_pass.render_config("percona", profile, resources,
                                          resource_pass)
            assert sorted(body["float"]["any"][0]["run_groups"]) == ["innodb"]
        # And the ops axis is two units again. It was one while `post` killed
        # the server at the old pin and the profile asked for `incremental`
        # alone; the capability in `config/engines/percona.yml` carries the
        # evidence for both the narrowing and the restoration.
        assert est_mod._ops_units(profile, "percona") == 2


# ---------------------------------------------------------------------------
# Coverage this addition would otherwise have lost
# ---------------------------------------------------------------------------

class TestTheInvariantsKeyedOnHardCodedEngineTuples:
    """Three existing tests iterate a hand-written engine list rather than the
    registry, so a seventh engine is silently NOT covered by them -- they go
    on passing while checking six of seven. Those tests live in suites this
    task may not add to, so the missing coverage is supplied here, derived
    from the registry so the eighth engine cannot slip through the same way.
    """

    @staticmethod
    def _resolved(engine, resource_pass):
        from orchestrator.config import load_resources, resolve_resources
        info = type("I", (), {})()
        info.cpu = type("C", (), {"arch": "x86_64", "hybrid": False,
                                  "performance_cpus": list(range(40)),
                                  "efficiency_cpus": [], "logical_cpus": 80,
                                  "physical_cores": 40, "threads_per_core": 2,
                                  "model": "Xeon"})()
        info.total_ram_bytes = int(192 * 1024 ** 3)
        return resolve_resources(load_resources(resource_pass), engine, info)

    def test_the_ann_fingerprint_is_still_engine_invariant(self):
        """`test_config.py`'s version iterates a six-name tuple.

        The fingerprint keys the shared `results/annb/` tree by the resource
        pass's declared knobs and is deliberately engine-INVARIANT: making it
        engine-specific fragments the tree and produces a seven-engine recall
        chart containing one engine. It has broken twice before, both times
        by hashing something that differs per engine.
        """
        from orchestrator import engines as engines_mod
        from orchestrator.ann_pass import ann_fingerprint
        names = sorted(engines_mod.registry(refresh=True))
        assert "percona" in names and len(names) >= 7, names
        for resource_pass in ("normalized", "tuned"):
            prints = {e: ann_fingerprint(self._resolved(e, resource_pass))
                      for e in names}
            assert len(set(prints.values())) == 1, (resource_pass, prints)

    def test_the_two_passes_still_do_not_collide_for_the_new_engine(self):
        from orchestrator.ann_pass import ann_fingerprint
        assert (ann_fingerprint(self._resolved("percona", "normalized"))
                != ann_fingerprint(self._resolved("percona", "tuned")))

    def test_percona_resolves_resources_at_all(self):
        """An engine the resource layer cannot resolve dies after the image is
        built and the server is up."""
        resolved = self._resolved("percona", "normalized")
        assert resolved.buffer_bytes > 0
        assert resolved.server_memory_bytes > 0

    def test_the_overlay_module_reports_nothing_hdf5_cannot_store(self):
        """`test_config.py`'s version iterates a six-name tuple too.

        ann-benchmarks' `store_results` writes `get_additional()` into HDF5
        attributes, and `None`, a list or a dict raises `TypeError: Object
        dtype dtype('O') has no native HDF5 equivalent` -- which ann-benchmarks
        catches per worker and exits ZERO for, so the phase is reported
        complete and writes no results. Percona Search failed that way four
        times across two datasets while its measurements ran perfectly.
        """
        path = os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "algorithms", "percona",
                            "module.py")
        source = open(path).read()
        assert "def get_additional" in source           # needle findable
        block = source.split("def get_additional", 1)[1].split("\n    def ",
                                                                1)[0]
        offenders = re.findall(r'"(\w+)":\s*(None\b|\[|\{)', block)
        assert not offenders, offenders


# ---------------------------------------------------------------------------
# The pin is one value in every file that spells one
# ---------------------------------------------------------------------------
#
# `tests/test_e1_percona_design.py`'s
# `test_the_pinned_sha_is_the_same_in_every_file_that_names_one` collects every
# 40-hex string in docs/09 and the overlay module and asserts the set is that
# file's own `PINNED_SHA`. Those are two of the eight files in this repository
# that spell a full commit SHA, and the two constants named `PINNED_SHA` --
# that file's and this one's -- are independent literals that nothing
# compares. So the chain that is supposed to make a re-pin all-or-nothing had
# a break in the middle: the engine config was tied to THIS file's constant,
# the document to that file's, and `docker/percona/Dockerfile`'s SHA note and
# `tests/test_e4_percona_build_flags.py`'s provenance paragraph were tied to
# nothing at all. Both of those spell the pin inside a comment that reasons
# from it, so a half-finished re-pin leaves the config naming one source while
# the cmake reasoning cites another -- which is worse than no re-pin, because
# it reads as deliberate.
#
# This is the whole-tree form of that guard. It is what turns the re-pin's own
# acceptance criterion -- `grep -rn <the old sha>` over the tree returns
# nothing -- into an assertion rather than a step somebody remembers to run.

#: Directories never walked. The same list and the same reasons as
#: `tests/test_e16_citation_ratchet.py`'s `SKIP_DIRS`: `.git` and
#: `__pycache__` hold no source; `.claude` is gitignored and holds other
#: agents' git worktrees, so walking it would scan whole second copies of this
#: repository; and the rest are generated and gitignored per CLAUDE.md, so a
#: scan that read them would differ between a fresh checkout and a machine
#: that had run a benchmark. That last one is not a tidiness point here: the
#: vendor trees exported under `sources/` spell a 40-hex SHA on nearly every
#: page, so including them would make this guard fail on exactly the machine
#: that had done the build it exists to protect.
SHA_SCAN_SKIP_DIRS = ("__pycache__", ".git", ".pytest_cache", ".claude",
                      ".cache", ".venv", "venv", "sources", "work", "results",
                      "datasets", "state")

#: Suffixes read. The last entry is a whole filename rather than an extension,
#: because the Dockerfile is where two of the pin's three unguarded copies
#: lived and `str.endswith` takes both forms from one tuple.
SHA_SCAN_SUFFIXES = (".py", ".yml", ".yaml", ".md", ".sh", ".sql", ".js",
                     ".txt", ".json", "Dockerfile")

#: This file, and the plan document, by relative path. Both are named
#: because both legitimately spell commits that are not the pin, and a guard
#: whose own data sits inside its own scan has to say so rather than quietly
#: skip itself -- `tests/test_e16_citation_ratchet.py` makes the same
#: declaration for the same reason, in
#: `test_this_file_makes_none_of_the_citations_it_counts`.
SELF_REL = "tests/test_e2_percona_wiring.py"
PLAN_DOC = "docs/12-percona-feedback-plan.md"

#: `orchestrator/estimate.py`, which dates its measured ingest rates to the
#: binary that produced them. Named here rather than edited: the sentence is
#: a fact about seventeen archived records and not a copy of the pin, and the
#: file belongs to another unit.
ESTIMATOR = "orchestrator/estimate.py"

#: Commits other than the pin that this tree may name, as
#: `{sha: (the files that may name it, why)}`.
#:
#: KEYED BY SHA AND NOT BY FILE, which is the correction a review made and
#: the principle this comment already stated two paragraphs up. The first
#: version exempted `docs/12-percona-feedback-plan.md` wholesale, so a SECOND
#: stale commit landing in that file would have passed -- an exemption
#: inferred from a filename rather than named.
#:
#: THIS FILE IS IN EVERY ENTRY'S FILE LIST, AND THAT IS NOT A SELF-EXEMPTION.
#: A guard that forbids a commit has to spell it, so the forbidding and the
#: forbidden look identical to a grep. The narrow check below --
#: `test_this_file_holds_nothing_but_the_commits_it_declares` -- is what
#: stops that from being a hiding place: every commit in this file must be
#: the pin or a declared entry, so a stale copy smuggled in here fails even
#: though the file is listed. The first version of this guard had no such
#: declaration at all and simply failed against its own data, which is how
#: the omission was found.
COMMITS_ALLOWED_BESIDE_THE_PIN = {
    "1fc5dadc7265e0080e70320edcf238c3743d2b7a": (
        (PLAN_DOC, SELF_REL),
        "the pre-rebase head of PR #6278's branch, and so the twin of the "
        "pin: it is the plan's evidence that the old branch tip is not an "
        "ancestor of the merged series. docs/12's figures describe the tree "
        "as it was when the plan was written and are deliberately not "
        "maintained -- the plan's own phase 0 says so."),
    "02a8b80c70b0ccf6c9035d8a2307e7fa988cc304": (
        (PLAN_DOC, SELF_REL, ESTIMATOR),
        "the superseded pin: the `vector-mvp` tip of 2026-10-13, left behind "
        "when the vector MVP was merged into the `9.7` series. docs/12 is "
        "the document that argues for that re-pin, so naming the pin it "
        "moved away from is its subject rather than a stale copy.\n"
        "\n"
        "`orchestrator/estimate.py` is the third entry and it was found by "
        "this guard rather than by reading: the estimator's rates landed on "
        "master while the old pin was still live, and its comment says `All "
        "seventeen records carry engine_version 9.7.1-1-vector-bench at "
        "engine_tag 02a8b80c. These are therefore rates for the OLD pin`. "
        "That is a fact about which binary measured seventeen archived "
        "records, not a copy of the pin, and the sentence is the reason the "
        "first run at the new pin has something explicit to supersede. The "
        "file belongs to another unit, so it is declared here and not "
        "edited -- and the check below means that if that sentence is ever "
        "removed, this allowance fails rather than lingering."),
    "384d9df7ab8a2afee9c8f99704e5a7ea90d32730": (
        ("config/engines/myvector.yml", "tests/test_e23_myvector_engine.py",
         SELF_REL),
        "a different engine's pin, not a copy of Percona's: MyVector "
        "release v1.26.9 (askdba/myvector), which config/engines/myvector.yml "
        "carries as its own `source.tag` and its own test pins. This guard "
        "scans the whole tree for 40-hex commits, so a second engine pinned "
        "by SHA has to be declared here or it reads as a half-finished "
        "Percona re-pin."),
    "93ed00873dd65f64c0788c80576d5cf3fba39831": (
        ("config/engines/villagesql.yml", "tests/test_e26_villagesql_engine.py",
         "config/engines/villagesql84.yml",
         "tests/test_e30_villagesql84_engine.py",
         "docs/04-engine-notes.md", SELF_REL),
        "a different engine's pin, not a copy of Percona's: vsql-vector-pk "
        "release 0.0.3 at 93ed008 (villagesql/vsql-vector-pk), which "
        "config/engines/villagesql.yml and config/engines/villagesql84.yml "
        "carry as their own `source.tag`."),
    "533ab4b58d2744b79bb45e41f44fa928f190a655": (
        ("config/engines/villagesql.yml", "tests/test_e26_villagesql_engine.py",
         "docs/04-engine-notes.md", SELF_REL),
        "VillageSQL's publish/mysql-9.7_0.0.7 tag, the server tree "
        "vsql-vector-pk is compiled against. Pinned by commit as well as by "
        "tag so prepare_villagesql can refuse a moved tag."),
    "689c74a9f88c91ae035f57e0ae8915a168274112": (
        ("config/engines/villagesql84.yml",
         "tests/test_e30_villagesql84_engine.py",
         "docs/04-engine-notes.md", SELF_REL),
        "VillageSQL's publish/mysql-8.4_0.0.7 tag (the same commit as "
        "release/0.0.7), the MySQL 8.4 server tree villagesql84 compiles the "
        "same vsql-vector-pk commit against. Pinned by commit as well as by "
        "tag so prepare_villagesql can refuse a moved tag."),
    "008e09c2834b98143a8c067d4d225c90953050cf": (
        ("config/engines/myvector.yml", "tests/test_e24_myvector_disclosures.py",
         SELF_REL),
        "MySQL's mysql-9.7.2 tag, the server tree MyVector is compiled into. "
        "Pinned by commit as well as by tag so prepare_myvector can refuse a "
        "moved tag, the way it refuses a moved plugin commit."),
}

#: Of those, the ones that were once this engine's `source.tag`.
#:
#: THE RE-PIN'S ACCEPTANCE CRITERION IS A GREP FOR THE SUPERSEDED PIN, AND
#: `_FORTY_HEX` CANNOT SEE ONE. CLAUDE.md's engine table and this engine's own
#: config comment write a pin as EIGHT characters, so reverting either of them
#: to the old abbreviation left the 40-hex scan green while the acceptance
#: grep failed -- a half-finished re-pin in a file the re-pin itself edited.
#: So the criterion gets its own assertion below, over abbreviations of seven
#: characters and up and case-insensitively: `git log --format=%H` is lower
#: case, but nothing stops a document shouting one.
#:
#: The files allowed to name each one come from the table above, so the two
#: assertions cannot disagree about where a commit may live.
SUPERSEDED_PINS = ("02a8b80c70b0ccf6c9035d8a2307e7fa988cc304",)

#: A commit as this repository writes one: seven to forty hex characters,
#: token-bounded, holding BOTH a digit and a letter.
#:
#: The digit-and-letter requirement is what makes a 7-character scan usable at
#: all. Without it the pattern matches ordinary English -- `effaced`, `decade`,
#: `deadbeef` are all hex -- and a guard that fires on prose is a guard the
#: next author deletes. With it, the tree yields 40 commit-shaped tokens, most
#: of them this repository's OWN commits cited in test docstrings, which is
#: why the assertions below are scoped to the superseded pin and to near
#: misses of the current one rather than demanding an allow-list of all 40.
_HEXISH = re.compile(r"(?<![0-9A-Za-z])([0-9a-fA-F]{7,40})(?![0-9A-Za-z])")

#: Sites the scan must actually reach, so that a walk which quietly stopped
#: finding files cannot pass every assertion it makes. One per shape the
#: suffix list has to handle: a YAML config, an extensionless Dockerfile, a
#: Markdown document, a Python module. The two test files that spell the pin
#: are deliberately absent -- a non-vacuity check naming only the guards
#: proves nothing about the tree they guard.
SHA_SCAN_WITNESSES = (
    "config/engines/percona.yml",
    "docker/percona/Dockerfile",
    "docs/09-adding-percona-server.md",
    "overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/module.py",
)

_FORTY_HEX = re.compile(r"\b[0-9a-f]{40}\b")


def _forty_hex_shas(root):
    """`{relative path: {every 40-hex string in it}}`, under `root`.

    Taken as a parameter rather than off `VB_ROOT` so the planted tree below
    exercises this walk and this regex, and not a copy of them.
    """
    found = {}
    for where, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SHA_SCAN_SKIP_DIRS]
        for name in files:
            if not name.endswith(SHA_SCAN_SUFFIXES):
                continue
            try:
                text = open(os.path.join(where, name), encoding="utf-8",
                            errors="replace").read()
            except OSError:
                continue
            shas = set(_FORTY_HEX.findall(text))
            if shas:
                rel = os.path.relpath(os.path.join(where, name), root)
                found[rel.replace(os.sep, "/")] = shas
    return found


def _hexish_tokens(root):
    """`{relative path: {every commit-shaped token in it, lower-cased}}`.

    Same walk and same suffix list as `_forty_hex_shas`; a different pattern,
    and lower-cased on the way out so an abbreviation shouted in prose is
    still recognised.
    """
    found = {}
    for where, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SHA_SCAN_SKIP_DIRS]
        for name in files:
            if not name.endswith(SHA_SCAN_SUFFIXES):
                continue
            try:
                text = open(os.path.join(where, name), encoding="utf-8",
                            errors="replace").read()
            except OSError:
                continue
            tokens = {
                tok.lower() for tok in _HEXISH.findall(text)
                if re.search(r"[0-9]", tok) and re.search(r"[a-fA-F]", tok)}
            if tokens:
                rel = os.path.relpath(os.path.join(where, name), root)
                found[rel.replace(os.sep, "/")] = tokens
    return found


class TestThePinIsOneValueInEveryFileThatSpellsOne:

    def test_every_forty_hex_sha_in_the_tree_is_the_pin(self):
        found = _forty_hex_shas(VB_ROOT)
        wrong = {}
        for rel, shas in sorted(found.items()):
            bad = sorted(
                s for s in shas
                if s != PINNED_SHA
                and rel not in COMMITS_ALLOWED_BESIDE_THE_PIN.get(
                    s, ((), ""))[0])
            if bad:
                wrong[rel] = bad
        assert not wrong, (
            "these files name a 40-hex commit that is not the pin, so the "
            "re-pin is half-finished; a pin that differs between two files "
            "is not a pin\n  "
            + "\n  ".join(f"{rel}: {', '.join(shas)}"
                          for rel, shas in wrong.items()))

    def test_the_scan_reaches_the_sites_nothing_else_pins(self):
        """Non-vacuity, and the reason it is four paths and not a count: a
        count goes stale the next time a file stops naming the pin, and then
        the repair is to lower the count, which is how a floor becomes a
        comment."""
        found = _forty_hex_shas(VB_ROOT)
        missing = [rel for rel in SHA_SCAN_WITNESSES if rel not in found]
        assert not missing, (
            "the scan no longer sees a SHA in these files, so it is not "
            "checking what it claims to check -- either the suffix list "
            "stopped matching them or the walk stopped reaching them: "
            + ", ".join(missing))

    def test_the_scan_sees_a_planted_sha_and_skips_a_generated_tree(self,
                                                                   tmp_path):
        """The walk, the suffix list and the regex proved end to end on a tree
        built for the purpose, because one guard in this series shipped green
        while scanning zero files.

        The last two plants assert the LIMITS rather than the behaviour: a
        generated tree is skipped even though it holds the needle, and a
        suffix outside the list is not read. Both are deliberate, and a
        reader of this file is entitled to know which omissions were chosen.
        """
        stale = "0" * 39 + "1"
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "engines.yml").write_text(f"tag: {stale}\n")
        (tmp_path / "Dockerfile").write_text(f"# built from {stale}\n")
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "vendor.md").write_text(stale + "\n")
        (tmp_path / "notes.rst").write_text(stale + "\n")
        found = _forty_hex_shas(str(tmp_path))
        assert found == {"config/engines.yml": {stale},
                         "Dockerfile": {stale}}, found

    def test_every_allowed_commit_is_still_where_it_says_it_is(self):
        """An allowance for a commit that is no longer in the file it names is
        pure allowance: it would wave through the next stale copy under the
        same SHA. Same shape as `tests/test_e16_citation_ratchet.py`'s check
        that no ceiling is slack by more than its file holds.
        """
        found = _hexish_tokens(VB_ROOT)
        assert COMMITS_ALLOWED_BESIDE_THE_PIN, "the allow-set is empty"
        for sha, (rels, reason) in sorted(
                COMMITS_ALLOWED_BESIDE_THE_PIN.items()):
            assert reason.strip(), sha
            assert sha != PINNED_SHA, sha
            assert rels, sha
            for rel in rels:
                assert os.path.isfile(os.path.join(VB_ROOT, rel)), rel
                # At ANY abbreviation, which is the whole point: docs/12
                # writes the superseded pin as eight characters and the full
                # forty never appears there, so a 40-hex check here declared
                # that allowance dead when it is live.
                assert any(sha.startswith(tok)
                           for tok in found.get(rel, set())), (
                    f"{sha} is no longer named in {rel}, at any "
                    f"abbreviation, so that half of its allowance is doing "
                    f"nothing but waiting to be abused -- remove it")

    def test_this_file_holds_nothing_but_the_commits_it_declares(self):
        """The narrow check that keeps this file's own listing from being a
        hiding place.

        This guard has to SPELL the commits it forbids, so to a grep the
        forbidding and the forbidden are the same string, and the file must
        therefore appear in its own allow-list. What stops that being an
        exemption is this: the only commits allowed in here are the pin and
        the declared entries, so a stale copy smuggled into this file fails
        even though the file is listed. `tests/test_e16_citation_ratchet.py`
        makes exactly this move for exactly this reason.
        """
        declared = {PINNED_SHA} | set(COMMITS_ALLOWED_BESIDE_THE_PIN)
        found = _forty_hex_shas(VB_ROOT).get(SELF_REL, set())
        assert found, f"{SELF_REL} names no commit at all; the scan is blind"
        assert found <= declared, sorted(found - declared)
        tokens = _hexish_tokens(VB_ROOT).get(SELF_REL, set())
        undeclared = sorted(
            tok for tok in tokens
            if not any(sha.startswith(tok) for sha in declared))
        assert not undeclared, (
            f"{SELF_REL} names commit-shaped tokens it does not declare, so "
            f"its place in its own allow-list is hiding them: {undeclared}")

    def test_no_abbreviation_of_a_superseded_pin_survives(self):
        """The re-pin's acceptance criterion, as an assertion.

        This is the one the 40-hex scan cannot make: the pin appears as eight
        characters in CLAUDE.md's engine table and in the engine config's own
        comment, and reverting either to the old abbreviation leaves
        `test_every_forty_hex_sha_in_the_tree_is_the_pin` green because
        neither file names a 40-hex string at all.
        """
        assert SUPERSEDED_PINS, "nothing is recorded as superseded"
        found = _hexish_tokens(VB_ROOT)
        stale = []
        for sha in sorted(SUPERSEDED_PINS):
            assert len(sha) == 40 and re.fullmatch(r"[0-9a-f]{40}", sha), sha
            allowed = COMMITS_ALLOWED_BESIDE_THE_PIN[sha][0]
            for rel, tokens in sorted(found.items()):
                if rel in allowed:
                    continue
                for token in sorted(tokens):
                    if sha.startswith(token):
                        stale.append(f"{rel}: {token} (abbreviates {sha})")
        assert not stale, (
            "a superseded pin is still named outside the one file allowed to "
            "name it, so the re-pin is half-finished:\n  "
            + "\n  ".join(stale))

    def test_a_near_miss_of_the_pin_is_not_mistaken_for_it(self):
        """A mistyped abbreviation is the failure the two assertions above
        both miss: it is not 40 hex and it is not the superseded pin.

        Scoped to tokens sharing the pin's first three characters, which is
        narrow enough to cost nothing -- no other commit-shaped token in the
        tree begins with them -- and wide enough to catch a transposed or
        truncated digit in a hand-copied pin.
        """
        found = _hexish_tokens(VB_ROOT)
        wrong = []
        for rel, tokens in sorted(found.items()):
            for token in sorted(tokens):
                if token[:3] == PINNED_SHA[:3] and not PINNED_SHA.startswith(token):
                    wrong.append(f"{rel}: {token}")
        assert not wrong, (
            "these look like the pin and are not it, so they are a typo in a "
            f"hand-copied abbreviation of {PINNED_SHA}:\n  "
            + "\n  ".join(wrong))


# ---------------------------------------------------------------------------
# The ann results tree is reused across a re-pin, and three files say so
# ---------------------------------------------------------------------------
#
# `orchestrator/ann_pass.py:ann_fingerprint` is DELIBERATELY engine-invariant:
# it hashes `ANN_MEASUREMENT_VERSION`, the server memory, the pass signature,
# the build threads and the CPU count, and no engine name and no source tag.
# That is the right design -- making it engine-specific fragments the tree and
# produces a seven-engine recall chart with one engine in it -- and it means a
# re-pin does not move `results/annb/<pass>/<fingerprint>/`. ann-benchmarks
# skips a configuration that already has a result file and reports that as
# success, so after a re-pin every Percona recall cell on disk is reused, the
# new binary's recall is never measured, and the report publishes the old
# binary's numbers under the new pin's `engine_version`.
#
# Two things notice, and neither prevents it. `ann_pass.run_engine` prints
# "already-computed configurations will be skipped" before the run, and
# `report/generate.py`'s `stale_ann` bucket compares each record's
# `extra.source_mtime` against the run's start and lists the cell in the
# Validity section afterwards. The remedy is to MOVE THE ENGINE'S SUBTREE of
# the ann results tree aside before the first run at a new pin:
# `_count_results` counts result files under directories named for the
# engine, so it returns 0 and ann-benchmarks measures the curve again. NOT
# `--force`: its help text offers exactly this re-run, but the same flag is
# the escape hatch on the free-disk refusal in `orchestrator/cli.py`, so
# passing it disables the one refusal a core-dump campaign most needs -- and
# for two rounds this file asserted `--force` as the remedy. NOT a fresh run
# directory: `annb_results` is `results/annb`, a sibling of the run
# directory, so a new `--run-id` leaves every recall cell where it was. And
# NOT a bump of `ANN_MEASUREMENT_VERSION`, which would move every pass's tree
# and re-measure every recall point for all seven engines and both resource
# passes.
#
# So the requirement is written in the three places an operator or an author
# actually passes through on the way to that run, and asserted here, because
# a requirement that lives only in a merged pull request body is a
# requirement nobody will read at the moment it matters.

#: Where the move-the-subtree-aside requirement is written, and why each site.
ANN_SUBTREE_SITES = {
    "config/engines/percona.yml":
        "beside the pin itself: moving `source.tag` is the act that creates "
        "the hazard, so the warning belongs in the diff that creates it",
    "CLAUDE.md":
        "in the invariant that already explains why the fingerprint is "
        "engine-invariant, which is the bullet a reader lands on when they "
        "ask why the tree is shared",
    "docs/09-adding-percona-server.md":
        "in the list of what the smoke run has to survive, which is the "
        "document the first post-re-pin run is planned from",
}


#: A line that separates two paragraphs: empty, or carrying nothing but a
#: comment marker. The second half is what a YAML or shell comment block uses
#: for a paragraph break, and leaving it out is what let the whole 77-line
#: `source:` comment in `config/engines/percona.yml` count as ONE block.
_PARAGRAPH_BREAK = re.compile(r"^[ \t]*(?:#|--|//)?[ \t]*$")

#: The start of a list item, bulleted or numbered. `5. ` is not a `-` bullet,
#: and leaving numbers out is what let all five items of docs/09 §5 count as
#: one 34-line block.
_LIST_ITEM = re.compile(r"\n(?=[ \t]{0,4}(?:[-*][ \t]|\d{1,2}[.)][ \t]))")


def _paragraphs(text):
    """One block per paragraph: blank-line separated, comment markers and
    list items understood.

    A block that is bigger than the claim is a smuggling hole, because the
    three-part conjunction below can then be satisfied by three unrelated
    sentences while the warning itself is gone. THIS IS NOT HYPOTHETICAL AND
    THE FIRST VERSION HAD IT: that version split on blank lines and on `-`
    bullets only, so a `  #` line was not a break and `5. ` was not an item,
    and the measured block sizes were 77 lines for the engine config's
    `source:` comment and 34 for docs/09 §5 against 8 and 10 for CLAUDE.md.
    A review then deleted the whole hazard paragraph from the config, left
    the bare line `# Rebuild, then run with --force.`, added the word
    "cached" to an unrelated comment line in the same block, and the guard
    stayed GREEN. With both breaks understood, no site's `--force` block
    exceeds seventeen lines and that deletion is red.
    """
    blocks, current = [], []
    for line in text.splitlines():
        if _PARAGRAPH_BREAK.match(line):
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    out = []
    for chunk in blocks:
        out.extend(p for p in _LIST_ITEM.split(chunk) if p.strip())
    return out


class TestTheAnnCacheHazardIsWrittenWhereItWillBeRead:

    _CLAIM = "a re-pin leaves the recall cells and the subtree must be moved aside"
    _DIRECTIVE = "a sentence that tells the operator to pass --force"

    @pytest.mark.parametrize("path", sorted(ANN_SUBTREE_SITES))
    def test_the_subtree_requirement_is_made_at_this_site(self, path):
        """Some ONE paragraph at this site makes the whole claim.

        One paragraph and not the file, because a conjunction over a whole
        file can be satisfied by sentences that have nothing to do with each
        other -- which is what `_paragraphs` exists to prevent and what a
        review demonstrated by deleting the warning and leaving the
        vocabulary behind.
        """
        pattern = PROSE_CLAIMS[self._CLAIM]["pattern"]
        blocks = _paragraphs(open(os.path.join(VB_ROOT, path)).read())
        hits = [b for b in blocks if re.search(pattern, _flat(b), re.I)]
        assert hits, (
            f"no single paragraph in {path} still makes the whole claim: "
            f"{self._CLAIM}\n"
            f"  reason this site was chosen: {ANN_SUBTREE_SITES[path]}\n"
            f"  pattern: {pattern}\n"
            f"  the pattern is a vocabulary, not a sentence -- if the claim "
            f"is still made and only the wording moved, widen PROSE_CLAIMS "
            f"rather than restoring the old phrasing")

    @pytest.mark.parametrize("path", sorted(ANN_SUBTREE_SITES))
    def test_this_site_does_not_tell_the_operator_to_pass_force(self, path):
        """The assertion that would have caught two rounds of being wrong.

        `--force` does re-run points that already have results, so for two
        rounds these three sites said to pass it. It is also the escape hatch
        on the free-disk refusal, so passing it disables the warning a
        core-dump campaign most needs, and the remedy is to move the engine's
        subtree aside instead. The old guard grepped for the string `--force`
        and could not tell an instruction from a warning, so it was green
        either way.

        WHAT THIS CHECK DOES NOT COVER, stated because a guard over prose
        always has an edge and naming it is the only honest terminus. It reads
        one shape -- a directive verb, or a `run-benchmark.sh` command line,
        reaching `--force` without a negation in between. A sentence that
        recommends the flag without any of those verbs ("the flag to use here
        is `--force`") is not caught, and neither is an instruction split
        across two sentences. It is not a proof that these documents never
        recommend `--force`; it is a proof that they do not recommend it in
        the form an operator would copy. The rest is a reviewer's job, and
        that is an acceptable terminus only because this docstring says so.
        """
        pattern = PROSE_CLAIMS[self._DIRECTIVE]["pattern"]
        offenders = [
            _flat(b)[:200]
            for b in _paragraphs(open(os.path.join(VB_ROOT, path)).read())
            if re.search(pattern, _flat(b), re.I)]
        assert not offenders, (
            f"{path} tells the operator to pass --force, and the remedy for "
            f"a re-pin is to move the engine's ann subtree aside instead: "
            f"--force is also the escape hatch on the free-disk refusal, so "
            f"it disables the warning a core-dump campaign most needs.\n  "
            + "\n  ".join(offenders))

    def test_every_site_exists_and_says_why_it_is_a_site(self):
        for path, reason in sorted(ANN_SUBTREE_SITES.items()):
            assert reason.strip(), path
            assert os.path.isfile(os.path.join(VB_ROOT, path)), path

    @pytest.mark.parametrize("shape,text,expected", [
        # A markdown bullet list: the shape CLAUDE.md's invariants use.
        ("markdown bullets",
         "- the first bullet says the pin moved\n"
         "- the second says pass --force\n", 2),
        # A NUMBERED list: the shape docs/09 §5 uses, and the one the first
        # version of this splitter could not see at all.
        ("numbered items",
         "4. the fourth item says the pin moved\n"
         "5. the fifth says pass --force\n", 2),
        # A YAML comment block broken by comment-marker-only lines: the shape
        # `config/engines/percona.yml` uses, and the site where the deletion
        # attack was actually open.
        ("yaml comment paragraphs",
         "  # the first paragraph says the pin moved\n"
         "  # and continues\n"
         "  #\n"
         "  # the second says pass --force\n", 2),
        # A shell comment block, same rule, because the entrypoint uses it.
        ("shell comment paragraphs",
         "# first paragraph\n#\n# second paragraph\n", 2),
        # Continuation lines stay with their item rather than becoming
        # blocks of their own, or every multi-line warning would be split
        # into fragments that each fail the conjunction.
        ("an item keeps its continuations",
         "5. the fifth item says the pin moved\n"
         "   and that the cells are reused\n"
         "   so pass --force\n", 1),
    ])
    def test_the_paragraph_split_isolates_each_shape(self, shape, text,
                                                     expected):
        """The smuggling hole `_paragraphs` exists to close, asserted on every
        shape the three sites actually use rather than on the one that already
        worked. The first version of this test pinned only the markdown
        bullet, which is why the YAML and numbered cases shipped broken."""
        got = _paragraphs(text)
        assert len(got) == expected, (shape, got)

    def test_no_force_block_is_large_enough_to_smuggle(self):
        """The property, stated as a number rather than trusted to the
        splitter's implementation.

        A block much larger than the claim can satisfy a three-part
        conjunction out of three unrelated sentences. Seventeen lines is the
        largest any site needs today; the bound is what fails if a future
        edit merges the warning back into a wall of comment.
        """
        for path in sorted(ANN_SUBTREE_SITES):
            text = open(os.path.join(VB_ROOT, path)).read()
            blocks = [b for b in _paragraphs(text) if "--force" in b]
            assert blocks, path
            worst = max(len(b.splitlines()) for b in blocks)
            assert worst <= 25, (
                f"{path}'s largest --force block is {worst} lines; a block "
                f"that big can satisfy the three-part claim out of three "
                f"unrelated sentences, which is what `_paragraphs` exists to "
                f"prevent")
