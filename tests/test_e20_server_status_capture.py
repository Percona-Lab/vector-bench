"""E20 -- the vendor's memory gauge is documented as recorded and is collected
nowhere.

`CLAUDE.md`, `README.md`, `config/engines/percona.yml` -- which calls it "the
audit trail" for the decision to leave `innodb_hnsw_max_memory` unlimited --
and `docs/09-adding-percona-server.md` all state that
`Innodb_hnsw_memory_used` is recorded per phase. Nothing in `harness/`,
`report/` or `orchestrator/` read a status variable from ANY engine, so the
audit trail for a deliberate fairness decision did not exist.
`docs/11-methodology-roadmap.md` §4.1 says so in its own words and calls it
the most embarrassing item on that page.

**The gauge is not a missing server feature.** It is in the status-variable
array at the pin (`docs/09-adding-percona-server.md` §2.6, from
`innobase_system_variables[]` and the status array beside it, which together
contain exactly one HNSW status entry and no second), and the vendor suite has
a result file for it. Only the collector was ours to write.

WHAT THE FIX IS, by symbol:

* `EngineDriver.status_gauges` -- `()` on the base class, so the six engines
  that have no such variable record exactly what they recorded before.
* `EngineDriver.status_counters` -- `{}` on the base class. One optional hook
  beside `prepare_churn`, `wait_index_synced` and `maintain`, same
  no-op-default discipline.
* `MySQLFamilyDriver.status_counters` -- one `SHOW {GLOBAL|SESSION} STATUS
  LIKE` per name, coerced to `int`, **omitting** a name the server does not
  expose rather than zeroing it. Zero is a legitimate reading of this
  particular gauge -- §2.6 records it reading `0` after a restart, because the
  graph loads lazily -- so "absent" and "zero" are two different facts and
  have to stay two different facts.
* `_matching_status_row` -- the row check, extracted into a named function
  after the first review found both of its halves untested: `_` is a LIKE
  wildcard, `Innodb_hnsw_memory_used` contains three of them, and the fixture
  compared names for equality, so neither the widened-pattern case nor the
  short-row case could be constructed. Deleting either half left all 29 tests
  passing. `RecordingCursor` now translates `_` and `%` the way a server does.
* `PerconaDriver.status_gauges` -- `("Innodb_hnsw_memory_used",)`, the only
  engine in the set that declares one.
* `harness/workloads/build.py` -- `_server_status`, and one `extra` key,
  `extra.server_status`, a `{name: int}` dict on the `index_build` record,
  written only when a gauge was actually read.

WHAT THESE TESTS DO NOT ESTABLISH, stated here because a collector that has
never run against a live server is unproven however green this file is.
Nothing here starts a container, a server or a build: every statement goes to
a recording cursor, and the value that comes back is one this file made up.
So these tests establish the *shape* of the capture -- which statement is
issued, in which scope, what the record carries, and that no other engine's
record moves -- and they establish nothing about whether a real Percona Server
answers `SHOW GLOBAL STATUS LIKE 'Innodb_hnsw_memory_used'` with a row this
collector can parse. The first ops run on the build host is what confirms
that, and the thing to look at is `extra.server_status` on the `index_build`
record in `ops-percona-*.jsonl`.

THE CAVEAT THAT TRAVELS WITH EVERY FIGURE FROM THIS GAUGE, and the reason it
is repeated in the hook's docstring, in the driver's declaration and in
`build.py`: it omits the node map (*HNSW::m_nodes*, a supplied 30-50 bytes per
node) and search scratch, so it is a **floor** on graph memory and not a
measurement of it. At glove-100's 1,183,514 rows that is 35.5-59.2 MB
unreported per vector index.

PROBE STYLE, per this suite's convention (`tests/test_e9_percona_bulk_build_crash.py`,
`tests/test_e13_percona_churn_repair_is_a_noop.py`): a decision test reports
its honest and mutant value rather than a bare pass/fail, and an identical
pair is called out as a dead probe. The one mutant that can be injected at run
time -- `status_gauges` declared on the base class instead of on
`PerconaDriver` -- is injected with `mock.patch.object`, never by editing a
file on disk: CPython invalidates a `.pyc` on (whole-second mtime, size), so a
same-size rewrite inside one second can leave stale bytecode valid and measure
the previous version. The other two mutants the spec names are source edits
and were applied and reverted by hand.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import datasets as ds                                   # noqa: E402
from harness.drivers.base import (ConnectionSpec, EngineDriver,       # noqa: E402
                                  IndexSpec)
from harness.drivers.mysql_family import (AliSQLDriver, MariaDBDriver,  # noqa: E402
                                          PerconaDriver)
from harness.drivers.postgres import _driver_classes                 # noqa: E402
from harness.metrics import records                                  # noqa: E402
from harness.workloads import build as build_workload                # noqa: E402
from harness.workloads.context import RunContext                     # noqa: E402
from tests.fake_driver import (DIM, FakeDriver, K, N_ROWS,           # noqa: E402
                               index_spec, tiny_dataset)

#: The one HNSW status variable the pinned server registers. Spelled once.
GAUGE = "Innodb_hnsw_memory_used"

#: A value this file invented. It is deliberately not round and deliberately
#: not zero: a collector that returned a default would have to return one of
#: those, and an assertion against either could pass without a statement ever
#: having been issued. Nothing about it came off a server -- see the module
#: docstring on what these tests do not establish.
MEMORY_USED = 2_013_265_921

#: The `extra` key the `index_build` record carries. One key holding a
#: `{name: int}` dict, not one key per variable, so a second gauge needs no
#: schema decision and a report can look for a name.
KEY = "server_status"


# ---------------------------------------------------------------------------
# A recording cursor, so the SQL the driver sends -- or does not send -- is the
# thing under test
# ---------------------------------------------------------------------------

#: `SHOW <scope> STATUS LIKE '<name>'`, as the cursor below reads it back.
#: Matching the scope as a group rather than accepting either is what makes
#: assertion 1's "GLOBAL, not SESSION" an assertion and not a hope: the gauge
#: is GLOBAL status (`docs/09-adding-percona-server.md` §2.6), and a SESSION
#: read of a global-only variable is a different question with a different
#: answer.
STATUS_RE = re.compile(r"^\s*SHOW\s+(GLOBAL|SESSION)\s+STATUS\s+LIKE\s+'([^']*)'",
                       re.IGNORECASE)

#: What a live server answers `SHOW CREATE TABLE` with. Needed only because
#: `PerconaDriver.maintain()` reads the index options back off the server; no
#: test here calls it, and it is here so a cursor that is asked cannot answer
#: `None` and send the driver down a fall-back path by accident.
SHOW_CREATE = (
    "t1",
    "CREATE TABLE `t1` (\n"
    "  `id` bigint unsigned NOT NULL,\n"
    "  `v` vector(16) NOT NULL,\n"
    "  PRIMARY KEY (`id`),\n"
    "  VECTOR KEY `vi` (`v`) USING `hnsw` "
    "(`M`=16, `ef_construction`=200, `metric`=`euclidean`)\n"
    ") ENGINE=InnoDB",
)

TABLE_BYTES = 123_456_789
VERSION_STRING = "8.4.0-1.percona"


class StatusError(RuntimeError):
    """What a dead connection raises. Named so the assertion can say which."""


class RecordingCursor:
    """Records every statement and answers by STATEMENT, like a server.

    Deliberately self-contained rather than imported from
    `tests/test_e13_percona_churn_repair_is_a_noop.py` or
    `tests/test_e2_percona_wiring.py`: the central claim of this file is "no
    status statement reached the cursor", and it must not be able to break
    because another test file reshaped its fixture.

    Keyed on the statement rather than queued because `build.run` issues a
    dozen statements in an order this file should not have to encode, and a
    queue drained by one of them hands the next caller another caller's row.

    `status` is the server's status table as this fixture pretends it to be: a
    name absent from it is a name the server does not expose, which is the
    case assertion 3 is about.
    """

    def __init__(self, status: Optional[Dict[str, Any]] = None,
                 rows: int = N_ROWS, raises: bool = False,
                 raw_rows: Optional[List[Any]] = None):
        self.statements: List[str] = []
        self._status = dict(status or {})
        self._rows = rows
        self._raises = raises
        #: Answers a status read with these rows verbatim, bypassing `status`.
        #: For the malformed-row case, which no dict of names and values can
        #: express.
        self._raw_rows = raw_rows
        self._answer: List[tuple] = []

    def execute(self, sql, params=None):
        self.statements.append(str(sql))
        if self._raises:
            raise StatusError(f"server has gone away: {sql}")
        self._answer = self._answer_for(str(sql))

    def _answer_for(self, sql: str) -> List[tuple]:
        status = STATUS_RE.match(sql)
        if status is not None:
            if self._raw_rows is not None:
                return list(self._raw_rows)
            # **LIKE semantics, not an equality test, and this is the whole
            # reason this fixture was rewritten.** An earlier version matched
            # the requested name against its own keys exactly (case-
            # insensitively), which cannot produce the one input the
            # collector's name check exists for: a row the server returns
            # because the PATTERN matched, naming a different variable.
            # Measured: with that fixture, deleting the name check from
            # `_matching_status_row` left all 29 tests passing. `_` matches
            # any single character and `%` any run of them, exactly as the
            # server does it, so the widened pattern can now actually hit
            # something.
            #
            # The server also returns the name in its own casing, which is
            # reproduced too: the collector is supposed to key its result by
            # the name it ASKED for, so a fixture that echoed the request
            # verbatim could not tell that apart from one that trusted the
            # server's string.
            pattern = re.escape(status.group(2)).replace("_", ".").replace(
                "%", ".*")
            return [(name, value) for name, value in self._status.items()
                    if re.fullmatch(pattern, name, re.IGNORECASE)]
        upper = sql.upper()
        if upper.startswith("SHOW CREATE TABLE"):
            return [SHOW_CREATE]
        if upper.startswith("SELECT COUNT(*)"):
            return [(self._rows,)]
        if upper.startswith("SELECT DATA_LENGTH"):
            return [(TABLE_BYTES,)]
        if upper.startswith("SELECT VERSION()"):
            return [(VERSION_STRING,)]
        return []

    def executemany(self, sql, rows):
        self.statements.append(str(sql))
        if self._raises:
            raise StatusError(f"server has gone away: {sql}")
        self._answer = []

    def fetchone(self):
        return self._answer[0] if self._answer else None

    def fetchall(self):
        return list(self._answer)

    def close(self):
        pass

    @property
    def status_statements(self) -> List[str]:
        """Every `SHOW ... STATUS` statement, which is where the proof lives."""
        return [s for s in self.statements if STATUS_RE.match(s)]


def wired(cls, *, status: Optional[Dict[str, Any]] = None,
          raises: bool = False, build_mode: str = "post",
          raw_rows: Optional[List[Any]] = None):
    """A real driver of `cls` wired to a recording cursor, with no server.

    `connect()` is not called: it needs the `mariadb` connector, which is a
    container-side dependency this suite does not have on the host, and the
    only thing it would do here is hand the driver a cursor -- which is what
    this function does instead.
    """
    driver = cls(ConnectionSpec(host="h", port=3306))
    cursor = RecordingCursor(status=status, raises=raises, raw_rows=raw_rows)
    driver._cur = cursor
    driver._index = IndexSpec(dim=DIM, m=16, metric=ds.EUCLIDEAN,
                              build_mode=build_mode)
    return driver


def cursor_of(driver) -> RecordingCursor:
    return driver._cur


# ---------------------------------------------------------------------------
# 1. The statement, its scope, and the value that comes back
# ---------------------------------------------------------------------------

class TestTheStatementIssued:
    def test_a_global_status_read_naming_the_gauge_returns_its_value(self):
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED})

        got = driver.status_counters((GAUGE,))

        assert got == {GAUGE: MEMORY_USED}
        issued = cursor_of(driver).status_statements
        assert len(issued) == 1, issued
        scope, named = STATUS_RE.match(issued[0]).groups()
        print(f"STATEMENT = {issued[0]!r}")
        # GLOBAL status, established in `docs/09-adding-percona-server.md`
        # §2.6 from the server's own status-variable array. A SESSION read
        # here would be a different question.
        assert scope.upper() == "GLOBAL"
        assert named == GAUGE

    def test_the_scope_argument_is_the_only_thing_that_changes_the_keyword(self):
        """A probe, because `scope` has a default and a default is easy to
        ignore. The session arm asserts the parameter is wired, not that a
        session read of this gauge is meaningful -- it is not; the variable is
        GLOBAL status and nothing in this repo asks for it per session."""
        honest = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        honest.status_counters((GAUGE,))
        mutant = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        mutant.status_counters((GAUGE,), scope="session")

        honest_scope = STATUS_RE.match(cursor_of(honest).status_statements[0]).group(1)
        mutant_scope = STATUS_RE.match(cursor_of(mutant).status_statements[0]).group(1)
        print(f"HONEST (default scope)   = {honest_scope}")
        print(f"MUTANT (scope='session') = {mutant_scope}")
        assert honest_scope.upper() != mutant_scope.upper(), (
            "DEAD PROBE: the scope argument changes nothing")
        assert honest_scope.upper() == "GLOBAL"
        assert mutant_scope.upper() == "SESSION"

    def test_a_scope_that_is_neither_is_refused_rather_than_interpolated(self):
        """`checked_filter_column`'s discipline, for the same reason: the
        scope reaches a statement string, so an unrecognised one is a
        `ValueError` and never an interpolation. A caller bug, not a server
        failure -- and `build.py` swallows it either way, so no run dies of
        it."""
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        with pytest.raises(ValueError):
            driver.status_counters((GAUGE,), scope="GLOBAL; DROP DATABASE ann")
        assert cursor_of(driver).statements == []

    def test_a_name_that_could_not_be_spelled_safely_is_refused(self):
        """Same reason. The names come from a code-declared tuple, so this can
        only fire on a typo in `status_gauges` -- and failing loudly is what
        makes that typo findable, where silently skipping the name would make
        the gauge absent forever and look exactly like a server that does not
        expose it."""
        driver = wired(PerconaDriver)
        for bad in ("Innodb_hnsw_memory_used'; SELECT 1 --", "Innodb%", "", "a b"):
            with pytest.raises(ValueError):
                driver.status_counters((bad,))
        assert cursor_of(driver).statements == []

    def test_a_row_the_widened_pattern_dragged_in_is_not_taken(self):
        """**The headline catch, and it was untested until the first review
        found it.** `_` is a LIKE wildcard and `Innodb_hnsw_memory_used`
        contains three of them, so `LIKE 'Innodb_hnsw_memory_used'` also
        matches `Innodb_hnsw_memoryXused`. The old fixture compared names for
        equality, so the only input that reaches
        `_matching_status_row`'s name check could not be constructed, and
        deleting that check left all 29 tests passing.

        The decoy is FIRST in the status dict on purpose: the collector takes
        the first matching row, so an unguarded version returns the decoy's
        value under the real name's key -- a byte count filed as another
        variable. Both values are distinct so the assertion can tell which
        one came back."""
        decoy = "Innodb_hnsw_memoryXused"
        driver = wired(PerconaDriver, status={decoy: 999,
                                              GAUGE: MEMORY_USED})
        got = driver.status_counters((GAUGE,))

        statement = cursor_of(driver).status_statements[0]
        served = cursor_of(driver)._answer_for(statement)
        print(f"the fixture's server answered {len(served)} rows: {served}")
        assert len(served) == 2, (
            "DEAD PROBE: the fixture did not reproduce LIKE widening, so the "
            "name check cannot be reached")
        print(f"HONEST (exact name wins) = {got!r}")
        print(f"MUTANT (first row wins)  = {{{GAUGE!r}: 999}}")
        assert got == {GAUGE: MEMORY_USED}

    def test_a_row_too_short_to_hold_a_value_is_skipped_not_fatal(self):
        """The other half of the same guard, and it fails differently: a
        1-tuple that reached `row[1]` would raise inside the per-name
        `except`, which omits the gauge AND reports it as a failed read --
        losing the good row that follows it in the same result."""
        driver = wired(PerconaDriver,
                       raw_rows=[(GAUGE,), (GAUGE, MEMORY_USED)])
        got = driver.status_counters((GAUGE,))
        print(f"HONEST (short row skipped) = {got!r}")
        print(f"MUTANT (length check gone) = {{}} plus a failed-read line")
        assert got == {GAUGE: MEMORY_USED}

    def test_the_guard_is_reachable_on_its_own(self):
        """`_matching_status_row` is a named function rather than a condition
        inside the loop precisely so these cases can be stated directly. A
        condition that only a live server can reach is a condition nothing
        tests, which is how the two mutants above survived."""
        from harness.drivers.mysql_family import _matching_status_row as pick
        assert pick([(GAUGE, 7)], GAUGE) == (GAUGE, 7)
        assert pick([("INNODB_HNSW_MEMORY_USED", 7)], GAUGE) is not None
        assert pick([("Innodb_hnsw_memoryXused", 7)], GAUGE) is None
        assert pick([(GAUGE,)], GAUGE) is None
        assert pick([], GAUGE) is None
        assert pick([(GAUGE,), ("x", 1), (GAUGE, 7)], GAUGE) == (GAUGE, 7)
        # Not a row at all. Skipped rather than allowed to abort the names
        # after it.
        assert pick([None, (GAUGE, 7)], GAUGE) == (GAUGE, 7)

    def test_the_result_is_keyed_by_the_name_asked_for(self):
        """Not by the server's string, which is a different object and whose
        casing is the server's business. `checked_filter_column` returns the
        whitelisted element rather than the caller's string for the same
        reason: the key set of `extra.server_status` has to be decidable from
        this repository, because that is what a report greps for."""
        driver = wired(PerconaDriver, status={"INNODB_HNSW_MEMORY_USED": MEMORY_USED})
        got = driver.status_counters((GAUGE,))
        print(f"server spelled it INNODB_HNSW_MEMORY_USED; record says "
              f"{list(got)}")
        assert got == {GAUGE: MEMORY_USED}


# ---------------------------------------------------------------------------
# 2. A cursor that raises loses no measurement
# ---------------------------------------------------------------------------

class TestAFailingServerIsNotAFailedRun:
    def test_a_cursor_that_raises_yields_nothing_and_propagates_nothing(self):
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED}, raises=True)
        got = driver.status_counters((GAUGE,))
        print(f"cursor raised {StatusError.__name__}; collector returned {got!r}")
        assert got == {}

    def test_one_dead_name_does_not_cost_the_others(self):
        """Per-name, not per-call. The loop swallows per statement, so a gauge
        a future server drops does not take the gauges beside it with it."""
        class HalfDead(RecordingCursor):
            def execute(self, sql, params=None):
                if "doomed" in str(sql):
                    self.statements.append(str(sql))
                    raise StatusError(sql)
                super().execute(sql, params)

        driver = wired(PerconaDriver)
        driver._cur = HalfDead(status={GAUGE: MEMORY_USED})
        got = driver.status_counters((GAUGE, "Innodb_doomed_counter"))
        assert got == {GAUGE: MEMORY_USED}

    def test_build_run_swallows_a_driver_whose_hook_raises(self):
        """`_driver_capabilities`'s discipline, which is the pattern the hook
        is modelled on: a measurement already taken is not lost to a driver
        that misbehaves afterwards. The record is written, and it carries no
        status key."""
        class Exploding(FakeDriver):
            status_gauges = (GAUGE,)

            def status_counters(self, names, scope="global"):
                raise StatusError("the hook itself is broken")

        built = _one_build(Exploding)
        assert KEY not in built["extra"]
        assert built["phase"] == records.PHASE_BUILD
        assert built["index_bytes"] > 0


# ---------------------------------------------------------------------------
# 3. Absent is not zero
# ---------------------------------------------------------------------------

class TestAbsentIsNotZero:
    """`docs/09-adding-percona-server.md` §2.6 is why this is the sharpest
    assertion in the file: `hnsw_memory_used.result` shows the gauge reading
    `0` after a restart, because the graph is faulted in lazily and rises on
    the first search. So zero is a MEASUREMENT of this variable, and a
    collector that zeroed a variable it could not read would publish "the
    graph is empty" and "this server has no such variable" as the same
    number."""

    def test_a_variable_the_server_does_not_expose_is_absent(self):
        driver = wired(PerconaDriver, status={})
        got = driver.status_counters((GAUGE,))
        print(f"HONEST (omit)        = {got!r}")
        print(f"MUTANT (zero-fill)   = {{{GAUGE!r}: 0}}")
        assert got == {}
        assert GAUGE not in got

    def test_a_real_zero_is_carried_through(self):
        """The other half, and the half that makes the one above mean
        something. Dropping zeros would be an equally wrong collector that
        passes the test above."""
        driver = wired(PerconaDriver, status={GAUGE: 0})
        got = driver.status_counters((GAUGE,))
        assert got == {GAUGE: 0}

    def test_a_value_that_is_not_a_number_is_omitted_rather_than_guessed(self):
        driver = wired(PerconaDriver, status={GAUGE: "disabled"})
        assert driver.status_counters((GAUGE,)) == {}

    @pytest.mark.parametrize("raw,expected", [
        (MEMORY_USED, MEMORY_USED),
        (str(MEMORY_USED), MEMORY_USED),
        (str(MEMORY_USED).encode(), MEMORY_USED),
        ("0", 0),
        ("  17  ", 17),
    ])
    def test_the_spellings_a_connector_may_hand_back(self, raw, expected):
        """A status value arrives as `int`, `str` or `bytes` depending on the
        connector and the column type. Bytes need handling not because `int()`
        refuses them -- `int(b'17')` is 17 -- but because `status_int` decodes
        them onto the string path, where the `float()` fallback turns a value
        like `b"0.000"` into 0 instead of a `ValueError`. The record field is
        an int either way."""
        driver = wired(PerconaDriver, status={GAUGE: raw})
        assert driver.status_counters((GAUGE,)) == {GAUGE: expected}


# ---------------------------------------------------------------------------
# 4. Which engines declare a gauge, and what that costs the ones that do not
# ---------------------------------------------------------------------------

class TestOnlyPerconaDeclaresAGauge:
    """The answer to "what happens for the engines that have no such
    variable", in code rather than by implication: `()` on the base class and
    an early return in `build.py`, so no statement is issued and no key is
    written."""

    def test_the_base_class_declares_none_and_collects_nothing(self):
        assert EngineDriver.status_gauges == ()
        # The no-op default, same discipline as `prepare_churn`,
        # `wait_index_synced` and `maintain`.
        assert EngineDriver.status_counters(
            mock.Mock(spec=EngineDriver), (GAUGE,)) == {}

    def test_percona_is_the_only_driver_in_the_registry_that_declares_one(self):
        """Derived from `_driver_classes()` rather than from a list written
        here, which would be the drift-by-omission hole the claim is meant to
        close: a driver added to the registry would otherwise be silently
        exempt from "nothing else moved". `_driver_classes` is one entry per
        ARCHITECTURE, which is the right granularity -- `status_gauges` is a
        class attribute, so mariadb and mariadb123 share one declaration."""
        declared = {name: tuple(cls.status_gauges)
                    for name, cls in sorted(_driver_classes().items())}
        print(f"DECLARED = {declared}")
        assert declared["PerconaDriver"] == (GAUGE,)
        assert all(gauges == () for name, gauges in declared.items()
                   if name != "PerconaDriver"), declared
        # Non-vacuity. Deriving the set from the registry catches a driver
        # ADDED without thinking about gauges, which is the drift this test
        # exists for -- but an "all of them" assertion over a registry that
        # had SHRUNK would pass while saying nothing. The six architectures
        # that exist today are named so that losing one is a failure here
        # rather than a quietly weaker test.
        assert set(declared) >= {"MariaDBDriver", "AliSQLDriver",
                                 "PostgresDriver", "MongoDriver",
                                 "ValkeyDriver", "PerconaDriver"}, declared

    def test_mariadb_and_alisql_send_no_status_statement_through_build_run(
            self, tmp_path, dataset):
        """Asserted on the cursor's statement list, which is where the proof
        lives: "the record has no key" could also be produced by a driver that
        asked the server and got nothing, and that is a different claim --
        it would mean a statement per build per engine against five servers
        that have nothing to answer."""
        for cls in (MariaDBDriver, AliSQLDriver):
            driver = wired(cls, status={GAUGE: MEMORY_USED})
            built = _run_build(tmp_path / f"no-gauge-{cls.__name__}",
                               dataset, driver)[1]
            issued = cursor_of(driver).status_statements
            print(f"{cls.__name__}: status statements = {issued}")
            assert issued == []
            assert KEY not in built["extra"]

    def test_declaring_the_gauge_on_the_base_class_is_the_mutant(
            self, tmp_path, dataset):
        """The run-time mutant for assertion 4, injected rather than edited
        onto disk. A `status_gauges` on `EngineDriver` instead of on
        `PerconaDriver` would be invisible to every assertion above -- Percona
        would still collect, and the result dict would still be right -- and
        would send a `SHOW GLOBAL STATUS` to MariaDB and AliSQL on every
        build, for a variable neither server has, and write them an `extra`
        key no record of theirs has ever carried."""
        def mariadb_build(suffix):
            driver = wired(MariaDBDriver, status={GAUGE: MEMORY_USED})
            built = _run_build(tmp_path / f"mutant-{suffix}", dataset, driver)[1]
            return cursor_of(driver).status_statements, built["extra"]

        honest_issued, honest_extra = mariadb_build("honest")
        with mock.patch.object(EngineDriver, "status_gauges", (GAUGE,)):
            mutant_issued, mutant_extra = mariadb_build("base-class")

        print(f"HONEST (gauge on PerconaDriver) = {len(honest_issued)} "
              f"statements, {KEY} in extra: {KEY in honest_extra}")
        print(f"MUTANT (gauge on EngineDriver)  = {len(mutant_issued)} "
              f"statements, {KEY} in extra: {KEY in mutant_extra}")
        assert (honest_issued, KEY in honest_extra) != \
               (mutant_issued, KEY in mutant_extra), (
            "DEAD PROBE: moving the declaration to the base class changed "
            "nothing about a MariaDB build")
        assert honest_issued == [] and KEY not in honest_extra
        assert len(mutant_issued) == 1 and mutant_extra[KEY] == {GAUGE: MEMORY_USED}


# ---------------------------------------------------------------------------
# 5. End to end through `build.run`: one new key, and only when there is one
# ---------------------------------------------------------------------------

def _run_build(path, dataset, driver, build_mode: str = "post"):
    """One build phase against `driver`, and the two records it wrote."""
    recorder = records.Recorder(str(path.parent / f"ops-{path.name}.jsonl"))
    ctx = RunContext(run_id="e20", resource_pass="normalized",
                     recorder=recorder, engine=driver.name, engine_tag="pin",
                     march="x86-64-v3", k=K,
                     cache_dir=str(path.parent / f"cache-{path.name}"))
    try:
        build_workload.run(ctx, driver, dataset,
                           index_spec(dataset, build_mode=build_mode))
    finally:
        recorder.close()
    written = list(records.read_records(recorder.path))
    assert [r["phase"] for r in written] == [records.PHASE_INGEST,
                                             records.PHASE_BUILD]
    return written


def _one_build(driver_cls, **kwargs):
    """The `index_build` record a `FakeDriver` subclass produces."""
    import tempfile
    from pathlib import Path
    dataset = tiny_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        driver = driver_cls(dim=dataset.dim, metric=dataset.metric, **kwargs)
        driver.connect()
        return _run_build(Path(tmp) / "one", dataset, driver)[1]


class NoGauges(FakeDriver):
    """Today's five unaffected engines, in the shape the suite already has for
    them: `status_gauges` inherited from the base class and untouched."""


class OneGauge(FakeDriver):
    """A driver that reports a gauge, without editing `tests/fake_driver.py`
    -- which belongs to no unit here and is shared by the whole suite."""

    status_gauges = (GAUGE,)

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        assert scope == "global"
        return {str(name): MEMORY_USED for name in names}


class ZeroGauge(FakeDriver):
    """Reports the gauge reading zero, which is the single most likely first
    real reading: `docs/09-adding-percona-server.md` §2.6 records
    `Innodb_hnsw_memory_used` at `0` after a restart because the graph faults
    in lazily."""

    status_gauges = (GAUGE,)

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        return {str(name): 0 for name in names}


class NeverAsked(FakeDriver):
    """Declares no gauges and records it if asked anyway.

    The only thing that notices `_server_status`' early return. Deleting that
    return sends no statement and writes no key -- the family loop iterates an
    empty tuple and the base hook touches no cursor -- so "the driver is not
    called" is the whole of what the return buys.

    **It records the call rather than raising, and that distinction is the
    difference between a live test and a dead one.** The first version of
    this class raised `AssertionError`, and the mutant survived: the raise
    lands in `_server_status`' own `except Exception`, which swallows it,
    prints a failed-read line and returns `{}` -- so the record looked
    identical and the test passed. A swallowed assertion is not an assertion.
    `asks` is a class attribute because `_one_build` constructs the driver
    itself and hands back only the record."""

    asks: List[tuple] = []

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        type(self).asks.append((tuple(names), scope))
        return {}


class SilentGauge(FakeDriver):
    """Declares a gauge the server does not answer for -- the shape a Percona
    build on a server without the variable would take. Assertion 3 at the
    record level: nothing to report is no key, not a key holding `{}` and not
    a key holding a zero."""

    status_gauges = (GAUGE,)

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        return {}


class TestTheRecordIsAdditive:
    def test_a_driver_that_reports_a_gauge_gets_the_key(self):
        built = _one_build(OneGauge)
        print(f"extra = {built['extra']}")
        assert built["extra"][KEY] == {GAUGE: MEMORY_USED}

    def test_a_recorded_zero_reaches_the_record(self):
        """**The layer a report reads, which the collector's own zero test
        does not cover.** `build.py` writes the key only when the dict is
        non-empty, and "non-empty" has to mean non-empty and not truthy: a
        gauge reading `0` is §2.6's post-restart measurement and the most
        likely first real reading on the build host. Changing both
        `if server_status` sites to `if any(server_status.values())` left all
        29 of the previous tests passing, because nothing drove a zero
        through `build.run`."""
        built = _one_build(ZeroGauge)
        print(f"HONEST (zero recorded)   = {built['extra'].get(KEY)!r}")
        print(f"MUTANT (truthiness test) = key absent")
        assert built["extra"][KEY] == {GAUGE: 0}

    def test_a_driver_declaring_no_gauges_is_never_asked(self):
        """What `_server_status`' early return actually buys, and the only
        observable it has. The hook is free for the six engines that want
        none of it -- a future implementation that opens a connection or logs
        cannot be triggered by a driver that declared nothing.

        Asserted on the recorded call and not on the record, because the
        record is identical either way: with the return deleted, the driver
        is asked for an empty tuple and answers `{}`."""
        NeverAsked.asks = []
        built = _one_build(NeverAsked)
        print(f"HONEST (not asked) = {NeverAsked.asks!r}")
        print(f"MUTANT (asked for ()) = [((), 'global')]")
        assert NeverAsked.asks == []
        assert KEY not in built["extra"]
        assert built["phase"] == records.PHASE_BUILD

    def test_a_declared_gauge_with_no_implementation_says_so(self, capsys):
        """The fifth way this comes back empty, and the one that used to be
        silent. A driver that declares a gauge and inherits the base no-op is
        a missing override in this repository; a server with no such variable
        is a fact about the server. They produce the same `{}` and call for
        opposite responses, so the run log has to separate them -- it is the
        only diagnostic this unit has in the field."""
        class DeclaresButInherits(FakeDriver):
            status_gauges = (GAUGE,)

        built = _one_build(DeclaresButInherits)
        printed = capsys.readouterr().out
        assert KEY not in built["extra"]
        assert "no driver implementation" in printed, printed
        assert GAUGE in printed

    def test_a_hook_that_raises_says_so_rather_than_looking_like_absence(
            self, capsys):
        """`status_counters` raising is either a server failure or a
        `ValueError` from a malformed `status_gauges` -- a bug in this
        repository. Without a line it is byte-for-byte indistinguishable from
        a server that has no such variable, which is what the first review
        demonstrated: the claim that a raise is "findable" was false at this
        call site."""
        class Exploding(FakeDriver):
            status_gauges = (GAUGE,)

            def status_counters(self, names, scope="global"):
                raise StatusError("the hook itself is broken")

        built = _one_build(Exploding)
        printed = capsys.readouterr().out
        assert KEY not in built["extra"]
        assert "status read failed" in printed, printed
        assert "the hook itself is broken" in printed

    def test_one_unparseable_value_does_not_lose_the_others(self, capsys):
        """The coercion is per name. In one comprehension inside the `try` --
        which is what this replaced -- a single value that would not become an
        `int` discarded the whole reading, including names that were fine."""
        class MixedBag(FakeDriver):
            status_gauges = (GAUGE, "Innodb_other_gauge")

            def status_counters(self, names, scope="global"):
                return {GAUGE: MEMORY_USED, "Innodb_other_gauge": "nonsense"}

        built = _one_build(MixedBag)
        printed = capsys.readouterr().out
        assert built["extra"][KEY] == {GAUGE: MEMORY_USED}
        assert "is not an int" in printed, printed

    def test_a_driver_that_reports_none_gets_no_key_at_all(self):
        built = _one_build(NoGauges)
        print(f"extra keys = {sorted(built['extra'])}")
        assert KEY not in built["extra"]

    def test_a_gauge_the_server_does_not_answer_for_gets_no_key_either(self):
        """The mutant this one exists for is a `build.py` that writes
        `extra.server_status: {}` unconditionally. It passes every assertion
        above -- the dict is right whenever there is a dict -- and it changes
        the key set of every record of all seven engines, which is the one
        thing "additive" has to mean."""
        built = _one_build(SilentGauge)
        print(f"HONEST (no key)              = {sorted(built['extra'])}")
        print(f"MUTANT (unconditional empty) = "
              f"{sorted(list(built['extra']) + [KEY])}")
        assert KEY not in built["extra"]

    def test_the_key_set_of_an_unaffected_engine_record_is_unchanged(self):
        """The assertion that makes "additive" true rather than claimed. The
        expected set is written out literally rather than diffed against
        another run of the same code, which would hold under a mutant that
        added the key to both."""
        built = _one_build(NoGauges)
        # `FakeDriver.capabilities()` declares exactly one key on purpose
        # (see its `_capabilities`), so this set is the fake's and not the
        # seven engines'. What it pins is that `build.py` added nothing.
        # Ops measurement version 9 added the index-size provenance to every
        # index_build record (`harness/workloads/index_size.py`); those four
        # are named here so the set stays literal.
        assert set(built["extra"]) == {
            "incremental_index", "separable_build",
            records.OPS_MEASUREMENT_VERSION_KEY,
            "size_read_at", "index_bytes_after_build",
            "index_size_settle_s", "index_size_settle",
        }, built["extra"]

    def test_only_the_index_build_record_carries_it(self):
        """Scope, stated so nobody reads an omission as an oversight: this
        unit collects at the build phase only. The `ingest` record written
        beside it is not a second place to look."""
        import tempfile
        from pathlib import Path
        dataset = tiny_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            driver = OneGauge(dim=dataset.dim, metric=dataset.metric)
            driver.connect()
            ingest, built = _run_build(Path(tmp) / "phases", dataset, driver)
        assert KEY not in (ingest.get("extra") or {})
        assert KEY in built["extra"]

    def test_a_percona_build_records_what_the_global_status_read_returned(
            self, tmp_path, dataset):
        """The whole unit in one assertion, through the real driver: a
        `PerconaDriver` in `post` mode, one bulk `ALTER TABLE`, one `SHOW
        GLOBAL STATUS`, and the value on the record. Against a recording
        cursor -- see the module docstring on what that does not
        establish."""
        driver = wired(PerconaDriver, status={GAUGE: MEMORY_USED})
        built = _run_build(tmp_path / "percona", dataset, driver)[1]

        print(f"statements = {cursor_of(driver).status_statements}")
        print(f"extra.{KEY} = {built['extra'][KEY]}")
        assert built["extra"][KEY] == {GAUGE: MEMORY_USED}
        assert built["engine"] == "percona"
        assert len(cursor_of(driver).status_statements) == 1
        # The gauge is read after the build and before the sizing queries, so
        # it describes the graph the build produced.
        statements = cursor_of(driver).statements
        alter = max(i for i, s in enumerate(statements)
                    if s.upper().startswith("ALTER TABLE"))
        status = statements.index(cursor_of(driver).status_statements[0])
        count = max(i for i, s in enumerate(statements)
                    if s.upper().startswith("SELECT COUNT(*)"))
        assert alter < status < count, statements

    def test_the_value_is_an_int_on_the_record_and_survives_jsonl(self):
        """The record is read back off disk, not out of the object, because
        `Recorder.write` is the only point every ops record passes through
        and JSON is what a report loads."""
        built = _one_build(OneGauge)
        value = built["extra"][KEY][GAUGE]
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value == MEMORY_USED


# ---------------------------------------------------------------------------
# The measurement-version verdict
# ---------------------------------------------------------------------------

class TestNoMeasurementVersionBumpIsOwed:
    """Decided against the charter above `OPS_MEASUREMENT_VERSION` in
    `harness/metrics/records.py` -- "bump in any change that alters what an
    ops number MEANS", extended by version 8 to re-partitioning -- rather than
    copied from anywhere.

    No existing number moves and no identity component changes. The version 8
    block names them by symbol: `OPS_UNIT_FIELDS` (`report/generate.py`),
    `OPS_UNIT_KEYS` (`workloads/churn_readers.py`), `build_signature` and
    `memory_stem` (`report/loaders.py`), and `charts.series_key` through the
    first.

    **Three of them read `extra`, and an earlier version of this docstring
    got all three wrong** -- it said `build_signature` read one key, named
    the wrong writer for it, and filed `series_key` under "named columns
    only". Corrected, by opening each one:

    * `build_signature` reads **two** `extra` keys, `source_file` and
      `quantization`.
    * `extra.source_file` is written by `report/loaders.py` from the ann
      result path, not by the overlay modules.
    * `extra.quantization` **is** written by this harness --
      `MongoDriver.capabilities()` returns it and `build.py`'s
      `_driver_capabilities` merges it into the very dict this unit appends
      to.
    * `charts.series_key` reads a third key, `extra.iterative_scan`, beside
      the engine and the signature.

    **The verdict survives all of that, and for a better reason than the one
    it had.** Every identity component reads `extra` by LITERAL KEY NAME --
    never by enumerating a record's key set -- so a key that none of them
    names cannot change what any of them returns, however many keys they
    read and whoever writes them. The test below pins that by running all
    five over a record with and without the key, `series_key` included; it
    ran three before, and the one it skipped was the one the prose had
    wrong.

    `ANN_MEASUREMENT_VERSION` is not in question: it is the first field of
    `ann_fingerprint`, which keys `results/annb/<pass>/<fingerprint>/`, and
    nothing in this unit touches the ann path at all. This unit writes to the
    ops records only.

    `docs/11-methodology-roadmap.md` §4.1 reaches the same verdict in its own
    words -- "stamped into `extra`, which is additive and owes no bump".
    """

    def test_the_ops_version_is_unchanged(self):
        # `>= 8` since version 9 (ops v9 bumped for F2-F11, a later change):
        # what this pins is that THIS change did not bump, not the value.
        assert records.OPS_MEASUREMENT_VERSION >= 8

    def test_the_new_key_reaches_no_identity_component(self):
        from report import charts, generate, loaders
        from harness.workloads import churn_readers
        assert KEY not in generate.OPS_UNIT_FIELDS
        assert KEY not in churn_readers.OPS_UNIT_KEYS

        # The `extra` of a record that carries every key an identity
        # component actually reads, so the comparison is made against a
        # populated signature rather than an empty one -- an all-None
        # signature would compare equal under almost any mutation.
        base_extra = {"source_file": "angular_M_16_quantization_none_400.hdf5",
                      "quantization": "none",
                      "iterative_scan": "relaxed_order"}
        without = {"engine": "percona", "dataset": "glove-100-angular",
                   "resource_pass": "normalized", "m": 16,
                   "build_mode": "post", "storage_engine": "InnoDB",
                   "extra": dict(base_extra)}
        with_status = dict(without,
                           extra={**base_extra, KEY: {GAUGE: MEMORY_USED}})

        for fn in (loaders.build_signature, loaders.memory_stem,
                   loaders.operating_point_key, generate.ops_unit,
                   charts.series_key):
            before, after = fn(without), fn(with_status)
            print(f"{fn.__name__}: {before!r}")
            assert before == after, fn.__name__
            # Non-vacuity: a component that returned None for both would
            # satisfy the equality above without reading anything.
            assert before is not None and before != (), fn.__name__


@pytest.fixture
def dataset():
    return tiny_dataset()


# ---------------------------------------------------------------------------
# 6. Server WAIT snapshots around each concurrency window
#
# The status gauge above is one figure read after a build. The wait snapshots
# apply the same no-op-default discipline per measured window:
# `EngineDriver.wait_snapshot` / `wait_delta`, read on a SEPARATE connection
# immediately before and after the window. The decision under test is that
# they are OUTSIDE the timed span -- a performance_schema scan inside the
# window would be charged to the engine as lost throughput.
# ---------------------------------------------------------------------------

from harness.drivers.engine_waits import (mysql_delta,          # noqa: E402
                                          mysql_snapshot)
from harness.drivers.mysql_family import MySQLFamilyDriver      # noqa: E402
from harness.drivers.server_waits import (NOT_IMPLEMENTED,      # noqa: E402
                                          counter_delta, event)
from harness.workloads import concurrency as concurrency_workload  # noqa: E402

#: Long enough that a snapshot taken inside a 0.2 s window would at least
#: quintuple its wall and cut its QPS by the same factor.
SNAPSHOT_SLEEP_S = 1.0
WINDOW_S = 0.2
MUTEX = "wait/synch/mutex/innodb/trx_sys_mutex"
DATAFILE = "wait/io/file/innodb/innodb_data_file"


class SlowWaits(FakeDriver):
    """A fake whose every snapshot costs a second and advances a counter.

    Class-level state, because the capture asks the factory for a driver of
    its own and the clients are forked; only the parent ever snapshots.
    """

    calls: List[float] = []

    def clone(self):
        sibling = super().clone()
        sibling.__class__ = type(self)
        return sibling

    def wait_snapshot(self):
        import time as _time
        _time.sleep(SNAPSHOT_SLEEP_S)
        SlowWaits.calls.append(_time.time())
        n = len(SlowWaits.calls)
        return {"events": {MUTEX: event(10 * n, 0.25 * n),
                           DATAFILE: event(3 * n, 0.5 * n)}}, None

    def wait_delta(self, before, after):
        return counter_delta(before, after, "fake")


def _loaded(cls):
    from tests.fake_driver import loaded_driver
    driver = loaded_driver(tiny_dataset(), ef_half=64.0, ef_search=32)
    driver.__class__ = cls
    return driver


def _concurrency(tmp_path, driver, clients=(1,)):
    recorder = records.Recorder(str(tmp_path / "ops-waits.jsonl"))
    ctx = RunContext(run_id="e20w", resource_pass="normalized",
                     recorder=recorder, engine=driver.name, engine_tag="pin",
                     march="x86-64-v3", k=K,
                     cache_dir=str(tmp_path / "cache"))
    dataset = tiny_dataset()
    try:
        concurrency_workload.run(
            ctx, driver.clone, dataset, index_spec(dataset), ef_search=32,
            client_counts=clients, duration_s=WINDOW_S, warmup=2,
            max_queries=50)
    finally:
        recorder.close()
    return list(records.read_records(recorder.path))


class TestWaitSnapshotsAreOutsideTheWindow:
    def test_a_one_second_snapshot_does_not_move_the_window(self, tmp_path):
        """Honest: the window's wall and its QPS-implied wall stay ~0.2 s
        although each snapshot sleeps 1 s.

        Mutant (the capture's `begin()` moved to after `pool.start`, inside
        the barrier-opened window): `window_ended_at - window_started_at`
        and `queries_executed / qps` both grow past 1.2 s.
        """
        SlowWaits.calls = []
        row = _concurrency(tmp_path, _loaded(SlowWaits))[0]
        extra = row["extra"]
        wall = extra["window_ended_at"] - extra["window_started_at"]
        implied = row["queries_executed"] / row["qps"]
        print(f"window wall {wall:.3f}s, implied {implied:.3f}s, "
              f"snapshots {len(SlowWaits.calls)}")
        assert len(SlowWaits.calls) == 2, "one snapshot each side"
        assert wall < WINDOW_S + 0.5 * SNAPSHOT_SLEEP_S
        assert implied < WINDOW_S * 1.5
        # The two snapshots bracket the window rather than sit in it. The
        # first returns before the window opens; the second only completes
        # a full sleep after it closed.
        before, after = SlowWaits.calls
        assert before <= extra["window_started_at"]
        assert after >= extra["window_ended_at"] + SNAPSHOT_SLEEP_S * 0.9

    def test_the_delta_is_after_minus_before_ranked_by_time(self, tmp_path):
        SlowWaits.calls = []
        extra = _concurrency(tmp_path, _loaded(SlowWaits))[0]["extra"]
        waits = extra["server_waits"]
        assert extra["server_waits_reason"] is None
        assert waits["ranked_by"] == "wait_s"
        assert [e["event"] for e in waits["events"]] == [DATAFILE, MUTEX]
        assert waits["events"][0] == {"event": DATAFILE, "count": 3,
                                      "wait_s": 0.5}
        assert waits["events"][1]["count"] == 10
        assert waits["total_wait_s"] == 0.75
        assert "warm-up" in waits["covers"]

    def test_a_driver_that_overrides_nothing_records_null_and_why(
            self, tmp_path):
        """Absent is not zero: `server_waits` is None, never `{events: []}`."""
        extra = _concurrency(tmp_path, _loaded(FakeDriver))[0]["extra"]
        assert extra["server_waits"] is None
        assert extra["server_waits_reason"] == NOT_IMPLEMENTED

    def test_the_base_class_default_is_none_and_a_reason(self):
        driver = FakeDriver()
        assert driver.wait_snapshot() == (None, NOT_IMPLEMENTED)
        assert driver.wait_delta({}, {}) == (None, NOT_IMPLEMENTED)
        assert EngineDriver.wait_sample_interval_s is None


class WaitCursor:
    """Answers the four statements `mysql_snapshot` sends."""

    def __init__(self, ps_on=True, timed=("wait/io/file/x",), rows=(),
                 mutex_rows=()):
        self.ps_on, self.timed = ps_on, list(timed)
        self.rows, self.mutex_rows = list(rows), list(mutex_rows)
        self.statements: List[str] = []
        self._answer: List[tuple] = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        upper = sql.upper()
        if "@@PERFORMANCE_SCHEMA" in upper:
            self._answer = [(1 if self.ps_on else 0,)]
        elif "SETUP_INSTRUMENTS" in upper:
            self._answer = [(n,) for n in self.timed]
        elif "EVENTS_WAITS_SUMMARY" in upper:
            self._answer = list(self.rows)
        elif "INNODB MUTEX" in upper:
            self._answer = list(self.mutex_rows)
        else:
            self._answer = []

    def fetchone(self):
        return self._answer[0] if self._answer else None

    def fetchall(self):
        return list(self._answer)


class TestMySQLFamilyWaits:
    def test_performance_schema_off_is_none_with_a_reason(self):
        """MariaDB's default. Mutant (`return {"events": {}}, None` on OFF):
        the record would claim a measured window in which nothing waited."""
        driver = MariaDBDriver(ConnectionSpec(host="h", port=3306))
        driver._cur = WaitCursor(ps_on=False)
        snap, reason = driver.wait_snapshot()
        assert snap is None
        assert "performance_schema is OFF" in reason

    def test_no_timed_wait_instrument_is_none_with_a_reason(self):
        snap, reason = mysql_snapshot(WaitCursor(timed=()))
        assert snap is None and "enabled and timed" in reason

    def test_picoseconds_become_seconds_and_idle_is_excluded(self):
        cur = WaitCursor(timed=("wait/io/file/x", MUTEX), rows=[
            (MUTEX, 4, 2_000_000_000_000)],
            mutex_rows=[("InnoDB", "trx_sys.cc:1", "waits=7")])
        snap, reason = mysql_snapshot(cur)
        assert reason is None
        assert snap["events"][MUTEX] == {"count": 4, "wait_s": 2.0}
        assert snap["synch_instruments_timed"] == 1
        assert snap["innodb_mutex"] == {"trx_sys.cc:1": 7}
        assert any("<> 'idle'" in s for s in cur.statements)

    def test_the_family_delta_differences_mutexes_too(self):
        before = {"events": {MUTEX: event(4, 2.0)},
                  "innodb_mutex": {"a": 7}, "synch_instruments_timed": 3}
        after = {"events": {MUTEX: event(9, 3.5), "new": event(1, 0.1)},
                 "innodb_mutex": {"a": 10}, "synch_instruments_timed": 3}
        waits, reason = mysql_delta(before, after)
        assert reason is None
        assert waits["events"][0] == {"event": MUTEX, "count": 5,
                                      "wait_s": 1.5}
        assert waits["events_total"] == 2
        assert waits["innodb_mutex_waits"] == {"a": 3}

    def test_a_counter_that_went_backwards_refuses_the_window(self):
        waits, reason = counter_delta({"events": {MUTEX: event(9, 3.0)}},
                                      {"events": {MUTEX: event(1, 0.1)}}, "s")
        assert waits is None and "went backwards" in reason

    def test_only_the_top_ten_are_kept_but_the_total_covers_all(self):
        after = {"events": {f"e{i:02d}": event(1, float(i))
                            for i in range(1, 16)}}
        waits, _ = counter_delta({"events": {}}, after, "s")
        assert len(waits["events"]) == 10
        assert waits["events"][0]["event"] == "e15"
        assert waits["events_total"] == 15
        assert waits["total_wait_s"] == float(sum(range(1, 16)))

    def test_every_mysql_family_driver_in_the_registry_reads_waits(self):
        family = [cls for cls in _driver_classes().values()
                  if issubclass(cls, MySQLFamilyDriver)]
        assert len(family) >= 4, family
        for cls in family:
            assert cls.wait_snapshot is not EngineDriver.wait_snapshot, cls
