"""E9 -- Percona's bulk vector-index build kills the server, and the default
build mode used to walk every profile straight into it.

WHAT WAS OBSERVED, and the only thing here that is not checkable from the
repo: on the build host, at `config/engines/percona.yml`'s pinned
`source.tag`, on fashion-mnist-784-euclidean with a 20,000-row subset, M=16,
ef_construction=200, normalized pass --

    [percona] loading 20,000 x 784 vectors with 1 thread(s); bulk build after load
    [percona] load: 7.7s (2,583 rows/s)
    [percona] ALTER TABLE t1 ADD VECTOR KEY vi (v) TYPE hnsw (M = 16, ...)
    mariadb.InterfaceError: Lost connection to server during query

-- raised from `harness/drivers/mysql_family.py`'s `create_index`. The load
completed; the ALTER killed the server. The same corpus, M and
ef_construction built and served correctly in `incremental` mode in the same
run, reaching recall@10=0.9895 at the smallest grid ef_search and 8,263 qps
at four clients, so this is the BULK path alone and not the index, the
dimensionality or the dataset.

THE DEFECT ON THIS SIDE was not the crash, which belongs to the server. It
was that `orchestrator/cli.py`'s ops loop defaulted every engine's build mode
to the literal `["post"]`, so EVERY profile that names no mode -- `main`,
`smoke`, `dev`, `full`, `quick` -- ran Percona in the one mode that cannot
complete, and did so after paying for the entire load first. The fix moves
the default into the engine's own declaration.

A SECOND, SMALLER DEFECT fixed alongside it: a failed ops unit printed
nothing. One ops phase is M values x build modes x storage engines, folded
into one `worst_rc`, so a phase in which `post` crashed and `incremental`
succeeded perfectly reported a bare `[FAIL] ... (exit 1)` naming neither. It
cost a real diagnosis -- the crash was first read as a failure of the unit
that had just succeeded.

Probe style, per the suite's convention: every decision test reports its
HONEST and MUTANT value rather than a bare pass/fail, and an identical pair
is called out as a dead probe. No file on disk is patched here -- the inputs
are a profile dict and an engine-config dict, so the mutant arm is a copy
with one key changed, which is cheaper and cannot leave residue. Nothing in
this file starts a container, a server or a build.
"""

import ast
import copy
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI_PATH = os.path.join(REPO, "orchestrator", "cli.py")
ENGINE_PATH = os.path.join(REPO, "config", "engines", "percona.yml")

CAPABILITY = "bulk_vector_index_build"


def _cli_source():
    with open(CLI_PATH) as fh:
        return fh.read()


def _percona_cfg():
    from orchestrator.config import load_engine
    return load_engine("percona")


def _with_capability(cfg, value):
    """The engine config as it would read if the server were fixed."""
    mutant = copy.deepcopy(cfg)
    mutant.setdefault("capabilities", {})[CAPABILITY] = value
    return mutant


# ---------------------------------------------------------------------------
# The declaration, and the evidence behind it
# ---------------------------------------------------------------------------

class TestTheEngineDeclaresWhatItCannotDo:

    def test_the_capability_is_declared_false(self):
        caps = _percona_cfg().get("capabilities", {}) or {}
        assert CAPABILITY in caps, (
            f"{CAPABILITY} is the engine's declaration that its bulk vector "
            f"index build is unusable at this pin; the ops loop reads it to "
            f"pick the default build mode")
        assert caps[CAPABILITY] is False

    def test_the_reason_is_recorded_next_to_the_declaration(self):
        """R7: the claim in the comment is part of the deliverable.

        A bare `false` would be re-flipped by the next person who wondered
        why, so the statement, the symptom and the condition for restoring it
        have to be written where the key is.
        """
        with open(ENGINE_PATH) as fh:
            text = fh.read()
        head = text.index(CAPABILITY + ":")
        # The note sits above the key; take a generous window back.
        window = text[max(0, head - 2600):head]
        for needle in ("ADD VECTOR KEY", "ALGORITHM=", "Lost connection",
                       "incremental"):
            assert needle in window, (
                f"{needle!r} is missing from the note above {CAPABILITY}; "
                f"without it the declaration cannot be audited or retired")

    def test_no_other_engine_declares_it(self):
        """A capability absent everywhere else is what keeps this a
        Percona-specific statement rather than a new axis every engine has to
        answer. Absence reads as `True` -- the bulk build works -- which is
        what every other engine's default depends on.
        """
        from orchestrator.config import load_engine
        from orchestrator.cli import KNOWN_ENGINES
        declared = {}
        for engine in KNOWN_ENGINES:
            caps = (load_engine(engine).get("capabilities", {}) or {})
            if CAPABILITY in caps:
                declared[engine] = caps[CAPABILITY]
        # VillageSQL is the second: its bulk CREATE INDEX intermittently
        # deadlocks and aborts the server at its pin (config/engines/
        # villagesql.yml), pinned in tests/test_e26_villagesql_engine.py.
        assert declared == {"percona": False, "villagesql": False}, declared


# ---------------------------------------------------------------------------
# The default, which is the part that was crashing
# ---------------------------------------------------------------------------

class TestTheDefaultComesFromTheEngineAndNotFromALiteral:

    def test_a_profile_that_names_no_mode_gets_the_mode_that_completes(self):
        from orchestrator.cli import _ops_build_modes
        cfg = _percona_cfg()
        honest = _ops_build_modes("percona", {"ops": {}}, cfg)
        mutant = _ops_build_modes("percona", {"ops": {}},
                                  _with_capability(cfg, True))
        print(f"HONEST (capability false) = {honest}")
        print(f"MUTANT (capability true)  = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the default did not move when the capability was "
            f"flipped, so this test is not measuring the fix. Both arms "
            f"returned {honest}")
        assert honest == ["incremental"], (
            "a profile naming no mode must not get the mode that kills the "
            f"server; got {honest}")
        assert mutant == ["post"], (
            "with the bulk build declared working, the default must return to "
            f"each engine's idiomatic build; got {mutant}")

    def test_the_default_is_read_through_the_capability_for_every_profile(self):
        """Not only the Percona-only profile: the crash was in the profiles
        that say nothing, which is most of them.
        """
        from orchestrator.cli import _ops_build_modes
        from orchestrator.config import load_profile
        cfg = _percona_cfg()
        for name in ("main", "smoke", "dev", "full", "quick"):
            profile = load_profile(name)
            modes = _ops_build_modes("percona", profile, cfg)
            print(f"{name}: percona build modes = {modes}")
            assert "post" not in modes, (
                f"profile {name!r} would run Percona's bulk build, which "
                f"cannot complete at this pin")

    @pytest.mark.parametrize("engine", ["mariadb", "mariadb123", "alisql",
                                        "pgvector", "mongodb", "valkey"])
    def test_no_other_engine_s_default_moved(self, engine):
        """The fix widens where the default comes from. It must not change
        the default for an engine that declares nothing, because for MariaDB
        and AliSQL `post` is a documented no-op and the measured build is the
        incremental one either way.
        """
        from orchestrator.cli import _ops_build_modes
        from orchestrator.config import load_engine
        modes = _ops_build_modes(engine, {"ops": {}, "ann": {}},
                                 load_engine(engine))
        print(f"{engine}: default build modes = {modes}")
        assert modes == ["post"]


class TestAnExplicitRequestIsHonouredRatherThanRefused:

    def test_post_is_still_reachable_on_purpose(self, capsys):
        """Whoever checks a newer Percona for the fix must be able to ask for
        the broken mode without editing the engine config. A refusal here
        would make this file the obstacle to retiring itself.
        """
        from orchestrator.cli import _ops_build_modes
        profile = {"ops": {"percona_build_modes": ["post"]}}
        modes = _ops_build_modes("percona", profile, _percona_cfg())
        warning = capsys.readouterr().err
        print(f"returned = {modes}")
        assert modes == ["post"], (
            "an explicit request must be honoured; a silent substitution "
            "would make the run's own records disagree with the profile")
        assert "WARNING" in warning and CAPABILITY in warning, warning
        assert "lost connection" in warning.lower(), (
            "the warning has to name the symptom, or the resulting failure "
            f"reads as a harness fault: {warning!r}")

    def test_the_warning_is_silent_when_the_mode_is_usable(self, capsys):
        from orchestrator.cli import _ops_build_modes
        profile = {"ops": {"percona_build_modes": ["post"]}}
        modes = _ops_build_modes("percona", profile,
                                 _with_capability(_percona_cfg(), True))
        warning = capsys.readouterr().err
        print(f"returned = {modes}, stderr = {warning!r}")
        assert modes == ["post"]
        assert warning == "", (
            "a warning on a working bulk build would train the operator to "
            f"ignore it: {warning!r}")


# ---------------------------------------------------------------------------
# The profile, and its agreement with the estimator
# ---------------------------------------------------------------------------

class TestTheProfileAsksForTheMeasurableHalfOnly:

    def test_the_percona_profile_names_incremental_alone(self):
        from orchestrator.config import load_profile
        modes = load_profile("percona")["ops"]["percona_build_modes"]
        assert modes == ["incremental"], modes

    def test_the_loop_and_the_estimator_agree_on_the_unit_count(self):
        """`tests/test_e2_percona_wiring.py` pins the estimator's count, and
        the two have to move together: a unit the runner executes and the
        estimator does not count is a printed figure short by a whole unit of
        every query term.
        """
        from orchestrator.cli import _ops_build_modes
        from orchestrator.config import load_profile
        from orchestrator import estimate as est_mod
        profile = load_profile("percona")
        modes = _ops_build_modes("percona", profile, _percona_cfg())
        m_values = len(profile["ops"]["m_values"])
        assert est_mod._ops_units(profile, "percona") == m_values * len(modes)

    def test_the_profile_records_what_the_narrowing_costs(self):
        """The axis this profile exists for is down to one value. Saying so
        is the difference between a measurement that is missing and one that
        was judged uninteresting.
        """
        with open(os.path.join(REPO, "config", "profiles",
                               "percona.yml")) as fh:
            text = fh.read()
        head = text.index("percona_build_modes: [incremental]")
        window = text[max(0, head - 1800):head]
        assert CAPABILITY in window, (
            "the profile must point at the engine's declaration, so the "
            "evidence lives in one place")
        for needle in ("post", "incremental"):
            assert needle in window, needle


# ---------------------------------------------------------------------------
# A failed unit has to say so
# ---------------------------------------------------------------------------

class TestAFailedUnitIsNoLongerSilent:

    def test_the_failing_unit_names_itself(self):
        src = _cli_source()
        assert "UNIT FAILED" in src, (
            "a unit that fails has to name its own tag; the phase-level "
            "[FAIL] line cannot say which of the units it covers broke")
        assert "failed_units.append(tag)" in src

    def test_the_phase_summarises_which_units_failed(self):
        src = _cli_source()
        assert "units failed:" in src, (
            "the phase's exit code is an aggregate over the units, so the "
            "summary is what makes a mixed result legible")

    def test_the_lifted_comment_does_not_still_claim_the_old_default(self):
        """Caught in review of this very change, which is why it is pinned.

        The argument for `_OPS_BUILD_MODE_KEYS` was written when the default
        was a literal `["post"]` for every engine, and it said so in those
        words. Moving it to module scope carried the sentence along, leaving
        a comment that contradicted the function immediately below it -- the
        same failure this series has hit repeatedly, a claim in a comment
        outliving the code it described.
        """
        src = _cli_source()
        stale = 'The default is `["post"]` for Percona'
        assert stale not in src, (
            "the default is now read from the engine's capability, so this "
            "sentence is false; `_ops_build_modes` is the behaviour it has "
            "to agree with")
        assert "The default WAS the literal" in src, (
            "the correction has to say what changed, not merely drop the "
            "false clause, or the next reader cannot tell which default a "
            "record came from")

    def test_the_loop_does_not_abort_on_the_first_failure(self):
        """Structural, because it is a property of the control flow rather
        than of any string: units are independent measurements, and stopping
        the loop would have discarded the successful `incremental` unit that
        proved which half was broken. `--fail-fast` stays phase-level, which
        is asserted by the absence of any `break` in this function.

        The one `continue` is the checkpoint skip, which is the opposite
        concern -- work already done.
        """
        fn = next(node for node in ast.parse(_cli_source()).body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_run_unit")
        breaks = [n for n in ast.walk(fn) if isinstance(n, ast.Break)]
        continues = [n for n in ast.walk(fn) if isinstance(n, ast.Continue)]
        print(f"break nodes = {len(breaks)}, continue nodes = {len(continues)}")
        assert not breaks, (
            "a break in the ops unit loop would abandon the remaining units, "
            "which is how the successful measurement would have been lost")
        assert len(continues) == 1, (
            "expected exactly the checkpoint skip; a second continue needs "
            "its own reasoning here")
