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

SINCE THEN, 2026-11-03: at the current pin (9.7.2-2, x86-64-v3) the same
shape built 25 times out of 25 (ALTER 3.4-5.8 s, nothing in the error log, no
core), and a 60,000-row post/incremental comparison gave the same index
quality. So the capability is now TRUE for Percona: profiles that name no mode
get `post` again, and `config/profiles/percona.yml` runs both modes. What this
file still protects is the mechanism -- the default comes from the engine and
not from a literal, an engine that declares the build broken (VillageSQL)
still gets `incremental`, and an explicit `post` against a false declaration
is honoured with a warning. The churn repair is NOT restored by this flip; it
is a separate key, pinned by `tests/test_e13_percona_churn_repair_is_a_noop.py`.

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

    def test_the_capability_is_declared_true_at_the_current_pin(self):
        caps = _percona_cfg().get("capabilities", {}) or {}
        assert CAPABILITY in caps, (
            f"{CAPABILITY} is the engine's declaration of whether its bulk "
            f"vector index build works at this pin; the ops loop reads it to "
            f"pick the default build mode, and it is declared explicitly "
            f"rather than left to the absent-reads-true default so the "
            f"history beside it has a key to sit next to")
        assert caps[CAPABILITY] is True
        # And the repair, a different statement, is NOT restored with it.
        assert caps.get("vector_index_replace") is False

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
        window = text[max(0, head - 4500):head]
        for needle in ("ADD VECTOR KEY", "ALGORITHM=", "Lost connection",
                       "incremental", "percona-20261015-203911",
                       "9.7.1-1", "9.7.2-2", "ATTEMPTS BUILT", "3.4-5.8 s",
                       "core.%p", "percona-build-modes", "0.9714",
                       "ann phase"):
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
        # villagesql84 is the same extension on the 8.4 server line and
        # declares the same, for the same storage ABI.
        assert declared == {"percona": True, "villagesql": False,
                            "villagesql84": False}, declared


# ---------------------------------------------------------------------------
# The default, which is the part that was crashing
# ---------------------------------------------------------------------------

class TestTheDefaultComesFromTheEngineAndNotFromALiteral:

    def test_a_profile_that_names_no_mode_gets_the_mode_that_completes(self):
        from orchestrator.cli import _ops_build_modes
        cfg = _percona_cfg()
        honest = _ops_build_modes("percona", {"ops": {}}, cfg)
        mutant = _ops_build_modes("percona", {"ops": {}},
                                  _with_capability(cfg, False))
        print(f"HONEST (capability true, as shipped) = {honest}")
        print(f"MUTANT (capability false)            = {mutant}")
        assert honest != mutant, (
            "DEAD PROBE: the default did not move when the capability was "
            f"flipped, so this test is not measuring the fix. Both arms "
            f"returned {honest}")
        assert honest == ["post"], (
            "with the bulk build declared working at this pin, the default "
            f"must be the engine's idiomatic build; got {honest}")
        assert mutant == ["incremental"], (
            "a profile naming no mode must not get a mode the engine declares "
            f"broken; got {mutant}")

    def test_the_default_is_read_through_the_capability_for_every_profile(self):
        """Not only the Percona-only profile: the crash was in the profiles
        that say nothing, which is most of them -- so they are the ones that
        must follow the declaration both ways.
        """
        from orchestrator.cli import _ops_build_modes
        from orchestrator.config import load_profile
        cfg = _percona_cfg()
        broken = _with_capability(cfg, False)
        for name in ("main", "smoke", "dev", "full", "quick",
                     "tuned-complete"):
            profile = load_profile(name)
            modes = _ops_build_modes("percona", profile, cfg)
            mutant = _ops_build_modes("percona", profile, broken)
            print(f"{name}: percona build modes = {modes} "
                  f"(capability false: {mutant})")
            assert modes == ["post"], (
                f"profile {name!r} names no mode and should get Percona's "
                f"declared-working bulk build; got {modes}")
            assert mutant == ["incremental"], (name, mutant)

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
        # The capability flipped back to false: the case the warning exists
        # for, now that the shipped config says the build works.
        modes = _ops_build_modes("percona", profile,
                                 _with_capability(_percona_cfg(), False))
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
        modes = _ops_build_modes("percona", profile, _percona_cfg())
        warning = capsys.readouterr().err
        print(f"returned = {modes}, stderr = {warning!r}")
        assert modes == ["post"]
        assert warning == "", (
            "a warning on a working bulk build would train the operator to "
            f"ignore it: {warning!r}")


# ---------------------------------------------------------------------------
# The profile, and its agreement with the estimator
# ---------------------------------------------------------------------------

class TestTheProfileRunsBothModesAgain:

    def test_the_percona_profile_names_both_modes(self):
        """It was cut to `[incremental]` while the bulk build killed the
        server, with a comment saying to restore both when it was fixed.
        """
        from orchestrator.cli import _ops_build_modes
        from orchestrator.config import load_profile
        modes = load_profile("percona")["ops"]["percona_build_modes"]
        assert modes == ["post", "incremental"], modes
        assert _ops_build_modes("percona", load_profile("percona"),
                                _percona_cfg()) == ["post", "incremental"]

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

    def test_the_profile_records_why_and_what_was_not_restored(self):
        """The list was narrowed and widened again; the note beside it has to
        say both, point at the evidence in the engine config, and say that
        restoring `post` did not restore the churn repair.
        """
        with open(os.path.join(REPO, "config", "profiles",
                               "percona.yml")) as fh:
            text = fh.read()
        head = text.index("percona_build_modes: [post, incremental]")
        window = text[max(0, head - 1800):head]
        assert CAPABILITY in window, (
            "the profile must point at the engine's declaration, so the "
            "evidence lives in one place")
        for needle in ("post", "incremental", "OLD PIN",
                       "vector_index_replace"):
            assert needle in window, needle
        assert "KILLS THE SERVER AT THIS PIN" not in text


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


# ---------------------------------------------------------------------------
# The report says which build each path measured
# ---------------------------------------------------------------------------

class TestTheReportSaysTheAnnCurveIsIncremental:
    """With `post` now Percona's ops default, the ops records describe a bulk
    build while every ann recall curve is still an incremental one (the ann
    overlay module has no `build_mode`). §3's build-mode rows read only the
    ops `index_build` records, so without a row of its own the report would
    say "Percona Server bulk-builds after load" beside curves that were not.
    """

    @staticmethod
    def _rows(*modes_seen, ann=True):
        from report.render import _build_mode_rows
        build = [{"engine": "percona", "phase": "index_build",
                  "build_mode": mode,
                  "extra": {"separable_build": mode == "post"}}
                 for mode in modes_seen]
        return _build_mode_rows({"build": build,
                                 "ann_engines": ["percona"] if ann else []})

    @staticmethod
    def _text(rows):
        return " ".join(" ".join(r) for r in rows)

    def test_a_both_modes_run_still_discloses_the_incremental_ann_build(self):
        """`percona-build-modes` puts Percona in `either`, not `bulk` -- and
        that is the run in which a reader compares modes against the curve.
        Found in review: the row read `bulk` alone and went missing there."""
        honest = self._text(self._rows("post", "incremental"))
        mutant = self._text(self._rows("incremental"))
        print(f"HONEST (both modes)       = {honest!r}")
        print(f"MUTANT (incremental only) = {mutant!r}")
        assert "recall curves are an incremental build" in honest
        assert "some of its ops units bulk-built" in honest
        assert "recall curves are an incremental build" not in mutant

    def test_an_ops_only_report_has_no_curve_to_disclose(self):
        """An ops-only report has no recall curve, so "its recall curves are
        an incremental build" would describe something the report does not
        contain. Found in review."""
        honest = self._text(self._rows("post", ann=False))
        mutant = self._text(self._rows("post", ann=True))
        print(f"HONEST (no ann records) = {honest!r}")
        print(f"MUTANT (ann records)    = {mutant!r}")
        assert "recall curves" not in honest
        assert "recall curves are an incremental build" in mutant

    def test_a_post_only_run_discloses_the_incremental_ann_build(self):
        honest = self._rows("post")
        mutant = self._rows("incremental")
        honest_text = " ".join(" ".join(r) for r in honest)
        mutant_text = " ".join(" ".join(r) for r in mutant)
        print(f"HONEST (ops built post)        = {honest_text!r}")
        print(f"MUTANT (ops built incremental) = {mutant_text!r}")
        assert "recall curves are an incremental build" in honest_text
        assert "recall curves are an incremental build" not in mutant_text, (
            "an incremental ops unit matches the ann build; no disclosure is "
            "owed and one would be noise")
