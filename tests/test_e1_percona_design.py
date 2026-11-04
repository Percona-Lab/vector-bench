"""Guards for the Percona Server addition designed in docs/09-adding-percona-server.md.

This file holds the checkable half of that document. It exists because three fix
rounds in this series were spent on prose alone — "code right, prose wrong" — and
a claim in a document is part of the deliverable.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
They cannot establish anything about Percona Server. No server is reachable from
a test run, and a test asserting "Percona requires a BIGINT UNSIGNED primary key"
would only be asserting that a document contains a sentence. Every claim about
the engine is sourced in docs/09 to a named symbol or MTR test on the pinned
SHA, in prose, and is settled by reading that tree, not here. docs/09 §0 is the
citation policy and the block below "Citations: symbols, never line numbers" is
what holds the document to it.

What they do establish is that the artefacts agree with each other and with the
document: that the DDL the overlay module emits is the DDL the document says it
emits, that docker/percona/ has the shape the other engines' directories have,
that init.sql does not carry the one line that would stop the container booting,
and that the arithmetic in docs/09 §3 reproduces from the formula it cites.

The module is loaded and CALLED rather than grepped. `tests/test_config.py`'s
existing overlay tests read module source as text because importing an overlay
module needs an upstream ann-benchmarks package that is not in this repo;
`_load_overlay()` below supplies the one missing piece so these tests can assert
on behaviour instead of on substrings. That matters here specifically: the
difference between MariaDB's DDL and Percona's is a handful of tokens, and a
substring test for `"BIGINT UNSIGNED"` would pass against a file that also still
said `id INT PRIMARY KEY` somewhere.

Each negative assertion below is paired with a positive one that proves the
needle is findable, because a negative assertion passes just as happily against
an empty string as against a correct file.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import types

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER = os.path.join(VB_ROOT, "docker")
OVERLAY = os.path.join(VB_ROOT, "overlay", "ann-benchmarks")
ALGORITHMS = os.path.join(OVERLAY, "ann_benchmarks", "algorithms")
PERCONA_DOC = os.path.join(VB_ROOT, "docs", "09-adding-percona-server.md")


def _doc_09() -> str:
    """docs/09 as text, for the citation guards.

    A function and not the `doc` fixture: the citation pipeline below is also
    run against synthetic documents, so it takes text rather than reading the
    file, and the helpers that do read it need a module-level reader.
    """
    return open(PERCONA_DOC, encoding="utf-8").read()

# The SHA docs/09 pins. Repeated here so that a test can check the document, the
# module docstring and this file all name the same commit -- the value is the
# whole point of pinning, and three copies that can drift is how a pin stops
# meaning anything.
PINNED_SHA = "56124b03fdc5954e2401ab4d2c7bf7cb3a767158"


# ---------------------------------------------------------------------------
# Loading the overlay module
# ---------------------------------------------------------------------------

def _load_overlay():
    """Import `vb_mysql` and the percona module with the upstream piece stubbed.

    `ann_benchmarks/vb_mysql.py` does `from .algorithms.base.module import
    BaseANN`, and `algorithms/base/` belongs to upstream ann-benchmarks, which
    this repo clones into `work/` at run time rather than vendoring. So the
    overlay cannot be imported as-is from a test.

    The stub is one empty class. Nothing in `Dialect.create_table`,
    `_query_sql` or `_verify_plan` touches `BaseANN`, so stubbing it does not
    stand in for any behaviour under test -- which is the trap this repo calls
    "fixtures supplying the thing under test". If a future test here needed
    `BaseANN`'s own behaviour it would have to clone upstream instead.

    Returns (vb_mysql_module, percona_module).
    """
    cached = getattr(_load_overlay, "_cache", None)
    if cached is not None:
        return cached

    def _pkg(name: str, path: str | None = None) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        return mod

    ab = _pkg("ann_benchmarks", os.path.join(OVERLAY, "ann_benchmarks"))
    algs = _pkg("ann_benchmarks.algorithms", ALGORITHMS)
    ab.algorithms = algs
    base = _pkg("ann_benchmarks.algorithms.base")
    algs.base = base

    base_module = types.ModuleType("ann_benchmarks.algorithms.base.module")

    class BaseANN:  # the whole stub
        pass

    base_module.BaseANN = BaseANN
    sys.modules["ann_benchmarks.algorithms.base.module"] = base_module
    base.module = base_module

    def _load(dotted: str, path: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(dotted, path)
        assert spec and spec.loader, f"cannot load {path}"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    # Importing from `overlay/` would otherwise write `__pycache__/` into the
    # overlay tree, and `overlay/ann-benchmarks/` is copied wholesale over a
    # disposable clone on every `run` and `render`. A .pyc compiled by the
    # host's interpreter has no business travelling into the bench image
    # beside the source it was built from, so bytecode is off for these loads.
    was_writing = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        vb = _load("ann_benchmarks.vb_mysql",
                   os.path.join(OVERLAY, "ann_benchmarks", "vb_mysql.py"))
        ab.vb_mysql = vb
        percona_pkg = _pkg("ann_benchmarks.algorithms.percona",
                           os.path.join(ALGORITHMS, "percona"))
        algs.percona = percona_pkg
        percona = _load("ann_benchmarks.algorithms.percona.module",
                        os.path.join(ALGORITHMS, "percona", "module.py"))
    finally:
        sys.dont_write_bytecode = was_writing

    _load_overlay._cache = (vb, percona)
    return vb, percona


@pytest.fixture(scope="module")
def overlay():
    return _load_overlay()


def test_the_loader_does_not_write_bytecode_into_the_overlay():
    """`overlay/ann-benchmarks/` is copied wholesale over a disposable clone on
    every `run` and `render`, so a host-compiled `.pyc` sitting beside the
    source would travel into the bench image. Importing from there is this
    file's doing, so suppressing the write is this file's responsibility.

    Verified by running the full suite from a clean tree and confirming no
    `.pyc` appeared anywhere under `overlay/`. This test guards the mechanism
    rather than the directory state, because a directory assertion would fail
    on a stale file left by something other than the suite and would then look
    like a broken checkout.
    """
    import inspect
    source = inspect.getsource(_load_overlay)
    assert "sys.dont_write_bytecode = True" in source
    # Restored afterwards rather than left on: leaving a global interpreter
    # flag flipped would change the behaviour of every test that runs after
    # this one.
    assert "sys.dont_write_bytecode = was_writing" in source
    assert "finally:" in source


def _strip_sql_comments(text: str) -> str:
    """Drop `--` line comments, keeping only statements.

    docs/09 and docker/percona/init.sql both discuss `mysql_native_password` at
    length in comments, so a test asking whether the file "contains" the string
    would fail on a correct file. What matters is whether the server is ever
    asked for the plugin, which means the statements.
    """
    out = []
    for line in text.splitlines():
        head = line.split("--", 1)[0]
        if head.strip():
            out.append(head)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# docker/percona/ shape
# ---------------------------------------------------------------------------

class TestDockerDirectoryShape:
    """docker/percona/ must look like the other MySQL-family engines' directories.

    The expected shape is DERIVED from docker/mariadb/ and docker/alisql/ rather
    than spelled out, so that a change to the pattern fails here instead of
    leaving the newest engine as the only one following the old one.
    """

    @staticmethod
    def _kinds(engine: str) -> set:
        """The file roles present, with the engine name factored out."""
        out = set()
        for name in os.listdir(os.path.join(DOCKER, engine)):
            out.add(name.replace(engine, "<engine>"))
        return out

    def test_the_directory_exists(self):
        assert os.path.isdir(os.path.join(DOCKER, "percona"))

    def test_same_file_roles_as_the_two_engines_it_is_modelled_on(self):
        mariadb = self._kinds("mariadb")
        alisql = self._kinds("alisql")
        # Non-vacuity: the two templates must agree with each other, or
        # "matches the pattern" would not be a statement about anything.
        assert mariadb == alisql, (
            "the two MySQL-family docker dirs no longer share a shape; "
            f"mariadb={sorted(mariadb)} alisql={sorted(alisql)}")
        assert mariadb == {"Dockerfile", "entrypoint-<engine>.sh", "init.sql"}
        assert self._kinds("percona") == mariadb

    def test_dockerfile_declares_the_same_build_stages(self):
        """A missing `bench` stage fails only when the ann pass is launched."""
        def stages(engine):
            src = open(os.path.join(DOCKER, engine, "Dockerfile")).read()
            return set(re.findall(r"^FROM\s+\S+\s+AS\s+(\w+)", src,
                                  re.MULTILINE))
        assert stages("alisql") == {"builder", "runtime", "bench"}
        assert stages("percona") == stages("alisql")

    def test_the_dockerfile_copies_the_entrypoint_that_exists(self):
        """A COPY of a filename nobody wrote is a build failure, late."""
        src = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        copied = re.findall(r"^COPY\s+(entrypoint-\S+)", src, re.MULTILINE)
        assert copied == ["entrypoint-percona.sh"]
        assert os.path.isfile(os.path.join(DOCKER, "percona", copied[0]))

    def test_the_entrypoint_offers_the_same_subcommands(self):
        """`vb_mysql._start_server` runs `vb-entrypoint server`; the ops path and
        docs/03 use `init` and `client`."""
        def subcommands(engine):
            path = os.path.join(DOCKER, engine, f"entrypoint-{engine}.sh")
            src = open(path).read()
            body = src.split("case ", 1)[-1]
            return set(re.findall(r"^\s{2}(\w+)\)", body, re.MULTILINE))
        assert {"server", "init", "client"} <= subcommands("alisql")
        assert subcommands("percona") == subcommands("alisql")

    def test_the_socket_path_agrees_between_dockerfile_and_entrypoint(self):
        """Two files name this path; a readiness probe waits on one of them.

        If they disagree the server comes up on one socket and everything that
        waits for it watches the other, which presents as a start-up timeout
        rather than as a typo.
        """
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        entrypoint = open(os.path.join(DOCKER, "percona",
                                       "entrypoint-percona.sh")).read()
        from_env = re.search(r"VB_SOCKET=(\S+)", dockerfile)
        from_default = re.search(r'SOCKET="\$\{VB_SOCKET:-([^}"]+)\}"',
                                 entrypoint)
        assert from_env and from_default
        assert from_env.group(1) == from_default.group(1)
        # And it must be engine-specific, or two servers sharing a data mount
        # would collide. Derived from the other engines' convention.
        assert from_env.group(1).endswith("/percona.sock")


class TestBuildArgConventions:
    """The Dockerfile has to receive the pin through the ARG the build script
    will pass, and the pin has to be under the key the orchestrator reads.

    Both of these were wrong in the first draft of docs/09. A mismatched ARG
    fails quietly: the image builds and carries `VB_ENGINE_TAG=unknown`. The
    wrong config key fails loudly, before any build: `_source_tag()` returns
    the empty string and `prepare_percona` dies, because it passes `""` as
    the default and `die`s on an empty value -- unlike the other `prepare_*`
    functions, which fall back to a hard-coded tag. An earlier revision of
    this docstring said this engine silently falls back too; it does not.
    """

    def test_the_dockerfile_arg_follows_the_build_script_convention(self):
        """build-images.sh:129-131 passes `<ENGINE>_TAG=${tag}`."""
        script = open(os.path.join(VB_ROOT, "scripts",
                                   "build-images.sh")).read()
        passed = dict(re.findall(r"^\s+(\w+)\)\s+bargs\+=\(--build-arg "
                                 r"\"(\w+_TAG)=", script, re.MULTILINE))
        # Non-vacuity: the convention must be visible for the existing engines.
        assert passed.get("mariadb") == "MARIADB_TAG"
        assert passed.get("alisql") == "ALISQL_TAG"
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        args = set(re.findall(r"^ARG (\w+_TAG)", dockerfile, re.MULTILINE))
        assert args == {"PERCONA_TAG"}, args

    def test_the_arg_is_wired_to_the_env_var_the_manifest_reads(self):
        """`ops_pass.py`'s `VB_ENGINE_TAG` puts source.tag into the
        environment the manifest reads; the image sets it too, and an ARG
        declared but never used leaves it `unknown`.

        A SYMBOL AND NOT A NUMBER. This used to cite a line number in
        `ops_pass.py`, and the change that converted it is an edit above that
        line -- which is #63's whole argument for the conversion, made again
        against a citation #63 did not reach. It survived #63 because it is
        bare docstring prose with a row in neither `CITED_SYMBOLS` nor the
        `CITED_IN_REPO` table #63 deleted, so nothing failed and nothing would
        have.

        NO NUMBERS IN THIS PARAGRAPH, deliberately: the earlier version named
        both the old line and the new one, and the new one went stale inside
        the same round -- in a docstring whose subject is line numbers going
        stale. A figure that needs maintaining to stay true does not belong in
        the explanation of why figures should not be cited.
        """
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        assert "VB_ENGINE_TAG=${PERCONA_TAG}" in dockerfile

    def test_the_orchestrator_has_no_source_sha_key(self):
        """docs/09 §4.3 tells the follow-on task to write `source: tag:` with a
        SHA as the value. That instruction is only correct while this holds."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "engines.py")).read()
        fn = source.split("def _source_tag", 1)[1].split("\ndef ", 1)[0]
        assert 'source.get("tag")' in fn          # needle findable
        assert 'source.get("sha")' not in fn

    def test_the_document_says_tag_and_not_sha(self):
        doc = open(PERCONA_DOC).read()
        skeleton = doc.split("```yaml", 1)[1].split("```", 1)[0]
        assert f"tag: {PINNED_SHA}" in skeleton
        assert f"sha: {PINNED_SHA}" not in skeleton

    def test_build_images_refuses_an_engine_it_has_no_case_for(self):
        """Which is why the missing case is a clear failure and not a silent
        one -- worth knowing before someone goes looking for a subtler bug."""
        script = open(os.path.join(VB_ROOT, "scripts",
                                   "build-images.sh")).read()
        assert 'die "engine' in script and "has no known build-arg" in script


class TestEverySubmoduleIsMitigated:
    """The build blocker review found: four submodules, four flags needed.

    `prepare-sources.sh` ships a `git archive`, which omits submodule paths, so
    cmake must be told not to descend into any of them. The first revision of
    the Dockerfile enumerated all four correctly, said "two", and passed three
    flags -- `extra/libkmip` was left out and cmake configure would have failed.

    These guards are self-consistency checks on the Dockerfile's own note: the
    note maps each submodule to the flag that handles it, and the flag has to
    appear in the cmake invocation. A count stated in prose and a list that
    disagrees with it is exactly what happened.
    """

    @pytest.fixture(scope="class")
    def dockerfile(self):
        return open(os.path.join(DOCKER, "percona", "Dockerfile")).read()

    @pytest.fixture(scope="class")
    def mapping(self, dockerfile):
        """submodule path -> cmake flag, parsed out of note 2's table."""
        note = dockerfile.split("NO SUBMODULE FETCH", 1)[1].split(
            "BOOST NEEDS NO FLAG", 1)[0]
        pairs = re.findall(r"^#\s+(\S+/\S+)\s+->\s+(-D\S+)", note,
                           re.MULTILINE)
        return dict(pairs)

    def test_all_four_submodules_are_named_with_a_flag_each(self, mapping):
        assert set(mapping) == {
            "storage/rocksdb/rocksdb",
            "extra/coredumper",
            "extra/jwt-cpp",
            "extra/libkmip",
        }, mapping

    def test_every_mapped_flag_is_actually_passed_to_cmake(self, dockerfile,
                                                           mapping):
        """The half that was wrong: the note listed four, cmake got three."""
        invocation = dockerfile.split("cmake /src/source", 1)[1].split(
            "make -j", 1)[0]
        for submodule, flag in sorted(mapping.items()):
            assert flag in invocation, f"{submodule} -> {flag} not passed"

    def test_libkmip_is_closed_by_either_accepted_spelling(self, mapping):
        """`extra/libkmip` must be gated, by one of the two spellings that work.

        BOTH work, and an earlier version of this test asserted otherwise. It
        required `startswith("-DWITHOUT_COMPONENT_KEYRING_KMIP")` and its
        docstring said `-DWITH_...=OFF` "would not close it" -- so a later
        engineer switching to the WITH_ spelling would have got a red test
        telling them something untrue, which is worse than no test at all.

        Measured under cmake 3.28.3, with CMakeLists.txt:2295-2298 transcribed
        verbatim into a standalone project:

            (no flags)                          -> WOULD_ADD_SUBDIRECTORY
            -DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED
            -DWITH_COMPONENT_KEYRING_KMIP=0     -> SKIPPED
            -DWITH_COMPONENT_KEYRING_KMIP=NO    -> SKIPPED
            -DWITH_COMPONENT_KEYRING_KMIP=ON    -> WOULD_ADD_SUBDIRECTORY
            -DWITHOUT_COMPONENT_KEYRING_KMIP=1  -> SKIPPED
            -DWITHOUT_COMPONENT_KEYRING_KMIP=0  -> SKIPPED

        The old reasoning conflated `DEFINED WITH_X` with `WITH_X`: defining it
        is exactly what makes the AND false, and the OR's second operand is
        `WITH_X` as a boolean, where OFF/0/NO are CMake false constants. The
        component's own guard keys off the WITH_ spelling too --
        components/keyrings/keyring_kmip/CMakeLists.txt:28 is
        `IF(NOT WITH_COMPONENT_KEYRING_KMIP) RETURN()`.

        What this asserts is the part that is load-bearing: the submodule is
        gated by a recognised spelling, and specifically not by the one value
        that re-opens it. The Dockerfile may prefer WITHOUT_ -- it does, because
        that gate tests definedness and never the value, so it cannot be broken
        by a value spelling -- but that is a preference, and a preference does
        not belong in an assertion.
        """
        flag = mapping["extra/libkmip"]
        symbol, _, value = flag.partition("=")
        assert symbol in ("-DWITH_COMPONENT_KEYRING_KMIP",
                          "-DWITHOUT_COMPONENT_KEYRING_KMIP"), flag
        if symbol == "-DWITH_COMPONENT_KEYRING_KMIP":
            assert value.upper() in ("OFF", "0", "NO", "FALSE", "N"), flag

    def test_the_note_does_not_claim_the_WITH_spelling_fails(self, dockerfile):
        """Polarity, on the prose this time.

        The falsehood lived in two places and the test enforced it. Deleting it
        from the assertion is not enough -- the note has to stop saying it, or
        the next reader believes the note and the test's silence looks like
        agreement.
        """
        note = dockerfile.split("NO SUBMODULE FETCH", 1)[1].split(
            "BOOST NEEDS NO FLAG", 1)[0]
        for wrong in ("would NOT help",
                      "so only the WITHOUT_ spelling closes it"):
            assert wrong not in note, wrong
        # ...and it carries the measurement instead of a replacement argument.
        assert "cmake 3.28.3" in note
        assert "-DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED" in note

    def test_the_note_does_not_undercount_its_own_list(self, dockerfile,
                                                       mapping):
        """The actual defect: prose said "two", the list had four.

        Anchored on ONE deterministic phrase in note 2's heading, "TAKES <n>
        FLAGS", and not on number words anywhere in the note. The first version
        of this test scanned the whole note for number words and compared the
        mapping length against the SET of them -- and the note's prose contains
        two, three, four and one, so `len(mapping) in {1,2,3,4}` held for any
        plausible mapping. Measured: with libkmip deleted from the note the
        mapping was 3, the word set was {1,2,3,4}, and the test passed.

        That is membership-not-value, which is on this repo's list of measured
        vacuity shapes. A count assertion has to compare against a single
        stated number.
        """
        heading = dockerfile.split("NO SUBMODULE FETCH", 1)[1].split("\n", 1)[0]
        words = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5}
        stated = re.search(r"TAKES (" + "|".join(words) + r") FLAGS", heading)
        assert stated, (
            f"note 2's heading states no flag count: {heading!r}")
        assert words[stated.group(1)] == len(mapping), (
            f"heading says {stated.group(1)} flags, mapping has {len(mapping)}")

    def test_no_submodules_key_is_proposed_for_the_engine_config(self):
        """The alternative path -- listing them in percona.yml's
        `source.submodules:` the way mariadb.yml lists libmariadb -- would make
        the build fetch from three more hosts. The document rules it out, so
        its skeleton must not quietly contain one.

        Comments are stripped first, because the skeleton *explains* the
        absence in a comment and the first version of this test matched its own
        explanation. That is the inverse of the usual vacuity: a negative
        assertion failing on prose that agrees with it.
        """
        doc = open(PERCONA_DOC).read()
        skeleton = doc.split("```yaml", 1)[1].split("```", 1)[0]
        keys = "\n".join(ln.split("#", 1)[0] for ln in skeleton.splitlines())
        assert "submodules:" not in keys
        # ...and the explanation is still there, so this is not passing because
        # the skeleton went silent on the question.
        assert "submodules:" in skeleton
        # Non-vacuity: mariadb.yml really does use that key, so the spelling
        # this is looking for is the right one, and the stripper does not eat
        # a real key.
        mariadb = open(os.path.join(VB_ROOT, "config", "engines",
                                    "mariadb.yml")).read()
        mariadb_keys = "\n".join(ln.split("#", 1)[0]
                                 for ln in mariadb.splitlines())
        assert "submodules:" in mariadb_keys


class TestGraphMemoryFallback:
    """The by-hand `docker run` must not get the 1 GiB compiled-in default, and
    the orchestrator's value must always win.

    docker/alisql/init.sql:13-15 sets the precedent -- "the server flag covers
    startup, this covers a server someone restarted without it" -- but the
    mechanism cannot be copied: --init-file runs AFTER the server flags, so a
    `SET GLOBAL` there would overwrite a measurement parameter and put a value
    in the manifest that was not in force.
    """

    @pytest.fixture(scope="class")
    def entrypoint(self):
        return open(os.path.join(DOCKER, "percona",
                                 "entrypoint-percona.sh")).read()

    def test_the_fallback_exists(self, entrypoint):
        assert "--innodb-hnsw-max-memory=0" in entrypoint

    def test_it_is_conditional_and_not_unconditional(self, entrypoint):
        """An unconditional append would override the resource pass."""
        start = entrypoint.split("start_server()", 1)[1]
        appended = start.split("--innodb-hnsw-max-memory=0", 1)[0]
        # The append sits inside a guard on whether one was already given.
        assert "have_max_memory" in appended
        assert re.search(r"if\s*\(\(\s*!\s*have_max_memory\s*\)\)", appended)

    def test_the_guard_matches_both_mysqld_spellings(self, entrypoint):
        """`--flag=value` and `--flag value` are both valid, and underscores and
        hyphens are interchangeable in mysqld option names. A guard matching
        only one spelling would silently double-specify the option."""
        start = entrypoint.split("start_server()", 1)[1]
        assert "${a//_/-}" in start
        assert "--innodb-hnsw-max-memory|--innodb-hnsw-max-memory=*" in start

    def test_init_sql_does_not_set_it(self):
        """The thing that must NOT be done, paired with the AliSQL precedent it
        would have been copied from."""
        alisql = _strip_sql_comments(
            open(os.path.join(DOCKER, "alisql", "init.sql")).read())
        percona = _strip_sql_comments(
            open(os.path.join(DOCKER, "percona", "init.sql")).read())
        # Non-vacuity: AliSQL really does repeat its global in the init file.
        assert "SET GLOBAL" in alisql
        assert "SET GLOBAL" not in percona
        assert "innodb_hnsw_max_memory" not in percona

    def test_init_sql_explains_the_asymmetry_rather_than_omitting_it(self):
        """A silent omission reads as an oversight; the next person adds it."""
        src = open(os.path.join(DOCKER, "percona", "init.sql")).read()
        assert "init.sql:13-15" in src
        assert "entrypoint-percona.sh" in src


class TestExplainFormatIsPinned:
    """`vector_ann` lives in the `type` column of TRADITIONAL EXPLAIN only.

    `explain_format` defaults to TRADITIONAL, so a bare EXPLAIN works today, but
    the variable is reachable through VB_SERVER_ARGS and under TREE or JSON the
    marker vanishes -- a false negative about a server that DID use the index.
    No MTR test could catch it, because MTR output is always TRADITIONAL.
    """

    def test_the_module_pins_the_format(self, overlay):
        _, percona = overlay
        body = open(os.path.join(ALGORITHMS, "percona", "module.py")).read()
        fn = body.split("def _verify_plan", 1)[1].split("\n    def ", 1)[0]
        # The executable line, not the docstring: look at the execute() call.
        call = fn.split("self._cur.execute(", 1)[1]
        assert '"EXPLAIN FORMAT=TRADITIONAL ' in call

    def test_the_shared_implementation_checks_the_name_and_not_the_access_type(
            self, overlay):
        """Non-vacuity: proves the module differs rather than matching.

        **Re-aimed from the EXPLAIN FORMAT at the access type, because the
        format version was a landmine.** It asserted
        `"FORMAT=TRADITIONAL" not in fn` for the SHARED reader, which pinned
        the ann path to *not* being hardened the way its ops sibling was -- so
        a later author hardening it would have read this failure as "you broke
        something". That is the shape of guard this file exists to remove, not
        to add.

        The load-bearing difference is the access type, and it cannot be
        acquired by accident: the shared reader asks only whether
        `dialect.index_name` appears anywhere in the joined plan text, which
        Percona's own recorded output satisfies for a full table scan.

        **Pinning the format in the shared reader would be safe and is not
        forbidden by anything here.** It is simply unnecessary: a name-only
        check is format-insensitive -- the index name appears in TREE and JSON
        output too -- so the keyword buys that reader nothing. It is pinned in
        the ops driver (`harness/drivers/mysql_family.py`) because that one
        tests `Dialect.plan_marker`, a value in the tabular `type` column that
        no other format has, and in `percona/module.py` for the same reason.
        All three servers the shared reader serves accept the keyword, so the
        choice is about need rather than risk.
        """
        shared = open(os.path.join(OVERLAY, "ann_benchmarks",
                                   "vb_mysql.py")).read()
        fn = shared.split("def _verify_plan", 1)[1].split("\n    def ", 1)[0]
        assert '"EXPLAIN ' in fn                      # needle findable
        # The shared reader does NOT look for an access type. This is what
        # makes the Percona override necessary rather than cosmetic.
        assert "vector_ann" not in fn
        assert "PLAN_ACCESS_TYPE" not in fn
        # ...and the Percona module does, which is the other half of the pair.
        percona = open(os.path.join(ALGORITHMS, "percona",
                                    "module.py")).read()
        own = percona.split("def _verify_plan", 1)[1].split("\n    def ", 1)[0]
        assert "PLAN_ACCESS_TYPE in plan" in own


class TestInitSqlDoesNotAskForARemovedPlugin:
    """The one line that must not be copied from docker/alisql/init.sql.

    docs/09 §7 item 2 cites mysql-test/r/mysql_native_password_removed.result:4-5
    on the pinned SHA: `CREATE USER ... IDENTIFIED WITH mysql_native_password`
    returns `ERROR HY000: Plugin 'mysql_native_password' is not loaded`. An
    --init-file statement that errors is fatal to start-up, so copying AliSQL's
    line would make the container fail to boot.
    """

    NEEDLE = "mysql_native_password"

    def _statements(self, engine: str) -> str:
        path = os.path.join(DOCKER, engine, "init.sql")
        return _strip_sql_comments(open(path).read())

    def test_the_needle_is_findable_by_this_test(self):
        """Non-vacuity, twice over.

        The negative assertion below would pass against an empty string, and it
        would also pass if `_strip_sql_comments` were over-eager enough to
        remove everything. AliSQL's init.sql really does request the plugin in a
        statement, so finding it there proves both the needle and the stripper.
        """
        alisql = self._statements("alisql")
        assert self.NEEDLE in alisql
        assert "CREATE USER" in alisql

    def test_percona_init_sql_never_requests_it_in_a_statement(self):
        assert self.NEEDLE not in self._statements("percona")

    def test_percona_init_sql_is_not_vacuously_empty(self):
        """A file whose statements were all deleted would pass the test above."""
        statements = self._statements("percona")
        assert statements.count("CREATE USER") == 2        # '%' and 'localhost'
        assert statements.count("GRANT ALL PRIVILEGES") == 2
        assert "FLUSH PRIVILEGES" in statements
        # The account the rest of the framework connects as.
        assert "'bench'" in statements

    def test_it_does_not_copy_alisqls_vector_enable_statement_either(self):
        """AliSQL's init.sql runs `SET GLOBAL vidx_disabled = OFF` because VIDX
        ships disabled. Percona's index needs no switch, and an unknown variable
        in an --init-file is also fatal."""
        assert "vidx_disabled" in self._statements("alisql")   # needle findable
        assert "vidx_disabled" not in self._statements("percona")


# ---------------------------------------------------------------------------
# The dialect the module actually emits
# ---------------------------------------------------------------------------

class TestEmittedDDL:
    """docs/09 §2.2 and §2.3, checked against the statement the code builds."""

    @pytest.fixture
    def ddl(self, overlay):
        _, percona = overlay
        dialect = percona.PERCONA
        return dialect.create_table(dim=128, m=16, metric="euclidean",
                                    storage_engine="InnoDB")

    def test_primary_key_is_bigint_unsigned(self, ddl, overlay):
        """index_syntax.result:186-204 records four refusals of anything else.

        Paired with the base dialect, which emits the INT key Percona refuses,
        so this asserts a difference rather than the presence of a string.
        """
        vb, percona = overlay
        base = vb.Dialect(name="probe").create_table(
            dim=128, m=16, metric="euclidean", storage_engine="InnoDB")
        assert "id INT PRIMARY KEY" in base
        assert "id BIGINT UNSIGNED PRIMARY KEY" in ddl
        assert "id INT PRIMARY KEY" not in ddl

    def test_the_index_clause_is_percona_spelling(self, ddl):
        assert "VECTOR KEY vi (v) TYPE hnsw (" in ddl
        # MariaDB's spelling must be gone, not merely joined by Percona's.
        assert "DISTANCE=" not in ddl
        assert "M=16" not in ddl          # Percona writes `M = 16`

    def test_all_three_index_options_are_stated(self, ddl):
        """Omitting one silently takes Percona's default -- M=25 rather than the
        profile's value (vec0vec.h:55). dd_params_resolved.result:8-19 shows the
        server filling in and storing whatever was not said."""
        opts = re.search(r"TYPE hnsw \(([^)]*)\)", ddl)
        assert opts, ddl
        names = {kv.split("=")[0].strip() for kv in opts.group(1).split(",")}
        assert names == {"M", "ef_construction", "metric"}

    def test_the_metric_is_a_bare_identifier(self, ddl):
        """`metric = 'euclidean'` is a parse error -- an option value is an
        identifier or an integer (Docs/percona-hnsw.md:1305-1307)."""
        assert "metric = euclidean" in ddl
        assert "metric = 'euclidean'" not in ddl
        assert 'metric = "euclidean"' not in ddl

    def test_m_and_ef_construction_are_unquoted_integers(self, ddl):
        """`M = 6.0` and `M = "6"` are both parse errors
        (index_syntax.test:356, :363)."""
        opts = re.search(r"TYPE hnsw \(([^)]*)\)", ddl).group(1)
        for pair in opts.split(","):
            name, _, value = (p.strip() for p in pair.partition("="))
            if name in ("M", "ef_construction"):
                assert re.fullmatch(r"\d+", value), pair

    def test_ef_construction_defaults_to_perconas_own_default(self, overlay):
        _, percona = overlay
        assert percona.DEFAULT_EF_CONSTRUCTION == 200       # vec0vec.h:56
        assert percona.PERCONA.ef_construction == 200
        ddl = percona.PERCONA.create_table(128, 16, "euclidean", "InnoDB")
        assert "ef_construction = 200" in ddl

    def test_the_vector_column_is_not_null(self, ddl):
        """An indexed vector column is NOT NULL (search.test:36)."""
        assert re.search(r"v VECTOR\(128\) NOT NULL", ddl)


class TestEmittedQuery:
    """docs/09 §2.4, checked against the SELECT the code builds."""

    @pytest.fixture
    def instance(self, overlay):
        """A PerconaServer with only the attributes `_query_sql` reads.

        Built without `__init__`, which starts a server. The attributes set here
        are the ones the real constructor would set; nothing under test is
        supplied by the fixture -- `_query_sql`'s output is computed, not given.
        """
        _, percona = overlay
        obj = object.__new__(percona.PerconaServer)
        obj.dialect = percona.PERCONA
        obj._metric_fn = "euclidean"
        return obj

    def test_it_is_a_three_argument_distance_call(self, instance):
        """sql/item_create.cc:1403 registers DISTANCE with arity 3."""
        sql = instance._query_sql(10)
        call = re.search(r"DISTANCE\(([^)]*)\)", sql)
        assert call, sql
        args = [a.strip() for a in call.group(1).split(",")]
        assert len(args) == 3, args
        assert args[0] == "v"
        assert args[1] == "%s"
        assert args[2] == "'EUCLIDEAN'"

    def test_mariadbs_per_metric_function_is_gone(self, instance, overlay):
        """Non-vacuity: the shared implementation emits it, so its absence here
        is a difference and not an empty assertion."""
        vb, percona = overlay
        shared = vb.VBMySQLBase._query_sql(instance, 10)
        assert "vec_distance_euclidean(v, %s)" in shared
        assert "vec_distance_" not in instance._query_sql(10)

    def test_exactly_one_bound_parameter(self, instance):
        """The caller passes `(vector,)`. A second placeholder would raise at
        execute time, inside a measured loop."""
        assert instance._query_sql(10).count("%s") == 1

    def test_ascending_order_and_a_limit(self, instance):
        """Both are required for the index to be used at all
        (search.test:49-55: no LIMIT and DESC are listed as shapes that must
        not use the index)."""
        sql = instance._query_sql(37)
        assert "ORDER BY DISTANCE(" in sql
        assert " DESC" not in sql.upper().replace("DESCRIBE", "")
        assert sql.rstrip().endswith("LIMIT 37")

    def test_the_metric_tracks_the_configured_one(self, instance):
        instance._metric_fn = "cosine"
        assert "'COSINE'" in instance._query_sql(10)
        assert "'EUCLIDEAN'" not in instance._query_sql(10)


class TestMetricMapping:
    """docs/09 §2.4: only euclidean and cosine can be index metrics."""

    def test_both_metrics_the_harness_uses_are_mapped(self, overlay):
        _, percona = overlay
        assert percona.PERCONA.metric_names == {"angular": "cosine",
                                                "euclidean": "euclidean"}

    def test_no_metric_that_cannot_be_indexed_is_offered(self, overlay):
        """index_syntax.result:789-808 refuses euclidean_squared, dot and
        manhattan as index metrics with ER_NOT_SUPPORTED_YET. Offering one here
        would build nothing and fall back to a full scan."""
        _, percona = overlay
        offered = set(percona.PERCONA.metric_names.values())
        assert offered.isdisjoint({"euclidean_squared", "dot", "manhattan"})

    def test_mapped_names_are_lowercase_for_the_ddl(self, overlay):
        """The DD stores the canonical lowercase form
        (dd_params_resolved.result:55)."""
        _, percona = overlay
        for value in percona.PERCONA.metric_names.values():
            assert value == value.lower()


class TestEfSearchStatement:
    """docs/09 §1: innodb_hnsw_ef_search, SESSION scope."""

    def test_it_formats_into_a_set_session_statement(self, overlay):
        _, percona = overlay
        stmt = percona.PERCONA.set_ef_search.format(ef_search=123)
        assert stmt == "SET SESSION innodb_hnsw_ef_search = 123"

    def test_it_names_perconas_variable_and_not_another_engines(self, overlay):
        _, percona = overlay
        template = percona.PERCONA.set_ef_search
        assert "innodb_hnsw_ef_search" in template
        for foreign in ("mhnsw_ef_search", "vidx_hnsw_ef_search"):
            assert foreign not in template

    def test_the_template_has_the_placeholder_the_caller_fills(self, overlay):
        """`vb_mysql.set_query_arguments` calls
        `.format(ef_search=...)`; a template without the field silently ignores
        the sweep and measures one operating point many times."""
        _, percona = overlay
        assert "{ef_search}" in percona.PERCONA.set_ef_search


class TestRangeValidation:
    """The module refuses out-of-range M and efConstruction itself.

    Only the raising path is exercised: a valid call reaches
    `super().__init__`, which starts a server.
    """

    def test_ef_construction_bounds_match_the_source(self, overlay):
        _, percona = overlay
        assert (percona.MIN_EF_CONSTRUCTION,
                percona.MAX_EF_CONSTRUCTION) == (1, 4096)   # vec0vec.h:71-72

    def test_m_bounds_match_the_source(self, overlay):
        _, percona = overlay
        assert (percona.MIN_M, percona.MAX_M) == (2, 200)   # vec0vec.h:64-65

    @pytest.mark.parametrize("ef", [0, 4097])
    def test_out_of_range_ef_construction_is_refused(self, overlay, ef):
        _, percona = overlay
        with pytest.raises(ValueError, match="efConstruction"):
            percona.PerconaServer("euclidean", {"M": 16, "efConstruction": ef})

    @pytest.mark.parametrize("m", [1, 201])
    def test_out_of_range_m_is_refused(self, overlay, m):
        _, percona = overlay
        with pytest.raises(ValueError, match="^M must be in"):
            percona.PerconaServer("euclidean", {"M": m})

    def test_the_swept_m_values_are_all_accepted(self, overlay):
        """The grid in the overlay config must be inside the engine's range, or
        a sweep dies partway through."""
        _, percona = overlay
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ALGORITHMS, "percona",
                                               "config.yml")))
        groups = cfg["float"]["any"][0]["run_groups"]["innodb"]["arg_groups"]
        swept = [m for g in groups for m in g["M"]]
        assert swept, "no M values are swept"
        assert all(percona.MIN_M <= m <= percona.MAX_M for m in swept), swept


# ---------------------------------------------------------------------------
# The plan check
# ---------------------------------------------------------------------------

class _FakeCursor:
    """A cursor that returns one canned EXPLAIN row."""

    def __init__(self, row):
        self._row = row

    def execute(self, sql, params=None):
        self._last = sql

    def fetchall(self):
        return [self._row]

    def fetchone(self):
        return (128,)          # VECTOR_DIM(v)


# Two plans, both taken from the server's own recorded output in
# mysql-test/suite/percona_vector/r/optimizer.result on the pinned SHA.
PLAN_USED = (1, "SIMPLE", "t1", None, "vector_ann", "vi", "vi", 0,
             "const", 100, 10.00, "Using where")          # :95-96
PLAN_NOT_USED = (1, "SIMPLE", "t1", None, "ALL", None, None, None,
                 None, 100, 100.00, "Using filesort")     # :127-128
# The plan this guard exists for: the vector index offered and not taken. The
# index name is in `possible_keys`, so any check that only looks for the name
# passes; `key` is another index and the access type is a range scan.
PLAN_OFFERED_NOT_TAKEN = (1, "SIMPLE", "t1", None, "range", "vi,tag_idx",
                          "tag_idx", 4, None, 50, 11.11, "Using where")


class TestPlanCheckIsNotSatisfiedByTheIndexName:
    """docs/09 §4.1: the shared substring check is vacuous on Percona.

    This is the repo's recurring defect shape -- a substring satisfied by
    something other than the thing being asserted -- and it is worth a test
    because the consequence is a full table scan recorded as an ANN search:
    exact results, slowly, which no output distinguishes from "accurate but
    slow".
    """

    @pytest.fixture
    def instance(self, overlay):
        _, percona = overlay
        obj = object.__new__(percona.PerconaServer)
        obj.dialect = percona.PERCONA
        obj._metric_fn = "euclidean"
        return obj

    def _verify(self, instance, plan_row):
        instance._cur = _FakeCursor(plan_row)
        return instance._verify_plan(10)

    def test_it_accepts_the_plan_that_used_the_index(self, instance):
        assert self._verify(instance, PLAN_USED) is True

    def test_it_rejects_the_plain_scan(self, instance):
        assert self._verify(instance, PLAN_NOT_USED) is False

    def test_it_rejects_a_plan_that_only_offered_the_index(self, instance):
        """The whole point of the override."""
        assert self._verify(instance, PLAN_OFFERED_NOT_TAKEN) is False

    def test_the_shared_check_would_have_accepted_that_plan(self, instance,
                                                            overlay):
        """Non-vacuity: proves the override is load-bearing.

        Without this, the three tests above would pass equally well against an
        override that changed nothing, and there would be no evidence that the
        inherited implementation had a hole.
        """
        vb, _ = overlay
        instance._cur = _FakeCursor(PLAN_OFFERED_NOT_TAKEN)
        assert vb.VBMySQLBase._verify_plan(instance, 10) is True
        # ...and the override rejects the same plan through the same cursor.
        assert instance._verify_plan(10) is False

    def test_the_marker_is_an_access_type_and_not_a_name(self, overlay):
        _, percona = overlay
        assert percona.PLAN_ACCESS_TYPE == "vector_ann"
        assert percona.PLAN_ACCESS_TYPE != percona.PERCONA.index_name

    def test_the_check_is_switched_on(self, overlay):
        """`verify_index_used=False` would skip it entirely."""
        _, percona = overlay
        assert percona.PERCONA.verify_index_used is True


class TestMarchIsReportedAndNotLostToAnUnknown:
    """docs/09 §4.8: BOTH `-march` readers know this image's path.

    "Never mix -march between engines" is a stated invariant, and `march` is the
    field that would catch a mixed build. A reader that does not know an
    engine's path reports `unknown` for every one of that engine's records,
    which is the "hard-coded probe whose figure reaches a published artefact
    looking measured" shape.

    **THIS CLASS WAS INVERTED BY THE CHANGE IT DESCRIBES, AND THAT IS WHY IT
    READS THIS WAY NOW.** §4.8 prescribes two steps -- add
    `/opt/percona/.march` to both tuples, and *then* drop the per-engine
    override in `percona/module.py`, "since the next engine added will hit the
    same thing twice". Until that was done these guards asserted the state the
    DESIGN task left behind: the path absent from the shared tuple and present
    in an override. Performing the prescribed change therefore failed four of
    them. They assert the finished state instead, and they are strictly
    stronger for it -- there are two readers, the old wording checked one, and
    the one it could not check (`harness/main.py`, in the harness container) is
    the one every ops record's `march` comes from.
    """

    @staticmethod
    def _march_paths(path_parts, name="_image_march") -> list:
        """The string literals a march reader actually iterates.

        Parsed out of the AST rather than matched against the function's text.
        The text version of this test was vacuous: the method's own DOCSTRING
        names `/opt/percona/.march`, so deleting the path from the real tuple
        left the substring present and the guard green. Measured, not assumed --
        with the path removed from the tuple the old assertion still passed.

        This matters more than most guards here because §4.8's failure mode is
        a wrong `march` reaching a published artefact looking measured, and
        `march` is the field that would catch a mixed-SIMD build.
        """
        import ast
        tree = ast.parse(open(os.path.join(*path_parts)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                # The docstring is an Expr statement, never a `for` iterable,
                # so walking the loops cannot pick it up.
                for inner in ast.walk(node):
                    if isinstance(inner, ast.For) and isinstance(
                            inner.iter, (ast.Tuple, ast.List)):
                        return [e.value for e in inner.iter.elts
                                if isinstance(e, ast.Constant)
                                and isinstance(e.value, str)]
        raise AssertionError(f"no {name} loop over a literal sequence")

    #: The two readers, in the two containers. Named as data so every test
    #: below covers both and one cannot be quietly dropped from a loop.
    ANN_READER = (OVERLAY, "ann_benchmarks", "vb_mysql.py")
    OPS_READER = (VB_ROOT, "harness", "main.py")
    READERS = ((ANN_READER, "_image_march"), (OPS_READER, "_detect_march"))

    @pytest.mark.parametrize("parts,name", READERS)
    def test_every_march_reader_knows_this_images_path(self, parts, name):
        """Both tuples, not one.

        `vb_mysql.py` serves the ann pass; `harness/main.py` serves the ops
        pass from a container that mounts no overlay, so no module override
        could ever have reached it and `orchestrator/ops_pass.py` passes no
        `--march` for its `or` to fall back from.
        """
        paths = self._march_paths(parts, name=name)
        assert paths, f"{name}: the tuple is empty"
        assert "/opt/mariadb/.march" in paths, paths   # needle findable
        assert "/opt/percona/.march" in paths, paths

    def test_the_per_engine_override_is_gone(self):
        """§4.8's second instruction, and the half that is easy to skip.

        Adding the path to the shared tuples while leaving the override in
        place would work, and would leave the next engine two places to miss
        instead of one. Asserted on the AST rather than on the text, because
        the comment that REPLACED the override names the method it replaced
        and a substring check would match that comment -- the same vacuity
        shape the docstring hole had.
        """
        import ast
        tree = ast.parse(open(os.path.join(ALGORITHMS, "percona",
                                           "module.py")).read())
        defined = [n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]
        assert "_image_march" not in defined, (
            "percona/module.py still overrides _image_march; §4.8 says the "
            "path belongs in both shared tuples instead")
        # Non-vacuity: the parser does find this file's other methods, so an
        # empty list is not what made the line above pass.
        assert "get_additional" in defined, defined
        # And the removal is explained where the override was, rather than
        # leaving a silent hole for the next reader to 'fix' by adding it back.
        body = open(os.path.join(ALGORITHMS, "percona", "module.py")).read()
        assert "_image_march" in body, "the reason it is absent is not recorded"

    @pytest.mark.parametrize("parts,name", READERS)
    def test_no_march_reader_can_be_satisfied_by_prose(self, parts, name):
        """Non-vacuity, stated as a test rather than as a comment.

        Proves the assertions above read the tuples and not the surrounding
        text: in both files the path appears in the function's text more often
        than in its tuple, so a text-matching guard cannot tell the two apart.
        """
        body = open(os.path.join(*parts)).read()
        slice_ = body.split(f"def {name}", 1)[1]
        slice_ = slice_.split("\n    def ", 1)[0].split("\ndef ", 1)[0]
        in_tuple = len([p for p in self._march_paths(parts, name=name)
                        if p == "/opt/percona/.march"])
        assert slice_.count("/opt/percona/.march") > in_tuple, name

    def test_the_dockerfile_writes_the_path_both_readers_read(self):
        """The two halves of one measurement, in three files."""
        dockerfile = open(os.path.join(DOCKER, "percona", "Dockerfile")).read()
        assert "/opt/percona/.march" in dockerfile


# ---------------------------------------------------------------------------
# The overlay config
# ---------------------------------------------------------------------------

class TestOverlayConfig:

    @pytest.fixture(scope="class")
    def cfg(self):
        import yaml
        return yaml.safe_load(open(os.path.join(ALGORITHMS, "percona",
                                                "config.yml")))

    @pytest.fixture(scope="class")
    def entry(self, cfg):
        return cfg["float"]["any"][0]

    def test_the_constructor_it_names_is_defined_in_the_module(self, entry):
        """A config naming a class the module does not define fails only once
        the image is built and the corpus is loaded -- the same check
        tests/test_config.py makes for every other engine."""
        import ast
        tree = ast.parse(open(os.path.join(ALGORITHMS, "percona",
                                           "module.py")).read())
        classes = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef)}
        assert entry["constructor"] in classes

    def test_the_module_path_matches_the_directory(self, entry):
        assert entry["module"] == "ann_benchmarks.algorithms.percona"
        assert os.path.isfile(os.path.join(ALGORITHMS, "percona", "module.py"))

    def test_the_ann_constructor_collides_with_no_existing_engine(self, entry):
        """ann-benchmarks keys result files on the constructor, so a duplicate
        makes the second engine report the first's recall. This is why
        mariadb123 exists as a separate name."""
        import ast
        others = set()
        for name in sorted(os.listdir(ALGORITHMS)):
            path = os.path.join(ALGORITHMS, name, "module.py")
            if name == "percona" or not os.path.isfile(path):
                continue
            tree = ast.parse(open(path).read())
            others |= {n.name for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef)}
        assert others, "no other overlay modules found to compare against"
        assert entry["constructor"] not in others

    def test_it_is_innodb_only(self, entry):
        """Percona's vector index is InnoDB's (storage/innobase), so there is no
        MyISAM curve -- the alisql shape, not the mariadb one."""
        groups = entry["run_groups"]
        assert set(groups) == {"innodb"}
        for g in groups["innodb"]["arg_groups"]:
            assert g["engine"] == "InnoDB"

    def test_the_ef_search_grid_is_inside_the_servers_range(self, entry):
        """1..100000, clamped with a warning outside
        (innodb_hnsw_ef_search_basic.result:31-50). A clamped point would be
        recorded under the value asked for, not the value used."""
        grid = entry["run_groups"]["innodb"]["query_args"][0]
        assert grid
        assert all(1 <= ef <= 100000 for ef in grid), grid

    def test_the_ef_search_grid_matches_the_other_mysql_family_engines(self,
                                                                      entry):
        """Three curves sampled at different points are not comparable."""
        import yaml
        grid = entry["run_groups"]["innodb"]["query_args"][0]
        other = yaml.safe_load(open(os.path.join(ALGORITHMS, "alisql",
                                                 "config.yml")))
        other_grid = (other["float"]["any"][0]["run_groups"]["innodb"]
                      ["query_args"][0])
        assert grid == other_grid

    def test_ef_construction_is_declared_and_pinned_to_the_vendor_default(
            self, entry, overlay):
        """Percona is the third engine here to expose the knob, and a curve that
        did not say which value built it could not be read beside MariaDB's.

        The name says "pinned" and not "swept" because the config pins it to a
        single value: the old name claimed a sweep while asserting only that the
        key was present, which would have passed on any grid including a
        one-element one. Checked both ways now.
        """
        _, percona = overlay
        groups = entry["run_groups"]["innodb"]["arg_groups"]
        assert groups
        for g in groups:
            assert "efConstruction" in g
            # Pinned, deliberately: a knob only one engine has adds a grid
            # dimension that cannot be compared across engines, so the fair
            # reading beside MariaDB is at the vendor default.
            assert g["efConstruction"] == [percona.DEFAULT_EF_CONSTRUCTION]

    def test_the_config_says_in_words_that_it_is_pinned_not_swept(self):
        """The comment is the only place a reader learns it was a choice."""
        src = open(os.path.join(ALGORITHMS, "percona", "config.yml")).read()
        assert "pinned" in src.lower()


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

class TestTheDocument:

    @pytest.fixture(scope="class")
    def doc(self):
        return open(PERCONA_DOC).read()

    def test_it_exists_and_is_the_file_docs_are_numbered_for(self, doc):
        assert doc.startswith("# Adding Percona Server as a seventh engine")

    def test_the_pinned_sha_is_the_same_in_every_file_that_names_one(self, doc):
        """A pin that differs between the document and the code it describes is
        not a pin.

        THE SECOND ASSERTION IS NOT REDUNDANT, and it was nearly lost. The set
        comparison alone is satisfied by a SHA in `percona/module.py` with
        docs/09 naming none at all -- so a document that dropped its pin would
        pass. That half used to live in
        `test_foreign_citations_are_skipped_explicitly`, as a repeated
        forty-character literal, and the citation conversion deleted that test
        along with the foreign-citation machinery it belonged to. It is here
        now, spelled `PINNED_SHA` rather than repeated, so the pin has one
        definition in this file instead of two that can drift -- which is the
        argument the comment above `PINNED_SHA` already makes about three
        copies.
        """
        module = open(os.path.join(ALGORITHMS, "percona", "module.py")).read()
        shas = set(re.findall(r"\b[0-9a-f]{40}\b", doc + module))
        assert shas == {PINNED_SHA}, shas
        assert PINNED_SHA in doc, (
            "docs/09 names no 40-character SHA; §0's pin table is where the "
            "document commits to having been read against a specific tree")

    @pytest.fixture(scope="class")
    def section1(self, doc):
        return doc.split("## 1.", 1)[1].split("## 2.", 1)[0]

    def test_it_answers_the_ef_search_question_affirmatively(self, section1):
        """The question this investigation existed to answer.

        An earlier version of this test required only that the section contain
        `innodb_hnsw_ef_search`, `**Yes.**` and `SESSION`. The whole section was
        replaced with waffle that kept those three tokens and deleted every
        citation, both bounds and both riders, and the suite stayed green.

        So this asserts POLARITY and PLACEMENT, not presence: the affirmative
        answer has to come before the evidence, and the negative answer must be
        absent. Presence alone is what the hedge attack exploits.
        """
        yes = re.search(r"\*\*Yes\.\*\*", section1)
        assert yes, "the section does not answer affirmatively"
        assert "innodb_hnsw_ef_search" in section1
        # Placement: the answer leads, the evidence follows. A section that
        # buried the verdict under hedging would fail here.
        # The evidence form changed with the citation policy: §1's sources
        # are vendor files, so they are named in italics rather than in code
        # font, and the first one is the placement marker the old
        # `.cc:<line>` search was.
        first_citation = re.search(
            r"\*[A-Za-z0-9_./-]+\.(?:result|test|yy|cc|h|md)\*", section1)
        assert first_citation, "the section cites nothing"
        assert yes.start() < first_citation.start()
        # Polarity: none of the phrasings that would mean the opposite.
        lowered = section1.lower()
        for wrong in ("exposes nothing equivalent", "no ef_search",
                      "build-time only", "not settable per query",
                      "has no query-time"):
            assert wrong not in lowered, wrong

    def test_it_states_both_scopes_and_the_bounds(self, section1):
        """The facts a driver author needs, each of which the hedge attack
        deleted while keeping the three tokens the old test looked for."""
        assert "SESSION" in section1 and "GLOBAL" in section1
        for value in ("40", "100000"):
            assert value in section1, value
        # The clamp, which is why an out-of-range grid point would be recorded
        # under a value the server did not use.
        assert "clamp" in section1.lower()

    def test_it_carries_both_riders(self, section1):
        """Each rider changes driver code, so neither may be dropped.

        The two tokens below are the QUOTED forms, not paraphrases, and that is
        the point. A second hedge attack got a §1 past all 94 guards: it led
        with `**Yes.**`, carried every token then demanded, met the citation
        floor exactly, and avoided all five banned phrasings -- while telling
        the follow-on task to do the opposite of what the investigation
        established, on the two sentences that become driver code. It kept the
        token `max(ef_search, k)` while arguing against implementing the clamp,
        because the old regex accepted either spelling.

        `max(ef, LIMIT)` is the server's own help string, quoted from the
        declaration this section reproduces, and
        `EngineDriver.effective_ef` in `harness/drivers/base.py` is the hook
        that has to be overridden; the attack carried neither. Requiring the
        quoted forms is the same technique that already works for the bounds.
        The hook used to be required as `8a82da8:harness/drivers/base.py:469`;
        it is required by name now, and by the name of the file it is in, so
        the pair survives an edit above it. That superseded number is written
        in the revision-scoped form `tests/test_e16_citation_ratchet.py`
        prescribes for a dead citation -- it is the spelling `git show` takes,
        so it names a line that cannot move rather than one that already
        has.
        """
        assert "effective_ef" in section1
        assert "max(ef, LIMIT)" in section1, (
            "the clamp must be quoted in the server's own spelling, not "
            "paraphrased")
        assert "harness/drivers/base.py" in section1
        assert "ef_construction" in section1

    def test_it_recommends_implementing_the_clamp_rather_than_inheriting(
            self, section1):
        """Polarity on the first rider's CONCLUSION, not just its tokens.

        The attack's sentence was "`effective_ef` should therefore be left as
        the inherited identity: overriding it would double-apply a clamp the
        server has already applied". Banning that wording alone would be
        brittle, and "the inherited identity" appears in the correct text too
        ("...and not leave the inherited identity"), so a substring ban would
        collide with the thing it is protecting -- hence the longer phrase
        `be left as the inherited identity`, which the correct text does not
        contain.

        REQUIRE THE INSTRUCTION PAIRED WITH ITS VALUE, NOT ONE VERB. An earlier
        version demanded the literal `should implement`, and that fired on
        rewordings preserving every fact and both conclusions:

            should **override** `effective_ef` as `max(ef, k)`   -> FAILED
            **must** implement `effective_ef` as `max(ef, k)`    -> FAILED

        `override` is if anything the more precise verb -- `effective_ef` is an
        inherited concrete method with a default on `EngineDriver`, so
        overriding is literally what happens -- and `must` is strictly stronger
        than `should`. A guard that fires on correct prose teaches the next
        author to weaken or delete it, which is how the vacuity shapes reached
        this repo in the first place: a false positive on a correct document is
        a slower-acting version of the same failure.

        So: either verb, any modal, any wording between, but the target AND the
        value must both be present. `[\\s\\S]` rather than `.` so a line wrap
        cannot fail it, and rather than `\\W` because the gap contains words
        ("as", "with", "returning") -- a `\\W` version of this pattern rejected
        all seven legitimate rewordings for exactly that reason before it was
        measured against them.
        """
        assert re.search(
            r"(implement|override)[\s\S]{0,30}effective_ef"
            r"[\s\S]{0,60}max\(ef,\s*k\)", section1), (
            "§1 must tell the follow-on task to implement or override "
            "effective_ef AS max(ef, k)")
        for wrong in ("double-apply", "be left as the inherited identity"):
            assert wrong not in section1, wrong

    def test_it_says_the_claude_md_asymmetry_sentence_becomes_false(
            self, section1):
        """Polarity on the second rider's conclusion.

        The attack wrote "`ef_construction` is a build-time concern rather than
        a knob this engine surfaces, so `CLAUDE.md`'s 'exposed only by pgvector
        and Valkey' sentence stands and needs no edit" -- the exact opposite of
        the finding, with the token `ef_construction` present throughout.

        BANS SCOPED TO THE SENTENCE CARRYING THE SUBJECT. An earlier version
        banned `needs no edit` anywhere in §1, and that is subject-blind: it
        fires on a true sentence about something else, for instance

            "`EngineDriver.effective_ef`'s inherited default is correct for
             every other engine here and **needs no edit**"

        which is correct prose about a different subject, and "needs no edit"
        is exactly the phrase that recurs in a change-list document. A ban has
        to name the subject it is a ban about, so these look only at sentences
        mentioning `CLAUDE.md` or `ef_construction` -- which is where the
        attack's claim lived, and where a claim of this kind has to live to
        mean anything.
        """
        assert re.search(r"sentence false|sentence.{0,40}false", section1), (
            "§1 must say the CLAUDE.md asymmetry sentence becomes false")
        subject_sentences = [
            s for s in re.split(r"(?<=[.!?])\s+", section1)
            if "CLAUDE.md" in s or "ef_construction" in s
        ]
        assert subject_sentences, (
            "§1 says nothing about ef_construction or CLAUDE.md at all")
        for sentence in subject_sentences:
            for wrong in ("sentence stands", "needs no edit"):
                assert wrong not in sentence, f"{wrong!r} in: {sentence[:90]}"

    def test_it_cites_more_than_a_token(self, section1):
        """A citation floor, because the hedge attack's tell was that every
        citation went away while the verdict stayed.

        The floor used to count `file:line` spellings. It now counts the
        ITALICISED VENDOR SOURCES the conversion replaced them with -- the
        same claim about the same section, over the citation form §0 now
        requires. Kept at 6 with §1 carrying more: the second attack on this
        section met a `>= 4` floor exactly, and a threshold three below the
        real count is a licence to delete three citations silently.
        """
        citations = re.findall(
            r"\*([A-Za-z0-9_./-]+\.(?:result|test|yy|cc|h|md))\*", section1)
        assert len(citations) >= 6, citations

    def test_the_sysvar_declaration_is_quoted_not_paraphrased(self, section1):
        """ha_innodb.cc's declaration is the primary source for every number in
        this section -- scope, default and both bounds in one place -- so the
        section quotes it rather than restating it."""
        assert "MYSQL_THDVAR_ULONG" in section1
        assert "*ha_innodb.cc*" in section1

    # -- the prose is cross-checked against that quote ------------------
    #
    # A fourth attack shape passed all 104 guards: every required form, no
    # banned form, `**Yes.**` before the first citation, enough citations, and
    # BOTH riders' conclusions stated correctly -- with the two facts a reader
    # actually looks up inverted. It claimed the variable was GLOBAL-only and
    # established once per connection, and that the default was 100000 with a
    # floor of 40.
    #
    # The detail that makes it fixable: the section quotes the correct
    # declaration -- `nullptr, nullptr, 40, 1, 100000, 0)` -- three lines above
    # the prose contradicting it. The guards required that quote and nothing
    # compared the two. These do, value against value, which is the technique
    # that makes the §3 arithmetic guard strong and which vocabulary cannot
    # satisfy.
    #
    # It is load-bearing because §1 is where the calibration-grid author and
    # the driver author read scope and range. "GLOBAL-only, once per
    # connection" sends the driver to `global_setup` instead of
    # `set_ef_search`, which is the whole calibration mechanism; "default
    # 100000, floor 40" licenses a grid below 40 that the server silently
    # clamps -- the exact failure the clamp sentence exists to warn about.

    @staticmethod
    def _declared(section1):
        """(default, min, max) parsed out of the quoted declaration."""
        decl = re.search(r"MYSQL_THDVAR_ULONG\(([\s\S]*?)\);", section1)
        assert decl, "§1 no longer quotes the MYSQL_THDVAR_ULONG declaration"
        tail = re.search(
            r"nullptr\s*,\s*nullptr\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            decl.group(1))
        assert tail, (
            "the quoted declaration's trailing arguments do not parse; the "
            "cross-checks below have nothing to compare against")
        return tail.group(1), tail.group(2), tail.group(3)

    @staticmethod
    def _flat(section1):
        """Whitespace-normalised section.

        Every check below runs on this, not on the raw text. Measured reason:
        the fourth attack wrote "a read-only\\nreflection of it", and a raw
        substring ban on `read-only reflection` did not fire because the phrase
        straddled a line wrap. A prose guard that a reflow defeats is not a
        guard.
        """
        return " ".join(section1.split())

    def test_the_declaration_quote_parses_to_a_default_and_two_bounds(
            self, section1):
        """Pin the quote itself, so the cross-checks cannot pass vacuously.

        If the declaration were reformatted or trimmed to the point where the
        trailing arguments stop parsing, every value comparison below would
        have nothing to compare and would need to fail loudly rather than
        quietly succeed. `_declared` asserts, so this test is that failure.
        """
        default, mn, mx = self._declared(section1)
        assert (default, mn, mx) == ("40", "1", "100000"), (default, mn, mx)

    def test_the_stated_default_and_maximum_match_the_declaration(
            self, section1):
        """The prose's numbers must agree with the quote three lines above it.

        Four comparisons, each tying a number to a ROLE rather than checking
        that the number appears somewhere: the attack contained both 40 and
        100000, so presence proves nothing. It swapped which was which.
        """
        default, mn, mx = self._declared(section1)
        flat = self._flat(section1)
        assert re.search(r"default\D{0,20}\b%s\b" % default, flat, re.I), (
            f"§1 must state the default as {default}, as its own quote does")
        assert not re.search(r"default\D{0,20}\b%s\b" % mx, flat, re.I), (
            f"§1 calls {mx} the default; the declaration it quotes says "
            f"{default} is the default and {mx} is the maximum")
        assert re.search(r"(maximum|upper bound|range)\D{0,24}\b%s\b" % mx,
                         flat, re.I), (
            f"§1 must state {mx} as the maximum or the top of the range")
        assert not re.search(r"(floor|minimum)\D{0,20}\b%s\b" % default, flat,
                             re.I), (
            f"§1 calls {default} the floor; the declaration it quotes says "
            f"the minimum is {mn}")

    def test_the_stated_scope_matches_the_session_variable_macro(
            self, section1):
        """`MYSQL_THDVAR_*` is the session-variable macro, and the engine reads
        it per query through `THDVAR`. Prose saying otherwise sends the driver
        to the wrong hook.
        """
        flat = self._flat(section1)
        assert "MYSQL_THDVAR_ULONG" in flat
        assert "SET SESSION innodb_hnsw_ef_search" in flat, (
            "§1 must show the variable being set per session")
        assert "THDVAR" in flat, (
            "§1 must cite the per-query read, or 'per query' is an assertion")
        assert not re.search(r"\bGLOBAL\b[^.]{0,40}\b(only|in scope)\b", flat), (
            "§1 describes the variable as GLOBAL-only; MYSQL_THDVAR_ULONG is "
            "session and global")
        assert not re.search(r"read-only\s+reflection", flat), (
            "§1 calls the session value a read-only reflection of the global "
            "one; it is settable per session")

    def test_it_keeps_the_live_server_questions_in_their_own_section(self, doc):
        """docs/09's own promise: what needs a server is listed separately so a
        reader can tell reading from measurement."""
        assert "## 7. What only a live server can settle" in doc
        section = doc.split("## 7. What only a live server can settle", 1)[1]
        # Enumerated, not a paragraph of hedging. Floor raised from 5 to 7:
        # §7 carries 8, and a threshold three below the real count would let
        # three questions be dropped without a red test -- the same loose-floor
        # shape the second §1 hedge attack exploited.
        items = re.findall(r"^\d+\. \*\*", section, re.MULTILINE)
        assert len(items) >= 7, items
        # Contiguous numbering, so striking an item renumbers rather than
        # leaving a gap a reader reads as a missing question.
        numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", section,
                                              re.MULTILINE)]
        assert numbers == list(range(1, len(numbers) + 1)), numbers

    def test_the_struck_items_heading_counts_its_own_bullets(self, doc):
        """Self-consistency, the same shape as the submodule-count guard.

        Caught by mutation: renaming the heading from "Three" to "Two" while
        leaving three bullets passed every other guard in this file. A count in
        a heading that disagrees with the list under it is how the submodule
        note came to say "two" over four entries, and the fix is the same --
        anchor on one deterministic phrase rather than scan for number words.
        """
        section = doc.split("## 7. What only a live server can settle", 1)[1]
        heading = re.search(r"^### (\w+) items struck from this list", section,
                            re.MULTILINE)
        assert heading, "the struck-items subsection is missing or renamed"
        words = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
        stated = words.get(heading.group(1))
        assert stated is not None, heading.group(1)
        struck = section.split(heading.group(0), 1)[1]
        bullets = re.findall(r"^- \*\*", struck, re.MULTILINE)
        assert len(bullets) == stated, (
            f"heading says {heading.group(1)}, {len(bullets)} bullets listed")

    def test_it_names_every_file_the_follow_on_task_must_change(self, doc):
        """The brief for this design lists these by name; a change list missing
        one of them gets as far as starting a server and then dies on
        `argument --engine: invalid choice`."""
        section = doc.split("## 4. File-by-file change list", 1)[1]
        for required in ("harness/drivers/mysql_family.py",
                         "_driver_classes()",
                         "harness/drivers/postgres.py",
                         "config/engines/percona.yml",
                         "GOLDEN_ORDER",
                         "tests/test_engine_registry.py",
                         "webui/static/fields.js",
                         "config/profiles/smoke.yml",
                         # Added after review found all three missing. Each is
                         # a file Percona breaks or silently degrades:
                         # ann_pass raises, main.py blanks `march`, lib.sh
                         # leaves a vendor checkout unprotected.
                         "orchestrator/ann_pass.py",
                         "harness/main.py",
                         "scripts/lib.sh"):
            assert required in section, required

    def test_it_records_the_family_wide_entrypoint_defect(self, doc):
        """§4.9. Found while matching the family's entrypoint, out of scope to
        fix, and the kind of thing that is lost if it is not written down.

        Reproduced rather than reasoned: the pattern extracted verbatim with
        `id -u` stubbed exits 1 before `exec` for a non-root uid and prints
        nothing. Latent only because no image here sets USER.
        """
        section = doc.split("### 4.9", 1)[1].split("### 4.10", 1)[0]
        # All three siblings named, so a fix cannot be applied to one.
        for path in ("entrypoint-percona.sh", "entrypoint-alisql.sh",
                     "entrypoint-mariadb.sh"):
            assert path in section, path
        assert "set -euo pipefail" in section
        assert "latent" in section.lower()
        # The reproduction, not just the claim.
        assert "FAKE_UID=1000" in section and "rc=1" in section

    def test_the_entrypoint_pattern_it_describes_is_really_there(self):
        """The claim is about three files in this checkout, so check them.

        If a sibling is ever fixed without the others, this fails and says so --
        which is the point of recording it as family-wide.
        """
        for engine in ("percona", "alisql", "mariadb"):
            path = os.path.join(DOCKER, engine, f"entrypoint-{engine}.sh")
            src = open(path).read()
            # The call site: this line IS byte-identical across all three.
            assert 'local u; u="$(user_args)"' in src, (
                f"{engine}: call-site pattern gone, update §4.9")
            assert "set -euo pipefail" in src, engine
            # The function body whose non-zero exit `set -e` acts on. Matched
            # on the test-and-printf rather than on a one-line definition:
            # mariadb's user_args() is formatted across four lines with a
            # comment, so the three are NOT byte-identical as functions even
            # though they behave identically and the call site is. A guard on
            # the one-line spelling passed for alisql and percona and failed
            # for mariadb, which would have read as "mariadb is already fixed".
            assert re.search(r'\[\[ "\$\(id -u\)" -eq 0 \]\] && printf', src), \
                engine

    def test_it_records_where_ann_pass_refuses_an_unknown_engine(self, doc):
        """§4.10's whole point is that Percona lands on this `raise`.

        The section used to reproduce the line with its gutter number and this
        test parsed the number back out. The number is gone: the section
        quotes the statement, names the function it is in, and the
        `CITED_SYMBOLS` row for `render_config` is what re-opens it. What
        stays here is the part that row cannot express -- that the quotation
        §4.10 prints is the statement `render_config` really ends with, and
        that it is the LAST thing in the function rather than one branch among
        several.
        """
        section = doc.split("### 4.10", 1)[1].split("\n## ", 1)[0]
        assert 'raise ValueError(f"unknown engine: {engine}")' in section
        assert "render_config" in section
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ann_pass.py")).read()
        body = source.split("def render_config", 1)[1].split("\ndef ", 1)[0]
        statement = 'raise ValueError(f"unknown engine: {engine}")'
        assert statement in body
        # "Percona matched none of them and landed on the `raise`" is a claim
        # about ORDER: the raise is the dispatch's fallback, so it comes after
        # every engine branch. Checked as an order rather than as "the last
        # line of the function", which it is not -- a `return` follows it.
        last_branch = body.rindex("elif engine ==")
        assert last_branch < body.index(statement), (
            "§4.10 says an unknown engine falls through every branch to this "
            "raise, and the raise now precedes a branch")

    def test_no_mysql_family_branch_in_ann_pass_emits_ef_construction(self,
                                                                      doc):
        """§4.10 says Percona needs a branch of its own because the MySQL-family
        branch cannot emit `efConstruction`. Checked, not asserted."""
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "ann_pass.py")).read()
        body = source.split("def render_config", 1)[1]
        mysql_branch = body.split('if engine in MARIADB_ENGINES', 1)[1]
        mysql_branch = mysql_branch.split('elif engine ==', 1)[0]
        # Non-vacuity: the pgvector branch right after it does emit the key.
        assert "efConstruction" in body
        assert "efConstruction" not in mysql_branch

    def test_the_ops_path_march_reader_covers_this_image(self, doc):
        """§4.8's second half, now done.

        A different container from the overlay, so no module override can
        reach it -- which is exactly why the path had to go in here and not
        only in `percona/module.py`. This assertion used to be `not in`,
        recording that the work was outstanding; it fired when the work was
        done, which is the right direction for it to fire in.
        """
        source = open(os.path.join(VB_ROOT, "harness", "main.py")).read()
        fn = source.split("def _detect_march", 1)[1].split("\ndef ", 1)[0]
        assert "/opt/mariadb/.march" in fn          # needle findable
        assert "/opt/percona/.march" in fn
        assert "harness/main.py" in doc
        # And nothing passes --march in from the orchestrator, so the detector
        # is what runs.
        ops = open(os.path.join(VB_ROOT, "orchestrator", "ops_pass.py")).read()
        assert "march" not in ops

    def test_it_does_not_claim_optimize_table_fails_to_compact(self, doc):
        """The review's HIGH 1. `handler0alter.cc:1393-1404` says a table with a
        vector index is rebuilt in place and the graph rebuilt from the copied
        rows; a DELETEd row is not copied, so OPTIMIZE compacts. An earlier
        revision said the opposite, having read a scare-quoted hedge in an MTR
        comment as a claim about the statement.

        Polarity, not presence: the section has to say it DOES compact.
        """
        section = doc.split("### 2.7", 1)[1].split("## 3.", 1)[0]
        assert "*handler0alter.cc*" in section
        assert "**does compact**" in section
        for wrong in ("does **not** compact graph tombstones",
                      "`OPTIMIZE TABLE` is not it"):
            assert wrong not in section, wrong
        # And it still explains why maintain() is the index replace anyway.
        assert "without rewriting the base table" in section

    def test_it_records_the_build_py_lines_it_calls_a_defect(self, doc):
        """§2.5's claim ABOUT build.py, beyond where its lines are.

        The line range itself is checked by the citation table below, as one
        row among all of them. What stays here is the part a line check cannot
        express: the document's claim that `build_mode == "post"` has no
        counterpart anywhere in the file. That is the defect §2.5 is about, and
        it would survive any amount of renumbering.
        """
        source = open(os.path.join(VB_ROOT, "harness", "workloads",
                                   "build.py")).read()
        assert 'index.build_mode == "incremental"' in source
        assert 'build_mode == "post"' not in source
        # The three lines §2.5 reproduces used to be cited as "Lines 93-95";
        # they are now quoted and attributed to the function they are in, and
        # the `CITED_SYMBOLS` row for `build.py`'s `run` re-opens them. What
        # this test keeps asserting is that the document still attributes the
        # quotation rather than leaving it floating.
        section = doc.split("### 2.5", 1)[1].split("### 2.6", 1)[0]
        assert "`harness/workloads/build.py`" in section
        assert "`run`" in section


# ---------------------------------------------------------------------------
# Citations: symbols, never line numbers
# ---------------------------------------------------------------------------
#
# WHAT THIS REPLACED, AND WHY IT HAD TO BE REPLACED RATHER THAN REPAIRED.
#
# docs/09 used to cite `path`-colon-line, and this file re-opened every cited
# line and compared it against what the document claimed was there. The table
# was real coverage and it caught real rot. It also had two defects that no
# amount of repair could reach:
#
# 1. A LINE NUMBER DIES TO ANY EDIT ABOVE IT. Line citations in docs/09 and
#    docs/10 went stale twice in two days and put `master` red both times
#    (#41, #43). Neither was a careless edit to a document: in both cases a
#    later round of edits inside the SAME change moved lines below the point
#    where the citations had been derived. The repair for one moved others,
#    which is how the second failure happened.
# 2. THE HALF NOBODY CHECKED WAS THE BIGGER HALF. The old parser saw 70 of
#    this document's ~198 citations; `FOREIGN_MARKERS` then deny-listed every
#    vendor path by name, correctly, because the pinned Percona tree is not in
#    this checkout. What was left was 31 occurrences, 27 of them distinct,
#    pointing into this repository. The other ~164 vendor line references were
#    checked by NOTHING -- not even for being in range -- and would have gone
#    stale in silence at the re-pin, in a document whose §0 provenance table
#    commits it to having been read against a specific tree. A citation that
#    cannot go red is a decoration that reads like coverage.
#
# So the document cites symbols, and the guard has three layers in place of
# one. Each is named here because each is a different strength:
#
# * THE RATCHET (`TestTheDocumentMakesNoLineCitation`). docs/09 must contain
#   ZERO line citations of any form. One assertion replacing twenty-seven
#   parameterised ones, and strictly stronger about the thing that was
#   rotting. The pattern is WIDER than the one it replaces -- MTR `.result`
#   and `.test` paths, `.yy` grammars, extensionless files and bare `:N`
#   continuations were all invisible to the old one -- and it is proved
#   against `CITATION_FIXTURE`, a frozen string carrying one of each form,
#   because a negative assertion over a pattern that matched nothing would
#   pass against a document full of citations. That is the same vacuity the
#   old `_CITE_RE`'s own note recorded being caught by.
#
# * THE SYMBOL TABLE (`CITED_SYMBOLS`). Every claim docs/09 makes about the
#   CONTENTS of something in this repository is a row keyed by
#   `(path, symbol)` and carrying the string the document points at. The test
#   opens the file, locates that symbol's declaration, and asserts the string
#   is inside it. Same strength as the old line check -- it is still the
#   contents being compared -- keyed by a name instead of a number, so it
#   survives an edit above it.
#
# * EXISTENCE OVER EVERYTHING ELSE. Every code-font path in docs/09 resolves
#   to a file in this tree and every code-font symbol appears somewhere in it,
#   using `tests/test_e15_methodology_roadmap_doc.py`'s classifier rather than
#   a second copy of it, so the three guarded documents agree about what a
#   path is and what a symbol is.
#
# WHAT IS WEAKER, STATED RATHER THAN IMPLIED. A vendor symbol citation is
# checked by nothing. It was checked by nothing before; what changed is that
# it now names a function, a class or an MTR test a reader holding the pinned
# tree can find by searching, instead of a line number that is wrong. There is
# no `FOREIGN_MARKERS` any more and there is nothing for one to do: the policy
# puts vendor material in prose, so a code-font path into the vendor tree is
# not a citation this guard has to classify -- it is a violation the path arm
# fails on. That is the one place this conversion is strictly stronger about
# the vendor half, and it is the only one.


#: Where a symbol may resolve, and the suffixes read into the index. Imported
#: rather than restated: docs/09, docs/10 and docs/11 are three documents under
#: one citation policy, and three copies of the classifier would be three
#: chances for them to disagree about what a path is.
from .test_e15_methodology_roadmap_doc import (  # noqa: E402
    CITATION_FIXTURE, CITATION_FIXTURE_EXPECTED, INDEXED_SUFFIXES,
    ROOT_SOURCES, SOURCE_DIRS, _LINE_CITATION_RE, _YAML_KEY_RE, _WORD_RE,
    _declared_in_python, classify, line_citations)

#: Generated and gitignored, per CLAUDE.md -- not walked for the index and not
#: searched for a path. A file under one of these does not exist on a fresh
#: checkout, so neither a symbol nor a path may resolve against it.
GENERATED_DIRS = ("sources", "work", "results", "datasets", "state")

#: The three document guards exclude ALL THREE of themselves from the symbol
#: index, and the reason is measured rather than supposed.
#: `tests/test_e15_methodology_roadmap_doc.py` records it: the first run of its
#: planted arm PASSED the live document and FAILED the plant, because `tests/`
#: is indexed and the planted name was spelled in the guard. These files spell
#: symbol names while describing what they would reject, so any one of them in
#: the index can certify a citation into thin air.
GUARD_FILES = (
    "tests/test_e1_percona_design.py",
    "tests/test_e3_rollout_doc.py",
    "tests/test_e15_methodology_roadmap_doc.py",
)

#: A commit SHA is not a symbol, and `_SYMBOL_RE` cannot tell the difference:
#: `a86503214416` is a run of hex that happens to start with a letter. docs/09
#: quotes twelve-character SHAs in §0's commit table and in §7, in code font,
#: and they resolve nowhere by design. Excluded by SHAPE rather than by an
#: allowlist of the particular SHAs, deliberately -- an allowlist would have to
#: be edited every time the pin moves, which is the one edit this document
#: makes most often and the one a guard must not fight.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: A fenced block is a QUOTATION and is reproduced verbatim, so the backticks
#: inside one are the source's, not this document's. The vendor comment §2.7
#: quotes spells `online` in backticks; classifying that as a citation would
#: ask this repository to contain a symbol named `online` because Percona's
#: source does. Fences are therefore stripped before classification -- and NOT
#: before the ratchet, which scans the whole file, because a line citation
#: smuggled into a fence is still a line citation.
_FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)


def _tree_files():
    """Every text file a citation may resolve against, repo-relative."""
    out = []
    for name in ROOT_SOURCES:
        if os.path.isfile(os.path.join(VB_ROOT, name)):
            out.append(name)
    for top in SOURCE_DIRS + ("docs",):
        for root, dirs, files in os.walk(os.path.join(VB_ROOT, top)):
            dirs[:] = [d for d in dirs
                       if d not in ("__pycache__", ".git", ".pytest_cache")
                       and d not in GENERATED_DIRS]
            for filename in files:
                out.append(os.path.relpath(os.path.join(root, filename),
                                           VB_ROOT))
    return out


_TREE = []


def tree_files():
    if not _TREE:
        _TREE.extend(_tree_files())
    return _TREE


def resolve_path(spelling):
    """`(resolved, candidates)` for a path as a document spells it.

    A document may abbreviate: §2.5 writes `build.py` where the file is
    `harness/workloads/build.py`, and spelling it out everywhere would make
    several sentences unreadable. An abbreviation is resolved by UNIQUE
    SUFFIX, which is a real check and not a loophole -- it fails when the file
    is renamed, when it is deleted, and when a second file of the same name
    makes the citation ambiguous. The ellipsis form the document used to use
    (`overlay/.../vb_mysql.py`) resolves to nothing under this rule, which is
    why it is gone: a citation a reader cannot paste into a search is not
    better than a line number.
    """
    if os.path.isfile(os.path.join(VB_ROOT, spelling)):
        return spelling, [spelling]
    hits = [f for f in tree_files()
            if f == spelling or f.endswith("/" + spelling)]
    return (hits[0] if len(hits) == 1 else None), hits


_INDEX = {}


def symbol_index():
    """Every name this tree mentions, with the guard files excluded."""
    if _INDEX:
        return _INDEX
    excluded = {os.path.join(VB_ROOT, g) for g in GUARD_FILES}
    mentioned, declared, read = set(), set(), []
    for rel in tree_files():
        if not rel.endswith(INDEXED_SUFFIXES):
            continue
        full = os.path.join(VB_ROOT, rel)
        if full in excluded or rel.startswith("docs/"):
            continue
        try:
            source = open(full, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        read.append(rel)
        mentioned.update(_WORD_RE.findall(source))
        if rel.endswith(".py"):
            declared.update(_declared_in_python(source))
        elif rel.endswith((".yml", ".yaml")):
            declared.update(_YAML_KEY_RE.findall(source))
    _INDEX["mentioned"] = mentioned
    _INDEX["declared"] = declared
    _INDEX["files"] = read
    return _INDEX


def doc_spans(text):
    """`(paths, symbols, prose)` for the document's own prose.

    Fenced quotations are stripped first and SHA-shaped spans are moved out of
    the symbol set, both for the reasons given above.
    """
    paths, symbols, prose = classify(_FENCE_RE.sub("", text))
    shas = {s for s in symbols if _SHA_RE.match(s)}
    return paths, symbols - shas, prose | shas


# -- the line-citation ratchet ----------------------------------------------
#
# THE PATTERN AND ITS FIXTURE ARE IMPORTED, NOT DEFINED HERE. They used to be
# defined here and in `tests/test_e15_methodology_roadmap_doc.py`, which
# carried a comment saying its copy was "kept deliberately identical" to
# `tests/test_e3_rollout_doc.py`'s -- three copies of one policy, and they
# were not identical: the other two were narrow enough to miss 130 of this
# document's 199 line references on `8a82da8`. There is one definition now,
# in the guard for the document that states the policy, and this file and
# docs/10's guard both use it. The direction is forced: this file already
# imports `classify` from there, so defining the pattern here and importing
# it back would be a circular import between two test modules.

#: A line inside a fenced block that opens with a bare number, which is the
#: shape of a reproduced source gutter.
#:
#: Fences are matched with `lstrip()` and not anchored at column 0, and that is
#: the whole reason this guard exists rather than a hand scan. The hand scan
#: this change first used anchored `^```, and docs/09 §1's `search.test`
#: excerpt sits inside a bullet, so its fence is indented two spaces and the
#: entire block was invisible. Four vendor line numbers survived a commit whose
#: message said all the blocks were clean, eleven lines under a sentence saying
#: the numbers were gone, and the citation directly above them had been
#: converted in the same change.
_GUTTER_LINE_RE = re.compile(r"^\s*\d+(\s\s|\s*$)")

#: Lines that open with a number inside a fence and are NOT gutters, each with
#: what the number is. Spelled as the whole line so an entry cannot widen into
#: a licence: a reproduced result row and an EXPLAIN row both legitimately
#: begin with a value, and a scan that refused them would push the document
#: into paraphrasing recorded output, which is the one thing §0 says a
#: `.result` file is preferred FOR.
RECORDED_VALUE_LINES = {
    "40  40":
        "§1's sysvar result: the global and session columns of a SELECT, both "
        "reading 40. The document's point is that number.",
    "40  200":
        "§1's sysvar result, after SET SESSION: global 40, session 200.",
    "1  SIMPLE      t1    NULL       ALL  v1            NULL NULL    NULL X    "
    "100.00   Using filesort":
        "§4.1's EXPLAIN row. The leading 1 is the `id` column, and the row is "
        "the full-table-scan plan that defeats a name-only index check.",
}

# -- locating a symbol's declaration ----------------------------------------

def _preceding_comments(lines, start):
    """The contiguous comment block immediately above line `start` (1-based).

    Part of the body on purpose. This repository's convention is that the
    rationale for a constant lives in the `#:` block above it -- the version
    charter above `OPS_MEASUREMENT_VERSION`, the measured-rates note above
    `_INGEST_ROWS_PER_S` -- and most of what these two documents quote is in
    those blocks rather than in the statement. Excluding them would force the
    documents to quote code they are not talking about.
    """
    i = start - 1
    while i > 0 and lines[i - 1].lstrip().startswith("#"):
        i -= 1
    return i


def _python_symbol_body(source, symbol):
    lines = source.splitlines()
    if symbol == "<module>":
        tree = ast.parse(source)
        doc = ast.get_docstring(tree)
        return doc if doc else None
    want = symbol.split(".")
    found = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            name = None
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                name = child.name
            elif isinstance(child, ast.Assign):
                target = child.targets[0]
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Attribute):
                    name = target.attr
            elif isinstance(child, ast.AnnAssign) and isinstance(
                    child.target, ast.Name):
                name = child.target.id
            if name is not None:
                qualified = stack + [name]
                if qualified[-len(want):] == want:
                    lo = _preceding_comments(lines, child.lineno)
                    hi = getattr(child, "end_lineno", child.lineno)
                    found.append("\n".join(lines[lo:hi]))
            if isinstance(child, ast.ClassDef):
                walk(child, stack + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.If, ast.Try, ast.With, ast.For,
                                    ast.While)):
                walk(child, stack)

    try:
        walk(ast.parse(source), [])
    except SyntaxError:
        return None
    return "\n".join(found) if found else None


def _yaml_symbol_body(source, symbol):
    """A YAML key's own line plus everything nested under it.

    Dotted spellings (`ann.m_values`) are matched on the LAST component and
    then checked for depth, which is enough here and avoids this growing into
    a YAML parser: `pyyaml` would lose the comments, and the comments are what
    both documents quote.
    """
    key = symbol.split(".")[-1]
    lines = source.splitlines()
    pattern = re.compile(r"^(\s*)" + re.escape(key) + r"\s*:")
    out = []
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        lo = _preceding_comments(lines, i + 1)
        hi = i + 1
        while hi < len(lines):
            nxt = lines[hi]
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            hi += 1
        out.append("\n".join(lines[lo:hi]))
    return "\n".join(out) if out else None


def _shell_symbol_body(source, symbol):
    """A shell function, or a variable the file exports or assigns."""
    lines = source.splitlines()
    fn = re.compile(r"^\s*(?:function\s+)?" + re.escape(symbol) + r"\s*\(\)")
    nxt_fn = re.compile(r"^\s*(?:function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\(\)")
    out = []
    for i, line in enumerate(lines):
        if fn.match(line):
            lo = _preceding_comments(lines, i + 1)
            hi = i + 1
            while hi < len(lines) and not nxt_fn.match(lines[hi]):
                hi += 1
            out.append("\n".join(lines[lo:hi]))
    if out:
        return "\n".join(out)
    assign = re.compile(r"^\s*(?:export\s+|local\s+)?" + re.escape(symbol)
                        + r"=")
    hits = [i for i, line in enumerate(lines) if assign.match(line)]
    if hits:
        return "\n".join(lines[_preceding_comments(lines, i + 1):i + 1]
                         for i in hits)
    return None


def _markdown_symbol_body(source, symbol):
    """A section, named by its heading text, down to the next heading of the
    same or a shallower level.

    A heading is the only thing in a document that a sibling document can
    point at and that survives an edit -- §-numbers are stable here by
    convention (docs/10 §6 records why §2.0 of docs/05 is numbered 2.0), and a
    renumbering is a deliberate act a reader can follow. Fenced blocks are
    skipped because MTR output quoted in docs/09 starts lines with `#`.
    """
    lines = source.splitlines()
    heads, fenced = [], False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            m = re.match(r"(#{1,6}) ", line)
            if m:
                heads.append((i, len(m.group(1)), line))
    for pos, (i, level, line) in enumerate(heads):
        if symbol not in line:
            continue
        hi = len(lines)
        for j, lvl, _text in heads[pos + 1:]:
            if lvl <= level:
                hi = j
                break
        return "\n".join(lines[i:hi])
    return None


def _generic_symbol_body(source, symbol):
    """A Dockerfile ARG/ENV, or any line that declares the token.

    The fallback, and narrow on purpose: it returns the declaring line and the
    comments above it, not the whole file, so an anchor that happens to appear
    somewhere else cannot satisfy a row.
    """
    lines = source.splitlines()
    pattern = re.compile(r"^\s*(?:ARG|ENV|LABEL)?\s*" + re.escape(symbol)
                         + r"\s*[=: ]")
    out = []
    for i, line in enumerate(lines):
        if pattern.match(line):
            lo = _preceding_comments(lines, i + 1)
            out.append("\n".join(lines[lo:i + 1]))
    return "\n".join(out) if out else None


def symbol_body(path, symbol):
    """The declaration of `symbol` in `path`, or None if it is not there.

    None is the failure a row must report, and it must report it with the
    FILE'S NAME: a row naming a symbol that no longer exists is the renamed
    case, which is the one case a symbol citation is supposed to catch, and
    "assert anchor in None" would say nothing about where to look.
    """
    full = os.path.join(VB_ROOT, path)
    source = open(full, encoding="utf-8", errors="replace").read()
    if path.endswith(".py"):
        return _python_symbol_body(source, symbol)
    if path.endswith((".yml", ".yaml")):
        return _yaml_symbol_body(source, symbol)
    if path.endswith(".sh"):
        return _shell_symbol_body(source, symbol)
    if path.endswith(".md"):
        return _markdown_symbol_body(source, symbol)
    return _generic_symbol_body(source, symbol)

# ---------------------------------------------------------------------------
# What docs/09 claims about this repository's contents
# ---------------------------------------------------------------------------

#: `(path, symbol)` -> the string, or strings, docs/09 says is inside that
#: symbol. The successor to the old `CITED_IN_REPO`, carrying every one of its
#: anchors and the four `BARE_CONTINUATIONS` rows folded in, re-keyed from a
#: line spelling to the symbol the line sat in.
#:
#: THE KEY IS A SYMBOL AND THE VALUE IS STILL THE CONTENTS, which is the part
#: worth being precise about: this is not "the symbol exists", it is "the
#: symbol exists AND the document is right about what is in it". The only
#: thing that changed is what survives an edit above it.
#:
#: Paths are spelled in full here even where the document abbreviates, because
#: a table is not prose and an unambiguous key is worth more than a short one.
#:
#: The value may be one anchor or a tuple of them. The tuple form is kept from
#: the old table for the reason recorded there: a value written `'a' and 'b'`
#: evaluates to `'b'`, so a row that read as two assertions made one, and the
#: half silently dropped was the half about the entry a follow-on task had to
#: add.
#:
#: An anchor must not span a line break unless the body it is matched against
#: contains that break -- bodies are joined with newlines, so a wrapped
#: sentence is matched by quoting the clause that fits on one line.
CITED_SYMBOLS = {
    # -- §2.7's churn hooks, and §1's clamp -----------------------------
    ("harness/drivers/base.py", "EngineDriver.repair_path"):
        ('repair_path: str = "none"', "Valkey and Percona Search inherit"),
    ("harness/drivers/base.py", "EngineDriver.prepare_churn"):
        "def prepare_churn",
    ("harness/drivers/base.py", "EngineDriver.wait_index_synced"):
        "def wait_index_synced",
    ("harness/drivers/base.py", "EngineDriver.effective_ef"):
        "def effective_ef(self, ef: int, k: int)",
    ("harness/workloads/churn.py", "run"):
        "what = driver.maintain()",

    # -- §2.5's build-mode defect ---------------------------------------
    # The module docstring is the thing §2.5 block-quotes, so `<module>` is
    # the symbol: a docstring has no name of its own, and matching the quote
    # against the whole file would let any sentence in `build.py` satisfy it.
    ("harness/workloads/build.py", "<module>"):
        ("ingest_wall_s",
         "pgvector's bulk build is N× cheaper than its own"),
    # The three lines §2.5 used to cite as "Lines 93-95" plus the two the
    # paragraph after them pointed at by bare number. All five are in one
    # function, which is why five line citations became one row.
    ("harness/workloads/build.py", "run"):
        ('incremental = driver.capabilities().get("incremental_index", True)',
         'if index.build_mode == "incremental":',
         "driver.create_index(index)",
         "effective_build = build_seconds if not incremental "
         "else load.wall_seconds"),

    # -- §2.6's post-build graph warm-up --------------------------------
    # §2.6 says which queries, at which ef, and which driver declares it.
    ("harness/workloads/graph_warmup.py", "plan"):
        ("test_vectors[:max_queries]", "top_ef(ef_grid, ef_search)"),
    ("harness/workloads/graph_warmup.py", "top_ef"):
        "max([int(ef) for ef in ef_grid] + [int(ef_search)])",
    ("harness/drivers/base.py", "EngineDriver.warm_graph_after_build"):
        "warm_graph_after_build: bool = False",
    ("harness/drivers/mysql_family.py",
     "PerconaDriver.warm_graph_after_build"):
        "warm_graph_after_build: bool = True",

    # -- §2.1, §2.2, §4.1: the dialect ----------------------------------
    ("harness/drivers/mysql_family.py", "to_binary_f32"):
        "def to_binary_f32",
    ("harness/drivers/mysql_family.py",
     "MySQLFamilyDriver._create_table_ddl"):
        "id {self.dialect.pk_type} PRIMARY KEY",
    ("harness/drivers/mysql_family.py", "MySQLFamilyDriver.connect"):
        "host=self.spec.host",
    # §4.1's two claims about the plan check: that the index-name test is a
    # substring test, and that the format is spelled out. Both are inside the
    # one method, and the second is the free hardening the section argues for.
    ("harness/drivers/mysql_family.py",
     "MySQLFamilyDriver.explain_uses_vector_index"):
        ("used = self.dialect.index_name in plan",
         "EXPLAIN FORMAT=TRADITIONAL"),
    ("harness/drivers/mysql_family.py", "MySQLFamilyDriver.capabilities"):
        ('"incremental_index"', '"ef_construction_tunable"'),

    # -- §4.2's registry, with both bare continuations folded in --------
    ("harness/drivers/postgres.py", "_driver_classes"):
        ("def _driver_classes()",
         "from .mysql_family import",
         '"PerconaDriver": PerconaDriver'),
    ("harness/drivers/postgres.py", "_driver_table"):
        "def _driver_table()",

    # -- §4.3's `tag` key, and where it reaches the manifest ------------
    ("orchestrator/engines.py", "_source_tag"):
        'return str(source.get("tag") or source.get("version") or "")',
    ("orchestrator/ops_pass.py", "OpsRun.run_harness"):
        "VB_ENGINE_TAG",

    # -- §4.8's two march readers, both tuples --------------------------
    # The gutter numbers in §4.8's fenced block are gone; the three paths the
    # block reproduces are the anchors, which is what the old
    # `BARE_CONTINUATIONS` row pinned by line and is now pinned by symbol.
    ("harness/main.py", "_detect_march"):
        ('for path in ("/opt/mariadb/.march"',
         '"/opt/pgvector-artifacts/.march"',
         '"/opt/percona/.march"'),
    ("harness/main.py", "main"):
        "args.march or _detect_march()",
    ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
     "VBMySQLBase._image_march"):
        "/opt/mariadb/.march",

    # -- §2.1, §2.2, §7: the ann overlay's copies -----------------------
    ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py", "to_binary_f32"):
        "def to_binary_f32",
    ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
     "Dialect.create_table"):
        "id INT PRIMARY KEY",
    ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
     "VBMySQLBase._conn_kwargs"):
        "unix_socket",
    ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py", "VBMySQLBase.hint"):
        "force_index_hint",
    ("overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py",
     "VBMySQLBase._dim_from_table"):
        "CHARACTER_OCTET_LENGTH",
    ("overlay/ann-benchmarks/ann_benchmarks/algorithms/alisql/module.py",
     "<module>"):
        "Field_vector::store()",

    # -- §4.10's branches, which used to be nine bare line numbers ------
    # Moved here from `tests/test_e10_failures_explain_themselves.py`, which
    # held them as positions in a file and is the test that caught §4.10
    # labelling Percona's own branch as mongodb's. The branch CONDITION is
    # both the citation and the anchor now, so the label and the thing it
    # labels are the same string and cannot disagree.
    # -- §5's two post-rebuild requirements, which name four symbols -----
    # The ann half and the ops half of the same hazard: the recall tree does
    # not move when the pin does, and the provenance table carries one source
    # per engine per RUN DIRECTORY. Each anchor is the thing the sentence
    # claims, not the symbol's name, so a rewrite that keeps the name and
    # drops the behaviour fails here.
    ("orchestrator/ann_pass.py", "ann_fingerprint"):
        ("ANN_MEASUREMENT_VERSION",
         "Deliberately engine-invariant"),
    ("orchestrator/ann_pass.py", "run_engine"):
        "already-computed configurations will be skipped",
    ("report/generate.py", "summarize"):
        '"stale_ann"',
    ("report/render.py", "_engine_rows"):
        'source.get("tag", "?")',
    ("orchestrator/manifest.py", "set_engine"):
        'self.data["engines"][engine] = {',

    ("orchestrator/ann_pass.py", "render_config"):
        ('raise ValueError(f"unknown engine',
         'if engine in MARIADB_ENGINES + ("alisql",)',
         'elif engine == "pgvector":',
         'elif engine == "valkey":',
         'elif engine == "percona":',
         'elif engine == "mongodb":',
         'for build_mode in ann.get("pgvector_build_modes", ["post"]):',
         'for build_mode in ann.get("valkey_build_modes", ["post"]):',
         '"efConstruction": ef_constructions'),

    # -- §4.4's registry golden values ----------------------------------
    ("tests/test_engine_registry.py", "GOLDEN_ORDER"):
        '"mariadb", "alisql", "pgvector", "mariadb123"',

    # -- §4.7's scripts -------------------------------------------------
    ("docker/percona/Dockerfile", "PERCONA_TAG"):
        "ARG PERCONA_TAG=unknown",
    ("scripts/build-images.sh", "build_engine"):
        ("has no known build-arg",
         'percona)  bargs+=(--build-arg "PERCONA_TAG=${tag}") ;;'),
    ("scripts/lib.sh", "assert_not_vendor_repo"):
        'die "refusing to write inside read-only vendor repo: $canon"',
    ("scripts/prepare-sources.sh", "prepare_percona"):
        "git archive",

    # -- §4.9's entrypoint defect, in all three files -------------------
    ("docker/alisql/entrypoint-alisql.sh", "start_server"):
        'local u; u="$(user_args)"',
    ("docker/mariadb/entrypoint-mariadb.sh", "start_server"):
        'local u; u="$(user_args)"',
    ("docker/percona/entrypoint-percona.sh", "start_server"):
        'local u; u="$(user_args)"',
    ("docker/alisql/entrypoint-alisql.sh", "user_args"):
        '[[ "$(id -u)" -eq 0 ]] && printf',

    # -- §3's supplied row count, and the sibling document it quotes ----
    ("harness/datasets.py", "tag_threshold_for"):
        "1,183,514 training rows",
    ("docs/07-planning-a-run.md", "8. A note on the numbers in this document"):
        "60,000-row smoke",

}

#: Repository code files docs/09 NAMES but makes no claim about the contents
#: of, with what covers each instead. Every entry is asserted to be cited by
#: the document and asserted NOT to have a `CITED_SYMBOLS` row, so an entry
#: that acquires a row is a contradiction rather than a silent overlap.
#:
#: THE HONEST LIMIT OF THE FORWARD CHECK, named here because it is the same
#: limit the old `BARE_CONTINUATIONS` note recorded about itself: this list
#: says the document makes no CONTENTS claim about these files, and nothing can
#: check that. A sentence asserting something about `orchestrator/cli.py` could
#: be added tomorrow and no test would ask for a row. What the list does buy is
#: that a NEW path cited by the document fails here until somebody decides
#: which side of the line it is on, which is the decision that used to be made
#: by not noticing.
PATHS_WITHOUT_A_CLAIM = {
    "config/engines/percona.yml":
        "§4.3 proposes this file rather than quoting it, and the claims §3 "
        "and §4.6 do make about it -- the memory flag, the capability keys -- "
        "are checked against the file itself by TestGraphMemoryFallback and "
        "TestBuildArgConventions.",
    "config/engines/mariadb.yml":
        "named in §4.3 as the config task C1 had already stripped: history, "
        "not a claim about its current contents.",
    "config/engines/alisql.yml":
        "named in §4.3 as the file the proposed config is modelled on.",
    "config/profiles/smoke.yml":
        "§4.6 names its 20,000-row subset; that figure is §3's input table, "
        "where it is marked supplied.",
    "config/profiles/main.yml":
        "named in §4.6 as a profile the memory policy has to be right for.",
    "config/profiles/percona.yml":
        "named in §4.6 as the profile to write; it is the follow-on task's.",
    "config/profiles/full.yml":
        "named in §2.5 as one of the profiles carrying "
        "`pgvector_build_modes`; the key, not the file, is the claim.",
    "config/resources/normalized.yml":
        "§3 names its `graph_cache_fraction` to say what feeding Percona's "
        "ceiling from it would do -- an argument about a policy, not a "
        "quotation.",
    "orchestrator/cli.py":
        "§4.10 records that it had to change and says nothing about what is "
        "in it.",
    "orchestrator/estimate.py":
        "§4.6 names `_INGEST_ROWS_PER_S` and `_effective_rate` as the reason "
        "Percona has no entry. docs/10 §5 item 2 is where that file's "
        "contents are quoted, and tests/test_e3_rollout_doc.py checks them.",
    "overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/module.py":
        "the module this task wrote. Its emitted DDL, query, metric map, "
        "ef_search statement, range checks and plan marker are CALLED and "
        "asserted on by the test classes above, which is stronger than any "
        "quotation would be.",
    "overlay/ann-benchmarks/ann_benchmarks/algorithms/percona/config.yml":
        "checked by TestOverlayConfig, which loads it.",
    "overlay/ann-benchmarks/ann_benchmarks/algorithms/pgvector/module.py":
        "§4.10 names it and valkey's as the two modules that DO read "
        "`build_mode`, which is a claim about the string being present; "
        "TestOverlayConfig's siblings in tests/test_config.py own those files.",
    "overlay/ann-benchmarks/ann_benchmarks/algorithms/valkey/module.py":
        "as above.",
    "docker/percona/init.sql":
        "\u00a74.7 marks it done and \u00a77 item 2 says what it does NOT contain. "
        "The "
        "claims are negative -- no `mysql_native_password` line, no "
        "measurement parameter -- and a negative claim about a file is not a "
        "quotation from a symbol in it. "
        "TestInitSqlDoesNotAskForARemovedPlugin and TestGraphMemoryFallback "
        "check them against the file, each paired with a positive assertion "
        "that the needle is findable.",
    "docker/alisql/init.sql":
        "named in \u00a77 item 2 as the file whose line must NOT be copied; "
        "the "
        "same negative claim, checked the same way.",
    "tests/test_c1_engine_sql.py":
        "§4.3 names it as the test a new `sql:` block would fail. That is a "
        "claim about a task in flight, not about a file's contents.",
    "tests/test_e1_percona_design.py":
        "this file. §6 describes it, and a row would be this guard checking "
        "itself -- the shape docs/11 §0 records as worse than no guard at all.",
    "tests/test_e10_failures_explain_themselves.py":
        "§4.10 names it as the guard on its branch list: a pointer to a test, "
        "not a quotation from one.",
    "tests/test_e15_methodology_roadmap_doc.py":
        "§6 names it as the source of the classifier this file imports.",
    "webui/static/fields.js":
        "§4.5 deliberately cites no location in it and says why: the two maps "
        "grow by a line per engine, so a citation into them rots on every "
        "addition while pointing at nothing a reader could not find by name "
        "in a twelve-line file.",
}

#: Paths docs/09 names that do not exist and must not, with the reason. Same
#: mechanism as `ABSENT_PATHS` in tests/test_e15_methodology_roadmap_doc.py,
#: and for the same reason: asserted absent rather than skipped, so a file that
#: lands makes the sentence saying it has not fail here instead of misleading a
#: reader.
ABSENT_PATHS = {
    "run-manifest.json":
        "written into a run directory by orchestrator/manifest.py and never "
        "committed; `results/` is gitignored. §3 names it as where a declared "
        "knob lands. If this resolves, somebody checked a run artefact in.",
}

#: Code-font spans that look like a symbol, are not one, and could not be
#: reworded out without making a sentence worse. Each is asserted NOT to
#: resolve: an entry that resolves is a real symbol being excused, which is how
#: an escape hatch becomes a hole.
#:
#: Empty, and that is a finding rather than a default. Every vendor identifier
#: docs/09 names moved into italics with the conversion, which §0 states as a
#: rule -- so none needed an entry. The vendor-looking names that stayed in
#: code font stayed because this repository really does name them:
#: `innodb_hnsw_max_memory` is a flag in `config/engines/percona.yml` and
#: `ER_CAPACITY_EXCEEDED` is in the overlay module.
VOCABULARY = {}

def cited_repo_files(text):
    """Every file in this repository the document names, resolved.

    Over ALL code-font spans and not only the ones `_PATH_RE` accepts, because
    `docker/percona/Dockerfile` has no extension and `_PATH_RE` is anchored on
    one. A span that resolves to no file, or to more than one, is not in this
    set -- `TestEverySymbolAndPathResolves` is where those are reported.
    """
    spans = {s.strip() for s in re.findall(r"`([^`\n]+)`",
                                           _FENCE_RE.sub("", text))}
    out = {}
    for span in spans:
        if "/" not in span and "." not in span:
            continue
        # A directory in code font is prose about a tree -- `harness/`,
        # `docker/percona/` -- and `os.path.isfile` is what keeps one out of
        # this set. The classifier reaches the same conclusion by a different
        # route, recognising a path only by its extension, and
        # tests/test_e15_methodology_roadmap_doc.py records the consequence:
        # a deep directory citation into a generated tree is as uncheckable
        # as a file citation into one, and is not flagged.
        if span.endswith("/"):
            continue
        resolved, _hits = resolve_path(span)
        if resolved:
            out.setdefault(resolved, set()).add(span)
    return out


class TestTheCitationParsersFoundSomething:
    """Floors and sentinels, because every assertion below is over a set.

    A parser that matched nothing would make this whole section pass against
    an empty document. One guard in this series shipped green because it
    scanned zero files, and the old `_CITE_RE`'s own note records a live error
    hidden by a pattern that could not see a whole class of citation. Both
    failure modes are the same shape and both are checked here.
    """

    def test_the_index_read_real_files(self):
        idx = symbol_index()
        assert len(idx["files"]) >= 150, len(idx["files"])
        assert len(idx["mentioned"]) >= 5000, len(idx["mentioned"])
        assert len(idx["declared"]) >= 1500, len(idx["declared"])

    def test_the_index_holds_things_it_must_hold(self):
        """Sentinels by KIND, one per route into the index, so a route that
        quietly produced nothing is visible: a function, a constant, a class
        and a YAML key."""
        idx = symbol_index()
        for name in ("to_binary_f32", "OPS_MEASUREMENT_VERSION",
                     "PerconaDriver", "bulk_vector_index_build"):
            assert name in idx["declared"], name
            assert name in idx["mentioned"], name

    def test_the_index_reads_none_of_the_three_document_guards(self):
        """Measured, not supposed, and all three rather than this one.

        tests/test_e15_methodology_roadmap_doc.py records what happens when a
        guard's index includes the guard: its planted bad symbol RESOLVED,
        because the plant's names were spelled in the file doing the
        resolving, so the arm passed the live document and failed the plant.
        These three files describe each other -- this one imports that one's
        classifier -- so any one of them left in the index can certify a
        citation that exists nowhere but in a test.
        """
        read = set(symbol_index()["files"])
        for guard in GUARD_FILES:
            assert guard not in read, guard

    def test_the_index_does_not_read_the_documentation(self):
        """`docs/` is excluded on purpose: a symbol that exists only in prose
        must not resolve, or this becomes a check that docs/09 agrees with
        docs/05. Both mechanisms are asserted, because
        tests/test_e15_methodology_roadmap_doc.py records that naming one of
        two is dead against a change to the other."""
        read = symbol_index()["files"]
        assert not any(f.startswith("docs/") for f in read)
        assert ".md" not in INDEXED_SUFFIXES

    def test_the_index_reads_nothing_from_a_generated_tree(self):
        """A symbol that resolves only against a fetched vendor tree resolves
        for the author and for nobody else. `sources/` is where the Percona
        tree would sit if it were ever fetched here."""
        read = symbol_index()["files"]
        for prefix in GENERATED_DIRS:
            assert not any(f.startswith(prefix + "/") for f in read), prefix
        ignore = open(os.path.join(VB_ROOT, ".gitignore")).read()
        rules = {line.strip() for line in ignore.splitlines()}
        for prefix in GENERATED_DIRS:
            assert "/" + prefix + "/" in rules, prefix
        # Non-vacuity: a `rules` set that came back empty would certify any
        # prefix at all.
        assert "/nonexistent-zq/" not in rules

    def test_the_classifier_found_paths_and_symbols(self):
        """Floors with about a tenth of slack, and no exact counts.

        An exact figure in a comment is checked by nobody and is false two
        commits later -- which is the #41/#43 mechanism, and writing it inside
        the file that exists to stop it is how this series learned the lesson
        twice.
        """
        paths, symbols, prose = doc_spans(_doc_09())
        assert len(paths) >= 45, sorted(paths)
        assert len(symbols) >= 180, len(symbols)
        assert prose, "nothing classified as prose; the classifier widened"

    def test_the_classifier_strips_fenced_quotations(self):
        """The fence rule, measured. docs/09 quotes a Percona source comment
        that spells `online` in backticks; counting that as a citation would
        ask this repository to declare a symbol because the vendor's source
        does."""
        paths, symbols, _prose = doc_spans(
            "prose `harness/main.py`\n\n```\n/* `no_such_symbol_zq` and "
            "`no/such/file_zq.py` */\n```\n")
        assert paths == {"harness/main.py"}
        assert "no_such_symbol_zq" not in symbols

    def test_a_commit_sha_is_not_classified_as_a_symbol(self):
        """§0's commit table quotes twelve-character SHAs in code font. They
        are hex, they start with a letter often enough, and `_SYMBOL_RE`
        cannot tell them from an identifier."""
        _paths, symbols, prose = doc_spans("`a86503214416` and `effective_ef`")
        assert symbols == {"effective_ef"}
        assert "a86503214416" in prose

    def test_the_document_names_the_repository_files_it_claims_about(self):
        """The resolver, exercised on the document rather than on a fixture."""
        found = cited_repo_files(_doc_09())
        assert len(found) >= 35, sorted(found)
        for sentinel in ("harness/drivers/mysql_family.py",
                         "docker/percona/Dockerfile",
                         "overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py"):
            assert sentinel in found, sentinel

    def test_the_suffix_resolver_refuses_what_it_should(self):
        """Unique-suffix resolution is the one convenience in this scheme, so
        its edges are pinned: an abbreviation resolves, a missing file does
        not, and an ambiguous one does not either -- `module.py` names six
        files under `overlay/` and must come back unresolved rather than
        picking one."""
        assert resolve_path("build.py")[0] == "harness/workloads/build.py"
        assert resolve_path("harness/main.py")[0] == "harness/main.py"
        assert resolve_path("no_such_file_zq.py")[0] is None
        resolved, hits = resolve_path("module.py")
        assert resolved is None and len(hits) > 1, hits


class TestTheDocumentMakesNoLineCitation:
    """The policy, executable. This is the keystone: a policy stated only in
    prose is a policy until the first hurry."""

    def test_docs_09_contains_no_line_citation(self):
        found = line_citations(_doc_09())
        assert not found, (
            "docs/09 cites a line number, which its own §0 forbids -- line "
            "citations in this repository rotted twice in two days (#41, "
            "#43) and the vendor ones rotted unseen. Name the symbol "
            "instead: " + ", ".join(sorted(set(found))))

    def test_the_detector_sees_every_form_it_has_to_see(self):
        """Non-vacuity, and not optional: the assertion above is NEGATIVE, and
        a negative assertion passes just as happily against a broken pattern
        as against a clean document.

        The fixture is a frozen literal rather than a slice of a document,
        because a fixture read out of the tree can come back empty and an
        empty fixture turns this test into the vacuity it exists to catch.
        """
        found = line_citations(CITATION_FIXTURE)
        assert len(found) == CITATION_FIXTURE_EXPECTED, found
        joined = " ".join(found)
        # One assertion per form, so a pattern that lost ONE of the four
        # widenings fails with the form it lost named.
        assert "foo.py:12" in joined
        assert "bar/baz.result:13-15" in joined           # MTR result
        assert "sql_yacc.yy:9" in joined                  # grammar
        assert "docker/percona/Dockerfile:725" in joined  # no extension
        assert ":701-707" in joined                       # bare continuation
        assert "r/x.result:3-5, :13-17, :23-26" in joined  # several ranges

    def test_the_detector_does_not_fire_on_correct_prose(self):
        """The other failure mode. A guard that fires on correct writing
        teaches the next author to delete it, which is how the vacuity shapes
        reached this repository in the first place."""
        assert not line_citations(
            "`max(ef, k)` `p50/p95/max` `harness/main.py` `x86-64-v3` "
            "`effective_ef()` `{'cause': 'vector'}` `M = 6.0`")

    def test_the_policy_is_stated_in_the_document(self):
        """The reason, not only the rule. A rule whose reason lives in a test
        file gets relaxed by the next author to find it inconvenient."""
        doc = _doc_09()
        assert "#41" in doc and "#43" in doc, (
            "docs/09 §0 must justify the policy with the two pull requests "
            "that repaired the rot; without them it is a style preference")
        assert re.search(r"invalidated by \*\*any\*\* edit", doc)
        assert re.search(
            r"only by a\s*\*\*rename\*\*|\*\*only by a\s*rename\*\*", doc)

    def test_the_illustration_of_the_banned_form_is_not_in_code_font(self):
        """The one place docs/09 has to SHOW a line citation is the paragraph
        banning it, and the ban is a pattern over code-font spans. Italics
        rather than an allowlist, exactly as docs/11 §0 chose: a carve-out in
        the ban is a carve-out, and the document would then be the one file
        permitted to make the citation it forbids."""
        assert re.search(r"\*some/file\.py:123\*", _doc_09()), (
            "docs/09 §0 must illustrate the banned form in italics; in code "
            "font it trips its own ban")

    def test_the_document_records_what_the_old_form_could_not_check(self):
        """§6's load-bearing admission. The conversion's real argument is not
        that line numbers rot -- it is that the vendor two thirds of them were
        checked by nothing and would have rotted in silence at the re-pin. A
        later tidy-up that drops this leaves the conversion looking like
        housekeeping."""
        doc = _doc_09()
        assert re.search(r"checked by nothing", doc, re.I)
        assert "164" in doc, (
            "§0 and §6 state the size of the unchecked half; without a figure "
            "the sentence is an impression")
        assert re.search(r"cannot go red", doc, re.I)

    def test_no_fenced_block_prints_a_gutter_number(self):
        """The form the ratchet cannot see, with its own check.

        A gutter number is not inside a code-font span, so `_LINE_CITATION_RE`
        -- which is deliberately a pattern over code-font spans -- cannot see
        one. That left the whole class to the author, and the author got it
        wrong: this change cleared eight of the eleven blocks that carried
        gutters, said in a commit message that all of them were clean, and
        left four vendor line numbers in §1 eleven lines below a sentence
        asserting they were gone. Eleven is the corrected census and the
        history only reconstructs at eleven -- 8, then 2, then the last 1 --
        and "ten" survived in this docstring and in §1 after the commit
        message and the pull request had both been corrected, which is the
        same one-place-and-not-the-other failure this whole change is about.

        The hand scan missed them because it anchored the fence at column 0
        and that block is indented inside a bullet.

        So the class is closed here instead of trusted. Reproducing a source
        with its gutters is citing it by line with extra steps: the numbers go
        stale at a re-pin exactly as a citation does, and nothing about a
        fenced block makes them easier to check.
        """
        fenced = False
        offenders = []
        blocks = 0
        for n, line in enumerate(_doc_09().splitlines(), 1):
            if line.lstrip().startswith("```"):
                if not fenced:
                    blocks += 1
                fenced = not fenced
                continue
            if not fenced or not _GUTTER_LINE_RE.match(line):
                continue
            if line.strip() in RECORDED_VALUE_LINES:
                continue
            offenders.append(f"{n}: {line.strip()[:70]}")
        # Non-vacuity: the scan has to have read the document's fences at all.
        assert blocks >= 18, blocks
        assert not offenders, (
            "docs/09 reproduces source gutter numbers inside a fenced block. "
            "They are line citations the ratchet cannot see and they rot at a "
            "re-pin like any other. Delete the prefixes and keep the lines:\n  "
            + "\n  ".join(offenders))

    def test_the_recorded_value_lines_are_named_and_still_there(self):
        """The escape hatch, in both directions.

        An entry that no longer appears is dead allowance -- the next line to
        open with that number would be waved through -- and an entry with no
        reason is an exemption nobody can audit. Both are checked, which is
        the shape `VOCABULARY` and `ABSENT_PATHS` use for the same reason.
        """
        doc = _doc_09()
        for line, reason in RECORDED_VALUE_LINES.items():
            assert reason.strip(), line
            assert any(raw.strip() == line for raw in doc.splitlines()), (
                f"RECORDED_VALUE_LINES carries {line!r} and docs/09 no longer "
                f"has that line; delete the entry rather than leaving an "
                f"allowance behind")

    def test_the_gutter_scan_sees_an_indented_fence(self):
        """The defect that let the four through, as a test rather than a note.

        The scan that missed them anchored the fence at column 0. This asserts
        the opposite behaviour on a fence indented inside a bullet, which is
        exactly where docs/09 §1's excerpt lives.
        """
        indented = "- a bullet:\n  ```\n  127  --echo # knob\n  ```\n"
        fenced = False
        seen = []
        for line in indented.splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced and _GUTTER_LINE_RE.match(line):
                seen.append(line.strip())
        assert seen == ["127  --echo # knob"], seen

class TestEverySymbolAndPathResolves:
    """The existence layer: coarser than the table below, over everything."""

    def test_every_cited_path_resolves_to_one_file(self):
        paths, _symbols, _prose = doc_spans(_doc_09())
        bad = []
        for path in sorted(paths):
            if path in ABSENT_PATHS:
                continue
            resolved, hits = resolve_path(path)
            if resolved is None:
                why = f"ambiguous, {hits}" if hits else "no such file"
                bad.append(f"{path}: {why}")
        assert not bad, (
            "docs/09 cites these paths and this tree cannot resolve them. A "
            "vendor path belongs in prose, not in code font -- §0 says so:\n  "
            + "\n  ".join(bad))

    def test_every_cited_symbol_appears_in_the_tree(self):
        _paths, symbols, _prose = doc_spans(_doc_09())
        bad = sorted(s for s in symbols
                     if s not in VOCABULARY
                     and s not in symbol_index()["mentioned"])
        assert not bad, (
            "docs/09 names these symbols in code font and nothing in this "
            "tree has them -- renamed, never existed, or a vendor name that "
            "belongs in italics per §0: " + ", ".join(bad))

    def test_enough_symbols_resolve_to_a_declaration(self):
        """The half that keeps the half above from being satisfied by prose. A
        word match accepts a symbol surviving only inside a comment, and a
        design document is unusually likely to name one."""
        _paths, symbols, _prose = doc_spans(_doc_09())
        declared = sorted(s for s in symbols
                          if s in symbol_index()["declared"])
        assert len(declared) >= 120, (
            f"only {len(declared)} of {len(symbols)} cited symbols resolve to "
            f"a declaration")

    def test_the_vocabulary_escapes_are_named_and_do_not_resolve(self):
        """An escape hatch that excuses a real symbol is a hole, not a hatch."""
        idx = symbol_index()
        for span, reason in VOCABULARY.items():
            assert reason.strip(), span
            assert span not in idx["mentioned"], (
                f"{span} is in VOCABULARY and resolves in the tree; it is a "
                f"citation, not vocabulary -- delete the entry")
        assert VOCABULARY == {}, (
            "VOCABULARY has entries; name them in the round report rather "
            "than letting them sit unchecked: " + ", ".join(VOCABULARY))

    @pytest.mark.parametrize("path", sorted(ABSENT_PATHS))
    def test_a_deliberately_absent_path_is_still_absent(self, path):
        assert ABSENT_PATHS[path].strip(), path
        assert not os.path.exists(os.path.join(VB_ROOT, path)), (
            f"{path} now exists; docs/09 describes it as generated rather "
            f"than committed: {ABSENT_PATHS[path]}")

    @pytest.mark.parametrize("path", sorted(ABSENT_PATHS))
    def test_a_deliberately_absent_path_is_actually_cited(self, path):
        """A row for a path the document stopped naming reads as coverage and
        is not."""
        paths, _symbols, _prose = doc_spans(_doc_09())
        assert path in paths, (
            f"ABSENT_PATHS carries {path} and docs/09 no longer names it")

    def test_no_vendor_path_is_cited_in_code_font(self):
        """§0's second rule, as a check rather than as a promise.

        This is the ONE place the conversion is strictly stronger about the
        vendor half, and it is why `FOREIGN_MARKERS` is gone rather than
        ported: a deny-list of vendor filenames existed to say "these are not
        checked". Under the policy a vendor path in code font is a violation,
        not a category, so the path arm above fails on it and there is nothing
        left for a deny-list to do. The spellings below are the ones the old
        deny-list carried, asserted against the live document.
        """
        paths, _symbols, _prose = doc_spans(_doc_09())
        retired_deny_list = (
            "percona-hnsw", "ha_innodb", "handler0alter", "field.cc",
            "field.h", "item_create", "sys_vars.cc", "vec0vec",
            "vector_constants", "CMakeLists", "keyring_kmip", "mysql-test",
            "sql_yacc", "ddl0loader",
        )
        offenders = sorted(p for p in paths
                           if any(m in p for m in retired_deny_list))
        assert not offenders, (
            "these vendor paths are in code font in docs/09; §0 requires them "
            "in prose, because a backticked path there means a path in this "
            "repository: " + ", ".join(offenders))
        # Non-vacuity: the document still TALKS about those files, in italics,
        # so a conversion that deleted the vendor evidence would fail here.
        doc = _doc_09()
        for vendor in ("*ha_innodb.cc*", "*handler0alter.cc*", "*field.cc*",
                       "*percona-hnsw.md*", "*vec0vec.h*",
                       "*vector_constants.h*", "*sys_vars.cc*",
                       "*sql_yacc.yy*", "*item_create.cc*",
                       "*CMakeLists.txt*", "*ddl0loader.cc*"):
            assert vendor in doc, (
                f"docs/09 no longer names {vendor} at all; the conversion was "
                f"supposed to move vendor material into prose, not delete it")


class TestEveryClaimAboutThisRepoIsOpenedAndMatched:
    """The symbol table: the old line check, re-keyed.

    Not "the symbol exists" -- the symbol is located and the string the
    document points at is asserted to be inside it. That is the same strength
    as re-opening a line range, which is the point: the conversion must not
    buy survivability by giving up the contents check.
    """

    @pytest.mark.parametrize("key", sorted(CITED_SYMBOLS))
    def test_the_symbol_holds_what_the_document_says(self, key):
        path, symbol = key
        anchors = CITED_SYMBOLS[key]
        if isinstance(anchors, str):
            anchors = (anchors,)
        full = os.path.join(VB_ROOT, path)
        assert os.path.isfile(full), f"{path} does not exist"
        body = symbol_body(path, symbol)
        assert body is not None, (
            f"docs/09 cites {symbol!r} in {path} and that file no longer "
            f"declares it -- renamed, or moved to another file. This is the "
            f"failure a symbol citation exists to catch.")
        for anchor in anchors:
            assert anchor in body, (
                f"docs/09 cites {path}'s {symbol!r} for {anchor!r}; that "
                f"symbol now reads: {body.strip()[:160]!r}")

    def test_every_repo_code_file_the_document_names_is_accounted_for(self):
        """Completeness, forward: the half that makes this closed rather than
        a snapshot. A file the document starts naming lands here until
        somebody decides whether it carries a claim."""
        named = cited_repo_files(_doc_09())
        with_rows = {path for path, _symbol in CITED_SYMBOLS}
        unaccounted = sorted(
            path for path in named
            if path not in with_rows
            and path not in PATHS_WITHOUT_A_CLAIM
            and not path.startswith("docs/")
            # THE CARVE-OUT, AND WHY IT IS NARROWER THAN IT LOOKS. The old
            # forward check was keyed on CITATIONS and had no exemption: every
            # `path:line` into this repo needed a row. This one is keyed on
            # FILES, so it needs one, because both documents name sibling
            # documents and the root files constantly -- `CLAUDE.md` for an
            # invariant, `README.md` for a profile table, `docs/05` and
            # `docs/07` for a figure's provenance -- and requiring a contents
            # row for each would force a row for every passing reference.
            #
            # What the exemption does NOT do is leave those uncovered. The
            # rows that matter are still here: each `.md` row names a section
            # of a sibling document, opens it, and matches the sentence quoted
            # from it. The exemption only removes the REQUIREMENT, and it
            # removes it for the class of reference that is a pointer rather
            # than a claim.
            #
            # NO COUNT OF THOSE ROWS IS STATED HERE, and that is deliberate
            # rather than vague. This note was pasted into both guards saying
            # "this guard carries three `.md` rows and docs/09's carries one".
            # Measured: the docs/09 guard carries ONE and the docs/10 guard
            # carries SIX, across two files -- so "three" was false in both
            # copies, and the second clause was true only when read from one
            # of them. A figure in a comment is checked by nobody; the floor
            # below is checked on every run, which is the whole argument this
            # change exists to make.
            #
            # The honest cost: a contents claim about a `docs/` path with no row
            # is not detected, exactly as a contents claim about a file in
            # PATHS_WITHOUT_A_CLAIM is not. Both are the same limit and it
            # is the one this check cannot close.
            and path != "CLAUDE.md")
        assert not unaccounted, (
            "docs/09 names these repository files and nothing says whether it "
            "claims anything about them. Give each a CITED_SYMBOLS row or a "
            "PATHS_WITHOUT_A_CLAIM reason: " + ", ".join(unaccounted))

    def test_no_row_cites_a_file_the_document_dropped(self):
        """Completeness, reverse: a row for something the document no longer
        names is dead weight that reads as coverage. It is how
        `ddl0loader.cc:273` was caught sitting unclassified the first time the
        old guard ran."""
        named = cited_repo_files(_doc_09())
        stale = sorted({path for path, _symbol in CITED_SYMBOLS}
                       - set(named))
        assert not stale, (
            "these rows cite files docs/09 no longer names: "
            + ", ".join(stale))

    def test_every_identifier_row_names_a_symbol_the_document_cites(self):
        """The other half of reverse completeness, at symbol granularity.

        Scoped to identifier-shaped symbols, and the scope is the honest part:
        a `.md` row's symbol is a section HEADING, which the citing document
        points at as "§3" rather than by quoting the heading, so requiring the
        heading text in docs/09 would fire on correct prose. Those rows are
        covered by the path half above and by their own anchor check.
        """
        _paths, symbols, _prose = doc_spans(_doc_09())
        bad = []
        for path, symbol in sorted(CITED_SYMBOLS):
            if path.endswith(".md") or symbol == "<module>":
                continue
            leaf = symbol.rsplit(".", 1)[-1]
            if leaf not in symbols:
                bad.append(f"{path}:{symbol}")
        assert not bad, (
            "these rows name symbols docs/09 does not cite in code font, so "
            "the row is checking something the document stopped saying: "
            + ", ".join(bad))

    @pytest.mark.parametrize("path", sorted(PATHS_WITHOUT_A_CLAIM))
    def test_a_path_without_a_claim_is_named_and_really_has_no_row(self, path):
        assert PATHS_WITHOUT_A_CLAIM[path].strip(), path
        with_rows = {p for p, _symbol in CITED_SYMBOLS}
        assert path not in with_rows, (
            f"{path} is listed as carrying no contents claim and also has a "
            f"CITED_SYMBOLS row; one of the two is wrong")
        assert path in cited_repo_files(_doc_09()), (
            f"{path} is listed as named-without-a-claim and docs/09 no longer "
            f"names it; delete the entry")

    @pytest.mark.parametrize("key", sorted(CITED_SYMBOLS))
    def test_the_anchors_are_not_empty(self, key):
        """`CITED_SYMBOLS`'s values ARE the assertions, and nothing checked
        that one existed.

        `assert "" in body` is true of every body, so a row whose anchor was
        an empty string -- or a tuple containing one, which is the likelier
        typo -- would pass against anything the symbol happened to contain and
        read as coverage. Every other escape hatch in this file is asserted
        non-empty (`VOCABULARY`, `ABSENT_PATHS`, `PATHS_WITHOUT_A_CLAIM` all
        check their reasons); this was the table it was missing from, and it
        is the table that matters most. Inherited from the line-citation
        version, which had it too.

        A floor on LENGTH as well as on emptiness, because a one- or
        two-character anchor is the same defect slightly diluted: `"("` is in
        most function bodies.
        """
        anchors = CITED_SYMBOLS[key]
        if isinstance(anchors, str):
            anchors = (anchors,)
        assert anchors, f"{key} has no anchor at all"
        for anchor in anchors:
            assert isinstance(anchor, str), (key, anchor)
            # NON-SPACE characters, which is what the docstring says and
            # what `len(anchor.strip())` did not do: `"a    b"` strips to six
            # and is two characters of evidence.
            assert len("".join(anchor.split())) >= 6, (
                f"{key}'s anchor {anchor!r} is too short to be evidence; an "
                f"anchor this small is satisfied by most bodies")

    def test_the_markdown_rows_open_a_real_section(self):
        """The premise of the carve-out above, checked instead of counted.

        The forward check exempts every `docs/` path, and the note beside it
        says the rows that matter are still here. That is a claim about this
        table, so it is asserted: at least one `.md` row, every one of them
        resolving to a section that exists, and its anchor inside that
        section. The note used to carry a COUNT of these rows instead and the
        count was wrong in both guards -- which is why there is a floor here
        and no number there.
        """
        md = sorted(k for k in CITED_SYMBOLS if k[0].endswith(".md"))
        assert md, (
            "no `.md` rows at all, so the carve-out note's claim that sibling "
            "documents are still covered by their own anchors is false")
        for path, heading in md:
            body = symbol_body(path, heading)
            assert body is not None, (
                f"{path} has no section matching {heading!r}; a sibling "
                f"document's heading was renumbered or reworded")
            anchors = CITED_SYMBOLS[(path, heading)]
            for anchor in ((anchors,) if isinstance(anchors, str) else anchors):
                assert anchor in body, (path, heading, anchor)

    def test_the_symbol_locator_can_fail(self):
        """The locator is what every row above rests on, so it is proved able
        to come back empty. A locator that returned the whole file would make
        every anchor pass, which is the vacuity this file has already been
        bitten by: a guard that sliced a function's TEXT was satisfied by the
        method's docstring and stayed green with the real tuple emptied.
        """
        assert symbol_body("harness/main.py", "no_such_symbol_zq") is None
        body = symbol_body("harness/main.py", "_detect_march")
        assert body is not None and "/opt/percona/.march" in body
        # And it is a SLICE, not the file: `main`'s body must not leak in.
        assert "args.march or _detect_march()" not in body

    def test_the_locator_handles_every_file_kind_the_table_uses(self):
        """One assertion per resolver, because each is a different parser and
        a silent None from any of them would be reported as a renamed symbol
        rather than as a broken locator."""
        checks = (
            ("harness/drivers/base.py", "EngineDriver.repair_path",
             'repair_path: str = "none"'),                       # python
            ("harness/workloads/build.py", "<module>",
             "ingest_wall_s"),                                   # docstring
            ("scripts/lib.sh", "assert_not_vendor_repo",
             "refusing to write inside read-only vendor repo"),   # shell
            ("config/engines/percona.yml", "capabilities",
             "bulk_vector_index_build"),                          # yaml
            ("docker/percona/Dockerfile", "PERCONA_TAG",
             "ARG PERCONA_TAG"),                                  # generic
            ("docs/07-planning-a-run.md",
             "8. A note on the numbers in this document",
             "60,000-row smoke"),                                 # markdown
        )
        for path, symbol, needle in checks:
            body = symbol_body(path, symbol)
            assert body is not None, (path, symbol)
            assert needle in body, (path, symbol, body[:120])

    def test_the_comment_block_above_a_declaration_is_part_of_its_body(self):
        """Measured, because several rows depend on it: this repository puts
        the rationale for a constant in the `#:` block above it, and that is
        what the documents quote."""
        body = symbol_body("harness/drivers/base.py",
                           "EngineDriver.repair_path")
        assert "Valkey and Percona Search inherit" in body
        assert 'repair_path: str = "none"' in body


class TestTheMemoryArithmeticReproduces:
    """docs/09 §3's table, recomputed from the formula it cites.

    The formula is the floor an MTR test asserts the server stays above:
    mysql-test/suite/percona_vector/r/hnsw_memory_used.result:63 is
    `SET @floor = 1000 * (64 * 4 + 2 * 16 * 8);` for 1000 nodes of VECTOR(64) at
    M = 16, generalising to rows * (dims * 4 + 2 * M * 8).

    These are NOT measurements and the document says so: rows, dims and M are
    all supplied, from this repo's dataset facts and profiles. What is checked
    here is only that the numbers printed are the ones the formula gives -- an
    arithmetic slip in a table that a reader will use to size a two-day run.
    """

    GIB = 1073741824

    @staticmethod
    def _floor(rows: int, dims: int, m: int) -> int:
        return rows * (dims * 4 + 2 * m * 8)

    @pytest.fixture(scope="class")
    def rows(self):
        """The §3 table, parsed out of the document."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1]
        out = []
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 7 or not re.fullmatch(r"[\d,]+|per 1M", cells[1]):
                continue
            out.append(cells)
        assert out, "could not parse the §3 table"
        return out

    def test_the_table_was_found_and_has_every_row(self, rows):
        """Non-vacuity: a parser that matched nothing would make every
        assertion below pass silently."""
        assert len(rows) == 5

    def test_every_floor_matches_the_formula(self, rows):
        for cells in rows:
            _, raw_rows, dims, m, floor, _, _ = cells
            n = 1_000_000 if raw_rows == "per 1M" else int(raw_rows.replace(",", ""))
            want = self._floor(n, int(dims), int(m))
            got = int(re.sub(r"[^\d]", "", floor))
            assert got == want, f"{cells[0]}: doc says {got:,}, formula gives {want:,}"

    def test_every_ratio_matches_its_floor(self, rows):
        for cells in rows:
            floor = int(re.sub(r"[^\d]", "", cells[4]))
            claimed = float(re.sub(r"[^\d.]", "", cells[5]))
            assert round(floor / self.GIB, 3) == claimed, cells[0]

    def test_the_over_the_default_column_follows_from_the_numbers(self, rows):
        """A row whose floor exceeds the default and whose verdict says "no"
        would be the dangerous kind of error: it reads as reassurance."""
        for cells in rows:
            floor = int(re.sub(r"[^\d]", "", cells[4]))
            verdict = cells[6].lower()
            if floor > self.GIB:
                assert verdict.startswith("**yes") or verdict.startswith("yes"), cells
            else:
                assert "floor no" in verdict or verdict == "no", cells

    def test_section_3_states_a_decided_policy_not_a_recommendation(self):
        """The human chose policy (1). §3 has to read as settled.

        Polarity, because the failure mode is a reader relitigating a decision
        whose reasoning is still on the page: "two defensible policies, the
        choice is the user's" is exactly right *before* the choice and
        misleading after it.
        """
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1].split(
            "## 4.", 1)[0]
        assert "### The policy: `innodb_hnsw_max_memory = 0`" in section
        assert "**This is decided, not proposed.**" in section
        assert "CHOSEN" in section
        for stale in ("### The recommendation",
                      "This document recommends (1)",
                      "the choice is the user's because it is a fairness "
                      "judgement and not a fact"):
            assert stale not in section, stale
        # The justification survives the decision -- both policies still argued.
        assert "graph_cache_bytes`, accepting that some configurations will "\
               "fail" in section

    def test_the_node_map_caveat_is_attached_to_the_gauge(self):
        """The caveat has to sit where a reader takes a figure from the
        variable, not in a footnote elsewhere: the chosen policy's audit trail
        IS that variable, and it is short by one node map per vector index."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1].split(
            "## 4.", 1)[0]
        assert "*percona-hnsw.md*" in section
        assert "lower bound" in section
        # The projection, with both ends, and marked as a projection.
        assert "35.5 to 59.2 MB" in section
        assert "projection" in section
        assert "supplied" in section

    def test_the_node_map_projection_reproduces(self):
        """Arithmetic, from the repo's own row count and Percona's stated range.

        Not a measurement and §3 says so: the row count is supplied from
        harness/datasets.py:492 and 30-50 bytes/node is Percona's own estimate.
        """
        datasets = open(os.path.join(VB_ROOT, "harness", "datasets.py")).read()
        assert "1,183,514 training rows" in datasets
        rows = 1183514
        assert rows * 30 == 35_505_420
        assert rows * 50 == 59_175_700
        assert round(rows * 30 / 1e6, 1) == 35.5
        assert round(rows * 50 / 1e6, 1) == 59.2

    def test_section_3_keeps_the_line_that_the_gate_cannot_surface_this(self):
        """The most load-bearing sentence in §3 for anyone reading a green
        smoke run as evidence the memory setting is fine."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1].split(
            "## 4.", 1)[0]
        assert "the gate cannot surface this" in section
        assert "0.19" in section

    def test_the_default_is_the_one_the_server_ships(self):
        """innodb_hnsw_max_memory_basic.result:3-5 reads 1073741824."""
        doc = open(PERCONA_DOC).read()
        assert "1073741824" in doc
        assert self.GIB == 1073741824

    def test_the_smoke_profile_really_does_fit_under_the_default(self):
        """§3's conclusion that the gate will not surface the ceiling, taken
        from the profile rather than from the document."""
        import yaml
        profile = yaml.safe_load(open(os.path.join(
            VB_ROOT, "config", "profiles", "smoke.yml")))
        subset = profile["ops"]["subset_rows"]
        m_values = profile["ann"]["m_values"]
        assert subset == 20000 and m_values == [16]
        # fashion-mnist-784: dims from the dataset name the profile asks for.
        assert profile["datasets"] == ["fashion-mnist-784-euclidean"]
        assert self._floor(subset, 784, max(m_values)) < self.GIB
        # ...and the ann pass loads the whole 60,000-row training set.
        assert self._floor(60000, 784, max(m_values)) < self.GIB

    def test_the_main_profile_does_not(self):
        """The finding that changes what can be run: glove-100 at M=32 exceeds
        the default on the floor alone, so the load is guaranteed to stop.

        The row count is this repo's own, quoted at harness/datasets.py:492.
        """
        datasets = open(os.path.join(VB_ROOT, "harness", "datasets.py")).read()
        assert "1,183,514 training rows" in datasets
        assert self._floor(1183514, 100, 32) > self.GIB
        # At M=16 the floor fits but the MTR's own 2x ceiling does not, which is
        # why §3 calls that configuration a coin-flip rather than safe.
        assert self._floor(1183514, 100, 16) < self.GIB
        assert 2 * self._floor(1183514, 100, 16) > self.GIB
