"""The Percona build inputs: the cmake flag list, its mirror, and the apt list.

WHY THIS FILE EXISTS
--------------------
2026-10-14. The first `./run-benchmark.sh build --engines percona` ever run
failed configure five times in a row, and the shape of the fix changed on the
way through. Each of the first two was an option left at a default that is
only correct on a machine carrying development packages this image does not
install, and each was "fixed" by switching the option off:

    CMake Error at CMakeLists.txt:2075 (MESSAGE):
      -DWITH_AUTHENTICATION_LDAP=ON, but missing system libraries
    CMake Error at CMakeLists.txt:2127 (MESSAGE):
      -DWITH_CURL=system, but missing system libraries.
      You can disable CURL with -DWITH_CURL=0

The third had no off switch at all -- `CMakeLists.txt:2224-2226` is
`IF(LINUX) MYSQL_CHECK_RPC() ENDIF()` and `cmake/rpc.cmake:106-110` is a bare
`IF(NOT RPC_INCLUDE_DIRS)` ending in FATAL_ERROR -- and the last two showed
why switching things off was the wrong strategy on this tree:

    CMake Error at plugin/auth_ldap/CMakeLists.txt:21 (MESSAGE):
      Required LDAP dev library not found.
    CMake Error at components/keyrings/keyring_vault/CMakeLists.txt:33:
      Not building Keyring Vault Component, could not find CURL library

The first of those is a PERCONA plugin keyed on a different symbol
(`WITH_PERCONA_AUTHENTICATION_LDAP`, CMakeLists.txt:2004-2011, default ON off
APPLE), which the LDAP flag never reached. The second was CAUSED by
`-DWITH_CURL=0`: the component is ON by default and tests `CURL_FOUND`, so
removing the target broke a Percona component that wanted it.

The fix for all of that landed separately, as commit 5fe565e (PR #22),
and this file is the half that commit did not carry: nothing in
`tests/` compared the two copies of the flag list, which is how they
could drift. What #22 changed, and what is asserted below:

  * the BUILDER stage's apt list gains thirteen packages, taken from the
    vendor's own `install_deps()` in
    `build-ps/percona-server-9.0_builder.sh` filtered to noble, which is this
    image's base. It is a superset and was not pruned -- see the Dockerfile's
    apt note. Eleven landed in 5fe565e; a8b8785 (PR #23) added the last two
    after a --no-cache rebuild with `libsasl2-dev` present STILL died at
    :2075, because `WARN_MISSING_SYSTEM_SASL` wants a SCRAM *plugin* from
    `libsasl2-modules` and not the dev package's headers. A check whose
    message says "development libraries" wanting a runtime plugin is the
    sharpest thing this episode turned up.
  * `-DWITH_CURL=bundled`, which is the value Percona's own
    `build-ps/build-binary.sh:304` uses. `=0` and `=system` are the two
    spellings that failed above.
  * NO authentication flag. Intermediate attempts at this fix passed
    `-DWITH_AUTHENTICATION_LDAP=OFF`, and #22 carries none: with libldap2-dev,
    libsasl2-dev and libkrb5-dev installed, the warnings that fed :2075 do not
    fire, so such a flag would gate nothing.
    `FORBIDDEN_FLAG_SYMBOLS` asserts it and its two obvious companions stay
    out, and says what would have to change to re-add one.

The package assertions are anchored to the BUILDER stage specifically. This
Dockerfile has three stages (builder, runtime, bench) and a whole-file grep
would be satisfied by a package sitting in the wrong one, which would not
build -- the mutation run for this file includes exactly that mutant.

THE FLAG LIST LIVES IN TWO PLACES THAT NOTHING KEPT IN STEP:

  * `config/engines/percona.yml` -> `build.cmake_flags`, which
    `scripts/build-images.sh:64 cmake_flags_for()` reads and passes as the
    CMAKE_FLAGS build-arg. This is the live path -- it is what
    `./run-benchmark.sh build` uses.
  * `docker/percona/Dockerfile`'s hardcoded list in the builder RUN, whose own
    comment says "Flags mirror config/engines/percona.yml". This is what a
    by-hand `docker build` uses, and docs/03-running-manually.md documents
    doing exactly that.

The Dockerfile appends `${CMAKE_FLAGS}` AFTER its own flags, so a value in the
YAML can confirm or override one above it but can never remove it. That makes
drift between the two lists silent in the worst direction: the framework path
and the by-hand path would compile different servers, and nothing would say so.
Before this file, `grep -rn cmake_flags tests/` returned nothing.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
Nothing here builds anything, and no build of this image has ever succeeded.
These are agreement guards over two text files plus one YAML parse: that the
two copies of the flag list say the same thing, that the one flag this change
adds is in both with the vendor's value, that the removed flags stay removed,
and that the eleven packages are declared in the stage that compiles. They
establish nothing about Percona Server, and they cannot. Every claim about
what cmake does with these options is a file:line on commit
20c57915626b0214bbfe49718beccc0b6ebd51a6 plus the operator's own configure
output. Those readings were taken at the 9.7.1 base and carried through two
re-pins; at this one (9.7.2-2 to 9.7.4-4, a fast-forward) the cited lines
were re-read against the new tree and two ranges renumbered, but no
configure has been run against it. The comment above the builder's apt list in
`docker/percona/Dockerfile`, and the one beside `build.cmake_flags` in
`config/engines/percona.yml`, carry that reading at length.

THE VACUITY SHAPES THESE GUARDS ARE BUILT TO AVOID
-------------------------------------------------
Every one of these is live in this file rather than hypothetical, which is why
each is asserted rather than assumed:

1. **A symbol match that cannot tell a right value from a wrong one.** This is
   the big one here, because `-DWITH_CURL` has three spellings that all look
   like fixes: `=bundled` is correct, `=system` caused the
   CMakeLists.txt:2127 fatal and `=0` caused the keyring_vault:33 fatal. A
   guard written `assert "WITH_CURL" in text`, or one checking only that the
   symbol is passed, is satisfied by all three. `REQUIRED_FLAGS` therefore
   pins symbol AND value, and
   `test_matching_the_curl_symbol_alone_cannot_tell_right_from_wrong` asserts
   that the Dockerfile really does carry `-DWITH_CURL=0` in its prose while
   passing `=bundled` as an argument -- and that the correct value appears in
   the prose too, so even counting the right string over the whole file
   settles nothing.
2. **Prefix containment, where the needle prefixes a different flag.**
   `-DWITH_R` prefixes both `-DWITH_ROCKSDB=OFF` and `-DWITH_ROUTER=OFF`, and
   `-DWITH` prefixes `-DWITHOUT_COMPONENT_KEYRING_KMIP=1` as well as every
   `-DWITH_*`. So prefix matching cannot name a flag in this list; every
   comparison is exact equality on a parsed token, and
   `test_a_symbol_prefix_cannot_identify_a_flag` asserts the ambiguity.
   (A review that fed into this file offered `WITH_LDAP` as a substring of
   `WITH_AUTHENTICATION_LDAP`. It is not one -- after `WITH_` comes
   `AUTHENTICATION`. `test_the_reviewers_substring_example_is_not_the_real_trap`
   records that it was checked rather than believed. The shape is real; that
   instance of it is not.)
3. **A text search that finds the flag in the prose documenting it.** The
   Dockerfile header quotes all five cmake errors verbatim, so
   `-DWITH_AUTHENTICATION_LDAP=ON`, `-DWITH_CURL=0` and `-DWITH_CURL=system`
   -- every wrong value, two of them for the flag that IS passed -- are in the
   file, and so is the correct `-DWITH_CURL=bundled`. A whole-file grep
   therefore proves
   nothing about what is passed and can prove the opposite of the truth; it is
   also why the removed-flag assertion works on parsed tokens, since
   `assert "-DWITH_AUTHENTICATION_LDAP" not in dockerfile_text` would fail
   against a correct tree. Both parsers are bounded to the builder's cmake
   invocation and `TestTheProseAroundTheInvocationIsNotMistakenForArguments`
   asserts the bounding works.
4. **A package assertion that cannot see the wrong stage.** The Dockerfile has
   three stages and only `builder` runs cmake, so `libtirpc-dev` in `runtime`
   would satisfy a grep and still fail to build. The package tests parse
   per-stage apt lists, and the mutation run includes the move-to-runtime
   mutant specifically.
5. **A package name that is a prefix of another package name.**
   `libsasl2-modules` is a strict prefix of `libsasl2-modules-gssapi-mit`, and
   `libsasl2` prefixes both plus `libsasl2-dev`. A guard written
   `any(p.startswith("libsasl2-modules") ...)` is satisfied by the gssapi
   package alone -- leaving the SCRAM plugin missing, :2075 still firing, and
   the test green. Package assertions are whole-token membership of a parsed
   install list, each name separately, and
   `test_the_sasl_names_collide_by_prefix_in_both_directions` asserts the
   collision is real.
6. **A value vocabulary assumed to be `OFF`.** `-DWITH_CURL=bundled`,
   `-DWITHOUT_COMPONENT_KEYRING_KMIP=1`, `-DWITH_SSL=system` and
   `-DWITH_ZLIB=system` do not end in `OFF`. A check that split on `"=OFF"`
   would skip four flags including the one this change adds. Values are parsed
   with `partition("=")`, and `test_the_value_vocabulary_is_mixed` fails if
   that stops being necessary.
7. **An absence assertion that is really a policy in disguise.**
   `FORBIDDEN_FLAG_SYMBOLS` asserts three flags stay out. That is a claim
   about THIS pin -- their gates do not fire with the dev packages installed,
   and two of them route through a `WITH_INTERNAL` that is 0 because
   `internal/CMakeLists.txt` is absent from the published tree. It is not
   "these are wrong forever": the constant's comment says what would have to
   change to re-add one, so a future pin that needs them gets an argument to
   make rather than a red test to delete.
8. **`'"X"' and "Y"` evaluating to `"Y"`.** No assertion below is a
   conjunction of string literals. Expectations live in a dict and three
   frozensets, and the non-vacuity classes assert the parsers returned
   something before anything else asks what they returned.
"""

from __future__ import annotations

import ast
import os
import re

import pytest
import yaml

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCONA_YML = os.path.join(VB_ROOT, "config", "engines", "percona.yml")
PERCONA_DOCKERFILE = os.path.join(VB_ROOT, "docker", "percona", "Dockerfile")

# The one flag this change adds, symbol -> value, spelled exactly as both
# copies must carry it. The value is Percona's own: build-ps/build-binary.sh:304
# is `-DWITH_CURL=bundled`, which satisfies
# components/keyrings/keyring_vault/CMakeLists.txt:32 by building extra/curl.
#
# THE VALUE IS THE WHOLE CLAIM HERE, which is why it is kept apart from the
# symbol. `-DWITH_CURL` has three spellings that all look like a fix and two of
# them break the build: `=system` is the default that produced the
# CMakeLists.txt:2127 fatal, and `=0` produced the keyring_vault:33 fatal. A
# guard matching the symbol, or the substring `WITH_CURL`, passes on all three.
REQUIRED_FLAGS = {
    "-DWITH_CURL": "bundled",
}

# Symbols that must NOT be in either list. Intermediate attempts at this fix
# passed `-DWITH_AUTHENTICATION_LDAP=OFF` and 5fe565e ships none of them,
# because the builder now installs libldap2-dev, libsasl2-dev and libkrb5-dev,
# so the warnings that fed the CMakeLists.txt:2075 fatal do not fire and such
# a flag would gate nothing.
# The other two are listed with it because they are the obvious companions to
# re-add, and they would be equally dead: both defaults route through
# `WITH_INTERNAL`, which is 0 whenever `internal/CMakeLists.txt` is absent, and
# it is absent from the published tree at this pin.
#
# This is an absence assertion, a shape worth justifying. It is not "these
# flags are wrong forever" -- it is "this list does not switch authentication
# plugins off, and a flag that gates nothing does not belong in a list whose
# header claims every entry cites the line that made it necessary". If a future
# pin makes one load-bearing, this constant is where that argument gets made,
# with the file:line that changed.
FORBIDDEN_FLAG_SYMBOLS = frozenset({
    "-DWITH_AUTHENTICATION_LDAP",
    "-DWITH_AUTHENTICATION_KERBEROS",
    "-DWITH_AUTHENTICATION_CLIENT_PLUGINS",
})

# Development headers and runtime plugins the builder stage installs, from the
# vendor's own `install_deps()` in build-ps/percona-server-9.0_builder.sh
# filtered to noble. It is a SUPERSET, not a minimal set -- the Dockerfile's
# apt note names the entries that are probably redundant and says why it was
# not pruned.
#
# THIRTEEN, AND THE LAST TWO ARE NOT DEV PACKAGES AT ALL. A --no-cache rebuild
# with libsasl2-dev installed and LDAP found still died at
# CMakeLists.txt:2075, because `WARN_MISSING_SYSTEM_SASL` returns quietly only
# when `FIND_LIBRARY(NAMES scram PATH_SUFFIXES sasl2)` finds a SCRAM *plugin*:
# libsasl2-dev ships sasl/sasl.h, libsasl2.so and libsasl2.a and nothing under
# sasl2/, while the unversioned sasl2/libscram.so symlink that FIND_LIBRARY
# needs comes from libsasl2-modules, a runtime package. A check whose message
# says "development libraries" wants a plugin, so the fix is two more packages
# rather than a different -dev one. (5fe565e had eleven; a8b8785 has thirteen.)
BUILDER_REQUIRED_PACKAGES = frozenset({
    "libldap2-dev", "libsasl2-dev", "libkrb5-dev", "libcurl4-openssl-dev",
    "libtirpc-dev", "libpam0g-dev", "liblz4-dev", "libzstd-dev",
    "libevent-dev", "libre2-dev", "libwrap0-dev",
    "libsasl2-modules", "libsasl2-modules-gssapi-mit",
})

# The three SASL package names, which collide by prefix in both directions and
# are the reason every package assertion here is whole-token.
#
#   libsasl2-dev                  headers; was present when :2075 still fired
#   libsasl2-modules              the SCRAM plugin the check actually wants
#   libsasl2-modules-gssapi-mit   a different plugin; CONTAINS the name above
#
# A check for the substring `libsasl2` is satisfied by any one of them. A
# check for `libsasl2-modules` as a substring or prefix is satisfied by
# `libsasl2-modules-gssapi-mit` alone -- which would leave the SCRAM plugin
# missing and :2075 still firing, with a green test. That is not hypothetical:
# the first of these three being present while the second was absent is
# exactly what cost the rebuild.
SASL_PACKAGES = (
    "libsasl2-dev",
    "libsasl2-modules",
    "libsasl2-modules-gssapi-mit",
)

# The spelling the vendor script uses, which must never appear here: it is
# virtual on noble (provided by libpam0g-dev), so an edit that "corrects" the
# list to match build-ps must fail here and not in a two-hour build.
VENDOR_VIRTUAL_PAM_NAME = "libpam-dev"

# Wrong values the Dockerfile header quotes out of the five cmake errors. Each
# must be findable in the FILE and unfindable in the ARGUMENTS; that gap is
# what says the parsers are bounded to the invocation rather than grepping
# documentation. Two of them are wrong values for a flag that IS passed, which
# is precisely what makes a substring check on `WITH_CURL` worthless.
WRONG_VALUES_QUOTED_IN_PROSE = (
    "-DWITH_AUTHENTICATION_LDAP=ON",
    "-DWITH_CURL=0",
    "-DWITH_CURL=system",
)

# The CORRECT value is quoted in the header too (the apt note explains the
# package list "with -DWITH_CURL=bundled and the vendor's bundled
# libevent/zstd"), so even counting occurrences of the right string across the
# whole file cannot establish what cmake is told. Only the bounded parse can.
RIGHT_VALUE_ALSO_QUOTED_IN_PROSE = "-DWITH_CURL=bundled"

# The Dockerfile legitimately passes seven flags the YAML does not, and this is
# the allowlist that says which. They are not policy: four name where this
# particular image puts things, one names the build it is, and two hand cmake
# the builder stage's own exported CFLAGS -- so none of them is a decision the
# YAML could sensibly express or a resource pass could vary.
#
# The last two are the ones with a trap behind them. `-DCMAKE_C_FLAGS=` and
# `-DCMAKE_CXX_FLAGS=` are half of the fix for the DEB package-flags overwrite
# (docker/percona/Dockerfile's note above its cmake line, and
# tests/test_e8_package_flags_overwrite.py): their value is `"$CFLAGS"`, a
# shell expansion of a variable exported two lines earlier in the same RUN. A
# YAML copy could only hold a literal, and because `${CMAKE_FLAGS}` is appended
# LAST that literal would WIN -- quietly replacing the flags the builder
# exported with whatever this file happened to say. So they belong here and the
# YAML must stay silent about them. `-DWITH_PACKAGE_FLAGS=OFF`, the other half,
# IS mirrored in the YAML, because a duplicate of it can only confirm.
#
# This set is asserted to be EXACTLY the Dockerfile-only symbols, not merely to
# contain them. An eighth Dockerfile-only flag added later therefore fails here
# rather than slipping in, which is the drift this file exists to stop; and an
# entry that goes stale because the Dockerfile stopped passing it fails too.
DOCKERFILE_ONLY_SYMBOLS = frozenset({
    "-DCMAKE_BUILD_TYPE",
    "-DCMAKE_INSTALL_PREFIX",
    "-DMYSQL_DATADIR",
    "-DSYSCONFDIR",
    "-DMYSQL_SERVER_SUFFIX",
    "-DCMAKE_C_FLAGS",
    "-DCMAKE_CXX_FLAGS",
})


def _split(flag: str) -> tuple[str, str]:
    """`-DWITH_SSL=system` -> `("-DWITH_SSL", "system")`.

    `partition` and not `split("=")`: a value may itself contain `=`, and a
    flag with no `=` at all (`-DFOO`) must come back with an empty value
    rather than raising, so the mirror can still report it. Nothing here
    assumes the value is `OFF` -- see vacuity shape 3 in the module docstring.
    """
    symbol, _, value = flag.partition("=")
    return symbol, value


@pytest.fixture(scope="module")
def yaml_flags() -> list[str]:
    """`build.cmake_flags`, parsed as YAML and not grepped as text.

    Grepping would pass against a flag that is present but commented out, and
    this config's own comment block quotes both failing values out of the cmake
    errors -- so a text search for either symbol finds a hit in the prose
    whichever way the real list is set.
    """
    with open(PERCONA_YML) as handle:
        cfg = yaml.safe_load(handle)
    return list((cfg or {}).get("build", {}).get("cmake_flags") or [])


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    with open(PERCONA_DOCKERFILE) as handle:
        return handle.read()


@pytest.fixture(scope="module")
def cmake_region(dockerfile_text: str) -> str:
    """Just the builder RUN's cmake invocation.

    Bounded at both ends on purpose. The header above it quotes `-D` flags in
    prose -- note 2's submodule table and the apt-list comment's account of
    the configure failures both do -- so a scan of the whole file would
    collect flags that are documentation and not arguments, including a
    value that caused one of the failures.
    """
    assert "cmake /src/source" in dockerfile_text, (
        "the builder's cmake invocation was not found; this test's anchor is "
        "stale or the Dockerfile was restructured")
    after = dockerfile_text.split("cmake /src/source", 1)[1]
    assert "make -j" in after, (
        "no 'make -j' after the cmake invocation to bound the region on")
    return after.split("make -j", 1)[0]


@pytest.fixture(scope="module")
def dockerfile_flags(cmake_region: str) -> list[str]:
    r"""Flags as passed: one per continuation line, anchored at line start.

    Anchoring matters. An unanchored `-D\S+` would also match inside
    `-march=${MARCH}` style text and inside any `-D` that appears mid-line.
    """
    return re.findall(r"^\s*(-D\S+)", cmake_region, re.MULTILINE)


# ---------------------------------------------------------------------------
# Non-vacuity: the parsers found something before anything asks what
# ---------------------------------------------------------------------------
#
# Most assertions further down are of the form "this set difference is empty",
# and two empty inputs satisfy all of them. A renamed file, a restructured
# Dockerfile or a `build:` key that parses to `{}` would otherwise turn this
# whole module green over nothing. These run first for that reason.

class TestTheInputsAreActuallyThere:

    def test_both_files_exist(self):
        assert os.path.isfile(PERCONA_YML), PERCONA_YML
        assert os.path.isfile(PERCONA_DOCKERFILE), PERCONA_DOCKERFILE

    def test_the_yaml_flag_list_is_not_empty(self, yaml_flags):
        assert yaml_flags, (
            "build.cmake_flags parsed to nothing -- an empty list, a renamed "
            "key or a file that is no longer the engine config")

    def test_the_dockerfile_text_is_not_empty(self, dockerfile_text):
        assert dockerfile_text.strip(), "the Dockerfile read back empty"

    def test_the_dockerfile_cmake_region_is_not_empty(self, cmake_region):
        assert cmake_region.strip(), (
            "the region between 'cmake /src/source' and 'make -j' is blank")

    def test_the_dockerfile_flag_list_is_not_empty(self, dockerfile_flags):
        assert dockerfile_flags, (
            "no -D flags parsed out of the builder's cmake invocation")

    def test_neither_list_is_implausibly_short(self, yaml_flags,
                                               dockerfile_flags):
        """A parse that returns one flag is not obviously broken the way one
        that returns zero is, so the floor is set above the trivial case.

        Ten is below both real counts and above anything a partial parse would
        plausibly produce; it is a smoke bound, not a pin on the contents.
        """
        assert len(yaml_flags) >= 10, yaml_flags
        assert len(dockerfile_flags) >= 10, dockerfile_flags

    def test_no_flag_is_duplicated_in_either_list(self, yaml_flags,
                                                  dockerfile_flags):
        """Set arithmetic below hides duplicates, and a duplicated flag with
        two different values is exactly the drift this file is about: the
        Dockerfile's `${CMAKE_FLAGS}` append means the LAST one wins, so two
        copies of one symbol make the effective value depend on ordering.
        """
        for label, flags in (("yaml", yaml_flags),
                             ("dockerfile", dockerfile_flags)):
            symbols = [_split(f)[0] for f in flags]
            dupes = sorted({s for s in symbols if symbols.count(s) > 1})
            assert not dupes, f"{label} repeats: {dupes}"


# ---------------------------------------------------------------------------
# The flags the two incidents required
# ---------------------------------------------------------------------------

class TestTheFlagsTheConfigureFailuresRequired:
    """Both flags, in both copies, with the right values. Asserted as exact
    membership of a parsed list and exact equality on the value."""

    @pytest.mark.parametrize("symbol,value", sorted(REQUIRED_FLAGS.items()))
    def test_the_yaml_carries_it(self, symbol, value, yaml_flags):
        assert f"{symbol}={value}" in yaml_flags, (
            f"{symbol}={value} missing from config/engines/percona.yml "
            f"build.cmake_flags; this is the live path that "
            f"./run-benchmark.sh build reads. Got: {yaml_flags}")

    @pytest.mark.parametrize("symbol,value", sorted(REQUIRED_FLAGS.items()))
    def test_the_dockerfile_carries_it(self, symbol, value, dockerfile_flags):
        assert f"{symbol}={value}" in dockerfile_flags, (
            f"{symbol}={value} missing from docker/percona/Dockerfile's "
            f"builder cmake invocation, so a by-hand `docker build` would "
            f"still fail configure. Got: {dockerfile_flags}")

    @pytest.mark.parametrize("source", ["yaml", "dockerfile"])
    @pytest.mark.parametrize("symbol,value", sorted(REQUIRED_FLAGS.items()))
    def test_the_value_is_the_disabling_one(self, symbol, value, source,
                                            yaml_flags, dockerfile_flags):
        """Presence is not the claim.

        `-DWITH_AUTHENTICATION_LDAP=ON` and `-DWITH_CURL=system` are the
        strings the two cmake errors print, and they are the defects. A test
        that only checked the symbol was present would pass on both.
        """
        flags = yaml_flags if source == "yaml" else dockerfile_flags
        values = [v for s, v in map(_split, flags) if s == symbol]
        assert values == [value], (
            f"{source}: expected exactly one {symbol} with value {value!r}, "
            f"got {values!r}")

    def test_the_value_vocabulary_is_mixed(self, yaml_flags):
        """Why `partition("=")` and not a trailing-`OFF` filter.

        Four of these flags do not end in `OFF`: `-DWITH_CURL=bundled`,
        `-DWITHOUT_COMPONENT_KEYRING_KMIP=1`, `-DWITH_SSL=system` and
        `-DWITH_ZLIB=system`. The first is the flag this change exists for, so
        a check that split on `"=OFF"` or filtered for a trailing `OFF` would
        miss precisely the regression it was written to catch. This fails if
        the vocabulary ever collapses to `OFF`, at which point the generality
        elsewhere can be dropped deliberately rather than forgotten.
        """
        values = {v for _, v in map(_split, yaml_flags)}
        non_off = sorted(values - {"OFF"})
        assert len(non_off) >= 3, (
            f"expected several non-OFF values, got {sorted(values)}")
        # ...and every required flag's value is among them, so the generality
        # is protecting the flag this change adds and not only its neighbours.
        for value in REQUIRED_FLAGS.values():
            assert value in values, (value, sorted(values))
            assert value != "OFF", REQUIRED_FLAGS

    def test_a_symbol_prefix_cannot_identify_a_flag(self, yaml_flags):
        """The prefix ambiguity that makes exact matching necessary, asserted.

        `-DWITH_R` is a prefix of both `-DWITH_ROCKSDB=OFF` and
        `-DWITH_ROUTER=OFF`, and `-DWITH` is a prefix of
        `-DWITHOUT_COMPONENT_KEYRING_KMIP=1` as well as of every `-DWITH_*`.
        So prefix matching cannot name a flag in this list, and this fails if
        that stops being true -- at which point the strictness everywhere else
        can be relaxed deliberately rather than by accident.
        """
        for prefix, least in (("-DWITH_R", 2), ("-DWITH", 2)):
            hits = sorted(f for f in yaml_flags if f.startswith(prefix))
            assert len(hits) >= least, (
                f"{prefix!r} no longer collides; hits={hits}")

    def test_matching_the_curl_symbol_alone_cannot_tell_right_from_wrong(
            self, yaml_flags, dockerfile_text):
        """Why `REQUIRED_FLAGS` pins a value and not a symbol.

        All three spellings of this flag appear in the Dockerfile -- `=bundled`
        as an argument, `=system` and `=0` quoted in the header out of the two
        fatals they caused. A guard written `assert "WITH_CURL" in text`, or
        one checking only that the symbol is passed, is satisfied by every one
        of them, including the two that break the build. This test asserts
        that the ambiguity is real, so the exactness elsewhere is visibly
        load-bearing rather than stylistic.
        """
        for wrong in ("-DWITH_CURL=0", "-DWITH_CURL=system"):
            assert wrong in dockerfile_text, (
                f"the header no longer quotes {wrong!r}, one of the two "
                f"values of this flag that failed a real configure, so this "
                f"test has stopped discriminating")
        # The right value is in the prose as well, so even a count of the
        # correct string over the whole file proves nothing about arguments.
        assert dockerfile_text.count(RIGHT_VALUE_ALSO_QUOTED_IN_PROSE) >= 2, (
            dockerfile_text.count(RIGHT_VALUE_ALSO_QUOTED_IN_PROSE))
        passed = [f for f in yaml_flags if _split(f)[0] == "-DWITH_CURL"]
        assert passed == ["-DWITH_CURL=bundled"], passed

    def test_no_other_flag_contains_the_curl_symbol(self, yaml_flags,
                                                    dockerfile_flags):
        """`-DWITH_CURL` is a substring of nothing else in either list, which
        is luck rather than design: adding a `-DWITH_CURL_PATH` later would
        make any substring check elsewhere ambiguous, and this makes that
        visible instead of silent."""
        for label, flags in (("yaml", yaml_flags),
                             ("dockerfile", dockerfile_flags)):
            carriers = sorted(f for f in flags if "-DWITH_CURL" in f)
            assert carriers == ["-DWITH_CURL=bundled"], (
                f"{label}: more than one flag now carries the text "
                f"'-DWITH_CURL': {carriers}")

    @pytest.mark.parametrize("symbol", sorted(FORBIDDEN_FLAG_SYMBOLS))
    @pytest.mark.parametrize("source", ["yaml", "dockerfile"])
    def test_the_authentication_flags_are_absent(self, symbol, source,
                                                 yaml_flags,
                                                 dockerfile_flags):
        """The flags an earlier revision added and this one removed.

        Parsed tokens and not a text search, which matters more here than
        anywhere else in this file: the Dockerfile header QUOTES
        `-DWITH_AUTHENTICATION_LDAP=ON` out of the CMakeLists.txt:2075 error,
        so `assert "-DWITH_AUTHENTICATION_LDAP" not in dockerfile_text` would
        fail against a correct tree and teach the next author to delete the
        guard.
        """
        flags = yaml_flags if source == "yaml" else dockerfile_flags
        present = sorted(f for f in flags if _split(f)[0] == symbol)
        assert not present, (
            f"{source}: {symbol} is back in the list as {present}. With "
            f"libldap2-dev, libsasl2-dev and libkrb5-dev installed in the "
            f"builder it gates nothing; see FORBIDDEN_FLAG_SYMBOLS for the "
            f"argument and what would have to change to re-add it.")

    def test_the_forbidden_set_is_not_vacuously_satisfied(self, yaml_flags):
        """Non-vacuity for the absence test above.

        An empty `FORBIDDEN_FLAG_SYMBOLS`, or a parse returning nothing, would
        satisfy it over no content. The symbols are also checked to be
        well-formed, so a typo'd entry cannot pass by never matching anything.
        """
        assert FORBIDDEN_FLAG_SYMBOLS, "the forbidden set is empty"
        assert yaml_flags, "nothing parsed to check against"
        for symbol in FORBIDDEN_FLAG_SYMBOLS:
            assert symbol.startswith("-D"), symbol
            assert "=" not in symbol, (
                f"{symbol} carries a value; absence is asserted on the symbol "
                f"so that =ON and =OFF are both caught")

    def test_the_reviewers_substring_example_is_not_the_real_trap(self):
        """Recorded because it was checked, not because it was believed.

        A review that fed into this file offered `WITH_LDAP` as a substring of
        `WITH_AUTHENTICATION_LDAP`. It is not one -- after `WITH_` comes
        `AUTHENTICATION`, not `LDAP`. The shape of the warning was right and
        the instance was wrong, and leaving a false example in a docstring is
        how a correct guard gets deleted by someone who tests the example.
        """
        assert "WITH_LDAP" not in "WITH_AUTHENTICATION_LDAP"


# ---------------------------------------------------------------------------
# The prose quotes the wrong values; the parsers must not see them
# ---------------------------------------------------------------------------

class TestTheProseAroundTheInvocationIsNotMistakenForArguments:
    """The reason both parsers are bounded, asserted rather than trusted.

    `docker/percona/Dockerfile`'s apt-list comment quotes the cmake errors
    verbatim, so the file contains `-DWITH_AUTHENTICATION_LDAP=ON` and
    `-DWITH_CURL=0` -- values that broke the build -- as well as the correct
    `-DWITH_CURL=bundled`. A whole-file text search finds the wrong value as
    readily
    as the right one, so it cannot establish what cmake is actually told.
    """

    @pytest.mark.parametrize("wrong", WRONG_VALUES_QUOTED_IN_PROSE)
    def test_the_wrong_value_is_in_the_file(self, wrong, dockerfile_text):
        """If this fails the documentation stopped quoting the failure, and
        the test below stops proving anything -- fix them together."""
        assert wrong in dockerfile_text, (
            f"{wrong!r} is no longer quoted anywhere in the Dockerfile, so "
            f"the bounding test below has nothing to discriminate against")

    @pytest.mark.parametrize("wrong", WRONG_VALUES_QUOTED_IN_PROSE)
    def test_the_wrong_value_is_not_an_argument(self, wrong, dockerfile_flags):
        assert wrong not in dockerfile_flags, (
            f"{wrong!r} is being passed to cmake as an argument, not merely "
            f"quoted in the header prose. Parsed: {dockerfile_flags}")

    def test_the_prose_lives_above_the_region_the_parser_reads(
            self, dockerfile_text, cmake_region):
        """Belt and braces on the bound itself: the quoted errors are in the
        header, and the header is not in the parsed region."""
        for wrong in WRONG_VALUES_QUOTED_IN_PROSE:
            assert wrong not in cmake_region, (
                f"{wrong!r} appears inside the builder's cmake invocation "
                f"region, so the bounding is not separating prose from "
                f"arguments")
        assert len(cmake_region) < len(dockerfile_text), (
            "the 'region' is the whole file; the bounding did nothing")


def _apt_packages(stage_text: str) -> frozenset[str]:
    """Package names from every `apt-get install` in one stage.

    Truncated at the first `&&` after `install`, because the runtime stage
    chains `&& useradd ...` onto its install and those tokens are not
    packages. Option words are dropped by their leading dash.
    """
    found: set[str] = set()
    for body in re.findall(r"apt-get install((?:[^\n]*\\\n)*[^\n]*)",
                           stage_text):
        for token in body.replace("\\\n", " ").split("&&")[0].split():
            if token.startswith("-") or token in {";", "\\"}:
                continue
            found.add(token)
    return frozenset(found)


@pytest.fixture(scope="module")
def dockerfile_stages(dockerfile_text) -> dict:
    """Stage name -> that stage's text, split on `FROM ... AS <name>`.

    The split is what makes the package assertion specific. `libtirpc-dev` in
    the runtime or bench stage would satisfy a whole-file grep and still fail
    to build, because it is the builder that runs cmake.
    """
    marks = list(re.finditer(r"^FROM\s+\S+\s+AS\s+(\w+)\s*$",
                             dockerfile_text, re.MULTILINE))
    assert marks, "no named build stages found in the Dockerfile"
    stages = {}
    for index, mark in enumerate(marks):
        end = (marks[index + 1].start() if index + 1 < len(marks)
               else len(dockerfile_text))
        stages[mark.group(1)] = dockerfile_text[mark.start():end]
    return stages


@pytest.fixture(scope="module")
def builder_packages(dockerfile_stages) -> frozenset:
    assert "builder" in dockerfile_stages, sorted(dockerfile_stages)
    return _apt_packages(dockerfile_stages["builder"])


# ---------------------------------------------------------------------------
# The package the unskippable check needs, in the right stage
# ---------------------------------------------------------------------------

class TestTheBuilderStageInstallsWhatTheUnskippableCheckNeeds:
    """`libtirpc-dev`, and the stage it has to be in.

    Unlike the two flags above, this gate has no off switch:
    `CMakeLists.txt:2224-2226` runs `MYSQL_CHECK_RPC()` on every Linux build
    and `cmake/rpc.cmake:106-110` is a bare `IF(NOT RPC_INCLUDE_DIRS)` ending
    in FATAL_ERROR. So the fix is a build dependency, and the only thing a
    test can usefully hold is that the dependency is declared in the stage
    that compiles.
    """

    def test_the_stage_split_found_the_stages_this_file_has(
            self, dockerfile_stages):
        """Non-vacuity for the split: a regex that matched nothing would make
        every stage lookup below raise rather than silently pass, but a split
        that found only ONE stage would make the bounding meaningless."""
        assert {"builder", "runtime", "bench"} <= set(dockerfile_stages), (
            f"stages found: {sorted(dockerfile_stages)}")

    def test_each_stage_is_a_proper_slice_of_the_file(self, dockerfile_stages,
                                                      dockerfile_text):
        for name, text in dockerfile_stages.items():
            assert text, f"stage {name} is empty"
            assert len(text) < len(dockerfile_text), (
                f"stage {name} is the whole file; the split did nothing")

    def test_the_builder_apt_list_is_not_empty(self, builder_packages):
        """If this parsed to nothing, the membership test below would be
        asserting against an empty set and would fail loudly -- but a parse
        that returned one junk token would not, so the floor is a count."""
        assert len(builder_packages) >= 10, sorted(builder_packages)

    def test_the_builder_apt_list_is_packages_and_not_option_words(
            self, builder_packages):
        """The parse drops flags by their leading dash; this is the assertion
        that says it did, so a future `-y` creeping in as a 'package' cannot
        inflate the count above."""
        assert not [p for p in builder_packages if p.startswith("-")], (
            sorted(builder_packages))
        for expected in ("build-essential", "cmake", "python3"):
            assert expected in builder_packages, sorted(builder_packages)

    @pytest.mark.parametrize("package", sorted(BUILDER_REQUIRED_PACKAGES))
    def test_the_builder_stage_installs_it(self, package, builder_packages):
        assert package in builder_packages, (
            f"{package} is not in the BUILDER stage's apt-get install list, "
            f"so cmake's unskippable MYSQL_CHECK_RPC would still fail at "
            f"rpc.cmake:108. Builder installs: {sorted(builder_packages)}")

    @pytest.mark.parametrize("package", sorted(BUILDER_REQUIRED_PACKAGES))
    def test_a_later_stage_does_not_carry_it_instead(self, package,
                                                     dockerfile_stages):
        """The wrong-stage mutant, which a whole-file grep cannot see.

        The runtime image deliberately carries runtime libraries and no `-dev`
        packages at all -- `libssl3t64` and not `libssl-dev`, and so on -- so a
        build-time `-dev` package appearing there is always a mistake, and the
        build would fail with the package present in the file.
        """
        for name in ("runtime", "bench"):
            if name not in dockerfile_stages:
                continue
            assert package not in _apt_packages(dockerfile_stages[name]), (
                f"{package} is installed in the {name} stage. It is needed by "
                f"cmake in the BUILDER stage; here it only grows the image "
                f"and would leave configure failing.")

    @pytest.mark.parametrize("package", SASL_PACKAGES)
    def test_each_sasl_package_is_installed_under_its_own_exact_name(
            self, package, builder_packages):
        """All three, separately, as whole tokens.

        `libsasl2-dev` alone was the state in which CMakeLists.txt:2075 still
        fired: the check calls
        `FIND_LIBRARY(NAMES scram PATH_SUFFIXES sasl2)` and the dev package
        ships nothing under `sasl2/`. So "a libsasl2 package is installed" is
        not the property that matters; which ones are is.
        """
        assert package in builder_packages, (
            f"{package} is not a whole token in the builder stage's install "
            f"list. Builder installs: {sorted(builder_packages)}")

    def test_the_sasl_names_collide_by_prefix_in_both_directions(self):
        """The hazard that makes whole-token matching necessary, asserted.

        This is a property of the names themselves, so it is checked against
        the names and not the file: `libsasl2-modules` is a strict prefix of
        `libsasl2-modules-gssapi-mit`, and `libsasl2` prefixes all three. A
        guard written `any(p.startswith("libsasl2-modules") for p in pkgs)` is
        therefore satisfied by the gssapi package alone, leaving the SCRAM
        plugin missing and :2075 still firing with a green test.

        If this ever fails, the names have changed and the whole-token
        strictness above can be reconsidered deliberately.
        """
        dev, modules, gssapi = SASL_PACKAGES
        assert gssapi.startswith(modules) and gssapi != modules, (
            modules, gssapi)
        assert len({p for p in SASL_PACKAGES if p.startswith("libsasl2")}) == 3
        # ...and the dev package is NOT a prefix of either plugin, so the
        # collision is specifically between the two plugin names.
        assert not modules.startswith(dev), (dev, modules)

    def test_the_gssapi_package_alone_does_not_satisfy_the_scram_requirement(
            self, builder_packages):
        """The mutant this test exists for, stated as an assertion.

        Written so that a list carrying only `libsasl2-modules-gssapi-mit`
        fails: membership of the exact name `libsasl2-modules` is required
        independently, and the two are asserted to be different packages.
        """
        _dev, modules, gssapi = SASL_PACKAGES
        assert modules in builder_packages, (
            f"{modules} is absent. {gssapi} being present does not provide "
            f"the SCRAM plugin that FIND_LIBRARY(NAMES scram) looks for, and "
            f"a prefix check would not have noticed. Builder installs: "
            f"{sorted(builder_packages)}")
        assert modules != gssapi

    def test_the_vendor_virtual_package_name_is_not_used(self,
                                                         builder_packages):
        """`libpam0g-dev`, never the vendor script's `libpam-dev`.

        build-ps/percona-server-9.0_builder.sh:551 installs `libpam-dev`, and
        on noble that is a VIRTUAL package whose only provider is
        `libpam0g-dev` -- packages.ubuntu.com files it under "virtual". Anyone
        re-deriving this list from the vendor script will copy the virtual
        name, so the failure belongs here and not two hours into a build.
        """
        assert VENDOR_VIRTUAL_PAM_NAME not in builder_packages, (
            f"{VENDOR_VIRTUAL_PAM_NAME!r} is the vendor's spelling and is "
            f"virtual on noble; use the providing package instead. Builder "
            f"installs: {sorted(builder_packages)}")
        assert "libpam0g-dev" in builder_packages, sorted(builder_packages)

    def test_the_required_package_set_is_not_vacuous(self, builder_packages):
        """Non-vacuity for the parametrized membership test.

        An empty `BUILDER_REQUIRED_PACKAGES` would make it collect no cases at
        all and the class would look green over nothing. The count is pinned
        loosely at the vendor-derived eleven so that silently emptying the
        constant fails here.
        """
        assert len(BUILDER_REQUIRED_PACKAGES) >= 13, (
            sorted(BUILDER_REQUIRED_PACKAGES))
        assert BUILDER_REQUIRED_PACKAGES <= builder_packages, sorted(
            BUILDER_REQUIRED_PACKAGES - builder_packages)

    def test_the_stage_package_lists_actually_differ(self, dockerfile_stages):
        """Non-vacuity for the test above: if every stage parsed to the same
        set, 'not in runtime' and 'in builder' could not both hold, and a
        bounding bug would show up here rather than as a confusing failure."""
        builder = _apt_packages(dockerfile_stages["builder"])
        runtime = _apt_packages(dockerfile_stages["runtime"])
        assert builder != runtime, (
            "builder and runtime parsed to the same package set; the stage "
            "split is not separating them")
        assert runtime, "the runtime stage parsed to no packages at all"


# ---------------------------------------------------------------------------
# The mirror, in both directions
# ---------------------------------------------------------------------------

class TestTheTwoFlagListsMirrorEachOther:
    """The guard that did not exist, and the reason the lists could drift.

    `docker/percona/Dockerfile`'s own comment says "Flags mirror
    config/engines/percona.yml", and `config/engines/percona.yml` says "Both
    copies carry the same values". Two files asserting agreement with each
    other and nothing checking it is the standing shape of this repo's silent
    bugs.
    """

    def test_every_yaml_flag_is_passed_by_the_dockerfile(self, yaml_flags,
                                                         dockerfile_flags):
        missing = sorted(set(yaml_flags) - set(dockerfile_flags))
        assert not missing, (
            "declared in config/engines/percona.yml but not passed by "
            f"docker/percona/Dockerfile: {missing}. A by-hand `docker build` "
            "would compile a different server from `./run-benchmark.sh build`.")

    def test_every_dockerfile_policy_flag_is_declared_in_the_yaml(
            self, yaml_flags, dockerfile_flags):
        yaml_symbols = {s for s, _ in map(_split, yaml_flags)}
        missing = sorted(
            flag for flag in dockerfile_flags
            if _split(flag)[0] not in yaml_symbols
            and _split(flag)[0] not in DOCKERFILE_ONLY_SYMBOLS)
        assert not missing, (
            "passed by docker/percona/Dockerfile, absent from "
            f"config/engines/percona.yml, and not allowlisted: {missing}. "
            "Either declare it in the YAML or add it to "
            "DOCKERFILE_ONLY_SYMBOLS with a reason.")

    def test_the_allowlist_is_exactly_the_dockerfile_only_set(
            self, yaml_flags, dockerfile_flags):
        """Equality, not containment, in both directions at once.

        Containment would let a sixth Dockerfile-only flag through if the
        allowlist happened to be a superset, and would never notice an entry
        that went stale. This is the assertion that makes the exclusion a
        decision with a shelf life rather than a blanket.
        """
        yaml_symbols = {s for s, _ in map(_split, yaml_flags)}
        docker_symbols = {s for s, _ in map(_split, dockerfile_flags)}
        unmirrored = docker_symbols - yaml_symbols
        assert unmirrored == set(DOCKERFILE_ONLY_SYMBOLS), (
            "the Dockerfile-only flag set moved.\n"
            f"  newly unmirrored (add to the YAML, or allowlist with a "
            f"reason): {sorted(unmirrored - DOCKERFILE_ONLY_SYMBOLS)}\n"
            f"  stale allowlist entries (no longer passed, or now mirrored): "
            f"{sorted(DOCKERFILE_ONLY_SYMBOLS - unmirrored)}")

    def test_the_shared_symbols_carry_identical_values(self, yaml_flags,
                                                       dockerfile_flags):
        """Agreeing on which flags exist is half of mirroring.

        `=OFF` against `=ON`, or `=0` against `=system`, for the same symbol
        passes every membership test above on the symbol alone; and because
        the Dockerfile appends `${CMAKE_FLAGS}` last the YAML's value is the
        one that would win -- so the two files would document opposite
        policies and the build would quietly follow the YAML.
        """
        yaml_pairs = dict(map(_split, yaml_flags))
        docker_pairs = dict(map(_split, dockerfile_flags))
        shared = sorted(set(yaml_pairs) & set(docker_pairs))
        assert shared, "the two lists share no symbol at all"
        disagree = {
            symbol: {"yaml": yaml_pairs[symbol],
                     "dockerfile": docker_pairs[symbol]}
            for symbol in shared
            if yaml_pairs[symbol] != docker_pairs[symbol]
        }
        assert not disagree, f"same flag, different value: {disagree}"

    def test_the_dockerfile_still_appends_the_yaml_flags_last(
            self, cmake_region):
        """The premise the two tests above rest on.

        If `${CMAKE_FLAGS}` were dropped, `build.cmake_flags` would stop
        reaching cmake entirely and every assertion in this file would be
        checking a list that no longer influences the build -- green, and
        measuring nothing.
        """
        assert "${CMAKE_FLAGS}" in cmake_region, (
            "the builder's cmake invocation no longer expands ${CMAKE_FLAGS}, "
            "so config/engines/percona.yml build.cmake_flags is dead config")


# ---------------------------------------------------------------------------
# The debugger, the stage it belongs in, and the audit entry it reverses
# ---------------------------------------------------------------------------
#
# 2026-10-27, request 1 of docs/12-percona-feedback-plan.md -- its §4, "get
# stack traces for the crashes". NOT request 3, which is filtered search
# against pgvector (§7) and another unit's subject; an earlier revision of
# this section said 3 at every site.
#
# Twelve of the twenty-four Percona ops units in run full-20261015-215006
# died inside the first INSERT -- exactly half. `config/profiles/full.yml` is
# four datasets and `ops.m_values: [8, 16, 32]`, so twelve units a pass and
# twenty-four across both; the twelve that died is the plan's count over the
# archived unit files, which `results/` being gitignored puts outside this
# checkout. The figure is NOT taken from `config/engines/percona.yml`'s
# `error_log` note, which still reads "13 of 24" -- wrong, and corrected by
# the crash-capture change.
#
# Nothing in the tree could turn such a crash into text: `gdb` was installed
# by no stage of any image here, and that was a RECORDED decision -- the
# audit note above the builder's apt list lists it beside
# valgrind, doxygen and the rest of Percona's packaging and test pipeline as
# "deliberately NOT taken, with the reason, so the audit does not have to be
# redone".
#
# So this is a reversal of a written decision and not merely a package, and
# the two halves are asserted separately:
#
#   * the package is in the BENCH stage and in no other. The runtime image is
#     the container the ops path measures -- `OpsRun._start_server` starts it
#     from `image.runtime` -- and a debugger in it is a package in the
#     measured server's own filesystem. The bench image is what
#     `OpsRun.run_script` starts for a lab script, which is the route that
#     opens a core.
#   * the audit note stops LISTING it and starts RECORDING the reversal. A
#     list that silently loses an entry is how an audit has to be redone,
#     which is the note's own stated reason for existing, so a deletion that
#     leaves no trace fails here just as a silent addition does.
#
# WHAT THESE ASSERTIONS CANNOT ESTABLISH, in the same spirit as the module
# docstring above: nothing here builds an image. Whether noble has a `gdb`
# candidate, whether the layer builds, and whether gdb can open a core
# written by this mysqld are all for the first rebuild. These are text guards
# over a Dockerfile, a script and a tuple.

#: The debugger the crash-capture route needs.
DEBUGGER_PACKAGE = "gdb"

#: The stage it belongs in, and the stages it must stay out of. `bench` is
#: `FROM runtime AS bench`, so a package in `runtime` is also in `bench` --
#: the containment is the reason the negative assertion is the load-bearing
#: one here rather than a formality.
DEBUGGER_STAGE = "bench"
DEBUGGER_FORBIDDEN_STAGES = ("runtime", "builder")

#: What a parsed package name may look like. Debian policy allows
#: lowercase alphanumerics plus `+ - .`, starting alphanumeric.
#:
#: THIS CONSTANT EXISTS BECAUSE THE PARSER READ A COMMENT. `_apt_packages`
#: matches `apt-get install` anywhere in a stage's text, prose included, and
#: this change's own note in the bench stage quoted that string while
#: explaining why gdb used to be absent. The bench stage then parsed to
#: `['3', 'Request', '`', 'any', 'build-essential', 'gdb', ...]` -- every
#: membership assertion in this file still passed, because junk tokens
#: collide with no package name anyone asserts. The note was reworded; this
#: is the guard that makes the next one fail loudly instead.
PACKAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9+.-]*$")

#: Runtime packages the runtime stage really does install, asserted beside
#: every "gdb is not in runtime" claim. A negative assertion passes just as
#: happily against an empty string, and `_apt_packages` returning nothing for
#: a restructured stage is exactly how that would happen.
RUNTIME_PACKAGES_IT_DOES_CARRY = ("procps", "libssl3t64")

#: The audit note, by its own words. The first is its anchor; the second and
#: third bound the enumeration of package names inside it, which is the part
#: `gdb` has to leave. The note's prose continues past the enumeration, and
#: the reversal sentence is added to that prose -- so the two regions are
#: parsed separately and a name cannot satisfy one by sitting in the other.
AUDIT_NOTE_ANCHOR = "Vendor packages deliberately NOT taken"
AUDIT_NOTE_END = "\nARG BASE_IMAGE"
AUDIT_ENUMERATION_OPEN = "does not have to be redone:"
AUDIT_ENUMERATION_CLOSE = "are their packaging and test pipeline"

#: An entry that must still be listed. Without it, "gdb is not in the
#: enumeration" is satisfied by deleting the paragraph.
AUDIT_ENTRY_THAT_STAYS = "valgrind"

#: The marker the reversal sentence must carry, and the request it must name.
#:
#: BOTH EXIST BECAUSE TOKEN CO-OCCURRENCE HAS NO POLARITY, and that was a
#: review finding against the first version of this file rather than a
#: hypothetical. `"gdb" in note` plus `"bench" in <the sentence naming gdb>`
#: is satisfied by a note reading "gdb is not taken in the bench stage
#: either, for the same reason" -- the OPPOSITE of what the image does -- and
#: by one reading literally `gdb bench.`. A marker is what distinguishes a
#: record of a reversal from any other sentence that happens to mention the
#: package and the stage, and the request number is what sends the next
#: auditor to the right section of the plan.
#:
#: Pinning prose in a test is deliberate here: the audit note's whole purpose
#: is that the audit not have to be redone, so what it SAYS is the artefact
#: under test, exactly as `tests/test_e8_package_flags_overwrite.py` holds
#: both Dockerfiles' notes to naming the flags they explain.
AUDIT_REVERSAL_MARKER = "ONE REVERSAL"
AUDIT_REQUEST = "request 1"

#: The reversal sentence exactly as the Dockerfile spells it, used ONLY as a
#: mutation needle. The PROPERTIES that matter are asserted above and apply
#: to any wording; this string is how the probes below reach the sentence in
#: raw text, where it is one line. A reword should update this and leave the
#: property assertions alone -- and is told to, because a probe whose needle
#: stopped matching would silently stop mutating anything, which is why its
#: presence is asserted rather than assumed.
AUDIT_REVERSAL_NEEDLE = " ONE REVERSAL: gdb, for request 1; see the bench stage."

#: The lab script that runs the debugger, and the flag that mounts the
#: directory holding the core.
LAB_SCRIPT = "gdb-backtrace.py"
LAB_ARTIFACTS_OPTION = "--artifacts"

#: The gdb command the plan asks for, and the name of the constant that has
#: to hold it. Pinned as a value because the value is the claim: `bt` alone
#: gives one thread, and the crashing thread of a multi-threaded server is
#: not reliably the one gdb stops in.
#:
#: READ OUT OF THE AST, NOT GREPPED, and this is vacuity shape 3 from the
#: module docstring caught live rather than argued. The script documents the
#: command it runs at length, so `"thread apply all bt full" in source` is
#: true no matter what the constant says -- the mutation run below weakened
#: the constant to `bt` and the grepped form of this assertion SURVIVED.
GDB_COMMAND_CONSTANT = "GDB_COMMAND"
GDB_BATCH_COMMAND = "thread apply all bt full"

#: The two code fragments that say the constant is actually what gdb is
#: handed. Spelled as code so that prose cannot satisfy them: a script that
#: defines the constant correctly and then passes a literal would otherwise
#: pass everything above.
GDB_INVOCATION_FRAGMENTS = ('[gdb, "--batch", "--nx",', '"-ex", GDB_COMMAND,')


def _audit_note(dockerfile_text: str) -> str:
    """The 'deliberately NOT taken' paragraph, flattened to one line.

    Comment markers and line wrapping are removed so that an assertion about
    a sentence is not also an assertion about where the 77th column fell.
    """
    assert AUDIT_NOTE_ANCHOR in dockerfile_text, (
        f"{AUDIT_NOTE_ANCHOR!r} is not in the Dockerfile; this parser's "
        f"anchor is stale or the audit note is gone")
    after = dockerfile_text.split(AUDIT_NOTE_ANCHOR, 1)[1]
    assert AUDIT_NOTE_END in after, (
        f"no {AUDIT_NOTE_END!r} after the audit note to bound it on")
    note = AUDIT_NOTE_ANCHOR + after.split(AUDIT_NOTE_END, 1)[0]
    return " ".join(note.replace("#", " ").split())


def _audit_enumeration(dockerfile_text: str) -> frozenset[str]:
    """The package names the audit note enumerates as not taken.

    Whole tokens off the comma list, not a substring scan of the paragraph:
    the reversal sentence names `gdb` on purpose, and a substring check would
    read that as the entry still being listed -- which is the one way this
    guard could pass while the edit it asks for had not happened.
    """
    note = _audit_note(dockerfile_text)
    assert AUDIT_ENUMERATION_OPEN in note, AUDIT_ENUMERATION_OPEN
    assert AUDIT_ENUMERATION_CLOSE in note, AUDIT_ENUMERATION_CLOSE
    listed = note.split(AUDIT_ENUMERATION_OPEN, 1)[1]
    listed = listed.split(AUDIT_ENUMERATION_CLOSE, 1)[0]
    names = set()
    for part in listed.replace(" and ", ", ").split(","):
        name = part.strip(" `.;:")
        if name:
            names.add(name)
    return frozenset(names)


def _script_constant(source: str, name: str):
    """A module-level constant of a lab script, read out of its AST.

    The parse exists because the grepped form of the assertion it serves was
    a mutation SURVIVOR: `scripts/gdb-backtrace.py` explains the gdb command
    it runs in its docstring, so a substring check for that command passes
    against a script whose constant has been changed to something else.
    """
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    return None


def _audit_sentences(dockerfile_text: str) -> list:
    r"""The audit note, split into sentences.

    `(?<=\.)\s+` and not `split(".")`: the paragraph cites `CMakeLists.txt`
    and `build-binary.sh`, and the dot in a filename is followed by no space.
    """
    return re.split(r"(?<=\.)\s+", _audit_note(dockerfile_text))


def _audit_reversal_sentence(dockerfile_text: str) -> str:
    """The sentence recording the reversal, or '' if it is not unique.

    SELECTED BY THE MARKER, NOT BY THE PACKAGE NAME. The first version of
    this helper returned the first sentence containing `gdb`, which has two
    faults: a legitimate earlier mention of the package would be read as the
    record and fail a correct file, and any sentence naming the package and
    the stage would satisfy the assertions whatever it claimed about them.
    Returning '' for zero matches AND for more than one is deliberate: two
    reversal sentences mean the caller cannot know which one it checked.
    """
    found = [s for s in _audit_sentences(dockerfile_text)
             if AUDIT_REVERSAL_MARKER in s]
    return found[0] if len(found) == 1 else ""


@pytest.fixture(scope="module")
def lab_script_source() -> str:
    path = os.path.join(VB_ROOT, "scripts", LAB_SCRIPT)
    with open(path) as handle:
        return handle.read()


class TestTheDebuggerIsInTheStageThatDiagnoses:
    """`gdb`, in `bench` and nowhere else."""

    def test_the_bench_stage_installs_it(self, dockerfile_stages):
        packages = _apt_packages(dockerfile_stages[DEBUGGER_STAGE])
        assert packages, (
            "the bench stage parsed to no packages at all, so the assertion "
            "below would be about nothing")
        assert DEBUGGER_PACKAGE in packages, (
            f"{DEBUGGER_PACKAGE} is not in the {DEBUGGER_STAGE} stage's "
            f"apt-get install list, so `lab ... {LAB_SCRIPT}` has no "
            f"debugger to run. {DEBUGGER_STAGE} installs: {sorted(packages)}")

    @pytest.mark.parametrize("stage", DEBUGGER_FORBIDDEN_STAGES)
    def test_no_other_stage_installs_it(self, stage, dockerfile_stages):
        """The runtime stage is the measured server's filesystem.

        Paired with a positive assertion about the same list, because an
        absence check is satisfied by a parse that returned nothing -- and a
        restructured stage is precisely how that happens.
        """
        packages = _apt_packages(dockerfile_stages[stage])
        assert len(packages) >= 5, (
            f"the {stage} stage parsed to {sorted(packages)}, which is too "
            f"few for the absence below to mean anything")
        assert DEBUGGER_PACKAGE not in packages, (
            f"{DEBUGGER_PACKAGE} is installed in the {stage} stage. The "
            f"runtime image is the container the ops path measures, and a "
            f"debugger belongs in the one a lab script runs in. "
            f"{stage} installs: {sorted(packages)}")

    def test_the_runtime_list_really_does_carry_packages_it_claims(
            self, dockerfile_stages):
        """Non-vacuity for the test above, by name rather than by count."""
        packages = _apt_packages(dockerfile_stages["runtime"])
        for expected in RUNTIME_PACKAGES_IT_DOES_CARRY:
            assert expected in packages, (
                f"{expected} is no longer in the runtime stage, so this "
                f"file's model of that stage is stale: {sorted(packages)}")

    def test_no_stage_parses_prose_as_packages(self, dockerfile_stages):
        """Every stage's parsed list is package names and nothing else.

        `_apt_packages` is not bounded to a RUN -- it matches the apt install
        step wherever it appears, and this Dockerfile is mostly comments. A
        note that quotes the string turns the following words into
        "packages": harmless for membership assertions, and not harmless for
        a count or for anyone reading a failure message. This is the
        assertion that names the cause.

        Every stage the fixture found, not a hardcoded three: a fourth stage
        would otherwise be added and never checked, which is the same shape
        as the flag added and forgotten that `_LAB_OPTIONS` guards against.
        """
        offenders = {}
        for stage, text in dockerfile_stages.items():
            junk = sorted(p for p in _apt_packages(text)
                          if not PACKAGE_NAME.match(p))
            if junk:
                offenders[stage] = junk
        assert not offenders, (
            f"these stages parsed prose as package names: {offenders}. A "
            f"comment in docker/percona/Dockerfile most likely quotes the "
            f"literal apt install step; the parser reads prose, so reword "
            f"the comment rather than widening this check.")

    def test_it_is_installed_in_exactly_one_stage(self, dockerfile_stages):
        """`bench` is `FROM runtime AS bench`, so an entry copied into
        runtime rather than moved would leave the bench assertion green and
        put the debugger in the measured server after all."""
        carrying = sorted(name for name, text in dockerfile_stages.items()
                          if DEBUGGER_PACKAGE in _apt_packages(text))
        assert carrying == [DEBUGGER_STAGE], carrying


class TestTheAuditNoteRecordsTheReversalRatherThanLosingIt:
    """The half that is about the decision and not about the package.

    The note's own reason for existing is "so the audit does not have to be
    redone". Dropping an entry from it silently is how it has to be redone.
    """

    def test_the_enumeration_parsed_to_something(self, dockerfile_text):
        """Non-vacuity: every assertion below is about this set."""
        listed = _audit_enumeration(dockerfile_text)
        assert len(listed) >= 10, sorted(listed)
        for name in ("doxygen", "patchelf", "copyright-update"):
            assert name in listed, sorted(listed)

    def test_the_enumeration_no_longer_lists_the_debugger(self,
                                                          dockerfile_text):
        listed = _audit_enumeration(dockerfile_text)
        assert DEBUGGER_PACKAGE not in listed, (
            f"the audit note still lists {DEBUGGER_PACKAGE} among the "
            f"packages deliberately not taken, while the {DEBUGGER_STAGE} "
            f"stage installs it. The note and the image disagree: "
            f"{sorted(listed)}")

    def test_the_paragraph_was_not_deleted_to_satisfy_that(self,
                                                           dockerfile_text):
        listed = _audit_enumeration(dockerfile_text)
        assert AUDIT_ENTRY_THAT_STAYS in listed, (
            f"{AUDIT_ENTRY_THAT_STAYS} has left the audit note too, so the "
            f"edit removed the paragraph rather than one entry: "
            f"{sorted(listed)}")

    def test_the_note_still_records_the_entry_it_reversed(self,
                                                          dockerfile_text):
        """The entry leaves the list and stays in the record.

        This is the assertion a silent deletion fails. The mutant is not
        hypothetical: removing the name is the smallest edit that satisfies
        the test above.
        """
        note = _audit_note(dockerfile_text)
        assert DEBUGGER_PACKAGE in note, (
            f"{DEBUGGER_PACKAGE} has vanished from the audit note entirely. "
            f"It was a recorded decision; a reader auditing this image "
            f"against the vendor's install_deps() would now have to redo the "
            f"work the note exists to save.")

    def test_the_record_is_marked_as_a_reversal(self, dockerfile_text):
        """Exactly one sentence, and it says that it is one.

        Without the marker the guards below are two token co-occurrences
        with no polarity, and a sentence asserting the opposite of what the
        image does satisfies them.
        """
        sentence = _audit_reversal_sentence(dockerfile_text)
        assert sentence, (
            f"the audit note carries no single sentence marked "
            f"{AUDIT_REVERSAL_MARKER!r}. Sentences naming the debugger: "
            f"{[s for s in _audit_sentences(dockerfile_text) if DEBUGGER_PACKAGE in s]}")

    def test_the_record_says_what_it_reversed_and_where_it_went(
            self, dockerfile_text):
        """The marked sentence has to name both, or it records nothing."""
        sentence = _audit_reversal_sentence(dockerfile_text)
        assert sentence, "no marked reversal sentence; see the test above"
        assert DEBUGGER_PACKAGE in sentence, (
            f"the reversal sentence does not name what was reversed: "
            f"{sentence!r}")
        assert DEBUGGER_STAGE in sentence, (
            f"the reversal sentence does not say where {DEBUGGER_PACKAGE} "
            f"now is. It reads: {sentence!r}")

    def test_the_record_names_the_request_it_serves(self, dockerfile_text):
        """The pointer an auditor follows.

        Every other refusal in that paragraph carries a citation; a reversal
        that carries only a purpose leaves the reader to find the reason
        somewhere in a 700-line file. The number is checked because getting
        it wrong is worse than omitting it: an earlier revision said
        "request 3", which is filtered search against pgvector.
        """
        sentence = _audit_reversal_sentence(dockerfile_text)
        assert sentence, "no marked reversal sentence; see the test above"
        assert AUDIT_REQUEST in sentence, (
            f"the reversal sentence does not name {AUDIT_REQUEST!r}, so the "
            f"next auditor has no section to go to. It reads: {sentence!r}")


class TestTheLabRouteThatOpensTheCore:
    """The script, and the flag that lets it see a saved core.

    `OpsRun.run_script` mounts no writable path and deliberately so -- a lab
    script's invented numbers must not be able to reach a report. The core,
    however, is in `results/<run-id>/`, which a lab session cannot name, so
    the directory has to be handed in explicitly and read-only.
    """

    def test_the_script_exists_and_is_the_file_it_names(self):
        path = os.path.join(VB_ROOT, "scripts", LAB_SCRIPT)
        assert os.path.isfile(path), path
        assert not os.path.islink(path), (
            f"{path} is a symlink; `lab` runs files in scripts/ so that the "
            f"name in the command is the code that runs")

    def test_the_script_parses_and_has_a_summary_line(self, lab_script_source):
        """`lab --list` prints the first line of the docstring, and
        `lab_scripts()` swallows a SyntaxError into an empty summary -- so a
        script that does not parse is listed with a blank description rather
        than failing."""
        tree = ast.parse(lab_script_source)
        summary = (ast.get_docstring(tree) or "").split("\n")[0].strip()
        assert summary, (
            f"scripts/{LAB_SCRIPT} has no one-line docstring summary, so "
            f"`lab --list` would print its name against nothing")

    def test_the_lab_would_actually_run_it(self):
        """The check `cmd_lab` makes before it starts anything."""
        from orchestrator.cli import _lab_script_problem, lab_scripts
        assert _lab_script_problem(LAB_SCRIPT) is None, (
            _lab_script_problem(LAB_SCRIPT))
        assert LAB_SCRIPT in dict(lab_scripts()), sorted(dict(lab_scripts()))

    def test_it_signs_the_contract_every_lab_script_signs(self,
                                                          lab_script_source):
        """`run_script` passes --host and --port to every script. A script
        that does not accept them dies on argparse after the server is up."""
        assert '"--host"' in lab_script_source
        assert '"--port"' in lab_script_source

    def test_it_runs_the_command_the_plan_asks_for(self, lab_script_source):
        """The value is the claim. `bt` alone reports the thread gdb happens
        to stop in, which for a crashed multi-threaded server is not
        reliably the one that crashed -- InnoDB alone contributes a dozen
        background threads.

        Read out of the AST. The grepped version of this assertion survived
        the mutant that weakens the constant to `bt`, because the script's
        own docstring quotes the full command while explaining it.
        """
        value = _script_constant(lab_script_source, GDB_COMMAND_CONSTANT)
        assert value == GDB_BATCH_COMMAND, (
            f"scripts/{LAB_SCRIPT}'s {GDB_COMMAND_CONSTANT} is {value!r}, "
            f"not {GDB_BATCH_COMMAND!r}")

    @pytest.mark.parametrize("fragment", GDB_INVOCATION_FRAGMENTS)
    def test_the_constant_is_what_gdb_is_actually_handed(
            self, fragment, lab_script_source):
        """A correct constant the invocation does not use is the other half
        of the same mutant."""
        assert fragment in lab_script_source, (
            f"scripts/{LAB_SCRIPT} no longer contains {fragment!r}, so "
            f"{GDB_COMMAND_CONSTANT} may not be what gdb is given")

    def test_lab_has_the_flag_that_mounts_the_artefacts(self):
        """`_LAB_OPTIONS` is lab's own list of its flags, and a test in
        tests/test_config.py asserts it matches the subparser exactly -- so
        the flag cannot be added here and forgotten there, or vice versa."""
        from orchestrator.cli import _LAB_OPTIONS
        assert LAB_ARTIFACTS_OPTION in _LAB_OPTIONS, sorted(_LAB_OPTIONS)

    def test_the_compatibility_refusal_is_gone_and_stays_gone(self):
        """THE RATCHET FIRED, AND THIS IS WHAT IS LEFT OF IT.

        It was `test_the_compatibility_refusal_collects_itself`, and it
        existed because `cli._artifacts_mount_problem()` was load-bearing
        only while `OpsRun.run_script` had no `artifacts_dir` parameter: it
        turned a TypeError raised minutes after a server came up into an
        immediate refusal. The change that adds the parameter did not list
        `orchestrator/cli.py` in its file set, so nothing in its path would
        have prompted the cleanup and the function would have sat there
        returning None forever. The ratchet's job was to fail on the day that
        happened and hand over a work order.

        It did exactly that. #68 added `artifacts_dir` and this test failed
        with its three items; the file set was widened by `orchestrator/cli.py`
        and this file to discharge them inside the same change, so `master`
        never carried the dead code or the red. All three are done: the
        function is deleted, `cmd_lab`'s refusal loop is one check instead of
        two, and the forward `supported` branch that was waiting for this
        moment is gone -- with it `inspect`, which `cli.py` imported for that
        one call and nothing else.

        WHAT SURVIVES IS THE NO-RETURN HALF, deliberately. A ratchet that
        deleted itself entirely would leave nothing saying the shim must not
        come back, and a reader who meets `--artifacts` failing on an old
        checkout is exactly the person who would reintroduce it. This
        assertion costs one line and is the cheaper half of the pair.
        """
        import inspect

        from orchestrator import cli, ops_pass

        # The precondition the deletion rests on. Asserted rather than
        # assumed, so this cannot pass by the parameter having been removed
        # again -- which would make the shim load-bearing a second time.
        assert "artifacts_dir" in inspect.signature(
            ops_pass.OpsRun.run_script).parameters, (
            "OpsRun.run_script has lost its `artifacts_dir` parameter. The "
            "refusal this test retired was what turned that into an "
            "immediate error instead of a TypeError raised after the server "
            "was up; if the parameter is really going, the refusal and its "
            "original ratchet have to come back with it")
        assert not hasattr(cli, "_artifacts_mount_problem"), (
            "cli._artifacts_mount_problem() is back. It can only ever return "
            "None now that `artifacts_dir` is unconditionally present, so it "
            "is dead code and the refusal in cmd_lab would be a no-op")


# ---------------------------------------------------------------------------
# Mutation probes for the four assertions above
# ---------------------------------------------------------------------------
#
# Each corrupts the text, asserts the corruption applied, and reports the
# honest reading beside the mutant one. The two package probes matter most:
# both halves of "in bench, not in runtime" are set-membership claims over a
# parse, and a parse that silently returned nothing would make the negative
# half green forever.

class TestTheseProbesDiscriminate:

    def test_moving_the_debugger_to_the_runtime_stage_is_seen(
            self, dockerfile_stages):
        """The mutant that a whole-file grep cannot see, and the one that
        would put a debugger in the measured server's filesystem.

        The slice comes from the fixture and the parse from `_apt_packages`,
        so the probe exercises the two the assertions use rather than a
        simpler copy of them.
        """
        honest = dockerfile_stages["runtime"]
        mutant = honest.replace(
            "      ca-certificates procps tzdata \\",
            "      ca-certificates procps tzdata gdb \\")
        assert mutant != honest, "mutation did not apply"
        assert DEBUGGER_PACKAGE not in _apt_packages(honest)
        assert DEBUGGER_PACKAGE in _apt_packages(mutant)

    def test_adding_the_package_without_editing_the_note_is_seen(
            self, dockerfile_text):
        """The mutant the audit assertions exist for: install gdb, leave the
        note claiming it was deliberately not taken."""
        mutant = dockerfile_text.replace(AUDIT_REVERSAL_NEEDLE, "")
        mutant = mutant.replace("redone: valgrind, doxygen,",
                                "redone: valgrind, gdb, doxygen,")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert DEBUGGER_PACKAGE not in _audit_enumeration(dockerfile_text)
        assert DEBUGGER_PACKAGE in _audit_enumeration(mutant)

    def test_deleting_the_entry_silently_is_seen(self, dockerfile_text):
        """The other direction, and the one an edit is likelier to take: drop
        the name from the list, record nothing."""
        mutant = dockerfile_text.replace(AUDIT_REVERSAL_NEEDLE, "")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert DEBUGGER_PACKAGE in _audit_note(dockerfile_text)
        assert DEBUGGER_PACKAGE not in _audit_note(mutant)

    def test_the_probe_needle_is_in_the_file(self, dockerfile_text):
        """Non-vacuity for every probe below that mutates the note.

        `str.replace` of a string that is not there returns the original,
        and each probe asserts `mutant != honest` for that reason -- but
        this says so once, with the needle named, so a reword produces one
        clear failure here instead of three confusing ones.
        """
        assert AUDIT_REVERSAL_NEEDLE in dockerfile_text, (
            f"{AUDIT_REVERSAL_NEEDLE!r} is no longer in the Dockerfile. If "
            f"the reversal sentence was reworded, update this constant; the "
            f"property assertions above do not need changing.")

    def test_a_note_asserting_the_opposite_is_seen(self, dockerfile_text):
        """The review finding, kept as an assertion.

        Token co-occurrence has no polarity: a sentence naming the package
        and the stage while claiming the package is NOT taken satisfies
        `"gdb" in note` and `"bench" in <sentence naming gdb>`. The marker
        is what tells the two apart, and this proves it does.
        """
        mutant = dockerfile_text.replace(
            AUDIT_REVERSAL_NEEDLE,
            " gdb is not taken in the bench stage either, for the same "
            "reason.")
        assert mutant != dockerfile_text, "mutation did not apply"
        # Both of the weak conditions still hold on the mutant...
        assert DEBUGGER_PACKAGE in _audit_note(mutant)
        assert any(DEBUGGER_PACKAGE in s and DEBUGGER_STAGE in s
                   for s in _audit_sentences(mutant))
        # ...and the marked-sentence selector finds nothing, which is what
        # every assertion in TestTheAuditNote... now rests on.
        assert _audit_reversal_sentence(dockerfile_text)
        assert _audit_reversal_sentence(mutant) == ""

    def test_a_contentless_note_is_seen(self, dockerfile_text):
        """The degenerate case of the same finding: the two tokens alone."""
        mutant = dockerfile_text.replace(AUDIT_REVERSAL_NEEDLE,
                                         " gdb bench.")
        assert mutant != dockerfile_text, "mutation did not apply"
        assert _audit_reversal_sentence(mutant) == ""

    def test_the_wrong_request_number_is_seen(self, dockerfile_text):
        """The blocking defect this file shipped once: request 3 is filtered
        search against pgvector, not the stack traces."""
        mutant = dockerfile_text.replace(AUDIT_REVERSAL_NEEDLE,
                                         AUDIT_REVERSAL_NEEDLE.replace(
                                             AUDIT_REQUEST, "request 3"))
        assert mutant != dockerfile_text, "mutation did not apply"
        sentence = _audit_reversal_sentence(mutant)
        assert sentence, "the mutant lost the marker as well; wrong mutant"
        assert AUDIT_REQUEST not in sentence
        assert AUDIT_REQUEST in _audit_reversal_sentence(dockerfile_text)

    def test_an_empty_stage_parse_would_not_pass_as_an_absence(
            self, dockerfile_stages):
        """What the paired positive assertions are for, stated directly."""
        assert DEBUGGER_PACKAGE not in _apt_packages("")
        assert _apt_packages("") == frozenset()
        assert _apt_packages(dockerfile_stages["runtime"]) != frozenset()

    def test_grepping_for_the_gdb_command_cannot_see_it_change(
            self, lab_script_source):
        """The survivor this file was corrected for, kept as an assertion.

        Both readings are reported: the grep is true of the mutant and the
        AST is not, which is the whole difference between the two forms.
        """
        mutant = lab_script_source.replace(
            f'{GDB_COMMAND_CONSTANT} = "{GDB_BATCH_COMMAND}"',
            f'{GDB_COMMAND_CONSTANT} = "bt"')
        assert mutant != lab_script_source, "mutation did not apply"
        assert GDB_BATCH_COMMAND in mutant, (
            "the grepped assertion would have failed on this mutant after "
            "all, so this probe no longer records anything")
        assert _script_constant(lab_script_source,
                                GDB_COMMAND_CONSTANT) == GDB_BATCH_COMMAND
        assert _script_constant(mutant, GDB_COMMAND_CONSTANT) == "bt"

    def test_a_substring_scan_of_the_note_cannot_answer_the_question(
            self, dockerfile_text):
        """Why the enumeration is parsed into tokens rather than grepped.

        After the edit the paragraph names `gdb` deliberately -- that is the
        record of the reversal -- so `"gdb" in note` is TRUE while the
        enumeration does not list it. A guard written as a substring check
        would therefore report the edit as not having happened.
        """
        note = _audit_note(dockerfile_text)
        assert DEBUGGER_PACKAGE in note
        assert DEBUGGER_PACKAGE not in _audit_enumeration(dockerfile_text)
