"""Why the Percona builder names a _FORTIFY_SOURCE level of its own.

WHY THIS FILE EXISTS
--------------------
2026-10-15. Three builds of the Percona image died with
`make: *** [Makefile:166: all] Error 2` and nothing in the tail naming a
file. The serial retry added in the previous commit put the real error last:

    extra/libfido2/libfido2-1.15.0/src/.../aes256.c.o
    <command-line>: error: "_FORTIFY_SOURCE" redefined
    <command-line>: note: this is the location of the previous definition

Two definitions with different values were reaching the preprocessor. The
tree asks for 2 (`CMakeLists.txt:1586` at the pin prepends
`-DNDEBUG -D_FORTIFY_SOURCE=2` for the non-debug build types), and the 3 came
from `dpkg-buildflags --get CPPFLAGS` by way of
`cmake/compile_flags.cmake:130`'s `ADD_LINUX_DEB_FLAGS()` -- a function that
also OVERWRITES CMAKE_C_FLAGS, discarding every flag the builder exported.
`tests/test_e8_package_flags_overwrite.py` owns that chain and the two cmake
flags that close it; the corrected account is in docker/percona/Dockerfile
above its cmake line.

That matters for how this file should be read. WHEN THIS FILE WAS WRITTEN THE
EMITTING SITE HAD NOT BEEN FOUND and the mechanism recorded here was the wrong
one: it supposed the tree's own 2 arrived as a `-Wp,-D...` the gcc driver's
scan could not see, so that the driver added its own default 3 beside it. No
such `-Wp,` define exists in this tree and the driver's default is not
involved -- both defines were ordinary `-D`s and the disagreeing one was
dpkg's. The flag this file guards therefore did NOT fix the build on its own,
because the export carrying it was thrown away too.

WHAT THIS FILE IS FOR
---------------------
The flag has an exact value and is not free to drift, but what it buys is much
narrower than this file first claimed, and the difference is worth stating
precisely.

What it buys is AGREEMENT. The level it names has to match the level the tree
prepends for its non-debug build types. A reader "modernising" 2 to 3 would
recreate the redefinition against the tree's 2; a reader dropping the value
entirely would get 1, which conflicts the same way.

What it does NOT buy is the level the project compiles at. With
`-DWITH_PACKAGE_FLAGS=OFF` now on the cmake line, deleting this token would
leave the tree's own `-D_FORTIFY_SOURCE=2` at `CMakeLists.txt:1586` stating the
level for every object in the cmake project; only sub-builds that read
`$CFLAGS` out of the ENVIRONMENT rather than from cmake would fall back to a
toolchain or distro default. So the token is belt-and-braces plus coverage of
those sub-builds -- not the thing that decides the project's hardening level.

AND THIS IMAGE WAS NOT COMPILING AT 3. An earlier version of this paragraph
said it was, and built an engine-to-engine codegen argument on top of that;
both are removed rather than softened. Pre-fix the compile line ended with the
tree's `-D_FORTIFY_SOURCE=2` from `CMAKE_C_FLAGS_RELWITHDEBINFO`, AFTER dpkg's
`=3` in `CMAKE_C_FLAGS`, and the last `-D` of a macro wins -- measured:
`gcc -O2 -D_FORTIFY_SOURCE=3 -D_FORTIFY_SOURCE=2 -dM -E -` prints
`#define _FORTIFY_SOURCE 2`, and the reverse order prints 3. The effective
level was 2, which AGREES with both MariaDB pins
(`mariadb-11.8.8:CMakeLists.txt:290`, `mariadb-12.3.4:CMakeLists.txt:303`,
read at both pins), while a redefinition warning was printed in every
translation unit and the build died in libfido2. The real engine-to-engine
divergence was `-march`, and
`tests/test_e8_package_flags_overwrite.py` is where that argument lives.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCONA_DOCKERFILE = os.path.join(VB_ROOT, "docker", "percona", "Dockerfile")

# The level the pinned tree itself asks for. Spelled out rather than derived,
# so that a tree that changes its mind fails this file with an explanation
# instead of silently re-deciding the hardening level of a published number.
TREE_FORTIFY_LEVEL = "2"

# A SYNTHETIC OBSERVER, NOT A CITATION, and not the mechanism of the observed
# failure either -- that one is dpkg-buildflags' plain `-D_FORTIFY_SOURCE=3`
# via `cmake/compile_flags.cmake:130`, which
# tests/test_e8_package_flags_overwrite.py measures.
#
# What this define is for is the probe below. The gcc driver's scan for a
# user-supplied level does not count a `-Wp,` define, so with this on the
# command line the driver still injects its own default -- which makes that
# injected default VISIBLE as a redefinition the moment our explicit `-D`
# stops suppressing it. It is how the probe tells a load-bearing flag from a
# decorative one on a host whose compiler defaults to 3. No `-Wp,` define of
# this macro exists in the pinned tree; the value 2 is chosen to match the
# tree's so that the honest side of the pair compiles clean.
UNSEEN_DEFINE_PROBE = "-Wp,-D_FORTIFY_SOURCE=2"

# libfido2's dialect and strictness at the pin, which is what makes the
# warning fatal there and nowhere else:
# extra/libfido2/libfido2-1.15.0/CMakeLists.txt:329-330 adds -pedantic and
# -pedantic-errors; lines 88-104 build the dialect flags.
LIBFIDO2_FLAGS = [
    "-std=c99", "-pedantic", "-pedantic-errors",
    "-D_POSIX_C_SOURCE=200809L", "-D_BSD_SOURCE", "-D_GNU_SOURCE",
    "-D_DEFAULT_SOURCE",
]


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    with open(PERCONA_DOCKERFILE, encoding="utf-8") as handle:
        return handle.read()


def build_commands(text: str) -> list[str]:
    """Non-comment lines only. The notes quote flags in prose at length."""
    return [line for line in text.splitlines()
            if not line.lstrip().startswith("#")]


def cflags_line(text: str) -> str:
    for line in build_commands(text):
        if line.strip().startswith("export CFLAGS="):
            return line
    raise AssertionError(
        "no `export CFLAGS=` line in the builder stage; this file's anchor "
        "is stale or the Dockerfile was restructured")


def cflags_value(text: str) -> str:
    line = cflags_line(text)
    match = re.search(r'export CFLAGS="([^"]*)"', line)
    assert match, f"could not read the CFLAGS value out of {line!r}"
    return match.group(1)


def expanded_cflags(text: str) -> list[str]:
    """CFLAGS as the shell would have it, with the ARG defaults applied."""
    value = cflags_value(text)
    args = {name: default.strip().strip('"').strip("'")
            for name, default in
            re.findall(r"^ARG (\w+)=(.*)$", text, re.MULTILINE)}
    value = value.replace("${MARCH}", args.get("MARCH", "x86-64-v3"))
    value = value.replace("${EXTRA_CXXFLAGS}", args.get("EXTRA_CXXFLAGS", ""))
    assert "${" not in value, (
        f"an unexpanded variable is left in CFLAGS: {value!r}. This file "
        f"compiles with these flags, so it has to know all of them.")
    return value.split()


# ---------------------------------------------------------------------------
# Non-vacuity. Every assertion below is about one token in one line, and a
# file that failed to parse would satisfy several of them by having nothing.
# ---------------------------------------------------------------------------

class TestTheFileWasActuallyRead:

    def test_the_dockerfile_exists(self):
        assert os.path.isfile(PERCONA_DOCKERFILE), PERCONA_DOCKERFILE

    def test_it_has_a_builder_cflags_line(self, dockerfile_text):
        assert cflags_line(dockerfile_text).strip()

    def test_the_cflags_carry_the_other_build_flags_too(self, dockerfile_text):
        """If this line stopped being the real CFLAGS, the rest is theatre."""
        value = cflags_value(dockerfile_text)
        for expected in ("-O3", "-march="):
            assert expected in value, (
                f"{expected} is not in the CFLAGS this file is guarding, so "
                f"it is reading the wrong line: {value!r}")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

class TestTheLevelIsNamedAndIsTheTreesLevel:

    def test_cflags_names_a_fortify_level(self, dockerfile_text):
        value = cflags_value(dockerfile_text)
        assert "-D_FORTIFY_SOURCE=" in value, (
            "the builder's CFLAGS no longer names a _FORTIFY_SOURCE level. "
            "The cmake project would still get the tree's own 2 from "
            "CMakeLists.txt:1586, but the sub-builds that read $CFLAGS out of "
            "the environment would fall back to a toolchain or distro "
            "default, and this image would stop stating a level it has a "
            "documented reason to state. CFLAGS: " + repr(value))

    def test_the_level_is_the_one_the_tree_asks_for(self, dockerfile_text):
        value = cflags_value(dockerfile_text)
        assert f"-D_FORTIFY_SOURCE={TREE_FORTIFY_LEVEL}" in value, (
            f"the level is not {TREE_FORTIFY_LEVEL}. It has to AGREE with "
            f"the level the pinned tree prepends for its non-debug build "
            f"types; a different number here is the redefinition conflict "
            f"again, not a stronger fix. CFLAGS: {value!r}")

    def test_no_second_level_contradicts_it(self, dockerfile_text):
        """One level in the build commands, not two."""
        carrying = [line for line in build_commands(dockerfile_text)
                    if "-D_FORTIFY_SOURCE" in line]
        assert len(carrying) == 1, (
            f"_FORTIFY_SOURCE is set in {len(carrying)} non-comment lines; "
            f"two levels in one build is the defect itself: {carrying}")

    def test_the_level_is_not_left_valueless(self, dockerfile_text):
        """`-D_FORTIFY_SOURCE` with no value means 1, and 1 against the
        tree's 2 is a redefinition exactly like 3 is."""
        value = cflags_value(dockerfile_text)
        assert not re.search(r"-D_FORTIFY_SOURCE(?![=\w])", value), (
            f"-D_FORTIFY_SOURCE is passed without a value: {value!r}")

    def test_cplusplus_inherits_it(self, dockerfile_text):
        """absl is C++ and showed the same redefinition, so CXXFLAGS has to
        carry the level too. It does by deriving from CFLAGS; if that ever
        becomes a separate literal, this file is not guarding C++ -- and
        `-DCMAKE_CXX_FLAGS="$CXXFLAGS"` on the cmake line would then pass the
        wrong string on."""
        commands = "\n".join(build_commands(dockerfile_text))
        assert 'export CXXFLAGS="$CFLAGS"' in commands, (
            "CXXFLAGS no longer derives from CFLAGS, so the fortify level "
            "this file guards may not reach the C++ half of the build")

    def test_the_level_is_in_cflags_and_not_a_cmake_argument(
            self, dockerfile_text):
        """It belongs to the compiler, not to cmake. Kept out of the cmake
        region on purpose: tests/test_e4_percona_build_flags.py collects
        line-anchored `-D` flags there and compares them against
        config/engines/percona.yml, which does not list this one."""
        after = dockerfile_text.split("cmake /src/source", 1)
        assert len(after) == 2, "the cmake invocation anchor is stale"
        region = after[1].split("make -j", 1)[0]
        anchored = re.findall(r"^\s*(-D\S+)", region, re.MULTILINE)
        assert not [f for f in anchored if "_FORTIFY_SOURCE" in f], (
            f"the fortify level is being passed as a cmake argument: "
            f"{anchored}")


# ---------------------------------------------------------------------------
# The live probe: does the flag actually do what the comment says?
# ---------------------------------------------------------------------------
#
# The assertions above are text matching, which cannot tell a load-bearing
# flag from a cargo-culted one. This compiles the libfido2 recipe with the
# Dockerfile's own CFLAGS and reports the honest result beside the result
# with the flag removed. It needs a compiler whose default fortify level is
# the hazard, which the documented host prerequisites do not include -- so it
# skips rather than failing a correctly set up machine.
#
# READ IT FOR WHAT IT IS: it establishes that the explicit level in CFLAGS
# suppresses the driver's own default, using UNSEEN_DEFINE_PROBE to make that
# default observable. It does NOT reproduce the build failure -- that needed
# dpkg-buildflags' `=3`, and tests/test_e8_package_flags_overwrite.py is where
# that is measured.

CC = shutil.which("cc") or shutil.which("gcc")


def _compile(flags: list[str], tmp_path) -> tuple[int, int]:
    """Returns (exit status, number of redefinition diagnostics)."""
    src = tmp_path / "fortify_probe.c"
    src.write_text("int main(void){return 0;}\n")
    done = subprocess.run(
        [CC, *flags, "-c", str(src), "-o", os.devnull],
        capture_output=True, text=True, timeout=120)
    return done.returncode, done.stderr.count('"_FORTIFY_SOURCE" redefined')


@pytest.mark.skipif(CC is None, reason="no host C compiler; see the module "
                                       "docstring -- docker/python3/pyyaml/"
                                       "git/curl are the documented host "
                                       "prerequisites")
class TestTheFlagIsLoadBearingOnThisHost:

    @pytest.fixture(scope="class")
    def host_default_level(self) -> str:
        done = subprocess.run([CC, "-O3", "-dM", "-E", "-"], input="",
                              capture_output=True, text=True, timeout=120)
        found = re.search(r"#define _FORTIFY_SOURCE (\S+)", done.stdout)
        return found.group(1) if found else ""

    @pytest.fixture(scope="class")
    def honest_flags(self, dockerfile_text) -> list[str]:
        return [*LIBFIDO2_FLAGS, *expanded_cflags(dockerfile_text),
                UNSEEN_DEFINE_PROBE]

    @pytest.fixture(scope="class")
    def probe(self, honest_flags, host_default_level, tmp_path_factory):
        """The pair, or a skip explaining why this host cannot produce one."""
        if not host_default_level:
            pytest.skip("this compiler supplies no default _FORTIFY_SOURCE, "
                        "so the conflict this flag removes cannot arise here "
                        "and the probe would prove nothing")
        if host_default_level == TREE_FORTIFY_LEVEL:
            pytest.skip(f"this compiler defaults to the tree's own level "
                        f"({host_default_level}), so there is nothing for it "
                        f"to disagree with")
        tmp_path = tmp_path_factory.mktemp("fortify")

        # -march=x86-64-v3 and the rest have to be accepted here, or a
        # failure below would be about the flags and not about fortify.
        usable, _ = _compile(
            [f for f in honest_flags if f != UNSEEN_DEFINE_PROBE], tmp_path)
        if usable != 0:
            pytest.skip("this host's compiler will not take the builder's "
                        "own CFLAGS, so the probe cannot run here")

        mutant_flags = [f for f in honest_flags
                        if not f.startswith("-D_FORTIFY_SOURCE=")]
        assert len(mutant_flags) == len(honest_flags) - 1, (
            "the mutation did not apply: no -D_FORTIFY_SOURCE= to remove")
        return (_compile(honest_flags, tmp_path),
                _compile(mutant_flags, tmp_path),
                host_default_level)

    def test_honest_and_mutant_differ(self, probe):
        honest, mutant, level = probe
        report = (f"honest (exit, redefinitions) = {honest}; "
                  f"mutant with the flag removed = {mutant}; "
                  f"host default level = {level}")

        assert honest != mutant, (
            f"DEAD PROBE -- removing the flag changed nothing, so this test "
            f"is measuring nothing on this host. {report}")
        assert honest == (0, 0), (
            f"the builder's own CFLAGS do not compile the libfido2 recipe "
            f"cleanly, which is the failure this flag exists to remove. "
            f"{report}")
        assert mutant[1] >= 1, (
            f"without the flag the redefinition did not appear, so this "
            f"host's compiler is not injecting a default level and the "
            f"suppression this probe is about cannot be observed here. "
            f"{report}")
        assert mutant[0] != 0, (
            f"without the flag the compile still succeeded, so "
            f"-pedantic-errors is not making the redefinition fatal here. "
            f"{report}")
