#
# Percona Server 9.7.4-4 (MySQL 9.7.4 base + native InnoDB HNSW vector index),
# built from a pinned SHA on the `release-9.7.4-4` branch of the `9.7` series.
#
# Targets:
#   runtime  — server binaries only. Use this to run Percona Server by hand
#              (see docs/03-running-manually.md).
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/percona/, produced by scripts/prepare-sources.sh.
#
# REVIEW STATUS: this file has never been built. docs/09-adding-percona-server.md
# §7 lists what the first build has to prove. Every flag below cites the line on
# the pinned tree that made it necessary; none was guessed from a commit subject.
#
# Differences from docker/alisql/Dockerfile, which is otherwise the closest
# template (both are MySQL-derived, both initialise with
# `mysqld --initialize-insecure`, both create the bench account from --init-file):
#
#   1. OUT-OF-SOURCE BUILD. AliSQL's own build.sh forces an in-source build.
#      Percona Server refuses one: CMakeLists.txt:583 is
#      `OPTION(FORCE_INSOURCE_BUILD "Allow in-source build" OFF)` and :606 says
#      "You *can* force in-source build by invoking cmake with
#      -DFORCE_INSOURCE_BUILD=1". We do not force it; cmake runs in /build.
#
#   2. NO SUBMODULE FETCH, WHICH TAKES FOUR FLAGS AND NOT TWO. The pinned
#      tree's .gitmodules lists four submodules, and prepare-sources.sh ships a
#      `git archive`, which omits submodule paths entirely. Every one of them
#      therefore has to be switched off, or cmake descends into a directory
#      that is not in the tarball:
#
#        storage/rocksdb/rocksdb  -> -DWITH_ROCKSDB=OFF
#        extra/coredumper         -> -DWITH_COREDUMPER=OFF
#                                    CMakeLists.txt:1558, default ON
#        extra/jwt-cpp            -> -DWITH_AUTH_OPENID_CONNECT=OFF
#                                    CMakeLists.txt:167, default ON
#        extra/libkmip            -> -DWITHOUT_COMPONENT_KEYRING_KMIP=1
#
#      The last one is the reason this list is four lines long. libkmip is not
#      behind an OPTION() with a default; it is behind a two-symbol condition
#      that fires when NEITHER symbol is defined — CMakeLists.txt:2295-2297:
#
#        IF(NOT DEFINED WITH_COMPONENT_KEYRING_KMIP AND NOT DEFINED WITHOUT_COMPONENT_KEYRING_KMIP
#           OR WITH_COMPONENT_KEYRING_KMIP)
#          ADD_SUBDIRECTORY(extra/libkmip)
#        ENDIF()
#
#      AND binds tighter than OR, so with neither defined that is
#      (TRUE AND TRUE) OR FALSE = TRUE — which is the original defect: an
#      earlier revision of this file enumerated all four submodules correctly,
#      said "two", and passed three flags, so the build would have failed at
#      configure time on a directory the tarball does not contain.
#
#      -DWITH_COMPONENT_KEYRING_KMIP=OFF closes the gate too. Measured, not
#      reasoned: the condition above was transcribed verbatim into a standalone
#      project and run under cmake 3.28.3 —
#
#        (no flags)                          -> WOULD_ADD_SUBDIRECTORY
#        -DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=0     -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=NO    -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=ON    -> WOULD_ADD_SUBDIRECTORY
#        -DWITHOUT_COMPONENT_KEYRING_KMIP=1  -> SKIPPED
#        -DWITHOUT_COMPONENT_KEYRING_KMIP=0  -> SKIPPED
#
#      An earlier revision of this note had that backwards, on the argument
#      that defining the WITH_ symbol leaves it defined. That conflated
#      `DEFINED WITH_X` with `WITH_X`: defining it is exactly what makes the
#      AND false, and the OR's second operand is `WITH_X` evaluated as a
#      boolean, where OFF, 0 and NO are CMake false constants. The component's
#      own guard agrees — components/keyrings/keyring_kmip/CMakeLists.txt:28 is
#      `IF(NOT WITH_COMPONENT_KEYRING_KMIP) RETURN()`, keyed off the WITH_
#      spelling. Reasoning about a tool's semantics when the tool is installed
#      and the answer is one command away is the mirror image of this round's
#      other lesson about quoted hedges.
#
#      WITHOUT_ is still the better choice here, for the reason the last row of
#      that table shows: the gate tests `DEFINED WITHOUT_...` and never its
#      value, so WITHOUT_ closes it on definedness alone and cannot be broken
#      by a value spelling. WITH_=OFF relies on false-constant evaluation,
#      which is correct but is one more thing to be right about.
#
#      The alternative (listing them in percona.yml's `source.submodules:`, the
#      way mariadb.yml lists libmariadb) would make the build fetch from three
#      more hosts for plugins no benchmark here exercises. With the four flags
#      below this build needs no network beyond the base image and apt — the
#      same property docker/alisql/Dockerfile relies on.
#
#   3. BOOST NEEDS NO FLAG AND NO DOWNLOAD. cmake/boost.cmake:26-27 reads
#      "# Always use the bundled version." / SET(BOOST_SOURCE_DIR
#      ${CMAKE_SOURCE_DIR}/extra/boost), and the pinned tree carries
#      extra/boost/boost_1_87_0. So there is no -DWITH_BOOST and no
#      -DDOWNLOAD_BOOST here, deliberately.
#
#   4. NO mysql_native_password ANYWHERE. See init.sql — the plugin is gone from
#      the server in this base and requesting it is a hard error.
#
#   5. WE DO NOT USE build-ps/build-binary.sh, AND THE REASON IS THE BENCHMARK
#      AND NOT CONVENIENCE. It is the obvious thing to reach for -- it is the
#      vendor's own build, so it must work -- and it was audited against the
#      pinned tree on 2026-10-14 rather than dismissed. Six of its choices
#      disqualify it here, and all six are read from the script at the pin:
#
#        :17, :55   TARGET_CFLAGS='' and the only -march it ever sets is
#                   `-m32 -march=i686`, for 32-bit. On x86_64 the script's own
#                   flags come from `rpm --eval %optflags` (:217-218) inside
#                   `if [ -n "$(command -v rpm)" ]` and this noble image has
#                   no rpm, so COMMON_FLAGS holds only :232's
#                   -DPERCONA_INNODB_VERSION and the SCRIPT names no -march.
#                   It does NOT erase a caller's: :233 is `export
#                   CFLAGS="$COMMON_FLAGS ${CFLAGS:-}"`, an APPEND. Naming one
#                   stays our job -- CLAUDE.md, "Never mix -march".
#        :270       WITH_PGO defaults to 1: a three-pass profile-guided build.
#                   Percona would be the only engine in the study with PGO.
#        :312       -DWITH_LTO=ON, against a tree default of OFF
#                   (CMakeLists.txt:1017, OPTION at :1020). Same objection.
#        :311       -DWITH_PERCONA_TELEMETRY=ON, against a tree default of OFF
#                   (CMakeLists.txt:2392). Periodic background work inside a
#                   server whose CPU and wall time this benchmark attributes to
#                   query execution. ON is Percona's PACKAGING choice.
#        :299, :313 -DWITH_ROCKSDB=ON and -DWITH_JS_LANG=ON with
#                   -DV8_LIB_DIR=${WITH_V8}/... — the rocksdb submodule note 2
#                   explains we do not ship, and a prebuilt V8 tree.
#        :308       -DFORCE_INSOURCE_BUILD=1, which note 1 exists to avoid.
#
#      Two of its flags are also simply dead on this tree, which is the other
#      half of why the list cannot be copied: `-DWITH_LIBEVENT=bundled` (:309)
#      and `-DWITH_INNODB_MEMCACHED=ON` (:302). Measured, not inferred --
#      `extra/libevent` and `plugin/innodb_memcached` are both 404 at this
#      commit, and neither symbol appears in the top-level CMakeLists.txt or
#      in any file under cmake/. cmake warns about unused -D variables instead
#      of failing, so copying them would be silent noise. Anyone re-deriving
#      this flag list from the vendor script will reach for them;
#      tests/test_e6_percona_vendor_alignment.py fails if they land.
#
#      What IS taken from the vendor is everything that is about making it
#      compile rather than about how fast it runs: the builder apt list (note
#      on it below), and `bundled` for the three third-party libraries where
#      this tree offers the choice -- curl (:304), zlib (:303) and zstd
#      (:310). zstd was already the default (cmake/zstd.cmake:25); zlib was
#      `system` here until this commit and is `bundled` in cmake/zlib.cmake's
#      own default too (:117). Bundled is the path Percona's CI exercises, and
#      every one of the five configure failures on 2026-10-14 was a missing
#      system library.
#
#      Vendor packages deliberately NOT taken, with the reason, so the audit
#      does not have to be redone: valgrind, doxygen, doxygen-gui, graphviz,
#      devscripts, debhelper, fakeroot, ccache, patchelf, python3-mysqldb,
#      libeatmydata, dh-apparmor and copyright-update are their packaging and
#      test pipeline, not compile inputs; libsystemd-dev and libjemalloc-dev
#      govern options this tree defaults OFF (CMakeLists.txt:1655 and :874);
#      libmecab-dev likewise, since build-binary.sh passes -DWITH_MECAB only
#      when asked (:84); and `libpam-dev` is the virtual name whose provider
#      we install directly -- see the test that exists for exactly that
#      mistake. ONE REVERSAL: gdb, for request 1; see the bench stage.
#

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG EXTRA_CXXFLAGS=""
ARG JOBS=0

# NOT the same shape as the AliSQL builder stage any more, and the difference
# is measured. `file` is used by the strip pass at the end; `python3` is
# required by the MySQL build itself (comp_err and the generated-source steps),
# not by the benchmark. Everything from libldap2-dev onwards is this tree's
# own requirement, found on 2026-10-14 by building it for the first time.
#
# AliSQL builds from the same MySQL lineage with none of them, so this is not
# a MySQL-family cost -- it is what Percona Server 9.0 plus its own components
# need. Five configure failures, in the order they happened:
#
#   CMakeLists.txt:2075   -DWITH_AUTHENTICATION_LDAP=ON, but missing system
#                         libraries  (fed by the LDAP, SASL and KERBEROS
#                         warnings together -- any one of the three trips it)
#   CMakeLists.txt:2127   -DWITH_CURL=system, but missing system libraries,
#                         preceded by an ALIAS failure for `ext::curl` at
#                         cmake/curl.cmake:246. WITH_CURL defaults to "system"
#                         on every UNIX (CMakeLists.txt:998-1002).
#   cmake/rpc.cmake:108   Could not find rpc/rpc.h in /usr/include or
#                         /usr/include/tirpc            -> libtirpc-dev
#   plugin/auth_ldap/CMakeLists.txt:21
#                         Required LDAP dev library not found. A PERCONA
#                         plugin, and the half of that which matters is that
#                         no upstream -DWITH_AUTHENTICATION_* flag governs it.
#                         The check is NOT unconditional, as an earlier
#                         revision of this note claimed: it sits inside
#                         `IF(WITH_PERCONA_AUTHENTICATION_LDAP)` at
#                         plugin/auth_ldap/CMakeLists.txt:16, declared at
#                         CMakeLists.txt:2004-2011 with the default ON
#                         everywhere but APPLE. So the governing flag exists;
#                         it is simply a different, Percona-specific one from
#                         the family we were setting, and
#                         -DWITH_PERCONA_AUTHENTICATION_LDAP=OFF would have
#                         disabled it.
#   components/keyrings/keyring_vault/CMakeLists.txt:33
#                         could not find CURL library, after -DWITH_CURL=0
#                         had been set to clear an earlier failure
#
#   CMakeLists.txt:2075   again, with libsasl2-dev installed and LDAP found:
#                         `Cannot find SASL development libraries` is what
#                         cmake/sasl.cmake prints, and the dev package is not
#                         what it was missing. MYSQL_CHECK_SASL looks for the
#                         SCRAM *plugin* -- FIND_LIBRARY(NAMES scram
#                         PATH_SUFFIXES sasl2) -- and returns without warning
#                         only when it finds one. libsasl2-dev ships nothing
#                         under sasl2/ at all; the unversioned
#                         sasl2/libscram.so symlink FIND_LIBRARY needs comes
#                         from libsasl2-modules, a RUNTIME package. The
#                         warning's own install hint names
#                         libsasl2-modules-gssapi-mit beside the dev package
#                         for this reason, and the vendor's install_deps lists
#                         both. Measured by unpacking all three .debs for
#                         noble, after a --no-cache rebuild proved the dev
#                         package was installed and the check failed anyway.
#
# EVERY ONE OF THESE HAS AN OFF SWITCH SOMEWHERE. An earlier revision of this
# note said only rpc.cmake did not, and that was wrong three times over. The
# switches are recorded so the strategy below reads as a choice rather than as
# the absence of one:
#
#   :2075              -DWITH_AUTHENTICATION_LDAP=OFF, and it was tried
#   :2127              -DWITH_CURL=0, and it was tried -- see keyring_vault
#   auth_ldap:21       -DWITH_PERCONA_AUTHENTICATION_LDAP=OFF, per above
#   keyring_vault:33   components/keyrings/keyring_vault/CMakeLists.txt:25 is
#                      `option(WITH_COMPONENT_KEYRING_VAULT ... ON)` with an
#                      early `return()` at :26-29, so =OFF closes it
#   rpc.cmake:108      -DWITH_TIRPC=bundled takes the branch at
#                      cmake/rpc.cmake:51-59, which sets TIRPC_FOUND itself
#                      and calls ADD_SUBDIRECTORY(extra/tirpc) -- and
#                      extra/tirpc IS in the tarball, since .gitmodules at
#                      this pin lists only the four submodules note 2 names
#
# UNPROVEN, AND DELIBERATELY LEFT THAT WAY: the vendor's own
# build-ps/percona-server-9.0_builder.sh:603-609 carries a comment asserting
# there is "no bundled-tirpc fallback despite extra/tirpc/ existing in the
# tree". That contradicts the branch cited on the last row above. Nothing here
# can settle it -- no build of this image has ever succeeded, so neither
# reading has been executed -- and it must not be resolved by argument.
#
# The strategy does not depend on which of those is right. Every flag that was
# tried moved the failure instead of removing it, and -DWITH_CURL=0 is the
# clearest case because it CAUSED keyring_vault:33; each further switch costs
# a multi-hour build to evaluate. Hence the rule recorded beside cmake_flags
# in config/engines/percona.yml: on this tree, switching a dependency off
# breaks a Percona component that wanted the target, while installing the
# dependency does not.
#
# WHY NO CONSUMER-SIDE FLAG COULD EVER HAVE SAVED THE CURL ONE, which is the
# sharpest thing here and the easiest to get backwards: the curl check at
# CMakeLists.txt:2121-2131 sits at TOP LEVEL, inside no IF() at all. It runs
# whether or not anything in the build wants curl, and :2127 is fatal on the
# library being absent rather than on anything needing it. OpenID Connect
# genuinely is a curl consumer -- plugin/auth_openid_connect/CMakeLists.txt:32-34
# links `curl`, inside `IF(WITH_AUTH_OPENID_CONNECT)` at :16 -- so the
# -DWITH_AUTH_OPENID_CONNECT=OFF in the cmake list below really does remove
# one, and it still does not help.
#
# The list is this tree's own build-ps/percona-server-9.0_builder.sh
# `install_deps()`, filtered to the development packages and plugins among its
# entries and to what noble actually has -- the vendor list also names gcc-8
# through gcc-15, valgrind, doxygen, python-mysqldb, dh-systemd, libzstd1-dev
# and libncurses5-dev, several of which have no candidate here. It is NOT "the
# subset that gates configure", which an earlier revision of this note claimed
# while also calling it a superset: it is a SUPERSET and it was not pruned.
# Only three entries are tied to a failure recorded above -- libldap2-dev,
# libtirpc-dev and libsasl2-modules. With -DWITH_CURL=bundled and the vendor's
# bundled libevent and zstd (build-binary.sh:309-310), libcurl4-openssl-dev,
# liblz4-dev, libzstd-dev, libevent-dev and libre2-dev are probably
# unnecessary; nothing here can run a build to find out which, so none were
# removed. Prune it against a build that succeeded, not against reasoning.
#
# libpam0g-dev AND NOT THE VENDOR'S `libpam-dev`, for a reason an earlier
# revision of this note also got wrong: `libpam-dev` does not resolve to
# nothing on noble -- it is a VIRTUAL package whose only provider is
# libpam0g-dev, and packages.ubuntu.com files it under "virtual". Naming the
# real package is still the right call, because a purely virtual name is one
# more thing for apt to be clever about, but no failure above asked for it:
# CMakeLists.txt:165 is `OPTION(WITH_PAM "Build with Percona PAM plugin" OFF)`,
# DEFAULT OFF, and only build-binary.sh:298 turns it on. libpam0g-dev is
# precautionary, like most of this list.
#
# TELEMETRY, recorded for whoever next compares this against build-binary.sh:
# that script passes -DWITH_PERCONA_TELEMETRY=ON at :311 and this build passes
# nothing at all. CMakeLists.txt:2392 is
# `option(WITH_PERCONA_TELEMETRY "Build Percona Telemetry component" OFF)` --
# already OFF -- so no telemetry component is built and no flag is needed to
# achieve that. Leave it alone rather than "fixing" the difference: a
# component doing periodic background work inside a server whose CPU and wall
# time this benchmark attributes to query execution is a confound, and ON is a
# choice Percona's PACKAGING makes, not the source default.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev libedit-dev \
      python3 file \
      libldap2-dev libsasl2-dev libkrb5-dev libcurl4-openssl-dev libtirpc-dev \
      libpam0g-dev liblz4-dev libzstd-dev libevent-dev libre2-dev libwrap0-dev \
      libsasl2-modules libsasl2-modules-gssapi-mit \
      perl gawk unzip curl

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# Flags mirror config/engines/percona.yml. -march is the only one that may ever
# differ between engines, and it must not: all seven engines compile SIMD
# distance kernels, and a different flag on one turns the benchmark into a
# comparison of compiler flags. The value is written to /opt/percona/.march so
# the ann overlay module can report which one the image actually used — see
# overlay/.../percona/module.py, which overrides _image_march() for exactly this.
#
# `make --output-sync` is a diagnosability flag and it is not optional.
# The first `build --engines percona` ever run ended with nothing more useful
# than
#
#   make: *** [Makefile:166: all] Error 2
#
# as its final line. The compiler diagnostic and the `[ 26%] Building CXX
# object ...` line that names the file it came from are written by two
# different processes, so above -j1 they land apart, and the sub-make's own
# `make[1]: *** Error 1` lands apart from the top-level `Error 2` as well.
# Measured on GNU Make 4.3, a two-level recursive build at -j4 with one target
# failing: without the flag the `error:` text sits three unrelated progress
# lines below the target that produced it, with three more after it before the
# top-level error; with `=target` it is the very next line. This image is built
# at `nproc` jobs by default, which on the reference host is 80.
#
# `=target` and not `=recurse`: `recurse` groups output per recursive make, and
# CMake's generated top-level Makefile recurses once into CMakeFiles/Makefile2
# for the whole build, so `=recurse` would buffer every line until the build
# ended -- no progress at all for the hours this takes. `=line` keeps progress
# live but still splits a multi-line diagnostic from its own target.
#
# `|| make -j1` is the other half, and it is about where the error lands
# rather than whether it is adjacent to its file. `--output-sync` puts
# the compiler diagnostic next to the `[ 26%] Building CXX object ...` line
# that names the file; it does NOT move either of them to the end of the log.
# When a target fails at -j80 make stops launching new work but lets the ~79
# already-running targets finish, and their blocks print AFTER the failing
# one -- so the last twenty lines of the log are the stragglers draining out,
# and three builds of this image were reported from that tail with the error
# never read. The serial retry resumes from wherever the parallel build got
# to, recompiles only what is missing, stops at the first error and prints it
# as the last thing in the log. It costs minutes rather than hours because
# everything before the failure is already built.
#
# Two consequences to know rather than discover:
#
#   * a failure that was a parallel-build RACE will not recur serially, so
#     the retry succeeds and the image builds. That is a better outcome than
#     failing, and it produces the same binary -- but it means a green build
#     here does not prove the parallel build is sound.
#   * `set -e` does not fire between the two, because `||` suppresses errexit
#     for its left side. If the serial retry also fails the chain still exits
#     non-zero, which is what fails the Docker build.
#
# The retry is spelled `--jobs=1` and not `-j1`, which is not a style choice.
# `make -j` is the region anchor two other suites split this file on, and it
# has to stay UNIQUE: with a second `make -j` token in the same command, the
# guard in tests/test_e5_percona_make_output_sync.py stops failing loudly
# when the flag is misplaced and starts silently falling through to the
# retry, widening the region past the invocation it is meant to bound.
# Measured: with `-j1` the "wrong side of the anchor" probe went from
# reporting an empty region to reporting a populated one, which is the
# failure the probe exists to detect. `--jobs=1` is identical to make.
#
# The flag goes AFTER -j"$JOBS" deliberately. `make -j` is the literal string
# tests/test_e1_percona_design.py and tests/test_e4_percona_build_flags.py
# split this file on to bound the cmake flag region, and they do not survive
# its loss equally: test_e4 asserts the anchor is present and fails loudly,
# while test_e1:350-351 splits without that assertion, so a missing separator
# silently widens its region from the cmake invocation to end of file -- every
# assertion still green over a bound that no longer exists. Measured, by
# running both parses over a copy of this file with the flag moved ahead of
# -j: the guarded region goes from every flag to none at all, while the
# unguarded one keeps every flag and swallows the rest of the file with them,
# as far down as `ARG PERCONA_TAG`. That asymmetry is why
# tests/test_e5_percona_make_output_sync.py pins the flag's side of the
# anchor rather than trusting either suite to notice.
#
# CFLAGS CANNOT BE HANDED TO THIS TREE BY `export` ALONE ON A DEBIAN HOST, AND
# THE cmake LINE BELOW CARRIES THREE FLAGS THAT EXIST ONLY FOR THAT. Every step
# of the chain is READ from the Percona tree -- at the 9.7.1 base this file was
# written against. At the current pin
# (20c57915626b0214bbfe49718beccc0b6ebd51a6) every READ line number below was
# re-read against the tree and holds, except the two that moved: the
# CMakeLists.txt comment block on the compile line is at :2963-2973 (it was
# :2959-2969 before four lines were added above it), and compiler_options.cmake
# moved down one line (a new INCLUDE at its top). That re-reading is not a
# configure, and a configure is a build. A step says MEASURED where it was
# measured on the build host rather than inside this image:
#
#   1. source CMakeLists.txt:747-764 -- for `LINUX_STANDALONE AND
#      (LINUX_DEB_PLATFORM OR LINUX_RPM_PLATFORM)` (:750) the tree offers a
#      WITH_PACKAGE_FLAGS option (:762-763), and :758 defaults it **ON** when
#      the build type is RELWITHDEBINFO. Every other branch defaults it OFF:
#      any sanitizer at :753-754, DEBUG/WITH_DEBUG at :755-756, everything
#      else at :759-760. This image is RelWithDebInfo (`ARG BUILD_TYPE` above,
#      and `build.type` in config/engines/percona.yml), so it is ON unless we
#      say otherwise. READ.
#      Both of that condition's terms hold here, and these two files were
#      fetched from the pin itself rather than inferred from AliSQL's copies:
#      `cmake/os/Linux.cmake:56-58` is
#      `IF(LINUX_DEBIAN OR LINUX_UBUNTU) SET(LINUX_DEB_PLATFORM 1)` and
#      CMakeLists.txt:286 is where `LINUX_UBUNTU` is set; the base image is
#      ubuntu:24.04. `cmake/install_layout.cmake:114-116` is
#      `IF(LINUX AND INSTALL_LAYOUT MATCHES "STANDALONE")
#      SET(LINUX_STANDALONE 1)`, and :80/:83 default INSTALL_LAYOUT to
#      STANDALONE, which nothing in this image overrides. READ.
#   2. source CMakeLists.txt:769-770 -- the escape hatch is
#      `IF(MY_COMPILER_IS_GNU AND WITH_PACKAGE_FLAGS AND NOT
#      INITIAL_CMAKE_C_FLAGS AND NOT INITIAL_CMAKE_CXX_FLAGS)`. READ.
#   3. source CMakeLists.txt:313-322 -- INITIAL_CMAKE_C_FLAGS is captured from
#      `IF(DEFINED CMAKE_C_FLAGS)`, and that block runs BEFORE `PROJECT()`;
#      :326 is where the tree says it has to ("Cannot
#      INCLUDE(CMakeDetermineSystem) prior to PROJECT initialization below").
#      The tree's own comment at :313-314 says the capture is for values picked
#      up "form the CACHE or from the command line" -- THE ENVIRONMENT IS NOT
#      IN THAT LIST, and cmake does not read $CFLAGS until the C language is
#      enabled, which happens inside PROJECT().
#      MEASURED, with that snippet alone on cmake 3.28.3 (noble's cmake): with
#      `CFLAGS` exported, `DEFINED CMAKE_C_FLAGS` is FALSE before PROJECT() and
#      the value appears only after it, so `NOT INITIAL_CMAKE_C_FLAGS` reads
#      TRUE and the hatch stays OPEN. With `-DCMAKE_C_FLAGS=` on the command
#      line it is DEFINED before PROJECT() and the hatch CLOSES. So exporting
#      CFLAGS does not arm the guard; only the command line does.
#      ALSO MEASURED, and the sharp edge on this flag: the command-line value
#      has to be NON-EMPTY. With `-DCMAKE_C_FLAGS=` and nothing after it,
#      CMAKE_C_FLAGS is DEFINED with an empty value, but the capture is
#      `SET(INITIAL_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})` with the expansion
#      UNQUOTED -- an empty expansion supplies zero arguments, so that is
#      `SET(INITIAL_CMAKE_C_FLAGS)`, which UNSETS the variable rather than
#      setting it to "". MEASURED: `IF(DEFINED INITIAL_CMAKE_C_FLAGS)` is
#      false in that case. Either way `IF(NOT ...)` reads TRUE, so the hatch
#      stays open and the overwrite happens anyway. That is why this image
#      passes the real
#      CFLAGS rather than a placeholder, and the second reason
#      `-DWITH_PACKAGE_FLAGS=OFF` is passed as well. (The vendor's own value
#      is non-empty too: build-binary.sh:282 sets TARBALL_WARN_SUPPRESS to
#      five `-Wno-` flags before :290-291 pass it. READ.)
#   4. source cmake/compile_flags.cmake:119-136 -- ADD_LINUX_DEB_FLAGS() then
#      runs. Its four `SET(... PARENT_SCOPE)` calls sit INSIDE the
#      `FOREACH(flag CPPFLAGS CFLAGS CXXFLAGS LDFLAGS)` at :120-135, not after
#      it, so they execute once per iteration and only the last pass has every
#      `GET_*` variable populated -- which is why the net effect is simply the
#      final values. The one that matters is
#      `SET(CMAKE_C_FLAGS "${GET_CFLAGS}   ${GET_CPPFLAGS}" PARENT_SCOPE)`
#      (:130), with the CXX equivalent at :131: an OVERWRITE and not an
#      append, so everything exported above is DISCARDED. :132-134 set
#      CMAKE_EXE_LINKER_FLAGS, CMAKE_MODULE_LINKER_FLAGS and
#      CMAKE_SHARED_LINKER_FLAGS from dpkg's LDFLAGS the same way. READ.
#   5. dpkg-buildflags is present in the builder because it installs
#      build-essential (the apt list above), so MY_DPKG_BUILDFLAGS is found and
#      step 4 fires. MEASURED on the build host (Ubuntu 24.04.5, dpkg 1.22.6 --
#      the same series as this image's ubuntu:24.04 base): `apt-cache depends
#      build-essential` lists `Depends: dpkg-dev`, and
#      `dpkg -S $(command -v dpkg-buildflags)` answers
#      `dpkg-dev: /usr/bin/dpkg-buildflags`. Its output there, also MEASURED:
#        --get CPPFLAGS -> `-Wdate-time -D_FORTIFY_SOURCE=3`
#        --get CFLAGS   -> `-g -O2 -fno-omit-frame-pointer
#            -mno-omit-leaf-frame-pointer -ffile-prefix-map=<cwd>=. -flto=auto
#            -ffat-lto-objects -fstack-protector-strong -fstack-clash-protection
#            -Wformat -Werror=format-security -fcf-protection`
#
# TWO CONSEQUENCES, AND THE SECOND IS THE SERIOUS ONE.
#
# (a) THE BUILD FAILURE. Three builds ended in `make: *** [Makefile:166: all]
#     Error 2` with nothing in the tail naming a file. The serial retry above
#     is what finally put the real error last:
#
#         extra/libfido2/libfido2-1.15.0/src/.../aes256.c.o
#         <command-line>: error: "_FORTIFY_SOURCE" redefined
#         <command-line>: note: this is the location of the previous definition
#
#     Both definitions are ordinary `-D`s on one command line and they
#     disagree. Step 5's `-D_FORTIFY_SOURCE=3` arrives in CMAKE_C_FLAGS through
#     CPPFLAGS, while the tree prepends its own `-DNDEBUG -D_FORTIFY_SOURCE=2`
#     into CMAKE_C_FLAGS_<BUILDTYPE> for the three non-debug types at source
#     CMakeLists.txt:1583-1588 (READ). gcc's own Ubuntu default never enters
#     it: the driver stands that default down as soon as a -D for the macro is
#     on the command line, and here there are two.
#
#     Only libfido2 FAILS on it. Everywhere else the same redefinition is a
#     warning and the build carries on -- bundled absl does, at 69% -- which is
#     why three builds reached 20-69% before dying. libfido2 adds
#     `-pedantic-errors` (source extra/libfido2/libfido2-1.15.0/CMakeLists.txt
#     :329-330, READ), and that turns this one diagnostic into a hard error.
#     -pedantic-errors is also the only flag tried that reproduces the log's
#     exact form: a bare `error:` with a `note:` and no `cc1: all warnings
#     being treated as errors` line. It is NOT -Werror, which prints a
#     `[-Werror]` tag and that summary. Nor can it be demoted --
#     -Wno-error=cpp, -Wno-pedantic and -Wno-builtin-macro-redefined all leave
#     the error standing (MEASURED).
#
# (b) `-march=x86-64-v3` IS SILENTLY DROPPED. Step 4 replaces CMAKE_C_FLAGS
#     wholesale, so the entire exported string goes and dpkg's arrives in its
#     place. THE WHOLE DELTA, because a partial list here would be the same
#     kind of mistake this note is about -- MEASURED by tokenising
#     `dpkg-buildflags --get CFLAGS` plus `--get CPPFLAGS` on the build host
#     against the exported string READ from the `export CFLAGS=` line below.
#     THE LOST COUNT IS THIS ENGINE'S: AliSQL loses FOUR, not five, because
#     its export line carries no `-D_FORTIFY_SOURCE=2` to lose. The gained
#     twelve and the kept two are the same for both.
#
#       LOST (5 here, 4 for AliSQL)
#         -march=x86-64-v3        the one that matters; see below
#         -fexceptions
#         -fno-strict-aliasing    see below
#         -O3                     replaced by dpkg's -O2, and of NO
#                                 consequence either way: the per-config
#                                 `-O2 -g -DNDEBUG` ends the command line
#                                 whatever CMAKE_C_FLAGS holds
#         -D_FORTIFY_SOURCE=2     replaced by dpkg's =3. The effective LEVEL
#                                 is still 2, because the tree's own =2 from
#                                 CMAKE_C_FLAGS_<BUILDTYPE> comes last -- but
#                                 the disagreement between the two is exactly
#                                 what emits consequence (a)'s diagnostic
#       GAINED (12)
#         -flto=auto -ffat-lto-objects -fstack-protector-strong
#         -fstack-clash-protection -fcf-protection
#         -mno-omit-leaf-frame-pointer -Wformat -Werror=format-security
#         -Wdate-time -ffile-prefix-map=<cwd>=.
#         -O2                     see -O3 above
#         -D_FORTIFY_SOURCE=3     see -D_FORTIFY_SOURCE=2 above
#       KEPT (2)
#         -g
#         -fno-omit-frame-pointer for two independent reasons: Ubuntu 24.04's
#                                 dpkg emits it too, AND
#                                 cmake/build_configurations/compiler_options
#                                 .cmake:42 and :52 set COMMON_C_FLAGS and
#                                 COMMON_CXX_FLAGS to it under
#                                 `IF(MY_COMPILER_IS_GNU)` (:41) and :79/:84
#                                 prepend them. READ at the pin.
#
#     `-fno-strict-aliasing` DESERVES ITS OWN SENTENCE. A MySQL-family C/C++
#     build silently losing it while compiling at -O2 is a codegen change and
#     arguably a correctness one, and it is precisely the class of thing this
#     note exists to catch. NOTHING IN THE THREE FILES WE HOLD AT THE PIN
#     re-adds it: `CMakeLists.txt`, `cmake/compile_flags.cmake` and
#     `cmake/build_configurations/compiler_options.cmake` contain zero
#     occurrences of either `-fno-strict-aliasing` or `-fexceptions`
#     (MEASURED). That is deliberately NOT a claim about the whole tree --
#     there is no Percona source tarball on this box and the rest of the tree
#     was never read, so a sub-directory may well re-add either one. What
#     those three files DO add, for contrast, is `-fno-omit-frame-pointer`
#     above and `SECTIONS_FLAG` = `-ffunction-sections -fdata-sections` at
#     compiler_options.cmake:35.
#
#     The LTO is real LTO and not just a compile flag, because
#     compile_flags.cmake:132-134 sets all three of cmake's LINKER_FLAGS
#     variables from dpkg's LDFLAGS as well -- MEASURED, `dpkg-buildflags
#     --get LDFLAGS` here is `-Wl,-Bsymbolic-functions -flto=auto
#     -ffat-lto-objects -Wl,-z,relro`. So link-time optimization, which
#     tests/test_e6_percona_vendor_alignment.py forbids this image from taking
#     off the vendor script, arrives through the back door instead.
#     THE BREACH HAS NO SYMPTOM. `echo "${MARCH}" > /opt/percona/.march` below
#     writes the build ARG, not what cmake honoured, so that file and every
#     `march` reader in this repo report x86-64-v3 whichever way the configure
#     went. CLAUDE.md's "Never mix -march between engines" is what this
#     violates, and a Percona number produced without the SIMD baseline is not
#     comparable with the three engines that demonstrably carry it -- mariadb,
#     mariadb123 and pgvector, which pass `-march` on the cmake command line
#     or through a Makefile and so never lost it. AliSQL is NOT in that list:
#     before this commit it had the identical defect and was almost certainly
#     built without `-march` too, which is why its published numbers have to
#     be re-measured rather than used as the comparison.
#
#     THE OPTIMISATION LEVEL IS NOT PART OF THIS, in either direction, and no
#     sentence here should imply otherwise. `-O3` in the exported CFLAGS has
#     NEVER reached the compiler for this engine, before this fix or after it.
#     Both trees say why in their own words -- source CMakeLists.txt:2963-2973
#     (AliSQL's copy at :2709-2719) records that the command line is
#     `g++ DEFINITIONS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_<configuration>` and
#     that RELWITHDEBINFO's own value is `-O2 -g -DNDEBUG`, and names the only
#     remedy: `cmake -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG"`,
#     which this image does not pass. MEASURED, on cmake 3.28.3 and gcc 13:
#     a RelWithDebInfo project configured with `-DCMAKE_C_FLAGS="-O3 -g
#     -march=x86-64-v3"` compiles with
#     `cc -O3 -g -march=x86-64-v3 -O2 -g -DNDEBUG`, and
#     `gcc -O3 -O2 -Q --help=optimizers` is byte-identical to `gcc -O2` and
#     differs from `gcc -O3` -- the LAST -O wins, so the effective level is 2.
#     (-march has no such competitor in the per-config flags, which is why it
#     DOES reach the compiler once step 4 stops discarding it.) That this
#     study mixes -O2 and -O3 across engines is a real and separate defect; it
#     is not this fix's to make or to claim.
#
# THE PREVIOUS FIX AIMED AT THE RIGHT SYMPTOM THROUGH THE WRONG MECHANISM AND
# DID NOT WORK, and this paragraph replaces its account rather than softening
# it. It added `-D_FORTIFY_SOURCE=2` to the exported CFLAGS on the theory that
# the tree's own 2 was reaching the preprocessor as a `-Wp,-D...` that the gcc
# driver's scan could not see, letting the driver add its own 3 beside it.
# There is no such `-Wp,` define in this tree. The emitting site is step 5, the
# conflicting value is 3 and it is a plain `-D`, the driver's own default is
# not involved at all, and -- decisively -- the export that fix added was
# itself discarded by step 4 along with everything else, so it could not have
# taken effect either way.
#
# THE FIX IS THE PAIR OF DEFENCES PERCONA'S OWN PACKAGING USES, and both are on
# the cmake line below: build-ps/build-binary.sh:307 passes
# `-DWITH_PACKAGE_FLAGS=OFF`, which closes step 1, and :290-291 pass
# `-DCMAKE_C_FLAGS=` and `-DCMAKE_CXX_FLAGS=` on the cmake COMMAND LINE, which
# closes step 3 (both READ at the pin). Either alone would be enough today;
# both are passed because they fail in different directions. A future tree that
# renames or re-defaults the option leaves the command line standing, and a
# future tree that learns to capture the environment leaves the option
# standing.
#
# THE `export CFLAGS=`/`export CXXFLAGS=` LINES STAY. Sub-builds that are not
# this cmake project read them from the environment -- extra/libfido2 and the
# bundled ExternalProjects -- and the exported string is literally what the new
# `-D` flags pass on, so the command line and the environment cannot drift
# apart.
#
# THE `-D_FORTIFY_SOURCE=2` TOKEN IN CFLAGS ALSO STAYS, AND ITS LEVEL IS NOT
# FREE TO DRIFT, but its job has changed and the change is worth being plain
# about. It must AGREE with the 2 the tree prepends at CMakeLists.txt:1583-1588
# -- 3 against that 2 is the redefinition conflict again rather than a stronger
# fix, and a bare `-D_FORTIFY_SOURCE` means 1, which conflicts the same way.
# WHAT IT DOES NOT BUY, AND AN EARLIER VERSION OF THIS NOTE CLAIMED IT DID:
# this image was NOT compiling at 3. Pre-fix the compile line ended with the
# tree's own `-D_FORTIFY_SOURCE=2` from CMAKE_C_FLAGS_RELWITHDEBINFO (:1586),
# AFTER dpkg's `=3` in CMAKE_C_FLAGS, and the last -D of a macro wins.
# MEASURED: `gcc -O2 -D_FORTIFY_SOURCE=3 -D_FORTIFY_SOURCE=2 -dM -E -` prints
# `#define _FORTIFY_SOURCE 2`, and the reverse order prints 3. So the image was
# compiling at 2 -- AGREEING with both MariaDB pins, which ask for 2 plainly
# (mariadb-11.8.8:CMakeLists.txt:290 and mariadb-12.3.2:CMakeLists.txt:303,
# both `MY_CHECK_AND_SET_COMPILER_FLAG("-D_FORTIFY_SOURCE=2" RELEASE
# RELWITHDEBINFO)`, READ at both pins) -- while printing a redefinition warning
# in every translation unit and dying in libfido2. There was no engine-to-engine
# codegen difference in the hardening level, and the argument that there was has
# been removed rather than softened. The divergence that WAS real is -march, and
# consequence (b) above is where that argument belongs.
# What this token is NOT, any more, is the thing that fixes the build: with
# WITH_PACKAGE_FLAGS=OFF there is no second, disagreeing definition left for it
# to agree with.
#
# THE PINNED AliSQL HAS THE IDENTICAL DEFECT. Its CMakeLists.txt:782-799 is
# this same option block (WITH_PACKAGE_FLAGS_DEFAULT ON for RELWITHDEBINFO at
# :793, sanitizer and DEBUG branches at :788-791), :801-820 is this same guard
# (the condition at :804-805), :303-306 is the same INITIAL_CMAKE_C_FLAGS
# capture, and its cmake/compile_flags.cmake:120-137 is this same function with
# the overwrite at :131 -- all READ at its own pin.
# docker/alisql/Dockerfile now carries the same three flags and a note of its
# own.
#
# WHY ALISQL NEVER HIT CONSEQUENCE (a), re-derived, because an earlier version
# of this note gave the wrong reason ("the string appears only in two
# comments"). MEASURED over sources/alisql-AliSQL-8.0.44-2.tar: `_FORTIFY_SOURCE`
# occurs 11 times on 10 lines in 9 files. Eight of those files are comments or
# inert -- source/CMakeLists.txt:783, gperftools' CMakeLists.txt:278 and
# configure.ac:349, libfido2's commented-out set() at :349, a
# `#ifndef`/`#define _FORTIFY_SOURCE 3` pair in libevent's evdns.c:56-57, and
# three rpm `%optflags` seds in packaging/rpm-* (4 hits: fedora names twice).
# Exactly ONE is live code:
# extra/libevent/libevent-2.1.11-stable/CMakeLists.txt:365,
# `add_definitions(-D_FORTIFY_SOURCE=2)`. But it is inside
# `if (EVENT__ENABLE_GCC_HARDENING)` (:357), an option declared with default
# **OFF** at :321 and set nowhere else in the tree (grepped: those two lines are
# its only occurrences) -- so it is not emitted. libevent itself IS compiled
# (cmake/libevent.cmake:164-166 default WITH_LIBEVENT to `bundled`, :148 and
# :153 set the path and ADD_SUBDIRECTORY it, and source/CMakeLists.txt:1965
# calls MYSQL_CHECK_LIBEVENT() unconditionally); it is the define that is
# switched off. So no second, disagreeing definition reached any AliSQL object:
# dpkg's `=3` simply replaced gcc's own default of 3 and nothing was redefined.
#
# AND INDEPENDENTLY, the escalation that killed this image cannot occur there at
# all. AliSQL vendors the same libfido2 1.15.0, with the same
# `-pedantic-errors` at extra/libfido2/libfido2-1.15.0/CMakeLists.txt:329-330 --
# but source/CMakeLists.txt:2110-2118 adds `extra/libfido2` only
# `IF((WITH_AUTHENTICATION_FIDO OR WITH_AUTHENTICATION_CLIENT_PLUGINS) AND NOT
# UDEV_WARN_MISSING)`, and config/engines/alisql.yml:96-97 pass BOTH of those
# OFF. So AliSQL compiles no libfido2 object and that flag is applied to
# nothing. (config/engines/percona.yml passes neither flag, which is why THIS
# image does compile it -- and why its log names
# extra/libfido2/libfido2-1.15.0/src/.../aes256.c.o.)
#
# INFERRED, NOT OBSERVED: no AliSQL build log was available. The two paragraphs
# above are read from the cmake sources at the pin, not from a recorded build.
# What is not in doubt is consequence (b), which applied to AliSQL in full and
# in silence.
#
# tests/test_e7_percona_fortify_level.py holds the fortify level and its
# reasoning; tests/test_e8_package_flags_overwrite.py holds this chain.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    mkdir -p /build; \
    cd /build; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} -D_FORTIFY_SOURCE=2 ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake /src/source \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DWITH_PACKAGE_FLAGS=OFF \
      -DCMAKE_C_FLAGS="$CFLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
      -DCMAKE_INSTALL_PREFIX=/opt/percona \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/percona \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      -DWITH_COREDUMPER=OFF \
      -DWITH_AUTH_OPENID_CONNECT=OFF \
      -DWITH_CURL=bundled \
      -DWITH_ROCKSDB=OFF \
      -DWITHOUT_COMPONENT_KEYRING_KMIP=1 \
      -DWITH_NDB=OFF \
      -DWITH_ROUTER=OFF \
      -DWITH_MYSQLX=OFF \
      -DWITH_UNIT_TESTS=OFF \
      -DWITH_SSL=system \
      -DWITH_ZLIB=bundled \
      -DWITH_ZSTD=bundled \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS" --output-sync=target || make --jobs=1; \
    make install; \
    echo "${MARCH}" > /opt/percona/.march

RUN find /opt/percona -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench Percona Server runtime"
ARG PERCONA_TAG=unknown
ENV VB_ENGINE=percona \
    VB_ENGINE_TAG=${PERCONA_TAG} \
    VB_ROOT_DIR=/opt/percona \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/percona.sock \
    PATH=/opt/percona/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g libedit2 \
      ca-certificates procps tzdata \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/percona /opt/percona
COPY entrypoint-percona.sh /usr/local/bin/vb-entrypoint
COPY init.sql /opt/vbench/init.sql
RUN chmod +x /usr/local/bin/vb-entrypoint \
    && mkdir -p "${VB_DATA_DIR}" /var/run/vbench \
    && chown -R mysql:mysql /var/lib/vbench /var/run/vbench \
    && chmod 0644 /opt/vbench/init.sql

ENV VB_INIT_SQL=/opt/vbench/init.sql \
    VB_DB_USER=bench \
    VB_DB_PASSWORD=bench

EXPOSE 3306
ENTRYPOINT ["/usr/local/bin/vb-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------------------
FROM runtime AS bench
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench Percona Server + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
# gdb IS THE ONE ENTRY REVERSED out of the "deliberately NOT taken" audit
# note above -- the SECOND of its fourteen entries, after valgrind -- and it
# is here rather than in `runtime` on purpose. TWELVE OF THE TWENTY-FOUR
# Percona ops units in run full-20261015-215006 died inside the first
# INSERT: exactly half. The denominator is checkable here --
# `config/profiles/full.yml` is four datasets and `ops.m_values: [8, 16,
# 32]`, so twelve units a pass and twenty-four across both -- and the twelve
# is the plan's count over the archived unit files, twelve of which are
# empty while twelve carry records; `results/` is gitignored, so that half
# is READ FROM THE PLAN and not re-counted here. Do not take this figure
# from `config/engines/percona.yml`'s `error_log` note, which still says
# "13 of 24": the plan establishes 13 as wrong and the crash-capture change
# corrects it there.
#
# Nothing in this repository could turn such a crash into text, because no
# image here installed a debugger. (The literal two words that name an apt
# install step are deliberately not written in this comment: the per-stage
# package parser in tests/test_e4_percona_build_flags.py matches them
# anywhere in the file, prose included, and this note poisoned the bench
# stage's parsed package list until it was reworded. There is now a test for
# that.) Request 1 of docs/12-percona-feedback-plan.md -- "get stack traces
# for the crashes", its §4, NOT request 3, which is filtered search against
# pgvector in §7 -- asks for the backtrace, so the decision is reversed and
# the note above records it rather than quietly dropping the entry. The
# note's own reason for existing is that the audit should not have to be
# redone, and a wrong request number in it sends the next auditor to the
# wrong section.
#
# WHY THE BENCH STAGE AND NOT THE RUNTIME STAGE. The runtime image is the
# container the ops path MEASURES: `OpsRun._start_server` starts it from
# `image.runtime`, under the pass's cpuset and memory limit, and a debugger
# in it is a package in the measured server's own filesystem. The bench
# image is the one `OpsRun.run_script` starts for a lab script, which is the
# route that opens a core. `FROM runtime AS bench` means this stage already
# has the mysqld the debugger needs, so nothing is copied between stages and
# no runtime layer changes -- `percona-runtime` is byte-for-byte what it
# would have been without this.
#
# THE ONE PLACE THAT IS NOT CLEAN, stated rather than left to be discovered:
# on the ANN path the engine and the benchmark share ONE container and it is
# this image -- `overlay/ann-benchmarks/ann_benchmarks/vb_mysql.py` starts
# the server with `Popen([ENTRYPOINT, "server"])` inside the bench
# container. So for recall/QPS the server does run in a filesystem that
# carries gdb. What that costs is disk and nothing else, and the package is
# the evidence rather than the assumption -- MEASURED with dpkg-query on a
# noble host, the same base this image uses: gdb 15.1-1ubuntu1~24.04.1 is
# 10.0 MiB installed, everything it ships outside /usr is
# /etc/gdb/gdbinit -- 39 bytes, one comment line and no commands -- and an
# empty /etc/gdb/gdbinit.d/ -- no service, no systemd unit, no apt hook --
# and its single Recommends is `libc-dbg`, which
# `--no-install-recommends` leaves out. Of its eighteen Depends the ones
# this image may not already carry through python3, build-essential and git
# come to at most 3.7 MiB (libmpfr6 1.2, libsource-highlight4t64 0.7,
# libgmp10 0.5, libbabeltrace1 0.5, libreadline8t64 0.5, libipt2 0.1,
# libxxhash0 0.1, libdebuginfod1t64 0.07). So order 15 MiB on an image whose
# pip layer alone is several hundred; the exact figure is whatever the first
# build's `docker images` prints. Nothing reads any of it unless a lab
# script invokes it, and no lab script runs during a measured pass --
# `cmd_lab` refuses to start beside a measurement for exactly that reason.
# Resident memory, cpuset and the cgroup accounting the report publishes are
# untouched: a file nobody opens occupies no page cache inside the limit.
#
# ONE GAP IN THAT ARGUMENT, since "nothing executes" is not the whole of it:
# gdb's Depends put SHARED LIBRARIES in the image, and some are libraries
# the measuring client links against -- libpython3.12t64, libstdc++6,
# zlib1g, libexpat1. REASONED, not measured, because resolving them needs
# an apt this change does not run: every one of them is already installed
# by this stage's python3-dev and build-essential from the same pocket, so
# gdb pulls in no library the client did not already link against and
# replaces no version of one. If a rebuild ever shows apt UPGRADING one of
# them to satisfy gdb, that is a different change and this paragraph is
# where the argument has to be remade.
#
# WHAT THE BACKTRACE WILL AND WILL NOT CONTAIN, and this is the part the
# plan gets optimistic about. The build is RelWithDebInfo, but the builder's
# last step strips it: `strip --strip-debug` over every ELF under
# /opt/percona. MEASURED on the build host with binutils 2.42 and gdb 15.1,
# on a toy program compiled -g -O0 and then stripped the same way -- the
# flag removes every `.debug_*` section and KEEPS `.symtab`:
#
#     with DWARF   #0 ... in inner (depth=5, seed=15) at c.c:2
#                         local_a = 35 / local_b = 50 / p = 0x0
#     after strip  #0 ... in inner ()
#                         No symbol table info available.
#
# So `thread apply all bt full` returns the call chain by function name --
# enough to attribute a crash to a code path, which is what request 1 asks
# for -- and no arguments, no locals and no line numbers. The strip is NOT
# removed here, and the magnitude of what that would add is NOT measured --
# which is itself the reason not to flip it in passing. Keeping the debug
# sections of every ELF under /opt/percona is a large multiple of the
# stripped tree for a MySQL-family RelWithDebInfo build, and it would land
# in the runtime image as well, since that is where `COPY --from=builder`
# puts the tree. Whoever needs variable values rather than a stack should
# make that change against a build that reports the figure.
#
# UNVERIFIED, all of it, because no build of this image has ever succeeded:
# that noble has a `gdb` candidate, that this layer builds, and that gdb can
# open a core written by this mysqld. The first
# `./run-benchmark.sh build --engines percona` is the gate, and it costs no
# extra build -- the re-pin rebuilds this image anyway.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config \
      gdb

# The same client stack as the MariaDB and AliSQL bench images, for the same
# reason: MariaDB Connector/Python 2.0 (pure Python) speaks the MySQL
# protocol, so every MySQL-family engine is driven through an identical client
# with binary-protocol prepared statements, and differences in the numbers
# belong to the servers. requirements-mysql.txt pins mariadb==2.0.0rc2 and
# says why.
#
# The auth handshake: this base has no mysql_native_password (see init.sql),
# so the bench account authenticates with caching_sha2_password. Over the unix
# socket the ann pass uses that is a cleartext exchange the server permits;
# over the TCP connection the ops harness uses, the client fetches the
# server's RSA public key, which is what the pinned `cryptography` is for.
# Verified with 2.0.0rc2 against this image's server over TCP on 2026-11-09.
# The prebuilt `mariadb-binary` wheel is the one that fails here, which is
# one reason it is not installed.
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; v = mariadb.__version__; assert v.startswith('2.0.0rc2') and v.endswith('+native'), v; print('Connector/Python', v, 'OK')"

WORKDIR /home/app
ENV PERCONA_ROOT_DIR=/opt/percona \
    PERCONA_DB_WORKSPACE=/var/lib/vbench \
    DO_INIT_PERCONA=yes
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
