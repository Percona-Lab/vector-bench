#
# AliSQL (MySQL 8.0.44 base + VIDX vector index), built from a pinned tag.
#
# Targets:
#   runtime  — server binaries only. Use this to run AliSQL by hand
#              (see docs/03-running-manually.md).
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/alisql/, produced by scripts/prepare-sources.sh.
#
# AliSQL vendors every third-party dependency it needs (including boost 1.77 at
# extra/boost/boost_1_77_0) and has no git submodules, so this build needs no
# network beyond the base image and apt.

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG EXTRA_CXXFLAGS=""
ARG JOBS=0

# git and ninja-build are for the bundled DuckDB. AliSQL's CMake calls
# MYSQL_CHECK_DUCKDB() unconditionally on Linux, so extra/duckdb is compiled as
# an ExternalProject on every build and its Makefile probes for both.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev \
      python3 file \
      git ninja-build

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# AliSQL's own build.sh forces an in-source build, so cmake runs in the tree.
# Flags mirror its release configuration; see config/engines/alisql.yml.
#
# THE THREE FLAGS AT THE TOP OF THE cmake LINE BELOW ARE NOT DECORATION, AND
# `export CFLAGS=` ALONE DOES NOT REACH THIS TREE ON A DEBIAN HOST. AliSQL is
# MySQL-family and carries the same package-flags machinery Percona does; the
# full chain, with its measurements, is written out in
# docker/percona/Dockerfile beside the identical flags there. In short, all
# READ from the pinned AliSQL tree unless marked MEASURED:
#
#   * source CMakeLists.txt:782-799 offers WITH_PACKAGE_FLAGS for a
#     `LINUX_STANDALONE` DEB/RPM platform (the condition is :785) and :793
#     defaults it **ON** for RELWITHDEBINFO, which is this image's build type
#     (`ARG BUILD_TYPE` above and `build.type` in config/engines/alisql.yml).
#     Every other branch defaults it OFF: any sanitizer at :788-789,
#     DEBUG/WITH_DEBUG at :790-791, everything else at :794-795.
#   * the guard at source CMakeLists.txt:801-820 only stands down if
#     INITIAL_CMAKE_C_FLAGS is set (the condition is :804-805), and :303-306
#     captures that from `IF(DEFINED CMAKE_C_FLAGS)`
#     BEFORE `PROJECT()` -- which cmake satisfies from the command line or the
#     cache, never from the environment. MEASURED on cmake 3.28.3 with that
#     snippet alone: an exported `CFLAGS` leaves `CMAKE_C_FLAGS` undefined at
#     that point, a command-line `-DCMAKE_C_FLAGS=` defines it.
#   * source cmake/compile_flags.cmake:120-137 then runs
#     ADD_LINUX_DEB_FLAGS(). Its four `SET(... PARENT_SCOPE)` calls sit INSIDE
#     the `FOREACH(flag CPPFLAGS CFLAGS CXXFLAGS LDFLAGS)` at :121-136, so they
#     run once per iteration and only the last pass has every `GET_*`
#     populated; the one that matters is
#     `SET(CMAKE_C_FLAGS "${GET_CFLAGS}   ${GET_CPPFLAGS}" PARENT_SCOPE)`
#     (:131) -- an OVERWRITE -- and :133-135 replace the three LINKER_FLAGS
#     from dpkg's LDFLAGS too. dpkg-buildflags is present because this builder
#     installs build-essential above, which depends on dpkg-dev (MEASURED on
#     the build host).
#
# The consequence here is the quiet one. `-march=${MARCH}` -- x86-64-v3, which
# config/engines/alisql.yml and CLAUDE.md's "Never mix -march between engines"
# both require to equal every other source-built engine's -- IS DISCARDED
# along with the whole exported string, and dpkg's arrives in its place.
#
# THE FULL DELTA is catalogued in docker/percona/Dockerfile. ITS LOST COUNT IS
# FIVE AND THIS ENGINE'S IS FOUR, and the difference is not a rounding of the
# same fact: Percona's `export CFLAGS=` carries `-D_FORTIFY_SOURCE=2` and the
# one below does not, so that token is Percona's to lose and not ours.
# MEASURED against this file's own export line, the four lost here are
# `-march=x86-64-v3`, `-fexceptions`, `-fno-strict-aliasing`, and `-O3`
# replaced by dpkg's `-O2` to no effect. The gained twelve and the kept two
# are the same for both engines: twelve gained (`-flto=auto
# -ffat-lto-objects -fstack-protector-strong -fstack-clash-protection
# -fcf-protection -mno-omit-leaf-frame-pointer -Wformat -Werror=format-security
# -Wdate-time -ffile-prefix-map=<cwd>=. -O2 -D_FORTIFY_SOURCE=3`) and two kept
# (`-g`, `-fno-omit-frame-pointer`). MEASURED on the build host by tokenising
# `dpkg-buildflags --get CFLAGS` plus `--get CPPFLAGS` against the exported
# string. `-fno-strict-aliasing` is the one besides `-march` worth naming on
# its own: a MySQL-family build losing it at -O2 is a codegen change. The LTO
# is real LTO because compile_flags.cmake:133-135 replaces the LINKER_FLAGS
# too, and `dpkg-buildflags --get LDFLAGS` carries `-flto=auto
# -ffat-lto-objects` (MEASURED).
# `echo "${MARCH}" > /opt/alisql/.march` below writes the build ARG and not
# what cmake honoured, so the file and every `march` reader in this repo say
# x86-64-v3 whichever way the configure went: THE BREACH HAS NO SYMPTOM.
#
# THE OPTIMISATION LEVEL IS NOT PART OF THIS. `-O3` in the exported CFLAGS has
# never reached the compiler for this engine, before this fix or after it: the
# compile line is `CMAKE_C_FLAGS CMAKE_C_FLAGS_<configuration>` and
# RelWithDebInfo's own value ends it with `-O2 -g -DNDEBUG`, which this tree
# spells out itself at source CMakeLists.txt:2709-2719 along with the remedy
# this image does not pass (`-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=...`). MEASURED
# on cmake 3.28.3 and gcc 13: the emitted command is
# `cc -O3 -g -march=x86-64-v3 -O2 -g -DNDEBUG` and the last -O wins. That the
# study mixes -O2 and -O3 across engines is a separate defect; this fix neither
# causes nor repairs it.
#
# ALISQL'S ALREADY-PUBLISHED SMOKE NUMBERS PREDATE THIS FIX and were therefore
# almost certainly produced by a binary compiled WITHOUT -march=x86-64-v3 and
# WITH LTO. They are not comparable with numbers from an image built after this
# commit and must be re-measured rather than reused. That conclusion rests on
# -march and LTO alone.
#
# WHY THE PERCONA BUILD FAILURE NEVER HAPPENED HERE, which is not for the
# reason an earlier version of this note gave. MEASURED over
# sources/alisql-AliSQL-8.0.44-2.tar: `_FORTIFY_SOURCE` occurs 11 times on 10
# lines in 9 files, and exactly ONE of them is live code --
# extra/libevent/libevent-2.1.11-stable/CMakeLists.txt:365,
# `add_definitions(-D_FORTIFY_SOURCE=2)`. That line sits inside
# `if (EVENT__ENABLE_GCC_HARDENING)` (:357), an option declared default **OFF**
# at :321 and set nowhere else in the tree, so it is never emitted -- libevent
# is compiled (cmake/libevent.cmake:164-166, :148, :153 and
# source/CMakeLists.txt:1965) but that define is not. With no second,
# disagreeing definition, dpkg's `=3` merely replaced gcc's own default of 3
# and nothing was redefined at all. Independently, the `-pedantic-errors` that
# made the diagnostic FATAL for Percona is applied to nothing here:
# source/CMakeLists.txt:2110-2118 adds `extra/libfido2` only
# `IF((WITH_AUTHENTICATION_FIDO OR WITH_AUTHENTICATION_CLIENT_PLUGINS) AND NOT
# UDEV_WARN_MISSING)`, and config/engines/alisql.yml:96-97 pass both OFF.
# INFERRED FROM THE SOURCES, NOT OBSERVED: no AliSQL build log was available.
#
# The fix is the pair of defences Percona's own packaging uses, and both are
# on the cmake line below: `-DWITH_PACKAGE_FLAGS=OFF` closes the option, and
# `-DCMAKE_C_FLAGS="$CFLAGS"`/`-DCMAKE_CXX_FLAGS="$CXXFLAGS"` on the COMMAND
# LINE close the guard. Both are READ off build-ps/build-binary.sh (:307 and
# :290-291) at the Percona pin. The `export CFLAGS=`/`export CXXFLAGS=` lines
# STAY: the bundled DuckDB ExternalProject is built by its own Makefile and
# reads them from the environment, and the exported string is literally what
# the new `-D` flags pass on, so the two cannot drift. The command-line value
# must be NON-EMPTY to arm the guard: `SET(INITIAL_CMAKE_C_FLAGS
# ${CMAKE_C_FLAGS})` expands UNQUOTED, so `-DCMAKE_C_FLAGS=` with nothing after
# it supplies zero arguments and UNSETS the variable rather than setting it to
# "" -- MEASURED on cmake 3.28.3, `IF(DEFINED INITIAL_CMAKE_C_FLAGS)` is false
# there. `IF(NOT ...)` reads true either way, so the hatch stays open.
#
# tests/test_e8_package_flags_overwrite.py holds this for both engines.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    cd /src/source; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake . \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DWITH_PACKAGE_FLAGS=OFF \
      -DCMAKE_C_FLAGS="$CFLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
      -DCMAKE_INSTALL_PREFIX=/opt/alisql \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/alisql \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS"; \
    make install; \
    echo "${MARCH}" > /opt/alisql/.march

RUN find /opt/alisql -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench AliSQL runtime"
ARG ALISQL_TAG=unknown
ENV VB_ENGINE=alisql \
    VB_ENGINE_TAG=${ALISQL_TAG} \
    VB_ROOT_DIR=/opt/alisql \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/alisql.sock \
    PATH=/opt/alisql/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g \
      ca-certificates procps tzdata \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/alisql /opt/alisql
COPY entrypoint-alisql.sh /usr/local/bin/vb-entrypoint
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
LABEL org.opencontainers.image.title="vector-bench AliSQL + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config

# Same client stack as the MariaDB bench image on purpose: MariaDB
# Connector/Python 2.0 (pure Python) speaks the MySQL protocol, so every
# MySQL-family engine is driven through an identical driver with
# binary-protocol prepared statements. Differences in the numbers then belong
# to the servers, not to the clients. See requirements-mysql.txt.
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; v = mariadb.__version__; assert v.startswith('2.0.0rc2') and v.endswith('+native'), v; print('Connector/Python', v, 'OK')"


WORKDIR /home/app
ENV ALISQL_ROOT_DIR=/opt/alisql \
    ALISQL_DB_WORKSPACE=/var/lib/vbench \
    DO_INIT_ALISQL=yes
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
