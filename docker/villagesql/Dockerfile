#
# VillageSQL (MySQL 9.7.2 base) + the vsql_vector extension (SVECTOR type and
# HNSW custom index), both compiled from pinned sources.
#
# Targets:
#   runtime  — mysqld, the client, and vsql_vector.veb, nothing else.
#   bench    — runtime + the Python stack ann-benchmarks and the ops harness drive.
#
# Build context is work/buildctx/villagesql/, produced by
# `scripts/prepare-sources.sh --engine villagesql`. It holds ONE tarball with
# two trees: `source/` is villagesql-server at `source.server_tag`, and
# `vsql-vector/` is the extension at `source.tag` (a commit SHA on `main`).
#
# Two compiles, in this order, and the order is forced:
#
#  1. The server. VillageSQL is a MySQL fork, so this is the same cmake build
#     docker/myvector/Dockerfile runs for its MySQL 9.7.2 tree -- the same
#     CFLAGS, the same cmake flags in config/engines/villagesql.yml -- so the
#     two 9.7.2-based servers here differ in their vector implementation and
#     not in how they were compiled. The `sdk` target is part of `all` at
#     this tag (villagesql/CMakeLists.txt: `add_custom_target(sdk ALL ...)`)
#     and stages `villagesql-extension-sdk-*/` in the build tree, which is
#     what the extension's FindVillageSQL looks for.
#  2. The extension, against the server's BUILD tree (`VillageSQL_BUILD_DIR`)
#     rather than an installed SDK, because vsql-vector sets
#     `VSQL_USE_DEV_ABI ON`: it links the dev/unstable ABI (`storage_api.h`,
#     the preview index-builder and session-variable headers), which is the
#     staged SDK's `include-dev/`. This is the upstream README's own recipe.
#
# THE EXTENSION'S OPTIMISATION LEVEL IS CHECKED, NOT ASSUMED. Its distance
# kernels -- the hot loop of every query -- live in the .veb, not in mysqld.
# vsql-vector's CMakeLists defaults an empty CMAKE_BUILD_TYPE to
# RelWithDebInfo, but an earlier revision of that file left it empty, which
# compiled the kernels at -O0 (VillageSQL's own adapter for this framework
# measured that as 6-10x slower). Release is passed explicitly and the build
# fails unless an -O1/-O2/-O3 flag reaches the svector target's compile line.
#
# -march reaches BOTH compiles through CMAKE_C/CXX_FLAGS: the extension's
# CMakeLists sets no -march of its own, and without it the kernels compile to
# SSE2 only. x86-64-v3, like every other engine built from source (CLAUDE.md,
# "Never mix -march between engines").
#
# ONE RUN for both compiles, ending in `rm -rf /build`, for the reason
# docker/myvector/Dockerfile gives: a RelWithDebInfo MySQL build directory is
# tens of GB, and committed as a layer it broke the next step on the build
# host. The cost is that an extension-only re-pin recompiles the server too.

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG EXTRA_CXXFLAGS=""
ARG JOBS=0

ENV DEBIAN_FRONTEND=noninteractive
# docker/myvector/Dockerfile's list (same MySQL 9.7.2 tree) plus `zip` and
# `jq`, which villagesql/bld_tools/setup_linux_build_env.sh -- the vendor's
# definitive dependency list -- names and the VEB packaging step uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev libedit-dev \
      python3 file git ca-certificates \
      libldap2-dev libsasl2-dev libkrb5-dev libcurl4-openssl-dev libtirpc-dev \
      libpam0g-dev liblz4-dev libzstd-dev libevent-dev libre2-dev \
      perl gawk unzip curl zip jq

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    mkdir -p /build/server /build/ext; \
    cd /build/server; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} -D_FORTIFY_SOURCE=2 ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake /src/source \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_C_FLAGS="$CFLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
      -DCMAKE_INSTALL_PREFIX=/opt/villagesql \
      -DINSTALL_LAYOUT=STANDALONE \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/villagesql \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      -DINSTALL_MYSQLTESTDIR= \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS" --output-sync=target || make --jobs=1; \
    make install; \
    ls -d /build/server/villagesql-extension-sdk-*/include-dev >/dev/null; \
    : "--- 2. the extension, against the build tree above ---"; \
    cd /build/ext; \
    cmake /src/vsql-vector \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS="-march=${MARCH} ${EXTRA_CXXFLAGS}" \
      -DCMAKE_CXX_FLAGS="-march=${MARCH} ${EXTRA_CXXFLAGS}" \
      -DVillageSQL_BUILD_DIR=/build/server ; \
    make -j"$JOBS" svector; \
    flags="$(find . -name flags.make -path '*svector.dir*' | head -1)"; \
    olevels="$(grep -oE -- '-O[0-3s]' "$flags" | sort -u | tr '\n' ' ')"; \
    echo "svector optimisation flags: ${olevels:-none}"; \
    echo "$olevels" | grep -qE -- '-O[123]' \
      || { echo "FATAL: svector compiled without -O1/-O2/-O3"; cat "$flags"; exit 1; }; \
    grep -q -- "-march=${MARCH}" "$flags" \
      || { echo "FATAL: svector compiled without -march=${MARCH}"; cat "$flags"; exit 1; }; \
    make -j"$JOBS"; \
    veb="$(find . -name 'vsql_vector.veb' | head -1)"; \
    test -n "$veb"; \
    mkdir -p /opt/villagesql/lib/veb; \
    cp "$veb" /opt/villagesql/lib/veb/vsql_vector.veb; \
    echo "${MARCH}" > /opt/villagesql/.march; \
    cd /; rm -rf /build /src; \
    find /opt/villagesql -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug; \
    rm -rf /opt/villagesql/mysql-test /opt/villagesql/lib/*.a

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench VillageSQL + vsql_vector runtime"
ARG VILLAGESQL_TAG=unknown
ARG VILLAGESQL_SERVER_TAG=unknown
ENV VB_ENGINE=villagesql \
    VB_ENGINE_TAG=${VILLAGESQL_TAG} \
    VB_VILLAGESQL_SERVER_TAG=${VILLAGESQL_SERVER_TAG} \
    VB_ROOT_DIR=/opt/villagesql \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/villagesql.sock \
    PATH=/opt/villagesql/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g libedit2 \
      libldap2 libsasl2-2 libcurl4t64 libtirpc3t64 liblz4-1 libzstd1 libevent-2.1-7t64 \
      libre2-10 ca-certificates procps tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/villagesql /opt/villagesql
COPY entrypoint-villagesql.sh /usr/local/bin/vb-entrypoint
COPY init.sql /opt/vbench/init.sql
RUN chmod +x /usr/local/bin/vb-entrypoint \
    && test -f /opt/villagesql/lib/veb/vsql_vector.veb \
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
LABEL org.opencontainers.image.title="vector-bench VillageSQL + vsql_vector + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Connector/Python over Connector/C, the same client every MySQL-family
# engine here is driven through (docker/_shared/requirements-mysql.txt).
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN MARIADB_CONFIG=/usr/bin/mariadb_config \
    pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; print('Connector/Python', mariadb.__version__, 'OK')"

WORKDIR /home/app
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
