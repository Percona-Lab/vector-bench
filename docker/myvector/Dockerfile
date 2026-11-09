#
# MySQL 9.7 LTS + the MyVector plugin, both compiled from pinned sources.
#
# Targets:
#   runtime  — mysqld, the client, and myvector.so, nothing else.
#   bench    — runtime + the Python stack ann-benchmarks and the ops harness drive.
#
# Build context is work/buildctx/myvector/, produced by
# `scripts/prepare-sources.sh --engine myvector`. It holds ONE tarball with two
# trees: `source/` is mysql-server at `source.server_tag`, `myvector/` is the
# plugin at `source.tag` (a commit SHA, release v1.26.9).
#
# Why the server is compiled here rather than taken from the `mysql:9.7`
# image, which is what upstream ships the plugin in:
#
#  1. ABI. A MySQL plugin is compiled against one server tree's headers and
#     loaded into whatever binary is on the image. Upstream compiles against
#     `mysql-9.7.0` and drops the .so into a `mysql:9.7` image whose patch
#     level moves under the tag. Building both from one tree removes the
#     question.
#  2. Portability and -march. The distance kernels are hnswlib's, inside the
#     plugin, and must be compiled at the same -march as every other engine's
#     (CLAUDE.md: "Never mix -march between engines"). Compiling the server in
#     the same build gives it the same flags Percona Server's build gets,
#     which is the other engine on this server line.
#  3. Base OS. The official image is Oracle Linux; every bench image here is
#     Ubuntu 24.04, and the Python stack in `bench` is shared with them.
#
# The plugin is placed in-tree exactly as upstream's own CI does it
# (`.github/workflows/docker-publish.yml`, step "Build MyVector plugin"):
# src/*.cc, include/*.h, include/*.i and CMakeLists.txt into plugin/myvector.
# The vendor CMakeLists then takes its `MYSQL_ADD_PLUGIN` branch.

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
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev libedit-dev \
      python3 file \
      libldap2-dev libsasl2-dev libkrb5-dev libcurl4-openssl-dev libtirpc-dev \
      libpam0g-dev liblz4-dev libzstd-dev libevent-dev libre2-dev \
      perl gawk unzip curl

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# The vendor's in-tree recipe, verbatim in effect. Then the plugin's
# install script, minus its `INSTALL PLUGIN` line: the entrypoint loads the
# plugin with --plugin-load-add on every start instead, so that the plugin's
# own system variables (`--myvector-index-dir` and friends) are accepted on
# the command line and therefore appear in the server flags the manifest
# records. INSTALL PLUGIN on top of that would fail with "already exists".
RUN set -eux; \
    mkdir -p /src/source/plugin/myvector; \
    cp /src/myvector/src/*.cc /src/source/plugin/myvector/; \
    cp /src/myvector/include/*.h /src/source/plugin/myvector/; \
    cp /src/myvector/include/*.i /src/source/plugin/myvector/ 2>/dev/null || true; \
    cp /src/myvector/CMakeLists.txt /src/source/plugin/myvector/; \
    mkdir -p /opt/myvector-share; \
    grep -v '^INSTALL PLUGIN' /src/myvector/sql/myvectorplugin.sql \
      > /opt/myvector-share/myvectorplugin.sql; \
    ! grep -q '^INSTALL PLUGIN' /opt/myvector-share/myvectorplugin.sql

# Compiler flags mirror docker/percona/Dockerfile, the other engine on this
# server line, so the two servers differ in their vector implementation and
# not in how they were compiled. -march reaches the plugin through
# CMAKE_CXX_FLAGS.
#
# ONE RUN, ending in `rm -rf /build`, and that is not tidiness. A
# RelWithDebInfo MySQL build directory is tens of GB; committed as its own
# layer and followed by a separate strip step (the shape percona's Dockerfile
# has), the next step's container failed to start on the build host with
# `exec: "/bin/sh": stat /bin/sh: no such file or directory` -- the snapshot
# of the huge layer came up empty. Nothing of the build directory is needed
# after `make install`, so it never becomes a layer.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    mkdir -p /build; \
    cd /build; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} -D_FORTIFY_SOURCE=2 ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake /src/source \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_C_FLAGS="$CFLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
      -DCMAKE_INSTALL_PREFIX=/opt/myvector \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/myvector \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      -DINSTALL_MYSQLTESTDIR= \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS" --output-sync=target || make --jobs=1; \
    make install; \
    test -f /opt/myvector/lib/plugin/myvector.so; \
    cp /opt/myvector-share/myvectorplugin.sql /opt/myvector/share/myvectorplugin.sql; \
    echo "${MARCH}" > /opt/myvector/.march; \
    cd /; rm -rf /build /src; \
    find /opt/myvector -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug; \
    rm -rf /opt/myvector/mysql-test /opt/myvector/lib/*.a

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench MySQL 9.7 + MyVector runtime"
ARG MYVECTOR_TAG=unknown
ARG MYSQL_SERVER_TAG=unknown
ENV VB_ENGINE=myvector \
    VB_ENGINE_TAG=${MYVECTOR_TAG} \
    VB_MYSQL_SERVER_TAG=${MYSQL_SERVER_TAG} \
    VB_ROOT_DIR=/opt/myvector \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/myvector.sock \
    PATH=/opt/myvector/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g libedit2 \
      libldap2 libsasl2-2 libcurl4t64 libtirpc3t64 liblz4-1 libzstd1 libevent-2.1-7t64 \
      libre2-10 ca-certificates procps tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/myvector /opt/myvector
COPY entrypoint-myvector.sh /usr/local/bin/vb-entrypoint
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
LABEL org.opencontainers.image.title="vector-bench MySQL 9.7 + MyVector + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config \
      gdb \
    && rm -rf /var/lib/apt/lists/*

# Connector/Python 2.0, pure Python, the same client every MySQL-family
# engine here is driven through (docker/_shared/requirements-mysql.txt).
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; v = mariadb.__version__; assert v.startswith('2.0.0rc2') and v.endswith('+native'), v; print('Connector/Python', v, 'OK')"

WORKDIR /home/app
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
