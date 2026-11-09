#
# MariaDB, built from a pinned release tag.
#
# Targets:
#   runtime  — server binaries only, no compiler, no source. Use this to run
#              MariaDB by hand (see docs/03-running-manually.md).
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/mariadb/, produced by scripts/prepare-sources.sh.

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG JOBS=0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build bison pkg-config \
      libncurses-dev libssl-dev libaio-dev zlib1g-dev libpcre2-dev \
      libcurl4-openssl-dev libxml2-dev libedit-dev libbz2-dev \
      liblz4-dev libzstd-dev libsnappy-dev libjemalloc-dev \
      libnuma-dev libkrb5-dev libjudy-dev gnutls-dev \
      python3 file

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# Vector distance kernels are the hot loop; -march decides whether they get
# AVX2/FMA (x86-64-v3), AVX-512 (native, on capable hardware), or neither.
# The chosen value is recorded in the image so the report can state it.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    cmake -S /src/source -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_INSTALL_PREFIX=/opt/mariadb \
      -DCMAKE_C_FLAGS="-march=${MARCH}" \
      -DCMAKE_CXX_FLAGS="-march=${MARCH}" \
      ${CMAKE_FLAGS} ; \
    cmake --build /build --parallel "$JOBS"; \
    cmake --install /build; \
    echo "${MARCH}" > /opt/mariadb/.march

# Strip debug info: RelWithDebInfo binaries are ~1.5 GB and the runtime image
# does not need them. Symbols stay in the builder stage if you need to profile.
RUN find /opt/mariadb -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench MariaDB runtime"
ARG MARIADB_TAG=unknown
ENV VB_ENGINE=mariadb \
    VB_ENGINE_TAG=${MARIADB_TAG} \
    VB_ROOT_DIR=/opt/mariadb \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/mariadb.sock \
    PATH=/opt/mariadb/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libaio1t64 libncurses6 libpcre2-8-0 libnuma1 \
      libcurl4t64 libxml2 libedit2 libzstd1 liblz4-1 libsnappy1v5 \
      libjemalloc2 zlib1g libjudydebian1 \
      ca-certificates procps tzdata \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/mariadb /opt/mariadb
COPY entrypoint-mariadb.sh /usr/local/bin/vb-entrypoint
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
LABEL org.opencontainers.image.title="vector-bench MariaDB + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config

# The MySQL-family engines deliberately share one client stack — MariaDB
# Connector/Python 2.0, pure Python, binary-protocol prepared statements — so
# their ingest and query numbers are not skewed by different client
# implementations or by one client behaving differently per server. See
# requirements-mysql.txt for why 2.0 and why pure Python; the build fails if
# any other implementation is what imports.
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; v = mariadb.__version__; assert v.startswith('2.0.0rc2') and v.endswith('+native'), v; print('Connector/Python', v, 'OK')"


WORKDIR /home/app
ENV MARIADB_ROOT_DIR=/opt/mariadb \
    MARIADB_DB_WORKSPACE=/var/lib/vbench \
    DO_INIT_MARIADB=yes
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
