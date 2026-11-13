#!/usr/bin/env bash
#
# Entrypoint for the vector-bench Percona Server runtime image.
#
#   vb-entrypoint server [extra mysqld args...]   start the server (init first run)
#   vb-entrypoint init                            initialise the data directory only
#   vb-entrypoint client [args...]                open a client on the unix socket
#   vb-entrypoint <anything else>                 exec it verbatim
#
# Percona Server's `9.7` series is MySQL 9.7.4 based at the pin (MYSQL_VERSION on the
# pinned tree: MAJOR=9 MINOR=7 PATCH=4 EXTRA=-4, MATURITY="LTS"), so initialisation is
# `mysqld --initialize-insecure`, the same as AliSQL's and not MariaDB's
# mariadb-install-db. This file is a close copy of entrypoint-alisql.sh for that
# reason; read the two side by side.
#
# REVIEW STATUS: never run. docs/09-adding-percona-server.md §7 lists what the
# smoke run has to prove.
#
# THREE THINGS THAT ARE NOT COPIED FROM THE AliSQL ENTRYPOINT:
#
# 1. There is no --vidx-disabled=OFF equivalent, and nothing else to enable.
#    AliSQL ships VIDX disabled (ER_VECTOR_DISABLED until vidx_disabled=OFF) and
#    MariaDB needs nothing; Percona's vector index is on in the build and needs
#    no flag. There is also no isolation-level requirement: AliSQL raises
#    ER_NOT_SUPPORTED_YET on vector DML at anything but READ COMMITTED, and
#    nothing in the pinned tree's percona_vector suite sets an isolation level
#    to make vector DML work.
#
# 2. ONE VECTOR TUNABLE GETS A FALLBACK HERE, AND ONLY IF NOBODY ELSE SET IT.
#    Percona has exactly two server-level vector knobs, and both belong to the
#    orchestrator, because they are fairness parameters:
#      * innodb_hnsw_max_memory  GLOBAL only, default 1073741824 (1 GiB), 0 = no
#        limit (mysql-test/suite/sys_vars/r/innodb_hnsw_max_memory_basic.result:3-5,
#        :13-17, :23-26). It is NOT a cache size. Exceeding it FAILS the
#        statement with ER_CAPACITY_EXCEEDED rather than evicting anything, so
#        the default silently caps how many rows can be indexed at all. The
#        resource pass must set it; see docs/09 §5.
#      * innodb_hnsw_ef_search   SESSION and GLOBAL, default 40, range 1..100000
#        (ha_innodb.cc:1205-1209). The harness sets it per session during
#        calibration, so a server-level value here would be overwritten anyway
#        and would only confuse the manifest. Nothing is done with it here.
#    Both arrive through VB_SERVER_ARGS from config/engines/percona.yml, which is
#    how every other engine's knobs arrive, and so land in run-manifest.json.
#
#    So why a fallback for the first one at all? Because docker/alisql/init.sql
#    sets the precedent and the reason transfers. Its lines 13-15 read "Vector
#    features ship disabled; the server flag covers startup, this covers a
#    server someone restarted without it", and repeat `vidx_disabled = OFF` in
#    the init file as well as the entrypoint flag. The case it protects is the
#    by-hand `docker run` that docs/03-running-manually.md documents. Without
#    an equivalent here, a by-hand Percona run gets the compiled-in 1 GiB and
#    dies with ER_CAPACITY_EXCEEDED partway through any corpus worth loading —
#    the by-hand instructions would be broken for this engine alone.
#
#    But the mechanism cannot be AliSQL's. `vidx_disabled = OFF` is the only
#    sane value, so repeating it in --init-file is harmless. `innodb_hnsw_max_memory`
#    is a MEASUREMENT PARAMETER, and --init-file runs after the server flags are
#    applied, so a `SET GLOBAL innodb_hnsw_max_memory = ...` in init.sql would
#    silently overwrite whatever the resource pass chose and put a value in the
#    manifest that was not the value in force. That is the whole class of bug
#    this repo is built to avoid.
#
#    Hence: the fallback lives here, in the server flags, and is applied ONLY
#    when no --innodb-hnsw-max-memory arrived from VB_SERVER_ARGS or argv. The
#    orchestrator always wins; a by-hand run gets a working server; and the
#    flag that was actually used is visible in the `exec` line this logs.
#
# 3. The RSA keypair matters here and does not for AliSQL. This base has no
#    mysql_native_password (see init.sql), so the bench account uses
#    caching_sha2_password, and a TCP client without TLS needs the server's
#    public key. `caching_sha2_password_auto_generate_rsa_keys` defaults ON and
#    writes the pair into the datadir during --initialize-insecure, so the key
#    exists; whether MariaDB Connector/C asks for it is the open question in
#    docs/09 §7. Nothing here forces it, because forcing an auth path this
#    file cannot test would hide the answer.

set -euo pipefail

ROOT_DIR="${VB_ROOT_DIR:-/opt/percona}"
DATA_DIR="${VB_DATA_DIR:-/var/lib/vbench/data}"
SOCKET="${VB_SOCKET:-/var/run/vbench/percona.sock}"
LOG_FILE="${VB_LOG_FILE:-/var/lib/vbench/percona.err}"
INIT_SQL="${VB_INIT_SQL:-/opt/vbench/init.sql}"

log() { printf '[vb-percona] %s\n' "$*" >&2; }

find_bin() {
  local name="$1" p
  for p in "$ROOT_DIR/bin/$name"; do
    [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
  done
  p="$(command -v "$name" 2>/dev/null || true)"
  [[ -n "$p" ]] && { printf '%s' "$p"; return 0; }
  return 1
}

MYSQLD="$(find_bin mysqld)" || { log "FATAL: no mysqld under $ROOT_DIR"; exit 1; }

user_args() { [[ "$(id -u)" -eq 0 ]] && printf '%s' "--user=root"; }

ensure_dirs() {
  mkdir -p "$DATA_DIR" "$(dirname "$SOCKET")" "$(dirname "$LOG_FILE")"
}

initialise() {
  if [[ -d "$DATA_DIR/mysql" ]]; then
    log "data directory already initialised at $DATA_DIR"
    return 0
  fi
  log "initialising data directory at $DATA_DIR"
  "$MYSQLD" \
    --no-defaults \
    --initialize-insecure \
    --datadir="$DATA_DIR" \
    --log-error="$LOG_FILE" \
    $(user_args) >&2
  log "initialisation complete"
}

start_server() {
  ensure_dirs
  initialise

  # --skip-grant-tables is deliberately absent, for the reason the AliSQL and
  # MariaDB entrypoints give: on MySQL 8 and later it implicitly enables
  # --skip-networking, which would make this server unreachable over TCP from
  # the harness container. The bench account is created from --init-file instead,
  # so all three MySQL-family engines share one startup and auth shape.
  local -a args=(
    --no-defaults
    --datadir="$DATA_DIR"
    --socket="$SOCKET"
    --log-error="$LOG_FILE"
    --pid-file=/var/run/vbench/percona.pid
    --skip-name-resolve
  )
  # NOTE: no --mysqlx=0 here, same as the AliSQL entrypoint. The image is built
  # with -DWITH_MYSQLX=OFF, so the X plugin does not exist and the option is
  # rejected outright ("unknown variable 'mysqlx=0'"). Pass it via VB_SERVER_ARGS
  # if you build with the X plugin enabled.
  [[ -f "$INIT_SQL" ]] && args+=( --init-file="$INIT_SQL" )
  # --no-defaults must be the server's first argument; a duplicate arriving
  # later makes it exit 2. It is already in `args` above, so strip any copy
  # that comes in through VB_SERVER_ARGS rather than letting it break startup.
  if [[ -n "${VB_SERVER_ARGS:-}" ]]; then
    for _arg in ${VB_SERVER_ARGS}; do
      [[ "$_arg" == "--no-defaults" ]] && continue
      args+=( "$_arg" )
    done
  fi
  args+=( "$@" )

  # The graph-memory fallback described in note 2. Applied only if nothing
  # already set it, so the orchestrator's value always wins and a by-hand
  # `docker run` still gets a server that can index more than ~300k rows.
  # Matched on the prefix because both `--flag=value` and `--flag value` are
  # valid mysqld spellings, and `innodb-hnsw-max-memory` and
  # `innodb_hnsw_max_memory` are both accepted.
  local have_max_memory=0 a
  for a in "${args[@]}"; do
    case "${a//_/-}" in
      --innodb-hnsw-max-memory|--innodb-hnsw-max-memory=*) have_max_memory=1 ;;
    esac
  done
  if (( ! have_max_memory )); then
    log "no --innodb-hnsw-max-memory given; defaulting to 0 (unlimited)."
    log "  The compiled-in default is 1 GiB, which refuses INSERTs with"
    log "  ER_CAPACITY_EXCEEDED once the graph reaches it. A benchmark run"
    log "  passes this explicitly from the resource pass; see docs/09 §3."
    args+=( --innodb-hnsw-max-memory=0 )
  fi

  local u; u="$(user_args)"; [[ -n "$u" ]] && args+=( "$u" )

  log "exec $MYSQLD ${args[*]}"
  exec "$MYSQLD" "${args[@]}"
}

cmd="${1:-server}"; shift || true
case "$cmd" in
  server) start_server "$@" ;;
  init)   ensure_dirs; initialise ;;
  client)
    CLIENT="$(find_bin mysql)" || { log "no client binary"; exit 1; }
    exec "$CLIENT" --socket="$SOCKET" "$@"
    ;;
  *) exec "$cmd" "$@" ;;
esac
