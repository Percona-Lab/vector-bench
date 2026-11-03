#!/usr/bin/env bash
#
# Entrypoint for the vector-bench VillageSQL + vsql_vector runtime image.
#
#   vb-entrypoint server [extra mysqld args...]   start the server (init first run)
#   vb-entrypoint init                            initialise the data directory only
#   vb-entrypoint client [args...]                open a client on the unix socket
#   vb-entrypoint <anything else>                 exec it verbatim
#
# The shape of entrypoint-myvector.sh -- both are MySQL 9.7.2 trees, so
# initialisation is `mysqld --initialize-insecure` -- with the extension
# installed once, at initialisation, instead of a plugin loaded per start:
#
# 1. `INSTALL EXTENSION vsql_vector` RUNS ON A SHORT-LIVED, SOCKET-ONLY
#    SERVER, NOT DURING --initialize. VillageSQL's own adapter for this
#    framework recorded that the preview capability framework is not
#    registered under --initialize, so the load fails there. The install is
#    recorded in the catalog and the extension is loaded again on every later
#    start; it is checked here (`SVECTOR` must resolve in a CREATE TABLE)
#    rather than trusted, because a missing type otherwise surfaces as a
#    parse error on the first benchmark statement.
#
# 2. `--vsql-allow-preview-extensions=ON` IS PASSED ON EVERY START, not SET
#    PERSIST-ed. vsql_vector is a preview extension (it uses the dev ABI and
#    the custom-index preview): once it is installed, a server started
#    without the flag refuses to start ("Failed to load VEF extension
#    'vsql_vector': extension requires preview capabilities but
#    vsql_allow_preview_extensions is OFF", then "Aborting"), and SET GLOBAL
#    of it is refused in favour of SET PERSIST. A command-line value is
#    visible in the exec line and in the engine config's server flags; a
#    persisted one lives in mysqld-auto.cnf where nothing in a run records it.
#
# 3. NO mysql_native_password. VillageSQL's adapter for the 8.4 base enabled
#    it so the client stack matched MariaDB's; MySQL 9.7 has removed the
#    plugin, so the bench account takes caching_sha2_password exactly as the
#    Percona and MyVector accounts on the same server line do (init.sql).

set -euo pipefail

ROOT_DIR="${VB_ROOT_DIR:-/opt/villagesql}"
DATA_DIR="${VB_DATA_DIR:-/var/lib/vbench/data}"
SOCKET="${VB_SOCKET:-/var/run/vbench/villagesql.sock}"
LOG_FILE="${VB_LOG_FILE:-/var/lib/vbench/villagesql.err}"
INIT_SQL="${VB_INIT_SQL:-/opt/vbench/init.sql}"
EXTENSION="${VB_VILLAGESQL_EXTENSION:-vsql_vector}"

log() { printf '[vb-villagesql] %s\n' "$*" >&2; }

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
CLIENT="$(find_bin mysql)" || { log "FATAL: no mysql client under $ROOT_DIR"; exit 1; }

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
    --basedir="$ROOT_DIR" \
    --datadir="$DATA_DIR" \
    --log-error="$LOG_FILE" \
    $(user_args) >&2

  local boot_sock="/var/run/vbench/bootstrap.sock"
  log "installing extension $EXTENSION on a socket-only bootstrap server"
  local -a boot=( --no-defaults --basedir="$ROOT_DIR" --datadir="$DATA_DIR"
                  --socket="$boot_sock" --skip-networking --skip-log-bin
                  --log-error="$LOG_FILE" --pid-file=/var/run/vbench/bootstrap.pid
                  --vsql-allow-preview-extensions=ON )
  local u; u="$(user_args)"; [[ -n "$u" ]] && boot+=( "$u" )
  "$MYSQLD" "${boot[@]}" &
  local pid=$! i
  for i in $(seq 1 120); do
    "$CLIENT" -uroot --socket="$boot_sock" -e 'SELECT 1' >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { log "FATAL: bootstrap server exited"; tail -50 "$LOG_FILE" >&2; exit 1; }
    sleep 0.5
  done
  "$CLIENT" -uroot --socket="$boot_sock" < "$INIT_SQL"
  "$CLIENT" -uroot --socket="$boot_sock" -e "INSTALL EXTENSION $EXTENSION"
  # Proof the type resolves, not a hope (note 1).
  "$CLIENT" -uroot --socket="$boot_sock" -e \
    "CREATE DATABASE vb_probe; CREATE TABLE vb_probe.t (id INT PRIMARY KEY, v SVECTOR(2)); DROP DATABASE vb_probe;" \
    || { log "FATAL: $EXTENSION installed but SVECTOR does not resolve"; exit 1; }
  "$CLIENT" -uroot --socket="$boot_sock" -e 'SHUTDOWN'
  wait "$pid" || true
  log "initialisation complete"
}

start_server() {
  ensure_dirs
  initialise

  local -a args=(
    --no-defaults
    --basedir="$ROOT_DIR"
    --datadir="$DATA_DIR"
    --socket="$SOCKET"
    --log-error="$LOG_FILE"
    --pid-file=/var/run/vbench/villagesql.pid
    --skip-name-resolve
    --vsql-allow-preview-extensions=ON
  )
  [[ -f "$INIT_SQL" ]] && args+=( --init-file="$INIT_SQL" )
  if [[ -n "${VB_SERVER_ARGS:-}" ]]; then
    for _arg in ${VB_SERVER_ARGS}; do
      [[ "$_arg" == "--no-defaults" ]] && continue
      args+=( "$_arg" )
    done
  fi
  args+=( "$@" )

  local u; u="$(user_args)"; [[ -n "$u" ]] && args+=( "$u" )

  log "exec $MYSQLD ${args[*]}"
  exec "$MYSQLD" "${args[@]}"
}

cmd="${1:-server}"; shift || true
case "$cmd" in
  server) start_server "$@" ;;
  init)   ensure_dirs; initialise ;;
  client) exec "$CLIENT" --socket="$SOCKET" "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
