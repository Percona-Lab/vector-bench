#!/usr/bin/env bash
# Reproduce Percona Server's bulk vector-index build crash and collect what a
# server developer needs to read it: the client log, mysqld's error log, the
# core, a backtrace from the core, and the mysqld binary the core came from.
#
#   ./scripts/repro-percona-bulk-build.sh               # 5 attempts
#   ATTEMPTS=10 ./scripts/repro-percona-bulk-build.sh
#   ROWS=60000  ./scripts/repro-percona-bulk-build.sh   # a larger load
#   SKIP_BUILD=1 ./scripts/repro-percona-bulk-build.sh  # image already built
#   NO_CORE=1   ./scripts/repro-percona-bulk-build.sh   # leave core_pattern alone
#   PROFILE=percona-bulk-1m ATTEMPTS=1 ./scripts/repro-percona-bulk-build.sh  # 1M rows
#
# Each attempt is one run of config/profiles/$PROFILE.yml (percona-bulk-repro
# unless PROFILE says otherwise): a fresh container and data directory, load
# the profile's rows (or ROWS of them) with no vector index, then
# ALTER TABLE ... ADD VECTOR KEY ... ALGORITHM=INPLACE. The crash is
# intermittent (one of two identical runs crashed on 2026-10-15), so it repeats.
# docs/repro-percona-bulk-build.md has the whole procedure from a clean machine.
#
# CORES. The framework already asks mysqld for one (--core-file, ulimit core
# unlimited) and copies it out before teardown, but it never touches the
# host's kernel.core_pattern, and on a stock Ubuntu that pipes to apport, so
# no file is written. This script sets it to `core.%p` -- the value the
# framework's capture and scripts/gdb-backtrace.py both expect -- with sudo,
# and puts the old value back on exit. It is host-wide while the script runs.
set -uo pipefail

ATTEMPTS=${ATTEMPTS:-5}
ROWS=${ROWS:-}
PROFILE=${PROFILE:-percona-bulk-repro}
CORE_PATTERN=core.%p
TS=$(date +%Y%m%d-%H%M%S)

cd "$(dirname "$0")/.." || exit 1
OUT=${OUT:-$PWD/results/percona-bulk-repro-$TS}

die() { echo "error: $*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is $USER in the docker group?)"
python3 -c 'import yaml' 2>/dev/null || die "python3 with PyYAML is required (apt-get install python3-yaml)"
[ -e "config/profiles/$PROFILE.yml" ] || die "no profile config/profiles/$PROFILE.yml"

# The dataset and row count come from the profile, so a PROFILE override needs
# nothing else to match it.
read -r DATASET PROFILE_ROWS < <(python3 -c '
import sys, yaml
p = yaml.safe_load(open(sys.argv[1]))
print(p["datasets"][0], p["ops"].get("subset_rows") or "all")
' "config/profiles/$PROFILE.yml") || die "cannot read config/profiles/$PROFILE.yml"

old_pattern=$(cat /proc/sys/kernel/core_pattern)
tmp_profile=
restore_pattern() {
    [ "$(cat /proc/sys/kernel/core_pattern)" = "$old_pattern" ] && return 0
    echo "restoring kernel.core_pattern to: $old_pattern"
    printf '%s\n' "$old_pattern" | sudo tee /proc/sys/kernel/core_pattern >/dev/null && return 0
    echo "WARNING: kernel.core_pattern is still '$(cat /proc/sys/kernel/core_pattern)'. Put it back with:" >&2
    echo "  echo '$old_pattern' | sudo tee /proc/sys/kernel/core_pattern" >&2
    return 1
}
cleanup() {
    [ -n "$tmp_profile" ] && rm -f "$tmp_profile"
    restore_pattern
}
trap cleanup EXIT

# ROWS other than the profile's goes through a copy of the profile rather than
# an edit of the committed one, so the tree stays clean.
if [ -n "$ROWS" ] && [ "$ROWS" != "$PROFILE_ROWS" ]; then
    base=$PROFILE
    PROFILE=$base-$ROWS
    tmp_profile=config/profiles/$PROFILE.yml
    sed "s/subset_rows: .*/subset_rows: $ROWS/;s/^name: .*/name: $PROFILE/" \
        "config/profiles/$base.yml" > "$tmp_profile"
fi

[ -e "datasets/$DATASET.hdf5" ] \
    || ./run-benchmark.sh fetch --datasets "$DATASET" || die "dataset download failed"
if [ -z "${SKIP_BUILD:-}" ]; then
    ./run-benchmark.sh build --engines percona --march x86-64-v3 || die "image build failed"
fi

mkdir -p "$OUT"
{
    echo "framework: $(git rev-parse HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
    grep -E '^\s*tag:' config/engines/percona.yml | head -1
    for image in vector-bench/percona-runtime vector-bench/percona-bench; do
        echo "$image: $(docker image inspect -f '{{.Id}}' "$image" 2>/dev/null)"
    done
    echo "host: $(uname -srm), $(nproc) cpus"
    echo "profile: $PROFILE, dataset: $DATASET, rows: ${ROWS:-$PROFILE_ROWS}, attempts: $ATTEMPTS"
} > "$OUT/versions.txt"
cat "$OUT/versions.txt"

# Only now, after the build: the build can outlast sudo's credential cache, and
# a restore that has to prompt at exit is one that gets missed.
if [ -z "${NO_CORE:-}" ] && [ "$old_pattern" != "$CORE_PATTERN" ]; then
    echo "kernel.core_pattern is '$old_pattern'; setting it to '$CORE_PATTERN' for this script (sudo)"
    printf '%s\n' "$CORE_PATTERN" | sudo tee /proc/sys/kernel/core_pattern >/dev/null \
        || die "could not set kernel.core_pattern; rerun with NO_CORE=1 to go without cores"
fi
echo "core_pattern during the attempts: $(cat /proc/sys/kernel/core_pattern)" | tee -a "$OUT/versions.txt"

crashed=0
for i in $(seq 1 "$ATTEMPTS"); do
    id=percona-bulk-repro-$TS-$i
    echo
    echo "=== attempt $i/$ATTEMPTS: $id"
    ./run-benchmark.sh run --profile "$PROFILE" --engines percona \
        --resource-pass normalized --phases ops --no-report --run-id "$id"
    rc=$?
    run=results/$id
    if grep -rqs "Lost connection" "$run/logs"; then
        crashed=$((crashed + 1))
        verdict="CRASHED (rc=$rc)"
        if [ -n "$(find "$run/cores" -type f 2>/dev/null)" ]; then
            verdict="$verdict, core saved"
            ./run-benchmark.sh lab --engine percona --resource-pass normalized \
                --artifacts "$run" gdb-backtrace.py > "$run/backtrace.txt" 2>&1 \
                && verdict="$verdict, backtrace in $id/backtrace.txt"
        else
            verdict="$verdict, no core"
        fi
    elif grep -rqs "bulk index build:" "$run/logs"; then
        verdict="built: $(grep -rhs 'bulk index build:' "$run/logs" | sed 's/.*bulk index build: //')"
    else
        verdict="other failure (rc=$rc), see $id/logs"
    fi
    echo "attempt $i: $verdict" | tee -a "$OUT/summary.txt"
    cp -r "$run" "$OUT/"
done

restore_pattern

# The core is only readable against the exact binary that wrote it.
if [ "$crashed" -gt 0 ]; then
    cid=$(docker create vector-bench/percona-runtime 2>/dev/null) \
        && docker cp "$cid:/opt/percona/bin/mysqld" "$OUT/mysqld" >/dev/null \
        && echo "mysqld binary: $OUT/mysqld"
    [ -n "${cid:-}" ] && docker rm "$cid" >/dev/null
fi

echo
echo "$crashed of $ATTEMPTS attempts crashed."
echo "everything is in $OUT"
echo "  summary.txt, versions.txt, <attempt>/logs/*ops-m16-post* (client),"
echo "  <attempt>/logs/*server-error* (mysqld), <attempt>/cores/, <attempt>/backtrace.txt"
echo "to send it: tar czf percona-bulk-repro-$TS.tar.gz -C results percona-bulk-repro-$TS"
