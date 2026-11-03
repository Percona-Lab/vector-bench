#!/usr/bin/env bash
#
# Build the vector-bench engine images from the pinned sources.
#
# Each engine produces two images:
#   <engine>-runtime  the server alone, for manual use (docs/03-running-manually.md)
#   <engine>-bench    runtime + the ann-benchmarks Python stack
#
# Usage:
#   scripts/build-images.sh [--engine mariadb|mariadb123|alisql|pgvector|webui|all]
#                           [--target runtime|bench|all]
#                           [--march x86-64-v3|native|...]
#                           [--jobs N] [--no-cache] [--pull]
#
# --march is the flag that decides which SIMD path the distance kernels take.
# Every engine gets the SAME value, because otherwise the benchmark compares
# compiler flags rather than implementations. Use `native` only when the build
# host and the benchmark host are the same machine.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ENGINE=all
TARGET=all
MARCH=""
JOBS=0
EXTRA_BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    --target=*) TARGET="${1#*=}"; shift ;;
    --march) MARCH="$2"; shift 2 ;;
    --march=*) MARCH="${1#*=}"; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --no-cache) EXTRA_BUILD_ARGS+=(--no-cache); shift ;;
    --pull) EXTRA_BUILD_ARGS+=(--pull); shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need_docker
need_cmd python3

# BuildKit is faster and gives better output, but it needs the buildx component,
# which plenty of hosts do not have. The Dockerfiles deliberately avoid
# BuildKit-only syntax so they build either way; pick whichever is available
# rather than failing on a machine that only has the classic builder.
if docker buildx version >/dev/null 2>&1; then
  export DOCKER_BUILDKIT=1
  info "using BuildKit (buildx present)"
else
  export DOCKER_BUILDKIT=0
  warn "buildx not found; using the classic builder (slower rebuilds, same result)"
fi

# ---------------------------------------------------------------------------

# Read the space-separated cmake flag list out of an engine config.
cmake_flags_for() {
  python3 - "$VB_CONFIG/engines/$1.yml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(" ".join(cfg.get("build", {}).get("cmake_flags", []) or []))
PY
}

# Read a space-separated list out of an engine config (yq_get cannot do lists).
list_for() {
  python3 - "$VB_CONFIG/engines/$1.yml" "$2" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
node = cfg
for part in sys.argv[2].split("."):
    node = (node or {}).get(part)
print(" ".join(node or []))
PY
}

build_engine() {
  local engine="$1"
  local cfg="$VB_CONFIG/engines/$engine.yml"
  [[ -f "$cfg" ]] || die "no config for engine '$engine'"

  local ctx="$VB_BUILDCTX/$engine"
  [[ -d "$ctx" ]] \
    || die "build context missing for $engine — run: scripts/prepare-sources.sh --engine $engine"
  # Only engines we compile have a source tarball. Percona Search arrives as
  # published images and Valkey as packages, so requiring source.tar of them
  # failed a build whose context prepare-sources had just reported ready.
  local kind; kind="$(yq_get "$cfg" source.kind "source")"
  if [[ "$kind" == "source" ]]; then
    [[ -f "$ctx/source.tar" ]] \
      || die "build context for $engine has no source.tar — run: scripts/prepare-sources.sh --engine $engine"
  fi

  # Engines we compile carry a git tag. Percona Search is a published image and
  # Valkey is a package set, so they carry a version instead and their images
  # would otherwise all be tagged ":unknown".
  local tag;      tag="$(yq_get "$cfg" source.tag "")"
  [[ -n "$tag" ]] || tag="$(yq_get "$cfg" source.version unknown)"
  local base;     base="$(yq_get "$cfg" image.base ubuntu:24.04)"
  local rt_image; rt_image="$(yq_get "$cfg" image.runtime "vector-bench/${engine}-runtime")"
  local bn_image; bn_image="$(yq_get "$cfg" image.bench   "vector-bench/${engine}-bench")"
  local btype;    btype="$(yq_get "$cfg" build.type RelWithDebInfo)"
  local march;    march="${MARCH:-$(yq_get "$cfg" build.march x86-64-v3)}"
  local cxxextra; cxxextra="$(yq_get "$cfg" build.extra_cxxflags "")"
  local optflags; optflags="$(yq_get "$cfg" build.optflags "")"
  local flags;    flags="$(cmake_flags_for "$engine")"

  local -a bargs=(
    --build-arg "BASE_IMAGE=${base}"
    --build-arg "MARCH=${march}"
    --build-arg "BUILD_TYPE=${btype}"
    --build-arg "CMAKE_FLAGS=${flags}"
    --build-arg "EXTRA_CXXFLAGS=${cxxextra}"
    --build-arg "JOBS=${JOBS}"
  )
  [[ -n "$optflags" ]] && bargs+=(--build-arg "OPTFLAGS=${optflags}")
  # An engine may declare `alias_of` to reuse another engine's Dockerfile and
  # tag build-arg. That is how a second MariaDB version costs one config file
  # instead of a branch in every script.
  local base; base="$(yq_get "$cfg" alias_of "$engine")"
  case "$base" in
    mariadb)  bargs+=(--build-arg "MARIADB_TAG=${tag}") ;;
    alisql)   bargs+=(--build-arg "ALISQL_TAG=${tag}") ;;
    # The value is a SHA rather than a tag name -- Percona is pinned to a
    # commit (the tip of Percona PR #6278 on base `9.7`), not to a release --
    # but it travels by the same `source.tag` key and through the same build
    # arg, because `git archive` and `docker build --build-arg` neither know
    # nor care which it is. docker/percona/Dockerfile declares `ARG PERCONA_TAG` to receive it.
    percona)  bargs+=(--build-arg "PERCONA_TAG=${tag}") ;;
    pgvector) bargs+=(--build-arg "PGVECTOR_TAG=${tag}") ;;
    # The MyVector plugin commit (the value of `source.tag`) plus the MySQL
    # server tag it is compiled into; both land in the image's environment and
    # the manifest's source record. docker/myvector/Dockerfile declares both.
    myvector) bargs+=(--build-arg "MYVECTOR_TAG=${tag}"
                      --build-arg "MYSQL_SERVER_TAG=$(yq_get "$cfg" source.server_tag unknown)") ;;
    # Nothing compiled: the module is copied out of its own published image.
    # The vsql_vector commit (the value of `source.tag`) plus the
    # villagesql-server tag it is compiled against; docker/villagesql/
    # Dockerfile declares both.
    villagesql) bargs+=(--build-arg "VILLAGESQL_TAG=${tag}"
                        --build-arg "VILLAGESQL_SERVER_TAG=$(yq_get "$cfg" source.server_tag unknown)") ;;
    mongodb)  bargs+=(--build-arg "MONGOT_IMAGE=$(yq_get "$cfg" source.mongot_image "")") ;;
    # Nothing compiled: installed from a Percona repository. The package list
    # comes from the config, so a name that does not resolve is one edit away
    # rather than a code change.
    valkey)   bargs+=(
                --build-arg "VALKEY_REPO=$(yq_get "$cfg" source.repository "")"
                --build-arg "VALKEY_PACKAGES=$(list_for "$engine" source.packages)"
              ) ;;
    *) die "engine '$engine' has no known build-arg (alias_of=$base)" ;;
  esac

  local -a targets=()
  case "$TARGET" in
    all)     targets=(runtime bench) ;;
    runtime) targets=(runtime) ;;
    bench)   targets=(bench) ;;
    *) die "unknown target: $TARGET" ;;
  esac

  local t img
  for t in "${targets[@]}"; do
    case "$t" in
      runtime) img="$rt_image" ;;
      bench)   img="$bn_image" ;;
    esac
    info "building ${img}:${tag} (target=$t, march=$march, base=$base)"
    local start; start=$(date +%s)
    docker build \
      --target "$t" \
      -t "${img}:${tag}" -t "${img}:latest" \
      -f "$VB_DOCKER/$base/Dockerfile" \
      "${bargs[@]}" "${EXTRA_BUILD_ARGS[@]}" \
      "$ctx" \
      || die "build failed for $engine target $t"
    ok "built ${img}:${tag} in $(( $(date +%s) - start ))s"
  done

  # Record what was actually built, for the run manifest.
  python3 - "$VB_SOURCES/${engine}.image.json" "$engine" "$tag" "$march" "$btype" \
           "$(image_digest "${rt_image}:${tag}")" "$(image_digest "${bn_image}:${tag}" 2>/dev/null || echo unbuilt)" <<'PY'
import json, sys
out, engine, tag, march, btype, rt, bn = sys.argv[1:8]
json.dump({"engine": engine, "tag": tag, "march": march, "build_type": btype,
           "runtime_image_id": rt, "bench_image_id": bn},
          open(out, "w"), indent=2, sort_keys=True)
open(out, "a").write("\n")
PY
}

build_webui() {
  # Not an engine: no source tree, no -march, no config/engines entry. Built
  # from docker/webui with the repo excluded, since the repo is bind-mounted at
  # run time rather than copied in.
  info "building vector-bench/webui"
  docker build --tag vector-bench/webui:latest "$VB_DOCKER/webui"
  ok "vector-bench/webui:latest"
}

case "$ENGINE" in
  # `all` stays the three baseline engines. Extra versions such as mariadb123
  # are opt-in, because each is another hour of compiling that nobody asked for
  # by typing "all".
  all)        build_engine mariadb; build_engine alisql; build_engine pgvector ;;
  webui)      build_webui ;;
  mariadb123) build_engine mariadb123 ;;
  # Nothing is compiled: the runtime image is the published PSMDB image plus
  # mongot copied out of its own image, both pinned by digest.
  mongodb)    build_engine mongodb ;;
  # Installed from Percona's valkey repository; nothing is compiled.
  valkey)     build_engine valkey ;;
  # Percona SERVER, compiled from source like mariadb and alisql and not from
  # a package like the two above. Opt-in rather than part of `all` for the
  # same reason mariadb123 is: it is another full MySQL build that nobody
  # asked for by typing "all".
  percona)    build_engine percona ;;
  # MySQL 9.7 compiled from source with the MyVector plugin in-tree. Opt-in
  # for the same reason percona is: another full MySQL build.
  myvector)   build_engine myvector ;;
  # VillageSQL compiled from source plus the vsql_vector extension. Opt-in
  # for the same reason: another full MySQL-sized build.
  villagesql) build_engine villagesql ;;
  mariadb|alisql|pgvector) build_engine "$ENGINE" ;;
  *) die "unknown engine: $ENGINE" ;;
esac

ok "images built"
