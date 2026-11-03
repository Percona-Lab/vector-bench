#!/usr/bin/env bash
#
# Materialise engine source trees at pinned tags into vector-bench/sources/
# and build the per-engine Docker build contexts under work/buildctx/.
#
# The vendor repositories in the parent directory are treated as READ-ONLY.
# We never check out, fetch into, or otherwise modify them:
#
#   * AliSQL has no submodules, so `git archive <tag>` straight out of the
#     local repo is enough and needs no network at all.
#   * MariaDB needs the `libmariadb` submodule, which is NOT populated in the
#     local checkout. `git archive` cannot include submodules, so for MariaDB we
#     clone into sources/ using the local repo as an object `--reference`
#     (fast, almost no network for the main history) and then initialise only
#     the submodules the build actually requires.
#   * Percona Server HAS four submodules and still takes the AliSQL path, not
#     the MariaDB one, because all four are switched off by cmake flags in the
#     build rather than populated. It is also the one engine pinned to a SHA on
#     a branch rather than to a tag -- see prepare_percona.
#
# Usage:
#   scripts/prepare-sources.sh [--engine mariadb|alisql|pgvector|percona|myvector|villagesql|all]
#                              [--force]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ENGINE=all
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need_cmd git tar python3
mkdir -p "$VB_SOURCES" "$VB_BUILDCTX"
assert_not_vendor_repo "$VB_SOURCES"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# record_meta <engine> <tag> <sha> <origin>
record_meta() {
  local engine="$1" tag="$2" sha="$3" origin="$4"
  python3 - "$VB_SOURCES/${engine}.source.json" "$engine" "$tag" "$sha" "$origin" <<'PY'
import json, sys
out, engine, tag, sha, origin = sys.argv[1:6]
with open(out, "w") as fh:
    json.dump({"engine": engine, "tag": tag, "commit": sha, "origin": origin},
              fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
  ok "$engine: tag=$tag commit=${sha:0:12} origin=$origin"
}

# Reuse an existing tarball when the pinned commit has not changed.
tarball_current() {
  local engine="$1" want_sha="$2" tar="$3"
  [[ $FORCE -eq 0 ]] || return 1
  [[ -f "$tar" ]] || return 1
  local meta="$VB_SOURCES/${engine}.source.json"
  [[ -f "$meta" ]] || return 1
  local have
  have="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("commit",""))' "$meta")"
  [[ "$have" == "$want_sha" ]]
}

# ---------------------------------------------------------------------------
# AliSQL — no submodules, pure offline `git archive`
# ---------------------------------------------------------------------------

prepare_alisql() {
  local tag; tag="$(yq_get "$VB_CONFIG/engines/alisql.yml" source.tag AliSQL-8.0.44-2)"
  local upstream; upstream="$(yq_get "$VB_CONFIG/engines/alisql.yml" source.upstream https://github.com/alibaba/AliSQL.git)"
  local tar="$VB_SOURCES/alisql-${tag}.tar"
  local repo="$VB_REPO_ALISQL" origin=local

  if [[ ! -d "$repo/.git" ]] || ! git -C "$repo" rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1; then
    warn "AliSQL tag $tag not available locally; cloning from $upstream"
    repo="$VB_SOURCES/alisql-git"
    origin=upstream
    if [[ ! -d "$repo/.git" ]]; then
      git clone --quiet --depth 1 --branch "$tag" "$upstream" "$repo"
    fi
  fi

  local sha; sha="$(git -C "$repo" rev-parse "refs/tags/$tag^{commit}")"
  if tarball_current alisql "$sha" "$tar"; then
    ok "alisql: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
    stage_context alisql "$tar"; return
  fi

  info "alisql: exporting $tag from $repo (read-only git archive)"
  # storage/duckdb/third_parties is excluded: the DuckDB engine is switched off
  # in the build, and shipping it would add hundreds of MB to the build context.
  git -C "$repo" archive --format=tar --prefix=source/ "$tag" \
      -- . ':(exclude)storage/duckdb/third_parties' > "$tar.tmp"
  mv "$tar.tmp" "$tar"
  record_meta alisql "$tag" "$sha" "$origin"
  stage_context alisql "$tar"
}

# ---------------------------------------------------------------------------
# Percona Server — pinned to a SHA, and deliberately WITHOUT its submodules
# ---------------------------------------------------------------------------
#
# Same shape as prepare_alisql: a pure offline `git archive`, no network when
# the commit is already in the local checkout. Two things differ and both are
# deliberate.
#
# 1. THE PIN IS A COMMIT, NOT A TAG. `source.tag` holds a SHA because the
#    pin is commit 56124b03, the tip of Percona PR #6278 on base `9.7`, and
#    has no release to name. `git archive` takes a commit-ish wherever it
#    takes a tag, so the mechanism is unchanged -- but `refs/tags/$tag` is not
#    how this one is resolved, which is why the verify below is
#    `rev-parse "$sha^{commit}"` against the bare object rather than against a
#    tag ref. Compare SHAs and never dates: the series was rebased onto `9.7`,
#    so its author and committer dates disagree.
#
# 2. NO SUBMODULES, AND THAT IS NOT AN OMISSION. The pin has four of them --
#    percona-xtradb-cluster-galera, storage/rocksdb/rocksdb, extra/coredumper
#    and extra/jwt-cpp -- and MariaDB's branch above exists precisely because
#    `git archive` cannot carry a submodule. Percona does not need the clone
#    treatment because all four are switched OFF in the build instead, which
#    takes four cmake flags (docker/percona/Dockerfile note 2). Cloning to
#    populate submodules we then disable would fetch from three more hosts for
#    code that is never compiled.
prepare_percona() {
  local cfg="$VB_CONFIG/engines/percona.yml"
  local sha; sha="$(yq_get "$cfg" source.tag "")"
  [[ -n "$sha" ]] || die "percona: config/engines/percona.yml has no source.tag"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/percona/percona-server.git)"
  local tar="$VB_SOURCES/percona-${sha}.tar"
  local repo="$VB_REPO_PERCONA" origin=local

  if [[ ! -d "$repo/.git" ]] \
     || ! git -C "$repo" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    warn "percona commit ${sha:0:12} not available locally; cloning from $upstream"
    repo="$VB_SOURCES/percona-git"
    origin=upstream
    if [[ ! -d "$repo/.git" ]]; then
      # No --depth: a shallow clone holds only the default branch's tip, and
      # the pin is an older commit on `9.7` (that branch has moved past it),
      # so the failure would arrive as "fatal: not a tree object" after the
      # download rather than before it.
      git clone --quiet "$upstream" "$repo"
    fi
    git -C "$repo" cat-file -e "${sha}^{commit}" 2>/dev/null \
      || git -C "$repo" fetch --quiet origin "$sha" 2>/dev/null || true
  fi

  # Resolve and re-verify: this is the line that makes the SHA an identity
  # check and not just a string that got passed along.
  local resolved; resolved="$(git -C "$repo" rev-parse "${sha}^{commit}")" \
    || die "percona: commit $sha is not in $repo"
  [[ "$resolved" == "$sha" ]] \
    || die "percona: source.tag $sha resolved to $resolved -- not the pinned commit"

  if tarball_current percona "$resolved" "$tar"; then
    ok "percona: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
    stage_context percona "$tar"; return
  fi

  info "percona: exporting ${sha:0:12} from $repo (read-only git archive)"
  git -C "$repo" archive --format=tar --prefix=source/ "$resolved" > "$tar.tmp"
  mv "$tar.tmp" "$tar"
  record_meta percona "$sha" "$resolved" "$origin"
  stage_context percona "$tar"
}

# ---------------------------------------------------------------------------
# MariaDB — needs the libmariadb submodule, so clone (referencing local objects)
# ---------------------------------------------------------------------------

prepare_mariadb() {
  # Takes the engine name so additional MariaDB versions (mariadb123, ...)
  # reuse this routine. Tarball and checkout names carry the tag, so two
  # versions never collide.
  local engine="${1:-mariadb}"
  local cfg="$VB_CONFIG/engines/$engine.yml"
  local tag; tag="$(yq_get "$cfg" source.tag mariadb-11.8.8)"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/MariaDB/server.git)"
  local submods; submods="$(yq_get "$cfg" source.submodules libmariadb)"
  local tar="$VB_SOURCES/mariadb-${tag}.tar"
  local checkout="$VB_SOURCES/mariadb-${tag}"

  local want_sha=""
  if [[ -d "$VB_REPO_MARIADB/.git" ]] \
     && git -C "$VB_REPO_MARIADB" rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1; then
    want_sha="$(git -C "$VB_REPO_MARIADB" rev-parse "refs/tags/$tag^{commit}")"
  fi

  if [[ -n "$want_sha" ]] && tarball_current "$engine" "$want_sha" "$tar"; then
    ok "$engine: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
    stage_context "$engine" "$tar"; return
  fi

  if [[ ! -d "$checkout/.git" ]]; then
    rm -rf "$checkout"
    local ref_args=()
    if [[ -d "$VB_REPO_MARIADB/.git" ]]; then
      # Reuse the local object store so the clone costs almost no network.
      # --dissociate makes the result independent of the reference afterwards,
      # so the vendor repo can be moved or deleted without breaking us.
      info "$engine: cloning $tag referencing local objects at $VB_REPO_MARIADB"
      ref_args=(--reference-if-able "$VB_REPO_MARIADB" --dissociate)
    else
      info "$engine: cloning $tag from $upstream (no local repo to reference)"
    fi
    git clone --quiet --branch "$tag" --single-branch "${ref_args[@]}" \
        "$upstream" "$checkout"
  else
    info "$engine: reusing existing checkout at $checkout"
  fi

  local sha; sha="$(git -C "$checkout" rev-parse HEAD)"

  # Only the submodules the build actually needs. ColumnStore, RocksDB, wsrep,
  # wolfssl, libmarias3 and duckdb are all disabled in the Dockerfile, so
  # pulling them would waste gigabytes.
  local IFS=,
  for sm in $submods; do
    [[ -n "$sm" ]] || continue
    if [[ -z "$(ls -A "$checkout/$sm" 2>/dev/null)" ]]; then
      info "$engine: initialising submodule $sm"
      git -C "$checkout" submodule update --init --depth 1 -- "$sm" \
        || die "failed to initialise submodule $sm (network required once)"
    fi
  done
  unset IFS

  info "$engine: packing build context tarball"
  tar --create --file "$tar.tmp" \
      --exclude-vcs --exclude='.git' \
      --transform='s,^\.,source,' \
      -C "$checkout" .
  mv "$tar.tmp" "$tar"
  record_meta "$engine" "$tag" "$sha" "$checkout"
  stage_context "$engine" "$tar"
}

# ---------------------------------------------------------------------------
# pgvector — small extension, cloned at its release tag
# ---------------------------------------------------------------------------

prepare_pgvector() {
  local cfg="$VB_CONFIG/engines/pgvector.yml"
  local tag; tag="$(yq_get "$cfg" source.tag v0.8.6)"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/pgvector/pgvector.git)"
  local checkout="$VB_SOURCES/pgvector-${tag}"
  local tar="$VB_SOURCES/pgvector-${tag}.tar"

  if [[ ! -d "$checkout/.git" ]]; then
    rm -rf "$checkout"
    info "pgvector: cloning $tag from $upstream"
    git clone --quiet --depth 1 --branch "$tag" "$upstream" "$checkout"
  fi
  local sha; sha="$(git -C "$checkout" rev-parse HEAD)"

  if tarball_current pgvector "$sha" "$tar"; then
    ok "pgvector: source tarball up to date"
    stage_context pgvector "$tar"; return
  fi

  git -C "$checkout" archive --format=tar --prefix=source/ HEAD > "$tar.tmp"
  mv "$tar.tmp" "$tar"
  record_meta pgvector "$tag" "$sha" "$upstream"
  stage_context pgvector "$tar"
}

# ---------------------------------------------------------------------------
# Build context staging
# ---------------------------------------------------------------------------

# Docker build contexts are per-engine directories holding exactly one source
# tarball plus that engine's auxiliary files. Keeping them separate means a
# MariaDB rebuild never has to ship AliSQL's 1.5 GB of source to the daemon.
stage_context() {
  local engine="$1" tar="$2"
  local ctx="$VB_BUILDCTX/$engine"
  mkdir -p "$ctx"
  # Hardlink when possible (same filesystem, zero copy), else fall back to copy.
  rm -f "$ctx/source.tar"
  ln "$tar" "$ctx/source.tar" 2>/dev/null || cp "$tar" "$ctx/source.tar"
  # Files shared by every engine image (the pinned Python stack), then the
  # engine's own auxiliary files (entrypoint, confs) which may override them.
  find "$VB_DOCKER/_shared" -maxdepth 1 -type f \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  # Auxiliary files (entrypoint, init.sql, confs) come from the engine's own
  # docker/ directory, or from the engine it aliases. Without the alias lookup
  # a second MariaDB version compiles for an hour and then dies on
  #   COPY failed: stat entrypoint-mariadb.sh: file does not exist
  # because docker/mariadb123/ does not exist and never needs to.
  local base; base="$(yq_get "$VB_CONFIG/engines/$engine.yml" alias_of "$engine")"
  find "$VB_DOCKER/$base" -maxdepth 1 -type f ! -name 'Dockerfile' \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  if [[ ! -f "$ctx/source.tar" ]]; then
    die "$engine: build context is missing source.tar at $ctx"
  fi
  ok "$engine: build context ready at $ctx (aux files from docker/$base)"
}

# ---------------------------------------------------------------------------

prepare_mongodb() {
  # The one engine with nothing to compile.
  #
  # Percona Search ships as tarballs, packages and images only, so there is no
  # pinned tag and commit to record and no source.tar to stage. What replaces
  # them is the image digest: a tag floats, and a run built from a floating tag
  # is unreproducible in exactly the dimension this script exists to guarantee.
  # Both digests are resolved here and written into the build context, where
  # the build reads them and the manifest records them.
  local engine=mongodb
  local cfg="$VB_CONFIG/engines/$engine.yml"
  local ctx="$VB_BUILDCTX/$engine"
  local mongot server
  mongot="$(yq_get "$cfg" source.mongot_image "")"
  server="$(yq_get "$cfg" source.server_image "")"
  [[ -n "$mongot" && -n "$server" ]] || die "$engine: source.mongot_image / source.server_image not set in $cfg"

  mkdir -p "$ctx"
  info "$engine: resolving image digests (nothing is compiled for this engine)"

  local digests="$ctx/image-digests.txt"
  : > "$digests"
  local ref name digest
  for ref in "$server" "$mongot"; do
    docker pull --quiet "$ref" >/dev/null || die "$engine: cannot pull $ref"
    digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$ref" 2>/dev/null || true)"
    [[ -n "$digest" ]] || die "$engine: $ref has no repo digest; push it to a registry or pin it by hand"
    name="$([[ "$ref" == "$server" ]] && echo server || echo mongot)"
    printf '%s=%s\n' "$name" "$digest" >> "$digests"
    ok "$engine: $name pinned to $digest"
  done

  find "$VB_DOCKER/_shared" -maxdepth 1 -type f -exec cp {} "$ctx/" \; 2>/dev/null || true
  find "$VB_DOCKER/$engine" -maxdepth 1 -type f ! -name 'Dockerfile' \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  ok "$engine: build context ready at $ctx (no source.tar: nothing is built from source)"
}

# ---------------------------------------------------------------------------

prepare_valkey() {
  # Nothing is fetched or compiled: Percona ships both the server and the
  # search module as packages, and the build installs them from the repository
  # named in the engine config. What this stages is the build context and the
  # package list the Dockerfile checks before installing, so a name that does
  # not resolve fails with the repository's actual contents rather than after
  # the corpus is loaded.
  local engine=valkey
  local cfg="$VB_CONFIG/engines/$engine.yml"
  local ctx="$VB_BUILDCTX/$engine"
  local repo; repo="$(yq_get "$cfg" source.repository "")"
  [[ -n "$repo" ]] || die "$engine: source.repository not set in $cfg"

  mkdir -p "$ctx"
  info "$engine: provenance is the $repo repository (nothing is compiled)"
  printf '%s\n' "$repo" > "$ctx/percona-repository.txt"

  find "$VB_DOCKER/_shared" -maxdepth 1 -type f -exec cp {} "$ctx/" \; 2>/dev/null || true
  find "$VB_DOCKER/$engine" -maxdepth 1 -type f ! -name 'Dockerfile' \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  ok "$engine: build context ready at $ctx (no source.tar: installed from packages)"
}


# ---------------------------------------------------------------------------
# MyVector — TWO pinned sources in one context: the plugin at a commit, and
# the MySQL server tree it is compiled into, at a tag.
# ---------------------------------------------------------------------------
#
# The plugin is built in-tree (`plugin/myvector/`), which is the vendor's own
# plugin recipe, so the context tarball carries the MySQL tree as `source/`
# and the plugin tree beside it as `myvector/`; the Dockerfile copies the
# plugin files into place the way upstream's CI does. Both are shallow
# fetches into sources/ -- nothing here touches a vendor checkout.
#
# The manifest's `engines.myvector.source` is this function's JSON, so it is
# where the plugin commit AND the server tag/commit are recorded.
prepare_myvector() {
  local cfg="$VB_CONFIG/engines/myvector.yml"
  local sha; sha="$(yq_get "$cfg" source.tag "")"
  [[ -n "$sha" ]] || die "myvector: $cfg has no source.tag"
  local release; release="$(yq_get "$cfg" source.release "")"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/askdba/myvector.git)"
  local server_tag; server_tag="$(yq_get "$cfg" source.server_tag "")"
  [[ -n "$server_tag" ]] || die "myvector: $cfg has no source.server_tag"
  local server_up; server_up="$(yq_get "$cfg" source.server_upstream https://github.com/mysql/mysql-server.git)"

  local plugin_repo="$VB_SOURCES/myvector-git"
  if [[ ! -d "$plugin_repo/.git" ]]; then
    info "myvector: fetching plugin commit ${sha:0:12} from $upstream"
    git init --quiet "$plugin_repo"
    git -C "$plugin_repo" remote add origin "$upstream"
  fi
  git -C "$plugin_repo" cat-file -e "${sha}^{commit}" 2>/dev/null \
    || git -C "$plugin_repo" fetch --quiet --depth 1 origin "$sha" \
    || die "myvector: cannot fetch commit $sha from $upstream"
  local resolved; resolved="$(git -C "$plugin_repo" rev-parse "${sha}^{commit}")"
  [[ "$resolved" == "$sha" ]] \
    || die "myvector: source.tag $sha resolved to $resolved -- not the pinned commit"

  local server_repo="$VB_SOURCES/mysql-server-${server_tag}"
  if [[ ! -d "$server_repo/.git" ]]; then
    info "myvector: cloning server $server_tag from $server_up (shallow)"
    git clone --quiet --depth 1 --branch "$server_tag" "$server_up" "$server_repo" \
      || die "myvector: cannot clone $server_tag from $server_up"
  fi
  local server_pin; server_pin="$(yq_get "$cfg" source.server_commit "")"
  [[ -n "$server_pin" ]] || die "myvector: $cfg has no source.server_commit"
  local server_sha; server_sha="$(git -C "$server_repo" rev-parse "refs/tags/${server_tag}^{commit}" 2>/dev/null \
                                  || git -C "$server_repo" rev-parse HEAD)"
  # The same refusal the plugin commit gets: a tag that now names another
  # commit is a different server tree under the pinned label.
  [[ "$server_sha" == "$server_pin" ]] \
    || die "myvector: server tag $server_tag resolved to $server_sha -- not the pinned $server_pin"

  local tar="$VB_SOURCES/myvector-${sha}-${server_tag}.tar"
  if [[ $FORCE -eq 0 && -f "$tar" ]]; then
    ok "myvector: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
  else
    info "myvector: exporting server $server_tag and plugin ${sha:0:12}"
    git -C "$server_repo" archive --format=tar --prefix=source/ HEAD > "$tar.tmp"
    git -C "$plugin_repo" archive --format=tar --prefix=myvector/ "$resolved" > "$tar.plugin"
    tar --concatenate --file="$tar.tmp" "$tar.plugin"
    rm -f "$tar.plugin"
    mv "$tar.tmp" "$tar"
  fi

  python3 - "$VB_SOURCES/myvector.source.json" "$sha" "$resolved" "$release" \
           "$upstream" "$server_tag" "$server_sha" "$server_up" <<'PY'
import json, sys
out, tag, commit, release, origin, server_tag, server_commit, server_origin = sys.argv[1:9]
with open(out, "w") as fh:
    json.dump({"engine": "myvector", "tag": tag, "commit": commit,
               "release": release, "origin": origin,
               "server_tag": server_tag, "server_commit": server_commit,
               "server_origin": server_origin},
              fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
  ok "myvector: plugin ${release} commit=${resolved:0:12}, server ${server_tag} commit=${server_sha:0:12}"
  stage_context myvector "$tar"
}

# ---------------------------------------------------------------------------
# VillageSQL -- TWO pinned sources in one context: the vsql_vector extension
# at a commit, and the villagesql-server tree it is compiled against, at a tag.
# ---------------------------------------------------------------------------
#
# MyVector's shape (prepare_myvector), and for the same reason: the extension
# is built against the server's own build tree, so the context tarball carries
# the server as `source/` and the extension beside it as `vsql-vector/`. The
# extension is pinned to a SHA on `main` because the HNSW index it is being
# measured for is not in any vsql-vector release (its only tag, 0.0.1,
# predates it). Both are shallow fetches into sources/ -- nothing here touches
# a vendor checkout. The manifest's `engines.villagesql.source` is this
# function's JSON, so the server tag and commit are recorded beside the
# extension commit.
prepare_villagesql() {
  local cfg="$VB_CONFIG/engines/villagesql.yml"
  local sha; sha="$(yq_get "$cfg" source.tag "")"
  [[ -n "$sha" ]] || die "villagesql: $cfg has no source.tag"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/villagesql/vsql-vector.git)"
  local server_tag; server_tag="$(yq_get "$cfg" source.server_tag "")"
  [[ -n "$server_tag" ]] || die "villagesql: $cfg has no source.server_tag"
  local server_pin; server_pin="$(yq_get "$cfg" source.server_commit "")"
  [[ -n "$server_pin" ]] || die "villagesql: $cfg has no source.server_commit"
  local server_up; server_up="$(yq_get "$cfg" source.server_upstream https://github.com/villagesql/villagesql-server.git)"

  local ext_repo="$VB_SOURCES/vsql-vector-git"
  if [[ ! -d "$ext_repo/.git" ]]; then
    info "villagesql: fetching vsql-vector commit ${sha:0:12} from $upstream"
    git init --quiet "$ext_repo"
    git -C "$ext_repo" remote add origin "$upstream"
  fi
  git -C "$ext_repo" cat-file -e "${sha}^{commit}" 2>/dev/null \
    || git -C "$ext_repo" fetch --quiet --depth 1 origin "$sha" \
    || die "villagesql: cannot fetch commit $sha from $upstream"
  local resolved; resolved="$(git -C "$ext_repo" rev-parse "${sha}^{commit}")"
  [[ "$resolved" == "$sha" ]] \
    || die "villagesql: source.tag $sha resolved to $resolved -- not the pinned commit"

  # The tag contains a slash, so the checkout name flattens it.
  local server_repo="$VB_SOURCES/villagesql-server-${server_tag//\//-}"
  if [[ ! -d "$server_repo/.git" ]]; then
    info "villagesql: cloning server $server_tag from $server_up (shallow)"
    git clone --quiet --depth 1 --branch "$server_tag" "$server_up" "$server_repo" \
      || die "villagesql: cannot clone $server_tag from $server_up"
  fi
  local server_sha; server_sha="$(git -C "$server_repo" rev-parse "refs/tags/${server_tag}^{commit}" 2>/dev/null \
                                  || git -C "$server_repo" rev-parse HEAD)"
  [[ "$server_sha" == "$server_pin" ]] \
    || die "villagesql: server tag $server_tag resolved to $server_sha -- not the pinned $server_pin"

  local tar="$VB_SOURCES/villagesql-${sha}-${server_sha}.tar"
  if [[ $FORCE -eq 0 && -f "$tar" ]]; then
    ok "villagesql: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
  else
    info "villagesql: exporting server $server_tag and vsql-vector ${sha:0:12}"
    git -C "$server_repo" archive --format=tar --prefix=source/ "$server_sha" > "$tar.tmp"
    git -C "$ext_repo" archive --format=tar --prefix=vsql-vector/ "$resolved" > "$tar.ext"
    tar --concatenate --file="$tar.tmp" "$tar.ext"
    rm -f "$tar.ext"
    mv "$tar.tmp" "$tar"
  fi

  python3 - "$VB_SOURCES/villagesql.source.json" "$sha" "$resolved" \
           "$upstream" "$server_tag" "$server_sha" "$server_up" <<'PY'
import json, sys
out, tag, commit, origin, server_tag, server_commit, server_origin = sys.argv[1:8]
with open(out, "w") as fh:
    json.dump({"engine": "villagesql", "tag": tag, "commit": commit,
               "origin": origin, "server_tag": server_tag,
               "server_commit": server_commit, "server_origin": server_origin},
              fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
  ok "villagesql: vsql-vector commit=${resolved:0:12}, server ${server_tag} commit=${server_sha:0:12}"
  stage_context villagesql "$tar"
}

case "$ENGINE" in
  all)        prepare_mariadb mariadb; prepare_alisql; prepare_pgvector ;;
  percona)    prepare_percona ;;
  myvector)   prepare_myvector ;;
  villagesql) prepare_villagesql ;;
  mariadb)    prepare_mariadb mariadb ;;
  mariadb123) prepare_mariadb mariadb123 ;;
  alisql)     prepare_alisql ;;
  pgvector)   prepare_pgvector ;;
  mongodb)    prepare_mongodb ;;
  valkey)     prepare_valkey ;;
  *) die "unknown engine: $ENGINE (expected mariadb|mariadb123|alisql|pgvector|mongodb|valkey|percona|myvector|villagesql|all)" ;;
esac

ok "sources prepared"
