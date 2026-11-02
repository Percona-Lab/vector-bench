"""Profile listing, reading, validation and writing.

Only config/profiles/*.yml is writable. Resource passes and engine definitions
encode the fairness invariants of the comparison and stay in git.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

REQUIRED_KEYS = ("name", "datasets")
# Not a style list: a key missing from here is reported to the operator as
# "unrecognised", so this set is a claim about what the orchestrator reads.
#
# `engines` was absent from it for as long as `config/profiles/percona.yml`
# declared `engines: [percona]`, and so the one validator in the tree that
# looks at profile keys called the one profile whose key was correct
# unrecognised -- a warning, in the profile editor, where nobody running
# `./run-benchmark.sh run --profile percona` would ever see it, while
# `cli.cmd_run` ignored the key outright and measured all seven engines
# (PS/percona-20261020-140151). Now that `config.profile_engines` reads it,
# leaving it out would warn about a key that decides what a run measures.
KNOWN_TOP_LEVEL = {
    "name", "description", "datasets", "engines", "k", "runs", "ann", "ops",
    "resources", "default_resource_pass",
    # Marks a profile a report is published from; such a profile must carry
    # `runs: 2` or more (tests/test_config.py, TestPublishableProfiles).
    "publishable",
}


def profiles_dir(config_dir: str) -> str:
    return os.path.join(config_dir, "profiles")


def profile_path(config_dir: str, name: str) -> Optional[str]:
    if not NAME_RE.match(name or ""):
        return None
    return os.path.join(profiles_dir(config_dir), f"{name}.yml")


def list_profiles(config_dir: str) -> List[Dict[str, Any]]:
    directory = profiles_dir(config_dir)
    out: List[Dict[str, Any]] = []
    try:
        names = sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".yml"))
    except OSError:
        return out

    for name in names:
        try:
            with open(os.path.join(directory, f"{name}.yml")) as fh:
                body = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            body = {}
        out.append({
            "name": name,
            "description": body.get("description"),
            "datasets": body.get("datasets") or [],
            "default_resource_pass": body.get("default_resource_pass"),
            "ann_enabled": bool((body.get("ann") or {}).get("enabled", True)),
            "ops_enabled": bool((body.get("ops") or {}).get("enabled", True)),
            "m_values": (body.get("ann") or {}).get("m_values") or [],
        })
    return out


def read_profile(config_dir: str, name: str) -> Optional[Dict[str, Any]]:
    path = profile_path(config_dir, name)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        parsed, errors = {}, [str(exc)]
    else:
        errors = []
    return {"name": name, "text": text, "parsed": parsed, "errors": errors}


def validate(name: str, text: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Return (errors, warnings, parsed)."""
    errors: List[str] = []
    warnings: List[str] = []

    if not NAME_RE.match(name or ""):
        errors.append("profile name must match [a-z0-9][a-z0-9_-]* and be <= 64 chars")

    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return errors + [f"YAML error: {exc}"], warnings, {}

    if not isinstance(parsed, dict):
        return errors + ["profile must be a YAML mapping"], warnings, {}

    for key in REQUIRED_KEYS:
        if key not in parsed:
            errors.append(f"missing required key: {key}")

    if parsed.get("name") and parsed["name"] != name:
        errors.append(f"name in file is {parsed['name']!r}, expected {name!r}")

    datasets = parsed.get("datasets")
    if datasets is not None and (not isinstance(datasets, list) or not datasets):
        errors.append("datasets must be a non-empty list")

    # Borrowed rather than restated, so the editor's complaint and `cmd_run`'s
    # refusal cannot drift into saying different things about one file. An
    # ERROR and not a warning: `write_profile` refuses on errors only, and this
    # editor is the realistic way `engines: [percona]` becomes `engines: []` --
    # which would hand all seven engines the Percona-only grid without a word.
    from orchestrator.config import profile_engines_problem
    problem = profile_engines_problem(parsed)
    if problem:
        errors.append(problem)

    for section in ("ann", "ops"):
        block = parsed.get(section)
        if block is not None and not isinstance(block, dict):
            errors.append(f"{section} must be a mapping")

    for key in parsed:
        if key not in KNOWN_TOP_LEVEL:
            warnings.append(f"unrecognised top-level key: {key}")

    resources = parsed.get("resources")
    if isinstance(resources, dict):
        empty = [k for k, v in resources.items() if v == {}]
        if empty:
            warnings.append(
                "resources keys set to {} do not clear the inherited value; "
                f"use null instead: {', '.join(sorted(empty))}")

    ann_m = ((parsed.get("ann") or {}).get("m_values") or [])
    if isinstance(ann_m, list) and len(ann_m) > 4:
        warnings.append(
            f"{len(ann_m)} M values: every M reloads the whole corpus, "
            "so ingest time scales with this list")

    return errors, warnings, parsed


def write_profile(config_dir: str, name: str, text: str) -> Tuple[bool, List[str], List[str]]:
    errors, warnings, _parsed = validate(name, text)
    if errors:
        return False, errors, warnings

    path = profile_path(config_dir, name)
    if path is None:
        return False, ["invalid profile name"], warnings

    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        os.replace(temporary, path)
        _match_owner(directory, path)
    except OSError as exc:
        return False, [f"could not write {path}: {exc}"], warnings

    return True, [], warnings


def _match_owner(reference_dir: str, path: str) -> None:
    """Give a written file the owner of its directory.

    A fallback, not the mechanism: `run-benchmark.sh web` runs the container as
    the invoking user, so this is a no-op there. It matters only when the server
    is started as root by hand, where a file written to a bind mount would
    otherwise be root-owned on the host and undeletable by its owner.
    """
    if os.geteuid() != 0:
        return
    try:
        stat = os.stat(reference_dir)
        os.chown(path, stat.st_uid, stat.st_gid)
    except OSError:
        pass
