"""Control surface: profile editing, run launching, live monitoring.

Every handler here refuses unless the server was started with --allow-control.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from . import profiles as profiles_mod
from . import runs as runs_mod
from .api import Api, Response

DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _denied() -> Response:
    return 403, {"error": "control is disabled; start the UI with --allow-control"}


def _guard(api: Api) -> Optional[Response]:
    return None if api.allow_control else _denied()


# -- reference data ----------------------------------------------------------

def list_engines(api: Api, _m, _q, _b=None) -> Response:
    from . import engines as engines_mod
    from orchestrator import docker_ctl

    listed = engines_mod.listing(api.config_dir)
    for engine in listed:
        images = engine.get("images") or {}
        engine["built"] = {
            kind: bool(ref) and docker_ctl.image_exists(ref)
            for kind, ref in images.items()
        }
    return 200, {
        "engines": listed,
        "drivers": engines_mod.available_drivers(),
        "control_enabled": api.allow_control,
    }


def get_engine(api: Api, match, _q, _b=None) -> Response:
    from . import engines as engines_mod
    found = engines_mod.read(api.config_dir, match.group("name"))
    if found is None:
        return 404, {"error": f"no such engine: {match.group('name')}"}
    return 200, found


def validate_engine(api: Api, match, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    from . import engines as engines_mod
    errors, warnings, parsed = engines_mod.validate(
        api.config_dir, match.group("name"), (body or {}).get("text", ""))
    return 200, {"ok": not errors, "errors": errors, "warnings": warnings,
                 "parsed": parsed}


def put_engine(api: Api, match, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    from . import engines as engines_mod
    name = match.group("name")
    text = (body or {}).get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "body must contain non-empty 'text'"}

    ok, errors, warnings = engines_mod.write(api.config_dir, name, text)
    if not ok:
        return 400, {"ok": False, "errors": errors, "warnings": warnings}
    return 200, {"ok": True, "errors": [], "warnings": warnings, "name": name,
                 "next": f"./run-benchmark.sh build --engines {name}"}


def clone_engine(api: Api, _m, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    from . import engines as engines_mod
    spec = body or {}
    text, errors = engines_mod.clone(api.config_dir, str(spec.get("base") or ""),
                                     str(spec.get("name") or ""))
    if errors:
        return 400, {"ok": False, "errors": errors}
    return 200, {"ok": True, "name": spec.get("name"), "text": text}


def list_datasets(api: Api, _m, _q, _b=None) -> Response:
    """Every dataset this framework knows, and whether it is here yet.

    A missing dataset used to be invisible until a run failed on it.
    """
    from harness.datasets import KNOWN_DATASETS

    on_disk: Dict[str, int] = {}
    try:
        for filename in os.listdir(api.datasets_dir):
            if filename.endswith(".hdf5"):
                path = os.path.join(api.datasets_dir, filename)
                try:
                    on_disk[filename[:-len(".hdf5")]] = os.path.getsize(path)
                except OSError:
                    on_disk[filename[:-len(".hdf5")]] = 0
    except OSError:
        pass

    known = []
    for name, facts in sorted(KNOWN_DATASETS.items()):
        generated = name.startswith("dbpedia-openai-")
        known.append({
            "name": name,
            "dim": facts.get("dim"),
            "train": facts.get("train"),
            "test": facts.get("test"),
            "metric": facts.get("metric"),
            "role": facts.get("role"),
            "approx_bytes": facts.get("approx_bytes"),
            "downloaded": name in on_disk,
            "bytes_on_disk": on_disk.get(name),
            # fetch cannot retrieve these; scripts/generate-dataset.sh builds them.
            "generated": generated,
        })

    extra = sorted(set(on_disk) - {d["name"] for d in known})
    return 200, {
        "datasets": known,
        "local_only": [{"name": n, "bytes_on_disk": on_disk[n]} for n in extra],
        "datasets_dir": api.datasets_dir,
    }


def setup_plan(api: Api, _m, _q, _b=None) -> Response:
    from . import setup as setup_mod
    return 200, {**setup_mod.plan(api.root, api.results_dir, api.datasets_dir),
                 "control_enabled": api.allow_control}


def status(api: Api, _m, _q, _b=None) -> Response:
    """Is this machine ready to measure, and if not what is missing.

    The answer used to be spread across `fetch --list`, `docker images` and a
    failed run.
    """
    import shutil

    from harness.datasets import KNOWN_DATASETS
    from orchestrator import docker_ctl
    from orchestrator import engines as engines_mod

    engines = []
    for name, engine in engines_mod.registry().items():
        from orchestrator.config import load_engine
        try:
            images = (load_engine(name).get("image") or {})
        except FileNotFoundError:
            images = {}
        engines.append({
            "name": name,
            "label": engine.label,
            "color": engine.color,
            "group": engine.group,
            "tag": engine.tag,
            "runtime_built": bool(images.get("runtime")) and
                             docker_ctl.image_exists(images["runtime"]),
            "bench_built": bool(images.get("bench")) and
                           docker_ctl.image_exists(images["bench"]),
        })

    present = set()
    try:
        present = {f[:-len(".hdf5")] for f in os.listdir(api.datasets_dir)
                   if f.endswith(".hdf5")}
    except OSError:
        pass

    try:
        usage = shutil.disk_usage(api.root)
        disk = {"free_bytes": usage.free, "total_bytes": usage.total}
    except OSError:
        disk = {}

    active = api.jobs.active()
    return 200, {
        "engines": engines,
        "engines_ready": sum(1 for e in engines if e["bench_built"]),
        "datasets_present": sorted(present),
        "datasets_known": len(KNOWN_DATASETS),
        "disk": disk,
        "runs": len(runs_mod.discover_runs(api.results_dir)),
        "active_job": active,
        "control_enabled": api.allow_control,
    }


def set_run_label(api: Api, match, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    from . import importing as importing_mod

    run_dir = runs_mod.resolve_run_dir(api.results_dir, match.group("run_id"))
    if run_dir is None:
        return 404, {"error": f"no such run: {match.group('run_id')}"}

    spec = body or {}
    label = spec.get("label")
    if label is None:
        return 400, {"error": "body must contain 'label' (empty string clears it)"}
    if str(label).strip():
        importing_mod.set_label(run_dir, str(label), spec.get("source"))
    else:
        importing_mod.clear_label(run_dir)
    return 200, {"ok": True}


def import_run(api: Api, _m, query, _b=None) -> Response:
    """Handled in the server: the body is an archive, not JSON."""
    return 400, {"error": "internal: import is handled by the server"}


def list_profiles(api: Api, _m, _q, _b=None) -> Response:
    return 200, {"profiles": profiles_mod.list_profiles(api.config_dir)}


def get_profile(api: Api, match, _q, _b=None) -> Response:
    profile = profiles_mod.read_profile(api.config_dir, match.group("name"))
    if profile is None:
        return 404, {"error": f"no such profile: {match.group('name')}"}
    return 200, profile


def validate_profile(api: Api, match, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    text = (body or {}).get("text", "")
    errors, warnings, parsed = profiles_mod.validate(match.group("name"), text)
    return 200, {"ok": not errors, "errors": errors, "warnings": warnings,
                 "parsed": parsed}


def put_profile(api: Api, match, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    name = match.group("name")
    text = (body or {}).get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "body must contain non-empty 'text'"}

    ok, errors, warnings = profiles_mod.write_profile(api.config_dir, name, text)
    if not ok:
        return 400, {"ok": False, "errors": errors, "warnings": warnings}
    return 200, {"ok": True, "errors": [], "warnings": warnings, "name": name}


# -- estimate ----------------------------------------------------------------

def estimate(api: Api, _m, _q, body=None) -> Response:
    from orchestrator.cli import (DEFAULT_ENGINES, KNOWN_ENGINES,
                                  estimate_load_hours)
    from orchestrator.config import load_profile, profile_engines

    spec = body or {}
    name = spec.get("profile")
    try:
        profile = load_profile(name) if name else {}
    except FileNotFoundError:
        return 404, {"error": f"no such profile: {name}"}

    # The profile's own engine list sits between the request and KNOWN_ENGINES,
    # exactly as in `cli.cmd_run`, and the dataset line below has always done
    # the same with `profile.datasets`. Not optional tidiness:
    # `estimate_load_hours` returns its breakdown instead of printing it
    # precisely "so the CLI and the web UI cannot report different numbers for
    # the same plan" (its docstring), and `control.js` opens with
    # `plan.engines: []` -- nothing checked. So picking the percona profile
    # here previewed a seven-engine estimate for a job that `cmd_run` now runs
    # on one, which is the same figure being wrong in a new direction.
    # DEFAULT_ENGINES, not KNOWN_ENGINES, for the same reason: `cmd_run`
    # leaves the opt-in engines out of a run that names none.
    engines = [e for e in (spec.get("engines") or profile_engines(profile)
                           or DEFAULT_ENGINES)
               if e in KNOWN_ENGINES]
    datasets = [d for d in (spec.get("datasets") or profile.get("datasets") or [])
                if DATASET_RE.match(str(d))]

    resource_pass = spec.get("resource_pass") or profile.get(
        "default_resource_pass") or "both"
    passes = ["normalized", "tuned"] if resource_pass == "both" else [resource_pass]

    phase = spec.get("phases") or "both"
    phases = ["ann", "ops"] if phase == "both" else [phase]

    return 200, estimate_load_hours(profile, engines, datasets, passes, phases)


# -- jobs --------------------------------------------------------------------

def list_jobs(api: Api, _m, _q, _b=None) -> Response:
    active = api.jobs.active()
    return 200, {
        "jobs": api.jobs.list_jobs(),
        "active": active["id"] if active else None,
        "control_enabled": api.allow_control,
    }


def create_job(api: Api, _m, _q, body=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    spec = body or {}
    # Asked before launching rather than inferred from the message afterwards:
    # a refusal because the machine is busy is a different answer from a
    # refusal because the request was wrong, and the status code should say so.
    conflict = api.jobs.conflict_for(str(spec.get("kind") or "run"))
    if conflict:
        return 409, {"ok": False, "errors": [conflict]}

    job, errors = api.jobs.launch(spec)
    if errors:
        return 400, {"ok": False, "errors": errors}
    return 201, {"ok": True, "job": job}


def get_job(api: Api, match, _q, _b=None) -> Response:
    job = api.jobs.get(match.group("job_id"))
    if job is None:
        return 404, {"error": f"no such job: {match.group('job_id')}"}
    return 200, {"job": job}


def get_job_log(api: Api, match, query, _b=None) -> Response:
    try:
        offset = int((query.get("offset") or ["0"])[0])
    except ValueError:
        offset = 0
    chunk = api.jobs.read_log(match.group("job_id"), offset)
    if chunk is None:
        return 404, {"error": f"no such job: {match.group('job_id')}"}
    return 200, chunk


def stop_job(api: Api, match, _q, _b=None) -> Response:
    denied = _guard(api)
    if denied:
        return denied
    ok, errors = api.jobs.stop(match.group("job_id"))
    return (200 if ok else 400), {"ok": ok, "errors": errors}


ROUTES: List[Tuple[str, Any, Any]] = [
    ("GET", re.compile(r"^/api/engines$"), list_engines),
    ("POST", re.compile(r"^/api/engines/clone$"), clone_engine),
    ("GET", re.compile(r"^/api/engines/(?P<name>[^/]+)$"), get_engine),
    ("PUT", re.compile(r"^/api/engines/(?P<name>[^/]+)$"), put_engine),
    ("POST", re.compile(r"^/api/engines/(?P<name>[^/]+)/validate$"), validate_engine),
    ("GET", re.compile(r"^/api/datasets$"), list_datasets),
    ("GET", re.compile(r"^/api/status$"), status),
    ("PUT", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/label$"), set_run_label),
    ("GET", re.compile(r"^/api/setup$"), setup_plan),
    ("GET", re.compile(r"^/api/profiles$"), list_profiles),
    ("GET", re.compile(r"^/api/profiles/(?P<name>[^/]+)$"), get_profile),
    ("PUT", re.compile(r"^/api/profiles/(?P<name>[^/]+)$"), put_profile),
    ("POST", re.compile(r"^/api/profiles/(?P<name>[^/]+)/validate$"), validate_profile),
    ("POST", re.compile(r"^/api/estimate$"), estimate),
    ("GET", re.compile(r"^/api/jobs$"), list_jobs),
    ("POST", re.compile(r"^/api/jobs$"), create_job),
    ("GET", re.compile(r"^/api/jobs/(?P<job_id>[^/]+)$"), get_job),
    ("GET", re.compile(r"^/api/jobs/(?P<job_id>[^/]+)/log$"), get_job_log),
    ("POST", re.compile(r"^/api/jobs/(?P<job_id>[^/]+)/stop$"), stop_job),
]
