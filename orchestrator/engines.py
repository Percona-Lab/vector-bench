"""The engine registry: one place that knows what engines exist.

These facts used to be six dictionaries keyed by engine name, in ann_pass,
ops_pass, cli, charts and render. Adding an engine meant finding all six, and
missing one failed hours into a run rather than at load -- `argument --engine:
invalid choice` after a server had already started, or a recall chart quietly
missing a series.

They are now read from config/engines/*.yml, where the rest of an engine already
lives, which is also what lets a variant be added without editing Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import CONFIG_DIR, load_engine

ENGINES_DIR = os.path.join(CONFIG_DIR, "engines")

#: Fallback chart colours for an engine whose config names none.
FALLBACK_COLORS = ("#ff7f0e", "#8c564b", "#17becf", "#bcbd22", "#7f7f7f")


@dataclass(frozen=True)
class EngineRuntime:
    """What the orchestrator needs to start an engine and talk about it."""

    name: str
    display_name: str
    driver: str
    ann_constructor: str
    port: int
    data_mount: str
    #: Read-only view of the server's data directory for exact index sizing.
    #: None where the index is not a separate on-disk artifact.
    server_data_mount: Optional[str]
    user: str
    password: str
    group: str
    order: int
    color: str
    marker: str
    linestyle: str
    chart_label: str
    label: str
    tag: str
    #: Readiness probe argv, run in the server container until it succeeds.
    probe: Tuple[str, ...]
    #: Absolute path, inside the container, of the file the server writes its
    #: own diagnostics to -- i.e. what the entrypoint gives `--log-error`.
    #:
    #: Set by the four engines started that way -- `mariadb`, `mariadb123`,
    #: `alisql`, `percona` -- whose container stdout is then the entrypoint
    #: banner while everything the server itself says goes into a file in the
    #: data volume that teardown removes. Also set by `mongodb`, which takes
    #: no such flag but redirects `mongot`'s log into that volume with a shell
    #: `>`; there `docker logs` still carries mongod, so the two captures are
    #: complementary rather than one replacing the other.
    #:
    #: None -- the default, and the case for `pgvector` and `valkey` -- keeps
    #: the `docker logs` capture that already carries everything for them.
    #: The ops path copies this file out before teardown either way.
    #:
    #: LAST IN THE DATACLASS AND DEFAULTED, deliberately, and in that order:
    #: Python forbids a field without a default after one that has it, so
    #: giving this a default -- which the config makes right, the key being
    #: genuinely optional -- is what forces it to the end rather than beside
    #: `data_mount` where it reads best.
    error_log: Optional[str] = None
    #: `docker run --ulimit` entries for this engine's server container, as
    #: `((name, value), ...)`. Empty for every engine that declares none, and
    #: `ContainerSpec.ulimits` says what the one live entry is for.
    #:
    #: A TUPLE OF PAIRS AND NOT THE DICT IT READS AS. This class is
    #: `@dataclass(frozen=True)`, so Python generates `__hash__` from its
    #: fields and `hash()` on a registry entry works today; one `Dict` field
    #: makes every engine unhashable, and that would surface at whatever first
    #: put one in a set rather than here. `probe` is a tuple for the same
    #: reason. Callers that want a mapping build one -- `dict(engine.ulimits)`
    #: -- which also keeps them from mutating the registry.
    ulimits: Tuple[Tuple[str, str], ...] = ()
    #: The build every summary figure for this engine is taken from, as
    #: `((axis, value), ...)` with `value` an int, a string or `"ops"` -- the
    #: last meaning "whatever the ops pass builds under this profile and pass",
    #: which `reference_build.resolve_reference_build` turns into a value. A
    #: tuple of pairs for the reason `ulimits` is one. Empty for an engine
    #: that declares none, and `reference_build_problem` refuses that.
    reference_build: Tuple[Tuple[str, Any], ...] = ()

    @property
    def credentials(self) -> Tuple[str, str]:
        return (self.user, self.password)

    @property
    def style(self) -> Dict[str, str]:
        return {"color": self.color, "marker": self.marker,
                "linestyle": self.linestyle, "label": self.chart_label}


def _source_tag(config: Dict[str, Any]) -> str:
    source = config.get("source") or {}
    return str(source.get("tag") or source.get("version") or "")


def _build(name: str, config: Dict[str, Any], fallback_index: int) -> EngineRuntime:
    runtime = config.get("runtime") or {}
    chart = runtime.get("chart") or {}
    credentials = runtime.get("credentials") or {}
    display = config.get("display_name", name)
    label = runtime.get("label") or chart.get("label") or display

    missing = [key for key in ("driver", "ann_constructor", "port", "data_mount")
               if runtime.get(key) in (None, "")]
    if missing:
        raise ValueError(
            f"config/engines/{name}.yml: runtime is missing {', '.join(missing)}")

    return EngineRuntime(
        name=name,
        display_name=display,
        driver=str(runtime["driver"]),
        ann_constructor=str(runtime["ann_constructor"]),
        port=int(runtime["port"]),
        data_mount=str(runtime["data_mount"]),
        server_data_mount=runtime.get("server_data_mount") or None,
        user=str(credentials.get("user", "")),
        password=str(credentials.get("password", "")),
        group=str(runtime.get("group", "extra")),
        order=int(runtime.get("order", 1000)),
        color=str(chart.get("color") or FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]),
        marker=str(chart.get("marker") or "o"),
        linestyle=str(chart.get("linestyle") or "-"),
        chart_label=str(chart.get("label") or label),
        label=str(label),
        tag=_source_tag(config),
        probe=tuple(str(a) for a in (runtime.get("probe") or ())),
        # Not in `missing` above: an engine that logs to stderr declares no
        # error_log and must keep working without one.
        error_log=str(runtime["error_log"]) if runtime.get("error_log") else None,
        # Sorted, so the argv an engine produces does not depend on the order
        # keys happen to appear in its YAML -- a server command line that
        # changes without the configuration changing is a diff nobody can
        # account for. Values are stringified because YAML reads `-1` as an
        # int and `"-1"` as a string and `--ulimit core=-1` is the same flag
        # either way.
        ulimits=tuple(sorted(
            (str(key), str(value))
            for key, value in (runtime.get("ulimits") or {}).items())),
        reference_build=_reference_build(name, runtime.get("reference_build")),
    )


#: The axes a reference build may name: the ones `report/loaders.build_params`
#: reads back off a recall record. An axis outside this set could never match
#: a point, so it is refused here rather than turning a whole row into "—".
REFERENCE_BUILD_AXES = ("m", "ef_construction", "quantization")


def _reference_build(name: str, declared: Any) -> Tuple[Tuple[str, Any], ...]:
    if not declared:
        return ()
    if not isinstance(declared, dict):
        raise ValueError(f"config/engines/{name}.yml: runtime.reference_build "
                         f"must be a mapping, got {declared!r}")
    unknown = sorted(set(declared) - set(REFERENCE_BUILD_AXES))
    if unknown:
        raise ValueError(
            f"config/engines/{name}.yml: runtime.reference_build names "
            f"{', '.join(unknown)}; a reference build can name only "
            f"{', '.join(REFERENCE_BUILD_AXES)}")
    return tuple(sorted(declared.items()))


_cache: Optional[Dict[str, EngineRuntime]] = None


def registry(refresh: bool = False) -> Dict[str, EngineRuntime]:
    """Every engine defined under config/engines/, in presentation order.

    `refresh` re-reads the directory, which the web UI needs after writing a
    new engine config in the same process.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    try:
        names = sorted(f[:-4] for f in os.listdir(ENGINES_DIR) if f.endswith(".yml"))
    except OSError:
        names = []

    built: List[EngineRuntime] = []
    for index, name in enumerate(names):
        try:
            built.append(_build(name, load_engine(name), index))
        except (ValueError, KeyError, TypeError) as exc:
            # A malformed engine must not take the whole registry with it, but
            # it must not be silently absent either.
            print(f"[engines] ignoring {name}: {exc}")

    built.sort(key=lambda e: (e.order, e.name))
    _cache = {engine.name: engine for engine in built}
    return _cache


def known_engines(refresh: bool = False) -> Tuple[str, ...]:
    return tuple(registry(refresh))


def engines_in_group(group: str) -> Tuple[str, ...]:
    return tuple(name for name, engine in registry().items() if engine.group == group)


def get(name: str) -> EngineRuntime:
    try:
        return registry()[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name} "
                         f"(known: {', '.join(known_engines())})") from None


def presentation(name: str) -> Dict[str, Any]:
    """Label, colour, marker and line style, for the run manifest.

    The report reads this from the manifest rather than from config/, because
    the report container mounts report/ and harness/ only.
    """
    try:
        engine = get(name)
    except ValueError:
        return {}
    return {"label": engine.label, "chart_label": engine.chart_label,
            "color": engine.color, "marker": engine.marker,
            "linestyle": engine.linestyle, "order": engine.order}


def as_dict(name: str) -> Dict[str, Any]:
    engine = get(name)
    return {
        "name": engine.name, "display_name": engine.display_name,
        "driver": engine.driver, "ann_constructor": engine.ann_constructor,
        "port": engine.port, "data_mount": engine.data_mount,
        "server_data_mount": engine.server_data_mount,
        "group": engine.group, "order": engine.order,
        "label": engine.label, "chart_label": engine.chart_label,
        "color": engine.color, "tag": engine.tag,
    }
