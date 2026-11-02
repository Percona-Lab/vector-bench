"""Chart generation. Every chart is written as both SVG and PNG.

Design constraints, in order:

* **Readable in both light and dark.** The HTML report is viewed in whatever
  theme the reader uses, so charts use a transparent background and a palette
  with sufficient contrast against either.
* **Colour is not the only channel.** Each engine gets a distinct marker and
  line style as well as a colour, so the charts survive greyscale printing and
  the most common colour-vision deficiencies.
* **Axes state their direction.** "Higher is better" is written on the axis
  rather than assumed, because half these charts invert that (latency, build
  time, index size).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# matplotlib is a container dependency, not a host one. Charts are drawn inside
# a bench image, which carries it; the host is promised nothing beyond python3
# and PyYAML, and importing this module must not quietly break that promise.
# It does get imported on the host: the test suite reaches report.generate for
# its pure-logic helpers, and on a machine set up from the documented
# prerequisites that failed with ModuleNotFoundError seventy-six times.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.ticker import FuncFormatter  # noqa: E402
except ImportError:  # pragma: no cover - every bench image has it
    matplotlib = None
    plt = None
    FuncFormatter = None


def have_matplotlib() -> bool:
    return plt is not None


def _require_matplotlib() -> None:
    """Refuse to draw rather than draw nothing.

    A report that came out with its charts silently missing would be worse than
    one that did not come out, which is the same reasoning that makes the
    generator refuse a run directory with no manifest.
    """
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed, so no chart can be drawn. Charts are "
            "generated inside a bench image which carries it: run "
            "`./run-benchmark.sh report --run-dir ...` rather than invoking "
            "report/generate.py on the host.")

# Engine identity is consistent across every chart in the report. Colour is never
# the only channel: each engine also gets a marker and a line style, so the
# charts survive greyscale printing and the common colour-vision deficiencies.
#
# Why a given engine has a given hue is recorded beside the value itself, in
# config/engines/<name>.yml.
#
# These are defaults. The authority is config/engines/*.yml, which the run copies
# into its manifest as `engines.<name>.presentation`; adopt_presentation() below
# applies it. The table stays because the report container mounts report/ and
# harness/ only -- importing the orchestrator from here crashed generation at the
# end of a twenty-hour run once -- and because a manifest written before that
# field existed still has to render.
STYLE: Dict[str, Dict[str, Any]] = {
    "mariadb":  {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "MariaDB 11.8 (MHNSW)"},
    "mariadb123": {"color": "#5fa8d3", "marker": "s", "linestyle": "--", "label": "MariaDB 12.3 (MHNSW)"},
    "alisql":   {"color": "#d62728", "marker": "s", "linestyle": "--", "label": "AliSQL (VIDX)"},
    "pgvector": {"color": "#2ca02c", "marker": "^", "linestyle": "-.", "label": "PostgreSQL (pgvector)"},
    "mongodb":  {"color": "#9467bd", "marker": "v", "linestyle": "-",  "label": "Percona Search (mongot)"},
    "valkey":   {"color": "#e377c2", "marker": "D", "linestyle": "-",  "label": "Valkey (valkey-search)"},
}


def adopt_presentation(manifest: Dict[str, Any]) -> None:
    """Take engine colours and labels from the run that produced the numbers.

    An engine added after this file was written has its style in its own config
    and therefore in the manifest, so it draws correctly without an edit here.
    """
    for name, info in (manifest.get("engines") or {}).items():
        presentation = (info or {}).get("presentation") or {}
        if not presentation:
            continue
        current = dict(STYLE.get(name, FALLBACK))
        for key, field in (("color", "color"), ("marker", "marker"),
                           ("linestyle", "linestyle"), ("label", "chart_label")):
            value = presentation.get(field)
            if value:
                current[key] = value
        STYLE[name] = current


FALLBACK = {"color": "#7f7f7f", "marker": "D", "linestyle": ":", "label": "unknown"}

GRID = {"alpha": 0.25, "linewidth": 0.6}


def style_for(engine: str) -> Dict[str, Any]:
    return STYLE.get(engine, {**FALLBACK, "label": engine})


# Axes a single engine can be swept over within one run. Two measurements of
# one engine that differ on either are different curves, not repeats of one:
# grouping a chart by engine name alone put several points at every x and drew
# a line through all of them that no configuration ever produced.
# Fallback axes, for records that arrive without a source file -- a report
# rebuilt with --from-records from an archive, for instance.
#
# Listing the axes by hand has now failed three times. It missed storage engine
# and ef_construction, then graph degree, and then quantization: two Percona
# Search curves were drawn as one line and came out as a sawtooth. Each time
# the fix was to add the axis that had just been added to the sweep, and each
# time the next axis was already waiting. So identity comes from the result
# filename now, which ann-benchmarks names after the build arguments and which
# therefore covers axes this code has never heard of.
SERIES_AXES = ("storage_engine", "ef_construction", "m")

# How each fallback axis names itself in a legend. Empty means the value speaks
# for itself, as a storage engine does.
AXIS_LABELS = {"storage_engine": "", "ef_construction": "ef_c=", "m": "M="}

# (engine, configuration, resource pass, iterative scan mode).
#
# This constant is the only place the width is written down, which is what made
# the two widenings this key has had cost one edit here and none at its seven
# call sites -- and why widening the shared primitive beat keying six charts
# locally. Three functions do arithmetic on it and none holds a literal index:
# `churn` and `churn_retention` WIDEN it, the first by one axis of its own and
# the second by the pass and the build mode for `churn_impact` to read back
# off; `churn` and `churn_impact` SLICE it back at this offset. (`charts.churn`
# names its own axis and says why this comment does not -- a prose count in
# `tests/test_t4b_readers.py` pins how often that word may appear across
# `report/*.py`, against a paragraph in a file this change may not touch.)
#
# `tests/test_config.py::test_the_key_width_matches_the_key` enforces that, and
# the rule is worth reading before widening again, because the guard itself has
# been wrong twice. It asserts this constant equals the key's real length; it
# forbids any numeric subscript on `key` except `key[0]` (the engine, read
# positionally on purpose) anywhere in the series-key family; and it requires
# every width-dependent function to name this constant. Both the family and the
# width-dependence are derived from the AST rather than listed -- the listed
# version named two functions and missed the second widener, and the version
# before that forbade the literals `key[:3]` and `key[3]`, which stopped being
# the dangerous ones the moment the key reached width 3.
SERIES_KEY_WIDTH = 4

# Colour stays with the engine so versions remain comparable at a glance; the
# storage engine is carried by the linestyle instead.
STORAGE_LINESTYLES = ("-", "--", ":", "-.")

# Linestyles for the resource pass, used ONLY by `pareto`. Everywhere else the
# linestyle already carries the storage engine (see `series_style`), so it is
# not free -- but `pareto` draws one frontier per engine and never splits by
# storage, which is exactly why it has a linestyle to spend. Without it the two
# passes of one engine come out the same colour, the same marker and the same
# line, distinguishable only by reading the legend, which is the defect
# `series_style`'s own comment records for a second graph degree.
PASS_LINESTYLES = ("-", "--", ":", "-.")

#: `qps_at_recall`'s "whichever pass this summary has", distinct from `None`,
#: which is a real pass key: records that carry no `resource_pass` group under
#: it, and a report rebuilt from an archive predating the field is exactly that
#: case. A `None` default would have made "no pass recorded" and "caller did
#: not say" the same request.
_ANY_PASS = object()


def series_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of one plotted line: the engine, which build it came from,
    which resource pass measured it, and which query strategy it used.

    **What leaving the pass out did, on real data.** Measured on
    `PS/new/full-20261015-215006`, a `--resource-pass both` run of all seven
    engines: the filtered chart for fashion-mnist drew 24 series and EVERY one
    of them had x = [0.01, 0.01, 0.1, 0.1, 0.5, 0.5] -- the normalized point
    and the tuned point at each selectivity, in one series, joined by a line no
    machine ever produced. The same pooling was in `build_cost`, `concurrency`,
    `churn`, `latency_percentiles` and `calibration_curves`, because all six
    group through this function. The two passes are two memory and CPU budgets;
    `pareto` splits on them and says so in `pareto_series_style`, §4's tables
    key on them, `loaders.operating_point_key` keys on them, and
    `generate.py`'s duplicate and inversion detectors name them explicitly
    beside the signature. Every consumer that had to tell two budgets apart
    already did. The charts in the middle were the exception.

    **Composed here rather than folded into `build_signature`, which is the
    established pattern and not a convenience.** `loaders.operating_point_key`
    is `(resource_pass, build_signature(record))` and its docstring records
    why: for the MySQL-family engines nothing else in the signature separates
    the two passes, so each consumer that needs the pass adds it at its own
    call site. The signature is deliberately pass-blind and that is pinned, not
    merely observed -- `tests/test_e14_per_pass_manifest.py`'s
    `test_the_fixture_really_does_collide` asserts that two records differing
    only in `resource_pass` produce ONE signature, so moving the pass into
    `build_signature` fails that test by design.

    Both branches of the signature are blind to the pass, so this is not a
    column-fallback problem. The ann result trees are per pass, so one build
    configuration measured in both produces the same RELATIVE path and
    therefore the same file-branch signature -- which is why
    `latency_percentiles`, the one chart here that plots ann records, pooled
    the passes exactly as the five ops charts did. `generate.py`'s `by_config`
    comment records the same mechanism from the detector's side.

    **`extra.iterative_scan` is here for the same reason and emphatically NOT
    in `build_signature`'s column fallback, and the difference is measured
    rather than stylistic.** It looks like a peer of `quantization`, which the
    fallback does list -- both are index or query properties with no column of
    their own, and the signature's own docstring is about what omitting
    `quantization` cost.

    **The rule that separates them is WHICH records carry the field, not how
    many.** An earlier draft of this comment said `quantization` was None on
    every record of every phase and that is simply false: 6 records in
    `full-20261015-215006` carry it, and they are mixed within one engine
    across the two passes -- mongodb's `index_build`, `none` normalized and
    `scalar` tuned. So "mixed within a unit" is not the discriminator either.

    The discriminator is `loaders.operating_points`. `build_signature` is not
    only a series key: that function joins a calibration sweep to the
    workloads that consumed it through `operating_point_key`, and a record
    enters the join iff its phase is `PHASE_CALIBRATION` (the curve half) or
    it carries `extra.calibration_status` (the chosen half). **An axis is safe
    in the signature iff it never appears on a record that joins**, because a
    field present on one member of a unit and absent on its siblings splits
    them apart. Counted on that bundle:

        quantization     6 carriers,  0 join -- `index_build` only, which is
                         neither phase and carries no status
        iterative_scan  72 carriers, 72 join -- `filtered` only, which IS the
                         join's own downstream half

    That is the test to apply to a third axis, and it is the one this change
    got right for the wrong reason first time.

    What the wrong placement costs, measured: adding `iterative_scan` to the
    fallback takes the operating-point table from 54 rows to 66. The 12 extra
    rows are pgvector's filtered records stranded on a key their own
    calibration curve cannot reach, so each arrives with no recall, no grid
    size and no point count -- twelve half-rows published in a table whose job
    is to say what ef each unit settled on. None of the original 54 is lost,
    so nothing in the suite would have caught it.

    Composed here, the field reaches only the charts, where every function
    filters to one phase first and the cross-phase join never happens.
    """
    from .loaders import build_signature
    # The whole signature, not its last element: the filename branch ends in
    # the build stem, but the column fallback ends in one field, and taking
    # [-1] blindly collapsed every fallback record into a single series.
    #
    # The pass goes AFTER the signature rather than before it, so that within
    # one engine the existing ordering of configurations is a prefix of the new
    # one and the two passes of a configuration sort adjacently. Charts order
    # their series by `str(key)`, so the legend a reader already knows keeps
    # its shape and only gains a split. The scan mode follows the pass for the
    # same reason, and in the order the filtered table prints the two columns.
    return (record.get("engine"), build_signature(record),
            record.get("resource_pass"),
            (record.get("extra") or {}).get("iterative_scan"))


def _config_tokens(record: Dict[str, Any]) -> Optional[List[str]]:
    """The build configuration as ann-benchmarks spelled it in the filename.

    `angular_M_16_quantization_scalar` -> [angular, M, 16, quantization, scalar]
    """
    from .loaders import build_signature
    signature = build_signature(record)
    if signature[0] != "file":
        return None
    return str(signature[-1]).split("_")


def _key_suffixes(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], str]:
    """How one engine's series name the axes that are in the key but not in
    the base label, or the empty string where the run did not vary them.

    `series_key` carries the resource pass and `extra.iterative_scan`, and
    neither branch of `series_labels` can name either: the filename branch
    reads ann-benchmarks' result stem, and the trees are per pass so the stem
    is identical in both and never mentions a query-time session setting at
    all; the fallback branch reads `SERIES_AXES`, which lists build axes. So
    splitting the series without this would have produced two lines with the
    same legend text -- the measurements separated and still unattributable,
    which is half the defect and the less obvious half.

    **For the scan mode that half is the whole point, and both sides of it are
    now measured.** `percona-20261020-140151`, which finished at
    2026-10-20T14:30:54Z, is the first run to put both modes on disk. It is one
    engine, one pass, one M, and both modes at the same selectivity -- the
    exact shape this key exists to separate -- and every one of the four points
    used the vector index:

        mode           column  selectivity  recall@k      QPS   p99 (ms)
        off            tag          0.1000    0.0985   2316.5      0.565
        relaxed_order  tag          0.1000    0.9995    692.9      2.569
        off            ctag         0.1252    0.1770   2263.7      0.575
        relaxed_order  ctag         0.1252    0.9990    151.1     27.989

    So the mode is worth about a factor of ten in recall and costs a factor of
    three to fifteen in throughput, and that trade is the whole content of
    PR #45's axis. What the chart did with it, measured at the point
    `charts.filtered` hands to `plot()`, before the mode joined the key:

        column=tag    1 series, 'PostgreSQL (pgvector)'
                      x = [0.1, 0.1]        y = [0.0985, 0.9995]
        column=ctag   1 series, 'PostgreSQL (pgvector)'
                      x = [0.1252, 0.1252]  y = [0.1770, 0.9990]

    One series per column, both of its points at the same x, so the line is a
    vertical segment spanning a factor of ten in recall under a legend entry
    that names neither end. After, two series per column, one point each, each
    legend naming its mode. This run is the only evidence on disk that the axis
    does anything, and it was the one the chart could not show.

    **An earlier revision of this paragraph said "no archived record reads
    `relaxed_order` at all" and used that to label the second mode's recall as
    an expectation.** The magnitude of the guess was right and the premise was
    never sound: the evidence behind it was a count over
    `full-20261015-215006` alone, written as a universal over every archived
    record, and the run above landed nineteen minutes before the commit
    carrying it. An unscoped universal over a corpus that grows is the wrong
    form for a comment; a figure attached to a named, dated run is not.

    Each axis is named only when the engine actually varied it. That is
    `pareto_series_style`'s rule and it is kept for its reason: on a run that
    varied neither, the label is byte-identical to what every chart drew
    before the key grew, so such a report does not change at all.

    That is also what settles the archived run, for a narrower reason than an
    earlier draft of this comment gave. **#45 added a second VALUE, not the
    field.** `cli._ops_iterative_scan` already pinned `off` explicitly, so all
    72 of pgvector's filtered records in `full-20261015-215006` read `off` and
    not one reads None; the records that read None are the other six engines'
    248 filtered records and every record of every other phase. "Absent on
    every pre-#45 record" was wrong both ways round. What matters is only that
    no engine on that run carries two values, and one value is not more than
    one value, so every label and every series there is untouched by this
    axis.

    `charts.churn` appends its own extra axis and names it under this same
    rule; `churn_impact` names the build mode off the key `churn_retention`
    appends for it.
    """
    passes = {r.get("resource_pass") for r in rows}
    scans = {(r.get("extra") or {}).get("iterative_scan") for r in rows}
    # Assigned per KEY and not accumulated per RECORD: a series holds one
    # record per x, so `+=` over the records spelled the pass once per point
    # -- "PostgreSQL (pgvector) / normalized / normalized / normalized" at
    # three selectivities. Every field read here is part of the key, so two
    # records that share a key compute the same suffix and the assignment is
    # idempotent whichever of them is reached last.
    suffixes: Dict[Tuple[Any, ...], str] = {}
    for r in rows:
        suffix = ""
        if len(passes) > 1:
            suffix += f" / {_pass_label(r.get('resource_pass'))}"
        if len(scans) > 1:
            # Spelled with the record's own field name, and the raw mode
            # beside it, because the filtered and strategy tables print that
            # exact string under an "Iterative scan" column -- a reader
            # carrying a line from the chart to its row has one spelling to
            # match. A series whose records carry no mode while a sibling's do
            # says so rather than going bare, or "the engine's default" and
            # "not recorded" would look like the same line.
            scan = (r.get("extra") or {}).get("iterative_scan")
            suffix += f" / iterative_scan={scan if scan else 'unset'}"
        suffixes[series_key(r)] = suffix
    return suffixes


def series_labels(records: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], str]:
    """Legend text per series, naming only what actually varies.

    An engine swept over one configuration keeps its bare label: naming an axis
    the run did not sweep implies a comparison it did not make.

    Where the records carry a result filename, the varying part is found by
    comparing those filenames token by token within one engine, so a new build
    knob names itself without this code being taught about it. The token before
    the one that differs is its name -- `..._quantization_scalar` against
    `..._quantization_none` gives `quantization=scalar`.
    """
    per_engine: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        per_engine.setdefault(r.get("engine"), []).append(r)

    labels: Dict[Tuple[Any, ...], str] = {}
    for engine, rows in per_engine.items():
        base = style_for(engine)["label"]
        tokenised = {series_key(r): _config_tokens(r) for r in rows}
        usable = [t for t in tokenised.values() if t]
        widths = {len(t) for t in usable}

        if len(usable) == len(tokenised) and len(widths) == 1 and len(usable) > 1:
            differing = [i for i in range(widths.pop())
                         if len({tuple(t)[i] for t in usable}) > 1]
            for key, tokens in tokenised.items():
                parts = [f"{tokens[i - 1]}={tokens[i]}" if i else tokens[i]
                         for i in differing]
                labels[key] = base + (" / " + " / ".join(parts) if parts else "")
            continue

        # No filenames, or filenames that cannot be compared: fall back to the
        # named axes, which is all a --from-records rebuild has.
        varying = [set() for _ in SERIES_AXES]
        for r in rows:
            for i, axis in enumerate(SERIES_AXES):
                varying[i].add(r.get(axis))
        for r in rows:
            label = base
            for i, axis in enumerate(SERIES_AXES):
                value = r.get(axis)
                if value is not None and len(varying[i]) > 1:
                    label += (f" / {AXIS_LABELS[axis]}{value}"
                              if AXIS_LABELS[axis] else f" / {value}")
            labels[series_key(r)] = label

    # After both branches rather than inside either, because the axes
    # `_key_suffixes` names are in the key and in neither branch's source of
    # truth -- the filename tokens above, and `SERIES_AXES` below. Folding it
    # into the branches would have meant writing it twice and re-indenting the
    # one that returns early.
    for rows in per_engine.values():
        for key, suffix in _key_suffixes(rows).items():
            labels[key] += suffix
    return labels


def series_style(record: Dict[str, Any],
                 storages: Optional[List[Optional[str]]] = None,
                 degrees: Optional[List[Optional[int]]] = None) -> Dict[str, Any]:
    """Engine colour and marker, with the linestyle carrying storage engine."""
    style = dict(style_for(record.get("engine")))
    if storages and len(storages) > 1:
        idx = storages.index(record.get("storage_engine"))
        style["linestyle"] = STORAGE_LINESTYLES[idx % len(STORAGE_LINESTYLES)]
    # Colour is the engine and linestyle is the storage engine, so a second
    # graph degree had nothing left to carry it and two AliSQL curves came out
    # identical. Width and transparency are free: the denser graph stays solid
    # and prominent, the sparser one recedes, and which is which is legible
    # without reading the legend.
    if degrees and len(degrees) > 1 and record.get("m") is not None:
        rank = degrees.index(record.get("m"))
        style["alpha"] = 1.0 if rank == len(degrees) - 1 else 0.55
        style["linewidth"] = 2.0 if rank == len(degrees) - 1 else 1.3
    return style


def _grouped(records: List[Dict[str, Any]]
             ) -> Tuple[Dict[Tuple[Any, ...], List[Dict[str, Any]]],
                        Dict[Tuple[Any, ...], str],
                        Dict[str, List[Optional[str]]],
                        Dict[str, List[Optional[int]]]]:
    """Split records into plotted series, with their labels, storage engines
    and graph degrees -- the last two because a style has to be chosen per
    series and both need to be distinguishable within one engine's colour."""
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(series_key(r), []).append(r)
    per_engine: Dict[str, List[Optional[str]]] = {}
    degrees: Dict[str, List[Optional[int]]] = {}
    for key, rows_in in groups.items():
        sample = rows_in[0]
        storages = per_engine.setdefault(key[0], [])
        if sample.get("storage_engine") not in storages:
            storages.append(sample.get("storage_engine"))
        ms = degrees.setdefault(key[0], [])
        if sample.get("m") not in ms:
            ms.append(sample.get("m"))
    for v in list(per_engine.values()) + list(degrees.values()):
        v.sort(key=lambda x: (x is None, x))
    return groups, series_labels(records), per_engine, degrees


# ---------------------------------------------------------------------------
# Figure geometry
#
# `_save` writes every chart with `bbox_inches="tight"`, which grows the SAVED
# image to contain whatever artists hang off the canvas. That is what makes a
# legend a width decision: an artist wider than the figure does not get
# clipped or wrapped, it enlarges the file. Everything in this section exists
# to keep the legend from being the thing that decides how wide a chart is.
# ---------------------------------------------------------------------------

#: Widest a saved chart may be, as a multiple of its own height.
#:
#: **The defect this bounds, measured.** `fig.legend(..., ncol=len(handles))`
#: laid every series on ONE legend row, so the row grew with the series count
#: and `bbox_inches="tight"` grew the PNG to match. On
#: `PS/new/full-20261015-215006`, the seven-engine `--resource-pass both` run
#: of 2026-10-15 -- fashion-mnist is the one corpus all seven engines ran, so
#: each (engine, M, pass) combination is its own legend entry:
#:
#:     churn-fashion-mnist-784-euclidean.png         64,730 x 792   81.7 : 1
#:     concurrency-fashion-mnist-784-euclidean.png   25,943 x 864   30.0 : 1
#:     filtered-tag-fashion-mnist-784-euclidean.png  25,326 x 817   31.0 : 1
#:     filtered-ctag-fashion-mnist-784-euclidean.png 25,326 x 817   31.0 : 1
#:     build-fashion-mnist-784-euclidean.png         25,326 x 792   32.0 : 1
#:     storage-fashion-mnist-784-euclidean.png        7,484 x 826    9.1 : 1
#:     churn-sift-128-euclidean.png                   4,514 x 792    5.7 : 1
#:
#: Seven of that run's fifty charts, and the report's headline corpus took six
#: of the seven. A 64,730-pixel strip is not a chart in any document: placed
#: in a 6.5-inch text column it is 0.08 inches tall.
#:
#: **And past a certain series count it stops being a presentation problem.**
#: Measured against `acaf032`, the revision before this fix, by rendering its
#: own `charts.churn` with seven engines over nine graph degrees -- 63 units,
#: 126 legend entries:
#:
#:     ValueError: Image size of 67288x792 pixels is too large.
#:     It must be less than 2^16 in each direction.
#:
#: raised from `_RendererAgg(int(width), int(height), dpi)` inside `savefig`.
#: The same revision renders 21 series fine at 22,494 x 792, so this is the
#: same defect one sweep further along, not a separate one. The failure lands
#: at REPORT GENERATION, after the measurement work is already spent: the run
#: has done its hours and comes away with no report at all. That is the real
#: severity here, and it is why `_require_matplotlib` further up refuses to
#: draw rather than draw nothing -- it would be a poor trade to protect the
#: report from a missing dependency and then lose it to a legend.
#: `tests/test_chart_aspect_ratio.py::TestTheWideShapeDoesNotDestroyTheReport`
#: pins that shape.
#:
#: **Why 4.0 rather than something nearer 3.** The widest FIXED layout in this
#: module is `build_cost`'s four side-by-side panels, `figsize=(20, 4.6)`,
#: which saves at 3.8:1 -- and the single-engine versions of the very charts
#: listed above measured 1.4:1 to 3.8:1 and are fine. A 3:1 bound would
#: therefore condemn a panel layout doing nothing wrong, and the fix would
#: become "redesign the panels", which is not what went wrong. 4:1 in a
#: 6.5-inch column is 1.6 inches tall, about the floor for a legible axis.
#:
#: **An earlier revision of this paragraph claimed 4.0 was "the first round
#: number above every intrinsic panel layout here". That was false, and the
#: way it was false is the thing to remember.** `latency_percentiles` at
#: THREE engines is one row of three panels, 13.2 x 3.1, which saves at
#: 4.07:1 -- over this bound, in the project's headline
#: MariaDB/AliSQL/pgvector comparison. It was missed because the survey behind
#: the sentence only swept series counts UPWARD, 21 and then 63, and at those
#: counts the same chart is a 3x3 grid at 1.41:1. The violating shape was
#: SMALLER than anything measured. A survey is an input to the number chosen
#: here, not a decoration on it: had that chart been in it, 4.0 would not have
#: been available without also bringing it under. `grid_figsize` now does
#: that, and `tests/test_chart_aspect_ratio.py` sweeps one engine through
#: seven as well as the high counts, so the claim below is checked rather than
#: asserted.
#:
#: Every figsize in this module that varies with the data now passes through
#: `bar_chart_figsize` or `grid_figsize`; the rest are literal constants. That
#: is what makes this a module-wide bound rather than a number referenced in
#: one place, which is how the exception above survived review once.
#:
#: This is a DESIGN bound those helpers honour, not a runtime assertion.
#: `_save` deliberately does not refuse a figure that exceeds it: a wide chart
#: is a bad chart, while a generator that raises at the end of a twenty-hour
#: run loses the report, and this module has already paid that price once for
#: an import. The bound is held by the helpers and checked by the tests.
MAX_ASPECT_RATIO = 4.0

#: Where a wrapped legend's top edge sits, in figure fractions, measured from
#: the bottom of the canvas. Just below it: the axes' own x labels are inside
#: the canvas once `tight_layout()` has run, so a legend anchored a hair under
#: zero clears them without having to reserve space inside the figure.
_WRAPPED_LEGEND_TOP = -0.015

#: Legend padding matplotlib charges per entry, in multiples of the legend's
#: own fontsize: the handle, the gap between handle and text, and the gap to
#: the next column. Read from `rcParams` rather than written as 2.0/0.8/2.0 so
#: the estimate stays honest if a style sheet ever changes them.
_LEGEND_PAD_KEYS = ("legend.handlelength", "legend.handletextpad",
                    "legend.columnspacing")


def bar_chart_figsize(count: int, per_bar: float, minimum: float,
                      height: float) -> Tuple[float, float]:
    """Figure size for a bar chart of `count` bars, capped by the aspect bound.

    Pure arithmetic and public so it is testable on a host with no matplotlib,
    which is the same reason `series_key` and friends are.

    `storage_breakdown` asked for `max(8, 1.5 * len(labels))` inches with no
    ceiling. On the 2026-10-15 run that chart had 33 bars -- seven engines,
    three graph degrees, two resource passes -- so it asked for 49.5 inches
    and got 7,484 x 826 pixels, 9.1:1. Growing a bar chart per bar is right;
    growing it without limit is what turns it into a strip.
    """
    return (min(max(minimum, per_bar * count), MAX_ASPECT_RATIO * height),
            height)


def grid_figsize(columns: int, rows: int, per_column: float, per_row: float,
                 minimum: float = 8.0) -> Tuple[float, float]:
    """Figure size for a grid of panels, raised to meet the aspect bound.

    The DUAL of `bar_chart_figsize`, and the asymmetry is the point rather
    than an inconsistency. A bar chart's width grows with the data and has no
    limit, so growing its height cannot save it -- a 49-inch-wide figure is
    unplaceable at any height -- and the width is what has to yield. A panel
    grid's width is already bounded by its own column cap: `min(3, ...)` at
    4.4 inches a column is 13.2 inches whatever the run holds, which is a
    perfectly placeable figure. Nothing about it is running away; only the
    RATIO is wrong, and the cheap correction is to give the row more height.

    `latency_percentiles` at THREE engines is one row of three panels, 13.2 x
    3.1, which saves at 4.07:1 -- over the bound. Three engines is the
    project's headline MariaDB/AliSQL/pgvector comparison and
    `--engines mariadb,alisql,pgvector` is a first-class invocation, so this
    was not a corner. It needs 3.3 inches of height instead of 3.1, and every
    other engine count clears the bound without a height raise.

    That is not the same as "every other engine count comes out untouched",
    which is what this sentence used to say and which the very next paragraph
    contradicts: ONE engine also moves, from 4.4 inches wide to 8.0, through
    `minimum` below. Two shapes change here, n=3 by height and n=1 by width.

    `minimum` is a FLOOR on the width and it is there for the opposite
    problem. At one engine the grid asked for 4.4 inches -- narrower than any
    other chart in this module, all of which floor at 8 or 9 -- and a figure
    that narrow cannot hold its own furniture: the suptitle "Query latency
    distribution — fashion-mnist-784-euclidean" at 13pt is wider than the
    canvas, so `bbox_inches="tight"` widened the PNG to fit it.
    `latency-gist-960-euclidean.png` and `latency-glove-100-angular.png` in
    the 2026-10-15 run are 837 and 835 pixels on a 660-pixel canvas for that
    reason, and they are single-engine charts.

    **The two helpers are duals in RATIO only, and this one does not bound
    absolute size.** `bar_chart_figsize` caps width outright, so its output is
    always placeable; this one caps nothing -- `grid_figsize(100, 1, 4.4,
    3.1)` is 440 x 110 inches at a perfect 4.00:1, which is past matplotlib's
    own pixel limit. That is latent rather than live: both call sites cap
    their own column count (`min(3, ...)` and a literal list of four), so
    `columns` cannot grow without an edit here. A caller that lets its column
    count follow the data needs a ceiling as well as a ratio.
    """
    width = max(minimum, per_column * columns)
    return (width, max(per_row * rows, width / MAX_ASPECT_RATIO))


def _renderer(fig):
    """A renderer to measure against, without forcing a full draw."""
    canvas = fig.canvas
    if not hasattr(canvas, "get_renderer"):
        canvas.draw()
    return canvas.get_renderer()


def _figure_fraction_y(fig, y_px: float) -> float:
    return y_px / (fig.get_size_inches()[1] * fig.dpi)


def _text_width_px(fig, text: str, fontsize: float) -> float:
    probe = fig.text(0, 0, text, fontsize=fontsize)
    try:
        return probe.get_window_extent(_renderer(fig)).width
    finally:
        probe.remove()


def _legend_columns(fig, labels: Sequence[Any], fontsize: float,
                    width_px: float) -> int:
    """How many legend columns fit across `width_px`.

    Conservative on purpose: every column is sized to the WIDEST label rather
    than packed to the labels it will actually hold. Packing to fit exactly is
    one font substitution away from sticking out of the figure again, and
    sticking out of the figure is the entire defect.
    """
    if not labels:
        return 1
    widest = max(_text_width_px(fig, str(text), fontsize) for text in labels)
    pad_em = sum(matplotlib.rcParams[key] for key in _LEGEND_PAD_KEYS)
    entry_px = widest + pad_em * fontsize * fig.dpi / 72.0
    if entry_px <= 0:
        return len(labels)
    return max(1, min(len(labels), int(width_px / entry_px)))


def _wrap_to_figure(fig, text: str, fontsize: float) -> str:
    """Break a caption into lines that each fit across the figure.

    A caption wider than the canvas widens the PNG for exactly the reason an
    unwrapped legend row did, and the archive has both: with the legend fixed,
    `latency-gist-960-euclidean.png` is still 837 pixels on a 660-pixel canvas
    and `storage-` on a two-engine run is 1,247 on 1,200, because each hangs a
    single-line `fig.text` off the side of a narrow figure.

    Returns `text` UNCHANGED when it already fits, which is every caption on
    every figure wide enough to hold it -- so the charts this never had to
    touch stay byte-identical.
    """
    width_px = fig.get_size_inches()[0] * fig.dpi
    if _text_width_px(fig, text, fontsize) <= width_px:
        return text
    lines: List[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and _text_width_px(fig, candidate, fontsize) > width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _series_legend(fig, handles: Sequence[Any], labels: Sequence[Any],
                   fontsize: float = 10) -> Optional[float]:
    """The one legend for a whole figure, wrapped so it never widens it.

    Returns the figure-fraction y of the legend's bottom edge when the legend
    hangs below the axes -- a negative number -- so a caller that also writes
    text under the axes can clear it, and `None` when it did not.

    **One row for as long as one row fits, which is not laziness.** The
    single-row branch is the pre-fix call verbatim, down to the anchor, so a
    chart whose legend already fitted comes out of this function byte for
    byte as it did before. Only a figure whose legend would have overflowed
    takes the second branch, which is exactly the set of charts that was
    broken. The reason to go to that trouble is that this is a LAYOUT change
    to a measurement instrument: a chart that moved is a chart a reader has
    to re-examine, and the narrower the blast radius the cheaper that is.
    (An earlier revision justified it as "the PNGs are byte-deterministic and
    tests compare them". Nothing compared them -- there are no golden PNGs in
    this repository, and the only byte-for-byte chart comparisons are the
    ones in `tests/test_chart_aspect_ratio.py`, which arrived with this
    change. The property is worth having; that reason for it was circular.)

    **Below the axes rather than wrapped in place above them.** Above, the
    legend grows upward into `suptitle`, which these four charts place at
    y=1.06 to 1.18 and which carries the symbol key a reader needs; every one
    of them would have had to move its title by an amount only this function
    can know. Below, the growth lands in empty space, and `bbox_inches="tight"`
    spends HEIGHT on it -- the same mechanism that used to spend width.
    """
    if not handles:
        return None
    width_px = fig.get_size_inches()[0] * fig.dpi
    columns = _legend_columns(fig, labels, fontsize, width_px)
    if columns >= len(handles):
        fig.legend(handles, labels, frameon=False, fontsize=fontsize,
                   loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.06))
        return None
    legend = fig.legend(handles, labels, frameon=False, fontsize=fontsize,
                        loc="upper center", ncol=columns,
                        bbox_to_anchor=(0.5, _WRAPPED_LEGEND_TOP))
    return _figure_fraction_y(
        fig, legend.get_window_extent(_renderer(fig)).y0)


def _below(floor: float, reserved: Optional[float],
           gap: float = 0.02) -> float:
    """Figure-fraction y for text that must sit under everything already there.

    `floor` is where the text went before anything could be below it, and is
    returned unchanged when nothing is -- which keeps the charts that were
    already fine byte-identical.
    """
    if reserved is None:
        return floor
    return min(floor, reserved - gap)


def _new_axes(title: str, xlabel: str, ylabel: str,
              figsize: Tuple[float, float] = (9, 5.5)):
    # Every chart starts here, so this is the one place that has to check.
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, **GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _save(fig, out_dir: str, stem: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for ext in ("svg", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, format=ext, bbox_inches="tight",
                    dpi=150 if ext == "png" else None, transparent=True)
        paths[ext] = path
    plt.close(fig)
    return paths


def _bytes_formatter(value: float, _pos: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.0f} PiB"


# ---------------------------------------------------------------------------
# Recall / QPS
# ---------------------------------------------------------------------------

def _pass_label(resource_pass: Optional[str]) -> str:
    """How a pass is named in a legend, including when it has no name.

    Delegated, so the legend, §4's headings, the floor chart's stem and
    `described_passes` all spell it the same way -- they spelled it three
    different ways, one of them `None`.

    The reason stated here used to conflate two inputs. A record the loader
    labelled carries the STRING `"unknown"` and never arrives as `None`;
    `None` means a record the loader never touched, which the tables would
    otherwise render as a Python repr. It is the second case this handles.
    """
    from .loaders import pass_or_unknown

    return pass_or_unknown(resource_pass)


def pareto_engine_of(key: Any) -> Any:
    """The engine half of a `pareto` key, which may be a bare engine name."""
    return key[0] if isinstance(key, tuple) else key


def pareto_pass_of(key: Any) -> Optional[str]:
    """The resource pass half of a `pareto` key, or None if it has none."""
    return key[1] if isinstance(key, tuple) and len(key) > 1 else None


def pareto_series_style(key: Any, ordered_passes: Sequence[Any]
                        ) -> Tuple[Dict[str, Any], str]:
    """The style and legend label for one `(engine, pass)` frontier.

    Split out of `pareto` so the decision is testable without matplotlib,
    which lives in the bench images rather than on a host built from the
    documented prerequisites. A test that had to draw could only assert that
    the call returned, which is how two mutants of this logic -- keying by
    engine alone, and not varying the linestyle -- both left the suite green.

    Colour and marker come from the engine, so one engine stays recognisable
    across both of its lines. The pass is added to the label and takes the
    linestyle, and both only when the run measured more than one pass: with a
    single pass the style and label are byte-identical to what this chart drew
    before the key grew. Without the linestyle the two passes of one engine
    come out the same colour, the same marker and the same line -- the defect
    `series_style`'s own comment records for a second graph degree.
    """
    style = dict(style_for(pareto_engine_of(key)))
    label = style["label"]
    resource_pass = pareto_pass_of(key)
    if len(ordered_passes) > 1:
        label += f" / {_pass_label(resource_pass)}"
        style["linestyle"] = PASS_LINESTYLES[
            list(ordered_passes).index(resource_pass) % len(PASS_LINESTYLES)]
    return style, label


def pareto_line_label(label: str, points: List[Dict[str, Any]]) -> str:
    """The legend entry, saying "curve at <build>" when only one was swept.

    A line through one build's ef_search points is that build's recall-
    throughput curve, not a Pareto frontier: nothing else was tried. The
    legend is where a reader of the chart alone learns which one they are
    looking at. Over ALL the engine's points, not the zoomed subset, so the
    zoomed companion does not call a multi-build frontier a curve because
    only one build survived its recall cut.
    """
    from .builds import build_label, build_signature
    signatures = {build_signature(p) for p in points}
    if len(signatures) == 1 and points:
        return f"{label} — curve at {build_label(points[0])}"
    return label


def pareto(records_by_engine: Dict[Any, List[Dict[str, Any]]], dataset: str,
           k: int, out_dir: str, stem: str,
           scatter: bool = True,
           recall_floor: Optional[float] = None) -> Optional[Dict[str, str]]:
    """Recall vs QPS, the headline chart.

    Plots each engine's Pareto frontier as a line and, faintly, every measured
    point behind it. Showing the raw points matters: a frontier drawn through
    three widely-scattered measurements deserves less trust than one drawn
    through thirty tight ones, and hiding the scatter hides that.

    Keyed by `(engine, resource_pass)` rather than by engine. A frontier taken
    across two passes is a frontier of nothing: it picks the faster of two
    machines at every recall and draws a curve no configuration ever produced.
    The colour and marker come from the engine half, so an engine stays
    recognisable across both of its lines, and the pass is added to the legend
    only when the run measured more than one -- which leaves a single-pass
    chart identical to what it drew before, legend included. The idiom is
    `_key_suffixes`', which suffixes `series_labels`' text the same way for the
    same reason -- and this chart keeps its own copy because its key is
    `(engine, resource_pass)` and not a `series_key`, so it never reaches that
    helper.

    That sentence used to credit `churn_impact`, which did name the pass itself
    until the pass joined `series_key` and the shared label took the job over.
    It is the second of two pointers at the same deleted block, the other being
    in `charts.churn`; both rotted in a change whose diff contained neither.

    A plain engine name is still accepted, so a caller that has no pass to
    offer -- `--from-records` over an archive that predates the field -- is
    not required to invent one.
    """
    # `None` is kept in the ordering, not filtered out: `('mariadb', None)`
    # beside `('mariadb', 'tuned')` drew one engine's two frontiers under one
    # label and one linestyle, and `_pass_label` names the unlabelled half
    # instead of leaving an empty suffix.
    #
    # Reachable only from a hand-assembled or third-party records file:
    # `load_annb_results` sets `resource_pass` unconditionally and the value
    # is always truthy, so nothing this pipeline writes produces the state.
    # Kept because the cost is one `key=str` and the failure is silent.
    ordered_passes = sorted({pareto_pass_of(key) for key in records_by_engine},
                            key=str)
    from .loaders import pareto_frontier

    if not any(records_by_engine.values()):
        return None

    title = f"Recall vs throughput — {dataset}"
    if recall_floor is not None:
        # The decision is almost always made in the high-recall region, which a
        # full 0-1 axis compresses into a few pixels. A zoomed companion chart
        # shows that region at a usable scale without hiding the full picture.
        title += f"   (recall ≥ {recall_floor:g})"
    fig, ax = _new_axes(
        title,
        f"Recall@{k}  (higher is better →)",
        "Queries per second  (higher is better ↑)",
    )
    ax.set_yscale("log")
    # Recall lives at the top of its range and the interesting differences are
    # in the last few percent, so a linear axis is the wrong instrument. On
    # 0.69-1.0 linear, MariaDB 12.3 (0.9784-0.9996) occupies the rightmost 7%
    # of the plot and AliSQL the rightmost 10%: both read as truncated rather
    # than as accurate. A logit axis gives 0.99-0.999 the same width as
    # 0.9-0.99, which is how ann-benchmarks plots recall and why.
    ax.set_xscale("logit")

    plotted = False
    for key, points in sorted(records_by_engine.items(), key=lambda kv: str(kv[0])):
        style, label = pareto_series_style(key, ordered_passes)
        label = pareto_line_label(label, points)
        usable = [p for p in points
                  if p.get("recall_at_k") is not None and p.get("qps")]
        if recall_floor is not None:
            usable = [p for p in usable if p["recall_at_k"] >= recall_floor]
        if not usable:
            continue
        plotted = True

        if scatter:
            ax.scatter(
                [p["recall_at_k"] for p in usable], [p["qps"] for p in usable],
                color=style["color"], marker=style["marker"], s=18,
                alpha=0.22, linewidths=0, zorder=2,
            )

        frontier = pareto_frontier(usable)
        if frontier:
            ax.plot(
                [p["recall_at_k"] for p in frontier], [p["qps"] for p in frontier],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2.0, markersize=6,
                label=label, zorder=3,
            )

    if not plotted:
        plt.close(fig)
        return None

    # A logit axis cannot reach 1.0, and the right edge has to leave room for
    # the highest point measured rather than sitting on top of it.
    highest = max(
        (p["recall_at_k"] for points in records_by_engine.values()
         for p in points if p.get("recall_at_k") is not None
         and p.get("qps")
         and (recall_floor is None or p["recall_at_k"] >= recall_floor)),
        default=0.99)
    right = min(0.9999, 1.0 - (1.0 - highest) * 0.5)
    if recall_floor is not None:
        ax.set_xlim(left=recall_floor, right=right)
    else:
        ax.set_xlim(left=max(0.5, ax.get_xlim()[0]), right=right)
    # Ticks the reader recognises, rather than logit's default 1-10^-n labels.
    ticks = [t for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
             if ax.get_xlim()[0] <= t <= right]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Build cost
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PR 5 §5.1 item 6: "show intervals as whiskers on churn and filtered charts".
# ---------------------------------------------------------------------------

#: `extra` key holding a recall interval, written by `stats.ci_fields`.
#: Spelled here rather than imported at module scope for the reason
#: `generate.consistency_checks` gives: `report/` and `harness/` mount
#: separately and a top-level `harness` import from this package has crashed
#: report generation at the end of a long run. `recall_whiskers` imports the
#: reader function in its body; this constant only names a dict key.
RECALL_CI_EXTRA_KEY = "recall_ci95"


def recall_whiskers(rows: Sequence[Dict[str, Any]], xs: Sequence[Any],
                    ys: Sequence[Any], key: str = RECALL_CI_EXTRA_KEY):
    """The `(x, y, yerr)` an `errorbar` call needs, for the rows that have one.

    Returns `None` when no row in the series carries a usable interval, so a
    caller can skip the `errorbar` entirely rather than drawing a series of
    zero-length bars -- which would read as "measured with no uncertainty" on
    a chart where most points simply predate the field.

    `yerr` is asymmetric and clamped at zero. The published mean is rounded to
    six places (`records` stores `round(recall, 6)`) while the interval bounds
    are not, so a point can sit a few units in the twelfth decimal outside its
    own interval; matplotlib raises `ValueError: 'yerr' must not contain
    negative values` on that, which would lose the whole figure at the end of
    a run to a rounding artifact. Clamped rather than widened, so the bar
    never claims more spread than was measured.
    """
    from harness.metrics.stats import Interval

    lower, upper, px, py = [], [], [], []
    for row, x, y in zip(rows, xs, ys):
        if y is None:
            continue
        interval = Interval.from_field((row.get("extra") or {}).get(key))
        if interval is None or not interval.usable:
            continue
        px.append(x)
        py.append(y)
        lower.append(max(0.0, float(y) - interval.low))
        upper.append(max(0.0, interval.high - float(y)))
    if not px:
        return None
    return px, py, [lower, upper]


def draw_recall_whiskers(ax, rows: Sequence[Dict[str, Any]],
                         xs: Sequence[Any], ys: Sequence[Any],
                         color: str) -> bool:
    """Draw the whiskers for one series onto one axis. True if it drew any.

    `zorder=2`, between the near_pass/near_fail band the filtered chart fills
    at `zorder=1` and the markers at `zorder=5`/`6`: a bar under the band is
    invisible and a bar over the "returned fewer than k" ring would hide the
    mark it is there to draw attention to.

    No `label`, so the legend keeps one entry per series. A second handle per
    engine would double the legend's width on a seven-engine run, and the
    footnote under the figure explains the bars once for all of them.
    """
    drawn = recall_whiskers(rows, xs, ys)
    if drawn is None:
        return False
    px, py, yerr = drawn
    ax.errorbar(px, py, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.2,
                capsize=3, capthick=1.2, alpha=0.75, zorder=2)
    return True


#: Added to a chart title when at least one series drew whiskers. Only then:
#: on a run of archived records nothing carries an interval, and a legend
#: explaining bars that are not on the figure is worse than no explanation.
WHISKER_TITLE_NOTE = "bars = 95% bootstrap CI of the recall"


def _is_build_cost_point(record: Dict[str, Any], dataset: str) -> bool:
    """Only the ops pass's index build belongs on the build-cost chart.

    This selection used to be dataset plus `build_wall_s is not None`, which
    was safe for exactly as long as no `recall_qps` record carried a build
    time -- and `loaders.load_annb_results` now reads `build_time` and
    `ingest_seconds` out of every ann-benchmarks result file, so they all do.
    Without this filter the ann path's builds land on an ops chart whose other
    two panels are `peak_rss_bytes` and `index_bytes`: figures measured in a
    different container, under a different memory budget, by a harness that
    does not run there. Nothing would have crashed and the chart would have
    gained three curves.

    That is the second time selecting a chart's records on "has the field"
    rather than "is the measurement" has produced a wrong published figure in
    this one function -- `build_cost`'s own docstring has the first -- which is
    why this is a named predicate with the reason attached and not a clause in
    a comprehension. `_is_concurrency_point` exists for the identical reason.

    The phase is a literal rather than `records.PHASE_BUILD` because
    `report/charts.py` must not import `harness` at module scope: see the note
    above `RECALL_CI_EXTRA_KEY`, and the long run that died of such an import.
    `tests/test_e18_ann_build_timing_reaches_the_records.py` asserts the
    literal against the constant so the two cannot drift.
    """
    return (record.get("phase") == "index_build"
            and record.get("dataset") == dataset
            and record.get("build_wall_s") is not None)


def build_cost(records: List[Dict[str, Any]], dataset: str, out_dir: str,
               stem: str) -> Optional[Dict[str, str]]:
    """Build cost against M: time to a queryable index, the bulk build step on
    its own, peak memory and index size.

    **The first two panels are a fix, and the defect was a published one.**
    This chart used to plot the raw `build_wall_s` of every engine on one axis
    under a "lower is better" arrow. That field is two different quantities:
    `workloads/build.py` writes `effective_build = build_seconds if not
    incremental else load.wall_seconds`, so for pgvector in bulk mode it is
    the `CREATE INDEX` call alone and for MariaDB MHNSW and AliSQL VIDX it is
    the ENTIRE million-row load, because they maintain the graph on every
    INSERT and there is no separable step to time.

    `build.py`'s own module docstring exists to prevent exactly this. It
    records the two figures separately "so the report can say 'pgvector's bulk
    build is N x cheaper than its own incremental build' and 'pgvector's
    incremental build costs X versus MHNSW's Y' -- two honest claims instead
    of one misleading one". The build TABLE honoured that through
    `render._index_build_kind`; this chart read `extra.separable_build`
    nowhere and selected on dataset and non-null alone.

    The gap had also just widened for a reason with nothing to do with index
    building: adding the `ctag` filter column slows the load, which raises the
    incremental engines' `build_wall_s` because it IS the load time, while
    pgvector's bulk figure is untouched -- the index build reads only the
    vector column.

    So the axes now compare like with like:

    * **Time to a queryable index** is `generate.time_to_queryable_index_s`:
      ingest + build for a bulk engine, the single figure for an incremental
      one. Every point on it is the same quantity, which is what makes both of
      the claims above answerable from it -- the first as one engine's two
      build modes, the second as two engines.
    * **Bulk index build step** is the raw `build_wall_s` restricted to
      records whose build WAS separable. An incremental engine has no point
      there, which is the honest rendering of "there is no such step", and no
      number on that panel includes a corpus load.

    A record whose `separable_build` is absent appears on neither: the two
    formulas differ by a whole corpus load and nothing in the record says
    which applies. Those engines are named in the subtitle rather than
    dropped silently -- see `generate.BUILD_KIND_UNRECORDED` for why absence
    has no honest reading here.
    """
    from report.generate import (BUILD_KIND_BULK, BUILD_KIND_UNRECORDED,
                                 build_kind, time_to_queryable_index_s)

    plotted_records = [r for r in records if _is_build_cost_point(r, dataset)]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    def bulk_build_step(record):
        """`build_wall_s`, but only where it is a build STEP.

        Returning None for every other kind is what keeps a corpus load off
        this panel. Reading the field directly here instead would restore the
        defect above in one line, which is why this is a named function with
        this docstring rather than a field name in the table below.
        """
        if build_kind(record) != BUILD_KIND_BULK:
            return None
        return record.get("build_wall_s")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    fig.patch.set_alpha(0.0)
    #: `(value of one record, axis label, tick formatter)`. A CALLABLE and not
    #: a field name: the first two panels are derivations rather than columns,
    #: and a bare field name is the form that let the misleading panel be
    #: written in the first place.
    panels = [
        (time_to_queryable_index_s,
         "Time to a queryable index (s)  (lower is better ↓)", None),
        (bulk_build_step,
         "Bulk index build step (s)  (bulk builds only)", None),
        (lambda r: r.get("peak_rss_bytes"),
         "Peak server memory  (lower is better ↓)", _bytes_formatter),
        (lambda r: r.get("index_bytes"),
         "Index size on disk  (lower is better ↓)", _bytes_formatter),
    ]

    any_data = False
    for ax, (value_of, label, formatter) in zip(axes, panels):
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("M (graph degree)")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if formatter:
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))

        for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
            style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
            points = sorted(
                ((r.get("m"), value_of(r)) for r in rows
                 if r.get("m") is not None and value_of(r) is not None),
                key=lambda p: p[0],
            )
            if not points:
                continue
            any_data = True
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                    label=labels[key])

    if not any_data:
        plt.close(fig)
        return None

    handles, labels = axes[0].get_legend_handles_labels()
    _series_legend(fig, handles, labels)
    unrecorded = sorted({str(r.get("engine")) for r in plotted_records
                         if build_kind(r) == BUILD_KIND_UNRECORDED})
    subtitle = f"Index build cost — {dataset}"
    if unrecorded:
        subtitle += (
            f"\n{len(unrecorded)} engine(s) left off the first two panels "
            f"({', '.join(unrecorded)}): their records do not say whether the "
            f"build was separable, so the build time cannot be placed")
    fig.suptitle(subtitle, fontsize=13, y=1.14)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def _is_concurrency_point(record: Dict[str, Any], dataset: str) -> bool:
    """Only the concurrency sweep belongs on the concurrency chart.

    `clients` and `qps` are recorded by the recall sweep, filtered search and
    churn as well, all at one client. Selecting on those two fields alone put
    every one of them on the x=1 gridline and let filtered search's 13-second
    p99 set the scale of the latency panel.
    """
    return (record.get("phase") == "concurrency"
            and record.get("dataset") == dataset
            and record.get("clients") and record.get("qps"))


#: The two statuses' footnote marks, imported from the renderer rather than
#: declared here so the mark in this chart's legend and the mark in the
#: operating-point table's footnote cannot drift apart -- a dagger on the
#: chart explained by a footnote under a different glyph in the table is worse
#: than no mark at all. `render.py` imports no chart code, so this is the
#: direction with no cycle; it is also where `CLIENT_BOUND_MARK` already
#: lives, and that one IS spelled twice (as a literal in the suptitle below),
#: which is exactly the drift this import avoids.
#: Imported, not re-spelled, for the same reason the marks are: the chart
#: title and the operating-point table's Target column name the same number
#: and must name it the same way. `_fmt_target` is `%g` rather than fixed
#: decimals, and its docstring has the measured reason. PR 2 §2.3.3 writes the
#: title with a literal `0.95`, and `{0.95:g}` is exactly `0.95`, so the
#: default target renders the spec's own string.
#: `FLOOR_MARK, UNCALIBRATED_MARK` stay adjacent and in that order, out of
#: alphabetical sequence, because
#: `tests/test_g2_operating_point_doc.py::test_the_chart_imports_the_marks_rather_than_respelling_them`
#: pins that exact substring -- §2.0's point being that a chart's mark and a
#: table's footnote cannot drift apart. A new name goes after them.
from .render import (FLOOR_MARK, UNCALIBRATED_MARK,  # noqa: E402
                     UNREACHED_MARK, SCALING_CONCURRENCY_NOT_ACHIEVED,
                     _fmt_target)


def _uncalibrated_engines(records: List[Dict[str, Any]], dataset: str
                          ) -> List[str]:
    """Engines plotted on this dataset's concurrency chart with no status.

    An engine qualifies when ANY of its plotted points lacks
    `extra.calibration_status`, not only when all of them do. One engine can
    be half-calibrated: the checkpoint grain is (pass, engine, dataset, phase)
    and within an ops phase (M, build_mode), so a resume after flipping
    `ops.ef_mode` can leave one M of an engine measured at the profile's
    literal ef and the next calibrated. The line drawn through both is then
    partly held to the target and partly not, and the honest treatment of a
    line like that is the same as of one held to it nowhere.
    """
    plotted = [r for r in records if _is_concurrency_point(r, dataset)]
    unheld = {r.get("engine") for r in plotted
              if not (r.get("extra") or {}).get("calibration_status")}
    return sorted(e for e in unheld if e is not None)


def _qps_panel_title(records: List[Dict[str, Any]], dataset: str) -> str:
    """What the concurrency chart's QPS panel is actually showing.

    PR 2 §2.3.3 specifies the title "QPS at recall >= 0.95". The 0.95 in it is
    `DEFAULT_TARGET_RECALL`, not a constant: `--target-recall` is a flag and
    `extra.target_recall` is stamped on every record of a calibrated run, so
    the title is built from the records being plotted and renders the spec's
    literal string whenever the run used the default. Hard-coding 0.95 would
    caption a `--target-recall 0.90` run with a floor none of its engines was
    held to -- the precise failure this title exists to prevent, inverted, and
    the reader has nothing else on the page to catch it with.

    Three cases, because the honest caption differs in each:

    * one target -- the spec's title, with that target's value.
    * none -- `ef_mode: fixed`. No calibration ran, every engine sat at the
      profile's literal ef_search, and what recall that bought differs per
      engine by up to 0.18 on these implementations. Claiming a recall floor
      here would be the chart asserting a comparison the run did not make, so
      it says the opposite.
    * more than one -- a `--from-records` rebuild over records merged from two
      runs. Naming either target would be false about the other half of the
      chart, so both are named and neither is presented as the floor.

    Then qualified once more, because a single target is not the same claim as
    a floor that held. An engine with no `calibration_status` was never put at
    the target, and the first version of this function got that right at RUN
    granularity -- no targets anywhere means no claim -- and wrong at ENGINE
    granularity: one calibrated engine beside one resumed from a fixed-ef pass
    titled the panel "QPS at recall >= 0.95" over a line that had never been
    held to 0.95, with no mark on it and no row in the operating-point table.
    See `_uncalibrated_engines` for how that state is reached without anyone
    hand-editing a file.

    `dataset` narrows the targets as well. `ops.target_recall` is one
    profile-level value (`orchestrator/ops_pass.py` passes a single
    `--target-recall`), so one run cannot vary it across datasets and this is
    unreachable from a single run directory -- but a `--from-records` rebuild
    over two merged runs can, and it put the sift run's target in the glove
    chart's title: measured, two datasets at 0.80 and 0.95 produced "QPS at
    mixed recall targets (0.8, 0.95)" on the glove panel, where every record
    plotted had been held to 0.95.
    """
    from .loaders import calibration_targets

    scoped = [r for r in records if r.get("dataset") == dataset]
    targets = calibration_targets(scoped)
    if not targets:
        return "QPS at a fixed ef_search — recall differs per engine"
    if len(targets) == 1:
        title = f"QPS at recall ≥ {_fmt_target(targets[0])}"
    else:
        title = ("QPS at mixed recall targets ("
                 + ", ".join(_fmt_target(t) for t in targets) + ")")
    if _uncalibrated_engines(records, dataset):
        title += f" — not for every engine ({UNCALIBRATED_MARK})"
    return title


def _calibration_marks(records: List[Dict[str, Any]], dataset: str
                       ) -> Dict[str, Dict[str, Any]]:
    """Per engine: which footnote marks it earned on this dataset, and why.

    Keyed on the ENGINE and not on the plotted series key, because the mark is
    a statement about the engine's floor and ceiling rather than about one
    line. One engine owns several plotted series -- `series_key` separates the
    build configuration, the resource pass and the scan mode, as
    `operating_points` keys the pass separately -- so an engine whose floor
    sits above the target in the tuned pass and not the normalized one earns
    the mark on all of its lines, including the normalized ones that reached
    the target.

    **The footnote does not say WHICH pass or M earned the mark, and that is
    stated rather than left to be discovered.** `recalls` carries both on
    every entry and `_calibration_footnote` unpacks them as `_pass` and `_m`
    and prints neither -- only the engine and its best achieved recall. The
    pass split narrows the consequence without removing it: a reader can now
    see which LINES carry the mark, where before one pooled line carried a
    mark earned in a pass it was half made of. Widening the sentence means
    deciding what it says for an engine daggered in both passes, which is a
    rendering question and not this key's.

    `recalls` carries the recall actually achieved, which §2.3.3 requires
    beside the `target_unreached` mark. It is the calibration sweep's recall at
    the chosen ef -- the largest in the grid for that status, which is the
    engine's best showing and what §2.1.6 says the engine is reported at. The
    concurrency record's own `recall_at_k` is a different measurement and is
    already its own column in the table.

    The dataset filter is load-bearing and is not a tidiness measure. Measured
    with it removed, over one engine daggered on glove and `ok` on sift: the
    sift chart's legend suffix went from `''` to `' †'` and its footnote from
    empty to an assertion that MariaDB cannot be slowed down to the target --
    on a chart of a dataset where that was never measured.

    `uncalibrated` is not a sweep outcome: the engine has plotted points with
    no `calibration_status` at all. It is collected here rather than beside
    the title so the legend, the footnote and the title all come from one
    derivation -- see UNCALIBRATED_MARK for how a run reaches that state.
    """
    from .loaders import operating_points

    marks: Dict[str, Dict[str, Any]] = {}

    def entry_for(engine: str) -> Dict[str, Any]:
        return marks.setdefault(engine, {"statuses": set(), "recalls": []})

    for row in operating_points(records):
        if row.get("dataset") != dataset:
            continue
        engine = row.get("engine")
        status = row.get("status")
        if engine is None or status not in ("floor_above_target",
                                            "target_unreached"):
            continue
        entry = entry_for(engine)
        entry["statuses"].add(status)
        entry["recalls"].append((status, row.get("resource_pass"),
                                 row.get("m"), row.get("recall_at_k")))

    # Only where a target was pursued at all. On `ef_mode: fixed` every engine
    # lacks a status, the title already says the chart is at a fixed ef, and
    # marking all seven as exceptions to a claim nobody made would be noise.
    from .loaders import calibration_targets
    if calibration_targets([r for r in records
                            if r.get("dataset") == dataset]):
        for engine in _uncalibrated_engines(records, dataset):
            entry_for(engine)["statuses"].add("uncalibrated")
    return marks


def _mark_suffix(entry: Optional[Dict[str, Any]]) -> str:
    """The marks one legend label should carry, in footnote order."""
    if not entry:
        return ""
    # Known statuses only, by construction: `_calibration_marks` tests against
    # the two literals and adds `uncalibrated` itself. An unrecognised
    # `calibration_status` -- a future value, or a merged file from newer code
    # -- therefore reaches the table as a raw cell with no mark here and no
    # footnote, under a title that still claims the floor held. The status set
    # is closed in `calibrate.py` (`ok`, `target_unreached`,
    # `floor_above_target`) and the measurement version is what guards the
    # merged-file case, so this is not reachable today; it is the thing to
    # revisit if a fourth status is ever added.
    statuses = entry["statuses"]
    out = ""
    if "uncalibrated" in statuses:
        out += " " + UNCALIBRATED_MARK
    if "floor_above_target" in statuses:
        out += " " + FLOOR_MARK
    if "target_unreached" in statuses:
        out += " " + UNREACHED_MARK
    return out


def _agree(engines: Sequence[str]) -> Dict[str, str]:
    """Verb and pronouns for a footnote written over a LIST of engines.

    Chosen explicitly rather than written singular, because for two of the
    three footnotes multi-engine is the normal case, not the edge: an operator
    who flips `ops.ef_mode` mid-run has every engine already measured carrying
    no status, and `target_unreached` lands on whole families at low M -- the
    tuned-profile notes record all three MySQL-family engines between 0.736
    and 0.818 at M=6, every one of them below a 0.95 target.

    Written singular, the uncalibrated paragraph read, verbatim over two
    engines: "AliSQL (VIDX), PostgreSQL (pgvector) WAS not calibrated: IT
    carries no calibration_status, so IT was measured at ... does not apply to
    IT. ITS throughput ...". An explicit choice is also how
    `_refusal_message` already handles "1 record" against "2 records".
    """
    if len(engines) == 1:
        return {"was": "was", "it": "it", "its": "its", "carries": "carries",
                "is": "is"}
    return {"was": "were", "it": "they", "its": "their", "carries": "carry",
            "is": "are"}


def _calibration_footnote(marks: Dict[str, Dict[str, Any]]) -> str:
    """The footnotes §2.3.3 asks for, each printed only when it applies.

    The `floor_above_target` sentence carries §2.1.6's requirement verbatim --
    "cannot be slowed down to the target" -- because that is the one thing a
    reader of this chart must not have to infer. An engine sitting above the
    comparison recall is not being judged on the same terms as the rest, and a
    dagger with no sentence behind it reads as a minor caveat.

    A third paragraph for the engines that were never calibrated. It comes
    first, because it is the only one of the three that says the title above
    the panel does not describe that line at all.
    """
    floors = sorted(e for e, v in marks.items()
                    if "floor_above_target" in v["statuses"])
    unreached = sorted(e for e, v in marks.items()
                       if "target_unreached" in v["statuses"])
    uncalibrated = sorted(e for e, v in marks.items()
                          if "uncalibrated" in v["statuses"])
    parts: List[str] = []
    if uncalibrated:
        g = _agree(uncalibrated)
        parts.append(
            f"{UNCALIBRATED_MARK} "
            f"{', '.join(style_for(e)['label'] for e in uncalibrated)}"
            f" {g['was']} NOT calibrated: {g['carries']} no"
            f" calibration_status, so the recall floor in the title above does"
            f" not apply. Measured at the profile's literal ef_search,"
            f" {g['its']} throughput is not comparable with the rest at any"
            f" known recall")
    if floors:
        parts.append(
            f"{FLOOR_MARK} {', '.join(style_for(e)['label'] for e in floors)}"
            f" cannot be slowed down to the target: the smallest grid"
            f" ef_search is already above it, so this QPS is measured at a"
            f" HIGHER recall than the rest")
    if unreached:
        shown = []
        for engine in unreached:
            for status, _pass, _m, recall in marks[engine]["recalls"]:
                if status != "target_unreached" or recall is None:
                    continue
                shown.append(f"{style_for(engine)['label']} {recall:.4f}")
        detail = f" (best achieved: {', '.join(shown)})" if shown else ""
        g = _agree(unreached)
        parts.append(
            f"{UNREACHED_MARK} "
            f"{', '.join(style_for(e)['label'] for e in unreached)}"
            f" reached the target at no grid point and {g['is']} shown at"
            f" {g['its']} best achievable recall{detail} — a LOWER recall than"
            f" the rest")
    return "\n".join(parts)


def concurrency(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                stem: str) -> Optional[Dict[str, str]]:
    """QPS, p99 latency and scaling efficiency against client count."""
    plotted_records = [r for r in records if _is_concurrency_point(r, dataset)]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_alpha(0.0)

    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("Concurrent clients")
        ax.grid(True, **GRID)
        ax.set_xscale("log", base=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Queries per second  (higher is better ↑)", fontsize=10)
    axes[1].set_ylabel("p99 latency (ms)  (lower is better ↓)", fontsize=10)
    axes[2].set_ylabel("Scaling efficiency  (1.0 = linear)", fontsize=10)

    # PR 2 §2.3.3. On the QPS panel and not as the figure's title, because it
    # is true of that panel only: the latency and efficiency panels are not
    # throughput at a recall. Built from `records` rather than
    # `plotted_records` so a run whose concurrency workload was skipped for one
    # engine still reports the target the others were held to.
    axes[0].set_title(_qps_panel_title(records, dataset), fontsize=10)
    marks = _calibration_marks(records, dataset)

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["clients"])
        clients = [r["clients"] for r in rows]

        axes[0].plot(clients, [r["qps"] for r in rows], color=style["color"],
                     marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=1.8, markersize=6,
                     label=labels[key] + _mark_suffix(marks.get(key[0])))

        p99 = [(c, r.get("latency_p99_ms")) for c, r in zip(clients, rows)
               if r.get("latency_p99_ms") is not None]
        if p99:
            axes[1].plot([p[0] for p in p99], [p[1] for p in p99],
                         color=style["color"], marker=style["marker"],
                         linestyle=style["linestyle"], linewidth=1.8, markersize=6)

        # The efficiency panel omits a point whose pool never ran the queries
        # its client count implies, and omitting it is the whole intent rather
        # than a shortcut. This panel's y axis is labelled "1.0 = linear" and
        # drawn against a linear-scaling reference line: a point placed on it
        # asserts that N clients delivered this share of N clients' work. For
        # a pool that served about one query, that assertion is false whatever
        # marker it is drawn with, and a hollow point would still put it on
        # the line to be read off. The table prints the refusal in words, the
        # Validity section names the point and shows the figure it withheld,
        # and the QPS and latency panels keep the point because those ARE
        # measurements. See `report/render.SCALING_CONCURRENCY_NOT_ACHIEVED`.
        eff = [(c, (r.get("extra") or {}).get("scaling_efficiency"))
               for c, r in zip(clients, rows)
               if (r.get("extra") or {}).get("scaling_status")
               != SCALING_CONCURRENCY_NOT_ACHIEVED]
        eff = [(c, v) for c, v in eff if v is not None]
        if eff:
            axes[2].plot([p[0] for p in eff], [p[1] for p in eff],
                         color=style["color"], marker=style["marker"],
                         linestyle=style["linestyle"], linewidth=1.8, markersize=6)

        # A point where the CLIENT container ran out of CPU, hollowed the same
        # way the filtered chart hollows a short result set. Both panels,
        # because both carry the misreading: the throughput panel shows a
        # curve flattening and the efficiency panel shows it falling away from
        # linear, and for a client-bound point neither is evidence about the
        # engine. The clients are forked OS processes serializing vectors
        # inside one container's cpuset, and at 16-32 clients against a fast
        # engine that cpuset is a real ceiling of this harness rather than of
        # the database under it. Since `ops_measurement_version` 6 it is the
        # only client-side ceiling this flag is about: the clients no longer
        # share an interpreter, so a point is hollow because the CPUs ran out
        # and not because a lock serialized them.
        for r in rows:
            extra = r.get("extra") or {}
            if not extra.get("client_bound"):
                continue
            axes[0].scatter([r["clients"]], [r["qps"]], s=140,
                            facecolors="none", edgecolors=style["color"],
                            linewidths=1.6, zorder=5)
            efficiency = extra.get("scaling_efficiency")
            # A point can be both client-bound and refused, and the refusal
            # wins on this panel: the line above omitted it, and hollowing it
            # here would put it back on the linear-scaling reference the panel
            # is read against. The QPS panel still hollows it, because a
            # client-bound throughput is a measurement with a caveat.
            if (efficiency is not None
                    and extra.get("scaling_status")
                    != SCALING_CONCURRENCY_NOT_ACHIEVED):
                axes[2].scatter([r["clients"]], [efficiency], s=140,
                                facecolors="none", edgecolors=style["color"],
                                linewidths=1.6, zorder=5)

    # Linear-scaling reference so the efficiency panel has a meaning without
    # the reader having to remember what 1.0 represents.
    axes[2].axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes[2].text(axes[2].get_xlim()[0], 1.02, " linear scaling",
                 fontsize=8, color="#888888", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    legend_bottom = _series_legend(fig, handles, labels)
    # The calibration footnotes go below the axes, not into the suptitle that
    # already carries the hollow-marker legend: they are two full sentences
    # each and naming an engine's achieved recall there would push the title
    # wider than the figure.
    footnote = _calibration_footnote(marks)
    fig.suptitle(
        f"Concurrency scaling — {dataset}   "
        f"○ = client CPU-bound (>80% of the client container's CPUs)",
        fontsize=12, y=1.14)
    fig.tight_layout()
    if footnote:
        # Under the legend when the legend came down here too, which on a
        # seven-engine run it does: both are anchored in figure fractions
        # below the canvas, so without this they are drawn on top of one
        # another.
        fig.text(0.01, _below(-0.04, legend_bottom), footnote,
                 fontsize=9, color="#444444", ha="left", va="top")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Filtered search
# ---------------------------------------------------------------------------

#: Grey, not an engine colour: a truncated point is a property of the
#: harness's budget rather than of the engine, and colouring it with the
#: engine's own palette entry would read as another measurement of it.
TIME_BOUNDED_GREY = "#8a8a8a"

#: What each filter column is, for the chart title. A reader who sees two
#: filtered charts needs to know why in the title and not in §2.4.
_COLUMN_BLURB = {
    "tag": "row_id % 100, independent of the vector space",
    "ctag": "cluster id, correlated with the vector space",
}


def filtered(records: List[Dict[str, Any]], dataset: str, out_dir: str,
             stem: str, column: Optional[str] = None
             ) -> Optional[Dict[str, str]]:
    """Filtered recall and QPS against selectivity, for one filter column.

    `column` is PR 3's "a column selector (or two charts)" done as two charts,
    which is the option that needs no JavaScript in a report that is also
    rendered to Markdown. `generate.py` calls this once per column present and
    names the files `filtered-tag-<dataset>` and `filtered-ctag-<dataset>`,
    both of which `render.py` picks up through its existing
    `figures("filtered-")` prefix.

    Drawing both columns on one pair of axes is what the previous version did
    by omission, and it is worse than it sounds: the x axis is the fraction of
    rows passing the filter, so a `tag` point at 0.0100 and a `ctag` point at
    0.0190 -- the share a nominal 1% actually delivers -- land at different x
    for the same request, and one line through both reads as an engine whose
    recall moved between two selectivities when it is two predicates.

    `column=None` plots every filtered record regardless, which is what an
    archived run needs: nothing written before PR 3 carries
    `extra.filter_column` at all.
    """
    plotted_records = [r for r in records if r.get("dataset") == dataset
                       and r.get("selectivity") is not None]
    if column is not None:
        from .generate import filter_column
        plotted_records = [r for r in plotted_records
                           if filter_column(r) == column]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("Fraction of rows passing the filter")
        ax.set_xscale("log")
        ax.grid(True, **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Filtered recall@k  (higher is better ↑)", fontsize=10)
    axes[1].set_ylabel("Queries per second  (higher is better ↑)", fontsize=10)
    axes[1].set_yscale("log")

    any_whiskers = False
    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["selectivity"])
        sel = [r["selectivity"] for r in rows]
        recalls = [r.get("recall_at_k") for r in rows]

        axes[0].plot(sel, recalls,
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=labels[key])
        # §5.1 item 6, on the recall panel only. The QPS panel's interval is
        # `extra.qps_ci95` and it is deliberately not drawn: that axis is
        # logarithmic and spans three decades here, so a bar a few percent
        # wide is shorter than the marker and says nothing a reader can see.
        # The number and its interval are both in the filtered table.
        if draw_recall_whiskers(axes[0], rows, sel, recalls, style["color"]):
            any_whiskers = True
        axes[1].plot(sel, [r.get("qps") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6)

        # The near_pass / near_fail split, on the recall panel only. A query
        # whose true unfiltered nearest neighbour passes the filter has its
        # best answer inside the candidate list the engine was going to walk
        # anyway; one whose nearest neighbour fails has to keep looking, and
        # that is where post-filtering gives up. Drawn as a thin band between
        # the two rather than two more lines: the headline is still the
        # overall recall, and the gap is the finding.
        if any((r.get("extra") or {}).get("recall_near_fail") is not None
               for r in rows):
            band_sel, lo, hi = [], [], []
            for r in rows:
                extra = r.get("extra") or {}
                a, b = extra.get("recall_near_fail"), extra.get("recall_near_pass")
                if a is None or b is None:
                    continue
                band_sel.append(r["selectivity"])
                lo.append(min(a, b))
                hi.append(max(a, b))
            if band_sel:
                axes[0].fill_between(band_sel, lo, hi, color=style["color"],
                                     alpha=0.13, linewidth=0, zorder=1)

        for r in rows:
            extra = r.get("extra") or {}
            # An engine returning fewer than k rows is a real behaviour, not
            # noise — mark it so a high-looking recall is not read as
            # unqualified success.
            if extra.get("returned_fewer_than_k"):
                axes[0].scatter([r["selectivity"]], [r.get("recall_at_k")],
                                s=140, facecolors="none",
                                edgecolors=style["color"], linewidths=1.6, zorder=5)
            # A point the budget stopped early, greyed as PR 3 asks. Its
            # recall is a mean over however many queries finished, so it
            # carries variance the other points do not; `insufficient_sample`
            # (under 50 queries) gets the heavier mark of the two.
            if extra.get("time_bounded"):
                heavy = bool(extra.get("insufficient_sample"))
                axes[0].scatter([r["selectivity"]], [r.get("recall_at_k")],
                                s=190 if heavy else 150, marker="X" if heavy else "s",
                                facecolors="none", edgecolors=TIME_BOUNDED_GREY,
                                linewidths=2.0 if heavy else 1.3, zorder=6)

    axes[0].set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    _series_legend(fig, handles, labels)
    titled = f"Filtered (hybrid) search — {dataset}"
    if column is not None:
        titled += f" — {column} ({_COLUMN_BLURB.get(column, column)})"
    fig.suptitle(
        f"{titled}\n○ = returned fewer than k results   "
        f"□ = stopped by the time budget   ✕ = under 50 queries   "
        f"band = near_pass vs near_fail recall"
        + (f"   {WHISKER_TITLE_NOTE}" if any_whiskers else ""),
        fontsize=11, y=1.18
    )
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------

def churn(records: List[Dict[str, Any]], dataset: str, out_dir: str,
          stem: str) -> Optional[Dict[str, str]]:
    """Recall and QPS after successive delete/re-insert cycles.

    **One series per (configuration, maintenance state), which is a fix.** PR 4
    §4.2 writes two records per churn fraction -- the index as the churn left
    it and the index after the engine's repair path ran -- and this chart
    grouped by configuration alone, so both landed at the same x in one
    series. Measured on the synthetic driver, 9 of 9 runs of one
    configuration: x = [0.0, 0.1, 0.1, 0.25, 0.25], one series, so each
    fraction was drawn as a vertical segment rather than as two curves. A line
    through both is a line no configuration ever produced.

    The pre-churn baseline record names no state, so it joins BOTH series: it
    is the x=0 point each of them is measured against, and giving it to one
    would leave the other starting at its first churn fraction.

    `churn_measurements` excludes the reader-pool records §4.3 writes. Their
    `qps` is aggregate throughput over `ops.churn_readers` clients against a
    corpus that is mid-change, and they carry no recall at all; on this chart
    they would be a third point per fraction, several times higher on the QPS
    panel and absent from the recall one.
    """
    from report.churn_state import churn_measurements, maintenance_state

    plotted_records = [r for r in churn_measurements(records)
                       if r.get("dataset") == dataset
                       and r.get("churn_fraction") is not None]
    if not plotted_records:
        return None
    _, labels, storages, degrees = _grouped(plotted_records)

    # Keyed `series_key(r) + (state,)` and sliced back with
    # `SERIES_KEY_WIDTH`, which is the idiom `churn_impact` below already uses
    # to widen this key by the build mode. The state stays a LOCAL extension
    # and did not follow the resource pass into `series_key`: the pass is an
    # axis of every ops chart, while this state exists on churn records and on
    # no other phase. (Worded without naming the key itself, deliberately:
    # `tests/test_t4b_readers.py` pins how many times that word occurs across
    # `report/*.py` against a paragraph in `harness/workloads/churn.py`, so a
    # comment here that spelled it would fail the suite on a prose count and
    # the fix would be an edit to a file this change has no business in.)
    states = {maintenance_state(r) for r in plotted_records} - {None}
    by_series: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in plotted_records:
        state = maintenance_state(r)
        for member in (sorted(states) if state is None else [state]):
            by_series.setdefault(series_key(r) + (member,), []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.set_xlabel("Cumulative fraction of rows deleted and re-inserted")
        ax.grid(True, **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Recall@k  (higher is better ↑)", fontsize=10)
    axes[1].set_ylabel("Queries per second  (higher is better ↑)", fontsize=10)

    any_whiskers = False
    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["churn_fraction"])
        fractions = [r["churn_fraction"] for r in rows]
        # The state is named only when the figure actually holds more than one
        # of them, the way `churn_impact` names the build mode: on a run of
        # engines with no repair path every entry would otherwise read
        # "... / before", which is a distinction the run never made.
        #
        # This sentence used to say "the pass and the build mode", and it was
        # true until the pass joined `series_key` and `churn_impact`'s local
        # pass suffix went with it -- `series_labels` has already put the pass
        # into the text being sliced out here. A cross-reference to a sibling's
        # block is only as current as that block: this one rotted in the same
        # change that deleted what it pointed at, five hundred lines away and
        # outside its diff.
        label = labels[key[:SERIES_KEY_WIDTH]]
        if len(states) > 1 and key[SERIES_KEY_WIDTH]:
            label += f" / {key[SERIES_KEY_WIDTH]} repair"
        recalls = [r.get("recall_at_k") for r in rows]
        axes[0].plot(fractions, recalls,
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6,
                     label=label)
        axes[1].plot(fractions, [r.get("qps") for r in rows],
                     color=style["color"], marker=style["marker"],
                     linestyle=style["linestyle"], linewidth=1.8, markersize=6)
        # §5.1 item 6. The recall panel only, for the reason the filtered
        # chart gives -- and here there is a second one: the x=0 baseline
        # record joins BOTH maintenance series, so its bar is drawn twice in
        # the same place. That is honest (it IS the same measurement, and it
        # is the point each series is compared against) and it is why the bar
        # at x=0 looks heavier than the others.
        if draw_recall_whiskers(axes[0], rows, fractions, recalls,
                                style["color"]):
            any_whiskers = True

    handles, labels = axes[0].get_legend_handles_labels()
    _series_legend(fig, handles, labels)
    fig.suptitle(f"Index degradation under churn — {dataset}"
                 + (f"\n{WHISKER_TITLE_NOTE}" if any_whiskers else ""),
                 fontsize=13, y=1.14)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def memory_timeline(series: Dict[str, List[Dict[str, Any]]], out_dir: str,
                    stem: str, phase: Optional[str] = None
                    ) -> Optional[Dict[str, str]]:
    """Server memory through one phase, every measurement from its own zero.

    What this chart is for is the shape: a cache filling to its ceiling looks
    different from a build spike, which looks different again from a steady
    climb that never plateaus. Shapes are compared by overlaying them, so every
    measurement starts at zero -- including two measurements that happen to
    live in one file because a phase was run twice.

    Anchoring them to the file's start instead put the second measurement 123
    hours to the right, which turned the whole chart into empty space with two
    thin bands at either end. That is the same picture as before the sessions
    were split, and it was not what splitting them was for.

    `phase` selects which measurements to draw. The ann phase loads a corpus
    and sweeps a grid; the ops phase loads it again and runs four workloads.
    They take different times and answer different questions, and twelve series
    on one axis was already crowded before any of them had two measurements.
    """
    from .loaders import split_sessions

    wanted = {}
    for name, rows in series.items():
        if phase and not name.endswith(f"-{phase}"):
            continue
        if phase is None and name.endswith("-ann"):
            continue
        rows = [r for r in rows if r.get("rss_bytes")]
        if rows:
            wanted[name] = rows
    if not wanted:
        return None

    title = "Server memory" + (f" — {phase} phase" if phase else " — ops phase")
    fig, ax = _new_axes(title, "Elapsed within the measurement (s)",
                        "Resident memory  (lower is better ↓)", figsize=(11, 4.6))
    ax.yaxis.set_major_formatter(FuncFormatter(_bytes_formatter))

    for name, rows in sorted(wanted.items()):
        engine = name.split("-")[0]
        style = style_for(engine)
        segments = split_sessions(rows)
        for index, segment in enumerate(segments):
            # Each measurement from its own zero, so the shapes overlay.
            t0 = segment[0]["t"]
            label = name if index == 0 else None
            if len(segments) > 1 and index == 0:
                label = f"{name}  ({len(segments)} measurements)"
            ax.plot([r["t"] - t0 for r in segment],
                    [r["rss_bytes"] for r in segment],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.4, alpha=1.0 if index == 0 else 0.55,
                    label=label)

    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, out_dir, stem)

def floor_bar_labels(entry: Dict[str, Any], floor: int
                     ) -> Tuple[str, Optional[str]]:
    """The text above one floor bar, and the build printed along it.

    The bars are the headline frontier, which may come from a different
    build at each floor; a bar without its build is the 25 Aug table again.
    The mismatch mark is the table's (`frontier_render.OPS_MISMATCH_MARK`),
    imported rather than spelled, so the footnote under the table explains
    the glyph on the chart. Split out so it is testable without matplotlib.
    """
    from .frontier_render import OPS_MISMATCH_MARK
    value = entry.get(f"qps_at_recall_{floor}")
    if not value:
        return "not reached", None
    top = f"{value:,.0f}"
    if entry.get(f"qps_at_recall_{floor}_is_ops_build") is False:
        top += f" {OPS_MISMATCH_MARK}"
    return top, entry.get(f"qps_at_recall_{floor}_build_label")


def qps_at_recall(summary: Dict[str, Any], dataset: str, out_dir: str,
                  stem: str, floors: Sequence[float] = (0.90, 0.95, 0.99),
                  resource_pass: Any = _ANY_PASS
                  ) -> Optional[Dict[str, str]]:
    """Grouped bars: best QPS each engine reaches at each recall floor.

    This is the comparison an operator actually makes — "how fast is it at an
    accuracy I can accept" — and it is hard to read off a Pareto curve, because
    the eye compares the curves rather than their heights at one x.

    One chart per resource pass, selected by `resource_pass`, because the bars
    are a comparison between engines at one budget. Pooling two passes into one
    axis would put fourteen bars at each floor on a nine-inch figure and ask
    the reader to pair them by legend; `generate.py` emits one entry per pass
    instead, the way it already does one per filter column. `_ANY_PASS` takes
    whichever single pass the summary holds, so a caller with nothing to say
    about passes still gets the chart it used to get.
    """
    by_pass = (summary.get("per_dataset", {}) or {}).get(dataset, {}) or {}
    if resource_pass is _ANY_PASS:
        if len(by_pass) != 1:
            return None
        per_engine = next(iter(by_pass.values()))
    else:
        per_engine = by_pass.get(resource_pass) or {}
    if not per_engine:
        return None

    engines = sorted(per_engine)
    fig, ax = _new_axes(
        f"Throughput at a recall floor — {dataset}",
        "", "Queries per second  (higher is better ↑)", figsize=(9.5, 5.2),
    )
    width = 0.8 / max(len(engines), 1)
    xs = range(len(floors))
    plotted = False

    for i, engine in enumerate(engines):
        style = style_for(engine)
        vals = [per_engine[engine].get(f"qps_at_recall_{int(f * 100)}") for f in floors]
        offs = [x + i * width - 0.4 + width / 2 for x in xs]
        heights = [v if v else 0 for v in vals]
        if any(heights):
            plotted = True
        bars = ax.bar(offs, heights, width * 0.92, color=style["color"],
                      label=style["label"], edgecolor="none")
        for b, v, f in zip(bars, vals, floors):
            # An engine that never reaches a floor gets an explicit marker, not
            # a zero bar that reads as "measured and very slow".
            top, build = floor_bar_labels(per_engine[engine], int(f * 100))
            # One label per bar, ABOVE it and along it, in the engine's
            # colour: value, mark, then build. The build used to be white
            # text inside the bar, which a short bar truncated ("quantiza"
            # for Percona Search on the 25 Aug chart), a tall one let run
            # off onto the white background, and a horizontal value above
            # collided with. Rotated, each label is one bar wide, so eight
            # engines do not overlap.
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() if v else 0,
                    f" {top}  {build}" if build else f" {top}",
                    ha="center", va="bottom", fontsize=7,
                    color=style["color"] if v else "#999999",
                    rotation=90)

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"recall ≥ {f:.2f}" for f in floors])
    # Headroom so the value labels above the tallest bars do not collide with
    # the legend. Larger than it was: each label now carries its build and
    # stands upright above the bar.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.75)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Normalized vs tuned
# ---------------------------------------------------------------------------

def pass_comparison(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                    stem: str) -> Optional[Dict[str, str]]:
    """What tuning actually bought, per engine and per dimension.

    Both passes are measured but nothing previously compared them directly, so
    the central question the two-pass design exists to answer — does tuning
    change the ranking? — had to be reconstructed by hand from two tables.
    """
    dims = [
        ("recall_qps", "qps", "Query throughput", lambda r: r.get("clients") in (1, None)),
        ("index_build", "build_wall_s", "Index build time", lambda r: True),
        ("concurrency", "qps", "Throughput @ max clients", lambda r: True),
        ("filtered", "qps", "Filtered throughput", lambda r: True),
    ]

    engines, panels = set(), []
    for phase, field, label, keep in dims:
        by = {}
        for r in records:
            if (r.get("phase") != phase or r.get("dataset") != dataset
                    or not keep(r) or r.get(field) is None):
                continue
            key = (r["engine"], r.get("resource_pass"))
            # Best observed value per (engine, pass): max for throughput,
            # min for build time, since lower is better there.
            prev = by.get(key)
            val = r[field]
            by[key] = val if prev is None else (
                min(prev, val) if field.endswith("_s") else max(prev, val))
            engines.add(r["engine"])
        panels.append((label, field, by))

    engines = sorted(engines)
    if not engines or not any(p[2] for p in panels):
        return None

    # Through `grid_figsize` and not `(4.0 * len(panels), 4.8)`, which is what
    # this said and which is a NO-OP today: `dims` above is a literal list of
    # four, so the width is always 16 inches and 16/4.8 is 3.33:1. It is
    # routed anyway because the expression reads as data-dependent and one
    # more panel would make it 4.17:1 -- over the bound, by the same arithmetic
    # that put `latency_percentiles` there, in a chart nobody would re-measure
    # while adding a dimension to a list.
    fig, axes = plt.subplots(1, len(panels),
                             figsize=grid_figsize(len(panels), 1, 4.0, 4.8))
    fig.patch.set_alpha(0.0)
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, field, by) in zip(axes, panels):
        ax.patch.set_alpha(0.0)
        ax.grid(True, axis="y", **GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        lower_better = field.endswith("_s")
        ax.set_ylabel(f"{label}  ({'lower' if lower_better else 'higher'} is better"
                      f" {'↓' if lower_better else '↑'})", fontsize=9)

        for i, engine in enumerate(engines):
            style = style_for(engine)
            n = by.get((engine, "normalized"))
            u = by.get((engine, "tuned"))
            ax.bar(i - 0.19, n or 0, 0.36, color=style["color"], alpha=0.45,
                   edgecolor="none")
            ax.bar(i + 0.19, u or 0, 0.36, color=style["color"], edgecolor="none")
            if n and u:
                delta = (u - n) / n
                better = (delta < 0) if lower_better else (delta > 0)
                ax.text(i, max(n, u), f"{delta:+.0%}", ha="center", va="bottom",
                        fontsize=8,
                        color="#2ca02c" if better else "#d62728")
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels([style_for(e)["label"].split(" (")[0] for e in engines],
                           fontsize=8, rotation=15, ha="right")

    fig.suptitle(f"Normalized (pale) vs tuned (solid) — {dataset}\n"
                 f"percentage is the change tuning produced", fontsize=12, y=1.06)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Latency distribution
# ---------------------------------------------------------------------------

def latency_percentiles(records: List[Dict[str, Any]], dataset: str,
                        out_dir: str, stem: str) -> Optional[Dict[str, str]]:
    """p50 and p99 against search width, one panel per engine.

    Mean latency hides the tail, and the tail is what a service-level objective
    is written against. An engine with a good p50 and a bad p99 is a different
    proposition from one with both merely acceptable.

    One panel each, rather than everything on one axis. Two things made the
    single-axis version unreadable and they compound: the sweep grew to a dozen
    series once graph degree and ef_construction were both being varied, and
    AliSQL's cliff put 384 ms on the same linear scale as Valkey's 0.7 ms, so
    five engines were pressed flat against the floor. Panels fix the crowding
    and a shared log scale fixes the range, while still letting a reader
    compare across panels -- which a per-engine y-scale would not.
    """
    plotted_records = [r for r in records
                       if r.get("phase") == "recall_qps" and r.get("dataset") == dataset
                       and r.get("ef_search") and r.get("latency_p99_ms")]
    if not plotted_records:
        return None
    by_series, labels, storages, degrees = _grouped(plotted_records)

    engines = sorted({key[0] for key in by_series})
    columns = min(3, len(engines))
    rows_count = (len(engines) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, sharex=True, sharey=True,
                             figsize=grid_figsize(columns, rows_count,
                                                  4.4, 3.1))
    fig.patch.set_alpha(0.0)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    values = [v for r in plotted_records
              for v in (r.get("latency_p50_ms"), r.get("latency_p99_ms"))
              if v]
    low = max(0.1, min(values) * 0.7)
    high = max(values) * 1.4

    for index, engine in enumerate(engines):
        ax = axes[index]
        ax.patch.set_alpha(0.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.set_title(style_for(engine)["label"], fontsize=10.5, pad=6)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)

        for key, series_rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
            if key[0] != engine:
                continue
            style = series_style(series_rows[0], storages.get(engine),
                                 degrees.get(engine))
            series_rows = sorted(series_rows, key=lambda r: r["ef_search"])
            ef = [r["ef_search"] for r in series_rows]
            p50 = [r.get("latency_p50_ms") for r in series_rows]
            p99 = [r.get("latency_p99_ms") for r in series_rows]
            # Band between p50 and p99 makes the spread visible at a glance;
            # the solid line is p50 so the medians stay comparable.
            ax.fill_between(ef, p50, p99, color=style["color"],
                            alpha=0.13, linewidth=0)
            # The engine is the panel, so within it the label names only what
            # varies -- the graph degree, or the build-quality setting.
            within = labels[key]
            within = within.split(" / ", 1)[1] if " / " in within else None
            ax.plot(ef, p50, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style.get("linewidth", 1.9),
                    alpha=style.get("alpha", 1.0), markersize=4,
                    label=within)
            ax.plot(ef, p99, color=style["color"], linestyle=":",
                    linewidth=1.1, alpha=0.8 * style.get("alpha", 1.0))

        if any(l.get_label() and not l.get_label().startswith("_")
               for l in ax.get_lines()):
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    for spare in axes[len(engines):]:
        spare.set_visible(False)

    for index, ax in enumerate(axes[:len(engines)]):
        if index >= len(engines) - columns:
            ax.set_xlabel("ef_search  (search width)", fontsize=9.5)
        if index % columns == 0:
            ax.set_ylabel("Latency (ms)  ↓", fontsize=9.5)

    fig.suptitle(f"Query latency distribution — {dataset}", fontsize=13)
    # Wrapped, because at one engine this figure is 4.4 inches wide and this
    # sentence is about 6: unwrapped it hung off the side and `_save`'s
    # `bbox_inches="tight"` paid for it in width. `latency-gist-960-euclidean`
    # and `latency-glove-100-angular` in the 2026-10-15 run are both 837 and
    # 835 pixels on a 660-pixel canvas for this one artist.
    note = _wrap_to_figure(
        fig, "solid = p50;  dotted = p99;  shaded = the span "
        "between them.  Log scale, shared across panels.", 8)
    fig.text(0.99, 0.005, note,
             ha="right", va="bottom", fontsize=8, color="#888888")
    # The band reserved under the axes grows with the caption, or a wrapped
    # caption climbs into the x labels it sits beneath. One line reserves the
    # 3% it always did, so a figure wide enough never to wrap is unchanged.
    fig.tight_layout(rect=(0, 0.03 * (note.count("\n") + 1), 1, 0.96))
    return _save(fig, out_dir, stem)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def storage_breakdown(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                      stem: str) -> Optional[Dict[str, str]]:
    """Index vs table bytes, stacked.

    Index size alone is misleading between these engines: pgvector's HNSW keeps
    the vectors inside the index, while MHNSW and VIDX keep them in the table
    and store only the graph. Only the total is comparable.
    """
    rows = [r for r in records
            if r.get("phase") == "index_build" and r.get("dataset") == dataset
            and (r.get("index_bytes") or r.get("table_bytes"))
            # An in-memory engine writes no files. Stacking its resident bytes
            # beside the others' on-disk bytes in a chart titled "on-disk
            # footprint" invites a comparison between two different quantities;
            # its memory cost is in the build table, labelled as resident.
            and not (r.get("extra") or {}).get("in_memory_only")]
    if not rows:
        return None

    labels, index_b, table_b, colors = [], [], [], []
    for r in sorted(rows, key=lambda r: (r["engine"], str(r.get("resource_pass")),
                                         str(r.get("build_mode")))):
        tag = style_for(r["engine"])["label"].split(" (")[0]
        # Storage engine belongs here for the same reason it belongs in the
        # build table: MariaDB gets an InnoDB bar and a MyISAM bar, and without
        # it they carry identical captions.
        storage = r.get("storage_engine")
        extra = [x for x in (r.get("resource_pass"),
                             storage if storage not in (None, "heap") else None,
                             r.get("build_mode")) if x]
        labels.append(f"{tag}\n{' / '.join(extra)}")
        index_b.append((r.get("index_bytes") or 0))
        table_b.append((r.get("table_bytes") or 0))
        colors.append(style_for(r["engine"])["color"])

    # 1.5 inches a bar until the aspect bound stops it. Past that the bars
    # keep their share of a fixed width and the captions turn upright, which
    # is the only one of the two that can be given up: a caption reads
    # rotated, a 49-inch figure does not read at all.
    figsize = bar_chart_figsize(len(labels), 1.5, 8.0, 5.2)
    crowded = figsize[0] < 1.5 * len(labels)
    fig, ax = _new_axes(f"On-disk footprint — {dataset}", "",
                        "Bytes  (lower is better ↓)", figsize=figsize)
    ax.yaxis.set_major_formatter(FuncFormatter(_bytes_formatter))
    xs = range(len(labels))
    ax.bar(xs, index_b, 0.62, color=colors, edgecolor="none", label="index")
    ax.bar(xs, table_b, 0.62, bottom=index_b, color=colors, alpha=0.4,
           edgecolor="none", label="table")
    for x, (i, tb) in enumerate(zip(index_b, table_b)):
        # The totals turn upright with the captions and for the same reason:
        # at the bar pitch a capped width gives, "191 MiB" is wider than the
        # bar it belongs to and runs into its neighbours' totals.
        ax.text(x, i + tb, _bytes_formatter(i + tb, 0), ha="center", va="bottom",
                fontsize=8, rotation=90 if crowded else 0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8, rotation=90 if crowded else 0)
    # Headroom for those totals: upright they stand about three times as tall
    # as they do flat, and without the extra room they climb into the title.
    ax.set_ylim(top=ax.get_ylim()[1] * (1.30 if crowded else 1.12))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    # Below the axes, not inside them: at seven bars the in-plot version
    # overlapped the data it was explaining. Below the CAPTIONS too once they
    # are upright, because then they are the thing that reaches furthest down.
    ticks_bottom = None
    if crowded:
        ticks_bottom = min(
            (_figure_fraction_y(fig, tick.get_window_extent(_renderer(fig)).y0)
             for tick in ax.get_xticklabels()), default=None)
    # Wrapped for the same reason the latency caption is: on a one- or
    # two-engine run this figure is at its 8-inch minimum and this sentence is
    # wider, which made the PNG 1,247 pixels on a 1,200-pixel canvas. Below
    # the axes already, so the extra line costs height and nothing else.
    fig.text(0.5, _below(-0.06, ticks_bottom),
             _wrap_to_figure(
                 fig,
                 "solid = index, pale = table.  pgvector stores vectors inside the index; "
                 "MHNSW and VIDX keep them in the table — only the totals are comparable.",
                 8),
             ha="center", va="top", fontsize=8, color="#888888")
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Churn impact
# ---------------------------------------------------------------------------

def churn_retention(records: List[Dict[str, Any]], dataset: str
                    ) -> Dict[Tuple[Any, ...], List[Tuple[float, float]]]:
    """Post-churn QPS as a fraction of each series' own pre-churn baseline.

    The key names the storage engine. Without it MariaDB's InnoDB and MyISAM
    runs shared one baseline slot, so whichever was recorded second overwrote
    the first and both post-churn points were divided by it -- turning a real
    3% MyISAM loss and an 84% InnoDB loss into two numbers that were each
    wrong by the ratio between the two baselines.

    **The repaired measurement is excluded, and that is the fix for a second
    instance of the same defect.** PR 4 §4.2 writes two records per churn
    fraction -- before and after the engine's repair path -- and this function
    put both in one series' point list. Since it returns `sorted(points)`, the
    pair arrived duplicated at every x AND indistinguishable by position:
    measured on the synthetic driver, x = [0.1, 0.1, 0.25, 0.25] in 9 of 9
    runs of one configuration.

    Excluded rather than split into two series, and the reason is the
    denominator rather than the plumbing. This ratio is against the PRE-CHURN
    baseline, a state before any maintenance ran, so a repaired measurement
    over it answers "what did churn plus the repair cost?" -- a different
    question from the one the axis asks, and one that cannot share an axis
    labelled "QPS as a fraction of pre-churn" without saying which of the two
    each line is. The repaired state is not lost: `charts.churn` draws it as
    its own series and `render._churn_table` gives it its own row with the
    `Repair` column naming what ran.

    This does NOT claim to fix the ratios above 1.0 that this chart can
    produce, and no mechanism is asserted for them. On the synthetic driver,
    which runs no vacuum at all, 2 of 9 runs had BOTH points at a fraction
    above 1.0 -- so the un-maintained measurement outran the baseline too, and
    "a vacuum makes it faster" is refuted as the explanation there. It is
    jitter in a 50-query QPS figure whose spread was 6,028 to 16,811 across
    those runs. Whether a real engine's post-repair throughput can beat its
    own pre-churn baseline is plausible and unmeasured.

    The reader-pool records §4.3 writes are excluded by `churn_measurements`
    for a harder reason: their `qps` is aggregate throughput over
    `ops.churn_readers` clients, so dividing it by a single-client baseline
    publishes a retention of roughly the reader count on an axis where 1.0
    means "no degradation".
    """
    from report.churn_state import churn_measurements, maintenance_state

    baseline: Dict[Tuple[Any, ...], float] = {}
    after: Dict[Tuple[Any, ...], List[Tuple[float, float]]] = {}
    for r in churn_measurements(records):
        if r.get("dataset") != dataset or not r.get("qps"):
            continue
        if maintenance_state(r) == "after":
            continue
        # The tuple is a CARRIER, not a discriminator, and both appended
        # fields are already inside `series_key`: `build_signature`'s column
        # fallback holds `build_mode` and has since it was written, and the
        # pass joined `series_key` when the ops charts stopped pooling the two
        # budgets. They are appended anyway so `churn_impact` can read them
        # back off the key at `SERIES_KEY_WIDTH` and `SERIES_KEY_WIDTH + 1` for
        # its legend, which is the shape `tests/test_config.py` and
        # `tests/test_t4b_readers.py` both look this function's result up by
        # and which `harness/workloads/churn.py` cites as "a key shape T4b does
        # not own". Appending a field the key already carries cannot change the
        # partition -- both are functions of the record -- so the redundancy
        # costs nothing and dropping it would break two consumers to tidy one
        # tuple.
        key = series_key(r) + (r.get("resource_pass"), r.get("build_mode"))
        if (r.get("churn_fraction") or 0) == 0:
            baseline[key] = r["qps"]
        else:
            after.setdefault(key, []).append((r["churn_fraction"], r["qps"]))

    return {key: [(f, q / baseline[key]) for f, q in sorted(points)]
            for key, points in after.items() if baseline.get(key)}


def churn_impact(records: List[Dict[str, Any]], dataset: str, out_dir: str,
                 stem: str) -> Optional[Dict[str, str]]:
    """Throughput retained after churn, as a fraction of the pre-churn baseline.

    Recall usually survives churn; throughput does not, and that is invisible in
    a recall-only view. Plotted as a retention ratio so engines with different
    absolute speeds can be compared on how well they hold up.
    """
    retained = churn_retention(records, dataset)
    if not retained:
        return None

    fig, ax = _new_axes(f"Throughput retained after churn — {dataset}",
                        "Cumulative fraction of rows deleted and re-inserted",
                        "QPS as a fraction of pre-churn  (higher is better ↑)",
                        figsize=(9, 5.2))
    # The same record set `churn_retention` derived `retained` from, so the
    # legend text and the `next(...)` style lookup below describe the series
    # that were actually plotted. `series_labels` compares records within an
    # engine to find what varies, so feeding it a set the ratio excluded can
    # name an axis no plotted line differs on.
    from report.churn_state import churn_measurements

    churn_records = [r for r in churn_measurements(records)
                     if r.get("dataset") == dataset and r.get("qps")]
    # `_grouped`, not a second copy of its loop, and this is the repair of a
    # defect rather than a tidy-up. The hand-rolled block that stood here was
    #
    #     storages = {}
    #     for key in {series_key(r) for r in churn_records}:
    #         storages.setdefault(key[0], []).append(None)
    #
    # -- one entry per series, and the entry a literal `None` instead of that
    # series' own `storage_engine`. `series_style` looks the record's storage
    # engine up in this list with `.index()`, so the list has to CONTAIN the
    # values, and it did not. The crash was therefore guaranteed on any engine
    # with more than one series and impossible on any engine with one, because
    # the lookup sits behind `len(storages) > 1`: a one-series engine gets
    # `[None]`, the guard stays shut and the literal is never read. That guard
    # is why this shipped. Regenerated against `PS/new/full-20261015-215006`,
    # the seven-engine `--resource-pass both` run of 2026-10-15, where each
    # engine has six churn series (three graph degrees over two passes) and
    # pgvector twelve:
    #
    #   [report] chart churnimpact-fashion-mnist-784-euclidean failed: 'InnoDB' is not in list
    #   [report] chart churnimpact-glove-100-angular failed: 'InnoDB' is not in list
    #   [report] chart churnimpact-sift-128-euclidean failed: 'InnoDB' is not in list
    #
    # -- every dataset that ran the churn workload. `generate.py` catches per
    # chart so one bad figure cannot kill a twenty-hour run's report, so the
    # stem never reached `chart_paths` and neither renderer emitted it or
    # linked it: the loss was three silently absent figures, not a broken
    # page. That bundle's own shipped report has 47 SVGs including `churn-*`
    # and no `churnimpact-*` at all.
    #
    # The shared helper is reused rather than the pattern copied because a
    # second hand-rolled copy of a de-duplicating loop is how this diverged in
    # the first place: `_grouped` is forty lines up and `charts.churn`, the
    # sibling chart on these same records, already calls it. The key shapes do
    # differ -- `retained` is keyed `series_key(r) + (resource_pass,
    # build_mode)` -- but that is the loop below's business and not the
    # helper's. What `_grouped` returns is keyed by the engine (`key[0]`) and
    # by `key[:SERIES_KEY_WIDTH]`, which is exactly how both dicts are read
    # here already, so the wider key never reaches it.
    #
    # `_grouped` samples `storage_engine` and `m` off one record per series.
    # Sound for these records and not by luck: the ops harness writes no
    # `extra.source_file`, so `build_signature` takes its column branch, which
    # names both fields -- all 203 churn records in that bundle took it and no
    # series held two values of either.
    #
    # `groups` is discarded: the series partition comes from `churn_retention`,
    # which is `_grouped`'s partition with the ratio's own exclusions applied
    # (the pre-churn baseline, the repaired measurement, the reader pool).
    _groups, labels, storages, degrees = _grouped(churn_records)

    # The build mode is named only when the figure actually holds more than one
    # of them. On a single-mode run it turned every legend entry into
    # "... (post)", which crowds out the storage engine -- the one distinction
    # this chart exists to show.
    #
    # **The resource pass used to be named here too, on the same rule, and is
    # not any more: `series_labels` names it now.** It had to be named locally
    # while the pass was absent from `series_key`, because the shared label
    # could not see it. Now that the pass is IN the key, `_key_suffixes` adds
    # it to the shared label under the identical "only when more than one"
    # rule, and keeping this block as well printed it twice. Measured over a
    # two-pass churn fixture with the removed lines reinstated: the shipped
    # label is "MariaDB 11.8 (MHNSW) / tuned" and theirs was
    # "MariaDB 11.8 (MHNSW) / tuned / tuned", with the build mode landing
    # between the two on a figure that varies it as well.
    modes = {key[SERIES_KEY_WIDTH + 1] for key in retained}

    for key, points in sorted(retained.items(), key=lambda kv: str(kv[0])):
        sample = next(r for r in churn_records
                      if series_key(r) == key[:SERIES_KEY_WIDTH])
        # `degrees` as well, which this chart never passed at all -- the same
        # root cause one step further on, since the block above had no
        # equivalent of `_grouped`'s second list to hand over. Colour is the
        # engine and the linestyle is now really the storage engine, so on the
        # 2026-10-15 run, which varies storage on no engine and sweeps three
        # graph degrees on every one, repairing only the storage list would
        # have published six MariaDB lines identical in colour, marker,
        # linestyle and width, separable only by reading the legend. That is
        # verbatim the defect `series_style`'s own comment records against two
        # AliSQL curves, and the alpha and width it returns are the channels
        # it exists to set.
        style = series_style(sample, storages.get(key[0]), degrees.get(key[0]))
        points = sorted(points)
        label = labels.get(key[:SERIES_KEY_WIDTH], key[0])
        if key[SERIES_KEY_WIDTH + 1] and len(modes) > 1:
            label += f" ({key[SERIES_KEY_WIDTH + 1]})"
        # `.get` with this chart's own 1.9 as the fallback, not `style["..."]`:
        # `series_style` adds `alpha` and `linewidth` only when the engine
        # actually swept more than one degree, which is the same "only when
        # more than one" rule the legend above follows. So a single-degree run
        # draws at the 1.9 and the full opacity it always did, and a report
        # that swept one M does not move. (`charts.churn` reads the linestyle
        # and hardcodes its own 1.8, so it computes `degrees` and discards the
        # distinction. That is its bug to fix, on its own figure, with its own
        # before and after; it is not fixed by a change whose diff is this
        # function.)
        ax.plot([0] + [p[0] for p in points], [1.0] + [p[1] for p in points],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style.get("linewidth", 1.9),
                alpha=style.get("alpha"), markersize=6,
                label=label)

    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.text(0.0, 1.01, " no degradation", fontsize=8, color="#888888", va="bottom")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# ef_search calibration (appendix)
# ---------------------------------------------------------------------------

def calibration_curves(records: List[Dict[str, Any]], dataset: str,
                       out_dir: str, stem: str) -> Optional[Dict[str, str]]:
    """PR 2 §2.3.4: recall against ef_search, one line per engine.

    The audit trail for every other number in a calibrated report. The
    operating-point table states the ef each engine was given; this is the
    curve that choice was made from, so a reader can see whether the chosen
    point sits on a plateau or on a slope — and a sweep that early-stopped
    after three of twelve grid points is visibly three points long rather than
    silently short.

    It "doubles as a sanity check against the ann frontier" in the sense that a
    reader can carry a recall off this chart to §4 and find the same engine at
    the same ef_search there. The two are deliberately NOT drawn on one axis:
    `PHASE_CALIBRATION`'s own docstring is explicit that these are not Pareto
    points — they are single-client probes against the index the ops pass just
    built, measured in a different container, over a different query count, by
    a different code path's recall computation. Overlaid they would read as one
    curve, and the first thing a reader would do with a disagreement is
    average it. Kept apart, a disagreement between the two charts is what it
    actually is: a signal that one of the two measurement paths is wrong about
    this engine.

    Returns None when no sweep ran, which `generate.py` also guards ahead of
    the call: the chart is optional because the sweep is.
    """
    from .loaders import (calibration_records, operating_point_key,
                          operating_points)

    _require_matplotlib()
    points = [r for r in calibration_records(records)
              if r.get("dataset") == dataset
              and r.get("ef_search") is not None
              and r.get("recall_at_k") is not None]
    if not points:
        return None

    targets = sorted({(r.get("extra") or {}).get("target_recall")
                      for r in points
                      if (r.get("extra") or {}).get("target_recall")
                      is not None})

    fig, ax = _new_axes(
        f"ef_search calibration — {dataset}",
        "ef_search (requested)", "Recall@k  (single-client sweep)",
        figsize=(9, 5.5))
    ax.set_xscale("log", base=2)

    by_series, labels, storages, degrees = _grouped(points)
    # Which (unit, ef) pairs the run actually settled on, so the ring marks the
    # point the downstream workloads were measured at rather than this
    # function's own guess at which point the rule should have picked. Matched
    # through `operating_point_key`, via the key each row carries, so there is
    # one definition of the unit: a ring on a point no workload ran at, or no
    # ring where a sweep succeeded, is then a real disagreement between the
    # sweep and what consumed it rather than two derivations drifting.
    settled = {(row["key"], row["ef_search"])
               for row in operating_points(records)
               if row.get("dataset") == dataset
               and row.get("ef_search") is not None}

    for key, rows in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        style = series_style(rows[0], storages.get(key[0]), degrees.get(key[0]))
        rows = sorted(rows, key=lambda r: r["ef_search"])
        ax.plot([r["ef_search"] for r in rows], [r["recall_at_k"] for r in rows],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.8, markersize=5,
                label=labels[key])
        chosen = [r for r in rows
                  if (operating_point_key(r), r["ef_search"]) in settled]
        if chosen:
            ax.scatter([r["ef_search"] for r in chosen],
                       [r["recall_at_k"] for r in chosen], s=170,
                       facecolors="none", edgecolors=style["color"],
                       linewidths=1.8, zorder=5)

    for target in targets:
        ax.axhline(target, color="#888888", linewidth=1.0, linestyle="--")
        ax.text(ax.get_xlim()[0], target + 0.002,
                f" target {_fmt_target(target)}", fontsize=8,
                color="#888888", va="bottom")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.0, -0.17,
            "◯ = the ef_search the concurrency, filtered and churn workloads "
            "were then measured at",
            transform=ax.transAxes, fontsize=8, color="#444444")
    return _save(fig, out_dir, stem)
