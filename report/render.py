"""Markdown and self-contained HTML rendering for the vector-bench report.

Two rules shape the output:

* **Environment before results.** The hardware, engine versions and resource
  limits come first, so nobody reads a number without the context that scopes it.
* **Caveats are not a footer.** Anything that invalidates or narrows a result —
  a full-scan fallback, a missing SIMD path, an engine that returned fewer than
  k rows — appears in a Validity section above the charts, not buried at the end.

The HTML is fully self-contained: SVG charts are inlined, CSS is inline, and no
request leaves the page. It renders in both light and dark themes.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from report import disclosures as _disclosures
from report import provenance_render as _provenance
from report import scope as _scope

# Defaults; the manifest's per-engine `presentation.label` overrides them, so an
# engine added after this file was written names itself correctly.
ENGINE_LABEL = {
    "mariadb": "MariaDB 11.8 (MHNSW)",
    "mariadb123": "MariaDB 12.3 (MHNSW)",
    "alisql": "AliSQL (VIDX)",
    "pgvector": "PostgreSQL (pgvector)",
    "mongodb": "Percona Search for MongoDB (mongot)",
    "valkey": "Valkey (valkey-search)",
}


def adopt_presentation(manifest: Dict[str, Any]) -> None:
    for name, info in (manifest.get("engines") or {}).items():
        label = ((info or {}).get("presentation") or {}).get("label")
        if label:
            ENGINE_LABEL[name] = label


def _fmt_bytes(value: Optional[float]) -> str:
    if value is None:
        return "—"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PiB"


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _label(engine: str) -> str:
    return ENGINE_LABEL.get(engine, engine)


# ---------------------------------------------------------------------------
# Shared section builders (returned as Markdown; HTML converts them)
# ---------------------------------------------------------------------------

def described_passes(summary: Dict[str, Any]) -> List[str]:
    """Every resource pass the report describes, as `summarize` computed it.

    One number, carried in the summary, so §4's headings and the floor chart's
    stem agree about how many passes there are. They did not: §4 counted the
    passes in `per_dataset`, which holds only those that produced recall for
    one dataset, so a report whose §1 named two passes could print §4 under a
    bare single-pass heading, and the stem qualified one dataset and not
    another.

    Those two are its only callers. An earlier version of this docstring said
    §1's row and §2's notice shared it as well; they do not -- §1 derives from
    `_pass_resources(manifest)` and the single-valued `resource_pass`, §2
    reads `summary["passes_without_a_tree"]` -- and claiming a wider
    agreement than exists is the same kind of false statement as the narrow
    ones this change went back to fix.

    The fallback derives it from `per_dataset` for a caller that builds a
    summary by hand -- several tests do -- and yields 1 for a single-pass one,
    which is what the byte-identical single-pass path requires.
    """
    described = summary.get("described_passes")
    if described is not None:
        return list(described)
    per_dataset = summary.get("per_dataset", {}) or {}
    return sorted({p for passes in per_dataset.values() for p in passes},
                  key=str)


def passes_that_ran(manifest: Dict[str, Any]) -> List[str]:
    """Passes the phase log says this run executed, in first-seen order.

    `phases[]` is appended per measured unit, so it is the one place a
    manifest states what the run actually did rather than what its config
    block was last told. It needs no fingerprint, which is why it can describe
    an archived both-pass run that recorded only one.
    """
    out: List[str] = []
    for phase in (manifest.get("phases") or []):
        name = phase.get("resource_pass")
        if name and str(name) not in out:
            out.append(str(name))
    return out


def recorded_passes(manifest: Dict[str, Any]
                    ) -> List[Tuple[str, Optional[str]]]:
    """(pass, ann fingerprint) for every pass this run executed, in run order.

    `config["passes"]` is accumulated one entry per pass by
    `orchestrator/manifest.py`. A manifest written before that list existed
    carries only the single-valued `resource_pass` / `ann_fingerprint`, which
    describe one pass -- whichever `set_config` saw last.

    **A pass `phases[]` names and the config block does not still appears
    here, with `None` for its fingerprint.** That is the archived both-pass
    run: `full-20261015-215006` ran `normalized` then `tuned`, and its config
    block names only `tuned`. Deriving the pass list from the config block
    alone made §1 and §11 describe a run of one pass, because the config block
    is exactly the thing that forgot. `None` means "this pass ran and its
    results tree cannot be located" -- the tree needs the fingerprint, but
    naming the pass and reproducing the run do not.

    **Ordered by `phases[]`.** The entry order in `config["passes"]` is the
    order `set_config` was called in, and on a resume that is not the run
    order: `_accumulate_passes` adopts the pass the config block carried
    before the one it is called for, so resuming the archived manifest records
    `['tuned', 'normalized']` for a run that measured normalized first. Any
    pass the phase log does not mention keeps its recorded position, after the
    ones it does.

    The fingerprint is read, never derived. `report/` runs inside a bench
    image that mounts report/ and harness/ only; importing the orchestrator to
    compute it crashed generation at the end of a twenty-hour run once.

    Lives here rather than in `generate.py` because the narrowing of the ann
    results tree, the two sections that describe the run and §2's Validity
    notice all have to agree about which passes ran -- a report naming one
    pass above trees read from two is the bug this list exists to end.
    `generate.py` imports it, for the same reason it imports `ENGINE_LABEL`:
    one reader, not two.
    """
    config = manifest.get("config") or {}
    out: List[Tuple[str, Optional[str]]] = []
    seen = set()
    for entry in (config.get("passes") or []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("resource_pass")
        if name and str(name) not in seen:
            seen.add(str(name))
            out.append((str(name), entry.get("ann_fingerprint")))
    # Consulted whether or not the list is populated, not only as a fallback.
    # `_pass_resources` adopts the carried block unconditionally, so a manifest
    # whose list omits the pass its single-valued keys name made the two
    # readers disagree: §1 printed that pass's real limits while §2 declared
    # its recall unlocatable, because only one of them had read its recorded
    # fingerprint. Unreachable from the fixed orchestrator, reachable from a
    # hand-edited manifest, and a disagreement between two sections of one
    # report is the class of defect this whole change is about.
    name = config.get("resource_pass")
    if name and str(name) not in seen:
        seen.add(str(name))
        out.append((str(name), config.get("ann_fingerprint")))
    for name in passes_that_ran(manifest):
        if name not in seen:
            seen.add(name)
            out.append((name, None))
    order = passes_that_ran(manifest)
    ranked = [pair for pair in out if pair[0] in order]
    ranked.sort(key=lambda pair: order.index(pair[0]))
    return ranked + [pair for pair in out if pair[0] not in order]


def passes_without_a_tree(manifest: Dict[str, Any]) -> List[str]:
    """Passes this run executed whose ann results tree cannot be located.

    A pass with no recorded `ann_fingerprint` -- an archived manifest's other
    pass, or one a resume declined to fingerprint because it was measured on
    another machine. The tree is keyed by that string and nothing in `report/`
    can recompute it.

    **That is the whole of what a name in this list means: the tree could not
    be narrowed to.** Whether the pass's recall is actually missing from the
    report is a second question with a different answer, because both
    whole-tree fallbacks in `generate._narrow_to_this_run` hand the loader the
    un-narrowed root and `_pass_from_path` then recovers the pass from the
    directory layout. On a host that has itself measured the pass -- the case
    `orchestrator/manifest.py`'s `_accumulate_passes` documents -- this
    function returns the pass, §4 publishes its recall, and §2 correctly says
    nothing.

    So §2 does NOT render this list. It renders the intersected one that
    `generate.summarize` builds, which drops any pass whose recall the
    fallback picked up anyway; an earlier version of this docstring claimed
    both that a name here means the measurements are absent and that §2
    renders it, and both halves are false of the un-intersected return. The
    reason that matters is not tidiness: read as a guarantee, it invites
    dropping the intersection as redundant, which is exactly the hole the
    retraction in `_accumulate_passes` exists to record.

    This list is what the generator's stdout warning reads, and that warning
    is worded to match -- it claims only that the tree cannot be narrowed to.
    stdout is not bundled by `./run-benchmark.sh export`, which is why §2
    exists as well.
    """
    return [name for name, fingerprint in recorded_passes(manifest)
            if not fingerprint]


def _pass_resources(manifest: Dict[str, Any]
                    ) -> List[Tuple[str, Dict[str, Any]]]:
    """(pass, its own resolved resources) for each pass, in run order.

    A pass with no limits of its own gets `{}`, which §1 renders as `?` and
    `—`. It must NOT fall back to the single-valued `resolved_resources`:
    those describe one named pass, and lending them to another is the
    misattribution this change exists to end -- `full-20261015-215006` would
    print `normalized — server cpuset | 0-75` for a pass that ran on 8 CPUs.
    """
    config = manifest.get("config") or {}
    by_name = {str(e.get("resource_pass")): (e.get("resolved_resources") or {})
               for e in (config.get("passes") or [])
               if isinstance(e, dict) and e.get("resource_pass")}
    # The single-valued block is a complete record of exactly one pass: the
    # one it names, and no other.
    carried = config.get("resource_pass")
    if carried and str(carried) not in by_name:
        by_name[str(carried)] = config.get("resolved_resources") or {}
    return [(name, by_name.get(name, {}))
            for name, _fingerprint in recorded_passes(manifest)]


def reproduction_pass_flag(manifest: Dict[str, Any]) -> str:
    """The `--resource-pass` value that would perform this run again.

    §11 printed the single-valued `config.resource_pass`, so a run of both
    passes was reproduced as `--resource-pass tuned` -- a command that does
    neither of the two things the run did. `both` is the flag `cli.py` expands
    into the pass list, so it is the only correct answer for a multi-pass run;
    a comma-separated list is not accepted there and would send the reader to
    an error.

    Taken from `recorded_passes`, which unions the config block with
    `phases[]`, so an archived manifest that recorded one pass and ran two
    still reproduces as `both`. No fingerprint is involved.
    """
    names = [name for name, _fingerprint in recorded_passes(manifest)]
    if len(names) > 1:
        return "both"
    if names:
        return names[0]
    return "normalized"


def _or_unknown(value: Any) -> str:
    """`?` for a limit the manifest does not state, matching the single-pass
    rows, which reach the same answer through `dict.get(key, "?")`."""
    return "?" if value is None else str(value)


#: The five limits §1 prints, and the label each carries, in its order.
#:
#: `server_cpuset`, `server_memory_bytes` and `build_threads` are pass-level
#: and engine-invariant by construction, so one row per pass states them
#: exactly. The buffer and graph-cache figures are NOT: `resolve_resources`
#: lets pgvector's buffer absorb the graph cache's share, and an in-memory
#: engine allocates neither, so on the tuned pass the same row differs by
#: engine. Measured on `full-20261015-215006`, whose host is 80 logical CPUs
#: (40 physical) and 187.54 GiB, and whose TUNED PASS is granted cpuset `0-75`
#: -- 76 CPUs -- and 112.52 GiB, that being `host_fraction: 0.6` of host RAM.
#: The buffer row is then 39.38 GiB for the MySQL family, 67.51 GiB for
#: pgvector and 0 for Valkey, being 0.35, 0.60 and 0 of that pass limit per
#: `config/resources/tuned.yml`.
#:
#: **Every figure here is binary.** They read 42.29 / 72.49 in decimal GB,
#: and an earlier version of this comment gave the limit in GiB and the
#: buffers in GB while asserting the ratio between them -- arithmetic that
#: only closes if the reader switches base mid-sentence. What §1 actually
#: prints is neither written form: `_fmt_bytes` renders that cell as
#: `39.4 GiB`.
#:
#: The pass limits are deliberately not called the host's capacity. An
#: earlier version labelled 76 CPUs and 112.5 GiB as "the host", in the one
#: sentence whose purpose is naming the machine; they are
#: `resolved_resources.server_cpu_count` and `server_memory_bytes`, which are
#: what the pass was given, and they scale with the host -- which is why the
#: run is named beside them.
#:
#: So those two rows are ONE engine's split, and -- the part that is new and
#: was glossed here before -- not necessarily the same engine's as before.
#: `config.resolved_resources` is whichever (pass, engine) `set_config` saw
#: last; a `config["passes"]` entry holds the FIRST engine of its pass,
#: because the accumulation is first-writer-wins so that a resume cannot
#: re-stamp it. A single-pass report and a two-pass one can therefore print
#: different buffer figures for the same pass. Both are a real resolved value
#: for that pass; neither is the pass's only one, and §1 is not the place that
#: claims otherwise -- the per-engine table below is.
_PASS_LIMIT_ROWS = (
    ("server_cpuset", "server cpuset", _or_unknown),
    ("server_memory_bytes", "server memory limit", _fmt_bytes),
    ("buffer_bytes", "buffer pool / shared_buffers", _fmt_bytes),
    ("graph_cache_bytes", "graph cache", _fmt_bytes),
    ("build_threads", "build threads", _or_unknown),
)


def _environment_rows(manifest: Dict[str, Any]) -> List[List[str]]:
    host = manifest.get("host", {}) or {}
    cpu = host.get("cpu", {}) or {}
    config = manifest.get("config", {}) or {}
    resolved = config.get("resolved_resources", {}) or {}
    # One block of limit rows per pass once a run measured more than one.
    # Printing a single pass's cpuset and memory limit above tables holding
    # another pass's rows is how `full-20261015-215006` published 8-core
    # normalized numbers under "cpuset 0-75, 112.5 GiB": the reader attributes
    # them to a 76-core machine, and nothing in the report contradicts that.
    per_pass = _pass_resources(manifest)
    multi = len(per_pass) > 1

    simd = ", ".join(cpu.get("simd_flags", []) or []) or "none detected"
    rows = [
        ["CPU", f"{cpu.get('model', '?')} — {cpu.get('logical_cpus', '?')} logical / "
                f"{cpu.get('physical_cores', '?')} physical cores"],
        ["SIMD", f"{simd}  (AVX-512: {'yes' if cpu.get('has_avx512') else 'no'})"],
        ["Hybrid cores", (f"yes — {len(cpu.get('performance_cpus', []))} performance / "
                          f"{len(cpu.get('efficiency_cpus', []))} efficiency"
                          if cpu.get("hybrid") else "no")],
        ["RAM", _fmt_bytes(host.get("total_ram_bytes"))],
        ["Kernel", host.get("kernel", "?")],
        ["Docker", host.get("docker_version", "?") or "?"],
        ["cgroup", host.get("cgroup_version", "?")],
        ["Resource pass",
         ", ".join(name for name, _r in per_pass) if multi
         else str(config.get("resource_pass", "?"))],
        # Which version of the ops measurement path produced these numbers.
        # The reason `OPS_MEASUREMENT_VERSION` exists at all is that a change
        # to `harness/` can alter what an ops number means without altering
        # the record's shape -- version 3 moved the concurrency query pool off
        # `ops.max_queries`, so a version 2 throughput and a version 3
        # throughput are the same column measured over different fractions of
        # the corpus. The report already refuses to put two versions on one
        # chart; until now it printed neither of them above the numbers, so a
        # reader of a PUBLISHED report could not tell which single version it
        # was. "not recorded" rather than "?" or 1, because an absent manifest
        # key means a run predating the field and the records themselves are
        # what say version 1 -- see `ops_measurement_version`.
        ["Ops measurement version",
         str(config.get("ops_measurement_version", "not recorded"))],
    ]
    if not multi:
        # Byte-for-byte what this table has always emitted for a run of one
        # pass. A single-pass report is not what was wrong, and changing its
        # wording would invalidate every published one for nothing.
        rows += [
            ["Server cpuset", str(resolved.get("server_cpuset", "?"))],
            ["Server memory limit", _fmt_bytes(resolved.get("server_memory_bytes"))],
            ["Buffer pool / shared_buffers", _fmt_bytes(resolved.get("buffer_bytes"))],
            ["Graph cache", _fmt_bytes(resolved.get("graph_cache_bytes"))],
            ["Build threads", str(resolved.get("build_threads", "?"))],
        ]
        return rows
    # One labelled block per pass, in the order the passes ran. The property
    # column carries the pass name because this table is two columns wide and
    # every row in it is a property of the run: a third column would have to
    # be added to, and left blank by, every row above.
    for name, limits in per_pass:
        for key, label, fmt in _PASS_LIMIT_ROWS:
            rows.append([f"{name} — {label}", fmt(limits.get(key))])
    return rows


def _engine_rows(manifest: Dict[str, Any]) -> List[List[str]]:
    rows = []
    for engine, info in sorted((manifest.get("engines", {}) or {}).items()):
        source = info.get("source", {}) or {}
        build = info.get("build", {}) or {}
        if source.get("kind") == "image":
            # No tag, no commit, no -march, because nothing was compiled. The
            # digest is the provenance. Printing "?" in those columns where
            # every other engine shows a commit reads as data we failed to
            # collect, and there is none to collect.
            digest = (source.get("mongot_digest") or source.get("server_digest")
                      or "")
            rows.append([
                _label(engine),
                source.get("version", "image"),
                digest.split("@")[-1][:19] if digest else "not pinned",
                "n/a (JVM)",
                "published image",
            ])
            continue
        rows.append([
            _label(engine),
            source.get("tag", "?"),
            (source.get("commit", "") or "")[:12] or "?",
            build.get("march", "?"),
            build.get("build_type", "?"),
        ])
    return rows


#: The character every table in this report is delimited by, and which no
#: cell may contain.
#:
#: It is a REFUSAL rather than an escape. `_md_tables_to_html` below splits a
#: row on this character to build the HTML page, so an escaped `\\|` would
#: render as two literal characters in the page while the Markdown looked
#: correct -- two renderers disagreeing about one table, which is the shape
#: this file has already shipped twice.
#:
#: The defect it exists for: PR 5 §5.2 names its tolerance column `|delta|`,
#: and writing that notation as a header cell put TWO extra pipes in the
#: header row. The header then had two more cells than every body row, the
#: HTML table came out shifted, and nothing in the suite noticed -- the only
#: cell-count assertion in the repo covers the concurrency table alone, and a
#: pipe added to a cell WITHOUT renaming it leaves every `header.index(...)`
#: lookup working. Measured: a ten-cell header over an eight-cell body with
#: the suite green at 1927 passed.
#:
#: **Headers raise; body cells are substituted**, because the two halves of
#: the table have different authors.
#:
#: A HEADER is written in this file, by a literal a few lines from the call.
#: A pipe there is a programming error, it is the same on every run, and
#: failing fast names the column. PR 5 §5.2's `|delta|` notation was exactly
#: that, and raising is safe because no run's data can produce it.
#:
#: A BODY cell can carry text this repository did not write:
#: `extra.maintenance` is whatever a driver's `maintain()` returned, a
#: driver's own sync note is free prose, and an engine label comes from the
#: manifest. (The sync note's key is deliberately not spelled here:
#: `tests/test_t4b_readers.py` counts occurrences of it across `report/` to
#: assert it is read NOWHERE, and a prose mention would make that count 1
#: while the claim it stands for stayed true. A count that cannot tell a
#: mention from a read is worth not disturbing.) Raising on one of those
#: would lose the entire report at the final
#: step of a run that may have taken twenty hours -- the failure
#: `harness/metrics/stats.bootstrap_ci`'s docstring refuses to risk for one
#: error bar. So the character is replaced and the row survives.
#:
#: Latent today: `postgres.maintain()` returns a pipe-free string and no
#: driver string in the tree contains one. Closed anyway, because closing it
#: now costs this comment and not closing it costs a report.
MD_TABLE_DELIMITER = "|"

#: What a pipe inside a body cell is rendered as. A BROKEN BAR, not an escape:
#: `_md_tables_to_html` below splits a row on the delimiter to build the HTML
#: page, so a backslash-escaped pipe would render as two literal characters
#: there while the Markdown looked right -- two renderers disagreeing about
#: one table. The broken bar reads as a pipe to a human and is not the
#: delimiter to either renderer, so the substitution is visible and loses
#: nothing a reader needs.
MD_TABLE_DELIMITER_SUBSTITUTE = "¦"


def _check_no_delimiter(cells, where: str) -> None:
    """Refuse a HEADER cell containing the table delimiter, naming the cell.

    Header cells only. `_sanitise_cell` handles the body, for the reason
    `MD_TABLE_DELIMITER`'s comment gives.
    """
    for cell in cells:
        if cell is not None and MD_TABLE_DELIMITER in str(cell):
            raise ValueError(
                f"{where} cell {str(cell)!r} contains the table delimiter "
                f"{MD_TABLE_DELIMITER!r}, which would split one cell into "
                f"several and leave this row a different width from the rest "
                f"of the table. Rename the column (PR 5 §5.2's `|delta|` "
                f"became `Recall gap`) or format the value without it.")


def _sanitise_cell(value) -> str:
    """One body cell as text, with any delimiter replaced rather than refused.

    Returns the em dash for the three empties `_md_table` has always treated
    as missing, so this is the single place a body value becomes a cell.
    """
    if value in (None, "None", ""):
        return "—"
    text = str(value)
    if MD_TABLE_DELIMITER in text:
        return text.replace(MD_TABLE_DELIMITER,
                            MD_TABLE_DELIMITER_SUBSTITUTE)
    return text


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    _check_no_delimiter(headers, "header")
    if not rows:
        return "\n_No data._\n"
    # Leading blank line: most Markdown renderers require one between a
    # paragraph and a table, and these tables always follow prose.
    out = ["", "| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_sanitise_cell(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _fmt_age(run_started: Any, measured_at: Any) -> str:
    try:
        delta = float(run_started) - float(measured_at)
    except (TypeError, ValueError):
        return "—"
    if delta < 3600:
        return f"{delta / 60:.0f} min"
    if delta < 86400:
        return f"{delta / 3600:.1f} h"
    return f"{delta / 86400:.1f} days"


#: Prose for each code `report/scope.py` files an excluded recall record
#: under. Keyed off that module's constants rather than off three retyped
#: literals, because a sentinel spelled in two files is a sentinel that
#: drifts, and the drift here would be silent: an unknown code falls through
#: to the generic phrase below and the row still renders.
#:
#: Four codes rather than one, and each sends a reader somewhere different.
#: In the reference bundle percona's own gist and glove curves were the
#: IN-SCOPE engine on an out-of-scope corpus; filing those under "engine"
#: would send a reader looking for an engine nobody ran. And a run can measure
#: an engine, measure a corpus, and never put the two together --
#: `new/full-20261015-215006` is percona over four corpora beside six engines
#: over fashion-mnist alone -- so that case needs saying too, or the row reads
#: as a contradiction of §1.
_OUT_OF_SCOPE_REASON = {
    _scope.OUT_OF_SCOPE_ENGINE: "this run did not measure this engine",
    _scope.OUT_OF_SCOPE_DATASET: "this run did not measure this corpus",
    _scope.OUT_OF_SCOPE_BOTH: "neither this engine nor this corpus",
    _scope.OUT_OF_SCOPE_PAIR: "not this engine on this corpus",
}


def _recall_pairs(summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The `(engine, corpus)` combinations this report's recall half carries.

    Read off `per_dataset`, which `summarize` builds one entry per
    `(dataset, pass, engine)` with at least one recall point -- so this is the
    set of pairs the charts and §4 actually draw, not a product derived from
    two separate lists.

    That distinction is the defect this function exists for. §2's opening
    sentence was built as "every engine in the report ON every corpus in the
    report", which is the PRODUCT of the axes, and the narrowing beside it is
    the phase log's PAIRS. On a run that measured percona over four corpora
    and six other engines over one -- `new/full-20261015-215006`'s shape --
    the product is 28 and the report carries 10, so the sentence claimed
    pgvector on gist two paragraphs above its own table row saying the run
    never measured pgvector on gist. The comment here used to assert that
    could not happen "because these come off the summary"; they did, and off
    the wrong two keys of it.
    """
    pairs = {(engine, dataset)
             for dataset, by_pass in (summary.get("per_dataset") or {}).items()
             for per_engine in (by_pass or {}).values()
             for engine in (per_engine or {})}
    return sorted(pairs)


def _covered_phrase(pairs: List[Tuple[str, str]]) -> str:
    """`pairs` as prose, grouped by engine, naming every one of them.

    Grouped only for length -- one clause per engine listing its corpora is
    the same set of pairs, rendered so that ten of them read as a sentence
    rather than as a list of twenty names. It must stay exhaustive: the whole
    point is that the reader can check it against the table below, and a
    truncated claim is the vague one this replaced.
    """
    if not pairs:
        return "none: no engine produced a recall measurement here"
    by_engine: Dict[str, List[str]] = {}
    for engine, dataset in pairs:
        by_engine.setdefault(engine, []).append(dataset)
    clauses = [f"{_label(engine)} on "
               + ", ".join(f"`{d}`" for d in sorted(datasets))
               for engine, datasets in sorted(by_engine.items())]
    count = (f"the {len(pairs)} engine/corpus combinations" if len(pairs) > 1
             else "the one engine/corpus combination")
    return f"{count} " + "; ".join(clauses)


def _widening_note(manifest: Dict[str, Any]) -> str:
    """How to bring an excluded combination into scope, per profile.

    Conditional, because the answer is not the same for all seven profiles
    and the unconditional version was false for six of them. `--engines` is
    refused only when the profile DECLARES an engine list:
    `config.engines_outside_profile` returns nothing when there is none --
    "there is nothing to be outside of" -- and `profile_engines`' own
    docstring records that every shipped profile but `percona` declares none.
    So `run --profile full --engines pgvector --datasets gist-960-euclidean`
    is accepted, and `full` is the profile behind the bundle this notice was
    written for. A sentence calling that refused would send an operator to
    edit a file they did not need to touch, in the one section of the report
    whose job is to be exact.
    """
    declared = (((manifest.get("config") or {}).get("profile") or {})
                .get("engines") or [])
    if not declared:
        return (" `--engines` and `--datasets` widen this profile, which "
                "declares no engine list of its own.")
    named = ", ".join(f"`{e}`" for e in declared)
    return (f" `--engines` will not do it: this profile declares {named} and "
            f"the orchestrator refuses a name outside that list, because a "
            f"grid, a corpus size and a set of resource extras belong to the "
            f"engines a profile names. Widening means editing the profile.")


def _inversion_likely(inversion: Dict[str, Any]) -> str:
    """`summarize`'s verdict; a summary written before it carried one falls
    back to the old rule, which was first step = warm-up."""
    return inversion.get("likely") or (
        "warm-up" if inversion.get("first_in_sweep") else "unexplained")


def _validity_section(manifest: Dict[str, Any], summary: Dict[str, Any],
                      records: Optional[List[Dict[str, Any]]] = None) -> str:
    parts: List[str] = []
    warnings = manifest.get("warnings", []) or []

    # Driver-declared facts the records carry: workloads an engine refused to
    # measure, and costs its own query path puts inside every timed query.
    # First, because both change how everything below reads.
    for note in (_disclosures.all_declined_lines(records or [], _label),
                 _disclosures.structural_lines(records or [], _label),
                 _disclosures.index_sync_lines(records or [], _label),
                 _disclosures.memory_lines(records or [], _label)):
        if note:
            parts.append(note)

    if summary.get("failed_phases"):
        parts.append(
            "### Phases that FAILED — these results are incomplete\n\n"
            "The run did not finish every unit it was asked to. Any engine or "
            "resource pass listed here is **missing from the comparisons below**, "
            "and its absence is a failure rather than a finding.\n"
        )
        rows = [[_label(f.get("engine", "?")), f.get("phase", "?"),
                 f.get("resource_pass", "?"), f.get("dataset", "?"),
                 str(f.get("exit_code", "?")),
                 f"{(f.get('duration_s') or 0) / 60:.1f} min"]
                for f in summary["failed_phases"]]
        parts.append(_md_table(
            ["Engine", "Phase", "Pass", "Dataset", "Exit code", "Ran for"], rows))

    if summary.get("refused_phases"):
        parts.append(
            "\n### Units refused before they ran\n\n"
            "Not failures: the orchestrator declined to start these because "
            "the engine's index build cannot fit the server memory limit of "
            "this run, and would have been killed mid-build after the load. "
            "The estimate is `harness/memory_fit.py`'s.\n\n")
        rows = [[_label(p.get("engine", "?")), p.get("phase", "?"),
                 p.get("resource_pass", "?"), p.get("dataset", "?"),
                 str(p.get("reason", "?"))]
                for p in summary["refused_phases"]]
        parts.append(_md_table(
            ["Engine", "Phase", "Pass", "Dataset", "Reason"], rows))

    if summary.get("silent_ann_failures"):
        parts.append(
            "\n### Recall phases that ran and measured nothing\n\n"
            "The engines below completed their recall phase and wrote **no "
            "measurements at all**. ann-benchmarks catches a per-algorithm "
            "exception and exits successfully, so a module that raises leaves "
            "the phase marked completed and the engine simply absent from the "
            "comparison below. That absence is a failure, not a finding, and "
            "the engine's other workloads may still have succeeded, which "
            "makes it easy to miss. The reason is in the run log for that "
            "phase, not in this report.\n"
        )
        rows = [[_label(f.get("engine", "?")), f.get("dataset") or "—",
                 f.get("resource_pass") or "—",
                 f"{(f.get('duration_s') or 0) / 60:.1f} min"]
                for f in summary["silent_ann_failures"][:20]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Pass", "Ran for"], rows))

    if summary.get("plan_failures"):
        parts.append(
            "### Measurements that did not use the vector index\n\n"
            "The following measurements ran with the vector index **absent from the "
            "query plan**. A full scan returns exact results slowly, so these appear "
            "in the data as high recall at low throughput. They are not ANN "
            "measurements and must not be read as such.\n"
        )
        rows = [[_label(f.get("engine", "?")), f.get("dataset", "?"),
                 f.get("phase", "?"), str(f.get("m", "—")),
                 str(f.get("ef_search", "—")),
                 # dict.get's default does not fire on an explicit None, which
                 # is what an unfiltered measurement stores — so "None" would
                 # leak into the table where "not applicable" is meant.
                 ("—" if f.get("selectivity") is None
                  else f"{f['selectivity']:.0%}")]
                for f in summary["plan_failures"][:40]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Phase", "M", "ef_search", "Selectivity"], rows))
        if len(summary["plan_failures"]) > 40:
            parts.append(f"\n_…and {len(summary['plan_failures']) - 40} more._\n")

    if "mongodb" in set(summary.get("engines") or []):
        parts.append(
            "\n### One engine is a Technical Preview\n\n"
            "**Percona Search for MongoDB is a Technical Preview and its own "
            "documentation says it is not for production use.** It is measured "
            "here beside four generally available engines. Treat a number for it "
            "as an indication of where the implementation currently stands "
            "rather than as a property of a shipped product, and expect it to "
            "move between releases in ways the others will not.\n"
        )

    # Appended only when it has something to say. `parts` being empty is what
    # selects the "no validity problems were detected" ending below, so a
    # blank string pushed on here would retire that sentence on every clean
    # run -- silently, because the section it replaces renders as nothing.
    scaling = _scaling_refused_section(summary)
    if scaling:
        parts.append(scaling)

    warmup = _index_warmup_section(summary)
    if warmup:
        parts.append(warmup)

    fixed = _fixed_ef_section(records or [])
    if fixed:
        parts.append(fixed)

    if summary.get("recall_floor_gaps"):
        parts.append(
            "\n### Recall floors an engine never approached\n\n"
            "`QPS @ recall≥F` is a comparison only where both engines have "
            "measurements near F. For each engine below, **every configuration "
            "swept returned recall above the floor**, so the figure reported at "
            "that floor is the same measurement as at every lower one, taken at "
            "a materially higher accuracy than the column asks for. This is a "
            "limit of the grid rather than a fault: `ef_search` cannot go below "
            "k, and where an engine exposes no `ef_construction` there is no "
            "setting that makes its index less accurate at a fixed M.\n"
        )
        rows = [[_label(g.get("engine", "?")), g.get("dataset", "?"),
                 f"≥{g['floor']:.2f}",
                 f"{g['lowest_recall']:.4f}",
                 f"{g['measured_at']:.4f}" if g.get("measured_at") is not None else "—",
                 _fmt(g.get("qps"), 0)]
                for g in summary["recall_floor_gaps"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Floor reported", "Lowest recall measured",
             "Figure taken at recall", "QPS"], rows))
        if len(summary["recall_floor_gaps"]) > 30:
            parts.append(
                f"\n_…and {len(summary['recall_floor_gaps']) - 30} more._\n")

    if summary.get("contended_phases"):
        parts.append(
            "\n### Phases that shared the machine\n\n"
            "Each sample records what the host burned as well as what the "
            "measured container did. Where the difference is more than two "
            "cores sustained, something outside the harness was running and "
            "the numbers from that phase describe the machine as much as the "
            "engine.\n"
        )
        rows = [[_label(c.get("engine", "?")), c.get("series", "?"),
                 f"{c.get('elapsed_s'):,.0f} s",
                 f"{c.get('container_cores')}",
                 f"**{c.get('foreign_cores')}**",
                 str(c.get("peak_load1") or "—")]
                for c in summary["contended_phases"][:30]]
        parts.append(_md_table(
            ["Engine", "Series", "Duration", "Our cores", "Foreign cores",
             "Peak load1"], rows))

    if summary.get("qps_inversions"):
        inversions = summary["qps_inversions"]
        severe = [i for i in inversions if _inversion_likely(i) != "warm-up"]
        parts.append(
            "\n### Throughput that rose as query effort rose\n\n"
            "Examining more candidates cannot be faster on the same index, so "
            "each row below is a measurement of something other than the "
            "engine — most often load on the host that the harness did not "
            "put there.\n\n"
            "**Read the ratio.** An inversion of roughly 1.5–2x at the "
            "*first* step of a sweep is the warm-up artifact: the opening "
            "configuration pays for a cold cache and the second looks faster. "
            "Anything larger, or anywhere other than the first step, is not "
            "explicable that way and the affected points should be "
            "re-measured rather than published. A sweep whose index was "
            "warmed before its first configuration has no cold first point, "
            "so a first-step inversion there is unexplained too.\n"
        )
        rows = [[_label(i.get("engine", "?")),
                 str(i.get("storage_engine") or "—"),
                 str(i.get("ef_construction") or "—"),
                 f"{i.get('from_ef')} → {i.get('to_ef')}",
                 f"{i.get('from_qps'):,} → {i.get('to_qps'):,}",
                 f"{i.get('ratio')}x",
                 ("warm-up" if _inversion_likely(i) == "warm-up"
                  else "**unexplained**")]
                for i in inversions[:30]]
        parts.append(_md_table(
            ["Engine", "Storage", "ef_construction", "ef_search",
             "QPS", "Ratio", "Likely"], rows))
        if len(inversions) > 30:
            parts.append(f"\n_…and {len(inversions) - 30} more._\n")
        if severe:
            worst = severe[0]
            parts.append(
                f"\n**{len(severe)} of these are not the cold-first-step artifact.** The "
                f"largest is {_label(worst['engine'])}, where ef_search "
                f"{worst['from_ef']} measured {worst['from_qps']:,} QPS and "
                f"ef_search {worst['to_ef']} measured {worst['to_qps']:,} — "
                f"{worst['ratio']}x faster for more work. Treat every point in "
                f"that engine's affected range as unmeasured.\n")

    if summary.get("duplicate_ann"):
        parts.append(
            "\n### The same configuration was measured more than once\n\n"
            "Every row below appeared twice or more **within one resource "
            "pass**, which means this report is reading more than one "
            "measurement tree for that pass — typically results from an older "
            "resource budget left in the tree beside the current ones. The "
            "charts pick the best value at each point, so those rows are a "
            "blend of two configurations rather than a measurement of either. "
            "Remove the stale directory under `results/annb/<pass>/` and "
            "regenerate. `--force` is not the remedy here: it recomputes "
            "within a tree, and what this section describes is a second tree "
            "being read alongside the current one.\n"
            # The remedy used to read "point the report at a single tree with
            # `--annb-results`, or delete the stale one", and both halves are
            # now wrong. The first is inert: the generator climbs back out of
            # a narrowed path on purpose, because a run of both passes writes
            # two trees and narrowing to one of them silently dropped 13.1
            # hours of measurement. The second invited deleting real results —
            # the rows this section used to list on a legitimate two-pass run
            # were that run's own two passes, 123 of them, and a reader who
            # followed the advice would have deleted a pass.
            "\n_Two resource passes are **not** a duplicate: they are two "
            "budgets, and they are keyed separately here and in every chart. "
            "A row appearing in this table names one pass measured twice._\n"
        )
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 c.get("resource_pass") or "—",
                 str(c.get("m", "—")), c.get("build_mode") or "—",
                 c.get("storage_engine") or "—",
                 str(c.get("ef_search", "—")),
                 str(c.get("count")), ", ".join(str(q) for q in c.get("qps", []))]
                for c in summary["duplicate_ann"][:30]]
        # The full unit, in the column order every other unit table in this
        # file uses. `build_signature` keys on the result filename, which
        # carries the storage engine and the build mode, so two rows here can
        # differ on either -- and printing neither made two configurations'
        # duplicates read as one configuration listed twice.
        parts.append(_md_table(
            ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage",
             "ef_search", "Copies", "QPS seen"], rows))
        if len(summary["duplicate_ann"]) > 30:
            parts.append(f"\n_…and {len(summary['duplicate_ann']) - 30} more._\n")

    if summary.get("stale_ann"):
        parts.append(
            "\n### Recall results that predate this run\n\n"
            "ann-benchmarks skips any configuration that already has a result "
            "file and reports that as success, so a re-run after a configuration "
            "change returns instantly with the **previous** numbers. The recall "
            "and throughput figures below were measured before this run started "
            "and do not reflect its resource limits or harness version. Re-run "
            "the ann phase with `--force` to recompute them.\n"
        )
        seen, rows = set(), []
        for c in summary["stale_ann"]:
            key = (c.get("engine"), c.get("dataset"))
            if key in seen:
                continue
            seen.add(key)
            rows.append([_label(c.get("engine", "?")), c.get("dataset", "?"),
                         _fmt_age(c.get("run_started"), c.get("measured_at"))])
        parts.append(_md_table(["Engine", "Dataset", "Measured before this run by"], rows))

    if summary.get("out_of_scope_ann"):
        # Deliberately NEXT to the staleness notice above, and deliberately
        # not merged into it. Both are facts about the same shared tree and
        # they need opposite remedies.
        #
        # A STALE row is a configuration this run asked for and ann-benchmarks
        # served from cache, because it skips any configuration that already
        # has a result file and reports that as success. `--force` recomputes
        # it, and in the reference bundle percona's own in-scope sweep was
        # exactly that: all 35 of its fashion-mnist files predate the run, 24
        # written on 2026-10-16 and 11 on 2026-10-20 against a run that
        # started on the 21st. The table above dedupes to one row per
        # (engine, corpus) and keeps the first it meets, which there was a
        # 10-16 file, so the published row read 5.0 days. A legitimate cache
        # hit, correctly flagged, and this notice must not take it over.
        #
        # A row HERE is a measurement of something this run never asked for.
        # There is no `--force` that makes MariaDB part of a percona-only run,
        # and offering one would send an operator to re-measure six engines to
        # fix a report that is now correct. Before this notice existed the
        # rows below appeared in the staleness table instead -- "MariaDB 12.3,
        # measured before this run by 1.8 days" -- which reads as a refreshable
        # cache hit and is the wrong instruction twice over.
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 str(c.get("files") or c.get("records") or 0),
                 _OUT_OF_SCOPE_REASON.get(c.get("reason"),
                                          "not part of this run")]
                for c in summary["out_of_scope_ann"][:30]]
        files = sum(c.get("files") or 0 for c in summary["out_of_scope_ann"])
        records_dropped = sum(c.get("records") or 0
                              for c in summary["out_of_scope_ann"])
        # The file count where there is one, the record count otherwise. A
        # record with no `extra.source_file` came from a hand-assembled or
        # third-party records file rather than out of the tree, and counting
        # it as a file would name a path nobody can go and look at.
        held = (f"**{files} result files**" if files != 1
                else "**1 result file**")
        if not files:
            held = (f"**{records_dropped} recall records**"
                    if records_dropped != 1 else "**1 recall record**")
        covered = _recall_pairs(summary)
        covers = _covered_phrase(covered)
        parts.append(
            "\n### Recall results from other runs in the shared tree\n\n"
            "Recall results live in a tree shared across runs at "
            "`results/annb/<pass>/<fingerprint>/`. That is deliberate: the "
            "fingerprint keys on the resource pass's declared limits and is "
            "engine-invariant, so every engine measured at one budget lands "
            "in one tree and one chart compares them. The cost is that the "
            "tree is not a record of any single run.\n"
            f"\nThe recall results in this report are {covers}. The tree "
            f"also held {held} for the combinations below, which this run did "
            f"not measure, and **they are excluded from every chart and table "
            f"in this report** and from the `records.jsonl` beside it.\n"
            "\n_This is not the staleness notice above, and `--force` is not "
            "the remedy. A stale result is a measurement this run asked for "
            "and ann-benchmarks served from cache. A row below was never part "
            "of this run at all, so there is nothing here to recompute: to "
            "report on it, report the run that measured it, or measure it "
            f"here.{_widening_note(manifest)}_\n"
        )
        parts.append(_md_table(
            ["Engine", "Dataset", "Result files",
             "Why it is not part of this run"], rows))
        more = len(summary["out_of_scope_ann"]) - 30
        if more > 0:
            parts.append(f"\n_…and {more} more._\n")

    # Only when there is something to say: `if not parts` below picks the
    # "no validity problems" ending, and an appended empty string retires it.
    provenance_notices = _provenance.validity_notices(manifest, summary)
    if provenance_notices:
        parts.append(provenance_notices)

    if summary.get("passes_without_a_tree"):
        # The notice that has to be IN the document. `full-20261015-215006`
        # ran both passes, recorded one pass's `ann_fingerprint`, and
        # published a report whose §1 stated one machine above 52 table rows
        # measured on another -- with nothing in the document to contradict
        # them. The generator said so on stdout; stdout is not what anybody
        # reads, and `export` does not bundle it.
        # Not escaped here: this function emits Markdown, and the HTML page
        # runs every line through `_inline_md`, which escapes first and then
        # applies the backtick and bold markup. Escaping twice prints
        # `&amp;#x27;` on the page and a literal entity in the Markdown.
        orphans = summary["passes_without_a_tree"]
        named = ", ".join(f"`{p}`" for p in orphans)
        many = len(orphans) > 1
        # Spelled out rather than glued together from a suffix: "pass" + "es"
        # + "'s" produced "those passes's recall measurements" in the first
        # version of this notice.
        subject = f"the {named} resource passes" if many \
            else f"the {named} resource pass"
        them, their = ("them", "those passes'") if many \
            else ("it", "that pass's")
        parts.append(
            f"\n### Recall measurements this report could not locate\n\n"
            f"This run executed {subject}, and the manifest records no "
            f"`ann_fingerprint` for {them}. Recall results live in a tree "
            f"shared across runs at `results/annb/<pass>/<fingerprint>/`, "
            f"keyed by that string, so **{their} recall and throughput "
            f"measurements are absent from every chart and table below**. "
            f"The ops measurements for {them} are unaffected and are "
            f"present. Nothing here guesses at the tree: it is shared across "
            f"runs, so a directory under that pass may belong to a different "
            f"resource configuration entirely.\n\n"
            f"A manifest predating per-pass recording can be completed "
            f"without re-measuring anything — "
            f"`./run-benchmark.sh run --profile <p> --resume --run-id "
            f"{manifest.get('run_id', '<run-id>')}` "
            f"records every pass's fingerprint and skips every finished "
            f"unit — after which this report regenerates complete.\n")

    if summary.get("memory_pressure"):
        parts.append(
            "\n### Engines that ran against their memory limit\n\n"
            "These phases spent part of their run at the container memory limit, "
            "which means the kernel was reclaiming continuously. Throughput and "
            "latency for them describe **the budget they were given rather than "
            "the implementation**, and they are not comparable with an engine that "
            "had headroom. Raise `memory.server_limit_gb` for this corpus and "
            "re-run before drawing any conclusion from these numbers.\n"
        )
        rows = [[_label(c.get("engine", "?")),
                 _fmt_bytes(c.get("peak_bytes")),
                 _fmt_bytes(c.get("limit_bytes")),
                 f"{c.get('fraction_at_ceiling', 0):.0%}",
                 f"{c.get('first_hit_fraction', 0):.0%} in"]
                for c in summary["memory_pressure"]]
        parts.append(_md_table(
            ["Engine", "Peak", "Limit", "Time at the limit", "First reached"], rows))

    if summary.get("short_result_cases"):
        parts.append(
            "\n### Filtered queries that returned fewer than k results\n\n"
            "With post-filtering, an engine can exhaust its candidate list before "
            "finding k qualifying rows. This is real engine behaviour, not a harness "
            "fault, but it means the recall figure for these points is computed over "
            "short result sets and should be read alongside the count.\n"
        )
        # Build mode and pass are what tell these apart. pgvector is measured
        # in two build modes across two passes, so four real measurements
        # rendered as the same row four times and read as a rendering fault.
        # The column joined them in PR 3 for the same reason: `tag` at 10% and
        # `ctag` at 10.85% otherwise differ only by a selectivity a reader
        # cannot attribute to a predicate.
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 c.get("resource_pass") or "—", c.get("build_mode") or "—",
                 c.get("filter_column") or "tag",
                 f"{(c.get('selectivity') or 0):.2%}",
                 str(c.get("queries", "—"))]
                for c in summary["short_result_cases"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Pass", "Build mode", "Column",
             "Selectivity", "Queries short of k"], rows))

    if summary.get("time_bounded_points"):
        parts.append(
            "\n### Filtered points the time budget stopped\n\n"
            "`ops.filtered_budget_s` caps each (column, selectivity, "
            "iterative-scan mode) point. These points spent it before "
            "answering every planned query, so their recall and latency are "
            "computed over the queries that finished and carry that sample's "
            "variance rather than the full query set's. **This is a finding "
            "about the configuration, not a harness fault** — a point that "
            "cannot finish 500 queries in five minutes is running below 1.7 "
            "QPS, which is itself the measurement. Points marked ✕ completed "
            "fewer than 50 queries and are drawn greyed with a cross on the "
            "filtered chart; □ points completed at least 50.\n"
        )
        rows = [[_label(c.get("engine", "?")), c.get("dataset", "?"),
                 c.get("resource_pass") or "—", c.get("build_mode") or "—",
                 c.get("filter_column") or "tag",
                 f"{(c.get('selectivity') or 0):.2%}",
                 f"{c.get('queries_executed', '—')}/"
                 f"{c.get('queries_planned', '—')}",
                 f"{c.get('budget_s', '—')}",
                 "✕ under 50" if c.get("insufficient_sample") else "□"]
                for c in summary["time_bounded_points"][:30]]
        parts.append(_md_table(
            ["Engine", "Dataset", "Pass", "Build mode", "Column",
             "Selectivity", "Queries done", "Budget (s)", "Sample"], rows))

    if warnings:
        parts.append("\n### Environment warnings\n\n")
        for w in warnings:
            parts.append(f"- {w}\n")

    if not parts:
        parts.append(
            "No validity problems were detected: every measurement used the vector "
            "index, no filtered query returned short, and the environment raised no "
            "warnings.\n"
        )
    return "".join(parts)


def _fixed_ef_section(records: List[Dict[str, Any]]) -> str:
    """Validity: throughput measured at one ef_search, not at one recall.

    `ef_mode: fixed` gives every engine the same `ef_search`, which is the
    same *intention* and not the same accuracy -- two engines at ef 100 can
    sit at recall 0.98 and 0.80, and the faster one may simply be doing less
    work per query. §6 says so for the operating-point table; a reader who
    starts at §7's throughput needs it in Validity, with the ef named.
    """
    from report.operating import is_fixed_ef
    # `is_fixed_ef`, not `ef_mode == "fixed"`: records from before ops
    # measurement version 4 carry no `ef_mode` and WERE fixed -- the 25 Aug
    # run is one, and it is the run this section was written for.
    fixed = [r for r in records
             if r.get("phase") in ("concurrency", "filtered", "churn")
             and is_fixed_ef(r)]
    if not fixed:
        return ""
    efs = sorted({r.get("ef_search") for r in fixed
                  if r.get("ef_search") is not None})
    named = ", ".join(str(ef) for ef in efs) or "unrecorded"
    return (
        "\n### Throughput at a fixed ef_search, not at a fixed recall\n\n"
        f"This run measured at a fixed ef -- `ef_mode: fixed`, or a harness "
        f"from before ops measurement version 4, which had no other mode and "
        f"records no `ef_mode`: every engine's concurrency, "
        f"filtered and churn figures were measured at ef_search = {named}, "
        f"whatever recall that bought each engine. Throughput is therefore "
        f"**not compared at equal recall** in §7-§9 -- an engine with a "
        f"higher QPS there may be answering a cheaper, less accurate query. "
        f"The Recall@k column beside each figure is the measure of how far "
        f"apart they were; `ef_mode: calibrated` holds every engine to one "
        f"recall target instead.\n")


def _scaling_refused_section(summary: Dict[str, Any]) -> str:
    """Validity: the scaling efficiencies this report refuses to publish.

    In Validity and not only in the concurrency footnote, because this is the
    section a reader is told to read before the charts and because the
    concurrency chart is the headline of the whole report. A caveat that lives
    only under the table it qualifies is a caveat read after the conclusion
    has been drawn.

    The row carries what was served against what was offered, the throughput
    that is still a real measurement, and the client CPU — which is the one
    figure that separates the two causes. Named per point rather than per
    engine: a pool can serve its concurrency at four clients and lose it at
    thirty-two, and "this engine is affected" would hide which half of its
    curve is readable.
    """
    refused = _scaling_refused(summary)
    if not refused:
        return ""
    parts = [
        "\n### Scaling efficiencies this report will not publish\n\n"
        "`Scaling efficiency` divides an N-client throughput by N times a "
        "one-client throughput. **That is a statement about the engine only "
        "if N queries were actually in flight**, and at the points below "
        "they were not: either the clients had fewer than half their queries "
        "outstanding on average — the sum of every in-window query's "
        "duration over the window, Little's law with no estimator in it — or "
        "neither the median estimate nor the throughput shows the queries "
        "being served at once. The efficiency figure there is not a "
        "measurement of how this engine scales, so the table prints "
        f"*{SCALING_REFUSED_CELL}* in that cell and the chart's efficiency "
        "panel omits the point. **The QPS and the latency percentiles are "
        "left in place and are real**: the pool did answer that many queries "
        "at those latencies. It is the division by the client count that has "
        "no basis.\n\n"
        "**Since `ops_measurement_version` 9 the mean-based figure is the "
        "primary one.** Until then this section was keyed to the MEDIAN "
        "estimate — queries × p50 ÷ window — because, while the clients were "
        "Python threads, a client waiting for the interpreter lock still "
        "counted as in flight and the mean was blind to it. The clients have "
        "been separate OS processes since version 6, so that reason is gone, "
        "and the median kept only its own bias: under a heavy tail it "
        "undercounts by p50 ÷ mean (MariaDB 12.3 at 32 clients on "
        "`full-20260825`, p50 34 ms against p99 1,043 ms). A skewed service "
        "time is therefore no longer a refusal; it is noted "
        f"`tail_dominated` ({TAIL_DOMINATED_MARK}) beside a published figure. "
        "What remains here is a pool whose clients did not keep their "
        "queries in flight — a fault in this harness, the case the guard is "
        "kept as a tripwire for — or an engine that answers one query at a "
        "time with a tail, which a client clock cannot tell apart from "
        "clients taking turns. Read the Client CPU column with it. Records "
        "from before version 9 were judged by the median rule and keep that "
        "verdict.\n",
    ]
    rows = []
    for r in sorted(refused, key=lambda r: (r.get("dataset") or "",
                                            r.get("engine") or "",
                                            r.get("clients") or 0)):
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            _storage(r), str(r.get("clients", "—")),
            _fmt(extra.get("outstanding_concurrency"), 2),
            _fmt(extra.get("served_concurrency"), 2),
            _fmt(r.get("qps"), 0),
            # The figure that is not published, shown here and nowhere else.
            # Withholding it from the table and then hiding it entirely would
            # make the refusal uncheckable; this is the one place a reader can
            # see what was refused and why.
            _fmt(extra.get("scaling_efficiency"), 2),
            _fmt(extra.get("client_cpu_seconds"), 1, " s"),
        ])
    parts.append(_md_table(
        ["Engine", "Dataset", "Storage", "Clients", "In service (mean)",
         "Median estimate", "QPS", "Efficiency withheld", "Client CPU"],
        rows))
    parts.append(
        "\n**In service (mean)** is the exact figure the refusal reads since "
        "version 9: the sum of every in-window query's duration over the "
        "window. It counts any wait as flight — from the client's clock a "
        "query it is blocked on is in flight whatever it is blocked on — "
        "which is why it could not be the guard while the clients shared one "
        "interpreter lock, and can be now. **Median estimate** is queries × "
        "p50 ÷ window, the figure the guard used before version 9; the two "
        "differ by exactly p50 ÷ mean, so a median estimate far under the "
        "mean says the latency had a tail and nothing more.\n")
    return "".join(parts)


def _index_warmup_section(summary: Dict[str, Any]) -> str:
    """Validity: ann sweeps whose one-time index warm-up did not converge.

    `summary["index_warmup_flags"]` (`report/generate.index_warmup_flags`):
    a warm-up that hit its cap without two batches agreeing, or that ran
    fewer than two batches. Either way the sweep's first configuration may
    have met the cold cache the warm-up exists to absorb, and its QPS is
    then the cache's, not the engine's.
    """
    flags = summary.get("index_warmup_flags") or []
    if not flags:
        return ""
    rows = [[_label(f.get("engine") or "?"), f.get("dataset") or "?",
             f.get("resource_pass") or "—", f.get("build") or "—",
             _fmt(f.get("queries"), 0),
             {True: "yes", False: "**no**"}.get(f.get("converged"), "—"),
             _fmt(f.get("seconds"), 1, " s")]
            for f in flags]
    return (
        "\n### Index warm-ups that did not settle\n\n"
        "Each ann sweep warms its index before the first configuration, "
        "querying until two consecutive 200-query batches agree to within 5% "
        "or 5,000 queries have run. The sweeps below either reached that cap "
        "without agreeing or ran fewer than 400 queries — two batches, the "
        "least that can be compared — so the **first configuration of each "
        "may still have been measured against a cold cache**, and a QPS "
        "rise from it to the next point is that artefact rather than the "
        "engine.\n"
        + _md_table(["Engine", "Dataset", "Pass", "Build",
                     "Warm-up queries", "Converged", "Warm-up time"], rows))


def _known_asymmetries(summary: Optional[Dict[str, Any]] = None) -> str:
    """Structural differences no configuration removes.

    Conditional on what actually ran. A row about a separate index process in a
    report of four in-engine indexes describes nothing, and a reader has to
    work out that it does not apply.
    """
    engines = set((summary or {}).get("engines") or [])
    # Which engines expose the build-quality knob is a property of the RUN and
    # of the drivers in it, never of a literal here. It was pgvector alone
    # until valkey-search joined, and the `{"pgvector", "valkey"}` literal
    # that replaced that sentence was wrong again the day Percona Server
    # landed: a Percona-only run intersected to nothing, fell through to the
    # single-engine arm and published "exposed only by pgvector" in a report
    # about the one engine here with BOTH a build-time and a query-time
    # effort knob -- contradicting `CLAUDE.md`'s invariants bullet, which
    # names all three, and `config/engines/percona.yml`, which declares the
    # capability. The discriminator needs no new channel:
    # `harness/workloads/build.py` merges `_driver_capabilities(driver)` into
    # every `index_build` record's `extra`, so `extra.ef_construction_tunable`
    # is the driver's own answer about the engine that actually ran -- and
    # reading it from the records keeps `report/` importing nothing from
    # `orchestrator`, which `tests/test_config.py` enforces.
    tunable = _ef_construction_engines(summary)
    # The same correction one row down. "pgvector bulk-builds after load;
    # MHNSW and VIDX build incrementally" was unconditional, so a
    # Percona-only report spent a row on three engines that did not run and
    # said nothing about the one that did -- the only engine in the MySQL
    # family whose `build_mode` is a comparison within itself. Derived from
    # `extra.separable_build` through `generate.build_kind`, the classifier
    # that §5's table and the build chart already share, so this row and the
    # figures below it cannot disagree about what a build was.
    rows = [
                _ef_construction_row(tunable, sorted(engines)),
                *_build_mode_rows(summary),
                ["AliSQL VIDX is InnoDB-only and requires READ COMMITTED",
                 "InnoDB is the headline storage engine for all engines. MariaDB's "
                 "MyISAM results, where present, are a MariaDB-only extra curve."],
                ["Client stacks differ for PostgreSQL",
                 "MariaDB and AliSQL share one client (MariaDB Connector/C) so their "
                 "numbers are mutually comparable. PostgreSQL necessarily uses "
                 "psycopg3; its client overhead differs by an unmeasured amount."],
                ["SIMD paths depend on the host CPU",
                 "Both MHNSW and VIDX document AVX-512 distance kernels. On a host "
                 "without AVX-512 both run narrower paths, and the ranking may differ "
                 "on AVX-512 hardware."],
    ]

    if "valkey" in engines:
        rows += [
            ["Valkey holds the whole dataset in memory",
             "The other engines keep a disk-backed table with a cache in front of "
             "it, so their container limit is a cache budget. Valkey's is the "
             "dataset. It has no buffer pool to size and no index on disk, and "
             "its index size below is resident memory rather than a file."],
            ["Valkey applies the predicate during the graph walk",
             "Its documentation describes a per-query planner that can "
             "pre-filter instead. This run did not observe that: its latency "
             "rises when the predicate narrows, which is what walking the "
             "graph costs and the opposite of what pre-filtering costs. The "
             "filtered section derives each engine's strategy from that "
             "measurement rather than from its documentation."],
            ["Valkey is installed from packages, not built from source",
             "valkey-search ships prebuilt, so it carries installed package "
             "versions instead of a tag, a commit and a `-march`, and the "
             "AVX-512 row above does not describe it."],
        ]

    if "mongodb" in engines:
        rows += [
            ["Percona Search keeps its index in a separate process",
             "mongod holds the documents and mongot holds a Lucene index fed from "
             "the change stream, where the other four keep the index inside the "
             "query engine. What is being compared includes where the index lives, "
             "not only how it is implemented."],
            ["Percona Search is not built from source",
             "It ships as packages and images only and runs on the JVM, so it "
             "carries an image digest instead of a tag, a commit and a `-march`. "
             "Its distance kernels come from the JVM's vector API, so the AVX-512 "
             "row above does not describe it."],
            ["Percona Search exposes no graph degree",
             "M is pinned at one value for every other engine and cannot be set "
             "here at all, so any row carrying an M for it is naming a sweep "
             "rather than a setting, and those comparisons are not at matched M."],
            ["Percona Search filters before the vector comparison",
             "Its `filter` is a pre-filter; the other four apply the predicate "
             "after the graph walk. That is why their filtered results either "
             "collapse in throughput or return fewer than k rows, and why a "
             "difference in the filtered section is a difference in kind."],
            ["Percona Search builds its index asynchronously",
             "createSearchIndex returns before the index exists and it is built "
             "while writes are still arriving, so its build cost is wall clock to "
             "a queryable index rather than the duration of a statement."],
        ]

    return (
        "These are structural differences between the engines that no configuration "
        "removes. They are why some comparisons below are presented in pairs rather "
        "than as a single ranking.\n\n"
        + _md_table(["Asymmetry", "Consequence for these results"], rows)
    )


def _headline_tables(summary: Dict[str, Any]) -> str:
    """§4's tables: one per (dataset, resource pass).

    The pass is in the heading rather than in a column, and that is a
    deliberate difference from §7-§10, which do carry a Pass cell. Their rows
    are individual measurement windows, so the pass identifies which machine
    one window ran on. A row here is a derived frontier: the floor columns,
    `Recall range` and `Points measured` are all claims taken across the whole
    point set, so the pass scopes the entire table. A scope that belongs to the
    table belongs in its heading, where the eye cannot drop it -- as a cell, an
    engine measured in both passes becomes two rows with the same label and the
    qualifier has to be carried by hand. The prose under the table is written
    in the singular about "the swept grid" and closes "this run pins M=16 for
    every engine", which stays true per table and becomes a per-row read under
    a column.

    The cross-pass comparison this does not give is `charts.pass_comparison`'s,
    which exists because that question "had to be reconstructed by hand from
    two tables" and whose first dimension is recall throughput.
    """
    parts: List[str] = []
    per_dataset = summary.get("per_dataset", {}) or {}
    # Named only when the RUN has more than one, so a single-pass report keeps
    # the heading `#### <dataset>` it has always had, byte for byte.
    #
    # Off `described_passes`, not off `per_dataset`: the latter holds only the
    # passes that produced recall for a dataset, so when one pass's tree was
    # missing or unnarrowable §1 said "normalized, tuned" and §4 printed a bare
    # heading over one of them -- and a reader took that table for the other
    # pass's. One number, computed once in `summarize`.
    #
    # NOT called `multi`: the engine loop below already binds that name for
    # "this engine was swept over more than one storage engine", so an outer
    # `multi` is shadowed from the first engine onward. It rendered the first
    # pass's heading correctly and every later one bare, which the §4 heading
    # test caught and inspection did not.
    many_passes = len(described_passes(summary)) > 1
    # Lazily: `frontier_render` imports this module's cell helpers at its
    # own import time, and this module is the one `generate` imports first.
    from report import frontier_render
    for dataset, by_pass in sorted(per_dataset.items()):
        for resource_pass, per_engine in sorted(
                by_pass.items(), key=lambda kv: str(kv[0])):
            heading = (f"{dataset} — {resource_pass}" if many_passes
                       else dataset)
            parts.append(f"\n#### {heading}\n\n")
            rows = []
            for engine, entry in sorted(per_engine.items()):
                # The frontier spans every configuration the engine was swept over,
                # so on the tuned pass MariaDB's headline figures are MyISAM ones.
                # Naming the winner keeps the number from being read as the result
                # of the default build.
                multi = len(entry.get("storage_engines") or []) > 1

                def cell(floor: int) -> str:
                    value = _fmt(entry.get(f"qps_at_recall_{floor}"), 0)
                    storage = entry.get(f"qps_at_recall_{floor}_storage")
                    if multi and storage and value != "—":
                        value = f"{value} ({storage})"
                    # The recall the figure was actually taken at, on every cell.
                    # A frontier reports the fastest configuration at or above the
                    # floor, and that configuration is often well above it -- so
                    # two floors can be served by one measurement and the columns
                    # come out identical. MariaDB reads 415 at both 0.90 and 0.95
                    # because its fastest setting already returns 0.9563, and it
                    # has three slower measurements in between. Without the recall
                    # beside it, equal columns read as a missing measurement.
                    at = entry.get(f"qps_at_recall_{floor}_recall")
                    if value != "—" and at is not None:
                        value += f" @{at:.3f}"
                    # An engine with no configuration below the floor has no
                    # measurement AT the floor either -- the best it can offer is
                    # the one above it. MariaDB 12.3's lowest setting returns
                    # recall 0.9784, so its "QPS at recall 0.90" is really its QPS
                    # at 0.9888, and printing a bare number invites a comparison
                    # with engines that do have a point near 0.90. The recall the
                    # figure was actually taken at travels with it.
                    # A bare marker now: the recall it was taken at is already on
                    # the cell, so repeating it beside a dagger said the same
                    # number twice.
                    lowest = entry.get("min_recall")
                    if (value != "—" and at is not None and lowest is not None
                            and lowest > floor / 100.0):
                        value += " †"
                    # The headline is the fastest build swept, which
                    # may not be the index §5 onward measured. Said beside
                    # the number, because that is where it gets copied from.
                    if value != "—":
                        value += frontier_render.headline_mark(entry, floor)
                    return _with_build(value, entry.get(
                        f"qps_at_recall_{floor}_build_label"))

                # The range, not only the best. Two identical floor columns mean
                # the engine has no measurement between them, and the lowest recall
                # it reached is the number that says so.
                span = _recall_span(entry.get("min_recall"),
                                    entry.get("max_recall"))
                rows.append([
                    _label(engine),
                    cell(90), cell(95), cell(99),
                    span,
                    str(entry.get("points", 0)),
                ])
            parts.append(_md_table(
                ["Engine", "QPS @ recall≥0.90", "QPS @ recall≥0.95",
                 "QPS @ recall≥0.99", "Recall range", "Points measured"],
                rows,
            ))
            parts.append(
                "\n_QPS at a recall floor is the comparison an operator makes: how fast "
                "is the engine at an accuracy I can accept, over every build this run "
                "swept. The build that produced each figure is printed under it. A "
                "dash means no build swept reached that recall._\n"
                "\n_Each cell also carries the recall the figure was measured at. "
                "A frontier reports the fastest configuration at or above the "
                "floor, and that configuration is frequently well above it, so two "
                "columns showing the same number are one measurement winning both "
                "— not a missing one. MariaDB reads the same at 0.90 and 0.95 "
                "because its fastest setting already returns 0.956, with three "
                "slower measurements in between._\n"
                "\n_A **†** marks a figure the engine has no measurement for at that "
                "floor: every configuration swept returned recall above it, so the "
                "number shown is its throughput at the higher recall printed beside "
                "the dagger. Those cells are not comparable with engines that do have "
                "a point near the floor — they are taken at a materially harder task. "
                "A lower graph degree M or build effort is what would produce points "
                "there; the swept-space table below says what this run varied._\n"
            )
            parts.append(frontier_render.mismatch_notes(per_engine))
            parts.append(frontier_render.swept_space_table(per_engine))
            parts.append(frontier_render.smallest_table(per_engine))
            parts.append(frontier_render.build_cost_table(per_engine))
            parts.append(frontier_render.ops_build_table(per_engine,
                                                         FLOOR_MARK))
            parts.append(_reference_build_notes(per_engine))
    return "".join(parts) or "_No recall/QPS data in this run._\n"


def _with_build(value: str, label: Optional[str]) -> str:
    """A floor figure with the build it came from on the line below it.

    `full-20260825` printed Percona Search's 138 with nothing to say it came
    from the unquantized index while every other figure for the engine came
    from the scalar one. A number that names its build cannot be mistaken
    for another build's.
    """
    if value == "—" or not label:
        return value
    return f"{value}<br>`{label}`"


def _frontier_intro(summary: Dict[str, Any]) -> str:
    """§4's opening sentence. It said "Pareto frontier" for every engine,
    including those measured at one build with only ef_search varied, which
    is a curve; `frontier_render.section_intro` words it per run."""
    from report import frontier_render
    return frontier_render.section_intro(summary)


def _recall_span(low: Optional[float], high: Optional[float]) -> str:
    if high is None:
        return "—"
    return f"{low:.4f} to {high:.4f}" if low is not None else f"{high:.4f}"


def _reference_build_notes(per_engine: Dict[str, Any]) -> str:
    """Why a row of the ops-build table holds dashes.

    One sentence per engine whose reference build left a cell empty, giving
    the build's own ceiling: "its reference build peaks at recall 0.892" is
    what the 25 Aug post needed beside Percona Search, and printed 138 from
    another index instead. The headline now does take the fastest build --
    named, and marked when it is not the ops build -- so these notes explain
    the secondary table rather than the headline.
    """
    lines = []
    for engine, entry in sorted(per_engine.items()):
        build = entry.get("reference_build_label") or "not recorded"
        if not any(entry.get(f"reference_qps_at_recall_{f}") is None
                   for f in (90, 95, 99)):
            continue
        source = entry.get("reference_source") or ""
        if entry.get("reference_build") is None and source.startswith(
                "ambiguous"):
            lines.append(
                f"- **{_label(engine)}**: the ops build is ambiguous "
                f"-- {source.split(': ', 1)[-1]} -- so this table prints no "
                f"figure rather than one pooled across them.\n")
        elif entry.get("reference_build") is None:
            lines.append(
                f"- **{_label(engine)}**: no ops build is recorded for this "
                f"run and the grid swept {entry.get('sweep_builds', 0)} "
                f"builds, so this table prints no figure rather than guess "
                f"which of them the ops pass measured.\n")
        elif entry.get("reference_max_recall") is None and any(
                value is None for value in entry["reference_build"].values()):
            # Valkey on the 25 Aug run: the ops pass left efC to the server,
            # so whether an ann build is the same index cannot be said --
            # which is not the same as "nothing was measured at it".
            lines.append(
                f"- **{_label(engine)}**: its ops build (`{build}`) left a "
                f"setting to the server default, which no record names, so "
                f"no swept build can be matched to it and every floor here "
                f"reads \"—\".\n")
        elif entry.get("reference_max_recall") is None:
            lines.append(
                f"- **{_label(engine)}**: no recall point was measured at its "
                f"ops build (`{build}`), so every floor here reads \"—\".\n")
        else:
            lines.append(
                f"- **{_label(engine)}**: its ops build (`{build}`) peaks at "
                f"recall {entry['reference_max_recall']:.3f}; a \"—\" here "
                f"means that build has no point at or above the floor, "
                f"whatever another build reached.\n")
    return (_chosen_build_notes(per_engine)
            + ("\n" + "".join(lines) if lines else ""))


#: How `report/builds.py` prefixes a source the ops pass chose per dataset
#: under `ops.build_selection: frontier`.
_CHOSEN_PREFIX = "manifest: "
_CHOSEN_WORDS = {
    "frontier": "chosen from this run's frontier as the build fastest at "
                "recall {target}",
    "below_target": "chosen from this run's frontier, but NO swept build "
                    "reached recall {target}; this is the one with the "
                    "highest recall",
    "fallback_reference": "the declared reference build, measured because "
                          "the ann phase left no frontier to choose from "
                          "(see the run's warnings)",
}


def _chosen_build_notes(per_engine: Dict[str, Any]) -> str:
    """One line per engine whose ops build was chosen rather than declared.

    `ops.build_selection: frontier` picks the ops build off the ann phase,
    so the reference build differs per engine and dataset and may be a
    fallback; a reader has to be told which, not left to assume the
    declared build.
    """
    lines = []
    for engine, entry in sorted(per_engine.items()):
        source = str(entry.get("reference_source") or "")
        if not source.startswith(_CHOSEN_PREFIX):
            continue
        how = source[len(_CHOSEN_PREFIX):]
        kind, _, target = how.partition("@")
        words = _CHOSEN_WORDS.get(kind, how).format(target=target or "?")
        build = entry.get("reference_build_label") or "not recorded"
        lines.append(f"- **{_label(engine)}**: `{build}` -- {words} "
                     f"(`{how}`).\n")
    if not lines:
        return ""
    return ("\n_The ops build was chosen per engine from this run's ann "
            "frontier (`ops.build_selection: frontier`) rather than declared; "
            "how each engine's was chosen:_\n\n" + "".join(lines))


# The ops harness stamps a storage engine on every unit and defaults to
# InnoDB, which is meaningless for PostgreSQL. Its own ann records call the
# same thing `heap`; this keeps the two paths from disagreeing on the page.
_STORAGE_OVERRIDES = {"pgvector": "heap", "mongodb": "wiredTiger",
                      "valkey": "memory"}


def _storage(record: Dict[str, Any]) -> str:
    override = _STORAGE_OVERRIDES.get(record.get("engine"))
    return override or record.get("storage_engine") or "—"


def _build_table(summary: Dict[str, Any]) -> str:
    # Ingest rate lives on the `ingest` record, not on `index_build`; reading it
    # off the build record produced an em dash in every row.
    #
    # **This key must name every axis `_run_unit` loops, because each one it
    # omits silently publishes one unit's rate beside another unit's build
    # time.** `storage_engine` was added first: MariaDB's InnoDB and MyISAM
    # builds collided and one rate was printed for both rows, which reported
    # MyISAM's 210 rows/s as 80 on 11.8 and its 305 as 51 on 12.3.
    #
    # `m` is the same bug one axis over. It cannot show at all until one
    # (engine, dataset, pass, mode, storage) holds two M values, which is why
    # it was found by `full-20261015-215006` (percona, 4 datasets x M 8/16/32
    # x both resource passes): that bundle published four self-refuting rows
    # out of eleven in section 5. With `m` absent
    # every M record for one (engine, dataset, pass, mode, storage) collapsed
    # into a single dict entry and the last one written won: fashion-mnist
    # printed 598 rows/s beside its own M=16 build of 167.1 s over 60,000 rows
    # (359), 545 beside M=32's 370.3 s (162) and 598 beside the other M=32's
    # 356.9 s (168), and sift-128 printed 206 beside M=16's 2,344.5 s over
    # 1,000,000 rows (427). For an incremental build the build wall clock IS
    # the load, so each of those rows carried the arithmetic that refuted it --
    # and a reader had no way to tell which of the two numbers was the stray.
    ingest_by_key = {
        (i.get("engine"), i.get("dataset"), i.get("resource_pass"),
         i.get("m"), i.get("build_mode"), i.get("storage_engine")):
            i.get("ingest_rows_per_s")
        for i in summary.get("ingest", [])
    }

    rows = []
    for r in sorted(summary.get("build", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   # The pass is in the sort as well as in the
                                   # key: it is an axis of the grid, and with
                                   # it absent two rows differing only by pass
                                   # landed in whatever order the records were
                                   # written in.
                                   #
                                   # **This changes section 5's row order for
                                   # an already-archived bundle**: it used to
                                   # run straight down M, and now groups by
                                   # pass and then M. That is deliberate --
                                   # the two passes are different machines and
                                   # reading down an M column that alternates
                                   # between them compares nothing. No cell
                                   # value moves; only which row prints first.
                                   r.get("resource_pass") or "",
                                   r.get("m") or 0, r.get("storage_engine") or "")):
        ingest_rate = ingest_by_key.get(
            (r.get("engine"), r.get("dataset"), r.get("resource_pass"),
             r.get("m"), r.get("build_mode"), r.get("storage_engine")))
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            # The resource pass, for the reason the key above carries it and
            # sections 7, 8 and 9's tables all print it: the two passes hand
            # the server different machines, so two rows differing only by
            # pass are not two readings of one configuration. In
            # `full-20261015-215006` fashion-mnist M=8 appeared twice --
            # 110.1 s at 6.5 GiB peak RSS and 100.3 s at 44.4 GiB -- with
            # nothing on the page saying which budget either row ran under,
            # and section 1 names a single pass (`tuned` there), so the
            # 8-core/16 GB normalized row read as a 76-core/112.5 GiB one.
            r.get("resource_pass") or "—",
            # `_fmt`, not `str(...)`. This was the one ops-unit M cell in the
            # file spelled `str(r.get("m", "—"))`, and it rendered correctly
            # only by accident: a record carrying `m: None` -- the key present
            # and the value absent, which `.get`'s default does not catch --
            # produced the literal string "None", and `_md_table`'s
            # `_sanitise_cell` happened to map "None" to an em dash. Remove
            # that arm and this cell prints `None` on the page with the whole
            # suite still green. `str()` also renders a float M as "16.0"
            # where every other table prints "16".
            _fmt(r.get("m"), 0),
            # MariaDB is measured on InnoDB and MyISAM under the tuned pass, and
            # without this column those two rows are identical on the page while
            # differing 6x in ingest rate.
            _storage(r),
            r.get("build_mode") or "—",
            _fmt(r.get("build_wall_s"), 1, " s"),
            _fmt(ingest_rate, 0, " rows/s"),
            _fmt_bytes(r.get("peak_rss_bytes")),
            _index_size(r),
            # The one figure every engine's build can be compared on, beside
            # the kind that says why the raw `build_wall_s` cannot be. PR 5's
            # added scope: for a bulk build this is ingest + build, for an
            # incremental one it is the single figure, and for a record that
            # does not say which it is a dash -- the two formulas differ by a
            # whole corpus load. One derivation, shared with the chart, so the
            # column and the figure cannot disagree.
            _fmt(_time_to_queryable(r), 1),
            _index_build_kind(r),
        ])
    table = _md_table(
        ["Engine", "Dataset", "Pass", "M", "Storage", "Build mode",
         "Build time", "Ingest rate", "Peak server RSS", "Index size",
         "Time to queryable index (s)", "Index build"],
        rows,
    )
    return table + _build_table_notes(summary)


def _index_size(record: Dict[str, Any]) -> str:
    """Index size, labelled when it is not a file.

    Valkey writes nothing to disk, so its figure is resident memory. Printing
    it bare in a column every other row fills with a file size invites a
    comparison between two different quantities.
    """
    size = _fmt_bytes(record.get("index_bytes"))
    if (record.get("extra") or {}).get("in_memory_only"):
        return f"{size} resident"
    return size


def _time_to_queryable(record: Dict[str, Any]):
    """`generate.time_to_queryable_index_s`, imported in the body.

    Function-local because `generate` imports this module at its own top
    level, so the reverse import has to stay inside a call -- the same shape
    `_filter_column` below uses.
    """
    from report.generate import time_to_queryable_index_s
    return time_to_queryable_index_s(record)


def _index_build_kind(record: Dict[str, Any]) -> str:
    """How the index came to exist, in four kinds rather than two.

    "no" was a fair answer while every engine either maintained its graph on
    INSERT or built it with a blocking statement afterwards. An index built by
    another process, asynchronously, while the writes are still arriving is
    neither, and calling it "no" files it beside MHNSW's incremental build,
    which is a different operation that also happens to lack a bulk step.

    The fourth kind is a FIX, not an addition. This read
    `extra.separable_build` with `.get`, so a record that does not carry the
    key at all -- every record written before it existed, which is every
    archived run -- came out as "incremental". Those runs contain pgvector,
    whose build is bulk, so the column asserted the opposite of the truth for
    one engine and there was no way for a reader to tell. `generate.build_kind`
    is the one classifier the chart uses too, so the figure and this column
    cannot disagree about what a row measured.
    """
    from report.generate import (BUILD_KIND_BULK, BUILD_KIND_UNRECORDED,
                                 build_kind)
    extra = record.get("extra") or {}
    if extra.get("async_index_build") or record.get("build_mode") == "async":
        ready = extra.get("index_ready_seconds")
        if ready:
            return f"async, ready {_fmt(ready / 60, 1)} min after load"
        return "async (separate process)"
    kind = build_kind(record)
    if kind == BUILD_KIND_UNRECORDED:
        return "not recorded"
    return "bulk after load" if kind == BUILD_KIND_BULK else "incremental"


def _build_table_notes(summary: Dict[str, Any]) -> str:
    """Footnotes for values that are labels rather than settings."""
    notes: List[str] = []
    unapplied = sorted({
        _label(r.get("engine", "?")) for r in summary.get("build", [])
        if (r.get("extra") or {}).get("m_applied") is False
    })
    if unapplied:
        notes.append(
            f"**M is not a setting for {', '.join(unapplied)}.** It exposes no "
            "graph degree, so the value says which sweep the row belongs to "
            "rather than what the engine was told. Comparisons in this table "
            "are not at matched M."
        )
    if any(_index_build_kind(r).startswith("async")
           for r in summary.get("build", [])):
        notes.append(
            "**An async build overlaps the load.** Build time for those rows is "
            "the wall clock from the first write to the index reporting ready, "
            "because neither the load nor the wait alone is the cost of having "
            "a queryable index."
        )
    unrecorded = sorted({
        _label(r.get("engine", "?")) for r in summary.get("build", [])
        if _index_build_kind(r) == "not recorded"})
    if unrecorded:
        notes.append(
            f"**The build kind was not recorded for {', '.join(unrecorded)}.** "
            f"Those records predate the field, so nothing in them says whether "
            f"their build time is a `CREATE INDEX` call or a whole corpus load "
            f"— the two differ by hours at 1M rows. The figure is shown as "
            f"measured and is **not** comparable with the rows around it; the "
            f"build chart leaves those rows off its like-for-like axis for the "
            f"same reason and says how many it dropped."
        )
    return ("\n" + "\n\n".join(notes) + "\n") if notes else ""


def _not_measured(workload: str, profile: Dict[str, Any],
                  records: Optional[List[Dict[str, Any]]] = None) -> str:
    """Say a workload was not enabled, rather than printing an empty table.

    A section with an empty table and a blank chart reads as a broken report.
    It usually just means `ops.workloads` did not include this one -- but NOT
    when a driver declined it (`extra.declined_workloads`): then the profile
    did ask, and saying it did not is the wrong reason in the one place a
    reader looks for the reason.
    """
    enabled = ((profile.get("ops") or {}).get("workloads")) or []
    declined = _disclosures.declined_lines(records or [], workload, _label)
    if declined:
        # Named from the engines that DECLINED, and nothing said about the
        # rest: another engine may be absent because its phase failed, which
        # Validity's failed-phase table reports, and a sentence here claiming
        # "every engine declined" would be a second, wrong reason for it.
        decliners = sorted(
            engine for engine, reasons in
            _disclosures.declined_by_engine(records or []).items()
            if workload in reasons)
        return (
            f"_Not measured in this run._ The "
            f"`{profile.get('name', 'profile')}` profile requested "
            f"`{workload}`; {_prose_list(decliners)} declined it, as "
            f"below, and no engine produced `{workload}` records.\n"
            + declined)
    return (
        f"_Not measured in this run._ The `{profile.get('name', 'profile')}` "
        f"profile ran `workloads: {list(enabled)}`, which does not include "
        f"`{workload}`. Add it to the profile and re-run to populate this "
        f"section; it reuses the corpus the build phase already loaded.\n"
    )


#: The mark a client-bound concurrency point carries, in the table and on the
#: chart. One symbol for one meaning, as the filtered chart already does for a
#: short result set.
CLIENT_BOUND_MARK = "○"

#: The mark on a point whose pool never ran the queries its client count
#: implies, so no scaling efficiency is rendered for it at all.
#:
#: A DIFFERENT mark from `CLIENT_BOUND_MARK` and a stronger one, because the
#: two say different things. A hollow point is a figure with a caveat: the
#: clients were busy, so read it carefully. This one is a figure that is not
#: published, because the premise it divides by -- that N clients offered N
#: concurrent queries -- did not hold. The cell carries no number to qualify.
#:
#: `⊘` and not a letter: it has to survive both output paths of this file and
#: it must not be mistaken for a footnote index. `_inline_md` converts only
#: `` ` `` and `**`, so it reaches the HTML unchanged.
SCALING_REFUSED_MARK = "⊘"

#: What the Scaling-efficiency cell says instead of a number, in the one case
#: where the harness refused it. Words and not only the mark: a column of bare
#: glyphs makes a reader hunt for the footnote, and this is the one cell in
#: the table whose content is a refusal rather than a measurement.
SCALING_REFUSED_CELL = f"{SCALING_REFUSED_MARK} not measurable"

#: `extra.scaling_status` when the pool served materially fewer queries than
#: it had clients. Declared here rather than imported from
#: `harness.workloads.clients`, and NOT because the import would fail -- the
#: report container mounts `harness/` too, which is why the two
#: `harness.metrics.stats` imports in this file work. It is that they are
#: function-scope on purpose: `harness.workloads.clients` pulls numpy and
#: every driver module in to obtain one string. The two declarations are
#: pinned together by `tests/test_g1_client_concurrency.py`.
SCALING_CONCURRENCY_NOT_ACHIEVED = "concurrency_not_achieved"

#: `extra.scaling_note` when the window's median sat under half its mean
#: (ops version 9; `harness/metrics/scaling.SCALING_TAIL_DOMINATED`, pinned
#: equal by a test). A NOTE: the cell carries the mark beside a published
#: figure, and nothing is withheld for it.
TAIL_DOMINATED = "tail_dominated"
TAIL_DOMINATED_MARK = "△"

#: Footnote marks for the two statuses the ef sweep can end on without having
#: put an engine at the target recall. `report/charts.py` imports these rather
#: than spelling the glyphs again: the chart's dagger and the table's dagger
#: have to be the same character or the footnote under one explains a mark the
#: reader saw on the other. `CLIENT_BOUND_MARK` above is duplicated as a
#: literal in the concurrency chart's suptitle, which is the drift this avoids.
#:
#: Two DIFFERENT marks, because PR 2 §2.3.3 asks for the distinction and the
#: two conditions are opposites: a dagger means the engine is above the target
#: and could not be slowed to it, so its throughput is understated relative to
#: the rest; a double dagger means it never got there, so its throughput is
#: overstated. One mark for both would present those as the same caveat.
FLOOR_MARK = "†"
UNREACHED_MARK = "‡"

#: A third condition, and the one that is not a sweep outcome at all: this
#: engine was never calibrated, so it was not held to the target the chart
#: title names. Runs are checkpointed per (pass, engine, dataset, phase), so an
#: operator who flips `ops.ef_mode` to `calibrated` and resumes keeps the
#: fixed-ef records of every engine already measured -- and nothing refuses
#: that, because the measurement version did not change. Without a mark those
#: engines sit on a panel titled with a recall floor they never met, under a
#: table whose own introduction says to read the Status column first, while
#: having no row in it to read.
#:
#: `*` is the conventional first footnote mark and survives this file's two
#: output paths: `_inline_md` converts only `` ` `` and `**`, so a lone
#: asterisk reaches the HTML unchanged (measured), and in the Markdown it is
#: always written mid-sentence with spaces either side -- never at the start
#: of a line, which is the position where it would become a bullet, and never
#: paired, which is what a renderer would read as emphasis.
UNCALIBRATED_MARK = "*"


def _fmt_target(target: float) -> str:
    """A recall TARGET as the report should spell it, in the table and the chart.

    `%g`, not `_fmt`'s fixed decimals, and this is not a style preference.
    Measured on this interpreter: `_fmt(0.999, 2)` is `1.00`, so a
    `--target-recall 0.999` run would have its Target column read as a demand
    for PERFECT recall; and `_fmt(0.995, 2)` is `0.99`, because 0.995 is
    slightly below 0.995 in binary, so the column would understate the floor
    the engines were held to. `%g` prints 0.95, 0.9, 0.98, 0.995 and 0.999
    exactly.

    A target is a declared threshold, not a measurement: rounding it is
    rewriting the run's configuration, which is why it does not share the
    four-decimal convention the Recall@k columns use for measured values.
    `report/charts.py` imports this so the chart title and this column cannot
    print one target two ways.
    """
    return f"{target:g}"


# ---------------------------------------------------------------------------
# Operating point (PR 2 §2.3.2)
# ---------------------------------------------------------------------------

def _operating_point_table(summary: Dict[str, Any]) -> str:
    """Which ef_search each engine was measured at, and what it bought.

    This table is the premise of the three sections after it. Concurrency,
    filtered search and churn are all measured at one ef_search per engine,
    and once that ef is chosen per engine rather than declared in the profile,
    a reader cannot check that the comparison was fair without seeing the
    choice. The column that makes it checkable is **Status**: `ok` means the
    engine was put at the target, and the other two values mean it was not --
    in opposite directions.

    Deliberately wider than §2.3.2's "engine · chosen ef · recall · status".
    The remaining columns are what make a row identify one measurement: a
    profile sweeping M calibrates once per M, so Dataset / Pass / M /
    Build mode / Storage are the difference between four rows and one row
    printed four times. The Grid column is there because a sweep can stop
    early, and "chose ef 640 out of a grid whose largest point is 640" is a
    different fact from "chose 640 out of 1000".

    **`Storage` is a fix, and the sentence above used to omit it.**
    `operating_point_key` separates units by `resource_pass` plus
    `build_signature`, and the signature contains the storage engine, so a
    MariaDB sweep on InnoDB and on MyISAM produces two rows -- two separate
    calibrations on two different indexes -- whose every printed identity
    cell was the same. `loaders.operating_points` has always carried
    `storage_engine` on the row; only this table dropped it. Unlike the
    `build_mode` omission in sections 7 and 9, this one is latent rather than
    live: every committed profile pins `mariadb_storage_engines: [InnoDB]`
    today, and `config/profiles/tuned-complete.yml` documents how to widen it
    again -- which the tuned-complete run did, and `_build_table`'s comment
    records what that measured.
    """
    rows = []
    for point in summary.get("operating_points", []) or []:
        status = point.get("status")
        marked = {"floor_above_target": f"floor_above_target {FLOOR_MARK}",
                  "target_unreached": f"target_unreached {UNREACHED_MARK}"}
        if point.get("status_conflict"):
            shown = ("conflict: "
                     + ", ".join(str(s) for s in point["status_conflict"]))
        else:
            shown = marked.get(status, status or "not recorded")
        if point.get("ef_conflict"):
            ef_cell = ("conflict: "
                       + ", ".join(str(e) for e in point["ef_conflict"]))
        else:
            ef_cell = (str(point["ef_search"])
                       if point.get("ef_search") is not None else "—")
        measured = point.get("points_measured")
        size = point.get("grid_size")
        grid = (f"{measured} of {size}" if measured and size
                else (str(measured) if measured else "—"))
        rows.append([
            _label(point.get("engine", "?")), point.get("dataset", "?"),
            point.get("resource_pass") or "—",
            # `_fmt`, not `str(... or "—")`: the latter renders a float M as
            # "16.0" where the five other ops-unit tables print "16", and
            # this table names Pass, M and Build mode beside each other, so a
            # reader scanning down M across sections sees two spellings of
            # one number.
            _fmt(point.get("m"), 0),
            point.get("build_mode") or "—",
            _storage(point),
            ef_cell,
            # Requested vs searched: MongoDB cannot search fewer candidates
            # than k, so its "ef 10" at k=100 is a search of 100, and two
            # engines compared at one chosen ef are not comparable without
            # this. Printed only where it differs, so the column is silent on
            # the engines that honour the request.
            (str(point["effective_ef"])
             if point.get("effective_ef") is not None
             and point.get("effective_ef") != point.get("ef_search")
             else "—"),
            # Four decimals for the measured recall, matching the concurrency
            # table's convention so the two can be read side by side; `%g` for
            # the declared target, because rounding a threshold rewrites the
            # run's configuration. See `_fmt_target`.
            _fmt(point.get("recall_at_k"), 4),
            (_fmt_target(point["target_recall"])
             if point.get("target_recall") is not None else "—"),
            shown,
            grid,
        ])
    if not rows:
        # Explicit, not an empty table. §2.3's whole point is that a reader
        # can see which ef each engine was measured at; a blank table here
        # would read as "the sweep ran and chose nothing", when the honest
        # statement is that no sweep ran at all and the ef_search below came
        # straight out of the profile.
        return (
            "_No ef_search calibration in this run._ Every number in the "
            "sections below was measured at the `ef_search` the profile "
            "declares (`ef_mode: fixed`), so the engines are compared at "
            "equal *search effort* rather than at equal *answer quality* — "
            "and on these implementations one `ef_search` can buy recalls "
            "0.18 apart. Re-run with `ef_mode: calibrated` to populate this "
            "section.\n")
    return _md_table(
        # `Build mode`, not `Build`. This column is `build_mode` -- the same
        # ops-unit axis the other six tables print -- and it was the only one
        # naming it differently, which is the `Pass` / `Resource pass`
        # divergence the convention guard in
        # `tests/test_e10_report_unit_join_keys.py` exists to prevent, sitting
        # inside this file. `Build` is also ambiguous here in a way it is not
        # elsewhere: section 5 has a `Build time` column, so a bare `Build`
        # beside `Chosen ef_search` reads as a duration.
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage",
         "Chosen ef_search", "Effective ef", "Recall@k", "Target", "Status",
         "Grid points"],
        rows,
    ) + _operating_point_notes(summary)


def _operating_point_notes(summary: Dict[str, Any]) -> str:
    """The footnotes, each printed only when this run earned it."""
    points = summary.get("operating_points", []) or []
    floors = [p for p in points if p.get("status") == "floor_above_target"]
    unreached = [p for p in points if p.get("status") == "target_unreached"]
    parts = [
        # `**bold**`, not `*italic*`, on the two direction words below and on
        # "large" in the appendix. `_inline_md` converts backticks and `**`
        # and has no italic rule at all -- its own comment says "bold before
        # italic" over code that does not do italic -- so `*higher*` reached
        # the HTML page as literal asterisks around the one token a misreading
        # inverts. Asterisk-italics are used elsewhere in this file
        # (`*first*` in `_validity_section`, `*intention*` in
        # `_CONCURRENCY_NOTES`) and have the same defect there; following the
        # convention on a word this load-bearing was the wrong trade, and
        # fixing those two is someone else's diff.
        "\n**Recall@k here is the calibration sweep's**, measured at the "
        "chosen `ef_search` over `ops.max_queries` single-client queries "
        "against warm caches — the measurement the choice was made on. The "
        "Recall@k in the concurrency table below is a different number: the "
        "median window's, over the shared query pool, with every other client "
        "contending. They are not expected to be equal and neither is a check "
        "on the other.\n",
    ]
    if floors:
        where = ", ".join(sorted(
            f"{_label(p.get('engine', '?'))} at M={p.get('m', '?')}"
            f" ({p.get('resource_pass') or 'unknown pass'}),"
            f" recall {_fmt(p.get('recall_at_k'), 4)} at ef_search"
            f" {p.get('ef_search')}" for p in floors))
        parts.append(
            f"\n{FLOOR_MARK} **These engines cannot be slowed down to the "
            f"target.** {where}. The smallest `ef_search` in the grid already "
            f"exceeds the target recall by more than 0.02, so no grid point "
            f"brings them down to it and every number below is measured at a "
            f"**higher** recall than the engines marked `ok` — which understates "
            f"their throughput relative to the others rather than flattering "
            f"it. The comparison below is not like-for-like on recall for "
            f"these rows.\n\nMHNSW and VIDX are both expected to land here at "
            f"M=16 — 0.94–0.98 at minimum effort, per the tuned-profile notes "
            f"— and for them it is a property of the implementation rather "
            f"than a fault of the run. For any other engine, check that "
            f"`ops.ef_grid` starts low enough to reach the target from "
            f"below.\n")
    if unreached:
        where = ", ".join(sorted(
            f"{_label(p.get('engine', '?'))} at M={p.get('m', '?')}"
            f" ({p.get('resource_pass') or 'unknown pass'}) reached "
            f"{_fmt(p.get('recall_at_k'), 4)} at ef_search "
            f"{p.get('ef_search')}" for p in unreached))
        parts.append(
            f"\n{UNREACHED_MARK} **These engines reached the target at no grid "
            f"point** and are reported at the best recall they achieved: "
            f"{where}. Their throughput below is bought at a **lower** recall "
            f"than the engines marked `ok`, so it is not comparable with "
            f"theirs and will tend to look better than it is. Extend "
            f"`ops.ef_grid` upwards and re-run if the target matters more than "
            f"the run time.\n")
    conflicts = [p for p in points
                 if p.get("ef_conflict") or p.get("status_conflict")]
    if conflicts:
        parts.append(
            f"\n**{len(conflicts)} row(s) show a conflict rather than a "
            f"value.** Two different chosen `ef_search` values, or two "
            f"different statuses, were found among the records of one "
            f"measurement unit — which means this report is reading more than "
            f"one measurement of that unit as though it were one. A run "
            f"directory cannot produce this (one JSONL file per unit, emptied "
            f"per attempt), so the records came from a merged file via "
            f"`--from-records`. Split them and report each measurement "
            f"separately; nothing in those rows is safe to read.\n")
    missing = [p for p in points if p.get("status") is None
               and not p.get("status_conflict")]
    if missing:
        parts.append(
            f"\n**{len(missing)} row(s) have no status.** The sweep's records "
            f"are present but no concurrency, filtered or churn record carries "
            f"`extra.calibration_status`, so the ef was chosen and then "
            f"nothing was measured at it — a unit whose downstream workloads "
            f"did not run. The sweep's cost was paid and produced no "
            f"comparison.\n")
    # The mirror case: a unit whose sweep records are missing but whose
    # downstream half is present. It renders an em dash in Recall@k, Effective
    # ef and Grid points, and the `missing` branch above cannot see it because
    # that keys on an absent STATUS and this row has one.
    no_curve = [p for p in points if p.get("points_measured") is None]
    if no_curve:
        parts.append(
            f"\n**{len(no_curve)} row(s) show an ef and a status but no "
            f"curve.** The workloads recorded which ef they were given, but "
            f"the `ef_calibration` records that ef was chosen from are not in "
            f"this report — narrowed away by `--datasets`, or a records file "
            f"assembled by hand. The ef and the status are trustworthy; the "
            f"empty Recall@k means the evidence for the choice is absent, not "
            f"that the recall was zero.\n")
    missing_points = _engines_without_an_operating_point(summary)
    if missing_points and points:
        where = ", ".join(f"{_label(e)} on {d}" for e, d in missing_points)
        parts.append(
            f"\n**{len(missing_points)} engine(s) were measured but never "
            f"calibrated: {where}.** They have throughput in the sections "
            f"below and no row here, so the Status column cannot be read for "
            f"them and the recall floor the concurrency chart's title names "
            f"does not apply to their curves — which carry the "
            f"{UNCALIBRATED_MARK} mark there. This is what a run that was "
            f"resumed after `ops.ef_mode` changed looks like: checkpoints are "
            f"per (pass, engine, dataset, phase), so engines already measured "
            f"keep their fixed-ef records and nothing refuses the mixture, "
            f"because the measurement version did not change. Re-measure "
            f"those engines, or compare them only with each other.\n")
    return "".join(parts)


def _engines_without_an_operating_point(summary: Dict[str, Any]
                                        ) -> List[Any]:
    """(engine, dataset) pairs with measurements in a calibrated workload and
    no operating point.

    Only the three workloads the sweep exists to choose an ef for -- an engine
    with an `ingest` or `index_build` record and nothing else was never going
    to have an operating point, and saying so would fire on every profile that
    builds an index without querying it.

    The caller prints this only when at least one operating point exists. On
    `ef_mode: fixed` every engine is in this list, the table has already said
    the run was not calibrated, and repeating it per engine would turn the one
    true sentence into six.
    """
    calibrated = {(p.get("engine"), p.get("dataset"))
                  for p in summary.get("operating_points", []) or []}
    seen = set()
    for key in ("concurrency", "filtered", "churn"):
        for r in summary.get(key, []) or []:
            pair = (r.get("engine"), r.get("dataset"))
            if all(pair) and pair not in calibrated:
                seen.add(pair)
    return sorted(seen)


# The tuned pass measures MariaDB on both InnoDB and MyISAM, so an engine name
# alone does not identify a row. Without this column the two appeared as
# unexplained duplicates, and the largest result in the tuned-complete run --
# that MariaDB loses 3-10% of its throughput to churn on MyISAM against 84-86%
# on InnoDB -- was invisible on the page.
#
# `Pass`, `M` and `Build mode` are here for the same reason and were added by
# the same kind of evidence. **The ops phase loops four axes** --
# `resource_pass` x `m_values` x `build_modes` x `storage_engines`
# (`orchestrator/cli.py`) -- and this table named one of the four, so the
# other three rendered as duplicate rows. In `full-20261015-215006`
# five fashion-mnist rows at one client read 1,677 / 2,037 / 1,952 / 1,673 /
# 1,789 QPS -- five different (pass, M) units, in that order
# (normalized M=32, normalized M=8, tuned M=16, tuned M=32, tuned M=8) -- and
# the page could not tell them apart. The pass is the damaging half: the
# normalized pass gives the server 8 cores and 16 GB
# (`config/resources/normalized.yml`) against the tuned pass's 76 cores and
# 112.5 GiB in that run, section 1 names one pass for the whole report, so an
# 8-core throughput read as a 76-core one. Section 8's filtered table already
# carried all three columns, which is why the same grid is legible there.
#
# **`Build mode` is not hypothetical and did not show in that run only
# because `config/profiles/percona.yml` pins `[incremental]`.**
# `full.yml`, `main.yml`, `smoke.yml` and `quick.yml` all set
# `pgvector_build_modes: [post, incremental]` and the ops loop runs a unit per
# mode, so a pgvector run publishes two concurrency points per (pass, M,
# storage) that are different indexes -- a bulk-built one and an incrementally
# maintained one. Rendered without this column they carry byte-identical
# identity cells across a 22% throughput spread, which is this section's
# defect exactly, one axis further out.
def _concurrency_table(summary: Dict[str, Any]) -> str:
    rows = []
    points = _operating_point_index(summary)
    for r in sorted(summary.get("concurrency", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   # All four ops-unit axes sort ahead of the
                                   # client count, so each unit's scaling
                                   # curve is contiguous instead of
                                   # interleaved with the others at every
                                   # client count. `m or 0`, not `str(m)`: as
                                   # text, M 8/16/32 sorts 16, 32, 8.
                                   r.get("resource_pass") or "",
                                   r.get("m") or 0,
                                   r.get("build_mode") or "",
                                   r.get("storage_engine") or "",
                                   r.get("clients") or 0)):
        extra = r.get("extra") or {}
        # The efficiency is read inside `_efficiency_cell`, not here: the
        # decision whether to print it at all is that function's, and a local
        # copy of the number beside it is how a caller comes to render one
        # thing while the cell says another.
        # A point measured once carries no spread, and the difference between
        # "measured once" and "repeated and stable" is the difference between
        # a number and a result.
        repeats = extra.get("repeats") or 1
        spread = ("—" if repeats < 2
                  else f"±{extra.get('qps_spread_pct', 0):.1f}% (n={repeats})")
        # A pool is five figures on the real datasets; `_fmt` groups a float
        # and prints an int bare, and "10000" beside a QPS of "5,100" reads as
        # a different kind of number.
        pool = extra.get("query_pool")
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            # All four unit axes, spelled and ordered exactly as
            # `_filtered_table` and the churn-under-writes table spell them
            # (Pass, M, Build mode, Storage), so one reader convention covers
            # every table that renders an ops unit.
            r.get("resource_pass") or "—",
            _fmt(r.get("m"), 0),
            r.get("build_mode") or "—",
            _storage(r),
            str(r.get("clients", "—")),
            _fmt(r.get("qps"), 0),
            spread,
            # Beside the throughput it qualifies, which is the whole reason
            # the workload keeps the ids. Two engines at one `ef_search` can
            # sit at 0.98 and 0.80, and a throughput table with no quality
            # column presents them as comparable while one is doing
            # materially less work per query.
            _fmt(r.get("recall_at_k"), 4),
            _fmt(r.get("latency_p50_ms"), 3),
            _fmt(r.get("latency_p95_ms"), 3),
            _fmt(r.get("latency_p99_ms"), 3),
            _efficiency_cell(extra),
            # The evidence for that cell, beside it. All three -- the client
            # count, what was served and the ratio's verdict -- for the reason
            # the client-CPU column carries its numerator and denominator: a
            # refusal a reader cannot check is indistinguishable from a bug in
            # the refusal.
            _served_cell(extra, r.get("clients")),
            # What the engines' caches were actually shown. A reader comparing
            # a version-2 report to a version-3 one needs these two to see why
            # the throughput moved: the same QPS over a pool of 500 and a pool
            # of 10,000 are measurements of two different things.
            _fmt(float(pool) if pool is not None else None, 0),
            _fmt(extra.get("query_reuse"), 2, "x"),
            _client_cpu_cell(extra),
            # Beside the client's, so a flattening curve says whose ceiling
            # it is: an engine out of cores, stalled, or idle behind a lock.
            *_server_cells(r, summary.get("server_windows")),
            *_operating_point_cells(r, points),
        ])
    return _md_table(
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage", "Clients",
         "QPS", "Spread", "Recall@k", "p50 (ms)", "p95 (ms)", "p99 (ms)",
         "Scaling efficiency", "Queries in service", "Query pool", "Reuse",
         "Client CPU", "Server CPU", "Server CPU pressure",
         "Calibrated ef", "Calibration recall"],
        rows,
    ) + BUILD_MODE_COLUMN_NOTE + OPERATING_POINT_NOTE \
        + _concurrency_notes(summary)


#: What the limiter table is and, as importantly, what it is not.
LIMITER_NOTE = (
    "\n_**What limited throughput** puts the measured components of each "
    "point side by side: the client container's CPU, the server's CPU and "
    "CPU pressure over the same window, and the server's own largest wait "
    "(performance_schema for the MySQL family, INFO commandstats for "
    "Valkey -- service time, not waiting -- and mongod's lock waits for "
    "MongoDB), read on a separate connection before and after the window "
    "and so also spanning the pool's warm-up. "
    "**Largest measured** names the largest of the first three shares and "
    "is not a verdict: a wait profile has no common unit with them and is "
    "shown, not ranked. A dash with a reason means the engine could not "
    "say, which is not the same as nothing waited. **Profile** links the "
    "perf stacks for the one EXTRA window `ops.profile_window_clients` "
    "asked for: same client count and length, run after the measured "
    "repeats and never one of them, so no published figure was taken under "
    "the profiler. pgvector has no cumulative wait counters; its wait column "
    "is a dash unless `ops.server_wait_sampler` sampled pg_stat_activity "
    "during the window, which every such record states._\n")


def _limiter_table(summary: Dict[str, Any]) -> str:
    """One "what limited throughput" row per concurrency point."""
    from report.limiters import (largest_component, profile_cell,
                                 top_wait_cell)
    from report.server_windows import window_key
    windows = summary.get("server_windows") or {}
    rows = []
    for r in sorted(summary.get("concurrency", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("resource_pass") or "",
                                   r.get("m") or 0,
                                   r.get("build_mode") or "",
                                   r.get("storage_engine") or "",
                                   r.get("clients") or 0)):
        extra = r.get("extra") or {}
        figures = windows.get(window_key(r)) or {}
        server, pressure = _server_cells(r, windows)
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            r.get("resource_pass") or "—", _fmt(r.get("m"), 0),
            r.get("build_mode") or "—", _storage(r),
            str(r.get("clients", "—")),
            _client_cpu_cell(extra), server, pressure,
            top_wait_cell(extra),
            profile_cell(r, summary.get("profiles")),
            largest_component(extra.get("client_cpu_util"),
                              figures.get("cpu_util"),
                              figures.get("cpu_pressure")),
        ])
    return ("\n#### What limited throughput\n" + _md_table(
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage", "Clients",
         "Client CPU", "Server CPU", "Server CPU pressure", "Top server wait",
         "Profile", "Largest measured"],
        rows,
    ) + LIMITER_NOTE)


#: What the two columns every concurrency and churn table ends with --
#: "Calibrated ef" and "Calibration recall" -- mean. Spelled as literals in
#: each header rather than unpacked from a constant, because
#: `tests/test_e10_report_unit_join_keys.py` finds the unit tables by an AST
#: scan for literal header lists. The calibration point is what makes a
#: throughput figure comparable: the policy chose that ef because the sweep
#: measured the target recall there.
OPERATING_POINT_NOTE = (
    "\n_**Calibrated ef** is the ef_search the calibration chose for this "
    "unit and every throughput figure in the row was measured at; "
    "**Calibration recall** is the recall the sweep measured there, which is "
    "the accuracy the throughput was bought at. A status beside it means the "
    "sweep did not put the engine at the target -- `target_unreached` "
    f"{UNREACHED_MARK} (never got there: the throughput is at a lower recall) "
    f"or `floor_above_target` {FLOOR_MARK} (could not be slowed to it) -- and "
    "that unit is not ranked in the peak-throughput headline. "
    "`<ef> (fixed)` with *not calibrated* means the unit ran at a "
    "declared ef_search (`ef_mode: fixed`, or a record from before ops "
    "measurement version 4), so its throughput is not at equal recall with "
    "any other row._\n")


#: The two cells of a unit measured at a declared ef. Words, not a dash: a
#: dash reads as "not recorded", and the point is that no recall target was
#: pursued, so the QPS beside it is not at any engine's equal recall.
FIXED_EF_RECALL_CELL = "not calibrated"


def _operating_point_cells(row: Dict[str, Any], points: Any) -> List[str]:
    from report.operating import UNRANKED_STATUSES, is_fixed_ef, status_of
    point = points.find(row)
    status = status_of(row, point)
    ef = (point or {}).get("ef_search")
    # Records only: a churn-under-writes row is derived and carries no
    # `extra`, so the pre-version-4 rule would read every one as fixed.
    if point is None and row.get("phase") and is_fixed_ef(row):
        return [f"{_fmt(row.get('ef_search'), 0)} (fixed)",
                FIXED_EF_RECALL_CELL]
    if ef is None and point is None:
        return ["—", "—"]
    recall = _fmt((point or {}).get("recall_at_k"), 4)
    if status in UNRANKED_STATUSES:
        mark = UNREACHED_MARK if status == "target_unreached" else FLOOR_MARK
        recall = f"{recall} {status} {mark}"
    return [_fmt(ef, 0), recall]


def _operating_point_index(summary: Dict[str, Any]) -> Any:
    from report.operating import OperatingPointIndex
    return OperatingPointIndex(summary.get("operating_points") or [])


def _peak_throughput_table(summary: Dict[str, Any]) -> str:
    """The cross-engine concurrency headline, ranked only where comparable.

    Built from `summary["concurrency_peaks"]` (`report/operating.py`):
    client-bound points never set a peak, and a unit whose calibration missed
    the target shows its status where the number would be and sorts after
    every ranked unit -- its throughput is at a recall the others were not
    held to.
    """
    peaks = summary.get("concurrency_peaks") or []
    if not peaks:
        return ""
    rows = []
    rank = 0
    for p in peaks:
        if p["ranked"]:
            rank += 1
            figure = _fmt(p["peak_qps"], 0)
            clients = str(p.get("peak_clients") or "—")
        else:
            figure = f"not ranked: {p.get('status') or 'no point'}"
            clients = "—"
        rows.append([
            _label(p.get("engine") or "?"), p.get("dataset") or "?",
            p.get("resource_pass") or "—", _fmt(p.get("m"), 0),
            p.get("build_mode") or "—", _storage(p),
            str(rank) if p["ranked"] else "—",
            figure, clients, _fmt(p.get("scaling_efficiency"), 2),
            (f"{_fmt(p.get('ef_search'), 0)} (fixed)" if p.get("fixed_ef")
             else _fmt(p.get("ef_search"), 0)),
            (FIXED_EF_RECALL_CELL if p.get("fixed_ef")
             else _fmt(p.get("calibration_recall"), 4)),
            str(p.get("client_bound_excluded") or 0),
        ])
    return (
        "\n**Peak throughput.** Each unit's highest concurrency QPS and the "
        "client count it scaled to, ranked within the run. Client-bound "
        f"points ({CLIENT_BOUND_MARK}) are excluded -- the clients were the "
        "ceiling there -- and counted in the last column; a unit whose "
        "calibration missed the recall target is listed but not ranked.\n"
        + _md_table(["Engine", "Dataset", "Pass", "M", "Build mode",
                     "Storage", "Rank", "Peak QPS", "At clients",
                     "Scaling efficiency",
                     "Calibrated ef", "Calibration recall",
                     "Client-bound points excluded"], rows)
        + "\n")


def _efficiency_cell(extra: Dict[str, Any]) -> str:
    """The Scaling-efficiency cell, or a refusal where it is not measurable.

    **The refusal is the point of this function.** `scaling_efficiency` is
    this section's headline, and it is an N-client throughput divided by N
    times a one-client throughput -- a statement about the engine only if N
    queries were in flight. A live `smoke` run of MariaDB measured 2,920.7 q/s
    at one client and 2,927.5 at four with every latency percentile unchanged,
    which is about one query in service: the 0.25 it printed describes the
    client, and rendering it in this column presents it as a property of the
    database.

    An em dash, as everywhere else in this file, where there is simply no
    figure -- a point with no one-client baseline to divide by. The refusal
    cell is a different thing and looks different: there IS a number in the
    record and this report will not print it.

    A record with no `scaling_status` at all renders as it always did, which
    is every record written before this existed. Absence is not a refusal:
    treating it as one would blank the efficiency column of every archived run
    in the results tree and read as a regeneration fault.
    """
    if extra.get("scaling_status") == SCALING_CONCURRENCY_NOT_ACHIEVED:
        return SCALING_REFUSED_CELL
    eff = extra.get("scaling_efficiency")
    return _fmt(eff, 2) if eff is not None else "—"


def _served_cell(extra: Dict[str, Any], clients: Any) -> str:
    """How many queries the pool had in service, against how many it had
    clients: the mean-based figure first, the median estimate second.

    Ops version 9 put `outstanding_concurrency` first, labelled "(mean)", and
    `served_concurrency` second as "median est.". The order was the other
    way round while the guard read the median, and on `full-20260825` that
    printed a MariaDB 12.3 32-client point as a fraction of the queries it
    held because its p99 was thirty times its p50. A record with no
    `outstanding_concurrency` (written before G1 recorded it) still renders
    the median figure alone, as it always did.

    `served / clients` written out as a pair rather than as the ratio the
    status is computed from, because the two numbers are what a reader needs
    in order to check the verdict: `2.0 of 4` is self-explaining where `0.50`
    needs the footnote to say what it is a share of. The same argument the
    Client CPU column's numerator and denominator are kept for.

    An em dash where the figure was not recorded, which is every record
    written before this existed, and where the window counted no query.
    """
    served = extra.get("served_concurrency")
    outstanding = extra.get("outstanding_concurrency")
    if served is None and outstanding is None:
        return "—"
    of = f" of {clients}" if clients else ""
    if outstanding is None:
        text = f"{served:,.2f}{of}"
    else:
        text = f"{outstanding:,.2f}{of} (mean)"
        if served is not None:
            text += f" · {served:,.2f} median est."
    if extra.get("scaling_note") == TAIL_DOMINATED:
        text += f" {TAIL_DOMINATED_MARK}"
    if extra.get("scaling_status") == SCALING_CONCURRENCY_NOT_ACHIEVED:
        return f"{text} {SCALING_REFUSED_MARK}"
    return text


def _server_cells(row: Dict[str, Any], windows: Dict[Any, Any]) -> List[str]:
    """Server CPU and server CPU pressure over this row's window.

    From `report/server_windows.py`, joined by the window's wall-clock ends.
    Cores and, when the manifest recorded the cpuset, utilisation; pressure
    is the share of the window in which a runnable task waited for CPU.
    Dashes for a record from before ops version 9, which has no window ends.
    """
    from report.server_windows import window_key
    figures = (windows or {}).get(window_key(row))
    if not figures:
        return ["—", "—"]
    cores = figures.get("cpu_cores")
    util = figures.get("cpu_util")
    if cores is None:
        cpu = "—"
    elif util is not None:
        cpu = f"{cores:,.1f} cores ({util:.0%})"
    else:
        cpu = f"{cores:,.1f} cores"
    pressure = figures.get("cpu_pressure")
    io = figures.get("io_pressure")
    cell = "—" if pressure is None else f"{pressure:.0%}"
    if io is not None:
        cell += f" (io {io:.0%})"
    return [cpu, cell]


def _client_cpu_cell(extra: Dict[str, Any]) -> str:
    """The client container's utilisation, marked when it was the ceiling.

    An em dash means no counter was readable, not that the clients were idle;
    `client_bound` is None in the same case for the same reason, and the
    footnote says so. Writing 0.00 there would be the silent version of the
    fault this column exists to expose.
    """
    util = extra.get("client_cpu_util")
    if util is None:
        return "—"
    text = _fmt(util, 2)
    return f"{text} {CLIENT_BOUND_MARK}" if extra.get("client_bound") else text


#: What the four columns the concurrency table grew actually mean, as prose
#: rather than as longer headers: each is a sentence about how the number was
#: taken, and the failure they guard against is a reader comparing two cells
#: that were not measuring the same thing. A module constant so
#: `_concurrency_notes` below stays short enough to read as the handful of
#: conditionals it is.
_CONCURRENCY_NOTES = (
        "\n**Recall@k** is the median window's, scored over the queries that "
        "window executed (`extra.recall_queries`) — the same set its QPS "
        "counts. It is recorded because a fixed `ef_search` is an *intention* "
        "that all engines sit at comparable recall, and two of these "
        "implementations at one `ef_search` can differ by 0.18.\n",
        "\n**Query pool** is how many distinct queries the clients drew from "
        "and **Reuse** is how many times the average query was answered "
        "inside the window. A small pool measures every graph cache against a "
        "hot set it can hold entirely, which erases the difference this "
        "section exists to show, so the two figures are what make one run's "
        "throughput comparable with another's.\n",
        "\n**QPS counts only the queries that both began and ended inside the "
        "window.** `extra.inflight_discarded` records how many were dropped "
        "at the close — at most one per client on any real driver, the query "
        "each was holding when the clock stopped, though a query cheaper than "
        "the gap between the stop instant and the stop flag can be discarded "
        "several times over. Their share of the uncorrected figure is "
        "about per-query latency ÷ window length, so the correction is a few "
        "per cent for a slow engine over a short window and 0.005% for a "
        "1.4 ms query over thirty seconds.\n",
        f"\n**Client CPU** is the *client* container's own cgroup CPU over the "
        f"window divided by (window × the size of its cpuset): 1.00 means the "
        f"harness saturated every CPU it was given. "
        f"{CLIENT_BOUND_MARK} marks a point above 0.80, drawn hollow on the "
        f"chart as well. Each client is a forked OS process serializing "
        f"vectors inside that cpuset, so a hollow point's throughput is "
        f"bounded by this harness as well as by the engine and **is not "
        f"evidence about where the engine saturates**. The reverse does not "
        f"follow, and the reason is structural rather than a matter of where "
        f"the limit was set: **a CPU-utilisation threshold cannot see an "
        f"interpreter-lock ceiling, because the lock prevents the very CPU "
        f"consumption the threshold looks for.** What that no longer "
        f"describes is *these* clients: each holds its own interpreter, so a "
        f"driver that keeps the lock across its socket wait can no longer "
        f"make them take turns. The shape is kept because it is still the "
        f"one to recognise, and the figure behind it is the RETIRED thread "
        f"pool's: driven by the synthetic driver that harness flattened at "
        f"2.06 cores — 0.51 of a four-CPU container at 4, 8, 16 and 32 "
        f"clients alike — while its throughput peaked at two clients and "
        f"halved by thirty-two, and this column marked none of those points. "
        f"So **an unmarked point is not a guarantee that the engine was the "
        f"limit**; a flat Client CPU column beside a falling QPS curve is "
        f"still the case to re-measure, and `Queries in service` beside it "
        f"is the column that was added to tell the two apart.\n",
        f"\n**Queries in service** is how many queries the pool actually had "
        f"in flight on average, against how many clients it ran, two ways. "
        f"**(mean)** is the exact figure — every in-window query's duration "
        f"summed over the window, Little's law with no estimator — and since "
        f"`ops_measurement_version` 9 it is the one the refusal reads. "
        f"**median est.** is the queries the window counted times their "
        f"MEDIAN latency over the window, the figure the guard read before "
        f"version 9; the two differ by exactly p50 ÷ mean, and where the "
        f"median estimate is under half the mean the cell carries "
        f"{TAIL_DOMINATED_MARK} (`tail_dominated`) — a note that the latency "
        f"had a heavy tail, never a refusal. **What the version-9 refusal "
        f"detects is client-side starvation** — the clients kept fewer than "
        f"half their queries in flight — and the case where the median "
        f"estimate and a credible efficiency are both under half. **It does "
        f"not detect a server that answers one query at a time**: from the "
        f"clients' clocks that looks exactly like a saturated server "
        f"(in service ≈ N, efficiency ≈ 1/N) and is published with its low "
        f"efficiency. Serialization and saturation inside the server are "
        f"told apart, where they can be, by the Server CPU and Server CPU "
        f"pressure columns, not by this one. The rule the rest of this "
        f"paragraph describes is the PRE-version-9 one, still the verdict "
        f"on any older record in this report. **Scaling efficiency divides "
        f"an N-client throughput by "
        f"N times a one-client throughput, so it describes the engine only if "
        f"N queries were in flight** — and where this column says they were "
        f"not, the efficiency cell reads *{SCALING_REFUSED_CELL}* and no "
        f"number is published. The threshold is half the client count, picked "
        f"at the middle of a measured gap: five synthetic drivers of "
        f"identical 0.325 ms query cost put this ratio at 0.22–0.25 where the "
        f"pool served one query and 0.94–1.00 where it served four, with "
        f"nothing in between. **A saturated server with deterministic "
        f"service time is not flagged**, and that is the case the threshold "
        f"was chosen to protect: a real queue raises the whole latency "
        f"distribution, median included, so the modelled saturated server "
        f"sits at 0.98 of its client count while its efficiency falls to "
        f"0.28. **That is the whole of what was measured, and the claim is "
        f"narrowed to it on purpose**: every calibration driver has a "
        f"deterministic per-query cost, so a server whose service time is "
        f"skewed enough to put its median well below its mean can still be "
        f"refused here. That skew-triggered refusal was this rule's known "
        f"false positive, and it is why version 9 moved the refusal to the "
        f"mean-based figure. What a refusal no longer means is Python: until "
        f"version 6 the clients shared one interpreter lock and a serializing "
        f"client library was the expected cause, and each client is now its "
        f"own process. An engine that answers one query at a time is the "
        f"cause that remains — read the Client CPU cell beside it, which "
        f"says whether the harness was busy while this column said it was "
        f"not. Since version 9 an efficiency divided by a one-client "
        f"baseline whose median was more than 1.4x slower than this point's "
        f"is withheld too (`extra.efficiency_reliability: "
        f"baseline_not_credible`): a warm baseline cannot get faster as "
        f"clients are added, so that efficiency describes a cold baseline.\n",
        "\n**Server CPU** is the server container's cgroup CPU over the "
        "median window, in cores and as a share of its cpuset where the "
        "manifest recorded one; **Server CPU pressure** is the share of the "
        "window in which at least one of its runnable tasks waited for a CPU "
        "(cgroup `cpu.pressure` some), with IO pressure beside it. Both are "
        "differences of cumulative counters across the samples inside the "
        "window, joined by the window's recorded wall-clock ends, so a dash "
        "means a record from before ops measurement version 9 or a kernel "
        "that publishes no pressure — never an idle server.\n",
)


def _client_cpu_gap_note(unmeasured: int, total: int) -> str:
    """Said when SOME points lack a client-CPU figure, never when all do.

    The distinction is the whole content of the note.

    A run measured end to end by a harness older than this column has no gap
    to report. Every point lacks the figure for one uniform reason, the column
    is an unbroken row of em dashes that says so by itself, and the honest
    statement about it is nothing. Printed anyway, the paragraph lands on every
    report of every pre-existing run directory and trains a reader to skip the
    footnote — after which it is also unread on the run where it matters.

    A run where some points carry the figure and others do not is a different
    thing: the counter was readable and then was not, so the harness lost a
    measurement part-way through a sweep and the points either side of it were
    not measured the same way. That is a fault, and it stays loud.
    """
    if not unmeasured or unmeasured >= total:
        return ""
    return (
        f"\n**{unmeasured} of {total} points have no client-CPU figure (em "
        f"dash), while the rest do.** The harness container's cgroup CPU "
        f"counter was readable for some windows of this run and not for "
        f"others, so the measurement was lost part-way through rather than "
        f"never taken — and whether the clients were the ceiling at those "
        f"points is unknown rather than known to be no. Treat the marked and "
        f"unmarked points in this section as having been measured "
        f"differently.\n")


def _scaling_refused(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Concurrency points whose scaling efficiency this report will not print.

    One place, read by the footnote under the table, by the Validity section
    and by the chart, so a point drawn as refused in one and published in
    another is not possible. Three readers of one condition each deciding it
    for themselves is how a chart comes to disagree with the table beside it,
    which reads as a rendering bug rather than as the measurement it is.
    """
    return [r for r in summary.get("concurrency", []) or []
            if (r.get("extra") or {}).get("scaling_status")
            == SCALING_CONCURRENCY_NOT_ACHIEVED]


def _concurrency_notes(summary: Dict[str, Any]) -> str:
    """The fixed footnote, plus whatever this run has to say for itself."""
    rows = summary.get("concurrency", []) or []
    bound = [r for r in rows if (r.get("extra") or {}).get("client_bound")]
    unmeasured = [r for r in rows
                  if (r.get("extra") or {}).get("client_cpu_util") is None]
    early = [r for r in rows
             if ((r.get("extra") or {}).get("prewindow_discarded") or 0) > 0]

    parts = list(_CONCURRENCY_NOTES)
    if bound:
        where = ", ".join(sorted(
            f"{_label(r.get('engine', '?'))} at {r.get('clients')} clients"
            for r in bound))
        parts.append(
            f"\nClient-bound points in this run: {where}. Re-measure these "
            f"with more client CPUs before reading them as engine ceilings.\n")
    refused = _scaling_refused(summary)
    if refused:
        where = ", ".join(sorted(
            f"{_label(r.get('engine', '?'))} at {r.get('clients')} clients"
            for r in refused))
        parts.append(
            f"\n**No scaling efficiency is published for {len(refused)} "
            f"point(s) in this run: {where}.** The pool served fewer than "
            f"half the queries its client count implies at those points, so "
            f"the efficiency figure there is not a measurement of how this "
            f"engine scales. The Validity section names them with the "
            f"numbers.\n")
    parts.append(_client_cpu_gap_note(len(unmeasured), len(rows)))
    if early:
        parts.append(
            f"\n**{len(early)} point(s) report `extra.prewindow_discarded` "
            f"above zero.** That count is zero by construction — the window "
            f"opens at the instant the clients' barrier releases and no "
            f"measured query can precede it — so a non-zero value means the "
            f"window's two ends disagree about when it began, and the QPS "
            f"above is suspect. This is a framework fault, not an engine "
            f"result.\n")
    return "".join(parts)


#: What a tie cell says when there is nothing to compare against, and when
#: there is but the interval cannot support the comparison. Two different
#: facts, so two different cells -- "alone here" is the profile measuring one
#: engine at this point, and "no interval" is a record with no usable
#: `recall_ci95`, which is every record written before PR 5.
TIE_ALONE_CELL = "—"
TIE_NO_INTERVAL_CELL = "no interval"
TIE_SEPARATED_CELL = "distinct"


def _tie_row_name(record: Dict[str, Any], storages_differ: bool) -> str:
    """How a tie cell names one of the rows it is tied with.

    The engine label, plus the storage engine where the group holds more than
    one of them: a MariaDB InnoDB row tied with a MariaDB MyISAM row would
    otherwise read "tied with MariaDB", which looks like a rendering fault
    and is the same complaint `_filtered_table` carries about M and build
    mode being unconditional columns.
    """
    name = _label(record.get("engine", "?"))
    if storages_differ:
        name += f" ({_storage(record)})"
    return name


def _tie_cells(records: List[Dict[str, Any]],
               group_of,
               ci_key: str = "recall_ci95") -> Dict[int, str]:
    """PR 5 §5.1 item 6: "mark two engines as tied when their intervals
    overlap instead of ranking them".

    Returns `id(record) -> cell`, so a caller can label the rows it is
    already building in whatever order it sorts them.

    `group_of(record)` is the comparison point: two records in one group are
    directly comparable, and the ONLY difference between them is the thing
    being compared. That is where this can go wrong the way this report has
    gone wrong before -- the filter-strategy verdict divided a narrow point
    from one ops unit by a wide point from another and published 50.0x, a
    figure neither unit produced -- so each caller's group spells out every
    axis it holds constant and the storage engine, the one axis engines
    legitimately differ on, is NAMED in the cell rather than silently pooled.

    The verdict is `stats.tie_verdict`, which is three-valued, and the three
    values are all rendered:

    * TIED -> "tied with X, Y". Two engines whose intervals overlap are not
      distinguishable by this measurement and the table must not imply an
      order between them.
    * SEPARATED -> "distinct". The intervals do not overlap, so the ordering
      the numbers suggest is supported.
    * UNKNOWN -> "no interval". Either record lacks a usable interval, which
      covers an archived record, a one-query point and a `nan` from a
      measurement that did not happen. Measured: a `(nan, nan)` interval
      against a real one is False under BOTH the closed and the strict
      overlap test, so a two-valued version of this column would print
      "distinct" for a point that was never measured.
    """
    from harness.metrics.stats import (TIE_TIED, TIE_UNKNOWN, Interval,
                                       tie_verdict)

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(group_of(r), []).append(r)

    cells: Dict[int, str] = {}
    for _key, members in groups.items():
        storages_differ = len({_storage(m) for m in members}) > 1
        intervals = {
            id(m): Interval.from_field((m.get("extra") or {}).get(ci_key))
            for m in members}
        for row in members:
            others = [m for m in members if m is not row]
            if not others:
                cells[id(row)] = TIE_ALONE_CELL
                continue
            verdicts = [tie_verdict(intervals[id(row)], intervals[id(m)])
                        for m in others]
            tied = [m for m, v in zip(others, verdicts) if v == TIE_TIED]
            if tied:
                cells[id(row)] = "tied with " + ", ".join(
                    sorted(_tie_row_name(m, storages_differ) for m in tied))
            elif all(v == TIE_UNKNOWN for v in verdicts):
                # Every comparison at this point was unanswerable, so saying
                # "distinct" would be an ordering claim with no measurement
                # behind it.
                cells[id(row)] = TIE_NO_INTERVAL_CELL
            else:
                cells[id(row)] = TIE_SEPARATED_CELL
    return cells


#: What the `Build mode` column in sections 7 and 9 is, and what it is not.
#:
#: **Those two sections print the DECLARED mode and cannot print what
#: happened, so they must not look as though they do.** `cli.py`'s ops loop
#: defaults the mode to `post` for any engine that does not declare
#: `capabilities.bulk_vector_index_build: false`, and for MHNSW and VIDX --
#: which maintain their graph on every INSERT and have no separable build
#: step -- `post` is a no-op. So a MariaDB concurrency row reads `post` beside
#: an index that was built incrementally. The column is still right as an
#: identity: it is the ops-unit axis, and two rows differing only in it are
#: two different indexes. It is wrong only if read as a description of the
#: build.
#:
#: Section 5 is where the description lives, and it is deliberately a SECOND
#: column there: `_index_build_kind` classifies what the record actually says
#: happened and carries a distinct "not recorded" value for records that
#: predate the field. Sections 7 and 9 cannot compute it -- `separable_build`
#: is written on `index_build` records and on nothing else (measured on
#: `full-20261015-215006`: present on 11 of 11 build records, absent from all
#: 55 concurrency and all 70 churn records), so deriving it here would mean
#: joining a concurrency row to a build row, which is the exact defect this
#: table was just fixed for. Pointing at section 5 is the honest move;
#: inventing the join is not.
#:
#: One constant, two readers, for the reason `TIE_COLUMN_NOTE` below is one:
#: two sections wording the same caveat differently is how one of them comes
#: to be edited into saying something the other contradicts.
BUILD_MODE_COLUMN_NOTE = (
    "\n_**“Build mode”** is the ops-unit axis this row was measured in, not a "
    "statement about how the index was built. The orchestrator defaults it to "
    "`post` for every engine that does not declare otherwise, and MHNSW and "
    "VIDX build their graph on every INSERT — for them `post` is a no-op and "
    "the index is incremental whatever this cell says. Two rows differing "
    "only in this column are two different indexes, which is why it is here. "
    "What each build actually did is section 5's `Index build` column, which "
    "is derived from the build record rather than from the requested mode._\n")


#: The footnote under every table that carries a tie column. One string, used
#: by both, because two copies of an explanation drift and the churn table and
#: the filtered table are explaining the same column.
TIE_COLUMN_NOTE = (
    "\n_**“Tied with”** is PR 5 §5.1 item 6: where two engines' 95% "
    "bootstrap intervals for the recall in this row overlap, the report says "
    "so instead of letting the two numbers imply an order. The interval is "
    "resampled from the per-query recalls behind that one mean, so it "
    "describes how well this measurement pins the engine's recall — not how "
    "much the engine varies between runs. **“distinct”** means the intervals "
    "do not overlap and the ordering the numbers suggest is supported. "
    "**“no interval”** means at least one row at this point carries no usable "
    "interval — an archived record, or a point that measured too few queries "
    "for a spread to mean anything — and the comparison is therefore not "
    "available rather than negative. A dash means no other engine was "
    "measured at this exact point, so there was nothing to compare._\n")


def _filtered_table(summary: Dict[str, Any]) -> str:
    """One row per (engine, dataset, ops unit, column, selectivity, mode).

    **Pass, M and build mode are columns because the ops phase loops them.**
    `_run_unit` iterates `m_values x build_modes x storage_engines` inside each
    resource pass, and every pass through it reloads the corpus, rebuilds the
    index and writes its own full column x selectivity grid. Without those
    columns `full` gives mariadb three indistinguishable rows per (column,
    selectivity) and pgvector six -- verbatim the "four real measurements
    rendered as the same row four times and read as a rendering fault" that
    the short-results block in Validity cites as its own reason for carrying
    them. They are rendered unconditionally rather than only when they vary,
    which is what that block does: a column that appears and disappears is a
    second thing for a reader to notice.

    Selectivity is printed to two decimal places, not zero. `tag` at a nominal
    1% delivers 1.00% and `ctag` delivers 1.90% on the 2,000-row fixture --
    and up to 4.96% on a 100,000-row one depending only on the clustering seed
    -- so `{:.0%}` renders both as "2%" or both as "1%" and erases the
    difference the second column exists to show. The share here is counted,
    not requested: `nominal` is beside it so a reader can see both.

    The near_pass / near_fail columns are the heart of PR 3 §3.3 item 3. A
    query whose true unfiltered nearest neighbour passes the filter has its
    answer inside the candidate list the engine was going to walk anyway; one
    whose nearest neighbour fails has to keep looking. For `tag` the two
    groups track the selectivity and the recalls are close; for `ctag` the
    near_fail group is where post-filtering collapses. A dash means the group
    was empty, which is a real state and not a missing measurement -- `tag` at
    1% put zero of fifty queries in near_pass, measured, because a uniform 1%
    of rows contains a query's nearest neighbour about 1% of the time.
    """
    # §5.1 item 6's comparison point for a filtered row: everything held
    # constant except the engine and the storage it ran on. The threshold and
    # not the measured selectivity, because the threshold is the knob the
    # engines were all given and is equal by construction, while the share it
    # delivers is counted from the corpus and would make a float the group
    # key -- `tag` at a nominal 1% delivers 1.000073% on glove, and two
    # engines agreeing to eleven places is a coincidence to rely on, not an
    # invariant. The ops-unit axes are all here (`m`, `build_mode`,
    # `resource_pass`); `storage_engine` is deliberately NOT, because engines
    # legitimately differ on it -- pgvector's heap against MariaDB's InnoDB --
    # and a group that held it constant would never compare two engines at
    # all. It is named in the cell instead, by `_tie_row_name`.
    def filtered_group(r):
        extra = r.get("extra") or {}
        threshold = extra.get("filter_threshold")
        if threshold is None:
            threshold = extra.get("tag_threshold")
        if threshold is None:
            # Archived records carry neither. Rounded hard, so two engines'
            # counted shares of the same corpus still land together.
            threshold = round(r.get("selectivity") or 0.0, 6)
        return (r.get("dataset"), _filter_column(r), threshold,
                extra.get("iterative_scan"), r.get("resource_pass"),
                r.get("m"), r.get("build_mode"), r.get("ef_search"))

    tie_cells = _tie_cells(summary.get("filtered", []), filtered_group)

    rows = []
    for r in sorted(summary.get("filtered", []),
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   r.get("resource_pass") or "",
                                   # `or 0`, not `str()`: as text, M 8/16/32
                                   # sorts 16, 32, 8. `full` sweeps exactly
                                   # that grid, and `_build_table` in this
                                   # same file already does it this way.
                                   r.get("m") or 0,
                                   r.get("build_mode") or "",
                                   r.get("storage_engine") or "",
                                   _filter_column(r),
                                   r.get("selectivity") or 0)):
        extra = r.get("extra") or {}
        nominal = extra.get("nominal_selectivity")
        executed, planned = r.get("queries_executed"), extra.get("queries_planned")
        queries = _fmt(executed, 0) if executed is not None else "—"
        if planned and executed is not None and executed < planned:
            queries = f"{executed:,}/{planned:,}"
            if extra.get("insufficient_sample"):
                queries += " ✕"
            elif extra.get("time_bounded"):
                queries += " □"
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            r.get("resource_pass") or "—",
            _fmt(r.get("m"), 0),
            r.get("build_mode") or "—",
            _storage(r),
            _filter_column(r),
            f"{(r.get('selectivity') or 0):.2%}",
            "—" if nominal is None else f"{nominal:.0%}",
            _fmt(r.get("recall_at_k"), 4),
            _fmt(extra.get("recall_near_pass"), 4),
            _fmt(extra.get("recall_near_fail"), 4),
            f"{extra.get('near_pass_queries', '—')}/"
            f"{extra.get('near_fail_queries', '—')}",
            _fmt(r.get("qps"), 0),
            _fmt(r.get("latency_p99_ms"), 3),
            "yes" if r.get("vector_index_used") else "no",
            str(extra.get("short_result_queries", 0)),
            queries,
            extra.get("iterative_scan") or "—",
            tie_cells.get(id(r), TIE_ALONE_CELL),
        ])
    return _md_table(
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage", "Column",
         "Selectivity", "Nominal", "Recall@k", "Recall near_pass",
         "Recall near_fail", "pass/fail n", "QPS", "p99 (ms)", "Index used",
         "Short results", "Queries", "Iterative scan", "Tied with"],
        rows,
    ) + TIE_COLUMN_NOTE + FILTERED_EF_NOTE


#: Filtered search is not calibrated on its own. The harness substitutes the
#: ef the UNFILTERED sweep chose and runs every predicate at it, so a filtered
#: recall below the target is the predicate's cost at that ef, not a failed
#: calibration -- and a reader who assumes each filtered point was tuned to
#: the target reads the recall columns backwards.
FILTERED_EF_NOTE = (
    "\n_Filtered search reuses the unfiltered calibrated ef: every row here "
    "ran at the ef_search the unfiltered sweep chose for its unit (§6), with "
    "no calibration of its own, so a recall below the target is what the "
    "predicate cost at that ef._\n")


def _filter_column(record: Dict[str, Any]) -> str:
    """The filter column a filtered record measured.

    Absence means `tag`: every filtered record written before PR 3 had one
    column and it was `tag`. Imported from `report.generate` rather than
    re-spelled, so the renderer and the summariser cannot disagree about what
    an archived record measured.
    """
    from report.generate import filter_column
    return filter_column(record)


def _filter_strategy_table(summary: Dict[str, Any]) -> str:
    """Which filtering strategy each engine used, from its own latency.

    Every engine's documentation says what it does. This says what it did. A
    pre-filter compares only the qualifying vectors, so a narrower predicate
    is less work; anything that walks the graph pays more, because it has to
    travel further to find k survivors. The direction separates them without
    anyone being taken at their word -- which matters, because the report
    asserted for weeks that one engine pre-filtered when its own numbers said
    otherwise.
    """
    strategies = summary.get("filter_strategies") or []
    if not strategies:
        return ""
    # The column is part of the key, so it has to be part of the row: the
    # ratio is derived from the narrowest and widest point of ONE column's
    # curve, and two rows for one engine are two predicates rather than a
    # repeated measurement.
    parts = [
        "\n#### What each engine actually did\n\n",
        "Derived from how median latency moves as the predicate narrows, not "
        "from what the engine documents. **A pre-filter gets faster** — fewer "
        "qualifying vectors is less work. Anything that applies the predicate "
        "during or after the graph walk gets slower, because it has to travel "
        "further to find k survivors.\n\n"
        "**One row per corpus per filter column per ops unit, and they can "
        "disagree.** "
        "The ratio is taken across the two ends of ONE curve — one dataset, "
        "one column, "
        "one iterative-scan mode, one resource pass, and one (M, build mode, "
        "storage engine) unit. Every one of those changes the answer, and the "
        "ops phase loops the last three: pgvector's `post` and `incremental` "
        "builds are different indexes, MariaDB's InnoDB and MyISAM are "
        "different storage, and the two resource passes give the engine "
        "different memory and cpusets. Pooling any of them divides two "
        "latencies that never ran together and publishes the result as a "
        "verdict about an engine — measured on one engine and one column over "
        "two build modes, 0.20x `pre-filter` and 1.00x `post-filter, abandons "
        "the query` separately, against a pooled 50.0x `post-filter, "
        "exhaustive` that neither build mode produced. An engine that "
        "pre-filters on an indexed `tag` predicate and walks the graph for a "
        "`ctag` one is a real and interesting outcome, which a single row "
        "could not express.\n",
    ]
    rows = [[_label(s["engine"]),
             # **`Dataset` is a fix, not an addition.** This table named every
             # ops-unit axis and not the corpus, so two datasets' curves
             # collapsed into repeated rows: on `full-20261015-215006` its 21
             # rows carried only 12 distinct identities, 8 of them appearing
             # two or three times -- `tuned | 16 | incremental | InnoDB | tag`
             # twice, once at `0.7 -> 10.1 ms` and once at `0.9 -> 15.8 ms`.
             # Those are two corpora, and a reader had the same two readings
             # available as everywhere else in this series: a noisy
             # measurement, or a duplicated row. `generate.py` has always put
             # `dataset` in the grouping key and carried it on every entry;
             # only the renderer dropped it.
             s.get("dataset") or "—",
             s.get("resource_pass") or "—",
             _fmt(s.get("m"), 0),
             s.get("build_mode") or "—",
             s.get("storage_engine") or "—",
             s.get("filter_column") or "tag",
             f"{s['wide']:.2%} → {s['narrow']:.2%}",
             f"{s['wide_p50']:,.1f} → {s['narrow_p50']:,.1f} ms",
             f"{s['ratio']}x",
             _fmt(s.get("narrow_recall"), 4),
             s.get("iterative_scan") or "—",
             s["observed"]]
            for s in strategies]
    # **Deliberately NOT sorted by the unit axes**, unlike sections 5, 7 and
    # 9. `generate.filter_strategies` ranks these by `ratio`, because the
    # question this table answers is "which engines' latency moved, and how
    # hard" -- reading it means scanning the Change column, so grouping by
    # unit would scatter the ranking that is its whole point. Contiguity is
    # the invariant in the grouped tables; it is not one here, and the
    # identity columns are what keeps two rows apart instead.
    parts.append(_md_table(
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage", "Column",
         "Selectivity", "Median latency", "Change",
         "Recall at the narrow end", "Iterative scan", "Observed strategy"],
        rows))
    return "".join(parts)


def _churn_table(summary: Dict[str, Any]) -> str:
    """One row per churn measurement, with the state that tells two apart.

    **The `Repair` column is a fix, not an addition.** PR 4 §4.2 writes a
    `"before"` and an `"after"` record per churn fraction -- the index as the
    churn left it, and the index after the engine's repair path ran -- and
    until this column existed nothing in `report/` read
    `extra.maintenance_state`. Two rows per fraction were emitted whose
    engine, dataset, storage and churn columns were byte-identical, carrying
    recalls 0.8960 and 0.9980 and index sizes 275.0 KiB and 250.0 KiB. Both
    pairs reproduced exactly in 9 of 9 runs of one configuration on the
    synthetic driver. A reader had two readings available -- "a noisy
    measurement" and "a duplicate row" -- and both were false.

    The dash is for a record that names no state: the pre-churn baseline, and
    every churn record written before §4.2. Those measure one state and do not
    say which, so claiming either would be this column inventing the thing it
    exists to report.

    `churn_measurements`, not `summary["churn"]`: the reader-pool records §4.3
    writes are the same phase with a `churn_fraction` and a multi-client QPS,
    and they belong in their own table below rather than in a column of
    single-client ones.

    **`Pass`, `M` and `Build mode` are a fix of the same shape as `Repair`.**
    `churn_group` below has always held all three constant when deciding what
    a row is tied with, so the renderer knew they were part of a row's
    identity while the header named none of them -- and the churn-under-writes
    table immediately below this one prints all three.
    `full-20261015-215006` swept M 8/16/32 over two resource passes, so every
    (dataset, fraction, state) appeared five times with the engine, dataset,
    storage, churn and repair cells byte-identical: at 10% before repair,
    recalls 0.9865 / 0.9594 / 0.9672 / 0.9860 / 0.9598 and QPS 1,503 / 1,898 /
    1,561 / 1,331 / 1,829, which reads as one noisy measurement of one index
    instead of five readings of five. The pass matters most: it gave the
    server 8 cores and 16 GB under `normalized` against 76 cores and
    112.5 GiB under `tuned` in that run, so a QPS column with no pass beside
    it is not a comparison at all. `build_mode` is in for the reason
    `_concurrency_table`'s comment gives -- the profiles that set
    `pgvector_build_modes: [post, incremental]` run an ops unit per mode, and
    a bulk-built index and an incrementally maintained one are two indexes,
    not one index measured twice.
    """
    from report.churn_state import churn_measurements, maintenance_state

    measurements = list(churn_measurements(summary.get("churn", [])))
    points = _operating_point_index(summary)

    # §5.1 item 6's comparison point for a churn row. The maintenance state
    # is in the key and that is the whole reason this table has a `Repair`
    # column: §4.2 writes a "before" and an "after" record per fraction, and
    # pooling them made two readings of two different indexes look like a
    # noisy measurement of one. Tying a degraded index to a repaired one
    # would be the same defect wearing §5.1's clothes.
    #
    # `storage_engine` is excluded for the reason `filtered_group` gives --
    # engines differ on it legitimately -- and named in the cell instead.
    def churn_group(r):
        return (r.get("dataset"), r.get("churn_fraction"),
                maintenance_state(r), r.get("resource_pass"), r.get("m"),
                r.get("build_mode"), r.get("ef_search"))

    tie_cells = _tie_cells(measurements, churn_group)

    rows = []
    for r in sorted(measurements,
                    key=lambda r: (r.get("dataset", ""), r.get("engine", ""),
                                   # **The four unit axes sort ahead of BOTH
                                   # the fraction and the state rank, and the
                                   # state rank stays last.** That precedence
                                   # is the whole legibility claim and it is
                                   # the one thing here a green suite does not
                                   # protect: move these four behind the state
                                   # rank and the page still renders, every
                                   # number still correct, but a unit's
                                   # `before` and `after` rows end up as many
                                   # rows apart as the grid is wide -- five, on
                                   # `full-20261015-215006` -- which is the
                                   # "two readings of one noisy index" shape
                                   # the `Repair` column exists to prevent.
                                   # `m or 0`, not `str(m)`, for the reason
                                   # `_filtered_table` gives.
                                   r.get("resource_pass") or "",
                                   r.get("m") or 0,
                                   r.get("build_mode") or "",
                                   r.get("storage_engine") or "",
                                   r.get("churn_fraction") or 0,
                                   # Ranked, not sorted as strings: "after"
                                   # sorts BEFORE "before" alphabetically,
                                   # which put the repaired row above the
                                   # degraded one and reversed the order the
                                   # two were measured in.
                                   _CHURN_STATE_ORDER.get(
                                       maintenance_state(r), 99))):
        extra = r.get("extra") or {}
        state = maintenance_state(r)
        rows.append([
            _label(r.get("engine", "?")), r.get("dataset", "?"),
            # The three axes `churn_group` above already holds constant, now
            # named on the page as well. Same spelling and same order as
            # `_filtered_table` and `_churn_during_writes_table`.
            r.get("resource_pass") or "—",
            _fmt(r.get("m"), 0),
            r.get("build_mode") or "—",
            _storage(r),
            f"{(r.get('churn_fraction') or 0):.0%}",
            _churn_state_cell(state, extra),
            _fmt(r.get("recall_at_k"), 4),
            _fmt(extra.get("recall_drop_vs_baseline"), 4),
            _fmt(r.get("qps"), 0),
            _fmt_bytes(r.get("index_bytes")),
            _reinsert_ratio_cell(r, summary.get("ingest") or []),
            tie_cells.get(id(r), TIE_ALONE_CELL),
            *_operating_point_cells(r, points),
        ])
    return _md_table(
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage", "Churn",
         "Repair", "Recall@k", "Recall drop vs baseline", "QPS", "Index size",
         "Re-insert / load", "Tied with", "Calibrated ef",
         "Calibration recall"],
        rows,
    ) + BUILD_MODE_COLUMN_NOTE + TIE_COLUMN_NOTE + OPERATING_POINT_NOTE \
        + REINSERT_RATIO_NOTE \
        + _churn_notes(summary) + _churn_during_writes_table(summary)


#: What the Re-insert / load column means and when it is withheld.
REINSERT_RATIO_NOTE = (
    "\n_**Re-insert / load** is the churn re-insert's accepted rate over the "
    "same unit's ingest rate, printed only where both were written with the "
    "same number of connections (`extra.reinsert_writers` = "
    "`extra.ingest_threads`, ops measurement version 9). `full-20260825` "
    "compared one re-insert writer with an eight-thread load, which measures "
    "the thread pool rather than the index; a row that cannot show equal "
    "counts says *withheld*. The load side is the WHOLE load's average rate: "
    "the ingest record keeps no progress samples, so the rate over its last "
    "10% of rows -- the honest comparison for an incremental graph, which "
    "slows as it grows -- is not available, and for MHNSW, VIDX and "
    "Percona's incremental build this ratio is therefore biased low._\n")


def _reinsert_ratio_cell(row: Dict[str, Any],
                         ingest: List[Dict[str, Any]]) -> str:
    """The re-insert/load ratio, or why it is not printed (ops v9, F9)."""
    extra = row.get("extra") or {}
    rate = extra.get("reinsert_rows_per_s_accepted")
    if rate is None:
        return "—"
    writers, threads = (extra.get("reinsert_writers"),
                        extra.get("ingest_threads"))
    if writers is None or threads is None or writers != threads:
        return (f"withheld ({writers} vs {threads} writers)"
                if writers is not None and threads is not None
                else "withheld")
    unit = ("engine", "dataset", "resource_pass", "m", "build_mode",
            "storage_engine")
    load = next((i.get("ingest_rows_per_s") for i in ingest
                 if all(i.get(k) == row.get(k) for k in unit)
                 and i.get("ingest_rows_per_s")), None)
    if not load:
        return "—"
    return f"{rate / load:.2f}x"


#: The order the churn states are measured in, which is the order they are
#: rendered in. A plain string sort puts "after" first.
_CHURN_STATE_ORDER = {None: 0, "before": 1, "after": 2}


def _empty_or(value: Any, is_empty: Optional[bool], places: int) -> str:
    """A figure, or an em dash where the window that produced it was empty.

    Separate from `_fmt` because the two absences mean different things:
    `_fmt(None)` is "nobody measured this", and this is "the window was open
    for less than one query, so the 0.0 in the record is arithmetic rather
    than a measurement".
    """
    if is_empty:
        return "—"
    return _fmt(value, places)


def _churn_state_cell(state: Optional[str], extra: Dict[str, Any]) -> str:
    """The `Repair` cell: which side of the repair path this row measured.

    Named with what actually ran where the driver said -- `extra.maintenance`
    is `maintain()`'s own description, "VACUUM (ANALYZE)" for pgvector -- so
    the column reports the mechanism rather than only the word "after". The
    engines with no repair path write one record per fraction carrying
    `repair_path = "none"`, which is what they wrote before PR 4, and they get
    the `before` cell and no second row.
    """
    if state is None:
        return "—"
    if state != "after":
        return state
    what = extra.get("maintenance")
    seconds = extra.get("maintenance_s")
    cell = f"after {what}" if what else "after"
    return cell + (f" ({seconds:,.1f} s)" if isinstance(seconds, (int, float))
                   else "")


def _churn_during_writes_table(summary: Dict[str, Any]) -> str:
    """PR 4 §4.3: what a read costs while the engine is taking writes.

    The headline figure is the p99 ratio, and the table shows both percentiles
    beside it so a reader can do the division. That is not a courtesy: it is
    the only way a reader can check that the two numbers being divided
    describe the same thing, and this repository has published two ratios that
    did not. The filter-strategy ratio divided a narrow point from one ops
    unit by a wide point from another; `charts.churn_retention` pairs records
    from two different maintenance states. Both read as findings.

    One row per (engine, dataset, fraction, ops unit), never one per engine,
    for the same reason the filter-strategy table carries Pass, M, Build mode
    and Storage: pgvector's `post` and `incremental` builds are different
    indexes and the two resource passes give the engine different memory and
    cpusets, so pooling them divides two latencies that never ran together.

    **A ratio here is not a figure to quote to two decimals.** It is a ratio
    of two wall-clock percentiles. On the synthetic driver the same churn
    configuration's QPS ranged from 6,028 to 16,811 over nine runs -- a factor
    of 2.79 -- while its seeded recall did not move in the last digit. What
    the measurement supports is the direction and the rough magnitude of the
    cost, and the reader count beside it.
    """
    rows_in = summary.get("churn_during_writes") or []
    if not rows_in:
        return ""
    points = _operating_point_index(summary)
    parts = [
        "\n**Reads while the engine is taking writes.** Every number in this "
        "table comes from one pool of readers measured twice: for 30 s before "
        "the step's deletes, and again throughout the re-insert. Both windows "
        "share the client count, the query pool, k, the ef_search and the ops "
        "unit — the harness refuses to divide them otherwise — so the ratio "
        "is a property of the write load and not of two configurations. "
        "Readers compute no recall here: the corpus is mid-change, so no "
        "ground truth describes it.\n\n"
        "The ratio is a ratio of two wall-clock percentiles and should be "
        "read as a direction and a magnitude rather than to two decimals. "
        "The writer rate beside it is the re-insert rate measured *with* "
        "those readers running, which is not the same quantity as the "
        "write-only rate; `ops.churn_readers: 0` measures that one.\n",
    ]
    table_rows = []
    for s in rows_in:
        ratio = s.get("p99_ratio")
        ratio_cell = f"{ratio:,.2f}x" if ratio is not None else "—"
        if s.get("ratio_disagrees"):
            # Loud in the cell, because a reader scanning the column is the
            # person who needs to know the record's two halves disagree.
            ratio_cell += " ⚠"
        if s.get("client_bound"):
            # The same hollow-point convention the concurrency chart uses: the
            # readers may have been part of the ceiling, so the figure is not
            # evidence about the engine.
            ratio_cell += " ○"
        if s.get("scaling_status") == SCALING_CONCURRENCY_NOT_ACHIEVED:
            # Stronger than the hollow point, and on the WRITER cell as well
            # below: the readers did not run concurrently, so this p99 is a
            # read that waited for a lock rather than one the engine was slow
            # to answer, and the writer beside it was starved in the client.
            ratio_cell += f" {SCALING_REFUSED_MARK}"
        # A window that counted no query has no throughput and no
        # percentile, and the cell says so rather than printing the 0.0 the
        # record carries. `clients.client_cpu` makes the argument for the
        # client-CPU figure and it is the same argument here: zero is the
        # claim "the readers did nothing", which is the one thing that cannot
        # be true of a window that executed queries. Found on a synthetic
        # driver whose re-insert returns in ~1 ms, where the row read
        # `Reader QPS: 186 → 0` with nothing to say the window never opened.
        baseline_empty = s.get("baseline_empty")
        under_empty = s.get("under_writes_empty")
        table_rows.append([
            _label(s.get("engine") or "?"), s.get("dataset") or "?",
            s.get("resource_pass") or "—",
            _fmt(s.get("m"), 0),
            s.get("build_mode") or "—",
            s.get("storage_engine") or "—",
            f"{(s.get('churn_fraction') or 0):.0%}",
            _fmt(s.get("readers"), 0),
            f"{_empty_or(s.get('baseline_p99_ms'), baseline_empty, 3)} → "
            f"{_empty_or(s.get('under_writes_p99_ms'), under_empty, 3)} ms",
            ratio_cell,
            f"{_empty_or(s.get('baseline_qps'), baseline_empty, 0)} → "
            f"{_empty_or(s.get('under_writes_qps'), under_empty, 0)}",
            _fmt(s.get("writer_rows_per_s"), 1, " rows/s")
            + (f" {SCALING_REFUSED_MARK}"
               if s.get("scaling_status")
               == SCALING_CONCURRENCY_NOT_ACHIEVED else ""),
            # What the readers actually had in service, against how many there
            # were. The evidence for both marks above, in the row that carries
            # them -- mean-based first and the median estimate second since
            # ops version 9, through the one cell function the concurrency
            # table uses, so the two tables cannot read the pair differently.
            _served_cell(s, s.get("readers")),
            *_operating_point_cells(s, points),
        ])
    parts.append(_md_table(
        ["Engine", "Dataset", "Pass", "M", "Build mode", "Storage", "Churn",
         "Readers", "Reader p99", "p99 under writes", "Reader QPS",
         "Writer rate", "Reads in service", "Calibrated ef",
         "Calibration recall"],
        table_rows))
    if any(s.get("ratio_disagrees") for s in rows_in):
        parts.append(
            "\n⚠ The ratio this report computed from the record's own two "
            "percentiles differs from the one the harness stamped into the "
            "same record by more than 1%. The two halves of that record do "
            "not describe each other, so neither number is reportable: read "
            "the raw `extra.reader_baseline_p99_ms` and "
            "`extra.reader_under_writes_p99_ms` before quoting anything from "
            "that row.\n")
    empty = [s for s in rows_in
             if s.get("baseline_empty") or s.get("under_writes_empty")]
    if empty:
        windows = ", ".join(
            f"{(s.get('churn_fraction') or 0):.0%} "
            f"({(s.get('under_writes_window_s') or 0) * 1000:.0f} ms)"
            for s in empty)
        parts.append(
            f"\n**A blank reader cell means the window closed before a single "
            f"reader query finished**, not that the engine answered none. A "
            f"during-writes window is exactly as long as the re-insert takes, "
            f"so this happens when the write path returns faster than one "
            f"read: {windows}. The record keeps the evidence — "
            f"`extra.reader_under_writes_queries_executed` is 0 and "
            f"`extra.reader_under_writes_window_s` is the length it managed. "
            f"A zero printed here would read as a throughput collapse, which "
            f"is the one thing it is not.\n")
    serialized = [s for s in rows_in
                  if s.get("scaling_status")
                  == SCALING_CONCURRENCY_NOT_ACHIEVED]
    if serialized:
        parts.append(
            f"\n{SCALING_REFUSED_MARK} **The readers did not run "
            f"concurrently at {len(serialized)} of these points**, so neither "
            f"the p99 ratio nor the writer rate beside it is evidence about "
            f"the engine. `Reads in service` is, first, the sum of every "
            f"in-window read's duration over the window (mean) and, second, "
            f"the reads counted times their median latency (median est.); "
            f"since ops measurement version 9 the refusal reads the first, so "
            f"below half the reader count the readers did not keep their reads "
            f"in flight. A median estimate far below the mean is "
            f"{TAIL_DOMINATED_MARK} `tail_dominated` -- a heavy-tailed read "
            f"latency, not a refusal; records from before version 9 were "
            f"judged on the median and keep that verdict. **The "
            f"writer rate was measured against those same reads**, so it is "
            f"not a figure about an engine under a known read load either; "
            f"the churn step record carries this status as "
            f"`extra.reader_scaling_status` for exactly that reason, so "
            f"`extra.reinsert_rows_per_s` cannot be read without it. **What "
            f"is no longer the mechanism**: until `ops_measurement_version` "
            f"6 the readers were threads in the writer's own interpreter, and "
            f"reads that serialized on the interpreter lock would starve the "
            f"writer inside the client rather than inside the engine, which "
            f"made that rate a measurement of Python. Each reader is now a "
            f"forked OS process and the writer is the harness's own, so no "
            f"interpreter lock is shared between them and whatever contention "
            f"this row shows is in the engine or in the client container's "
            f"CPUs. `extra.client_cpu_seconds` is the figure for the second "
            f"of those.\n")
    if any(s.get("client_bound") for s in rows_in):
        parts.append(
            "\n○ The readers used more than 80% of their container's CPUs, so "
            "the reader figures at that point are bounded by the readers as "
            "well as by the engine. Note the limit is sufficient and not "
            "necessary — an unmarked point is not a guarantee that the "
            "readers were not part of the ceiling, because a CPU threshold "
            "cannot see an interpreter-lock ceiling.\n")
    return "".join(parts)


def _churn_notes(summary: Dict[str, Any]) -> str:
    """An engine that could not write its rows back.

    Blank cells read as a measurement that was not taken. This one was: the
    engine took the writes and did not finish them, and how far it got is the
    result. Reported here rather than left to be inferred from an empty row.
    """
    stalled = [r for r in summary.get("churn", [])
               if (r.get("extra") or {}).get("reinsert_completed") is False]
    if not stalled:
        return ""
    lines = ["\n**Re-insertion did not complete for every engine.** Recall and "
             "throughput are omitted for these rows on purpose: the corpus is "
             "missing rows, so a figure taken then describes neither the "
             "baseline nor a churned corpus. What the engine managed is the "
             "measurement.\n"]
    rows = []
    for r in stalled:
        extra = r.get("extra") or {}
        rows.append([
            _label(r.get("engine", "?")),
            f"{extra.get('rows_reinserted', 0):,} of "
            f"{extra.get('rows_expected', 0):,}",
            _fmt(extra.get("insert_seconds"), 0, " s"),
            _fmt(extra.get("reinsert_rows_per_s"), 1, " rows/s"),
            _fmt(extra.get("churn_budget_s"), 0, " s"),
        ])
    lines.append(_md_table(
        ["Engine", "Rows re-inserted", "Spent", "Rate", "Budget"], rows))
    return "".join(lines)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

#: The pass/fail marks on a cross-check row. Characters and not words, so the
#: column stays narrow enough that the two recalls sit side by side; the
#: footnote under the table says what each one means, and both are also
#: spelled in the row's own "Within tolerance?" wording so a reader who
#: cannot see the glyph is not left guessing.
CROSSCHECK_PASS_MARK = "✓"

#: Marks a cell whose axis was not part of the match. Not a failure and not a
#: warning: the row is a real match, and this says which axis it was made
#: without, because a reader comparing the row against the raw JSONL would
#: otherwise find a `build_mode` the table shows and the match ignored.
CROSSCHECK_UNMATCHED_AXIS_MARK = "°"
CROSSCHECK_FAIL_MARK = "✗"


def _consistency_section(summary: Dict[str, Any]) -> str:
    """PR 5 §5.2 item 3's "Consistency checks" section.

    Every match with both values and a pass/fail mark, the coverage that
    explains an empty list, and §5.2 item 5's QPS-order inversions.

    The empty case is the one this function is really for. §5.2 item 3 says
    "An empty match list is itself reported, not hidden", and an empty list
    has two causes that must not be printed the same way: a profile whose
    `ann.m_values` and `ops.m_values` do not overlap cannot match anything and
    nothing is wrong, while a profile whose grids DO overlap and still matches
    nothing has a fault. The coverage table names which of the two happened,
    per engine and dataset.

    **On `main` the grids meet, so an empty section there is the fault case.**
    That profile sweeps `ann.m_values: [8, 16, 32]` against
    `ops.m_values: [16]`; the 16 is in the ann grid precisely so this check
    has something to compare, and every engine should produce matched rows.

    An earlier revision of this docstring said the opposite -- it described
    the `[8, 32]` grid `main` carried before PR 5 §5.2 added the 16, and
    called an empty section there "nothing is wrong". The same pull request
    changed the profile, so that sentence shipped already false. An operator
    who met it after an empty section on a `main` run would have filed a
    fault as expected behaviour, which is the one reading this function's
    coverage table exists to prevent.
    """
    consistency = summary.get("consistency") or {}
    matches = consistency.get("matches") or []
    coverage = consistency.get("coverage") or []
    inversions = consistency.get("qps_inversions") or []
    parts: List[str] = []

    if matches:
        rows = []
        for row in matches:
            mark = (CROSSCHECK_PASS_MARK if row["passed"]
                    else CROSSCHECK_FAIL_MARK)
            verdict = ("agrees" if row["passed"] else "DISAGREES")
            rows.append([
                _label(row.get("engine", "?")), row.get("dataset", "?"),
                str(row.get("resource_pass") or "—"),
                str(row.get("m") if row.get("m") is not None else "—"),
                str(row.get("ef_search")
                    if row.get("ef_search") is not None else "—"),
                str(row.get("storage_engine") or "—"),
                # The engine's own build mode, with a mark where the axis was
                # not part of the match because the recall path never declared
                # one. Without the mark this cell would read like a matched
                # axis, which for the MySQL family it is not.
                ((str(row.get("ops_build_mode") or "—")
                  + (" " + CROSSCHECK_UNMATCHED_AXIS_MARK
                     if row.get("build_mode_axis_dropped") else ""))),
                _fmt(row["recall_ann"], 4),
                _fmt(row["recall_ops"], 4),
                _fmt(row["delta"], 4),
                # The tolerance AND which branch of `max(0.01, 2 x half)`
                # produced it. Without the basis a reader cannot tell a
                # CI-derived tolerance from the 0.01 floor a record with no
                # usable interval falls back to, and those are different
                # statements about how well the comparison is pinned.
                f"{_fmt(row['tolerance'], 4)} "
                f"({'CI' if row['tolerance_basis'] == 'ci' else 'floor'})",
                f"{mark} {verdict}",
            ])
        parts.append(_md_table(
            # "Recall gap" and not the specification's "|Δ recall|".
            # Every table here is pipe-delimited markdown and
            # `_md_tables_to_html` splits a row on every pipe, so a header
            # cell containing two of them became three cells and the header
            # row carried two more than the body. Measured on the synthetic
            # run before this was renamed.
            ["Engine", "Dataset", "Pass", "M", "ef_search", "Storage",
             "Build mode", "Recall (ann)", "Recall (ops)", "Recall gap",
             "Tolerance", "Within tolerance?"],
            rows,
        ))
        failed = [r for r in matches if not r["passed"]]
        # What the sentence below may NOT claim. The MariaDB, MariaDB 12.3 and
        # AliSQL ann modules write no `build_mode`, so for those engines the
        # axis is dropped and the match was made without it. Saying "matched
        # on build mode" and leaving it there would be the same class of
        # defect as the filter-strategy verdict this report carried for weeks
        # off a vendor's documentation, having measured nothing of the kind.
        dropped = sorted({_label(r.get("engine", "?")) for r in matches
                          if r.get("build_mode_axis_dropped")})
        # Quantization is in the key and reads None on both sides for every
        # engine in every shipped profile: `context.record_defaults` does not
        # put `IndexSpec.quantization` on an ops record, and the ann loader
        # does not copy the attribute the mongodb module writes into the
        # result file. Saying the rows "matched on quantization" without
        # saying that credits the check with a discrimination it did not make
        # -- the same class of claim as the filter-strategy verdict above.
        #
        # Neither of these two comments quotes that verdict's wording, and
        # that is a constraint rather than a style choice:
        # `tests/test_config.py:TestFilterStrategyIsMeasuredNotQuoted`
        # asserts the sentence is absent from this file, and the guard is a
        # substring search -- it cannot tell a cautionary citation from the
        # claim, which is correct of it, because the next reader cannot
        # either. It also searches the file as written, so a quotation broken
        # across a line wrap passes it while the sentence is still there to
        # anyone who reflows the paragraph. That is exactly how an earlier
        # revision of these two comments carried it.
        quantized = sorted({str(r.get("quantization")) for r in matches
                            if r.get("quantization") is not None})
        if quantized:
            quantization_note = (
                "\n_Quantization was matched on, and the values present are "
                + ", ".join("`%s`" % q for q in quantized) + "._\n")
        else:
            quantization_note = (
                "\n_**Quantization is in the match key but told nothing "
                "apart here**: every row on both sides reads no quantization. "
                "Percona Search is the only engine that compresses indexed "
                "vectors, and neither path puts the value where the other can "
                "see it \u2014 the ops harness takes `--quantization` but does "
                "not record it, and the recall path writes it into the result "
                "file's attributes without the loader copying it onto the "
                "record. The key carries it so that a path which starts "
                "recording it cannot silently pool two quantizations, not "
                "because this table distinguished them._\n")

        dropped_axis_note = ""
        if dropped:
            dropped_axis_note = (
                f"\n_A **{CROSSCHECK_UNMATCHED_AXIS_MARK}** beside a build "
                f"mode means the axis was **not** matched on for that row. "
                f"{', '.join(dropped)} expose one build mode and their "
                f"ann-benchmarks modules declare none, while the ops harness "
                f"stamps `post` on every record from its own default "
                f"regardless of what the engine did \u2014 for MHNSW and VIDX "
                f"the graph is maintained on every INSERT whatever that field "
                f"says. Requiring the two to agree would empty this table for "
                f"those engines on a field neither side measures, so the axis "
                f"is dropped wherever the recall sweep is silent about it and "
                f"kept wherever it is not._\n")
        parts.append(
            f"\n_{len(matches)} matched point(s), {len(failed)} outside "
            f"tolerance._\n"
            f"\n_The two paths measure the same index with different code in "
            f"different containers over different query sets — ann-benchmarks "
            f"computes its own recall over **all** test vectors in-process, "
            f"while the ops calibration sweep scores the **first 1,000** over "
            f"a TCP connection. A gap is therefore expected to be small but "
            f"not zero, which is why the tolerance is "
            f"`max(0.01, 2 × the ops 95% CI half-width)` rather than a bare "
            f"equality: one recall point is allowed unconditionally, and a "
            f"noisier ops measurement is allowed proportionally more. "
            f"A **{CROSSCHECK_FAIL_MARK}** is the two paths disagreeing by "
            f"more than the ops measurement's own uncertainty can account "
            f"for, which is a result about this framework and not about the "
            f"engine._\n"
            f"\n_Matched on engine, dataset, resource pass, M, `ef_search`, "
            f"quantization and build mode. **Not** on storage engine: the ops "
            f"harness stamps `InnoDB` on every engine from its own default "
            f"while the ann modules report `heap`, `memory` and `wiredTiger`, "
            f"so requiring equality there would empty this table for three of "
            f"seven engines for a naming reason. The Storage column carries the "
            f"normalised value each row was paired on, so two units differing "
            f"only in storage engine are two rows here rather than one._\n"
            + dropped_axis_note + quantization_note
        )
    else:
        parts.append(
            "_No point was measured by both paths in this run._ That is a "
            "statement about the configuration, not a missing measurement — "
            "the table below says which case each engine and dataset is in.\n"
        )

    if coverage:
        parts.append("\n#### Which configurations could be cross-checked\n\n")
        parts.append(_md_table(
            ["Engine", "Dataset", "M measured by the recall sweep",
             "M measured by calibration", "M in both", "Matches", "Status"],
            [[_label(c.get("engine", "?")), c.get("dataset", "?"),
              ", ".join(str(m) for m in c["ann_m"]) or "—",
              ", ".join(str(m) for m in c["ops_m"]) or "—",
              ", ".join(str(m) for m in c["overlap_m"]) or "—",
              str(c["matches"]), c["status"]]
             for c in coverage],
        ))
        from report.generate import (COVERAGE_NO_M_OVERLAP,
                                     COVERAGE_NOTHING_MATCHED)
        if any(c["status"] == COVERAGE_NO_M_OVERLAP for c in coverage):
            parts.append(
                "\n_Rows reading **“no M is measured by both paths (by "
                "configuration)”** are a profile choice and not a fault: the "
                "recall sweep varies M because it is the only build knob "
                "MHNSW and VIDX expose, while the ops phase pins one M "
                "because every extra value reloads the whole corpus. To make "
                "this check non-empty, add one of the ops M values to "
                "`ann.m_values` for a single dataset, or run calibration at "
                "the ann M values in a cross-check profile._\n"
            )
        if any(c["status"] == COVERAGE_NOTHING_MATCHED for c in coverage):
            parts.append(
                "\n_Rows reading **“M overlaps but no point matched”** are a "
                "**fault**, not a configuration: both paths measured a shared "
                "M and no pair formed anyway, which means one of the other "
                "matched axes — resource pass, `ef_search`, quantization, "
                "build mode — disagrees between the two paths for a reason "
                "the records do not explain._\n"
            )

    if inversions:
        parts.append("\n#### Throughput order that differs between the two "
                     "paths\n\n")
        parts.append(_md_table(
            ["Dataset", "M", "ef_search", "Engines", "Faster in-process",
             "Faster over TCP", "QPS (ann)", "QPS (ops)"],
            [[row.get("dataset", "?"),
              str(row.get("m") if row.get("m") is not None else "—"),
              str(row.get("ef_search")
                  if row.get("ef_search") is not None else "—"),
              " vs ".join(_label(e) for e in row["engines"]),
              _label(row["faster_in_ann"]), _label(row["faster_in_ops"]),
              " / ".join(_fmt(q, 0) for q in row["qps_ann"]),
              " / ".join(_fmt(q, 0) for q in row["qps_ops"])]
             for row in inversions],
        ))
        parts.append(
            "\n_**Not a failure.** §5.2 item 5 flags these rather than "
            "failing on them: the recall path queries the engine in-process "
            "and the ops path goes over TCP with one client, so a fixed "
            "per-query overhead is added to every ops measurement. Two "
            "engines close together in-process can therefore swap places once "
            "that overhead is paid, and that is a real difference an operator "
            "experiences. It is listed because the alternative reading — that "
            "one of the two measurements is wrong — is also possible and "
            "nothing else in this report would surface it._\n"
        )
    elif matches:
        parts.append(
            "\n_No engine pair changed places between the two paths._\n")

    return "".join(parts)


def render_markdown(manifest: Dict[str, Any], summary: Dict[str, Any],
                    records: List[Dict[str, Any]],
                    chart_paths: Dict[str, Dict[str, str]], title: str) -> str:
    run_id = manifest.get("run_id", "?")
    config = manifest.get("config", {}) or {}
    profile = config.get("profile", {}) or {}

    parts: List[str] = [
        f"# {title}\n\n",
        f"**MariaDB vs AliSQL vs PostgreSQL + pgvector** — HNSW vector search.\n\n",
        f"- Run: `{run_id}`\n",
        f"- Started: {manifest.get('started_at', '?')} · Finished: {manifest.get('finished_at', '?')}\n",
        f"- Profile: **{profile.get('name', '?')}** — {profile.get('description', '')}\n",
        f"- Status: {manifest.get('status', '?')}\n",
        f"- Records: {len(records):,}\n\n",
        "---\n\n## 1. Environment\n\n",
        "Every number below is scoped to this environment. Vector-search results do "
        "not transfer across CPU generations, and both MariaDB MHNSW and AliSQL VIDX "
        "select distance kernels from CPU features at runtime.\n\n",
        _md_table(["Property", "Value"], _environment_rows(manifest)),
        "\n### Engines under test\n\n",
        _md_table(["Engine", "Tag", "Commit", "-march", "Build type"],
                  _engine_rows(manifest)),
        _provenance.engine_budget_section(manifest),
        "\n---\n\n## 2. Validity\n\n",
        _validity_section(manifest, summary, records),
        "\n---\n\n## 3. Known asymmetries\n\n",
        _known_asymmetries(summary),
        "\n---\n\n## 4. Recall vs throughput\n\n",
        _frontier_intro(summary),
        _provenance.section4_note(summary),
        _headline_tables(summary),
    ]

    for stem, paths in sorted(chart_paths.items()):
        if stem.split("-")[0] in ("pareto", "paretozoom", "qpsatrecall", "latency"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")
    # Under the QPS and latency charts, where the asymmetry bites.
    parts.append(_disclosures.structural_lines(records, _label))

    parts += [
        "\n---\n\n## 5. Index build cost\n\n",
        "ann-benchmarks does not measure this. **`Index build` says which "
        "operation the build time describes**, and the two are not the same "
        "quantity: pgvector builds its graph as one bulk operation after a "
        "plain load, while MHNSW and VIDX maintain theirs on every INSERT and "
        "have no separable build step at all \u2014 so for them the build cost "
        "IS the load cost. `Time to queryable index` is the column that "
        "compares across that difference: ingest plus build for a bulk engine, "
        "the single figure for an incremental one. Read the two together, and "
        "see the chart's first two panels for the same split drawn against "
        "M.\n\n",
        _build_table(summary),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.split("-")[0] in ("build", "storage"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")
    parts.append(_disclosures.memory_lines(records, _label))

    parts += [
        # Before the concurrency section and not after it, per PR 2 §2.3.2,
        # because it is the premise of the three sections that follow rather
        # than a note about them: concurrency, filtered search and churn are
        # all measured at the one ef_search this table reports, and a reader
        # who meets those numbers first has already read them as
        # equal-effort comparisons by the time the caveat arrives.
        "\n---\n\n## 6. Operating point\n\n",
        "Which `ef_search` each engine was measured at in the three sections "
        "below, and what recall that bought. A fixed `ef_search` is not one "
        "quantity across these engines — it is `mhnsw_ef_search`, VIDX's "
        "equivalent, `hnsw.ef_search`, MongoDB's `numCandidates` and Valkey's "
        "`EF_RUNTIME` — so a shared literal buys different amounts of search "
        "effort and recalls up to 0.18 apart. The sweep chooses the smallest "
        "grid `ef_search` that reaches the target recall and — where the grid "
        "has a next point — is confirmed by it; a target first met at the "
        "largest `ef_search` in the grid is used unconfirmed, because the "
        "grid ran out rather than because the check failed. That makes the "
        "sections below comparisons at equal answer quality instead of at "
        "equal knob setting. **Read the Status column before the "
        "throughput:** it names the engines for which that did not work, and "
        "in which direction.\n\n",
        _operating_point_table(summary),
    ]
    # The curve the choice was made from is in Appendix A, not here: §2.3.4
    # places it in an appendix, and the reason that is right is that this
    # section answers "is the comparison below fair" in one table, while the
    # curve answers "was the ef chosen well" — a different question, asked by a
    # smaller audience, and up to twelve points per engine wide.
    #
    # Conditional on the appendix actually being there. A cross-reference to a
    # section the report does not contain is the one kind of wrong this page can
    # be while every number on it is right, and it is reached by an ordinary
    # route: matplotlib is absent from the host this report may be generated
    # on, so the chart is skipped and every other section still renders.
    calibration = sorted(stem for stem in chart_paths
                         if stem.startswith("calibration-"))
    if calibration:
        parts.append(
            "\nThe curves these efs were chosen from are in **Appendix A**.\n")

    parts += [
        "\n---\n\n## 7. Concurrency scaling\n\n",
        "Where a database should differ from an ANN library. MariaDB caches the "
        "MHNSW graph per `TABLE_SHARE`; AliSQL keeps a shared cache plus a "
        "per-transaction cache; pgvector serves graph pages from `shared_buffers`. "
        "Scaling efficiency of 1.0 is linear scaling. **Spread** is the range "
        "across repeated measurement windows at that point; a dash means the "
        "point was measured once and carries no error bar.\n\n",
        (_concurrency_table(summary) + _peak_throughput_table(summary)
         + _limiter_table(summary)
         if summary.get("concurrency")
         else _not_measured("concurrency", profile, records)),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("concurrency-"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")
    parts.append(_disclosures.structural_lines(records, _label))

    parts += [
        "\n---\n\n## 8. Filtered (hybrid) search\n\n",
        "Vector search with a scalar predicate, scored against ground truth "
        "recomputed exactly over the qualifying rows. **Short results** counts "
        "queries that returned fewer than k rows — a real consequence of "
        "post-filtering, not a harness fault.\n\n",
        (_filtered_table(summary) if summary.get("filtered")
         else _not_measured("filtered", profile, records)),
        # In a mixed run the other engines fill the table and the decliner
        # would simply be absent from it; name it and say why.
        (_disclosures.declined_lines(records, "filtered", _label)
         if summary.get("filtered") else ""),
        _filter_strategy_table(summary),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.startswith("filtered-"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        "\n---\n\n## 9. Churn\n\n",
        "Recall and throughput after deleting and re-inserting a fraction of rows. "
        "HNSW graphs degrade under deletion; how much differs by implementation.\n\n",
        (_churn_table(summary) if summary.get("churn")
         else _not_measured("churn", profile, records)),
        _disclosures.index_sync_lines(records, _label),
    ]
    for stem, paths in sorted(chart_paths.items()):
        if stem.split("-")[0] in ("churn", "churnimpact", "passcompare") or stem.startswith("memory-timeline"):
            parts.append(f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")

    parts += [
        # PR 5 §5.2 item 3. Its own numbered section rather than a note under
        # §2 Validity, because it is not a list of faults: it reports every
        # point the two measurement paths both measured, passing ones
        # included, and an empty list with the reason it is empty. A reader
        # who finds only the failures here would read the absence of failures
        # as the absence of the check.
        #
        # After §9 and before the reproduction instructions, so the two
        # sections it compares -- §4's recall frontier and §6's operating
        # point -- have both been read by the time it arrives. §6's own prose
        # already asks the reader to do this comparison by hand off Appendix
        # A's curves; this is that comparison done by the generator.
        "\n---\n\n## 10. Consistency checks\n\n",
        "The recall/throughput sweep in §4 and the calibration sweep in §6 "
        "measure the same index with different code: ann-benchmarks in-process "
        "against the whole test set, and the ops harness over TCP against the "
        "first 1,000 queries. Where a configuration was measured by both, the "
        "two recalls should agree to within the ops measurement's own "
        "uncertainty. Where they do not, the disagreement is a fact about this "
        "framework and belongs in the report.\n\n",
        _consistency_section(summary),
    ]

    parts += [
        "\n---\n\n## 11. Reproducing this run\n\n",
        "```bash\n"
        f"./run-benchmark.sh build --march {(config.get('resolved_resources') or {}).get('march', 'x86-64-v3')}\n"
        f"./run-benchmark.sh fetch --datasets {','.join(summary.get('datasets', []) or ['glove-100-angular'])}\n"
        f"./run-benchmark.sh run --profile {profile.get('name', 'quick')} "
        f"--resource-pass {reproduction_pass_flag(manifest)}\n"
        "```\n\n",
        "The raw merged records are in `records.jsonl` next to this file, and the "
        "full provenance — image ids, source commits, resolved limits — is in "
        "`../run-manifest.json`.\n\n",
        "See `docs/05-methodology.md` for what is measured and how, and "
        "`docs/03-running-manually.md` to reproduce any single measurement by hand "
        "without the framework.\n",
    ]

    # PR 2 §2.3.4's appendix. Emitted only when the sweep ran: an appendix
    # heading over a "not measured" paragraph is the empty-chart problem with
    # extra steps, and `ef_mode: fixed` already says so in §6. `calibration` is
    # the same list the §6 cross-reference above is conditioned on, so the
    # pointer and its target cannot disagree.
    if calibration:
        parts.append(
            "\n---\n\n## Appendix A. Calibration curves\n\n"
            "Recall against requested `ef_search` from the sweep in §6, one "
            "line per engine, with the chosen point ringed. A choice sitting "
            "on a plateau is robust; one on a steep slope means a small change "
            "in the error model moves the ef, and the engines whose curves "
            "cross in this region are the ones whose ranking in §7 depends on "
            "the target.\n\n"
            "**Use it as a cross-check against §4.** Carry an engine's recall "
            "at some `ef_search` off this chart and find the same engine at the "
            "same `ef_search` and build on the §4 recall/throughput chart. The two are "
            "measured by different code in different containers over different "
            "query counts — ann-benchmarks computes its own recall, this sweep "
            "computes ours — so they are not plotted together and are not "
            "expected to match to the fourth decimal. A **large** disagreement "
            "is the finding: it means one of the two measurement paths is "
            "wrong about that engine, and nothing else in this report would "
            "show it.\n")
        for stem in calibration:
            paths = chart_paths[stem]
            parts.append(
                f"\n![{stem}](charts/{os.path.basename(paths['svg'])})\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# §3's two generated rows
#
# These helpers belong beside `_known_asymmetries` and are down here instead
# for a reason that NO LONGER HOLDS, which is why this paragraph says so
# rather than being deleted with the constraint.
#
# WHAT IT WAS. `docs/10-rollout.md` cited three locations in this module by
# LINE NUMBER, two of them below `_known_asymmetries` -- the "grids meet"
# sentence in `_consistency_section` and the section-10 heading in
# `render_markdown` -- and `tests/test_e3_rollout_doc.py` resolved each
# against that exact line. Inserting anything above them shifted both and put
# `master` red; `CLAUDE.md` records that happening twice in two days
# (#41, #43). docs/10 was not that change's to edit, so the call site above
# was rewritten within its own line count and the bodies went here. Nor did
# extracting them into a module help: removing ~100 lines near the call site
# shifts the later citations UP and fails the same guard an insertion does.
#
# WHAT CHANGED. #63 converted those citations to SYMBOLS. `CITED_IN_REPO` is
# now keyed by `(path, symbol)` pairs -- `("report/render.py",
# "_consistency_section")` and the two beside it -- and resolves them by
# finding the symbol, so a line number is no longer load-bearing anywhere in
# this file and these bodies can move freely.
#
# THE MOTION IS PENDING AND TRAVELS ALONE. ~240 lines sit ~2,290 lines from
# their only call site; moving them back beside `_known_asymmetries` is now
# a legal change and is deliberately not made here. A relocation of that size
# is the diff shape that hides a substantive edit inside it, and doing it in
# a change with behaviour in it means re-measuring the suite to achieve
# nothing observable. Nothing enforces the motion: the citation ratchet in
# `tests/test_e16_citation_ratchet.py` counts citations and cannot see a
# block that wants moving, so this paragraph is the record of the debt and
# should go with the block when it moves.
# ---------------------------------------------------------------------------

#: How this table names an engine in PROSE, which is not how a chart legend
#: names it. Every other row here is hand-written English -- "pgvector
#: bulk-builds", "AliSQL VIDX is InnoDB-only", "Valkey holds the whole
#: dataset in memory" -- so a row generated from the records has to be able
#: to say the same words. `_label` is the legend, and "`ef_construction` is
#: exposed only by PostgreSQL (pgvector)" reads as a label dropped into a
#: sentence; the sentence this replaces said "pgvector", and
#: `tests/test_config.py` pins that spelling.
#:
#: NOT A GATE, which is the thing to check before adding a map of engine
#: names to this repo: `_prose_name` falls back to `_label`, so an engine
#: added after this was written names itself from the manifest instead of
#: printing its config key, and nothing here has to be kept in step with
#: `config/engines/*.yml`. `KNOWN_ENGINES` stays whatever that directory
#: holds.
_PROSE_NAME = {
    "mariadb": "MariaDB MHNSW",
    "mariadb123": "MariaDB 12.3 MHNSW",
    "alisql": "AliSQL VIDX",
    "pgvector": "pgvector",
    "mongodb": "Percona Search",
    "valkey": "Valkey",
    "percona": "Percona Server",
    "myvector": "MyVector",
    "villagesql": "VillageSQL",
    "villagesql84": "VillageSQL on MySQL 8.4",
}

#: Which engines expose `ef_construction` when no record says either way.
#:
#: Absence is not a denial. `extra.ef_construction_tunable` arrived with
#: `build.py:_driver_capabilities`, and the engines that expose the knob
#: exposed it before that key was ever written, so reading silence as "no"
#: would withdraw a true claim from every run archived before it -- and from
#: every synthetic summary a test builds without the key. The shape and the
#: precedent are `generate.FILTER_COLUMN_WHEN_ABSENT`: a statement of fact
#: about the records that predate the key, not a default.
#:
#: A record that DOES carry the key always wins, so this set cannot overrule
#: a driver: it is consulted only for an engine whose every build record is
#: silent.
#:
#: MEASURED, because both halves of that argument are claims about archived
#: data. Of the `index_build` records in the reachable archives, 132 carry no
#: `ef_construction_tunable`: pgvector 38, mariadb 44, alisql 34,
#: mariadb123 12, mongodb 4. The pgvector 38 are why absence must not be read
#: as a denial -- a legacy pgvector-only directory would otherwise regenerate
#: with the knob withdrawn from the one engine that had it. The other 94 are
#: why absence must not be read as an affirmation either: read that way, a
#: legacy mariadb-only directory regenerates as "exposed only by MariaDB
#: MHNSW", which is the original defect with its sign flipped. Both
#: directions are pinned in
#: `tests/test_e19_asymmetry_rows_name_the_engines_that_ran.py`, and so is
#: this set's equality with the engine configs, because a set that may be
#: widened by a one-word edit is a literal again.
EF_CONSTRUCTION_TUNABLE_WHEN_UNRECORDED = frozenset(
    {"pgvector", "valkey", "percona", "myvector", "villagesql",
     "villagesql84"})


def _prose_name(engine: str) -> str:
    return _PROSE_NAME.get(engine) or _label(engine)


def _prose_list(engines: List[str]) -> str:
    names = [_prose_name(e) for e in engines]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _capability_states(summary: Optional[Dict[str, Any]],
                       key: str) -> Dict[str, Any]:
    """Per engine, the values its `index_build` records state for `key`.

    An engine missing from the result said nothing on any record, which is a
    third state and not a `False` -- the distinction
    `generate.build_kind`'s `BUILD_KIND_UNRECORDED` exists for, and the one
    that decides whether the fallback above is consulted.

    ONLY A REAL BOOLEAN COUNTS AS AN ANSWER, and the discipline is the point
    rather than defensiveness. This read `bool(extra[key])` on whatever was
    there, which turns the three states back into two in both directions: a
    hand-edited or YAML-round-tripped `"false"` is truthy, so a report would
    have said "exposed only by MariaDB MHNSW" off a string; and a recorded
    `null` -- which this repo uses deliberately to mean "documented, not
    observed", as `config/engines/valkey.yml` does for
    `hybrid_filter_planner` with a comment about exactly this hazard --
    collapsed to a denial WITHOUT consulting the fallback, which is the one
    thing the three-state rule exists to prevent.

    So a value that is not `True` or `False` is not an answer: the record is
    silent for that key and the engine falls to the fallback, the same as a
    record that never carried it. `harness/metrics/records.py` guards the
    same shape the same way in `ops_versions_measure_the_same_thing`, with a
    comment about hand-edited manifests. All 685 values in the reachable
    archives are real booleans, so nothing in the published numbers moves;
    this is about what the next hand-edited file does.
    """
    states: Dict[str, Any] = {}
    for record in (summary or {}).get("build") or []:
        value = (record.get("extra") or {}).get(key)
        if isinstance(value, bool):
            states.setdefault(record.get("engine"), set()).add(value)
    return states


def _ef_construction_engines(summary: Optional[Dict[str, Any]]) -> List[str]:
    """The engines in this run whose records say the build knob is theirs."""
    states = _capability_states(summary, "ef_construction_tunable")
    return [engine
            for engine in sorted(set((summary or {}).get("engines") or []))
            if (True in states[engine] if engine in states
                else engine in EF_CONSTRUCTION_TUNABLE_WHEN_UNRECORDED)]


def _ef_construction_row(tunable: List[str],
                         engines: List[str]) -> List[str]:
    """§3's `ef_construction` row, in three arms rather than two.

    The third arm is the one a literal set could not reach: a run in which no
    engine exposes the knob at all. The old code's `else` branch covered both
    "one engine has it" and "none of them does" and said "only by pgvector"
    for both, so a MariaDB-and-AliSQL report named an engine that had not run
    and a Percona report denied the engine that had.

    NO ROW HERE SAYS WHAT THE SWEEP VARIES INSTEAD, and that absence is
    deliberate. The obvious sentence -- "so the sweep varies `M` alone" --
    is false for one of the seven engines: `orchestrator/ann_pass.py` gives
    Percona Search `m_values[:1]` under its own comment, "M is carried, not
    applied: mongot exposes no graph degree", where every other engine gets
    the whole grid. `--profile full --engines mongodb` is an ordinary
    invocation, and in that report this row would have sat three rows above
    "M is pinned at one value for every other engine and cannot be set here
    at all" and contradicted it. Writing a second claim about a second knob
    to explain the first is how this row went wrong in the first place.
    """
    others = [e for e in engines if e not in tunable]
    if not tunable:
        return [
            "`ef_construction` is exposed by none of the engines in this run",
            "Build quality cannot be tuned here at all, so the normalized "
            "pass has no knob to pin."
            + (f" {_prose_list(others)} "
               + ("exposes" if len(others) == 1 else "expose")
               + " no equivalent." if others else ""),
        ]
    tuned = _prose_list(tunable)
    consequence = (
        f"Build quality can be tuned for {tuned}, "
        + (f"and not for {_prose_list(others)}, which "
           + ("has" if len(others) == 1 else "have")
           + " no equivalent. " if others else
           "an axis the engines without the knob do not have. ")
        + "It is pinned to the default in the normalized pass, so no engine "
          "is given an axis the others lack, and swept only in the tuned "
          "pass.")
    if "percona" in tunable:
        # `CLAUDE.md`'s invariants bullet, in the row that is about the knob:
        # the build-time knob is half of what makes this engine the
        # interesting one, and a reader of a report whose subject is
        # `ef_construction` has to be told the query-time knob is there too.
        consequence += (
            " Percona Server is the only engine in this study with both a "
            "build-time and a query-time effort knob, so for it the sweep is "
            "a second axis rather than the only one.")
    return [
        ("`ef_construction` is exposed only by " if len(tunable) == 1
         else "`ef_construction` is exposed by ") + tuned,
        consequence,
    ]


def _build_mode_engines(summary: Optional[Dict[str, Any]]
                        ) -> Tuple[List[str], List[str], List[str]]:
    """Which engines bulk-built, which built incrementally, which did both.

    Four kinds in and three lists out, and the arithmetic is the point.

    An ASYNCHRONOUS build is excluded rather than filed as a bulk one.
    mongot's `separable_build` is True and its index is built by another
    process while the writes are still arriving, so calling that a bulk build
    would contradict the "builds its index asynchronously" row the mongodb
    block adds a few rows down. `_index_build_kind` splits the same four
    kinds for §5's table and this follows it.

    An UNRECORDED build -- no `separable_build` key at all -- lands in no
    list. Nothing in such a record says which operation `build_wall_s` timed,
    and this row is a claim about an operation.

    That third case is a GUARD rather than a described population, and the
    difference is worth stating because the fallback above is the other way
    round. Counted over every archived run reachable here, 817 `index_build`
    records carry `separable_build` and none is missing it, so no archive
    exercises this branch; 132 of those same records are missing
    `ef_construction_tunable` -- 38 of them pgvector's -- which is why
    `EF_CONSTRUCTION_TUNABLE_WHEN_UNRECORDED` is backed by data and this is
    not.
    """
    from report.generate import (BUILD_KIND_BULK, BUILD_KIND_INCREMENTAL,
                                 build_kind)
    kinds: Dict[str, Any] = {}
    for record in (summary or {}).get("build") or []:
        extra = record.get("extra") or {}
        if (extra.get("async_index_build")
                or record.get("build_mode") == "async"):
            continue
        kinds.setdefault(record.get("engine"), set()).add(build_kind(record))
    bulk = {e for e, seen in kinds.items() if BUILD_KIND_BULK in seen}
    incremental = {e for e, seen in kinds.items()
                   if BUILD_KIND_INCREMENTAL in seen}
    return (sorted(bulk - incremental), sorted(incremental - bulk),
            sorted(bulk & incremental))


def _build_mode_rows(summary: Optional[Dict[str, Any]]) -> List[List[str]]:
    """§3's bulk-versus-incremental rows, naming what this run observed.

    Two rows at most. An engine swept over both modes is not a side of the
    bulk/incremental asymmetry -- it is the asymmetry held inside one engine
    -- so it gets its own row rather than being named on both sides of the
    first one.

    A run that observed neither gets no row. The unconditional sentence this
    replaces is why: in a Percona-only report it described MHNSW, VIDX and
    pgvector, none of which had run.
    """
    bulk, incremental, either = _build_mode_engines(summary)
    rows: List[List[str]] = []
    clauses = []
    if bulk:
        clauses.append(
            f"{_prose_list(bulk)} "
            + ("bulk-builds" if len(bulk) == 1 else "bulk-build")
            + " after load")
    if incremental:
        clauses.append(
            f"{_prose_list(incremental)} "
            + ("builds" if len(incremental) == 1 else "build")
            + " incrementally on every INSERT")
    if clauses:
        # The CONSEQUENCE is conditional too, and leaving it unconditional
        # while deriving the subject was half a fix. Master's sentence says
        # "reported separately for both modes", which on
        # `percona-20261021-134142` -- one engine, one mode, incremental --
        # described a second column the report does not contain, directly
        # beside a correctly derived subject naming one mode. A row whose two
        # cells disagree about how many modes ran is worse than the
        # unconditional row it replaced, because the derivation makes it look
        # checked.
        rows.append(["; ".join(clauses),
                     "Build cost is reported separately for both modes. A "
                     "single 'build time' comparison would compare different "
                     "operations."
                     if bulk and incremental else
                     "Build cost is reported as the operation that ran. A "
                     "single 'build time' comparison against the other mode "
                     "-- in this run or in another -- would compare different "
                     "operations."])
    if either:
        consequence = (
            "Both modes were measured for it, so that pair is a comparison "
            "within one implementation rather than against another one.")
        if "percona" in either:
            # The half of this that a single ratio hides, and it is not a
            # detail: `docs/09-adding-percona-server.md` §2 quotes the
            # vendor's own `index_build.test` -- "First ADD, explicit
            # INPLACE: rebuilds the table with the column" -- so the bulk
            # path pays for a base-table rebuild that pgvector's bulk path
            # does not. Reporting "the bulk build is N times cheaper" with
            # that inside the number compares two operations and calls the
            # difference a build cost.
            consequence += (
                " For Percona Server the two are not even the same "
                "operation: its first *ADD VECTOR KEY* rebuilds the base "
                "table where pgvector's *CREATE INDEX* does not, so a single "
                "'the bulk build is faster' figure hides a table rebuild "
                "inside one engine's number.")
        rows.append([
            f"{_prose_list(either)} {'does' if len(either) == 1 else 'do'} "
            "either, so `build_mode` is a comparison within one engine",
            consequence,
        ])
    ann_engines = (summary or {}).get("ann_engines") or []
    if "percona" in ann_engines and ("percona" in bulk or "percona" in either):
        # **THE TWO PATHS BUILD PERCONA DIFFERENTLY BY DEFAULT, and the rows
        # above cannot say so because they read only ops `index_build`
        # records.** Since `capabilities.bulk_vector_index_build` became true
        # at the current pin, an ops unit that names no mode builds `post`;
        # the ann overlay module has no `build_mode` and declares the index in
        # CREATE TABLE, so every recall curve is an incremental build. At
        # 60,000 fashion-mnist rows the two gave the same recall (0.9714 vs
        # 0.9709 at ef_search 10), which is why this is a disclosure and not
        # a refusal -- but a reader joining §4's curve to §5's build cost is
        # joining two different builds, and has to be told.
        #
        # Two conditions, both found in review. Percona's ops units swept
        # over BOTH modes land in `either`, not `bulk`, and that is exactly
        # the run in which a reader compares modes against the curve, so
        # `either` counts. And an ops-only report has no curve to be an
        # incremental build of, so the row needs Percona's recall records.
        ops_side = ("some of its ops units bulk-built"
                    if "percona" in either else "its ops units bulk-built")
        rows.append([
            f"Percona Server: {ops_side}; its recall curves are an "
            "incremental build",
            "The ann phase declares the index in CREATE TABLE and has no "
            "build mode, so the recall/QPS curves come from an index grown "
            "row by row, while the ops figures for a bulk-built unit (build "
            "cost, concurrency, filtered, churn) come from the bulk build "
            "the ops pass ran."])
    return rows


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5a6270; --border: #e2e5ea;
  --accent: #1f77b4; --card: #f7f8fa; --warn-bg: #fff6e5; --warn-br: #e6a23c;
  --code-bg: #f2f4f7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c; --fg: #e6e8eb; --muted: #9aa4b2; --border: #2a2f38;
    --accent: #5aa9e6; --card: #1b1f26; --warn-bg: #2e2618; --warn-br: #c98a2b;
    --code-bg: #1f242c;
  }
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5a6270; --border: #e2e5ea;
  --accent: #1f77b4; --card: #f7f8fa; --warn-bg: #fff6e5; --warn-br: #e6a23c;
  --code-bg: #f2f4f7;
}
:root[data-theme="dark"] {
  --bg: #14171c; --fg: #e6e8eb; --muted: #9aa4b2; --border: #2a2f38;
  --accent: #5aa9e6; --card: #1b1f26; --warn-bg: #2e2618; --warn-br: #c98a2b;
  --code-bg: #1f242c;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 5rem; background: var(--bg); color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 .35em; letter-spacing: -.01em; }
h2 { font-size: 1.35rem; margin: 2.4em 0 .6em; padding-bottom: .3em;
     border-bottom: 1px solid var(--border); }
h3 { font-size: 1.08rem; margin: 1.8em 0 .5em; }
h4 { font-size: .98rem; margin: 1.4em 0 .4em; color: var(--muted);
     text-transform: uppercase; letter-spacing: .04em; }
p, li { color: var(--fg); }
.sub { color: var(--muted); font-size: .95rem; margin-top: 0; }
a { color: var(--accent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: .875em; }
code { background: var(--code-bg); padding: .12em .38em; border-radius: 4px; }
pre { background: var(--code-bg); padding: 1rem; border-radius: 8px;
      overflow-x: auto; border: 1px solid var(--border); }
pre code { background: none; padding: 0; }
.tablewrap { overflow-x: auto; margin: 1rem 0; border: 1px solid var(--border);
             border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { padding: .55rem .8rem; text-align: left; border-bottom: 1px solid var(--border);
         white-space: nowrap; }
th { background: var(--card); font-weight: 600; position: sticky; top: 0; }
tr:last-child td { border-bottom: none; }
td:first-child, th:first-child { white-space: normal; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: .8rem; margin: 1.2rem 0; }
.meta div { background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; padding: .7rem .9rem; }
.meta dt { color: var(--muted); font-size: .75rem; text-transform: uppercase;
           letter-spacing: .05em; margin: 0 0 .2rem; }
.meta dd { margin: 0; font-size: .95rem; font-weight: 600; word-break: break-word; }
.warn { background: var(--warn-bg); border-left: 3px solid var(--warn-br);
        padding: .85rem 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
.warn p:first-child { margin-top: 0; } .warn p:last-child { margin-bottom: 0; }
figure { margin: 1.4rem 0; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--muted); font-size: .85rem; margin-top: .4rem; }
.note { color: var(--muted); font-size: .9rem; font-style: italic; }

/* ---------------------------------------------------------------------------
   Print / Save as PDF.

   The browser's own PDF export is the whole mechanism -- no extra dependency,
   and it works on the standalone file as well as in the app. What it needs is
   for the page to stop being a screen: one light palette whatever the reader's
   theme, and page breaks that do not fall through a chart or a table.
   --------------------------------------------------------------------------- */
@media print {
  /* A dark theme prints as a wall of ink, and the reader's theme is not a
     property of the document being archived. */
  :root, :root[data-theme="dark"], :root[data-theme="light"] {
    --bg: #ffffff; --fg: #111111; --muted: #444444; --border: #bbbbbb;
    --accent: #14507d; --card: #f6f6f6; --warn-bg: #fff8e8; --warn-br: #b8860b;
    --code-bg: #f2f2f2;
  }
  body { background: #fff; color: #111; font-size: 10.5pt; }
  .wrap { max-width: none; margin: 0; padding: 0; }

  /* Charts and tables are the content; splitting one across a page break makes
     it unreadable rather than merely untidy. */
  figure, svg, img, table, .tablewrap, .warn, .note { break-inside: avoid; page-break-inside: avoid; }
  h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
  h2 { break-before: page; page-break-before: page; }
  h1 + p, h2:first-of-type { break-before: auto; page-break-before: auto; }

  /* Nothing scrolls on paper, so a wide table has to fit instead. Without a
     fixed layout it keeps its natural width and simply runs off the right of
     the page, taking the last column with it. */
  .tablewrap { overflow: visible !important; }
  table { width: 100%; table-layout: fixed; font-size: 8.5pt; }
  /* On screen cells never wrap and the wrapper scrolls. On paper there is
     nowhere to scroll, so nowrap is what pushed the last column off the page
     -- table-layout alone cannot rescue a cell that refuses to break. */
  th, td { white-space: normal; overflow-wrap: anywhere; word-break: break-word;
           padding: .35rem .5rem; }
  th { position: static; }
  svg { max-width: 100% !important; height: auto !important; }

  a { color: inherit; text-decoration: none; }
  /* The environment and validity sections carry the caveats; keep their
     emphasis rather than flattening them into the body text. */
  .warn { border-left: 3px solid #b8860b; background: #fff8e8; }
}

@page { margin: 16mm 14mm; }
"""


def _svg_inline(path: str) -> str:
    """Inline an SVG so the page has no external requests at all."""
    try:
        with open(path, encoding="utf-8") as fh:
            svg = fh.read()
    except OSError:
        return ""
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else ""


def _md_tables_to_html(markdown_fragment: str) -> str:
    """Convert the Markdown tables built above into HTML.

    A tiny purpose-built converter rather than a Markdown dependency: the only
    constructs used in these fragments are tables, list items, headings and
    emphasis, and the report should not gain a dependency for that.
    """
    lines = markdown_fragment.split("\n")
    out: List[str] = []
    table: List[List[str]] = []

    def flush_table() -> None:
        if not table:
            return
        header, *body = table
        out.append('<div class="tablewrap"><table><thead><tr>')
        # .extend, not `out +=`: augmented assignment would rebind `out` as a
        # local inside this closure and shadow the enclosing list.
        out.extend(f"<th>{_html.escape(c)}</th>" for c in header)
        out.append("</tr></thead><tbody>")
        for row in body:
            out.append("<tr>" + "".join(
                f"<td>{_inline_md(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table></div>")
        table.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue  # separator row
            table.append(cells)
            continue
        flush_table()
        if not stripped:
            continue
        if stripped.startswith("#### "):
            out.append(f"<h4>{_inline_md(stripped[5:])}</h4>")
        elif stripped.startswith("### "):
            out.append(f"<h3>{_inline_md(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            out.append(f"<li>{_inline_md(stripped[2:])}</li>")
        elif stripped.startswith("_") and stripped.endswith("_"):
            out.append(f'<p class="note">{_inline_md(stripped.strip("_"))}</p>')
        else:
            out.append(f"<p>{_inline_md(stripped)}</p>")
    flush_table()

    joined = "\n".join(out)
    # Wrap runs of <li> in a <ul>.
    joined = joined.replace("<li>", "<ul><li>", 1) if "<li>" in joined else joined
    if "<li>" in joined:
        idx = joined.rfind("</li>")
        joined = joined[:idx + 5] + "</ul>" + joined[idx + 5:]
    return joined


def _inline_md(text: str) -> str:
    escaped = _html.escape(text)
    # The one tag a Markdown table cell may carry, and the only way to put a
    # second line in one: §4 prints each figure's build UNDER the number.
    # GitHub's renderer honours `<br>` in a cell; this converter escaped it.
    escaped = escaped.replace("&lt;br&gt;", "<br>")
    # Order matters: bold before italic, code before both.
    while "`" in escaped:
        first = escaped.find("`")
        second = escaped.find("`", first + 1)
        if second < 0:
            break
        escaped = (escaped[:first] + "<code>" + escaped[first + 1:second]
                   + "</code>" + escaped[second + 1:])
    while "**" in escaped:
        first = escaped.find("**")
        second = escaped.find("**", first + 2)
        if second < 0:
            break
        escaped = (escaped[:first] + "<strong>" + escaped[first + 2:second]
                   + "</strong>" + escaped[second + 2:])
    # `[text](href)`, which the limiter table's profile column uses. After
    # escaping, so the href is already attribute-safe.
    escaped = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>',
                     escaped)
    return escaped


def _pass_note(summary: Dict[str, Any]) -> str:
    """Explain an absent normalized-vs-tuned comparison.

    The chart hangs off section 9 — section 8 until "Operating point" was
    inserted ahead of it — so when it is skipped that section simply loses a
    figure with nothing said about it.
    """
    passes = sorted({p for p in summary.get("passes", []) if p})
    if len(passes) >= 2:
        return ""
    only = passes[0] if passes else "one"
    return (f"<p><em>The normalized-vs-tuned comparison is not shown: this run "
            f"measured the <code>{_html.escape(only)}</code> pass only. Run with "
            f"<code>--resource-pass both</code> to populate it.</em></p>")


def render_html(manifest: Dict[str, Any], summary: Dict[str, Any],
                records: List[Dict[str, Any]],
                chart_paths: Dict[str, Dict[str, str]], title: str) -> str:
    run_id = manifest.get("run_id", "?")
    host = manifest.get("host", {}) or {}
    cpu = host.get("cpu", {}) or {}
    config = manifest.get("config", {}) or {}
    profile = config.get("profile", {}) or {}
    resolved = config.get("resolved_resources", {}) or {}

    def figures(prefix: str) -> str:
        out = []
        for stem, paths in sorted(chart_paths.items()):
            if stem.startswith(prefix):
                svg = _svg_inline(paths["svg"])
                if svg:
                    out.append(f"<figure>{svg}<figcaption>{_html.escape(stem)}"
                               f"</figcaption></figure>")
        return "\n".join(out)

    # The headline cards are the first thing a reader of the page sees, and
    # they carried the same single-pass claim §1's table did: one cpuset and
    # one memory limit, taken from whichever pass `set_config` saw last. On
    # `full-20261015-215006` that put `0-75` and 112.5 GiB at the top of a
    # page whose tables held 8-CPU, 16 GB rows. With two passes each card
    # names both, qualified, rather than picking one.
    per_pass = _pass_resources(manifest)
    if len(per_pass) > 1:
        pass_card = ", ".join(name for name, _limits in per_pass)
        cpuset_card = ", ".join(f"{_or_unknown(limits.get('server_cpuset'))} "
                                f"({name})" for name, limits in per_pass)
        memory_card = ", ".join(
            f"{_fmt_bytes(limits.get('server_memory_bytes'))} ({name})"
            for name, limits in per_pass)
    else:
        pass_card = config.get("resource_pass", "?")
        cpuset_card = resolved.get("server_cpuset", "?")
        memory_card = _fmt_bytes(resolved.get("server_memory_bytes"))
    cards = [
        ("Run", run_id),
        ("Profile", profile.get("name", "?")),
        ("Resource pass", pass_card),
        ("CPU", cpu.get("model", "?")),
        ("AVX-512", "yes" if cpu.get("has_avx512") else "no"),
        ("Server cpuset", cpuset_card),
        ("Server memory", memory_card),
        ("Records", f"{len(records):,}"),
    ]
    cards_html = "".join(
        f"<div><dt>{_html.escape(str(k))}</dt><dd>{_html.escape(str(v))}</dd></div>"
        for k, v in cards
    )

    warnings = manifest.get("warnings", []) or []
    warn_html = ""
    if warnings:
        items = "".join(f"<p>⚠ {_html.escape(w)}</p>" for w in warnings)
        warn_html = f'<div class="warn">{items}</div>'

    sections = [
        ("1. Environment",
         _md_tables_to_html(_md_table(["Property", "Value"], _environment_rows(manifest)))
         + "<h3>Engines under test</h3>"
         + _md_tables_to_html(_md_table(
             ["Engine", "Tag", "Commit", "-march", "Build type"], _engine_rows(manifest)))
         + _md_tables_to_html(_provenance.engine_budget_section(manifest)),
         ""),
        ("2. Validity",
         _md_tables_to_html(_validity_section(manifest, summary, records)), ""),
        ("3. Known asymmetries",
         _md_tables_to_html(_known_asymmetries(summary)), ""),
        ("4. Recall vs throughput",
         _md_tables_to_html(_provenance.section4_note(summary)
                            + _headline_tables(summary)),
         figures("pareto-") + figures("paretozoom-") + figures("qpsatrecall-") + figures("latency-")
         + _md_tables_to_html(_disclosures.structural_lines(records, _label))),
        ("5. Index build cost",
         _md_tables_to_html(_build_table(summary)), figures("build-") + figures("storage-")
         + _md_tables_to_html(_disclosures.memory_lines(records, _label))),
        # Before concurrency, and carrying no figure: the calibration chart is
        # Appendix A below. Numbered and worded to match the markdown, for the
        # reason the comment under this list gives -- this page is the one a
        # reader actually opens, and a section that exists in one renderer and
        # not the other is how the HTML came to say "No data." for workloads
        # the markdown already explained.
        ("6. Operating point",
         _md_tables_to_html(_operating_point_table(summary)), ""),
        # These three must use the same not-measured wording as the markdown.
        # The HTML is built from its own section list, so fixing only the
        # markdown left the page a reader actually opens still saying
        # "No data." for a workload the profile never ran.
        ("7. Concurrency scaling",
         _md_tables_to_html(_concurrency_table(summary)
                            + _peak_throughput_table(summary)
                            + _limiter_table(summary)
                            if summary.get("concurrency")
                            else _not_measured("concurrency", profile, records)),
         figures("concurrency-")
         + _md_tables_to_html(_disclosures.structural_lines(records, _label))),
        ("8. Filtered (hybrid) search",
         _md_tables_to_html(
             (_filtered_table(summary)
              + _disclosures.declined_lines(records, "filtered", _label))
             if summary.get("filtered")
             else _not_measured("filtered", profile, records)),
         figures("filtered-")),
        ("9. Churn",
         _md_tables_to_html((_churn_table(summary) if summary.get("churn")
                             else _not_measured("churn", profile, records))
                            + _disclosures.index_sync_lines(records, _label))
         + _pass_note(summary),
         figures("churn-") + figures("churnimpact-") + figures("passcompare-") + figures("memory-timeline")),
        # PR 5 §5.2 item 3, in the renderer a reader actually opens. The
        # comment above this list is about exactly this failure: the HTML is
        # built from its own section list, so a section added to the markdown
        # alone is a section the page does not have. No figure -- the check is
        # a table of two recalls per row, and a chart of 1,000 matched pairs
        # would say less than the pass/fail column does.
        ("10. Consistency checks",
         _md_tables_to_html(_consistency_section(summary)), ""),
    ]
    body = "".join(
        f"<h2>{_html.escape(name)}</h2>{content}{figs}"
        for name, content, figs in sections
    )

    # Last on the page, after "Reproducing this run", which is where the
    # markdown puts it too. Emitted only when the sweep ran: an appendix
    # heading over nothing is the empty-chart problem with extra steps, and
    # §6 already says what a fixed-ef run means.
    calibration_figures = figures("calibration-")
    appendix = ""
    if calibration_figures:
        appendix = (
            "<h2>Appendix A. Calibration curves</h2>"
            "<p class=\"note\">Recall against requested <code>ef_search</code> "
            "from the sweep in §6, one line per engine, with the chosen point "
            "ringed. Cross-check it against §4: the same engine at the same "
            "<code>ef_search</code> should sit at a similar recall there. The "
            "two are measured by different code in different containers over "
            "different query counts, so they are not plotted together and are "
            "not expected to agree to the fourth decimal — a large "
            "disagreement means one of the two measurement paths is wrong "
            "about that engine, and nothing else in this report would show "
            "it.</p>" + calibration_figures)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)} — {_html.escape(run_id)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{_html.escape(title)}</h1>
<p class="sub">MariaDB vs AliSQL vs PostgreSQL + pgvector — HNSW vector search ·
{_html.escape(str(manifest.get('started_at', '')))} → {_html.escape(str(manifest.get('finished_at', '')))}</p>
<div class="meta">{cards_html}</div>
{warn_html}
{body}
<h2>11. Reproducing this run</h2>
<pre><code>./run-benchmark.sh build
./run-benchmark.sh fetch --datasets {_html.escape(','.join(summary.get('datasets', []) or ['glove-100-angular']))}
./run-benchmark.sh run --profile {_html.escape(str(profile.get('name', 'quick')))} --resource-pass {_html.escape(reproduction_pass_flag(manifest))}</code></pre>
<p class="note">Raw merged records are in <code>records.jsonl</code>; full provenance
is in <code>run-manifest.json</code>. See <code>docs/05-methodology.md</code> for
what is measured, and <code>docs/03-running-manually.md</code> to reproduce any
single measurement by hand without this framework.</p>
{appendix}
</div>
</body>
</html>
"""
