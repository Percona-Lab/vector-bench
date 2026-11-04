"""Guards for the rollout runbook in docs/10-rollout.md.

This file holds the checkable half of that document. It exists because a runbook
is read once, under time press, by someone about to spend two days of machine
time -- so a figure that has gone stale in it is worse than an absent one, and a
step quietly dropped from it is the failure mode nobody notices until the run is
over.

WHAT THESE TESTS CAN AND CANNOT ESTABLISH
-----------------------------------------
They cannot establish anything about MariaDB, AliSQL, pgvector, Valkey, MongoDB
or Percona. No server is reachable from a test run. Every claim docs/10 makes
about an engine is marked supplied in its own §0 and is settled by the source it
cites, not here.

They also cannot re-derive the archived corpus count in §1. That tree lives
outside this repository -- `tests/test_config.py` resolves it relative to the
checkout's parent, which in a git worktree is not where it sits -- so the
figure is pinned to the comment it is quoted from and
`TestTheArchivedCorpusFigureIsPinnedToItsSource` checks the quote rather than the
files. That is the one figure in docs/10 that is not recomputed.

What they do establish is that every other figure in the document is the one this
repository holds at test time:

* docs/10 makes no line citation at all, over a pattern proved against a
  frozen fixture carrying one of each form it has to catch;
* every symbol docs/10 cites into this repository resolves in the tree, and
  every claim it makes about a named symbol's CONTENTS is checked by opening
  that symbol, in both directions -- a file the document starts naming with
  nothing checking it fails completeness, and a row whose anchor has moved
  fails its own check;
* the measurement version, the equivalence set, the client-bound threshold and
  its arithmetic, both profiles' `churn_readers`, every driver's `repair_path`
  and churn-hook overrides, and the estimator's fallback and its ratio are all
  read out of the modules and compared against the numbers the document prints;
* the cross-version refusal is CALLED rather than described, because "the report
  refuses" is a claim about behaviour and a grep for the word "refuses" would
  pass against a module that had stopped doing it;
* all six rollout steps and all seven checklist items are present;
* the item docs/10 §3 says is NOT satisfiable still says so, with its
  mechanical reason, and the condition that removed the row beside it still
  holds -- `report/crosscheck.py` exists and the report renders section 10;
* the two statements docs/10 §4 item 5 used to call stale now say that
  `main`'s grids DO meet, pinned by polarity so a differently wrong rewrite
  fails too, and neither still carries the `[8, 32]` claim in any spelling;
* the D1 cost table in §5 item 3 is recomputed from the estimator for both M
  grids and both engine sets, and agrees with the comment that states it.

This file was written one pull request earlier, when `report/crosscheck.py` did
not exist and `main`'s two M grids did not meet. Both guards fired on #16, which
is what they were for; what they assert now is the other side of the same
condition, so neither claim can drift back silently.

The staleness guard did the same thing one round later: it was written to fail
when the two sentences were FIXED, and it did, which is how docs/10 came to
stop flagging them. What stands in its place asserts the correction, in the
direction that cannot be satisfied by a sentence that is wrong a different
way.

The prose claims are pinned as PATTERNS and not as sentences -- see `CLAIMS`
below, which carries for each claim the rewordings it must accept and the
tidied-up forms it must refuse. A guard that fires on correct prose is a
slower-acting vacuous one, because it teaches the next author to delete it.

Each negative assertion is paired with a positive one that proves the needle is
findable, because a negative assertion passes just as happily against an empty
string as against a correct file. The same applies to every parser here: one
that matched nothing would make the assertions below vacuous, so each has a
test asserting what it found.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLOUT_DOC = os.path.join(VB_ROOT, "docs", "10-rollout.md")
METHODOLOGY_DOC = os.path.join(VB_ROOT, "docs", "05-methodology.md")
PERCONA_DOC = os.path.join(VB_ROOT, "docs", "09-adding-percona-server.md")


def _doc() -> str:
    return open(ROLLOUT_DOC).read()


def _plain(text: str) -> str:
    """The document's prose with markdown emphasis and whitespace normalised.

    Backticks and `**` are removed rather than matched, so a verbatim quotation
    from the spec can be checked against a sentence this document legitimately
    marks up -- the spec writes `client_bound` as bare text and docs/10 writes
    it in code font, and that difference must not read as a dropped step.
    """
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", "")).strip()


# ---------------------------------------------------------------------------
# The prose claims, as vocabularies rather than sentences
# ---------------------------------------------------------------------------
#
# A guard that demands one exact sentence fires on a legitimate rewrite, and a
# guard that fires on correct prose is a slower-acting vacuous one: it teaches
# the next author to delete it. Every claim below is therefore a pattern that
# pins the subject, the mechanism and the POLARITY and leaves the wording free.
#
# `accepts` is the rewordings the pattern was actually checked against and
# `rejects` the tidied-up forms it must still catch. Both are asserted by
# `TestTheClaimPatternsAreNeitherBrittleNorVacuous`, so the tolerance is
# executable rather than a promise in a comment -- and a pattern loose enough to
# accept its own negation fails there rather than passing quietly forever.
#
# Patterns run against `_plain(...)`, so a line break inside a sentence is a
# single space and markdown emphasis is gone.
CLAIMS = {
    "an absent version key means version 1": dict(
        pattern=r"\bare version 1\b",
        accepts=[
            "so all of them are version 1 by definition",
            "every one of them are version 1, not \"unknown\"",
            "records that predate the constant are version 1 by construction",
        ],
        rejects=[
            "records that predate the constant are of unknown version",
            "so all of them are version 5",
        ]),
    "the refusal costs the whole report, not one chart": dict(
        pattern=r"(there is no report|no report is produced|"
                r"the whole report fails|a non-zero exit)",
        accepts=[
            "prints the message and returns 2, so there is no report.",
            "returns 2, so no report is produced at all.",
            "the whole report fails rather than one chart losing a series.",
        ],
        rejects=[
            "so the chart is drawn with a caveat above it.",
            "so the older half is dropped and the report is generated.",
        ]),
    "the archived corpus is not in this repository": dict(
        pattern=r"(not part of this repository|outside this repository|"
                r"not in this repository|beside the checkout)",
        accepts=[
            "That tree is not part of this repository and is resolved",
            "The corpus lives outside this repository, so a test",
            "it sits beside the checkout rather than inside it",
        ],
        rejects=[
            "The corpus in this repository holds 239 files",
        ]),
    "client_bound cannot be read off the flag": dict(
        section='### 1. "Every concurrency record has recall, and no point is `client_bound` (or the report marks it)"',
        pattern=r"client_bound half\b.{0,120}?"
                r"\b(cannot|can not|is not|does not)\b.{0,60}?"
                r"\b(satisf|establish|answer|settl|discharg|read|tell)",
        accepts=[
            "The client_bound half cannot be satisfied by reading that flag",
            "The client_bound half of this item cannot be established by "
            "reading the flag",
            "The client_bound half is not answered by that flag at all",
        ],
        rejects=[
            "The client_bound half is satisfied by reading that flag",
            "The client_bound half is exactly what the flag reports",
        ]),
    # Was "only pgvector has a repair path", which was true until Percona
    # Server declared `index-replace`. The vocabulary fixes the SET -- both
    # engines named, and MongoDB excluded -- rather than a count or a
    # sentence, so the next engine to acquire one fails this instead of
    # quietly joining a number nobody re-read.
    "two engines have a repair path: pgvector and percona server": dict(
        pattern=r"(two engines|pgvector and percona|percona server and "
                r"pgvector)\b.{0,160}?a repair path"
                r"|a repair path\b.{0,160}?"
                r"(pgvector and percona|percona server and pgvector)",
        accepts=[
            "Two engines have a repair path: pgvector and Percona Server.",
            "pgvector and Percona Server are the two that declare a repair "
            "path; MongoDB inherits none.",
            "There is a repair path on pgvector and Percona Server, and "
            "nowhere else.",
            "Exactly two engines have a repair path, pgvector and Percona "
            "Server, and they are not comparable in cost.",
        ],
        rejects=[
            "Only pgvector has a repair path.",
            "pgvector and MongoDB both have a repair path.",
            "Every engine has a repair path of some kind.",
        ]),
    "mongodb inherits the no-repair default": dict(
        pattern=r"MongoDB\b.{0,60}?"
                r"(inherits|does not override|keeps the default|has none)",
        accepts=[
            "MongoDB inherits \"none\".",
            "MongoDB does not override it and reports none.",
            "MongoDB keeps the default, so the panel would be empty.",
        ],
        rejects=[
            "MongoDB declares a repair path of its own.",
        ]),
    "mongodb and valkey are the two that override the sync wait": dict(
        pattern=r"(MongoDB and Valkey|Valkey and MongoDB)\b.{0,60}?"
                r"(override|implement|supply)",
        accepts=[
            "returns 0.0 by default and MongoDB and Valkey override it.",
            "MongoDB and Valkey both override it; nothing else does.",
            "Valkey and MongoDB implement it and the rest inherit the no-op.",
        ],
        rejects=[
            "only MongoDB overrides it.",
        ]),
    "valkey is no longer a no-op": dict(
        pattern=r"Valkey\b.{0,40}?no longer\b",
        accepts=[
            "Valkey is no longer a no-op either.",
            "Valkey no longer records what it recorded before.",
            "Since #14 Valkey is no longer silent here.",
        ],
        rejects=[
            "Valkey remains a no-op, as before.",
        ]),
    "mariadb and alisql override no churn hook": dict(
        pattern=r"MariaDB and AliSQL\b.{0,60}?"
                r"(override no|override none|no churn hook|no-ops?)",
        accepts=[
            "MariaDB and AliSQL override no churn hook at all",
            "MariaDB and AliSQL override none of them",
            "MariaDB and AliSQL are still the no-ops here",
        ],
        rejects=[
            "MariaDB and AliSQL now override the sync wait as well",
        ]),
    "no readers means no during-writes record": dict(
        pattern=r"during_writes churn record is\b.{0,40}?"
                r"(never written|not written|absent)",
        accepts=[
            "the during_writes churn record is never written",
            "the during_writes churn record is not written at all",
            "the during_writes churn record is absent",
        ],
        rejects=[
            "the during_writes churn record is written with zero readers",
        ]),
    "main pins four readers and dev pins none": dict(
        pattern=r"main\.yml pins\b.{0,40}?churn_readers.{0,8}?4\b"
                r".{0,120}?dev\.yml pins\b.{0,20}?0\b",
        accepts=[
            "main.yml pins ops.churn_readers: 4 (a) and dev.yml pins 0 (b).",
            "main.yml pins ops.churn_readers: 4, while dev.yml pins 0.",
        ],
        rejects=[
            "main.yml pins ops.churn_readers: 4 and dev.yml pins 4.",
        ]),
    "smoke inherits the default reader count": dict(
        pattern=r"smoke\.yml\b.{0,40}?(does not pin|leaves|never pins)"
                r".{0,40}?churn_readers.{0,80}?default of 4",
        accepts=[
            "config/profiles/smoke.yml does not pin ops.churn_readers, so it "
            "inherits the default of 4",
            "smoke.yml leaves ops.churn_readers alone and takes the default "
            "of 4",
        ],
        rejects=[
            "smoke.yml pins ops.churn_readers to 4",
        ]),
    "a green smoke is not evidence about the memory policy": dict(
        pattern=r"(green )?smoke\b.{0,60}?not evidence",
        accepts=[
            "a green smoke run is not evidence the memory policy is sound",
            "a smoke run that passes is not evidence about the ceiling",
        ],
        rejects=[
            "a green smoke run is evidence the memory policy is sound",
        ]),
    "there is no crosscheck profile": dict(
        pattern=r"no crosscheck profile",
        accepts=[
            "and there is no crosscheck profile in config/profiles/.",
            "there is no crosscheck profile either, so D1 is open.",
        ],
        rejects=[
            "the crosscheck profile already covers it.",
        ]),
    "T5a landed and T5c did not": dict(
        pattern=r"T5a (has landed|landed|is merged)\b.{0,900}?"
                r"T5c\b.{0,160}?(does not exist|is absent|has not landed)",
        accepts=[
            "T5a has landed - stats.py is there. T5c is optional and the "
            "file does not exist.",
            "T5a landed in PR 0's branch. T5c (rate.py) has not landed.",
        ],
        rejects=[
            "T5a has landed and T5c has landed with it.",
        ]),
    "the consistency section now exists": dict(
        section='### 5. "Consistency checks pass, or each failure is explained"',
        # Tempered rather than a list of spellings: the window between
        # "section" and "exists" must not contain a negation, so word order is
        # free and polarity is not.
        pattern=r"section\b(?:(?!\bnot\b|\bno longer\b|\buntil\b)[\s\S])"
                r"{0,40}?\bexists\b",
        accepts=[
            "**This section now exists.** `report/crosscheck.py:327`'s",
            "This section exists now, and here is what to read in it.",
            "The consistency section exists as of #16.",
        ],
        rejects=[
            "This section does not exist at this commit.",
            "The consistency section is not reviewable until T5b lands.",
            "This section no longer exists.",
        ]),
    "item 5 left the unsatisfiable table": dict(
        section='## 3. One checklist item is not satisfiable as written',
        pattern=r"item 5\b.{0,80}?(was in this table and has left|"
                r"has left (it|this table)|left this table|is no longer here)",
        accepts=[
            "**Item 5 was in this table and has left it.** It was",
            "Item 5 has left this table, and the condition is why.",
            "Item 5 is no longer here, because the section now exists.",
        ],
        rejects=[
            "Item 5 is in this table because the section does not exist.",
        ]),
    "an empty section on main is a fault": dict(
        section='### 5. "Consistency checks pass, or each failure is explained"',
        # Replaces "two places still say the cross-check is empty on main",
        # which this document made until the two sentences were corrected.
        # Tempered on polarity rather than spelled: the window between `main`
        # and `fault` must not contain a word that flips it, so wording is
        # free and the claim is not.
        pattern=r"`?main`?\b(?:(?!\bby design\b|\bnot a fault\b|"
                r"\bno fault\b|\bexpected\b)[\s\S]){0,160}?\bfault\b",
        accepts=[
            "**On `main` an empty section is a fault, not a design.**",
            "On main, an empty consistency section is a fault to investigate.",
            "main's grids meet, so an empty section is a fault.",
        ],
        rejects=[
            "On `main` an empty section is empty by design.",
            "On `main` an empty section is expected, not a fault.",
            "On `main` an empty section is no fault at all.",
        ]),
    "the README figure and the estimator are different bases": dict(
        section='### 3. The cost envelope, and what D1 cost',
        # Either order: the sentence that introduces the warning names the
        # difference before it names README.
        pattern=r"(README\b.{0,200}?(hand method|different bas|"
                r"not the number)"
                r"|(hand method|different bas|not the number)"
                r".{0,200}?README\b)",
        accepts=[
            "Those are ingest hours, and they are not the number in "
            "`README.md`.",
            "README reports ~61 h by the hand method of docs/07 §1",
            "The README table and the estimator are different bases.",
        ],
        rejects=[
            "README prints the same figure the estimator does.",
        ]),
    "the measured MySQL-family ingest rates": dict(
        pattern=r"mariadb\b.{0,12}?150\b.{0,60}?mariadb123\b.{0,12}?110\b"
                r".{0,60}?alisql\b.{0,12}?55\b",
        accepts=[
            "are mariadb 150, mariadb123 110 and alisql 55",
            "are mariadb at 150, mariadb123 at 110, alisql at 55 rows/s",
        ],
        rejects=[
            "are mariadb 150, mariadb123 110 and alisql 150",
        ]),
}


def _section(heading: str) -> str:
    """The slice of docs/10 under `heading`, up to the next heading of its kind.

    Claims about one checklist item have to be asserted INSIDE that item. A
    document-wide search let §4 item 5 revert to "This section does not exist"
    while still passing, because the words it looked for existed elsewhere on
    the page -- measured, not supposed: that mutation came back with an
    identical honest and mutant value.
    """
    assert heading in _doc(), f"docs/10 has no section headed {heading!r}"
    rest = _doc().split(heading, 1)[1]
    depth = len(heading) - len(heading.lstrip("#"))
    return rest.split("\n" + "#" * depth + " ", 1)[0]


def _claim(name: str):
    """Assert docs/10 still makes the named claim, however it is worded.

    A claim carrying `section` is searched only there, which is what keeps a
    negated sentence in one item from being satisfied by a positive one in
    another.
    """
    entry = CLAIMS[name]
    pattern = entry["pattern"]
    where = entry.get("section")
    haystack = _section(where) if where else _doc()
    found = re.search(pattern, _plain(haystack), re.I | re.S)
    assert found, (
        f"docs/10 no longer makes this claim: {name}\n"
        f"  looked in: {where or 'the whole document'}\n"
        f"  pattern: {pattern}\n"
        f"  the pattern is a vocabulary, not a sentence -- if the claim is "
        f"still true and only the wording moved, widen CLAIMS rather than "
        f"restoring the old sentence")
    return found


# ---------------------------------------------------------------------------
# §1, §2, §4, §5: every claim docs/10 makes about this repository's contents
# ---------------------------------------------------------------------------
#
# WHAT THIS REPLACED. docs/10 cited `path`-colon-line and this file re-opened
# every cited line. The table was real coverage -- 51 distinct citations, all
# of them into this repository, every one checked -- and it kept firing on
# correct documents, because a line number dies to ANY edit above it. Line
# citations here and in docs/09 went stale twice in two days and put `master`
# red both times (#41, #43), and in both cases a later round of edits inside
# the SAME change moved lines below the point where the citations had been
# derived. This file's own preamble to that table recorded the mechanism
# happening to the table's own comment: two numbers in prose ABOUT the table
# said `48-49` and `43-51` until an import line was added above
# `_INGEST_ROWS_PER_S` and moved both by one.
#
# So docs/10 cites symbols, and the machinery is imported from
# `tests/test_e1_percona_design.py` rather than copied. ONE implementation for
# the two documents under one policy, deliberately: the previous arrangement
# had two `_CITE_RE`s that were supposed to be identical, and
# `tests/test_e15_methodology_roadmap_doc.py` carries a third with a comment
# explaining that it is kept "deliberately identical" to one of them. Three
# patterns that must agree are three chances to disagree. The classifier
# itself lives in the docs/11 guard and is imported by both of the others, so
# all three documents resolve a path and a symbol the same way.
from .test_e1_percona_design import (  # noqa: E402
    CITATION_FIXTURE, CITATION_FIXTURE_EXPECTED, GENERATED_DIRS, GUARD_FILES,
    cited_repo_files, doc_spans, line_citations, resolve_path, symbol_body,
    symbol_index)


def _symbol_text(path: str, symbol: str) -> str:
    """A symbol's declaration, or a failure naming the file and the symbol.

    The successor to `_cited_body(path, "lo-hi")`, which several tests below
    used to quote a comment out of a module by line. Those calls were line
    arithmetic inside the guard that enforces the no-line-number policy, and
    they rotted the same way: an insertion anywhere above the range moved it.
    They are symbols now, and the coarsening is real and stated -- a function
    body is wider than a five-line window, so where the narrower region
    mattered the test below asserts the ORDER of two anchors inside the body
    rather than their distance.
    """
    body = symbol_body(path, symbol)
    assert body is not None, (
        f"{path} no longer declares {symbol!r}; docs/10 cites it, and a "
        f"renamed symbol is the one failure a symbol citation exists to catch")
    return body


#: `(path, symbol)` -> the string, or strings, docs/10 says is inside that
#: symbol. Every anchor from the old `CITED_IN_REPO` is here, re-keyed from a
#: line spelling to the symbol that line sat in. The value may be a tuple,
#: which is how several citations into one function collapse into one row.
#:
#: STILL THE CONTENTS, not the existence: the symbol is located and the string
#: is asserted to be inside it. The conversion changed what survives an edit,
#: not what is being compared.
#:
#: Markdown rows name a SECTION HEADING. A heading is the only thing in a
#: sibling document that this one can point at and that survives an edit, and
#: §6 of this document records why the §-numbers are stable by convention
#: (docs/05's §2.0 is numbered 2.0 precisely so that nothing below it was
#: renumbered).
CITED_SYMBOLS = {
    # -- §1's measurement versions ---------------------------------------
    # Four citations into one comment block, which is the version charter
    # above the constant. The old table had them as four line ranges; they are
    # one row because they are one piece of writing, and `symbol_body`
    # includes the `#:` block above a declaration for exactly this case.
    ("harness/metrics/records.py", "OPS_MEASUREMENT_VERSION"):
        ("OPS_MEASUREMENT_VERSION = 11",
         "239 ops record files and 2,134 ops records",
         "CHANGES WHAT IT MEASURES",
         "is a diagnostic, not a guard"),
    ("harness/metrics/records.py", "IMPLICIT_OPS_MEASUREMENT_VERSION"):
        "IMPLICIT_OPS_MEASUREMENT_VERSION = 1",
    ("harness/metrics/records.py", "OPS_MEASUREMENT_EQUIVALENT_VERSIONS"):
        "OPS_MEASUREMENT_EQUIVALENT_VERSIONS = frozenset({",
    ("harness/metrics/stats.py", "bootstrap_ci"):
        "def bootstrap_ci",
    ("report/measurement_versions.py", "refuse_mixed_measurement_versions"):
        "if not unmatched and len(groups) < 2:",
    ("report/measurement_versions.py", "_SAME_MEASUREMENT_FACT"):
        "These versions denote the same measurement",
    ("report/generate.py", "main"):
        'print(f"[report] {exc}", file=sys.stderr)',

    # -- §2 and §4 item 1: the reader pool and the client-bound flag -----
    ("config/profiles/dev.yml", "churn_readers"):
        "churn_readers: 0",
    ("harness/workloads/churn_readers.py", "_reader_count"):
        "if readers <= 0:",
    ("harness/workloads/churn.py", "run"):
        ("readers are off, which is the one", "if reader_count:"),
    ("harness/workloads/pool_window.py", "CLIENT_BOUND_UTIL"):
        "CLIENT_BOUND_UTIL = 0.80",
    ("harness/workloads/pool_window.py", "PoolWindow.client_bound"):
        "return self.client_cpu_util > CLIENT_BOUND_UTIL",
    ("harness/workloads/clients.py", "<module>"):
        ("flattens at 2.06 cores", "synthetic SHAPE may not reproduce"),

    # -- §2 step C and §4 item 2: the calibration statuses ---------------
    ("harness/workloads/calibrate.py", "STATUS_FLOOR_ABOVE_TARGET"):
        'STATUS_FLOOR_ABOVE_TARGET = "floor_above_target"',
    ("harness/workloads/calibrate.py", "STATUS_TARGET_UNREACHED"):
        'STATUS_TARGET_UNREACHED = "target_unreached"',

    # -- §4 item 3: the selectivity overshoot ----------------------------
    ("harness/datasets.py", "FilterThreshold"):
        "1.04%-3.47% of the rows (median 1.43%)",

    # -- §4 item 4: the repair paths and the index-sync hook -------------
    ("harness/drivers/base.py", "EngineDriver.repair_path"):
        ('repair_path: str = "none"', "Valkey and Percona Search inherit"),
    ("harness/drivers/base.py", "EngineDriver.wait_index_synced"):
        "def wait_index_synced",
    ("harness/drivers/postgres.py", "PostgresDriver.repair_path"):
        'repair_path = "vacuum_analyze"',
    ("harness/drivers/mysql_family.py", "PerconaDriver.repair_path"):
        'repair_path = "index-replace"',
    ("config/engines/percona.yml", "capabilities"):
        "bulk_vector_index_build",

    # -- §4 item 5 and §2 step E: the consistency section ----------------
    ("report/crosscheck.py", "consistency_checks"):
        "def consistency_checks",
    ("report/crosscheck.py", "CROSSCHECK_MIN_TOLERANCE"):
        "CROSSCHECK_MIN_TOLERANCE = 0.01",
    ("report/render.py", "render_markdown"):
        "## 10. Consistency checks",
    ("report/render.py", "_consistency_section"):
        "the grids meet, so an empty section there is the fault case",

    # -- §4 item 7: the manifest -----------------------------------------
    ("orchestrator/manifest.py", "Manifest.__init__"):
        '"ops_measurement_version": OPS_MEASUREMENT_VERSION',
    ("report/render.py", "_environment_rows"):
        '"Ops measurement version"',

    # -- §5 items 2 and 3: the estimator and the cost table --------------
    ("orchestrator/estimate.py", "_INGEST_ROWS_PER_S"):
        ("Measured ingest rates, rows/s, on a Xeon Gold 6230.",
         "by 3x is worse than none",
         '"mariadb": 150'),
    ("orchestrator/estimate.py", "_effective_rate"):
        "_INGEST_ROWS_PER_S.get(engine)",
    # `m_values` appears twice in this file, under `ann` and under `ops`, and
    # the YAML locator concatenates both blocks -- so the anchors are chosen
    # to belong to the `ann` one and to nothing else.
    ("config/profiles/main.yml", "m_values"):
        ("m_values: [8, 16, 32]", "+17.04 h of ingest for"),
    ("config/profiles/main.yml", "churn_readers"):
        "churn_readers: 4",

    # -- the two sibling documents this page quotes ----------------------
    ("docs/05-methodology.md", "The two filter modes"):
        "The selectivity recorded is the one that was delivered",
    ("docs/05-methodology.md", "2.7 Cross-checking the two measurement paths"):
        "the grids meet, so an empty section there is the fault case",
    ("docs/05-methodology.md", "5. Reproducibility"):
        "config.ops_measurement_version",
    ("docs/09-adding-percona-server.md",
     "is not a cache, and that is a fairness problem"):
        "This is the finding most likely to change what gets run",
    ("docs/09-adding-percona-server.md",
     "What the default costs, with the inputs named"):
        "These figures are projections, not measurements.",
    ("docs/09-adding-percona-server.md",
     "The policy: `innodb_hnsw_max_memory = 0`"):
        "the gate cannot surface this at all",
}

#: Repository files docs/10 NAMES but makes no claim about the contents of,
#: with what covers each instead. Asserted cited and asserted row-less, so an
#: entry that acquires a row is a contradiction rather than a silent overlap.
#:
#: Same honest limit as the sibling table in tests/test_e1_percona_design.py:
#: this says the document makes no CONTENTS claim about these files, and
#: nothing can check that. What it buys is that a NEW file named by the
#: document fails here until somebody decides which side of the line it is on.
PATHS_WITHOUT_A_CLAIM = {
    "config/profiles/smoke.yml":
        "§2 step C's claim is that it does NOT pin `ops.churn_readers`, so it "
        "inherits the default of 4. A negative claim about a file has no "
        "symbol to open; TestTheProfileReaderCountsAreRead loads both "
        "profiles and recomputes the inheritance.",
    "tests/fake_driver.py":
        "§0 and §4 item 1 name it to say what a figure measured against it is "
        "a measurement OF -- this harness, not an engine. That is a claim "
        "about provenance, not about its contents.",
    "tests/test_stats.py":
        "named in §1 as the test that landed beside `bootstrap_ci`.",
    "tests/test_e2_percona_wiring.py":
        "named in \u00a75 item 2, after #62 landed, as one of the two files "
        "that "
        "exercise the gap list with a probe engine nothing could ever have "
        "measured. A claim about what a test covers, not about its contents "
        "-- and this entry is here because the forward check demanded it on "
        "the rebase onto #62 rather than because anyone remembered to look.",
    "tests/test_estimate.py":
        "named in §5 item 3 as what pins both cost bases; "
        "TestTheDecisionD1IsDecidedAndItsCostIsRecomputed recomputes the "
        "table from the estimator rather than quoting that file.",
    "tests/verify-alisql-traps.sh":
        "§2 step B names it as a script the operator runs against a live "
        "server. Nothing in a test run can reach one.",
    "tests/test_e3_rollout_doc.py":
        "this file. §6 describes it, and a row would be this guard checking "
        "itself -- the shape docs/11 §0 records as worse than no guard at all.",
}

#: Paths docs/10 names BECAUSE THEY DO NOT EXIST, with the reason. Asserted
#: absent rather than skipped -- the same mechanism, and the same single entry,
#: as `ABSENT_PATHS` in tests/test_e15_methodology_roadmap_doc.py, which is
#: where the reasoning is written out at length.
ABSENT_PATHS = {
    "harness/workloads/rate.py":
        "T5c, the open-loop latency workload: optional in the spec and never "
        "written. §1's PR 5 row says so, and docs/11 §6 is the design for it. "
        "If this file appears, §1 is wrong and has to say what landed.",
}

#: Code-font spans that look like a symbol and are not one. Empty, and that is
#: a finding rather than a default: every identifier docs/10 puts in code font
#: is a thing in this tree, and the one external source it quotes -- the
#: project specification -- is named in italics as *SPEC.md*, which is what
#: retired `FOREIGN_MARKERS`.
VOCABULARY = {}


_MERGE_SUBJECT_RE = re.compile(r"Merge pull request #(\d+) from \S+/(\S+)")

#: The numbers §1 calls absent from the merge numbering, as a PATTERN and not
#: as one sentence -- the same choice `CLAIMS` makes above and for the same
#: reason. The first version searched for the literal "#4 does not appear",
#: which a reword to "#4 is absent from this history" would have reddened
#: `master` for no defect at all: the brittleness this file spends its whole
#: preamble arguing against.
#:
#: The subject is the number, the polarity is absence, and the wording is
#: free. Bounded to one sentence (no period) and stopped at the next `#N`, so
#: "#5 landed, and #4 does not appear" names #4 and not both.
#: `TestTheAbsenceVocabularyIsNeitherBrittleNorVacuous` carries the rewordings
#: it must accept and the forms it must still refuse.
_ABSENT_NUMBER_RE = re.compile(
    r"#(\d+)\b(?:(?!#\d)[^.])*?"
    r"(?:does not appear|never appears|does not occur|is absent|"
    r"does not exist|is missing|appears nowhere)")

#: Rewordings the pattern must accept, and forms it must still refuse. Both
#: are asserted, so the tolerance is executable rather than promised.
ABSENCE_ACCEPTS = (
    ("#4 does not appear in this history at all", [4]),
    ("#4 is absent from this history", [4]),
    ("#4 never appears as a merge", [4]),
    ("#4 appears nowhere in the merge subjects", [4]),
    ("the repository skipped a number: #4 does not occur", [4]),
    # The absence phrase belongs to the nearer number, not to both: a
    # pattern that swept up #5 here would read the document as naming two
    # gaps and fail on correct prose.
    ("#5 landed. #4 does not appear", [4]),
)
ABSENCE_REFUSES = (
    # Present rather than absent: the polarity is the whole point.
    "#4 appears in this history",
    "#4 is in this history",
    # A number that merged, described as merged.
    "#16 landed as t5b-stats",
    # No number at all.
    "one pull request number does not appear in this history",
)


def _numbers_called_absent(text: str) -> set:
    return {int(m) for m in _ABSENT_NUMBER_RE.findall(_plain(text))}


def _merges_at(revision: str) -> Dict[int, str]:
    """`{pull request number: branch}` for the merges reachable from `revision`.

    Takes a revision rather than reading HEAD, so the same parser answers both
    "what does the history say now" and "what did it say at the commit where
    this guard went red", and the two cannot disagree about how a merge subject
    is spelled.
    """
    try:
        out = subprocess.run(
            ["git", "log", "--merges", "--format=%s", revision],
            cwd=VB_ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:
        pytest.skip(f"{revision} is not resolvable in this checkout")
    found: Dict[int, str] = {}
    for line in out.stdout.splitlines():
        m = _MERGE_SUBJECT_RE.match(line)
        if m:
            found[int(m.group(1))] = m.group(2)
    return found


class TestTheClaimPatternsAreNeitherBrittleNorVacuous:
    """What `CLAIMS` is allowed to accept, and what it must still refuse.

    This is the half that keeps the prose guards usable. A pattern tight enough
    to demand one sentence fails the `accepts` rows and gets deleted by the next
    author; a pattern loose enough to match anything passes the `accepts` rows
    and fails the `rejects` ones. Both failures land here rather than in a
    document that looks guarded and is not.
    """

    @pytest.mark.parametrize("name", sorted(CLAIMS))
    def test_the_document_still_makes_the_claim(self, name):
        _claim(name)

    @pytest.mark.parametrize("name", sorted(CLAIMS))
    def test_the_pattern_accepts_a_legitimate_rewording(self, name):
        pattern = CLAIMS[name]["pattern"]
        accepts = CLAIMS[name]["accepts"]
        assert len(accepts) >= 2, (
            f"{name}: give the pattern at least two rewordings to prove it is "
            f"not keyed to one sentence")
        for wording in accepts:
            assert re.search(pattern, _plain(wording), re.I | re.S), (
                f"{name}: the pattern rejects a legitimate rewording, which "
                f"makes it a guard that fires on correct prose:\n  {wording}")

    @pytest.mark.parametrize("name", sorted(CLAIMS))
    def test_the_pattern_refuses_the_tidied_up_form(self, name):
        """Non-vacuity, and the trap this repo measured today: a check keyed to
        the right answer cannot see a wrong one."""
        pattern = CLAIMS[name]["pattern"]
        rejects = CLAIMS[name]["rejects"]
        assert rejects, f"{name}: name at least one form that must not pass"
        for wording in rejects:
            assert not re.search(pattern, _plain(wording), re.I | re.S), (
                f"{name}: the pattern accepts a form that reverses the claim, "
                f"so it cannot see the edit it exists to catch:\n  {wording}")

    @pytest.mark.parametrize(
        "name", sorted(n for n in CLAIMS if CLAIMS[n].get("section")))
    def test_a_scoped_claim_really_is_scoped(self, name):
        """Non-vacuity for `_section`: a slice that silently fell back to the
        whole document would make scoping decorative, which is the state that
        let §4 item 5 revert to its opposite and still pass."""
        heading = CLAIMS[name]["section"]
        slice_ = _section(heading)
        assert slice_.strip(), heading
        assert len(slice_) < len(_doc()), heading
        assert heading not in slice_, heading

    def test_every_claim_is_used(self):
        """A claim nobody calls `_claim()` for reads as coverage and is not."""
        source = open(os.path.abspath(__file__)).read()
        unused = [name for name in CLAIMS
                  if f'_claim("{name}")' not in source]
        assert not unused, unused


class TestTheCitationParsersFoundSomething:
    """Floors and sentinels for the imported machinery, run against docs/10.

    The parsers are shared with tests/test_e1_percona_design.py and are
    exercised there too. These are not duplicates of those: the floors here
    are docs/10's, and a classifier that silently stopped seeing this
    document's spans would pass that file's floors and fail these.
    """

    def test_the_classifier_found_paths_and_symbols(self):
        paths, symbols, prose = doc_spans(_doc())
        assert len(paths) >= 30, sorted(paths)
        assert len(symbols) >= 55, len(symbols)
        assert prose, "nothing classified as prose; the classifier widened"

    def test_the_document_names_the_repository_files_it_claims_about(self):
        found = cited_repo_files(_doc())
        assert len(found) >= 25, sorted(found)
        for sentinel in ("harness/metrics/records.py",
                         "report/crosscheck.py",
                         "config/profiles/main.yml"):
            assert sentinel in found, sentinel

    def test_the_index_reads_none_of_the_three_document_guards(self):
        """The self-poisoning exclusion, asserted from this file as well as
        from the one that builds the index. docs/10 names a guard class and a
        fake driver, so the exclusion is load-bearing here too, and a test
        that only lives beside the implementation is a test nobody runs when
        the importer changes."""
        read = set(symbol_index()["files"])
        for guard in GUARD_FILES:
            assert guard not in read, guard

    def test_the_index_reads_nothing_from_a_generated_tree(self):
        read = symbol_index()["files"]
        for prefix in GENERATED_DIRS:
            assert not any(f.startswith(prefix + "/") for f in read), prefix


class TestTheDocumentMakesNoLineCitation:
    """The policy, executable."""

    def test_docs_10_contains_no_line_citation(self):
        found = line_citations(_doc())
        assert not found, (
            "docs/10 cites a line number, which its own §0 forbids -- line "
            "citations in this repository rotted twice in two days (#41, "
            "#43). Name the symbol instead: "
            + ", ".join(sorted(set(found))))

    def test_the_detector_sees_every_form_it_has_to_see(self):
        """Non-vacuity, over the frozen fixture rather than over a document.

        THIS IS WHERE THE OLD ARRANGEMENT BROKE. docs/11's guard proved its
        own detector by running it against docs/10 and asserting at least
        thirty matches -- "docs/10 is made of these". That was true and it was
        a dependency between two documents' citation styles: converting this
        page would have reddened the docs/11 guard for no defect at all. A
        frozen literal cannot be converted out from under a test.
        """
        found = line_citations(CITATION_FIXTURE)
        assert len(found) == CITATION_FIXTURE_EXPECTED, found

    def test_the_detector_does_not_fire_on_correct_prose(self):
        assert not line_citations(
            "`max(0.01, 2 x the ops 95% CI half-width)` `p50/p95/max` "
            "`report/crosscheck.py` `ops.churn_readers: 4`")

    def test_the_policy_is_stated_in_the_document(self):
        doc = _doc()
        assert "#41" in doc and "#43" in doc, (
            "docs/10 §0 must justify the policy with the two pull requests "
            "that repaired the rot; without them it is a style preference")
        assert re.search(r"invalidated by \*\*any\*\* edit", doc)
        assert re.search(r"only by a rename", doc)

    def test_the_illustration_of_the_banned_form_is_not_in_code_font(self):
        assert re.search(r"\*some/file\.py:123\*", _doc()), (
            "docs/10 §0 must illustrate the banned form in italics; in code "
            "font it trips its own ban")

    def test_the_specification_is_never_cited_in_code_font(self):
        """What retired `FOREIGN_MARKERS`, and why there is nothing for one to
        do any more.

        The deny-list existed to say "SPEC.md is not in this checkout, so do
        not resolve a citation into it against a local file of the same name".
        Under the policy the specification is named in prose -- *SPEC.md* --
        so a backticked one is a violation the path arm fails on, not a
        category to be skipped. The old entry's reasoning survives as the
        assertion below: if a SPEC.md ever lands in the tree, this still fails
        rather than quietly resolving against it.
        """
        assert "`SPEC.md" not in _doc(), (
            "docs/10 cites the specification in code font; §0 requires it in "
            "prose, because a backticked path there means a path in this "
            "repository")
        assert not os.path.exists(os.path.join(VB_ROOT, "SPEC.md")), (
            "SPEC.md now exists in this repo; the spec references in docs/10 "
            "can and should become checkable citations")
        # Non-vacuity: the document still quotes the spec, in italics, and
        # still marks every quotation supplied.
        assert "*SPEC.md*" in _doc()
        assert "supplied" in _doc()


class TestEverySymbolAndPathResolves:
    """The existence layer: coarser than the table below, over everything."""

    def test_every_cited_path_resolves_to_one_file(self):
        paths, _symbols, _prose = doc_spans(_doc())
        bad = []
        for path in sorted(paths):
            if path in ABSENT_PATHS:
                continue
            resolved, hits = resolve_path(path)
            if resolved is None:
                bad.append(
                    f"{path}: "
                    + (f"ambiguous, {hits}" if hits else "no such file"))
        assert not bad, (
            "docs/10 cites these paths and this tree cannot resolve them:\n  "
            + "\n  ".join(bad))

    def test_every_cited_symbol_appears_in_the_tree(self):
        _paths, symbols, _prose = doc_spans(_doc())
        bad = sorted(s for s in symbols
                     if s not in VOCABULARY
                     and s not in symbol_index()["mentioned"])
        assert not bad, (
            "docs/10 names these symbols in code font and nothing in this "
            "tree has them -- renamed, or never existed: " + ", ".join(bad))

    def test_enough_symbols_resolve_to_a_declaration(self):
        """The half that keeps the half above from being satisfied by prose."""
        _paths, symbols, _prose = doc_spans(_doc())
        declared = sorted(s for s in symbols
                          if s in symbol_index()["declared"])
        assert len(declared) >= 45, (
            f"only {len(declared)} of {len(symbols)} cited symbols resolve to "
            f"a declaration: {declared}")

    def test_the_vocabulary_escapes_are_named_and_do_not_resolve(self):
        idx = symbol_index()
        for span, reason in VOCABULARY.items():
            assert reason.strip(), span
            assert span not in idx["mentioned"], (
                f"{span} is in VOCABULARY and resolves in the tree; it is a "
                f"citation, not vocabulary -- delete the entry")
        assert VOCABULARY == {}, (
            "VOCABULARY has entries; name them in the round report rather "
            "than letting them sit unchecked: " + ", ".join(VOCABULARY))

    @pytest.mark.parametrize("path", sorted(ABSENT_PATHS))
    def test_a_deliberately_absent_path_is_still_absent(self, path):
        assert ABSENT_PATHS[path].strip(), path
        assert not os.path.exists(os.path.join(VB_ROOT, path)), (
            f"{path} now exists; docs/10 describes it as absent and has to be "
            f"updated: {ABSENT_PATHS[path]}")

    @pytest.mark.parametrize("path", sorted(ABSENT_PATHS))
    def test_a_deliberately_absent_path_is_actually_cited(self, path):
        paths, _symbols, _prose = doc_spans(_doc())
        assert path in paths, (
            f"ABSENT_PATHS carries {path} and docs/10 no longer names it")


class TestEveryClaimAboutThisRepoIsOpenedAndMatched:
    """The symbol table: the old line check, re-keyed.

    Not "the symbol exists" -- the symbol is located and the string the
    document points at is asserted to be inside it, which is the same
    comparison the line check made.
    """

    @pytest.mark.parametrize("key", sorted(CITED_SYMBOLS))
    def test_the_symbol_holds_what_the_document_says(self, key):
        path, symbol = key
        anchors = CITED_SYMBOLS[key]
        if isinstance(anchors, str):
            anchors = (anchors,)
        full = os.path.join(VB_ROOT, path)
        assert os.path.isfile(full), f"{path} does not exist"
        body = symbol_body(path, symbol)
        assert body is not None, (
            f"docs/10 cites {symbol!r} in {path} and that file no longer "
            f"declares it -- renamed, or moved to another file. This is the "
            f"failure a symbol citation exists to catch.")
        for anchor in anchors:
            assert anchor in body, (
                f"docs/10 cites {path}'s {symbol!r} for {anchor!r}; that "
                f"symbol now reads: {body.strip()[:160]!r}")

    def test_every_repo_code_file_the_document_names_is_accounted_for(self):
        """Completeness, forward."""
        named = cited_repo_files(_doc())
        with_rows = {path for path, _symbol in CITED_SYMBOLS}
        unaccounted = sorted(
            path for path in named
            if path not in with_rows
            and path not in PATHS_WITHOUT_A_CLAIM
            and not path.startswith("docs/")
            # THE CARVE-OUT, AND WHY IT IS NARROWER THAN IT LOOKS. The old
            # forward check was keyed on CITATIONS and had no exemption: every
            # `path:line` into this repo needed a row. This one is keyed on
            # FILES, so it needs one, because both documents name sibling
            # documents and the root files constantly -- `CLAUDE.md` for an
            # invariant, `README.md` for a profile table, `docs/05` and
            # `docs/07` for a figure's provenance -- and requiring a contents
            # row for each would force a row for every passing reference.
            #
            # What the exemption does NOT do is leave those uncovered. The
            # rows that matter are still here: each `.md` row names a section
            # of a sibling document, opens it, and matches the sentence quoted
            # from it. The exemption only removes the REQUIREMENT, and it
            # removes it for the class of reference that is a pointer rather
            # than a claim.
            #
            # NO COUNT OF THOSE ROWS IS STATED HERE, and that is deliberate
            # rather than vague. This note was pasted into both guards saying
            # "this guard carries three `.md` rows and docs/09's carries one".
            # Measured: the docs/09 guard carries ONE and the docs/10 guard
            # carries SIX, across two files -- so "three" was false in both
            # copies, and the second clause was true only when read from one
            # of them. A figure in a comment is checked by nobody; the floor
            # below is checked on every run, which is the whole argument this
            # change exists to make.
            #
            # The honest cost: a contents claim about a `docs/` path with no row
            # is not detected, exactly as a contents claim about a file in
            # PATHS_WITHOUT_A_CLAIM is not. Both are the same limit and it
            # is the one this check cannot close.
            and path != "CLAUDE.md"
            and path != "README.md")
        assert not unaccounted, (
            "docs/10 names these repository files and nothing says whether it "
            "claims anything about them. Give each a CITED_SYMBOLS row or a "
            "PATHS_WITHOUT_A_CLAIM reason: " + ", ".join(unaccounted))

    def test_no_row_cites_a_file_the_document_dropped(self):
        """Completeness, reverse: a row for something the document no longer
        names is dead weight that reads as coverage."""
        named = cited_repo_files(_doc())
        stale = sorted({path for path, _symbol in CITED_SYMBOLS}
                       - set(named))
        assert not stale, (
            "these rows cite files docs/10 no longer names: "
            + ", ".join(stale))

    def test_every_identifier_row_names_a_symbol_the_document_cites(self):
        """The other half of reverse completeness, at symbol granularity.

        Scoped to identifier-shaped symbols. A `.md` row's symbol is a section
        heading, which this document points at as "§2.7" rather than by
        quoting the heading, and a YAML key may be reached through a dotted
        path the document never spells; both are covered by the path half
        above and by their own anchor check.
        """
        _paths, symbols, _prose = doc_spans(_doc())
        bad = []
        for path, symbol in sorted(CITED_SYMBOLS):
            if path.endswith((".md", ".yml", ".yaml")) or symbol == "<module>":
                continue
            leaf = symbol.rsplit(".", 1)[-1]
            if leaf not in symbols:
                bad.append(f"{path}:{symbol}")
        assert not bad, (
            "these rows name symbols docs/10 does not cite in code font, so "
            "the row is checking something the document stopped saying: "
            + ", ".join(bad))

    @pytest.mark.parametrize("path", sorted(PATHS_WITHOUT_A_CLAIM))
    def test_a_path_without_a_claim_is_named_and_really_has_no_row(self, path):
        assert PATHS_WITHOUT_A_CLAIM[path].strip(), path
        with_rows = {p for p, _symbol in CITED_SYMBOLS}
        assert path not in with_rows, (
            f"{path} is listed as carrying no contents claim and also has a "
            f"CITED_SYMBOLS row; one of the two is wrong")
        assert path in cited_repo_files(_doc()), (
            f"{path} is listed as named-without-a-claim and docs/10 no longer "
            f"names it; delete the entry")

    @pytest.mark.parametrize("key", sorted(CITED_SYMBOLS))
    def test_the_anchors_are_not_empty(self, key):
        """`CITED_SYMBOLS`'s values ARE the assertions, and nothing checked
        that one existed.

        `assert "" in body` is true of every body, so a row whose anchor was
        an empty string -- or a tuple containing one, which is the likelier
        typo -- would pass against anything the symbol happened to contain and
        read as coverage. Every other escape hatch in this file is asserted
        non-empty (`VOCABULARY`, `ABSENT_PATHS`, `PATHS_WITHOUT_A_CLAIM` all
        check their reasons); this was the table it was missing from, and it
        is the table that matters most. Inherited from the line-citation
        version, which had it too.

        A floor on LENGTH as well as on emptiness, because a one- or
        two-character anchor is the same defect slightly diluted: `"("` is in
        most function bodies.
        """
        anchors = CITED_SYMBOLS[key]
        if isinstance(anchors, str):
            anchors = (anchors,)
        assert anchors, f"{key} has no anchor at all"
        for anchor in anchors:
            assert isinstance(anchor, str), (key, anchor)
            # NON-SPACE characters, which is what the docstring says and
            # what `len(anchor.strip())` did not do: `"a    b"` strips to six
            # and is two characters of evidence.
            assert len("".join(anchor.split())) >= 6, (
                f"{key}'s anchor {anchor!r} is too short to be evidence; an "
                f"anchor this small is satisfied by most bodies")

    def test_the_markdown_rows_open_a_real_section(self):
        """The premise of the carve-out above, checked instead of counted.

        The forward check exempts every `docs/` path, and the note beside it
        says the rows that matter are still here. That is a claim about this
        table, so it is asserted: at least one `.md` row, every one of them
        resolving to a section that exists, and its anchor inside that
        section. The note used to carry a COUNT of these rows instead and the
        count was wrong in both guards -- which is why there is a floor here
        and no number there.
        """
        md = sorted(k for k in CITED_SYMBOLS if k[0].endswith(".md"))
        assert md, (
            "no `.md` rows at all, so the carve-out note's claim that sibling "
            "documents are still covered by their own anchors is false")
        for path, heading in md:
            body = symbol_body(path, heading)
            assert body is not None, (
                f"{path} has no section matching {heading!r}; a sibling "
                f"document's heading was renumbered or reworded")
            anchors = CITED_SYMBOLS[(path, heading)]
            for anchor in ((anchors,) if isinstance(anchors, str) else anchors):
                assert anchor in body, (path, heading, anchor)

    def test_the_symbol_locator_can_fail(self):
        """Proved able to come back empty, from this file as well, because
        every row above rests on it and a locator that returned the whole file
        would make every anchor pass."""
        assert symbol_body("report/crosscheck.py", "no_such_symbol_zq") is None
        body = symbol_body("report/crosscheck.py", "CROSSCHECK_MIN_TOLERANCE")
        assert body is not None and "CROSSCHECK_MIN_TOLERANCE = 0.01" in body
        assert "def consistency_checks" not in body


# ---------------------------------------------------------------------------
# §1: the measurement versions, recomputed
# ---------------------------------------------------------------------------

class TestTheVersionFiguresComeFromTheConstants:
    """Every version number docs/10 prints, read back out of `harness/`."""

    def test_the_document_states_the_current_version_the_code_stamps(self):
        from harness.metrics import records
        claimed = re.search(r"`OPS_MEASUREMENT_VERSION` is (\d+)", _doc())
        assert claimed, "docs/10 no longer states the measurement version"
        assert int(claimed.group(1)) == records.OPS_MEASUREMENT_VERSION

    def test_the_document_states_the_implicit_version(self):
        from harness.metrics import records
        assert records.IMPLICIT_OPS_MEASUREMENT_VERSION == 1
        _claim("an absent version key means version 1")

    def test_the_equivalent_versions_are_the_pair_the_document_names(self):
        """§1 says versions 1 and 2 denote the same measurement, and that the
        set is what changes the refusal's wording. Both halves are read from
        the module, because a third member would make the sentence false."""
        from harness.metrics import records
        assert sorted(records.OPS_MEASUREMENT_EQUIVALENT_VERSIONS) == [1, 2]
        assert records.ops_versions_measure_the_same_thing([1, 2]) is True
        assert records.ops_versions_measure_the_same_thing([1, 5]) is False
        plain = _plain(_doc())
        # Quotations from the module, so matched literally: the document is
        # quoting `_SAME_MEASUREMENT_FACT`, and a paraphrase of a quotation is
        # a different claim.
        assert re.search(r"version 1 and (a )?version 2", plain)
        assert "denote the same measurement" in plain


class TestTheRefusalIsCalledAndNotDescribed:
    """§1 claims the refusal is unconditional on any two distinct versions.

    Called, not grepped. "The report refuses to mix versions" is a claim about
    behaviour, and a prose check for the word would pass unchanged against a
    module whose early return had been widened to let the equivalent pair
    through -- which is precisely the tidy-up §1 warns about, because the
    equivalence set makes it look reasonable.
    """

    @staticmethod
    def _rec(version=None, phase="churn"):
        extra = {} if version is None else {"ops_measurement_version": version}
        return {"phase": phase, "engine": "e", "dataset": "d", "extra": extra}

    def test_version_1_beside_version_5_is_refused(self):
        from report import measurement_versions as mv
        with pytest.raises(mv.MixedMeasurementVersions):
            mv.refuse_mixed_measurement_versions(
                [self._rec(), self._rec(5)], source="S")

    def test_the_equivalent_pair_is_refused_too(self):
        """The distinction §1 exists to get right: versions 1 and 2 mean the
        same thing and are still refused, so the refusal is not "two versions
        that disagree" -- it is any two distinct values."""
        from report import measurement_versions as mv
        with pytest.raises(mv.MixedMeasurementVersions):
            mv.refuse_mixed_measurement_versions(
                [self._rec(), self._rec(2)], source="S")

    def test_the_equivalence_set_changes_the_wording_and_not_the_decision(self):
        from report import measurement_versions as mv
        texts = {}
        for pair in ((None, 2), (None, 5)):
            with pytest.raises(mv.MixedMeasurementVersions) as caught:
                mv.refuse_mixed_measurement_versions(
                    [self._rec(pair[0]), self._rec(pair[1])], source="S")
            texts[pair] = str(caught.value)
        assert "denote the same measurement" in texts[(None, 2)]
        assert "denote the same measurement" not in texts[(None, 5)]

    def test_one_version_alone_is_not_refused(self):
        """Non-vacuity: a guard that raised on everything would pass all three
        assertions above while saying nothing."""
        from report import measurement_versions as mv
        mv.refuse_mixed_measurement_versions(
            [self._rec(5), self._rec(5)], source="S")
        mv.refuse_mixed_measurement_versions(
            [self._rec(), self._rec()], source="S")

    def test_the_refusal_is_a_non_zero_exit_and_not_a_caveat(self):
        """§1 says there is no report at all, which is stronger than "the
        chart is refused" and is the half an operator plans around."""
        body = _symbol_text("report/generate.py", "main")
        assert "MixedMeasurementVersions" in body
        assert "return 2" in body
        _claim("the refusal costs the whole report, not one chart")


class TestTheArchivedCorpusFigureIsPinnedToItsSource:
    """The one figure in docs/10 that is not recomputed, and why.

    `SMOKE_RESULTS/` sits beside the checkout, not inside it, and
    `tests/test_config.py:1499` resolves it from the checkout's parent -- which
    in a git worktree is the worktrees directory. So there is nothing for a test
    to count. The figures were counted independently while docs/10 was written
    (239 files, 2,134 records, no record carrying the version key) and they
    agreed with the comment; what is pinned here is that the document and the
    comment still say the same thing, so neither can be corrected alone.
    """

    NUMBERS = ("239", "2,134")

    def test_the_document_and_the_comment_agree(self):
        comment = _symbol_text("harness/metrics/records.py",
                               "OPS_MEASUREMENT_VERSION")
        doc = _doc()
        for number in self.NUMBERS:
            assert number in comment, (
                f"{number} is no longer in records.py:240-246; docs/10 §1 "
                f"quotes it and must be re-counted")
            assert number in doc, f"docs/10 no longer states {number}"

    def test_the_document_says_where_the_figure_came_from(self):
        """A reader must be able to tell this figure from the recomputed ones."""
        _claim("the archived corpus is not in this repository")
        assert "checkout" in _plain(_doc())

    def test_the_count_reproduces_where_the_corpus_is_reachable(self):
        """Derived rather than pinned, on a checkout that can see the tree.

        Whether it can depends on where the checkout sits -- `SMOKE_RESULTS/`
        is resolved from the parent directory, which in a git worktree is the
        worktrees directory. So this is a measurement in the main checkout and
        a skip elsewhere, and `test_the_document_and_the_comment_agree` above is
        what holds in both.
        """
        import json
        root = os.path.join(os.path.dirname(VB_ROOT), "SMOKE_RESULTS")
        if not os.path.isdir(root):
            pytest.skip(f"the archived corpus is not reachable from {VB_ROOT}")
        files = records = with_key = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not (name.startswith("ops-") and name.endswith(".jsonl")):
                    continue
                files += 1
                with open(os.path.join(dirpath, name)) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        records += 1
                        extra = json.loads(line).get("extra") or {}
                        if "ops_measurement_version" in extra:
                            with_key += 1
        assert with_key == 0, (
            f"{with_key} archived records now carry a version key; docs/10 "
            f"§1 says none do")
        # Anchored on the phrase, not the bare number, so the failure can quote
        # what the document actually claims instead of what it should claim.
        # Against `_plain`, because the phrase wraps across two lines in the
        # document and a raw-text regex matched neither the honest nor the
        # mutant form -- which is a probe that cannot fail, not a passing one.
        said = re.search(r"([\d,]+) ops record files, ([\d,]+) ops records",
                         _plain(_doc()))
        assert said, "docs/10 §1 no longer states the corpus counts"
        assert said.group(1) == f"{files}", (
            f"the corpus holds {files} ops record files; docs/10 §1 says "
            f"{said.group(1)}")
        assert said.group(2) == f"{records:,}", (
            f"the corpus holds {records:,} ops records; docs/10 §1 says "
            f"{said.group(2)}")


# ---------------------------------------------------------------------------
# §2 and §4: the six steps and the seven checklist items
# ---------------------------------------------------------------------------
#
# Verbatim from the spec's rollout section and pre-publication checklist. Pinned
# as literals because the spec is not in this repository: there is nothing to
# derive them from, and a runbook that has silently lost a step is the failure
# this file was asked to make impossible.
SPEC_STEPS = (
    "1. Merge PR 0 and PR 1; run dev for all six engines, then smoke for "
    "pgvector and MariaDB 11.8.",
    "2. Merge PR 2; run smoke for all engines. Check the operating-point "
    "table: MHNSW and VIDX should show floor_above_target or a low chosen ef.",
    "3. Merge PR 3 (largest); run dev per engine, then "
    "verify-alisql-traps.sh, then smoke. Compare tag results with the "
    "previous smoke run: they should match within noise, since only the "
    "measurement version changed.",
    "4. Merge PR 4; run churn-only smoke (--workloads build,churn) on "
    "pgvector and MongoDB first, where the new hooks do something.",
    "5. Merge PR 5; regenerate reports for the PR 1–4 smoke runs (no "
    "rerun needed) and review the consistency section.",
    "6. Only then start main. Old main results stay valid as "
    "ops_measurement_version 1 and are not mixed into new charts.",
)

SPEC_CHECKLIST = (
    "Every concurrency record has recall, and no point is client_bound (or "
    "the report marks it)",
    "Operating-point table reviewed; every non-ok status explained in the text",
    "Filtered: near_fail recall shown for ctag; time-bounded points greyed",
    "Churn: before/after maintenance shown where a repair path exists; "
    "repair_path: none stated elsewhere",
    "Consistency checks pass, or each failure is explained",
    "05-methodology.md updated for all five PRs",
    "Run manifest records ops_measurement_version, seeds and the resolved "
    "profile",
)


class TestEveryRolloutStepIsPresent:
    """A step dropped from a runbook is the defect nobody notices in time."""

    def test_there_are_six_of_them(self):
        """Non-vacuity for the table below: if SPEC_STEPS were ever trimmed,
        every assertion that walks it would pass over a shorter list."""
        assert len(SPEC_STEPS) == 6
        assert len({s[:2] for s in SPEC_STEPS}) == 6

    @pytest.mark.parametrize("step", SPEC_STEPS, ids=lambda s: s[:2])
    def test_the_step_is_quoted_verbatim(self, step):
        assert _plain(step) in _plain(_doc()), (
            "docs/10 no longer quotes rollout step " + step[:2])

    def test_each_step_says_where_it_lands(self):
        """The table is a mapping, not a list: a row with no destination is a
        step the document mentions and does not place."""
        rows = [r for r in _step_table_rows() if r]
        assert len(rows) == 6, rows
        for number, _spec_text, destination in rows:
            assert destination.strip(), f"step {number} has no destination"
            assert re.search(r"step[s]? [A-F]", destination), (
                f"step {number} does not name a revised step: {destination!r}")

    def test_the_revised_order_runs_a_to_f(self):
        plain = _plain(_doc())
        for letter in "ABCDEF":
            assert f"Step {letter} —" in plain, letter


def _step_table_rows():
    """The §2 step table, as (number, spec text, destination) triples."""
    text = _doc().split("## 2. The revised order", 1)[1].split("\n## ", 1)[0]
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        m = re.match(r"(\d)\.\s", cells[0])
        if not m:
            continue
        out.append((m.group(1), cells[0], cells[1]))
    return out


class TestEveryChecklistItemIsPresent:
    """All seven, each with a section of its own."""

    def test_there_are_seven_of_them(self):
        assert len(SPEC_CHECKLIST) == 7

    @pytest.mark.parametrize("item", SPEC_CHECKLIST,
                             ids=lambda s: s.split(";")[0][:30])
    def test_the_item_is_quoted_verbatim(self, item):
        assert _plain(item) in _plain(_doc()), (
            "docs/10 no longer quotes checklist item " + item[:40])

    def test_each_item_has_its_own_annotated_heading(self):
        """Quoting the item somewhere is not the same as annotating it. The
        document is asked for a per-item judgement, so each needs a heading.

        Scoped to §4: §5's subheadings are numbered too, and a document-wide
        regex would mix the two lists and then disagree with both.
        """
        section = _doc().split("## 4. The seven-item checklist", 1)[1]
        section = section.split("\n## ", 1)[0]
        headings = re.findall(r"^### (\d)\. (.+)$", section, re.M)
        assert [h[0] for h in headings] == list("1234567"), headings
        for index, (_number, title) in enumerate(headings):
            assert _plain(SPEC_CHECKLIST[index]) in _plain(title), title


# ---------------------------------------------------------------------------
# §3: the item that is not satisfiable as written, and the one that left
# ---------------------------------------------------------------------------

def _unsatisfiable_rows():
    """§3's table, as (item number, what, why) triples."""
    text = _doc().split("## 3. One checklist item", 1)[1].split("\n## ", 1)[0]
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"\d+", cells[0]):
            out.append(tuple(cells))
    return out


class TestTheUnsatisfiableItemStillSaysSo:
    """The claim a later reader is most likely to tidy into something false.

    It is a negative result. A reader who finds "this cannot be done" in a
    runbook and can see no reason for it deletes the sentence, so the row has
    to carry its mechanism -- and the mechanism has to be checkable, or the row
    is just a more confident sentence.

    THIS TABLE HELD TWO ROWS AND NOW HOLDS ONE, and the difference is worth
    writing down because it is the shape of every claim in this file. Item 5
    was unsatisfiable because `report/crosscheck.py` did not exist; the file
    exists now, so the row is gone and
    `TestItemFiveLeftTheTableForAReasonThatStillHolds` asserts the condition
    that removed it. Item 1 has no such condition: nothing short of replacing
    the quantity `client_bound` is computed from retires it.
    """

    def test_the_table_has_exactly_the_one_row(self):
        rows = _unsatisfiable_rows()
        assert len(rows) == 1, rows
        assert [r[0] for r in rows] == ["1"], rows

    def test_the_row_carries_a_mechanism_and_not_only_a_verdict(self):
        rows = {r[0]: r for r in _unsatisfiable_rows()}
        _n, what, why = rows["1"]
        assert "client_bound" in what
        # The mechanism, not the verdict: a CPU threshold, and the reason it
        # cannot see the case. Keyed on the two halves of the argument so that
        # a rewrite keeping the argument passes and one keeping only the
        # conclusion does not.
        assert "CPU" in why
        assert "sufficient but not necessary" in why or "not necessary" in why
        assert re.search(r"burns no CPU|no CPU|does not", why), why

    def test_the_client_bound_section_still_states_the_blindness(self):
        section = _doc().split("### 1. \"Every concurrency record", 1)[1]
        section = section.split("\n### ", 1)[0]
        _claim("client_bound cannot be read off the flag")
        assert "CLIENT_BOUND_UTIL" in section
        # Quotations, matched literally for the reason above.
        assert "fake_driver.py" in section
        assert "may not reproduce" in section


class TestItemFiveLeftTheTableForAReasonThatStillHolds:
    """The inverted tripwire.

    The earlier revision of this file asserted that `report/crosscheck.py` did
    NOT exist, and said in its own docstring that the guard failing would mean
    docs/10 needed rewriting. It did, and it did. The guard was right and this
    is what it becomes: the same condition, asserted the other way round, so
    that a document which stopped saying item 5 is satisfiable -- or a tree
    which stopped making it so -- fails here rather than drifting.

    Behaviour, not file existence. "The consistency section exists" is a claim
    about what the report can produce, and a module that imported but computed
    nothing would satisfy a `path.exists` check while leaving item 5 exactly as
    unreviewable as before.
    """

    def test_the_module_is_there_and_computes_something(self):
        from report import crosscheck
        out = crosscheck.consistency_checks([])
        assert isinstance(out, dict), out
        for key in ("matches", "coverage"):
            assert key in out, sorted(out)
        assert not os.path.exists(
            os.path.join(VB_ROOT, "harness", "workloads", "rate.py")), (
            "T5c has landed too; docs/10 §1 still calls it absent")

    def test_the_report_renders_a_section_for_it(self):
        """§4 item 5 sends the reader to section 10 by name."""
        render = open(os.path.join(VB_ROOT, "report", "render.py")).read()
        assert "## 10. Consistency checks" in render
        assert "def _consistency_section" in render

    def test_the_document_says_the_section_exists_and_what_to_read(self):
        section = _doc().split("### 5. \"Consistency checks pass", 1)[1]
        section = section.split("\n### ", 1)[0]
        _claim("the consistency section now exists")
        # The three things §4 item 5 tells the reader to look at. Named
        # individually because "the section exists" is not advice.
        assert "coverage" in section.lower(), section[:200]
        assert "`CROSSCHECK_MIN_TOLERANCE`" in section
        assert "`consistency_checks` in `report/crosscheck.py`" in section

    def test_the_document_records_why_the_row_left(self):
        """An audit trail, not a deletion. The next reader of §3 should be able
        to tell a row that was wrong from a row whose condition was removed."""
        section = _doc().split("## 3. One checklist item", 1)[1]
        section = section.split("\n## ", 1)[0]
        _claim("item 5 left the unsatisfiable table")
        assert "report/crosscheck.py" in section


class TestTheStatementsTheDocumentFlaggedAreCorrected:
    """§4 item 5 flagged two places that said `main`'s M grids do not meet.

    Both were falsified by the pull request that wrote them --
    `config/profiles/main.yml` moved to `[8, 16, 32]` in the same change that
    added the section those sentences describe -- so an operator reading
    either one after an empty section on a `main` run would have filed a real
    fault as expected behaviour.

    The previous revision of this class asserted they were STILL stale, and
    failed when they were fixed, which is how docs/10 came to stop flagging
    them. This asserts the other side, and it is deliberately not a literal:
    pinning the corrected sentence by its exact text would accept any other
    wrong sentence and reject a harmless reflow. So the claim is pinned by
    POLARITY -- `main` and `meet` with nothing between them that flips it --
    and the stale spellings are asserted absent after whitespace
    normalisation, because one of these sentences spans a line break and a
    prose guard a reflow defeats is not a guard.
    """

    FILES = ("report/render.py", "docs/05-methodology.md")

    #: The claim each file must make: `main`'s grids meet.
    #:
    #: Word order is free and polarity is not, which is the shape `CLAIMS`
    #: above uses. A regex pinning "main ... grids ... meet" in that order
    #: rejects "the grids overlap on main", which says the same thing -- and
    #: a guard that fires on correct prose teaches the next author to delete
    #: it. So this looks for the three ideas inside one window and refuses
    #: the window if anything in it flips the sense.
    MEET_VERB = r"\b(?:meet|intersect|overlap)\b"
    NEGATION = r"\b(?:not|never|cannot|no longer)\b|\bby design\b"
    #: Characters either side of `grids`. Wide enough for a sentence, narrow
    #: enough not to reach the paragraph above -- which in `render.py` is the
    #: general two-causes explanation and legitimately contains "do not
    #: overlap". Measured against both files.
    WINDOW = 110

    def _says_grids_meet(self, text):
        """Whether `text` states somewhere that main's grids meet."""
        for match in re.finditer(r"\bgrids\b", text, re.I):
            window = text[max(0, match.start() - self.WINDOW):
                          match.end() + self.WINDOW]
            if not re.search(r"`?main`?\b", window, re.I):
                continue
            if not re.search(self.MEET_VERB, window, re.I):
                continue
            if re.search(self.NEGATION, window, re.I):
                continue
            return window
        return None

    #: Spellings of the stale claim. Patterns, not one sentence: the figure
    #: that rotted can come back worded differently, and a guard keyed to the
    #: exact old text would miss that.
    STALE = (
        r"ann\.m_values:\s*`?\s*\[8,\s*32\]`?\s*against",
        r"grids\s+do\s+not\s+(meet|intersect)",
        r"nothing\s+can\s+match",
    )

    #: Where a labelled history clause starts. Both corrected passages record
    #: what they used to say, and one of them quotes "empty by design" while
    #: doing it -- so the absence check skips the clause that declares itself
    #: history, exactly as `tests/test_t4b_readers.py` does for the line
    #: counts it stopped restating. Keying the exemption to the LABEL rather
    #: than to the wording is the only way to keep an honest record legal
    #: while still catching the same words used as a current claim.
    HISTORY = "An earlier revision"
    HISTORY_WINDOW = 600

    def _live_prose(self, path):
        """The file's prose with labelled history removed, whitespace flat."""
        text = open(os.path.join(VB_ROOT, path)).read()
        flat = re.sub(r"\s+", " ", re.sub(r"\s*#\s*", " ", text))
        while self.HISTORY in flat:
            at = flat.index(self.HISTORY)
            flat = flat[:at] + flat[at + self.HISTORY_WINDOW:]
        return flat

    @pytest.mark.parametrize("path", FILES)
    def test_the_file_says_the_grids_meet(self, path):
        flat = self._live_prose(path)
        assert self._says_grids_meet(flat), (
            f"{path} no longer states that `main`'s M grids meet. The "
            f"profile is `[8, 16, 32]` against `[16]`, so an empty "
            f"consistency section on `main` is a fault; a file that does not "
            f"say so inverts the diagnostic the coverage table is for.")

    @pytest.mark.parametrize("path", FILES)
    def test_the_stale_claim_is_gone_in_every_spelling(self, path):
        flat = self._live_prose(path)
        for pattern in self.STALE:
            hit = re.search(pattern, flat, re.I)
            assert hit is None, (
                f"{path} states the stale claim again ({hit.group(0)!r}). "
                f"`main` runs `[8, 16, 32]` against `[16]`. If this is a "
                f"record of what the file used to say, introduce it with "
                f"{self.HISTORY!r} as both corrected passages do.")

    def test_the_patterns_are_neither_brittle_nor_vacuous(self):
        """The needles are findable and the polarity is load-bearing.

        Without this the two tests above would pass against an empty string
        and against any sentence containing the right nouns, which is how a
        negative assertion goes quietly vacuous.
        """
        accepts = [
            "On `main` the grids meet, so an empty section is the fault case.",
            "main sweeps [8, 16, 32] against [16]; the grids intersect.",
            "The grids overlap on main.",
        ]
        rejects = [
            "In `main` these grids do not intersect",
            "On `main` the grids never meet",
            "`main` configures [8, 32] against [16], so the check is empty "
            "by design",
            "",
        ]
        for text in accepts:
            assert self._says_grids_meet(text), text
        for text in rejects:
            assert not self._says_grids_meet(text), text
        # And each stale pattern matches the thing it bans.
        stale_text = ("`main` configures `ann.m_values: [8, 32]` against "
                      "`ops.m_values: [16]`, so these grids do not intersect "
                      "and nothing can match there")
        for pattern in self.STALE:
            assert re.search(pattern, stale_text, re.I), pattern

    def test_the_history_exemption_is_scoped_and_not_a_hole(self):
        """The exemption keeps an honest record legal without blanket amnesty.

        Honest: both corrected files pass, and each does mention the old
            `[8, 32]` grid inside a clause introduced by the label.
        Mutant (the same clause with the label removed): the stale pattern
            fires, so the exemption cannot be had by accident.
        """
        for path in self.FILES:
            raw = open(os.path.join(VB_ROOT, path)).read()
            assert self.HISTORY in raw, (
                f"{path} does not record what it used to say")
            assert "[8, 32]" in raw, (
                f"{path} dropped the record of the old grid entirely; the "
                f"exemption below is then describing nothing")
        unlabelled = ("the two grids do not intersect on main, so the check "
                      "is empty by design")
        assert re.search(self.STALE[1], unlabelled, re.I)

    def test_the_profile_is_what_makes_the_correction_true(self):
        """The claim is only right because the profile says so."""
        main = _profile("main.yml")
        ann, ops = set(main["ann"]["m_values"]), set(main["ops"]["m_values"])
        assert ops and ops <= ann, (sorted(ann), sorted(ops))
        _claim("an empty section on main is a fault")

    def test_the_document_names_both_places(self):
        section = _doc().split("### 5. \"Consistency checks pass", 1)[1]
        section = section.split("\n### ", 1)[0]
        assert "`_consistency_section`" in section
        assert "`report/render.py`" in section
        assert "`docs/05-methodology.md` §2.7" in section


# ---------------------------------------------------------------------------
# §4 item 1: the client-bound threshold and its arithmetic
# ---------------------------------------------------------------------------

class TestTheClientBoundArithmeticIsRecomputed:
    """§4 item 1's numbers, taken from the module rather than restated."""

    @staticmethod
    def _window(util):
        from harness.workloads.clients import PoolWindow
        kwargs = {field: None for field in PoolWindow._fields}
        kwargs["client_cpu_util"] = util
        return PoolWindow(**kwargs)

    def test_the_document_prints_the_threshold_the_module_holds(self):
        from harness.workloads import clients
        claimed = re.search(r"`CLIENT_BOUND_UTIL = ([\d.]+)`", _doc())
        assert claimed, "docs/10 no longer states the threshold"
        assert float(claimed.group(1)) == clients.CLIENT_BOUND_UTIL

    def test_the_flag_is_the_comparison_the_document_says_it_is(self):
        """Called, not grepped: the claim is that nothing but CPU is consulted."""
        assert self._window(0.9).client_bound is True
        assert self._window(0.1).client_bound is False
        assert self._window(None).client_bound is None

    def test_the_cited_utilisation_is_under_the_threshold(self):
        """The whole of §4 item 1 turns on 2.06 cores in a four-CPU container
        not reaching the threshold. Both inputs are read from the comment the
        document cites, the quotient is recomputed, and the flag is then asked
        -- so a changed threshold, a changed comment or a changed flag each
        fail here rather than leaving a plausible sentence in place."""
        from harness.workloads import clients
        comment = _symbol_text("harness/workloads/clients.py", "<module>")
        cores = re.search(r"flattens at ([\d.]+) cores", comment)
        cpus = re.search(r"of a (\w+)-CPU client container", comment)
        assert cores and cpus, comment
        assert cpus.group(1) == "four"
        util = float(cores.group(1)) / 4
        assert util < clients.CLIENT_BOUND_UTIL, util
        assert self._window(util).client_bound is False
        # Three places, not two: `clients.py` rounds its own quotient to 0.51
        # and docs/10 prints the exact one, so rounding to two decimals here
        # would make the document agree with the comment's rounding rather than
        # with the division. It caught this document printing 0.51.
        assert round(util, 3) == self._doc_quotient(), (
            "docs/10 prints a different quotient than the comment implies")

    @staticmethod
    def _doc_quotient():
        claimed = re.search(r"([\d.]+) of a four-CPU container", _doc())
        assert claimed, "docs/10 no longer prints the quotient"
        return float(claimed.group(1))

    def test_the_document_attributes_the_shape_to_the_fake_driver(self):
        """The correction this section exists for: the figure is a measurement
        of this harness against `tests/fake_driver.py`, and the module itself
        says the shape may not reproduce against a socket-bound driver. A
        document that dropped the attribution would be presenting it as a
        measurement of an engine."""
        plain = _plain(_doc())
        assert "tests/fake_driver.py" in plain
        assert "may not reproduce" in plain
        comment = _symbol_text("harness/workloads/clients.py", "<module>")
        assert "synthetic SHAPE may not reproduce" in comment


# ---------------------------------------------------------------------------
# §2 and §4 item 4: the profiles and the churn hooks
# ---------------------------------------------------------------------------

def _profile(name):
    import yaml
    with open(os.path.join(VB_ROOT, "config", "profiles", name)) as fh:
        return yaml.safe_load(fh)


class TestTheProfileReaderCountsAreRead:
    """§2's pooling hazard depends on two profile values and a third absence."""

    def test_main_and_dev_pin_the_values_the_document_names(self):
        assert _profile("main.yml")["ops"]["churn_readers"] == 4
        assert _profile("dev.yml")["ops"]["churn_readers"] == 0
        _claim("main pins four readers and dev pins none")

    def test_smoke_inherits_the_default_rather_than_pinning_it(self):
        """§2 step C turns on this: `smoke` is where the reader pool first
        runs, and it is where it runs because the key is absent."""
        from harness.workloads.churn_readers import DEFAULT_CHURN_READERS
        assert "churn_readers" not in _profile("smoke.yml")["ops"]
        assert DEFAULT_CHURN_READERS == 4
        _claim("smoke inherits the default reader count")

    def test_no_shipped_profile_sweeps_the_value(self):
        """§2's closing claim. Checked over every profile rather than the two
        the document names, because the claim is about all of them."""
        directory = os.path.join(VB_ROOT, "config", "profiles")
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".yml"):
                continue
            ops = (_profile(name) or {}).get("ops") or {}
            value = ops.get("churn_readers", 4)
            assert isinstance(value, (int, type(None))) \
                and not isinstance(value, bool), (name, value)

    def test_the_resolver_cannot_express_a_sweep(self):
        """The stronger half: a list would not survive resolution, so the
        hazard really is pooling by hand and not a profile nobody has written
        yet."""
        from orchestrator.ops_pass import _churn_readers
        assert _churn_readers({"churn_readers": 4}) == 4
        assert _churn_readers({"churn_readers": 0}) == 0
        assert _churn_readers({}) == 4
        assert _churn_readers({"churn_readers": None}) == 4
        with pytest.raises(TypeError):
            _churn_readers({"churn_readers": [0, 4]})


class TestTheChurnHooksAreReadFromTheDrivers:
    """§4 item 4's correction: one repair path, two sync waits."""

    @staticmethod
    def _drivers():
        import inspect
        from harness.drivers import base, mongo, mysql_family, postgres, valkey
        out = {}
        for module in (postgres, valkey, mongo, mysql_family):
            for obj in vars(module).values():
                if (inspect.isclass(obj)
                        and issubclass(obj, base.EngineDriver)
                        and obj is not base.EngineDriver
                        and obj.__module__ == module.__name__):
                    out[obj.__name__] = obj
        return out

    def test_the_drivers_were_found(self):
        """Non-vacuity: an empty mapping would make the two tests below pass."""
        found = self._drivers()
        assert len(found) >= 5, sorted(found)
        for name in ("PostgresDriver", "ValkeyDriver", "MongoDriver",
                     "MariaDBDriver", "AliSQLDriver"):
            assert name in found, sorted(found)

    def test_exactly_two_drivers_declare_a_repair_path(self):
        """**This read `only pgvector` and fired when Percona Server landed**,
        which is the direction it should fire in: §4 item 4 is about which
        engines get a before/after maintenance panel, and an engine that
        acquired a repair path without this page noticing would have been
        given one silently -- or denied one it had earned.

        Percona's is the expensive one of the two BY DESIGN: a whole graph
        rebuild per churn fraction against pgvector's VACUUM. A value
        assertion, so a third fails here rather than passing.

        **"declares a repair mechanism" and "can execute it at this pin" are
        two different claims, and this test is about the first.** It reads the
        CLASS attributes, which is why
        `PerconaDriver.declare_bulk_vector_index_build` shadowing
        `repair_path` with `"none"` per instance leaves it green and should.
        At the pinned `source.tag` Percona's records carry
        `repair_path: "none"` and no before/after maintenance panel, because
        that ALTER is the same statement
        `capabilities.bulk_vector_index_build: false` records as killing the
        server; the engine still *has* the mechanism, and the class is where
        that is declared. A reader who wants the per-pin answer wants
        `docs/05-methodology.md`'s churn section, not this assertion.
        """
        from harness.drivers.base import EngineDriver
        assert EngineDriver.repair_path == "none"
        overriding = {name: cls.repair_path
                      for name, cls in self._drivers().items()
                      if cls.repair_path != "none"}
        assert overriding == {"PostgresDriver": "vacuum_analyze",
                              "PerconaDriver": "index-replace"}, overriding
        _claim("two engines have a repair path: pgvector and percona server")
        _claim("mongodb inherits the no-repair default")

    def test_mongodb_and_valkey_override_the_sync_wait_and_not_the_repair(self):
        from harness.drivers.base import EngineDriver
        assert EngineDriver.wait_index_synced(None, 0) == 0.0
        waiting = sorted(
            name for name, cls in self._drivers().items()
            if cls.wait_index_synced is not EngineDriver.wait_index_synced)
        assert waiting == ["MongoDriver", "ValkeyDriver"], waiting
        _claim("mongodb and valkey are the two that override the sync wait")
        _claim("valkey is no longer a no-op")

    def test_mariadb_and_alisql_override_no_churn_hook(self):
        """The one sentence in §4 item 4 that is still about no-ops."""
        from harness.drivers.base import EngineDriver
        hooks = ("prepare_churn", "wait_index_synced", "maintain",
                 "restore_churn")
        drivers = self._drivers()
        for name in ("MariaDBDriver", "AliSQLDriver"):
            for hook in hooks:
                assert getattr(drivers[name], hook, None) is \
                    getattr(EngineDriver, hook, None), (name, hook)
        _claim("mariadb and alisql override no churn hook")

    def test_the_during_writes_record_needs_a_reader_pool(self):
        """§2 step A's correction, in two halves.

        The resolver must give `ops.churn_readers: 0` a pool of zero, and the
        churn workload must gate the record on that count. Only checking the
        first leaves the claim half-established: a resolver returning zero into
        a workload that wrote the record anyway would still pass.
        """
        from harness.workloads.churn_readers import _reader_count
        import numpy
        queries = numpy.zeros((16, 4), dtype="<f4")

        def factory():                  # never called on the readers<=0 path
            raise AssertionError("factory must not be called")

        assert _reader_count(0, queries, factory, "e") == 0
        assert _reader_count(4, queries, factory, "e") == 4
        assert _reader_count(4, queries, None, "e") == 0
        # The gate. `reader_baseline` stays None without it, and the record
        # is written only when both windows exist -- which is what the
        # comment above `reader_baseline` in `churn.run` states and this
        # document quotes.
        #
        # ONE BODY INSTEAD OF TWO LINE WINDOWS, and the coarsening is
        # recovered as an ORDER claim. The two citations used to be five-line
        # and one-line slices; `run` is one function and contains both, so a
        # pair of `in` tests over it would be weaker than what they replaced.
        # The gate has to come BEFORE the guarded write, which is the whole
        # content of "no readers means no during-writes record", so that is
        # what is asserted.
        body = _symbol_text("harness/workloads/churn.py", "run")
        assert "if reader_count:" in body, body[:200]
        assert "reader_baseline is not None" in body, body[:200]
        assert (body.index("if reader_count:")
                < body.index("reader_baseline is not None")), (
            "the reader-count gate no longer precedes the during-writes "
            "record it guards")
        _claim("no readers means no during-writes record")


# ---------------------------------------------------------------------------
# §4 item 6: which names docs/05-methodology.md is missing
# ---------------------------------------------------------------------------

class TestTheMethodologyGapsAreDerived:
    """§4 item 6 lists names absent from docs/05, and closes as they land.

    Both directions, deliberately. A one-sided check that the absent names are
    absent would pass against a document that had lost the names it calls
    present too, and the item's judgement -- "PR 3 and the version scheme are
    well covered" -- is half of what makes the gap meaningful.
    """

    #: PR 2's four names, which are the whole of what docs/05 still does not
    #: describe. `client_bound`, `client_cpu_util` and `during_writes` were
    #: here until they landed in §2.3 and §2.5; they moved to PRESENT below
    #: rather than being deleted, so the list cannot shrink by forgetting.
    #: Empty, and that is the item's TERMINAL state rather than a disabled
    #: guard: PR 2's four names moved to `PRESENT` when docs/05 §2.0 landed,
    #: as PR 1's and PR 4's did before them. `test_an_empty_absent_list_is_the
    #: _closed_state` asserts the emptiness is deliberate, because a loop over
    #: an empty tuple passes for the same reason a dead probe does.
    ABSENT = ()
    PRESENT = ("maintenance_state", "churn_readers", "ctag",
               "ops_measurement_version", "index_sync_s",
               "client_bound", "client_cpu_util", "during_writes",
               "ef_calibration", "floor_above_target", "target_unreached",
               "operating-point")

    @pytest.fixture(scope="class")
    def methodology(self):
        return open(METHODOLOGY_DOC).read()

    def test_the_names_the_document_calls_absent_are_absent(self, methodology):
        still_there = [name for name in self.ABSENT if name in methodology]
        assert not still_there, (
            "docs/05 now covers these and docs/10 §4 item 6 is stale: "
            + ", ".join(still_there))

    def test_the_names_the_document_calls_covered_are_present(self,
                                                              methodology):
        """The paired positive: it proves the needle is findable, so the
        assertion above is about docs/05 and not about an unreadable file."""
        missing = [name for name in self.PRESENT if name not in methodology]
        assert not missing, missing

    @staticmethod
    def _item6() -> str:
        return _doc().split("### 6. \"", 1)[1].split("\n### ", 1)[0]

    def test_the_document_names_every_absence_it_claims(self):
        """The list in §4 item 6 and the list here must be the same list."""
        section = self._item6()
        for name in self.ABSENT:
            assert name in section, (
                f"docs/10 §4 item 6 no longer names {name}, which this guard "
                f"still asserts is missing from docs/05")

    def test_an_empty_absent_list_is_the_closed_state(self):
        """`ABSENT` is empty, so the loop above passes over nothing.

        That is the state the item was designed to reach, but an empty loop
        and a deleted guard are indistinguishable from the outside, so the
        emptiness is asserted here together with the sentence docs/10 has to
        be making once it holds. A name returned to `ABSENT` fails the first
        assertion; a document still calling a gap open while `ABSENT` is empty
        fails the second.
        """
        assert self.ABSENT == (), self.ABSENT
        plain = _plain(self._item6())
        for still_open in ("is absent entirely", "has nothing in the "
                           "methodology describing", "one gap", "one left",
                           "Not satisfied at this commit"):
            assert still_open not in plain, (
                f"docs/10 §4 item 6 still says {still_open!r} while this "
                f"guard holds that every name it listed is now in docs/05")
        # The paired positive: the item has to say the thing it now means,
        # or the negatives above would pass against an item that had simply
        # lost its text.
        assert "now covered" in plain, plain[:200]

    def test_the_selectivity_range_is_the_one_the_source_measured(self):
        """§4 item 3 quotes three figures out of `harness/datasets.py`.

        The citation guard pins the anchor, so a change in the source fails
        there. This closes the other direction: a figure edited in docs/10
        alone. Parsed from both sides rather than written twice, because this
        repo's commonest defect is a number corrected in one of its two homes.
        """
        body = _symbol_text("harness/datasets.py", "FilterThreshold")
        source = re.search(
            r"nominal 1% delivered ([\d.]+)%-([\d.]+)% of the rows "
            r"\(median ([\d.]+)%\)", body)
        assert source, body
        said = re.search(
            r"nominal 1% delivering\s+between ([\d.]+)% and ([\d.]+)% of the "
            r"rows with a median of\s+([\d.]+)%", _plain(_doc()))
        assert said, "docs/10 §4 item 3 no longer states the delivered range"
        assert said.groups() == source.groups(), (said.groups(),
                                                  source.groups())

    def test_the_calibration_names_it_calls_absent_really_exist(self):
        """Otherwise the item would be complaining that docs/05 omits nothing.

        The three calibration statuses are real; what is missing is any
        description of them in the methodology.
        """
        from harness.workloads import calibrate
        assert calibrate.STATUS_OK == "ok"
        assert calibrate.STATUS_TARGET_UNREACHED == "target_unreached"
        assert calibrate.STATUS_FLOOR_ABOVE_TARGET == "floor_above_target"


# ---------------------------------------------------------------------------
# §5 item 2: the estimator's fallback
# ---------------------------------------------------------------------------

class TestTheEstimatorFallbackIsRecomputed:
    """§5 item 2's figures and its ratio, from the table itself."""

    def test_an_unmeasured_engine_is_named_rather_than_modelled(self):
        """Called, not read off the literal.

        `_effective_rate` is what the estimate is built from, so what it
        returns for an unknown engine IS the behaviour this section describes.
        A test that only matched a number in the source would pass against a
        `.get` whose result had stopped reaching the estimate.

        **THE DEFECT THIS SECTION DESCRIBED IS FIXED, AND THIS TEST WAS
        INVERTED BY THE FIX.** It asserted `_effective_rate(unknown, …) == 300`
        and that `estimate.py` still contained `.get(engine, 300)` -- it was
        pinning a live defect in place, correctly, while it was live. The
        fallback now returns `None` and the engine is named in
        `engines_without_ingest_rate`, so the assertions are the mirror
        image: no number, and a name in the gap list.

        **And Percona has stopped being the example, which is the second
        thing this test records.** It demonstrated the gap list with a real
        engine that had no rate; that engine's rate was measured into
        `_MEASURED_ROWS_PER_S` and it is counted now, so the demonstration
        moved to `no-such-engine`, which was already asserted three lines
        above for exactly this reason. The section's point is unchanged -- an
        unmeasured engine is named rather than modelled -- and is now carried
        by an engine nothing could ever have measured, so it cannot expire
        again the next time a rate is written down.

        The historical figure is still checked against the document, because
        §5 item 2 quotes it to make the argument -- it just no longer has to
        be what the code does.
        """
        from orchestrator import estimate
        claimed = re.search(r"modelled at (\d+) rows/s", _doc())
        assert claimed, "docs/10 no longer states what the old fallback was"
        assert float(claimed.group(1)) == 300.0

        assert "no-such-engine" not in estimate._INGEST_ROWS_PER_S
        # No number at all, which is the repair.
        assert estimate._effective_rate("no-such-engine",
                                        "no-such-dataset") is None
        # And a measured engine still gets one, so this is a gap report and
        # not a function that has stopped working.
        assert estimate._effective_rate("alisql", "no-such-dataset") > 0
        # The defect's spelling must not come back as CODE. Matched with the
        # assignment, because both `estimate.py` and this page quote the old
        # form in prose to explain the repair -- a bare substring check fired
        # on that explanation, which is the inverse vacuity: a negative
        # assertion failing on text that agrees with it.
        source = open(os.path.join(VB_ROOT, "orchestrator",
                                   "estimate.py")).read()
        assert "base = _INGEST_ROWS_PER_S.get(engine, 300)" not in source
        assert "base = _INGEST_ROWS_PER_S.get(engine)" in source
        # An engine with no rate is named, not modelled, in the estimate the
        # section is about -- and an engine WITH one is counted, which is the
        # half that proves the list is derived and not a constant.
        e = estimate.estimate_load_hours(
            {"name": "t", "ann": {"m_values": [16]}, "ops": {"m_values": [16]}},
            ["mariadb", "percona", "no-such-engine"], ["glove-100-angular"],
            ["normalized"], ["ann", "ops"])
        assert e["engines_without_ingest_rate"] == ["no-such-engine"]
        assert "no-such-engine" not in e["per_engine_hours"]
        assert e["per_engine_hours"]["percona"] > 0

    def test_the_mysql_family_rates_are_the_ones_the_document_prints(self):
        from orchestrator import estimate
        for engine, printed in (("mariadb", 150), ("mariadb123", 110),
                                ("alisql", 55)):
            assert estimate._INGEST_ROWS_PER_S[engine] == printed, engine
        _claim("the measured MySQL-family ingest rates")

    def test_the_ratio_the_document_quotes_is_the_one_the_table_gives(self):
        """The contradiction §5 item 2 rests on, recomputed. A ratio restated
        by hand is exactly the figure that goes stale when a rate is
        re-measured, and the comment above the table says rates get
        re-measured."""
        from orchestrator import estimate
        fallback = estimate._INGEST_ROWS_PER_S.get("absent", 300)
        slowest = min(estimate._INGEST_ROWS_PER_S[e]
                      for e in ("mariadb", "mariadb123", "alisql"))
        claimed = re.search(r"\*\*([\d.]+)× optimistic", _doc())
        assert claimed, "docs/10 no longer prints the ratio"
        assert round(fallback / slowest, 2) == float(claimed.group(1))

    def test_the_comment_still_sets_the_rule_the_fallback_breaks(self):
        """Without this sentence §5 item 2 is a preference, not a
        contradiction."""
        body = _symbol_text("orchestrator/estimate.py",
                            "_INGEST_ROWS_PER_S")
        assert "Measured ingest rates" in body
        assert "by 3x is worse than none" in body
        assert "3x" in _doc()


class TestTheDecisionD1IsDecidedAndItsCostIsRecomputed:
    """§5 item 3 says D1 is settled and what it cost.

    The earlier revision of this class asserted the grids did NOT meet, and
    docs/10 removed the hour figures as unsourced because at that commit
    nothing in the repo supported them. Both have reversed. The figures are now
    recomputed from `orchestrator.estimate` rather than read off the comment
    that states them, so this closes the loop the other way: a profile change
    moves them here, and a figure edited in docs/10 alone fails.
    """

    #: `main.yml`'s ann grid before D1, kept so the "before" column of §5 item
    #: 3's table is a computation rather than a memory.
    GRID_BEFORE = [8, 32]
    #: The three engines README and docs/07 call the original comparison, which
    #: is the engine set §5 item 3's second row is over. Named rather than
    #: sliced off `KNOWN_ENGINES`, because "the first three" is not a fact.
    HEADLINE = ["mariadb", "alisql", "pgvector"]

    @staticmethod
    def _hours(profile, engines):
        from orchestrator.estimate import estimate_load_hours
        return estimate_load_hours(
            profile, list(engines), list(profile.get("datasets", [])),
            ["normalized"], ["ann", "ops"])["total_hours"]

    def test_the_ann_and_ops_m_grids_meet(self):
        main = _profile("main.yml")
        ann_m, ops_m = main["ann"]["m_values"], main["ops"]["m_values"]
        shared = set(ann_m) & set(ops_m)
        assert shared, (
            f"ann {ann_m} and ops {ops_m} no longer intersect; docs/10 §5 "
            f"item 3 and §4 item 5 both depend on their meeting")
        doc = _doc()
        assert f"`m_values: {ann_m}`" in doc, ann_m
        assert f"`m_values` of `{ops_m}`" in doc, ops_m

    def test_there_is_still_no_crosscheck_profile(self):
        """The option that was NOT taken, and docs/10 says why it was not."""
        names = os.listdir(os.path.join(VB_ROOT, "config", "profiles"))
        assert not [n for n in names if "crosscheck" in n], names
        _claim("there is no crosscheck profile")

    def _doc_cost_table(self):
        """§5 item 3's table, as {label: (before, after, added)}."""
        import copy
        text = _doc().split("### 3. The cost envelope", 1)[1]
        text = text.split("\n## ", 1)[0]
        out = {}
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 4:
                continue
            nums = [re.sub(r"[^\d.]", "", c) for c in cells[1:]]
            if not all(re.fullmatch(r"\d+\.\d\d", n) for n in nums):
                continue
            out[cells[0]] = tuple(float(n) for n in nums)
        assert out, "could not parse §5 item 3's cost table"
        return out, copy.deepcopy(_profile("main.yml"))

    def test_the_cost_table_was_found_and_has_both_rows(self):
        """Non-vacuity: a parser that matched nothing would make the two
        assertions below pass over an empty table."""
        rows, _ = self._doc_cost_table()
        assert len(rows) == 2, sorted(rows)
        # "seven", not "six": the first row was labelled for the six engines
        # with a measured ingest rate while it was computed over all seven and
        # Percona fell out of it. Percona's rate is measured, so the row
        # prices every engine and says so -- and this assertion is what makes
        # the label and the engine list below it move together.
        assert any("seven" in label for label in rows), sorted(rows)
        assert not any("six" in label for label in rows), sorted(rows)
        assert any("three" in label for label in rows), sorted(rows)

    def test_every_figure_in_it_comes_from_the_estimator(self):
        from orchestrator.cli import KNOWN_ENGINES
        rows, now = self._doc_cost_table()
        before = {**now, "ann": {**now["ann"],
                                 "m_values": list(self.GRID_BEFORE)}}
        for label, (said_before, said_after, said_added) in rows.items():
            engines = (list(KNOWN_ENGINES) if "seven" in label
                       else list(self.HEADLINE))
            got_before = self._hours(before, engines)
            got_after = self._hours(now, engines)
            assert got_before == said_before, (label, got_before, said_before)
            assert got_after == said_after, (label, got_after, said_after)
            assert round(got_after - got_before, 2) == said_added, (
                label, round(got_after - got_before, 2), said_added)

    def test_the_comment_that_states_them_agrees_with_the_estimator(self):
        """`main.yml:84-87` quotes the two added figures in prose. Checked
        against the same computation, because a comment and a document that
        disagree about one number is this repo's commonest defect -- and here
        the document is derived while the comment is written by hand."""
        from orchestrator.cli import KNOWN_ENGINES
        comment = _symbol_text("config/profiles/main.yml", "m_values")
        said = re.findall(r"\+(\d+\.\d\d) h", comment)
        assert len(said) >= 2, comment
        now = _profile("main.yml")
        before = {**now, "ann": {**now["ann"],
                                 "m_values": list(self.GRID_BEFORE)}}
        added = {
            "headline": round(self._hours(now, self.HEADLINE)
                              - self._hours(before, self.HEADLINE), 2),
            "all": round(self._hours(now, list(KNOWN_ENGINES))
                         - self._hours(before, list(KNOWN_ENGINES)), 2),
        }
        assert float(said[0]) == added["headline"], (said, added)
        assert float(said[1]) == added["all"], (said, added)

    def test_the_readme_figure_is_a_different_basis_and_docs10_says_so(self):
        """§5 item 3 warns that README's ~61 h and the table above are not the
        same quantity. The warning is only worth keeping while they differ."""
        from orchestrator.cli import KNOWN_ENGINES
        del KNOWN_ENGINES
        readme = open(os.path.join(VB_ROOT, "README.md")).read()
        row = re.search(r"\|\s*\*\*`main`\*\*\s*\|[^|]*\|\s*\*\*~(\d+) h\*\*",
                        readme)
        assert row, "README's main row no longer prints an hours figure"
        estimator = self._hours(_profile("main.yml"), self.HEADLINE)
        assert abs(float(row.group(1)) - estimator) > 1.0, (
            "README's figure and the estimator's now agree; docs/10 §5 item 3 "
            "warns that they are different bases and should stop")
        _claim("the README figure and the estimator are different bases")


# ---------------------------------------------------------------------------
# §5 item 1: docs/09's numbers must not be copied here
# ---------------------------------------------------------------------------

class TestTheMemoryTableIsCitedAndNotRestated:
    """The defect this repo has hit most often is a figure in two places.

    docs/09 §3 carries its own guard that recomputes every row of its table from
    the formula it cites. A copy here would be a second thing to correct, so
    docs/10 cites the section and states the conclusion in words. This test is
    what keeps that true.
    """

    @staticmethod
    def _table_numbers():
        """The §3 table's floors and ratios, straight out of docs/09."""
        doc = open(PERCONA_DOC).read()
        section = doc.split("## 3. `innodb_hnsw_max_memory`", 1)[1]
        out = set()
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 7 or not re.fullmatch(r"[\d,]+|per 1M", cells[1]):
                continue
            out.add(cells[4])
            out.add(re.sub(r"[^\d.]", "", cells[5]))
        return out

    def test_the_numbers_were_found(self):
        """Non-vacuity, and the shape the deletion mutant needs: a parser that
        matched nothing would make the next assertion trivially true."""
        numbers = self._table_numbers()
        assert len(numbers) >= 8, sorted(numbers)
        percona = open(PERCONA_DOC).read()
        for number in numbers:
            assert number in percona, number

    def test_none_of_them_appears_in_the_rollout_document(self):
        rollout = _doc()
        copied = sorted(n for n in self._table_numbers() if n in rollout)
        assert not copied, (
            "docs/10 restates docs/09 §3's figures instead of citing them, "
            "which is the second-copy defect §5 item 1 is written to avoid: "
            + ", ".join(copied))

    def test_it_cites_the_section_and_states_the_conclusion(self):
        """The paired positive: not restating is only right if the document
        still tells the operator where to look and what it means."""
        flat = re.sub(r"\s+", " ", _doc())
        assert ("`docs/09-adding-percona-server.md` §3's *What the default "
                "costs, with the inputs named*" in flat)
        _claim("a green smoke is not evidence about the memory policy")


# ---------------------------------------------------------------------------
# Figures the next merge would falsify
# ---------------------------------------------------------------------------

_TEST_COUNT_RE = re.compile(r"\b\d[\d,]*\s+tests\b")


class TestNoFigureTheNextMergeWouldFalsify:
    """A count of the test suite is wrong the moment anyone adds a test.

    This repository has shipped a stale one three times. The rule docs/10
    follows is to cite a test rather than count them, and this is what enforces
    it.
    """

    def test_the_document_states_no_test_count(self):
        found = _TEST_COUNT_RE.findall(_doc())
        assert not found, (
            "docs/10 states a test count, which the next merge falsifies: "
            + ", ".join(found))

    def test_the_pattern_would_find_one(self):
        """Non-vacuity: a regex that matched nothing would make the assertion
        above pass against a document full of counts."""
        assert _TEST_COUNT_RE.findall("verified only by 1,884 tests against")
        assert _TEST_COUNT_RE.findall("2057 tests collected")
        assert not _TEST_COUNT_RE.findall("tests/test_e3_rollout_doc.py")

    def test_the_document_cites_a_test_file_instead(self):
        assert "tests/test_e3_rollout_doc.py" in _doc()
        assert "tests/verify-alisql-traps.sh" in _doc()


class TestThePullRequestMappingMatchesTheHistory:
    """§1's mapping table, checked against the merge commits.

    The spec numbers its pull requests 0 to 5 and the repository numbered its
    own 1 to 15, and the draft of this document conflated them. The table is
    the only place a reader can recover which is which, so it is checked rather
    than trusted.
    """

    #: §1's mapping table, as (repository pull request number, branch). Also
    #: the definition of the span the gap assertion below runs over, which is
    #: why it is a constant here rather than a literal in one decorator --
    #: see `test_the_gap_in_the_numbering_is_the_one_the_document_names`.
    MAPPING_ROWS = [
        (2, "pr-0-fake-driver"),
        (3, "pr-1-concurrency"),
        (5, "pr-3-filter"),
        (7, "t3b-drivers"),
        (8, "t3c-filtered"),
        (9, "t4a-churn"),
        (12, "t4b-readers"),
        (16, "t5b-stats"),
    ]

    #: The last merge of the repository whose numbering §1 describes:
    #: EvgeniyPatlan/vector-bench's #74, the tip Percona-Lab/vector-bench was
    #: fast-forwarded to on 2026-11-03 when it became the upstream. Percona-Lab
    #: numbers its own pull requests from #1 again, and reading HEAD put this
    #: class red the moment its #4 merged: the gap §1 names was "filled" by a
    #: pull request from a different repository. Every number §1 talks about
    #: is reachable from here, and nothing merged later can change that set.
    OLD_NUMBERING_TIP = "0ddf29a"

    @pytest.fixture(scope="class")
    def merges(self):
        found = _merges_at(self.OLD_NUMBERING_TIP)
        if not found:
            pytest.skip("no merge commits in this checkout")
        return found

    def test_the_history_was_read(self, merges):
        """Non-vacuity: an empty mapping would make the rows below pass."""
        assert len(merges) >= 10, sorted(merges)

    @pytest.mark.parametrize("number,branch", MAPPING_ROWS)
    def test_the_row_names_a_merge_that_happened(self, merges, number,
                                                 branch):
        assert number in merges, (
            f"docs/10 §1 says #{number} landed {branch}; the history has no "
            f"such merge")
        assert branch in merges[number], (merges[number], branch)
        # The row has to name both, on one line; the separator between them is
        # the document's business.
        rows = [line for line in _doc().splitlines()
                if f"#{number}" in line and branch in line]
        assert rows, (
            f"the history has #{number} landing {branch}, and no line in "
            f"docs/10 names both")

    def test_the_gap_in_the_numbering_is_the_one_the_document_names(self,
                                                                    merges):
        """§1 says #4 is absent from the history. Derived, not asserted: the
        reader is being told the two numbering schemes do not line up, and a
        gap the document invented would undercut exactly that.

        **The highest merged number is deliberately no longer pinned, and
        that is the fix rather than an omission.** This test used to assert
        `highest == 16` and that §1 said "up to #16". Both went stale the
        next time anything merged -- which was this document's own pull
        request, within the hour -- and the gate failed on
        `AssertionError: 17` for a figure no reader acts on. It is the same
        class as the spliced line counts `tests/test_t4b_readers.py` stopped
        restating: a monotonically increasing fact cannot be written down and
        stay true.

        **THE GAP SET IS NOW COMPUTED OVER A CLOSED SPAN, AND THAT IS THE
        SECOND HALF OF THE SAME FIX.** The pin that replaced `highest == 16`
        was `gaps == [4]` over the whole log, justified in this docstring and
        in docs/10 §1 by the claim that "history does not acquire new gaps".
        **That claim is false, and it put `master` red on 2026-10-21.** #47
        merged at 10:35 while #46 was still open, the whole-log computation
        returned `[4, 46]`, and the gate failed on a document that was
        correct; #46 merged at 16:17 and the gate went green again on its
        own. A guard that fails on correct prose and then repairs itself when
        an unrelated branch lands is the failure mode this file already
        names twice -- it is not measuring its subject, and it teaches the
        next author to delete it.

        History acquires a gap whenever a higher-numbered pull request merges
        before a lower-numbered one, and on this repository's cadence that is
        routine rather than exceptional: #12 landed before #11, #30 before
        #27, #47 before #46. Every one of those opened a gap that closed when
        the lower number merged.

        So the span is closed instead. **#1 to #16 is settled history**: the
        span §1's mapping table covers, every number in it long since merged
        or not, no gap inside it able to open or close again. The ceiling is
        read off `MAPPING_ROWS` rather than written down, so it moves with the
        table it is the span of. Over that span the gap set is frozen, #4 is
        the whole of it, and the assertion can never be falsified by a merge.

        **The span is closed because the table describes SETTLED history, not
        merely because it is a table.** `era` inherits that property and
        nothing else: a row added for a number above a pull request still in
        flight would re-open the original defect verbatim. Two things keep
        that narrow rather than theoretical.
        `test_the_row_names_a_merge_that_happened` refuses a row docs/10 does
        not name on one line with its branch, and
        `test_the_open_span_form_really_did_fail_and_the_closed_one_does_not`
        fails outright for any `era >= 46`, because #46 is unmerged at the
        commit it replays. The table is the spec series, which closed at #16;
        if a row is ever added for a later pull request, read this paragraph
        before trusting the span.

        **The limit this accepts, stated rather than hidden.** A permanent gap
        ABOVE the span -- a pull request number that is closed-unmerged and
        never will merge -- is not detected. It cannot be, from inside a unit
        test: git records that #4 never merged and that #46 had not merged
        yet, and nothing in the object store tells those two apart. Answering
        it needs GitHub, and a guard that reaches the network is a guard that
        fails on a train. What is kept instead is the part that does not need
        the network: `4 in gaps` over the whole log, below, and the document
        naming exactly one gap and no other.
        """
        era = max(number for number, _ in self.MAPPING_ROWS)
        # Non-vacuity for the span itself: an empty or mis-parsed MAPPING_ROWS
        # would make `range(1, era + 1)` a handful of numbers and the gap set
        # trivially right.
        assert era >= 16, era
        gaps = sorted(set(range(1, era + 1)) - set(merges))
        assert gaps == [4], (
            f"over #1-#{era} -- the span docs/10 §1's mapping table covers, "
            f"and closed history -- the gaps in the merge numbering are "
            f"{gaps}, not [4]. A gap here is permanent: every number in this "
            f"span is resolved. Either a merge subject stopped parsing or §1 "
            f"is describing a history this repository does not have.")
        # A FLOOR, not a value. The quantity only increases, so a lower bound
        # never goes stale, and it is here only to keep the gap computation
        # above from being satisfied by a mis-parsed handful of merges.
        # `test_the_history_was_read` is the other half of that.
        assert max(merges) >= era, max(merges)
        # The one fact about the WHOLE log that is durable: #4 never merged,
        # and a number once skipped is skipped forever. The complement -- that
        # no OTHER gap is permanent -- is the part that needs the network, and
        # is not asserted.
        whole = sorted(set(range(1, max(merges) + 1)) - set(merges))
        assert 4 in whole, whole
        # §1 must name that gap, and must name no other. This is the half that
        # still watches the whole document: an invented gap anywhere in it
        # fails here even though the gap COMPUTATION stops at #16.
        named = sorted(_numbers_called_absent(_doc()))
        assert named == [4], (
            f"docs/10 §1 names {named} as absent from the merge numbering; "
            f"the gap over #1-#{era} is [4]")
        # And §1 must not reintroduce the figure that rotted. Checked as a
        # pattern rather than against today's number, because the defect is
        # any such claim and not one particular spelling of it.
        stale = re.search(r"(up to|as high as|through)\s+#\d+", _plain(_doc()))
        assert stale is None, (
            f"docs/10 §1 names the highest merged pull request again "
            f"({stale.group(0)!r}). That figure goes stale on the next "
            f"merge; the mapping table carries the numbers instead.")

    def test_the_open_span_form_really_did_fail_and_the_closed_one_does_not(
            self):
        """The commit where the old pin went red, replayed against both forms.

        Not a hypothetical. `a695c6d` is #47's merge commit -- the tree as it
        stood between 10:35 and 16:17 on 2026-10-21, with #47 merged and #46
        still open. Replaying the two computations there is what makes "the
        closed span is immune to the event that broke the open one" an
        executable statement rather than a paragraph of reasoning.

        Pinned to a sha, so it is frozen: the merge set reachable from
        `a695c6d` cannot change, which is the same property the closed span
        relies on one level up.

        **THIS TEST, AND NOT THE ONE ABOVE, IS WHAT CARRIES THE FIX.** A
        review measured it: reverting the test above to the open span passes
        at HEAD, because #46 has since landed and the whole-log gap set is
        `[4]` again. The regression it was written for is undetectable from
        today's tree. That division of labour is deliberate and worth stating
        plainly -- the closed span prevents recurrence and can never again
        demonstrate the defect, so the demonstration has to live at a commit
        where the defect was visible. Delete this test and the change becomes
        a claim in a docstring.
        """
        merges = _merges_at("a695c6d")
        # Non-vacuity: a parse that found nothing would make both gap sets
        # whatever `range(1, 1)` is and both assertions below meaningless.
        assert len(merges) >= 10, sorted(merges)
        assert 47 in merges and 46 not in merges, sorted(merges)

        open_span = sorted(set(range(1, max(merges) + 1)) - set(merges))
        assert open_span == [4, 46], open_span

        era = max(number for number, _ in self.MAPPING_ROWS)
        closed_span = sorted(set(range(1, era + 1)) - set(merges))
        assert closed_span == [4], closed_span

    def test_what_the_document_says_about_16_matches_the_tree(self):
        """Whether §1 calls #16 open has to match whether #16 is merged.

        Two-way, so it never fires spuriously: whichever side the tree is on,
        the document has to be on the same one.

        **It asks about #16's own merge commit now, and it used to ask about
        HEAD.** `HEAD in origin/master` is only a proxy for "#16 is merged"
        while HEAD is #16's branch. It stopped being one as soon as this
        document was written on a later branch: #16 was already merged, HEAD
        was not, so the test demanded §1 call #16 open and failed at its own
        base. It then self-resolved when this document merged, which is worse
        than failing -- a guard that goes green because an unrelated branch
        landed was never measuring its subject.

        Searching `origin/master`'s own merge subjects answers the question
        directly and does not depend on where HEAD sits.
        """
        try:
            out = subprocess.run(
                ["git", "log", "--merges", "--format=%s", "origin/master"],
                cwd=VB_ROOT, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            pytest.skip(f"git unavailable: {exc}")
        if out.returncode != 0 or not out.stdout.strip():
            pytest.skip("origin/master not available in this checkout")
        subjects = out.stdout.splitlines()
        # Non-vacuity: a parse that found no merges at all would make the
        # `merged = False` branch below assert the opposite of the truth.
        assert any(re.match(r"Merge pull request #\d+ ", s) for s in subjects), (
            "no merge subjects parsed from origin/master")
        merged = any(re.match(r"Merge pull request #16 ", s) for s in subjects)
        plain = _plain(_doc())
        says_open = bool(re.search(r"#16 is (the )?open", plain))
        if merged:
            assert not says_open, (
                "#16 is merged into origin/master, so docs/10 §1 must stop "
                "calling it open and §2 must stop waiting on it")
        else:
            assert says_open, (
                "#16 is not in origin/master; docs/10 §1 should say it is "
                "still open, because §2's order waits on the merge")

    def test_the_absence_vocabulary_accepts_a_legitimate_rewording(self):
        """The half that keeps this guard usable. A check keyed to one
        sentence reddens `master` on a correct rewrite, and the author who
        pays for that twice deletes it."""
        for wording, expected in ABSENCE_ACCEPTS:
            assert sorted(_numbers_called_absent(wording)) == expected, wording

    def test_the_absence_vocabulary_refuses_the_wrong_polarity(self):
        """Non-vacuity: a pattern loose enough to accept its own negation
        would report #4 as named by a document that said the opposite."""
        for wording in ABSENCE_REFUSES:
            assert _numbers_called_absent(wording) == set(), wording

    def test_t5a_landed_and_t5c_did_not(self):
        """The distinction §1 draws inside PR 5, which is what unblocks the
        order in §2 being about T5b alone.

        Deliberately does not take the `merges` fixture: it is answered by the
        filesystem, and taking the fixture made it skip wherever git does not
        answer -- which is every exported tree, and which is how a mutation run
        found it passing and failing identically.
        """
        assert os.path.isfile(
            os.path.join(VB_ROOT, "harness", "metrics", "stats.py"))
        assert os.path.isfile(os.path.join(VB_ROOT, "tests", "test_stats.py"))
        assert not os.path.exists(
            os.path.join(VB_ROOT, "harness", "workloads", "rate.py")), (
            "harness/workloads/rate.py now exists: T5c has landed and "
            "docs/10 §1 is stale")
        _claim("T5a landed and T5c did not")
