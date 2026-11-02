"""Engine driver interface for the ops harness.

The ops harness measures what ann-benchmarks does not: index build cost,
concurrency scaling, filtered search and churn. Those need operations
ann-benchmarks' BaseANN has no concept of (delete rows, size an index, open N
independent connections), so they get their own interface.

A driver talks to a server that is already running in its own container. The
harness never starts or configures the server — the orchestrator does, so that
resource limits and server flags are applied identically for every engine and
are recorded in the manifest.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy

from ..datasets import FILTER_COLUMNS, FILTER_CTAG, FILTER_TAG

TABLE = "t1"
DATABASE = "ann"


@dataclass
class ConnectionSpec:
    host: str
    port: int
    user: str = "bench"
    password: str = "bench"
    database: str = DATABASE
    # Path where this process can read the server's data directory, when the
    # orchestrator has shared it read-only. Enables exact on-disk index sizing
    # for engines whose catalogs cannot report it.
    data_dir: Optional[str] = None


@dataclass
class IndexSpec:
    dim: int
    m: int
    metric: str                       # "angular" | "euclidean"
    ef_construction: Optional[int] = None
    storage_engine: str = "InnoDB"
    build_mode: str = "post"          # "post" | "incremental"
    # Percona Search is the only engine here that compresses indexed vectors.
    # "none" | "scalar" | "binary"; pinned off in the normalized pass so it is
    # not an axis the others lack, and set to the vendor's recommendation in
    # the tuned pass, where following each vendor's own advice is the point.
    quantization: Optional[str] = None


@dataclass
class LoadResult:
    rows: int
    wall_seconds: float
    threads: int

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.wall_seconds if self.wall_seconds > 0 else 0.0


@dataclass
class RowAttrs:
    """The scalar columns a row carries, as one object.

    Filtered search is measured twice over the same index: once on `tag`,
    whose values are independent of the vector space, and once on `ctag`,
    whose values are nearest-pivot cluster ids and therefore follow it
    (`harness/datasets.py`). Loading both columns in one build is PR 3's
    option A. Option B -- one column whose values are chosen per run -- needs
    no driver change at all, and was rejected because it needs a full reload
    per filter mode: MHNSW and VIDX maintain their graph on every INSERT at
    ~150 and ~55 rows/s, so a second mode at 1M rows is ~1.9 h and ~5 h of
    ingest per engine, per M value.

    Both columns travel in one object rather than as two parameters because
    `load` (the bulk ingest) and `insert_rows` (churn's re-insert) have to
    agree about how many columns an INSERT names, and they are in different
    files. Two parameters is two places to forget one, and forgetting is not
    equally loud everywhere: an INSERT whose column list and value tuple
    disagree in length fails outright on the SQL engines, while a Valkey HSET
    that omits a field writes a hash successfully -- the write returns OK --
    and whether the module then counts that as an indexing failure or simply
    indexes a document no `@ctag:[-inf (t]` can match, the first thing anyone
    sees is a filtered recall of zero.

    Not frozen, unlike `_Rows` in `tests/fake_driver.py` but like
    `ConnectionSpec`, `IndexSpec` and `LoadResult` above it: this is a
    parameter object handed straight to a cursor, not shared mutable state
    that several threads read.
    """

    tag: numpy.ndarray
    ctag: numpy.ndarray

    def __post_init__(self) -> None:
        """Both columns, one value per row, or nothing.

        The invariant the whole structure exists for. Every write path below
        reads the two arrays at the same index -- `row(i)`, `fields(i)` -- so
        a pair of different lengths does not fail where it was built, it fails
        at whichever row runs out first, or does not fail at all and pairs one
        row's vector with another row's cluster id. Churn slices this, and
        Valkey slices those slices again, so there are three chances to get a
        bound wrong before anything is written.
        """
        if len(self.tag) != len(self.ctag):
            raise ValueError(
                f"a RowAttrs holds one value per column per row; got "
                f"{len(self.tag)} {FILTER_TAG} and {len(self.ctag)} "
                f"{FILTER_CTAG} values")

    @classmethod
    def coerce(cls, attrs: Union["RowAttrs", Sequence[int], numpy.ndarray]
               ) -> "RowAttrs":
        """A `RowAttrs` from either a `RowAttrs` or a bare array of tag values.

        The bare-array form is not deprecation politeness. Counted by AST,
        there are **13 such call sites across 6 files**, and the ones that
        decide the question are the ones PR 3 may not edit:
        `tests/test_workloads.py:1060` and `tests/test_calibrate.py:668` call
        `load()` with one array, and `tests/test_config.py` drives
        `ValkeyDriver.insert_rows` with a plain list of ints at six sites to
        exercise the write-batch ladder. All three files are off limits to
        this task, so a required `RowAttrs` is not a migration, it is a
        broken suite. The rest -- `scripts/probe-valkey-filter.py:151`,
        `scripts/verify-valkey-churn.py:197` and `:245`, and the helpers at
        the bottom of `tests/fake_driver.py` -- are diagnostics with no cache
        directory to compute a cluster assignment from, which is the reason
        they have no `ctag` to give rather than merely no habit of giving one.

        `ctag` is a `copy()` of the array, not the same object. Nothing today
        writes through either column, so sharing one buffer was harmless and
        measurably free -- and it put two names on one array in a repo whose
        stated default is immutability, which is a trap set for whoever adds
        the first in-place write. The copy costs one pass over an int64 array
        per call, against a load that is doing one INSERT per row.

        A bare array becomes `ctag = tag`. That is a named conversion rather
        than a default argument precisely so the compromise is visible at the
        call site. Zeros were the alternative and are worse: `ctag INT NOT
        NULL` is satisfied either way, but a column of zeros makes `ctag < t`
        true for every row at any t > 0, so a filtered measurement over it
        would deliver the whole corpus while the record said 1%. Mirroring
        keeps the delivered share exactly what the threshold asks for, so the
        only thing wrong is the correlation -- and a run where `ctag` and
        `tag` agree shows up as the two filter modes producing the same
        numbers, which is the one result PR 3 exists to show they do not.
        """
        if isinstance(attrs, cls):
            return attrs
        values = numpy.asarray(attrs)
        return cls(tag=values, ctag=values.copy())

    def __len__(self) -> int:
        return int(len(self.tag))

    def __getitem__(self, index) -> "RowAttrs":
        """Slice both columns together, so a sliced row keeps its own values.

        Churn re-inserts in slices and Valkey re-inserts in slices of those,
        and slicing the columns independently is how a row ends up with
        another row's cluster id.
        """
        return RowAttrs(tag=self.tag[index], ctag=self.ctag[index])

    def column(self, name: str) -> numpy.ndarray:
        """One column's values, by whitelisted name.

        The field names here are `FILTER_COLUMNS`' own strings, which is what
        lets this be a lookup rather than a branch -- and is the coupling to
        know about if a third filter column is ever added: it needs a field on
        this dataclass spelled exactly as the dataset module spells it, and
        `checked_filter_column` would otherwise accept a name that
        `getattr` cannot find.
        """
        return getattr(self, checked_filter_column(name))

    def row(self, i: int) -> Tuple[int, ...]:
        """Row `i`'s scalar values, in `FILTER_COLUMNS` order.

        For the engines whose INSERT or COPY takes a positional tuple. The
        order is the dataset module's, which is also the order the column
        lists in those statements are written in; `tests/test_t3b_drivers.py`
        pins the two against each other rather than trusting the eye.
        """
        return tuple(int(self.column(name)[i]) for name in FILTER_COLUMNS)

    def fields(self, i: int) -> Dict[str, int]:
        """Row `i`'s scalar values keyed by column name.

        For the two document/hash engines, where the field names are part of
        the write. Valkey writes the same mapping from three separate places
        (bulk load, churn re-insert, and the stall probe); building it here
        means a column cannot be present in one of the three and missing from
        another, which is the shape of the bug the three sites invite.
        """
        return {name: int(self.column(name)[i]) for name in FILTER_COLUMNS}


def checked_filter_column(column: Any) -> str:
    """The filter column's own name, or `ValueError`. Never interpolation.

    Every predicate in every driver is built by formatting a column name into
    a query string -- `WHERE {column} < %s`, `@{column}:[-inf (t]`, a BSON
    key. A column name that reached one of those from outside would be
    injection in four dialects at once, so the name is matched against
    `datasets.FILTER_COLUMNS` and what is returned is **the matching element
    of that tuple**, not the caller's string. The predicate is therefore
    always built from a module constant, even if the caller handed in a `str`
    subclass whose `__eq__` lies or whose `__str__` differs from its value.

    Deriving the whitelist from `datasets.FILTER_COLUMNS` rather than keeping
    a tuple here is the point: a second literal list of legal columns is a
    second thing to update, and the failure when the two disagree is a column
    the dataset module will happily compute ground truth for and the driver
    refuses to query -- or worse, the reverse.
    """
    for allowed in FILTER_COLUMNS:
        if column == allowed:
            return allowed
    raise ValueError(
        f"{column!r} is not a filter column; "
        f"expected one of {list(FILTER_COLUMNS)}"
    )


class EngineDriver(abc.ABC):
    """One connection-owning driver instance per client."""

    name: str = "unknown"
    #: True when the engine maintains its graph during INSERT and therefore has
    #: no separable bulk build step. Drives how build cost is attributed.
    incremental_index: bool = True
    #: True when the index is built by another process after the statement
    #: returns, so build cost is load time plus time-to-ready and the two
    #: overlap. Neither value of incremental_index describes that.
    async_index_build: bool = False
    #: Short name of the repair path `maintain()` runs, or "none" for an
    #: engine that has none. Declared rather than inferred from `maintain()`'s
    #: return value, because the churn workload has to stamp it onto the
    #: record it writes BEFORE maintenance runs -- that record is the
    #: "before maintenance" measurement, and writing it first is what keeps a
    #: completed measurement on disk when `maintain()` itself fails. A value
    #: derived from a call that has not happened yet would be "none" for
    #: pgvector too, which is the one thing PR 4 §4.1 item 4 asks this key to
    #: be able to say apart.
    #:
    #: A class attribute next to the two above because it is the same kind of
    #: fact: a property of the engine, not of a measurement. MariaDB, AliSQL,
    #: Valkey and Percona Search inherit "none" and need no edit, which is
    #: what "No hooks" in §4.1 item 4 means.
    repair_path: str = "none"
    #: What the bytes `index_bytes()` counts hold of the vectors (ops
    #: measurement version 9): `float32`, `int16`, `int8`, `none` (a graph
    #: of ids only) or `unknown`, and whether the index carries its own copy
    #: of them at all (None = not verified). Copied onto every record that
    #: carries `index_bytes`, because a size comparison that does not say
    #: this compares storage formats. Set ONLY from the engine's source as
    #: pinned: MHNSW and VIDX store `int16_t dims[]` per graph node
    #: (`FVector` in `sql/vector_mhnsw.cc` / `sql/vidx/vidx_hnsw.cc`),
    #: pgvector's HNSW element tuples carry the `vector` value. Every other
    #: engine's on-disk format is not in this checkout, so it says unknown.
    index_vector_format: str = "unknown"
    index_includes_vectors: Optional[bool] = None
    #: Server status variables worth recording, by name, or `()` for an engine
    #: that exposes none. Read today by one caller only,
    #: `harness/workloads/build.py:_server_status`, so the values land on the
    #: `index_build` record and no other phase's. Declared per DRIVER and not per call site,
    #: so a workload asks `driver.status_gauges` rather than carrying a table
    #: of engine names -- and so the answer to "what happens for the engines
    #: that have no such variable" is a value in this file rather than an
    #: implication: the empty tuple, which `harness/workloads/build.py` turns
    #: into no statement and no record key at all.
    #:
    #: `()` here is therefore load-bearing twice over. It is what six of the
    #: seven engines use, and it is what makes their records byte-identical to
    #: the ones written before any of this existed.
    #:
    #: Only `PerconaDriver` overrides it. Its one gauge,
    #: `Innodb_hnsw_memory_used`, is the only engine-reported live graph
    #: memory figure in the study -- see that override for the caveat that
    #: travels with every figure quoted from it.
    status_gauges: Tuple[str, ...] = ()
    #: Workloads this engine cannot run honestly, as `((workload, reason),
    #: ...)`, or `()` for an engine that runs all of them.
    #:
    #: A declaration and not an exception raised mid-workload, for the reason
    #: `repair_path` is one: `harness/main.py` has to know BEFORE the first
    #: record is written, so it can stamp `extra.declined_workloads` onto every
    #: record of the run and skip the workload with the reason printed, rather
    #: than fail the whole ops unit after build and concurrency have already
    #: been paid for. The alternative a driver has without this -- answering
    #: the filtered queries some other way -- publishes a number under a label
    #: it does not deserve, which is the failure this repository is built
    #: against.
    #:
    #: A tuple of pairs rather than a dict because it is a class attribute: a
    #: mutable default here would be one object shared by every driver class.
    #: Only `MyVectorDriver` declares anything (filtered search, which its
    #: pinned release does not have).
    declined_workloads: Tuple[Tuple[str, str], ...] = ()

    def __init__(self, spec: ConnectionSpec):
        self.spec = spec
        self._conn: Any = None

    def query_path_facts(self, index: IndexSpec) -> Dict[str, Any]:
        """Structural facts about this engine's query path, for every record.

        `{}` for every engine whose query is one statement on a vectorised
        distance kernel, which is all of them but one. `harness/main.py`
        stamps whatever this returns onto every record of the run, and
        `report/disclosures.py` renders a footnote from it -- CLAUDE.md
        requires a structural asymmetry to be stated in the report, and a fact
        that lived only in a driver's docstring would not be.

        Keys: `round_trips_per_query` (int) and `distance_impl` (a string
        starting `scalar` or `simd`).
        """
        return {}

    # -- lifecycle ------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Open a connection and apply session setup."""

    @abc.abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "EngineDriver":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- schema ---------------------------------------------------------

    @abc.abstractmethod
    def drop_schema(self) -> None: ...

    @abc.abstractmethod
    def create_schema(self, index: IndexSpec) -> None:
        """Create the table, and the index too if the engine builds it inline."""

    @abc.abstractmethod
    def create_index(self, index: IndexSpec) -> None:
        """Create the vector index. No-op where create_schema already did it."""

    def storage_engine_in_use(self, index: IndexSpec) -> Optional[str]:
        """The storage engine this driver's records should name, or None.

        `IndexSpec.storage_engine` is a MySQL-family `ENGINE=` clause and
        nothing else. `harness/main.py` defaults `--storage-engine` to
        `InnoDB` and `orchestrator/cli.py:_ops_storage_engines` passes
        `InnoDB` for every engine that is not MariaDB, so the field arrives
        populated whether or not the driver has anywhere to put it. Only the
        MySQL family does: `MySQLFamilyDriver._create_table_ddl` ends its
        CREATE TABLE with `) ENGINE={index.storage_engine}`, and every other
        driver here ignores the field completely.

        `RunContext.record_defaults` used to stamp the raw field, so the ops
        records claimed an InnoDB for PostgreSQL, Valkey and Percona Search --
        three engines that have no storage engines at all, let alone that one.
        Measured over the seven-engine run `full-20261015-215006`, by
        (engine, storage_engine):

            pgvector     InnoDB      268      pgvector     heap     200
            valkey       InnoDB      128      valkey       memory   100
            mongodb      InnoDB      120      mongodb      wiredTiger 37

        The honest halves of those three pairs are the `recall_qps` records,
        which the ann overlay modules write and which name what the engine
        actually stores in; every other phase is this harness, and it
        overwrote all three with the same wrong default. So one engine
        reported two different storage engines inside a single run, and the
        half a reader is most likely to trust -- the ops half, which is where
        build cost, concurrency, filtered search and churn live -- was the
        false one.

        Returning None is what makes the field ABSENT: `Record.to_dict` drops
        every None, which is the schema's own convention for a field the
        engine has no answer for (`ef_construction` is already absent for
        MHNSW and VIDX, which expose no such knob). A reader that wants a
        display name for PostgreSQL's heap has one in
        `report/render.py:_STORAGE_OVERRIDES`, keyed on the engine, which is
        where a presentation name belongs; a measurement record saying nothing
        is not the same claim as a measurement record saying InnoDB.

        **Why absent rather than the honest name, which is the obvious
        alternative and would make the ops half agree with the ann half
        verbatim.** `report/loaders.py:memory_stem` reconstructs the filename
        the orchestrator gave this unit's `mem-*.jsonl` sampler, and that name
        suffixes the storage engine only when it is not InnoDB
        (`orchestrator/cli.py`: `if storage.lower() != "innodb"`). The
        orchestrator names the file from ITS value, which is `InnoDB` for all
        three engines, so the file on disk is unsuffixed -- in that run,
        `mem-pgvector-...-m16-post.jsonl`. A record saying `heap` would make
        `memory_stem` ask for `...-m16-post-heap`, match nothing, and drop the
        sampled peak RSS for three of seven engines out of the report. An
        absent field takes the same `or ""` branch InnoDB takes and keeps that
        join intact, so this is the one spelling that corrects the claim
        without moving a second number. Naming the real engine here is a
        change to `report/` as well, and is not this one.

        That join is not theoretical and the cost is not small: in
        `full-20261015-215006` EVERY index_build record for those three
        engines carries `extra.peak_rss_source: "sampled"`, so
        `report/generate.py:attach_peak_rss` is the only thing that ever gave
        them a `peak_rss_bytes` at all -- and it is every engine, not only
        these three: all 54 of that run's index_build records are sampled. A
        stem that stopped matching would not publish a wrong peak memory
        figure, it would publish none.

        Withdrawing the claim carries an `OPS_MEASUREMENT_VERSION` bump to 8,
        even though no measured quantity moves, because this field is an
        identity component of `OPS_UNIT_FIELDS`, `build_signature` and
        `memory_stem` rather than a label: see the version 8 block in
        `harness/metrics/records.py` for the legend it otherwise publishes
        across the boundary.

        WHAT THIS DOES NOT FIX, because it cannot be fixed from inside
        `harness/`: `webui/static/fields.js` declares `storage_engine` as a
        FILTER on all six ops views, and
        `tests/test_c2_webui_phases.py` asserts for the ingest and
        calibration views that every declared filter is a facet every record
        of that phase carries. That assertion held only because this method's
        predecessor lied -- it is a guard that depended on the defect. With
        the claim withdrawn, an engine with no storage engine carries no such
        facet, and the declaration is the half that is now wrong. Whether the
        six views drop the filter, or `explore.js` learns to hide a facet no
        record in the current view carries, is a decision about the Explore
        tab and belongs with it.

        A method rather than a class attribute beside `incremental_index` and
        `repair_path`, which is where an engine's fixed properties live,
        because the MySQL family's answer is not fixed: MariaDB's
        InnoDB-versus-MyISAM sweep is a real within-engine comparison
        (`orchestrator/cli.py:_ops_storage_engines`, `docs/05-methodology.md`),
        and both values have to reach the record for the report to keep the
        two curves apart.
        """
        return None

    # -- data -----------------------------------------------------------

    @abc.abstractmethod
    def load(self, vectors: numpy.ndarray,
             attrs: Union[RowAttrs, Sequence[int], numpy.ndarray],
             threads: int = 1, start_id: int = 0) -> LoadResult:
        """Ingest `vectors` with their scalar columns.

        `attrs` is a `RowAttrs`; a bare array of tag values is accepted and
        mirrored into `ctag` (see `RowAttrs.coerce`). Every implementation
        coerces on entry rather than branching afterwards, so the write path
        below the coercion only ever sees both columns.
        """

    @abc.abstractmethod
    def delete_ids(self, ids: Sequence[int]) -> None: ...

    @abc.abstractmethod
    def insert_rows(self, ids: Sequence[int], vectors: numpy.ndarray,
                    attrs: Union[RowAttrs, Sequence[int], numpy.ndarray]
                    ) -> None: ...

    @abc.abstractmethod
    def count_rows(self) -> int: ...

    # -- churn ----------------------------------------------------------
    # Three optional hooks with no-op defaults, so an engine that has no
    # mechanism for one of them needs no code at all. The defaults are what
    # MariaDB, AliSQL and (for two of the three) Valkey and Percona Search
    # use, and a driver that overrides none of them measures exactly what the
    # churn workload measured before these existed.

    def prepare_churn(self) -> None:
        """Put the engine in a deterministic state before the deletes.

        Called once per churn run, before the baseline measurement, so that
        whatever it changes is in force for the baseline as well as for the
        post-churn measurements -- a knob flipped between the two would make
        the comparison a comparison of knobs.

        pgvector is the engine this exists for: autovacuum decides on its own
        schedule whether the tombstones a churn leaves behind are still there
        when recall is re-measured, so the "before maintenance" number was
        partly a vacuumed index and partly not, with nothing in the record to
        say which.
        """

    def wait_index_synced(self, expected_rows: int) -> float:
        """Block until an asynchronously maintained index reflects the writes.

        Returns the seconds spent waiting -- 0.0 for the engines that maintain
        their index on the write path, where there is nothing to wait for.

        The return value is a measurement and not a status: the churn workload
        records it as `extra.index_sync_s`, so an engine whose index lags by
        minutes after a re-insert says so in the record rather than reporting
        a recall taken against an index that had not caught up. A hook that
        returned nothing would make "the index was ready" and "nobody asked"
        the same observation.
        """
        return 0.0

    def maintain(self) -> Optional[str]:
        """Run the engine's repair path; return a short description of what ran.

        `None` means this engine has no repair path, and the churn workload
        then writes one record per fraction instead of a before/after pair.
        A description is what the record carries as `extra.maintenance`, so it
        is read by a human: "VACUUM (ANALYZE)", not a row count.

        Deliberately not timed here. The caller times the call, because the
        number belongs beside the recall figures it explains and a driver that
        timed itself would have to be trusted about the clock it used.
        """
        return None

    # -- server-reported status ------------------------------------------

    def status_counters(self, names: Sequence[str],
                        scope: str = "global") -> Dict[str, int]:
        """Server status variables by name, as integers. `{}` by default.

        A fourth optional hook with a no-op default, beside the three churn
        hooks above and for the same reason: six of the seven engines here
        expose nothing of the kind, and they must need no code and record
        exactly what they recorded before this existed. An engine that
        overrides nothing collects nothing and writes no record key -- see
        `status_gauges` and `harness/workloads/build.py:_server_status`.

        **It must not cost a measurement.** A status read happens AFTER the
        thing it describes has been built and timed, so a server that has
        gone away by then has already been paid for. An implementation
        therefore omits what it cannot read rather than raising, and the call
        site swallows anyway -- the pattern is
        `harness/workloads/build.py:_driver_capabilities`, which keeps a
        completed measurement when a driver misbehaves afterwards.

        **Omitting, not zeroing, and the distinction is the whole point of
        the return type being a dict rather than a row.** A variable the
        server does not expose and a variable reading zero are different
        facts about different servers, and for the one gauge this hook exists
        for they are both reachable: `docs/09-adding-percona-server.md` §2.6
        records `Innodb_hnsw_memory_used` reading `0` after a restart,
        because the graph is faulted in lazily and the figure rises on the
        first search. So zero is a legitimate measurement here, and a
        collector that zero-filled a missing name would publish "the graph is
        not loaded" and "this server has no such variable" as one number.

        `scope` is `"global"` or `"session"`. It is a parameter rather than a
        property of the name because the MySQL family has variables in both
        scopes and the same statement serves either; the default is global,
        which is what the only declared gauge needs.

        **THE CAVEAT THAT TRAVELS WITH THE NUMBER, repeated here because this
        is where a future reader will look for what the number means.** For
        `Innodb_hnsw_memory_used` the gauge omits the node map
        (*HNSW::m_nodes*, a supplied 30-50 bytes per node) and search
        scratch, so it is a **floor** on graph memory and not a measurement
        of it -- at glove-100's 1,183,514 rows, 35.5-59.2 MB unreported per
        vector index. `config/engines/percona.yml` states this in capitals,
        `CLAUDE.md` restates it, and a report quoting the figure without it
        is quoting a number a reader will reconcile against a container's
        resident set and find short.
        """
        return {}

    # -- server-reported waits, per measured window ------------------------
    #
    # A fifth optional hook pair, added the way `status_counters` /
    # `status_gauges` were: a no-op default on the base class, so an engine
    # that overrides nothing writes `server_waits: null` with the reason
    # below and needs no code. The capture lives in
    # `harness/workloads/server_waits.py`; see `harness/drivers/server_waits.py`
    # for why a wait profile is the evidence the version-9 server CPU and
    # pressure columns could not supply.

    #: Seconds between samples for an engine whose wait accounting is a
    #: point-in-time view rather than a cumulative counter (pgvector's
    #: `pg_stat_activity`). None for every engine with counters, for which
    #: two snapshots -- one before the window, one after -- are enough.
    wait_sample_interval_s: Optional[float] = None

    def wait_snapshot(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """`(snapshot, None)` or `(None, why not)`. Never raises.

        Called on a SEPARATE connection -- the capture opens its own driver
        through the factory -- immediately before and immediately after a
        window, and never inside it: a performance_schema scan is not free,
        and paying for it inside the span the QPS divides by would charge
        the instrument to the engine. A sampling engine
        (`wait_sample_interval_s` set) is instead called once per interval
        from a sampler process the capture forks before the clients (a
        thread alive at their fork is what `tests/test_g4_pool_teardown.py`
        forbids), on the sampler's own connection.
        """
        # Imported here, not at the top: line numbers in this file are cited
        # by `tests/test_e13_percona_churn_repair_is_a_noop.py`.
        from .server_waits import NOT_IMPLEMENTED
        return None, NOT_IMPLEMENTED

    def wait_delta(self, before: Dict[str, Any], after: Dict[str, Any]
                   ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """The window's waits from two snapshots, in the shape
        `harness/drivers/server_waits.py` documents; `(None, why)` when the
        two cannot be differenced."""
        from .server_waits import NOT_IMPLEMENTED
        return None, NOT_IMPLEMENTED

    # -- query ----------------------------------------------------------

    @abc.abstractmethod
    def set_ef_search(self, ef_search: int) -> None: ...

    def effective_ef(self, ef: int, k: int) -> int:
        """How many candidates this engine will really search for `ef`.

        For most engines here the answer is `ef`, which is why this is a
        concrete method rather than another abstract one. THREE clamp, and all
        clamp to k (the third, MyVector, is hnswlib's own `searchKnn`, which
        searches at `std::max(ef_, k)`):

        * MongoDB -- `numCandidates` below `limit` cannot return `limit` rows,
          so it raises the low end of the grid to k.
        * Percona Server -- `innodb_hnsw_ef_search`'s own help text states the
          contract ("the effective width is max(ef, LIMIT)") and the engine
          passes the variable beside `std::max<size_t>(limit, 1)` on the scan
          path. Unlike MongoDB's, this clamp bites at no grid point this repo
          currently sweeps at k=10; it is implemented anyway because the
          inherited identity would be wrong at the one point where it mattered
          and silently right everywhere else.

        Either way the calibration records have to say so -- a reader
        comparing two engines at "ef 10" needs to know that one of them looked
        at 10 candidates and the other at k.

        Takes `k` because the only clamp in this codebase is against k. The
        spec names the shape `effective_ef(ef)`, and a one-argument version
        would have to keep its own copy of k: k arrives per query
        (`query(vector, k)`) and no driver stores it, so the value would be
        stale exactly when a profile changed it.

        Reports rather than enforces. The engine does the clamping; this says
        what it did, and a driver that returned a value its queries do not use
        would be worse than no method at all.
        """
        return int(ef)

    @abc.abstractmethod
    def query(self, vector, k: int) -> List[int]: ...

    @abc.abstractmethod
    def query_filtered(self, vector, k: int, threshold: int,
                       column: str = FILTER_TAG) -> List[int]:
        """Top-k among the rows satisfying `column < threshold`.

        `column` is one of `datasets.FILTER_COLUMNS` and is put through
        `checked_filter_column` before it reaches any query string. It
        defaults to `tag`, which is what every caller measured before `ctag`
        existed, so an existing three-argument call keeps measuring exactly
        what it measured.

        The third parameter is `threshold` rather than the `tag_threshold` it
        was: it is no longer a threshold on `tag` specifically. Every caller
        passes it positionally -- `harness/workloads/filtered.py` at 110 and
        125, and `tests/test_fake_driver.py` at 291, 300, 304, 310, 317 and
        319, eight sites counted by AST rather than by eye -- so the rename
        reaches nobody. That is also why `explain_uses_vector_index` below
        could NOT be renamed with it: its third argument is passed by
        KEYWORD, at `filtered.py:104` and `test_fake_driver.py:65`.
        """

    @abc.abstractmethod
    def explain_uses_vector_index(self, vector, k: int,
                                  tag_threshold: Optional[int] = None,
                                  column: str = FILTER_TAG) -> bool:
        """Whether the planner actually chose the vector index.

        Not a nicety. Every one of these engines can fall back to a full scan,
        which returns exact results slowly — indistinguishable in the output
        from "high recall, low QPS" unless it is checked explicitly.

        `tag_threshold` keeps its name here while `query_filtered`'s became
        `threshold`, and the asymmetry is deliberate rather than an oversight:
        this one is passed by keyword by two callers that may not be edited
        from this task -- `harness/workloads/filtered.py:104` and
        `tests/test_fake_driver.py:65` -- so renaming it is a TypeError in
        both. `column` is appended instead, which is what the spec asks for
        and all it asks for: it writes this signature as
        `explain_uses_vector_index(..., column="tag")`, eliding the existing
        parameters rather than restating them.
        """

    # -- sizing / metadata ----------------------------------------------

    @abc.abstractmethod
    def index_bytes(self) -> int: ...

    @abc.abstractmethod
    def table_bytes(self) -> int: ...

    @abc.abstractmethod
    def server_version(self) -> str: ...

    def capabilities(self) -> Dict[str, Any]:
        return {
            "incremental_index": self.incremental_index,
            "async_index_build": self.async_index_build,
            "ef_construction_tunable": False,
            **self.index_format(),
        }

    def index_format(self) -> Dict[str, Any]:
        """The two `index_vector_format` keys, for every `capabilities()`."""
        return {"index_vector_format": self.index_vector_format,
                "index_includes_vectors": self.index_includes_vectors}

    def settle_index_bytes(self):
        """(bytes, seconds, detail) for the size after the build has settled.

        Immediate by default. Percona Search overrides it -- its directory
        sum halves over minutes as Lucene merges -- and the InnoDB engines do
        not: a tablespace grows when pages are ALLOCATED, not when they are
        flushed, so a checkpoint would not move the `.ibd` size a reader sees.
        """
        return self.index_bytes(), 0.0, {"method": "immediate"}


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
