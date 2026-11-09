"""One client pool, shared by the concurrency and churn workloads.


PR 4 §4.3 item 3 asks for this module by name, and the reason is that churn now
needs readers running while the writer works. Every piece of machinery that
takes is already in `concurrency.py` and was argued for there over three
separate corrections, so the choice was to lift it or to write a second
implementation that would start out agreeing and then drift:

* **The barrier that opens the window.** `threading.Barrier(action=...)` reads
  the clock in the last party to arrive, after every party is waiting and
  before any is released, so the window's start instant provably precedes
  every measured query. Reading it in the parent thread after `wait()` returns
  is the obvious place and is wrong: with eight clients on one GIL, six of them
  issued their first query first, those queries looked as though they began
  before the window, and dropping them pinned `extra.recall_queries` at 198 of
  a 200-query pool however long the window ran.

* **PR 1's window trimming.** A fixed-duration window ends by setting a flag;
  each client then finishes the query it is holding. Counting those put more
  queries in the numerator than the denominator's wall covered, by
  `inflight_discarded / queries_executed` -- measured flat at 2.564% from 1 to
  32 clients, because the client count cancels, and ~0.005% at `main`'s 30 s
  windows. It also reported more queries in flight on average than there were
  clients to hold them, which no client count can do. One
  `span_disposition` call decides which latencies AND which returned ids
  count, so the recall beside a throughput figure is over the queries that
  figure counted.

* **PR 1's CPU accounting.** The clients are forked OS processes sharing one
  container's cpuset. `client_cpu` divides a cgroup counter by a wall clock
  and a CPU count, and the record keeps all three, because each of the three
  can be wrong while the quotient stays plausibly between 0 and 1.

**What a pool's numbers are NOT evidence about, and this matters more now that
a second workload inherits it.** `CLIENT_BOUND_UTIL` is sufficient and not
necessary, and the gap is structural rather than a question of where the limit
was set: a CPU-utilisation threshold cannot see a GIL ceiling, because the lock
prevents the CPU consumption the threshold looks for. Measured on the RETIRED
thread pool against `tests/fake_driver.py`, on a twenty-CPU box: its own CPU
flattens at 2.06 cores -- 0.51 of a four-CPU client container at 4, 8, 16 and
32 clients alike -- while aggregate throughput peaks at two clients
(16,614 q/s) and falls to 8,127 by thirty-two. That is the exact shape a reader
would call an engine ceiling and the flag fired on none of those points. **No
GIL ceiling is reachable now that the clients are processes**; what is left is
a cpuset, and a reader pool's QPS is still this harness against this engine.

**And the synthetic figure above overstates a real client's cost, so the
synthetic SHAPE may not reproduce.** The fake's queries are numpy-bound
in-process, whereas a real driver spends most of a query blocked on a socket
with the lock released -- which is why the decision to run the clients as
processes waited on a real dev or smoke run rather than on this measurement.
This paragraph lived at `concurrency.py:118` before PR 4 §4.3 lifted the flag
into this module, which deleted it; it is restored here because two workloads
now inherit the flag and because it is the qualitative form of a measurement
PR 4 went on to make. **That run has now happened -- see THE CLIENTS ARE
PROCESSES below -- and it did not reproduce the synthetic shape: it was
worse.**

That measurement, for the one quantity where the two regimes were compared
directly: a reader pool's aggregate QPS divided by a single-client baseline --
the spurious retention `report/churn_state.py` exists to keep out of the churn
chart -- came out **3.74 to 4.26** over 9 runs with readers whose query
releases the lock (a `latency_s` sleep, the socket-like regime) and **0.89 to
2.62** over 9 runs with numpy-bound readers, which is the regime the 2.06-core
figure above was taken in. Two independent measurements agree on both rows:
a second party measured 3.83-4.24 and 1.06-1.93, each inside the range above.

So "may not reproduce" is not hedging. The socket-like regime puts that
quantity at roughly the reader count and the numpy-bound one puts it near 1,
which are different answers to "do four readers deliver four readers' worth of
throughput" rather than two estimates of one answer. The ranges overlap at
their edges, so no single factor separates them and none is quoted here. The
numpy-bound regime is the pessimistic one for scaling and the optimistic one
for this flag, which is the direction that matters: a flag calibrated there
under-reports what a real client costs.

**THE MECHANISM, AND WHICH PARTS OF IT ARE MEASURED.** A live `smoke` run
exposed this rather than a reading of the code, so the claims are separated by
how each one is known.

*Observed (SUPPLIED -- a `smoke` run of MariaDB on fashion-mnist-784 at
M=16, not reproduced here).* Throughput flat across a four-fold rise in
clients -- 2,920.7 q/s at one and 2,927.5 at four -- with every percentile
unchanged: p50 0.325 ms, p95 0.442/0.439, p99 0.552/0.544. `qps x p50` is
0.95, so about one query was in service. The same run's churn writer managed
12.0 rows/s against four readers at 2,819 q/s, against a modelled ~150.

*Measured here.* A saturated server and a serializing client ARE
distinguishable from the client's own spans, and `served_concurrency` is the
quantity that does it -- see `SERVED_CONCURRENCY_FLOOR` for the five-driver
calibration and `metrics/latency.WindowConcurrency` for why the exact
span-sum does not. The pool itself was never the fault: driven by a
lock-releasing driver it scales 1.02 at four clients. `client_cpu_util` on
the serializing point reads 0.177 against a 0.80 threshold.

*Read, not measured, and it CANNOT be measured here -- this host starts no
server.* Whether a client library releases the interpreter lock around its
socket wait is a property of that library's C extension, and the engines do
not share one: `drivers/postgres.py` imports **psycopg** (psycopg3) while
`drivers/mysql_family.py` imports **mariadb** (MariaDB Connector/Python;
over Connector/C until 1.1.10, pure Python since 2.0), which `docker/_shared/requirements-mysql.txt` says is used for
BOTH MariaDB and AliSQL on purpose. The question is per-library, not
per-engine, and nothing in this repository settles it.

**THE CLIENTS ARE PROCESSES, AND `client_process.py` HAS THE MECHANISM.**
The paragraph that used to end this docstring said running them as processes
was "the fix for the underlying limit" and "deliberately NOT attempted", and
named the next real run as the thing that would settle whether the limit was
the client library. That run finished on 2026-10-14 and settled it in the
direction the paragraph feared -- three of seven engines pinned near one
query in service at four clients, exactly the three sharing MariaDB
Connector/Python. `harness/workloads/client_process.py` carries that run's
table, the fork-versus-spawn decision, and §7's cost list measured item by
item; `docs/05-methodology.md` §7 carries the same figures for a reader who
is not in the code. What stays here is the pool: the barrier, the window, the
two concurrency figures, the client-CPU accounting and the sharding.

The split is this repository's 800-line ceiling, which this file stood at 791
lines against before G3. **A re-export does not SHARE a name**, so the split
had to be checked against every test that patches one of these: see
`client_process.py`'s docstring, which lists them and says which single
target moved.

`_one_window` deliberately stays in `concurrency.py`: it owns the fixed
duration, the recall scoring and the repeats, none of which churn wants.
"""

from __future__ import annotations

import threading
import time
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple)

import numpy

from ..drivers.base import IndexSpec
# The client itself is an OS process and lives next door; see that module for
# why the split is where it is and for what a re-export does not share.
# Re-exported here under the names `concurrency.py`, `churn.py`,
# `churn_readers.py` and four test files already import.
from .client_process import (CLIENT_START_METHOD,  # noqa: F401
                             Answer, Client, ClientDied, ClientFailed,
                             DriverFactory, PoolError, Timeouts, _MP,
                             _exit_description)
# The window's value types and the refusal that guards a ratio across two
# of them. Re-exported because `report/`, `churn.py` and four test files
# import them from here; `CLIENT_BOUND_UTIL` comes with them because
# `PoolWindow.client_bound` is its only reader, and that module's
# docstring records that rebinding it HERE therefore reaches nothing.
from .pool_window import (CLIENT_BOUND_UTIL,  # noqa: F401
                          PoolSpec, PoolWindow, window_ratio)
# The scaling-status policy lives in `metrics/latency.py`, beside the
# quantity it judges and because this module was 846 lines against the repo's
# 800-line ceiling with it inlined -- the reason `churn_readers.py` exists.
# Re-exported under the names it was written with, because that is how
# `concurrency.py`, `churn_readers.py` and the tests reach them. **A
# re-export does not SHARE the name**: rebinding `clients.SERVED_CONCURRENCY_FLOOR`
# does not reach `scaling_status`, which reads it out of `latency`'s globals,
# and `tests/test_g1_client_concurrency.py` asserts that asymmetry so a
# mutation aimed here is not taken for a clean result.
from ..metrics.latency import (INFLIGHT_DISCARDED,  # noqa: F401
                               PREWINDOW_DISCARDED,
                               SCALING_CONCURRENCY_NOT_ACHIEVED, SCALING_OK,
                               SCALING_STATUSES, SERVED_CONCURRENCY_FLOOR,
                               LatencyCollector, merge_collectors,
                               scaling_status, served_fraction,
                               span_disposition, window_concurrency)
from ..metrics.resources import own_cpu_count, read_own_cpu_usec
from ..progress import Heartbeat

#: Seconds the PARENT waits at the barrier, and it does **not** bind. Kept at
#: the value it had before the lift so nothing about the window moved, and
#: documented honestly: a client that is still warming up when
#: `CLIENT_BARRIER_TIMEOUT_S` expires breaks the barrier from its own side,
#: which makes this wait raise `BrokenBarrierError` immediately. So warm-up is
#: bounded at roughly `CLIENT_BARRIER_TIMEOUT_S` after the first client
#: arrives, and 300 s is a ceiling that cannot be reached while the client
#: side is the smaller of the two.
#:
#: An earlier revision of this comment called it "the only thing that stops a
#: stuck driver holding the machine". That was the 120 below, unnamed at the
#: time, and naming it is what made the inconsistency visible.
BARRIER_TIMEOUT_S = 300.0

#: Seconds a CLIENT waits at the barrier, which is the one that binds. A
#: client reaching the barrier waits for its siblings; the slowest warm-up
#: therefore has this long from the first arrival, not `BARRIER_TIMEOUT_S`.
#:
#: Named rather than left as the bare `timeout=120` it was before the lift,
#: because the two numbers only make sense read together: the smaller one is
#: the bound, and a future edit that raised this above 300 would silently move
#: the bound to the parent without either comment changing.
#:
#: It is also the bound on how long the parent waits for a client PROCESS to
#: say it is warm, which is what made the pair bind in the one-client case
#: too. A single hanging thread used to leave the parent's 300 s as the only
#: limit, because there was no sibling to time out at the barrier and break
#: it; `Client.attend` aborts the barrier itself when its client has not
#: reported inside this, so the smaller number now binds at every client
#: count. That is the direction the comment above already claimed.
CLIENT_BARRIER_TIMEOUT_S = 120.0

#: Seconds a client is given to notice the stop flag and return. One query's
#: worth of slack over any latency this framework has measured; a client still
#: running after it is reported as a failure rather than waited on, because a
#: join with no timeout is how a reader exception becomes a hang.
#:
#: It bounds the span transport as well as the join, and the order matters: a
#: client PROCESS blocks writing its spans into a 64 KiB pipe until the parent
#: reads them, so `stop` reads every client's payload BEFORE it joins any of
#: them. Joining first is a deadlock rather than a slow path.
JOIN_TIMEOUT_S = 120.0


class Instruments(NamedTuple):
    """The clocks, counters and primitives a pool measures itself with.

    Injectable, and the reason is not abstraction for its own sake: five
    existing concurrency tests calibrate a window by replacing exactly these
    names **on `harness.workloads.concurrency`** -- a CPU counter that
    advances by one CPU-second per CPU per wall second so the expected
    utilisation is 1.0 by construction, a `merge_collectors` that runs twice
    to show what the untrimmed merge counted, and a `threading` whose `Event`
    takes a visible amount of time to `set()` so the gap between the window's
    end and the flag is measurable.

    Rebinding a name on the concurrency module does **not** reach a function
    this module captured at import time -- that is how three mutants got a
    fully green suite in an earlier task here -- so the pool cannot simply
    import them. `concurrency._one_window` therefore passes its own
    module-level names in, which keeps the concurrency module the one place a
    test replaces an instrument, and keeps PR 4's acceptance ("concurrency
    tests unchanged") meaningful rather than something the lift quietly broke.

    `threading` needs only `Event`, `Barrier` and `BrokenBarrierError`: the
    thread class itself is resolved in this module and a test that wanted to
    replace it would be replacing the client, not an instrument.

    **IT IS STILL `threading`, AND THAT IS DELIBERATE NOW THAT THE CLIENTS ARE
    PROCESSES.** The obvious reading of "make the clients processes" is to
    hand this field a `multiprocessing` context, and it is wrong twice. It
    would not even import: `multiprocessing` and its contexts have **no**
    `BrokenBarrierError` attribute -- measured, `hasattr` is False on both --
    because `multiprocessing.synchronize.Barrier` SUBCLASSES
    `threading.Barrier` and raises `threading.BrokenBarrierError` itself, the
    same class object. And `concurrency._instruments()` passes
    `threading=threading` literally, so a pool that needed a multiprocessing
    bundle here would be broken by its own caller. So the barrier and the stop
    flag stay THREAD primitives in the PARENT, where the barrier's action
    still reads the window's opening instant in the last party to arrive;
    `Client.attend` stands in for each client process at that barrier and
    `ClientPool._go` / `_halt` are the two inter-process flags that carry the
    decision across the fork. `tests/test_g3_process_clients.py` asserts the
    `BrokenBarrierError` fact, so a future edit that reaches for a context
    here fails on the reason rather than on a traceback.

    **`time` is deliberately NOT in this bundle, and the asymmetry is worth
    knowing about.** Every window instant -- the barrier action's
    `perf_counter`, the stop instant, each client's per-query span -- is read
    through THIS module's `time`. `concurrency.time` is still live because
    `_one_window` sleeps on it, so a test that patched `concurrency.time`
    would reach the sleep and not the measurement, where before the lift it
    reached both. Nobody does that today. Two things did get safer in the same
    move, in the direction that fails loudly: `span_disposition` and
    `LatencyCollector` are no longer attributes of `concurrency` at all, so a
    `monkeypatch.setattr` on either now raises `AttributeError` instead of
    passing green against a patch that reached nothing.

    A client process reads `time` out of its own copy of this module, which is
    the same module object the parent forked -- and the readings are
    comparable because `perf_counter` is `clock_gettime(CLOCK_MONOTONIC)`,
    whose reference point is the machine's rather than the process's. That is
    verified rather than assumed; the module docstring has the 200-fork
    measurement.
    """

    threading: Any = threading
    merge: Callable[..., Dict[str, float]] = merge_collectors
    cpu_usec: Callable[[], Optional[int]] = read_own_cpu_usec
    cpu_count: Callable[[], int] = own_cpu_count


DEFAULT_INSTRUMENTS = Instruments()



def shard_queries(queries: numpy.ndarray,
                  clients: int) -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
    """Each client's query slice, paired with those queries' positions.

    The slice and the offsets are built in one place and returned together
    because they are two spellings of one fact: `queries[c::clients]` is a
    strided view and `arange(c, n, clients)` is where its elements came from.
    Let them drift apart and recall is scored against the wrong rows of the
    ground truth — client 1 of 4 answering queries 1, 5, 9, ... while its
    results are compared against the neighbours of queries 0, 1, 2, ... That
    produces a plausible-looking low recall rather than an error, which is the
    exact failure mode this repository opens by warning about, and it would be
    read as the engine's ANN quality.

    Strided rather than contiguous blocks, which is the older decision this
    preserves: two clients handed adjacent slices of a sorted query set warm
    each other's caches, and part of what the window measured would be that
    rather than the engine.

    A VIEW and not a copy, which is now load-bearing for a second reason.
    `docs/05-methodology.md` §7 states as a cost of client processes that
    "each client becomes an OS process with its own copy of the query set".
    It does not: these are strided views of one array, a client never WRITES
    to one, and fork's copy-on-write therefore copies no page of it. Which is
    not the same as a child leaving the pages alone -- `queries[c::clients]`
    is strided across the whole array, so every child READS pages spread over
    the entire query set and only the write is absent. SUPPLIED: 0.13x the
    array summed over eight children against a 104.9 MB array, with a control
    arm of 8.12x for eight children that write to it. No test here reproduces
    those; `client_process.py`'s docstring has both arms and says so.
    """
    n = len(queries)
    return [(queries[c::clients], numpy.arange(c, n, clients))
            for c in range(clients)]


def client_cpu(cpu_start: Optional[int], cpu_end: Optional[int],
               wall_s: float,
               cpus: int) -> Tuple[Optional[float], Optional[float]]:
    """The clients' own CPU over a window, in seconds and as a utilisation.

    Both numbers from one place because the second is the first divided by
    `wall_s * cpus`, and a record that carried only the ratio would leave a
    reader unable to tell a busy window from a short one or from a narrow
    cpuset. Each of those three inputs can be wrong in a way that still
    produces a number between 0 and 1: the counter in the wrong unit (cgroup
    v1 publishes nanoseconds), the wall taken around the wrong interval, or
    the CPU count taken from the host rather than from the cpuset. So all
    three are recorded and the arithmetic is one expression in one function
    that a test can calibrate against a known number of saturated cores.

    `(None, None)`, never `(0.0, 0.0)`, when the counter is unavailable or the
    window is degenerate. Zero is the claim "the clients did nothing", which
    is the one thing that cannot be true of a window that executed queries,
    and it would make the client-bound flag read false on a run where nothing
    was ever measured.

    Still a measurement of the clients now that they are separate processes,
    and the reason is structural rather than empirical: `fork` does not change
    cgroup membership -- only an explicit write to `cgroup.procs` does -- so
    every client is inside the harness container's cgroup and the one counter
    `read_own_cpu_usec` reads covers all of them. Verified by comparing
    `/proc/self/cgroup` between parent and children, which is exact; the
    magnitude cannot be verified outside a container, because there the same
    file is the root cgroup and reads the whole machine (§2.3).
    """
    if cpu_start is None or cpu_end is None or wall_s <= 0 or cpus <= 0:
        return None, None
    # Clamped at zero: the counter is monotonic, so a negative delta means the
    # two readings did not come from the same cgroup and the magnitude is
    # meaningless either way.
    seconds = max(0.0, (cpu_end - cpu_start) / 1e6)
    return seconds, seconds / (wall_s * cpus)



class ClientPool:
    """A pool of query clients and one measured window over it.

    Two calls, in order: `start()` runs the warm-up and opens the window at
    the barrier; `stop()` closes it and returns a `PoolWindow`. The caller
    decides what happens in between -- `concurrency._one_window` sleeps for a
    fixed duration, `churn` waits for the writer -- and that is the only
    difference between the two workloads' use of this machinery.

    A pool measures ONE window. Processes cannot be restarted any more than
    threads could, and the churn workload needs its reader windows either side
    of a delete step it must not be reading through (§4.3 item 4 starts the
    readers after the delete), so it builds a second pool from the same
    `PoolSpec` instead. `PoolSpec` equality is what makes "the same pool"
    checkable rather than asserted -- see `PoolSpec` for the two
    published-wrong ratios that motivated it.

    **THE THREE THINGS THAT CROSS THE FORK, and why they are not more.** The
    clients are processes, so the barrier, the stop flag and the per-client
    collectors can no longer all be one kind of object:

    * the **barrier** stays a thread barrier in the parent, with one
      `Client.attend` thread standing in for each client, because its action
      reads the window's opening instant into parent memory and because it is
      an injected instrument (`Instruments`);
    * **`_go` and `_halt`** are the two inter-process `Event`s that carry the
      barrier's decision and the stop flag's decision to the children. Two
      for the POOL and not two per client, which is what keeps the `/dev/shm`
      footprint at a measured 40,960 bytes whatever the client count;
    * the **spans** come back over one pipe per client and are rebuilt into
      the same `LatencyCollector` objects the merge has always been handed,
      so `merge_collectors` and `window_concurrency` see exactly what they
      saw -- one shared wall, one `span_disposition` rule, no new definition
      of "inside the window".
    """

    def __init__(self, driver_factory: DriverFactory, queries: numpy.ndarray,
                 clients: int, k: int, ef_search: int, index: IndexSpec,
                 warmup: int, collect_results: bool = True,
                 unit: Tuple[Any, ...] = (),
                 instruments: Instruments = DEFAULT_INSTRUMENTS):
        if clients <= 0:
            raise ValueError(f"a pool needs at least one client, got {clients}")
        self.spec = PoolSpec(clients=clients, queries=len(queries), k=k,
                             ef_search=ef_search, unit=tuple(unit))
        self._instruments = instruments
        self._stop = instruments.threading.Event()
        # The two flags the CLIENTS watch. The parent's `_stop` above is the
        # instrument a test replaces and the witness that the window's end is
        # read before any flag goes up; `_halt` is what actually ends a
        # client's loop, and it is set immediately after `_stop`. Setting
        # `_halt` first would let a slow `_stop.set()` -- which one
        # concurrency test injects on purpose -- stop the clients before the
        # ordering it exists to demonstrate could be observed.
        self._go = _MP.Event()
        self._halt = _MP.Event()
        # The instant the window opens, read by the barrier itself -- see this
        # module's docstring for what reading it in the parent thread did to
        # `extra.recall_queries`.
        #
        # The CPU counter is read in the same barrier action, immediately
        # before the wall instant, so the two readings bracket one interval
        # instead of two. Reading it in the parent after `wait()` returns
        # would put the clients' first queries' CPU outside a window whose
        # denominator covers them; reading it before the barrier would include
        # every client's warm-up, which is hundreds of queries of CPU charged
        # to a window that did not run them.
        #
        # CPU first, then the wall: the file read costs microseconds, and
        # spending them before the wall starts means the window contains no
        # CPU that its start reading has already counted.
        self._opened: List[Tuple[Optional[int], float]] = []
        self._ready = instruments.threading.Barrier(
            clients + 1, action=self._open_window)
        self._workers = [
            Client(driver_factory, shard, offsets, k, ef_search, index,
                   warmup, self._ready, self._stop,
                   collect_results=collect_results,
                   go=self._go, halt=self._halt,
                   broken_barrier=instruments.threading.BrokenBarrierError,
                   timeouts=self._timeouts())
            for shard, offsets in shard_queries(queries, clients)
        ]
        #: The parent threads that stand in for the clients at the barrier.
        #: Empty until `start`; they are finished before the window opens.
        self._attendants: List[threading.Thread] = []
        self._started = False

    @staticmethod
    def _timeouts() -> Timeouts:
        """The three waits, read out of THIS module's globals at construction.

        **Read here and not imported by `client_process.py`, and that is the
        one thing the split had to get right.** A module that re-exports a
        name does not share it, so a `Client` that did `from .clients import
        CLIENT_BARRIER_TIMEOUT_S` would freeze the value at import and a
        `monkeypatch.setattr(cl, "CLIENT_BARRIER_TIMEOUT_S", 0.3)` would reach
        nothing -- which is exactly how three mutants got a fully green suite
        in an earlier task here. `tests/test_t4b_readers.py` sets both barrier
        numbers on this module and then measures when the barrier broke, so a
        frozen copy would have turned that measurement into a reading of the
        shipped 120 s against itself.

        A STATICMETHOD reading globals rather than a module-level constant
        bundle, for the reason `concurrency._instruments()` is a function: a
        bundle built at import would hold the originals. The values are
        captured once per pool, at construction, which every patch site in the
        tree already precedes.
        """
        return Timeouts(barrier_s=BARRIER_TIMEOUT_S,
                        client_barrier_s=CLIENT_BARRIER_TIMEOUT_S,
                        join_s=JOIN_TIMEOUT_S)

    def _open_window(self) -> None:
        self._opened.append(
            (self._instruments.cpu_usec(), time.perf_counter()))

    def start(self, label: str, prefix: str) -> None:
        """Run the warm-up and open the measured window at the barrier.

        `label` and `prefix` are the heartbeat's, because warm-up is unbounded
        from the operator's point of view: every client runs its warm-up
        queries before the barrier releases, and on a slow configuration that
        is minutes of silence before the window even starts.

        The clients are FORKED FIRST, before any attendant thread and before
        the heartbeat, and the order is deliberate. A fork copies only the
        calling thread, so a child inherits any lock a sibling thread happened
        to be holding at that instant -- already locked, with no thread left
        to unlock it. Forking before this method starts a thread of its own
        keeps the window of exposure to threads the CALLER already had.

        **CPYTHON SAYS SO OUT LOUD, AND THE FIRST THING IT CAUGHT WAS THIS
        POOL'S OWN THREADS.** Since 3.12 `os.fork()` raises
        `DeprecationWarning: This process is multi-threaded, use of fork()
        may lead to deadlocks in the child` whenever the forking process has
        more than one thread. A full suite run reported **54** of them, and
        the first reading of that was wrong: it blamed the test fixtures that
        hold a `threading.Lock` to count queries. The real cause was here --
        a sweep builds one pool per window and `stop` did not join its
        attendants, so the next pool forked while the previous pool's were
        still being collected. `stop` joins them now and the count is **0**.
        `tests/test_g3_process_clients.py` carries the honest/mutant pair.

        What remains is genuinely not this method's to remove: a CALLER that
        already has threads of its own. A real run reaches here from
        `harness/main.py` with none. The suite's remaining 53 are all one
        caller and were bisected rather than guessed:
        `tests/test_webui_control.py` leaves a `webui/jobs.py` worker thread
        alive when it finishes, so every pool a later test builds forks beside
        it -- measured, that file paired with `tests/test_workloads.py`
        reproduces the warning and the other three `test_webui_*` files paired
        the same way produce none. The fix belongs in that test (join the job
        before the test returns), not here and not in a warning filter: a
        measurement framework that silenced a deadlock warning would be
        choosing a quiet log over a run that finishes, and this warning is how
        the pool's own un-joined attendants were found.
        """
        for worker in self._workers:
            worker.launch()
        self._started = True
        # One attendant per client, started after the fork so no child
        # inherits them. Each waits for its client to report warm and then
        # takes that client's place at the barrier.
        self._attendants = [
            threading.Thread(target=worker.attend, daemon=True,
                             name=f"client-attendant-{i}")
            for i, worker in enumerate(self._workers)]
        for attendant in self._attendants:
            attendant.start()
        try:
            with Heartbeat(label, prefix=prefix):
                self._ready.wait(timeout=BARRIER_TIMEOUT_S)
        except self._instruments.threading.BrokenBarrierError:
            self._ready.abort()
            self._stop.set()
            self._halt.set()
            first = self._first_error()
            # The clients are ended rather than left to notice a flag. A
            # thread that never noticed died with the interpreter; a process
            # that never notices goes on querying the server the next
            # configuration is about to measure, which would put one engine's
            # concurrency figure on top of the previous one's leftovers.
            for worker in self._workers:
                worker.reap()
            # And the attendants, for the reason `_join_attendants` gives.
            # **This path did not join them and that was the same defect the
            # 54 -> 0 fork-warning repair fixed in `stop`, reached from the
            # other side.** An attendant blocked in `_await_warmup`'s
            # `poll(client_barrier_s)` can outlive this method, and
            # `churn.py`'s three `except PoolError` sites go on to the next
            # churn fraction -- so the next pool's `launch` forks beside the
            # previous pool's live threads, which is exactly the
            # multi-threaded fork CPython 3.12 warns about. Joining here
            # rather than only in `stop` is what makes the warning count 0 on
            # the FAILURE path too; `stop` is never reached on it.
            self._join_attendants()
            raise PoolError(
                f"a client failed during warm-up: "
                f"{first or 'barrier timed out'}")
        # The window is open: its instant has been read in the action above,
        # before this line, so no measured query can precede it and
        # `extra.prewindow_discarded` is still zero by construction.
        self._go.set()

    def stop(self) -> PoolWindow:
        """Close the window and return it, or raise if a client failed."""
        if not self._started or not self._opened:
            raise PoolError("the window was never opened")
        cpu_start, wall_start = self._opened[0]
        # The stop instant, read BEFORE the flag is set and used as both the
        # denominator's end and the numerator's cutoff. Those were different
        # instants until this was recorded: the wall ended here, while every
        # client went on to finish the query it was holding and record it, so
        # QPS counted up to one query per client that its own wall did not
        # cover.
        stop_at = time.perf_counter()
        # And the CPU counter immediately after, for the same reason in the
        # same direction: read after `stop.set()`, it would include whatever
        # the clients burned finishing the query each was holding -- work the
        # wall above deliberately excludes.
        cpu_end = self._instruments.cpu_usec()
        self._stop.set()
        # The flag the clients actually watch, set immediately after the
        # instrument above and never before it. See `__init__`.
        self._halt.set()
        wall = stop_at - wall_start
        cpus = self._instruments.cpu_count()
        cpu_seconds, cpu_util = client_cpu(cpu_start, cpu_end, wall, cpus)

        # Every payload BEFORE any join. A client blocks writing more than the
        # pipe's buffer until the parent reads it, so joining first deadlocks
        # the pool on exactly the windows that produced the most spans. One
        # deadline across the pool, so a slow first client cannot hand the
        # last one a fresh timeout.
        deadline = time.monotonic() + JOIN_TIMEOUT_S
        for worker in self._workers:
            worker.absorb(deadline)
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

        failures = [w.error for w in self._workers if w.error is not None]
        if failures:
            for worker in self._workers:
                worker.reap()
            raise PoolError(
                f"{len(failures)} of {self.spec.clients} clients failed; "
                f"first error: {failures[0]}")
        still_running = [w for w in self._workers if w.is_alive()]
        if still_running:
            for worker in self._workers:
                worker.reap()
            raise PoolError(
                f"{len(still_running)} of {self.spec.clients} clients did not "
                f"return within {JOIN_TIMEOUT_S:.0f}s of the window closing")
        # A client that sent its spans and was THEN killed leaves no error and
        # a window that looks complete. Only the exit status says otherwise,
        # and the realistic case is a container's memory limit reaping a
        # client at the end of a long window: without this check the pool
        # would publish the partial window as a measurement of the engine.
        # Checked after the two above so the more specific message wins.
        #
        # EXCEPT a client THIS POOL killed at the span-transfer deadline,
        # which is not something that happened to the window. Two states, and
        # neither belongs here: if the transfer really was cut short, `absorb`
        # recorded a `ClientDied` and the `failures` branch above already
        # raised with the right message; if the payload landed complete while
        # the watchdog was firing, the spans are ENTIRE and the only thing
        # wrong is the signal this pool sent. Counting that as "ended
        # abnormally after reporting ... the spans it sent cover only part of
        # the window" would fail a sound window and blame the harness's own
        # SIGKILL on the engine. `Client.ended_by_this_pool` has the race.
        killed = [(w, w.exitcode) for w in self._workers
                  if w.exitcode not in (0, None) and not w.ended_by_this_pool]
        if killed:
            worker, code = killed[0]
            for each in self._workers:
                each.reap()
            raise PoolError(
                f"{len(killed)} of {self.spec.clients} client processes ended "
                f"abnormally after reporting; first: client {worker.name}, "
                f"{_exit_description(code)}. The spans it sent cover only "
                f"part of the window, so this is not a measurement.")

        merged = self._instruments.merge(
            [w.collector for w in self._workers], wall,
            cutoff=stop_at, window_start=wall_start)

        # The concurrency this pool actually achieved, over the same spans
        # and by the same `span_disposition` rule the merge above used. NOT
        # routed through `Instruments.merge`: that bundle exists so a test can
        # replace the merge to show what an untrimmed one counted, and a guard
        # computed by the replaced instrument would be calibrated against the
        # test's fixture rather than against the window.
        outstanding, served = window_concurrency(
            [w.collector for w in self._workers], wall,
            cutoff=stop_at, window_start=wall_start)

        # One dict across every client. The shards are disjoint and the
        # offsets are absolute positions, so no two clients can write the same
        # key and the merge cannot lose a result.
        #
        # Trimmed by the same predicate and the same window as the latencies,
        # so a recall reported beside this window's QPS is over the queries
        # that QPS counted. Letting the two diverge puts `queries_executed`
        # below `extra.recall_queries` for any window inside its first pass,
        # and `extra.query_reuse` -- executed over distinct -- below 1.0 for a
        # window in which nothing was repeated.
        results: Dict[int, Answer] = {}
        for worker in self._workers:
            for offset, answer in worker.results.items():
                if span_disposition(answer.started, answer.finished,
                                    wall_start, stop_at) is None:
                    results[offset] = answer

        for worker in self._workers:
            worker.reap()
        self._join_attendants()

        return PoolWindow(
            qps=merged.pop("qps"),
            inflight_discarded=merged.pop(INFLIGHT_DISCARDED),
            prewindow_discarded=merged.pop(PREWINDOW_DISCARDED),
            merged=merged,
            client_cpu_seconds=cpu_seconds,
            client_cpus=cpus,
            client_cpu_util=cpu_util,
            wall_s=wall,
            spec=self.spec,
            results=results,
            outstanding_concurrency=outstanding,
            served_concurrency=served,
        )

    def _first_error(self) -> Optional[BaseException]:
        """The first client error the parent knows about.

        **It does NOT wait for the attendants, and an earlier revision of
        this method did.** The reasoning for waiting was that an attendant
        might be one `poll` return away from naming the driver's own
        exception, which is worth more to an operator than 'barrier timed
        out'. It was wrong twice. An attendant that has not reported is
        blocked on its client's PIPE, not on the barrier, so breaking the
        barrier does not wake it and the wait runs to its full length every
        time. And the wait lands inside the failure path, where it delays the
        very timeout it is reporting: measured through
        `tests/test_t4b_readers.py`'s inverted barrier pair -- client 6.0 s,
        parent 1.5 s -- a one-second grace made the pool fail after **2.51 s**
        against a 1.5 s timeout, which is a `PoolError` whose message
        misstates its own cause by two thirds.

        Nothing is lost by not waiting, because the attendant assigns
        `worker.error` BEFORE it touches the barrier: `attend` sets it in the
        `try` and aborts or waits in the `finally`. So every error that could
        have broken the barrier is already readable here, and the only case
        that reaches the fallback is the one that has no error to name -- a
        client still warming up when the barrier timed out.
        """
        for worker in self._workers:
            if worker.error is not None:
                return worker.error
        return None

    def _join_attendants(self) -> None:
        """Collect the parent threads that stood in at the barrier.

        **One method rather than two copies, because the two call sites are
        the same repair and a copy is how one of them came to be missing.**
        `stop` has always joined them. `start`'s `BrokenBarrierError` path
        did not, and it raises `PoolError` instead of reaching `stop` -- so a
        pool that failed during warm-up left its attendants running, and
        `churn.py`'s three `except PoolError` sites go on to the next churn
        fraction and fork the next pool beside them.

        NOT joining them cost something real on the success path and the
        failure path is the same cost: a sweep builds one pool per window, so
        the next pool's `launch` forked while the previous pool's attendants
        were still being collected, which is a fork from a multi-threaded
        parent. CPython 3.12 warns about exactly that, and the warning is
        right to -- a child forked beside a live thread can inherit a lock
        that thread was holding, with nothing left to unlock it. Measured, a
        full suite run: 54 of those warnings before the `stop` join and 0
        after. `start`'s docstring has the rest of the argument.

        **Both callers call this AFTER reaping the clients, and the order is
        the whole reason this does not hang.** `_first_error`'s docstring has
        the measurement: an attendant that has not reported is blocked on its
        client's PIPE and not on the barrier, so aborting the barrier does not
        wake it and a wait placed before the reap runs to its full length --
        it made the pool fail after 2.51 s against a 1.5 s timeout. `reap`
        ends the client process, which closes the write end of that pipe, so
        the attendant's `poll` returns at once and this join is the
        milliseconds it takes the thread to finish. The timeout is kept
        anyway, and bounds it at `JOIN_TIMEOUT_S` like every other wait here:
        a join with no timeout is how a reader exception becomes a hang.
        """
        for attendant in self._attendants:
            attendant.join(timeout=JOIN_TIMEOUT_S)


__all__ = [
    "Answer", "BARRIER_TIMEOUT_S", "CLIENT_BARRIER_TIMEOUT_S",
    "CLIENT_BOUND_UTIL", "CLIENT_START_METHOD", "Client", "ClientDied",
    "ClientFailed", "ClientPool", "DEFAULT_INSTRUMENTS", "DriverFactory",
    "Instruments", "JOIN_TIMEOUT_S", "PoolError", "PoolSpec", "PoolWindow",
    "SCALING_CONCURRENCY_NOT_ACHIEVED", "SCALING_OK", "SCALING_STATUSES",
    "SERVED_CONCURRENCY_FLOOR", "Timeouts", "client_cpu", "shard_queries",
    "window_ratio",
]
