"""`served / clients` below a half is latency SKEW, not a serialized pool.

`metrics/latency.scaling_status` refused a scaling efficiency whenever
`served_concurrency` fell below half the client count, and the harness
publishes that refusal as "the clients serialized on the interpreter lock, or
the engine answers one query at a time". For a pool with no think time --
which is every concurrency point, since the client loop has no pause between
queries -- that condition is neither of those things. It is `p50 < mean / 2`,
and a two-mode service time puts it there while the pool scales perfectly.

**The arithmetic, which `TestTheConditionIsAStatementAboutLatencySkew` runs
rather than quotes.** `served` and `outstanding` are sums over the same kept
set, so `served / outstanding == p50 / mean` identically, and
`outstanding / clients` is 0.949-0.996 over everything this file rebuilds and
as low as 0.84 over the wider sweep behind it, whose low came from a CPU-bound
spinning thread arm at 8 clients that no test here reproduces. So
`served / clients` is `p50 / mean` to within that factor. For a
service time that is `f` with probability `1-p` and `s` otherwise the median
is `f` and the mean is `(1-p) f + p s`, so the refusal fires exactly when
`s/f > (1+p)/p` -- 4.3x at p=0.3, 11x at p=0.1, 21x at p=0.05. A graph cache
smaller than the graph is that distribution, and `CLAUDE.md` names cache
sizing as a first-class fairness concern for MHNSW and VIDX. The next run is
1M rows; the 60k-row smoke run that motivated the guard fits any cache, which
is why this did not show.

**MEASURED HERE, not supplied.** Through the real `ClientPool` in process
mode, 4 clients, synthetic drivers that sleep -- releasing the lock, no shared
queue, so the server is explicitly not the bottleneck:

    regime                        efficiency   served/clients   p50/mean
    narrow latency (control)           0.998            0.973      0.988
    30% of queries at 20x cost         0.984            0.163      0.164
    10% at 50x                         0.959            0.191      0.193
     5% at 50x (NOT rebuilt here)      1.000            0.327      0.331
    saturated server, narrow           0.277            0.989      0.994
    saturated AND 30% at 20x           0.233            0.155      0.156
    misses queue, 30% at 20x           0.299            0.050      0.051
    clients took turns (threads)       0.248            0.246      0.252

Rows two to four scaled LINEARLY and were all flagged
`concurrency_not_achieved`. Every row but the 5%-at-50x one is rebuilt by
`REGIMES` below, and that one is reported and not rebuilt: the closed form
already covers p=0.05 and an eighth regime costs 3 s of suite time for a third
miss rate. The table was taken at 2 s windows on a box whose twenty cores were
otherwise busy; the fixture rebuilds the shape at 0.5 s and asserts verdicts
and ratios rather than decimals, for the reason
`tests/test_g3_process_clients.py` gives.

**The repair is a second estimator of one quantity, not a second threshold.**
`scaling_efficiency` is `qps_n / (n x qps_1)`, and at one client a span holds
no queue and no lock, so `qps_1` is the reciprocal of the per-query CYCLE cost
-- service plus the client loop's bookkeeping, which `served_concurrency` at
one client measures as 0.965-0.992 rather than 1.0. Little's law then makes
`efficiency` the share of the clients in service, computed from the one-client
mean instead of the n-client median. The two are biased in largely disjoint
regimes -- skew drags the median one down, a server-side queue drags the
throughput one down -- and both only downwards, so the larger is the better
bound and the SAME floor of 0.5 judges it. Measured, the efficiency column is
0.96-1.00 where the pool scaled and 0.22-0.30 where it did not, so 0.5 is
again the midpoint of a measured gap -- a factor of 1.9 below the lowest
scaling row and 1.7 above the highest non-scaling one.

**THE EFFICIENCY TERM REACHES `clients >= 3` AND NOT BELOW, which is the
review's finding and not a design choice.** A completely serialized pool's
efficiency is exactly `1 / clients`, so at one and two clients it sits ON the
floor and cannot discriminate; admitting it there dropped a true positive the
old rule caught. `throughput_fraction` therefore withholds the term wherever
`1 / clients` is not itself below the floor -- and that withholds the FIX
there too, so the two-client point keeps the old rule's false positive until a
profile stops asking for it. `config/profiles/main.yml` sweeps two clients
today. `TestTwoClientsIsAKnifeEdgeForBothEstimators` measures both halves.

**ONE CLIENT IS HANDLED SEPARATELY AND IS NOW `ok` BY DEFINITION**, because
it is a different statement: a one-client pool has no concurrency to fail to
achieve. One client cannot take turns with itself, and "the engine answers one
query at a time" is not a defect at n=1 but the definition of n=1. That was a
live defect -- `clients == 1` is in every profile, and the 30%-at-20x regime
at a single client measured `served / clients` 0.1428, 0.1716, 0.1764, 0.1850
and 0.2050 over five builds, every one refused, so every run of a skewed
workload printed "the clients serialized" about one client while withholding
an efficiency that is 1.0 by definition.
`TestOneClientHasNoConcurrencyToFailToAchieve` asserts that it is the branch
and NOT the tautological efficiency that publishes those points.

**WHAT THIS DOES NOT FIX, asserted rather than admitted in prose.**
`TestTheResidueHoldsBothCausesAndCannotSeparateThem` measures a saturated
server whose service time is ALSO two-moded, and a server whose cache misses
queue on one shared resource. Both land at low efficiency AND low
`served / clients` -- the same quadrant as clients genuinely taking turns, on
both quantities the status is computed from. **One quantity OUTSIDE the status
does separate one of the two**, and claiming otherwise was wrong: `p50` grows
by a factor of 3.5 with the client count for the saturated regimes and not at
all for a stalled client. It is not promoted to a third term because the
miss-queue regime -- the realistic 1M-row shape, whose median query is a cache
hit that never reaches the bottleneck -- fails to scale with a FLAT median and
would be got wrong by it. `test_median_growth_sorts_one_residue_regime_and_
not_the_other` measures both halves. So no third status was added: the status
names the observation, both its causes are refused, and a value meaning
"withheld, cannot attribute" would be a synonym for the one already there.

**That is not the same as `report/` needing no change, which an earlier
version of this docstring claimed.** Keeping one status string means no new
value has to be rendered, and no more. `report/render.py`'s concurrency
footnote is FIXED TEXT saying that where the queries-in-service column is
below half the client count the efficiency cell publishes nothing -- which a
skewed-but-scaling point will now contradict on the page. The follow-up owes
that fix, in a file this change does not touch.

**Mutation discipline (R4).** Every claim below is checked by injecting the
regression at run time -- a monkeypatch, or a verbatim copy of the pre-change
rule run over the same window -- never by editing a file. Each probe asserts
that its mutation actually landed before reading it, and each docstring
carries the honest value and the mutant value. An identical pair is called out
as a dead probe rather than reported as a guard that holds.

**No engine is started here and no number in this file comes from one.**
"""

from __future__ import annotations

import ctypes
import ctypes.util
import multiprocessing
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.drivers.base import IndexSpec                            # noqa: E402
from harness.metrics import latency as latency_mod                    # noqa: E402
from harness.metrics import scaling as scaling_mod                    # noqa: E402
from harness.metrics.latency import (LatencyCollector,                # noqa: E402
                                     merge_collectors, window_concurrency)
# The scaling POLICY is `scaling.py`'s, not `latency.py`'s. Imported from its
# own module rather than through `latency`'s re-export so that the mutation
# probes below name the module whose globals the rule actually reads -- the
# asymmetry `tests/test_g1_client_concurrency.py` pins.
from harness.metrics.scaling import (CREDIBLE_BASELINE_MEDIAN_RATIO,  # noqa: E402
                                     SCALING_CONCURRENCY_NOT_ACHIEVED,
                                     SCALING_OK, SERVED_CONCURRENCY_FLOOR,
                                     achieved_fraction, baseline_is_credible,
                                     scaling_status, served_fraction,
                                     throughput_fraction)
from harness.workloads import clients as cl                           # noqa: E402
from tests.fake_driver import (FakeDriver, K, N_QUERIES, N_ROWS,      # noqa: E402
                               index_spec, tiny_dataset)
# The pool G3 retired. The only thing in the tree that can still make clients
# genuinely take turns, which is the true positive this file must not drop.
from tests.retired_thread_pool import thread_window                   # noqa: E402

MP = multiprocessing.get_context("fork")

#: The fast mode's wall cost, identical across every regime here. 0.5 ms for
#: the reason `tests/test_g1_client_concurrency.py:QUERY_S` records: well
#: below the interpreter's 5 ms switch interval, so a lock-holding query's
#: wait lands in the tail rather than in the median, and well above the pool's
#: own per-query bookkeeping.
QUERY_S = 0.0005

#: One measured window. Short, because the fixture sweeps five regimes at
#: three client counts twice; long enough that a one-client window of the
#: WIDEST regime completes over a hundred queries, so its median is the
#: driver's and not the box's.
WINDOW_S = 0.5

#: Windows per point, reduced by MAX qps. Foreign load can only depress a
#: window, and the question asked of each point is what the pool can achieve.
REPEATS = 2

#: Client counts the fixture sweeps. Two and four, because the defect and the
#: repair both have to hold at a count other than the one they were found at,
#: and because `config/profiles/main.yml` sweeps `[1, 2, 4, 8, 16, 32]`.
CLIENT_COUNTS = (1, 2, 4)

#: What every synthetic query returns. Fixed ids rather than the fake's own
#: neighbours, and load-bearing for the reason `test_g1`'s `SYNTHETIC_IDS`
#: gives: `FakeDriver.query` runs a NumPy scan and NumPy RELEASES the
#: interpreter lock inside it, which would land inside the span the instrument
#: sums and stop the lock-holding regime from holding the lock.
SYNTHETIC_IDS = list(range(K))

#: `usleep` through `ctypes.PyDLL`, which does NOT release the interpreter
#: lock around the call -- so four THREADS take turns while the call itself
#: costs almost no CPU. `tests/test_g3_process_clients.py:_HELD_USLEEP` has
#: the measurement that says why a CPU-free wait and not a spin: the spin's
#: verdict is about how many cores the box has spare, and this one's is about
#: the lock.
_LIBC_NAME = ctypes.util.find_library("c") or "libc.so.6"
_HELD_USLEEP = ctypes.PyDLL(_LIBC_NAME).usleep
_HELD_USLEEP.argtypes = [ctypes.c_uint]
_USLEEP_US = int(QUERY_S * 1e6)

_rngs: Dict[int, random.Random] = {}


def _rng() -> random.Random:
    """A generator per PROCESS, keyed by pid.

    The pool forks its clients, so every child inherits one `random` state
    and would draw the identical miss sequence. Not a correctness problem for
    an independent-clients model, but it would make the regimes' slow queries
    land in lockstep, which is a different experiment from the one described.
    """
    pid = os.getpid()
    generator = _rngs.get(pid)
    if generator is None:
        generator = random.Random(0x5EED ^ pid)
        _rngs[pid] = generator
    return generator


# ---------------------------------------------------------------------------
# The regimes
# ---------------------------------------------------------------------------

def regime_narrow() -> Callable[[], None]:
    """The control: one service time, the lock released for all of it."""
    return lambda: time.sleep(QUERY_S)


def regime_skew(p: float, penalty: float) -> Callable[[], None]:
    """A two-mode service time, the lock released for all of it.

    The cache-miss shape: `penalty` x the fast cost on a share `p` of
    queries. Nothing is shared between clients, so the pool MUST scale and
    any refusal over it is a false positive.
    """
    def spend() -> None:
        time.sleep(QUERY_S * (penalty if _rng().random() < p else 1.0))
    return spend


def regime_saturated() -> Callable[[], None]:
    """A server that IS the bottleneck, narrow service time.

    A shared FIFO: a query waits as long as there are queries in flight, and
    the client is blocked on it for all of it. With n clients looping each
    sees depth n, so throughput is flat and efficiency about 1/n -- the shape
    a reader calls an engine ceiling, and it is one. **This is the case the
    guard must not fire on**, and the case the repair's `max` exists for.

    The depth counter is a `multiprocessing.Value` created BEFORE any fork
    and not a `threading.Lock`: under fork each client would otherwise see
    its own queue at depth one, every query would cost the base latency, the
    pool would scale and the row would protect nothing. The same trap
    `tests/test_g1_client_concurrency.py:saturated` records being caught by.
    """
    depth = MP.Value("i", 0)

    def spend() -> None:
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        try:
            time.sleep(QUERY_S * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


def regime_saturated_skewed(p: float, penalty: float) -> Callable[[], None]:
    """The combination: a shared FIFO whose SERVICE time is two-moded.

    The case the repair cannot separate from a serialized pool, measured
    rather than argued. Same shared counter as `regime_saturated`, with this
    query's own draw as its cost, so the wait is the depth times a service
    time that has a tail.
    """
    depth = MP.Value("i", 0)

    def spend() -> None:
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        cost = QUERY_S * (penalty if _rng().random() < p else 1.0)
        try:
            time.sleep(cost * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


def regime_miss_queue(p: float, penalty: float) -> Callable[[], None]:
    """Cache HITS run free; only the MISSES queue on one shared resource.

    **The realistic 1M-row shape, and the regime the no-third-status argument
    rests on.** A graph cache smaller than the graph, with the misses
    contending for one thing: a hit costs the fast mode and never reaches the
    bottleneck, a miss costs `penalty` x fast x (misses in flight).

    It matters because it is the one regime in which the median span does NOT
    grow with the client count -- the median query is a HIT -- while the pool
    still fails to scale. `p50` growth therefore separates the saturated
    regimes from a stalled client and fails on this one, which is why it is
    not promoted to a third term in the rule.
    """
    depth = MP.Value("i", 0)

    def spend() -> None:
        if _rng().random() >= p:
            time.sleep(QUERY_S)
            return
        with depth.get_lock():
            depth.value += 1
            here = depth.value
        try:
            time.sleep(QUERY_S * penalty * here)
        finally:
            with depth.get_lock():
                depth.value -= 1
    return spend


def regime_held() -> Callable[[], None]:
    """Clients genuinely taking turns: a C call that holds the lock.

    The TRUE POSITIVE. Only serializes the retired THREAD pool -- separate
    interpreters do not share one lock -- so every reading of it below comes
    from `tests/retired_thread_pool.py`, which exists for exactly this.
    """
    return lambda: _HELD_USLEEP(_USLEEP_US)


#: The regimes, and which arm each is measured in. The lock-holding one has to
#: be the thread arm or it does not serialize at all; the rest are the real
#: pool, because what they claim is about the pool that ships.
REGIMES: Dict[str, Callable[[], Callable[[], None]]] = {
    "narrow": regime_narrow,
    "skew30x20": lambda: regime_skew(0.30, 20.0),
    "skew10x50": lambda: regime_skew(0.10, 50.0),
    "saturated": regime_saturated,
    "sat_skew30x20": lambda: regime_saturated_skewed(0.30, 20.0),
    "miss30x20": lambda: regime_miss_queue(0.30, 20.0),
    "held": regime_held,
}

#: Measured in the retired thread pool. One regime, and that is the point:
#: the live pool cannot be made to serialize through Python's own lock.
THREAD_ARM = ("held",)

#: Scaled linearly and has a skewed latency, so a refusal over it is the
#: false positive this file is about.
SCALED_BUT_SKEWED = ("skew30x20", "skew10x50")

#: The one of them the strict VERDICT is keyed to, and the choice is about
#: variance rather than about preference. Both scale; the efficiency of the
#: 10%-at-50x regime is estimated from a window in which only a tenth of the
#: queries are slow, so its one-client baseline is the noisiest number in the
#: table -- measured 0.959 over 2 s windows, 1.237 over 1 s, 0.789 over 0.6 s
#: and 0.747 in this fixture's own 0.5 s. The 30%-at-20x regime measured
#: 0.973-1.051 over every window length tried. The other row is
#: reported and its DEFECT SHAPE is asserted, which is the part that does not
#: depend on the box.
LOW_VARIANCE_SKEW = ("skew30x20",)

#: Did not scale, and the reason is the engine. Must not be refused.
ENGINE_CEILING = ("saturated",)

#: Did not scale and is skewed too: the residue, which holds both causes.
#: Both are measured, because they fail DIFFERENTLY -- the saturated one's
#: median grows with the client count and the miss-queue one's does not, and
#: the second is why `p50` growth cannot become a third term.
AMBIGUOUS = ("sat_skew30x20", "miss30x20")

#: The residue regime whose median DOES grow with the client count, and the
#: one whose median does not. Named so the pair cannot be mixed up by a
#: reader of the test that reports it.
RESIDUE_WITH_A_GROWING_MEDIAN = "sat_skew30x20"
RESIDUE_WITH_A_FLAT_MEDIAN = "miss30x20"

#: The counts at which a regime that fails to scale is UNAMBIGUOUSLY below
#: the floor on the throughput estimator -- which is the same condition
#: `throughput_fraction` admits the term under, `1 / clients < floor`, so this
#: tuple and that predicate move together if the floor moves.
#:
#: **Two clients is not one of them, and the first version of this comment was
#: WRONG about why.** It said the n=2 knife edge was "a property of the
#: PRE-EXISTING floor and not of this change". It is not: a completely
#: serialized pool's efficiency is exactly `1 / n`, so at two clients the
#: throughput estimator sits ON the floor and carries no information, while
#: the median-span estimator carries a systematic DOWNWARD bias -- its
#: `outstanding / clients` factor is always under 1.0 -- that put it below the
#: floor 3 times out of 3. Measured on the retired thread pool at two
#: clients: `served / clients` 0.4873 / 0.4919 / 0.4945 against efficiencies
#: of 0.4987 / 0.4913 / 0.4992, so a `max` over the two takes the larger and
#: lands on the floor. A reviewer of this change measured 0.5008 there and
#: PUBLISHED a pool whose clients were taking turns -- a true positive the old
#: rule caught, dropped by the repair. `throughput_fraction` now withholds the
#: term at `clients <= 2` for exactly that reason, and
#: `TestTwoClientsIsAKnifeEdgeForBothEstimators` measures the pair.
#: `config/profiles/main.yml` sweeps two clients today.
UNAMBIGUOUS_COUNTS = tuple(c for c in CLIENT_COUNTS if c >= 4)


# ---------------------------------------------------------------------------
# Fixtures: one table, measured once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    return tiny_dataset()


@pytest.fixture(scope="module")
def queries(dataset):
    return dataset.test[:N_QUERIES]


@pytest.fixture(scope="module")
def base(dataset):
    driver = FakeDriver(dim=dataset.dim, metric=dataset.metric, ef_search=32,
                        ef_half=64.0)
    driver.connect()
    driver.create_schema(index_spec(dataset))
    driver.load(dataset.train[:N_ROWS], list(range(N_ROWS)))
    return driver


def factory_for(base: FakeDriver, spend: Callable[[], None]):
    """A driver factory whose clones spend `spend()` and return fixed ids.

    Bound on the INSTANCE and not on a subclass: `FakeDriver.clone` returns a
    plain `FakeDriver`, so an override would not reach the clients the pool
    builds. Inside the timed region deliberately -- the client brackets
    `driver.query` alone and the spans are the whole input.
    """
    def clone():
        client = base.clone()

        def query(vector, k):
            spend()
            return SYNTHETIC_IDS[:k]

        client.query = query
        return client
    return clone


def _process_point(factory, queries, clients: int) -> Dict[str, float]:
    """One window of the real pool, reduced to the figures the rule reads."""
    pool = cl.ClientPool(factory, queries, clients, K, 32,
                         IndexSpec(dim=queries.shape[1], m=16,
                                   metric="euclidean"), warmup=2)
    pool.start(f"skew-probe {clients}-client warm-up", prefix="skew")
    time.sleep(WINDOW_S)
    window = pool.stop()
    return {"qps": window.qps, "served": window.served_concurrency,
            "outstanding": window.outstanding_concurrency,
            "p50": window.merged.get("latency_p50_ms"),
            "mean": window.merged.get("latency_mean_ms"),
            "clients": clients}


def _thread_point(factory, queries, clients: int) -> Dict[str, float]:
    """One window of the RETIRED thread pool, reduced the same way.

    Through the shared helper, so the only difference from the live pool is
    where the queries ran. The percentiles come from the real
    `merge_collectors` over the very spans the helper's `served` summed.
    """
    raw = thread_window(factory, queries, clients, K, WINDOW_S)
    merged = merge_collectors(raw["collectors"], raw["wall"],
                              cutoff=raw["cutoff"],
                              window_start=raw["window_start"])
    return {"qps": raw["qps"], "served": raw["served"],
            "outstanding": raw["outstanding"],
            "p50": merged.get("latency_p50_ms"),
            "mean": merged.get("latency_mean_ms"),
            "clients": clients}


def _best(arm, factory, queries, clients: int) -> Dict[str, float]:
    return max((arm(factory, queries, clients) for _ in range(REPEATS)),
               key=lambda point: point["qps"])


@pytest.fixture(scope="module")
def table(base, queries) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Every regime at every client count, measured once for the module.

    One fixture because the whole thing is one experiment: every regime pays
    the same fast-mode cost, and measuring the rows at different times under
    different foreign load is how two columns of one table come to disagree
    for a reason that is not the regime.
    """
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name, build in REGIMES.items():
        arm = _thread_point if name in THREAD_ARM else _process_point
        factory = factory_for(base, build())
        one = _best(arm, factory, queries, 1)
        row = {1: dict(one, efficiency=1.0)}
        for clients in CLIENT_COUNTS:
            if clients == 1:
                continue
            point = _best(arm, factory, queries, clients)
            point["efficiency"] = (point["qps"] / (clients * one["qps"])
                                   if one["qps"] else float("nan"))
            row[clients] = point
        out[name] = row
    return out


def fractions(point: Dict[str, float]) -> Dict[str, float]:
    """The three shares the rule and its evidence are expressed in."""
    clients = point["clients"]
    return {"served": point["served"] / clients,
            "outstanding": point["outstanding"] / clients,
            "efficiency": point["efficiency"],
            "p50_over_mean": point["p50"] / point["mean"]}


def report(table) -> str:
    """The table as a reader sees it, for a failure message."""
    lines = [f"{'regime':15s} {'n':>2s} {'eff':>7s} {'srv/n':>7s} "
             f"{'out/n':>7s} {'p50/mean':>9s}"]
    for name, row in table.items():
        for clients, point in sorted(row.items()):
            share = fractions(point)
            lines.append(
                f"{name:15s} {clients:2d} {share['efficiency']:7.3f} "
                f"{share['served']:7.3f} {share['outstanding']:7.3f} "
                f"{share['p50_over_mean']:9.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The pre-change rule, kept verbatim so every verdict can be compared
# ---------------------------------------------------------------------------

def pre_change_scaling_status(served: Optional[float],
                              clients: int) -> Optional[str]:
    """`scaling_status` as it stood at fbb48cd, copied body for body.

    R2: a repair has to be shown against the form it replaced, run over the
    same input. Calling the live function with no `efficiency` reaches the
    same arithmetic today, and that is checked below rather than assumed --
    but it is a property of the current default and not of the old file, so
    the old body is kept rather than relied upon.
    """
    fraction = served_fraction(served, clients)
    if fraction is None:
        return None
    if fraction < scaling_mod.SERVED_CONCURRENCY_FLOOR:
        return SCALING_CONCURRENCY_NOT_ACHIEVED
    return SCALING_OK


# ---------------------------------------------------------------------------
# The condition, as arithmetic
# ---------------------------------------------------------------------------

class TestTheConditionIsAStatementAboutLatencySkew:
    """`served / clients < 0.5` is `p50 < mean / 2`, over hand-built spans.

    No pool and no driver: which side of a floor a ratio falls on is a
    property of the arithmetic, and the end-to-end classes below measure the
    real thing.
    """

    @staticmethod
    def _collector(durations):
        """A collector holding spans of exactly `durations`, laid end to end
        so every one of them is inside the window."""
        collector = LatencyCollector(warmup=0)
        at = 0.0
        for seconds in durations:
            collector.add_span(at, at + seconds)
            at += seconds
        return collector

    @staticmethod
    def _two_mode(p, penalty, count=1000, fast=0.001):
        """`count` service times, a share `p` of them `penalty` x `fast`.

        Deterministic rather than drawn: the closed form below is about the
        proportion, and a sampled proportion would make the assertion about
        the seed.
        """
        slow = int(round(count * p))
        return [fast * penalty] * slow + [fast] * (count - slow)

    def test_served_over_outstanding_is_exactly_p50_over_mean(self):
        """The identity the whole defect rests on, read off one window.

        Both figures are sums over the SAME kept set, so the ratio is the
        median over the mean with nothing left over. Honest: 0.1644 for a
        30%-at-20x set, equal to `p50 / mean` to 1e-12. Mutant (`percentile`
        rebound on `harness.metrics.latency` to return the mean, which is the
        shape of a mean-latency implementation of `served`): 1.0 over the very
        same window -- so the two differ by a factor of six and the probe is
        not vacuous.

        **Stated as `served / outstanding` and not as `served / clients`,
        because only the first is exact.** `served / clients` is this ratio
        times `outstanding / clients`, which is 0.84-1.00 measured but is not
        1 by construction. The end-to-end classes carry that factor.

        Over whatever set the window KEEPS, which is the invariant that makes
        it exact: both figures are sums over one `span_disposition` call, so a
        span the cutoff trims leaves both of them. Measured unbounded here so
        the closed form below is the only thing the input has to satisfy; the
        trimming rule itself is `test_g1_client_concurrency.py`'s.

        Measured: `served / outstanding` 0.14925373134332912 against a
        `p50 / mean` of 0.14925373134332912 -- a difference of exactly 0.0 --
        and the PUBLISHED columns, which `as_dict` rounds to four decimals,
        give 0.14925373134328357, which is the closed form
        `1 / (1 - p + p s/f)` to the last digit.
        """
        durations = self._two_mode(0.30, 20.0)
        collector = self._collector(durations)
        wall = sum(durations)
        got = window_concurrency([collector], wall)
        merged = merge_collectors([collector], wall)
        ratio = got.served / got.outstanding
        # The unrounded quantities, off the very spans the collector holds.
        kept = sorted(sample.seconds for sample in collector._samples)
        exact = (latency_mod.percentile(kept, 0.50)
                 / (sum(kept) / len(kept)))
        assert ratio == exact
        published = merged["latency_p50_ms"] / merged["latency_mean_ms"]
        assert ratio == pytest.approx(published, abs=1e-12)
        assert published == pytest.approx(1.0 / (0.7 + 0.3 * 20.0), abs=1e-15)

        original = latency_mod.percentile
        try:
            latency_mod.percentile = (
                lambda samples, q: sum(samples) / len(samples))
            # The mutation landed. Four probes in the PR under review aborted
            # silently because what they replaced was ambiguous, so this is
            # asserted and not assumed.
            assert latency_mod.percentile is not original
            mutant = window_concurrency([collector], wall, cutoff=wall,
                                        window_start=0.0)
        finally:
            latency_mod.percentile = original
        assert latency_mod.percentile is original
        mutant_ratio = mutant.served / mutant.outstanding
        assert mutant_ratio == pytest.approx(1.0, rel=1e-9)
        assert mutant_ratio != pytest.approx(ratio, rel=1e-3)

    @pytest.mark.parametrize("p", [0.30, 0.10, 0.05])
    def test_the_old_rule_trips_at_the_penalty_the_closed_form_predicts(
            self, p):
        """`s/f > (1+p)/p`, with the verdict read on both sides of it.

        4.333x at p=0.3, 11x at p=0.1, 21x at p=0.05. The set is built with
        `outstanding == clients` exactly -- the spans tile the window and the
        client count is the number laid down -- so the factor the previous
        test warns about is 1 here and the condition is exactly `p50/mean`.

        Honest and mutant are the two SIDES: a penalty 10% under the
        predicted one reads `ok`, 10% over reads `concurrency_not_achieved`.
        A differing pair, so the boundary is where the closed form says and
        not somewhere the test would have accepted either way.
        """
        boundary = (1.0 + p) / p
        verdicts = {}
        for label, penalty in (("under", boundary * 0.9),
                               ("over", boundary * 1.1)):
            durations = self._two_mode(p, penalty)
            # One collector per client, each holding the same distribution,
            # so the window is `clients` spans deep throughout.
            clients = 4
            collectors = [self._collector(durations) for _ in range(clients)]
            wall = sum(durations)
            got = window_concurrency(collectors, wall, cutoff=wall,
                                     window_start=0.0)
            assert got.outstanding == pytest.approx(clients, rel=1e-9)
            verdicts[label] = pre_change_scaling_status(got.served, clients)
        assert verdicts["under"] == SCALING_OK, (p, verdicts)
        assert verdicts["over"] == SCALING_CONCURRENCY_NOT_ACHIEVED, (p,
                                                                     verdicts)
        assert verdicts["under"] != verdicts["over"]

    def test_the_new_rule_publishes_both_sides_when_the_pool_scaled(self):
        """The repair, over the same hand-built input.

        A 30%-at-20x set at four clients: honest (`efficiency` 1.0 supplied,
        which is what a linearly scaling pool measures) reads `ok`; mutant
        (the pre-change rule, same window) reads
        `concurrency_not_achieved`. Those are the two verdicts the defect is
        the difference between.
        """
        durations = self._two_mode(0.30, 20.0)
        clients = 4
        collectors = [self._collector(durations) for _ in range(clients)]
        wall = sum(durations)
        got = window_concurrency(collectors, wall, cutoff=wall,
                                 window_start=0.0)
        honest = scaling_status(got.served, clients, 1.0)
        mutant = pre_change_scaling_status(got.served, clients)
        assert honest == SCALING_OK
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert honest != mutant


# ---------------------------------------------------------------------------
# The repair's shape, with no measurement in it
# ---------------------------------------------------------------------------

class TestTheRepairIsOneFloorOverTwoEstimators:
    """`achieved_fraction` and what the rule does with it.

    Hand-built: these are properties of the comparison. The threshold is the
    one that was already there, and that is the claim -- a second constant
    would be a second thing to justify.
    """

    def test_the_fraction_is_the_larger_of_the_two_estimators(self):
        """`max`, and the two orderings are both exercised.

        Honest: 0.9 whichever estimator holds it. Mutant (`min`, the shape of
        a conjunction written as an intersection of refusals): 0.2 and 0.3 on
        the same two inputs, which is below the floor in both -- so the
        mutation moves the verdict and is not a dead probe.
        """
        skewed = achieved_fraction(0.8, 4, 0.9)       # served/n 0.2, eff 0.9
        queued = achieved_fraction(3.6, 4, 0.3)       # served/n 0.9, eff 0.3
        assert skewed == pytest.approx(0.9)
        assert queued == pytest.approx(0.9)
        assert min(0.2, 0.9) == pytest.approx(0.2)
        assert min(0.9, 0.3) == pytest.approx(0.3)
        assert scaling_status(0.8, 4, 0.9) == SCALING_OK
        assert scaling_status(3.6, 4, 0.3) == SCALING_OK

    def test_a_min_rule_would_refuse_both_and_is_run_to_show_it(self):
        """The mutation inside `scaling_status`, not beside it.

        `scaling_status` resolves `achieved_fraction` through this module's
        globals on every call, so rebinding it reaches the rule. Honest: `ok`
        for a skewed-but-scaling window and `ok` for a queued one. Mutant
        (`achieved_fraction` rebound to the `min` form):
        `concurrency_not_achieved` for both.
        """
        original = scaling_mod.achieved_fraction
        try:
            scaling_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: min(
                    served_fraction(served, clients), efficiency))
            assert scaling_mod.achieved_fraction is not original
            mutant_skewed = scaling_status(0.8, 4, 0.9)
            mutant_queued = scaling_status(3.6, 4, 0.3)
        finally:
            scaling_mod.achieved_fraction = original
        assert scaling_mod.achieved_fraction is original
        assert scaling_status(0.8, 4, 0.9) == SCALING_OK
        assert scaling_status(3.6, 4, 0.3) == SCALING_OK
        assert mutant_skewed == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert mutant_queued == SCALING_CONCURRENCY_NOT_ACHIEVED

    def test_the_floor_is_the_one_that_was_already_there(self):
        """No second constant, and the rule reads the live one.

        Honest (floor 0.5): a window at `served / clients` 0.2 and efficiency
        0.45 is refused. Mutant (floor rebound to 0.4 **on
        `harness.metrics.scaling`**): the same window reads `ok`. The rule
        looks the floor up in its own module globals on every call, so the
        rebinding lands; a floor captured in a default argument would make
        this probe dead.
        """
        original = scaling_mod.SERVED_CONCURRENCY_FLOOR
        honest = scaling_status(0.8, 4, 0.45)
        try:
            scaling_mod.SERVED_CONCURRENCY_FLOOR = 0.4
            assert scaling_mod.SERVED_CONCURRENCY_FLOOR != original
            mutant = scaling_status(0.8, 4, 0.45)
        finally:
            scaling_mod.SERVED_CONCURRENCY_FLOOR = original
        assert scaling_mod.SERVED_CONCURRENCY_FLOOR == original
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert mutant == SCALING_OK
        assert honest != mutant

    def test_an_unmeasured_window_stays_none_even_with_an_efficiency(self):
        """None is "nobody could look", which an efficiency cannot repair.

        A window that counted no query has nothing in it to have been in
        service, and an efficiency over it divides a throughput that was never
        measured. Honest: None. Mutant (`achieved_fraction` rebound to fall
        back to the efficiency, which is the obvious reading of "take
        whichever you have"): `ok` -- a window nobody measured published as a
        pass, which is the fault `PoolWindow.client_bound` draws the same
        distinction to avoid.
        """
        honest = scaling_status(None, 4, 1.0)
        original = scaling_mod.achieved_fraction
        try:
            scaling_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: (
                    served_fraction(served, clients)
                    if served is not None else efficiency))
            assert scaling_mod.achieved_fraction is not original
            mutant = scaling_status(None, 4, 1.0)
        finally:
            scaling_mod.achieved_fraction = original
        # Restoration, asserted and not left to the `finally`. The two calls
        # below reach the module-level import binding rather than this
        # module's global, so they would pass with the mutant still in place
        # and cannot stand in for this line.
        assert scaling_mod.achieved_fraction is original
        assert honest is None
        assert mutant == SCALING_OK
        assert honest != mutant
        assert achieved_fraction(None, 4, 1.0) is None
        assert achieved_fraction(1.0, 0, 1.0) is None

    def test_omitting_the_efficiency_is_byte_for_byte_the_old_rule(self):
        """The migration state, asserted so it cannot drift.

        Two callers in `harness/workloads/` still compute this status without
        an efficiency, so the two-argument form has to keep its old verdicts
        exactly -- otherwise this change moves published numbers in a path
        nobody routed. Checked across the floor rather than at one point.

        Honest and mutant here are the two FORMS over one input set, and they
        are deliberately IDENTICAL: that is the claim. The differing pair for
        the same inputs is in the test above, where an efficiency is supplied.
        """
        for served in (0.0, 0.5, 1.9, 2.0, 2.1, 3.9, 4.0, 8.0):
            assert (scaling_status(served, 4)
                    == pre_change_scaling_status(served, 4)), served
        assert scaling_status(None, 4) is pre_change_scaling_status(None, 4)
        # And the new argument is what makes the difference, on an input the
        # two-argument form refuses.
        assert scaling_status(0.8, 4) == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert scaling_status(0.8, 4, 0.99) == SCALING_OK

    def test_throughput_fraction_is_the_efficiency_where_it_is_evidence(self):
        """Named rather than computed, and withheld rather than rescaled.

        `scaling_efficiency` IS a concurrency fraction -- `qps_n / qps_1` is
        Little's law with the one-client cycle cost, divided by n -- so a
        function that RESCALED it would be claiming otherwise. What it does
        instead is return `None` where the reading is not evidence.
        """
        assert throughput_fraction(0.25, 4) == 0.25
        assert throughput_fraction(None, 4) is None
        assert achieved_fraction(0.8, 4, 0.9) == pytest.approx(
            max(served_fraction(0.8, 4), throughput_fraction(0.9, 4)))

    @pytest.mark.parametrize("clients,admitted", [(1, False), (2, False),
                                                  (3, True), (4, True),
                                                  (8, True), (32, True)])
    def test_the_term_is_withheld_where_one_over_clients_is_not_below_the_floor(
            self, clients, admitted):
        """The n=2 repair, as the predicate rather than as a verdict.

        A COMPLETELY serialized pool delivers one client's throughput, so its
        efficiency is exactly `1 / clients`. At one and two clients that is
        not below the floor of 0.5, so the reading cannot distinguish total
        serialization from perfect concurrency and is withheld. From three
        clients up, `1 / clients` is 0.333 or less and the reading separates.

        Honest: `throughput_fraction(0.9, 2)` is None and
        `throughput_fraction(0.9, 3)` is 0.9. Mutant: the pre-repair form,
        which returned the efficiency at every client count, is the identity
        -- so the pair below is None against 0.9 at two clients.
        """
        got = throughput_fraction(0.9, clients)
        mutant = 0.9           # the pre-repair body: `return efficiency`
        assert (got == 0.9) is admitted, (clients, got)
        if not admitted:
            assert got is None
            assert got != mutant
        else:
            assert got == mutant

    def test_the_boundary_moves_with_the_floor_and_is_not_a_hard_coded_two(
            self):
        """Derived from the floor, read back by moving the floor.

        Honest (floor 0.5): the term is withheld at two clients and admitted
        at three. Mutant (floor rebound to 0.3 on
        `harness.metrics.scaling`): withheld at three and admitted at four,
        because `1/3` is 0.333 which is no longer below it. A `clients > 2`
        spelling would read 0.9 at three clients under both floors, which is
        the dead-probe signature this names.
        """
        original = scaling_mod.SERVED_CONCURRENCY_FLOOR
        honest = (throughput_fraction(0.9, 2), throughput_fraction(0.9, 3),
                  throughput_fraction(0.9, 4))
        try:
            scaling_mod.SERVED_CONCURRENCY_FLOOR = 0.3
            assert scaling_mod.SERVED_CONCURRENCY_FLOOR != original
            mutant = (throughput_fraction(0.9, 2), throughput_fraction(0.9, 3),
                      throughput_fraction(0.9, 4))
        finally:
            scaling_mod.SERVED_CONCURRENCY_FLOOR = original
        assert scaling_mod.SERVED_CONCURRENCY_FLOOR == original
        assert honest == (None, 0.9, 0.9)
        assert mutant == (None, None, 0.9)
        assert honest != mutant

    def test_a_superlinear_reading_is_withheld_rather_than_clamped(self):
        """A share of the clients in service cannot exceed 1.

        Honest (ceiling 2.0): an efficiency of 1e308 or of `inf` is withheld,
        so a window at `served / clients` 0.2 is refused on the median alone.
        Mutant (`CREDIBLE_EFFICIENCY_CEILING` rebound to `inf`, which is the
        pre-repair behaviour of admitting any finite reading): the same window
        reads `ok` -- a corrupt baseline publishing a point whose own spans
        say one query in five was in service.

        **Clamping is not the repair and the difference is read back here**:
        `min(efficiency, 1.0)` would turn both of these into a confident 1.0,
        the most permissive value the rule has.
        """
        original = scaling_mod.CREDIBLE_EFFICIENCY_CEILING
        honest = (scaling_status(0.8, 4, 1e308),
                  scaling_status(0.8, 4, float("inf")))
        try:
            scaling_mod.CREDIBLE_EFFICIENCY_CEILING = float("inf")
            assert scaling_mod.CREDIBLE_EFFICIENCY_CEILING != original
            mutant = (scaling_status(0.8, 4, 1e308),
                      scaling_status(0.8, 4, float("inf")))
        finally:
            scaling_mod.CREDIBLE_EFFICIENCY_CEILING = original
        assert scaling_mod.CREDIBLE_EFFICIENCY_CEILING == original
        assert honest == (SCALING_CONCURRENCY_NOT_ACHIEVED,) * 2
        assert mutant == (SCALING_OK,) * 2
        assert honest != mutant
        # The clamp, run rather than described.
        assert scaling_status(0.8, 4, min(1e308, 1.0)) == SCALING_OK
        # And a legitimate reading just under the ceiling still counts, so
        # the ceiling is not refusing everything above 1.0. The highest
        # legitimate reading measured behind this file was 1.327.
        assert throughput_fraction(1.327, 4) == 1.327
        assert throughput_fraction(2.0, 4) == 2.0
        assert throughput_fraction(2.0001, 4) is None

    def test_a_nan_efficiency_degrades_to_the_old_rule(self):
        """The argument order of `max` is load-bearing, and this reads it.

        `qps_n / (clients * qps_1)` is NaN when the baseline window completed
        nothing, and every comparison against a NaN is False -- so
        `max(fraction, nan)` returns the FRACTION and the median-span rule
        decides. The ceiling does not catch it, because `nan > ceiling` is
        False too.

        Honest: a NaN efficiency on a window at `served / clients` 0.2 reads
        `concurrency_not_achieved`, exactly as the old rule did. Mutant
        (`achieved_fraction` rebound with the arguments to `max` REVERSED):
        the NaN propagates, `NaN < floor` is False, and the window reads
        `ok` -- which would publish every point in a run whose baseline
        measured nothing.
        """
        nan = float("nan")
        honest = scaling_status(0.8, 4, nan)
        original = scaling_mod.achieved_fraction
        try:
            scaling_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: max(
                    throughput_fraction(efficiency, clients),
                    served_fraction(served, clients)))
            assert scaling_mod.achieved_fraction is not original
            mutant = scaling_status(0.8, 4, nan)
        finally:
            scaling_mod.achieved_fraction = original
        assert scaling_mod.achieved_fraction is original
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert mutant == SCALING_OK
        assert honest != mutant
        # And the honest path really did take the fraction, not the NaN.
        assert achieved_fraction(0.8, 4, nan) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# The false positive, through the real pool
# ---------------------------------------------------------------------------

class TestAPoolThatScaledIsNotRefusedForBeingSkewed:
    """The defect and the repair, measured end to end at 2 and 4 clients.

    Verdicts and ratios against the floor rather than decimals, because this
    box is shared -- the same discipline `tests/test_g3_process_clients.py`
    adopts and for the same reason.
    """

    @pytest.mark.parametrize("regime", LOW_VARIANCE_SKEW)
    @pytest.mark.parametrize("clients", UNAMBIGUOUS_COUNTS)
    def test_the_old_rule_refused_it_and_the_new_one_does_not(
            self, table, regime, clients):
        """Honest (new rule) against mutant (the pre-change body), one window.

        Measured at 4 clients and 2 s windows: 30%-at-20x reads efficiency
        0.984 with `served / clients` 0.163, and 10%-at-50x 0.959 with 0.191.
        Both scaled -- nothing is shared between these clients -- and both
        were refused on the median alone.

        The two verdicts must differ, which is what makes this a repair
        rather than a restatement.
        """
        point = table[regime][clients]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the median-span estimator no longer "
            f"falls under the floor, so this probe is not measuring the "
            f"defect it claims\n{report(table)}")
        assert share["efficiency"] > SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the pool did not scale, so it is not "
            f"the false-positive regime\n{report(table)}")
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = pre_change_scaling_status(point["served"], clients)
        assert honest == SCALING_OK, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest != mutant

    @pytest.mark.parametrize("regime", SCALED_BUT_SKEWED)
    def test_the_throughput_really_rose_with_the_client_count(self, table,
                                                              regime):
        """Non-vacuity: the pool scaled, and not only its efficiency column.

        A verdict that moved because the efficiency was mismeasured would be
        the instrument loosened rather than the defect removed. Honest: qps
        at four clients is more than twice qps at one. Mutant: the same
        comparison at one client against itself, which is 1.0 by construction
        and is the reading a broken baseline would give.
        """
        one = table[regime][1]
        four = table[regime][4]
        honest = four["qps"] / one["qps"]
        mutant = one["qps"] / one["qps"]
        assert honest > 2.0, (regime, honest, report(table))
        assert mutant == pytest.approx(1.0)
        assert honest != pytest.approx(mutant, rel=0.1)

    @pytest.mark.parametrize("regime", SCALED_BUT_SKEWED)
    def test_every_skewed_regime_has_the_defect_shape(self, table, regime):
        """The part of the claim that does not depend on the box.

        For BOTH skewed regimes, including the noisier one the verdict is not
        keyed to: the pool's throughput rose with the client count AND its
        `served / clients` is under the floor. That conjunction is the defect
        -- a pool that scaled, refused -- and it holds whatever the efficiency
        estimator's variance is, because the throughput comparison is a ratio
        of counts.

        Honest: the pair above. Mutant (the narrow control over the same two
        quantities): it scaled too, and its `served / clients` is ABOVE the
        floor, so the refusal is attributable to the skew and to nothing else
        about these drivers.
        """
        point = table[regime][4]
        share = fractions(point)
        scaled = point["qps"] / table[regime][1]["qps"]
        assert scaled > 2.0, (regime, scaled, report(table))
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, report(table)
        control = fractions(table["narrow"][4])
        control_scaled = table["narrow"][4]["qps"] / table["narrow"][1]["qps"]
        assert control_scaled > 2.0, report(table)
        assert control["served"] > SERVED_CONCURRENCY_FLOOR, report(table)
        assert pre_change_scaling_status(point["served"], 4) != \
            pre_change_scaling_status(table["narrow"][4]["served"], 4)

    def test_the_control_is_not_refused_by_either_rule(self, table):
        """The narrow-latency regime, so the probe above is about the skew.

        Honest and mutant are the two RULES, and they are deliberately the
        same here: a control that moved would mean the repair had changed
        something other than the skewed case. Called out as such, because an
        identical pair is otherwise the signature of a dead probe.
        """
        for clients in (c for c in CLIENT_COUNTS if c > 1):
            point = table["narrow"][clients]
            share = fractions(point)
            assert share["served"] > SERVED_CONCURRENCY_FLOOR, report(table)
            assert share["efficiency"] > SERVED_CONCURRENCY_FLOOR, (
                report(table))
            new = scaling_status(point["served"], clients,
                                 share["efficiency"])
            old = pre_change_scaling_status(point["served"], clients)
            assert new == old == SCALING_OK, report(table)


# ---------------------------------------------------------------------------
# The trap: a low efficiency is not on its own a refusal
# ---------------------------------------------------------------------------

class TestAnEngineCeilingIsStillPublished:
    """The saturated server, which has a LOW efficiency and must not be
    refused.

    The whole reason the repair is a `max` over two estimators and not a
    second threshold on the efficiency: that column is 0.277 at four clients
    for a server that genuinely is the bottleneck -- indistinguishable from
    the true positive's 0.248 -- and what saves it is `served / clients`
    staying at its client count, because a real queue raises the median along
    with everything else.
    """

    @pytest.mark.parametrize("regime", ENGINE_CEILING)
    @pytest.mark.parametrize("clients", UNAMBIGUOUS_COUNTS)
    def test_the_queue_keeps_its_median_at_the_client_count(
            self, table, regime, clients):
        """Honest (the rule) against mutant (the efficiency term alone).

        Measured at four clients: efficiency 0.277, `served / clients` 0.989,
        `p50 / mean` 0.994, and the median span rose from 0.576 ms at one
        client to 2.075 ms at four -- a factor of 3.6, about the client count,
        which is what a queue does and a stalled client does not.

        Honest: `ok`. Mutant (the efficiency fed through the served slot, so
        the rule judges it alone): `concurrency_not_achieved` -- a real engine
        ceiling refused, which the floor's docstring names as worse than the
        defect.
        """
        point = table[regime][clients]
        share = fractions(point)
        assert share["efficiency"] < SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the saturated regime scaled, so it is "
            f"not modelling an engine ceiling\n{report(table)}")
        assert share["served"] > SERVED_CONCURRENCY_FLOOR, (
            f"{regime} at {clients}: the queue did not raise the median, so "
            f"this row is not the one the floor protects\n{report(table)}")
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = scaling_status(share["efficiency"] * clients, clients)
        assert honest == SCALING_OK, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest != mutant

    def test_the_median_grew_with_the_client_count(self, table):
        """The mechanism behind the row above, read off the same windows.

        Honest: the saturated regime's median span at four clients is more
        than twice its median at one. Mutant: the same ratio for the
        lock-holding regime, where the median did not move -- measured 0.558
        to 0.559 ms. The pair is what "a real queue raises the whole
        distribution" means, and it is the only quantity measured here that
        tells the two apart.
        """
        honest = (table["saturated"][4]["p50"] / table["saturated"][1]["p50"])
        mutant = table["held"][4]["p50"] / table["held"][1]["p50"]
        assert honest > 2.0, (honest, report(table))
        assert mutant < 1.5, (mutant, report(table))
        assert honest != pytest.approx(mutant, rel=0.2)


# ---------------------------------------------------------------------------
# The true positive, which must survive the repair
# ---------------------------------------------------------------------------

class TestClientsTakingTurnsAreStillRefused:
    """The tripwire, through the retired thread pool.

    `concurrency_not_achieved` is close to unreachable since the clients
    became OS processes, so the guard is a tripwire rather than an expected
    finding -- and a tripwire nothing can trip is indistinguishable from a
    broken one. `tests/retired_thread_pool.py` is the one thing in the tree
    that can still trip it, and the repair has to leave it tripping.
    """

    @pytest.mark.parametrize("clients", UNAMBIGUOUS_COUNTS)
    def test_both_estimators_are_low_so_the_refusal_stands(self, table,
                                                           clients):
        """Honest (new rule) and mutant (pre-change rule), deliberately EQUAL.

        Measured in the thread arm: efficiency and `served / clients` are the
        same number at every client count -- at 2 s windows, 0.499/0.494 at
        two clients, 0.248/0.246 at four and 0.124/0.123 at eight (two and
        eight are measured but not asserted here; see
        `UNAMBIGUOUS_COUNTS`) -- because flat throughput and
        a flat median make `served` a constant and `1/n` both figures. So the
        larger of the two is still under the floor and the verdict does not
        move.

        An identical pair is normally a dead probe. It is the deliverable
        here, and the differing pair that proves the input is not inert is in
        the next test.
        """
        point = table["held"][clients]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, (
            f"the thread arm no longer serializes at {clients} clients, so "
            f"this file is not measuring the positive it claims"
            f"\n{report(table)}")
        assert share["efficiency"] < SERVED_CONCURRENCY_FLOOR, (
            f"the thread arm's throughput scaled at {clients} clients"
            f"\n{report(table)}")
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = pre_change_scaling_status(point["served"], clients)
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest == mutant

    def test_a_mismeasured_efficiency_is_what_would_drop_it(self, table):
        """What the new term costs if the baseline it divides by is wrong.

        The efficiency is a ratio across TWO windows, and this repository's
        `window_ratio` exists because ratios of separately measured
        quantities have published two wrong results here. Honest (the measured
        efficiency, about a quarter at four clients):
        `concurrency_not_achieved`. Mutant (efficiency 1.0, which is what a
        one-client baseline depressed by foreign load would produce): `ok` --
        the true positive dropped. Run so the failure mode is recorded rather
        than described, and so the report can name the mitigation.
        """
        point = table["held"][4]
        share = fractions(point)
        honest = scaling_status(point["served"], 4, share["efficiency"])
        mutant = scaling_status(point["served"], 4, 1.0)
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert mutant == SCALING_OK
        assert honest != mutant

    def test_the_pool_delivered_one_clients_throughput_and_not_more(
            self, table):
        """Non-vacuity for the row above: the throughput really is flat.

        Honest: four lock-holding threads deliver less than 1.5x one thread.
        Mutant: the control regime over the same arm is not available in this
        table, so the comparison is against the same regime at one client,
        which is 1.0 -- and the honest figure has to sit near it rather than
        near four. Both read off the fixture.
        """
        honest = table["held"][4]["qps"] / table["held"][1]["qps"]
        assert honest < 1.5, (honest, report(table))
        assert honest > 0.5, (honest, report(table))


# ---------------------------------------------------------------------------
# What the repair does NOT fix
# ---------------------------------------------------------------------------

class TestTwoClientsIsAKnifeEdgeForBothEstimators:
    """At one and two clients the throughput estimator is withheld; at two
    that leaves the median-span rule deciding alone.

    **THE FIRST VERSION OF THIS FILE CALLED THIS "a property of the floor as
    it already stood, surfaced by this work rather than created by it", AND
    THAT WAS WRONG.** A pool that serializes completely delivers one client's
    throughput however many clients it has, so its efficiency is exactly
    `1 / n` -- 0.5 at two clients, ON the floor. The old rule's quantity is
    not symmetric with it: `served / clients` is `(p50/mean) x
    (outstanding/clients)` and that second factor is ALWAYS under 1.0, so the
    old rule carried a systematic downward bias that put a serialized
    two-client pool below the floor 3 times out of 3 -- measured
    `served / clients` 0.4873 / 0.4919 / 0.4945 and 0.4949 / 0.4912 / 0.4943
    over two sets of three builds, by margins of 0.0055 to 0.0127. Replacing
    the deciding quantity with an unbiased one centred on 0.5 therefore turns
    a reliable refusal into a coin flip, and a reviewer of this change
    measured 0.5008 and published a pool whose clients were taking turns.

    So `throughput_fraction` withholds the term wherever `1 / clients` is not
    itself below the floor. That is symmetric and it CUTS BOTH WAYS: at TWO
    clients the skew false positive this file exists to fix is also not fixed,
    because the term that fixes it is not admitted there. The term reaches
    `clients >= 3`.

    **One client is not in that trade**, and the asymmetry is the point:
    there the withholding costs nothing, because a separate branch in
    `scaling_status` returns `ok` for a reason that has nothing to do with
    either estimator -- a one-client pool has no concurrency to fail to
    achieve. See `TestOneClientHasNoConcurrencyToFailToAchieve`. So the
    verdicts run ok / refused / ok / ok at n=1,2,3,4, which is measured below
    rather than asserted in prose.
    """

    def test_the_verdict_at_every_client_count_the_withholding_touches(self):
        """The reach of the fix, as the verdicts themselves.

        A skewed-but-SCALING pool -- `served / clients` 0.17, efficiency 0.98,
        the 30%-at-20x regime's measured shape -- reads:

            n=1   ok        by DEFINITION: no concurrency to fail to achieve
            n=2   refused   the term is withheld, the old rule decides
            n=3   ok        the term is admitted and the repair applies
            n=4   ok

        **The asymmetry between n=1 and n=2 is the whole content of this
        probe, and the two `ok`s have different causes.** At one client the
        efficiency is withheld as well -- `1/1` is not below the floor -- so
        the `ok` comes from the one-client branch and not from the term; at
        two the withholding leaves the old rule in charge and it refuses.

        Honest: the four verdicts above. Mutant (the pre-review form, which
        admitted the efficiency at every client count): `ok` at all four,
        including at two, where a serialized pool is indistinguishable from
        this one. The pair differs at exactly n=2.
        """
        skewed = {}
        unguarded = {}
        for clients in (1, 2, 3, 4):
            served = 0.17 * clients
            skewed[clients] = scaling_status(served, clients, 0.98)
            unguarded[clients] = (
                SCALING_OK if max(0.17, 0.98) >= SERVED_CONCURRENCY_FLOOR
                else SCALING_CONCURRENCY_NOT_ACHIEVED)
        assert skewed == {1: SCALING_OK,
                          2: SCALING_CONCURRENCY_NOT_ACHIEVED,
                          3: SCALING_OK, 4: SCALING_OK}, skewed
        assert unguarded == {c: SCALING_OK for c in (1, 2, 3, 4)}, unguarded
        assert skewed != unguarded
        # Where the term is withheld AND there is a concurrency question to
        # ask -- which is n=2 and not n=1 -- the verdict is exactly the old
        # rule's, so the withholding costs nothing beyond the status quo.
        assert (skewed[2] == pre_change_scaling_status(0.34, 2)
                == SCALING_CONCURRENCY_NOT_ACHIEVED)
        # And at one client the two rules DISAGREE, which is the next class.
        assert skewed[1] != pre_change_scaling_status(0.17, 1)

    def test_a_serialized_two_client_pool_is_still_refused(self, table):
        """The true positive the review found dropped, read off the fixture.

        Honest (the term withheld at two clients): the thread arm's
        two-client window is `concurrency_not_achieved`, which is what the
        old rule said. Mutant (the term admitted there, which is this
        change's pre-review form): the verdict is whichever of the two
        estimators happened to come out larger, and the supplied reviewer
        reading of efficiency 0.5008 against `served / clients` 0.4898
        publishes it.

        The mutant is run on the SUPPLIED pair as well as on the measured
        one, because the measured one does not always cross -- my own three
        builds read efficiencies of 0.4987, 0.4913 and 0.4992, all just
        under. A probe that only used the local reading would come back
        identical and prove nothing, so it is reported both ways.
        """
        point = table["held"][2]
        share = fractions(point)
        honest = scaling_status(point["served"], 2, share["efficiency"])
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest == pre_change_scaling_status(point["served"], 2)
        # The pre-review form over the same window: `max` of the two shares
        # against the floor, with no withholding.
        local = max(share["served"], share["efficiency"])
        local_mutant = (SCALING_OK if local >= SERVED_CONCURRENCY_FLOOR
                        else SCALING_CONCURRENCY_NOT_ACHIEVED)
        # SUPPLIED by the reviewer of this change, not measured here.
        supplied = max(0.4898, 0.5008)
        supplied_mutant = (SCALING_OK if supplied >= SERVED_CONCURRENCY_FLOOR
                           else SCALING_CONCURRENCY_NOT_ACHIEVED)
        assert supplied_mutant == SCALING_OK
        assert honest != supplied_mutant
        if local_mutant == honest:
            # A dead probe on the local reading, and said so rather than
            # left to look like a guard that held: this build's efficiency
            # came out under the floor on its own.
            assert local < SERVED_CONCURRENCY_FLOOR, (local, report(table))

    def test_the_two_regimes_are_not_separated_at_two_clients(self, table):
        """The pair, read off the fixture and asserted only as an overlap.

        Measured at this fixture's 0.5 s windows: the lock-holding thread arm
        at two clients read efficiency 0.4913-0.4998 and `served / clients`
        0.4873-0.4949 over six builds; the saturated server read 0.538 and
        0.985. So the THROUGHPUT estimator does not separate them at two
        clients -- the readings are within 10% of each other and straddle the
        floor -- while the MEDIAN one still does, by a factor of two.

        Honest: that overlap on the efficiency column. Mutant: the same two
        regimes at four clients, where the efficiency readings are 0.248 and
        0.278 -- still overlapping, which is the point the `max` exists for --
        and the median column separates them 0.246 against 0.989. Both pairs
        read back, so the probe reports the structure rather than a verdict.
        """
        turns_two = fractions(table["held"][2])
        ceiling_two = fractions(table["saturated"][2])
        gap = ceiling_two["efficiency"] / turns_two["efficiency"]
        # Measured 1.082 (0.538 against 0.497). The bound is wide because
        # both halves are near 0.5 by construction and a short window on a
        # busy box moves either of them; what is being asserted is that they
        # do not separate, and a factor of 1.6 either way is still no
        # separation.
        assert 0.625 < gap < 1.6, (gap, report(table))
        assert (ceiling_two["served"] / turns_two["served"]) > 1.6, (
            report(table))
        turns_four = fractions(table["held"][4])
        ceiling_four = fractions(table["saturated"][4])
        assert (ceiling_four["efficiency"] / turns_four["efficiency"]) < 2.0, (
            report(table))
        assert (ceiling_four["served"] / turns_four["served"]) > 2.0, (
            report(table))

    def test_a_completely_serialized_pool_sits_on_the_floor_at_two_clients(
            self):
        """The arithmetic behind it, with no measurement in it.

        Honest (`1/n` at two clients, the exact serialization prediction, on
        both estimators): `ok` -- 0.5 is not BELOW the floor, and the floor is
        documented as a `<`. Mutant (a hair under, which is what the thread
        arm actually measured): `concurrency_not_achieved`. So which verdict a
        perfectly serialized two-client pool gets is decided by noise, under
        the old rule and the new one alike.
        """
        honest = scaling_status(1.0, 2, 0.5)
        mutant = scaling_status(0.999, 2, 0.4995)
        assert honest == SCALING_OK
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert honest != mutant
        # And the old rule agrees with both, so this is not something the
        # efficiency term introduced.
        assert pre_change_scaling_status(1.0, 2) == SCALING_OK
        assert (pre_change_scaling_status(0.999, 2)
                == SCALING_CONCURRENCY_NOT_ACHIEVED)


class TestOneClientHasNoConcurrencyToFailToAchieve:
    """`clients == 1` is `SCALING_OK` whenever the window measured anything.

    **Not a special case bolted on, and not the efficiency's accident.** There
    is no true positive at one client: one client cannot take turns with
    itself, so "the clients serialized on the interpreter lock" is impossible,
    and "the engine answers one query at a time" is not a defect at n=1 -- it
    is the definition of n=1. Whatever `served / clients` comes out as at one
    client is a statement about the latency distribution and nothing else.

    It is reached as that statement rather than through the efficiency, which
    is 1.0 by construction at one client: `throughput_fraction` withholds any
    reading that `1 / clients` does not put below the floor, and `1 / 1` does
    not, so the term is withheld here too. A right answer from a constructed
    value would not be evidence.

    **This was a live defect, not a hypothetical.** `clients == 1` is in every
    profile -- `smoke`, `percona` and `dev` among them -- so every run of a
    skewed workload printed "the clients serialized" about a single client and
    withheld a scaling efficiency that is 1.0 by definition.
    """

    def test_a_skewed_one_client_window_is_published(self, table):
        """Honest against mutant on the measured one-client window.

        Honest (the new rule): `ok`. Mutant (the pre-change body, same
        window): `concurrency_not_achieved`. Measured `served / clients` at
        one client for the 30%-at-20x regime: 0.1428, 0.1716, 0.1764, 0.1850
        and 0.2050 over five builds, every one of them a refusal before.

        The efficiency is passed as None as well as as 1.0, because the two
        callers in `harness/workloads/` pass nothing and this branch has to
        fire for them too -- it is the one part of this change that is NOT
        waiting on plumbing.
        """
        point = table["skew30x20"][1]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, (
            f"the one-client skewed window is no longer under the floor, so "
            f"this probe is not measuring the defect\n{report(table)}")
        honest_with = scaling_status(point["served"], 1, 1.0)
        honest_without = scaling_status(point["served"], 1)
        mutant = pre_change_scaling_status(point["served"], 1)
        assert honest_with == honest_without == SCALING_OK, report(table)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest_with != mutant

    def test_it_is_the_branch_and_not_the_efficiency_that_publishes_it(self):
        """The term is withheld at one client, so the `ok` is the branch's.

        Honest: `throughput_fraction(1.0, 1)` is None -- the efficiency plays
        no part -- and `scaling_status(0.17, 1, 1.0)` is still `ok`. Mutant
        (the one-client branch removed, which is the pre-change shape of the
        rule): the same call reads `concurrency_not_achieved`, because the
        withheld term leaves `served_fraction` of 0.17 against the floor.

        So the two readings differ, and the difference is attributable to the
        branch rather than to the efficiency -- which is the distinction the
        whole class exists to make.
        """
        assert throughput_fraction(1.0, 1) is None
        honest = scaling_status(0.17, 1, 1.0)
        # The rule without the branch, run over the same input.
        fraction = achieved_fraction(0.17, 1, 1.0)
        mutant = (SCALING_CONCURRENCY_NOT_ACHIEVED
                  if fraction < SERVED_CONCURRENCY_FLOOR else SCALING_OK)
        assert fraction == pytest.approx(0.17)
        assert honest == SCALING_OK
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED
        assert honest != mutant

    def test_a_one_client_window_that_measured_nothing_is_still_none(self):
        """`None` wins over the branch, and that ordering is asserted.

        "Nobody could look" is not "the pool was fine", and a one-client
        window that counted no query is the first of those. Honest: None.
        Mutant (the branch moved ABOVE the None check, which is the obvious
        way to write it): `ok` -- a window nobody measured published as a
        pass, which is the fault `PoolWindow.client_bound` draws the same
        distinction to avoid.
        """
        honest = scaling_status(None, 1, 1.0)
        mutant = SCALING_OK          # the branch, taken before the None check
        assert honest is None
        assert honest != mutant
        assert scaling_status(None, 1) is None
        # And a one-client window that DID measure something is not None, so
        # the probe is not passing because everything at n=1 is None.
        assert scaling_status(0.17, 1) == SCALING_OK


class TestAnUnderWarmedBaselineIsTheRepairsOwnExposure:
    """What a depressed `qps_1` does to the term, and the signal that sees it.

    **A hazard this change creates and cannot fix from here.** The efficiency
    is a ratio across TWO windows, and `harness/workloads/concurrency.py` sets
    `warmup` PER CLIENT -- so the one-client baseline gets the fewest warmed
    queries of any point in the sweep, reaches cache steady state slowest in
    wall clock, and is measured FIRST, right after load. A baseline depressed
    by a factor k inflates every efficiency in the sweep by about k, and
    `CREDIBLE_EFFICIENCY_CEILING` is no defence because the inflated value is
    nowhere near it.

    Measured here, with the n-client window held FIXED and only the baseline's
    per-query cost scaled, against the retired thread pool at four clients
    whose clients genuinely take turns:

        baseline     efficiency   p50_1 / p50_n   status
        honest            0.248           0.996   concurrency_not_achieved
        2.0x cold         0.468           1.889   concurrency_not_achieved
        2.5x cold         0.579           2.338   ok          <- PUBLISHED
        3.0x cold         0.690           2.782   ok

    So the crossing is at about 2.1x, and past it a pool whose clients are
    taking turns publishes a scaling efficiency. The fix belongs in the
    workload's warm-up, not here.

    **The signal that sees it is already a published column.** `p50_1 / p50_n`
    above 1 means the median query got FASTER as clients were added, which no
    engine can do: a queue raises the median and a stalled client leaves it
    alone, so nothing legitimate lowers it. `latency_p50_ms` is recorded at
    every point of the sweep, so a caller could pass it with no schema change
    -- it is simply not an input to this module today.
    """

    @pytest.fixture(scope="class")
    def cold_baseline(self, base, queries):
        """A one-client window at TWICE the per-query cost.

        The model of an under-warmed baseline: same driver, same arm, same
        window length, a cold per-query cost. Two windows, best by qps, like
        every other point in this file.
        """
        factory = factory_for(base, lambda: time.sleep(QUERY_S * 2.0))
        return _best(_thread_point, factory, queries, 1)

    def test_a_cold_baseline_inflates_the_efficiency_towards_the_floor(
            self, table, cold_baseline):
        """Honest baseline against under-warmed, over ONE n-client window.

        The four-client window is the fixture's, unchanged, so the only thing
        that differs between the two readings is the baseline -- which is what
        makes this a measurement of the warm-up and not of two runs.

        Honest (the fixture's own baseline): efficiency about a quarter, and
        the window is refused. Mutant (the 2x cold baseline): the efficiency
        roughly doubles. Both read back. The verdict does not necessarily move
        at 2x -- measured 0.468 against a floor of 0.5, a 6% margin -- and
        that near-miss is the finding, so the assertion is on the INFLATION
        rather than on a verdict the box's noise would decide.
        """
        four = table["held"][4]
        honest = four["qps"] / (4.0 * table["held"][1]["qps"])
        mutant = four["qps"] / (4.0 * cold_baseline["qps"])
        assert honest < SERVED_CONCURRENCY_FLOOR, (honest, report(table))
        assert mutant > 1.6 * honest, (honest, mutant, report(table))
        assert honest != pytest.approx(mutant, rel=0.1)
        # And the inflation is entirely in the baseline: the n-client window
        # is the same object, so `served / clients` cannot have moved.
        assert four["served"] / 4.0 < SERVED_CONCURRENCY_FLOOR, report(table)

    def test_the_median_that_got_faster_is_what_flags_it(self, table,
                                                         cold_baseline):
        """The detector, as the pair it has to separate.

        Honest (the fixture's baseline): `p50_1 / p50_n` is about 1.0 --
        measured 0.996 -- because a stalled client leaves the median alone.
        Mutant (the 2x cold baseline): 1.889. A median that got faster as
        clients were added, which no engine can do.
        """
        four = table["held"][4]
        honest = table["held"][1]["p50"] / four["p50"]
        mutant = cold_baseline["p50"] / four["p50"]
        assert honest < 1.3, (honest, report(table))
        assert mutant > 1.5, (mutant, report(table))
        assert mutant > 1.3 * honest, (honest, mutant, report(table))

    def test_baseline_is_credible_reads_the_two_windows_that_way(
            self, table, cold_baseline):
        """`baseline_is_credible`, over the very windows above.

        Honest (the fixture's own baseline against the four-client window):
        True. Mutant (the 2x cold baseline against the same four-client
        window): False. Both read back off measured medians rather than off
        hand-picked numbers, so the function is exercised on the shape it
        exists for.

        **Nothing in `harness/` calls this yet.** It is a handover: the
        caller is `concurrency.py`, which holds both medians already as
        `latency_p50_ms`. This probe is what makes it a tested handover
        rather than a paragraph.
        """
        four = table["held"][4]
        honest = baseline_is_credible(table["held"][1]["p50"], four["p50"])
        mutant = baseline_is_credible(cold_baseline["p50"], four["p50"])
        assert honest is True, report(table)
        assert mutant is False, report(table)
        assert honest != mutant

    def test_the_ratio_threshold_sits_in_the_gap_it_was_measured_from(self):
        """Either side of 1.4, with no measurement in it.

        The gap it is the midpoint of: legitimate readings of this ratio ran
        0.97-1.02 over every regime and client count the fixture rebuilds,
        and the lowest corrupt reading measured was 1.889. Honest (1.4): 1.02
        is credible and 1.889 is not. Mutant (the constant rebound to 2.0 on
        `harness.metrics.scaling`): 1.889 reads credible too -- and 1.889 is
        the row whose efficiency came within 6% of publishing a serialized
        pool, so that is the miss that matters.
        """
        original = scaling_mod.CREDIBLE_BASELINE_MEDIAN_RATIO
        honest = (baseline_is_credible(1.02, 1.0),
                  baseline_is_credible(1.889, 1.0))
        try:
            scaling_mod.CREDIBLE_BASELINE_MEDIAN_RATIO = 2.0
            assert scaling_mod.CREDIBLE_BASELINE_MEDIAN_RATIO != original
            mutant = (baseline_is_credible(1.02, 1.0),
                      baseline_is_credible(1.889, 1.0))
        finally:
            scaling_mod.CREDIBLE_BASELINE_MEDIAN_RATIO = original
        assert scaling_mod.CREDIBLE_BASELINE_MEDIAN_RATIO == original
        assert honest == (True, False)
        assert mutant == (True, True)
        assert honest != mutant
        # The constant is read out of module globals on every call, so the
        # rebinding above reached the function rather than a captured copy.
        assert CREDIBLE_BASELINE_MEDIAN_RATIO == original
        # Measured rows, read through the function rather than described.
        assert baseline_is_credible(0.996, 1.0) is True
        assert [baseline_is_credible(r, 1.0)
                for r in (1.889, 2.338, 2.782)] == [False, False, False]

    def test_a_missing_median_is_none_and_never_false(self):
        """None is "nobody could look"; False would say "it was corrupt".

        Honest: None for a missing or non-positive median either side.
        Mutant (the obvious `bool(...)` spelling, which turns every absence
        into a verdict): False, which a consumer would render as a warning
        about a baseline nobody measured.
        """
        for pair in ((None, 1.0), (1.0, None), (None, None),
                     (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
            honest = baseline_is_credible(*pair)
            assert honest is None, pair
            assert honest is not False, pair
        # Non-vacuity: a measured pair does return a bool.
        assert baseline_is_credible(1.0, 1.0) is True


class TestTheResidueHoldsBothCausesAndCannotSeparateThem:
    """A saturated server whose service time is also two-moded.

    **This is the reject inside the repair, and it is measured rather than
    conceded.** Such a server has a low efficiency -- it IS the bottleneck --
    and a low `served / clients` -- its service time has a tail -- so it lands
    in the same quadrant as clients genuinely taking turns and within that
    regime's measured range on every quantity here. The `max` does not rescue
    it and nothing available in one window would.

    That is not a mislabel: `concurrency_not_achieved` names the OBSERVATION
    and both its causes are refused, and "the engine answers one query at a
    time" is true of this regime. It is why no third status was added -- a
    value meaning "withheld, cannot attribute" would be a synonym for the one
    that is already there, and would have to be plumbed through `report/` to
    say nothing new.
    """

    @pytest.mark.parametrize("regime", AMBIGUOUS)
    @pytest.mark.parametrize("clients", UNAMBIGUOUS_COUNTS)
    def test_the_combination_is_refused_by_both_rules(self, table, regime,
                                                      clients):
        """Honest and mutant are the two RULES and are deliberately EQUAL.

        Measured at four clients in this fixture: efficiency 0.260 and
        `served / clients` 0.166, against the lock-holding arm's 0.248 and
        0.246; at 2 s windows, 0.233 and 0.155 against the same pair. The
        repair
        leaves this verdict exactly where it was, which is the honest outcome
        for a case one window cannot attribute -- and it means the repair
        never flags anything the old rule did not.
        """
        point = table[regime][clients]
        share = fractions(point)
        assert share["served"] < SERVED_CONCURRENCY_FLOOR, report(table)
        assert share["efficiency"] < SERVED_CONCURRENCY_FLOOR, report(table)
        honest = scaling_status(point["served"], clients, share["efficiency"])
        mutant = pre_change_scaling_status(point["served"], clients)
        assert honest == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)
        assert honest == mutant

    @pytest.mark.parametrize("regime", AMBIGUOUS)
    def test_the_quantities_the_status_reads_do_not_separate_it(self, table,
                                                                regime):
        """**On every quantity the STATUS IS COMPUTED FROM**, not on every
        quantity -- and the narrowing is the review's, correctly.

        The status reads two numbers: `served / clients` and the efficiency.
        Both of them are on the same side of the floor for the two residue
        regimes as for clients genuinely taking turns, and the efficiency
        columns overlap within a factor of two. Measured at four clients:
        efficiency 0.233 and 0.217 for the two residue regimes against 0.248
        for the true positive.

        `outstanding / clients` is included because it is the quantity a
        reviewer reaches for next, and it is blind -- 0.93-1.00 everywhere.
        `p50` growth is NOT included, because it is not blind and it does
        separate one of these two: that is the next test's business.

        Honest: the overlap. Mutant: the saturated NARROW regime, whose
        `served / clients` is a factor of two or more from the true
        positive's, so the overlap is a property of the skew and not of the
        measurement.
        """
        turns = fractions(table["held"][4])
        both = fractions(table[regime][4])
        ceiling = fractions(table["saturated"][4])
        for key in ("served", "efficiency"):
            assert both[key] < SERVED_CONCURRENCY_FLOOR, report(table)
            assert turns[key] < SERVED_CONCURRENCY_FLOOR, report(table)
        # Measured at four clients: 0.233/0.248 = 0.94 for the saturated-and-
        # skewed regime and 0.217/0.248 = 0.88 for the miss-queue one. The
        # bound asserts the OVERLAP rather than a decimal.
        ratio = both["efficiency"] / turns["efficiency"]
        assert 0.4 < ratio < 2.5, (regime, ratio, report(table))
        # Blind, which is the claim: measured 0.949-0.996 over 7 regimes x
        # 3 client counts x 2 builds of this very fixture, and as low as 0.84
        # for a CPU-bound spinning thread arm at 8 clients that no test here
        # reproduces. The bound is deliberately far wider than either, since
        # the only thing that has to hold is that this quantity never comes
        # near the floor and so can never be a term.
        assert 0.8 < both["outstanding"] < 1.05, report(table)
        assert 0.8 < turns["outstanding"] < 1.05, report(table)
        separated = ceiling["served"] / turns["served"]
        assert separated > 2.0, (separated, report(table))

    def test_median_growth_sorts_one_residue_regime_and_not_the_other(
            self, table):
        """Why `p50` growth is reported beside the status and not inside it.

        A queue moves the median and a stalled client does not, so
        `p50@4 / p50@1` separates a saturated server from clients taking
        turns. Measured over five builds: x3.659-x3.726 for the saturated
        server and x3.624-x3.660 for the saturated-and-skewed one, against
        x0.994-x1.013 for the thread arm. **So a third term on that growth
        WOULD sort the residue regime this file used to build alone** -- the
        first version of this file said the median "failed to grow in two of
        the three", naming a regime it did not build, and was wrong about the
        two it did.

        It is still not promoted, and this is the measurement that says why:
        the miss-queue regime fails to scale with a median that does NOT grow
        -- measured x1.020, x0.998 and x1.002 -- because its median query is a
        cache hit that never reaches the bottleneck. Honest and mutant here
        are the two residue regimes' growth over one fixture: a differing
        pair, which is the point.
        """
        def growth(name):
            return table[name][4]["p50"] / table[name][1]["p50"]

        queued = growth(RESIDUE_WITH_A_GROWING_MEDIAN)
        flat = growth(RESIDUE_WITH_A_FLAT_MEDIAN)
        turns = growth("held")
        assert queued > 2.0, (queued, report(table))
        assert flat < 1.6, (flat, report(table))
        assert turns < 1.6, (turns, report(table))
        # The pair that refuses the third term: one residue regime is a
        # factor of two clear of the stalled client on this quantity and the
        # other is indistinguishable from it.
        assert queued / turns > 2.0, (queued, turns, report(table))
        assert 0.5 < flat / turns < 2.0, (flat, turns, report(table))

    def test_the_repair_never_refuses_what_the_old_rule_published(self,
                                                                  table):
        """The whole table, as one claim: the change is one-directional.

        A refusal the old rule did not make would be a new false positive,
        and this is the only probe that would catch one in a regime nobody
        thought to classify. Honest: every point either keeps its verdict or
        moves from refused to `ok`. Mutant (`achieved_fraction` rebound to the
        `min` form): points move the other way, and the count of them is
        read back so the probe cannot pass by finding nothing to compare.
        """
        def verdicts(rule):
            out = {}
            for name, row in table.items():
                for clients, point in row.items():
                    share = fractions(point)
                    out[(name, clients)] = (
                        rule(point["served"], clients, share["efficiency"]),
                        pre_change_scaling_status(point["served"], clients))
            return out

        honest = verdicts(scaling_status)
        loosened = [key for key, (new, old) in honest.items()
                    if new != old]
        tightened = [key for key, (new, old) in honest.items()
                     if new == SCALING_CONCURRENCY_NOT_ACHIEVED
                     and old == SCALING_OK]
        assert tightened == [], (tightened, report(table))
        assert loosened, (
            f"no verdict moved at all, so this probe compared nothing"
            f"\n{report(table)}")

        original = scaling_mod.achieved_fraction
        try:
            scaling_mod.achieved_fraction = (
                lambda served, clients, efficiency=None: min(
                    served_fraction(served, clients), efficiency)
                if served is not None else None)
            assert scaling_mod.achieved_fraction is not original
            mutant = verdicts(scaling_status)
        finally:
            scaling_mod.achieved_fraction = original
        assert scaling_mod.achieved_fraction is original
        mutant_tightened = [key for key, (new, old) in mutant.items()
                            if new == SCALING_CONCURRENCY_NOT_ACHIEVED
                            and old == SCALING_OK]
        assert mutant_tightened, (
            f"the min form refused nothing the old rule published, so this "
            f"mutation measured nothing\n{report(table)}")


# ---------------------------------------------------------------------------
# Ops version 9: `outstanding` is the primary figure
# ---------------------------------------------------------------------------

def regime_lognormal(sigma: float, cap: float) -> Callable[[], None]:
    """A HEAVY continuous tail, the lock released for all of it.

    The shape `full-20260825` measured on MariaDB 12.3 at 32 clients -- p50
    34 ms, p99 1,043 ms -- rather than the two-mode cache-miss model above.
    `exp(sigma**2 / 2)` is mean over median, 3.1 at sigma 1.5; capped so one
    draw cannot outlast the window. Nothing is shared between clients, so
    the pool MUST scale.
    """
    def spend() -> None:
        draw = min(cap, _rng().lognormvariate(0.0, sigma))
        time.sleep(QUERY_S * draw)
    return spend


class TestTheMeanBasedFigureIsThePrimaryOne:
    """F2 item 1-2: `scaling_status(..., outstanding=)` refuses on the exact
    mean-based occupancy, and a heavy tail is a NOTE (`tail_dominated`).

    Measured through the real process pool, because the claim is that the
    lock blindness that made the median necessary is gone from the pool
    that ships.
    """

    @pytest.fixture(scope="class")
    def lognormal(self, base, queries):
        factory = factory_for(base, regime_lognormal(1.5, 60.0))
        one = _best(_process_point, factory, queries, 1)
        four = _best(_process_point, factory, queries, 4)
        four["efficiency"] = four["qps"] / (4 * one["qps"])
        return {1: one, 4: four}

    def test_a_heavy_tail_at_full_concurrency_is_ok_and_noted(self,
                                                              lognormal):
        """Honest: `ok` + `tail_dominated`. Mutant (the pre-change rule, the
        median alone): `concurrency_not_achieved` -- the false positive the
        report published as "the clients serialized"."""
        four = lognormal[4]
        assert four["outstanding"] / 4 > 0.8, four
        assert four["served"] / four["outstanding"] < 0.5, four
        honest = scaling_status(four["served"], 4, four["efficiency"],
                                outstanding=four["outstanding"])
        assert honest == SCALING_OK, four
        assert scaling_mod.scaling_note(four["served"], four["outstanding"]) \
            == scaling_mod.SCALING_TAIL_DOMINATED
        # Without an efficiency term too: a low median is never a refusal on
        # its own under the version-9 rule.
        assert scaling_status(four["served"], 4,
                              outstanding=four["outstanding"]) == SCALING_OK
        mutant = pre_change_scaling_status(four["served"], 4)
        assert mutant == SCALING_CONCURRENCY_NOT_ACHIEVED, four

    def test_the_note_is_not_a_status(self):
        """A note never appears among the statuses a report treats as a
        verdict, and a narrow window carries no note."""
        assert (scaling_mod.SCALING_TAIL_DOMINATED
                not in scaling_mod.SCALING_STATUSES)
        assert scaling_mod.scaling_note(3.9, 4.0) is None
        assert scaling_mod.scaling_note(None, 4.0) is None
        assert scaling_mod.tail_dominated(1.0, 0.0) is None

    @pytest.mark.parametrize("regime", ("held", "sat_skew30x20"))
    def test_a_serialising_pool_is_still_refused(self, table, regime):
        """The true positives survive the move to `outstanding`.

        `held` is clients genuinely taking turns (the retired thread pool --
        its `outstanding` stays at the client count, the lock wait sitting
        inside the spans) and `sat_skew30x20` is ONE SERVER SLOT, a shared
        FIFO, whose service time has a tail. Both deliver one client's
        throughput and keep the median low, so both in-service estimators are
        under the floor and the residue refusal holds.
        """
        point = table[regime][4]
        status = scaling_status(point["served"], 4, point["efficiency"],
                                outstanding=point["outstanding"])
        assert status == SCALING_CONCURRENCY_NOT_ACHIEVED, report(table)

    def test_a_pool_whose_clients_were_not_in_flight_is_refused(self):
        """The case the client clock measures EXACTLY: hand-built spans
        covering 1.2 of four clients' worth of the window. Refused whatever
        the efficiency says."""
        clients = []
        for offset in range(4):
            collector = LatencyCollector()
            # Each client in flight 30% of a 1 s window.
            collector.add_span(offset * 0.01, offset * 0.01 + 0.3)
            clients.append(collector)
        measured = window_concurrency(clients, 1.0)
        assert measured.outstanding == pytest.approx(1.2)
        assert scaling_status(measured.served, 4, 1.0,
                              outstanding=measured.outstanding) \
            == SCALING_CONCURRENCY_NOT_ACHIEVED

    def test_a_saturated_server_is_still_published(self, table):
        """The engine ceiling: outstanding at the client count, efficiency
        about 1/n. Must stay `ok`."""
        point = table["saturated"][4]
        assert scaling_status(point["served"], 4, point["efficiency"],
                              outstanding=point["outstanding"]) == SCALING_OK
