"""Whether a scaling efficiency measured over a client pool is reportable.

**This module exists because `metrics/latency.py` ran out of room, and that
is the same reason `latency.py` holds the quantity this policy judges.** The
policy was written in `harness/workloads/clients.py`, which stood at 846
lines against this repository's 800-line ceiling with it inlined; it moved to
`latency.py` beside `window_concurrency`, the thing it reads. Three rounds of
measured rationale later -- a floor, a credibility ceiling, a baseline
detector and the arguments for all three -- `latency.py` stood at 844 lines
and the same ceiling applied again. `pool_window.py` and `churn_readers.py`
were split out of `clients.py` for exactly this, and the alternative here was
deleting measurements to make room, which in a file whose whole subject is
measurements is the wrong trade.

So `latency.py` keeps the QUANTITY -- `window_concurrency`, the spans, the
percentiles -- and this module keeps the POLICY: the floor, the two statuses,
the two estimators of the share of clients in service, and the rule over
them. The division is the one the floor's own docstring already drew.

**ONE CONSEQUENCE HAS TO BE WRITTEN DOWN, because this repository has been
caught by it three times.** `latency.py` re-exports every name below so that
`harness/workloads/` and `report/` keep reaching them where they always did.
A module that re-exports a name does NOT share it: rebinding
`latency.SERVED_CONCURRENCY_FLOOR` reaches nothing, because `scaling_status`
reads this module's globals. A mutation probe aimed at `latency` is therefore
DEAD, and `tests/test_g1_client_concurrency.py` had one aimed exactly there
-- repointed in the same commit as this split, because a split that silently
neutered a mutation probe would be worse than the ceiling breach it fixed.
`pool_window.CLIENT_BOUND_UTIL` carries the same warning for the same reason.

Nothing here imports anything from this package, let alone `orchestrator`:
the policy is arithmetic over two numbers a caller already has.
"""

from __future__ import annotations

from typing import Optional

#: Share of the client count that must have been IN SERVICE before a scaling
#: efficiency is reportable as a property of the engine. Below it the harness
#: records a status instead and the report refuses to render the figure.
#: `scaling_status` below says what the statuses mean;
#: `latency.WindowConcurrency` says why `served` and not the exact
#: `outstanding` is compared.
#:
#: 0.5 IS MEASURED, and the measurement is the GAP it sits in. Five synthetic
#: drivers of identical 0.325 ms query cost through the real `ClientPool` at 1
#: and 4 clients, three 2 s windows each, median window
#: (CPython 3.12.3; `docs/05-methodology.md` §7 has the full table and
#: `tests/test_g1_client_concurrency.py` the reproducible form):
#:
#:     driver                     efficiency   served/clients   flagged?
#:     releases the lock                1.00             0.96   no
#:     saturated server                 0.28             0.98   no
#:     holds the lock (two forms)  0.25-0.27        0.22-0.25   YES
#:     half released, half held         0.52             0.96   no
#:     any driver at 1 client           1.00        0.94-1.00   no
#:
#: Two populations, 0.22-0.25 and 0.94-1.00, with nothing measured between:
#: 0.5 is a factor of two from each side and no plausible move of it changes
#: a verdict. The midpoint of a measured gap is the whole justification.
#:
#: **The saturated-server row is the one this has to get right.** That driver
#: IS the bottleneck -- latency rises with the client count, throughput is
#: flat and efficiency falls to 0.28, the shape a reader would call an engine
#: ceiling, and it is one. Its median span rose with it, 0.389 ms to
#: 1.377 ms, so its `served` stayed at the client count and it is NOT
#: flagged. A real queue raises the whole distribution; a lock stall raises
#: only the tail.
#:
#: **THAT LAST SENTENCE IS TRUE OF THOSE FIVE AND FALSE IN GENERAL, WHICH IS
#: WHY `efficiency` WAS ADMITTED TO THE RULE BELOW.** All five have a NARROW
#: service time, so for all five `served / clients` and `p50 / mean` are the
#: same number, mechanically: `served / outstanding == p50 / mean` identically
#: (both are sums over one kept set) and `outstanding / clients` is 0.949-0.996
#: over everything `tests/test_skewed_latency_is_not_serialization.py`
#: rebuilds, 0.84 at worst over the wider sweep behind this work. So
#: `served/clients` below a half IS `p50` below half the mean -- a claim about
#: LATENCY SKEW, not about how many clients ran. A two-mode service time with
#: miss rate p and penalty s/f drives it under the floor whenever
#: `s/f > (1+p)/p`: 4.3x at p=0.3, 11x at p=0.1, 21x at p=0.05, which is what
#: a graph cache smaller than the graph produces.
#:
#: Measured on this tree through the real `ClientPool`, 4 clients, 2 s
#: windows, better of two by qps, drivers that SLEEP -- releasing the lock, no
#: shared queue, so the server is explicitly not the bottleneck (CPython
#: 3.12.3; the lock-holding row is the retired thread pool, the only arm that
#: can still produce it). Every row but the fourth is rebuilt by
#: `tests/test_skewed_latency_is_not_serialization.py`; the 5%-at-50x row is
#: reported and not rebuilt, the closed form already covering p=0.05:
#:
#:     regime                        efficiency   served/clients   p50/mean
#:     narrow latency (control)           0.998            0.973      0.988
#:     30% of queries at 20x cost         0.984            0.163      0.164
#:     10% at 50x                         0.959            0.191      0.193
#:      5% at 50x (not rebuilt)           1.000            0.327      0.331
#:     saturated server, narrow           0.277            0.989      0.994
#:     saturated AND 30% at 20x           0.233            0.155      0.156
#:     misses queue, 30% at 20x           0.299            0.050      0.051
#:     clients took turns (threads)       0.248            0.246      0.252
#:
#: Rows two to four SCALED LINEARLY and were flagged `concurrency_not_achieved`
#: on `served / clients` alone -- published as "the clients serialized, or the
#: engine answers one query at a time", neither of which was true of them. The
#: column that holds is `efficiency`: 0.96-1.00 where the pool scaled and
#: 0.23-0.30 where it did not, nothing between at this window length, so the
#: same 0.5 is again the midpoint of a measured gap -- 1.9 below the lowest
#: scaling row, 1.7 above the highest non-scaling one. ONE floor judges both
#: estimators; there is no second constant to choose. A floor at 0.35 would
#: publish a pool that delivered 40% of its clients' throughput, the refusal's
#: own definition of not achieving it; at 0.75 the 10%-at-50x row's 0.959 is
#: refused on noise, and over 0.6 s windows it measured 0.789.
SERVED_CONCURRENCY_FLOOR = 0.5

#: The largest `scaling_efficiency` still read as a MEASUREMENT of the share
#: of clients in service; above it `throughput_fraction` withholds and the
#: median-span estimator decides alone. A share of clients in service cannot
#: exceed 1, so a reading above 1 is sampling noise up to some point and a
#: corrupt one-client baseline beyond it; the value cannot say which, so what
#: is chosen is where to stop treating it as evidence. **Clamping to
#: `min(efficiency, 1.0)` is the wrong repair** -- it turns an arbitrarily
#: broken baseline into a confident 1.0, the rule's most permissive value.
#:
#: 2.0 IS MEASURED, and the measurement is the TAIL it clears: 42 readings of
#: pools that genuinely scaled, through the real `ClientPool` at 1 and 4
#: clients, best of two windows, as min / median / max over 14 readings each:
#:
#:     narrow       0.993 / 1.000 / 1.006   7 over 1.0
#:     30% at 20x   0.889 / 1.010 / 1.126   9 over 1.0
#:     10% at 50x   0.707 / 0.940 / 1.155   4 over 1.0
#:
#: plus 14 more at other window lengths whose maximum was 1.327. HALF of all
#: legitimate readings exceed 1.0 and none of the 56 reached 1.5, so a ceiling
#: AT 1.0 would withhold the term on half the points it exists for and restore
#: the false positive on each. 2.0 is 1.5x above the highest legitimate
#: reading measured, and is where a reading starts asserting twice as many
#: queries in service as there were clients to hold them. At 1.5 the 1.327
#: reading has 13% clearance, so the 10%-at-50x regime is refused again on
#: variance; at 1.2 that reading and a measured 1.268 are both withheld; at 10
#: it is not a credibility test at all.
#:
#: **IT CATCHES A GROSSLY CORRUPT BASELINE AND NOTHING SUBTLER** -- a
#: baseline depressed 2x inflates a genuine 0.25 to 0.50, under every ceiling
#: above and over the floor. `baseline_is_credible` below is for that.
CREDIBLE_EFFICIENCY_CEILING = 2.0

#: Largest `p50_one_client / p50_n_client` still consistent with a baseline
#: that was warmed like the rest of the sweep. `baseline_is_credible` below
#: carries the measurement that chose it; the constant is here so that a
#: caller wiring the check up has one name to move.
CREDIBLE_BASELINE_MEDIAN_RATIO = 1.4


def baseline_is_credible(p50_one_client_ms: Optional[float],
                         p50_n_client_ms: Optional[float]
                         ) -> Optional[bool]:
    """Whether the one-client baseline behind a `scaling_efficiency` was warm.

    **CALLED BY `harness/workloads/concurrency.py` SINCE OPS VERSION 9**,
    which holds both medians -- `latency_p50_ms` is recorded at every point.
    Where this returns False the efficiency is neither published nor passed to
    `scaling_status`: the record carries `scaling_efficiency: null` and
    `extra.efficiency_reliability: "baseline_not_credible"`, because a number
    divided by a cold baseline is a number about the baseline. It is here and
    not there because it defends `scaling_status`'s `efficiency` term, the
    argument `SERVED_CONCURRENCY_FLOOR` sits here under.

    WHAT IT DEFENDS AGAINST: `warmup` in that workload is PER CLIENT, so the
    one-client baseline gets the fewest warmed queries of any point, reaches
    cache steady state slowest in wall clock, and is measured FIRST. A
    baseline depressed by k inflates every efficiency by about k, and
    `CREDIBLE_EFFICIENCY_CEILING` cannot see it. This can: a median that got
    FASTER as clients were added is impossible, a queue raising the median and
    a stalled client leaving it alone.

    MEASURED with the four-client window held FIXED -- clients genuinely
    taking turns, retired thread pool -- scaling only the baseline's cost:

        baseline     efficiency   p50_1/p50_n   scaling_status
        honest            0.248         0.996   concurrency_not_achieved
        2.0x cold         0.468         1.889   concurrency_not_achieved
        2.5x cold         0.579         2.338   ok          <- PUBLISHED
        3.0x cold         0.690         2.782   ok

    A serialized pool therefore starts publishing an efficiency at about 2.1x
    under-warmth, and the 2.0x row sits 6% from the floor, which is noise.

    1.4 IS THE MIDPOINT OF A MEASURED GAP, as the other two constants are.
    Legitimate readings ran 0.97-1.02 over every regime and client count
    `tests/test_skewed_latency_is_not_serialization.py` rebuilds -- a scaling
    pool leaves the median alone, a queueing one RAISES it, so the legitimate
    side is bounded above by 1 plus noise -- against a lowest corrupt reading
    of 1.889: a factor of 1.37 from each side. At 1.1 the clearance over
    legitimate noise is 8%, which a one-client median on a busy box crosses on
    its own; at 1.8 the 2.0x row clears by 5%, so anything slightly less cold
    than that already-dangerous row is missed; at 2.0 it is missed outright.

    `None`, never False, when either median is missing or non-positive: that
    is "nobody could look", and False would read as "the baseline was
    corrupt".
    """
    if p50_one_client_ms is None or p50_n_client_ms is None:
        return None
    if p50_one_client_ms <= 0 or p50_n_client_ms <= 0:
        return None
    return (p50_one_client_ms / p50_n_client_ms
            <= CREDIBLE_BASELINE_MEDIAN_RATIO)


#: The pool served about as many queries as it had clients, so a scaling
#: efficiency measured over it describes the engine.
SCALING_OK = "ok"

#: The pool served materially fewer queries at once than it had clients, so
#: whatever `scaling_efficiency` came out of it is not n-client scaling of
#: this engine. Named for the OBSERVATION and not for a cause, deliberately:
#: see `scaling_status` for the two causes a client's own clock cannot
#: separate.
SCALING_CONCURRENCY_NOT_ACHIEVED = "concurrency_not_achieved"

#: Both statuses, for a report or a test that wants to assert it has handled
#: every one. A tuple rather than a set so the order is the order they are
#: documented in.
SCALING_STATUSES = (SCALING_OK, SCALING_CONCURRENCY_NOT_ACHIEVED)

#: A NOTE, never a status and never a refusal: the window's median query sat
#: below half its mean, so the median-based "in service" figure undercounts
#: the queries the pool really had in the engine. Recorded as
#: `extra.scaling_note` beside an `ok` or a refusal, never instead of one.
#:
#: Ops version 9. The incident: `full-20260825` published MariaDB 12.3 at 32
#: clients with p50 34 ms and p99 1,043 ms, so `queries x p50 / window` --
#: the figure the report printed as Little's law -- counted a fraction of the
#: queries that were outstanding, and the same median artefact printed "74-75%
#: of the window inside the call" at ONE client where `qps x mean latency`
#: is 1.00 for all six engines. The tail is a property of the engine worth
#: printing; it is not evidence that the clients failed to keep their queries
#: in flight, which is what a refusal claims.
SCALING_TAIL_DOMINATED = "tail_dominated"

#: Every note, for a report that wants to assert it has rendered each.
SCALING_NOTES = (SCALING_TAIL_DOMINATED,)

#: `served / outstanding` below which a window is `tail_dominated`. Equal to
#: `p50 / mean` identically (both are sums over one kept span set -- see
#: `latency.WindowConcurrency`), so this is "the median is under half the
#: mean". Half for the reason the floor is half: the measured populations sit
#: either side of it -- 0.16-0.19 for the skewed regimes the floor's table
#: rebuilds, 0.95-0.99 for every narrow one.
TAIL_DOMINATED_RATIO = 0.5


def tail_dominated(served: Optional[float],
                   outstanding: Optional[float]) -> Optional[bool]:
    """Whether the median-span estimate sits under half the exact figure.

    `None` when either is missing or `outstanding` is not positive: a window
    that counted nothing has no tail to describe, and False would read as a
    measured narrow distribution.
    """
    if served is None or outstanding is None or outstanding <= 0:
        return None
    return served / outstanding < TAIL_DOMINATED_RATIO


def scaling_note(served: Optional[float],
                 outstanding: Optional[float]) -> Optional[str]:
    """`SCALING_TAIL_DOMINATED` or None -- the value `extra.scaling_note`
    carries. One spelling for the harness and the report."""
    return SCALING_TAIL_DOMINATED if tail_dominated(served, outstanding) \
        else None


def served_fraction(served: Optional[float],
                    clients: int) -> Optional[float]:
    """`served` as a share of the client count.

    The quantity the floor is compared against, as one expression in one
    place, so `scaling_status` and anything printing the evidence cannot
    divide by two different denominators. `None` propagates: a window that
    counted no query has no fraction, and a 0.0 here reads as total
    serialization to every consumer.
    """
    if served is None or clients <= 0:
        return None
    return served / clients


def throughput_fraction(efficiency: Optional[float],
                        clients: int) -> Optional[float]:
    """`scaling_efficiency` read as the share of the clients IN SERVICE, or
    `None` where that reading is not evidence.

    THE SAME QUANTITY `served_fraction` ESTIMATES, which is why one floor
    judges both: at one client a span holds no queue and no lock, so `qps_1`
    is the reciprocal of the per-query CYCLE cost -- service plus the client
    loop's own bookkeeping. Little's law then gives the queries in service at
    n clients as `qps_n x cycle_cost == qps_n / qps_1`, and dividing by n is
    `scaling_efficiency`. One estimator uses the n-client MEDIAN SPAN for that
    cost and the other the ONE-CLIENT MEAN; that is their whole difference.

    CYCLE AND NOT SERVICE, and the gap is measured: `served_concurrency` at
    ONE client is `qps_1 x p50_1`, which would be 1.0 were they the same thing
    and reads 0.965-0.992 here. So this estimator over-reports the in-service
    share by 0.8-3.5% for these sleep-based drivers -- negligible against the
    1.8x margin at the floor, and the SIGN of it is the whole reason the term
    is withheld at n=2 below.

    Their failure modes are largely disjoint, which is the point:
    `served_fraction` is biased down by latency SKEW and this one by a
    SERVER-SIDE QUEUE -- the saturated row's 0.277.

    **SKEW DOES NOT LEAVE THIS ESTIMATOR ALONE EITHER, AND SAYING IT DID WAS
    WRONG.** A count has no quantile, but this is a RATIO of two counts from
    windows of different sample size: `qps_1` estimates `1 / mean` from the
    fewest samples in the sweep and `1 / mean` is convex, so by Jensen the
    baseline is biased up and this fraction down, growing with service-time
    variance. Measured on 10%-at-50x, true efficiency 1.00: 0.707, 0.762 and
    0.881 against a median of 0.940. **The direction is the safe one** -- an
    under-reported efficiency can only leave the old false positive in place,
    never publish a point that should have been refused -- so it bounds the
    repair's reach rather than holing it.

    THREE REASONS TO WITHHOLD, each making this `None` and leaving
    `served_fraction` to decide alone:

    * No efficiency at all -- no one-client baseline to divide by. A 0.0 here
      would read as total serialization.
    * `1 / clients` is not itself below the floor -- `clients <= 2` at a floor
      of 0.5. A COMPLETELY serialized pool delivers one client's throughput,
      so its efficiency IS `1 / clients`: 0.5 at two clients, exactly the
      floor, deciding nothing. The median-span estimator works there because
      its `outstanding / clients` factor is always under 1.0 and that bias
      carries it below the floor -- measured on the retired thread pool at two
      clients, `served / clients` 0.4873/0.4919/0.4945 against efficiencies
      0.4987/0.4913/0.4992: three refusals out of three by margins of 0.0055
      to 0.0127, where a `max` takes the larger and sits on the floor. A
      reviewer measured 0.5008 there and published a pool whose clients were
      taking turns. Derived from the floor, not spelled `clients > 2`.
    * The reading is above `CREDIBLE_EFFICIENCY_CEILING`, far enough over 1 to
      be a corrupt baseline rather than a measurement.
    """
    if efficiency is None or clients <= 0:
        return None
    if 1.0 / clients >= SERVED_CONCURRENCY_FLOOR:
        return None
    if efficiency > CREDIBLE_EFFICIENCY_CEILING:
        return None
    return efficiency


def achieved_fraction(served: Optional[float], clients: int,
                      efficiency: Optional[float] = None) -> Optional[float]:
    """The best estimate of the share of `clients` the pool had in service.

    THE LARGER of the two estimators above, and that choice is the whole
    content of this function. Each is biased in the regime the other is exact
    in -- see `throughput_fraction` -- and both are biased DOWNWARDS only:
    skew cannot inflate a median above a mean, and a server-side queue cannot
    make a pool deliver throughput it did not deliver. Two lower bounds on one
    quantity, so the larger is the better bound and the smaller carries no
    information the larger lacks.

    `None` when `served_fraction` is None, and NOT the efficiency on its own:
    a window that counted no query has nothing in it to have been in service,
    and an efficiency over it is a ratio of two throughputs one of which was
    never measured. See `scaling_status` for why that is not `ok`.
    """
    fraction = served_fraction(served, clients)
    if fraction is None:
        return None
    throughput = throughput_fraction(efficiency, clients)
    if throughput is None:
        return fraction
    # THE ARGUMENT ORDER IS LOAD-BEARING, NOT STYLE. `max` returns its FIRST
    # argument unless a later one compares greater, and every comparison
    # against a NaN is False -- so a NaN efficiency, which `qps_n / (clients *
    # qps_1)` gives from a zero baseline, degrades to the median-span rule
    # here. Reversed it returns the NaN, `NaN < floor` is False too, and every
    # point in the run publishes. The ceiling does not catch NaN either
    # (`NaN > ceiling` is False); `float("inf")` it does.
    return max(fraction, throughput)


def scaling_status(served: Optional[float],
                   clients: int,
                   efficiency: Optional[float] = None,
                   *, outstanding: Optional[float] = None) -> Optional[str]:
    """Whether a scaling efficiency over this window is reportable.

    **SINCE OPS VERSION 9 THE PRIMARY FIGURE IS `outstanding`, AND EVERY
    LIVE CALLER PASSES IT.** The rule below this paragraph was built when the
    clients were threads, and its reason for rejecting the exact mean-based
    figure was the interpreter lock: a thread waiting for the lock is still
    inside its span, so `outstanding` read the client count however little was
    served (`latency.WindowConcurrency` has the five-driver table). The clients
    have been separate OS processes since version 6, so that blindness has no
    cause left in the measurement path, and keying the guard to the MEDIAN
    kept only its cost: on `full-20260825` MariaDB 12.3 at 32 clients had p50
    34 ms against p99 1,043 ms, and `queries x p50` counted a fraction of the
    queries the engine actually held. Given `outstanding`:

    * `outstanding / clients < SERVED_CONCURRENCY_FLOOR` refuses -- the
      clients did not keep their queries in flight, which is the claim the
      efficiency's denominator rests on and the one case a client clock
      measures exactly.
    * The residue still refuses: BOTH in-service estimators below the floor --
      `served / clients` and a credible `throughput_fraction` -- is the
      quadrant `TestTheResidueHoldsBothCausesAndCannotSeparateThem` measures,
      where a pool neither scaled nor kept its median at its client count and
      the cause cannot be attributed. Without an efficiency term (two clients,
      no baseline, an incredible baseline, a NaN) a low `served` alone no
      longer refuses: it is `tail_dominated` (`scaling_note`), a note.
    * Otherwise `ok`, including a saturated server -- outstanding at the
      client count, efficiency low, which is the engine ceiling this guard has
      always had to publish.

    **WHAT THIS GUARD DOES NOT DETECT, said plainly because the rule's name
    invites the opposite reading.** A server that genuinely answers ONE query
    at a time is, from the clients' clocks, indistinguishable from a
    saturated one: every client waits inside its call, so `outstanding` is
    about N; the queue raises the whole latency distribution, so `served` is
    about N too; and the throughput is flat, so the efficiency is about 1/N.
    It is published `ok`, with that low efficiency. What the guard detects is
    CLIENT-SIDE starvation (`outstanding / N < 0.5`) and the both-low residue.
    Server-side serialization versus saturation is not separated by any
    figure computed here; the server's own CPU and pressure over the window
    (`report/server_windows.py`, the report's Server CPU columns) are the
    evidence for that, and they are beside the figure rather than in it.

    `outstanding=None` is the pre-version-9 rule, byte for byte: kept so the
    mutation probes and the archived-record recomputations that call it
    without the keyword keep testing the arithmetic they were written about.

    Here rather than in `harness/workloads/clients.py`, where it was written
    and where `PoolWindow.scaling_status` still exposes it: this module owns
    the quantity, the floor belongs beside the quantity it judges, and
    `clients.py` was 846 lines against this repository's 800-line ceiling
    with the policy inlined. `churn_readers.py` exists for the same reason.

    `efficiency` is OPTIONAL AND THE DEFAULT IS THE OLD RULE, which is a
    migration state and not a design: the two callers that have an efficiency
    to pass are in `harness/workloads/`, and a caller that does not pass one
    gets a status computed from the median span alone and therefore gets the
    false positive the floor's docstring measures. `PoolWindow.scaling_status`
    is one such caller -- the window does not know the one-client baseline
    that `concurrency.run` divides by -- and the churn readers are the other,
    and they have no one-client baseline at all. Until those are routed this
    argument changes nothing about any published figure.

    `None` -- not `SCALING_OK` -- when there is nothing to judge: a window
    that counted no query has no spans to sum, and "the pool achieved its
    concurrency" and "nobody could look" are different statements, exactly as
    `PoolWindow.client_bound` distinguishes them for an unreadable CPU
    counter. The report must render the two differently or the absence reads
    as a pass.

    `SCALING_CONCURRENCY_NOT_ACHIEVED` NAMES THE OBSERVATION, NOT A CAUSE,
    deliberately. Two things produce it and a client's own clock cannot tell
    them apart: the CLIENTS serialized on Python's interpreter lock, the case
    the live smoke run exposed, which makes the efficiency a property of the
    harness; or the SERVER serialized, in which case there is a real finding
    -- but it is "this engine serializes queries", not "this engine scales at
    0.25", which is a different claim needing a different presentation. Both
    make the published number wrong, so both are refused and the report
    states both readings. `client_cpu_seconds` separates them in practice,
    roughly one core against roughly none, and travels beside the status in
    the record. Putting the CPU figure INSIDE the condition was rejected for
    the reason `clients.CLIENT_BOUND_UTIL` is a cautionary tale: a CPU term
    reintroduces the blindness this quantity was added to fix.

    **WHICH IS WHY NO THIRD STATUS WAS ADDED, having been considered.** The
    residue -- a pool that neither scaled nor kept its median at its client
    count -- holds BOTH causes, indistinguishable from THIS WINDOW, which is
    the whole weight the argument carries. Measured on this tree, clients
    taking turns on the interpreter lock, a saturated server whose service
    time is also two-moded, and a server whose cache misses queue on one
    shared resource all deliver a throughput that does NOT rise with the
    client count, at an efficiency of 0.23-0.30 and a `served / clients` of
    0.05-0.25: the two quantities this status reads, and they do not sort the
    three.

    **ONE QUANTITY OUTSIDE THE STATUS SORTS TWO OF THE THREE, and saying
    otherwise was wrong.** `p50@n / p50@1` measured x3.66-x3.73 for the
    saturated server and x3.62-x3.66 for the saturated-and-skewed one over
    five builds, against x0.99-x1.01 for clients taking turns -- a queue moves
    the median and a stalled client does not -- so a third term on it would
    separate THOSE two. It is not promoted because of the third regime, where
    cache misses queue on one shared resource and the growth measured x1.00:
    its median query is a HIT that never reaches the bottleneck. That is the
    realistic 1M-row shape and the case such a term would get wrong, so the
    growth is reported beside the status rather than inside it. A value
    meaning "withheld, cannot attribute" would still be a synonym for this
    one.

    **`report/` IS NOT LEFT UNTOUCHED BY THIS, WHICH AN EARLIER VERSION OF
    THIS DOCSTRING CLAIMED.** Keeping one status string means no new value has
    to be rendered, and no more. `report/render.py`'s concurrency footnote is
    FIXED TEXT asserting that where the queries-in-service column says the
    pool lacked its clients' worth in flight, the efficiency cell reads "not
    measurable" and no number is published. Once a caller passes an
    `efficiency`, a skewed-but-scaling point renders a served column well
    under half its client count with an unmarked efficiency beside it, and
    that footnote then says the opposite of what the table shows. Its "a real
    queue raises the whole latency distribution, median included" is the claim
    `SERVED_CONCURRENCY_FLOOR` now records as true only of a NARROW service
    time. Both are the follow-up's, in a file this does not touch.

    **What the `efficiency` term is for, then, is the case where NEITHER
    reading is true**: a pool that scaled linearly and whose latency is merely
    skewed. `SERVED_CONCURRENCY_FLOOR`'s docstring has that measurement, three
    regimes of it. A pool that delivered n times a one-client throughput
    cannot have been starved, whatever shape its latency has.

    **ONE CLIENT IS `SCALING_OK` BY DEFINITION, AND THAT IS A SEPARATE
    STATEMENT FROM EVERYTHING ABOVE.** A one-client pool has NO CONCURRENCY TO
    FAIL TO ACHIEVE. There is no true positive at one client: one client
    cannot take turns with itself, so "the clients serialized on the
    interpreter lock" is impossible, and "the engine answers one query at a
    time" is not a defect at n=1 -- it is the definition of n=1. Whatever
    `served / clients` comes out as, it is a statement about the latency
    distribution and about nothing else.

    It is said as that statement and NOT reached through the efficiency, which
    is 1.0 by construction at one client and would publish the same points for
    no reason at all: a right answer from a constructed value is not evidence,
    and `throughput_fraction` withholds exactly that tautology.

    Measured, because it was a live defect: the 30%-at-20x regime at a SINGLE
    client read `served / clients` 0.1428, 0.1716, 0.1764, 0.1850 and 0.2050
    over five builds, and the pre-change rule refused all five. Every run of a
    skewed workload printed "the clients serialized" about one client while
    withholding an efficiency that is 1.0 by definition, and `clients == 1` is
    in every profile. `None` still wins over this branch: a one-client window
    that counted no query is "nobody could look", not "the pool was fine".
    """
    if outstanding is not None:
        return _outstanding_rule(served, clients, efficiency, outstanding)
    fraction = achieved_fraction(served, clients, efficiency)
    if fraction is None:
        return None
    if clients == 1:
        # Not a short cut past the comparison below: there is nothing for a
        # single client to have failed to achieve. Below the None check, so a
        # one-client window that counted no query is still "nobody looked".
        return SCALING_OK
    if fraction < SERVED_CONCURRENCY_FLOOR:
        return SCALING_CONCURRENCY_NOT_ACHIEVED
    return SCALING_OK


def _outstanding_rule(served: Optional[float], clients: int,
                      efficiency: Optional[float],
                      outstanding: float) -> Optional[str]:
    """The version-9 rule `scaling_status` documents, given `outstanding`."""
    occupied = served_fraction(outstanding, clients)
    if occupied is None:
        return None
    if clients == 1:
        return SCALING_OK
    if occupied < SERVED_CONCURRENCY_FLOOR:
        return SCALING_CONCURRENCY_NOT_ACHIEVED
    in_service = served_fraction(served, clients)
    throughput = throughput_fraction(efficiency, clients)
    # `throughput == throughput` drops a NaN efficiency (a zero baseline) to
    # "no efficiency term", which is the safe direction: without the term a
    # low median is a note and not a refusal.
    if (in_service is not None and throughput is not None
            and throughput == throughput
            and in_service < SERVED_CONCURRENCY_FLOOR
            and throughput < SERVED_CONCURRENCY_FLOOR):
        return SCALING_CONCURRENCY_NOT_ACHIEVED
    return SCALING_OK


__all__ = [
    "SERVED_CONCURRENCY_FLOOR",
    "CREDIBLE_EFFICIENCY_CEILING",
    "CREDIBLE_BASELINE_MEDIAN_RATIO",
    "SCALING_OK",
    "SCALING_CONCURRENCY_NOT_ACHIEVED",
    "SCALING_STATUSES",
    "SCALING_TAIL_DOMINATED",
    "SCALING_NOTES",
    "TAIL_DOMINATED_RATIO",
    "tail_dominated",
    "scaling_note",
    "baseline_is_credible",
    "served_fraction",
    "throughput_fraction",
    "achieved_fraction",
    "scaling_status",
]
