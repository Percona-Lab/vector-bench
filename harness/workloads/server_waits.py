"""Capture the server's wait profile around one measured window.

`WaitCapture.begin()` runs immediately BEFORE a window and `finish()`
immediately AFTER it; neither runs inside the span a QPS is divided by. That is
the property the whole feature is worth having for: a performance_schema scan
or a serverStatus is not free, and taken inside the window it would charge the
instrument to the engine being measured. The deltas therefore also cover the
pool's warm-up (a few dozen queries per client) and its teardown, which is
said on every record by `server_waits.covers`.

**A separate connection, always.** The capture asks the driver factory for a
driver of its own. Reusing the main harness driver would be harmless for the
counter engines and wrong for pgvector, whose sampler would then be reading
`pg_stat_activity` on a connection a workload also uses; and a client
connection is never touched, because the clients are forked processes that own
theirs.

**pgvector's sampler is OPT-IN, and the one thing here that runs DURING a
window.** Postgres keeps no cumulative wait-event counters, so the only way to
see what its backends wait on is to sample `pg_stat_activity` while they run --
an extra backend and a query whose cost grows with the client count, inside
the span the QPS divides by. Every other engine only snapshots at the edges,
so sampling by default would measure pgvector under load the others are not
under. It therefore runs only with `ops.server_wait_sampler: true`, which
every record then states (`extra.server_wait_sampler`); off, pgvector records
`server_waits: None` with `SAMPLER_OFF_REASON`, and edge-only samples are NOT
substituted -- a point-in-time view of idle backends before and after a window
is not a delta of anything.

**The sampler is a forked PROCESS, not a thread.** The window's clients are
forked after the sampler starts. A thread alive in the parent at that fork is exactly what
`tests/test_g4_pool_teardown.py` exists to forbid (CPython's multi-threaded
fork `DeprecationWarning`, and a lock held by the thread is held forever in
every child), and `ClientPool` starts its own attendant threads only after
its forks for that reason. A process forked before the pool leaves the parent
single-threaded; it opens its own connection, samples, and sends one result
back over a pipe when told to stop.

**It never raises and never zeroes.** Every failure -- the side connection
refused, performance_schema off, a counter reset -- ends as
`server_waits: None` with `server_waits_reason` saying which. A measured
window with no waits is `{"events": []}`, a different fact.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..drivers.server_waits import NOT_IMPLEMENTED
from .client_process import _MP

#: What a record says the deltas span. One phrase, so a reader of the JSONL
#: does not take the wait seconds as the measured window's alone.
COVERS = "the window plus its pool warm-up and teardown"

#: Why a sampling engine has no wait profile by default.
SAMPLER_OFF_REASON = (
    "postgres has no cumulative wait-event counters; sampling is opt-in "
    "(ops.server_wait_sampler) because it runs during the window")

#: The record keys. A prefix is prepended for the churn reader windows.
WAITS_KEY = "server_waits"
REASON_KEY = "server_waits_reason"


class WaitCapture:
    """One window's before/after snapshots (or samples), on a side connection.
    """

    def __init__(self, driver_factory: Optional[Callable[[], Any]],
                 like: Any = None, sampler: bool = False):
        """`like` is a driver the workload already holds, of the class the
        factory makes. When it overrides nothing, the factory is never
        called: a driver costs a connection, and on the test fake it costs a
        clone id, which reseeds every client after it -- the concurrency
        suite's recall figures moved by 0.01 when this was not checked."""
        self._factory = driver_factory
        self._like = like
        #: Whether a sampling engine may sample during the window
        #: (`ops.server_wait_sampler`). Off: no process is started.
        self._sampler_allowed = bool(sampler)
        self._driver: Any = None
        self._before: Optional[Dict[str, Any]] = None
        self._reason: Optional[str] = None
        self._interval: Optional[float] = None
        self._sampler: Any = None
        self._stop: Any = None
        self._results: Any = None

    # -- lifecycle ----------------------------------------------------------

    def begin(self) -> "WaitCapture":
        if self._factory is None:
            self._reason = "no driver factory to open a side connection with"
            return self
        if self._like is not None and not _overrides(self._like,
                                                     "wait_snapshot"):
            self._reason = NOT_IMPLEMENTED
            return self
        if (self._like is not None and not self._sampler_allowed
                and getattr(self._like, "wait_sample_interval_s", None)):
            self._reason = SAMPLER_OFF_REASON
            return self
        try:
            driver = self._factory()
            self._driver = driver
            if not _overrides(driver, "wait_snapshot"):
                self._reason = NOT_IMPLEMENTED
                return self
            self._interval = getattr(driver, "wait_sample_interval_s", None)
            if self._interval and not self._sampler_allowed:
                self._reason = SAMPLER_OFF_REASON
                self._interval = None
                return self
            if self._interval:
                # The sampler opens its own connection in the child; this
                # parent-side instance is only used for `wait_delta`.
                self._start_sampler()
            else:
                driver.connect()
                self._before, self._reason = driver.wait_snapshot()
        except Exception as exc:
            self._reason = f"side connection for the wait snapshot failed: {exc}"
        return self

    def finish(self, prefix: str = "") -> Dict[str, Any]:
        """The record keys for this window. Closes the side connection."""
        waits: Optional[Dict[str, Any]] = None
        reason = self._reason
        try:
            if reason is None:
                waits, reason = self._result()
        except Exception as exc:
            waits, reason = None, f"wait delta failed: {exc}"
        finally:
            self._close()
        if waits is not None:
            waits = {**waits, "covers": COVERS}
            reason = None
        elif reason is None:
            reason = "no wait data was returned"
        return {prefix + WAITS_KEY: waits, prefix + REASON_KEY: reason}

    # -- internals ----------------------------------------------------------

    def _start_sampler(self) -> None:
        self._stop = _MP.Event()
        receive, send = _MP.Pipe(duplex=False)
        self._results = receive
        self._sampler = _MP.Process(
            target=_sample_loop,
            args=(self._factory, float(self._interval), self._stop, send),
            name="vb-wait-sampler", daemon=True)
        self._sampler.start()
        send.close()

    def _result(self):
        driver = self._driver
        if self._interval:
            self._stop.set()
            timeout = max(10.0, 3 * self._interval)
            if not self._results.poll(timeout):
                return None, (f"the wait sampler sent nothing within "
                              f"{timeout:.0f}s of being told to stop")
            accumulated, samples, error = self._results.recv()
            if not samples:
                return None, error or "the wait sampler took no sample"
            return driver.wait_delta({"events": {}, "samples": 0},
                                     {"events": accumulated,
                                      "samples": samples})
        after, reason = driver.wait_snapshot()
        if after is None:
            return None, reason
        return driver.wait_delta(self._before, after)

    def _close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=10.0)
            if self._sampler.is_alive():
                self._sampler.terminate()
                self._sampler.join(timeout=5.0)
        if self._results is not None:
            try:
                self._results.close()
            except Exception:
                pass
            self._results = None
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    @property
    def sampling(self) -> bool:
        """Whether a sampler process is (still) running -- for the tests."""
        return self._sampler is not None and self._sampler.is_alive()

    @property
    def side_driver(self) -> Any:
        return self._driver


def _sample_loop(factory: Callable[[], Any], interval: float, stop: Any,
                 send: Any) -> None:
    """The sampler process: own connection, one sample per interval.

    The first sample is taken at once, so a window shorter than the interval
    still gets one. Sends `(events, samples, last_error)` exactly once.
    """
    accumulated: Dict[str, Dict[str, Any]] = {}
    samples, error = 0, None
    driver = None
    try:
        driver = factory()
        driver.connect()
        while True:
            try:
                sample, reason = driver.wait_snapshot()
            except Exception as exc:
                sample, reason = None, f"wait sample failed: {exc}"
            if sample is None:
                error = reason
            else:
                samples += int(sample.get("samples") or 1)
                for name, value in (sample.get("events") or {}).items():
                    slot = accumulated.setdefault(
                        name, {"count": 0, "wait_s": None})
                    slot["count"] += int(value.get("count") or 0)
            if stop.wait(interval):
                break
    except Exception as exc:
        error = f"wait sampler connection failed: {exc}"
    finally:
        try:
            send.send((accumulated, samples, error))
        except Exception:
            pass
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


def _overrides(driver: Any, name: str) -> bool:
    """Whether the driver's class replaces the base no-op.

    Checked before connecting, so an engine with nothing to say costs no
    connection per window. Duck-typed drivers without the base class count as
    overriding when they have the method at all.
    """
    from ..drivers.base import EngineDriver
    method = getattr(type(driver), name, None)
    if method is None:
        return False
    return method is not getattr(EngineDriver, name, None)


def capture_window(driver_factory: Optional[Callable[[], Any]],
                   run: Callable[[], Any], prefix: str = "",
                   like: Any = None, sampler: bool = False):
    """`(run(), wait keys)` with the capture wrapped around the call."""
    capture = WaitCapture(driver_factory, like, sampler).begin()
    try:
        result = run()
    except BaseException:
        capture.finish(prefix)
        raise
    return result, capture.finish(prefix)


def sampler_disclosure(driver: Any, enabled: bool) -> Optional[Dict[str, Any]]:
    """`extra.server_wait_sampler` for a run, or None when nothing sampled.

    Only an engine that samples gets the key, and only when the profile
    turned sampling on: that is the one case in which the measured windows
    carried load the other engines' did not.
    """
    interval = getattr(driver, "wait_sample_interval_s", None)
    if not (enabled and interval):
        return None
    return {"interval_s": float(interval), "concurrent": True}


__all__ = ["COVERS", "REASON_KEY", "SAMPLER_OFF_REASON", "WAITS_KEY",
           "WaitCapture", "capture_window", "sampler_disclosure"]
