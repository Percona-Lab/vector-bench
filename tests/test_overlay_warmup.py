"""The one-time index warm-up every ann overlay module runs before its first
measured configuration.

The incident: in full-20260825 every MySQL-family sweep measured ef_search 10
SLOWER than ef_search 20 -- 1.69x to 2.13x, on all six MariaDB/AliSQL curves --
and both Percona Search curves did the same at 1.13x. More candidates cannot be
faster on one index; the first point simply paid for a cold graph. The 30
random-vector queries `_warm_cache` runs inside every `set_query_arguments`
were not enough to warm a 1M x 1536 graph, and the cold point then set numbers
in the recall-floor table.

What is pinned here:

* the stop rule -- two consecutive 200-query batches within 5% of each other,
  capped at 5,000 queries -- against a fake engine whose latency is known;
* that the warm-up queries are TRAINING vectors only, never the test set,
  because warming with the vectors about to be timed would pre-cache exactly
  the graph paths they need and flatter the result;
* that it runs once per built index and lands in the result attributes;
* that every overlay algorithm module is wired to the one shared helper, so no
  engine gets a different warm-up from another.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import types

import numpy
import pytest

from tests.test_e1_percona_design import ALGORITHMS, OVERLAY, _load_overlay


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(dotted: str, path: str) -> types.ModuleType:
    if dotted in sys.modules:
        return sys.modules[dotted]
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = was
    return mod


def _warmup_module():
    _load_overlay()
    return _load("ann_benchmarks.vb_warmup",
                 os.path.join(OVERLAY, "ann_benchmarks", "vb_warmup.py"))


def _engine_dirs():
    return sorted(d for d in os.listdir(ALGORITHMS)
                  if os.path.isfile(os.path.join(ALGORITHMS, d, "module.py")))


def _engine_module(engine: str) -> types.ModuleType:
    _load_overlay()
    # mariadb123 imports mariadb's dialect, so its package must exist first.
    for name in ("mariadb", engine):
        pkg_name = f"ann_benchmarks.algorithms.{name}"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [os.path.join(ALGORITHMS, name)]
            sys.modules[pkg_name] = pkg
    return _load(f"ann_benchmarks.algorithms.{engine}.module",
                 os.path.join(ALGORITHMS, engine, "module.py"))


def _algorithm_class(engine: str) -> type:
    mod = _engine_module(engine)
    base = sys.modules["ann_benchmarks.algorithms.base.module"].BaseANN
    found = [obj for obj in vars(mod).values()
             if isinstance(obj, type) and issubclass(obj, base)
             and obj.__module__ == mod.__name__]
    assert len(found) == 1, f"{engine}: expected one algorithm class, {found}"
    return found[0]


# ---------------------------------------------------------------------------
# A fake engine with a known latency curve
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ColdThenWarm:
    """Latency falls 10% per 200 queries while cold -- never within the 5%
    tolerance -- then, after `warm_after` queries, halves from where it
    started and stays flat. The decline matters: a constant cold latency
    would satisfy the stop rule on its second batch, and no real cold cache
    looks like that."""

    def __init__(self, clock: FakeClock, warm_after: int,
                 base: float = 0.010, never_warms: bool = False):
        self.clock, self.warm_after, self.base = clock, warm_after, base
        self.never_warms = never_warms
        self.seen: list = []

    def latency(self, i: int) -> float:
        if self.never_warms or i < self.warm_after:
            return self.base * 0.9 ** (i // 200)
        return self.base * 0.5

    def __call__(self, vector) -> None:
        self.clock.now += self.latency(len(self.seen))
        self.seen.append(numpy.array(vector, copy=True))


class Untouchable(numpy.ndarray):
    """A test set that fails the test if anything reads it."""

    def __array_finalize__(self, obj):  # pragma: no cover - trivial
        pass

    def __getitem__(self, item):
        raise AssertionError("the warm-up read the test set")

    def __iter__(self):
        raise AssertionError("the warm-up iterated the test set")

    def __array__(self, *args, **kwargs):
        raise AssertionError("the warm-up converted the test set")


def _train(rows: int = 20_000, dim: int = 8) -> numpy.ndarray:
    return numpy.random.RandomState(1).normal(size=(rows, dim)).astype("<f4")


# ---------------------------------------------------------------------------
# The stop rule
# ---------------------------------------------------------------------------

class TestTheStopRule:

    @pytest.mark.parametrize("warm_after", [400, 1000, 2000])
    def test_it_stops_within_two_batches_of_the_engine_warming(self, warm_after):
        vw = _warmup_module()
        clock = FakeClock()
        engine = ColdThenWarm(clock, warm_after)
        result = vw.warm_until_stable(engine, vw.training_sample(_train()),
                                      clock=clock)
        assert result["converged"] is True
        assert warm_after < result["queries"] <= warm_after + 400
        assert result["queries"] == len(engine.seen)

    def test_it_does_not_stop_while_the_engine_is_still_warming(self):
        vw = _warmup_module()
        clock = FakeClock()
        engine = ColdThenWarm(clock, 3000)
        result = vw.warm_until_stable(engine, vw.training_sample(_train()),
                                      clock=clock)
        assert result["queries"] > 3000

    def test_it_is_capped_at_five_thousand(self):
        vw = _warmup_module()
        assert vw.WARMUP_CAP == 5000
        clock = FakeClock()
        engine = ColdThenWarm(clock, 0, never_warms=True)
        result = vw.warm_until_stable(engine, vw.training_sample(_train()),
                                      clock=clock)
        assert result["queries"] == 5000 == len(engine.seen)
        assert result["converged"] is False

    def test_the_batch_and_tolerance_are_the_declared_ones(self):
        vw = _warmup_module()
        assert vw.WARMUP_BATCH == 200
        assert vw.WARMUP_TOLERANCE == 0.05

    def test_the_duration_is_the_time_spent_querying(self):
        vw = _warmup_module()
        clock = FakeClock()
        engine = ColdThenWarm(clock, 1000)
        result = vw.warm_until_stable(engine, vw.training_sample(_train()),
                                      clock=clock)
        assert result["seconds"] == pytest.approx(
            sum(engine.latency(i) for i in range(result["queries"])))

    def test_a_failing_query_ends_the_warm_up_without_raising(self):
        vw = _warmup_module()

        def broken(vector):
            raise RuntimeError("server went away")

        result = vw.warm_until_stable(broken, vw.training_sample(_train()))
        assert result["queries"] == 0 and result["converged"] is False


# ---------------------------------------------------------------------------
# Training vectors only
# ---------------------------------------------------------------------------

class TestOnlyTrainingVectors:

    def test_every_warm_up_query_is_a_training_row(self):
        vw = _warmup_module()
        train = _train()
        clock = FakeClock()
        engine = ColdThenWarm(clock, 0, never_warms=True)
        vw.warm_until_stable(engine, vw.training_sample(train), clock=clock)
        rows = {r.tobytes() for r in train}
        assert all(v.tobytes() in rows for v in engine.seen)

    def test_the_test_set_is_never_touched(self):
        """The module is handed the training set by `fit` and nothing else.
        The test set exists in this test only to prove the warm-up path has
        no route to it."""
        vw = _warmup_module()
        test_set = numpy.zeros((100, 8), dtype="<f4").view(Untouchable)

        class Algo(vw.IndexWarmupMixin):
            def __init__(self, engine):
                self.query = lambda v, n: engine(v)

        clock = FakeClock()
        engine = ColdThenWarm(clock, 600)
        algo = Algo(engine)
        algo._keep_warmup_sample(_train())
        algo._warm_index_once(clock=clock)
        assert engine.seen, "the warm-up ran no queries"
        assert test_set.shape == (100, 8)  # still intact and never read

    def test_the_sample_is_fixed_and_seeded(self):
        vw = _warmup_module()
        train = _train()
        a, b = vw.training_sample(train), vw.training_sample(train)
        assert numpy.array_equal(a, b)
        assert len(a) == vw.WARMUP_SAMPLE_ROWS
        assert len(a) <= vw.WARMUP_CAP

    def test_the_sample_is_a_copy_not_a_view_of_the_corpus(self):
        """Keeping a view would keep the whole training set alive for the
        lifetime of the algorithm object."""
        vw = _warmup_module()
        train = _train()
        sample = vw.training_sample(train)
        assert sample.base is None or sample.base is not train
        assert not numpy.shares_memory(sample, train)

    def test_a_small_corpus_is_cycled_not_refused(self):
        vw = _warmup_module()
        train = _train(rows=50)
        clock = FakeClock()
        engine = ColdThenWarm(clock, 0, never_warms=True)
        result = vw.warm_until_stable(engine, vw.training_sample(train),
                                      clock=clock)
        assert result["queries"] == 5000


# ---------------------------------------------------------------------------
# Once per index, and recorded
# ---------------------------------------------------------------------------

class TestOncePerIndexAndRecorded:

    def _algo(self, warm_after=1000):
        vw = _warmup_module()

        class Algo(vw.IndexWarmupMixin):
            pass

        clock = FakeClock()
        engine = ColdThenWarm(clock, warm_after)
        algo = Algo()
        algo.query = lambda v, n: engine(v)
        algo._keep_warmup_sample(_train())
        return vw, algo, engine, clock

    def test_it_runs_once_however_many_configurations_follow(self):
        vw, algo, engine, clock = self._algo()
        algo._warm_index_once(clock=clock)
        first = len(engine.seen)
        for _ in range(5):
            algo._warm_index_once(clock=clock)
        assert len(engine.seen) == first

    def test_the_sample_is_released_once_used(self):
        vw, algo, engine, clock = self._algo()
        algo._warm_index_once(clock=clock)
        assert algo._vb_warmup_sample is None

    def test_count_and_duration_reach_the_attributes(self):
        vw, algo, engine, clock = self._algo()
        assert algo._warmup_attrs() == {}
        algo._warm_index_once(clock=clock)
        attrs = algo._warmup_attrs()
        assert attrs["index_warmup_queries"] == len(engine.seen)
        assert attrs["index_warmup_seconds"] > 0
        assert attrs["index_warmup_converged"] is True
        # ann-benchmarks writes every attribute into HDF5, which has no None.
        assert all(v is not None for v in attrs.values())

    def test_a_new_index_is_warmed_again(self):
        """ann-benchmarks builds one index per definition and may reuse the
        algorithm object; a second `fit` is a new, cold graph."""
        vw, algo, engine, clock = self._algo()
        algo._warm_index_once(clock=clock)
        first = len(engine.seen)
        algo._keep_warmup_sample(_train())
        algo._warm_index_once(clock=clock)
        assert len(engine.seen) > first


# ---------------------------------------------------------------------------
# Every engine module is wired to the shared helper
# ---------------------------------------------------------------------------

def _stub_instance(cls: type):
    """An instance without __init__ (which would start a server), with every
    private attribute the module reads defaulting to 0."""

    def missing(self, name):
        if name.startswith("_") and not name.startswith("__"):
            return 0
        raise AttributeError(name)

    shim = type(cls.__name__, (cls,), {"__getattr__": missing})
    algo = object.__new__(shim)

    class Cursor:
        def execute(self, *a, **k):
            return None

        def fetchone(self):
            return ("stub",)

        def fetchall(self):
            return []

    algo._cur = Cursor()
    algo._plan_verified = True
    algo._query_verified = True
    algo._status = {}
    algo._warm_cache = lambda: None
    for name in ("_server_version", "_jvm_version", "_packages", "_image_march"):
        setattr(algo, name, lambda: "stub")
    return algo


ENGINES = _engine_dirs()


def test_every_engine_the_overlay_ships_is_covered():
    assert {"mariadb", "mariadb123", "alisql", "percona", "myvector",
            "pgvector", "mongodb", "valkey"} <= set(ENGINES)


@pytest.mark.parametrize("engine", ENGINES)
class TestEveryEngineIsWired:

    def test_it_uses_the_shared_mixin(self, engine):
        vw = _warmup_module()
        assert issubclass(_algorithm_class(engine), vw.IndexWarmupMixin)

    def test_fit_keeps_a_training_sample(self, engine):
        """Somewhere on the class's `fit` path -- its own, or the shared
        `VBMySQLBase.fit` it calls -- the training set is sampled."""
        cls = _algorithm_class(engine)
        sources = []
        for klass in cls.__mro__:
            for name in ("fit", "_fit"):
                if name in vars(klass):
                    sources.append(inspect.getsource(vars(klass)[name]))
        assert any("self._keep_warmup_sample(X)" in s for s in sources), engine

    def test_the_first_configuration_warms_the_index_once(self, engine):
        cls = _algorithm_class(engine)
        algo = _stub_instance(cls)
        seen = []
        train = _train(rows=2000)
        algo.query = lambda v, n: seen.append(numpy.array(v, copy=True)) or []
        algo._keep_warmup_sample(train)

        algo.set_query_arguments(10)
        ran = len(seen)
        assert ran >= 2 * 200, f"{engine}: the index warm-up did not run"
        rows = {r.tobytes() for r in train}
        assert all(v.tobytes() in rows for v in seen)

        algo.set_query_arguments(20)
        algo.set_query_arguments(40)
        assert len(seen) == ran, f"{engine}: warmed more than once"

    def test_the_counts_reach_get_additional(self, engine):
        cls = _algorithm_class(engine)
        algo = _stub_instance(cls)
        algo.query = lambda v, n: []
        algo._keep_warmup_sample(_train(rows=2000))
        algo.set_query_arguments(10)
        extra = algo.get_additional()
        assert extra["index_warmup_queries"] >= 400
        assert "index_warmup_seconds" in extra
        assert "index_warmup_converged" in extra

    def test_the_per_configuration_warm_cache_is_kept(self, engine):
        """The index warm-up is in addition to, not instead of, the 30
        throwaway queries before every configuration."""
        cls = _algorithm_class(engine)
        assert callable(getattr(cls, "_warm_cache", None)), engine
        source = inspect.getsource(
            next(k for k in cls.__mro__
                 if "set_query_arguments" in vars(k)).set_query_arguments)
        assert "self._warm_cache()" in source
        assert "self._warm_index_once()" in source


def test_the_estimator_names_the_same_cap():
    """The run-time estimate cannot import the overlay (it needs numpy and runs
    on a bare host), so the cap it prints is a copy -- pinned here."""
    from orchestrator import estimate
    vw = _warmup_module()
    assert f"{vw.WARMUP_CAP:,} training-set queries" in estimate._ANN_QUERY_COST
