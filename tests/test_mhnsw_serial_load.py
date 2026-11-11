"""MariaDB's ann load uses one writer, because MHNSW deadlocks under more.

Concurrent INSERTs into an MHNSW index X-lock neighbour rows of the graph
table (`t1#i#01`) in graph order, so two writers take them in opposite orders
and InnoDB kills one (errno 1213). Run percona-mariadb-20261110 lost both
MariaDB ann units to it at 8 threads, and the threads bought nothing --
measured 2026-11-11 on MariaDB 11.8.9 (16,000 x 784-d rows, 500-row INSERTs):

    threads      1      2      4      8
    rows/s     1141   1214   1196   1217
    deadlocks     0      0      6     46
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

VB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VB_ROOT)

from tests.test_e1_percona_design import ALGORITHMS, _load_overlay  # noqa: E402


def _mariadb_module(name="mariadb"):
    vb, _ = _load_overlay()
    dotted = f"ann_benchmarks.algorithms.{name}"
    if f"{dotted}.module" in sys.modules:
        return vb, sys.modules[f"{dotted}.module"]
    pkg = types.ModuleType(dotted)
    pkg.__path__ = [os.path.join(ALGORITHMS, name)]
    sys.modules[dotted] = pkg
    if name != "mariadb":
        _mariadb_module("mariadb")
    spec = importlib.util.spec_from_file_location(
        f"{dotted}.module", os.path.join(ALGORITHMS, name, "module.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{dotted}.module"] = mod
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = was
    return vb, mod


@pytest.mark.parametrize("env", ("0", "8", "32"))
def test_mariadb_loads_with_one_thread_whatever_is_asked(monkeypatch, env):
    vb, mod = _mariadb_module()
    monkeypatch.setenv("VB_INSERT_THREADS", env)
    assert mod.MARIADB.max_insert_threads == 1
    assert vb.VBMySQLBase._resolve_insert_threads(
        mod.MARIADB.max_insert_threads) == 1


def test_mariadb123_shares_the_cap():
    _, mod = _mariadb_module("mariadb123")
    assert mod.MariaDB123.dialect.max_insert_threads == 1


def test_percona_keeps_its_parallel_load(monkeypatch):
    """The cap is MHNSW's, not the family's: Percona's HNSW is not capped."""
    vb, percona = _load_overlay()
    monkeypatch.setenv("VB_INSERT_THREADS", "8")
    cap = percona.PerconaServer.dialect.max_insert_threads
    assert cap is None
    assert vb.VBMySQLBase._resolve_insert_threads(cap) == 8


def test_no_cap_leaves_the_default_alone(monkeypatch):
    vb, _ = _load_overlay()
    monkeypatch.setenv("VB_INSERT_THREADS", "0")
    expected = max(1, min(8, len(os.sched_getaffinity(0))))
    assert vb.VBMySQLBase._resolve_insert_threads() == expected
    assert vb.VBMySQLBase._resolve_insert_threads(None) == expected


def test_the_constructor_applies_the_dialect_s_cap():
    src = open(os.path.join(VB_ROOT, "overlay", "ann-benchmarks",
                            "ann_benchmarks", "vb_mysql.py")).read()
    assert ("self._insert_threads = self._resolve_insert_threads(\n"
            "            self.dialect.max_insert_threads)") in src
