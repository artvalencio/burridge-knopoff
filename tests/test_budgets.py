"""Time and memory budgets and long-run statistics (run with: pytest -m slow).

Budgets have about 3x margin over the measured values (benchmarks/run.py).
"""
import time
import tracemalloc

import numpy as np
import pytest

from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module', autouse=True)
def compiled():
    run(SimulationConfig(n_blocks_x=3, n_blocks_y=3, simulation_years=5))


def test_default_run_time():
    start = time.perf_counter()
    run(SimulationConfig())  # 100 x 100 blocks, 100 years: ~12 s measured
    assert time.perf_counter() - start < 40


def test_default_run_memory():
    tracemalloc.start()
    run(SimulationConfig())  # ~44 MB measured (the old code needed ~4.8 GB)
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    assert peak < 120


def test_streamed_run_memory(tmp_path):
    cfg = SimulationConfig(n_blocks_x=50, n_blocks_y=50, simulation_years=100, snapshot_interval_years=0.05)
    tracemalloc.start()
    run(cfg, store_dir=str(tmp_path))  # 2001 snapshots x 2500 blocks go to disk
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    assert peak < 30


def test_heterogeneous_fault_produces_a_broad_magnitude_distribution():
    res = run(SimulationConfig(n_blocks_x=50, n_blocks_y=50, simulation_years=200.0))
    moments = np.array([e.moment_n_m for e in res.events])
    magnitudes = np.array([e.magnitude for e in res.events])
    assert moments.max() / moments.min() > 1e3  # more than three decades of moment
    # fewer large events than small ones (Gutenberg-Richter-like decay)
    counts = np.histogram(magnitudes, bins=np.arange(np.floor(magnitudes.min()), magnitudes.max() + 0.5, 0.5))[0]
    assert counts[0] > 10 * counts[-1]
