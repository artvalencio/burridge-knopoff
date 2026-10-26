"""Time and memory budgets and long-run statistics (run with: pytest -m slow).

Budgets have about 3x margin over the measured values (benchmarks/run.py).
"""
import time
import tracemalloc

import numpy as np
import pytest

from burridge_knopoff import analysis
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module', autouse=True)
def compiled():
    run(SimulationConfig(n_blocks_x=3, n_blocks_y=3, simulation_years=5))


def test_default_run_time():
    start = time.perf_counter()
    run(SimulationConfig())  # 100 x 100 blocks, 100 years: ~10 s measured
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


def local_slopes(config):
    stats = analysis.gutenberg_richter(run(config))
    return stats, [(w.mc - stats['mc'], w.b) for w in stats['local']]


def test_geometric_springs_give_a_power_law_over_two_magnitude_units():
    # default 100 x 100 x 100 years: b ~ 0.55-0.8 from Mc up to Mc + 3 (measured)
    stats, slopes = local_slopes(SimulationConfig())
    assert 0.5 < stats['b'].b < 1.0
    assert all(b < 1.0 for offset, b in slopes if offset <= 2.0)


def test_weak_coupling_cuts_the_distribution_off_early():
    # the version-2.0 springs (k_c / k_p = 2.5): b jumps above 1.4 within one unit of Mc
    stats, slopes = local_slopes(SimulationConfig(coupling_stiffness_n_m=6e13, loading_stiffness_n_m=2.4e13))
    assert any(b > 1.4 for offset, b in slopes if offset <= 1.0)


def test_long_range_run_time():
    # 60 x 60 blocks, range 3 (28 coupled blocks), 150 years: 185 s before
    # incremental forces during slip
    cfg = SimulationConfig(n_blocks_x=60, n_blocks_y=60, simulation_years=150.0, sample_interval_years=1.0,
                           interaction_range_blocks=3)
    run(SimulationConfig(n_blocks_x=4, n_blocks_y=4, simulation_years=40.0, interaction_range_blocks=3))
    start = time.perf_counter()
    run(cfg)
    assert time.perf_counter() - start < 45


def test_velocity_weakening_gives_a_gutenberg_richter_b_value():
    # v_w = 0.1 m/s, onset drop 0.3: b = 0.97 +/- 0.01 measured (two seeds), local
    # slopes 0.85-1.05 over the first two magnitude units; Coulomb friction gives ~0.66
    cfg = SimulationConfig(n_blocks_x=60, n_blocks_y=60, simulation_years=150.0, sample_interval_years=1.0,
                           weakening_velocity_m_s=0.1, onset_drop=0.3)
    stats, slopes = local_slopes(cfg)
    assert 0.85 < stats['b'].b < 1.15
    assert all(0.75 < b < 1.3 for offset, b in slopes if offset <= 2.0)
