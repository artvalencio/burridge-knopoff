"""Runner: stepping, snapshots, per-block random streams, regional moments, energy series."""
import dataclasses

import numpy as np
import pytest

from burridge_knopoff.config import SECONDS_PER_YEAR, SimulationConfig
from burridge_knopoff.model import counter_uniform
from burridge_knopoff.simulation import Runner, region_of_blocks, run

SMALL = dict(n_blocks_x=8, n_blocks_y=6, n_sensors_x=3, n_sensors_y=2, simulation_years=40.0,
             sample_interval_years=0.25, seed=5)


def test_advancing_in_pieces_gives_the_same_catalogue():
    cfg = SimulationConfig(**SMALL)
    whole = run(cfg)
    runner = Runner(cfg)
    for years in (3.0, 7.5, 21.0, 40.0):
        runner.advance(years * SECONDS_PER_YEAR)
    runner.record_until(np.inf)
    pieces = runner.result()
    # splitting a plate load in two changes rounding in the last digits only
    assert [(e.hypocentre, e.n_slipped) for e in pieces.events] == [(e.hypocentre, e.n_slipped) for e in whole.events]
    np.testing.assert_allclose([e.start_years for e in pieces.events], [e.start_years for e in whole.events], rtol=1e-9)
    np.testing.assert_allclose(pieces.gravity, whole.gravity, rtol=1e-9, atol=1e-20)


def test_restoring_a_snapshot_repeats_the_future_exactly():
    for mode in ('global', 'per-block'):
        runner = Runner(SimulationConfig(**SMALL, rng_mode=mode))
        runner.advance(10 * SECONDS_PER_YEAR)
        state = runner.snapshot()
        n = len(runner.events)
        runner.advance(20 * SECONDS_PER_YEAR)
        first = runner.events[n:]
        runner.restore(state)
        del runner.events[n:]
        runner.advance(20 * SECONDS_PER_YEAR)
        assert len(first) >= 3
        assert runner.events[n:] == first


def test_counter_streams_depend_only_on_seed_block_and_count():
    u = counter_uniform(3, np.arange(1000), np.zeros(1000, dtype=np.uint64))
    assert 0 <= u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.03
    np.testing.assert_array_equal(counter_uniform(3, [7, 2], [4, 9]), [counter_uniform(3, [7], [4])[0],
                                                                       counter_uniform(3, [2], [9])[0]])
    assert counter_uniform(3, [7], [4])[0] != counter_uniform(4, [7], [4])[0]
    assert counter_uniform(3, [7], [4])[0] != counter_uniform(3, [7], [5])[0]
    # successive draws of one block are uncorrelated
    seq = counter_uniform(1, np.zeros(5000, dtype=np.int64), np.arange(5000))
    assert abs(np.corrcoef(seq[:-1], seq[1:])[0, 1]) < 0.05


def test_per_block_mode_is_reproducible_and_the_default_unchanged():
    a = run(SimulationConfig(**SMALL, rng_mode='per-block'))
    b = run(SimulationConfig(**SMALL, rng_mode='per-block'))
    assert a.events == b.events and len(a.events) > 20
    assert run(SimulationConfig(**SMALL)).events != a.events     # a different (but valid) noise realisation
    with pytest.raises(ValueError, match='rng_mode'):
        SimulationConfig(rng_mode='other').validate()


def test_regional_moments_add_up_to_the_event_moments():
    res = run(SimulationConfig(**SMALL, n_regions_x=4, n_regions_y=3))
    assert res.regional_moment.shape == (len(res.events), 12)
    np.testing.assert_allclose(res.regional_moment.sum(axis=1), [e.moment_n_m for e in res.events], rtol=1e-5)
    regions = region_of_blocks(SimulationConfig(**SMALL, n_regions_x=4, n_regions_y=3))
    assert sorted(set(regions)) == list(range(12)) and regions[0] == 0 and regions[-1] == 11


def test_energy_series_matches_a_direct_computation():
    cfg = SimulationConfig(**{**SMALL, 'sample_interval_years': 0.5}, record_energy=True)
    res = run(cfg)
    assert res.elastic_energy.shape == res.times_years.shape
    # check against stepping a runner to sample times and computing the energy there
    runner = Runner(dataclasses.replace(cfg, record_energy=False), samples=False)
    for i in (5, 33, 70):
        runner.advance(res.times_years[i] * SECONDS_PER_YEAR)
        assert res.elastic_energy[i] == pytest.approx(runner.model.elastic_energy(), rel=1e-9)
    assert run(SimulationConfig(**SMALL)).elastic_energy is None
