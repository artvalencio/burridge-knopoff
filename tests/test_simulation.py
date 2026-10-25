import dataclasses
import json
import os

import numpy as np
import pytest

from burridge_knopoff import gravity
from burridge_knopoff.config import SECONDS_PER_YEAR, SimulationConfig
from burridge_knopoff.simulation import run

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def result():
    cfg = SimulationConfig(n_blocks_x=6, n_blocks_y=4, n_sensors_x=3, n_sensors_y=2,
                           simulation_years=30.0, sample_interval_years=0.1, snapshot_interval_years=1.0, seed=1)
    return run(cfg)


def test_sampling_schedule(result):
    np.testing.assert_allclose(result.times_years, np.arange(301) * 0.1)
    np.testing.assert_allclose(result.snapshot_years, np.arange(31) * 1.0)
    assert result.potentials.shape == (301, 6)
    assert result.displacements.shape == (31, 24) and result.displacements.dtype == np.float32


def test_events_are_ordered_and_inside_the_run(result):
    starts = [e.start_years for e in result.events]
    assert len(starts) > 5
    assert starts == sorted(starts) and 0 < starts[0] and starts[-1] <= 30.0
    for e in result.events:
        assert e.moment_n_m > 0 and e.n_slipped >= 1 and e.duration_s > 0
        assert e.max_slip_m <= e.total_slip_m


def test_potentials_change_only_when_an_event_happened(result):
    changed = np.flatnonzero(np.any(np.diff(result.potentials, axis=0) != 0, axis=1))
    event_samples = {int(np.ceil(e.start_years / 0.1 - 1e-9)) for e in result.events}
    # a change between samples k and k+1 needs an event in (t_k, t_k+1]
    for k in changed:
        assert any(k < s <= k + 1 for s in event_samples)


def test_incremental_potentials_match_the_final_displacements(result):
    sensors = gravity.SensorArray(result.config)
    final = sensors.potential_change(result.final_displacement)
    np.testing.assert_allclose(result.potentials[-1], final, rtol=1e-10, atol=1e-12 * np.abs(final).max())
    np.testing.assert_allclose(result.displacements[-1], result.final_displacement, rtol=1e-6)


def test_seed_reproduces_the_catalogue(result):
    again = run(result.config)
    assert again.events == result.events
    other = run(dataclasses.replace(result.config, seed=7))
    assert other.events != result.events


def test_single_block_events_are_periodic():
    cfg = SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1, heterogeneity=0.0,
                           redraw_thresholds=False, simulation_years=100.0, sample_interval_years=1.0)
    res = run(cfg)
    recurrence = cfg.single_block_recurrence_years()
    first = (cfg.static_force_n - cfg.dynamic_force_n) / cfg.loading_stiffness_n_m / cfg.plate_velocity_m_per_year
    starts = np.array([e.start_years for e in res.events])
    # each slip carries the O(dt^2) error of the integration (~3e-6 at 100 steps per period)
    np.testing.assert_allclose(starts, first + recurrence * np.arange(len(starts)), rtol=1e-5)
    assert len(starts) == int((100 - first) // recurrence) + 1


def test_an_event_running_at_the_end_is_recorded():
    base = SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1, heterogeneity=0.0)
    first_s = (base.static_force_n - base.dynamic_force_n) / (base.loading_stiffness_n_m * base.plate_velocity_m_s)
    cfg = dataclasses.replace(base, simulation_years=(first_s + 0.5) / SECONDS_PER_YEAR, sample_interval_years=0.5)
    res = run(cfg)
    assert len(res.events) == 1 and res.events[0].duration_s > 0.5


def test_progress_is_reported(small_config):
    seen = []
    run(dataclasses.replace(small_config, simulation_years=5.0), progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0) and seen == sorted(seen)


def test_results_can_be_streamed_to_disk(small_config, tmp_path):
    res = run(small_config, store_dir=str(tmp_path))
    assert isinstance(res.displacements, np.memmap) and isinstance(res.potentials, np.memmap)
    in_memory = run(small_config)
    np.testing.assert_array_equal(np.asarray(res.potentials), in_memory.potentials)
    np.testing.assert_array_equal(np.asarray(res.displacements), in_memory.displacements)


# ---- golden run: guards refactoring and performance work ----------------------------
GOLDEN = os.path.join(HERE, 'golden_catalogue.json')
GOLDEN_CONFIG = dict(n_blocks_x=8, n_blocks_y=8, n_sensors_x=3, n_sensors_y=3, simulation_years=40.0,
                     sample_interval_years=0.5, snapshot_interval_years=5.0, seed=2024)


def golden_summary():
    res = run(SimulationConfig(**GOLDEN_CONFIG))
    return {'events': [[e.start_years, e.hypocentre, e.n_slipped, e.total_slip_m, e.magnitude] for e in res.events],
            'final_potentials': res.potentials[-1].tolist()}


def test_golden_run():
    with open(GOLDEN) as fh:
        expected = json.load(fh)
    got = golden_summary()
    assert len(got['events']) == len(expected['events'])
    for g, e in zip(got['events'], expected['events']):
        assert g[1:3] == e[1:3]
        np.testing.assert_allclose([g[0], g[3], g[4]], [e[0], e[3], e[4]], rtol=1e-9)
    np.testing.assert_allclose(got['final_potentials'], expected['final_potentials'], rtol=1e-9)


if __name__ == '__main__':  # regenerate the golden file (only for intended changes)
    with open(GOLDEN, 'w') as fh:
        json.dump(golden_summary(), fh, indent=1)
