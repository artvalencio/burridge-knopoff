import numpy as np

from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run, run_ensemble


def config():
    return SimulationConfig(n_blocks_x=5, n_blocks_y=4, n_sensors_x=2, n_sensors_y=2,
                            simulation_years=20.0, sample_interval_years=0.5)


def test_ensemble_runs_one_simulation_per_seed():
    results = run_ensemble(config(), seeds=[3, 4, 5], n_jobs=1)
    assert [r.config.seed for r in results] == [3, 4, 5]
    for seed, r in zip([3, 4, 5], results):
        single = run(SimulationConfig(**{**config().__dict__, 'seed': seed}))
        assert r.events == single.events


def test_ensemble_results_do_not_depend_on_the_number_of_workers():
    serial = run_ensemble(config(), seeds=range(4), n_jobs=1)
    parallel = run_ensemble(config(), seeds=range(4), n_jobs=2)
    for a, b in zip(serial, parallel):
        assert a.events == b.events
        np.testing.assert_array_equal(a.potentials, b.potentials)
