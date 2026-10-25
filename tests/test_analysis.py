import dataclasses

import numpy as np
import pytest

from burridge_knopoff import analysis
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.events import Event


def gutenberg_richter_sample(b, mc, n, seed=0, upper=None):
    """Magnitudes with N(>=m) ~ 10^(-b m) above mc (optionally truncated at upper)."""
    rng = np.random.default_rng(seed)
    beta = b * np.log(10)
    if upper is None:
        return mc + rng.exponential(1 / beta, n)
    u = rng.random(n)  # inverse CDF of the truncated exponential
    return mc - np.log(1 - u * (1 - np.exp(-beta * (upper - mc)))) / beta


@pytest.mark.parametrize('b', [0.7, 1.0, 1.3])
def test_aki_estimator_recovers_b(b):
    m = gutenberg_richter_sample(b, 4.0, 20000, seed=int(b * 10))
    est = analysis.b_value(m, mc=4.0)
    assert est.n == 20000 and est.sigma == pytest.approx(est.b / np.sqrt(20000), rel=1e-9)
    assert abs(est.b - b) < 3 * est.sigma


def test_events_below_mc_are_ignored():
    m = np.concatenate([gutenberg_richter_sample(1.0, 4.0, 5000, seed=1), np.full(3000, 3.5)])
    assert analysis.b_value(m, mc=4.0).n == 5000


@pytest.mark.parametrize('b', [0.6, 1.0])
def test_truncated_estimator_is_unbiased_where_aki_is_not(b):
    m = gutenberg_richter_sample(b, 5.0, 20000, seed=3, upper=6.0)
    est = analysis.b_value(m, mc=5.0, upper=6.0)
    assert abs(est.b - b) < 3 * est.sigma
    naive = analysis.b_value(m, mc=5.0)
    assert naive.b - b > 0.2  # Aki ignores the upper edge of the window


def test_truncated_estimator_tends_to_aki_for_a_far_upper_edge():
    m = gutenberg_richter_sample(1.0, 4.0, 5000, seed=4)
    assert analysis.b_value(m, 4.0, upper=4.0 + 60).b == pytest.approx(analysis.b_value(m, 4.0).b, rel=1e-6)


def test_magnitude_of_completeness_finds_the_rollover():
    rng = np.random.default_rng(5)
    m = gutenberg_richter_sample(1.0, 2.0, 200000, seed=5)
    detected = m[rng.random(len(m)) < np.minimum(1.0, np.exp(6 * (m - 3.0)))]  # incomplete below 3
    assert analysis.magnitude_of_completeness(detected) == pytest.approx(3.0, abs=0.15)


def test_local_b_values_on_a_pure_power_law():
    m = gutenberg_richter_sample(0.9, 4.0, 50000, seed=6)
    windows = analysis.local_b_values(m, width=1.0, step=0.5, min_events=200)
    assert [w.mc for w in windows][:3] == [4.0, 4.5, 5.0]
    for w in windows:
        assert w.upper == pytest.approx(w.mc + 1.0) and w.n >= 200
        assert abs(w.b - 0.9) < 3 * w.sigma + 0.02


def test_too_few_events_is_an_error():
    with pytest.raises(ValueError, match='events'):
        analysis.b_value(np.array([4.2]), mc=4.0)


def make_result(magnitudes, starts, config):
    from burridge_knopoff.simulation import SimulationResult
    events = [Event(start_years=t, duration_s=1.0, hypocentre=0, n_slipped=1, total_slip_m=1.0, max_slip_m=1.0,
                    peak_velocity_m_s=1.0, moment_n_m=10 ** (1.5 * m + 9.1), magnitude=m)
              for m, t in zip(magnitudes, starts)]
    return SimulationResult(config=config, times_years=np.zeros(1), potentials=np.zeros((1, 1)),
                            snapshot_years=np.zeros(1), displacements=np.zeros((1, 1)), events=events,
                            final_displacement=np.zeros(1), dissipated_energy_j=0.0)


def test_statistics_skip_the_spin_up():
    cfg = SimulationConfig(spinup_years=10.0, simulation_years=100.0)
    late = gutenberg_richter_sample(1.0, 4.0, 4000, seed=7)
    result = make_result(np.r_[np.full(500, 8.0), late], np.r_[np.full(500, 5.0), np.linspace(11, 99, 4000)], cfg)
    stats = analysis.gutenberg_richter(result)
    assert stats['n_events'] == 4000 and stats['spinup_years'] == 10.0
    assert abs(stats['b'].b - 1.0) < 3 * stats['b'].sigma
    assert analysis.gutenberg_richter(result, spinup_years=0.0)['n_events'] == 4500


def test_spinup_is_validated():
    with pytest.raises(ValueError, match='spinup_years'):
        SimulationConfig(spinup_years=-1.0).validate()
    assert any('spin-up' in w for w in SimulationConfig(spinup_years=200.0).check_parameters())
