"""Nonlinear-dynamics estimators on reference systems, then on the model (plan, section 6.10)."""
import math

import numpy as np
import pytest

from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.dynamics import embedding as em
from burridge_knopoff.dynamics import lyapunov as ly
from burridge_knopoff.dynamics import ordinal as od
from burridge_knopoff.dynamics import recurrence as rq
from burridge_knopoff.dynamics import surrogates as sg
from burridge_knopoff.dynamics import synchronization as sync
from burridge_knopoff.model import Model


@pytest.fixture(scope='module')
def henon():
    x = np.zeros(6000)
    y = 0.0
    x[0] = 0.1
    for i in range(1, len(x)):
        x[i], y = 1 - 1.4 * x[i - 1] ** 2 + y, 0.3 * x[i - 1]
    return x[1000:]


@pytest.fixture(scope='module')
def lorenz():
    # x component, sampled every 0.02 time units
    def f(s):
        return np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])
    dt, s, out = 0.01, np.array([1.0, 1.0, 1.0]), []
    for i in range(60000):
        k1 = f(s)
        k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= 10000 and i % 2 == 0:
            out.append(s[0])
    return np.array(out)


# ---- reference systems -----------------------------------------------------------------
def test_henon_dimension_and_exponent(henon):
    fnn = em.false_nearest_neighbours(henon, 1, max_dim=4)
    assert fnn[0] > 0.3 and fnn[1] < 0.01                          # embedding dimension 2
    assert em.correlation_dimension(em.delay_vectors(henon, 2, 1), theiler=2)['dimension'] == pytest.approx(1.21, abs=0.08)
    assert ly.rosenstein(henon, 2, 1, theiler=5, max_steps=12, fit=(0, 6))['slope'] == pytest.approx(0.419, abs=0.04)
    assert ly.kantz(henon, 2, 1, eps=0.01, theiler=5, max_steps=10, fit=(1, 6))['slope'] == pytest.approx(0.419, abs=0.04)


def test_lorenz_delay_dimension_and_exponent(lorenz):
    tau = em.first_minimum(em.auto_mutual_information(lorenz, 40))
    assert 8 <= tau <= 14                                          # ~0.2 time units
    fnn = em.false_nearest_neighbours(lorenz, tau, max_dim=4, theiler=2 * tau)
    assert fnn[1] > 0.05 and fnn[2] < 0.02                         # embedding dimension 3
    d2 = em.correlation_dimension(em.delay_vectors(lorenz, 3, tau), theiler=50, fit=(1e-4, 1e-2))['dimension']
    assert d2 == pytest.approx(2.05, abs=0.12)
    slope = ly.rosenstein(lorenz, 3, tau, theiler=50, max_steps=150, fit=(40, 150))['slope'] / 0.02
    assert slope == pytest.approx(0.906, rel=0.12)
    e1, e2 = em.cao(lorenz[:8000], tau, max_dim=5, theiler=2 * tau)
    assert np.all(np.diff(e1) > 0) and e1[-1] > 0.95               # E1 rises and saturates near 1
    assert e2.min() < 0.5                                          # E2 != 1: deterministic, not noise
    assert em.eckmann_ruelle_bound(len(lorenz)) == pytest.approx(2 * math.log10(len(lorenz)))


def test_rqa_and_ordinal_patterns_separate_order_from_noise():
    t = np.arange(3000) * 0.1
    sine = rq.rqa(rq.recurrence_matrix(em.delay_vectors(np.sin(t), 2, 16), rate=0.05), theiler=1)
    noise = rq.rqa(rq.recurrence_matrix(np.random.default_rng(1).normal(size=1500), rate=0.05), theiler=1)
    assert sine['DET'] > 0.95 and noise['DET'] < 0.3 and sine['RR'] == pytest.approx(0.05, abs=0.005)
    assert sine['L_max'] > 10 * noise['L_max'] and sine['LAM'] > noise['LAM']
    white = np.random.default_rng(2).normal(size=20000)
    assert od.permutation_entropy(white, 5) > 0.99
    h, c = od.statistical_complexity(white, 5)
    assert c < 0.05
    starts, values = rq.windowed_rqa(em.delay_vectors(np.sin(t), 2, 16), window=300, step=300)
    assert len(starts) == 9 and np.all(values['DET'] > 0.9)


def test_chaos_sits_low_on_entropy_and_high_on_complexity(henon):
    h, c = od.statistical_complexity(henon, 5)
    assert h < 0.75 and c > 0.3


def test_surrogates_keep_what_they_should():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=4096))
    s = sg.iaaft(x, rng=1)
    np.testing.assert_allclose(np.sort(s), np.sort(x))
    spec = lambda v: np.abs(np.fft.rfft(v - v.mean()))
    assert np.corrcoef(np.log(spec(s)[1:200]), np.log(spec(x)[1:200]))[0, 1] > 0.9
    t = np.cumsum(rng.exponential(1.0, 500))
    shuffled = sg.interval_shuffle(t, rng=2)
    np.testing.assert_allclose(np.sort(np.diff(shuffled)), np.sort(np.diff(t)))
    value, sur, p = sg.surrogate_test(lambda v: np.corrcoef(v[:-1], v[1:])[0, 1], x, lambda v, r: r.permutation(v),
                                      n_surrogates=49)
    assert p == pytest.approx(1 / 50)


def test_jaccard_similarity():
    sim = rq.jaccard_similarity([[1, 2, 3], [2, 3, 4], [7]])
    assert sim[0, 1] == pytest.approx(0.5) and sim[0, 2] == 0 and np.all(np.diag(sim) == 1)


# ---- synchronisation ---------------------------------------------------------------
def test_phase_and_event_synchronisation():
    rng = np.random.default_rng(3)
    t1 = np.cumsum(rng.uniform(5, 15, 300))
    times = np.linspace(t1[1], t1[-2], 5000)
    phi = sync.event_phase(t1, times)
    assert sync.phase_locking_value(phi, phi) == pytest.approx(1.0)
    other = sync.event_phase(np.cumsum(rng.uniform(5, 15, 300)), times)
    assert sync.phase_locking_value(phi, other) < 0.2
    assert np.all(sync.kuramoto_order(np.column_stack([phi, phi, phi]))[np.isfinite(phi)] > 0.999)
    Q, q = sync.event_synchronization(t1, t1 + 0.5)                # t1 leads by 0.5
    assert Q > 0.9 and q > 0.9
    independent = np.cumsum(rng.uniform(5, 15, 300))
    Q_fixed, q_fixed = sync.event_synchronization(t1, independent, tau=1.0)
    assert Q_fixed < 0.3 and abs(q_fixed) < 0.15
    assert sync.event_synchronization(t1, t1 + 0.5, tau=1.0)[0] > 0.9
    # the adaptive window gives chance coincidences at these rates, but no direction
    assert abs(sync.event_synchronization(t1, independent)[1]) < 0.15
    assert sync.rotation_number(np.arange(10) * 3.0, 2.0) == pytest.approx(1.5)


class _Tide:
    def __init__(self, amplitude, period, end, step=600.0):
        self.step_s = step
        self.times = np.arange(int(end / step) + 4) * step
        self.coulomb = np.zeros((len(self.times), 3))
        self.coulomb[:, 0] = amplitude * np.sin(2 * math.pi * self.times / period)

    def max_spatial_variation(self, *args):
        return 0.0


def _forced_single_block(amplitude_hours, n_events=40):
    # a fast plate (recurrence ~0.44 years) keeps the tide table short; locking depends on the
    # tidal amplitude in hours of loading only
    model = Model(SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1, heterogeneity=0.0,
                                   redraw_thresholds=False, plate_velocity_m_per_year=9.0))
    rate = model.kp * model.vp
    period = 12.42 * 3600
    end = (n_events + 2) * 2 * (model.static_force[0] - model.fd) / rate
    tide = _Tide(amplitude_hours * 3600 * rate, period, end, step=3600.0)
    times = []
    for __ in range(n_events):
        wait, idx = model.time_to_failure_tidal(tide, end)
        model.load(wait)
        times.append(model.time_s)
        model.slip(idx)
    return np.array(times), period


def test_strong_periodic_forcing_locks_the_firing_phase():
    locked, period = _forced_single_block(amplitude_hours=200)
    free, __ = _forced_single_block(amplitude_hours=0.0)
    plv = lambda t: abs(np.mean(np.exp(1j * sync.firing_phases(t[5:], period))))
    assert plv(locked) > 0.9                                        # every failure at the same tidal phase
    assert plv(free) < 0.6                                          # free recurrence: phase drifts
    assert sync.rotation_number(locked, period) == pytest.approx(np.mean(np.diff(locked)) / period)


# ---- the model -------------------------------------------------------------------------
DET = dict(n_sensors_x=1, n_sensors_y=1, redraw_thresholds=False, simulation_years=3000.0, spinup_years=0.0)


def test_single_block_event_map_is_stable():
    sp = ly.event_map_spectrum(SimulationConfig(**DET, n_blocks_x=1, n_blocks_y=1, heterogeneity=0.0),
                               n_events=50, n_transient=20)
    assert sp['per_event'][0] < 0


def test_event_map_and_two_trajectory_exponents_agree_on_a_chaotic_grid():
    cfg = SimulationConfig(**DET, n_blocks_x=8, n_blocks_y=8, seed=3)
    sp = ly.event_map_spectrum(cfg, n_events=300, n_transient=300)
    assert sp['per_event'][0] > 0.1 and sp['kaplan_yorke'] > 2      # chaotic, several unstable directions
    assert sp['retries'] < 0.05 * 300 * 64
    tt = ly.two_trajectory(cfg, delta=1e-9, renorm_years=1.0, total_years=300.0, start_years=100.0)
    assert tt['per_year'] == pytest.approx(sp['per_year'][0], rel=0.15)
    assert ly.kaplan_yorke([1.0, -0.5, -2.0]) == pytest.approx(2.25)


def test_identical_trajectories_do_not_diverge():
    cfg = SimulationConfig(**DET, n_blocks_x=6, n_blocks_y=4, seed=2)
    ref, pert = ly.perturbed_pair(cfg, 50.0, 1e-9)
    assert ly.distance(ref, pert) == pytest.approx(1e-9)
    pert.restore(ref.snapshot())
    for t in (60.0, 90.0):
        ref.advance(t * 3.15576e7)
        pert.advance(t * 3.15576e7)
        assert ly.distance(ref, pert) == 0.0


def test_finite_size_exponent_and_predictability():
    cfg = SimulationConfig(**DET, n_blocks_x=8, n_blocks_y=8, seed=3)
    fs = ly.finite_size(cfg, delta0=1e-10, n_scales=12, check_years=0.25, max_years=60.0, start_years=100.0,
                        n_samples=3)
    small = fs['exponent'][:4]
    assert np.all(np.isfinite(small)) and np.all(small > 0)
    hz = ly.predictability_horizon(cfg, delta=1e-9, horizon_years=80.0, start_years=100.0, n_samples=3,
                                   magnitude_classes=(0.0,))
    assert np.all(np.isfinite(hz[0.0])) and np.all(hz[0.0] > 0)
