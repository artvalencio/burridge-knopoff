"""Precursory creep (rate-and-state nucleation) against exact and brute-force solutions."""
import dataclasses
import math

import numpy as np
import pytest

from burridge_knopoff.config import SECONDS_PER_YEAR, SimulationConfig
from burridge_knopoff.model import Model
from burridge_knopoff.nucleation import _tide, creep_over, time_to_instability
from burridge_knopoff.simulation import Runner, run


def _config(**kw):
    base = dict(n_blocks_x=3, n_blocks_y=3, n_sensors_x=2, n_sensors_y=2, simulation_years=1.0,
                spinup_years=0.0, preslip=True)
    base.update(kw)
    return SimulationConfig(**base)


def _gammas(model, t=None, hourly=None):
    c = 0.0
    if hourly is not None:
        c = np.array([_tide(hourly.coulomb, hourly.step_s, t, x, y)[0] for x, y in zip(model.x_centre, model.y_centre)])
    return (model.current_force - model.static_force + c) / model.config.block_area_m2


def _set_gaps(model, gammas_pa):
    model.static_force[:] = model.current_force - np.asarray(gammas_pa) * model.config.block_area_m2


def _reference(model, t_max, hourly=None, frac=0.002):
    """Brute-force integration of the coupled creep (RK4 on the creep of every block, steps
    of a small fraction of the fastest e-folding time); returns (t_fail, block, creep)."""
    cfg = model.config
    a, area, vs = model.a_sigma, cfg.block_area_m2, cfg.rs_seismic_velocity_m_s
    f0 = model.current_force.copy()
    s0 = model.static_force.copy()
    kp_vp = model.kp * model.vp

    def gamma(delta, t):
        f = f0 + kp_vp * (t - t0) + model.coupling_force(delta) - model.kp * delta
        g = (f - (s0 - model.weaken * delta)) / area
        if hourly is not None:
            g = g + np.array([_tide(hourly.coulomb, hourly.step_s, t, x, y)[0]
                              for x, y in zip(model.x_centre, model.y_centre)]) / area
        return g

    def rate(delta, t):
        return vs * np.exp(np.minimum(gamma(delta, t), 0.0) / a)

    t0 = model.time_s
    t, delta = t0, np.zeros(cfg.n_blocks)
    while t < t0 + t_max:
        g = gamma(delta, t)
        if g.max() >= 0:
            return t, int(np.argmax(g)), delta
        v = rate(delta, t)
        speed = np.max(model.h_sigma * v + model.kp * model.vp / area)
        dt = frac * a / speed
        if hourly is not None:
            dt = min(dt, 60.0)
        k1 = rate(delta, t)
        k2 = rate(delta + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rate(delta + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rate(delta + dt * k3, t + dt)
        new = delta + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if gamma(new, t + dt).max() >= 0:
            return t + dt, int(np.argmax(gamma(new, t + dt))), new
        delta, t = new, t + dt
    return np.inf, -1, delta


def test_exact_formulas_agree_with_each_other():
    a, hs = 1e4, 4e8
    for r in (4e-3, 0.0, -2e-3):
        v0 = 1e-11
        t_inst = time_to_instability(v0, r, a, hs)
        # creep up to just before the instability diverges logarithmically, and is finite before
        assert creep_over(v0, r, 0.5 * t_inst, a, hs) > 0
        assert creep_over(v0, r, 1.01 * t_inst, a, hs) == -1.0
        # the creep rate implied by the formula: d(delta)/dh at h = 0 is v0
        h = 1e-3
        assert creep_over(v0, r, h, a, hs) / h == pytest.approx(v0, rel=1e-6)
    # a strongly negative stressing rate never lets the block fail
    assert time_to_instability(1e-15, -1.0, a, hs) == np.inf


def test_single_block_matches_dieterich():
    cfg = _config(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1)
    model = Model(cfg)
    gamma0 = -150e3
    _set_gaps(model, [gamma0])
    a, area = model.a_sigma, cfg.block_area_m2
    r = model.kp * model.vp / area
    v0 = cfg.rs_seismic_velocity_m_s * math.exp(gamma0 / a)
    expected_t = time_to_instability(v0, r, a, model.h_sigma[0])
    failed = model.creep(10 * SECONDS_PER_YEAR)
    assert failed == 0
    assert model.time_s == pytest.approx(expected_t, rel=1e-6)
    # Gamma went from gamma0 to 0 through loading and creep: r T + H sigma delta = -gamma0
    assert model.preslip_m[0] == pytest.approx((-gamma0 - r * expected_t) / model.h_sigma[0], rel=1e-6)
    assert _gammas(model)[0] == pytest.approx(0.0, abs=1e-6 * abs(gamma0))


def test_far_blocks_do_not_creep():
    model = Model(_config())
    _set_gaps(model, np.full(9, -2e6))
    assert model.creep(0.1 * SECONDS_PER_YEAR) == -1
    assert np.all(model.preslip_m == 0)
    assert model.time_s == pytest.approx(0.1 * SECONDS_PER_YEAR)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_coupled_creep_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    model = Model(_config())
    _set_gaps(model, rng.uniform(-200e3, -150e3, 9))
    ref = Model(_config())
    ref.current_force[:] = model.current_force
    ref.static_force[:] = model.static_force
    t_ref, i_ref, creep_ref = _reference(ref, 5 * SECONDS_PER_YEAR)
    failed = model.creep(5 * SECONDS_PER_YEAR)
    assert failed == i_ref
    assert model.time_s == pytest.approx(t_ref, rel=5e-3)
    big = creep_ref > 0.05 * creep_ref.max()
    np.testing.assert_allclose(model.preslip_m[big], creep_ref[big], rtol=0.02)
    # the failing block stops at the instability, not past it
    assert (model.current_force[failed] - model.static_force[failed]) / model.config.block_area_m2 ==         pytest.approx(0.0, abs=0.05 * model.a_sigma)
    # the forces stayed consistent with the displacements
    loaded = model.force()
    np.testing.assert_allclose(model.current_force, loaded, rtol=1e-9, atol=1e-6 * np.abs(loaded).max())


def test_creep_with_tides_matches_brute_force():
    cfg = _config(tides=True, tide_scale=50.0)
    runner = Runner(cfg, samples=False)
    model, hourly = runner.model, runner.hourly
    _set_gaps(model, np.linspace(-170e3, -160e3, 9))
    ref = Model(cfg)
    ref.current_force[:] = model.current_force
    ref.static_force[:] = model.static_force
    t_ref, i_ref, creep_ref = _reference(ref, 0.5 * SECONDS_PER_YEAR, hourly=hourly)
    failed = model.creep(0.5 * SECONDS_PER_YEAR, hourly)
    assert failed == i_ref
    assert model.time_s == pytest.approx(t_ref, rel=5e-3)


def test_preslip_run_conserves_slip_and_records_creep():
    cfg = SimulationConfig(n_blocks_x=20, n_blocks_y=20, n_sensors_x=4, n_sensors_y=4, simulation_years=40.0,
                           spinup_years=10.0, preslip=True)
    res = run(cfg)
    cat = res.catalogue
    assert len(cat) > 50
    # every displacement is seismic slip or creep: the sum of the catalogue slips is the total
    # displacement minus the creep
    total_seismic = cat['total_slip_m'].sum()
    assert total_seismic < res.final_displacement.sum()
    creep_total = res.final_displacement.sum() - total_seismic
    assert creep_total > 0
    # long-term rate: the blocks keep up with the plate
    plate = cfg.plate_velocity_m_per_year * cfg.simulation_years
    assert res.final_displacement.mean() == pytest.approx(plate, rel=0.3)


def test_preslip_gravity_follows_the_displacements():
    cfg = SimulationConfig(n_blocks_x=15, n_blocks_y=15, n_sensors_x=3, n_sensors_y=3, simulation_years=15.0,
                           spinup_years=0.0, sample_interval_years=0.05, preslip=True)
    runner = Runner(cfg)
    runner.advance(runner.end_s)
    runner.flush_creep(tolerance_m=0.0)
    direct = runner.sensors.gravity_change(runner.model.u)
    np.testing.assert_allclose(runner.delta_g, direct, rtol=1e-8, atol=1e-6 * np.abs(direct).max())


def test_preslip_off_is_unchanged():
    cfg = SimulationConfig(n_blocks_x=15, n_blocks_y=15, n_sensors_x=3, n_sensors_y=3, simulation_years=10.0,
                           spinup_years=0.0)
    a = run(cfg)
    b = run(dataclasses.replace(cfg, rs_dc_m=1.0))  # creep parameters are ignored without preslip
    assert a.events == b.events


def test_invalid_creep_parameters():
    with pytest.raises(ValueError, match='rs_dc_m'):
        SimulationConfig(preslip=True, rs_dc_m=1e-3).validate()
    with pytest.raises(ValueError, match='rs_b_over_a'):
        SimulationConfig(preslip=True, rs_b_over_a=0.9).validate()


def test_replay_with_creep_reproduces_the_event():
    from burridge_knopoff.preseismic import replay
    cfg = SimulationConfig(n_blocks_x=20, n_blocks_y=20, n_sensors_x=4, n_sensors_y=4, simulation_years=12.0,
                           spinup_years=0.0, preslip=True, gravity_model='dislocation')
    res = run(cfg)
    cat = res.catalogue
    late = np.flatnonzero(cat['start_years'] > 6.0)
    k = int(late[np.argmax(cat['magnitude'][late])])
    window = replay(res, k, window_years=0.5, frame_hours=6.0)
    assert window.event == res.events[k]
    # between events the slip deficit grows and the blocks creep: the frames are not constant
    g = window.gravity
    assert np.abs(np.diff(g, axis=0)).max() > 0


def test_decluster_keeps_mainshocks_only():
    from burridge_knopoff.tide_analysis import decluster
    day = 86400.0
    t = np.array([0.0, 0.5, 2.0, 3.0, 30.0, 30.1]) * day
    x = np.array([0.0, 5e3, 100e3, 1e3, 0.0, 2e3])
    y = np.zeros(6)
    mw = np.array([6.0, 5.0, 5.5, 6.5, 5.0, 4.0])
    keep = decluster(t, x, y, mw, window_days=10.0, radius_m=30e3)
    # 1: aftershock of 0; 2: too far; 3: larger than 0, a mainshock; 4: window over; 5: aftershock of 4
    np.testing.assert_array_equal(keep, [True, False, True, True, True, False])
