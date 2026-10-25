import dataclasses
import math

import numpy as np
import pytest

from burridge_knopoff import grid
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.model import Model


@pytest.fixture
def model(small_config):
    return Model(small_config)


# ---- forces ----------------------------------------------------------------

def test_no_coupling_force_at_rest_and_newtons_third_law(model, small_config):
    np.testing.assert_array_equal(model.coupling_force(np.zeros(small_config.n_blocks)), 0.0)
    u = np.random.default_rng(0).random(small_config.n_blocks)
    assert abs(model.coupling_force(u).sum()) < 1e-9 * small_config.coupling_stiffness_n_m


def test_coupling_acts_between_geometric_neighbours_only(model, small_config):
    for idx in range(small_config.n_blocks):
        u = np.zeros(small_config.n_blocks)
        u[idx] = 1.0
        f = model.coupling_force(u)
        pulled = set(np.flatnonzero(f).tolist()) - {idx}
        assert pulled == set(grid.neighbours(idx, small_config))
        np.testing.assert_allclose(f[list(pulled)], small_config.coupling_stiffness_n_m)
        assert f[idx] == pytest.approx(-len(pulled) * small_config.coupling_stiffness_n_m)


def test_total_force_adds_the_loading_spring(model, small_config):
    u = np.random.default_rng(1).random(small_config.n_blocks)
    model.u[:] = u
    expected = model.coupling_force(u) + small_config.loading_stiffness_n_m * (model.plate_m - u)
    np.testing.assert_allclose(model.force(), expected)


# ---- strengths -------------------------------------------------------------

def test_strengths_without_heterogeneity(small_config):
    m = Model(dataclasses.replace(small_config, heterogeneity=0.0))
    np.testing.assert_allclose(m.static_force, small_config.static_force_n)


def test_strengths_are_heterogeneous_and_seeded(small_config):
    m1, m2, m3 = Model(small_config), Model(small_config), Model(dataclasses.replace(small_config, seed=99))
    excess = small_config.static_force_n - small_config.dynamic_force_n
    h = small_config.heterogeneity
    assert np.all(m1.static_force >= small_config.dynamic_force_n + (1 - h) * excess)
    assert np.all(m1.static_force <= small_config.dynamic_force_n + (1 + h) * excess)
    np.testing.assert_array_equal(m1.static_force, m2.static_force)
    assert not np.array_equal(m1.static_force, m3.static_force)


# ---- loading ---------------------------------------------------------------

def test_initial_stress_is_dynamic_and_next_failure_is_the_weakest_block(model, small_config):
    np.testing.assert_allclose(model.force(), small_config.dynamic_force_n)
    dt, idx = model.time_to_failure()
    assert idx == int(np.argmin(model.static_force))
    rate = small_config.loading_stiffness_n_m * small_config.plate_velocity_m_s
    assert dt == pytest.approx((model.static_force.min() - small_config.dynamic_force_n) / rate)
    model.load(dt)
    assert model.force()[idx] == pytest.approx(model.static_force[idx], rel=1e-12)
    assert model.time_s == pytest.approx(dt)


# ---- dynamics: analytic oracle -----------------------------------------------

@pytest.fixture
def single():
    return Model(SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1,
                                  heterogeneity=0.0, redraw_thresholds=False, steps_per_period=400))


def test_single_block_stick_slip_cycle(single):
    cfg = single.config
    kp, excess = cfg.loading_stiffness_n_m, cfg.static_force_n - cfg.dynamic_force_n
    omega = math.sqrt(kp / cfg.block_mass_kg)
    dt, idx = single.time_to_failure()
    assert dt == pytest.approx(excess / (kp * cfg.plate_velocity_m_s))
    single.load(dt)
    result = single.slip(idx)
    assert result.slip[0] == pytest.approx(2 * excess / kp, rel=1e-4)
    assert result.duration_s == pytest.approx(math.pi / omega, abs=2 * cfg.slip_time_step_s)
    assert result.peak_velocity_m_s == pytest.approx(excess / kp * omega, rel=1e-3)
    assert single.v[0] == 0.0
    # reloading from 2 F_d - F_s back to F_s
    dt2, __ = single.time_to_failure()
    assert dt2 == pytest.approx(2 * excess / (kp * cfg.plate_velocity_m_s), rel=1e-4)


# ---- dynamics: invariants ------------------------------------------------------

def run_one_event(model):
    dt, idx = model.time_to_failure()
    model.load(dt)
    return model.slip(idx)


def test_energy_balance_during_an_event():
    cfg = SimulationConfig(n_blocks_x=8, n_blocks_y=8, n_sensors_x=2, n_sensors_y=2, seed=3)
    model = Model(cfg)
    for __ in range(10):
        dt, idx = model.time_to_failure()
        model.load(dt)
        before = model.elastic_energy()
        result = model.slip(idx)
        after = model.elastic_energy()
        balance = before + result.plate_work_j - after - result.dissipated_j
        assert abs(balance) < 1e-3 * result.dissipated_j


def test_after_an_event_every_block_is_stuck_and_slip_is_forward(small_config):
    model = Model(small_config)
    for __ in range(20):
        u_before = model.u.copy()
        result = run_one_event(model)
        assert np.all(model.v == 0)
        assert np.all(model.force() <= model.static_force * (1 + 1e-12))
        assert np.all(model.u - u_before >= 0)
        np.testing.assert_allclose(result.slip, (model.u - u_before)[result.slipped])
        assert set(np.flatnonzero(model.u != u_before)) <= set(result.slipped)


def test_redrawn_strengths_change_only_where_blocks_slipped(small_config):
    model = Model(small_config)
    before = model.static_force.copy()
    result = run_one_event(model)
    changed = np.flatnonzero(model.static_force != before)
    assert set(changed) <= set(result.slipped) and len(changed) > 0
    quenched = Model(dataclasses.replace(small_config, redraw_thresholds=False))
    before = quenched.static_force.copy()
    run_one_event(quenched)
    np.testing.assert_array_equal(quenched.static_force, before)


def test_events_are_reproducible_with_a_seed(small_config):
    a, b = Model(small_config), Model(small_config)
    for __ in range(10):
        ra, rb = run_one_event(a), run_one_event(b)
        np.testing.assert_array_equal(ra.slipped, rb.slipped)
    np.testing.assert_array_equal(a.u, b.u)
