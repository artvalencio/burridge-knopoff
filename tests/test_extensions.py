"""Velocity-weakening friction and long-range elastic stress transfer."""
import dataclasses
import math

import numpy as np
import pytest

from burridge_knopoff import grid
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.model import Model


def single(**kw):
    return Model(SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1, heterogeneity=0.0,
                                  redraw_thresholds=False, steps_per_period=400, **kw))


def one_event(model):
    dt, idx = model.time_to_failure()
    model.load(dt)
    return model.slip(idx)


# ---- velocity-weakening friction ---------------------------------------------------

def test_friction_law_drops_at_onset_and_weakens_with_velocity():
    model = single(weakening_velocity_m_s=0.5, onset_drop=0.2)
    cfg = model.config
    excess = cfg.static_force_n - cfg.dynamic_force_n
    assert model.sliding_friction(0.0, 0) == pytest.approx(cfg.dynamic_force_n + 0.8 * excess)
    assert model.sliding_friction(0.5, 0) == pytest.approx(cfg.dynamic_force_n + 0.4 * excess)
    v = np.linspace(0, 10, 50)
    f = [model.sliding_friction(x, 0) for x in v]
    assert np.all(np.diff(f) < 0) and f[-1] > cfg.dynamic_force_n


def test_coulomb_friction_by_default():
    model = single()
    assert model.sliding_friction(0.0, 0) == model.sliding_friction(3.0, 0) == model.config.dynamic_force_n


def test_slow_weakening_is_coulomb_at_the_onset_level():
    # v_w >> slip velocities: friction stays at F_d + (1 - s) (F_s - F_d), so the block
    # slips 2 s (F_s - F_d) / k_p in half a period
    model = single(weakening_velocity_m_s=1e9, onset_drop=0.3)
    cfg = model.config
    excess = cfg.static_force_n - cfg.dynamic_force_n
    result = one_event(model)
    assert result.slip[0] == pytest.approx(2 * 0.3 * excess / cfg.k_loading_n_m, rel=1e-4)
    omega = math.sqrt(cfg.k_loading_n_m / cfg.block_mass_kg)
    assert result.duration_s == pytest.approx(math.pi / omega, abs=2 * cfg.slip_time_step_s)


def test_fast_weakening_tends_to_coulomb_at_the_dynamic_level():
    model = single(weakening_velocity_m_s=1e-9, onset_drop=0.3)
    cfg = model.config
    result = one_event(model)
    assert result.slip[0] == pytest.approx(2 * (cfg.static_force_n - cfg.dynamic_force_n) / cfg.k_loading_n_m, rel=1e-3)


def test_weakening_slips_more_than_the_onset_level_alone():
    slow = one_event(single(weakening_velocity_m_s=1e9, onset_drop=0.3)).slip[0]
    weak = one_event(single(weakening_velocity_m_s=0.3, onset_drop=0.3)).slip[0]
    fast = one_event(single(weakening_velocity_m_s=1e-9, onset_drop=0.3)).slip[0]
    assert slow < weak < fast


@pytest.mark.parametrize('kw', [dict(weakening_velocity_m_s=0.3, onset_drop=0.2),
                                dict(interaction_range_blocks=3),
                                dict(weakening_velocity_m_s=0.3, onset_drop=0.2, interaction_range_blocks=3)])
def test_energy_balance_with_extensions(kw):
    model = Model(SimulationConfig(n_blocks_x=9, n_blocks_y=7, n_sensors_x=2, n_sensors_y=2, seed=3, **kw))
    for __ in range(15):
        dt, idx = model.time_to_failure()
        model.load(dt)
        before = model.elastic_energy()
        result = model.slip(idx)
        balance = before + result.plate_work_j - model.elastic_energy() - result.dissipated_j
        assert abs(balance) < 2e-3 * result.dissipated_j
        assert np.all(model.force() <= model.static_force * (1 + 1e-12))


@pytest.mark.parametrize('field,value', [('onset_drop', 0.0), ('onset_drop', 1.5), ('weakening_velocity_m_s', -1.0),
                                         ('interaction_range_blocks', 0), ('interaction_exponent', 0.0)])
def test_invalid_extension_parameters(field, value):
    kw = {field: value}
    if field == 'onset_drop':
        kw['weakening_velocity_m_s'] = 0.5
    with pytest.raises(ValueError, match=field):
        SimulationConfig(**kw).validate()


# ---- long-range elastic stress transfer -----------------------------------------------

@pytest.fixture
def long_range():
    return SimulationConfig(n_blocks_x=7, n_blocks_y=5, n_sensors_x=2, n_sensors_y=2, interaction_range_blocks=3, seed=4)


def pairwise_stiffness(cfg):
    pos = grid.block_positions(cfg)[:, :2] / cfg.block_spacing_m
    r = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    with np.errstate(divide='ignore'):
        k = cfg.k_coupling_n_m * r ** (-cfg.interaction_exponent)
    k[(r == 0) | (r > cfg.interaction_range_blocks + 1e-9)] = 0.0
    return k


def test_nearest_neighbour_stencil_is_the_default(small_config):
    drow, dcol, weight = grid.interaction_stencil(small_config)
    assert list(zip(drow, dcol)) == [(0, -1), (0, 1), (-1, 0), (1, 0)] and np.all(weight == 1.0)


def test_long_range_coupling_matches_the_pairwise_sum(long_range):
    k = pairwise_stiffness(long_range)
    u = np.random.default_rng(0).random(long_range.n_blocks)
    expected = (k * (u[None, :] - u[:, None])).sum(axis=1)
    np.testing.assert_allclose(Model(long_range).coupling_force(u), expected, rtol=1e-12, atol=1e-3)


def test_long_range_coupling_is_translation_invariant_and_conservative(long_range):
    model = Model(long_range)
    np.testing.assert_allclose(model.coupling_force(np.full(long_range.n_blocks, 3.0)), 0.0, atol=1e-2)
    u = np.random.default_rng(1).random(long_range.n_blocks)
    assert abs(model.coupling_force(u).sum()) < 1e-9 * long_range.k_coupling_n_m


def test_long_range_elastic_energy_matches_the_pairwise_sum(long_range):
    model = Model(long_range)
    model.u[:] = np.random.default_rng(2).random(long_range.n_blocks)
    k = pairwise_stiffness(long_range)
    du = model.u[None, :] - model.u[:, None]
    expected = 0.25 * (k * du ** 2).sum() + 0.5 * long_range.k_loading_n_m * ((model.plate_m - model.u) ** 2).sum()
    assert model.elastic_energy() == pytest.approx(expected, rel=1e-12)


def test_long_range_maintained_forces_stay_exact(long_range):
    model = Model(dataclasses.replace(long_range, n_blocks_x=10, n_blocks_y=8))
    for __ in range(200):
        one_event(model)
    np.testing.assert_allclose(model.current_force, model.force(), rtol=1e-12, atol=1e-2)


def test_time_step_accounts_for_all_couplings(long_range):
    weights = grid.interaction_stencil(long_range)[2]
    stiffness = long_range.k_loading_n_m + long_range.k_coupling_n_m * weights.sum()
    expected = 2 * math.pi / math.sqrt(stiffness / long_range.block_mass_kg) / long_range.steps_per_period
    assert long_range.slip_time_step_s == pytest.approx(expected)
