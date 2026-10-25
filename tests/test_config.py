import dataclasses

import numpy as np
import pytest

from burridge_knopoff.config import SimulationConfig, SECONDS_PER_YEAR


def test_defaults_are_valid_and_physical():
    cfg = SimulationConfig()
    cfg.validate()
    assert cfg.n_blocks == 100 * 100 and cfg.n_sensors == 10 * 10
    assert cfg.block_area_m2 == pytest.approx(2000.0 ** 2)
    assert cfg.block_mass_kg == pytest.approx(2700 * 2000.0 ** 2 * 10_000)
    assert cfg.plate_velocity_m_s == pytest.approx(0.09 / SECONDS_PER_YEAR)
    assert cfg.static_force_n > cfg.dynamic_force_n > 0


def test_default_parameters_produce_events_within_the_simulation():
    cfg = SimulationConfig()
    assert cfg.single_block_recurrence_years() < cfg.simulation_years / 2  # 44 years; events are mostly collective
    assert cfg.check_parameters() == []


def test_parameter_check_warns_about_long_recurrence():
    cfg = SimulationConfig(loading_stiffness_n_m=1e10)
    assert any('recurrence' in w for w in cfg.check_parameters())


@pytest.mark.parametrize('field,value', [
    ('n_blocks_x', 0), ('n_sensors_y', 0), ('block_spacing_m', -1.0), ('simulation_years', 0.0),
    ('sample_interval_years', 0.0), ('sensor_height_m', 0.0), ('heterogeneity', 1.0), ('heterogeneity', -0.1),
    ('dynamic_friction_stress_pa', 2e7), ('coupling_stiffness_n_m', -1.0), ('steps_per_period', 5),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        SimulationConfig(**{field: value}).validate()


def test_config_is_immutable_and_replaceable():
    cfg = SimulationConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_blocks_x = 3
    assert dataclasses.replace(cfg, n_blocks_x=3).n_blocks == 300


def test_stiffnesses_follow_from_the_fault_geometry():
    cfg = SimulationConfig()
    assert cfg.k_coupling_n_m == pytest.approx(cfg.shear_modulus_pa * cfg.seismogenic_thickness_m)  # 3e14
    assert cfg.k_loading_n_m == pytest.approx(cfg.shear_modulus_pa * cfg.block_area_m2 / cfg.loading_distance_m)
    assert cfg.k_coupling_n_m / cfg.k_loading_n_m == pytest.approx(50)
    thicker = SimulationConfig(seismogenic_thickness_m=20_000.0, loading_distance_m=40_000.0)
    assert thicker.k_coupling_n_m == pytest.approx(2 * cfg.k_coupling_n_m)
    assert thicker.k_loading_n_m == pytest.approx(cfg.k_loading_n_m / 2)


def test_explicit_stiffnesses_override_the_geometry():
    cfg = SimulationConfig(coupling_stiffness_n_m=6e13, loading_stiffness_n_m=2.4e13)
    assert (cfg.k_coupling_n_m, cfg.k_loading_n_m) == (6e13, 2.4e13)


def test_loading_distance_must_be_positive():
    with pytest.raises(ValueError, match='loading_distance_m'):
        SimulationConfig(loading_distance_m=0.0).validate()
