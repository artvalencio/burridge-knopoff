import numpy as np
import pytest

from burridge_knopoff import gravity, grid
from burridge_knopoff.config import G, SimulationConfig

MASS = 1e14


def test_point_mass_potential_and_superposition():
    blocks = np.array([[0.0, 0.0, 0.0], [3000.0, 0.0, 0.0]])
    sensors = np.array([[0.0, 0.0, 4000.0]])
    expected = -G * MASS / 4000.0 - G * MASS / 5000.0
    assert gravity.potential(blocks, sensors, MASS)[0] == pytest.approx(expected, rel=1e-14)


def test_acceleration_points_towards_the_mass():
    g = gravity.acceleration(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1000.0]]), MASS)
    np.testing.assert_allclose(g[0], [0.0, 0.0, -G * MASS / 1000.0 ** 2])


def test_acceleration_is_minus_the_gradient_of_the_potential():
    rng = np.random.default_rng(0)
    blocks = rng.random((5, 3)) * 4000
    sensor = np.array([[1500.0, 2200.0, 5000.0]])
    h = 0.5
    grad = [(gravity.potential(blocks, sensor + h * e, MASS) - gravity.potential(blocks, sensor - h * e, MASS))[0] / (2 * h)
            for e in np.eye(3)]
    np.testing.assert_allclose(gravity.acceleration(blocks, sensor, MASS)[0], -np.array(grad), rtol=1e-6)


@pytest.fixture
def sensors(small_config):
    return gravity.SensorArray(small_config)


def test_no_displacement_no_change(sensors, small_config):
    np.testing.assert_array_equal(sensors.potential_change(np.zeros(small_config.n_blocks)), 0.0)


def test_change_of_one_displaced_block_is_exact(sensors, small_config):
    u = np.zeros(small_config.n_blocks)
    u[7] = 2.5
    # reference in 50-digit decimal arithmetic (a float64 difference of 1/r terms
    # would itself lose ~1e-11 to cancellation)
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    gm = Decimal(G) * Decimal(sensors.block_mass)
    expected = []
    for d in sensors.positions - sensors.block_positions[7]:
        dx, dy, dz = (Decimal(float(c)) for c in d)
        rest = dy * dy + dz * dz
        expected.append(float(-gm * (1 / ((dx - Decimal(2.5)) ** 2 + rest).sqrt() - 1 / (dx * dx + rest).sqrt())))
    np.testing.assert_allclose(sensors.potential_change(u), expected, rtol=1e-13)


def test_change_matches_difference_of_totals(sensors, small_config):
    u = np.random.default_rng(1).random(small_config.n_blocks) * 3
    moved = sensors.block_positions + np.column_stack([u, np.zeros_like(u), np.zeros_like(u)])
    naive = (gravity.potential(moved, sensors.positions, sensors.block_mass)
             - gravity.potential(sensors.block_positions, sensors.positions, sensors.block_mass))
    np.testing.assert_allclose(sensors.potential_change(u), naive, rtol=1e-7)


def test_incremental_updates_equal_full_recomputation(sensors, small_config):
    rng = np.random.default_rng(2)
    u = np.zeros(small_config.n_blocks)
    delta = np.zeros(small_config.n_sensors)
    for __ in range(20):
        slipped = rng.choice(small_config.n_blocks, size=5, replace=False)
        u_new = u.copy()
        u_new[slipped] += rng.random(5) * 2
        sensors.update(delta, u, u_new, slipped)
        u = u_new
    full = sensors.potential_change(u)
    np.testing.assert_allclose(delta, full, rtol=1e-12, atol=1e-12 * np.abs(full).max())


def test_slip_along_x_gives_a_map_symmetric_about_the_x_axis():
    cfg = SimulationConfig(n_blocks_x=5, n_blocks_y=5, n_sensors_x=4, n_sensors_y=4)
    sensors = gravity.SensorArray(cfg)
    change = grid.as_grid(sensors.potential_change(np.ones(cfg.n_blocks)), cfg, sensors=True)
    np.testing.assert_allclose(change, change[::-1, :], rtol=1e-12)
    # mass moves towards +x: the potential (-G m / r) deepens there and rises at -x
    assert np.all(change[:, -1] < 0) and np.all(change[:, 0] > 0)
