"""Surface displacement and gravity of fault slip in a half-space (Okada 1985, Okubo 1992)."""
import math

import numpy as np
import pytest

from burridge_knopoff import dislocation as dl
from burridge_knopoff.config import G, SimulationConfig
from burridge_knopoff.simulation import Runner


def _okada(x, y, d, dip, length, width, nu=0.25):
    sd, cd = math.sin(math.radians(dip)), math.cos(math.radians(dip))
    f = dl._patch(x, y, d, length, width, sd, cd, 1.0 - 2.0 * nu)
    return -f[:6] / (2 * math.pi), f[6:]


def test_okada_1985_check_list():
    # Okada (1985), Table 2, case 2: x = 2, y = 3, d = 4, dip 70, L = 3, W = 2, lambda = mu
    u, _ = _okada(2.0, 3.0, 4.0, 70.0, 3.0, 2.0)
    expected = [-8.689e-3, -4.298e-3, -2.747e-3,   # strike slip: ux, uy, uz
                -4.682e-3, -3.527e-2, -3.564e-2]   # dip slip
    np.testing.assert_allclose(u, expected, rtol=2e-4)


def test_vertical_fault_is_the_limit_of_steep_ones():
    u90, g90 = _okada(2.0, 3.0, 4.0, 90.0, 3.0, 2.0)
    u89, g89 = _okada(2.0, 3.0, 4.0, 89.99, 3.0, 2.0)
    np.testing.assert_allclose(u90, u89, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(g90, g89, rtol=1e-3, atol=1e-7)


def _config(**kw):
    base = dict(n_blocks_x=2, n_blocks_y=2, block_spacing_m=2000.0, n_sensors_x=2, n_sensors_y=2,
                gravity_model='dislocation', fault_dip_deg=30.0, fault_rake_deg=70.0, fault_top_depth_m=3000.0)
    base.update(kw)
    return SimulationConfig(**base)


def test_four_patches_make_one_twice_as_large():
    cfg = _config()
    L = cfg.block_spacing_m
    rng = np.random.default_rng(0)
    points = rng.uniform(-20e3, 25e3, (30, 2))
    tiled = dl.surface_field(cfg, points, np.arange(4), np.ones(4))
    # one patch covering the four: down dip from the top edge over 2 L, along strike over 2 L
    geo = dl._geometry(cfg)
    big = np.zeros((len(points), 4))
    for k, (X, Y) in enumerate(points):
        d = geo['top_depth'] + 2 * L * geo['sd']
        x_bot = geo['x_top'] + 2 * L * geo['cd']
        f = dl._patch(Y + 0.5 * L, x_bot - X, d, 2 * L, 2 * L, geo['sd'], geo['cd'], geo['m'])
        u1, u2 = geo['cr'], geo['sr']
        ux = -(u1 * f[0] + u2 * f[3]) / (2 * math.pi)
        uy = -(u1 * f[1] + u2 * f[4]) / (2 * math.pi)
        uz = -(u1 * f[2] + u2 * f[5]) / (2 * math.pi)
        big[k] = (-uy, ux, uz, geo['rho_g'] * (u1 * f[6] + u2 * f[7]) - geo['beta'] * uz)
    np.testing.assert_allclose(tiled, big, rtol=1e-9, atol=1e-12 * np.abs(big).max())


def test_coarse_patches_are_exact_for_uniform_slip():
    cfg = _config(n_blocks_x=4, n_blocks_y=4)
    points = np.random.default_rng(2).uniform(-30e3, 40e3, (20, 2))
    fine = dl.surface_field(cfg, points, np.arange(16), np.full(16, 1.5))
    coarse = dl.surface_field(cfg, points, np.arange(16), np.full(16, 1.5), coarsen=2)
    np.testing.assert_allclose(coarse, fine, rtol=1e-9, atol=1e-12 * np.abs(fine).max())


def test_shear_slip_conserves_mass():
    # the surface mass change (gravity without the free-air term, over 2 pi G) integrates to 0:
    # the net mass inside a window of half-width R is a far-field tail falling as 1 / R
    cfg = _config(n_blocks_x=1, n_blocks_y=1, fault_top_depth_m=15e3, fault_dip_deg=25.0)
    fractions = []
    for half_width in (300e3, 600e3):
        h = 3e3
        xs = np.arange(-half_width, half_width, h) + h / 2
        X, Y = np.meshgrid(xs, xs)
        field = dl.surface_field(cfg, np.column_stack([X.ravel(), Y.ravel()]), [0], [1.0])
        mass = (field[:, 3] + dl.FREE_AIR_GRADIENT * field[:, 2]) / (2 * math.pi * G) * h * h
        fractions.append(mass.sum() / np.abs(mass).sum())
    assert abs(fractions[1]) < 0.025
    assert fractions[1] / fractions[0] == pytest.approx(0.5, abs=0.05)


def test_thrust_lifts_and_pushes_the_surface_trenchwards():
    cfg = _config(n_blocks_x=1, n_blocks_y=1, fault_rake_deg=90.0, fault_dip_deg=20.0)
    above = dl.surface_field(cfg, np.array([[0.0, 0.0]]), [0], [1.0])[0]
    assert above[2] > 0          # uplift above the slip
    assert above[0] < 0          # the hanging wall moves up dip, towards -X (the trench)
    assert abs(above[1]) < 1e-12  # no along-strike motion on the axis of a pure thrust
    assert above[3] < 0          # the free-air drop of the uplift dominates the gravity change


def test_matrix_matches_the_direct_sum():
    cfg = _config(n_blocks_x=4, n_blocks_y=3, n_sensors_x=3, n_sensors_y=3)
    matrix, total = dl.gravity_matrix(cfg)
    points = dl.surface_points(cfg)
    slip = np.random.default_rng(1).uniform(0, 2, cfg.n_blocks)
    direct = dl.surface_field(cfg, points, np.arange(cfg.n_blocks), slip)[:, 3]
    np.testing.assert_allclose(slip @ matrix.astype(float), direct, rtol=1e-6)
    np.testing.assert_allclose(total, matrix.sum(axis=0, dtype=float))


def test_runner_gravity_is_that_of_the_slip_deficit():
    cfg = SimulationConfig(n_blocks_x=15, n_blocks_y=15, n_sensors_x=4, n_sensors_y=4, simulation_years=20.0,
                           spinup_years=0.0, sample_interval_years=0.1, gravity_model='dislocation')
    runner = Runner(cfg)
    runner.advance(runner.end_s)
    m = runner.model
    matrix, total = dl.gravity_matrix(cfg)
    expected = (m.u - (m.plate_m - runner.plate0)) @ matrix.astype(float)
    np.testing.assert_allclose(runner.gravity_at(), expected, rtol=1e-5, atol=1e-6 * np.abs(expected).max())
    # samples between events follow the back-slip trend: the gravity changes even without events
    sample = runner.gravity
    assert np.any(np.abs(np.diff(sample, axis=0)).max(axis=1) > 0)


def test_dislocation_needs_valid_geometry():
    with pytest.raises(ValueError, match='fault_top_depth_m'):
        SimulationConfig(gravity_model='dislocation', fault_top_depth_m=0.0).validate()
    with pytest.raises(ValueError, match='gravity_model'):
        SimulationConfig(gravity_model='okubo').validate()
