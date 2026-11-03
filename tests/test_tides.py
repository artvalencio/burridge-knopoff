import math

import numpy as np
import pytest

from burridge_knopoff import tides
from burridge_knopoff.config import SimulationConfig

CFG = SimulationConfig(tides=True)
SITE = ([38.3], [143.0])
HOUR = 3600.0


def julian_centuries(jd_tt):
    return (jd_tt - tides.J2000) / 36525.0


# ---- ephemerides ---------------------------------------------------------------------
def test_moon_matches_meeus_example_47a():
    # 1992 April 12, 0h TD: lambda = 133.162655 deg, beta = -3.229126 deg, distance 368409.7 km
    lon, lat, dist = tides.moon_ecliptic(julian_centuries(2448724.5))
    assert math.degrees(lon) == pytest.approx(133.162655, abs=0.05)
    assert math.degrees(lat) == pytest.approx(-3.229126, abs=0.02)
    assert dist / 1e3 == pytest.approx(368409.7, abs=300)


def test_sun_matches_meeus_example_25a():
    # 1992 October 13, 0h TD: true longitude 199.90988 deg, R = 0.99766 AU
    lon, lat, dist = tides.sun_ecliptic(julian_centuries(2448908.5))
    assert math.degrees(lon) == pytest.approx(199.90988, abs=0.05)
    assert dist / tides.AU_M == pytest.approx(0.99766, abs=2e-4)


def test_sidereal_time_and_epoch():
    assert tides.epoch_julian_date('2000-01-01T12:00:00') == pytest.approx(tides.J2000)
    assert tides.epoch_julian_date('1987-04-10T00:00:00') == pytest.approx(2446895.5)
    # Meeus example 12.a: 1987 April 10, 0h UT: GMST = 13h10m46.3668s
    assert math.degrees(tides.gmst_rad(2446895.5)) / 15 == pytest.approx(13 + 10 / 60 + 46.3668 / 3600, abs=1e-6)


def test_mean_moon_sun_tidal_ratio():
    # GM / d^3 averaged over a year: ~2.2 (Moon / Sun), the classic ratio of the tide-raising forces
    t = np.arange(0, 365.25 * 86400, 6 * HOUR)
    pos = tides.body_positions(t, tides.J2000)
    k = {name: np.mean(gm / np.linalg.norm(p, axis=1) ** 3) for name, (gm, p) in pos.items()}
    assert k['moon'] / k['sun'] == pytest.approx(2.2, rel=0.03)


# ---- potential, strain, stress ------------------------------------------------------
def _fields_at_offsets(offsets_enu, t, config=CFG, lat0=38.3, lon0=143.0):
    # fields at points displaced by (east, north, up) metres from the site
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    p0 = np.array([math.cos(lat0r) * math.cos(lon0r), math.cos(lat0r) * math.sin(lon0r), math.sin(lat0r)])
    e = np.array([-math.sin(lon0r), math.cos(lon0r), 0.0])
    n = np.array([-math.sin(lat0r) * math.cos(lon0r), -math.sin(lat0r) * math.sin(lon0r), math.cos(lat0r)])
    pts = tides.EARTH_RADIUS_M * p0 + offsets_enu @ np.array([e, n, p0])
    r = np.linalg.norm(pts, axis=1)
    lat = np.degrees(np.arcsin(pts[:, 2] / r))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return tides.TidePoints(lat, lon, config, depth_m=tides.EARTH_RADIUS_M - r).fields(t)


def test_potential_is_harmonic_and_acceleration_is_its_gradient():
    h = 2000.0
    offsets = np.array([[0, 0, 0], [h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]], dtype=float)
    f = _fields_at_offsets(offsets, np.array([0.0, 5 * HOUR, 11 * HOUR]))
    W = f['potential']
    laplacian = (W[:, 1:].sum(axis=1) - 6 * W[:, 0]) / h ** 2
    grad = np.stack([(W[:, 1] - W[:, 2]), (W[:, 3] - W[:, 4]), (W[:, 5] - W[:, 6])], axis=1) / (2 * h)
    assert np.all(np.abs(laplacian) < 1e-3 * np.abs(grad).max() / h)
    np.testing.assert_allclose(f['acceleration'][:, 0], grad, rtol=1e-4, atol=1e-4 * np.abs(grad).max())


def test_areal_strain_follows_the_love_numbers():
    sun = SimulationConfig(tides=True, tide_bodies='sun')   # degree 2 only
    f = tides.TidePoints(*SITE, sun).fields(np.arange(0, 2 * 86400, HOUR))
    expected = (2 * sun.love_h2 - 6 * sun.love_l2) * f['potential'] / (tides.EARTH_GRAVITY * tides.EARTH_RADIUS_M)
    np.testing.assert_allclose(f['strain_ee'] + f['strain_nn'], expected, rtol=1e-12, atol=1e-25)


def test_amplitudes_are_realistic():
    f = tides.TidePoints(*SITE, CFG).fields(np.arange(0, 30 * 86400, HOUR))
    strain = np.abs(np.stack([f['strain_ee'], f['strain_nn']])).max()
    assert 1e-8 < strain < 1e-7
    gravity_ugal = f['gravity'] * 1e8
    assert 150 < gravity_ugal.max() - gravity_ugal.min() < 400
    stress = np.abs(np.stack([f['sigma_ee'], f['sigma_nn']])).max()
    assert 1e3 < stress < 1e4                                     # kPa range
    assert 1e2 < np.abs(f['shear']).max() < stress                # reduced on a 10 degree thrust


def test_main_constituents_over_a_nodal_cycle():
    # least-squares harmonic fit over 18.61 years: Cartwright & Edden (1973) amplitude ratios
    t = np.arange(0, 18.6129 * 365.25 * 86400, HOUR)
    W = tides.TidePoints(*SITE, CFG).fields(t)['potential'][:, 0]
    speeds = dict(M2=28.9841042, S2=30.0, N2=28.4397295, K2=30.0821373, K1=15.0410686, O1=13.9430356,
                  P1=14.9589314, Q1=13.3986609)
    hours = t / HOUR
    cols = [np.ones_like(hours)]
    for s in speeds.values():
        cols += [np.cos(np.radians(s) * hours), np.sin(np.radians(s) * hours)]
    coef = np.linalg.lstsq(np.column_stack(cols), W, rcond=None)[0]
    amp = {k: math.hypot(coef[1 + 2 * i], coef[2 + 2 * i]) for i, k in enumerate(speeds)}
    assert amp['M2'] / amp['S2'] == pytest.approx(0.63192 / 0.29400, rel=0.03)
    assert amp['K1'] / amp['O1'] == pytest.approx(0.36878 / 0.26221, rel=0.03)
    assert amp['N2'] / amp['M2'] == pytest.approx(0.12099 / 0.63192, rel=0.03)


# ---- fault resolution ---------------------------------------------------------------
def test_compression_across_strike_drives_a_thrust():
    normal, slip = tides.fault_vectors(0.0, 30.0, 90.0)          # strike north, dips east, pure thrust
    sigma = np.diag([-1.0, 0.0, 0.0])                            # east-west compression
    assert slip @ sigma @ normal > 0                             # promotes thrust slip
    assert normal @ sigma @ normal < 0                           # and clamps the fault
    assert abs(normal @ slip) < 1e-12 and abs(np.linalg.norm(normal) - 1) < 1e-12


def test_switches_scale_and_superposition():
    t = np.arange(0, 3 * 86400, HOUR)
    both = tides.TideModel(CFG).coefficients(t)
    moon = tides.TideModel(SimulationConfig(tides=True, tide_bodies='moon')).coefficients(t)
    sun = tides.TideModel(SimulationConfig(tides=True, tide_bodies='sun')).coefficients(t)
    for key in ('drive', 'normal', 'gravity'):
        np.testing.assert_allclose(both[key], moon[key] + sun[key], rtol=1e-9, atol=1e-12 * np.abs(both[key]).max())
    zero = tides.TideModel(SimulationConfig(tides=True, tide_scale=0.0)).coefficients(t)
    assert all(np.all(v == 0) for v in zero.values())
    off = tides.TideModel(SimulationConfig(tides=True, tide_stress=False, tide_body_force=False)).coefficients(t)
    assert np.all(off['drive'] == 0) and np.all(off['normal'] == 0) and np.any(off['gravity'] != 0)
    double = tides.TideModel(SimulationConfig(tides=True, tide_scale=2.0)).coefficients(t)
    np.testing.assert_allclose(double['drive'], 2 * both['drive'])


def test_body_force_is_small_next_to_the_stress():
    t = np.arange(0, 15 * 86400, HOUR)
    body = tides.TideModel(SimulationConfig(tides=True, tide_stress=False)).coefficients(t)['drive'][:, 0]
    stress = tides.TideModel(SimulationConfig(tides=True, tide_body_force=False)).coefficients(t)['drive'][:, 0]
    assert 0 < np.abs(body).max() < 0.5 * np.abs(stress).max()


def test_linear_fit_represents_the_area():
    model = tides.TideModel(CFG)
    t = np.arange(0, 2 * 86400, 3 * HOUR)
    coef = model.coefficients(t)
    rng = np.random.default_rng(0)
    x0, y0 = tides.grid_centre(CFG)
    x, y = rng.uniform(-x0, x0, 30), rng.uniform(-y0, y0, 30)
    lat, lon = tides.local_to_geographic(x, y, CFG)
    f = tides.TidePoints(lat, lon, CFG).fields(t)
    exact = CFG.block_area_m2 * f['shear'] + CFG.block_mass_kg * f['accel_motion']
    fitted = tides.evaluate(coef['drive'], x, y)
    assert np.abs(fitted - exact).max() < 0.02 * np.abs(exact).max()


def test_hourly_interpolation_and_semidiurnal_phase():
    cfg = SimulationConfig(tides=True, simulation_years=0.2)
    hourly = tides.HourlyTide(cfg, end_s=0.1 * 365.25 * 86400)
    half_hours = np.arange(10, 400) * HOUR + 0.5 * HOUR
    exact = hourly.model.coefficients(half_hours)['drive']
    interp = hourly.at(half_hours)[0]
    assert np.abs(interp - exact).max() < 0.01 * np.abs(exact).max()
    times, phase = hourly.semidiurnal_phase()
    slope = np.polyfit(times[200:-200], np.unwrap(phase[200:-200]), 1)[0]
    assert 2 * math.pi / slope / HOUR == pytest.approx(12.3, rel=0.03)   # M2/S2 mix: ~12.2-12.4 h
