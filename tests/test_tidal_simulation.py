"""The 2-D model with tidal forcing: hourly stepping, frozen tide during events, gravity."""
import dataclasses
import math

import numpy as np
import pytest

from burridge_knopoff import tide_analysis, tides
from burridge_knopoff.config import SECONDS_PER_YEAR, SimulationConfig
from burridge_knopoff.gravity import SensorArray
from burridge_knopoff.model import Model
from burridge_knopoff.simulation import Runner, run

HOUR = 3600.0
SMALL = dict(n_blocks_x=6, n_blocks_y=4, n_sensors_x=3, n_sensors_y=2, simulation_years=60.0,
             sample_interval_years=0.5, seed=7)


class SyntheticTide:
    """A tide table with a uniform Coulomb force A sin(omega t) (test hook)."""

    def __init__(self, amplitude_n, period_s, end_s, step_s=HOUR):
        self.step_s = step_s
        self.times = np.arange(int(end_s / step_s) + 4) * step_s
        self.coulomb = np.zeros((len(self.times), 3))
        self.coulomb[:, 0] = amplitude_n * np.sin(2 * math.pi * self.times / period_s)

    def max_spatial_variation(self, x_extent, y_extent):
        return 0.0


def test_hourly_stepping_without_tide_reproduces_the_exact_jumps():
    exact = run(SimulationConfig(**SMALL))
    stepped = run(SimulationConfig(**SMALL, tides=True, tide_scale=0.0))
    assert len(stepped.events) == len(exact.events) > 20
    for a, b in zip(exact.events, stepped.events):
        assert (a.hypocentre, a.n_slipped) == (b.hypocentre, b.n_slipped)
        assert abs(a.start_years - b.start_years) * SECONDS_PER_YEAR < 1e-3
        assert a.magnitude == pytest.approx(b.magnitude, rel=1e-9)
    np.testing.assert_allclose(stepped.potentials, exact.potentials, rtol=1e-9, atol=1e-18)


def _first_root(f, start, end, grid_s=10.0):
    # first t in [start, end] with f(t) > 0: fine scan, then bisection
    t = np.arange(start, end, grid_s)
    first = t[np.argmax(f(t) > 0)]
    lo, hi = first - grid_s, first
    for __ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return hi


def _catmull_rom(table, t):
    # the interpolation of tides.HourlyTide, for a 1-column table
    tau = np.asarray(t, dtype=float) / table.step_s
    k = np.floor(tau).astype(int)
    u = tau - k
    p = table.coulomb[:, 0]
    p0, p1, p2, p3 = p[np.maximum(k - 1, 0)], p[k], p[k + 1], p[k + 2]
    return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3)


def _single_block():
    model = Model(SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1, heterogeneity=0.0,
                                   redraw_thresholds=False))
    return model, model.kp * model.vp, model.static_force[0] - model.current_force[0]


def test_single_block_failure_is_the_exact_first_crossing_of_the_interpolated_tide():
    model, rate, gap = _single_block()
    amplitude, period = 300 * rate * HOUR, 12.42 * HOUR      # tidal swing = 300 h of tectonic loading
    end = gap / rate + 10 * period
    tide = SyntheticTide(amplitude, period, end)
    wait, idx = model.time_to_failure_tidal(tide, end)
    earliest = (gap - 1.01 * amplitude) / rate                # no crossing is possible before this
    expected = _first_root(lambda t: rate * t + _catmull_rom(tide, t) - gap, earliest, end, grid_s=1.0)
    assert idx == 0
    assert wait == pytest.approx(expected, abs=1.0)
    assert wait < gap / rate - 50 * HOUR                       # the tide advances the failure


def test_shorter_tide_steps_converge_to_the_exact_sinusoid():
    model, rate, gap = _single_block()
    amplitude, period = 300 * rate * HOUR, 12.42 * HOUR
    end = gap / rate + 10 * period
    wait, __ = model.time_to_failure_tidal(SyntheticTide(amplitude, period, end, step_s=600.0), end)
    earliest = (gap - amplitude) / rate
    exact = _first_root(lambda t: rate * t + amplitude * np.sin(2 * math.pi * np.asarray(t) / period) - gap,
                        earliest, end)
    assert wait == pytest.approx(exact, abs=60.0)


def test_energy_balance_with_a_frozen_tide():
    cfg = SimulationConfig(n_blocks_x=8, n_blocks_y=8, n_sensors_x=2, n_sensors_y=2, seed=3)
    model = Model(cfg)
    rng = np.random.default_rng(0)
    for __ in range(10):
        dt, idx = model.time_to_failure()
        model.load(dt)
        model.tide_drive = rng.normal(0, 1e-4, cfg.n_blocks) * cfg.static_force_n
        model.tide_normal = rng.normal(0, 1e-4, cfg.n_blocks) * cfg.static_force_n
        before = model.elastic_energy()
        result = model.slip(idx)
        after = model.elastic_energy()
        balance = before + result.plate_work_j + result.tide_work_j - after - result.dissipated_j
        assert abs(balance) < 1e-3 * result.dissipated_j
        assert result.tide_work_j != 0.0


def test_no_block_is_left_above_its_strength_after_an_event_with_tides():
    runner = Runner(SimulationConfig(**SMALL, tides=True, tide_scale=200.0))
    m = runner.model
    for __ in range(60):
        if runner.step(runner.end_s) is None:
            break
        effective = m.current_force + m.tide_drive + m.mus * m.tide_normal - m.static_force
        assert effective.max() <= 1e-9 * m.static_force.max()


def test_strong_tides_phase_lock_failures_to_rising_tidal_stress():
    res = run(SimulationConfig(**SMALL, tides=True, tide_scale=50.0))
    cat = res.catalogue
    first_in_sequence = np.r_[True, np.diff(cat['start_years']) * SECONDS_PER_YEAR > 60.0]
    rates = cat['tide_coulomb_rate_pa_h'][first_in_sequence]
    assert len(rates) > 10 and np.mean(rates > 0) > 0.9
    assert np.all(cat['tide_shear_pa'] != 0)
    p_value = tide_analysis.schuster_p(cat['tide_phase_rad'][first_in_sequence])
    assert p_value < 1e-3


def test_tide_columns_and_series():
    res = run(SimulationConfig(**SMALL, tides=True))
    assert res.tide_coulomb_pa is not None and len(res.tide_times_s) == len(res.tide_coulomb_pa)
    assert np.abs(res.tide_coulomb_pa).max() < 5e3                     # kPa range at scale 1
    cat = res.catalogue
    assert np.all(np.abs(cat['tide_phase_rad']) <= math.pi)
    no_tides = run(SimulationConfig(**SMALL))
    assert no_tides.tide_coulomb_pa is None and np.all(no_tides.catalogue['tide_coulomb_pa'] == 0)


# ---- gravity --------------------------------------------------------------------------
def test_gravity_change_matches_a_direct_sum():
    cfg = SimulationConfig(n_blocks_x=5, n_blocks_y=4, n_sensors_x=3, n_sensors_y=3)
    sensors = SensorArray(cfg)
    u = np.random.default_rng(1).random(cfg.n_blocks) * 3
    G, m = 6.67430e-11, cfg.block_mass_kg

    def downward(disp):
        pos = sensors.block_positions + np.column_stack([disp, np.zeros_like(disp), np.zeros_like(disp)])
        d = sensors.positions[:, None, :] - pos[None, :, :]
        return G * m * (d[..., 2] / np.linalg.norm(d, axis=2) ** 3).sum(axis=1)

    expected = downward(u) - downward(np.zeros(cfg.n_blocks))
    np.testing.assert_allclose(sensors.gravity_change(u), expected, rtol=1e-6)


def test_gravity_change_is_cancellation_free():
    import mpmath
    mpmath.mp.dps = 50
    cfg = SimulationConfig(n_blocks_x=1, n_blocks_y=1, n_sensors_x=1, n_sensors_y=1, sensor_height_m=5000.0)
    sensors = SensorArray(cfg)
    s, b = sensors.positions[0], sensors.block_positions[0]
    for du in (1e-6, 1e-3, 1.0):
        def g(u):
            dx, dy, dz = (mpmath.mpf(s[0]) - b[0] - u), mpmath.mpf(s[1]) - b[1], mpmath.mpf(s[2]) - b[2]
            return dz / (dx * dx + dy * dy + dz * dz) ** mpmath.mpf(1.5)
        exact = float(6.67430e-11 * cfg.block_mass_kg * (g(mpmath.mpf(du)) - g(0)))
        got = sensors.gravity_change(np.array([du]))[0]
        assert got == pytest.approx(exact, rel=1e-9)


def test_runs_record_gravity_and_coseismic_steps():
    res = run(SimulationConfig(**SMALL, gravity_map_min_magnitude=5.5))
    assert res.gravity.shape == res.potentials.shape
    cat = res.catalogue
    assert np.all(cat['dg_max_abs_ugal'] >= cat['dg_rms_ugal'])
    big = np.flatnonzero(cat['magnitude'] >= 5.5)
    np.testing.assert_array_equal(res.gravity_map_events, big)
    for k, index in enumerate(res.gravity_map_events):
        assert np.abs(res.gravity_map_gravity[k]).max() * 1e8 == pytest.approx(cat['dg_max_abs_ugal'][index])
    # the gravity series is the sum of the co-seismic steps (steps of all events, not only the maps)
    total = np.abs(res.gravity[-1]).max()
    assert total > 0


def test_tide_analysis_tools():
    rng = np.random.default_rng(0)
    assert tide_analysis.schuster_p(rng.uniform(-math.pi, math.pi, 2000)) > 0.01
    assert tide_analysis.schuster_p(rng.normal(0.0, 0.3, 200)) < 1e-10
    t = np.arange(0, 100 * 86400, 3600.0)
    stress = np.sin(2 * math.pi * t / (12.42 * 3600))
    events = t[stress > 0.9][::5] + 1.0
    table = tide_analysis.rate_vs_stress(events, t, stress, n_bins=4)
    assert table['ratio'][-1] > 2 and table['ratio'][0] == 0
