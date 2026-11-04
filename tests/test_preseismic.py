"""Pre-seismic windows: deterministic replay, hourly frames, tide removal, animation."""
import math
import os

import numpy as np
import pytest
from PIL import Image

from burridge_knopoff import preseismic
from burridge_knopoff import visualization as vis
from burridge_knopoff.config import SECONDS_PER_YEAR, SimulationConfig
from burridge_knopoff.simulation import Runner, run

BASE = dict(n_blocks_x=8, n_blocks_y=6, n_sensors_x=3, n_sensors_y=3, simulation_years=40.0,
            sample_interval_years=0.01, seed=3, spinup_years=5.0)


@pytest.fixture(scope='module')
def plain():
    return run(SimulationConfig(**BASE))


@pytest.fixture(scope='module')
def tidal():
    return run(SimulationConfig(**BASE, tides=True))


def _target(result, min_years=2.0):
    # a large event with others before it in its 1-year window
    starts = np.array([e.start_years for e in result.events])
    mags = np.array([e.magnitude for e in result.events])
    ok = np.flatnonzero(starts > min_years)
    k = int(ok[np.argmax(mags[ok])])
    assert np.count_nonzero((starts < starts[k]) & (starts >= starts[k] - 1.0)) > 0
    return k


def test_replay_frames_cover_one_year_hour_by_hour(plain):
    k = _target(plain)
    w = preseismic.replay(plain, k)
    assert len(w.frame_times_s) == 8767                       # 365.25 days x 24 h, both ends included
    np.testing.assert_allclose(np.diff(w.frame_times_s), 3600.0)
    assert w.frame_times_s[-1] == pytest.approx(plain.events[k].start_years * SECONDS_PER_YEAR)
    assert np.all(w.gravity[0] == 0) and np.all(w.potential[0] == 0)
    assert w.tidal_gravity_abs is None and np.array_equal(w.residual(), w.gravity)


def test_last_frame_excludes_the_target_event(plain):
    k = _target(plain)
    w = preseismic.replay(plain, k)
    runner = Runner(plain.config, samples=False)
    for __ in range(k):
        runner.step(runner.end_s)
    np.testing.assert_array_equal(w.gravity_abs[-1], runner.delta_g)   # state after event k - 1
    assert [j for j, __ in w.events] == [j for j, e in enumerate(plain.events[:k])
                                         if e.start_years * SECONDS_PER_YEAR >= w.frame_times_s[0]]


def test_frames_agree_with_the_regular_samples(plain):
    k = _target(plain)
    w = preseismic.replay(plain, k)
    starts = np.array([e.start_years for e in plain.events]) * SECONDS_PER_YEAR
    sample_s = plain.times_years * SECONDS_PER_YEAR
    checked = 0
    for i in np.flatnonzero((sample_s >= w.frame_times_s[0]) & (sample_s < w.frame_times_s[-1])):
        j = int(np.searchsorted(w.frame_times_s, sample_s[i], side='right')) - 1
        lo, hi = w.frame_times_s[j], sample_s[i]
        if np.any((starts > lo) & (starts <= hi)):
            continue                                              # an event between frame and sample
        np.testing.assert_allclose(w.gravity_abs[j], plain.gravity[i], rtol=1e-12, atol=1e-25)
        checked += 1
    assert checked > 20


def test_frames_change_only_after_events(plain):
    k = _target(plain)
    w = preseismic.replay(plain, k)
    starts = np.array([e.start_years for e in plain.events]) * SECONDS_PER_YEAR
    changed = np.flatnonzero(np.any(np.diff(w.gravity_abs, axis=0) != 0, axis=1)) + 1
    assert len(changed) > 0
    for j in changed:
        assert np.any((starts > w.frame_times_s[j - 1]) & (starts <= w.frame_times_s[j]))


def test_several_windows_in_one_pass_and_early_events(plain):
    k = _target(plain)
    windows = preseismic.replay(plain, [k, 0, 1])
    assert [w.event_index for w in windows] == sorted({k, 0, 1})
    np.testing.assert_array_equal(windows[-1].gravity_abs, preseismic.replay(plain, k).gravity_abs)
    # a window longer than the time before the event starts at t = 0
    t_ev = plain.events[0].start_years * SECONDS_PER_YEAR
    w = preseismic.replay(plain, 0, window_years=plain.events[0].start_years + 5, frame_hours=24.0)
    assert w.frame_times_s[0] >= 0 and len(w.frame_times_s) == int(t_ev // 86400) + 1
    assert w.events == [] and np.all(w.gravity == 0)


def test_replay_with_tides(tidal):
    k = _target(tidal)
    w = preseismic.replay(tidal, k)
    assert w.tidal_gravity_abs is not None and w.tidal_coulomb_pa is not None
    tide_ugal = w.tidal_gravity * 1e8
    assert 100 < tide_ugal.max() - tide_ugal.min() < 400              # the tide dominates the raw record
    np.testing.assert_allclose(w.raw - w.residual('model'), w.tidal_gravity, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(w.residual('model'), w.gravity, rtol=1e-9, atol=1e-18)


def test_fitted_tide_removal_recovers_steps_and_constituents():
    hours = np.arange(0, 365.25 * 24 + 1)
    t = hours * 3600.0
    m2 = 60e-8 * np.cos(np.radians(28.9841042) * hours + 0.3)
    k1 = 40e-8 * np.cos(np.radians(15.0410686) * hours - 1.0)
    steps = 2e-8 * (hours > 2000) - 1e-8 * (hours > 6000)
    drift = 0.5e-8 * hours / hours[-1]
    raw = (m2 + k1 + steps + drift)[:, None]
    tide = preseismic.fitted_tide(t, raw)
    np.testing.assert_allclose(tide[:, 0], m2 + k1, atol=0.01 * 60e-8)
    residual = raw[:, 0] - tide[:, 0]
    residual -= residual[0]
    assert residual[3000] - residual[1000] == pytest.approx(2e-8 + 0.5e-8 * 2000 / hours[-1], rel=0.1)
    assert residual[7000] - residual[5000] == pytest.approx(-1e-8 + 0.5e-8 * 2000 / hours[-1], rel=0.1)


def test_largest_event_respects_the_spinup(plain):
    k = preseismic.largest_event(plain)
    assert plain.events[k].start_years >= plain.config.spinup_years
    with pytest.raises(ValueError):
        preseismic.largest_event(plain, spinup_years=1e6)


@pytest.mark.filterwarnings('ignore:Animation was deleted')
def test_animation_has_one_frame_per_record_and_a_fixed_scale(tidal, tmp_path):
    k = _target(tidal)
    w = preseismic.replay(tidal, k, window_years=0.01, frame_hours=6.0)     # 15 frames
    fig, anim, update = vis.preseismic_animation(w, path=None)
    mesh = fig.axes[0].collections[0]
    limits = (mesh.norm.vmin, mesh.norm.vmax)
    for j in range(len(w.frame_times_s)):
        update(j)
        assert (mesh.norm.vmin, mesh.norm.vmax) == limits
    assert 'tides removed (model)' in fig.axes[1].get_ylabel()            # colour bar: residuals
    vis.plt.close(fig)
    path = vis.preseismic_animation(w, str(tmp_path / 'pre.gif'), writer='pillow', tide_removal='fit')
    assert Image.open(path).n_frames == len(w.frame_times_s)


@pytest.mark.filterwarnings('ignore:Animation was deleted')
def test_animation_without_tides_shows_gravity_changes(plain):
    w = preseismic.replay(plain, _target(plain), window_years=0.005, frame_hours=6.0)
    fig, anim, update = vis.preseismic_animation(w, path=None)
    assert 'Gravity change since t0' in fig.axes[1].get_ylabel()
    vis.plt.close(fig)


def test_animation_requires_ffmpeg_for_video(plain, tmp_path, monkeypatch):
    w = preseismic.replay(plain, _target(plain), window_years=0.002, frame_hours=6.0)
    monkeypatch.setattr(vis, '_ffmpeg_available', lambda: False)
    with pytest.raises(RuntimeError, match='FFmpeg'):
        vis.preseismic_animation(w, str(tmp_path / 'pre.mp4'))
