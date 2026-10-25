import dataclasses
import os

import numpy as np
import pytest
from PIL import Image

from burridge_knopoff import visualization as vis
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run


@pytest.fixture(scope='module')
def result():
    return run(SimulationConfig(n_blocks_x=6, n_blocks_y=4, n_sensors_x=3, n_sensors_y=2,
                                simulation_years=20.0, sample_interval_years=0.1, seed=1))


@pytest.fixture(scope='module')
def quiet():
    """No events: every potential change is zero."""
    return run(SimulationConfig(n_blocks_x=3, n_blocks_y=3, n_sensors_x=2, n_sensors_y=2,
                                simulation_years=0.5, sample_interval_years=0.1))


@pytest.mark.parametrize('values', [np.zeros(5), np.array([1e-9, 2e-9]), np.array([-3.0, -1.0]), np.array([-1.0, 4.0])])
def test_colour_scale_is_symmetric_and_never_fails(values):
    norm = vis.symmetric_norm(values)
    assert norm.vmin == -norm.vmax and norm.vmax > 0
    assert norm.vmax >= np.abs(values).max()


def test_frames_are_subsampled_and_keep_both_ends():
    assert vis.frame_indices(10, 300).tolist() == list(range(10))
    idx = vis.frame_indices(10001, 300)
    assert len(idx) <= 300 and idx[0] == 0 and idx[-1] == 10000 and np.all(np.diff(idx) > 0)


@pytest.mark.filterwarnings('ignore:Animation was deleted')
def test_animation_title_follows_the_frames(result):
    fig, anim, update = vis.potential_animation(result, path=None, max_frames=5)
    frames = vis.frame_indices(len(result.times_years), 5)
    update(len(frames) - 1)
    assert '%.2f' % result.times_years[frames[-1]] in fig.axes[0].get_title()
    vis.plt.close(fig)


def test_animation_falls_back_to_gif(result, tmp_path, monkeypatch):
    monkeypatch.setattr(vis, '_ffmpeg_available', lambda: False)
    path = vis.potential_animation(result, path=str(tmp_path / 'anim.mp4'), max_frames=6)
    assert path.endswith('.gif') and os.path.exists(path)
    assert Image.open(path).n_frames == 6


def test_quiet_run_plots_do_not_fail(quiet, tmp_path):
    paths = vis.create_all(quiet, str(tmp_path), max_frames=3, writer='pillow')
    assert all(os.path.exists(p) for p in paths)
    assert not any('event_' in os.path.basename(p) for p in paths)


def test_event_comparison_marks_the_hypocentre(result):
    k = int(np.argmax([e.magnitude for e in result.events]))
    fig = vis.event_comparison(result, k, path=None)
    x_km, y_km = (c / 1000 for c in result.events[k].hypocentre_xy(result.config))
    markers = [line for ax in fig.axes for line in ax.get_lines() if line.get_marker() == '*']
    assert markers and all(np.allclose(m.get_data(), [[x_km], [y_km]]) for m in markers)
    vis.plt.close(fig)


def test_magnitude_frequency_is_cumulative(result):
    fig = vis.magnitude_frequency(result, path=None)
    m = np.array([e.magnitude for e in result.events])
    line = fig.axes[0].get_lines()[0]
    mags, counts = line.get_data()
    assert counts[0] == len(m) and np.all(np.diff(counts) <= 0)
    vis.plt.close(fig)


def test_create_all_writes_every_plot(result, tmp_path):
    paths = vis.create_all(result, str(tmp_path), max_frames=4, writer='pillow', n_event_plots=2)
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ['event_1_comparison.png', 'event_2_comparison.png', 'gravity_potential_animation.gif',
                     'magnitude_frequency.png']
