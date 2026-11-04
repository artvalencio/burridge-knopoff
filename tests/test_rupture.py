"""Displacement during an event: rupture replay, surface fields and maps."""
import numpy as np
import pytest

from burridge_knopoff import dislocation, rupture, visualization
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run


@pytest.fixture(scope='module')
def result():
    cfg = SimulationConfig(n_blocks_x=30, n_blocks_y=30, n_sensors_x=5, n_sensors_y=5, simulation_years=25.0,
                           spinup_years=5.0)
    return run(cfg)


@pytest.fixture(scope='module')
def rup(result):
    cat = result.catalogue
    late = np.flatnonzero(cat['start_years'] > 5.0)
    k = int(late[np.argmax(cat['magnitude'][late])])
    return rupture.replay(result, k, n_frames=20)


def test_replay_reproduces_the_event(result, rup):
    event = result.events[rup.event_index]
    assert rup.event == event
    assert len(rup.slipped) == event.n_slipped
    assert rup.final_slip.sum() == pytest.approx(event.total_slip_m, rel=1e-5)
    assert rup.frame_times_s[0] == 0.0
    assert rup.frame_times_s[-1] == pytest.approx(event.duration_s)
    assert 10 <= len(rup.frame_times_s) <= 22
    # blocks never slip backwards: the slip grows from frame to frame, from zero
    assert np.all(rup.slip[0] == 0)
    assert np.all(np.diff(rup.slip, axis=0) >= -1e-6)


def test_surface_field_is_that_of_the_slip(rup):
    points, _, _ = rupture.map_points(rup.config, n=6)
    frames = rupture.surface_frames(rup, points, factor=1)
    direct = dislocation.surface_field(rup.config, points, rup.slipped, rup.final_slip)
    np.testing.assert_allclose(frames[-1], direct, rtol=1e-5, atol=1e-9 * np.abs(direct).max())
    np.testing.assert_allclose(frames[0], 0.0, atol=1e-15)
    # coarse patches agree with the block sum away from the fault edges
    coarse = rupture.surface_frames(rup, points, factor=2)[-1]
    assert np.abs(coarse - direct).max() < 0.1 * np.abs(direct).max()


def test_maps_are_written(rup, tmp_path):
    still = visualization.displacement_map(rup, str(tmp_path / 'map.png'), n_points=10)
    movie = visualization.rupture_animation(rup, str(tmp_path / 'map.gif'), n_points=8, writer='pillow')
    assert (tmp_path / 'map.png').stat().st_size > 0 and still.endswith('.png')
    assert (tmp_path / 'map.gif').stat().st_size > 0 and movie.endswith('.gif')
