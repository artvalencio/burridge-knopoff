import numpy as np
import pytest

from burridge_knopoff import events


@pytest.mark.parametrize('moment,magnitude', [(1e18, 5.9333333), (10 ** 9.1, 0.0), (3.98e22, 9.0)])
def test_moment_magnitude(moment, magnitude):
    assert events.moment_magnitude(moment) == pytest.approx(magnitude, abs=1e-3)


def test_moment_sums_slip_over_the_ruptured_area(small_config):
    slip = np.zeros(small_config.n_blocks)
    slip[[2, 3, 9]] = [1.0, 0.5, 0.25]
    expected = small_config.shear_modulus_pa * small_config.block_area_m2 * 1.75
    assert events.seismic_moment(slip, small_config) == pytest.approx(expected)


def test_catalogue_columns_and_hypocentre_coordinates(small_config):
    ev = events.Event(start_years=2.5, duration_s=3.0, hypocentre=9, n_slipped=3, total_slip_m=1.75,
                      max_slip_m=1.0, peak_velocity_m_s=0.8, moment_n_m=2.1e17, magnitude=5.5)
    row, col = ev.hypocentre_row_col(small_config)
    x, y = ev.hypocentre_xy(small_config)
    assert (row, col) == (1, 3)
    assert (x, y) == (3 * small_config.block_spacing_m, 1 * small_config.block_spacing_m)
    table = events.catalogue([ev, ev], small_config)
    assert table['magnitude'].tolist() == [5.5, 5.5]
    assert table['hypocentre_x_m'][0] == x and table['hypocentre_y_m'][0] == y
    assert set(events.CATALOGUE_COLUMNS) == set(table.dtype.names)


def test_empty_catalogue(small_config):
    assert len(events.catalogue([], small_config)) == 0
