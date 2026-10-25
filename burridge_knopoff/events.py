"""
Earthquake events: the catalogue entries produced by the simulation, seismic
moment and moment magnitude.

An event starts when a stuck block first exceeds its strength (the
hypocentre) and ends when every block has stopped again. Its size is
measured by the seismic moment M0 = shear modulus x block area x total slip,
and by the moment magnitude Mw = (2/3) (log10 M0 - 9.1) (Hanks & Kanamori,
1979; M0 in N m).
"""
import dataclasses
import math

import numpy as np

from . import grid

CATALOGUE_COLUMNS = ('start_years', 'duration_s', 'hypocentre', 'hypocentre_row', 'hypocentre_col',
                     'hypocentre_x_m', 'hypocentre_y_m', 'n_slipped', 'total_slip_m', 'max_slip_m',
                     'peak_velocity_m_s', 'moment_n_m', 'magnitude')


def seismic_moment(slip, config):
    """M0 = shear modulus x block area x sum of the slips (N m)."""
    return config.shear_modulus_pa * config.block_area_m2 * float(np.sum(slip))


def moment_magnitude(moment):
    """Mw = (2/3) (log10 M0 - 9.1), with M0 in N m."""
    return (2.0 / 3.0) * (math.log10(moment) - 9.1)


@dataclasses.dataclass(frozen=True)
class Event:
    """One earthquake.

    start_years: time of the first failure; duration_s: until all blocks stopped;
    hypocentre: index of the first failing block; n_slipped: number of blocks that
    slipped; total_slip_m, max_slip_m: sum and maximum of their slips;
    peak_velocity_m_s: fastest block velocity; moment_n_m, magnitude: M0 and Mw.
    """
    start_years: float
    duration_s: float
    hypocentre: int
    n_slipped: int
    total_slip_m: float
    max_slip_m: float
    peak_velocity_m_s: float
    moment_n_m: float
    magnitude: float

    def hypocentre_row_col(self, config):
        return grid.row_col(self.hypocentre, config)

    def hypocentre_xy(self, config):
        """(x, y) of the hypocentre block (m)."""
        row, col = self.hypocentre_row_col(config)
        return col * config.block_spacing_m, row * config.block_spacing_m


def catalogue(event_list, config):
    """The events as a numpy structured array with the CATALOGUE_COLUMNS."""
    dtype = [(name, np.int64 if name in ('hypocentre', 'hypocentre_row', 'hypocentre_col', 'n_slipped')
              else np.float64) for name in CATALOGUE_COLUMNS]
    table = np.zeros(len(event_list), dtype=dtype)
    for k, ev in enumerate(event_list):
        row, col = ev.hypocentre_row_col(config)
        x, y = ev.hypocentre_xy(config)
        table[k] = (ev.start_years, ev.duration_s, ev.hypocentre, row, col, x, y, ev.n_slipped,
                    ev.total_slip_m, ev.max_slip_m, ev.peak_velocity_m_s, ev.moment_n_m, ev.magnitude)
    return table
