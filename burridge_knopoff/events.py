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
                     'peak_velocity_m_s', 'moment_n_m', 'magnitude',
                     'dg_max_abs_ugal', 'dg_rms_ugal', 'dg_max_sensor', 'dphi_max_abs',
                     'tide_shear_pa', 'tide_normal_pa', 'tide_coulomb_pa', 'tide_coulomb_rate_pa_h', 'tide_phase_rad')
INTEGER_COLUMNS = ('hypocentre', 'hypocentre_row', 'hypocentre_col', 'n_slipped', 'dg_max_sensor')


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

    Co-seismic gravity (the step the event causes at the sensors): dg_max_abs_ugal,
    dg_rms_ugal: largest and RMS |change of the downward gravity| over the sensors
    (microGal); dg_max_sensor: where the largest is; dphi_max_abs: largest |potential
    change| (J/kg).

    Tides at the hypocentre at the origin time (all 0 without tides): tide_shear_pa: the
    driving stress (shear on the fault plus the body force per unit area, along the slip);
    tide_normal_pa: normal stress (tension positive); tide_coulomb_pa: shear + mu_s normal;
    tide_coulomb_rate_pa_h: its rate; tide_phase_rad: semidiurnal phase of the Coulomb
    stress at the grid centre (0 at its maxima).
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
    dg_max_abs_ugal: float = math.nan
    dg_rms_ugal: float = math.nan
    dg_max_sensor: int = -1
    dphi_max_abs: float = math.nan
    tide_shear_pa: float = 0.0
    tide_normal_pa: float = 0.0
    tide_coulomb_pa: float = 0.0
    tide_coulomb_rate_pa_h: float = 0.0
    tide_phase_rad: float = 0.0

    def hypocentre_row_col(self, config):
        return grid.row_col(self.hypocentre, config)

    def hypocentre_xy(self, config):
        """(x, y) of the hypocentre block (m)."""
        row, col = self.hypocentre_row_col(config)
        return col * config.block_spacing_m, row * config.block_spacing_m


def catalogue(event_list, config):
    """The events as a numpy structured array with the CATALOGUE_COLUMNS."""
    dtype = [(name, np.int64 if name in INTEGER_COLUMNS else np.float64) for name in CATALOGUE_COLUMNS]
    table = np.zeros(len(event_list), dtype=dtype)
    for k, ev in enumerate(event_list):
        row, col = ev.hypocentre_row_col(config)
        x, y = ev.hypocentre_xy(config)
        table[k] = (ev.start_years, ev.duration_s, ev.hypocentre, row, col, x, y, ev.n_slipped,
                    ev.total_slip_m, ev.max_slip_m, ev.peak_velocity_m_s, ev.moment_n_m, ev.magnitude,
                    ev.dg_max_abs_ugal, ev.dg_rms_ugal, ev.dg_max_sensor, ev.dphi_max_abs,
                    ev.tide_shear_pa, ev.tide_normal_pa, ev.tide_coulomb_pa, ev.tide_coulomb_rate_pa_h,
                    ev.tide_phase_rad)
    return table
