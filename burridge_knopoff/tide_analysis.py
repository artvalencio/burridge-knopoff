"""
Tidal statistics of an event catalogue.

- schuster_p: Schuster (1897) test of the clustering of tidal phases; the
  p-value is exp(-R^2 / N), with R the length of the sum of the unit phase
  vectors of N events (small p: the events prefer a phase).
- rate_vs_stress: observed / expected number of events per bin of tidal
  stress, the expected number following the time spent in each bin
  (Cochran et al., 2004).
- tidal_statistics: both for a SimulationResult. The tidal series is
  computed from the configuration even when the run had no tides, which is
  the null control: a catalogue that never felt the tide must show no
  correlation with it.
"""
import dataclasses
import math

import numpy as np

from .config import SECONDS_PER_YEAR


def schuster_p(phases):
    """Schuster p-value of the phases (rad): exp(-R^2 / N); nan for no events."""
    phases = np.asarray(phases, dtype=float)
    n = len(phases)
    if n == 0:
        return math.nan
    r2 = np.sum(np.cos(phases)) ** 2 + np.sum(np.sin(phases)) ** 2
    return math.exp(-r2 / n)


def rate_vs_stress(event_times_s, series_times_s, stress, n_bins=10):
    """Events per bin of the tidal stress at their times, against the number expected if
    they ignored the tide (proportional to the time spent in each bin). Bins are quantiles
    of the stress over the series, so each covers the same time. Returns a dict of arrays:
    edges, observed, expected, ratio."""
    stress = np.asarray(stress, dtype=float)
    at_events = np.interp(event_times_s, series_times_s, stress)
    edges = np.quantile(stress, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    observed = np.histogram(at_events, bins=edges)[0].astype(float)
    time_share = np.histogram(stress, bins=edges)[0] / len(stress)
    expected = time_share * len(at_events)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(expected > 0, observed / expected, np.nan)
    return dict(edges=edges, observed=observed, expected=expected, ratio=ratio)


def tidal_series(config):
    """(times_s, Coulomb stress at the grid centre in Pa, unwrapped semidiurnal phase) of the
    configuration's tide, whether or not the run used it (tide_scale is ignored: the null
    control uses the unscaled tide)."""
    from .tides import HourlyTide
    cfg = dataclasses.replace(config, tides=True, tide_scale=1.0)
    hourly = HourlyTide(cfg, config.simulation_years * SECONDS_PER_YEAR)
    times, phase = hourly.semidiurnal_phase()
    return times, hourly.coulomb[:, 0] / config.block_area_m2, np.unwrap(phase)


def tidal_statistics(result, spinup_years=None, min_magnitude=None, n_bins=10, series=None):
    """Schuster p-value and rate vs stress of a result's events after the spin-up.

    Event phases come from the configuration's tidal series (so a run without tides gives
    the null control); `series` passes a precomputed tidal_series(config) to reuse it across
    runs of the same geometry. Returns a dict: n, schuster_p, rate (see rate_vs_stress)."""
    cfg = result.config
    spinup = cfg.spinup_years if spinup_years is None else spinup_years
    cat = result.catalogue
    keep = cat['start_years'] >= spinup
    if min_magnitude is not None:
        keep &= cat['magnitude'] >= min_magnitude
    t_events = cat['start_years'][keep] * SECONDS_PER_YEAR
    times, coulomb, phase = tidal_series(cfg) if series is None else series
    phases = np.interp(t_events, times, phase)
    inside = (times >= spinup * SECONDS_PER_YEAR)
    return dict(n=int(keep.sum()), schuster_p=schuster_p(phases),
                rate=rate_vs_stress(t_events, times[inside], coulomb[inside], n_bins=n_bins))
