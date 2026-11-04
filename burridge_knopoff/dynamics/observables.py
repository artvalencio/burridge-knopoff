"""
Series derived from a SimulationResult for the dynamics analyses (plan, section 6.2).
"""
import numpy as np

from ..config import SECONDS_PER_YEAR


def _after_spinup(result, spinup_years):
    spinup = result.config.spinup_years if spinup_years is None else spinup_years
    starts = np.array([e.start_years for e in result.events])
    return spinup, starts, starts >= spinup


def regional_moment_series(result, bin_days=1.0, spinup_years=None):
    """(bin start times in s, (n_bins, n_regions) moment released per bin and region, N m)."""
    spinup, starts, keep = _after_spinup(result, spinup_years)
    width = bin_days * 86400.0
    t0, t1 = spinup * SECONDS_PER_YEAR, result.config.simulation_years * SECONDS_PER_YEAR
    edges = np.arange(t0, t1 + width, width)
    idx = np.clip(np.searchsorted(edges, starts[keep] * SECONDS_PER_YEAR, side='right') - 1, 0, len(edges) - 2)
    series = np.zeros((len(edges) - 1, result.regional_moment.shape[1]))
    np.add.at(series, idx, result.regional_moment[keep])
    return edges[:-1], series


def event_train(result, min_magnitude=None, spinup_years=None):
    """Start times (s) of the events after the spin-up, optionally above a magnitude."""
    spinup, starts, keep = _after_spinup(result, spinup_years)
    if min_magnitude is not None:
        keep &= np.array([e.magnitude >= min_magnitude for e in result.events])
    return starts[keep] * SECONDS_PER_YEAR


def regional_event_trains(result, min_moment_n_m, spinup_years=None):
    """For every region, the times (s) of the events that released at least min_moment_n_m
    in it (a regional "large event")."""
    spinup, starts, keep = _after_spinup(result, spinup_years)
    rm = result.regional_moment
    return [starts[keep & (rm[:, r] >= min_moment_n_m)] * SECONDS_PER_YEAR for r in range(rm.shape[1])]


def inter_event_times(times):
    return np.diff(np.sort(np.asarray(times, dtype=float)))


def gravity_modes(result, n_modes=3, spinup_years=None):
    """Leading empirical orthogonal functions of the gravity field after the spin-up:
    (times in years, (n_samples, n_modes) amplitudes, (n_modes, n_sensors) patterns,
    explained variance fractions)."""
    spinup = result.config.spinup_years if spinup_years is None else spinup_years
    keep = result.times_years >= spinup
    g = np.asarray(result.gravity)[keep]
    g = g - g.mean(axis=0)
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    var = s ** 2 / max(np.sum(s ** 2), 1e-300)
    return result.times_years[keep], u[:, :n_modes] * s[:n_modes], vt[:n_modes], var[:n_modes]


def region_centres(config):
    """(n_regions, 2) centres of the regions (m), in the block coordinates."""
    L = config.block_spacing_m
    w = config.n_blocks_x * L / config.n_regions_x
    h = config.n_blocks_y * L / config.n_regions_y
    rows, cols = np.divmod(np.arange(config.n_regions_x * config.n_regions_y), config.n_regions_x)
    return np.column_stack([(cols + 0.5) * w - L / 2, (rows + 0.5) * h - L / 2])
