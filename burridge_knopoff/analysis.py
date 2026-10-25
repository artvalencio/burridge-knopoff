"""
Magnitude-frequency statistics of an event catalogue.

The Gutenberg-Richter law N(>= M) ~ 10^(-b M) means that magnitudes above a
threshold Mc are exponentially distributed with rate beta = b ln 10. The
estimators here are maximum-likelihood, for continuous magnitudes (the
simulated Mw are not binned):

- b_value(m, mc): Aki (1965), b = log10(e) / (mean(m) - mc), with standard
  error b / sqrt(n).
- b_value(m, mc, upper): Page (1968) for magnitudes truncated to [mc, upper):
  solves mean(m - mc) = 1/beta - w / (exp(beta w) - 1), w = upper - mc. Aki's
  formula applied inside a finite window overestimates b (for b = 1 and a
  one-unit window it gives ~1.34), which is why local slopes need this one.
- magnitude_of_completeness: the most populated 0.1-wide bin (maximum
  curvature); below it the catalogue is not a power law (here: the smallest
  events are single blocks, whose size is set by the grid).
- local_b_values: b in sliding windows, to see where the power law holds and
  where it is cut off.
- gutenberg_richter(result): all of the above for the events after the
  spin-up (the first years are dominated by the synchronised initial state).
"""
import dataclasses
import math

import numpy as np

LOG10_E = math.log10(math.e)


@dataclasses.dataclass(frozen=True)
class BValue:
    """b-value estimate from n events with mc <= M (< upper, if upper is set)."""
    b: float
    sigma: float
    n: int
    mc: float
    upper: float = None


def _truncated_mean_excess(beta, w):
    # mean of x in [0, w) for a density ~ exp(-beta x), stable near beta = 0
    if abs(beta * w) < 1e-6:
        return w / 2 - beta * w * w / 12
    return 1 / beta - w / math.expm1(beta * w)


def _truncated_fisher(beta, w):
    # variance of x for the same density = Fisher information per event
    if abs(beta * w) < 1e-4:
        return w * w / 12
    e = math.exp(beta * w)
    return 1 / beta ** 2 - w * w * e / (e - 1) ** 2


def b_value(magnitudes, mc, upper=None):
    """Maximum-likelihood b-value of the magnitudes >= mc (and < upper)."""
    m = np.asarray(magnitudes, dtype=float)
    m = m[(m >= mc) & ((m < upper) if upper is not None else True)]
    n = len(m)
    if n < 2:
        raise ValueError('need at least 2 events above mc to estimate b, got %d' % n)
    excess = float(m.mean() - mc)
    if upper is None:
        b = LOG10_E / excess
        return BValue(b=b, sigma=b / math.sqrt(n), n=n, mc=mc)
    w = upper - mc
    # the mean excess decreases monotonically with beta: bisection
    lo, hi = -200.0 / w, 200.0 / w
    for __ in range(200):
        mid = 0.5 * (lo + hi)
        if _truncated_mean_excess(mid, w) > excess:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    sigma_beta = 1 / math.sqrt(n * _truncated_fisher(beta, w))
    return BValue(b=beta / math.log(10), sigma=sigma_beta / math.log(10), n=n, mc=mc, upper=upper)


def magnitude_of_completeness(magnitudes, bin_width=0.1):
    """Lower edge of the most populated magnitude bin (maximum-curvature method)."""
    m = np.asarray(magnitudes, dtype=float)
    start = math.floor(m.min() / bin_width) * bin_width
    edges = np.arange(start, m.max() + bin_width, bin_width)
    counts = np.histogram(m, edges)[0]
    return round(float(edges[np.argmax(counts)]), 10)


def local_b_values(magnitudes, width=1.0, step=0.5, min_events=50):
    """Truncated b-values in windows [a, a + width), a = start, start + step, ...;
    windows with fewer than min_events events are skipped."""
    m = np.asarray(magnitudes, dtype=float)
    out = []
    a = math.floor(m.min() / step) * step
    while a < m.max():
        inside = np.count_nonzero((m >= a) & (m < a + width))
        if inside >= min_events:
            out.append(b_value(m, round(a, 10), upper=round(a + width, 10)))
        a += step
    return out


def gutenberg_richter(result, spinup_years=None, min_events=50):
    """Statistics of the events starting after the spin-up (default: config.spinup_years).

    Returns a dict: n_events, spinup_years, mc, b (BValue above mc, or None if
    too few events) and local (list of BValue in one-magnitude windows).
    """
    spinup = result.config.spinup_years if spinup_years is None else spinup_years
    m = np.array([e.magnitude for e in result.events if e.start_years >= spinup])
    stats = {'n_events': len(m), 'spinup_years': spinup, 'mc': None, 'b': None, 'local': []}
    if len(m) < 2:
        return stats
    stats['mc'] = magnitude_of_completeness(m)
    try:
        stats['b'] = b_value(m, stats['mc'])
    except ValueError:
        pass
    stats['local'] = local_b_values(m, min_events=min_events)
    return stats
