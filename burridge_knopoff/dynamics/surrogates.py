"""
Surrogate data for tests of nonlinearity and of coupling.

- iaaft: same amplitude distribution and (nearly) the same power spectrum as x,
  otherwise random (Schreiber & Schmitz, 1996): the null of a linear Gaussian
  process seen through a static nonlinearity
- time_shift: circular shift by a random offset (keeps all structure of x but
  breaks its timing relative to another series)
- interval_shuffle: event train with the same inter-event intervals in random
  order (keeps the interval distribution, destroys their sequence)
- surrogate_test: rank of a statistic among its surrogate values
"""
import numpy as np


def iaaft(x, rng=None, n_iter=100):
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    sorted_x = np.sort(x)
    amplitudes = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    for __ in range(n_iter):
        phases = np.angle(np.fft.rfft(s))
        s = np.fft.irfft(amplitudes * np.exp(1j * phases), n=len(x))
        ranks = np.argsort(np.argsort(s))
        s_new = sorted_x[ranks]
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


def time_shift(x, rng=None, min_fraction=0.1):
    rng = np.random.default_rng(rng)
    n = len(x)
    lo = int(min_fraction * n)
    return np.roll(np.asarray(x), int(rng.integers(lo, n - lo)) if n - lo > lo else n // 2)


def interval_shuffle(event_times, rng=None):
    rng = np.random.default_rng(rng)
    t = np.sort(np.asarray(event_times, dtype=float))
    if len(t) < 2:
        return t
    return t[0] + np.concatenate([[0.0], np.cumsum(rng.permutation(np.diff(t)))])


def surrogate_test(statistic, x, make_surrogate, n_surrogates=99, seed=0, larger=True):
    """(value, surrogate values, p-value): one-sided rank test of statistic(x) against
    statistic(make_surrogate(x, rng))."""
    rng = np.random.default_rng(seed)
    value = statistic(x)
    sur = np.array([statistic(make_surrogate(x, rng)) for __ in range(n_surrogates)])
    extreme = np.sum(sur >= value) if larger else np.sum(sur <= value)
    return value, sur, (extreme + 1) / (n_surrogates + 1)
