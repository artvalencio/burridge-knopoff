"""
Ordinal patterns: permutation entropy (Bandt & Pompe, 2002) and the
complexity-entropy plane (Rosso et al., 2007), which places deterministic
chaos (low entropy, high complexity) and correlated noise in different
regions.
"""
import math

import numpy as np


def ordinal_distribution(x, order=4, delay=1):
    """Probabilities of the order! ordinal patterns of x (ties broken by position)."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (order - 1) * delay
    if n <= 0:
        raise ValueError('series too short')
    windows = np.column_stack([x[k * delay:k * delay + n] for k in range(order)])
    ranks = np.argsort(windows, axis=1, kind='stable')
    codes = np.zeros(n, dtype=np.int64)
    for k in range(order):
        codes = codes * order + ranks[:, k]
    counts = np.unique(codes, return_counts=True)[1]
    p = np.zeros(math.factorial(order))
    p[:len(counts)] = counts / n
    return p


def _entropy(p):
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def permutation_entropy(x, order=4, delay=1):
    """Normalised permutation entropy H in [0, 1]."""
    p = ordinal_distribution(x, order, delay)
    return _entropy(p) / math.log(len(p))


def statistical_complexity(x, order=4, delay=1):
    """(H, C): normalised permutation entropy and the Jensen-Shannon statistical complexity."""
    p = ordinal_distribution(x, order, delay)
    n = len(p)
    u = np.full(n, 1.0 / n)
    h = _entropy(p) / math.log(n)
    js = _entropy(0.5 * (p + u)) - 0.5 * _entropy(p) - 0.5 * _entropy(u)
    js_max = -0.5 * ((n + 1) / n * math.log(n + 1) + math.log(n) - 2 * math.log(2 * n))
    return h, float(js / js_max * h)
