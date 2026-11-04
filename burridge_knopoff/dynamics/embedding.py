"""
Delay embedding and dimensions (Takens, 1981; Sauer et al., 1991).

- delay_vectors: x[n], x[n + tau], ..., x[n + (m - 1) tau]
- auto_mutual_information / first_minimum: the delay (Fraser & Swinney, 1986),
  by equal-population histograms, or by cami.mutual_info (method='cami')
- false_nearest_neighbours: fraction of false neighbours vs dimension (Kennel et
  al., 1992); cao: E1 and E2 (Cao, 1997)
- correlation_sum / correlation_dimension: Grassberger & Procaccia (1983) with a
  Theiler window; eckmann_ruelle_bound is the largest dimension N points can
  support (2 log10 N; Eckmann & Ruelle, 1992)
- return_map: (s[n], s[n + 1]) pairs, e.g. of inter-event times
"""
import math

import numpy as np


def delay_vectors(x, dim, tau):
    """(N - (dim - 1) tau, dim) delay vectors of a scalar series."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError('series too short for dim=%d, tau=%d' % (dim, tau))
    return np.column_stack([x[k * tau:k * tau + n] for k in range(dim)])


def _rank_bins(x, bins):
    ranks = np.argsort(np.argsort(x, kind='stable'), kind='stable')
    return (ranks * bins // len(x)).astype(np.int64)


def auto_mutual_information(x, max_lag, bins=16, method='histogram'):
    """I(x[n]; x[n + lag]) for lag = 0..max_lag, in bits."""
    x = np.asarray(x, dtype=float)
    if method == 'cami':
        import cami
        return np.array([cami.mutual_info(x[:len(x) - lag], x[lag:], n_symbols=bins, symbolic_length=1,
                                          symbolic_type='equal-points') if lag else
                         cami.entropy(x, n_symbols=bins, symbolic_length=1, symbolic_type='equal-points')
                         for lag in range(max_lag + 1)])
    s = _rank_bins(x, bins)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        a, b = s[:len(s) - lag], s[lag:]
        joint = np.bincount(a * bins + b, minlength=bins * bins).reshape(bins, bins) / len(a)
        pa, pb = joint.sum(axis=1), joint.sum(axis=0)
        nz = joint > 0
        out[lag] = np.sum(joint[nz] * np.log2(joint[nz] / np.outer(pa, pb)[nz]))
    return out


def first_minimum(values):
    """Index of the first local minimum (the last index if the series decreases throughout)."""
    v = np.asarray(values)
    for k in range(1, len(v) - 1):
        if v[k] < v[k - 1] and v[k] <= v[k + 1]:
            return k
    return len(v) - 1


def _tree(points):
    from scipy.spatial import cKDTree
    return cKDTree(points)


def _nearest(points, theiler=0):
    """Index and distance of the nearest neighbour of every point, excluding |i - j| <= theiler."""
    tree = _tree(points)
    k = min(len(points), 2 * theiler + 3)
    while True:
        dist, idx = tree.query(points, k=k)
        own = np.arange(len(points))[:, None]
        ok = np.abs(idx - own) > theiler
        ok &= np.isfinite(dist)
        if np.all(ok.any(axis=1)) or k >= len(points):
            first = np.argmax(ok, axis=1)
            rows = np.arange(len(points))
            return idx[rows, first], dist[rows, first]
        k = min(len(points), 2 * k)


def false_nearest_neighbours(x, tau, max_dim=8, rtol=15.0, atol=2.0, theiler=0):
    """Fraction of false nearest neighbours for dim = 1..max_dim (Kennel et al., 1992)."""
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    out = []
    for dim in range(1, max_dim + 1):
        X = delay_vectors(x, dim + 1, tau)
        base = X[:, :dim]
        j, d = _nearest(base, theiler)
        extra = np.abs(X[:, dim] - X[j, dim])
        d = np.maximum(d, 1e-300)
        false = (extra / d > rtol) | (np.sqrt(d ** 2 + extra ** 2) / sd > atol)
        out.append(float(np.mean(false)))
    return np.array(out)


def cao(x, tau, max_dim=8, theiler=0):
    """Cao's E1(d) and E2(d) for d = 1..max_dim (maximum norm). E1 saturates near 1 above the
    embedding dimension; E2 stays near 1 for all d for noise."""
    x = np.asarray(x, dtype=float)
    a, estar = [], []
    for dim in range(1, max_dim + 2):
        X = delay_vectors(x, dim + 1, tau)
        base = X[:, :dim]
        tree = _tree(base)
        k = min(len(base), 2 * theiler + 3)
        dist, idx = tree.query(base, k=k, p=np.inf)
        own = np.arange(len(base))[:, None]
        ok = (np.abs(idx - own) > theiler) & (dist > 0)
        first = np.argmax(ok, axis=1)
        rows = np.arange(len(base))
        j, d = idx[rows, first], dist[rows, first]
        d_next = np.maximum(np.abs(X[:, :dim + 1] - X[j, :dim + 1]).max(axis=1), d)
        a.append(np.mean(d_next / np.maximum(d, 1e-300)))
        estar.append(np.mean(np.abs(X[:, dim] - X[j, dim])))
    a, estar = np.array(a), np.array(estar)
    return a[1:] / a[:-1], estar[1:] / estar[:-1]


def correlation_sum(points, radii, theiler=0, max_points=4000, seed=0):
    """C(r): fraction of pairs (|i - j| > theiler) closer than r (maximum norm), from up to
    max_points reference points."""
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    ref = np.sort(rng.choice(len(points), size=min(max_points, len(points)), replace=False))
    radii = np.asarray(radii, dtype=float)
    counts = np.zeros(len(radii))
    pairs = 0
    index = np.arange(len(points))
    for i in ref:
        d = np.max(np.abs(points - points[i]), axis=1)
        dd = np.sort(d[np.abs(index - i) > theiler])
        counts += np.searchsorted(dd, radii, side='left')
        pairs += len(dd)
    return counts / max(pairs, 1)


def correlation_dimension(points, theiler=0, n_radii=40, fit=(1e-3, 1e-1), max_points=4000, seed=0):
    """Correlation dimension: slope of log C(r) vs log r over the radii where
    fit[0] <= C(r) <= fit[1]. Returns a dict: dimension (nan without 3 points in the band),
    radii, C, local slopes."""
    points = np.asarray(points, dtype=float)
    spread = np.max(points.max(axis=0) - points.min(axis=0))
    radii = spread * np.logspace(-3.5, 0, n_radii)
    C = correlation_sum(points, radii, theiler=theiler, max_points=max_points, seed=seed)
    ok = C > 0
    lr, lc = np.log(radii[ok]), np.log(C[ok])
    slopes = np.gradient(lc, lr) if len(lr) > 2 else np.zeros(0)
    band = (C[ok] >= fit[0]) & (C[ok] <= fit[1])
    dimension = float(np.polyfit(lr[band], lc[band], 1)[0]) if band.sum() >= 3 else math.nan
    return dict(dimension=dimension, radii=radii, C=C, slopes=slopes)


def eckmann_ruelle_bound(n_points):
    """Largest dimension a correlation-sum estimate from n points can support: 2 log10 N."""
    return 2.0 * math.log10(max(n_points, 1))


def return_map(series):
    s = np.asarray(series, dtype=float)
    return s[:-1], s[1:]
