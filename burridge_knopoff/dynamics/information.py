"""
Information flow with the CaMI toolbox (cami >= 1.5; plan, section 6.7).

Series are encoded with explicit divisions (divisions()): equal-population
quantiles for ordinary series and, for sparse event series (at least half of
the bins empty: moments or counts per bin), one symbol for "no event" and
quantiles of the non-empty bins for the others. Equal-population partitions
alone would collapse on the repeated zeros and see no information at all.
Moments are taken as log1p; tau is fixed (1 bin) for reproducibility.

- te_network: transfer entropy between every ordered pair of series, with
  surrogate significance (circular shifts, which keep each series' own
  structure): the causal network of fault regions
- distance_decay: mean transfer entropy vs the distance between regions
- pair_flow: TE and CaMI both ways, and the directionality index, for one pair
  (e.g. gravity at a sensor and the moment of the region below it, or the
  tidal Coulomb stress and the event rate)
- pooled_windows: many short windows of the same phenomenon (e.g. the year
  before each large event) pooled by cami.multithread_causality
"""
import numpy as np

DEFAULTS = dict(n_symbols=3, symbolic_length=2, symbolic_type='equal-points', tau=1)


def _cami():
    import cami
    if not hasattr(cami, 'surrogate_test'):
        raise ImportError('cami >= 1.5 is needed (pip install -e ../cami-python)')
    return cami


def prepare(series, log=True):
    """log1p of non-negative series (moments, counts), so that a few huge events do not use up
    the partition; other series are returned unchanged."""
    s = np.asarray(series, dtype=float)
    return np.log1p(s) if log and np.all(s >= 0) else s


def divisions(x, n_symbols):
    """Symbol boundaries for x: n_symbols - 1 equal-population quantiles or, when at least
    half of x is zero (sparse event series), a boundary just above zero followed by quantiles
    of the positive values."""
    x = np.asarray(x, dtype=float)
    positive = x[x > 0]
    if np.mean(x == 0) >= 0.5 and len(positive):
        cuts = [0.5 * positive.min()]
        if n_symbols > 2:
            cuts += list(np.quantile(positive, np.arange(1, n_symbols - 1) / (n_symbols - 1)))
        return np.unique(cuts)
    return np.unique(np.quantile(x, np.arange(1, n_symbols) / n_symbols))


def _options(x, y, kwargs):
    opts = {**DEFAULTS, **kwargs}
    n = opts.pop('n_symbols')
    opts.pop('symbolic_type', None)
    xd, yd = divisions(x, n), divisions(y, n)
    if len(xd) != len(yd):           # cami needs the same alphabet for both
        k = min(len(xd), len(yd))
        xd, yd = divisions(x, k + 1)[:k], divisions(y, k + 1)[:k]
    return dict(opts, x_divs=list(xd), y_divs=list(yd))


def te_network(series, n_surrogates=19, alpha=0.05, seed=0, log=True, **kwargs):
    """Transfer entropy TE[i, j] from series[:, i] to series[:, j], its surrogate p-values and
    significance. series: (n_times, n_nodes). Constant series give zero and p = 1."""
    cami = _cami()
    X = np.column_stack([prepare(c, log) for c in np.asarray(series, dtype=float).T])
    n = X.shape[1]
    te = np.zeros((n, n))
    p = np.ones((n, n))
    varying = np.array([np.ptp(X[:, i]) > 0 for i in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j or not (varying[i] and varying[j]):
                continue
            res = cami.surrogate_test(X[:, i], X[:, j], measure='transfer_entropy', n_surrogates=n_surrogates,
                                      alpha=alpha, seed=seed + i * n + j, **_options(X[:, i], X[:, j], kwargs))
            te[i, j], p[i, j] = res['value'], res['p_value']
    return dict(te=te, p=p, significant=p <= alpha)


def distance_decay(matrix, centres, bins=None):
    """Mean of matrix[i, j] (i != j) by the distance between centres i and j: (distances,
    means, counts)."""
    centres = np.asarray(centres, dtype=float)
    d = np.hypot(*(centres[:, None, :] - centres[None, :, :]).transpose(2, 0, 1))
    off = ~np.eye(len(centres), dtype=bool)
    dist, vals = d[off], np.asarray(matrix)[off]
    if bins is None:
        bins = np.unique(np.round(dist, 6))
        means = np.array([vals[np.isclose(dist, b)].mean() for b in bins])
        counts = np.array([np.isclose(dist, b).sum() for b in bins])
        return bins, means, counts
    idx = np.digitize(dist, bins)
    means = np.array([vals[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(bins))])
    counts = np.array([np.sum(idx == k) for k in range(1, len(bins))])
    return 0.5 * (bins[1:] + bins[:-1]), means, counts


def pair_flow(x, y, n_surrogates=0, seed=0, log=False, **kwargs):
    """Information flow between x and y: dict with te_xy, te_yx, cami_xy, cami_yx, mutual_info,
    directionality (> 0: net flow x -> y) and, with surrogates, p-values of te_xy and te_yx."""
    cami = _cami()
    x, y = prepare(x, log), prepare(y, log)
    out = dict(cami.measures(x, y, **_options(x, y, kwargs)))
    if n_surrogates:
        out['p_xy'] = cami.surrogate_test(x, y, measure='transfer_entropy', n_surrogates=n_surrogates, seed=seed,
                                          **_options(x, y, kwargs))['p_value']
        out['p_yx'] = cami.surrogate_test(y, x, measure='transfer_entropy', n_surrogates=n_surrogates,
                                          seed=seed + 1, **_options(y, x, kwargs))['p_value']
    return out


def pooled_windows(x_windows, y_windows, log=False, **kwargs):
    """cami.multithread_causality over many short windows (lists of arrays, or 2-D arrays with
    one window per column)."""
    cami = _cami()
    opts = {**DEFAULTS, **kwargs}
    def windows(w):
        return list(np.asarray(w, dtype=float).T) if isinstance(w, np.ndarray) and w.ndim == 2 else list(w)

    xs = [prepare(w, log) for w in windows(x_windows)]
    ys = [prepare(w, log) for w in windows(y_windows)]
    return cami.multithread_causality(xs, ys, **opts)


def event_rate(event_times_s, times_s):
    """Number of events in each interval [times_s[k], times_s[k + 1])."""
    edges = np.append(np.asarray(times_s, dtype=float), np.inf)
    return np.histogram(event_times_s, bins=edges)[0].astype(float)
