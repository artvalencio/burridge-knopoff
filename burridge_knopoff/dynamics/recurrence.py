"""
Recurrence plots and recurrence quantification analysis (Eckmann et al., 1987;
Marwan et al., 2007).

- recurrence_matrix: R[i, j] = 1 if |x_i - x_j| <= eps (maximum norm); eps can be
  set by a target recurrence rate, so that runs can be compared
- rqa: recurrence rate RR, determinism DET, longest diagonal L_max, entropy of
  the diagonal lengths ENTR, laminarity LAM and trapping time TT; the main
  diagonal and lines within the Theiler window are excluded
- windowed_rqa: the measures in sliding windows of a long series
- cross_recurrence, joint_recurrence (Romano et al., 2004)
- jaccard_similarity: overlap of the block sets of ruptures
"""
import numpy as np
from numba import njit


def _distances(x, y=None):
    x = np.asarray(x, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    y = x if y is None else (np.asarray(y, dtype=float)[:, None] if np.ndim(y) == 1 else np.asarray(y, dtype=float))
    return np.max(np.abs(x[:, None, :] - y[None, :, :]), axis=2)


def threshold_for_rate(distances, rate, theiler=0):
    """The eps giving recurrence rate `rate` outside the Theiler window."""
    n = distances.shape[0]
    i, j = np.triu_indices(n, k=theiler + 1)
    return float(np.quantile(distances[i, j], rate))


def recurrence_matrix(x, eps=None, rate=None, theiler=0):
    """Boolean recurrence matrix of the states x (n, dim) or a scalar series; give eps or a
    target recurrence rate."""
    d = _distances(x)
    if eps is None:
        if rate is None:
            raise ValueError('give eps or rate')
        eps = threshold_for_rate(d, rate, theiler)
    return d <= eps


def cross_recurrence(x, y, eps):
    return _distances(x, y) <= eps


def joint_recurrence(x, y, rate=0.05, theiler=0):
    """Joint recurrence: both systems recur at the same times (each at its own threshold)."""
    return recurrence_matrix(x, rate=rate, theiler=theiler) & recurrence_matrix(y, rate=rate, theiler=theiler)


@njit(cache=True)
def _line_histograms(R, theiler):
    # histograms of diagonal line lengths (upper triangle, |i - j| > theiler) and of vertical
    # line lengths (all columns, excluding the Theiler band)
    n = R.shape[0]
    diag = np.zeros(n + 1, dtype=np.int64)
    for k in range(theiler + 1, n):
        run = 0
        for i in range(n - k):
            if R[i, i + k]:
                run += 1
            else:
                if run:
                    diag[run] += 1
                run = 0
        if run:
            diag[run] += 1
    vert = np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        run = 0
        for i in range(n):
            if R[i, j] and abs(i - j) > theiler:
                run += 1
            else:
                if run:
                    vert[run] += 1
                run = 0
        if run:
            vert[run] += 1
    return diag, vert


def rqa(R, lmin=2, vmin=2, theiler=0):
    """Recurrence quantification of a (square, symmetric) recurrence matrix. Returns a dict:
    RR, DET, L_max, L_mean, ENTR, LAM, TT."""
    R = np.ascontiguousarray(np.asarray(R, dtype=np.bool_))
    n = R.shape[0]
    diag, vert = _line_histograms(R, theiler)
    lengths = np.arange(n + 1)
    points_upper = float(np.sum(diag * lengths))
    pairs = (n - theiler - 1) * (n - theiler) / 2.0
    rr = points_upper / pairs if pairs > 0 else 0.0
    long_d = lengths >= lmin
    det = float(np.sum(diag[long_d] * lengths[long_d]) / points_upper) if points_upper else 0.0
    n_long = diag[long_d].sum()
    l_mean = float(np.sum(diag[long_d] * lengths[long_d]) / n_long) if n_long else 0.0
    l_max = int(lengths[diag > 0].max()) if np.any(diag > 0) else 0
    if n_long:
        p = diag[long_d] / n_long
        p = p[p > 0]
        entr = float(-np.sum(p * np.log(p)))
    else:
        entr = 0.0
    points_all = float(np.sum(vert * lengths))
    long_v = lengths >= vmin
    lam = float(np.sum(vert[long_v] * lengths[long_v]) / points_all) if points_all else 0.0
    n_v = vert[long_v].sum()
    tt = float(np.sum(vert[long_v] * lengths[long_v]) / n_v) if n_v else 0.0
    return dict(RR=rr, DET=det, L_max=l_max, L_mean=l_mean, ENTR=entr, LAM=lam, TT=tt)


def windowed_rqa(states, window, step, rate=0.05, theiler=1, lmin=2):
    """RQA in windows of `window` states every `step`: (window starts, dict of arrays). The
    threshold is set per window from the recurrence rate."""
    states = np.asarray(states, dtype=float)
    starts = np.arange(0, len(states) - window + 1, step)
    rows = [rqa(recurrence_matrix(states[s:s + window], rate=rate, theiler=theiler), lmin=lmin, theiler=theiler)
            for s in starts]
    return starts, {key: np.array([r[key] for r in rows]) for key in (rows[0] if rows else {})}


def jaccard_similarity(block_sets):
    """Matrix of |A & B| / |A | B| between the block sets of ruptures."""
    sets = [set(np.asarray(b).tolist()) for b in block_sets]
    n = len(sets)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(sets[i] | sets[j])
            out[i, j] = out[j, i] = len(sets[i] & sets[j]) / union if union else 0.0
    return out
