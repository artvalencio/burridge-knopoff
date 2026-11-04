"""
Lyapunov exponents and predictability (plan, section 6.3).

The model is piecewise smooth (stick-slip switches, failure thresholds) and,
by default, noise-driven (strengths are redrawn at random). Exponents are
defined for the deterministic model (redraw_thresholds=False) and for the
noise-driven model with shared noise (rng_mode='per-block').

State distance: the stress-gap vector D = F - S (force minus strength of each
block), RMS over the blocks, in units of the mean strength excess F_s - F_d.
Perturbations are applied to the displacements (and, when strengths differ,
to the strengths), which change D linearly; rescaling the differences of u
and S rescales the difference of D by the same factor.

- event_map_spectrum: full spectrum of the event-to-event map on small grids.
  The map w_n -> w_{n+1} (w = u - plate, the state just after an event) is
  differentiated by directional finite differences inside one branch (same
  hypocentre, same set of slipping blocks); this includes the shift of the
  failure time and the switching of blocks, i.e. the saltation terms of the
  tangent dynamics (Mueller, 1995), and is re-orthonormalised by QR (Benettin
  et al., 1980). Also the Kaplan-Yorke dimension.
- two_trajectory: largest exponent from a reference and a perturbed run,
  renormalised at regular times.
- finite_size: finite-size Lyapunov exponent lambda(delta) (Aurell et al., 1997).
- predictability_horizon: time until a perturbed run's catalogue differs.
- rosenstein, kantz: estimators from a scalar time series.
"""
import dataclasses
import math

import numpy as np

from ..config import SECONDS_PER_YEAR
from ..model import Model
from ..simulation import Runner


# ---- the event map (small grids, deterministic) ---------------------------------------
class EventMap:
    """The event-to-event map of a deterministic configuration (redraw_thresholds=False)."""

    def __init__(self, config, n_transient=200, seed_state=None):
        if config.redraw_thresholds:
            raise ValueError('the event map needs fixed strengths (redraw_thresholds=False)')
        self.model = Model(config)
        for __ in range(n_transient):
            self._event(self.model)

    @staticmethod
    def _event(model):
        dt, idx = model.time_to_failure()
        model.load(dt)
        result = model.slip(idx)
        return idx, result

    def state(self):
        m = self.model
        return m.u - m.plate_m

    def _save(self):
        m = self.model
        return m.u.copy(), m.v.copy(), m.plate_m, m.time_s, m.current_force.copy()

    def _load(self, saved):
        m = self.model
        m.u[:], m.v[:], m.plate_m, m.time_s, m.current_force[:] = saved[0], saved[1], saved[2], saved[3], saved[4]

    def apply(self, w):
        """Image of state w (relative displacements just after an event) under one event, with
        its branch (hypocentre, slipped set) and duration until the event (s). The model is
        left unchanged."""
        m = self.model
        saved = self._save()
        m.u[:] = w + m.plate_m
        m.v[:] = 0.0
        m.current_force[:] = m.force()
        t0 = m.time_s
        idx, result = self._event(m)
        out = m.u - m.plate_m, (idx, frozenset(result.slipped.tolist())), m.time_s - t0
        self._load(saved)
        return out

    def step(self):
        """Advance the model by one event; returns the elapsed time (s)."""
        t0 = self.model.time_s
        self._event(self.model)
        return self.model.time_s - t0


def event_map_spectrum(config, n_events=300, n_transient=200, eps=1e-7, n_vectors=None, seed=0):
    """Lyapunov spectrum of the event map: dict with exponents per event and per year (sorted,
    largest first), Kaplan-Yorke dimension, number of branch retries, and the events used.
    eps is the finite-difference step in metres."""
    emap = EventMap(config, n_transient=n_transient)
    n = config.n_blocks
    k = n if n_vectors is None else min(n_vectors, n)
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, k)))[0]
    logs = np.zeros(k)
    elapsed = 0.0
    retries = 0
    for __ in range(n_events):
        w0 = emap.state()
        base, branch, __ = emap.apply(w0)
        Z = np.empty((n, k))
        for j in range(k):
            h = eps
            for attempt in range(6):
                image, b, __ = emap.apply(w0 + h * Q[:, j])
                if b == branch:
                    break
                image_m, b_m, __ = emap.apply(w0 - h * Q[:, j])
                if b_m == branch:
                    image, h = image_m, -h
                    break
                retries += 1
                h *= 0.1
            Z[:, j] = (image - base) / h
        Q, R = np.linalg.qr(Z)
        diag = np.abs(np.diag(R))
        logs += np.log(np.maximum(diag, 1e-300))
        elapsed += emap.step()
    per_event = np.sort(logs / n_events)[::-1]
    per_year = per_event * n_events / (elapsed / SECONDS_PER_YEAR)
    return dict(per_event=per_event, per_year=per_year, kaplan_yorke=kaplan_yorke(per_event),
                retries=retries, n_events=n_events, years=elapsed / SECONDS_PER_YEAR)


def kaplan_yorke(exponents):
    """Kaplan-Yorke (Lyapunov) dimension of a spectrum."""
    lam = np.sort(np.asarray(exponents, dtype=float))[::-1]
    cum = np.cumsum(lam)
    if cum[0] < 0:
        return 0.0
    j = int(np.max(np.flatnonzero(cum >= 0)))
    if j == len(lam) - 1:
        return float(len(lam))
    return float(j + 1 + cum[j] / abs(lam[j + 1]))


# ---- trajectories of the full model -----------------------------------------------------
def _gap(runner):
    m = runner.model
    return m.current_force - m.static_force


def _excess(config):
    return config.static_force_n - config.dynamic_force_n


def distance(a, b):
    """RMS difference of the stress gaps of two runners, in units of the strength excess."""
    return float(np.sqrt(np.mean((_gap(a) - _gap(b)) ** 2)) / _excess(a.config))


def _rescale(ref, pert, factor):
    # pert <- ref + factor (pert - ref) for displacements and strengths (D changes linearly)
    mr, mp = ref.model, pert.model
    mp.u[:] = mr.u + factor * (mp.u - mr.u)
    mp.static_force[:] = mr.static_force + factor * (mp.static_force - mr.static_force)
    mp.current_force[:] = mp.force()


def perturbed_pair(config, start_years, delta, seed=0):
    """(reference, perturbed) runners at start_years, the perturbed one displaced in a random
    direction so that their distance is delta."""
    ref = Runner(config, samples=False)
    ref.advance(start_years * SECONDS_PER_YEAR)
    pert = Runner(config, samples=False)
    perturb(ref, pert, delta, seed)
    return ref, pert


def perturb(ref, pert, delta, seed=0):
    """Puts pert at distance delta from ref, in a random direction of the displacements."""
    pert.restore(ref.snapshot())
    rng = np.random.default_rng(seed)
    pert.model.u[:] += rng.normal(size=ref.config.n_blocks) * 1e-3
    pert.model.current_force[:] = pert.model.force()
    _rescale(ref, pert, delta / distance(ref, pert))


def two_trajectory(config, delta=1e-9, renorm_years=0.5, total_years=50.0, start_years=None, seed=0):
    """Largest Lyapunov exponent (per year and per event of the reference run) by renormalised
    divergence of two trajectories (Benettin). Returns a dict with the exponent, its running
    estimate and the log growth factors."""
    start = config.spinup_years if start_years is None else start_years
    run_cfg = dataclasses.replace(config, simulation_years=start + total_years + 1.0)
    ref, pert = perturbed_pair(run_cfg, start, delta, seed)
    growth = []
    n0 = len(ref.events)
    t = start
    while t < start + total_years - 1e-9:
        t = min(t + renorm_years, start + total_years)
        ref.advance(t * SECONDS_PER_YEAR)
        pert.advance(t * SECONDS_PER_YEAR)
        d = distance(ref, pert)
        if d == 0.0:
            growth.append(-np.inf)                 # the perturbation healed: start a new one
            perturb(ref, pert, delta, seed + len(growth))
            continue
        growth.append(math.log(d / delta))
        _rescale(ref, pert, delta / d)
    g = np.array(growth)
    finite = np.isfinite(g)
    per_year = float(np.sum(g[finite]) / (renorm_years * finite.sum())) if finite.any() else math.nan
    n_events = len(ref.events) - n0
    return dict(per_year=per_year, per_event=per_year * total_years / max(n_events, 1),
                running=np.cumsum(np.where(finite, g, 0)) / (renorm_years * np.cumsum(finite).clip(min=1)),
                log_growth=g, n_events=n_events)


def finite_size(config, delta0=1e-10, ratio=2.0, n_scales=20, check_years=0.05, max_years=50.0, start_years=None,
                n_samples=5, seed=0):
    """Finite-size Lyapunov exponent: lambda(delta) = ln(ratio) / <time for a perturbation to
    grow from delta to ratio delta>, for delta = delta0 ratio^n (per year). Perturbations
    that never reach a scale within max_years do not contribute to it."""
    start = config.spinup_years if start_years is None else start_years
    run_cfg = dataclasses.replace(config, simulation_years=start + n_samples * max_years + max_years + 1)
    scales = delta0 * ratio ** np.arange(n_scales + 1)
    times = [[] for __ in range(n_scales)]
    for s in range(n_samples):
        t_start = start + s * max_years
        ref, pert = perturbed_pair(run_cfg, t_start, delta0, seed + s)
        reached = np.full(n_scales + 1, np.nan)
        reached[0] = 0.0
        t = t_start
        while t < t_start + max_years and np.isnan(reached[-1]):
            t += check_years
            ref.advance(t * SECONDS_PER_YEAR)
            pert.advance(t * SECONDS_PER_YEAR)
            d = distance(ref, pert)
            newly = np.isnan(reached) & (scales <= d)
            reached[newly] = t - t_start
        for n in range(n_scales):
            if np.isfinite(reached[n]) and np.isfinite(reached[n + 1]):
                times[n].append(max(reached[n + 1] - reached[n], check_years))
    lam = np.array([math.log(ratio) / np.mean(ts) if ts else math.nan for ts in times])
    return dict(scales=scales[:-1], exponent=lam, samples=[len(ts) for ts in times])


def predictability_horizon(config, delta=1e-9, horizon_years=30.0, start_years=None, n_samples=5,
                           magnitude_classes=(0.0, 5.0, 6.0, 7.0), seed=0):
    """For each magnitude class (events with Mw >= m), the time after a perturbation of size
    delta until the perturbed catalogue first differs from the reference: a different
    hypocentre, or a magnitude differing by more than 0.1, or a missing event. Returns a dict
    {m: array of horizons (years; nan if no difference within the horizon)}."""
    start = config.spinup_years if start_years is None else start_years
    run_cfg = dataclasses.replace(config, simulation_years=start + n_samples * horizon_years + horizon_years + 1)
    out = {m: [] for m in magnitude_classes}
    for s in range(n_samples):
        t0 = start + s * horizon_years
        ref, pert = perturbed_pair(run_cfg, t0, delta, seed + s)
        n_r, n_p = len(ref.events), len(pert.events)
        ref.advance((t0 + horizon_years) * SECONDS_PER_YEAR)
        pert.advance((t0 + horizon_years) * SECONDS_PER_YEAR)
        er, ep = ref.events[n_r:], pert.events[n_p:]
        for m in magnitude_classes:
            a = [e for e in er if e.magnitude >= m]
            b = [e for e in ep if e.magnitude >= m]
            horizon = math.nan
            for i in range(max(len(a), len(b))):
                if i >= len(a) or i >= len(b):
                    horizon = (a if i < len(a) else b)[i].start_years - t0
                    break
                if a[i].hypocentre != b[i].hypocentre or abs(a[i].magnitude - b[i].magnitude) > 0.1:
                    horizon = min(a[i].start_years, b[i].start_years) - t0
                    break
            out[m].append(horizon)
    return {m: np.array(v) for m, v in out.items()}


# ---- estimators from data ------------------------------------------------------------
def rosenstein(x, dim, tau, theiler, max_steps, fit=(0, None)):
    """Mean log divergence of initially nearest neighbours (Rosenstein et al., 1993): dict with
    the curve <ln d(k)>, k = 0..max_steps, and its slope over fit (per sample)."""
    from .embedding import _nearest, delay_vectors
    X = delay_vectors(x, dim, tau)
    n = len(X)
    j, __ = _nearest(X, theiler)
    i = np.arange(n)
    curve = np.full(max_steps + 1, np.nan)
    for k in range(max_steps + 1):
        ok = (i + k < n) & (j + k < n)
        d = np.max(np.abs(X[i[ok] + k] - X[j[ok] + k]), axis=1)
        d = d[d > 0]
        curve[k] = np.mean(np.log(d)) if len(d) else np.nan
    lo, hi = fit[0], (max_steps if fit[1] is None else fit[1])
    ks = np.arange(lo, hi + 1)
    slope = float(np.polyfit(ks, curve[lo:hi + 1], 1)[0])
    return dict(curve=curve, slope=slope)


def kantz(x, dim, tau, eps, theiler, max_steps, fit=(0, None), max_points=2000, seed=0):
    """Kantz (1994): S(k) = < ln < |x_{i+k} - x_{j+k}| >_{j in U_eps(i)} >_i (last delay
    coordinate), and its slope over fit (per sample)."""
    from .embedding import delay_vectors
    X = delay_vectors(x, dim, tau)
    n = len(X) - max_steps
    rng = np.random.default_rng(seed)
    ref = rng.choice(n, size=min(max_points, n), replace=False)
    from scipy.spatial import cKDTree
    tree = cKDTree(X[:n])
    S = np.zeros(max_steps + 1)
    used = 0
    for i in ref:
        nb = np.array(tree.query_ball_point(X[i], eps, p=np.inf))
        nb = nb[np.abs(nb - i) > theiler] if len(nb) else nb
        if len(nb) == 0:
            continue
        for k in range(max_steps + 1):
            S[k] += math.log(max(np.mean(np.abs(X[nb + k, -1] - X[i + k, -1])), 1e-300))
        used += 1
    S /= max(used, 1)
    lo, hi = fit[0], (max_steps if fit[1] is None else fit[1])
    ks = np.arange(lo, hi + 1)
    return dict(curve=S, slope=float(np.polyfit(ks, S[lo:hi + 1], 1)[0]), used=used)
