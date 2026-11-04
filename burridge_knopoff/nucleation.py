"""
Precursory creep of stuck blocks: rate-and-state nucleation (config.preslip).

A stuck block creeps at

    v = v_s exp(Gamma / (a sigma)),     Gamma = (F + C - S) / A,

with F the spring force, C the tidal Coulomb force (0 without tides), S the strength
and A the block area: Gamma is the stress minus the strength (Pa), and the block fails
(the event starts) when Gamma reaches 0, creeping at v_s. Creep by delta lowers F by
k_sum delta (the block's own springs) and S by (b sigma / D_c) A delta (the evolution of
the state variable during nucleation, theta ~ exp(-delta / D_c); Dieterich, 1992), so

    dGamma/dt = r + H sigma v,     H sigma = b sigma / D_c - k_sum / A  (> 0),

where r is the stressing rate from the plate, the tide and the creep of the coupled
blocks. For constant r this is solved exactly: with s = r / (a sigma), v0 the rate at
the start and E(h) = expm1(s h) / s,

    creep over h:          delta = -(a sigma / H sigma) log1p(-H sigma v0 E(h) / (a sigma))
    time to instability:   T = log1p(s Q) / s,   Q = a sigma / (H sigma v0)   (inf if s Q <= -1)

(T = Q for s = 0). For r = plate rate this is Dieterich's (1992) nucleation: v grows
as a sigma / (H sigma T) in the last stage, and the nucleation lasts about
t_a = a sigma / r, a sigma ln(v_s H sigma / r) below the nominal strength.

Integration (advance): the coupled blocks interact through r, which is frozen over each
step and updated between steps. Blocks creeping faster than rs_min_velocity_m_s are
candidates. Fast candidates (those that could fail before the end of the call) are
stepped with the step halved towards the next instability, so the failing block and
its neighbours are resolved down to the last milliseconds; slow candidates creep by the
exact formula once, at the end of the call. Blocks are promoted to fast when the creep
of a neighbour or the tide brings them close to failure. Plate loading is not applied
to the force array here (Model.load does it after the call).
"""
import math

import numpy as np
from numba import njit

from .model import _cubic_coefficients


@njit(cache=True)
def _tide(coulomb, step_s, t, x, y):
    # tidal Coulomb force (N) at t and its rate (N/s) at block coordinates (x, y), from the
    # Catmull-Rom interpolation of the tabulated coefficients (c0, cx, cy)
    k = int(np.floor(t / step_s)) + 1
    u = t / step_s - (k - 1)
    value = 0.0
    rate = 0.0
    for j in range(3):
        a0, a1, a2, a3 = _cubic_coefficients(coulomb, k, j)
        w = 1.0 if j == 0 else (x if j == 1 else y)
        value += w * (a0 + u * (a1 + u * (a2 + u * a3)))
        rate += w * (a1 + u * (2.0 * a2 + u * 3.0 * a3)) / step_s
    return value, rate


@njit(cache=True, inline='always')
def _expm1_over(s, h):
    # expm1(s h) / s, -> h as s -> 0
    x = s * h
    if abs(x) < 1e-8:
        return h * (1.0 + 0.5 * x)
    return math.expm1(x) / s


@njit(cache=True, inline='always')
def time_to_instability(v0, r, a_sigma, h_sigma):
    """Time until a block creeping at v0 (m/s) under a constant stressing rate r (Pa/s)
    reaches Gamma = 0 (inf if never)."""
    if v0 <= 0.0:
        return np.inf
    q = a_sigma / (h_sigma * v0)
    s = r / a_sigma
    x = s * q
    if abs(x) < 1e-8:
        return q * (1.0 - 0.5 * x)
    if x <= -1.0:
        return np.inf
    return math.log1p(x) / s


@njit(cache=True, inline='always')
def creep_over(v0, r, h, a_sigma, h_sigma):
    """Creep (m) in a time h of a block starting at v0 (m/s) under a constant stressing rate r
    (Pa/s); -1 if it reaches instability within h."""
    if v0 <= 0.0:
        return 0.0
    arg = -h_sigma * v0 * _expm1_over(r / a_sigma, h) / a_sigma
    if arg <= -1.0:
        return -1.0
    return -(a_sigma / h_sigma) * math.log1p(arg)


@njit(cache=True)
def _neighbour_rate(i, vel, nx, ny, drow, dcol, kw):
    # sum_j k_ij v_j over the blocks coupled to i (N/s)
    row = i // nx
    col = i - row * nx
    acc = 0.0
    for s in range(drow.shape[0]):
        r = row + drow[s]
        c = col + dcol[s]
        if 0 <= r < ny and 0 <= c < nx:
            acc += kw[s] * vel[r * nx + c]
    return acc


@njit(cache=True)
def advance(u, force, static, preslip, ksum, kw, drow, dcol, nx, ny, kp_vp, area, a_sigma, h_sigma,
            weaken, v_s, gamma_cut, t0, t_stop, tidal, coulomb, step_s, xc, yc, tide_margin, tide_cap_s,
            state, cand, fast, vel, t_join, dstep, dcorr, max_iterations):
    """Creeps every candidate block from t0 until the first instability or t_stop.

    u, force, static, preslip are updated in place (force without the plate loading of the
    call, which the caller applies). Work arrays (size n): state (int8, all 0 on entry and
    return: 0 locked, 1 slow, 2 fast candidate), cand, fast (int64), vel, t_join, dstep, dcorr (float).
    Returns (t_end, failed block or -1)."""
    n = u.shape[0]
    n_st = drow.shape[0]
    plate_rate = kp_vp / area  # Pa/s
    horizon = t_stop - t0
    margin = plate_rate * horizon + tide_margin
    # ---- candidates and their creep rates at t0
    n_cand = 0
    for i in range(n):
        gamma = (force[i] - static[i]) / area
        if gamma > gamma_cut - margin:
            state[i] = 1
            cand[n_cand] = i
            n_cand += 1
            t_join[i] = t0
    for k in range(n_cand):
        i = cand[k]
        c = 0.0
        if tidal:
            c = _tide(coulomb, step_s, t0, xc[i], yc[i])[0]
        gamma = (force[i] - static[i] + c) / area
        vel[i] = v_s * math.exp(gamma / a_sigma) if gamma > gamma_cut else 0.0
    # fast: could fail before t_stop (with the most favourable tide), or already unstable
    n_fast = 0
    for k in range(n_cand):
        i = cand[k]
        gamma = (force[i] - static[i]) / area
        v_best = v_s * math.exp(min(gamma + tide_margin, 0.0) / a_sigma)
        r = plate_rate + _neighbour_rate(i, vel, nx, ny, drow, dcol, kw) / area
        if vel[i] > 0.0 and (gamma >= 0.0 or time_to_instability(v_best, r, a_sigma, h_sigma[i]) < 2.0 * horizon):
            state[i] = 2
            fast[n_fast] = i
            n_fast += 1
    # ---- step the fast blocks
    t = t0
    failed = -1
    it = 0
    while t < t_stop and n_fast > 0:
        it += 1
        if it > max_iterations:
            raise RuntimeError('creep integration did not converge')
        shift = kp_vp * (t - t0)
        # rates and the next instability
        t_min = np.inf
        i_min = -1
        for k in range(n_fast):
            i = fast[k]
            c = 0.0
            if tidal:
                c = _tide(coulomb, step_s, t, xc[i], yc[i])[0]
            gamma = (force[i] + shift - static[i] + c) / area
            if gamma >= 0.0:
                t_min = 0.0
                i_min = i
                vel[i] = v_s
                continue
            vel[i] = v_s * math.exp(gamma / a_sigma) if gamma > gamma_cut else 0.0
        if t_min == 0.0:
            failed = i_min
            break
        r_max = plate_rate
        for k in range(n_fast):
            i = fast[k]
            r = kp_vp + _neighbour_rate(i, vel, nx, ny, drow, dcol, kw)
            if tidal:
                r += _tide(coulomb, step_s, t, xc[i], yc[i])[1]
            r /= area
            t_join[i] = r  # reuse as the frozen rate of this step (fast blocks only)
            if abs(r) > r_max:
                r_max = abs(r)
            tt = time_to_instability(vel[i], r, a_sigma, h_sigma[i])
            if tt < t_min:
                t_min = tt
                i_min = i
        h = min(t_stop - t, 0.5 * a_sigma / r_max)
        if tidal:
            h = min(h, tide_cap_s)
        final = False
        if t_min <= h:
            if t_min <= 10.0 * a_sigma / (h_sigma[i_min] * v_s):
                h = t_min
                final = True
            else:
                h = 0.5 * t_min
        # predictor: creep of every fast block over h at its frozen rate; corrector: the same with
        # the coupled fast blocks loading at their mean creep rate over the step (dstep / h). The
        # step is halved until the loading assumed by the corrector matches the corrected creep
        # within 0.05 a sigma on every block (so no block overshoots Gamma = 0).
        n_old = n_fast
        while True:
            for k in range(n_old):
                i = fast[k]
                dstep[i] = _step_creep(i, final and i == i_min, t_join[i], h, t, shift, force, static, vel, area,
                                       a_sigma, h_sigma, tidal, coulomb, step_s, xc, yc)
            for k in range(n_old):
                i = fast[k]
                extra = _fast_neighbours(i, dstep, vel, state, h, nx, ny, drow, dcol, kw, True)
                dcorr[i] = _step_creep(i, final and i == i_min, t_join[i] + extra / area, h, t, shift, force,
                                       static, vel, area, a_sigma, h_sigma, tidal, coulomb, step_s, xc, yc)
            err = 0.0
            for k in range(n_old):
                i = fast[k]
                e = abs(_fast_neighbours(i, dcorr, dstep, state, h, nx, ny, drow, dcol, kw, False)) / area
                if e > err:
                    err = e
            if err <= 0.05 * a_sigma or h < 1e-9:
                break
            h *= 0.5
            final = False
        for k in range(n_old):
            i = fast[k]
            d = dcorr[i]
            if d == 0.0:
                continue
            u[i] += d
            preslip[i] += d
            force[i] -= ksum[i] * d
            static[i] -= weaken * d
            row = i // nx
            col = i - row * nx
            for s in range(n_st):
                rr = row + drow[s]
                cc = col + dcol[s]
                if 0 <= rr < ny and 0 <= cc < nx:
                    j = rr * nx + cc
                    if state[j] == 1:
                        # a slow candidate creeps at its old rate up to now before it is loaded
                        _settle(j, t + h, u, force, static, preslip, ksum, kw, drow, dcol, nx, ny, kp_vp, area,
                                a_sigma, h_sigma, weaken, vel, t_join, t0)
                    force[j] += kw[s] * d
                    if state[j] == 2:
                        continue
                    shift_end = kp_vp * (t + h - t0)
                    gamma = (force[j] + shift_end - static[j]) / area
                    if state[j] == 0:
                        if gamma <= gamma_cut - margin:
                            continue
                        state[j] = 1
                        cand[n_cand] = j
                        n_cand += 1
                    # the slow creep restarts from now at the new rate
                    t_join[j] = t + h
                    vel[j] = v_s * math.exp(min(gamma, 0.0) / a_sigma) if gamma > gamma_cut else 0.0
                    # a coupled block brought close to failure joins the fast set
                    v_best = v_s * math.exp(min(gamma + tide_margin, 0.0) / a_sigma)
                    if gamma >= 0.0 or time_to_instability(v_best, plate_rate, a_sigma, h_sigma[j]) < 2.0 * (t_stop - t):
                        state[j] = 2
                        fast[n_fast] = j
                        n_fast += 1
        t = t_stop if (not final and h >= t_stop - t) else t + h
        if final:
            failed = i_min
            break
    t_end = t if failed >= 0 else t_stop
    # ---- slow candidates creep once over their time in the call
    for k in range(n_cand):
        i = cand[k]
        if state[i] == 1:
            _settle(i, t_end, u, force, static, preslip, ksum, kw, drow, dcol, nx, ny, kp_vp, area,
                    a_sigma, h_sigma, weaken, vel, t_join, t0)
        state[i] = 0
        vel[i] = 0.0
    return t_end, failed


@njit(cache=True)
def _fast_neighbours(i, d_new, d_ref, state, h, nx, ny, drow, dcol, kw, rate):
    # sum over the fast blocks j coupled to i of k_ij (d_new[j] / h - d_ref[j]) (rate) or
    # k_ij (d_new[j] - d_ref[j])
    row = i // nx
    col = i - row * nx
    acc = 0.0
    for s in range(drow.shape[0]):
        rr = row + drow[s]
        cc = col + dcol[s]
        if 0 <= rr < ny and 0 <= cc < nx:
            j = rr * nx + cc
            if state[j] == 2:
                acc += kw[s] * (d_new[j] / h - d_ref[j] if rate else d_new[j] - d_ref[j])
    return acc


@njit(cache=True)
def _step_creep(i, final, r, h, t, shift, force, static, vel, area, a_sigma, h_sigma, tidal, coulomb, step_s,
                xc, yc):
    # creep of fast block i over h at the stressing rate r; the failing block (final) and a block
    # that would become unstable within h stop exactly at Gamma = 0
    d = -1.0 if final else creep_over(vel[i], r, h, a_sigma, h_sigma[i])
    if d < 0.0:
        c = 0.0
        if tidal:
            c = _tide(coulomb, step_s, t, xc[i], yc[i])[0]
        gamma = (force[i] + shift - static[i] + c) / area
        d = max((-gamma - r * h) / h_sigma[i], 0.0)
    return d


@njit(cache=True)
def _settle(i, t_end, u, force, static, preslip, ksum, kw, drow, dcol, nx, ny, kp_vp, area, a_sigma, h_sigma,
            weaken, vel, t_join, t0):
    # creep of a slow candidate from t_join[i] to t_end at the plate stressing rate, from the
    # rate it had when it joined (stopped short of instability)
    h = t_end - t_join[i]
    if h <= 0.0 or vel[i] <= 0.0:
        return
    r = kp_vp / area
    d = creep_over(vel[i], r, h, a_sigma, h_sigma[i])
    if d < 0.0:
        shift = kp_vp * (t_join[i] - t0)
        gamma = (force[i] + shift - static[i]) / area
        d = max((-gamma - r * h) / h_sigma[i], 0.0) * 0.999
    if d == 0.0:
        return
    u[i] += d
    preslip[i] += d
    force[i] -= ksum[i] * d
    static[i] -= weaken * d
    row = i // nx
    col = i - row * nx
    for s in range(drow.shape[0]):
        rr = row + drow[s]
        cc = col + dcol[s]
        if 0 <= rr < ny and 0 <= cc < nx:
            force[rr * nx + cc] += kw[s] * d
