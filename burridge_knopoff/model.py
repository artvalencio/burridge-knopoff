"""
The Burridge-Knopoff spring-block model.

Each block i slides along x on a fixed base. Its displacement from the
reference position is u_i and its velocity v_i >= 0. The forces on it are

    coupling:  sum_j k_ij (u_j - u_i)            (free boundaries)
    loading:   k_p (x_p - u_i)                    (leaf spring to the plate)
    friction:  stuck: whatever keeps it at rest, up to its strength F_s,i
               slipping: f_i(v) opposing the motion

where x_p(t) = x_p(0) + v_p t is the plate position. The plate starts at
x_p(0) = F_d / k_p, so every block starts at the dynamic friction level.

Coupling: k_ij = k_c (L / r_ij)^p for blocks within interaction_range_blocks
(r in spacings, p = interaction_exponent; grid.interaction_stencil). The
default range 1 couples the four nearest neighbours (classic model); larger
ranges approximate the 1/r^3 static stress transfer of an elastic body.

Friction while slipping: f_i(v) = F_d by default (Coulomb). With velocity
weakening, f_i(v) = F_d + (1 - s) (F_s,i - F_d) / (1 + v / v_w): friction drops
by the onset fraction s when slip starts, then decays towards F_d.

Event-driven integration
------------------------
While every block is stuck, only the plate moves and each force grows
linearly at the rate k_p v_p: the time until the weakest block reaches its
strength is computed exactly (time_to_failure, load). The dynamics are then
integrated only during slip (slip), and only for the slipping blocks, with
velocity Verlet steps of a fraction of the fastest oscillation period. A
block stops when its velocity returns to zero and the force on it is within
its strength; stuck blocks coupled to moving ones start slipping when their
force exceeds their strength. Blocks never slip backwards. The event ends
when no block moves. This replaces ~1e9 explicit steps through years of
quiet loading by a few hundred steps per event.

Performance: the spring force on every block is maintained incrementally
(current_force). Loading adds k_p v_p dt to all blocks; after an event only
the slipped blocks and the blocks coupled to them are recomputed. The slip
kernel reuses preallocated work arrays, so an event costs O(blocks involved x
coupled offsets) plus a few O(n) vector operations.
"""
import dataclasses

import numpy as np
from numba import njit

from . import grid


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def counter_uniform(seed, blocks, counts):
    """Uniform numbers in [0, 1) that depend only on (seed, block, count): a SplitMix64
    hash of the three. The k-th redraw of a block is then the same whatever the order of
    events, so two nearby trajectories share their noise."""
    with np.errstate(over='ignore'):
        x = (np.uint64(seed) * _GOLDEN + np.asarray(blocks, dtype=np.uint64) * _MIX1
             + np.asarray(counts, dtype=np.uint64) * _MIX2 + _GOLDEN)
        x = (x ^ (x >> np.uint64(30))) * _MIX1
        x = (x ^ (x >> np.uint64(27))) * _MIX2
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


@dataclasses.dataclass
class SlipResult:
    """Outcome of one event: slipped block indices (in order of first failure),
    their displacements before the event and their slips (m), duration (s), peak
    velocity (m/s), energy dissipated by friction (J), work done by the plate
    during the event (J), integration steps, and the work of the tidal force (J)."""
    slipped: np.ndarray
    u_before: np.ndarray
    slip: np.ndarray
    duration_s: float
    peak_velocity_m_s: float
    dissipated_j: float
    plate_work_j: float
    steps: int
    tide_work_j: float = 0.0


@njit(cache=True, inline='always')
def _force_nearest(i, u, plate, nx, ny, kp, kw):
    # loading spring + the four nearest neighbours (left, right, below, above)
    row = i // nx
    col = i - row * nx
    f = kp * (plate - u[i])
    if col > 0:
        f += kw[0] * (u[i - 1] - u[i])
    if col < nx - 1:
        f += kw[1] * (u[i + 1] - u[i])
    if row > 0:
        f += kw[2] * (u[i - nx] - u[i])
    if row < ny - 1:
        f += kw[3] * (u[i + nx] - u[i])
    return f


@njit(cache=True)
def _force_stencil(i, u, plate, nx, ny, kp, drow, dcol, kw):
    # loading spring + coupling to every block of the stencil inside the grid
    row = i // nx
    col = i - row * nx
    f = kp * (plate - u[i])
    for s in range(drow.shape[0]):
        r = row + drow[s]
        c = col + dcol[s]
        if 0 <= r < ny and 0 <= c < nx:
            f += kw[s] * (u[r * nx + c] - u[i])
    return f


@njit(cache=True, inline='always')
def _force(i, u, plate, nx, ny, kp, drow, dcol, kw):
    # the nearest-neighbour stencil gets a small, inlined function (the general loop
    # made the default model ~50% slower)
    if drow.shape[0] == 4:
        return _force_nearest(i, u, plate, nx, ny, kp, kw)
    return _force_stencil(i, u, plate, nx, ny, kp, drow, dcol, kw)


@njit(cache=True, inline='always')
def _friction(v, strength, fd, vw, keep):
    # sliding friction: F_d, or F_d + keep (F_s,i - F_d) / (1 + v / v_w) with keep = 1 - onset drop
    if vw <= 0.0:
        return fd
    if v < 0.0:
        v = 0.0
    return fd + keep * (strength - fd) / (1.0 + v / vw)


@njit(cache=True, inline='always')
def _friction_tidal(v, strength, fd, vw, keep, normal, mus, mud):
    # sliding friction when the normal force changes by `normal` (tension positive): the
    # static strength shifts by -mus normal and F_d by -mud normal (exact zero shift when
    # normal == 0, so the tide-free results are unchanged)
    f = _friction(v, strength, fd, vw, keep) - mud * normal
    if vw > 0.0:
        if v < 0.0:
            v = 0.0
        f -= keep * (mus - mud) * normal / (1.0 + v / vw)
    return f


@njit(cache=True)
def _activate(i, u, active, slipped, order, u_start, n_order, act, n_act):
    # starts (or restarts) block i slipping; the first time, remember where it was
    active[i] = True
    act[n_act] = i
    if not slipped[i]:
        slipped[i] = True
        order[n_order] = i
        u_start[i] = u[i]
        n_order += 1
    return n_order, n_act + 1


@njit(cache=True, inline='always')
def _threshold(static, normal, i, mus, tidal):
    # strength of block i, lowered by the tidal unclamping when tides are on
    return static[i] - mus * normal[i] if tidal else static[i]


@njit(cache=True, inline='always')
def _sliding(v, static, normal, i, fd, vw, keep, mus, mud, tidal):
    if tidal:
        return _friction_tidal(v, static[i], fd, vw, keep, normal[i], mus, mud)
    return _friction(v, static[i], fd, vw, keep)


@njit(cache=True)
def _slip_kernel(u, v, force, static, fd, vw, keep, kp, mass, nx, ny, drow, dcol, kw, ksum, plate, vp, dt,
                 hypocentre, max_steps, active, slipped, act, moved, acc, disp, order, u_start,
                 drive, normal, mus, mud, tidal):
    """Integrates one event in place (u, v, force).

    Tides (frozen during the event): drive[i] is added to the force on block i, and the
    normal force normal[i] (tension positive) lowers its strength by mus normal[i] and its
    sliding friction by mud normal[i]. With tidal False these arrays are not read (the
    tide-free model, unchanged and as fast as before).

    Forces: with the nearest-neighbour stencil each force is recomputed from the
    displacements when needed (4 terms). With longer ranges (incremental mode),
    force[] is updated as blocks move: a block moving by du changes its own force
    by -ksum[i] du (ksum = k_p + its coupling stiffnesses) and the force of each
    coupled block j by k_ij du; the plate motion is added as one common shift.
    Every lookup is then O(1) instead of O(stencil).

    Work arrays: active, slipped (all False on entry and on return), act, moved,
    acc, disp, u_start (size n). On return order[:n_slipped] lists the slipped
    blocks in order of first failure and u_start[order[k]] their displacements
    before. Returns (plate, steps, n_slipped, peak velocity, dissipated energy,
    plate work, completed, tidal work).
    """
    n = u.shape[0]
    n_stencil = drow.shape[0]
    incremental = n_stencil > 4
    plate_start = plate
    n_order, n_act = _activate(hypocentre, u, active, slipped, order, u_start, 0, act, 0)
    sum_u = u.sum()
    peak = 0.0
    dissipated = 0.0
    work = 0.0
    tide_work = 0.0
    steps = 0
    completed = False
    while steps < max_steps:
        if n_act == 0:
            # quiet: did the (tiny) plate motion during the event push any other block over?
            for i in range(n):
                f_i = force[i] + kp * (plate - plate_start) if incremental else _force(i, u, plate, nx, ny, kp, drow, dcol, kw)
                if (f_i + drive[i] if tidal else f_i) > _threshold(static, normal, i, mus, tidal):
                    n_order, n_act = _activate(i, u, active, slipped, order, u_start, n_order, act, n_act)
            if n_act == 0:
                completed = True
                break
        # velocity Verlet: a(t), then u(t + dt), then a(t + dt) and v(t + dt)
        for k in range(n_act):
            i = act[k]
            f_i = force[i] + kp * (plate - plate_start) if incremental else _force(i, u, plate, nx, ny, kp, drow, dcol, kw)
            if tidal:
                f_i += drive[i]
            acc[i] = (f_i - _sliding(v[i], static, normal, i, fd, vw, keep, mus, mud, tidal)) / mass
        sum_before = sum_u
        for k in range(n_act):
            i = act[k]
            du = v[i] * dt + 0.5 * acc[i] * dt * dt
            if du < 0.0:
                du = 0.0  # no backward slip
            u[i] += du
            sum_u += du
            if tidal:
                tide_work += drive[i] * du
            disp[i] = du
            moved[k] = i
        if incremental:
            # forces are linear in u: apply every displacement of this step
            for k in range(n_act):
                i = moved[k]
                du = disp[i]
                if du == 0.0:
                    continue
                force[i] -= ksum[i] * du
                row = i // nx
                col = i - row * nx
                for s in range(n_stencil):
                    r = row + drow[s]
                    c = col + dcol[s]
                    if 0 <= r < ny and 0 <= c < nx:
                        force[r * nx + c] += kw[s] * du
        # plate work during the step: k_p v_p dt sum_i (x_p - u_i), at mid-step
        work += kp * vp * dt * (n * (plate + 0.5 * vp * dt) - 0.5 * (sum_before + sum_u))
        plate += vp * dt
        n_moved = n_act
        n_keep = 0
        for k in range(n_act):
            i = act[k]
            f_old = _sliding(v[i], static, normal, i, fd, vw, keep, mus, mud, tidal)
            v_pred = v[i] + acc[i] * dt  # velocity-dependent friction at the end of the step
            f_new = force[i] + kp * (plate - plate_start) if incremental else _force(i, u, plate, nx, ny, kp, drow, dcol, kw)
            if tidal:
                f_new += drive[i]
            fric_new = _sliding(v_pred, static, normal, i, fd, vw, keep, mus, mud, tidal)
            v_new = v[i] + 0.5 * (acc[i] + (f_new - fric_new) / mass) * dt
            # friction work: mean friction over the step times the distance slid
            dissipated += 0.5 * (f_old + fric_new) * disp[i]
            if v_new <= 0.0:
                v[i] = 0.0
                if f_new <= _threshold(static, normal, i, mus, tidal):
                    active[i] = False  # stops: friction can hold it again
                    continue
            else:
                v[i] = v_new
                if v_new > peak:
                    peak = v_new
            act[n_keep] = i
            n_keep += 1
        n_act = n_keep
        # stuck blocks coupled to the moving ones may now exceed their strength
        for k in range(n_moved):
            i = moved[k]
            row = i // nx
            col = i - row * nx
            for s in range(n_stencil):
                r = row + drow[s]
                c = col + dcol[s]
                if 0 <= r < ny and 0 <= c < nx:
                    j = r * nx + c
                    if active[j]:
                        continue
                    f_j = force[j] + kp * (plate - plate_start) if incremental else _force(j, u, plate, nx, ny, kp, drow, dcol, kw)
                    if (f_j + drive[j] if tidal else f_j) > _threshold(static, normal, j, mus, tidal):
                        n_order, n_act = _activate(j, u, active, slipped, order, u_start, n_order, act, n_act)
        steps += 1
    # update the maintained forces: the plate moved for everyone, and the slipped
    # blocks and the blocks coupled to them are recomputed exactly (this also
    # removes any rounding accumulated by the incremental updates)
    shift = kp * (plate - plate_start)
    for i in range(n):
        force[i] += shift
    for k in range(n_order):
        i = order[k]
        row = i // nx
        col = i - row * nx
        force[i] = _force(i, u, plate, nx, ny, kp, drow, dcol, kw)
        for s in range(n_stencil):
            r = row + drow[s]
            c = col + dcol[s]
            if 0 <= r < ny and 0 <= c < nx:
                force[r * nx + c] = _force(r * nx + c, u, plate, nx, ny, kp, drow, dcol, kw)
    for k in range(n_order):  # restore the work-array invariant
        slipped[order[k]] = False
        active[order[k]] = False
    return plate, steps, n_order, peak, dissipated, work, completed, tide_work


@njit(cache=True)
def _cubic_coefficients(series, k, j):
    # Catmull-Rom cubic a0 + a1 u + a2 u^2 + a3 u^3 of component j on step [k-1, k], u in [0, 1]
    last = series.shape[0] - 1
    p0 = series[max(k - 2, 0), j]
    p1 = series[max(k - 1, 0), j]
    p2 = series[min(k, last), j]
    p3 = series[min(k + 1, last), j]
    return p1, 0.5 * (p2 - p0), 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3), 0.5 * (3.0 * p1 - p0 - 3.0 * p2 + p3)


@njit(cache=True)
def _poly(a0, a1, a2, a3, u):
    return a0 + u * (a1 + u * (a2 + u * a3))


@njit(cache=True)
def _first_positive(a0, a1, a2, a3, u_lo):
    # first u in [u_lo, 1] where the cubic is > 0, or -1. Between consecutive critical points
    # the cubic is monotone, so the first piece that ends above zero holds the crossing,
    # found by bisection.
    points = np.empty(4)
    n = 0
    points[n] = u_lo
    n += 1
    # critical points: a1 + 2 a2 u + 3 a3 u^2 = 0
    qa, qb, qc = 3.0 * a3, 2.0 * a2, a1
    r1 = r2 = -1.0
    if qa != 0.0:
        disc = qb * qb - 4.0 * qa * qc
        if disc >= 0.0:
            sq = np.sqrt(disc)
            r1 = (-qb - sq) / (2.0 * qa)
            r2 = (-qb + sq) / (2.0 * qa)
            if r1 > r2:
                r1, r2 = r2, r1
    elif qb != 0.0:
        r1 = -qc / qb
    for r in (r1, r2):
        if u_lo < r < 1.0:
            points[n] = r
            n += 1
    points[n] = 1.0
    n += 1
    if _poly(a0, a1, a2, a3, points[0]) > 0.0:
        return points[0]
    for i in range(1, n):
        if _poly(a0, a1, a2, a3, points[i]) > 0.0:
            lo, hi = points[i - 1], points[i]
            for __ in range(80):
                mid = 0.5 * (lo + hi)
                if mid <= lo or mid >= hi:
                    break
                if _poly(a0, a1, a2, a3, mid) > 0.0:
                    hi = mid
                else:
                    lo = mid
            return hi
    return -1.0


@njit(cache=True)
def _first_failure(gap, x, y, coulomb, step_s, rate, t0, t_end):
    # earliest time >= t0 (and <= t_end) at which some candidate c has
    #     gap[c] + rate (t - t0) + c0(t) + cx(t) x[c] + cy(t) y[c] > 0,
    # with the tide interpolated by Catmull-Rom cubics between tabulated steps (so short
    # excursions above the strength inside a step are not missed). Returns (t, c) or (-1, -1).
    k = int(np.floor(t0 / step_s)) + 1
    u_lo = t0 / step_s - (k - 1)
    while k < coulomb.shape[0] and (k - 1) * step_s <= t_end:
        b0 = _cubic_coefficients(coulomb, k, 0)
        bx = _cubic_coefficients(coulomb, k, 1)
        by = _cubic_coefficients(coulomb, k, 2)
        base = rate * ((k - 1) * step_s - t0)
        best_u = 2.0
        best_c = -1
        for c in range(gap.shape[0]):
            a0 = gap[c] + base + b0[0] + bx[0] * x[c] + by[0] * y[c]
            a1 = rate * step_s + b0[1] + bx[1] * x[c] + by[1] * y[c]
            a2 = b0[2] + bx[2] * x[c] + by[2] * y[c]
            a3 = b0[3] + bx[3] * x[c] + by[3] * y[c]
            u = _first_positive(a0, a1, a2, a3, u_lo)
            if u >= 0.0 and u < best_u:
                best_u = u
                best_c = c
        if best_c >= 0:
            t = (k - 1 + best_u) * step_s
            if t > t_end:
                return -1.0, -1
            return max(t, t0), best_c
        k += 1
        u_lo = 0.0
    return -1.0, -1


class Model:
    """State and dynamics of the spring-block system for a SimulationConfig.

    Attributes
    ----------
    u, v : ndarray (n_blocks,)
        Displacement (m, along x) and velocity (m/s) of each block, in the
        grid index order (see grid).
    plate_m : float
        Plate position x_p (m).
    time_s : float
        Simulated time (s).
    static_force : ndarray (n_blocks,)
        Strength F_s,i of each block (N).
    current_force : ndarray (n_blocks,)
        Spring force on each block (N), maintained incrementally (equal to force()).
    tide_drive, tide_normal : ndarray (n_blocks,)
        Tidal force along the slip and tidal normal force (tension positive) on each
        block (N), frozen during an event (zero without tides; see set_tide).
    """

    max_event_steps = 10_000_000

    def __init__(self, config):
        self.config = config.validate()
        self.rng = np.random.default_rng(config.seed)
        n = config.n_blocks
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        self.kc = config.k_coupling_n_m
        self.kp = config.k_loading_n_m
        self.fd = config.dynamic_force_n
        self.mass = config.block_mass_kg
        self.vp = config.plate_velocity_m_s
        self.dt = config.slip_time_step_s
        # coupled offsets and their stiffnesses k_c (L / r)^p
        self.drow, self.dcol, weight = grid.interaction_stencil(config)
        self.kw = self.kc * weight
        # k_p + the coupling stiffnesses of each block (fewer partners at the edges)
        self.ksum = self.kp + self._coupling_degree()
        # sliding friction: v_w <= 0 encodes Coulomb friction for the kernel
        self.vw = config.weakening_velocity_m_s if config.weakening_velocity_m_s is not None else 0.0
        self.keep = 1.0 - config.onset_drop
        self.plate_m = self.fd / self.kp  # every block starts at the dynamic friction level
        self.time_s = 0.0
        # random strengths: one global stream (default), or a counter per block so that the
        # k-th redraw of block i is always the same number (rng_mode='per-block')
        self.redraw_counts = np.zeros(n, dtype=np.uint64) if config.rng_mode == 'per-block' else None
        self.static_force = self._draw_strengths(n)
        self.current_force = self.force()
        # tides: per-block forcing during events, block coordinates relative to the grid centre
        self.mus = config.friction_coefficient
        self.mud = config.dynamic_friction_coefficient
        self.tide_drive = np.zeros(n)
        self.tide_normal = np.zeros(n)
        positions = grid.block_positions(config)
        self.x_centre = positions[:, 0] - positions[:, 0].mean()
        self.y_centre = positions[:, 1] - positions[:, 1].mean()
        # work arrays reused by every event (see _slip_kernel)
        self._work = dict(active=np.zeros(n, dtype=np.bool_), slipped=np.zeros(n, dtype=np.bool_),
                          act=np.empty(n, dtype=np.int64), moved=np.empty(n, dtype=np.int64),
                          acc=np.zeros(n), disp=np.zeros(n), order=np.empty(n, dtype=np.int64),
                          u_start=np.zeros(n))

    def _draw_strengths(self, size, blocks=None):
        # F_s,i = F_d + (F_s - F_d) x uniform[1 - h, 1 + h]
        cfg = self.config
        excess = cfg.static_force_n - cfg.dynamic_force_n
        h = cfg.heterogeneity
        if self.redraw_counts is None:
            uniform = self.rng.random(size)
        else:
            blocks = np.arange(size) if blocks is None else np.asarray(blocks)
            uniform = counter_uniform(cfg.seed, blocks, self.redraw_counts[blocks])
            self.redraw_counts[blocks] += np.uint64(1)
        return self.fd + excess * (1 + h * (2 * uniform - 1))

    # ---- forces and energy --------------------------------------------------
    def _pairs(self, g):
        # for every stencil offset: stiffness, and slices of the blocks that have a
        # partner at that offset (target) and of the partners (source)
        ny, nx = g.shape
        for dr, dc, k in zip(self.drow, self.dcol, self.kw):
            target = (slice(max(0, -dr), ny - max(0, dr)), slice(max(0, -dc), nx - max(0, dc)))
            source = (slice(max(0, dr), ny + min(0, dr)), slice(max(0, dc), nx + min(0, dc)))
            yield k, target, source

    def _coupling_degree(self):
        # sum of the coupling stiffnesses of each block's partners inside the grid
        g = np.zeros((self.config.n_blocks_y, self.config.n_blocks_x))
        for k, target, source in self._pairs(g):
            g[target] += k
        return g.ravel()

    def coupling_force(self, u):
        """Force from the coupling springs on each block (N), for displacements u."""
        g = grid.as_grid(u, self.config)
        f = np.zeros_like(g, dtype=float)
        for k, target, source in self._pairs(g):
            f[target] += k * (g[source] - g[target])
        return f.ravel()

    def sliding_friction(self, velocity, index):
        """Friction (N) on block `index` while it slides at `velocity` (m/s)."""
        return _friction(float(velocity), self.static_force[index], self.fd, self.vw, self.keep)

    def force(self):
        """Total spring force on each block (N), recomputed from scratch: coupling + loading."""
        return self.coupling_force(self.u) + self.kp * (self.plate_m - self.u)

    def elastic_energy(self):
        """Energy stored in the loading and coupling springs (J). Every coupled pair
        appears at two opposite offsets, hence 1/4 rather than 1/2."""
        g = grid.as_grid(self.u, self.config)
        coupling = sum(k * np.sum((g[source] - g[target]) ** 2) for k, target, source in self._pairs(g))
        return 0.5 * self.kp * np.sum((self.plate_m - self.u) ** 2) + 0.25 * coupling

    # ---- loading --------------------------------------------------------------
    def time_to_failure(self):
        """(seconds, index): time until the first block reaches its strength while all
        are stuck (forces grow at k_p v_p), and that block (0 s if one already has)."""
        idx = int(np.argmax(self.current_force - self.static_force))
        wait = (self.static_force[idx] - self.current_force[idx]) / (self.kp * self.vp)
        return max(float(wait), 0.0), idx

    def time_to_failure_tidal(self, hourly, end_s):
        """(seconds, index): time until the first block fails under plate loading plus the
        tide (HourlyTide), or (inf, -1) if none fails by end_s.

        The tide is tabulated every tide_step_hours and interpolated by cubics in between;
        inside each step the first crossing of every candidate is found exactly (critical
        points of its cubic, then bisection), so no failure is missed. Only candidate blocks
        are checked: those whose gap is within twice the largest spatial variation of the
        tide of the largest gap, since no other block can overtake them before the next event."""
        gap = self.current_force - self.static_force
        if getattr(self, '_tide_spread_source', None) is not hourly:
            self._tide_spread = hourly.max_spatial_variation(np.abs(self.x_centre).max(), np.abs(self.y_centre).max())
            self._tide_spread_source = hourly
        top = gap.max()
        cand = np.flatnonzero(gap >= top - 2.0 * self._tide_spread - 1e-9 * abs(top))
        g, x, y = gap[cand], self.x_centre[cand], self.y_centre[cand]
        t0 = self.time_s
        t_fail, c = _first_failure(g, x, y, hourly.coulomb, hourly.step_s, self.kp * self.vp, t0, end_s)
        if c < 0:
            return np.inf, -1
        return t_fail - t0, int(cand[c])

    @property
    def tidal_event(self):
        """Whether the next event feels a tide (any non-zero tidal force)."""
        return bool(self.tide_drive.any() or self.tide_normal.any())

    def set_tide(self, coefficients_drive, coefficients_normal):
        """Freezes the tidal forcing for the next event, from fitted coefficients (c0, cx, cy)."""
        cd, cn = coefficients_drive, coefficients_normal
        self.tide_drive = cd[0] + cd[1] * self.x_centre + cd[2] * self.y_centre
        self.tide_normal = cn[0] + cn[1] * self.x_centre + cn[2] * self.y_centre

    def clear_tide(self):
        self.tide_drive = np.zeros(self.config.n_blocks)
        self.tide_normal = np.zeros(self.config.n_blocks)

    def load(self, seconds):
        """Advance the plate while every block stays stuck."""
        self.plate_m += self.vp * seconds
        self.time_s += seconds
        self.current_force += self.kp * self.vp * seconds

    # ---- slip -------------------------------------------------------------------
    def slip(self, hypocentre):
        """Integrate one event starting with the failure of block `hypocentre`."""
        w = self._work
        plate, steps, n_slipped, peak, dissipated, work, completed, tide_work = _slip_kernel(
            self.u, self.v, self.current_force, self.static_force, self.fd, self.vw, self.keep, self.kp,
            self.mass, self.config.n_blocks_x, self.config.n_blocks_y, self.drow, self.dcol, self.kw, self.ksum,
            self.plate_m, self.vp, self.dt, hypocentre, self.max_event_steps, w['active'], w['slipped'],
            w['act'], w['moved'], w['acc'], w['disp'], w['order'], w['u_start'],
            self.tide_drive, self.tide_normal, self.mus, self.mud, self.tidal_event)
        if not completed:
            raise RuntimeError('event did not stop within %d steps' % self.max_event_steps)
        self.time_s += steps * self.dt
        self.plate_m = plate
        slipped = w['order'][:n_slipped].copy()
        u_before = w['u_start'][slipped]
        if self.config.redraw_thresholds:
            # annealed disorder: slipped blocks heal with a new strength from the same
            # distribution, but a block that has just stopped can hold the load it
            # carries, so the new strength is never below its current force (with tides: the
            # force including the tidal drive and unclamping it feels when it stops)
            held = self.current_force[slipped] + self.tide_drive[slipped] + self.mus * self.tide_normal[slipped]
            self.static_force[slipped] = np.maximum(self._draw_strengths(n_slipped, slipped), held)
        return SlipResult(slipped=slipped, u_before=u_before, slip=self.u[slipped] - u_before,
                          duration_s=steps * self.dt, peak_velocity_m_s=peak, dissipated_j=dissipated,
                          plate_work_j=work, steps=steps, tide_work_j=tide_work)
