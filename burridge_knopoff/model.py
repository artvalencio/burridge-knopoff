"""
The Burridge-Knopoff spring-block model (classic formulation).

Each block i slides along x on a fixed base. Its displacement from the
reference position is u_i and its velocity v_i >= 0. The forces on it are

    coupling:  k_c sum_{neighbours j} (u_j - u_i)      (free boundaries)
    loading:   k_p (x_p - u_i)                          (leaf spring to the plate)
    friction:  stuck: whatever keeps it at rest, up to its strength F_s,i
               slipping: F_d opposing the motion

where x_p(t) = x_p(0) + v_p t is the plate position. The plate starts at
x_p(0) = F_d / k_p, so every block starts at the dynamic friction level.

Event-driven integration
------------------------
While every block is stuck, only the plate moves and each force grows
linearly at the rate k_p v_p: the time until the weakest block reaches its
strength is computed exactly (time_to_failure, load). The dynamics are then
integrated only during slip (slip), and only for the slipping blocks, with
velocity Verlet steps of a fraction of the fastest oscillation period. A
block stops when its velocity returns to zero and the force on it is within
its strength; stuck blocks next to moving ones start slipping when their
force exceeds their strength. Blocks never slip backwards. The event ends
when no block moves. This replaces ~1e9 explicit steps through years of
quiet loading by a few hundred steps per event.

Performance: the spring force on every block is maintained incrementally
(current_force). Loading adds k_p v_p dt to all blocks; after an event only
the slipped blocks and their neighbours are recomputed. The slip kernel
reuses preallocated work arrays, so an event costs O(blocks involved) plus a
few O(n) vector operations, with no per-event allocation of grid-sized arrays.
"""
import dataclasses

import numpy as np
from numba import njit

from . import grid


@dataclasses.dataclass
class SlipResult:
    """Outcome of one event: slipped block indices (in order of first failure),
    their displacements before the event and their slips (m), duration (s), peak
    velocity (m/s), energy dissipated by friction (J), work done by the plate
    during the event (J), integration steps."""
    slipped: np.ndarray
    u_before: np.ndarray
    slip: np.ndarray
    duration_s: float
    peak_velocity_m_s: float
    dissipated_j: float
    plate_work_j: float
    steps: int


@njit(cache=True)
def _force(i, u, plate, nx, ny, kc, kp):
    # coupling to the existing neighbours + loading spring
    row = i // nx
    col = i - row * nx
    f = kp * (plate - u[i])
    if col > 0:
        f += kc * (u[i - 1] - u[i])
    if col < nx - 1:
        f += kc * (u[i + 1] - u[i])
    if row > 0:
        f += kc * (u[i - nx] - u[i])
    if row < ny - 1:
        f += kc * (u[i + nx] - u[i])
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


@njit(cache=True)
def _slip_kernel(u, v, force, static, fd, kc, kp, mass, nx, ny, plate, vp, dt, hypocentre, max_steps,
                 active, slipped, act, moved, acc, order, u_start):
    """Integrates one event in place (u, v, force).

    Work arrays: active, slipped (all False on entry and on return), act, moved,
    acc, u_start (size n). On return order[:n_slipped] lists the slipped blocks in
    order of first failure and u_start[order[k]] their displacements before.
    Returns (plate, steps, n_slipped, peak velocity, dissipated energy, plate work, completed).
    """
    n = u.shape[0]
    plate_start = plate
    n_order, n_act = _activate(hypocentre, u, active, slipped, order, u_start, 0, act, 0)
    sum_u = u.sum()
    peak = 0.0
    dissipated = 0.0
    work = 0.0
    steps = 0
    completed = False
    while steps < max_steps:
        if n_act == 0:
            # quiet: did the (tiny) plate motion during the event push any other block over?
            for i in range(n):
                if _force(i, u, plate, nx, ny, kc, kp) > static[i]:
                    n_order, n_act = _activate(i, u, active, slipped, order, u_start, n_order, act, n_act)
            if n_act == 0:
                completed = True
                break
        # velocity Verlet: a(t), then u(t + dt), then a(t + dt) and v(t + dt)
        for k in range(n_act):
            i = act[k]
            acc[i] = (_force(i, u, plate, nx, ny, kc, kp) - fd) / mass
        sum_before = sum_u
        for k in range(n_act):
            i = act[k]
            du = v[i] * dt + 0.5 * acc[i] * dt * dt
            if du < 0.0:
                du = 0.0  # no backward slip
            u[i] += du
            sum_u += du
            dissipated += fd * du  # friction work: F_d times the distance slid
            moved[k] = i
        # plate work during the step: k_p v_p dt sum_i (x_p - u_i), at mid-step
        work += kp * vp * dt * (n * (plate + 0.5 * vp * dt) - 0.5 * (sum_before + sum_u))
        plate += vp * dt
        n_moved = n_act
        n_keep = 0
        for k in range(n_act):
            i = act[k]
            f_new = _force(i, u, plate, nx, ny, kc, kp)
            v_new = v[i] + 0.5 * (acc[i] + (f_new - fd) / mass) * dt
            if v_new <= 0.0:
                v[i] = 0.0
                if f_new <= static[i]:
                    active[i] = False  # stops: friction can hold it again
                    continue
            else:
                v[i] = v_new
                if v_new > peak:
                    peak = v_new
            act[n_keep] = i
            n_keep += 1
        n_act = n_keep
        # stuck neighbours of the moving blocks may now exceed their strength
        for k in range(n_moved):
            i = moved[k]
            row = i // nx
            col = i - row * nx
            for j in (i - 1 if col > 0 else -1, i + 1 if col < nx - 1 else -1,
                      i - nx if row > 0 else -1, i + nx if row < ny - 1 else -1):
                if j >= 0 and not active[j] and _force(j, u, plate, nx, ny, kc, kp) > static[j]:
                    n_order, n_act = _activate(j, u, active, slipped, order, u_start, n_order, act, n_act)
        steps += 1
    # update the maintained forces: the plate moved for everyone, and the slipped
    # blocks and their neighbours are recomputed exactly
    shift = kp * (plate - plate_start)
    for i in range(n):
        force[i] += shift
    for k in range(n_order):
        i = order[k]
        row = i // nx
        col = i - row * nx
        force[i] = _force(i, u, plate, nx, ny, kc, kp)
        for j in (i - 1 if col > 0 else -1, i + 1 if col < nx - 1 else -1,
                  i - nx if row > 0 else -1, i + nx if row < ny - 1 else -1):
            if j >= 0:
                force[j] = _force(j, u, plate, nx, ny, kc, kp)
    for k in range(n_order):  # restore the work-array invariant
        slipped[order[k]] = False
        active[order[k]] = False
    return plate, steps, n_order, peak, dissipated, work, completed


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
        self.plate_m = self.fd / self.kp  # every block starts at the dynamic friction level
        self.time_s = 0.0
        self.static_force = self._draw_strengths(n)
        self.current_force = self.force()
        # work arrays reused by every event (see _slip_kernel)
        self._work = dict(active=np.zeros(n, dtype=np.bool_), slipped=np.zeros(n, dtype=np.bool_),
                          act=np.empty(n, dtype=np.int64), moved=np.empty(n, dtype=np.int64),
                          acc=np.zeros(n), order=np.empty(n, dtype=np.int64), u_start=np.zeros(n))

    def _draw_strengths(self, size):
        # F_s,i = F_d + (F_s - F_d) x uniform[1 - h, 1 + h]
        cfg = self.config
        excess = cfg.static_force_n - cfg.dynamic_force_n
        h = cfg.heterogeneity
        return self.fd + excess * (1 + h * (2 * self.rng.random(size) - 1))

    # ---- forces and energy --------------------------------------------------
    def coupling_force(self, u):
        """Force from the coupling springs on each block (N), for displacements u."""
        g = grid.as_grid(u, self.config)
        f = np.zeros_like(g, dtype=float)
        dx = self.kc * (g[:, 1:] - g[:, :-1])   # spring between columns c and c+1
        f[:, :-1] += dx
        f[:, 1:] -= dx
        dy = self.kc * (g[1:, :] - g[:-1, :])   # spring between rows r and r+1
        f[:-1, :] += dy
        f[1:, :] -= dy
        return f.ravel()

    def force(self):
        """Total spring force on each block (N), recomputed from scratch: coupling + loading."""
        return self.coupling_force(self.u) + self.kp * (self.plate_m - self.u)

    def elastic_energy(self):
        """Energy stored in the loading and coupling springs (J)."""
        g = grid.as_grid(self.u, self.config)
        return 0.5 * (self.kp * np.sum((self.plate_m - self.u) ** 2)
                      + self.kc * (np.sum(np.diff(g, axis=1) ** 2) + np.sum(np.diff(g, axis=0) ** 2)))

    # ---- loading --------------------------------------------------------------
    def time_to_failure(self):
        """(seconds, index): time until the first block reaches its strength while all
        are stuck (forces grow at k_p v_p), and that block (0 s if one already has)."""
        idx = int(np.argmax(self.current_force - self.static_force))
        wait = (self.static_force[idx] - self.current_force[idx]) / (self.kp * self.vp)
        return max(float(wait), 0.0), idx

    def load(self, seconds):
        """Advance the plate while every block stays stuck."""
        self.plate_m += self.vp * seconds
        self.time_s += seconds
        self.current_force += self.kp * self.vp * seconds

    # ---- slip -------------------------------------------------------------------
    def slip(self, hypocentre):
        """Integrate one event starting with the failure of block `hypocentre`."""
        w = self._work
        plate, steps, n_slipped, peak, dissipated, work, completed = _slip_kernel(
            self.u, self.v, self.current_force, self.static_force, self.fd, self.kc, self.kp, self.mass,
            self.config.n_blocks_x, self.config.n_blocks_y, self.plate_m, self.vp, self.dt,
            hypocentre, self.max_event_steps, w['active'], w['slipped'], w['act'], w['moved'], w['acc'],
            w['order'], w['u_start'])
        if not completed:
            raise RuntimeError('event did not stop within %d steps' % self.max_event_steps)
        self.time_s += steps * self.dt
        self.plate_m = plate
        slipped = w['order'][:n_slipped].copy()
        u_before = w['u_start'][slipped]
        if self.config.redraw_thresholds:
            # annealed disorder: slipped blocks heal with a new strength from the same
            # distribution, but a block that has just stopped can hold the load it
            # carries, so the new strength is never below its current force
            self.static_force[slipped] = np.maximum(self._draw_strengths(n_slipped),
                                                    self.current_force[slipped])
        return SlipResult(slipped=slipped, u_before=u_before, slip=self.u[slipped] - u_before,
                          duration_s=steps * self.dt, peak_velocity_m_s=peak, dissipated_j=dissipated,
                          plate_work_j=work, steps=steps)
