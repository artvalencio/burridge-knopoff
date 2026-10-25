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
"""
import dataclasses

import numpy as np
from numba import njit

from . import grid


@dataclasses.dataclass
class SlipResult:
    """Outcome of one event: slipped block indices (in order of first failure),
    their slips (m), duration (s), peak velocity (m/s), energy dissipated by
    friction (J), work done by the plate during the event (J), integration steps."""
    slipped: np.ndarray
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
def _activate(i, active, slipped, order, n_order, act, n_act):
    active[i] = True
    act[n_act] = i
    if not slipped[i]:
        slipped[i] = True
        order[n_order] = i
        n_order += 1
    return n_order, n_act + 1


@njit(cache=True)
def _slip_kernel(u, v, static, fd, kc, kp, mass, nx, ny, plate, vp, dt, hypocentre, max_steps, order):
    """Integrates one event in place (u, v). Returns
    (plate, steps, n_slipped, peak velocity, dissipated energy, plate work, completed)."""
    n = u.shape[0]
    active = np.zeros(n, dtype=np.bool_)
    slipped = np.zeros(n, dtype=np.bool_)
    act = np.empty(n, dtype=np.int64)       # indices of the slipping blocks
    moved = np.empty(n, dtype=np.int64)     # blocks that moved in the current step
    acc = np.zeros(n)
    n_order, n_act = _activate(hypocentre, active, slipped, order, 0, act, 0)
    sum_u = u.sum()
    peak = 0.0
    dissipated = 0.0
    work = 0.0
    steps = 0
    while steps < max_steps:
        if n_act == 0:
            # quiet: did the (tiny) plate motion during the event push any other block over?
            for i in range(n):
                if _force(i, u, plate, nx, ny, kc, kp) > static[i]:
                    n_order, n_act = _activate(i, active, slipped, order, n_order, act, n_act)
            if n_act == 0:
                return plate, steps, n_order, peak, dissipated, work, True
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
                    n_order, n_act = _activate(j, active, slipped, order, n_order, act, n_act)
        steps += 1
    return plate, steps, n_order, peak, dissipated, work, False


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
    """

    max_event_steps = 10_000_000

    def __init__(self, config):
        self.config = config.validate()
        self.rng = np.random.default_rng(config.seed)
        n = config.n_blocks
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        self.kc = config.coupling_stiffness_n_m
        self.kp = config.loading_stiffness_n_m
        self.fd = config.dynamic_force_n
        self.mass = config.block_mass_kg
        self.vp = config.plate_velocity_m_s
        self.dt = config.slip_time_step_s
        self.plate_m = self.fd / self.kp  # every block starts at the dynamic friction level
        self.time_s = 0.0
        self.static_force = self._draw_strengths(n)

    def _draw_strengths(self, size):
        # F_s,i = F_d + (F_s - F_d) x uniform[1 - h, 1 + h]
        cfg = self.config
        excess = cfg.static_force_n - cfg.dynamic_force_n
        h = cfg.heterogeneity
        return self.fd + excess * (1 + h * (2 * self.rng.random(size) - 1))

    # ---- forces and energy --------------------------------------------------
    def coupling_force(self, u):
        """Force from the coupling springs on each block (N), for displacements u."""
        cfg = self.config
        g = grid.as_grid(u, cfg)
        f = np.zeros_like(g, dtype=float)
        dx = self.kc * (g[:, 1:] - g[:, :-1])   # spring between columns c and c+1
        f[:, :-1] += dx
        f[:, 1:] -= dx
        dy = self.kc * (g[1:, :] - g[:-1, :])   # spring between rows r and r+1
        f[:-1, :] += dy
        f[1:, :] -= dy
        return f.ravel()

    def force(self):
        """Total spring force on each block (N): coupling + loading."""
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
        times = (self.static_force - self.force()) / (self.kp * self.vp)
        idx = int(np.argmin(times))
        return max(float(times[idx]), 0.0), idx

    def load(self, seconds):
        """Advance the plate while every block stays stuck."""
        self.plate_m += self.vp * seconds
        self.time_s += seconds

    # ---- slip -------------------------------------------------------------------
    def slip(self, hypocentre):
        """Integrate one event starting with the failure of block `hypocentre`."""
        u_before = self.u.copy()
        order = np.empty(self.config.n_blocks, dtype=np.int64)
        plate, steps, n_slipped, peak, dissipated, work, completed = _slip_kernel(
            self.u, self.v, self.static_force, self.fd, self.kc, self.kp, self.mass,
            self.config.n_blocks_x, self.config.n_blocks_y, self.plate_m, self.vp, self.dt,
            hypocentre, self.max_event_steps, order)
        if not completed:
            raise RuntimeError('event did not stop within %d steps' % self.max_event_steps)
        self.time_s += steps * self.dt
        self.plate_m = plate
        slipped = order[:n_slipped].copy()
        if self.config.redraw_thresholds:
            # annealed disorder: slipped blocks heal with a new strength from the same
            # distribution, but a block that has just stopped can hold the load it
            # carries, so the new strength is never below its current force
            current = self.force()[slipped]
            self.static_force[slipped] = np.maximum(self._draw_strengths(n_slipped), current)
        return SlipResult(slipped=slipped, slip=self.u[slipped] - u_before[slipped], duration_s=steps * self.dt,
                          peak_velocity_m_s=peak, dissipated_j=dissipated, plate_work_j=work, steps=steps)
