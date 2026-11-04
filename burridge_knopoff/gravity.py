"""
Gravitational potential of the blocks at the sensors.

Each block is a point mass m. Its potential at a sensor is -G m / r, and the
sensors report the change relative to the reference grid,

    dPhi_s = -G m sum_b ( 1/|r_s - r_b - u_b e_x| - 1/|r_s - r_b| ),

where u_b is the displacement of block b along x (the slip direction).

Precision: dPhi (~1e-4 J/kg for metre-scale slip) is tiny next to Phi itself
(~-1e2 J/kg), so it is never computed as a difference of two totals. Each
term is evaluated in a cancellation-free form:

    1/a - 1/b = (b - a) / (a b),   b - a = (b^2 - a^2) / (a + b),
    b^2 - a^2 = (u_new - u_old) (2 d_x - u_new - u_old),

with a = |d - u_new e_x|, b = |d - u_old e_x| and d = r_s - r_b. The same
formula updates the potentials incrementally after an event, touching only
the blocks that slipped (cost: sensors x slipped blocks).

Gravity. A gravimeter measures the downward gravity G m sum_b d_z / r^3 (the
sensors are above the blocks, d_z > 0). Its change when a block moves is
G m d_z (1/a^3 - 1/b^3), evaluated in the same cancellation-free way,

    1/a^3 - 1/b^3 = (b - a)(a^2 + a b + b^2) / (a^3 b^3),

sharing b - a with the potential (gravity_change, update_blocks(..., delta_g)).

With gravity_model 'dislocation' the gravity is instead that of slip on a fault in an
elastic half-space at the free surface (dislocation.py): a fixed matrix of the gravity
change at each sensor per metre of slip of each block (float32, blocks x sensors), and
its row total for the back-slip trend. The potential stays the point-mass one.
"""
import numpy as np
from numba import njit

from . import grid
from .config import G


@njit(cache=True)
def _change_kernel(delta, delta_g, sensors, blocks, indices, u_old, u_new, gm, with_gravity):
    # adds -G m sum_k (1/a - 1/b) to delta and (with_gravity) G m sum_k d_z (1/a^3 - 1/b^3)
    # to delta_g, for blocks indices[k] moving from u_old[k] to u_new[k] (compact arrays
    # aligned with indices; see module docstring)
    for s in range(sensors.shape[0]):
        acc = 0.0
        acc_g = 0.0
        for k in range(indices.shape[0]):
            b = indices[k]
            du = u_new[k] - u_old[k]
            if du == 0.0:
                continue
            dx = sensors[s, 0] - blocks[b, 0]
            dy = sensors[s, 1] - blocks[b, 1]
            dz = sensors[s, 2] - blocks[b, 2]
            rest = dy * dy + dz * dz
            a = np.sqrt((dx - u_new[k]) ** 2 + rest)
            bb = np.sqrt((dx - u_old[k]) ** 2 + rest)
            b_minus_a = du * (2.0 * dx - u_new[k] - u_old[k]) / (a + bb)
            acc += b_minus_a / (a * bb)
            if with_gravity:
                ab = a * bb
                acc_g += dz * b_minus_a * (a * a + ab + bb * bb) / (ab * ab * ab)
        delta[s] -= gm * acc
        if with_gravity:
            delta_g[s] += gm * acc_g


def potential(block_positions, sensor_positions, block_mass):
    """Absolute potential -G m sum_b 1/|r_s - r_b| at each sensor (J/kg)."""
    d = sensor_positions[:, None, :] - block_positions[None, :, :]
    return -G * block_mass * (1.0 / np.linalg.norm(d, axis=2)).sum(axis=1)


def acceleration(block_positions, sensor_positions, block_mass):
    """Gravitational acceleration g = -grad Phi at each sensor, shape (n_sensors, 3), in m/s^2.
    It points towards the masses: g = -G m sum_b (r_s - r_b) / |r_s - r_b|^3."""
    d = sensor_positions[:, None, :] - block_positions[None, :, :]
    r = np.linalg.norm(d, axis=2)
    return -G * block_mass * (d / r[:, :, None] ** 3).sum(axis=1)


class SensorArray:
    """The sensor grid of a configuration and its potential changes.

    Attributes
    ----------
    positions : ndarray (n_sensors, 3)
        Sensor positions (m), see grid.sensor_positions.
    block_positions : ndarray (n_blocks, 3)
        Reference block positions (m).
    block_mass : float
        Mass of each block (kg).
    """

    def __init__(self, config):
        self.config = config
        self.positions = grid.sensor_positions(config)
        self.block_positions = grid.block_positions(config)
        self.block_mass = config.block_mass_kg
        self.green = self.green_total = None
        if config.gravity_model == 'dislocation':
            from .dislocation import gravity_matrix
            self.green, self.green_total = gravity_matrix(config)

    def potential_change(self, u):
        """Potential change (J/kg) at each sensor for block displacements u (m, along x)."""
        u = np.asarray(u, dtype=float)
        moved = np.flatnonzero(u)
        delta = np.zeros(len(self.positions))
        return self.update_blocks(delta, moved, np.zeros(len(moved)), u[moved])

    def update(self, delta, u_old, u_new, indices):
        """Adds to delta (in place) the change due to the blocks `indices` moving
        from u_old to u_new; exact, and only these blocks are visited."""
        indices = np.asarray(indices, dtype=np.int64)
        return self.update_blocks(delta, indices, np.asarray(u_old, dtype=float)[indices],
                                  np.asarray(u_new, dtype=float)[indices])

    def update_blocks(self, delta, indices, u_before, u_after, delta_g=None):
        """Like update, with u_before/u_after given only for the blocks `indices`. If delta_g
        is given, the change of the downward gravity (m/s^2) is added to it as well."""
        indices = np.asarray(indices, dtype=np.int64)
        u_before = np.asarray(u_before, dtype=float)
        u_after = np.asarray(u_after, dtype=float)
        point_gravity = delta_g is not None and self.green is None
        _change_kernel(delta, delta_g if point_gravity else delta, self.positions, self.block_positions,
                       indices, u_before, u_after, G * self.block_mass, point_gravity)
        if delta_g is not None and self.green is not None:
            from .dislocation import accumulate
            accumulate(self.green, indices, u_after - u_before, delta_g)
        return delta

    def gravity_change(self, u):
        """Change of the downward gravity (m/s^2) at each sensor for displacements u (m, along x)."""
        u = np.asarray(u, dtype=float)
        moved = np.flatnonzero(u)
        delta_g = np.zeros(len(self.positions))
        self.update_blocks(np.zeros(len(self.positions)), moved, np.zeros(len(moved)), u[moved], delta_g)
        return delta_g
