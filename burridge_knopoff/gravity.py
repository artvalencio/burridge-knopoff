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
"""
import numpy as np
from numba import njit

from . import grid
from .config import G


@njit(cache=True)
def _change_kernel(delta, sensors, blocks, u_old, u_new, indices, gm):
    # adds -G m sum_b (1/a - 1/b) for the listed blocks to delta (see module docstring)
    for s in range(sensors.shape[0]):
        acc = 0.0
        for k in range(indices.shape[0]):
            b = indices[k]
            du = u_new[b] - u_old[b]
            if du == 0.0:
                continue
            dx = sensors[s, 0] - blocks[b, 0]
            dy = sensors[s, 1] - blocks[b, 1]
            dz = sensors[s, 2] - blocks[b, 2]
            rest = dy * dy + dz * dz
            a = np.sqrt((dx - u_new[b]) ** 2 + rest)
            bb = np.sqrt((dx - u_old[b]) ** 2 + rest)
            acc += du * (2.0 * dx - u_new[b] - u_old[b]) / (a + bb) / (a * bb)
        delta[s] -= gm * acc


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

    def potential_change(self, u):
        """Potential change (J/kg) at each sensor for block displacements u (m, along x)."""
        u = np.asarray(u, dtype=float)
        delta = np.zeros(len(self.positions))
        _change_kernel(delta, self.positions, self.block_positions, np.zeros_like(u), u,
                       np.flatnonzero(u), G * self.block_mass)
        return delta

    def update(self, delta, u_old, u_new, indices):
        """Adds to delta (in place) the change due to the blocks `indices` moving
        from u_old to u_new; exact, and only these blocks are visited."""
        _change_kernel(delta, self.positions, self.block_positions, np.asarray(u_old, dtype=float),
                       np.asarray(u_new, dtype=float), np.asarray(indices, dtype=np.int64), G * self.block_mass)
        return delta
