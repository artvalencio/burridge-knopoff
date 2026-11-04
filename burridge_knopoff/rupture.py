"""
Displacement during an event: the rupture replayed frame by frame.

replay() runs the configuration of a result again up to an event (results are
deterministic for a seed) and records the block displacements during it, every few
integration steps (Model.record). A Rupture holds the slip of the blocks that slip at
each frame. surface_frames() turns it into the displacement vector field and the
gravity change at the free surface (Okada, 1985; Okubo, 1992; see dislocation.py) of
the slip reached at each frame: the static field of the slip so far (the elastic waves
that carry it are not modelled), for whichever gravity model the run used.

Large ruptures are summed over coarse patches of 2 x 2, 4 x 4, ... blocks carrying
their mean slip (dislocation.coarse_patches), at most MAX_PATCHES of them.
"""
import dataclasses
import math

import numpy as np
from numba import njit, prange

from . import dislocation, grid
from .simulation import Runner

MAX_PATCHES = 3000


@dataclasses.dataclass
class Rupture:
    """The slip during one event.

    frame_times_s : (n_frames,) time since the start of the event (s); the last frame is the
        end of the event
    slipped : (n_slipped,) indices of the blocks that slip, in order of first failure
    slip : (n_frames, n_slipped) float32 slip of those blocks since the start (m)
    """
    config: object
    event_index: int
    event: object
    frame_times_s: np.ndarray
    slipped: np.ndarray
    slip: np.ndarray

    @property
    def final_slip(self):
        return self.slip[-1]

    def save(self, path):
        np.savez_compressed(path, event_index=self.event_index, frame_times_s=self.frame_times_s,
                            slipped=self.slipped, slip=self.slip)


def replay(result, event_index, n_frames=60):
    """Rupture of result.events[event_index], with about n_frames frames.
    Raises RuntimeError if the replay does not reproduce the event."""
    config = result.config
    k = int(event_index)
    target = result.events[k]
    runner = Runner(config, samples=False)
    while len(runner.events) < k:
        if runner.step(runner.end_s) is None:
            raise RuntimeError('replay ended before event %d' % k)
    dt = config.slip_time_step_s
    steps = max(1, int(round(target.duration_s / dt)))
    every = max(1, int(math.ceil(steps / max(n_frames - 1, 1))))
    frames = np.zeros((steps // every + 2, config.n_blocks), dtype=np.float32)
    runner.model.record = (every, frames)
    event = runner.step(runner.end_s)
    if event is None or event.hypocentre != target.hypocentre or event.n_slipped != target.n_slipped:
        raise RuntimeError('replay diverged before event %d' % k)
    filled = runner.model.recorded_frames
    model = runner.model
    slipped = runner.last_slip.slipped
    u_start = runner.last_slip.u_before
    rows = list(frames[:filled, slipped].astype(np.float64) - u_start)
    times = list(np.arange(filled) * every * dt)
    if times[-1] < event.duration_s:  # the end of the event
        rows.append(model.u[slipped] - u_start)
        times.append(event.duration_s)
    slip = np.maximum(np.array(rows), 0.0).astype(np.float32)
    return Rupture(config=config, event_index=k, event=event, frame_times_s=np.array(times), slipped=slipped,
                   slip=slip)


def map_points(config, n=30, margin=0.15):
    """A regular n x n grid of surface points (X, Y) covering the projection of the fault with
    a margin (fraction of its size) on every side; returns (points (n*n, 2), xs, ys)."""
    L = config.block_spacing_m
    dip = math.radians(config.fault_dip_deg)
    x_lo, x_hi = -0.5 * L, -0.5 * L + config.n_blocks_x * L * math.cos(dip)
    y_lo, y_hi = -0.5 * L, (config.n_blocks_y - 0.5) * L
    mx, my = margin * (x_hi - x_lo), margin * (y_hi - y_lo)
    xs = np.linspace(x_lo - mx, x_hi + mx, n)
    ys = np.linspace(y_lo - my, y_hi + my, n)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()]), xs, ys


@njit(cache=True, parallel=True, error_model='numpy')
def _patch_fields(points, centres, size, top_depth, x_top, sd, cd, cr, sr, m, rho_g, beta, out):
    # out[p, s, :] = (dX, dY, dZ, dg) at point s for unit slip of patch p
    for p in prange(centres.shape[0]):
        for s in range(points.shape[0]):
            r = dislocation._response(points[s, 0], points[s, 1], centres[p, 0], centres[p, 1], size, top_depth,
                                      x_top, sd, cd, cr, sr, m, rho_g, beta)
            for k in range(4):
                out[p, s, k] = r[k]


def coarsening(config, slipped):
    """Smallest patch factor (1, 2, 4, ...) giving at most MAX_PATCHES patches for these blocks."""
    factor = 1
    while len(dislocation.coarse_patches(config, slipped, np.zeros(len(slipped)), factor)[1]) > MAX_PATCHES:
        factor *= 2
    return factor


def surface_frames(rupture, points, factor=None):
    """Surface field of the slip at every frame: array (n_frames, n_points, 4) of dX, dY, dZ (m)
    and the gravity change (m/s^2), for the surface points (X, Y)."""
    config = rupture.config
    factor = coarsening(config, rupture.slipped) if factor is None else factor
    centres, _, size = dislocation.coarse_patches(config, rupture.slipped, np.zeros(len(rupture.slipped)), factor)
    geo = dislocation._geometry(config)
    geo['size'] = size
    green = np.zeros((len(centres), len(points), 4))
    _patch_fields(np.ascontiguousarray(points, dtype=float), np.ascontiguousarray(centres), out=green, **geo)
    out = np.empty((len(rupture.slip), len(points), 4))
    for f, slip in enumerate(rupture.slip):
        mean = dislocation.coarse_patches(config, rupture.slipped, slip, factor)[1]
        out[f] = np.einsum('p,psk->sk', mean, green)
    return out


def slip_grid(rupture, frame=-1):
    """Slip of every block at a frame, as a (rows, columns) grid (m; zero where no slip)."""
    full = np.zeros(rupture.config.n_blocks)
    full[rupture.slipped] = rupture.slip[frame]
    return grid.as_grid(full, rupture.config)
