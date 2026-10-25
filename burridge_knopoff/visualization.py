"""
Plots and animation of a SimulationResult.

All maps share a symmetric colour scale around zero (blue: potential
decrease, red: increase), which also works when every value is zero or of
one sign. Hypocentres are drawn at their (x, y) positions in km. Each
function saves to `path` and returns the path written; with path=None it
returns the matplotlib objects instead (for further editing or tests).
"""
import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from . import grid

CMAP = 'RdBu_r'
LABEL = 'Gravitational potential change (J/kg)'


def symmetric_norm(values):
    """Colour normalisation from -max|v| to +max|v| (a tiny range if all are zero)."""
    m = float(np.max(np.abs(values))) if np.size(values) else 0.0
    m = m if m > 0 else 1e-30
    return Normalize(vmin=-m, vmax=m)


def frame_indices(n_samples, max_frames):
    """At most max_frames sample indices, evenly spread, always including the first and last."""
    if n_samples <= max_frames:
        return np.arange(n_samples)
    return np.unique(np.round(np.linspace(0, n_samples - 1, max_frames)).astype(int))


def _sensor_mesh(config):
    pos = grid.sensor_positions(config)
    return (grid.as_grid(pos[:, 0], config, sensors=True) / 1000,
            grid.as_grid(pos[:, 1], config, sensors=True) / 1000)


def _map(ax, result, values, norm, title):
    x, y = _sensor_mesh(result.config)
    mesh = ax.pcolormesh(x, y, grid.as_grid(values, result.config, sensors=True), cmap=CMAP, norm=norm,
                         shading='nearest')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(title)
    ax.set_aspect('equal')
    return mesh


def _ffmpeg_available():
    return animation.writers.is_available('ffmpeg')


def potential_animation(result, path, max_frames=300, fps=20, writer=None):
    """Animation of the potential change at the sensors; stars mark hypocentres of
    the events since the previous frame. Uses ffmpeg (MP4) when available,
    otherwise Pillow (GIF, the extension of `path` is changed)."""
    frames = frame_indices(len(result.times_years), max_frames)
    pots = np.asarray(result.potentials)
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = symmetric_norm(pots[frames])
    mesh = _map(ax, result, pots[frames[0]], norm, '')
    fig.colorbar(mesh, ax=ax, label=LABEL)
    starts = np.array([e.start_years for e in result.events])
    xy = np.array([e.hypocentre_xy(result.config) for e in result.events]).reshape(-1, 2) / 1000
    stars, = ax.plot([], [], '*', color='yellow', markersize=14, markeredgecolor='black')

    def update(k):
        i = frames[k]
        t = result.times_years[i]
        t_prev = result.times_years[frames[k - 1]] if k > 0 else -np.inf
        mesh.set_array(grid.as_grid(pots[i], result.config, sensors=True).ravel())
        recent = (starts > t_prev) & (starts <= t)
        stars.set_data(xy[recent, 0], xy[recent, 1])
        # no blitting: the title (time) is redrawn with the rest of the frame
        ax.set_title('Potential change, t = %.2f years' % t)
        return mesh, stars

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    if path is None:
        return fig, anim, update
    if writer is None:
        writer = 'ffmpeg' if _ffmpeg_available() else 'pillow'
    if writer == 'pillow' and not path.endswith('.gif'):
        path = os.path.splitext(path)[0] + '.gif'
    anim.save(path, writer=writer, fps=fps, dpi=80)
    plt.close(fig)
    return path


def event_comparison(result, event_index, path):
    """Potential change just before and just after an event, and their difference,
    with the hypocentre marked."""
    event = result.events[event_index]
    times = result.times_years
    before = max(int(np.searchsorted(times, event.start_years, side='left')) - 1, 0)
    after = min(int(np.searchsorted(times, event.start_years, side='right')), len(times) - 1)
    pots = np.asarray(result.potentials)
    panels = [(pots[before], 'Before (t = %.2f years)' % times[before]),
              (pots[after], 'After (t = %.2f years)' % times[after]),
              (pots[after] - pots[before], 'Change due to the event')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x_km, y_km = (c / 1000 for c in event.hypocentre_xy(result.config))
    for ax, (values, title) in zip(axes, panels):
        mesh = _map(ax, result, values, symmetric_norm(values), title)
        fig.colorbar(mesh, ax=ax, label=LABEL, shrink=0.8)
        ax.plot(x_km, y_km, '*', color='yellow', markersize=16, markeredgecolor='black')
    fig.suptitle('Event at t = %.3f years, Mw %.2f, %d blocks slipped'
                 % (event.start_years, event.magnitude, event.n_slipped))
    fig.tight_layout()
    if path is None:
        return fig
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def magnitude_frequency(result, path):
    """Cumulative number of events with magnitude >= M (log scale), the
    Gutenberg-Richter representation."""
    m = np.sort([e.magnitude for e in result.events])
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(m):
        ax.plot(m, len(m) - np.arange(len(m)), '.-')  # N(>= m[k]) = n - k
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'no events', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Moment magnitude Mw')
    ax.set_ylabel('Number of events with magnitude >= Mw')
    ax.set_title('Magnitude-frequency distribution (%d events)' % len(m))
    ax.grid(True, which='both', alpha=0.3)
    if path is None:
        return fig
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def create_all(result, directory, max_frames=300, writer=None, n_event_plots=3):
    """Animation, comparison plots of the n_event_plots largest events and the
    magnitude-frequency plot, saved in `directory`; returns the paths."""
    os.makedirs(directory, exist_ok=True)
    paths = [potential_animation(result, os.path.join(directory, 'gravity_potential_animation.mp4'),
                                 max_frames=max_frames, writer=writer)]
    largest = np.argsort([-e.magnitude for e in result.events], kind='stable')[:n_event_plots]
    for rank, k in enumerate(largest, start=1):
        paths.append(event_comparison(result, int(k), os.path.join(directory, 'event_%d_comparison.png' % rank)))
    paths.append(magnitude_frequency(result, os.path.join(directory, 'magnitude_frequency.png')))
    return paths
