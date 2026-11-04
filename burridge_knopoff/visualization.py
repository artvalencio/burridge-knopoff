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
import matplotlib.quiver
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from . import analysis, grid
from .config import SECONDS_PER_YEAR

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
    """Cumulative number of events with magnitude >= M (log scale) after the
    spin-up, with the magnitude of completeness and the maximum-likelihood
    Gutenberg-Richter fit (analysis.gutenberg_richter)."""
    stats = analysis.gutenberg_richter(result)
    spinup = stats['spinup_years']
    m = np.sort([e.magnitude for e in result.events if e.start_years >= spinup])
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(m):
        ax.plot(m, len(m) - np.arange(len(m)), '.-', label='events')  # N(>= m[k]) = n - k
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'no events', ha='center', va='center', transform=ax.transAxes)
    if stats['b'] is not None:
        est, mc = stats['b'], stats['mc']
        line = np.linspace(mc, m.max(), 50)
        ax.plot(line, np.count_nonzero(m >= mc) * 10 ** (-est.b * (line - mc)), '--',
                label='b = %.2f ± %.2f (Mw ≥ Mc)' % (est.b, est.sigma))
        ax.axvline(mc, color='grey', linestyle=':', label='Mc = %.1f' % mc)
        ax.legend()
    ax.set_xlabel('Moment magnitude Mw')
    ax.set_ylabel('Number of events with magnitude >= Mw')
    ax.set_title('Magnitude-frequency distribution (%d events after %g years of spin-up)' % (len(m), spinup))
    ax.grid(True, which='both', alpha=0.3)
    if path is None:
        return fig
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def preseismic_animation(window, path, tide_removal='model', fps=30, writer=None, percentile=100.0,
                         fade_days=7.0, dpi=80):
    """Animation of a preseismic.PreseismicWindow: one frame per recorded time.

    One map: the gravity change since the first frame; with tides, the gravity residuals
    (tides removed by `tide_removal`, 'model' or 'fit'). Below it, the series at the sensor
    nearest the hypocentre with a moving cursor (and, with tides, the tidal Coulomb stress
    at the hypocentre). The colour scale is fixed for the whole window (the `percentile` of
    |values| over all frames). The target hypocentre is a star; earlier events appear as
    circles when they happen, sized by magnitude and fading over fade_days.

    Writes MP4 with ffmpeg (required for path ending in .mp4; writer overrides, e.g. 'pillow'
    for a GIF in tests). With path=None returns (fig, anim, update)."""
    cfg = window.config
    tides = window.tidal_gravity_abs is not None
    values = (window.residual(tide_removal) if tides else window.gravity) * 1e8      # microGal
    scale = float(np.percentile(np.abs(values), percentile)) if values.size else 0.0
    norm = Normalize(vmin=-scale if scale > 0 else -1e-30, vmax=scale if scale > 0 else 1e-30)
    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0])
    x, y = _sensor_mesh(cfg)
    mesh = ax.pcolormesh(x, y, grid.as_grid(values[0], cfg, sensors=True), cmap=CMAP, norm=norm, shading='nearest',
                         animated=True)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')
    label = ('Gravity residual since t0, tides removed (%s) (µGal)' % tide_removal if tides
             else 'Gravity change since t0 (µGal)')
    fig.colorbar(mesh, ax=ax, label=label, shrink=0.85)
    hx, hy = (c / 1000 for c in window.event.hypocentre_xy(cfg))
    ax.plot(hx, hy, '*', color='yellow', markersize=18, markeredgecolor='black', zorder=5)
    ev_t = np.array([e.start_years * SECONDS_PER_YEAR for __, e in window.events])
    ev_xy = np.array([e.hypocentre_xy(cfg) for __, e in window.events]).reshape(-1, 2) / 1000
    ev_size = np.array([8 + 6 * max(e.magnitude - 4, 0) ** 2 for __, e in window.events])
    dots = ax.scatter(ev_xy[:, 0] if len(ev_xy) else [], ev_xy[:, 1] if len(ev_xy) else [], s=ev_size if len(ev_size) else [],
                      facecolors='none', edgecolors='black', linewidths=1.0, animated=True, zorder=4)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', va='bottom', fontsize=11, animated=True)
    strip = fig.add_subplot(gs[1])
    days = window.time_to_event_s / 86400.0
    s = window.nearest_sensor()
    strip.plot(days, values[:, s], color='tab:blue', lw=0.8)
    strip.set_ylabel('sensor %d (µGal)' % s, color='tab:blue')
    strip.set_xlabel('time to the event (days)')
    strip.set_xlim(days[0], days[-1] if days[-1] > days[0] else days[0] + 1)
    if tides and window.tidal_coulomb_pa is not None:
        twin = strip.twinx()
        twin.plot(days, window.tidal_coulomb_pa / 1e3, color='tab:orange', lw=0.5, alpha=0.7)
        twin.set_ylabel('tidal Coulomb (kPa)', color='tab:orange')
    cursor = strip.axvline(days[0], color='black', lw=1.0, animated=True)
    fade_s = fade_days * 86400.0

    def update(k):
        t = window.frame_times_s[k]
        mesh.set_array(grid.as_grid(values[k], cfg, sensors=True).ravel())
        if len(ev_t):
            age = t - ev_t
            alpha = np.where((age >= 0) & (age < fade_s), 1.0 - age / fade_s, 0.0)
            colours = np.zeros((len(ev_t), 4))
            colours[:, 3] = alpha
            dots.set_edgecolors(colours)
        lead = window.frame_times_s[-1] - t
        n_so_far = int(np.count_nonzero(ev_t <= t)) if len(ev_t) else 0
        title.set_text('Mw %.2f event: t - t_event = -%d d %02d h   (%d earlier events in the window)'
                       % (window.event.magnitude, lead // 86400, (lead % 86400) // 3600, n_so_far))
        cursor.set_xdata([days[k], days[k]])
        return mesh, dots, title, cursor

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=len(window.frame_times_s), interval=1000 / fps, blit=True)
    if path is None:
        return fig, anim, update
    if writer is None:
        if not _ffmpeg_available():
            plt.close(fig)
            raise RuntimeError('the pre-seismic animation needs FFmpeg (%d frames; a GIF would be too large)'
                               % len(window.frame_times_s))
        writer = 'ffmpeg'
    anim.save(path, writer=writer, fps=fps, dpi=dpi)
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


# ---- displacement during an event ------------------------------------------------------------
def _rupture_panels(rupture, n_points):
    from . import rupture as rp
    cfg = rupture.config
    points, xs, ys = rp.map_points(cfg, n=n_points)
    field = rp.surface_frames(rupture, points)
    L = cfg.block_spacing_m
    cd = np.cos(np.radians(cfg.fault_dip_deg))
    h = rupture.event.hypocentre
    row, col = divmod(h, cfg.n_blocks_x)
    hypo_fault = (col * L / 1e3, row * L / 1e3)
    epicentre = ((-0.5 * L + (col + 0.5) * L * cd) / 1e3, row * L / 1e3)
    outline_x = np.array([-0.5, -0.5 + cfg.n_blocks_x * cd, -0.5 + cfg.n_blocks_x * cd, -0.5, -0.5]) * L / 1e3
    outline_y = np.array([-0.5, -0.5, cfg.n_blocks_y - 0.5, cfg.n_blocks_y - 0.5, -0.5]) * L / 1e3
    return field, xs / 1e3, ys / 1e3, hypo_fault, epicentre, (outline_x, outline_y)


def _slip_arrows(cfg, slip_grid, n_arrows=14):
    # hanging-wall motion on the fault at a sparse sample of blocks: up dip (-x) by sin(rake),
    # along strike (+y) by cos(rake)
    ny, nx = slip_grid.shape
    step = max(1, int(np.ceil(max(nx, ny) / n_arrows)))
    rr, cc = np.mgrid[step // 2:ny:step, step // 2:nx:step]
    s = slip_grid[rr, cc]
    rake = np.radians(cfg.fault_rake_deg)
    L = cfg.block_spacing_m / 1e3
    return cc * L, rr * L, -np.sin(rake) * s, np.cos(rake) * s


def displacement_map(rupture, path=None, frame=-1, n_points=30, field=None):
    """Three maps of an event at one frame (default: its end): slip on the fault with the motion of
    the hanging wall; the displacement vector field at the surface (arrows: horizontal, colour:
    vertical); the gravity change at the surface (dislocation model). Returns the path written,
    or (fig, axes) with path=None."""
    from . import rupture as rp
    cfg = rupture.config
    panels = _rupture_panels(rupture, n_points) if field is None else field
    field, xs, ys, hypo_fault, epicentre, outline = panels
    f = field[frame]
    slip = rp.slip_grid(rupture, frame)
    L = cfg.block_spacing_m / 1e3
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    ax = axes[0]
    extent = (-0.5 * L, (cfg.n_blocks_x - 0.5) * L, -0.5 * L, (cfg.n_blocks_y - 0.5) * L)
    im = ax.imshow(slip, origin='lower', extent=extent, cmap='magma_r', vmin=0,
                   vmax=max(float(rupture.final_slip.max()), 1e-9))
    fig.colorbar(im, ax=ax, label='slip (m)')
    qx, qy, qu, qv = _slip_arrows(cfg, slip)
    final_max = max(float(rupture.final_slip.max()), 1e-12)
    ax.quiver(qx, qy, qu, qv, color='tab:cyan', angles='xy', width=0.004, scale=final_max * qx.shape[1] / 1.2)
    ax.plot(*hypo_fault, marker='*', ms=16, mfc='yellow', mec='k')
    ax.set(xlabel='x, down dip on the fault (km)', ylabel='y, along strike (km)', title='Slip on the fault')
    ax = axes[1]
    dz = f[:, 2].reshape(len(ys), len(xs))
    im = ax.pcolormesh(xs, ys, dz, cmap=CMAP, norm=symmetric_norm(field[-1][:, 2]), shading='auto')
    fig.colorbar(im, ax=ax, label='vertical displacement (m)')
    scale = max(float(np.hypot(field[-1][:, 0], field[-1][:, 1]).max()), 1e-12)
    ax.quiver(xs[None, :].repeat(len(ys), 0), ys[:, None].repeat(len(xs), 1), f[:, 0].reshape(dz.shape),
              f[:, 1].reshape(dz.shape), angles='xy', scale=scale * len(xs) / 1.5, width=0.003)
    ax.plot(*outline, color='0.4', lw=1)
    ax.plot(*epicentre, marker='*', ms=16, mfc='yellow', mec='k')
    ax.set(xlabel='X, horizontal down-dip (km)', ylabel='Y, along strike (km)',
           title='Surface displacement (max horizontal %.2g m)' % scale)
    ax = axes[2]
    dg = f[:, 3].reshape(dz.shape) * 1e8
    im = ax.pcolormesh(xs, ys, dg, cmap=CMAP, norm=symmetric_norm(field[-1][:, 3] * 1e8), shading='auto')
    fig.colorbar(im, ax=ax, label='gravity change (uGal)')
    ax.plot(*outline, color='0.4', lw=1)
    ax.plot(*epicentre, marker='*', ms=16, mfc='yellow', mec='k')
    ax.set(xlabel='X, horizontal down-dip (km)', ylabel='Y, along strike (km)', title='Surface gravity change')
    for a in axes:
        a.set_aspect('equal')
    ev = rupture.event
    fig.suptitle('Mw %.2f event at %.2f years: t = %.1f s after the start'
                 % (ev.magnitude, ev.start_years, rupture.frame_times_s[frame]))
    fig.tight_layout()
    if path is None:
        return fig, axes
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def rupture_animation(rupture, path, n_points=30, fps=12, writer=None):
    """Animation of displacement_map over the frames of the rupture (MP4 with ffmpeg, else GIF)."""
    panels = _rupture_panels(rupture, n_points)
    fig, axes = displacement_map(rupture, None, frame=0, field=panels)
    field, xs, ys = panels[0], panels[1], panels[2]
    from . import rupture as rp
    image = axes[0].images[0]
    meshes = [axes[1].collections[0], axes[2].collections[0]]
    quiver = [c for c in axes[1].collections if isinstance(c, matplotlib.quiver.Quiver)][0]
    slip_quiver = [c for c in axes[0].collections if isinstance(c, matplotlib.quiver.Quiver)][0]
    title = fig._suptitle
    ev = rupture.event

    def update(k):
        slip = rp.slip_grid(rupture, k)
        image.set_data(slip)
        slip_quiver.set_UVC(*_slip_arrows(rupture.config, slip)[2:])
        f = field[k]
        meshes[0].set_array(f[:, 2].reshape(len(ys), len(xs)).ravel())
        meshes[1].set_array((f[:, 3] * 1e8).reshape(len(ys), len(xs)).ravel())
        quiver.set_UVC(f[:, 0], f[:, 1])
        title.set_text('Mw %.2f event at %.2f years: t = %.1f s after the start'
                       % (ev.magnitude, ev.start_years, rupture.frame_times_s[k]))
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(rupture.frame_times_s), interval=1000 / fps, blit=False)
    if writer is None:
        writer = 'ffmpeg' if _ffmpeg_available() else 'pillow'
    if writer == 'pillow' and not path.endswith('.gif'):
        path = os.path.splitext(path)[0] + '.gif'
    anim.save(path, writer=writer, fps=fps, dpi=80)
    plt.close(fig)
    return path
