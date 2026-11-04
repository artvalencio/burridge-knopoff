"""
Pre-seismic gravity: the sensor field in the year (window) before an event,
frame by frame.

A run samples the sensors every sample_interval_years, too coarsely for
this, and the event of interest is chosen after the run. replay() therefore
runs the same configuration again (results are deterministic for a seed) up
to the target event(s), recording the sensors every frame_hours from
t0 = t_event - window_years to t_event, just before the event: its own
co-seismic step is not included. Several targets are recorded in one pass.

Gravity is the change of the downward gravity (m/s^2). With tides, the
sensors also feel the tidal gravity; PreseismicWindow keeps the tectonic
part (from slip) and the tidal part separately, and residual() gives the
tide-removed series either exactly ('model': the tide the simulation
applied) or as one would with real data ('fit': least-squares fit of the
main tidal constituents plus a constant and a linear drift, per sensor, over
the whole window).
"""
import dataclasses
import math

import numpy as np

from . import grid
from .config import SECONDS_PER_YEAR
from .simulation import Runner

# speeds of the main tidal constituents, degrees per hour
CONSTITUENTS = dict(M2=28.9841042, S2=30.0, N2=28.4397295, K2=30.0821373, K1=15.0410686, O1=13.9430356,
                    P1=14.9589314, Q1=13.3986609, Mf=1.0980331, Mm=0.5443747)


@dataclasses.dataclass
class PreseismicWindow:
    """Sensor frames before one event.

    frame_times_s : (n_frames,) times (s), the last one at the event
    gravity_abs, potential_abs : (n_frames, n_sensors) tectonic gravity change (m/s^2) and
        potential change (J/kg) since the start of the run
    tidal_gravity_abs : (n_frames, n_sensors) tidal gravity (m/s^2), or None without tides
    tidal_coulomb_pa : (n_frames,) tidal Coulomb stress at the hypocentre, or None
    events : [(index, Event)] of the events in [t0, t_event)
    """
    config: object
    event_index: int
    event: object
    frame_times_s: np.ndarray
    gravity_abs: np.ndarray
    potential_abs: np.ndarray
    tidal_gravity_abs: np.ndarray = None
    tidal_coulomb_pa: np.ndarray = None
    events: list = dataclasses.field(default_factory=list)

    @property
    def gravity(self):
        """Tectonic gravity change relative to the first frame (m/s^2)."""
        return self.gravity_abs - self.gravity_abs[0]

    @property
    def potential(self):
        return self.potential_abs - self.potential_abs[0]

    @property
    def tidal_gravity(self):
        return None if self.tidal_gravity_abs is None else self.tidal_gravity_abs - self.tidal_gravity_abs[0]

    @property
    def raw(self):
        """What the sensors record relative to the first frame: tectonic + tidal (m/s^2)."""
        return self.gravity if self.tidal_gravity is None else self.gravity + self.tidal_gravity

    def residual(self, method='model'):
        """The raw change with the tides removed (m/s^2), relative to the first frame."""
        if self.tidal_gravity is None:
            return self.gravity
        if method == 'model':
            return self.raw - self.tidal_gravity
        if method == 'fit':
            removed = self.raw - fitted_tide(self.frame_times_s, self.raw)
            return removed - removed[0]
        raise ValueError("tide removal must be 'model' or 'fit', got %r" % method)

    @property
    def time_to_event_s(self):
        return self.frame_times_s - self.frame_times_s[-1]

    def nearest_sensor(self):
        """Index of the sensor nearest the hypocentre (horizontally)."""
        x, y = self.event.hypocentre_xy(self.config)
        pos = grid.sensor_positions(self.config)
        return int(np.argmin(np.hypot(pos[:, 0] - x, pos[:, 1] - y)))

    def save(self, path):
        np.savez(path, event_index=self.event_index, frame_times_s=self.frame_times_s,
                 gravity_abs=self.gravity_abs, potential_abs=self.potential_abs,
                 tidal_gravity_abs=np.zeros(0) if self.tidal_gravity_abs is None else self.tidal_gravity_abs,
                 tidal_coulomb_pa=np.zeros(0) if self.tidal_coulomb_pa is None else self.tidal_coulomb_pa,
                 window_event_indices=np.array([k for k, __ in self.events], dtype=np.int64))


def fitted_tide(times_s, series, constituents=CONSTITUENTS):
    """The tidal part of each column of series: least-squares fit of the constituents plus a
    constant and a linear drift, returning only the harmonic (tidal) part."""
    hours = (np.asarray(times_s, dtype=float) - times_s[0]) / 3600.0
    harmonics = []
    for speed in constituents.values():
        w = math.radians(speed) * hours
        harmonics += [np.cos(w), np.sin(w)]
    harmonics = np.column_stack(harmonics)
    span = max(hours[-1], 1.0)
    design = np.column_stack([np.ones_like(hours), hours / span, harmonics])
    coef = np.linalg.lstsq(design, np.asarray(series, dtype=float), rcond=None)[0]
    return harmonics @ coef[2:]


def largest_event(result, spinup_years=None):
    """Index of the largest event after the spin-up (the default target)."""
    spinup = result.config.spinup_years if spinup_years is None else spinup_years
    mags = np.array([e.magnitude if e.start_years >= spinup else -np.inf for e in result.events])
    if not len(mags) or not np.isfinite(mags.max()):
        raise ValueError('no event after the spin-up of %g years' % spinup)
    return int(np.argmax(mags))


def replay(result, event_indices, window_years=1.0, frame_hours=1.0):
    """PreseismicWindow(s) of result.events[k] for k in event_indices (an int gives one window).

    The configuration of `result` is run again up to the last target; for each target the
    sensors are recorded every frame_hours from max(0, t_event - window_years) to t_event.
    Raises RuntimeError if the replay does not reproduce a target event."""
    single = np.isscalar(event_indices)
    targets = sorted({int(k) for k in np.atleast_1d(event_indices)})
    config = result.config
    runner = Runner(config, samples=False)
    step_s = frame_hours * 3600.0
    sensors_rel = grid.sensor_positions(config)[:, :2] - np.array(_centre(config))
    windows = {}
    for k in targets:
        event = result.events[k]
        t_ev = event.start_years * SECONDS_PER_YEAR
        n = int(math.floor(min(window_years * SECONDS_PER_YEAR, t_ev) / step_s + 1e-9)) + 1
        times = t_ev - (n - 1 - np.arange(n)) * step_s
        g = np.zeros((n, config.n_sensors))
        phi = np.zeros((n, config.n_sensors))

        def fill(start, stop, g=g, phi=phi, times=times):
            g[start:stop] = runner.gravity_at(times[start:stop])
            phi[start:stop] = runner.delta

        runner.add_recorder(times, fill)
        windows[k] = (event, times, g, phi)
    for k in targets:
        event, times, g, phi = windows[k]
        while len(runner.events) < k:
            if runner.step(runner.end_s) is None:
                raise RuntimeError('replay ended before event %d' % k)
        if config.preslip:
            # the blocks creep before the event: run it, the frames being filled on the way
            again = runner.step(runner.end_s)
            if again is None or again.hypocentre != event.hypocentre or                     abs(again.start_years - event.start_years) > 1e-9 * max(event.start_years, 1.0):
                raise RuntimeError('replay diverged before event %d' % k)
            continue
        wait, hypocentre = runner.next_failure(runner.end_s)
        t_next = runner.model.time_s + wait
        if hypocentre != event.hypocentre or abs(t_next - times[-1]) > 1e-6 * max(times[-1], 1.0):
            raise RuntimeError('replay diverged before event %d' % k)
        # only the plate moves before the event: fill the frames without touching the model
        # (loading it here would split one load in two and could change rounding)
        runner.record_until(t_next)
    out = []
    for k in targets:
        event, times, g, phi = windows[k]
        tidal_g = coulomb = None
        if runner.hourly is not None:
            from .tides import evaluate
            grav = runner.hourly.at(times)[3]
            tidal_g = evaluate(grav, sensors_rel[:, 0], sensors_rel[:, 1])
            x_c, y_c = runner.model.x_centre[event.hypocentre], runner.model.y_centre[event.hypocentre]
            coul = runner.hourly.at(times)[2]
            coulomb = (coul[:, 0] + coul[:, 1] * x_c + coul[:, 2] * y_c) / config.block_area_m2
        in_window = [(j, e) for j, e in enumerate(runner.events[:k])
                     if e.start_years * SECONDS_PER_YEAR >= times[0]]
        out.append(PreseismicWindow(config=config, event_index=k, event=event, frame_times_s=times, gravity_abs=g,
                                    potential_abs=phi, tidal_gravity_abs=tidal_g, tidal_coulomb_pa=coulomb,
                                    events=in_window))
    return out[0] if single else out


def _centre(config):
    from .tides import grid_centre
    return grid_centre(config)
