"""
The event-driven simulation loop.

Between events every block is stuck, only the plate moves, and nothing the
sensors see changes: the potential changes are piecewise constant in time.
The loop therefore alternates

1. load: advance the plate exactly to the next failure (Model.time_to_failure),
   filling every sample and snapshot that falls before it with the current
   (unchanged) state;
2. slip: integrate the event (Model.slip), update the sensor potentials and
   gravity from the blocks that slipped only (SensorArray.update_blocks) and
   record the event with its co-seismic gravity step.

An event that starts before the end of the simulation is always completed
and recorded, even if it ends slightly after it. Samples that fall inside an
event (which lasts seconds) show the state after the event.

With tides (config.tides) the loading is no longer linear in time: step 1
advances every tide_step_hours and locates the failure inside the step
(Model.time_to_failure_tidal), and the tide at the start of each event is
frozen for its duration (Model.set_tide).

Runner holds the state of a simulation and advances it (advance, step);
run() is the usual one-shot use. Recorders fill arrays at given times with
the current state, so replays can record extra series (e.g. hourly frames)
on the same code path.
"""
import copy
import dataclasses
import math
import os

import numpy as np

from . import events as ev
from . import grid
from .config import SECONDS_PER_YEAR
from .gravity import SensorArray
from .model import Model


@dataclasses.dataclass
class SimulationResult:
    """Results of run().

    times_years : (n_samples,) sample times
    potentials : (n_samples, n_sensors) potential change at each sensor (J/kg)
    snapshot_years : (n_snapshots,) times of the stored displacement fields
    displacements : (n_snapshots, n_blocks) float32 block displacements (m)
    events : list of events.Event, in time order
    final_displacement : (n_blocks,) displacements at the end (m, float64)
    dissipated_energy_j : total energy dissipated by friction
    gravity : (n_samples, n_sensors) change of the downward gravity at each sensor (m/s^2)
    gravity_map_events : indices of the events (Mw >= gravity_map_min_magnitude) whose
        co-seismic steps are stored in gravity_map_potential and gravity_map_gravity
        (n, n_sensors; J/kg and m/s^2)
    tide_times_s, tide_coulomb_pa : with tides, the tidal Coulomb stress at the grid centre
        (driving stress + mu_s normal stress), every tide_step_hours
    regional_moment : (n_events, n_regions) float32 moment released by each event in each
        region (N m; regions of n_regions_x x n_regions_y, row-major like the blocks)
    elastic_energy : (n_samples,) energy in the springs at the sample times (J), with
        record_energy
    """
    config: object
    times_years: np.ndarray
    potentials: np.ndarray
    snapshot_years: np.ndarray
    displacements: np.ndarray
    events: list
    final_displacement: np.ndarray
    dissipated_energy_j: float
    gravity: np.ndarray = None
    gravity_map_events: np.ndarray = None
    gravity_map_potential: np.ndarray = None
    gravity_map_gravity: np.ndarray = None
    tide_times_s: np.ndarray = None
    tide_coulomb_pa: np.ndarray = None
    regional_moment: np.ndarray = None
    elastic_energy: np.ndarray = None

    @property
    def catalogue(self):
        return ev.catalogue(self.events, self.config)

    @property
    def block_positions(self):
        return grid.block_positions(self.config)

    @property
    def sensor_positions(self):
        return grid.sensor_positions(self.config)


def _schedule(total_years, interval_years):
    # 0, dt, 2 dt, ... up to and including the end (tolerant to rounding)
    return np.arange(int(np.floor(total_years / interval_years + 1e-9)) + 1) * interval_years


def _array(store_dir, name, shape, dtype):
    if store_dir is None:
        return np.zeros(shape, dtype=dtype)
    os.makedirs(store_dir, exist_ok=True)
    return np.lib.format.open_memmap(os.path.join(store_dir, name + '.npy'), mode='w+', dtype=dtype, shape=shape)


class Recorder:
    """Calls fill(start, stop) whenever the entries times_s[start:stop] become due; the
    state seen is the one just after the last event before each time."""

    def __init__(self, times_s, fill):
        self.times_s = np.asarray(times_s, dtype=float)
        self.fill = fill
        self.next = 0

    def record_until(self, t_s):
        stop = int(np.searchsorted(self.times_s, t_s * (1 + 1e-12), side='right'))
        if stop > self.next:
            self.fill(self.next, stop)
            self.next = stop


class Runner:
    """A simulation in progress: the model, the sensor state, the catalogue so far.

    advance(until_s) runs every event that starts by until_s and loads the plate up to
    it; record_until fills the recorders. Attributes: model, sensors, delta (potential
    change at the sensors), delta_g (gravity change), events, dissipated_j, hourly (the
    tide table, or None).
    """

    def __init__(self, config, store_dir=None, samples=True):
        self.config = config.validate()
        self.model = Model(config)
        self.sensors = SensorArray(config)
        self.end_s = config.simulation_years * SECONDS_PER_YEAR
        n_s = config.n_sensors
        self.delta = np.zeros(n_s)
        self.delta_g = np.zeros(n_s)
        self._step_phi = np.zeros(n_s)
        self._step_g = np.zeros(n_s)
        self.events = []
        self.dissipated_j = 0.0
        self.map_events, self.map_phi, self.map_g = [], [], []
        self.regional = []
        self.region_of_block = region_of_blocks(config)
        self.n_regions = config.n_regions_x * config.n_regions_y
        self.recorders = []
        self.hourly = self._phase_times = self._phase = None
        if config.tides:
            from .tides import HourlyTide
            self.hourly = HourlyTide(config, self.end_s)
            self._phase_times, phase = self.hourly.semidiurnal_phase()
            self._phase = np.unwrap(phase)
        if samples:
            self.times = _schedule(config.simulation_years, config.sample_interval_years)
            self.snap_times = _schedule(config.simulation_years, config.snapshot_interval_years)
            self.potentials = _array(store_dir, 'potentials', (len(self.times), n_s), np.float64)
            self.gravity = _array(store_dir, 'gravity', (len(self.times), n_s), np.float64)
            self.displacements = _array(store_dir, 'displacements', (len(self.snap_times), config.n_blocks),
                                        np.float32)

            def fill_samples(start, stop):
                self.potentials[start:stop] = self.delta
                self.gravity[start:stop] = self.delta_g

            def fill_snapshots(start, stop):
                self.displacements[start:stop] = self.model.u.astype(np.float32)

            self.recorders += [Recorder(self.times * SECONDS_PER_YEAR, fill_samples),
                               Recorder(self.snap_times * SECONDS_PER_YEAR, fill_snapshots)]
            self.energy = None
            if config.record_energy:
                self.energy = np.zeros(len(self.times))
                sample_s = self.times * SECONDS_PER_YEAR

                def fill_energy(start, stop):
                    # blocks are stuck between events: only the plate moves, so the loading-spring
                    # energy at each sample follows from the plate position at that time
                    m = self.model
                    plate = m.plate_m - m.vp * (m.time_s - sample_s[start:stop])
                    total = m.elastic_energy()
                    loading_now = 0.5 * m.kp * np.sum((m.plate_m - m.u) ** 2)
                    n, su, su2 = len(m.u), m.u.sum(), np.sum(m.u ** 2)
                    loading = 0.5 * m.kp * (n * plate ** 2 - 2 * plate * su + su2)
                    self.energy[start:stop] = total - loading_now + loading

                self.recorders.append(Recorder(sample_s, fill_energy))

    # ---- recording -----------------------------------------------------------------
    def add_recorder(self, times_s, fill):
        recorder = Recorder(times_s, fill)
        recorder.record_until(self.model.time_s)
        self.recorders.append(recorder)
        return recorder

    def record_until(self, t_s):
        for recorder in self.recorders:
            recorder.record_until(t_s)

    # ---- dynamics ------------------------------------------------------------------
    def next_failure(self, limit_s):
        """(wait, hypocentre) of the next failure, or a wait beyond limit_s."""
        if self.hourly is None:
            return self.model.time_to_failure()
        return self.model.time_to_failure_tidal(self.hourly, min(limit_s, self.end_s))

    def tide_at(self, t_s, x_c, y_c):
        """Tidal stresses at (x_c, y_c) relative to the grid centre at time t_s: driving
        stress, normal stress, Coulomb stress (Pa), its rate (Pa/h) and the semidiurnal
        phase (rad)."""
        from .tides import evaluate
        area = self.config.block_area_m2
        drive, normal, coulomb, __ = self.hourly.at(np.array([t_s - 60.0, t_s, t_s + 60.0]))
        values = [float(evaluate(c, x_c, y_c)[0]) for c in (drive[1], normal[1], coulomb[1])]
        rate = (float(evaluate(coulomb[2], x_c, y_c)[0]) - float(evaluate(coulomb[0], x_c, y_c)[0])) / 120.0 * 3600.0
        phase = math.remainder(float(np.interp(t_s, self._phase_times, self._phase)), 2 * math.pi)
        return values[0] / area, values[1] / area, values[2] / area, rate / area, phase

    def step(self, limit_s):
        """Runs the next event if it starts by limit_s and returns it; otherwise loads the
        plate to limit_s and returns None."""
        model, config = self.model, self.config
        wait, hypocentre = self.next_failure(limit_s)
        if model.time_s + wait > limit_s:
            model.load(limit_s - model.time_s)
            self.record_until(limit_s)
            return None
        model.load(wait)
        self.record_until(model.time_s)
        start_s = model.time_s
        tide = {}
        if self.hourly is not None:
            drive, normal, __, __ = self.hourly.at(start_s)
            model.set_tide(drive[0], normal[0])
            values = self.tide_at(start_s, model.x_centre[hypocentre], model.y_centre[hypocentre])
            tide = dict(zip(('tide_shear_pa', 'tide_normal_pa', 'tide_coulomb_pa', 'tide_coulomb_rate_pa_h',
                             'tide_phase_rad'), values))
        result = model.slip(hypocentre)
        # only the slipped blocks moved: the co-seismic step at the sensors comes from them alone
        step_phi, step_g = self._step_phi, self._step_g
        step_phi[:] = 0.0
        step_g[:] = 0.0
        self.sensors.update_blocks(step_phi, result.slipped, result.u_before, model.u[result.slipped], step_g)
        self.delta += step_phi
        self.delta_g += step_g
        self.dissipated_j += result.dissipated_j
        moment = ev.seismic_moment(result.slip, config)
        magnitude = ev.moment_magnitude(moment)
        dg_ugal = np.abs(step_g) * 1e8
        event = ev.Event(
            start_years=start_s / SECONDS_PER_YEAR, duration_s=result.duration_s, hypocentre=hypocentre,
            n_slipped=len(result.slipped), total_slip_m=float(result.slip.sum()),
            max_slip_m=float(result.slip.max()), peak_velocity_m_s=result.peak_velocity_m_s,
            moment_n_m=moment, magnitude=magnitude, dg_max_abs_ugal=float(dg_ugal.max()),
            dg_rms_ugal=float(np.sqrt(np.mean(dg_ugal ** 2))), dg_max_sensor=int(np.argmax(dg_ugal)),
            dphi_max_abs=float(np.abs(step_phi).max()), **tide)
        self.events.append(event)
        self.regional.append((config.shear_modulus_pa * config.block_area_m2 * np.bincount(
            self.region_of_block[result.slipped], weights=result.slip, minlength=self.n_regions)).astype(np.float32))
        if magnitude >= config.gravity_map_min_magnitude:
            self.map_events.append(len(self.events) - 1)
            self.map_phi.append(step_phi.copy())
            self.map_g.append(step_g.copy())
        self.last_slip = result
        return event

    def advance(self, until_s, progress=None):
        """Runs every event that starts by until_s, then loads the plate to until_s.
        progress: callable receiving the fraction of [0, end] done (about 1000 calls)."""
        next_report = 0.0
        while self.step(until_s) is not None:
            if progress is not None and self.model.time_s >= next_report:
                next_report = self.model.time_s + self.end_s / 1000
                progress(min(self.model.time_s / self.end_s, 1.0))

    # ---- state ---------------------------------------------------------------------
    def snapshot(self):
        """A copy of the dynamical state (model, sensor state, random generator); events and
        recorders are not included."""
        m = self.model
        return dict(u=m.u.copy(), v=m.v.copy(), plate_m=m.plate_m, time_s=m.time_s,
                    static_force=m.static_force.copy(), current_force=m.current_force.copy(),
                    rng=copy.deepcopy(m.rng.bit_generator.state), delta=self.delta.copy(),
                    delta_g=self.delta_g.copy(),
                    rng_counts=None if getattr(m, 'redraw_counts', None) is None else m.redraw_counts.copy())

    def restore(self, state):
        m = self.model
        m.u[:] = state['u']
        m.v[:] = state['v']
        m.plate_m, m.time_s = state['plate_m'], state['time_s']
        m.static_force[:] = state['static_force']
        m.current_force[:] = state['current_force']
        m.rng.bit_generator.state = copy.deepcopy(state['rng'])
        if state.get('rng_counts') is not None:
            m.redraw_counts[:] = state['rng_counts']
        self.delta[:] = state['delta']
        self.delta_g[:] = state['delta_g']

    def result(self):
        if not hasattr(self, 'times'):
            raise ValueError('result() needs a Runner created with samples=True')
        tide_times = tide_coulomb = None
        if self.hourly is not None:
            keep = self.hourly.times <= self.end_s
            tide_times = self.hourly.times[keep]
            tide_coulomb = self.hourly.coulomb[keep, 0] / self.config.block_area_m2
        n_s = self.config.n_sensors
        return SimulationResult(
            config=self.config, times_years=self.times, potentials=self.potentials, snapshot_years=self.snap_times,
            displacements=self.displacements, events=self.events, final_displacement=self.model.u.copy(),
            dissipated_energy_j=self.dissipated_j, gravity=self.gravity,
            gravity_map_events=np.array(self.map_events, dtype=np.int64),
            gravity_map_potential=np.array(self.map_phi).reshape(-1, n_s),
            gravity_map_gravity=np.array(self.map_g).reshape(-1, n_s),
            tide_times_s=tide_times, tide_coulomb_pa=tide_coulomb,
            regional_moment=np.array(self.regional, dtype=np.float32).reshape(-1, self.n_regions),
            elastic_energy=self.energy)


def region_of_blocks(config):
    """Region index of every block: the grid split into n_regions_x x n_regions_y regions
    (row-major, rows along y)."""
    rows, cols = np.divmod(np.arange(config.n_blocks), config.n_blocks_x)
    return ((rows * config.n_regions_y // config.n_blocks_y) * config.n_regions_x
            + cols * config.n_regions_x // config.n_blocks_x).astype(np.int64)


def run(config, progress=None, store_dir=None):
    """Simulate `config.simulation_years` and return a SimulationResult.

    progress: True to show a progress bar (tqdm, in simulated years, on
        stderr), or a callable receiving the completed fraction (0..1) whenever
        at least 0.1% of the simulated time has passed, and at the end. Default (None or False): no progress report.
    store_dir: if given, the potentials, gravity and displacement snapshots are
        written to .npy files there (memory-mapped) instead of being kept in RAM.
    """
    config.validate()
    bar = None
    if progress is False:
        progress = None
    if progress is True:
        from tqdm import tqdm
        bar = tqdm(total=config.simulation_years, unit='yr', desc='Simulating', dynamic_ncols=True,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f} yr '
                              '[{elapsed}<{remaining}]{postfix}')

        def progress(fraction):
            # tqdm throttles redraws itself, so this is cheap even for many events
            bar.set_postfix_str('%d events' % len(runner.events), refresh=False)
            bar.update(fraction * config.simulation_years - bar.n)
    try:
        runner = Runner(config, store_dir=store_dir)
        runner.advance(runner.end_s, progress)
        runner.record_until(np.inf)  # an event completed after the end fills the last samples
        if progress is not None:
            progress(1.0)
        if store_dir is not None:
            for array in (runner.potentials, runner.gravity, runner.displacements):
                array.flush()
        return runner.result()
    finally:
        if bar is not None:
            bar.close()


def _run_seed(args):
    config, seed = args
    return run(dataclasses.replace(config, seed=int(seed)))


def run_ensemble(config, seeds, n_jobs=1):
    """Runs `config` once per seed and returns the results in seed order.

    n_jobs > 1 runs the simulations in that many worker processes (-1: one per
    CPU). Each run depends only on its own seed, so the results do not depend
    on n_jobs. Results are held in memory: use small snapshot rates for big
    ensembles.
    """
    tasks = [(config, seed) for seed in seeds]
    if n_jobs == 1:
        return [_run_seed(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
        return list(pool.map(_run_seed, tasks))
