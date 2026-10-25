"""
The event-driven simulation loop.

Between events every block is stuck, only the plate moves, and nothing the
sensors see changes: the potential changes are piecewise constant in time.
The loop therefore alternates

1. load: advance the plate exactly to the next failure (Model.time_to_failure),
   filling every sample and snapshot that falls before it with the current
   (unchanged) state;
2. slip: integrate the event (Model.slip), update the sensor potentials from
   the blocks that slipped only (SensorArray.update) and record the event.

An event that starts before the end of the simulation is always completed
and recorded, even if it ends slightly after it. Samples that fall inside an
event (which lasts seconds) show the state after the event.
"""
import dataclasses
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
    """
    config: object
    times_years: np.ndarray
    potentials: np.ndarray
    snapshot_years: np.ndarray
    displacements: np.ndarray
    events: list
    final_displacement: np.ndarray
    dissipated_energy_j: float

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


def run(config, progress=None, store_dir=None):
    """Simulate `config.simulation_years` and return a SimulationResult.

    progress: optional callable receiving the completed fraction (0..1) after
        each event and at the end.
    store_dir: if given, the potentials and displacement snapshots are written
        to .npy files there (memory-mapped) instead of being kept in RAM.
    """
    config.validate()
    model = Model(config)
    sensors = SensorArray(config)
    end_s = config.simulation_years * SECONDS_PER_YEAR
    times = _schedule(config.simulation_years, config.sample_interval_years)
    snap_times = _schedule(config.simulation_years, config.snapshot_interval_years)
    potentials = _array(store_dir, 'potentials', (len(times), config.n_sensors), np.float64)
    displacements = _array(store_dir, 'displacements', (len(snap_times), config.n_blocks), np.float32)
    times_s, snap_s = times * SECONDS_PER_YEAR, snap_times * SECONDS_PER_YEAR
    delta = np.zeros(config.n_sensors)  # current potential change at the sensors
    next_sample = next_snap = 0
    event_list = []
    dissipated = 0.0

    def record_until(t_s):
        # every sample/snapshot due by t_s sees the current, unchanged state
        nonlocal next_sample, next_snap
        stop = np.searchsorted(times_s, t_s * (1 + 1e-12), side='right')
        potentials[next_sample:stop] = delta
        next_sample = max(next_sample, stop)
        stop = np.searchsorted(snap_s, t_s * (1 + 1e-12), side='right')
        if stop > next_snap:
            displacements[next_snap:stop] = model.u.astype(np.float32)
            next_snap = stop

    while True:
        wait, hypocentre = model.time_to_failure()
        if model.time_s + wait > end_s:
            model.load(end_s - model.time_s)
            record_until(end_s)
            break
        model.load(wait)
        record_until(model.time_s)
        start_s = model.time_s
        result = model.slip(hypocentre)
        # only the slipped blocks moved: update the sensors from them alone
        sensors.update_blocks(delta, result.slipped, result.u_before, model.u[result.slipped])
        dissipated += result.dissipated_j
        moment = ev.seismic_moment(result.slip, config)
        event_list.append(ev.Event(
            start_years=start_s / SECONDS_PER_YEAR, duration_s=result.duration_s, hypocentre=hypocentre,
            n_slipped=len(result.slipped), total_slip_m=float(result.slip.sum()),
            max_slip_m=float(result.slip.max()), peak_velocity_m_s=result.peak_velocity_m_s,
            moment_n_m=moment, magnitude=ev.moment_magnitude(moment)))
        if progress is not None:
            progress(min(model.time_s / end_s, 1.0))
    record_until(np.inf)  # an event completed after the end fills the last samples
    if progress is not None:
        progress(1.0)
    if store_dir is not None:
        potentials.flush()
        displacements.flush()
    return SimulationResult(config=config, times_years=times, potentials=potentials, snapshot_years=snap_times,
                            displacements=displacements, events=event_list, final_displacement=model.u.copy(),
                            dissipated_energy_j=dissipated)
