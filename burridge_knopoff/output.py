"""
Saving and loading simulation results.

save_results writes to a directory:

    results.npz              everything, binary (reload with load_results)
    events.csv               the event catalogue (events.CATALOGUE_COLUMNS)
    sensor_potentials.csv    time_years, sensor_0 ... sensor_{n-1}   (J/kg)
    sensor_positions.csv     sensor, x_m, y_m, z_m
    block_positions.csv      block, row, col, x_m, y_m               (reference grid)
    block_displacements.csv  time_years, block_0 ... (m)    only with block_csv=True

The CSV files are "wide" and normalised: coordinates are written once, and
each time row holds only values. Large tables are written in chunks of
CHUNK_ROWS rows, so no per-row Python objects are built.
"""
import csv
import dataclasses
import json
import os

import numpy as np

from . import events as ev
from . import grid
from .config import SimulationConfig
from .simulation import SimulationResult

CHUNK_ROWS = 1000


def _write_table(path, header, rows, fmt='%.17g'):
    """Writes header + rows (2-D array-like), CHUNK_ROWS rows at a time."""
    with open(path, 'w', newline='') as fh:
        fh.write(','.join(header) + '\n')
        for start in range(0, len(rows), CHUNK_ROWS):
            np.savetxt(fh, np.asarray(rows[start:start + CHUNK_ROWS], dtype=float), delimiter=',', fmt=fmt)


def save_results(result, directory, csv_files=True, block_csv=False):
    """Writes results.npz and (unless csv_files=False) the CSV files to `directory`."""
    os.makedirs(directory, exist_ok=True)
    cfg = result.config
    np.savez(os.path.join(directory, 'results.npz'), config=json.dumps(dataclasses.asdict(cfg)),
             times_years=result.times_years, potentials=np.asarray(result.potentials),
             snapshot_years=result.snapshot_years, displacements=np.asarray(result.displacements),
             catalogue=result.catalogue, final_displacement=result.final_displacement,
             dissipated_energy_j=result.dissipated_energy_j)
    if not csv_files:
        return
    table = result.catalogue
    with open(os.path.join(directory, 'events.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(ev.CATALOGUE_COLUMNS)
        writer.writerows(table.tolist())
    pots = np.asarray(result.potentials)
    _write_table(os.path.join(directory, 'sensor_potentials.csv'),
                 ['time_years'] + ['sensor_%d' % k for k in range(cfg.n_sensors)],
                 _with_time(result.times_years, pots))
    sensors = grid.sensor_positions(cfg)
    _write_table(os.path.join(directory, 'sensor_positions.csv'), ['sensor', 'x_m', 'y_m', 'z_m'],
                 np.column_stack([np.arange(cfg.n_sensors), sensors]), fmt='%.10g')
    rows, cols = np.divmod(np.arange(cfg.n_blocks), cfg.n_blocks_x)
    blocks = grid.block_positions(cfg)
    _write_table(os.path.join(directory, 'block_positions.csv'), ['block', 'row', 'col', 'x_m', 'y_m'],
                 np.column_stack([np.arange(cfg.n_blocks), rows, cols, blocks[:, :2]]), fmt='%.10g')
    if block_csv:
        _write_table(os.path.join(directory, 'block_displacements.csv'),
                     ['time_years'] + ['block_%d' % k for k in range(cfg.n_blocks)],
                     _with_time(result.snapshot_years, result.displacements), fmt='%.9g')


class _with_time:
    """Rows [time, values...] built lazily, chunk by chunk (no full copy)."""

    def __init__(self, times, values):
        self.times, self.values = times, values

    def __len__(self):
        return len(self.times)

    def __getitem__(self, chunk):
        return np.column_stack([self.times[chunk], np.asarray(self.values[chunk], dtype=float)])


def load_results(directory):
    """Reads results.npz written by save_results back into a SimulationResult."""
    with np.load(os.path.join(directory, 'results.npz')) as data:
        cfg = SimulationConfig(**json.loads(str(data['config'])))
        table = data['catalogue']
        fields = [f.name for f in dataclasses.fields(ev.Event)]
        events = [ev.Event(**{name: row[name].item() for name in fields}) for row in table]
        return SimulationResult(config=cfg, times_years=data['times_years'], potentials=data['potentials'],
                                snapshot_years=data['snapshot_years'], displacements=data['displacements'],
                                events=events, final_displacement=data['final_displacement'],
                                dissipated_energy_j=float(data['dissipated_energy_j']))
