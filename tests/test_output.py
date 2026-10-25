import csv
import os

import numpy as np
import pytest

from burridge_knopoff import output
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run


@pytest.fixture(scope='module')
def result():
    return run(SimulationConfig(n_blocks_x=5, n_blocks_y=3, n_sensors_x=2, n_sensors_y=2,
                                simulation_years=20.0, sample_interval_years=0.5, seed=4))


def read_csv(path):
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    return rows[0], np.array(rows[1:], dtype=float) if len(rows) > 1 else np.zeros((0, len(rows[0])))


def test_save_and_load_round_trip(result, tmp_path):
    output.save_results(result, str(tmp_path))
    loaded = output.load_results(str(tmp_path))
    assert loaded.config == result.config
    np.testing.assert_array_equal(loaded.potentials, result.potentials)
    np.testing.assert_array_equal(loaded.displacements, result.displacements)
    np.testing.assert_array_equal(loaded.catalogue, result.catalogue)
    np.testing.assert_array_equal(loaded.final_displacement, result.final_displacement)


def test_csv_files(result, tmp_path):
    output.save_results(result, str(tmp_path))
    header, events = read_csv(tmp_path / 'events.csv')
    assert header[:3] == ['start_years', 'duration_s', 'hypocentre'] and len(events) == len(result.events)
    np.testing.assert_allclose(events[:, header.index('magnitude')], [e.magnitude for e in result.events])
    header, pots = read_csv(tmp_path / 'sensor_potentials.csv')
    assert header == ['time_years'] + ['sensor_%d' % k for k in range(4)]
    np.testing.assert_allclose(pots[:, 0], result.times_years)
    np.testing.assert_allclose(pots[:, 1:], result.potentials, rtol=1e-15)
    header, sensors = read_csv(tmp_path / 'sensor_positions.csv')
    assert header == ['sensor', 'x_m', 'y_m', 'z_m'] and len(sensors) == 4
    header, blocks = read_csv(tmp_path / 'block_positions.csv')
    assert header == ['block', 'row', 'col', 'x_m', 'y_m'] and len(blocks) == 15
    assert not os.path.exists(tmp_path / 'block_displacements.csv')


def test_block_displacement_csv_is_optional_and_written_in_chunks(result, tmp_path, monkeypatch):
    monkeypatch.setattr(output, 'CHUNK_ROWS', 2)
    output.save_results(result, str(tmp_path), block_csv=True)
    header, rows = read_csv(tmp_path / 'block_displacements.csv')
    assert header == ['time_years'] + ['block_%d' % k for k in range(15)]
    np.testing.assert_allclose(rows[:, 0], result.snapshot_years)
    np.testing.assert_allclose(rows[:, 1:], result.displacements, rtol=1e-6)


def test_empty_catalogue(tmp_path):
    res = run(SimulationConfig(n_blocks_x=2, n_blocks_y=2, n_sensors_x=1, n_sensors_y=1, simulation_years=0.5,
                               sample_interval_years=0.1))
    assert res.events == []
    output.save_results(res, str(tmp_path))
    header, rows = read_csv(tmp_path / 'events.csv')
    assert len(rows) == 0 and header[0] == 'start_years'
    assert len(output.load_results(str(tmp_path)).catalogue) == 0
