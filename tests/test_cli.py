import os
import subprocess
import sys

import pytest

from burridge_knopoff import cli, visualization
from burridge_knopoff.config import SimulationConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ['--nx', '4', '--ny', '3', '--sensors', '2', '2', '--years', '15', '--sample-interval', '0.5']


def test_run_writes_data_and_plots(tmp_path, capsys):
    code = cli.main(TINY + ['--output', str(tmp_path), '--max-frames', '3'])
    out = capsys.readouterr().out
    assert code == 0
    for name in ['results.npz', 'events.csv', 'sensor_potentials.csv']:
        assert os.path.exists(tmp_path / 'data' / name)
    plots = os.listdir(tmp_path / 'plots')
    assert 'magnitude_frequency.png' in plots and any(p.startswith('gravity_potential_animation') for p in plots)
    assert '[1/4]' in out and '[4/4]' in out and 'Events:' in out


def test_no_plots_and_streaming(tmp_path):
    assert cli.main(TINY + ['--output', str(tmp_path), '--no-plots', '--stream']) == 0
    assert not os.path.exists(tmp_path / 'plots')
    assert os.path.exists(tmp_path / 'data' / 'potentials.npy')


def test_invalid_parameters_exit_with_status_2(capsys):
    assert cli.main(['--nx', '0', '--no-plots']) == 2
    assert 'n_blocks_x' in capsys.readouterr().err


def test_failures_are_reported_with_status_1(tmp_path, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError('no display')
    monkeypatch.setattr(visualization, 'create_all', broken)
    assert cli.main(TINY + ['--output', str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert 'no display' in captured.err and 'successfully' not in captured.out
    assert os.path.exists(tmp_path / 'data' / 'results.npz')  # data was still saved


def test_summary_without_events(tmp_path, capsys):
    assert cli.main(['--nx', '2', '--ny', '2', '--sensors', '1', '1', '--years', '0.3',
                     '--sample-interval', '0.1', '--output', str(tmp_path), '--no-plots']) == 0
    assert 'Events: 0' in capsys.readouterr().out


def test_defaults_come_from_the_given_config(tmp_path):
    cfg = SimulationConfig(n_blocks_x=3, n_blocks_y=2, n_sensors_x=1, n_sensors_y=1, simulation_years=5.0)
    assert cli.main(['--output', str(tmp_path), '--no-plots'], config=cfg) == 0
    from burridge_knopoff.output import load_results
    assert load_results(str(tmp_path / 'data')).config == cfg


def test_main_script_runs_from_anywhere(tmp_path):
    proc = subprocess.run([sys.executable, os.path.join(REPO, 'main.py'), '--help'], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and '--years' in proc.stdout


def test_summary_reports_the_b_value(tmp_path, capsys):
    args = ['--nx', '15', '--ny', '15', '--sensors', '2', '2', '--years', '60', '--sample-interval', '1',
            '--spinup', '10', '--output', str(tmp_path), '--no-plots']
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert 'b = ' in out and 'spin-up 10' in out


def test_summary_without_events_after_the_spinup(tmp_path, capsys):
    assert cli.main(TINY + ['--spinup', '20', '--output', str(tmp_path), '--no-plots']) == 0
    assert 'too few events after the spin-up' in capsys.readouterr().out
