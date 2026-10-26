import re

from burridge_knopoff import cli
from burridge_knopoff.config import SimulationConfig
from burridge_knopoff.simulation import run

SMALL = SimulationConfig(n_blocks_x=6, n_blocks_y=5, n_sensors_x=2, n_sensors_y=2, simulation_years=40.0,
                         sample_interval_years=1.0)
ARGS = ['--nx', '6', '--ny', '5', '--sensors', '2', '2', '--years', '40', '--sample-interval', '1', '--no-plots']


def test_run_can_show_a_progress_bar_in_simulated_years(capsys):
    run(SMALL, progress=True)
    err = capsys.readouterr().err
    assert 'Simulating' in err and '100%' in err
    assert re.search(r'40(\.0)?/40(\.0)? *yr', err)


def test_no_progress_bar_by_default_in_the_api(capsys):
    run(SMALL)
    assert capsys.readouterr().err == ''


def test_progress_callbacks_still_work():
    seen = []
    run(SMALL, progress=seen.append)
    assert seen[-1] == 1.0


def test_cli_shows_the_progress_bar(tmp_path, capsys):
    assert cli.main(ARGS + ['--output', str(tmp_path)]) == 0
    assert '100%' in capsys.readouterr().err


def test_cli_progress_bar_can_be_turned_off(tmp_path, capsys):
    assert cli.main(ARGS + ['--output', str(tmp_path), '--no-progress']) == 0
    assert 'Simulating' not in capsys.readouterr().err
