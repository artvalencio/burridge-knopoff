import matplotlib
import pytest

matplotlib.use('Agg')

from burridge_knopoff.config import SimulationConfig


@pytest.fixture
def small_config():
    """A 6 x 4 grid (non-square on purpose) over 30 years."""
    return SimulationConfig(n_blocks_x=6, n_blocks_y=4, n_sensors_x=3, n_sensors_y=2,
                            simulation_years=30.0, sample_interval_years=0.1,
                            snapshot_interval_years=1.0, seed=1)
