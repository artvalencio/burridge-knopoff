import numpy as np
import pytest

from burridge_knopoff import grid
from burridge_knopoff.config import SimulationConfig


def test_block_index_is_row_major_with_rows_along_y(small_config):
    pos = grid.block_positions(small_config)
    L = small_config.block_spacing_m
    assert pos.shape == (24, 3)
    for idx in range(24):
        row, col = grid.row_col(idx, small_config)
        assert grid.block_index(row, col, small_config) == idx
        np.testing.assert_allclose(pos[idx], [col * L, row * L, 0.0])


def test_neighbours_are_geometric_neighbours_on_non_square_grids(small_config):
    pos = grid.block_positions(small_config)
    L = small_config.block_spacing_m
    for idx in range(small_config.n_blocks):
        expected = {j for j in range(small_config.n_blocks)
                    if j != idx and np.isclose(np.linalg.norm(pos[j] - pos[idx]), L)}
        assert set(grid.neighbours(idx, small_config)) == expected


def test_sensors_are_centred_over_the_blocks(small_config):
    blocks = grid.block_positions(small_config)
    sensors = grid.sensor_positions(small_config)
    assert sensors.shape == (6, 3)
    np.testing.assert_allclose(sensors[:, :2].mean(axis=0), blocks[:, :2].mean(axis=0))
    np.testing.assert_allclose(sensors[:, 2], small_config.sensor_height_m)
    # row-major like the blocks: x varies fastest
    assert sensors[1, 0] > sensors[0, 0] and sensors[1, 1] == sensors[0, 1]


def test_single_sensor_sits_above_the_centre():
    cfg = SimulationConfig(n_blocks_x=5, n_blocks_y=3, n_sensors_x=1, n_sensors_y=1)
    np.testing.assert_allclose(grid.sensor_positions(cfg)[0], [4000.0, 2000.0, 5000.0])


def test_grid_shapes(small_config):
    assert grid.as_grid(np.arange(24.0), small_config).shape == (4, 6)
    assert grid.as_grid(np.arange(6.0), small_config, sensors=True).shape == (2, 3)
