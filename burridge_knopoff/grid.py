"""
Index conventions and coordinates.

Blocks form an n_blocks_y x n_blocks_x grid on the fault plane z = 0, spaced
block_spacing_m apart and starting at the origin. Every per-block array uses
the same row-major order: block index = row * n_blocks_x + column, with
columns along x (the plate-motion direction) and rows along y. Reshaping a
per-block array to (n_blocks_y, n_blocks_x) gives the grid (as_grid). Sensors
follow the same convention with their own grid size.
"""
import numpy as np


def block_index(row, col, config):
    return row * config.n_blocks_x + col


def row_col(index, config):
    """(row, column) of a block index; row along y, column along x."""
    return divmod(index, config.n_blocks_x)


def neighbours(index, config):
    """Indices of the (up to four) nearest neighbours; edges have fewer (free boundaries)."""
    row, col = row_col(index, config)
    out = []
    if col > 0:
        out.append(index - 1)
    if col < config.n_blocks_x - 1:
        out.append(index + 1)
    if row > 0:
        out.append(index - config.n_blocks_x)
    if row < config.n_blocks_y - 1:
        out.append(index + config.n_blocks_x)
    return out


def interaction_stencil(config):
    """Coupled offsets and their relative stiffness: (drow, dcol, weight) arrays.

    All offsets with 0 < r <= interaction_range_blocks (r in block spacings) are
    coupled with weight r^-interaction_exponent. The four nearest neighbours come
    first, in the order left, right, below, above; the others follow by distance.
    """
    reach = config.interaction_range_blocks
    offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    others = [(dr, dc) for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)
              if (dr, dc) not in offsets and 0 < dr * dr + dc * dc <= reach * reach]
    offsets += sorted(others, key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    drow = np.array([o[0] for o in offsets], dtype=np.int64)
    dcol = np.array([o[1] for o in offsets], dtype=np.int64)
    weight = np.hypot(drow, dcol) ** -float(config.interaction_exponent)
    return drow, dcol, weight


def block_positions(config):
    """Reference positions (x, y, z) of the blocks, shape (n_blocks, 3), in m."""
    rows, cols = np.mgrid[0:config.n_blocks_y, 0:config.n_blocks_x]
    L = config.block_spacing_m
    return np.column_stack([cols.ravel() * L, rows.ravel() * L, np.zeros(config.n_blocks)])


def _cell_centres(n, extent):
    # n points at the centres of n equal cells covering [0, extent]
    return (np.arange(n) + 0.5) * extent / n


def sensor_positions(config):
    """Sensor positions (x, y, z), shape (n_sensors, 3), in m: a grid with the same
    centre and extent as the block grid, at sensor_height_m above it."""
    L = config.block_spacing_m
    xs = _cell_centres(config.n_sensors_x, (config.n_blocks_x - 1) * L)
    ys = _cell_centres(config.n_sensors_y, (config.n_blocks_y - 1) * L)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')  # rows along y, x varies fastest
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(config.n_sensors, config.sensor_height_m)])


def as_grid(values, config, sensors=False):
    """Reshapes per-block (or per-sensor) values to a (rows, columns) grid."""
    shape = (config.n_sensors_y, config.n_sensors_x) if sensors else (config.n_blocks_y, config.n_blocks_x)
    return np.asarray(values).reshape(shape)
