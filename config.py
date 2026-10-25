"""
Configuration of the Burridge-Knopoff earthquake simulation.

All parameters are module-level constants read by the other modules when
their objects are created. Units are SI unless the comment says otherwise;
note the exceptions: SENSOR_HEIGHT is in km, SIMULATION_TIME, DT_SAMPLE and
PLATE_VELOCITY use years.

The values are inspired by the Japan Trench (plate rate ~9 cm/year) but are
not a calibrated model of it: with these masses, springs and friction
coefficients, about 147 m of spring stretch is needed to overcome static
friction (about 1600 years of plate motion), so events are not expected in
the default 100 years.
"""

import numpy as np

# Grid dimensions
N_BLOCKS_X = 100  # Number of blocks in x-direction
N_BLOCKS_Y = 100  # Number of blocks in y-direction
N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y  # Total blocks

# Sensor array
N_SENSORS_X = 10  # Sensors in x-direction
N_SENSORS_Y = 10  # Sensors in y-direction
N_SENSORS = N_SENSORS_X * N_SENSORS_Y  # Total sensors
SENSOR_HEIGHT = 5.0  # km above fault plane

# Simulation time parameters
SIMULATION_TIME = 100.0  # years (total simulation time)
DT_SAMPLE = 0.01  # years (data sampling interval)
T_INITIAL_EQUILIBRATION = 10.0  # years (initial equilibration period; not used yet)

# Physical parameters (SI units)
BLOCK_MASS = 1e12  # kg (mass of each block, ~crustal segment)
BLOCK_SPACING = 2000.0  # meters (spacing between blocks, 2 km)
# Coupling springs between nearest neighbours, both in the fault plane:
# "horizontal" couples neighbours along the force kernel's first grid index i,
# "vertical" along its second index j (see compute_forces_numba).
SPRING_K_HORIZONTAL = 1e10  # N/m
SPRING_K_VERTICAL = 1e10  # N/m

# Friction parameters (rate-and-state friction)
MU_STATIC = 0.6  # Static friction coefficient
MU_DYNAMIC = 0.4  # Dynamic friction coefficient
V_CHARACTERISTIC = 1e-6  # m/s (characteristic velocity for friction)
A_FRICTION = 0.01  # Rate-and-state parameter a
B_FRICTION = 0.015  # Rate-and-state parameter b (b > a for instability)
DC = 0.01  # m (characteristic slip distance)

# Plate motion
PLATE_VELOCITY = 0.09  # m/year (Pacific plate subduction rate ~9 cm/year)
PLATE_VELOCITY_SI = PLATE_VELOCITY / (365.25 * 24 * 3600)  # m/s

# Gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Linear damping force -DAMPING * velocity. Note: with BLOCK_MASS = 1e12 kg the
# decay time m/DAMPING is ~1e7 s (months), so this damping is negligible.
DAMPING = 1e5  # kg/s

# Numerical solver parameters
RTOL = 1e-6  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance
MAX_STEP = 0.01 * 365.25 * 24 * 3600  # Maximum time step in seconds (0.01 year)

# Event detection thresholds. Events are detected from the velocities at the
# sampling times only (every DT_SAMPLE), so measured durations are multiples of
# the sampling interval (~3.65 days by default).
VELOCITY_THRESHOLD = 1e-3  # m/s (threshold for earthquake detection)
MIN_EVENT_DURATION = 1.0  # seconds (minimum duration to count as event)

# Output settings
OUTPUT_DIR = 'output'
CSV_DIR = 'output/csv'
ANIMATION_DIR = 'output/animations'

# Derived parameters
BLOCK_AREA = BLOCK_SPACING ** 2  # m^2
NORMAL_FORCE = BLOCK_MASS * 9.81  # N (normal force per block: its weight)

# Convert time units
SECONDS_PER_YEAR = 365.25 * 24 * 3600

def get_sensor_positions():
    """
    Fixed positions of the gravity sensors.

    Sensors form an N_SENSORS_X x N_SENSORS_Y grid at height SENSOR_HEIGHT
    (converted from km to m) above the fault plane z = 0. Like the blocks,
    they are ordered row by row: index = row * N_SENSORS_X + column, rows
    along y.

    Note: the sensors span BLOCK_SPACING/2 .. N*BLOCK_SPACING - BLOCK_SPACING/2,
    while the blocks span 0 .. (N-1)*BLOCK_SPACING, so the sensor array is
    shifted by half a spacing relative to the blocks.

    Returns
    -------
    positions : ndarray, shape (N_SENSORS, 3)
        (x, y, z) of each sensor, in m.
    """
    x_range = N_BLOCKS_X * BLOCK_SPACING
    y_range = N_BLOCKS_Y * BLOCK_SPACING
    
    # Create grid of sensor positions
    x_sensors = np.linspace(BLOCK_SPACING/2, x_range - BLOCK_SPACING/2, N_SENSORS_X)
    y_sensors = np.linspace(BLOCK_SPACING/2, y_range - BLOCK_SPACING/2, N_SENSORS_Y)
    
    xx, yy = np.meshgrid(x_sensors, y_sensors)
    
    # All sensors at height h above the fault (z = 0 for blocks)
    z_sensors = np.full_like(xx, SENSOR_HEIGHT * 1000)  # Convert km to m
    
    positions = np.stack([xx.flatten(), yy.flatten(), z_sensors.flatten()], axis=1)
    
    return positions

def get_block_positions():
    """
    Reference (initial) positions of the blocks on the fault plane z = 0.

    Blocks are spaced BLOCK_SPACING apart, starting at the origin.

    Index convention: np.meshgrid(x, y) produces arrays of shape
    (N_BLOCKS_Y, N_BLOCKS_X), so after flattening the block index is
    row * N_BLOCKS_X + column, with rows along y and columns along x.
    Note: compute_forces_numba and the event detection assume instead
    index = i * N_BLOCKS_Y + j with i along x; the two agree only on square
    grids, and even then x and y are swapped in the event epicentres.

    Returns
    -------
    positions : ndarray, shape (N_BLOCKS, 3)
        (x, y, z) of each block, in m.
    """
    x_blocks = np.linspace(0, (N_BLOCKS_X - 1) * BLOCK_SPACING, N_BLOCKS_X)
    y_blocks = np.linspace(0, (N_BLOCKS_Y - 1) * BLOCK_SPACING, N_BLOCKS_Y)

    xx, yy = np.meshgrid(x_blocks, y_blocks)  # shape (N_BLOCKS_Y, N_BLOCKS_X)
    
    # Initial z = 0 (fault plane)
    zz = np.zeros_like(xx)
    
    positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    return positions
