"""
Configuration file for Burridge-Knopoff earthquake simulation
Physical parameters based on Japanese Trench characteristics
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
T_INITIAL_EQUILIBRATION = 10.0  # years (initial equilibration period)

# Physical parameters (SI units)
BLOCK_MASS = 1e12  # kg (mass of each block, ~crustal segment)
BLOCK_SPACING = 2000.0  # meters (spacing between blocks, 2 km)
SPRING_K_HORIZONTAL = 1e10  # N/m (spring stiffness horizontal)
SPRING_K_VERTICAL = 1e10  # N/m (spring stiffness vertical)

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

# Damping coefficient (small damping for numerical stability)
DAMPING = 1e5  # kg/s

# Numerical solver parameters
RTOL = 1e-6  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance
MAX_STEP = 0.01 * 365.25 * 24 * 3600  # Maximum time step in seconds (0.01 year)

# Event detection thresholds
VELOCITY_THRESHOLD = 1e-3  # m/s (threshold for earthquake detection)
MIN_EVENT_DURATION = 1.0  # seconds (minimum duration to count as event)

# Output settings
OUTPUT_DIR = 'output'
CSV_DIR = 'output/csv'
ANIMATION_DIR = 'output/animations'

# Derived parameters
BLOCK_AREA = BLOCK_SPACING ** 2  # m^2
NORMAL_FORCE = BLOCK_MASS * 9.81  # N (normal force per block)

# Convert time units
SECONDS_PER_YEAR = 365.25 * 24 * 3600

def get_sensor_positions():
    """
    Calculate fixed positions of gravity sensors.
    Returns positions in meters as (x, y, z) coordinates.
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
    Calculate initial positions of blocks in the grid.
    Returns positions in meters as (x, y, z) coordinates.
    """
    x_blocks = np.linspace(0, (N_BLOCKS_X - 1) * BLOCK_SPACING, N_BLOCKS_X)
    y_blocks = np.linspace(0, (N_BLOCKS_Y - 1) * BLOCK_SPACING, N_BLOCKS_Y)
    
    xx, yy = np.meshgrid(x_blocks, y_blocks)
    
    # Initial z = 0 (fault plane)
    zz = np.zeros_like(xx)
    
    positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    return positions
