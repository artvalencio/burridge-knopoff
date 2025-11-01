"""
Gravity sensor array for measuring gravitational potential changes
due to block displacements in the Burridge-Knopoff model
"""

import numpy as np
from numba import jit
import config

class GravitySensorArray:
    """
    Array of gravity sensors measuring gravitational potential changes
    """
    
    def __init__(self):
        """Initialize sensor array"""
        self.n_sensors = config.N_SENSORS
        self.sensor_positions = config.get_sensor_positions()
        
        # Gravitational constant
        self.G = config.G
        
        # Block properties
        self.block_mass = config.BLOCK_MASS
        self.n_blocks = config.N_BLOCKS
        
        # Reference potential (at initial block positions)
        self.ref_block_positions = config.get_block_positions()
        self.ref_potential = self.compute_potential(self.ref_block_positions)
        
    def compute_potential(self, block_positions):
        """
        Compute gravitational potential at all sensor locations
        
        Args:
            block_positions: numpy array of shape (n_blocks, 3) with block positions
            
        Returns:
            potentials: numpy array of shape (n_sensors,) with potential at each sensor
        """
        potentials = compute_potential_numba(
            block_positions,
            self.sensor_positions,
            self.block_mass,
            self.G
        )
        
        return potentials
    
    def compute_potential_change(self, block_positions):
        """
        Compute change in gravitational potential relative to reference
        
        Args:
            block_positions: numpy array of shape (n_blocks, 3) with block positions
            
        Returns:
            delta_potential: numpy array of shape (n_sensors,) with potential change
        """
        current_potential = self.compute_potential(block_positions)
        delta_potential = current_potential - self.ref_potential
        
        return delta_potential
    
    def get_sensor_grid_coordinates(self):
        """
        Get sensor coordinates as 2D grid for visualization
        
        Returns:
            x_grid, y_grid: 2D arrays of x and y coordinates
        """
        x_coords = self.sensor_positions[:, 0].reshape(config.N_SENSORS_Y, config.N_SENSORS_X)
        y_coords = self.sensor_positions[:, 1].reshape(config.N_SENSORS_Y, config.N_SENSORS_X)
        
        return x_coords, y_coords


@jit(nopython=True)
def compute_potential_numba(block_positions, sensor_positions, block_mass, G):
    """
    Compute gravitational potential at sensor locations (optimized with numba)
    
    The gravitational potential due to a point mass m at distance r is:
    Φ = -G * m / r
    
    Total potential is the sum over all blocks.
    
    Args:
        block_positions: array of shape (n_blocks, 3)
        sensor_positions: array of shape (n_sensors, 3)
        block_mass: mass of each block (scalar)
        G: gravitational constant
        
    Returns:
        potentials: array of shape (n_sensors,)
    """
    n_sensors = sensor_positions.shape[0]
    n_blocks = block_positions.shape[0]
    
    potentials = np.zeros(n_sensors)
    
    for s_idx in range(n_sensors):
        sensor_pos = sensor_positions[s_idx]
        potential = 0.0
        
        for b_idx in range(n_blocks):
            block_pos = block_positions[b_idx]
            
            # Calculate distance
            dx = sensor_pos[0] - block_pos[0]
            dy = sensor_pos[1] - block_pos[1]
            dz = sensor_pos[2] - block_pos[2]
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Avoid division by zero
            if distance < 1e-6:
                distance = 1e-6
            
            # Add contribution to potential
            potential += -G * block_mass / distance
        
        potentials[s_idx] = potential
    
    return potentials


def compute_gravity_acceleration(block_positions, sensor_positions, block_mass, G):
    """
    Compute gravitational acceleration vector at sensor locations
    
    This is useful for comparison with actual gravity measurements (gravimeters)
    g = -∇Φ
    
    Args:
        block_positions: array of shape (n_blocks, 3)
        sensor_positions: array of shape (n_sensors, 3)
        block_mass: mass of each block
        G: gravitational constant
        
    Returns:
        accelerations: array of shape (n_sensors, 3) with [gx, gy, gz]
    """
    n_sensors = sensor_positions.shape[0]
    n_blocks = block_positions.shape[0]
    
    accelerations = np.zeros((n_sensors, 3))
    
    for s_idx in range(n_sensors):
        sensor_pos = sensor_positions[s_idx]
        
        for b_idx in range(n_blocks):
            block_pos = block_positions[b_idx]
            
            # Vector from block to sensor
            r_vec = sensor_pos - block_pos
            distance = np.linalg.norm(r_vec)
            
            if distance < 1e-6:
                continue
            
            # Gravitational acceleration: g = G*m/r^2 * (r_vec/r)
            g_magnitude = G * block_mass / (distance ** 2)
            g_vec = g_magnitude * (r_vec / distance)
            
            accelerations[s_idx] += g_vec
    
    return accelerations
