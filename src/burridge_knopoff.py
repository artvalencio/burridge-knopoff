"""
Core Burridge-Knopoff model implementation for 2D grid of blocks
Implements stick-slip friction dynamics for earthquake simulation
"""

import numpy as np
from numba import jit
import config

class BurridgeKnopoffSystem:
    """
    2D Burridge-Knopoff earthquake model with rate-and-state friction
    """
    
    def __init__(self):
        """Initialize the block system"""
        self.n_x = config.N_BLOCKS_X
        self.n_y = config.N_BLOCKS_Y
        self.n_blocks = config.N_BLOCKS
        
        # Block properties
        self.mass = config.BLOCK_MASS
        self.spacing = config.BLOCK_SPACING
        
        # Spring constants
        self.k_h = config.SPRING_K_HORIZONTAL
        self.k_v = config.SPRING_K_VERTICAL
        
        # Friction parameters
        self.mu_s = config.MU_STATIC
        self.mu_d = config.MU_DYNAMIC
        self.v_char = config.V_CHARACTERISTIC
        self.a = config.A_FRICTION
        self.b = config.B_FRICTION
        self.dc = config.DC
        
        # Plate motion
        self.v_plate = config.PLATE_VELOCITY_SI
        
        # Damping
        self.damping = config.DAMPING
        
        # Normal force
        self.f_normal = config.NORMAL_FORCE
        
        # Initialize positions (reference positions for springs)
        self.block_positions_init = config.get_block_positions()
        
        # Initialize state variable for rate-and-state friction
        self.state_variable = np.ones(self.n_blocks) * self.dc / self.v_char
        
        # Plate displacement (increases with time)
        self.plate_displacement = 0.0
        
    def get_initial_state(self):
        """
        Get initial state vector for ODE solver
        State: [x1, y1, z1, ..., xN, yN, zN, vx1, vy1, vz1, ..., vxN, vyN, vzN, theta1, ..., thetaN]
        """
        # Initial positions (at reference)
        positions = self.block_positions_init.copy()
        
        # Initial velocities (zero)
        velocities = np.zeros_like(positions)
        
        # Initial state variables
        theta = np.ones(self.n_blocks) * self.dc / self.v_char
        
        # Combine into state vector
        state = np.concatenate([
            positions.flatten(),
            velocities.flatten(),
            theta
        ])
        
        return state
    
    def compute_forces(self, t, state):
        """
        Compute forces on all blocks for ODE solver
        
        Args:
            t: Current time (seconds)
            state: State vector [positions, velocities, state_variables]
            
        Returns:
            derivatives: [velocities, accelerations, d_theta/dt]
        """
        # Extract state components
        n = self.n_blocks
        positions = state[:3*n].reshape(n, 3)
        velocities = state[3*n:6*n].reshape(n, 3)
        theta = state[6*n:]
        
        # Update plate displacement
        self.plate_displacement = self.v_plate * t
        
        # Compute forces using optimized function
        forces = compute_forces_numba(
            positions, velocities, theta,
            self.n_x, self.n_y,
            self.mass, self.spacing,
            self.k_h, self.k_v,
            self.mu_s, self.mu_d, self.v_char,
            self.a, self.b, self.dc,
            self.plate_displacement,
            self.damping, self.f_normal,
            self.block_positions_init
        )
        
        # Compute accelerations
        accelerations = forces / self.mass
        
        # Evolution of state variable (rate-and-state friction)
        v_magnitude = np.linalg.norm(velocities, axis=1)
        d_theta_dt = 1.0 - (v_magnitude * theta) / self.dc
        
        # Combine derivatives
        derivatives = np.concatenate([
            velocities.flatten(),
            accelerations.flatten(),
            d_theta_dt
        ])
        
        return derivatives
    
    def extract_positions_velocities(self, state):
        """Extract positions and velocities from state vector"""
        n = self.n_blocks
        positions = state[:3*n].reshape(n, 3)
        velocities = state[3*n:6*n].reshape(n, 3)
        return positions, velocities
    
    def get_block_index(self, i, j):
        """Convert 2D grid indices to 1D block index"""
        return i * self.n_y + j


@jit(nopython=True)
def compute_forces_numba(positions, velocities, theta,
                         n_x, n_y, mass, spacing,
                         k_h, k_v, mu_s, mu_d, v_char,
                         a, b, dc, plate_disp,
                         damping, f_normal, ref_positions):
    """
    Compute forces on all blocks (optimized with numba)
    
    This function computes:
    1. Spring forces from neighboring blocks
    2. Friction forces from the moving plate
    3. Damping forces
    """
    n_blocks = n_x * n_y
    forces = np.zeros((n_blocks, 3))
    
    for idx in range(n_blocks):
        # Convert 1D index to 2D grid position
        i = idx // n_y
        j = idx % n_y
        
        # Current position and velocity
        pos = positions[idx]
        vel = velocities[idx]
        
        # Spring forces from neighbors
        spring_force = np.zeros(3)
        
        # Left neighbor (i-1, j)
        if i > 0:
            neighbor_idx = (i-1) * n_y + j
            displacement = positions[neighbor_idx] - pos
            ref_displacement = ref_positions[neighbor_idx] - ref_positions[idx]
            stretch = displacement - ref_displacement
            spring_force += k_h * stretch
        
        # Right neighbor (i+1, j)
        if i < n_x - 1:
            neighbor_idx = (i+1) * n_y + j
            displacement = positions[neighbor_idx] - pos
            ref_displacement = ref_positions[neighbor_idx] - ref_positions[idx]
            stretch = displacement - ref_displacement
            spring_force += k_h * stretch
        
        # Bottom neighbor (i, j-1)
        if j > 0:
            neighbor_idx = i * n_y + (j-1)
            displacement = positions[neighbor_idx] - pos
            ref_displacement = ref_positions[neighbor_idx] - ref_positions[idx]
            stretch = displacement - ref_displacement
            spring_force += k_v * stretch
        
        # Top neighbor (i, j+1)
        if j < n_y - 1:
            neighbor_idx = i * n_y + (j+1)
            displacement = positions[neighbor_idx] - pos
            ref_displacement = ref_positions[neighbor_idx] - ref_positions[idx]
            stretch = displacement - ref_displacement
            spring_force += k_v * stretch
        
        # Friction force (acts in x-direction, opposing motion relative to plate)
        v_rel = vel[0] - plate_disp  # Relative velocity to plate
        v_magnitude = abs(v_rel)
        
        # Rate-and-state friction coefficient
        if v_magnitude < 1e-12:
            # Static friction
            mu = mu_s
            friction_force_x = -spring_force[0]  # Balance spring force
            # Limit to maximum static friction
            max_friction = mu * f_normal
            if abs(friction_force_x) > max_friction:
                friction_force_x = max_friction * np.sign(-spring_force[0])
        else:
            # Dynamic friction with rate-and-state
            mu = mu_d + a * np.log(v_magnitude / v_char + 1e-12) + b * np.log(v_char * theta[idx] / dc + 1e-12)
            mu = max(mu, mu_d)  # Ensure mu >= mu_dynamic
            friction_force_x = -mu * f_normal * np.sign(v_rel)
        
        friction_force = np.array([friction_force_x, 0.0, 0.0])
        
        # Damping force
        damping_force = -damping * vel
        
        # Total force
        forces[idx] = spring_force + friction_force + damping_force
    
    return forces
