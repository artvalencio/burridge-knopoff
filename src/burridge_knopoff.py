"""
Core of the 2-D Burridge-Knopoff model: the state vector and the forces.

Each block i has a position r_i (3 components), a velocity v_i and a
rate-and-state friction variable theta_i. The equations of motion are

    dr_i/dt     = v_i
    m dv_i/dt   = F_springs + F_friction + F_damping       (compute_forces_numba)
    dtheta_i/dt = 1 - |v_i| theta_i / D_c                  (aging law)

Known issues of the current formulation (to be fixed test-first):
- there is no loading spring to the plate; instead the plate *displacement*
  is subtracted from the block *velocity* in the friction law (unit error),
  which acts as a constant push along x;
- the friction coefficient is not the Dieterich-Ruina law (see the kernel);
- only x has friction or loading, so y and z carry no physics;
- the kernel's index convention differs from config.get_block_positions on
  non-square grids.
"""

import numpy as np
from numba import jit
import config

class BurridgeKnopoffSystem:
    """
    2-D Burridge-Knopoff earthquake model with rate-and-state friction.

    Holds the parameters read from config and provides the right-hand side of
    the ODE (compute_forces) for scipy.integrate.solve_ivp.

    Attributes
    ----------
    n_x, n_y, n_blocks : int
        Grid size and number of blocks.
    block_positions_init : ndarray, shape (n_blocks, 3)
        Reference positions (m); spring rest lengths are measured from them.
    plate_displacement : float
        v_plate * t at the last call of compute_forces (m).
    """

    def __init__(self):
        """Read the model parameters from config and build the reference grid."""
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
        # (not used: the ODE carries its own theta, see get_initial_state)
        self.state_variable = np.ones(self.n_blocks) * self.dc / self.v_char
        
        # Plate displacement (increases with time)
        self.plate_displacement = 0.0
        
    def get_initial_state(self):
        """
        Initial state vector for the ODE solver.

        Layout (7N values):
        [x1, y1, z1, ..., xN, yN, zN,  vx1, vy1, vz1, ..., vxN, vyN, vzN,  theta1, ..., thetaN]

        Blocks start at their reference positions, at rest, with
        theta = D_c / v_char, the steady-state value at v = v_char.

        Returns
        -------
        state : ndarray, shape (7 * n_blocks,)
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
        Right-hand side of the ODE: time derivative of the state vector.

        Parameters
        ----------
        t : float
            Time (s).
        state : ndarray, shape (7 * n_blocks,)
            State vector (layout in get_initial_state).

        Returns
        -------
        derivatives : ndarray, shape (7 * n_blocks,)
            [velocities, accelerations, dtheta/dt], in the same layout.
        """
        # Views of the state components (no copies)
        n = self.n_blocks
        positions = state[:3*n].reshape(n, 3)
        velocities = state[3*n:6*n].reshape(n, 3)
        theta = state[6*n:]

        # Plate displacement at time t (m). KNOWN ISSUE: it is passed to the
        # kernel, which subtracts it from a velocity instead of using it to
        # stretch a loading spring.
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
        
        # Aging law for the rate-and-state variable: dtheta/dt = 1 - v*theta/D_c,
        # which relaxes theta towards D_c/v (KNOWN ISSUE: uses the norm of the
        # absolute 3-D velocity, while friction uses the x slip velocity)
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
        """Positions and velocities of the blocks as (n_blocks, 3) views of the state vector."""
        n = self.n_blocks
        positions = state[:3*n].reshape(n, 3)
        velocities = state[3*n:6*n].reshape(n, 3)
        return positions, velocities
    
    def get_block_index(self, i, j):
        """1-D block index of grid indices (i, j), in the force kernel's convention
        (i * n_y + j). Not used; differs from the order of config.get_block_positions
        (row * n_x + column) on non-square grids."""
        return i * self.n_y + j


@jit(nopython=True)
def compute_forces_numba(positions, velocities, theta,
                         n_x, n_y, mass, spacing,
                         k_h, k_v, mu_s, mu_d, v_char,
                         a, b, dc, plate_disp,
                         damping, f_normal, ref_positions):
    """
    Total force on every block (compiled with numba).

    For each block the force is the sum of
    1. linear springs to its four nearest neighbours: k * (stretch), where the
       stretch is the change of the neighbour offset from its reference value;
    2. friction along x (static or rate-and-state branch, see below);
    3. linear damping, -damping * velocity.

    Parameters
    ----------
    positions, velocities : ndarray, shape (n_blocks, 3)
        Current positions (m) and velocities (m/s).
    theta : ndarray, shape (n_blocks,)
        Rate-and-state variables (s).
    n_x, n_y : int
        Grid size; block idx is treated as (i, j) = (idx // n_y, idx % n_y).
    mass, spacing : float
        Block mass (kg; unused here) and spacing (m; unused here).
    k_h, k_v : float
        Spring constants along i and along j (N/m).
    mu_s, mu_d, v_char, a, b, dc : float
        Friction parameters.
    plate_disp : float
        Plate displacement v_plate * t (m).
    damping, f_normal : float
        Damping coefficient (kg/s) and normal force per block (N).
    ref_positions : ndarray, shape (n_blocks, 3)
        Reference positions (m).

    Returns
    -------
    forces : ndarray, shape (n_blocks, 3)
        Force on each block (N).
    """
    n_blocks = n_x * n_y
    forces = np.zeros((n_blocks, 3))

    for idx in range(n_blocks):
        # 1-D index -> grid indices. KNOWN ISSUE: config.get_block_positions
        # orders blocks as row * n_x + column (rows along y); this
        # i * n_y + j convention matches it only on square grids, and on
        # non-square grids the "neighbours" below are not adjacent blocks.
        i = idx // n_y
        j = idx % n_y
        
        # Current position and velocity
        pos = positions[idx]
        vel = velocities[idx]
        
        # Spring forces from the (up to) four neighbours. The force is
        # k * stretch with stretch = (r_n - r) - (r_n_ref - r_ref): zero at the
        # reference configuration, so only displacements produce forces.
        # Blocks on the edges simply have fewer springs (free boundaries).
        spring_force = np.zeros(3)  # (allocated per block: slow in the inner loop)
        
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
        
        # Friction along x, opposing the slip velocity v_rel.
        # KNOWN ISSUE: plate_disp is a displacement (m) subtracted from a
        # velocity (m/s); as soon as the plate has moved, v_rel ~ -plate_disp
        # and friction becomes a constant push of +mu*N along x. In a standard
        # Burridge-Knopoff model the slip velocity is the block velocity and
        # the plate loads each block through a spring k_p*(v_plate*t - u).
        v_rel = vel[0] - plate_disp
        v_magnitude = abs(v_rel)

        if v_magnitude < 1e-12:
            # "Stuck" branch: friction cancels the spring force along x, up to
            # the static limit mu_s * N. (An ODE state practically never has
            # |v_rel| < 1e-12, so blocks do not really stick.)
            mu = mu_s
            friction_force_x = -spring_force[0]
            max_friction = mu * f_normal
            if abs(friction_force_x) > max_friction:
                friction_force_x = max_friction * np.sign(-spring_force[0])
        else:
            # Sliding branch: rate-and-state-like coefficient
            #   mu = mu_d + a ln(v/v_char) + b ln(v_char theta / D_c),
            # clamped at mu >= mu_d. KNOWN ISSUE: Dieterich-Ruina uses a
            # reference coefficient mu_0 (not mu_d) and no clamp; the clamp
            # removes velocity weakening below v_char.
            mu = mu_d + a * np.log(v_magnitude / v_char + 1e-12) + b * np.log(v_char * theta[idx] / dc + 1e-12)
            mu = max(mu, mu_d)
            friction_force_x = -mu * f_normal * np.sign(v_rel)
        
        friction_force = np.array([friction_force_x, 0.0, 0.0])
        
        # Linear (viscous) damping in all three directions
        damping_force = -damping * vel
        
        # Total force
        forces[idx] = spring_force + friction_force + damping_force
    
    return forces
