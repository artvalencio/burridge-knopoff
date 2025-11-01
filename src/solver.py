"""
Numerical solver for the Burridge-Knopoff system
Handles time integration and event detection
"""

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import config

class EarthquakeSimulator:
    """
    Main simulator class that integrates the system and detects events
    """
    
    def __init__(self, bk_system, gravity_sensors):
        """
        Initialize simulator
        
        Args:
            bk_system: BurridgeKnopoffSystem instance
            gravity_sensors: GravitySensorArray instance
        """
        self.bk_system = bk_system
        self.gravity_sensors = gravity_sensors
        
        # Simulation parameters
        self.t_total = config.SIMULATION_TIME * config.SECONDS_PER_YEAR
        self.dt_sample = config.DT_SAMPLE * config.SECONDS_PER_YEAR
        
        # Storage for results
        self.times = []
        self.positions = []
        self.velocities = []
        self.potentials = []
        self.events = []
        
        # Event detection
        self.v_threshold = config.VELOCITY_THRESHOLD
        self.in_event = False
        self.event_start_time = 0.0
        self.event_max_velocity = 0.0
        self.event_epicenter = None
        
    def run_simulation(self):
        """
        Run the full simulation with event detection
        """
        print("Initializing Burridge-Knopoff simulation...")
        print(f"Grid size: {config.N_BLOCKS_X}x{config.N_BLOCKS_Y} blocks")
        print(f"Sensor array: {config.N_SENSORS_X}x{config.N_SENSORS_Y} sensors")
        print(f"Simulation time: {config.SIMULATION_TIME} years")
        
        # Initial state
        y0 = self.bk_system.get_initial_state()
        
        # Time points for sampling
        t_eval = np.arange(0, self.t_total, self.dt_sample)
        n_samples = len(t_eval)
        
        print(f"\nStarting integration ({n_samples} time samples)...")
        
        # Custom integration with progress bar and event detection
        t_current = 0.0
        state_current = y0
        
        with tqdm(total=n_samples, desc="Simulating") as pbar:
            for i, t_target in enumerate(t_eval):
                # Integrate to next sampling point
                sol = solve_ivp(
                    self.bk_system.compute_forces,
                    [t_current, t_target],
                    state_current,
                    method='RK45',
                    rtol=config.RTOL,
                    atol=config.ATOL,
                    max_step=config.MAX_STEP
                )
                
                if not sol.success:
                    print(f"\nWarning: Integration failed at t={t_current/config.SECONDS_PER_YEAR:.3f} years")
                    break
                
                # Update state
                state_current = sol.y[:, -1]
                t_current = t_target
                
                # Extract positions and velocities
                positions, velocities = self.bk_system.extract_positions_velocities(state_current)
                
                # Compute gravitational potential
                delta_potential = self.gravity_sensors.compute_potential_change(positions)
                
                # Store results
                self.times.append(t_current / config.SECONDS_PER_YEAR)  # Convert to years
                self.positions.append(positions.copy())
                self.velocities.append(velocities.copy())
                self.potentials.append(delta_potential.copy())
                
                # Event detection
                self._detect_events(t_current, velocities)
                
                pbar.update(1)
        
        print(f"\nSimulation complete!")
        print(f"Total events detected: {len(self.events)}")
        
        return self.times, self.positions, self.velocities, self.potentials, self.events
    
    def _detect_events(self, t, velocities):
        """
        Detect earthquake events based on velocity threshold
        
        Args:
            t: current time (seconds)
            velocities: array of shape (n_blocks, 3)
        """
        # Compute maximum velocity magnitude
        v_magnitudes = np.linalg.norm(velocities, axis=1)
        v_max = np.max(v_magnitudes)
        v_mean = np.mean(v_magnitudes)
        
        # Check if we're in an event
        if v_max > self.v_threshold:
            if not self.in_event:
                # Event starting
                self.in_event = True
                self.event_start_time = t
                self.event_max_velocity = v_max
                
                # Find epicenter (block with maximum velocity)
                epicenter_idx = np.argmax(v_magnitudes)
                i = epicenter_idx // config.N_BLOCKS_Y
                j = epicenter_idx % config.N_BLOCKS_Y
                self.event_epicenter = (i, j)
            else:
                # Event continuing, update max velocity
                if v_max > self.event_max_velocity:
                    self.event_max_velocity = v_max
                    epicenter_idx = np.argmax(v_magnitudes)
                    i = epicenter_idx // config.N_BLOCKS_Y
                    j = epicenter_idx % config.N_BLOCKS_Y
                    self.event_epicenter = (i, j)
        else:
            if self.in_event:
                # Event ending
                event_duration = t - self.event_start_time
                
                # Only record if duration exceeds minimum
                if event_duration > config.MIN_EVENT_DURATION:
                    # Estimate magnitude from maximum velocity (simplified)
                    # Real magnitude would require moment tensor calculation
                    magnitude = np.log10(self.event_max_velocity) + 5.0
                    
                    event_data = {
                        'time': self.event_start_time / config.SECONDS_PER_YEAR,  # years
                        'duration': event_duration,  # seconds
                        'magnitude': magnitude,
                        'max_velocity': self.event_max_velocity,  # m/s
                        'epicenter_i': self.event_epicenter[0],
                        'epicenter_j': self.event_epicenter[1]
                    }
                    
                    self.events.append(event_data)
                
                # Reset event state
                self.in_event = False
                self.event_start_time = 0.0
                self.event_max_velocity = 0.0
                self.event_epicenter = None
