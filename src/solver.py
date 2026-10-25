"""
Time integration of the Burridge-Knopoff system and event detection.

The ODE is integrated with scipy.integrate.solve_ivp (RK45) from one sampling
time to the next. At every sample the block positions, velocities and the
sensor potential changes are stored, and the sampled velocities are used to
detect earthquakes.

Known issues:
- the spring oscillations (periods ~30 s) force RK45 to take on the order of
  1e9 steps over 100 years, so the default run does not finish in practice;
- events are detected only at the sampling times (every DT_SAMPLE), so short
  slip events are mostly missed, and an event still running at the end of the
  run is not recorded;
- the "magnitude" is log10(peak velocity) + 5, not a moment magnitude;
- epicentre (i, j) follow the force kernel's convention, which swaps x and y
  relative to the block positions.
"""

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import config

class EarthquakeSimulator:
    """
    Integrates a BurridgeKnopoffSystem, samples it and detects events.

    Results accumulate in lists (times in years; positions and velocities as
    (n_blocks, 3) arrays; potential changes as (n_sensors,) arrays; events as
    dicts). Every sample of every block is kept in memory: about 4.8 GB for
    the default 100 x 100 grid over 100 years.
    """

    def __init__(self, bk_system, gravity_sensors):
        """
        Parameters
        ----------
        bk_system : BurridgeKnopoffSystem
            The model (right-hand side of the ODE and initial state).
        gravity_sensors : GravitySensorArray
            Sensors evaluated at every sample.
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
        
        # Event detection state: an event is "open" while the fastest block
        # moves faster than v_threshold at consecutive samples
        self.v_threshold = config.VELOCITY_THRESHOLD
        self.in_event = False
        self.event_start_time = 0.0
        self.event_max_velocity = 0.0
        self.event_epicenter = None
        
    def run_simulation(self):
        """
        Run the full simulation with event detection.

        Returns
        -------
        times : list of float
            Sample times (years).
        positions, velocities : list of ndarray, each (n_blocks, 3)
            Block positions (m) and velocities (m/s) at each sample.
        potentials : list of ndarray, each (n_sensors,)
            Potential change at each sensor (J/kg) relative to the reference grid.
        events : list of dict
            Detected events (see _detect_events).
        """
        print("Initializing Burridge-Knopoff simulation...")
        print(f"Grid size: {config.N_BLOCKS_X}x{config.N_BLOCKS_Y} blocks")
        print(f"Sensor array: {config.N_SENSORS_X}x{config.N_SENSORS_Y} sensors")
        print(f"Simulation time: {config.SIMULATION_TIME} years")
        
        # Initial state
        y0 = self.bk_system.get_initial_state()
        
        # Sampling times (s): 0, dt, 2 dt, ... < t_total (the end time itself
        # is not sampled, and the first "interval" is [0, 0])
        t_eval = np.arange(0, self.t_total, self.dt_sample)
        n_samples = len(t_eval)
        
        print(f"\nStarting integration ({n_samples} time samples)...")
        
        # Custom integration with progress bar and event detection
        t_current = 0.0
        state_current = y0
        
        with tqdm(total=n_samples, desc="Simulating") as pbar:
            for i, t_target in enumerate(t_eval):
                # Integrate to the next sampling time. The integrator is restarted
                # for every interval (losing its step-size history), and only the
                # final state of the interval is kept.
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
                
                # Event detection from this sample's velocities only (motion
                # between samples is not seen)
                self._detect_events(t_current, velocities)
                
                pbar.update(1)
        
        print(f"\nSimulation complete!")
        print(f"Total events detected: {len(self.events)}")
        
        return self.times, self.positions, self.velocities, self.potentials, self.events
    
    def _detect_events(self, t, velocities):
        """
        Update the event detection with the velocities of one sample.

        An event starts at the first sample where the fastest block exceeds
        VELOCITY_THRESHOLD and ends at the first sample where none does. The
        epicentre is the fastest block at the moment of peak velocity. When an
        event ends and lasted more than MIN_EVENT_DURATION, a dict is appended
        to self.events with keys 'time' (start, years), 'duration' (s),
        'magnitude' (log10(peak velocity) + 5, a placeholder, not a moment
        magnitude), 'max_velocity' (m/s), 'epicenter_i', 'epicenter_j'.

        Parameters
        ----------
        t : float
            Current time (s).
        velocities : ndarray, shape (n_blocks, 3)
            Block velocities (m/s).
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
                
                # Epicentre = fastest block. KNOWN ISSUE: (i, j) use the force
                # kernel's idx = i * N_BLOCKS_Y + j, but blocks are stored as
                # row * N_BLOCKS_X + column (rows along y), so i is the y index.
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
                
                # Only record if duration exceeds minimum (with sampled detection the
                # duration is a multiple of the sampling interval, so this always holds)
                if event_duration > config.MIN_EVENT_DURATION:
                    # Placeholder "magnitude" from the peak velocity. A moment
                    # magnitude needs the seismic moment M0 = shear modulus *
                    # area * slip: Mw = (2/3)(log10 M0 - 9.1), M0 in N m.
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
