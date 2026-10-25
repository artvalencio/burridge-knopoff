"""
Simulation parameters.

All quantities are SI, and every field name carries its unit (_m, _pa, _n_m,
_kg_m3, ...), except the simulation schedule, which is in years (_years). The
defaults describe a 200 km x 200 km fault patch loaded at the Japan Trench
plate rate (9 cm/year). They are chosen to be mutually consistent (events
occur every few years), not calibrated to a real fault.
"""
import dataclasses
import math

SECONDS_PER_YEAR = 365.25 * 24 * 3600
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Parameters of a Burridge-Knopoff simulation (immutable; use dataclasses.replace).

    Grid
        n_blocks_x, n_blocks_y: blocks along x (plate motion) and y.
        block_spacing_m: distance between neighbouring blocks.
        seismogenic_thickness_m, rock_density_kg_m3: give the block mass
            (density x spacing^2 x thickness).
    Springs
        coupling_stiffness_n_m: springs between neighbouring blocks (default
            about shear modulus x spacing).
        loading_stiffness_n_m: "leaf" springs pulling each block towards the
            moving plate.
    Friction (forces = stress x block area)
        static_friction_stress_pa: mean strength. A stuck block starts slipping
            when the force on it exceeds its strength.
        dynamic_friction_stress_pa: friction while slipping (below static).
        heterogeneity: relative spread h of the strength excess (static minus
            dynamic) between blocks, uniform in [1 - h, 1 + h], 0 <= h < 1.
        redraw_thresholds: draw a new strength for a block after each slip
            (annealed disorder) instead of keeping it fixed (quenched).
        weakening_velocity_m_s: if set, sliding friction decays from static to
            dynamic as F_d + (F_s - F_d) / (1 + v / v_w); None: constant F_d.
    Loading and moment
        plate_velocity_m_per_year: plate rate.
        shear_modulus_pa: for the seismic moment M0 = shear modulus x area x slip.
    Sensors
        n_sensors_x, n_sensors_y, sensor_height_m: sensor grid centred over the
            blocks, at this height above the fault plane z = 0.
    Schedule
        simulation_years: simulated time.
        sample_interval_years: interval of the sensor time-series.
        snapshot_interval_years: interval of the stored block displacement field.
        steps_per_period: time steps per period of the fastest oscillation
            during slip (accuracy of the dynamics).
        seed: random seed of the heterogeneity (results are reproducible).
    """
    n_blocks_x: int = 100
    n_blocks_y: int = 100
    block_spacing_m: float = 2000.0
    seismogenic_thickness_m: float = 10_000.0
    rock_density_kg_m3: float = 2700.0

    coupling_stiffness_n_m: float = 6e13
    loading_stiffness_n_m: float = 2.4e13

    static_friction_stress_pa: float = 10e6
    dynamic_friction_stress_pa: float = 7e6
    heterogeneity: float = 0.5
    redraw_thresholds: bool = True
    weakening_velocity_m_s: float = None

    plate_velocity_m_per_year: float = 0.09
    shear_modulus_pa: float = 3e10

    n_sensors_x: int = 10
    n_sensors_y: int = 10
    sensor_height_m: float = 5000.0

    simulation_years: float = 100.0
    sample_interval_years: float = 0.01
    snapshot_interval_years: float = 1.0
    steps_per_period: int = 100
    seed: int = 0

    # ---- derived quantities -------------------------------------------------
    @property
    def n_blocks(self):
        return self.n_blocks_x * self.n_blocks_y

    @property
    def n_sensors(self):
        return self.n_sensors_x * self.n_sensors_y

    @property
    def block_area_m2(self):
        return self.block_spacing_m ** 2

    @property
    def block_mass_kg(self):
        return self.rock_density_kg_m3 * self.block_area_m2 * self.seismogenic_thickness_m

    @property
    def plate_velocity_m_s(self):
        return self.plate_velocity_m_per_year / SECONDS_PER_YEAR

    @property
    def static_force_n(self):
        return self.static_friction_stress_pa * self.block_area_m2

    @property
    def dynamic_force_n(self):
        return self.dynamic_friction_stress_pa * self.block_area_m2

    @property
    def slip_time_step_s(self):
        """Time step during slip: a fraction of the period of the fastest oscillation
        (a block held by its loading spring and four coupling springs)."""
        omega = math.sqrt((self.loading_stiffness_n_m + 4 * self.coupling_stiffness_n_m) / self.block_mass_kg)
        return 2 * math.pi / omega / self.steps_per_period

    def single_block_recurrence_years(self):
        """Recurrence time of an isolated block of mean strength: it slips
        2 (F_s - F_d) / k_p per event and reloads at the rate k_p v_p."""
        slip = 2 * (self.static_force_n - self.dynamic_force_n) / self.loading_stiffness_n_m
        return slip / self.plate_velocity_m_per_year

    # ---- checks -------------------------------------------------------------
    def validate(self):
        """Raises ValueError naming the first invalid field; returns self."""
        positive = ['n_blocks_x', 'n_blocks_y', 'block_spacing_m', 'seismogenic_thickness_m', 'rock_density_kg_m3',
                    'coupling_stiffness_n_m', 'loading_stiffness_n_m', 'static_friction_stress_pa',
                    'plate_velocity_m_per_year', 'shear_modulus_pa', 'n_sensors_x', 'n_sensors_y',
                    'sensor_height_m', 'simulation_years', 'sample_interval_years', 'snapshot_interval_years']
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError('%s must be positive, got %r' % (name, getattr(self, name)))
        if not 0 <= self.dynamic_friction_stress_pa < self.static_friction_stress_pa:
            raise ValueError('dynamic_friction_stress_pa must be in [0, static_friction_stress_pa)')
        if not 0 <= self.heterogeneity < 1:
            raise ValueError('heterogeneity must be in [0, 1), got %r' % self.heterogeneity)
        if self.weakening_velocity_m_s is not None and not self.weakening_velocity_m_s > 0:
            raise ValueError('weakening_velocity_m_s must be positive or None')
        if self.steps_per_period < 10:
            raise ValueError('steps_per_period must be at least 10, got %r' % self.steps_per_period)
        return self

    def check_parameters(self):
        """Physical sanity warnings (list of strings, empty when nothing looks off)."""
        warnings = []
        recurrence = self.single_block_recurrence_years()
        if recurrence > self.simulation_years:
            warnings.append('single-block recurrence time %.3g years exceeds the simulated %.3g years: '
                            'few or no events expected' % (recurrence, self.simulation_years))
        if self.sample_interval_years > self.simulation_years:
            warnings.append('sample interval longer than the simulation')
        return warnings
