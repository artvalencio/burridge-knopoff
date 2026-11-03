"""
Simulation parameters.

All quantities are SI, and every field name carries its unit (_m, _pa, _n_m,
_kg_m3, ...), except the simulation schedule, which is in years (_years). The
defaults describe a 500 km x 500 km fault patch loaded at the Japan Trench
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
    Springs (derived from the geometry unless given explicitly)
        coupling_stiffness_n_m: springs between neighbouring blocks. Default
            None: shear modulus x seismogenic thickness (the elastic stiffness of
            the slab between two blocks: modulus x cross-section H L / distance L).
        loading_stiffness_n_m: "leaf" springs pulling each block towards the
            moving plate. Default None: shear modulus x block area /
            loading_distance_m (shear of the block's column over the depth at
            which the plate loads the fault).
        loading_distance_m: that loading depth. The ratio k_c / k_p sets how far
            stress is passed on (correlation length ~ sqrt(k_c / k_p) blocks); the
            default 20 km gives 50, i.e. ~7 blocks. Much smaller ratios cut the
            magnitude distribution off early (see analysis).
        The effective values are k_coupling_n_m and k_loading_n_m.
        interaction_range_blocks: blocks within this distance (in spacings) are
            coupled, with stiffness k_c (L / r)^interaction_exponent. 1 (default):
            the four nearest neighbours only (classic model). Larger values
            approximate the long-range static stress transfer of an elastic
            body, which decays as 1 / r^3.
    Friction (forces = stress x block area)
        static_friction_stress_pa: mean strength. A stuck block starts slipping
            when the force on it exceeds its strength.
        dynamic_friction_stress_pa: friction while slipping (below static).
        heterogeneity: relative spread h of the strength excess (static minus
            dynamic) between blocks, uniform in [1 - h, 1 + h], 0 <= h < 1.
        weakening_velocity_m_s, onset_drop: velocity-weakening friction
            (after Carlson & Langer, 1989). While a block slips, friction is
            F_d + (1 - onset_drop) (F_s,i - F_d) / (1 + v / weakening_velocity_m_s):
            it drops by onset_drop of the strength excess when slip starts (so
            that slip can start) and then weakens towards F_d as v grows.
            weakening_velocity_m_s None (default): constant friction F_d.
        redraw_thresholds: draw a new strength for a block after each slip
            (annealed disorder; never below the force it holds when it stops)
            instead of keeping it fixed (quenched).
    Loading and moment
        plate_velocity_m_per_year: plate rate.
        shear_modulus_pa: for the seismic moment M0 = shear modulus x area x slip.
    Sensors
        n_sensors_x, n_sensors_y, sensor_height_m: sensor grid centred over the
            blocks, at this height above the fault plane z = 0 (default 25 x 25:
            one sensor every 20 km over the 500 km patch).
    Earth tides (see tides.py)
        tides: Moon and Sun tidal forcing; the time between events is then
            stepped every tide_step_hours instead of jumped exactly. Off by default.
        tide_bodies: comma-separated 'moon', 'sun'.
        tide_stress, tide_body_force: the tidal stress on the fault (shear, and the
            normal stress that unclamps it) and the direct tidal force on the block
            masses, switchable separately.
        tide_scale: amplification factor (0: hourly stepping without tide).
        tide_spatial: 'linear' (fit over the area) or 'uniform' (centre value).
        epoch_utc: ISO time of t = 0.
        centre_latitude_deg, centre_longitude_deg: centre of the block grid.
        fault_strike_deg, fault_dip_deg, fault_rake_deg: orientation used to resolve
            the tidal stress (default: the 2011 Tohoku-oki mechanism). Model x points
            along the down-dip azimuth (strike + 90), y along the strike.
        friction_coefficient: static friction coefficient mu_s; the normal stress
            changes the strength by mu_s and the sliding friction by mu_d =
            mu_s F_d / F_s, per unit area.
        love_h2, love_l2, love_k2, poisson_ratio: elastic response of the Earth.
        skempton_b: undrained pore-pressure response (0: drained).
    Schedule
        simulation_years: simulated time.
        sample_interval_years: interval of the sensor time-series.
        snapshot_interval_years: interval of the stored block displacement field.
        steps_per_period: time steps per period of the fastest oscillation
            during slip (accuracy of the dynamics).
        spinup_years: events starting earlier are left out of the statistics
            (analysis.gutenberg_richter): every block starts at the same stress,
            so the first years are dominated by system-wide events.
        seed: random seed of the heterogeneity (results are reproducible).
    """
    n_blocks_x: int = 250
    n_blocks_y: int = 250
    block_spacing_m: float = 2000.0
    seismogenic_thickness_m: float = 10_000.0
    rock_density_kg_m3: float = 2700.0

    coupling_stiffness_n_m: float = None
    loading_stiffness_n_m: float = None
    loading_distance_m: float = 20_000.0
    interaction_range_blocks: int = 1
    interaction_exponent: float = 3.0

    static_friction_stress_pa: float = 10e6
    dynamic_friction_stress_pa: float = 7e6
    heterogeneity: float = 0.5
    weakening_velocity_m_s: float = None
    onset_drop: float = 0.2
    redraw_thresholds: bool = True

    plate_velocity_m_per_year: float = 0.09
    shear_modulus_pa: float = 3e10

    n_sensors_x: int = 25
    n_sensors_y: int = 25
    sensor_height_m: float = 5000.0

    tides: bool = False
    tide_bodies: str = 'moon,sun'
    tide_stress: bool = True
    tide_body_force: bool = True
    tide_scale: float = 1.0
    tide_step_hours: float = 1.0
    tide_spatial: str = 'linear'
    epoch_utc: str = '2000-01-01T12:00:00'
    centre_latitude_deg: float = 38.3
    centre_longitude_deg: float = 143.0
    fault_strike_deg: float = 203.0
    fault_dip_deg: float = 10.0
    fault_rake_deg: float = 88.0
    friction_coefficient: float = 0.6
    love_h2: float = 0.6078
    love_l2: float = 0.0847
    love_k2: float = 0.30
    poisson_ratio: float = 0.25
    skempton_b: float = 0.0

    simulation_years: float = 100.0
    sample_interval_years: float = 0.01
    snapshot_interval_years: float = 1.0
    steps_per_period: int = 100
    spinup_years: float = 20.0
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
    def k_coupling_n_m(self):
        """Coupling stiffness: explicit, or shear modulus x seismogenic thickness."""
        if self.coupling_stiffness_n_m is not None:
            return self.coupling_stiffness_n_m
        return self.shear_modulus_pa * self.seismogenic_thickness_m

    @property
    def k_loading_n_m(self):
        """Loading stiffness: explicit, or shear modulus x block area / loading distance."""
        if self.loading_stiffness_n_m is not None:
            return self.loading_stiffness_n_m
        return self.shear_modulus_pa * self.block_area_m2 / self.loading_distance_m

    @property
    def static_force_n(self):
        return self.static_friction_stress_pa * self.block_area_m2

    @property
    def dynamic_force_n(self):
        return self.dynamic_friction_stress_pa * self.block_area_m2

    @property
    def slip_time_step_s(self):
        """Time step during slip: a fraction of the period of the fastest oscillation
        (a block held by its loading spring and all its coupling springs)."""
        from .grid import interaction_stencil
        weights = interaction_stencil(self)[2]
        omega = math.sqrt((self.k_loading_n_m + self.k_coupling_n_m * weights.sum()) / self.block_mass_kg)
        return 2 * math.pi / omega / self.steps_per_period

    @property
    def dynamic_friction_coefficient(self):
        """mu_d = mu_s F_d / F_s: how much the sliding friction changes with normal stress."""
        return self.friction_coefficient * self.dynamic_friction_stress_pa / self.static_friction_stress_pa

    def single_block_recurrence_years(self):
        """Recurrence time of an isolated block of mean strength: it slips
        2 (F_s - F_d) / k_p per event and reloads at the rate k_p v_p."""
        slip = 2 * (self.static_force_n - self.dynamic_force_n) / self.k_loading_n_m
        return slip / self.plate_velocity_m_per_year

    # ---- checks -------------------------------------------------------------
    def validate(self):
        """Raises ValueError naming the first invalid field; returns self."""
        positive = ['n_blocks_x', 'n_blocks_y', 'block_spacing_m', 'seismogenic_thickness_m', 'rock_density_kg_m3',
                    'loading_distance_m', 'static_friction_stress_pa',
                    'plate_velocity_m_per_year', 'shear_modulus_pa', 'n_sensors_x', 'n_sensors_y',
                    'sensor_height_m', 'simulation_years', 'sample_interval_years', 'snapshot_interval_years']
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError('%s must be positive, got %r' % (name, getattr(self, name)))
        for name in ('coupling_stiffness_n_m', 'loading_stiffness_n_m'):
            if getattr(self, name) is not None and not getattr(self, name) > 0:
                raise ValueError('%s must be positive or None, got %r' % (name, getattr(self, name)))
        if not 0 <= self.dynamic_friction_stress_pa < self.static_friction_stress_pa:
            raise ValueError('dynamic_friction_stress_pa must be in [0, static_friction_stress_pa)')
        if not 0 <= self.heterogeneity < 1:
            raise ValueError('heterogeneity must be in [0, 1), got %r' % self.heterogeneity)
        if not (isinstance(self.interaction_range_blocks, int) and self.interaction_range_blocks >= 1):
            raise ValueError('interaction_range_blocks must be an integer >= 1, got %r' % self.interaction_range_blocks)
        if not self.interaction_exponent > 0:
            raise ValueError('interaction_exponent must be positive, got %r' % self.interaction_exponent)
        if self.weakening_velocity_m_s is not None:
            if not self.weakening_velocity_m_s > 0:
                raise ValueError('weakening_velocity_m_s must be positive or None, got %r' % self.weakening_velocity_m_s)
            if not 0 < self.onset_drop <= 1:
                raise ValueError('onset_drop must be in (0, 1] with velocity weakening, got %r' % self.onset_drop)
        if self.tides:
            bodies = [b.strip() for b in self.tide_bodies.split(',') if b.strip()]
            if not bodies or any(b not in ('moon', 'sun') for b in bodies):
                raise ValueError("tide_bodies must list 'moon' and/or 'sun', got %r" % self.tide_bodies)
            if not self.tide_step_hours > 0:
                raise ValueError('tide_step_hours must be positive, got %r' % self.tide_step_hours)
            if self.tide_spatial not in ('linear', 'uniform'):
                raise ValueError("tide_spatial must be 'linear' or 'uniform', got %r" % self.tide_spatial)
            if not self.tide_scale >= 0:
                raise ValueError('tide_scale must be >= 0, got %r' % self.tide_scale)
            if not 0 <= self.poisson_ratio < 0.5:
                raise ValueError('poisson_ratio must be in [0, 0.5), got %r' % self.poisson_ratio)
            if not self.friction_coefficient >= 0:
                raise ValueError('friction_coefficient must be >= 0, got %r' % self.friction_coefficient)
            if not -90 <= self.centre_latitude_deg <= 90:
                raise ValueError('centre_latitude_deg must be in [-90, 90], got %r' % self.centre_latitude_deg)
        if not self.spinup_years >= 0:
            raise ValueError('spinup_years must be >= 0, got %r' % self.spinup_years)
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
        if self.spinup_years >= self.simulation_years:
            warnings.append('spin-up of %g years covers the whole simulation: no events for the statistics'
                            % self.spinup_years)
        if self.sample_interval_years > self.simulation_years:
            warnings.append('sample interval longer than the simulation')
        return warnings
