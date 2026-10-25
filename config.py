"""
Default parameters for `python main.py`. Edit the values below; command-line
options (python main.py --help) override some of them for a single run.

Units are in the names (SI; schedule in years). The meaning of every field is
documented in burridge_knopoff/config.py (help(SimulationConfig)).
"""
from burridge_knopoff.config import SimulationConfig

CONFIG = SimulationConfig(
    # grid: 100 x 100 blocks, 2 km apart (a 200 km x 200 km fault patch)
    n_blocks_x=100,
    n_blocks_y=100,
    block_spacing_m=2000.0,
    seismogenic_thickness_m=10_000.0,   # with the density, gives the block mass
    rock_density_kg_m3=2700.0,

    # springs, derived from the geometry: coupling = shear modulus x thickness (3e14 N/m),
    # loading = shear modulus x block area / loading distance (6e12 N/m); set
    # coupling_stiffness_n_m / loading_stiffness_n_m to override them
    loading_distance_m=20_000.0,        # depth at which the plate loads the fault

    # friction (stress x block area = force) and its heterogeneity
    static_friction_stress_pa=10e6,
    dynamic_friction_stress_pa=7e6,
    heterogeneity=0.5,                  # strength excess spread: uniform in [1 - h, 1 + h]
    redraw_thresholds=True,

    # loading and seismic moment
    plate_velocity_m_per_year=0.09,     # Pacific plate at the Japan Trench, ~9 cm/year
    shear_modulus_pa=3e10,

    # gravity sensors, centred over the blocks
    n_sensors_x=10,
    n_sensors_y=10,
    sensor_height_m=5000.0,

    # schedule
    simulation_years=100.0,
    sample_interval_years=0.01,         # sensor time-series
    snapshot_interval_years=1.0,        # stored block displacement fields
    steps_per_period=100,               # accuracy of the slip dynamics
    spinup_years=20.0,                  # left out of the magnitude statistics
    seed=0,
)
