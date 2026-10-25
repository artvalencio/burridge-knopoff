# Burridge-Knopoff Earthquake Simulation with Gravity Sensors

A Python implementation of a two-dimensional Burridge-Knopoff spring-block
earthquake model, combined with an array of gravity sensors that record the
change in gravitational potential caused by the motion of the blocks.

> **Status: research prototype under repair.** The code runs, but the current
> physics core has known errors and the default configuration cannot finish
> in practice. Read [Status and known issues](#status-and-known-issues) before
> using any result.

## Overview

- A grid of `N_BLOCKS_X × N_BLOCKS_Y` blocks (default 100 × 100) on the fault
  plane z = 0, each connected to its four nearest neighbours by linear springs
- Friction along the plate-motion direction (x), with a static branch and a
  rate-and-state branch, plus linear damping
- A `N_SENSORS_X × N_SENSORS_Y` array of gravity sensors (default 10 × 10) at a
  height above the fault, measuring the potential Φ = −G Σ m/r of the blocks
- Integration with SciPy's `solve_ivp` (RK45), sampling every `DT_SAMPLE` years
- Event detection from the sampled block velocities
- CSV export and plots/animations of the potential changes

## Status and known issues

The following problems were found by running the code on small grids. They
are being fixed test-first; until then, treat outputs as illustrative only.

**Physics**
- The plate does not load the blocks through springs: the plate's
  *displacement* (m) is subtracted from a block's *velocity* (m/s) in the
  friction law, which acts as a constant push along x.
- The friction law is not the Dieterich-Ruina law (wrong reference
  coefficient, clamped at μ_d, θ evolves with the absolute 3-D velocity), and
  blocks never truly stick.
- Only x carries forces besides the springs; y and z are integrated without
  physics.
- With the default parameters, overcoming static friction needs about 147 m of
  spring stretch, about 1600 years at 9 cm/year, so no earthquakes are expected
  within the default 100 years even after the loading is fixed.
- All blocks are identical and start identically. Without heterogeneity the
  model does not produce Gutenberg-Richter statistics.

**Grid indexing**
- Block positions are stored row by row along x, but the force kernel assumes
  the opposite order: on non-square grids springs connect the wrong blocks.
- Epicentre x and y are swapped in the event catalogue and in the plots.
- The sensor array is offset by half a block spacing relative to the blocks.

**Numerics and events**
- Spring oscillations have periods of about 30 s, so the explicit RK45
  integrator needs on the order of 10⁹ steps for 100 years. A 3 × 3 grid did
  not complete 0.03 years (three samples) in 5 minutes; the default run does
  not finish in practice.
- Events are detected only at the sampling times (every 3.65 days by default),
  so second-long slip events are mostly missed, and an event still running at
  the end is not recorded.
- The reported "magnitude" is log10(peak velocity) + 5, not a moment magnitude.

**Gravity, memory, output**
- `compute_gravity_acceleration` (not used by the pipeline) has the wrong sign.
- Positions and velocities of every block are kept in memory at every sample:
  about 4.8 GB for the default run. The block CSV would have 10⁸ rows (several
  GB) and is built from one Python dict per row, needing tens of GB of RAM.
- Plots fail when all potential changes are zero or of one sign (e.g. the
  first frame), because of the diverging colour normalisation.

## Installation

Requirements:
- Python 3.8 or later
- numpy, scipy, matplotlib, pandas, numba and tqdm (see `requirements.txt`)
- FFmpeg, only for the MP4 animation

```bash
pip install -r requirements.txt
```

FFmpeg: `choco install ffmpeg` (Windows), `sudo apt-get install ffmpeg`
(Linux), `brew install ffmpeg` (macOS), or https://ffmpeg.org/download.html.

## Usage

Run from the repository root (the `src` modules import `config` from there):

```bash
python main.py
```

The script initialises the blocks and sensors, integrates the model, exports
CSV files to `output/csv/` and writes plots and the animation to
`output/animations/`.

Because of the integration cost described above, even very small
configurations (a few blocks, a few hundredths of a year, set in `config.py`)
take many minutes. Fixing this is the first priority of the ongoing rework.

### Configuration

All parameters live in `config.py` (SI units unless stated):

| Group | Parameters |
|---|---|
| Grid | `N_BLOCKS_X`, `N_BLOCKS_Y`, `BLOCK_SPACING` (m), `BLOCK_MASS` (kg) |
| Sensors | `N_SENSORS_X`, `N_SENSORS_Y`, `SENSOR_HEIGHT` (**km**) |
| Time | `SIMULATION_TIME`, `DT_SAMPLE` (**years**) |
| Springs | `SPRING_K_HORIZONTAL` (neighbours along the first grid index), `SPRING_K_VERTICAL` (along the second), N/m |
| Friction | `MU_STATIC`, `MU_DYNAMIC`, `A_FRICTION`, `B_FRICTION`, `DC` (m), `V_CHARACTERISTIC` (m/s) |
| Loading | `PLATE_VELOCITY` (**m/year**), `DAMPING` (kg/s) |
| Solver | `RTOL`, `ATOL`, `MAX_STEP` (s) |
| Events | `VELOCITY_THRESHOLD` (m/s), `MIN_EVENT_DURATION` (s) |
| Output | `OUTPUT_DIR`, `CSV_DIR`, `ANIMATION_DIR` |

`T_INITIAL_EQUILIBRATION` is defined but not used.

## Outputs

| File | Content |
|---|---|
| `output/csv/block_positions_time.csv` | `time, block_id, x, y, z`: every block at every sample (long format; very large) |
| `output/csv/sensor_potential_time.csv` | `time, sensor_id, x_sensor, y_sensor, z_sensor, potential`: potential change in J/kg relative to the initial configuration |
| `output/csv/earthquake_events.csv` | `time, duration, magnitude, max_velocity, epicenter_i, epicenter_j, epicenter_x, epicenter_y` (see the known issues about magnitude and epicentres) |
| `output/animations/gravity_potential_animation.mp4` | map of the potential change over time; stars mark events within one year |
| `output/animations/event_N_comparison.png` | potential change before, during and after the first three events |
| `output/animations/magnitude_frequency.png` | histogram of the event magnitudes |

Times are in years in all outputs except event durations (seconds).

## Project structure

```
burridge-knopoff/
├── config.py               # parameters and block/sensor coordinate grids
├── main.py                 # runs simulation, export and visualisation
├── requirements.txt
└── src/
    ├── burridge_knopoff.py # state vector, forces (numba kernel), friction state
    ├── solver.py           # time integration, sampling, event detection
    ├── gravity_sensors.py  # gravitational potential at the sensors
    ├── data_export.py      # CSV export
    └── visualization.py    # animation and plots
```

### State vector

The ODE state for N blocks is

```
[x1, y1, z1, ..., xN, yN, zN,  vx1, vy1, vz1, ..., vxN, vyN, vzN,  θ1, ..., θN]
```

(positions in m, velocities in m/s, rate-and-state variables θ in s): 7N values.

## Scientific background

### Burridge-Knopoff model

The Burridge-Knopoff model (Burridge & Knopoff, 1967) represents a fault as
blocks connected to each other by coupling springs and to a slowly moving
plate by loading ("leaf") springs, sliding with friction on a fixed surface.
Stress builds up while the blocks stick; when the force on a block exceeds
static friction it slips, possibly triggering its neighbours: an earthquake.
With heterogeneity or velocity-weakening friction, such models produce broad,
power-law-like event size distributions (Carlson & Langer, 1989).

### Rate-and-state friction

In the Dieterich-Ruina formulation, with slip velocity v and state θ,

```
μ = μ₀ + a·ln(v/v*) + b·ln(v*·θ/D_c)
dθ/dt = 1 − v·θ/D_c                    (aging law)
```

At steady state θ = D_c/v and μ_ss = μ₀ + (a − b)·ln(v/v*): friction weakens
with velocity when b > a, which allows stick-slip instabilities.

### Gravitational potential

The sensors record

```
Φ(r) = −G Σᵢ mᵢ / |r − rᵢ|
```

and the outputs are ΔΦ = Φ(t) − Φ(0). The corresponding acceleration is
g = −∇Φ = −G Σᵢ mᵢ (r − rᵢ)/|r − rᵢ|³, pointing towards the masses.

### Moment magnitude

A physically meaningful magnitude uses the seismic moment
M₀ = μ_shear · A · s̄ (shear modulus, rupture area, mean slip):
Mw = (2/3)(log₁₀ M₀ − 9.1), with M₀ in N·m.

## References

1. Burridge, R., & Knopoff, L. (1967). Model and theoretical seismicity. *Bulletin of the Seismological Society of America*, 57(3), 341-371.
2. Carlson, J. M., & Langer, J. S. (1989). Mechanical model of an earthquake fault. *Physical Review A*, 40(11), 6470-6484.
3. Dieterich, J. H. (1979). Modeling of rock friction: 1. Experimental results and constitutive equations. *Journal of Geophysical Research*, 84(B5), 2161-2168.
4. Ruina, A. (1983). Slip instability and state variable friction laws. *Journal of Geophysical Research*, 88(B12), 10359-10370.
5. Scholz, C. H. (2002). *The Mechanics of Earthquakes and Faulting* (2nd ed.). Cambridge University Press.

## License

This project is provided for educational and research purposes.

---

**Note**: this is a simplified model for research and education. Real
earthquake forecasting requires far more complex models (3-D geometry,
heterogeneous materials, fluids and more).
