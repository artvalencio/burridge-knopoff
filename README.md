# Burridge-Knopoff Earthquake Simulation with Gravity Sensors

A comprehensive Python implementation of the Burridge-Knopoff earthquake model with a gravity sensor array for measuring gravitational potential changes during seismic events.

## Overview

This simulation models earthquake dynamics using:
- **100×100 grid** of blocks connected by springs (10,000 blocks total)
- **Rate-and-state friction** for realistic stick-slip behavior
- **10×10 gravity sensor array** positioned at height h above the fault
- **Realistic parameters** based on Japanese Trench seismicity
- **Comprehensive data export** to CSV files
- **Animations** showing gravitational potential changes before/after seismic events

## Features

### Physical Model
- 2D Burridge-Knopoff system with spring-block dynamics
- Rate-and-state friction laws (Dieterich-Ruina)
- Plate motion driving stress accumulation
- Realistic earthquake parameters (Japanese subduction zone)

### Gravity Sensing
- 10×10 array of sensors at configurable height
- Gravitational potential calculations using Newton's law
- Tracks potential changes due to block displacements
- Optimized with Numba for fast computation

### Output
- **CSV Files**:
  - `block_positions_time.csv`: Position (x, y, z) of each block over time
  - `sensor_potential_time.csv`: Gravitational potential at each sensor over time
  - `earthquake_events.csv`: Detected seismic events with magnitude, location, time
  
- **Visualizations**:
  - Animated 2D map of gravitational potential changes
  - Before/after comparison plots for major events
  - Magnitude-frequency distribution (Gutenberg-Richter)

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (required for animation generation)

### Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- numba >= 0.57.0
- tqdm >= 4.65.0

## Usage

### Basic Execution

Run the simulation with default parameters:

```bash
python main.py
```

The simulation will:
1. Initialize the 100×100 block system
2. Set up the 10×10 gravity sensor array
3. Run the simulation for 100 years (configurable)
4. Export data to CSV files in `output/csv/`
5. Generate animations in `output/animations/`

### Configuration

Edit `config.py` to customize simulation parameters:

```python
# Grid size
N_BLOCKS_X = 100
N_BLOCKS_Y = 100

# Sensor array
N_SENSORS_X = 10
N_SENSORS_Y = 10
SENSOR_HEIGHT = 5.0  # km above fault

# Simulation duration
SIMULATION_TIME = 100.0  # years
DT_SAMPLE = 0.01  # sampling interval (years)

# Physical parameters
BLOCK_MASS = 1e12  # kg
BLOCK_SPACING = 2000.0  # meters (2 km)
PLATE_VELOCITY = 0.09  # m/year (9 cm/year)

# Friction parameters
MU_STATIC = 0.6
MU_DYNAMIC = 0.4
```

## Project Structure

```
burridge-knopoff/
├── config.py                   # Configuration parameters
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── burridge_knopoff.py    # Core physics engine
│   ├── gravity_sensors.py      # Gravity sensor array
│   ├── solver.py               # Numerical integration
│   ├── data_export.py          # CSV export functionality
│   └── visualization.py        # Plotting and animation
└── output/
    ├── csv/                    # Exported CSV files
    └── animations/             # Generated animations
```

## Physical Parameters

### Japanese Trench Parameters

The default configuration uses realistic values for the Japan Trench subduction zone:

- **Plate velocity**: 9 cm/year (Pacific plate subduction)
- **Block spacing**: 2 km (crustal segment size)
- **Seismogenic depth**: 10-30 km
- **Friction coefficients**: μ_s = 0.6, μ_d = 0.4
- **Spring stiffness**: 10^10 N/m

### Gravity Sensor Configuration

- **Sensor height**: 5 km above fault plane
- **Array coverage**: Spans entire fault system
- **Measurement**: Gravitational potential changes (J/kg)
- **Expected magnitude**: 10^-8 to 10^-6 m²/s² for major events

## Output Files

### CSV Files

1. **block_positions_time.csv**
   - Columns: `time, block_id, x, y, z`
   - Contains position of every block at each time step
   - File size: ~hundreds of MB for 100-year simulation

2. **sensor_potential_time.csv**
   - Columns: `time, sensor_id, x_sensor, y_sensor, z_sensor, potential`
   - Gravitational potential at each sensor over time
   - File size: ~10-50 MB

3. **earthquake_events.csv**
   - Columns: `time, duration, magnitude, max_velocity, epicenter_i, epicenter_j, epicenter_x, epicenter_y`
   - Detected seismic events with characteristics
   - File size: Small (~KB to few MB)

### Visualizations

1. **gravity_potential_animation.mp4**
   - Time-lapse of gravitational potential changes
   - Yellow stars mark earthquake epicenters
   - Duration: Proportional to simulation time

2. **event_X_comparison.png**
   - Before/during/after comparison for major events
   - Shows spatial distribution of potential changes
   - Multiple files for different events

3. **magnitude_frequency.png**
   - Histogram of earthquake magnitudes
   - Tests Gutenberg-Richter relationship

## Performance Considerations

### Computational Requirements

- **Memory**: ~2-4 GB RAM for 100×100 grid
- **Runtime**: 
  - ~10-30 minutes for 100 years (depends on CPU)
  - Optimized with Numba JIT compilation
  
### Optimization Tips

1. **Reduce grid size** for faster testing:
   ```python
   N_BLOCKS_X = 50
   N_BLOCKS_Y = 50
   ```

2. **Increase sampling interval**:
   ```python
   DT_SAMPLE = 0.1  # years (less frequent sampling)
   ```

3. **Shorten simulation time**:
   ```python
   SIMULATION_TIME = 50.0  # years
   ```

4. **Disable animations** if FFmpeg is unavailable (comment out visualization calls in `main.py`)

## Scientific Background

### Burridge-Knopoff Model

The Burridge-Knopoff model represents the Earth's crust as a grid of blocks connected by springs, sliding on a moving plate. Key features:

- **Stick-slip dynamics**: Blocks stick due to friction, then slip when stress exceeds threshold
- **Earthquake generation**: Slip events propagate through the grid
- **Statistical properties**: Reproduces Gutenberg-Richter frequency-magnitude scaling

### Rate-and-State Friction

The simulation uses Dieterich-Ruina rate-and-state friction:

```
μ = μ_d + a·ln(v/v*) + b·ln(θv*/D_c)
dθ/dt = 1 - vθ/D_c
```

Where:
- μ: friction coefficient
- v: slip velocity
- θ: state variable
- a, b: empirical parameters (b > a for instability)
- D_c: characteristic slip distance

### Gravitational Potential

Sensors measure changes in gravitational potential:

```
Φ(r) = -G Σ(m_i / |r - r_i|)
```

Where:
- G: gravitational constant
- m_i: block masses
- r: sensor position
- r_i: block positions

## Validation

Expected behaviors:
- ✓ Periodic earthquake events (pseudo-periodic recurrence)
- ✓ Gutenberg-Richter magnitude-frequency relation
- ✓ Spatial clustering of aftershocks
- ✓ Gravity changes of 10^-8 to 10^-6 m²/s² for large events
- ✓ Energy conservation (within numerical tolerance)

## Troubleshooting

### Common Issues

1. **"Integration failed"**
   - Solution: Reduce `MAX_STEP` in config.py
   - Or increase tolerance (RTOL, ATOL)

2. **"FFmpeg not found"**
   - Solution: Install FFmpeg or disable animations

3. **Out of memory**
   - Solution: Reduce grid size or sampling frequency

4. **Numba compilation warnings**
   - Usually safe to ignore
   - Ensure numpy arrays have correct dtypes

### Getting Help

For issues or questions:
- Check parameter values in `config.py`
- Review error messages in console output
- Verify all dependencies are installed correctly

## References

1. Burridge, R., & Knopoff, L. (1967). Model and theoretical seismicity. *Bulletin of the Seismological Society of America*, 57(3), 341-371.

2. Dieterich, J. H. (1979). Modeling of rock friction: 1. Experimental results and constitutive equations. *Journal of Geophysical Research*, 84(B5), 2161-2168.

3. Ruina, A. (1983). Slip instability and state variable friction laws. *Journal of Geophysical Research*, 88(B12), 10359-10370.

4. Scholz, C. H. (2002). *The Mechanics of Earthquakes and Faulting* (2nd ed.). Cambridge University Press.

## License

This project is provided for educational and research purposes.

## Author

Implemented with realistic Japanese Trench parameters for earthquake simulation research.

---

**Note**: This is a simplified model for research and educational purposes. Real earthquake prediction requires significantly more complex models incorporating 3D geometry, heterogeneous material properties, fluid effects, and other factors.
