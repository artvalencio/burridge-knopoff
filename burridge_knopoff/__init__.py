"""
Burridge-Knopoff earthquake simulation with a gravity sensor array.

    from burridge_knopoff import SimulationConfig, run
    result = run(SimulationConfig(n_blocks_x=50, n_blocks_y=50))

Modules
-------
config         SimulationConfig: all parameters, with units in their names
grid           index conventions, block and sensor coordinates
model          the spring-block model: forces, loading, event-driven slip dynamics
gravity        gravitational potential changes at the sensors
events         event catalogue, seismic moment and moment magnitude
simulation     run(), run_ensemble() and SimulationResult
output         save_results(), load_results() (NPZ and CSV)
visualization  plots and animation
cli            command-line interface (python main.py / burridge-knopoff)
"""
from .config import SimulationConfig
from .output import load_results, save_results
from .simulation import SimulationResult, run, run_ensemble

__version__ = '2.0.0'
__all__ = ['SimulationConfig', 'SimulationResult', 'run', 'run_ensemble', 'save_results', 'load_results']
