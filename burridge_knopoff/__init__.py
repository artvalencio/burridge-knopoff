"""
Burridge-Knopoff earthquake simulation with a gravity sensor array.

Modules
-------
config        SimulationConfig: all parameters, with units in their names
grid          index conventions, block and sensor coordinates
"""
from .config import SimulationConfig

__version__ = '2.0.0.dev0'
__all__ = ['SimulationConfig']
