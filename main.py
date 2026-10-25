"""
Runs the Burridge-Knopoff simulation with the parameters of config.py:
simulates, saves the data to output/data and the plots to output/plots.

    python main.py                 # defaults from config.py
    python main.py --years 20      # override some of them (see --help)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works from any directory

import config  # noqa: E402
from burridge_knopoff.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(config=config.CONFIG))
