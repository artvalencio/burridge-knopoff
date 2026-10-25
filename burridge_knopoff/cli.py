"""
Command-line interface: simulate, save the results, plot them.

    python main.py [options]          (from the repository; defaults from config.py)
    burridge-knopoff [options]        (after pip install; defaults from SimulationConfig)

Exit status: 0 on success, 1 if saving or plotting failed (the other steps
still run), 2 for invalid parameters.
"""
import argparse
import dataclasses
import os
import sys
import time

from . import output, visualization
from .config import SimulationConfig
from .simulation import run


def _parser(defaults):
    p = argparse.ArgumentParser(description='Burridge-Knopoff earthquake simulation with gravity sensors.')
    p.add_argument('--nx', type=int, default=defaults.n_blocks_x, help='blocks along x (plate motion)')
    p.add_argument('--ny', type=int, default=defaults.n_blocks_y, help='blocks along y')
    p.add_argument('--sensors', type=int, nargs=2, metavar=('NX', 'NY'),
                   default=(defaults.n_sensors_x, defaults.n_sensors_y), help='sensor grid')
    p.add_argument('--years', type=float, default=defaults.simulation_years, help='simulated time')
    p.add_argument('--sample-interval', type=float, default=defaults.sample_interval_years,
                   help='sensor sampling interval (years)')
    p.add_argument('--snapshot-interval', type=float, default=defaults.snapshot_interval_years,
                   help='interval of stored block displacement fields (years)')
    p.add_argument('--seed', type=int, default=defaults.seed, help='random seed of the heterogeneity')
    p.add_argument('--output', default='output', help='output directory (data/ and plots/ inside)')
    p.add_argument('--stream', action='store_true', help='write series to disk while simulating (low memory)')
    p.add_argument('--block-csv', action='store_true', help='also write the block displacements as CSV')
    p.add_argument('--no-plots', action='store_true', help='skip the plots and animation')
    p.add_argument('--max-frames', type=int, default=300, help='frames of the animation')
    return p


def _summary(result):
    events = result.events
    lines = ['Simulated %.2f years: %d samples, %d events' % (result.times_years[-1], len(result.times_years),
                                                            len(events)),
             'Events: %d' % len(events)]
    if events:
        mags = [e.magnitude for e in events]
        big = max(events, key=lambda e: e.magnitude)
        x, y = big.hypocentre_xy(result.config)
        lines += ['  Mw %.2f to %.2f' % (min(mags), max(mags)),
                  '  largest: Mw %.2f at t = %.2f years, hypocentre (%.1f, %.1f) km, %d blocks slipped'
                  % (big.magnitude, big.start_years, x / 1000, y / 1000, big.n_slipped)]
    return '\n'.join(lines)


def main(argv=None, config=None):
    defaults = config if config is not None else SimulationConfig()
    args = _parser(defaults).parse_args(argv)
    try:
        cfg = dataclasses.replace(defaults, n_blocks_x=args.nx, n_blocks_y=args.ny, n_sensors_x=args.sensors[0],
                                  n_sensors_y=args.sensors[1], simulation_years=args.years,
                                  sample_interval_years=args.sample_interval,
                                  snapshot_interval_years=args.snapshot_interval, seed=args.seed).validate()
    except ValueError as error:
        print('Invalid parameters: %s' % error, file=sys.stderr)
        return 2
    data_dir = os.path.join(args.output, 'data')
    status = 0
    print('[1/4] %d x %d blocks, %d x %d sensors, %g years (seed %d)'
          % (cfg.n_blocks_x, cfg.n_blocks_y, cfg.n_sensors_x, cfg.n_sensors_y, cfg.simulation_years, cfg.seed))
    for warning in cfg.check_parameters():
        print('      warning: ' + warning)
    print('[2/4] Simulating...', flush=True)
    start = time.perf_counter()
    result = run(cfg, store_dir=data_dir if args.stream else None)
    print('      done in %.1f s' % (time.perf_counter() - start))
    print('[3/4] Saving data to %s' % data_dir, flush=True)
    try:
        output.save_results(result, data_dir, block_csv=args.block_csv)
    except Exception as error:  # report and go on with the plots
        print('      ERROR while saving: %s' % error, file=sys.stderr)
        status = 1
    if args.no_plots:
        print('[4/4] Plots skipped')
    else:
        plot_dir = os.path.join(args.output, 'plots')
        print('[4/4] Plotting to %s' % plot_dir, flush=True)
        try:
            visualization.create_all(result, plot_dir, max_frames=args.max_frames)
        except Exception as error:
            print('      ERROR while plotting: %s' % error, file=sys.stderr)
            status = 1
    print(_summary(result))
    print('Finished successfully.' if status == 0 else 'Finished with errors (see above).')
    return status


if __name__ == '__main__':
    sys.exit(main())
