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

from . import analysis, output, preseismic, rupture, visualization
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
    p.add_argument('--spinup', type=float, default=defaults.spinup_years,
                   help='years left out of the magnitude statistics')
    p.add_argument('--weakening', type=float, nargs=2, metavar=('V_W', 'DROP'), default=None,
                   help='velocity-weakening friction: weakening velocity (m/s) and onset drop (0-1]')
    p.add_argument('--range', type=int, default=defaults.interaction_range_blocks, dest='interaction_range',
                   help='coupling range in blocks (1: nearest neighbours; >1: long-range 1/r^3)')
    p.add_argument('--tides', action='store_true', default=defaults.tides,
                   help='Moon and Sun tidal forcing (time between events stepped every --tide-step hours)')
    p.add_argument('--tide-scale', type=float, default=defaults.tide_scale, help='tide amplification factor')
    p.add_argument('--tide-bodies', nargs='+', choices=('moon', 'sun'), default=defaults.tide_bodies.split(','),
                   help='bodies raising the tide')
    p.add_argument('--tide-step', type=float, default=defaults.tide_step_hours, metavar='HOURS',
                   help='largest step between tidal checks (hours)')
    p.add_argument('--preseismic', nargs='?', const='largest', default=None, metavar='EVENT',
                   help="animate the gravity before an event ('largest' after the spin-up, or a catalogue index)")
    p.add_argument('--preseismic-years', type=float, default=1.0, help='length of the pre-seismic window')
    p.add_argument('--frame-hours', type=float, default=1.0, help='time between pre-seismic frames')
    p.add_argument('--tide-removal', choices=('model', 'fit'), default='model',
                   help='how tides are removed from the pre-seismic gravity (runs with tides)')
    p.add_argument('--gravity-model', choices=('point-mass', 'dislocation'), default=defaults.gravity_model,
                   help="gravity of the blocks as point masses, or of fault slip in a half-space (Okubo 1992, "
                        "with the interseismic back-slip trend)")
    p.add_argument('--fault-depth', type=float, default=defaults.fault_top_depth_m, metavar='METRES',
                   help='depth of the top edge of the fault (dislocation model)')
    p.add_argument('--preslip', action='store_true', default=defaults.preslip,
                   help='precursory creep of the blocks before they fail (rate-and-state nucleation)')
    p.add_argument('--rs-a-sigma', type=float, default=defaults.rs_a_sigma_pa, metavar='PA',
                   help='a sigma of the creep law (Pa)')
    p.add_argument('--rs-dc', type=float, default=defaults.rs_dc_m, metavar='METRES',
                   help='characteristic slip distance D_c of the creep law')
    p.add_argument('--rupture', nargs='?', const='largest', default=None, metavar='EVENT',
                   help='displacement maps and animation of an event (default: the largest after the spin-up)')
    p.add_argument('--rupture-frames', type=int, default=60, help='frames of the rupture animation')
    p.add_argument('--output', default='output', help='output directory (data/ and plots/ inside)')
    p.add_argument('--stream', action='store_true', help='write series to disk while simulating (low memory)')
    p.add_argument('--block-csv', action='store_true', help='also write the block displacements as CSV')
    p.add_argument('--no-plots', action='store_true', help='skip the plots and animation')
    p.add_argument('--no-progress', action='store_true', help='hide the progress bar')
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
    stats = analysis.gutenberg_richter(result)
    if stats['b'] is None:
        lines.append('  b-value: too few events after the spin-up (%g years)' % stats['spinup_years'])
    else:
        est = stats['b']
        lines.append('  b = %.2f +/- %.2f for Mw >= Mc = %.1f (%d events after the spin-up %g years)'
                     % (est.b, est.sigma, stats['mc'], est.n, stats['spinup_years']))
        lines.append('  local b (1-unit windows): ' + ', '.join('%.1f-%.1f: %.2f' % (w.mc, w.upper, w.b)
                                                               for w in stats['local']))
    return '\n'.join(lines)


def _preseismic(result, args, data_dir):
    try:
        k = preseismic.largest_event(result) if args.preseismic == 'largest' else int(args.preseismic)
        event = result.events[k]
        print('[+]   Pre-seismic window of event %d (Mw %.2f at %.2f years): replaying...'
              % (k, event.magnitude, event.start_years), flush=True)
        window = preseismic.replay(result, k, window_years=args.preseismic_years, frame_hours=args.frame_hours)
        os.makedirs(data_dir, exist_ok=True)
        window.save(os.path.join(data_dir, 'preseismic_event_%d.npz' % k))
        plot_dir = os.path.join(args.output, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        path = visualization.preseismic_animation(window, os.path.join(plot_dir, 'preseismic_event_%d.mp4' % k),
                                                  tide_removal=args.tide_removal)
        print('      %d frames written to %s' % (len(window.frame_times_s), path))
        return 0
    except Exception as error:
        print('      ERROR in the pre-seismic animation: %s' % error, file=sys.stderr)
        return 1


def _rupture(result, args, data_dir):
    try:
        k = preseismic.largest_event(result) if args.rupture == 'largest' else int(args.rupture)
        event = result.events[k]
        print('[+]   Rupture of event %d (Mw %.2f at %.2f years): replaying...'
              % (k, event.magnitude, event.start_years), flush=True)
        rup = rupture.replay(result, k, n_frames=args.rupture_frames)
        os.makedirs(data_dir, exist_ok=True)
        rup.save(os.path.join(data_dir, 'rupture_event_%d.npz' % k))
        plot_dir = os.path.join(args.output, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        still = visualization.displacement_map(rup, os.path.join(plot_dir, 'rupture_event_%d.png' % k))
        movie = visualization.rupture_animation(rup, os.path.join(plot_dir, 'rupture_event_%d.mp4' % k))
        print('      %s and %s (%d frames)' % (still, movie, len(rup.frame_times_s)))
        return 0
    except Exception as error:
        print('      ERROR in the rupture maps: %s' % error, file=sys.stderr)
        return 1


def main(argv=None, config=None):
    defaults = config if config is not None else SimulationConfig()
    args = _parser(defaults).parse_args(argv)
    try:
        cfg = dataclasses.replace(defaults, n_blocks_x=args.nx, n_blocks_y=args.ny, n_sensors_x=args.sensors[0],
                                  n_sensors_y=args.sensors[1], simulation_years=args.years,
                                  sample_interval_years=args.sample_interval,
                                  snapshot_interval_years=args.snapshot_interval, seed=args.seed,
                                  spinup_years=args.spinup, interaction_range_blocks=args.interaction_range,
                                  tides=args.tides, tide_scale=args.tide_scale,
                                  tide_bodies=','.join(args.tide_bodies), tide_step_hours=args.tide_step,
                                  gravity_model=args.gravity_model, fault_top_depth_m=args.fault_depth,
                                  preslip=args.preslip, rs_a_sigma_pa=args.rs_a_sigma, rs_dc_m=args.rs_dc,
                                  **({} if args.weakening is None else
                                     dict(weakening_velocity_m_s=args.weakening[0], onset_drop=args.weakening[1])))
        cfg.validate()
    except ValueError as error:
        print('Invalid parameters: %s' % error, file=sys.stderr)
        return 2
    data_dir = os.path.join(args.output, 'data')
    status = 0
    print('[1/4] %d x %d blocks, %d x %d sensors, %g years (seed %d)'
          % (cfg.n_blocks_x, cfg.n_blocks_y, cfg.n_sensors_x, cfg.n_sensors_y, cfg.simulation_years, cfg.seed))
    friction = ('Coulomb' if cfg.weakening_velocity_m_s is None else
                'velocity weakening (v_w = %g m/s, onset drop %g)' % (cfg.weakening_velocity_m_s, cfg.onset_drop))
    print('      friction: %s; coupling range: %d block(s)' % (friction, cfg.interaction_range_blocks))
    if cfg.tides:
        print('      tides: %s, scale %g, steps of at most %g h' % (cfg.tide_bodies.replace(',', ' + '),
                                                                 cfg.tide_scale, cfg.tide_step_hours))
    if cfg.gravity_model == 'dislocation':
        print('      gravity: fault slip in a half-space (dip %g, rake %g, top at %g km)'
              % (cfg.fault_dip_deg, cfg.fault_rake_deg, cfg.fault_top_depth_m / 1e3))
    if cfg.preslip:
        print('      precursory creep: a sigma %g kPa, b/a %g, D_c %g um'
              % (cfg.rs_a_sigma_pa / 1e3, cfg.rs_b_over_a, cfg.rs_dc_m * 1e6))
    for warning in cfg.check_parameters():
        print('      warning: ' + warning)
    print('[2/4] Simulating...', flush=True)
    start = time.perf_counter()
    result = run(cfg, store_dir=data_dir if args.stream else None, progress=not args.no_progress)
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
    if args.preseismic is not None:
        status = max(status, _preseismic(result, args, data_dir))
    if args.rupture is not None:
        status = max(status, _rupture(result, args, data_dir))
    print(_summary(result))
    print('Finished successfully.' if status == 0 else 'Finished with errors (see above).')
    return status


if __name__ == '__main__':
    sys.exit(main())
