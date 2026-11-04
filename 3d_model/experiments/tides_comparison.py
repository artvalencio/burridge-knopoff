"""
Ensembles with and without Earth tides (plan, section 4).

    python 3d_model/experiments/tides_comparison.py run --size dev --seeds 0-9 --conditions A B C --jobs 4
    python 3d_model/experiments/tides_comparison.py report --size dev

Conditions (plan 4.1):
    A   no tides, exact jumps (reference)
    B   tides on with tide_scale = 0 (control: hourly stepping alone)
    C   Moon + Sun, scale 1 (main comparison, C vs A)
    Dm  Moon only;  Ds  Sun only
    E10, E100, E1000   Moon + Sun amplified 10x, 100x, 1000x
    W-A, W-C   as A and C with velocity-weakening friction (0.1 m/s, onset drop 0.3)

Sizes: 'dev' is a 100 km x 100 km grid (50 x 50 blocks, 10 x 10 sensors), 'full' the
default 500 km grid (250 x 250 blocks, 25 x 25 sensors). Runs last 120 years, the first
20 excluded. Each run stores a compact summary (results/<size>/<condition>_seed<k>.npz):
the catalogue, the gravity field at the end of the spin-up and at the end (for the
cumulative co-seismic change) and, with tides, the tidal Coulomb series.
Only the 2-D model is available until the 3-D model (plan, section 3) exists.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root

from burridge_knopoff import analysis, tide_analysis                  # noqa: E402
from burridge_knopoff.config import SimulationConfig                   # noqa: E402
from burridge_knopoff.simulation import run                            # noqa: E402

RESULTS = os.path.join(HERE, 'results')
SIZES = {
    'dev': dict(n_blocks_x=50, n_blocks_y=50, n_sensors_x=10, n_sensors_y=10),
    'full': dict(),
}
COMMON = dict(simulation_years=120.0, spinup_years=20.0, sample_interval_years=1.0, snapshot_interval_years=120.0)
WEAKENING = dict(weakening_velocity_m_s=0.1, onset_drop=0.3)
CONDITIONS = {
    'A': dict(),
    'B': dict(tides=True, tide_scale=0.0),
    'C': dict(tides=True),
    'Dm': dict(tides=True, tide_bodies='moon'),
    'Ds': dict(tides=True, tide_bodies='sun'),
    'E10': dict(tides=True, tide_scale=10.0),
    'E100': dict(tides=True, tide_scale=100.0),
    'E1000': dict(tides=True, tide_scale=1000.0),
    'W-A': dict(**WEAKENING),
    'W-C': dict(tides=True, **WEAKENING),
}
THRESHOLDS_UGAL = (0.1, 1.0, 10.0)


def config(size, condition, seed):
    return SimulationConfig(**SIZES[size], **COMMON, **CONDITIONS[condition], seed=seed)


def _path(size, condition, seed):
    return os.path.join(RESULTS, size, '%s_seed%d.npz' % (condition, seed))


def _one(task):
    size, condition, seed = task
    cfg = config(size, condition, seed)
    start = time.perf_counter()
    res = run(cfg)
    elapsed = time.perf_counter() - start
    k_spin = int(np.searchsorted(res.times_years, cfg.spinup_years))
    out = dict(config=json.dumps(dataclasses.asdict(cfg)), catalogue=res.catalogue, elapsed_s=elapsed,
               gravity_spinup=np.asarray(res.gravity[k_spin]), gravity_end=np.asarray(res.gravity[-1]))
    if res.tide_coulomb_pa is not None:
        out.update(tide_times_s=res.tide_times_s[::3], tide_coulomb_pa=res.tide_coulomb_pa[::3])
    os.makedirs(os.path.dirname(_path(size, condition, seed)), exist_ok=True)
    np.savez_compressed(_path(size, condition, seed), **out)
    return condition, seed, elapsed, len(res.events)


def _seeds(text):
    seeds = []
    for part in text.split(','):
        lo, __, hi = part.partition('-')
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def run_ensembles(size, conditions, seeds, jobs, overwrite=False):
    tasks = [(size, c, s) for c in conditions for s in seeds if overwrite or not os.path.exists(_path(size, c, s))]
    print('%d runs to do (%s, conditions %s, seeds %s)' % (len(tasks), size, ' '.join(conditions), seeds), flush=True)
    if jobs == 1:
        done = map(_one, tasks)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=jobs)
        done = pool.map(_one, tasks)
    for condition, seed, elapsed, n in done:
        print('  %-6s seed %3d: %7d events, %6.1f s' % (condition, seed, n, elapsed), flush=True)


# ---- analysis ------------------------------------------------------------------------
class _Result:
    """The parts of a SimulationResult the analysis tools need, from a summary file."""

    def __init__(self, path):
        from burridge_knopoff import events as ev
        with np.load(path) as d:
            self.config = SimulationConfig(**json.loads(str(d['config'])))
            self.table = d['catalogue']
            self.gravity_change = d['gravity_end'] - d['gravity_spinup']
            self.elapsed_s = float(d['elapsed_s'])
        fields = [f.name for f in dataclasses.fields(ev.Event)]
        self.events = [ev.Event(**{n: row[n].item() for n in fields if n in self.table.dtype.names})
                       for row in self.table]

    @property
    def catalogue(self):
        return self.table


def summarise(res, series=None):
    cfg = res.config
    cat = res.catalogue
    keep = cat['start_years'] >= cfg.spinup_years
    c = cat[keep]
    years = cfg.simulation_years - cfg.spinup_years
    gr = analysis.gutenberg_richter(res)
    mc = gr['mc'] if gr['mc'] is not None else np.nan
    out = dict(n_events=int(keep.sum()), n_above_mc=int(np.sum(c['magnitude'] >= mc)),
               n_mw7=int(np.sum(c['magnitude'] >= 7.0)), b=gr['b'].b if gr['b'] is not None else np.nan,
               b_sigma=gr['b'].sigma if gr['b'] is not None else np.nan, mc=mc,
               moment_rate=float(c['moment_n_m'].sum() / years), elapsed_s=res.elapsed_s)
    dt = np.diff(c['start_years'])
    out['interevent_cv'] = float(dt.std() / dt.mean()) if len(dt) > 1 else np.nan
    for m in np.arange(4.0, 8.5, 0.5):
        out['n_mw_%.1f' % m] = int(np.sum((c['magnitude'] >= m) & (c['magnitude'] < m + 0.5)))
    for th in THRESHOLDS_UGAL:
        out['n_dg_above_%g' % th] = int(np.sum(c['dg_max_abs_ugal'] >= th))
    big = c['magnitude'] >= mc
    if big.sum() > 10:
        out['dg_mw_slope'] = float(np.polyfit(c['magnitude'][big], np.log10(c['dg_max_abs_ugal'][big]), 1)[0])
    ts = tide_analysis.tidal_statistics(res, series=series)
    out['schuster_p'] = ts['schuster_p']
    out['rate_ratio'] = ts['rate']['ratio'].tolist()
    return out


def _bootstrap_ci(values, n=2000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(size, reference='A'):
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = os.path.join(RESULTS, size)
    runs = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith('.npz'):
            condition, seed = name[:-4].rsplit('_seed', 1)
            runs.setdefault(condition, {})[int(seed)] = _Result(os.path.join(folder, name))
    # the tidal series depends only on geometry, epoch and bodies: compute each once
    cache = {}

    def series(cfg):
        if cfg.tide_bodies not in cache:
            cache[cfg.tide_bodies] = tide_analysis.tidal_series(cfg)
        return cache[cfg.tide_bodies]

    summaries = {c: {s: summarise(r, series(r.config)) for s, r in rs.items()} for c, rs in runs.items()}
    keys = ['n_events', 'n_above_mc', 'n_mw7', 'b', 'mc', 'moment_rate', 'interevent_cv', 'schuster_p',
            'n_dg_above_0.1', 'n_dg_above_1', 'n_dg_above_10', 'dg_mw_slope', 'elapsed_s']
    rows = []
    for c in sorted(summaries):
        seeds = sorted(summaries[c])
        row = dict(condition=c, n_seeds=len(seeds))
        for k in keys:
            vals = np.array([summaries[c][s].get(k, np.nan) for s in seeds], dtype=float)
            row[k] = float(np.nanmean(vals))
            row[k + '_ci'] = _bootstrap_ci(vals)
        if c != reference and reference in summaries:
            common = sorted(set(seeds) & set(summaries[reference]))
            if len(common) >= 5:
                for k in ('n_events', 'n_above_mc', 'n_mw7', 'b'):
                    diff = np.array([summaries[c][s][k] - summaries[reference][s][k] for s in common], dtype=float)
                    row['paired_diff_' + k] = float(np.mean(diff))
                    row['paired_diff_%s_ci' % k] = _bootstrap_ci(diff)
                    nonzero = diff[diff != 0]
                    row['wilcoxon_p_' + k] = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else np.nan
                mags = [np.concatenate([runs[x][s].catalogue['magnitude'][runs[x][s].catalogue['start_years'] >= 20]
                                        for s in common]) for x in (reference, c)]
                dgs = [np.concatenate([runs[x][s].catalogue['dg_max_abs_ugal'][runs[x][s].catalogue['start_years'] >= 20]
                                       for s in common]) for x in (reference, c)]
                row['ad_p_magnitude'] = float(stats.anderson_ksamp(mags).pvalue)
                row['ad_p_dg'] = float(stats.anderson_ksamp(dgs).pvalue)
        rows.append(row)
    with open(os.path.join(folder, 'summary.json'), 'w') as fh:
        json.dump(dict(rows=rows, per_run=summaries), fh, indent=1, default=float)
    _table(rows, os.path.join(folder, 'summary.md'))
    _figures(runs, summaries, folder, plt, reference)
    print(open(os.path.join(folder, 'summary.md')).read())


def _fmt(v, ci=None, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '-'
    text = ('%.' + str(digits) + 'g') % v
    if ci is not None and all(np.isfinite(ci)):
        text += ' [%s, %s]' % (('%.' + str(digits) + 'g') % ci[0], ('%.' + str(digits) + 'g') % ci[1])
    return text


def _table(rows, path):
    lines = ['| condition | seeds | events after spin-up | Mw >= 7 | b | Schuster p | events with dg >= 1 uGal '
             '| dg-Mw slope | paired diff. events (vs A) | Wilcoxon p | AD p (Mw) | AD p (dg) | time per run (s) |',
             '|' + '---|' * 13]
    for r in rows:
        lines.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['condition'], r['n_seeds'], _fmt(r['n_events'], r['n_events_ci'], 6), _fmt(r['n_mw7']),
            _fmt(r['b'], r['b_ci']), _fmt(r['schuster_p'], None, 2), _fmt(r['n_dg_above_1'], None, 6),
            _fmt(r['dg_mw_slope']), _fmt(r.get('paired_diff_n_events'), r.get('paired_diff_n_events_ci'), 4),
            _fmt(r.get('wilcoxon_p_n_events'), None, 2), _fmt(r.get('ad_p_magnitude'), None, 2),
            _fmt(r.get('ad_p_dg'), None, 2), _fmt(r['elapsed_s'])))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def _figures(runs, summaries, folder, plt, reference):
    order = [c for c in CONDITIONS if c in summaries]
    # event counts per condition
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, c in enumerate(order):
        vals = [summaries[c][s]['n_events'] for s in summaries[c]]
        ax.plot(np.full(len(vals), i), vals, 'o', alpha=0.5, color='tab:blue')
        lo, hi = _bootstrap_ci(vals)
        ax.errorbar(i, np.mean(vals), yerr=[[np.mean(vals) - lo], [hi - np.mean(vals)]], fmt='s', color='black',
                    capsize=4)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel('events after the spin-up')
    ax.set_title('Event counts per condition (mean and 95% bootstrap CI)')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'event_counts.png'), dpi=120)
    plt.close(fig)
    # magnitude-frequency and co-seismic gravity CCDFs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in [x for x in ('A', 'C', 'E100', 'E1000') if x in runs]:
        cats = [r.catalogue[r.catalogue['start_years'] >= 20] for r in runs[c].values()]
        for ax, key in zip(axes, ('magnitude', 'dg_max_abs_ugal')):
            v = np.sort(np.concatenate([cat[key] for cat in cats]))
            ax.plot(v, 1 - np.arange(len(v)) / len(v), label=c)
    axes[0].set_xlabel('Mw')
    axes[1].set_xlabel('largest co-seismic |dg| over the sensors (uGal)')
    axes[1].set_xscale('log')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylabel('fraction of events >= value')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'magnitude_and_gravity_ccdf.png'), dpi=120)
    plt.close(fig)
    # rate vs tidal stress decile
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in order:
        ratios = np.array([summaries[c][s]['rate_ratio'] for s in summaries[c]])
        ax.plot(np.arange(1, ratios.shape[1] + 1), ratios.mean(axis=0), 'o-', label=c)
    ax.axhline(1, color='grey', lw=0.8)
    ax.set_xlabel('tidal Coulomb stress decile at the origin time')
    ax.set_ylabel('observed / expected events')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'rate_vs_tidal_stress.png'), dpi=120)
    plt.close(fig)
    # cumulative co-seismic gravity change: mean maps and their difference
    if reference in runs and 'C' in runs:
        cfg = next(iter(runs[reference].values())).config
        shape = (cfg.n_sensors_y, cfg.n_sensors_x)
        mean = {c: np.mean([r.gravity_change for r in runs[c].values()], axis=0).reshape(shape) * 1e8
                for c in (reference, 'C')}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (title, m) in zip(axes, [('no tides (A)', mean[reference]), ('tides (C)', mean['C']),
                                         ('C - A', mean['C'] - mean[reference])]):
            lim = np.abs(m).max() or 1
            im = ax.imshow(m, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_title('cumulative co-seismic dg, ' + title)
            fig.colorbar(im, ax=ax, label='uGal')
        fig.tight_layout()
        fig.savefig(os.path.join(folder, 'cumulative_gravity_maps.png'), dpi=120)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--size', choices=sorted(SIZES), default='dev')
    r.add_argument('--conditions', nargs='+', choices=sorted(CONDITIONS), default=['A', 'B', 'C'])
    r.add_argument('--seeds', default='0-9', help='e.g. 0-9 or 0,3,5-7')
    r.add_argument('--jobs', type=int, default=1)
    r.add_argument('--overwrite', action='store_true')
    s = sub.add_parser('report')
    s.add_argument('--size', choices=sorted(SIZES), default='dev')
    args = p.parse_args(argv)
    if args.command == 'run':
        run_ensembles(args.size, args.conditions, _seeds(args.seeds), args.jobs, args.overwrite)
    else:
        report(args.size)


if __name__ == '__main__':
    main()
