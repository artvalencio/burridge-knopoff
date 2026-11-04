"""
Nonlinear-dynamics analyses of the 2-D model (plan, section 6), on development grids.

    python 3d_model/experiments/nonlinear_analysis.py all            # every analysis
    python 3d_model/experiments/nonlinear_analysis.py spectrum sync  # some of them

Analyses (results in results/nonlinear/<name>.json and figures):
    spectrum     Lyapunov spectrum of the event map and Kaplan-Yorke dimension vs grid size
                 (deterministic model): is the chaos extensive? (H5)
    lyapunov     largest exponent (two trajectories), finite-size exponent and predictability
                 horizons on the 100 km grid, deterministic and per-block noise (H4)
    recurrence   windowed RQA of the elastic energy: windows before large events vs random ones
    sync         regional event phases: Kuramoto order, phase locking vs distance, event
                 synchronisation; without and with tides (H6)
    information  transfer-entropy network of the regions and its distance decay; tide ->
                 seismicity and gravity -> seismicity flow, with no-tide null runs (H7)
    ordinal      complexity-entropy positions of the deterministic, per-block and default models
"""
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from burridge_knopoff.config import SECONDS_PER_YEAR, SimulationConfig     # noqa: E402
from burridge_knopoff.dynamics import embedding as em                       # noqa: E402
from burridge_knopoff.dynamics import information as inf                    # noqa: E402
from burridge_knopoff.dynamics import lyapunov as ly                        # noqa: E402
from burridge_knopoff.dynamics import observables as obs                    # noqa: E402
from burridge_knopoff.dynamics import ordinal as od                         # noqa: E402
from burridge_knopoff.dynamics import recurrence as rq                      # noqa: E402
from burridge_knopoff.dynamics import synchronization as sync               # noqa: E402
from burridge_knopoff.simulation import run                                 # noqa: E402

OUT = os.path.join(HERE, 'results', 'nonlinear')
DEV = dict(n_blocks_x=50, n_blocks_y=50, n_sensors_x=10, n_sensors_y=10, simulation_years=220.0, spinup_years=20.0,
           sample_interval_years=0.01, snapshot_interval_years=220.0)
DETERMINISTIC = dict(redraw_thresholds=False)


def _save(name, data):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + '.json'), 'w') as fh:
        json.dump(data, fh, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else float(v))
    print('%s: %s' % (name, json.dumps(data, default=lambda v: np.round(v, 4).tolist() if hasattr(v, 'tolist')
                                        else round(float(v), 4))[:1500]), flush=True)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# ---- H5: spectrum vs grid size ---------------------------------------------------------
def spectrum():
    rows = []
    for n in (2, 4, 6, 8, 10, 12):
        cfg = SimulationConfig(n_blocks_x=n, n_blocks_y=n, n_sensors_x=1, n_sensors_y=1, simulation_years=1e5,
                               spinup_years=0.0, seed=3, **DETERMINISTIC)
        start = time.perf_counter()
        sp = ly.event_map_spectrum(cfg, n_events=300, n_transient=300)
        rows.append(dict(n_blocks=n * n, lambda1_per_event=sp['per_event'][0], lambda1_per_year=sp['per_year'][0],
                         n_positive=int(np.sum(sp['per_event'] > 1e-3)), kaplan_yorke=sp['kaplan_yorke'],
                         retries=sp['retries'], seconds=time.perf_counter() - start, spectrum=sp['per_event']))
    _save('spectrum', rows)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    nb = [r['n_blocks'] for r in rows]
    axes[0].plot(nb, [r['kaplan_yorke'] for r in rows], 'o-', label='Kaplan-Yorke dimension')
    axes[0].plot(nb, [r['n_positive'] for r in rows], 's--', label='positive exponents')
    axes[0].set_xlabel('number of blocks')
    axes[0].legend()
    axes[0].set_title('Extensivity of the chaos (deterministic model)')
    for r in rows:
        axes[1].plot(np.arange(1, len(r['spectrum']) + 1) / r['n_blocks'], r['spectrum'], '.-', label='%d' % r['n_blocks'])
    axes[1].set_xlabel('exponent index / number of blocks')
    axes[1].set_ylabel('exponent per event')
    axes[1].set_ylim(-3, None)
    axes[1].legend(title='blocks', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'spectrum.png'), dpi=120)
    plt.close(fig)


# ---- H4: exponents and predictability on the 100 km grid -------------------------------
def lyapunov():
    out = {}
    for mode, kw in (('deterministic', DETERMINISTIC), ('per-block noise', dict(rng_mode='per-block'))):
        cfg = SimulationConfig(**{**DEV, 'n_sensors_x': 1, 'n_sensors_y': 1, 'sample_interval_years': 10.0}, **kw)
        tt = ly.two_trajectory(cfg, delta=1e-9, renorm_years=0.5, total_years=80.0, start_years=30.0)
        fs = ly.finite_size(cfg, delta0=1e-10, n_scales=24, check_years=0.05, max_years=30.0, start_years=30.0,
                            n_samples=4)
        hz = ly.predictability_horizon(cfg, delta=1e-9, horizon_years=40.0, start_years=30.0, n_samples=4,
                                       magnitude_classes=(0.0, 5.0, 6.0, 6.5))
        out[mode] = dict(lambda1_per_year=tt['per_year'], lambda1_per_event=tt['per_event'],
                         e_folding_years=1 / tt['per_year'] if tt['per_year'] > 0 else math.inf,
                         fsle_scales=fs['scales'], fsle=fs['exponent'],
                         horizon_median_years={str(m): float(np.nanmedian(v)) if np.isfinite(v).any() else None
                                               for m, v in hz.items()})
    _save('lyapunov', out)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, r in out.items():
        ax.loglog(r['fsle_scales'], np.where(np.asarray(r['fsle']) > 0, r['fsle'], np.nan), 'o-', label=mode)
    ax.set_xlabel('perturbation size delta (RMS stress gap / strength excess)')
    ax.set_ylabel('finite-size exponent (1/year)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fsle.png'), dpi=120)
    plt.close(fig)


# ---- recurrence: energy before large events vs random windows ---------------------------
def recurrence():
    cfg = SimulationConfig(**DEV, record_energy=True)
    res = run(cfg)
    keep = res.times_years >= cfg.spinup_years
    t, energy = res.times_years[keep], res.elastic_energy[keep]
    x = np.diff(energy)                                   # energy increments per sample (0.01 year)
    tau = max(1, em.first_minimum(em.auto_mutual_information(x, 30)))
    X = em.delay_vectors(x, 3, tau)
    window = 100                                          # one year of samples
    big = [e.start_years for e in res.events if e.magnitude >= 6.5 and e.start_years > cfg.spinup_years + 1.5]
    pre = []
    for tb in big:
        k = int(np.searchsorted(t, tb)) - 1
        if k - window >= 0 and k < len(X):
            pre.append(rq.rqa(rq.recurrence_matrix(X[k - window:k], rate=0.1, theiler=1), theiler=1))
    rng = np.random.default_rng(0)
    rand = [rq.rqa(rq.recurrence_matrix(X[s:s + window], rate=0.1, theiler=1), theiler=1)
            for s in rng.integers(0, len(X) - window, 200)]
    keys = ('DET', 'LAM', 'L_max', 'ENTR', 'TT')
    from scipy import stats
    summary = {k: dict(pre_mean=float(np.mean([r[k] for r in pre])) if pre else None,
                       random_mean=float(np.mean([r[k] for r in rand])),
                       mannwhitney_p=float(stats.mannwhitneyu([r[k] for r in pre], [r[k] for r in rand]).pvalue)
                       if len(pre) >= 5 else None) for k in keys}
    h, c = od.statistical_complexity(x, 5)
    _save('recurrence', dict(n_large_events=len(pre), tau_samples=tau, rqa=summary,
                             energy_increments_entropy=h, energy_increments_complexity=c))


# ---- H6: synchronisation of regions -------------------------------------------------------
def _regional_phases(res, min_moment):
    trains = obs.regional_event_trains(res, min_moment)
    t0, t1 = res.config.spinup_years * SECONDS_PER_YEAR, res.config.simulation_years * SECONDS_PER_YEAR
    times = np.linspace(t0, t1, 20000)
    return times, np.column_stack([sync.event_phase(tr, times) for tr in trains]), trains


def sync_analysis():
    out = {}
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, kw in (('no tides', {}), ('tides', dict(tides=True)), ('tides x100', dict(tides=True, tide_scale=100.0))):
        cfg = SimulationConfig(**{**DEV, 'sample_interval_years': 1.0}, **kw)
        res = run(cfg)
        min_moment = np.quantile(res.regional_moment[res.regional_moment > 0], 0.95)
        times, phases, trains = _regional_phases(res, min_moment)
        r = sync.kuramoto_order(phases)
        centres = obs.region_centres(cfg)
        n = phases.shape[1]
        plv = np.full((n, n), np.nan)
        es = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                plv[i, j] = plv[j, i] = sync.phase_locking_value(phases[:, i], phases[:, j])
                es[i, j] = es[j, i] = sync.event_synchronization(trains[i], trains[j])[0]
        d, plv_by_d, __ = inf.distance_decay(np.nan_to_num(plv), centres)
        __, es_by_d, __ = inf.distance_decay(np.nan_to_num(es), centres)
        out[label] = dict(kuramoto_mean=float(np.nanmean(r)), kuramoto_p95=float(np.nanpercentile(r, 95)),
                          plv_mean=float(np.nanmean(plv)), event_sync_mean=float(np.nanmean(es)),
                          distance_km=d / 1e3, plv_by_distance=plv_by_d, event_sync_by_distance=es_by_d,
                          events_per_region=[len(tr) for tr in trains])
        ax.plot(d / 1e3, plv_by_d, 'o-', label=label)
    ax.set_xlabel('distance between region centres (km)')
    ax.set_ylabel('phase-locking value')
    ax.legend()
    ax.set_title('Synchronisation of regional large events')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sync.png'), dpi=120)
    plt.close(fig)
    _save('sync', out)


# ---- H7: information flow -------------------------------------------------------------------
def information():
    out = {}
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, kw in (('no tides', {}), ('tides', dict(tides=True))):
        cfg = SimulationConfig(**{**DEV, 'sample_interval_years': 1.0, 'n_regions_x': 4, 'n_regions_y': 4}, **kw)
        res = run(cfg)
        t, series = obs.regional_moment_series(res, bin_days=7.0)
        net = inf.te_network(series, n_surrogates=19, n_symbols=2, symbolic_length=2)
        centres = obs.region_centres(cfg)
        d, te_by_d, counts = inf.distance_decay(net['te'], centres)
        # tide -> seismicity: hourly tidal Coulomb vs hourly event counts (tide series of the config,
        # used for the no-tide run as the null)
        from burridge_knopoff import tide_analysis
        times, coulomb, __ = tide_analysis.tidal_series(cfg)
        keep = times >= cfg.spinup_years * SECONDS_PER_YEAR
        counts_h = inf.event_rate(obs.event_train(res), times[keep])
        tide_flow = inf.pair_flow(coulomb[keep], counts_h, n_surrogates=19, n_symbols=2, symbolic_length=2,
                                  symbolic_type='equal-points')
        out[label] = dict(te_mean=float(net['te'][~np.eye(len(centres), dtype=bool)].mean()),
                          fraction_significant=float(net['significant'][~np.eye(len(centres), dtype=bool)].mean()),
                          distance_km=d / 1e3, te_by_distance=te_by_d,
                          tide_to_events=dict(te=tide_flow['te_xy'], p=tide_flow['p_xy'], reverse_te=tide_flow['te_yx'],
                                              directionality=tide_flow['directionality']))
        ax.plot(d / 1e3, te_by_d, 'o-', label=label)
    ax.set_xlabel('distance between region centres (km)')
    ax.set_ylabel('mean transfer entropy (bits)')
    ax.legend()
    ax.set_title('Information flow between regions (weekly moment release)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'information.png'), dpi=120)
    plt.close(fig)
    _save('information', out)


# ---- chaos or noise -----------------------------------------------------------------------
def ordinal():
    out = {}
    for label, kw in (('deterministic', DETERMINISTIC), ('per-block noise', dict(rng_mode='per-block')),
                      ('default noise', {})):
        cfg = SimulationConfig(**{**DEV, 'sample_interval_years': 1.0}, **kw)
        res = run(cfg)
        dt = obs.inter_event_times(obs.event_train(res))
        mags = np.array([e.magnitude for e in res.events if e.start_years >= cfg.spinup_years])
        out[label] = dict(interevent=od.statistical_complexity(dt, 5), magnitude=od.statistical_complexity(mags, 5),
                          n_events=len(mags))
    _save('ordinal', out)


ANALYSES = dict(spectrum=spectrum, lyapunov=lyapunov, recurrence=recurrence, sync=sync_analysis,
                information=information, ordinal=ordinal)


def main(argv=None):
    names = (argv if argv is not None else sys.argv[1:]) or ['all']
    if 'all' in names:
        names = list(ANALYSES)
    for name in names:
        start = time.perf_counter()
        ANALYSES[name]()
        print('[%s done in %.0f s]' % (name, time.perf_counter() - start), flush=True)


if __name__ == '__main__':
    main()
