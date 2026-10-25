"""Benchmarks: best-of-k wall time and tracemalloc peak per configuration.

    python benchmarks/run.py LABEL [--only SUBSTRING]

Results are stored under LABEL in benchmarks/results.json and compared with 'baseline'.
"""
import json
import os
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from burridge_knopoff import SimulationConfig  # noqa: E402
from burridge_knopoff.simulation import run  # noqa: E402

RESULTS = os.path.join(HERE, 'results.json')
CASES = {
    'default 100x100, 100 years': (SimulationConfig(), 1),
    '50x50, 100 years': (SimulationConfig(n_blocks_x=50, n_blocks_y=50), 2),
    '20x20, 100 years': (SimulationConfig(n_blocks_x=20, n_blocks_y=20), 3),
}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'latest'
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else ''
    stored = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    baseline = stored.get('baseline', {})
    results = dict(stored.get(label, {}))
    run(SimulationConfig(n_blocks_x=3, n_blocks_y=3, simulation_years=5))  # compile the kernels
    for name, (cfg, repeats) in CASES.items():
        if only not in name:
            continue
        times = []
        for __ in range(repeats):
            start = time.perf_counter()
            res = run(cfg)
            times.append(time.perf_counter() - start)
        tracemalloc.start()
        run(cfg)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        results[name] = r = {'time_s': round(min(times), 3), 'peak_mb': round(peak / 1e6, 1), 'events': len(res.events)}
        base = baseline.get(name)
        extra = '  (baseline %.1f s, %.1fx faster)' % (base['time_s'], base['time_s'] / r['time_s']) if base and label != 'baseline' else ''
        print('%-28s %8.2f s %7.1f MB %7d events%s' % (name, r['time_s'], r['peak_mb'], r['events'], extra), flush=True)
    stored[label] = results
    with open(RESULTS, 'w') as fh:
        json.dump(stored, fh, indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
