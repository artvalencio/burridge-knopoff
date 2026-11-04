"""Information-flow pipeline (CaMI toolbox), validated on coupled maps with known links."""
import numpy as np
import pytest

cami = pytest.importorskip('cami')
coupledlogistic = pytest.importorskip('coupledlogistic')
if not hasattr(cami, 'surrogate_test'):
    pytest.skip('cami >= 1.5 needed', allow_module_level=True)

from burridge_knopoff.config import SimulationConfig                      # noqa: E402
from burridge_knopoff.dynamics import information as inf                 # noqa: E402
from burridge_knopoff.dynamics import observables as obs                 # noqa: E402
from burridge_knopoff.simulation import run                              # noqa: E402

BINARY = dict(n_symbols=2, symbolic_length=1, symbolic_type='equal-divs')


@pytest.fixture(scope='module')
def chain():
    # 0 -> 1 -> 2, coupled logistic maps
    return coupledlogistic.coupledlogistic(4000, 4, [[0, 1, 0], [0, 0, 1], [0, 0, 0]], 0.15, 'diffusive', seed=1)


def test_network_recovers_the_known_links(chain):
    # transfer entropy measures information flow, not only direct coupling: the indirect 0 -> 2
    # and weak reverse flows can be significant too, but the true links are the strongest
    net = inf.te_network(chain, n_surrogates=49, log=False, **{**BINARY, 'symbolic_length': 2})
    te = net['te']
    assert net['significant'][0, 1] and net['significant'][1, 2]
    assert te[0, 1] > 5 * te[1, 0] and te[1, 2] > 5 * te[2, 1]
    off = te[~np.eye(3, dtype=bool)]
    assert sorted(off)[-2:] == sorted([te[0, 1], te[1, 2]])


def test_pair_flow_direction(chain):
    flow = inf.pair_flow(chain[:, 0], chain[:, 1], n_surrogates=19, **BINARY)
    assert flow['directionality'] > 0 and flow['p_xy'] <= 0.05 < flow['p_yx']


def test_pooled_short_windows_find_the_flow(chain):
    windows = chain[:3900].reshape(39, 100, 3)              # 39 short trials of 100 steps
    res = inf.pooled_windows([w[:, 0] for w in windows], [w[:, 1] for w in windows], **BINARY)
    assert res['te_xy'] > 5 * res['te_yx']


def test_distance_decay_and_helpers():
    centres = np.array([[0, 0], [1, 0], [3, 0]])
    matrix = np.array([[0, 5, 1], [5, 0, 5], [1, 5, 0]])
    d, mean, count = inf.distance_decay(matrix, centres)
    np.testing.assert_allclose(d, [1, 2, 3])
    np.testing.assert_allclose(mean, [5, 5, 1])
    np.testing.assert_allclose(inf.event_rate([0.5, 0.7, 2.5], [0, 1, 2]), [2, 0, 1])
    assert np.allclose(inf.prepare([0, np.e - 1]), [0, 1]) and np.allclose(inf.prepare([-1, 2]), [-1, 2])


def test_constant_series_give_no_flow():
    x = np.column_stack([np.zeros(500), np.random.default_rng(0).random(500)])
    net = inf.te_network(x, n_surrogates=9, log=False, **BINARY)
    assert np.all(net['te'] == 0) and not net['significant'].any()


def test_observables_from_a_run():
    cfg = SimulationConfig(n_blocks_x=10, n_blocks_y=10, n_sensors_x=3, n_sensors_y=3, simulation_years=60.0,
                           sample_interval_years=0.1, spinup_years=10.0, n_regions_x=2, n_regions_y=2)
    res = run(cfg)
    t, series = obs.regional_moment_series(res, bin_days=30.0)
    after = [e for e in res.events if e.start_years >= 10.0]
    assert series.shape[1] == 4 and series.sum() == pytest.approx(sum(e.moment_n_m for e in after), rel=1e-5)
    trains = obs.regional_event_trains(res, min_moment_n_m=1e17)
    assert len(trains) == 4 and all(np.all(np.diff(tr) >= 0) for tr in trains)
    times, amps, patterns, var = obs.gravity_modes(res, n_modes=2)
    assert amps.shape == (len(times), 2) and patterns.shape == (2, 9) and 0 < var[0] <= 1
    centres = obs.region_centres(cfg)
    assert centres.shape == (4, 2) and centres[0, 0] < centres[1, 0] and centres[0, 1] < centres[2, 1]


def test_sparse_event_series_keep_their_information():
    # mostly empty bins: equal-population partitions would collapse on the zeros
    rng = np.random.default_rng(4)
    x = (rng.random(5000) < 0.05) * rng.lognormal(0, 1, 5000)
    y = np.roll(x, 1) * (rng.random(5000) < 0.9)
    assert list(inf.divisions(x, 2)) == [0.5 * x[x > 0].min()]
    flow = inf.pair_flow(x, y, n_surrogates=19, n_symbols=2, symbolic_length=1)
    assert flow['te_xy'] > 0.05 and flow['p_xy'] <= 0.05 and flow['te_yx'] < 0.2 * flow['te_xy']
