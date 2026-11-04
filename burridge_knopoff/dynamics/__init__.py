"""
Nonlinear-dynamics analyses of the model (plan, section 6).

    observables      series from results: regional moment rates, event trains, gravity modes
    embedding        delay vectors, auto-mutual information, false nearest neighbours, Cao,
                     correlation sum and dimension, return maps
    lyapunov         two-trajectory and finite-size exponents, predictability horizon, the
                     event-map spectrum on small grids, and data estimators (Rosenstein, Kantz)
    recurrence       recurrence plots, RQA, windowed RQA, cross/joint recurrence, rupture similarity
    synchronization  event phases, phase locking, Kuramoto order, event synchronisation,
                     firing-phase maps and rotation numbers
    ordinal          permutation entropy and the complexity-entropy plane
    surrogates       IAAFT, time-shift and interval-shuffle surrogates
    information      transfer entropy / CaMI analyses through the cami toolbox

The estimators need scipy (neighbour searches); information needs cami >= 1.5.
"""
