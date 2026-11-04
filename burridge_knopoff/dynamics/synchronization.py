"""
Synchronisation of event trains and forcing (Pikovsky et al., 2001).

- event_phase: the phase of a train of events at given times, increasing by
  2 pi between successive events (linear interpolation)
- phase_locking_value: |mean exp(i (phi_1 - phi_2))|
- kuramoto_order: |mean_k exp(i phi_k)| over oscillators, at every time
- event_synchronization: Quian Quiroga, Kreuz & Grassberger (2002): Q (0..1)
  and the delay asymmetry q (> 0: the first train tends to lead)
- firing_phases / rotation_number: phases of events relative to a periodic
  forcing, the return map phi_{n+1} vs phi_n, and the mean phase advance per
  event (in cycles); locked trains have a constant rotation number
"""
import math

import numpy as np


def event_phase(event_times, times):
    """Phase (rad, unwrapped) of the event train at `times`: 2 pi k at the k-th event, linear
    in between; nan before the first and after the last event."""
    ev = np.sort(np.asarray(event_times, dtype=float))
    t = np.asarray(times, dtype=float)
    out = np.full(t.shape, np.nan)
    if len(ev) < 2:
        return out
    k = np.searchsorted(ev, t, side='right') - 1
    ok = (k >= 0) & (k < len(ev) - 1)
    kk = k[ok]
    out[ok] = 2 * math.pi * (kk + (t[ok] - ev[kk]) / (ev[kk + 1] - ev[kk]))
    return out


def phase_locking_value(phi1, phi2):
    d = np.asarray(phi1) - np.asarray(phi2)
    d = d[np.isfinite(d)]
    return float(np.abs(np.mean(np.exp(1j * d)))) if len(d) else math.nan


def kuramoto_order(phases):
    """Order parameter r(t) of phases (n_times, n_oscillators); nan columns are ignored."""
    z = np.exp(1j * np.asarray(phases))
    valid = np.isfinite(np.asarray(phases))
    z = np.where(valid, z, 0)
    count = valid.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(z.sum(axis=1)) / count


def event_synchronization(t1, t2, tau=None):
    """(Q, q) of two event trains (Quian Quiroga et al., 2002). tau None: the local, adaptive
    time window (half the smallest neighbouring inter-event interval)."""
    a, b = np.sort(np.asarray(t1, dtype=float)), np.sort(np.asarray(t2, dtype=float))
    if len(a) < 3 or len(b) < 3:
        return math.nan, math.nan

    def c(x, y):
        # number of times an event of x follows one of y within the window (ties count 1/2)
        total = 0.0
        for i in range(1, len(x) - 1):
            reach = tau if tau is not None else 0.5 * min(x[i + 1] - x[i], x[i] - x[i - 1])
            j0 = np.searchsorted(y, x[i] - reach, side='left')
            j1 = np.searchsorted(y, x[i], side='right')
            for j in range(max(j0, 1), min(j1, len(y) - 1)):
                d = x[i] - y[j]
                window = tau if tau is not None else 0.5 * min(x[i + 1] - x[i], x[i] - x[i - 1],
                                                               y[j + 1] - y[j], y[j] - y[j - 1])
                if d == 0.0:
                    total += 0.5
                elif 0.0 < d < window:
                    total += 1.0
        return total

    c12, c21 = c(a, b), c(b, a)
    norm = math.sqrt((len(a) - 2) * (len(b) - 2))
    return (c12 + c21) / norm, (c21 - c12) / norm


def firing_phases(event_times, period, phase0=0.0):
    """Phases (rad, in [0, 2 pi)) of events relative to a periodic forcing."""
    return np.mod(2 * math.pi * np.asarray(event_times, dtype=float) / period + phase0, 2 * math.pi)


def rotation_number(event_times, period):
    """Mean number of forcing cycles per event interval."""
    t = np.sort(np.asarray(event_times, dtype=float))
    if len(t) < 2:
        return math.nan
    return float((t[-1] - t[0]) / period / (len(t) - 1))
