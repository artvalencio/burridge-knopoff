"""
Earth tides from the Moon and the Sun, and their effect on the fault.

Positions of the bodies
-----------------------
Low-precision analytic series (Montenbruck & Gill, 2000, sec. 3.3.2; terms
as in Meeus, 1998, ch. 25 and 47): the Moon to ~0.01 degree and ~100 km, the
Sun to ~0.03 degree. Ecliptic coordinates of date are rotated to the
Earth-fixed frame with the mean obliquity and the Greenwich mean sidereal
time (Meeus ch. 12); nutation and polar motion are neglected. Time is
seconds since config.epoch_utc; dynamical time is UTC + 69.184 s.

Tidal potential and its effects at a point P (unit vector p, radius r)
----------------------------------------------------------------------
For a body of mass parameter GM at distance d in the direction R:

    W_n = GM / d (r / d)^n P_n(c),   c = p . R       (n = 2; and 3 for the Moon)

With q = R - c p (the tangential part of R), the angular Hessian of W_n on
the sphere is K r^n [P_n''(c) q q^T - c P_n'(c) I] (a closed form, regular
at every c), its tangential gradient K r^(n-1) P_n'(c) q, and its radial
derivative n W_n / r. The deformation of the elastic Earth (Love and Shida
numbers h_n, l_n; Agnew, 2007) gives the horizontal strain

    eps = ( h_n W I + l_n Hessian ) / (g a),

whose trace is (2 h_n - n (n + 1) l_n) W / (g a). At the free surface the
vertical stresses vanish (plane stress), so the horizontal stress is
2 mu [eps + nu / (1 - nu) tr(eps) I]; with a Skempton coefficient B > 0 the
undrained pore pressure -B K eps_kk is added to the normal stress.

On the fault (strike, dip, rake) with normal n (footwall to hanging wall) and
hanging-wall slip vector s:

    shear  tau = s . sigma . n     (positive: promotes slip in the tectonic sense)
    normal sigma_n = n . sigma . n + p   (tension positive: unclamping)

and the body force on a block along its own motion (-s). For a block of area
A and mass m, the tidal force that drives it is A tau + m a_s; the normal
stress shifts its static strength by -mu_s A sigma_n and its dynamic friction
by -mu_d A sigma_n (mu_s = config.friction_coefficient, mu_d = mu_s F_d / F_s).

Tidal gravity at the sensors (change of the downward gravity, m/s^2):
-delta_n dW_n/dr with the gravimetric factor delta_n = 1 + 2 h_n / n - (n + 1) k_n / n.

Spatial representation
----------------------
The tide varies smoothly over the model area. Each quantity is evaluated at a
5 x 5 grid of control points and fitted as c0 + cx x + cy y in the local
model coordinates (origin at the centre of the block grid); with
tide_spatial='uniform' only the centre value is used. HourlyTide holds these
coefficients every tide_step_hours and interpolates them in between.
"""
import datetime
import math

import numpy as np

from .config import SECONDS_PER_YEAR

DEG = math.pi / 180.0
ARCSEC = DEG / 3600.0
GM_MOON = 4.9028e12          # m^3 s^-2
GM_SUN = 1.32712440018e20    # m^3 s^-2
EARTH_RADIUS_M = 6.371e6
EARTH_GRAVITY = 9.81         # m s^-2
AU_M = 1.495978707e11
TT_MINUS_UTC_S = 69.184
J2000 = 2451545.0

# degree-3 Love numbers (IERS Conventions 2010, nominal)
LOVE_H3, LOVE_L3, LOVE_K3 = 0.292, 0.015, 0.093


# ---- time -----------------------------------------------------------------------
def epoch_julian_date(epoch_utc):
    """Julian date (UTC) of an ISO 8601 time such as '2000-01-01T12:00:00'."""
    t = datetime.datetime.fromisoformat(epoch_utc)
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return 2440587.5 + (t - datetime.datetime(1970, 1, 1)).total_seconds() / 86400.0


def gmst_rad(jd_ut):
    """Greenwich mean sidereal time (rad) at Julian date jd_ut (Meeus 12.4)."""
    d = jd_ut - J2000
    T = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T * T - T ** 3 / 38710000.0
    return np.mod(deg, 360.0) * DEG


# ---- ephemerides -------------------------------------------------------------------
def moon_ecliptic(T):
    """Ecliptic longitude, latitude (rad, mean equinox of date) and distance (m) of the
    Moon, T in Julian centuries of dynamical time since J2000."""
    T = np.asarray(T, dtype=float)
    L0 = (218.31617 + 481267.88088 * T) * DEG
    l = (134.96292 + 477198.86753 * T) * DEG
    lp = (357.52543 + 35999.04944 * T) * DEG
    F = (93.27283 + 483202.01873 * T) * DEG
    D = (297.85027 + 445267.11135 * T) * DEG
    lon = L0 + ARCSEC * (22640 * np.sin(l) + 769 * np.sin(2 * l) - 4586 * np.sin(l - 2 * D) + 2370 * np.sin(2 * D)
                         - 668 * np.sin(lp) - 412 * np.sin(2 * F) - 212 * np.sin(2 * l - 2 * D)
                         - 206 * np.sin(l + lp - 2 * D) + 192 * np.sin(l + 2 * D) - 165 * np.sin(lp - 2 * D)
                         + 148 * np.sin(l - lp) - 125 * np.sin(D) - 110 * np.sin(l + lp) - 55 * np.sin(2 * F - 2 * D))
    lat = ARCSEC * (18520 * np.sin(F + lon - L0 + ARCSEC * (412 * np.sin(2 * F) + 541 * np.sin(lp)))
                    - 526 * np.sin(F - 2 * D) + 44 * np.sin(l + F - 2 * D) - 31 * np.sin(-l + F - 2 * D)
                    - 25 * np.sin(-2 * l + F) - 23 * np.sin(lp + F - 2 * D) + 21 * np.sin(-l + F)
                    + 11 * np.sin(-lp + F - 2 * D))
    dist_km = (385000 - 20905 * np.cos(l) - 3699 * np.cos(2 * D - l) - 2956 * np.cos(2 * D) - 570 * np.cos(2 * l)
               + 246 * np.cos(2 * l - 2 * D) - 205 * np.cos(lp - 2 * D) - 171 * np.cos(l + 2 * D)
               - 152 * np.cos(l + lp - 2 * D))
    return np.mod(lon, 2 * math.pi), lat, dist_km * 1e3


def sun_ecliptic(T):
    """Ecliptic longitude, latitude (0) and distance (m) of the Sun, mean equinox of date."""
    T = np.asarray(T, dtype=float)
    M = (357.5256 + 35999.049 * T) * DEG
    lon = (282.9400 + 1.3972 * T) * DEG + M + ARCSEC * (6892 * np.sin(M) + 72 * np.sin(2 * M))
    dist = (149.619 - 2.499 * np.cos(M) - 0.021 * np.cos(2 * M)) * 1e9
    return np.mod(lon, 2 * math.pi), np.zeros_like(lon), dist


def earth_fixed(lon, lat, dist, jd_ut, T):
    """Ecliptic coordinates of date -> Earth-fixed Cartesian position (m), shape (..., 3)."""
    eps = (23.439291 - 0.0130042 * T) * DEG
    x = dist * np.cos(lat) * np.cos(lon)
    y = dist * np.cos(lat) * np.sin(lon)
    z = dist * np.sin(lat)
    ye = y * np.cos(eps) - z * np.sin(eps)       # ecliptic -> equatorial
    ze = y * np.sin(eps) + z * np.cos(eps)
    theta = gmst_rad(jd_ut)                       # equatorial -> Earth-fixed
    return np.stack([np.cos(theta) * x + np.sin(theta) * ye, -np.sin(theta) * x + np.cos(theta) * ye, ze], axis=-1)


def body_positions(t_s, epoch_jd, bodies=('moon', 'sun')):
    """{name: (GM, Earth-fixed position (len(t), 3) in m)} at times t_s (seconds since the epoch)."""
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
    jd_ut = epoch_jd + t_s / 86400.0
    T = (jd_ut + TT_MINUS_UTC_S / 86400.0 - J2000) / 36525.0
    out = {}
    for name in bodies:
        if name == 'moon':
            out[name] = (GM_MOON, earth_fixed(*moon_ecliptic(T), jd_ut, T))
        elif name == 'sun':
            out[name] = (GM_SUN, earth_fixed(*sun_ecliptic(T), jd_ut, T))
        else:
            raise ValueError('unknown tide body %r (moon, sun)' % name)
    return out


# ---- geometry ----------------------------------------------------------------------
def _legendre(n, c):
    # P_n, P_n', P_n'' at c
    if n == 2:
        return 0.5 * (3 * c * c - 1), 3 * c, 3 * np.ones_like(c)
    if n == 3:
        return 0.5 * (5 * c ** 3 - 3 * c), 0.5 * (15 * c * c - 3), 15 * c
    raise ValueError(n)


def fault_vectors(strike_deg, dip_deg, rake_deg):
    """Unit vectors in local (east, north, up): fault normal n (footwall to hanging wall)
    and hanging-wall slip vector s."""
    phi, delta, lam = strike_deg * DEG, dip_deg * DEG, rake_deg * DEG
    strike = np.array([math.sin(phi), math.cos(phi), 0.0])
    down_dip = np.array([math.cos(delta) * math.cos(phi), -math.cos(delta) * math.sin(phi), -math.sin(delta)])
    normal = np.array([math.sin(delta) * math.cos(phi), -math.sin(delta) * math.sin(phi), math.cos(delta)])
    slip = math.cos(lam) * strike - math.sin(lam) * down_dip
    return normal, slip


def local_to_geographic(x_m, y_m, config):
    """Latitude, longitude (deg) of model coordinates relative to the centre of the block
    grid: x points along the down-dip azimuth (strike + 90 deg), y along the strike."""
    phi = config.fault_strike_deg * DEG
    east = np.asarray(x_m) * math.cos(phi) + np.asarray(y_m) * math.sin(phi)
    north = -np.asarray(x_m) * math.sin(phi) + np.asarray(y_m) * math.cos(phi)
    lat = config.centre_latitude_deg + north / EARTH_RADIUS_M / DEG
    lon = config.centre_longitude_deg + east / (EARTH_RADIUS_M * math.cos(config.centre_latitude_deg * DEG)) / DEG
    return lat, lon


def grid_centre(config):
    L = config.block_spacing_m
    return (config.n_blocks_x - 1) * L / 2.0, (config.n_blocks_y - 1) * L / 2.0


class TidePoints:
    """Tidal fields at fixed points (latitude, longitude in degrees, depth in m)."""

    def __init__(self, lat_deg, lon_deg, config, depth_m=0.0):
        lat, lon = np.asarray(lat_deg, dtype=float) * DEG, np.asarray(lon_deg, dtype=float) * DEG
        self.config = config
        self.radius = EARTH_RADIUS_M - np.broadcast_to(np.asarray(depth_m, dtype=float), lat.shape)
        self.p = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
        self.east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
        self.north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
        self.normal, self.slip = fault_vectors(config.fault_strike_deg, config.fault_dip_deg, config.fault_rake_deg)
        self.bodies = tuple(b.strip() for b in config.tide_bodies.split(',') if b.strip())
        self.epoch_jd = epoch_julian_date(config.epoch_utc)

    def fields(self, t_s):
        """Tidal fields at times t_s (s since the epoch), each of shape (len(t), n_points):
        potential W (J/kg), horizontal stress tensor (sigma_ee, sigma_nn, sigma_en; Pa),
        pore pressure (Pa), acceleration (east, north, up; m/s^2), gravity change (m/s^2,
        downward positive), shear and normal stress on the fault (Pa), and the acceleration
        along the block motion (m/s^2). Not scaled by tide_scale."""
        cfg = self.config
        mu, nu = cfg.shear_modulus_pa, cfg.poisson_ratio
        love = {2: (cfg.love_h2, cfg.love_l2, cfg.love_k2), 3: (LOVE_H3, LOVE_L3, LOVE_K3)}
        t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
        shape = (len(t_s),) + self.p.shape[:-1]
        W = np.zeros(shape)
        eps = np.zeros(shape + (2, 2))
        acc = np.zeros(shape + (3,))
        grav = np.zeros(shape)
        for name, (gm, pos) in body_positions(t_s, self.epoch_jd, self.bodies).items():
            d = np.linalg.norm(pos, axis=-1)
            R = pos / d[:, None]
            Rb = R.reshape((len(t_s),) + (1,) * (self.p.ndim - 1) + (3,))
            db = d.reshape((len(t_s),) + (1,) * (self.p.ndim - 1))
            c = np.sum(self.p * Rb, axis=-1)
            q = Rb - c[..., None] * self.p
            qe, qn = np.sum(q * self.east, axis=-1), np.sum(q * self.north, axis=-1)
            for n in ((2, 3) if name == 'moon' else (2,)):
                h, l, k = love[n]
                P, dP, ddP = _legendre(n, c)
                K = gm / db * (self.radius / db) ** n          # GM r^n / d^(n+1)
                Wn = K * P
                hess = np.empty(shape + (2, 2))
                hess[..., 0, 0] = K * (ddP * qe * qe - c * dP)
                hess[..., 1, 1] = K * (ddP * qn * qn - c * dP)
                hess[..., 0, 1] = hess[..., 1, 0] = K * ddP * qe * qn
                ge = EARTH_GRAVITY * EARTH_RADIUS_M
                eps += (h * Wn[..., None, None] * np.eye(2) + l * hess) / ge
                tangential = K / self.radius * dP
                acc[..., 0] += tangential * qe
                acc[..., 1] += tangential * qn
                acc[..., 2] += n * Wn / self.radius
                grav -= (1 + 2.0 * h / n - (n + 1) * k / n) * n * Wn / self.radius
                W += Wn
        tr = eps[..., 0, 0] + eps[..., 1, 1]
        sigma = 2 * mu * (eps + (nu / (1 - nu)) * tr[..., None, None] * np.eye(2))
        bulk = 2 * mu * (1 + nu) / (3 * (1 - 2 * nu))
        pore = -cfg.skempton_b * bulk * tr * (1 - 2 * nu) / (1 - nu)
        n2, s2 = self.normal[:2], self.slip[:2]   # vertical stress components vanish
        shear = np.einsum('i,...ij,j->...', s2, sigma, n2)
        normal = np.einsum('i,...ij,j->...', n2, sigma, n2) + pore
        motion = -self.slip                                    # the block moves against the hanging wall
        acc_motion = acc[..., 0] * motion[0] + acc[..., 1] * motion[1] + acc[..., 2] * motion[2]
        return dict(potential=W, sigma_ee=sigma[..., 0, 0], sigma_nn=sigma[..., 1, 1], sigma_en=sigma[..., 0, 1],
                    strain_ee=eps[..., 0, 0], strain_nn=eps[..., 1, 1], strain_en=eps[..., 0, 1], pore=pore,
                    acceleration=acc, gravity=grav, shear=shear, normal=normal, accel_motion=acc_motion)


# ---- coefficients for the model -------------------------------------------------
class TideModel:
    """Tidal forcing of a SimulationConfig, as linear fits c0 + cx x + cy y over the model area.

    coefficients(t_s) returns a dict of (len(t), 3) arrays:
        drive   A tau + m a_motion   (N): the force pushing a block along its slip
        normal  A sigma_n            (N): tension positive
        gravity tidal gravity change (m/s^2, downward positive) at the sensors
    all multiplied by tide_scale, with tide_stress / tide_body_force switching terms off.
    Coordinates are relative to grid_centre(config).
    """

    n_control = 5

    def __init__(self, config):
        self.config = config
        x0, y0 = grid_centre(config)
        if config.tide_spatial == 'uniform':
            xs = ys = np.zeros(1)
        else:
            xs = np.linspace(-x0, x0, self.n_control) if x0 > 0 else np.zeros(1)
            ys = np.linspace(-y0, y0, self.n_control) if y0 > 0 else np.zeros(1)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        self.x, self.y = xx.ravel(), yy.ravel()
        lat, lon = local_to_geographic(self.x, self.y, config)
        self.points = TidePoints(lat, lon, config)
        basis = np.column_stack([np.ones_like(self.x), self.x, self.y])
        self.fit = np.linalg.pinv(basis)            # (3, n_points)

    def fit_coefficients(self, values):
        return values @ self.fit.T

    def coefficients(self, t_s):
        cfg = self.config
        f = self.points.fields(t_s)
        area, mass = cfg.block_area_m2, cfg.block_mass_kg
        stress = 1.0 if cfg.tide_stress else 0.0
        body = 1.0 if cfg.tide_body_force else 0.0
        drive = stress * area * f['shear'] + body * mass * f['accel_motion']
        normal = stress * area * f['normal']
        s = cfg.tide_scale
        return dict(drive=s * self.fit_coefficients(drive), normal=s * self.fit_coefficients(normal),
                    gravity=s * self.fit_coefficients(f['gravity']))


def evaluate(coefficients, x, y):
    """Values c0 + cx x + cy y at positions relative to the grid centre, for one time
    (coefficients of shape (3,)) or many ((n, 3) -> (n, len(x)))."""
    c = np.asarray(coefficients)
    return c[..., 0:1] + c[..., 1:2] * np.asarray(x) + c[..., 2:3] * np.asarray(y)


class HourlyTide:
    """The coefficients of TideModel every tide_step_hours from t = 0 to beyond end_s,
    with Catmull-Rom (cubic) interpolation in between."""

    def __init__(self, config, end_s, chunk_steps=8766):
        self.config = config
        self.model = TideModel(config)
        self.step_s = config.tide_step_hours * 3600.0
        n = int(math.ceil(end_s / self.step_s)) + 3
        self.times = np.arange(n) * self.step_s
        parts = [self.model.coefficients(self.times[k:k + chunk_steps]) for k in range(0, n, chunk_steps)]
        self.drive = np.concatenate([p['drive'] for p in parts])
        self.normal = np.concatenate([p['normal'] for p in parts])
        self.gravity = np.concatenate([p['gravity'] for p in parts])
        mus = config.friction_coefficient
        self.coulomb = self.drive + mus * self.normal          # force against the static strength

    def _interp(self, series, t_s):
        t = np.atleast_1d(np.asarray(t_s, dtype=float)) / self.step_s
        k = np.clip(np.floor(t).astype(np.int64), 0, len(self.times) - 2)
        u = (t - k)[:, None]
        p0 = series[np.maximum(k - 1, 0)]
        p1, p2 = series[k], series[k + 1]
        p3 = series[np.minimum(k + 2, len(self.times) - 1)]
        return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                      + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3)

    def at(self, t_s):
        """Interpolated coefficients (drive, normal, coulomb, gravity), each (len(t), 3)."""
        return (self._interp(self.drive, t_s), self._interp(self.normal, t_s), self._interp(self.coulomb, t_s),
                self._interp(self.gravity, t_s))

    def max_spatial_variation(self, x_extent, y_extent):
        """Largest |cx x + cy y| of the Coulomb force over the run and the area (N)."""
        return float(np.max(np.abs(self.coulomb[:, 1]) * x_extent + np.abs(self.coulomb[:, 2]) * y_extent))

    def semidiurnal_phase(self):
        """Phase (rad) of the Coulomb force at the grid centre in its semidiurnal band
        (periods 10-14 h), from the analytic signal: 0 at the maxima. Returns (times, phase)."""
        x = self.coulomb[:, 0] - self.coulomb[:, 0].mean()
        freq = np.fft.fftfreq(len(x), d=self.step_s)
        band = (freq >= 1 / (14 * 3600.0)) & (freq <= 1 / (10 * 3600.0))   # positive frequencies only
        analytic = np.fft.ifft(np.where(band, 2.0 * np.fft.fft(x), 0.0))  # band-passed analytic signal
        return self.times, np.angle(analytic)


def seconds_to_years(t_s):
    return np.asarray(t_s) / SECONDS_PER_YEAR
