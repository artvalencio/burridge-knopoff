"""
Surface displacement and gravity change of slip on the fault in an elastic half-space.

Each block is a rectangular patch (side block_spacing_m) of a planar fault dipping at
fault_dip_deg, whose top edge is at fault_top_depth_m. Block coordinates on the fault
are x (down dip, from half a spacing below the top edge) and y (along strike); the
surface frame X, Y is the horizontal projection, X towards the dip direction and Y
along the strike, with the top edge at X = -spacing / 2 (so X = x for a flat fault).
A block displacement u is slip of that amount in the rake direction fault_rake_deg
(the hanging wall moves along the rake; 90 is a thrust).

- Displacement: Okada (1985), closed form at the free surface.
- Gravity: Okubo (1992), the change of gravity at a point fixed to the deforming
  surface: dilatation, the raised or lowered surface layer and the free-air change of
  the gravimeter's height, beta u_z with beta = 3.086e-6 s^-2.

Both are linear in slip, so the field of any slip distribution is a sum over patches.
Interseismic deformation follows from back slip (Savage, 1983): steady slip of the
whole interface at the plate rate moves the plates rigidly, so the deformation is that
of the slip deficit u - x_p on the locked patches (gravity_change(u - x_p)).

The corner functions follow the okubo92.m transcription of Okubo's and Okada's
equations by F. Beauducel (2013, BSD licence), restricted to shear slip.
"""
import math

import numpy as np
from numba import njit, prange

from . import grid
from .config import G

FREE_AIR_GRADIENT = 3.086e-6  # s^-2, normal free-air gradient (0.3086 mGal/m)


@njit(cache=True, error_model='numpy', inline='always')
def _corner(xi, eta, q, sd, cd, m):
    # Okada (1985) shear displacement terms and Okubo (1992) gravity terms at one Chinnery
    # corner: (fx_s, fy_s, fz_s, fx_d, fy_d, fz_d, g_s, g_d); u = -U/(2 pi) f and
    # dg = rho G (U1 g_s + U2 g_d). m = 1 - 2 nu = mu / (lambda + mu).
    r = math.sqrt(xi * xi + eta * eta + q * q)
    yb = eta * cd + q * sd
    db = eta * sd - q * cd
    re = r + eta
    rx = r + xi
    ln_re = math.log(re)
    theta = math.atan(xi * eta / (q * r)) if q != 0.0 else 0.0
    if abs(cd) < 1e-6:
        rd = r + db
        i1 = -0.5 * m * xi * q / (rd * rd)
        i3 = 0.5 * m * (eta / rd + yb * q / (rd * rd) - ln_re)
        i4 = -m * q / rd
        i5 = -m * xi * sd / rd
    else:
        i4 = m / cd * (math.log(r + db) - sd * ln_re)
        if abs(xi) < 1e-9 * r:
            i5 = 0.0
        else:
            i5 = m * 2.0 / cd * math.atan((-q * cd + (1.0 + sd) * re) / (xi * cd))
        i3 = m * (yb / (cd * (r + db)) - ln_re) + sd / cd * i4
        i1 = m * (-xi / (cd * (r + db))) - sd / cd * i5
    i2 = -m * ln_re - i3
    fx_s = xi * q / (r * re) + theta + i1 * sd
    fy_s = yb * q / (r * re) + q * cd / re + i2 * sd
    fz_s = db * q / (r * re) + q * sd / re + i4 * sd
    fx_d = q / r - i3 * sd * cd
    fy_d = yb * q / (r * rx) + cd * theta - i1 * sd * cd
    fz_d = db * q / (r * rx) + sd * theta - i5 * sd * cd
    i2g = math.atan((r + xi + eta) / q) if q != 0.0 else 0.5 * math.pi
    g_s = -q * sd / r + q * q * cd / (r * re)
    g_d = 2.0 * i2g * sd - q * db / (r * rx)
    return fx_s, fy_s, fz_s, fx_d, fy_d, fz_d, g_s, g_d


@njit(cache=True, error_model='numpy')
def _patch(x, y, d, length, width, sd, cd, m):
    # Chinnery sums f(x, p) - f(x, p - W) - f(x - L, p) + f(x - L, p - W) in Okada's frame
    # (x along strike from the patch start, y horizontal towards the up-dip side from the
    # bottom edge, d the depth of the bottom edge). Returns the 8 summed terms.
    p = y * cd + d * sd
    q = y * sd - d * cd
    out = np.zeros(8)
    for corner in range(4):
        xi = x if corner < 2 else x - length
        eta = p if corner % 2 == 0 else p - width
        sign = 1.0 if corner == 0 or corner == 3 else -1.0
        t = _corner(xi, eta, q, sd, cd, m)
        for k in range(8):
            out[k] += sign * t[k]
    return out


@njit(cache=True, error_model='numpy')
def _response(X, Y, xs, ys, size, top_depth, x_top, sd, cd, cr, sr, m, rho_g, beta):
    # surface displacement (dX, dY, dZ) and gravity change at (X, Y) for unit slip of the square
    # patch of side `size` centred at fault coordinates (xs, ys); the fault's top edge is at
    # fault coordinate x_top, below X = x_top
    s_bottom = xs - x_top + 0.5 * size  # distance along dip from the top edge to the patch bottom
    d = top_depth + s_bottom * sd
    x_bot = x_top + s_bottom * cd
    ok_x = Y - (ys - 0.5 * size)
    ok_y = x_bot - X
    f = _patch(ok_x, ok_y, d, size, size, sd, cd, m)
    u1, u2 = cr, sr  # unit slip along the rake: strike-slip and dip-slip parts
    ux = -(u1 * f[0] + u2 * f[3]) / (2.0 * math.pi)
    uy = -(u1 * f[1] + u2 * f[4]) / (2.0 * math.pi)
    uz = -(u1 * f[2] + u2 * f[5]) / (2.0 * math.pi)
    dg = rho_g * (u1 * f[6] + u2 * f[7]) - beta * uz
    # Okada's x is along strike (+Y), his y points up dip (-X)
    return -uy, ux, uz, dg


def _geometry(config):
    dip = math.radians(config.fault_dip_deg)
    rake = math.radians(config.fault_rake_deg)
    return dict(size=config.block_spacing_m, top_depth=config.fault_top_depth_m,
                x_top=-0.5 * config.block_spacing_m, sd=math.sin(dip), cd=math.cos(dip),
                cr=math.cos(rake), sr=math.sin(rake), m=1.0 - 2.0 * config.poisson_ratio,
                rho_g=config.rock_density_kg_m3 * G, beta=FREE_AIR_GRADIENT)


@njit(cache=True, parallel=True, error_model='numpy')
def _gravity_matrix(points, blocks, size, top_depth, x_top, sd, cd, cr, sr, m, rho_g, beta, out):
    # out[b, s]: gravity change at point s for unit slip of block b
    for b in prange(blocks.shape[0]):
        for s in range(points.shape[0]):
            out[b, s] = _response(points[s, 0], points[s, 1], blocks[b, 0], blocks[b, 1], size, top_depth, x_top,
                                  sd, cd, cr, sr, m, rho_g, beta)[3]


@njit(cache=True, parallel=True, error_model='numpy')
def _field(points, blocks, slip, size, top_depth, x_top, sd, cd, cr, sr, m, rho_g, beta, out):
    # out[s] = (dX, dY, dZ, dg) at point s for the slips of the given blocks
    for s in prange(points.shape[0]):
        a0 = a1 = a2 = a3 = 0.0
        for b in range(blocks.shape[0]):
            if slip[b] == 0.0:
                continue
            r = _response(points[s, 0], points[s, 1], blocks[b, 0], blocks[b, 1], size, top_depth, x_top,
                          sd, cd, cr, sr, m, rho_g, beta)
            a0 += r[0] * slip[b]
            a1 += r[1] * slip[b]
            a2 += r[2] * slip[b]
            a3 += r[3] * slip[b]
        out[s, 0] = a0
        out[s, 1] = a1
        out[s, 2] = a2
        out[s, 3] = a3


def surface_points(config):
    """Horizontal (X, Y) positions of the sensors on the free surface (m): the sensor grid of
    grid.sensor_positions without its height."""
    return np.ascontiguousarray(grid.sensor_positions(config)[:, :2])


def coarse_patches(config, blocks, slip, factor):
    """Slip of the given blocks on square patches of factor x factor blocks: (centres (k, 2) in
    fault coordinates, mean slip (k,), patch side). Exact for slip uniform over each patch,
    and accurate at distances larger than the patch side."""
    blocks = np.asarray(blocks, dtype=np.int64)
    rows, cols = np.divmod(blocks, config.n_blocks_x)
    key = (rows // factor) * ((config.n_blocks_x + factor - 1) // factor) + cols // factor
    uniq, inverse = np.unique(key, return_inverse=True)
    mean = np.bincount(inverse, weights=np.asarray(slip, dtype=float)) / factor ** 2
    n_cx = (config.n_blocks_x + factor - 1) // factor
    pr, pc = np.divmod(uniq, n_cx)
    L = config.block_spacing_m
    centres = np.column_stack([(pc * factor + 0.5 * (factor - 1)) * L, (pr * factor + 0.5 * (factor - 1)) * L])
    return centres, mean, factor * L


def surface_field(config, points, blocks, slip, coarsen=1):
    """Surface displacement and gravity for slip (m) of the given blocks: array (n_points, 4)
    of dX, dY, dZ (m, X down-dip, Y along strike, Z up) and the gravity change (m/s^2, at a
    point moving with the ground, free-air included). coarsen > 1 sums patches of
    coarsen x coarsen blocks carrying their mean slip (coarse_patches)."""
    geo = _geometry(config)
    if coarsen > 1:
        pos, slip, geo['size'] = coarse_patches(config, blocks, slip, coarsen)
    else:
        pos = grid.block_positions(config)[np.asarray(blocks, dtype=np.int64), :2]
    out = np.zeros((len(points), 4))
    _field(np.ascontiguousarray(points, dtype=float), np.ascontiguousarray(pos, dtype=float),
           np.ascontiguousarray(slip, dtype=float), out=out, **geo)
    return out


def gravity_matrix(config, points=None):
    """(matrix, total): matrix[b, s] is the gravity change (m/s^2) at surface point s for 1 m of
    slip on block b (float32, blocks x points), total[s] the sum over all blocks (float64)."""
    points = surface_points(config) if points is None else np.ascontiguousarray(points, dtype=float)
    blocks = np.ascontiguousarray(grid.block_positions(config)[:, :2])
    out = np.zeros((len(blocks), len(points)), dtype=np.float32)
    _gravity_matrix(points, blocks, out=out, **_geometry(config))
    return out, out.sum(axis=0, dtype=np.float64)


@njit(cache=True)
def accumulate(matrix, indices, du, delta_g):
    """delta_g += sum_k matrix[indices[k]] * du[k] (rows of a blocks x points matrix)."""
    for k in range(indices.shape[0]):
        if du[k] == 0.0:
            continue
        row = matrix[indices[k]]
        for s in range(row.shape[0]):
            delta_g[s] += row[s] * du[k]
