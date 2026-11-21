# -*- coding: utf-8 -*-
"""
OTWNDATA seamless PBR stone-wall textures
=========================================

Procedural, seamlessly tiling, high-resolution stone masonry in the style of
the OTWNDATA.VGA wall tiles of Wizardry VII (1992).

Masonry structure
-----------------
* One periodic multi-scale power diagram: 49 large stones plus 16 medium
  stones in the natural junction voids.  All stones are real cells, so the
  medium stones never sit on top of the large ones.
* Large-stone linear size spread is 4.0x (max/min); actual layout is close
  to that target.  Outline corners are rounded with a periodic opening, so
  the stones read as worn, rounded polygons with no side-count limit.
* Joints are narrow (10 px at 2048 for gold/aged, 9 px for cyclopean).
  The height map uses a soft edge bevel and a continuous shallow joint
  band instead of a deep trench, so the joints are smooth and readable.
* 15 chips / pockmarks per texture.  They are subtle in albedo but retain
  strong relief in height/normal/AO.
* Procedural fractures are implemented but disabled by default; enable
  them with --cracks N.
* Colour palette sampled from OTWNDATA tile t000 (golden sandstone,
  salmon and yellow speckles, dark umber joints).

Outputs (2K, 2048x2048)
-----------------------
albedo_2k.png, height_2k.png, normal_gl_2k.png, normal_dx_2k.png,
roughness_2k.png, ao_2k.png  for every variant in ./textures_otwn/.

Requires numpy + Pillow.  On this machine they live in
E:\\Temp\\opencode\\pylibs.
"""

import os
import sys
import csv
import math
import argparse

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.path.insert(0, r"E:\Temp\opencode\pylibs")
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------------
# periodic noise helpers
# ----------------------------------------------------------------------------
def fft_noise(shape, beta, seed):
    """Periodic (tileable) Gaussian noise with spectral slope beta."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape).astype(np.float64)
    F = np.fft.rfft2(white)
    h, w = shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    n = np.fft.irfft2(F * amp, s=shape)
    n -= n.mean()
    s = n.std()
    if s > 1e-12:
        n /= s
    return n.astype(np.float32)


def gauss_blur(a, sigma):
    """Periodic Gaussian blur via FFT."""
    if sigma <= 0:
        return a.astype(np.float32, copy=True)
    h, w = a.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (kx * kx + ky * ky))
    F = np.fft.rfft2(a.astype(np.float64))
    return np.fft.irfft2(F * g, s=a.shape).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def periodic_distance(mask, pad):
    """Approximate Euclidean distance (chamfer 3x3) to True, wrapped."""
    h, w = mask.shape
    m = np.pad(mask.astype(bool), pad, mode="wrap")
    H, W = m.shape
    dist = np.where(m, 0.0, 1.0e9).astype(np.float32)
    idx = np.arange(W, dtype=np.float32)
    diag = np.float32(np.sqrt(2.0))
    for y in range(1, H):
        cur = dist[y]
        prev = dist[y - 1]
        cand = np.minimum(
            prev + 1.0,
            np.minimum(np.roll(prev, 1), np.roll(prev, -1)) + diag)
        np.minimum(cur, cand, out=cur)
        cur[:] = np.minimum(cur, idx + np.minimum.accumulate(cur - idx))
    for y in range(H - 2, -1, -1):
        cur = dist[y]
        nxt = dist[y + 1]
        cand = np.minimum(
            nxt + 1.0,
            np.minimum(np.roll(nxt, 1), np.roll(nxt, -1)) + diag)
        np.minimum(cur, cand, out=cur)
        rev = cur[::-1].copy()
        rev[:] = np.minimum(rev, idx + np.minimum.accumulate(rev - idx))
        cur[:] = rev[::-1]
    return dist[pad:pad + h, pad:pad + w]


def make_cracks(size, seed, count=9):
    """A few thin random-walk fractures, seamless across tile edges."""
    rng = np.random.default_rng(seed + 30)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    width = max(2, int(round(2.6 * size / 512.0)))
    for _ in range(count):
        x = rng.uniform(0.0, size)
        y = rng.uniform(0.0, size)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        segs = int(rng.integers(7, 13))
        step = rng.uniform(0.28, 0.52) * size / segs
        for s in range(segs):
            ang += rng.normal(0.0, 0.48)
            nx = x + math.cos(ang) * step
            ny = y + math.sin(ang) * step
            for ox in (-size, 0, size):
                for oy in (-size, 0, size):
                    draw.line([(x + ox, y + oy), (nx + ox, ny + oy)],
                              fill=255, width=width)
            x, y = nx, ny
        # one short side branch
        if rng.random() < 0.75:
            bx = x + rng.normal(0.0, step * 1.6)
            by = y + rng.normal(0.0, step * 1.6)
            for ox in (-size, 0, size):
                for oy in (-size, 0, size):
                    draw.line([(x + ox, y + oy), (bx + ox, by + oy)],
                              fill=255, width=max(1, width - 1))
    crack = np.asarray(img, np.float32) / 255.0
    crack = gauss_blur(crack, max(0.7, 0.0014 * size))
    return np.clip(crack * 1.25, 0.0, 1.0)


def srgb_to_lin(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def save_png(path, arr):
    a = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path, optimize=True)


def wrap(d):
    return (d + 0.5) % 1.0 - 0.5


# ----------------------------------------------------------------------------
# palettes sampled from OTWNDATA t000 (k-means, sRGB)
# ----------------------------------------------------------------------------
GOLD_PAL = np.array([
    [0.824, 0.714, 0.384],   # d2b662  pale sand
    [0.824, 0.667, 0.314],   # d2aa50  golden
    [0.761, 0.600, 0.282],   # c29948
    [0.698, 0.537, 0.267],   # b28944
    [0.605, 0.455, 0.202],   # 9a7433  shadow gold
    [0.833, 0.691, 0.093],   # d4b018  yellow mineral
    [0.932, 0.696, 0.489],   # eeb17d  salmon fleck
    [0.667, 0.522, 0.365],   # aa855d  dusty
], np.float32)
GOLD_W = np.array([0.48, 0.33, 0.12, 0.05, 0.02, 0.00, 0.00, 0.00],
                  np.float32)

AGED_PAL = np.array([
    [0.604, 0.522, 0.310],
    [0.588, 0.478, 0.260],
    [0.537, 0.439, 0.235],
    [0.478, 0.384, 0.208],
    [0.412, 0.325, 0.170],
    [0.627, 0.557, 0.380],
    [0.552, 0.503, 0.352],
    [0.500, 0.435, 0.290],
], np.float32)
AGED_W = np.array([0.28, 0.22, 0.16, 0.11, 0.08, 0.06, 0.05, 0.04],
                  np.float32)


# ----------------------------------------------------------------------------
# even (Lloyd-relaxed) distribution of big-stone centres on a periodic domain
# ----------------------------------------------------------------------------
def relax_centers(n, seed, iterations=8, res=256, init_jitter=0.32):
    rng = np.random.default_rng(seed)
    g = int(math.ceil(math.sqrt(n)))
    pts = []
    for j in range(g):
        for i in range(g):
            if len(pts) >= n:
                break
            pts.append([(i + 0.5 + rng.uniform(-init_jitter, init_jitter)) / g,
                        (j + 0.5 + rng.uniform(-init_jitter, init_jitter)) / g])
    pts = np.array(pts[:n], np.float64)

    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    xx = (xx + 0.5) / res
    yy = (yy + 0.5) / res
    for _ in range(iterations):
        d1 = np.full((res, res), 1e9, np.float32)
        id1 = np.zeros((res, res), np.int16)
        for k, (px, py) in enumerate(pts):
            dx = (xx - px + 0.5) % 1.0 - 0.5
            dy = (yy - py + 0.5) % 1.0 - 0.5
            d = dx * dx + dy * dy
            m = d < d1
            d1 = np.where(m, d, d1)
            id1 = np.where(m, k, id1)

        ids = id1.ravel()
        cnt = np.bincount(ids, minlength=n).astype(np.float64)
        for k in range(n):
            m = ids == k
            if not m.any():
                continue
            dx = (xx.ravel()[m] - pts[k, 0] + 0.5) % 1.0 - 0.5
            dy = (yy.ravel()[m] - pts[k, 1] + 0.5) % 1.0 - 0.5
            pts[k, 0] = (pts[k, 0] + dx.mean()) % 1.0
            pts[k, 1] = (pts[k, 1] + dy.mean()) % 1.0

    # tiny irregularity so the packing does not look like a perfect lattice
    pts += rng.uniform(-0.005, 0.005, pts.shape)
    pts %= 1.0
    return pts.astype(np.float32)


def optimize_weights(centers, target, res=256, iterations=80, step=0.55):
    """Simple power-weight optimisation for prescribed cell areas."""
    n = len(centers)
    weights = np.zeros(n, np.float32)
    for _ in range(iterations):
        ids = power_ids(weights, centers, res)
        area = np.bincount(ids.ravel(), minlength=n).astype(np.float32)
        area /= float(res * res)
        weights += step * (target - area)
    return weights


def power_fields(centers, weights, size, warp=None):
    """Full-resolution power diagram labels and distance to the cell edge."""
    h = w = size
    Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
    X0 = (X0 + 0.5) / w
    Y0 = (Y0 + 0.5) / h
    if warp is not None:
        X0 = (X0 + warp[0]) % 1.0
        Y0 = (Y0 + warp[1]) % 1.0
    pd1 = np.full((h, w), 1e9, np.float32)
    pd2 = np.full((h, w), 1e9, np.float32)
    ids = np.zeros((h, w), np.int16)
    ids2 = np.zeros((h, w), np.int16)
    for si, (cx, cy) in enumerate(centers):
        dx = (X0 - cx + 0.5) % 1.0 - 0.5
        dy = (Y0 - cy + 0.5) % 1.0 - 0.5
        pd = dx * dx + dy * dy - weights[si]
        m1 = pd < pd1
        m2 = (~m1) & (pd < pd2)
        ids2 = np.where(m1, ids, np.where(m2, si, ids2))
        pd2 = np.where(m1, pd1, np.where(m2, pd, pd2))
        ids = np.where(m1, si, ids)
        pd1 = np.where(m1, pd, pd1)

    p1 = centers[ids]
    p2 = centers[ids2]
    ddx = wrap(p2[..., 0] - p1[..., 0])
    ddy = wrap(p2[..., 1] - p1[..., 1])
    site_dist = np.sqrt(ddx * ddx + ddy * ddy)
    bedge = (pd2 - pd1) / (2.0 * np.maximum(site_dist, 1e-6))
    bedge = np.maximum(bedge, 0.0)
    return ids, bedge


def build_multi_layout(size, seed, spread=1.1, medium_count=10, # 4  24
                       medium_area=0.09, joint_half=16.0 / 4096.0,
                       round_sigma=0.015, verbose=True): # 0.005
    """One power diagram with strongly varied big stones and smaller
    stones in the junction voids.  Returns ids, bedge_raw, bedge_s, prof,
    rim, inradius, cell_area, centers, weights, n_big, n_med."""
    h = w = size
    n_big = 49
    rng = np.random.default_rng(seed + 21)
    cxy = relax_centers(n_big, seed + 21, iterations=8)
    cxy[:, 0] = (cxy[:, 0]
                 + 0.010 * np.sin(2.0 * math.pi * (2.0 * cxy[:, 1] + 0.17))
                 + 0.005 * np.sin(2.0 * math.pi * (3.0 * cxy[:, 0] + 0.41))) % 1.0
    cxy[:, 1] = (cxy[:, 1]
                 + 0.010 * np.sin(2.0 * math.pi * (2.0 * cxy[:, 0] + 0.53))
                 + 0.005 * np.sin(2.0 * math.pi * (3.0 * cxy[:, 1] + 0.23))) % 1.0

    # Most stones cluster around the middle, with a few true outliers.
    # Linear sizes span 1..spread, so areas span 1..spread^2, but a
    # beta distribution avoids the thin slivers produced by a log ramp.
    u = rng.beta(2.0, 2.0, n_big)
    u[0], u[-1] = 0.0, 1.0
    lin = 1.0 + (float(max(1.0, spread)) - 1.0) * u
    rng.shuffle(lin)
    target_big = lin * lin
    target_big /= target_big.sum()

    # preliminary big-only diagram, just to find the junction voids
    w_big = optimize_weights(cxy, target_big, res=192, iterations=140, step=0.35)
    verts = find_power_vertices(cxy, w_big, verbose=verbose)

    # Reject junctions sitting exactly on the periodic border: their
    # cells become thin strips split between opposite tile edges.
    border = 0.005
    pool = np.nonzero((verts[:, 0] > border) & (verts[:, 0] < 1.0 - border) &
                      (verts[:, 1] > border) & (verts[:, 1] < 1.0 - border))[0]
    n_med = min(max(0, int(medium_count)), len(pool))
    med_pts = np.zeros((0, 2), np.float64)
    if n_med > 0:
        # independent stream: medium placement must not drift when the
        # big-stone size distribution changes
        rng_med = np.random.default_rng(seed + 22)
        used = np.zeros(len(pool), bool)
        dist = np.full(len(pool), 1e9, np.float64)
        chosen = []
        for s in range(n_med):
            if s == 0:
                idx = int(rng_med.integers(len(pool)))
            else:
                valid = (~used) & (dist > 0.055)
                if not valid.any():
                    valid = ~used
                cand = np.nonzero(valid)[0]
                score = dist[cand] + rng_med.normal(0.0, 0.018, len(cand))
                idx = int(cand[int(np.argmax(score))])
            used[idx] = True
            chosen.append(idx)
            dd = wrap(verts[pool] - verts[pool][idx])
            dist = np.minimum(dist, np.sqrt((dd * dd).sum(axis=1)))
        med_pts = verts[pool[chosen]].copy()
        med_pts += rng_med.normal(0.0, 0.004, med_pts.shape)
        med_pts %= 1.0

    if n_med > 0:
        area_med = float(np.clip(medium_area, 0.0, 0.35))
        target = np.concatenate([
            target_big * (1.0 - area_med),
            np.full(n_med, area_med / n_med),
        ]).astype(np.float32)
    else:
        target = target_big.astype(np.float32)
    target /= target.sum()

    centers = np.vstack([cxy, med_pts]).astype(np.float32)
    weights = optimize_weights(centers, target, res=256,
                               iterations=250, step=0.30)

    # smooth domain warp: turns straight Voronoi edges into curved,
    # stone-like outlines while every shared edge stays shared
    warp_x = fft_noise((h, w), 2.2, seed + 20)
    warp_y = fft_noise((h, w), 2.2, seed + 21)
    warp_x = gauss_blur(warp_x, max(0.8, 0.0012 * size))
    warp_y = gauss_blur(warp_y, max(0.8, 0.0012 * size))
    warp = (0.0055 * warp_x, 0.0055 * warp_y)

    ids, bedge = power_fields(centers, weights, size, warp=warp)
    bedge_s = gauss_blur(bedge, round_sigma * size)
    n_shape = fft_noise((h, w), 3.0, seed + 6)
    n_shape = gauss_blur(n_shape, max(1.0, 0.0018 * size))
    n_edge = fft_noise((h, w), 9.0, seed + 11)
    n_edge = gauss_blur(n_edge, max(0.8, 0.0012 * size))
    bedge_s += 0.0022 * n_shape + 0.0030 * n_edge

    inradius = np.zeros(len(centers), np.float32)
    np.maximum.at(inradius, ids.ravel(), bedge_s.ravel())
    inradius = np.maximum(inradius, 1e-3)
    cell_area = np.bincount(ids.ravel(), minlength=len(centers)).astype(np.float32)
    cell_area /= float(h * w)

    t = np.clip((bedge_s - joint_half) /
                np.maximum(inradius[ids] - joint_half, 1e-4), 0.0, 1.0)
    prof = (1.0 - (1.0 - t) ** 2.3) ** 0.55
    prof *= smoothstep(0.0, 0.085, t)
    rim = np.exp(-((bedge_s - joint_half) / 0.006) ** 2)

    if verbose:
        eq = np.sqrt(cell_area[:n_big] / math.pi)
        ratio = float(eq.max() / max(eq.min(), 1e-9))
        print("    multi layout: %d big + %d junction stones, "
              "big size max/min %.2f (target %.2f)"
              % (n_big, n_med, ratio, spread))
    return (ids, bedge, bedge_s, prof, rim, inradius, cell_area,
            centers, weights, n_big, n_med)


# ----------------------------------------------------------------------------
# big stones: weighted periodic Voronoi, rounded corners, domed tops
# ----------------------------------------------------------------------------
def power_ids(weights, centers, res):
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    xx = (xx + 0.5) / res
    yy = (yy + 0.5) / res
    best = np.full((res, res), 1e9, np.float32)
    ids = np.zeros((res, res), np.int16)
    for k, (cx, cy) in enumerate(centers):
        dx = (xx - cx + 0.5) % 1.0 - 0.5
        dy = (yy - cy + 0.5) % 1.0 - 0.5
        pd = dx * dx + dy * dy - weights[k]
        m = pd < best
        best = np.where(m, pd, best)
        ids = np.where(m, k, ids)
    return ids


def build_big_layout(size, seed, joint_half=0.0085, round_sigma=0.022,
                     target_lo=0.90, target_hi=1.10, verbose=True):
    """Return big_id, raw bedge, smoothed bedge, prof, rim, inradius,
    cell_area, cxy, weights, target."""
    h = w = size
    n = 49
    rng = np.random.default_rng(seed + 21)
    cxy = relax_centers(n, seed + 21, iterations=8)
    # smooth periodic warp: bends the 7x7 rows so they do not read as a grid
    cxy[:, 0] = (cxy[:, 0]
                 + 0.010 * np.sin(2.0 * math.pi * (2.0 * cxy[:, 1] + 0.17))
                 + 0.005 * np.sin(2.0 * math.pi * (3.0 * cxy[:, 0] + 0.41))) % 1.0
    cxy[:, 1] = (cxy[:, 1]
                 + 0.010 * np.sin(2.0 * math.pi * (2.0 * cxy[:, 0] + 0.53))
                 + 0.005 * np.sin(2.0 * math.pi * (3.0 * cxy[:, 1] + 0.23))) % 1.0

    # target sizes: linear spread <= 30 %, shuffled over the layout
    lin = np.linspace(target_lo, target_hi, n).astype(np.float32)
    rng.shuffle(lin)
    target = lin * lin
    target /= target.sum()

    # optimise power weights at a coarse resolution
    res = 256
    weights = np.zeros(n, np.float32)
    for _ in range(80):
        ids = power_ids(weights, cxy, res)
        area = np.bincount(ids.ravel(), minlength=n).astype(np.float32)
        area /= float(res * res)
        weights += 0.55 * (target - area)

    # full-resolution power diagram and second nearest cell
    Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
    X0 = (X0 + 0.5) / w
    Y0 = (Y0 + 0.5) / h
    pd1 = np.full((h, w), 1e9, np.float32)
    pd2 = np.full((h, w), 1e9, np.float32)
    big_id = np.zeros((h, w), np.int16)
    big_id2 = np.zeros((h, w), np.int16)
    for si, (cx, cy) in enumerate(cxy):
        dx = (X0 - cx + 0.5) % 1.0 - 0.5
        dy = (Y0 - cy + 0.5) % 1.0 - 0.5
        pd = dx * dx + dy * dy - weights[si]
        m1 = pd < pd1
        m2 = (~m1) & (pd < pd2)
        big_id2 = np.where(m1, big_id, np.where(m2, si, big_id2))
        pd2 = np.where(m1, pd1, np.where(m2, pd, pd2))
        big_id = np.where(m1, si, big_id)
        pd1 = np.where(m1, pd, pd1)

    p1 = cxy[big_id]
    p2 = cxy[big_id2]
    ddx = wrap(p2[..., 0] - p1[..., 0])
    ddy = wrap(p2[..., 1] - p1[..., 1])
    site_dist = np.sqrt(ddx * ddx + ddy * ddy)
    bedge = (pd2 - pd1) / (2.0 * np.maximum(site_dist, 1e-6))
    bedge = np.maximum(bedge, 0.0)

    # Gaussian smoothing of the distance field rounds the polygon corners
    bedge_s = gauss_blur(bedge, round_sigma * size)
    # gentle low-frequency wobble: hand-laid, not laser-cut outlines
    n_shape = fft_noise((h, w), 3.0, seed + 6)
    n_shape = gauss_blur(n_shape, max(1.0, 0.0018 * size))
    bedge_s += 0.0012 * n_shape

    inradius = np.zeros(n, np.float32)
    np.maximum.at(inradius, big_id.ravel(), bedge_s.ravel())
    inradius = np.maximum(inradius, 1e-3)

    cell_area = np.bincount(big_id.ravel(), minlength=n).astype(np.float32)
    cell_area /= float(h * w)

    # rounded dome: steep bevel at the joint, gentle top
    t = np.clip((bedge_s - joint_half) /
                np.maximum(inradius[big_id] - joint_half, 1e-4), 0.0, 1.0)
    prof = (1.0 - (1.0 - t) ** 2.3) ** 0.55
    prof *= smoothstep(0.0, 0.085, t)
    rim = np.exp(-((bedge_s - joint_half) / 0.010) ** 2)

    if verbose:
        eq = np.sqrt(cell_area / math.pi)
        ratio = eq.max() / max(eq.min(), 1e-9)
        # neighbour/side counts of the polygon cells (coarse diagram);
        # ignore hairline slivers shorter than ~3 % of the texture
        ids = power_ids(weights, cxy, 384)
        counts = np.zeros((n, n), np.int64)
        for axis in (0, 1):
            if axis == 0:
                a, b = ids[:-1, :].ravel(), ids[1:, :].ravel()
            else:
                a, b = ids[:, :-1].ravel(), ids[:, 1:].ravel()
            m = a != b
            lo = np.minimum(a[m], b[m]).astype(np.int64)
            hi = np.maximum(a[m], b[m]).astype(np.int64)
            cnt = np.bincount(lo * n + hi, minlength=n * n).reshape(n, n)
            counts += cnt + cnt.T
        nb = (counts >= 12).sum(axis=1)
        print("    big stones: %d, sides %d..%d, size max/min %.2f"
              % (n, int(nb[nb > 0].min()), int(nb.max()), ratio))
    return (big_id, bedge, bedge_s, prof, rim, inradius, cell_area, cxy,
            weights, target)


# ----------------------------------------------------------------------------
# medium junction stones: placed at power-diagram vertices
# ----------------------------------------------------------------------------
def find_power_vertices(cxy, weights, verbose=True):
    """Vertices of the periodic power diagram (triple points)."""
    n = len(cxy)
    pts = []
    seen = set()
    for i in range(n - 2):
        ci = cxy[i]
        wi = float(weights[i])
        for j in range(i + 1, n - 1):
            d1 = wrap(cxy[j] - ci)
            l1 = float(d1 @ d1)
            wj = float(weights[j])
            for k in range(j + 1, n):
                d2 = wrap(cxy[k] - ci)
                det = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(det) < 1e-10:
                    continue
                b1 = 0.5 * (l1 + wi - wj)
                b2 = 0.5 * (float(d2 @ d2) + wi - float(weights[k]))
                qx = (b1 * d2[1] - d1[1] * b2) / det
                qy = (d1[0] * b2 - b1 * d2[0]) / det
                p = (ci + np.array([qx, qy], np.float64)) % 1.0

                dd = wrap(cxy - p)
                pd = np.einsum("ij,ij->i", dd, dd) - weights
                idx = np.argpartition(pd, 3)[:3]
                if not {i, j, k}.issubset(set(idx.tolist())):
                    continue
                pds = pd[[i, j, k]]
                if float(pds.max() - pds.min()) > 1e-5:
                    continue

                key = (int(round(p[0] * 200000.0)),
                       int(round(p[1] * 200000.0)))
                if key in seen:
                    continue
                seen.add(key)
                pts.append(p)

    pts = np.array(pts, np.float64) if pts else np.zeros((0, 2))
    if verbose:
        print("    junction candidates: %d power-diagram vertices" % len(pts))
    return pts


def build_medium_layer(size, seed, cxy, weights, inradius, joint_half,
                       count=44, scale=0.32, verbose=True):
    """Rounded full-height stones wedged into the big-cell junctions.

    They replace area of the big stones; the joint width stays the same.
    Returns med_id, med_q, med_prof, med_mask, med_carve, pts, radii.
    """
    h = w = size
    rng = np.random.default_rng(seed + 303)
    verts = find_power_vertices(cxy, weights, verbose=verbose)
    if count <= 0 or len(verts) == 0:
        z = np.zeros((h, w), np.float32)
        return (np.zeros((h, w), np.int32), z + 1.0, z, z, z,
                np.zeros((0, 2)), np.zeros(0), 0)

    # local big-stone size at each junction
    local_inr = np.zeros(len(verts), np.float32)
    for q, p in enumerate(verts):
        dd = wrap(cxy - p)
        pd = np.einsum("ij,ij->i", dd, dd) - weights
        nb = np.argsort(pd)[:3]
        local_inr[q] = float(np.median(inradius[nb]))

    # irregular selection with a minimum spacing, biased to spread out
    n_sel = min(int(count), len(verts))
    min_sep = 0.045
    used = np.zeros(len(verts), bool)
    dist = np.full(len(verts), 1e9, np.float64)
    chosen = []
    for s in range(n_sel):
        if s == 0:
            idx = int(rng.integers(len(verts)))
        else:
            valid = (~used) & (dist > min_sep)
            if not valid.any():
                valid = ~used
            cand = np.nonzero(valid)[0]
            score = dist[cand] + rng.normal(0.0, 0.018, len(cand))
            idx = int(cand[int(np.argmax(score))])
        used[idx] = True
        chosen.append(idx)
        dd = wrap(verts - verts[idx])
        dist = np.minimum(dist, np.sqrt((dd * dd).sum(axis=1)))

    pts = verts[chosen].copy()
    # small hand-laid jitter, still inside the junction area
    pts += rng.normal(0.0, 0.005, pts.shape)
    pts %= 1.0

    radii = scale * local_inr[chosen]
    radii *= rng.uniform(0.78, 1.22, n_sel)
    radii = np.clip(radii, 0.010, 0.050).astype(np.float32)

    asp = rng.uniform(0.80, 1.22, n_sel)
    rot = rng.uniform(0.0, math.pi, n_sel).astype(np.float32)
    lobe_a = rng.uniform(0.025, 0.080, n_sel)
    lobe_b = rng.uniform(0.012, 0.045, n_sel)
    ph1 = rng.uniform(0.0, 2.0 * math.pi, n_sel)
    ph2 = rng.uniform(0.0, 2.0 * math.pi, n_sel)

    Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
    X0 = (X0 + 0.5) / w
    Y0 = (Y0 + 0.5) / h

    qmin = np.full((h, w), 1e9, np.float32)
    mid = np.zeros((h, w), np.int32)
    body = np.zeros((h, w), np.float32)
    carve = np.zeros((h, w), np.float32)

    for k in range(n_sel):
        r = float(radii[k])
        dx = wrap(X0 - float(pts[k, 0]))
        dy = wrap(Y0 - float(pts[k, 1]))
        ca, sa = math.cos(float(rot[k])), math.sin(float(rot[k]))
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        rx = r * math.sqrt(float(asp[k]))
        ry = r / math.sqrt(float(asp[k]))
        q = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        th = np.arctan2(v, u)
        q *= (1.0 + lobe_a[k] * np.sin(4.0 * th + ph1[k])
              + lobe_b[k] * np.sin(6.0 * th + ph2[k]))

        # body ends one joint-half before the nominal radius; the big
        # stones are cut back one joint-half after it, leaving a dark ring
        jr = min(0.75, joint_half / max(r, 1e-6))
        soft_q = max(0.028, 1.4 / (size * max(r, 1e-6)))
        b = 1.0 - smoothstep(1.0 - jr, 1.0 - jr + soft_q, q)
        c = 1.0 - smoothstep(1.0 + jr - soft_q, 1.0 + jr + soft_q, q)

        m = q < qmin
        qmin = np.where(m, q, qmin)
        mid = np.where(m, k, mid)
        body = np.where(m, b, body)
        carve = np.where(m, c, carve)

    prof = (1.0 - np.clip(qmin, 0.0, 1.0) ** 2) ** 0.55
    prof = (prof * body).astype(np.float32)
    if verbose:
        print("    medium junction stones: %d, radius %.1f..%.1f px, "
              "area %.1f%%"
              % (n_sel, float(radii.min() * size), float(radii.max() * size),
                 100.0 * float(body.mean())))
    return (mid.astype(np.int32), qmin.astype(np.float32), prof,
            body, carve, pts, radii, n_sel)


# ----------------------------------------------------------------------------
# small rounded joint fillers (legacy, not used by the junction layout)
# ----------------------------------------------------------------------------
def build_small_layer(size, seed, grid=86, verbose=True):
    """Return small_id, small_q, small_prof.  Periodic jittered pebble grid."""
    h = w = size
    rng = np.random.default_rng(seed + 101)
    g = grid
    # jittered centres and per-stone elliptical shape
    jx = rng.uniform(-0.40, 0.40, (g, g)).astype(np.float32)
    jy = rng.uniform(-0.40, 0.40, (g, g)).astype(np.float32)
    asp = rng.uniform(0.84, 1.18, (g, g)).astype(np.float32)
    rot = rng.uniform(0.0, math.pi, (g, g)).astype(np.float32)
    rad = rng.uniform(0.58, 0.64, (g, g)).astype(np.float32)  # in cell units
    ph = rng.uniform(0.0, 2.0 * math.pi, (g, g)).astype(np.float32)

    Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
    X0 = (X0 + 0.5) / w
    Y0 = (Y0 + 0.5) / h

    qmin = np.full((h, w), 1e9, np.float32)
    dmin = np.full((h, w), 1e9, np.float32)
    sid = np.zeros((h, w), np.int32)
    cell = 1.0 / g

    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            ix = np.floor(X0 * g).astype(np.int32) + ox
            iy = np.floor(Y0 * g).astype(np.int32) + oy
            ii = (ix % g).astype(np.int32)
            jj = (iy % g).astype(np.int32)
            cx = (ix + 0.5 + jx[jj, ii]) * cell
            cy = (iy + 0.5 + jy[jj, ii]) * cell
            dx = wrap(X0 - cx)
            dy = wrap(Y0 - cy)
            ca = np.cos(rot[jj, ii])
            sa = np.sin(rot[jj, ii])
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            a = cell * rad[jj, ii] * np.sqrt(asp[jj, ii])
            b = cell * rad[jj, ii] / np.sqrt(asp[jj, ii])
            q = np.sqrt((u / a) ** 2 + (v / b) ** 2)
            # slightly irregular outline
            th = np.arctan2(v, u)
            q *= (1.0 + 0.018 * np.sin(3.0 * th + ph[jj, ii]))
            # rounded pebble: the smallest normalised radius wins, so two
            # overlapping pebbles meet along a smooth curved boundary
            m = q < qmin
            qmin = np.where(m, q, qmin)
            sid = np.where(m, jj * g + ii, sid)

    qmin = np.maximum(qmin, 0.0)
    prof = (1.0 - np.clip(qmin, 0.0, 1.0) ** 2) ** 0.45
    if verbose:
        cover = float((qmin < 1.0).mean())
        print("    joint fillers: %dx%d rounded pebbles (%s), coverage %.1f%%"
              % (g, g, "full height", cover * 100.0))
    return sid, qmin, prof


# ----------------------------------------------------------------------------
# texture generator
# ----------------------------------------------------------------------------
def generate(size=2048, seed=7, variant="gold", suffix="2k", outdir=".",
             verbose=True, joint_px=None, dome=1.0, chip_count=15,
             crack_count=0, spread=4.0, medium_count=16, medium_area=0.09,
             seam_halo=0.0050, seam_halo_amt=0.30,
             normal_detail=1.0):
    h = w = size
    rng = np.random.default_rng(seed)

    # ---- periodic noise fields -------------------------------------------
    n_low = fft_noise((h, w), 3.2, seed + 1)
    n_med = fft_noise((h, w), 2.0, seed + 2)
    n_fine = fft_noise((h, w), 0.9, seed + 3)
    n_fine2 = fft_noise((h, w), 0.7, seed + 4)
    n_tint = fft_noise((h, w), 2.4, seed + 5)
    n_shape = fft_noise((h, w), 3.0, seed + 6)
    n_pit = fft_noise((h, w), 1.5, seed + 7)
    n_stri = fft_noise((h, w), 1.2, seed + 8)
    n_dark = fft_noise((h, w), 0.8, seed + 9)
    n_spot = fft_noise((h, w), 1.3, seed + 10)
    n_micro = fft_noise((h, w), 0.55, seed + 11)

    n_low_s = gauss_blur(n_low, 1.0)
    n_med_s = gauss_blur(n_med, 1.0)
    n_fine_s = gauss_blur(n_fine, 0.8)
    n_shape_s = gauss_blur(n_shape, 1.6)

    Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
    X0 = (X0 + 0.5) / w
    Y0 = (Y0 + 0.5) / h

    # ---- variant parameters ----------------------------------------------
    if variant == "aged":
        big_amp, big_base = 0.74, 0.25
        joint_px_default = 10.0
        big_rough, small_rough = 0.60, 0.68
        moss_amt = 0.38
        pal, pal_w = AGED_PAL, AGED_W
        mortar = np.array([0.105, 0.080, 0.050], np.float32)
        dirt = np.array([0.160, 0.145, 0.086], np.float32)
        small_pal = pal * np.array([0.94, 0.96, 0.92], np.float32)
        fill_grid = 88
    elif variant == "cyclopean":
        big_amp, big_base = 0.95, 0.24
        joint_px_default = 9.0
        big_rough, small_rough = 0.50, 0.60
        moss_amt = 0.0
        pal, pal_w = GOLD_PAL, GOLD_W
        mortar = np.array([0.160, 0.110, 0.058], np.float32)
        dirt = np.array([0.240, 0.170, 0.090], np.float32)
        small_pal = pal[:5] * np.array([0.90, 0.88, 0.84], np.float32)
        fill_grid = 80
    else:
        big_amp, big_base = 0.82, 0.25
        joint_px_default = 10.0
        big_rough, small_rough = 0.54, 0.64
        moss_amt = 0.0
        pal, pal_w = GOLD_PAL, GOLD_W
        mortar = np.array([0.145, 0.100, 0.052], np.float32)
        dirt = np.array([0.235, 0.165, 0.088], np.float32)
        small_pal = pal[:5] * np.array([0.92, 0.90, 0.86], np.float32)
        fill_grid = 86

    # ---- joint width parameter -------------------------------------------
    # joint_px is the *full* visible joint width in pixels at 2048.
    # 0 means: no small fillers at all, just a thin dark mortar seam.
    req_joint_px = float(joint_px_default if joint_px is None else joint_px)
    joint_on = req_joint_px > 0.5
    if joint_on:
        joint_px_eff = req_joint_px
        joint_half = joint_px_eff / 4096.0
        seam_half = 0.0
    else:
        # stones touch; a thin dark mortar line is painted on the seam
        joint_px_eff = 8.0
        joint_half = 0.0
        seam_half = joint_px_eff / 4096.0
    dome = float(np.clip(dome, 0.0, 2.0))

    # ---- multi-scale stone layout -----------------------------------------
    (ids, bedge_raw, bedge, prof, rim, inradius, cell_area,
     centers, weights, n_big, n_med) = build_multi_layout(
        size, seed, spread=spread, medium_count=medium_count if joint_on else 0,
        medium_area=medium_area, joint_half=joint_half,
        round_sigma=0.009, verbose=verbose)
    n_total = n_big + n_med

    # morphological opening with a real wrapped distance transform:
    # erode the cells by round_px, then dilate by the same radius.  This
    # rounds the polygon corners while keeping the original joint width.
    soft_px = max(0.8, 0.0015 * size)
    pad = max(16, int(0.020 * size))
    if joint_on:
        round_px = max(2.0, 0.0140 * size)
        core = bedge_raw > (round_px / size)
        dist_core = periodic_distance(core, pad)
        stone_mask = 1.0 - smoothstep(round_px - soft_px,
                                      round_px + soft_px, dist_core)
        # rounded rim shading and seam follow the opened silhouette
        rim_w = max(1.0, 0.003 * size)
        rim = np.exp(-((dist_core - round_px) / rim_w) ** 2)
    else:
        # touching-stone mode: keep the original silhouette, no gap fill
        dist_core = np.zeros_like(bedge_raw)
        stone_mask = smoothstep(-soft_px / size, soft_px / size, bedge_raw)
        rim_w = max(1.0, 0.003 * size)
        rim = np.exp(-(bedge_raw / rim_w) ** 2)
    big_mask = stone_mask * (ids < n_big)
    med_mask = stone_mask * (ids >= n_big)
    gap = 1.0 - stone_mask
    # A wider, soft mask is used only for albedo.  It keeps the thin dark
    # joint line, but gives the joint -> stone colour transition a gradual
    # mortar/stone gradient instead of a hard cut.
    # The edge width must be continuous across a large/medium boundary.
    # A per-cell radius jumps at the seam, so use a smooth local-minimum
    # radius instead: a p-norm over a tiny blur approximates min() and keeps
    # both sides of the joint on the same transition width.
    inr_norm = np.maximum(inradius[ids], 1.0 / size)
    invp = 1.0 / inr_norm ** 6
    local_inr_norm = gauss_blur(invp, max(1.0, 0.0100 * size)) ** (-1.0 / 6.0)
    local_inr = np.maximum(local_inr_norm * size, 1.0)
    large_gate = smoothstep(0.035, 0.055, local_inr_norm)
    soft_col = max(2.0, 0.010 * size)  # 0.0035
    soft_col_px = soft_col * (0.40 + 0.60 * large_gate)
    if joint_on:
        stone_col_mask = 1.0 - smoothstep(round_px - soft_col_px,
                                          round_px + soft_col_px, dist_core)
    else:
        stone_col_mask = smoothstep(-soft_col_px / size,
                                    soft_col_px / size, bedge_raw)
    gap_col_mask = 1.0 - stone_col_mask
    # a wider bevel mask is used only for height: it adds a soft edge facet
    # and removes the hard, jagged step at the joints
    soft_h = max(1.5, 0.0035 * size)
    if joint_on:
        stone_h = 1.0 - smoothstep(round_px - soft_h,
                                   round_px + soft_h, dist_core)
    else:
        stone_h = smoothstep(-soft_h / size, soft_h / size, bedge_raw)
    # smooth the bevel contour itself: removes the last hard pixel steps
    stone_h = gauss_blur(stone_h, max(0.8, 0.0012 * size))
    gap_h = 1.0 - stone_h
    # body profile uses the already smoothed cell-edge distance; the raw
    # field has small slope breaks at medial axes that become visible
    # radial facets in the 2K normal map
    inr_s = np.zeros(n_total, np.float32)
    np.maximum.at(inr_s, ids.ravel(), bedge.ravel())
    t = np.clip((bedge - joint_half) /
                np.maximum(inr_s[ids] - joint_half, 1e-4), 0.0, 1.0)
    prof = (1.0 - (1.0 - t) ** 2.3) ** 0.55
    prof *= smoothstep(0.0, 0.085, t)

    # Shared, per-stone normalised distance from the rounded edge to the
    # centre.  The albedo already uses a wide gradient of this kind; height,
    # normals and roughness now use the same edge band instead of a narrow
    # hard joint.
    edge_start = joint_half if joint_on else 0.0
    t_edge = np.clip((bedge - edge_start) /
                     np.maximum(inr_s[ids] - edge_start, 1e-4), 0.0, 1.0)
    edge_frac = float(np.clip(seam_halo * 5.5, 0.05, 0.15))
    # Match the albedo's wide colour transition, but never let the band eat
    # more than a quarter of a small junction stone.  `local_inr` is the
    # same seam-continuous radius used by the albedo.
    height_small = 0.55 + 0.45 * large_gate
    edge_px = np.maximum(
        edge_frac * height_small * local_inr,
        np.minimum(soft_col * height_small, 0.25 * local_inr))
    edge_ramp = smoothstep(0.0, np.maximum(edge_px, 1e-4),
                           (bedge - edge_start) * size)

    # procedural fractures are disabled for now; keep the generator for later
    if crack_count > 0:
        crack = make_cracks(size, seed, crack_count) * stone_mask
    else:
        crack = np.zeros_like(stone_mask)

    # ---- per-cell style ---------------------------------------------------
    cell_col = np.zeros((n_total, 3), np.float32)
    pal_p = pal_w / pal_w.sum()
    cell_col[:n_big] = pal[rng.choice(len(pal), n_big, p=pal_p)]
    cell_col[:n_big] *= rng.uniform(0.97, 1.08, (n_big, 1)).astype(np.float32)
    if n_med:
        cell_col[n_big:] = pal[rng.choice(len(pal), n_med, p=pal_p)]
        cell_col[n_big:] *= rng.uniform(0.95, 1.06, (n_med, 1)).astype(np.float32)

    # keep every stone inside the warm sandstone family: pull dark/odd
    # palette picks toward the variant's average and clamp their luminance
    pal_base = (pal * pal_w[:, None]).sum(0) / max(pal_w.sum(), 1e-6)
    cell_col = 0.90 * cell_col + 0.10 * pal_base[None, :]
    lum = cell_col @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum_floor = {"aged": 0.50, "cyclopean": 0.58}.get(variant, 0.60)
    cell_col *= np.maximum(1.0, lum_floor / np.maximum(lum, 1e-6))[:, None]

    cell_hj = rng.normal(0.0, 1.0, n_total).astype(np.float32)
    cell_rot = rng.uniform(0.0, math.pi, n_total).astype(np.float32)
    cell_asp = rng.uniform(0.82, 1.18, n_total).astype(np.float32)
    cell_freq = rng.uniform(7.0, 13.0, n_total).astype(np.float32)
    cell_phase = rng.uniform(0.0, 2.0 * math.pi, n_total).astype(np.float32)
    cell_rj = rng.normal(0.0, 1.0, n_total).astype(np.float32)

    dxc = wrap(X0 - centers[ids, 0])
    dyc = wrap(Y0 - centers[ids, 1])
    ca = np.cos(cell_rot[ids])
    sa = np.sin(cell_rot[ids])
    u = dxc * ca + dyc * sa
    v = -dxc * sa + dyc * ca

    # rotated elliptical top cap, silently blended into the cell dome
    a_cap = inradius[ids] * 0.62 * cell_asp[ids]
    b_cap = inradius[ids] * 0.62 / cell_asp[ids]
    rr = np.sqrt((u / np.maximum(a_cap, 1e-4)) ** 2 +
                 (v / np.maximum(b_cap, 1e-4)) ** 2)
    cap = (1.0 - np.clip(rr, 0.0, 1.0) ** 2) ** 0.55
    prof_shape = prof * (0.90 + 0.10 * cap)

    # rotated bedding striations (very subtle relief + colour)
    stri = np.sin(2.0 * math.pi * (u * cell_freq[ids] + cell_phase[ids]))
    stri *= 0.5 + 0.5 * n_stri

    # painted dark mortar seam when --joint 0 (stones touch)
    if joint_on:
        seam = np.zeros((h, w), np.float32)
    else:
        seam = 1.0 - smoothstep(seam_half * 0.45, seam_half * 1.35, bedge_raw)

    # ---- sparse deep chips: <= one per three large stones ----------------
    n_chips = int(np.clip(chip_count, 0, max(0, n_big // 3)))
    chip_ids = rng.choice(n_big, size=n_chips, replace=False)
    # guaranteed size tiers: many tiny pockmarks, a few medium chips and
    # one or two large broken-out hollows per texture
    n_chip_large = max(1, int(round(n_chips * 0.15))) if n_chips >= 4 else 0
    n_chip_med = max(1, int(round(n_chips * 0.30))) if n_chips >= 2 else 0
    n_chip_small = max(0, n_chips - n_chip_large - n_chip_med)
    if n_chip_large + n_chip_med + n_chip_small < n_chips:
        n_chip_small += n_chips - n_chip_large - n_chip_med
    chip_scales = np.concatenate([
        10.0 ** rng.uniform(math.log10(0.030), math.log10(0.080), n_chip_small),
        10.0 ** rng.uniform(math.log10(0.090), math.log10(0.200), n_chip_med),
        10.0 ** rng.uniform(math.log10(0.220), math.log10(0.360), n_chip_large),
    ]).astype(np.float32)
    rng.shuffle(chip_scales)
    if verbose:
        print("    chip sizes (fraction of inradius): %s"
              % ", ".join("%.3f" % v for v in np.sort(chip_scales)[::-1]))
    chip_mask = np.zeros((h, w), np.float32)
    chip_depth = np.zeros((h, w), np.float32)
    chip_core = np.zeros((h, w), np.float32)
    for k, cid in enumerate(chip_ids):
        cx, cy = centers[cid]
        inr = float(inradius[cid])
        r_chip = inr * float(chip_scales[k])
        # keep even large chips away from the stone boundary
        max_off = 0.55 * max(0.0, 1.0 - r_chip / inr) * inr
        off = rng.uniform(0.0, max_off)
        aoff = rng.uniform(0.0, 2.0 * math.pi)
        ccx = (cx + off * math.cos(aoff)) % 1.0
        ccy = (cy + off * math.sin(aoff)) % 1.0
        # keep the chip on its own stone
        px = int(ccx * w) % w
        py = int(ccy * h) % h
        if int(ids[py, px]) != int(cid) or stone_mask[py, px] < 0.55:
            ccx, ccy = cx, cy

        aspect = rng.uniform(1.0, 2.0)
        rx = r_chip * math.sqrt(aspect)
        ry = r_chip / math.sqrt(aspect)
        ang = rng.uniform(0.0, math.pi)
        lobes = int(rng.integers(3, 7))
        lobe_a = rng.uniform(0.03, 0.17)
        lobe_b = rng.uniform(0.02, 0.08)
        ph1 = rng.uniform(0.0, 2.0 * math.pi)
        ph2 = rng.uniform(0.0, 2.0 * math.pi)
        depth = rng.uniform(0.12, 0.26) + 0.20 * (r_chip / inr)

        dx = wrap(X0 - ccx)
        dy = wrap(Y0 - ccy)
        ca2, sa2 = math.cos(ang), math.sin(ang)
        uu = dx * ca2 + dy * sa2
        vv = -dx * sa2 + dy * ca2
        qc = np.sqrt((uu / rx) ** 2 + (vv / ry) ** 2)
        th = np.arctan2(vv, uu)
        qc *= (1.0 + lobe_a * np.sin(lobes * th + ph1)
               + lobe_b * np.sin((lobes + 2) * th + ph2))
        qc *= (1.0 + 0.06 * n_pit)
        hole = 1.0 - smoothstep(0.72, 1.15, qc)
        hole *= stone_mask * (ids == cid)
        core = 1.0 - smoothstep(0.10, 0.60, qc)
        chip_mask = np.maximum(chip_mask, hole)
        chip_depth = np.maximum(chip_depth, hole * depth)
        chip_core = np.maximum(chip_core, core * stone_mask * (ids == cid))

    # ---- height -----------------------------------------------------------
    # dome scales the stone bulge; at dome=0 the top is flat and only a
    # narrow chamfer at the joint remains.
    dome_shape = dome * prof_shape + 0.05 * (1.0 - dome) * prof
    h_cell = (big_base + big_amp * dome_shape
              + 0.010 * n_med_s + 0.0022 * n_fine_s
              + 0.006 * stri + 0.010 * cell_hj[ids])
    mortar_h = big_base + (0.035 if joint_on else -0.002)
    # The same wide, per-stone edge band as the albedo.  The stone body is
    # untouched further inside, but the last few percent of the radius are
    # a smooth bevel instead of a narrow, deep joint trench.
    height = mortar_h + (h_cell - mortar_h) * edge_ramp
    if not joint_on:
        height -= 0.012 * seam
    # keep chips readable, but do not turn their centres into black pits
    height -= 0.65 * chip_depth
    height -= 0.200 * crack
    height = gauss_blur(height, max(1.2, 0.0018 * size))
    # extra smoothing only inside the wide joint band, not over the whole
    # stone body
    edge_w = smoothstep(0.0, 0.35, 1.0 - edge_ramp)
    h_soft = gauss_blur(height, max(2.5, 0.0040 * size))
    height = height * (1.0 - 0.45 * edge_w) + h_soft * (0.45 * edge_w)
    lo, hi = np.percentile(height, [0.2, 99.8])
    height = np.clip((height - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    # a shallow joint floor: the seam remains readable, but there are no
    # black gashes or deep pits at the junction of several stones
    height = 0.32 + 0.68 * height
    height = np.clip(height, 0.0, 1.0).astype(np.float32)

    # ---- albedo -----------------------------------------------------------
    cell_col_px = cell_col[ids]
    cell_col_px *= (0.98 + 0.06 * n_low_s + 0.030 * n_med_s
                    + 0.020 * n_fine_s)[..., None]
    cell_col_px *= (1.0 + 0.028 * stri)[..., None]
    shade_amt = np.where(ids >= n_big, 0.09, 0.05)
    cell_col_px *= (1.0 - shade_amt[..., None] * dome
                    * (1.0 - prof_shape[..., None]))
    cell_col_px *= (1.0 + 0.030 * n_tint)[..., None]

    # joint = 0 gets a dark painted mortar seam instead of stones
    mortar_col = (mortar if joint_on else
                  np.array([0.095, 0.070, 0.042], np.float32))
    gap_col = np.clip(mortar_col * 2.20 + 0.04, 0.0, 1.0)
    gap_wide = gauss_blur(gap, max(1.0, 0.010 * size))
    wide_fill = smoothstep(0.30, 0.60, gap_wide)   # для секции normal ниже
    wide_line = smoothstep(0.68, 0.90, gap_wide)
    fill_col = gap_col[None, None, :]

    # мягкий перетон между соседними камнями (блюр поканальный)
    _cs = max(1.5, 0.004 * size)
    cell_col_px = np.stack(
        [gauss_blur(cell_col_px[..., c], _cs)
         for c in range(cell_col_px.shape[-1])], axis=-1).astype(np.float32)

    # перо узкое и одинаковое везде: тёмная зона = joint_half + w_edge
    # (~0.011, то есть ~22 px на 2048 от кромки камня). При 0.020 тёмная
    # полоса съедала средние камни целиком — отсюда тёмные озёра в центре.
    # нормировка градиента на РАДИУС КАЖДОГО камня (как в v7):
    # t = 0 на кромке, 1 в центре клетки. Без масок резкой границы
    # взять неоткуда, а мелкие камни автоматически получают светлое
    # ядро, потому что градиент успевает дойти до 1 в их центре
    s = bedge - joint_half
    # Use the local-minimum radius instead of this cell's radius: at a
    # large/medium stone boundary a per-cell radius changes abruptly and
    # makes the seam look like it has a step.  local_inr_norm is continuous
    # across that boundary.
    inr_cell = local_inr_norm
    t = np.clip(s / np.maximum(inr_cell - joint_half, 1e-4), 0.0, 1.0)
    # тёмная зона занимает только edge_frac радиуса камня у кромки,
    # дальше камень чистый: шов тонкий и при этом пропорционален камню
    edge_frac = float(np.clip(seam_halo * 12.0, 0.10, 0.30))
    ramp = smoothstep(0.0, edge_frac, t)
    ramp = gauss_blur(ramp, max(0.5, 0.0020 * size))



    if joint_on:
        a = float(np.clip(seam_halo_amt, 0.0, 1.0))
        light_col = 0.55 * mortar_col + 0.45 * pal_base
        grad_col = (light_col[None, None, :] * (1.0 - a)
                    + mortar_col[None, None, :] * a)
        r3 = ramp[..., None]
        col = cell_col_px * r3 + grad_col * (1.0 - r3)

        # тонкое тёмное ядро ровно на кромке — акцент, не зона
        w_k = max(1.0, 0.0022 * size) / size
        joint_line = np.exp(-((s / w_k) ** 2))
        _nls = np.asarray(n_low_s, np.float32)
        if _nls.ndim == 3:
            _nls = _nls[..., 0]
        joint_strength = 0.40 + 0.20 * smoothstep(-0.9, 0.9, _nls)
        joint_mix = (0.30 * joint_strength * joint_line)[..., None]
        col = col * (1.0 - joint_mix) + mortar_col[None, None, :] * joint_mix

        vis = ramp
        col *= (1.0 - 0.07 * rim[..., None] * vis[..., None])
        col = (col * (1.0 - 0.06 * (1.0 - vis)[..., None])
               + dirt * (0.06 * (1.0 - vis))[..., None])
    else:
        col = (cell_col_px * stone_col_mask[..., None]
               + fill_col * gap_col_mask[..., None])
        col *= (1.0 - 0.07 * rim[..., None] * stone_col_mask[..., None])
        col = (col * (1.0 - 0.06 * gap_col_mask[..., None])
               + dirt * (0.06 * gap_col_mask)[..., None])



    col *= (1.0 - 0.07 * smoothstep(0.9, 1.15, n_pit)[..., None])
    # трещины остаются в height/normal, в альбедо почти не видны
    col *= (1.0 - 0.035 * crack[..., None])

    # OTWNDATA speckles: sparse salmon and yellow mineral flecks
    fleck = smoothstep(1.75, 2.25, n_fine2)
    salmon = np.array([0.96, 0.74, 0.52], np.float32)
    yellow = np.array([0.88, 0.74, 0.12], np.float32)
    pick = smoothstep(0.0, 1.0, 0.5 + 0.5 * n_tint)[..., None]
    fleck_col = salmon[None, None, :] * pick + yellow[None, None, :] * (1.0 - pick)
    stone_vis = ramp if joint_on else np.clip(big_mask + med_mask, 0.0, 1.0)
    col = (col * (1.0 - 0.42 * fleck[..., None] * stone_vis[..., None])
           + fleck_col * (0.42 * fleck * stone_vis)[..., None])

    # a few larger mineral spots, like the 1-2 px dots of the original tile
    spot = smoothstep(2.05, 2.45, n_spot) * stone_vis
    spot_col = np.array([0.86, 0.58, 0.38], np.float32)
    col = (col * (1.0 - 0.28 * spot[..., None])
           + spot_col * (0.28 * spot)[..., None])

    # tiny dark mineral grains, the brown counterpart of the flecks
    grain = smoothstep(2.15, 2.60, n_dark) * stone_vis
    grain_col = np.array([0.36, 0.25, 0.12], np.float32)
    col = (col * (1.0 - 0.30 * grain[..., None])
           + grain_col * (0.30 * grain)[..., None])

    # moss in the aged variant, mostly in joints
    if moss_amt > 0:
        moss_mask = smoothstep(0.55, 0.95, n_pit) * (1.0 - (ramp if joint_on
                                                        else stone_col_mask))
        moss_mask *= smoothstep(-0.2, 0.6, n_low)
        moss = np.array([0.205, 0.250, 0.110], np.float32)
        col = (col * (1.0 - moss_amt * moss_mask[..., None])
               + moss * (moss_amt * moss_mask)[..., None])

    # chips are mainly a height/normal feature; keep albedo almost quiet
    chip_col = np.array([0.290, 0.220, 0.125], np.float32)
    col = (col * (1.0 - 0.22 * chip_mask[..., None])
           + chip_col * (0.22 * chip_mask)[..., None])
    col = (col * (1.0 - 0.07 * chip_core[..., None])
           + chip_col * (0.07 * chip_core)[..., None])
    lip = chip_mask * (1.0 - chip_core)
    col = np.clip(col * (1.0 + 0.055 * lip[..., None]), 0.0, 1.0)

    # joint = 0: painted dark mortar line over the touching stone edges
    if not joint_on:
        seam_col = np.array([0.070, 0.050, 0.030], np.float32)
        col = (col * (1.0 - 0.78 * seam[..., None])
               + seam_col * (0.78 * seam)[..., None])


    # ---- AO ---------------------------------------------------------------
    hs = gauss_blur(height, 1.4)
    cav = np.maximum(0.0, gauss_blur(hs, 4.0) - hs)
    cav2 = np.maximum(0.0, gauss_blur(hs, 18.0) - hs)
    ao = 1.0 - np.clip(cav * 4.2, 0.0, 0.55) - np.clip(cav2 * 1.7, 0.0, 0.35)
    ao *= (1.0 - 0.08 * gap_col_mask)
    ao *= (1.0 - 0.08 * rim * stone_col_mask)
    ao *= (1.0 - 0.35 * crack)
    ao *= (1.0 - 0.30 * seam)
    ao = np.clip(ao, 0.20, 1.0).astype(np.float32)
    ao = gauss_blur(ao, 0.8)

    # subtle baked AO in albedo
    col *= (0.96 + 0.04 * ao[..., None])
    col = np.clip(col, 0.0, 1.0)

    # ---- roughness --------------------------------------------------------
    rough_cell = (big_rough + 0.035 * n_med_s
                 + 0.020 * cell_rj[ids] + 0.030 * stri
                 + 0.030 * (ids >= n_big))
    rough = (rough_cell * stone_col_mask + 0.80 * gap_col_mask)
    rough += 0.08 * chip_mask
    rough += 0.06 * crack
    rough = rough * (1.0 - seam) + 0.82 * seam
    rough = np.clip(rough, 0.30, 0.95).astype(np.float32)

    # ---- normal -----------------------------------------------------------
    # The main stone body gets fine, rough tooling.  The detail is faded out
    # over the same wide edge band as the albedo, so the rounded silhouette
    # stays smooth and the joints do not turn into hard creases.
    micro = (0.0100 * n_fine_s + 0.0060 * n_fine2
             + 0.0040 * n_micro + 0.0030 * n_med_s)
    detail_mask = edge_ramp * (1.0 - 0.80 * wide_fill)
    hn = gauss_blur(height, 1.1) + normal_detail * micro * detail_mask
    gx = (np.roll(hn, -1, axis=1) - np.roll(hn, 1, axis=1)) * 0.5
    gy = (np.roll(hn, -1, axis=0) - np.roll(hn, 1, axis=0)) * 0.5
    S_big = 68.0
    S_med = 58.0
    S_gap = 12.0
    S_stone = np.where(ids < n_big, S_big, S_med)
    S = S_gap + (S_stone - S_gap) * stone_col_mask
    S *= (size / 2048.0)
    nx = -gx * S
    ny = -gy * S
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= ln
    ny /= ln
    nz /= ln

    # ---- save maps --------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    save_png(os.path.join(outdir, "albedo_%s.png" % suffix), col)
    save_png(os.path.join(outdir, "height_%s.png" % suffix), height)
    save_png(os.path.join(outdir, "roughness_%s.png" % suffix), rough)
    save_png(os.path.join(outdir, "ao_%s.png" % suffix), ao)
    save_png(os.path.join(outdir, "normal_gl_%s.png" % suffix),
             np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5,
                       nz * 0.5 + 0.5], -1))
    save_png(os.path.join(outdir, "normal_dx_%s.png" % suffix),
             np.stack([nx * 0.5 + 0.5, 0.5 - ny * 0.5,
                       nz * 0.5 + 0.5], -1))

    # ---- statistics -------------------------------------------------------
    vis_area = np.bincount(ids[big_mask > 0.5].ravel(),
                           minlength=n_total).astype(np.float64)
    vis_area /= float(h * w)
    eq = np.sqrt(cell_area[:n_big] / math.pi)
    size_ratio = float(eq.max() / max(eq.min(), 1e-9))
    big_area = float(big_mask.mean())
    med_area = float(med_mask.mean())
    seal = (float(np.mean(np.abs(col[:, 0] - col[:, -1]))),
            float(np.mean(np.abs(col[0, :] - col[-1, :]))))
    if verbose:
        print("    %s: %dx%d, %d big + %d junction stones, "
              "size max/min %.2fx, %d chips"
              % (variant, size, size, n_big, n_med, size_ratio, n_chips))
        print("    seam L-R %.4f  T-B %.4f  joint %.1f px (%s)  dome %.2f  "
              "big area %.1f%%  medium area %.1f%%"
              % (seal[0], seal[1], joint_px_eff,
                 "stones" if joint_on else "dark mortar", dome,
                 100.0 * big_area, 100.0 * med_area))
    stats = dict(variant=variant, size=size, stones=n_big, chips=n_chips,
                 cracks=int(crack_count), medium=n_med, spread=float(spread),
                 size_ratio=size_ratio,
                 joint_px=req_joint_px, joint_eff_px=joint_px_eff,
                 dome=dome, joint_on=int(joint_on),
                 seam_lr=seal[0], seam_tb=seal[1],
                 big_area=big_area, med_area=med_area)
    return col, height, nx, ny, nz, ao, stats


# ----------------------------------------------------------------------------
# previews
# ----------------------------------------------------------------------------
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def load_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0


def lit_render(albedo, normal, ao):
    """Gamma-correct directional light preview (warm key + sky ambient)."""
    nx = normal[..., 0] * 2.0 - 1.0
    ny = normal[..., 1] * 2.0 - 1.0
    nz = normal[..., 2] * 2.0 - 1.0
    L = np.array([-0.55, -0.68, 0.48], np.float32)
    L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)

    H = L + np.array([0.0, 0.0, 1.0], np.float32)
    H /= np.linalg.norm(H)
    spec = np.clip(nx * H[0] + ny * H[1] + nz * H[2], 0.0, 1.0)
    spec = spec ** 20.0

    lin = srgb_to_lin(albedo.astype(np.float32))
    sky = np.array([0.20, 0.21, 0.24], np.float32)
    out = (lin * (sky + 0.92 * diff[..., None])
           + 0.10 * spec[..., None])
    out *= (0.50 + 0.50 * ao[..., None])
    return np.clip(lin_to_srgb(out), 0.0, 1.0)


def make_previews(root, variants):
    prev = os.path.join(root, "previews")
    os.makedirs(prev, exist_ok=True)
    thumbs = []

    for v in variants:
        d = os.path.join(root, v)
        alb = load_rgb(os.path.join(d, "albedo_2k.png"))
        nrm = load_rgb(os.path.join(d, "normal_gl_2k.png"))
        ao = load_gray(os.path.join(d, "ao_2k.png"))

        small = np.asarray(Image.fromarray(
            (alb * 255 + 0.5).clip(0, 255).astype(np.uint8))
            .resize((1024, 1024), Image.LANCZOS)).astype(np.float32) / 255.0
        save_png(os.path.join(prev, "%s_tiled_2x2.png" % v),
                 np.tile(small, (2, 2, 1)))

        full = np.tile(alb, (2, 2, 1))
        s = full.shape[0] // 2
        save_png(os.path.join(prev, "%s_seam_1k.png" % v),
                 full[s - 512:s + 512, s - 512:s + 512])

        lit = lit_render(alb, nrm, ao)
        save_png(os.path.join(prev, "%s_lit.png" % v), lit)

        th = np.asarray(Image.fromarray(
            (lit * 255 + 0.5).clip(0, 255).astype(np.uint8))
            .resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255.0
        thumbs.append((v, th))

    ref_thumb = None
    ref_path = os.path.join(os.path.dirname(root), "png_vga", "tiles",
                            "OTWNDATA", "t000.png")
    if os.path.exists(ref_path):
        rim = Image.open(ref_path).convert("RGBA")
        bg = Image.new("RGBA", rim.size, (32, 28, 22, 255))
        bg.alpha_composite(rim)
        rim = bg.convert("RGB")
        k = min(512.0 / rim.width, 512.0 / rim.height)
        rim = rim.resize((int(rim.width * k), int(rim.height * k)),
                         Image.LANCZOS)
        canvas = Image.new("RGB", (512, 512), (26, 24, 20))
        canvas.paste(rim, ((512 - rim.width) // 2,
                           (512 - rim.height) // 2))
        canvas.save(os.path.join(prev, "reference_OTWNDATA_t000.png"))
        ref_thumb = np.asarray(canvas).astype(np.float32) / 255.0

    cols, labels = [], []
    if ref_thumb is not None:
        cols.append(ref_thumb)
        labels.append("reference t000")
    for v, th in thumbs:
        cols.append(th)
        labels.append(v)

    n = len(cols)
    gap = 18
    label_h = 40
    title_h = 60
    W = n * 512 + (n + 1) * gap
    H = title_h + 512 + label_h + gap
    sheet = Image.new("RGB", (W, H), (22, 22, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 30)
        font_label = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
    except Exception:
        font_title = font_label = ImageFont.load_default()
    draw.text((gap, 14), "OTWNDATA  --  seamless PBR stone wall (2K)",
              fill=(235, 225, 200), font=font_title)
    for i, (c, lab) in enumerate(zip(cols, labels)):
        x = gap + i * (512 + gap)
        im = Image.fromarray((c * 255 + 0.5).clip(0, 255).astype(np.uint8))
        sheet.paste(im, (x, title_h))
        draw.text((x + 8, title_h + 512 + 8), lab,
                  fill=(220, 210, 190), font=font_label)
    sheet.save(os.path.join(prev, "contact_sheet.png"), optimize=True)


# ----------------------------------------------------------------------------
# docs / manifest
# ----------------------------------------------------------------------------
def write_readme(root, stats):
    lines = []
    a = lines.append
    a(u"OTWNDATA — бесшовные PBR-текстуры каменной стены")
    a(u"=================================================")
    a(u"")
    a(u"Стиль: тайлы стен OTWNDATA.VGA (Wizardry VII, 1992), тайл t000.")
    a(u"Текстуры синтезированы процедурно и бесшовно; это не апскейл")
    a(u"8-битных тайлов, а новая кладка в той же палитре и манере.")
    a(u"")
    a(u"Кладка")
    a(u"------")
    a(u"- 49 крупных камней и 16 средних камней в естественных развилках")
    a(u"  швов. Все камни — ячейки одной периодической степенной диаграммы,")
    a(u"  поэтому средние камни не накладываются на крупные, а занимают")
    a(u"  свободные полости между ними.")
    a(u"- Крупные камни — сложные многоугольники без ограничения числа")
    a(u"  сторон, плотно упакованные. Углы скруглены периодическим")
    a(u"  раскрытием, поэтому силуэты выглядят обкатанными, а не")
    a(u"  многогранными.")
    a(u"- Разброс линейного размера крупных камней — около 4.0x (max/min).")
    a(u"  Фактически по вариантам:")
    for s in stats:
        if s.get("stones"):
            a(u"    %-9s  max/min = %.2f  (%.0f %%)"
              % (s["variant"], s["size_ratio"],
                 (s["size_ratio"] - 1.0) * 100.0))
    a(u"- Швы узкие: 10 px при 2048 для gold/aged и 9 px для cyclopean.")
    a(u"  В карте высот край камня имеет широкую мягкую фаску, а стык —")
    a(u"  неглубокую полосу раствора без узких ям и чёрных провалов.")
    a(u"  Тело камня остаётся куполообразным.")
    a(u"- Переход цвета от шва к камню мягкий: тёмная линия шва")
    a(u"  переходит в основной цвет через плавную полосу раствора,")
    a(u"  без резкой границы.")
    a(u"- В normal map основная площадь камня имеет мелкую грубую")
    a(u"  обработку. У швов детализация плавно сходит на нет, поэтому")
    a(u"  округлый силуэт и широкая фаска читаются мягко, без ступеней.")
    a(u"- height, roughness и AO используют ту же широкую зону перехода")
    a(u"  у края камня, что и albedo; это убирает рассинхрон карт.")
    a(u"- Ширина мягкой зоны задана в долях радиуса камня, поэтому")
    a(u"  средние и мелкие камни получают такой же плавный градиент, как")
    a(u"  крупные. На стыке крупного и среднего камня ширина считается")
    a(u"  по локальному минимуму радиуса, поэтому она не прыгает на")
    a(u"  границе. Контраст у мелких ниже, чтобы тёмный раствор не")
    a(u"  превращал небольшой камень в одно тёмное пятно.")
    a(u"- Сколы/выщербины: 15 на текстуру, от мелких точек до крупных")
    a(u"  выломов. В альбедо они почти не читаются, а в height/normal/AO")
    a(u"  дают заметный рельеф.")
    a(u"- Трещины: процедурная генерация реализована, но по умолчанию")
    a(u"  отключена. Включить можно параметром --cracks N.")
    a(u"")
    a(u"Палитра (sRGB, взята из OTWNDATA t000)")
    a(u"---------------------------------------")
    a(u"  #d2b662, #d2aa50, #c29948, #b28944, #9a7433 — золотистый песчаник")
    a(u"  #d4b018 — жёлтые минеральные вкрапления")
    a(u"  #eeb17d — лососёвые/розовые вкрапления")
    a(u"  #6d501e / тёмно-коричневый — раствор и грязь в швах")
    a(u"")
    a(u"Варианты")
    a(u"--------")
    a(u"gold/       — золотисто-песчаная кладка, ближе всего к оригиналу")
    a(u"aged/       — выветренная, темнее, с мхом в швах")
    a(u"cyclopean/  — более высокие купола, более контрастная кладка")
    a(u"")
    a(u"Файлы (каждый вариант, 2048x2048)")
    a(u"--------------------------------")
    a(u"albedo_2k.png       — базовый цвет (sRGB)")
    a(u"height_2k.png       — карта высот (линейная, grayscale)")
    a(u"normal_gl_2k.png    — normal map, OpenGL (+Y вверх)")
    a(u"normal_dx_2k.png    — normal map, DirectX (-Y)")
    a(u"roughness_2k.png    — шероховатость (линейная, grayscale)")
    a(u"ao_2k.png           — ambient occlusion (линейная, grayscale)")
    a(u"")
    a(u"Свойства")
    a(u"--------")
    a(u"- Все карты бесшовные по X и Y.")
    a(u"- albedo — sRGB; height / normal / roughness / ao — линейные данные.")
    a(u"- normal_gl: зелёный канал +Y; normal_dx: зелёный канал -Y.")
    a(u"- Камень немагнитный: metallic = 0.")
    a(u"- Масштаб: 7 крупных камней по ширине текстуры; средние камни")
    a(u"  занимают развилки швов и заметно меньше крупных.")
    a(u"")
    a(u"Three.js (MeshStandardMaterial)")
    a(u"-------------------------------")
    a(u"  const tex = new THREE.TextureLoader();")
    a(u"  mat.map = tex.load('gold/albedo_2k.png');")
    a(u"  mat.normalMap = tex.load('gold/normal_gl_2k.png');")
    a(u"  mat.roughnessMap = tex.load('gold/roughness_2k.png');")
    a(u"  mat.aoMap = tex.load('gold/ao_2k.png');")
    a(u"  mat.metalness = 0.0; mat.roughness = 1.0;")
    a(u"  mat.map.colorSpace = THREE.SRGBColorSpace;")
    a(u"  // aoMap требует второго UV-набора (uv2) в геометрии.")
    a(u"")
    a(u"Превью")
    a(u"------")
    a(u"previews/<variant>_tiled_2x2.png — тайлинг 2x2")
    a(u"previews/<variant>_lit.png       — рендер с направленным светом")
    a(u"previews/<variant>_seam_1k.png   — кроп 1:1 на стыке четырёх тайлов")
    a(u"previews/contact_sheet.png       — все варианты и оригинал")
    a(u"")
    a(u"Генерация")
    a(u"---------")
    a(u"  python make_otwn_textures.py")
    a(u"  python make_otwn_textures.py --size 512 --variants gold  # быстрый тест")
    a(u"")
    a(u"Параметры")
    a(u"----------")
    a(u"  --joint PX   полная ширина шва в пикселях при 2048 (по умолчанию")
    a(u"               10 для gold/aged и 9 для cyclopean). 0 — не")
    a(u"               генерировать средние камни, оставить тонкий тёмный")
    a(u"               шов-раствор.")
    a(u"  --dome K     амплитуда купола крупных камней, 1.0 — обычная,")
    a(u"               0.0 — плоские верхние грани (остаётся фаска у шва).")
    a(u"  --chips N    число сколов (по умолчанию 15, максимум stones/3).")
    a(u"  --cracks N   число трещин (по умолчанию 0 — отключено).")
    a(u"  --spread K   целевой разброс линейного размера крупных камней")
    a(u"               (по умолчанию 4.0).")
    a(u"  --medium N   максимум средних камней в развилках (по умолчанию 16).")
    a(u"  --medium-area F  доля площади под средние камни (по умолчанию 0.09).")
    a(u"  --seam-halo F    ширина общей мягкой зоны у шва (доля размера,")
    a(u"                   по умолчанию 0.005).")
    a(u"  --seam-halo-amt F  сила градиента цвета у шва (0..1, по умолчанию 0.30).")
    a(u"  --normal-detail K  сила мелкой шероховатости normal map (по умолчанию 1.0).")
    a(u"")
    a(u"Примеры:")
    a(u"  python make_otwn_textures.py --joint 0 --dome 0")
    a(u"  python make_otwn_textures.py --cracks 9 --size 512 --variants gold")
    a(u"  python make_otwn_textures.py --spread 3.0 --medium 12")
    a(u"")
    if stats:
        s0 = stats[0]
        joints = sorted({float(s["joint_px"]) for s in stats})
        domes = sorted({float(s["dome"]) for s in stats})
        joint_txt = ("%.0f–%.0f" % (joints[0], joints[-1])
                     if len(joints) > 1 else "%.0f" % joints[0])
        dome_txt = ("%.2f–%.2f" % (domes[0], domes[-1])
                    if len(domes) > 1 else "%.2f" % domes[0])
        a(u"Текущий пак: %d крупных + %d средних камней, %d сколов, "
          u"%d трещин, разброс %.1fx, шов %s px, купол %s."
          % (s0["stones"], s0.get("medium", 0), s0["chips"],
             s0.get("cracks", 0), float(s0["spread"]), joint_txt, dome_txt))
    else:
        a(u"Текущий пак: 49 крупных + 16 средних камней, 15 сколов, "
          u"0 трещин, разброс 4.0x, шов 9–10 px, купол 1.00.")
    a(u"Параметры каждого варианта записаны в stats.csv.")
    a(u"")
    a(u"Исходные экспорты: png/ (.PIC -> PNG), png_vga/ (.VGA -> PNG).")
    a(u"Пак текстур: textures_otwn/.")
    # The body above is plain text; convert its underlined headings so the
    # generator emits a proper Markdown README.md.
    import re as _re
    md = []
    _i = 0
    while _i < len(lines):
        _line = lines[_i]
        if (_i + 1 < len(lines) and _line.strip() and
                _re.fullmatch(r"[=-]{3,}", lines[_i + 1].strip())):
            _level = "#" if set(lines[_i + 1].strip()) == {"="} else "##"
            md.append(_level + " " + _line.strip())
            _i += 2
            continue
        md.append(_line)
        _i += 1
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write(u"\n".join(md) + u"\n")


def write_manifest(root):
    maps = ("albedo", "height", "normal_gl", "normal_dx",
            "roughness", "ao")
    valid = {"%s/%s_2k.png" % (v, m)
             for v in ("gold", "aged", "cyclopean") for m in maps}
    rows = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(".png"):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root).replace("\\", "/")
            if rel not in valid and not rel.startswith("previews/"):
                continue
            im = Image.open(p)
            rows.append([rel, im.width, im.height, im.mode,
                         os.path.getsize(p)])
    with open(os.path.join(root, "manifest.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "width", "height", "mode", "bytes"])
        wr.writerows(rows)


def write_stats(root, stats):
    keys = ["variant", "size", "stones", "chips", "cracks", "medium",
            "spread", "size_ratio", "joint_px", "joint_eff_px", "dome",
            "joint_on", "seam_lr", "seam_tb", "big_area", "med_area"]
    with open(os.path.join(root, "stats.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(keys)
        for s in stats:
            if s.get("stones"):
                wr.writerow([("%.6f" % s[k]) if isinstance(s[k], float)
                             else s[k] for k in keys])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--variants", default="gold,aged,cyclopean")
    ap.add_argument("--joint", type=float, default=None,
                    help="full joint width in pixels at 2048; "
                         "0 = no fillers, thin dark mortar seam")
    ap.add_argument("--dome", type=float, default=1.0,
                    help="large-stone dome amount; 0 = flat tops")
    ap.add_argument("--chips", type=int, default=15,
                    help="number of chips/pockmarks (capped at stones/3)")
    ap.add_argument("--cracks", type=int, default=0,
                    help="number of thin fractures (0 = disabled)")
    ap.add_argument("--spread", type=float, default=4.0,
                    help="big-stone linear size spread (max/min)")
    ap.add_argument("--medium", type=int, default=16,
                    help="max number of medium stones in junction voids")
    ap.add_argument("--medium-area", type=float, default=0.09,
                    help="fraction of surface taken by medium stones")
    ap.add_argument("--seam-halo", type=float, default=0.0050,
                    help="width of the soft albedo halo around the seam "
                         "(fraction of texture size)")
    ap.add_argument("--seam-halo-amt", type=float, default=0.30,
                    help="opacity of the soft albedo seam halo")
    ap.add_argument("--normal-detail", type=float, default=1.0,
                    help="strength of the fine rough-stone normal detail")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, "textures_otwn")
    os.makedirs(out, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    jobs = {"gold": 7, "aged": 19, "cyclopean": 31}
    stats = []
    for name in variants:
        if name not in jobs:
            print("unknown variant:", name)
            continue
        print("generating %s (%d) ..." % (name, args.size))
        _, _, _, _, _, _, st = generate(
            size=args.size, seed=jobs[name], variant=name,
            suffix="2k" if args.size == 2048 else "test",
            outdir=os.path.join(out, name),
            joint_px=args.joint, dome=args.dome, chip_count=args.chips,
            crack_count=args.cracks,
            spread=args.spread, medium_count=args.medium,
            medium_area=args.medium_area, seam_halo=args.seam_halo,
            seam_halo_amt=args.seam_halo_amt,
            normal_detail=args.normal_detail)
        stats.append(st)

    # stale test/4K maps from older runs would contradict the 2K pack
    if args.size == 2048:
        for name in variants:
            d = os.path.join(out, name)
            for f in os.listdir(d) if os.path.isdir(d) else []:
                if f.endswith("_4k.png") or f.endswith("_test.png"):
                    os.remove(os.path.join(d, f))

    if args.size == 2048 and set(variants) >= {"gold", "aged", "cyclopean"}:
        print("building previews ...")
        make_previews(out, ["gold", "aged", "cyclopean"])
        print("writing README / manifest / stats ...")
        # preserve stats of variants generated in earlier runs
        existing = {}
        for s in stats:
            existing[s["variant"]] = s
        stats = [existing.get(v) for v in ["gold", "aged", "cyclopean"]]
        stats = [s for s in stats if s]
        write_readme(out, stats)
        write_manifest(out)
        write_stats(out, stats)
    print("done ->", out)


if __name__ == "__main__":
    main()
