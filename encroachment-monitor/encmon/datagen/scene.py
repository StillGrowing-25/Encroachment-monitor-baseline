"""Synthetic evaluation scene (paper Sec. VI-C and Fig. 3 right).

A 512 x 512 px Sentinel-2-like scene (5.12 km at 10 m) observed at t1 and
t2.  The scene contains a protected government zone with an irregular
boundary, pre-existing construction, water and vegetation; between t1 and
t2 new encroachments appear inside the protected zone (small buildings,
some straddling the boundary), legal new construction appears outside, and
seasonal bare-soil flushes occur everywhere.

Ground truth per pixel: land-cover label, and an encroachment-change mask.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    SCENE_ENC_SIZE_RANGE,
    SCENE_N_ENCROACHMENTS,
    SCENE_N_FLASH,
    SCENE_N_LEGAL_NEW,
    SCENE_SIZE,
    SCENE_STRADDLE_FRACTION,
)
from .patches import SURF_SPECTRA

SURF_TO_ID = {"open": 0, "vegetation": 1, "water": 2, "built": 3}


def _render(label: np.ndarray, rng: np.random.Generator, gain: float | None = None) -> np.ndarray:
    """Render (H, W, 5) reflectance from a surface-id map with pixel noise."""
    g = rng.normal(1.0, 0.05) if gain is None else gain
    out = np.empty(label.shape + (5,), dtype=float)
    for surf, sid in SURF_TO_ID.items():
        spec = SURF_SPECTRA[surf]
        mu = np.asarray(spec["band_mean"])
        sigma = spec["band_std"]
        mask = label == sid
        out[mask] = np.maximum(mu * g + rng.normal(0.0, sigma, size=(int(mask.sum()), 5)), 0.001)
    return out


def _protected_zone_mask(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Irregular protected-zone polygon as a raster mask + its boundary ring.

    Built as the union of a large soft ellipse with low-frequency harmonic
    boundary perturbation, giving an organic, map-like government boundary.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.42, h * 0.55
    ang = np.arctan2(yy - cy, xx - cx)
    r_base = min(h, w) * 0.36
    # low-frequency radius perturbation -> irregular but compact boundary
    pert = (
        0.16 * np.sin(3 * ang + 0.7)
        + 0.10 * np.sin(5 * ang + 2.1)
        + 0.06 * np.sin(8 * ang + 4.2)
    )
    r = r_base * (1 + pert)
    dist = np.hypot(xx - cx, yy - cy)
    return dist <= r


def _stamp(label: np.ndarray, cy: int, cx: int, half: int, sid: int, rng) -> None:
    """Stamp a jittered square blob of surface id into the label map."""
    h, w = label.shape
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    # per-pixel jitter on the rim so polygons are not perfectly crisp
    sub = label[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (np.maximum(np.abs(yy - cy), np.abs(xx - cx)) <= half)
    sub[inside] = sid


def generate_scene(seed: int):
    """Generate the bi-temporal scene and all ground-truth masks.

    Returns dict with t1, t2 (H, W, 5), protected mask, boundary ring,
    encroachment pixel mask, legal-new pixel mask, and land-cover labels.
    """
    rng = np.random.default_rng(seed + 23)
    n = SCENE_SIZE
    prot = _protected_zone_mask((n, n), rng)

    # --- base landscape t1 -------------------------------------------------
    label1 = np.full((n, n), SURF_TO_ID["open"], dtype=np.int8)
    # vegetation regions via smooth noise thresholding
    yy, xx = np.mgrid[0:n, 0:n]
    veg_noise = (
        np.sin(xx / 67.0) * np.cos(yy / 49.0)
        + 0.7 * np.sin(xx / 23.0 + 1.2) * np.sin(yy / 31.0 + 0.4)
        + rng.normal(0, 0.25, (n, n))
    )
    label1[veg_noise > 0.15] = SURF_TO_ID["vegetation"]
    # a river / lake along the south-east
    river = (np.abs(yy - (n * 0.85 + 30 * np.sin(xx / 40.0))) < 14)
    label1[river] = SURF_TO_ID["water"]
    # pre-existing town (built) in the north, outside the protected zone
    town_x, town_y = int(n * 0.72), int(n * 0.18)
    town_noise = rng.random((n, n))
    town = (np.hypot((xx - town_x) / (n * 0.16), (yy - town_y) / (n * 0.10)) < 1.0)
    label1[town & (town_noise < 0.75)] = SURF_TO_ID["built"]
    # some pre-existing government buildings inside the protected zone
    for _ in range(14):
        cy, cx = int(rng.uniform(0.3, 0.8) * n), int(rng.uniform(0.25, 0.6) * n)
        while not prot[cy, cx]:
            cy, cx = int(rng.uniform(0.2, 0.8) * n), int(rng.uniform(0.2, 0.6) * n)
        _stamp(label1, cy, cx, int(rng.integers(3, 7)), SURF_TO_ID["built"], rng)

    label2 = label1.copy()

    # --- t2 changes ---------------------------------------------------------
    gt_enc = np.zeros((n, n), dtype=bool)      # true encroachment pixels
    gt_legal_new = np.zeros((n, n), dtype=bool)
    prot_ys, prot_xs = np.nonzero(prot)
    border = _boundary_ring(prot)

    # encroachments: small structures inside the protected zone
    placed = 0
    guard = 0
    while placed < SCENE_N_ENCROACHMENTS and guard < 4000:
        guard += 1
        i = int(rng.integers(0, len(prot_ys)))
        cy, cx = int(prot_ys[i]), int(prot_xs[i])
        half = int(rng.integers(SCENE_ENC_SIZE_RANGE[0], SCENE_ENC_SIZE_RANGE[1] + 1))
        straddle = rng.random() < SCENE_STRADDLE_FRACTION
        if straddle:
            # push centre to sit on the boundary ring
            bi = int(rng.integers(0, len(border[0])))
            cy, cx = int(border[0][bi]), int(border[1][bi])
        stamp = np.zeros((n, n), dtype=bool)
        _stamp_into(stamp, cy, cx, half, rng)
        if not straddle:
            # keep interior encroachments fully inside the protected zone
            grow = _binary_dilate(stamp, 3)
            if (grow & ~prot).any():
                continue
        before = label2[stamp]
        label2[stamp] = SURF_TO_ID["built"]
        newly = stamp.copy()
        newly[stamp] = before != SURF_TO_ID["built"]     # infill isn't new
        gt_enc |= newly
        placed += 1

    # legal new construction: larger formal layouts outside the protected zone
    placed = 0
    guard = 0
    while placed < SCENE_N_LEGAL_NEW and guard < 2000:
        guard += 1
        cy = int(rng.uniform(0, n))
        cx = int(rng.uniform(0, n))
        half = int(rng.integers(8, 16))
        stamp = np.zeros((n, n), dtype=bool)
        _stamp_into(stamp, cy, cx, half, rng)
        # keep formal layouts clear of the protected boundary so their window
        # halo does not spill into the protected zone (realistic zoning)
        if (_binary_dilate(stamp, 4) & prot).any():
            continue
        label2[stamp] = SURF_TO_ID["built"]
        gt_legal_new[stamp] = True
        placed += 1

    # seasonal bare-soil flushes (change-rule confusers), both zones
    gt_flash = np.zeros((n, n), dtype=bool)
    for _ in range(SCENE_N_FLASH):
        cy = int(rng.uniform(0, n))
        cx = int(rng.uniform(0, n))
        half = int(rng.integers(3, 10))
        stamp = np.zeros((n, n), dtype=bool)
        _stamp_into(stamp, cy, cx, half, rng)
        was_veg = label2[stamp] == SURF_TO_ID["vegetation"]
        label2[stamp] = SURF_TO_ID["open"]
        gt_flash[stamp] = was_veg

    gain1 = float(rng.normal(1.0, 0.02))
    gain2 = float(rng.normal(1.0, 0.02))
    t1 = _render(label1, rng, gain=gain1)
    t2 = _render(label2, rng, gain=gain2)

    # sub-pixel mis-registration of t2 relative to t1 (< 1 px, paper Table III)
    shift_y, shift_x = 0.4, -0.3
    t2s = _fractional_shift(t2, shift_y, shift_x)

    return dict(
        t1=t1,
        t2=t2s,
        t2_ideal=t2,
        label1=label1,
        label2=label2,
        protected=prot,
        boundary=border,
        gt_encroachment=gt_enc & ~gt_flash,
        gt_legal_new=gt_legal_new,
        gt_flash=gt_flash,
        shift=(shift_y, shift_x),
    )


def _stamp_into(target: np.ndarray, cy: int, cx: int, half: int, rng) -> None:
    h, w = target.shape
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.maximum(np.abs(yy - cy), np.abs(xx - cx)) <= half
    target[y0:y1, x0:x1] |= inside


def _binary_dilate(mask: np.ndarray, it: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(it):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        )
    return out


def _boundary_ring(mask: np.ndarray):
    """Pixel coordinates of the protected-zone boundary."""
    er = ~_binary_dilate(~mask, 1)
    edge = mask & ~er
    ys, xs = np.nonzero(edge)
    return ys, xs


def _fractional_shift(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Bilinear fractional shift (simulates residual mis-registration)."""
    h, w = arr.shape[:2]
    y0 = int(np.floor(dy))
    x0 = int(np.floor(dx))
    fy, fx = dy - y0, dx - x0
    out = np.empty_like(arr)
    def sh(a, by, bx):
        out = np.zeros_like(a)
        ys = slice(max(0, by), min(h, h + by))
        xs = slice(max(0, bx), min(w, w + bx))
        ys_src = slice(max(0, -by), min(h, h - by))
        xs_src = slice(max(0, -bx), min(w, w - bx))
        out[ys, xs] = a[ys_src, xs_src]
        return out
    a = sh(arr, y0, x0)
    b = sh(arr, y0 + 1, x0)
    c = sh(arr, y0, x0 + 1)
    d = sh(arr, y0 + 1, x0 + 1)
    return (1 - fy) * (1 - fx) * a + fy * (1 - fx) * b + (1 - fy) * fx * c + fy * fx * d
