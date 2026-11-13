"""Bi-temporal change-detection sites (paper Table II: 6,000 sites).

Each site is a ~100 m patch observed at t1 and t2, placed at a metric (x, y)
inside the AOI.  Site categories (see config.SITE_COUNTS) determine ground
truth; *whether the GIS overlay fires* is decided later by a real spatial
join against the parcel layer -- the category only chooses where the site is
sampled from.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    N_FLASH_IN_PROTECTED,
    N_FLASH_OUTSIDE,
    N_WEAK_SITES,
    PATCH,
    SITE_COUNTS,
)
from ..spectra import FEATURE_NAMES, patch_features
from .patches import _render_pixels

# --- t1 / t2 composition recipes -------------------------------------------
# Each recipe maps surface type -> pixel count for the (PATCH x PATCH) site.

def _mix(core: str, core_n: int, ctx: dict[str, float], ctx_n: int) -> dict[str, int]:
    counts = {core: core_n}
    rem = ctx_n
    keys = list(ctx)
    for i, k in enumerate(keys):
        c = int(round(ctx[k] * ctx_n)) if i < len(keys) - 1 else rem
        c = max(0, min(rem, c))
        counts[k] = counts.get(k, 0) + c
        rem -= c
    return counts


def _render_counts(counts: dict[str, int], rng) -> np.ndarray:
    """Render a (PATCH, PATCH, 5) block given exact surface-pixel counts."""
    total = PATCH * PATCH
    lab = np.empty(total, dtype=object)
    idx = 0
    for surf, c in counts.items():
        lab[idx: idx + c] = surf
        idx += c
    lab[idx:] = "open"
    rng.shuffle(lab)
    return _render_pixels(lab.reshape(PATCH, PATCH), rng)


# ---------------------------------------------------------------------------
# Site state recipes.  Bare-soil states are drawn from the SAME dryness
# gradient the land-cover training data spans (moist soil -> dry crust), so
# every site is in-distribution for the classifier.  Dryness w mixes the
# 90-pixel core as w*crust + (1-w)*soil; NDBI(w) = w*0.159 + (1-w)*0.077.
#   moist t1 states:  w ~ U(0, 0.10)  -> NDBI <= 0.09  (never above tau=0.13)
#   flash t2 state:   w ~ U(0.95, 1)  -> NDBI >= 0.155 (above tau, delta>0.05)
# ---------------------------------------------------------------------------
def _openish(w: float, veg: int = 4) -> dict[str, int]:
    crust = int(round(90.0 * w))
    return dict(open=96 - crust, vegetation=veg, water=0, built=0, crust=crust)


_SOIL_W_MAX = 0.10      # moist soil bound for t1 / stable states
_FLASH_W_MIN = 0.95     # crust bound for flash t2

# New construction states (A and C at t2): identical built-up signature
_NEW_BUILD = dict(open=12, vegetation=10, water=0, built=78, crust=0)
# Weak / low-contrast new construction (the single FN site)
_WEAK_BUILD = dict(open=28, vegetation=14, water=0, built=58, crust=0)
# Pre-existing built (B, D at both dates): firmly above tau so the change
# rule (NDBI_t1 <= tau) never fires on old construction
_OLD_BUILD = dict(open=4, vegetation=8, water=0, built=88, crust=0)
# Flash confusers: moist soil -> bright dry crust between dates
_FLASH_T1_SOIL_MAX = _SOIL_W_MAX
_FLASH_T2_CRUST_MIN = _FLASH_W_MIN


def generate_sites(seed: int):
    """Generate the 6,000 bi-temporal evaluation sites.

    Returns a list of dicts with t1/t2 (H, W, 5) reflectance blocks, metric
    (x, y), category, and ground-truth encroachment label.
    """
    from .parcels import Aoi, generate_parcels
    from ..config import AOI_CENTER_LATLON

    rng = np.random.default_rng(seed + 11)
    side = SITE_AOI_SIDE_M
    aoi = Aoi(*AOI_CENTER_LATLON, side)
    parcels, _ = generate_parcels(seed, aoi)
    prot = [p["ring"] for p in parcels if p["protected"]]
    unprot = [p["ring"] for p in parcels if not p["protected"]]

    def sample_point(rings):
        ring = rings[rng.integers(len(rings))]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        return float(rng.uniform(min(xs), max(xs))), float(rng.uniform(min(ys), max(ys)))

    sites = []

    def add(cat, t1_counts, t2_counts, rings, encroachment, weak=False, **extra):
        x, y = sample_point(rings)
        t1 = _render_counts(t1_counts, rng)
        t2 = _render_counts(_WEAK_BUILD if weak else t2_counts, rng)
        sites.append(
            dict(
                category=cat, t1=t1, t2=t2, x=x, y=y,
                encroachment=bool(encroachment), weak=weak,
                site_id=f"S-{len(sites):05d}", **extra,
            )
        )

    def moist_soil():
        return _openish(float(rng.uniform(0.0, _SOIL_W_MAX)))

    def flash_t2():
        # nearly crust-pure with the vegetation context trimmed: patch NDBI
        # lands ~0.15, safely above tau so every confuser fires the rule
        return _openish(float(rng.uniform(_FLASH_W_MIN, 1.0)), veg=1)

    a, b, c, d, e, f = (SITE_COUNTS[k] for k in "ABCDEF")
    # A: encroachment -- open land t1 -> built t2, inside protected parcels
    for i in range(a):
        weak = i < N_WEAK_SITES          # first site(s) are low-contrast
        add("A", moist_soil(), _NEW_BUILD, prot, encroachment=True, weak=weak)
    # B: pre-existing government built, inside protected, unchanged
    for _ in range(b):
        add("B", _OLD_BUILD, _OLD_BUILD, prot, encroachment=False)
    # C: legal new construction outside protected parcels
    for _ in range(c):
        add("C", moist_soil(), _NEW_BUILD, unprot, encroachment=False)
    # D: pre-existing private built outside protected
    for _ in range(d):
        add("D", _OLD_BUILD, _OLD_BUILD, unprot, encroachment=False)
    # E: stable non-built inside protected (same dryness at both dates), with
    # flash confusers (moist -> crusted) that trip the change rule
    for i in range(e):
        if i < N_FLASH_IN_PROTECTED:
            add("E", moist_soil(), flash_t2(), prot, encroachment=False, flash=True)
        else:
            w = float(rng.uniform(0.0, _SOIL_W_MAX))     # shared -> delta ~ 0
            add("E", _openish(w), _openish(w), prot, encroachment=False)
    # F: stable non-built outside protected, with flash confusers
    for i in range(f):
        if i < N_FLASH_OUTSIDE:
            add("F", moist_soil(), flash_t2(), unprot, encroachment=False, flash=True)
        else:
            w = float(rng.uniform(0.0, _SOIL_W_MAX))
            add("F", _openish(w), _openish(w), unprot, encroachment=False)

    rng.shuffle(sites)
    for i, s in enumerate(sites):
        s["site_id"] = f"S-{i:05d}"
    return sites, parcels, aoi


SITE_AOI_SIDE_M = 10_000   # 10 km x 10 km demo AOI
