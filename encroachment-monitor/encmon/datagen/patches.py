"""Generator for the 12,000-patch five-class land-cover dataset (Table II).

Each patch is a 10 x 10 pixel composition at 10 m resolution (~100 m patch):
class-core pixels carry the class's spectral signature, the remaining
"context" pixels carry a class-specific surrounding mix.  Legal and Illegal
Construction share the *identical* built-up core signature -- by construction
-- and differ only weakly through their surroundings, which is exactly the
spectral ambiguity the paper measures.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    CLASS_SPECTRA,
    DRY_CRUST,
    PATCH,
    PATCH_CONTEXT_MIX,
    PATCH_CORE_FRACTION,
    CLASSES,
)
from ..spectra import FEATURE_NAMES, patch_features


def _sample_composition(cls: str, rng: np.random.Generator) -> np.ndarray:
    """Return a (PATCH, PATCH) label map of pixel-level surface materials.

    Open Land carries a per-patch *dryness gradient*: the core is a mixture
    of moist soil and bright dry crust with weight w ~ U(0, 1), which spans
    the natural NDBI range of bare soil (0.08 -> 0.16).  This is what lets
    seasonal crust flashes be in-distribution for the classifier while the
    change rule (tau=0.13) still separates moist soil from new construction.
    """
    surf = ["open", "vegetation", "water", "built"]
    core_frac = PATCH_CORE_FRACTION[cls]
    mix = PATCH_CONTEXT_MIX[cls]
    n_total = PATCH * PATCH
    n_context = n_total - int(round(core_frac * n_total))

    lab = np.empty(n_total, dtype=object)
    if cls == "open":
        w = float(rng.uniform(0.0, 1.0))               # patch dryness
        n_crust = int(round((n_total - n_context) * w))
        lab[:] = "open"                                 # moist soil base
        lab[:n_crust] = "crust"                         # dry-crust fraction
    elif cls in ("vegetation", "water"):
        lab[:] = cls
    else:
        lab[:] = "built"                                # legal/illegal core

    # context pixels drawn from the class's surrounding mix
    p = np.array([mix[s] for s in surf], dtype=float)
    p = p / p.sum()
    ctx = rng.choice(len(surf), size=n_context, p=p)
    idx = rng.permutation(n_total)[:n_context]
    lab[idx] = [surf[i] for i in ctx]
    return lab.reshape(PATCH, PATCH)


SURF_SPECTRA = {
    # surface-type spectra include the class cores and the context materials
    "open": CLASS_SPECTRA["open"],
    "vegetation": CLASS_SPECTRA["vegetation"],
    "water": CLASS_SPECTRA["water"],
    "built": CLASS_SPECTRA["legal"],   # one shared built-up signature
    "crust": DRY_CRUST,                # bright dry-crusted soil (flash)
}


def _render_pixels(label_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Render (PATCH, PATCH, 5) surface-reflectance pixels from a label map."""
    out = np.empty((PATCH, PATCH, 5), dtype=float)
    for surf, spec in SURF_SPECTRA.items():
        mask = label_map == surf
        if not mask.any():
            continue
        mu = np.asarray(spec["band_mean"])
        sigma = spec["band_std"]
        # pixel noise + a patch-scale brightness variation (illumination/soil)
        patch_gain = rng.normal(1.0, 0.06)
        noise = rng.normal(0.0, sigma, size=(int(mask.sum()), 5))
        out[mask] = np.maximum(mu * patch_gain + noise, 0.001)
    return out


def generate_patches(n_patches: int, seed: int):
    """Generate the balanced five-class patch dataset.

    Returns (X [n,18] feature matrix, y [n] class index, meta list of dicts).
    """
    from ..config import CLASSES  # local import to avoid cycles in tests

    rng = np.random.default_rng(seed)
    per_class = n_patches // len(CLASSES)
    X, y, meta = [], [], []
    for ci, cls in enumerate(CLASSES):
        for k in range(per_class):
            lab = _sample_composition(cls, rng)
            bands = _render_pixels(lab, rng)
            X.append(patch_features(bands))
            y.append(ci)
            meta.append({"class": cls, "patch_id": f"{cls}_{k:05d}"})
    X = np.asarray(X)
    y = np.asarray(y)
    order = rng.permutation(len(y))
    return X[order], y[order], [meta[i] for i in order]


FEATURE_NAMES_ = FEATURE_NAMES
