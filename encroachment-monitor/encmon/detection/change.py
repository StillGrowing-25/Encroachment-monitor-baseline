"""Bi-temporal NDBI change detection (paper Methodology).

    new(t) = (NDBI_t2 > TAU) AND (NDBI_t1 <= TAU) AND (delta NDBI > DELTA)

A significant jump from a non-built to a built-up state marks new
construction; the double threshold (absolute level and delta) is robust to
the seasonal and illumination noise that defeats raw pixel differencing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChangeConfig:
    tau: float = 0.12      # absolute built-up level threshold (NDBI)
    delta: float = 0.05    # minimum NDBI increase between dates


def site_ndbi(ndbi_t1: np.ndarray, ndbi_t2: np.ndarray, cfg: ChangeConfig | None = None) -> np.ndarray:
    """Vectorized change rule for arrays of per-site NDBI values."""
    cfg = cfg or ChangeConfig()
    return (ndbi_t2 > cfg.tau) & (ndbi_t1 <= cfg.tau) & ((ndbi_t2 - ndbi_t1) > cfg.delta)


def detect_sites(sites: list[dict], cfg: ChangeConfig | None = None) -> list[dict]:
    """Run the change rule over bi-temporal sites.

    Each site dict needs 't1' and 't2' (H, W, 5) reflectance blocks.  The
    returned dicts add: ndbi_t1, ndbi_t2, d_ndbi, change (bool).
    """
    from ..spectra import ndbi

    cfg = cfg or ChangeConfig()
    out = []
    for s in sites:
        n1 = float(ndbi(s["t1"]).mean())
        n2 = float(ndbi(s["t2"]).mean())
        d = n2 - n1
        fired = bool((n2 > cfg.tau) and (n1 <= cfg.tau) and (d > cfg.delta))
        out.append({**s, "ndbi_t1": n1, "ndbi_t2": n2, "d_ndbi": d, "change": fired})
    return out


def detect_raster(ndbi_t1: np.ndarray, ndbi_t2: np.ndarray, cfg: ChangeConfig | None = None) -> np.ndarray:
    """Pixel-wise change mask for full rasters of NDBI maps."""
    cfg = cfg or ChangeConfig()
    return (ndbi_t2 > cfg.tau) & (ndbi_t1 <= cfg.tau) & ((ndbi_t2 - ndbi_t1) > cfg.delta)
