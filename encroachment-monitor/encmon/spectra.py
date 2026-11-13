from __future__ import annotations
import numpy as np
from .config import BANDS
_EPS = 1e-8


def ndvi(bands: np.ndarray, axis: int = -1) -> np.ndarray:
    nir = bands[..., 3]
    red = bands[..., 2]
    return (nir - red) / (nir + red + _EPS)


def ndwi(bands: np.ndarray, axis: int = -1) -> np.ndarray:
    green = bands[..., 1]
    nir = bands[..., 3]
    return (green - nir) / (green + nir + _EPS)


def ndbi(bands: np.ndarray, axis: int = -1) -> np.ndarray:
    swir = bands[..., 4]
    nir = bands[..., 3]
    return (swir - nir) / (swir + nir + _EPS)


INDICES_FN = {"ndvi": ndvi, "ndwi": ndwi, "ndbi": ndbi}


def texture_local_variance(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype=float)
    pad = np.pad(band, [(0, 0)] * (band.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    stack = np.stack(
        [
            pad[..., :-2, :-2], pad[..., :-2, 1:-1], pad[..., :-2, 2:],
            pad[..., 1:-1, :-2], pad[..., 1:-1, 1:-1], pad[..., 1:-1, 2:],
            pad[..., 2:, :-2], pad[..., 2:, 1:-1], pad[..., 2:, 2:],
        ],
        axis=0,
    )
    var = stack.var(axis=0)
    return var


FEATURE_NAMES = (
    [f"{b}_mean" for b in BANDS]
    + [f"{b}_std" for b in BANDS]
    + [f"{i}_{s}" for i in ("ndvi", "ndwi", "ndbi") for s in ("mean", "std")]
    + ["texture_mean", "texture_max"]
)


def patch_features(bands: np.ndarray) -> np.ndarray:
    feats = []
    for b in range(5):
        feats.append(bands[..., b].mean())
    for b in range(5):
        feats.append(bands[..., b].std())
    for fn in (ndvi, ndwi, ndbi):
        idx = fn(bands)
        feats.append(idx.mean())
        feats.append(idx.std())
    tex = texture_local_variance(bands[..., 3])
    feats.append(tex.mean())
    feats.append(tex.max())
    return np.asarray(feats, dtype=float)


def raster_features(bands: np.ndarray, window: int = 5) -> np.ndarray:
    bands = np.asarray(bands, dtype=float)
    pad = window // 2
    feats = []
    for b in range(5):
        band = bands[..., b]
        m = _window_mean(band, pad)
        feats.append(m)
        feats.append(np.sqrt(np.maximum(_window_mean(band * band, pad) - m * m, 0.0)))
    for fn in (ndvi, ndwi, ndbi):
        idx = fn(bands)
        m = _window_mean(idx, pad)
        feats.append(m)
        feats.append(np.sqrt(np.maximum(_window_mean(idx * idx, pad) - m * m, 0.0)))
    tex = texture_local_variance(bands[..., 3])
    feats.append(_window_mean(tex, pad))
    feats.append(_window_max(tex, pad))
    return np.stack(feats, axis=-1)


def _window_mean(a: np.ndarray, pad: int) -> np.ndarray:
    a = np.pad(a, [(pad, pad), (pad, pad)], mode="edge")
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    k = 2 * pad + 1
    s = (
        ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    )
    return s / (k * k)


def _window_max(a: np.ndarray, pad: int) -> np.ndarray:
    a = np.pad(a, [(pad, pad), (pad, pad)], mode="edge")
    h, w = a.shape[0] - 2 * pad, a.shape[1] - 2 * pad
    best = np.full((h, w), -np.inf)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = a[pad + dy: pad + dy + h, pad + dx: pad + dx + w]
            best = np.maximum(best, shifted)
    return best
