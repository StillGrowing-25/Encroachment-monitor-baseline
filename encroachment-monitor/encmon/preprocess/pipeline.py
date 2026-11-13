"""Preprocessing pipeline (paper Table III).

Radiometric correction (TOA -> surface reflectance), geometric correction /
co-registration (sub-pixel shift estimation and resampling), cloud masking
from a scene-classification band, and training augmentation.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Radiometric correction (top-of-atmosphere -> surface reflectance)
# ---------------------------------------------------------------------------
# Per-band forward model used by the simulator:
#   TOA = transmittance * SR + path_radiance
# correction inverts it with the sensor calibration constants.
TRANSMITTANCE = np.array([0.90, 0.92, 0.95, 0.96, 0.97])   # blue..swir
PATH_RADIANCE = np.array([0.040, 0.030, 0.020, 0.008, 0.003])


def to_toa(sr: np.ndarray) -> np.ndarray:
    """Forward model: surface reflectance -> top-of-atmosphere radiance-equivalent."""
    return TRANSMITTANCE * sr + PATH_RADIANCE


def to_surface_reflectance(toa: np.ndarray) -> np.ndarray:
    """Radiometric correction: invert the atmospheric term (Table III, step 1)."""
    return np.maximum((toa - PATH_RADIANCE) / TRANSMITTANCE, 0.0)


# ---------------------------------------------------------------------------
# Geometric correction / co-registration
# ---------------------------------------------------------------------------
def estimate_shift(img_ref: np.ndarray, img_mov: np.ndarray) -> tuple[float, float]:
    """Estimate sub-pixel (dy, dx) that aligns img_mov onto img_ref, i.e.
    shift_image(img_mov, dy, dx) registers the pair (< 1 px RMSE).

    Phase correlation on the mean band gives an integer peak; sub-pixel
    refinement uses the parabolic fit around the peak.
    """
    a = np.mean(img_ref, axis=-1)
    b = np.mean(img_mov, axis=-1)
    a = a - a.mean()
    b = b - b.mean()
    corr = np.abs(np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] if peak[0] <= a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] <= a.shape[1] // 2 else peak[1] - a.shape[1]

    def refine(cm1, c0, cp1):
        denom = (cm1 - 2 * c0 + cp1)
        if abs(denom) < 1e-12:
            return 0.0
        return float(0.5 * (cm1 - cp1) / denom)

    fy = refine(corr[(peak[0] - 1) % a.shape[0], peak[1]], corr[peak],
                corr[(peak[0] + 1) % a.shape[0], peak[1]])
    fx = refine(corr[peak[0], (peak[1] - 1) % a.shape[1]], corr[peak],
                corr[peak[0], (peak[1] + 1) % a.shape[1]])
    return dy + fy, dx + fx


def shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Bilinearly shift img by (dy, dx) with zero fill (resampling step)."""
    from ..datagen.scene import _fractional_shift

    out = _fractional_shift(img, dy, dx)
    h, w = img.shape[:2]
    iy, ix = int(np.ceil(abs(dy))), int(np.ceil(abs(dx)))
    if dy > 0:
        out[:iy] = 0
    elif dy < 0:
        out[h - iy:] = 0
    if dx > 0:
        out[:, :ix] = 0
    elif dx < 0:
        out[:, w - ix:] = 0
    return out


def co_register(t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Bring t2 onto the t1 grid to sub-pixel accuracy (< 1 px RMSE)."""
    dy, dx = estimate_shift(t1, t2)
    return shift_image(t2, dy, dx), (dy, dx)


# ---------------------------------------------------------------------------
# Cloud masking (scene-classification / QA band)
# ---------------------------------------------------------------------------
SCL_IDS = {
    "nodata": 0, "saturated": 1, "dark": 2, "cloud_shadow": 3, "vegetation": 4,
    "bare": 5, "water": 6, "cloud_low": 7, "cloud_medium": 8, "cloud_high": 9,
    "cirrus": 10,
}
CLOUD_IDS = {SCL_IDS["cloud_low"], SCL_IDS["cloud_medium"], SCL_IDS["cloud_high"],
             SCL_IDS["cirrus"], SCL_IDS["cloud_shadow"], SCL_IDS["nodata"],
             SCL_IDS["saturated"]}


def simulate_scl(label_map: np.ndarray, cloud_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Build a scene-classification band from a label map + synthetic cloud."""
    from ..datagen.scene import SURF_TO_ID

    scl = np.full(label_map.shape, SCL_IDS["bare"], dtype=np.int16)
    scl[label_map == SURF_TO_ID["vegetation"]] = SCL_IDS["vegetation"]
    scl[label_map == SURF_TO_ID["water"]] = SCL_IDS["water"]
    scl[label_map == SURF_TO_ID["built"]] = SCL_IDS["dark"]
    n = label_map.size
    n_cloud = int(cloud_frac * n)
    if n_cloud:
        idx = rng.choice(n, size=n_cloud, replace=False)
        ids = rng.choice(
            [SCL_IDS["cloud_low"], SCL_IDS["cloud_medium"], SCL_IDS["cirrus"]],
            size=n_cloud,
        )
        scl.ravel()[idx] = ids
    return scl


def cloud_mask(scl: np.ndarray) -> np.ndarray:
    """Boolean mask: True where the pixel is unusable (cloud / shadow / nodata)."""
    return np.isin(scl, list(CLOUD_IDS))


def apply_cloud_mask(scene: np.ndarray, scl: np.ndarray):
    """Zero out cloudy pixels; returns (masked scene, valid-pixel mask)."""
    valid = ~cloud_mask(scl)
    out = scene.copy()
    out[~valid] = 0.0
    return out, valid


# ---------------------------------------------------------------------------
# Training augmentation (flip, rotate, scale)
# ---------------------------------------------------------------------------
def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flip / 90-degree rotation / mild brightness scaling of a patch."""
    out = patch
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        out = np.rot90(out, k)
    if rng.random() < 0.3:
        f = float(rng.uniform(0.9, 1.1))
        out = np.clip(out * f, 0.0, 1.0)
    return np.ascontiguousarray(out)
