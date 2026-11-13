"""Pixel-level segmentation quality on the synthetic scene (paper Sec. VI-C).

The three-stage pipeline runs per pixel: land-cover classification from
windowed spectral features, the NDBI change rule between dates, and the
protected-zone overlay.  Results are evaluated against the explicit
encroachment polygons (TP / FP / FN raster, Fig. 3 right).
"""
from __future__ import annotations

import numpy as np


def run_scene_pipeline(scene: dict, classifier, cfg=None):
    """Run detection over the bi-temporal scene; returns masks and timing."""
    import time

    from ..config import NDBI_DELTA, NDBI_TAU, WINDOW
    from ..spectra import raster_features

    tau = cfg.tau if cfg else NDBI_TAU
    delta = cfg.delta if cfg else NDBI_DELTA

    t0 = time.perf_counter()
    f1 = raster_features(scene["t1"], window=WINDOW)
    f2 = raster_features(scene["t2"], window=WINDOW)
    t_feat = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = classifier.predict(f2.reshape(-1, f2.shape[-1])).reshape(f2.shape[:2])
    built_pred = (pred == 3) | (pred == 4)
    t_clf = time.perf_counter() - t0

    ndbi1, ndbi2 = f1[..., 14], f2[..., 14]
    change = (ndbi2 > tau) & (ndbi1 <= tau) & ((ndbi2 - ndbi1) > delta)
    protected = scene["protected"]

    return dict(
        built_pred=built_pred,
        change_mask=change,
        flag_spectral=built_pred,
        flag_change=change,
        flag_full=(change & protected),
        protected=protected,
        timing=dict(feature_s=t_feat, classify_s=t_clf, total_s=t_feat + t_clf),
        pixels=f2.shape[0] * f2.shape[1],
    )


def segmentation_metrics(res: dict, gt: np.ndarray) -> dict:
    """Pixel-level P/R/F1/IoU of the full pipeline vs ground truth."""
    from .metrics import binary_counts, prf_iou

    return prf_iou(binary_counts(gt, res["flag_full"]))
