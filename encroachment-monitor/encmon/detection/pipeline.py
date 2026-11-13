"""Three-stage encroachment pipeline and its ablation (paper Sec. Methodology,
Table VI).

Configurations:
  1. spectral only      -- land-cover classifier on the t2 observation; every
                           built-up pixel/site is treated as an encroachment.
  2. + change detection -- NDBI bi-temporal rule keeps only *new* construction.
  3. + GIS overlay      -- new construction intersected with the protected
                           parcel layer; only then is it an encroachment.
"""
from __future__ import annotations

import numpy as np

from .change import ChangeConfig, detect_sites
from .gis import ProtectedLayer


class EncroachmentPipeline:
    """Full pipeline: spectral classification -> change detection -> GIS."""

    def __init__(self, classifier, parcels: list[dict], cfg: ChangeConfig | None = None):
        self.clf = classifier
        self.layer = ProtectedLayer(parcels)
        self.cfg = cfg or ChangeConfig()

    # ------------------------------------------------------------------
    def classify_built(self, sites: list[dict]) -> list[dict]:
        """Stage 1: predict land cover at t2 for every site."""
        from ..spectra import patch_features

        X = np.array([patch_features(s["t2"]) for s in sites])
        pred = self.clf.predict(X)
        return [{**s, "pred_class": int(p)} for s, p in zip(sites, pred)]

    # ------------------------------------------------------------------
    def run(self, sites: list[dict]) -> list[dict]:
        """Annotate sites with all three configuration's decisions."""
        detected = detect_sites(sites, self.cfg)
        out = []
        for s in detected:
            protected, parcel_id = self.layer.contains_point(s["x"], s["y"])
            spectral_only = s["pred_class"] in (3, 4)          # legal/illegal built
            out.append(
                {
                    **s,
                    "protected": bool(protected),
                    "parcel_id": parcel_id,
                    "flag_spectral": spectral_only,
                    "flag_change": bool(s["change"]),
                    "flag_full": bool(s["change"] and protected),
                }
            )
        return out


def ablation(site_results: list[dict]) -> list[dict]:
    """Compute the Table VI ablation rows from annotated sites."""
    from ..evaluation.metrics import binary_counts, prf_iou

    y = np.array([s["encroachment"] for s in site_results], dtype=bool)
    rows = []
    for name, key in (
        ("Spectral only", "flag_spectral"),
        ("+ Change detection", "flag_change"),
        ("+ Change + GIS", "flag_full"),
    ):
        pred = np.array([bool(s[key]) for s in site_results], dtype=bool)
        rows.append(dict(config=name, **prf_iou(binary_counts(y, pred))))
    return rows
