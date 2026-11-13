"""Permutation feature importance (paper Sec. VI-D, Fig. 4).

For each feature, the column is randomly permuted on the test set and the
accuracy drop is measured -- a model-agnostic, fully auditable explanation
of what the land-cover classifier relies on.
"""
from __future__ import annotations

import numpy as np


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = float((model.predict(X) == y).mean())
    out = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            acc = float((model.predict(Xp) == y).mean())
            drops.append(base - acc)
        out.append(
            dict(
                feature=name,
                importance_mean=float(np.mean(drops)),
                importance_std=float(np.std(drops)),
            )
        )
    out.sort(key=lambda d: -d["importance_mean"])
    return out
