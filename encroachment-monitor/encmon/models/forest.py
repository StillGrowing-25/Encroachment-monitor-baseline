from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest(RandomForestClassifier):
    def __init__(
        self,
        n_trees: int = 120,
        max_depth: int | None = 14,
        min_samples: int = 2,
        bins: int = 48,
        random_state: int = 0,
        **kwargs,
    ):
        n_estimators = kwargs.pop("n_estimators", n_trees)
        min_samples_split = kwargs.pop("min_samples_split", min_samples)
        max_features = kwargs.pop("max_features", "sqrt")
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            **kwargs,
        )
        self.n_trees = n_estimators
        self.min_samples = min_samples_split
        self.bins = bins

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        super().fit(X, y)
        return self

    @property
    def trees(self):
        return getattr(self, "estimators_", [])

    @property
    def n_classes(self) -> int | None:
        if hasattr(self, "classes_"):
            return len(self.classes_)
        return None
