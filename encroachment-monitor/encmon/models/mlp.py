from __future__ import annotations

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


class MLP(MLPClassifier):

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 60,
        lr: float = 2e-3,
        batch: int = 128,
        l2: float = 1e-4,
        random_state: int = 0,
        **kwargs,
    ):
        hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", hidden)
        max_iter = kwargs.pop("max_iter", epochs)
        learning_rate_init = kwargs.pop("learning_rate_init", lr)
        batch_size = kwargs.pop("batch_size", batch)
        alpha = kwargs.pop("alpha", l2)
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        self.hidden = hidden_layer_sizes
        self.epochs = max_iter
        self.lr = learning_rate_init
        self.batch = batch_size
        self.l2 = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "MLP":
        if verbose:
            self.verbose = True
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        # Adapt batch_size if sample size is smaller than batch
        if isinstance(self.batch_size, int) and len(X) < self.batch_size:
            self.batch_size = min(len(X), self.batch)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            super().fit(X, y)
        return self

    @property
    def n_classes(self) -> int | None:
        if hasattr(self, "classes_"):
            return len(self.classes_)
        return None
