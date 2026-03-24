"""
One-Class SVM Model for S.A.F.E - FIXED (Research-Grade)
==========================================================
FIXES APPLIED:
  1. fit() trains ONLY on normal data; y is ignored.
  2. Decision threshold is computed from training scores (p95) and stored,
     so predict() on val/test uses the same boundary without re-fitting.
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from typing import Dict
from src.models.base_model import BaseAnomalyModel


class OneClassSVMDetector(BaseAnomalyModel):
    """One-Class SVM for anomaly detection."""

    def __init__(self, kernel: str = 'rbf', nu: float = 0.05,
                 gamma: str = 'auto', random_state: int = 42):
        """
        Parameters
        ----------
        nu : float
            Upper bound on the fraction of training errors.
            Set to a realistic deployment anomaly rate (0.05 = 5 %).
        """
        super().__init__(
            name="OneClassSVM",
            config={'kernel': kernel, 'nu': nu, 'gamma': gamma,
                    'random_state': random_state}
        )
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self._train_threshold: float = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Train on NORMAL data only; y is ignored."""
        self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.model.fit(X)
        self._compute_train_stats(X)

        # Cache training threshold
        train_scores = self._raw_scores(X)
        self._train_threshold = float(np.percentile(train_scores, 95))
        print(f"  OneClassSVM: training threshold (p95) = {self._train_threshold:.6f}")

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Normalised positive anomaly scores ([0, 1], higher = more anomalous)."""
        raw = -self.model.decision_function(X)
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            return (raw - lo) / (hi - lo)
        return raw

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._raw_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stored TRAINING threshold."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            X = X.values

        scores = self._raw_scores(X)
        if self._train_threshold is not None:
            return (scores > self._train_threshold).astype(int)
        # Fallback
        raw_pred = self.model.predict(X)
        return (raw_pred == -1).astype(int)