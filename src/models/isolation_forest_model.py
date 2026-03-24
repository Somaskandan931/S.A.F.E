"""
Isolation Forest Model for S.A.F.E - FIXED (Research-Grade)
=============================================================
FIXES APPLIED:
  1. fit() accepts ONLY normal (non-anomalous) data.
     The caller (SAFEPipeline.train_models) is responsible for passing
     normal-only data; this model does not filter by label.
  2. threshold is computed from training scores at the 95th percentile and
     stored; predict() uses this stored threshold so it is consistent
     across val/test sets.
  3. contamination parameter reflects the true expected anomaly rate in
     deployment (~5-10 %), not the training data composition.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict
from src.models.base_model import BaseAnomalyModel


class IsolationForestDetector(BaseAnomalyModel):
    """Isolation Forest for anomaly detection."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05,
                 max_samples: int = 256, random_state: int = 42):
        """
        Parameters
        ----------
        contamination : float
            Expected fraction of anomalies in deployment data.
            Set to a realistic value (0.05 = 5 %) — NOT to the fraction of
            injected anomalies in evaluation, which would be circular.
        """
        super().__init__(
            name="IsolationForest",
            config={
                'n_estimators':  n_estimators,
                'contamination': contamination,
                'max_samples':   max_samples,
                'random_state':  random_state,
            }
        )

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        # Stored training-score threshold (set after fit)
        self._train_threshold: float = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Train on NORMAL data only.

        y is accepted for API compatibility but IGNORED.
        If y contains anomaly labels the caller should have already
        filtered to y == 0 before passing X here.
        """
        self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.model.fit(X)
        self._compute_train_stats(X)

        # Store threshold at p95 of training anomaly scores
        train_scores = self._raw_scores(X)
        self._train_threshold = float(np.percentile(train_scores, 95))
        print(f"  IsolationForest: training threshold (p95) = {self._train_threshold:.6f}")

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Return positive anomaly scores (higher = more anomalous), [0, 1]."""
        raw = -self.model.decision_function(X)           # higher = more anomalous
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
        """
        Predict anomaly labels using the TRAINING threshold.

        This ensures that the decision boundary is never re-fitted on
        validation or test data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            X = X.values

        scores = self._raw_scores(X)

        if self._train_threshold is not None:
            return (scores > self._train_threshold).astype(int)
        # Fallback: sklearn's internal decision (contamination-based)
        raw_pred = self.model.predict(X)
        return (raw_pred == -1).astype(int)

    # ------------------------------------------------------------------
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importance = self.train_stats['feature_stds']
        importance = importance / np.sum(importance)
        return dict(zip(self.feature_names, importance))

    def get_contamination(self) -> float:
        return self.config['contamination']

    def set_contamination(self, contamination: float):
        self.config['contamination'] = contamination
        self.model.set_params(contamination=contamination)