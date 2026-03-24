"""
Statistical Anomaly Detection Models for S.A.F.E - FIXED (Research-Grade)
===========================================================================
FIXES APPLIED:
  1. All models compute thresholds from TRAINING scores only and store them.
     predict() on val/test uses the stored threshold — no recalculation.
  2. ZScoreAnomalyDetector: threshold is chosen as the score at the 95th
     percentile of the training distribution, not a fixed z-score constant.
     This makes the model adaptive without looking at the test set.
  3. MovingAverageAnomalyDetector: same pattern.
  4. ThresholdAnomalyDetector: kept as-is but uses percentile-based bounds
     learned from training data only.
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.models.base_model import BaseAnomalyModel


class ZScoreAnomalyDetector(BaseAnomalyModel):
    """Z-score based anomaly detection."""

    def __init__(self, threshold: float = 3.0, window_size: int = None):
        super().__init__(
            name="ZScoreDetector",
            config={'threshold': threshold, 'window_size': window_size}
        )
        self.threshold = threshold
        self.window_size = window_size
        self.mean_: np.ndarray = None
        self.std_:  np.ndarray = None
        self._train_threshold: float = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit on NORMAL training data; y is ignored."""
        self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0)
        self.std_  = np.where(self.std_ == 0, 1, self.std_)

        self._compute_train_stats(X)

        # Threshold from training score distribution (p95)
        train_scores = self._compute_scores(X)
        self._train_threshold = float(np.percentile(train_scores, 95))
        print(f"  ZScoreDetector: training threshold (p95) = {self._train_threshold:.6f}")

        self.is_fitted = True
        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mean_) / self.std_)
        return np.max(z, axis=1)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._compute_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use TRAINING threshold — not the raw z-score constant."""
        scores = self.predict_score(X)
        t = self._train_threshold if self._train_threshold is not None else self.threshold
        return (scores > t).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importance = self.std_ / np.sum(self.std_)
        return dict(zip(self.feature_names, importance))


class MovingAverageAnomalyDetector(BaseAnomalyModel):
    """Moving-average based anomaly detection."""

    def __init__(self, window_size: int = 10, threshold: float = 2.0):
        super().__init__(
            name="MovingAverageDetector",
            config={'window_size': window_size, 'threshold': threshold}
        )
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_mean_: np.ndarray = None
        self.baseline_std_:  np.ndarray = None
        self._train_threshold: float = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit on NORMAL training data; y is ignored."""
        self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.baseline_mean_ = np.mean(X, axis=0)
        self.baseline_std_  = np.std(X, axis=0)
        self.baseline_std_  = np.where(self.baseline_std_ == 0, 1, self.baseline_std_)

        self._compute_train_stats(X)

        # Cache threshold from training scores
        train_scores = self._compute_scores(X)
        self._train_threshold = float(np.percentile(train_scores, 95))
        print(f"  MovingAverageDetector: training threshold (p95) = {self._train_threshold:.6f}")

        self.is_fitted = True
        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X))
        for i in range(len(X)):
            start = max(0, i - self.window_size)
            window = X[start:i + 1]
            if len(window) > 0:
                w_mean = np.mean(window, axis=0)
                w_std  = np.std(window, axis=0)
                w_std  = np.where(w_std == 0, 1, w_std)
                dev    = np.abs((X[i] - w_mean) / w_std)
                scores[i] = np.max(dev)
        return scores

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._compute_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use TRAINING threshold."""
        scores = self.predict_score(X)
        t = self._train_threshold if self._train_threshold is not None else self.threshold
        return (scores > t).astype(int)


class ThresholdAnomalyDetector(BaseAnomalyModel):
    """Percentile-based threshold anomaly detection."""

    def __init__(self, thresholds: Dict[str, tuple] = None,
                 percentile_based: bool = True):
        super().__init__(
            name="ThresholdDetector",
            config={'thresholds': thresholds, 'percentile_based': percentile_based}
        )
        self.thresholds          = thresholds or {}
        self.percentile_based    = percentile_based
        self.learned_thresholds_ = {}
        self._train_threshold: float = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Learn thresholds from NORMAL training data; y is ignored."""
        self._validate_input(X)

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_vals = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_vals = X

        for idx, feature in enumerate(self.feature_names):
            if self.percentile_based:
                lower = np.percentile(X_vals[:, idx], 5)
                upper = np.percentile(X_vals[:, idx], 95)
            else:
                mu  = np.mean(X_vals[:, idx])
                sig = np.std(X_vals[:, idx])
                lower = mu - 3 * sig
                upper = mu + 3 * sig
            self.learned_thresholds_[feature] = (lower, upper)

        if self.thresholds:
            self.learned_thresholds_.update(self.thresholds)

        self._compute_train_stats(X_vals)

        # Cache training score p95
        train_scores = self._compute_scores(X_vals)
        self._train_threshold = float(np.percentile(train_scores, 95))
        print(f"  ThresholdDetector: training threshold (p95) = {self._train_threshold:.6f}")

        self.is_fitted = True
        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X))
        for idx, feature in enumerate(self.feature_names):
            lower, upper = self.learned_thresholds_[feature]
            below = np.maximum(0, lower - X[:, idx])
            above = np.maximum(0, X[:, idx] - upper)
            rng   = upper - lower
            if rng > 0:
                violations = (below + above) / rng
            else:
                violations = below + above
            scores += violations
        return scores / max(len(self.feature_names), 1)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._compute_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use TRAINING threshold."""
        scores = self.predict_score(X)
        t = self._train_threshold if self._train_threshold is not None else 0.0
        return (scores > t).astype(int)

    def get_thresholds(self) -> Dict[str, tuple]:
        return self.learned_thresholds_