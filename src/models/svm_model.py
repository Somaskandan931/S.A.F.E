"""
One-Class SVM Model for S.A.F.E
Boundary-based anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from typing import Dict
from src.models.base_model import BaseAnomalyModel


class OneClassSVMDetector( BaseAnomalyModel ) :
    """One-Class SVM for anomaly detection"""

    def __init__ ( self, kernel: str = 'rbf', nu: float = 0.1,
                   gamma: str = 'auto', random_state: int = 42 ) :
        """
        Initialize One-Class SVM

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            nu: Upper bound on fraction of outliers
            gamma: Kernel coefficient
            random_state: Random seed
        """
        super().__init__(
            name="OneClassSVM",
            config={
                'kernel' : kernel,
                'nu' : nu,
                'gamma' : gamma,
                'random_state' : random_state
            }
        )

        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Train One-Class SVM"""
        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            self.feature_names = list( X.columns )
            X = X.values
        else :
            self.feature_names = [f"feature_{i}" for i in range( X.shape[1] )]

        self.model.fit( X )
        self._compute_train_stats( X )
        self.is_fitted = True

        return self

    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """Get anomaly scores (distance from decision boundary)"""
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        # Get decision function (negative for outliers)
        scores = -self.model.decision_function( X )

        # Normalize to [0, 1] range
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score :
            scores = (scores - min_score) / (max_score - min_score)

        return scores

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """Predict anomaly labels"""
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        # SVM returns -1 for outliers, 1 for inliers
        predictions = self.model.predict( X )

        # Convert to 1 for anomaly, 0 for normal
        return (predictions == -1).astype( int )