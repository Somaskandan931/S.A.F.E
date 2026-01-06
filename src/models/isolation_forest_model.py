"""
Isolation Forest Model for S.A.F.E
Tree-based anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict
from src.models.base_model import BaseAnomalyModel


class IsolationForestDetector( BaseAnomalyModel ) :
    """Isolation Forest for anomaly detection"""

    def __init__ ( self, n_estimators: int = 100, contamination: float = 0.1,
                   max_samples: int = 256, random_state: int = 42 ) :
        """
        Initialize Isolation Forest

        Args:
            n_estimators: Number of trees
            contamination: Expected proportion of anomalies
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
        """
        super().__init__(
            name="IsolationForest",
            config={
                'n_estimators' : n_estimators,
                'contamination' : contamination,
                'max_samples' : max_samples,
                'random_state' : random_state
            }
        )

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Train Isolation Forest"""
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
        """
        Get anomaly scores

        Returns:
            Negative scores (lower = more anomalous)
            Converted to positive scores (higher = more anomalous)
        """
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        # Get decision scores (negative for anomalies)
        scores = self.model.decision_function( X )

        # Convert to positive anomaly scores
        # Higher score = more anomalous
        anomaly_scores = -scores

        # Normalize to [0, 1] range
        min_score, max_score = anomaly_scores.min(), anomaly_scores.max()
        if max_score > min_score :
            anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)

        return anomaly_scores

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """
        Predict anomaly labels

        Returns:
            1 for anomaly, 0 for normal
        """
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        # Sklearn returns -1 for anomaly, 1 for normal
        predictions = self.model.predict( X )

        # Convert to 1 for anomaly, 0 for normal
        return (predictions == -1).astype( int )

    def get_feature_importance ( self ) -> Dict[str, float] :
        """
        Approximate feature importance using path lengths

        Note: Isolation Forest doesn't have built-in feature importance
        This is an approximation
        """
        if not self.is_fitted :
            return {}

        # Use a simple heuristic based on variance
        importance = self.train_stats['feature_stds']
        importance = importance / np.sum( importance )

        return dict( zip( self.feature_names, importance ) )

    def get_contamination ( self ) -> float :
        """Get contamination parameter"""
        return self.config['contamination']

    def set_contamination ( self, contamination: float ) :
        """Update contamination parameter"""
        self.config['contamination'] = contamination
        self.model.set_params( contamination=contamination )


if __name__ == "__main__" :
    # Example usage
    from src.data.synthetic_generator import SyntheticCrowdGenerator
    from src.features.feature_engineering import FeatureEngineer
    from sklearn.metrics import precision_score, recall_score, f1_score

    print( "Testing Isolation Forest Model..." )

    # Generate data
    generator = SyntheticCrowdGenerator()
    df = generator.generate_normal_pattern( n_samples=1000 )
    df_anomaly = generator.inject_anomalies( df, anomaly_ratio=0.1 )

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.create_feature_matrix( df_anomaly )

    # Prepare data
    feature_cols = engineer.get_feature_names()
    X = df_features[feature_cols].values
    y = df_features['is_anomaly'].values

    # Split data
    split_idx = int( len( X ) * 0.8 )
    X_train, X_test = X[:split_idx], X[split_idx :]
    y_train, y_test = y[:split_idx], y[split_idx :]

    # Train model
    model = IsolationForestDetector(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )

    print( "Training model..." )
    model.fit( X_train )

    # Predict
    print( "Making predictions..." )
    y_pred = model.predict( X_test )
    scores = model.predict_score( X_test )

    # Evaluate
    precision = precision_score( y_test, y_pred )
    recall = recall_score( y_test, y_pred )
    f1 = f1_score( y_test, y_pred )

    print( f"\nResults:" )
    print( f"Precision: {precision:.3f}" )
    print( f"Recall: {recall:.3f}" )
    print( f"F1 Score: {f1:.3f}" )
    print( f"Anomalies detected: {y_pred.sum()} / {y_test.sum()}" )
    print( f"Score range: {scores.min():.3f} - {scores.max():.3f}" )

    # Feature importance
    importance = model.get_feature_importance()
    print( f"\nTop 5 important features:" )
    sorted_features = sorted( importance.items(), key=lambda x : x[1], reverse=True )
    for feat, imp in sorted_features[:5] :
        print( f"  {feat}: {imp:.4f}" )