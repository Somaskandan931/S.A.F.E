"""
Statistical Anomaly Detection Models for S.A.F.E
Implements baseline statistical methods
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.models.base_model import BaseAnomalyModel


class ZScoreAnomalyDetector( BaseAnomalyModel ) :
    """Z-score based anomaly detection"""

    def __init__ ( self, threshold: float = 3.0, window_size: int = None ) :
        """
        Initialize Z-score detector

        Args:
            threshold: Z-score threshold for anomaly
            window_size: Rolling window size (None for global statistics)
        """
        super().__init__(
            name="ZScoreDetector",
            config={'threshold' : threshold, 'window_size' : window_size}
        )
        self.threshold = threshold
        self.window_size = window_size
        self.mean_ = None
        self.std_ = None

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Calculate mean and std from training data"""
        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            self.feature_names = list( X.columns )
            X = X.values
        else :
            self.feature_names = [f"feature_{i}" for i in range( X.shape[1] )]

        # Calculate global statistics
        self.mean_ = np.mean( X, axis=0 )
        self.std_ = np.std( X, axis=0 )

        # Prevent division by zero
        self.std_ = np.where( self.std_ == 0, 1, self.std_ )

        self._compute_train_stats( X )
        self.is_fitted = True
        return self

    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """Calculate anomaly scores based on z-scores"""
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        # Calculate z-scores for each feature
        z_scores = np.abs( (X - self.mean_) / self.std_ )

        # Aggregate across features (max z-score)
        anomaly_scores = np.max( z_scores, axis=1 )

        return anomaly_scores

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """Predict anomaly labels"""
        scores = self.predict_score( X )
        return (scores > self.threshold).astype( int )

    def get_feature_importance ( self ) -> Dict[str, float] :
        """Get feature importance based on variance"""
        if not self.is_fitted :
            return {}

        # Features with higher std are more important
        importance = self.std_ / np.sum( self.std_ )
        return dict( zip( self.feature_names, importance ) )


class MovingAverageAnomalyDetector( BaseAnomalyModel ) :
    """Moving average based anomaly detection"""

    def __init__ ( self, window_size: int = 10, threshold: float = 2.0 ) :
        """
        Initialize moving average detector

        Args:
            window_size: Size of moving window
            threshold: Threshold in standard deviations
        """
        super().__init__(
            name="MovingAverageDetector",
            config={'window_size' : window_size, 'threshold' : threshold}
        )
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_mean_ = None
        self.baseline_std_ = None

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Calculate baseline statistics"""
        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            self.feature_names = list( X.columns )
            X = X.values
        else :
            self.feature_names = [f"feature_{i}" for i in range( X.shape[1] )]

        # Use overall statistics as baseline
        self.baseline_mean_ = np.mean( X, axis=0 )
        self.baseline_std_ = np.std( X, axis=0 )
        self.baseline_std_ = np.where( self.baseline_std_ == 0, 1, self.baseline_std_ )

        self._compute_train_stats( X )
        self.is_fitted = True
        return self

    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """Calculate anomaly scores using moving average"""
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        anomaly_scores = np.zeros( len( X ) )

        # Calculate moving average for each feature
        for i in range( len( X ) ) :
            start_idx = max( 0, i - self.window_size )
            window_data = X[start_idx :i + 1]

            if len( window_data ) > 0 :
                window_mean = np.mean( window_data, axis=0 )
                window_std = np.std( window_data, axis=0 )
                window_std = np.where( window_std == 0, 1, window_std )

                # Calculate deviation from moving average
                deviations = np.abs( (X[i] - window_mean) / window_std )
                anomaly_scores[i] = np.max( deviations )

        return anomaly_scores

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """Predict anomaly labels"""
        scores = self.predict_score( X )
        return (scores > self.threshold).astype( int )


class ThresholdAnomalyDetector( BaseAnomalyModel ) :
    """Simple threshold-based anomaly detection"""

    def __init__ ( self, thresholds: Dict[str, tuple] = None,
                   percentile_based: bool = True ) :
        """
        Initialize threshold detector

        Args:
            thresholds: Dict mapping feature names to (min, max) thresholds
            percentile_based: Use percentiles instead of absolute values
        """
        super().__init__(
            name="ThresholdDetector",
            config={'thresholds' : thresholds, 'percentile_based' : percentile_based}
        )
        self.thresholds = thresholds or {}
        self.percentile_based = percentile_based
        self.learned_thresholds_ = {}

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Learn thresholds from data"""
        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            self.feature_names = list( X.columns )
            X_df = X
            X = X.values
        else :
            self.feature_names = [f"feature_{i}" for i in range( X.shape[1] )]
            X_df = pd.DataFrame( X, columns=self.feature_names )

        # Learn thresholds for each feature
        for idx, feature in enumerate( self.feature_names ) :
            if self.percentile_based :
                # Use percentiles (5th and 95th)
                lower = np.percentile( X[:, idx], 5 )
                upper = np.percentile( X[:, idx], 95 )
            else :
                # Use mean ± k*std
                mean = np.mean( X[:, idx] )
                std = np.std( X[:, idx] )
                lower = mean - 3 * std
                upper = mean + 3 * std

            self.learned_thresholds_[feature] = (lower, upper)

        # Override with manual thresholds if provided
        if self.thresholds :
            self.learned_thresholds_.update( self.thresholds )

        self._compute_train_stats( X )
        self.is_fitted = True
        return self

    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """Calculate anomaly scores based on threshold violations"""
        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        self._validate_input( X )

        if isinstance( X, pd.DataFrame ) :
            X = X.values

        anomaly_scores = np.zeros( len( X ) )

        for idx, feature in enumerate( self.feature_names ) :
            lower, upper = self.learned_thresholds_[feature]

            # Calculate how much values exceed thresholds
            below_lower = np.maximum( 0, lower - X[:, idx] )
            above_upper = np.maximum( 0, X[:, idx] - upper )

            # Normalize by threshold range
            threshold_range = upper - lower
            if threshold_range > 0 :
                violations = (below_lower + above_upper) / threshold_range
            else :
                violations = below_lower + above_upper

            anomaly_scores += violations

        # Normalize by number of features
        anomaly_scores /= len( self.feature_names )

        return anomaly_scores

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """Predict anomaly labels"""
        scores = self.predict_score( X )
        return (scores > 0).astype( int )  # Any threshold violation

    def get_thresholds ( self ) -> Dict[str, tuple] :
        """Get learned thresholds"""
        return self.learned_thresholds_


if __name__ == "__main__" :
    # Example usage
    from src.data.synthetic_generator import SyntheticCrowdGenerator
    from src.features.feature_engineering import FeatureEngineer

    # Generate data
    generator = SyntheticCrowdGenerator()
    df = generator.generate_normal_pattern( n_samples=500 )
    df_anomaly = generator.inject_anomalies( df, anomaly_ratio=0.1 )

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.create_feature_matrix( df_anomaly )

    # Prepare data
    feature_cols = engineer.get_feature_names()
    X = df_features[feature_cols].values
    y = df_features['is_anomaly'].values

    # Test Z-score detector
    print( "Testing Z-Score Detector..." )
    zscore_model = ZScoreAnomalyDetector( threshold=2.5 )
    zscore_model.fit( X[:400] )
    predictions = zscore_model.predict( X[400 :] )
    scores = zscore_model.predict_score( X[400 :] )
    print( f"Detected {predictions.sum()} anomalies" )
    print( f"Score range: {scores.min():.2f} - {scores.max():.2f}" )

    # Test threshold detector
    print( "\nTesting Threshold Detector..." )
    threshold_model = ThresholdAnomalyDetector( percentile_based=True )
    threshold_model.fit( X[:400] )
    predictions = threshold_model.predict( X[400 :] )
    print( f"Detected {predictions.sum()} anomalies" )
    print( f"Thresholds: {list( threshold_model.get_thresholds().items() )[:3]}" )