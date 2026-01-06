"""
Base Model Class for S.A.F.E
Abstract base class for all anomaly detection models
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib


class BaseAnomalyModel( ABC ) :
    """Abstract base class for anomaly detection models"""

    def __init__ ( self, name: str, config: Dict = None ) :
        """
        Initialize base model

        Args:
            name: Model name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names = []
        self.train_stats = {}

    @abstractmethod
    def fit ( self, X: np.ndarray, y: np.ndarray = None ) -> 'BaseAnomalyModel' :
        """
        Train the model

        Args:
            X: Training features
            y: Training labels (optional for unsupervised)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """
        Predict anomaly labels

        Args:
            X: Input features

        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        pass

    @abstractmethod
    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """
        Predict anomaly scores

        Args:
            X: Input features

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        pass

    def fit_predict ( self, X: np.ndarray, y: np.ndarray = None ) -> np.ndarray :
        """
        Fit model and predict on same data

        Args:
            X: Features
            y: Labels (optional)

        Returns:
            Predictions
        """
        self.fit( X, y )
        return self.predict( X )

    def save_model ( self, filepath: str ) :
        """Save model to disk"""
        if not self.is_fitted :
            raise ValueError( "Cannot save unfitted model" )

        joblib.dump( {
            'model' : self,
            'config' : self.config,
            'feature_names' : self.feature_names,
            'train_stats' : self.train_stats
        }, filepath )
        print( f"Model saved to {filepath}" )

    @classmethod
    def load_model ( cls, filepath: str ) -> 'BaseAnomalyModel' :
        """Load model from disk"""
        data = joblib.load( filepath )
        model = data['model']
        model.config = data['config']
        model.feature_names = data['feature_names']
        model.train_stats = data['train_stats']
        return model

    def get_params ( self ) -> Dict[str, Any] :
        """Get model parameters"""
        return self.config

    def set_params ( self, **params ) :
        """Set model parameters"""
        self.config.update( params )
        return self

    def _validate_input ( self, X: np.ndarray ) :
        """Validate input data"""
        if not isinstance( X, (np.ndarray, pd.DataFrame) ) :
            raise ValueError( "Input must be numpy array or pandas DataFrame" )

        if len( X.shape ) != 2 :
            raise ValueError( f"Input must be 2D, got shape {X.shape}" )

        if self.is_fitted and X.shape[1] != len( self.feature_names ) :
            raise ValueError(
                f"Expected {len( self.feature_names )} features, got {X.shape[1]}"
            )

    def _compute_train_stats ( self, X: np.ndarray ) :
        """Compute training statistics"""
        self.train_stats = {
            'n_samples' : X.shape[0],
            'n_features' : X.shape[1],
            'feature_means' : np.mean( X, axis=0 ),
            'feature_stds' : np.std( X, axis=0 ),
            'feature_mins' : np.min( X, axis=0 ),
            'feature_maxs' : np.max( X, axis=0 )
        }

    def get_feature_importance ( self ) -> Dict[str, float] :
        """
        Get feature importance scores

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Default implementation - override in specific models
        return {name : 1.0 / len( self.feature_names )
                for name in self.feature_names}

    def __repr__ ( self ) :
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"


class EnsembleAnomalyModel( BaseAnomalyModel ) :
    """Ensemble of multiple anomaly detection models"""

    def __init__ ( self, models: list, weights: list = None,
                   aggregation: str = 'mean' ) :
        """
        Initialize ensemble

        Args:
            models: List of BaseAnomalyModel instances
            weights: Optional weights for each model
            aggregation: How to combine predictions ('mean', 'max', 'voting')
        """
        super().__init__( name="Ensemble", config={} )
        self.models = models
        self.weights = weights or [1.0] * len( models )
        self.aggregation = aggregation

        if len( self.models ) != len( self.weights ) :
            raise ValueError( "Number of models and weights must match" )

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Fit all models in ensemble"""
        for model in self.models :
            model.fit( X, y )

        self.is_fitted = True
        self.feature_names = self.models[0].feature_names
        return self

    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """Get ensemble anomaly scores"""
        scores = []

        for model, weight in zip( self.models, self.weights ) :
            model_scores = model.predict_score( X )
            scores.append( model_scores * weight )

        scores = np.array( scores )

        if self.aggregation == 'mean' :
            return np.mean( scores, axis=0 )
        elif self.aggregation == 'max' :
            return np.max( scores, axis=0 )
        elif self.aggregation == 'median' :
            return np.median( scores, axis=0 )
        else :
            raise ValueError( f"Unknown aggregation: {self.aggregation}" )

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """Get ensemble predictions"""
        if self.aggregation == 'voting' :
            predictions = []
            for model in self.models :
                predictions.append( model.predict( X ) )

            # Majority voting
            predictions = np.array( predictions )
            return (np.mean( predictions, axis=0 ) > 0.5).astype( int )
        else :
            scores = self.predict_score( X )
            threshold = np.percentile( scores, 90 )
            return (scores > threshold).astype( int )