"""
Explainability Module for S.A.F.E
Provides interpretable insights into model predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try :
    import shap

    SHAP_AVAILABLE = True
except ImportError :
    SHAP_AVAILABLE = False
    print( "⚠️  SHAP not available. Install with: pip install shap" )


class ExplainabilityAnalyzer :
    """Analyze and explain model predictions"""

    def __init__ ( self ) :
        self.feature_names = None
        self.explainer = None

    def calculate_feature_importance ( self, model, X: np.ndarray,
                                       feature_names: List[str] ) -> Dict[str, float] :
        """
        Calculate feature importance scores

        Args:
            model: Trained model with feature_importances_ or get_feature_importance()
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        self.feature_names = feature_names

        # Try different methods to get feature importance
        if hasattr( model, 'feature_importances_' ) :
            # Tree-based models (e.g., Isolation Forest)
            importances = model.feature_importances_
        elif hasattr( model, 'get_feature_importance' ) :
            # Custom implementation
            importance_dict = model.get_feature_importance()
            return importance_dict
        elif hasattr( model, 'coef_' ) :
            # Linear models
            importances = np.abs( model.coef_[0] )
        else :
            # Fallback: use variance as proxy
            importances = np.var( X, axis=0 )

        # Normalize
        importances = importances / np.sum( importances )

        return dict( zip( feature_names, importances ) )

    def plot_feature_importance ( self, importance_dict: Dict[str, float],
                                  top_n: int = 10,
                                  save_path: Optional[str] = None ) -> plt.Figure :
        """
        Plot feature importance

        Args:
            importance_dict: Dictionary of feature importances
            top_n: Number of top features to display
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Sort by importance
        sorted_features = sorted( importance_dict.items(),
                                  key=lambda x : x[1],
                                  reverse=True )[:top_n]

        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]

        # Create plot
        fig, ax = plt.subplots( figsize=(10, 6) )
        ax.barh( features, importances )
        ax.set_xlabel( 'Importance Score' )
        ax.set_title( f'Top {top_n} Feature Importances' )
        ax.invert_yaxis()

        # Add values on bars
        for i, v in enumerate( importances ) :
            ax.text( v, i, f' {v:.4f}', va='center' )

        plt.tight_layout()

        if save_path :
            Path( save_path ).parent.mkdir( parents=True, exist_ok=True )
            plt.savefig( save_path, dpi=300, bbox_inches='tight' )

        return fig

    def explain_prediction_shap ( self, model, X_train: np.ndarray,
                                  X_test: np.ndarray,
                                  feature_names: List[str] ) -> Dict :
        """
        Use SHAP to explain predictions

        Args:
            model: Trained model
            X_train: Training data (for background)
            X_test: Test data to explain
            feature_names: List of feature names

        Returns:
            Dictionary with SHAP values and explanations
        """
        if not SHAP_AVAILABLE :
            return {
                'error' : 'SHAP not available',
                'message' : 'Install with: pip install shap'
            }

        self.feature_names = feature_names

        # Create SHAP explainer
        # Use a sample of training data as background
        background = shap.sample( X_train, min( 100, len( X_train ) ) )

        try :
            # Try TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer( model.model if hasattr( model, 'model' ) else model )
        except :
            # Fallback to KernelExplainer
            def predict_fn ( x ) :
                if hasattr( model, 'predict_score' ) :
                    return model.predict_score( x )
                else :
                    return model.predict( x )

            self.explainer = shap.KernelExplainer( predict_fn, background )

        # Calculate SHAP values
        shap_values = self.explainer.shap_values( X_test[:min( 100, len( X_test ) )] )

        return {
            'shap_values' : shap_values,
            'feature_names' : feature_names,
            'base_value' : self.explainer.expected_value if hasattr( self.explainer, 'expected_value' ) else 0
        }

    def create_explanation_report ( self, prediction: int,
                                    score: float,
                                    features: Dict[str, float],
                                    feature_importance: Dict[str, float] ) -> str :
        """
        Create human-readable explanation report

        Args:
            prediction: Model prediction (0 or 1)
            score: Anomaly score
            features: Feature values
            feature_importance: Feature importance scores

        Returns:
            String report
        """
        report = "\n" + "=" * 60 + "\n"
        report += "PREDICTION EXPLANATION\n"
        report += "=" * 60 + "\n\n"

        # Prediction summary
        risk_level = "HIGH RISK" if prediction == 1 else "NORMAL"
        report += f"Risk Level: {risk_level}\n"
        report += f"Anomaly Score: {score:.4f}\n\n"

        # Top contributing features
        report += "Top Contributing Features:\n"
        report += "-" * 60 + "\n"

        # Sort features by importance
        sorted_features = sorted( feature_importance.items(),
                                  key=lambda x : x[1],
                                  reverse=True )[:5]

        for i, (feature, importance) in enumerate( sorted_features, 1 ) :
            value = features.get( feature, 0 )
            report += f"{i}. {feature}\n"
            report += f"   Value: {value:.4f} | Importance: {importance:.4f}\n"

        report += "\n" + "=" * 60 + "\n"

        return report

    def analyze_anomaly_patterns ( self, df: pd.DataFrame,
                                   anomaly_col: str = 'is_anomaly' ) -> Dict :
        """
        Analyze patterns in detected anomalies

        Args:
            df: DataFrame with predictions
            anomaly_col: Column name for anomaly labels

        Returns:
            Dictionary with pattern analysis
        """
        if anomaly_col not in df.columns :
            return {'error' : 'Anomaly column not found'}

        anomalies = df[df[anomaly_col] == 1]

        analysis = {
            'total_anomalies' : len( anomalies ),
            'percentage' : len( anomalies ) / len( df ) * 100,
        }

        # Temporal patterns
        if 'timestamp' in df.columns :
            df['timestamp'] = pd.to_datetime( df['timestamp'] )
            anomalies['hour'] = anomalies['timestamp'].dt.hour
            analysis['by_hour'] = anomalies['hour'].value_counts().to_dict()

        # Spatial patterns
        if 'zone_id' in df.columns :
            analysis['by_zone'] = anomalies['zone_id'].value_counts().to_dict()

        # Feature patterns
        numeric_cols = df.select_dtypes( include=[np.number] ).columns
        numeric_cols = [c for c in numeric_cols if c != anomaly_col]

        analysis['feature_stats'] = {}
        for col in numeric_cols[:10] :  # Top 10 features
            analysis['feature_stats'][col] = {
                'anomaly_mean' : float( anomalies[col].mean() ) if col in anomalies else 0,
                'normal_mean' : float( df[df[anomaly_col] == 0][col].mean() ) if col in df else 0
            }

        return analysis

    def generate_rule_based_explanation ( self, features: Dict[str, float] ) -> List[str] :
        """
        Generate rule-based explanation

        Args:
            features: Feature values

        Returns:
            List of explanation strings
        """
        explanations = []

        # Density rules
        if 'density' in features :
            if features['density'] > 100 :
                explanations.append( "Very high crowd density detected" )
            elif features['density'] < 10 :
                explanations.append( "Unusually low crowd density" )

        # Speed rules
        if 'speed_mean' in features :
            if features['speed_mean'] < 0.5 :
                explanations.append( "Crowd movement significantly slowed" )
            elif features['speed_mean'] > 2.5 :
                explanations.append( "Unusually fast crowd movement" )

        # Direction variance rules
        if 'direction_variance' in features :
            if features['direction_variance'] > 1.5 :
                explanations.append( "High directional conflict - crowd moving in multiple directions" )

        # Entry/exit imbalance
        if 'entry_exit_ratio' in features :
            if features['entry_exit_ratio'] > 3 :
                explanations.append( "Significant entry/exit imbalance - crowd accumulating" )
            elif features['entry_exit_ratio'] < 0.3 :
                explanations.append( "More people exiting than entering" )

        # Risk scores
        if 'footfall_risk' in features and features['footfall_risk'] > 0.7 :
            explanations.append( "Abnormal footfall pattern detected" )

        if 'flow_risk' in features and features['flow_risk'] > 0.7 :
            explanations.append( "Flow disruption detected" )

        if not explanations :
            explanations.append( "Normal crowd behavior - no significant anomalies" )

        return explanations