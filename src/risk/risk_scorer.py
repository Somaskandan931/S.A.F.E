"""
Risk Scoring Module for S.A.F.E
Computes comprehensive risk scores based on multiple signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class RiskScorer:
    """Calculate risk scores for crowd situations"""

    def __init__(self, config_path: str = None):
        # Import config loader
        try:
            from src.utils.config_loader import load_config
            self.config = load_config(config_path)
        except ImportError:
            try:
                from src.config.default_config import DEFAULT_CONFIG
                self.config = DEFAULT_CONFIG.copy()
            except ImportError:
                # Minimal fallback
                self.config = {
                    'risk_scoring': {
                        'weights': {
                            'footfall_anomaly': 0.35,
                            'flow_disruption': 0.35,
                            'temporal_abnormality': 0.30
                        },
                        'thresholds': {
                            'low': 0.3,
                            'medium': 0.6,
                            'high': 0.8
                        }
                    }
                }

        self.weights = self.config['risk_scoring']['weights']
        self.thresholds = self.config['risk_scoring']['thresholds']

    def calculate_footfall_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate footfall-based risk component

        Args:
            df: DataFrame with footfall features

        Returns:
            Series with footfall risk scores
        """
        risk_scores = pd.Series(0.0, index=df.index)

        # Density anomaly
        if 'density_zscore' in df.columns:
            density_risk = np.clip(np.abs(df['density_zscore']) / 5.0, 0, 1)
            risk_scores += density_risk * 0.4

        # Rapid change
        if 'density_change_rate' in df.columns:
            change_risk = np.clip(np.abs(df['density_change_rate']) * 2, 0, 1)
            risk_scores += change_risk * 0.3

        # Footfall anomaly score if available
        if 'footfall_anomaly_score' in df.columns:
            footfall_risk = np.clip(df['footfall_anomaly_score'] / 5.0, 0, 1)
            risk_scores += footfall_risk * 0.3

        return np.clip(risk_scores, 0, 1)

    def calculate_flow_disruption_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate flow disruption risk component

        Args:
            df: DataFrame with flow features

        Returns:
            Series with flow disruption risk scores
        """
        risk_scores = pd.Series(0.0, index=df.index)

        # Speed reduction (handle both column names)
        speed_col = 'speed_mean' if 'speed_mean' in df.columns else 'velocity_mean'
        if speed_col in df.columns:
            speed_risk = 1.0 - np.clip(df[speed_col] / 2.0, 0, 1)
            risk_scores += speed_risk * 0.3

        # Directional conflict
        if 'direction_variance' in df.columns:
            direction_risk = np.clip(df['direction_variance'] / 2.0, 0, 1)
            risk_scores += direction_risk * 0.3

        # Flow disruption score if available
        if 'flow_disruption' in df.columns:
            disruption_risk = np.clip(df['flow_disruption'] / 2.0, 0, 1)
            risk_scores += disruption_risk * 0.4

        return np.clip(risk_scores, 0, 1)

    def calculate_temporal_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate temporal abnormality risk

        Args:
            df: DataFrame with temporal features

        Returns:
            Series with temporal risk scores
        """
        risk_scores = pd.Series(0.0, index=df.index)

        # High density during non-peak hours is more concerning
        if 'is_peak_hour' in df.columns and 'density' in df.columns:
            non_peak_density = df['density'] * (1 - df['is_peak_hour'])
            max_density = df['density'].max()
            if max_density > 0:
                risk_scores += np.clip(non_peak_density / max_density, 0, 1) * 0.5

        # Weekend unusual activity
        if 'is_weekend' in df.columns and 'density' in df.columns:
            weekend_density = df['density'] * df['is_weekend']
            max_density = df['density'].max()
            if max_density > 0:
                risk_scores += np.clip(weekend_density / max_density * 0.5, 0, 1) * 0.3

        # Late night activity
        if 'hour_of_day' in df.columns and 'density' in df.columns:
            late_night = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
            late_night_density = df['density'] * late_night
            max_density = df['density'].max()
            if max_density > 0:
                risk_scores += np.clip(late_night_density / max_density, 0, 1) * 0.2

        return np.clip(risk_scores, 0, 1)

    def calculate_composite_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite risk score from all components

        Args:
            df: DataFrame with all features

        Returns:
            DataFrame with risk scores added
        """
        df = df.copy()

        # Calculate individual risk components
        footfall_risk = self.calculate_footfall_risk(df)
        flow_risk = self.calculate_flow_disruption_risk(df)
        temporal_risk = self.calculate_temporal_risk(df)

        # Composite risk with weights
        df['footfall_risk'] = footfall_risk
        df['flow_risk'] = flow_risk
        df['temporal_risk'] = temporal_risk

        df['composite_risk_score'] = (
            footfall_risk * self.weights['footfall_anomaly'] +
            flow_risk * self.weights['flow_disruption'] +
            temporal_risk * self.weights['temporal_abnormality']
        )

        # Ensure risk is in [0, 1]
        df['composite_risk_score'] = np.clip(df['composite_risk_score'], 0, 1)

        return df

    def assign_risk_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign categorical risk levels

        Args:
            df: DataFrame with risk scores

        Returns:
            DataFrame with risk level categories
        """
        df = df.copy()

        def get_risk_level(score):
            if score < self.thresholds['low']:
                return 'Low'
            elif score < self.thresholds['medium']:
                return 'Low'
            elif score < self.thresholds['high']:
                return 'Medium'
            else:
                return 'High'

        df['risk_level'] = df['composite_risk_score'].apply(get_risk_level)

        return df

    def get_risk_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of risk scores

        Args:
            df: DataFrame with risk scores

        Returns:
            Dictionary with risk statistics
        """
        summary = {
            'mean_risk': df['composite_risk_score'].mean(),
            'max_risk': df['composite_risk_score'].max(),
            'std_risk': df['composite_risk_score'].std(),
            'risk_distribution': df['risk_level'].value_counts().to_dict(),
            'high_risk_zones': [],
            'high_risk_times': []
        }

        # Identify high-risk zones
        if 'zone_id' in df.columns:
            high_risk_df = df[df['risk_level'] == 'High']
            if not high_risk_df.empty:
                zone_risk = high_risk_df.groupby('zone_id')['composite_risk_score'].agg([
                    'mean', 'max', 'count'
                ])
                summary['high_risk_zones'] = zone_risk.to_dict('index')

        # Identify high-risk time periods
        if 'timestamp' in df.columns:
            high_risk_df = df[df['risk_level'] == 'High']
            if not high_risk_df.empty:
                summary['high_risk_times'] = high_risk_df['timestamp'].tolist()

        return summary

    def detect_persistent_risk(self, df: pd.DataFrame,
                              window: int = 3) -> pd.DataFrame:
        """
        Detect persistent high-risk conditions

        Args:
            df: DataFrame with risk scores
            window: Number of consecutive time periods

        Returns:
            DataFrame with persistent risk flags
        """
        df = df.copy()
        df = df.sort_values(['zone_id', 'timestamp'])

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone

            # Check for consecutive high risk periods
            high_risk = (df.loc[zone_mask, 'risk_level'] == 'High').astype(int)

            # Rolling sum to find persistent risk
            persistent = high_risk.rolling(window=window, min_periods=window).sum()
            df.loc[zone_mask, 'persistent_risk'] = (persistent >= window).astype(int)

        df['persistent_risk'] = df['persistent_risk'].fillna(0)

        return df