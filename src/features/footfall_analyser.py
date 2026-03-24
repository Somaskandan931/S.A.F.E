"""
Footfall Analysis Module for S.A.F.E
Specialized module for footfall pattern analysis and anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class FootfallAnalyzer:
    """Analyze footfall patterns and detect abnormal surges"""

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
                    'features': {
                        'footfall': {
                            'baseline_window': 50,
                            'anomaly_threshold': 2.5,
                            'surge_threshold': 1.5
                        }
                    }
                }

        self.baseline_window = self.config['features']['footfall']['baseline_window']
        self.anomaly_threshold = self.config['features']['footfall']['anomaly_threshold']
        self.surge_threshold = self.config['features']['footfall']['surge_threshold']

        self.baselines = {}  # Store baseline patterns per zone

    def calculate_baseline(self, df: pd.DataFrame,
                          zone_id: int = None) -> Dict:
        """
        Calculate baseline footfall patterns for a zone

        Args:
            df: Historical crowd data
            zone_id: Specific zone ID (None for all zones)

        Returns:
            Dictionary with baseline statistics
        """
        if zone_id is not None:
            df = df[df['zone_id'] == zone_id].copy()

        baselines = {}

        for zone in df['zone_id'].unique():
            zone_data = df[df['zone_id'] == zone]

            # Overall statistics
            baseline = {
                'mean_density': zone_data['density'].mean(),
                'std_density': zone_data['density'].std(),
                'median_density': zone_data['density'].median(),
                'max_density': zone_data['density'].quantile(0.95),  # 95th percentile
            }

            # Time-based baselines (hourly)
            if 'hour_of_day' in zone_data.columns:
                hourly_baseline = zone_data.groupby('hour_of_day')['density'].agg([
                    'mean', 'std', 'median'
                ]).to_dict('index')
                baseline['hourly_patterns'] = hourly_baseline
            else:
                baseline['hourly_patterns'] = {}

            # Day-based baselines
            if 'day_of_week' in zone_data.columns:
                daily_baseline = zone_data.groupby('day_of_week')['density'].agg([
                    'mean', 'std', 'median'
                ]).to_dict('index')
                baseline['daily_patterns'] = daily_baseline
            else:
                baseline['daily_patterns'] = {}

            # Peak hour baseline
            if 'is_peak_hour' in zone_data.columns:
                peak_data = zone_data[zone_data['is_peak_hour'] == 1]
                baseline['peak_hour_mean'] = peak_data['density'].mean()
                baseline['peak_hour_std'] = peak_data['density'].std()
            else:
                baseline['peak_hour_mean'] = baseline['mean_density']
                baseline['peak_hour_std'] = baseline['std_density']

            baselines[zone] = baseline

        self.baselines = baselines
        return baselines

    def detect_footfall_anomaly(self, current_density: float,
                               zone_id: int,
                               hour: int,
                               is_peak: bool) -> Tuple[float, str]:
        """
        Detect if current footfall is anomalous

        Args:
            current_density: Current crowd density
            zone_id: Zone identifier
            hour: Hour of day
            is_peak: Whether it's peak hour

        Returns:
            Tuple of (anomaly_score, anomaly_type)
        """
        if zone_id not in self.baselines:
            return 0.0, "no_baseline"

        baseline = self.baselines[zone_id]

        # Get appropriate baseline for comparison
        if hour in baseline.get('hourly_patterns', {}):
            expected_mean = baseline['hourly_patterns'][hour]['mean']
            expected_std = baseline['hourly_patterns'][hour]['std']
        else:
            expected_mean = baseline['mean_density']
            expected_std = baseline['std_density']

        # Calculate z-score
        if expected_std > 0:
            z_score = (current_density - expected_mean) / expected_std
        else:
            z_score = 0

        # Determine anomaly type and score
        anomaly_score = abs(z_score)

        if z_score > self.anomaly_threshold:
            if current_density > baseline['max_density'] * self.surge_threshold:
                anomaly_type = "severe_surge"
                anomaly_score *= 1.5
            else:
                anomaly_type = "moderate_surge"
        elif z_score < -self.anomaly_threshold:
            anomaly_type = "unusual_drop"
        else:
            anomaly_type = "normal"
            anomaly_score = 0.0

        return min(anomaly_score, 10.0), anomaly_type

    def analyze_footfall_trends(self, df: pd.DataFrame,
                               window: int = 10) -> pd.DataFrame:
        """
        Analyze footfall trends over time

        Args:
            df: DataFrame with crowd data
            window: Rolling window size

        Returns:
            DataFrame with trend analysis
        """
        df = df.copy()
        df = df.sort_values(['zone_id', 'timestamp'])

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone

            # Rolling statistics
            df.loc[zone_mask, 'footfall_rolling_mean'] = (
                df.loc[zone_mask, 'density'].rolling(window=window, min_periods=1).mean()
            )

            df.loc[zone_mask, 'footfall_rolling_std'] = (
                df.loc[zone_mask, 'density'].rolling(window=window, min_periods=1).std()
            )

            # Trend direction
            df.loc[zone_mask, 'footfall_trend'] = (
                df.loc[zone_mask, 'density'].diff().rolling(window=3, min_periods=1).mean()
            )

            # Acceleration (second derivative)
            df.loc[zone_mask, 'footfall_acceleration'] = (
                df.loc[zone_mask, 'footfall_trend'].diff()
            )

        df = df.fillna(0)
        return df

    def detect_rapid_accumulation(self, df: pd.DataFrame,
                                 threshold_rate: float = 0.5) -> pd.DataFrame:
        """
        Detect rapid crowd accumulation patterns

        Args:
            df: DataFrame with crowd data
            threshold_rate: Rate threshold for rapid accumulation

        Returns:
            DataFrame with accumulation flags
        """
        df = df.copy()
        df = df.sort_values(['zone_id', 'timestamp'])

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone

            # Calculate accumulation rate
            density_change = df.loc[zone_mask, 'density'].diff()
            time_diff = df.loc[zone_mask, 'timestamp'].diff().dt.total_seconds()

            df.loc[zone_mask, 'accumulation_rate'] = density_change / (time_diff + 1)

            # Flag rapid accumulation
            df.loc[zone_mask, 'rapid_accumulation'] = (
                df.loc[zone_mask, 'accumulation_rate'] > threshold_rate
            ).astype(int)

        df['accumulation_rate'] = df['accumulation_rate'].fillna(0)
        df['rapid_accumulation'] = df['rapid_accumulation'].fillna(0)

        return df

    def compute_footfall_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute footfall-specific risk score

        Args:
            df: DataFrame with footfall features

        Returns:
            DataFrame with footfall risk scores
        """
        df = df.copy()

        # Ensure baselines are calculated
        if not self.baselines:
            self.calculate_baseline(df)

        # Calculate anomaly scores
        anomaly_scores = []
        anomaly_types = []

        for _, row in df.iterrows():
            score, anom_type = self.detect_footfall_anomaly(
                current_density=row['density'],
                zone_id=row['zone_id'],
                hour=row.get('hour_of_day', 12),
                is_peak=row.get('is_peak_hour', 0)
            )
            anomaly_scores.append(score)
            anomaly_types.append(anom_type)

        df['footfall_anomaly_score'] = anomaly_scores
        df['footfall_anomaly_type'] = anomaly_types

        # Normalize to 0-1 range
        max_score = df['footfall_anomaly_score'].max()
        if max_score > 0:
            df['footfall_risk'] = df['footfall_anomaly_score'] / max_score
        else:
            df['footfall_risk'] = 0

        return df

    def get_zone_summary(self, zone_id: int) -> Dict:
        """Get summary statistics for a specific zone"""
        if zone_id in self.baselines:
            return self.baselines[zone_id]
        else:
            return {"error": "No baseline calculated for this zone"}