"""
Escalation Detection Module for S.A.F.E
Identifies patterns that indicate escalating risk situations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class EscalationDetector:
    """Detect escalating risk patterns in crowd data"""

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
                        'escalation': {
                            'persistence_window': 5,
                            'concurrent_anomalies_threshold': 3
                        }
                    }
                }

        self.persistence_window = self.config['risk_scoring']['escalation']['persistence_window']
        self.concurrent_threshold = self.config['risk_scoring']['escalation']['concurrent_anomalies_threshold']

    def detect_sustained_high_risk(self, df: pd.DataFrame,
                                   window: int = None) -> pd.DataFrame:
        """
        Detect sustained high-risk periods

        Args:
            df: DataFrame with risk scores
            window: Number of consecutive periods (uses config if None)

        Returns:
            DataFrame with sustained_risk flag
        """
        if window is None:
            window = self.persistence_window

        df = df.copy()
        df = df.sort_values(['zone_id', 'timestamp'])

        df['sustained_risk'] = 0

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone

            # Check for consecutive high risk
            high_risk = (df.loc[zone_mask, 'risk_level'] == 'High').astype(int)

            # Rolling sum to find sustained periods
            rolling_high = high_risk.rolling(window=window, min_periods=window).sum()
            df.loc[zone_mask, 'sustained_risk'] = (rolling_high >= window).astype(int)

        return df

    def detect_rapid_escalation(self, df: pd.DataFrame,
                               threshold_increase: float = 0.3) -> pd.DataFrame:
        """
        Detect rapid increase in risk scores

        Args:
            df: DataFrame with risk scores
            threshold_increase: Minimum increase to flag

        Returns:
            DataFrame with rapid_escalation flag
        """
        df = df.copy()
        df = df.sort_values(['zone_id', 'timestamp'])

        df['rapid_escalation'] = 0

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone

            # Calculate rate of change
            risk_change = df.loc[zone_mask, 'composite_risk_score'].diff()

            # Flag rapid increases
            df.loc[zone_mask, 'rapid_escalation'] = (
                risk_change > threshold_increase
            ).astype(int)

        return df

    def detect_spreading_risk(self, df: pd.DataFrame,
                             time_window: int = 3) -> pd.DataFrame:
        """
        Detect risk spreading across multiple zones

        Args:
            df: DataFrame with risk scores
            time_window: Number of recent time periods to check

        Returns:
            DataFrame with spreading_risk flag
        """
        df = df.copy()
        df = df.sort_values('timestamp')

        df['spreading_risk'] = 0

        # Get unique timestamps
        timestamps = df['timestamp'].unique()

        for i in range(len(timestamps)):
            current_time = timestamps[i]

            # Look at recent time window
            start_idx = max(0, i - time_window)
            recent_times = timestamps[start_idx:i + 1]

            # Count high-risk zones over time
            high_risk_counts = []
            for t in recent_times:
                time_data = df[df['timestamp'] == t]
                high_risk_count = (time_data['risk_level'] == 'High').sum()
                high_risk_counts.append(high_risk_count)

            # Check if increasing trend
            if len(high_risk_counts) >= 2:
                is_increasing = all(
                    high_risk_counts[i] <= high_risk_counts[i + 1]
                    for i in range(len(high_risk_counts) - 1)
                )

                if is_increasing and high_risk_counts[-1] >= self.concurrent_threshold:
                    df.loc[df['timestamp'] == current_time, 'spreading_risk'] = 1

        return df

    def detect_concurrent_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect multiple concurrent anomalies across zones

        Args:
            df: DataFrame with anomaly data

        Returns:
            DataFrame with concurrent_anomalies count
        """
        df = df.copy()

        # Count anomalies per timestamp
        anomaly_counts = df.groupby('timestamp')['risk_level'].apply(
            lambda x: (x == 'High').sum()
        ).reset_index()
        anomaly_counts.columns = ['timestamp', 'concurrent_anomalies']

        df = df.merge(anomaly_counts, on='timestamp', how='left')

        return df

    def calculate_escalation_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite escalation score

        Args:
            df: DataFrame with escalation indicators

        Returns:
            DataFrame with escalation_score
        """
        df = df.copy()

        # Detect all escalation patterns
        df = self.detect_sustained_high_risk(df)
        df = self.detect_rapid_escalation(df)
        df = self.detect_spreading_risk(df)
        df = self.detect_concurrent_anomalies(df)

        # Calculate composite score
        max_concurrent = df['concurrent_anomalies'].max()
        if max_concurrent > 0:
            concurrent_normalized = df['concurrent_anomalies'] / max_concurrent
        else:
            concurrent_normalized = 0

        df['escalation_score'] = (
            df['sustained_risk'] * 0.3 +
            df['rapid_escalation'] * 0.3 +
            df['spreading_risk'] * 0.2 +
            concurrent_normalized * 0.2
        )

        # Normalize to 0-1
        df['escalation_score'] = df['escalation_score'].clip(0, 1)

        return df

    def classify_escalation_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify escalation into levels

        Args:
            df: DataFrame with escalation_score

        Returns:
            DataFrame with escalation_level
        """
        df = df.copy()

        def get_escalation_level(score):
            if score < 0.3:
                return 'None'
            elif score < 0.6:
                return 'Moderate'
            else:
                return 'Severe'

        df['escalation_level'] = df['escalation_score'].apply(get_escalation_level)

        return df

    def get_escalation_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of escalation events

        Args:
            df: DataFrame with escalation data

        Returns:
            Dictionary with escalation statistics
        """
        summary = {
            'total_records': len(df),
            'sustained_risk_events': int(df['sustained_risk'].sum()) if 'sustained_risk' in df else 0,
            'rapid_escalation_events': int(df['rapid_escalation'].sum()) if 'rapid_escalation' in df else 0,
            'spreading_risk_events': int(df['spreading_risk'].sum()) if 'spreading_risk' in df else 0,
            'max_concurrent_anomalies': int(df['concurrent_anomalies'].max()) if 'concurrent_anomalies' in df else 0,
            'avg_escalation_score': float(df['escalation_score'].mean()) if 'escalation_score' in df else 0,
            'escalation_distribution': {}
        }

        if 'escalation_level' in df.columns:
            summary['escalation_distribution'] = df['escalation_level'].value_counts().to_dict()

        return summary

    def identify_critical_periods(self, df: pd.DataFrame,
                                 threshold: float = 0.7) -> pd.DataFrame:
        """
        Identify critical time periods requiring intervention

        Args:
            df: DataFrame with escalation scores
            threshold: Escalation score threshold

        Returns:
            DataFrame of critical periods
        """
        df = df.copy()

        critical = df[df['escalation_score'] >= threshold].copy()

        # Add urgency indicator
        critical['urgency'] = critical['escalation_score'].apply(
            lambda x: 'Immediate' if x >= 0.9 else 'High' if x >= 0.7 else 'Medium'
        )

        return critical[['timestamp', 'zone_id', 'escalation_score',
                        'escalation_level', 'urgency', 'concurrent_anomalies']]

    def detect_pre_escalation_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect early warning signs before escalation

        Args:
            df: DataFrame with crowd data

        Returns:
            DataFrame with pre_escalation flags
        """
        df = df.copy()
        df = df.sort_values(['zone_id', 'timestamp'])

        df['pre_escalation'] = 0

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone
            zone_data = df.loc[zone_mask].copy()

            # Look for gradual increase in density
            zone_data['density_trend'] = zone_data['density'].rolling(window=5).mean()
            density_increasing = zone_data['density_trend'].diff() > 0

            # Look for decreasing mobility
            speed_col = 'speed_mean' if 'speed_mean' in zone_data.columns else 'velocity_mean'
            if speed_col in zone_data.columns:
                speed_decreasing = zone_data[speed_col].diff() < 0
            else:
                speed_decreasing = False

            # Look for increasing direction variance
            if 'direction_variance' in zone_data.columns:
                direction_increasing = zone_data['direction_variance'].diff() > 0
            else:
                direction_increasing = False

            # Flag pre-escalation
            pre_escalation = (
                density_increasing &
                (speed_decreasing | direction_increasing)
            )

            df.loc[zone_mask, 'pre_escalation'] = pre_escalation.astype(int)

        return df