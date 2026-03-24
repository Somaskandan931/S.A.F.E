"""
Feature Engineering Module for S.A.F.E
Extracts meaningful features from crowd signals
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class FeatureEngineer:
    """Extract and engineer features from crowd data"""

    def __init__(self, config_path: str = None):
        # Import config loader
        try:
            from src.utils.config_loader import load_config
            self.config = load_config(config_path)
        except ImportError:
            # Fallback if utils not available
            try:
                from src.config.default_config import DEFAULT_CONFIG
                self.config = DEFAULT_CONFIG.copy()
            except ImportError:
                # Minimal fallback
                self.config = {
                    'features': {
                        'temporal_features': True,
                        'crowd_features': True
                    }
                }

        self.temporal_features = self.config.get('features', {}).get('temporal_features', True)
        self.crowd_features = self.config.get('features', {}).get('crowd_features', True)

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal context features

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with temporal features
        """
        df = df.copy()

        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Hour of day
        if 'hour_of_day' not in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour

        # Day of week (0=Monday, 6=Sunday)
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Is peak hour (8-10 AM, 5-7 PM)
        if 'is_peak_hour' not in df.columns:
            df['is_peak_hour'] = df['hour_of_day'].apply(
                lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 19) else 0
            )

        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Time of day categories
        df['time_category'] = pd.cut(df['hour_of_day'],
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['night', 'morning', 'afternoon', 'evening'],
                                     include_lowest=True)

        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def calculate_footfall_change(self, df: pd.DataFrame,
                                  window: int = 5) -> pd.DataFrame:
        """
        Calculate rate of change in footfall/density

        Args:
            df: DataFrame with density column
            window: Number of time steps for calculation

        Returns:
            DataFrame with footfall change features
        """
        df = df.copy()

        # Sort by zone and time
        df = df.sort_values(['zone_id', 'timestamp'])

        # Calculate rolling statistics per zone
        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone

            # Density change rate
            df.loc[zone_mask, 'density_change'] = df.loc[zone_mask, 'density'].diff()
            df.loc[zone_mask, 'density_change_rate'] = (
                df.loc[zone_mask, 'density'].pct_change()
            )

            # Rolling mean and std
            df.loc[zone_mask, 'density_rolling_mean'] = (
                df.loc[zone_mask, 'density'].rolling(window=window, min_periods=1).mean()
            )
            df.loc[zone_mask, 'density_rolling_std'] = (
                df.loc[zone_mask, 'density'].rolling(window=window, min_periods=1).std()
            )

            # Z-score relative to recent history
            df.loc[zone_mask, 'density_zscore'] = (
                (df.loc[zone_mask, 'density'] - df.loc[zone_mask, 'density_rolling_mean']) /
                (df.loc[zone_mask, 'density_rolling_std'] + 1e-6)
            )

        # Fill NaN values (fix pandas FutureWarning)
        df['density_change'] = df['density_change'].fillna(0)
        df['density_change_rate'] = df['density_change_rate'].fillna(0)
        df['density_zscore'] = df['density_zscore'].fillna(0)

        return df

    def calculate_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flow-based features

        Args:
            df: DataFrame with speed and direction data

        Returns:
            DataFrame with flow features
        """
        df = df.copy()

        # Check if required columns exist
        if 'direction_variance' in df.columns:
            # Flow efficiency (inverse of direction variance)
            df['flow_efficiency'] = 1 / (df['direction_variance'] + 1e-6)

        # Check for speed variance or create it
        if 'velocity_std' in df.columns and 'speed_variance' not in df.columns:
            df['speed_variance'] = df['velocity_std'] ** 2

        if 'speed_variance' in df.columns:
            # Speed consistency (inverse of speed variance)
            df['speed_consistency'] = 1 / (df['speed_variance'] + 1e-6)

        # Speed mean handling
        if 'velocity_mean' in df.columns and 'speed_mean' not in df.columns:
            df['speed_mean'] = df['velocity_mean']

        if 'speed_mean' in df.columns and 'direction_variance' in df.columns:
            # Flow disruption score (high when speed is low and variance is high)
            max_speed = df['speed_mean'].max()
            if max_speed > 0:
                df['flow_disruption'] = (
                    (1 - df['speed_mean'] / max_speed) *
                    df['direction_variance']
                )
            else:
                df['flow_disruption'] = 0

        if 'speed_mean' in df.columns and 'flow_efficiency' in df.columns:
            # Crowd mobility index
            df['mobility_index'] = df['speed_mean'] * df['flow_efficiency']

        return df

    def calculate_zone_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features based on neighboring zones

        Args:
            df: DataFrame with zone_id

        Returns:
            DataFrame with zone interaction features
        """
        df = df.copy()

        # For each timestamp, calculate zone statistics
        for timestamp in df['timestamp'].unique():
            time_mask = df['timestamp'] == timestamp

            # Average density across all zones
            avg_density = df.loc[time_mask, 'density'].mean()
            df.loc[time_mask, 'zone_density_ratio'] = (
                df.loc[time_mask, 'density'] / (avg_density + 1e-6)
            )

            # Density variance across zones
            density_variance = df.loc[time_mask, 'density'].var()
            df.loc[time_mask, 'cross_zone_density_variance'] = density_variance

        return df

    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature matrix for modeling

        Args:
            df: Raw crowd data

        Returns:
            DataFrame with all engineered features
        """
        # Extract temporal features
        df = self.extract_temporal_features(df)

        # Calculate footfall changes
        df = self.calculate_footfall_change(df)

        # Calculate flow features
        df = self.calculate_flow_features(df)

        # Calculate zone interactions
        df = self.calculate_zone_interactions(df)

        return df

    def get_feature_names(self, include_temporal: bool = True,
                         include_raw: bool = True) -> List[str]:
        """
        Get list of feature names for modeling

        Args:
            include_temporal: Include temporal features
            include_raw: Include raw crowd features

        Returns:
            List of feature names
        """
        features = []

        if include_raw:
            features.extend([
                'density', 'speed_mean', 'speed_variance',
                'direction_variance', 'velocity_mean', 'velocity_std'
            ])

        # Engineered features
        features.extend([
            'density_change', 'density_change_rate', 'density_zscore',
            'flow_efficiency', 'speed_consistency', 'flow_disruption',
            'mobility_index', 'zone_density_ratio', 'cross_zone_density_variance'
        ])

        if include_temporal:
            features.extend([
                'hour_of_day', 'day_of_week', 'is_peak_hour', 'is_weekend',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ])

        return features