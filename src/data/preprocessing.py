"""
Data Preprocessing Module for S.A.F.E - ROBUST VERSION
Handles different column sets between train and test datasets
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """Preprocess crowd data for modeling with robust column handling"""

    def __init__(self, config: dict = None):
        """Initialize preprocessor with configuration"""
        try:
            from src.config.default_config import DEFAULT_CONFIG
            self.config = config if config is not None else DEFAULT_CONFIG
        except ImportError:
            self.config = config if config is not None else {}

        self.scaler = None
        self.feature_ranges = {}
        self.train_columns = None  # Store columns seen during training

    def handle_missing_values(self, df: pd.DataFrame,
                              strategy: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing values in dataset"""
        df = df.copy()

        if strategy == 'forward_fill':
            df = df.ffill()
        elif strategy == 'backward_fill':
            df = df.bfill()
        elif strategy == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        elif strategy == 'drop':
            df = df.dropna()

        # Fill remaining NaNs with 0
        df = df.fillna(0)

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional temporal features"""
        df = df.copy()

        if 'timestamp' not in df.columns:
            return df

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract temporal components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

        return df

    def harmonize_columns(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Harmonize column names and handle missing columns

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (stores column list)

        Returns:
            DataFrame with harmonized columns
        """
        df = df.copy()

        if is_training:
            # Store columns for later
            self.train_columns = df.columns.tolist()
            return df

        # Transform phase: ensure all training columns exist
        if self.train_columns is not None:
            for col in self.train_columns:
                if col not in df.columns:
                    # Add missing column with zeros
                    df[col] = 0.0
                    print(f"⚠️ Added missing column '{col}' with zeros")

            # Keep only training columns (remove any extra columns)
            df = df[self.train_columns]

        return df

    def normalize_features(self, df: pd.DataFrame,
                          columns: List[str] = None,
                          method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features"""
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude certain columns
            exclude = ['zone_id', 'timestamp', 'is_anomaly', 'hour_of_day', 'day_of_week']
            columns = [c for c in columns if c not in exclude]

        if len(columns) == 0:
            return df

        # Only normalize columns that exist
        existing_cols = [c for c in columns if c in df.columns]

        if len(existing_cols) == 0:
            print("⚠️ No columns to normalize")
            return df

        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                df[existing_cols] = self.scaler.fit_transform(df[existing_cols])
            else:
                # Transform - only use columns that scaler knows about
                scaler_cols = [c for c in existing_cols if c in self.scaler.feature_names_in_]
                if len(scaler_cols) > 0:
                    df[scaler_cols] = self.scaler.transform(df[scaler_cols])
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                df[existing_cols] = self.scaler.fit_transform(df[existing_cols])
            else:
                scaler_cols = [c for c in existing_cols if c in self.scaler.feature_names_in_]
                if len(scaler_cols) > 0:
                    df[scaler_cols] = self.scaler.transform(df[scaler_cols])

        return df

    def add_anomaly_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weakly-supervised anomaly labeling"""
        df = df.copy()

        # Only add if density exists
        if 'density' not in df.columns:
            df['is_anomaly'] = 0
            return df

        # Density anomaly
        if df['density'].std() > 0:
            density_z = (df["density"] - df["density"].mean()) / (df["density"].std() + 1e-6)
        else:
            density_z = 0

        # Velocity anomaly (if available)
        if 'velocity_mean' in df.columns and df['velocity_mean'].std() > 0:
            velocity_z = (df["velocity_mean"] - df["velocity_mean"].mean()) / (df["velocity_mean"].std() + 1e-6)
        else:
            velocity_z = 0

        df["is_anomaly"] = (
            (np.abs(density_z) > 3.0) |
            (np.abs(velocity_z) > 3.0)
        ).astype(int)

        print(f"✓ Anomaly labels added: {df['is_anomaly'].sum()} / {len(df)}")

        return df

    def select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric features for ML models - PRESERVES zone_id"""
        df = df.copy()

        # Save zone_id BEFORE dropping non-numeric columns
        zone_id_preserved = None
        if 'zone_id' in df.columns:
            zone_id_preserved = df['zone_id'].copy()

        # Columns to explicitly drop
        drop_cols = ["dataset", "timestamp"]

        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # RESTORE zone_id if it was present
        if zone_id_preserved is not None:
            # Convert zone_id to numeric if it's a string
            if zone_id_preserved.dtype == 'object':
                zone_mapping = {zone: idx for idx, zone in enumerate(zone_id_preserved.unique())}
                df['zone_id'] = zone_id_preserved.map(zone_mapping)
            else:
                df['zone_id'] = zone_id_preserved

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data"""
        print("Starting preprocessing pipeline...")

        # 1. Handle missing values
        df = self.handle_missing_values(df, strategy="interpolate")
        print(f"✓ Handled missing values")

        # 2. Create temporal features
        df = self.create_temporal_features(df)
        print(f"✓ Created temporal features")

        # 3. Harmonize columns (store training columns)
        df = self.harmonize_columns(df, is_training=True)
        print(f"✓ Stored training columns: {len(self.train_columns)}")

        # 4. Normalize features
        df = self.normalize_features(df, method='standard')
        print(f"✓ Normalized features")

        # 5. Add anomaly labels
        df = self.add_anomaly_labels(df)

        # 6. Select numeric features (PRESERVES zone_id)
        df = self.select_numeric_features(df)

        print(f"Preprocessing complete: {len(df)} records")

        # Verify zone_id is preserved
        if 'zone_id' in df.columns:
            print(f"✓ zone_id preserved: {df['zone_id'].nunique()} unique zones")
        else:
            print("⚠️ WARNING: zone_id was lost!")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform validation/test data - ROBUST VERSION"""
        df = df.copy()

        print(f"\nTransforming test data...")
        print(f"  Input columns: {len(df.columns)}")
        print(f"  Input records: {len(df)}")

        # 1. Handle missing values
        df = self.handle_missing_values(df, strategy="interpolate")

        # 2. Create temporal features
        df = self.create_temporal_features(df)

        # 3. Harmonize columns (add missing, remove extra)
        df = self.harmonize_columns(df, is_training=False)
        print(f"  After harmonization: {len(df.columns)} columns")

        # 4. Normalize using TRAIN scaler
        if self.scaler is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = ['zone_id', 'timestamp', 'is_anomaly']
            numeric_cols = [c for c in numeric_cols if c not in exclude and c in df.columns]

            # Only normalize columns that scaler knows about
            if hasattr(self.scaler, 'feature_names_in_'):
                scaler_cols = [c for c in numeric_cols if c in self.scaler.feature_names_in_]
                if len(scaler_cols) > 0:
                    print(f"  Normalizing {len(scaler_cols)} columns")
                    df[scaler_cols] = self.scaler.transform(df[scaler_cols])
                else:
                    print(f"  ⚠️ No common columns to normalize")

        # 5. Select numeric features (PRESERVES zone_id)
        df = self.select_numeric_features(df)

        print(f"  Final columns: {len(df.columns)}")
        print(f"  zone_id preserved: {'zone_id' in df.columns}")

        return df