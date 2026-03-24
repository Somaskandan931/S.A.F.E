"""
Data Preprocessing Module for S.A.F.E - FIXED (Research-Grade)
=================================================================
FIXES APPLIED:
  1. REMOVED add_anomaly_labels() from training pipeline (was causing label leakage)
  2. Scaler is fit ONLY on training data, transform applied to test/val (no leakage)
  3. Anomaly injection is now a separate, explicit method only used for evaluation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """Preprocess crowd data for modeling with zero label/scaling leakage."""

    def __init__(self, config: dict = None):
        try:
            from src.config.default_config import DEFAULT_CONFIG
            self.config = config if config is not None else DEFAULT_CONFIG
        except ImportError:
            self.config = config if config is not None else {}

        self.scaler = None
        self.feature_ranges = {}
        self.train_columns = None

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------
    def handle_missing_values(self, df: pd.DataFrame,
                              strategy: str = 'forward_fill') -> pd.DataFrame:
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
        df = df.fillna(0)
        return df

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'timestamp' not in df.columns:
            return df
        df['timestamp'] = pd.to_datetime(df['timestamp'])
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

    # ------------------------------------------------------------------
    # Column harmonisation
    # ------------------------------------------------------------------
    def harmonize_columns(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        df = df.copy()
        if is_training:
            self.train_columns = df.columns.tolist()
            return df
        if self.train_columns is not None:
            for col in self.train_columns:
                if col not in df.columns:
                    df[col] = 0.0
                    print(f"  Added missing column '{col}' with zeros")
            df = df[self.train_columns]
        return df

    # ------------------------------------------------------------------
    # Normalisation — CRITICAL: fit only on training, transform on test
    # ------------------------------------------------------------------
    def normalize_features(self, df: pd.DataFrame,
                           columns: List[str] = None,
                           method: str = 'standard') -> pd.DataFrame:
        """
        Normalise features.

        Rules that prevent data leakage:
          - If self.scaler is None  → fit on this data  (training phase only)
          - If self.scaler exists   → transform only    (val/test phase)

        The pipeline MUST call fit_transform() for training data and
        transform() for validation/test data, so these paths are triggered
        correctly.
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = ['zone_id', 'timestamp', 'is_anomaly', 'hour_of_day', 'day_of_week']
            columns = [c for c in columns if c not in exclude]

        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            print("  No columns to normalise")
            return df

        if method == 'standard':
            if self.scaler is None:
                # TRAINING: fit + transform
                self.scaler = StandardScaler()
                df[existing_cols] = self.scaler.fit_transform(df[existing_cols])
            else:
                # VALIDATION / TEST: transform only — no fit
                if hasattr(self.scaler, 'feature_names_in_'):
                    scaler_cols = [c for c in existing_cols
                                   if c in self.scaler.feature_names_in_]
                else:
                    scaler_cols = existing_cols
                if scaler_cols:
                    df[scaler_cols] = self.scaler.transform(df[scaler_cols])
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                df[existing_cols] = self.scaler.fit_transform(df[existing_cols])
            else:
                if hasattr(self.scaler, 'feature_names_in_'):
                    scaler_cols = [c for c in existing_cols
                                   if c in self.scaler.feature_names_in_]
                else:
                    scaler_cols = existing_cols
                if scaler_cols:
                    df[scaler_cols] = self.scaler.transform(df[scaler_cols])
        return df

    # ------------------------------------------------------------------
    # Anomaly injection (for evaluation only — NOT used in training)
    # ------------------------------------------------------------------
    def inject_anomalies(self, df: pd.DataFrame,
                         anomaly_ratio: float = 0.05,
                         random_state: int = 42) -> pd.DataFrame:
        """
        Inject realistic anomalies into a DataFrame for evaluation purposes.

        IMPORTANT: This method must NEVER be called before or during model
        training.  It exists solely to create ground-truth labels for
        evaluation on datasets that do not ship with anomaly annotations.

        Injected anomaly types:
          1. Sudden velocity spike  — crowd suddenly accelerates
          2. Direction chaos        — abrupt heading reversal
          3. Density surge          — abnormal crowd compression

        Features used for injection are DIFFERENT from statistical summaries,
        so the model cannot exploit the labelling rule.
        """
        df = df.copy()
        rng = np.random.default_rng(random_state)
        n = len(df)
        n_anomaly = max(1, int(n * anomaly_ratio))

        df['is_anomaly'] = 0

        # --- 1. Velocity spike (sudden acceleration) ---
        if 'velocity_mean' in df.columns:
            idx = rng.choice(n, size=n_anomaly // 3, replace=False)
            df.loc[idx, 'velocity_mean'] *= rng.uniform(2.5, 5.0, size=len(idx))
            df.loc[idx, 'is_anomaly'] = 1
            print(f"  Injected {len(idx)} velocity-spike anomalies")

        # --- 2. Direction reversal (chaos) ---
        if 'direction_variance' in df.columns:
            available = df[df['is_anomaly'] == 0].index.tolist()
            if available:
                idx = rng.choice(available, size=min(n_anomaly // 3, len(available)), replace=False)
                df.loc[idx, 'direction_variance'] += rng.uniform(1.5, 3.0, size=len(idx))
                df.loc[idx, 'is_anomaly'] = 1
                print(f"  Injected {len(idx)} direction-chaos anomalies")

        # --- 3. Density surge ---
        if 'density' in df.columns:
            available = df[df['is_anomaly'] == 0].index.tolist()
            if available:
                idx = rng.choice(available, size=min(n_anomaly // 3, len(available)), replace=False)
                df.loc[idx, 'density'] *= rng.uniform(3.0, 6.0, size=len(idx))
                df.loc[idx, 'is_anomaly'] = 1
                print(f"  Injected {len(idx)} density-surge anomalies")

        total = int(df['is_anomaly'].sum())
        print(f"  Total anomalies injected: {total} / {n} ({total/n*100:.1f}%)")
        return df

    # ------------------------------------------------------------------
    # Feature selection helper
    # ------------------------------------------------------------------
    def select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric features; preserve zone_id."""
        df = df.copy()

        zone_id_preserved = None
        if 'zone_id' in df.columns:
            zone_id_preserved = df['zone_id'].copy()

        for col in ['dataset', 'timestamp']:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.select_dtypes(include=[np.number])

        if zone_id_preserved is not None:
            if zone_id_preserved.dtype == 'object':
                mapping = {z: i for i, z in enumerate(zone_id_preserved.unique())}
                df['zone_id'] = zone_id_preserved.map(mapping)
            else:
                df['zone_id'] = zone_id_preserved

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing on TRAINING data and transform it.

        Steps:
          1. Handle missing values
          2. Create temporal features
          3. Store training column list
          4. Fit scaler and normalise (training data only)
          5. Select numeric features

        NOTE: add_anomaly_labels() is intentionally absent from this
        pipeline.  Models are trained on normal data only (unsupervised).
        Anomaly labels are injected separately at evaluation time via
        inject_anomalies().
        """
        print("Starting training preprocessing pipeline...")

        df = self.handle_missing_values(df, strategy="interpolate")
        print("  Handled missing values")

        df = self.create_temporal_features(df)
        print("  Created temporal features")

        df = self.harmonize_columns(df, is_training=True)
        print(f"  Stored {len(self.train_columns)} training columns")

        # FIT scaler on training data only
        df = self.normalize_features(df, method='standard')
        print("  Fitted scaler and normalised training features")

        df = self.select_numeric_features(df)
        print(f"  Preprocessing complete: {len(df)} records, {len(df.columns)} features")

        if 'zone_id' in df.columns:
            print(f"  zone_id preserved: {df['zone_id'].nunique()} unique zones")
        else:
            print("  WARNING: zone_id was lost during preprocessing")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform VALIDATION / TEST data using the already-fitted scaler.

        The scaler is NEVER re-fitted here — only transform() is called.
        This prevents any test data from influencing the normalisation.
        """
        if self.scaler is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_transform() on training data first."
            )

        df = df.copy()
        print(f"\nTransforming data ({len(df)} records)...")

        df = self.handle_missing_values(df, strategy="interpolate")
        df = self.create_temporal_features(df)
        df = self.harmonize_columns(df, is_training=False)
        print(f"  After harmonisation: {len(df.columns)} columns")

        # TRANSFORM only — never fit on test data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['zone_id', 'timestamp', 'is_anomaly']
        numeric_cols = [c for c in numeric_cols if c not in exclude and c in df.columns]

        if hasattr(self.scaler, 'feature_names_in_'):
            scaler_cols = [c for c in numeric_cols
                           if c in self.scaler.feature_names_in_]
        else:
            scaler_cols = numeric_cols

        if scaler_cols:
            print(f"  Transforming {len(scaler_cols)} columns with training scaler")
            df[scaler_cols] = self.scaler.transform(df[scaler_cols])
        else:
            print("  WARNING: No common columns to normalise")

        df = self.select_numeric_features(df)
        print(f"  Final: {len(df.columns)} columns, zone_id present: {'zone_id' in df.columns}")

        return df