"""
S.A.F.E Pipeline with Fixed Plotting - COMPLETE VERSION
Includes adaptive thresholding AND visualization generation
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from src import models
from src.evaluation.metrics import MetricsCalculator


class SAFEPipeline:
    """Complete training and evaluation pipeline with plots"""

    def __init__(self, data_loader, preprocessing, train_scenes, validation_scenes,
                 test_dataset, mall_path, model_output_dir, results_output_dir, plot_output_dir):
        """Initialize S.A.F.E Pipeline"""
        self.data_loader = data_loader
        self.preprocessing = preprocessing
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.test_dataset = test_dataset
        self.mall_path = Path(mall_path)

        self.model_output_dir = Path(model_output_dir)
        self.results_output_dir = Path(results_output_dir)
        self.plot_output_dir = Path(plot_output_dir)

        # Create output directories
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.results_output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator()

        # Feature columns
        self.BASE_FEATURES = [
            'time_window', 'footfall_count', 'velocity_mean', 'velocity_std',
            'direction_variance', 'zone_center_x', 'zone_center_y', 'frame_id', 'density'
        ]

        self.TEMPORAL_FEATURES = [
            'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'quarter',
            'is_weekend', 'is_business_hours'
        ]

        self.FEATURE_COLUMNS = self.BASE_FEATURES + self.TEMPORAL_FEATURES

    def _get_available_features(self, df: pd.DataFrame) -> list:
        """Get list of available features from DataFrame"""
        exclude_cols = ['dataset', 'timestamp', 'is_anomaly', 'zone_id', 'pedestrian_id']

        available = []
        for col in self.FEATURE_COLUMNS:
            if col in df.columns and col not in exclude_cols:
                available.append(col)

        # Also include any other numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in available and col not in exclude_cols:
                available.append(col)

        print(f"\n📊 Available features: {len(available)}")
        print(f"   Features: {available}")

        return available

    def calculate_adaptive_threshold(self, scores: np.ndarray,
                                    method: str = 'percentile',
                                    percentile: float = 90) -> float:
        """Calculate adaptive threshold based on score distribution"""
        if method == 'percentile':
            threshold = np.percentile(scores, percentile)
        elif method == 'iqr':
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        elif method == 'zscore':
            threshold = scores.mean() + 2 * scores.std()
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return float(threshold)

    def evaluate_with_plots(self, models, df, tag="validation", use_adaptive_threshold=True):
        """Evaluate models with adaptive thresholding AND generate plots"""
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION WITH PLOTS - {tag.upper()}")
        if use_adaptive_threshold:
            print("(Using Adaptive Thresholding)")
        print(f"{'='*60}")

        # Get available features
        available_features = self._get_available_features(df)

        if len(available_features) == 0:
            print("⚠️ No valid features found for evaluation")
            return {}

        # Check if labels exist
        has_labels = "is_anomaly" in df.columns
        if has_labels:
            y_true = df["is_anomaly"].values
        else:
            # Create synthetic labels based on risk thresholds
            y_true = np.zeros(len(df))
            print("⚠️ No labels found - creating synthetic labels for visualization")

        X = df[available_features]
        results = {}

        for name, model in models.items():
            if name == "lstm_autoencoder":
                continue

            print(f"\n📊 Evaluating {name}...")

            try:
                scores = model.predict_score(X)

                # Calculate adaptive threshold
                if use_adaptive_threshold:
                    threshold = self.calculate_adaptive_threshold(scores, method='percentile', percentile=90)
                    predictions = (scores > threshold).astype(int)
                    print(f"   Adaptive threshold: {threshold:.4f}")
                else:
                    predictions = model.predict(X)

                # Generate plots even without true labels
                if not has_labels:
                    # Create pseudo-labels for visualization
                    y_true = (scores > np.percentile(scores, 90)).astype(int)

                # Create evaluation report with ALL plots
                plot_dir = self.plot_output_dir / tag
                plot_dir.mkdir(parents=True, exist_ok=True)

                print(f"   📊 Generating plots in {plot_dir}...")
                metrics = self.metrics_calc.create_evaluation_report(
                    y_true=y_true,
                    y_pred=predictions,
                    y_scores=scores,
                    model_name=f"{name}_{tag}",
                    save_dir=str(plot_dir)
                )

                if has_labels:
                    from sklearn.metrics import precision_score, recall_score, f1_score

                    precision = precision_score(y_true, predictions, zero_division=0)
                    recall = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)

                    print(f"   Precision: {precision:.4f}")
                    print(f"   Recall:    {recall:.4f}")
                    print(f"   F1 Score:  {f1:.4f}")

                    metrics.update({
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "threshold": threshold if use_adaptive_threshold else "model_default"
                    })
                else:
                    anomaly_rate = float(predictions.mean())
                    metrics.update({
                        "anomaly_rate": anomaly_rate,
                        "threshold": threshold if use_adaptive_threshold else "model_default"
                    })
                    print(f"   Anomaly Rate: {anomaly_rate:.2%}")

                results[name] = metrics

            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save results
        if results:
            results_df = pd.DataFrame(results).T
            results_path = self.results_output_dir / f"{tag}_results_with_plots.csv"
            results_df.to_csv(results_path)
            print(f"\n💾 Results saved to {results_path}")
            print(f"📊 Plots saved to {self.plot_output_dir / tag}")

        return results

    def evaluate_lstm_with_plots(self, lstm_model, df, tag="validation", use_adaptive_threshold=True):
        """Evaluate LSTM model with plots"""
        print(f"\n{'='*60}")
        print(f"LSTM EVALUATION WITH PLOTS - {tag.upper()}")
        if use_adaptive_threshold:
            print("(Using Adaptive Thresholding)")
        print(f"{'='*60}")

        try:
            from src.data.sequence_builder import SequenceBuilder

            # Get available features
            available_features = self._get_available_features(df)

            # Check zone_id
            if 'zone_id' not in df.columns:
                print("❌ ERROR: 'zone_id' not found!")
                return {}

            # Sort by zone and time
            df_sorted = df.sort_values(['zone_id', 'time_window']).reset_index(drop=True)

            print(f"📊 Evaluation data stats:")
            print(f"   Total records: {len(df_sorted)}")
            print(f"   Unique zones: {df_sorted['zone_id'].nunique()}")

            # Build sequences
            seq_builder = SequenceBuilder(window_size=30, stride=5)
            X_seq = seq_builder.build_sequences(df_sorted, available_features)

            print(f"📦 Built {X_seq.shape[0]} sequences for evaluation")

            # Get scores
            scores = lstm_model.predict_score(X_seq)

            # Calculate adaptive threshold
            if use_adaptive_threshold:
                threshold = self.calculate_adaptive_threshold(scores, method='percentile', percentile=90)
                predictions = (scores > threshold).astype(int)
                print(f"   Adaptive threshold: {threshold:.4f} (vs training: {lstm_model.threshold:.4f})")
                print(f"   Threshold ratio: {threshold / lstm_model.threshold:.2f}x")
            else:
                predictions = lstm_model.predict(X_seq)

            # Check if labels exist
            has_labels = 'is_anomaly' in df.columns

            if has_labels:
                # Map sequence predictions back
                y_true_seq = []
                for i in range(len(X_seq)):
                    start_idx = i * 5
                    if start_idx < len(df_sorted):
                        y_true_seq.append(df_sorted['is_anomaly'].iloc[start_idx])
                y_true_seq = np.array(y_true_seq)
            else:
                # Create pseudo-labels
                y_true_seq = (scores > np.percentile(scores, 90)).astype(int)
                print("⚠️ No labels - using pseudo-labels for visualization")

            # Generate plots
            plot_dir = self.plot_output_dir / tag
            plot_dir.mkdir(parents=True, exist_ok=True)

            print(f"   📊 Generating LSTM plots in {plot_dir}...")
            metrics = self.metrics_calc.create_evaluation_report(
                y_true=y_true_seq,
                y_pred=predictions,
                y_scores=scores,
                model_name=f"lstm_autoencoder_{tag}",
                save_dir=str(plot_dir)
            )

            if has_labels:
                from sklearn.metrics import precision_score, recall_score, f1_score

                precision = precision_score(y_true_seq, predictions, zero_division=0)
                recall = recall_score(y_true_seq, predictions, zero_division=0)
                f1 = f1_score(y_true_seq, predictions, zero_division=0)

                print(f"   Precision: {precision:.4f}")
                print(f"   Recall:    {recall:.4f}")
                print(f"   F1 Score:  {f1:.4f}")

                metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "threshold": threshold if use_adaptive_threshold else lstm_model.threshold
                })
            else:
                anomaly_rate = float(predictions.mean())
                metrics.update({
                    "anomaly_rate": anomaly_rate,
                    "threshold": threshold if use_adaptive_threshold else lstm_model.threshold
                })
                print(f"   Anomaly Rate: {anomaly_rate:.2%}")

            # Save results
            results_df = pd.DataFrame([metrics])
            results_path = self.results_output_dir / f"{tag}_lstm_results_with_plots.csv"
            results_df.to_csv(results_path)
            print(f"💾 Results saved to {results_path}")
            print(f"📊 Plots saved to {plot_dir}")

            return metrics

        except Exception as e:
            print(f"❌ LSTM evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def train_models(self, df):
        """Train all anomaly detection models"""
        models = {}

        from src.models.model_registry import MODEL_REGISTRY
        from src.data.sequence_builder import SequenceBuilder

        # Get available features
        available_features = self._get_available_features(df)

        if len(available_features) == 0:
            print("❌ No valid features found for training!")
            return models

        # Train tabular models
        print("\n" + "=" * 60)
        print("TRAINING TABULAR MODELS")
        print("=" * 60)

        for name, Model in MODEL_REGISTRY.items():
            if name == "lstm_autoencoder":
                continue

            print(f"\n🔧 Training {name}...")
            try:
                model = Model()
                X_train = df[available_features]
                model.fit(X_train)
                model.save_model(self.model_output_dir / f"{name}.pkl")
                models[name] = model
                print(f"   ✅ {name} trained and saved")

            except Exception as e:
                print(f"   ❌ {name} training failed: {e}")
                continue

        # Train LSTM model
        print("\n" + "=" * 60)
        print("TRAINING LSTM AUTOENCODER (ZONE-AWARE)")
        print("=" * 60)

        try:
            if 'zone_id' not in df.columns:
                print("❌ ERROR: 'zone_id' not found!")
                raise ValueError("zone_id is required for LSTM training")

            if df['zone_id'].dtype == 'object':
                print("⚠️ Converting zone_id to numeric...")
                zone_mapping = {zone: idx for idx, zone in enumerate(df['zone_id'].unique())}
                df['zone_id'] = df['zone_id'].map(zone_mapping)

            print(f"📊 Training data stats:")
            print(f"   Total records: {len(df)}")
            print(f"   Unique zones: {df['zone_id'].nunique()}")

            df_sorted = df.sort_values(['zone_id', 'time_window']).reset_index(drop=True)

            seq_builder = SequenceBuilder(window_size=30, stride=5)
            X_seq = seq_builder.build_sequences(df_sorted, available_features)

            print(f"📦 Built sequences: {X_seq.shape}")

            lstm_model = MODEL_REGISTRY["lstm_autoencoder"](
                sequence_length=X_seq.shape[1],
                hidden_size=32,
                num_layers=2,
                epochs=20,
                batch_size=16
            )

            print(f"\n🔥 Training LSTM Autoencoder...")
            lstm_model.fit(X_seq)
            lstm_model.save_model(self.model_output_dir / "lstm_autoencoder.pkl")

            models["lstm_autoencoder"] = lstm_model
            print("   ✅ LSTM Autoencoder trained and saved")

        except Exception as e:
            print(f"   ❌ LSTM training failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n✅ Training complete! Trained {len(models)} models")
        return models

    def run(self):
        """Execute complete S.A.F.E pipeline with plots"""
        print("\n" + "🚀 " * 35)
        print("S.A.F.E COMPLETE PIPELINE (WITH PLOTS)")
        print("🚀 " * 35)

        # Load and train
        print("\n" + "=" * 60)
        print("STEP 1: LOADING TRAINING DATA")
        print("=" * 60)

        raw_train_df = self.data_loader.load_eth_ucy(self.train_scenes)
        zone_id_backup = raw_train_df['zone_id'].copy() if 'zone_id' in raw_train_df.columns else None
        train_df = self.preprocessing.fit_transform(raw_train_df.copy())

        if zone_id_backup is not None and 'zone_id' not in train_df.columns:
            train_df['zone_id'] = zone_id_backup.values

        # Train models
        print("\n" + "=" * 60)
        print("STEP 2: TRAINING MODELS")
        print("=" * 60)

        models = self.train_models(train_df)

        # Validation with plots
        print("\n" + "=" * 60)
        print("STEP 3: VALIDATION WITH PLOTS")
        print("=" * 60)

        try:
            raw_val_df = self.data_loader.load_eth_ucy(self.validation_scenes)
            zone_id_backup = raw_val_df['zone_id'].copy() if 'zone_id' in raw_val_df.columns else None
            val_df = self.preprocessing.transform(raw_val_df.copy())

            if zone_id_backup is not None and 'zone_id' not in val_df.columns:
                val_df['zone_id'] = zone_id_backup.values

            # Evaluate with plots
            self.evaluate_with_plots(models, val_df, tag="validation", use_adaptive_threshold=True)

            if "lstm_autoencoder" in models:
                self.evaluate_lstm_with_plots(models["lstm_autoencoder"], val_df,
                                            tag="validation", use_adaptive_threshold=True)

        except Exception as e:
            print(f"⚠️ Validation skipped: {e}")

        # Test on Mall with plots
        if self.test_dataset == "mall":
            print("\n" + "=" * 60)
            print("STEP 4: TESTING ON MALL WITH PLOTS")
            print("=" * 60)

            try:
                raw_test_df = self.data_loader.load_mall_dataset()

                if raw_test_df.empty:
                    print("⚠️ Mall dataset is empty")
                    return

                zone_id_backup = raw_test_df['zone_id'].copy() if 'zone_id' in raw_test_df.columns else None
                test_df = self.preprocessing.transform(raw_test_df.copy())

                if zone_id_backup is not None and 'zone_id' not in test_df.columns:
                    test_df['zone_id'] = zone_id_backup.values

                # Evaluate with plots
                print("\n🎯 Using adaptive thresholding for cross-domain testing...")
                self.evaluate_with_plots(models, test_df, tag="mall", use_adaptive_threshold=True)

                if "lstm_autoencoder" in models:
                    self.evaluate_lstm_with_plots(models["lstm_autoencoder"], test_df,
                                                tag="mall", use_adaptive_threshold=True)

            except Exception as e:
                print(f"❌ Mall testing failed: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "✅ " * 35)
        print("PIPELINE COMPLETE!")
        print(f"📊 All plots saved to: {self.plot_output_dir}")
        print("✅ " * 35)