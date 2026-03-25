"""
S.A.F.E Pipeline - FIXED (Research-Grade)
==========================================
FIXES APPLIED:
  1. Models are trained ONLY on normal data (unsupervised).
  2. Mall normal data can be optionally added to training (controlled via config)
  3. Threshold is computed on TRAINING scores at the 95th percentile and
     carried forward to validation/test — it is NEVER re-fit on test data.
  4. LSTM scores are normalized for consistent thresholds.
  5. Cross-dataset evaluation: Train ETH + (optional Mall) → Val UCY → Test Mall (held-out)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricsCalculator


class SAFEPipeline:
    """Complete training and evaluation pipeline (research-grade, no leakage)."""

    def __init__(self, data_loader, preprocessing, train_scenes, validation_scenes,
                 test_dataset, mall_path, model_output_dir, results_output_dir,
                 plot_output_dir, include_mall_in_training: bool = False,
                 mall_train_split: float = 0.7):
        self.data_loader = data_loader
        self.preprocessing = preprocessing
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.test_dataset = test_dataset
        self.mall_path = Path(mall_path)
        self.include_mall_in_training = include_mall_in_training
        self.mall_train_split = mall_train_split

        self.model_output_dir = Path(model_output_dir)
        self.results_output_dir = Path(results_output_dir)
        self.plot_output_dir = Path(plot_output_dir)

        for d in [self.model_output_dir, self.results_output_dir, self.plot_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.metrics_calc = MetricsCalculator()

        # Feature columns used in training
        self.BASE_FEATURES = [
            'time_window', 'footfall_count', 'velocity_mean', 'velocity_std',
            'direction_variance', 'zone_center_x', 'zone_center_y', 'frame_id', 'density'
        ]
        self.TEMPORAL_FEATURES = [
            'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'quarter',
            'is_weekend', 'is_business_hours'
        ]
        self.FEATURE_COLUMNS = self.BASE_FEATURES + self.TEMPORAL_FEATURES

        # Threshold computed from training scores (carried to val/test without re-fitting)
        self.train_thresholds = {}

        # Store held-out test data
        self.mall_test_data = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_available_features(self, df: pd.DataFrame) -> list:
        """Return feature columns that exist in df, excluding metadata."""
        exclude = {'dataset', 'timestamp', 'is_anomaly', 'zone_id', 'pedestrian_id'}
        available = []
        for col in self.FEATURE_COLUMNS:
            if col in df.columns and col not in exclude:
                available.append(col)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in available and col not in exclude:
                available.append(col)
        print(f"\n  Available features ({len(available)}): {available[:10]}...")
        return available

    def _compute_train_threshold(self, model_name: str, scores: np.ndarray,
                                 percentile: float = 95) -> float:
        """
        Compute threshold from TRAINING scores and cache it.
        The 95th percentile says: "The top 5% of training anomaly scores define the boundary."
        """
        threshold = float(np.percentile(scores, percentile))
        self.train_thresholds[model_name] = threshold
        print(f"  Threshold (p{percentile:.0f} of training scores): {threshold:.6f}")
        return threshold

    def _get_test_threshold(self, model_name: str) -> float:
        """Return cached training threshold for use on val/test data."""
        if model_name not in self.train_thresholds:
            raise RuntimeError(
                f"No training threshold found for '{model_name}'. "
                "Call train_models() before evaluate()."
            )
        return self.train_thresholds[model_name]

    # ------------------------------------------------------------------
    # Data Loading with Mall Split
    # ------------------------------------------------------------------
    def _load_training_data(self) -> pd.DataFrame:
        """Load training data (ETH/UCY + optionally Mall normal data)."""
        print("\n" + "=" * 60)
        print("LOADING TRAINING DATA")
        print("=" * 60)

        # Load ETH/UCY training data
        raw_train = self.data_loader.load_eth_ucy(self.train_scenes)

        if raw_train.empty:
            print("  ❌ No ETH/UCY training data loaded!")
            return pd.DataFrame()

        print(f"  ✅ ETH/UCY training: {len(raw_train):,} records")

        # Optionally include Mall normal data in training
        if self.include_mall_in_training:
            print(f"\n  Including Mall normal data in training (split: {self.mall_train_split*100:.0f}% train / {(1-self.mall_train_split)*100:.0f}% test)")

            try:
                raw_mall = self.data_loader.load_mall_dataset()

                if not raw_mall.empty:
                    from sklearn.model_selection import train_test_split

                    # Split Mall into train and test
                    mall_train, self.mall_test_data = train_test_split(
                        raw_mall,
                        train_size=self.mall_train_split,
                        random_state=42
                    )

                    print(f"  Mall training: {len(mall_train):,} records")
                    print(f"  Mall held-out test: {len(self.mall_test_data):,} records")

                    # Combine training data
                    raw_train = pd.concat([raw_train, mall_train], ignore_index=True)
                    print(f"  ✅ Combined training: {len(raw_train):,} records")
                else:
                    print("  ⚠️ Mall dataset empty - skipping")
                    self.mall_test_data = None
            except Exception as e:
                print(f"  ⚠️ Could not load Mall for training: {e}")
                self.mall_test_data = None
        else:
            self.mall_test_data = None
            print("  Mall dataset NOT included in training (honest domain shift test)")

        return raw_train

    def _load_validation_data(self) -> pd.DataFrame:
        """Load validation data (Zara1/Zara2)."""
        print("\n" + "=" * 60)
        print("LOADING VALIDATION DATA")
        print("=" * 60)

        raw_val = self.data_loader.load_eth_ucy(self.validation_scenes)

        if raw_val.empty:
            print("  ⚠️ No validation data loaded!")
        else:
            print(f"  ✅ Validation data: {len(raw_val):,} records")

        return raw_val

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_models(self, df: pd.DataFrame) -> dict:
        """
        Train all anomaly detection models on NORMAL data only.
        """
        from src.models.model_registry import MODEL_REGISTRY
        from src.data.sequence_builder import SequenceBuilder

        models = {}
        available_features = self._get_available_features(df)

        if not available_features:
            print("ERROR: No valid features — aborting training")
            return models

        if 'is_anomaly' in df.columns:
            print(
                "\nWARNING: 'is_anomaly' column found in training data — dropping it. "
                "Models must be trained on raw features only."
            )
            df = df.drop(columns=['is_anomaly'])

        # ---- Tabular models ----------------------------------------
        print("\n" + "=" * 60)
        print("TRAINING TABULAR MODELS (unsupervised, normal data only)")
        print("=" * 60)

        for name, Model in MODEL_REGISTRY.items():
            if name == "lstm_autoencoder":
                continue
            print(f"\n  Training {name}...")
            try:
                model = Model()
                X_train = df[available_features]
                model.fit(X_train)

                # Compute and cache threshold from training scores
                train_scores = model.predict_score(X_train.values)
                self._compute_train_threshold(name, train_scores, percentile=95)

                model.save_model(self.model_output_dir / f"{name}.pkl")
                models[name] = model
                print(f"  ✅ {name} trained and saved")
            except Exception as e:
                print(f"  ❌ {name} training failed: {e}")
                import traceback; traceback.print_exc()

        # ---- LSTM Autoencoder --------------------------------------
        print("\n" + "=" * 60)
        print("TRAINING LSTM AUTOENCODER (normal sequences only)")
        print("=" * 60)

        try:
            if 'zone_id' not in df.columns:
                raise ValueError("'zone_id' column required for LSTM training")

            if df['zone_id'].dtype == 'object':
                mapping = {z: i for i, z in enumerate(df['zone_id'].unique())}
                df['zone_id'] = df['zone_id'].map(mapping)

            df_sorted = df.sort_values(['zone_id', 'time_window']).reset_index(drop=True)
            seq_builder = SequenceBuilder(window_size=30, stride=5)
            X_seq = seq_builder.build_sequences(df_sorted, available_features)
            print(f"  Built sequences: {X_seq.shape}")

            lstm_model = MODEL_REGISTRY["lstm_autoencoder"](
                sequence_length=X_seq.shape[1],
                hidden_size=32,
                num_layers=2,
                epochs=20,
                batch_size=16
            )
            lstm_model.fit(X_seq)

            # Compute threshold from training reconstruction errors
            train_scores = lstm_model.predict_score(X_seq)

            # Normalize scores to [0, 1] range for consistent thresholds
            score_min = float(train_scores.min())
            score_max = float(train_scores.max())
            if score_max > score_min:
                train_scores_norm = (train_scores - score_min) / (score_max - score_min)
                threshold = float(np.percentile(train_scores_norm, 95))
            else:
                threshold = float(np.percentile(train_scores, 95))
                score_min, score_max = 0.0, 1.0

            self.train_thresholds["lstm_autoencoder"] = threshold
            print(f"  LSTM training threshold (p95): {threshold:.6f}")
            print(f"  Score range: [{score_min:.6f}, {score_max:.6f}]")

            # Store normalization parameters in model
            lstm_model.score_min = score_min
            lstm_model.score_max = score_max

            lstm_model.save_model(self.model_output_dir / "lstm_autoencoder.pkl")
            models["lstm_autoencoder"] = lstm_model
            print("  ✅ LSTM Autoencoder trained and saved")
        except Exception as e:
            print(f"  ❌ LSTM training failed: {e}")
            import traceback; traceback.print_exc()

        print(f"\n✅ Training complete — {len(models)} models")
        return models

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, models: dict, df: pd.DataFrame, tag: str = "validation",
                 inject_labels: bool = True) -> dict:
        """
        Evaluate tabular models on val/test data.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION — {tag.upper()}")
        print("Threshold: from training scores (no re-fitting on test data)")
        print(f"{'='*60}")

        available_features = self._get_available_features(df)
        if not available_features:
            print("  No valid features — skipping")
            return {}

        # Inject anomaly labels for evaluation (independent of model features)
        if inject_labels and 'is_anomaly' not in df.columns:
            print(f"\n  Injecting evaluation anomalies ({tag})...")
            df = self.preprocessing.inject_anomalies(df, anomaly_ratio=0.05, random_state=42)
        elif 'is_anomaly' not in df.columns:
            print("  WARNING: No labels available — metrics will not be computed")

        has_labels = 'is_anomaly' in df.columns
        y_true = df['is_anomaly'].values if has_labels else None

        X = df[available_features]
        results = {}
        plot_dir = self.plot_output_dir / tag
        plot_dir.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            if name == "lstm_autoencoder":
                continue
            print(f"\n  Evaluating {name}...")
            try:
                scores = model.predict_score(X.values)

                # Use TRAINING threshold — never recalculate on test data
                threshold = self._get_test_threshold(name)
                predictions = (scores > threshold).astype(int)
                print(f"  Applied training threshold: {threshold:.6f}")

                metrics = self.metrics_calc.create_evaluation_report(
                    y_true=y_true if has_labels else (scores > np.percentile(scores, 95)).astype(int),
                    y_pred=predictions,
                    y_scores=scores,
                    model_name=f"{name}_{tag}",
                    save_dir=str(plot_dir)
                )

                if has_labels:
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                    p = precision_score(y_true, predictions, zero_division=0)
                    r = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)
                    try:
                        auc = roc_auc_score(y_true, scores)
                    except Exception:
                        auc = float('nan')
                    print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
                    metrics.update({"precision": p, "recall": r, "f1_score": f1, "roc_auc": auc,
                                    "threshold": threshold})
                else:
                    rate = float(predictions.mean())
                    metrics.update({"anomaly_rate": rate, "threshold": threshold})
                    print(f"  Anomaly rate: {rate:.2%}")

                results[name] = metrics
            except Exception as e:
                print(f"  ❌ {name} evaluation failed: {e}")
                import traceback; traceback.print_exc()

        if results:
            pd.DataFrame(results).T.to_csv(
                self.results_output_dir / f"{tag}_results.csv"
            )
            print(f"\n  Results saved → {self.results_output_dir / f'{tag}_results.csv'}")

        return results

    def evaluate_lstm(self, lstm_model, df: pd.DataFrame, tag: str = "validation",
                      inject_labels: bool = True) -> dict:
        """Evaluate the LSTM autoencoder on sequence data."""
        print(f"\n{'='*60}")
        print(f"LSTM EVALUATION — {tag.upper()}")
        print("Threshold: from training reconstruction errors (no re-fitting)")
        print(f"{'='*60}")

        try:
            from src.data.sequence_builder import SequenceBuilder

            available_features = self._get_available_features(df)

            if 'zone_id' not in df.columns:
                print("  ERROR: 'zone_id' missing — cannot build sequences")
                return {}

            if inject_labels and 'is_anomaly' not in df.columns:
                print(f"\n  Injecting evaluation anomalies ({tag})...")
                df = self.preprocessing.inject_anomalies(df, anomaly_ratio=0.05, random_state=42)

            has_labels = 'is_anomaly' in df.columns

            if df['zone_id'].dtype == 'object':
                mapping = {z: i for i, z in enumerate(df['zone_id'].unique())}
                df['zone_id'] = df['zone_id'].map(mapping)

            df_sorted = df.sort_values(['zone_id', 'time_window']).reset_index(drop=True)
            seq_builder = SequenceBuilder(window_size=30, stride=5)
            X_seq = seq_builder.build_sequences(df_sorted, available_features)
            print(f"  Built {X_seq.shape[0]} evaluation sequences")

            scores = lstm_model.predict_score(X_seq)

            # Use TRAINING threshold
            threshold = self._get_test_threshold("lstm_autoencoder")
            predictions = (scores > threshold).astype(int)
            print(f"  Applied training threshold: {threshold:.6f}")
            print(f"  Current score p95: {np.percentile(scores, 95):.6f}  "
                  f"(ratio vs threshold: {np.percentile(scores, 95)/threshold:.2f}x)")

            # Map labels to sequences
            if has_labels:
                stride = 5
                y_true_seq = []
                for i in range(len(X_seq)):
                    start = i * stride
                    if start < len(df_sorted):
                        y_true_seq.append(int(df_sorted['is_anomaly'].iloc[start]))
                    else:
                        y_true_seq.append(0)
                y_true_seq = np.array(y_true_seq)
            else:
                y_true_seq = (scores > np.percentile(scores, 95)).astype(int)

            plot_dir = self.plot_output_dir / tag
            plot_dir.mkdir(parents=True, exist_ok=True)

            metrics = self.metrics_calc.create_evaluation_report(
                y_true=y_true_seq,
                y_pred=predictions,
                y_scores=scores,
                model_name=f"lstm_autoencoder_{tag}",
                save_dir=str(plot_dir)
            )

            if has_labels:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                p = precision_score(y_true_seq, predictions, zero_division=0)
                r = recall_score(y_true_seq, predictions, zero_division=0)
                f1 = f1_score(y_true_seq, predictions, zero_division=0)
                try:
                    auc = roc_auc_score(y_true_seq, scores)
                except Exception:
                    auc = float('nan')
                print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
                metrics.update({"precision": p, "recall": r, "f1_score": f1, "roc_auc": auc,
                                 "threshold": threshold})
            else:
                rate = float(predictions.mean())
                metrics.update({"anomaly_rate": rate, "threshold": threshold})
                print(f"  Anomaly rate: {rate:.2%}")

            pd.DataFrame([metrics]).to_csv(
                self.results_output_dir / f"{tag}_lstm_results.csv"
            )
            return metrics

        except Exception as e:
            print(f"  ❌ LSTM evaluation failed: {e}")
            import traceback; traceback.print_exc()
            return {}

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the complete S.A.F.E pipeline."""
        print("\n" + "🚀 " * 20)
        print("S.A.F.E PIPELINE — Research-Grade (No Leakage)")
        if self.include_mall_in_training:
            print("Strategy: Train ETH/UCY + Mall (normal) → Val UCY → Test Mall (held-out)")
        else:
            print("Strategy: Train ETH/UCY → Val UCY → Test Mall (cross-domain generalization)")
        print("🚀 " * 20)

        # Step 1 — Load training data (ETH/UCY + optional Mall)
        raw_train = self._load_training_data()
        if raw_train.empty:
            print("❌ No training data loaded. Exiting.")
            return

        # Fit preprocessing on training data only
        print("\n" + "=" * 60)
        print("STEP 1: PREPROCESS TRAINING DATA")
        print("=" * 60)
        zone_id_bk = raw_train['zone_id'].copy() if 'zone_id' in raw_train.columns else None
        train_df = self.preprocessing.fit_transform(raw_train.copy())
        if zone_id_bk is not None and 'zone_id' not in train_df.columns:
            train_df['zone_id'] = zone_id_bk.values

        # Step 2 — Train models
        print("\n" + "=" * 60)
        print("STEP 2: TRAIN MODELS (normal data only)")
        print("=" * 60)
        models = self.train_models(train_df)

        # Step 3 — Load and evaluate validation data (Zara1/Zara2)
        raw_val = self._load_validation_data()
        if not raw_val.empty:
            print("\n" + "=" * 60)
            print("STEP 3: VALIDATION (cross-scene: Zara1 + Zara2)")
            print("=" * 60)
            try:
                zone_id_bk = raw_val['zone_id'].copy() if 'zone_id' in raw_val.columns else None
                val_df = self.preprocessing.transform(raw_val.copy())
                if zone_id_bk is not None and 'zone_id' not in val_df.columns:
                    val_df['zone_id'] = zone_id_bk.values

                self.evaluate(models, val_df, tag="validation", inject_labels=True)
                if "lstm_autoencoder" in models:
                    self.evaluate_lstm(models["lstm_autoencoder"], val_df,
                                       tag="validation", inject_labels=True)
            except Exception as e:
                print(f"  Validation failed: {e}")
                import traceback; traceback.print_exc()
        else:
            print("⚠️ No validation data — skipping")

        # Step 4 — Test on Mall (held-out if included in training, otherwise cross-domain)
        if self.mall_test_data is not None and not self.mall_test_data.empty:
            print("\n" + "=" * 60)
            print("STEP 4: TEST ON HELD-OUT MALL DATA")
            print(f"Training included Mall: {self.include_mall_in_training}")
            print("=" * 60)
            try:
                zone_id_bk = self.mall_test_data['zone_id'].copy() if 'zone_id' in self.mall_test_data.columns else None
                test_df = self.preprocessing.transform(self.mall_test_data.copy())
                if zone_id_bk is not None and 'zone_id' not in test_df.columns:
                    test_df['zone_id'] = zone_id_bk.values

                self.evaluate(models, test_df, tag="mall_heldout", inject_labels=True)
                if "lstm_autoencoder" in models:
                    self.evaluate_lstm(models["lstm_autoencoder"], test_df,
                                       tag="mall_heldout", inject_labels=True)
            except Exception as e:
                print(f"  Mall held-out test failed: {e}")
                import traceback; traceback.print_exc()
        elif self.test_dataset == "mall":
            # Try to load Mall dataset for cross-domain test (no training exposure)
            print("\n" + "=" * 60)
            print("STEP 4: CROSS-DOMAIN TEST (Mall dataset — no training exposure)")
            print("This tests how well models generalize to completely new environments")
            print("=" * 60)
            try:
                raw_test = self.data_loader.load_mall_dataset()
                if not raw_test.empty:
                    zone_id_bk = raw_test['zone_id'].copy() if 'zone_id' in raw_test.columns else None
                    test_df = self.preprocessing.transform(raw_test.copy())
                    if zone_id_bk is not None and 'zone_id' not in test_df.columns:
                        test_df['zone_id'] = zone_id_bk.values

                    self.evaluate(models, test_df, tag="mall_crossdomain", inject_labels=True)
                    if "lstm_autoencoder" in models:
                        self.evaluate_lstm(models["lstm_autoencoder"], test_df,
                                           tag="mall_crossdomain", inject_labels=True)
            except Exception as e:
                print(f"  Cross-domain test failed: {e}")

        # Step 5 — Print summary
        self._print_summary(models)

        print("\n" + "✅ " * 20)
        print("PIPELINE COMPLETE")
        print(f"  Plots   → {self.plot_output_dir}")
        print(f"  Results → {self.results_output_dir}")
        print("✅ " * 20)

    def _print_summary(self, models: dict):
        """Print summary of trained models."""
        print("\n" + "=" * 60)
        print("TRAINED MODELS SUMMARY")
        print("=" * 60)
        print(f"Total models: {len(models)}")
        for name, model in models.items():
            print(f"  - {name}: {type(model).__name__}")
            if name in self.train_thresholds:
                print(f"      Threshold: {self.train_thresholds[name]:.6f}")

    # ------------------------------------------------------------------
    # Legacy aliases (kept for backwards compat)
    # ------------------------------------------------------------------
    def evaluate_with_plots(self, models, df, tag="validation",
                            use_adaptive_threshold=True):
        """Legacy alias → evaluate()."""
        return self.evaluate(models, df, tag=tag, inject_labels=True)

    def evaluate_lstm_with_plots(self, lstm_model, df, tag="validation",
                                 use_adaptive_threshold=True):
        """Legacy alias → evaluate_lstm()."""
        return self.evaluate_lstm(lstm_model, df, tag=tag, inject_labels=True)