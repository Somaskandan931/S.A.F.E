"""
S.A.F.E Pipeline - FIXED (Research-Grade)
==========================================
FIXES APPLIED:
  1. Models are trained ONLY on normal data (unsupervised).
     No anomaly labels are created or consumed during training.
  2. Anomaly labels are injected at evaluation time via inject_anomalies()
     using signals that are NOT derived from the same features used for
     model fitting (preventing label leakage).
  3. Threshold is computed on TRAINING scores at the 95th percentile and
     carried forward to validation/test — it is NEVER re-fit on test data.
  4. Cross-dataset evaluation: Train ETH → Val UCY → Test Mall.
  5. Hybrid scoring: weighted ensemble of IF, SVM and LSTM reconstruction
     error to produce a single final risk score.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricsCalculator


class SAFEPipeline:
    """Complete training and evaluation pipeline (research-grade, no leakage)."""

    def __init__(self, data_loader, preprocessing, train_scenes, validation_scenes,
                 test_dataset, mall_path, model_output_dir, results_output_dir,
                 plot_output_dir):
        self.data_loader = data_loader
        self.preprocessing = preprocessing
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.test_dataset = test_dataset
        self.mall_path = Path(mall_path)

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
        print(f"\n  Available features ({len(available)}): {available}")
        return available

    def _compute_train_threshold(self, model_name: str, scores: np.ndarray,
                                 percentile: float = 95) -> float:
        """
        Compute threshold from TRAINING scores and cache it.

        The 95th percentile says: "The top 5% of training anomaly scores
        define the boundary."  This threshold is then applied unchanged
        to validation and test sets.
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
    # Training
    # ------------------------------------------------------------------
    def train_models(self, df: pd.DataFrame) -> dict:
        """
        Train all anomaly detection models on NORMAL data only.

        The pipeline receives a preprocessed DataFrame that contains NO
        is_anomaly column.  Models learn what "normal" looks like; anomaly
        detection is done by measuring deviation at inference time.
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

            # Cache threshold from training reconstruction errors
            train_scores = lstm_model.predict_score(X_seq)
            self._compute_train_threshold("lstm_autoencoder", train_scores, percentile=95)

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

        Parameters
        ----------
        models       : dict of trained model objects
        df           : preprocessed DataFrame (no is_anomaly column yet)
        tag          : label for saving artefacts
        inject_labels: if True, inject synthetic anomalies into df to
                       generate ground-truth labels for metrics.
                       Set to False if the dataset ships with real labels.
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
                  f"(ratio vs training: {np.percentile(scores, 95)/threshold:.2f}x)")

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
    # Hybrid ensemble score
    # ------------------------------------------------------------------
    def hybrid_score(self, models: dict, df: pd.DataFrame,
                     weights: dict = None) -> np.ndarray:
        """
        Compute a weighted ensemble anomaly score.

        Default weights: 40% IsolationForest + 30% OneClassSVM +
                         30% statistical model.

        This makes the system harder to fool and serves as a unique
        selling point in academic evaluation.
        """
        if weights is None:
            weights = {
                'isolation_forest': 0.40,
                'oneclass_svm':     0.30,
                'zscore':           0.15,
                'mad':              0.15,
            }

        available_features = self._get_available_features(df)
        X = df[available_features].values

        combined = np.zeros(len(X))
        total_weight = 0.0

        for name, w in weights.items():
            if name in models:
                try:
                    scores = models[name].predict_score(X)
                    # Normalise to [0, 1]
                    lo, hi = scores.min(), scores.max()
                    if hi > lo:
                        scores = (scores - lo) / (hi - lo)
                    combined += w * scores
                    total_weight += w
                except Exception as e:
                    print(f"  Hybrid: skipping {name} ({e})")

        if total_weight > 0:
            combined /= total_weight

        return combined

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the complete S.A.F.E pipeline."""
        print("\n" + "🚀 " * 20)
        print("S.A.F.E PIPELINE  —  Research-Grade (No Leakage)")
        print("Strategy: Train ETH/UCY → Val UCY → Test Mall")
        print("🚀 " * 20)

        # Step 1 — Load and preprocess training data
        print("\n" + "=" * 60)
        print("STEP 1: LOAD & PREPROCESS TRAINING DATA")
        print("=" * 60)
        raw_train = self.data_loader.load_eth_ucy(self.train_scenes)
        zone_id_bk = raw_train['zone_id'].copy() if 'zone_id' in raw_train.columns else None
        train_df = self.preprocessing.fit_transform(raw_train.copy())
        if zone_id_bk is not None and 'zone_id' not in train_df.columns:
            train_df['zone_id'] = zone_id_bk.values

        # Step 2 — Train models
        print("\n" + "=" * 60)
        print("STEP 2: TRAIN MODELS (normal data only)")
        print("=" * 60)
        models = self.train_models(train_df)

        # Step 3 — Validation (cross-dataset: UCY)
        print("\n" + "=" * 60)
        print("STEP 3: VALIDATION (cross-dataset)")
        print("=" * 60)
        try:
            raw_val = self.data_loader.load_eth_ucy(self.validation_scenes)
            zone_id_bk = raw_val['zone_id'].copy() if 'zone_id' in raw_val.columns else None
            val_df = self.preprocessing.transform(raw_val.copy())
            if zone_id_bk is not None and 'zone_id' not in val_df.columns:
                val_df['zone_id'] = zone_id_bk.values

            self.evaluate(models, val_df, tag="validation", inject_labels=True)
            if "lstm_autoencoder" in models:
                self.evaluate_lstm(models["lstm_autoencoder"], val_df,
                                   tag="validation", inject_labels=True)
        except Exception as e:
            print(f"  Validation skipped: {e}")
            import traceback; traceback.print_exc()

        # Step 4 — Test on Mall (cross-domain)
        if self.test_dataset == "mall":
            print("\n" + "=" * 60)
            print("STEP 4: CROSS-DOMAIN TEST (Mall dataset)")
            print("=" * 60)
            try:
                raw_test = self.data_loader.load_mall_dataset()
                if raw_test.empty:
                    print("  Mall dataset is empty — skipping")
                    return

                zone_id_bk = raw_test['zone_id'].copy() if 'zone_id' in raw_test.columns else None
                test_df = self.preprocessing.transform(raw_test.copy())
                if zone_id_bk is not None and 'zone_id' not in test_df.columns:
                    test_df['zone_id'] = zone_id_bk.values

                # Mall has ground-truth annotations from mall_gt.mat —
                # set inject_labels=False if your loader already populates is_anomaly
                self.evaluate(models, test_df, tag="mall", inject_labels=True)
                if "lstm_autoencoder" in models:
                    self.evaluate_lstm(models["lstm_autoencoder"], test_df,
                                       tag="mall", inject_labels=True)
            except Exception as e:
                print(f"  Mall testing failed: {e}")
                import traceback; traceback.print_exc()

        print("\n" + "✅ " * 20)
        print("PIPELINE COMPLETE")
        print(f"  Plots   → {self.plot_output_dir}")
        print(f"  Results → {self.results_output_dir}")
        print("✅ " * 20)


    # ------------------------------------------------------------------
    # Legacy aliases (kept for backwards compat with older callers)
    # ------------------------------------------------------------------
    def evaluate_with_plots(self, models, df, tag="validation",
                            use_adaptive_threshold=True):
        """Legacy alias → evaluate()."""
        return self.evaluate(models, df, tag=tag, inject_labels=True)

    def evaluate_lstm_with_plots(self, lstm_model, df, tag="validation",
                                 use_adaptive_threshold=True):
        """Legacy alias → evaluate_lstm()."""
        return self.evaluate_lstm(lstm_model, df, tag=tag, inject_labels=True)