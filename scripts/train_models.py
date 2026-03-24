"""
S.A.F.E Training Script - FIXED for your dataset structure
==========================================================
FIXES APPLIED:
  1. Fixed Mall dataset path handling
  2. Added proper file mapping for your CSV files
  3. Added debug output for Mall dataset loading
  4. Fixed working directory issues
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change working directory to project root for consistent path handling
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

from src.pipeline.safe_pipeline import SAFEPipeline
from src.data.data_loader import DatasetManager
from src.data.preprocessing import DataPreprocessor
import src.config.default_config as config

# ===================== CONFIGURATION =====================
TRAIN_SCENES      = ["eth", "hotel", "univ"]
VALIDATION_SCENES = ["zara1", "zara2"]
TEST_DATASET      = "mall"

# Create output directories
ANALYSIS_DIR = Path("results/analysis")
PLOTS_DIR    = Path("results/plots")
MODELS_DIR   = Path("scripts/models")
RESULTS_DIR  = Path("results")
MALL_PATH    = project_root / "data" / "raw" / "mall_dataset"

for d in [ANALYSIS_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("S.A.F.E TRAINING SCRIPT")
print("=" * 60)
print(f"Project Root: {project_root}")
print(f"Working Directory: {os.getcwd()}")
print(f"Models Dir: {MODELS_DIR}")
print(f"Results Dir: {RESULTS_DIR}")

# ===================== CHECK DATASETS =====================
print("\n" + "=" * 60)
print("CHECKING DATASETS")
print("=" * 60)

# Check ETH/UCY files
eth_ucy_path = project_root / "data" / "raw" / "eth_ucy"
print(f"\n📁 ETH/UCY Directory: {eth_ucy_path}")
if eth_ucy_path.exists():
    csv_files = list(eth_ucy_path.glob("*.csv"))
    print(f"   Found {len(csv_files)} CSV file(s):")
    # Show first 10 files to avoid too much output
    for f in csv_files[:10]:
        print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    if len(csv_files) > 10:
        print(f"     ... and {len(csv_files) - 10} more files")
else:
    print(f"   ❌ ETH/UCY directory not found: {eth_ucy_path}")

# Check Mall dataset
print(f"\n📁 Mall Dataset Directory: {MALL_PATH}")
if MALL_PATH.exists():
    gt_file = MALL_PATH / "mall_gt.mat"
    frames_dir = MALL_PATH / "frames"

    if gt_file.exists():
        print(f"   ✅ mall_gt.mat found ({gt_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ❌ mall_gt.mat not found")

    if frames_dir.exists():
        frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        print(f"   ✅ frames directory found with {len(frames)} images")
    else:
        print(f"   ❌ frames directory not found")

    # Debug mall_gt.mat structure
    try:
        import scipy.io
        print(f"\n   🔍 Debugging mall_gt.mat structure...")
        mat_data = scipy.io.loadmat(str(gt_file))
        print(f"   Variables in mall_gt.mat: {[k for k in mat_data.keys() if not k.startswith('__')]}")

        if 'frame' in mat_data:
            frames_data = mat_data['frame']
            print(f"   frame shape: {frames_data.shape}")
            print(f"   frame type: {type(frames_data)}")

            # Check first few frames
            for i in range(min(3, len(frames_data[0]))):
                frame_item = frames_data[0][i]
                print(f"   Frame {i}: type={type(frame_item)}")
                if hasattr(frame_item, 'dtype'):
                    print(f"      dtype: {frame_item.dtype}")
                if hasattr(frame_item, 'loc'):
                    print(f"      Has 'loc' attribute")
                    if hasattr(frame_item.loc, 'shape'):
                        print(f"      loc shape: {frame_item.loc.shape}")
    except Exception as e:
        print(f"   ⚠️ Could not debug mall_gt.mat: {e}")
else:
    print(f"   ⚠️ Mall dataset directory not found")
    print("   Cross-domain test will be skipped")
    TEST_DATASET = None

# ===================== OPTIONAL DATA ANALYSER =====================
USE_ANALYZER = False
try:
    from data_analysis_plots import DataAnalyzer
    analyzer = DataAnalyzer(output_dir=str(ANALYSIS_DIR))
    USE_ANALYZER = True
    print("\n  ✅ DataAnalyzer loaded")
except ImportError:
    print("\n  ℹ️ DataAnalyzer not found — skipping distribution plots")

# ===================== INITIALISE COMPONENTS =====================
data_loader   = DatasetManager()
preprocessing = DataPreprocessor()

# Print available datasets before proceeding
print("\n" + "=" * 60)
print("AVAILABLE DATASETS IN DATA LOADER")
print("=" * 60)
available = data_loader.get_available_datasets()
if available:
    for name, exists in available.items():
        print(f"  {name}: {'✅' if exists else '❌'}")
else:
    print("  No datasets found! Please check your data directory.")

pipeline = SAFEPipeline(
    data_loader=data_loader,
    preprocessing=preprocessing,
    train_scenes=TRAIN_SCENES,
    validation_scenes=VALIDATION_SCENES,
    test_dataset=TEST_DATASET if TEST_DATASET else "none",
    mall_path=str(MALL_PATH),
    model_output_dir=str(MODELS_DIR),
    results_output_dir=str(RESULTS_DIR),
    plot_output_dir=str(PLOTS_DIR),
)

# ===================== STEP 1 — TRAINING DATA =====================
print("\n" + "=" * 60)
print("STEP 1: LOADING TRAINING DATA")
print("=" * 60)

try:
    raw_train_df = data_loader.load_eth_ucy(TRAIN_SCENES)

    if raw_train_df.empty:
        print("  ❌ No training data loaded! Check dataset files.")
        print("  Attempting to load with alternative method...")
        # Try to load individual files
        all_data = []
        for scene in TRAIN_SCENES:
            df = data_loader.load_dataset(scene)
            if not df.empty:
                all_data.append(df)
        if all_data:
            raw_train_df = pd.concat(all_data, ignore_index=True)
            print(f"  ✅ Loaded {len(raw_train_df):,} training records using fallback method")
        else:
            print("  ❌ Still no data. Please check your CSV files.")
            sys.exit(1)
    else:
        print(f"  ✅ Loaded {len(raw_train_df):,} training records")

    if USE_ANALYZER:
        analyzer.generate_all_plots(raw_train_df, dataset_name="01_training_raw")

    # Fit scaler on training data — this is the ONLY fit call
    zone_id_bk = raw_train_df['zone_id'].copy() if 'zone_id' in raw_train_df.columns else None
    train_df = preprocessing.fit_transform(raw_train_df.copy())
    if zone_id_bk is not None and 'zone_id' not in train_df.columns:
        train_df['zone_id'] = zone_id_bk.values

    if USE_ANALYZER:
        analyzer.generate_all_plots(train_df, dataset_name="02_training_processed")

    print(f"  Training data shape: {train_df.shape}")
    print(f"  Features: {list(train_df.columns)[:10]}...")

except Exception as e:
    print(f"  ✗ Failed to load training data: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ===================== STEP 2 — TRAIN MODELS =====================
print("\n" + "=" * 60)
print("STEP 2: TRAINING MODELS (unsupervised, normal data only)")
print("=" * 60)

try:
    models = pipeline.train_models(train_df)
    print(f"  ✅ Trained {len(models)} models")
    print(f"  Models: {list(models.keys())}")

    # Save model list for reference
    with open(MODELS_DIR / "trained_models.txt", "w") as f:
        f.write("Trained Models:\n")
        for model_name in models.keys():
            f.write(f"  - {model_name}\n")
except Exception as e:
    print(f"  ✗ Training failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ===================== STEP 3 — VALIDATION =====================
print("\n" + "=" * 60)
print("STEP 3: VALIDATION (cross-scene: Zara1 + Zara2)")
print("NOTE: Thresholds carried from training — not re-fitted")
print("=" * 60)

val_results = {}
try:
    raw_val_df = data_loader.load_eth_ucy(VALIDATION_SCENES)

    if raw_val_df.empty:
        print("  ⚠️ No validation data found, trying individual loading...")
        all_val = []
        for scene in VALIDATION_SCENES:
            df = data_loader.load_dataset(scene)
            if not df.empty:
                all_val.append(df)
        if all_val:
            raw_val_df = pd.concat(all_val, ignore_index=True)
            print(f"  ✅ Loaded {len(raw_val_df):,} validation records using fallback method")
        else:
            print("  ⚠️ No validation data found, skipping validation")
            raw_val_df = pd.DataFrame()

    if not raw_val_df.empty:
        print(f"  ✅ Loaded {len(raw_val_df):,} validation records")

        if USE_ANALYZER:
            analyzer.generate_all_plots(raw_val_df, dataset_name="03_validation_raw")

        # Transform only — scaler already fitted on training data
        zone_id_bk = raw_val_df['zone_id'].copy() if 'zone_id' in raw_val_df.columns else None
        val_df = preprocessing.transform(raw_val_df.copy())
        if zone_id_bk is not None and 'zone_id' not in val_df.columns:
            val_df['zone_id'] = zone_id_bk.values

        if USE_ANALYZER:
            analyzer.generate_all_plots(val_df, dataset_name="04_validation_processed")

        # Evaluate — inject anomaly labels for metrics only
        val_results = pipeline.evaluate(models, val_df, tag="validation", inject_labels=True)

        if "lstm_autoencoder" in models:
            lstm_val = pipeline.evaluate_lstm(
                models["lstm_autoencoder"], val_df, tag="validation", inject_labels=True
            )
            val_results["lstm_autoencoder"] = lstm_val

except Exception as e:
    print(f"  Validation failed: {e}")
    import traceback; traceback.print_exc()

# ===================== STEP 4 — CROSS-DOMAIN TEST =====================
mall_results = {}
if TEST_DATASET == "mall" and MALL_PATH.exists():
    print("\n" + "=" * 60)
    print("STEP 4: CROSS-DOMAIN TEST (Mall dataset)")
    print("NOTE: Training thresholds applied unchanged")
    print("=" * 60)

    try:
        print("Loading Mall dataset...")
        raw_test_df = data_loader.load_mall_dataset()

        if raw_test_df.empty:
            print("  ⚠️ Mall dataset is empty — trying direct loader...")
            # Try direct loading
            from src.data.mall_loader import MallDatasetLoader
            loader = MallDatasetLoader(str(MALL_PATH))
            raw_test_df = loader.load_mall_dataset(zone_grid_size=(3, 3), time_window=5)

        if raw_test_df.empty:
            print("  ❌ Mall dataset could not be loaded")
        else:
            print(f"  ✅ Loaded {len(raw_test_df):,} Mall records")
            print(f"  Mall data columns: {raw_test_df.columns.tolist()}")
            print(f"  Mall zone IDs: {raw_test_df['zone_id'].unique()}")

            if USE_ANALYZER:
                analyzer.generate_all_plots(raw_test_df, dataset_name="05_mall_raw")

            zone_id_bk = raw_test_df['zone_id'].copy() if 'zone_id' in raw_test_df.columns else None
            test_df = preprocessing.transform(raw_test_df.copy())
            if zone_id_bk is not None and 'zone_id' not in test_df.columns:
                test_df['zone_id'] = zone_id_bk.values

            if USE_ANALYZER:
                analyzer.generate_all_plots(test_df, dataset_name="06_mall_processed")

            print(f"  Evaluating models on Mall dataset...")
            mall_results = pipeline.evaluate(models, test_df, tag="mall", inject_labels=True)

            if "lstm_autoencoder" in models:
                lstm_mall = pipeline.evaluate_lstm(
                    models["lstm_autoencoder"], test_df, tag="mall", inject_labels=True
                )
                mall_results["lstm_autoencoder"] = lstm_mall

    except Exception as e:
        print(f"  Mall testing failed: {e}")
        import traceback; traceback.print_exc()
else:
    print("\n" + "=" * 60)
    print("STEP 4: CROSS-DOMAIN TEST SKIPPED")
    print("Mall dataset not available")
    print("=" * 60)

# ===================== CROSS-DATASET SUMMARY TABLE =====================
print("\n" + "=" * 60)
print("CROSS-DATASET GENERALISATION SUMMARY")
print("(This table is what reviewers look for)")
print("=" * 60)
print(f"{'Model':<25} {'Val F1':>8} {'Val AUC':>9} {'Mall F1':>8} {'Mall AUC':>9}")
print("-" * 60)

all_model_names = set(list(val_results.keys()) + list(mall_results.keys()))
for name in sorted(all_model_names):
    vf1  = val_results.get(name, {}).get('f1_score', float('nan'))
    vauc = val_results.get(name, {}).get('roc_auc',  float('nan'))
    mf1  = mall_results.get(name, {}).get('f1_score', float('nan'))
    mauc = mall_results.get(name, {}).get('roc_auc',  float('nan'))

    # Format with better display
    vf1_str = f"{vf1:.4f}" if not np.isnan(vf1) else "N/A"
    vauc_str = f"{vauc:.4f}" if not np.isnan(vauc) else "N/A"
    mf1_str = f"{mf1:.4f}" if not np.isnan(mf1) else "N/A"
    mauc_str = f"{mauc:.4f}" if not np.isnan(mauc) else "N/A"

    print(f"  {name:<23} {vf1_str:>8} {vauc_str:>9} {mf1_str:>8} {mauc_str:>9}")

print("=" * 60)

# ===================== MODEL PERFORMANCE RANKING =====================
if val_results:
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE RANKING (Validation F1 Score)")
    print("=" * 60)

    # Sort models by F1 score
    sorted_models = sorted(val_results.items(),
                          key=lambda x: x[1].get('f1_score', 0),
                          reverse=True)

    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        f1 = metrics.get('f1_score', 0)
        auc = metrics.get('roc_auc', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)

        print(f"\n{rank}. {model_name.upper()}")
        print(f"   F1 Score: {f1:.4f} | AUC: {auc:.4f}")
        print(f"   Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Recommendation
    best_model = sorted_models[0][0] if sorted_models else None
    if best_model:
        print(f"\n🎯 Recommended Model: {best_model.upper()}")
        print(f"   Best F1 Score: {sorted_models[0][1].get('f1_score', 0):.4f}")

# ===================== FINAL SUMMARY =====================
print("\n" + "🎉 " * 20)
print("TRAINING AND EVALUATION COMPLETE")
print("🎉 " * 20)

print(f"\n  Models   → {MODELS_DIR}/")
print(f"  Results  → {RESULTS_DIR}/")
print(f"  Plots    → {PLOTS_DIR}/")
if USE_ANALYZER:
    print(f"  Analysis → {ANALYSIS_DIR}/")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("""
1. Start the backend API server:
   python main.py

2. In a new terminal, start the dashboard:
   streamlit run dashboard.py

3. Load a model in the dashboard sidebar

4. Upload a video and start monitoring

5. For Mall dataset testing, ensure:
   - mall_gt.mat is in data/raw/mall_dataset/
   - frames/ directory contains the frame images
   
6. To view results:
   - Check results/validation_results.csv
   - Open results/plots/validation/ for visualizations
""")

# ===================== HELPER FUNCTION FOR QUICK TEST =====================
def quick_test():
    """Quick test to verify models are working"""
    print("\n" + "=" * 60)
    print("QUICK TEST: Verifying trained models")
    print("=" * 60)

    import joblib
    import numpy as np

    models_dir = Path("scripts/models")
    model_files = list(models_dir.glob("*.pkl"))

    if not model_files:
        print("No model files found!")
        return

    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        print(f"  - {model_file.name}")

        # Try loading each model
        try:
            model_data = joblib.load(model_file)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data

            print(f"    ✅ Loaded successfully")
            if hasattr(model, 'predict'):
                # Test with random data
                test_data = np.random.randn(1, 10)
                try:
                    pred = model.predict(test_data)
                    print(f"    ✅ Prediction test passed")
                except:
                    print(f"    ⚠️ Prediction test failed (may need specific input shape)")
        except Exception as e:
            print(f"    ❌ Failed to load: {e}")

# Run quick test
quick_test()

print("\n" + "✅ " * 20)
print("Setup complete! You can now start the backend and dashboard.")
print("✅ " * 20)