"""
Enhanced Training Script for S.A.F.E
Train on ETH/UCY datasets with zone-aware LSTM
Test on Mall dataset
Includes comprehensive data analysis and visualization
"""

from src.pipeline.safe_pipeline import SAFEPipeline
from src.data.data_loader import DatasetManager
from src.data.preprocessing import DataPreprocessor
from pathlib import Path
import src.config.default_config as config
# Import the data analyzer
import sys
import os

# Add src to path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("\n" + "🚀 " * 35)
print("S.A.F.E TRAINING PIPELINE WITH COMPREHENSIVE VISUALIZATION")
print("STRATEGY: Train on ETH/UCY | Test on Mall")
print("🚀 " * 35)

# ===================== CONFIGURATION =====================
TRAIN_SCENES = ["eth", "hotel", "univ"]
VALIDATION_SCENES = ["zara1", "zara2"]
TEST_DATASET = "mall"

# Output directories
ANALYSIS_DIR = Path("results/analysis")
PLOTS_DIR = Path("results/plots")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

# Create all directories
for directory in [ANALYSIS_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Mall dataset path (adjust if needed)
MALL_PATH = Path("data/raw/mall_dataset")

# Check if Mall dataset exists
print("\n" + "=" * 60)
print("CHECKING MALL DATASET")
print("=" * 60)

if MALL_PATH.exists():
    print(f"✓ Mall directory found: {MALL_PATH}")

    # Check for required files
    mall_gt = MALL_PATH / "mall_gt.mat"
    frames_dir = MALL_PATH / "frames"

    if mall_gt.exists():
        print(f"✓ Ground truth file found: {mall_gt}")
    else:
        print(f"✗ Ground truth file missing: {mall_gt}")
        print("\n⚠️  Cannot test on Mall dataset without mall_gt.mat")
        TEST_DATASET = None

    if frames_dir.exists():
        frame_count = len(list(frames_dir.glob("*.jpg")))
        print(f"✓ Frames directory found: {frames_dir}")
        print(f"  Contains {frame_count} frame images")
    else:
        print(f"⚠️  Frames directory not found: {frames_dir}")
        print("  (Optional - not required for zone-level analysis)")
else:
    print(f"✗ Mall directory not found: {MALL_PATH}")
    print("\n⚠️  Mall dataset testing will be skipped")
    print("\nTo enable Mall testing:")
    print("  1. Download Mall dataset from: http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html")
    print("  2. Extract to: data/raw/mall_dataset/")
    print("  3. Ensure mall_gt.mat exists in that directory")
    TEST_DATASET = None

# ===================== INITIALIZE DATA ANALYZER =====================
print("\n" + "=" * 60)
print("INITIALIZING DATA ANALYZER")
print("=" * 60)

try:
    # Try to import the data analyzer
    from data_analysis_plots import DataAnalyzer

    analyzer = DataAnalyzer(output_dir=str(ANALYSIS_DIR))
    print(f"✅ Data Analyzer initialized")
    print(f"   Output directory: {ANALYSIS_DIR}")
    USE_ANALYZER = True

except ImportError as e:
    print(f"⚠️  Data Analyzer not found: {e}")
    print("   Skipping data distribution plots")
    USE_ANALYZER = False

# ===================== INITIALIZE PIPELINE =====================
print("\n" + "=" * 60)
print("INITIALIZING PIPELINE")
print("=" * 60)

# Initialize components
data_loader = DatasetManager()
preprocessing = DataPreprocessor()

train_data = data_loader.load_trajectory_data(
    datasets=config.DATASETS["training"],
    zone_size=config.GRID_SIZE[0],
    time_window=config.TIME_WINDOW
)

# Create pipeline
pipeline = SAFEPipeline(
    data_loader=data_loader,
    preprocessing=preprocessing,
    train_scenes=TRAIN_SCENES,
    validation_scenes=VALIDATION_SCENES,
    test_dataset=TEST_DATASET if TEST_DATASET else "none",
    mall_path=str(MALL_PATH),
    model_output_dir=str(MODELS_DIR),
    results_output_dir=str(RESULTS_DIR),
    plot_output_dir=str(PLOTS_DIR)
)

print(f"✅ Pipeline initialized")
print(f"  Training scenes: {TRAIN_SCENES}")
print(f"  Validation scenes: {VALIDATION_SCENES}")
print(f"  Test dataset: {TEST_DATASET if TEST_DATASET else 'None (Mall not available)'}")

# ===================== LOAD AND ANALYZE TRAINING DATA =====================
print("\n" + "=" * 60)
print("STEP 1: LOADING TRAINING DATA")
print("=" * 60)

try:
    # Load raw training data
    raw_train_df = data_loader.load_eth_ucy(TRAIN_SCENES)

    print(f"✅ Loaded training data: {len(raw_train_df)} records")

    # Generate data distribution plots for RAW data
    if USE_ANALYZER:
        print("\n" + "=" * 60)
        print("ANALYZING RAW TRAINING DATA")
        print("=" * 60)
        analyzer.generate_all_plots(raw_train_df, dataset_name="01_training_raw")

    # Preprocess data
    zone_id_backup = raw_train_df['zone_id'].copy() if 'zone_id' in raw_train_df.columns else None
    train_df = preprocessing.fit_transform(raw_train_df.copy())

    if zone_id_backup is not None and 'zone_id' not in train_df.columns:
        train_df['zone_id'] = zone_id_backup.values

    # Generate data distribution plots for PROCESSED data
    if USE_ANALYZER:
        print("\n" + "=" * 60)
        print("ANALYZING PROCESSED TRAINING DATA")
        print("=" * 60)
        analyzer.generate_all_plots(train_df, dataset_name="02_training_processed")

except Exception as e:
    print(f"✗ Failed to load training data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===================== TRAIN MODELS =====================
print("\n" + "=" * 60)
print("STEP 2: TRAINING MODELS")
print("=" * 60)

try:
    models = pipeline.train_models(train_df)
    print(f"✅ Trained {len(models)} models")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===================== VALIDATION WITH PLOTS =====================
print("\n" + "=" * 60)
print("STEP 3: VALIDATION WITH ANALYSIS")
print("=" * 60)

try:
    # Load validation data
    raw_val_df = data_loader.load_eth_ucy(VALIDATION_SCENES)
    print(f"✅ Loaded validation data: {len(raw_val_df)} records")

    # Generate data distribution plots for validation data
    if USE_ANALYZER:
        print("\n" + "=" * 60)
        print("ANALYZING VALIDATION DATA")
        print("=" * 60)
        analyzer.generate_all_plots(raw_val_df, dataset_name="03_validation_raw")

    # Preprocess validation data
    zone_id_backup = raw_val_df['zone_id'].copy() if 'zone_id' in raw_val_df.columns else None
    val_df = preprocessing.transform(raw_val_df.copy())

    if zone_id_backup is not None and 'zone_id' not in val_df.columns:
        val_df['zone_id'] = zone_id_backup.values

    # Generate plots for processed validation data
    if USE_ANALYZER:
        analyzer.generate_all_plots(val_df, dataset_name="04_validation_processed")

    # Evaluate models with plots
    print("\n" + "=" * 60)
    print("EVALUATING MODELS ON VALIDATION DATA")
    print("=" * 60)

    pipeline.evaluate_with_plots(models, val_df, tag="validation", use_adaptive_threshold=True)

    if "lstm_autoencoder" in models:
        pipeline.evaluate_lstm_with_plots(models["lstm_autoencoder"], val_df,
                                        tag="validation", use_adaptive_threshold=True)

except Exception as e:
    print(f"⚠️ Validation failed: {e}")
    import traceback
    traceback.print_exc()

# ===================== TEST ON MALL WITH PLOTS =====================
if TEST_DATASET == "mall":
    print("\n" + "=" * 60)
    print("STEP 4: TESTING ON MALL WITH ANALYSIS")
    print("=" * 60)

    try:
        # Load Mall test data
        raw_test_df = data_loader.load_mall_dataset()

        if raw_test_df.empty:
            print("⚠️ Mall dataset is empty")
        else:
            print(f"✅ Loaded Mall test data: {len(raw_test_df)} records")

            # Generate data distribution plots for Mall data
            if USE_ANALYZER:
                print("\n" + "=" * 60)
                print("ANALYZING MALL TEST DATA (RAW)")
                print("=" * 60)
                analyzer.generate_all_plots(raw_test_df, dataset_name="05_mall_raw")

            # Preprocess Mall data
            zone_id_backup = raw_test_df['zone_id'].copy() if 'zone_id' in raw_test_df.columns else None
            test_df = preprocessing.transform(raw_test_df.copy())

            if zone_id_backup is not None and 'zone_id' not in test_df.columns:
                test_df['zone_id'] = zone_id_backup.values

            # Generate plots for processed Mall data
            if USE_ANALYZER:
                analyzer.generate_all_plots(test_df, dataset_name="06_mall_processed")

            # Evaluate models with plots
            print("\n" + "=" * 60)
            print("EVALUATING MODELS ON MALL DATA")
            print("=" * 60)
            print("🎯 Using adaptive thresholding for cross-domain testing...")

            pipeline.evaluate_with_plots(models, test_df, tag="mall", use_adaptive_threshold=True)

            if "lstm_autoencoder" in models:
                pipeline.evaluate_lstm_with_plots(models["lstm_autoencoder"], test_df,
                                                tag="mall", use_adaptive_threshold=True)

    except Exception as e:
        print(f"✗ Mall testing failed: {e}")
        import traceback
        traceback.print_exc()

# ===================== SUMMARY =====================
print("\n" + "🎉 " * 35)
print("TRAINING AND ANALYSIS COMPLETE!")
print("🎉 " * 35)

print("\n" + "=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)

print(f"\n📊 Data Analysis Plots:")
print(f"   {ANALYSIS_DIR}/")
if USE_ANALYZER:
    analysis_files = list(ANALYSIS_DIR.glob("*.png"))
    for f in sorted(analysis_files):
        print(f"   ├── {f.name}")
    print(f"   Total: {len(analysis_files)} plots")
else:
    print("   (Data analyzer not available)")

print(f"\n📈 Model Evaluation Plots:")
print(f"   {PLOTS_DIR}/")
plot_subdirs = [d for d in PLOTS_DIR.iterdir() if d.is_dir()]
for subdir in sorted(plot_subdirs):
    print(f"   ├── {subdir.name}/")
    plot_files = list(subdir.glob("*.png"))
    for f in sorted(plot_files):
        print(f"   │   ├── {f.name}")
    print(f"   │   Total: {len(plot_files)} plots")

print(f"\n🤖 Trained Models:")
print(f"   {MODELS_DIR}/")
model_files = list(MODELS_DIR.glob("*.pkl"))
for f in sorted(model_files):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"   ├── {f.name} ({size_mb:.2f} MB)")
print(f"   Total: {len(model_files)} models")

print(f"\n📋 Evaluation Results:")
print(f"   {RESULTS_DIR}/")
result_files = list(RESULTS_DIR.glob("*.csv"))
for f in sorted(result_files):
    print(f"   ├── {f.name}")
print(f"   Total: {len(result_files)} reports")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Review data analysis plots in results/analysis/")
print("2. Review model evaluation plots in results/plots/")
print("3. Check evaluation metrics in results/*.csv")
print("4. Use trained models in models/ for deployment")
print("=" * 60)