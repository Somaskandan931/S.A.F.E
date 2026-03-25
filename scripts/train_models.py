# train_models.py - FIXED VERSION
"""
S.A.F.E Training Script - FIXED VERSION
===========================================
Proper evaluation with no data leakage
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Ensure project root is on the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

from src.pipeline.safe_pipeline import SAFEPipeline
from src.data.data_loader import DatasetManager
from src.data.preprocessing import DataPreprocessor
import src.config.default_config as config
from src.evaluation.metrics import MetricsCalculator

# ===================== CONFIGURATION =====================
INCLUDE_MALL_IN_TRAINING = False  # False = honest cross-domain test
MALL_TRAIN_SPLIT = 0.7

TRAIN_SCENES = ["eth", "hotel", "univ"]
VALIDATION_SCENES = ["zara1", "zara2"]
TEST_DATASET = "mall"

# Create output directories
ANALYSIS_DIR = Path("results/analysis")
PLOTS_DIR = Path("results/plots")
MODELS_DIR = Path("scripts/models")
RESULTS_DIR = Path("results")
MALL_PATH = project_root / "data" / "raw" / "mall_dataset"

for d in [ANALYSIS_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("S.A.F.E TRAINING SCRIPT - FIXED VERSION")
print("=" * 60)
print(f"Project Root: {project_root}")
print(f"Models Dir: {MODELS_DIR}")
print(f"Results Dir: {RESULTS_DIR}")

# ===================== PRINT TRAINING STRATEGY =====================
print("\n" + "=" * 60)
print("TRAINING STRATEGY")
print("=" * 60)
if INCLUDE_MALL_IN_TRAINING:
    print(f"✅ INCLUDING Mall normal data in training ({MALL_TRAIN_SPLIT*100:.0f}% split)")
    print("   Strategy: Train on ETH/UCY + Mall (normal) → Test on held-out Mall")
else:
    print("❌ EXCLUDING Mall data from training")
    print("   Strategy: Train only on ETH/UCY → Test on Mall")
    print("   This is an HONEST cross-domain generalization test")
print("=" * 60)

# ===================== CHECK DATASETS =====================
print("\n" + "=" * 60)
print("CHECKING DATASETS")
print("=" * 60)

eth_ucy_path = project_root / "data" / "raw" / "eth_ucy"
print(f"\n📁 ETH/UCY Directory: {eth_ucy_path}")
if eth_ucy_path.exists():
    csv_files = list(eth_ucy_path.glob("*.csv"))
    print(f"   Found {len(csv_files)} CSV file(s)")
    for f in csv_files[:5]:
        print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    print(f"   ❌ ETH/UCY directory not found")

print(f"\n📁 Mall Dataset Directory: {MALL_PATH}")
if MALL_PATH.exists():
    gt_file = MALL_PATH / "mall_gt.mat"
    frames_dir = MALL_PATH / "frames"
    if gt_file.exists():
        print(f"   ✅ mall_gt.mat found")
    else:
        print(f"   ❌ mall_gt.mat not found")
    if frames_dir.exists():
        frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        print(f"   ✅ frames directory found with {len(frames)} images")
    else:
        print(f"   ⚠️ frames directory not found")
else:
    print(f"   ⚠️ Mall dataset directory not found")

# ===================== DATA ANALYZER =====================
USE_ANALYZER = False
try:
    from data_analysis_plots import DataAnalyzer
    analyzer = DataAnalyzer(output_dir=str(ANALYSIS_DIR))
    USE_ANALYZER = True
    print("\n  ✅ DataAnalyzer loaded")
except ImportError:
    print("\n  ℹ️ DataAnalyzer not found")

# ===================== INITIALISE COMPONENTS =====================
data_loader = DatasetManager()
preprocessing = DataPreprocessor()

print("\n" + "=" * 60)
print("AVAILABLE DATASETS")
print("=" * 60)
available = data_loader.get_available_datasets()
if available:
    for name, exists in available.items():
        print(f"  {name}: {'✅' if exists else '❌'}")
else:
    print("  No datasets found!")

# ===================== CREATE PIPELINE =====================
pipeline = SAFEPipeline(
    data_loader=data_loader,
    preprocessing=preprocessing,
    train_scenes=TRAIN_SCENES,
    validation_scenes=VALIDATION_SCENES,
    test_dataset=TEST_DATASET,
    mall_path=str(MALL_PATH),
    model_output_dir=str(MODELS_DIR),
    results_output_dir=str(RESULTS_DIR),
    plot_output_dir=str(PLOTS_DIR),
    include_mall_in_training=INCLUDE_MALL_IN_TRAINING,
    mall_train_split=MALL_TRAIN_SPLIT,
)

# ===================== RUN MAIN PIPELINE =====================
try:
    # Run training and validation
    pipeline.run()

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

    # Load results and generate summary
    results_files = list(RESULTS_DIR.glob("*_results.csv"))
    if results_files:
        print(f"\n📊 Results Summary:")
        for rf in results_files:
            df = pd.read_csv(rf)
            print(f"\n  {rf.name}:")
            print(df.to_string(index=False))

    # Generate data analysis plots if analyzer available
    if USE_ANALYZER:
        print("\n" + "=" * 60)
        print("GENERATING DATA ANALYSIS PLOTS")
        print("=" * 60)

        # Load training data for analysis
        raw_train = data_loader.load_eth_ucy(TRAIN_SCENES)
        if not raw_train.empty:
            print("\n📈 ETH/UCY Training Data Analysis:")
            analyzer.generate_all_plots(raw_train, dataset_name="eth_ucy_training")

        # Load validation data
        raw_val = data_loader.load_eth_ucy(VALIDATION_SCENES)
        if not raw_val.empty:
            print("\n📈 Validation Data Analysis:")
            analyzer.generate_all_plots(raw_val, dataset_name="validation")

        # Load Mall dataset
        raw_mall = data_loader.load_mall_dataset()
        if not raw_mall.empty:
            print("\n📈 Mall Dataset Analysis:")
            analyzer.generate_all_plots(raw_mall, dataset_name="mall")

except Exception as e:
    print(f"\n❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

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
print("EVALUATION SUMMARY")
print("=" * 60)
if INCLUDE_MALL_IN_TRAINING:
    print("✅ Models were trained on Mall normal data")
    print("   Test results reflect performance on held-out Mall data")
else:
    print("❌ Models were NOT trained on Mall data")
    print("   Test results reflect cross-domain generalization")

# Display results files
print("\n📁 Generated Files:")
for f in RESULTS_DIR.glob("*"):
    if f.suffix in ['.csv', '.txt', '.png']:
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
for f in PLOTS_DIR.glob("*"):
    if f.suffix == '.png':
        print(f"  {f.name}")
for f in ANALYSIS_DIR.glob("*"):
    if f.suffix == '.png':
        print(f"  {f.name}")

print("\n" + "✅ " * 20)
print("Setup complete! You can now:")
print("  1. Check results/validation_results.csv for model metrics")
print("  2. View plots in results/analysis/ for data understanding")
print("  3. Check results/plots/ for model evaluation plots")
print("  4. Start the backend: python main.py")
print("  5. Start the dashboard: streamlit run dashboard.py")
print("✅ " * 20)