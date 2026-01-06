"""
Training Script for S.A.F.E
Train on ETH/UCY datasets with zone-aware LSTM
Test on Mall dataset
"""

from src.pipeline.safe_pipeline import SAFEPipeline
from src.data.data_loader import DatasetManager
from src.data.preprocessing import DataPreprocessor
from pathlib import Path

print("\n" + "🚀 " * 35)
print("S.A.F.E TRAINING PIPELINE")
print("STRATEGY: Train on ETH/UCY | Test on Mall")
print("🚀 " * 35)

# ===================== CONFIGURATION =====================
TRAIN_SCENES = ["eth", "hotel", "univ"]
VALIDATION_SCENES = ["zara1", "zara2"]
TEST_DATASET = "mall"

# Mall dataset path (adjust if needed)
MALL_PATH = Path("C:/Users/somas/PycharmProjects/S.A.F.E/data/raw/mall_dataset")

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
        print(f"❌ Ground truth file missing: {mall_gt}")
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
    print(f"❌ Mall directory not found: {MALL_PATH}")
    print("\n⚠️  Mall dataset testing will be skipped")
    print("\nTo enable Mall testing:")
    print("  1. Download Mall dataset from: http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html")
    print("  2. Extract to: data/raw/mall_dataset/")
    print("  3. Ensure mall_gt.mat exists in that directory")
    TEST_DATASET = None

# ===================== INITIALIZE PIPELINE =====================
print("\n" + "=" * 60)
print("INITIALIZING PIPELINE")
print("=" * 60)

# Initialize components
data_loader = DatasetManager()
preprocessing = DataPreprocessor()

# Create pipeline
pipeline = SAFEPipeline(
    data_loader=data_loader,
    preprocessing=preprocessing,
    train_scenes=TRAIN_SCENES,
    validation_scenes=VALIDATION_SCENES,
    test_dataset=TEST_DATASET if TEST_DATASET else "none",
    mall_path=str(MALL_PATH),
    model_output_dir="models",
    results_output_dir="results",
    plot_output_dir="results/plots"
)

print(f"✓ Pipeline initialized")
print(f"  Training scenes: {TRAIN_SCENES}")
print(f"  Validation scenes: {VALIDATION_SCENES}")
print(f"  Test dataset: {TEST_DATASET if TEST_DATASET else 'None (Mall not available)'}")

# ===================== RUN PIPELINE =====================
print("\n" + "=" * 60)
print("STARTING TRAINING PIPELINE")
print("=" * 60)

try:
    pipeline.run()

    print("\n" + "🎉 " * 35)
    print("TRAINING COMPLETE!")
    print("🎉 " * 35)

except Exception as e:
    print("\n" + "❌ " * 35)
    print(f"PIPELINE FAILED: {e}")
    print("❌ " * 35)
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("SCRIPT FINISHED")
print("=" * 60)

