"""
Quick test to verify ETH-UCY dataset
Fixed path resolution for Windows
"""
import pandas as pd
from pathlib import Path
import sys
import os

# Get project root directory (parent of scripts folder)
if '__file__' in globals():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
else:
    project_root = Path.cwd()

print("=" * 60)
print("ETH-UCY Dataset Verification")
print("=" * 60)
print(f"\nProject Root: {project_root}")
print(f"Current Working Dir: {Path.cwd()}")

# Set correct data directory
data_dir = project_root / "data" / "raw" / "eth_ucy"
print(f"Looking in: {data_dir}")
print(f"Directory exists: {data_dir.exists()}")

if data_dir.exists():
    print(f"\nContents of {data_dir}:")
    for item in data_dir.iterdir():
        print(f"  - {item.name} ({item.stat().st_size / 1024:.2f} KB)")

# Find all CSV files
csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []

print(f"\n{'='*60}")
if not csv_files:
    print("❌ NO CSV FILES FOUND!")
    print(f"\nPlease check:")
    print(f"1. Files exist in: {data_dir}")
    print(f"2. Run from project root: C:\\Users\\somas\\PycharmProjects\\S.A.F.E")

    # Try alternative locations
    print(f"\n🔍 Searching in alternative locations...")
    alternative_paths = [
        Path.cwd() / "data" / "raw" / "eth_ucy",
        Path("C:/Users/somas/PycharmProjects/S.A.F.E/data/raw/eth_ucy"),
        project_root / "external" / "ETH-UCY-Preprocessing"
    ]

    for alt_path in alternative_paths:
        if alt_path.exists():
            alt_csvs = list(alt_path.glob("*.csv"))
            if alt_csvs:
                print(f"\n✓ Found {len(alt_csvs)} CSV files in: {alt_path}")
                csv_files = alt_csvs
                data_dir = alt_path
                break
else:
    print(f"✓ FOUND {len(csv_files)} CSV FILES")

print(f"{'='*60}\n")

# Process each CSV file
if csv_files:
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*60}")
        print(f"File {i}/{len(csv_files)}: {csv_file.name}")
        print(f"{'='*60}")

        try:
            # Load CSV
            df = pd.read_csv(csv_file)

            print(f"✓ Successfully loaded!")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Size: {csv_file.stat().st_size / 1024:.2f} KB")

            # Show columns
            print(f"\n  Column names:")
            for col in df.columns:
                print(f"    - {col}")

            # Show basic stats
            print(f"\n  Data types:")
            for col, dtype in df.dtypes.items():
                print(f"    - {col}: {dtype}")

            # Show sample data
            print(f"\n  First 3 rows:")
            print(df.head(3).to_string(index=False))

            # Check for key columns
            key_columns = ['pedestrian_id', 'person_id', 'agent_id', 'x', 'y', 'X', 'Y', 'frame', 'frame_id']
            found_cols = [col for col in key_columns if col in df.columns]

            if found_cols:
                print(f"\n  ✓ Found key columns: {', '.join(found_cols)}")

                # Count unique pedestrians
                for ped_col in ['pedestrian_id', 'person_id', 'agent_id']:
                    if ped_col in df.columns:
                        n_peds = df[ped_col].nunique()
                        print(f"    - Unique pedestrians: {n_peds}")
                        break

                # Count frames
                for frame_col in ['frame', 'frame_id', 'timestep']:
                    if frame_col in df.columns:
                        n_frames = df[frame_col].nunique()
                        frame_range = f"{df[frame_col].min()} - {df[frame_col].max()}"
                        print(f"    - Frame range: {frame_range} ({n_frames} frames)")
                        break

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            import traceback
            traceback.print_exc()

print(f"\n{'='*60}")
print("VERIFICATION COMPLETE!")
print(f"{'='*60}")

if csv_files:
    print(f"\n✅ SUCCESS! Found and verified {len(csv_files)} datasets")
    print(f"\nDataset location: {data_dir}")
    print(f"\nNext steps:")
    print(f"1. Create eth_ucy_loader.py (see guide)")
    print(f"2. Integrate with your data pipeline")
    print(f"3. Train models on this data")
else:
    print(f"\n⚠️  No datasets found. Please run setup script:")
    print(f"   .\\setup_datasets.ps1")