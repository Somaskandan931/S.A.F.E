"""
Mall Dataset Loader - Complete for your dataset structure
Handles mall_gt.mat, mall_feat.mat, and perspective_roi.mat
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from typing import Tuple, Optional, Dict
import cv2
import warnings
warnings.filterwarnings("ignore")


class MallDatasetLoader:
    """Load and process Mall Dataset with complete support"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.frames_path = self.dataset_path / "frames"
        self.gt_path = self.dataset_path / "mall_gt.mat"
        self.features_path = self.dataset_path / "mall_feat.mat"
        self.roi_path = self.dataset_path / "perspective_roi.mat"

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Mall dataset not found: {self.dataset_path}")

        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_path}")

        # frames directory is optional - we can work without images
        if not self.frames_path.exists():
            print(f"⚠️  Frames directory not found: {self.frames_path}")
            print("   Will use default dimensions for zone calculation")

        print(f"\n✅ Mall Dataset Loader initialized")
        print(f"   Ground truth: {self.gt_path.name}")
        if self.features_path.exists():
            print(f"   Features: {self.features_path.name}")
        if self.roi_path.exists():
            print(f"   ROI/Perspective: {self.roi_path.name}")

    def load_ground_truth(self) -> dict:
        """
        Load mall_gt.mat file

        The mat file contains a variable 'frame' which is a cell array
        Each cell contains a structure with 'loc' field (pedestrian positions)
        """
        print(f"\n📂 Loading Mall ground truth from {self.gt_path.name}...")
        mat_data = loadmat(str(self.gt_path))

        # Print available variables for debugging
        print(f"   Variables in mat file: {[k for k in mat_data.keys() if not k.startswith('__')]}")

        # The ground truth is stored in 'frame' variable
        if 'frame' in mat_data:
            frames = mat_data['frame'][0]  # Get the cell array
            print(f"✓ Loaded {len(frames)} frames with ground truth")
            return frames
        else:
            raise ValueError(f"mall_gt.mat does not contain 'frame' variable. Available: {list(mat_data.keys())}")

    def load_features(self) -> Optional[np.ndarray]:
        """Load pre-computed features from mall_feat.mat if available"""
        if not self.features_path.exists():
            return None

        print(f"\n📊 Loading features from {self.features_path.name}...")
        mat_data = loadmat(str(self.features_path))

        # Try to find feature variable (often 'feat' or 'features')
        for var_name in ['feat', 'features', 'data', 'X']:
            if var_name in mat_data:
                features = mat_data[var_name]
                print(f"✓ Loaded features: {features.shape}")
                return features

        print(f"   Variables in features file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
        return None

    def load_roi(self) -> Optional[Dict]:
        """Load region of interest and perspective information"""
        if not self.roi_path.exists():
            return None

        print(f"\n🗺️  Loading ROI/Perspective from {self.roi_path.name}...")
        mat_data = loadmat(str(self.roi_path))

        roi_info = {}

        # Look for ROI mask
        for var_name in ['roi', 'mask', 'ROI']:
            if var_name in mat_data:
                roi_info['mask'] = mat_data[var_name]
                print(f"✓ Loaded ROI mask: {roi_info['mask'].shape}")
                break

        # Look for perspective mapping
        for var_name in ['perspective', 'H', 'homography']:
            if var_name in mat_data:
                roi_info['perspective'] = mat_data[var_name]
                print(f"✓ Loaded perspective mapping: {roi_info['perspective'].shape}")
                break

        return roi_info if roi_info else None

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions from frames or use defaults"""
        try:
            if self.frames_path.exists():
                # Try to find first frame
                frame_files = list(self.frames_path.glob("seq_*.jpg")) + \
                             list(self.frames_path.glob("seq_*.png"))
                if frame_files:
                    first_img = cv2.imread(str(frame_files[0]))
                    if first_img is not None:
                        h, w = first_img.shape[:2]
                        print(f"🖼  Image dimensions from frames: {w}x{h}")
                        return w, h
        except Exception as e:
            print(f"   Could not read frame: {e}")

        # Default dimensions (Mall dataset typical size)
        print(f"🖼  Using default image dimensions: 640x480")
        return 640, 480

    def extract_pedestrian_data(self, frames: list,
                                zone_grid_size: Tuple[int, int] = (3, 3),
                                frame_rate: int = 10) -> pd.DataFrame:
        """
        Convert Mall ground truth to pedestrian-level data

        Args:
            frames: Cell array from mall_gt.mat
            zone_grid_size: Number of zones (rows, cols) - default 3x3 grid
            frame_rate: Frames per second for timestamp generation
        """
        records = []

        # Get image dimensions
        img_width, img_height = self.get_image_dimensions()

        # Calculate zone dimensions
        zone_width = img_width / zone_grid_size[1]
        zone_height = img_height / zone_grid_size[0]

        print(f"\n🔲 Creating {zone_grid_size[0]}x{zone_grid_size[1]} zone grid...")
        print(f"   Each zone: {zone_width:.1f}x{zone_height:.1f} pixels")

        total_pedestrians = 0
        frames_with_data = 0

        for frame_idx, frame_data in enumerate(frames):
            # Extract locations from frame_data
            locations = None

            # Different ways to access the data based on mat file structure
            if isinstance(frame_data, np.ndarray):
                if frame_data.dtype.names:
                    # Structured array
                    if 'loc' in frame_data.dtype.names:
                        locations = frame_data['loc'][0, 0]
                elif len(frame_data.shape) == 2 and frame_data.shape[0] == 2:
                    # Direct coordinates (2 x N)
                    locations = frame_data.T
                elif len(frame_data.shape) == 2 and frame_data.shape[1] == 2:
                    # Direct coordinates (N x 2)
                    locations = frame_data
            elif hasattr(frame_data, 'loc'):
                # Object with loc attribute
                locations = frame_data.loc
            elif hasattr(frame_data, '__getitem__') and len(frame_data) > 0:
                # Try first element
                try:
                    if hasattr(frame_data[0], 'loc'):
                        locations = frame_data[0].loc
                except:
                    pass

            # Skip if no locations
            if locations is None:
                continue

            # Ensure locations is a numpy array
            if not isinstance(locations, np.ndarray):
                locations = np.array(locations)

            # Handle shape
            if len(locations.shape) == 1:
                # Single pedestrian
                if len(locations) == 2:
                    locations = locations.reshape(1, 2)
                else:
                    continue
            elif len(locations.shape) == 2:
                if locations.shape[0] == 2 and locations.shape[1] > 0:
                    # Shape is (2, N) - transpose to (N, 2)
                    locations = locations.T
                elif locations.shape[1] != 2:
                    continue
            else:
                continue

            # Skip if no pedestrians
            if len(locations) == 0:
                continue

            frames_with_data += 1

            # Process each pedestrian in this frame
            for ped_idx, (x, y) in enumerate(locations):
                # Ensure x, y are numbers
                x = float(x)
                y = float(y)

                # Skip if outside image bounds
                if x < 0 or x > img_width or y < 0 or y > img_height:
                    continue

                # Determine which zone this pedestrian belongs to
                zone_col = int(x / zone_width)
                zone_row = int(y / zone_height)

                # Clamp to valid zone indices
                zone_col = max(0, min(zone_col, zone_grid_size[1] - 1))
                zone_row = max(0, min(zone_row, zone_grid_size[0] - 1))

                # Create unique zone ID
                zone_id = zone_row * zone_grid_size[1] + zone_col

                records.append({
                    'frame_id': frame_idx,
                    'pedestrian_id': f"mall_{frame_idx}_{ped_idx}",
                    'pos_x': x,
                    'pos_y': y,
                    'zone_id': zone_id,
                    'timestamp': pd.Timestamp('2012-01-01') + pd.Timedelta(seconds=frame_idx / frame_rate),
                    'dataset': 'mall'
                })
                total_pedestrians += 1

        df = pd.DataFrame(records)

        if df.empty:
            print("⚠️ No pedestrian records extracted")
            return df

        print(f"\n✓ Extracted {len(df)} pedestrian records")
        print(f"  - Frames with data: {frames_with_data}/{len(frames)}")
        print(f"  - Total pedestrians: {total_pedestrians}")
        print(f"  - Unique zones: {df['zone_id'].nunique()}")
        print(f"  - Zone distribution:\n{df['zone_id'].value_counts().sort_index()}")

        return df

    def add_synthetic_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic velocity features based on density changes over time

        This creates realistic velocity estimates from crowd density gradients,
        solving the "all zeros" problem that breaks LSTM models.
        """
        print(f"\n🔧 Adding synthetic velocity features...")

        df = df.copy()
        df = df.sort_values(['zone_id', 'frame_id'])  # Sort by zone and frame

        # Initialize velocity columns
        df['velocity_mean'] = 0.0
        df['velocity_std'] = 0.0
        df['direction_variance'] = 0.0

        for zone in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone
            zone_data = df[zone_mask].copy()

            if len(zone_data) < 2:
                continue

            # Calculate velocity from density change
            density_diff = zone_data['density'].diff().fillna(0)

            # Smooth with rolling window
            velocity_raw = density_diff.rolling(3, min_periods=1).mean()

            # Scale to realistic range (ETH/UCY velocity_mean typically 0-2 m/s)
            velocity_scaled = velocity_raw * 0.3

            # Calculate velocity standard deviation
            velocity_std = velocity_scaled.rolling(5, min_periods=1).std().fillna(0)

            # Assign back
            df.loc[zone_mask, 'velocity_mean'] = velocity_scaled.values
            df.loc[zone_mask, 'velocity_std'] = velocity_std.values

        # Calculate direction variance
        df['direction_variance'] = df.groupby('zone_id')['velocity_mean'].transform(
            lambda x: x.rolling(5, min_periods=1).std()
        ).fillna(0)

        # Statistics
        print(f"✓ Synthetic velocity added:")
        print(f"  velocity_mean: μ={df['velocity_mean'].mean():.4f}, σ={df['velocity_mean'].std():.4f}")
        print(f"  velocity_std: μ={df['velocity_std'].mean():.4f}, σ={df['velocity_std'].std():.4f}")
        print(f"  direction_variance: μ={df['direction_variance'].mean():.4f}, σ={df['direction_variance'].std():.4f}")

        # Verify non-zero
        non_zero = (df['velocity_mean'] != 0).sum()
        print(f"  Non-zero velocities: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%)")

        return df

    def aggregate_to_zone_level(self, df: pd.DataFrame, time_window: int = 5) -> pd.DataFrame:
        """
        Aggregate pedestrian-level data to zone-level time windows
        """
        print(f"\n🔄 Aggregating to zone-level (window={time_window} frames)...")

        if df.empty:
            return pd.DataFrame()

        # Create time windows
        df['time_window'] = df['frame_id'] // time_window

        # Aggregate per zone and time window
        zone_agg = df.groupby(['zone_id', 'time_window']).agg(
            footfall_count=('pedestrian_id', 'count'),
            zone_center_x=('pos_x', 'mean'),
            zone_center_y=('pos_y', 'mean'),
            frame_id=('frame_id', 'min'),
            timestamp=('timestamp', 'first')
        ).reset_index()

        # Calculate density (normalized footfall count)
        max_footfall = zone_agg['footfall_count'].max()
        if max_footfall > 0:
            zone_agg['density'] = zone_agg['footfall_count'] / max_footfall * 100
        else:
            zone_agg['density'] = 0

        # Sort
        zone_agg = zone_agg.sort_values(['zone_id', 'time_window'])

        # Add dataset label
        zone_agg['dataset'] = 'mall'

        # Initialize velocity columns
        zone_agg['velocity_mean'] = 0.0
        zone_agg['velocity_std'] = 0.0
        zone_agg['direction_variance'] = 0.0

        # Add synthetic velocity
        zone_agg = self.add_synthetic_velocity(zone_agg)

        print(f"\n✓ Aggregated to {len(zone_agg)} zone records")
        print(f"  Unique zones: {zone_agg['zone_id'].nunique()}")
        print(f"  Time windows: {zone_agg['time_window'].nunique()}")
        print(f"  Density range: [{zone_agg['density'].min():.2f}, {zone_agg['density'].max():.2f}]")

        return zone_agg

    def load_mall_dataset(self, zone_grid_size: Tuple[int, int] = (3, 3),
                         time_window: int = 5) -> pd.DataFrame:
        """
        Load Mall dataset with synthetic velocity features

        Args:
            zone_grid_size: Number of zones (rows, cols)
            time_window: Number of frames to aggregate per time window

        Returns:
            DataFrame with zone-level aggregated data
        """
        print("\n" + "=" * 60)
        print("LOADING MALL DATASET")
        print("=" * 60)

        # Load ground truth
        frames = self.load_ground_truth()

        # Load optional features and ROI
        features = self.load_features()
        roi = self.load_roi()

        # Extract pedestrian data with zone assignment
        df_pedestrians = self.extract_pedestrian_data(frames, zone_grid_size=zone_grid_size)

        if df_pedestrians.empty:
            print("❌ Mall dataset contains no usable data")
            return pd.DataFrame()

        # Aggregate to zone level
        df_zones = self.aggregate_to_zone_level(df_pedestrians, time_window=time_window)

        if df_zones.empty:
            print("❌ Failed to aggregate Mall data")
            return pd.DataFrame()

        # Add features if available
        if features is not None:
            print(f"\n📊 Features available but not yet integrated")
            # You can add feature integration logic here

        print("\n" + "=" * 60)
        print("✅ MALL DATASET LOADED")
        print("=" * 60)
        print(f"Total zone records: {len(df_zones)}")
        print(f"Unique zones: {df_zones['zone_id'].nunique()}")
        print(f"Columns: {df_zones.columns.tolist()}")
        print("=" * 60 + "\n")

        return df_zones


if __name__ == "__main__":
    # Test the loader
    loader = MallDatasetLoader("data/raw/mall_dataset")
    df_mall = loader.load_mall_dataset(zone_grid_size=(3, 3), time_window=5)

    if not df_mall.empty:
        print("\nDataFrame Info:")
        print(df_mall.info())

        print("\nSample data (first 10 rows):")
        print(df_mall.head(10))

        print("\nVelocity Statistics:")
        print(df_mall[['velocity_mean', 'velocity_std', 'direction_variance']].describe())

        print("\nDensity Statistics by Zone:")
        print(df_mall.groupby('zone_id')['density'].describe())