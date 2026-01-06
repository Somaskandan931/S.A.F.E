"""
Dataset Loader for S.A.F.E - FIXED VERSION
Works with preprocessed ETH-UCY CSV files and Mall dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta


class DatasetManager:
    """
    Dataset Manager for S.A.F.E
    Loads preprocessed ETH-UCY CSV files and Mall dataset
    """

    def __init__(self, config_path: str = None):
        """Initialize dataset manager"""
        from src.config.default_config import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG.copy()

        # Use absolute paths for your datasets
        self.raw_path = Path(r"C:\Users\somas\PycharmProjects\S.A.F.E\data\raw\eth_ucy")
        self.mall_path = Path(r"C:\Users\somas\PycharmProjects\S.A.F.E\data\raw\mall_dataset")
        self.processed_path = Path(self.config['data']['processed_data_path'])

        # Create directories
        for path in [self.raw_path, self.processed_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Cache for loaded datasets
        self.dataset_cache = {}

        # Scan for available files
        self.available_files = self._scan_files()

    def _scan_files(self) -> Dict[str, Path]:
        """Scan directory for available CSV files"""
        files = {}

        if not self.raw_path.exists():
            print(f"⚠️ Data directory not found: {self.raw_path}")
            return files

        # Find all CSV files
        csv_files = list(self.raw_path.glob("*.csv"))

        print(f"\n{'='*60}")
        print(f"Scanning {self.raw_path}")
        print(f"{'='*60}")
        print(f"Found {len(csv_files)} CSV files:")

        for csv_file in csv_files:
            # Extract scene name from filename
            name = csv_file.stem.lower()

            if 'eth' in name and 'hotel' not in name:
                scene = 'eth'
            elif 'hotel' in name:
                scene = 'hotel'
            elif 'univ' in name or 'student' in name:
                scene = 'univ'
            elif 'zara' in name:
                if '1' in name or 'zara1' in name:
                    scene = 'zara1'
                elif '2' in name or 'zara2' in name:
                    scene = 'zara2'
                else:
                    scene = 'zara'
            else:
                scene = csv_file.stem

            files[scene] = csv_file
            size_kb = csv_file.stat().st_size / 1024
            print(f"  ✓ {scene:10s} -> {csv_file.name} ({size_kb:.2f} KB)")

        print(f"{'='*60}\n")
        return files

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load dataset from CSV file (NO HEADER format)

        Args:
            dataset_name: Name of dataset (eth, hotel, univ, zara1, zara2)

        Returns:
            DataFrame with trajectory data
        """
        # Check cache
        if dataset_name in self.dataset_cache:
            print(f"Using cached {dataset_name}")
            return self.dataset_cache[dataset_name]

        if dataset_name not in self.available_files:
            available = list(self.available_files.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found!\n"
                f"Available datasets: {available}"
            )

        try:
            file_path = self.available_files[dataset_name]
            print(f"\nLoading {dataset_name} from {file_path.name}...")

            # Load CSV WITHOUT header (standard ETH-UCY format)
            df = pd.read_csv(
                file_path,
                header=None,
                names=['frame_id', 'pedestrian_id', 'pos_x', 'pos_y']
            )

            # Convert to appropriate types
            df['frame_id'] = df['frame_id'].astype(int)
            df['pedestrian_id'] = df['pedestrian_id'].astype(int)
            df['pos_x'] = df['pos_x'].astype(float)
            df['pos_y'] = df['pos_y'].astype(float)

            # Add dataset label
            df['dataset'] = dataset_name

            # Cache it
            self.dataset_cache[dataset_name] = df

            print(f"✓ Loaded {len(df):,} records")
            print(f"  Unique pedestrians: {df['pedestrian_id'].nunique()}")
            print(f"  Frame range: {df['frame_id'].min()} - {df['frame_id'].max()}")

            return df

        except Exception as e:
            raise RuntimeError(f"Error loading {dataset_name}: {e}")

    def compute_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocities and speeds for trajectories"""
        df = df.copy()
        df = df.sort_values(['pedestrian_id', 'frame_id'])

        # Compute velocities
        df['vel_x'] = df.groupby('pedestrian_id')['pos_x'].diff()
        df['vel_y'] = df.groupby('pedestrian_id')['pos_y'].diff()

        # Compute speed and direction
        df['speed'] = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
        df['direction'] = np.arctan2(df['vel_y'], df['vel_x'])

        # Scale by frame rate (2.5 fps for ETH/UCY)
        fps = 2.5
        df['speed'] = df['speed'] * fps
        df['vel_x'] = df['vel_x'] * fps
        df['vel_y'] = df['vel_y'] * fps

        # Fill NaN for first frame of each agent
        df['vel_x'] = df['vel_x'].fillna(0)
        df['vel_y'] = df['vel_y'].fillna(0)
        df['speed'] = df['speed'].fillna(0)
        df['direction'] = df['direction'].fillna(0)

        return df

    def aggregate_to_zones(self, df: pd.DataFrame, zone_size: float = 5.0,
                          time_window: int = 5) -> pd.DataFrame:
        """
        Aggregate trajectory data to zone-based format

        Args:
            df: DataFrame with trajectory data
            zone_size: Size of each zone in meters
            time_window: Number of frames per time window

        Returns:
            Zone-aggregated DataFrame
        """
        df = df.copy()

        # Create zones based on spatial position
        df['zone_x'] = (df['pos_x'] / zone_size).astype(int)
        df['zone_y'] = (df['pos_y'] / zone_size).astype(int)

        # Create numeric zone_id (CRITICAL for LSTM)
        df['zone_id'] = df['zone_y'] * 100 + df['zone_x']  # Numeric encoding

        # Create time windows
        df['time_window'] = df['frame_id'] // time_window

        # Compute velocities if not present
        if 'speed' not in df.columns:
            df = self.compute_velocities(df)

        # Aggregate by zone and time window
        aggregated = df.groupby(['time_window', 'zone_id', 'dataset']).agg({
            'pedestrian_id': 'nunique',
            'speed': ['mean', 'std'],
            'direction': 'std',
            'pos_x': 'mean',
            'pos_y': 'mean',
            'frame_id': 'min'
        }).reset_index()

        # Flatten column names
        aggregated.columns = [
            'time_window', 'zone_id', 'dataset',
            'footfall_count', 'velocity_mean', 'velocity_std',
            'direction_variance', 'zone_center_x', 'zone_center_y',
            'frame_id'
        ]

        # Fill NaN values
        aggregated['velocity_mean'] = aggregated['velocity_mean'].fillna(0)
        aggregated['velocity_std'] = aggregated['velocity_std'].fillna(0)
        aggregated['direction_variance'] = aggregated['direction_variance'].fillna(0)

        # Create timestamps
        fps = 2.5
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        aggregated['timestamp'] = aggregated['frame_id'].apply(
            lambda x: base_time + timedelta(seconds=x / fps)
        )

        # Calculate density
        zone_area = zone_size * zone_size
        aggregated['density'] = aggregated['footfall_count'] / zone_area

        print(f"  Aggregated to {len(aggregated):,} zone records")
        print(f"  Unique zones: {aggregated['zone_id'].nunique()}")
        print(f"  Zone IDs (sample): {sorted(aggregated['zone_id'].unique())[:10]}")

        return aggregated

    def load_trajectory_data(self, datasets: List[str] = ['eth', 'hotel'],
                           zone_size: float = 5.0, time_window: int = 5) -> pd.DataFrame:
        """
        Load and aggregate multiple datasets

        Args:
            datasets: List of dataset names
            zone_size: Spatial zone size in meters
            time_window: Time window in frames

        Returns:
            Combined and aggregated DataFrame
        """
        all_data = []

        for dataset_name in datasets:
            try:
                print(f"\n{'='*60}")
                print(f"Processing {dataset_name}...")
                print('='*60)

                # Load raw data
                raw_data = self.load_dataset(dataset_name)

                # Compute velocities
                raw_data = self.compute_velocities(raw_data)

                # Aggregate to zones
                aggregated = self.aggregate_to_zones(
                    raw_data, zone_size=zone_size, time_window=time_window
                )

                all_data.append(aggregated)

            except Exception as e:
                print(f"✗ Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_data:
            raise ValueError("No datasets loaded successfully")

        # Combine all datasets
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['dataset', 'timestamp', 'zone_id'])

        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"{'='*60}")
        print(f"Total records: {len(combined):,}")
        print(f"Datasets: {combined['dataset'].unique().tolist()}")
        print(f"Unique zones: {combined['zone_id'].nunique()}")
        print(f"Zone_id type: {combined['zone_id'].dtype}")
        print('='*60)

        return combined

    def load_eth_ucy(self, scenes: List[str], zone_size: float = 5.0,
                     time_window: int = 5) -> pd.DataFrame:
        """Load ETH-UCY datasets (pipeline adapter method)"""
        return self.load_trajectory_data(
            datasets=scenes, zone_size=zone_size, time_window=time_window
        )

    def load_mall_dataset(self) -> pd.DataFrame:
        """
        Load Mall Dataset using MallDatasetLoader

        Returns:
            Zone-level aggregated DataFrame compatible with ETH/UCY format
        """
        from src.data.mall_loader import MallDatasetLoader

        print("\n" + "=" * 60)
        print("LOADING MALL DATASET")
        print("=" * 60)

        try:
            loader = MallDatasetLoader(str(self.mall_path))
            df = loader.load_mall_dataset(zone_grid_size=(3, 3), time_window=5)

            if df.empty:
                print("⚠️ Mall dataset is empty")
                return pd.DataFrame()

            # Ensure zone_id is numeric (CRITICAL)
            if df['zone_id'].dtype == 'object':
                print("⚠️ Converting zone_id to numeric...")
                zone_mapping = {zone: idx for idx, zone in enumerate(df['zone_id'].unique())}
                df['zone_id'] = df['zone_id'].map(zone_mapping)

            print(f"✅ Mall dataset loaded successfully")
            print(f"  Records: {len(df)}")
            print(f"  Unique zones: {df['zone_id'].nunique()}")
            print(f"  Zone_id type: {df['zone_id'].dtype}")
            print(f"  Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            print(f"❌ Failed to load Mall dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_available_datasets(self) -> Dict[str, bool]:
        """Get list of available datasets"""
        return {name: True for name in self.available_files.keys()}

    def save_processed(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        filepath = self.processed_path / filename
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to {filepath}")
        print(f"  Size: {filepath.stat().st_size / (1024*1024):.2f} MB")