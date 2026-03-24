"""
Dataset Loader for S.A.F.E - UPDATED for your dataset structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class DatasetManager:
    """
    Dataset Manager for S.A.F.E.
    Loads raw ETH-UCY CSV files and the Mall dataset.
    """

    def __init__(self, config_path: str = None):
        try:
            import src.config.default_config as config
            self.config = config
            # Use paths from config
            self.raw_path = getattr(config, 'ETH_UCY_PATH', Path('data/raw/eth_ucy'))
            self.mall_path = getattr(config, 'MALL_PATH', Path('data/raw/mall_dataset'))
            self.processed_path = getattr(config, 'DATA_PATH', Path('data/processed'))
        except ImportError:
            # Fallback
            self.raw_path = Path('data/raw/eth_ucy')
            self.mall_path = Path('data/raw/mall_dataset')
            self.processed_path = Path('data/processed')

        # Create directories if they don't exist
        for p in [self.raw_path, self.processed_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.dataset_cache = {}
        self.available_files = self._scan_files()

    # ------------------------------------------------------------------
    def _scan_files(self) -> Dict[str, Path]:
        """Scan for CSV files in the ETH/UCY directory"""
        files = {}

        if not self.raw_path.exists():
            print(f"  Data directory not found: {self.raw_path}")
            return files

        csv_files = list(self.raw_path.glob("*.csv"))

        if not csv_files:
            print(f"  No CSV files found in {self.raw_path}")
            return files

        print(f"\n{'='*60}")
        print(f"Scanning {self.raw_path}")
        print(f"Found {len(csv_files)} CSV file(s):")

        for csv_file in csv_files:
            name = csv_file.stem.lower()

            # Map filenames to scene names
            if 'eth' in name and 'hotel' not in name:
                scene = 'eth'
            elif 'hotel' in name:
                scene = 'hotel'
            elif 'univ' in name or 'student' in name or 'ucy' in name:
                scene = 'univ'
            elif 'zara' in name:
                if '1' in name or '01' in name:
                    scene = 'zara1'
                elif '2' in name or '02' in name:
                    scene = 'zara2'
                else:
                    scene = 'zara'
            else:
                # Try to infer from filename
                print(f"  Unrecognized file: {csv_file.name}")
                # You can add custom mapping here for your files
                if 'crowd' in name:
                    scene = 'univ'
                elif 'data' in name:
                    scene = 'eth'
                else:
                    scene = csv_file.stem

            files[scene] = csv_file
            print(f"  {scene:10s} → {csv_file.name} ({csv_file.stat().st_size/1024:.1f} KB)")

        print('=' * 60)
        return files

    # ------------------------------------------------------------------
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a specific dataset by name"""
        if dataset_name in self.dataset_cache:
            return self.dataset_cache[dataset_name]

        if dataset_name not in self.available_files:
            print(f"  ⚠️ Dataset '{dataset_name}' not found in available files")
            print(f"  Available: {list(self.available_files.keys())}")
            return pd.DataFrame()

        file_path = self.available_files[dataset_name]
        print(f"\nLoading {dataset_name} from {file_path.name}...")

        # Try to read CSV with different formats
        try:
            # Try standard format (frame_id, pedestrian_id, pos_x, pos_y)
            df = pd.read_csv(
                file_path, header=None,
                names=['frame_id', 'pedestrian_id', 'pos_x', 'pos_y']
            )
            df['frame_id'] = df['frame_id'].astype(int)
            df['pedestrian_id'] = df['pedestrian_id'].astype(int)
            df['pos_x'] = df['pos_x'].astype(float)
            df['pos_y'] = df['pos_y'].astype(float)
        except Exception as e:
            # Try different formats
            print(f"  Standard format failed: {e}")
            try:
                # Try with header
                df = pd.read_csv(file_path)
                # Check if columns match expected names
                if 'frame' in df.columns and 'id' in df.columns and 'x' in df.columns and 'y' in df.columns:
                    df = df.rename(columns={
                        'frame': 'frame_id',
                        'id': 'pedestrian_id',
                        'x': 'pos_x',
                        'y': 'pos_y'
                    })
                else:
                    raise ValueError("Unknown CSV format")
            except Exception as e2:
                print(f"  Alternative format also failed: {e2}")
                return pd.DataFrame()

        df['dataset'] = dataset_name
        self.dataset_cache[dataset_name] = df

        print(f"  Loaded {len(df):,} records | "
              f"{df['pedestrian_id'].nunique()} pedestrians | "
              f"frames {df['frame_id'].min()}–{df['frame_id'].max()}")
        return df

    # ------------------------------------------------------------------
    def compute_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocities from position data"""
        df = df.copy().sort_values(['pedestrian_id', 'frame_id'])
        fps = 2.5  # ETH-UCY frame rate

        df['vel_x'] = df.groupby('pedestrian_id')['pos_x'].diff() * fps
        df['vel_y'] = df.groupby('pedestrian_id')['pos_y'].diff() * fps
        df['speed'] = np.sqrt(df['vel_x'] ** 2 + df['vel_y'] ** 2)
        df['direction'] = np.arctan2(df['vel_y'], df['vel_x'])

        for col in ['vel_x', 'vel_y', 'speed', 'direction']:
            df[col] = df[col].fillna(0)

        return df

    # ------------------------------------------------------------------
    def aggregate_to_zones(self, df: pd.DataFrame,
                           zone_size: float = 5.0,
                           time_window: int = 5) -> pd.DataFrame:
        """Aggregate trajectory data to zone-based crowd signals"""
        df = df.copy()

        # Spatial zone encoding
        df['zone_x'] = (df['pos_x'] / zone_size).astype(int)
        df['zone_y'] = (df['pos_y'] / zone_size).astype(int)
        df['zone_id'] = df['zone_y'] * 100 + df['zone_x']
        df['time_window'] = df['frame_id'] // time_window

        if 'speed' not in df.columns:
            df = self.compute_velocities(df)

        agg = df.groupby(['time_window', 'zone_id', 'dataset']).agg(
            footfall_count=('pedestrian_id', 'nunique'),
            velocity_mean=('speed', 'mean'),
            velocity_std=('speed', 'std'),
            direction_variance=('direction', 'std'),
            zone_center_x=('pos_x', 'mean'),
            zone_center_y=('pos_y', 'mean'),
            frame_id=('frame_id', 'min'),
        ).reset_index()

        agg[['velocity_mean', 'velocity_std', 'direction_variance']] = \
            agg[['velocity_mean', 'velocity_std', 'direction_variance']].fillna(0)

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        fps = 2.5
        agg['timestamp'] = agg['frame_id'].apply(
            lambda x: base_time + timedelta(seconds=x / fps)
        )

        zone_area = zone_size * zone_size
        agg['density'] = agg['footfall_count'] / zone_area

        print(f"  Aggregated → {len(agg):,} zone records | "
              f"{agg['zone_id'].nunique()} unique zones")
        return agg

    # ------------------------------------------------------------------
    def load_trajectory_data(self, datasets: List[str] = ('eth', 'hotel'),
                             zone_size: float = 5.0,
                             time_window: int = 5) -> pd.DataFrame:
        """Load and aggregate trajectory data for multiple datasets"""
        all_data = []
        for name in datasets:
            try:
                print(f"\n{'='*60}\nProcessing {name}...\n{'='*60}")
                raw = self.load_dataset(name)
                if raw.empty:
                    print(f"  ⚠️ No data for {name}, skipping")
                    continue
                raw = self.compute_velocities(raw)
                agg = self.aggregate_to_zones(raw, zone_size=zone_size,
                                              time_window=time_window)
                all_data.append(agg)
            except Exception as e:
                print(f"  ✗ Error processing {name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_data:
            raise ValueError("No datasets loaded successfully")

        combined = pd.concat(all_data, ignore_index=True).sort_values(
            ['dataset', 'timestamp', 'zone_id']
        )
        print(f"\n{'='*60}")
        print(f"Combined: {len(combined):,} records | "
              f"datasets={combined['dataset'].unique().tolist()} | "
              f"zones={combined['zone_id'].nunique()}")
        print('=' * 60)
        return combined

    def load_eth_ucy(self, scenes: List[str], zone_size: float = 5.0,
                     time_window: int = 5) -> pd.DataFrame:
        """Load ETH-UCY dataset for specified scenes"""
        return self.load_trajectory_data(
            datasets=scenes, zone_size=zone_size, time_window=time_window
        )

    # ------------------------------------------------------------------
    def load_mall_dataset(self) -> pd.DataFrame:
        """Load Mall dataset using the updated loader"""
        from src.data.mall_loader import MallDatasetLoader

        print("\n" + "=" * 60)
        print("LOADING MALL DATASET")
        print("=" * 60)

        try:
            if not hasattr(self, 'mall_path') or not self.mall_path:
                self.mall_path = Path("data/raw/mall_dataset")

            print(f"Looking for Mall dataset at: {self.mall_path}")

            if not self.mall_path.exists():
                print(f"  ✗ Mall dataset directory not found: {self.mall_path}")
                return pd.DataFrame()

            loader = MallDatasetLoader(str(self.mall_path))
            df = loader.load_mall_dataset(zone_grid_size=(3, 3), time_window=5)

            if df.empty:
                print("  Mall dataset is empty")
                return pd.DataFrame()

            # Ensure numeric zone_id
            if df['zone_id'].dtype == 'object':
                mapping = {z: i for i, z in enumerate(df['zone_id'].unique())}
                df['zone_id'] = df['zone_id'].map(mapping)

            print(f"  ✅ Loaded {len(df):,} records | {df['zone_id'].nunique()} zones")
            return df

        except Exception as e:
            print(f"  ✗ Failed to load Mall dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def get_available_datasets(self) -> Dict[str, bool]:
        """Get dictionary of available datasets"""
        return {name: True for name in self.available_files}

    def save_processed(self, df: pd.DataFrame, filename: str):
        """Save processed data to file"""
        path = self.processed_path / filename
        df.to_csv(path, index=False)
        print(f"  Saved → {path} ({path.stat().st_size/(1024*1024):.2f} MB)")