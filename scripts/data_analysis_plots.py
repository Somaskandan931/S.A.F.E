# data_analysis_plots.py - FIXED VERSION
"""
Data Analysis and Visualization Module for S.A.F.E
Generates comprehensive data distribution and analysis plots
FIXED VERSION - Robust error handling for all plot types
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """Generate data analysis plots for S.A.F.E with robust error handling"""

    def __init__(self, output_dir: str = "results/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style with fallback
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        sns.set_palette("husl")

        print(f"📊 Data Analyzer initialized")
        print(f"   Output directory: {self.output_dir}")

    def plot_feature_distributions(self, df: pd.DataFrame,
                                   dataset_name: str = "dataset",
                                   max_features: int = 12) -> None:
        """Plot distribution of all numeric features"""
        print(f"\n📈 Plotting feature distributions for {dataset_name}...")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove ID columns and anomaly labels
        exclude = ['zone_id', 'pedestrian_id', 'frame_id', 'time_window', 'is_anomaly']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if len(numeric_cols) == 0:
            print("   ⚠️ No numeric features to plot")
            return

        # Limit number of features
        numeric_cols = numeric_cols[:max_features]

        # Calculate grid size
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]

            # Plot histogram with robust bin selection
            data = df[col].dropna()
            if len(data) > 0:
                bins = min(50, max(10, len(data) // 20))
                ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
                ax.set_title(f'{col}', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()

                stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}'
                ax.text(0.95, 0.95, stats_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        save_path = self.output_dir / f"{dataset_name}_feature_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✅ Saved: {save_path}")

    def plot_correlation_matrix(self, df: pd.DataFrame,
                               dataset_name: str = "dataset") -> None:
        """Plot correlation matrix heatmap"""
        print(f"\n🔥 Plotting correlation matrix for {dataset_name}...")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['zone_id', 'pedestrian_id', 'frame_id', 'time_window', 'is_anomaly']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if len(numeric_cols) < 2:
            print("   ⚠️ Not enough features for correlation matrix")
            return

        # Calculate correlation
        corr = df[numeric_cols].corr()

        # Create figure with adaptive size
        fig_size = max(10, len(numeric_cols) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

        # Plot heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   annot_kws={'size': 8})

        ax.set_title(f'Feature Correlation Matrix - {dataset_name}',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        save_path = self.output_dir / f"{dataset_name}_correlation_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✅ Saved: {save_path}")

    def plot_temporal_patterns(self, df: pd.DataFrame,
                              dataset_name: str = "dataset") -> None:
        """Plot temporal patterns in the data"""
        print(f"\n⏰ Plotting temporal patterns for {dataset_name}...")

        if 'timestamp' not in df.columns:
            print("   ⚠️ No timestamp column found")
            return

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # 1. Density by hour of day
        if 'density' in df.columns:
            hourly_density = df.groupby('hour')['density'].mean()
            axes[0, 0].plot(hourly_density.index, hourly_density.values,
                           marker='o', linewidth=2, markersize=6, color='steelblue')
            axes[0, 0].set_title('Average Density by Hour', fontweight='bold')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Avg Density')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(range(0, 24, 3))
        else:
            axes[0, 0].text(0.5, 0.5, 'No density data', ha='center', va='center')

        # 2. Density by day of week
        if 'density' in df.columns:
            daily_density = df.groupby('day_of_week')['density'].mean()
            x_positions = list(daily_density.index)
            y_values = list(daily_density.values)
            x_labels = [day_names[d] for d in x_positions]

            axes[0, 1].bar(x_positions, y_values, color='coral', edgecolor='black')
            axes[0, 1].set_title('Average Density by Day', fontweight='bold')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Avg Density')
            axes[0, 1].set_xticks(x_positions)
            axes[0, 1].set_xticklabels(x_labels)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[0, 1].text(0.5, 0.5, 'No density data', ha='center', va='center')

        # 3. Velocity by hour
        velocity_col = 'velocity_mean' if 'velocity_mean' in df.columns else 'speed_mean'
        if velocity_col in df.columns:
            hourly_velocity = df.groupby('hour')[velocity_col].mean()
            axes[1, 0].plot(hourly_velocity.index, hourly_velocity.values,
                           marker='s', linewidth=2, markersize=6, color='forestgreen')
            axes[1, 0].set_title('Average Velocity by Hour', fontweight='bold')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Avg Velocity')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xticks(range(0, 24, 3))
        else:
            axes[1, 0].text(0.5, 0.5, 'No velocity data', ha='center', va='center')

        # 4. Activity heatmap (hour vs day)
        if 'density' in df.columns:
            try:
                pivot_data = df.groupby(['day_of_week', 'hour'])['density'].mean().unstack()
                if not pivot_data.empty:
                    sns.heatmap(pivot_data, cmap='YlOrRd', ax=axes[1, 1],
                               cbar_kws={'label': 'Avg Density'})
                    axes[1, 1].set_title('Density Heatmap', fontweight='bold')
                    axes[1, 1].set_xlabel('Hour of Day')
                    axes[1, 1].set_ylabel('Day of Week')
                    y_labels = [day_names[i] for i in pivot_data.index if i < len(day_names)]
                    axes[1, 1].set_yticklabels(y_labels, rotation=0)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Heatmap error', ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No density data', ha='center', va='center')

        plt.tight_layout()

        save_path = self.output_dir / f"{dataset_name}_temporal_patterns.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✅ Saved: {save_path}")

    def plot_zone_analysis(self, df: pd.DataFrame,
                          dataset_name: str = "dataset") -> None:
        """Plot zone-level analysis"""
        print(f"\n🗺️ Plotting zone analysis for {dataset_name}...")

        if 'zone_id' not in df.columns:
            print("   ⚠️ No zone_id column found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Zone occupancy distribution
        zone_counts = df['zone_id'].value_counts().sort_index()
        axes[0, 0].bar(zone_counts.index, zone_counts.values,
                      color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Record Count by Zone', fontweight='bold')
        axes[0, 0].set_xlabel('Zone ID')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # 2. Average density by zone
        if 'density' in df.columns:
            zone_density = df.groupby('zone_id')['density'].mean().sort_index()
            axes[0, 1].bar(zone_density.index, zone_density.values,
                          color='coral', edgecolor='black')
            axes[0, 1].set_title('Average Density by Zone', fontweight='bold')
            axes[0, 1].set_xlabel('Zone ID')
            axes[0, 1].set_ylabel('Avg Density')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[0, 1].text(0.5, 0.5, 'No density data', ha='center', va='center')

        # 3. Density boxplot by zone
        if 'density' in df.columns and df['zone_id'].nunique() <= 20:
            try:
                df.boxplot(column='density', by='zone_id', ax=axes[1, 0])
                axes[1, 0].set_title('Density Distribution by Zone', fontweight='bold')
                axes[1, 0].set_xlabel('Zone ID')
                axes[1, 0].set_ylabel('Density')
                axes[1, 0].tick_params(axis='x', rotation=45)
            except:
                axes[1, 0].text(0.5, 0.5, 'Boxplot failed', ha='center', va='center')
        else:
            axes[1, 0].text(0.5, 0.5, 'No density data or too many zones', ha='center', va='center')

        # 4. Zone activity over time
        if 'timestamp' in df.columns and 'density' in df.columns:
            try:
                df_sorted = df.sort_values('timestamp')
                top_zones = df['zone_id'].value_counts().head(5).index

                for zone in top_zones:
                    zone_data = df_sorted[df_sorted['zone_id'] == zone]
                    if len(zone_data) > 1:
                        window = min(10, max(2, len(zone_data) // 10))
                        rolling_avg = zone_data['density'].rolling(window, min_periods=1).mean()
                        axes[1, 1].plot(zone_data['timestamp'], rolling_avg,
                                       label=f'Zone {zone}', alpha=0.7, linewidth=2)

                axes[1, 1].set_title('Density Over Time (Top 5 Zones)', fontweight='bold')
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Density (Rolling Avg)')
                axes[1, 1].legend(loc='best')
                axes[1, 1].grid(True, alpha=0.3)
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No temporal data', ha='center', va='center')

        plt.tight_layout()

        save_path = self.output_dir / f"{dataset_name}_zone_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✅ Saved: {save_path}")

    def plot_data_quality(self, df: pd.DataFrame,
                         dataset_name: str = "dataset") -> None:
        """Plot data quality metrics"""
        print(f"\n✅ Plotting data quality metrics for {dataset_name}...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            axes[0, 0].barh(range(len(missing)), missing.values, color='salmon')
            axes[0, 0].set_yticks(range(len(missing)))
            axes[0, 0].set_yticklabels(missing.index, fontsize=10)
            axes[0, 0].set_title('Missing Values by Feature', fontweight='bold')
            axes[0, 0].set_xlabel('Count')
            axes[0, 0].invert_yaxis()
        else:
            axes[0, 0].text(0.5, 0.5, '✓ No Missing Values', ha='center', va='center', fontsize=16)
            axes[0, 0].set_title('Missing Values', fontweight='bold')
            axes[0, 0].set_xlim(0, 1)
            axes[0, 0].set_ylim(0, 1)

        # 2. Data type distribution
        dtypes = df.dtypes.value_counts()
        axes[0, 1].pie(dtypes.values, labels=[str(t) for t in dtypes.index], autopct='%1.1f%%')
        axes[0, 1].set_title('Data Type Distribution', fontweight='bold')

        # 3. Record count over time
        if 'timestamp' in df.columns:
            try:
                df_temp = df.copy()
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
                df_temp['date'] = df_temp['timestamp'].dt.date

                daily_counts = df_temp.groupby('date').size()
                axes[1, 0].plot(daily_counts.index, daily_counts.values,
                               marker='o', linewidth=2, markersize=4, color='steelblue')
                axes[1, 0].set_title('Record Count Over Time', fontweight='bold')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].grid(True, alpha=0.3)
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
        else:
            axes[1, 0].text(0.5, 0.5, 'No timestamp data', ha='center', va='center')

        # 4. Summary statistics table
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]

            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')

            table = axes[1, 1].table(cellText=summary.round(3).values,
                                    rowLabels=summary.index,
                                    colLabels=summary.columns,
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            axes[1, 1].set_title('Summary Statistics', fontweight='bold', pad=20)
        else:
            axes[1, 1].text(0.5, 0.5, 'No numeric features', ha='center', va='center')

        plt.tight_layout()

        save_path = self.output_dir / f"{dataset_name}_data_quality.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✅ Saved: {save_path}")

    def plot_anomaly_analysis(self, df: pd.DataFrame,
                             dataset_name: str = "dataset") -> None:
        """Plot anomaly distribution if labels exist"""
        print(f"\n🚨 Plotting anomaly analysis for {dataset_name}...")

        if 'is_anomaly' not in df.columns:
            print("   ⚠️ No anomaly labels found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Anomaly distribution
        anomaly_counts = df['is_anomaly'].value_counts()
        labels = ['Normal', 'Anomaly']
        colors = ['#4caf50', '#f44336']

        axes[0, 0].pie(anomaly_counts.values, labels=labels, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Anomaly Distribution', fontweight='bold')

        # 2. Anomalies by zone
        if 'zone_id' in df.columns:
            zone_anomalies = df.groupby('zone_id')['is_anomaly'].sum()
            axes[0, 1].bar(zone_anomalies.index, zone_anomalies.values,
                          color='#f44336', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Anomaly Count by Zone', fontweight='bold')
            axes[0, 1].set_xlabel('Zone ID')
            axes[0, 1].set_ylabel('Anomaly Count')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[0, 1].text(0.5, 0.5, 'No zone data', ha='center', va='center')

        # 3. Anomalies over time
        if 'timestamp' in df.columns:
            try:
                df_temp = df.copy()
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
                df_temp['hour'] = df_temp['timestamp'].dt.hour

                hourly_anomalies = df_temp.groupby('hour')['is_anomaly'].sum()
                axes[1, 0].bar(hourly_anomalies.index, hourly_anomalies.values,
                              color='#f44336', alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Anomalies by Hour', fontweight='bold')
                axes[1, 0].set_xlabel('Hour of Day')
                axes[1, 0].set_ylabel('Anomaly Count')
                axes[1, 0].set_xticks(range(0, 24, 3))
                axes[1, 0].grid(True, alpha=0.3, axis='y')
            except:
                axes[1, 0].text(0.5, 0.5, 'Error plotting temporal anomalies', ha='center', va='center')
        else:
            axes[1, 0].text(0.5, 0.5, 'No timestamp data', ha='center', va='center')

        # 4. Feature comparison (normal vs anomaly)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['zone_id', 'is_anomaly', 'frame_id', 'time_window', 'pedestrian_id']
        numeric_cols = [c for c in numeric_cols if c not in exclude][:6]

        if len(numeric_cols) > 0:
            normal_data = df[df['is_anomaly'] == 0][numeric_cols]
            anomaly_data = df[df['is_anomaly'] == 1][numeric_cols]

            if len(normal_data) > 0 and len(anomaly_data) > 0:
                normal_means = normal_data.mean()
                anomaly_means = anomaly_data.mean()

                x = np.arange(len(numeric_cols))
                width = 0.35

                axes[1, 1].bar(x - width/2, normal_means, width,
                              label='Normal', color='#4caf50', edgecolor='black')
                axes[1, 1].bar(x + width/2, anomaly_means, width,
                              label='Anomaly', color='#f44336', edgecolor='black')

                axes[1, 1].set_title('Feature Means: Normal vs Anomaly', fontweight='bold')
                axes[1, 1].set_xlabel('Feature')
                axes[1, 1].set_ylabel('Mean Value')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(numeric_cols, rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No numeric features', ha='center', va='center')

        plt.tight_layout()

        save_path = self.output_dir / f"{dataset_name}_anomaly_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✅ Saved: {save_path}")

    def generate_all_plots(self, df: pd.DataFrame,
                          dataset_name: str = "dataset") -> None:
        """Generate all analysis plots with error handling"""
        print(f"\n{'='*60}")
        print(f"GENERATING ALL PLOTS FOR: {dataset_name}")
        print(f"{'='*60}")

        # List of plot functions
        plot_functions = [
            ('feature_distributions', self.plot_feature_distributions),
            ('correlation_matrix', self.plot_correlation_matrix),
            ('temporal_patterns', self.plot_temporal_patterns),
            ('zone_analysis', self.plot_zone_analysis),
            ('data_quality', self.plot_data_quality),
            ('anomaly_analysis', self.plot_anomaly_analysis)
        ]

        success_count = 0
        for name, plot_func in plot_functions:
            try:
                plot_func(df, dataset_name)
                success_count += 1
            except Exception as e:
                print(f"   ⚠️ Failed to generate {name}: {e}")

        print(f"\n{'='*60}")
        print(f"✅ COMPLETED {success_count}/{len(plot_functions)} PLOTS FOR: {dataset_name}")
        print(f"   📁 Saved to: {self.output_dir}")
        print(f"{'='*60}")