"""
Data Analysis and Visualization Module for S.A.F.E
Generates comprehensive data distribution and analysis plots
FIXED VERSION - handles missing days/hours gracefully
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


class DataAnalyzer:
    """Generate data analysis plots for S.A.F.E"""

    def __init__(self, output_dir: str = "results/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
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

        # Remove ID columns
        exclude = ['zone_id', 'pedestrian_id', 'frame_id', 'time_window']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        # Limit number of features
        numeric_cols = numeric_cols[:max_features]

        if len(numeric_cols) == 0:
            print("   ⚠️ No numeric features to plot")
            return

        # Calculate grid size
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]

            # Plot histogram
            df[col].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col} Distribution', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')

            # Add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()

            stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}'
            ax.text(0.95, 0.95, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})

        ax.set_title(f'Feature Correlation Matrix - {dataset_name}',
                    fontsize=16, fontweight='bold', pad=20)

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

        # 1. Density by hour of day
        if 'density' in df.columns:
            hourly_density = df.groupby('hour')['density'].mean()
            axes[0, 0].plot(hourly_density.index, hourly_density.values,
                           marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_title('Average Density by Hour of Day', fontweight='bold')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Avg Density')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Density by day of week - FIXED
        if 'density' in df.columns:
            # Get actual days present in data
            daily_density = df.groupby('day_of_week')['density'].mean()

            # Create full week array with NaN for missing days
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            full_week_density = pd.Series(index=range(7), dtype=float)
            for day in daily_density.index:
                full_week_density[day] = daily_density[day]

            # Plot only days that exist
            x_positions = []
            y_values = []
            x_labels = []

            for day in range(7):
                if day in daily_density.index:
                    x_positions.append(day)
                    y_values.append(daily_density[day])
                    x_labels.append(day_names[day])

            if len(x_positions) > 0:
                axes[0, 1].bar(x_positions, y_values, color='skyblue', edgecolor='black')
                axes[0, 1].set_title('Average Density by Day of Week', fontweight='bold')
                axes[0, 1].set_xlabel('Day of Week')
                axes[0, 1].set_ylabel('Avg Density')
                axes[0, 1].set_xticks(x_positions)
                axes[0, 1].set_xticklabels(x_labels)
                axes[0, 1].grid(True, alpha=0.3, axis='y')
            else:
                axes[0, 1].text(0.5, 0.5, 'No day-of-week data available',
                              ha='center', va='center')

        # 3. Velocity by hour
        velocity_col = 'velocity_mean' if 'velocity_mean' in df.columns else 'speed_mean'
        if velocity_col in df.columns:
            hourly_velocity = df.groupby('hour')[velocity_col].mean()
            axes[1, 0].plot(hourly_velocity.index, hourly_velocity.values,
                           marker='s', linewidth=2, markersize=8, color='orange')
            axes[1, 0].set_title('Average Velocity by Hour of Day', fontweight='bold')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Avg Velocity')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Activity heatmap (hour vs day) - FIXED
        if 'density' in df.columns:
            # Check if we have enough data for heatmap
            pivot_data = df.groupby(['day_of_week', 'hour'])['density'].mean().unstack()

            if len(pivot_data) > 0:
                # Create y-axis labels for only the days present
                y_labels = [day_names[i] if i < len(day_names) else str(i)
                           for i in pivot_data.index]

                sns.heatmap(pivot_data, cmap='YlOrRd', ax=axes[1, 1],
                           cbar_kws={'label': 'Avg Density'})
                axes[1, 1].set_title('Density Heatmap (Day vs Hour)', fontweight='bold')
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Day of Week')
                axes[1, 1].set_yticklabels(y_labels, rotation=0)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data for heatmap',
                              ha='center', va='center')

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

        # 3. Density boxplot by zone
        if 'density' in df.columns and df['zone_id'].nunique() <= 20:
            df.boxplot(column='density', by='zone_id', ax=axes[1, 0])
            axes[1, 0].set_title('Density Distribution by Zone', fontweight='bold')
            axes[1, 0].set_xlabel('Zone ID')
            axes[1, 0].set_ylabel('Density')
            plt.sca(axes[1, 0])
            plt.xticks(rotation=45)

        # 4. Zone activity over time
        if 'timestamp' in df.columns and 'density' in df.columns:
            df_sorted = df.sort_values('timestamp')
            # Sample top 5 zones
            top_zones = df['zone_id'].value_counts().head(5).index

            for zone in top_zones:
                zone_data = df_sorted[df_sorted['zone_id'] == zone]
                if len(zone_data) > 0:
                    # Use rolling mean only if we have enough data
                    window = min(10, max(2, len(zone_data) // 10))
                    axes[1, 1].plot(zone_data['timestamp'],
                                   zone_data['density'].rolling(window, min_periods=1).mean(),
                                   label=f'Zone {zone}', alpha=0.7, linewidth=2)

            axes[1, 1].set_title('Density Over Time (Top 5 Zones)', fontweight='bold')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Density (Rolling Avg)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)

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
            axes[0, 0].set_yticklabels(missing.index)
            axes[0, 0].set_title('Missing Values by Feature', fontweight='bold')
            axes[0, 0].set_xlabel('Count')
            axes[0, 0].invert_yaxis()
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values ✓',
                          ha='center', va='center', fontsize=16)
            axes[0, 0].set_title('Missing Values', fontweight='bold')

        # 2. Data type distribution
        dtypes = df.dtypes.value_counts()
        axes[0, 1].pie(dtypes.values, labels=dtypes.index, autopct='%1.1f%%',
                      colors=sns.color_palette('pastel'))
        axes[0, 1].set_title('Data Type Distribution', fontweight='bold')

        # 3. Record count over time
        if 'timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
            df_temp['date'] = df_temp['timestamp'].dt.date

            daily_counts = df_temp.groupby('date').size()
            axes[1, 0].plot(daily_counts.index, daily_counts.values,
                           marker='o', linewidth=2, markersize=6)
            axes[1, 0].set_title('Record Count Over Time', fontweight='bold')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Summary statistics table
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]

            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')

            table = axes[1, 1].table(cellText=summary.round(2).values,
                                    rowLabels=summary.index,
                                    colLabels=summary.columns,
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 1].set_title('Summary Statistics (Top 6 Features)',
                                fontweight='bold', pad=20)
        else:
            axes[1, 1].text(0.5, 0.5, 'No numeric features found',
                          ha='center', va='center')

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
        colors = ['lightgreen', 'salmon']

        axes[0, 0].pie(anomaly_counts.values, labels=labels, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Anomaly Distribution', fontweight='bold')

        # 2. Anomalies by zone
        if 'zone_id' in df.columns:
            zone_anomalies = df.groupby('zone_id')['is_anomaly'].sum()
            axes[0, 1].bar(zone_anomalies.index, zone_anomalies.values,
                          color='red', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Anomaly Count by Zone', fontweight='bold')
            axes[0, 1].set_xlabel('Zone ID')
            axes[0, 1].set_ylabel('Anomaly Count')
            axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Anomalies over time
        if 'timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
            df_temp['hour'] = df_temp['timestamp'].dt.hour

            hourly_anomalies = df_temp.groupby('hour')['is_anomaly'].sum()
            axes[1, 0].plot(hourly_anomalies.index, hourly_anomalies.values,
                           marker='o', linewidth=2, markersize=8, color='red')
            axes[1, 0].set_title('Anomalies by Hour of Day', fontweight='bold')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Anomaly Count')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature comparison (normal vs anomaly)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['zone_id', 'is_anomaly', 'frame_id', 'time_window']
        numeric_cols = [c for c in numeric_cols if c not in exclude][:6]

        if len(numeric_cols) > 0:
            normal_data = df[df['is_anomaly'] == 0][numeric_cols]
            anomaly_data = df[df['is_anomaly'] == 1][numeric_cols]

            # Only calculate means if we have data
            if len(normal_data) > 0 and len(anomaly_data) > 0:
                normal_means = normal_data.mean()
                anomaly_means = anomaly_data.mean()

                x = np.arange(len(numeric_cols))
                width = 0.35

                axes[1, 1].bar(x - width/2, normal_means, width,
                              label='Normal', color='lightgreen', edgecolor='black')
                axes[1, 1].bar(x + width/2, anomaly_means, width,
                              label='Anomaly', color='salmon', edgecolor='black')

                axes[1, 1].set_title('Feature Means: Normal vs Anomaly', fontweight='bold')
                axes[1, 1].set_xlabel('Feature')
                axes[1, 1].set_ylabel('Mean Value')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(numeric_cols, rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data for comparison',
                              ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No numeric features found',
                          ha='center', va='center')

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

        # List of plot functions to try
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
                import traceback
                traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"✅ COMPLETED {success_count}/{len(plot_functions)} PLOTS FOR: {dataset_name}")
        print(f"   📁 Saved to: {self.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Test the analyzer
    print("Testing Data Analyzer...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'zone_id': np.random.randint(0, 10, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'density': np.random.exponential(20, n_samples),
        'velocity_mean': np.random.normal(1.0, 0.3, n_samples),
        'velocity_std': np.random.gamma(2, 0.1, n_samples),
        'direction_variance': np.random.gamma(3, 0.2, n_samples),
        'is_anomaly': np.random.binomial(1, 0.1, n_samples)
    })

    # Generate all plots
    analyzer = DataAnalyzer(output_dir="results/analysis")
    analyzer.generate_all_plots(df, dataset_name="test_dataset")

    print("\n✅ Test complete! Check results/analysis/ for plots")