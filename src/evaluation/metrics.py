"""
Evaluation Metrics Module for S.A.F.E - FIXED PLOTTING
Comprehensive metrics with working visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """Calculate and visualize performance metrics with fixed plotting"""

    def __init__(self):
        self.metrics_history = []
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def calculate_basic_metrics(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)

            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

        return metrics

    def calculate_threshold_metrics(self, y_true: np.ndarray,
                                      y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate metrics that require scores"""
        metrics = {}

        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['roc_auc'] = 0.0

        try:
            metrics['average_precision'] = average_precision_score(y_true, y_scores)
        except:
            metrics['average_precision'] = 0.0

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                save_path: str = None) -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")

        plt.close(fig)
        return fig

    def plot_roc_curve(self, y_true: np.ndarray,
                         y_scores: np.ndarray,
                         save_path: str = None) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curve saved to {save_path}")

        plt.close(fig)
        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray,
                                      y_scores: np.ndarray,
                                      save_path: str = None) -> plt.Figure:
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(recall, precision, 'b-', linewidth=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        ax.axhline(y=y_true.mean(), color='r', linestyle='--', linewidth=2,
                   label=f'Baseline (AP = {y_true.mean():.3f})')

        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Precision-Recall curve saved to {save_path}")

        plt.close(fig)
        return fig

    def plot_score_distribution(self, y_true: np.ndarray,
                                  y_scores: np.ndarray,
                                  save_path: str = None) -> plt.Figure:
        """Plot anomaly score distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Separate scores by class
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]

        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')

        ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Score distribution saved to {save_path}")

        plt.close(fig)
        return fig

    def create_evaluation_report(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_scores: np.ndarray,
                                   model_name: str,
                                   save_dir: str = None) -> Dict:
        """Create complete evaluation with all plots"""
        print(f"\n{'='*60}")
        print(f"Creating Evaluation Report: {model_name}")
        print(f"{'='*60}")

        # Calculate metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred)

        if y_scores is not None:
            threshold_metrics = self.calculate_threshold_metrics(y_true, y_scores)
            metrics.update(threshold_metrics)

        # Create plots if save directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📊 Generating visualizations...")

            try:
                # Confusion Matrix
                self.plot_confusion_matrix(
                    y_true, y_pred,
                    save_path=str(save_dir / f"{model_name}_confusion_matrix.png")
                )
            except Exception as e:
                print(f"⚠️ Could not create confusion matrix: {e}")

            if y_scores is not None:
                try:
                    # ROC Curve
                    self.plot_roc_curve(
                        y_true, y_scores,
                        save_path=str(save_dir / f"{model_name}_roc_curve.png")
                    )
                except Exception as e:
                    print(f"⚠️ Could not create ROC curve: {e}")

                try:
                    # Precision-Recall Curve
                    self.plot_precision_recall_curve(
                        y_true, y_scores,
                        save_path=str(save_dir / f"{model_name}_pr_curve.png")
                    )
                except Exception as e:
                    print(f"⚠️ Could not create PR curve: {e}")

                try:
                    # Score Distribution
                    self.plot_score_distribution(
                        y_true, y_scores,
                        save_path=str(save_dir / f"{model_name}_score_distribution.png")
                    )
                except Exception as e:
                    print(f"⚠️ Could not create score distribution: {e}")

        # Create text report
        report = self.create_metrics_report(y_true, y_pred, y_scores, model_name)

        if save_dir:
            report_path = save_dir / f"{model_name}_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"✅ Text report saved to {report_path}")

        print(report)

        return metrics

    def create_metrics_report(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_scores: np.ndarray = None,
                                model_name: str = "Model") -> str:
        """Create a text report of metrics"""
        metrics = self.calculate_basic_metrics(y_true, y_pred)

        if y_scores is not None:
            threshold_metrics = self.calculate_threshold_metrics(y_true, y_scores)
            metrics.update(threshold_metrics)

        report = f"\n{'='*60}\n"
        report += f"Performance Report: {model_name}\n"
        report += f"{'='*60}\n\n"

        report += f"Classification Metrics:\n"
        report += f"  Precision:        {metrics['precision']:.4f}\n"
        report += f"  Recall:           {metrics['recall']:.4f}\n"
        report += f"  F1 Score:         {metrics['f1_score']:.4f}\n"
        report += f"  Accuracy:         {metrics.get('accuracy', 0):.4f}\n\n"

        if 'roc_auc' in metrics:
            report += f"Threshold Metrics:\n"
            report += f"  ROC AUC:          {metrics['roc_auc']:.4f}\n"
            report += f"  Avg Precision:    {metrics['average_precision']:.4f}\n\n"

        report += f"Confusion Matrix:\n"
        report += f"  True Positives:   {metrics.get('true_positives', 0)}\n"
        report += f"  False Positives:  {metrics.get('false_positives', 0)}\n"
        report += f"  True Negatives:   {metrics.get('true_negatives', 0)}\n"
        report += f"  False Negatives:  {metrics.get('false_negatives', 0)}\n"

        report += f"\n{'='*60}\n"

        return report


if __name__ == "__main__":
    # Test the fixed metrics
    print("Testing Fixed Metrics Calculator with Plotting...")

    # Generate sample data
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, 1000)
    y_scores = np.random.random(1000)
    y_pred = (y_scores > 0.5).astype(int)

    # Create calculator
    calculator = MetricsCalculator()

    # Create complete evaluation report with plots
    metrics = calculator.create_evaluation_report(
        y_true, y_pred, y_scores,
        model_name="Test_Model",
        save_dir="results/plots"
    )

    print("\n✅ Test complete! Check results/plots/ directory for visualizations")