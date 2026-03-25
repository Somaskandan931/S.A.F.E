# metrics.py - FIXED VERSION with proper plots
"""
Evaluation Metrics Module for S.A.F.E
Comprehensive metrics with working visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculate and visualize performance metrics"""

    def __init__(self):
        self.metrics_history = []
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        sns.set_palette("husl")

    def calculate_all_metrics(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all classification metrics"""
        metrics = {}

        # Basic metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Score-based metrics
        if y_scores is not None:
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
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 14})

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def plot_roc_curve(self, y_true: np.ndarray,
                         y_scores: np.ndarray,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray,
                                      y_scores: np.ndarray,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, 'b-', linewidth=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        ax.axhline(y=y_true.mean(), color='r', linestyle='--', linewidth=2,
                   label=f'Baseline (AP = {y_true.mean():.3f})')

        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def plot_score_distribution(self, y_true: np.ndarray,
                                  y_scores: np.ndarray,
                                  threshold: Optional[float] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot anomaly score distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separate scores by class
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]

        bins = 50
        ax.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='#4caf50', density=True)
        ax.hist(anomaly_scores, bins=bins, alpha=0.6, label='Anomaly', color='#f44336', density=True)

        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold:.3f}')

        ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def plot_model_comparison(self, results: Dict[str, Dict],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of multiple models"""
        if not results:
            return None

        metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(results.keys())

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [results[name].get(metric, 0) for name in model_names]

            bars = ax.bar(model_names, values, color='steelblue', edgecolor='black')
            ax.set_title(metric.upper(), fontweight='bold', fontsize=12)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def create_evaluation_report(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_scores: np.ndarray,
                                   model_name: str,
                                   save_dir: Optional[str] = None,
                                   threshold: Optional[float] = None) -> Dict:
        """Create complete evaluation with all plots"""
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {model_name}")
        print(f"{'='*60}")

        # Calculate metrics
        metrics = self.calculate_all_metrics(y_true, y_pred, y_scores)

        # Print metrics
        print(f"\nClassification Metrics:")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1 Score:   {metrics['f1_score']:.4f}")
        print(f"  Accuracy:   {metrics.get('accuracy', 0):.4f}")

        if 'roc_auc' in metrics:
            print(f"\nScore-based Metrics:")
            print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
            print(f"  Avg Precision: {metrics['average_precision']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics.get('true_positives', 0)}")
        print(f"  FP: {metrics.get('false_positives', 0)}")
        print(f"  TN: {metrics.get('true_negatives', 0)}")
        print(f"  FN: {metrics.get('false_negatives', 0)}")

        # Create plots if save directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📊 Generating visualizations...")

            try:
                self.plot_confusion_matrix(
                    y_true, y_pred,
                    save_path=str(save_dir / f"{model_name}_confusion_matrix.png")
                )
                print(f"  ✅ Confusion matrix saved")
            except Exception as e:
                print(f"  ⚠️ Confusion matrix failed: {e}")

            try:
                self.plot_roc_curve(
                    y_true, y_scores,
                    save_path=str(save_dir / f"{model_name}_roc_curve.png")
                )
                print(f"  ✅ ROC curve saved")
            except Exception as e:
                print(f"  ⚠️ ROC curve failed: {e}")

            try:
                self.plot_precision_recall_curve(
                    y_true, y_scores,
                    save_path=str(save_dir / f"{model_name}_pr_curve.png")
                )
                print(f"  ✅ PR curve saved")
            except Exception as e:
                print(f"  ⚠️ PR curve failed: {e}")

            try:
                self.plot_score_distribution(
                    y_true, y_scores, threshold,
                    save_path=str(save_dir / f"{model_name}_score_dist.png")
                )
                print(f"  ✅ Score distribution saved")
            except Exception as e:
                print(f"  ⚠️ Score distribution failed: {e}")

        return metrics