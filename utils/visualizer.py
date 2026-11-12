"""
Visualization utilities for patient-level results
===============================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from pathlib import Path
import logging
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization utilities for experiment results"""

    @staticmethod
    def plot_comprehensive_results(fold_results: List[Dict[str, Any]], config, save_path: str):
        """Generate comprehensive visualization plots"""

        # Create plots directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Plot performance metrics
        Visualizer._plot_performance_metrics(fold_results, save_path)

        # Plot FP-constrained results
        Visualizer._plot_fp_constrained_results(fold_results, save_path)

        # Plot confusion matrices only if volume-level data exists
        if 'true_positives' in fold_results[0]:
            Visualizer._plot_confusion_matrices(fold_results, save_path)

        # Plot AUROC curves for each fold
        Visualizer._plot_roc_curves(fold_results, save_path)

        logger.info("Visualization plots saved")

    @staticmethod
    def _plot_performance_metrics(fold_results: List[Dict[str, Any]], save_path: str):
        """Plot performance metrics across folds"""
        
        # Detect available metrics
        sample_result = fold_results[0]
        
        # Patient-level metrics
        patient_metrics = {}
        if 'patient_auc' in sample_result:
            patient_metrics['Patient AUC'] = [r['patient_auc'] for r in fold_results]
        
        # Volume-level metrics (if available)
        volume_metric_keys = ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        volume_metrics = {}
        
        for key in volume_metric_keys:
            if key in sample_result:
                display_name = key.replace('_', ' ').title()
                if key == 'auc':
                    display_name = 'Volume AUC'
                volume_metrics[display_name] = [r[key] for r in fold_results]
        
        # Combine metrics
        all_metrics = {**patient_metrics, **volume_metrics}
        
        if not all_metrics:
            logger.warning("No metrics found for visualization")
            return
        
        # Create figure
        n_metrics = len(all_metrics)
        
        # Calculate rows and columns
        if n_metrics == 1:
            n_rows, n_cols = 1, 1
        elif n_metrics == 2:
            n_rows, n_cols = 1, 2
        else:
            n_rows = 2
            n_cols = (n_metrics + 1) // 2
        
        fig, axes_raw = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        
        # Convert to list properly
        if n_metrics == 1:
            axes = [axes_raw]
        else:
            axes = axes_raw.flatten().tolist()
        
        colors = sns.color_palette("husl", len(fold_results))
        
        for idx, (metric_name, values) in enumerate(all_metrics.items()):
            ax = axes[idx]
            
            # Bar plot with individual fold results
            x_pos = np.arange(len(values))
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            
            # Formatting
            ax.set_xlabel('Fold', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(f'{metric_name} Across Folds', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{i+1}' for i in range(len(values))])
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide extra subplots
        for idx in range(len(all_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/comprehensive_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_fp_constrained_results(fold_results: List[Dict[str, Any]], save_path: str):
        """Plot FP-constrained results"""
        
        if 'fp_constrained_results' not in fold_results[0]:
            logger.info("No FP-constrained results to plot")
            return
        
        # Get FP constraint keys
        fp_keys = list(fold_results[0]['fp_constrained_results'].keys())
        metrics_to_plot = ['accuracy', 'recall', 'specificity', 'f1_score']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Prepare data for each FP constraint
            fp_labels = []
            fold_data = {f'Fold {i+1}': [] for i in range(len(fold_results))}
            
            for fp_key in sorted(fp_keys):
                fp_val = fp_key.split('_')[1]
                fp_labels.append(f'FP={fp_val}')
                
                for fold_idx, result in enumerate(fold_results):
                    value = result['fp_constrained_results'][fp_key].get(metric, 0)
                    fold_data[f'Fold {fold_idx+1}'].append(value)
            
            # Plot grouped bar chart
            x = np.arange(len(fp_labels))
            width = 0.8 / len(fold_results)
            
            for fold_idx, (fold_name, values) in enumerate(fold_data.items()):
                offset = width * fold_idx - (width * len(fold_results) / 2) + width / 2
                ax.bar(x + offset, values, width, label=fold_name, alpha=0.8)
            
            ax.set_xlabel('FP Constraint', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} by FP Constraint', 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(fp_labels)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/fp_constrained_results.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_confusion_matrices(fold_results: List[Dict[str, Any]], save_path: str):
        """Plot confusion matrices for each fold (volume-level only)"""
        
        n_folds = len(fold_results)
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
        
        if n_folds == 1:
            axes = [axes]
        
        for idx, result in enumerate(fold_results):
            # Check if confusion matrix data exists
            if not all(key in result for key in ['true_positives', 'false_positives', 
                                                  'true_negatives', 'false_negatives']):
                continue
                
            cm = np.array([
                [result['true_negatives'], result['false_positives']],
                [result['false_negatives'], result['true_positives']]
            ])
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Diseased'],
                       yticklabels=['Normal', 'Diseased'],
                       cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
            ax.set_title(f'Fold {idx+1} Confusion Matrix\n'
                        f'Acc: {result.get("accuracy", 0):.3f}',
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_detailed_analysis(fold_results: List[Dict[str, Any]], save_path: str):
        """Plot detailed analysis including error distributions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Threshold distribution
        ax = axes[0, 0]
        thresholds = [r.get('threshold', 0) for r in fold_results]
        if any(thresholds):
            ax.bar(range(len(thresholds)), thresholds, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
            ax.set_ylabel('Optimal Threshold', fontsize=11, fontweight='bold')
            ax.set_title('Optimal Threshold per Fold', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels([f'{i+1}' for i in range(len(thresholds))])
            ax.grid(axis='y', alpha=0.3)
        
        # 2. Patient count distribution
        ax = axes[0, 1]
        normal_counts = [r.get('normal_patient_count', 0) for r in fold_results]
        diseased_counts = [r.get('diseased_patient_count', 0) for r in fold_results]
        
        x = np.arange(len(fold_results))
        width = 0.35
        ax.bar(x - width/2, normal_counts, width, label='Normal', alpha=0.8)
        ax.bar(x + width/2, diseased_counts, width, label='Diseased', alpha=0.8)
        ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
        ax.set_ylabel('Patient Count', fontsize=11, fontweight='bold')
        ax.set_title('Patient Distribution per Fold', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in range(len(fold_results))])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Training time
        ax = axes[1, 0]
        training_times = [r.get('fold_training_time', 0) for r in fold_results]
        if any(training_times):
            ax.bar(range(len(training_times)), training_times, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
            ax.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
            ax.set_title('Training Time per Fold', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(training_times)))
            ax.set_xticklabels([f'{i+1}' for i in range(len(training_times))])
            ax.grid(axis='y', alpha=0.3)
        
        # 4. Best epoch
        ax = axes[1, 1]
        best_epochs = [r.get('best_epoch', 0) for r in fold_results]
        if any(best_epochs):
            ax.bar(range(len(best_epochs)), best_epochs, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
            ax.set_ylabel('Best Epoch', fontsize=11, fontweight='bold')
            ax.set_title('Best Epoch per Fold', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(best_epochs)))
            ax.set_xticklabels([f'{i+1}' for i in range(len(best_epochs))])
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _plot_roc_curves(fold_results: List[Dict[str, Any]], save_path: str):
        """Plot ROC curves for each fold and aggregate"""

        # Check if we have the necessary data
        if 'all_labels' not in fold_results[0] or 'all_scores' not in fold_results[0]:
            logger.info("No score data available for ROC curves")
            return

        n_folds = len(fold_results)
        colors = sns.color_palette("husl", n_folds)

        # Create figure with two subplots: individual folds and mean ROC
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Store TPR values for averaging
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        # Plot 1: Individual fold ROC curves
        for idx, result in enumerate(fold_results):
            y_true = result['all_labels']
            y_scores = result['all_scores']

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            # Interpolate for averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            # Plot individual fold
            ax1.plot(fpr, tpr, color=colors[idx], lw=2, alpha=0.7,
                    label=f'Fold {idx+1} (AUC = {roc_auc:.3f})')

        # Plot diagonal line
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('ROC Curves - Individual Folds', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])

        # Plot 2: Mean ROC curve with confidence interval
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax2.plot(mean_fpr, mean_tpr, color='b', lw=3,
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

        # Calculate standard deviation of TPR
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                        label=r'$\pm$ 1 std. dev.')

        # Plot individual fold curves with less emphasis
        for idx, interp_tpr in enumerate(tprs):
            ax2.plot(mean_fpr, interp_tpr, color=colors[idx], lw=1, alpha=0.3)

        # Plot diagonal line
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Mean ROC Curve with Variability', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"ROC curves saved: Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")

        # Also create individual ROC curve plots for each fold
        Visualizer._plot_individual_roc_curves(fold_results, save_path)

    @staticmethod
    def _plot_individual_roc_curves(fold_results: List[Dict[str, Any]], save_path: str):
        """Plot individual ROC curves for each fold in separate files"""

        # Create subdirectory for individual ROC curves
        roc_dir = Path(save_path) / "individual_roc_curves"
        roc_dir.mkdir(parents=True, exist_ok=True)

        for idx, result in enumerate(fold_results):
            if 'all_labels' not in result or 'all_scores' not in result:
                continue

            y_true = result['all_labels']
            y_scores = result['all_scores']

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 7))

            # Plot ROC curve
            ax.plot(fpr, tpr, color='darkorange', lw=3,
                   label=f'ROC curve (AUC = {roc_auc:.4f})')

            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

            # Find and plot optimal threshold point (if available)
            if 'threshold' in result:
                threshold = result['threshold']
                # Find the point on ROC curve closest to this threshold
                threshold_idx = np.argmin(np.abs(thresholds - threshold))
                ax.plot(fpr[threshold_idx], tpr[threshold_idx], 'ro', markersize=10,
                       label=f'Optimal Threshold = {threshold:.6f}')

            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'Fold {idx+1} ROC Curve\nAUC = {roc_auc:.4f}',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])

            plt.tight_layout()
            plt.savefig(f'{roc_dir}/fold_{idx+1}_roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Individual ROC curves saved in {roc_dir}")