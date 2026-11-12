"""
Professional report generation utilities
=======================================
"""

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Professional report generation utilities"""

    @staticmethod
    def generate_detailed_report(fold_results: List[Dict], statistical_summary: Dict,
                                 config: DictConfig, save_path: str) -> str:
        """Generate comprehensive text and markdown reports"""

        logger.info("Generating detailed reports...")

        # Generate text report
        text_report = ReportGenerator._generate_text_report(
            fold_results, statistical_summary, config
        )

        # Generate markdown report
        markdown_report = ReportGenerator._generate_markdown_report(
            fold_results, statistical_summary, config
        )

        # Save reports
        with open(os.path.join(save_path, 'detailed_report.txt'), 'w', encoding='utf-8') as f:
            f.write(text_report)

        with open(os.path.join(save_path, 'detailed_report.md'), 'w', encoding='utf-8') as f:
            f.write(markdown_report)

        # Generate executive summary
        executive_summary = ReportGenerator._generate_executive_summary(
            fold_results, statistical_summary, config
        )

        with open(os.path.join(save_path, 'executive_summary.md'), 'w', encoding='utf-8') as f:
            f.write(executive_summary)

        logger.info("All reports generated successfully")

        return text_report

    @staticmethod
    def _generate_text_report(fold_results: List[Dict], statistical_summary: Dict,
                              config: DictConfig) -> str:
        """Generate comprehensive text report"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("K-FOLD CROSS VALIDATION DETAILED REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Run Name: {config.run_name}")
        report_lines.append(f"Random Seed: {config.random_seed}")
        report_lines.append("")

        # Experiment configuration
        report_lines.append("EXPERIMENT CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"K-Folds: {config.k_folds}")
        report_lines.append(f"Epochs: {config.epochs}")
        report_lines.append(f"Batch Size: {config.batch_size}")
        report_lines.append(f"Learning Rate: {config.lr}")
        report_lines.append(f"Model Dimensions: {config.dims}")
        report_lines.append(f"Memory Dimension: {config.mem_dim}")
        report_lines.append(f"Confidence Level: {config.confidence_level}")
        report_lines.append("")

        # Statistical summary
        report_lines.append("STATISTICAL SUMMARY")
        report_lines.append("-" * 40)

        # Patient-level metrics
        if 'patient_auc' in statistical_summary:
            stats = statistical_summary['patient_auc']
            report_lines.append("Patient-Level AUC:")
            report_lines.append(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            report_lines.append(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            report_lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report_lines.append(f"  Median (IQR): {stats['median']:.4f} ({stats['q25']:.4f}, {stats['q75']:.4f})")
            report_lines.append("")

        # FP-constrained results
        if 'fp_constrained_summary' in statistical_summary:
            report_lines.append("FP-Constrained Results:")
            for fp_key, fp_stats in statistical_summary['fp_constrained_summary'].items():
                fp_val = fp_key.split('_')[1]
                report_lines.append(f"  FP={fp_val}:")
                for metric_name, metric_stats in fp_stats.items():
                    report_lines.append(f"    {metric_name.title()}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")
            report_lines.append("")

        # Volume-level metrics (if available)
        volume_metrics = {
            'auc': 'Volume-Level AUC',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall (Sensitivity)',
            'f1_score': 'F1-Score',
            'specificity': 'Specificity'
        }

        volume_metrics_found = False
        for metric_key, metric_name in volume_metrics.items():
            if metric_key in statistical_summary:
                if not volume_metrics_found:
                    report_lines.append("Volume-Level Metrics:")
                    volume_metrics_found = True
                stats = statistical_summary[metric_key]
                report_lines.append(f"{metric_name}:")
                report_lines.append(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                report_lines.append(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
                report_lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                report_lines.append(f"  Median (IQR): {stats['median']:.4f} ({stats['q25']:.4f}, {stats['q75']:.4f})")
                report_lines.append("")

        # Individual fold results
        report_lines.append("INDIVIDUAL FOLD RESULTS")
        report_lines.append("-" * 40)

        for i, result in enumerate(fold_results):
            report_lines.append(f"Fold {i + 1}:")
            
            # Patient-level metrics
            if 'patient_auc' in result:
                report_lines.append(f"  Patient AUC: {result['patient_auc']:.4f}")
                report_lines.append(f"  Normal Patients: {result.get('normal_patient_count', 'N/A')}")
                report_lines.append(f"  Diseased Patients: {result.get('diseased_patient_count', 'N/A')}")
            
            # Volume-level metrics (if available)
            if 'auc' in result:
                report_lines.append(f"  Volume AUC: {result['auc']:.4f}")
                report_lines.append(f"  Accuracy: {result['accuracy']:.4f}")
                report_lines.append(f"  Precision: {result['precision']:.4f}")
                report_lines.append(f"  Recall: {result['recall']:.4f}")
                report_lines.append(f"  F1-Score: {result['f1_score']:.4f}")
                report_lines.append(f"  Specificity: {result['specificity']:.4f}")
                report_lines.append(f"  Threshold: {result['threshold']:.6f}")
            
            report_lines.append(f"  Training Time: {result.get('fold_training_time', 0):.2f}s")
            report_lines.append(f"  Best Epoch: {result.get('best_epoch', 'N/A')}/{result.get('total_epochs', 'N/A')}")

            # Confusion Matrix (if available)
            if 'true_negatives' in result:
                report_lines.append(f"  Confusion Matrix:")
                report_lines.append(f"    TN: {result['true_negatives']}, FP: {result['false_positives']}")
                report_lines.append(f"    FN: {result['false_negatives']}, TP: {result['true_positives']}")

            # Error Statistics (if available)
            if 'normal_mean_error' in result:
                report_lines.append(f"  Normal Error: {result['normal_mean_error']:.6f} ± {result['normal_std_error']:.6f}")
                report_lines.append(f"  Diseased Error: {result['diseased_mean_error']:.6f} ± {result['diseased_std_error']:.6f}")
            
            report_lines.append("")

        # Threshold analysis (if available)
        if 'threshold_stats' in statistical_summary:
            report_lines.append("THRESHOLD ANALYSIS")
            report_lines.append("-" * 40)
            thresh_stats = statistical_summary['threshold_stats']
            report_lines.append(f"Mean Threshold: {thresh_stats['mean']:.6f} ± {thresh_stats['std']:.6f}")
            report_lines.append(f"Threshold Range: {np.min(thresh_stats['values']):.6f} - {np.max(thresh_stats['values']):.6f}")
            report_lines.append("")

        # Performance analysis
        report_lines.extend(ReportGenerator._generate_performance_analysis(fold_results, statistical_summary))

        # Conclusions
        report_lines.extend(ReportGenerator._generate_conclusions(fold_results, statistical_summary))

        report_lines.append("")
        report_lines.append("=" * 80)

        return '\n'.join(report_lines)

    @staticmethod
    def _generate_markdown_report(fold_results: List[Dict], statistical_summary: Dict,
                                  config: DictConfig) -> str:
        """Generate markdown report for better readability"""

        md_lines = []
        md_lines.append("# K-Fold Cross Validation Report")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**Run Name:** `{config.run_name}`")
        md_lines.append(f"**Random Seed:** `{config.random_seed}`")
        md_lines.append("")

        # Configuration table
        md_lines.append("## Experiment Configuration")
        md_lines.append("")
        md_lines.append("| Parameter | Value |")
        md_lines.append("|-----------|-------|")
        md_lines.append(f"| K-Folds | {config.k_folds} |")
        md_lines.append(f"| Epochs | {config.epochs} |")
        md_lines.append(f"| Batch Size | {config.batch_size} |")
        md_lines.append(f"| Learning Rate | {config.lr} |")
        md_lines.append(f"| Model Dimensions | {config.dims} |")
        md_lines.append(f"| Memory Dimension | {config.mem_dim} |")
        md_lines.append(f"| Confidence Level | {config.confidence_level} |")
        md_lines.append("")

        # Statistical summary - Patient-level
        if 'patient_auc' in statistical_summary:
            md_lines.append("## Patient-Level Results")
            md_lines.append("")
            stats = statistical_summary['patient_auc']
            md_lines.append(f"**Patient AUC:** {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
            md_lines.append("")

        # FP-constrained results table
        if 'fp_constrained_summary' in statistical_summary:
            md_lines.append("## FP-Constrained Results")
            md_lines.append("")
            md_lines.append("| FP Constraint | Accuracy | Precision | Recall | Specificity | F1-Score |")
            md_lines.append("|--------------|----------|-----------|--------|-------------|----------|")
            
            for fp_key in sorted(statistical_summary['fp_constrained_summary'].keys()):
                fp_stats = statistical_summary['fp_constrained_summary'][fp_key]
                fp_val = fp_key.split('_')[1]
                md_lines.append(f"| FP={fp_val} | "
                              f"{fp_stats.get('accuracy', {}).get('mean', 0):.4f} | "
                              f"{fp_stats.get('precision', {}).get('mean', 0):.4f} | "
                              f"{fp_stats.get('recall', {}).get('mean', 0):.4f} | "
                              f"{fp_stats.get('specificity', {}).get('mean', 0):.4f} | "
                              f"{fp_stats.get('f1_score', {}).get('mean', 0):.4f} |")
            md_lines.append("")

        # Volume-level metrics table (if available)
        volume_metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        if any(m in statistical_summary for m in volume_metrics):
            md_lines.append("## Volume-Level Metrics")
            md_lines.append("")
            md_lines.append("| Metric | Mean ± Std | 95% CI | Range |")
            md_lines.append("|--------|------------|--------|-------|")

            volume_metric_names = {
                'auc': 'AUC-ROC',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1-Score',
                'specificity': 'Specificity'
            }

            for metric_key, metric_name in volume_metric_names.items():
                if metric_key in statistical_summary:
                    stats = statistical_summary[metric_key]
                    md_lines.append(f"| **{metric_name}** | {stats['mean']:.4f} ± {stats['std']:.4f} | "
                                  f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] | "
                                  f"[{stats['min']:.4f}, {stats['max']:.4f}] |")
            md_lines.append("")

        # Individual fold results
        md_lines.append("## Individual Fold Results")
        md_lines.append("")
        
        # Determine which metrics are available
        has_patient_auc = 'patient_auc' in fold_results[0]
        has_volume_auc = 'auc' in fold_results[0]
        
        if has_patient_auc and has_volume_auc:
            md_lines.append("| Fold | Patient AUC | Volume AUC | Accuracy | Time (s) |")
            md_lines.append("|------|-------------|------------|----------|----------|")
            for i, result in enumerate(fold_results):
                md_lines.append(f"| {i + 1} | {result['patient_auc']:.4f} | {result['auc']:.4f} | "
                              f"{result['accuracy']:.4f} | {result.get('fold_training_time', 0):.1f} |")
        elif has_patient_auc:
            md_lines.append("| Fold | Patient AUC | Normal Pts | Diseased Pts | Time (s) |")
            md_lines.append("|------|-------------|------------|--------------|----------|")
            for i, result in enumerate(fold_results):
                md_lines.append(f"| {i + 1} | {result['patient_auc']:.4f} | "
                              f"{result.get('normal_patient_count', 'N/A')} | "
                              f"{result.get('diseased_patient_count', 'N/A')} | "
                              f"{result.get('fold_training_time', 0):.1f} |")
        else:
            md_lines.append("| Fold | Training Time (s) | Best Epoch |")
            md_lines.append("|------|-------------------|------------|")
            for i, result in enumerate(fold_results):
                md_lines.append(f"| {i + 1} | {result.get('fold_training_time', 0):.1f} | "
                              f"{result.get('best_epoch', 'N/A')} |")

        md_lines.append("")

        # Performance analysis
        md_lines.extend(ReportGenerator._generate_markdown_performance_analysis(fold_results, statistical_summary))

        # Conclusions
        md_lines.extend(ReportGenerator._generate_markdown_conclusions(fold_results, statistical_summary))

        return '\n'.join(md_lines)

    @staticmethod
    def _generate_executive_summary(fold_results: List[Dict], statistical_summary: Dict,
                                    config: DictConfig) -> str:
        """Generate executive summary for stakeholders"""

        md_lines = []
        md_lines.append("# Executive Summary")
        md_lines.append("")
        md_lines.append(f"**Project:** {config.run_name}")
        md_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        md_lines.append("")

        # Determine primary metric
        if 'patient_auc' in statistical_summary:
            primary_metric = 'patient_auc'
            metric_name = 'Patient-Level AUC'
        elif 'auc' in statistical_summary:
            primary_metric = 'auc'
            metric_name = 'AUC-ROC'
        else:
            md_lines.append("No performance metrics available.")
            return '\n'.join(md_lines)

        # Key results
        auc_stats = statistical_summary[primary_metric]

        md_lines.append("## Key Results")
        md_lines.append("")
        md_lines.append(f"- **Model Performance:** {metric_name} = {auc_stats['mean']:.3f} ± {auc_stats['std']:.3f}")
        
        if 'accuracy' in statistical_summary:
            acc_stats = statistical_summary['accuracy']
            md_lines.append(f"- **Classification Accuracy:** {acc_stats['mean']:.1%} ± {acc_stats['std']:.1%}")
        
        md_lines.append(f"- **Cross-Validation:** {config.k_folds}-fold with {len(fold_results)} successful runs")
        md_lines.append("")

        # Performance assessment
        if auc_stats['mean'] >= 0.9:
            performance = "Excellent"
            recommendation = "Model is ready for deployment"
        elif auc_stats['mean'] >= 0.8:
            performance = "Good"
            recommendation = "Model shows strong performance, consider additional validation"
        elif auc_stats['mean'] >= 0.7:
            performance = "Fair"
            recommendation = "Model needs improvement before deployment"
        else:
            performance = "Poor"
            recommendation = "Model requires significant improvement"

        stability = "High" if auc_stats['std'] < 0.05 else "Moderate" if auc_stats['std'] < 0.1 else "Low"

        md_lines.append("## Performance Assessment")
        md_lines.append("")
        md_lines.append(f"- **Overall Performance:** {performance}")
        md_lines.append(f"- **Model Stability:** {stability}")
        md_lines.append(f"- **Recommendation:** {recommendation}")
        md_lines.append("")

        # Risk assessment (if confusion matrix data available)
        if 'true_positives' in fold_results[0]:
            md_lines.append("## Risk Assessment")
            md_lines.append("")

            total_fp = sum([r.get('false_positives', 0) for r in fold_results])
            total_tn = sum([r.get('true_negatives', 0) for r in fold_results])
            total_fn = sum([r.get('false_negatives', 0) for r in fold_results])
            total_tp = sum([r.get('true_positives', 0) for r in fold_results])

            fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
            fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0

            md_lines.append(f"- **False Positive Rate:** {fpr:.1%} (Risk of false alarms)")
            md_lines.append(f"- **False Negative Rate:** {fnr:.1%} (Risk of missed detections)")
            md_lines.append("")

        # Next steps
        md_lines.append("## Next Steps")
        md_lines.append("")
        if auc_stats['mean'] >= 0.85:
            md_lines.append("1. Proceed with deployment preparation")
            md_lines.append("2. Conduct final validation on holdout test set")
            md_lines.append("3. Monitor performance in production environment")
        else:
            md_lines.append("1. Model improvement required")
            md_lines.append("2. Consider hyperparameter tuning")
            md_lines.append("3. Analyze failure cases and edge conditions")

        md_lines.append("")

        return '\n'.join(md_lines)

    @staticmethod
    def _generate_performance_analysis(fold_results: List[Dict], statistical_summary: Dict) -> List[str]:
        """Generate detailed performance analysis section"""

        lines = []
        lines.append("PERFORMANCE ANALYSIS")
        lines.append("-" * 40)

        # Determine primary metric
        if 'patient_auc' in statistical_summary:
            auc_key = 'patient_auc'
            metric_name = 'Patient AUC'
        elif 'auc' in statistical_summary:
            auc_key = 'auc'
            metric_name = 'Volume AUC'
        else:
            lines.append("No AUC metrics available for analysis")
            lines.append("")
            return lines

        # Model consistency analysis
        auc_values = [r[auc_key] for r in fold_results if auc_key in r]
        if auc_values:
            auc_range = np.max(auc_values) - np.min(auc_values)

            lines.append(f"Model Consistency ({metric_name}):")
            lines.append(f"  {metric_name} Range: {auc_range:.4f}")
            lines.append(f"  Coefficient of Variation: {np.std(auc_values) / np.mean(auc_values) * 100:.2f}%")

            # Best and worst performing folds
            best_fold_idx = np.argmax(auc_values)
            worst_fold_idx = np.argmin(auc_values)

            lines.append(f"  Best Fold: {best_fold_idx + 1} ({metric_name}: {auc_values[best_fold_idx]:.4f})")
            lines.append(f"  Worst Fold: {worst_fold_idx + 1} ({metric_name}: {auc_values[worst_fold_idx]:.4f})")
            lines.append("")

        # Error analysis (if available)
        if 'normal_mean_error' in fold_results[0]:
            normal_errors = [r['normal_mean_error'] for r in fold_results]
            diseased_errors = [r['diseased_mean_error'] for r in fold_results]
            separation = np.mean(diseased_errors) - np.mean(normal_errors)

            lines.append(f"Error Analysis:")
            lines.append(f"  Average Normal Error: {np.mean(normal_errors):.6f}")
            lines.append(f"  Average Diseased Error: {np.mean(diseased_errors):.6f}")
            lines.append(f"  Error Separation: {separation:.6f}")
            lines.append("")

        return lines

    @staticmethod
    def _generate_conclusions(fold_results: List[Dict], statistical_summary: Dict) -> List[str]:
        """Generate conclusions section"""

        lines = []
        lines.append("CONCLUSIONS")
        lines.append("-" * 40)

        # Determine primary metric
        if 'patient_auc' in statistical_summary:
            auc_stats = statistical_summary['patient_auc']
            metric_name = "Patient-Level AUC"
        elif 'auc' in statistical_summary:
            auc_stats = statistical_summary['auc']
            metric_name = "AUC"
        else:
            lines.append("Insufficient metrics for conclusion generation")
            return lines

        # Performance rating
        if auc_stats['mean'] >= 0.9:
            performance = "Excellent"
            interpretation = "The model demonstrates outstanding discriminative ability."
        elif auc_stats['mean'] >= 0.8:
            performance = "Good"
            interpretation = "The model shows strong performance suitable for most applications."
        elif auc_stats['mean'] >= 0.7:
            performance = "Fair"
            interpretation = "The model has moderate performance and may need improvement."
        else:
            performance = "Poor"
            interpretation = "The model requires significant improvement before deployment."

        lines.append(f"Overall Performance ({metric_name}): {performance}")
        lines.append(f"Interpretation: {interpretation}")
        lines.append("")

        # Stability assessment
        stability = "High" if auc_stats['std'] < 0.05 else "Moderate" if auc_stats['std'] < 0.1 else "Low"
        lines.append(f"Model Stability: {stability}")

        if stability == "High":
            stability_note = "The model shows consistent performance across different data splits."
        elif stability == "Moderate":
            stability_note = "The model shows reasonable consistency with some variation."
        else:
            stability_note = "The model shows high variability, indicating potential overfitting or data issues."

        lines.append(f"Stability Note: {stability_note}")
        lines.append("")

        # Recommendations
        lines.append("Recommendations:")

        if auc_stats['mean'] >= 0.85 and stability == "High":
            lines.append("  - Model is ready for deployment")
            lines.append("  - Conduct final validation on independent test set")
            lines.append("  - Implement monitoring for production performance")
        elif auc_stats['mean'] >= 0.75:
            lines.append("  - Consider hyperparameter optimization")
            lines.append("  - Validate on larger dataset if available")
            lines.append("  - Analyze failure cases for improvement opportunities")
        else:
            lines.append("  - Investigate data quality and feature engineering")
            lines.append("  - Consider alternative model architectures")
            lines.append("  - Expand dataset or improve data collection")

        lines.append("")

        return lines

    @staticmethod
    def _generate_markdown_performance_analysis(fold_results: List[Dict], statistical_summary: Dict) -> List[str]:
        """Generate markdown performance analysis section"""

        lines = []
        lines.append("## Performance Analysis")
        lines.append("")

        # Determine primary metric
        if 'patient_auc' in statistical_summary:
            auc_key = 'patient_auc'
            metric_name = 'Patient AUC'
        elif 'auc' in statistical_summary:
            auc_key = 'auc'
            metric_name = 'Volume AUC'
        else:
            return lines

        # Model consistency
        auc_values = [r[auc_key] for r in fold_results if auc_key in r]
        if auc_values:
            auc_range = np.max(auc_values) - np.min(auc_values)
            cv = np.std(auc_values) / np.mean(auc_values) * 100

            lines.append("### Model Consistency")
            lines.append("")
            lines.append(f"- **{metric_name} Range:** {auc_range:.4f}")
            lines.append(f"- **Coefficient of Variation:** {cv:.2f}%")

            best_fold_idx = np.argmax(auc_values)
            worst_fold_idx = np.argmin(auc_values)

            lines.append(f"- **Best Performing Fold:** {best_fold_idx + 1} ({metric_name}: {auc_values[best_fold_idx]:.4f})")
            lines.append(f"- **Worst Performing Fold:** {worst_fold_idx + 1} ({metric_name}: {auc_values[worst_fold_idx]:.4f})")
            lines.append("")

        # Training efficiency
        training_times = [r.get('fold_training_time', 0) for r in fold_results]
        total_time = sum(training_times)

        if total_time > 0:
            lines.append("### Training Efficiency")
            lines.append("")
            lines.append(f"- **Total Training Time:** {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
            lines.append(f"- **Average Time per Fold:** {np.mean(training_times):.1f} seconds")
            lines.append("")

        return lines

    @staticmethod
    def _generate_markdown_conclusions(fold_results: List[Dict], statistical_summary: Dict) -> List[str]:
        """Generate markdown conclusions section"""

        lines = []
        lines.append("## Conclusions")
        lines.append("")

        # Determine primary metric
        if 'patient_auc' in statistical_summary:
            auc_stats = statistical_summary['patient_auc']
        elif 'auc' in statistical_summary:
            auc_stats = statistical_summary['auc']
        else:
            lines.append("Insufficient metrics for conclusions")
            return lines

        # Performance assessment
        if auc_stats['mean'] >= 0.9:
            performance = "Excellent"
        elif auc_stats['mean'] >= 0.8:
            performance = "Good"
        elif auc_stats['mean'] >= 0.7:
            performance = "Fair"
        else:
            performance = "Poor"

        stability = "High" if auc_stats['std'] < 0.05 else "Moderate" if auc_stats['std'] < 0.1 else "Low"

        lines.append(f"- **Overall Performance:** {performance}")
        lines.append(f"- **Model Stability:** {stability}")
        lines.append("")

        # Key metrics summary table
        lines.append("### Key Metrics Summary")
        lines.append("")
        lines.append(f"| Metric | Value | Confidence Interval |")
        lines.append(f"|--------|-------|---------------------|")

        # Add available metrics
        available_metrics = []
        if 'patient_auc' in statistical_summary:
            available_metrics.append(('patient_auc', 'Patient AUC'))
        if 'auc' in statistical_summary:
            available_metrics.append(('auc', 'Volume AUC'))
        for metric_key in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric_key in statistical_summary:
                available_metrics.append((metric_key, metric_key.replace('_', ' ').title()))

        for metric_key, metric_name in available_metrics:
            stats = statistical_summary[metric_key]
            lines.append(f"| {metric_name} | {stats['mean']:.4f} ± {stats['std']:.4f} | "
                       f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] |")

        lines.append("")

        # Final recommendations
        lines.append("### Recommendations")
        lines.append("")

        if auc_stats['mean'] >= 0.85 and auc_stats['std'] < 0.05:
            lines.append("1. **Ready for Deployment** - Model shows excellent and consistent performance")
            lines.append("2. **Final Validation** - Conduct testing on independent holdout dataset")
            lines.append("3. **Production Monitoring** - Implement performance tracking in production")
        elif auc_stats['mean'] >= 0.75:
            lines.append("1. **Optimization Phase** - Fine-tune hyperparameters for better performance")
            lines.append("2. **Extended Validation** - Test on larger or more diverse datasets")
            lines.append("3. **Error Analysis** - Investigate failure cases and edge conditions")
        else:
            lines.append("1. **Model Improvement Required** - Significant performance enhancement needed")
            lines.append("2. **Data Analysis** - Review data quality and feature engineering")
            lines.append("3. **Architecture Review** - Consider alternative model designs")

        lines.append("")
        lines.append(f"**Bottom Line:** {ReportGenerator._get_bottom_line(auc_stats, statistical_summary)}")
        lines.append("")

        return lines

    @staticmethod
    def _get_bottom_line(auc_stats: Dict, statistical_summary: Dict) -> str:
        """Generate bottom line assessment"""

        if auc_stats['mean'] >= 0.9 and auc_stats['std'] < 0.05:
            return "Excellent model performance with high consistency - recommended for production deployment."
        elif auc_stats['mean'] >= 0.8:
            return "Good model performance suitable for most applications - proceed with careful validation."
        elif auc_stats['mean'] >= 0.7:
            return "Fair model performance - improvement recommended before deployment."
        else:
            return "Poor model performance - significant improvement required before consideration for deployment."