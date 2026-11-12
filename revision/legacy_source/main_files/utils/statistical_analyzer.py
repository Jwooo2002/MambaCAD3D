"""
Statistical analysis utilities with accuracy-based threshold optimization
======================================================================
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.metrics import roc_curve, accuracy_score
from scipy import stats
from omegaconf import DictConfig


class StatisticalAnalyzer:
    """Advanced statistical analysis utilities"""

    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval using t-distribution"""
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        h = sem * stats.t.ppf((1 + confidence_level) / 2., n - 1)
        return mean - h, mean + h

    @staticmethod
    def find_optimal_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold using Youden's Index"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold, youden_index[optimal_idx]

    @staticmethod
    def find_optimal_threshold_accuracy(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold that maximizes accuracy"""
        # Get unique thresholds from scores
        thresholds = np.unique(y_scores)

        # If too many thresholds, sample them
        if len(thresholds) > 1000:
            thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)

        best_accuracy = 0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            # Make predictions with current threshold
            y_pred = (y_scores > threshold).astype(int)

            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)

            # Update best threshold if accuracy is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold, best_accuracy

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, method: str = 'accuracy') -> Tuple[
        float, float]:
        """
        Find optimal threshold using specified method

        Args:
            y_true: True binary labels
            y_scores: Anomaly scores
            method: 'accuracy' or 'youden'

        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        if method == 'accuracy':
            return StatisticalAnalyzer.find_optimal_threshold_accuracy(y_true, y_scores)
        elif method == 'youden':
            return StatisticalAnalyzer.find_optimal_threshold_youden(y_true, y_scores)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'accuracy' or 'youden'")

    @staticmethod
    def calculate_statistical_summary(fold_results: List[Dict], config: DictConfig) -> Dict[str, Any]:
        """Calculate comprehensive statistical summary - now supports patient-level metrics"""

        # Check what metrics are available in fold_results
        if not fold_results:
            raise ValueError("fold_results is empty")

        # Detect available metrics from first fold
        sample_result = fold_results[0]

        # Patient-level metrics (always should be present)
        patient_metrics = ['patient_auc']

        # Volume-level metrics (optional - for backward compatibility)
        volume_metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']

        # Determine which metrics to analyze
        available_metrics = []
        for metric in patient_metrics + volume_metrics:
            if metric in sample_result:
                available_metrics.append(metric)

        if not available_metrics:
            raise ValueError(f"No valid metrics found in fold_results. Available keys: {list(sample_result.keys())}")

        summary = {}

        # Calculate statistics for each available metric
        for metric in available_metrics:
            values = [result[metric] for result in fold_results if metric in result]

            if not values:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation

            # Confidence interval
            ci_lower, ci_upper = StatisticalAnalyzer.calculate_confidence_interval(
                np.array(values), config.confidence_level
            )

            summary[metric] = {
                'mean': mean_val,
                'std': std_val,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': values
            }

        # FP-constrained metrics summary
        if 'fp_constrained_results' in sample_result:
            summary['fp_constrained_summary'] = {}

            # Get all FP keys from first fold
            fp_keys = list(sample_result['fp_constrained_results'].keys())

            for fp_key in fp_keys:
                fp_summary = {}
                fp_metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']

                for metric in fp_metrics:
                    values = [
                        result['fp_constrained_results'][fp_key][metric]
                        for result in fold_results
                        if fp_key in result['fp_constrained_results']
                    ]

                    if values:
                        fp_summary[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values, ddof=1),
                            'values': values
                        }

                summary['fp_constrained_summary'][fp_key] = fp_summary

        # Threshold statistics (if available)
        if 'threshold' in sample_result:
            summary['threshold_stats'] = {
                'mean': np.mean([result['threshold'] for result in fold_results]),
                'std': np.std([result['threshold'] for result in fold_results], ddof=1),
                'values': [result['threshold'] for result in fold_results]
            }

        return summary