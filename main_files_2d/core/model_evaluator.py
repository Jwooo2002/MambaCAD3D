"""
Model evaluation utilities
=========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import logging
from omegaconf import DictConfig

from data.ImageDataset2D import ImageDataset2D
from utils.statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utilities"""

    @staticmethod
    def calculate_anomaly_scores(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
        """Calculate anomaly scores for a dataset"""
        model.eval()
        scores = []

        with torch.no_grad():
            for item_idx, data in dataloader:
                data = data.to(device)
                recon_x, _ = model(data)

                error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                recon_error_per_sample = torch.mean(
                    error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                )
                scores.extend(recon_error_per_sample.cpu().numpy())

        return np.array(scores)

    @staticmethod
    def comprehensive_evaluation(model: nn.Module, normal_test_files: List[str],
                                 diseased_files: List[str], data_roots: Dict[str, str],
                                 transform, device: torch.device, config: DictConfig,
                                 fold_idx: Optional[int] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation of a model"""

        # Create datasets
        normal_dataset = ImageDataset2D(
            data_root=data_roots['normal'],
            pkl_files=normal_test_files,
            transform=transform
        )
        normal_dataloader = DataLoader(
            normal_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers
        )

        diseased_dataset = ImageDataset2D(
            data_root=data_roots['diseased'],
            pkl_files=diseased_files,
            transform=transform
        )
        diseased_dataloader = DataLoader(
            diseased_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers
        )

        # Calculate anomaly scores
        normal_scores = ModelEvaluator.calculate_anomaly_scores(model, normal_dataloader, device)
        diseased_scores = ModelEvaluator.calculate_anomaly_scores(model, diseased_dataloader, device)

        # Prepare labels and data
        normal_labels = np.zeros(len(normal_scores))
        diseased_labels = np.ones(len(diseased_scores))
        all_scores = np.concatenate([normal_scores, diseased_scores])
        all_labels = np.concatenate([normal_labels, diseased_labels])

        # Calculate AUC
        auc = roc_auc_score(all_labels, all_scores)

        # Determine threshold
        if config.use_optimal_threshold:
            optimal_threshold, youden_idx = StatisticalAnalyzer.find_optimal_threshold(
                all_labels, all_scores
            )
            threshold = optimal_threshold
            logger.info(f"  Optimal threshold (Youden's Index): {threshold:.6f} (Youden: {youden_idx:.4f})")
        else:
            threshold = config.anomaly_threshold
            youden_idx = None

        # Calculate predictions and metrics
        predictions = (all_scores > threshold).astype(int)
        accuracy = accuracy_score(all_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, predictions, average='binary'
        )

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)  # Same as recall

        # Error statistics
        normal_mean_error = np.mean(normal_scores)
        normal_std_error = np.std(normal_scores)
        diseased_mean_error = np.mean(diseased_scores)
        diseased_std_error = np.std(diseased_scores)

        # Percentile statistics
        normal_percentiles = np.percentile(normal_scores, [25, 50, 75, 90, 95, 99])
        diseased_percentiles = np.percentile(diseased_scores, [25, 50, 75, 90, 95, 99])

        return {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'threshold': threshold,
            'youden_index': youden_idx,

            # Confusion matrix components
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),

            # Error statistics
            'normal_mean_error': normal_mean_error,
            'normal_std_error': normal_std_error,
            'diseased_mean_error': diseased_mean_error,
            'diseased_std_error': diseased_std_error,

            # Percentiles
            'normal_percentiles': {
                'p25': normal_percentiles[0], 'p50': normal_percentiles[1], 'p75': normal_percentiles[2],
                'p90': normal_percentiles[3], 'p95': normal_percentiles[4], 'p99': normal_percentiles[5]
            },
            'diseased_percentiles': {
                'p25': diseased_percentiles[0], 'p50': diseased_percentiles[1], 'p75': diseased_percentiles[2],
                'p90': diseased_percentiles[3], 'p95': diseased_percentiles[4], 'p99': diseased_percentiles[5]
            },

            # Raw data for further analysis
            'normal_scores': normal_scores,
            'diseased_scores': diseased_scores,
            'all_scores': all_scores,
            'all_labels': all_labels,
            'predictions': predictions
        }