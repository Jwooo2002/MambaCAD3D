"""
Patient-level anomaly evaluation utilities
=========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import logging
from omegaconf import DictConfig

from data.VideoDataset import VideoDataset
from utils.statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class PatientLevelEvaluator:
    """Patient-level anomaly evaluation"""

    @staticmethod
    def calculate_patient_anomaly_scores(model: nn.Module, dataloader: DataLoader,
                                         device: torch.device, file_list: List[str]) -> Dict[str, np.ndarray]:
        """Calculate anomaly scores for each patient (volume)"""
        model.eval()
        patient_scores = {}

        with torch.no_grad():
            for i, (item_idx, data) in enumerate(dataloader):
                # Get patient ID from file list
                patient_id = file_list[i].replace('.pkl', '')

                data = data.to(device)
                recon_x, _ = model(data)

                # Calculate reconstruction error for each slice
                error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                slice_errors = torch.mean(
                    error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                )

                patient_scores[patient_id] = slice_errors.cpu().numpy()

        return patient_scores

    @staticmethod
    def patient_level_evaluation(model: nn.Module, normal_test_files: List[str],
                                 diseased_files: List[str], data_roots: Dict[str, str],
                                 transform, device: torch.device, config: DictConfig,
                                 fold_idx: Optional[int] = None) -> Dict[str, Any]:
        """Perform patient-level evaluation"""

        # Calculate patient scores for normal data
        normal_dataset = VideoDataset(
            data_root=data_roots['normal'],
            pkl_files=normal_test_files,
            transform=transform
        )

        # Create individual dataloaders for each patient to maintain patient-file mapping
        normal_patient_scores = {}
        for pkl_file in normal_test_files:
            patient_id = pkl_file.replace('.pkl', '')
            single_dataset = VideoDataset(
                data_root=data_roots['normal'],
                pkl_files=[pkl_file],
                transform=transform
            )
            single_dataloader = DataLoader(
                single_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=config.num_workers
            )

            model.eval()
            slice_scores = []
            with torch.no_grad():
                for item_idx, data in single_dataloader:
                    data = data.to(device)
                    recon_x, _ = model(data)

                    error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                    slice_errors = torch.mean(
                        error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                    )
                    slice_scores.extend(slice_errors.cpu().numpy())

            normal_patient_scores[patient_id] = np.array(slice_scores)

        # Calculate patient scores for diseased data
        diseased_patient_scores = {}
        for pkl_file in diseased_files:
            patient_id = pkl_file.replace('.pkl', '')
            single_dataset = VideoDataset(
                data_root=data_roots['diseased'],
                pkl_files=[pkl_file],
                transform=transform
            )
            single_dataloader = DataLoader(
                single_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=config.num_workers
            )

            model.eval()
            slice_scores = []
            with torch.no_grad():
                for item_idx, data in single_dataloader:
                    data = data.to(device)
                    recon_x, _ = model(data)

                    error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                    slice_errors = torch.mean(
                        error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                    )
                    slice_scores.extend(slice_errors.cpu().numpy())

            diseased_patient_scores[patient_id] = np.array(slice_scores)

        # Calculate patient-level statistics
        normal_patient_stats = {}
        diseased_patient_stats = {}

        for patient_id, scores in normal_patient_scores.items():
            normal_patient_stats[patient_id] = {
                'mean_score': np.mean(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'std_score': np.std(scores),
                'slice_count': len(scores)
            }

        for patient_id, scores in diseased_patient_scores.items():
            diseased_patient_stats[patient_id] = {
                'mean_score': np.mean(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'std_score': np.std(scores),
                'slice_count': len(scores)
            }

        # Collect all slice scores for threshold calculation
        all_normal_scores = np.concatenate(list(normal_patient_scores.values()))
        all_diseased_scores = np.concatenate(list(diseased_patient_scores.values()))
        all_scores = np.concatenate([all_normal_scores, all_diseased_scores])
        all_labels = np.concatenate([
            np.zeros(len(all_normal_scores)),
            np.ones(len(all_diseased_scores))
        ])

        # Calculate AUC at slice level
        slice_auc = roc_auc_score(all_labels, all_scores)

        # Determine threshold
        if config.use_optimal_threshold:
            threshold_method = getattr(config, 'threshold_method', 'accuracy')
            optimal_threshold, metric_value = StatisticalAnalyzer.find_optimal_threshold(
                all_labels, all_scores, method=threshold_method
            )
            threshold = optimal_threshold
            if threshold_method == 'accuracy':
                logger.info(f"  Optimal threshold (Max Accuracy): {threshold:.6f} (Slice Accuracy: {metric_value:.4f})")
            else:
                logger.info(f"  Optimal threshold (Youden's Index): {threshold:.6f} (Youden: {metric_value:.4f})")
        else:
            threshold = config.anomaly_threshold
            metric_value = None

        # Patient-level predictions using "any slice anomaly" rule
        normal_patient_predictions = {}
        diseased_patient_predictions = {}

        for patient_id, scores in normal_patient_scores.items():
            # If ANY slice exceeds threshold, patient is anomalous
            patient_prediction = 1 if np.any(scores > threshold) else 0
            normal_patient_predictions[patient_id] = patient_prediction

        for patient_id, scores in diseased_patient_scores.items():
            # If ANY slice exceeds threshold, patient is anomalous
            patient_prediction = 1 if np.any(scores > threshold) else 0
            diseased_patient_predictions[patient_id] = patient_prediction

        # Calculate patient-level metrics
        patient_labels = np.concatenate([
            np.zeros(len(normal_patient_predictions)),  # Normal patients = 0
            np.ones(len(diseased_patient_predictions))  # Diseased patients = 1
        ])
        patient_predictions = np.concatenate([
            list(normal_patient_predictions.values()),
            list(diseased_patient_predictions.values())
        ])

        # Patient-level performance metrics
        patient_accuracy = accuracy_score(patient_labels, patient_predictions)
        patient_precision, patient_recall, patient_f1, _ = precision_recall_fscore_support(
            patient_labels, patient_predictions, average='binary', zero_division=0
        )

        # Patient-level confusion matrix
        tn, fp, fn, tp = confusion_matrix(patient_labels, patient_predictions).ravel()
        patient_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        patient_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate patient-level scores for AUC
        # Use max score per patient as the patient-level anomaly score
        normal_patient_max_scores = [np.max(scores) for scores in normal_patient_scores.values()]
        diseased_patient_max_scores = [np.max(scores) for scores in diseased_patient_scores.values()]

        patient_scores = np.concatenate([normal_patient_max_scores, diseased_patient_max_scores])
        try:
            patient_auc = roc_auc_score(patient_labels, patient_scores)
        except:
            patient_auc = 0.5  # Default when AUC cannot be calculated

        return {
            # Patient-level metrics
            'patient_auc': patient_auc,
            'patient_accuracy': patient_accuracy,
            'patient_precision': patient_precision,
            'patient_recall': patient_recall,
            'patient_f1_score': patient_f1,
            'patient_sensitivity': patient_sensitivity,
            'patient_specificity': patient_specificity,
            'patient_threshold': threshold,

            # Patient-level confusion matrix
            'patient_true_positives': int(tp),
            'patient_false_positives': int(fp),
            'patient_true_negatives': int(tn),
            'patient_false_negatives': int(fn),

            # Slice-level metrics for comparison
            'slice_auc': slice_auc,

            # Patient details
            'normal_patient_count': len(normal_patient_predictions),
            'diseased_patient_count': len(diseased_patient_predictions),
            'normal_patient_stats': normal_patient_stats,
            'diseased_patient_stats': diseased_patient_stats,
            'normal_patient_predictions': normal_patient_predictions,
            'diseased_patient_predictions': diseased_patient_predictions,

            # Raw scores for further analysis
            'normal_patient_scores': normal_patient_scores,
            'diseased_patient_scores': diseased_patient_scores,
            'patient_labels': patient_labels,
            'patient_predictions': patient_predictions,
            'patient_max_scores': patient_scores,

            # Threshold info
            'threshold_method': getattr(config, 'threshold_method', 'accuracy'),
            'threshold_metric_value': metric_value
        }