"""
FP-based patient-level evaluation utilities
==========================================
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

logger = logging.getLogger(__name__)


class FPBasedEvaluator:
    """Patient-level evaluation with FP-constrained thresholds"""

    @staticmethod
    def fp_based_patient_evaluation(model: nn.Module, normal_test_files: List[str],
                                    diseased_files: List[str], data_roots: Dict[str, str],
                                    transform, device: torch.device, config: DictConfig,
                                    fold_idx: Optional[int] = None) -> Dict[str, Any]:
        """Perform patient-level evaluation with FP=0~5 constraints"""

        logger.info(f"Starting FP-based patient evaluation for fold {fold_idx + 1}")

        # Calculate patient scores
        normal_patient_scores = FPBasedEvaluator._calculate_patient_scores(
            model, normal_test_files, data_roots['normal'], transform, device, config
        )

        diseased_patient_scores = FPBasedEvaluator._calculate_patient_scores(
            model, diseased_files, data_roots['diseased'], transform, device, config
        )

        # Patient-level statistics
        normal_patient_stats = {}
        diseased_patient_stats = {}

        for patient_id, scores in normal_patient_scores.items():
            normal_patient_stats[patient_id] = {
                'mean_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
                'min_score': float(np.min(scores)),
                'std_score': float(np.std(scores)),
                'patch_count': len(scores)
            }

        for patient_id, scores in diseased_patient_scores.items():
            diseased_patient_stats[patient_id] = {
                'mean_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
                'min_score': float(np.min(scores)),
                'std_score': float(np.std(scores)),
                'patch_count': len(scores)
            }

        # Calculate patient max scores for threshold determination
        normal_patient_max_scores = [np.max(scores) for scores in normal_patient_scores.values()]
        diseased_patient_max_scores = [np.max(scores) for scores in diseased_patient_scores.values()]

        # Evaluate multiple FP constraints
        fp_results = {}
        for target_fp in range(6):  # FP = 0, 1, 2, 3, 4, 5
            result = FPBasedEvaluator._evaluate_with_fp_constraint(
                normal_patient_max_scores, diseased_patient_max_scores,
                list(normal_patient_scores.keys()), list(diseased_patient_scores.keys()),
                target_fp
            )
            fp_results[f'FP_{target_fp}'] = result

        # Calculate overall AUC using patient max scores
        patient_labels = np.concatenate([
            np.zeros(len(normal_patient_max_scores)),  # Normal = 0
            np.ones(len(diseased_patient_max_scores))  # Diseased = 1
        ])
        patient_scores = np.concatenate([normal_patient_max_scores, diseased_patient_max_scores])

        try:
            patient_auc = roc_auc_score(patient_labels, patient_scores)
        except:
            patient_auc = 0.5

        return {
            'patient_auc': patient_auc,
            'normal_patient_count': len(normal_patient_scores),
            'diseased_patient_count': len(diseased_patient_scores),
            'normal_patient_stats': normal_patient_stats,
            'diseased_patient_stats': diseased_patient_stats,
            'normal_patient_scores': normal_patient_scores,
            'diseased_patient_scores': diseased_patient_scores,
            'fp_constrained_results': fp_results,
            'patient_labels': patient_labels,
            'patient_max_scores': patient_scores
        }

    @staticmethod
    def _calculate_patient_scores(model: nn.Module, file_list: List[str], data_root: str,
                                  transform, device: torch.device, config: DictConfig) -> Dict[str, np.ndarray]:
        """Calculate anomaly scores for each patient"""

        patient_scores = {}

        for pkl_file in file_list:
            patient_id = pkl_file.replace('.pkl', '')

            single_dataset = VideoDataset(
                data_root=data_root,
                pkl_files=[pkl_file],
                transform=transform
            )
            single_dataloader = DataLoader(
                single_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers
            )

            model.eval()
            patch_scores = []

            with torch.no_grad():
                for item_idx, data in single_dataloader:
                    data = data.to(device)
                    recon_x, _ = model(data)

                    error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                    patch_error = torch.mean(
                        error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                    )
                    patch_scores.extend(patch_error.cpu().numpy())

            patient_scores[patient_id] = np.array(patch_scores)

        return patient_scores

    @staticmethod
    def _evaluate_with_fp_constraint(normal_max_scores: List[float], diseased_max_scores: List[float],
                                     normal_patient_ids: List[str], diseased_patient_ids: List[str],
                                     target_fp: int) -> Dict[str, Any]:
        """Evaluate performance with specific FP constraint"""

        normal_scores = np.array(normal_max_scores)
        diseased_scores = np.array(diseased_max_scores)

        # Sort normal scores in descending order to find threshold
        sorted_normal_scores = np.sort(normal_scores)[::-1]  # Descending

        if target_fp == 0:
            # FP = 0: threshold should be higher than all normal scores
            if len(sorted_normal_scores) > 0:
                threshold = sorted_normal_scores[0] + 1e-6
            else:
                threshold = 1.0
        elif target_fp <= len(sorted_normal_scores):
            # FP = target_fp: threshold should be set to allow exactly target_fp false positives
            threshold = (sorted_normal_scores[target_fp - 1] +
                         (sorted_normal_scores[target_fp] if target_fp < len(sorted_normal_scores) else 0)) / 2
        else:
            # target_fp > number of normal patients: all normal patients will be FP
            threshold = sorted_normal_scores[-1] - 1e-6 if len(sorted_normal_scores) > 0 else 0.0

        # Make predictions
        normal_predictions = (normal_scores > threshold).astype(int)
        diseased_predictions = (diseased_scores > threshold).astype(int)

        # Calculate metrics
        fp = np.sum(normal_predictions)  # Normal predicted as diseased
        tn = len(normal_predictions) - fp  # Normal predicted as normal
        tp = np.sum(diseased_predictions)  # Diseased predicted as diseased
        fn = len(diseased_predictions) - tp  # Diseased predicted as normal

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Identify which patients are misclassified
        fp_patients = [normal_patient_ids[i] for i, pred in enumerate(normal_predictions) if pred == 1]
        fn_patients = [diseased_patient_ids[i] for i, pred in enumerate(diseased_predictions) if pred == 0]

        return {
            'threshold': float(threshold),
            'fp': int(fp),
            'tn': int(tn),
            'tp': int(tp),
            'fn': int(fn),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(recall),  # Same as recall
            'specificity': float(specificity),
            'f1_score': float(f1),
            'fp_patients': fp_patients,
            'fn_patients': fn_patients,
            'total_normal_patients': len(normal_patient_ids),
            'total_diseased_patients': len(diseased_patient_ids)
        }

    @staticmethod
    def log_fp_results(fp_results: Dict[str, Dict], fold_idx: int):
        """Log FP-constrained results in a readable format"""

        logger.info("=" * 60)
        logger.info(f"FP-CONSTRAINED RESULTS - FOLD {fold_idx + 1}")
        logger.info("=" * 60)

        # Table header
        logger.info(
            f"{'FP':<3} {'Threshold':<12} {'TP':<3} {'TN':<3} {'FN':<3} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'Spec':<6} {'F1':<6}")
        logger.info("-" * 70)

        for fp_key in sorted(fp_results.keys()):
            result = fp_results[fp_key]
            fp_val = fp_key.split('_')[1]

            logger.info(
                f"{fp_val:<3} {result['threshold']:<12.6f} "
                f"{result['tp']:<3} {result['tn']:<3} {result['fn']:<3} "
                f"{result['accuracy']:<6.3f} {result['precision']:<6.3f} "
                f"{result['recall']:<6.3f} {result['specificity']:<6.3f} {result['f1_score']:<6.3f}"
            )

            # Log misclassified patients if any
            if result['fp_patients']:
                logger.info(f"    FP Patients: {', '.join(result['fp_patients'])}")
            if result['fn_patients']:
                logger.info(f"    FN Patients: {', '.join(result['fn_patients'])}")

        logger.info("=" * 60)