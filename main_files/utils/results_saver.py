"""
Enhanced results saving utilities
================================
Saves individual volume anomaly scores and creates patient-wise plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultsSaver:
    """Enhanced results saving with volume-wise analysis"""

    @staticmethod
    def save_volume_scores_and_plots(model, normal_test_files, diseased_files, data_roots,
                                     transform, device, config, fold_idx, save_dir):
        """
        Save individual volume anomaly scores as .npy files and create patient plots
        """

        # Create directories
        scores_dir = os.path.join(save_dir, f'fold_{fold_idx + 1}_volume_scores')
        plots_dir = os.path.join(save_dir, f'fold_{fold_idx + 1}_patient_plots')
        os.makedirs(scores_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        logger.info(f"Saving volume-wise results for Fold {fold_idx + 1}")

        # Process normal volumes
        normal_volume_scores = ResultsSaver._process_volumes(
            model, normal_test_files, data_roots['normal'], transform, device,
            scores_dir, plots_dir, 'Normal', config.batch_size, config.num_workers
        )

        # Process diseased volumes
        diseased_volume_scores = ResultsSaver._process_volumes(
            model, diseased_files, data_roots['diseased'], transform, device,
            scores_dir, plots_dir, 'Diseased', config.batch_size, config.num_workers
        )

        # Save summary statistics
        ResultsSaver._save_summary_stats(
            normal_volume_scores, diseased_volume_scores, scores_dir, fold_idx
        )

        logger.info(f"Volume scores saved to: {scores_dir}")
        logger.info(f"Patient plots saved to: {plots_dir}")

        return normal_volume_scores, diseased_volume_scores

    @staticmethod
    def _process_volumes(model, file_list, data_root, transform, device,
                         scores_dir, plots_dir, label, batch_size, num_workers):
        """Process individual volumes and save scores/plots"""

        from torch.utils.data import DataLoader
        from data.VideoDataset import VideoDataset
        import torch.nn.functional as F
        import torch

        volume_scores = {}

        for pkl_file in file_list:
            # Extract patient ID from filename
            patient_id = pkl_file.replace('.pkl', '')

            # Create dataset for single volume
            single_dataset = VideoDataset(
                data_root=data_root,
                pkl_files=[pkl_file],
                transform=transform
            )
            single_dataloader = DataLoader(
                single_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

            # Calculate anomaly scores for this volume
            model.eval()
            volume_slice_scores = []

            with torch.no_grad():
                for item_idx, data in single_dataloader:
                    data = data.to(device)
                    recon_x, _ = model(data)

                    # Calculate reconstruction error for each slice
                    error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                    slice_errors = torch.mean(
                        error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                    )
                    volume_slice_scores.extend(slice_errors.cpu().numpy())

            volume_slice_scores = np.array(volume_slice_scores)
            volume_scores[patient_id] = volume_slice_scores

            # Save individual volume scores as .npy
            score_file = os.path.join(scores_dir, f'{patient_id}_{label.lower()}_scores.npy')
            np.save(score_file, volume_slice_scores)

            # Create patient plot
            ResultsSaver._create_patient_plot(
                volume_slice_scores, patient_id, label, plots_dir
            )

        return volume_scores

    @staticmethod
    def _create_patient_plot(scores, patient_id, label, plots_dir):
        """Create and save plot for individual patient"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Slice-wise anomaly scores
        slice_indices = np.arange(len(scores))
        ax1.plot(slice_indices, scores, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(slice_indices, scores, alpha=0.3)
        ax1.set_xlabel('Slice Index', fontweight='bold')
        ax1.set_ylabel('Anomaly Score (MSE)', fontweight='bold')
        ax1.set_title(f'{patient_id} - Slice-wise Anomaly Scores', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)

        stats_text = f'Mean: {mean_score:.6f}\nMax: {max_score:.6f}\nMin: {min_score:.6f}\nStd: {std_score:.6f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Plot 2: Distribution histogram
        ax2.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.6f}')
        ax2.set_xlabel('Anomaly Score', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title(f'{patient_id} - Score Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Set overall title
        color = 'red' if label == 'Diseased' else 'blue'
        fig.suptitle(f'Patient: {patient_id} ({label})', fontsize=16, fontweight='bold', color=color)

        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(plots_dir, f'{patient_id}_{label.lower()}_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _save_summary_stats(normal_scores, diseased_scores, scores_dir, fold_idx):
        """Save summary statistics for the fold"""

        summary_stats = {
            'fold_idx': fold_idx + 1,
            'normal_patients': list(normal_scores.keys()),
            'diseased_patients': list(diseased_scores.keys()),
            'normal_stats': {},
            'diseased_stats': {}
        }

        # Calculate statistics for normal patients
        all_normal_scores = []
        for patient_id, scores in normal_scores.items():
            patient_mean = np.mean(scores)
            patient_max = np.max(scores)
            patient_std = np.std(scores)

            summary_stats['normal_stats'][patient_id] = {
                'mean': float(patient_mean),
                'max': float(patient_max),
                'std': float(patient_std),
                'num_slices': len(scores)
            }
            all_normal_scores.extend(scores)

        # Calculate statistics for diseased patients
        all_diseased_scores = []
        for patient_id, scores in diseased_scores.items():
            patient_mean = np.mean(scores)
            patient_max = np.max(scores)
            patient_std = np.std(scores)

            summary_stats['diseased_stats'][patient_id] = {
                'mean': float(patient_mean),
                'max': float(patient_max),
                'std': float(patient_std),
                'num_slices': len(scores)
            }
            all_diseased_scores.extend(scores)

        # Overall statistics
        summary_stats['overall_stats'] = {
            'normal_overall': {
                'mean': float(np.mean(all_normal_scores)) if all_normal_scores else None,
                'std': float(np.std(all_normal_scores)) if all_normal_scores else None,
                'total_slices': len(all_normal_scores)
            },
            'diseased_overall': {
                'mean': float(np.mean(all_diseased_scores)) if all_diseased_scores else None,
                'std': float(np.std(all_diseased_scores)) if all_diseased_scores else None,
                'total_slices': len(all_diseased_scores)
            }
        }

        # Save as JSON
        import json
        stats_file = os.path.join(scores_dir, f'fold_{fold_idx + 1}_summary_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        # Save combined scores arrays
        if all_normal_scores:
            np.save(os.path.join(scores_dir, f'fold_{fold_idx + 1}_all_normal_scores.npy'),
                    np.array(all_normal_scores))
        if all_diseased_scores:
            np.save(os.path.join(scores_dir, f'fold_{fold_idx + 1}_all_diseased_scores.npy'),
                    np.array(all_diseased_scores))

        logger.info(f"Summary statistics saved for fold {fold_idx + 1}")
        logger.info(f"Normal patients: {len(normal_scores)}, Diseased patients: {len(diseased_scores)}")


def create_comprehensive_patient_overview(results_dir):
    """Create a comprehensive overview of all patients across all folds"""

    overview_dir = os.path.join(results_dir, 'patient_overview')
    os.makedirs(overview_dir, exist_ok=True)

    # Find all volume score directories
    fold_dirs = [d for d in os.listdir(results_dir) if d.startswith('fold_') and 'volume_scores' in d]

    if not fold_dirs:
        logger.warning("No fold volume score directories found")
        return

    all_patient_stats = {}

    for fold_dir in fold_dirs:
        fold_path = os.path.join(results_dir, fold_dir)
        stats_file = os.path.join(fold_path, f'{fold_dir.replace("_volume_scores", "")}_summary_stats.json')

        if os.path.exists(stats_file):
            import json
            with open(stats_file, 'r') as f:
                fold_stats = json.load(f)

            fold_num = fold_stats['fold_idx']
            all_patient_stats[f'fold_{fold_num}'] = fold_stats

    if not all_patient_stats:
        logger.warning("No patient statistics found")
        return

    # Create overview plots
    _create_patient_overview_plots(all_patient_stats, overview_dir)

    logger.info(f"Patient overview saved to: {overview_dir}")


def _create_patient_overview_plots(all_stats, overview_dir):
    """Create overview plots comparing all patients"""

    # Collect all patient data
    normal_means = []
    diseased_means = []

    for fold_key, fold_stats in all_stats.items():
        for patient_id, stats in fold_stats.get('normal_stats', {}).items():
            normal_means.append(stats['mean'])

        for patient_id, stats in fold_stats.get('diseased_stats', {}).items():
            diseased_means.append(stats['mean'])

    # Check if we have any data
    if not normal_means and not diseased_means:
        logger.warning("No patient data found for overview plots")
        return

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    # Box plot comparison (only if we have data)
    box_data = []
    box_labels = []

    if normal_means:
        box_data.append(normal_means)
        box_labels.append('Normal')
    if diseased_means:
        box_data.append(diseased_means)
        box_labels.append('Diseased')

    if box_data:
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor('lightblue' if box_labels[i] == 'Normal' else 'lightcoral')
        ax1.set_ylabel('Mean Anomaly Score')
        ax1.set_title('Patient-wise Mean Anomaly Scores')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)

    # Histogram comparison
    if normal_means:
        ax2.hist(normal_means, bins=20, alpha=0.7, label='Normal', color='lightblue')
    if diseased_means:
        ax2.hist(diseased_means, bins=20, alpha=0.7, label='Diseased', color='lightcoral')

    if normal_means or diseased_means:
        ax2.set_xlabel('Mean Anomaly Score')
        ax2.set_ylabel('Number of Patients')
        ax2.set_title('Distribution of Patient Mean Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)

    # Scatter plot
    if normal_means:
        normal_x = np.ones(len(normal_means)) + np.random.normal(0, 0.05, len(normal_means))
        ax3.scatter(normal_x, normal_means, alpha=0.6, color='blue', s=50, label='Normal')

    if diseased_means:
        diseased_x = np.ones(len(diseased_means)) * 2 + np.random.normal(0, 0.05, len(diseased_means))
        ax3.scatter(diseased_x, diseased_means, alpha=0.6, color='red', s=50, label='Diseased')

    if normal_means or diseased_means:
        ax3.set_xlim([0.5, 2.5])
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Normal', 'Diseased'])
        ax3.set_ylabel('Mean Anomaly Score')
        ax3.set_title('Individual Patient Scores')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(overview_dir, 'all_patients_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary statistics with safe handling
    summary = {}

    if normal_means:
        summary['normal_patients'] = {
            'count': len(normal_means),
            'mean_score': float(np.mean(normal_means)),
            'std_score': float(np.std(normal_means)),
            'min_score': float(np.min(normal_means)),
            'max_score': float(np.max(normal_means))
        }
    else:
        summary['normal_patients'] = {
            'count': 0,
            'mean_score': None,
            'std_score': None,
            'min_score': None,
            'max_score': None
        }

    if diseased_means:
        summary['diseased_patients'] = {
            'count': len(diseased_means),
            'mean_score': float(np.mean(diseased_means)),
            'std_score': float(np.std(diseased_means)),
            'min_score': float(np.min(diseased_means)),
            'max_score': float(np.max(diseased_means))
        }
    else:
        summary['diseased_patients'] = {
            'count': 0,
            'mean_score': None,
            'std_score': None,
            'min_score': None,
            'max_score': None
        }

    import json
    with open(os.path.join(overview_dir, 'overall_patient_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Patient overview plots created: Normal={len(normal_means)}, Diseased={len(diseased_means)}")