"""
Anomalous patch saving utilities
===============================
Saves original and reconstructed images for slices that exceed threshold
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import json
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AnomalyPatchSaver:
    """Save anomalous patches (original and reconstructed)"""

    @staticmethod
    def save_anomalous_patches(model, normal_test_files, diseased_files, data_roots,
                              transform, device, config, fold_idx, threshold, save_base_dir):
        """
        Save original and reconstructed volume patches for slices exceeding threshold
        """

        # Create directories for different views
        patches_dir = os.path.join(save_base_dir, f'fold_{fold_idx + 1}_anomalous_patches')

        # Original directories
        original_sagittal_dir = os.path.join(patches_dir, 'original', 'sagittal')
        original_axial_dir = os.path.join(patches_dir, 'original', 'axial')
        original_coronal_dir = os.path.join(patches_dir, 'original', 'coronal')

        # Reconstructed directories
        recon_sagittal_dir = os.path.join(patches_dir, 'reconstructed', 'sagittal')
        recon_axial_dir = os.path.join(patches_dir, 'reconstructed', 'axial')
        recon_coronal_dir = os.path.join(patches_dir, 'reconstructed', 'coronal')

        # Comparison directory
        comparison_dir = os.path.join(patches_dir, 'comparison')

        # Create all directories
        for dir_path in [original_sagittal_dir, original_axial_dir, original_coronal_dir,
                        recon_sagittal_dir, recon_axial_dir, recon_coronal_dir, comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)

        dir_structure = {
            'original': {
                'sagittal': original_sagittal_dir,
                'axial': original_axial_dir,
                'coronal': original_coronal_dir
            },
            'reconstructed': {
                'sagittal': recon_sagittal_dir,
                'axial': recon_axial_dir,
                'coronal': recon_coronal_dir
            },
            'comparison': comparison_dir
        }

        logger.info(f"Saving anomalous volume patches for Fold {fold_idx + 1} (threshold: {threshold:.6f})")

        # Process normal test files
        normal_anomaly_count = AnomalyPatchSaver._process_and_save_patches(
            model, normal_test_files, data_roots['normal'], transform, device,
            config, threshold, dir_structure, 'Normal', fold_idx
        )

        # Process diseased files
        diseased_anomaly_count = AnomalyPatchSaver._process_and_save_patches(
            model, diseased_files, data_roots['diseased'], transform, device,
            config, threshold, dir_structure, 'Diseased', fold_idx
        )

        # Create summary
        summary = {
            'fold_idx': fold_idx + 1,
            'threshold': float(threshold),  # Convert to Python float
            'normal_anomalous_patches': normal_anomaly_count,
            'diseased_anomalous_patches': diseased_anomaly_count,
            'total_anomalous_patches': normal_anomaly_count + diseased_anomaly_count
        }

        # Save summary
        import json
        summary_file = os.path.join(patches_dir, f'fold_{fold_idx + 1}_anomaly_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Anomalous volume patches saved:")
        logger.info(f"  Normal anomalous patches: {normal_anomaly_count}")
        logger.info(f"  Diseased anomalous patches: {diseased_anomaly_count}")
        logger.info(f"  Total anomalous patches: {normal_anomaly_count + diseased_anomaly_count}")
        logger.info(f"  Saved to: {patches_dir}")

        return summary

    @staticmethod
    def _process_and_save_patches(model, file_list, data_root, transform, device,
                                 config, threshold, dir_structure, label, fold_idx):
        """Process files and save anomalous volume patches (limited count)"""

        from data.VideoDataset import VideoDataset
        import cv2

        model.eval()
        total_anomaly_count = 0
        MAX_PATCHES_PER_PATIENT = 10  # 환자당 최대 10개만 저장

        for pkl_file in file_list:
            patient_id = pkl_file.replace('.pkl', '')

            # Create dataset for single volume
            single_dataset = VideoDataset(
                data_root=data_root,
                pkl_files=[pkl_file],
                transform=transform
            )
            single_dataloader = DataLoader(
                single_dataset,
                batch_size=1,  # Process one volume patch at a time
                shuffle=False,
                num_workers=0
            )

            # Collect all patches with their scores first
            patient_patches = []
            patch_idx = 0

            with torch.no_grad():
                for item_idx, data in single_dataloader:
                    data = data.to(device)
                    recon_x, _ = model(data)

                    # Calculate reconstruction error for this volume patch
                    error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                    patch_error = torch.mean(
                        error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                    )

                    # Check if this patch exceeds threshold
                    if patch_error.item() > threshold:
                        patient_patches.append({
                            'data': data.clone(),
                            'recon_x': recon_x.clone(),
                            'error': patch_error.item(),
                            'patch_idx': patch_idx
                        })

                    patch_idx += 1

            # Sort by error score (highest first) and take top N
            patient_patches.sort(key=lambda x: x['error'], reverse=True)
            top_patches = patient_patches[:MAX_PATCHES_PER_PATIENT]

            # Save only top anomalous patches
            for patch_info in top_patches:
                AnomalyPatchSaver._save_volume_patch_fast(
                    patch_info['data'], patch_info['recon_x'], patch_info['error'],
                    patient_id, patch_info['patch_idx'], label, dir_structure
                )
                total_anomaly_count += 1

            if len(top_patches) > 0:
                logger.info(f"  {patient_id} ({label}): {len(top_patches)} top anomalous patches saved (out of {len(patient_patches)} total)")

        return total_anomaly_count

    @staticmethod
    def _save_volume_patch_fast(original_data, recon_data, error_score, patient_id, patch_idx,
                               label, dir_structure):
        """Save volume patch with fast CV2 method"""

        import cv2

        # Convert tensors to numpy
        original_np = original_data.squeeze().cpu().numpy()
        recon_np = recon_data.squeeze().cpu().numpy()

        # Handle different tensor dimensions
        if len(original_np.shape) == 4:  # (B, C, H, W) or (B, C, D, H, W)
            original_np = original_np[0]  # Remove batch dimension
            recon_np = recon_np[0]

        if len(original_np.shape) == 4:  # (C, D, H, W) - volume
            original_np = original_np[0]  # Take first channel
            recon_np = recon_np[0]
        elif len(original_np.shape) == 3 and original_np.shape[0] == 1:  # (1, H, W)
            original_np = original_np[0]  # Take first channel
            recon_np = recon_np[0]

        # Denormalize if needed (reverse the (x + 300) / 700 transform)
        original_np = original_np * 700 - 300
        recon_np = recon_np * 700 - 300

        # Create base filename
        base_filename = f"{patient_id}_{label.lower()}_patch{patch_idx:03d}_score{error_score:.6f}"

        # If it's a volume (3D), save different views with CV2
        if len(original_np.shape) == 3:  # (D, H, W) volume
            AnomalyPatchSaver._save_volume_views_fast(
                original_np, recon_np, base_filename, dir_structure
            )
        else:  # 2D slice
            AnomalyPatchSaver._save_2d_slice_fast(
                original_np, recon_np, base_filename, dir_structure
            )

        # Create comparison visualization (only one comprehensive image)
        comparison_file = os.path.join(dir_structure['comparison'], f"{base_filename}_comparison.png")
        AnomalyPatchSaver._create_volume_comparison_plot_fast(
            original_np, recon_np, error_score, patient_id, patch_idx, label, comparison_file
        )

    @staticmethod
    def _save_volume_views_fast(original_vol, recon_vol, base_filename, dir_structure):
        """Save different views of a 3D volume using CV2 (faster)"""

        import cv2

        depth, height, width = original_vol.shape

        # Central slices for each view
        sagittal_slice = width // 2
        axial_slice = depth // 2
        coronal_slice = height // 2

        # Normalize to 0-255 for CV2
        def normalize_for_cv2(img):
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return (img_norm * 255).astype(np.uint8)

        # Sagittal view - save as single combined image
        original_sagittal = normalize_for_cv2(original_vol[:, :, sagittal_slice])
        recon_sagittal = normalize_for_cv2(recon_vol[:, :, sagittal_slice])

        # Combine original and reconstructed side by side
        combined_sagittal = np.hstack([original_sagittal, recon_sagittal])
        sagittal_file = os.path.join(dir_structure['original']['sagittal'], f"{base_filename}_sagittal.png")
        cv2.imwrite(sagittal_file, combined_sagittal)

        # Axial view
        original_axial = normalize_for_cv2(original_vol[axial_slice, :, :])
        recon_axial = normalize_for_cv2(recon_vol[axial_slice, :, :])

        combined_axial = np.hstack([original_axial, recon_axial])
        axial_file = os.path.join(dir_structure['original']['axial'], f"{base_filename}_axial.png")
        cv2.imwrite(axial_file, combined_axial)

        # Coronal view
        original_coronal = normalize_for_cv2(original_vol[:, coronal_slice, :])
        recon_coronal = normalize_for_cv2(recon_vol[:, coronal_slice, :])

        combined_coronal = np.hstack([original_coronal, recon_coronal])
        coronal_file = os.path.join(dir_structure['original']['coronal'], f"{base_filename}_coronal.png")
        cv2.imwrite(coronal_file, combined_coronal)

    @staticmethod
    def _save_2d_slice_fast(original_slice, recon_slice, base_filename, dir_structure):
        """Save 2D slice using CV2 (faster)"""

        import cv2

        def normalize_for_cv2(img):
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return (img_norm * 255).astype(np.uint8)

        original_norm = normalize_for_cv2(original_slice)
        recon_norm = normalize_for_cv2(recon_slice)

        # Combine side by side
        combined = np.hstack([original_norm, recon_norm])
        axial_file = os.path.join(dir_structure['original']['axial'], f"{base_filename}_axial.png")
        cv2.imwrite(axial_file, combined)

    @staticmethod
    def _create_volume_comparison_plot_fast(original, reconstructed, error_score, patient_id, patch_idx,
                                           label, save_path):
        """Create simplified comparison plot (faster)"""

        if len(original.shape) == 3:  # Volume
            # Use central slices for comparison
            depth, height, width = original.shape

            # Get central slice (axial view only for speed)
            axial_orig = original[depth // 2, :, :]
            axial_recon = reconstructed[depth // 2, :, :]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(axial_orig, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(axial_recon, cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')

            axes[2].imshow(np.abs(axial_orig - axial_recon), cmap='hot')
            axes[2].set_title('Difference')
            axes[2].axis('off')

        else:  # 2D slice
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(reconstructed, cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')

            axes[2].imshow(np.abs(original - reconstructed), cmap='hot')
            axes[2].set_title('Difference')
            axes[2].axis('off')

        # Simplified title
        color = 'red' if label == 'Diseased' else 'blue'
        fig.suptitle(f'{patient_id} ({label}) - Patch {patch_idx} - Error: {error_score:.6f}',
                    fontsize=12, fontweight='bold', color=color)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Lower DPI for speed
        plt.close()

    @staticmethod
    def create_anomaly_overview(results_dir, fold_count):
        """Create overview of all anomalous patches across folds"""
        import json

        overview_dir = os.path.join(results_dir, 'anomaly_overview')
        os.makedirs(overview_dir, exist_ok=True)

        # Collect summaries from all folds
        all_summaries = []
        total_normal_anomalies = 0
        total_diseased_anomalies = 0

        for fold_idx in range(fold_count):
            summary_file = os.path.join(
                results_dir, f'fold_{fold_idx + 1}_anomalous_patches',
                f'fold_{fold_idx + 1}_anomaly_summary.json'
            )

            if os.path.exists(summary_file):
                # import json
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    all_summaries.append(summary)
                    total_normal_anomalies += summary['normal_anomalous_slices']
                    total_diseased_anomalies += summary['diseased_anomalous_slices']

        # Create overview plot
        if all_summaries:
            AnomalyPatchSaver._create_anomaly_overview_plot(
                all_summaries, overview_dir, total_normal_anomalies, total_diseased_anomalies
            )

        # Save overall summary
        overall_summary = {
            'total_folds': len(all_summaries),
            'total_normal_anomalies': total_normal_anomalies,
            'total_diseased_anomalies': total_diseased_anomalies,
            'total_anomalies': total_normal_anomalies + total_diseased_anomalies,
            'fold_summaries': all_summaries
        }

        with open(os.path.join(overview_dir, 'overall_anomaly_summary.json'), 'w') as f:
            json.dump(overall_summary, f, indent=2)

        logger.info(f"Anomaly overview created:")
        logger.info(f"  Total normal anomalies: {total_normal_anomalies}")
        logger.info(f"  Total diseased anomalies: {total_diseased_anomalies}")
        logger.info(f"  Overview saved to: {overview_dir}")

        return overall_summary

    @staticmethod
    def _create_anomaly_overview_plot(summaries, overview_dir, total_normal, total_diseased):
        """Create overview plot of anomalies across folds"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Anomalies per fold
        folds = [s['fold_idx'] for s in summaries]
        normal_counts = [s['normal_anomalous_slices'] for s in summaries]
        diseased_counts = [s['diseased_anomalous_slices'] for s in summaries]

        x = np.arange(len(folds))
        width = 0.35

        ax1.bar(x - width/2, normal_counts, width, label='Normal', color='lightblue', alpha=0.8)
        ax1.bar(x + width/2, diseased_counts, width, label='Diseased', color='lightcoral', alpha=0.8)

        ax1.set_xlabel('Fold', fontweight='bold')
        ax1.set_ylabel('Number of Anomalous Slices', fontweight='bold')
        ax1.set_title('Anomalous Slices per Fold', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Fold {f}' for f in folds])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Overall distribution
        labels = ['Normal\nAnomalies', 'Diseased\nAnomalies']
        sizes = [total_normal, total_diseased]
        colors = ['lightblue', 'lightcoral']

        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontweight': 'bold'})
        ax2.set_title('Overall Anomaly Distribution', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(overview_dir, 'anomaly_overview.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()