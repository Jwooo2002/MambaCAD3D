"""
Extreme Anomaly Score Visualizer
=================================
Visualizes top 10 highest and lowest anomaly score patches with sagittal and axial views
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from tqdm import tqdm
from omegaconf import DictConfig

from data.ImageDataset2D import ImageDataset2D

logger = logging.getLogger(__name__)


class ExtremeScoreVisualizer:
    """Visualize patches with extreme (high/low) anomaly scores"""

    @staticmethod
    def calculate_scores_and_get_patches(model: nn.Module, dataloader: DataLoader,
                                        device: torch.device) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Calculate anomaly scores and get original and reconstructed patches

        Returns:
            scores: Array of anomaly scores
            original_patches: List of original patches
            reconstructed_patches: List of reconstructed patches
        """
        model.eval()
        scores = []
        original_patches = []
        reconstructed_patches = []

        with torch.no_grad():
            for item_idx, data in tqdm(dataloader, desc="Calculating scores", leave=False):
                data = data.to(device)
                recon_x, _ = model(data)

                # Calculate reconstruction error per sample
                error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                recon_error_per_sample = torch.mean(
                    error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                )

                scores.extend(recon_error_per_sample.cpu().numpy())

                # Store original and reconstructed patches
                for i in range(data.shape[0]):
                    original_patches.append(data[i].cpu().numpy())
                    reconstructed_patches.append(recon_x[i].cpu().numpy())

        return np.array(scores), original_patches, reconstructed_patches

    @staticmethod
    def visualize_volume_patch(patch: np.ndarray, title: str, save_path: str):
        """
        Visualize a volume patch with sagittal and axial views

        Args:
            patch: Volume patch of shape (C, D, H, W) - NORMALIZED
            title: Title for the plot
            save_path: Path to save the figure
        """
        # Squeeze channel dimension if present
        if patch.shape[0] == 1:
            patch = patch[0]  # Shape: (D, H, W)

        # Denormalize to original HU values
        # Forward transform was: (x + 300) / 700
        # Inverse: x * 700 - 300
        patch = patch * 700 - 300

        depth, height, width = patch.shape

        # Create figure with 2 rows: sagittal and axial views
        fig = plt.figure(figsize=(20, 8))

        # Calculate middle slices
        mid_depth = depth // 2
        mid_width = width // 2

        # Sagittal view (show multiple slices)
        sagittal_slices = [
            int(width * 0.25),
            int(width * 0.375),
            int(width * 0.5),
            int(width * 0.625),
            int(width * 0.75)
        ]

        for idx, slice_idx in enumerate(sagittal_slices):
            ax = plt.subplot(2, 5, idx + 1)
            sagittal_slice = patch[:, :, slice_idx]
            im = ax.imshow(sagittal_slice, cmap='gray', aspect='auto')
            ax.set_title(f'Sagittal Slice {slice_idx}/{width}', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Axial view (show multiple slices)
        axial_slices = [
            int(depth * 0.25),
            int(depth * 0.375),
            int(depth * 0.5),
            int(depth * 0.625),
            int(depth * 0.75)
        ]

        for idx, slice_idx in enumerate(axial_slices):
            ax = plt.subplot(2, 5, idx + 6)
            axial_slice = patch[slice_idx, :, :]
            im = ax.imshow(axial_slice, cmap='gray', aspect='auto')
            ax.set_title(f'Axial Slice {slice_idx}/{depth}', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def visualize_comparison(original: np.ndarray, reconstructed: np.ndarray,
                           score: float, patch_idx: int, save_dir: str, prefix: str):
        """
        Create comparison visualization of original and reconstructed patches

        Args:
            original: Original patch (C, D, H, W) - NORMALIZED
            reconstructed: Reconstructed patch (C, D, H, W) - NORMALIZED
            score: Anomaly score
            patch_idx: Index of the patch
            save_dir: Directory to save visualizations
            prefix: Prefix for filename (e.g., 'high' or 'low')
        """
        # Squeeze channel dimension
        if original.shape[0] == 1:
            original = original[0]
            reconstructed = reconstructed[0]

        # Denormalize to original HU values
        # Forward transform was: (x + 300) / 700
        # Inverse: x * 700 - 300
        original = original * 700 - 300
        reconstructed = reconstructed * 700 - 300

        depth, height, width = original.shape

        # Calculate difference
        difference = np.abs(original - reconstructed)

        # Create figure with 3 rows (original, reconstructed, difference) x 10 columns (5 sagittal + 5 axial)
        fig = plt.figure(figsize=(25, 12))

        # Sagittal slices
        sagittal_slices = [
            int(width * 0.25),
            int(width * 0.375),
            int(width * 0.5),
            int(width * 0.625),
            int(width * 0.75)
        ]

        # Axial slices
        axial_slices = [
            int(depth * 0.25),
            int(depth * 0.375),
            int(depth * 0.5),
            int(depth * 0.625),
            int(depth * 0.75)
        ]

        # Row 1: Original
        for idx, slice_idx in enumerate(sagittal_slices):
            ax = plt.subplot(3, 10, idx + 1)
            im = ax.imshow(original[:, :, slice_idx], cmap='gray', aspect='auto')
            if idx == 0:
                ax.set_ylabel('Original\nSagittal', fontsize=11, fontweight='bold')
            ax.set_title(f'Sag {slice_idx}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for idx, slice_idx in enumerate(axial_slices):
            ax = plt.subplot(3, 10, idx + 6)
            im = ax.imshow(original[slice_idx, :, :], cmap='gray', aspect='auto')
            if idx == 0:
                ax.set_ylabel('Original\nAxial', fontsize=11, fontweight='bold')
            ax.set_title(f'Ax {slice_idx}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 2: Reconstructed
        for idx, slice_idx in enumerate(sagittal_slices):
            ax = plt.subplot(3, 10, idx + 11)
            im = ax.imshow(reconstructed[:, :, slice_idx], cmap='gray', aspect='auto')
            if idx == 0:
                ax.set_ylabel('Reconstructed\nSagittal', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for idx, slice_idx in enumerate(axial_slices):
            ax = plt.subplot(3, 10, idx + 16)
            im = ax.imshow(reconstructed[slice_idx, :, :], cmap='gray', aspect='auto')
            if idx == 0:
                ax.set_ylabel('Reconstructed\nAxial', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 3: Difference
        for idx, slice_idx in enumerate(sagittal_slices):
            ax = plt.subplot(3, 10, idx + 21)
            im = ax.imshow(difference[:, :, slice_idx], cmap='hot', aspect='auto')
            if idx == 0:
                ax.set_ylabel('Difference\nSagittal', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for idx, slice_idx in enumerate(axial_slices):
            ax = plt.subplot(3, 10, idx + 26)
            im = ax.imshow(difference[slice_idx, :, :], cmap='hot', aspect='auto')
            if idx == 0:
                ax.set_ylabel('Difference\nAxial', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'{prefix.capitalize()} Score Patch #{patch_idx}: Score = {score:.6f}',
                    fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save comparison
        comparison_file = f'{prefix}_score_patch_{patch_idx:04d}_score_{score:.6f}.png'
        plt.savefig(f'{save_dir}/{comparison_file}', dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_extreme_score_visualizations(model: nn.Module, normal_files: List[str],
                                         diseased_files: List[str], data_roots: Dict[str, str],
                                         transform, device: torch.device, config: DictConfig,
                                         fold_idx: int, base_save_dir: str):
        """
        Save visualizations of top 10 highest and lowest anomaly score patches

        Args:
            model: Trained model
            normal_files: List of normal test files
            diseased_files: List of diseased files
            data_roots: Dictionary with data root paths
            transform: Data transformation
            device: Torch device
            config: Configuration
            fold_idx: Fold index
            base_save_dir: Base directory to save visualizations
        """
        logger.info(f"Generating extreme score visualizations for Fold {fold_idx + 1}")

        # Create datasets
        normal_dataset = ImageDataset2D(
            data_root=data_roots['normal'],
            pkl_files=normal_files,
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

        # Calculate scores for normal samples
        logger.info("Processing normal samples...")
        normal_scores, normal_originals, normal_reconstructed = \
            ExtremeScoreVisualizer.calculate_scores_and_get_patches(
                model, normal_dataloader, device
            )

        # Calculate scores for diseased samples
        logger.info("Processing diseased samples...")
        diseased_scores, diseased_originals, diseased_reconstructed = \
            ExtremeScoreVisualizer.calculate_scores_and_get_patches(
                model, diseased_dataloader, device
            )

        # Combine all scores
        all_scores = np.concatenate([normal_scores, diseased_scores])
        all_originals = normal_originals + diseased_originals
        all_reconstructed = normal_reconstructed + diseased_reconstructed

        # Get indices of top 10 highest and lowest scores
        top10_high_indices = np.argsort(all_scores)[-10:][::-1]
        top10_low_indices = np.argsort(all_scores)[:10]

        # Create save directories
        fold_vis_dir = Path(base_save_dir) / f'fold_{fold_idx + 1}_extreme_scores'

        high_dir = fold_vis_dir / 'top10_highest_scores'
        low_dir = fold_vis_dir / 'top10_lowest_scores'

        high_before_dir = high_dir / 'before'
        high_after_dir = high_dir / 'after'
        low_before_dir = low_dir / 'before'
        low_after_dir = low_dir / 'after'

        for dir_path in [high_before_dir, high_after_dir, low_before_dir, low_after_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Visualizing top 10 highest scores...")
        # Visualize top 10 highest scores
        for rank, idx in enumerate(top10_high_indices):
            score = all_scores[idx]
            original = all_originals[idx]
            reconstructed = all_reconstructed[idx]

            # Determine if normal or diseased
            if idx < len(normal_scores):
                label = 'normal'
            else:
                label = 'diseased'

            # Before (original)
            ExtremeScoreVisualizer.visualize_volume_patch(
                original,
                f'Rank {rank+1}: Original (Score: {score:.6f}, Label: {label})',
                str(high_before_dir / f'rank_{rank+1:02d}_idx_{idx:05d}_score_{score:.6f}_{label}_original.png')
            )

            # After (reconstructed)
            ExtremeScoreVisualizer.visualize_volume_patch(
                reconstructed,
                f'Rank {rank+1}: Reconstructed (Score: {score:.6f}, Label: {label})',
                str(high_after_dir / f'rank_{rank+1:02d}_idx_{idx:05d}_score_{score:.6f}_{label}_recon.png')
            )

            # Comparison
            ExtremeScoreVisualizer.visualize_comparison(
                original, reconstructed, score, idx,
                str(high_dir), f'high_rank_{rank+1:02d}'
            )

        logger.info(f"Visualizing top 10 lowest scores...")
        # Visualize top 10 lowest scores
        for rank, idx in enumerate(top10_low_indices):
            score = all_scores[idx]
            original = all_originals[idx]
            reconstructed = all_reconstructed[idx]

            # Determine if normal or diseased
            if idx < len(normal_scores):
                label = 'normal'
            else:
                label = 'diseased'

            # Before (original)
            ExtremeScoreVisualizer.visualize_volume_patch(
                original,
                f'Rank {rank+1}: Original (Score: {score:.6f}, Label: {label})',
                str(low_before_dir / f'rank_{rank+1:02d}_idx_{idx:05d}_score_{score:.6f}_{label}_original.png')
            )

            # After (reconstructed)
            ExtremeScoreVisualizer.visualize_volume_patch(
                reconstructed,
                f'Rank {rank+1}: Reconstructed (Score: {score:.6f}, Label: {label})',
                str(low_after_dir / f'rank_{rank+1:02d}_idx_{idx:05d}_score_{score:.6f}_{label}_recon.png')
            )

            # Comparison
            ExtremeScoreVisualizer.visualize_comparison(
                original, reconstructed, score, idx,
                str(low_dir), f'low_rank_{rank+1:02d}'
            )

        # Create summary statistics
        summary = {
            'fold_idx': fold_idx + 1,
            'total_samples': len(all_scores),
            'normal_samples': len(normal_scores),
            'diseased_samples': len(diseased_scores),
            'top10_highest': {
                'indices': top10_high_indices.tolist(),
                'scores': [float(all_scores[i]) for i in top10_high_indices],
                'labels': ['normal' if i < len(normal_scores) else 'diseased' for i in top10_high_indices]
            },
            'top10_lowest': {
                'indices': top10_low_indices.tolist(),
                'scores': [float(all_scores[i]) for i in top10_low_indices],
                'labels': ['normal' if i < len(normal_scores) else 'diseased' for i in top10_low_indices]
            }
        }

        # Save summary
        import json
        with open(fold_vis_dir / 'extreme_scores_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Extreme score visualizations saved to: {fold_vis_dir}")
        logger.info(f"  - Top 10 highest scores: {high_dir}")
        logger.info(f"  - Top 10 lowest scores: {low_dir}")

        return summary
