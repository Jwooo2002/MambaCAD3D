"""
Test-Only Evaluation Script for MemAE_VSSM3D Model
===================================================

This script loads a pre-trained model and performs testing/evaluation only.
No training is performed.

Features:
- Load pre-trained model from .pt file
- Calculate AUROC and generate ROC curve
- Visualize top 10 highest and lowest error cases (sagittal and axial views)
- Save all results and visualizations

Usage:
    python test_only.py

Then modify the parameters below as needed.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
import logging
from typing import List, Dict, Tuple
import pandas as pd
import cv2

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'main_files'))

from main_files.model import MemAE_VSSM3D
from main_files.data.VideoDataset import VideoDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ROC CURVE IMPLEMENTATION (without sklearn)
# ============================================================================

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute ROC curve without sklearn

    Args:
        y_true: True binary labels (0 or 1)
        y_score: Predicted scores

    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Thresholds used
    """
    # Sort by scores descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Get unique thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_score[threshold_idxs]

    # Count positives and negatives
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add endpoint
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    # Calculate rates
    if tps[-1] <= 0:
        logger.warning("No positive samples in y_true, cannot compute ROC curve")
        return np.array([0, 1]), np.array([0, 1]), thresholds

    tpr = tps / tps[-1]

    if fps[-1] <= 0:
        logger.warning("No negative samples in y_true")
        fpr = np.zeros_like(fps)
    else:
        fpr = fps / fps[-1]

    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute Area Under the Curve using trapezoidal rule"""
    # Sort by fpr
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    # Compute AUC using trapezoidal rule
    return np.trapz(tpr, fpr)


def compute_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC score"""
    fpr, tpr, _ = compute_roc_curve(y_true, y_score)
    return compute_auc(fpr, tpr)


# ============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# ============================================================================

# Model path - UPDATE THIS PATH TO YOUR .pt FILE
MODEL_PATH = "./trainResults/results_data_root_bs2_d32_64_128/data_0_05_8_32_CTCA_64/data_0_05_8_32_CTCA_64_20251023_231210/models/fold_1_model.pt"

# Test data directories - UPDATE THESE PATHS
NORMAL_TEST_DIR = "./dataset/v11/0.05_8_32_CTCA_64/Normal_fp1"
DISEASED_TEST_DIR = "./dataset/v11/0.05_8_32_CTCA_64/Diseased"

# Test data file lists (leave as None to use all files in directories)
# Or specify list of files like: ["D_10_patch.pkl", "D_11_patch.pkl"]
NORMAL_TEST_FILES = None  # Use all normal files
DISEASED_TEST_FILES = None  # Use all diseased files

# Model architecture parameters (must match the trained model)
MODEL_CONFIG = {
    'in_chans': 1,
    'depths': [2, 2, 2],
    'dims': [32, 64, 128],
    'ssm_d_state': 8,
    'mem_dim': 300,
    'drop_path_rate': 0.0,
    'shrink_thres': 0.0025
}


# Testing parameters
BATCH_SIZE = 2
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues
GPU_ID = '0'

# Output directory for results
OUTPUT_DIR = "./test_results_AAsdasfasdfasdfasfddafreyrffhgtfyujgfcyujcfgyudsfgsadgfasdgfasdgasdfgas"

# Number of extreme cases to visualize
NUM_EXTREME_CASES = 10

# ============================================================================
# END OF CONFIGURABLE PARAMETERS
# ============================================================================


def setup_device(gpu_id: str) -> torch.device:
    """Setup computing device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model(model_path: str, model_config: dict, device: torch.device) -> nn.Module:
    """Load pre-trained model from .pt file"""
    logger.info(f"Loading model from: {model_path}")

    # Create model architecture
    model = MemAE_VSSM3D(
        in_chans=model_config['in_chans'],
        depths=model_config['depths'],
        dims=model_config['dims'],
        ssm_d_state=model_config['ssm_d_state'],
        mem_dim=model_config['mem_dim'],
        drop_path_rate=model_config['drop_path_rate'],
        shrink_thres=model_config['shrink_thres']
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    logger.info("Model loaded successfully")
    return model


def get_test_files(data_dir: str, file_list: List[str] = None) -> List[str]:
    """Get list of test files from directory"""
    if file_list is not None:
        return file_list

    # Get all .pkl files in directory
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    logger.info(f"Found {len(files)} files in {data_dir}")
    return files


def calculate_anomaly_scores_detailed(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Calculate anomaly scores and keep reconstruction data for visualization

    Returns:
        scores: Anomaly scores for each sample
        original_data: Original input data
        reconstructed_data: Reconstructed data
        item_indices: Item indices (patch IDs) for each sample
    """
    model.eval()
    scores = []
    all_originals = []
    all_reconstructions = []
    all_item_indices = []

    with torch.no_grad():
        for item_idx, data in dataloader:
            data = data.to(device)
            recon_x, _ = model(data)

            # Calculate per-sample reconstruction error
            error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
            recon_error_per_sample = torch.mean(
                error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
            )
            scores.extend(recon_error_per_sample.cpu().numpy())

            # Store data for visualization
            all_originals.append(data.cpu().numpy())
            all_reconstructions.append(recon_x.cpu().numpy())
            all_item_indices.extend(item_idx.cpu().numpy() if isinstance(item_idx, torch.Tensor) else item_idx)

    scores = np.array(scores)
    originals = np.concatenate(all_originals, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)

    return scores, originals, reconstructions, all_item_indices


def calculate_patient_level_scores(
    normal_files: List[str],
    diseased_files: List[str],
    normal_test_dir: str,
    diseased_test_dir: str,
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Calculate anomaly scores for each patient (all volume patches per patient file)

    Returns:
        normal_patient_scores: Dict mapping patient_id -> array of volume scores
        diseased_patient_scores: Dict mapping patient_id -> array of volume scores
    """
    from main_files.data.VideoDataset import VideoDataset

    # Calculate scores for each normal patient
    normal_patient_scores = {}
    for pkl_file in normal_files:
        patient_id = pkl_file.replace('.pkl', '')

        single_dataset = VideoDataset(
            data_root=normal_test_dir,
            pkl_files=[pkl_file],
            transform=transform
        )
        single_dataloader = DataLoader(
            single_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0
        )

        patient_scores = []
        model.eval()
        with torch.no_grad():
            for item_idx, data in single_dataloader:
                data = data.to(device)
                recon_x, _ = model(data)

                error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                volume_errors = torch.mean(
                    error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                )
                patient_scores.extend(volume_errors.cpu().numpy())

        normal_patient_scores[patient_id] = np.array(patient_scores)

    # Calculate scores for each diseased patient
    diseased_patient_scores = {}
    for pkl_file in diseased_files:
        patient_id = pkl_file.replace('.pkl', '')

        single_dataset = VideoDataset(
            data_root=diseased_test_dir,
            pkl_files=[pkl_file],
            transform=transform
        )
        single_dataloader = DataLoader(
            single_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0
        )

        patient_scores = []
        model.eval()
        with torch.no_grad():
            for item_idx, data in single_dataloader:
                data = data.to(device)
                recon_x, _ = model(data)

                error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                volume_errors = torch.mean(
                    error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                )
                patient_scores.extend(volume_errors.cpu().numpy())

        diseased_patient_scores[patient_id] = np.array(patient_scores)

    return normal_patient_scores, diseased_patient_scores


def plot_auroc_curve(
    all_labels: np.ndarray,
    all_scores: np.ndarray,
    save_path: str,
    level: str = "Patient"
):
    """Generate and save AUROC curve"""
    logger.info(f"Generating {level}-level AUROC curve...")

    # Calculate ROC curve using our implementation
    fpr, tpr, thresholds = compute_roc_curve(all_labels, all_scores)
    roc_auc = compute_auc(fpr, tpr)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC curve (AUC = {roc_auc:.4f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    # Formatting
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'{level}-level Receiver Operating Characteristic (ROC) Curve',
              fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"{level}-level AUROC curve saved to: {save_path}")
    logger.info(f"{level}-level AUROC Score: {roc_auc:.4f}")

    return roc_auc, fpr, tpr, thresholds


def visualize_extreme_cases(
    scores: np.ndarray,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    labels: np.ndarray,
    item_indices: List[int],
    output_dir: str,
    num_cases: int = 10
):
    """
    Visualize top N highest and lowest error cases
    Show axial views only
    """
    logger.info(f"Visualizing top {num_cases} highest and lowest error cases...")

    # Create output directories
    high_error_dir = Path(output_dir) / "high_error_cases"
    low_error_dir = Path(output_dir) / "low_error_cases"
    high_error_dir.mkdir(parents=True, exist_ok=True)
    low_error_dir.mkdir(parents=True, exist_ok=True)

    # Get indices of highest and lowest error cases
    sorted_indices = np.argsort(scores)
    lowest_error_indices = sorted_indices[:num_cases]
    highest_error_indices = sorted_indices[-num_cases:][::-1]  # Reverse to get descending order

    # Visualize highest error cases
    logger.info("Generating visualizations for highest error cases...")
    for rank, idx in enumerate(highest_error_indices, 1):
        _visualize_single_case(
            original=originals[idx],
            reconstruction=reconstructions[idx],
            score=scores[idx],
            label=labels[idx],
            patch_id=item_indices[idx],
            rank=rank,
            case_type='high',
            output_dir=high_error_dir
        )

    # Visualize lowest error cases
    logger.info("Generating visualizations for lowest error cases...")
    for rank, idx in enumerate(lowest_error_indices, 1):
        _visualize_single_case(
            original=originals[idx],
            reconstruction=reconstructions[idx],
            score=scores[idx],
            label=labels[idx],
            patch_id=item_indices[idx],
            rank=rank,
            case_type='low',
            output_dir=low_error_dir
        )

    logger.info(f"Extreme case visualizations saved to: {output_dir}")


def _visualize_single_case(
    original: np.ndarray,
    reconstruction: np.ndarray,
    score: float,
    label: int,
    patch_id: int,
    rank: int,
    case_type: str,
    output_dir: Path
):
    """
    Visualize a single case with axial planes split into groups of 9 slices per image
    Show original and reconstruction side by side for each slice

    Args:
        original: Original data with shape (C, D, H, W)
        reconstruction: Reconstructed data with shape (C, D, H, W)
        score: Anomaly score
        label: Ground truth label (0=Normal, 1=Diseased)
        patch_id: Patch/volume ID
        rank: Ranking (1 to num_cases)
        case_type: 'high' or 'low'
        output_dir: Directory to save visualization
    """
    # Remove channel dimension if present
    if original.shape[0] == 1:
        original = original[0]
        reconstruction = reconstruction[0]

    D, H, W = original.shape

    label_str = "Diseased" if label == 1 else "Normal"
    error_type = "Highest" if case_type == 'high' else "Lowest"

    # Process all slices
    all_slice_images = []
    for d_idx in range(D):
        # Get 2D slice
        original_slice = original[d_idx, :, :]
        reconstruction_slice = reconstruction[d_idx, :, :]

        # Inverse transform: denormalize back to original HU values
        original_hu = original_slice * 700 - 300
        reconstruction_hu = reconstruction_slice * 700 - 300

        # Resize images to increase resolution (448x448) with INTER_CUBIC interpolation
        original_hu = cv2.resize(original_hu, (448, 448), interpolation=cv2.INTER_CUBIC)
        reconstruction_hu = cv2.resize(reconstruction_hu, (448, 448), interpolation=cv2.INTER_CUBIC)

        # Adjust brightness: make images darker (alpha=0.8, beta=0)
        original_hu = cv2.convertScaleAbs(original_hu, alpha=0.8, beta=0)
        reconstruction_hu = cv2.convertScaleAbs(reconstruction_hu, alpha=0.8, beta=0)

        all_slice_images.append((original_hu, reconstruction_hu, d_idx))

    # Split slices into groups of 9
    slices_per_image = 9
    num_images = (D + slices_per_image - 1) // slices_per_image  # Ceiling division

    # Create separate image for each group
    for img_idx in range(num_images):
        start_idx = img_idx * slices_per_image
        end_idx = min(start_idx + slices_per_image, D)
        group_slices = all_slice_images[start_idx:end_idx]

        # 3x6 grid layout (3 rows, each row has 3 pairs of original+reconstruction = 6 columns)
        rows = 3
        cols = 6
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

        # Title for this part
        if num_images > 1:
            fig.suptitle(f'{error_type} Error Case - Rank {rank} - Patch ID: {patch_id} (Part {img_idx + 1}/{num_images})\\n'
                         f'Anomaly Score: {score:.6f} | Ground Truth: {label_str}',
                         fontsize=18, fontweight='bold')
        else:
            fig.suptitle(f'{error_type} Error Case - Rank {rank} - Patch ID: {patch_id}\\n'
                         f'Anomaly Score: {score:.6f} | Ground Truth: {label_str}',
                         fontsize=18, fontweight='bold')

        # Plot slices in this group
        for local_idx, (orig_img, recon_img, slice_idx) in enumerate(group_slices):
            row = local_idx // 3
            col_pair = local_idx % 3
            col_orig = col_pair * 2
            col_recon = col_pair * 2 + 1

            # Original image
            im1 = axes[row, col_orig].imshow(orig_img, cmap='gray', interpolation='bicubic')
            axes[row, col_orig].set_title(f'Slice {slice_idx + 1}/{D} - Original',
                                           fontweight='bold', fontsize=10)
            axes[row, col_orig].axis('off')
            plt.colorbar(im1, ax=axes[row, col_orig], fraction=0.046)

            # Reconstruction image
            im2 = axes[row, col_recon].imshow(recon_img, cmap='gray', interpolation='bicubic')
            axes[row, col_recon].set_title(f'Slice {slice_idx + 1}/{D} - Recon',
                                            fontweight='bold', fontsize=10)
            axes[row, col_recon].axis('off')
            plt.colorbar(im2, ax=axes[row, col_recon], fraction=0.046)

        # Hide unused subplots (remaining slots in the 3x3 grid)
        for local_idx in range(len(group_slices), slices_per_image):
            row = local_idx // 3
            col_pair = local_idx % 3
            col_orig = col_pair * 2
            col_recon = col_pair * 2 + 1
            axes[row, col_orig].axis('off')
            axes[row, col_recon].axis('off')

        plt.tight_layout()

        # Save figure with part number if multiple images
        if num_images > 1:
            filename = f"rank_{rank:02d}_patch_{patch_id}_{label_str}_part{img_idx + 1}.png"
        else:
            filename = f"rank_{rank:02d}_patch_{patch_id}_{label_str}.png"
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_results_summary(
    output_dir: str,
    patient_auroc: float,
    volume_auroc: float,
    normal_patient_scores: Dict[str, np.ndarray],
    diseased_patient_scores: Dict[str, np.ndarray],
    normal_volume_scores: np.ndarray,
    diseased_volume_scores: np.ndarray
):
    """Save summary of test results"""
    summary_path = Path(output_dir) / "test_summary.txt"

    # Calculate patient-level max scores
    normal_patient_max_scores = [np.max(scores) for scores in normal_patient_scores.values()]
    diseased_patient_max_scores = [np.max(scores) for scores in diseased_patient_scores.values()]

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Normal Test Directory: {NORMAL_TEST_DIR}\n")
        f.write(f"Diseased Test Directory: {DISEASED_TEST_DIR}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Patient-level AUROC: {patient_auroc:.4f}\n")
        f.write(f"Volume-level AUROC:  {volume_auroc:.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PATIENT-LEVEL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Normal Patients: {len(normal_patient_scores)}\n")
        f.write(f"Diseased Patients: {len(diseased_patient_scores)}\n\n")

        f.write("Normal Patients - Max Scores (per patient):\n")
        for patient_id, scores in normal_patient_scores.items():
            f.write(f"  {patient_id}: max={np.max(scores):.6f}, mean={np.mean(scores):.6f}, "
                   f"volumes={len(scores)}\n")
        f.write("\n")

        f.write("Diseased Patients - Max Scores (per patient):\n")
        for patient_id, scores in diseased_patient_scores.items():
            f.write(f"  {patient_id}: max={np.max(scores):.6f}, mean={np.mean(scores):.6f}, "
                   f"volumes={len(scores)}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("VOLUME-LEVEL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Normal Volumes: {len(normal_volume_scores)}\n")
        f.write(f"Total Diseased Volumes: {len(diseased_volume_scores)}\n")
        f.write(f"Total Volumes: {len(normal_volume_scores) + len(diseased_volume_scores)}\n\n")

        f.write("Normal Volumes - Error Statistics:\n")
        f.write(f"  Mean Error:   {np.mean(normal_volume_scores):.6f}\n")
        f.write(f"  Std Error:    {np.std(normal_volume_scores):.6f}\n")
        f.write(f"  Min Error:    {np.min(normal_volume_scores):.6f}\n")
        f.write(f"  Max Error:    {np.max(normal_volume_scores):.6f}\n")
        f.write(f"  Median Error: {np.median(normal_volume_scores):.6f}\n\n")

        percentiles = [25, 50, 75, 90, 95, 99]
        f.write("  Percentiles:\n")
        for p in percentiles:
            val = np.percentile(normal_volume_scores, p)
            f.write(f"    {p}th percentile: {val:.6f}\n")
        f.write("\n")

        f.write("Diseased Volumes - Error Statistics:\n")
        f.write(f"  Mean Error:   {np.mean(diseased_volume_scores):.6f}\n")
        f.write(f"  Std Error:    {np.std(diseased_volume_scores):.6f}\n")
        f.write(f"  Min Error:    {np.min(diseased_volume_scores):.6f}\n")
        f.write(f"  Max Error:    {np.max(diseased_volume_scores):.6f}\n")
        f.write(f"  Median Error: {np.median(diseased_volume_scores):.6f}\n\n")

        f.write("  Percentiles:\n")
        for p in percentiles:
            val = np.percentile(diseased_volume_scores, p)
            f.write(f"    {p}th percentile: {val:.6f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Test summary saved to: {summary_path}")


def plot_score_distributions(
    normal_scores: np.ndarray,
    diseased_scores: np.ndarray,
    output_dir: str
):
    """Plot distribution of anomaly scores for normal and diseased samples"""
    logger.info("Plotting score distributions...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Histogram
    axes[0].hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    axes[0].hist(diseased_scores, bins=50, alpha=0.6, label='Diseased', color='red', density=True)
    axes[0].set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Plot 2: Box plot
    data_to_plot = [normal_scores, diseased_scores]
    bp = axes[1].boxplot(data_to_plot, labels=['Normal', 'Diseased'],
                         patch_artist=True, showfliers=True)

    # Color the box plots
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axes[1].set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Anomaly Score Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / "score_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Score distribution plot saved to: {save_path}")


def save_reconstruction_errors(
    normal_patient_scores: Dict[str, np.ndarray],
    diseased_patient_scores: Dict[str, np.ndarray],
    output_dir: str
):
    """
    Save patient-grouped volume-level reconstruction errors to .npy and Excel files

    Args:
        normal_patient_scores: Dict mapping patient_id -> array of volume scores
        diseased_patient_scores: Dict mapping patient_id -> array of volume scores
        output_dir: Directory to save the files
    """
    logger.info("Saving patient-grouped reconstruction errors to .npy and Excel files...")

    output_path = Path(output_dir)

    # ===== 1. Save Patient-level summary statistics =====
    # Save to CSV directly without pandas
    csv_patient_path = output_path / "patient_level_errors.csv"
    with open(csv_patient_path, 'w') as f:
        # Write header
        f.write("Patient_ID,Class,Label,Num_Volumes,Max_Error,Mean_Error,Median_Error,Min_Error,Std_Error,P25_Error,P75_Error,P95_Error\n")

        # Normal patients
        for patient_id, scores in normal_patient_scores.items():
            f.write(f"{patient_id},Normal,0,{len(scores)},{np.max(scores):.6f},{np.mean(scores):.6f},{np.median(scores):.6f}," +
                   f"{np.min(scores):.6f},{np.std(scores):.6f},{np.percentile(scores, 25):.6f},{np.percentile(scores, 75):.6f}," +
                   f"{np.percentile(scores, 95):.6f}\n")

        # Diseased patients
        for patient_id, scores in diseased_patient_scores.items():
            f.write(f"{patient_id},Diseased,1,{len(scores)},{np.max(scores):.6f},{np.mean(scores):.6f},{np.median(scores):.6f}," +
                   f"{np.min(scores):.6f},{np.std(scores):.6f},{np.percentile(scores, 25):.6f},{np.percentile(scores, 75):.6f}," +
                   f"{np.percentile(scores, 95):.6f}\n")

    logger.info(f"Patient-level errors saved to CSV: {csv_patient_path}")

    # ===== 2. Save patient-grouped volume-level errors (all volumes per patient) =====
    # Save to CSV directly without pandas
    csv_volumes_path = output_path / "patient_grouped_volume_errors.csv"
    with open(csv_volumes_path, 'w') as f:
        # Write header
        f.write("Patient_ID,Class,Label,Volume_Index_In_Patient,Reconstruction_Error\n")

        # Normal patients - all volumes
        for patient_id, scores in normal_patient_scores.items():
            for vol_idx, score in enumerate(scores):
                f.write(f"{patient_id},Normal,0,{vol_idx},{score:.6f}\n")

        # Diseased patients - all volumes
        for patient_id, scores in diseased_patient_scores.items():
            for vol_idx, score in enumerate(scores):
                f.write(f"{patient_id},Diseased,1,{vol_idx},{score:.6f}\n")

    logger.info(f"Patient-grouped volume errors saved to CSV: {csv_volumes_path}")

    # Save to .npy (preserves patient dictionary structure)
    patient_errors_dict = {
        'normal_patient_scores': normal_patient_scores,
        'diseased_patient_scores': diseased_patient_scores
    }
    npy_patient_path = output_path / "patient_grouped_volume_errors.npy"
    np.save(npy_patient_path, patient_errors_dict, allow_pickle=True)
    logger.info(f"Patient-grouped volume errors saved to .npy: {npy_patient_path}")

    logger.info("All reconstruction errors saved successfully!")


class CustomToTensor:
    """Custom ToTensor that handles numpy array compatibility issues"""
    def __call__(self, pic):
        # Ensure it's a proper numpy array
        pic = np.asarray(pic, dtype=np.float32)

        # Handle different input shapes
        if pic.ndim == 2:
            # H x W -> 1 x H x W
            pic = np.expand_dims(pic, axis=0)
        elif pic.ndim == 3:
            # H x W x C -> C x H x W
            pic = pic.transpose((2, 0, 1))

        # Create tensor using torch.tensor instead of torch.from_numpy to avoid version compatibility issues
        tensor = torch.tensor(pic, dtype=torch.float32)
        return tensor


def main():
    """Main test execution function"""
    logger.info("=" * 80)
    logger.info("STARTING TEST-ONLY EVALUATION (PATIENT-LEVEL)")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Setup device
    device = setup_device(GPU_ID)

    # Load model
    model = load_model(MODEL_PATH, MODEL_CONFIG, device)

    # Setup data transforms (MUST match training transforms!)
    transform = transforms.Compose([
        CustomToTensor(),  # Custom ToTensor to handle numpy compatibility
        transforms.Lambda(lambda x: (x + 300) / 700),  # Same normalization as training
    ])

    # Get test files
    normal_test_files = get_test_files(NORMAL_TEST_DIR, NORMAL_TEST_FILES)
    diseased_test_files = get_test_files(DISEASED_TEST_DIR, DISEASED_TEST_FILES)

    logger.info(f"Normal test files (patients): {len(normal_test_files)}")
    logger.info(f"Diseased test files (patients): {len(diseased_test_files)}")

    # ===== PATIENT-LEVEL EVALUATION =====
    logger.info("=" * 80)
    logger.info("CALCULATING PATIENT-LEVEL SCORES")
    logger.info("=" * 80)

    # Calculate patient-level scores
    normal_patient_scores, diseased_patient_scores = calculate_patient_level_scores(
        normal_test_files, diseased_test_files,
        NORMAL_TEST_DIR, DISEASED_TEST_DIR,
        model, transform, device, BATCH_SIZE
    )

    # Calculate patient-level max scores (used for patient-level AUC)
    normal_patient_max_scores = np.array([np.max(scores) for scores in normal_patient_scores.values()])
    diseased_patient_max_scores = np.array([np.max(scores) for scores in diseased_patient_scores.values()])

    patient_scores = np.concatenate([normal_patient_max_scores, diseased_patient_max_scores])
    patient_labels = np.concatenate([
        np.zeros(len(normal_patient_max_scores)),  # Normal = 0
        np.ones(len(diseased_patient_max_scores))   # Diseased = 1
    ])

    # Calculate and plot Patient-level AUROC curve
    logger.info("-" * 80)
    patient_auroc_path = output_dir / "patient_level_auroc_curve.png"
    patient_auroc, patient_fpr, patient_tpr, patient_thresholds = plot_auroc_curve(
        patient_labels, patient_scores, str(patient_auroc_path), level="Patient"
    )

    # ===== VOLUME-LEVEL EVALUATION (for visualization) =====
    logger.info("=" * 80)
    logger.info("CALCULATING VOLUME-LEVEL SCORES (for visualization)")
    logger.info("=" * 80)

    # Create datasets and dataloaders for volume-level processing
    normal_dataset = VideoDataset(
        data_root=NORMAL_TEST_DIR,
        pkl_files=normal_test_files,
        transform=transform
    )
    normal_dataloader = DataLoader(
        normal_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    diseased_dataset = VideoDataset(
        data_root=DISEASED_TEST_DIR,
        pkl_files=diseased_test_files,
        transform=transform
    )
    diseased_dataloader = DataLoader(
        diseased_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    # Calculate volume-level scores and get data for visualization
    logger.info("Processing normal volumes...")
    normal_volume_scores, normal_originals, normal_reconstructions, normal_item_indices = \
        calculate_anomaly_scores_detailed(model, normal_dataloader, device)

    logger.info("Processing diseased volumes...")
    diseased_volume_scores, diseased_originals, diseased_reconstructions, diseased_item_indices = \
        calculate_anomaly_scores_detailed(model, diseased_dataloader, device)

    # Combine volume-level data
    all_volume_scores = np.concatenate([normal_volume_scores, diseased_volume_scores])
    all_volume_labels = np.concatenate([
        np.zeros(len(normal_volume_scores)),  # Normal = 0
        np.ones(len(diseased_volume_scores))   # Diseased = 1
    ])
    all_originals = np.concatenate([normal_originals, diseased_originals], axis=0)
    all_reconstructions = np.concatenate([normal_reconstructions, diseased_reconstructions], axis=0)
    all_item_indices = normal_item_indices + diseased_item_indices

    # Calculate and plot Volume-level AUROC curve (for reference)
    logger.info("-" * 80)
    volume_auroc_path = output_dir / "volume_level_auroc_curve.png"
    volume_auroc, volume_fpr, volume_tpr, volume_thresholds = plot_auroc_curve(
        all_volume_labels, all_volume_scores, str(volume_auroc_path), level="Volume"
    )

    # Plot score distributions (patient-level)
    logger.info("-" * 80)
    plot_score_distributions(normal_patient_max_scores, diseased_patient_max_scores, str(output_dir))

    # Visualize extreme cases (using volume-level scores)
    logger.info("-" * 80)
    visualize_extreme_cases(
        scores=all_volume_scores,
        originals=all_originals,
        reconstructions=all_reconstructions,
        labels=all_volume_labels,
        item_indices=all_item_indices,
        output_dir=str(output_dir),
        num_cases=NUM_EXTREME_CASES
    )

    # Save results summary
    logger.info("-" * 80)
    save_results_summary(
        output_dir=str(output_dir),
        patient_auroc=patient_auroc,
        volume_auroc=volume_auroc,
        normal_patient_scores=normal_patient_scores,
        diseased_patient_scores=diseased_patient_scores,
        normal_volume_scores=normal_volume_scores,
        diseased_volume_scores=diseased_volume_scores
    )

    # Save reconstruction errors to .npy and Excel files
    logger.info("-" * 80)
    save_reconstruction_errors(
        normal_patient_scores=normal_patient_scores,
        diseased_patient_scores=diseased_patient_scores,
        output_dir=str(output_dir)
    )

    logger.info("=" * 80)
    logger.info("TEST EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info(f"All results saved to: {output_dir}")
    logger.info(f"Patient-level AUROC: {patient_auroc:.4f}")
    logger.info(f"Volume-level AUROC:  {volume_auroc:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
