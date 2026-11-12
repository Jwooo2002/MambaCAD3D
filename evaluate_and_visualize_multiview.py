"""
Evaluate Model and Visualize Multi-View (Axial, Sagittal, Coronal)
===================================================================

Features:
- Test on one dataset (calculate threshold)
- Visualize on another dataset (with calculated threshold)
- Multiple views: Axial (4x4), Sagittal (1x5), Coronal (1x5)
- Window/Level: exp01 (W=700, L=50) only
- Original images only (no reconstruction)

Usage:
    python evaluate_and_visualize_multiview.py
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import logging
from typing import List, Tuple
import cv2
from multiprocessing import Pool, cpu_count
import time

project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'main_files'))

from main_files.model import MemAE_VSSM3D
from main_files.data.VideoDataset import VideoDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

MODEL_PATH = "./trainResults/results_data_root_bs2_d32_64_128/data_0_05_8_32_CTCA_64/data_0_05_8_32_CTCA_64_20251023_231210/models/fold_1_model.pt"

# TEST DATA (for threshold calculation)
TEST_NORMAL_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Normal_fp1"
TEST_DISEASED_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Diseased"

# VISUALIZATION DATA (for generating images with threshold)
VIZ_NORMAL_DIR = "./main_files/dataset/v13/0.05_8_8_CTCA_64/Normal_fp1"
VIZ_DISEASED_DIR = "./main_files/dataset/v13/0.05_8_8_CTCA_64/Diseased"

MODEL_CONFIG = {
    'in_chans': 1,
    'depths': [2, 2, 2],
    'dims': [32, 64, 128],
    'ssm_d_state': 8,
    'mem_dim': 300,
    'drop_path_rate': 0.0,
    'shrink_thres': 0.0025
}

BATCH_SIZE = 2
NUM_WORKERS = 0
GPU_ID = '0'

OUTPUT_DIR = "./multiview_results_exp01"

# Window/Level setting (exp01 only)
WINDOW_WIDTH = 700
WINDOW_LEVEL = 50

# Gamma correction (>1.0 = compress dark, expand bright)
GAMMA = 1.5  # Adjust this: 1.0=linear, 1.5-2.5=emphasize bright areas

# Resolution and views
RESOLUTION_AXIAL = 448  # For 4x4 axial grid
RESOLUTION_OTHER = 640  # For sagittal/coronal
NUM_REPRESENTATIVE_SLICES = 5  # For sagittal/coronal

NUM_VIZ_WORKERS = min(10, cpu_count())

# ============================================================================


def apply_window_level(hu_data: np.ndarray, window_width: float, window_level: float, gamma: float = 1.0) -> np.ndarray:
    """Apply Window/Level to HU data with gamma correction

    Args:
        hu_data: HU values
        window_width: Window width
        window_level: Window level
        gamma: Gamma value (>1.0 = compress dark, expand bright)
    """
    min_hu = window_level - window_width / 2.0
    max_hu = window_level + window_width / 2.0
    windowed = np.clip(hu_data, min_hu, max_hu)

    if window_width != 0:
        # Normalize to 0-1 range
        normalized = (windowed - min_hu) / window_width

        # Apply gamma correction
        if gamma != 1.0:
            normalized = np.power(normalized, gamma)

        # Scale to 0-255
        normalized = normalized * 255.0
    else:
        normalized = windowed

    return normalized.astype(np.uint8)


def resize_keep_aspect(img: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = img.shape
    if h == w:
        return cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_CUBIC)
    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Compute ROC curve"""
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_score[threshold_idxs]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if tps[-1] <= 0:
        return np.array([0, 1]), np.array([0, 1]), thresholds

    tpr = tps / tps[-1]
    if fps[-1] <= 0:
        fpr = np.zeros_like(fps)
    else:
        fpr = fps / fps[-1]

    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC"""
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    return np.trapz(tpr, fpr)


def find_optimal_threshold_patient_level(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    """Find optimal threshold using Youden's J statistic at patient level"""
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    return optimal_threshold, sensitivity, specificity


def setup_device(gpu_id: str) -> torch.device:
    """Setup device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model(model_path: str, model_config: dict, device: torch.device) -> nn.Module:
    """Load model"""
    logger.info(f"Loading model from: {model_path}")
    model = MemAE_VSSM3D(
        in_chans=model_config['in_chans'],
        depths=model_config['depths'],
        dims=model_config['dims'],
        ssm_d_state=model_config['ssm_d_state'],
        mem_dim=model_config['mem_dim'],
        drop_path_rate=model_config['drop_path_rate'],
        shrink_thres=model_config['shrink_thres']
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info("Model loaded")
    return model


def get_test_files(data_dir: str) -> List[str]:
    """Get test files"""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    return files


def calculate_patient_scores(model, data_dir, pkl_files, transform, device, batch_size):
    """Calculate patient-level scores (max score per patient)"""
    patient_scores = {}

    for pkl_file in pkl_files:
        patient_id = pkl_file.replace('.pkl', '').replace('_patch', '')
        dataset = VideoDataset(data_dir, transform=transform, pkl_files=[pkl_file])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        scores = []
        model.eval()
        with torch.no_grad():
            for item_idx, data in dataloader:
                data = data.to(device)
                recon_x, _ = model(data)
                error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
                patch_errors = torch.mean(
                    error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
                )
                scores.extend(patch_errors.cpu().numpy())

        patient_scores[patient_id] = np.array(scores)

    return patient_scores


def calculate_anomaly_scores_detailed(model, dataloader, device):
    """Calculate patch scores with data"""
    model.eval()
    scores = []
    all_originals = []
    all_reconstructions = []
    all_item_indices = []

    with torch.no_grad():
        for item_idx, data in dataloader:
            data = data.to(device)
            recon_x, _ = model(data)

            error_per_pixel = F.mse_loss(recon_x, data, reduction='none')
            recon_error_per_sample = torch.mean(
                error_per_pixel.view(error_per_pixel.shape[0], -1), dim=1
            )
            scores.extend(recon_error_per_sample.cpu().numpy())

            all_originals.append(data.cpu().numpy())
            all_reconstructions.append(recon_x.cpu().numpy())
            all_item_indices.extend(item_idx.cpu().numpy() if isinstance(item_idx, torch.Tensor) else item_idx)

    scores = np.array(scores)
    originals = np.concatenate(all_originals, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)

    return scores, originals, reconstructions, all_item_indices


def get_representative_indices(total_slices: int, num_slices: int = 5) -> List[int]:
    """Get evenly spaced indices for representative slices"""
    if total_slices <= num_slices:
        return list(range(total_slices))
    indices = []
    for i in range(num_slices):
        idx = int((i + 0.5) * total_slices / num_slices)
        indices.append(idx)
    return indices


def visualize_single_patch_multiview(args):
    """Visualize single patch with axial + sagittal + coronal views (original only)"""
    volume_data, output_info = args

    original = volume_data['original']
    score = volume_data['score']
    patch_id = volume_data['patch_id']
    patient_source = volume_data['patient_source']

    output_dir = Path(output_info['output_dir'])
    window_width = output_info['window_width']
    window_level = output_info['window_level']
    gamma = output_info['gamma']

    # Create subdirectories for each view
    axial_dir = output_dir / 'axial'
    sagittal_dir = output_dir / 'sagittal'
    coronal_dir = output_dir / 'coronal'

    axial_dir.mkdir(parents=True, exist_ok=True)
    sagittal_dir.mkdir(parents=True, exist_ok=True)
    coronal_dir.mkdir(parents=True, exist_ok=True)

    # Remove channel
    if original.shape[0] == 1:
        original = original[0]

    D, H, W = original.shape

    # 1. AXIAL VIEW (4x4 grid) - original only
    _visualize_axial(original, axial_dir, patch_id, patient_source,
                     score, D, window_width, window_level, RESOLUTION_AXIAL, gamma)

    # 2. SAGITTAL VIEW (1x5 grid) - original only
    _visualize_sagittal(original, sagittal_dir, patch_id, patient_source,
                       score, W, window_width, window_level, RESOLUTION_OTHER, NUM_REPRESENTATIVE_SLICES, gamma)

    # 3. CORONAL VIEW (1x5 grid) - original only
    _visualize_coronal(original, coronal_dir, patch_id, patient_source,
                      score, H, window_width, window_level, RESOLUTION_OTHER, NUM_REPRESENTATIVE_SLICES, gamma)


def _visualize_axial(original, output_dir, patch_id, patient_source,
                     score, D, window_width, window_level, resolution, gamma):
    """Visualize axial slices (4x4 grid) - original only"""
    original_slices = []

    for d_idx in range(D):
        orig_slice = original[d_idx, :, :]

        # Denormalize to HU
        orig_hu = orig_slice * 700 - 300

        # Resize
        orig_hu = cv2.resize(orig_hu, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

        # Apply Window/Level with gamma
        orig_wl = apply_window_level(orig_hu, window_width, window_level, gamma)

        original_slices.append((orig_wl, d_idx))

    # 4x4 grid
    slices_per_image = 16
    num_images = (D + slices_per_image - 1) // slices_per_image

    for img_idx in range(num_images):
        start_idx = img_idx * slices_per_image
        end_idx = min(start_idx + slices_per_image, D)

        _save_4x4_grid(
            original_slices[start_idx:end_idx],
            output_dir, patch_id, patient_source, score,
            "axial", img_idx, num_images, D,
            window_width, window_level
        )


def _visualize_sagittal(original, output_dir, patch_id, patient_source,
                       score, W, window_width, window_level, max_size, num_rep_slices, gamma):
    """Visualize sagittal slices (1x5 grid) - original only"""
    representative_indices = get_representative_indices(W, num_rep_slices)

    original_slices = []

    for w_idx in representative_indices:
        orig_slice = original[:, :, w_idx]  # (D, H)

        # Denormalize to HU
        orig_hu = orig_slice * 700 - 300

        # Resize keeping aspect ratio
        orig_hu = resize_keep_aspect(orig_hu, max_size)

        # Apply Window/Level with gamma
        orig_wl = apply_window_level(orig_hu, window_width, window_level, gamma)

        original_slices.append((orig_wl, w_idx))

    _save_1xN_grid(original_slices, output_dir, patch_id, patient_source, score,
                   "sagittal", W, window_width, window_level)


def _visualize_coronal(original, output_dir, patch_id, patient_source,
                      score, H, window_width, window_level, max_size, num_rep_slices, gamma):
    """Visualize coronal slices (1x5 grid) - original only"""
    representative_indices = get_representative_indices(H, num_rep_slices)

    original_slices = []

    for h_idx in representative_indices:
        orig_slice = original[:, h_idx, :]  # (D, W)

        # Denormalize to HU
        orig_hu = orig_slice * 700 - 300

        # Resize keeping aspect ratio
        orig_hu = resize_keep_aspect(orig_hu, max_size)

        # Apply Window/Level with gamma
        orig_wl = apply_window_level(orig_hu, window_width, window_level, gamma)

        original_slices.append((orig_wl, h_idx))

    _save_1xN_grid(original_slices, output_dir, patch_id, patient_source, score,
                   "coronal", H, window_width, window_level)


def _save_4x4_grid(slices, output_dir, patch_id, patient_source, score,
                   img_type, img_idx, num_images, total_slices,
                   window_width, window_level):
    """Save 4x4 grid"""
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    if num_images > 1:
        title = f'{img_type.upper()} (Part {img_idx + 1}/{num_images})\n' \
               f'Patch ID: {patch_id} ({patient_source}) | Score: {score:.6f}\n' \
               f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'
    else:
        title = f'{img_type.upper()}\n' \
               f'Patch ID: {patch_id} ({patient_source}) | Score: {score:.6f}\n' \
               f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for local_idx, (slice_img, slice_idx) in enumerate(slices):
        im = axes[local_idx].imshow(slice_img, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
        axes[local_idx].set_title(f'Slice {slice_idx + 1}/{total_slices}', fontweight='bold', fontsize=10)
        axes[local_idx].axis('off')
        plt.colorbar(im, ax=axes[local_idx], fraction=0.046)

    for local_idx in range(len(slices), 16):
        axes[local_idx].axis('off')

    plt.tight_layout()

    if num_images > 1:
        filename = f"patch_{patch_id}_{patient_source}_{img_type}_part{img_idx + 1}.png"
    else:
        filename = f"patch_{patch_id}_{patient_source}_{img_type}.png"

    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_1xN_grid(slices, output_dir, patch_id, patient_source, score,
                   img_type, total_slices, window_width, window_level):
    """Save 1xN grid"""
    rows, cols = 1, len(slices)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    if len(slices) == 1:
        axes = [axes]

    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    title = f'{img_type.upper()}\n' \
           f'Patch ID: {patch_id} ({patient_source}) | Score: {score:.6f}\n' \
           f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for local_idx, (slice_img, slice_idx) in enumerate(slices):
        im = axes[local_idx].imshow(slice_img, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
        axes[local_idx].set_title(f'Slice {slice_idx + 1}/{total_slices}', fontweight='bold', fontsize=12)
        axes[local_idx].axis('off')
        plt.colorbar(im, ax=axes[local_idx], fraction=0.046)

    plt.tight_layout()

    filename = f"patch_{patch_id}_{patient_source}_{img_type}.png"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_patches_parallel(volume_data_list, output_info_list, num_workers):
    """Parallel visualization"""
    logger.info(f"Visualizing {len(volume_data_list)} patches with {num_workers} workers...")
    start_time = time.time()

    args_list = list(zip(volume_data_list, output_info_list))
    with Pool(processes=num_workers) as pool:
        pool.map(visualize_single_patch_multiview, args_list)

    elapsed = time.time() - start_time
    logger.info(f"Visualization completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


class CustomToTensor:
    """Custom ToTensor"""
    def __call__(self, pic):
        pic = np.asarray(pic, dtype=np.float32)
        if pic.ndim == 2:
            pic = np.expand_dims(pic, axis=0)
        elif pic.ndim == 3:
            pic = pic.transpose((2, 0, 1))
        return torch.tensor(pic, dtype=torch.float32)


def main():
    """Main"""
    logger.info("=" * 80)
    logger.info("EVALUATE AND VISUALIZE MULTI-VIEW")
    logger.info("=" * 80)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(GPU_ID)
    model = load_model(MODEL_PATH, MODEL_CONFIG, device)

    transform = transforms.Compose([
        CustomToTensor(),
        transforms.Lambda(lambda x: (x + 300) / 700),
    ])

    # ========================================================================
    # STEP 1: Calculate threshold and find high-score patch indices from TEST data
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: Calculate Threshold and Find High-Score Patches from TEST Data")
    logger.info(f"  Test Normal: {TEST_NORMAL_DIR}")
    logger.info(f"  Test Diseased: {TEST_DISEASED_DIR}")
    logger.info("=" * 80)

    test_normal_files = get_test_files(TEST_NORMAL_DIR)
    test_diseased_files = get_test_files(TEST_DISEASED_DIR)

    # Calculate patient-level threshold
    normal_patient_scores = calculate_patient_scores(
        model, TEST_NORMAL_DIR, test_normal_files, transform, device, BATCH_SIZE
    )
    diseased_patient_scores = calculate_patient_scores(
        model, TEST_DISEASED_DIR, test_diseased_files, transform, device, BATCH_SIZE
    )

    normal_patient_max = np.array([np.max(scores) for scores in normal_patient_scores.values()])
    diseased_patient_max = np.array([np.max(scores) for scores in diseased_patient_scores.values()])

    patient_scores = np.concatenate([normal_patient_max, diseased_patient_max])
    patient_labels = np.concatenate([np.zeros(len(normal_patient_max)), np.ones(len(diseased_patient_max))])

    threshold, sensitivity, specificity = find_optimal_threshold_patient_level(patient_labels, patient_scores)
    logger.info(f"Optimal threshold: {threshold:.6f}")
    logger.info(f"Sensitivity: {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

    # Calculate all patch scores from TEST data to find high-score indices
    logger.info("Calculating all TEST patch scores...")
    test_normal_dataset = VideoDataset(TEST_NORMAL_DIR, transform=transform, pkl_files=test_normal_files)
    test_diseased_dataset = VideoDataset(TEST_DISEASED_DIR, transform=transform, pkl_files=test_diseased_files)

    test_normal_loader = DataLoader(test_normal_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_diseased_loader = DataLoader(test_diseased_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_normal_scores, _, _, _ = calculate_anomaly_scores_detailed(model, test_normal_loader, device)
    test_diseased_scores, _, _, _ = calculate_anomaly_scores_detailed(model, test_diseased_loader, device)

    test_all_scores = np.concatenate([test_normal_scores, test_diseased_scores])
    test_all_sources = ['Normal'] * len(test_normal_scores) + ['Diseased'] * len(test_diseased_scores)

    # Find indices exceeding threshold in TEST data
    test_high_score_mask = test_all_scores > threshold
    test_high_score_indices = np.where(test_high_score_mask)[0]

    logger.info(f"Found {len(test_high_score_indices)} patches in TEST data exceeding threshold")
    logger.info(f"These patch indices will be used to select from VIZ data")

    # ========================================================================
    # STEP 2: Load VIZ data and visualize patches at same indices
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: Load VIZ Data and Visualize Patches at Same Indices")
    logger.info(f"  Viz Normal: {VIZ_NORMAL_DIR}")
    logger.info(f"  Viz Diseased: {VIZ_DISEASED_DIR}")
    logger.info("=" * 80)

    viz_normal_files = get_test_files(VIZ_NORMAL_DIR)
    viz_diseased_files = get_test_files(VIZ_DISEASED_DIR)

    viz_normal_dataset = VideoDataset(VIZ_NORMAL_DIR, transform=transform, pkl_files=viz_normal_files)
    viz_diseased_dataset = VideoDataset(VIZ_DISEASED_DIR, transform=transform, pkl_files=viz_diseased_files)

    viz_normal_loader = DataLoader(viz_normal_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    viz_diseased_loader = DataLoader(viz_diseased_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    viz_normal_scores, viz_normal_orig, viz_normal_recon, viz_normal_idx = \
        calculate_anomaly_scores_detailed(model, viz_normal_loader, device)
    viz_diseased_scores, viz_diseased_orig, viz_diseased_recon, viz_diseased_idx = \
        calculate_anomaly_scores_detailed(model, viz_diseased_loader, device)

    viz_all_scores = np.concatenate([viz_normal_scores, viz_diseased_scores])
    viz_all_orig = np.concatenate([viz_normal_orig, viz_diseased_orig], axis=0)
    viz_all_recon = np.concatenate([viz_normal_recon, viz_diseased_recon], axis=0)
    viz_all_idx = viz_normal_idx + viz_diseased_idx
    viz_all_patient_sources = ['Normal'] * len(viz_normal_scores) + ['Diseased'] * len(viz_diseased_scores)

    logger.info(f"Total VIZ patches: {len(viz_all_scores)}")

    # Use the SAME indices from TEST data (not based on VIZ threshold)
    logger.info(f"Visualizing patches at indices found in TEST data: {len(test_high_score_indices)} patches")

    if len(test_high_score_indices) == 0:
        logger.warning("No patches to visualize!")
        return

    # Prepare visualization data using TEST indices
    volume_data_list = []
    output_info_list = []

    for idx in test_high_score_indices:
        if idx >= len(viz_all_orig):
            logger.warning(f"Index {idx} out of range for VIZ data (max: {len(viz_all_orig)-1}), skipping")
            continue

        volume_data_list.append({
            'original': viz_all_orig[idx],
            'reconstruction': viz_all_recon[idx],
            'score': viz_all_scores[idx],  # Using VIZ score for display
            'patch_id': idx,  # Global index
            'patient_source': viz_all_patient_sources[idx]
        })
        output_info_list.append({
            'output_dir': str(output_dir),
            'window_width': WINDOW_WIDTH,
            'window_level': WINDOW_LEVEL,
            'gamma': GAMMA
        })

    logger.info(f"Prepared {len(volume_data_list)} patches for visualization")

    # Visualize
    visualize_patches_parallel(volume_data_list, output_info_list, NUM_VIZ_WORKERS)

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-VIEW EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n\n")

        f.write("TEST DATA (for threshold and index selection):\n")
        f.write(f"  Normal: {TEST_NORMAL_DIR}\n")
        f.write(f"  Diseased: {TEST_DISEASED_DIR}\n")
        f.write(f"  Total TEST patches: {len(test_all_scores)}\n")
        f.write(f"  Threshold: {threshold:.6f}\n")
        f.write(f"  Sensitivity: {sensitivity:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write(f"  Patches exceeding threshold: {len(test_high_score_indices)}\n\n")

        f.write("VISUALIZATION DATA (same indices from TEST):\n")
        f.write(f"  Normal: {VIZ_NORMAL_DIR}\n")
        f.write(f"  Diseased: {VIZ_DISEASED_DIR}\n")
        f.write(f"  Total VIZ patches: {len(viz_all_scores)}\n")
        f.write(f"  Visualized patches: {len(volume_data_list)}\n\n")

        f.write("WINDOW/LEVEL:\n")
        f.write(f"  Window: {WINDOW_WIDTH} HU\n")
        f.write(f"  Level: {WINDOW_LEVEL} HU\n\n")

        f.write("VIEWS GENERATED:\n")
        f.write("  - Axial (4x4 grid, original only)\n")
        f.write("  - Sagittal (1x5 grid, original only)\n")
        f.write("  - Coronal (1x5 grid, original only)\n\n")

        from_normal = sum([1 for p in volume_data_list if p['patient_source'] == 'Normal'])
        from_diseased = sum([1 for p in volume_data_list if p['patient_source'] == 'Diseased'])

        f.write(f"Breakdown by patient source:\n")
        f.write(f"  From Normal: {from_normal}\n")
        f.write(f"  From Diseased: {from_diseased}\n\n")

        f.write("=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED!")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Threshold: {threshold:.6f}")
    logger.info(f"TEST patches exceeding threshold: {len(test_high_score_indices)}")
    logger.info(f"VIZ patches visualized: {len(volume_data_list)}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
