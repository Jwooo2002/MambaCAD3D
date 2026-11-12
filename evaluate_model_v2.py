"""
Enhanced Evaluation with Patient-Level Threshold
=================================================

Process:
1. Calculate patient-level scores (max patch score per patient)
2. Find optimal threshold using patient-level data (high sensitivity & specificity)
3. Visualize ALL patches exceeding threshold with different W/L settings:
   - exp01: Full HU range (W=700, L=50)
   - exp02: Standard view (W=500, L=100)

Features:
- Window/Level visualization (medical imaging standard)
- Parallel visualization for speed
- Separate original and reconstruction (4x4 grid)
- Patient-level optimal threshold

Usage:
    python evaluate_model_v2.py
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
from typing import List, Tuple, Dict
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

NORMAL_TEST_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Normal_fp1"
DISEASED_TEST_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Diseased"

NORMAL_TEST_FILES = None
DISEASED_TEST_FILES = None

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

OUTPUT_DIR = "./test_results_v2_windowlevel"

# Window/Level experiments
EXPERIMENTS = [
    {
        'name': 'exp01_full_range',
        'window': 700,
        'level': 50,
        'description': 'Full HU range [-300, 400]'
    },
    {
        'name': 'exp02_standard',
        'window': 500,
        'level': 100,
        'description': 'Standard balanced view [-150, 350]'
    }
]

RESOLUTION = 448
NUM_VIZ_WORKERS = min(10, cpu_count())

# ============================================================================


def apply_window_level(hu_data: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Apply Window/Level to HU data"""
    min_hu = window_level - window_width / 2.0
    max_hu = window_level + window_width / 2.0
    windowed = np.clip(hu_data, min_hu, max_hu)
    if window_width != 0:
        normalized = ((windowed - min_hu) / window_width) * 255.0
    else:
        normalized = windowed
    return normalized.astype(np.uint8)


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
    """
    Find optimal threshold using Youden's J statistic at patient level
    Returns: (threshold, sensitivity, specificity)
    """
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)

    # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]  # TPR
    specificity = 1 - fpr[optimal_idx]  # 1 - FPR

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


def get_test_files(data_dir: str, file_list: List[str] = None) -> List[str]:
    """Get test files"""
    if file_list is not None:
        return file_list
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    logger.info(f"Found {len(files)} files in {data_dir}")
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


def visualize_single_patch(args):
    """Visualize single patch - creates separate original and recon 4x4 grids"""
    volume_data, output_info = args

    original = volume_data['original']
    reconstruction = volume_data['reconstruction']
    score = volume_data['score']
    patch_id = volume_data['patch_id']
    patient_source = volume_data['patient_source']  # 'Normal' or 'Diseased'

    output_dir = Path(output_info['output_dir'])
    window_width = output_info['window_width']
    window_level = output_info['window_level']
    resolution = output_info['resolution']
    exp_name = output_info['exp_name']

    # Remove channel
    if original.shape[0] == 1:
        original = original[0]
        reconstruction = reconstruction[0]

    D, H, W = original.shape

    # Process slices
    original_slices = []
    recon_slices = []

    for d_idx in range(D):
        orig_slice = original[d_idx, :, :]
        recon_slice = reconstruction[d_idx, :, :]

        # Denormalize to HU
        orig_hu = orig_slice * 700 - 300
        recon_hu = recon_slice * 700 - 300

        # Resize
        orig_hu = cv2.resize(orig_hu, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        recon_hu = cv2.resize(recon_hu, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

        # Apply Window/Level
        orig_wl = apply_window_level(orig_hu, window_width, window_level)
        recon_wl = apply_window_level(recon_hu, window_width, window_level)

        original_slices.append((orig_wl, d_idx))
        recon_slices.append((recon_wl, d_idx))

    # 4x4 grid = 16 slices per image
    slices_per_image = 16
    num_images = (D + slices_per_image - 1) // slices_per_image

    for img_idx in range(num_images):
        start_idx = img_idx * slices_per_image
        end_idx = min(start_idx + slices_per_image, D)

        # Save ORIGINAL 4x4
        _save_4x4_grid(
            original_slices[start_idx:end_idx],
            output_dir, patch_id, patient_source, score,
            "original", img_idx, num_images, D,
            window_width, window_level, exp_name
        )

        # Save RECONSTRUCTION 4x4
        _save_4x4_grid(
            recon_slices[start_idx:end_idx],
            output_dir, patch_id, patient_source, score,
            "reconstruction", img_idx, num_images, D,
            window_width, window_level, exp_name
        )


def _save_4x4_grid(slices, output_dir, patch_id, patient_source, score,
                   img_type, img_idx, num_images, total_slices,
                   window_width, window_level, exp_name):
    """Save 4x4 grid"""
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    # Title
    if num_images > 1:
        title = f'{img_type.upper()} - {exp_name} (Part {img_idx + 1}/{num_images})\n' \
               f'Patch ID: {patch_id} | Score: {score:.6f} | Patient: {patient_source}\n' \
               f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'
    else:
        title = f'{img_type.upper()} - {exp_name}\n' \
               f'Patch ID: {patch_id} | Score: {score:.6f} | Patient: {patient_source}\n' \
               f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot
    for local_idx, (slice_img, slice_idx) in enumerate(slices):
        im = axes[local_idx].imshow(slice_img, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
        axes[local_idx].set_title(f'Slice {slice_idx + 1}/{total_slices}', fontweight='bold', fontsize=10)
        axes[local_idx].axis('off')
        plt.colorbar(im, ax=axes[local_idx], fraction=0.046)

    # Hide unused
    for local_idx in range(len(slices), 16):
        axes[local_idx].axis('off')

    plt.tight_layout()

    # Filename
    if num_images > 1:
        filename = f"patch_{patch_id}_{patient_source}_{img_type}_part{img_idx + 1}.png"
    else:
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
        pool.map(visualize_single_patch, args_list)

    elapsed = time.time() - start_time
    logger.info(f"Visualization completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


def visualize_experiment(all_scores, all_originals, all_recons, all_indices, all_patient_sources,
                        threshold, output_base_dir, exp_config, resolution, num_workers):
    """Visualize all patches exceeding threshold for one experiment"""
    exp_name = exp_config['name']
    window_width = exp_config['window']
    window_level = exp_config['level']
    description = exp_config['description']

    logger.info("=" * 80)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Description: {description}")
    logger.info(f"Window/Level: {window_width}/{window_level} HU")
    logger.info("=" * 80)

    high_score_mask = all_scores > threshold
    high_score_indices = np.where(high_score_mask)[0]

    logger.info(f"Found {len(high_score_indices)} patches with score > threshold ({threshold:.6f})")

    if len(high_score_indices) == 0:
        logger.warning(f"No patches exceed threshold for {exp_name}!")
        return

    exp_dir = Path(output_base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    volume_data_list = []
    output_info_list = []

    for idx in high_score_indices:
        volume_data_list.append({
            'original': all_originals[idx],
            'reconstruction': all_recons[idx],
            'score': all_scores[idx],
            'patch_id': all_indices[idx],
            'patient_source': all_patient_sources[idx]
        })
        output_info_list.append({
            'output_dir': str(exp_dir),
            'window_width': window_width,
            'window_level': window_level,
            'resolution': resolution,
            'exp_name': exp_name
        })

    visualize_patches_parallel(volume_data_list, output_info_list, num_workers)

    # Save summary
    summary_path = exp_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EXPERIMENT: {exp_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Description: {description}\n")
        f.write(f"Window Width: {window_width} HU\n")
        f.write(f"Window Level: {window_level} HU\n")
        f.write(f"Display Range: [{window_level - window_width/2:.0f}, {window_level + window_width/2:.0f}] HU\n\n")
        f.write(f"Patient-level Threshold: {threshold:.6f}\n")
        f.write(f"Total patches exceeding threshold: {len(high_score_indices)}\n\n")

        # Count by patient source
        from_normal = sum([1 for idx in high_score_indices if all_patient_sources[idx] == 'Normal'])
        from_diseased = sum([1 for idx in high_score_indices if all_patient_sources[idx] == 'Diseased'])

        f.write(f"Breakdown by patient source:\n")
        f.write(f"  From Normal patients: {from_normal}\n")
        f.write(f"  From Diseased patients: {from_diseased}\n\n")

        f.write("Patch details:\n")
        for idx in high_score_indices:
            f.write(f"  Patch {all_indices[idx]}: score={all_scores[idx]:.6f}, from {all_patient_sources[idx]} patient\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Experiment {exp_name} results saved to: {exp_dir}")


def plot_patient_roc(patient_labels, patient_scores, save_path):
    """Plot patient-level ROC curve"""
    fpr, tpr, _ = compute_roc_curve(patient_labels, patient_scores)
    auc = compute_auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'darkorange', lw=3, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlabel('FPR', fontsize=14, fontweight='bold')
    plt.ylabel('TPR', fontsize=14, fontweight='bold')
    plt.title('Patient-level ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Patient-level AUROC: {auc:.4f}")
    return auc


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
    logger.info("EVALUATION WITH PATIENT-LEVEL THRESHOLD")
    logger.info("=" * 80)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(GPU_ID)
    model = load_model(MODEL_PATH, MODEL_CONFIG, device)

    transform = transforms.Compose([
        CustomToTensor(),
        transforms.Lambda(lambda x: (x + 300) / 700),
    ])

    normal_files = get_test_files(NORMAL_TEST_DIR, NORMAL_TEST_FILES)
    diseased_files = get_test_files(DISEASED_TEST_DIR, DISEASED_TEST_FILES)

    # Step 1: Calculate patient-level scores
    logger.info("=" * 80)
    logger.info("STEP 1: Calculate Patient-Level Scores")
    logger.info("=" * 80)

    normal_patient_scores = calculate_patient_scores(
        model, NORMAL_TEST_DIR, normal_files, transform, device, BATCH_SIZE
    )
    diseased_patient_scores = calculate_patient_scores(
        model, DISEASED_TEST_DIR, diseased_files, transform, device, BATCH_SIZE
    )

    # Get patient-level max scores
    normal_patient_max = np.array([np.max(scores) for scores in normal_patient_scores.values()])
    diseased_patient_max = np.array([np.max(scores) for scores in diseased_patient_scores.values()])

    patient_scores = np.concatenate([normal_patient_max, diseased_patient_max])
    patient_labels = np.concatenate([np.zeros(len(normal_patient_max)), np.ones(len(diseased_patient_max))])

    logger.info(f"Normal patients: {len(normal_patient_max)}")
    logger.info(f"Diseased patients: {len(diseased_patient_max)}")

    # Step 2: Find optimal threshold at patient level
    logger.info("=" * 80)
    logger.info("STEP 2: Find Optimal Patient-Level Threshold")
    logger.info("=" * 80)

    roc_path = output_dir / "patient_level_roc.png"
    patient_auroc = plot_patient_roc(patient_labels, patient_scores, str(roc_path))

    threshold, sensitivity, specificity = find_optimal_threshold_patient_level(patient_labels, patient_scores)
    logger.info(f"Optimal threshold: {threshold:.6f}")
    logger.info(f"Sensitivity (TPR): {sensitivity:.4f}")
    logger.info(f"Specificity (TNR): {specificity:.4f}")

    # Step 3: Get all patch-level data
    logger.info("=" * 80)
    logger.info("STEP 3: Calculate All Patch Scores")
    logger.info("=" * 80)

    normal_dataset = VideoDataset(NORMAL_TEST_DIR, transform=transform, pkl_files=normal_files)
    diseased_dataset = VideoDataset(DISEASED_TEST_DIR, transform=transform, pkl_files=diseased_files)

    normal_loader = DataLoader(normal_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    diseased_loader = DataLoader(diseased_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    normal_scores, normal_orig, normal_recon, normal_idx = \
        calculate_anomaly_scores_detailed(model, normal_loader, device)
    diseased_scores, diseased_orig, diseased_recon, diseased_idx = \
        calculate_anomaly_scores_detailed(model, diseased_loader, device)

    # Combine with patient source labels
    all_scores = np.concatenate([normal_scores, diseased_scores])
    all_orig = np.concatenate([normal_orig, diseased_orig], axis=0)
    all_recon = np.concatenate([normal_recon, diseased_recon], axis=0)
    all_idx = normal_idx + diseased_idx
    all_patient_sources = ['Normal'] * len(normal_scores) + ['Diseased'] * len(diseased_scores)

    logger.info(f"Total patches: {len(all_scores)} (Normal: {len(normal_scores)}, Diseased: {len(diseased_scores)})")

    # Step 4: Visualize patches exceeding threshold
    logger.info("=" * 80)
    logger.info("STEP 4: Visualize High-Score Patches")
    logger.info("=" * 80)

    for exp_config in EXPERIMENTS:
        visualize_experiment(
            all_scores, all_orig, all_recon, all_idx, all_patient_sources,
            threshold, str(output_dir), exp_config, RESOLUTION, NUM_VIZ_WORKERS
        )

    # Save overall summary
    summary_path = output_dir / "overall_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n\n")

        f.write("PATIENT-LEVEL PERFORMANCE:\n")
        f.write(f"  AUROC: {patient_auroc:.4f}\n")
        f.write(f"  Optimal Threshold: {threshold:.6f}\n")
        f.write(f"  Sensitivity: {sensitivity:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n\n")

        f.write("PATCH-LEVEL STATISTICS:\n")
        f.write(f"  Total patches: {len(all_scores)}\n")
        f.write(f"    From Normal patients: {len(normal_scores)}\n")
        f.write(f"    From Diseased patients: {len(diseased_scores)}\n\n")

        high_score_count = np.sum(all_scores > threshold)
        f.write(f"  Patches exceeding threshold: {high_score_count}\n\n")

        f.write("EXPERIMENTS:\n")
        for exp in EXPERIMENTS:
            f.write(f"  {exp['name']}: W={exp['window']}, L={exp['level']} - {exp['description']}\n")
        f.write("\n" + "=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED!")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Patient-level AUROC: {patient_auroc:.4f}")
    logger.info(f"Threshold: {threshold:.6f} (Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f})")
    logger.info(f"Experiments: {', '.join([e['name'] for e in EXPERIMENTS])}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
