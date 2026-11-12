"""
Visualize Normal Patches with Multiple Views
=============================================

Features:
- Visualize all Normal patches automatically (or selected IDs if specified)
- Load data directly from dataset (no model inference)
- Multiple viewing planes: Axial (4x4 grid), Sagittal and Coronal (5 representative slices each)
- Two experiment settings with different Window/Level and gamma correction
- Results saved in separate directories per experiment

Usage:
    python visualize_selected_patches.py
"""

import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import logging
from typing import List
import cv2
import torch
from multiprocessing import Pool, cpu_count
import time

project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'main_files'))

from main_files.data.VideoDataset import VideoDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Data directories (same as evaluate_model_v2.py)
NORMAL_TEST_DIR = "./main_files/dataset/v13/0.05_8_8_CTCA_64/Normal_fp1"
DISEASED_TEST_DIR = "./main_files/dataset/v13/0.05_8_8_CTCA_64/Diseased"
#
# NORMAL_TEST_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Normal_fp1"
# DISEASED_TEST_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Diseased"

# IMPORTANT: Specify patch IDs to visualize
# Leave empty to visualize ALL Normal patches
SELECTED_PATCH_IDS = [
    8, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 96, 110, 218, 280, 281
    # Add your patch IDs here, e.g.:
    # 119, 122, 123, 164, ...
]

BATCH_SIZE = 2

OUTPUT_DIR = "./normal_patches_visualization_8_sel"

# Viewing planes to generate
VIEW_PLANES = [
    'axial',
               'sagittal',
               'coronal'
]

# Window/Level settings
EXPERIMENTS = [
    # {
    #     'name': 'exp01_full_range',
    #     'window': 700,
    #     'level': 50,
    #     'gamma': 1.0,
    #     'description': 'Full HU range [-300, 400]'
    # },
    {
        'name': 'exp02_standard',
        'window': 500,
        'level': 100,
        'gamma': 1.5,
        'description': 'Standard balanced view [-150, 350] with gamma=1.5'
    }
]

# Maximum dimension for display (maintains aspect ratio)
MAX_DISPLAY_SIZE = 640

# Resolution for axial view (4x4 grid)
RESOLUTION_AXIAL = 448

# Number of slices to show for sagittal/coronal views
NUM_REPRESENTATIVE_SLICES = 5

# Parallel processing
NUM_WORKERS = min(10, cpu_count())  # Use up to 10 workers

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
        return cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_LINEAR)

    # Keep aspect ratio
    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def get_test_files(data_dir: str) -> List[str]:
    """Get all test files"""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    return files


def collect_selected_patches(data_dir, pkl_files, selected_ids, transform, batch_size, patient_source):
    """Collect data for patches matching selected IDs"""
    selected_data = []

    for pkl_file in pkl_files:
        dataset = VideoDataset(data_dir, transform=transform, pkl_files=[pkl_file])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for item_idx, data in dataloader:
            # Convert to numpy
            data_np = data.cpu().numpy()
            indices = item_idx.cpu().numpy() if isinstance(item_idx, torch.Tensor) else np.array(item_idx)

            # Check if any of these patches match selected IDs
            for i, idx in enumerate(indices):
                if idx in selected_ids:
                    selected_data.append({
                        'data': data_np[i],
                        'patch_id': idx,
                        'patient_source': patient_source,
                        'patient_file': pkl_file
                    })

    return selected_data


def get_representative_indices(total_slices: int, num_slices: int = 5) -> List[int]:
    """Get evenly spaced indices for representative slices"""
    if total_slices <= num_slices:
        return list(range(total_slices))

    # Evenly divide the volume into num_slices sections
    indices = []
    for i in range(num_slices):
        idx = int((i + 0.5) * total_slices / num_slices)
        indices.append(idx)
    return indices


def visualize_patch_worker(args):
    """Worker function for parallel processing"""
    patch_data, exp_config, view_plane, output_dir, max_size, num_rep_slices, resolution_axial = args
    return visualize_patch(patch_data, exp_config, view_plane, output_dir, max_size, num_rep_slices, resolution_axial)


def visualize_patch(patch_data, exp_config, view_plane, output_dir, max_size, num_rep_slices, resolution_axial):
    """Visualize single patch for one experiment and view plane"""

    data = patch_data['data']
    patch_id = patch_data['patch_id']
    patient_source = patch_data['patient_source']
    patient_file = patch_data['patient_file']

    window_width = exp_config['window']
    window_level = exp_config['level']
    gamma = exp_config['gamma']
    exp_name = exp_config['name']

    # Remove channel if present
    if data.shape[0] == 1:
        data = data[0]

    D, H, W = data.shape

    # Process slices based on view plane
    slices_data = []

    if view_plane == 'axial':
        # Axial: slice along D axis (4x4 grid)
        for d_idx in range(D):
            slice_2d = data[d_idx, :, :]  # (H, W)

            # Denormalize to HU
            slice_hu = slice_2d * 700 - 300

            # Resize to fixed resolution for axial
            slice_hu = cv2.resize(slice_hu, (resolution_axial, resolution_axial), interpolation=cv2.INTER_CUBIC)

            # Apply Window/Level with gamma
            slice_wl = apply_window_level(slice_hu, window_width, window_level, gamma)

            slices_data.append((slice_wl, d_idx))

        # Save as 4x4 grid
        _save_4x4_grid(
            slices_data, output_dir, patch_id, patient_source, patient_file,
            D, window_width, window_level, exp_name, view_plane
        )
        return

    elif view_plane == 'sagittal':
        # Sagittal: slice along W axis
        total_slices = W
        representative_indices = get_representative_indices(W, num_rep_slices)

        for w_idx in representative_indices:
            slice_2d = data[:, :, w_idx]  # (D, H)

            # Denormalize to HU
            slice_hu = slice_2d * 700 - 300

            # Resize keeping aspect ratio
            slice_hu = resize_keep_aspect(slice_hu, max_size)

            # Apply Window/Level with gamma
            slice_wl = apply_window_level(slice_hu, window_width, window_level, gamma)

            slices_data.append((slice_wl, w_idx))

    elif view_plane == 'coronal':
        # Coronal: slice along H axis
        total_slices = H
        representative_indices = get_representative_indices(H, num_rep_slices)

        for h_idx in representative_indices:
            slice_2d = data[:, h_idx, :]  # (D, W)

            # Denormalize to HU
            slice_hu = slice_2d * 700 - 300

            # Resize keeping aspect ratio
            slice_hu = resize_keep_aspect(slice_hu, max_size)

            # Apply Window/Level with gamma
            slice_wl = apply_window_level(slice_hu, window_width, window_level, gamma)

            slices_data.append((slice_wl, h_idx))

    # Save as 1xN grid
    _save_grid(
        slices_data, output_dir, patch_id, patient_source, patient_file,
        total_slices, window_width, window_level, exp_name, view_plane
    )


def _save_grid(slices, output_dir, patch_id, patient_source, patient_file,
               total_slices, window_width, window_level, exp_name, view_plane):
    """Save grid of slices (1xN for sagittal/coronal)"""
    rows, cols = 1, len(slices)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Handle single slice case
    if len(slices) == 1:
        axes = [axes]

    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    patient_file_base = patient_file.replace('.pkl', '').replace('_patch', '')

    # Title
    title = f'{view_plane.upper()} - {exp_name}\n' \
           f'Patch ID: {patch_id} | From: {patient_file_base} ({patient_source})\n' \
           f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot
    for local_idx, (slice_img, slice_idx) in enumerate(slices):
        im = axes[local_idx].imshow(slice_img, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
        axes[local_idx].set_title(f'Slice {slice_idx + 1}/{total_slices}', fontweight='bold', fontsize=12)
        axes[local_idx].axis('off')
        plt.colorbar(im, ax=axes[local_idx], fraction=0.046)

    plt.tight_layout()

    # Filename
    filename = f"patch_{patch_id}_{patient_file_base}_{patient_source}_{view_plane}.png"

    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_4x4_grid(slices, output_dir, patch_id, patient_source, patient_file,
                   total_slices, window_width, window_level, exp_name, view_plane):
    """Save 4x4 grid for axial slices"""
    rows, cols = 4, 4
    slices_per_image = 16
    num_images = (len(slices) + slices_per_image - 1) // slices_per_image

    patient_file_base = patient_file.replace('.pkl', '').replace('_patch', '')
    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    for img_idx in range(num_images):
        start_idx = img_idx * slices_per_image
        end_idx = min(start_idx + slices_per_image, len(slices))
        current_slices = slices[start_idx:end_idx]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        if num_images > 1:
            title = f'{view_plane.upper()} (Part {img_idx + 1}/{num_images}) - {exp_name}\n' \
                   f'Patch ID: {patch_id} | From: {patient_file_base} ({patient_source})\n' \
                   f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'
        else:
            title = f'{view_plane.upper()} - {exp_name}\n' \
                   f'Patch ID: {patch_id} | From: {patient_file_base} ({patient_source})\n' \
                   f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'

        fig.suptitle(title, fontsize=16, fontweight='bold')

        for local_idx, (slice_img, slice_idx) in enumerate(current_slices):
            im = axes[local_idx].imshow(slice_img, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
            axes[local_idx].set_title(f'Slice {slice_idx + 1}/{total_slices}', fontweight='bold', fontsize=10)
            axes[local_idx].axis('off')
            plt.colorbar(im, ax=axes[local_idx], fraction=0.046)

        for local_idx in range(len(current_slices), 16):
            axes[local_idx].axis('off')

        plt.tight_layout()

        if num_images > 1:
            filename = f"patch_{patch_id}_{patient_file_base}_{patient_source}_{view_plane}_part{img_idx + 1}.png"
        else:
            filename = f"patch_{patch_id}_{patient_file_base}_{patient_source}_{view_plane}.png"

        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


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
    logger.info("VISUALIZE NORMAL PATCHES - MULTIPLE VIEWS")
    logger.info("=" * 80)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        CustomToTensor(),
        transforms.Lambda(lambda x: (x + 300) / 700),
    ])

    # Get all test files
    normal_files = get_test_files(NORMAL_TEST_DIR)
    diseased_files = get_test_files(DISEASED_TEST_DIR)

    logger.info(f"Found {len(normal_files)} Normal files and {len(diseased_files)} Diseased files")

    # Load EXACTLY as evaluate_model_v2.py does (lines 572-588)
    logger.info("=" * 80)
    logger.info("Loading patches exactly as evaluate_model_v2.py...")

    normal_dataset = VideoDataset(NORMAL_TEST_DIR, transform=transform, pkl_files=normal_files)
    diseased_dataset = VideoDataset(DISEASED_TEST_DIR, transform=transform, pkl_files=diseased_files)

    normal_loader = DataLoader(normal_dataset, BATCH_SIZE, shuffle=False, num_workers=0)
    diseased_loader = DataLoader(diseased_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    # Process Normal patches
    logger.info("Processing Normal patches...")
    normal_data = []
    for item_idx, data in normal_loader:
        normal_data.append(data.cpu().numpy())
    normal_all = np.concatenate(normal_data, axis=0) if normal_data else np.array([])

    # Process Diseased patches
    logger.info("Processing Diseased patches...")
    diseased_data = []
    for item_idx, data in diseased_loader:
        diseased_data.append(data.cpu().numpy())
    diseased_all = np.concatenate(diseased_data, axis=0) if diseased_data else np.array([])

    # Combine EXACTLY as evaluate_model_v2.py (line 584-588)
    # all_scores = np.concatenate([normal_scores, diseased_scores])
    # all_patient_sources = ['Normal'] * len(normal_scores) + ['Diseased'] * len(diseased_scores)
    all_data = np.concatenate([normal_all, diseased_all], axis=0)
    all_patient_sources_full = ['Normal'] * len(normal_all) + ['Diseased'] * len(diseased_all)

    logger.info(f"Total patches: {len(all_data)}")
    logger.info(f"  Normal: 0-{len(normal_all)-1}")
    logger.info(f"  Diseased: {len(normal_all)}-{len(all_data)-1}")

    # Determine which patches to visualize
    if not SELECTED_PATCH_IDS:
        # If empty, use all Normal patch IDs
        selected_ids_set = set(range(len(normal_all)))
        logger.info(f"SELECTED_PATCH_IDS is empty, visualizing ALL Normal patches: {len(selected_ids_set)} patches")
    else:
        selected_ids_set = set(SELECTED_PATCH_IDS)
        logger.info(f"Selected Patch IDs: {sorted(SELECTED_PATCH_IDS)}")

    logger.info(f"View planes: {VIEW_PLANES}")
    logger.info(f"Representative slices per view: {NUM_REPRESENTATIVE_SLICES}")
    logger.info(f"Experiments: {[e['name'] for e in EXPERIMENTS]}")

    # Extract selected patches
    all_selected_patches = []
    for selected_id in selected_ids_set:
        if selected_id < len(all_data):
            all_selected_patches.append({
                'data': all_data[selected_id],
                'patch_id': selected_id,
                'patient_source': all_patient_sources_full[selected_id],
                'patient_file': f'{all_patient_sources_full[selected_id]}_{selected_id}'
            })
        else:
            logger.warning(f"Patch ID {selected_id} out of range (max: {len(all_data)-1})")

    normal_patches = [p for p in all_selected_patches if p['patient_source'] == 'Normal']
    diseased_patches = [p for p in all_selected_patches if p['patient_source'] == 'Diseased']
    logger.info(f"Found {len(all_selected_patches)} patches matching selected IDs")
    logger.info(f"  From Normal: {len(normal_patches)}")
    logger.info(f"  From Diseased: {len(diseased_patches)}")

    if len(all_selected_patches) == 0:
        logger.warning("No patches found matching the selected IDs!")
        return

    # Visualize for each experiment and view plane (parallel processing)
    logger.info("=" * 80)
    logger.info(f"Starting parallel visualization with {NUM_WORKERS} workers...")

    # Prepare all tasks for parallel processing
    all_tasks = []
    for exp_config in EXPERIMENTS:
        for view_plane in VIEW_PLANES:
            exp_name = exp_config['name']
            exp_view_dir = output_dir / exp_name / view_plane
            exp_view_dir.mkdir(parents=True, exist_ok=True)

            for patch_data in all_selected_patches:
                all_tasks.append((
                    patch_data,
                    exp_config,
                    view_plane,
                    exp_view_dir,
                    MAX_DISPLAY_SIZE,
                    NUM_REPRESENTATIVE_SLICES,
                    RESOLUTION_AXIAL
                ))

    logger.info(f"Total tasks to process: {len(all_tasks)}")
    start_time = time.time()

    # Execute in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(visualize_patch_worker, all_tasks)

    elapsed = time.time() - start_time
    logger.info(f"Visualization completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    logger.info(f"Average time per task: {elapsed/len(all_tasks):.2f}s")

    # Log summary by experiment and view
    for exp_config in EXPERIMENTS:
        for view_plane in VIEW_PLANES:
            exp_name = exp_config['name']
            exp_view_dir = output_dir / exp_name / view_plane
            logger.info(f"Saved {len(all_selected_patches)} patches to {exp_view_dir}")

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NORMAL PATCHES VISUALIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        if SELECTED_PATCH_IDS:
            f.write(f"SELECTED PATCH IDs: {sorted(SELECTED_PATCH_IDS)}\n\n")
        else:
            f.write(f"MODE: Visualizing ALL Normal patches\n\n")

        f.write(f"FOUND PATCHES: {len(all_selected_patches)} total\n")
        f.write(f"  From Normal patients: {len(normal_patches)}\n")
        f.write(f"  From Diseased patients: {len(diseased_patches)}\n\n")

        f.write("PATCH DETAILS:\n")
        for patch_data in all_selected_patches:
            patient_file = patch_data['patient_file'].replace('.pkl', '').replace('_patch', '')
            f.write(f"  Patch ID {patch_data['patch_id']}: "
                   f"from {patient_file} ({patch_data['patient_source']})\n")
        f.write("\n")

        f.write("VIEW PLANES:\n")
        for view in VIEW_PLANES:
            if view == 'axial':
                f.write(f"  - {view} (4x4 grid, all slices)\n")
            else:
                f.write(f"  - {view} ({NUM_REPRESENTATIVE_SLICES} representative slices, aspect ratio preserved)\n")
        f.write("\n")

        f.write("EXPERIMENTS:\n")
        for exp in EXPERIMENTS:
            f.write(f"  {exp['name']}: W={exp['window']}, L={exp['level']}, Gamma={exp['gamma']} - {exp['description']}\n")
        f.write("\n" + "=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("VISUALIZATION COMPLETED!")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Total patches visualized: {len(all_selected_patches)}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
