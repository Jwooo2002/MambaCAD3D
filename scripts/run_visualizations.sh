#!/bin/bash

# Visualize selected patches with different configurations
# Usage: bash run_visualizations.sh

echo "=========================================="
echo "Running visualizations with multiple configs"
echo "=========================================="

# Configuration arrays
DATASETS=("0.05_8_8_CTCA_64" "0.05_16_16_CTCA_64")
RESOLUTIONS=(448 640 800)

# Selected patch IDs (same for all runs)
PATCH_IDS="119, 122, 123, 164, 165, 166, 167, 172, 175, 180, 181, 215, 220, 221, 232, 249, 333, 334, 336, 347, 348, 351, 373, 376, 377, 392, 400, 423, 424, 425, 430, 431, 432, 441, 442, 800"

# Loop through datasets and resolutions
for dataset in "${DATASETS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Dataset: $dataset"
        echo "Resolution: $resolution"
        echo "=========================================="

        # Extract dataset name for output directory
        dataset_name=$(echo $dataset | cut -d'_' -f2-3)  # e.g., "8_8" or "16_16"

        # Set paths
        NORMAL_DIR="./main_files/dataset/v13/${dataset}/Normal_fp1"
        DISEASED_DIR="./main_files/dataset/v13/${dataset}/Diseased"
        OUTPUT_DIR="./selected_patches_visualization_${dataset_name}_res${resolution}"

        echo "Normal: $NORMAL_DIR"
        echo "Diseased: $DISEASED_DIR"
        echo "Output: $OUTPUT_DIR"

        # Create temporary Python script with current config
        cat > temp_visualize.py << EOF
"""Temporary visualization script - auto-generated"""

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

project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'main_files'))

from main_files.data.VideoDataset import VideoDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CONFIGURATION
NORMAL_TEST_DIR = "${NORMAL_DIR}"
DISEASED_TEST_DIR = "${DISEASED_DIR}"
SELECTED_PATCH_IDS = [${PATCH_IDS}]
BATCH_SIZE = 2
OUTPUT_DIR = "${OUTPUT_DIR}"
VIEW_PLANES = ['sagittal', 'coronal']
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
MAX_DISPLAY_SIZE = ${resolution}
NUM_REPRESENTATIVE_SLICES = 5

def apply_window_level(hu_data: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    min_hu = window_level - window_width / 2.0
    max_hu = window_level + window_width / 2.0
    windowed = np.clip(hu_data, min_hu, max_hu)
    if window_width != 0:
        normalized = ((windowed - min_hu) / window_width) * 255.0
    else:
        normalized = windowed
    return normalized.astype(np.uint8)

def resize_keep_aspect(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape
    if h == w:
        return cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def get_test_files(data_dir: str) -> List[str]:
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    return files

def get_representative_indices(total_slices: int, num_slices: int = 5) -> List[int]:
    if total_slices <= num_slices:
        return list(range(total_slices))
    indices = []
    for i in range(num_slices):
        idx = int((i + 0.5) * total_slices / num_slices)
        indices.append(idx)
    return indices

def visualize_patch(patch_data, exp_config, view_plane, output_dir, max_size, num_rep_slices):
    data = patch_data['data']
    patch_id = patch_data['patch_id']
    patient_source = patch_data['patient_source']
    patient_file = patch_data['patient_file']
    window_width = exp_config['window']
    window_level = exp_config['level']
    exp_name = exp_config['name']

    if data.shape[0] == 1:
        data = data[0]
    D, H, W = data.shape
    slices_data = []

    if view_plane == 'sagittal':
        total_slices = W
        representative_indices = get_representative_indices(W, num_rep_slices)
        for w_idx in representative_indices:
            slice_2d = data[:, :, w_idx]
            slice_hu = slice_2d * 700 - 300
            slice_hu = resize_keep_aspect(slice_hu, max_size)
            slice_wl = apply_window_level(slice_hu, window_width, window_level)
            slices_data.append((slice_wl, w_idx))
    elif view_plane == 'coronal':
        total_slices = H
        representative_indices = get_representative_indices(H, num_rep_slices)
        for h_idx in representative_indices:
            slice_2d = data[:, h_idx, :]
            slice_hu = slice_2d * 700 - 300
            slice_hu = resize_keep_aspect(slice_hu, max_size)
            slice_wl = apply_window_level(slice_hu, window_width, window_level)
            slices_data.append((slice_wl, h_idx))

    _save_grid(slices_data, output_dir, patch_id, patient_source, patient_file,
               total_slices, window_width, window_level, exp_name, view_plane)

def _save_grid(slices, output_dir, patch_id, patient_source, patient_file,
               total_slices, window_width, window_level, exp_name, view_plane):
    rows, cols = 1, len(slices)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if len(slices) == 1:
        axes = [axes]
    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2
    patient_file_base = patient_file.replace('.pkl', '').replace('_patch', '')
    title = f'{view_plane.upper()} - {exp_name}\\n' \
           f'Patch ID: {patch_id} | From: {patient_file_base} ({patient_source})\\n' \
           f'W/L: {window_width}/{window_level} HU | Range: [{min_hu:.0f}, {max_hu:.0f}] HU'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for local_idx, (slice_img, slice_idx) in enumerate(slices):
        im = axes[local_idx].imshow(slice_img, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
        axes[local_idx].set_title(f'Slice {slice_idx + 1}/{total_slices}', fontweight='bold', fontsize=12)
        axes[local_idx].axis('off')
        plt.colorbar(im, ax=axes[local_idx], fraction=0.046)
    plt.tight_layout()
    filename = f"patch_{patch_id}_{patient_file_base}_{patient_source}_{view_plane}.png"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

class CustomToTensor:
    def __call__(self, pic):
        pic = np.asarray(pic, dtype=np.float32)
        if pic.ndim == 2:
            pic = np.expand_dims(pic, axis=0)
        elif pic.ndim == 3:
            pic = pic.transpose((2, 0, 1))
        return torch.tensor(pic, dtype=torch.float32)

def main():
    logger.info("=" * 80)
    logger.info("VISUALIZE SELECTED PATCHES - MULTIPLE VIEWS")
    logger.info(f"Dataset: ${dataset}")
    logger.info(f"Resolution: ${resolution}")
    logger.info("=" * 80)

    if not SELECTED_PATCH_IDS:
        logger.error("SELECTED_PATCH_IDS is empty!")
        return

    selected_ids_set = set(SELECTED_PATCH_IDS)
    logger.info(f"Selected Patch IDs: {sorted(SELECTED_PATCH_IDS)}")
    logger.info(f"View planes: {VIEW_PLANES}")
    logger.info(f"Representative slices per view: {NUM_REPRESENTATIVE_SLICES}")
    logger.info(f"Experiments: {[e['name'] for e in EXPERIMENTS]}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        CustomToTensor(),
        transforms.Lambda(lambda x: (x + 300) / 700),
    ])

    normal_files = get_test_files(NORMAL_TEST_DIR)
    diseased_files = get_test_files(DISEASED_TEST_DIR)
    logger.info(f"Found {len(normal_files)} Normal files and {len(diseased_files)} Diseased files")

    logger.info("=" * 80)
    logger.info("Loading patches exactly as evaluate_model_v2.py...")

    normal_dataset = VideoDataset(NORMAL_TEST_DIR, transform=transform, pkl_files=normal_files)
    diseased_dataset = VideoDataset(DISEASED_TEST_DIR, transform=transform, pkl_files=diseased_files)

    normal_loader = DataLoader(normal_dataset, BATCH_SIZE, shuffle=False, num_workers=0)
    diseased_loader = DataLoader(diseased_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info("Processing Normal patches...")
    normal_data = []
    for item_idx, data in normal_loader:
        normal_data.append(data.cpu().numpy())
    normal_all = np.concatenate(normal_data, axis=0) if normal_data else np.array([])

    logger.info("Processing Diseased patches...")
    diseased_data = []
    for item_idx, data in diseased_loader:
        diseased_data.append(data.cpu().numpy())
    diseased_all = np.concatenate(diseased_data, axis=0) if diseased_data else np.array([])

    all_data = np.concatenate([normal_all, diseased_all], axis=0)
    all_patient_sources_full = ['Normal'] * len(normal_all) + ['Diseased'] * len(diseased_all)

    logger.info(f"Total patches: {len(all_data)}")
    logger.info(f"  Normal: 0-{len(normal_all)-1}")
    logger.info(f"  Diseased: {len(normal_all)}-{len(all_data)-1}")

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

    for exp_config in EXPERIMENTS:
        for view_plane in VIEW_PLANES:
            logger.info("=" * 80)
            exp_name = exp_config['name']
            description = exp_config['description']
            logger.info(f"Experiment: {exp_name} - {view_plane.upper()}")
            logger.info(f"Description: {description}")

            exp_view_dir = output_dir / exp_name / view_plane
            exp_view_dir.mkdir(parents=True, exist_ok=True)

            for patch_data in all_selected_patches:
                visualize_patch(patch_data, exp_config, view_plane,
                              exp_view_dir, MAX_DISPLAY_SIZE, NUM_REPRESENTATIVE_SLICES)

            logger.info(f"  Saved {len(all_selected_patches)} patches to {exp_view_dir}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\\n")
        f.write("SELECTED PATCHES VISUALIZATION SUMMARY\\n")
        f.write("=" * 80 + "\\n\\n")
        f.write(f"Dataset: ${dataset}\\n")
        f.write(f"Resolution: ${resolution}\\n\\n")
        f.write(f"SELECTED PATCH IDs: {sorted(SELECTED_PATCH_IDS)}\\n\\n")
        f.write(f"FOUND PATCHES: {len(all_selected_patches)} total\\n")
        f.write(f"  From Normal patients: {len(normal_patches)}\\n")
        f.write(f"  From Diseased patients: {len(diseased_patches)}\\n\\n")
        f.write("PATCH DETAILS:\\n")
        for patch_data in all_selected_patches:
            patient_file = patch_data['patient_file'].replace('.pkl', '').replace('_patch', '')
            f.write(f"  Patch ID {patch_data['patch_id']}: from {patient_file} ({patch_data['patient_source']})\\n")
        f.write("\\n")
        f.write("VIEW PLANES:\\n")
        for view in VIEW_PLANES:
            f.write(f"  - {view} ({NUM_REPRESENTATIVE_SLICES} representative slices, aspect ratio preserved)\\n")
        f.write("\\n")
        f.write("EXPERIMENTS:\\n")
        for exp in EXPERIMENTS:
            f.write(f"  {exp['name']}: W={exp['window']}, L={exp['level']} - {exp['description']}\\n")
        f.write("\\n" + "=" * 80 + "\\n")

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
EOF

        # Run the temporary script
        python temp_visualize.py

        if [ $? -eq 0 ]; then
            echo "SUCCESS: ${dataset_name} with resolution ${resolution}"
        else
            echo "FAILED: ${dataset_name} with resolution ${resolution}"
        fi

        # Clean up
        rm temp_visualize.py

    done
done

echo ""
echo "=========================================="
echo "All visualizations completed!"
echo "=========================================="
echo "Output directories:"
for dataset in "${DATASETS[@]}"; do
    dataset_name=$(echo $dataset | cut -d'_' -f2-3)
    for resolution in "${RESOLUTIONS[@]}"; do
        echo "  - selected_patches_visualization_${dataset_name}_res${resolution}"
    done
done
