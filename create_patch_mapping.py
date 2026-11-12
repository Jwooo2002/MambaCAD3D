"""
Create Patch Index Mapping CSV
================================

This script creates a CSV file that maps global patch indices to their source files.
Useful for understanding which patient/file a patch index (e.g., 801) corresponds to.

Output CSV format:
    global_index,category,patient_file,patient_id,patch_in_file

Example:
    0,Normal,N_1_patch.pkl,N_1,0
    1,Normal,N_1_patch.pkl,N_1,1
    ...
    801,Diseased,D_10_patch.pkl,D_10,12

Usage:
    python create_patch_mapping.py
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from typing import List

# ============================================================================
# CONFIGURATION
# ============================================================================

NORMAL_TEST_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Normal_fp1"
DISEASED_TEST_DIR = "./main_files/dataset/v11/0.05_8_32_CTCA_64/Diseased"

OUTPUT_CSV = "./patch_index_mapping.csv"

# You can specify specific files or set to None to use all .pkl files
NORMAL_TEST_FILES = None  # None = all files
DISEASED_TEST_FILES = None  # None = all files

# ============================================================================


def get_pkl_files(directory: str, file_list: List[str] = None) -> List[str]:
    """Get list of pkl files from directory"""
    if file_list is not None:
        return file_list

    files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl')])
    return files


def create_patch_mapping(
    normal_dir: str,
    diseased_dir: str,
    normal_files: List[str],
    diseased_files: List[str],
    output_csv: str
):
    """
    Create CSV mapping of global patch indices to patient files

    Args:
        normal_dir: Directory containing normal patient pkl files
        diseased_dir: Directory containing diseased patient pkl files
        normal_files: List of normal pkl filenames
        diseased_files: List of diseased pkl filenames
        output_csv: Path to output CSV file

    Returns:
        DataFrame with mapping information
    """
    print("=" * 80)
    print("CREATING PATCH INDEX MAPPING")
    print("=" * 80)
    print(f"Normal directory: {normal_dir}")
    print(f"Diseased directory: {diseased_dir}")
    print(f"Normal files: {len(normal_files)}")
    print(f"Diseased files: {len(diseased_files)}")
    print()

    mapping_data = []
    global_idx = 0

    # Process normal files
    print("Processing Normal files...")
    for pkl_file in normal_files:
        patient_id = pkl_file.replace('.pkl', '').replace('_patch', '')
        pkl_path = os.path.join(normal_dir, pkl_file)

        with open(pkl_path, 'rb') as f:
            clips = pickle.load(f)
            num_patches = len(clips)

        print(f"  {pkl_file}: {num_patches} patches (global index {global_idx} to {global_idx + num_patches - 1})")

        for patch_idx in range(num_patches):
            mapping_data.append({
                'global_index': global_idx,
                'category': 'Normal',
                'patient_file': pkl_file,
                'patient_id': patient_id,
                'patch_in_file': patch_idx,
                'patch_label': f"{patient_id}_{patch_idx}"
            })
            global_idx += 1

    print()

    # Process diseased files
    print("Processing Diseased files...")
    for pkl_file in diseased_files:
        patient_id = pkl_file.replace('.pkl', '').replace('_patch', '')
        pkl_path = os.path.join(diseased_dir, pkl_file)

        with open(pkl_path, 'rb') as f:
            clips = pickle.load(f)
            num_patches = len(clips)

        print(f"  {pkl_file}: {num_patches} patches (global index {global_idx} to {global_idx + num_patches - 1})")

        for patch_idx in range(num_patches):
            mapping_data.append({
                'global_index': global_idx,
                'category': 'Diseased',
                'patient_file': pkl_file,
                'patient_id': patient_id,
                'patch_in_file': patch_idx,
                'patch_label': f"{patient_id}_{patch_idx}"
            })
            global_idx += 1

    print()
    print("=" * 80)

    # Save to CSV
    df = pd.DataFrame(mapping_data)
    df.to_csv(output_csv, index=False)

    print(f"✓ Patch mapping saved: {output_csv}")
    print(f"✓ Total patches mapped: {len(df)}")
    print()

    # Show some examples
    print("Sample mappings:")
    print(df.head(10).to_string(index=False))
    print("...")
    print(df.tail(10).to_string(index=False))
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    normal_count = len(df[df['category'] == 'Normal'])
    diseased_count = len(df[df['category'] == 'Diseased'])
    print(f"Total patches: {len(df)}")
    print(f"  Normal: {normal_count}")
    print(f"  Diseased: {diseased_count}")
    print()
    print(f"To find a specific patch index (e.g., 801):")
    print(f"  df = pd.read_csv('{output_csv}')")
    print(f"  df[df['global_index'] == 801]")
    print("=" * 80)

    return df


def main():
    # Get file lists
    normal_files = get_pkl_files(NORMAL_TEST_DIR, NORMAL_TEST_FILES)
    diseased_files = get_pkl_files(DISEASED_TEST_DIR, DISEASED_TEST_FILES)

    # Create mapping
    df = create_patch_mapping(
        normal_dir=NORMAL_TEST_DIR,
        diseased_dir=DISEASED_TEST_DIR,
        normal_files=normal_files,
        diseased_files=diseased_files,
        output_csv=OUTPUT_CSV
    )


if __name__ == '__main__':
    main()
