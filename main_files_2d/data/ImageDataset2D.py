"""
2D Image Dataset - extracts 2D slices from 3D volumes
"""

from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np


class ImageDataset2D(Dataset):
    """
    2D Image dataset - extracts 2D slices from 3D volumes stored in pkl files
    Compatible with the k-fold framework
    """
    def __init__(self, data_root, transform=None, pkl_files=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform

        # Use specified pkl files or all files in directory
        if pkl_files is not None:
            video_list = pkl_files
        else:
            print(f"Loading all files from {data_root}...")
            video_list = sorted([f for f in os.listdir(self.data_root) if f.endswith('.pkl')])

        all_slices = []
        for video_file in video_list:
            v_path = os.path.join(self.data_root, video_file)
            with open(v_path, "rb") as f:
                clips_in_file = pickle.load(f)  # List of 3D volumes
                for clip in clips_in_file:
                    # clip shape: (Z, H, W)
                    clip = np.array(clip, dtype=np.float32)

                    # Extract all Z slices as 2D images
                    if len(clip.shape) == 3:
                        for z_idx in range(clip.shape[0]):
                            slice_2d = clip[z_idx]  # (H, W)
                            all_slices.append(slice_2d)
                    else:
                        # If already 2D
                        all_slices.append(clip)

        self.all_slices = np.array(all_slices, dtype=np.float32)
        self.num_slices = len(self.all_slices)

        if pkl_files is None:
            print(f"Dataset loaded. Total number of 2D slices: {self.num_slices}")

    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        slice_2d = self.all_slices[index]  # (H, W)

        # Apply transform if provided
        if self.transform is not None:
            # Add channel dimension for transform
            slice_2d = np.expand_dims(slice_2d, axis=2)  # (H, W, 1)
            slice_tensor = self.transform(slice_2d)  # (1, H, W)
        else:
            # Just convert to tensor
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0)  # (1, H, W)

        return index, slice_tensor
