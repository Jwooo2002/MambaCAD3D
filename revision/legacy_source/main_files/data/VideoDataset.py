# data.py (최종 수정본)

from __future__ import print_function, absolute_import
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pickle
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, data_root, transform=None, pkl_files=None):  # pkl_files 인자 추가
        super().__init__()
        self.data_root = data_root
        self.transform = transform

        # ## 수정된 부분 ##
        # pkl_files 인자가 주어지면 해당 파일 리스트를 사용하고,
        # 주어지지 않으면 data_root의 모든 .pkl 파일을 사용합니다.
        if pkl_files is not None:
            video_list = pkl_files
        else:
            print(f"Loading all files from {data_root}...")
            video_list = sorted([f for f in os.listdir(self.data_root) if f.endswith('.pkl')])

        all_clips = []
        for video_file in video_list:
            v_path = os.path.join(self.data_root, video_file)
            with open(v_path, "rb") as f:
                clips_in_file = pickle.load(f)
                all_clips.extend(clips_in_file)

        self.all_clips = np.array(all_clips, dtype=np.float32)
        self.num_clips = len(self.all_clips)
        if pkl_files is None:
            print(f"Dataset loaded. Total number of clips: {self.num_clips}")

    def __len__(self):
        return self.num_clips

    def __getitem__(self, index):
        video_clip = self.all_clips[index]

        frames_tensor = torch.cat([
            self.transform(
                np.expand_dims(frame, axis=2)
            ).unsqueeze(1)
            for frame in video_clip
        ], dim=1)

        return index, frames_tensor