#!/usr/bin/env python3

import logging
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset
from iopath.common.file_io import g_pathmgr

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class SingleFrameDataset(VisionDataset):
    """Dataset for single-frame segmentation tasks, treating each frame as a 1-frame video"""

    def __init__(
            self,
            transforms,
            training: bool,
            img_folder: str,
            gt_folder: str,
            file_list_txt: str,
            multiplier: int = 1,
            always_target: bool = True,
    ):
        self._transforms = transforms
        self.training = training
        self.img_folder = Path(img_folder)
        self.gt_folder = Path(gt_folder)
        self.always_target = always_target

        # Load sequence list
        self.sequences = []
        with open(file_list_txt, 'r') as f:
            for line in f:
                seq_name = line.strip()
                if seq_name:
                    self.sequences.append(seq_name)

        # Build frame list
        self.samples = []
        for seq_name in self.sequences:
            seq_img_dir = self.img_folder / seq_name
            if seq_img_dir.exists():
                frame_files = sorted(seq_img_dir.glob("*.jpg"))
                for frame_file in frame_files:
                    frame_name = frame_file.stem
                    mask_file = self.gt_folder / seq_name / f"{frame_name}.png"
                    if mask_file.exists():
                        self.samples.append({
                            'sequence': seq_name,
                            'frame_name': frame_name,
                            'image_path': frame_file,
                            'mask_path': mask_file
                        })

        # Apply multiplier
        self.samples = self.samples * multiplier

        # Add repeat_factors attribute for compatibility with SAM2's ConcatDataset
        self.repeat_factors = torch.ones(len(self.samples), dtype=torch.float32)

        self.curr_epoch = 0
        print(f"SingleFrameDataset: {len(self.samples)} samples from {len(self.sequences)} sequences")

    def _get_datapoint(self, idx):
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()

                sample = self.samples[idx]
                datapoint = self.construct_single_frame(sample)
                break

            except Exception as e:
                if self.training:
                    logging.warning(f"Loading failed (id={idx}); Retry {retry} with exception: {e}")
                    idx = random.randrange(0, len(self.samples))
                else:
                    raise e

        # Apply transforms
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct_single_frame(self, sample):
        """Construct a single-frame VideoDatapoint"""

        # Load image
        with g_pathmgr.open(str(sample['image_path']), "rb") as fopen:
            rgb_image = PILImage.open(fopen).convert("RGB")

        w, h = rgb_image.size

        # Load mask
        with g_pathmgr.open(str(sample['mask_path']), "rb") as fopen:
            mask_image = PILImage.open(fopen).convert("L")

        # Convert mask to tensor
        mask_array = np.array(mask_image)
        mask_tensor = torch.from_numpy(mask_array).to(torch.uint8)

        # Ensure binary mask (0 or 255)
        mask_tensor = (mask_tensor > 127).to(torch.uint8) * 255

        # Create frame with single object
        frame = Frame(
            data=rgb_image,
            objects=[
                Object(
                    object_id=1,  # Single object per frame
                    frame_index=0,  # Single frame index
                    segment=mask_tensor,
                )
            ]
        )

        # Create VideoDatapoint (single frame "video")
        # Use numeric video_id for compatibility with collate_fn
        video_id = hash(f"{sample['sequence']}_{sample['frame_name']}") % (2 ** 31)
        return VideoDatapoint(
            frames=[frame],
            video_id=video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch