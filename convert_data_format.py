#!/usr/bin/env python3
"""
Convert data cleaner data to SAM2 training format with timestamp grouping
"""

import os
import cv2
import json
import shutil
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename format: 2024-12-17T18-51-24.428733Z_slot-1_left_101.234"""
    parts = filename.split('_')
    return parts[0].split('.')[0] if len(parts) > 0 else filename


def convert_to_sam2_format(input_root: str, output_root: str) -> Tuple[int, pd.DataFrame]:
    """Convert data to SAM2 format grouped by timestamp"""
    input_root = Path(input_root)
    output_root = Path(output_root)

    jpeg_dir = output_root / "JPEGImages"
    ann_dir = output_root / "Annotations"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images_dir = input_root / "Images"
    masks_dir = input_root / "Masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"Missing directories: {images_dir} or {masks_dir}")
        return 0, pd.DataFrame()

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    # Group files by timestamp
    timestamp_groups = {}
    file_mappings = []

    for img_path in image_files:
        basename = img_path.stem
        mask_path = masks_dir / f"{basename}.jpg"
        if not mask_path.exists():
            mask_path = masks_dir / f"{basename}.png"
        if not mask_path.exists():
            continue

        timestamp = extract_timestamp_from_filename(basename)
        if timestamp not in timestamp_groups:
            timestamp_groups[timestamp] = []

        timestamp_groups[timestamp].append({
            'basename': basename,
            'img_path': img_path,
            'mask_path': mask_path
        })

    metadata = {}
    processed_sequences = 0
    total_samples = 0

    for timestamp, files in tqdm(timestamp_groups.items(), desc="Converting groups"):
        seq_jpeg_dir = jpeg_dir / timestamp
        seq_ann_dir = ann_dir / timestamp
        seq_jpeg_dir.mkdir(exist_ok=True)
        seq_ann_dir.mkdir(exist_ok=True)

        sequence_frames = []

        # Sort files by basename to ensure consistent frame ordering
        files.sort(key=lambda x: x['basename'])

        for frame_idx, file_info in enumerate(files):
            basename = file_info['basename']
            img_path = file_info['img_path']
            mask_path = file_info['mask_path']

            # Use zero-padded frame numbers: 00000, 00001, etc.
            frame_name = f"{frame_idx:05d}"
            dst_img = seq_jpeg_dir / f"{frame_name}.jpg"
            dst_mask = seq_ann_dir / f"{frame_name}.png"

            # Copy/convert image
            if img_path.suffix.lower() == '.jpg':
                shutil.copy2(img_path, dst_img)
            else:
                img = cv2.imread(str(img_path))
                if img is not None:
                    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    continue

            # Process mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(dst_mask), mask_binary)
                sequence_frames.append(frame_name)

                # Record mapping
                file_mappings.append({
                    'original_image': str(img_path.name),
                    'original_mask': str(mask_path.name),
                    'original_basename': basename,
                    'timestamp': timestamp,
                    'frame_idx': frame_idx,
                    'sam2_image': f"{timestamp}/{frame_name}.jpg",
                    'sam2_mask': f"{timestamp}/{frame_name}.png"
                })
                total_samples += 1

        if sequence_frames:
            metadata[timestamp] = {
                "sequence": timestamp,
                "frames": sequence_frames,
                "objects": {
                    "1": {"frames": sequence_frames}
                }
            }
            processed_sequences += 1

    # Save metadata
    with open(output_root / "meta.json", 'w') as f:
        json.dump({"sequences": metadata}, f, indent=2)

    mapping_df = pd.DataFrame(file_mappings)
    print(f"Converted {total_samples} samples into {processed_sequences} timestamp sequences")
    return total_samples, mapping_df


def create_file_lists(data_root: str, train_ratio: float = 0.9, val_ratio: float = 0.1) -> Dict[str, List[str]]:
    """Create train/val/test file lists"""
    data_root = Path(data_root)
    jpeg_dir = data_root / "JPEGImages"

    timestamp_dirs = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    total = len(timestamp_dirs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': timestamp_dirs[:train_end],
        'val': timestamp_dirs[train_end:val_end],
        'test': timestamp_dirs[val_end:]
    }

    for split_name, sequences in splits.items():
        with open(data_root / f"{split_name}_list.txt", 'w') as f:
            for seq in sequences:
                f.write(f"{seq}\n")

    print(f"Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
    return splits


def save_mapping_documentation(mapping_df: pd.DataFrame, splits: Dict[str, List[str]], output_root: str):
    """Save file mapping documentation"""
    output_root = Path(output_root)

    # Add split info
    split_mapping = {}
    for split_name, sequences in splits.items():
        for seq in sequences:
            split_mapping[seq] = split_name

    mapping_df['split'] = mapping_df['timestamp'].map(split_mapping).fillna('unknown')

    # Save mappings
    mapping_df.to_csv(output_root / "file_mapping.csv", index=False)

    for split_name in ['train', 'val', 'test']:
        split_df = mapping_df[mapping_df['split'] == split_name]
        if len(split_df) > 0:
            split_df.to_csv(output_root / f"{split_name}_mapping.csv", index=False)

    # Summary
    with open(output_root / "conversion_summary.txt", 'w') as f:
        f.write("SAM2 Data Conversion Summary\n")
        f.write("===========================\n\n")
        f.write(f"Total samples: {len(mapping_df)}\n")
        f.write(f"Total sequences: {len(mapping_df['timestamp'].unique())}\n\n")

        for split_name in ['train', 'val', 'test']:
            count = len(mapping_df[mapping_df['split'] == split_name])
            f.write(f"{split_name}: {count} samples\n")

        f.write("\nStructure: timestamp-grouped sequences with zero-padded frame indices\n")


def validate_input_data(input_root: str) -> bool:
    """Validate input data structure"""
    input_root = Path(input_root)
    images_dir = input_root / "Images"
    masks_dir = input_root / "Masks"

    if not all([input_root.exists(), images_dir.exists(), masks_dir.exists()]):
        print("Missing required directories")
        return False

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    mask_files = list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.png"))

    image_basenames = {f.stem for f in image_files}
    mask_basenames = {f.stem for f in mask_files}
    matching_samples = image_basenames & mask_basenames

    print(f"Images: {len(image_files)}, Masks: {len(mask_files)}")
    print(f"Complete pairs: {len(matching_samples)}")

    if len(matching_samples) == 0:
        print("No matching image-mask pairs found")
        return False

    # Show timestamp grouping
    timestamp_counts = {}
    for name in list(matching_samples)[:10]:  # Sample first 10
        timestamp = extract_timestamp_from_filename(name)
        timestamp_counts[timestamp] = timestamp_counts.get(timestamp, 0) + 1

    print(f"Sample timestamp groups: {len(timestamp_counts)}")
    for ts, count in list(timestamp_counts.items())[:3]:
        print(f"  {ts}: {count} frames")

    return True


def main():
    parser = argparse.ArgumentParser(description='Convert data to SAM2 format with timestamp grouping')
    parser.add_argument('--input', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output SAM2 directory')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Training ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--validate-only', action='store_true', help='Only validate input')

    args = parser.parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        print("Error: train_ratio + val_ratio must be <= 1.0")
        return

    if not validate_input_data(args.input):
        print("Input validation failed")
        return

    if args.validate_only:
        print("Validation completed")
        return

    num_samples, mapping_df = convert_to_sam2_format(args.input, args.output)

    if num_samples > 0:
        splits = create_file_lists(args.output, args.train_ratio, args.val_ratio)
        save_mapping_documentation(mapping_df, splits, args.output)
        print(f"Conversion completed: {args.output}")
    else:
        print("No data converted")


if __name__ == "__main__":
    main()