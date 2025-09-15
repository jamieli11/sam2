import argparse
import cv2
import glob
import os
import shutil
import torch
import numpy as np

from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2DataCleaner:
    def __init__(self, sam2_checkpoint, model_cfg, device=None):
        """
        Initialize SAM2 Data Cleaner

        Args:
            sam2_checkpoint: Path to SAM2 checkpoint
            model_cfg: Path to model config
            device: Device to use (cuda/mps/cpu), auto-detected if None
        """
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = device
        print(f"Using device: {device}")

        # Setup device-specific optimizations
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Load SAM2 model
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        print("SAM2 model loaded successfully!")

    def load_bbox_annotation(self, bbox_path):
        """Load bbox annotation from txt file (xywh format)"""
        if not os.path.exists(bbox_path):
            return None

        try:
            with open(bbox_path, 'r') as f:
                bbox_line = f.readline().strip()
                bbox_xywh = np.array(bbox_line.split(), dtype=int)
                return bbox_xywh  # Return in xywh format for saving
        except:
            print(f"Failed to load bbox from {bbox_path}")
            return None

    def xywh_to_xyxy(self, bbox_xywh):
        """Convert bbox from xywh to xyxy format"""
        return np.array([
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3]
        ])

    def xyxy_to_xywh(self, bbox_xyxy):
        """Convert bbox from xyxy to xywh format"""
        return np.array([
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1]
        ])

    def calculate_bbox_area(self, bbox_xywh):
        """Calculate bbox area from xywh format"""
        return bbox_xywh[2] * bbox_xywh[3]


    def get_mask_bbox(self, mask, min_area=100):
        """Generate bbox from mask using contours, filter out small connected components"""
        # Convert to uint8 for opencv operations
        mask_uint8 = mask.astype(np.uint8)

        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        if num_labels <= 1:  # Only background
            return None

        # Filter components by area (excluding background label 0)
        valid_labels = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                valid_labels.append(i)

        if not valid_labels:
            return None

        # Find the largest valid component
        largest_component_label = max(valid_labels,
                                      key=lambda x: stats[x, cv2.CC_STAT_AREA])

        # Create mask with only the largest valid component
        filtered_mask = (labels == largest_component_label).astype(np.uint8)

        # Get bounding box from filtered mask
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return np.array([x, y, x + w, y + h])  # Return in xyxy format

    def should_reject_frame(self, original_bbox_xywh, updated_bbox_xywh, area_threshold=0.5):
        """
        Check if frame should be rejected based on bbox area change

        Args:
            original_bbox_xywh: Original bbox in xywh format
            updated_bbox_xywh: Updated bbox in xywh format
            area_threshold: Threshold for area change (0.5 means reject if area < 0.5x or > 2x)

        Returns:
            bool: True if frame should be rejected
        """
        original_area = self.calculate_bbox_area(original_bbox_xywh)
        updated_area = self.calculate_bbox_area(updated_bbox_xywh)

        if original_area == 0:
            return True

        area_ratio = updated_area / original_area

        # Reject if area is less than threshold or greater than 1/threshold
        return area_ratio < area_threshold or area_ratio > (1 / area_threshold)

    def process_single_image(self, image_path, bbox_path, mask_path, output_mask_path, output_bbox_path):
        """
        Process a single image with SAM2 and update mask/bbox

        Returns:
            tuple: (success, rejection_reason, original_area, updated_area)
        """
        try:
            # Load image
            image = Image.open(image_path)
            image_np = np.array(image.convert("RGB"))

            # Load original bbox
            original_bbox_xywh = self.load_bbox_annotation(bbox_path)
            if original_bbox_xywh is None:
                return False, "No original bbox", 0, 0

            # Convert to xyxy for SAM2
            original_bbox_xyxy = self.xywh_to_xyxy(original_bbox_xywh)

            # Set image for predictor
            self.predictor.set_image(image_np)

            # Run SAM2 prediction
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=original_bbox_xyxy[None, :],
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                return False, "No SAM2 prediction", 0, 0

            # Get updated mask
            updated_mask = masks[0]

            # Generate updated bbox from mask
            updated_bbox_xyxy = self.get_mask_bbox(updated_mask)
            if updated_bbox_xyxy is None:
                return False, "No bbox from mask", 0, 0

            # Convert to xywh for comparison and saving
            updated_bbox_xywh = self.xyxy_to_xywh(updated_bbox_xyxy)

            # Check if frame should be rejected
            original_area = self.calculate_bbox_area(original_bbox_xywh)
            updated_area = self.calculate_bbox_area(updated_bbox_xywh)

            if self.should_reject_frame(original_bbox_xywh, updated_bbox_xywh):
                area_ratio = updated_area / original_area if original_area > 0 else 0
                return False, f"Area change too large (ratio: {area_ratio:.2f})", original_area, updated_area

            # Save updated mask
            updated_mask_uint8 = (updated_mask * 255).astype(np.uint8)
            cv2.imwrite(output_mask_path, updated_mask_uint8)

            # Save updated bbox
            with open(output_bbox_path, 'w') as f:
                f.write(
                    f"{updated_bbox_xywh[0]} {updated_bbox_xywh[1]} {updated_bbox_xywh[2]} {updated_bbox_xywh[3]}\n")

            return True, "Success", original_area, updated_area

        except Exception as e:
            return False, f"Error: {str(e)}", 0, 0

    def clean_dataset(self, input_root, output_root, area_threshold=0.5, backup_rejected=True):
        """
        Clean the entire dataset using SAM2

        Args:
            input_root: Input directory containing Images, Masks, Annotations
            output_root: Output directory for cleaned data
            area_threshold: Threshold for bbox area change
            backup_rejected: Whether to backup rejected frames
        """
        # Create output directories
        output_images_dir = os.path.join(output_root, 'Images')
        output_masks_dir = os.path.join(output_root, 'Masks')
        output_annotations_dir = os.path.join(output_root, 'Annotations')

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_masks_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)

        # Create rejected frames backup directory if needed
        if backup_rejected:
            rejected_dir = os.path.join(output_root, 'rejected_frames')
            rejected_images_dir = os.path.join(rejected_dir, 'Images')
            rejected_masks_dir = os.path.join(rejected_dir, 'Masks')
            rejected_annotations_dir = os.path.join(rejected_dir, 'Annotations')

            os.makedirs(rejected_images_dir, exist_ok=True)
            os.makedirs(rejected_masks_dir, exist_ok=True)
            os.makedirs(rejected_annotations_dir, exist_ok=True)

        # Get all image files
        input_images_dir = os.path.join(input_root, 'Images')
        input_masks_dir = os.path.join(input_root, 'Masks')
        input_annotations_dir = os.path.join(input_root, 'Annotations')

        image_files = sorted(glob.glob(os.path.join(input_images_dir, '*.jpg')) +
                             glob.glob(os.path.join(input_images_dir, '*.png')))

        print(f"Found {len(image_files)} images to process")

        # Statistics
        successful_count = 0
        rejected_count = 0
        error_count = 0
        rejected_frames = []

        # Process log
        log_path = os.path.join(output_root, 'cleaning_log.txt')
        rejected_log_path = os.path.join(output_root, 'rejected_frames.txt')

        with open(log_path, 'w') as log_file:
            log_file.write(f"SAM2 Data Cleaning Log\n")
            log_file.write(f"Started at: {datetime.now()}\n")
            log_file.write(f"Area threshold: {area_threshold}\n")
            log_file.write(f"Total images: {len(image_files)}\n\n")

            for image_path in tqdm(image_files, desc="Processing images"):
                basename = os.path.splitext(os.path.basename(image_path))[0]

                # Input paths
                mask_path = os.path.join(input_masks_dir, basename + '.jpg')
                bbox_path = os.path.join(input_annotations_dir, basename + '.txt')

                # Output paths
                output_image_path = os.path.join(output_images_dir, os.path.basename(image_path))
                output_mask_path = os.path.join(output_masks_dir, basename + '.jpg')
                output_bbox_path = os.path.join(output_annotations_dir, basename + '.txt')

                # Check if input files exist
                if not os.path.exists(mask_path) or not os.path.exists(bbox_path):
                    log_file.write(f"SKIP: {basename} - Missing mask or bbox file\n")
                    error_count += 1
                    continue

                # Process the image
                success, reason, original_area, updated_area = self.process_single_image(
                    image_path, bbox_path, mask_path, output_mask_path, output_bbox_path
                )

                if success:
                    # Copy original image to output
                    shutil.copy2(image_path, output_image_path)
                    successful_count += 1
                    log_file.write(
                        f"SUCCESS: {basename} - Original area: {original_area}, Updated area: {updated_area}\n")
                else:
                    rejected_count += 1
                    rejected_frames.append((basename, reason, original_area, updated_area))
                    log_file.write(f"REJECTED: {basename} - {reason}\n")

                    # Backup rejected frame if requested
                    if backup_rejected:
                        try:
                            shutil.copy2(image_path, os.path.join(rejected_images_dir, os.path.basename(image_path)))
                            if os.path.exists(mask_path):
                                shutil.copy2(mask_path, os.path.join(rejected_masks_dir, basename + '.jpg'))
                            if os.path.exists(bbox_path):
                                shutil.copy2(bbox_path, os.path.join(rejected_annotations_dir, basename + '.txt'))
                        except:
                            pass

            # Write summary
            log_file.write(f"\n=== SUMMARY ===\n")
            log_file.write(f"Successful: {successful_count}\n")
            log_file.write(f"Rejected: {rejected_count}\n")
            log_file.write(f"Errors: {error_count}\n")
            log_file.write(f"Total processed: {successful_count + rejected_count + error_count}\n")
            log_file.write(f"Success rate: {successful_count / len(image_files) * 100:.2f}%\n")
            log_file.write(f"Completed at: {datetime.now()}\n")

        # Write rejected frames list
        with open(rejected_log_path, 'w') as rejected_file:
            rejected_file.write(f"Rejected Frames Log\n")
            rejected_file.write(f"Total rejected: {rejected_count}\n")
            rejected_file.write(f"Area threshold: {area_threshold}\n\n")

            for frame_name, reason, orig_area, upd_area in rejected_frames:
                rejected_file.write(f"{frame_name}: {reason}\n")

        print(f"\n=== Cleaning Complete ===")
        print(f"Successful: {successful_count}")
        print(f"Rejected: {rejected_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {successful_count / len(image_files) * 100:.2f}%")
        print(f"Results saved to: {output_root}")
        print(f"Logs saved to: {log_path}")
        print(f"Rejected frames list: {rejected_log_path}")


def main():
    parser = argparse.ArgumentParser(description='SAM2 Data Cleaner for dataset processing')

    # Model configuration arguments
    parser.add_argument('--sam2_checkpoint',
                        default='checkpoints/sam2.1_hiera_large.pt',
                        help='Path to SAM2 checkpoint file')
    parser.add_argument('--model_cfg',
                        default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='Path to model configuration file')

    # Data path arguments
    parser.add_argument('--input_root',
                        required=True,
                        help='Input root directory (contains Images, Masks, Annotations directories)')
    parser.add_argument('--output_root',
                        required=True,
                        help='Output root directory for cleaned dataset')

    # Processing parameters
    parser.add_argument('--area_threshold',
                        type=float,
                        default=0.6,
                        help='Area threshold for rejection (reject if area < threshold x or > 1/threshold x original)')
    parser.add_argument('--backup_rejected',
                        action='store_true',
                        default=True,
                        help='Backup rejected frames')
    parser.add_argument('--no_backup_rejected',
                        action='store_false',
                        dest='backup_rejected',
                        help='Do not backup rejected frames')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)

    # Initialize cleaner
    print("Initializing SAM2 Data Cleaner...")
    cleaner = SAM2DataCleaner(args.sam2_checkpoint, args.model_cfg)

    # Start cleaning
    print("Starting dataset cleaning...")
    print(f"Input: {args.input_root}")
    print(f"Output: {args.output_root}")
    print(f"Area threshold: {args.area_threshold}")
    print(f"Backup rejected: {args.backup_rejected}")

    cleaner.clean_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        area_threshold=args.area_threshold,
        backup_rejected=args.backup_rejected
    )

    print("Dataset cleaning completed!")


if __name__ == "__main__":
    main()
