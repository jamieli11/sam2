import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

np.random.seed(3)


class SAM2InteractiveViewer:
    def __init__(self, sam2_checkpoint, model_cfg, device=None):
        """
        Initialize SAM2 Interactive Viewer

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
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

        # Load SAM2 model
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        print("SAM2 model loaded successfully!")

    def load_bbox_annotation(self, bbox_path):
        """Load bbox annotation from txt file"""
        if not os.path.exists(bbox_path):
            return None

        try:
            with open(bbox_path, 'r') as f:
                bbox_line = f.readline().strip()
                bbox_xywh = np.array(bbox_line.split(), dtype=int)
                # Convert from xywh to xyxy format
                bbox_xyxy = np.array([
                    bbox_xywh[0],
                    bbox_xywh[1],
                    bbox_xywh[0] + bbox_xywh[2],
                    bbox_xywh[1] + bbox_xywh[3]
                ])
                return bbox_xyxy
        except:
            print(f"Failed to load bbox from {bbox_path}")
            return None

    def show_mask(self, mask, ax, random_color=False, borders=True):
        """Display mask overlay"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        ax.imshow(mask_image)

    def show_box(self, box, ax):
        """Display bounding box"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=1))

    def show_box_color(self, box, ax, color='green'):
        """Display bounding box with specified color"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=1))


    def predict_and_visualize(self, image_path, bbox_path=None, save_path=None):
        """
        Run SAM2 prediction and create visualization

        Returns:
            visualization_img: Combined visualization image for display
        """
        # Load image
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))

        # Set image for predictor
        self.predictor.set_image(image_np)

        # Load bounding box
        bbox = self.load_bbox_annotation(bbox_path) if bbox_path else None

        masks, scores = None, None
        if bbox is not None:
            # Run prediction with bbox
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,
            )

            # Generate updated bbox from SAM2 mask
            if masks is not None and len(masks) > 0:
                mask = masks[0].astype(np.uint8)
                # Find contours to get bounding box
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    updated_bbox = np.array([x, y, x + w, y + h])


        # Load ground truth mask
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'Masks')
        basename = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(mask_dir, basename + '.jpg')
        gt_mask = None
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'SAM2 Interactive Viewer: {os.path.basename(image_path)}', fontsize=16)

        # Left top: Original image
        axes[0, 0].imshow(image_np)
        # if bbox is not None:
        #     self.show_box(bbox, axes[0, 0])
        #     axes[0, 0].set_title(f'Original + BBox: {bbox[2] - bbox[0]}x{bbox[3] - bbox[1]}')
        # else:
        axes[0, 0].set_title('Original Image (No BBox)')
        axes[0, 0].axis('off')

        # Right top: GT Mask overlay + bbox
        axes[0, 1].imshow(image_np)
        if gt_mask is not None:
            self.show_mask(gt_mask > 127, axes[0, 1], borders=True)
        if bbox is not None:
            self.show_box(bbox, axes[0, 1])
        axes[0, 1].set_title('GT Mask Overlay + BBox')
        axes[0, 1].axis('off')

        # Left bottom: SAM2 prediction overlay
        axes[1, 0].imshow(image_np)
        if masks is not None and len(masks) > 0:
            self.show_mask(masks[0], axes[1, 0], borders=True)
            axes[1, 0].set_title(f'SAM2 Prediction (Score: {scores[0]:.3f})')
        else:
            axes[1, 0].set_title('No SAM2 Prediction')
        axes[1, 0].axis('off')

        # Right bottom: SAM2 overlay + bbox
        axes[1, 1].imshow(image_np)
        if masks is not None and len(masks) > 0:
            self.show_mask(masks[0], axes[1, 1], borders=True)
        # if bbox is not None:
        #     self.show_box(bbox, axes[1, 1])
        #     axes[1, 1].set_title(f'SAM2 Overlay + BBox (Score: {scores[0]:.3f})')
        # else:
        #     axes[1, 1].set_title('SAM2 Overlay + BBox')

        if updated_bbox is not None:
            # Updated bbox in blue (need to add this method)
            self.show_box_color(updated_bbox, axes[1, 1], color='green')
            axes[1, 1].set_title(
                f'SAM2 Overlay + Updated BBox: {updated_bbox[2] - updated_bbox[0]}x{updated_bbox[3] - updated_bbox[1]} (Score: {scores[0]:.3f})')
        else:
            axes[1, 1].set_title('SAM2 Overlay + BBox')

        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Convert to numpy array for OpenCV display
        fig.canvas.draw()

        # Handle different matplotlib versions
        try:
            # New matplotlib versions
            buf = fig.canvas.buffer_rgba()
            vis_img = np.asarray(buf)[:, :, :3]  # Remove alpha channel
        except AttributeError:
            try:
                # Older matplotlib versions
                vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Alternative method
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                vis_img = buf[:, :, :3]  # Remove alpha channel

        plt.close(fig)

        return vis_img

    def interactive_visualization(self, data_root, save_dir=None, start_index=0):
        """
        Interactive visualization mode with keyboard controls

        Controls:
        - 'n' or 'd' or Right Arrow: Next image
        - 'p' or 'a' or Left Arrow: Previous image
        - 's': Save current visualization
        - 'r': Rerun prediction on current image
        - 'q' or ESC: Quit
        - Space: Next image
        """
        # Get all image files
        image_dir = os.path.join(data_root, 'Images')
        bbox_dir = os.path.join(data_root, 'Annotations')

        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) +
                             glob.glob(os.path.join(image_dir, '*.png')))

        if not image_files:
            print("No image files found!")
            return

        print(f"Found {len(image_files)} images")
        print("\nControls:")
        print("- 'n'/'d'/Right Arrow/Space: Next image")
        print("- 'p'/'a'/Left Arrow: Previous image")
        print("- 's': Save current visualization")
        print("- 'r': Rerun prediction on current image")
        print("- 'q'/ESC: Quit")
        print("- Any other key: Show help")

        current_index = max(0, min(start_index, len(image_files) - 1))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        cv2.namedWindow('SAM2 Interactive Viewer', cv2.WINDOW_NORMAL)

        # Cache for visualizations to avoid recomputation
        vis_cache = {}

        while True:
            # Get current file paths
            img_path = image_files[current_index]
            basename = os.path.splitext(os.path.basename(img_path))[0]

            bbox_path = os.path.join(bbox_dir, basename + '.txt')
            if not os.path.exists(bbox_path):
                bbox_path = None

            try:
                # Check cache first
                cache_key = f"{current_index}_{basename}"
                if cache_key not in vis_cache:
                    print(f"Processing {basename}...")
                    vis_img = self.predict_and_visualize(img_path, bbox_path)
                    vis_cache[cache_key] = vis_img
                else:
                    vis_img = vis_cache[cache_key]

                if vis_img is None:
                    print(f"Failed to process {basename}")
                    current_index = (current_index + 1) % len(image_files)
                    continue

                # Convert RGB to BGR for OpenCV display
                display_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

                # Add status bar
                status_text = f"Image {current_index + 1}/{len(image_files)}: {basename}"
                bbox_status = "with BBox" if bbox_path else "no BBox"
                full_status = f"{status_text} ({bbox_status})"

                cv2.putText(display_img, full_status, (10, display_img.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_img, full_status, (10, display_img.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                # Resize for display
                scale_factor = 1.5
                new_width = int(display_img.shape[1] * scale_factor)
                new_height = int(display_img.shape[0] * scale_factor)
                display_img = cv2.resize(display_img, (new_width, new_height))
                cv2.resizeWindow('SAM2 Interactive Viewer', new_width, new_height)
                cv2.imshow('SAM2 Interactive Viewer', display_img)

                # Wait for key press
                key = cv2.waitKey(0) & 0xFF

                # Handle key presses
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('n') or key == ord('d') or key == 83:  # 'n', 'd', or Right Arrow
                    current_index = (current_index + 1) % len(image_files)
                elif key == ord('p') or key == ord('a') or key == 81:  # 'p', 'a', or Left Arrow
                    current_index = (current_index - 1) % len(image_files)
                elif key == 32:  # Space
                    current_index = (current_index + 1) % len(image_files)
                elif key == ord('r'):  # Rerun prediction
                    cache_key = f"{current_index}_{basename}"
                    if cache_key in vis_cache:
                        del vis_cache[cache_key]
                    print(f"Rerunning prediction for {basename}...")
                elif key == ord('s') and save_dir:  # Save
                    save_path = os.path.join(save_dir, f'{basename}_sam2_prediction.png')
                    cv2.imwrite(save_path, display_img)
                    print(f"Saved: {save_path}")
                else:
                    # Show help
                    print("\nControls:")
                    print("- 'n'/'d'/Right Arrow/Space: Next image")
                    print("- 'p'/'a'/Left Arrow: Previous image")
                    print("- 's': Save current visualization")
                    print("- 'r': Rerun prediction on current image")
                    print("- 'q'/ESC: Quit")

            except Exception as e:
                print(f"Error processing {basename}: {e}")
                current_index = (current_index + 1) % len(image_files)

        cv2.destroyAllWindows()
        print(f"Interactive session ended. Last viewed: {current_index + 1}/{len(image_files)}")


# Usage example
if __name__ == "__main__":
    # Configuration
    sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # Data paths
    data_root = "/home/jamie/labsvision_corpus/s3/chevron_ml_debug_merged/2025-04-09T19-34-56.365198Z"

    # Create timestamp for output
    timestamp = data_root.split('/')[-2] if '/' in data_root else datetime.now().strftime("%Y%m%d_%H%M%S")
    slot_name = data_root.split('/')[-1] if '/' in data_root else "unknown_slot"
    output_dir = f"./sam2_vis_output/{timestamp}/{slot_name}"

    # Initialize viewer
    print("Initializing SAM2 Interactive Viewer...")
    viewer = SAM2InteractiveViewer(sam2_checkpoint, model_cfg)

    # Start interactive mode
    print("\nStarting interactive SAM2 visualization...")
    viewer.interactive_visualization(data_root, output_dir, start_index=0)