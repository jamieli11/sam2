# SAM2 Data Cleaner for ML marker detection Model

## Overview
This repository contains customized SAM2 implementation for data cleaning tasks in the [ML marker detection project](https://github.com/LabsCubed-Inc/ML_marker_detect/tree/dev/jamie).

## Key Features (Newly Added)
- [sam2_data_cleaner.py](sam2_data_cleaner.py): Data cleaner script for ML marker detection datasets
- [convert_data_format.py](convert_data_format.py): Data preparation for finetuning SAM2 on ML marker detection model 
- [data_check_interactive.py](data_check_interactive.py): Dataset quality check script
- [sam2_data_cleaner.yaml](sam2/configs/sam2.1_training/sam2_data_cleaner.yaml) and [sam2_data_cleaner_gpu.yaml](sam2/configs/sam2.1_training/sam2_data_cleaner_gpu.yaml): Single frame model training using cpu and gpu

## Quick Start
### SAM2 data cleaner
**Input**

Use [data generation code](https://github.com/LabsCubed-Inc/labsvision/tree/dev/ddarolfi/ml_marker_detect_data_generation) to generate datasets from [pull videos](s3://labscubed-dev/labscubedone/cubeone-2.0/labsvision_corpus/SN0035_data/). 
ML marker detection datasets should be organized in the following directory structure:
```
./data_root
├── Annotations
│   ├── anno1.txt
│   └── anno2.txt
├── Images
│   ├── image1.jpg
│   └── image2.jpg
└── Masks
    ├── mask1.jpg
    └── mask2.jpg
```

**Usage**
```bash
python sam2_cleaner.py --input_root $INPUT_PATH --output_root $OUTPUT_PATH
```
where:
- `--input_root`: input directory containing Images, Masks, and Annotations folders
- `--output_root`: output directory for cleaned dataset
- `--area_threshold`: area change threshold for frame rejection (default: 0.6)
- `--backup_rejected`: backup rejected frames (default: enabled)

**Output**

The output directory follows the same structure as the input, with additional information about rejected frames.

**Important Notice**

The current data cleaner CANNOT guarantee 100% accuracy. 
Manual verification using [data_check_interactive.py](data_check_interactive.py) is recommended to identify and filter any mislabeled data before using the cleaned dataset.

For example, after manual inspection, 682 out of 9,670 samples from the [chevron cube10 dataset](https://labscubed-dev.s3.us-east-2.amazonaws.com/labscubedone/cubeone-2.0/labsvision_corpus/Ml_marker_detect/Dataset/chevron_cube10_final.zip) were filtered out. 
The filtered-out dataset can be found in the [S3 bucket](https://labscubed-dev.s3.us-east-2.amazonaws.com/labscubedone/cubeone-2.0/labsvision_corpus/Ml_marker_detect/Dataset/chevron_250828_manually_filtered_rejected.zip).

### Finetune SAM2 data cleaner

The SAM2 data cleaner can be fine-tuned using the following commands:
```bash
# cpu debug
python3 $SAM2_REPO_PATH/sam2/training/train.py -c configs/sam2.1_training/sam2_data_cleaner.yaml --use-cluster 0 --num-gpus 1

# gpu training
python3 $SAM2_REPO_PATH/sam2/training/train.py -c configs/sam2.1_training/sam2_data_cleaner_gpu.yaml --use-cluster 0 --num-gpus $NUM_GPUS
```

**Training Requirements**

SAM2's training code is primarily designed for GPU execution, with memory monitoring and performance statistics that call CUDA functions.

Custom modifications have been made to enable CPU training for debugging purposes, though GPU training is strongly recommended for performance.

Even for CPU training, you must set `--num-gpus` 1 to ensure the training script enters the correct execution branch. 
Setting `--num-gpus 0` causes `num_proc = 0`, leading to premature function termination.


## Original SAM2 Documentation
For complete SAM2 documentation, see the original [README.md](README.md)

