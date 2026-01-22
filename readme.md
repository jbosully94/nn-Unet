# nnU-Net Guide for 3D Soil CT Segmentation

## Installation

I like to setup a new python environment. I've moved away from conda recently, just using the normal python one for this
```bash
python -m venv ~/pathtowherever/nnunet
source ~/pathtowherever/nnunet/bin/activate
pip install nnunetv2 tifffile
```

## Environment Setup

Run these each session, there's a way to make it so you don't have to run this every time. I think its called bashrc?

```bash
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```
eberproc1 and 2 has two GPU's. You can specify which one you want to use, or both!

To specify GPU:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
```

## Understanding Dataset ID

Every nnU-Net dataset needs a unique 3-digit ID (001-999). This ID is used throughout the pipeline.

```
Dataset100_SoilCT     → -d 100
Dataset101_Roots      → -d 101
Dataset200_NewProject → -d 200
```

The folder must be named Dataset{ID}_{Name}. The name after the underscore can be anything.

## Data Preparation

### What You Need

1. **Images** - 3D TIFF stacks of your CT scans
2. **Labels** - 3D TIFF stacks with annotation values (same dimensions as images)

### Folder Structure

```
$nnUNet_raw/Dataset100_SoilCT/
├── dataset.json
├── imagesTr/
│   ├── s000_0000.tif    # Image 1
│   ├── s000.json        # Spacing for image 1
│   ├── s001_0000.tif    # Image 2
│   ├── s001.json        # Spacing for image 2
│   └── ...
└── labelsTr/
    ├── s000.tif         # Labels for image 1
    ├── s000.json        # Spacing for labels 1
    ├── s001.tif         # Labels for image 2
    ├── s001.json        # Spacing for labels 2
    └── ...
```

### Naming Rules

| File Type | Format | Example |
|-----------|--------|---------|
| Image | `{scan_id}_0000.tif` | `s000_0000.tif` |
| Label | `{scan_id}.tif` | `s000.tif` |
| Spacing (both) | `{scan_id}.json` | `s000.json` |

The `_0000` suffix on images indicates channel 0 (CT data has one channel). Labels don't have this suffix.

### Spacing JSON

Each image and label needs a spacing file:

```bash
echo '{"spacing": [1.1, 1.1, 1.1]}' > s000.json
```

I don't really understand what this is for. I literally just set this as the resolution but I don't think that's what it's for. Perhaps uneven sampling in diffrent X, Y and Z directions.
### Label Values

nnU-Net requires specific label values:

0 is background, 1,2,3 etc. are your classes (pores, POM) while the ignore labels are the highest value. For example in soil we might do

0 unlabelled
1 is matrix
2 is pores
3 is POM

this is the typical output from draongfly labeling. But the issue is, the highest number needs to the unlabeled voxels in nnUnet! So we have to remap the values so 

0 is background (soil matrix)
1 is pores
2 is POM
3 is unlabeled

I do this in a python script below.

## Complete Data Preparation Script

This script handles everything - copying, renaming, remapping, and JSON creation:

```python
#!/usr/bin/env python3
"""
Prep data for nnU-Net.

Input labels: 0=unlabeled, 1=matrix, 2=pores, 3=POM
Output labels: 0=matrix, 1=pores, 2=POM, 3=ignore
"""
import numpy as np
import tifffile
import json
from pathlib import Path

image_dir = Path("/path/to/images")
label_dir = Path("/path/to/labels")
output_dir = Path.home() / "nnUNet_raw"
dataset_id = 100
dataset_name = "SoilCT"
spacing = [1.1, 1.1, 1.1]

out = output_dir / f"Dataset{dataset_id}_{dataset_name}"
(out / "imagesTr").mkdir(parents=True, exist_ok=True)
(out / "labelsTr").mkdir(parents=True, exist_ok=True)

count = 0
for img_path in sorted(image_dir.glob("*.tif*")):
    lbl_path = label_dir / img_path.name
    if not lbl_path.exists():
        print(f"SKIP: {img_path.name}")
        continue
    
    scan_id = f"s{count:03d}"
    
    img = tifffile.imread(img_path)
    tifffile.imwrite(out / "imagesTr" / f"{scan_id}_0000.tif", img.astype(np.float32))
    
    lbl = tifffile.imread(lbl_path)
    remapped = np.full(lbl.shape, 3, dtype=np.uint8)
    remapped[lbl == 1] = 0
    remapped[lbl == 2] = 1
    remapped[lbl == 3] = 2
    tifffile.imwrite(out / "labelsTr" / f"{scan_id}.tif", remapped)
    
    for folder in ["imagesTr", "labelsTr"]:
        json.dump({"spacing": spacing}, open(out / folder / f"{scan_id}.json", "w"))
    
    print(f"{img_path.name} -> {scan_id}")
    count += 1

json.dump({
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "pores": 1, "POM": 2, "ignore": 3},
    "numTraining": count,
    "file_ending": ".tif"
}, open(out / "dataset.json", "w"), indent=2)

print(f"\nDone: {count} samples")
```

## Training Pipeline

### 1. Preprocess

```bash
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
```

This basically determines the best settings based on your data. It's creating a training plan. Use `-np 1` if you run out of RAM.

I think it is in this step where the data normalisation occurs. Essentially, it normalises each class in the training data. I found it works well, much better than whatever Dragonfly was doing. 


### 2. Train

```bash
# Single GPU
nnUNetv2_train 100 3d_fullres all -device cuda

# Two GPUs
nnUNetv2_train 100 3d_fullres all -device cuda -num_gpus 2
```

- `100` = your dataset ID
- `3d_fullres` = 3D network at full resolution
- `all` = train on all data (no validation split)
- Default is 1000 epochs

I use 3d_fullres but because eBERproc1 and 2 have sufficient resources to do so. Running on 2 GPU's (2 A100) instead of 1 effectively halves training time. Still 60 seconds an epoch though! It will also train over 1000 epochs, no early stopping like Dragonfly.

Resume interrupted training:
```bash
nnUNetv2_train 100 3d_fullres all -device cuda --c
```

### 3. Predict (Command Line)

```bash
# Prepare input
mkdir -p ~/predict_input ~/predict_output
cp myimage.tif ~/predict_input/test_0000.tif
echo '{"spacing": [1.1, 1.1, 1.1]}' > ~/predict_input/test.json

# Run prediction
nnUNetv2_predict -i ~/predict_input -o ~/predict_output -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth

# Faster prediction (disable test-time augmentation)
nnUNetv2_predict -i ~/predict_input -o ~/predict_output -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth --disable_tta
```

The test-time augmentation is good, especially if you have small images. It spee

## Batch Prediction with Python API

You can predict images directly, 
```python
#!/usr/bin/env python3
import numpy as np
import tifffile
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path("/path/to/images")
output_dir = Path("/path/to/output")
output_dir.mkdir(exist_ok=True)

predictor = nnUNetPredictor(device=torch.device('cuda', 0))
predictor.initialize_from_trained_model_folder(
    "/path/to/nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres",
    use_folds="all",
    checkpoint_name="checkpoint_best.pth"
)
predictor.use_mirroring = False  # disable TTA for speed

for tif in sorted(input_dir.glob("*.tif")):
    name = tif.stem
    
    if (output_dir / f"{name}.tif").exists():
        print(f"{name}: skip")
        continue
    
    print(f"{name}: loading...")
    stack = tifffile.imread(tif)
    
    print(f"{name}: segmenting...")
    pred = predictor.predict_single_npy_array(
        input_image=stack[None],
        image_properties={'spacing': [1.1, 1.1, 1.1]},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False
    )
    
    tifffile.imwrite(output_dir / f"{name}.tif", pred.astype(np.uint8))
    print(f"{name}: done\n")
    
    del stack, pred
```

## Parallel Prediction on Two GPUs

Run two separate Python scripts:

**Terminal 1:**
```bash
export CUDA_VISIBLE_DEVICES=0
python segment_batch1.py
```

**Terminal 2:**
```bash
export CUDA_VISIBLE_DEVICES=1
python segment_batch2.py
```

## Fine-Tuning an Existing Model

To continue training with new data:

```bash
nnUNetv2_train 100 3d_fullres all -device cuda -pretrained_weights /path/to/checkpoint_best.pth
```

Requirements:
- Same number of classes
- Same number of input channels
- New data must be preprocessed first

## Sparse Annotation

You don't need to label every voxel. nnU-Net supports partial annotation:

- Label only some slices (e.g., every 10th slice)
- Paint sparse regions/scribbles
- Set unlabeled voxels to ignore label (highest value)

The network sees full 3D context but only trains on labeled voxels. 

## Region-Based Labels
Let's say you have an aggregate, and you want to segment the pores and POM within that aggregate this might be a good option. I am yet to try this but could be worth a shot. 

```json
{
  "channel_names": {"0": "CT"},
  "labels": {
    "background": 0,
    "aggregate": [1, 2, 3],
    "pores": 1,
    "matrix": 2,
    "POM": 3
  },
  "numTraining": 3,
  "file_ending": ".tif"
}
```

Network learns that aggregate = pores + matrix + POM. You annotate:
- 0 = air outside aggregate
- 1 = pores inside aggregate
- 2 = matrix inside aggregate  
- 3 = POM inside aggregate

HAVEN'T TESTED THIS THOUGH!

## File Locations

| What | Where |
|------|-------|
| Raw data | `$nnUNet_raw/Dataset100_SoilCT/` |
| Preprocessed | `$nnUNet_preprocessed/Dataset100_SoilCT/` |
| Trained model | `$nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/` |
| Checkpoints | `checkpoint_best.pth`, `checkpoint_final.pth` |
| Training progress | `progress.png` in fold directory |
