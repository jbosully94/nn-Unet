# nnU-Net Guide for 3D Soil CT Segmentation

A practical guide for using nnU-Net v2 for semantic segmentation of soil CT data.

## Installation

```bash
pip install nnunetv2 tifffile
```

## Environment Setup

Run these each session (or add to `~/.bashrc`):

```bash
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

To specify GPU:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
```
As both eBERproc machines have 2 GPU's each, we are able to simultaniously train across both GPU's (significantly speeds up training) or we can segment two seperate images across both GPUs in parallel. 

## Data Preparation

### Folder Structure

```
$nnUNet_raw/Dataset100_SampleID/
├── dataset.json
├── imagesTr/
│   ├── s000_0000.tif
│   ├── s000.json
│   ├── s001_0000.tif
│   └── s001.json
└── labelsTr/
    ├── s000.tif
    ├── s000.json
    ├── s001.tif
    └── s001.json
```
So this in the naming convention, here we call each thing a "Dataset" the number has to be unique for each case and what follows after the underscore can be whatever you want to call it. For example, an experiment or a sample ID.

### Naming Convention
Here we are specifying case ID as an individual scan.
- **Images:** `{case_id}_0000.tif` (the `_0000` is the channel index)
- **Labels:** `{case_id}.tif` (no channel suffix)
- **Spacing JSON:** `{case_id}.json` (matches case_id, not full filename)

### Spacing JSON

Each image and label needs a spacing JSON file:

```bash
echo '{"spacing": [1.1, 1.1, 1.1]}' > s000.json
```
The spacing is used to determine the optimum uNet architecture.

Format: `[Z, Y, X]` in any consistent units. Only matters if you have mixed resolutionsm such as unequal voxel sizes.

### Label Values

nnU-Net requires:
- Background = 0
- Classes = consecutive integers (1, 2, ...)
- Ignore label (for sparse annotation) = highest integer

Example for soil CT:
| Class | Value |
|-------|-------|
| Matrix (background) | 0 |
| Pores | 1 |
| POM | 2 |
| Ignore (unlabeled) | 3 |

### Label Remapping

If your labels use different values, remap them:

```python
import numpy as np
import tifffile

lbl = tifffile.imread('label.tif')

# Remap: 0=unlabeled, 1=matrix, 2=pores, 3=POM
# To: 0=matrix, 1=pores, 2=POM, 3=ignore
remapped = np.full(lbl.shape, 3, dtype=np.uint8)  # default = ignore
remapped[lbl == 1] = 0  # matrix -> background
remapped[lbl == 2] = 1  # pores -> class 1
remapped[lbl == 3] = 2  # POM -> class 2

tifffile.imwrite('label_remapped.tif', remapped)
```


### dataset.json

Create `$nnUNet_raw/Dataset100_SoilCT/dataset.json`:

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "pores": 1, "POM": 2, "ignore": 3},
  "numTraining": 3,
  "file_ending": ".tif"
}
```

Update `numTraining` to match your sample count.

## Pipeline

### 1. Preprocess

```bash
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
```

This:
- Extracts dataset fingerprint (shapes, spacings, intensities)
- Plans network architecture (patch size, depth, etc.)
- Preprocesses data (resampling, normalization)

Use `-np 1` if you run out of RAM.

### 2. Train

```bash
# Single GPU
nnUNetv2_train 100 3d_fullres all -device cuda

# Multi-GPU
nnUNetv2_train 100 3d_fullres all -device cuda -num_gpus 2

# CPU only
nnUNetv2_train 100 3d_fullres all -device cpu

# Mac MPS
nnUNetv2_train 100 3d_fullres all -device mps
```

Notes:
- `100` = dataset ID
- `3d_fullres` = configuration (alternatives: `2d`, `3d_lowres`)
- `all` = train on all data (use `0-4` for cross-validation folds)
- Default: 1000 epochs

To resume interrupted training:
```bash
nnUNetv2_train 100 3d_fullres all -device cuda --c
```

### 3. Predict

Prepare input folder:
```bash
mkdir -p ~/predict_input ~/predict_output

cp myimage.tif ~/predict_input/sample_0000.tif
echo '{"spacing": [1.1, 1.1, 1.1]}' > ~/predict_input/sample.json
```

Run prediction:
```bash
nnUNetv2_predict -i ~/predict_input -o ~/predict_output -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth
```

Speed up prediction (minimal quality loss):
```bash
nnUNetv2_predict -i ~/predict_input -o ~/predict_output -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth --disable_tta --step_size 0.7
```

- `--disable_tta`: Skip test-time augmentation (~4x faster)
- `--step_size 0.7`: Less overlap between patches (default 0.5)

## Parallel Prediction on Multiple GPUs

**Terminal 1:**
```bash
tmux new -s predict0

export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

nnUNetv2_predict -i ~/input1 -o ~/output1 -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth --disable_tta

# Detach: Ctrl+B then D
```

**Terminal 2:**
```bash
tmux new -s predict1

export CUDA_VISIBLE_DEVICES=1
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

nnUNetv2_predict -i ~/input2 -o ~/output2 -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth --disable_tta

# Detach: Ctrl+B then D
```

**Reattach:** `tmux attach -t predict0`

**List sessions:** `tmux ls`

## Converting Image Sequences

If your data is a folder of 2D TIFFs:

```python
import numpy as np
import tifffile
from pathlib import Path

slices = sorted(Path('/path/to/sequence/').glob('*.tif'))
stack = np.stack([tifffile.imread(s) for s in slices])
tifffile.imwrite('volume.tif', stack)
```

## Sparse Annotation Notes

nnU-Net supports partial/sparse annotation:

- Set unlabeled voxels to the ignore label (highest value, e.g., 3)
- The network sees full 3D context but only computes loss on labeled voxels
- For 3D patches (~32 slices deep), annotate 10-15 slices spread across Z
- Warning "No annotated pixels in image!" is expected - some patches land in unlabeled regions

## Output

Prediction outputs have class values:
- 0 = matrix/background
- 1 = pores
- 2 = POM

## File Locations

| What | Where |
|------|-------|
| Raw data | `$nnUNet_raw/Dataset100_SoilCT/` |
| Preprocessed | `$nnUNet_preprocessed/Dataset100_SoilCT/` |
| Trained model | `$nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/` |
| Checkpoints | `checkpoint_best.pth`, `checkpoint_final.pth`, `checkpoint_latest.pth` |
| Training progress | `progress.png` in fold directory |
| Plans | `$nnUNet_preprocessed/Dataset100_SoilCT/nnUNetPlans.json` |

## Reference

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). 
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. 
Nature methods, 18(2), 203-211.
```
