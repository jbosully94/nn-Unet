# nnU-Net for 3D Soil CT Segmentation

## Installation

I use a plain Python venv for this rather than conda:

```bash
python -m venv ~/pathtowherever/nnunet
source ~/pathtowherever/nnunet/bin/activate
pip install nnunetv2 tifffile
```

## Environment Setup

These need to be set each session. Add them to your `~/.bashrc` to avoid doing it manually every time:

```bash
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

eberproc1 and 2 each have two GPUs. You can pick one or use both:

```bash
export CUDA_VISIBLE_DEVICES=0    # GPU 0
export CUDA_VISIBLE_DEVICES=1    # GPU 1
export CUDA_VISIBLE_DEVICES=0,1  # both
```

## Dataset ID

Every nnU-Net dataset needs a unique 3-digit ID (001–999) used throughout the pipeline:

```
Dataset100_SoilCT     → -d 100
Dataset101_Roots      → -d 101
Dataset200_NewProject → -d 200
```

The folder must follow `Dataset{ID}_{Name}`. The name after the underscore can be anything.

## Data Preparation

### What you need

1. **Images** — 3D TIFF stacks of your CT scans
2. **Labels** — 3D TIFF stacks with annotation values, same dimensions as images

### Folder structure

```
$nnUNet_raw/Dataset100_SoilCT/
├── dataset.json
├── imagesTr/
│   ├── s000_0000.tif
│   ├── s000.json
│   ├── s001_0000.tif
│   ├── s001.json
│   └── ...
└── labelsTr/
    ├── s000.tif
    ├── s000.json
    ├── s001.tif
    ├── s001.json
    └── ...
```

### Naming rules

| File type | Format | Example |
|-----------|--------|---------|
| Image | `{scan_id}_0000.tif` | `s000_0000.tif` |
| Label | `{scan_id}.tif` | `s000.tif` |
| Spacing (both) | `{scan_id}.json` | `s000.json` |

The `_0000` suffix on images indicates channel 0. CT data has one channel, labels don't have this suffix.

### Spacing JSON

Each image and label needs a spacing file:

```bash
echo '{"spacing": [1.1, 1.1, 1.1]}' > s000.json
```

This describes the voxel spacing across X, Y, Z — useful if your sampling is uneven between axes.

### Label values

nnU-Net requires the ignore label to be the highest value. Dragonfly outputs:

```
0 = unlabelled
1 = matrix
2 = pores
3 = POM
```

This needs remapping so unlabelled voxels become the ignore label:

```
0 = background (matrix)
1 = pores
2 = POM
3 = ignore (unlabelled)
```

`scripts/prepare_dataset.py` handles copying, renaming, remapping, and JSON creation. Set the paths and dataset ID at the top of the script, then run it.

## Training Pipeline

### 1. Preprocess

```bash
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
```

This works out the best configuration based on your data — patch size, batch size, architecture. Use `-np 1` if you run out of RAM. Normalisation also happens here, per class, and it works well.

### 2. Train

```bash
# Single GPU
nnUNetv2_train 100 3d_fullres all -device cuda

# Two GPUs
nnUNetv2_train 100 3d_fullres all -device cuda -num_gpus 2
```

- `100` = dataset ID
- `3d_fullres` = 3D network at full resolution
- `all` = train on all data (no validation split)
- Runs for 1000 epochs, no early stopping

Running on two A100s roughly halves training time — still ~60 seconds per epoch. Resume interrupted training with `--c`:

```bash
nnUNetv2_train 100 3d_fullres all -device cuda --c
```

### 3. Predict

**Command line:**

```bash
mkdir -p ~/predict_input ~/predict_output
cp myimage.tif ~/predict_input/test_0000.tif
echo '{"spacing": [1.1, 1.1, 1.1]}' > ~/predict_input/test.json

nnUNetv2_predict -i ~/predict_input -o ~/predict_output -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth
```

Disable test-time augmentation for faster prediction (less accurate, more noticeable on small images):

```bash
nnUNetv2_predict ... --disable_tta
```

**Python API (batch):**

`scripts/segment.py` loads the model once and iterates over a directory of volumes. Set `base`, `output_dir`, and `model_dir` at the top. Label values in the output are `0=matrix, 1=pores, 2=POM`.

For parallel prediction across two GPUs, run two separate instances with different `CUDA_VISIBLE_DEVICES` pointing at non-overlapping inputs.

## Fine-Tuning

To continue training from an existing checkpoint with new data:

```bash
nnUNetv2_train 100 3d_fullres all -device cuda -pretrained_weights /path/to/checkpoint_best.pth
```

Requirements: same number of classes, same number of input channels. New data must be preprocessed first.

## Sparse Annotation

You don't need to label every voxel. nnU-Net supports partial annotation:

- Label only some slices (e.g. every 10th)
- Paint sparse regions or scribbles
- Set unlabelled voxels to the ignore label (highest value)

The network sees full 3D context but only trains on labelled voxels.

## Region-Based Labels

If you have an aggregate and want to segment pores and POM within it, you can define hierarchical labels in `dataset.json`. Haven't tested this yet but it could be useful:

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

The network learns that aggregate = pores + matrix + POM. You annotate:
- `0` = air outside aggregate
- `1` = pores inside aggregate
- `2` = matrix inside aggregate
- `3` = POM inside aggregate

## File Locations

| What | Where |
|------|-------|
| Raw data | `$nnUNet_raw/Dataset100_SoilCT/` |
| Preprocessed | `$nnUNet_preprocessed/Dataset100_SoilCT/` |
| Trained model | `$nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/` |
| Checkpoints | `checkpoint_best.pth`, `checkpoint_final.pth` |
| Training progress | `progress.png` in fold directory |
