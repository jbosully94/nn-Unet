# Batch Segmentation with nnU-Net API

## Setup

```bash
export CUDA_VISIBLE_DEVICES=1
```
This will use the second GPU by default when you set this. 
## Script

```python
#!/usr/bin/env python3
import numpy as np
import tifffile
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

base = Path("/gdata/dm/EBERLIGHT/Shabtai/7BM/2025-3/Shabtai-20251021-e281641/analysis")
output_dir = base / "Segmented_NoMask"
output_dir.mkdir(exist_ok=True)

predictor = nnUNetPredictor()
predictor.initialize_from_trained_model_folder(
    "/home/beams/OSULLIVANJ/nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all",
    use_folds="all",
    checkpoint_name="checkpoint_best.pth"
)

for folder in sorted(base.glob("*/recon_denoised")):
    name = folder.parent.name
    
    if (output_dir / f"{name}.tif").exists():
        print(f"{name}: skip")
        continue
    
    slices = sorted(folder.glob("*.tif*"))
    if not slices:
        continue
    
    print(f"{name}: loading {len(slices)} slices...")
    stack = np.stack([tifffile.imread(s) for s in slices])
    
    print(f"{name}: segmenting...")
    pred = predictor.predict_single_npy_array(
        stack[None],
        {'spacing': [1.1, 1.1, 1.1]},
        None,
        None,
        False
    )
    
    tifffile.imwrite(output_dir / f"{name}.tif", pred.astype(np.uint8))
    print(f"{name}: done\n")
    
    del stack, pred
```

## Run

```bash
python batch_segment.py
```

## Output

- Location: `{base}/Segmented_NoMask/`
- Format: 3D TIFF per sample
- Values: 0=matrix, 1=pores, 2=POM

## Notes

- Model loads once, predicts all volumes
- No temp files created
- Skips already-processed samples
- Adjust `base` path and `spacing` as needed
