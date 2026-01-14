# Batch Segmentation Workflow

## Prep script

```python
#!/usr/bin/env python3
import numpy as np
import tifffile
import json
from pathlib import Path

base = Path("/gdata/dm/EBERLIGHT/Shabtai/7BM/2025-3/Shabtai-20251021-e281641/analysis")
input_dir = Path.home() / "nnUNet_batch/input"
input_dir.mkdir(parents=True, exist_ok=True)

for folder in sorted(base.glob("*/recon_denoised")):
    name = folder.parent.name
    slices = sorted(folder.glob("*.tif*"))
    if not slices:
        continue
    
    print(f"{name}: {len(slices)} slices")
    stack = np.stack([tifffile.imread(s) for s in slices])
    
    tifffile.imwrite(input_dir / f"{name}_0000.tif", stack)
    json.dump({"spacing": [1.1, 1.1, 1.1]}, open(input_dir / f"{name}.json", "w"))
```

## Run prediction

```bash
export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

nnUNetv2_predict -i ~/nnUNet_batch/input -o ~/nnUNet_batch/Segmented_NoMask -d 100 -c 3d_fullres -f all -chk checkpoint_best.pth --disable_tta
```

## Output

Segmented volumes saved to `~/nnUNet_batch/Segmented_NoMask/`

Values: 0=matrix, 1=pores, 2=POM
