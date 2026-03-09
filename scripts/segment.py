#!/usr/bin/env python3
import numpy as np
import tifffile
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

base = Path("/path/to/analysis")
output_dir = base / "Segmented"
model_dir = Path.home() / "nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all"
spacing = [1.1, 1.1, 1.1]

output_dir.mkdir(exist_ok=True)

predictor = nnUNetPredictor()
predictor.initialize_from_trained_model_folder(
    str(model_dir),
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
        {'spacing': spacing},
        None,
        None,
        False
    )

    tifffile.imwrite(output_dir / f"{name}.tif", pred.astype(np.uint8))
    print(f"{name}: done\n")

    del stack, pred
