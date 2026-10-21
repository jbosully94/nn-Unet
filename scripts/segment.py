import numpy as np
import tifffile
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# folders
input_dir = Path("/path/to/analysis")
output_dir = Path("/path/to/analysis/Segmented")
model_dir = Path("/path/to/nnUNet_results/Dataset100_SoilCT/nnUNetTrainer__nnUNetPlans__3d_fullres")

# settings
folds = ("all",)
checkpoint = "checkpoint_final.pth"
spacing = [1.1, 1.1, 1.1]
tile_step_size = 0.5  # 0.5 means patches overlap by 50%
use_gaussian = True
use_mirroring = True

output_dir.mkdir(exist_ok=True)

# load the model
predictor = nnUNetPredictor(
    tile_step_size=tile_step_size,
    use_gaussian=use_gaussian,
    use_mirroring=use_mirroring,
    device=torch.device("cuda"),
)
predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=folds, checkpoint_name=checkpoint)

# go through each sample
samples = sorted(input_dir.glob("*/recon_denoised"))

for folder in samples:
    name = folder.parent.name
    out_file = output_dir / (name + ".tif")

    if out_file.exists():
        print(name, "already done, skipping")
        continue

    slices = sorted(folder.glob("*.tif*"))
    if len(slices) == 0:
        print(name, "no tif files found")
        continue

    print(name, "loading", len(slices), "slices")
    stack = np.stack([tifffile.imread(s) for s in slices])

    print(name, "segmenting")
    pred = predictor.predict_single_npy_array(stack[None], {"spacing": spacing}, None, None, False)

    tifffile.imwrite(out_file, pred.astype(np.uint8))
    print(name, "done")
