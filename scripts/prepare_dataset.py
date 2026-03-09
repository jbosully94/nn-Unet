#!/usr/bin/env python3
"""
Prep data for nnU-Net.

Input labels: 0=unlabeled, 1=matrix, 2=pores, 3=POM
Output labels: 0=matrix, 1=pores, 2=POM, 3=ignore
"""
import json
import numpy as np
import tifffile
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
