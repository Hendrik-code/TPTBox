# Segmentation (`TPTBox.segmentation`)

Integration with external segmentation pipelines.  Provides a consistent `NII`-based interface
over SPINEPS, VibeSeg/TotalVibeSeg, and nnU-Net.

## Public API

```python
from TPTBox.segmentation import (
    run_spineps,
    run_vibeseg,
    run_totalvibeseg,
    run_nnunet,
    run_inference_on_file,
    extract_vertebra_bodies_from_VibeSeg,
)
```

## Key functions

| Function | Module | Description |
|---|---|---|
| `run_spineps(img_nii, model, ...)` | `spineps.py` | Run SPINEPS spine segmentation on a NIfTI; returns vertebra + subregion masks |
| `run_vibeseg(img_nii, ...)` | `VibeSeg/vibeseg.py` | Run VibeSeg body composition segmentation |
| `run_totalvibeseg(img_nii, ...)` | `VibeSeg/vibeseg.py` | Run TotalVibeSeg — extended label set |
| `run_nnunet(img_nii, model_dir, ...)` | `VibeSeg/vibeseg.py` | Generic nnU-Net inference on a single NIfTI |
| `run_inference_on_file(path, ...)` | `nnUnet_utils/inference_api.py` | Low-level nnU-Net inference on a file path |
| `extract_vertebra_bodies_from_VibeSeg(seg)` | `VibeSeg/vibeseg.py` | Post-process VibeSeg output to isolate vertebra bodies |

## Dependencies

| Pipeline | Requirement |
|---|---|
| SPINEPS | `pip install spineps` + model weights |
| VibeSeg / TotalVibeSeg | `pip install nnunetv2` + model weights (auto-downloaded on first run) |
| Generic nnU-Net | `pip install nnunetv2` + custom model directory |

All external tools are imported lazily — the core TPTBox package installs and imports cleanly
without them.

## Example

```python
from TPTBox import NII
from TPTBox.segmentation import run_spineps

ct = NII.load("ct.nii.gz", seg=False)
vert_seg, subreg_seg = run_spineps(ct, model="small")
vert_seg.save("vertebrae.nii.gz")
```
