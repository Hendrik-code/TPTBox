# Segmentation (`TPTBox.segmentation`)

Integration with external segmentation pipelines.  Provides a consistent `NII`-based interface
over SPINEPS, VibeSeg/TotalVibeSeg, and nnU-Net.

## Public API

```python
from TPTBox.segmentation import (
    run_spineps,
    run_vibeseg,
    run_nnunet,
    run_inference_on_file,
    extract_vertebra_bodies_from_VibeSeg,
    add_ribs_to_vert_spine,
)
```

## Key functions

| Function | Module | Description |
|---|---|---|
| `run_spineps(file_path, dataset=None, ...)` | `spineps.py` | Run SPINEPS spine segmentation on a file path / `BIDS_FILE`; returns a `dict` of output paths |
| `run_vibeseg(img_nii, ...)` | `VibeSeg/vibeseg.py` | Run VibeSeg body composition segmentation |
| `run_nnunet(img_nii, model_dir, ...)` | `VibeSeg/vibeseg.py` | Generic nnU-Net inference on a single NIfTI |
| `run_inference_on_file(path, ...)` | `nnUnet_utils/inference_api.py` | Low-level nnU-Net inference on a file path |
| `add_ribs_to_vert_spine(vert, spine, ...)` | `rib/add_ribs.py` | Merge left/right rib labels into an existing vertebra + spine segmentation; optionally runs VibeSeg (dataset 12) on the source CT to obtain the raw rib mask |

## Dependencies

| Pipeline | Requirement |
|---|---|
| SPINEPS | `pip install spineps` + model weights |
| VibeSeg | `pip install "TPTBox[seg]"` + model weights (auto-downloaded on first run) |
| Generic nnU-Net | `pip install "TPTBox[seg]"` + custom model directory |
| Rib assignment (`add_ribs_to_vert_spine`) | calls into VibeSeg/SPINEPS if the segmentation is missing. |

All external tools are optional: `import TPTBox.segmentation` succeeds without them, and each
entry point only raises (naming the extra to install) when it is actually called.

About the nnU-Net version. `TPTBox.segmentation.nnUnet_utils` is a self-contained fork: it reads
the checkpoint's own `plans.json` and builds its own `PlansManager`, so it is not tied to the
plans layout of whichever nnU-Net you have installed. The `seg` extra therefore mirrors SPINEPS'
own constraint rather than imposing one of its own — `nnunetv2>=2.8,<3.0` on Python 3.10+, and
`nnunetv2==2.4.2` on Python 3.9 (the last release that still supports it). That way
`TPTBox[seg]` and `spineps` can be installed side by side.

## Example

```python
from TPTBox import to_nii
from TPTBox.segmentation import run_spineps

# run_spineps takes a *path* (or BIDS_FILE), not an NII, and writes its results
# into the dataset's derivatives folder. It returns the output paths it produced.
output_paths = run_spineps("sub-001_T2w.nii.gz", model_semantic="t2w")

vert_seg = to_nii(output_paths["out_vert"], seg=True)
subreg_seg = to_nii(output_paths["out_spine"], seg=True)
print(vert_seg.unique())
```


Full script example for VIBEseg:
```python
"""
Example usage of VIBESeg for full-body MRI segmentation.

This script demonstrates how to run the VIBESeg pipeline on a single
NIfTI image and store the resulting segmentation to disk.
"""

from TPTBox.segmentation import run_vibeseg


def main() -> None:
    """
    Run VIBESeg on a single input image.
    """
    image = "path_or_nii_of_img.nii.gz"
    output_path = "VIBESeg.nii.gz"

    run_vibeseg(
        image=image,
        out_path=output_path,
        override=True,
        gpu=0,
        ddevice="cuda",
        # dataset_id=100,  # defaults to the newest available model
        padd=5,
        # Update the memory estimation
        memory_base=5000,  # Base memory in MB, default is 5GB
        memory_factor=160,  # prod(shape)*memory_factor/1000, 160 -> 30 GB
        memory_max=16000,  # in MB, here is 16GB
        wait_till_gpu_percent_is_free=0.1,
    )


if __name__ == "__main__":
    main()
```

## Adding ribs to an existing spine segmentation

```python
from TPTBox import to_nii
from TPTBox.segmentation import add_ribs_to_vert_spine

# Case 1: raw rib mask already exists
vert_out, spine_out = add_ribs_to_vert_spine(
    vert="sub-01_seg-vert.nii.gz",
    spine="sub-01_seg-spine.nii.gz",
    rib_seg="sub-01_seg-VIBESeg-12.nii.gz",
    save=True,  # writes back to vert / spine paths
)

# Case 2: no rib mask — VibeSeg dataset 12 is run on the CT
vert_out, spine_out = add_ribs_to_vert_spine(
    vert="sub-01_seg-vert.nii.gz",
    spine="sub-01_seg-spine.nii.gz",
    ct="sub-01_ct.nii.gz",
    rib_seg_out="sub-01_seg-VIBESeg-12.nii.gz",
)
```

Pass `split_touching=True` (default) so ribs of adjacent vertebrae that touch
front / middle / back get separated via erosion before assignment.
