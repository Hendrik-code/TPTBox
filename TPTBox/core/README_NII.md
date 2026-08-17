# TPTBox: `nii_wrapper.py` — NIfTI image wrapper

The `core` subpackage is the foundation of TPTBox. It provides the three primary abstractions —
`NII`, `POI`, and `BIDS_FILE` — along with helper utilities for array operations and anatomical constants.

| Symbol | Description |
|---|---|
| `NII` | Wraps `nibabel.Nifti1Image`; the central image type throughout TPTBox |
| `NII.load(path, seg)` | Load a NIfTI file from disk (classmethod) |
| `NII.from_numpy(arr, affine, seg)` | Construct from a numpy array and affine matrix |
| `NII.reorient(axcodes_to)` | Reorient to a canonical axis code (e.g. `("R","A","S")`) |
| `NII.rescale(voxel_spacing)` | Resample to new voxel spacing in mm |
| `NII.resample_from_to(other)` | Resample to match the grid of another `NII` |
| `NII.apply_mask(mask)` | Zero-out voxels outside a binary/label mask |
| `NII.map_labels(label_map)` | Remap integer labels |
| `NII.save(path)` | Save to disk as `.nii` or `.nii.gz` |
| `NII.get_array()` | Return a copy of the underlying numpy array |
| `NII.get_seg_array()` | Same as `get_array()` but asserts `seg=True` |
| `Image_Reference` | Type alias: `BIDS_FILE | Nifti1Image | Path | str | NII` |

```python
from TPTBox import NII

# Image
nii = NII.load("path/to/img.nii.gz", seg=False)
# Segmentation
seg = NII.load("path/to/seg.nii.gz", seg=True)

# Standardize the image to a fixed orientation (Right-Anterior-Superior in the nibabel coordinate system)
# and resample it to an isotropic resolution of 1 mm x 1 mm x 1 mm
nii_rescaled = nii.reorient("RAS").rescale((1, 1, 1))

# One-line function to resample another image to match a reference image
seg_resampled = seg.resample_from_to(nii_rescaled)

# The appropriate resampling method is automatically selected depending on
# whether the image represents a segmentation or a continuous-valued image.
```
