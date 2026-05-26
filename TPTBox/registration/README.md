# Registration (`TPTBox.registration`)

Image registration utilities supporting rigid (point- and intensity-based) and deformable
registration.  Wraps ANTs (via SimpleITK) and the optional DeepALI deep learning backend.

## Public API

```python
from TPTBox.registration import (
    Point_Registration,
    ridged_points_from_poi,
    ridged_points_from_subreg_vert,
    Deformable_Registration,
    Template_Registration,
    General_Registration,          # requires hf-deepali
    Rigid_Elements_Registration,   # requires hf-deepali
)
```

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `Point_Registration` | `_ridged_points/point_registration.py` | Rigid registration from paired 3D landmark sets |
| `ridged_points_from_poi(fixed, moving, poi_fixed, poi_moving)` | same | Convenience wrapper: align two NIIs using POI correspondences |
| `ridged_points_from_subreg_vert(...)` | same | Same but derives POIs from vertebra+subregion segmentations automatically |
| `Deformable_Registration` | `_deformable/deformable_reg.py` | ANTs-based deformable (SyN) registration |
| `Template_Registration` | `_deformable/deformable_reg.py` | Deformable registration to an atlas/template |
| `General_Registration` | `_deepali/` | DeepALI deep-learning registration (requires `hf-deepali`) |
| `Rigid_Elements_Registration` | `_deepali/` | Per-element rigid registration via DeepALI |

## Installation of optional dependency

```bash
pip install hf-deepali   # only needed for General_Registration / Rigid_Elements_Registration
```

## Example

```python
from TPTBox.registration import ridged_points_from_poi

fixed_nii = NII.load("fixed.nii.gz", seg=False)
moving_nii = NII.load("moving.nii.gz", seg=False)

registered, transform = ridged_points_from_poi(
    fixed_nii, moving_nii,
    poi_fixed=poi_fixed,
    poi_moving=poi_moving,
)
registered.save("registered.nii.gz")
```
