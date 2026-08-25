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
    General_Registration,  # requires hf-deepali
    Rigid_Elements_Registration,  # requires hf-deepali
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
from TPTBox import NII, POI
from TPTBox.registration import Point_Registration

poi_fixed = POI.load("path/to/poi.json")
poi_moving = POI.load("path/to/poi.json")
# update resolution/orientation of poi_fixed, if you would like the resampe into an specific space
reg_obj = Point_Registration(poi_fixed, poi_moving)
# appling the transformation
nii_moving = NII.load("path/to/moving_img.nii.gz", False)
nii_moved = reg_obj.transform_nii(nii_moving)
poi_moved = reg_obj.transform_poi(poi_moving)
```
