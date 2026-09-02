# Registration (`TPTBox.registration`)

Image registration utilities supporting rigid (point- and intensity-based) and deformable
registration.  Point registration is built on SimpleITK; every intensity-based and
deformable backend is built on [DeepALI](https://github.com/BioMedIA/deepali) (PyTorch) and
needs the optional `hf-deepali` package.

## Public API

```python
from TPTBox.registration import (
    Point_Registration,
    ridged_points_from_poi,
    ridged_points_from_subreg_vert,
    Deepali_Point_Registration,  # requires hf-deepali
    ridged_points_from_poi_deepali,  # requires hf-deepali
    ridged_points_from_subreg_vert_deepali,  # requires hf-deepali
    Deformable_Registration,  # requires hf-deepali
    Template_Registration,  # requires hf-deepali
    Template_Registration2,  # requires hf-deepali
    General_Registration,  # requires hf-deepali
    Rigid_Elements_Registration,  # requires hf-deepali
)
```

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `Point_Registration` | `_ridged_points/point_registration.py` | Rigid registration from paired 3D landmark sets |
| `ridged_points_from_poi(poi_fixed, poi_moving, ...)` | same | Convenience wrapper: rigid transform from two POI sets |
| `ridged_points_from_subreg_vert(...)` | same | Same but derives POIs from vertebra+subregion segmentations automatically |
| `Deepali_Point_Registration` | `_ridged_points/deepali_point_registration.py` | Closed-form point registration on the DeepALI backend (requires `hf-deepali`) |
| `ridged_points_from_poi_deepali(...)` | same | DeepALI variant of `ridged_points_from_poi` (requires `hf-deepali`) |
| `ridged_points_from_subreg_vert_deepali(...)` | same | DeepALI variant of `ridged_points_from_subreg_vert` (requires `hf-deepali`) |
| `Deformable_Registration` | `_deformable/deformable_reg.py` | DeepALI/PyTorch deformable registration (requires `hf-deepali`) |
| `Template_Registration` | `_deformable/multilabel_segmentation.py` | Deformable registration to an atlas/template (requires `hf-deepali`) |
| `Template_Registration2` | `_deformable/multilabel_segmentation.py` | Variant of `Template_Registration` with an optional pre-registration (requires `hf-deepali`) |
| `General_Registration` | `_deepali/deepali_model.py` | DeepALI deep-learning registration (requires `hf-deepali`) |
| `Rigid_Elements_Registration` | `_deepali/spine_rigid_elements_reg.py` | Per-element rigid registration via DeepALI (requires `hf-deepali`) |

## Installation of optional dependency

```bash
pip install torch hf-deepali   # needed by every entry point except Point_Registration
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
