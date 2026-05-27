# Spine Statistics (`spine/spinestats`)

Clinical spine measurements computed from `POI` objects and `NII` segmentations.

## Modules

| Module | Description |
|---|---|
| `distances.py` | Distances between anatomical landmarks (IVD height, canal diameter, …) |
| `angles.py` | Cobb angle and other spine curvature measurements |
| `ivd_pois.py` | Intervertebral disc (IVD) Point of Interest computation |
| `body_quadrants.py` | Subdivides vertebra bodies into anatomical quadrants |
| `make_endplate.py` | Generates superior/inferior endplate surfaces from segmentations |

## Key functions

| Function | Module | Description |
|---|---|---|
| `compute_cobb_angle(poi)` | `angles.py` | Cobb angle between vertebra pairs from POI coordinates |
| `compute_ivd_height(poi)` | `distances.py` | Inter-vertebral disc height at a given level |
| `calc_ivd_pois(vert_nii, subreg_nii)` | `ivd_pois.py` | Compute IVD POIs from vertebra + subregion segmentations |
| `make_endplate(nii, poi, ...)` | `make_endplate.py` | Fit an endplate surface to a vertebra body |

## Coordinate convention

All measurement functions consume `POI` objects (voxel or world space) produced by
`calc_centroids` or `calc_poi_from_subreg_vert` from the `core` module.
