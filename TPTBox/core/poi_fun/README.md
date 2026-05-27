# POI Strategies (`core/poi_fun`)

Internal subpackage implementing the different strategies for computing Points of Interest (POIs)
from NIfTI segmentation volumes.  End users typically call the high-level helpers in `poi.py` and
`vert_constants.py`; the modules here provide the underlying algorithms.

## Modules

| Module | Description |
|---|---|
| `ray_casting.py` | Casts rays through a volume to find surface-intersection points |
| `vertebra_direction.py` | Derives superior/inferior/anterior/posterior unit vectors per vertebra |
| `vertebra_pois_non_centroids.py` | Computes non-centroid anatomical landmarks (process tips, endplates) |
| `pixel_based_point_finder.py` | Locates extreme/boundary pixels within a label for POI placement |
| `strategies.py` | Pluggable strategy objects for POI computation |
| `save_load.py` | JSON serialisation/deserialisation for `POI` objects |
| `save_mkr.py` | Export POIs as 3D Slicer markup files (`.fcsv`, `.mrk.json`) |
| `poi_abstract.py` | Abstract base classes shared across POI modules |
| `poi_global.py` | `POI_Global` — POI container in world-coordinate (mm) space |
| `_help.py` | Internal helpers (not part of the public API) |

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `POI_Global` | `poi_global.py` | World-space POI container; used after `POI.to_global()` |
| `save_poi` | `save_load.py` | Serialise a `POI` to a JSON file |
| `load_poi` | `save_load.py` | Deserialise a `POI` from a JSON file |
| `save_poi_as_slicer_markup` | `save_mkr.py` | Write POIs as a 3D Slicer `.mrk.json` markup file |

## Coordinate system

POIs inside this package use the same convention as `poi.py`:
- **Local** (`is_global() == False`): voxel indices `(i, j, k)` into the reference NIfTI
- **Global** (`is_global() == True`): world coordinates in mm, consistent with the NIfTI affine
