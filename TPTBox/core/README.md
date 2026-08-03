# TPTBox Core

The `core` subpackage is the foundation of TPTBox. It provides the three primary abstractions —
`NII`, `POI`, and `BIDS_FILE` — along with helper utilities for array operations and anatomical constants.


## The three pillars -- NII, POI, BIDS

### <a href=TPTBox/core/README_NII.md>NII: nii_wrapper.py -- NIfTI image wrapper </a>
This is the core of image handling, this takes care of loading images and segmentations, and any data processing

### <a href=TPTBox/core/README_POI.md>POI: poi.py -- Points of Interests </a>
This is the core of handling 2D/3D coordinates in any defined space. Center of mass locations can be computed in this format, and other landmarks. Similar to Niftis, this contains an affine matrix so it is aware of its global space relation, voxel spacing, ...

### <a href=TPTBox/core/README_BIDS.md>BIDS: bids_file.py -- Dataset Handling </a>
This is the core of handling datasets that are BIDS-compliant. Easily search through your datasets and find all images following your constraints, such as every CT that also has a specific segmentation available.


## Other Key Classes and Functions

### `np_utils.py` — NumPy utilities

Numpy functionalities that a lot f NII functions above utilize under the hood. Most of them are optimized to run on uint numpy arrays.

| Symbol | Description |
|---|---|
| `np_extract_label(arr, label)` | Extract a single label as a binary mask |
| `np_center_of_mass(arr)` | Per-label centre-of-mass |
| `np_volume(arr)` | Per-label voxel count |
| `np_bbox_binary(mask)` | Bounding-box slice tuple for a binary array |
| `np_dilate_msk(arr, mm, zoom)` | Morphological dilation by `mm` millimetres |
| `np_erode_msk(arr, mm, zoom)` | Morphological erosion |
| `np_fill_holes(arr)` | Fill holes per label |
| `np_connected_components(arr)` | Label connected components |
| `np_map_labels(arr, label_map)` | Remap label integers via a dict |
| `np_unique(arr)` | Unique values (faster than `np.unique` for uint arrays) |

```python
from TPTBox.core.np_utils import np_unique, np_center_of_mass

a = np.array([0,1,2,3], [4,5,6,7], dtype=np.uint8)

label = np_unique(a)
center_of_mass_of_label_four = np_center_of_mass(a)[4]
```

### `vert_constants.py` — Anatomical constants

| Symbol | Description |
|---|---|
| `Location` | `IntEnum` of anatomical subregion IDs (used as POI keys) |
| `Vertebra_Instance` | Maps integer IDs → anatomical names (C1–S1) |
| `v_name2idx` | Dict: `"L1" → 20`, etc. |
| `v_idx2name` | Dict: `20 → "L1"`, etc. |
| `v_idx_order` | Canonical sort order for vertebra IDs |
| `ZOOMS` | Type alias: `tuple[float, float, float]` |
| `AX_CODES` | Type alias: `tuple[str, str, str]` |
| `AFFINE` | Type alias: `np.ndarray` (4×4) |

```python
from TPTBox import NII, Location
# Segmentation
seg = NII.load("path/to/seg.nii.gz", seg=True)

# Get the segmentation of the Vertebra Corpus
seg_corpus = seg.extract_label(Location.Vertebra_Corpus)
```
