# TPTBox Core

The `core` subpackage is the foundation of TPTBox. It provides the three primary abstractions —
`NII`, `POI`, and `BIDS_FILE` — along with helper utilities for array operations and anatomical constants.

## Key Classes and Functions

### `nii_wrapper.py` — NIfTI image wrapper

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

### `bids_files.py` — BIDS dataset navigation

| Symbol | Description |
|---|---|
| `BIDS_Global_info` | Scans a dataset root and indexes all BIDS files |
| `BIDS_Global_info.enumerate_subjects()` | Iterate over subjects as `(subject_id, Subject_Container)` |
| `Subject_Container` | Per-subject file index; entry point for queries |
| `Subject_Container.new_query()` | Returns a `Searchquery` for this subject |
| `BIDS_FILE` | One file parsed into BIDS entities (sub, ses, format, …) |
| `BIDS_FILE.open_nii()` | Load this file's NIfTI |
| `BIDS_FILE.get_changed_path(...)` | Derive a new path with changed BIDS entities |
| `Searchquery` | Fluent query builder: `.filter()`, `.loop_dict()`, `.first()` |
| `BIDS_Family` | `dict[str, list[BIDS_FILE]]` grouping files by format |

### `poi.py` — Points of Interest

| Symbol | Description |
|---|---|
| `POI` | Maps `(vertebra_id, subregion_id) → (x, y, z)` |
| `calc_centroids(seg_nii)` | Compute centroids for every label in a segmentation |
| `calc_poi_from_subreg_vert(vert, subreg)` | Compute POIs from paired vertebra + subregion segmentations |
| `POI.save(path)` | Serialise to JSON |
| `POI.load(path)` | Deserialise from JSON |
| `POI.to_global(ref)` | Convert from voxel to world (mm) coordinates |
| `POI.to_local(ref)` | Convert from world to voxel coordinates |

### `np_utils.py` — NumPy utilities

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

## Quick Example

```python
from TPTBox import NII, BIDS_Global_info, calc_centroids

# Load and resample a CT
ct = NII.load("sub-001_ct.nii.gz", seg=False)
ct_ras = ct.reorient(("R", "A", "S")).rescale((1.0, 1.0, 1.0))

# Compute centroids from a segmentation
seg = NII.load("sub-001_seg.nii.gz", seg=True)
poi = calc_centroids(seg)
print(poi)

# Scan a BIDS dataset
bids = BIDS_Global_info(["dataset/"], parents=["rawdata"])
for subj, container in bids.enumerate_subjects():
    t2 = container.new_query().filter("format", "T2w").first()
```
