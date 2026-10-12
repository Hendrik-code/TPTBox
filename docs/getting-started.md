# Getting Started

## Installation

### From PyPI (recommended)

```bash
pip install TPTBox
```

### From source (development)

```bash
git clone https://github.com/Hendrik-code/TPTBox.git
cd TPTBox
pip install poetry
poetry install --with dev
```

### Optional dependencies

The core install stays light. The DICOM, segmentation and registration backends are guarded:
importing `TPTBox.core.dicom`, `TPTBox.segmentation` or `TPTBox.registration` always succeeds, and
only *calling* an entry point whose backend is missing raises an `ImportError` naming what to
install. (`antspyx` is the exception — its call sites still surface a plain `ModuleNotFoundError`.)

```bash
# DICOM -> NIfTI conversion (TPTBox.core.dicom)
pip install "TPTBox[dicom]"          # pydicom, dicom2nifti

# nnU-Net / VibeSeg inference (TPTBox.segmentation)
pip install "TPTBox[seg]"            # torch, nnunetv2, acvl_utils, batchgenerators

# Intensity-based and deformable registration (DeepALI)
pip install "TPTBox[reg]"            # torch, hf-deepali

# Several at once
pip install "TPTBox[dicom,seg,reg]"

# SPINEPS spine segmentation - no extra, see the note below
pip install spineps

# 3D mesh visualisation
pip install pyvista vtk

# N4 bias-field correction and some NII resampling helpers
pip install antspyx
```

!!! note "nnU-Net version"
    `TPTBox.segmentation.nnUnet_utils` is a self-contained fork of nnU-Net's inference code: it
    reads the checkpoint's own `plans.json` and builds its own `PlansManager`, so it is not tied
    to the plans layout of the installed nnU-Net. TPTBox therefore does not impose a version of
    its own — the `seg` extra mirrors what SPINEPS asks for: `nnunetv2>=2.8,<3.0` on Python 3.10+,
    and `nnunetv2==2.4.2` on Python 3.9, which is the last release supporting it. `TPTBox[seg]`
    and `spineps` can be installed side by side.

    If you also use `totalspineseg`, note that it requires `nnunetv2<=2.4.2`, which cannot be
    satisfied together with SPINEPS on Python 3.10+. That constraint comes from `totalspineseg`,
    not from TPTBox.

!!! note "SPINEPS has no extra"
    `spineps` depends on TPTBox itself, so a `TPTBox[spineps]` extra would be a circular
    dependency. Install it directly with `pip install spineps`.

## Core Concepts

### NII — NIfTI image wrapper

[`NII`][TPTBox.core.nii_wrapper.NII] wraps a nibabel `Nifti1Image` and adds convenient reorientation,
resampling, masking, and arithmetic operations. Set `seg=True` for integer-labelled segmentation
images to keep the smallest integer dtype.

```python
from TPTBox import NII

# Load from file
ct = NII.load("ct.nii.gz", seg=False)
seg = NII.load("seg.nii.gz", seg=True)

# Reorient to RAS
ct_ras = ct.reorient(("R", "A", "S"))

# Resample to 1 mm isotropic
ct_1mm = ct_ras.rescale((1.0, 1.0, 1.0))

# Apply a segmentation mask
masked = ct_1mm.apply_mask(seg)

# Save
ct_1mm.save("ct_1mm.nii.gz")
```

### POI — Points of Interest

[`POI`][TPTBox.core.poi.POI] maps `(vertebra_id, subregion_id) → 3D coordinate`.  Coordinates can be in
voxel or world (mm) space.  Use [`calc_centroids`][TPTBox.core.poi.calc_centroids] to compute centroids
from a segmentation.

```python
from TPTBox import NII, calc_centroids

seg = NII.load("seg.nii.gz", seg=True)
poi = calc_centroids(seg)
print(poi)
```

### BIDS dataset navigation

[`BIDS_Global_info`][TPTBox.core.bids_files.BIDS_Global_info] scans a dataset root and lets you filter
subjects, sessions, and modalities with a query interface.

```python
from TPTBox import BIDS_Global_info

bids = BIDS_Global_info(
    datasets=["path/to/dataset"],
    parents=["rawdata", "derivatives/spineps"],
)

for subject, container in bids.enumerate_subjects(sort=True):
    query = container.new_query(flatten=False)
    query.filter("format", "T2w")
    for family in query.loop_dict(key_addendum=["acq"]):
        t2w = family["T2w"][0].open_nii()
        ...
```

## Running Tests

```bash
pytest unit_tests/ -x -q
```

## Building the Documentation Locally

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve   # live-reload preview at http://127.0.0.1:8000
mkdocs build   # static build into site/
```
