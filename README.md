[![PyPI version tptbox](https://badge.fury.io/py/tptbox.svg)](https://pypi.python.org/pypi/tptbox/)

# Torso Processing ToolBox (TPTBox)

This is a multi-functional package to handle any sort of bids-conform dataset (CT, MRI, ...)
It can find, filter, search any BIDS_Family and subjects, and has many functionalities, among them:
- Easily loop over datasets, and the required files
- Read, Write Niftys, centroid jsons, ...
- Reorient, Resample, Shift Niftys, Centroids, labels
- Modular 2D snapshot generation (different views, MIPs, ...)
- 3D Mesh generation from segmentation and snapshots from them
- Running the Anduin docker smartly
- Registration
- Logging everything consistently
- ...

## Install the package
```bash
conda create -n 3.10 python=3.10 
conda activate 3.10
pip install TPTBox
```
### Install via github:
(you should be in the project folder)
```bash
pip install poetry
poetry install
```
or:
Develop mode is really, really nice:
```bash
pip install poetry
poetry install --with dev 
```

## Functionalities

Each folder in this package represents a different functionality.

The top-level-hierarchy incorporates the most important files, the BIDS_files.

### BIDS_Files

This file builds a data model out of the BIDS file names.
It can load a dataset as a BIDS_Global_info file, from which search queries and loops over the dataset can be started.
See ```tutorial_BIDS_files.ipynb``` for details.

### bids_constants
Defines constants for the BIDS nomenclature (sequence-splitting keys, naming conventions...)

### vert_constants

Contains definitions and sort order for our intern labels, for vertebrae, POI, ...

### Rotation and Resampling

Example rotate and resample.

```python

from TPTBox import NII

nii = NII.load("...path/xyz.nii.gz", seg=True)
img_rot = nii.reorient(axcodes_to=("P", "I", "R"))
img_scale = nii.rescale((1.5, 5, 1))  # in mm as currently rotated
# resample to an other image
img_resampled_to_other = nii.resample_from_to(img_scale)

from TPTBox import NII

nii = NII.load("...path/xyz.nii.gz", seg=True)
# R right, L left
# S superior/up, I inferior/down
# A anterior/front, P posterior/back
img_rot = nii.reorient(axcodes_to=("P", "I", "R"))
img_scale = nii.rescale((1.5, 5, 1))  # in mm as currently rotated
# resample to an other image
img_resampled_to_other = nii.resample_from_to(img_scale)

nii.get_array()  # get numpy array
nii.affine  # Affine matrix
nii.header  # NIFTY header
nii.orientation  # Orientation in 3-Letters
nii.zoom # Scale of the three image axis
nii.shape #shape
```

### Snapshot2D Spine

The snapshot function automatically generates sag, cor, axial cuts in the center of a segmentation.

```python
from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

ct = Path("Path to CT")
mri = Path("Path to MRI")
vert = Path("Path to Vertebra segmentation")
subreg = Path("Path to Vertebra subregions")
cdt = (vert, subreg, [50])  # 50 is subregion of the vertebra body
# cdt can be also loaded as a json. See definition Centroid_DictList in nii_utils

ct_frame = Snapshot_Frame(image=ct, segmentation=vert, centroids=cdt, mode="CT", coronal=True, axial=True)
mr_frame = Snapshot_Frame(image=mri, segmentation=vert, centroids=None, mode="MRI", coronal=True, axial=True)
create_snapshot(snp_path="snapshot.jpg", frames=[ct_frame, mr_frame])
```


### Snapshot3D

```python
TBD
```

### Docker

```python
TBD
```

### Logger

```python
TBD
```