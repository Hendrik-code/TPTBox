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
#### Make venv:
```
conda create -n 3.10 python=3.10 
conda activate 3.10
```
### One of the following:
(you should be in the project folder)
```
pip install -e ./
```
or:

Develop mode is really, really nice:
```
python setup.py develop
sudo python3 setup.py develop
```

If "python3" don't know where to install, use
```
which python
sudo <result from which python> setup.py develop
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
# R right, L left .. {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
img_rot = reorient_to(img, axcodes_to=("P", "I", "R")) 
img_rot_iso = resample_nib(img_rot, voxel_spacing=(1, 1, 1), order=3, c_val=0)
```

### Snapshot2D

The snapshot function automatically generates sag, cor, axial cuts in the center of a segmentation.

```python
from pathlib import Path
from BIDS.wrapper.snapshot_mr_fun2 import Snapshot_Frame,create_snapshot
ct = Path('Path to CT')
mri = Path('Path to MRI')
vert = Path('Path to Vertebra segmentation')
subreg = Path('Path to Vertebra subregions')
cdt = (vert,subreg,[50]) # 50 is subregion of the vertebra body
# cdt can be also loaded as a json. See definition Centroid_DictList in nii_utils

ct_frame = Snapshot_Frame(image=ct, segmentation=vert, centroids=cdt, mode="CT", coronal=True, axial=True)
mr_frame = Snapshot_Frame(image=mri, segmentation=vert, centroids=None, mode="MRI", coronal=True, axial=True)
create_snapshot(snp_path='snapshot.jpg',frames=[ct_frame, mr_frame])
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