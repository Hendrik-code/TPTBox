<h1 align="center">
<img src="TPTBox/images/TPTBox_overview.png" width="800">
</h1><br>


[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.103055-B31B1B)](https://doi.org/10.1016/j.softx.2026.103055)
[![Stable Version](https://img.shields.io/pypi/v/tptbox?label=stable)](https://pypi.python.org/pypi/tptbox/)
[![Python Versions](https://img.shields.io/pypi/pyversions/tptbox)](https://pypi.org/project/tptbox/)
[![Downloads](https://img.shields.io/pepy/dt/tptbox?label=downloads)](https://pepy.tech/project/tptbox)
[![tests](https://github.com/Hendrik-code/TPTBox/actions/workflows/tests.yml/badge.svg)](https://github.com/Hendrik-code/TPTBox/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Hendrik-code/TPTBox/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/Hendrik-code/TPTBox)
[![Documentation](https://readthedocs.org/projects/tptbox/badge/?version=latest)](https://tptbox.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <a href="#quick-use">Quick use</a> ·
  <a href="https://tptbox.readthedocs.io">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>


The Torso Processing ToolBox (TPTBox) is a multi-functional package to handle any sort of bids-conform dataset (CT, MRI, ...)

## Publication

This is the official repository of "TPTBox: Extensive torso processing toolbox for simple and automatic analysis of CT and MR imaging"

If you use this toolbox, please cite our publication: https://doi.org/10.1016/j.softx.2026.103055
```
@article{MollerGraf2026TPTBox,
  title = {TPTBox: Extensive torso processing toolbox for simple and automatic analysis of CT and MR imaging},
  journal = {SoftwareX},
  pages = {103055},
  year = {2026},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2026.103055},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711026005467},
  author = {Hendrik Möller and Robert Graf and Tanja Lerchl and Daniel Rueckert and Jan S. Kirschke},
}
```

## Features

- **Dataset Handling**: Loop over datasets, search query, and find images and their corresponding derivatives


- **I/O Handling**: Read and Write nifti files and point coordinate files as JSONs
- **Image Processing**: Reorient, Resample, Shift Niftys, Centroids, labels, compute connected components, and so much more
- **Visualization**: Modular 2D snapshot generation (different views, maximum intensity projections, depth-color map)
- **3D Mesh generation**: Use 3D segmentations to create 3D meshes and then take snapshots for visualization
- **Registration**: Register two images to each other, using available data (image, segmentation, points)
- **Stitching**: You have multiple MRI of the same person, split into different regions? Use our stitching algorithm to create one unified view.
- **Logger**: Log every function in a file automatically, color important messages in the terminal for easy recognition.


## Install the package
```bash
conda create -n 3.10 python=3.10
conda activate 3.10
pip install TPTBox
# Optional dependency Registration (deepali backend)
pip install "TPTBox[reg]"
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


### Quick Use:
```python
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
nii.zoom  # Scale of the three image axis
nii.shape  # shape
```


## Documentation

Full API reference and usage guides are available at **https://tptbox.readthedocs.io**.

The docs cover all sub-packages — `NII`, `POI`, `BIDS_FILE`, NumPy utilities,
vertebra constants, spine analysis, registration, segmentation, mesh3D,
stitching, and the logger — with hyperlinks back to the GitHub source.

## The three pillars

### <a href=TPTBox/core/README_NII.md>NII: nii_wrapper.py -- NIfTI image wrapper </a>
This is the core of image handling, this takes care of loading images and segmentations, and any data processing

### <a href=TPTBox/core/README_POI.md>POI: poi.py -- Points of Interests </a>
This is the core of handling 2D/3D coordinates in any defined space. Center of mass locations can be computed in this format, and other landmarks. Similar to Niftis, this contains an affine matrix so it is aware of its global space relation, voxel spacing, ...

### <a href=TPTBox/core/README_BIDS.md>BIDS: bids_files.py -- Dataset Handling </a>
This is the core of handling datasets that are BIDS-compliant. Easily search through your datasets and find all images following your constraints, such as every CT that also has a specific segmentation available.

## Modules

Each sub-package has its own README with API tables and examples. Click on the name to get the corresponding README with quick examples and more explanations.


| Module | Description |
|---|---|
| [`core`](https://tptbox.readthedocs.io/en/latest/modules/core/) | `NII` (NIfTI I/O and transforms), `POI` (anatomical landmarks), BIDS dataset navigation, NumPy utilities, vertebra constants |
| [`core/poi_fun`](https://tptbox.readthedocs.io/en/latest/modules/poi_fun/) | Internal POI computation strategies (surface points, corpus centers, disc points) |
| [`spine`](https://tptbox.readthedocs.io/en/latest/modules/spine/) | Spine-specific tools: 2D snapshot generation and statistical measurements |
| [`spine/snapshot2D`](https://tptbox.readthedocs.io/en/latest/modules/snapshot2d/) | Modular 2D image generation — axial/sagittal/coronal slices, MIPs, segmentation overlays |
| [`spine/spinestats`](https://tptbox.readthedocs.io/en/latest/modules/spinestats/) | Clinical spine measurements: distances, angles, disc heights, IVD landmarks |
| [`registration`](https://tptbox.readthedocs.io/en/latest/modules/registration/) | Rigid and deformable image registration via ANTs and DeepALI |
| [`segmentation`](https://tptbox.readthedocs.io/en/latest/modules/segmentation/) | Integration with SPINEPS, VibeSeg/TotalVibeSeg, and nnU-Net pipelines |
| [`mesh3D`](https://tptbox.readthedocs.io/en/latest/modules/mesh3d/) | 3D surface mesh generation and rendering from segmentation volumes |
| [`stitching`](https://tptbox.readthedocs.io/en/latest/modules/stitching/) | Multi-station NIfTI stitching for whole-body or long-spine acquisitions |
| [`logger`](https://tptbox.readthedocs.io/en/latest/modules/logger/) | Structured, consistent logging for medical image processing pipelines |
