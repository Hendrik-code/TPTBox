# TPTBox

**The Torso Processing Toolbox (TPTBox)** is a multi-functional Python package for processing BIDS-compliant medical imaging datasets (CT, MRI, and more).

[![PyPI version](https://badge.fury.io/py/tptbox.svg)](https://pypi.python.org/pypi/tptbox/)
[![Python Versions](https://img.shields.io/pypi/pyversions/tptbox)](https://pypi.org/project/tptbox/)
[![tests](https://github.com/Hendrik-code/TPTBox/actions/workflows/tests.yml/badge.svg)](https://github.com/Hendrik-code/TPTBox/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Hendrik-code/TPTBox/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/Hendrik-code/TPTBox)

## Overview

TPTBox provides a unified interface for the most common tasks in medical image processing research:

- **BIDS dataset navigation** — load, filter, and iterate over BIDS-compliant datasets
- **NIfTI I/O** — read and write NIfTI files with consistent reorientation and resampling
- **Points of Interest (POI)** — compute and manipulate anatomical landmarks on vertebrae
- **2D snapshots** — modular, multi-view MIP and overlay image generation
- **3D mesh generation** — surface meshes from segmentations with configurable rendering
- **Registration** — rigid point registration via SimpleITK, plus intensity-based and deformable registration via DeepALI (optional `hf-deepali`)
- **Segmentation** — integration with SPINEPS and nnU-Net inference pipelines
- **Image stitching** — multi-station field-of-view stitching
- **Logging** — structured, consistent logging across long-running pipelines

## Quick Example

```python
from TPTBox import NII, BIDS_Global_info

# Load a NIfTI and reorient + resample
nii = NII.load("path/to/image.nii.gz", seg=False)
nii_ras = nii.reorient(("R", "A", "S"))
nii_1mm = nii_ras.rescale((1.0, 1.0, 1.0))

# Iterate over a BIDS dataset
bids = BIDS_Global_info(["path/to/dataset"], parents=["rawdata"])
for subject, container in bids.enumerate_subjects():
    query = container.new_query(flatten=True)
    query.filter_format("T2w")
    query.filter_filetype("nii.gz")
    for t2w in query.loop_list():
        nii = t2w.open_nii()
```

See [Getting Started](getting-started.md) for installation instructions and more examples.
