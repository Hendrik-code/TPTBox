# Torso Processing ToolBox (TPTBox) - Stitching

This tool can merge multiple Nifti images if they are already aligned in global space. You can check this by opening them in ITKSnap with "open additional image."

![Example of a stitching](stitching.jpg?raw=true "Example of a stitching")


### Standalone
This script can be run directly from the console. Copy 'stiching.py' and install the necessary package.

```
stitching.py
[-h] print the help message
[-i IMAGES [IMAGES ...]] a list of input image paths
[-o OUTPUT] The output image path
[-v] verbose - if set, there will be more printouts.
[-min_value MIN_VALUE] New pixels not present will get this value. Recommended 0 for MRI and for CT -1024 or the known min-value.
[-seg] This flag is required if you merge segmentation Niftis.
Switches:
[-no_bias] If set: Do not use n4_bias_field_correction. It speeds up the process, but n4_bias_field_correction helps in roughly aligning the histogram.
[-bias_crop] crop empty spaces by the bias field mask.
[-crop] crop empty space away
[-sr] Store the ramp and stitching of the images in a 4d nii.gz
Optional:
[-hists] Use histogram matching to put the images in the roughly same histogram. The previous image is used when hist_n is not set.
[-hist_n HISTOGRAM_NAME] path to an image that should be used for histogram matching
[-ramp_e RAMP_EDGE_MIN_VALUE] The ramp is only considering values above this minimum value
[-ms MIN_SPACING] Set the minimum Spacing (in mm)
[-dtype DTYPE] Force a dtype
```

Example:

Given the image a.nii.gz,b.nii.gz,c.nii.gz and the segmentations a_msk.nii.gz,b_msk.nii.gz,c_msk.nii.gz. The images can be merged with:

```bash
stitching.py  -i a.nii.gz b.nii.gz c.nii.gz -o out.nii.gz
stitching.py  -i a_msk.nii.gz b_msk.nii.gz c_msk.nii.gz -o out_msk.nii.gz -seg
```

### Install as a package

Install on Python 3.10 or higher
```bash
pip install TPTBox
```

```python
from TPTBox import NII
from TPTBox.stitching import stitching
out_nii,_ = stitching([NII.load("a.nii.gz",seg=False), NII.load("b.nii.gz",seg=False), NII.load("c.nii.gz",seg=False)], out="out.nii.gz")

```

or


```python
from TPTBox.stitching import stitching_raw
stitching_raw(["a.nii.gz", "b.nii.gz", "c.nii.gz"], "out.nii.gz", is_segmentation=False)
```
