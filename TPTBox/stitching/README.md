# Stitching (`TPTBox.stitching`)

Merges multiple NIfTI images that are already aligned in global space into a single volume.
Useful for whole-body or long-spine multi-station acquisitions.
You can verify alignment by opening the images in ITKSnap with "open additional image."

## API

| Function | Description |
|---|---|
| `stitching(inputs, out, ...)` | High-level wrapper. `inputs` accepts any mix of `BIDS_FILE`, `NII`, `str`, or `Path`. Resolves the output path from a `BIDS_FILE` when one is passed. Returns `(result_nii, ramp_nii)` as `NII` objects. |
| `stitching_raw(images, out, ...)` | Low-level driver. `images` accepts file paths, pre-loaded `NII` objects, or (fallback) `Nifti1Image` objects. Returns `(result_nii, ramp_nii)` as `NII` objects. |
| `NAKO_stitch_T2w(HWS, BWS, LWS, n4_after_stitch=False)` | Stitch the three NAKO sagittal T2w spine stations (HWS cervical, BWS thoracic, LWS lumbar) into one volume |

![Example of a stitching](https://raw.githubusercontent.com/Hendrik-code/TPTBox/main/TPTBox/stitching/stitching.jpg "Example of a stitching")


### Standalone
This script can be run directly from the console. Copy 'stitching.py' and install the necessary package.

```
stitching.py
[-h] print the help message
[-i IMAGES [IMAGES ...]] a list of input image paths
[-o OUTPUT] The output image path
[-v] verbose - if set, there will be more printouts.
[-min_value MIN_VALUE] Background fill used when resampling each chunk, and — when set explicitly — a hard floor applied to the stitched output. Pass 0 for MRI magnitude, -1024 for CT. Omitting it (the Python-API default `None`) uses 0 as the internal background and only applies a hard floor when the output dtype cannot represent negatives (unsigned integer types) — signed / float outputs then keep legitimate negatives (Philips-scaled fat-fraction, phase, B0 offsets).
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
from TPTBox.stitching import stitching

list_of_files = [
    "File_A.nii.gz",
    "File_B.nii.gz",
    "File_C.nii.gz",
]

# Call the stitching function
# This will combine your images into a single NIfTI file
stitching(
    list_of_files,  # BIDS_FILE / NII / str / Path (any mix)
    out="out_path_stitched_image.nii.gz",  # Path or BIDS_FILE for the stitched output
    is_seg=False,  # Set True for segmentation masks (forces integer dtype, nearest-neighbour resample)
    is_ct=False,  # Sets min_value to -1024 (CT air) when min_value is not passed explicitly
    kick_out_fully_integrated_images=True,
    dtype=float,  # Output dtype; "auto" picks the smallest lossless type from the inputs (float32 when any input has a non-trivial scl_slope/inter)
    match_histogram=False,  # Match intensity histograms across images
    store_ramp=False,  # Store blending ramp (optional)
    min_value=None,  # Explicit background/floor. None (default) = clip only when the output dtype is unsigned integer (protects against negative-to-huge wraparound) and let signed / float outputs keep legitimate negatives. Pass 0 to floor magnitude MR at 0 so cubic-spline resample ringing doesn't leak small negatives into what should be a non-negative volume; pass -1024 for CT.
)
```


### Cite
```
Graf, R., Platzek, PS., Riedel, E.O. et al. Generating synthetic high-resolution spinal STIR and T1w images from T2w FSE and low-resolution axial Dixon. Eur Radiol (2024). https://doi.org/10.1007/s00330-024-11047-1

```

```
@article{graf2024generating,
  title={Generating synthetic high-resolution spinal STIR and T1w images from T2w FSE and low-resolution axial Dixon},
  author={Graf, Robert and Platzek, Paul-S{\"o}ren and Riedel, Evamaria Olga and Kim, Su Hwan and Lenhart, Nicolas and Ramsch{\"u}tz, Constanze and Paprottka, Karolin Johanna and Kertels, Olivia Ruriko and M{\"o}ller, Hendrik Kristian and Atad, Matan and others},
  journal={European Radiology},
  pages={1--11},
  year={2024},
  publisher={Springer}
}

```
