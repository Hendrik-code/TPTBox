# Stitching

Combine multiple NIfTI volumes with overlapping field-of-views into a single
volume — typical for whole-body or long-spine multi-station acquisitions where
each station is stored as its own NIfTI file.

The pipeline resamples every input into a common bounding box, optionally
applies N4 bias-field correction and histogram matching, and blends
overlapping regions using distance-transform-based weight ramps. Both intensity
and segmentation stitching are supported; for segmentations the blending
degenerates to a majority-style selection so labels remain integer-valued.

Use [`stitching`][TPTBox.stitching.stitching_tools.stitching] as the high-level
entry point (accepts `BIDS_FILE`, `NII`, `str`, or `Path` inputs), or the
lower-level [`stitching_raw`][TPTBox.stitching.stitching.main] when you already
have `Nifti1Image` objects.

## Stitching

::: TPTBox.stitching.stitching
    options:
      show_source: true
      filters: ["!^_"]

## Stitching Tools

::: TPTBox.stitching.stitching_tools
    options:
      show_source: true
      filters: ["!^_"]
