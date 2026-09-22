# Segmentation

Integration with external segmentation pipelines: SPINEPS (spine segmentation) and
VibeSeg / nnU-Net (general deep learning inference).

## SPINEPS

::: TPTBox.segmentation.spineps
    options:
      show_source: true
      filters: ["!^_"]

## VibeSeg

::: TPTBox.segmentation.VibeSeg.vibeseg
    options:
      show_source: true
      filters: ["!^_"]

## nnU-Net Utilities

::: TPTBox.segmentation.nnUnet_utils.inference_api
    options:
      show_source: true
      filters: ["!^_"]

## Rib assignment

Adds left/right rib labels to an existing vertebra + spine segmentation.
When no raw rib mask is supplied, VibeSeg dataset 12 is invoked on the
source CT.

::: TPTBox.segmentation.rib.add_ribs
    options:
      show_source: true
      filters: ["!^_"]
