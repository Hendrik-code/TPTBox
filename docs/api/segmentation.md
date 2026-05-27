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
