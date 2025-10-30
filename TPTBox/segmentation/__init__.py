from __future__ import annotations

from TPTBox.segmentation.spineps import run_spineps_all, run_spineps_single
from TPTBox.segmentation.VibeSeg.vibeseg import extract_vertebra_bodies_from_VibeSeg, run_inference_on_file, run_nnunet, run_vibeseg
from TPTBox.segmentation.VibeSeg.vibeseg import run_vibeseg as run_totalvibeseg  # TODO deprecate
