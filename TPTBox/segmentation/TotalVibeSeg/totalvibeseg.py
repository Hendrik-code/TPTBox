from pathlib import Path

from TPTBox import Image_Reference, to_nii
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_inference_on_file


def run_totalvibeseg(i: Image_Reference, out_seg: str | Path):
    run_inference_on_file(80, [to_nii(i)], out_file=out_seg)
