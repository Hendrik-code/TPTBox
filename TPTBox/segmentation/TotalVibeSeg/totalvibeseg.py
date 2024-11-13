from pathlib import Path

from TPTBox import Image_Reference, to_nii
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_inference_on_file


def run_totalvibeseg(i: Image_Reference, out_seg: str | Path, override=False, **args):
    run_inference_on_file(80, [to_nii(i)], out_file=out_seg, override=override, **args)


if __name__ == "__main__":
    from TPTBox import BIDS_FILE
    from TPTBox.segmentation import run_totalvibeseg

    # run_totalvibeseg
    # You can alos use a string/Path if you want to set the path yourself.
    dataset = "/media/data/robert/datasets/dicom_example/dataset-VR-DICOM2/"
    in_file = BIDS_FILE(
        f"{dataset}/derivative_stiched/sub-111168222/T2w/sub-111168222_sequ-301-stiched_acq-ax_part-water_T2w.nii.gz", dataset
    )
    out_file = in_file.get_changed_path(
        "nii.gz", "msk", parent="derivative", info={"seg": "TotalVibeSegmentator", "mod": in_file.bids_format}
    )
    run_totalvibeseg(in_file, out_file)
