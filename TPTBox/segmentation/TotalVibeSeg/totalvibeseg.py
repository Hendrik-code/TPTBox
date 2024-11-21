from pathlib import Path
from typing import Literal

from TPTBox import Image_Reference, to_nii
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_inference_on_file


def run_totalvibeseg(
    i: Image_Reference, out_seg: str | Path, override=False, gpu=0, ddevice: Literal["cpu", "cuda", "mps"] = "cuda", **args
):
    run_inference_on_file(80, [to_nii(i)], out_file=out_seg, override=override, gpu=gpu, ddevice=ddevice, **args)


def extract_vertebra_bodies_from_totalVibe(
    nii_total: Image_Reference,
    num_thoracic_verts: int = 12,
    num_lumbar_verts: int = 5,
    out_path: str | Path | None = None,
    out_path_poi: str | Path | None = None,
):
    """
    Extracts and labels vertebra bodies from a totalVibe segmentation NIfTI file.

    This function processes a segmentation mask containing vertebrae and intervertebral discs (IVDs).
    It separates individual vertebra bodies by eroding and splitting the mask at IVD regions, labels the vertebrae
    from bottom to top (lumbar and thoracic), and optionally saves the labeled mask and point-of-interest (POI) data.

    Args:
        nii_total (Image_Reference): Path or reference to the NIfTI file containing the totalVibe segmentation mask.
        num_thoracic_verts (int, optional): Number of thoracic vertebrae to include. Defaults to 12.
        num_lumbar_verts (int, optional): Number of lumbar vertebrae to include. Defaults to 5.
        out_path (str | Path | None, optional): Path to save the processed mask data. If None, no files are saved. Defaults to None.
        out_path_poi (str | Path | None, optional):  Path to save the processed POI data (ending json). If None, no files are saved. Defaults to None.

    Returns:
        tuple:
            - components (NII): A labeled NIfTI mask of the segmented vertebra bodies.
            - centroids_mapped (POI): Centroids of the labeled vertebrae as a point-of-interest (POI) dataset.

    Notes:
        - Labels for the vertebrae follow the naming convention: L1=20 to L5=24 for lumbar and T1=8 to T12=19 for thoracic; T13 = 28.
        - Cervical vertebrae and any unclassified regions are excluded (set to 0).
        - The output files, if saved, will include the mask and POI data:
          - Mask file: `<out_path>`
          - POI file: `<out_path>` with `_poi.json` suffix recommended.
    Example:
        >>> nii_total = "/path/to/vibe_segmentation.nii.gz"
        >>> labeled_mask, centroids = extract_vertebra_bodies_from_totalVibe(nii_total, out_path="output_mask.nii.gz")
    """
    from TPTBox import Vertebra_Instance, calc_centroids

    # Load the totalVibe segmentation
    nii = to_nii(nii_total, seg=True)
    vertebrae = nii.extract_label(69)
    ivds = nii.extract_label(68)

    # Erode vertebra masks and split them by IVDs
    split_masks = vertebrae.erode_msk(1, connectivity=3, verbose=False)
    split_masks[ivds.dilate_msk(1, connectivity=1, verbose=False) == 1] = 0

    # Get connected components and clean them
    vert_bodys = split_masks.get_connected_components()
    vert_bodys.dilate_msk_(3, verbose=False)
    vert_bodys[vertebrae != 1] = 0

    # Calculate centroids for vertebra bodies
    centroids_unsorted = calc_centroids(vert_bodys, second_stage=50)
    centroids_unsorted_srp = centroids_unsorted.reorient(("S", "R", "P"))
    centroids_sorted = dict(
        sorted({i: centroids_unsorted_srp[i, 50][0] for i in centroids_unsorted_srp.keys_region()}.items(), key=lambda x: x[1])
    )

    # Map centroids to labels based on thoracic and lumbar vertebra counts
    def map_to_label(index):
        if index >= num_thoracic_verts + num_lumbar_verts:
            return 0  # Remove cervical vertebrae
        if index < num_lumbar_verts:
            return Vertebra_Instance.name2idx()[f"L{num_lumbar_verts - index}"]
        return Vertebra_Instance.name2idx()[f"T{num_thoracic_verts - (index - num_lumbar_verts)}"]

    label_mapping = {k: map_to_label(i) for i, k in enumerate(centroids_sorted)}
    vert_bodys.map_labels_(label_mapping, verbose=False)
    centroids = centroids_unsorted.map_labels(label_map_region=label_mapping)
    # Save outputs if an output path is specified
    if out_path:
        vert_bodys.save(out_path)

    if out_path_poi:
        centroids.save(out_path_poi)

    return vert_bodys, centroids


if __name__ == "__main__":
    from TPTBox import BIDS_FILE
    from TPTBox.segmentation import run_totalvibeseg

    # run_totalvibeseg
    # You can also use a string/Path if you want to set the path yourself.
    dataset = "/media/data/robert/datasets/dicom_example/dataset-VR-DICOM2/"
    in_file = BIDS_FILE(
        f"{dataset}/derivative_stiched/sub-111168222/T2w/sub-111168222_sequ-301-stiched_acq-ax_part-water_T2w.nii.gz", dataset
    )
    out_file = in_file.get_changed_path(
        "nii.gz", "msk", parent="derivative", info={"seg": "TotalVibeSegmentator", "mod": in_file.bids_format}
    )
    run_totalvibeseg(in_file, out_file)
