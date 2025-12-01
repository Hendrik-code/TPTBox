from __future__ import annotations

from pathlib import Path
from typing import Literal

from TPTBox import Image_Reference, to_nii
from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

VibeSeg_map = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "intestine",
    19: "duodenum",
    20: "unused/rib (ct only)",
    21: "urinary_bladder",
    22: "prostate",
    23: "sacrum",
    24: "heart",
    25: "aorta",
    26: "pulmonary_vein",
    27: "brachiocephalic_trunk",
    28: "subclavian_artery_right",
    29: "subclavian_artery_left",
    30: "common_carotid_artery_right",
    31: "common_carotid_artery_left",
    32: "brachiocephalic_vein_left",
    33: "brachiocephalic_vein_right",
    34: "atrial_appendage_left",
    35: "superior_vena_cava",
    36: "inferior_vena_cava",
    37: "portal_vein_and_splenic_vein",
    38: "iliac_artery_left",
    39: "iliac_artery_right",
    40: "iliac_vena_left",
    41: "iliac_vena_right",
    42: "humerus_left",
    43: "humerus_right",
    44: "scapula_left",
    45: "scapula_right",
    46: "clavicula_left",
    47: "clavicula_right",
    48: "femur_left",
    49: "femur_right",
    50: "hip_left",
    51: "hip_right",
    52: "spinal_cord",
    53: "gluteus_maximus_left",
    54: "gluteus_maximus_right",
    55: "gluteus_medius_left",
    56: "gluteus_medius_right",
    57: "gluteus_minimus_left",
    58: "gluteus_minimus_right",
    59: "autochthon_left",
    60: "autochthon_right",
    61: "iliopsoas_left",
    62: "iliopsoas_right",
    63: "sternum",
    64: "costal_cartilages",
    65: "subcutaneous_fat",
    66: "muscle",
    67: "inner_fat",
    68: "IVD",
    69: "vertebra_body",
    70: "vertebra_posterior_elements",
    71: "spinal_channel",
    72: "bone_other",
}


def run_vibeseg(
    i: Image_Reference,
    out_seg: str | Path,
    override=False,
    gpu=0,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    dataset_id=100,
    padd=0,
    keep_size=False,  # Keep size of the model Segmentation
    **args,
):
    return run_inference_on_file(
        dataset_id,
        [to_nii(i)],
        out_file=out_seg,
        override=override,
        gpu=gpu,
        ddevice=ddevice,
        padd=padd,
        keep_size=keep_size,
        **args,
    )[0]


def run_nnunet(
    i: list[Image_Reference],
    out_seg: str | Path,
    override=False,
    gpu=0,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    dataset_id=80,
    **args,
):
    run_inference_on_file(
        dataset_id,
        [to_nii(i) for i in i],
        out_file=out_seg,
        override=override,
        gpu=gpu,
        ddevice=ddevice,
        **args,
    )


def extract_vertebra_bodies_from_VibeSeg(
    nii_vibeSeg: Image_Reference,
    num_thoracic_verts: int = 12,
    num_lumbar_verts: int = 5,
    out_path: str | Path | None = None,
    out_path_poi: str | Path | None = None,
):
    """
    Extracts and labels vertebra bodies from a VibeSeg segmentation NIfTI file.

    This function processes a segmentation mask containing vertebrae and intervertebral discs (IVDs).
    It separates individual vertebra bodies by eroding and splitting the mask at IVD regions, labels the vertebrae
    from bottom to top (lumbar and thoracic), and optionally saves the labeled mask and point-of-interest (POI) data.

    Args:
        nii_vibeSeg (Image_Reference): Path or reference to the NIfTI file containing the VibeSeg segmentation mask.
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
        >>> nii_vibeSeg = "/path/to/vibe_segmentation.nii.gz"
        >>> labeled_mask, centroids = extract_vertebra_bodies_from_nii_vibeSeg(nii_vibeSeg, out_path="output_mask.nii.gz")
    """
    from TPTBox import Vertebra_Instance, calc_centroids

    # Load the nii_vibeSeg segmentation
    nii = to_nii(nii_vibeSeg, seg=True)
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
        sorted(
            {i: centroids_unsorted_srp[i, 50][0] for i in centroids_unsorted_srp.keys_region()}.items(),
            key=lambda x: x[1],
        )
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
    from TPTBox.segmentation import run_vibeseg

    # You can also use a string/Path if you want to set the path yourself.
    dataset = "/media/data/robert/datasets/dicom_example/dataset-VR-DICOM2/"
    in_file = BIDS_FILE(
        f"{dataset}/derivative_stiched/sub-111168222/T2w/sub-111168222_sequ-301-stiched_acq-ax_part-water_T2w.nii.gz",
        dataset,
    )
    out_file = in_file.get_changed_path(
        "nii.gz",
        "msk",
        parent="derivative",
        info={"seg": "VibeSeg", "mod": in_file.bids_format},
    )
    run_vibeseg(in_file, out_file)
