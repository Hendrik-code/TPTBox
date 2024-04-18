from pathlib import Path

import numpy as np

from TPTBox import BIDS_FILE, NII, Image_Reference, No_Logger, to_nii
from TPTBox.stitching.stitching import main as stitching_raw

logger = No_Logger()


def stitching(
    *bids_files: BIDS_FILE | NII,
    out: BIDS_FILE | str | Path,
    verbose_stitching=False,
    bias_field=False,
    kick_out_fully_integrated_images=True,
    verbose=True,
    dtype: type = float,
    ct: bool = False,
):
    out = str(out.file["nii.gz"]) if isinstance(out, BIDS_FILE) else str(out)
    files = [bf.file["nii.gz"] if isinstance(bf, BIDS_FILE) else bf.nii for bf in bids_files]
    is_seg = bids_files[0].seg if hasattr(bids_files[0], "seg") else bids_files[0].get_interpolation_order() == 0  # type: ignore
    logger.print("stitching", out, verbose=verbose)
    return stitching_raw(
        files,
        out,
        False,
        False,
        verbose_stitching,
        min_value=-1024 if ct else 0,
        bias_field=bias_field,
        kick_out_fully_integrated_images=kick_out_fully_integrated_images,
        is_segmentation=is_seg,
        dtype=dtype,
    )


def _crop_borders(nii: NII, chunk_info: str, cut: dict[str, tuple[slice, slice, slice]]) -> NII:
    if chunk_info not in cut:
        logger.print("chunk_info must be in [HWS, BWS, LWS]")
    ori = nii.orientation
    return nii.reorient_().apply_crop_(cut[chunk_info]).reorient_(ori)


def GNC_stitch_T2w(
    HWS: Image_Reference,  # noqa: N803
    BWS: Image_Reference,  # noqa: N803
    LWS: Image_Reference,  # noqa: N803
    n4_after_stitch: bool = False,
    # cut={
    #    "HWS": (slice(None), slice(0, 400), slice(None)),
    #    "BWS": (slice(None), slice(80, 400), slice(None)),
    #    "LWS": (slice(None), slice(48, 448), slice(None)),
    # },
):
    """Preprocessing steps where n4 each chunk, then stitch, then n4
    Args:
        HWS (NII | str | Path): Cervical region
        BWS (NII | str | Path): Thoracic region
        LWS (NII | str | Path): Lumbar region
        n4_after_stitch (bool): where to do n4 correction after stitching again
    Returns:
        NII: Stitched and n4 corrected nifty
    """
    chunks = {"HWS": {}, "BWS": {}, "LWS": {}}
    chunks["HWS"]["nii"] = NII.load(HWS, seg=False).reorient_()
    chunks["BWS"]["nii"] = NII.load(BWS, seg=False).reorient_()
    chunks["LWS"]["nii"] = NII.load(LWS, seg=False).reorient_()
    # for k in chunks.keys():
    #    # chunks[k]["n4"] = _crop_borders(n4_bias(chunks[k]["nii"], spline_param=200)[0], k, cut)
    #    # chunks[k]["n4"].apply_crop_slice_(cut[k])
    # chunks_m = {k: chunks[k]["n4"] for k in chunks.keys()}
    # chunks_a = list([l.nii for l in chunks_m.values()])
    chunks_a = [a["nii"].nii for a in chunks.values()]
    # Stitch three chunks together
    stitched, _ = stitching_raw(
        chunks_a,
        output=None,
        match_histogram=False,
        store_ramp=False,
        verbose=False,
        bias_field=False,
        save=False,
    )
    stitched_nii = to_nii(stitched)
    if n4_after_stitch:
        stitched_nii, _ = n4_bias(stitched_nii)
    slices = (
        _center_frontal(stitched_nii.shape[0]),
        slice(None),
        slice(None),
    )
    stitched_nii.apply_crop_(slices)
    return stitched_nii.set_dtype_(np.uint16)


def n4_bias(
    nii: NII,
    threshold: int = 70,
    spline_param: int = 200,
    dtype2nii: bool = False,
    norm: int = -1,
):
    from ants.utils.convert_nibabel import from_nibabel

    # print("n4 bias", nii.dtype)
    mask = nii.get_array()
    mask[mask < threshold] = 0
    mask[mask != 0] = 1
    mask_nii = nii.set_array(mask)
    mask_nii.seg = True
    mask_nii.dilate_msk_(mm=3, verbose=False)
    n4: NII = nii.n4_bias_field_correction(mask=from_nibabel(mask_nii.nii), spline_param=spline_param)
    if norm != -1:
        n4 *= norm / n4.max()
    if dtype2nii:
        n4.set_dtype_(nii.dtype)
    return n4, mask_nii


def _center_frontal(size):
    """Calculating the location of the frontalplain +- 256"""
    # The multiplicator-factor was empirically evaluated in an excel sheet with 11 random MRIs
    lower_bound = size * 0.2
    upper_bound = size * 0.75
    distance = int((upper_bound - lower_bound) / 2)
    middle = int(lower_bound + distance)
    # rather push it little back over the top in dorsal direction rather then not putting it into account
    small_value = middle - 122
    big_value = middle + 134
    return slice(small_value, big_value)
