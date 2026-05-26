from __future__ import annotations

from pathlib import Path

import numpy as np

from TPTBox import BIDS_FILE, NII, Image_Reference, No_Logger, to_nii
from TPTBox.stitching.stitching import main as stitching_raw

logger = No_Logger()


def stitching(
    bids_files: list[BIDS_FILE | NII | str | Path] | list,
    out: BIDS_FILE | str | Path,
    is_seg: bool = False,
    is_ct: bool = False,
    verbose_stitching: bool = False,
    bias_field: bool = False,
    kick_out_fully_integrated_images: bool = True,
    verbose: bool = True,
    dtype: type = float,
    match_histogram: bool = False,
    store_ramp: bool = False,
    ramp_path=None,
) -> tuple:
    """Stitch a list of BIDS/NII volumes into a single output NIfTI file.

    Convenience wrapper around :func:`stitching_raw` that accepts BIDS_FILE
    objects, NII instances, or raw file paths and resolves the output path from
    a BIDS_FILE if necessary.

    Args:
        bids_files: Input volumes to stitch.  Accepts any mix of
            :class:`BIDS_FILE`, :class:`NII`, ``str``, or :class:`Path`.
        out: Destination path for the stitched volume. When a
            :class:`BIDS_FILE` is provided, the ``"nii.gz"`` file path is used.
        is_seg: If True, treats the inputs as segmentation images (disables
            bias field and histogram matching, uses integer dtypes).
        is_ct: If True, sets the background ``min_value`` to ``-1024`` (CT
            air) instead of ``0`` (MRI).
        verbose_stitching: If True, forwards verbose output from the low-level
            stitching routine.
        bias_field: If True, applies N4 bias-field correction to each input.
        kick_out_fully_integrated_images: If True, removes volumes that are
            fully contained within another volume before stitching.
        verbose: If True, logs the output path before stitching.
        dtype: NumPy dtype for the output array.
        match_histogram: If True, matches histograms between consecutive inputs.
        store_ramp: If True, also returns the per-volume blending weight array.

    Returns:
        A 2-tuple ``(stitched_nii, ramp_nii)`` as returned by
        :func:`stitching_raw`.
    """
    out = str(out.file["nii.gz"]) if isinstance(out, BIDS_FILE) else str(out)
    files = [to_nii(bf).nii for bf in bids_files]
    logger.print("stitching", out, verbose=verbose)
    return stitching_raw(
        files,
        out,
        match_histogram=match_histogram,
        store_ramp=store_ramp,
        verbose=verbose_stitching,
        min_value=-1024 if is_ct else 0,
        bias_field=bias_field,
        kick_out_fully_integrated_images=kick_out_fully_integrated_images,
        is_segmentation=is_seg,
        dtype=dtype,
        ramp_path=ramp_path,
    )


def _crop_borders(nii: NII, chunk_info: str, cut: dict[str, tuple[slice, slice, slice]]) -> NII:
    """Crop a NII volume to predefined borders for a given spine-chunk key."""
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
) -> NII:
    """Apply N4 bias correction to each chunk, stitch them, then apply N4 again.

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
) -> tuple[NII, NII]:
    """Apply N4 bias-field correction to a NII image with automatic mask generation.

    Voxels below ``threshold`` are excluded from the bias estimation via a
    dilated binary mask.

    Args:
        nii: Input image to correct.
        threshold: Intensity threshold below which voxels are excluded from
            the bias estimation mask.
        spline_param: B-spline control point spacing for the bias field model.
        dtype2nii: If True, casts the corrected image back to the original
            dtype of ``nii``.
        norm: If != -1, normalizes the corrected image so its maximum equals
            ``norm``.

    Returns:
        A 2-tuple ``(n4_corrected_nii, mask_nii)`` where ``mask_nii`` is the
        dilated binary mask used during correction.
    """
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


def _center_frontal(size: int) -> slice:
    """Compute a slice centred on the frontal plane of a spinal MRI volume."""
    # The multiplicator-factor was empirically evaluated in an excel sheet with 11 random MRIs
    lower_bound = size * 0.2
    upper_bound = size * 0.75
    distance = int((upper_bound - lower_bound) / 2)
    middle = int(lower_bound + distance)
    # rather push it little back over the top in dorsal direction rather then not putting it into account
    small_value = middle - 122
    big_value = middle + 134
    return slice(small_value, big_value)
