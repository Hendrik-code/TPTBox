from __future__ import annotations

import contextlib
import itertools
import warnings
from functools import cache
from pathlib import Path

import numpy as np
from nibabel.affines import apply_affine
from scipy.ndimage import binary_opening, distance_transform_edt
from scipy.spatial import ConvexHull
from skimage.exposure import match_histograms

from TPTBox.core.nii_wrapper import NII, to_nii
from TPTBox.core.np_utils import np_bbox_binary
from TPTBox.logger import Print_Logger

logger = Print_Logger()


@contextlib.contextmanager
def _suppress_dtype_warning():
    """Silence the "Loaded NIfTY: incorrect dtype detected" ``UserWarning``.

    ``NII._check_if_nifty_is_lying_about_its_dtype`` (nii_wrapper.py, three
    ``warnings.warn`` sites around line 151/163/172) fires this warning
    during ``_unpack`` whenever the on-disk dtype doesn't match the array's
    actual range — typical for Philips-scaled magnitude MR arriving as
    ``int16`` with a large ``scl_slope``. The stitching pipeline handles
    that case explicitly (rebuilds a scale-1 image with the widened dtype
    in ``main``'s loading loop), so the warning is noise here — but we
    only suppress it for the current call's stack, not globally.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Loaded NIfTY: incorrect dtype detected.*",
            category=UserWarning,
        )
        yield


def get_rotation_and_spacing_from_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a NIfTI affine into its rotation matrix and voxel spacing.

    Adapted from nibabel.orientations.

    Args:
        affine: 4x4 affine transformation matrix.

    Returns:
        A 2-tuple of ``(rotation, spacing)`` where ``rotation`` is a 3x3
        orthonormal matrix and ``spacing`` is a 1-D array of three voxel sizes.
    """
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def get_ras_affine(rotation: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Build a RAS affine matrix from rotation, voxel spacing, and image origin.

    Adapted from TorchIO's IO utilities.

    Args:
        rotation: 3x3 orthonormal rotation matrix.
        spacing: 1-D array of three voxel spacings (mm).
        origin: 1-D array giving the index-space origin coordinates.

    Returns:
        A 4x4 RAS affine matrix.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    rotation_zoom = rotation * spacing
    translation_ras = rotation.dot(origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation_ras
    return affine


def get_all_corner_points(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Compute the eight world-space corner points of a voxel volume.

    Args:
        affine: 4x4 affine mapping voxel indices to world coordinates.
        shape: Volume shape (X, Y, Z).

    Returns:
        Array of shape (8, 3) with the world-space coordinates of all eight
        corners of the bounding box.
    """
    lst = list(itertools.product([0, 1], repeat=3))
    lst = np.array(lst) * np.array(shape)
    lst += 1

    return apply_affine(affine, lst)


def _occupancy_bbox(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned bounding box (min, max inclusive) of the nonzero region of `occ`.

    Thin wrapper around :func:`TPTBox.core.np_utils.np_bbox_binary` (which
    shares a 2-D projection between two of the three axis reductions for 3-D
    inputs, so ~2× faster than three independent ``np.any`` reductions). The
    slice tuple is repacked as ``(lo, hi)`` int64 arrays because the ramp
    math further down works on plain vectors, and ``None`` is returned for
    an all-zero occupancy so ``_aabb_overlaps`` can treat it as "no overlap
    possible".
    """
    try:
        slices = np_bbox_binary(occ > 0)
    except ValueError:
        return None
    lo = np.array([s.start for s in slices], dtype=np.int64)
    hi = np.array([s.stop - 1 for s in slices], dtype=np.int64)
    return lo, hi


def _aabb_overlaps(a: tuple[np.ndarray, np.ndarray] | None, b: tuple[np.ndarray, np.ndarray] | None) -> bool:
    """Whether two axis-aligned bounding boxes touch or intersect. ``None`` means empty."""
    if a is None or b is None:
        return False
    a_lo, a_hi = a
    b_lo, b_hi = b
    return bool(np.all(a_lo <= b_hi) and np.all(b_lo <= a_hi))


def get_max_affine_and_shape(
    points: np.ndarray,
    affines: list[np.ndarray],
    min_spacing: float | None = None,
    dtype: type = float,
    verbose: bool = False,
) -> NII:
    """Determine the optimal output affine and shape that encloses all input volumes.

    Iterates over all input affines and selects the rotation that minimises the
    bounding-box volume of the convex hull of ``points``.  The finest (minimum)
    voxel spacing across all inputs is used, optionally clipped from below by
    ``min_spacing``.

    Args:
        points: World-space corner coordinates of all input volumes, shape (N, 3).
        affines: List of 4x4 affine matrices, one per input volume.
        min_spacing: Optional lower-bound on the output voxel spacing (mm).
        dtype: NumPy dtype for the output image data.
        verbose: If True, prints chosen spacing, shape, origin, and optimal
            rotation to stdout.

    Returns:
        A zeroed :class:`NII` with the computed affine and shape, ready
        to be used as a resampling target.

    Raises:
        ValueError: If no valid rotation could be determined from ``affines``.
    """
    hull = ConvexHull(points)

    min_possible_volume = hull.volume
    min_rotation = None
    min_volume = float("inf")
    min_shape = [0, 0, 0]
    origen = [0, 0, 0]
    spacings = []
    opt_id = -1

    # print(points[hull.vertices])
    # Find best rotation
    for idx, affine in enumerate(affines, 1):
        rotation, spacing = get_rotation_and_spacing_from_affine(affine)
        spacings.append(np.abs(spacing))

        points_rotated = points.copy()
        for i in range(points.shape[0]):
            points_rotated[i] = rotation.T.dot(points[i])

        hull_np = points_rotated[hull.vertices]

        max_v = np.max(hull_np, axis=0)
        min_v = np.min(hull_np, axis=0)
        dif = max_v - min_v
        v = dif[0] * dif[1] * dif[2]

        if v <= min_volume:
            min_volume = v
            min_rotation = rotation
            min_shape = dif
            origen = (min_v, max_v)
            opt_id = idx
    if min_rotation is None:
        raise ValueError(affines)
    # TODO check if an other orientation fails the code
    # TODO add option to pick the spacing and not the max_spacing

    new_spacing = np.min(np.round(np.stack(spacings), decimals=6), 0)
    if min_spacing is not None and min_spacing != 0:
        new_spacing = np.maximum(min_spacing, new_spacing)

    shape: np.ndarray = np.ceil(min_shape / new_spacing)
    logger.on_neutral("Choose the following spacing:", new_spacing, verbose=verbose)
    logger.on_neutral(
        f"Output shape is {shape}, which utilizes {min_possible_volume / min_volume * 100:.1f} % of all voxels.",
        verbose=verbose,
    )
    affine = get_ras_affine(min_rotation, new_spacing, origen[0])
    logger.on_neutral("The new origin is ", np.round(affine[:3, 3], 2), verbose=verbose)
    logger.on_neutral(
        "The optimal rotation came from file number ",
        opt_id,
        " ",
        np.round(min_rotation.reshape(-1), 2),
        verbose=verbose,
    )
    return NII((np.zeros(shape.astype(int), dtype=dtype), affine, None))  # type: ignore


@cache
def buffer_reference(path: str | Path, bias_field: bool, crop: bool = False) -> NII:
    """Load (and optionally bias-correct) a NIfTI file, caching the result.

    Subsequent calls with the same ``(path, bias_field, crop)`` return the
    cached NII without re-reading or re-correcting the file. Caching by all
    three arguments (rather than just ``path`` as the previous module-level
    dict did) means a call with ``crop=True`` no longer poisons a later call
    with ``crop=False`` for the same path. ``lru_cache`` is also thread-safe,
    so this is safe under a ``ThreadPoolExecutor``-based dispatcher.

    Args:
        path: File path of the NIfTI image.
        bias_field: If True, applies N4 bias-field correction before caching.
        crop: Passed to :meth:`NII.n4_bias_field_correction` when ``bias_field``
            is True.

    Returns:
        The loaded (and optionally bias-corrected) :class:`NII`.
    """
    reference = NII.load(path, False)
    if bias_field:
        reference = reference.n4_bias_field_correction(crop=crop)
    return reference


def _auto_output_dtype(niis: list[NII]) -> type:
    """Pick the smallest lossless output dtype from the inputs' on-disk dtypes.

    Reads only the NIfTI headers — no pixel data is loaded. If every input is
    integer-typed AND has a trivial slope/inter (so the raw storage range is
    the true value range), returns the widest integer dtype that covers them
    all; otherwise falls back to float32. This lets the stitcher store
    magnitude MR outputs as uint16 (or int16) when the source already fit in
    16 bits, halving the on-disk and downstream RAM footprint versus float64.

    The slope check is load-bearing: Philips exports the axial VIBE-DIXON as
    ``int16`` with a large ``scl_slope`` (~641 or ~2260) so the raw bytes span
    the signed range but ``fdata = raw * slope`` reaches ~1e6. Returning
    ``int16`` (or ``uint32`` after `_check_if_nifty_is_lying_about_its_dtype`
    widens the array) and casting the blended float back to it wraps values
    outside the target range into garbage — the stitched output ends up as a
    static-noise pattern or an unexpectedly wide dtype (e.g. ``uint32`` for
    the 6-echo mDIX magnitudes). When ANY input has slope != 1 or inter != 0
    we bail to float32 so the accumulator's true range survives the save.
    ``dataobj.slope`` / ``dataobj.inter`` on the underlying ``Nifti1Image``
    give the ORIGINAL header values (before NII's dtype-widening pass), which
    is what we need for this decision.
    """
    dtypes = [np.dtype(nii.dtype) for nii in niis]
    if not all(np.issubdtype(d, np.integer) for d in dtypes):
        return np.float32
    for nii in niis:
        slope = getattr(nii.nii.dataobj, "slope", 1.0)
        inter = getattr(nii.nii.dataobj, "inter", 0.0)
        if slope is None or inter is None:
            return np.float32
        if not (np.isfinite(slope) and np.isfinite(inter)):
            return np.float32
        if float(slope) != 1.0 or float(inter) != 0.0:
            return np.float32
    return max(dtypes, key=lambda d: d.itemsize).type  # e.g. np.uint16


def main(  # noqa: C901
    images: list[str] | list[Path] | list[NII],
    output: str | None,
    match_histogram: bool = False,
    store_ramp: bool = False,
    verbose: bool = False,
    min_value: float | None = None,
    bias_field: bool = True,
    crop_to_bias_field: bool = False,
    crop_empty: bool = False,
    histogram: str | None = None,
    ramp_edge_min_value: int = 5,
    min_spacing: int | None = None,
    kick_out_fully_integrated_images: bool = False,
    is_segmentation: bool = False,
    dtype: type | str = float,
    save: bool = True,
    ramp_path=None,
) -> tuple[NII | None, NII | None]:
    """Stitch multiple overlapping NIfTI volumes into a single output volume.

    The algorithm:

    1. Optionally applies N4 bias-field correction and histogram matching to
       each input volume.
    2. Finds the minimum bounding-box affine that encloses all inputs.
    3. Resamples every volume into that common space.
    4. Computes per-voxel blending weights using distance-transform-based
       ramps in overlap regions.
    5. Combines all resampled volumes with those weights and saves the result.

    Args:
        images: Input volumes as file paths or pre-loaded :class:`NII` (or
            :class:`Nifti1Image`) objects. At least two are required.
        output: Output file path (``".nii.gz"`` extension is appended if
            absent). If None the result is returned without writing to disk
            (``save`` must also be False).
        match_histogram: If True, matches the histogram of each volume to the
            previous one before stitching.
        store_ramp: If True, also saves the per-volume blend weights as a 4-D
            NIfTI alongside the stitched output.
        verbose: If True, prints progress messages to stdout.
        min_value: Background fill used as ``cval`` when resampling each chunk
            into the target space, and — when explicitly set — a hard floor
            applied to the stitched output. Pass ``0`` for MR, ``-1024`` for
            CT. Default ``None``: use ``0`` internally as the background /
            NaN-fill and only apply a hard floor when the output dtype cannot
            represent negatives (unsigned integer types), which prevents
            negative-to-huge-positive wraparound on the cast. Explicit values
            (segmentations force ``0``, CT callers pass ``-1024``) are always
            enforced; ``None`` lets signed / float outputs keep legitimate
            negative signal (Philips-scaled fat-fraction, phase, B0 offsets).
        bias_field: If True, applies N4 bias-field correction to each input
            before stitching. Forced to False for segmentations.
        crop_to_bias_field: If True, crops each bias-corrected volume to the
            region affected by the correction.
        crop_empty: If True, crops the final output to its non-background
            bounding box.
        histogram: Path or index string used as the histogram reference for
            ``match_histogram``.
        ramp_edge_min_value: Minimum thickness (voxels) of non-overlapping
            regions used when computing distance-transform ramps.
        min_spacing: Minimum allowed output voxel spacing (mm). Overrides the
            finest input spacing when specified.
        kick_out_fully_integrated_images: If True, recursively removes volumes
            that are fully enclosed within another volume.
        is_segmentation: If True, disables bias field, histogram matching, and
            uses nearest-neighbour resampling with integer dtype selection.
        dtype: Output data type. Accepts a Python type (e.g. ``float``,
            ``np.uint16``) or a string key from the internal type mapping.
        save: If True, writes the stitched image to ``output``.
        ramp_path: Optional explicit output path for the ramp NIfTI. Only used
            when ``store_ramp`` is True; when None the ramp path is derived
            from ``output``.

    Returns:
        A 2-tuple ``(stitched_nii, ramp_nii)`` where ``ramp_nii`` is None
        unless ``store_ramp`` is True. Returns ``(None, None)`` when fewer
        than two images are supplied.
    """
    # Suppress `_check_if_nifty_is_lying_about_its_dtype` UserWarnings for
    # the whole main() call — stitching handles the "dtype mismatches
    # actual range" case explicitly, and the warning is just noise here.
    # See `_suppress_dtype_warning` for scope / rationale.
    with _suppress_dtype_warning():
        return _main(
            images,
            output,
            match_histogram,
            store_ramp,
            verbose,
            min_value,
            bias_field,
            crop_to_bias_field,
            crop_empty,
            histogram,
            ramp_edge_min_value,
            min_spacing,
            kick_out_fully_integrated_images,
            is_segmentation,
            dtype,
            save,
            ramp_path,
        )


def _main(  # noqa: C901
    images: list[str] | list[Path] | list[NII],
    output: str | None,
    match_histogram: bool = False,
    store_ramp: bool = False,
    verbose: bool = False,
    min_value: float | None = None,
    bias_field: bool = True,
    crop_to_bias_field: bool = False,
    crop_empty: bool = False,
    histogram: str | None = None,
    ramp_edge_min_value: int = 5,
    min_spacing: int | None = None,
    kick_out_fully_integrated_images: bool = False,
    is_segmentation: bool = False,
    dtype: type | str = float,
    save: bool = True,
    ramp_path=None,
) -> tuple[NII | None, NII | None]:
    """Body of :func:`main`, split out so :func:`main` can wrap it in the
    dtype-warning suppression context.
    """
    np.set_printoptions(precision=2, floatmode="fixed")
    if is_segmentation:
        bias_field = False
        crop_to_bias_field = False
        min_value = 0
        match_histogram = False
        histogram = None
    _bg_value: float = 0.0 if min_value is None else float(min_value)
    if len(images) == 0 or len(images) == 1:
        logger.on_fail("Need at least two images (-i ...nii.gz ...nii.gz) to stitch. Got:", images)
        return None, None
    corners = []
    affines = []
    niis: list[NII] = []
    logger.on_log("### loading ###", verbose=verbose)
    for f_name in images:
        if isinstance(f_name, (Path, str)):
            logger.on_neutral("Load ", f_name, Path(f_name), verbose=verbose)
            # Load NII
            nii = to_nii(Path(f_name), seg=is_segmentation)
        elif isinstance(f_name, (NII)):
            nii = f_name
            nii.seg = is_segmentation
        else:
            nii = NII(f_name, seg=is_segmentation)
        if bias_field:
            nii = nii.n4_bias_field_correction(crop=crop_to_bias_field)
        ## Histogram equalization.
        if match_histogram:
            if histogram is None:
                if len(niis) == 0:
                    reference = None
                else:
                    logger.on_neutral("Histogram equalization with previous file", verbose=verbose)
                    reference = niis[-1].get_array()
            elif histogram.isdigit():
                logger.on_neutral("Histogram equalization", images[int(histogram)], verbose=verbose)
                reference = buffer_reference(images[int(histogram)], bias_field=bias_field, crop=crop_to_bias_field)  # type: ignore
            else:
                logger.on_neutral("Histogram equalization with file", histogram, verbose=verbose)
                reference = buffer_reference(histogram, bias_field=bias_field, crop=crop_to_bias_field)  # type: ignore
            if reference is not None:
                image = nii.get_array()

                matched = match_histograms(image.astype(float), reference.astype(float))
                matched[matched <= _bg_value] = _bg_value
                nii = nii.set_array(matched)

        niis.append(nii)
        # Get affine and points for minimum enclosing Rectangle calculation
        affine = nii.affine
        affines.append(affine)

        corners_current = get_all_corner_points(affine, nii.shape)
        corners.append(corners_current)

    corners_current = np.concatenate(corners, axis=0)

    # compute output shape and affine
    logger.on_log("### compute output shape and affine ###", verbose=verbose)
    if is_segmentation:
        max_value = max([x.max() for x in niis])
        if max_value < 256:
            dtype2 = np.uint8
        elif max_value < 256 * 256:
            dtype2 = np.uint16
        elif max_value < 256 * 256 * 256 * 256:
            dtype2 = np.uint32
        else:
            dtype2 = np.uint64
        dtype = dtype2
    else:
        # Auto-detect the output dtype from the input headers when the caller
        # asked for "auto". Blending math runs in float32 (was float64) —
        # halves peak RAM of target_list/occupancy_list on large stitched
        # volumes; the final cast at save time uses `dtype` (uint16 etc.).
        if isinstance(dtype, str) and dtype == "auto":
            dtype = _auto_output_dtype(niis)
        dtype2 = np.float32
    nii_out = get_max_affine_and_shape(corners_current, affines, min_spacing=min_spacing, dtype=dtype2, verbose=verbose)
    target_list = []
    occupancy_list = []
    # get resampled arrays and occupancy
    logger.on_log("### resample to new space ###", verbose=verbose)
    for i, nii in enumerate(niis, 1):
        logger.on_neutral(f"{i:2}/{len(niis):2} resampled", end="\r", verbose=verbose)
        nii_new = nii.resample_from_to(nii_out, order=0 if is_segmentation else 3, mode="constant", c_val=_bg_value, verbose=False)
        arr_new = nii_new.get_array()
        if not is_segmentation and np.issubdtype(arr_new.dtype, np.floating):
            np.nan_to_num(arr_new, copy=False, nan=_bg_value, posinf=_bg_value, neginf=_bg_value)
        target_list.append(arr_new)
        b = NII((np.ones(nii.shape, dtype=np.float32), nii.affine, None))  # type: ignore
        b = b.resample_from_to(nii_new, order=0, c_val=0, mode="constant", verbose=False)
        if is_segmentation:
            x = arr_new > 0
            occupancy_list.append((b.get_array() * x.astype(np.int8)).astype(np.float32))  # Keep segmentation if other is 0

        else:
            occupancy_list.append(b.get_array().astype(np.float32))

    logger.on_log("\n### ramp stitching ###", verbose=verbose)
    # Per-chunk axis-aligned bounding box in target-space voxel coords.
    # Precomputing once avoids the O(N_chunks^2) full-volume copy + multiply
    # inside the ramp loop for pairs that don't touch. `_occupancy_bbox`
    # returns None for an empty occupancy — treated as "no overlap possible".
    bboxes = [_occupancy_bbox(occ) for occ in occupancy_list]
    grid_shape_arr = np.asarray(occupancy_list[0].shape, dtype=np.int64)
    # Padding around the joint AABB: `ramp_edge_min_value` so binary_opening's
    # erode+dilate at the crop boundary yields the same result as on the full
    # volume; +1 slack for the distance transform.
    _ramp_pad = max(int(ramp_edge_min_value), 1) + 1
    # ramp stitching
    combinations = list(itertools.combinations(range(len(target_list)), 2))
    _ramp_done = 0
    _ramp_skipped_aabb = 0
    _ramp_skipped_no_voxel_overlap = 0
    for idx, item in enumerate(combinations, 1):
        logger.on_neutral(f"{idx:2}/{len(combinations):2} ramp stitching", end="\r", verbose=verbose)
        # Skip disjoint pairs before touching the full-volume arrays.
        if not _aabb_overlaps(bboxes[item[0]], bboxes[item[1]]):
            _ramp_skipped_aabb += 1
            continue
        # Work on the union-AABB sub-volume of the two chunks. Outside the
        # union `arr_i / sum_` equals the original `arr_i_full` (there,
        # overlap = 0, arr_i_ = binary mask of arr_i and the "other" mask
        # is 0, so sum_ = arr_i_ and arr_i / sum_ = arr_i). And chunk i's
        # occupancy support is fully contained in bboxes[i] ⊆ union, so
        # writing the result back only at the sub-slice is functionally
        # identical to the previous full-volume compute — but the ramp's
        # peak RAM drops from ~full-volume to ~union-AABB size (typically
        # 2 adjacent chunks tall).
        lo_i, hi_i = bboxes[item[0]]
        lo_j, hi_j = bboxes[item[1]]
        lo = np.maximum(np.minimum(lo_i, lo_j) - _ramp_pad, 0)
        hi = np.minimum(np.maximum(hi_i, hi_j) + _ramp_pad, grid_shape_arr - 1)
        sub = (
            slice(int(lo[0]), int(hi[0]) + 1),
            slice(int(lo[1]), int(hi[1]) + 1),
            slice(int(lo[2]), int(hi[2]) + 1),
        )
        # TODO fix intersection with more than two occupancies
        arr_1_full = occupancy_list[item[0]]
        arr_2_full = occupancy_list[item[1]]
        ###
        structure = np.ones((ramp_edge_min_value, ramp_edge_min_value, ramp_edge_min_value), dtype=bool)
        arr_1: np.ndarray = arr_1_full[sub].astype(np.float32, copy=True)
        arr_2: np.ndarray = arr_2_full[sub].astype(np.float32, copy=True)
        overlap = (arr_1 * arr_2) > 0.0
        if overlap.sum() > 0:
            _ramp_done += 1
            arr_1_ = (arr_1 > 0.0).astype(np.float32) - overlap
            arr_2_ = (arr_2 > 0.0).astype(np.float32) - overlap
            if ramp_edge_min_value == 0:
                arr_1_opened: np.ndarray = arr_1_
                arr_2_opened: np.ndarray = arr_2_
            else:
                arr_1_opened: np.ndarray = binary_opening(arr_1_, structure=structure, iterations=1, brute_force=True)
                arr_2_opened: np.ndarray = binary_opening(arr_2_, structure=structure, iterations=1, brute_force=True)

            arr_1[overlap] = distance_transform_edt(1.0 - arr_2_opened)[overlap]  # type: ignore
            arr_2[overlap] = distance_transform_edt(1.0 - arr_1_opened)[overlap]  # type: ignore
            arr_1_[overlap] = arr_1[overlap]
            arr_2_[overlap] = arr_2[overlap]
            sum_ = arr_1_ + arr_2_
            sum_[sum_ == 0] = 1.0
            arr_1_sub = arr_1 / sum_
            arr_2_sub = arr_2 / sum_
            # Chunk i's occupancy is fully contained inside bboxes[i] ⊆ sub,
            # so the sub-array max equals the volume-wide max.
            max_1 = float(arr_1_sub.max())
            max_2 = float(arr_2_sub.max())
            if max_1 != 1:
                import warnings

                warnings.warn(
                    str((float(arr_1_sub.min()), max_1)) + " the image in fully incorporated insight of an other " + str(images),
                    stacklevel=4,
                )
                if kick_out_fully_integrated_images:
                    images.pop(item[0])

            elif max_2 != 1:
                import warnings

                warnings.warn(
                    str((float(arr_2_sub.min()), max_2)) + " the image in fully incorporated insight of an other " + str(images),
                    stacklevel=4,
                )
                if kick_out_fully_integrated_images:
                    images.pop(item[1])
            if (max_1 != 1 or max_2 != 1) and kick_out_fully_integrated_images:
                logger.on_warning("kick_out_fully_integrated_images")
                logger.on_warning(images)
                # Pass EVERY argument by keyword — the positional form used
                # to silently drop `is_segmentation`, `dtype`, `ramp_path` and
                # shift `save` onto `is_segmentation`, which flipped the
                # recursion into the segmentation code path and produced an
                # unexpected uint dtype for what was actually magnitude MR.
                return main(
                    images=images,
                    output=output,
                    match_histogram=match_histogram,
                    store_ramp=store_ramp,
                    verbose=verbose,
                    min_value=min_value,
                    bias_field=bias_field,
                    crop_to_bias_field=crop_to_bias_field,
                    crop_empty=crop_empty,
                    histogram=histogram,
                    ramp_edge_min_value=ramp_edge_min_value,
                    min_spacing=min_spacing,
                    kick_out_fully_integrated_images=kick_out_fully_integrated_images,
                    is_segmentation=is_segmentation,
                    dtype=dtype,
                    save=save,
                    ramp_path=ramp_path,
                )
            arr_1_full[sub] = arr_1_sub.astype(arr_1_full.dtype, copy=False)
            arr_2_full[sub] = arr_2_sub.astype(arr_2_full.dtype, copy=False)
        else:
            _ramp_skipped_no_voxel_overlap += 1
            continue
    logger.on_neutral(
        f"\nramp summary: {_ramp_done} computed, "
        f"{_ramp_skipped_aabb} skipped (disjoint AABB), "
        f"{_ramp_skipped_no_voxel_overlap} skipped (no voxel overlap) "
        f"of {len(combinations)} pairs",
        verbose=verbose,
    )
    # Aggregate: in-place accumulate `t * occupancy` per chunk instead of
    # `np.stack(target_list) * np.stack(occupancy_list)` which would peak at
    # ~2 × N × volume of temporary float arrays.
    target_arr = np.zeros(target_list[0].shape, dtype=dtype2)
    if is_segmentation:
        for t, o in zip(target_list, occupancy_list):
            target_arr += (t * np.round(o)).astype(dtype2)  # TODO assuming only two intersecting regions
    else:
        for t, o in zip(target_list, occupancy_list):
            target_arr += t * o
    # Hard-floor policy:
    #   * If the caller passed `min_value` explicitly (segmentations force 0,
    #     CT typically passes -1024), always apply that floor.
    #   * If `min_value is None` (the default), only apply a floor when the
    #     output dtype cannot represent negatives (unsigned integer types) —
    #     otherwise negative-to-huge-positive wraparound would silently corrupt
    #     the save. Signed/float outputs keep legitimate negative signal
    #     (Philips-scaled fat-fraction, phase, B0 offsets).
    _floor: float | None
    if min_value is not None:
        _floor = float(min_value)
    elif np.issubdtype(np.dtype(dtype), np.unsignedinteger):
        _floor = 0.0
    else:
        _floor = None
    if _floor is not None:
        target_arr[target_arr <= _floor] = _floor
    logger.on_log("\n### Save ###", verbose=verbose)
    if output is not None:
        output = str(output)
        if not output.endswith(".nii.gz"):
            output = output.replace(".nii", "") + ".nii.gz"
        if "/" not in output and "\\" not in output:
            assert isinstance(images[0], (str, Path)), "automatic path fetching only possible if images are strings or Path, not objects"
            output = str(Path(Path(images[0]).parent, output))
    # Accept a string dtype name ("uint8", "float", …) from the CLI argparse
    # path. `np.dtype(...)` covers every alias `type_mapping` used to define
    # explicitly. Non-string dtypes (types picked by `_auto_output_dtype` or
    # passed in directly) pass through unchanged.
    if isinstance(dtype, str):
        dtype = np.dtype(dtype).type
    nii_out = nii_out.set_array(target_arr.astype(dtype))
    if bias_field:
        nii_out = nii_out.n4_bias_field_correction()
    if crop_empty:
        # Crop to the union of per-chunk occupancy AABBs. The previous path
        # went through compute_crop_slice on a 4-D (N, X, Y, Z) stack, which
        # sliced the wrong axes when applied to 3-D nii_out; deriving the
        # crop directly from `bboxes` is both cheaper and correct.
        valid_bboxes = [b for b in bboxes if b is not None]
        if valid_bboxes:
            lo = np.stack([b[0] for b in valid_bboxes]).min(axis=0)
            hi = np.stack([b[1] for b in valid_bboxes]).max(axis=0)
            ex_slice = (
                slice(int(lo[0]), int(hi[0]) + 1),
                slice(int(lo[1]), int(hi[1]) + 1),
                slice(int(lo[2]), int(hi[2]) + 1),
            )
            nii_out = nii_out[ex_slice]
        else:
            ex_slice = ()
    else:
        ex_slice = ()

    nii_out.set_dtype_(dtype)

    if save:
        nii_out.save(output)  # type: ignore
        logger.on_save("Saved ", output, verbose=verbose)

    if store_ramp:
        occupancy_arr = np.stack(occupancy_list, -1)
        if crop_empty:
            occupancy_arr = occupancy_arr[ex_slice]
        assert output is not None
        nii_occ = nii_out.set_array(occupancy_arr)
        nii_occ.set_dtype_(np.int8)
        output = output.replace(".nii.gz", "_ramps.nii.gz").replace("_msk_", "_") if ramp_path is None else ramp_path
        if save:
            nii_occ.save(output)  # type: ignore
            logger.on_save("Saved ", output, verbose=verbose)
        return nii_out, nii_occ
    logger.on_ok("\n### Finished ###", verbose=verbose)
    return nii_out, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nii-stitching")
    parser.add_argument("-i", "--images", nargs="+", default=[], help="filenames of images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.nii.gz",
        help="filename of output image",
    )
    parser.add_argument(
        "-hist_n",
        "--histogram_name",
        type=str,
        default=None,
        help="use this file for histogram_matching instead",
    )
    help_str = "fits the histogram, for the previous in the file list. "
    parser.add_argument(
        "-hists",
        "--match_histogram",
        default=False,
        action="store_true",
        help=help_str,
    )
    help_str = "n4_bias_field_correction"
    parser.add_argument(
        "-no_bias",
        "--no_bias_field_correction",
        default=False,
        action="store_true",
        help=help_str,
    )
    help_str = "crop with generated n4_bias_field_correction"
    parser.add_argument(
        "-bias_crop",
        "--bias_field_correction_crop",
        default=False,
        action="store_true",
        help=help_str,
    )
    help_str = "crop black spaces"
    parser.add_argument("-crop", "--crop", default=False, action="store_true", help=help_str)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    help_str = "intersecting images are bended together by there distance from vowels that are not intersecting. This flag saves the blending as a 4d nii."
    parser.add_argument("-sr", "--store_ramp", default=False, action="store_true", help=help_str)
    help_str = "If two images cut in a way, that would leave a thin slice of less than x voxels pixel, it will not be considered for the ramp calculation."
    parser.add_argument("-ramp_e", "--ramp_edge_min_value", type=int, default=5, help=help_str)
    help_str = "all values below will be set to min_value. (MRI=0, CT<=-1024)"
    parser.add_argument("-min_value", "--min_value", type=int, default=0, help=help_str)
    parser.add_argument("-ms", "--min_spacing", type=int, default=None, help="")
    parser.add_argument("-seg", "--is_segmentation", default=False, action="store_true")
    parser.add_argument("-dtype", "--dtype", default=float, type=str, help="output type")
    args = parser.parse_args()
    if args.verbose:
        try:
            from pprint import pprint

            pprint(args.__dict__)
        except Exception:
            print(args)

    main(
        args.images,
        args.output,
        args.match_histogram,
        args.store_ramp,
        args.verbose,
        bias_field=not args.no_bias_field_correction,
        crop_to_bias_field=args.bias_field_correction_crop,
        crop_empty=args.crop,
        ramp_edge_min_value=args.ramp_edge_min_value,
        histogram=args.histogram_name,
        min_value=args.min_value,
        min_spacing=args.min_spacing,
        is_segmentation=args.is_segmentation,
        dtype=args.dtype,
    )
