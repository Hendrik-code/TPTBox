from __future__ import annotations

import itertools
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np
from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import binary_opening, distance_transform_edt
from scipy.spatial import ConvexHull
from skimage.exposure import match_histograms


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


def get_array(nii: Nifti1Image) -> np.ndarray:
    """Extract the voxel data from a NIfTI image as a writable NumPy array.

    Args:
        nii: Source NIfTI image.

    Returns:
        A copy of the image data array with the original dtype preserved.
    """
    return np.asanyarray(nii.dataobj, dtype=nii.dataobj.dtype).copy()  # type: ignore


def set_array(nii: Nifti1Image, arr: np.ndarray) -> Nifti1Image:
    """Return a new NIfTI image with the given array, preserving header and affine.

    If the dtype of ``arr`` differs from the existing image, the header dtype is
    updated accordingly.

    Args:
        nii: Source NIfTI image whose header and affine are reused.
        arr: Replacement voxel data array.

    Returns:
        A new :class:`Nifti1Image` backed by ``arr``.
    """
    if nii.dataobj.dtype == arr.dtype:  # type: ignore
        nii = Nifti1Image(arr, nii.affine, nii.header)
    else:
        nii = Nifti1Image(get_array(nii), nii.affine, nii.header)
        nii.set_data_dtype(arr.dtype)
        nii = Nifti1Image(arr, nii.affine, nii.header)
    return nii


def argmin(lst: list) -> int:
    """Return the index of the minimum element in a list.

    Args:
        lst: Input list of comparable elements.

    Returns:
        Zero-based index of the smallest element.
    """
    return lst.index(min(lst))


def get_max_affine_and_shape(
    points: np.ndarray,
    affines: list[np.ndarray],
    min_spacing: float | None = None,
    dtype: type = float,
    verbose: bool = False,
) -> Nifti1Image:
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
        A zeroed :class:`Nifti1Image` with the computed affine and shape, ready
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
    print("Choose the following spacing:", new_spacing) if verbose else None
    print(f"Output shape is {shape}, which utilizes {min_possible_volume / min_volume * 100:.1f} % of all voxels.") if verbose else None
    affine = get_ras_affine(min_rotation, new_spacing, origen[0])
    print("The new origin is ", np.round(affine[:3, 3], 2)) if verbose else None
    print("The optimal rotation came from file number ", opt_id, " ", np.round(min_rotation.reshape(-1), 2)) if verbose else None
    return nib.Nifti1Image(np.zeros(shape.astype(int), dtype=dtype), affine)  # type: ignore


def compute_crop_slice(nii: Nifti1Image, minimum=0, dist=0) -> tuple[slice, slice, slice]:
    """Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

    Args:
        minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
        dist (int): The amount of padding to be added to the cropped image. Default value is 0.
        other_crop (tuple[slice,...], optional): A tuple of slice objects representing the slice of an other image to be combined with the current slice. Default value is None.

    Returns:
        ex_slice: A tuple of slice objects that need to be applied to crop the image.
        origin_shift: A tuple of integers representing the shift required to obtain the centroids of the cropped image.

    Note:
        - The computed slice removes the unused space from the image based on the minimum value.
        - The padding is added to the computed slice.
        - If the computed slice reduces the array size to zero, a ValueError is raised.
        - If other_crop is not None, the computed slice is combined with the slice of another image to obtain a common region of interest.
        - Only None slice is supported for combining slices.
    """
    shp = nii.shape
    zms = nii.header.get_zooms()  # type: ignore
    d = np.around(dist / np.asarray(zms)).astype(int)
    array = get_array(nii)  # + minimum
    msk_bin = np.zeros(array.shape, dtype=bool)
    # bool_arr[array<minimum] = 0
    msk_bin[array > minimum] = 1
    # msk_bin = np.asanyarray(bool_arr, dtype=bool)
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)
    if cor_msk[0].shape[0] == 0:
        raise ValueError("Array would be reduced to zero size")
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = max(0, c_min[0] - d[0])
    y0 = max(0, c_min[1] - d[1])
    z0 = max(0, c_min[2] - d[2])
    x1 = min(shp[0], c_max[0] + d[0])
    y1 = min(shp[1], c_max[1] + d[1])
    z1 = min(shp[2], c_max[2] + d[2])
    ex_slice = (slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1))
    return ex_slice


def dilate_msk(msk_i_data: np.ndarray, mm: int = 5, connectivity: int = 3) -> np.ndarray:
    """Dilate each label in a segmentation mask by a fixed number of voxels.

    Args:
        msk_i_data: Integer-valued 3-D segmentation array. Label 0 is background.
        mm: Number of dilation iterations to apply per label.
        connectivity: Structuring-element connectivity (1 = face-connected,
            3 = fully-connected including diagonals).

    Returns:
        A dilated ``uint8`` array of the same shape as ``msk_i_data``.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    struct = generate_binary_structure(3, connectivity)
    out = msk_i_data.copy() * 0
    for i in np.unique(msk_i_data):
        if i == 0:
            continue
        data = msk_i_data.copy()
        data[i != data] = 0
        msk_ibe_data = binary_dilation(data, structure=struct, iterations=mm)
        out[out == 0] = msk_ibe_data[out == 0]
    return out.astype(np.uint8)


def n4_bias_field_correction(
    nib: Nifti1Image,
    mask: np.ndarray | None = None,
    threshold: int = 60,
    shrink_factor: int = 4,
    convergence: dict | None = None,
    spline_param: int = 150,
    verbose: bool = False,
    weight_mask: np.ndarray | None = None,
    crop: bool = False,
) -> Nifti1Image:
    """Apply N4 bias-field correction to a NIfTI image using ANTsPy.

    A binary mask is derived automatically from voxels above ``threshold``
    and dilated by 3 voxels before correction is applied.

    Args:
        nib: Input NIfTI image to correct.
        mask: Optional pre-computed binary mask passed to ANTsPy. Overridden
            when ``threshold != 0``.
        threshold: Voxel intensity threshold for automatic mask generation.
            Set to 0 to disable automatic masking.
        shrink_factor: Image downsampling factor used inside ANTsPy to
            speed up computation.
        convergence: ANTsPy convergence dict with keys ``"iters"`` and
            ``"tol"``. Defaults to ``{"iters": [50, 50, 50, 50], "tol": 1e-7}``.
        spline_param: B-spline control point spacing for the bias field model.
        verbose: If True, ANTsPy prints progress information.
        weight_mask: Optional spatial weight mask passed to ANTsPy.
        crop: If True, crops the corrected image to the region where the bias
            field differed from the input.

    Returns:
        The bias-field-corrected NIfTI image.

    Raises:
        ModuleNotFoundError: If ``antspyx`` is not installed.
    """
    try:
        import ants
        import ants.utils.bias_correction as bc  # pip install antspyx==0.4.2
        from ants.utils.convert_nibabel import from_nibabel, to_nibabel

    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("n4 bias field correction uses ants install it with pip install antspyx==0.4.2") from err
        # 5.3 or higher
        import ants
        import ants.ops.bias_correction as bc  # pip install antspyx

        # TODO add conversion and remove this

        def from_nibabel(nib_image):
            """Converts a given Nifti image into an ANTsPy image.

            Parameters
            ----------
                img: NiftiImage

            Returns:
            -------
                ants_image: ANTsImage
            """
            ndim = nib_image.ndim

            if ndim < 3:
                print("Dimensionality is less than 3.")
                return None

            q_form = nib_image.get_qform()
            spacing = nib_image.header["pixdim"][1 : ndim + 1]

            origin = np.zeros(ndim)
            origin[:3] = q_form[:3, 3]

            direction = np.diag(np.ones(ndim))
            direction[:3, :3] = q_form[:3, :3] / spacing[:3]

            ants_img = ants.from_numpy(data=nib_image.get_fdata(), origin=origin.tolist(), spacing=spacing.tolist(), direction=direction)

            return ants_img

        def to_nibabel(img: ants.core.ants_image.ANTsImage):
            try:
                from nibabel.nifti1 import Nifti1Image
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Could not import nibabel, for conversion to nibabel. Install nibabel with pip install nibabel"
                ) from e
            affine = get_ras_affine(rotation=img.direction, spacing=img.spacing, origin=img.origin)
            return Nifti1Image(img.numpy(), affine, nib.header)

    if convergence is None:
        convergence = {"iters": [50, 50, 50, 50], "tol": 1e-07}
    input_ants = from_nibabel(nib)

    if threshold != 0:
        mask = get_array(nib)
        mask[mask < threshold] = 0
        mask[mask != 0] = 1
        mask = mask.astype(np.uint8)
        mask = dilate_msk(mask, mm=3)
        mask = from_nibabel(set_array(nib, mask))

    out = bc.n4_bias_field_correction(
        input_ants,
        mask=mask,
        shrink_factor=shrink_factor,
        convergence=convergence,
        spline_param=spline_param,
        verbose=verbose,
        weight_mask=weight_mask,
    )
    out_nib = to_nibabel(out)
    if crop:
        # Crop to regions that had a normalization applied. Removes a lot of dead space
        dif = to_nibabel(input_ants - out)
        da = get_array(dif)
        da[da != 0] = 1
        dif = set_array(dif, da)
        ex_slice = compute_crop_slice(dif)
        out_nib = out_nib.slicer[ex_slice]

    return out_nib


buffer_references = {}


def buffer_reference(path: str | Path, bias_field: bool, crop: bool = False) -> np.ndarray | Nifti1Image:
    """Load (and optionally bias-correct) a NIfTI file, caching the result.

    Subsequent calls with the same ``path`` return the cached result without
    re-reading or re-correcting the file.

    Args:
        path: File path of the NIfTI image.
        bias_field: If True, applies N4 bias-field correction before caching.
        crop: Passed to :func:`n4_bias_field_correction` when ``bias_field`` is True.

    Returns:
        The image data array (if ``bias_field`` is False) or the corrected
        :class:`Nifti1Image` (if ``bias_field`` is True).
    """
    if path in buffer_references:
        return buffer_references[path]
    reference = n4_bias_field_correction(nib.load(path), crop) if bias_field else get_array(nib.load(path))  # type: ignore
    buffer_references[path] = reference
    return reference


type_mapping = {
    "float": float,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}


def main(  # noqa: C901
    images: list[str] | list[Path] | list[nib.nifti1.Nifti1Image],
    output: str | None,
    match_histogram: bool = False,
    store_ramp: bool = False,
    verbose: bool = False,
    min_value: float = 0,
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
) -> tuple[Nifti1Image | None, Nifti1Image | None]:
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
        images: Input volumes as file paths or pre-loaded :class:`Nifti1Image`
            objects. At least two are required.
        output: Output file path (``".nii.gz"`` extension is appended if
            absent). If None the result is returned without writing to disk
            (``save`` must also be False).
        match_histogram: If True, matches the histogram of each volume to the
            previous one before stitching.
        store_ramp: If True, also saves the per-volume blend weights as a 4-D
            NIfTI alongside the stitched output.
        verbose: If True, prints progress messages to stdout.
        min_value: Background value (0 for MRI, -1024 for CT). Voxels at or
            below this value are replaced by it in the output.
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

    Returns:
        A 2-tuple ``(stitched_nii, ramp_nii)`` where ``ramp_nii`` is None
        unless ``store_ramp`` is True. Returns ``(None, None)`` when fewer
        than two images are supplied.
    """
    np.set_printoptions(precision=2, floatmode="fixed")
    if is_segmentation:
        bias_field = False
        crop_to_bias_field = False
        min_value = 0
        match_histogram = False
        histogram = None
    if len(images) == 0 or len(images) == 1:
        print("!!! Need at least two images (-i ...nii.gz ...nii.gz) to stitch!!!\n Got " + str(images))
        return None, None
    corners = []
    affines = []
    niis: list[nib.nifti1.Nifti1Image] = []
    print("### loading ###") if verbose else None
    for f_name in images:
        if isinstance(f_name, (Path, str)):
            print("Load ", f_name, Path(f_name)) if verbose else None
            # Load Nii
            nii: nib.nifti1.Nifti1Image = nib.load(f_name)  # type: ignore

        else:
            nii = f_name
        if bias_field:
            nii = n4_bias_field_correction(nii, crop=crop_to_bias_field)
        ## Histogram equalization.
        if match_histogram:
            if histogram is None:
                if len(niis) == 0:
                    reference = None
                else:
                    print("Histogram equalization with previous file") if verbose else None
                    reference = get_array(niis[-1])
            elif histogram.isdigit():
                print("Histogram equalization", images[int(histogram)]) if verbose else None
                reference = buffer_reference(images[int(histogram)], bias_field=bias_field, crop=crop_to_bias_field)  # type: ignore
            else:
                print("Histogram equalization with file", histogram) if verbose else None
                reference = buffer_reference(histogram, bias_field=bias_field, crop=crop_to_bias_field)  # type: ignore
            if reference is not None:
                image = get_array(nii)

                matched = match_histograms(image.astype(float), reference.astype(float))
                matched[matched <= min_value] = min_value
                nii = set_array(nii, matched)

        niis.append(nii)
        # Get affine and points for minimum enclosing Rectangle calculation
        affine = nii.affine
        affines.append(affine)

        corners_current = get_all_corner_points(affine, nii.shape)
        corners.append(corners_current)

    corners_current = np.concatenate(corners, axis=0)

    # compute output shape and affine
    print("### compute output shape and affine ###") if verbose else None
    if is_segmentation:
        max_value = max([x.get_fdata().max() for x in niis])
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
        dtype2 = float
    nii_out = get_max_affine_and_shape(corners_current, affines, min_spacing=min_spacing, dtype=dtype2, verbose=verbose)
    target_list = []
    occupancy_list = []
    # get resampled arrays and occupancy
    print("### resample to new space ###") if verbose else None
    for i, nii in enumerate(niis, 1):
        print(f"{i:2}/{len(niis):2} resampled", end="\r") if verbose else None
        nii_new = nip.resample_from_to(nii, nii_out, 0 if is_segmentation else 3, mode="constant", cval=min_value)
        arr_new = get_array(nii_new)
        target_list.append(arr_new)
        b = nib.Nifti1Image(get_array(nii) * 0 + 1, affine=nii.affine)  # type: ignore
        b = nip.resample_from_to(b, nii_new, 0, cval=0, mode="constant")
        if is_segmentation:
            x = arr_new > 0
            occupancy_list.append((get_array(b) * x.astype(np.int8)).astype(np.float32))  # Keep segmentation if other is 0

        else:
            occupancy_list.append(get_array(b).astype(np.float32))

    print("\n### ramp stitching ###") if verbose else None
    # ramp stitching
    combinations = list(itertools.combinations(range(len(target_list)), 2))
    for idx, item in enumerate(combinations, 1):
        print(f"{idx:2}/{len(combinations):2} ramp stitching", end="\r") if verbose else None
        # TODO fix intersection with more than two occupancies
        arr_1_full = occupancy_list[item[0]]
        arr_2_full = occupancy_list[item[1]]
        ###
        structure = np.ones((ramp_edge_min_value, ramp_edge_min_value, ramp_edge_min_value), dtype=bool)
        arr_1: np.ndarray = arr_1_full.copy()
        arr_2: np.ndarray = arr_2_full.copy()
        overlap = (arr_1 * arr_2) > 0.0
        if overlap.sum() > 0:
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
            arr_1_full = arr_1 / sum_
            arr_2_full = arr_2 / sum_
            if arr_1_full.max() != 1:
                import warnings

                warnings.warn(
                    str((arr_1_full.min(), arr_1_full.max())) + " the image in fully incorporated insight of an other " + str(images),
                    stacklevel=4,
                )
                if kick_out_fully_integrated_images:
                    images.pop(item[0])

            elif arr_2_full.max() != 1:
                import warnings

                warnings.warn(
                    str((arr_2_full.min(), arr_2_full.max())) + " the image in fully incorporated insight of an other " + str(images),
                    stacklevel=4,
                )
                if kick_out_fully_integrated_images:
                    images.pop(item[1])
            if (arr_1_full.max() != 1 or arr_2_full.max() != 1) and kick_out_fully_integrated_images:
                print("kick_out_fully_integrated_images")

                print(images)
                return main(
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
                    save,
                )
            # assert arr_1_full.max() == 1, (arr_1_full.min(), arr_1_full.max())
            # assert arr_2_full.max() == 1, (arr_2_full.min(), arr_2_full.max())
            occupancy_list[item[0]] = arr_1_full
            occupancy_list[item[1]] = arr_2_full
        else:
            continue
    occupancy_arr = np.stack(occupancy_list)
    if is_segmentation:
        occupancy_arr = np.round(occupancy_arr)  # TODO assuming only two intersecting regions
    target_arr = np.stack(target_list) * occupancy_arr
    if is_segmentation:
        target_arr = target_arr.astype(dtype2)
    target_arr = target_arr.sum(0)
    target_arr[target_arr <= min_value] = min_value
    print("\n### Save ###") if verbose else None
    if output is not None:
        output = str(output)
        if not output.endswith(".nii.gz"):
            output = output.replace(".nii", "") + ".nii.gz"
        if "/" not in output and "\\" not in output:
            assert isinstance(images[0], (str, Path)), "automatic path fetching only possible if images are strings or Path, not objects"
            output = str(Path(Path(images[0]).parent, output))
    dtype = type_mapping.get(dtype, dtype)  # type: ignore
    nii_out = set_array(nii_out, target_arr.astype(dtype))
    if bias_field:
        nii_out = n4_bias_field_correction(nii_out)
    if crop_empty:
        nii_occ = set_array(nii_out, occupancy_arr)
        ex_slice = compute_crop_slice(nii_occ)
        nii_out = nii_out.slicer[ex_slice]
    else:
        ex_slice = ()

    nii_out.set_data_dtype(dtype)

    if save:
        nib.save(nii_out, output)  # type: ignore
        print("Saved ", output) if verbose else None

    if store_ramp:
        occupancy_arr = np.stack(occupancy_list, -1)
        if crop_empty:
            occupancy_arr = occupancy_arr[ex_slice]
        assert output is not None
        nii_occ = set_array(nii_out, occupancy_arr)
        nii_occ.set_data_dtype(np.int8)
        output = output.replace(".nii.gz", "_ramps.nii.gz").replace("_msk_", "_") if ramp_path is None else ramp_path
        if save:
            nib.save(nii_occ, output)  # type: ignore
            print("Saved ", output) if verbose else None
        return nii_out, nii_occ
    print("\n### Finished ###") if verbose else None
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
