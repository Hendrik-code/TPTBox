from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813

if TYPE_CHECKING:  # pragma: no cover
    from nibabel import Nifti1Image

    from TPTBox import NII, POI


def nii_to_sitk(nii: NII) -> sitk.Image:
    """Create a SimpleITK image from a TPTBox NII wrapper.

    Args:
        nii: Source ``NII`` object.

    Returns:
        A ``SimpleITK.Image`` with origin, spacing, and direction set from the
        NIfTI affine.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L289
    return nib_to_sitk(nii.nii)


def nib_to_sitk(nii: Nifti1Image) -> sitk.Image:
    """Create a SimpleITK image from a nibabel ``Nifti1Image``.

    Transposes the voxel array from nibabel's (X, Y, Z) memory layout to
    SimpleITK's (Z, Y, X) convention and computes origin, spacing, and direction
    from the RAS affine matrix.

    Args:
        nii: Source nibabel ``Nifti1Image``.

    Returns:
        A 3-D ``SimpleITK.Image`` with metadata derived from the NIfTI affine.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L289
    array = np.asarray(nii.dataobj)
    affine = np.asarray(nii.affine).astype(np.float64)
    assert array.ndim == 3
    array = array.transpose()
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(float)

    image = sitk.GetImageFromArray(array, isVector=False)  # isVector = True, First dimension is used in parallel
    origin, spacing, direction = get_sitk_metadata_from_ras_affine(affine)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    assert len(image.GetSize()) == 3
    return image


###########################################################################################
# Functions are simplified from https://github.com/fepegar/torchio
def sitk_to_nib(image: sitk.Image) -> Nifti1Image:
    """Convert a SimpleITK image to a nibabel ``Nifti1Image``.

    Transposes the voxel array from SimpleITK's (Z, Y, X) convention back to
    nibabel's (X, Y, Z) layout and reconstructs the RAS affine.

    Args:
        image: Source 3-D ``SimpleITK.Image``.

    Returns:
        A nibabel ``Nifti1Image`` with the corresponding RAS affine matrix.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L332
    data = sitk.GetArrayFromImage(image).transpose()
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert image.GetDimension() == 3
    affine = get_ras_affine_from_sitk(image)
    import nibabel

    return nibabel.Nifti1Image(data, affine)


def sitk_to_nii(image: sitk.Image, seg: bool) -> NII:
    """Convert a SimpleITK image to a TPTBox ``NII`` wrapper.

    Args:
        image: Source 3-D ``SimpleITK.Image``.
        seg: Set to ``True`` if the image is a segmentation label map.

    Returns:
        A ``NII`` instance wrapping the converted image.
    """
    import TPTBox

    return TPTBox.NII(sitk_to_nib(image), seg)


def transform_centroid(ctd: POI, transform: sitk.Transform, img_fixed: sitk.Image, img_moving: sitk.Image, reg_type: str) -> POI:
    """Apply a SimpleITK registration transform to a set of POI centroids.

    For deformable registration the transform is applied directly in world
    coordinates.  For other registration types (e.g. rigid or affine) the point
    is first mapped from moving-image voxel space to world space, then through
    the inverse transform, and finally back to fixed-image voxel space.

    Args:
        ctd: Input POI object whose centroid coordinates (in voxel space) are to
            be transformed.
        transform: Fitted SimpleITK spatial transform.
        img_fixed: Fixed reference SimpleITK image (defines the output voxel grid).
        img_moving: Moving SimpleITK image (defines the input voxel-to-world mapping).
        reg_type: Registration type string; use ``"deformable"`` for deformable
            transforms and any other value for rigid/affine transforms.

    Returns:
        A new ``POI`` object resampled to the fixed-image grid with transformed
        centroid coordinates.
    """
    import TPTBox

    out = TPTBox.core.poi.POI_Descriptor()

    if reg_type == "deformable":
        for key, key2, (x, y, z) in ctd.items():
            ctr_b = transform.TransformPoint((x, y, z))
            out[key, key2] = ctr_b
            out[key, key2] = ctr_b
    else:
        for key, key2, (x, y, z) in ctd.items():
            ctr_b = img_moving.TransformContinuousIndexToPhysicalPoint((x, y, z))
            ctr_b = transform.GetInverse().TransformPoint(ctr_b)
            ctr_b = img_fixed.TransformPhysicalPointToContinuousIndex(ctr_b)
            out[key, key2] = ctr_b
    nii = sitk_to_nii(img_fixed, True)
    return nii.get_empty_POI(out)


def get_sitk_metadata_from_ras_affine(affine: np.ndarray) -> tuple[tuple, tuple, tuple]:
    """Decompose a NIfTI RAS affine into SimpleITK metadata components.

    SimpleITK uses LPS coordinates, so the origin and direction are converted
    from RAS before being returned.

    Args:
        affine: 4x4 homogeneous RAS affine matrix.

    Returns:
        A 3-tuple ``(origin, spacing, direction)`` where *origin* and *direction*
        are LPS tuples and *spacing* is an isotropic voxel-size tuple — all
        suitable for direct use with ``sitk.Image.SetOrigin`` etc.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L385
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    origin_array = np.dot(np.diag([-1, -1, 1]), origin_ras)
    direction_array = np.dot(np.diag([-1, -1, 1]), direction_ras).flatten()
    direction = tuple(direction_array)
    origin = tuple(origin_array)
    spacing = tuple(spacing_array)
    return origin, spacing, direction


def get_rotation_and_spacing_from_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 4x4 affine into a unit rotation matrix and per-axis voxel spacing.

    Args:
        affine: 4x4 homogeneous affine matrix.

    Returns:
        A 2-tuple ``(rotation, spacing)`` where *rotation* is the 3x3 unit-norm
        direction matrix and *spacing* is a 1-D array of voxel sizes in mm.
    """
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def get_ras_affine_from_sitk(sitk_object: sitk.Image | sitk.ImageFileReader) -> np.ndarray:
    """Build a NIfTI RAS affine matrix from a SimpleITK image or file reader.

    Handles 2-D (4-element direction), 3-D (9-element direction), and
    degenerate 4-D (16-element direction) images.

    Args:
        sitk_object: A ``SimpleITK.Image`` or ``SimpleITK.ImageFileReader``
            with spatial metadata set.

    Returns:
        A 4x4 homogeneous RAS affine matrix as a ``numpy.ndarray``.

    Raises:
        NotImplementedError: If the direction cosine array length is not 4, 9,
            or 16.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    spacing = np.array(sitk_object.GetSpacing())
    direction_lps = np.array(sitk_object.GetDirection())
    origin_lps = np.array(sitk_object.GetOrigin())
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise NotImplementedError()
    rotation_ras = np.dot(np.diag([-1, -1, 1]), rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(np.diag([-1, -1, 1]), origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine


def get_ras_affine_from_sitk_meta(
    spacing: np.ndarray | tuple,
    direction_lps: np.ndarray | tuple,
    origin_lps: np.ndarray | tuple,
) -> np.ndarray:
    """Build a NIfTI RAS affine matrix from explicit SimpleITK metadata values.

    Accepts the individual spacing, direction-cosine, and origin values rather
    than a ``SimpleITK.Image`` object.  Handles the same dimensionality edge
    cases as :func:`get_ras_affine_from_sitk`.

    Args:
        spacing: Per-axis voxel sizes (length 2, 3, or 4 depending on image
            dimensionality).
        direction_lps: Flattened LPS direction cosine matrix (length 4, 9, or 16).
        origin_lps: LPS physical origin of the image (length 2, 3, or 4).

    Returns:
        A 4x4 homogeneous RAS affine matrix as a ``numpy.ndarray``.

    Raises:
        NotImplementedError: If the direction cosine array length is not 4, 9,
            or 16.
    """
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    spacing = np.array(spacing)
    direction_lps = np.array(direction_lps)
    origin_lps = np.array(origin_lps)
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise NotImplementedError()
    rotation_ras = np.dot(np.diag([-1, -1, 1]), rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(np.diag([-1, -1, 1]), origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine
