from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image


def nifti_to_ants(nib_image: Nifti1Image, **args) -> object:
    """Convert a NiBabel NIfTI image to an ANTsPy image.

    First attempts to use :func:`ants.utils.from_nibabel_nifti`. Falls back to
    manually constructing the ANTs image from the q-form affine when the built-in
    conversion raises an exception (e.g. for non-3D images or unusual metadata).

    Args:
        nib_image: The NiBabel NIfTI image to convert.
        **args: Additional keyword arguments forwarded to
            :func:`ants.utils.from_nibabel_nifti` when available.

    Returns:
        The converted ``ants.ANTsImage``.

    Raises:
        NotImplementedError: If the image has fewer than 3 spatial dimensions.
    """
    import ants

    try:
        return ants.utils.from_nibabel_nifti(nib_image, **args)
    except Exception:
        pass
    ndim = nib_image.ndim

    if ndim < 3:
        raise NotImplementedError("Conversion is only implemented for 3D or higher images.")
    q_form = nib_image.get_qform()
    spacing = nib_image.header["pixdim"][1 : ndim + 1]

    origin = np.zeros(ndim)
    origin[:3] = np.dot(np.diag([-1, -1, 1]), q_form[:3, 3])

    direction = np.eye(ndim)
    direction[:3, :3] = np.dot(np.diag([-1, -1, 1]), q_form[:3, :3]) / spacing[:3]

    ants_img = ants.from_numpy(
        data=nib_image.get_fdata(),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        direction=direction,
    )
    "add nibabel conversion (lacey import to prevent forced dependency)"

    return ants_img


def get_ras_affine_from_ants(ants_img) -> np.ndarray:
    """Convert an ANTs image affine matrix to the RAS coordinate system.

    Adapted from
    https://github.com/fepegar/torchio/blob/main/src/torchio/data/io.py.
    Handles 3-D (direction length 9), 2-D (direction length 4), and
    degenerate 4-D (direction length 16) cases.

    Args:
        ants_img: An ``ants.ANTsImage`` whose LPS-convention affine is to be
            converted.

    Returns:
        A ``(4, 4)`` NumPy affine matrix in RAS coordinates compatible with
        NiBabel conventions.

    Raises:
        NotImplementedError: If the direction matrix has an unexpected number
            of elements.
    """
    spacing = np.array(ants_img.spacing)
    direction_lps = np.array(ants_img.direction)
    origin_lps = np.array(ants_img.origin)
    direction_length = direction_lps.shape[0] * direction_lps.shape[1]
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # 2D case (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # Fix potential bad NIfTI
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise NotImplementedError(f"Unexpected direction length = {direction_length}.")

    rotation_ras = np.dot(np.diag([-1, -1, 1]), rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(np.diag([-1, -1, 1]), origin_lps)

    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras

    return affine


def ants_to_nifti(img, header=None) -> Nifti1Image:
    """Convert an ANTsPy image to a NiBabel NIfTI image.

    First attempts to use :func:`ants.utils.to_nibabel_nifti`. Falls back to
    manually building a :class:`~nibabel.nifti1.Nifti1Image` from the RAS affine
    when the built-in conversion raises an exception.

    Args:
        img: An ``ants.ANTsImage`` to convert.
        header: Optional :class:`~nibabel.nifti1.Nifti1Header` to attach to the
            output image. The data dtype is updated to match the array dtype when
            provided.

    Returns:
        The converted :class:`~nibabel.nifti1.Nifti1Image`.
    """
    import ants

    try:
        return ants.utils.to_nibabel_nifti(img, header=header)
    except Exception:
        pass
    from nibabel.nifti1 import Nifti1Image

    affine = get_ras_affine_from_ants(img)
    arr = img.numpy()

    if header is not None:
        header.set_data_dtype(arr.dtype)

    return Nifti1Image(arr, affine, header)


# Legacy names for backwards compatibility
from_nibabel = nifti_to_ants
to_nibabel = ants_to_nifti

if __name__ == "__main__":
    import ants
    import nibabel as nib

    fn = ants.get_ants_data("mni")
    ants_img = ants.image_read(fn)
    nii_mni: Nifti1Image = nib.load(fn)
    ants_mni = to_nibabel(ants_img)
    assert (ants_mni.get_qform() == nii_mni.get_qform()).all()
    assert (ants_mni.affine == nii_mni.affine).all()
    temp = from_nibabel(nii_mni)

    assert ants.image_physical_space_consistency(ants_img, temp)

    fn = ants.get_data("ch2")
    ants_mni = ants.image_read(fn)
    nii_mni = nib.load(fn)
    ants_mni = to_nibabel(ants_mni)
    assert (ants_mni.get_qform() == nii_mni.get_qform()).all()

    nii_org = nib.load(fn)
    ants_org = ants.image_read(fn)
    temp = ants_org
    for _ in range(10):
        temp = to_nibabel(ants_org)
        assert (temp.get_qform() == nii_org.get_qform()).all()
        assert (ants_mni.affine == nii_mni.affine).all()
        temp = from_nibabel(temp)
        assert ants.image_physical_space_consistency(ants_org, temp)
    for _ in range(10):
        temp = from_nibabel(nii_org)
        assert ants.image_physical_space_consistency(ants_org, temp)
        temp = to_nibabel(temp)

        assert (temp.get_qform() == nii_org.get_qform()).all()
        assert (ants_mni.affine == nii_mni.affine).all()
