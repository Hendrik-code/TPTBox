from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import nibabel as nib

from TPTBox import Print_Logger

try:
    import nipy.algorithms.registration as nipy_reg
    import numpy as np
    from nipy.algorithms.registration.affine import Affine
except ModuleNotFoundError:
    err = Print_Logger()
    err.on_fail("This subscript needs nipy as an additonal package")
    err.on_fail("Please install: pip install nipy")
    raise
from typing_extensions import Self

from TPTBox import AX_CODES, NII

"""
Wrapper functions for different registration methods with ants and nipy.
"""
Similarity_Measures = Literal["slr", "mi", "pmi", "dpmi", "cc", "cr", "crl1"]
Affine_Transforms = Literal["affine", "affine2d", "similarity", "similarity2d", "rigid", "rigid2d"]


class HiddenPrints:
    """Context manager that suppresses all stdout output while active."""

    def __enter__(self) -> Self:
        """Redirect stdout to /dev/null."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore the original stdout."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


def registrate_ants(
    moving: NII,
    fixed: NII,
    type_of_transform: str = "DenseRigid",
    verbose: bool = False,
    **qargs,
) -> tuple[NII, list]:
    """Register *moving* to *fixed* using ANTs and return the warped image plus forward transforms.

    Args:
        moving: Moving (source) image.
        fixed: Fixed (target/reference) image.
        type_of_transform: ANTs transform type string (e.g. ``"DenseRigid"``, ``"SyN"``).
        verbose: If True, pass verbose output through ANTs.
        **qargs: Additional keyword arguments forwarded to ``ants.registration``.

    Returns:
        A tuple of (warped_moving_NII, forward_transform_paths).
    """
    import ants

    mytx = ants.registration(fixed=fixed.to_ants(), moving=moving.to_ants(), type_of_transform=type_of_transform, verbose=verbose, **qargs)

    warped_moving = mytx["warpedmovout"]
    print(mytx)
    return NII(ants.to_nibabel(warped_moving)), mytx["fwdtransforms"]


def registrate_nipy(
    moving: NII,
    fixed: NII,
    similarity: Similarity_Measures = "cc",
    optimizer: Affine_Transforms = "rigid",
    other_moving: list[NII] | None = None,
) -> tuple[NII, Affine, list[NII]]:
    """Register *moving* to *fixed* using nipy's histogram-based registration.

    Args:
        moving: Moving (source) image.
        fixed: Fixed (target/reference) image.
        similarity: Similarity metric used for histogram registration.  One of
            ``"slr"``, ``"mi"``, ``"pmi"``, ``"dpmi"``, ``"cc"``, ``"cr"``,
            ``"crl1"``.
        optimizer: Affine transform family to optimise over.  One of
            ``"affine"``, ``"affine2d"``, ``"similarity"``, ``"similarity2d"``,
            ``"rigid"``, ``"rigid2d"``.
        other_moving: Additional images to warp with the same transform.

    Returns:
        A tuple of (aligned_moving, transform, aligned_other_moving).
    """
    if other_moving is None:
        other_moving = []
    hist_reg = nipy_reg.HistogramRegistration(fixed.nii, moving.nii, similarity=similarity)
    with HiddenPrints():
        transform: Affine = hist_reg.optimize(optimizer, iterations=100)
    aligned_img = apply_registration_nipy(moving, fixed, transform)
    out_arr = [apply_registration_nipy(i, fixed, transform) for i in other_moving]
    for out, other in zip(out_arr, other_moving):
        out.seg = other.seg
    return aligned_img, transform, out_arr


def only_change_affine(nii: NII, transform: Affine) -> NII:
    """Return a copy of *nii* whose affine matrix has been updated by *transform*.

    Args:
        nii: Source NIfTI image.
        transform: nipy affine transform to compose with the existing affine.

    Returns:
        New ``NII`` with the modified affine and the original voxel data.
    """
    aff = nii.affine
    t_affine = transform.as_affine()
    t_affine = np.dot(t_affine, aff)
    return NII(nib.nifti1.Nifti1Image(nii.get_array(), t_affine), nii.seg)


def apply_registration_nipy(moving: NII, fixed: NII, transform: Affine) -> NII:
    """Apply a pre-computed nipy affine transform to resample *moving* into the *fixed* space.

    Args:
        moving: Moving (source) image to be resampled.
        fixed: Fixed (target/reference) image that defines the output space.
        transform: Pre-computed nipy affine transform.

    Returns:
        Resampled ``NII`` in the fixed image space.
    """
    aligned_img = nipy_reg.resample(moving.nii, transform, fixed.nii, interp_order=0 if moving.seg else 3)
    aligned_img = fixed.set_array(aligned_img.get_data())
    aligned_img.seg = moving.seg
    return aligned_img


def register_native_res(
    moving: NII,
    fixed: NII,
    similarity: Similarity_Measures = "cc",
    optimizer: Affine_Transforms = "rigid",
    other_moving: list[NII] | None = None,
) -> tuple[NII, NII, Affine, list[NII]]:
    """Register *moving* to *fixed* at the native resolution of *moving*.

    The fixed image is first resampled into *moving*'s voxel space before the
    histogram registration is performed, so the alignment uses global (physical)
    coordinates throughout.

    Args:
        moving: Moving (source) image.
        fixed: Fixed (target/reference) image.
        similarity: Similarity metric for histogram registration.  One of
            ``"slr"``, ``"mi"``, ``"pmi"``, ``"dpmi"``, ``"cc"``, ``"cr"``,
            ``"crl1"``.
        optimizer: Affine transform family to optimise.  One of
            ``"affine"``, ``"affine2d"``, ``"similarity"``, ``"similarity2d"``,
            ``"rigid"``, ``"rigid2d"``.
        other_moving: Additional images to warp with the same transform.

    Returns:
        A tuple of (aligned_moving, fixed_resampled_to_moving, transform,
        aligned_other_moving).
    """
    if other_moving is None:
        other_moving = []
    fixed_m_res = fixed.copy()
    fixed_m_res.resample_from_to_(moving)
    aligned_img, transform, out_arr = registrate_nipy(moving, fixed_m_res, similarity, optimizer, other_moving)
    return aligned_img, fixed_m_res, transform, out_arr


def crop_shared_(a: NII, b: NII) -> tuple:
    """Crop both images in-place to their shared non-background bounding box.

    Args:
        a: First NIfTI image (modified in place).
        b: Second NIfTI image (modified in place).

    Returns:
        The crop slice tuple applied to both images.
    """
    crop = a.compute_crop()
    crop = b.compute_crop(other_crop=crop)
    print(crop)
    a.apply_crop_(crop)
    b.apply_crop_(crop)
    return crop


if __name__ == "__main__":
    p = "/media/data/new_NAKO/NAKO/MRT/rawdata/105/sub-105013/"
    moving = NII.load(Path(p, "t1dixon", "sub-105013_acq-ax_rec-in_chunk-2_t1dixon.nii.gz"), False)
    fixed = NII.load(Path(p, "T2w", "sub-105013_acq-sag_chunk-LWS_sequ-31_T2w.nii.gz"), False)
    fixed.resample_from_to_(moving)
    # fixed.save("fixed_rep.nii.gz")
    aligned_img = registrate_nipy(moving, fixed)
