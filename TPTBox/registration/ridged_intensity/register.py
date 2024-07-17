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
    sys.exit()
from TPTBox import AX_CODES, NII

Similarity_Measures = Literal["slr", "mi", "pmi", "dpmi", "cc", "cr", "crl1"]
Affine_Transforms = Literal["affine", "affine2d", "similarity", "similarity2d", "rigid", "rigid2d"]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def registrate_ants(moving: NII, fixed: NII, type_of_transform="DenseRigid", verbose=False, **qargs):
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
):
    if other_moving is None:
        other_moving = []
    hist_reg = nipy_reg.HistogramRegistration(fixed.nii, moving.nii, similarity=similarity)
    with HiddenPrints():
        transform: Affine = hist_reg.optimize(optimizer, iterations=100)
    aligned_img = apply_registration_nipy(moving, fixed, transform)
    out_arr = [apply_registration_nipy(i, fixed, transform) for i in other_moving]
    for out, other in zip(out_arr, other_moving, strict=False):
        out.seg = other.seg
    return aligned_img, transform, out_arr


def only_change_affine(nii: NII, transform: Affine):
    aff = nii.affine
    t_affine = transform.as_affine()
    t_affine = np.dot(t_affine, aff)
    return NII(nib.nifti1.Nifti1Image(nii.get_array(), t_affine), nii.seg)


def apply_registration_nipy(moving: NII, fixed: NII, transform: Affine):
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
    """register an image to an other, with its native resolution of moving. Uses Global coordinates.

    Args:
        moving (NII): _description_
        fixed (NII): _description_
        similarity (Similarity_Measures, optional): _description_. Defaults to "cc".
        optimizer (Affine_Transforms, optional): _description_. Defaults to "rigid".

    Returns:
        (NII,NII): _description_
    """
    if other_moving is None:
        other_moving = []
    fixed_m_res = fixed.copy()
    fixed_m_res.resample_from_to_(moving)
    aligned_img, transform, out_arr = registrate_nipy(moving, fixed_m_res, similarity, optimizer, other_moving)
    return aligned_img, fixed_m_res, transform, out_arr


def crop_shared_(a: NII, b: NII):
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
