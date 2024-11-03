import itertools  # noqa: INP001
import os
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np
import SimpleITK as sitk  # noqa: N813
from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import binary_opening, distance_transform_edt
from scipy.spatial import ConvexHull
from skimage.exposure import match_histograms

from TPTBox.core.sitk_utils import nib_to_sitk, sitk_to_nib


def n4_bias_field_correction_nib(
    image: nib.nifti1.Nifti1Image,
    mask: nib.nifti1.Nifti1Image | None = None,
    rescale_intensities: bool = False,
    shrink_factor: int = 4,
    convergence: dict[str, list[int] | float] | None = None,
    spline_param: float | list[float] | None = None,
    return_bias_field: bool = False,
    verbose: bool = False,
    weight_mask: nib.nifti1.Nifti1Image | None = None,
):
    image_ = nib_to_sitk(image)
    mask_ = nib_to_sitk(mask) if mask is not None else None
    weight_mask_ = nib_to_sitk(weight_mask) if weight_mask is not None else None
    out = n4_bias_field_correction_sitk(
        image_, mask_, rescale_intensities, shrink_factor, convergence, spline_param, return_bias_field, verbose, weight_mask_
    )
    if isinstance(out, tuple):
        return sitk_to_nib(out[0]), sitk_to_nib(out[1])
    return sitk_to_nib(out)


def n4_bias_field_correction_sitk(
    image: sitk.Image,
    mask: sitk.Image | None = None,
    rescale_intensities: bool = False,
    shrink_factor: int = 4,
    convergence: dict[str, list[int] | float] | None = None,
    spline_param: float | list[float] | None = None,
    return_bias_field: bool = False,
    verbose: bool = False,
    weight_mask: sitk.Image | None = None,
):
    """
    N4 Bias Field Correction using SimpleITK

    Arguments
    ---------
    image : SimpleITK.Image
        Image to bias correct.

    mask : SimpleITK.Image
        Input mask. If not specified, the entire image is used.

    rescale_intensities : boolean
        If True, rescale intensities to the [min,max] range of the original image intensities
        within the user-specified mask.

    shrink_factor : scalar
        Shrink factor for multi-resolution correction, typically an integer less than 4.

    convergence : dict with keys `iters` and `tol`
        iters : List of maximum number of iterations for each level.
        tol : The convergence tolerance. Default tolerance is 1e-7.

    spline_param : float or list
        Parameter controlling number of control points in spline. Either a single value,
        indicating the spacing in each direction, or a list with one entry per
        dimension of the image, indicating the mesh size. If None, defaults to a spacing of 1.0 in all
        dimensions.

    return_bias_field : boolean
        If True, return the bias field instead of the bias corrected image.

    verbose : boolean
        Enables verbose output.

    weight_mask : SimpleITK.Image (optional)
        Image of weight mask.

    Returns
    -------
    SimpleITK.Image
        Bias corrected image or the estimated bias field if return_bias_field is True.
    """
    if convergence is None:
        convergence = {"iters": [50, 50, 50, 50], "tol": 1e-07}

    if mask is None:
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        mask = sitk.BinaryFillhole(mask)

    if spline_param is None:
        spline_param = [1.0] * image.GetDimension()

    # Apply weight mask if provided
    if weight_mask is not None:
        image = sitk.Mask(image, weight_mask)

    # Rescale intensities if required
    if rescale_intensities:
        original_min = sitk.Minimum(image, mask)
        original_max = sitk.Maximum(image, mask)
        image = sitk.RescaleIntensity(image, outputMinimum=original_min, outputMaximum=original_max)
    if shrink_factor is not None:
        shrink_factor = int(shrink_factor)
        if shrink_factor > 1:
            image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
            mask = sitk.Shrink(mask, [shrink_factor] * image.GetDimension())

    # Perform N4 Bias Field Correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfControlPoints(spline_param)
    corrector.SetMaximumNumberOfIterations(convergence["iters"])
    corrector.SetConvergenceThreshold(convergence["tol"])
    if verbose:
        print("Performing N4 bias field correction...")

    bias_corrected_image = corrector.Execute(image, mask)

    if return_bias_field:
        bias_field = sitk.Divide(image, bias_corrected_image)
        return bias_corrected_image, bias_field

    return bias_corrected_image
