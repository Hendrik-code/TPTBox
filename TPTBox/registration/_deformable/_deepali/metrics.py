from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class NMILOSS(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(
        self,
        intensity_range: tuple[float, float] | None = None,
        nbins: int = 32,
        sigma: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss(fixed, warped, bins_fixed, bins_warped, sigma=self.sigma).mean()


def nmi_gauss(x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of similar shape"

    def gaussian_window(x, bins, sigma):
        assert x.ndim == 2, "Input tensor should be 2-dimensional."
        return torch.exp(-((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)

    x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
    x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
    p_xy = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
    p_xy = p_xy + e  # deal with numerical instability

    p_xy = p_xy / p_xy.sum((1, 2))[:, None, None]

    p_x = p_xy.sum(1)
    p_y = p_xy.sum(2)

    i = (p_xy * torch.log(p_xy / (p_x[:, None] * p_y[:, :, None]))).sum((1, 2))

    marg_ent_0 = (p_x * torch.log(p_x)).sum(1)
    marg_ent_1 = (p_y * torch.log(p_y)).sum(1)

    normalized = -1 * 2 * i / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


def calculate_dice(mask1, mask2, label_class=0):
    """
    from https://github.com/voxelmorph/
    Dice score of a specified class between two label masks.
    (classes are encoded but by label class number not one-hot )

    Args:
        mask1: (numpy.array, shape (N, 1, *sizes)) segmentation mask 1
        mask2: (numpy.array, shape (N, 1, *sizes)) segmentation mask 2
        label_class: (int or float)

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).astype(np.float32)
    mask2_pos = (mask2 == label_class).astype(np.float32)

    assert mask1.ndim == mask2.ndim
    axes = tuple(range(2, mask1.ndim))
    pos1and2 = np.sum(mask1_pos * mask2_pos, axis=axes)
    pos1 = np.sum(mask1_pos, axis=axes)
    pos2 = np.sum(mask2_pos, axis=axes)
    return np.mean(2 * pos1and2 / (pos1 + pos2 + 1e-7))


def dice(a, b, label):
    return calculate_dice(a, b, label_class=label)


def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.
    from https://github.com/voxelmorph/

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    folding_ratio = []
    mag_grad_jac_det = []
    for n in range(disp.shape[0]):
        disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
        jac_det_n = calculate_jacobian_det(disp_n)
        folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
        mag_grad_jac_det += [np.abs(np.gradient(jac_det_n)).mean()]
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)


def calculate_jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)
    from https://github.com/voxelmorph/

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det
