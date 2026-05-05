import time

import elasticdeform
import numpy as np
from numpy.typing import NDArray

from TPTBox import NII


def deformed_nii(
    nii_dic: dict[str, NII],
    sigma: float | None = None,
    points=None,
    deform_factor=1.0,
    deform_padding=10,
    normalize=True,
    joint_normalize=False,
) -> dict[str, NII]:
    """
    Deform a dictionary of NII objects using random grid deformation. Requires elasticdeform. 'pip install elasticdeform'

    IMPORTANT: Normalize your image data to 0,1. The .seg property of NII shows if this is a segmentation. (NII is form our TPTBox and is a wrapper for nibable)

    This function takes a dictionary of NII objects and applies random grid deformation to each object
    using specified deformation parameters or, if not provided, random parameters generated based on
    the `deform_factor`. The deformed objects are returned as a dictionary.

    Args:
        arr_dic (dict[str, NII]): A dictionary containing NII objects to be deformed.
        sigma (float, optional): The standard deviation of the deformation field. If not provided,
            it will be generated based on the `deform_factor`.
        points (int, optional): The number of control points for the deformation grid. If not provided,
            it will be generated based on the `deform_factor`.
        deform_factor (float, optional): A factor used to determine the deformation parameters if
            `sigma` and `points` are not specified. Larger values result in stronger deformations.
        deform_padding (int, optional): The padding added to the deformed objects to avoid edge artifacts.
        verbose (bool, optional): If True, enable verbose logging. Default is True.

    Returns:
        dict[str, NII]: A dictionary where keys correspond to the input dictionary keys, and values
        correspond to the deformed NII objects.

    Example:
        # Deform a dictionary of NII objects using default deformation parameters
        deformed_data = deformed_NII(arr_dic)

        # Deform a dictionary of NII objects with specific deformation parameters
        sigma = 1.0
        points = 20
        deformed_data = deformed_NII(arr_dic, sigma=sigma, points=points)
    """
    if sigma is None or points is None:
        sigma, points = get_random_deform_parameter(deform_factor=deform_factor)

    print("deformation parameter sigma = ", round(sigma, 4), "; n_points = ", points)
    t = time.time()
    values = list(nii_dic.values())
    # Deform
    if joint_normalize:
        max_v = max([img.max() for img in nii_dic.values() if not img.seg])
        nii_dic = {k: img if img.seg else img.set_dtype(np.float32) / max_v for k, img in nii_dic.items()}
    elif normalize:
        nii_dic = {k: img if img.seg else img.set_dtype(np.float32).normalize() for k, img in nii_dic.items()}
    else:
        nii_dic = {k: img if img.seg else img.set_dtype(np.float32) for k, img in nii_dic.items()}
    assert sigma is not None
    p = deform_padding
    out: list[NDArray] = elasticdeform.deform_random_grid(
        [pad(v.get_array(), p=p) for v in values],
        sigma=sigma,  # type: ignore
        points=points,
        order=[0 if v.seg else 3 for v in values],  # type: ignore
    )
    out2: dict[str, NII] = {}
    for (k, nii), arr in zip(nii_dic.items(), out, strict=True):
        out2[k] = nii.set_array(arr[p:-p, p:-p, p:-p])
    print("Deformation took", round(time.time() - t, 1), "Seconds")
    return out2


def pad(arr, p=10):
    return np.pad(arr, p, mode="reflect")


def get_random_deform_parameter(deform_factor: float = 1):
    """
    Generate random deformation parameters for use in 3D deformation.

    This function generates random values for the deformation parameters, including 'sigma' and 'points',
    based on the specified deformation factor. These parameters are used for 3D deformation operations.

    Args:
        deform_factor (float, optional): A factor to control the strength of deformation. Default is 1.

    Returns:
        tuple[float, int]: A tuple containing the generated 'sigma' (float) and 'points' (int) parameters.

    Example:
        # Generate random deformation parameters with a deformation factor of 1
        sigma, points = get_random_deform_parameter()

        # Generate random deformation parameters with a deformation factor of 2
        sigma, points = get_random_deform_parameter(deform_factor=2)
    """
    sigma = 2 + np.random.uniform() * 2.5  # 1,5 - 4.5
    min_points = 3
    max_points = 17
    if sigma < 2:
        max_points = 17
    elif sigma < 1.7:
        max_points = 16
    elif sigma < 2.1:
        max_points = 15
    elif sigma < 2.3:
        max_points = 14
    elif sigma < 2.5:
        max_points = 13
    elif sigma < 2.6:
        max_points = 12
    elif sigma < 2.7:
        max_points = 11
    elif sigma < 2.8:
        max_points = 10
    elif sigma < 3:
        max_points = 9
    elif sigma < 3.5:
        max_points = 8
    elif sigma < 4.0:
        max_points = 7
    elif sigma < 4.3:
        max_points = 6
    else:
        max_points = 5
    points = np.random.randint(max_points - min_points + 1) + min_points
    # Stronger
    sigma *= deform_factor
    # points *= deform_factor
    points = max(round(points), 1)
    return (sigma, points)
