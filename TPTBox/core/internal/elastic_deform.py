import time

import numpy as np
from numpy.typing import NDArray

from TPTBox import NII


def _elasticdeform_install_hint() -> str:
    """Build an install hint tailored to the currently active NumPy version.

    The PyPI ``elasticdeform`` wheel is compiled against NumPy 1.x and fails
    to import under NumPy 2.x. The fix is to install from source so the C
    extension is rebuilt against whatever NumPy is available in the current
    environment. Reporting the detected NumPy version makes it obvious to the
    reader why the wheel broke and lets us suggest an install line that
    references the concrete environment they are on.

    See https://github.com/gvtulder/elasticdeform/issues/24 for context.
    """
    numpy_version = np.__version__
    numpy_major = int(numpy_version.split(".", 1)[0]) if numpy_version[:1].isdigit() else 0
    tarball = "https://github.com/gvtulder/elasticdeform/archive/refs/tags/v0.5.1.tar.gz"
    if numpy_major >= 2:
        return (
            f"elasticdeform could not be imported (NumPy {numpy_version} detected). "
            "The published wheel is built against NumPy 1.x and cannot load under "
            "NumPy 2.x. Rebuild from source against your current NumPy with:\n"
            f"    pip install --no-binary :all: --no-build-isolation --force-reinstall --no-deps {tarball}\n"
            "See https://github.com/gvtulder/elasticdeform/issues/24 for details."
        )
    return f"elasticdeform could not be imported (NumPy {numpy_version} detected). Install it with:\n    pip install elasticdeform\n"


try:
    import elasticdeform  # noqa: E402 - kept after the helper so the message can be built
except ImportError as _exc:
    raise ImportError(_elasticdeform_install_hint()) from _exc


def deformed_nii(
    nii_dic: dict[str, NII],
    sigma: float | None = None,
    points=None,
    deform_factor=1.0,
    deform_padding=10,
    normalize=True,
    joint_normalize=False,
) -> dict[str, NII]:
    """Deform a dictionary of NII objects using random grid deformation (requires ``pip install elasticdeform``).

    IMPORTANT: Normalize your image data to 0,1. The .seg property of NII shows if this is a segmentation. (NII is form our TPTBox and is a wrapper for nibable)

    This function takes a dictionary of NII objects and applies random grid deformation to each object
    using specified deformation parameters or, if not provided, random parameters generated based on
    the `deform_factor`. The deformed objects are returned as a dictionary.

    Args:
        nii_dic (dict[str, NII]): A dictionary containing NII objects to be deformed.
        sigma (float, optional): The standard deviation of the deformation field. If not provided,
            it will be generated based on the `deform_factor`.
        points (int, optional): The number of control points for the deformation grid. If not provided,
            it will be generated based on the `deform_factor`.
        deform_factor (float, optional): A factor used to determine the deformation parameters if
            `sigma` and `points` are not specified. Larger values result in stronger deformations.
        deform_padding (int, optional): The padding added to the deformed objects to avoid edge artifacts.
        normalize (bool, optional): If True, per-entry normalise non-segmentation images to [0, 1] before
            deforming and re-scale them back afterwards. Ignored when ``joint_normalize`` is True. Defaults to True.
        joint_normalize (bool, optional): If True, use a single shared max across all non-segmentation
            images for normalisation instead of per-entry min/max. Defaults to False.

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
        np.random.seed(None)
        sigma, points = get_random_deform_parameter(deform_factor=deform_factor)
    print("deformation parameter sigma = ", round(sigma, 4), "; n_points = ", points)
    t = time.time()

    # Deform
    max_v = None
    if joint_normalize:
        max_v = max([img.max() for img in nii_dic.values() if not img.seg])
        nii_dic = {k: img if img.seg else img.set_dtype(np.float32) / max_v for k, img in nii_dic.items()}
    elif normalize:
        max_v = {k: None if img.seg else (float(max(img.max() - img.min(), 1)), float(img.min())) for k, img in nii_dic.items()}
        nii_dic = {k: img if img.seg else (img.set_dtype(np.float32) - max_v[k][1]) / max_v[k][0] for k, img in nii_dic.items()}
    else:
        nii_dic = {k: img if img.seg else img.set_dtype(np.float32) for k, img in nii_dic.items()}

    values = list(nii_dic.values())
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
    if joint_normalize:
        out2 = {k: img if img.seg else img.set_dtype(np.float32) * max_v for k, img in out2.items()}
    elif normalize:
        out2 = {k: img if img.seg else ((img.set_dtype(np.float32) * max_v[k][0]) + max_v[k][1]) for k, img in out2.items()}
    return out2


def pad(arr: np.ndarray, p: int = 10) -> np.ndarray:
    """Reflect-pad a 3-D array by ``p`` voxels on every side."""
    return np.pad(arr, p, mode="reflect")


def get_random_deform_parameter(deform_factor: float = 1) -> tuple[float, int]:
    """Generate random deformation parameters for use in 3D deformation.

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
