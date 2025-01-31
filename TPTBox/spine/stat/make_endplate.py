from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import ball, binary_dilation, binary_erosion, disk
from sklearn.cluster import KMeans

from TPTBox import NII
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun.vertebra_direction import get_direction
from TPTBox.core.vert_constants import Location, Vertebra_Instance

"""
Author: Amirhossein Bayat
amir.bayat@tum.de
"""
vertebra_body = (
    49,
    50,
    Location.Endplate.value,
    Location.Vertebral_Body_Endplate_Inferior.value,
    Location.Vertebral_Body_Endplate_Superior.value,
)


def _dilate_erode_special(np_array: np.ndarray, ball_size=3, normal: None | np.ndarray = None) -> np.ndarray:
    if normal is None:
        struct = ball(ball_size)
    else:
        d = disk(ball_size)
        struct = np.stack([d, d, d], axis=np.argmax(normal))
    # print(struct)
    np_array = binary_dilation(np_array, footprint=struct)
    np_array = binary_erosion(np_array, footprint=struct)
    return np_array


def _get_largest_CC(segmentation: np.ndarray) -> np.ndarray:
    """
    Extracts the largest connected component from a binary segmentation.

    Parameters:
    - segmentation (np.ndarray): Input binary segmentation array.

    Returns:
    - np.ndarray: Binary array of the largest connected component.
    """
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)  # type: ignore
    largest_label = unique[1:][np.argmax(counts[1:])]  # Ignore the background label (0)
    return (labels == largest_label).astype(int)


def _get_endplate(body, mult, axis=1):
    """
    calculating the endplates using projection method
    mult variable is a 3D numpy array of indices with the same size as body,
    descending and ascending indices determents wether we are extracting the
    upper or lower endplate.
    """
    body = body * mult

    indices = np.argmax(body, axis=axis)
    k_means = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(indices.reshape(-1, 1))
    mask = k_means.labels_.reshape(indices.shape[0], indices.shape[1])
    mask += 1

    mean_1 = np.mean([mask == 1] * indices)
    mean_2 = np.mean([mask == 2] * indices)
    mean_3 = np.mean([mask == 3] * indices)
    if np.abs(mean_1 - mean_2) < 1:
        mask[mask == 2] = 1
    if np.abs(mean_1 - mean_3) < 1:
        mask[mask == 3] = 1
    if np.abs(mean_2 - mean_3) < 1:
        mask[mask == 3] = 2

    argmax_ind = np.argmax([np.mean([mask == 1] * indices), np.mean([mask == 2] * indices), np.mean([mask == 3] * indices)]) + 1
    mask[mask != argmax_ind] = 0
    mask[mask != 0] = 1

    mask = mask * indices
    mask[mask == 0] = -1

    if axis == 0:
        tmp = (np.arange(body.shape[0]) == mask[..., None]).astype(int)
        tmp = np.transpose(tmp, (2, 0, 1))
    elif axis == 2:
        tmp = (np.arange(body.shape[2]) == mask[..., None]).astype(int)
    elif axis == 1:
        tmp = (np.arange(body.shape[1]) == mask[..., None]).astype(int)
        tmp = np.swapaxes(tmp, 1, 2)
    else:
        raise ValueError(axis)
    return tmp


def _extract_endplate_np(body: np.ndarray, projected: np.ndarray, normal: np.ndarray, lower: bool = False) -> np.ndarray:
    """
    Extracts the upper or lower endplate of the given body.

    Parameters:
    - body (np.ndarray): 3D binary array representing the segmented body.
    - lower (bool): If True, extracts the lower endplate. Otherwise, extracts the upper endplate.

    Returns:
    - np.ndarray: Binary mask representing the extracted endplate.
    """
    # Adjust projection for lower endplate if needed
    if lower:
        projected = projected.max() + 1 - projected

    endplate_mask = _get_endplate(body, projected)
    endplate_mask = _get_largest_CC(_dilate_erode_special(endplate_mask, ball_size=3, normal=normal)) * np.clip(endplate_mask, 0, 1)
    return endplate_mask


def _endplate_extraction_msk(subreg_tmp: NII, normal: np.ndarray, _extract=True) -> NII:
    """
    Extracts upper and lower endplates from the given NII object and labels them.

    Parameters:
    - subreg_tmp (NII): Input NII object containing the body segmentation.

    Returns:
    - NII: Updated NII object with labeled endplates.
    """
    out_arr = np.zeros_like(subreg_tmp.get_array())
    if _extract:
        subreg_tmp = subreg_tmp.extract_label(vertebra_body)
    body = subreg_tmp.get_array()

    # Compute the projection along the normal vector
    grid = np.mgrid[0 : body.shape[0], 0 : body.shape[1], 0 : body.shape[2]]
    projected = np.tensordot(grid, normal, axes=(0, 0))  # type: ignore
    # Normalize and convert to integers starting from 1
    projected -= projected.min()  # Shift to start from 0
    projected = projected / projected.max() * (body.shape[1] - 1)  # Scale to body dimensions
    projected = np.round(projected).astype(int) + 1  # Convert to integers starting from 1

    out_arr[_extract_endplate_np(body, projected, normal) == 1] = Location.Vertebral_Body_Endplate_Inferior.value
    out_arr[_extract_endplate_np(body, projected, normal, lower=True) == 1] = Location.Vertebral_Body_Endplate_Superior.value

    return subreg_tmp.set_array(out_arr)


def endplate_extraction(idx, vert: NII, subreg: NII, poi: POI) -> NII | None:
    """
    Retrospectively computes an endplate to a vertebra (Two-Sided)

    Returns:
    - NII: endplates labeled 52, 53.
    """
    if isinstance(idx, Enum):
        idx = idx.value
    if Vertebra_Instance(idx) in Vertebra_Instance.sacrum()[1:]:
        return None

    vert_ = vert.reorient(verbose=False)
    subreg = subreg.reorient(verbose=False)

    out = vert_.extract_label(idx) * subreg.extract_label(vertebra_body, keep_label=True)
    crop = out.compute_crop(dist=3)
    out_c = out.apply_crop(crop)
    try:
        normal = get_direction("S", poi, vert_id=idx) / np.array(poi.zoom)
    except KeyError:
        return None
    out_c = _endplate_extraction_msk(out_c, normal, _extract=False)
    out[crop] = out_c
    out.reorient_(vert.orientation, verbose=False)
    return out


if __name__ == "__main__":
    from TPTBox import Location, calc_poi_from_subreg_vert
    from TPTBox.tests.test_utils import get_test_ct, get_tests_dir

    ct, subreg, vert, idx = get_test_ct()
    vert.reorient_()
    subreg.reorient_()
    poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=Location.Vertebra_Direction_Posterior)
    subreg2 = endplate_extraction(idx, vert, subreg, poi)
    assert subreg2 is not None
    subreg2.reorient_(ct.orientation)
    subreg2.save(get_tests_dir() / "sample_ct" / "endplate-test.nii.gz")
