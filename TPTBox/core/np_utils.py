import itertools
import warnings
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import scipy
from cc3d import (
    connected_components,  # pip install connected-components-3d
    contacts,
    region_graph,
    voxel_connectivity_graph,
)
from cc3d import statistics as _cc3dstats
from fill_voids import fill
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, center_of_mass, generate_binary_structure
from skimage.measure import euler_number, label

from TPTBox.core.vert_constants import COORDINATE, LABEL_MAP, LABEL_REFERENCE

UINT = TypeVar("UINT", bound=np.unsignedinteger[Any])
INT = TypeVar("INT", bound=np.signedinteger[Any])
UINTARRAY = NDArray[UINT]
INTARRAY = UINTARRAY | NDArray[INT]


def np_extract_label(arr: np.ndarray, label: int, to_label: int = 1, inplace: bool = True) -> np.ndarray:
    """Extracts a label from an given arr (works with zero as well!)

    Args:
        arr (np.ndarray): input arr
        label (int): label to be extracted (all other values are set to zero, label will be set to one, even if label==0!)
        to_label (int): the value of the entries that had the <label> value. Defaults to 1.
        inplace (bool, optional): If False, will make a copy of the arr. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    if to_label == 0:
        warnings.warn("np_extract_label: to_label is zero, this can have unforeseen consequences!", UserWarning, stacklevel=4)
    if not inplace:
        arr = arr.copy()

    if label != 0:
        arr[arr != label] = 0
        arr[arr == label] = to_label
        return arr
    # label == 0
    arr[arr != 0] = to_label + 1
    arr[arr == 0] = to_label
    arr[arr != to_label] = 0
    return arr


def cc3dstatistics(arr: UINTARRAY) -> dict:
    assert np.issubdtype(arr.dtype, np.unsignedinteger), f"cc3dstatistics expects uint type, got {arr.dtype}"
    return _cc3dstats(arr)


def np_volume(arr: UINTARRAY, include_zero: bool = False) -> dict[int, int]:
    """Returns a dictionary mapping array label to voxel_count (not including zero!)

    Args:
        arr (np.ndarray): _description_

    Returns:
        dict[int, int]: _description_
    """
    if include_zero:
        return {idx: i for idx, i in dict(enumerate(cc3dstatistics(arr)["voxel_counts"])).items() if i > 0}
    else:
        return {idx: i for idx, i in dict(enumerate(cc3dstatistics(arr)["voxel_counts"])).items() if i > 0 and idx != 0}


def np_count_nonzero(arr: np.ndarray) -> int:
    """Returns number of nonzero entries in the array

    Args:
        arr (np.ndarray): _description_

    Returns:
        int: _description_
    """
    return np.count_nonzero(arr)


def np_unique(arr: np.ndarray) -> list[int]:
    """Returns each existing label in the array (including zero!)

    Args:
        arr (np.ndarray): _description_

    Returns:
        list[int]: _description_
    """
    if not np.issubdtype(arr.dtype, np.unsignedinteger):
        return np.unique(arr)
    try:
        return [idx for idx, i in enumerate(cc3dstatistics(arr)["voxel_counts"]) if i > 0]
    except Exception:
        pass
    return np.unique(arr)


def np_unique_withoutzero(arr: UINTARRAY) -> list[int]:
    """Returns each existing label in the array (including zero!)

    Args:
        arr (np.ndarray): _description_

    Returns:
        list[int]: _description_
    """
    try:
        return [idx for idx, i in enumerate(cc3dstatistics(arr)["voxel_counts"]) if i > 0 and idx != 0]
    except Exception:
        pass
    return [i for i in np.unique(arr) if i != 0]


def np_center_of_mass(arr: UINTARRAY) -> dict[int, COORDINATE]:
    """Calculates center of mass, mapping label in array to a coordinate (float) (exluding zero)

    Args:
        arr (np.ndarray): _description_

    Returns:
        dict[int, Coordinate]: _description_
    """
    stats = cc3dstatistics(arr)
    # Does not use the other calls for speed reasons
    unique = [idx for idx, i in enumerate(stats["voxel_counts"]) if i > 0 and idx != 0]
    return {idx: v for idx, v in enumerate(stats["centroids"]) if idx in unique}


def np_bounding_boxes(arr: UINTARRAY) -> dict[int, tuple[slice, slice, slice]]:
    """Calculates bounding boxes for each different label (not zero!) in the array, returning a mapping

    Args:
        arr (np.ndarray): _description_

    Returns:
        dict[int, tuple[slice, slice, slice]]: _description_
    """
    stats = cc3dstatistics(arr)
    # Does not use the other calls for speed reasons
    unique = [idx for idx, i in enumerate(stats["voxel_counts"]) if i > 0 and idx != 0]
    return {idx: v for idx, v in enumerate(stats["bounding_boxes"]) if idx in unique}


def np_contacts(arr: UINTARRAY, connectivity: int):
    """Calculates the contacting labels and the amount of touching voxels based on connectivity

    Args:
        arr (UINTARRAY): _description_
        connectivity (int): _description_

    Returns:
        dict[tuple[int,int], int]: mapping touching labels as tuple to number of touching voxels
    """
    assert 2 <= arr.ndim <= 3, f"expected 2D or 3D, but got {arr.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    connectivity = min(connectivity * 4, 8) if arr.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26
    return contacts(arr, connectivity=connectivity)


def np_region_graph(arr: UINTARRAY, connectivity: int):
    """Returns the unique tuples of different labels in the array touching each other

    Args:
        arr (UINTARRAY): _description_
        connectivity (int): _description_

    Returns:
        set[tuple[int,int]]: touching labels in the array
    """
    assert 2 <= arr.ndim <= 3, f"expected 2D or 3D, but got {arr.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    connectivity = min(connectivity * 4, 8) if arr.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26
    return region_graph(arr, connectivity=connectivity)


def np_voxel_connectivity_graph(arr: UINTARRAY, connectivity: int):
    """Returns a voxel connectivity graph of the input array

    For 2D connectivity, the output is an 8-bit unsigned integer.

    Bits 1-4: edges     (4,8 way)
        5-8: corners   (8 way only, zeroed in 4 way)

        8      7      6      5      4      3      2      1
        ------ ------ ------ ------ ------ ------ ------ ------
        -x-y    x-y    -xy     xy     -x     +y     -x     +x

    For a 3D 26 and 18 connectivity, the output requires 32-bit unsigned integers,
        for 6-way the output are 8-bit unsigned integers.

    Bits 1-6: faces     (6,18,26 way)
        7-19: edges     (18,26 way)
        18-26: corners   (26 way)
        26-32: unused (zeroed)

    Args:
        arr (UINTARRAY): _description_
        connectivity (int): _description_

    Returns:
        uint8 or uint32 numpy array the same size as the input
    """
    assert 2 <= arr.ndim <= 3, f"expected 2D or 3D, but got {arr.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    connectivity = min(connectivity * 4, 8) if arr.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26
    return voxel_connectivity_graph(arr, connectivity=connectivity)


def np_dice(seg: np.ndarray, gt: np.ndarray, binary_compare: bool = False, label: int = 1):
    """Calculates the dice similarity between two numpy arrays

    Args:
        seg: segmentation array
        gt: other segmentation array
        binary_compare: if the should be binarized before (0/1)
        label: if not binary_compare, use this label for dice score

    Returns:
        float: dice value
    """
    assert seg.shape == gt.shape, f"shape mismatch, got {seg.shape}, and {gt.shape}"
    if binary_compare:
        seg = seg.copy()
        seg[seg != 0] = 1
        gt = gt.copy()
        gt[gt != 0] = 1
        label = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in double_scalars")
        dice = float(np.sum(seg[gt == label]) * 2.0 / (np.sum(seg) + np.sum(gt)))
    if np.isnan(dice):
        return 1.0
    return dice


def np_dilate_msk(
    arr: np.ndarray,
    label_ref: LABEL_REFERENCE = None,
    mm: int = 5,
    connectivity: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Dilates the given array by the specified number of voxels (not including the zero).

    Args:
        mm (int, optional): The number of voxels to dilate the mask by. Defaults to 5.
        connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
        mask (nparray, optional): If set, after each iteration, will zero out everything based on this mask

    Returns:
        nparray: The dilated mask.

    Notes:
        The method uses binary dilation with a 3D structuring element to dilate the mask by the specified number of voxels.

    """
    labels: list[int] = _to_labels(arr, label_ref)
    # present_labels = np_unique(arr)

    if mask is not None:
        mask[mask != 0] = 1

    struct = generate_binary_structure(arr.ndim, connectivity)

    labels: list[int] = [l for l in labels if l != 0]  # and l in present_labels]

    out = arr.copy()
    for _ in range(mm):
        for i in labels:
            data = out.copy()
            data[i != data] = 0
            msk_ibe_data = _binary_dilation(data, struct=struct)
            out[out == 0] = msk_ibe_data[out == 0] * i
            if mask is not None:
                out[mask == 0] = 0
    return out


def np_erode_msk(
    arr: np.ndarray,
    label_ref: LABEL_REFERENCE = None,
    mm: int = 5,
    connectivity: int = 3,
    border_value=0,
) -> np.ndarray:
    """
    Erodes the given array by the specified number of voxels.

    Args:
        mm (int, optional): The number of voxels to erode the mask by. Defaults to 5.
        connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).

    Returns:
        nparray: The eroded mask.

    Notes:
        The method uses binary erosion with a 3D structuring element to erode the mask by the specified number of voxels.
    """
    labels: list[int] = _to_labels(arr, label_ref)
    # present_labels = np_unique(arr)

    struct = generate_binary_structure(arr.ndim, connectivity)
    msk_i_data = arr.copy()
    out = arr.copy()
    for i in labels:
        if i == 0:  # or i not in present_labels:
            continue
        data = msk_i_data.copy()
        data[i != data] = 0
        msk_ibe_data = binary_erosion(data, structure=struct, iterations=mm, border_value=border_value)
        data[~msk_ibe_data] = 0  # type: ignore
        out[(msk_i_data == i) & (data == 0)] = 0
    return out


def np_map_labels(arr: UINTARRAY, label_map: LABEL_MAP) -> np.ndarray:
    """Maps labels in the given array according to the label_map dictionary.
    Args:
        label_map (dict): A dictionary that maps the original label values (str or int) to the new label values (int).

    Returns:
        np.ndarray: Returns a copy of the remapped array
    """
    k = np.array(list(label_map.keys()))
    v = np.array(list(label_map.values()))

    max_value = max(arr.max(), *k, *v) + 1

    mapping_ar = np.arange(max_value, dtype=arr.dtype)
    mapping_ar[k] = v
    return mapping_ar[arr]


def np_calc_crop_around_centerpoint(
    poi: tuple[int, ...] | tuple[float, ...],
    arr: np.ndarray,
    cutout_size: tuple[int, ...],
    pad_to_size: Sequence[int] | np.ndarray | int = 0,
) -> tuple[np.ndarray, tuple, tuple]:
    """

    Args:
        poi: center point of cutout
        arr: input array to cut
        cutout_size: size of the image cutout
        pad_to_size: additional padding amount

    Returns:
        np.ndarray: cut array
        tuple of the cutout coordinates
        tuple of the paddings
    """
    n_dim = len(poi)
    if isinstance(pad_to_size, int):
        pad_to_size = np.ones(n_dim) * pad_to_size
    assert (
        n_dim == len(arr.shape) == len(cutout_size) == len(pad_to_size)
    ), f"dimension mismatch, got dim {n_dim}, poi {poi}, arr shape {arr.shape}, cutout {cutout_size}, pad_to_size {pad_to_size}"

    poi = tuple(int(i) for i in poi)
    shape = arr.shape
    # Get cutout range
    cutout_coords = []
    padding = []
    for d in range(n_dim):
        _min, _max, _pad_min, _pad_max = _np_get_min_max_pad(poi[d], shape[d], cutout_size[d] // 2, pad_to_size[d] // 2)
        cutout_coords += [_min, _max]
        padding.append((int(_pad_min), int(_pad_max)))
    # cutout_coords = (x_min, x_max, y_min, y_max, z_min, z_max)
    # padding = ((x_pad_min, x_pad_max), (y_pad_min, y_pad_max), (z_pad_min, z_pad_max))

    cutout_coords_slices = tuple([slice(cutout_coords[i], cutout_coords[i + 1]) for i in range(0, n_dim * 2, 2)])
    arr_cut = arr[cutout_coords_slices]
    arr_cut = np.pad(
        arr_cut,
        tuple(padding),
    )
    return (
        arr_cut,
        cutout_coords_slices,
        tuple(padding),
        # tuple([slice(padding[i][0], padding[i][1]) for i in range(n_dim)]),
    )


def np_bbox_binary(img: np.ndarray, px_dist: int | Sequence[int] | np.ndarray = 0) -> tuple[slice, ...]:
    """calculates a bounding box in n dimensions given a image (factor ~2 times faster than compute_crop_slice)

    Args:
        img: input array
        px_dist: int | tuple[int]: dist (int): The amount of padding to be added to the cropped image. If int, will apply the same padding to each dim. Default value is 0.

    Returns:
        list of boundary coordinates [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    assert img is not None, "bbox_nd: received None as image"
    assert np_count_nonzero(img) > 0, "bbox_nd: img is empty, cannot calculate a bbox"
    n = img.ndim
    shp = img.shape
    if isinstance(px_dist, int):
        px_dist = np.ones(n, dtype=np.uint8) * px_dist
    assert len(px_dist) == n, f"dimension mismatch, got img shape {shp} and px_dist {px_dist}"

    bbox: list[float] = []
    for ax in itertools.combinations(reversed(range(n)), n - 1):
        nonzero = np.any(a=img, axis=ax)
        bbox.extend(np.where(nonzero)[0][[0, -1]])  # type: ignore
    out: tuple[slice, ...] = tuple(
        slice(
            max(bbox[i] - px_dist[i // 2], 0),
            min(bbox[i + 1] + px_dist[i // 2], shp[i // 2]) + 1,
        )
        for i in range(0, len(bbox), 2)
    )
    return out


def np_center_of_bbox_binary(img: np.ndarray, px_dist: int | Sequence[int] | np.ndarray = 0):
    bbox_nd = np_bbox_binary(img, px_dist=px_dist)
    ctd_bbox = []
    for i in range(len(bbox_nd)):
        size_t = bbox_nd[i].stop - bbox_nd[i].start
        # print(i, size_t)
        ctd_bbox.append(bbox_nd[i].start + (size_t // 2))
    return ctd_bbox


def np_approx_center_of_mass(seg: np.ndarray, label_ref: LABEL_REFERENCE = None) -> dict[int, COORDINATE]:
    import warnings

    warnings.warn("np_approx_center_of_mass deprecated use np_center_of_mass instead", stacklevel=3)  # TODO remove in version 1.0
    assert label_ref is None, "the new function does not need a label_ref"
    return np_center_of_mass(seg)


def _np_get_min_max_pad(pos: int, img_size: int, cutout_size: int, add_pad_size: int = 0) -> tuple[int, int, int, int]:
    """calc the min and max position around a center "pos" of a img and cutout size and whether it needs to be padded

    Args:
        pos: center position in one dim
        img_size: size of image in that dim
        cutout_size: cutout length in that dim

    Returns:
        pos_min, pos_max, pad_min, pad_max
    """
    if pos - cutout_size > 0:
        pos_min = pos - cutout_size
        pad_min = 0
    else:
        pos_min = 0
        pad_min = cutout_size - pos
    if pos + cutout_size < img_size:
        pos_max = pos + cutout_size
        pad_max = 0
    else:
        pos_max = img_size
        pad_max = pos + cutout_size - img_size
    return pos_min, pos_max, int(pad_min + add_pad_size), int(pad_max + add_pad_size)


def np_find_index_of_k_max_values(arr: np.ndarray, k: int = 2) -> list[int]:
    """Calculates the indices of the k-highest values in the given arr

    Args:
        arr: input array
        k: number of higest values to calculate the index for

    Returns:
        list[int]: list of indices sorted. First entry corresponds to the index of the highest value in arr, ...
    """
    idx = np.argpartition(arr, -k)[-k:]
    indices = idx[np.argsort((-arr)[idx])]
    return list(indices)


def np_connected_components(
    arr: UINTARRAY,
    connectivity: int = 3,
    label_ref: LABEL_REFERENCE = None,
    verbose: bool = False,
) -> tuple[dict[int, UINTARRAY], dict[int, int]]:
    """Calculates the connected components of a given array (works with zeros as well!)

    Args:
        arr: input arr
        labels (int | list[int] | None, optional): Labels that the connected components algorithm should be applied to. If none, applies on all labels found in arr. Defaults to None.
        connectivity: in range [1,3]. For 2D images, 2 and 3 is the same.
        verbose: If true, will print out if the array does not have any CC

    Returns:
        subreg_cc: dict[label, cc_idx, arr], subreg_cc_N: dict[label, n_connected_components]
    """

    assert np.min(arr) == 0, f"min value of mask not zero, got {np.min(arr)}"
    assert np.max(arr) >= 0, f"wrong normalization, max value is not >= 0, got {np_unique(arr)}"
    assert 2 <= arr.ndim <= 3, f"expected 2D or 3D, but got {arr.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    connectivity = min(connectivity * 2, 8) if arr.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26

    labels: Sequence[int] = _to_labels(arr, label_ref)

    subreg_cc = {}
    subreg_cc_n = {}
    for subreg in labels:  # type:ignore
        img_subreg = np_extract_label(arr, subreg, inplace=False)
        labels_out, n = connected_components(img_subreg, connectivity=connectivity, return_N=True)
        subreg_cc[subreg] = labels_out
        subreg_cc_n[subreg] = n
    if verbose:
        print(
            "Components founds (label,N): ",
            {i: subreg_cc_n[i] for i in labels},
        )
    return subreg_cc, subreg_cc_n


def np_get_largest_k_connected_components(
    arr: UINTARRAY,
    k: int | None = None,
    label_ref: LABEL_REFERENCE = None,
    connectivity: int = 3,
    return_original_labels: bool = True,
) -> UINTARRAY:
    """finds the largest k connected components in a given array (does NOT work with zero as label!)

    Args:
        arr (np.ndarray): input array
        k (int | None): finds the k-largest components. If k is None, will find all connected components and still sort them by size
        labels (int | list[int] | None, optional): Labels that the algorithm should be applied to. If none, applies on all labels found in arr. Defaults to None.
        connectivity: in range [1,3]. For 2D images, 2 and 3 is the same.
        return_original_labels (bool): If set to False, will label the components from 1 to k. Defaults to True

    Returns:
        np.ndarray: array with the largest k connected components
    """

    assert k is None or k > 0
    assert 2 <= arr.ndim <= 3, f"expected 2D or 3D, but got {arr.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    if arr.ndim == 2:  # noqa: SIM108
        connectivity = min(connectivity * 2, 8)  # 1:4, 2:8, 3:8
    else:
        connectivity = 6 if connectivity == 1 else 18 if connectivity == 2 else 26

    arr2 = arr.copy()
    labels: Sequence[int] = _to_labels(arr, label_ref)
    arr2[np.isin(arr, labels, invert=True)] = 0  # type:ignore

    labels_out, n = connected_components(arr2, connectivity=connectivity, return_N=True)
    if k is None:
        k = n
    k = min(k, n)  # if k > N, will return all N but still sorted
    label_volume_pairs = [(i, ct) for i, ct in np_volume(labels_out).items() if ct > 0]
    label_volume_pairs.sort(key=lambda x: x[1], reverse=True)
    preserve: list[int] = [x[0] for x in label_volume_pairs[:k]]

    cc_out = np.zeros(arr.shape, dtype=arr.dtype)
    for i, preserve_label in enumerate(preserve):
        cc_out[labels_out == preserve_label] = i + 1
    if return_original_labels:
        arr *= cc_out > 0  # to get original labels
        return arr
    return cc_out


def np_get_connected_components_center_of_mass(
    arr: UINTARRAY, label: int, connectivity: int = 3, sort_by_axis: int | None = None
) -> list[COORDINATE]:
    """Calculates the center of mass of the different connected components of a given label in an array

    Args:
        arr (np.ndarray): input array
        label (int): the label of the connected components
        connectivity (int, optional): Connectivity for the connected components. Defaults to 3.
        sort_by_axis (int | None, optional): If not none, will sort the center of mass list by this axis values. Defaults to None.

    Returns:
        _type_: _description_
    """
    if sort_by_axis is not None:
        assert 0 <= sort_by_axis <= len(arr.shape) - 1, f"sort_by_axis {sort_by_axis} invalid with an array of shape {arr.shape}"  # type:ignore
    subreg_cc, _ = np_connected_components(
        arr.copy(),
        connectivity=connectivity,
        label_ref=label,
        verbose=False,
    )
    coms = list(np_center_of_mass(subreg_cc[label]).values())

    if sort_by_axis is not None:
        coms.sort(key=lambda a: a[sort_by_axis])
    return coms


def np_translate_to_center_of_array(image: np.ndarray) -> np.ndarray:
    """Moves the nonzero values of an array so its center of mass is in the center of the array shape

    Args:
        image: input array

    Returns:
        np.ndarray: array of the same shape translated to the center
    """
    shape = image.shape
    shape_center = tuple(i // 2 for i in shape)
    com = center_of_mass(image)
    translation_vector: tuple[int, int] | tuple[int, int, int] = tuple(np.int32(np.asarray(shape_center) - np.asarray(com)))
    return np_translate_arr(image, translation_vector)


def np_translate_arr(arr: np.ndarray, translation_vector: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    """Translates nonzero values of an input array according to a 2D or 3D translation vector. Values that would be shifted beyond the boundary are removed!

    Args:
        arr: input array
        translation_vector: vector to translated the array with (2D or 3D)

    Returns:
        np.ndarray: the translated array

    Examples:
        >>> a = np.array([[0, 1, 0], [0, 2, 1], [1, 0, 0]])
        >>> b = np_translate_arr(a, translation_vector=(1, 0))
        >>> print(b)
        >>> [[0 0 0],[0 1 0],[0 2 1]]
    """
    assert 2 <= len(translation_vector) <= 3, f"expected translation vector to be 2D or 3D, but got {translation_vector}"
    assert len(arr.shape) == len(translation_vector), f"mismatch dimensions, got arr shape {arr.shape} and vector {translation_vector}"
    arr_translated = np.zeros_like(arr)
    if len(translation_vector) == 3:
        tx, ty, tz = translation_vector  # type:ignore
        H, W, D = arr.shape  # noqa: N806
        arr_translated[
            max(tx, 0) : H + min(tx, 0),
            max(ty, 0) : W + min(ty, 0),
            max(tz, 0) : D + min(tz, 0),
        ] = arr[
            -min(tx, 0) : H - max(tx, 0),
            -min(ty, 0) : W - max(ty, 0),
            -min(tz, 0) : D - max(tz, 0),
        ]
    else:
        tx, ty = translation_vector  # type:ignore
        H, W = arr.shape  # noqa: N806
        arr_translated[max(tx, 0) : H + min(tx, 0), max(ty, 0) : W + min(ty, 0)] = arr[
            -min(tx, 0) : H - max(tx, 0), -min(ty, 0) : W - max(ty, 0)
        ]
    return arr_translated


def np_fill_holes(arr: np.ndarray, label_ref: LABEL_REFERENCE = None, slice_wise_dim: int | None = None) -> np.ndarray:
    """Fills holes in segmentations

    Args:
        arr (np.ndarray): Input segmentation array
        labels (int | list[int] | None, optional): Labels that the hole-filling should be applied to. If none, applies on all labels found in arr. Defaults to None.
        slice_wise_dim (int | None, optional): If the input is 3D, the specified dimension here cna be used for 2D slice-wise filling. Defaults to None.

    Returns:
        np.ndarray: The array with holes filled
    """
    assert 2 <= arr.ndim <= 3
    assert arr.ndim == 3 or slice_wise_dim is None, "slice_wise_dim set but array is 3D"
    labels: Sequence[int] = _to_labels(arr, label_ref)
    for l in labels:  # type:ignore
        arr_l = arr.copy()
        arr_l = np_extract_label(arr_l, l)
        if slice_wise_dim is None:
            filled = fill(arr_l).astype(arr.dtype)
        else:
            assert 0 <= slice_wise_dim <= arr.ndim - 1, f"slice_wise_dim needs to be in range [0, {arr.ndim -1}]"
            filled = np.swapaxes(arr_l.copy(), 0, slice_wise_dim)
            filled = np.stack([fill(x) for x in filled])
            filled = np.swapaxes(filled, 0, slice_wise_dim)
        filled[filled != 0] = l
        arr[arr == 0] = filled[arr == 0]
    return arr


def np_calc_convex_hull(
    arr: INTARRAY,
    axis: int | None = None,
    verbose: bool = False,
):
    """Calculates the convex hull of a given array, returning the array filled with its convex hull

    Args:
        arr (INTARRAY): Input array
        axis (int | None, optional): If given axis, will calculate convex hull along that axis (remaining dimension must be at least 2). Defaults to None.
    """
    n_dims = arr.ndim
    if axis is None:
        assert 2 <= n_dims <= 3, f"If axis is none, array must be 2- or 3-dimensional, but got {n_dims} with shape {arr.shape}"
        return _convex_hull(arr, verbose=verbose)[0]
    else:
        assert 3 <= n_dims <= 4, f"If axis is given, the array must be 3- or 4-dimensional, but got {n_dims} with shape {arr.shape}"
        assert 0 <= axis <= n_dims, f"Specified axis must be in range of dimension, but got axis={axis} and n_dims={n_dims}"
        h = arr * 0
        for i in range(arr.shape[axis]):
            slices = _select_axis_dynamically(axis=axis, index=i, n_dims=n_dims)
            if np.count_nonzero(arr[slices]) == 0:
                continue
            try:
                convex_hull_slice = _convex_hull(arr[slices], verbose=verbose)[0].astype(arr.dtype)
                h[slices] += convex_hull_slice
            except Exception:
                pass
        return h


def _select_axis_dynamically(axis: int, index: int | slice, n_dims: int = 3):
    slices = tuple([slice(None) if i != axis else index for i in range(n_dims)])
    return slices


def _convex_hull(
    arr: INTARRAY,
    verbose: bool = False,
):
    """Calculates the convex hull of a given array (all labels are taken)

    Args:
        arr (INTARRAY): integer array

    Returns:
        _type_: out_img, hull
    """
    points = np.transpose(np.where(arr))
    if len(points) <= 3:
        print("To few points") if verbose else None
        return np.zeros_like(arr, dtype=arr.dtype)
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(arr.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(arr.shape, dtype=arr.dtype)
    out_img[out_idx] = 1
    return out_img, hull


def np_calc_boundary_mask(
    img: np.ndarray,
    threshold: float = 0,
    adjust_intensity_for_ct=False,
):
    """
    Calculate a boundary mask based on the input image.

    Parameters:
    - img (NII): The image used to create the boundary mask.
    - threshold(float): threshold
    - adjust_intensity_for_ct (bool): If True, adjust the image intensity by adding 1000.

    Returns:
    NII: A segmentation of the boundary.


    This function takes a NII and generates a boundary mask by marking specific regions.
    The intensity of the image can be adjusted for CT scans by adding 1000. The boundary mask is created by initializing
    corner points and using an "infect" process to mark neighboring points. The boundary mask is initiated with
    zeros, and specific boundary points are set to 1. The "infect" function iteratively marks neighboring points in the mask.
    The process starts from the initial points and corner points of the image. The infection process continues until the
    infect_list is empty. The resulting boundary mask is modified by subtracting 1 from all non-zero values and setting
    the remaining zeros to 2. The sum of the boundary mask values is printed before returning the modified NII object.

    """
    if adjust_intensity_for_ct:
        img = img + 1000
    boundary = img.copy()
    boundary[boundary > threshold] = 2
    boundary[boundary <= threshold] = 0
    infect_list = []

    def infect(x, y, z):
        if any([x < 0, y < 0, z < 0, x == boundary.shape[0], y == boundary.shape[1], z == boundary.shape[2]]):
            return
        if boundary[x, y, z] == 0:
            boundary[x, y, z] = 1
            for a, b, c in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]:
                infect_list.append((x + a, y + b, z + c))

        else:
            pass

    infect(0, 0, 0)
    infect(boundary.shape[0] - 1, 0, 0)
    infect(0, boundary.shape[1] - 1, 0)
    infect(boundary.shape[0] - 1, boundary.shape[1] - 1, 0)

    infect(0, 0, boundary.shape[2] - 1)
    infect(boundary.shape[0] - 1, 0, boundary.shape[2] - 1)
    infect(0, boundary.shape[1] - 1, boundary.shape[2] - 1)
    infect(boundary.shape[0] - 1, boundary.shape[1] - 1, boundary.shape[2] - 1)
    while len(infect_list) != 0:
        infect(*infect_list.pop())
    boundary[boundary == 0] = 2
    boundary -= 1
    print(boundary.sum())
    return boundary


def np_betti_numbers(img: np.ndarray, verbose=False) -> tuple[int, int, int]:
    """
    calculates the Betti number B0, B1, and B2 for a 3D img
    from the Euler characteristic number

    B0: Number of connected compotes
    B1: Number of holes
    B2: Number of fully engulfed empty spaces

    code prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)
    Source: https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py
    """
    # make sure the image is 3D (for connectivity settings)
    assert len(img.shape) == 3
    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected, and conversely
    N6 = 1  # noqa: N806
    N26 = 3  # noqa: N806
    # important first step is to
    # pad the image with background (0) around the border!
    padded = np.pad(img, pad_width=1)
    # make sure the image is binary with
    assert set(np_unique(padded)).issubset({0, 1})
    # calculate the Betti numbers B0, B2
    # then use Euler characteristic to get B1
    # get the label connected regions for foreground
    _, b0 = label(padded, return_num=True, connectivity=N26)  # 26 neighborhoods for foreground
    euler_char_num = euler_number(padded, connectivity=N26)  # 26 neighborhoods for foreground
    # get the label connected regions for background
    _, b2 = label(1 - padded, return_num=True, connectivity=N6)  # 6 neighborhoods for background
    # NOTE: need to subtract 1 from b2
    b2 -= 1
    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Betti:1 + Betti:2
    if verbose:
        print(f"Betti number: b0 = {b0}, b1 = {b1}, b2 = {b2}")
    return b0, b1, b2


def np_calc_overlapping_labels(
    reference_arr: np.ndarray,
    prediction_arr: np.ndarray,
) -> list[tuple[int, int]]:
    """Calculates the pairs of labels that are overlapping in at least one voxel (fast)

    Args:
        prediction_arr (np.ndarray): Numpy array containing the prediction labels.
        reference_arr (np.ndarray): Numpy array containing the reference labels.
        ref_labels (list[int]): List of unique reference labels.

    Returns:
        list[tuple[int, int]]: List of tuples of labels that overlap in at least one voxel
    """
    ref_labels = np_unique_withoutzero(reference_arr)
    overlap_arr = prediction_arr.astype(np.uint32)
    max_ref = max(ref_labels) + 1
    overlap_arr = (overlap_arr * max_ref) + reference_arr
    overlap_arr[reference_arr == 0] = 0
    # overlapping_indices = [(i % (max_ref), i // (max_ref)) for i in np.unique(overlap_arr) if i > max_ref]
    # instance_pairs = [(reference_arr, prediction_arr, i, j) for i, j in overlapping_indices]

    # (ref, pred)
    return [(int(i % (max_ref)), int(i // (max_ref))) for i in np.unique(overlap_arr) if i > max_ref]


def _to_labels(arr: np.ndarray, labels: LABEL_REFERENCE = None) -> Sequence[int]:
    if labels is None:
        labels = list(np_unique(arr))
    if not isinstance(labels, Sequence):
        labels = [labels]
    return labels


def _generate_binary_structure(n_dim: int, connectivity: int, kernel_size: int = 3):
    assert kernel_size >= 3, f"kernel_size must be >= 3, got {kernel_size}"
    assert kernel_size % 2 == 1, f"kernel_size must be an odd number, got {kernel_size}"
    connectivity = max(connectivity, 1)
    kernel_delta = (kernel_size - 3) // 2
    if n_dim < 1:
        return np.array(True, dtype=bool)
    output = np.fabs(np.indices([kernel_size] * n_dim) - 1 - kernel_delta)
    output = np.add.reduce(output, 0)
    return output <= connectivity + kernel_delta


# Fast Binary Morphological operations (taken from https://github.com/shoheiogawa/binmorphopy/blob/master/binmorphopy/morphology.py)
def _binary_dilation(image: np.ndarray, struct=None):
    selem = struct
    dim = image.ndim
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if image.dtype != bool:
        image = image.astype(bool)
    if selem is None:
        if dim == 1:
            selem = np.ones(shape=[3], dtype=bool)
        elif dim == 2:
            selem = np.zeros(shape=[3, 3], dtype=bool)
            selem[1, :] = True
            selem[:, 1] = True
        elif dim == 3:
            selem = np.zeros(shape=[3, 3, 3], dtype=bool)
            selem[:, 1, 1] = True
            selem[1, :, 1] = True
            selem[1, 1, :] = True
    else:
        if not isinstance(selem, np.ndarray):
            selem = np.asarray(selem, dtype=bool)
        if selem.dtype != bool:
            selem = selem.astype(bool)
        if any(num_pixels % 2 == 0 for num_pixels in selem.shape):
            raise ValueError("Only structure element of odd dimension " "in each direction is supported.")
    perimeter_image = _get_perimeter_image(image)
    perimeter_coords = np.where(perimeter_image)
    out = image.copy()
    assert selem is not None
    if dim == 1:
        sx = selem.shape[0]
        rx = sx // 2
        lx = image.shape[0]
        for ix in perimeter_coords[0]:
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            out[jx_b:jx_e] |= selem[kx_b:kx_e]

    if dim == 2:
        rx, ry = (n // 2 for n in selem.shape)
        lx = image.shape
        sx, sy = selem.shape
        lx, ly = image.shape
        for ix, iy in zip(perimeter_coords[0], perimeter_coords[1], strict=False):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            out[jx_b:jx_e, jy_b:jy_e] |= selem[kx_b:kx_e, ky_b:ky_e]

    if dim == 3:
        rx, ry, rz = (n // 2 for n in selem.shape)
        sx, sy, sz = selem.shape
        lx, ly, lz = image.shape
        for ix, iy, iz in zip(perimeter_coords[0], perimeter_coords[1], perimeter_coords[2], strict=False):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            (jz_b, jz_e), (kz_b, kz_e) = _generate_array_indices(iz, rz, sz, lz)
            out[jx_b:jx_e, jy_b:jy_e, jz_b:jz_e] |= selem[kx_b:kx_e, ky_b:ky_e, kz_b:kz_e]
    return out


def _binary_erosion(image: np.ndarray, struct=None) -> np.ndarray:
    image = image.astype(bool)
    image = np.pad(image, 1, constant_values=0)
    out_image = _binary_dilation(~image, struct)
    out_image = ~out_image
    out_image = _unpad(out_image, 1)
    # out_image = np.pad(cropped, 1, constant_values=0)
    return out_image


def _unpad(image: np.ndarray, pad_width: tuple[tuple[int, int], ...] | int):
    slices = []
    if isinstance(pad_width, int):
        ndim = image.ndim
        pad_width = tuple((pad_width, pad_width) for i in range(ndim))
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return image[tuple(slices)]


def _binary_closing(image: np.ndarray, struct=None):
    out_image = _binary_erosion(_binary_dilation(image, struct), struct)
    return out_image


def _binary_opening(image, struct=None):
    out_image = _binary_dilation(_binary_erosion(image, struct), struct)
    return out_image


def _get_perimeter_image(image):
    """Return the image of the perimeter structure of the input image

    Args:
        image (Numpy array): Image data as an array

    Returns:
        Numpy array: Perimeter image
    """
    dim = image.ndim
    if dim > 3:
        raise RuntimeError("Binary image in 4D or above is not supported.")
    count = np.zeros_like(image, dtype=np.uint8)
    inner = np.zeros_like(image, dtype=bool)

    count[1:] += image[:-1]
    count[:-1] += image[1:]

    if dim == 1:
        inner |= image == 2
        for i in [0, -1]:
            inner[i] |= count[i] == 1
        return image & (~inner)

    count[:, 1:] += image[:, :-1]
    count[:, :-1] += image[:, 1:]
    if dim == 2:
        inner |= count == 4
        for i in [0, -1]:
            inner[i] |= count[i] == 3
            inner[:, i] |= count[:, i] == 3
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 2
        return image & (~inner)

    count[:, :, 1:] += image[:, :, :-1]
    count[:, :, :-1] += image[:, :, 1:]

    if dim == 3:
        inner |= count == 6
        for i in [0, -1]:
            inner[i] |= count[i] == 5
            inner[:, i] |= count[:, i] == 5
            inner[:, :, i] |= count[:, :, i] == 5
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
        for i in [0, -1]:
            for j in [0, -1]:
                for k in [0, -1]:
                    inner[i, j, k] |= count[i, j, k] == 3
        return image & (~inner)
    raise RuntimeError("This line should not be reached.")


def _generate_array_indices(selem_center, selem_radius, selem_length, result_length):
    """Return the correct indices for slicing considering near-edge regions

    Args:
        selem_center (int): The index of the structuring element's center
        selem_radius (int): The radius of the structuring element
        selem_length (int): The length of the structuring element
        result_length (int): The length of the operating image

    Returns:
        (int, int): The range begin and end indices for the operating image
        (int, int): The range begin and end indices for the structuring element image
    """
    # First index for the result array
    result_begin = selem_center - selem_radius
    # Last index for the result array
    result_end = selem_center + selem_radius + 1
    # First index for the structuring element array
    selem_begin = -result_begin if result_begin < 0 else 0
    result_begin = max(0, result_begin)
    # Last index for the structuring element array
    selem_end = selem_length - (result_end - result_length) if result_end > result_length else selem_length
    return (result_begin, result_end), (selem_begin, selem_end)


if __name__ == "__main__":
    array = np.array(
        [
            [0, 0, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [1, 1, 0, 0, 0],
        ]
    )

    print(array)
    print(np_voxel_connectivity_graph(array, connectivity=1))
