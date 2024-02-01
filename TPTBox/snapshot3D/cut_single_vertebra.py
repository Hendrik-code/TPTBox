from __future__ import annotations

import numpy as np
import warnings
import itertools


def make_bounding_box_cutout(
    reference_mask_arr: np.ndarray, subreg_mask_arr: np.ndarray, cutout_size: tuple[int, int, int] | None, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Makes a cutout around the center of the bounding box

    Args:
        reference_mask_arr: used to calculate the bounding box (is made binary)
        subreg_mask_arr: subregion mask array, is cut and mutliplied with the cut reference mask
        cutout_size: size of the cutout. Gives a warning if pixels of the reference mask would be cut away. If none, will cut to bounding box
    Returns:
        vert_mask_arr_cut, subreg_mask_arr_cut, mask_subreg_combi_cut
    """
    shape = np.shape(reference_mask_arr)

    # make reference mask binary
    ref_mask_arr = reference_mask_arr.copy()
    ref_mask_arr[ref_mask_arr <= 0] = 0
    ref_mask_arr[ref_mask_arr != 0] = 1

    cutout_coords, (x_pad_min, x_pad_max, y_pad_min, y_pad_max, z_pad_min, z_pad_max) = calculate_cbbox_cutout_coords(
        ref_mask_arr, cutout_size=cutout_size, cutout_margin=1.1 if cutout_size is None else 1.0, verbose=verbose
    )

    # Cut masks
    vert_mask_arr_cut = make_3d_cutout(ref_mask_arr, cutout_coords)
    subreg_mask_arr_cut = make_3d_cutout(subreg_mask_arr, cutout_coords)
    # Combine vertebra binary mask with subregion
    mask_subreg_combi_cut = vert_mask_arr_cut * subreg_mask_arr_cut

    # Padding
    # img_arr_cut = iso_cut.get_fdata()
    # img_arr_cut = np.pad(img_arr_cut, ((x_pad_min, x_pad_max), (y_pad_min, y_pad_max), (z_pad_min, z_pad_max)))
    mask_subreg_combi_cut = np.pad(mask_subreg_combi_cut, ((x_pad_min, x_pad_max), (y_pad_min, y_pad_max), (z_pad_min, z_pad_max)))

    return vert_mask_arr_cut, subreg_mask_arr_cut, mask_subreg_combi_cut


def calculate_cbbox_cutout_coords(
    reference_mask_arr: np.ndarray, cutout_size: tuple[int, int, int] | None, cutout_margin: float = 1.0, verbose: bool = False
):
    """
    Args:
        reference_mask_arr: np array (is made binary)
        cutout_size: size of the cutout. Gives a warning if pixels of the reference mask would be cut away. If none, will cut to bounding box
        cutout_margin: the size of the cutout in each dimension is multiplied with this factor (so 1.1 means 10% additional space beyond the bbox is used as well)
        verbose:

    Returns:
        cutout_coords(x_min, x_max, y_min, y_max, z_min, z_max), paddings(x_pad_min, x_pad_max, y_pad_min, y_pad_max, z_pad_min, z_pad_max)
    """
    shape = np.shape(reference_mask_arr)

    # make reference mask binary
    ref_mask_arr = reference_mask_arr.copy()
    ref_mask_arr[ref_mask_arr <= 0] = 0
    ref_mask_arr[ref_mask_arr != 0] = 1
    bbox_3d = bbox_nd(ref_mask_arr)
    size_t: tuple[float, float, float] = (bbox_3d[1] - bbox_3d[0], bbox_3d[3] - bbox_3d[2], bbox_3d[5] - bbox_3d[4])

    if cutout_size is None:
        used_cutout_size: tuple[int, ...] = tuple(int(i) for i in size_t)
    else:
        used_cutout_size = cutout_size

    used_cutout_size = tuple(int(i * cutout_margin) for i in used_cutout_size)

    if (size_t[0] > used_cutout_size[0] or size_t[1] > used_cutout_size[1] or size_t[2] > used_cutout_size[2]) and verbose:
        warnings.warn(f"reference mask has bigger bounding box than cutout_size, got {size_t}, and {used_cutout_size}", UserWarning)

    # center around the bounding box
    x, y, z = center_of_bbox_nd(bbox_nd=bbox_3d)

    # Get cutout range
    x_min, x_max, x_pad_min, x_pad_max = get_min_max_pad(x, shape[0], used_cutout_size[0])
    y_min, y_max, y_pad_min, y_pad_max = get_min_max_pad(y, shape[1], used_cutout_size[1])
    z_min, z_max, z_pad_min, z_pad_max = get_min_max_pad(z, shape[2], used_cutout_size[2])
    return (x_min, x_max, y_min, y_max, z_min, z_max), (x_pad_min, x_pad_max, y_pad_min, y_pad_max, z_pad_min, z_pad_max)


def make_3d_cutout(arr: np.ndarray, cutout_coords: tuple[int, ...]) -> np.ndarray:
    """
    makes a 3d cutout from a given array

    Args:
        arr: array to cut
        cutout_coords: coordinates (idx) to cut with

    Returns:
        array cut given the cutout_coords
    """
    assert len(cutout_coords) == 6, f"cutout_size is not a list with 6 entries, instead got {cutout_coords}"
    return arr[cutout_coords[0] : cutout_coords[1], cutout_coords[2] : cutout_coords[3], cutout_coords[4] : cutout_coords[5]]


def get_min_max_pad(pos: int, img_size: int, cutout_size: int) -> tuple[int, int, int, int]:
    """
    @note: calc the min and max position around a center "pos" of a img and cutout size and whether it needs to be padded
    @return: pos_min, pos_max, pad_min, pad_max
    """
    cutout_low = cutout_size // 2
    cutout_high = cutout_low
    if cutout_size % 2 != 0:
        cutout_low += 1

    if pos - cutout_low > 0:
        pos_min = pos - cutout_low
        pad_min = 0
    else:
        pos_min = 0
        pad_min = cutout_low - pos
    if pos + cutout_high < img_size:
        pos_max = pos + cutout_high
        pad_max = 0
    else:
        pos_max = img_size
        pad_max = pos + cutout_high - img_size
    return pos_min, pos_max, pad_min, pad_max


def bbox_nd(img: np.ndarray) -> tuple[float, ...]:
    """
    calculates a bounding box in n dimensions given a image
    """
    assert img is not None, "bbox_nd: received None as image"
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(a=img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def center_of_bbox_nd(bbox_nd: tuple[float, ...]) -> list[float]:
    """
    calculates the center of a given bounding box
    @param bbox_nd: tuple with exactly dim * 2 entries
    """
    assert len(bbox_nd) % 2 == 0, f"bounding box has no even number of values, got {bbox_nd}"
    n_dim = len(bbox_nd) // 2
    ctd_bbox = []
    for i in range(0, len(bbox_nd), 2):
        size_t = bbox_nd[i + 1] - bbox_nd[i]
        # print(i, size_t)
        ctd_bbox.append((bbox_nd[i] + (size_t // 2)))
    return ctd_bbox
