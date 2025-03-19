if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import (
        _connected_components,
        _to_labels,
        np_bbox_binary,
        np_calc_overlapping_labels,
        np_connected_components,
        np_connected_components_per_label,
        np_connected_components_per_label2,
        np_extract_label,
        np_unique,
        np_unique_withoutzero,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(10, 31)
        nii, points, orientation, sizes = get_nii(x=(300, 300, 300), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint8)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def np_cc_labelwise1(arr: np.ndarray):
        return np_connected_components_per_label(arr)[0][1]

    def np_cc_labelwise2(arr: np.ndarray):
        return np_connected_components_per_label2(arr)[0][1]

    def np_cc_labelwise2crop(arr: np.ndarray):
        return np_connected_components_per_label2(arr, use_crop=True)[0][1]

    # def np_cc_once(arr: np.ndarray):
    #    # call cc once, then relabel
    #    connectivity = 3
    #    connectivity = min((connectivity + 1) * 2, 8) if arr.ndim == 2 else 6 if connectivity == 1 else 18 if connectivity == 2 else 26
    #
    #    labels: list[int] = np_unique(arr)
    #
    #    subreg_cc = {}
    #    subreg_cc_n = {}
    #    crop = np_bbox_binary(arr)
    #    arrc = arr[crop]
    #    zarr = np.zeros((len(labels), *arr.shape), dtype=arr.dtype)
    #
    #    labels_out = connected_components(arrc, connectivity=connectivity, return_N=False)
    #    for sidx, subreg in enumerate(labels):  # type:ignore
    #        arrcc[crop][np.logical_and()]
    #        # arr[s == subreg]
    #        # img_subreg = np_extract_label(arrc, subreg, inplace=False)
    #        # lcrop = np_bbox_binary(img_subreg)
    #        img_subregc = img_subreg[lcrop]
    #        img_subreg[lcrop] = labels_out[lcrop] * img_subregc
    #
    #        arrcc = zarr[sidx]
    #        arrcc[crop] = img_subreg
    #        subreg_cc[subreg] = arrcc
    #        subreg_cc_n[subreg] = len(np_unique_withoutzero(img_subreg[lcrop]))
    #    return subreg_cc[1]  # , subreg_cc_n

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[
            np_cc_labelwise1,
            np_cc_labelwise2,
        ],
        assert_equal_function=lambda x, y: True,  # np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
