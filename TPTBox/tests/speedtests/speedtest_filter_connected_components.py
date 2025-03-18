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
        np_extract_label,
        np_filter_connected_components,
        np_unique,
        np_unique_withoutzero,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(10, 31)
        nii, points, orientation, sizes = get_nii(x=(500, 500, 350), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint8)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def np_cc_labelwise1(arr: np.ndarray):
        return np_filter_connected_components(arr, min_volume=10, max_volume=50, largest_k_components=3)

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[
            np_cc_labelwise1,
        ],
        assert_equal_function=lambda x, y: True,  # np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
