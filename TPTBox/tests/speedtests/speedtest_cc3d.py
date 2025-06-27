from __future__ import annotations

if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.np_utils import (
        _to_labels,
        np_bbox_binary,
        np_bounding_boxes,
        np_center_of_mass,
        np_map_labels,
        np_unique,
        np_unique_withoutzero,
        np_volume,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 30)
        nii, points, orientation, sizes = get_nii(x=(140, 140, 150), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint8)
        # arr[arr == 1] = -1
        arr_r = arr.copy()
        return arr_r

    def cc3d_com(arr: np.ndarray):
        return np_center_of_mass(arr)

    def center_of_mass_one(arr: np.ndarray):
        coms = center_of_mass(arr)
        return coms

    def center_of_mass_(arr: np.ndarray):
        cc_label_set = np_unique(arr)
        coms = {}
        for c in cc_label_set:
            if c == 0:
                continue
            com = center_of_mass(arr == c)
            coms[c] = com
        return coms

    def bbox_(arr: np.ndarray):
        cc_label_set = np_unique(arr)
        coms = {}
        for c in cc_label_set:
            if c == 0:
                continue
            com = np_bbox_binary(arr == c)
            coms[c] = com
        return coms

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[cc3d_com, center_of_mass_],
        assert_equal_function=lambda x, y: np.all([x[i][0] == y[i][0] for i in x.keys()]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
