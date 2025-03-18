if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import (
        _to_labels,
        np_bbox_binary,
        np_extract_label,
        np_map_labels,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 15)
        nii, points, orientation, sizes = get_nii(x=(150, 150, 150), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(int)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def map_labels(arr: np.ndarray):
        return np_map_labels(arr, {1: 2})

    def map_labels2(arr: np.ndarray):
        crop = np_bbox_binary(arr == 1)
        arr2 = arr[crop]
        arr2 = np_map_labels(arr2, {1: 2})
        arr[crop] = arr2
        return arr

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[map_labels, map_labels2],
        assert_equal_function=lambda x, y: np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
