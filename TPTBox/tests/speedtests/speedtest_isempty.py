if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import _to_labels, np_bbox_binary, np_count_nonzero, np_unique_withoutzero, np_volume
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 10)
        nii, points, orientation, sizes = get_nii(x=(550, 550, 550), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(int) * 0
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def np_naive_count(arr: np.ndarray):
        return np.count_nonzero(arr) > 0

    def np_max(arr: np.ndarray):
        return arr.max() != 0

    def np_sum(arr: np.ndarray):
        return arr.sum() > 0

    def np_any(arr: np.ndarray):
        return np.any(arr)

    def np_nonzero(arr: np.ndarray):
        # super slow
        return arr.nonzero()[0].size != 0

    def np_sunique(arr: np.ndarray):
        # super slow
        return len(np_unique_withoutzero(arr)) != 0

    speed_test(
        repeats=100,
        get_input_func=get_nii_array,
        functions=[np_naive_count, np_max, np_any],
        assert_equal_function=lambda x, y: x == y,  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
