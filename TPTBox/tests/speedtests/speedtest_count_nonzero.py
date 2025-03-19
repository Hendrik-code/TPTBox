if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import _to_labels, np_bbox_binary, np_count_nonzero, np_volume
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 10)
        nii, points, orientation, sizes = get_nii(x=(350, 350, 350), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def np_naive_count(arr: np.ndarray):
        return np.count_nonzero(arr)

    def np_count(arr: np.ndarray):
        return sum(np_volume(arr).values())

    def np_countgreater(arr: np.ndarray):
        return (arr > 0).sum()

    def np_countcrop(arr: np.ndarray):
        crop = np_bbox_binary(arr)
        arrc = arr[crop]
        return np.count_nonzero(arrc)

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[np_naive_count, np_count, np_countgreater, np_countcrop],
        assert_equal_function=lambda x, y: x == y,  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
