if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import (
        _to_labels,
        np_extract_label,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(5, 10)
        nii, points, orientation, sizes = get_nii(x=(1000, 1000, 300), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        # arr = nii.get_seg_array().astype(int)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return nii

    def nii_extract(nii: NII):
        return nii.extract_label([1, 2, 3, 4, 5]).get_seg_array()

    def np_extract(nii: NII):
        return np_extract_label(nii.get_seg_array(), 1)

    def np_extractlist(nii: NII):
        return np_extract_label(nii.get_seg_array(), [1, 2, 3, 4, 5], use_crop=False)

    def np_extractlist_crop(nii: NII):
        return np_extract_label(nii.get_seg_array(), [1, 2, 3, 4, 5], use_crop=True)

    speed_test(
        repeats=10,
        get_input_func=get_nii_array,
        functions=[np_extractlist_crop, np_extractlist],
        assert_equal_function=lambda x, y: np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
