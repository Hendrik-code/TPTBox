if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from panoptica.utils.input_check_and_conversion.sanity_checker import sanity_check_and_convert_to_array
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
        nii, points, orientation, sizes = get_nii(x=(10, 10, 10), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint8)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def check_sanity(arr: np.ndarray):
        _, c = sanity_check_and_convert_to_array(arr, arr)
        return c.name

    speed_test(
        repeats=5,
        get_input_func=get_nii_array,
        functions=[check_sanity],
        assert_equal_function=lambda x, y: np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
