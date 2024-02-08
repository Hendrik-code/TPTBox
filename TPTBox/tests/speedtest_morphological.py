if __name__ == "__main__":
    # speed test dilation
    import random
    from time import perf_counter

    import numpy as np
    from tqdm import tqdm
    from TPTBox.tests.speedtest import speed_test

    from TPTBox.core.np_utils import (
        _binary_dilation,
        _binary_erosion,
        _unpad,
        binary_dilation,
        binary_erosion,
        generate_binary_structure,
        np_dilate_msk,
        np_erode_msk,
        np_erode_msknew,
    )
    from TPTBox.unit_tests.test_centroids import get_nii

    def get_nii_array():
        num_points = 0 if random.random() < 0.01 else 5
        nii, points, orientation, sizes = get_nii(x=(300, 300, 50), num_point=num_points)
        arr = nii.get_seg_array()
        arr_r = arr.copy()
        return arr_r

    speed_test(
        repeats=100,
        get_input_func=get_nii_array,
        functions=[np_erode_msk, np_erode_msknew],
        assert_equal_function=lambda x, y: np.count_nonzero(x) == np.count_nonzero(y),
        mm=10,
    )
    # print(time_measures)
