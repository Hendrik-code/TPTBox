if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import (
        _to_labels,
        cc3dstatistics,
        np_bbox_binary,
        np_extract_label,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(5, 10)
        nii, points, orientation, sizes = get_nii(x=(300, 300, 300), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    def normal(arr):
        return cc3dstatistics(arr, use_crop=False)

    def crop(arr):
        crop = np_bbox_binary(arr)
        arr = arr[crop]
        return cc3dstatistics(arr, use_crop=False)

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[normal, crop],
        assert_equal_function=lambda x, y: True,  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
