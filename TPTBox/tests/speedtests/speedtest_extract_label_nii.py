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
        nii, points, orientation, sizes = get_nii(x=(300, 300, 300), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        # arr = nii.get_seg_array().astype(int)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return nii

    extract_label = [2, 3, 4, 5]

    def nii_extract(nii: NII):
        return nii.extract_label(extract_label)

    def nii_extract2(nii: NII):
        return nii.set_array(np_extract_label(nii.get_seg_array(), extract_label, inplace=False))

    def nii_extract3(nii: NII):
        return nii.set_array(np_extract_label(nii.get_seg_array(), extract_label, inplace=True))

    speed_test(
        repeats=100,
        get_input_func=get_nii_array,
        functions=[
            nii_extract,
            nii_extract2,
            nii_extract3,
        ],
        # functions=[extractloop_e, extractloop_indexing],
        assert_equal_function=lambda x, y: np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
