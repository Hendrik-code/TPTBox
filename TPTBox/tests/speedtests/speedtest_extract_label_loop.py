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
        arr = nii.get_seg_array().astype(np.uint8)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    # def nii_extract(nii: NII):
    #    return nii.extract_label([1, 2, 3, 4, 5]).get_seg_array()

    extract_label_one = 2
    extract_label = [2, 3, 4, 5]

    def dummy(arr_bin: np.ndarray):
        return arr_bin + arr_bin

    def np_extractloop(arr: np.ndarray):
        if 1 not in extract_label:
            arr[arr == 1] = 0
        for idx in extract_label:
            arr[arr == idx] = 1
        arr[arr != 1] = 0
        return arr

    def np_extractloop_indexing(arr: np.ndarray):
        arrl = arr.copy()
        for l in extract_label:
            arr += dummy(arrl == l)
        return arr

    def np_extractloop_indexing2(arr: np.ndarray):
        arrl = arr.copy()
        for l in extract_label:
            arrl[arrl != l] = 0
            arrl[arrl == l] = 1
            arr += dummy(arrl)
        return arr

    def np_extractloop_e(arr: np.ndarray):
        arrl = arr.copy()
        for l in extract_label:
            arr += dummy(np_extract_label(arrl, l))
        return arr

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[np_extractloop, np_extractloop_indexing, np_extractloop_indexing2, np_extractloop_e],
        assert_equal_function=lambda x, y: np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
