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
        nii, points, orientation, sizes = get_nii(x=(500, 500, 500), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint8)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr

    # def nii_extract(nii: NII):
    #    return nii.extract_label([1, 2, 3, 4, 5]).get_seg_array()

    extract_one_label = 2
    extract_label = [2, 3, 4, 5]

    def dummy(arr_bin: np.ndarray):
        return arr_bin + arr_bin

    # EXTRACT ONE LABEL

    def np_extract_one(arr: np.ndarray):
        return np_extract_label(arr, extract_one_label)

    def np_extract_one_nii(arr: np.ndarray):
        arr = arr.copy()
        arr[arr != extract_one_label] = 0
        arr[arr == extract_one_label] = 1
        return arr

    def np_extract_one_equal(arr: np.ndarray):
        return arr == extract_one_label

    # EXTRACT LIST OF LABELS

    def np_extractlist(arr: np.ndarray):
        return np_extract_label(arr, extract_label, inplace=False)

    def np_extractlist_isin(arr: np.ndarray):
        arr = arr.copy()
        arr_msk = np.isin(arr, extract_label)
        arr[arr_msk] = 1
        arr[~arr_msk] = 0
        return arr

    def np_extractlist_for(arr: np.ndarray):
        arr = arr.copy()
        if 1 not in extract_label:
            arr[arr == 1] = 0
        for idx in extract_label:
            arr[arr == idx] = 1
        arr[arr != 1] = 0
        return arr

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[np_extractlist, np_extractlist_isin, np_extractlist_for],
        assert_equal_function=lambda x, y: np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
