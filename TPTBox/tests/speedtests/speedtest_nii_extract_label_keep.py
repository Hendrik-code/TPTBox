if __name__ == "__main__":
    # speed test NII.extract_label(keep_label=True):
    # current calls get_seg_array() twice (two full copies) + np_extract_label + multiply;
    # optimized uses one copy + np_isin mask + masked write.
    import random

    import numpy as np

    from TPTBox.core.np_utils import np_extract_label, np_isin
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    extract_label = [2, 3, 4, 5]

    def old_keep(nii):
        # frozen baseline: two get_seg_array() copies + np_extract_label + multiply
        seg_arr = nii.get_seg_array()
        seg_arr = np_extract_label(seg_arr, extract_label, to_label=1, inplace=True)
        seg_arr = seg_arr * nii.get_seg_array()
        return seg_arr

    def new_keep(nii):
        seg_arr = nii.get_seg_array()
        seg_arr[~np_isin(seg_arr, extract_label)] = 0
        return seg_arr

    def get_input():
        nii, *_ = get_nii(x=(300, 300, 300), num_point=random.randint(5, 10))
        return nii

    print("\n=== NII.extract_label keep_label=True (300^3, labels [2,3,4,5]) ===")
    speed_test(
        repeats=50, get_input_func=get_input, functions=[old_keep, new_keep], assert_equal_function=lambda x, y: np.array_equal(x, y)
    )
