if __name__ == "__main__":
    # speed test the mask extraction inside truncate_labels_beyond_reference_:
    # current does two full extract_label(...).get_seg_array() round-trips; the two binary masks
    # can be obtained directly from one get_array() via np_isin.
    import random

    import numpy as np

    from TPTBox.core.np_utils import np_isin
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    idx = 1
    not_beyond = 2

    def old_masks(nii):
        nii.get_array()
        cond = nii.extract_label(idx).get_seg_array() == 1
        nb = nii.extract_label(not_beyond).get_seg_array() == 1
        return (cond, nb)

    def new_masks(nii):
        np_array = nii.get_array()
        cond = np_isin(np_array, idx)
        nb = np_isin(np_array, not_beyond)
        return (cond, nb)

    def get_input():
        nii, *_ = get_nii(x=(300, 300, 300), num_point=random.randint(5, 10))
        return nii

    print("\n=== truncate_labels_beyond_reference_ mask extraction (300^3) ===")
    speed_test(
        repeats=50,
        get_input_func=get_input,
        functions=[old_masks, new_masks],
        assert_equal_function=lambda x, y: np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]),
    )
