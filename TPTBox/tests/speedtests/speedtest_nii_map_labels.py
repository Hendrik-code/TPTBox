if __name__ == "__main__":
    # speed test NII.map_labels with verbose=False:
    # current computes np_unique BEFORE and AFTER mapping purely to feed log.print(verbose=verbose),
    # i.e. two wasted full-array scans when verbose=False. Optimized skips them.
    import random

    import numpy as np

    from TPTBox.core.np_utils import np_map_labels, np_unique
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    label_map_ = {1: 10, 2: 11, 3: 12, 4: 13, 5: 14}

    def old_map_verbose_false(arr):
        # mirrors current map_labels internals with verbose=False (still scans!)
        labels_before = [v for v in np_unique(arr) if v > 0]  # noqa: F841
        data = np_map_labels(arr, label_map_)
        labels_after = [v for v in np_unique(data) if v > 0]  # noqa: F841
        return data.astype(np.uint16)

    def new_map_verbose_false(arr):
        data = np_map_labels(arr, label_map_)
        return data.astype(np.uint16)

    def get_input():
        nii, *_ = get_nii(x=(300, 300, 300), num_point=random.randint(5, 10))
        return nii.get_seg_array().astype(np.uint8)

    print("\n=== NII.map_labels verbose=False (300^3): skip the two logging np_unique scans ===")
    speed_test(
        repeats=50,
        get_input_func=get_input,
        functions=[old_map_verbose_false, new_map_verbose_false],
        assert_equal_function=lambda x, y: np.array_equal(x, y),
    )
