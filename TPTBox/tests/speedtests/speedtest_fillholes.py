if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import _to_labels, np_bbox_binary, np_connected_components, np_fill_holes
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 10)
        nii, points, orientation, sizes = get_nii(x=(350, 350, 150), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return nii, arr

    def nii_fill_holes(nii: NII, arr):  # noqa: ARG001
        return nii.fill_holes(use_crop=True).get_seg_array()

    def np_nfill_holes(nii: NII, arr):  # noqa: ARG001
        return np_fill_holes(arr, use_crop=False)

    def np_nfill_holes_crop(nii, arr: np.ndarray):  # noqa: ARG001
        return np_fill_holes(arr, use_crop=True)

    speed_test(
        repeats=10,
        get_input_func=get_nii_array,
        functions=[nii_fill_holes, np_nfill_holes_crop],
        assert_equal_function=lambda x, y: np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
