if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import (
        _to_labels,
        np_dilate_msk,
        np_erode_msk,
    )
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 10)
        nii, points, orientation, sizes = get_nii(x=(100, 100, 100), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        # arr = nii.get_seg_array().astype(int)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return nii

    def nii_dilate_withoutcrop(nii: NII):
        return nii.dilate_msk_(n_pixel=1, use_crop=False).get_seg_array()

    def nii_dilate_withcrop(nii: NII):
        return nii.dilate_msk_(n_pixel=1, use_crop=True).get_seg_array()

    def np_dilate_withoutcrop(nii: NII):
        return np_dilate_msk(nii.get_seg_array(), n_pixel=1, use_crop=False)

    def np_dilate_withcrop(nii: NII):
        return np_dilate_msk(nii.get_seg_array(), n_pixel=1, use_crop=True)

    def np_dilate_withLcrop(nii: NII):
        return np_dilate_msk(nii.get_seg_array(), n_pixel=1, use_crop=False, use_local_crop=True)

    def np_dilate_withBcrop(nii: NII):
        return np_dilate_msk(nii.get_seg_array(), n_pixel=1, use_crop=True, use_local_crop=True)

    #### ERODE

    def nii_erode_withoutcrop(nii: NII):
        return nii.erode_msk_(n_pixel=1, use_crop=False).get_seg_array()

    def nii_erode_withcrop(nii: NII):
        return nii.erode_msk_(n_pixel=2, use_crop=True).get_seg_array()

    def np_erode_withoutcrop(nii: NII):
        return np_erode_msk(nii.get_seg_array(), n_pixel=1, use_crop=False)

    def np_erode_withcrop(nii: NII):
        return np_erode_msk(nii.get_seg_array(), n_pixel=1, use_crop=True)

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[
            np_erode_withoutcrop,
            np_erode_withcrop,
        ],
        assert_equal_function=lambda x, y: np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
