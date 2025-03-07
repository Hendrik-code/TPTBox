if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import _to_labels, np_bbox_binary, np_extract_label, np_unique_withoutzero
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(50, 51)
        nii, points, orientation, sizes = get_nii(x=(350, 350, 150), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(np.uint)
        crop = np_bbox_binary(arr)
        arrc = arr[crop]
        labels = np_unique_withoutzero(arrc)
        # arr[arr == 1] = -1
        # arr_r = arr.copy()
        return arr, crop, labels, arrc

    def uncrop_naive(arr: np.ndarray, crop, labels, arrc: np.ndarray):  # noqa: ARG001
        results = {}
        for l in labels:
            img_l = np_extract_label(arrc, l, inplace=False)
            lcrop = np_bbox_binary(img_l)
            img_lc = img_l[lcrop]
            # Process here
            # need to uncrop somehow
            arrn = np.zeros(arr.shape, dtype=arr.dtype)
            arrn[crop][lcrop] = img_lc
            results[l] = arrn
        return results[1]

    def copy_uncrop(arr: np.ndarray, crop, labels, arrc: np.ndarray):
        zarr = np.zeros((len(labels), *arr.shape), dtype=arr.dtype)
        results = {}
        for idx, l in enumerate(labels):
            img_l = np_extract_label(arrc, l, inplace=False)
            lcrop = np_bbox_binary(img_l)
            img_lc = img_l[lcrop]
            # Process here
            img_lc *= img_lc
            # need to uncrop somehow
            arrn = zarr[idx]
            arrn[crop][lcrop] = img_lc
            results[l] = arrn
        return results[1]

    def uncrop2(arr: np.ndarray, crop, labels, arrc: np.ndarray):
        zarr = np.zeros((len(labels), *arr.shape), dtype=arr.dtype)
        results = {}
        for idx, l in enumerate(labels):
            img_l = np_extract_label(arrc, l, inplace=False)
            # lcrop = np_bbox_binary(img_l)
            img_lc = img_l  # [lcrop]
            # Process here
            img_lc *= img_lc
            # need to uncrop somehow
            arrn = zarr[idx]
            arrn[crop] = img_lc
            results[l] = arrn
        return results[1]

    speed_test(
        repeats=20,
        get_input_func=get_nii_array,
        functions=[uncrop2, copy_uncrop],
        assert_equal_function=lambda x, y: np.all([x[i] == y[i] for i in range(x.shape[0])]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
