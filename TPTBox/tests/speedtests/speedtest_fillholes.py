if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import _fill, _to_labels, np_bbox_binary, np_connected_components, np_extract_label, np_fill_holes
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 10)
        nii, points, orientation, sizes = get_nii(x=(150, 150, 150), num_point=num_points)
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

    def np_fill_holes_extract2(nii, arr: np.ndarray):  # noqa: ARG001
        slice_wise_dim = None
        assert 2 <= arr.ndim <= 3
        assert arr.ndim == 3 or slice_wise_dim is None, "slice_wise_dim set but array is 3D"
        labels: list[int] = _to_labels(arr, None)

        gcrop = np_bbox_binary(arr, px_dist=1)
        arrc = arr[gcrop]

        for l in labels:  # type:ignore
            arr_l = np_extract_label(arrc, l)
            crop = np_bbox_binary(arr_l, px_dist=1)
            arr_lc = arr_l[crop]

            if slice_wise_dim is None:
                filled = _fill(arr_lc).astype(arr.dtype)
            else:
                assert 0 <= slice_wise_dim <= arr.ndim - 1, f"slice_wise_dim needs to be in range [0, {arr.ndim - 1}]"
                filled = np.swapaxes(arr_lc, 0, slice_wise_dim)
                filled = np.stack([_fill(x) for x in filled])
                filled = np.swapaxes(filled, 0, slice_wise_dim)
            filled[filled != 0] = l
            arrc[crop][arrc[crop] == 0] = filled[arrc[crop] == 0]

        arr[gcrop] = arrc
        return arr

    def np_fill_holes_extract(nii, arr: np.ndarray):  # noqa: ARG001
        slice_wise_dim = None
        assert 2 <= arr.ndim <= 3
        assert arr.ndim == 3 or slice_wise_dim is None, "slice_wise_dim set but array is 3D"
        labels: list[int] = _to_labels(arr, None)

        gcrop = np_bbox_binary(arr, px_dist=1)
        arrc = arr[gcrop]

        for l in labels:  # type:ignore
            arr_l = arrc == l
            crop = np_bbox_binary(arr_l, px_dist=1)
            arr_lc = arr_l[crop]

            if slice_wise_dim is None:
                filled = _fill(arr_lc).astype(arr.dtype)
            else:
                assert 0 <= slice_wise_dim <= arr.ndim - 1, f"slice_wise_dim needs to be in range [0, {arr.ndim - 1}]"
                filled = np.swapaxes(arr_lc, 0, slice_wise_dim)
                filled = np.stack([_fill(x) for x in filled])
                filled = np.swapaxes(filled, 0, slice_wise_dim)
            filled[filled != 0] = l

            arrc[crop][arrc[crop] == 0] = filled[arrc[crop] == 0]

        arr[gcrop] = arrc
        return arr

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[np_nfill_holes, np_fill_holes_extract, np_nfill_holes_crop],
        assert_equal_function=lambda x, y: np.array_equal(x, y),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
