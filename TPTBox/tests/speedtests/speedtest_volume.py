if __name__ == "__main__":
    # speed test np_volume: cc3d statistics vs np.bincount across label-count regimes.
    # Finding: cc3d wins for FEW labels (typical seg); bincount wins for MANY labels (CC maps).
    import random

    import numpy as np

    from TPTBox.core.np_utils import cc3dstatistics, np_bbox_binary, np_connected_components
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    # ---- candidate implementations -------------------------------------------------

    def np_volume_cc3d(arr: np.ndarray):
        # current implementation (frozen baseline)
        return {idx: i for idx, i in dict(enumerate(cc3dstatistics(arr)["voxel_counts"])).items() if i > 0 and idx != 0}

    def np_volume_bincount_full(arr: np.ndarray):
        counts = np.bincount(arr.ravel())
        return {idx: c for idx, c in enumerate(counts) if c > 0 and idx != 0}

    def np_volume_bincount_crop(arr: np.ndarray):
        crop = np_bbox_binary(arr, raise_error=False, px_dist=2)
        counts = np.bincount(arr[crop].ravel())
        return {idx: c for idx, c in enumerate(counts) if c > 0 and idx != 0}

    def np_volume_hybrid(arr: np.ndarray):
        # cheap max() pass decides: many labels -> bincount, few labels -> cc3d(+crop)
        if int(arr.max()) > 64:
            counts = np.bincount(arr.ravel())
            return {idx: c for idx, c in enumerate(counts) if c > 0 and idx != 0}
        return {idx: i for idx, i in dict(enumerate(cc3dstatistics(arr)["voxel_counts"])).items() if i > 0 and idx != 0}

    functions = [np_volume_cc3d, np_volume_bincount_full, np_volume_bincount_crop, np_volume_hybrid]
    assert_equal = lambda x, y: x == y  # noqa: E731

    # ---- regime 1: few labels (typical segmentation) -------------------------------
    def get_few_uint8():
        nii, *_ = get_nii(x=(256, 256, 256), num_point=random.randint(5, 25))
        return nii.get_seg_array().astype(np.uint8)

    def get_few_uint16():
        nii, *_ = get_nii(x=(256, 256, 256), num_point=random.randint(5, 25))
        return nii.get_seg_array().astype(np.uint16)

    print("\n=== regime 1a: few labels (256^3, uint8) ===")
    speed_test(repeats=20, get_input_func=get_few_uint8, functions=functions, assert_equal_function=assert_equal)

    print("\n=== regime 1b: few labels (256^3, uint16) ===")
    speed_test(repeats=20, get_input_func=get_few_uint16, functions=functions, assert_equal_function=assert_equal)

    # ---- regime 2: many connected components (grid of isolated voxels) -------------
    def get_many_cc():
        base = np.zeros((160, 160, 160), dtype=np.uint8)
        base[::4, ::4, ::4] = 1  # ~64k isolated single-voxel components
        cc, n = np_connected_components(base)
        return cc.astype(np.uint32)

    print("\n=== regime 2: many CC labels (160^3 grid, ~64k labels, uint32) ===")
    speed_test(repeats=15, get_input_func=get_many_cc, functions=functions, assert_equal_function=assert_equal)

    # ---- regime 3: moderate label count (realistic filter_connected_components) -----
    def get_moderate_labels():
        arr = np.random.randint(0, 400, size=(160, 160, 160)).astype(np.uint16)
        return arr

    print("\n=== regime 3: moderate labels (160^3 randint<400, uint16) ===")
    speed_test(repeats=15, get_input_func=get_moderate_labels, functions=functions, assert_equal_function=assert_equal)
