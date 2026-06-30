if __name__ == "__main__":
    # speed test NII.remove_labels: per-label `seg_arr[seg_arr == l] = 0` loop (N masked passes)
    # vs a single np_map_labels gather ({label: removed_to_label}).
    import random

    import numpy as np

    from TPTBox.core.np_utils import np_isin, np_map_labels
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def old_remove(arr, remove):
        arr = arr.copy()
        for l in remove:
            arr[arr == l] = 0
        return arr

    def new_remove_map(arr, remove):
        return np_map_labels(arr, dict.fromkeys(remove, 0))

    def new_remove_isin(arr, remove):
        arr = arr.copy()
        arr[np_isin(arr, remove)] = 0
        return arr

    functions = [old_remove, new_remove_map, new_remove_isin]
    eq = lambda x, y: np.array_equal(x, y)  # noqa: E731

    # realistic: get_nii blocks (sparse foreground)
    def get_sparse(n_remove):
        def _f():
            nii, *_ = get_nii(x=(300, 300, 300), num_point=random.randint(n_remove + 2, n_remove + 14))
            return (nii.get_seg_array().astype(np.uint16), list(range(2, 2 + n_remove)))

        return _f

    for n_remove in (6, 20):
        print(f"\n=== NII.remove_labels (300^3 get_nii blocks, remove {n_remove} labels, uint16) ===")
        speed_test(repeats=40, get_input_func=get_sparse(n_remove), functions=functions, assert_equal_function=eq)

    # dense: many abundant labels
    def get_dense(n_remove):
        def _f():
            return (np.random.randint(0, 60, size=(200, 200, 200)).astype(np.uint16), list(range(2, 2 + n_remove)))

        return _f

    print("\n=== NII.remove_labels (200^3 dense randint<60, remove 20 labels, uint16) ===")
    speed_test(repeats=40, get_input_func=get_dense(20), functions=functions, assert_equal_function=eq)
