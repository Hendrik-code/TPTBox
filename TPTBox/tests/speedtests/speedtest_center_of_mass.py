if __name__ == "__main__":
    # speed test np_center_of_mass / np_bounding_boxes:
    # drop the O(max_label x n_unique) `idx in unique` list-membership post-processing.
    import numpy as np

    from TPTBox.core.np_utils import cc3dstatistics, np_connected_components
    from TPTBox.tests.speedtests.speedtest import speed_test

    # ---- center of mass ------------------------------------------------------------
    def com_current(arr: np.ndarray):
        stats = cc3dstatistics(arr, use_crop=False)
        unique = [idx for idx, i in enumerate(stats["voxel_counts"]) if i > 0 and idx != 0]
        return {idx: v for idx, v in enumerate(stats["centroids"]) if idx in unique}

    def com_direct(arr: np.ndarray):
        stats = cc3dstatistics(arr, use_crop=False)
        vc = stats["voxel_counts"]
        return {idx: v for idx, v in enumerate(stats["centroids"]) if idx != 0 and vc[idx] > 0}

    # ---- bounding boxes ------------------------------------------------------------
    def bbox_current(arr: np.ndarray):
        stats = cc3dstatistics(arr)
        unique = [idx for idx, i in enumerate(stats["voxel_counts"]) if i > 0 and idx != 0]
        return {idx: v for idx, v in enumerate(stats["bounding_boxes"]) if idx in unique}

    def bbox_direct(arr: np.ndarray):
        stats = cc3dstatistics(arr)
        vc = stats["voxel_counts"]
        return {idx: v for idx, v in enumerate(stats["bounding_boxes"]) if idx != 0 and vc[idx] > 0}

    def com_equal(x, y):
        return x.keys() == y.keys() and all(np.allclose(x[k], y[k]) for k in x)

    def bbox_equal(x, y):
        return x == y

    def get_many_cc():
        # grid of isolated single voxels -> ~4k connected components (high max label).
        # exercises the O(max_label x n_unique) post-processing the optimization removes.
        base = np.zeros((64, 64, 64), dtype=np.uint8)
        base[::4, ::4, ::4] = 1
        cc, _ = np_connected_components(base)
        return cc.astype(np.uint32)

    def get_moderate_labels():
        return np.random.randint(0, 2000, size=(120, 120, 120)).astype(np.uint16)

    for name, gen in [("~4k CC labels (64^3 grid)", get_many_cc), ("~2000 labels (120^3 randint)", get_moderate_labels)]:
        print(f"\n=== np_center_of_mass | {name} ===")
        speed_test(repeats=12, get_input_func=gen, functions=[com_current, com_direct], assert_equal_function=com_equal)
        print(f"\n=== np_bounding_boxes | {name} ===")
        speed_test(repeats=12, get_input_func=gen, functions=[bbox_current, bbox_direct], assert_equal_function=bbox_equal)
