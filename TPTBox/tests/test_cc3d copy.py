if __name__ == "__main__":
    # speed test dilation
    import random
    from time import perf_counter

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import (
        _to_labels,
        _connected_components,
        np_bbox_binary,
        np_connected_components,
        np_extract_label,
        np_unique,
        np_volume,
        np_filter_connected_components,
        np_filter_connected_components2,
    )
    from TPTBox.tests.test_utils import get_nii

    arr = np.array(
        [
            [0, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 2, 2, 1, 3, 0],
            [0, 0, 0, 3, 3, 0],
            [0, 0, 0, 1, 3, 0],
        ],
        dtype=np.uint8,
    )
    arr = arr.astype(np.uint8)

    kwargs = {
        # "min_volume": 3,
        "max_volume": 4,
        "largest_k_components": 1,
    }

    b = np_filter_connected_components(arr.copy(), **kwargs)
    print(b)
    c = np_filter_connected_components2(arr.copy(), **kwargs)
    print(c)
    print()
    print(np_volume(b))
    print(np_volume(c))

    print()
    print(np.array_equal(b, c))
