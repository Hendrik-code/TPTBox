if __name__ == "__main__":
    # speed test dilation
    import random
    from time import perf_counter

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.np_utils import _to_labels, connected_components, np_bbox_binary, np_connected_components, np_extract_label, np_unique
    from TPTBox.tests.test_utils import get_nii

    arr = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ]
    )

    labels_out = connected_components(arr, connectivity=4, return_N=False)
    print(arr)
    print(labels_out)

    # crop and uncrop
    #
    crop = np_bbox_binary(arr)
    arr_c = arr[crop]
    print(crop)
    start = perf_counter()
