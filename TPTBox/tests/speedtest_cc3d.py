if __name__ == "__main__":
    # speed test dilation
    import random

    import numpy as np
    from cc3d import statistics as cc3dstats
    from scipy.ndimage import center_of_mass

    from TPTBox.core.np_utils import (
        _to_labels,
        np_approx_center_of_mass,
        np_bbox_binary,
        np_bounding_boxes,
        np_center_of_mass,
        np_map_labels,
        np_unique,
        np_unique_withoutzero,
        np_volume,
    )
    from TPTBox.tests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def get_nii_array():
        num_points = random.randint(1, 30)
        nii, points, orientation, sizes = get_nii(x=(140, 140, 150), num_point=num_points)
        # nii.map_labels_({1: -1}, verbose=False)
        arr = nii.get_seg_array().astype(int)
        arr[arr == 1] = -1
        arr_r = arr.copy()
        return arr_r

    def cc3d_com(arr: np.ndarray):
        return np_center_of_mass(arr)

    def center_of_mass_one(arr: np.ndarray):
        coms = center_of_mass(arr)
        return coms

    def center_of_mass_(arr: np.ndarray):
        cc_label_set = np.unique(arr)
        coms = {}
        for c in cc_label_set:
            if c == 0:
                continue
            c_l = arr.copy()
            c_l[c_l != c] = 0
            com = center_of_mass(c_l)
            coms[c] = com
        return coms

    def bbox_(arr: np.ndarray):
        cc_label_set = np.unique(arr)
        coms = {}
        for c in cc_label_set:
            if c == 0:
                continue
            c_l = arr.copy()
            c_l[c_l != c] = 0
            com = np_bbox_binary(c_l)
            coms[c] = com
        return coms

    # def cc3d_volume(arr: np.ndarray):
    #    volumes = dict(enumerate(cc3dstats(arr)["voxel_counts"]))
    #    volumes.pop(0)
    #    return volumes

    # def cc3d_countnonzero(arr: np.ndarray):
    #    return sum(cc3d_volume(arr).values())

    # def cc3d_bbox(arr: np.ndarray):
    #    return dict(enumerate(cc3dstats(arr)["bounding_boxes"]))

    # def cc3d_unique(arr: np.ndarray) -> list[int]:
    #    return [i for i, v in cc3d_volume(arr).items() if v > 0]

    # a = np.ones((100, 100, 50), dtype=np.uint16)
    # a[0, 0, 0] = 0
    # print(type(a[0, 0, 0]))
    # print(cc3d_volume(a))

    arr = get_nii_array()
    # print(np_unique_withoutzero(arr))
    # print("npunique", np.unique(arr))
    # print()
    # print(center_of_mass_(arr))
    # print()
    # print(np_center_of_mass(arr))
    print()
    print(np_unique(arr))
    print(np.unique(arr))

    speed_test(
        repeats=50,
        get_input_func=get_nii_array,
        functions=[np_unique, np.unique],
        assert_equal_function=lambda x, y: True,  # np.all([x[i] == y[i] for i in range(len(x))]),  # noqa: ARG005
        # np.all([x[i] == y[i] for i in range(len(x))])
    )
    # print(time_measures)
