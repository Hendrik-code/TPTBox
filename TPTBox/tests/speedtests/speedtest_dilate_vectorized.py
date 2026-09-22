if __name__ == "__main__":
    # speed test np_dilate_msk: per-iteration full copy+mask vs boolean (out == i) mask.
    # Dilation has iterative inter-label competition, so the n_pixel loop is preserved;
    # only the per-label per-iteration `out.copy(); data[i != data] = 0` is replaced by `out == i`.
    import random

    import numpy as np
    from scipy.ndimage import generate_binary_structure

    from TPTBox.core.np_utils import _binary_dilation, _to_labels, np_bbox_binary
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def dilate_loop_baseline(arr, label_ref=None, n_pixel=5, connectivity=3, use_crop=True, ignore_axis=None):
        labels = _to_labels(arr, label_ref)
        if use_crop:
            arr_bin = arr.copy()
            arr_bin[np.isin(arr_bin, labels, invert=True)] = 0
            crop = np_bbox_binary(arr_bin, px_dist=1 + n_pixel, raise_error=False)
            arrc = arr[crop]
        else:
            arrc = arr
        if ignore_axis is None:
            struct = generate_binary_structure(arr.ndim, connectivity)
        else:
            struct = generate_binary_structure(arr.ndim - 1, connectivity)
            struct = np.expand_dims(struct, ignore_axis)
        labels = [l for l in labels if l != 0]
        out = arrc
        for _ in range(n_pixel):
            for i in labels:
                data = out.copy()
                data[i != data] = 0
                if use_crop:
                    lcrop = np_bbox_binary(data, px_dist=2 + n_pixel, raise_error=False)
                    data = data[lcrop]
                msk_ibe_data = _binary_dilation(data, struct=struct)
                if use_crop:
                    oc = out[lcrop] == 0
                    out[lcrop][oc] = msk_ibe_data[oc] * i
                else:
                    out[out == 0] = msk_ibe_data[out == 0] * i
        if use_crop:
            arr[crop] = out
            return arr
        return out

    def dilate_m2a(arr, label_ref=None, n_pixel=5, connectivity=3, use_crop=True, ignore_axis=None):
        labels = _to_labels(arr, label_ref)
        if use_crop:
            arr_bin = arr.copy()
            arr_bin[np.isin(arr_bin, labels, invert=True)] = 0
            crop = np_bbox_binary(arr_bin, px_dist=1 + n_pixel, raise_error=False)
            arrc = arr[crop]
        else:
            arrc = arr
        if ignore_axis is None:
            struct = generate_binary_structure(arr.ndim, connectivity)
        else:
            struct = generate_binary_structure(arr.ndim - 1, connectivity)
            struct = np.expand_dims(struct, ignore_axis)
        labels = [l for l in labels if l != 0]
        out = arrc
        for _ in range(n_pixel):
            for i in labels:
                data = out == i  # boolean mask instead of full copy + masked write
                if use_crop:
                    lcrop = np_bbox_binary(data, px_dist=2 + n_pixel, raise_error=False)
                    data = data[lcrop]
                msk_ibe_data = _binary_dilation(data, struct=struct)
                if use_crop:
                    oc = out[lcrop] == 0
                    out[lcrop][oc] = msk_ibe_data[oc] * i
                else:
                    out[out == 0] = msk_ibe_data[out == 0] * i
        if use_crop:
            arr[crop] = out
            return arr
        return out

    functions = [dilate_loop_baseline, dilate_m2a]
    eq = lambda x, y: np.array_equal(x, y)  # noqa: E731

    def get_few():
        nii, *_ = get_nii(x=(150, 150, 150), num_point=random.randint(5, 12))
        return nii.get_seg_array().astype(np.uint8)

    # NOTE: np_dilate_msk mutates in place; the harness runs each function 5x via timeit on the
    # same deep-copied input, so absolute numbers compound across runs. Both candidates mutate
    # identically, so the *relative* comparison stays fair. Dilation is only ever run on
    # few-label anatomical segmentations, so we benchmark that regime only.
    for n_pixel in (1, 3):
        for connectivity in (1, 3):
            print(f"\n=== np_dilate_msk few labels (150^3) | n_pixel={n_pixel} conn={connectivity} ===")
            speed_test(
                repeats=12,
                get_input_func=get_few,
                functions=functions,
                assert_equal_function=eq,
                n_pixel=n_pixel,
                connectivity=connectivity,
            )
