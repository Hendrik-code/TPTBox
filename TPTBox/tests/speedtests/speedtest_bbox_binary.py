if __name__ == "__main__":
    # speed test np_bbox_binary: 3 full np.any passes vs 2 full passes (3D specialization)
    import itertools
    import random

    import numpy as np

    from TPTBox.core.np_utils import np_is_empty
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def _finalize(bbox, shp, px_dist):
        return tuple(
            slice(max(bbox[i] - px_dist[i // 2], 0), min(bbox[i + 1] + px_dist[i // 2], shp[i // 2]) + 1) for i in range(0, len(bbox), 2)
        )

    def bbox_3pass(img: np.ndarray, px_dist=0, raise_error=True):
        # current implementation (frozen baseline)
        if np_is_empty(img):
            if raise_error:
                raise ValueError("empty")
            return tuple([slice(None)] * img.ndim)
        n = img.ndim
        shp = img.shape
        if isinstance(px_dist, int):
            px_dist = np.ones(n, dtype=np.uint8) * px_dist
        bbox: list = []
        for ax in itertools.combinations(reversed(range(n)), n - 1):
            nonzero = np.any(a=img, axis=ax)
            bbox.extend(np.where(nonzero)[0][[0, -1]])
        return _finalize(bbox, shp, px_dist)

    def bbox_2pass(img: np.ndarray, px_dist=0, raise_error=True):
        if np_is_empty(img):
            if raise_error:
                raise ValueError("empty")
            return tuple([slice(None)] * img.ndim)
        n = img.ndim
        shp = img.shape
        if isinstance(px_dist, int):
            px_dist = np.ones(n, dtype=np.uint8) * px_dist
        if n == 3:
            p = np.any(img, axis=2)  # full pass -> 2D (d0, d1)
            projections = (np.any(p, axis=1), np.any(p, axis=0), np.any(img, axis=(0, 1)))
            bbox: list = []
            for nz in projections:
                w = np.where(nz)[0]
                bbox.extend((w[0], w[-1]))
        else:
            bbox = []
            for ax in itertools.combinations(reversed(range(n)), n - 1):
                nonzero = np.any(a=img, axis=ax)
                bbox.extend(np.where(nonzero)[0][[0, -1]])
        return _finalize(bbox, shp, px_dist)

    functions = [bbox_3pass, bbox_2pass]
    assert_equal = lambda x, y: x == y  # noqa: E731  (tuples of slices compare elementwise)

    def make_input(size, n_points, px_dist):
        def _f():
            num_points = random.randint(1, n_points)
            nii, *_ = get_nii(x=(size, size, size), num_point=num_points)
            return (nii.get_seg_array().astype(np.uint8), px_dist)

        return _f

    for size, npts, px in [(512, 20, 0), (512, 20, 2), (256, 8, 0)]:
        print(f"\n=== np_bbox_binary | size {size}^3, up to {npts} blocks, px_dist={px} ===")
        speed_test(repeats=30, get_input_func=make_input(size, npts, px), functions=functions, assert_equal_function=assert_equal)
