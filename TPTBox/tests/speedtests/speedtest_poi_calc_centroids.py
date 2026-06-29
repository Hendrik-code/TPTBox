if __name__ == "__main__":
    # speed test calc_centroids core: per-label extract_label + crop + scipy center_of_mass loop
    # vs a single np_center_of_mass (cc3d) call that returns every label's centroid at once.
    import random

    import numpy as np
    from scipy.ndimage import center_of_mass

    from TPTBox.core.np_utils import np_center_of_mass
    from TPTBox.tests.speedtests.speedtest import speed_test
    from TPTBox.tests.test_utils import get_nii

    def old_centroids(nii):
        out = {}
        for i in nii.unique():
            m = nii.extract_label(i)
            crop = m.compute_crop()
            m2 = m[crop]
            cm = center_of_mass(m2.get_seg_array())
            out[int(i)] = tuple(round(x + c.start, 3) for x, c in zip(cm, crop))
        return out

    def new_centroids(nii):
        coms = np_center_of_mass(nii.get_seg_array())
        return {int(i): tuple(round(float(x), 3) for x in c) for i, c in coms.items()}

    eq = lambda x, y: x == y  # noqa: E731

    def make(size, npts):
        def _f():
            nii, *_ = get_nii(x=(size, size, size), num_point=random.randint(npts[0], npts[1]))
            return nii

        return _f

    for size, npts in [(200, (8, 16)), (150, (20, 40))]:
        print(f"\n=== calc_centroids ({size}^3, {npts[0]}-{npts[1]} labels) ===")
        speed_test(repeats=30, get_input_func=make(size, npts), functions=[old_centroids, new_centroids], assert_equal_function=eq)
