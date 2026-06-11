if __name__ == "__main__":
    # speed test html_preview._get_mesh: reorient_()+rescale_() were done per label (inside
    # from_segmentation_nii(extract_label(u))). Hoisting them out of the loop reorients/rescales
    # the image once. reorient/rescale commute with extract_label for nearest-neighbour seg
    # resampling, so each label's array (and thus its marching-cubes mesh) is unchanged.
    import nibabel as nib
    import numpy as np

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.tests.speedtests.speedtest import speed_test

    def old_per_label(nii):
        out = {}
        for u in nii.unique():
            m = nii.extract_label(u)  # NII
            m.reorient_()
            m.rescale_()
            out[u] = m.get_seg_array()
        return out

    def new_hoisted(nii):
        base = nii.reorient()
        base = base.rescale()
        return {u: base.extract_label(u).get_seg_array() for u in base.unique()}

    def eq(x, y):
        return set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x)

    def make(n_labels, spacing_z):
        def _f():
            aff = np.diag([1.0, 1.0, float(spacing_z), 1.0])
            arr = np.random.randint(0, n_labels, size=(96, 98, 44)).astype(np.uint16)
            return NII(nib.Nifti1Image(arr, aff), seg=True)

        return _f

    for n_labels, sz in [(12, 3.0), (25, 2.0)]:
        print(f"\n=== html_preview per-label reorient+rescale ({n_labels} labels, z-spacing {sz}) ===")
        speed_test(repeats=10, get_input_func=make(n_labels, sz), functions=[old_per_label, new_hoisted], assert_equal_function=eq)
