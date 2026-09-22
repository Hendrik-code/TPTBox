if __name__ == "__main__":
    # speed test POI global coordinate conversion: per-point local_to_global / global_to_local
    # loops vs a single batched (N,3) affine matmul.
    import nibabel as nib
    import numpy as np

    from TPTBox.core.nii_wrapper import NII
    from TPTBox.core.poi import calc_centroids
    from TPTBox.core.vert_constants import ROUNDING_LVL
    from TPTBox.tests.speedtests.speedtest import speed_test

    def make_poi(n_labels):
        def _f():
            arr = np.random.randint(0, n_labels, size=(110, 110, 110)).astype(np.uint16)
            nii = NII(nib.Nifti1Image(arr, affine=np.diag([1.5, 1.5, 2.0, 1.0])), seg=True)
            return calc_centroids(nii, second_stage=50)

        return _f

    def old_to_global(poi):
        out = {}
        for k1, k2, v in poi.items():
            out[(k1, k2)] = poi.local_to_global(v, False)
        return out

    def new_to_global(poi):
        items = list(poi.items())
        arr = np.asarray([v for _, _, v in items])
        a = (arr * np.asarray(poi.zoom)) @ np.asarray(poi.rotation).T + np.asarray(poi.origin)
        a = np.round(a, ROUNDING_LVL)
        return {(k1, k2): tuple(row) for (k1, k2, _), row in zip(items, a.tolist())}

    eq = lambda x, y: x == y  # noqa: E731

    for n_labels in (100, 400):
        print(f"\n=== POI.to_global affine ({n_labels} points) ===")
        speed_test(repeats=30, get_input_func=make_poi(n_labels), functions=[old_to_global, new_to_global], assert_equal_function=eq)
