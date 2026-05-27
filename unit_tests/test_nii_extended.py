"""Extended unit tests for NII: properties, label operations, spatial ops, and morphology."""

from __future__ import annotations

import unittest

import numpy as np

from TPTBox import NII
from TPTBox.tests.test_utils import get_nii, get_random_ax_code, repeats


def _make_nii(arr: np.ndarray, seg: bool = True, zoom=(1.0, 1.0, 1.0)) -> NII:
    affine = np.diag([*zoom, 1.0])
    return NII.from_numpy(arr, affine=affine, seg=seg)


def _make_seg(shape=(20, 20, 20), num_labels=3) -> NII:
    arr = np.zeros(shape, dtype=np.uint8)
    for i in range(1, num_labels + 1):
        s = slice(i * 3, i * 3 + 4)
        arr[s, s, s] = i
    return _make_nii(arr)


def _make_two_label_seg() -> NII:
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[1:6, 1:6, 1:6] = 1
    arr[10:15, 10:15, 10:15] = 2
    return _make_nii(arr)


class Test_NII_Properties(unittest.TestCase):
    def test_from_numpy_shape(self):
        arr = np.zeros((10, 12, 14), dtype=np.uint8)
        nii = _make_nii(arr)
        self.assertEqual(nii.shape, (10, 12, 14))

    def test_from_numpy_seg_flag(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        self.assertTrue(_make_nii(arr, seg=True).seg)
        self.assertFalse(_make_nii(arr, seg=False).seg)

    def test_from_numpy_round_trip(self):
        arr = np.arange(27, dtype=np.int16).reshape(3, 3, 3)
        nii = _make_nii(arr, seg=False)
        np.testing.assert_array_equal(nii.get_array(), arr)

    def test_zoom_from_affine(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        nii = _make_nii(arr, zoom=(2.0, 3.0, 4.0))
        self.assertAlmostEqual(nii.zoom[0], 2.0)
        self.assertAlmostEqual(nii.zoom[1], 3.0)
        self.assertAlmostEqual(nii.zoom[2], 4.0)

    def test_voxel_volume(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        nii = _make_nii(arr, zoom=(2.0, 3.0, 4.0))
        self.assertAlmostEqual(nii.voxel_volume(), 24.0)

    def test_is_empty_true(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        self.assertTrue(_make_nii(arr).is_empty)

    def test_is_empty_false(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        self.assertFalse(_make_nii(arr).is_empty)

    def test_unique_returns_nonzero_labels(self):
        nii = _make_seg(num_labels=4)
        self.assertEqual(sorted(nii.unique()), [1, 2, 3, 4])

    def test_unique_excludes_background(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        self.assertEqual(_make_nii(arr).unique(), [])

    def test_unique_after_modification(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 5
        nii = _make_nii(arr)
        self.assertEqual(nii.unique(), [5])

    def test_volumes_correct_counts(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[0:3, 0:3, 0:3] = 1  # 27 voxels
        arr[5:7, 5:7, 5:7] = 2  # 8 voxels
        vols = _make_nii(arr).volumes()
        self.assertEqual(vols[1], 27)
        self.assertEqual(vols[2], 8)

    def test_volumes_sum(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[0:4, 0:4, 0:4] = 1
        arr[5:8, 5:8, 5:8] = 2
        vols = _make_nii(arr).volumes()
        self.assertEqual(sum(vols.values()), int((arr > 0).sum()))

    def test_center_of_masses_single_voxel(self):
        arr = np.zeros((11, 11, 11), dtype=np.uint8)
        arr[5, 6, 7] = 1
        coms = _make_nii(arr).center_of_masses()
        self.assertIn(1, coms)
        for coord, expected in zip(coms[1], (5.0, 6.0, 7.0)):
            self.assertAlmostEqual(coord, expected, places=3)


class Test_NII_DTypeAndCopy(unittest.TestCase):
    def test_set_dtype_float32(self):
        arr = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)
        nii = _make_nii(arr).set_dtype(np.float32)
        self.assertEqual(nii.get_array().dtype, np.float32)
        np.testing.assert_array_almost_equal(nii.get_array(), arr.astype(np.float32))

    def test_astype_alias(self):
        arr = np.ones((4, 4, 4), dtype=np.uint8)
        nii = _make_nii(arr, seg=False).astype(np.float64)
        self.assertEqual(nii.get_array().dtype, np.float64)

    def test_set_dtype_preserves_values(self):
        arr = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
        nii = _make_nii(arr, seg=False).set_dtype(np.float32)
        np.testing.assert_array_equal(nii.get_array(), arr.astype(np.float32))

    def test_copy_is_independent(self):
        nii = _make_two_label_seg()
        nii2 = nii.copy()
        nii2.set_array_(nii2.get_array() + 1)
        self.assertFalse(np.array_equal(nii.get_array(), nii2.get_array()))

    def test_clone_matches_copy(self):
        nii = _make_two_label_seg()
        np.testing.assert_array_equal(nii.copy().get_array(), nii.clone().get_array())

    def test_set_array_replaces_data(self):
        nii = _make_two_label_seg()
        new_arr = np.ones((20, 20, 20), dtype=np.uint8) * 7
        nii2 = nii.set_array(new_arr)
        np.testing.assert_array_equal(nii2.get_array(), new_arr)

    def test_set_array_non_inplace(self):
        nii = _make_two_label_seg()
        orig_arr = nii.get_array().copy()
        new_arr = np.zeros((20, 20, 20), dtype=np.uint8)
        nii.set_array(new_arr)
        # original nii is unchanged (non-inplace)
        np.testing.assert_array_equal(nii.get_array(), orig_arr)


class Test_NII_LabelOps(unittest.TestCase):
    def test_extract_label_binary(self):
        nii = _make_two_label_seg()
        ext = nii.extract_label(1)
        arr = ext.get_array()
        orig = nii.get_array()
        # label-2 region is zeroed
        self.assertTrue(np.all(arr[orig == 2] == 0))
        # label-1 region is retained
        self.assertTrue(np.all(arr[orig == 1] == 1))

    def test_extract_label_list(self):
        # extract_label always binarises: all matched labels → 1, rest → 0.
        nii = _make_seg(num_labels=3)
        ext = nii.extract_label([1, 3])
        labels = sorted(ext.unique())
        # Only 1 is present (both 1 and 3 become 1); 2 is excluded.
        self.assertEqual(labels, [1])
        # Volume of extracted mask equals volume of labels 1 + 3 in original.
        vols = nii.volumes()
        self.assertEqual(int(ext.volumes().get(1, 0)), vols[1] + vols[3])

    def test_remove_labels_zeros_out(self):
        nii = _make_two_label_seg()
        result = nii.remove_labels(1)
        self.assertNotIn(1, result.unique())
        self.assertIn(2, result.unique())

    def test_remove_labels_multiple(self):
        nii = _make_seg(num_labels=3)
        result = nii.remove_labels([1, 2])
        self.assertNotIn(1, result.unique())
        self.assertNotIn(2, result.unique())
        self.assertIn(3, result.unique())

    def test_apply_mask_zeroes_outside(self):
        arr = np.ones((10, 10, 10), dtype=np.uint8)
        nii = _make_nii(arr)
        mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_arr[3:7, 3:7, 3:7] = 1
        mask = _make_nii(mask_arr)
        result = nii.apply_mask(mask)
        res_arr = result.get_array()
        self.assertTrue(np.all(res_arr[:3, :, :] == 0))
        self.assertTrue(np.all(res_arr[3:7, 3:7, 3:7] == 1))

    def test_map_labels_remaps_ids(self):
        nii = _make_two_label_seg()
        mapped = nii.map_labels({1: 10, 2: 20})
        self.assertIn(10, mapped.unique())
        self.assertIn(20, mapped.unique())
        self.assertNotIn(1, mapped.unique())
        self.assertNotIn(2, mapped.unique())

    def test_map_labels_preserves_volume(self):
        nii = _make_two_label_seg()
        vols_before = nii.volumes()
        mapped = nii.map_labels({1: 10, 2: 20})
        vols_after = mapped.volumes()
        self.assertEqual(vols_before[1], vols_after[10])
        self.assertEqual(vols_before[2], vols_after[20])

    def test_map_labels_inplace(self):
        nii = _make_two_label_seg()
        nii.map_labels_({1: 99})
        self.assertIn(99, nii.unique())
        self.assertNotIn(1, nii.unique())


class Test_NII_Spatial(unittest.TestCase):
    def test_flip_reverses_axis(self):
        arr = np.zeros((10, 5, 5), dtype=np.uint8)
        arr[0, :, :] = 1  # marker at start of axis 0
        nii = _make_nii(arr)
        flipped = nii.flip(0)
        arr_f = flipped.get_array()
        self.assertTrue(np.all(arr_f[-1, :, :] == 1))
        self.assertTrue(np.all(arr_f[0, :, :] == 0))

    def test_flip_double_is_identity(self):
        arr = np.random.randint(0, 4, (10, 8, 6), dtype=np.uint8)
        nii = _make_nii(arr)
        np.testing.assert_array_equal(nii.flip(0).flip(0).get_array(), arr)

    def test_pad_to_increases_shape(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        nii = _make_nii(arr)
        padded = nii.pad_to((10, 10, 10))
        self.assertEqual(padded.shape, (10, 10, 10))
        self.assertIn(1, padded.unique())

    def test_compute_crop_reduces_shape(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[5:10, 5:10, 5:10] = 1
        nii = _make_nii(arr)
        crop = nii.compute_crop()
        cropped = nii.apply_crop(crop)
        self.assertLessEqual(max(cropped.shape), 20)
        self.assertIn(1, cropped.unique())

    def test_get_histogram_total_count(self):
        arr = np.array([[[1, 1, 2, 2, 3]]], dtype=np.uint8)
        nii = _make_nii(arr)
        counts, bins = nii.get_histogram()
        self.assertEqual(int(counts.sum()), 5)

    def test_get_intersecting_volume_same_fov(self):
        # get_intersecting_volume binarises b's entire field-of-view, then
        # counts how many voxels in self's grid are within b's extent.
        # Two images with the same shape and affine fully overlap.
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        nii1 = _make_nii(arr)
        nii2 = _make_nii(arr)
        self.assertEqual(nii1.get_intersecting_volume(nii2), 1000)

    def test_get_intersecting_volume_partial(self):
        # Smaller image whose field-of-view is a strict subset → count < nii1's total voxels.
        arr_large = np.zeros((10, 10, 10), dtype=np.uint8)
        arr_small = np.zeros((5, 5, 5), dtype=np.uint8)
        nii_large = _make_nii(arr_large)
        # small image placed at origin covers 5x5x5 = 125 voxels in nii_large's 10x10x10 grid
        nii_small = _make_nii(arr_small)
        overlap = nii_large.get_intersecting_volume(nii_small)
        self.assertEqual(overlap, 125)

    def test_boundary_mask_nonzero(self):
        # boundary_mask uses threshold to distinguish foreground (>threshold) from background.
        # threshold must be between 0 and the actual foreground value.
        arr = np.zeros((15, 15, 15), dtype=np.uint8)
        arr[3:12, 3:12, 3:12] = 1
        nii = _make_nii(arr, seg=True)
        bm = nii.boundary_mask(threshold=0.5)
        bm_arr = bm.get_array()
        # Foreground voxels should appear in the result.
        self.assertGreater(int((bm_arr != 0).sum()), 0)

    def test_compute_surface_mask_smaller_than_original(self):
        arr = np.zeros((15, 15, 15), dtype=np.uint8)
        arr[2:13, 2:13, 2:13] = 1
        nii = _make_nii(arr)
        surface = nii.compute_surface_mask()
        orig_vol = int((nii.get_array() > 0).sum())
        surf_vol = int((surface.get_array() > 0).sum())
        self.assertLess(surf_vol, orig_vol)


class Test_NII_Smoothing(unittest.TestCase):
    def test_smooth_gaussian_same_shape(self):
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        nii = _make_nii(arr, seg=False)
        smoothed = nii.smooth_gaussian(sigma=1.0)
        self.assertEqual(smoothed.shape, nii.shape)

    def test_smooth_gaussian_reduces_variance(self):
        arr = np.random.rand(20, 20, 20).astype(np.float32)
        nii = _make_nii(arr, seg=False)
        smoothed = nii.smooth_gaussian(sigma=2.0)
        self.assertLess(smoothed.get_array().std(), arr.std())

    def test_smooth_gaussian_inplace(self):
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        nii = _make_nii(arr, seg=False)
        nii.smooth_gaussian_(sigma=1.0)
        self.assertEqual(nii.shape, (10, 10, 10))


class Test_NII_Morphology_Euclid(unittest.TestCase):
    @staticmethod
    def _sphere_seg(radius=6, shape=(25, 25, 25)):
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        arr = np.zeros(shape, dtype=np.uint8)
        xs, ys, zs = np.ogrid[: shape[0], : shape[1], : shape[2]]
        arr[(xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2 <= radius**2] = 1
        return _make_nii(arr)

    def test_erode_msk_euclid_reduces_volume(self):
        nii = self._sphere_seg()
        eroded = nii.erode_msk_euclid(n_pixel=2)
        orig_vol = int((nii.get_array() > 0).sum())
        ero_vol = int((eroded.get_array() > 0).sum())
        self.assertLess(ero_vol, orig_vol)

    def test_dilate_msk_euclid_increases_volume(self):
        nii = self._sphere_seg()
        dilated = nii.dilate_msk_euclid(n_pixel=2)
        orig_vol = int((nii.get_array() > 0).sum())
        dil_vol = int((dilated.get_array() > 0).sum())
        self.assertGreater(dil_vol, orig_vol)

    def test_erode_then_dilate_is_subset(self):
        nii = self._sphere_seg()
        processed = nii.erode_msk_euclid(n_pixel=2).dilate_msk_euclid(n_pixel=2)
        orig_arr = nii.get_array()
        proc_arr = processed.get_array()
        # no new voxels should appear outside the original mask
        self.assertTrue(np.all(proc_arr[orig_arr == 0] == 0))

    def test_erode_msk_euclid_inplace(self):
        nii = self._sphere_seg()
        orig_vol = int((nii.get_array() > 0).sum())
        nii.erode_msk_euclid(n_pixel=2, inplace=True)
        ero_vol = int((nii.get_array() > 0).sum())
        self.assertLess(ero_vol, orig_vol)

    def test_dilate_msk_euclid_inplace(self):
        nii = self._sphere_seg()
        orig_vol = int((nii.get_array() > 0).sum())
        nii.dilate_msk_euclid(n_pixel=2, inplace=True)
        dil_vol = int((nii.get_array() > 0).sum())
        self.assertGreater(dil_vol, orig_vol)

    def test_erode_msk_euclid_zero_pixel_unchanged(self):
        nii = self._sphere_seg()
        orig_arr = nii.get_array().copy()
        eroded = nii.erode_msk_euclid(n_pixel=0)
        np.testing.assert_array_equal(eroded.get_array(), orig_arr)


class Test_NII_Reorient_Rescale(unittest.TestCase):
    """Verify that reorient/rescale leave the affine consistent after round-trips."""

    def test_reorient_round_trip(self):
        for _ in range(repeats):
            with self.subTest():
                msk, _, _, _ = get_nii()
                ax1 = get_random_ax_code()
                ax2 = get_random_ax_code()
                msk2 = msk.reorient(ax1).reorient(ax2).reorient(msk.orientation)
                self.assertEqual(msk2.orientation, msk.orientation)
                self.assertEqual(msk2.shape, msk.shape)

    def test_rescale_round_trip_shape(self):
        arr = np.zeros((10, 12, 14), dtype=np.uint8)
        arr[3:7, 3:9, 3:11] = 1
        nii = _make_nii(arr, zoom=(1.0, 1.0, 1.0))
        nii2 = nii.rescale((2.0, 2.0, 2.0)).rescale((1.0, 1.0, 1.0))
        # shape should approximately return to original after double rescale
        for orig_s, new_s in zip(nii.shape, nii2.shape):
            self.assertAlmostEqual(orig_s, new_s, delta=2)


if __name__ == "__main__":
    unittest.main()
