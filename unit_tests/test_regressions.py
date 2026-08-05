"""Regression tests for bugs found in the repository audit.

Each test pins down behaviour that was previously broken; the comment on the class
names the symptom that used to occur.
"""

from __future__ import annotations

import unittest

import numpy as np

from TPTBox import NII
from TPTBox.core.np_utils import np_dilate_msk_euclid


def _make_nii(arr: np.ndarray, seg: bool = True, zoom=(1.0, 1.0, 1.0)) -> NII:
    affine = np.diag([*zoom, 1.0])
    return NII.from_numpy(arr, affine=affine, seg=seg)


def _cube(shape=(10, 10, 10)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[3:7, 3:7, 3:7] = 1
    return arr


class Test_Rescale_Scalar(unittest.TestCase):
    """`rescale(1.5)` used to build a generator and raise TypeError on len()."""

    def test_scalar_voxel_spacing(self):
        nii = _make_nii(_cube())
        out = nii.rescale(1.5)
        self.assertEqual(len(out.shape), 3)

    def test_scalar_matches_tuple(self):
        nii = _make_nii(_cube())
        self.assertEqual(nii.rescale(1.5).shape, nii.rescale((1.5, 1.5, 1.5)).shape)


class Test_Threshold(unittest.TestCase):
    """`threshold` used to zero out voxels exactly equal to the threshold."""

    def test_value_equal_to_threshold_is_kept(self):
        nii = _make_nii(_cube())
        self.assertEqual(list(nii.threshold(1).unique()), [1])

    def test_below_threshold_is_dropped(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[0, 0, 0] = 1
        arr[1, 1, 1] = 5
        out = _make_nii(arr).threshold(5).get_seg_array()
        self.assertEqual(out[1, 1, 1], 1)
        self.assertEqual(out[0, 0, 0], 0)


class Test_Normalize(unittest.TestCase):
    """`normalize` divided by max_out instead of scaling into [min_out, max_out]."""

    def test_range_0_255(self):
        nii = _make_nii(_cube(), seg=False)
        out = nii.normalize(0, 255)
        self.assertAlmostEqual(float(out.min()), 0.0, places=4)
        self.assertAlmostEqual(float(out.max()), 255.0, places=3)

    def test_range_1_2(self):
        nii = _make_nii(_cube(), seg=False)
        out = nii.normalize(1, 2)
        self.assertAlmostEqual(float(out.min()), 1.0, places=4)
        self.assertAlmostEqual(float(out.max()), 2.0, places=4)

    def test_constant_image_does_not_raise(self):
        arr = np.full((5, 5, 5), 3.0, dtype=np.float32)
        _make_nii(arr, seg=False).normalize(0, 1)


class Test_Dilate_Msk_Mask(unittest.TestCase):
    """`dilate_msk(mask=...)` raised IndexError for multi-label segmentations."""

    @staticmethod
    def _two_labels() -> NII:
        arr = np.zeros((30, 30, 30), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1
        arr[22:26, 22:26, 22:26] = 2
        return _make_nii(arr)

    def test_multilabel_with_mask_does_not_raise(self):
        nii = self._two_labels()
        full = _make_nii(np.ones((30, 30, 30), dtype=np.uint8))
        self.assertEqual(list(nii.dilate_msk(n_pixel=2, mask=full, verbose=False).unique()), [1, 2])

    def test_mask_restricts_output(self):
        nii = self._two_labels()
        half = np.zeros((30, 30, 30), dtype=np.uint8)
        half[:15] = 1
        out = nii.dilate_msk(n_pixel=2, mask=_make_nii(half), verbose=False).get_seg_array()
        self.assertEqual(int((out[15:] != 0).sum()), 0)

    def test_caller_mask_is_not_mutated(self):
        nii = self._two_labels()
        mask_arr = np.zeros((30, 30, 30), dtype=np.uint8)
        mask_arr[:15] = 7
        before = mask_arr.copy()
        nii.dilate_msk(n_pixel=2, mask=_make_nii(mask_arr), verbose=False)
        np.testing.assert_array_equal(mask_arr, before)


class Test_Dilate_Euclid_Labels(unittest.TestCase):
    """`np_dilate_msk_euclid(labels=...)` ignored `labels` when use_crop=True."""

    @staticmethod
    def _adjacent() -> np.ndarray:
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[5, 5, 5] = 1
        arr[5, 5, 7] = 2
        return arr

    def test_crop_and_nocrop_agree(self):
        arr = self._adjacent()
        with_crop = np_dilate_msk_euclid(arr.copy(), n_pixel=2, labels=[1], use_crop=True)
        without_crop = np_dilate_msk_euclid(arr.copy(), n_pixel=2, labels=[1], use_crop=False)
        np.testing.assert_array_equal(with_crop, without_crop)

    def test_unselected_label_is_not_dilated(self):
        arr = self._adjacent()
        out = np_dilate_msk_euclid(arr.copy(), n_pixel=2, labels=[1], use_crop=True)
        self.assertEqual(int((out == 2).sum()), 0)


class Test_Inplace_Contract(unittest.TestCase):
    """Early-return branches had `inplace` inverted, leaking/copying the wrong object."""

    def test_rescale_inplace_returns_self(self):
        nii = _make_nii(_cube())
        self.assertIs(nii.rescale_((1, 1, 1)), nii)

    def test_rescale_out_of_place_returns_copy(self):
        nii = _make_nii(_cube())
        self.assertIsNot(nii.rescale((1, 1, 1)), nii)

    def test_reorient_out_of_place_returns_copy(self):
        nii = _make_nii(_cube())
        self.assertIsNot(nii.reorient(nii.orientation), nii)

    def test_extract_label_none_keep_label_returns_copy(self):
        nii = _make_nii(_cube())
        self.assertIsNot(nii.extract_label(None, keep_label=True), nii)


class Test_Float_Seg_Downcast(unittest.TestCase):
    """`isinstance(dtype, np.floating)` is never True, so float segs stayed float (8x memory)."""

    def test_float64_seg_is_downcast_on_construction(self):
        arr = np.zeros((8, 8, 8), dtype=np.float64)
        arr[2:5, 2:5, 2:5] = 1
        self.assertEqual(_make_nii(arr).dtype, np.uint8)

    def test_float_seg_keeps_its_labels(self):
        arr = np.zeros((8, 8, 8), dtype=np.float32)
        arr[2:5, 2:5, 2:5] = 7
        self.assertEqual(list(_make_nii(arr).unique()), [7])

    def test_non_seg_float_is_left_alone(self):
        arr = np.zeros((8, 8, 8), dtype=np.float32)
        self.assertEqual(_make_nii(arr, seg=False).dtype, np.float32)


class Test_Map_Labels_Dtype(unittest.TestCase):
    """np_map_labels built its lookup table in the input dtype, wrapping out-of-range targets."""

    @staticmethod
    def _arr() -> np.ndarray:
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1
        return arr

    def test_target_above_input_dtype_range(self):
        from TPTBox.core.np_utils import np_map_labels

        self.assertIn(300, np_map_labels(self._arr(), {1: 300}))

    def test_negative_target(self):
        from TPTBox.core.np_utils import np_map_labels

        self.assertIn(-5, np_map_labels(self._arr(), {1: -5}))

    def test_in_range_target_keeps_dtype(self):
        from TPTBox.core.np_utils import np_map_labels

        self.assertEqual(np_map_labels(self._arr(), {1: 2}).dtype, np.uint8)


class Test_Smallest_Int_Dtype(unittest.TestCase):
    """set_dtype('smallest_int') picked from max() only, wrapping negative values."""

    def test_negative_range_uses_wide_enough_type(self):
        arr = np.full((4, 4, 4), -200, dtype=np.int32)
        arr[0, 0, 0] = 100
        nii = _make_nii(arr, seg=False)
        self.assertEqual(nii.set_dtype("smallest_int").dtype, np.int16)

    def test_values_survive_the_cast(self):
        arr = np.full((4, 4, 4), -200, dtype=np.int32)
        arr[0, 0, 0] = 100
        out = _make_nii(arr, seg=False).set_dtype("smallest_int").get_array()
        self.assertEqual(sorted(set(out.ravel().tolist())), [-200, 100])

    def test_smallest_uint_rejects_negatives(self):
        arr = np.full((4, 4, 4), -1, dtype=np.int32)
        with self.assertRaises(AssertionError):
            _make_nii(arr, seg=False).set_dtype("smallest_uint")


if __name__ == "__main__":
    unittest.main()
