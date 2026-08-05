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


class Test_Bbox_Binary(unittest.TestCase):
    """np_bbox_binary clamped before adding 1 and stored px_dist as uint8."""

    @staticmethod
    def _touching_border() -> np.ndarray:
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[:, 5, 5] = 1
        return arr

    def test_stop_never_exceeds_shape(self):
        from TPTBox.core.np_utils import np_bbox_binary

        for sl, dim in zip(np_bbox_binary(self._touching_border(), px_dist=2), (20, 20, 20)):
            self.assertLessEqual(sl.stop, dim)

    def test_large_px_dist_does_not_overflow(self):
        from TPTBox.core.np_utils import np_bbox_binary

        self.assertEqual(np_bbox_binary(self._touching_border(), px_dist=300)[0], slice(0, 20))

    def test_interior_bbox_is_unchanged(self):
        from TPTBox.core.np_utils import np_bbox_binary

        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[5:8, 5:8, 5:8] = 1
        self.assertEqual(np_bbox_binary(arr, px_dist=0)[0], slice(5, 8))


class Test_Unique_Return_Types(unittest.TestCase):
    """np_unique returned numpy scalars on the fallback paths, breaking json.dumps."""

    def test_all_dtypes_return_native_scalars(self):
        import json

        from TPTBox.core.np_utils import np_unique, np_unique_withoutzero

        for dtype in (np.uint8, np.int16, np.float32):
            arr = np.zeros((8, 8, 8), dtype=dtype)
            arr[0, 0, 0] = 3
            for values in (np_unique(arr), np_unique_withoutzero(arr)):
                json.dumps(values)  # would raise for numpy scalars
                for value in values:
                    self.assertIn(type(value), (int, float), f"{dtype} produced {type(value)}")

    def test_values_are_correct(self):
        from TPTBox.core.np_utils import np_unique, np_unique_withoutzero

        arr = np.zeros((8, 8, 8), dtype=np.int16)
        arr[0, 0, 0] = 3
        self.assertEqual(np_unique(arr), [0, 3])
        self.assertEqual(np_unique_withoutzero(arr), [3])


class Test_Segmentation_In_Border(unittest.TestCase):
    """An empty mask was reported as touching the border."""

    def test_empty_is_not_in_border(self):
        self.assertFalse(_make_nii(np.zeros((10, 10, 10), dtype=np.uint8)).is_segmentation_in_border())

    def test_touching_border_is_detected(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[0, 5, 5] = 1
        self.assertTrue(_make_nii(arr).is_segmentation_in_border())

    def test_centred_is_not_in_border(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[9:11, 9:11, 9:11] = 1
        self.assertFalse(_make_nii(arr).is_segmentation_in_border())


class Test_Global_Vert_Order(unittest.TestCase):
    """sag_cor_curve_projection aliased the module-level v_idx_order list and extended it in place."""

    def test_v_idx_order_matches_its_definition(self):
        from TPTBox.core.vert_constants import v_idx2name, v_idx_order

        self.assertEqual(list(v_idx_order), list(v_idx2name.keys()))

    def test_snapshot_module_does_not_extend_it(self):
        import TPTBox
        from TPTBox.spine.snapshot2D import snapshot_modular  # noqa: F401

        self.assertEqual(len(TPTBox.v_idx_order), len(TPTBox.v_idx2name))


class Test_POI_PIR_Cache(unittest.TestCase):
    """_vert_orientation_pir was a class attribute, so it was shared by every POI in the process."""

    @staticmethod
    def _poi(value: float):
        from TPTBox import POI

        return POI({1: {50: (value, value, value)}}, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=(10, 10, 10))

    def test_cache_is_not_shared_between_instances(self):
        first = self._poi(1.0)
        second = self._poi(4.0)
        first._vert_orientation_pir[99] = "subject-1"
        self.assertNotIn(99, second._vert_orientation_pir)

    def test_own_cache_is_retained(self):
        poi = self._poi(1.0)
        poi._vert_orientation_pir[99] = "subject-1"
        self.assertEqual(poi._vert_orientation_pir[99], "subject-1")

    def test_copy_starts_with_an_empty_cache(self):
        poi = self._poi(1.0)
        poi._vert_orientation_pir[99] = "subject-1"
        self.assertNotIn(99, poi.copy()._vert_orientation_pir)


class Test_Public_API(unittest.TestCase):
    """__all__ advertised load_poi, but nothing imported it, so `import *` raised."""

    def test_load_poi_is_exported(self):
        import TPTBox

        self.assertTrue(callable(TPTBox.load_poi))

    def test_everything_in_all_is_importable(self):
        import TPTBox

        missing = [name for name in TPTBox.__all__ if not hasattr(TPTBox, name)]
        self.assertEqual(missing, [])


class Test_Getitem_Slice(unittest.TestCase):
    """nii[0:5] delegated to a key containing Ellipsis, which the same method rejects."""

    def test_single_slice(self):
        nii = _make_nii(_cube())
        self.assertEqual(nii[0:5].shape, (5, 10, 10))

    def test_full_slice_tuple_still_works(self):
        nii = _make_nii(_cube())
        self.assertEqual(nii[2:5, 2:5, 2:5].shape, (3, 3, 3))


class Test_Label_As_String(unittest.TestCase):
    """str is a Sequence, so the str branch was unreachable and '12' iterated as characters."""

    @staticmethod
    def _seg() -> NII:
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1, 1, 1] = 12
        arr[2:5, 2:5, 2:5] = 3
        return _make_nii(arr)

    def test_extract_label_string_matches_int(self):
        seg = self._seg()
        self.assertEqual(int(seg.extract_label("12").sum()), int(seg.extract_label(12).sum()))

    def test_remove_labels_string(self):
        self.assertEqual(list(self._seg().remove_labels("12", verbose=False).unique()), [3])


class Test_Metrics_On_Integer_Images(unittest.TestCase):
    """ssim/psnr used in-place `/=`, which fails on integer arrays."""

    @staticmethod
    def _int_nii() -> NII:
        arr = np.zeros((10, 10, 10), dtype=np.int16)
        arr[2:5, 2:5, 2:5] = 300
        return _make_nii(arr, seg=False)

    def test_ssim_identical_is_one(self):
        nii = self._int_nii()
        self.assertAlmostEqual(float(nii.ssim(nii)), 1.0, places=5)

    def test_psnr_runs_on_int16(self):
        nii = self._int_nii()
        self.assertTrue(np.isinf(nii.psnr(nii)))


class Test_Calc_Centroids_Level_Info(unittest.TestCase):
    """type() was taken after unwrapping the enum to .value, so it was always int."""

    def test_level_two_info_records_the_enum_class(self):
        from TPTBox import Location, calc_centroids

        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1
        arr[6:9, 6:9, 6:9] = 2
        poi = calc_centroids(_make_nii(arr), second_stage=Location.Vertebra_Corpus)
        self.assertIs(poi.level_two_info, Location)


if __name__ == "__main__":
    unittest.main()
