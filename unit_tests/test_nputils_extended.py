"""Extended unit tests for np_utils: contacts, bounding boxes, surface, translate, normalize, euclid ops."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

import numpy as np  # noqa: E402

from TPTBox.core import np_utils  # noqa: E402


def _make_two_block_array() -> np.ndarray:
    """3-D array with two separate blocks of label 1 and one block of label 2."""
    arr = np.zeros((15, 15, 15), dtype=np.uint8)
    arr[1:5, 1:5, 1:5] = 1  # block A, label 1
    arr[9:13, 9:13, 9:13] = 1  # block B, label 1
    arr[1:5, 9:13, 1:5] = 2  # block C, label 2
    return arr


class Test_np_is_empty(unittest.TestCase):
    def test_empty_array(self):
        self.assertTrue(np_utils.np_is_empty(np.zeros((5, 5, 5), dtype=np.uint8)))

    def test_nonempty_array(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        self.assertFalse(np_utils.np_is_empty(arr))

    def test_all_ones(self):
        self.assertFalse(np_utils.np_is_empty(np.ones((3, 3), dtype=np.uint8)))


class Test_np_count_nonzero(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(np_utils.np_count_nonzero(np.zeros((4, 4), dtype=np.uint8)), 0)

    def test_some_nonzero(self):
        arr = np.zeros((5, 5), dtype=np.uint8)
        arr[0, 0] = 7
        arr[1, 2] = 3
        self.assertEqual(np_utils.np_count_nonzero(arr), 2)

    def test_all_nonzero(self):
        arr = np.ones((3, 3, 3), dtype=np.uint16)
        self.assertEqual(np_utils.np_count_nonzero(arr), 27)


class Test_np_unique(unittest.TestCase):
    def test_basic(self):
        arr = np.array([[[0, 1, 2], [2, 3, 0]]], dtype=np.uint8)
        result = np_utils.np_unique(arr)
        self.assertEqual(sorted(result), [0, 1, 2, 3])

    def test_all_same(self):
        arr = np.full((4, 4), 5, dtype=np.uint8)
        self.assertEqual(np_utils.np_unique(arr), [5])

    def test_unique_withoutzero(self):
        arr = np.array([0, 1, 2, 0, 3, 1], dtype=np.uint8)
        result = np_utils.np_unique_withoutzero(arr)
        self.assertEqual(sorted(result), [1, 2, 3])

    def test_unique_withoutzero_all_zeros(self):
        arr = np.zeros((3, 3), dtype=np.uint8)
        self.assertEqual(np_utils.np_unique_withoutzero(arr), [])


class Test_np_bounding_boxes(unittest.TestCase):
    def test_single_label_size(self):
        # Bounding boxes are in the coordinate frame of the internally cropped array,
        # so we only verify that the box spans the correct number of voxels per axis.
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:6, 3:7, 1:4] = 1  # size (4, 4, 3)
        boxes = np_utils.np_bounding_boxes(arr)
        self.assertIn(1, boxes)
        s0, s1, s2 = boxes[1]
        self.assertEqual(s0.stop - s0.start, 4)
        self.assertEqual(s1.stop - s1.start, 4)
        self.assertEqual(s2.stop - s2.start, 3)

    def test_multiple_labels(self):
        arr = _make_two_block_array()
        boxes = np_utils.np_bounding_boxes(arr)
        self.assertIn(1, boxes)
        self.assertIn(2, boxes)
        self.assertNotIn(0, boxes)

    def test_empty_array(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        boxes = np_utils.np_bounding_boxes(arr)
        self.assertEqual(boxes, {})


class Test_np_contacts(unittest.TestCase):
    def test_adjacent_labels_contact(self):
        arr = np.array([[[1, 1, 2, 2]]], dtype=np.uint8)
        contacts = np_utils.np_contacts(arr, connectivity=1)
        self.assertIn((1, 2), contacts)
        self.assertGreater(contacts[(1, 2)], 0)

    def test_non_adjacent_no_contact(self):
        arr = np.array([[[1, 0, 0, 2]]], dtype=np.uint8)
        contacts = np_utils.np_contacts(arr, connectivity=1)
        self.assertNotIn((1, 2), contacts)

    def test_contact_key_ordering(self):
        # np_contacts returns only one direction: (lower, higher) or (higher, lower)
        # — we just verify the pair exists in some order and has a positive count.
        arr = np.array([[[1, 2]]], dtype=np.uint8)
        contacts = np_utils.np_contacts(arr, connectivity=1)
        exists = (1, 2) in contacts or (2, 1) in contacts
        self.assertTrue(exists)
        count = contacts.get((1, 2), contacts.get((2, 1), 0))
        self.assertGreater(count, 0)


class Test_np_region_graph(unittest.TestCase):
    def test_adjacent_connected(self):
        arr = np.array([[[1, 1, 2, 2]]], dtype=np.uint8)
        graph = np_utils.np_region_graph(arr, connectivity=1)
        self.assertIn((1, 2), graph)

    def test_empty_graph(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0:2, 0:2, 0:2] = 1
        graph = np_utils.np_region_graph(arr, connectivity=1)
        # only one label → no edges between labels
        for a, b in graph:
            self.assertNotEqual(a, b)


class Test_np_translate_arr(unittest.TestCase):
    def test_shift_2d(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[3, 3] = 1
        shifted = np_utils.np_translate_arr(arr, (2, 2))
        self.assertEqual(shifted[5, 5], 1)
        self.assertEqual(shifted[3, 3], 0)

    def test_shift_3d(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2, 2, 2] = 5
        shifted = np_utils.np_translate_arr(arr, (1, 1, 1))
        self.assertEqual(shifted[3, 3, 3], 5)
        self.assertEqual(shifted[2, 2, 2], 0)

    def test_negative_shift(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[5, 5, 5] = 9
        shifted = np_utils.np_translate_arr(arr, (-2, 0, 0))
        self.assertEqual(shifted[3, 5, 5], 9)

    def test_shape_preserved(self):
        arr = np.ones((8, 9, 10), dtype=np.uint8)
        shifted = np_utils.np_translate_arr(arr, (1, 2, 3))
        self.assertEqual(shifted.shape, arr.shape)


class Test_np_compute_surface(unittest.TestCase):
    def test_surface_subset_of_mask(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:8, 2:8, 2:8] = 1
        surface = np_utils.np_compute_surface(arr)
        # surface voxels only where mask is
        self.assertTrue(np.all(surface[arr == 0] == 0))

    def test_surface_nonzero(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:8, 2:8, 2:8] = 1
        surface = np_utils.np_compute_surface(arr)
        self.assertGreater(np_utils.np_count_nonzero(surface), 0)

    def test_dilated_surface_larger(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[3:9, 3:9, 3:9] = 1
        surf_normal = np_utils.np_compute_surface(arr, dilated_surface=False)
        surf_dilated = np_utils.np_compute_surface(arr, dilated_surface=True)
        # dilated surface must cover at least as many voxels
        self.assertGreaterEqual(np_utils.np_count_nonzero(surf_dilated), np_utils.np_count_nonzero(surf_normal))


class Test_np_normalize_to_range(unittest.TestCase):
    def test_min_shifted_to_zero(self):
        # After normalization, the minimum value should be ≥ min_value (it is shifted exactly to min_value).
        arr = np.array([-100.0, 0.0, 500.0, 1000.0, 2000.0], dtype=np.float32)
        out = np_utils.np_normalize_to_range(arr, min_value=0, max_value=1500)
        self.assertAlmostEqual(float(out.min()), 0.0, places=3)

    def test_identity_in_range(self):
        # Array already in [0, max_value] should not be scaled down.
        arr = np.array([0.0, 500.0, 1000.0, 1500.0], dtype=np.float32)
        out = np_utils.np_normalize_to_range(arr, min_value=0, max_value=1500)
        np.testing.assert_allclose(out, arr, atol=1e-3)

    def test_scales_down_when_max_exceeded(self):
        # Old max 2000 > max_value 1000 → values should be scaled; check min stays 0.
        arr = np.array([0.0, 500.0, 1000.0, 2000.0], dtype=np.float32)
        out = np_utils.np_normalize_to_range(arr, min_value=0, max_value=1000)
        self.assertAlmostEqual(float(out.min()), 0.0, places=3)
        self.assertLessEqual(float(out.max()), 1000.0 + 1e-3)

    def test_shape_preserved(self):
        arr = np.random.rand(5, 6, 7).astype(np.float32) * 2000 - 500
        out = np_utils.np_normalize_to_range(arr)
        self.assertEqual(out.shape, arr.shape)


class Test_np_erode_dilate_euclid(unittest.TestCase):
    def test_erode_shrinks(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[4:16, 4:16, 4:16] = 1
        volume_before = np_utils.np_volume(arr)
        eroded = np_utils.np_erode_msk_euclid(arr, n_pixel=2)
        volume_after = np_utils.np_volume(eroded)
        self.assertLess(volume_after.get(1, 0), volume_before[1])

    def test_dilate_grows(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[7:13, 7:13, 7:13] = 1
        volume_before = np_utils.np_volume(arr)
        dilated = np_utils.np_dilate_msk_euclid(arr, n_pixel=2)
        volume_after = np_utils.np_volume(dilated)
        self.assertGreater(volume_after.get(1, 0), volume_before[1])

    def test_erode_then_dilate_subset(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[3:17, 3:17, 3:17] = 1
        eroded = np_utils.np_erode_msk_euclid(arr, n_pixel=2)
        dilated = np_utils.np_dilate_msk_euclid(eroded, n_pixel=2)
        # after erode+dilate, result should be subset of original
        self.assertTrue(np.all(dilated[arr == 0] == 0))

    def test_multilabel_euclid(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[2:9, 2:9, 2:9] = 1
        arr[11:18, 11:18, 11:18] = 2
        vol_before = np_utils.np_volume(arr)
        dilated = np_utils.np_dilate_msk_euclid(arr, n_pixel=1)
        vol_after = np_utils.np_volume(dilated)
        self.assertGreater(vol_after.get(1, 0), vol_before[1])
        self.assertGreater(vol_after.get(2, 0), vol_before[2])

    def test_zero_pixels_noop(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[3:7, 3:7, 3:7] = 1
        result = np_utils.np_erode_msk_euclid(arr, n_pixel=0)
        np.testing.assert_array_equal(result, arr)


class Test_np_calc_overlapping_labels(unittest.TestCase):
    def test_overlap_detected(self):
        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        a[2:8, 2:8, 2:8] = 1
        b[4:9, 4:9, 4:9] = 2
        result = np_utils.np_calc_overlapping_labels(a, b)
        self.assertIn((1, 2), result)

    def test_no_overlap(self):
        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        a[0:3, 0:3, 0:3] = 1
        b[7:10, 7:10, 7:10] = 2
        result = np_utils.np_calc_overlapping_labels(a, b)
        self.assertNotIn((1, 2), result)


class Test_np_bbox_binary(unittest.TestCase):
    """Tests for np_bbox_binary — tight bounding box of all non-zero voxels."""

    def test_basic_cube(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:6, 3:7, 1:5] = 1
        slices = np_utils.np_bbox_binary(arr)
        self.assertEqual(slices[0].start, 2)
        self.assertEqual(slices[0].stop, 6)
        self.assertEqual(slices[1].start, 3)
        self.assertEqual(slices[1].stop, 7)
        self.assertEqual(slices[2].start, 1)
        self.assertEqual(slices[2].stop, 5)

    def test_single_voxel(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[4, 5, 6] = 1
        slices = np_utils.np_bbox_binary(arr)
        self.assertEqual(slices[0].start, 4)
        self.assertEqual(slices[0].stop, 5)
        self.assertEqual(slices[1].start, 5)
        self.assertEqual(slices[1].stop, 6)

    def test_empty_raises(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        with self.assertRaises(ValueError):
            np_utils.np_bbox_binary(arr, raise_error=True)

    def test_with_padding_expands_box(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[4:6, 4:6, 4:6] = 1
        no_pad = np_utils.np_bbox_binary(arr, px_dist=0)
        with_pad = np_utils.np_bbox_binary(arr, px_dist=1)
        self.assertLessEqual(with_pad[0].start, no_pad[0].start)
        self.assertGreaterEqual(with_pad[0].stop, no_pad[0].stop)


class Test_np_point_coordinates(unittest.TestCase):
    """Tests for np_point_coordinates — non-zero voxel coordinates in 3D."""

    def test_single_point(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 3, 4] = 1
        coords = np_utils.np_point_coordinates(arr)
        self.assertEqual(len(coords), 1)
        self.assertEqual(coords[0], (2, 3, 4))

    def test_multiple_points(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[1, 1, 1] = 1
        arr[3, 3, 3] = 1
        coords = np_utils.np_point_coordinates(arr)
        self.assertEqual(len(coords), 2)
        self.assertIn((1, 1, 1), coords)
        self.assertIn((3, 3, 3), coords)

    def test_empty_array(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        coords = np_utils.np_point_coordinates(arr)
        self.assertEqual(len(coords), 0)

    def test_requires_3d(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            np_utils.np_point_coordinates(arr)


class Test_np_translate_to_center(unittest.TestCase):
    """Tests for np_translate_to_center_of_array — moves content toward array center."""

    def test_output_shape_preserved(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1, 1, 1] = 1
        out = np_utils.np_translate_to_center_of_array(arr)
        self.assertEqual(out.shape, arr.shape)

    def test_sum_preserved(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[1:4, 1:4, 1:4] = 1
        out = np_utils.np_translate_to_center_of_array(arr)
        self.assertEqual(int(out.sum()), int(arr.sum()))

    def test_content_moves_toward_center(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[0, 0, 0] = 1
        out = np_utils.np_translate_to_center_of_array(arr)
        xs, ys, zs = np.where(out)
        if len(xs) > 0:
            self.assertGreater(xs[0], 0)


class Test_np_calc_convex_hull(unittest.TestCase):
    """Tests for np_calc_convex_hull — fills the convex hull of non-zero voxels."""

    def test_2d_output_shape(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:8, 2:8] = 1
        hull = np_utils.np_calc_convex_hull(arr)
        self.assertEqual(hull.shape, arr.shape)

    def test_hull_contains_original(self):
        # need > 3 non-zero points so ConvexHull can construct a hull
        arr = np.zeros((12, 12), dtype=np.uint8)
        arr[1, 6] = 1
        arr[6, 1] = 1
        arr[6, 11] = 1
        arr[11, 6] = 1
        hull = np_utils.np_calc_convex_hull(arr)
        self.assertEqual(hull.shape, arr.shape)
        self.assertTrue(np.all(hull[arr > 0] > 0))

    def test_3d_output_same_shape(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        hull = np_utils.np_calc_convex_hull(arr)
        self.assertEqual(hull.shape, arr.shape)

    def test_hull_not_smaller_than_input(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:8, 2:8] = 1
        hull = np_utils.np_calc_convex_hull(arr)
        self.assertGreaterEqual(int(hull.sum()), int(arr.sum()))


class Test_np_betti_numbers(unittest.TestCase):
    """Tests for np_betti_numbers — topological descriptors B0, B1, B2."""

    def test_single_ball_b0_is_1(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:8, 2:8, 2:8] = 1
        b0, _b1, _b2 = np_utils.np_betti_numbers(arr)
        self.assertEqual(b0, 1)

    def test_two_components_b0_is_2(self):
        arr = np.zeros((15, 15, 15), dtype=np.uint8)
        arr[1:4, 1:4, 1:4] = 1
        arr[10:13, 10:13, 10:13] = 1
        b0, _b1, _b2 = np_utils.np_betti_numbers(arr)
        self.assertEqual(b0, 2)

    def test_empty_b0_is_0(self):
        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        b0, _b1, _b2 = np_utils.np_betti_numbers(arr)
        self.assertEqual(b0, 0)

    def test_returns_three_ints(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        result = np_utils.np_betti_numbers(arr)
        self.assertEqual(len(result), 3)
        for val in result:
            self.assertIsInstance(val, int)


class Test_np_majority_label_overlap(unittest.TestCase):
    """Tests for np_map_labels_based_on_majority_label_mask_overlap."""

    def test_simple_remap(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[3:7, 3:7, 3:7] = 1
        label_mask = np.zeros_like(arr)
        label_mask[2:8, 2:8, 2:8] = 5
        result = np_utils.np_map_labels_based_on_majority_label_mask_overlap(arr, label_mask)
        self.assertIn(5, np_utils.np_unique_withoutzero(result))
        self.assertNotIn(1, np_utils.np_unique(result))

    def test_no_overlap_maps_to_zero(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        label_mask = np.zeros_like(arr)
        result = np_utils.np_map_labels_based_on_majority_label_mask_overlap(arr, label_mask, no_match_label=0)
        self.assertNotIn(1, np_utils.np_unique(result))

    def test_inplace(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[3:7, 3:7, 3:7] = 2
        label_mask = np.zeros_like(arr)
        label_mask[2:8, 2:8, 2:8] = 9
        result = np_utils.np_map_labels_based_on_majority_label_mask_overlap(arr, label_mask, inplace=True)
        self.assertIs(result, arr)


class Test_np_find_index_of_k_max_values(unittest.TestCase):
    """Tests for np_find_index_of_k_max_values — indices of k largest values."""

    def test_top1(self):
        arr = np.array([3.0, 1.0, 9.0, 5.0])
        idx = np_utils.np_find_index_of_k_max_values(arr, k=1)
        self.assertEqual(idx[0], 2)

    def test_top2_order(self):
        arr = np.array([1.0, 7.0, 3.0, 9.0, 2.0])
        idx = np_utils.np_find_index_of_k_max_values(arr, k=2)
        self.assertEqual(idx[0], 3)
        self.assertEqual(idx[1], 1)

    def test_default_k_returns_two(self):
        arr = np.array([5.0, 1.0, 3.0])
        idx = np_utils.np_find_index_of_k_max_values(arr)
        self.assertEqual(len(idx), 2)


class Test_old_unique_variants(unittest.TestCase):
    """The cc3d-backed ``old_*`` variants of np_unique."""

    def test_old_np_unique(self):
        arr = np.array([0, 1, 2, 2, 3], dtype=np.uint8)
        self.assertEqual(sorted(np_utils.old_np_unique(arr)), [0, 1, 2, 3])

    def test_old_np_unique_withoutzero(self):
        arr = np.array([0, 1, 2, 2, 3], dtype=np.uint8)
        self.assertEqual(sorted(np_utils.old_np_unique_withoutzero(arr)), [1, 2, 3])

    def test_old_np_unique_non_uint_fallback(self):
        arr = np.array([0, 1, -2, 3], dtype=np.int16)
        self.assertEqual(sorted(np_utils.old_np_unique(arr)), [-2, 0, 1, 3])

    def test_old_np_unique_withoutzero_non_uint_fallback(self):
        # int16 is rejected by cc3dstatistics -> the np.unique fallback path is used.
        arr = np.array([0, 1, 2, 3], dtype=np.int16)
        self.assertEqual(sorted(np_utils.old_np_unique_withoutzero(arr)), [1, 2, 3])


class Test_np_voxel_connectivity_graph(unittest.TestCase):
    def test_2d(self):
        arr = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        out = np_utils.np_voxel_connectivity_graph(arr, connectivity=1)
        self.assertEqual(out.shape, arr.shape)

    def test_3d(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[1:4, 1:4, 1:4] = 1
        out = np_utils.np_voxel_connectivity_graph(arr, connectivity=2)
        self.assertEqual(out.shape, arr.shape)


class Test_np_dice(unittest.TestCase):
    def test_two_empty_returns_one(self):
        z = np.zeros((5, 5, 5), dtype=np.uint8)
        self.assertEqual(np_utils.np_dice(z, z, label=1), 1.0)

    def test_perfect_overlap(self):
        a = np.zeros((6, 6, 6), dtype=np.uint8)
        a[1:4, 1:4, 1:4] = 1
        self.assertAlmostEqual(np_utils.np_dice(a, a.copy(), label=1), 1.0)

    def test_binary_compare(self):
        a = np.zeros((6, 6, 6), dtype=np.uint8)
        b = np.zeros((6, 6, 6), dtype=np.uint8)
        a[1:4, 1:4, 1:4] = 1
        b[1:4, 1:4, 1:4] = 7
        self.assertAlmostEqual(np_utils.np_dice(a, b, binary_compare=True), 1.0)


class Test_euclid_morphology_branches(unittest.TestCase):
    """Exercise label/mask/use_crop branches of np_erode/dilate_msk_euclid."""

    @staticmethod
    def _two_label_arr():
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[3:9, 3:9, 3:9] = 1
        arr[11:17, 11:17, 11:17] = 2
        return arr

    def test_erode_euclid_labels_and_mask_crop(self):
        arr = self._two_label_arr()
        mask = np.ones_like(arr)
        out = np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=1, use_crop=True, labels=[1], mask=mask)
        self.assertIsInstance(out, np.ndarray)

    def test_erode_euclid_no_crop_with_labels_and_mask(self):
        arr = self._two_label_arr()
        mask = np.ones_like(arr)
        out = np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=1, use_crop=False, labels=[1], mask=mask)
        self.assertIsInstance(out, np.ndarray)

    def test_dilate_euclid_labels_and_mask_crop(self):
        arr = self._two_label_arr()
        mask = np.ones_like(arr)
        out = np_utils.np_dilate_msk_euclid(arr.copy(), n_pixel=1, use_crop=True, labels=[1], mask=mask)
        self.assertIsInstance(out, np.ndarray)

    def test_dilate_euclid_no_crop_with_mask(self):
        arr = self._two_label_arr()
        mask = np.ones_like(arr)
        out = np_utils.np_dilate_msk_euclid(arr.copy(), n_pixel=1, use_crop=False, mask=mask)
        self.assertIsInstance(out, np.ndarray)


class Test_dilate_erode_msk_branches(unittest.TestCase):
    """Exercise use_crop/mask/ignore_axis branches of np_dilate_msk and np_erode_msk."""

    def test_dilate_msk_crop_with_mask(self):
        # label nearly fills the array so crop and per-label crop both span the whole array,
        # which keeps the in-loop mask application shape-consistent.
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1:9, 1:9, 1:9] = 1
        mask = np.ones_like(arr)
        out = np_utils.np_dilate_msk(arr.copy(), n_pixel=1, use_crop=True, mask=mask)
        self.assertIsInstance(out, np.ndarray)

    def test_dilate_msk_no_crop_ignore_axis_and_mask(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[3:9, 3:9, 3:9] = 1
        mask = np.ones_like(arr)
        out = np_utils.np_dilate_msk(arr.copy(), label_ref=1, n_pixel=1, use_crop=False, mask=mask, ignore_axis=0)
        self.assertEqual(out.shape, arr.shape)

    def test_erode_msk_no_crop_ignore_axis_with_zero_label(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[3:9, 3:9, 3:9] = 1
        # label_ref includes 0 -> the i == 0 "continue" branch is taken
        out = np_utils.np_erode_msk(arr.copy(), label_ref=[0, 1], n_pixel=1, use_crop=False, ignore_axis=0)
        self.assertEqual(out.shape, arr.shape)


class Test_np_map_labels_empty(unittest.TestCase):
    def test_empty_map_returns_input(self):
        arr = np.array([1, 2, 3], dtype=np.uint8)
        out = np_utils.np_map_labels(arr, {})
        np.testing.assert_array_equal(out, arr)


class Test_connected_components_include_zero(unittest.TestCase):
    def test_cc_include_zero(self):
        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        cc, n = np_utils.np_connected_components(arr.copy(), include_zero=True)
        self.assertEqual(cc.shape, arr.shape)
        self.assertGreaterEqual(n, 1)

    def test_cc_per_label_include_zero(self):
        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        out = np_utils.np_connected_components_per_label(arr.copy(), include_zero=True)
        self.assertIn(0, out)
        self.assertIn(1, out)


class Test_filter_connected_components_branches(unittest.TestCase):
    @staticmethod
    def _multi():
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[1:4, 1:4, 1:4] = 1
        arr[1:4, 16:19, 1:4] = 1
        arr[16:19, 1:4, 1:4] = 2
        arr[16:19, 16:19, 16:19] = 2
        return arr

    def test_k_none_3d(self):
        out = np_utils.np_filter_connected_components(self._multi(), largest_k_components=None)
        self.assertGreater(np_utils.np_count_nonzero(out), 0)

    def test_2d_connectivity(self):
        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[1:4, 1:4] = 1
        arr[10:14, 10:14] = 1
        out = np_utils.np_filter_connected_components(arr, largest_k_components=1, connectivity=2)
        self.assertGreater(np_utils.np_count_nonzero(out), 0)

    def test_per_label_k_with_removed_label(self):
        out = np_utils.np_filter_connected_components(self._multi(), largest_k_components=1, removed_to_label=9)
        self.assertIn(9, np_utils.np_unique(out))

    def test_relabeled_output_with_removed_label(self):
        out = np_utils.np_filter_connected_components(
            self._multi(), largest_k_components=1, return_original_labels=False, removed_to_label=9
        )
        self.assertGreater(np_utils.np_count_nonzero(out), 0)


class Test_cc_center_of_mass_sorted(unittest.TestCase):
    def test_sort_by_axis(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[1:4, 1:4, 1:4] = 1
        arr[15:18, 15:18, 15:18] = 1
        coms = np_utils.np_get_connected_components_center_of_mass(arr, label=1, sort_by_axis=0)
        self.assertEqual(len(coms), 2)
        self.assertLessEqual(coms[0][0], coms[1][0])


class Test_fill_holes_pbar(unittest.TestCase):
    def test_fill_holes_with_pbar(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[2:10, 2:10, 2:10] = 1
        arr[5, 5, 5] = 0  # interior hole
        out = np_utils.np_fill_holes(arr.copy(), pbar=True)
        self.assertEqual(out[5, 5, 5], 1)


class Test_smooth_gaussian_labelwise_options(unittest.TestCase):
    def test_all_option_branches(self):
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[3:8, 3:8, 3:8] = 1
        arr[9:14, 9:14, 9:14] = 2
        out = np_utils.np_smooth_gaussian_labelwise(
            arr.copy(),
            label_to_smooth=[1, 2],
            label_weights={1: 1.5, 0: 0.5},
            dilate_prior=1,
            dilate_channelwise=True,
            background_threshold=0.1,
        )
        self.assertEqual(out.shape, arr.shape)


class Test_convex_hull_axiswise(unittest.TestCase):
    def test_axis_slicewise(self):
        arr = np.zeros((5, 14, 14), dtype=np.uint8)
        # slice 0 empty -> continue
        arr[1, 3, 3] = 1
        arr[1, 5, 5] = 1  # only 2 points -> _convex_hull returns zeros
        arr[2, 6, 2:10] = 1  # collinear -> ConvexHull raises -> except branch
        arr[3, 3:11, 3:11] = 1  # real hull
        arr[4, 3:11, 3:11] = 1
        hull = np_utils.np_calc_convex_hull(arr, axis=0)
        self.assertEqual(hull.shape, arr.shape)

    def test_select_axis_dynamically(self):
        slices = np_utils._select_axis_dynamically(axis=1, index=2, n_dims=3)
        self.assertEqual(slices, (slice(None), 2, slice(None)))


class Test_calc_boundary_mask(unittest.TestCase):
    def test_boundary_mask_basic(self):
        img = np.zeros((10, 10, 10), dtype=np.float32)
        img[3:7, 3:7, 3:7] = 100
        with contextlib.redirect_stdout(io.StringIO()):
            out = np_utils.np_calc_boundary_mask(img, threshold=50)
        self.assertEqual(out.shape, img.shape)

    def test_boundary_mask_ct(self):
        img = np.full((8, 8, 8), -1000, dtype=np.float32)
        img[2:6, 2:6, 2:6] = 200
        with contextlib.redirect_stdout(io.StringIO()):
            out = np_utils.np_calc_boundary_mask(img, threshold=0, adjust_intensity_for_ct=True)
        self.assertEqual(out.shape, img.shape)


class Test_betti_verbose(unittest.TestCase):
    def test_verbose(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            b0, _b1, _b2 = np_utils.np_betti_numbers(arr, verbose=True)
        self.assertEqual(b0, 1)


class Test_pad_to_parameters(unittest.TestCase):
    def test_mixed_pad_and_crop(self):
        padding, crop, requires_crop = np_utils._pad_to_parameters((10, 11, 8), (12, 10, 10))
        self.assertEqual(len(padding), 3)
        self.assertEqual(len(crop), 3)
        self.assertTrue(requires_crop)

    def test_pure_crop(self):
        padding, crop, requires_crop = np_utils._pad_to_parameters((12, 12, 12), (8, 8, 8))
        self.assertTrue(requires_crop)
        for c in crop:
            self.assertIsInstance(c, slice)


class Test_generate_binary_structure(unittest.TestCase):
    def test_3d(self):
        s = np_utils._generate_binary_structure(3, 1)
        self.assertEqual(s.shape, (3, 3, 3))

    def test_zero_dim(self):
        s = np_utils._generate_binary_structure(0, 1)
        self.assertTrue(bool(s))

    def test_larger_kernel(self):
        s = np_utils._generate_binary_structure(2, 2, kernel_size=5)
        self.assertEqual(s.shape, (5, 5))


class Test_fast_binary_morphology(unittest.TestCase):
    def test_dilation_1d_default_selem(self):
        img = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
        out = np_utils._binary_dilation(img)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(bool(out.any()))

    def test_dilation_2d_default_selem(self):
        img = np.zeros((5, 5), dtype=bool)
        img[2, 2] = True
        out = np_utils._binary_dilation(img)
        self.assertTrue(out[2, 1])

    def test_dilation_3d_default_selem(self):
        img = np.zeros((5, 5, 5), dtype=bool)
        img[2, 2, 2] = True
        out = np_utils._binary_dilation(img)
        self.assertTrue(out[1, 2, 2])

    def test_dilation_list_selem(self):
        img = np.zeros((5, 5), dtype=bool)
        img[2, 2] = True
        out = np_utils._binary_dilation(img, struct=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(out.shape, img.shape)

    def test_dilation_nonbool_selem(self):
        img = np.zeros((5, 5), dtype=bool)
        img[2, 2] = True
        out = np_utils._binary_dilation(img, struct=np.ones((3, 3), dtype=np.uint8))
        self.assertEqual(out.shape, img.shape)

    def test_dilation_even_selem_raises(self):
        img = np.zeros((5, 5), dtype=bool)
        img[2, 2] = True
        with self.assertRaises(ValueError):
            np_utils._binary_dilation(img, struct=np.ones((2, 2)))

    def test_binary_erosion(self):
        img = np.zeros((6, 6), dtype=bool)
        img[1:5, 1:5] = True
        out = np_utils._binary_erosion(img)
        self.assertEqual(out.shape, img.shape)

    def test_binary_closing_and_opening(self):
        img = np.zeros((6, 6), dtype=bool)
        img[1:5, 1:5] = True
        closed = np_utils._binary_closing(img)
        opened = np_utils._binary_opening(img)
        self.assertEqual(closed.shape, img.shape)
        self.assertEqual(opened.shape, img.shape)

    def test_unpad_int(self):
        arr = np.pad(np.ones((3, 3), dtype=bool), 1)
        out = np_utils._unpad(arr, 1)
        self.assertEqual(out.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
