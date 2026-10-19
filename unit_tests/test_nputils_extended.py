"""Extended unit tests for np_utils: contacts, bounding boxes, surface, translate, normalize, euclid ops."""

from __future__ import annotations

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


class TestNpIsin(unittest.TestCase):
    """`np_isin` is a lookup-table fast path for `np.isin` on label arrays."""

    def test_matches_numpy_isin(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 12, size=(8, 9, 10)).astype(np.uint8)
        for labels in ([3], [1, 5, 7], [], [0], list(range(12))):
            with self.subTest(labels=labels):
                np.testing.assert_array_equal(np_utils.np_isin(arr, labels), np.isin(arr, labels))

    def test_invert_matches_numpy(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 8, size=(6, 6, 6)).astype(np.uint8)
        labels = [2, 4]
        np.testing.assert_array_equal(np_utils.np_isin(arr, labels, invert=True), np.isin(arr, labels, invert=True))

    def test_labels_outside_the_array_range(self):
        arr = np.array([[[0, 1, 2]]], dtype=np.uint8)
        np.testing.assert_array_equal(np_utils.np_isin(arr, [99]), np.zeros_like(arr, dtype=bool))
        np.testing.assert_array_equal(np_utils.np_isin(arr, [99], invert=True), np.ones_like(arr, dtype=bool))

    def test_result_is_boolean_and_shape_preserving(self):
        arr = np.zeros((3, 4, 5), dtype=np.uint16)
        out = np_utils.np_isin(arr, [0])
        self.assertEqual(out.dtype, np.dtype(bool))
        self.assertEqual(out.shape, arr.shape)


class TestNpCalcBoundaryMask(unittest.TestCase):
    """Floods the sub-threshold region inward from the image border."""

    def test_an_all_background_image_marks_nothing(self):
        out = np_utils.np_calc_boundary_mask(np.zeros((8, 8, 8), dtype=np.int16), threshold=0)
        self.assertEqual(out.shape, (8, 8, 8))
        self.assertEqual(int((out != 0).sum()), 0)

    def test_an_enclosed_bright_region_is_marked(self):
        img = np.full((10, 10, 10), -1000, dtype=np.int16)
        img[3:7, 3:7, 3:7] = 300
        out = np_utils.np_calc_boundary_mask(img, threshold=0)
        self.assertEqual(int((out != 0).sum()), 4 * 4 * 4, "the above-threshold block is what remains marked")

    def test_ct_mode_shifts_the_intensities_by_1000(self):
        # Soft tissue at -500 HU is below a 0 threshold on the raw values but above it
        # once the CT offset is applied, so the two modes must disagree here.
        img = np.full((10, 10, 10), -1000, dtype=np.int16)
        img[3:7, 3:7, 3:7] = -500
        plain = np_utils.np_calc_boundary_mask(img, threshold=0)
        ct = np_utils.np_calc_boundary_mask(img, threshold=0, adjust_intensity_for_ct=True)
        self.assertEqual(int((plain != 0).sum()), 0)
        self.assertEqual(int((ct != 0).sum()), 4 * 4 * 4)

    def test_threshold_selects_what_counts_as_foreground(self):
        # Regression: the function wrote its sentinel value 2 into the array and then
        # compared that array against the threshold, so every threshold >= 2 erased the
        # foreground and the mask came back empty. NII.boundary_mask(threshold) forwards
        # straight here, so any realistic HU threshold was affected.
        img = np.full((10, 10, 10), -1000, dtype=np.int16)
        img[3:7, 3:7, 3:7] = 300
        for threshold in (0, 2, 100, 299):
            with self.subTest(threshold=threshold):
                self.assertEqual(int((np_utils.np_calc_boundary_mask(img, threshold=threshold) != 0).sum()), 4 * 4 * 4)
        self.assertEqual(int((np_utils.np_calc_boundary_mask(img, threshold=500) != 0).sum()), 0, "nothing is above 500")


class TestNpVoxelConnectivityGraph(unittest.TestCase):
    """Per-voxel bit field of which neighbours carry the same label."""

    def test_a_lone_voxel_has_no_same_label_neighbour(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        graph = np_utils.np_voxel_connectivity_graph(arr, connectivity=1)
        self.assertEqual(graph.shape, arr.shape)
        self.assertEqual(int(graph[2, 2, 2]), 0)
        # Background is a label too, so background voxels *are* connected to each other.
        self.assertGreater(int(graph[0, 0, 0]), 0)

    def test_two_touching_voxels_are_connected(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        arr[2, 2, 3] = 1
        graph = np_utils.np_voxel_connectivity_graph(arr, connectivity=1)
        self.assertGreater(int(graph[2, 2, 2]), 0)
        self.assertGreater(int(graph[2, 2, 3]), 0)

    def test_a_uniform_array_is_fully_connected(self):
        arr = np.ones((6, 6, 6), dtype=np.uint8)
        graph = np_utils.np_voxel_connectivity_graph(arr, connectivity=1)
        self.assertEqual(int(graph[3, 3, 3]), 63, "all six 6-neighbourhood bits set in the interior")

    def test_higher_connectivity_reports_at_least_as_many_links(self):
        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1
        low = np_utils.np_voxel_connectivity_graph(arr, connectivity=1)
        high = np_utils.np_voxel_connectivity_graph(arr, connectivity=3)
        self.assertGreaterEqual(int(np.sum(high != 0)), int(np.sum(low != 0)))


class TestNpFilterConnectedComponentsByBboxChain(unittest.TestCase):
    @staticmethod
    def _chain():
        # Three blobs stacked along axis 0 with a far-away outlier.
        arr = np.zeros((40, 12, 12), dtype=np.uint8)
        for z in (2, 8, 14):
            arr[z : z + 3, 4:8, 4:8] = 1
        arr[34:37, 1:3, 1:3] = 1  # outlier, off the chain
        return arr

    def test_the_outlier_is_dropped(self):
        out = np_utils.np_filter_connected_components_by_bbox_chain(self._chain())
        self.assertEqual(int(out[35, 2, 2]), 0, "the off-chain blob must be removed")
        self.assertGreater(int((out != 0).sum()), 0)

    def test_a_single_component_survives(self):
        arr = np.zeros((20, 10, 10), dtype=np.uint8)
        arr[5:8, 4:7, 4:7] = 1
        out = np_utils.np_filter_connected_components_by_bbox_chain(arr)
        self.assertEqual(int((out != 0).sum()), int((arr != 0).sum()))

    def test_a_generous_margin_keeps_more(self):
        tight = int((np_utils.np_filter_connected_components_by_bbox_chain(self._chain()) != 0).sum())
        loose = int(
            (
                np_utils.np_filter_connected_components_by_bbox_chain(self._chain(), margin=30.0, extra_margin=30.0, extra_margin_axis=0)
                != 0
            ).sum()
        )
        self.assertGreaterEqual(loose, tight)

    def test_extra_margin_requires_an_axis(self):
        with self.assertRaises(AssertionError):
            np_utils.np_filter_connected_components_by_bbox_chain(self._chain(), extra_margin=5.0)


class TestNpCalcConvexHull(unittest.TestCase):
    def test_hull_fills_a_concave_notch(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[5:15, 5:15, 5:15] = 1
        arr[5:10, 5:10, 5:15] = 0  # bite a corner out
        hull = np_utils.np_calc_convex_hull(arr, axis=None)
        self.assertGreater(int((hull != 0).sum()), int((arr != 0).sum()))

    def test_a_convex_input_is_unchanged_in_volume(self):
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[4:12, 4:12, 4:12] = 1
        hull = np_utils.np_calc_convex_hull(arr, axis=None)
        self.assertAlmostEqual(int((hull != 0).sum()), int((arr != 0).sum()), delta=int(0.05 * (arr != 0).sum()))

    def test_per_slice_hull_along_an_axis(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[4:8, 4:8, 4:8] = 1
        arr[5, 5, 5] = 0  # interior hole
        hull = np_utils.np_calc_convex_hull(arr, axis=0)
        self.assertEqual(int(hull[5, 5, 5]), 1, "the hole is filled inside its slice")


class TestNpErodeMskEuclid(unittest.TestCase):
    def test_erosion_removes_a_shell_of_the_requested_thickness(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[5:15, 5:15, 5:15] = 1
        out = np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=2)
        self.assertEqual(int((out != 0).sum()), 6 * 6 * 6, "a 10-voxel cube eroded by 2 leaves a 6-voxel cube")
        self.assertEqual(int(out[10, 10, 10]), 1, "the core survives")
        self.assertEqual(int(out[5, 10, 10]), 0, "the surface is removed")

    def test_more_erosion_removes_more(self):
        arr = np.zeros((24, 24, 24), dtype=np.uint8)
        arr[4:20, 4:20, 4:20] = 1
        light = int((np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=1) != 0).sum())
        heavy = int((np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=4) != 0).sum())
        self.assertGreater(light, heavy)

    def test_labels_restricts_which_label_is_eroded(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[2:9, 2:9, 2:9] = 1
        arr[11:18, 11:18, 11:18] = 2
        before_1 = int((arr == 1).sum())
        out = np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=2, labels=[1])
        self.assertEqual(int((out == 1).sum()), 3 * 3 * 3)
        self.assertLess(int((out == 1).sum()), before_1)
        self.assertGreater(int((out == 2).sum()), 300, "label 2 is left essentially untouched")

    def test_eroding_a_thin_structure_away_entirely(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[5, 5, 5] = 1
        self.assertEqual(int((np_utils.np_erode_msk_euclid(arr, n_pixel=2) != 0).sum()), 0)

    def test_use_crop_does_not_change_the_result(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[5:15, 5:15, 5:15] = 1
        cropped = np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=2, use_crop=True)
        plain = np_utils.np_erode_msk_euclid(arr.copy(), n_pixel=2, use_crop=False)
        np.testing.assert_array_equal(cropped, plain)

    def test_the_morphology_helpers_work_in_place(self):
        # Undocumented but consistent convention across np_erode_msk,
        # np_erode_msk_euclid and np_dilate_msk_euclid: the input array is modified and
        # also returned. Callers that need their array intact must pass a copy.
        for name, setup in (
            ("np_erode_msk", lambda a: a.__setitem__((slice(5, 15),) * 3, 1)),
            ("np_erode_msk_euclid", lambda a: a.__setitem__((slice(5, 15),) * 3, 1)),
            ("np_dilate_msk_euclid", lambda a: a.__setitem__((slice(8, 12),) * 3, 1)),
        ):
            with self.subTest(function=name):
                arr = np.zeros((20, 20, 20), dtype=np.uint8)
                setup(arr)
                snapshot = arr.copy()
                getattr(np_utils, name)(arr, n_pixel=2)
                self.assertFalse(np.array_equal(arr, snapshot), "the caller's array is modified in place")


if __name__ == "__main__":
    unittest.main()
