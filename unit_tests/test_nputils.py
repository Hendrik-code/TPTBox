from __future__ import annotations

import random
import unittest

import numpy as np

from TPTBox.core import np_utils
from TPTBox.tests.test_utils import get_nii, repeats


def make_test_array_repeating(shape=(4, 4), labels=(0, 1, 2, 3)) -> np.ndarray:
    """Creates a test array with a repeating pattern of the given labels."""
    size = np.prod(shape)
    label_pattern = np.resize(np.array(labels), size)
    return label_pattern.reshape(shape)


def make_labeled_3d_array() -> np.ndarray:
    """Create a small 3D array with labeled connected components.
    Components:
    - One block of 1s
    - One isolated voxel of 2
    """
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1  # a small cube
    arr[4, 4, 4] = 2  # single voxel
    return arr


class Test_np_utils(unittest.TestCase):
    def test_np_extract_label(self):
        arr = make_test_array_repeating()
        print(make_test_array_repeating())
        arr_copy = arr.copy()

        # Test extracting single label
        out = np_utils.np_extract_label(arr, label=2, to_label=99, inplace=False)
        assert np.all(arr == arr_copy), "Original array should not be modified when inplace=False"
        assert np.all((out == 99) == (arr == 2)), "Only positions with label 2 should be 99"
        assert np.all((out == 0) == (arr != 2)), "All other positions should be 0"

        # Test extracting zero label
        out_zero = np_utils.np_extract_label(arr, label=0, to_label=42, inplace=False)
        assert np.all((out_zero == 42) == (arr == 0)), "Label 0 should be correctly extracted"

        # Test multiple labels
        out_multi = np_utils.np_extract_label(arr, label=[1, 3], to_label=7, inplace=False)
        mask = np.isin(arr, [1, 3])
        assert np.all((out_multi == 7) == mask), "Labels 1 and 3 should be set to 7"

        # Test inplace modification
        arr_inplace = make_test_array_repeating()
        result = np_utils.np_extract_label(arr_inplace, label=1, to_label=5, inplace=True)
        assert result is arr_inplace, "Should return the same array if inplace=True"
        assert np.all((arr_inplace == 5) == (arr_copy == 1)), "Inplace modification should match label positions"

    def test_np_volume(self):
        arr = make_test_array_repeating().astype(np.uint16)
        arr_c = arr.copy()
        unique, counts = np.unique(arr, return_counts=True)
        expected = dict(zip(unique, counts))

        # Test without zero
        expected_no_zero = {k: v for k, v in expected.items() if k != 0}
        result = np_utils.np_volume(arr, include_zero=False)
        assert np.all(arr_c == arr), "arr changed"
        assert result == expected_no_zero, "Should exclude label 0 when include_zero=False"

        # Test with zero
        result_with_zero = np_utils.np_volume(arr, include_zero=True)
        assert np.all(arr_c == arr), "arr changed"
        assert result_with_zero == expected, "Should include label 0 when include_zero=True"

    def test_dice(self):
        for value in range(repeats):
            dims = random.randint(2, 3)
            shape = tuple(random.randint(5, 100) for d in range(dims))
            arr = np.ones(shape=shape, dtype=np.uint8) * value
            binary_compare = random.random() < 0.5
            dice = np_utils.np_dice(arr, arr, label=value, binary_compare=binary_compare)
            self.assertEqual(dice, 1.0)

    def test_erode_dilate(self):
        for value in range(repeats):
            nii, points, orientation, sizes = get_nii()
            arr = nii.get_seg_array()
            volume = np_utils.np_volume(arr)
            func = np_utils.np_erode_msk if value % 2 == 0 else np_utils.np_dilate_msk
            arr2 = func(arr, n_pixel=1, connectivity=1)
            volume2 = np_utils.np_volume(arr2)

            for k, v in volume.items():
                if value % 2 == 0:
                    if k not in volume2:
                        self.assertTrue(True)
                    else:
                        self.assertTrue(
                            v >= volume2[k] if k != 0 else v <= volume2[k],
                            msg=f"{volume}, {volume2}",
                        )
                else:
                    self.assertTrue(
                        v <= volume2[k] if k != 0 else v >= volume2[k],
                        msg=f"{volume}, {volume2}",
                    )

    def test_erodedilate_notpresentlabel(self):
        for value in range(repeats):
            nii, points, orientation, sizes = get_nii()
            arr = nii.get_seg_array()
            volume = np_utils.np_volume(arr)
            label = max(list(volume.keys())) + 1
            func = np_utils.np_erode_msk if value % 2 == 0 else np_utils.np_dilate_msk
            arr2 = func(arr, n_pixel=1, connectivity=1, label_ref=label)
            volume2 = np_utils.np_volume(arr2)

            for k, v in volume.items():
                self.assertEqual(v, volume2[k])

    def test_maplabels(self):
        for _value in range(repeats):
            nii, points, orientation, sizes = get_nii()
            arr = nii.get_seg_array().astype(np.uint16)
            volume = np_utils.np_volume(arr)
            labelmap = {i: random.randint(0, 10) for i in volume.keys()}
            arr2 = np_utils.np_map_labels(arr, labelmap)
            volume2 = np_utils.np_volume(arr2)

            correct = {}
            for source, target in labelmap.items():
                v = volume[source]
                if target not in correct:
                    correct[target] = 0
                correct[target] += v

            print(volume)
            print(volume2)

            for k, v in volume2.items():
                self.assertTrue(v == correct[k])

    def test_cutout(self):
        for _value in range(repeats):
            nii, points, orientation, sizes = get_nii()
            arr = nii.get_seg_array()
            shape = arr.shape
            cutout_size = tuple(int(random.random() * i * 2) for i in shape)
            cutout_size = tuple(c if c % 2 == 0 else c + 1 for c in cutout_size)
            cp = points[(1, 50)]
            arr_cut, _, _ = np_utils.np_calc_crop_around_centerpoint(cp, arr, cutout_size=cutout_size)

            shp = arr_cut.shape
            self.assertTrue(shp[0] == cutout_size[0], msg=f"{shp}, {cutout_size}")
            self.assertTrue(shp[1] == cutout_size[1], msg=f"{shp}, {cutout_size}")
            self.assertTrue(shp[2] == cutout_size[2], msg=f"{shp}, {cutout_size}")

    def test_fillholes_2D_simple(self):
        a = np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        b = np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        c = np_utils.np_fill_holes(a)
        print(c)
        self.assertTrue(np.all(b == c))

    def test_fillholes_2D_border(self):
        a = np.array(
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        b = np.array(
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        c = np_utils.np_fill_holes(a)
        print(c)
        self.assertTrue(np.all(b == c))

    def test_fillholes_3D_slicewise(self):
        a = np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        b = np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        for ax in [0, 1, 2]:
            aa = np.expand_dims(a, axis=ax)
            bb = np.expand_dims(b, axis=ax)
            c = np_utils.np_fill_holes(aa, slice_wise_dim=ax)
            self.assertTrue(np.all(bb == c))

    def test_fillholes(self):
        for _value in range(repeats):
            nii, points, orientation, sizes = get_nii(min_size=3)
            arr = nii.get_seg_array()
            volume = np_utils.np_volume(arr)
            for (p1, _p2), com in points.items():
                rand_point_in_cube = tuple(
                    int(c) + random.randint(-sizes[p1 - 1][idx] // 2, sizes[p1 - 1][idx] // 2) for idx, c in enumerate(com)
                )
                arr[int(rand_point_in_cube[0])][int(rand_point_in_cube[1])][int(rand_point_in_cube[2])] = 0
                filled = np_utils.np_fill_holes(arr, use_crop=False)
                volume_filled = np_utils.np_volume(filled)
                self.assertTrue(volume[p1] == volume_filled[p1])

    def test_connected_components(self):
        for _value in range(repeats):
            num_points = 3
            nii, points, orientation, sizes = get_nii(min_size=3, num_point=num_points)
            arr = nii.get_seg_array()
            volume = np_utils.np_volume(arr)
            subreg_cc, subreg_cc_n = np_utils.np_connected_components(arr)
            cc_volume = np_utils.np_volume(subreg_cc)
            self.assertTrue(subreg_cc_n == len(np_utils.np_unique_withoutzero(subreg_cc)))
            # for this test, this is also true
            self.assertTrue(subreg_cc_n == len(np_utils.np_unique_withoutzero(arr)))
            for label in range(1, num_points + 1):
                self.assertTrue(volume[label], cc_volume[label])  # type: ignore

    def test_connected_components_per_label(self):
        for _value in range(repeats):
            nii, points, orientation, sizes = get_nii(min_size=3)
            arr = nii.get_seg_array()
            volume = np_utils.np_volume(arr)
            subreg_cc = np_utils.np_connected_components_per_label(arr)
            for label in [1, 2, 3]:
                volume_cc = np_utils.np_volume(subreg_cc[label])
                self.assertTrue(volume[label], np.sum(volume_cc.values()))  # type: ignore

                # see if get center of masses match with stats centroids
                coms = np_utils.np_get_connected_components_center_of_mass(arr, label)
                n_coms = len(np_utils.np_unique_withoutzero(subreg_cc[label]))
                coms_compare = np_utils.np_center_of_mass(subreg_cc[label])
                if n_coms == 1:
                    print(coms)
                    print(coms_compare)
                    self.assertTrue(
                        np.array_equal(coms[0], next(iter(coms_compare.values()))),
                        msg=f"{coms[0][0]}, {coms_compare}",
                    )

    def test_get_largest_k_connected_components_non_global(self):
        a = np.zeros((50, 50), dtype=np.uint16)
        a[10:20, 10:20] = 1
        a[30:50, 30:50] = 1
        a[1:4, 1:4] = 1

        # k less than N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=2, return_original_labels=False)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 2, a_volume)
        self.assertTrue(a_volume[1] > a_volume[2])

        # k == N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=3, return_original_labels=False)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 3)
        self.assertTrue(a_volume[1] > a_volume[2] > a_volume[3])

        # k > N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=20, return_original_labels=False)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 3)
        self.assertTrue(a_volume[1] > a_volume[2] > a_volume[3])

        a = np.zeros((50, 50), dtype=np.uint16)
        a[10:20, 10:20] = 7
        a[30:50, 30:50] = 5
        a[1:4, 1:4] = 1

        # k less than N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=2, return_original_labels=False)
        a_volume = np_utils.np_volume(a_cc)
        self.assertTrue(len(a_volume) == 3, a_volume)

    def test_get_largest_k_connected_components(self):
        a = np.zeros((50, 50), dtype=np.uint16)
        a[10:20, 10:20] = 5
        a[30:50, 30:50] = 7
        a[1:4, 1:4] = 1

        # k less than N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=2, return_original_labels=False, k_larges_global=True)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 2, a_volume)
        self.assertTrue(a_volume[1] > a_volume[2])

        # k == N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=3, return_original_labels=False, k_larges_global=True)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 3)
        self.assertTrue(a_volume[1] > a_volume[2] > a_volume[3])

        # k > N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=20, return_original_labels=False, k_larges_global=True)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 3)
        self.assertTrue(a_volume[1] > a_volume[2] > a_volume[3])

        a = np.zeros((50, 50), dtype=np.uint16)
        a[10:20, 10:20] = 1
        a[30:50, 30:50] = 1
        a[1:4, 1:4] = 1

        # k less than N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=2, return_original_labels=False, k_larges_global=True)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 2, a_volume)
        self.assertTrue(a_volume[1] > a_volume[2])

    def test_fill_holes(self):
        # Create a test NII object with a segmentation mask
        arr = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=np.int16)

        # Fill the holes in the segmentation mask
        arr = np_utils.np_fill_holes(arr, label_ref=1)

        # Check that the holes are filled correctly
        expected_result = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(arr, expected_result), (arr, expected_result))

    def test_np_center_of_bbox_binary(self):
        arr = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 1]]], dtype=np.uint8)
        result = np_utils.np_center_of_bbox_binary(arr)
        print(result)
        # expected = [1, 1, 0]  # Adjust based on your bounding box logic
        # self.assertEqual(result, expected)

    def test_smooth_msk(self):
        # Create a test NII object with a segmentation mask
        data = np.zeros((10, 10), dtype=np.uint16)
        data[3:8, 3:8] = 1

        print(data)

        # Dilate the segmentation mask
        smoothed = np_utils.np_smooth_gaussian_labelwise(
            data,
            label_to_smooth=1,
            sigma=1,
            radius=4,
            truncate=4,
            boundary_mode="nearest",
            dilate_prior=0,
            smooth_background=True,
        )

        print()
        print(smoothed)

        # Check that the dilated mask is correct
        expected = np.zeros((10, 10), dtype=np.uint16)
        expected[3:8, 3:8] = 1
        expected[3, 3] = 0
        expected[3, 7] = 0
        expected[7, 3] = 0
        expected[7, 7] = 0
        self.assertTrue(np.array_equal(smoothed, expected), (smoothed[5], expected[5]))

    def test_smooth_msk2(self):
        # Create a test NII object with a segmentation mask
        data = np.zeros((10, 10), dtype=np.uint16)
        data[3:8, 3:8] = 1

        print(data)

        # Dilate the segmentation mask
        smoothed = np_utils.np_smooth_gaussian_labelwise(
            data,
            label_to_smooth=1,
            sigma=3,
            radius=4,
            truncate=4,
            boundary_mode="nearest",
            dilate_prior=1,
            smooth_background=True,
        )

        print()
        print(smoothed)

        # Check that the dilated mask is correct
        expected = np.zeros((10, 10), dtype=np.uint16)
        expected[3:8, 3:8] = 1
        expected[2, 5] = 1
        expected[5, 2] = 1
        expected[8, 5] = 1
        expected[5, 8] = 1
        self.assertTrue(np.array_equal(smoothed, expected), (smoothed[5], expected[5]))

    def test_np_binary_fill_holes_and_set_inter_labels_based_on_majority(self):
        # Create a test NII object with a segmentation mask
        data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 2],
                [0, 0, 1, 0, 2],
                [0, 0, 2, 2, 2],
            ],
            dtype=np.uint16,
        )

        print(data)

        # Dilate the segmentation mask
        filled = np_utils.np_fill_holes_global_with_majority_voting(
            data,
            connectivity=1,
        )

        print()
        print(filled)

        # Check that the dilated mask is correct
        expected = data.copy()
        expected[2, 2] = 1
        expected[3, 3] = 2
        self.assertTrue(np.array_equal(filled, expected))


class Test_point_helpers(unittest.TestCase):
    def test_np_index_single_and_multiple_hits(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        self.assertTrue(np.array_equal(np_utils.np_index(arr, [4, 5, 6]), [1]))
        self.assertTrue(np.array_equal(np_utils.np_index(arr, [1, 2, 3]), [0, 2]))

    def test_np_index_no_hit_is_empty(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        idx = np_utils.np_index(arr, [7, 8, 9])
        self.assertEqual(len(idx), 0)

    def test_np_find_closest_point_index_exact_match_returns_first(self):
        arr = np.array([[0, 0, 0], [5, 5, 5], [0, 0, 0]])
        # both row 0 and row 2 match exactly; the lowest index wins
        self.assertEqual(np_utils.np_find_closest_point_index(arr, [0, 0, 0]), 0)

    def test_np_find_closest_point_index_matches_bruteforce(self):
        rng = np.random.default_rng(42)
        for dtype in (np.int32, np.float64):
            for _ in range(repeats):
                arr = (rng.random((200, 3)) * 50).astype(dtype)
                point = (rng.random(3) * 50).astype(dtype)
                expected = int(np.argmin(np.linalg.norm(arr - point, axis=1)))
                got = np_utils.np_find_closest_point_index(arr, point)
                # ties may resolve to a different index, so compare distances not indices
                self.assertAlmostEqual(
                    float(np.linalg.norm(arr[got] - point)),
                    float(np.linalg.norm(arr[expected] - point)),
                    places=6,
                )

    def test_np_find_closest_point_index_rejects_empty(self):
        with self.assertRaises(AssertionError):
            np_utils.np_find_closest_point_index(np.zeros((0, 3), dtype=int), [0, 0, 0])


class Test_boundary_normals(unittest.TestCase):
    def make_two_slabs(self):
        """Label 1 fills x < 6, label 2 fills x >= 6, so the interface is the plane x == 5/6."""
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[:6] = 1
        arr[6:] = 2
        return arr

    def test_interface_voxels_are_on_the_boundary_of_the_first_label(self):
        arr = self.make_two_slabs()
        coords, normals = np_utils.np_compute_boundary_normals(arr, 1, 2)
        self.assertEqual(len(coords), 12 * 12)
        self.assertTrue((coords[:, 0] == 5).all())
        self.assertTrue((arr[tuple(coords.T)] == 1).all())
        self.assertEqual(len(normals), len(coords))

    def test_normals_are_unit_length_and_point_into_the_label(self):
        arr = self.make_two_slabs()
        _, normals = np_utils.np_compute_boundary_normals(arr, 1, 2)
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3))
        # label 1 lies towards -x, and the normal follows increasing mask density
        self.assertTrue(np.allclose(normals[:, 0], -1.0, atol=1e-3))
        self.assertTrue(np.allclose(normals[:, 1:], 0.0, atol=1e-3))

    def test_labels_that_do_not_touch_give_empty_result(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[1:3] = 1
        arr[7:9] = 2
        coords, normals = np_utils.np_compute_boundary_normals(arr, 1, 2)
        self.assertEqual(coords.shape, (0, 3))
        self.assertEqual(normals.shape, (0, 3))

    def test_diagonal_only_contact_does_not_count(self):
        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        arr[2, 2, 2] = 1
        arr[3, 3, 3] = 2  # shares only a corner, not a face
        coords, _ = np_utils.np_compute_boundary_normals(arr, 1, 2)
        self.assertEqual(len(coords), 0)


class Test_vector_helpers(unittest.TestCase):
    def test_np_unit_vector(self):
        self.assertTrue(np.allclose(np_utils.np_unit_vector(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0]))

    def test_np_angle_between(self):
        self.assertAlmostEqual(np_utils.np_angle_between((1, 0, 0), (0, 1, 0)), np.pi / 2)
        self.assertAlmostEqual(np_utils.np_angle_between((1, 0, 0), (1, 0, 0)), 0.0)
        self.assertAlmostEqual(np_utils.np_angle_between((1, 0, 0), (-1, 0, 0)), np.pi)
        self.assertAlmostEqual(np_utils.np_angle_between((1, 0, 0), (0, 1, 0), degrees=True), 90.0)


def _dumbbell(neck: int = 2) -> np.ndarray:
    """Two cubes joined by a thin neck, forming a single connected component."""
    arr = np.zeros((40, 24, 24), dtype=np.uint8)
    arr[4:16, 6:18, 6:18] = 1
    arr[24:36, 6:18, 6:18] = 1
    c = 12
    arr[16:24, c - neck : c + neck, c - neck : c + neck] = 1
    return arr


class Test_split_connected_component(unittest.TestCase):
    def test_input_really_is_one_component(self):
        _, n = np_utils.np_connected_components(_dumbbell(), connectivity=3)
        self.assertEqual(n, 1)

    def test_erosion_splits_into_two_parts(self):
        out = np_utils.np_split_connected_component(_dumbbell(), method="erosion")
        self.assertEqual(sorted(int(v) for v in np.unique(out)), [0, 1, 2])
        self.assertGreater((out == 1).sum(), 0)
        self.assertGreater((out == 2).sum(), 0)

    def test_erosion_full_partition_keeps_every_voxel(self):
        arr = _dumbbell()
        out = np_utils.np_split_connected_component(arr, method="erosion", full_partition=True)
        self.assertEqual((out != 0).sum(), (arr != 0).sum())
        # and never paints outside the input
        self.assertEqual(((out != 0) & (arr == 0)).sum(), 0)

    def test_erosion_without_full_partition_returns_only_the_cores(self):
        arr = _dumbbell()
        out = np_utils.np_split_connected_component(arr, method="erosion", full_partition=False)
        self.assertLess((out != 0).sum(), (arr != 0).sum())

    def test_the_two_parts_land_on_opposite_cubes(self):
        arr = _dumbbell()
        out = np_utils.np_split_connected_component(arr, method="erosion")
        low = out[4:16][out[4:16] != 0]
        high = out[24:36][out[24:36] != 0]
        # each cube must be dominated by a single, and different, label
        self.assertNotEqual(np.bincount(low).argmax(), np.bincount(high).argmax())

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            np_utils.np_split_connected_component(_dumbbell(), method="bogus")

    def test_solid_block_that_cannot_split_raises(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[5:15, 5:15, 5:15] = 1
        with self.assertRaises(ValueError):
            np_utils.np_split_connected_component(arr, method="erosion", max_iter=2)


class Test_split_connected_component_mincut(unittest.TestCase):
    def setUp(self):
        try:
            import networkx as nx  # noqa: F401
        except ImportError:
            self.skipTest("networkx not installed")

    def test_mincut_partitions_the_whole_volume(self):
        arr = _dumbbell()
        out = np_utils.np_split_connected_component(arr, method="mincut", connectivity=1)
        self.assertEqual(sorted(int(v) for v in np.unique(out)), [0, 1, 2])
        self.assertEqual((out != 0).sum(), (arr != 0).sum())

    def test_rejects_input_that_is_already_two_components(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        arr[14:18, 14:18, 14:18] = 1
        with self.assertRaises(ValueError):
            np_utils.np_split_connected_component(arr, method="mincut", connectivity=1)

    def test_min_volume_guard(self):
        with self.assertRaises(ValueError):
            np_utils.np_split_connected_component(_dumbbell(), method="mincut", connectivity=1, min_volume=10**6)

    def test_max_cut_guard(self):
        with self.assertRaises(ValueError):
            np_utils.np_split_connected_component(_dumbbell(), method="mincut", connectivity=1, max_cut=0.5)

    def test_anisotropic_zoom_is_accepted(self):
        out = np_utils.np_split_connected_component(_dumbbell(), method="mincut", connectivity=1, zoom=(1.0, 1.0, 3.0))
        self.assertEqual(sorted(int(v) for v in np.unique(out)), [0, 1, 2])

    def test_diagonal_edges_branch_runs(self):
        """The upstream version raised NameError here (undefined ``geom.norm``)."""
        out = np_utils.np_split_connected_component(_dumbbell(), method="mincut", connectivity=1, add_diagonal_edges=True)
        self.assertEqual(sorted(int(v) for v in np.unique(out)), [0, 1, 2])

    def test_max_ignore_none_runs(self):
        """The upstream version raised NameError here (``max_errors`` never assigned)."""
        out = np_utils.np_split_connected_component(_dumbbell(), method="mincut", connectivity=1, max_ignore=None)
        self.assertEqual(sorted(int(v) for v in np.unique(out)), [0, 1, 2])


class Test_connected_component_contact_map(unittest.TestCase):
    def test_contact_map_labels_and_contact_zone(self):
        out = np_utils.np_split_connected_component(_dumbbell(), method="erosion", full_partition=False)
        contact = np_utils.np_connected_component_contact_map(out == 1, out == 2)
        self.assertTrue({int(v) for v in np.unique(contact)} <= {0, 1, 2, 3})
        self.assertGreater((contact == 3).sum(), 0)

    def test_inputs_are_not_mutated(self):
        out = np_utils.np_split_connected_component(_dumbbell(), method="erosion", full_partition=False)
        a, b = out == 1, out == 2
        a_before, b_before = a.copy(), b.copy()
        np_utils.np_connected_component_contact_map(a, b)
        self.assertTrue(np.array_equal(a, a_before))
        self.assertTrue(np.array_equal(b, b_before))


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
