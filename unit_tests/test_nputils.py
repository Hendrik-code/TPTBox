# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import random  # noqa: E402
import unittest  # noqa: E402

import numpy as np  # noqa: E402

from TPTBox.core import np_utils  # noqa: E402
from TPTBox.tests.test_utils import get_nii, repeats  # noqa: E402


class Test_np_utils(unittest.TestCase):
    def test_dice(self):
        for value in range(repeats):
            dims = random.randint(2, 3)
            shape = tuple(random.randint(5, 100) for d in range(dims))
            arr = np.ones(shape=shape, dtype=np.uint8) * value
            binary_compare = random.random() < 0.5
            dice = np_utils.np_dice(arr, arr, label=value, binary_compare=binary_compare)
            self.assertEqual(dice, 1.0)

    def test_erodedilate(self):
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

    def test_get_largest_k_connected_components(self):
        a = np.zeros((50, 50), dtype=np.uint16)
        a[10:20, 10:20] = 5
        a[30:50, 30:50] = 7
        a[1:4, 1:4] = 1

        # k less than N
        a_cc = np_utils.np_filter_connected_components(a, largest_k_components=2, return_original_labels=False)
        a_volume = np_utils.np_volume(a_cc)
        print(a_volume)
        self.assertTrue(len(a_volume) == 2)
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


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
