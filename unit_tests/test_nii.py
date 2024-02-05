# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import random
import unittest

import nibabel as nib
import numpy as np

from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI as Centroids  # _centroids_to_dict_list
from TPTBox.core.poi import calc_centroids, v_idx2name
from TPTBox.tests.test_utils import get_nii, get_random_ax_code, repeats


def get_all_corner_points(affine, shape) -> np.ndarray:
    import itertools

    from nibabel.affines import apply_affine

    lst = list(itertools.product([0, 1], repeat=3))
    lst = np.array(lst) * (np.array(shape))  # counting starts an 0 to shape-1
    a = apply_affine(affine, lst)
    return a


class Test_bids_file(unittest.TestCase):
    def test_rescale(self):
        for _ in range(repeats // 10):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            cent = Centroids(cent, orientation=order)

            axcode_start = get_random_ax_code()
            # msk.reorient_(axcode_start)

            cdt = calc_centroids(msk)
            voxel_spacing = (
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
            )
            voxel_spacing2 = (1.0, 1.0, 1.0)
            msk2 = msk.rescale(voxel_spacing=voxel_spacing, verbose=False, inplace=False)
            msk2 = msk2.rescale(voxel_spacing=voxel_spacing2, verbose=False)
            cdt2 = calc_centroids(msk2)

            for (k1, k2, v), (k1_2, k2_2, v2) in zip(cdt.items(), cdt2.items()):
                self.assertEqual(k1, k1_2)
                self.assertEqual(k2, k2_2)
                for v, v2 in zip(v, v2):
                    self.assertAlmostEqual(v, v2)

    def test_rescale_corners(self):
        for _ in range(repeats // 4):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            cent = Centroids(cent, order)

            axcode_start = get_random_ax_code()
            msk.reorient_(axcode_start)
            voxel_spacing = (
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
            )
            msk2 = msk.rescale(voxel_spacing=voxel_spacing)

            def sub_test(msk, msk2, delta=0):
                p1 = get_all_corner_points(msk.affine, msk.shape)
                p2 = get_all_corner_points(msk2.affine, msk2.shape)
                if not (np.abs(np.subtract(p1, p2)) <= delta).all():
                    print(p1, p2, np.subtract(p1, p2))
                self.assertTrue(np.isclose(p1, p2).sum() >= 12)
                self.assertTrue((np.abs(np.subtract(p1, p2)) <= delta).all(), msg=f"{p1}, {p2}")

            sub_test(msk, msk2, delta=0)
            voxel_spacing = (
                random.choice([1, 2, 3, 4]),
                random.choice([1, 2, 3, 4]),
                random.choice([1, 2, 3, 4]),
            )
            msk3 = msk.rescale(voxel_spacing=voxel_spacing)
            sub_test(msk, msk3, delta=max(voxel_spacing) // 2)
            msk4 = msk2.rescale(voxel_spacing=(1, 1, 1))
            sub_test(msk, msk4, delta=max(voxel_spacing) // 2)

    def test_reorient_corners(self):
        for _ in range(repeats):  # repeats // 10
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            cent = Centroids(cent, orientation=order)

            msk.reorient_(get_random_ax_code())
            a = get_random_ax_code()
            msk2 = msk.reorient(a)

            def sub_test(msk, msk2):
                p1 = get_all_corner_points(msk.affine, [a - 1 for a in msk.shape])

                p2 = get_all_corner_points(msk2.affine, [a - 1 for a in msk2.shape])
                p1 = [p1[i] for i in range(p1.shape[0])]

                self.assertEqual(len(p1), 8)

                for i in range(p2.shape[0]):
                    p = p2[i]
                    for j, p_other in enumerate(p1):
                        if np.isclose(p, p_other).all():
                            p1.pop(j)
                            break
                self.assertEqual(len(p1), 0, msg=str(np.array(p1)) + "__" + str(p2))

            sub_test(msk, msk2)
            msk2 = msk2.reorient(get_random_ax_code())
            sub_test(msk, msk2)
            msk2 = msk2.reorient(get_random_ax_code())
            sub_test(msk, msk2)
            msk2 = msk2.reorient(msk.orientation)
            sub_test(msk, msk2)
            self.assertTrue(np.isclose(msk.get_array(), msk2.get_array()).all())

    def test_rescale_and_reorient(self):
        for _ in range(repeats // 10):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            cent = Centroids(cent, order)
            axcode_start = order  # get_random_ax_code()
            msk.reorient_(axcode_start)
            cdt = calc_centroids(msk)
            axcode = order  # get_random_ax_code()
            voxel_spacing = (
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
                random.choice([1, 1 / 2, 1 / 3, 1 / 4]),
            )
            voxel_spacing2 = (1.0, 1.0, 1.0)
            msk2 = msk.rescale_and_reorient(axcode, voxel_spacing=voxel_spacing, verbose=False, inplace=False)
            msk2 = msk2.rescale_and_reorient(axcode_start, voxel_spacing=voxel_spacing2, verbose=False)
            cdt2 = calc_centroids(msk2)
            for (k1, k2, v), (k1_2, k2_2, v2) in zip(cdt.items(), cdt2.items()):
                self.assertEqual(k1, k1_2)
                self.assertEqual(k2, k2_2)
                for v, v2 in zip(v, v2):
                    self.assertAlmostEqual(v, v2)

    def test_get_plane(self):
        for _ in range(repeats):
            a = np.zeros((10, 20, 30), dtype=np.uint16)
            nii = NII(nib.nifti1.Nifti1Image(a, np.eye(4)))
            self.assertEqual("iso", nii.get_plane())
            nii.reorient_(("R", "I", "P"))

            def r(idx):
                o = [random.random() * 0.5 + 0.5 for i in range(3)]
                o[idx] += 1
                return tuple(o)

            nii.rescale_(r(0))
            self.assertEqual("sag", nii.get_plane())
            nii.rescale_(r(1))
            self.assertEqual("ax", nii.get_plane())
            nii.rescale_(r(2))
            self.assertEqual("cor", nii.get_plane())

            nii.reorient_(("I", "P", "L"))
            nii.rescale_(r(2))
            self.assertEqual("sag", nii.get_plane())
            nii.rescale_(r(0))
            self.assertEqual("ax", nii.get_plane())
            nii.rescale_(r(1))
            self.assertEqual("cor", nii.get_plane())

            nii.reorient_(("S", "A", "L"))
            nii.rescale_(r(2))
            self.assertEqual("sag", nii.get_plane())
            nii.rescale_(r(0))
            self.assertEqual("ax", nii.get_plane())
            nii.rescale_(r(1))
            self.assertEqual("cor", nii.get_plane())

    def test_map(self):
        for _ in range(repeats // 10):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            a = random.sample(range(1, 29), len(cent))
            cdt = Centroids(cent, **msk._extract_affine())
            mapping = dict(enumerate(a, start=1))
            mapping_invert = {v: k for k, v in mapping.items()}
            # print(mapping)
            msk2 = msk.map_labels(mapping, verbose=False)
            cdt2 = calc_centroids(msk2)
            self.assertEqual(len(cent), len(cdt2))
            for key in a:
                self.assertTrue((key, 50) in cdt2)
            self.assertNotEqual(cdt, cdt2)
            msk2.map_labels_(mapping_invert, verbose=False)
            cdt2 = calc_centroids(msk2)
            self.assertEqual(cdt, cdt2)

    def test_map2(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            a = random.sample(range(1, 29), len(cent))
            cdt = Centroids(cent, **msk._extract_affine())

            def r(i):
                if random.random() > 0.5:
                    return v_idx2name[i]
                if random.random() > 0.5:
                    return str(i)
                return i

            mapping = {r(i): r(k) for i, k in enumerate(a, start=1)}
            mapping_invert = {v: k for k, v in mapping.items()}
            # print(mapping)
            msk2 = msk.map_labels(mapping, verbose=False)  # type: ignore
            cdt2 = calc_centroids(msk2)
            self.assertEqual(len(cent), len(cdt2))
            for key in a:
                self.assertTrue((key, 50) in cdt2)
            self.assertNotEqual(cdt, cdt2)
            msk2.map_labels_(mapping_invert, verbose=False)
            cdt2 = calc_centroids(msk2)
            self.assertEqual(cdt, cdt2)

    def test_erode_msk(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            msk2 = msk.erode_msk(verbose=False)
            self.assertNotEqual(msk.get_array().sum(), msk2.get_array().sum())

    def test_assert_affine(self):
        # asserts with itself
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))

            self.assertTrue(msk.assert_affine(other=msk))
            self.assertTrue(
                msk.assert_affine(affine=msk.affine, zoom=msk.zoom, orientation=msk.orientation, origin=msk.origin, shape=msk.shape)
            )

        # TODO cases where it should return False


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
