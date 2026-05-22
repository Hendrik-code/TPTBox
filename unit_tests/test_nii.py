# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import operator
import random
import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from TPTBox import NII, v_idx2name
from TPTBox.core import np_utils
from TPTBox.core.compat import zip_strict
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI as Centroids  # _centroids_to_dict_list  # noqa: N811
from TPTBox.core.poi import calc_centroids
from TPTBox.tests.test_utils import get_nii, get_random_ax_code, repeats


def get_all_corner_points(affine, shape) -> np.ndarray:
    import itertools

    from nibabel.affines import apply_affine

    lst = list(itertools.product([0, 1], repeat=3))
    lst = np.array(lst) * (np.array(shape))  # counting starts an 0 to shape-1
    a = apply_affine(affine, lst)
    return a


class TestNII_MathOperators(unittest.TestCase):
    """Tests that NII_Math operators match explicit NumPy array operations.

    Operator domain assumptions:
    - Arithmetic & comparison ops operate on float arrays
    - Bitwise ops (and, or, xor, invert) require integer arrays
    """

    @staticmethod
    def make_nii(shape=(8, 9, 10), seed=0, dtype=float):
        rng = np.random.default_rng(seed)
        arr = rng.normal(size=shape) if dtype is float else rng.integers(0, 8, size=shape, dtype=dtype)
        import nibabel as nib

        nii = NII((arr, np.eye(4), nib.nifti1.Nifti1Header()))
        return nii

    def test_binary_operator_equivalence_float(self):
        """Binary operators valid for float arrays."""
        binary_ops = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.floordiv,
            operator.mod,
            operator.pow,
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
        ]

        for op in binary_ops:
            with self.subTest(op=op):
                nii1 = self.make_nii(seed=1, dtype=float)
                nii2 = self.make_nii(seed=2, dtype=float)

                out_op = op(nii1, nii2)
                expected = op(nii1.get_array(), nii2.get_array())

                out_manual = nii1.set_array(expected, inplace=False)

                self.assertTrue(np.allclose(out_op.get_array(), out_manual.get_array(), equal_nan=True))

    def test_binary_operator_equivalence_bitwise(self):
        """Bitwise binary operators require integer arrays."""
        bitwise_ops = [
            operator.and_,
            operator.or_,
            operator.xor,
        ]

        for op in bitwise_ops:
            with self.subTest(op=op):
                nii1 = self.make_nii(seed=3, dtype=np.int32)
                nii2 = self.make_nii(seed=4, dtype=np.int32)

                out_op = op(nii1, nii2)
                expected = op(nii1.get_array(), nii2.get_array())

                out_manual = nii1.set_array(expected, inplace=False)

                np.testing.assert_array_equal(out_op.get_array(), out_manual.get_array())

    def test_unary_operator_equivalence_float(self):
        """Unary operators valid for float arrays."""
        unary_ops = [operator.neg, operator.pos, operator.abs, np.floor, np.ceil]

        for op in unary_ops:
            with self.subTest(op=op):
                nii = self.make_nii(seed=5, dtype=float)

                out_op = op(nii)
                expected = op(nii.get_array())

                out_manual = nii.set_array(expected, inplace=False)

                self.assertTrue(np.allclose(out_op.get_array(), out_manual.get_array()))

    def test_unary_operator_invert_integer(self):
        """Bitwise invert requires integer arrays."""
        nii = self.make_nii(seed=6, dtype=np.int32)

        out_op = ~nii
        expected = ~nii.get_array()

        out_manual = nii.set_array(expected, inplace=False)

        np.testing.assert_array_equal(out_op.get_array(), out_manual.get_array())

    def test_inplace_binary_operator(self):
        nii1 = self.make_nii(seed=7, dtype=float)
        nii2 = self.make_nii(seed=8, dtype=float)

        arr_before = nii1.get_array().copy()

        nii1 += nii2
        expected = arr_before + nii2.get_array()

        self.assertTrue(np.allclose(nii1.get_array(), expected))

    def test_inplace_unary_operator(self):
        nii = self.make_nii(seed=9, dtype=float)
        arr_before = nii.get_array().copy()

        nii *= 2
        self.assertTrue(np.allclose(nii.get_array(), arr_before * 2))

    def test_round_equivalence(self):
        nii = self.make_nii(seed=4)
        out_op = round(nii, 2)
        expected = np.round(nii.get_array(), 2)
        out_manual = nii.set_array(expected, inplace=False)
        print((out_op.get_array()[0], out_manual.get_array()[0]))
        self.assertTrue(np.allclose(out_op.get_array(), out_manual.get_array()))


class Test_bids_file(unittest.TestCase):
    def test_rescale_corners(self):
        for _ in range(repeats // 4):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            cent = Centroids(cent, orientation=order)

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
        for _ in range(repeats // 5):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            cent = Centroids(cent, orientation=order)
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
            for (k1, k2, v), (k1_2, k2_2, v2) in zip_strict(cdt.items(), cdt2.items()):
                self.assertEqual(k1, k1_2)
                self.assertEqual(k2, k2_2)
                for v, v2 in zip_strict(v, v2):  # noqa: B020, PLW2901
                    self.assertTrue(abs(v - v2) <= 1.01, msg=f"{v},{v2}")

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

    def test_dilate_msk(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            msk2 = msk.dilate_msk(verbose=False)
            self.assertNotEqual(msk.get_array().sum(), msk2.get_array().sum())

    def test_get_segmentation_connected_components(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            label = 1
            subreg_cc = msk.get_connected_components_per_label(labels=label)
            subreg_cc_n = {i: len(subreg_cc[i].unique()) for i in subreg_cc}
            volume = msk.volumes()
            volume_cc = subreg_cc[label].volumes()
            self.assertTrue(volume[label], np.sum(volume_cc.values()))

            # see if get center of masses match with stats centroids
            coms = msk.get_segmentation_connected_components_center_of_mass(label=label)
            n_coms = len(np_utils.np_unique_withoutzero(subreg_cc[label]))
            print(n_coms)
            print(subreg_cc_n)
            self.assertTrue(n_coms == subreg_cc_n[label])
            coms_compare = np_utils.np_center_of_mass(subreg_cc[label].get_seg_array())
            if n_coms == 1:
                print(coms)
                print(coms_compare)
                self.assertTrue(
                    np.array_equal(coms[0], next(iter(coms_compare.values()))),
                    msg=f"{coms[0][0]}, {coms_compare}",
                )
            # if n_coms == 1:
            #    first_centroid = np_utils.np_center_of_mass(subreg_cc[label].get_seg_array())
            #    self.assertTrue(
            #        abs(coms[0][0] - first_centroid[0]) <= 0.00001,
            #        msg=f"{coms[0][0]}, {first_centroid[0]}",
            #    )

    def test_apply_center_crop(self):
        for _ in range(repeats):
            crop_sizes = (random.randint(10, 200), random.randint(10, 200), random.randint(10, 200))
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            msk2 = msk.apply_center_crop(crop_sizes, verbose=random.randint(0, 1) == 0)
            self.assertEqual(msk2.shape, crop_sizes)
            self.assertTrue(msk2.assert_affine(shape=crop_sizes))

    def test_assert_affine(self):
        # asserts with itself
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))

            self.assertTrue(msk.assert_affine(other=msk))
            self.assertTrue(
                msk.assert_affine(affine=msk.affine, zoom=msk.zoom, orientation=msk.orientation, origin=msk.origin, shape=msk.shape)
            )

        # TODO cases where it should return False

    def test_center_of_masses(self):
        # asserts with itself
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))

            coms = msk.center_of_masses()
            np_coms = np_utils.np_center_of_mass(msk.get_seg_array())

            print("coms", coms)
            print("np_coms", np_coms)
            print("cent", cent)

            for i, g in coms.items():
                self.assertTrue(i in np_coms)
                self.assertTrue(np.all([g[idx] == np_coms[i][idx] for idx in range(3)]))
                self.assertTrue((i, 50) in cent)
                self.assertTrue(np.all([g[idx] == cent[i, 50][idx] for idx in range(3)]))
            # self.assertEqual(coms, np_coms)

    def test_apply_pad(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            msk2 = msk.apply_pad(padd=[(2, 2), (2, 2), (2, 2)], inplace=False)
            print(msk)
            print(msk2)
            for i in range(3):
                self.assertEqual(msk.shape[i], msk2.shape[i] - 4)
            self.assertFalse(msk.assert_affine(other=msk2, raise_error=False))

            msk3 = msk2.pad_to(msk.shape, inplace=False)
            print(msk3)
            for i in range(3):
                self.assertEqual(msk.shape[i], msk3.shape[i])
            self.assertTrue(msk3.assert_affine(other=msk))

    def test_errode(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(3, 10))
            msk2 = msk.copy()

            r = random.randint(1, 2)
            s1 = msk.erode_msk(r, verbose=False).sum()
            s2 = msk2.erode_msk(r, use_crop=False, verbose=False).sum()
            assert s1 == s2, (s1, s2)
            s1 = msk.dilate_msk(r, verbose=False).sum()
            s2 = msk2.dilate_msk(r, use_crop=False, verbose=False).sum()
            assert s1 == s2, (s1, s2)
            s1 = msk.dilate_msk(r, verbose=False).erode_msk(r, verbose=False).sum()
            s2 = msk2.dilate_msk(r, verbose=False).erode_msk(r, use_crop=False, verbose=False).sum()
            assert s1 == s2, (s1, s2)


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
