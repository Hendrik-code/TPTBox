"""Unit tests for the NII_Math mixin (TPTBox/core/nii_wrapper_math.py).

Covers arithmetic/comparison/bitwise operator dunders (and their in-place
variants), unary operators, reductions, clamp/normalize/threshold helpers and
the image-quality metrics (ssim/psnr/dice/betti_numbers).
"""

from __future__ import annotations

import math
import operator
import unittest

import nibabel as nib
import numpy as np
import pytest

from TPTBox import NII
from TPTBox.tests.test_utils import get_test_ct, get_test_mri


def _mk(arr: np.ndarray) -> NII:
    """Wrap a numpy array in an NII (identity affine, explicit header so dtype is preserved)."""
    return NII((arr, np.eye(4), nib.nifti1.Nifti1Header()))


def _farr(shape=(8, 9, 10), seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def _iarr(shape=(8, 9, 10), seed=0, high=8) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, high, size=shape, dtype=np.int32)


class Test_Math_BinaryOperators(unittest.TestCase):
    def test_binary_float_ops_match_numpy(self):
        ops = [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod, operator.pow]
        # positive operands so floor-div / mod / pow are all well-defined (no NaNs)
        a, b = np.abs(_farr(seed=1)) + 1.0, np.abs(_farr(seed=2)) + 1.0
        for op in ops:
            with self.subTest(op=op.__name__):
                out = op(_mk(a), _mk(b)).get_array()
                np.testing.assert_allclose(out, op(a, b), equal_nan=True)

    def test_binary_with_scalar(self):
        a = _farr(seed=3)
        np.testing.assert_allclose((_mk(a) + 5).get_array(), a + 5)
        np.testing.assert_allclose((_mk(a) * 2).get_array(), a * 2)

    def test_binary_with_ndarray(self):
        a, b = _farr(seed=4), _farr(seed=5)
        np.testing.assert_allclose((_mk(a) - b).get_array(), a - b)

    def test_right_hand_add_sub(self):
        a = _farr(seed=6)
        np.testing.assert_allclose((2.0 + _mk(a)).get_array(), 2.0 + a)
        np.testing.assert_allclose((10.0 - _mk(a)).get_array(), 10.0 - a)

    def test_bitshift_ops(self):
        a = _iarr(seed=7)
        np.testing.assert_array_equal((_mk(a) << 1).get_array(), a << 1)
        np.testing.assert_array_equal((_mk(a) >> 1).get_array(), a >> 1)


class Test_Math_Comparisons(unittest.TestCase):
    def test_comparison_ops_match_numpy(self):
        ops = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge]
        a, b = _farr(seed=1), _farr(seed=2)
        for op in ops:
            with self.subTest(op=op.__name__):
                out = op(_mk(a), _mk(b)).get_array().astype(bool)
                np.testing.assert_array_equal(out, op(a, b))


class Test_Math_Bitwise(unittest.TestCase):
    def test_bitwise_int_ops_match_numpy(self):
        a, b = _iarr(seed=1), _iarr(seed=2)
        np.testing.assert_array_equal((_mk(a) & _mk(b)).get_array(), a & b)
        np.testing.assert_array_equal((_mk(a) | _mk(b)).get_array(), a | b)
        np.testing.assert_array_equal((_mk(a) ^ _mk(b)).get_array(), a ^ b)
        np.testing.assert_array_equal((~_mk(a)).get_array(), ~a)

    def test_bitwise_on_float_raises(self):
        f = _mk(_farr(seed=3))
        with pytest.raises(TypeError):
            _ = f & f
        with pytest.raises(TypeError):
            _ = f | f
        with pytest.raises(TypeError):
            _ = f ^ f
        with pytest.raises(TypeError):
            _ = ~f


class Test_Math_InplaceOperators(unittest.TestCase):
    @staticmethod
    def _full(value: float) -> NII:
        return _mk(np.full((2, 2, 2), value, dtype=float))

    def test_iadd(self):
        n = self._full(8.0)
        n += 2
        np.testing.assert_allclose(n.get_array(), 10.0)

    def test_isub(self):
        n = self._full(8.0)
        n -= 3
        np.testing.assert_allclose(n.get_array(), 5.0)

    def test_imul(self):
        n = self._full(8.0)
        n *= 2
        np.testing.assert_allclose(n.get_array(), 16.0)

    def test_itruediv(self):
        n = self._full(8.0)
        n /= 2
        np.testing.assert_allclose(n.get_array(), 4.0)

    def test_ifloordiv(self):
        n = self._full(9.0)
        n //= 2
        np.testing.assert_allclose(n.get_array(), 4.0)

    def test_imod(self):
        n = self._full(9.0)
        n %= 2
        np.testing.assert_allclose(n.get_array(), 1.0)

    def test_ipow(self):
        n = self._full(2.0)
        n **= 3
        np.testing.assert_allclose(n.get_array(), 8.0)


class Test_Math_Unary(unittest.TestCase):
    def test_neg_pos_abs(self):
        a = _farr(seed=1)
        np.testing.assert_allclose((-_mk(a)).get_array(), -a)
        np.testing.assert_allclose((+_mk(a)).get_array(), +a)
        np.testing.assert_allclose(abs(_mk(a)).get_array(), np.abs(a))

    def test_round_dunder_and_method(self):
        a = _farr(seed=2)
        np.testing.assert_allclose(round(_mk(a), 2).get_array(), np.round(a, 2))
        np.testing.assert_allclose(_mk(a).round(2).get_array(), np.round(a, 2))

    def test_floor_ceil(self):
        a = _farr(seed=3)
        np.testing.assert_allclose(math.floor(_mk(a)).get_array(), np.floor(a))
        np.testing.assert_allclose(math.ceil(_mk(a)).get_array(), np.ceil(a))


class Test_Math_Reductions(unittest.TestCase):
    def setUp(self):
        self.arr = np.arange(27, dtype=float).reshape(3, 3, 3)
        self.nii = _mk(self.arr)

    def test_max_min(self):
        self.assertEqual(self.nii.max(), self.arr.max())
        self.assertEqual(self.nii.min(), self.arr.min())

    def test_sum_mean_median_std(self):
        self.assertTrue(np.isclose(self.nii.sum(), self.arr.sum()))
        self.assertTrue(np.isclose(self.nii.mean(), self.arr.mean()))
        self.assertTrue(np.isclose(self.nii.median(), np.median(self.arr)))
        self.assertTrue(np.isclose(self.nii.std(), self.arr.std()))

    def test_sum_mean_with_nii_mask(self):
        mask = np.zeros((3, 3, 3), np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1, 1] = 1
        mask[2, 2, 2] = 1
        m = NII.from_numpy(mask, np.eye(4), seg=True)
        sel = mask.astype(bool)
        self.assertTrue(np.isclose(self.nii.sum(where=m), self.arr[sel].sum()))
        self.assertTrue(np.isclose(self.nii.mean(where=m), self.arr[sel].mean()))
        self.assertTrue(np.isclose(self.nii.std(where=m), self.arr[sel].std()))


class Test_Math_Clamp(unittest.TestCase):
    def test_clamp_both_bounds(self):
        a = np.array([[[0.0, 5.0, 10.0]]])
        out = _mk(a).clamp(min=2, max=8).get_array().ravel()
        np.testing.assert_allclose(out, [2, 5, 8])

    def test_clamp_only_min(self):
        a = np.array([[[-3.0, 1.0, 4.0]]])
        out = _mk(a).clamp(min=0).get_array().ravel()
        np.testing.assert_allclose(out, [0, 1, 4])

    def test_clamp_inplace(self):
        a = np.array([[[0.0, 5.0, 10.0]]])
        n = _mk(a)
        n.clamp_(min=2, max=8)
        np.testing.assert_allclose(n.get_array().ravel(), [2, 5, 8])


class Test_Math_Normalize(unittest.TestCase):
    def test_normalize_ct_range(self):
        out = get_test_ct()[0].normalize_ct()
        self.assertAlmostEqual(float(out.min()), 0.0)
        self.assertAlmostEqual(float(out.max()), 1.0)

    def test_normalize_mri_range(self):
        out = get_test_mri()[0].normalize_mri()
        self.assertAlmostEqual(float(out.min()), 0.0)
        self.assertAlmostEqual(float(out.max()), 1.0)

    def test_normalize_default_range(self):
        out = get_test_mri()[0].normalize()
        self.assertAlmostEqual(float(out.min()), 0.0)
        self.assertAlmostEqual(float(out.max()), 1.0)

    def test_normalize_out_of_place_does_not_mutate(self):
        mri = get_test_mri()[0]
        before = mri.get_array().copy()
        mri.normalize()
        np.testing.assert_array_equal(mri.get_array(), before)

    def test_normalize_inplace(self):
        mri = get_test_mri()[0]
        mri.normalize_()
        self.assertAlmostEqual(float(mri.max()), 1.0)


class Test_Math_ThresholdNan(unittest.TestCase):
    def test_threshold_binarises(self):
        a = np.array([[[0.0, 0.3, 0.6, 1.0]]])
        out = _mk(a).threshold(0.5)
        np.testing.assert_array_equal(out.get_array().ravel(), [0, 0, 1, 1])
        self.assertTrue(out.seg)

    def test_nan_to_num(self):
        a = np.array([[[1.0, np.nan, 3.0]]])
        out = _mk(a).nan_to_num(num=-1)
        np.testing.assert_array_equal(out.get_array().ravel(), [1, -1, 3])


class Test_Math_Metrics(unittest.TestCase):
    @staticmethod
    def _posf(shape=(20, 20, 20), seed=0) -> NII:
        return NII.from_numpy(np.random.default_rng(seed).random(shape).astype(np.float32), np.eye(4), seg=False)

    @staticmethod
    def _seg(arr: np.ndarray) -> NII:
        return NII.from_numpy(arr, np.eye(4), seg=True)

    def test_ssim_identical_is_one(self):
        n = self._posf(seed=1)
        self.assertAlmostEqual(n.ssim(n.copy()), 1.0, places=5)

    def test_ssim_in_range(self):
        v = self._posf(seed=1).ssim(self._posf(seed=2))
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)

    def test_psnr_finite_positive(self):
        a = self._posf(seed=1)
        noisy = a.get_array() + 0.05 * np.random.default_rng(3).random((20, 20, 20)).astype(np.float32)
        v = a.psnr(NII.from_numpy(noisy, np.eye(4), seg=False))
        self.assertTrue(np.isfinite(v))
        self.assertGreater(v, 0)

    def test_dice_identical_is_one(self):
        arr = np.zeros((16, 16, 16), np.uint8)
        arr[2:8, 2:8, 2:8] = 1
        arr[10:14, 10:14, 10:14] = 2
        s = self._seg(arr)
        d = s.dice(s.copy(), bar=False)
        self.assertAlmostEqual(d[1], 1.0)
        self.assertAlmostEqual(d[2], 1.0)
        # also exercise the tqdm progress-bar branch
        self.assertAlmostEqual(s.dice(s.copy(), bar=True)[1], 1.0)

    def test_dice_partial_overlap(self):
        a = np.zeros((16, 16, 16), np.uint8)
        a[2:10, 2:10, 2:10] = 1
        b = np.zeros((16, 16, 16), np.uint8)
        b[6:14, 2:10, 2:10] = 1
        d = self._seg(a).dice(self._seg(b), bar=False)
        self.assertGreater(d[1], 0.0)
        self.assertLess(d[1], 1.0)

    def test_betti_numbers_solid_blob(self):
        arr = np.zeros((20, 20, 20), np.uint8)
        arr[5:15, 5:15, 5:15] = 1
        b = self._seg(arr).betti_numbers(verbose=True)
        self.assertEqual(b[1][0], 1)  # exactly one connected component


if __name__ == "__main__":
    unittest.main()
