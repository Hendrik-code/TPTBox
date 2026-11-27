"""Unit tests for the array kernels behind np_unique, connected components, distance transforms,
morphology and resampling (`unique_integers`, `connected_component_labels`,
`euclidean_distance_transform`, `binary_dilation_perimeter`, `binary_erosion_radius1`,
`affine_resample`).

They run against whichever implementation is active: the Python originals, or the Rust ports when
they were built into `.fast/lib` (see `benchmarks/fast_anything/README.md`). Most cases compare with
the library the Python original calls (numpy, cc3d, scipy), with the same inputs.
"""

from __future__ import annotations

import itertools
import unittest

import cc3d
import numpy as np
from scipy import ndimage

from TPTBox.core.internal.nii_help import affine_resample
from TPTBox.core.np_utils import (
    binary_dilation_perimeter,
    binary_erosion_radius1,
    connected_component_labels,
    euclidean_distance_transform,
    np_unique,
    np_unique_withoutzero,
    unique_integers,
)

INT_DTYPES = (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)


def _blobs(rng: np.random.Generator, shape, labels=3, density=0.3) -> np.ndarray:
    """Label map with blob-like regions: thresholded smoothed noise, labelled by octant of the noise value."""
    noise = ndimage.gaussian_filter(rng.random(shape), 1.0)
    out = np.zeros(shape, np.uint8)
    cut = np.quantile(noise, 1 - density)
    out[noise > cut] = 1 + (rng.integers(0, labels, size=shape)[noise > cut] if labels > 1 else 0)
    return out


def _assert_same_array(test: unittest.TestCase, got: np.ndarray, expected: np.ndarray) -> None:
    test.assertEqual(got.dtype, expected.dtype)
    test.assertEqual(got.shape, expected.shape)
    test.assertEqual(got.tobytes(), expected.tobytes())  # bit for bit, NaN and -0.0 included


class TestUniqueIntegers(unittest.TestCase):
    def test_matches_numpy_unique_for_every_integer_dtype(self):
        rng = np.random.default_rng(0)
        for dtype in INT_DTYPES:
            info = np.iinfo(dtype)
            cases = {
                "labels": rng.integers(0, min(info.max, 60) + 1, size=(9, 7, 5)).astype(dtype),
                "full range": rng.integers(info.min, info.max, size=(40, 3), dtype=dtype, endpoint=True),
                "extremes": np.array([info.max, info.min, 0, info.max], dtype=dtype),
            }
            for name, arr in cases.items():
                for variant, a in {
                    "c": arr,
                    "fortran": np.asfortranarray(arr),
                    "strided": arr[::2],
                    "byteswapped": arr.astype(arr.dtype.newbyteorder(">")),
                }.items():
                    with self.subTest(dtype=np.dtype(dtype).name, case=name, variant=variant):
                        got = unique_integers(a)
                        self.assertEqual(got, np.unique(a).tolist())
                        self.assertTrue(all(type(v) is int for v in got))

    def test_empty_arrays(self):
        # the unsigned path calls arr.max(), which has no identity on an empty array
        for dtype in INT_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                empty = np.zeros((0, 4), dtype)
                if np.dtype(dtype).kind == "u":
                    with self.assertRaises(ValueError):
                        unique_integers(empty)
                else:
                    self.assertEqual(unique_integers(empty), [])

    def test_non_integer_raises_and_np_unique_handles_it(self):
        for arr in (np.array([True, False]), np.array([0.5, 0.5, 2.0])):
            with self.subTest(dtype=arr.dtype.name):
                with self.assertRaises(TypeError):
                    unique_integers(arr)
                self.assertEqual(np_unique(arr), np.unique(arr).tolist())

    def test_np_unique_withoutzero(self):
        arr = np.array([[0, 5, 3], [3, 0, 7]], np.uint16)
        self.assertEqual(np_unique(arr), [0, 3, 5, 7])
        self.assertEqual(np_unique_withoutzero(arr), [3, 5, 7])
        self.assertEqual(np_unique_withoutzero(np.zeros(4, np.uint8)), [])


class TestConnectedComponentLabels(unittest.TestCase):
    def test_matches_cc3d(self):
        rng = np.random.default_rng(1)
        for (ndim, connectivity), labels in itertools.product([(2, 4), (2, 8), (2, 6), (2, 26), (3, 6), (3, 18), (3, 26)], (1, 3)):
            shape = (23, 31) if ndim == 2 else (13, 17, 11)
            arr = _blobs(rng, shape, labels=labels)
            for variant, a in {
                "c": arr,
                "fortran": np.asfortranarray(arr),
                "strided": arr[:, ::2],
                "bool": arr > 0,
                "int64": arr.astype(np.int64),
                "float32": arr.astype(np.float32),
            }.items():
                with self.subTest(connectivity=connectivity, labels=labels, variant=variant):
                    got, n = connected_component_labels(a, connectivity)
                    contiguous = a if a.flags.c_contiguous or a.flags.f_contiguous else np.ascontiguousarray(a)
                    expected, n_expected = cc3d.connected_components(contiguous, connectivity=connectivity, return_N=True)
                    _assert_same_array(self, got, expected)
                    self.assertEqual(
                        (got.flags.c_contiguous, got.flags.f_contiguous), (expected.flags.c_contiguous, expected.flags.f_contiguous)
                    )
                    self.assertEqual(n, n_expected)

    def test_numbering_follows_the_raster_scan(self):
        arr = np.zeros((4, 5, 6), np.uint8)
        arr[3, 4, 5] = arr[0, 0, 1] = 2
        arr[1, 0, 0] = 3  # touches (0, 0, 1) and (1, 1, 1), but neighbours with other values stay apart
        arr[1, 1, 1] = 7
        out, n = connected_component_labels(arr, 26)
        self.assertEqual(n, 4)
        self.assertEqual([out[0, 0, 1], out[1, 0, 0], out[1, 1, 1], out[3, 4, 5]], [1, 2, 3, 4])

    def test_connectivity_decides_diagonal_contacts(self):
        arr = np.zeros((3, 3, 3), np.uint8)
        arr[0, 0, 0] = arr[1, 1, 0] = arr[2, 2, 1] = 1  # edge contact, then corner contact
        self.assertEqual([connected_component_labels(arr, c)[1] for c in (6, 18, 26)], [3, 2, 1])

    def test_nan_voxels_never_connect(self):
        arr = np.array([[np.nan, np.nan, 1.0, 1.0, -0.0, 2.0]], np.float64)
        out, n = connected_component_labels(arr, 8)
        self.assertEqual(out.tolist(), [[1, 2, 3, 3, 0, 4]])
        self.assertEqual(n, 4)

    def test_output_dtype_follows_the_run_estimate(self):
        runs = np.zeros((65535, 1, 2), np.uint8)
        runs[:, 0, 0] = 1  # 65535 isolated runs
        self.assertEqual(connected_component_labels(runs, 6)[0].dtype, np.uint32)
        self.assertEqual(connected_component_labels(runs[:65534], 6)[0].dtype, np.uint16)

    def test_strided_view_is_labelled_like_a_copy(self):
        # cc3d itself raises IndexError on this view: it locates the foreground in the buffer behind
        # the view (where index 5 is set) instead of the view
        view = np.array([[1, 0, 0, 0, 0, 1, 0, 1, 0, 0]], np.uint8)[:, ::2]
        out, n = connected_component_labels(view, 8)
        self.assertEqual((out.tolist(), n), ([[1, 0, 0, 0, 0]], 1))

    def test_empty_input_comes_back_flat(self):
        out, n = connected_component_labels(np.zeros((2, 0, 3), np.int16), 26)
        self.assertEqual((out.shape, out.dtype, n), ((0,), np.dtype(np.int16), 0))

    def test_invalid_inputs(self):
        cases = {
            "float16": (np.zeros((2, 2), np.float16), 4, TypeError),
            "1D": (np.zeros(5, np.uint8), 4, ValueError),
            "3D with 4": (np.zeros((2, 2, 2), np.uint8), 4, ValueError),
            "2D with 0": (np.zeros((2, 2), np.uint8), 0, ValueError),
        }
        for name, (arr, connectivity, error) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                connected_component_labels(arr, connectivity)


class TestEuclideanDistanceTransform(unittest.TestCase):
    def test_matches_scipy_including_ties_and_spacing(self):
        rng = np.random.default_rng(2)
        for trial in range(60):
            ndim = 1 + trial % 3
            shape = tuple(int(s) for s in rng.integers(1, 12, ndim))
            mask = rng.random(shape) < (0.5, 0.8, 0.97)[trial % 3]  # dense masks give many ties
            sampling = None if trial % 2 else tuple(float(s) for s in rng.choice([0.5, 0.7, 1.0, 1.3, 1.5, 2.0], ndim))
            with self.subTest(shape=shape, sampling=sampling):
                dist, idx = euclidean_distance_transform(mask, sampling=sampling, return_indices=True)
                ref_dist, ref_idx = ndimage.distance_transform_edt(mask, sampling=sampling, return_indices=True)
                _assert_same_array(self, dist, ref_dist)
                _assert_same_array(self, idx, ref_idx)

    def test_requested_outputs_only(self):
        mask = np.ones((5, 4), bool)
        mask[2, 1] = False
        dist, idx = euclidean_distance_transform(mask)
        self.assertIsNone(idx)
        self.assertEqual(dist[0, 0], np.sqrt(5.0))
        dist, idx = euclidean_distance_transform(mask, return_distances=False, return_indices=True)
        self.assertIsNone(dist)
        self.assertEqual(idx[:, 0, 3].tolist(), [2, 1])
        with self.assertRaises(RuntimeError):
            euclidean_distance_transform(mask, return_distances=False)

    def test_mask_without_background_matches_scipy(self):
        for shape in ((4,), (3, 4), (2, 3, 4)):
            with self.subTest(shape=shape):
                mask = np.ones(shape, bool)
                dist, idx = euclidean_distance_transform(mask, return_indices=True)
                ref_dist, ref_idx = ndimage.distance_transform_edt(mask, return_indices=True)
                _assert_same_array(self, dist, ref_dist)
                _assert_same_array(self, idx, ref_idx)

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            euclidean_distance_transform(np.zeros((3, 3), np.uint8))
        with self.assertRaises(RuntimeError):
            euclidean_distance_transform(np.zeros((3, 3), bool), sampling=(1.0, 1.0, 1.0))


class TestBinaryMorphology(unittest.TestCase):
    def test_dilation_matches_scipy_for_generated_structures(self):
        # stamping only perimeter voxels equals a full dilation for these structuring elements
        rng = np.random.default_rng(3)
        for ndim, connectivity in itertools.product((1, 2, 3), (1, 2, 3)):
            if connectivity > ndim:
                continue
            struct = ndimage.generate_binary_structure(ndim, connectivity)
            shape = ((40,), (17, 23), (11, 13, 9))[ndim - 1]
            mask = _blobs(rng, shape, labels=1, density=0.4) > 0
            with self.subTest(ndim=ndim, connectivity=connectivity):
                _assert_same_array(self, binary_dilation_perimeter(mask, struct), ndimage.binary_dilation(mask, struct))

    def test_dilation_stamps_perimeter_voxels_only(self):
        # the element reaches two voxels down without the voxel in between, so only the block's
        # interior voxel (3, 3) could set (5, 3), and interior voxels are not stamped
        selem = np.zeros((5, 5), bool)
        selem[2, 2] = selem[4, 2] = True
        mask = np.zeros((7, 7), bool)
        mask[2:5, 2:5] = True
        out = binary_dilation_perimeter(mask, selem)
        expected = mask.copy()
        expected[5:7, 2:5] = True
        expected[5, 3] = False
        np.testing.assert_array_equal(out, expected)

    def test_dilation_invalid_inputs(self):
        cases = {
            "even element": (np.zeros((4, 4), bool), np.ones((2, 3), bool), ValueError),
            "dimension mismatch": (np.zeros((4, 4), bool), np.ones(3, bool), ValueError),
            "4D": (np.zeros((2, 2, 2, 2), bool), np.ones((3, 3, 3, 3), bool), RuntimeError),
            "not boolean": (np.zeros((4, 4), np.uint8), np.ones((3, 3), bool), TypeError),
        }
        for name, (image, selem, error) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                binary_dilation_perimeter(image, selem)

    def test_erosion_matches_scipy(self):
        rng = np.random.default_rng(4)
        for trial in range(80):
            ndim = 1 + trial % 3
            shape = tuple(int(s) for s in rng.integers(1, 14, ndim))
            image = rng.random(shape) < 0.85
            struct = rng.random(tuple(int(s) for s in rng.choice([1, 3], ndim))) < 0.7
            struct[tuple(s // 2 for s in struct.shape)] = True
            iterations, border_value = int(rng.choice([1, 2, 3, 0])), int(rng.choice([0, 1]))
            with self.subTest(shape=shape, struct=struct.shape, iterations=iterations, border_value=border_value):
                got = binary_erosion_radius1(image, struct, iterations, border_value)
                _assert_same_array(self, got, ndimage.binary_erosion(image, struct, iterations=iterations, border_value=border_value))

    def test_erosion_invalid_inputs(self):
        cases = {
            "wide element": (np.ones((6, 6), bool), np.ones((5, 5), bool), ValueError),
            "dimension mismatch": (np.ones((6, 6), bool), np.ones(3, bool), RuntimeError),
            "not boolean": (np.ones((6, 6), np.uint8), np.ones((3, 3), bool), TypeError),
        }
        for name, (image, struct, error) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                binary_erosion_radius1(image, struct, 1, 0)


class TestAffineResample(unittest.TestCase):
    def test_matches_scipy(self):
        rng = np.random.default_rng(5)
        for trial in range(48):
            order, mode = trial % 2, ("nearest", "constant")[(trial // 2) % 2]
            dtype = (np.uint8, np.int16, np.float32, np.float64, np.uint16, np.int32)[trial % 6]
            arr = (rng.random((9, 8, 7)) * 200 - (0 if np.dtype(dtype).kind == "u" else 100)).astype(dtype)
            rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            matrix = rotation * rng.uniform(0.5, 2.0, 3)
            offset = rng.uniform(-3, 3, 3) + np.array(arr.shape) / 2 - matrix @ (np.array((10, 9, 8)) / 2)
            output_dtype = (None, "float32", "uint8", "int16")[trial % 4]
            with self.subTest(order=order, mode=mode, dtype=np.dtype(dtype).name, output_dtype=output_dtype):
                got = affine_resample(arr, matrix, offset, (10, 9, 8), order, mode, -5.0, output_dtype)
                ref = ndimage.affine_transform(
                    arr,
                    matrix,
                    offset,
                    (10, 9, 8),
                    order=order,
                    mode=mode,
                    cval=-5.0,
                    output=None if output_dtype is None else np.dtype(output_dtype),
                )
                _assert_same_array(self, got, ref)

    def test_integer_output_rounds_half_away_from_zero_and_saturates(self):
        arr = np.array([0.0, 3.0, -3.0, 600.0])
        out = affine_resample(arr, np.zeros((1, 1)), np.array([0.5]), (1,), 1, "nearest", 0.0, "int16")
        self.assertEqual(out.tolist(), [2])  # 1.5 rounds to 2
        out = affine_resample(arr, np.zeros((1, 1)), np.array([1.5]), (1,), 1, "nearest", 0.0, "int16")
        self.assertEqual(out.tolist(), [0])
        out = affine_resample(arr, np.zeros((1, 1)), np.array([3.0]), (1,), 0, "nearest", 0.0, "uint8")
        self.assertEqual(out.tolist(), [255])

    def test_invalid_inputs(self):
        arr, eye, zero = np.zeros((4, 4)), np.eye(2), np.zeros(2)
        cases = {
            "order 3": ((arr, eye, zero, (4, 4), 3, "nearest", 0.0, None), ValueError),
            "mode reflect": ((arr, eye, zero, (4, 4), 0, "reflect", 0.0, None), ValueError),
            "float16": ((arr.astype(np.float16), eye, zero, (4, 4), 0, "nearest", 0.0, None), TypeError),
            "float32 matrix": ((arr, eye.astype(np.float32), zero, (4, 4), 0, "nearest", 0.0, None), TypeError),
            "rank mismatch": ((arr, eye, zero, (4, 4, 4), 0, "nearest", 0.0, None), ValueError),
            "empty nearest": ((arr[:0], eye, zero, (4, 4), 0, "nearest", 0.0, None), ValueError),
        }
        for name, (args, error) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                affine_resample(*args)


if __name__ == "__main__":
    unittest.main()
