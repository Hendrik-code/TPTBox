"""Unit tests for `corner_bisection_search`, the search behind the corner POIs (`strategy_find_corner`).

They run against whichever implementation is active: the Python original, or the Rust port when it
was built into `.fast/lib` (see `benchmarks/fast_anything/README.md`).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from TPTBox.core.poi_fun.strategies import corner_bisection_search


def _box(shape=(12, 12, 12), hi=10) -> np.ndarray:
    region = np.zeros(shape, np.uint8)
    region[:hi, :hi, :hi] = 1
    return region


START = np.array([5.0, 5.0, 5.0])
V1 = np.array([10.0, 0.0, 0.0])
V2 = np.array([0.0, 10.0, 0.0])


class TestCornerBisectionSearch(unittest.TestCase):
    def test_ends_on_the_rounded_corner_of_a_box(self):
        # voxels 0..9 are inside; near the x/y corner the trilinear value is (10 - x) * (10 - y),
        # so the 0.5 iso-line is that hyperbola, not the square corner at (9.5, 9.5)
        out = corner_bisection_search(_box(), START, V1, V2, 5e-8)
        assert out is not None
        self.assertAlmostEqual((10 - out[0]) * (10 - out[1]), 0.5, delta=1e-6)
        self.assertTrue(9.0 < out[0] < 9.5 and 9.0 < out[1] < 9.5, out)
        self.assertEqual(out[2], 5.0)

    def test_any_segmentation_label_dtype_gives_the_same_corner(self):
        out = corner_bisection_search(_box(), START, V1, V2, 5e-8)
        for dtype in (np.uint16, np.int32, np.float32):
            with self.subTest(dtype=dtype.__name__):
                cast = np.asarray(_box().astype(dtype), dtype=np.uint8)
                np.testing.assert_array_equal(corner_bisection_search(cast, START, V1, V2, 5e-8), out)

    def test_start_outside_the_mask_returns_none(self):
        # used to loop forever: the shrinking factors get stuck at the smallest subnormal
        cases = {
            "start outside": (_box(), np.array([11.0, 11.0, 11.0])),
            "empty axis": (_box()[:0], START),
            "infinite start": (_box(), np.array([math.inf, 5.0, 5.0])),
        }
        for name, (region, start) in cases.items():
            with self.subTest(name):
                self.assertIsNone(corner_bisection_search(region, start, V1, V2, 5e-8))

    def test_tolerance_below_float_resolution_terminates(self):
        # used to loop forever once the steps were absorbed by rounding
        ref = corner_bisection_search(_box(), START, V1, V2, 5e-8)
        for delta in (0.0, -0.0, 5e-324, -math.inf):
            with self.subTest(delta=delta):
                np.testing.assert_allclose(corner_bisection_search(_box(), START, V1, V2, delta), ref, atol=1e-5)

    def test_nan_coordinate_raises(self):
        with self.assertRaises(ValueError):
            corner_bisection_search(_box(), np.array([math.nan, 5.0, 5.0]), V1, V2, 5e-8)


if __name__ == "__main__":
    unittest.main()
