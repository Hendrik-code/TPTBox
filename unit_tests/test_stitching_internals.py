"""Unit tests for the pure geometry helpers in `TPTBox.stitching`.

`unit_tests/test_stiching.py` drives `stitching_raw` end to end; this file pins the
affine/bbox/dtype helpers it is built on, where the expected values are analytic.
"""

from __future__ import annotations

import unittest

import numpy as np

from TPTBox.stitching.stitching_tools import _center_frontal


class TestCenterFrontal(unittest.TestCase):
    """`_center_frontal` cuts a ~256-voxel dorsal window out of a stitched sagittal volume."""

    def test_slice_is_always_inside_the_volume(self):
        # Regression: `middle - 122` is negative for any volume below ~258 voxels, and numpy
        # reads a negative start as an offset from the end -> the crop silently took the
        # wrong side of the image (size=100 used to yield slice(-75, 181)).
        for size in (1, 10, 50, 100, 200, 256, 257, 258, 300, 600):
            with self.subTest(size=size):
                s = _center_frontal(size)
                self.assertGreaterEqual(s.start, 0)
                self.assertGreater(s.stop, s.start)
                self.assertLessEqual(s.stop, size)

    def test_crop_keeps_voxels_for_small_volumes(self):
        for size in (50, 100, 200):
            with self.subTest(size=size):
                arr = np.arange(size)
                kept = arr[_center_frontal(size)]
                self.assertGreater(kept.size, 0)
                # The window starts at the volume border for anything this small.
                self.assertEqual(kept[0], 0)

    def test_large_volume_window_is_unchanged(self):
        # The empirically tuned behaviour for full-size NAKO volumes must not move.
        self.assertEqual(_center_frontal(600), slice(163, 419))
        self.assertEqual(_center_frontal(600).stop - _center_frontal(600).start, 256)


if __name__ == "__main__":
    unittest.main()
