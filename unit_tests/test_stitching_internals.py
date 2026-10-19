"""Unit tests for the pure geometry helpers in `TPTBox.stitching`.

`unit_tests/test_stiching.py` drives `stitching_raw` end to end; this file pins the
affine/bbox/dtype helpers it is built on, where the expected values are analytic.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from TPTBox import NII
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


class TestAffineDecomposition(unittest.TestCase):
    """`get_rotation_and_spacing_from_affine` / `get_ras_affine` are inverses."""

    @staticmethod
    def _affines():
        import math  # noqa: PLC0415

        diag = np.diag([1.5, 2.0, 3.0, 1.0])
        diag[:3, 3] = [10.0, -5.0, 2.0]

        flipped = np.diag([-1.0, -2.0, 3.0, 1.0])
        flipped[:3, 3] = [1.0, 2.0, 3.0]

        def rot(axis, deg):
            t = math.radians(deg)
            c, s = math.cos(t), math.sin(t)
            if axis == "x":
                return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            if axis == "y":
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        turned = np.eye(4)
        turned[:3, :3] = rot("x", 13) @ rot("y", -7) @ rot("z", 21) @ np.diag([0.8, 1.2, 4.0])
        turned[:3, 3] = [3.0, -4.0, 5.0]
        return {"diagonal": diag, "double_flip": flipped, "rotated_anisotropic": turned}

    def test_decomposition_invariants(self):
        from TPTBox.stitching.stitching import get_rotation_and_spacing_from_affine  # noqa: PLC0415

        for name, affine in self._affines().items():
            with self.subTest(affine=name):
                rotation, spacing = get_rotation_and_spacing_from_affine(affine)
                self.assertTrue((spacing >= 0).all(), "spacing is a norm, so never negative")
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
                np.testing.assert_allclose(rotation @ np.diag(spacing), affine[:3, :3], atol=1e-9)

    def test_round_trip_reproduces_the_affine(self):
        from TPTBox.stitching.stitching import get_ras_affine, get_rotation_and_spacing_from_affine  # noqa: PLC0415

        for name, affine in self._affines().items():
            with self.subTest(affine=name):
                rotation, spacing = get_rotation_and_spacing_from_affine(affine)
                # `origin` is the translation expressed in the rotated frame, not the
                # world-space origin - which is what makes this round-trip exact.
                origin = rotation.T @ affine[:3, 3]
                np.testing.assert_allclose(get_ras_affine(rotation, spacing, origin), affine, atol=1e-9)

    def test_spacing_matches_the_column_norms(self):
        from TPTBox.stitching.stitching import get_rotation_and_spacing_from_affine  # noqa: PLC0415

        affine = self._affines()["rotated_anisotropic"]
        _, spacing = get_rotation_and_spacing_from_affine(affine)
        np.testing.assert_allclose(spacing, np.linalg.norm(affine[:3, :3], axis=0), atol=1e-9)


class TestCornerPoints(unittest.TestCase):
    def test_corners_are_the_box_corners_offset_by_one_voxel(self):
        from TPTBox.stitching.stitching import get_all_corner_points  # noqa: PLC0415

        corners = np.asarray(get_all_corner_points(np.eye(4), (4, 5, 6)))
        self.assertEqual(corners.shape, (8, 3))
        # Pinned, not endorsed: the implementation adds 1 to every index before
        # applying the affine, so the box spans 1..shape+1 instead of 0..shape.
        # See the PR description - the stitched FOV depends on it.
        np.testing.assert_allclose(corners.min(axis=0), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(corners.max(axis=0), [5.0, 6.0, 7.0])

    def test_corners_follow_the_affine(self):
        from TPTBox.stitching.stitching import get_all_corner_points  # noqa: PLC0415

        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [100.0, 0.0, 0.0]
        corners = np.asarray(get_all_corner_points(affine, (4, 4, 4)))
        self.assertAlmostEqual(float(corners[:, 0].min()), 102.0)
        self.assertEqual(len({tuple(c) for c in corners}), 8, "all eight corners are distinct")


class TestOccupancyBoxes(unittest.TestCase):
    def test_bbox_of_a_known_block(self):
        from TPTBox.stitching.stitching import _occupancy_bbox  # noqa: PLC0415

        occ = np.zeros((10, 10, 10), dtype=np.uint8)
        occ[2:5, 3:7, 1:2] = 1
        lo, hi = _occupancy_bbox(occ)
        np.testing.assert_array_equal(lo, [2, 3, 1])
        np.testing.assert_array_equal(hi, [4, 6, 1], "hi is inclusive")

    def test_empty_occupancy_is_none(self):
        from TPTBox.stitching.stitching import _occupancy_bbox  # noqa: PLC0415

        self.assertIsNone(_occupancy_bbox(np.zeros((5, 5, 5), dtype=np.uint8)))

    def test_aabb_overlaps(self):
        from TPTBox.stitching.stitching import _aabb_overlaps  # noqa: PLC0415

        box = (np.array([0, 0, 0]), np.array([4, 4, 4]))
        cases = {
            "identical": ((np.array([0, 0, 0]), np.array([4, 4, 4])), True),
            "touching": ((np.array([4, 4, 4]), np.array([8, 8, 8])), True),
            "disjoint": ((np.array([5, 5, 5]), np.array([8, 8, 8])), False),
            "contained": ((np.array([1, 1, 1]), np.array([2, 2, 2])), True),
            "empty": (None, False),
        }
        for name, (other, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_aabb_overlaps(box, other), expected)
                self.assertEqual(_aabb_overlaps(other, box), expected, "the relation is symmetric")


class TestMaxAffineAndShape(unittest.TestCase):
    def test_output_grid_covers_every_input_corner(self):
        from TPTBox.stitching.stitching import get_all_corner_points, get_max_affine_and_shape  # noqa: PLC0415

        a = np.diag([1.0, 1.0, 1.0, 1.0])
        b = np.diag([1.0, 1.0, 1.0, 1.0])
        b[:3, 3] = [10.0, 0.0, 0.0]
        points = np.concatenate([get_all_corner_points(a, (8, 8, 8)), get_all_corner_points(b, (8, 8, 8))])
        out = get_max_affine_and_shape(points, [a, b])
        self.assertIsInstance(out, NII)
        self.assertGreaterEqual(out.shape[0], 18, "the union spans both volumes along x")

    def test_min_spacing_clamps_the_output_zoom(self):
        from TPTBox.stitching.stitching import get_all_corner_points, get_max_affine_and_shape  # noqa: PLC0415

        a = np.diag([0.5, 0.5, 0.5, 1.0])
        points = get_all_corner_points(a, (8, 8, 8))
        fine = get_max_affine_and_shape(points, [a])
        coarse = get_max_affine_and_shape(points, [a], min_spacing=3.0)
        self.assertAlmostEqual(min(fine.zoom), 0.5, delta=1e-6)
        self.assertGreaterEqual(min(coarse.zoom), 3.0 - 1e-6)
        self.assertLess(max(coarse.shape), max(fine.shape), "a coarser grid needs fewer voxels")

    def test_the_finest_input_spacing_wins(self):
        from TPTBox.stitching.stitching import get_all_corner_points, get_max_affine_and_shape  # noqa: PLC0415

        fine = np.diag([0.5, 0.5, 0.5, 1.0])
        coarse = np.diag([3.0, 3.0, 3.0, 1.0])
        points = np.concatenate([get_all_corner_points(fine, (8, 8, 8)), get_all_corner_points(coarse, (4, 4, 4))])
        out = get_max_affine_and_shape(points, [fine, coarse])
        np.testing.assert_allclose(out.zoom, (0.5, 0.5, 0.5), atol=1e-6)


class TestAutoOutputDtype(unittest.TestCase):
    @staticmethod
    def _nii(dtype, slope=None):
        import nibabel as nib  # noqa: PLC0415

        from TPTBox import NII  # noqa: PLC0415

        img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=dtype), np.eye(4))
        if slope is not None:
            img.header.set_slope_inter(slope, 0.0)
        return NII(img, seg=False)

    def test_all_uint8_stays_uint8(self):
        from TPTBox.stitching.stitching import _auto_output_dtype  # noqa: PLC0415

        self.assertEqual(np.dtype(_auto_output_dtype([self._nii(np.uint8), self._nii(np.uint8)])), np.dtype(np.uint8))

    def test_mixed_integer_widths_take_the_wider_one(self):
        from TPTBox.stitching.stitching import _auto_output_dtype  # noqa: PLC0415

        out = np.dtype(_auto_output_dtype([self._nii(np.uint8), self._nii(np.uint16)]))
        self.assertGreaterEqual(out.itemsize, np.dtype(np.uint16).itemsize)

    def test_a_float_input_forces_float32(self):
        from TPTBox.stitching.stitching import _auto_output_dtype  # noqa: PLC0415

        self.assertEqual(np.dtype(_auto_output_dtype([self._nii(np.uint8), self._nii(np.float32)])), np.dtype(np.float32))

    def test_a_non_trivial_slope_forces_float32(self):
        import nibabel as nib  # noqa: PLC0415

        from TPTBox import NII  # noqa: PLC0415
        from TPTBox.stitching.stitching import _auto_output_dtype  # noqa: PLC0415

        # The Philips case the docstring is about: int16 storage with a large slope.
        # The slope is read off `nii.dataobj`, which only carries it for a file-backed
        # image, so the fixture has to go through disk.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slope.nii.gz"
            img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))
            img.header.set_slope_inter(641.0, 0.0)
            nib.save(img, path)
            nii = NII.load(path, seg=False)
            self.assertEqual(np.dtype(_auto_output_dtype([nii])), np.dtype(np.float32))


class TestStitchingTools(unittest.TestCase):
    def test_crop_borders_cuts_the_requested_slices(self):
        import nibabel as nib  # noqa: PLC0415

        from TPTBox import NII  # noqa: PLC0415
        from TPTBox.stitching.stitching_tools import _crop_borders  # noqa: PLC0415

        nii = NII(nib.Nifti1Image(np.ones((20, 20, 20), dtype=np.uint16), np.eye(4)), seg=False)
        cut = {"HWS": (slice(None), slice(2, 18), slice(None))}
        out = _crop_borders(nii, "HWS", cut)
        # The cut is applied in the reoriented frame and the result is turned back, so
        # assert the voxel count rather than a particular axis.
        self.assertEqual(int(np.prod(out.shape)), 20 * 16 * 20)
        self.assertEqual(out.orientation, nii.orientation)

    def test_crop_borders_with_an_unknown_key_raises(self):
        import nibabel as nib  # noqa: PLC0415

        from TPTBox import NII  # noqa: PLC0415
        from TPTBox.stitching.stitching_tools import _crop_borders  # noqa: PLC0415

        # The guard only logs a message and then indexes `cut` anyway. Pinned as-is;
        # it is a developer-facing helper and the KeyError names the missing key.
        nii = NII(nib.Nifti1Image(np.ones((20, 20, 20), dtype=np.uint16), np.eye(4)), seg=False)
        with self.assertRaises(KeyError):
            _crop_borders(nii, "NOPE", {})


if __name__ == "__main__":
    unittest.main()
