"""Unit tests for `TPTBox.core.poi_fun.ray_casting` and `pixel_based_point_finder`.

These are pure numpy/NII geometry routines, so the expected answers are analytic:
trilinear interpolation must reproduce a linear field exactly, a ray through a box
of known extent must exit at a known distance, the two convex ray casters must agree
with each other, and a plane through three points must split a volume into the exact
voxel counts on either side.
"""

from __future__ import annotations

import unittest

import nibabel as nib
import numpy as np

from TPTBox import NII
from TPTBox.core.poi_fun.pixel_based_point_finder import cdist_to_point, get_nearest_neighbor, project_pois_onto_set_of_points
from TPTBox.core.poi_fun.ray_casting import (
    add_ray_to_img,
    calculate_pca_normal_np,
    max_distance_ray_cast_convex_np,
    max_distance_ray_cast_convex_npfast,
    max_distance_ray_cast_non_convex,
    ray_cast_pixel_lvl,
    set_label_above_3_point_plane,
    trilinear_interpolate,
)


def _nii(arr: np.ndarray, seg: bool = True, zoom=(1.0, 1.0, 1.0)) -> NII:
    return NII(nib.Nifti1Image(arr, np.diag([*zoom, 1.0])), seg=seg)


def _box(shape=(30, 30, 30), lo=(10, 10, 10), hi=(21, 21, 21)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 1
    return arr


class TestTrilinearInterpolate(unittest.TestCase):
    """Trilinear interpolation reproduces a linear field exactly."""

    @staticmethod
    def _linear_volume(shape=(8, 9, 10)):
        gx, gy, gz = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
        return 2.0 * gx + 3.0 * gy - 1.0 * gz

    def test_exact_on_a_linear_field(self):
        volume = self._linear_volume()
        for x, y, z in ((1.5, 2.25, 3.75), (0.5, 0.5, 0.5), (6.1, 7.2, 8.3)):
            with self.subTest(point=(x, y, z)):
                self.assertAlmostEqual(trilinear_interpolate(volume, x, y, z), 2.0 * x + 3.0 * y - z, places=9)

    def test_voxel_centres_return_the_stored_value(self):
        volume = self._linear_volume()
        for x, y, z in ((1, 2, 3), (4, 5, 6)):
            with self.subTest(point=(x, y, z)):
                self.assertAlmostEqual(trilinear_interpolate(volume, x, y, z), float(volume[x, y, z]), places=9)

    def test_outside_the_interior_returns_zero(self):
        volume = self._linear_volume(shape=(5, 5, 5))
        for point in ((-1.0, 1.0, 1.0), (1.0, -0.5, 1.0), (4.0, 1.0, 1.0), (1.0, 1.0, 4.5)):
            with self.subTest(point=point):
                self.assertEqual(trilinear_interpolate(volume, *point), 0.0)


class TestConvexRayCast(unittest.TestCase):
    def test_exit_point_of_a_box_is_its_far_face(self):
        arr = _box()
        start = np.array([15.0, 15.0, 15.0])
        out = max_distance_ray_cast_convex_np(arr, start, np.array([1.0, 0.0, 0.0]))
        self.assertIsNotNone(out)
        # the box spans index 10..20 inclusive, so the boundary sits at ~20.5
        self.assertAlmostEqual(float(np.asarray(out)[0]), 20.5, delta=0.6)
        np.testing.assert_allclose(np.asarray(out)[1:], [15.0, 15.0], atol=1e-6)

    def test_the_two_implementations_agree(self):
        arr = _box()
        start = np.array([15.0, 15.0, 15.0])
        for direction in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]):
            with self.subTest(direction=tuple(direction)):
                slow = np.asarray(max_distance_ray_cast_convex_np(arr, start, np.array(direction)))
                fast = np.asarray(max_distance_ray_cast_convex_npfast(arr, start, np.array(direction)))
                np.testing.assert_allclose(slow, fast, atol=0.6)

    def test_distance_grows_with_the_region(self):
        small = max_distance_ray_cast_convex_np(_box(hi=(16, 21, 21)), np.array([12.0, 15.0, 15.0]), np.array([1.0, 0.0, 0.0]))
        large = max_distance_ray_cast_convex_np(_box(hi=(25, 21, 21)), np.array([12.0, 15.0, 15.0]), np.array([1.0, 0.0, 0.0]))
        self.assertLess(float(np.asarray(small)[0]), float(np.asarray(large)[0]))

    def test_a_start_outside_the_region_is_returned_unchanged(self):
        arr = _box()
        start = np.array([1.0, 1.0, 1.0])
        out = max_distance_ray_cast_convex_np(arr, start, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(np.asarray(out), start)

    def test_non_convex_caster_stops_at_the_near_gap(self):
        # A box with a slab removed in the middle: the bisecting convex caster reports
        # the far outer wall, the stepping one stops at the near edge of the gap.
        arr = _box()
        arr[15:18, :, :] = 0
        start = np.array([12.0, 15.0, 15.0])
        direction = np.array([1.0, 0.0, 0.0])
        convex = float(np.asarray(max_distance_ray_cast_convex_np(arr, start, direction))[0])
        non_convex = max_distance_ray_cast_non_convex(_nii(arr), start, direction, step_size=0.25)
        self.assertIsNotNone(non_convex)
        self.assertLess(float(np.asarray(non_convex)[0]), convex)
        self.assertAlmostEqual(float(np.asarray(non_convex)[0]), 15.0, delta=1.0)

    def test_non_convex_caster_returns_none_when_the_region_does_not_end_in_range(self):
        # The box continues past 12 + 2 voxels, so there is no exit point to report.
        arr = _box()
        out = max_distance_ray_cast_non_convex(
            _nii(arr), np.array([12.0, 15.0, 15.0]), np.array([1.0, 0.0, 0.0]), step_size=0.5, max_distance=2.0
        )
        self.assertIsNone(out)

    def test_non_convex_caster_finds_the_exit_with_enough_range(self):
        arr = _box()
        out = max_distance_ray_cast_non_convex(
            _nii(arr), np.array([12.0, 15.0, 15.0]), np.array([1.0, 0.0, 0.0]), step_size=0.25, max_distance=50.0
        )
        self.assertIsNotNone(out)
        self.assertAlmostEqual(float(np.asarray(out)[0]), 20.5, delta=1.0)


class TestRayCastPixelLevel(unittest.TestCase):
    def test_every_voxel_is_inside_the_shape_and_ordered(self):
        shape = (20, 20, 20)
        coords, _ = ray_cast_pixel_lvl(np.array([10.0, 10.0, 2.0]), np.array([0.0, 0.0, 1.0]), shape)
        self.assertIsNotNone(coords)
        coords = np.asarray(coords)
        self.assertTrue(((coords >= 0) & (coords < np.array(shape))).all())
        self.assertEqual(list(coords[:, 2]), sorted(coords[:, 2]))

    def test_two_sided_is_a_superset(self):
        shape = (20, 20, 20)
        one, _ = ray_cast_pixel_lvl(np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 1.0]), shape)
        both, _ = ray_cast_pixel_lvl(np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 1.0]), shape, two_sided=True)
        self.assertGreater(len(np.asarray(both)), len(np.asarray(one)))
        self.assertTrue(((np.asarray(both) >= 0) & (np.asarray(both) < np.array(shape))).all())

    def test_a_diagonal_ray_stays_in_bounds(self):
        shape = (12, 12, 12)
        coords, _ = ray_cast_pixel_lvl(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), shape)
        coords = np.asarray(coords)
        self.assertTrue(((coords >= 0) & (coords < np.array(shape))).all())


class TestAddRayToImg(unittest.TestCase):
    def test_painting_a_ray_adds_voxels(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        out = add_ray_to_img(np.array([10, 10, 2]), np.array([0.0, 0.0, 1.0]), _nii(arr), value=5, dilate=1)
        self.assertIsNotNone(out)
        self.assertGreater(int((out.get_seg_array() != 0).sum()), 0)

    def test_inplace_modifies_the_input(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        nii = _nii(arr)
        add_ray_to_img(np.array([10, 10, 2]), np.array([0.0, 0.0, 1.0]), nii, value=5, dilate=1, inplace=True)
        self.assertGreater(int((nii.get_seg_array() != 0).sum()), 0)

    def test_dilate_grows_the_ray(self):
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        thin = add_ray_to_img(np.array([10, 10, 2]), np.array([0.0, 0.0, 1.0]), _nii(arr), value=5, dilate=1)
        thick = add_ray_to_img(np.array([10, 10, 2]), np.array([0.0, 0.0, 1.0]), _nii(arr), value=5, dilate=3)
        self.assertLess(int((thin.get_seg_array() != 0).sum()), int((thick.get_seg_array() != 0).sum()))


class TestSetLabelAbovePlane(unittest.TestCase):
    def test_axis_aligned_plane_splits_the_volume_exactly(self):
        arr = np.ones((10, 10, 10), dtype=np.uint8)
        # A plane at z = 4 spanned by three points in that plane.
        out = set_label_above_3_point_plane(arr.copy(), [0.0, 0.0, 4.0], [9.0, 0.0, 4.0], [0.0, 9.0, 4.0], value=0)
        out = np.asarray(out if isinstance(out, np.ndarray) else out.get_seg_array())
        zeroed = int((out == 0).sum())
        self.assertEqual(zeroed + int((out == 1).sum()), arr.size)
        self.assertGreater(zeroed, 0)
        # every zeroed voxel is on one side of z = 4
        zs = np.argwhere(out == 0)[:, 2]
        self.assertTrue((zs >= 4).all() or (zs <= 4).all())

    def test_invert_flips_which_side_is_cleared(self):
        arr = np.ones((10, 10, 10), dtype=np.uint8)
        kwargs = {"p1": [0.0, 0.0, 4.0], "p2": [9.0, 0.0, 4.0], "p3": [0.0, 9.0, 4.0], "value": 0}
        up = np.asarray(set_label_above_3_point_plane(arr.copy(), invert=1, **kwargs))
        down = np.asarray(set_label_above_3_point_plane(arr.copy(), invert=-1, **kwargs))
        self.assertNotEqual(int((up == 0).sum()), int((down == 0).sum()))
        # The two strict sides plus the z == 4 plane itself account for the whole volume.
        on_plane = arr[:, :, 4].size
        self.assertEqual(int((up == 0).sum()) + int((down == 0).sum()) + on_plane, arr.size)

    def test_accepts_a_nii_and_returns_a_nii(self):
        nii = _nii(np.ones((8, 8, 8), dtype=np.uint8))
        out = set_label_above_3_point_plane(nii, [0.0, 0.0, 4.0], [7.0, 0.0, 4.0], [0.0, 7.0, 4.0], value=0)
        self.assertIsInstance(out, NII)
        self.assertLess(int(out.get_seg_array().sum()), int(nii.get_seg_array().sum()))


class TestPcaNormal(unittest.TestCase):
    def test_first_component_of_an_elongated_blob(self):
        arr = np.zeros((40, 12, 12), dtype=np.uint8)
        arr[4:36, 5:8, 5:8] = 1
        normal = calculate_pca_normal_np(arr, pca_component=0)
        self.assertAlmostEqual(abs(float(np.dot(normal / np.linalg.norm(normal), [1.0, 0.0, 0.0]))), 1.0, places=3)

    def test_zoom_only_rescales_the_returned_vector(self):
        # Pins the current semantics: the PCA is fitted on raw voxel indices, and `zoom`
        # is applied afterwards as a component-wise division of the resulting vector. It
        # therefore does not change *which* axis is principal, and the result is no
        # longer a unit vector. See the PR description.
        arr = np.zeros((20, 20, 12), dtype=np.uint8)
        arr[4:16, 6:14, 5:8] = 1
        zoom = (1.0, 4.0, 2.0)
        plain = calculate_pca_normal_np(arr, pca_component=1)
        scaled = calculate_pca_normal_np(arr, pca_component=1, zoom=zoom)
        np.testing.assert_allclose(scaled, plain / np.array(zoom), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(plain)), 1.0, places=9)

    def test_components_are_orthogonal(self):
        arr = np.zeros((30, 20, 12), dtype=np.uint8)
        arr[4:26, 4:16, 4:8] = 1
        first = calculate_pca_normal_np(arr, 0)
        second = calculate_pca_normal_np(arr, 1)
        self.assertAlmostEqual(float(np.dot(first, second)), 0.0, places=5)


class TestPixelBasedPointFinder(unittest.TestCase):
    def test_cdist_to_point(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        out = cdist_to_point((0.0, 0.0, 0.0), points)
        np.testing.assert_allclose(np.asarray(out).ravel(), [0.0, 5.0])

    def test_get_nearest_neighbor_picks_the_closest_labelled_voxel(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[8, 8, 8] = 3
        arr[1, 1, 1] = 3
        out = get_nearest_neighbor(np.array([0.0, 0.0, 0.0]), _nii(arr), 3)
        np.testing.assert_allclose(np.asarray(out), [1.0, 1.0, 1.0])

    def test_project_pois_onto_set_of_points_snaps_to_the_nearest(self):
        from TPTBox import POI  # noqa: PLC0415

        poi = POI({1: {50: (0.0, 0.0, 0.0)}}, orientation=("R", "A", "S"), zoom=(1.0, 1.0, 1.0), shape=(10.0, 10.0, 10.0))
        point_set = np.array([[5.0, 5.0, 5.0], [1.0, 0.0, 0.0]])
        out = project_pois_onto_set_of_points(poi, point_set)
        np.testing.assert_allclose(out[1, 50], (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
