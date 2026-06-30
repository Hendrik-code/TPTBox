# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import nibabel as nib
import numpy as np

from TPTBox import Location
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import calc_poi_from_subreg_vert
from TPTBox.core.poi_fun._help import sacrum_w_o_arcus
from TPTBox.core.poi_fun.ray_casting import (
    add_ray_to_img,
    add_spline_to_img,
    calculate_pca_normal_np,
    max_distance_ray_cast_convex,
    max_distance_ray_cast_convex_np,
    max_distance_ray_cast_convex_npfast,
    max_distance_ray_cast_convex_poi,
    ray_cast_pixel_lvl,
    set_label_above_3_point_plane,
    shift_point,
    trilinear_interpolate,
    unit_vector,
)
from TPTBox.tests.test_utils import get_test_ct


def _quiet():
    """Return a fresh context manager that swallows ``stdout`` noise."""
    return redirect_stdout(io.StringIO())


def _solid_cube(shape=(40, 40, 40), lo=10, hi=30, label=1) -> NII:
    """Build a binary ``NII`` containing one filled (convex) cube."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[lo:hi, lo:hi, lo:hi] = label
    return NII(nib.Nifti1Image(arr, np.eye(4)), seg=True)


class Test_Ray_Casting_Numpy(unittest.TestCase):
    """Functions that operate on plain numpy arrays / a hand-made cube."""

    def setUp(self):
        self.nii = _solid_cube()
        self.arr = self.nii.get_array().astype(float)
        self.center = np.array([20.0, 20.0, 20.0])

    def test_unit_vector(self):
        v = unit_vector(np.array([3.0, 0.0, 0.0]))
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(unit_vector(np.array([1.0, 2.0, -2.0])))), 1.0)

    def test_trilinear_interpolate(self):
        self.assertEqual(trilinear_interpolate(self.arr, 20.0, 20.0, 20.0), 1.0)
        # outside the valid interior -> 0.0
        self.assertEqual(trilinear_interpolate(self.arr, -1.0, 0.0, 0.0), 0.0)
        self.assertEqual(trilinear_interpolate(self.arr, 0.0, 0.0, 0.0), 0.0)
        # half-way across the boundary face -> in (0, 1)
        val = trilinear_interpolate(self.arr, 29.5, 20.0, 20.0)
        self.assertTrue(0.0 <= val <= 1.0)

    def test_max_distance_convex_variants_agree(self):
        direction = np.array([1.0, 0.0, 0.0])
        with _quiet():
            fast = max_distance_ray_cast_convex_npfast(self.arr, self.center, direction)
        npv = max_distance_ray_cast_convex_np(self.arr, self.center, direction)
        niiv = max_distance_ray_cast_convex(self.nii, self.center, direction)
        # exit point on the +x face is at x ~= 29.5, y/z unchanged
        for exit_point in (fast, npv, niiv):
            self.assertEqual(len(exit_point), 3)
            self.assertAlmostEqual(exit_point[0], 29.5, delta=0.5)
            self.assertAlmostEqual(exit_point[1], 20.0, delta=0.5)
        # the distance travelled is non-negative
        self.assertGreaterEqual(float(np.linalg.norm(npv - self.center)), 0.0)

    def test_max_distance_convex_outside_start(self):
        outside = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(max_distance_ray_cast_convex_np(self.arr, outside, direction), outside)
        np.testing.assert_array_equal(max_distance_ray_cast_convex(self.nii, outside, direction), outside)
        with _quiet():
            np.testing.assert_array_equal(max_distance_ray_cast_convex_npfast(self.arr, outside, direction), outside)

    def test_max_distance_convex_np_max_v(self):
        out = max_distance_ray_cast_convex_np(self.arr, self.center, np.array([1.0, 0.0, 0.0]), max_v=5)
        self.assertEqual(len(out), 3)

    def test_ray_cast_pixel_lvl(self):
        direction = np.array([1.0, 0.5, 0.25])  # all non-zero -> no /0 warning
        plane_coords, arange = ray_cast_pixel_lvl(self.center, direction, self.nii.shape)
        self.assertEqual(plane_coords.shape[0], arange.shape[0])
        self.assertEqual(plane_coords.shape[1], 3)
        self.assertGreaterEqual(int(plane_coords.min()), 0)
        for axis in range(3):
            self.assertLess(int(plane_coords[:, axis].max()), self.nii.shape[axis])
        self.assertEqual(float(arange[0]), 0.0)
        # two-sided concatenates both halves
        pc2, ar2 = ray_cast_pixel_lvl(self.center, direction, self.nii.shape, two_sided=True)
        self.assertEqual(pc2.shape[0], ar2.shape[0])
        self.assertGreater(pc2.shape[0], plane_coords.shape[0])

    def test_add_ray_to_img(self):
        direction = np.array([1.0, 0.5, 0.25])
        with _quiet():
            composed = add_ray_to_img(self.center, direction, self.nii, add_to_img=True, value=5, dilate=1)
        self.assertIsInstance(composed, NII)
        self.assertIn(5, composed.unique())
        # the cube (label 1) is still present in the composite
        self.assertIn(1, composed.unique())
        with _quiet():
            ray_only = add_ray_to_img(self.center, direction, self.nii, add_to_img=False, value=7, dilate=0)
        self.assertIsInstance(ray_only, NII)
        np.testing.assert_array_equal(ray_only.unique(), np.array([7]))

    def test_calculate_pca_normal_np(self):
        # elongate the region along axis 0 so PC1 is well-defined
        arr = np.zeros((40, 20, 20), dtype=np.uint8)
        arr[5:35, 8:12, 8:12] = 1
        for component in (0, 1, 2):
            n = calculate_pca_normal_np(arr, component)
            self.assertEqual(n.shape, (3,))
            self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=5)
        # primary axis should align with the elongated (first) axis
        with _quiet():
            pc1 = calculate_pca_normal_np(arr, 0, verbose=True)
        self.assertGreater(abs(pc1[0]), abs(pc1[1]))
        self.assertGreater(abs(pc1[0]), abs(pc1[2]))
        # zoom scales the vector (it is no longer unit length)
        scaled = calculate_pca_normal_np(arr, 0, zoom=(2.0, 1.0, 1.0))
        self.assertFalse(np.allclose(np.linalg.norm(scaled), 1.0))

    def test_set_label_above_3_point_plane(self):
        for inp in (self.nii.copy(), self.arr.copy()):
            before = float(np.asarray(inp).sum()) if not isinstance(inp, NII) else float(inp.sum())
            out = set_label_above_3_point_plane(inp, [20, 20, 10], [20, 10, 20], [10, 20, 20], value=0)
            self.assertIsInstance(out, type(inp))
            after = float(out.sum()) if isinstance(out, NII) else float(np.asarray(out).sum())
            # zeroing a half-space cannot increase the foreground sum
            self.assertLessEqual(after, before)
        # invert flips which side is cleared
        out_pos = set_label_above_3_point_plane(self.arr.copy(), [20, 20, 10], [20, 10, 20], [10, 20, 20], value=0, invert=1)
        out_neg = set_label_above_3_point_plane(self.arr.copy(), [20, 20, 10], [20, 10, 20], [10, 20, 20], value=0, invert=-1)
        self.assertNotAlmostEqual(float(out_pos.sum()), float(out_neg.sum()))

    def test_set_label_inplace(self):
        cube = _solid_cube()
        out = set_label_above_3_point_plane(cube, [20, 20, 10], [20, 10, 20], [10, 20, 20], value=0, inplace=True)
        self.assertIs(out, cube)

    def test_set_label_inferior_superior_orientation(self):
        # an NII whose superior axis points "I" flips the invert convention internally
        cube = _solid_cube().reorient(("R", "A", "I"))
        out = set_label_above_3_point_plane(cube, [20, 20, 10], [20, 10, 20], [10, 20, 20], value=0)
        self.assertIsInstance(out, NII)
        self.assertLessEqual(float(out.sum()), float(cube.sum()))


class Test_Ray_Casting_POI(unittest.TestCase):
    """POI-driven ray casting (incidentally covers _help + pixel_based_point_finder)."""

    @classmethod
    def setUpClass(cls):
        ct, subreg, vert, label = get_test_ct()
        cls.vert = vert
        # this set yields the articular-process landmarks (45-48) shift_point needs
        # *and* the Vertebra_Direction landmarks (128-130) get_direction needs.
        subreg_id = [
            Location.Vertebra_Corpus,
            Location.Superior_Articular_Right,
            Location.Superior_Articular_Left,
            Location.Inferior_Articular_Right,
            Location.Inferior_Articular_Left,
            128,
            129,
            130,
        ]
        with _quiet():
            cls.poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=subreg_id, verbose=False)
        cls.vert_ids = cls.poi.keys_region()
        cls.vid = cls.vert_ids[len(cls.vert_ids) // 2]
        region = vert.extract_label(cls.vid)
        cls.bb = region.compute_crop()
        cls.region = region.apply_crop(cls.bb)

    def test_max_distance_ray_cast_convex_poi_direction(self):
        with _quiet():
            point = max_distance_ray_cast_convex_poi(
                self.poi, self.region, self.vid, self.bb, normal_vector_points="R", start_point=Location.Vertebra_Corpus
            )
        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(len(point), 3)
        # exit point lies (approximately) inside the cropped region
        for axis in range(3):
            self.assertGreaterEqual(point[axis], -1.0)
            self.assertLessEqual(point[axis], self.region.shape[axis] + 1.0)

    def test_max_distance_ray_cast_convex_poi_location_pair(self):
        with _quiet():
            point = max_distance_ray_cast_convex_poi(
                self.poi,
                self.region,
                self.vid,
                self.bb,
                normal_vector_points=(Location.Superior_Articular_Right, Location.Superior_Articular_Left),
                start_point=Location.Vertebra_Corpus,
            )
        if point is not None:
            self.assertEqual(len(point), 3)
        # a tuple whose second landmark is absent short-circuits to None
        with _quiet():
            missing = max_distance_ray_cast_convex_poi(
                self.poi,
                self.region,
                self.vid,
                self.bb,
                normal_vector_points=(Location.Superior_Articular_Right, Location(81)),
                start_point=Location.Vertebra_Corpus,
            )
        self.assertIsNone(missing)

    def test_max_distance_ray_cast_convex_poi_missing_direction(self):
        # start point resolves but the vertebra has no direction landmarks ->
        # get_direction raises KeyError, which is caught and yields None.
        from TPTBox import POI

        cube = _solid_cube(shape=(20, 20, 20), lo=5, hi=15)
        poi = POI(
            {7: {Location.Vertebra_Corpus.value: (10.0, 10.0, 10.0)}},
            orientation=("R", "A", "S"),
            zoom=(1, 1, 1),
            shape=(20, 20, 20),
            origin=(0, 0, 0),
            rotation=np.eye(3),
        )
        with _quiet():
            point = max_distance_ray_cast_convex_poi(poi, cube, 7, None, normal_vector_points="R", start_point=Location.Vertebra_Corpus)
        self.assertIsNone(point)

    def test_max_distance_ray_cast_convex_poi_missing_vert(self):
        with _quiet():
            point = max_distance_ray_cast_convex_poi(
                self.poi, self.region, 999, self.bb, normal_vector_points="R", start_point=Location.Vertebra_Corpus
            )
        self.assertIsNone(point)

    def test_max_distance_ray_cast_convex_poi_ndarray_start(self):
        start = np.array([2.0, 2.0, 2.0])
        with _quiet():
            point = max_distance_ray_cast_convex_poi(self.poi, self.region, self.vid, self.bb, normal_vector_points="R", start_point=start)
        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(len(point), 3)

    def test_shift_point(self):
        with _quiet():
            shifted = shift_point(self.poi, self.vid, self.bb, start_point=Location.Vertebra_Corpus, direction="R")
        self.assertIsNotNone(shifted)
        assert shifted is not None
        self.assertEqual(len(shifted), 3)
        # direction=None returns the raw local start point (no displacement)
        with _quiet():
            raw = shift_point(self.poi, self.vid, self.bb, start_point=Location.Vertebra_Corpus, direction=None)
        self.assertIsNotNone(raw)
        assert raw is not None
        self.assertEqual(len(raw), 3)

    def test_shift_point_sacrum_skipped(self):
        # vertebra ids without arcus are skipped -> None
        with _quiet():
            out = shift_point(self.poi, sacrum_w_o_arcus[0], self.bb, start_point=Location.Vertebra_Corpus, direction="R")
        self.assertIsNone(out)

    def test_add_spline_to_img(self):
        with _quiet():
            composed = add_spline_to_img(self.vert.copy(), self.poi, location=50, add_to_img=True, value=100, dilate=2)
        self.assertIsInstance(composed, NII)
        self.assertIn(100, composed.unique())
        # standalone spline image (only the spline label present)
        with _quiet():
            spline = add_spline_to_img(self.vert.copy(), self.poi, location=50, add_to_img=False, value=77, dilate=1)
        np.testing.assert_array_equal(spline.unique(), np.array([77]))
        # override_seg=False only fills background voxels
        with _quiet():
            merged = add_spline_to_img(self.vert.copy(), self.poi, location=50, add_to_img=True, override_seg=False, value=123, dilate=1)
        self.assertIsInstance(merged, NII)


if __name__ == "__main__":
    unittest.main()
