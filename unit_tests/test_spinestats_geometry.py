"""Unit tests for the geometry machinery behind `measure_ivd_and_vertebra_geometry`
and `spinestats.poi_fun.endplates`.

The private helpers are pure geometry, so they are tested against shapes whose
answers are known in closed form: a box of known side lengths, a ray through a
triangle whose plane distance is known, and a spherical cap of known radius.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from TPTBox.spine.spinestats.poi_fun import endplates as ep

# `TPTBox.spine.spinestats.__init__` re-exports the *function*
# `measure_ivd_and_vertebra_geometry` under the same name as its module, so the module
# has to be fetched explicitly rather than with a plain `from ... import`.
geom = importlib.import_module("TPTBox.spine.spinestats.measure_ivd_and_vertebra_geometry")

VOXEL = (2.0, 2.0, 5.0)
# Ray spacing for the sample-data measurements: the 0.5 mm default costs ~60 s per call.
STEP_MM = 2.0

# trimesh's ray engine does its broad phase through `rtree`, which trimesh only pulls in
# under its `easy` extra. Without it every mesh ray cast raises ModuleNotFoundError.
HAS_RTREE = importlib.util.find_spec("rtree") is not None
RTREE_REASON = "requires rtree (trimesh's ray engine broad phase)"


def _box_mask(shape=(20, 20, 20), lo=(5, 5, 5), hi=(15, 15, 15)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 1
    return arr


class TestVoxelGeometryHelpers(unittest.TestCase):
    def test_center_of_mass_of_a_box(self):
        arr = _box_mask(lo=(4, 6, 8), hi=(10, 12, 14))
        com = geom._center_of_mass_voxels(arr, voxel_size=1)
        np.testing.assert_allclose(com, [6.5, 8.5, 10.5])

    def test_center_of_mass_scales_with_the_voxel_size(self):
        arr = _box_mask(lo=(4, 6, 8), hi=(10, 12, 14))
        plain = geom._center_of_mass_voxels(arr, voxel_size=1)
        scaled = geom._center_of_mass_voxels(arr, voxel_size=np.array(VOXEL))
        np.testing.assert_allclose(scaled, plain * np.array(VOXEL))

    def test_pca_axes_of_an_elongated_box_include_its_long_axis(self):
        arr = np.zeros((40, 20, 20), dtype=np.uint8)
        arr[4:36, 8:12, 8:12] = 1  # long along axis 0
        axes = geom._pca_principal_axes(arr, voxel_size=np.ones(3))
        # Returns (up, v1, v2) where "up" is chosen to follow `up_axis`, not the dominant
        # component, so the long axis shows up as one of the three (here as v1).
        dots = [abs(float(np.dot(np.asarray(a), [1.0, 0.0, 0.0]))) for a in axes]
        self.assertAlmostEqual(max(dots), 1.0, places=6)

    def test_pca_axes_are_an_orthonormal_triple(self):
        axes = geom._pca_principal_axes(_box_mask(), voxel_size=np.ones(3))
        self.assertEqual(len(axes), 3)
        mat = np.stack([np.asarray(a) for a in axes])
        for i in range(3):
            with self.subTest(axis=i):
                self.assertAlmostEqual(float(np.linalg.norm(mat[i])), 1.0, places=6)
        np.testing.assert_allclose(np.abs(mat @ mat.T), np.eye(3), atol=1e-6)

    def test_swap(self):
        self.assertEqual(geom._swap(1, 2), (2, 1))

    def test_grid_origins_spans_the_requested_plane(self):
        base = np.array([1.0, 2.0, 3.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        xs = np.array([-1.0, 0.0, 1.0])
        ys = np.array([0.0, 2.0])
        origins, flat_x, flat_y = geom._grid_origins(base, v1, v2, xs, ys)
        self.assertEqual(origins.shape, (6, 3))
        self.assertEqual(len(flat_x), 6)
        np.testing.assert_allclose(origins[0], base + xs[0] * v1 + ys[0] * v2)
        # the offsets are exactly the requested grid
        self.assertEqual(sorted(set(flat_x)), [-1.0, 0.0, 1.0])
        self.assertEqual(sorted(set(flat_y)), [0.0, 2.0])

    def test_segment_length_is_the_entry_to_exit_distance(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
        self.assertAlmostEqual(geom._segment_length(pts), 7.0, places=9)

    def test_segment_length_uses_first_and_last_point_only(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 10.0]])
        self.assertAlmostEqual(geom._segment_length(pts), 10.0, places=9)

    def test_segment_length_of_a_missed_ray_is_zero(self):
        self.assertEqual(geom._segment_length(np.zeros((0, 3))), 0)


class TestMeshConstruction(unittest.TestCase):
    """Marching-cubes meshing and orthographic projection; no ray casting, so no rtree."""

    def test_marching_cubes_mesh_of_a_box_is_watertight_and_spans_the_mask(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))  # 10 voxels per side
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        self.assertTrue(mesh.is_watertight)
        np.testing.assert_allclose(mesh.bounds[1] - mesh.bounds[0], [10.0, 10.0, 10.0])
        # The surface winding makes trimesh report a negative volume, which is why the
        # caller stores abs(mesh.volume); corner cutting keeps |volume| just under 10^3.
        self.assertLess(mesh.volume, 0.0)
        self.assertAlmostEqual(abs(mesh.volume), 10.0**3, delta=0.03 * 10.0**3)

    def test_mesh_volume_scales_with_the_voxel_size(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))
        plain = abs(geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3)).volume)
        scaled = abs(geom._segmentation_to_surface_mesh(arr, voxel_size=np.array(VOXEL)).volume)
        self.assertAlmostEqual(scaled, plain * VOXEL[0] * VOXEL[1] * VOXEL[2], delta=1e-6 * scaled)

    def test_projecting_a_box_along_its_axis_gives_the_face_area(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        flat = geom._project_mesh_onto_plane(mesh, np.array([0.0, 0.0, 1.0]))
        # Flattening collapses the two 10x10 faces perpendicular to "up" onto each other
        # (the four side faces degenerate to lines and contribute no area).
        self.assertAlmostEqual(geom._projected_area(flat), 2 * 10 * 10, delta=0.02 * 200)

    def test_projection_is_idempotent(self):
        arr = _box_mask()
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        up = np.array([0.0, 0.0, 1.0])
        once = geom._project_mesh_onto_plane(mesh, up)
        twice = geom._project_mesh_onto_plane(once, up)
        np.testing.assert_allclose(once.vertices, twice.vertices, atol=1e-9)


@unittest.skipUnless(HAS_RTREE, RTREE_REASON)
class TestMeshRayHelpers(unittest.TestCase):
    def test_intersect_ray_through_a_box_measures_its_height(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        up = np.array([0.0, 0.0, 1.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        hits = geom._intersect_ray(up, v1, v2, mesh)
        # The marching-cubes surface spans the full 10 voxels (bounds 4.5 .. 14.5).
        self.assertAlmostEqual(geom._segment_length(hits), 10.0, delta=0.2)

    def test_intersect_ray_from_dirs_matches_intersect_ray(self):
        arr = _box_mask()
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        dirs = (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        a = geom._intersect_ray_from_dirs(dirs, mesh)
        b = geom._intersect_ray(dirs[0], dirs[1], dirs[2], mesh)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_batched_rays_agree_with_single_rays(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        direction = np.array([0.0, 0.0, 1.0])
        origins = np.array([[9.0, 9.0, -50.0], [10.0, 10.0, -50.0]])
        lengths = geom._batched_ray_segments(mesh, direction, origins)[0]
        for length in np.asarray(lengths).ravel():
            with self.subTest(length=float(length)):
                self.assertAlmostEqual(float(length), 10.0, delta=0.2)

    def test_batched_rays_that_miss_return_zero(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        lengths = geom._batched_ray_segments(mesh, np.array([0.0, 0.0, 1.0]), np.array([[-100.0, -100.0, -50.0]]))[0]
        self.assertAlmostEqual(float(np.asarray(lengths).ravel()[0]), 0.0, places=9)


class TestResultContainers(unittest.TestCase):
    def test_nan_result_is_all_nan_plus_the_error(self):
        out = geom._nan_result("boom")
        self.assertEqual(out["error"], "boom")
        for key, value in out.items():
            if key == "error":
                continue
            with self.subTest(key=key):
                self.assertTrue(math.isnan(value))

    def test_height_statistics_of_an_empty_sample_are_nan(self):
        info = geom._StructureMeasurements(volume_voxel=1.0, volume_mesh=1.0, height_center=2.0, area=1.0)
        info._update_height_statistics([])
        self.assertTrue(math.isnan(info.mean_height))
        self.assertTrue(math.isnan(info.max_height))

    def test_nan_result_without_an_error_has_no_error_key(self):
        self.assertNotIn("error", geom._nan_result())

    def test_result_from_info_rounds_every_field(self):
        info = geom._StructureMeasurements(volume_voxel=1.23456789, volume_mesh=2.0, height_center=3.0, area=4.0)
        info._update_height_statistics([1.0, 2.0])  # mean/max are set lazily by this call
        out = geom._result_from_info(info)
        self.assertEqual(out["volume_voxel"], 1.2346)
        self.assertEqual(out["volume_mesh"], 2.0)
        # the public result carries exactly the documented field set
        self.assertEqual(sorted(out), sorted(geom._nan_result()))

    def test_height_statistics_track_the_samples(self):
        info = geom._StructureMeasurements(volume_voxel=1.0, volume_mesh=1.0, height_center=2.0, area=1.0)
        info._update_height_statistics([3.0, 1.0, 4.0, 2.0])  # unsorted input is sorted internally
        self.assertAlmostEqual(info.max_height, 4.0, places=6)
        self.assertAlmostEqual(info.mean_height, 2.5, places=6)
        # get_quantile indexes into the sorted sample list rather than interpolating,
        # so the 50th percentile of four samples is the third one.
        self.assertAlmostEqual(info.get_quantile(50, [1.0, 2.0, 3.0, 4.0]), 3.0, places=6)
        self.assertIsNone(geom._StructureMeasurements(1.0, 1.0, 1.0, 1.0).get_quantile(99))


class TestRayTriangleIntersect(unittest.TestCase):
    """Moeller-Trumbore: exact hit/miss decisions for a known triangle."""

    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([4.0, 0.0, 0.0])
    V2 = np.array([0.0, 4.0, 0.0])

    def _cast(self, origin, direction):
        return ep._ray_triangle_intersect(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float), self.V0, self.V1, self.V2)

    def test_ray_through_the_centroid_hits_at_the_plane_distance(self):
        centroid = (self.V0 + self.V1 + self.V2) / 3.0
        t = self._cast(centroid + np.array([0.0, 0.0, 7.0]), [0.0, 0.0, -1.0])
        self.assertIsNotNone(t)
        self.assertAlmostEqual(float(t), 7.0, places=6)

    def test_ray_outside_the_triangle_misses(self):
        self.assertIsNone(self._cast([5.0, 5.0, 3.0], [0.0, 0.0, -1.0]))

    def test_ray_parallel_to_the_triangle_misses(self):
        self.assertIsNone(self._cast([1.0, 1.0, 3.0], [1.0, 0.0, 0.0]))

    def test_triangle_behind_the_origin_misses(self):
        self.assertIsNone(self._cast([1.0, 1.0, 3.0], [0.0, 0.0, 1.0]))

    def test_distance_scales_with_the_origin_distance(self):
        centroid = (self.V0 + self.V1 + self.V2) / 3.0
        for height in (1.0, 10.0, 100.0):
            with self.subTest(height=height):
                t = self._cast(centroid + np.array([0.0, 0.0, height]), [0.0, 0.0, -1.0])
                self.assertAlmostEqual(float(t), height, places=5)

    def test_non_unit_direction_is_handled(self):
        centroid = (self.V0 + self.V1 + self.V2) / 3.0
        t = self._cast(centroid + np.array([0.0, 0.0, 6.0]), [0.0, 0.0, -2.0])
        self.assertIsNotNone(t)
        self.assertGreater(float(t), 0.0)


@unittest.skipUnless(HAS_RTREE, RTREE_REASON)
class TestRayCastToMesh(unittest.TestCase):
    """Pins the *current* behaviour of `_ray_cast_to_mesh`.

    Its comment says "closest hit", but it returns `origin + d.mean() * direction`,
    i.e. the mean over every hit along the ray. For a convex mesh entered and exited
    once that is the midpoint *inside* the mesh, not the surface point. These tests
    therefore assert the midpoint; see the PR description - changing this would move
    every computed endplate point, so it is reported rather than silently fixed.
    """

    def test_returns_the_midpoint_between_entry_and_exit(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))  # surface spans 4.5 .. 14.5
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        hit = ep._ray_cast_to_mesh(mesh, np.array([9.5, 9.5, -20.0]), np.array([0.0, 0.0, 1.0]))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(float(np.asarray(hit)[2]), 9.5, delta=0.2)

    def test_a_ray_pointing_away_misses(self):
        arr = _box_mask(lo=(5, 5, 5), hi=(15, 15, 15))
        mesh = geom._segmentation_to_surface_mesh(arr, voxel_size=np.ones(3))
        self.assertIsNone(ep._ray_cast_to_mesh(mesh, np.array([9.5, 9.5, -20.0]), np.array([0.0, 0.0, -1.0])))

    def test_hand_built_box_centred_on_the_origin(self):
        mesh = trimesh.creation.box(extents=(4.0, 4.0, 4.0))  # spans -2 .. 2
        hit = ep._ray_cast_to_mesh(mesh, np.array([0.0, 0.0, -10.0]), np.array([0.0, 0.0, 1.0]))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(float(np.asarray(hit)[2]), 0.0, places=6)

    def test_the_returned_point_lies_on_the_ray(self):
        mesh = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
        origin = np.array([0.5, -0.25, -10.0])
        direction = np.array([0.0, 0.0, 1.0])
        hit = np.asarray(ep._ray_cast_to_mesh(mesh, origin, direction))
        np.testing.assert_allclose(hit[:2], origin[:2], atol=1e-9)


class TestSampleRayVoxels(unittest.TestCase):
    def test_samples_stay_inside_the_volume_and_advance_monotonically(self):
        shape = (20, 20, 20)
        coords, t = ep._sample_ray_voxels(shape, np.array([10.0, 10.0, 2.0]), np.array([0.0, 0.0, 1.0]), max_steps=30)
        self.assertGreater(len(coords), 0)
        self.assertEqual(len(coords), len(t))
        for idx in coords:
            with self.subTest(idx=tuple(int(c) for c in idx)):
                self.assertTrue(all(0 <= int(c) < s for c, s in zip(idx, shape)))
        # one step per voxel, strictly forward
        np.testing.assert_allclose(np.diff(t), 1.0)
        self.assertEqual([int(c[2]) for c in coords], sorted(int(c[2]) for c in coords))

    def test_two_sided_adds_samples_behind_the_origin(self):
        shape = (20, 20, 20)
        one = ep._sample_ray_voxels(shape, np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 1.0]))[1]
        both = ep._sample_ray_voxels(shape, np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 1.0]), two_sided=True)[1]
        self.assertGreater(len(both), len(one))
        self.assertLess(both.min(), 0.0)
        self.assertEqual(set(one.tolist()).issubset(set(both.tolist())), True)

    def test_a_ray_pointing_straight_out_of_the_volume_yields_nothing(self):
        coords, t = ep._sample_ray_voxels((10, 10, 10), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), max_steps=5)
        self.assertEqual(len(coords), 1 if len(t) == 1 else len(coords))
        self.assertTrue(all(0 <= int(c[2]) < 10 for c in coords))


class TestEndplatePointsOnSampleData(unittest.TestCase):
    """`calc_endplate_points_` on the shipped single-vertebra CT."""

    @classmethod
    def setUpClass(cls):
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        _, cls.subreg, cls.vert, cls.label = get_test_ct()

    def test_without_an_endplate_label_the_inputs_come_back_unchanged(self):
        from TPTBox import Location  # noqa: PLC0415

        self.assertNotIn(Location.Endplate.value, self.subreg.unique(), "sample data has no Endplate label")
        vert_out, spine_out = ep.endplate_to_super_infer_endplate(self.vert.copy(), self.subreg.copy())
        np.testing.assert_array_equal(vert_out.get_array(), self.vert.get_array())
        np.testing.assert_array_equal(spine_out.get_array(), self.subreg.get_array())

    def test_endplate_voxels_are_relabelled_superior_or_inferior(self):
        from TPTBox import Location  # noqa: PLC0415

        # Paint a slab of Endplate between two vertebra instances and let the function
        # assign each voxel to the nearer vertebra's superior/inferior surface.
        spine = self.subreg.copy()
        vert = self.vert.copy()
        arr = spine.get_array()
        vert_arr = vert.get_array()
        border = (vert_arr == 22) & (arr == Location.Vertebra_Corpus_border.value)
        if int(border.sum()) == 0:
            self.skipTest("sample vertebra has no corpus border voxels")
        arr[border] = Location.Endplate.value
        spine.set_array_(arr)

        _, spine_out = ep.endplate_to_super_infer_endplate(vert, spine)
        labels = {int(u) for u in spine_out.unique()}
        self.assertNotIn(Location.Endplate.value, labels, "the generic Endplate label must be resolved")
        self.assertTrue(
            {Location.Vertebral_Body_Endplate_Superior.value, Location.Vertebral_Body_Endplate_Inferior.value} & labels,
            labels,
        )


@unittest.skipUnless(HAS_RTREE, RTREE_REASON)
class TestMeasureGeometryOnSampleData(unittest.TestCase):
    """The public entry point on the shipped sample CT (one vertebra, label 22).

    Vertebra mode is `structure_label=0` (labels above 0 and below 100); the IVD mode
    default of 100 selects disc labels instead.
    """

    @classmethod
    def setUpClass(cls):
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        cls.ct, cls.subreg, cls.vert, cls.label = get_test_ct()
        # The measurement is expensive (POI computation plus mesh ray casting), so it
        # runs exactly once per class and every assertion below reads the cached result.
        # Restricting it to the one fully contained vertebra (21 and 23 are truncated by
        # the sample volume) and coarsening the ray step takes it from ~60 s to ~17 s.
        # buffer_poi is left at its documented default of None on purpose: that path
        # used to raise AssertionError (see test_works_without_a_poi_buffer).
        cls.out = geom.measure_ivd_and_vertebra_geometry(
            None, cls.vert.copy(), cls.subreg.copy(), structure_label=0, step_size_mm=STEP_MM, instance_labels=[cls.label]
        )

    def test_works_without_a_poi_buffer(self):
        # Regression: buffer_poi=None is the documented default, but the function asked
        # the buffering decorator to write the result back unconditionally, which
        # asserts the path is not None. setUpClass already exercises that path.
        self.assertIn(self.label, self.out)

    def test_reports_finite_positive_geometry_for_the_vertebra(self):
        row = self.out[self.label]
        self.assertNotIn("error", row)
        for key in ("volume_voxel", "volume_mesh", "mean_height", "max_height", "width_lateral_x5", "width_sagittal_x6"):
            with self.subTest(key=key):
                self.assertTrue(np.isfinite(row[key]), key)
                self.assertGreater(row[key], 0.0, key)

    def test_voxel_volume_matches_the_mask(self):
        row = self.out[self.label]
        expected = float((self.vert.get_array() == self.label).sum()) * self.vert.voxel_volume()
        self.assertAlmostEqual(row["volume_voxel"], expected, delta=0.01 * expected)

    def test_mesh_volume_is_close_to_the_voxel_volume(self):
        row = self.out[self.label]
        self.assertAlmostEqual(row["volume_mesh"], row["volume_voxel"], delta=0.25 * row["volume_voxel"])

    def test_heights_are_ordered(self):
        row = self.out[self.label]
        self.assertLessEqual(row["lower_10_percent_height"], row["mean_height"])
        self.assertLessEqual(row["mean_height"], row["max_height"])

    def test_every_documented_field_is_present(self):
        self.assertEqual(sorted(self.out[self.label]), sorted(geom._nan_result()))

    def test_signal_fields_stay_nan_without_a_t2w_image(self):
        row = self.out[self.label]
        for key in ("signal", "structure_signal", "spinal_canal_signal"):
            with self.subTest(key=key):
                self.assertTrue(math.isnan(row[key]))

    def test_instance_labels_restricts_the_output(self):
        self.assertEqual(sorted(self.out), [self.label])


if __name__ == "__main__":
    unittest.main()
