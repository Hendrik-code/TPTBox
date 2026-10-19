"""Unit tests for the parts of `NII` / `NII_Math` / `Has_Grid` that the existing
`test_nii*.py` files do not reach.

Everything here is built from small hand-written arrays whose expected result can be
written down: a seed label diffusing into a reference mask, a slab truncated at a
reference label, a cube's surface voxel count, a histogram match against a known
reference, and the reflected/in-place arithmetic operators.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import NII
from TPTBox.core.nii_poi_abstract import Grid


def _nii(arr: np.ndarray, seg: bool = True, zoom=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) -> NII:
    affine = np.diag([*zoom, 1.0])
    affine[:3, 3] = origin
    return NII(nib.Nifti1Image(arr, affine), seg=seg)


class TestInfect(unittest.TestCase):
    """`infect` diffuses existing labels into a reference mask."""

    def test_two_seeds_split_the_corridor_by_distance(self):
        seeds = np.zeros((11, 3, 3), dtype=np.uint8)
        seeds[0] = 1
        seeds[10] = 2
        reference = np.ones((11, 3, 3), dtype=np.uint8)
        out = _nii(seeds).infect(_nii(reference), verbose=False)
        arr = out.get_seg_array()
        self.assertEqual(int(arr[1, 1, 1]), 1, "the near end belongs to seed 1")
        self.assertEqual(int(arr[9, 1, 1]), 2, "the far end belongs to seed 2")
        self.assertEqual(int((arr == 0).sum()), 0, "the whole corridor must be claimed")

    def test_labels_do_not_leave_the_reference_mask(self):
        seeds = np.zeros((9, 9, 9), dtype=np.uint8)
        seeds[4, 4, 4] = 3
        reference = np.zeros((9, 9, 9), dtype=np.uint8)
        reference[3:6, 3:6, 3:6] = 1
        out = _nii(seeds).infect(_nii(reference), verbose=False)
        arr = out.get_seg_array()
        self.assertEqual(set(np.unique(arr)) - {0}, {3})
        outside = arr.copy()
        outside[3:6, 3:6, 3:6] = 0
        self.assertEqual(int(outside.sum()), 0)

    def test_max_depth_limits_the_diffusion(self):
        seeds = np.zeros((11, 3, 3), dtype=np.uint8)
        seeds[0] = 1
        reference = np.ones((11, 3, 3), dtype=np.uint8)
        shallow = _nii(seeds).infect(_nii(reference), verbose=False, max_depth=2).get_seg_array()
        deep = _nii(seeds).infect(_nii(reference), verbose=False).get_seg_array()
        self.assertLess(int((shallow != 0).sum()), int((deep != 0).sum()))

    def test_inplace_variant_modifies_the_receiver(self):
        seeds = np.zeros((7, 3, 3), dtype=np.uint8)
        seeds[0] = 1
        nii = _nii(seeds)
        before = int((nii.get_seg_array() != 0).sum())
        nii.infect_(_nii(np.ones((7, 3, 3), dtype=np.uint8)), verbose=False)
        self.assertGreater(int((nii.get_seg_array() != 0).sum()), before)

    def test_mismatched_affine_is_rejected(self):
        seeds = np.zeros((5, 5, 5), dtype=np.uint8)
        seeds[0, 0, 0] = 1
        other = _nii(np.ones((5, 5, 5), dtype=np.uint8), zoom=(2.0, 2.0, 2.0))
        with self.assertRaises(AssertionError):
            _nii(seeds).infect(other, verbose=False)


class TestTruncateLabelsBeyondReference(unittest.TestCase):
    def test_voxels_beyond_the_reference_are_removed(self):
        arr = np.zeros((3, 3, 10), dtype=np.uint8)
        arr[..., :] = 1  # label 1 fills the column
        arr[1, 1, 4] = 2  # the reference label sits in the middle
        out = _nii(arr).truncate_labels_beyond_reference(idx=1, not_beyond=2, axis="S")
        kept = out.get_seg_array()
        self.assertEqual(int((kept[..., 5:] == 1).sum()), 0, "nothing of label 1 survives above the reference")
        self.assertGreater(int((kept[..., :4] == 1).sum()), 0, "below the reference is untouched")

    def test_inclusion_also_drops_the_reference_slice(self):
        arr = np.ones((3, 3, 10), dtype=np.uint8)
        arr[1, 1, 4] = 2
        without = _nii(arr).truncate_labels_beyond_reference(idx=1, not_beyond=2, inclusion=False).get_seg_array()
        with_incl = _nii(arr).truncate_labels_beyond_reference(idx=1, not_beyond=2, inclusion=True).get_seg_array()
        self.assertLessEqual(int((with_incl == 1).sum()), int((without == 1).sum()))

    def test_fill_value_is_used(self):
        arr = np.ones((3, 3, 10), dtype=np.uint8)
        arr[1, 1, 4] = 2
        out = _nii(arr).truncate_labels_beyond_reference(idx=1, not_beyond=2, fill=7).get_seg_array()
        self.assertIn(7, np.unique(out))

    def test_inplace_variant(self):
        arr = np.ones((3, 3, 10), dtype=np.uint8)
        arr[1, 1, 4] = 2
        nii = _nii(arr)
        nii.truncate_labels_beyond_reference_(idx=1, not_beyond=2)
        self.assertEqual(int((nii.get_seg_array()[..., 5:] == 1).sum()), 0)


class TestTranslateArr(unittest.TestCase):
    def test_tuple_translation_shifts_the_content(self):
        arr = np.zeros((7, 7, 7), dtype=np.uint8)
        arr[1, 1, 1] = 5
        out = _nii(arr).translate_arr((2, 0, 0), verbose=False).get_seg_array()
        self.assertEqual(int(out[3, 1, 1]), 5)
        self.assertEqual(int(out[1, 1, 1]), 0)

    def test_direction_dict_translation(self):
        arr = np.zeros((7, 7, 7), dtype=np.uint8)
        arr[3, 3, 3] = 5
        out = _nii(arr).translate_arr({"S": 2}, verbose=False).get_seg_array()
        self.assertEqual(int(out.sum()), 5, "the label must still be in the volume exactly once")
        self.assertEqual(int(np.argwhere(out == 5).shape[0]), 1)
        self.assertFalse(np.array_equal(out, arr))

    def test_zero_translation_is_a_noop(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        np.testing.assert_array_equal(_nii(arr).translate_arr((0, 0, 0), verbose=False).get_seg_array(), arr)

    def test_content_shifted_out_is_dropped(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[4, 2, 2] = 1
        out = _nii(arr).translate_arr((3, 0, 0), verbose=False).get_seg_array()
        self.assertEqual(int(out.sum()), 0)


class TestSegmentationComparisons(unittest.TestCase):
    def _pair(self):
        gt = np.zeros((8, 8, 8), dtype=np.uint8)
        gt[2:6, 2:6, 2:6] = 1
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        pred[3:7, 2:6, 2:6] = 1
        return _nii(pred), _nii(gt)

    def test_difference_marks_false_positive_and_false_negative(self):
        pred, gt = self._pair()
        diff = pred.get_segmentation_difference_to(gt).get_seg_array()
        self.assertGreater(len(set(np.unique(diff)) - {0}), 1, "at least FP and FN classes are marked")

    def test_identical_masks_differ_only_in_the_true_positive_class(self):
        pred, _ = self._pair()
        diff = pred.get_segmentation_difference_to(pred.copy()).get_seg_array()
        self.assertEqual(len(set(np.unique(diff)) - {0}), 1)

    def test_ignore_background_tp_drops_the_true_positive_class(self):
        pred, gt = self._pair()
        with_tp = pred.get_segmentation_difference_to(gt, ignore_background_tp=False).get_seg_array()
        without = pred.get_segmentation_difference_to(gt, ignore_background_tp=True).get_seg_array()
        self.assertLessEqual(int((without != 0).sum()), int((with_tp != 0).sum()))

    def test_overlapping_labels(self):
        a = np.zeros((6, 6, 6), dtype=np.uint8)
        a[0:3] = 1
        a[3:6] = 2
        b = np.zeros((6, 6, 6), dtype=np.uint8)
        b[0:3] = 7
        b[3:6] = 8
        pairs = _nii(a).get_overlapping_labels_to(_nii(b))
        self.assertEqual(sorted(tuple(p) for p in pairs), [(1, 7), (2, 8)])

    def test_map_labels_based_on_majority_overlap(self):
        src = np.zeros((6, 6, 6), dtype=np.uint8)
        src[0:3] = 1
        reference = np.zeros((6, 6, 6), dtype=np.uint8)
        reference[0:3] = 9
        out = _nii(src).map_labels_based_on_majority_label_mask_overlap(_nii(reference))
        self.assertEqual(set(np.unique(out.get_seg_array())) - {0}, {9})


class TestConnectedComponentsAndSurface(unittest.TestCase):
    def _three_blobs(self):
        arr = np.zeros((20, 8, 8), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1  # 8 voxels
        arr[6:9, 1:4, 1:4] = 1  # 27 voxels
        arr[12:16, 1:5, 1:5] = 1  # 64 voxels
        return _nii(arr)

    def test_filter_connected_components_by_min_volume(self):
        out = self._three_blobs().filter_connected_components(labels=1, min_volume=20)
        self.assertEqual(int((out.get_seg_array() != 0).sum()), 27 + 64)

    def test_filter_connected_components_by_max_count(self):
        out = self._three_blobs().filter_connected_components(labels=1, max_count_component=1)
        self.assertEqual(int((out.get_seg_array() != 0).sum()), 64, "the largest component survives")

    def test_largest_k_components_is_deprecated(self):
        # The method only raises now; filter_connected_components is the replacement.
        with self.assertRaises(DeprecationWarning):
            self._three_blobs().get_largest_k_segmentation_connected_components(2, labels=1)

    def test_get_connected_components_labels_each_blob(self):
        out = self._three_blobs().get_connected_components(labels=1)
        self.assertEqual(len(set(np.unique(out.get_seg_array())) - {0}), 3)

    def test_compute_surface_points_are_on_the_boundary(self):
        arr = np.zeros((9, 9, 9), dtype=np.uint8)
        arr[3:6, 3:6, 3:6] = 1
        points = _nii(arr).compute_surface_points()
        self.assertGreater(len(points), 0)
        self.assertNotIn((4, 4, 4), [tuple(p) for p in points], "the centre voxel is interior")

    def test_fill_holes_global_with_majority_voting(self):
        arr = np.zeros((9, 9, 9), dtype=np.uint8)
        arr[2:7, 2:7, 2:7] = 1
        arr[4, 4, 4] = 0  # a one-voxel hole
        out = _nii(arr).fill_holes_global_with_majority_voting()
        self.assertEqual(int(out.get_seg_array()[4, 4, 4]), 1)

    def test_extract_background(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[1:4, 1:4, 1:4] = 1
        out = _nii(arr).extract_background().get_seg_array()
        self.assertEqual(int(out.sum()), int((arr == 0).sum()))


class TestIntensityOperations(unittest.TestCase):
    def test_normalize_to_range_rescales_in_place(self):
        arr = np.linspace(-100, 500, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        nii = _nii(arr, seg=False)
        nii.normalize_to_range_(min_value=0, max_value=1000, verbose=False)
        out = nii.get_array()
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1000.0)

    def test_match_histograms_moves_towards_the_reference(self):
        rng = np.random.default_rng(0)
        source = _nii((rng.normal(100, 10, (16, 16, 16))).astype(np.float32), seg=False)
        reference = _nii((rng.normal(500, 10, (16, 16, 16))).astype(np.float32), seg=False)
        matched = source.match_histograms(reference)
        self.assertAlmostEqual(float(matched.get_array().mean()), 500.0, delta=25.0)
        self.assertAlmostEqual(float(source.get_array().mean()), 100.0, delta=5.0, msg="the source must be untouched")

    def test_match_histograms_inplace(self):
        rng = np.random.default_rng(1)
        source = _nii((rng.normal(100, 10, (12, 12, 12))).astype(np.float32), seg=False)
        reference = _nii((rng.normal(400, 10, (12, 12, 12))).astype(np.float32), seg=False)
        source.match_histograms_(reference)
        self.assertAlmostEqual(float(source.get_array().mean()), 400.0, delta=25.0)

    def test_smooth_gaussian_labelwise_keeps_the_label_set(self):
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[2:8, 2:8, 2:8] = 1
        arr[9:14, 9:14, 9:14] = 2
        out = _nii(arr).smooth_gaussian_labelwise(label_to_smooth=1, sigma=1.0, radius=2)
        self.assertTrue(set(np.unique(out.get_seg_array())).issubset({0, 1, 2}))
        self.assertIn(2, np.unique(out.get_seg_array()), "other labels survive")


class TestGridAndAccessors(unittest.TestCase):
    def test_accessors(self):
        nii = _nii(np.zeros((4, 5, 6), dtype=np.uint8), zoom=(1.0, 2.0, 3.0))
        self.assertEqual(nii.dims, 3)
        self.assertEqual(len(nii.direction), 9)
        self.assertEqual(len(nii.direction_itk), 9)
        self.assertEqual(nii.orientation_ornt.shape, (3, 2))
        self.assertEqual(nii.get_num_dims(), 3)
        self.assertEqual(nii.shape_int, (4, 5, 6))
        np.testing.assert_allclose(nii.spacing, (1.0, 2.0, 3.0))

    def test_numpy_and_ravel(self):
        arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        nii = _nii(arr)
        np.testing.assert_array_equal(nii.numpy(), arr)
        np.testing.assert_array_equal(nii.ravel(), arr.ravel())

    def test_set_description_is_stored_in_the_header(self):
        nii = _nii(np.zeros((3, 3, 3), dtype=np.uint8))
        nii.set_description("hello")
        self.assertIn("hello", str(nii.header["descrip"]))

    def test_corners_and_obb_quads(self):
        nii = _nii(np.zeros((4, 4, 4), dtype=np.uint8), zoom=(2.0, 2.0, 2.0))
        corners = nii.get_corners()
        self.assertEqual(np.asarray(corners).shape[0], 8)
        quads = nii.get_obb_quads()
        self.assertEqual(len(quads), 6, "a box has six faces")

    def test_global_to_local_arr_inverts_the_affine(self):
        nii = _nii(np.zeros((6, 6, 6), dtype=np.uint8), zoom=(1.5, 2.0, 3.0), origin=(5.0, -2.0, 7.0))
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
        world = (nii.affine[:3, :3] @ points.T).T + nii.affine[:3, 3]
        np.testing.assert_allclose(nii.global_to_local_arr(world), points, atol=1e-9)

    def test_global_to_local_arr_agrees_with_the_scalar_version(self):
        nii = _nii(np.zeros((6, 6, 6), dtype=np.uint8), zoom=(1.5, 2.0, 3.0), origin=(5.0, -2.0, 7.0))
        world = np.array([[7.0, 1.0, 10.0], [5.0, -2.0, 7.0]])
        batched = nii.global_to_local_arr(world)
        for i, point in enumerate(world):
            with self.subTest(i=i):
                np.testing.assert_allclose(batched[i], nii.global_to_local(tuple(point)), atol=1e-9)

    def test_make_empty_nii_keeps_the_grid(self):
        nii = _nii(np.ones((4, 5, 6), dtype=np.uint8), zoom=(1.0, 2.0, 3.0))
        empty = nii.make_empty_nii()
        self.assertEqual(empty.shape, nii.shape)
        np.testing.assert_allclose(empty.affine, nii.affine)
        self.assertEqual(int(empty.get_array().sum()), 0)

    def test_to_gird_returns_a_grid_with_the_same_geometry(self):
        nii = _nii(np.zeros((4, 5, 6), dtype=np.uint8), zoom=(1.0, 2.0, 3.0))
        grid = nii.to_gird()
        self.assertIsInstance(grid, Grid)
        np.testing.assert_allclose(grid.affine, nii.affine)
        self.assertEqual(tuple(grid.shape), nii.shape)

    def test_grid_can_be_constructed_directly(self):
        nii = _nii(np.zeros((4, 5, 6), dtype=np.uint8), zoom=(1.0, 2.0, 3.0))
        grid = Grid(affine=nii.affine, shape=nii.shape)
        np.testing.assert_allclose(grid.affine, nii.affine)
        self.assertEqual(tuple(grid.shape), nii.shape)

    def test_split_4d_image_to_3d(self):
        arr = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        for i in range(3):
            arr[..., i] = i + 1
        parts = _nii(arr).split_4D_image_to_3D()
        self.assertEqual(len(parts), 3)
        for i, part in enumerate(parts):
            with self.subTest(i=i):
                self.assertEqual(part.shape, (4, 4, 4))
                self.assertEqual(int(part.get_seg_array().max()), i + 1)

    def test_reorient_same_as(self):
        a = _nii(np.zeros((4, 5, 6), dtype=np.uint8))
        b = a.reorient(("P", "I", "R"))
        self.assertEqual(a.reorient_same_as(b).orientation, b.orientation)
        c = a.copy()
        c.reorient_same_as_(b)
        self.assertEqual(c.orientation, b.orientation)


class TestMathMixin(unittest.TestCase):
    @staticmethod
    def _float_nii(seed: int = 0):
        rng = np.random.default_rng(seed)
        return _nii(rng.normal(10.0, 3.0, (6, 6, 6)).astype(np.float32), seg=False)

    def test_summary_statistics_match_numpy(self):
        nii = self._float_nii()
        arr = nii.get_array()
        self.assertAlmostEqual(nii.mean(), float(arr.mean()), places=4)
        self.assertAlmostEqual(nii.median(), float(np.median(arr)), places=4)
        self.assertAlmostEqual(nii.std(), float(arr.std()), places=4)

    def test_clamp(self):
        nii = self._float_nii()
        out = nii.clamp(min=9.0, max=11.0)
        self.assertGreaterEqual(float(out.get_array().min()), 9.0)
        self.assertLessEqual(float(out.get_array().max()), 11.0)
        copy = nii.copy()
        copy.clamp_(min=9.0, max=11.0)
        np.testing.assert_allclose(copy.get_array(), out.get_array())

    def test_nan_to_num(self):
        arr = np.array([[[1.0, np.nan], [np.inf, -np.inf]]], dtype=np.float32)
        out = _nii(arr, seg=False).nan_to_num(num=0).get_array()
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[0, 0, 1]), 0.0)

    def test_normalize_variants_stay_in_range(self):
        nii = self._float_nii(2)
        for name in ("normalize", "normalize_mri", "normalize_ct"):
            with self.subTest(method=name):
                out = getattr(nii, name)(min_out=0, max_out=1)
                arr = out.get_array()
                self.assertGreaterEqual(float(arr.min()), 0.0)
                self.assertLessEqual(float(arr.max()), 1.0)

    def test_normalize_inplace(self):
        nii = self._float_nii(3)
        nii.normalize_(min_out=0, max_out=10)
        self.assertLessEqual(float(nii.get_array().max()), 10.0)

    def test_dice_of_identical_masks_is_one(self):
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        nii = _nii(arr)
        self.assertAlmostEqual(nii.dice(nii.copy(), bar=False)[1], 1.0, places=6)

    def test_dice_of_disjoint_masks_is_zero(self):
        a = np.zeros((8, 8, 8), dtype=np.uint8)
        a[0:3] = 1
        b = np.zeros((8, 8, 8), dtype=np.uint8)
        b[5:8] = 1
        self.assertAlmostEqual(_nii(a).dice(_nii(b), bar=False)[1], 0.0, places=6)

    def test_betti_numbers_of_a_solid_block(self):
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:8, 2:8, 2:8] = 1
        betti = _nii(arr).betti_numbers()
        self.assertEqual(betti[1][0], 1, "one connected component")

    def test_reflected_operators(self):
        arr = np.full((3, 3, 3), 2.0, dtype=np.float32)
        nii = _nii(arr, seg=False)
        np.testing.assert_allclose((1 + nii).get_array(), arr + 1)
        np.testing.assert_allclose((1 - nii).get_array(), 1 - arr)

    def test_bit_shift_operators(self):
        nii = _nii(np.full((3, 3, 3), 4, dtype=np.int32))
        np.testing.assert_array_equal((nii << 1).get_array(), 8)
        np.testing.assert_array_equal((nii >> 1).get_array(), 2)

    def test_inplace_operators(self):
        arr = np.full((3, 3, 3), 8.0, dtype=np.float32)
        cases = {
            "isub": (lambda n: n.__isub__(3), 5.0),
            "itruediv": (lambda n: n.__itruediv__(2), 4.0),
            "ifloordiv": (lambda n: n.__ifloordiv__(3), 2.0),
            "imod": (lambda n: n.__imod__(3), 2.0),
            "ipow": (lambda n: n.__ipow__(2), 64.0),
        }
        for name, (op, expected) in cases.items():
            with self.subTest(op=name):
                nii = _nii(arr.copy(), seg=False)
                op(nii)
                np.testing.assert_allclose(nii.get_array(), expected)


class TestMeshExport(unittest.TestCase):
    def test_to_stl_writes_a_mesh_for_one_label(self):
        arr = np.zeros((12, 12, 12), dtype=np.uint8)
        arr[3:9, 3:9, 3:9] = 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "label.stl"
            mesh = _nii(arr).to_stl(1, out_path=out)
            self.assertTrue(out.exists())
            self.assertGreater(len(mesh.vectors), 0)

    def test_to_stls_writes_one_mesh_per_label(self):
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[2:7, 2:7, 2:7] = 1
        arr[9:14, 9:14, 9:14] = 2
        with tempfile.TemporaryDirectory() as tmp:
            meshes = _nii(arr).to_stls(out_path=Path(tmp) / "out.stl")
            self.assertEqual(sorted(meshes), [1, 2])
            # Regression: `number_path` (default True for a non-dict out_path) computed
            # out_path.with_name(...) without assigning it, so both labels were written
            # to out.stl and only the last one survived.
            self.assertEqual(sorted(p.name for p in Path(tmp).glob("*.stl")), ["out_1.stl", "out_2.stl"])

    def test_to_stls_accepts_a_per_label_path_dict(self):
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[2:7, 2:7, 2:7] = 1
        arr[9:14, 9:14, 9:14] = 2
        with tempfile.TemporaryDirectory() as tmp:
            paths = {1: Path(tmp) / "one.stl", 2: Path(tmp) / "two.stl"}
            _nii(arr).to_stls(out_path=paths)
            for label, path in paths.items():
                with self.subTest(label=label):
                    self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
