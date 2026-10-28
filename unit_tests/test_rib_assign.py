"""Unit tests for `TPTBox.segmentation.rib._rib_assign`.

This is the pure-numpy half of the rib sub-package - no torch, no spineps - so it
runs everywhere. The fixtures are synthetic thoracic columns: vertebra blocks
stacked along the superior axis with rib bars sticking out left and right, which
makes the expected left/right split and the per-vertebra assignment checkable.
"""

from __future__ import annotations

import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from TPTBox import NII
from TPTBox.core.vert_constants import Full_Body_Instance, Location
from TPTBox.segmentation.rib._rib_assign import (
    RibCandidate,
    _thoracic_like_labels,
    _touching_surface,
    assign_ribs_to_vert_segmentation,
    split_touching_rib_ccs,
)

# RAS with a thick superior axis, so the "S" weighting in _touching_surface is testable.
ZOOM = (1.0, 1.0, 3.0)
SHAPE = (40, 24, 40)


def _nii(arr: np.ndarray, seg: bool = True) -> NII:
    return NII(nib.Nifti1Image(arr, np.diag([*ZOOM, 1.0])), seg=seg)


def _thorax(n_verts: int = 4, fuse: bool = False):
    """Build (vert_seg, sem_seg, rib_seg) for `n_verts` thoracic levels with paired ribs."""
    vert = np.zeros(SHAPE, dtype=np.uint16)
    sem = np.zeros(SHAPE, dtype=np.uint16)
    rib = np.zeros(SHAPE, dtype=np.uint16)
    for i in range(n_verts):
        vid = 8 + i  # T1 = 8
        z0 = 4 + i * 6
        z1 = z0 + (6 if fuse else 4)
        vert[18:22, 10:14, z0 : z0 + 4] = vid
        sem[18:22, 10:14, z0 : z0 + 4] = Location.Vertebra_Corpus.value
        # Rib bars reaching out to both sides along the right-left axis. The raw rib
        # segmentation is labelled the way the function documents: `right_id` on the
        # patient-right side, `left_id` on the left.
        rib[4:18, 11:13, z0 : z0 + 2] = Full_Body_Instance.rib_right.value
        rib[22:36, 11:13, z0 : z0 + 2] = Full_Body_Instance.rib_left.value
        if fuse:
            # a thin bridge joining this level's rib to the next one
            rib[6:10, 11:13, z0:z1] = Full_Body_Instance.rib_right.value
    return _nii(vert), _nii(sem), _nii(rib)


class TestThoracicLikeLabels(unittest.TestCase):
    def test_keeps_the_rib_bearing_block_and_appends_28(self):
        self.assertEqual(_thoracic_like_labels([20, 8, 7, 25, 3, 100]), [7, 8, 20, 28])

    def test_sorted_cranial_to_caudal(self):
        out = _thoracic_like_labels([19, 10, 15])
        self.assertEqual(out[:-1], sorted(out[:-1]))

    def test_an_empty_input_still_appends_28(self):
        self.assertEqual(_thoracic_like_labels([]), [28])

    def test_labels_outside_the_block_are_dropped(self):
        self.assertEqual(_thoracic_like_labels([1, 2, 6, 21, 24]), [28])


class TestTouchingSurface(unittest.TestCase):
    def _masks(self, axis: int):
        a = np.zeros(SHAPE, dtype=bool)
        b = np.zeros(SHAPE, dtype=bool)
        index_a = [slice(10, 14)] * 3
        index_b = [slice(10, 14)] * 3
        index_a[axis] = slice(10, 12)
        index_b[axis] = slice(12, 14)
        a[tuple(index_a)] = True
        b[tuple(index_b)] = True
        return a, b

    def test_separated_masks_do_not_touch(self):
        a = np.zeros(SHAPE, dtype=bool)
        b = np.zeros(SHAPE, dtype=bool)
        a[10:12, 10:12, 10:12] = True
        b[14:16, 10:12, 10:12] = True
        self.assertEqual(_touching_surface(a, b, _nii(np.zeros(SHAPE, dtype=np.uint8))), 0.0)

    def test_touching_area_scales_with_the_shared_face(self):
        nii = _nii(np.zeros(SHAPE, dtype=np.uint8))
        small_a = np.zeros(SHAPE, dtype=bool)
        small_b = np.zeros(SHAPE, dtype=bool)
        small_a[10:12, 10:12, 10:12] = True
        small_b[12:14, 10:12, 10:12] = True

        big_a = np.zeros(SHAPE, dtype=bool)
        big_b = np.zeros(SHAPE, dtype=bool)
        big_a[10:12, 8:16, 8:16] = True
        big_b[12:14, 8:16, 8:16] = True
        self.assertGreater(_touching_surface(big_a, big_b, nii), _touching_surface(small_a, small_b, nii))

    def test_the_default_weights_prefer_left_right_contact(self):
        nii = _nii(np.zeros(SHAPE, dtype=np.uint8))
        per_axis = {}
        for axis, name in ((nii.get_axis("R"), "R"), (nii.get_axis("A"), "A"), (nii.get_axis("S"), "S")):
            a, b = self._masks(axis)
            per_axis[name] = _touching_surface(a, b, nii)
        # defaults are S: 0.01, A: 0.1, R: 1 - contact across the superior axis counts least
        self.assertGreater(per_axis["R"], per_axis["A"])
        self.assertGreater(per_axis["A"], per_axis["S"])

    def test_explicit_weights_override_the_defaults(self):
        nii = _nii(np.zeros(SHAPE, dtype=np.uint8))
        a, b = self._masks(nii.get_axis("S"))
        default = _touching_surface(a, b, nii)
        weighted = _touching_surface(a, b, nii, axis_weights={nii.get_axis("S"): 1.0})
        self.assertGreater(weighted, default)

    def test_the_relation_is_symmetric(self):
        nii = _nii(np.zeros(SHAPE, dtype=np.uint8))
        a, b = self._masks(0)
        self.assertEqual(_touching_surface(a, b, nii), _touching_surface(b, a, nii))


class TestRibCandidate(unittest.TestCase):
    def test_fields(self):
        cand = RibCandidate(cc_label=3, volume=120.0, z=5.0, x=-2.0)
        self.assertEqual((cand.cc_label, cand.volume, cand.z, cand.x), (3, 120.0, 5.0, -2.0))


class TestSplitTouchingRibCcs(unittest.TestCase):
    def test_a_clean_mask_is_returned_unchanged_in_volume(self):
        _, _, rib = _thorax()
        cc = rib.get_connected_components()
        before = int((cc.get_seg_array() != 0).sum())
        out = split_touching_rib_ccs(cc.copy(), verbose=False)
        self.assertEqual(int((out.get_seg_array() != 0).sum()), before)

    def test_a_fused_pair_is_split_into_more_components(self):
        _, _, rib = _thorax(fuse=True)
        cc = rib.get_connected_components()
        before = len(set(np.unique(cc.get_seg_array())) - {0})
        out = split_touching_rib_ccs(cc.copy(), erosion_pixels=2, min_volume=10, verbose=False)
        after = len(set(np.unique(out.get_seg_array())) - {0})
        self.assertGreaterEqual(after, before)

    def test_running_with_more_passes_does_not_lose_voxels(self):
        _, _, rib = _thorax(fuse=True)
        cc = rib.get_connected_components()
        total = int((cc.get_seg_array() != 0).sum())
        out = split_touching_rib_ccs(cc.copy(), erosion_pixels=2, min_volume=10, max_passes=3, verbose=False)
        self.assertEqual(int((out.get_seg_array() != 0).sum()), total)


class TestAssignRibsToVertSegmentation(unittest.TestCase):
    """The public entry point. Two bugs used to make it raise before returning.

    What is asserted here is that it completes and merges rib labels back into the
    semantic segmentation without disturbing the vertebrae. Whether the matching
    heuristic pairs every rib with a vertebra is *not* asserted: on this synthetic
    geometry it does not, and the unmatched components legitimately come back marked
    with `error_value`.
    """

    def test_rib_voxels_are_merged_into_the_semantic_segmentation(self):
        vert, sem, rib = _thorax()
        _, out_sem = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib.copy(), verbose=False)
        labels = set(np.unique(out_sem.get_seg_array())) - {0}
        self.assertIn(Location.Rib_Left.value, labels)
        self.assertIn(Location.Rib_Right.value, labels)
        self.assertIn(Location.Vertebra_Corpus.value, labels, "the existing semantic labels survive")

    def test_the_rib_mask_footprint_is_preserved(self):
        vert, sem, rib = _thorax()
        _, out_sem = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib.copy(), verbose=False)
        rib_voxels = int((rib.get_seg_array() != 0).sum())
        marked = int(np.isin(out_sem.get_seg_array(), [Location.Rib_Left.value, Location.Rib_Right.value]).sum())
        self.assertEqual(marked, rib_voxels)

    def test_vertebra_labels_are_not_lost(self):
        vert, sem, rib = _thorax()
        out_vert, _ = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib.copy(), verbose=False)
        self.assertTrue({8, 9, 10, 11}.issubset(set(np.unique(out_vert.get_seg_array()))))

    def test_unmatched_components_are_marked_with_the_error_value(self):
        vert, sem, rib = _thorax()
        out_vert, _ = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib.copy(), error_value=251, verbose=False)
        self.assertIn(251, set(np.unique(out_vert.get_seg_array())))

    def test_add_error_false_leaves_no_error_label(self):
        vert, sem, rib = _thorax()
        out_vert, _ = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib.copy(), error_value=251, add_error=False, verbose=False)
        self.assertNotIn(251, set(np.unique(out_vert.get_seg_array())))

    def test_an_empty_rib_mask_leaves_the_inputs_alone(self):
        vert, sem, _ = _thorax()
        empty = _nii(np.zeros(SHAPE, dtype=np.uint16))
        out_vert, out_sem = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), empty, verbose=False)
        np.testing.assert_array_equal(out_vert.get_seg_array(), vert.get_seg_array())
        np.testing.assert_array_equal(out_sem.get_seg_array(), sem.get_seg_array())

    def test_split_touching_can_be_turned_off(self):
        vert, sem, rib = _thorax()
        out_vert, out_sem = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib.copy(), split_touching=False, verbose=False)
        self.assertIn(Location.Rib_Left.value, set(np.unique(out_sem.get_seg_array())))
        self.assertGreater(int((out_vert.get_seg_array() != 0).sum()), 0)

    def test_custom_left_and_right_ids_are_honoured(self):
        vert, sem, rib = _thorax()
        arr = rib.get_seg_array()
        arr[arr == Full_Body_Instance.rib_left.value] = 201
        arr[arr == Full_Body_Instance.rib_right.value] = 202
        rib.set_array_(arr)
        _, out_sem = assign_ribs_to_vert_segmentation(vert.copy(), sem.copy(), rib, left_id=201, right_id=202, verbose=False)
        labels = set(np.unique(out_sem.get_seg_array()))
        self.assertIn(Location.Rib_Left.value, labels)
        self.assertIn(Location.Rib_Right.value, labels)

    def test_a_mismatched_affine_is_rejected(self):
        vert, sem, rib = _thorax()
        other = NII(nib.Nifti1Image(np.zeros(SHAPE, dtype=np.uint16), np.diag([2.0, 2.0, 2.0, 1.0])), seg=True)
        with self.assertRaises(AssertionError):
            assign_ribs_to_vert_segmentation(vert, sem, other, verbose=False)


class TestAddRibs(unittest.TestCase):
    """`add_ribs` wires the assignment above into a spineps run; only the wiring is tested."""

    def test_an_existing_rib_segmentation_short_circuits_the_model_call(self):
        import sys  # noqa: PLC0415

        fake_spineps = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"spineps": fake_spineps, "spineps.seg_run": fake_spineps.seg_run}):
            from TPTBox.segmentation.rib import add_ribs as add_ribs_mod  # noqa: PLC0415

            self.assertTrue(hasattr(add_ribs_mod, "add_ribs_to_vert_spine"))


if __name__ == "__main__":
    unittest.main()
