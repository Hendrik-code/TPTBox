"""Unit tests for `TPTBox.registration._deepali.polyrigid_reg.Polyrigid_Registration`.

Synthetic, CPU-only: two cuboid "vertebrae" that move by known translations, hand-made POIs, one landmark
outside the bone. Needs torch and hf-deepali.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import NII, POI, Location, Vertebra_Instance

_HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("deepali") is not None
if _HAS_TORCH:
    from TPTBox.registration._deepali.polyrigid_reg import Polyrigid_Registration

SHAPE = (48, 48, 48)
SUB = [Location.Vertebra_Corpus.value, Location.Spinosus_Process.value, Location.Arcus_Vertebrae.value]
LAND = 99  # arbitrary sub-region id used for the landmark


def _nii(arr: np.ndarray, zoom=(1.0, 1.0, 1.0)) -> NII:
    aff = np.diag([*zoom, 1.0])
    return NII(nib.Nifti1Image(arr.astype(np.int32), aff), seg=True)


def _cuboids(shift1=(0, 0, 0), shift2=(0, 0, 0)) -> np.ndarray:
    a = np.zeros(SHAPE, dtype=np.int32)
    s1, s2 = np.array(shift1), np.array(shift2)
    a[tuple(slice(8 + s, 16 + s) for s in s1)] = 1
    a[tuple(slice(8 + s, 16 + s) for s in s2 + np.array([0, 0, 20]))] = 2
    return a


def _pois(ref: NII, shift1, shift2, landmark, ids=(1, 2)) -> POI:
    """``landmark``: None or a list of positions (voxel coordinates) outside the bone."""
    poi = ref.make_empty_POI()
    base = {1: np.array([12.0, 12.0, 12.0]), 2: np.array([12.0, 12.0, 32.0])}
    offs = [np.array(o, dtype=float) for o in ([0, 0, 0], [3, 0, 0], [0, 3, 0])]
    for v in ids:
        sh = np.array(shift1 if v == 1 else shift2, dtype=float)
        for s, o in zip(SUB, offs):
            poi[(v, s)] = tuple(base[v] + o + sh)
    for n, lm in enumerate(landmark or []):
        poi[(1, LAND + n)] = tuple(np.array(lm, dtype=float))
    return poi


@unittest.skipUnless(_HAS_TORCH, "torch/hf-deepali not installed")
class TestPolyrigid(unittest.TestCase):
    def _make(self, s1=(2, 0, 0), s2=(0, 2, 1), zoom=(1.0, 1.0, 1.0), landmark=((30, 30, 20), (32, 20, 28)), **kw):
        fv, mv = _nii(_cuboids(), zoom), _nii(_cuboids(s1, s2), zoom)
        fp = _pois(fv, (0, 0, 0), (0, 0, 0), landmark)
        mp = _pois(mv, s1, s2, [tuple(np.array(lm) + d) for lm, d in zip(landmark, ([2, 1, 0], [1, -2, 1]))] if landmark else None)
        return (
            Polyrigid_Registration(fv, fv, mv, mv, fixed_poi=fp, moving_poi=mp, landmarks=landmark is not None, verbose=False, **kw),
            fv,
            mv,
        )

    def test_rigid_inside_and_landmark_exact(self):
        reg, *_ = self._make()
        with self.subTest("vertebrae move rigidly"):
            for v, r in reg.rigid_residual().items():
                self.assertLess(r, 1e-3, v)
        with self.subTest("landmark interpolated exactly"):
            self.assertEqual(len(reg.landmarks_fixed), 2)
            self.assertLess(reg.landmark_residual(), 1e-2)
        with self.subTest("no NaN in field (also in air)"):
            self.assertTrue(np.isfinite(reg.get_field()).all())

    def test_translation_recovered(self):
        reg, *_ = self._make()
        # L_1 = translation (+2,0,0) fixed -> moving
        i = reg.ids.index(1)
        np.testing.assert_allclose(reg.L[i, :, 1:], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(reg.L[i, :, 0], [2, 0, 0], atol=1e-6)

    def test_identity(self):
        reg, *_ = self._make(s1=(0, 0, 0), s2=(0, 0, 0), landmark=None)
        f = reg.get_field()
        x = reg._grid_world().reshape(f.shape)
        np.testing.assert_allclose(f, x, atol=1e-3)

    def test_no_landmarks_fallback(self):
        reg, *_ = self._make(landmark=None)
        self.assertEqual(len(reg.landmarks_fixed), 0)
        self.assertTrue(np.isfinite(reg.get_field()).all())

    def test_single_landmark_is_ignored(self):
        reg, *_ = self._make(landmark=((30, 30, 20),))
        self.assertEqual(len(reg.landmarks_fixed), 0)

    def test_min_dist_mm_excludes_landmarks_close_to_a_surface(self):
        # vertebra 1 spans voxels 8..15 (1 mm voxels). (18, 12, 12) is ~2.5 mm from its surface, (30, 30, 20) is far away.
        lm = ((18, 12, 12), (30, 30, 20))
        with self.subTest("small distance keeps both"):
            reg, *_ = self._make(landmark=lm, min_dist_mm=1.0)
            self.assertEqual(len(reg.landmarks_fixed), 2)
        with self.subTest("larger distance drops the close one -> single landmark is ignored"):
            reg, *_ = self._make(landmark=lm, min_dist_mm=4.0)
            self.assertEqual(len(reg.landmarks_fixed), 0)

    def test_reference_grid_overrides_crop(self):
        fv, mv = _nii(_cuboids()), _nii(_cuboids((2, 0, 0), (0, 2, 1)))
        ref = fv.apply_crop((slice(4, 40), slice(2, 36), slice(0, 48)))
        reg = Polyrigid_Registration(fv, None, mv, None, reference=ref, refine_independent=False, verbose=False)
        self.assertEqual(tuple(reg.ref.shape), (36, 34, 48))
        np.testing.assert_allclose(reg.ref.affine, ref.affine)
        self.assertTrue(np.isfinite(reg.get_field()).all())

    def test_landmark_inside_segmentation_is_filtered(self):
        reg, *_ = self._make(landmark=((12, 12, 12), (13, 13, 13)))  # inside vertebra 1
        self.assertEqual(len(reg.landmarks_fixed), 0)

    def _two_vertebrae_one_with_two_pois(self, shift2=(0, 0, 0)):
        fv, mv = _nii(_cuboids()), _nii(_cuboids(shift2=shift2))
        fp, mp = _pois(fv, (0, 0, 0), (0, 0, 0), None, ids=(1,)), _pois(mv, (0, 0, 0), shift2, None, ids=(1,))
        for p_, sh in ((fp, (0, 0, 0)), (mp, shift2)):  # vertebra 2: only 2 shared POIs
            p_[(2, SUB[0])] = tuple(np.array((12.0, 12.0, 32.0)) + sh)
            p_[(2, SUB[1])] = tuple(np.array((15.0, 12.0, 32.0)) + sh)
        return fv, mv, fp, mp

    def test_vertebra_with_too_few_pois_gets_translation_only(self):
        fv, mv, fp, mp = self._two_vertebrae_one_with_two_pois(shift2=(2, 0, 0))
        reg = Polyrigid_Registration(fv, fv, mv, mv, fixed_poi=fp, moving_poi=mp, refine_independent=False, verbose=False)
        self.assertEqual(reg.ids, [1, 2])
        self.assertEqual(reg.init_kinds, {1: "horn", 2: "points"})
        i = reg.ids.index(2)
        np.testing.assert_allclose(reg.L[i, :, 1:], np.eye(3), atol=1e-9)  # no rotation from 2 points
        np.testing.assert_allclose(reg.L[i, :, 0], [2, 0, 0], atol=1e-6)

    def test_init_poi_drops_vertebra_with_too_few_pois(self):
        fv, mv, fp, mp = self._two_vertebrae_one_with_two_pois()
        reg = Polyrigid_Registration(fv, fv, mv, mv, fixed_poi=fp, moving_poi=mp, init="poi", verbose=False)
        self.assertEqual(reg.ids, [1])

    def test_masks_only_translation_init(self):
        """No POIs and no sub-region masks: centroid translation, identity rotation, B = 0."""
        fv, mv = _nii(_cuboids()), _nii(_cuboids((2, 0, 0), (0, 2, 1)))
        reg = Polyrigid_Registration(fv, None, mv, None, refine_independent=False, verbose=False)
        self.assertEqual(reg.init_kinds, {1: "centroid", 2: "centroid"})
        np.testing.assert_allclose(reg.L[0, :, 0], [2, 0, 0], atol=1e-6)
        np.testing.assert_allclose(reg.L[1, :, 0], [0, 2, 1], atol=1e-6)
        np.testing.assert_allclose(reg.L[:, :, 1:], np.stack([np.eye(3)] * 2), atol=1e-9)
        self.assertEqual(len(reg.landmarks_fixed), 0)
        self.assertEqual(len(reg.B), 0)

    def test_default_ids_include_t13_and_sacrum_and_order_is_anatomical(self):
        from TPTBox import Vertebra_Instance
        from TPTBox.registration._deepali.polyrigid_reg import default_rigid_ids

        ids = default_rigid_ids()
        self.assertEqual(Vertebra_Instance.vertebra_label_without_sacrum()[-1], 28)
        for i in (28, 26, 27, 29):
            self.assertIn(i, ids)
        self.assertEqual(Polyrigid_Registration._anatomical([20, 28, 19, 1]), [1, 19, 28, 20])
        self.assertEqual(Polyrigid_Registration._anatomical([26, 25, 20]), [20, 25, 26])
        # T13 and the sacrum can be rigid objects without any extra argument; absent labels are ignored
        a = np.zeros(SHAPE, np.int32)
        a[8:16, 8:16, 8:16] = 28
        a[8:16, 8:16, 24:32] = 20
        a[8:16, 8:16, 40:48] = 26
        b = np.roll(a, 2, axis=0)
        reg = Polyrigid_Registration(_nii(a), None, _nii(b), None, refine_independent=False, verbose=False)
        self.assertEqual(reg.ids, [28, 20, 26])
        np.testing.assert_allclose(reg.L[reg.ids.index(26), :, 0], [2, 0, 0], atol=1e-6)

    def test_sacrum_only_in_one_image_is_not_a_rigid_object(self):
        a = np.zeros(SHAPE, np.int32)
        a[8:16, 8:16, 8:16] = 20
        b = a.copy()
        b[8:16, 8:16, 40:48] = 26
        reg = Polyrigid_Registration(_nii(a), None, _nii(b), None, refine_independent=False, verbose=False)
        self.assertEqual(reg.ids, [20])
        reg = Polyrigid_Registration(_nii(b), None, _nii(a), None, refine_independent=False, verbose=False)
        self.assertEqual(reg.ids, [20])

    def test_explicit_vert_ids_can_exclude_the_sacrum(self):
        a = np.zeros(SHAPE, np.int32)
        a[8:16, 8:16, 8:16] = 20
        a[8:16, 8:16, 40:48] = 26
        reg = Polyrigid_Registration(_nii(a), None, _nii(np.roll(a, 2, 0)), None, vert_ids=[20], refine_independent=False, verbose=False)
        self.assertEqual(reg.ids, [20])

    def test_anisotropic_spacing_world_consistent(self):
        # same physical situation: world coordinates identical -> residuals stay ~0
        reg, *_ = self._make(zoom=(1.0, 1.0, 1.0))
        self.assertLess(max(reg.rigid_residual().values()), 1e-3)

    def test_transform_nii_moves_image(self):
        reg, fv, mv = self._make(landmark=None)
        warped = reg.transform_nii(mv)
        # the moving vertebra 1 sits shifted by +2 in x; warping it back must overlap the fixed vertebra 1
        w, f = warped.get_seg_array() == 1, reg.ref.get_seg_array() == 1
        self.assertGreater((w & f).sum() / f.sum(), 0.99)

    def test_transform_poi_maps_fixed_to_moving(self):
        reg, fv, mv = self._make()
        fp = _pois(fv, (0, 0, 0), (0, 0, 0), None)
        out = reg.transform_poi(fp)
        np.testing.assert_allclose(out[(1, SUB[0])], (14.0, 12.0, 12.0), atol=1e-2)

    def test_save_load_roundtrip(self):
        reg, fv, mv = self._make()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "polyrigid.pkl"
            reg.save(p)
            reg2 = Polyrigid_Registration.load(p, mv, fv)
        np.testing.assert_allclose(reg2.get_field(), reg.get_field(), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
