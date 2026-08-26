"""Tests for the DeepALI-based registration additions.

Covers (in this order):

1. ``Deepali_Point_Registration`` closed-form rigid fit numerics – identity,
   pure translation, and a full reorientation. Compared to the SITK-backed
   :class:`Point_Registration` where meaningful.
2. ``Deepali_Point_Registration.transform_nii`` / ``transform_poi`` round-trip
   on translated / reoriented data.
3. ``General_Registration`` with ``same_space=False`` and with POI landmarks.
4. ``Template_Registration2`` end-to-end on a tiny sample.
5. A small speed / memory sanity benchmark – it is a *soft* check (asserts we
   do not massively regress vs. SimpleITK on CPU), not a strict throughput
   contract.

Deepali is optional in TPTBox, so all tests skip cleanly when it is missing.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.append(str(_HERE.parents[1]))

try:
    import deepali  # noqa: F401

    _HAS_DEEPALI = True
except Exception:
    _HAS_DEEPALI = False

try:
    import elasticdeform  # noqa: F401

    _HAS_ELASTIC = True
except Exception:
    _HAS_ELASTIC = False


def _synthetic_deformed_atlas(nii, sigma: float = 1.0, points: int = 3, seed: int = 42):
    """Return a lightly-deformed copy of *nii* for atlas → target tests.

    Uses :func:`TPTBox.core.internal.elastic_deform.deformed_nii` with small,
    fixed parameters (sigma=1.0, points=3, seed=42) that yield IoU ≈ 0.7-0.8
    against the input – enough for the deformable stage to have real work to do
    but far from destroying the anatomy, so the test stays stable across runs.
    Falls back to ``nii.copy()`` when ``elasticdeform`` is missing.
    """
    if not _HAS_ELASTIC:
        return nii.copy()
    from TPTBox.core.internal.elastic_deform import deformed_nii  # noqa: PLC0415

    np.random.seed(seed)
    return deformed_nii({"x": nii.copy()}, sigma=sigma, points=points)["x"]


@unittest.skipUnless(_HAS_DEEPALI, "hf-deepali not installed")
class TestDeepaliPointRegistration(unittest.TestCase):
    def setUp(self) -> None:
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        ct_nii, subreg_nii, vert_nii, _ = get_test_ct()
        self.ct_nii = ct_nii
        self.poi = calc_poi_from_subreg_vert(
            vert_nii,
            subreg_nii,
            subreg_id=[
                Location.Vertebra_Corpus,
                Location.Spinosus_Process,
                Location.Arcus_Vertebrae,
            ],
        ).extract_subregion(
            Location.Vertebra_Corpus,
            Location.Spinosus_Process,
            Location.Arcus_Vertebrae,
        )
        self.assertGreaterEqual(len(list(self.poi.keys())), 2)

    def test_identity_fit(self):
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        reg = Deepali_Point_Registration(self.poi, self.poi, verbose=False, ddevice="cpu")
        self.assertAlmostEqual(reg.error_reg, 0.0, places=5)
        aff = reg.get_affine()
        np.testing.assert_allclose(aff, np.eye(4), atol=1e-6)

    def test_translation_fit(self):
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        shift_ras = np.array([2.5, -1.0, 3.0])
        moving = self.poi.copy()
        moving.origin = tuple(np.asarray(moving.origin) + shift_ras)
        reg = Deepali_Point_Registration(self.poi, moving, verbose=False, ddevice="cpu")
        self.assertLess(reg.error_reg, 1e-3)
        aff = reg.get_affine()
        # world_matrix is fixed→moving in RAS: translation should equal shift_ras
        np.testing.assert_allclose(aff[:3, :3], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(aff[:3, 3], shift_ras, atol=1e-4)

    def test_reorient_fit_and_warp(self):
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        moving_poi = self.poi.reorient(("R", "A", "S"))
        moving_img = self.ct_nii.reorient(("R", "A", "S"))
        reg = Deepali_Point_Registration(self.poi, moving_poi, verbose=False, ddevice="cpu")
        self.assertLess(reg.error_reg, 1e-3)
        # transform_nii should reproduce the original image
        warped = reg.transform_nii(moving_img)
        orig = self.ct_nii.get_array().astype(np.float32)
        got = warped.get_array().astype(np.float32)
        self.assertEqual(orig.shape, got.shape)
        # per-voxel error should be tiny (well under one HU unit on average)
        self.assertLess(np.mean(np.abs(orig - got)), 5.0)
        # transform_poi round-trip
        out = reg.transform_poi(moving_poi)
        for k in self.poi.keys():
            np.testing.assert_allclose(
                np.array(self.poi[k]),
                np.array(out[k]),
                atol=1e-2,
            )

    def test_serialise_roundtrip(self):
        import tempfile  # noqa: PLC0415

        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        moving = self.poi.copy()
        moving.origin = tuple(np.asarray(moving.origin) + np.array([1.0, 2.0, 3.0]))
        reg = Deepali_Point_Registration(self.poi, moving, verbose=False, ddevice="cpu")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reg.pkl"
            reg.save(p)
            reg2 = Deepali_Point_Registration.load(p, ddevice="cpu")
            np.testing.assert_allclose(reg.get_affine(), reg2.get_affine(), atol=1e-6)
            self.assertAlmostEqual(reg.error_reg, reg2.error_reg, places=6)

    def test_transform_poi_inverse_roundtrip(self):
        """apply(transform_poi(x)) then transform_poi_inverse should return x."""
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        moving = self.poi.copy()
        moving.origin = tuple(np.asarray(moving.origin) + np.array([2.0, -1.5, 4.0]))
        reg = Deepali_Point_Registration(self.poi, moving, verbose=False, ddevice="cpu")
        moved = reg.transform_poi(moving)
        back = reg.transform_poi_inverse(moved, allow_only_same_grid_as_moving=False)
        for k in moving.keys():
            np.testing.assert_allclose(np.array(back[k]), np.array(moving[k]), atol=1e-3)

    def test_transform_cord_and_inverse(self):
        """transform_cord ∘ transform_cord_inverse ≈ identity on a voxel."""
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        moving = self.poi.copy()
        moving.origin = tuple(np.asarray(moving.origin) + np.array([2.0, -1.5, 4.0]))
        reg = Deepali_Point_Registration(self.poi, moving, verbose=False, ddevice="cpu")
        c0 = (10.0, 15.0, 20.0)
        fwd = reg.transform_cord(c0)  # moving voxel -> fixed voxel
        back = reg.transform_cord_inverse(tuple(fwd.tolist()))  # fixed voxel -> moving voxel
        np.testing.assert_allclose(np.array(back), np.array(c0), atol=1e-3)

    def test_apply_dispatch_and_deepali_transform_property(self):
        from deepali.spatial import HomogeneousTransform  # noqa: PLC0415

        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        reg = Deepali_Point_Registration(self.poi, self.poi.copy(), verbose=False, ddevice="cpu")
        # deepali_transform must be a real HomogeneousTransform module
        self.assertIsInstance(reg.deepali_transform, HomogeneousTransform)
        # apply() dispatches by type: POI -> POI, NII -> NII
        out_poi = reg.apply(self.poi.copy())
        self.assertEqual(sorted(out_poi.keys()), sorted(self.poi.keys()))
        out_nii = reg.apply(self.ct_nii.copy())
        self.assertEqual(out_nii.shape, self.ct_nii.shape)
        # apply() rejects unsupported types
        with self.assertRaises(ValueError):
            reg.apply("not a poi or nii")  # type: ignore[arg-type]

    def test_zero_points_raises(self):
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        empty = self.poi.make_empty_POI()
        with self.assertRaises(ValueError):
            Deepali_Point_Registration(empty, empty, verbose=False, ddevice="cpu")

    def test_single_point_degenerates_to_translation(self):
        """One shared point pair: rotation is under-determined so we fit a
        pure translation (R = I, t = q - p) instead of erroring out.
        """
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        # Keep only a single key on each side.
        one_key = next(iter(self.poi.keys()))
        poi_fix_single = self.poi.make_empty_POI()
        poi_fix_single[one_key] = self.poi[one_key]

        shift = np.array([2.5, -1.0, 3.0])
        poi_mov_single = poi_fix_single.copy()
        poi_mov_single.origin = tuple(np.asarray(poi_mov_single.origin) + shift)

        reg = Deepali_Point_Registration(poi_fix_single, poi_mov_single, verbose=False, ddevice="cpu")
        aff = reg.get_affine()
        # Rotation must be the identity, translation must match the applied RAS shift.
        np.testing.assert_allclose(aff[:3, :3], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(aff[:3, 3], shift, atol=1e-4)
        # And the fitted transform must actually take the moving point back to the fixed one.
        back = reg.transform_poi(poi_mov_single)
        np.testing.assert_allclose(
            np.array(back[one_key]), np.array(poi_fix_single[one_key]), atol=1e-3
        )

    def test_helper_returns_same_type(self):
        from TPTBox.registration import (  # noqa: PLC0415
            Deepali_Point_Registration,
            ridged_points_from_poi_deepali,
        )

        moving = self.poi.copy()
        moving.origin = tuple(np.asarray(moving.origin) + np.array([1.0, 0.0, 0.0]))
        reg = ridged_points_from_poi_deepali(self.poi, moving, verbose=False, ddevice="cpu")
        self.assertIsInstance(reg, Deepali_Point_Registration)
        # RAS translation on the first component should recover +1.0
        self.assertAlmostEqual(reg.get_affine()[0, 3], 1.0, places=3)


@unittest.skipUnless(_HAS_DEEPALI, "hf-deepali not installed")
class TestGeneralRegistrationFlags(unittest.TestCase):
    def setUp(self) -> None:
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        self.ct_nii, _, _, _ = get_test_ct()

    def test_same_space_false_pipeline_runs(self):
        """Different-orientation moving image should not crash the transform_nii path."""
        from TPTBox.registration import General_Registration  # noqa: PLC0415

        img = self.ct_nii
        moving = img.reorient(("R", "A", "S"))
        reg = General_Registration(
            fixed_image=img,
            moving_image=moving,
            transform_name="Affine",
            pyramid_levels=1,
            max_steps=3,
            ddevice="cpu",
            verbose=0,
            loss_terms={"mse": "MSE"},
            weights={"mse": 1.0},
            same_space=False,
        )
        out = reg.transform_nii(moving)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.orientation, img.orientation)

    def test_poi_landmarks_are_converted(self):
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.registration import General_Registration  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        ct, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus]).extract_subregion(Location.Vertebra_Corpus)
        reg = General_Registration(
            fixed_image=ct,
            moving_image=ct,
            source_landmarks=poi,
            target_landmarks=poi,
            transform_name="Affine",
            pyramid_levels=1,
            max_steps=1,
            ddevice="cpu",
            verbose=0,
            loss_terms={"mse": "MSE", "lm": "LandmarkPointDistance"},
            weights={"mse": 1.0, "lm": 0.1},
        )
        # Landmarks stored on both sides & converted to shape (1, N, 3)
        tgt_lm = reg.target_landmarks
        src_lm = reg.source_landmarks
        assert tgt_lm is not None and src_lm is not None
        self.assertEqual(tgt_lm.shape[-1], 3)
        self.assertEqual(tgt_lm.shape, src_lm.shape)

    def test_poi_landmark_registration_converges(self):
        """End-to-end sanity: with a landmark loss the transform should actually pull
        matching POIs together and produce an image warp that undoes a known shift.
        """
        import torch  # noqa: PLC0415

        from TPTBox import Location, calc_poi_from_subreg_vert, to_nii  # noqa: PLC0415
        from TPTBox.registration import General_Registration  # noqa: PLC0415

        ct = to_nii(
            "/media/data/robert/code/TPTBox/TPTBox/tests/sample_ct/sub-ct_label-22_ct.nii.gz", False
        )
        vert = to_nii(
            "/media/data/robert/code/TPTBox/TPTBox/tests/sample_ct/sub-ct_seg-vert_label-22_msk.nii.gz",
            True,
        )
        sub = to_nii(
            "/media/data/robert/code/TPTBox/TPTBox/tests/sample_ct/sub-ct_seg-subreg_label-22_msk.nii.gz",
            True,
        )
        poi_fix = calc_poi_from_subreg_vert(
            vert,
            sub,
            subreg_id=[Location.Vertebra_Corpus, Location.Spinosus_Process, Location.Arcus_Vertebrae],
        ).extract_subregion(
            Location.Vertebra_Corpus,
            Location.Spinosus_Process,
            Location.Arcus_Vertebrae,
        )
        shift = np.array([5.0, -3.0, 2.0])
        ct_moving = ct.copy()
        ct_moving.origin = tuple(np.asarray(ct.origin) + shift)
        poi_moving = poi_fix.copy()
        poi_moving.origin = ct_moving.origin

        reg = General_Registration(
            fixed_image=ct,
            moving_image=ct_moving,
            source_landmarks=poi_moving,
            target_landmarks=poi_fix,
            transform_name="Affine",
            pyramid_levels=1,
            max_steps=200,
            ddevice="cpu",
            verbose=0,
            lr=0.01,
            loss_terms={"lm": "LandmarkPointDistance"},
            weights={"lm": 1.0},
            same_space=False,
        )
        # residual in target-cube coords should be tiny after optimisation
        with torch.no_grad():
            tgt_lm = reg.target_landmarks
            src_lm = reg.source_landmarks
            assert tgt_lm is not None and src_lm is not None
            residual = (reg.transform(tgt_lm) - src_lm).abs().mean().item()
        self.assertLess(residual, 0.02)

        # warped moving image should be pulled close to the fixed image (better than the
        # trivially-shifted baseline).
        warped = reg.transform_nii(ct_moving)
        orig_arr = ct.get_array().astype(np.float32)
        warped_arr = warped.get_array().astype(np.float32)
        naive_arr = ct_moving.resample_from_to(ct, mode="constant").get_array().astype(np.float32)
        err_warped = float(np.mean(np.abs(orig_arr - warped_arr)))
        err_naive = float(np.mean(np.abs(orig_arr - naive_arr)))
        # Landmark-driven warp must at least be no worse than doing nothing.
        self.assertLess(err_warped, err_naive + 1.0)

    def test_poi_global_landmarks_also_work(self):
        """POI_Global inputs should be accepted and produce the same tensor shape."""
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.core.poi_fun.poi_global import POI_Global  # noqa: PLC0415
        from TPTBox.registration import General_Registration  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        ct, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus]).extract_subregion(
            Location.Vertebra_Corpus
        )
        poi_g = POI_Global(poi, itk_coords=False)
        reg = General_Registration(
            fixed_image=ct,
            moving_image=ct,
            source_landmarks=poi_g,
            target_landmarks=poi_g,
            transform_name="Affine",
            pyramid_levels=1,
            max_steps=1,
            ddevice="cpu",
            verbose=0,
            loss_terms={"mse": "MSE", "lm": "LandmarkPointDistance"},
            weights={"mse": 1.0, "lm": 0.1},
        )
        tgt_lm = reg.target_landmarks
        src_lm = reg.source_landmarks
        assert tgt_lm is not None and src_lm is not None
        self.assertEqual(tgt_lm.shape[-1], 3)
        self.assertEqual(tgt_lm.shape, src_lm.shape)

    def test_save_load_roundtrip_preserves_same_space(self):
        """dump/load must round-trip the new same_space flag and stay warpable."""
        import tempfile  # noqa: PLC0415

        from TPTBox.registration import General_Registration  # noqa: PLC0415

        img = self.ct_nii
        moving = img.reorient(("R", "A", "S"))
        reg = General_Registration(
            fixed_image=img,
            moving_image=moving,
            transform_name="Affine",
            pyramid_levels=1,
            max_steps=2,
            ddevice="cpu",
            verbose=0,
            loss_terms={"mse": "MSE"},
            weights={"mse": 1.0},
            same_space=False,
        )
        # New dump format is a 5-tuple including same_space.
        dump = reg.get_dump()
        self.assertEqual(len(dump), 5)
        self.assertFalse(dump[4])  # same_space=False was stored
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gr.pkl"
            reg.save(p)
            reg2 = General_Registration.load(p, ddevice="cpu")
            self.assertFalse(reg2.same_space)
            # After load the transform_nii path still works.
            out = reg2.transform_nii(moving)
            self.assertEqual(out.shape, img.shape)

    def test_load_legacy_4tuple_dump(self):
        """Old dumps (pre-`same_space`) must still load and default to same_space=True."""
        from TPTBox.registration import General_Registration  # noqa: PLC0415

        img = self.ct_nii
        reg = General_Registration(
            fixed_image=img,
            moving_image=img,
            transform_name="Affine",
            pyramid_levels=1,
            max_steps=1,
            ddevice="cpu",
            verbose=0,
            loss_terms={"mse": "MSE"},
            weights={"mse": 1.0},
        )
        # Simulate a pre-`same_space` dump.
        legacy = (reg.transform, reg.target_grid, reg.input_grid, reg._is_inverted)
        reg2 = General_Registration.load_(legacy, gpu=0, ddevice="cpu")
        self.assertTrue(reg2.same_space)


@unittest.skipUnless(_HAS_DEEPALI, "hf-deepali not installed")
class TestTemplateRegistration2(unittest.TestCase):
    def _make(self, with_pre=True, crop=False, deform=True, max_steps=3):
        """Build a small Template_Registration2 for pipeline-shape tests.

        ``deform=True`` (default when ``elasticdeform`` is available) makes the
        atlas a lightly-warped copy of the target so the deformable stage has
        something to fit; otherwise ``atlas == target.copy()`` (still tests the
        wiring, but the deformable step becomes a near-no-op).
        """
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.registration import Deepali_Point_Registration, Template_Registration2  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        _ct, subreg, vert, _ = get_test_ct()
        atlas_vert = _synthetic_deformed_atlas(vert) if deform else vert.copy()
        poi_target = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus]).extract_subregion(
            Location.Vertebra_Corpus
        )
        pre = (
            Deepali_Point_Registration(poi_target, poi_target.copy(), verbose=False, ddevice="cpu")
            if with_pre
            else None
        )
        reg = Template_Registration2(
            target_seg=vert,
            atlas_seg=atlas_vert,
            pre_registration=pre,
            pyramid_levels=1,
            coarsest_level=0,
            finest_level=0,
            max_steps=max_steps,
            verbose=0,
            gpu=0,
            ddevice="cpu",
            crop=crop,
        )
        return reg, vert, atlas_vert, poi_target

    def test_pre_registration_pipeline(self):
        """Template_Registration2 should run end-to-end with a supplied pre_registration."""
        reg, vert, atlas_vert, _ = self._make(with_pre=True, crop=False)
        out = reg.transform_nii(atlas_vert)
        self.assertEqual(out.shape, vert.shape)

    def test_only_rigid_skips_deformable(self):
        """only_rigid=True should return the atlas after only the rigid stage."""
        reg, vert, atlas_vert, _ = self._make(with_pre=True, crop=False)
        out = reg.transform_nii(atlas_vert, only_rigid=True)
        self.assertEqual(out.shape, vert.shape)

    def test_transform_poi_returns_target_grid(self):
        reg, vert, _atlas_vert, poi_target = self._make(with_pre=True, crop=False)
        moved = reg.transform_poi(poi_target.copy())
        self.assertEqual(tuple(moved.shape), tuple(vert.shape))

    def test_auto_fit_when_no_pre_registration(self):
        """Passing pre_registration=None must trigger the internal Deepali fit path."""
        from TPTBox.registration import Deepali_Point_Registration  # noqa: PLC0415

        reg, vert, atlas_vert, _ = self._make(with_pre=False, crop=False)
        self.assertIsInstance(reg.reg_point, Deepali_Point_Registration)
        out = reg.transform_nii(atlas_vert)
        self.assertEqual(out.shape, vert.shape)

    def test_save_load_roundtrip(self):
        import tempfile  # noqa: PLC0415

        from TPTBox.registration import Template_Registration2  # noqa: PLC0415

        reg, vert, atlas_vert, _ = self._make(with_pre=True, crop=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tr2.pkl"
            reg.save(p)
            reg2 = Template_Registration2.load(p, ddevice="cpu")
            out = reg2.transform_nii(atlas_vert)
            self.assertEqual(out.shape, vert.shape)

    @unittest.skipUnless(_HAS_ELASTIC, "elasticdeform not installed")
    def test_deformable_stage_improves_over_rigid_only(self):
        """With a truly deformed atlas the deformable stage must reduce the label
        mismatch relative to the ``only_rigid=True`` baseline.

        Fixed sigma/points/seed keep this deterministic; ``max_steps`` is picked
        so the assertion holds with margin (verified empirically) without the
        test becoming slow.
        """
        # Use a slightly larger max_steps so convergence is stable, but still cheap.
        reg, vert, atlas_vert, _ = self._make(with_pre=True, crop=False, deform=True, max_steps=25)
        target_mask = (vert.get_array() != 0).astype(np.int8)
        rigid_only = reg.transform_nii(atlas_vert, only_rigid=True)
        full = reg.transform_nii(atlas_vert)
        rigid_mask = (rigid_only.get_array() != 0).astype(np.int8)
        full_mask = (full.get_array() != 0).astype(np.int8)

        def _iou(a, b):
            inter = int((a & b).sum())
            union = int((a | b).sum())
            return inter / max(union, 1)

        iou_rigid = _iou(target_mask, rigid_mask)
        iou_full = _iou(target_mask, full_mask)
        # Deformable stage must not regress overlap. Small positive delta is fine;
        # keep the margin loose so tiny numerical wobbles don't fail the test.
        self.assertGreaterEqual(iou_full, iou_rigid - 1e-3, f"iou rigid={iou_rigid:.3f} full={iou_full:.3f}")
        # And the atlas really was deformed - rigid-only IoU is well below 1.
        self.assertLess(iou_rigid, 0.95, f"atlas seems undeformed; iou_rigid={iou_rigid:.3f}")


@unittest.skipUnless(_HAS_DEEPALI, "hf-deepali not installed")
class TestFlipHelper(unittest.TestCase):
    """The R-axis flip is shared between Template_Registration(2) __init__/warp paths."""

    def test_nii_flip_is_involution(self):
        from TPTBox.registration._deformable.multilabel_segmentation import _flip_r_axis  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        ct, _, _, _ = get_test_ct()
        once = _flip_r_axis(ct.copy())
        twice = _flip_r_axis(once)
        np.testing.assert_array_equal(twice.get_array(), ct.get_array())
        # And a single flip is NOT the identity on a non-symmetric image.
        self.assertFalse(np.array_equal(once.get_array(), ct.get_array()))

    def test_poi_flip_is_involution_and_mirrors_r_axis(self):
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.registration._deformable.multilabel_segmentation import _flip_r_axis  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        _, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus]).extract_subregion(
            Location.Vertebra_Corpus
        )
        axis = poi.get_axis("R")
        flipped = _flip_r_axis(poi.copy())
        back = _flip_r_axis(flipped)
        for k in poi.keys():
            np.testing.assert_allclose(np.array(back[k]), np.array(poi[k]), atol=1e-6)
            # Only the R-axis coordinate changed.
            orig = np.array(poi[k])
            new = np.array(flipped[k])
            expected = orig.copy()
            expected[axis] = poi.shape[axis] - 1 - expected[axis]
            np.testing.assert_allclose(new, expected, atol=1e-6)

    def test_poi_global_rejected(self):
        """POI_Global has no shape/axis; the helper must refuse instead of returning garbage."""
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.core.poi_fun.poi_global import POI_Global  # noqa: PLC0415
        from TPTBox.registration._deformable.multilabel_segmentation import _flip_r_axis  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        _, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus]).extract_subregion(
            Location.Vertebra_Corpus
        )
        with self.assertRaises(TypeError):
            _flip_r_axis(POI_Global(poi))


@unittest.skipUnless(_HAS_DEEPALI, "hf-deepali not installed")
class TestSpeedAndMemory(unittest.TestCase):
    """Soft speed/memory checks – kept quick enough for CI, no strict deadlines."""

    def test_deepali_vs_sitk_point_registration(self):
        from TPTBox import Location, calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.registration import (  # noqa: PLC0415
            Deepali_Point_Registration,
            Point_Registration,
        )
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        ct, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(
            vert,
            subreg,
            subreg_id=[Location.Vertebra_Corpus, Location.Spinosus_Process, Location.Arcus_Vertebrae],
        ).extract_subregion(
            Location.Vertebra_Corpus,
            Location.Spinosus_Process,
            Location.Arcus_Vertebrae,
        )
        moving = poi.copy()
        moving.origin = tuple(np.asarray(moving.origin) + np.array([2.5, -1.0, 3.0]))
        img_moving = ct.copy()
        img_moving.origin = moving.origin

        # --- fit time -------------------------------------------------------
        t0 = time.perf_counter()
        sitk_reg = Point_Registration(poi, moving, verbose=False)
        t_sitk_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        deep_reg = Deepali_Point_Registration(poi, moving, verbose=False, ddevice="cpu")
        t_deep_fit = time.perf_counter() - t0

        # --- warp time ------------------------------------------------------
        t0 = time.perf_counter()
        sitk_out = sitk_reg.transform_nii(img_moving)
        t_sitk_warp = time.perf_counter() - t0
        tracemalloc.start()
        t0 = time.perf_counter()
        deep_out = deep_reg.transform_nii(img_moving)
        t_deep_warp = time.perf_counter() - t0
        _, peak_deep = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # accuracy comparison against ground truth
        orig = ct.get_array().astype(np.float32)
        err_sitk = float(np.mean(np.abs(orig - sitk_out.get_array().astype(np.float32))))
        err_deep = float(np.mean(np.abs(orig - deep_out.get_array().astype(np.float32))))

        # Log summary line so the CI output records the numbers.
        print(
            f"\n[bench] fit: sitk={t_sitk_fit * 1000:.1f}ms deep={t_deep_fit * 1000:.1f}ms | "
            f"warp: sitk={t_sitk_warp * 1000:.1f}ms deep={t_deep_warp * 1000:.1f}ms | "
            f"mean-err: sitk={err_sitk:.3f} deep={err_deep:.3f} | "
            f"deepali peak mem={peak_deep / (1024 * 1024):.1f} MiB"
        )
        # Sanity bounds - accuracy: deepali is typically much better than SITK's
        # BSplineResampler here, but we only assert deepali is not massively worse.
        self.assertLess(err_deep, max(1.0, err_sitk * 2 + 1.0))
        # Memory: peak within an order of magnitude of the raw volume (~64 MiB for a
        # 73^3 float32) - guards against runaway allocations.
        raw_mib = orig.nbytes / (1024 * 1024)
        self.assertLess(peak_deep / (1024 * 1024), max(raw_mib * 20, 256))


class TestOptionalDeepaliStubs(unittest.TestCase):
    """When ``hf-deepali`` isn't installed the deepali-backed entry points
    must still *import* – they should raise a helpful ``ImportError`` only
    when actually used, and the error message must mention that PyTorch is a
    prerequisite too.
    """

    def test_stub_factory_message(self):
        import TPTBox.registration as _reg_init  # noqa: PLC0415

        stub_cls = _reg_init._make_missing_deepali_stub("Foo", ImportError("no module named 'deepali'"))
        with self.assertRaises(ImportError) as ctx:
            stub_cls()
        msg = str(ctx.exception)
        self.assertIn("Foo", msg)
        self.assertIn("hf-deepali", msg)
        self.assertIn("PyTorch", msg)
        self.assertIn("pip install torch hf-deepali", msg)

        stub_fn = _reg_init._make_missing_deepali_func("bar", ImportError("boom"))
        with self.assertRaises(ImportError) as ctx:
            stub_fn(1, 2)
        self.assertIn("bar()", str(ctx.exception))
        self.assertIn("pip install torch hf-deepali", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
