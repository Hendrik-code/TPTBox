"""Unit tests for `TPTBox.registration._deepali.polyrigid_optim` and the mask-based path of `Polyrigid_Registration`.

Synthetic, CPU-only. Needs torch and hf-deepali.
"""

from __future__ import annotations

import importlib.util
import unittest
from typing import ClassVar

import nibabel as nib
import numpy as np

from TPTBox import NII

_HAS_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("deepali") is not None
if _HAS_DEPS:
    import torch
    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation

    from TPTBox.registration._deepali.polyrigid_optim import (
        PolyrigidOptimConfig,
        fold_penalty,
        incompressibility_penalty,
        jacobian_det,
        rotation_from_axis_angle,
    )
    from TPTBox.registration._deepali.polyrigid_reg import Polyrigid_Registration

AFF = np.diag([2.0, 2.0, 2.0, 1.0])


def _nii(arr: np.ndarray) -> NII:
    return NII(nib.Nifti1Image(arr.astype(np.int32), AFF), seg=True)


def _move(arr: np.ndarray, label: int, angle_deg, shift) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/translate one label of ``arr`` about its own centroid. Returns the mask and the rotation matrix."""
    m = (arr == label).astype(np.float32)
    c = np.argwhere(m > 0).mean(0)
    r = Rotation.from_euler("xyz", angle_deg, degrees=True).as_matrix()
    inv = r.T
    return affine_transform(m, inv, offset=c - inv @ (c + np.asarray(shift, float)), order=0) > 0.5, r


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def _two_bodies_with_block():
    """Two rigid vertebrae (1, 2) with a non-rigid block (7) between them that rotate in opposite senses."""
    s = (48, 48, 64)
    fa = np.zeros(s, np.int32)
    fa[14:30, 14:30, 6:22] = 1
    fa[14:30, 14:30, 40:56] = 2
    fa[14:30, 14:30, 24:38] = 7
    m1, r1 = _move(fa, 1, (15, 0, 0), (0, 0, -2))
    m2, r2 = _move(fa, 2, (-15, 0, 0), (0, 0, 2))
    ma = np.zeros(s, np.int32)
    ma[m1], ma[m2] = 1, 2
    return fa, ma, (r1, r2)


@unittest.skipUnless(_HAS_DEPS, "torch/hf-deepali not installed")
class TestHelpers(unittest.TestCase):
    def test_rotation_from_axis_angle(self):
        w = torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5]], requires_grad=True)
        r = rotation_from_axis_angle(w)
        with self.subTest("identity at zero"):
            np.testing.assert_allclose(r[0].detach().numpy(), np.eye(3), atol=1e-7)
        with self.subTest("matches scipy rotation vector, orthonormal"):
            np.testing.assert_allclose(r[1].detach().numpy(), Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix(), atol=1e-6)
            self.assertAlmostEqual(float(torch.linalg.det(r[1].detach())), 1.0, places=5)
        with self.subTest("finite gradient at identity"):
            r.sum().backward()
            self.assertTrue(torch.isfinite(w.grad).all())

    def test_jacobian_det(self):
        zoom = (1.5, 2.0, 3.0)
        idx = torch.stack(
            torch.meshgrid(*[torch.arange(n, dtype=torch.float32) * z for n, z in zip((12, 14, 16), zoom)], indexing="ij"), -1
        )
        with self.subTest("scale field has det s^3, identity has 1"):
            np.testing.assert_allclose(jacobian_det(idx, zoom).numpy(), 1.0, atol=1e-5)
            np.testing.assert_allclose(jacobian_det(idx * 1.2, zoom).numpy(), 1.2**3, atol=1e-4)
        with self.subTest("equals deepali.core.flow.jacobian_det on an axis-aligned grid"):
            from deepali.core.flow import jacobian_det as ref_det

            f = idx + 0.2 * torch.sin(idx / 7) + 0.1 * torch.cos(idx.flip(-1) / 5)
            flow = (f - idx).flip(-1).permute(3, 0, 1, 2)[None]  # channels first, components x,y,z
            ref = ref_det(flow, spacing=tuple(reversed(zoom)))[0, 0]
            np.testing.assert_allclose(jacobian_det(f, zoom)[1:-1, 1:-1, 1:-1].numpy(), ref[1:-1, 1:-1, 1:-1].numpy(), atol=1e-5)

    def test_penalties(self):
        det = torch.tensor([1.0, 0.5, -0.3, 2.0])
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
        with self.subTest("fold penalty only counts det < eps inside the mask"):
            self.assertAlmostEqual(float(fold_penalty(det, mask, eps=0.1)), (0.4**2) / 3, places=6)
            self.assertEqual(float(fold_penalty(torch.ones(4), mask)), 0.0)
        with self.subTest("incompressibility is 0 for det = 1 and weighted per voxel"):
            w = torch.tensor([0.0, 1.0, 0.0, 2.0])
            self.assertEqual(float(incompressibility_penalty(torch.ones(4), w)), 0.0)
            expect = (np.log(0.5) ** 2 * 1.0 + np.log(2.0) ** 2 * 2.0) / 2
            self.assertAlmostEqual(float(incompressibility_penalty(det.clamp(0.5, 2.0), w)), expect, places=5)


@unittest.skipUnless(_HAS_DEPS, "torch/hf-deepali not installed")
class TestMaskRegistration(unittest.TestCase):
    CFG: ClassVar[dict] = {"levels_mm": (8.0, 4.0), "steps": (120, 80), "joint_steps": (80, 60), "blur_sigma_mm": (6.0, 2.0)}

    def _reg(self, **kw):
        fa, ma, rots = _two_bodies_with_block()
        fixed = _nii(fa)
        reg = Polyrigid_Registration(fixed, None, _nii(ma), None, verbose=False, crop_margin_vox=None, **kw)
        return reg, fa, rots

    def test_masks_only_recovers_rotation(self):
        """No POIs: centroid translation + identity rotation, then the independent stage finds the 15 degree rotations."""
        reg, _fa, (r1, r2) = self._reg(optim=PolyrigidOptimConfig(**self.CFG))
        self.assertEqual(reg.init_kinds, {1: "centroid", 2: "centroid"})
        with self.subTest("rotations"):
            self.assertLess(_angle_deg(reg.L[0][:, 1:], r1), 3.0)
            self.assertLess(_angle_deg(reg.L[1][:, 1:], r2), 3.0)
        with self.subTest("no landmarks -> B = 0, field finite"):
            self.assertEqual(len(reg.B), 0)
            self.assertTrue(np.isfinite(reg.get_field()).all())
        with self.subTest("config controls the working grids"):
            self.assertEqual([s["level_mm"] for s in reg.optim_log.stages], [8.0, 4.0])
            self.assertTrue(all(s["name"] == "independent" for s in reg.optim_log.stages))

    def test_refinement_is_optional(self):
        reg, *_ = self._reg(refine_independent=False)
        self.assertEqual(reg.optim_log.stages, [])
        np.testing.assert_allclose(reg.L[:, :, 1:], np.stack([np.eye(3)] * 2), atol=1e-9)

    def test_joint_stage_runs_and_does_not_fold(self):
        reg, fa, _ = self._reg(joint_refine=True, optim=PolyrigidOptimConfig(**self.CFG))
        self.assertEqual([s["name"] for s in reg.optim_log.stages], ["independent", "joint", "independent", "joint"])
        for s in reg.optim_log.stages:
            self.assertLessEqual(s["loss_best"], s["loss_first"] + 1e-6)
        det = jacobian_det(torch.as_tensor(reg.get_field()), reg.ref.zoom).numpy()
        self.assertEqual(float((det[fa == 0] <= 0).mean()), 0.0)

    def test_compressibility_weight_controls_volume_change_in_block(self):
        """The block (label 7) keeps its volume when it is incompressible and may change it when marked compressible."""

        def block_log_det(comp):
            cfg = PolyrigidOptimConfig(**self.CFG, w_incomp=5.0, compressibility=comp)
            reg, fa, _ = self._reg(joint_refine=True, optim=cfg)
            det = jacobian_det(torch.as_tensor(reg.get_field()), reg.ref.zoom).numpy()
            return float(np.abs(np.log(np.clip(det[fa == 7], 1e-3, None))).mean())

        stiff, free = block_log_det(None), block_log_det({7: 0.0})
        self.assertLess(stiff, 0.5 * free)

    def test_incompressibility_needs_no_other_structures(self):
        """Without non-rigid labelled structures the joint stage still runs (no incompressibility term)."""
        fa, ma, _ = _two_bodies_with_block()
        fa[fa == 7] = 0
        reg = Polyrigid_Registration(
            _nii(fa), None, _nii(ma), None, joint_refine=True, verbose=False, crop_margin_vox=None, optim=PolyrigidOptimConfig(**self.CFG)
        )
        self.assertIn("joint", [s["name"] for s in reg.optim_log.stages])


def _squashed_pair(factor: float = 0.7):
    """Fixed: two cuboid vertebrae (1 above 2). Moving: vertebra 1 shortened to ``factor`` of its height (a compression fracture)."""
    s = (48, 48, 64)
    fa = np.zeros(s, np.int32)
    fa[14:30, 14:30, 6:26] = 1
    fa[14:30, 14:30, 36:56] = 2
    ma = fa.copy()
    ma[ma == 1] = 0
    h = round(20 * factor)
    ma[14:30, 14:30, 16 - h // 2 : 16 - h // 2 + h] = 1
    return fa, ma


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1.0)


@unittest.skipUnless(_HAS_DEPS, "torch/hf-deepali not installed")
class TestShapeChange(unittest.TestCase):
    CFG: ClassVar[dict] = {"levels_mm": (8.0, 4.0), "steps": (150, 100), "joint_steps": (60, 60), "blur_sigma_mm": (6.0, 2.0)}

    def _reg(self, **kw):
        fa, ma = _squashed_pair()
        return (
            Polyrigid_Registration(
                _nii(fa), None, _nii(ma), None, verbose=False, crop_margin_vox=None, optim=PolyrigidOptimConfig(**self.CFG), **kw
            ),
            fa,
            ma,
        )

    def test_default_is_rigid(self):
        reg, *_ = self._reg()
        self.assertEqual(reg.shape_scales, {})
        for i in range(2):
            self.assertAlmostEqual(float(np.linalg.det(reg.L[i][:, 1:])), 1.0, places=4)

    def test_auto_keeps_only_the_squashed_vertebra_flexible(self):
        """Both vertebrae are fitted in step 1; the unchanged one is reset to rigid because its scale stays within the tolerance."""
        reg, *_ = self._reg(nonrigid_ids="auto")
        self.assertEqual(reg.shape_flagged, [1])
        self.assertEqual(list(reg.shape_scales), [1])
        self.assertAlmostEqual(float(np.linalg.det(reg.L[0][:, 1:])), reg.shape_scales[1][0] * reg.shape_scales[1][1] ** 3, delta=0.02)
        self.assertAlmostEqual(float(np.linalg.det(reg.L[1][:, 1:])), 1.0, places=4)

    def test_squashed_vertebra_is_fitted_then_stiff(self):
        rigid, fa, ma = self._reg()
        flex, _, _ = self._reg(nonrigid_ids=[1])
        with self.subTest("only the listed vertebra has a shape"):
            self.assertEqual(list(flex.shape_scales), [1])
            self.assertAlmostEqual(float(np.linalg.det(flex.L[1][:, 1:])), 1.0, places=4)
        with self.subTest("SI scale recovers the compression (fixed -> moving maps to 0.7 of the height)"):
            k, s = flex.shape_scales[1]
            self.assertAlmostEqual(k, 0.7, delta=0.12)
            self.assertAlmostEqual(s, 1.0, delta=0.12)
        with self.subTest("overlap of the squashed vertebra improves"):

            def warped_dice(reg):
                w = reg.transform_nii(_nii(ma), padding_value=0).get_seg_array()
                return _dice(w == 1, fa == 1)

            self.assertGreater(warped_dice(flex), warped_dice(rigid) + 0.05)
        with self.subTest("after freezing, the pose stages run rigidly (shape unchanged by them)"):
            both = self._reg(nonrigid_ids=[1], joint_refine=True)[0]
            self.assertEqual(both.shape_scales, both.optim_log.shape)
            self.assertAlmostEqual(both.shape_scales[1][0], flex.shape_scales[1][0], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
