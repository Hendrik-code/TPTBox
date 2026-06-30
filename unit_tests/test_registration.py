# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import io
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk  # noqa: N813

from TPTBox import NII, POI
from TPTBox.core import sitk_utils as su
from TPTBox.core.poi import calc_centroids, calc_poi_from_subreg_vert
from TPTBox.registration._ridged_points.point_registration import (
    Point_Registration,
    ridged_points_from_poi,
    ridged_points_from_subreg_vert,
)
from TPTBox.tests.test_utils import get_nii, get_poi, get_test_ct, repeats

has_deepali = False
try:
    import deepali

    has_deepali = True
except ModuleNotFoundError:
    has_deepali = False


def _quiet():
    """Return a fresh context manager that swallows ``stdout`` noise."""
    return redirect_stdout(io.StringIO())


def _translated_poi(poi: POI, shift=(3.0, -2.0, 4.0)) -> POI:
    """Return a copy of *poi* with every centroid translated by *shift* (voxel space)."""
    out = poi.copy()
    for k1, k2, (x, y, z) in poi.copy().items():
        out[k1, k2] = (x + shift[0], y + shift[1], z + shift[2])
    return out


def _make_mock_deform():
    """Build a MagicMock standing in for a ``Deformable_Registration`` instance."""
    inst = mock.MagicMock()
    inst.transform_nii.side_effect = lambda nii, *_, **__: nii
    inst.transform_poi.side_effect = lambda poi, *_, **__: poi
    inst.inverse.return_value.transform_poi.side_effect = lambda poi, *_, **__: poi
    inst.get_dump.return_value = (None, None, None, False)
    return inst


class Test_sitk_utils(unittest.TestCase):
    def test_nii_to_sitk_roundtrip(self):
        for i in range(repeats):
            with self.subTest(i=i):
                nii = get_nii()[0]
                img = su.nii_to_sitk(nii)
                self.assertEqual(len(img.GetSize()), 3)
                self.assertEqual(tuple(img.GetSize()), tuple(nii.shape))
                back = su.sitk_to_nii(img, seg=True)
                np.testing.assert_array_equal(nii.get_array(), back.get_array())
                self.assertTrue(np.allclose(nii.affine, back.affine))

    def test_nib_to_sitk_roundtrip(self):
        for i in range(repeats):
            with self.subTest(i=i):
                nii = get_nii()[0]
                img = su.nib_to_sitk(nii.nii)
                nib_back = su.sitk_to_nib(img)
                self.assertTrue(np.allclose(np.asarray(nii.affine), np.asarray(nib_back.affine)))
                np.testing.assert_array_equal(np.asarray(nii.get_array()), np.asarray(nib_back.dataobj))

    def test_affine_metadata_helpers(self):
        for i in range(repeats):
            with self.subTest(i=i):
                nii = get_nii()[0]
                affine = nii.affine
                origin, spacing, direction = su.get_sitk_metadata_from_ras_affine(affine)
                self.assertEqual(len(origin), 3)
                self.assertEqual(len(spacing), 3)
                self.assertEqual(len(direction), 9)
                rotation, sp = su.get_rotation_and_spacing_from_affine(affine)
                self.assertEqual(rotation.shape, (3, 3))
                self.assertTrue((np.asarray(sp) > 0).all())
                img = su.nii_to_sitk(nii)
                rec = su.get_ras_affine_from_sitk(img)
                self.assertTrue(np.allclose(affine, rec))
                rec2 = su.get_ras_affine_from_sitk_meta(img.GetSpacing(), img.GetDirection(), img.GetOrigin())
                self.assertTrue(np.allclose(affine, rec2))

    def test_ras_affine_dimensionality_edge_cases(self):
        # 2-D image -> 4-element direction; 4-D image -> 16-element direction.
        img2d = sitk.Image([8, 9], sitk.sitkFloat32)
        self.assertEqual(su.get_ras_affine_from_sitk(img2d).shape, (4, 4))
        img4d = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.assertEqual(su.get_ras_affine_from_sitk(img4d).shape, (4, 4))
        # Same edge cases via the explicit-metadata helper.
        self.assertEqual(su.get_ras_affine_from_sitk_meta((1.0, 1.0), (1, 0, 0, 1), (5.0, 6.0)).shape, (4, 4))
        self.assertEqual(
            su.get_ras_affine_from_sitk_meta((1.0, 1.0, 1.0, 1.0), tuple(np.eye(4).flatten()), (0.0, 0.0, 0.0, 0.0)).shape,
            (4, 4),
        )
        with pytest.raises(NotImplementedError):
            su.get_ras_affine_from_sitk_meta((1.0, 1.0, 1.0), (1, 2, 3), (0.0, 0.0, 0.0))

    def test_transform_centroid_known_bug(self):
        # transform_centroid ends with ``nii.get_empty_POI(out)`` but NII exposes
        # ``make_empty_POI`` -> the call always raises AttributeError. We exercise
        # both branches (rigid + deformable) to cover the function body.
        poi = get_poi(num_vert=3, min_subreg=50, max_subreg=50, rotation=False)
        img = su.nii_to_sitk(poi.make_empty_nii())
        with pytest.raises(AttributeError):
            su.transform_centroid(poi, sitk.VersorRigid3DTransform(), img, img, "rigid")
        with pytest.raises(AttributeError):
            su.transform_centroid(poi, sitk.TranslationTransform(3, [0.0, 0.0, 0.0]), img, img, "deformable")


class Test_point_registration(unittest.TestCase):
    def _build(self, num_vert=6, shift=(3.0, -2.0, 4.0)):
        poi_fixed = get_poi(x=(50, 40, 30), num_vert=num_vert, num_subreg=1, rotation=False, min_subreg=50, max_subreg=50)
        poi_moving = _translated_poi(poi_fixed, shift)
        with _quiet():
            reg = ridged_points_from_poi(poi_fixed, poi_moving, verbose=False)
        return reg, poi_fixed, poi_moving

    def test_translation_recovery(self):
        for i in range(5):
            with self.subTest(i=i):
                reg, pf, pm = self._build()
                self.assertAlmostEqual(reg.error_reg, 0.0, places=3)
                self.assertAlmostEqual(reg.error_natural, 0.0, places=3)
                out = reg.transform_poi(pm)
                for k1, k2, coord in out.items():
                    self.assertTrue(np.allclose(coord, pf[k1, k2], atol=1e-2))

    def test_get_affine(self):
        reg, _pf, _pm = self._build()
        affine = reg.get_affine()
        self.assertEqual(affine.shape, (4, 4))
        # A pure translation -> identity rotation block.
        self.assertTrue(np.allclose(affine[:3, :3], np.eye(3), atol=1e-3))

    def test_transform_dispatch(self):
        reg, _pf, pm = self._build()
        with _quiet():
            via_poi = reg.transform(pm)
            via_nii = reg.transform(pm.make_empty_nii(seg=True))
        self.assertIsInstance(via_poi, POI)
        self.assertIsInstance(via_nii, NII)
        with pytest.raises(ValueError):
            reg.transform(42)

    def test_transform_nii(self):
        reg, _pf, pm = self._build()
        seg = pm.make_empty_nii(seg=True)
        arr = seg.get_array()
        arr[5:10, 5:10, 5:10] = 1
        seg = seg.set_array(arr)
        with _quiet():
            out_seg = reg.transform_nii(seg)  # c_val=None -> derived
            out_img = reg.transform_nii(pm.make_empty_nii(seg=False), c_val=0.0)
        self.assertEqual(out_seg.shape, reg.out_poi.shape_int)
        self.assertTrue(out_seg.seg)
        self.assertEqual(out_img.shape, reg.out_poi.shape_int)

    def test_resamplers(self):
        reg, _pf, _pm = self._build()
        self.assertIsInstance(reg.get_resampler(True, 0.0), sitk.ResampleImageFilter)
        self.assertIsInstance(reg.get_resampler(False, -1000.0), sitk.ResampleImageFilter)

    def test_inverse_roundtrip(self):
        reg, _pf, pm = self._build()
        fwd = reg.transform_poi(pm)
        back = reg.transform_poi_inverse(fwd)
        for k1, k2, coord in back.items():
            self.assertTrue(np.allclose(coord, pm[k1, k2], atol=1e-2))

    def test_get_dump_and_load_(self):
        reg, _pf, _pm = self._build()
        dump = reg.get_dump()
        self.assertEqual(dump[0], 1)
        self.assertEqual(len(dump), 8)
        reg2 = Point_Registration.load_(reg.get_dump())
        self.assertTrue(np.allclose(reg2.get_affine(), reg.get_affine()))

    def test_save_load(self):
        reg, _pf, pm = self._build()
        with TemporaryDirectory() as td:
            path = Path(td) / "reg.pkl"
            reg.save(path)
            reg2 = Point_Registration.load(path)
        o1 = reg.transform_poi(pm)
        o2 = reg2.transform_poi(pm)
        for k1, k2, coord in o1.items():
            self.assertTrue(np.allclose(coord, o2[k1, k2], atol=1e-6))

    def test_exclusion(self):
        poi_fixed = get_poi(x=(50, 40, 30), num_vert=4, rotation=False, min_subreg=50, max_subreg=50)
        poi_moving = _translated_poi(poi_fixed)
        with _quiet():
            reg = ridged_points_from_poi(poi_fixed, poi_moving, exclusion=[4], verbose=True)
        self.assertIsInstance(reg, Point_Registration)

    def test_leave_worst_percent_out(self):
        poi_fixed = get_poi(x=(50, 40, 30), num_vert=6, rotation=False, min_subreg=50, max_subreg=50)
        poi_moving = _translated_poi(poi_fixed)
        with _quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = ridged_points_from_poi(poi_fixed, poi_moving, leave_worst_percent_out=0.3, verbose=False)
        self.assertIsInstance(reg, Point_Registration)

    def test_ax_code_and_zooms(self):
        poi_fixed = get_poi(x=(50, 40, 30), num_vert=4, rotation=False, min_subreg=50, max_subreg=50)
        poi_moving = _translated_poi(poi_fixed)
        with _quiet():
            reg = ridged_points_from_poi(poi_fixed, poi_moving, ax_code=("P", "I", "R"), zooms=(2, 2, 2), verbose=False)
        self.assertIsInstance(reg, Point_Registration)

    def test_too_few_points_raises(self):
        poi_fixed = get_poi(x=(50, 40, 30), num_vert=2, min_subreg=50, max_subreg=50)
        poi_moving = get_poi(x=(50, 40, 30), num_vert=2, min_subreg=88, max_subreg=88)
        with _quiet(), pytest.raises(ValueError):
            ridged_points_from_poi(poi_fixed, poi_moving, verbose=False)

    def test_c_val_deprecation(self):
        poi_fixed = get_poi(x=(50, 40, 30), num_vert=3, rotation=False, min_subreg=50, max_subreg=50)
        poi_moving = _translated_poi(poi_fixed)
        with _quiet(), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ridged_points_from_poi(poi_fixed, poi_moving, c_val=-1000, verbose=False)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_ridged_points_from_subreg_vert(self):
        _ct, subreg, vert, _idx = get_test_ct()
        moving = calc_poi_from_subreg_vert(vert, subreg, subreg_id=50).extract_subregion_(50)
        with _quiet():
            reg = ridged_points_from_subreg_vert(moving, vert, subreg, subreg_id=50, verbose=False, save_buffer_file=False)
        self.assertIsInstance(reg, Point_Registration)
        self.assertTrue(np.allclose(reg.get_affine()[:3, :3], np.eye(3), atol=1e-3))


@unittest.skipIf(not has_deepali, "requires deepali to be installed")
class Test_deepali_model(unittest.TestCase):
    def _build(self, inverted=False, x=(20, 24, 28)):
        import torch

        from TPTBox.registration._deepali.deepali_model import General_Registration

        nii = get_nii(x=x)[0]
        reg = General_Registration.load_((torch.zeros(1), nii.to_gird(), nii.to_gird(), inverted), gpu=0, ddevice="cpu")
        return reg, nii

    def test_center_of_mass(self):
        import torch

        import TPTBox.registration._deepali.deepali_model as dm

        com = dm.center_of_mass(torch.ones(4, 4, 4))
        self.assertTrue(np.allclose(com.numpy(), [1.5, 1.5, 1.5]))

    def test_time_it(self):
        import TPTBox.registration._deepali.deepali_model as dm

        @dm.time_it
        def add(a, b):
            return a + b

        with _quiet():
            self.assertEqual(add(2, 3), 5)

    def test_load_config(self):
        import json

        import TPTBox.registration._deepali.deepali_model as dm

        with TemporaryDirectory() as td:
            (Path(td) / "c.json").write_text(json.dumps({"a": 1}))
            (Path(td) / "c.yaml").write_text("b: 2\n")
            self.assertEqual(dm._load_config(Path(td) / "c.json"), {"a": 1})
            self.assertEqual(dm._load_config(Path(td) / "c.yaml"), {"b": 2})

    def test_load_and_dump(self):
        reg, _nii = self._build()
        dump = reg.get_dump()
        self.assertEqual(len(dump), 4)
        self.assertFalse(dump[3])
        self.assertEqual(str(reg.device), "cpu")

    def test_save_load(self):
        from TPTBox.registration._deepali.deepali_model import General_Registration

        reg, _nii = self._build()
        with TemporaryDirectory() as td:
            path = Path(td) / "g.pkl"
            reg.save(path)
            reg2 = General_Registration.load(path, gpu=0, ddevice="cpu")
        self.assertEqual(reg2._is_inverted, reg._is_inverted)
        self.assertEqual(reg2.target_grid.shape_int, reg.target_grid.shape_int)

    def test_inverse(self):
        from TPTBox.registration._deepali.deepali_model import General_Registration

        reg, _nii = self._build()
        inv = reg.inverse()
        self.assertIsInstance(inv, General_Registration)
        self.assertIsNot(inv, reg)
        self.assertNotEqual(inv._is_inverted, reg._is_inverted)

    def test_transform_nii_and_call(self):
        import torch

        import TPTBox.registration._deepali.deepali_model as dm

        reg, nii = self._build()
        tshape = reg.target_grid.shape_int

        def fake(*_, **__):
            return torch.zeros(tuple(tshape[::-1]))

        with mock.patch.object(dm, "_warp_image", side_effect=fake), _quiet():
            out = reg.transform_nii(nii.copy(), ddevice="cpu")
            called = reg(nii.copy(), ddevice="cpu")  # __call__
        self.assertEqual(out.shape, tshape)
        self.assertEqual(called.shape, tshape)

    def test_transform_nii_inverted(self):
        import torch

        import TPTBox.registration._deepali.deepali_model as dm

        reg, nii = self._build(inverted=True)
        tshape = reg.target_grid.shape_int
        with mock.patch.object(dm, "_warp_image", side_effect=lambda *_, **__: torch.zeros(tuple(tshape[::-1]))), _quiet():
            out = reg.transform_nii(nii.copy(), ddevice="cpu", inverse=True)
        self.assertEqual(out.shape, tshape)

    def test_transform_poi(self):
        import TPTBox.registration._deepali.deepali_model as dm

        for inverted in (False, True):
            with self.subTest(inverted=inverted):
                reg, _nii = self._build(inverted=inverted)
                poi = reg.target_grid.make_empty_POI({1: {50: (5.0, 6.0, 7.0)}, 2: {50: (8.0, 9.0, 10.0)}})
                with mock.patch.object(dm, "_warp_poi", side_effect=lambda *a, **__: a[0]), _quiet():
                    out = reg.transform_poi(poi, ddevice="cpu")
                self.assertEqual(set(out.keys()), {(1, 50), (2, 50)})

    def test_transform_points(self):
        import torch
        from deepali.core import Axes

        import TPTBox.registration._deepali.deepali_model as dm

        reg, _nii = self._build()
        pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        captured = {}

        def cap(*a, **__):
            captured["grid"] = type(a[3]).__name__
            return torch.tensor(a[0])

        with mock.patch.object(dm, "_warp_points", side_effect=cap):
            out = reg.transform_points(pts, Axes.GRID, Axes.GRID, reg.target_grid, reg.input_grid, ddevice="cpu")
        self.assertEqual(tuple(out.shape), (2, 3))
        # Has_Grid args are converted to deepali Grid before reaching _warp_points.
        self.assertEqual(captured["grid"], "Grid")

        # ``_is_inverted`` toggles the inverse flag inside transform_points.
        reg_inv, _ = self._build(inverted=True)
        with mock.patch.object(dm, "_warp_points", side_effect=lambda *a, **__: torch.tensor(a[0])):
            out_inv = reg_inv.transform_points(pts, Axes.GRID, Axes.GRID, reg_inv.target_grid, reg_inv.input_grid, ddevice="cpu")
        self.assertEqual(tuple(out_inv.shape), (2, 3))


@unittest.skipIf(not has_deepali, "requires deepali to be installed")
class Test_deformable_reg(unittest.TestCase):
    def _images(self):
        return get_nii(x=(16, 18, 20))[0], get_nii(x=(16, 18, 20))[0]

    def test_default_wiring(self):
        from TPTBox.registration._deepali.deepali_model import General_Registration
        from TPTBox.registration._deformable.deformable_reg import Deformable_Registration

        fixed, moving = self._images()
        with mock.patch.object(General_Registration, "__init__", return_value=None) as m:
            Deformable_Registration(fixed, moving, auto_run=False)
        kw = m.call_args.kwargs
        self.assertEqual(set(kw["loss_terms"].keys()), {"be", "lncc"})
        self.assertEqual(kw["weights"], {"be": 0.001, "lncc": 1})
        self.assertEqual(kw["transform_args"], {"stride": [8, 8, 8], "transpose": False})
        self.assertEqual(kw["transform_name"], "SVFFD")
        self.assertEqual(kw["lr"], 0.001)
        self.assertEqual(kw["max_steps"], 1000)
        self.assertEqual(kw["pyramid_levels"], 3)
        self.assertFalse(kw["auto_run"])

    def test_svf_transpose_pop(self):
        from TPTBox.registration._deepali.deepali_model import General_Registration
        from TPTBox.registration._deformable.deformable_reg import Deformable_Registration

        fixed, moving = self._images()
        with mock.patch.object(General_Registration, "__init__", return_value=None) as m:
            Deformable_Registration(
                fixed, moving, auto_run=False, transform_name="SVF", transform_args={"stride": [4, 4, 4], "transpose": True}
            )
        # For SVF-family transforms the "transpose" key is dropped.
        self.assertEqual(m.call_args.kwargs["transform_args"], {"stride": [4, 4, 4]})

    def test_explicit_loss_terms(self):
        from TPTBox.registration._deepali.deepali_model import General_Registration
        from TPTBox.registration._deformable.deformable_reg import Deformable_Registration

        fixed, moving = self._images()
        with mock.patch.object(General_Registration, "__init__", return_value=None) as m:
            Deformable_Registration(fixed, moving, auto_run=False, loss_terms=["lncc"], weights=[1.0])
        kw = m.call_args.kwargs
        self.assertEqual(kw["loss_terms"], ["lncc"])
        self.assertEqual(kw["weights"], [1.0])


@unittest.skipIf(not has_deepali, "requires deepali to be installed")
class Test_template_registration(unittest.TestCase):
    def _segs(self):
        target = get_nii(x=(40, 44, 48), num_point=4)[0]
        return target, target.copy()

    def test_construct_transform_same_side(self):
        import TPTBox.registration._deformable.multilabel_segmentation as mls
        from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

        target, atlas = self._segs()
        with mock.patch.object(mls, "Deformable_Registration") as MD:
            MD.return_value = _make_mock_deform()
            with _quiet():
                tr = Template_Registration(target, atlas, same_side=True, crop=True, verbose=0, ddevice="cpu")
                out = tr.transform_nii(atlas.copy())
                out_rigid = tr.transform_nii(atlas.copy(), only_rigid=True)
                poi = calc_centroids(atlas, second_stage=40)
                pout = tr.transform_poi(poi)
        self.assertTrue(MD.called)
        self.assertIsNotNone(tr.crop)
        self.assertIsInstance(out, NII)
        self.assertEqual(out.shape, tr.target_grid_org.shape_int)
        self.assertIsInstance(out_rigid, NII)
        self.assertEqual(set(pout.keys()), set(poi.keys()))

    def test_flip_same_side_false(self):
        import TPTBox.registration._deformable.multilabel_segmentation as mls
        from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

        target, atlas = self._segs()
        with mock.patch.object(mls, "Deformable_Registration") as MD:
            MD.return_value = _make_mock_deform()
            with _quiet():
                tr = Template_Registration(target, atlas, same_side=False, crop=True, verbose=0, ddevice="cpu")
                out = tr.transform_nii(atlas.copy())
                poi = calc_centroids(atlas, second_stage=40)
                pout = tr.transform_poi(poi)
        self.assertFalse(tr.same_side)
        self.assertEqual(out.shape, tr.target_grid_org.shape_int)
        self.assertEqual(set(pout.keys()), set(poi.keys()))

    def test_crop_false_and_inverse(self):
        import TPTBox.registration._deformable.multilabel_segmentation as mls
        from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

        target, atlas = self._segs()
        with mock.patch.object(mls, "Deformable_Registration") as MD:
            MD.return_value = _make_mock_deform()
            with _quiet():
                tr = Template_Registration(target, atlas, same_side=True, crop=False, verbose=0, ddevice="cpu")
                poi = calc_centroids(target, second_stage=40)
                inv = tr.transform_poi_inverse(poi)
        self.assertIsNone(tr.crop)
        self.assertEqual(set(inv.keys()), set(poi.keys()))

    def test_with_images_and_pois(self):
        import TPTBox.registration._deformable.multilabel_segmentation as mls
        from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

        target, atlas = self._segs()
        target_img = target.copy()
        target_img.seg = False
        atlas_img = atlas.copy()
        atlas_img.seg = False
        poi_target = calc_centroids(target, second_stage=40)
        poi_atlas = calc_centroids(atlas, second_stage=40)
        with mock.patch.object(mls, "Deformable_Registration") as MD:
            MD.return_value = _make_mock_deform()
            with _quiet():
                tr = Template_Registration(
                    target,
                    atlas,
                    target_img=target_img,
                    atlas_img=atlas_img,
                    poi_cms=poi_atlas,
                    poi_target_cms=poi_target,
                    same_side=True,
                    crop=True,
                    verbose=0,
                    ddevice="cpu",
                )
        self.assertIsNotNone(tr.crop)

    def test_poi_provided_flip_and_inverse(self):
        import TPTBox.registration._deformable.multilabel_segmentation as mls
        from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

        target, atlas = self._segs()
        poi_target = calc_centroids(target, second_stage=40)
        poi_atlas = calc_centroids(atlas, second_stage=40)
        with mock.patch.object(mls, "Deformable_Registration") as MD:
            MD.return_value = _make_mock_deform()
            with _quiet():
                tr = Template_Registration(
                    target, atlas, poi_cms=poi_atlas, poi_target_cms=poi_target, same_side=False, crop=False, verbose=0, ddevice="cpu"
                )
                inv = tr.transform_poi_inverse(calc_centroids(target, second_stage=40))
        self.assertFalse(tr.same_side)
        self.assertEqual(set(inv.keys()), set(poi_target.keys()))

    def test_dump_save_load(self):
        import TPTBox.registration._deformable.multilabel_segmentation as mls
        from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

        target, atlas = self._segs()
        with mock.patch.object(mls, "Deformable_Registration") as MD:
            MD.return_value = _make_mock_deform()
            MD.load_ = mock.MagicMock(return_value=_make_mock_deform())
            with _quiet():
                tr = Template_Registration(target, atlas, same_side=True, crop=True, verbose=0, ddevice="cpu")
                dump = tr.get_dump()
                self.assertEqual(dump[0], 1)
                with TemporaryDirectory() as td:
                    path = Path(td) / "t.pkl"
                    tr.save(path)
                    tr2 = Template_Registration.load(path)
        self.assertEqual(tr2.same_side, tr.same_side)
        self.assertIsInstance(tr2.reg_point, Point_Registration)


if __name__ == "__main__":
    unittest.main()
