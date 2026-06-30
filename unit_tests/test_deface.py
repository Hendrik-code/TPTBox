from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pytest

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

from TPTBox import NII  # noqa: E402

try:
    import torch  # noqa: F401

    has_torch = True
except ModuleNotFoundError:
    has_torch = False


def _nii(arr: np.ndarray, seg: bool, affine=None) -> NII:
    if affine is None:
        affine = np.eye(4)
    return NII(nib.Nifti1Image(arr, affine), seg=seg)


def _ct_air(shape=(40, 40, 40), bone_block=True) -> NII:
    """Synthetic CT: all air (-1000) with an optional high-intensity bone block."""
    arr = np.full(shape, -1000, dtype=np.int16)
    if bone_block:
        # bone-valued block (max >= 128) so set_dtype('smallest_int') chooses int16
        arr[2:6, 2:6, 2:6] = 1000
    return _nii(arr, seg=False)


def _face_block(ref: NII, lo=10, hi=26) -> NII:
    arr = np.zeros(ref.shape, dtype=np.uint8)
    arr[lo:hi, lo:hi, lo:hi] = 1
    return _nii(arr, seg=True, affine=ref.affine.copy())


@unittest.skipIf(not has_torch, "requires torch to import the segmentation module")
class Test_extend_mask(unittest.TestCase):
    def test_extends_anterior(self):
        from TPTBox.segmentation._deface import _extend_mask

        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[8:12, 8:11, 8:12] = 1
        m = _nii(arr, seg=True)
        self.assertEqual(m.orientation[m.get_axis("A")], "A")
        before = int(m.get_array().sum())
        out = _extend_mask(m.copy(), 4, "A")
        self.assertIsInstance(out, NII)
        after = int(out.get_array().sum())
        # mask was dragged anteriorly -> strictly more voxels
        self.assertGreater(after, before)
        # voxels beyond original anterior extent (A axis index 10) now set in the block column
        col = out.get_array()[9, :, 9]
        self.assertEqual(col[11], 1)
        self.assertEqual(col[13], 1)

    def test_empty_mask_unchanged(self):
        from TPTBox.segmentation._deface import _extend_mask

        m = _nii(np.zeros((10, 10, 10), dtype=np.uint8), seg=True)
        out = _extend_mask(m.copy(), 3, "A")
        self.assertEqual(int(out.get_array().sum()), 0)

    def test_opposite_direction_branch(self):
        # direction="P" on an RAS mask hits the else-branch; n>=min coord keeps the
        # (buggy) inner loop empty so it is a safe no-op.
        from TPTBox.segmentation._deface import _extend_mask

        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[8:12, 8:11, 8:12] = 1
        m = _nii(arr, seg=True)
        before = int(m.get_array().sum())
        out = _extend_mask(m.copy(), 20, "P")
        self.assertIsInstance(out, NII)
        self.assertEqual(int(out.get_array().sum()), before)


@unittest.skipIf(not has_torch, "requires torch to import the segmentation module")
class Test_deface_img(unittest.TestCase):
    def test_masked_region_set_to_min(self):
        from TPTBox.segmentation._deface import deface_img

        ct = _ct_air(shape=(16, 16, 16), bone_block=False)
        ct_arr = ct.get_array()
        ct_arr[:] = 1000  # high max -> smallest_int picks int16, -1024 survives
        ct = ct.set_array(ct_arr)
        fm_arr = np.zeros((16, 16, 16), dtype=np.uint8)
        fm_arr[4:8, 4:8, 4:8] = 1
        fm = _nii(fm_arr, seg=True)
        out = deface_img(ct, fm, min_value=-1024, to_int=True)
        oarr = out.get_array()
        self.assertTrue((oarr[4:8, 4:8, 4:8] == -1024).all())
        self.assertTrue((oarr[0:4, 0:4, 0:4] == 1000).all())

    def test_to_int_false_exact_value(self):
        from TPTBox.segmentation._deface import deface_img

        ct = _nii(np.full((12, 12, 12), 50, dtype=np.int16), seg=False)
        fm_arr = np.zeros((12, 12, 12), dtype=np.uint8)
        fm_arr[3:6, 3:6, 3:6] = 1
        fm = _nii(fm_arr, seg=True)
        out = deface_img(ct, fm, min_value=-777, to_int=False)
        self.assertTrue((out.get_array()[3:6, 3:6, 3:6] == -777).all())

    def test_save_roundtrip(self):
        from TPTBox.segmentation._deface import deface_img

        ct = _nii(np.full((10, 10, 10), 1000, dtype=np.int16), seg=False)
        fm_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        fm_arr[2:5, 2:5, 2:5] = 1
        fm = _nii(fm_arr, seg=True)
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "defaced.nii.gz"
            out = deface_img(ct, fm, min_value=-1024, ct_out=out_path)
            self.assertTrue(out_path.exists())
            reloaded = NII.load(out_path, seg=False)
            self.assertEqual(reloaded.shape, ct.shape)
            self.assertTrue((reloaded.get_array()[2:5, 2:5, 2:5] == -1024).all())
            self.assertIsInstance(out, NII)

    def test_shape_mismatch_raises(self):
        from TPTBox.segmentation._deface import deface_img

        ct = _nii(np.zeros((10, 10, 10), dtype=np.int16), seg=False)
        fm = _nii(np.zeros((8, 8, 8), dtype=np.uint8), seg=True)
        with pytest.raises(AssertionError):
            deface_img(ct, fm)


@unittest.skipIf(not has_torch, "requires torch to import the segmentation module")
class Test_compute_deface_mask_cta(unittest.TestCase):
    def test_internal_passthrough(self):
        import TPTBox.segmentation._deface as df

        ct = _ct_air()
        face = _face_block(ct)
        with mock.patch.object(df, "run_VibeSeg", return_value=face) as m:
            out = df._compute_deface_mask_cta(ct, outpath=None, override=True, gpu=3)
        self.assertIs(out, face)
        m.assert_called_once()
        # dataset_id=1 / keep_size=False are hard-wired for the defacing model
        self.assertEqual(m.call_args.kwargs["dataset_id"], 1)
        self.assertEqual(m.call_args.kwargs["keep_size"], False)
        self.assertEqual(m.call_args.kwargs["gpu"], 3)

    def test_internal_early_return_when_exists(self):
        import TPTBox.segmentation._deface as df

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "exists.nii.gz"
            _ct_air().set_dtype("smallest_uint").save(out_path)
            with mock.patch.object(df, "run_VibeSeg") as m:
                # pass a str outpath -> exercises the str->Path coercion branch
                out = df._compute_deface_mask_cta(_ct_air(), outpath=str(out_path), override=False)
            self.assertEqual(Path(out), out_path)
            m.assert_not_called()

    def test_full_pipeline_returns_binary_mask(self):
        import TPTBox.segmentation._deface as df

        ct = _ct_air()
        face = _face_block(ct)
        with mock.patch.object(df, "run_VibeSeg", return_value=face) as m:
            mask = df.compute_deface_mask_cta(ct, outpath=None, override=True)
        m.assert_called_once()
        self.assertIsInstance(mask, NII)
        self.assertEqual(mask.shape, ct.shape)
        self.assertTrue(set(mask.unique()).issubset({0, 1}))
        # the morphology pipeline must leave a non-empty mask
        self.assertGreater(int(mask.get_array().sum()), 0)

    def test_full_pipeline_partially_defaced_and_save(self):
        import TPTBox.segmentation._deface as df

        ct = _ct_air()
        face = _face_block(ct)
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "mask.nii.gz"
            with mock.patch.object(df, "run_VibeSeg", return_value=face):
                mask = df.compute_deface_mask_cta(ct, outpath=out_path, override=True, partially_defaced=True)
            self.assertTrue(out_path.exists())
            self.assertTrue(set(mask.unique()).issubset({0, 1}))

    def test_full_pipeline_early_return_when_exists(self):
        import TPTBox.segmentation._deface as df

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "exists.nii.gz"
            _ct_air().set_dtype("smallest_uint").save(out_path)
            with mock.patch.object(df, "run_VibeSeg") as m:
                # pass a str outpath -> exercises the str->Path coercion branch
                out = df.compute_deface_mask_cta(_ct_air(), outpath=str(out_path), override=False)
            self.assertEqual(Path(out), out_path)
            m.assert_not_called()


if __name__ == "__main__":
    unittest.main()
