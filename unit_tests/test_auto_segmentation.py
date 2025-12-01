# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from TPTBox.core.nii_wrapper import to_nii

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

import unittest  # noqa: E402

from TPTBox import NII, Location, Print_Logger, calc_poi_from_subreg_vert  # noqa: E402
from TPTBox.tests.test_utils import get_test_ct, get_test_mri, get_tests_dir  # noqa: E402

has_spineps = False
try:
    import spineps

    has_spineps = True
except ModuleNotFoundError:
    has_spineps = False


class Test_test_samples(unittest.TestCase):
    # def test_load_ct(self):
    #    ct_nii, subreg_nii, vert_nii, label = get_test_ct()
    #    self.assertTrue(ct_nii.assert_affine(other=subreg_nii, raise_error=False))
    #    self.assertTrue(ct_nii.assert_affine(other=vert_nii, raise_error=False))

    #    l3 = vert_nii.extract_label(label)
    #    l3_subreg = subreg_nii.apply_mask(l3, inplace=False)
    #    self.assertEqual(l3.volumes()[1], sum(l3_subreg.volumes(include_zero=False).values()))
    @unittest.skipIf(not has_spineps, "requires spineps to be installed")
    def test_get_outpaths_spineps(self):
        tests_path = get_tests_dir()

        from TPTBox.segmentation.spineps import get_outpaths_spineps

        mri_path = tests_path.joinpath("sample_mri")
        mri_path = mri_path.joinpath("sub-mri_label-6_T2w.nii.gz")
        out = get_outpaths_spineps(mri_path, tests_path)
        assert "out_spine" in out
        assert "out_vert" in out

    @unittest.skipIf(not has_spineps, "requires spineps to be installed")
    def test_spineps(self):
        tests_path = get_tests_dir()
        if (tests_path / "derivative").exists():
            shutil.rmtree(tests_path / "derivative")

        mri_nii, subreg_nii, vert_nii, label = get_test_mri()
        from TPTBox.segmentation.spineps import run_spineps

        mri_path = tests_path.joinpath("sample_mri")
        mri_path = mri_path.joinpath("sub-mri_label-6_T2w.nii.gz")
        out = run_spineps(mri_path, tests_path, ignore_compatibility_issues=True)
        assert "out_spine" in out
        assert "out_vert" in out
        assert out["out_spine"].exists()
        assert out["out_vert"].exists()
        assert out["out_snap"].exists()
        assert out["out_ctd"].exists()

        vert_nii = to_nii(out["out_vert"], True)
        assert label in vert_nii.unique(), (label, vert_nii.unique())
        shutil.rmtree(tests_path / "derivative")

    @unittest.skipIf(not has_spineps, "requires spineps to be installed")
    def test_VIBESeg(self):
        tests_path = get_tests_dir()
        from TPTBox.segmentation import run_vibeseg

        for i in [100, 11, 278]:
            mri_path = tests_path.joinpath("sample_mri")
            mri_path = mri_path.joinpath("sub-mri_label-6_T2w.nii.gz")
            seg_out_path = tests_path / f"{i}_test_VIBESeg.nii.gz"
            out = run_vibeseg(mri_path, seg_out_path, dataset_id=i)
            assert isinstance(out, (NII, Path))
            assert seg_out_path.exists()
            seg_out_path.unlink(missing_ok=True)

    @unittest.skipIf(not has_spineps, "requires spineps to be installed")
    def test_VIBESeg_ct(self):
        tests_path = get_tests_dir()
        from TPTBox.segmentation import run_vibeseg

        for i in [100, 11, 520]:
            tests_path = get_tests_dir()
            ct_path = tests_path.joinpath("sample_ct")
            ct_path = ct_path.joinpath("sub-ct_label-22_ct.nii.gz")
            seg_out_path = tests_path / f"{i}_test_ct_VIBESeg.nii.gz"
            out = run_vibeseg(ct_path, seg_out_path, dataset_id=i)
            assert isinstance(out, (NII, Path))
            assert seg_out_path.exists()
            seg_out_path.unlink(missing_ok=True)
