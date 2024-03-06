# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import unittest  # noqa: E402

from TPTBox.tests.test_utils import get_test_ct, get_test_mri  # noqa: E402


class Test_testsamples(unittest.TestCase):
    def test_load_ct(self):
        ct_nii, subreg_nii, vert_nii, label = get_test_ct()
        self.assertTrue(ct_nii.assert_affine(other=subreg_nii, raise_error=False))
        self.assertTrue(ct_nii.assert_affine(other=vert_nii, raise_error=False))

        l3 = vert_nii.extract_label(label)
        l3_subreg = subreg_nii.apply_mask(l3, inplace=False)
        self.assertEqual(l3.volumes()[1], sum(l3_subreg.volumes(include_zero=False).values()))

    def test_load_mri(self):
        mri_nii, subreg_nii, vert_nii, label = get_test_mri()
        self.assertTrue(mri_nii.assert_affine(other=subreg_nii, raise_error=False))
        self.assertTrue(mri_nii.assert_affine(other=vert_nii, raise_error=False))

        l3 = vert_nii.extract_label(label)
        l3_subreg = subreg_nii.apply_mask(l3, inplace=False)
        self.assertEqual(l3.volumes()[1], sum(l3_subreg.volumes(include_zero=False).values()))
