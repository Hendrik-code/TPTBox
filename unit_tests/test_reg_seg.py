# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import random
import sys
import unittest
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

# TODO OUTDATED
# from TPTBox.registration.ridged_points.reg_segmentation import ridged_segmentation_from_seg

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
import nibabel as nib

from TPTBox import to_nii

test_data = [
    "BIDS/test/test_data/sub-fxclass0001_seg-subreg_msk.nii.gz",
    "BIDS/test/test_data/sub-fxclass0001_seg-vert_msk.nii.gz",
    "BIDS/test/test_data/sub-fxclass0004_seg-subreg_msk.nii.gz",
    "BIDS/test/test_data/sub-fxclass0004_seg-vert_msk.nii.gz",
]
out_name_sub = "BIDS/test/test_data/sub-fxclass0004_seg-subreg_reg-0001_msk.nii.gz"
out_name_vert = "BIDS/test/test_data/sub-fxclass0004_seg-vert_reg-0001_msk.nii.gz"


class Test_registration(unittest.TestCase):
    @unittest.skipIf(not Path(test_data[0]).exists(), "requires real data test data")
    def test_seg_registration(self):
        pass
        # TODO OUTDATED
        # t = ridged_segmentation_from_seg(*test_data, verbose=True, ids=list(range(40, 50)), exclusion=[19])
        # slice = t.compute_crop(dist=20)
        # nii_out = t.transform_nii(moving_img_nii=(test_data[2], True), slice=slice)
        # nii_out.save(out_name_sub)
        # nii_out = t.transform_nii(moving_img_nii=(test_data[3], True), slice=slice)
        # nii_out.save(out_name_vert)


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
