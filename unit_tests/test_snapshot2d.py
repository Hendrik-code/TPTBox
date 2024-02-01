# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from pathlib import Path
import unittest
import sys
import os
import random

import numpy as np

if not os.path.isdir("test"):
    sys.path.append("..")
file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

import TPTBox.core.bids_files as bids
import TPTBox
from TPTBox.snapshot2D import snapshot_modular, snapshot_templates, Snapshot_Frame, create_snapshot


def load_bids(in_path: Path):
    return bids.BIDS_Global_info(
        datasets=[in_path], parents=["rawdata", "derivatives", "cutout"], additional_key=["snp", "ovl"], verbose=False
    )


folder_fx = Path("D:/Repositories/Work/transitional_vertebra/data/data_tv/fxclass")
folder_vers = Path("D:/Repositories/Work/transitional_vertebra/data/Verse20_original/01_training")
import nibabel as nib


class Test_snapshot2d(unittest.TestCase):
    # def test_artificial(self):
    #    # make snapshot of artifical 3d numpy array
    #    test_array = np.zeros((150, 150, 300))  # ASL (F, U, L)
    #    seg_array = np.zeros((150, 150, 300))  # ASL (F, U, L)
    #    func = lambda x, y, z: (np.sqrt(x / 85) + ((z - 150) / 150) + (np.cos(y) / 2))
    #    # test_array = np.fromfunction(func, test_array)
    #    for x in range(test_array.shape[0]):
    #        for y in range(test_array.shape[1]):
    #            for z in range(test_array.shape[2]):
    #                if x == y or x == z or y == z:
    #                    test_array[x, y, z] = 1.0
    #                else:
    #                    test_array[x, y, z] = func(x, y, z)
    #    # test_array = np.vectorize(func)(test_array)
    #    centroids = {5: (60, 75, 60), 6: (75, 65, 140), 7: (78, 90, 240)}
    #    ctd = BIDS.Centroids(orientation=("L", "A", "S"), centroids=centroids)
    #    cube_size = 20
    #    for label, c in centroids.items():
    #        test_array[c[0] - 20 : c[0] + 20, c[1] - 20 : c[1] + 20, c[2] - 20 : c[2] + 20] = 1 - (0.1 * label - 5)
    #        seg_array[c[0] - 20 : c[0] + 20, c[1] - 20 : c[1] + 20, c[2] - 20 : c[2] + 20] = label
    #    affine = np.array([[-2.0, 0.0, 0.0, 117.86], [-0.0, 1.97, -0.36, -35.72], [0.0, 0.32, 2.17, -7.25], [0.0, 0.0, 0.0, 1.0]])
    #    test_nii = nib.Nifti1Image(dataobj=test_array, affine=affine)
    #    seg_nii = nib.Nifti1Image(dataobj=seg_array, affine=affine)
    #    ct_snap_seg = [
    #        Snapshot_Frame(
    #            image=test_nii,
    #            segmentation=seg_nii,
    #            centroids=ctd,
    #            coronal=True,
    #        ),
    #        Snapshot_Frame(
    #            image=test_nii,
    #            segmentation=seg_nii,
    #            centroids=ctd,
    #            sagittal=False,
    #            coronal=True,
    #            hide_segmentation=True,
    #            crop_img=True,
    #        ),
    #        Snapshot_Frame(
    #            image=test_nii,
    #            segmentation=seg_nii,
    #            centroids=ctd,
    #            sagittal=False,
    #            coronal=True,
    #            hide_segmentation=True,
    #            image_threshold=0.4,
    #            denoise_threshold=0.5,
    #        ),
    #    ]
    #    create_snapshot(os.getcwd() + "test_snap.png", ct_snap_seg)

    @unittest.skipIf(not Path(folder_vers).exists(), "Required files do not exit")
    def test_verse20(self):
        # this test runs over verse20 original and should not process anything because of missing files
        bids_ds = load_bids(folder_vers)
        count_samples, count_done = make_mip_snapshot_and_count(bids_ds, count_done_break=5)
        self.assertEqual(count_done, 5)
        self.assertNotEqual(count_samples, count_done)

    @unittest.skipIf(not Path(folder_fx).exists(), "Required files do not exit")
    def test_fxclass(self):
        bids_ds = load_bids(folder_fx)
        count_samples, count_done = make_mip_snapshot_and_count(bids_ds, count_done_break=5)
        self.assertTrue(count_samples == count_done)
        self.assertTrue(count_done <= 5)


def make_mip_snapshot_and_count(bids_ds, count_done_break: int):
    count_samples = 0
    count_done = 0
    for name, sample in bids_ds.enumerate_subjects(sort=True):
        count_samples += 1
        if isinstance(sample, bids.BIDS_Family):
            families = [sample]
        else:
            raise NotImplementedError()
            families = bids_utils.filter_ct_2_family(name, sample, False)
        if families is None:
            continue
        for family in families:
            ct_ref = family.get("ct")
            vert_ref = family.get(["vert", "msk_vert"])
            if ct_ref is None or vert_ref is None:
                # print(f"Did not find ct, vert_msk, and/or subreg_msk, in sample {name}, {family_key_len}")
                continue
            # check centroid jsons
            ctd_vert = family.get(["ctd_vert", "ctd_subreg"], None)
            if ctd_vert is None:
                # print(f"skipped due to missing json centroid, only got {family_key_len}")
                continue
            snapshot_templates.mip_shot(ct_ref[0], vert_msk=vert_ref[0], subreg_ctd=ctd_vert[0])
            count_done += 1
            if count_done > count_done_break:
                break
    return count_samples, count_done


if __name__ == "__main__":
    unittest.main()
