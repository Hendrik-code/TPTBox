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

skip = False
try:
    import pyvista
    from TPTBox.snapshot3D import snapshot_subreg_3d, cut_single_vertebra
except ImportError as e:
    skip = True


def load_bids(in_path: Path | str):
    return bids.BIDS_Global_info(
        datasets=[in_path], parents=["rawdata", "derivatives", "cutout"], additional_key=["snp", "ovl"], verbose=False
    )


folder_fx = Path("D:/Repositories/Work/transitional_vertebra/data/data_tv/fxclass")
folder_vers = Path("D:/Repositories/Work/transitional_vertebra/data/Verse20_original/01_training")


class Test_snapshot_utils(unittest.TestCase):
    @unittest.skipIf(not Path(folder_fx).exists() or skip, "Required files do not exit")
    def test_make_bbox_cutout(self):
        bids_ds = load_bids(folder_fx)
        count_samples = 0
        count_done = 0
        for name, sample in bids_ds.enumerate_subjects(sort=True):
            count_samples += 1
            if isinstance(sample, bids.BIDS_Family):
                families = [sample]
            else:
                raise NotImplementedError()
                families = bids_utils.filter_ct_2_family(name, sample, False)
            for family in families:
                vert_ref = family.get(["vert", "msk_vert"])
                subreg_ref = family.get(["subreg", "msk_subreg"])
                assert vert_ref is not None and subreg_ref is not None
                vert_arr = vert_ref[0].open_nii().get_array()
                subreg_arr = subreg_ref[0].open_nii().get_array()

                # actual test bbox code
                for test_cutsize in [(128, 128, 80), (50, 50, 50), (75, 20, 18), (9, 13, 17)]:
                    vert_mask_arr_cut, subreg_mask_arr_cut, mask_subreg_combi_cut = cut_single_vertebra.make_bounding_box_cutout(
                        vert_arr, subreg_arr, cutout_size=test_cutsize
                    )
                    self.assertEqual(vert_mask_arr_cut.shape, test_cutsize)
                    self.assertEqual(subreg_mask_arr_cut.shape, test_cutsize)
                    self.assertEqual(mask_subreg_combi_cut.shape, test_cutsize)
                    self.assertEqual(np.count_nonzero(mask_subreg_combi_cut), np.count_nonzero(subreg_mask_arr_cut))

                # test make cutout_npz
                mask_subreg_combi_cut, out_path = snapshot_subreg_3d.make_3d_bbox_cutout_npz(
                    vert_arr, subreg_arr, bids_file=vert_ref[0], vert_idx=19, verbose=True
                )
                self.assertEqual(mask_subreg_combi_cut.shape, (128, 128, 80))
                self.assertEqual(
                    str(out_path).replace("////", "/").replace("//", "/"),
                    "D:/Repositories/Work/transitional_vertebra/data/data_tv/fxclass/derivatives/ctfu00006/ses-20120920/vert_cutout_npz/sub-ctfu00006_ses-20120920_seg-vert_vertebra-T12_msk.npz",
                )
                # print(out_path)
                count_done += 1
                break
            if count_done > 0:
                break
        print("test_bbox", count_samples, count_done)


class Test_snapshot3d(unittest.TestCase):
    @unittest.skipIf(not Path(folder_vers).exists() or skip, "Required files do not exit")
    def test_verse20(self):
        # this test runs over verse20 original and should not process anything because of missing files
        bids_ds = load_bids("D:/Repositories/Work/transitional_vertebra/data/Verse20_original/01_training")
        count_samples, count_done = make_subregion_and_count(bids_ds, count_done_break=10)
        self.assertEqual(count_done, 0)
        self.assertNotEqual(count_samples, count_done)

    @unittest.skipIf(not Path(folder_fx).exists() or skip, "Required files do not exit")
    def test_fxclass(self):
        bids_ds = load_bids("D:/Repositories/Work/transitional_vertebra/data/data_tv/fxclass")
        count_samples, count_done = make_subregion_and_count(bids_ds, count_done_break=10)
        self.assertTrue(count_samples == count_done)
        self.assertTrue(count_done <= 10)


def make_subregion_and_count(bids_ds, count_done_break: int):
    count_samples = 0
    count_done = 0
    for name, sample in bids_ds.enumerate_subjects(sort=True):
        count_samples += 1
        if isinstance(sample, bids.BIDS_Family):
            families = [sample]
        else:
            families = bids_utils.filter_ct_2_family(name, sample, False)
        if families is None:
            continue
        for family in families:
            family_key_len = family.get_key_len()

            vert_ref = family.get(["vert", "msk_vert"])
            subreg_ref = family.get(["subreg", "msk_subreg"])
            if vert_ref is None or subreg_ref is None:
                # print(f"Did not find ct, vert_msk, and/or subreg_msk, in sample {name}, {family_key_len}")
                continue
            # check centroid jsons
            ctd_vert = family.get(["ctd_vert", "ctd_subreg"], None)
            if ctd_vert is None:
                # print(f"skipped due to missing json centroid, only got {family_key_len}")
                continue
            multiples = family.get_files_with_multiples()
            # if len(multiples) > 1:  # or len(family["subreg"]) > 1:  # or len(family["ctd_vert"]) > 1:
            #    print(family_key_len, multiples)

            snapshot_subreg_3d.make_subregion_3d_and_snapshot(
                vert_ref[0],
                sub_ref=subreg_ref[0],
                vert_idx_list=[18, 19, 28, 20, 21],
                verbose=False,
                save_combined_model=True,
                save_individual_snapshots=True,
            )
            count_done += 1
            if count_done > count_done_break:
                break
    return count_samples, count_done


if __name__ == "__main__":
    unittest.main()
