# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import TPTBox
import TPTBox.core.bids_files as bids
from TPTBox.tests.test_utils import a, get_BIDS_test


class Test_bids_file(unittest.TestCase):
    @unittest.skipIf(not Path("/media/robert/Expansion/dataset-Testset").exists(), "requires real data to be opened")
    def test_non_flatten_dixon(self):
        global_info = get_BIDS_test()
        for subj_name, subject in global_info.enumerate_subjects():
            with self.subTest(dataset=f"{subj_name} filter"):
                query = subject.new_query()
                # A nii.gz must exist
                query.filter("Filetype", "nii.gz")
                # It must exist a dixon and a msk
                query.filter("format", "dixon")
                query.filter("format", "msk")
                # Example of lamda function filtering
                query.filter("sequ", lambda x: int(x) == 303, required=True)  # type: ignore
            with self.subTest(dataset=f"{subj_name} loop"):
                for sequences in query.loop_dict():
                    self.assertIsInstance(sequences, dict)
                    self.assertTrue("dixon" in sequences)
                    self.assertEqual(len(sequences["dixon"]), 3)
                    # sequences.

    @unittest.skipIf(not Path("/media/robert/Expansion/dataset-Testset").exists(), "requires real data to be opened")
    def test_non_flatten_ct(self):
        global_info = get_BIDS_test()
        for subj_name, subject in global_info.enumerate_subjects():
            subject: bids.Subject_Container
            with self.subTest(dataset=f"{subj_name} filter"):
                query = subject.new_query()
                # A nii.gz must exist
                query.filter("Filetype", "nii.gz")
                # It must exist a dixon and a msk
                query.filter("format", "ct")
                # Example of lamda function filtering
                query.filter("sequ", lambda x: x != "None" and isinstance(x, str) and int(x) == 203, required=True)
            with self.subTest(dataset=f"{subj_name} loop"):
                for sequences in query.loop_dict():
                    self.assertIsInstance(sequences, dict)
                    self.assertTrue("ct" in sequences)
                    self.assertTrue("vert" in sequences)
                    self.assertTrue("subreg" in sequences)
                    self.assertTrue("snp" in sequences)

    @unittest.skipIf(not Path("/media/robert/Expansion/dataset-Testset").exists(), "requires real data to be opened")
    def test_get_sequence_files(self):
        global_info = get_BIDS_test()
        for _, subject in global_info.enumerate_subjects():
            if "20210111_301" in subject.sequences.keys():
                sequences = subject.get_sequence_files("20210111_301")
                self.assertIsInstance(sequences, TPTBox.BIDS_Family)
                self.assertTrue("dixon" in sequences)
                self.assertTrue("msk" in sequences)
                self.assertTrue("snp" in sequences)
                self.assertTrue("ctd_subreg" in sequences)
                self.assertIsInstance(sequences["dixon"], list)
                self.assertTrue(len(sequences["dixon"]) == 3)
                self.assertTrue("dixon" in sequences)

            else:

                def mapping(x: bids.BIDS_FILE):
                    if x.format == "ctd" and x.info["seg"] == "subreg":
                        return "other_key_word"
                    return None

                sequences = subject.get_sequence_files("20220517_406", key_transform=mapping)
                self.assertIsInstance(sequences, TPTBox.BIDS_Family)
                self.assertTrue("other_key_word" in sequences)
                self.assertTrue("snp" in sequences)
                self.assertTrue("msk_seg-subreg" in sequences)
                self.assertTrue("msk_vert" in sequences)
                self.assertTrue("ct" in sequences)

    @unittest.skipIf(not Path("/media/robert/Expansion/dataset-Testset").exists(), "requires real data to be opened")
    def test_conditional_action(self):
        global_info = get_BIDS_test()
        for subj_name, subject in global_info.enumerate_subjects():
            with self.subTest(dataset=f"{subj_name} filter"):
                query = subject.new_query(flatten=True)
                # A nii.gz must exist
                query.filter("Filetype", "nii.gz")
                # It must exist a dixon
                query.filter("format", "dixon")
            with self.subTest(dataset=f"{subj_name} action"):
                # Find if the dixon is a inphase image by looking into the json and set "part" to real
                query.action(
                    # Set Part key to real. Will be called if the filter = True
                    action_fun=lambda x: x.set("part", "real"),
                    # x is the json, because of the key="json". We return True if the json confirms that this is a real-part image
                    filter_fun=lambda x: "IP" in x["ImageType"],  # type: ignore
                    key="json",
                    # The json is required
                    required=True,
                )

            with self.subTest(dataset=f"{subj_name} loop"):
                real_count = 0
                dix_count = 0
                for sequences in query.loop_list():
                    self.assertIsInstance(sequences, bids.BIDS_FILE)
                    if "part" in sequences.info:
                        real_count += 1
                    dix_count += 1
                self.assertEqual(dix_count / real_count, 3.0)


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
