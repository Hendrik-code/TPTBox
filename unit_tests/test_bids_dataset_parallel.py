# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

if not os.path.isdir("test"):  # noqa: PTH112
    sys.path.append("..")
file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import random  # noqa: E402
import unittest.mock  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402

from TPTBox import BIDS_Global_info  # noqa: E402
from TPTBox.tests.test_utils import get_tests_dir, repeats  # noqa: E402


class Test_bids_global_info(unittest.TestCase):
    def dataset_test_successfull_load(self, tests_path: Path, parent: str):
        bids_ds = BIDS_Global_info(datasets=[tests_path], parents=[parent])

        expected_keys_mri = ["msk_seg-subreg_label-6", "msk_seg-vert_label-6", "T2w_label-6"]
        expected_keys_ct = ["ct_label-22", "msk_seg-vert_label-22", "msk_seg-subreg_label-22"]
        expected_keys = expected_keys_mri if parent == "sample_mri" else expected_keys_ct

        expected_subject_name = "mri" if parent == "sample_mri" else "ct"

        for name, subject in bids_ds.enumerate_subjects(sort=True):
            self.assertEqual(name, expected_subject_name)
            q = subject.new_query()
            families = q.loop_dict(sort=True)
            for f in families:
                print(f.family_id, f.get_key_len())
                key_len_dict = f.get_key_len()
                for k in expected_keys:
                    self.assertTrue(k in key_len_dict)
                    self.assertTrue(key_len_dict[k] == 1)

    def test_load_same_dataset_multiple_times(self):
        tests_path = get_tests_dir()
        parent = "sample_ct"
        for _i in range(repeats):
            self.dataset_test_successfull_load(tests_path=tests_path, parent=parent)

    def test_load_different_datasets_multiple_times(self):
        tests_path = get_tests_dir()
        for _i in range(repeats):
            parent = "sample_ct" if random.random() < 0.5 else "sample_mri"
            self.dataset_test_successfull_load(tests_path=tests_path, parent=parent)

    def test_load_same_dataset_multiple_times_parallel(self):
        tests_path = get_tests_dir()
        parent = "sample_ct"
        Parallel(n_jobs=5, backend="threading")(
            delayed(self.dataset_test_successfull_load)(tests_path=tests_path, parent=parent) for i in range(repeats)
        )

    def test_load_different_datasets_multiple_times_parallel(self):
        tests_path = get_tests_dir()
        Parallel(n_jobs=5, backend="threading")(
            delayed(self.dataset_test_successfull_load)(
                tests_path=tests_path, parent="sample_ct" if random.random() < 0.5 else "sample_mri"
            )
            for i in range(repeats)
        )
