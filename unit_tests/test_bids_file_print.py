# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from pathlib import Path
import unittest
import sys
import os

if not os.path.isdir("test"):
    sys.path.append("..")
file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import TPTBox.core.bids_files as bids
from TPTBox import BIDS_FILE
import io
import unittest
import unittest.mock
from TPTBox.tests.test_utils import get_BIDS_test, a


class Test_bids_file(unittest.TestCase):
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_non_verbose(self, mock_stdout):
        bids.validate_entities("xxx", "yyy", "zzz", False)
        self.assertEqual(mock_stdout.getvalue(), "")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_set(self, mock_stdout):
        f = BIDS_FILE("sub-spinegan0026_ses-20210109_sequ-203_seg-subreg_ctd.json", "/media/robert/Expansion/dataset-Testset", verbose=True)
        f.set("seg", "mimi")
        self.assertEqual(mock_stdout.getvalue(), "")
        f.set("TUT", "mimi")
        self.assertNotEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        self.assertEqual(f.get("seg"), "mimi")
        self.assertEqual(mock_stdout.getvalue(), "")
        self.assertEqual(f.get("TUT"), "mimi")
        self.assertEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        self.assertEqual(f.get("sequ"), "203")
        self.assertEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        self.assertEqual(f.get("b", default="999"), "999")
        self.assertEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        try:
            f.get("x")
            self.assertFalse(True)
        except Exception:
            pass
        f.set("task", "LR")
        # sys.__stdout__.write(mock_stdout.getvalue())
        self.assertEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        f.set("dir", "LR123")
        self.assertEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        f.set("task", "LR-123")
        self.assertNotEqual(mock_stdout.getvalue(), "")
        mock_stdout.truncate(0)
        # for key in ["run", "mod", "echo", "flip", "inv"]:
        #    f.set(key, str(random.randint(0, 100000)))
        #    self.assertEqual(mock_stdout.getvalue(), "", msg=f"{key},{f.get(key)}")
        #    mock_stdout.truncate(0)
        #    letters = string.ascii_lowercase
        #    f.set(key, "".join(random.choice(letters) for i in range(10)))
        #    self.assertNotEqual(mock_stdout.getvalue(), "")
        #    mock_stdout.truncate(0)


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
