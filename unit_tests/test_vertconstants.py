# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np

from TPTBox import Vertebra_Instance

structures = ["rib", "ivd", "endplate"]


class Test_Locations(unittest.TestCase):
    def test_vertebra_instance_uniqueness(self):
        name2idx = Vertebra_Instance.name2idx()

        print(name2idx)
        label_values = list(name2idx.values())
        label_valuesset, counts = np.unique(label_values, return_counts=True)

        for idx, c in enumerate(counts):
            if c > 1:
                l = label_valuesset[idx]

                structures_overlap = {i: g for i, g in name2idx.items() if g == l}

                self.assertTrue(False, f"Overlapping labels in {structures_overlap}")

    def test_vertebra_instance_completeness(self):
        name2idx = Vertebra_Instance.name2idx()
        idx2name = Vertebra_Instance.idx2name()

        for start in name2idx:
            i = name2idx[start]
            i2 = idx2name[i]
            self.assertTrue(i2 == start)

    def test_vertebra_instance_correctness(self):
        for i in [
            Vertebra_Instance.C7,
            Vertebra_Instance.T1,
            Vertebra_Instance.T2,
            Vertebra_Instance.T3,
            Vertebra_Instance.T4,
            Vertebra_Instance.T5,
            Vertebra_Instance.T12,
            Vertebra_Instance.T13,
        ]:
            self.assertIsNotNone(i.RIB)

        for i in [
            Vertebra_Instance.C6,
            Vertebra_Instance.C1,
            Vertebra_Instance.C3,
            Vertebra_Instance.L3,
            Vertebra_Instance.L6,
            Vertebra_Instance.COCC,
            Vertebra_Instance.S1,
            Vertebra_Instance.S3,
            Vertebra_Instance.S6,
        ]:
            with self.assertRaises(AssertionError):
                self.assertIsNone(i.RIB)
