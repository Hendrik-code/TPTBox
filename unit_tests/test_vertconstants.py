from __future__ import annotations

import unittest

import numpy as np

from TPTBox import Vertebra_Instance
from TPTBox.core.vert_constants import Location, vert_subreg_labels


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


class Test_Vertebra_Regions(unittest.TestCase):
    """Tests for Vertebra_Instance region accessors and class-level queries."""

    def test_cervical_count(self):
        self.assertEqual(len(Vertebra_Instance.cervical()), 7)

    def test_thoracic_count(self):
        self.assertEqual(len(Vertebra_Instance.thoracic()), 13)

    def test_lumbar_count(self):
        self.assertEqual(len(Vertebra_Instance.lumbar()), 6)

    def test_sacrum_includes_cocc(self):
        self.assertIn(Vertebra_Instance.COCC, Vertebra_Instance.sacrum())

    def test_order_starts_c1_ends_cocc(self):
        order = Vertebra_Instance.order()
        self.assertEqual(order[0], Vertebra_Instance.C1)
        self.assertEqual(order[-1], Vertebra_Instance.COCC)

    def test_order_dict_length(self):
        od = Vertebra_Instance.order_dict()
        self.assertEqual(len(od), len(Vertebra_Instance.order()))

    def test_is_sacrum_true(self):
        self.assertTrue(Vertebra_Instance.is_sacrum(Vertebra_Instance.S1.value))
        self.assertTrue(Vertebra_Instance.is_sacrum(Vertebra_Instance.COCC.value))

    def test_is_sacrum_false(self):
        self.assertFalse(Vertebra_Instance.is_sacrum(Vertebra_Instance.L5.value))
        self.assertFalse(Vertebra_Instance.is_sacrum(Vertebra_Instance.T1.value))

    def test_rib_label_nonempty(self):
        self.assertGreater(len(Vertebra_Instance.rib_label()), 0)

    def test_endplate_label_nonempty(self):
        self.assertGreater(len(Vertebra_Instance.endplate_label()), 0)

    def test_ivd_and_endplate_offsets(self):
        v = Vertebra_Instance.L3  # no rib, has IVD
        self.assertEqual(v.IVD, v.value + 100)
        self.assertEqual(v.ENDPLATE, v.value + 200)

    def test_vertebra_label_without_sacrum(self):
        labels = Vertebra_Instance.vertebra_label_without_sacrum()
        self.assertNotIn(Vertebra_Instance.S1.value, labels)
        self.assertNotIn(Vertebra_Instance.COCC.value, labels)


class Test_Vertebra_Navigation(unittest.TestCase):
    """Tests for get_next_poi and get_previous_poi navigation helpers."""

    def test_get_next_poi_finds_next(self):
        labels = [Vertebra_Instance.L3.value, Vertebra_Instance.L5.value]
        nxt = Vertebra_Instance.L3.get_next_poi(labels)
        self.assertEqual(nxt, Vertebra_Instance.L5)

    def test_get_next_poi_returns_none_at_end(self):
        labels = [Vertebra_Instance.COCC.value]
        self.assertIsNone(Vertebra_Instance.COCC.get_next_poi(labels))

    def test_get_previous_poi_finds_prev(self):
        labels = [Vertebra_Instance.T10.value, Vertebra_Instance.T12.value]
        prev = Vertebra_Instance.T12.get_previous_poi(labels)
        self.assertEqual(prev, Vertebra_Instance.T10)

    def test_get_previous_poi_returns_none_at_start(self):
        labels = [Vertebra_Instance.C1.value]
        self.assertIsNone(Vertebra_Instance.C1.get_previous_poi(labels))

    def test_get_next_skips_absent(self):
        # T1 and T3 present, T2 missing — next from T1 should be T3
        labels = [Vertebra_Instance.T1.value, Vertebra_Instance.T3.value]
        nxt = Vertebra_Instance.T1.get_next_poi(labels)
        self.assertEqual(nxt, Vertebra_Instance.T3)


class Test_Location_Enum(unittest.TestCase):
    """Tests for the Location enum and vert_subreg_labels utility."""

    def test_vert_subreg_labels_nonempty(self):
        self.assertGreater(len(vert_subreg_labels()), 0)

    def test_vert_subreg_labels_contains_corpus(self):
        self.assertIn(Location.Vertebra_Corpus, vert_subreg_labels())

    def test_vert_subreg_labels_with_border_not_smaller(self):
        with_border = vert_subreg_labels(with_border=True)
        without_border = vert_subreg_labels(with_border=False)
        self.assertGreaterEqual(len(with_border), len(without_border))

    def test_location_save_as_name_false(self):
        self.assertFalse(Location.save_as_name())

    def test_location_get_id_from_value(self):
        val = Location.Vertebra_Corpus.value
        self.assertEqual(Location._get_id(val), val)

    def test_location_unique_values(self):
        values = [loc.value for loc in Location]
        self.assertEqual(len(values), len(set(values)), "Location enum has duplicate values")
