"""Tests for POI label-name handling: nested format + migration, accessors, and metadata transfer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from TPTBox.core.poi import POI, calc_poi_average
from TPTBox.core.poi_fun.poi_abstract import normalize_label_name
from TPTBox.core.vert_constants import Location, Vertebra_Instance


def _poi():
    return POI(
        centroids={1: {50: (1.0, 2.0, 3.0)}, 2: {50: (4.0, 5.0, 6.0)}},
        orientation=("R", "A", "S"),
        zoom=(1.0, 1.0, 1.0),
        shape=(10.0, 10.0, 10.0),
        rotation=np.eye(3),
        origin=(0.0, 0.0, 0.0),
        level_one_info=Vertebra_Instance,
        level_two_info=Location,
    )


class TestLabelNameMigration(unittest.TestCase):
    def test_old_flat_to_nested(self):
        self.assertEqual(
            normalize_label_name({"(1, 2)": "C2", "(1, 3)": "C3", "(2, 50)": "Body"}),
            {1: {2: "C2", 3: "C3"}, 2: {50: "Body"}},
        )

    def test_json_string_keys_to_int(self):
        self.assertEqual(normalize_label_name({"1": {"2": "C2", "name": "Spine"}}), {1: {2: "C2", "name": "Spine"}})

    def test_idempotent_and_empty(self):
        nested = {1: {2: "C2", "name": "Spine"}}
        self.assertEqual(normalize_label_name(nested), nested)
        self.assertEqual(normalize_label_name(None), {})
        self.assertEqual(normalize_label_name({}), {})


class TestAccessors(unittest.TestCase):
    def test_set_get_point_and_group_name(self):
        p = _poi()
        p.set_label_name(1, 50, "MyCorpus")
        p.set_level_one_name(1, "CervicalGroup")
        self.assertEqual(p.label_name(1, 50), "MyCorpus")
        self.assertEqual(p.level_one_name(1), "CervicalGroup")
        self.assertEqual(p.info["label_name"], {1: {50: "MyCorpus", "name": "CervicalGroup"}})

    def test_enum_fallback(self):
        p = _poi()
        # Vertebra_Instance saves as name -> group name falls back to the enum name
        self.assertEqual(p.level_one_name(2), Vertebra_Instance._get_name(2))
        # accepts Enum members too
        self.assertEqual(p.level_one_name(Vertebra_Instance.C2), Vertebra_Instance._get_name(2))


class TestRoundTrip(unittest.TestCase):
    def test_save_new_format_load_back(self):
        p = _poi()
        p.set_label_name(1, 50, "MyCorpus")
        p.set_level_one_name(1, "CervicalGroup")
        fp = os.path.join(tempfile.mkdtemp(), "sub-x_ctd.json")
        p.save(fp, verbose=False)
        on_disk = json.loads(Path(fp).read_text())[0]["label_name"]
        self.assertEqual(on_disk, {"1": {"50": "MyCorpus", "name": "CervicalGroup"}})  # nested, JSON string keys
        q = POI.load(fp)
        self.assertEqual(q.info["label_name"], {1: {50: "MyCorpus", "name": "CervicalGroup"}})
        self.assertEqual(q.label_name(1, 50), "MyCorpus")

    def test_load_old_flat_format_migrates(self):
        p = _poi()
        fp = os.path.join(tempfile.mkdtemp(), "sub-x_ctd.json")
        p.save(fp, verbose=False)
        raw = json.loads(Path(fp).read_text())
        raw[0]["label_name"] = {"(1, 50)": "OldName"}  # legacy flat format on disk
        Path(fp).write_text(json.dumps(raw))
        q = POI.load(fp)
        self.assertEqual(q.info["label_name"], {1: {50: "OldName"}})


class TestMetadataTransfer(unittest.TestCase):
    def _check(self, q):
        self.assertIs(q.level_one_info, Vertebra_Instance)
        self.assertIs(q.level_two_info, Location)
        self.assertIn("label_name", q.info)

    def test_metadata_survives_ops(self):
        p = _poi()
        p.set_label_name(1, 50, "MyCorpus")
        for q in (
            p.copy(),
            p.reorient(("P", "I", "R")),
            p.rescale((2.0, 2.0, 2.0), verbose=False),
            p.extract_subregion(Location.Vertebra_Corpus),
            p.extract_region(1),
            p.map_labels(label_map_region={1: 10}),
            calc_poi_average([p, p.copy()]),
        ):
            self._check(q)


class TestNameIndexing(unittest.TestCase):
    def test_index_by_level_names(self):
        p = _poi()
        self.assertEqual(p[1, "Vertebra_Corpus"], (1.0, 2.0, 3.0))  # [idx, "level2name"]
        self.assertEqual(p["C1", "Vertebra_Corpus"], (1.0, 2.0, 3.0))  # ["level1name", "level2name"]
        self.assertEqual(p["C2", "Vertebra_Corpus"], (4.0, 5.0, 6.0))
        self.assertIn(("C2", "Vertebra_Corpus"), p)
        p["C1", "Vertebra_Disc"] = (7.0, 8.0, 9.0)
        self.assertEqual(p[1, 100], (7.0, 8.0, 9.0))  # Vertebra_Disc == 100

    def test_enum_with_level_set_no_warning(self):
        import warnings

        p = _poi()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(p[Vertebra_Instance.C1, Location.Vertebra_Corpus], (1.0, 2.0, 3.0))

    def test_enum_without_level_warns(self):
        import warnings

        from TPTBox.core.vert_constants import Any

        q = POI(
            centroids={1: {50: (1.0, 2.0, 3.0)}},
            orientation=("R", "A", "S"),
            zoom=(1.0, 1.0, 1.0),
            shape=(10.0, 10.0, 10.0),
        )
        self.assertIs(q.level_one_info, Any)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = q[Vertebra_Instance.C1, Location.Vertebra_Corpus]
        self.assertTrue(any("not set" in str(x.message) for x in w))


if __name__ == "__main__":
    unittest.main()
