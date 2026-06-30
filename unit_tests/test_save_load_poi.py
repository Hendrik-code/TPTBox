# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from TPTBox import POI, POI_Global
from TPTBox.core.poi_fun.poi_abstract import POI_Descriptor
from TPTBox.core.poi_fun.save_load import (
    _get_poi_idx_from_text,
    _load_docker_centroids,
    _load_form_POI_spine_r2,
    _load_landmark_txt,
    _parse_coords,
    _parse_header_value,
    load_poi,
    save_poi,
)
from TPTBox.core.vert_constants import LABEL_MAX, conversion_poi2text
from TPTBox.tests.test_utils import get_poi, repeats


def _quiet():
    """Return a fresh context manager that swallows ``stdout`` noise."""
    return redirect_stdout(io.StringIO())


def _gruber_poi(num_vert: int = 6) -> POI:
    """Build a POI whose subregions are all valid Gruber (FORMAT_GRUBER) keys."""
    subs = list(conversion_poi2text)
    cdt = POI_Descriptor()
    for v in range(1, num_vert + 1):
        sub = subs[v % len(subs)]
        cdt[v, sub] = tuple(random.random() * 40 for _ in range(3))
    return POI(cdt, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=(50, 50, 50), origin=(0, 0, 0), rotation=np.eye(3))


class Test_Save_Load_POI(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # round-trips of writable on-disk formats                            #
    # ------------------------------------------------------------------ #
    def test_roundtrip_docker_and_poi(self):
        for save_hint in (0, 2):
            with self.subTest(save_hint=save_hint):
                p = get_poi(num_vert=12, num_subreg=2)
                file = Path(tempfile.gettempdir(), "test_rt_docker_poi.json")
                p.save(file, verbose=False, save_hint=save_hint)
                c = load_poi(file)
                file.unlink(missing_ok=True)
                self.assertEqual(c, p)
                self.assertTrue(np.isclose(np.asarray(c.affine), np.asarray(p.affine), atol=1e-6).all())

    def test_roundtrip_gruber(self):
        for _ in range(repeats):
            p = _gruber_poi(num_vert=6)
            file = Path(tempfile.gettempdir(), "test_rt_gruber.json")
            p.save(file, verbose=False, save_hint=1)
            c = load_poi(file)
            file.unlink(missing_ok=True)
            self.assertEqual(c, p)

    def test_roundtrip_old_poi(self):
        # FORMAT_OLD_POI (10) is lossy: it stores 1mm-iso ("R","P","I") coords
        # without shape/rotation. The loaded POI (with None metadata) must be the
        # *other* argument so the missing fields are skipped during comparison.
        p = get_poi(num_vert=5, num_subreg=2)
        file = Path(tempfile.gettempdir(), "test_rt_old.json")
        p.save(file, verbose=False, save_hint=10)
        c = load_poi(file)
        file.unlink(missing_ok=True)
        expected = p.rescale((1, 1, 1), verbose=False).reorient_(("R", "P", "I"))
        expected.shape = None  # type: ignore
        expected.rotation = None  # type: ignore
        self.assertEqual(expected, c)

    def test_roundtrip_global(self):
        for _ in range(repeats):
            p = get_poi(num_vert=8, num_subreg=2).to_global()
            file = Path(tempfile.gettempdir(), "test_rt_global.json")
            p.save(file, verbose=False)
            c = POI_Global.load(file)
            file.unlink(missing_ok=True)
            self.assertEqual(c, p)

    def test_roundtrip_mrk(self):
        for _ in range(repeats):
            p = get_poi(num_vert=8, num_subreg=2).to_global()
            file = Path(tempfile.gettempdir(), "test_rt.mrk.json")
            with _quiet():
                p.save_mrk(file)
            c = POI_Global.load(file)
            file.unlink(missing_ok=True)
            self.assertEqual(c, p)

    def test_save_poi_function_make_parents_and_info(self):
        p = get_poi(num_vert=3, num_subreg=1)
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "sub", "dir", "poi.json")
            save_poi(p, file, make_parents=True, additional_info={"my_key": "my_value"}, verbose=False, save_hint=2)
            self.assertTrue(file.exists())
            with file.open() as f:
                header = json.load(f)[0]
            self.assertEqual(header["my_key"], "my_value")
            c = load_poi(file)
            self.assertEqual(c, p)
            self.assertEqual(c.info.get("my_key"), "my_value")

    # ------------------------------------------------------------------ #
    # error / edge handling of save_poi                                  #
    # ------------------------------------------------------------------ #
    def test_save_bad_file_ending(self):
        p = get_poi(num_vert=2, num_subreg=1)
        with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError):
            save_poi(p, Path(d, "poi.txt"), verbose=False)

    def test_save_empty_poi_writes_nothing(self):
        empty = POI(POI_Descriptor(), orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=(10, 10, 10))
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "empty.json")
            save_poi(empty, file, verbose=False)
            self.assertFalse(file.exists())

    # ------------------------------------------------------------------ #
    # load-only formats: craft a tiny valid file and load it             #
    # ------------------------------------------------------------------ #
    def test_load_spine_r2(self):
        data = {
            "centroids": {
                "centroids": [
                    {"direction": ["R", "A", "S"]},
                    {"label": 5, "X": 1.0, "Y": 2.0, "Z": 3.0},
                    {"label": 7, "X": 4.0, "Y": 5.0, "Z": 6.0},
                ]
            },
            "Spacing": [1.0, 1.0, 1.0],
            "Shape": [50, 50, 50],
        }
        # direct loader
        poi = _load_form_POI_spine_r2(data)
        self.assertEqual(poi[5, 50], (1.0, 2.0, 3.0))
        self.assertEqual(poi[7, 50], (4.0, 5.0, 6.0))
        self.assertEqual(tuple(poi.orientation), ("R", "A", "S"))
        # via load_poi dispatch on a real file
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "spine_r2.json")
            file.write_text(json.dumps(data))
            poi2 = load_poi(file)
            self.assertEqual(poi2[5, 50], (1.0, 2.0, 3.0))

    def test_load_format_poi_old_crafted(self):
        dict_list = [
            {"vert_label": "8", "85": "(281, 185, 274)", "81": "(1.5, 2.5, 3.5)"},
            {"vert_label": "9", "50": "(10, 20, 30)"},
        ]
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "old.json")
            file.write_text(json.dumps(dict_list))
            poi = load_poi(file)
        self.assertEqual(poi[8, 85], (281.0, 185.0, 274.0))
        self.assertEqual(poi[8, 81], (1.5, 2.5, 3.5))
        self.assertEqual(poi[9, 50], (10.0, 20.0, 30.0))
        self.assertEqual(tuple(poi.orientation), ("R", "P", "I"))

    def test_load_docker_centroids_variants(self):
        dict_list = [
            {"direction": ["R", "A", "S"]},
            {"label": 50 * LABEL_MAX + 5, "X": 1.0, "Y": 2.0, "Z": 3.0},  # int -> (5, 50)
            {"label": 7, "X": 4.0, "Y": 5.0, "Z": 6.0},  # int, subreg 0 -> 50
            {"label": float("nan"), "X": 0.0, "Y": 0.0, "Z": 0.0},  # NaN -> skipped
            {"label": "TH1_SSL", "X": 7.0, "Y": 8.0, "Z": 9.0},  # gruber-name -> (8, 81)
        ]
        centroids = POI_Descriptor()
        with _quiet():
            _load_docker_centroids(dict_list, centroids, None)
        self.assertIn((5, 50), centroids)
        self.assertIn((7, 50), centroids)
        self.assertIn((8, 81), centroids)
        self.assertEqual(centroids[5, 50], (1.0, 2.0, 3.0))
        self.assertEqual(centroids[8, 81], (7.0, 8.0, 9.0))
        # the NaN entry must not have produced any extra key
        self.assertEqual(len(list(centroids.keys())), 3)

    def test_load_mrk_crafted_lps(self):
        mrk = {
            "@schema": "https://x/markups-schema-v1.0.3.json#",
            "coordinateSystem": "LPS",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "controlPoints": [
                        {
                            "id": "5-50",
                            "label": "5-50",
                            "position": [1.0, 2.0, 3.0],
                            "description": "vert5",
                            "associatedNodeID": "node5",
                        },
                        {"id": "5-81", "label": "5-81", "position": [4.0, 5.0, 6.0]},
                    ],
                }
            ],
            "display": {"color": [0.1, 0.2, 0.3]},
        }
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "points.mrk.json")
            file.write_text(json.dumps(mrk))
            with _quiet():
                poi = load_poi(file)
        self.assertIsInstance(poi, POI_Global)
        self.assertTrue(poi.itk_coords)  # LPS -> itk
        self.assertEqual(len(poi), 2)
        self.assertEqual(poi[5, 50], (1.0, 2.0, 3.0))
        self.assertEqual(poi.info.get("color"), [0.1, 0.2, 0.3])

    def test_load_mrk_warning_branches(self):
        # exercises the many defensive log.on_warning / skip branches of _load_mkr_POI:
        # missing @schema, non-Fiducial type, unknown coordinate system, unknown units,
        # a markup with no controlPoints, and a measurements key.
        mrk = {
            "markups": [
                {"type": "Line", "coordinateSystem": "RAS"},
                {"type": "Fiducial", "coordinateSystem": "GIBBERISH"},
                {"type": "Fiducial", "coordinateSystem": "RAS", "coordinateUnits": "inch"},
                {"type": "Fiducial", "coordinateSystem": "RAS", "coordinateUnits": "mm"},
                {
                    "type": "Fiducial",
                    "coordinateSystem": "RAS",
                    "coordinateUnits": "mm",
                    "measurements": [{"name": "len"}],
                    "controlPoints": [{"id": "3", "label": "3", "position": [1.0, 2.0, 3.0]}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "warn.mrk.json")
            file.write_text(json.dumps(mrk))
            with _quiet():
                poi = load_poi(file)
        self.assertIsInstance(poi, POI_Global)
        self.assertFalse(poi.itk_coords)  # RAS -> not itk
        self.assertEqual(len(poi), 1)
        self.assertEqual(poi[3, 1], (1.0, 2.0, 3.0))

    def test_load_landmark_txt(self):
        txt = (
            "format: POINT_LIST\n"
            "coordinate_system: nib\n"
            "Matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n"
            "shape: 256 931 27\n"
            "\n"  # blank line -> skipped
            "a comment line without a colon\n"  # no colon -> skipped
            "Femur proximal:\n"
            "hip_center: (10.0, 20.0, 30.0)\n"
            "knee: (40.0, 50.0, 60.0)\n"
            "Pelvis:\n"
            "asis: (1.0, 2.0, 3.0)\n"
        )
        with tempfile.TemporaryDirectory() as d:
            file = Path(d, "landmarks.txt")
            file.write_text(txt)
            # direct parser
            header, points = _load_landmark_txt(file)
            self.assertEqual(header["Matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.assertEqual(header["shape"], [256, 931, 27])
            self.assertEqual(points[1][1], [10.0, 20.0, 30.0])
            self.assertEqual(points[2][1], [1.0, 2.0, 3.0])
            # via load_poi -> POI_Global (FORMAT_PLST)
            poi = load_poi(file)
        self.assertIsInstance(poi, POI_Global)
        self.assertEqual(len(poi), 3)

    # ------------------------------------------------------------------ #
    # small parser helpers                                               #
    # ------------------------------------------------------------------ #
    def test_parse_coords(self):
        self.assertEqual(_parse_coords("(1.0, 2.5, -3.0)"), [1.0, 2.5, -3.0])
        self.assertEqual(_parse_coords(" ( 4 , 5 , 6 ) "), [4.0, 5.0, 6.0])
        with pytest.raises(ValueError):
            _parse_coords("1, 2, 3")  # missing parentheses
        with pytest.raises(ValueError):
            _parse_coords("(1, 2)")  # wrong number of values

    def test_parse_header_value(self):
        self.assertEqual(_parse_header_value("5"), 5)
        self.assertEqual(_parse_header_value("3.14"), 3.14)
        self.assertEqual(_parse_header_value("-3.5e-12"), -3.5e-12)
        self.assertEqual(_parse_header_value("256 931 27"), [256, 931, 27])
        self.assertEqual(_parse_header_value("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(_parse_header_value("[[1, 0, 0], [0, 1, 0]]"), [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(_parse_header_value("[]"), [])
        self.assertEqual(_parse_header_value("hello"), "hello")
        self.assertEqual(_parse_header_value("a b c"), "a b c")  # mixed -> kept as string

    def test_get_poi_idx_from_text(self):
        centroids = POI_Descriptor()
        self.assertEqual(_get_poi_idx_from_text("x", "3-7", centroids), (3, 7))
        self.assertEqual(_get_poi_idx_from_text("4-9", "lbl", centroids), (4, 9))
        self.assertEqual(_get_poi_idx_from_text("12", "lbl", centroids), (12, 1))
        # collision: (1, 1) taken -> bumps subregion
        centroids[(1, 1)] = (0.0, 0.0, 0.0)
        self.assertEqual(_get_poi_idx_from_text("1", "lbl", centroids), (1, 2))

    def test_get_poi_idx_from_text_name_fallbacks(self):
        # Non-integer names trigger the Any-registry resolution fall-backs (in the
        # label-dash, id-dash and bare-id branches). They must always yield a
        # valid (region, subregion) integer pair without raising.
        centroids = POI_Descriptor()
        for idx, label in (("x", "C2-Vertebra_Corpus"), ("C3-Vertebra_Corpus", "lbl"), ("Vertebra_Corpus", "lbl")):
            with self.subTest(idx=idx, label=label):
                with _quiet():
                    region, subregion = _get_poi_idx_from_text(idx, label, centroids)
                self.assertIsInstance(region, int)
                self.assertIsInstance(subregion, int)


if __name__ == "__main__":
    unittest.main()
