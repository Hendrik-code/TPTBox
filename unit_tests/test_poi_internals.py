"""Unit tests for POI serialization and the descriptor internals.

`unit_tests/test_poi_save.py` round-trips the current save format; this file covers
the readers for the other formats the loader still accepts (landmark TXT, Slicer
.mrk.json, the two legacy POI layouts), the header/coordinate parsers underneath
them, and the `POI_Descriptor` name-resolution and set-algebra methods.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from TPTBox import POI, Location, Vertebra_Instance
from TPTBox.core.poi_fun.poi_abstract import POI_Descriptor, label_name_dict, unpack_poi_id
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox.core.poi_fun.save_load import (
    _is_Point3D,
    _load_landmark_txt,
    _parse_coords,
    _parse_header_value,
    load_poi,
    save_poi,
)


def _poi(points=None) -> POI:
    points = points or {20: {50: (1.0, 2.0, 3.0)}, 21: {50: (4.0, 5.0, 6.0)}}
    return POI(
        points,
        orientation=("R", "A", "S"),
        zoom=(1.0, 1.0, 1.0),
        shape=(50.0, 50.0, 50.0),
        origin=(0.0, 0.0, 0.0),
        rotation=np.eye(3),
        level_one_info=Vertebra_Instance,
        level_two_info=Location,
    )


class TestParseCoords(unittest.TestCase):
    def test_valid_triples(self):
        for text, expected in (("(1, 2, 3)", [1.0, 2.0, 3.0]), ("( -1.5,2 , 3e2 )", [-1.5, 2.0, 300.0])):
            with self.subTest(text=text):
                self.assertEqual(_parse_coords(text), expected)

    def test_rejects_malformed_input(self):
        for text in ("1, 2, 3", "(1, 2)", "(1, 2, 3, 4)", "()"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _parse_coords(text)


class TestParseHeaderValue(unittest.TestCase):
    def test_documented_examples(self):
        cases = [
            ("3.14", 3.14),
            ("7", 7),
            ("-3.5e-12", -3.5e-12),
            ("256 931 27", [256, 931, 27]),
            ("[1, 2, 3]", [1, 2, 3]),
            ("[[1, 0, 0], [0, 1, 0]]", [[1, 0, 0], [0, 1, 0]]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_parse_header_value(text), expected)

    def test_types_are_preserved(self):
        self.assertIsInstance(_parse_header_value("7"), int)
        self.assertIsInstance(_parse_header_value("7.5"), float)
        self.assertIsInstance(_parse_header_value("[1, 2]"), list)
        self.assertIsInstance(_parse_header_value("plain text"), str)

    def test_a_whole_float_collapses_to_int(self):
        # "7.0" parses as int(7), not float(7.0) - int conversion is tried first.
        self.assertIsInstance(_parse_header_value("7.0"), int)

    def test_whitespace_is_stripped(self):
        self.assertEqual(_parse_header_value("  42  "), 42)


class TestIsPoint3D(unittest.TestCase):
    def test_accepts_a_full_point(self):
        self.assertTrue(_is_Point3D({"X": 1.0, "Y": 2.0, "Z": 3.0, "label": 1, "desc": "d"}))

    def test_rejects_incomplete_or_wrongly_typed_dicts(self):
        for obj in ({"X": 1.0, "Y": 2.0}, {"X": "a", "Y": 2.0, "Z": 3.0}, {}, []):
            with self.subTest(obj=obj):
                self.assertFalse(_is_Point3D(obj))

    def test_non_container_input_is_not_supported(self):
        # Documented limitation: the guard does membership tests, so a scalar raises.
        for obj in (None, 5):
            with self.subTest(obj=obj), self.assertRaises(TypeError):
                _is_Point3D(obj)


class TestLandmarkTxt(unittest.TestCase):
    TEXT = """some_meta: 256 931 27
GroupA:
left_hip: (1.0, 2.0, 3.0)
right_hip: (4.5, 5.5, 6.5)
GroupB:
apex: (7.0, 8.0, 9.0)
"""

    def _write(self, folder: Path, text: str | None = None) -> Path:
        path = folder / "landmarks.txt"
        path.write_text(self.TEXT if text is None else text, encoding="utf-8")
        return path

    def test_header_and_points_are_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            header, points = _load_landmark_txt(self._write(Path(tmp)))
            self.assertEqual(header["coordinate_system"], "nib")
            self.assertEqual(header["some_meta"], [256, 931, 27])
            self.assertEqual(len(points), 2, "two groups")

    def test_group_names_and_point_names_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            header, _ = _load_landmark_txt(self._write(Path(tmp)))
            self.assertIn("GroupA", header["label_group_name"].values())
            self.assertIn("GroupB", header["label_group_name"].values())
            flat_names = [n for group in header["label_name"].values() for n in group.values()]
            self.assertIn("left_hip", flat_names)
            self.assertIn("apex", flat_names)

    def test_coordinates_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, points = _load_landmark_txt(self._write(Path(tmp)))
            all_coords = [tuple(c) for group in points.values() for c in group.values()]
            self.assertIn((1.0, 2.0, 3.0), all_coords)
            self.assertIn((7.0, 8.0, 9.0), all_coords)

    def test_blank_and_colonless_lines_are_skipped(self):
        text = "some_meta: 1\n\nno colon here\nGroupA:\np: (1, 1, 1)\n"
        with tempfile.TemporaryDirectory() as tmp:
            _, points = _load_landmark_txt(self._write(Path(tmp), text))
            self.assertEqual(sum(len(g) for g in points.values()), 1)

    def test_loading_through_load_poi_gives_a_poi(self):
        with tempfile.TemporaryDirectory() as tmp:
            poi = load_poi(self._write(Path(tmp)), verbose=False)
            self.assertEqual(len(list(poi.keys())), 3)


class TestMarkupJson(unittest.TestCase):
    """Slicer .mrk.json is read as a world-space POI_Global."""

    def _mrk(self) -> dict:
        return {
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "controlPoints": [
                        {"id": "1", "label": "L1", "position": [1.0, 2.0, 3.0], "description": ""},
                        {"id": "2", "label": "L2", "position": [4.0, 5.0, 6.0], "description": ""},
                    ],
                }
            ]
        }

    def test_round_trip_through_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.mrk.json"
            path.write_text(json.dumps(self._mrk()), encoding="utf-8")
            poi = load_poi(path, verbose=False)
            self.assertIsInstance(poi, POI_Global)
            self.assertEqual(len(list(poi.keys())), 2)

    def test_saving_a_global_poi_as_mrk_and_reading_it_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.mrk.json"
            glob = _poi().to_global()
            glob.save_mrk(path)
            self.assertTrue(path.exists())
            reloaded = load_poi(path, verbose=False)
            self.assertEqual(sorted(reloaded.keys()), sorted(glob.keys()))
            for key in glob.keys():
                with self.subTest(key=key):
                    np.testing.assert_allclose(reloaded[key], glob[key], atol=1e-4)


class TestSavePoiFormats(unittest.TestCase):
    def test_every_save_hint_round_trips(self):
        # hint 0 = docker, 1 = gruber, 2 = POI (the default). Gruber names its points
        # from a fixed landmark vocabulary, so it gets points from that vocabulary.
        gruber_points = {20: {101: (1.0, 2.0, 3.0)}, 21: {102: (4.0, 5.0, 6.0)}}
        for hint in (0, 1, 2):
            with self.subTest(save_hint=hint), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "poi.json"
                poi = _poi(gruber_points if hint == 1 else None)
                save_poi(poi, path, save_hint=hint, verbose=False)
                self.assertTrue(path.exists())
                reloaded = load_poi(path, verbose=False)
                self.assertEqual(sorted(reloaded.keys()), sorted(poi.keys()))
                for key in poi.keys():
                    np.testing.assert_allclose(np.asarray(reloaded[key]), np.asarray(poi[key]), atol=1e-3)

    def test_the_gruber_format_rejects_points_outside_its_vocabulary(self):
        # Documented limitation: save_hint=1 maps each subregion id through a fixed
        # landmark-name table, so an ordinary corpus centroid (50) has no name.
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(KeyError):
            save_poi(_poi(), Path(tmp) / "poi.json", save_hint=1, verbose=False)

    def test_additional_info_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poi.json"
            save_poi(_poi(), path, additional_info={"my_key": "my_value"}, verbose=False)
            self.assertIn("my_value", path.read_text(encoding="utf-8"))

    def test_make_parents_creates_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "poi.json"
            save_poi(_poi(), path, make_parents=True, verbose=False)
            self.assertTrue(path.exists())

    def test_round_trip_survives_reorient_and_rescale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poi.json"
            poi = _poi()
            save_poi(poi, path, verbose=False)
            reloaded = load_poi(path, verbose=False)
            reloaded.reorient_(("P", "I", "R")).rescale_((2.0, 2.0, 2.0)).rescale_((1.0, 1.0, 1.0)).reorient_(("R", "A", "S"))
            for key in poi.keys():
                with self.subTest(key=key):
                    np.testing.assert_allclose(np.asarray(reloaded[key]), np.asarray(poi[key]), atol=1e-3)


class TestPoiDescriptor(unittest.TestCase):
    def test_str_to_int_resolves_names(self):
        desc = POI_Descriptor()
        self.assertEqual(desc.str_to_int("T12", subregion=False), Vertebra_Instance.T12.value)
        self.assertEqual(desc.str_to_int("Vertebra_Corpus", subregion=True), Location.Vertebra_Corpus.value)

    def test_str_to_int_list(self):
        desc = POI_Descriptor()
        self.assertEqual(desc.str_to_int_list("L1", "L2"), [Vertebra_Instance.L1.value, Vertebra_Instance.L2.value])

    def test_str_to_int_dict(self):
        desc = POI_Descriptor()
        out = desc.str_to_int_dict({"L1": "L2"})
        self.assertEqual(out, {Vertebra_Instance.L1.value: Vertebra_Instance.L2.value})

    def test_get_and_pop(self):
        poi = _poi()
        desc = poi.centroids
        np.testing.assert_allclose(desc.get((20, 50)), (1.0, 2.0, 3.0))
        popped = desc.pop((20, 50), None)
        np.testing.assert_allclose(popped, (1.0, 2.0, 3.0))
        self.assertNotIn((20, 50), poi, "pop removes the point")
        self.assertIsNone(desc.pop((20, 50), None), "popping a missing key returns the default")

    def test_items_2d(self):
        poi = _poi()
        nested = dict(poi.centroids.items_2d())
        self.assertEqual(sorted(nested), [20, 21])
        self.assertEqual(sorted(nested[20]), [50])

    def test_delitem(self):
        poi = _poi()
        del poi.centroids[20, 50]
        self.assertNotIn((20, 50), poi)

    def test_normalize_input_data_accepts_flat_and_nested(self):
        flat = POI_Descriptor.normalize_input_data({(1, 50): (0.0, 0.0, 0.0)})
        nested = POI_Descriptor.normalize_input_data({1: {50: (0.0, 0.0, 0.0)}})
        self.assertEqual(sorted(flat.keys()), sorted(nested.keys()))

    def test_unpack_poi_id_accepts_every_key_form(self):
        definition = _poi().centroids.definition
        packed = unpack_poi_id((20, 50), definition)
        self.assertEqual(packed, (20, 50))
        self.assertEqual(unpack_poi_id(slice(20, 50), definition), (20, 50))
        self.assertEqual(unpack_poi_id((Vertebra_Instance.L1, Location.Vertebra_Corpus), definition), (20, 50))

    def test_unpack_poi_id_round_trips_a_packed_integer(self):
        from TPTBox.core import vert_constants  # noqa: PLC0415

        definition = _poi().centroids.definition
        key = 20 + 50 * vert_constants.LABEL_MAX
        self.assertEqual(unpack_poi_id(key, definition), (20, 50))

    def test_label_name_dict_normalises_keys(self):
        self.assertEqual(label_name_dict({"label_name": {"1": {"2": "C2"}}}), {1: {2: "C2"}})


class TestAbstractPoiOperations(unittest.TestCase):
    def test_extract_keeps_only_the_requested_pairs(self):
        poi = _poi({20: {50: (1.0, 1.0, 1.0), 51: (2.0, 2.0, 2.0)}, 21: {50: (3.0, 3.0, 3.0)}})
        out = poi.extract((20, 50))
        self.assertEqual(sorted(out.keys()), [(20, 50)])
        self.assertEqual(len(list(poi.keys())), 3, "the source must be untouched")

    def test_distances_between_two_pois(self):
        a = _poi({20: {50: (0.0, 0.0, 0.0)}})
        b = _poi({20: {50: (3.0, 4.0, 0.0)}})
        out = a.calculate_distances_poi(b)
        self.assertAlmostEqual(float(next(iter(out.values()))), 5.0, places=6)

    def test_distances_any_to_any(self):
        a = _poi({20: {50: (0.0, 0.0, 0.0)}})
        b = _poi({20: {50: (3.0, 4.0, 0.0)}, 21: {50: (0.0, 0.0, 10.0)}})
        out = a.calculate_distances_poi_any_2_any(b)
        self.assertTrue(out)
        flat = [float(v) for inner in out.values() for v in (inner.values() if isinstance(inner, dict) else [inner])]
        self.assertAlmostEqual(min(flat), 5.0, places=6)

    def test_distances_across_regions(self):
        a = _poi({20: {50: (0.0, 0.0, 0.0)}, 21: {50: (0.0, 0.0, 6.0)}})
        out = a.calculate_distances_poi_across_regions(a.copy())
        self.assertTrue(out)

    def test_operators(self):
        a = _poi({20: {50: (1.0, 1.0, 1.0)}})
        b = _poi({21: {50: (2.0, 2.0, 2.0)}})
        self.assertEqual(sorted((a + b).keys()), [(20, 50), (21, 50)])
        self.assertEqual(sorted((a >> b).keys()), [(20, 50), (21, 50)])
        self.assertEqual(sorted((a - a.copy()).keys()), [])
        self.assertEqual(sorted((a & a.copy()).keys()), [(20, 50)])

    def test_remove_centroid_inplace(self):
        poi = _poi()
        poi.remove_centroid_((20, 50))
        self.assertNotIn((20, 50), poi)

    def test_join_right_inplace(self):
        a = _poi({20: {50: (1.0, 1.0, 1.0)}})
        b = _poi({21: {50: (2.0, 2.0, 2.0)}})
        a.join_right_(b)
        self.assertEqual(sorted(a.keys()), [(20, 50), (21, 50)])

    def test_fit_spline_through_four_points(self):
        poi = _poi({v: {Location.Vertebra_Corpus.value: (0.0, 0.0, float(10 * i))} for i, v in enumerate((20, 21, 22, 23))})
        points, derivatives = poi.fit_spline(location=Location.Vertebra_Corpus, vertebra=True)
        self.assertGreater(len(points), 4)
        self.assertEqual(np.asarray(points).shape[1], 3)
        self.assertEqual(np.asarray(derivatives).shape[1], 3)


class TestPoiCropAndGlobal(unittest.TestCase):
    def test_apply_crop_reverse_is_the_inverse_of_apply_crop(self):
        poi = _poi({20: {50: (10.0, 12.0, 14.0)}})
        crop = (slice(5, 40), slice(5, 40), slice(5, 40))
        cropped = poi.apply_crop(crop)
        restored = cropped.apply_crop_reverse(crop, poi.shape)
        np.testing.assert_allclose(restored[20, 50], poi[20, 50], atol=1e-9)
        self.assertEqual(restored.shape, poi.shape)

    def test_local_to_global_arr_round_trip(self):
        poi = _poi()
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        world = poi.local_to_global_arr(points)
        np.testing.assert_allclose(poi.global_to_local_arr(world), points, atol=1e-9)

    def test_to_global_and_back(self):
        poi = _poi()
        back = poi.to_global().to_local(poi)
        for key in poi.keys():
            with self.subTest(key=key):
                np.testing.assert_allclose(back[key], poi[key], atol=1e-6)

    def test_to_cord_system_is_an_involution(self):
        glob = _poi().to_global()
        flipped = glob.to_cord_system(itk_coords=True)
        back = flipped.to_cord_system(itk_coords=False)
        for key in glob.keys():
            with self.subTest(key=key):
                np.testing.assert_allclose(back[key], glob[key], atol=1e-9)

    def test_to_other_maps_onto_another_grid(self):
        poi = _poi()
        other = poi.copy()
        other.rescale_((2.0, 2.0, 2.0))
        out = poi.to_global().to_other(other)
        self.assertEqual(sorted(out.keys()), sorted(poi.keys()))

    def test_make_point_cloud_nii(self):
        poi = _poi()
        cloud, sphere = poi.make_point_cloud_nii(s=3)
        self.assertGreater(int((cloud.get_seg_array() != 0).sum()), 0)
        self.assertGreater(int((sphere.get_seg_array() != 0).sum()), 0)

    def test_resample_from_to_inplace(self):
        poi = _poi()
        target = poi.copy()
        target.rescale_((2.0, 2.0, 2.0))
        poi.resample_from_to_(target)
        np.testing.assert_allclose(poi.zoom, target.zoom)


if __name__ == "__main__":
    unittest.main()
