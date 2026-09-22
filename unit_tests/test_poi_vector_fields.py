"""Unit tests for auto-transform of direction-vector and label-keyed fields in ``POI.info``.

Covers:
- ``POI.reorient`` — signed permutation of direction vectors.
- ``POI.resample_from_to`` — full rotation (``R_ref.T @ R_self``).
- ``POI.rescale`` — no-op for mm-space vectors.
- ``POI_Global.to_cord_system`` — RAS <-> LPS axis flips.
- ``POI.map_labels`` — key remap for both vector and label-keyed fields.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

from typing import ClassVar  # noqa: E402

import numpy as np  # noqa: E402

from TPTBox.core.poi import POI  # noqa: E402
from TPTBox.core.poi_fun.vector_fields import (  # noqa: E402
    POI_INFO_LABEL_KEYED_FIELDS_KEY,
    POI_INFO_VECTOR_FIELDS_KEY,
)


def _make_poi(orientation=("P", "I", "R"), zoom=(1.0, 1.0, 1.0)) -> POI:
    """Build a small POI with identity rotation and one L1 corpus point."""
    centroids: dict[int, dict[int, tuple[float, float, float]]] = {20: {50: (1.0, 2.0, 3.0)}}
    return POI(
        centroids,
        orientation=orientation,
        zoom=zoom,
        shape=(100, 100, 100),
        origin=(0.0, 0.0, 0.0),
        rotation=np.eye(3),
    )


def _register(poi: POI, vec_fields=(), lbl_fields=()) -> None:
    if vec_fields:
        poi.info.setdefault(POI_INFO_VECTOR_FIELDS_KEY, []).extend(vec_fields)
    if lbl_fields:
        poi.info.setdefault(POI_INFO_LABEL_KEYED_FIELDS_KEY, []).extend(lbl_fields)


class Test_Vector_Field_Reorient(unittest.TestCase):
    def test_pir_to_las_signed_permutation(self):
        # In PIR, +x=P, +y=I, +z=R. In LAS, +x=L=-R, +y=A=-P, +z=S=-I.
        # v_PIR = (P=0.020, I=-0.092, R=0.996). Same physical direction in LAS is:
        #   L=-R=-0.996, A=-P=-0.020, S=-I=0.092 -> (-0.996, -0.020, 0.092).
        poi = _make_poi(orientation=("P", "I", "R"))
        _register(poi, vec_fields=["v"])
        v_in = np.array([0.02, -0.09, 0.996])
        v_in = v_in / np.linalg.norm(v_in)
        poi.info["v"] = {"L1": tuple(v_in)}
        out = poi.reorient(("L", "A", "S"))
        got = np.asarray(out.info["v"]["L1"])
        expected = np.array([-v_in[2], -v_in[0], -v_in[1]])
        np.testing.assert_allclose(got, expected, atol=1e-9)
        # Norm is preserved by a signed permutation.
        self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=6)
        # Full round-trip: PIR -> LAS -> PIR equals original.
        back = out.reorient(("P", "I", "R"))
        np.testing.assert_allclose(back.info["v"]["L1"], v_in, atol=1e-9)

    def test_unregistered_field_untouched(self):
        poi = _make_poi()
        # not registered
        poi.info["orphan_vec"] = {"L1": (1.0, 0.0, 0.0)}
        out = poi.reorient(("L", "A", "S"))
        self.assertEqual(tuple(out.info["orphan_vec"]["L1"]), (1.0, 0.0, 0.0))

    def test_none_or_malformed_values_skipped(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        poi.info["v"] = {"L1": (0.0, 1.0, 0.0), "L2": None, "L3": (1.0, 2.0)}
        out = poi.reorient(("L", "A", "S"))
        self.assertIsNone(out.info["v"]["L2"])
        # 2-tuple stays unchanged, since shape mismatch is silently skipped
        self.assertEqual(tuple(out.info["v"]["L3"]), (1.0, 2.0))
        # 3-tuple got transformed
        self.assertEqual(len(out.info["v"]["L1"]), 3)


class Test_Vector_Field_Rescale(unittest.TestCase):
    def test_rescale_leaves_vectors_untouched(self):
        poi = _make_poi(zoom=(0.5, 0.5, 3.0))
        _register(poi, vec_fields=["v"])
        v = (0.1, -0.9, 0.05)
        poi.info["v"] = {"L1": v}
        out = poi.rescale((1.0, 1.0, 1.0))
        np.testing.assert_allclose(out.info["v"]["L1"], v, atol=1e-12)


class Test_Vector_Field_ResampleFromTo(unittest.TestCase):
    def test_identity_grid_is_noop(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        v = (0.09, -0.99, -0.02)
        poi.info["v"] = {"L1": v}
        # ref = a copy of the same POI -> same rotation -> R_ref.T @ R_self = I
        ref = poi.copy()
        out = poi.resample_from_to(ref)
        np.testing.assert_allclose(out.info["v"]["L1"], v, atol=1e-6)

    def test_rotated_ref_applies_rotation(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        v = (1.0, 0.0, 0.0)
        poi.info["v"] = {"L1": v}
        # Target POI has a 90-degree z-rotation
        R_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        ref = poi.copy()
        ref.rotation = R_z90
        out = poi.resample_from_to(ref)
        # v_target = R_ref.T @ R_self @ v = R_z90.T @ I @ [1,0,0] = R_z90.T @ [1,0,0] = [0, -1, 0]
        np.testing.assert_allclose(out.info["v"]["L1"], (0.0, -1.0, 0.0), atol=1e-9)


class Test_ToCordSystem(unittest.TestCase):
    def test_ras_to_lps_flips_xy_only(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        poi.info["v"] = {"L1": (0.1, 0.2, 0.3)}
        g = poi.to_global()  # RAS by default
        self.assertFalse(g.itk_coords)
        self.assertEqual(tuple(g.info["v"]["L1"]), (0.1, 0.2, 0.3))
        g_itk = g.to_cord_system(itk_coords=True)
        np.testing.assert_allclose(g_itk.info["v"]["L1"], (-0.1, -0.2, 0.3), atol=1e-12)
        # roundtrip
        g_back = g_itk.to_cord_system(itk_coords=False)
        np.testing.assert_allclose(g_back.info["v"]["L1"], (0.1, 0.2, 0.3), atol=1e-12)


class Test_MapLabels(unittest.TestCase):
    def test_vector_field_keys_remapped(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        v = (0.1, -0.9, 0.4)
        poi.info["v"] = {"L1": v}
        out = poi.map_labels(label_map_region={20: 2})  # L1 -> C2
        self.assertNotIn("L1", out.info["v"])
        self.assertIn("C2", out.info["v"])
        np.testing.assert_allclose(out.info["v"]["C2"], v, atol=1e-12)

    def test_label_keyed_scalar_field_remapped(self):
        poi = _make_poi()
        _register(poi, lbl_fields=["endplate_internal_angle"])
        poi.info["endplate_internal_angle"] = {"L1": 3.14}
        out = poi.map_labels(label_map_region={20: 2})
        self.assertNotIn("L1", out.info["endplate_internal_angle"])
        self.assertAlmostEqual(out.info["endplate_internal_angle"]["C2"], 3.14)

    def test_int_keys_also_remapped(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        poi.info["v"] = {20: (0.0, 1.0, 0.0), 21: (1.0, 0.0, 0.0)}
        out = poi.map_labels(label_map_region={20: 2})
        self.assertNotIn(20, out.info["v"])
        self.assertIn(2, out.info["v"])
        # unaffected key
        self.assertIn(21, out.info["v"])
        np.testing.assert_allclose(out.info["v"][2], (0.0, 1.0, 0.0), atol=1e-12)

    def test_unknown_string_key_kept(self):
        poi = _make_poi()
        _register(poi, lbl_fields=["s"])
        poi.info["s"] = {"NOT_A_VERT": 42.0, "L1": 1.0}
        out = poi.map_labels(label_map_region={20: 2})
        self.assertIn("NOT_A_VERT", out.info["s"])
        self.assertNotIn("L1", out.info["s"])
        self.assertIn("C2", out.info["s"])

    def test_empty_map_is_noop(self):
        poi = _make_poi()
        _register(poi, vec_fields=["v"])
        v = (0.1, -0.9, 0.4)
        poi.info["v"] = {"L1": v}
        out = poi.map_labels(label_map_region={})
        self.assertEqual(tuple(out.info["v"]["L1"]), v)


class Test_LabelName_MapLabels(unittest.TestCase):
    """label_name (nested {region: {subregion: name, "name": group}}) rides map_labels."""

    def _poi_with_label_name(self) -> POI:
        poi = _make_poi()
        # nested format directly
        poi.info["label_name"] = {
            20: {50: "L1_corpus", 100: "L1_disc", "name": "Spine"},
            21: {50: "L2_corpus"},
        }
        return poi

    def test_region_remap(self):
        poi = self._poi_with_label_name()
        out = poi.map_labels(label_map_region={20: 2})
        ln = out.info["label_name"]
        self.assertNotIn(20, ln)
        self.assertIn(2, ln)
        # inner subregion keys unchanged, group name preserved
        self.assertEqual(ln[2][50], "L1_corpus")
        self.assertEqual(ln[2][100], "L1_disc")
        self.assertEqual(ln[2]["name"], "Spine")
        # unaffected region still there
        self.assertEqual(ln[21][50], "L2_corpus")

    def test_subregion_remap(self):
        poi = self._poi_with_label_name()
        out = poi.map_labels(label_map_subregion={50: 51})
        ln = out.info["label_name"]
        # region keys unchanged
        self.assertIn(20, ln)
        self.assertNotIn(50, ln[20])
        self.assertIn(51, ln[20])
        self.assertEqual(ln[20][51], "L1_corpus")
        # group name key survives
        self.assertEqual(ln[20]["name"], "Spine")
        # unrelated subregion 100 preserved
        self.assertEqual(ln[20][100], "L1_disc")

    def test_region_and_subregion_remap(self):
        poi = self._poi_with_label_name()
        out = poi.map_labels(label_map_region={20: 2}, label_map_subregion={50: 51})
        ln = out.info["label_name"]
        self.assertEqual(ln[2][51], "L1_corpus")
        self.assertEqual(ln[2]["name"], "Spine")

    def test_region_collision_last_write_wins_on_inner(self):
        # Both 20 and 21 map onto 2: their inner dicts should merge; overlapping
        # inner keys let the later region's value win.
        poi = _make_poi()
        poi.info["label_name"] = {
            20: {50: "A", "name": "grp20"},
            21: {50: "B", 60: "unique"},
        }
        out = poi.map_labels(label_map_region={20: 2, 21: 2})
        ln = out.info["label_name"]
        self.assertIn(2, ln)
        # value at key 50 should come from the second insert (21 -> "B")
        self.assertEqual(ln[2][50], "B")
        # inner keys from both are preserved
        self.assertEqual(ln[2][60], "unique")
        # group name from region 20 comes along
        self.assertEqual(ln[2].get("name"), "grp20")

    def test_flat_legacy_format_migrated_and_remapped(self):
        # Ensure the migration path in label_name_dict / normalize_label_name is
        # triggered by the remap, so old flat "(region, subreg)" strings work too.
        poi = _make_poi()
        poi.info["label_name"] = {"(20, 50)": "L1_corpus"}
        out = poi.map_labels(label_map_region={20: 2})
        ln = out.info["label_name"]
        self.assertIn(2, ln)
        self.assertEqual(ln[2][50], "L1_corpus")


class Test_LabelName_LegacyMigration(unittest.TestCase):
    """The old flat ``{"(region, subreg)": name}`` format must migrate cleanly.

    Real-world fixture: a leg-atlas POI with 38 flat entries across 5 regions,
    including multi-digit subregion ids like ``(2, 10)`` — those must survive
    ``ast.literal_eval`` parsing.
    """

    _ATLAS_FLAT: ClassVar[dict[str, str]] = {
        "(1, 1)": "TGT",
        "(1, 2)": "FHC",
        "(1, 3)": "FNC",
        "(1, 4)": "FAAP",
        "(2, 1)": "FLCD",
        "(2, 2)": "FMCD",
        "(2, 3)": "FLCP",
        "(2, 4)": "FMCP",
        "(2, 5)": "FNP",
        "(2, 6)": "FADP",
        "(2, 7)": "TGPP",
        "(2, 8)": "TGCP",
        "(2, 9)": "FMCPC",
        "(2, 10)": "FLCPC",
        "(2, 11)": "TRMP",
        "(2, 12)": "TRLP",
        "(3, 1)": "TLCL",
        "(3, 2)": "TMCM",
        "(3, 3)": "TKC",
        "(3, 4)": "TLCA",
        "(3, 5)": "TLCP",
        "(3, 6)": "TMCA",
        "(3, 7)": "TMCP",
        "(3, 8)": "TTP",
        "(3, 9)": "TAAP",
        "(3, 10)": "TMIT",
        "(3, 11)": "TLIT",
        "(4, 1)": "FLM",
        "(4, 2)": "TMM",
        "(4, 3)": "TAC",
        "(4, 4)": "TADP",
        "(5, 1)": "PPP",
        "(5, 2)": "PDP",
        "(5, 3)": "PMP",
        "(5, 4)": "PLP",
        "(5, 5)": "PRPP",
        "(5, 6)": "PRDP",
        "(5, 7)": "PRHP",
    }

    def test_normalize_label_name_migrates_flat_atlas(self):
        from TPTBox.core.poi_fun.poi_abstract import normalize_label_name

        nested = normalize_label_name(dict(self._ATLAS_FLAT))
        # region keys are ints
        self.assertEqual(set(nested.keys()), {1, 2, 3, 4, 5})
        # multi-digit inner keys survive parsing
        self.assertEqual(nested[2][10], "FLCPC")
        self.assertEqual(nested[2][12], "TRLP")
        self.assertEqual(nested[3][11], "TLIT")
        # inner keys are ints too, no leftover string keys
        for region, inner in nested.items():
            for k in inner:
                self.assertIsInstance(k, int, f"inner key {k!r} in region {region} is not int")
        # count is preserved
        total = sum(len(v) for v in nested.values())
        self.assertEqual(total, len(self._ATLAS_FLAT))

    def test_normalize_is_idempotent(self):
        from TPTBox.core.poi_fun.poi_abstract import normalize_label_name

        once = normalize_label_name(dict(self._ATLAS_FLAT))
        twice = normalize_label_name({**once})
        self.assertEqual(once, twice)

    def test_normalize_empty_and_none(self):
        from TPTBox.core.poi_fun.poi_abstract import normalize_label_name

        self.assertEqual(normalize_label_name(None), {})
        self.assertEqual(normalize_label_name({}), {})

    def test_label_name_dict_caches_migration_in_info(self):
        from TPTBox.core.poi_fun.poi_abstract import label_name_dict

        info = {"label_name": dict(self._ATLAS_FLAT)}
        out = label_name_dict(info)
        # returned dict is the normalized form
        self.assertEqual(out[1][1], "TGT")
        # and the migrated form is cached back into info
        self.assertIs(info["label_name"], out)
        # a second call is a no-op (still nested)
        out2 = label_name_dict(info)
        self.assertIs(out2, info["label_name"])

    def test_load_poi_migrates_atlas_from_disk(self):
        """End-to-end: write the legacy JSON, load it, expect the nested form."""
        import json
        import tempfile

        from TPTBox.core.poi import POI

        payload = [
            {
                "direction": ["R", "A", "S"],
                "zoom": [1.0, 1.0, 1.0],
                "origin": [0.0, 0.0, 0.0],
                "shape": [10, 10, 10],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "format": "POI",
                "label_name": dict(self._ATLAS_FLAT),
            },
            # one dummy point so the file loads as a POI: {region: {subregion: (x,y,z)}}
            {"1": {"1": [1.0, 2.0, 3.0]}},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix="_poi.json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            poi = POI.load(path)
            ln = poi.info["label_name"]
            self.assertEqual(ln[1][1], "TGT")
            self.assertEqual(ln[2][10], "FLCPC")
            # no more flat "(...)"-style keys
            self.assertFalse(any(isinstance(k, str) and k.startswith("(") for k in ln))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_migration_then_map_labels(self):
        """A flat-format POI still remaps correctly through map_labels."""
        poi = _make_poi()
        poi.info["label_name"] = dict(self._ATLAS_FLAT)  # legacy flat
        # rename region 2 -> 20 and region 5 -> 50 in one shot
        out = poi.map_labels(label_map_region={2: 20, 5: 50})
        ln = out.info["label_name"]
        self.assertIn(20, ln)
        self.assertIn(50, ln)
        self.assertNotIn(2, ln)
        self.assertNotIn(5, ln)
        self.assertEqual(ln[20][10], "FLCPC")
        self.assertEqual(ln[50][7], "PRHP")


class Test_LabelName_Accessors(unittest.TestCase):
    """`set_label_name` / `set_level_one_name` write into ``info['label_name']``
    and get read back by ``label_name`` / ``level_one_name`` and by the
    Slicer/mkr exporter.
    """

    def _poi_with_enums(self) -> POI:
        from TPTBox.core.vert_constants import Location, Vertebra_Instance

        poi = _make_poi()
        poi.level_one_info = Vertebra_Instance
        poi.level_two_info = Location
        return poi

    def test_set_and_get_label_name(self):
        poi = self._poi_with_enums()
        poi.set_label_name(region=20, subregion=50, name="L1_corpus")
        self.assertEqual(poi.label_name(20, 50), "L1_corpus")
        # unset points fall back to the level_two_info enum name (or the raw id when
        # no enum entry matches).
        fallback = poi.label_name(21, 50)
        self.assertIsInstance(fallback, str)
        self.assertNotEqual(fallback, "L1_corpus")

    def test_set_and_get_level_one_name(self):
        poi = self._poi_with_enums()
        poi.set_level_one_name(region=20, name="Vertebra L1 custom")
        self.assertEqual(poi.level_one_name(20), "Vertebra L1 custom")
        # unset region falls back to the level_one_info enum name (L1 -> "L1")
        self.assertEqual(poi.level_one_name(21), "L2")

    def test_set_label_name_persists_in_info(self):
        poi = _make_poi()
        poi.set_label_name(20, 50, "L1_corpus")
        poi.set_level_one_name(20, "Femur")
        ln = poi.info["label_name"]
        self.assertEqual(ln[20][50], "L1_corpus")
        self.assertEqual(ln[20]["name"], "Femur")

    def test_names_flow_into_slicer_export(self):
        """`get_desc` (used by save_mrk) reads label/group name from label_name."""
        from TPTBox.core.poi_fun.save_mkr import get_desc

        poi = _make_poi()
        poi.set_label_name(20, 50, "L1_corpus")
        poi.set_level_one_name(20, "Spine")
        g = poi.to_global()
        name, name2, label = get_desc(g, region=20, subregion=50)
        # `label` is the per-point custom name; `name2` is the region group name.
        self.assertEqual(label, "L1_corpus")
        self.assertEqual(name2, "Spine")


class Test_Composition(unittest.TestCase):
    def test_reorient_and_map_labels_commute(self):
        # For a vector-transform + key-remap, order shouldn't matter.
        poi_a = _make_poi()
        poi_b = _make_poi()
        for poi in (poi_a, poi_b):
            _register(poi, vec_fields=["v"])
            poi.info["v"] = {"L1": (0.02, -0.09, 0.996)}
        out_a = poi_a.reorient(("L", "A", "S")).map_labels(label_map_region={20: 2})
        out_b = poi_b.map_labels(label_map_region={20: 2}).reorient(("L", "A", "S"))
        np.testing.assert_allclose(out_a.info["v"]["C2"], out_b.info["v"]["C2"], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
