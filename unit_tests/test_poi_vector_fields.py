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
