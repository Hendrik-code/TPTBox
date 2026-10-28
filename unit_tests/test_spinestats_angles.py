"""Unit tests for `TPTBox.spine.spinestats` angle, curvature and pelvic measurements.

The shipped sample data holds a single vertebra, so every spine here is synthesised
analytically: vertebra bodies stacked along the superior axis, each carrying the
direction POIs (`Vertebra_Direction_Posterior/Inferior/Right`) and the two endplate
median points that `MoveTo.TOP/BOTTOM` resolve to. Tilting one vertebra by a known
angle therefore gives a known expected measurement, which is what these tests assert -
a circle of radius r must yield curvature 1/r, a 30 degree coronal tilt must yield a
30 degree Cobb angle, and pelvic incidence must equal pelvic tilt plus sacral slope.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from TPTBox import POI, Location, Vertebra_Instance
from TPTBox.spine.spinestats import angles as angles_mod
from TPTBox.spine.spinestats import curvature as curv_mod
from TPTBox.spine.spinestats.angles import (
    MoveTo,
    compute_angel_between_two_points_,
    compute_lordosis_and_kyphosis,
    compute_max_cobb_angle,
    compute_max_cobb_angle_multi,
    cosine_distance,
    get_to_space,
)
from TPTBox.spine.spinestats.curvature import (
    _find_apices,
    _numerical_curvature,
    compute_axial_rotation,
    compute_coronal_balance,
    compute_curvature_profile,
    compute_multi_cobb,
    compute_segmental_endplate_angles,
    compute_sva,
    compute_wedge_metrics,
)
from TPTBox.spine.spinestats.pelvic_parameters import (
    _pelvic_from_endplate_pair,
    _sag,
    compute_pelvic_parameters,
    resolve_fullbody_poi_path,
)

V = Vertebra_Instance
SPINE_IDS = [v.value for v in list(V.cervical()) + list(V.thoracic()) + list(V.lumbar()) if v.value <= V.L5.value]


def rot(axis: str, deg: float) -> np.ndarray:
    """Rotation matrix about a RAS axis: x = R-L, y = A-P, z = S-I."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def spine_poi(
    ids=None,
    sagittal_tilt=None,
    coronal_tilt=None,
    axial_tilt=None,
    spacing: float = 20.0,
    half_height: float = 5.0,
    coronal_amp: float = 0.0,
    sagittal_radius: float | None = None,
    anterior_shift=None,
) -> POI:
    """Build a synthetic spine POI in RAS at 1 mm isotropic spacing.

    Args:
        ids: Vertebra ids, inferior-to-superior order of increasing z. Defaults to C1..L5.
        sagittal_tilt / coronal_tilt / axial_tilt: per-id rotation in degrees about the
            R-L / A-P / S-I axis, applied to that vertebra's direction and endplate POIs.
        spacing: distance between consecutive body centres in mm.
        half_height: distance from body centre to each endplate median point in mm.
        coronal_amp: amplitude of a half-sine lateral deviation across the whole spine.
        sagittal_radius: if given, place the bodies on a circular arc of that radius.
        anterior_shift: per-id extra offset along the anterior axis in mm.
    """
    ids = SPINE_IDS if ids is None else ids
    sagittal_tilt = sagittal_tilt or {}
    coronal_tilt = coronal_tilt or {}
    axial_tilt = axial_tilt or {}
    anterior_shift = anterior_shift or {}
    cent: dict[int, dict[int, tuple[float, float, float]]] = {}
    n = len(ids)
    for k, vid in enumerate(ids):
        if sagittal_radius:
            # Pure circular arc of +-20 degrees: curvature is exactly 1/sagittal_radius.
            th = math.radians(-20 + 40 * k / max(n - 1, 1))
            y = sagittal_radius * math.sin(th)
            z = sagittal_radius * (1 - math.cos(th))
        else:
            y, z = 0.0, spacing * k
        x = coronal_amp * math.sin(math.pi * k / max(n - 1, 1)) if n > 1 else 0.0
        centre = np.array([x, y + anterior_shift.get(vid, 0.0), z])
        r = rot("y", coronal_tilt.get(vid, 0.0)) @ rot("x", sagittal_tilt.get(vid, 0.0)) @ rot("z", axial_tilt.get(vid, 0.0))
        cent[vid] = {
            Location.Vertebra_Corpus.value: tuple(centre),
            Location.Vertebra_Direction_Posterior.value: tuple(centre + r @ np.array([0.0, -1.0, 0.0])),
            Location.Vertebra_Direction_Inferior.value: tuple(centre + r @ np.array([0.0, 0.0, -1.0])),
            Location.Vertebra_Direction_Right.value: tuple(centre + r @ np.array([1.0, 0.0, 0.0])),
            Location.Additional_Vertebral_Body_Middle_Superior_Median.value: tuple(centre + r @ np.array([0.0, 0.0, half_height])),
            Location.Additional_Vertebral_Body_Middle_Inferior_Median.value: tuple(centre + r @ np.array([0.0, 0.0, -half_height])),
        }
    return POI(
        cent,
        orientation=("R", "A", "S"),
        zoom=(1.0, 1.0, 1.0),
        shape=(400.0, 400.0, 400.0),
        origin=(0.0, 0.0, 0.0),
        rotation=np.eye(3),
        level_one_info=V,
        level_two_info=Location,
    )


class TestNumericalCurvature(unittest.TestCase):
    """A circle of radius r has curvature exactly 1/r everywhere."""

    def test_circle_curvature_is_one_over_radius(self):
        for radius in (5.0, 17.3, 50.0):
            for step in (0.05, 0.2):
                with self.subTest(radius=radius, step=step):
                    t = np.arange(0.0, math.pi / 2, step)
                    pts = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
                    kappa = _numerical_curvature(pts)[3:-3]
                    np.testing.assert_allclose(kappa, 1.0 / radius, rtol=2e-2)

    def test_circle_in_a_tilted_plane_gives_the_same_curvature(self):
        # Proves the 3D cross-product branch agrees with the planar case.
        radius = 12.0
        t = np.arange(0.0, math.pi, 0.05)
        flat = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
        basis = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
        kappa = _numerical_curvature(flat @ basis.T)[3:-3]
        np.testing.assert_allclose(kappa, 1.0 / radius, rtol=2e-2)

    def test_straight_line_has_zero_curvature(self):
        pts = np.stack([np.arange(20.0), np.zeros(20), np.zeros(20)], axis=1)
        np.testing.assert_allclose(_numerical_curvature(pts), 0.0, atol=1e-12)

    def test_scaling_the_curve_scales_curvature_inversely(self):
        t = np.arange(0.0, 1.5, 0.05)
        pts = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        base = _numerical_curvature(pts)[3:-3]
        scaled = _numerical_curvature(pts * 4.0)[3:-3]
        np.testing.assert_allclose(scaled, base / 4.0, rtol=1e-6)

    def test_too_few_points(self):
        self.assertEqual(len(_numerical_curvature(np.zeros((2, 3)))), 2)


class TestFindApices(unittest.TestCase):
    def test_returns_local_maxima_sorted_by_arc_position(self):
        arc = np.arange(11.0)
        kappa = np.array([0.0, 0.5, 0.1, 0.9, 0.2, 0.3, 0.2, 0.7, 0.1, 0.4, 0.0])
        apices = _find_apices(arc, kappa, top_k=3)
        self.assertEqual([a["arc_mm"] for a in apices], [1.0, 3.0, 7.0])
        self.assertEqual([a["kappa_1_per_mm"] for a in apices], [0.5, 0.9, 0.7])

    def test_top_k_keeps_the_strongest_peaks(self):
        arc = np.arange(11.0)
        kappa = np.array([0.0, 0.5, 0.1, 0.9, 0.2, 0.3, 0.2, 0.7, 0.1, 0.4, 0.0])
        self.assertEqual([a["arc_mm"] for a in _find_apices(arc, kappa, top_k=1)], [3.0])

    def test_no_peak_and_too_short(self):
        self.assertEqual(_find_apices(np.arange(5.0), np.arange(5.0), top_k=3), [])
        self.assertEqual(_find_apices(np.arange(2.0), np.zeros(2), top_k=3), [])


class TestAngleBetweenTwoPoints(unittest.TestCase):
    def test_sagittal_tilt_is_recovered_exactly(self):
        for tilt in (10.0, 30.0, 45.0):
            with self.subTest(tilt=tilt):
                poi = spine_poi(sagittal_tilt={V.T5.value: tilt})
                for project_2d in (False, True):
                    angle = compute_angel_between_two_points_(
                        poi.copy(), V.T4, V.T5, "P", MoveTo.CENTER, MoveTo.CENTER, project_2D=project_2d
                    )
                    self.assertAlmostEqual(angle, tilt, delta=0.05)

    def test_untilted_pair_is_zero(self):
        poi = spine_poi()
        self.assertEqual(compute_angel_between_two_points_(poi.copy(), V.T4, V.T6, "P", MoveTo.CENTER, MoveTo.CENTER, False), 0.0)

    def test_coronal_tilt_is_measured_in_the_r_direction(self):
        poi = spine_poi(coronal_tilt={V.T5.value: 20.0})
        angle = compute_angel_between_two_points_(poi.copy(), V.T4, V.T5, "R", MoveTo.CENTER, MoveTo.CENTER, project_2D=False)
        self.assertAlmostEqual(angle, 20.0, delta=0.01)

    def test_opposite_directions_give_the_same_unsigned_angle(self):
        poi = spine_poi(sagittal_tilt={V.T5.value: 25.0})
        a = compute_angel_between_two_points_(poi.copy(), V.T4, V.T5, "P", MoveTo.CENTER, MoveTo.CENTER, False)
        b = compute_angel_between_two_points_(poi.copy(), V.T4, V.T5, "A", MoveTo.CENTER, MoveTo.CENTER, False)
        self.assertAlmostEqual(a, b, delta=1e-6)

    def test_argument_order_does_not_matter(self):
        poi = spine_poi(sagittal_tilt={V.T5.value: 18.0})
        a = compute_angel_between_two_points_(poi.copy(), V.T4, V.T5, "P", MoveTo.CENTER, MoveTo.CENTER, False)
        b = compute_angel_between_two_points_(poi.copy(), V.T5, V.T4, "P", MoveTo.CENTER, MoveTo.CENTER, False)
        self.assertAlmostEqual(a, b, delta=1e-6)

    def test_none_for_missing_vertebra_or_none_input(self):
        poi = spine_poi(ids=[V.T4.value, V.T5.value])
        self.assertIsNone(compute_angel_between_two_points_(poi.copy(), None, V.T5, "P"))
        self.assertIsNone(compute_angel_between_two_points_(poi.copy(), V.T4, None, "P"))
        self.assertIsNone(compute_angel_between_two_points_(poi.copy(), V.T4, V.L5, "P"))

    def test_unknown_direction_raises(self):
        poi = spine_poi(ids=[V.T4.value, V.T5.value])
        with self.assertRaises(NotImplementedError):
            compute_angel_between_two_points_(poi.copy(), V.T4, V.T5, "Q")  # type: ignore[arg-type]

    def test_same_vertebra_is_rejected(self):
        poi = spine_poi(ids=[V.T4.value, V.T5.value])
        with self.assertRaises(AssertionError):
            compute_angel_between_two_points_(poi.copy(), V.T4, V.T4, "P")

    def test_superior_direction_is_supported(self):
        poi = spine_poi(sagittal_tilt={V.T5.value: 15.0})
        angle = compute_angel_between_two_points_(poi.copy(), V.T4, V.T5, "S", MoveTo.CENTER, MoveTo.CENTER, project_2D=False)
        self.assertAlmostEqual(angle, 15.0, delta=0.01)


class TestMoveTo(unittest.TestCase):
    def test_center_resolves_to_the_body_centroid(self):
        poi = spine_poi()
        self.assertEqual(MoveTo.CENTER.get_location(V.T5, poi), (V.T5, 50))
        np.testing.assert_allclose(MoveTo.CENTER.get_point(V.T5, poi), poi[V.T5.value, 50])

    def test_top_and_bottom_resolve_to_the_endplate_medians(self):
        poi = spine_poi(half_height=5.0)
        top = MoveTo.TOP.get_point(V.T5, poi)
        bottom = MoveTo.BOTTOM.get_point(V.T5, poi)
        # Both are averaged with the neighbour, so they sit between the two bodies.
        self.assertGreater(top[2], bottom[2])

    def test_has_point(self):
        poi = spine_poi()
        self.assertTrue(MoveTo.CENTER.has_point(V.T5, poi))
        self.assertTrue(MoveTo.CENTER.has_point(V.T5.value, poi))
        self.assertFalse(MoveTo.CENTER.has_point(V.S1, poi))


class TestLordosisAndKyphosis(unittest.TestCase):
    def test_straight_spine_has_no_curvature(self):
        out = compute_lordosis_and_kyphosis(spine_poi())
        angle_keys = sorted(k for k in out if not k.endswith("_apex"))
        self.assertEqual(angle_keys, ["cervical_lordosis", "lumbar_lordosis", "thoracic_kyphosis"])
        for key in angle_keys:
            with self.subTest(key=key):
                self.assertEqual(out[key], 0.0)

    def test_tilting_c2_only_moves_the_cervical_value(self):
        out = compute_lordosis_and_kyphosis(spine_poi(sagittal_tilt={V.C2.value: 20.0}))
        # C2 is an end vertebra, so its normal is averaged with its single neighbour -> half.
        self.assertAlmostEqual(out["cervical_lordosis"], 10.0, delta=0.01)
        self.assertEqual(out["thoracic_kyphosis"], 0.0)
        self.assertEqual(out["lumbar_lordosis"], 0.0)

    def test_requires_the_posterior_direction(self):
        poi = spine_poi()
        poi.remove_centroid_(*[(v, Location.Vertebra_Direction_Posterior.value) for v in SPINE_IDS])
        with self.assertRaises(AssertionError):
            compute_lordosis_and_kyphosis(poi)

    def test_input_is_not_mutated(self):
        poi = spine_poi(sagittal_tilt={V.C2.value: 20.0})
        before = tuple(poi[V.C2.value, 50])
        compute_lordosis_and_kyphosis(poi)
        self.assertEqual(tuple(poi[V.C2.value, 50]), before)


class TestMaxCobbAngle(unittest.TestCase):
    def test_straight_spine_has_no_cobb_angle(self):
        self.assertEqual(compute_max_cobb_angle(spine_poi()), (0, None, None, None))

    def test_opposed_coronal_tilts_sum_into_the_cobb_angle(self):
        poi = spine_poi(coronal_tilt={V.T5.value: 15.0, V.T11.value: -15.0})
        angle, from_vert, to_vert, apex = compute_max_cobb_angle(poi.copy(), vert_id1_mv=MoveTo.CENTER, vert_id2_mv=MoveTo.CENTER)
        self.assertAlmostEqual(angle, 30.0, delta=0.05)
        self.assertEqual((from_vert, to_vert), (V.T5.value, V.T11.value))
        self.assertIsNotNone(apex)
        self.assertLessEqual(from_vert, apex)
        self.assertLessEqual(apex, to_vert)

    def test_default_endplate_moves_average_with_the_neighbour(self):
        poi = spine_poi(coronal_tilt={V.T5.value: 15.0, V.T11.value: -15.0})
        angle, _, _, _ = compute_max_cobb_angle(poi.copy())
        self.assertAlmostEqual(angle, 15.0, delta=0.05)

    def test_restricting_the_vertebra_list_restricts_the_search(self):
        poi = spine_poi(coronal_tilt={V.T5.value: 15.0, V.T11.value: -15.0})
        angle, _, _, _ = compute_max_cobb_angle(poi.copy(), vertebrae_list=[V.L1, V.L2, V.L3])
        self.assertEqual(angle, 0)

    def test_multi_returns_the_segments_above_the_threshold(self):
        poi = spine_poi(coronal_tilt={V.T5.value: 15.0, V.T11.value: -15.0})
        out = compute_max_cobb_angle_multi(poi.copy(), vert_id1_mv=MoveTo.CENTER, vert_id2_mv=MoveTo.CENTER)
        self.assertGreaterEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 30.0, delta=0.05)

    def test_multi_is_empty_for_a_straight_spine(self):
        self.assertEqual(compute_max_cobb_angle_multi(spine_poi()), [])


class TestSagittalAndCoronalBalance(unittest.TestCase):
    def test_sva_is_the_signed_anterior_offset_of_c7(self):
        poi = spine_poi(ids=[*SPINE_IDS, V.S1.value], anterior_shift={V.C7.value: 25.0})
        out = compute_sva(poi.copy())
        self.assertEqual(out["top_vertebra"], "C7")
        self.assertEqual(out["base_vertebra"], "S1")
        self.assertAlmostEqual(out["sva_mm"], 25.0, delta=0.6)

    def test_sva_is_zero_for_a_plumb_spine(self):
        poi = spine_poi(ids=[*SPINE_IDS, V.S1.value])
        self.assertAlmostEqual(compute_sva(poi.copy())["sva_mm"], 0.0, delta=0.6)

    def test_sva_falls_back_from_s1_to_l5(self):
        out = compute_sva(spine_poi())  # no S1 in this spine
        self.assertEqual(out["base_vertebra"], "L5")
        self.assertIn("L5", out["base_landmark"])

    def test_sva_reports_a_missing_top_vertebra(self):
        poi = spine_poi(ids=[V.L1.value, V.L2.value, V.L3.value])
        out = compute_sva(poi.copy())
        self.assertIsNone(out["sva_mm"])
        self.assertIn("C7", out["error"])

    def test_sva_reports_a_missing_base_vertebra(self):
        poi = spine_poi(ids=[V.C6.value, V.C7.value, V.T1.value])
        out = compute_sva(poi.copy())
        self.assertIsNone(out["sva_mm"])
        self.assertIn("base vertebra", out["error"])

    def test_coronal_balance_tracks_the_lateral_offset(self):
        poi = spine_poi(ids=[*SPINE_IDS, V.S1.value], coronal_amp=0.0)
        self.assertAlmostEqual(compute_coronal_balance(poi.copy())["coronal_balance_mm"], 0.0, delta=0.6)

    def test_coronal_balance_reports_a_missing_top_vertebra(self):
        out = compute_coronal_balance(spine_poi(ids=[V.L1.value, V.L2.value, V.L3.value]))
        self.assertIsNone(out["coronal_balance_mm"])
        self.assertIn("C7", out["error"])


class TestSegmentalEndplateAnglesAndAxialRotation(unittest.TestCase):
    def test_tilting_one_vertebra_shows_up_in_both_neighbouring_segments(self):
        out = compute_segmental_endplate_angles(spine_poi(sagittal_tilt={V.T5.value: 10.0}))
        non_zero = {k: v for k, v in out.items() if v != 0.0}
        self.assertEqual(non_zero, {"T4-T5": 10.0, "T5-T6": -10.0})

    def test_straight_spine_is_all_zero(self):
        out = compute_segmental_endplate_angles(spine_poi())
        self.assertTrue(out)
        self.assertEqual(set(out.values()), {0.0})

    def test_axial_rotation_is_zero_without_rotation(self):
        out = compute_axial_rotation(spine_poi())
        self.assertTrue(out)
        self.assertEqual(set(out.values()), {0.0})

    def test_axial_rotation_recovers_a_transverse_rotation(self):
        out = compute_axial_rotation(spine_poi(axial_tilt={V.T5.value: 12.0}))
        self.assertAlmostEqual(abs(out["T5"]), 12.0, delta=0.1)


class TestWedgeMetrics(unittest.TestCase):
    def test_analytic_values(self):
        out = compute_wedge_metrics({7: {"anterior_height_x1": 12.0, "posterior_height_x2": 10.0, "width_sagittal_x6": 20.0}})
        self.assertAlmostEqual(out[7]["sagittal_wedge_deg"], math.degrees(math.atan2(2.0, 20.0)), places=3)
        self.assertAlmostEqual(out[7]["sagittal_wedge_index"], 2.0 / 11.0, places=4)
        self.assertTrue(math.isnan(out[7]["coronal_wedge_deg"]))

    def test_coronal_values(self):
        out = compute_wedge_metrics({1: {"right_height_x3": 11.0, "left_height_x4": 9.0, "width_lateral_x5": 30.0}})
        self.assertAlmostEqual(out[1]["coronal_wedge_deg"], math.degrees(math.atan2(2.0, 30.0)), places=3)
        self.assertAlmostEqual(out[1]["coronal_wedge_index"], 2.0 / 10.0, places=4)

    def test_symmetric_structure_has_zero_wedge(self):
        out = compute_wedge_metrics({1: {"anterior_height_x1": 10.0, "posterior_height_x2": 10.0, "width_sagittal_x6": 20.0}})
        self.assertEqual(out[1]["sagittal_wedge_deg"], 0.0)
        self.assertEqual(out[1]["sagittal_wedge_index"], 0.0)

    def test_missing_and_invalid_inputs_become_nan(self):
        out = compute_wedge_metrics(
            {
                1: {"anterior_height_x1": None, "posterior_height_x2": 10.0, "width_sagittal_x6": 20.0},
                2: {"anterior_height_x1": 12.0, "posterior_height_x2": 10.0, "width_sagittal_x6": 0.0},
                3: {"anterior_height_x1": float("nan"), "posterior_height_x2": 10.0, "width_sagittal_x6": 20.0},
                4: "not a dict",
            }
        )
        self.assertNotIn(4, out)
        for label in (1, 2, 3):
            with self.subTest(label=label):
                self.assertTrue(math.isnan(out[label]["sagittal_wedge_deg"]))

    def test_input_is_not_modified(self):
        geometry = {1: {"anterior_height_x1": 12.0, "posterior_height_x2": 10.0, "width_sagittal_x6": 20.0}}
        snapshot = json.dumps(geometry, sort_keys=True)
        compute_wedge_metrics(geometry)
        self.assertEqual(json.dumps(geometry, sort_keys=True), snapshot)


class TestCurvatureProfileAndMultiCobb(unittest.TestCase):
    def test_straight_spine_profile(self):
        out = compute_curvature_profile(spine_poi())
        self.assertAlmostEqual(out["tortuosity"], 1.0, places=3)
        self.assertAlmostEqual(out["curvature_max_1_per_mm"], 0.0, places=6)
        self.assertAlmostEqual(out["arc_length_mm"], out["chord_length_mm"], places=3)

    def test_arc_profile_recovers_one_over_the_radius(self):
        radius = 200.0
        out = compute_curvature_profile(spine_poi(sagittal_radius=radius))
        self.assertAlmostEqual(out["curvature_mean_1_per_mm"], 1.0 / radius, delta=1e-4)
        self.assertGreater(out["tortuosity"], 1.0)
        self.assertGreater(out["arc_length_mm"], out["chord_length_mm"])

    def test_profile_needs_four_corpus_points(self):
        out = compute_curvature_profile(spine_poi(ids=[V.T4.value, V.T5.value]))
        self.assertIn("too few", out["error"])

    def test_multi_cobb_is_zero_for_a_straight_spine(self):
        out = compute_multi_cobb(spine_poi())
        self.assertEqual(out["max_cobb_deg"], 0.0)

    def test_multi_cobb_grows_with_the_lateral_deviation(self):
        small = compute_multi_cobb(spine_poi(coronal_amp=5.0))["max_cobb_deg"]
        large = compute_multi_cobb(spine_poi(coronal_amp=30.0))["max_cobb_deg"]
        self.assertGreater(large, small)
        self.assertGreater(small, 0.0)
        for curve in compute_multi_cobb(spine_poi(coronal_amp=30.0))["curves"]:
            with self.subTest(curve=curve):
                self.assertIn(curve["handedness"], ("left", "right"))
                self.assertGreaterEqual(curve["arc_end_mm"], curve["arc_start_mm"])

    def test_multi_cobb_needs_four_corpus_points(self):
        self.assertIn("too few", compute_multi_cobb(spine_poi(ids=[V.T4.value, V.T5.value]))["error"])


class TestAngleHelpers(unittest.TestCase):
    def test_cosine_distance(self):
        self.assertAlmostEqual(cosine_distance(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])), 1.0, places=6)
        self.assertAlmostEqual(cosine_distance(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), 0.0, places=6)
        self.assertAlmostEqual(cosine_distance(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])), -1.0, places=6)

    def test_get_to_space_is_an_invertible_orthonormal_basis(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        to_space, from_space = get_to_space(a, b, c)
        np.testing.assert_allclose(from_space @ to_space, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(to_space @ a, [1, 0, 0], atol=1e-9)

    def test_last_lumbar_and_thoracic(self):
        poi = spine_poi()
        self.assertEqual(angles_mod._get_last_lumbar(poi), V.L5)
        self.assertEqual(angles_mod._get_last_thoracic(poi), V.T12)

    def test_last_lumbar_is_none_without_lumbar_vertebrae(self):
        self.assertIsNone(angles_mod._get_last_lumbar(spine_poi(ids=[V.T4.value, V.T5.value])))

    def test_last_present_picks_the_first_available_candidate(self):
        poi = spine_poi()
        self.assertEqual(curv_mod._last_present(poi, [V.S1, V.L5, V.L4]), V.L5)
        self.assertIsNone(curv_mod._last_present(poi, [V.S1]))

    def test_get_and_corpus_helpers(self):
        poi = curv_mod._prep(spine_poi())
        self.assertIsNotNone(curv_mod._corpus(poi, V.T5))
        self.assertIsNotNone(curv_mod._corpus(poi, V.T5.value))
        self.assertIsNone(curv_mod._get(poi, V.S1, Location.Vertebra_Corpus))

    def test_endplate_falls_back_to_corpus_plus_direction(self):
        poi = curv_mod._prep(spine_poi(half_height=5.0))
        sup = curv_mod._endplate(poi, V.T5, superior=True)
        inf = curv_mod._endplate(poi, V.T5, superior=False)
        corpus = curv_mod._corpus(poi, V.T5)
        self.assertAlmostEqual(float(np.linalg.norm(sup - corpus)), 5.0, delta=0.6)
        self.assertAlmostEqual(float(np.linalg.norm(inf - corpus)), 5.0, delta=0.6)


class TestPelvicParameters(unittest.TestCase):
    S_POST = np.array([0.0, -8.66, 5.0])
    S_ANT = np.array([0.0, 8.66, -5.0])
    FH_R = np.array([80.0, 50.0, -100.0])
    FH_L = np.array([-80.0, 50.0, -100.0])

    def test_pi_equals_pt_plus_ss(self):
        out = _pelvic_from_endplate_pair(self.S_POST, self.S_ANT, self.FH_R, self.FH_L)
        self.assertAlmostEqual(out["pi_deg"], out["pt_deg"] + out["ss_deg"], places=9)

    def test_pelvic_incidence_is_invariant_under_sagittal_rotation(self):
        base = _pelvic_from_endplate_pair(self.S_POST, self.S_ANT, self.FH_R, self.FH_L)
        theta = 12.0
        r = rot("x", theta)
        turned = _pelvic_from_endplate_pair(r @ self.S_POST, r @ self.S_ANT, r @ self.FH_R, r @ self.FH_L)
        # PI is an anatomical constant; PT and SS trade the rotation between them.
        self.assertAlmostEqual(turned["pi_deg"], base["pi_deg"], places=3)
        self.assertAlmostEqual(turned["pt_deg"], base["pt_deg"] + theta, places=3)
        self.assertAlmostEqual(turned["ss_deg"], base["ss_deg"] - theta, places=3)

    def test_all_angles_are_scale_invariant(self):
        base = _pelvic_from_endplate_pair(self.S_POST, self.S_ANT, self.FH_R, self.FH_L)
        scaled = _pelvic_from_endplate_pair(self.S_POST * 3, self.S_ANT * 3, self.FH_R * 3, self.FH_L * 3)
        for key in ("pi_deg", "pt_deg", "ss_deg"):
            with self.subTest(key=key):
                self.assertAlmostEqual(scaled[key], base[key], places=6)

    def test_hip_centre_is_the_midpoint_of_the_femoral_heads(self):
        out = _pelvic_from_endplate_pair(self.S_POST, self.S_ANT, self.FH_R, self.FH_L)
        np.testing.assert_allclose(out["hip_center_mm"], (self.FH_R + self.FH_L) / 2)
        np.testing.assert_allclose(out["s1_endplate_center_mm"], (self.S_POST + self.S_ANT) / 2)

    def test_degenerate_endplate(self):
        out = _pelvic_from_endplate_pair(self.S_POST, self.S_POST, self.FH_R, self.FH_L)
        self.assertIsNone(out["pi_deg"])
        self.assertIn("degenerate", out["error"])

    def test_degenerate_hip_axis(self):
        zero = np.zeros(3)
        out = _pelvic_from_endplate_pair(self.S_POST, self.S_ANT, zero, zero)
        self.assertIsNone(out["pi_deg"])
        self.assertIn("coincides", out["error"])

    def test_sag_projects_onto_the_anterior_superior_plane(self):
        # Drops the left-right component entirely and returns a 2-vector.
        np.testing.assert_allclose(_sag(np.array([5.0, 2.0, 3.0])), [2.0, 3.0])

    def _write_poi_json(self, folder: Path, body: dict) -> Path:
        path = folder / "sub-x_seg-fullbody_poi.json"
        path.write_text(json.dumps([{"meta": 1}, body]), encoding="utf-8")
        return path

    def _full_body(self) -> dict:
        return {
            "sacrum": {
                "1": self.S_POST.tolist(),
                "19": self.S_ANT.tolist(),
                "27": [-10.0, 6.0, 4.0],
                "28": [10.0, 6.0, 4.0],
            },
            "femur_right": {"11": self.FH_R.tolist()},
            "femur_left": {"11": self.FH_L.tolist()},
        }

    def test_compute_pelvic_parameters_both_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_poi_json(Path(tmp), self._full_body())
            out = compute_pelvic_parameters(path, lumbar_lordosis_deg=40.0)
            self.assertNotIn("error", out)
            for variant in ("poi_ap", "poi_ala"):
                with self.subTest(variant=variant):
                    pi = out[variant]["pi_deg"]
                    self.assertIsNotNone(pi)
                    self.assertAlmostEqual(out[variant]["pi_ll_mismatch_deg"], round(pi - 40.0, 3), places=3)

    def test_mismatch_is_none_without_lumbar_lordosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = compute_pelvic_parameters(self._write_poi_json(Path(tmp), self._full_body()))
            self.assertIsNone(out["poi_ap"]["pi_ll_mismatch_deg"])

    def test_missing_ala_landmarks_only_affects_the_ala_variant(self):
        body = self._full_body()
        del body["sacrum"]["27"]
        with tempfile.TemporaryDirectory() as tmp:
            out = compute_pelvic_parameters(self._write_poi_json(Path(tmp), body))
            self.assertIsNotNone(out["poi_ap"]["pi_deg"])
            self.assertIn("ala", out["poi_ala"]["error"])

    def test_error_paths(self):
        self.assertIn("no fullbody POI json path", compute_pelvic_parameters(None)["error"])
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            self.assertIn("does not exist", compute_pelvic_parameters(missing)["error"])

            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            self.assertIn("failed to load", compute_pelvic_parameters(broken)["error"])

            wrong_shape = Path(tmp) / "shape.json"
            wrong_shape.write_text(json.dumps({"body": {}}), encoding="utf-8")
            self.assertIn("unexpected", compute_pelvic_parameters(wrong_shape)["error"])

            body = self._full_body()
            del body["sacrum"]["19"]
            self.assertIn("missing required landmarks", compute_pelvic_parameters(self._write_poi_json(Path(tmp), body))["error"])

    def test_resolve_fullbody_poi_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(resolve_fullbody_poi_path(root, "100000"))


class TestPlottingEntryPoints(unittest.TestCase):
    """The plot_* wrappers must return the same numbers as the bare computation.

    `plot_cobb_and_lordosis_and_kyphosis` is not covered here: it resamples the POI
    onto the image in a way that is singular for a synthetic spine on the shipped
    single-vertebra sample, and inventing a fixture that satisfies it would not be
    testing anything real.
    """

    @classmethod
    def setUpClass(cls):
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use("Agg")
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        cls.ct, cls.subreg, cls.vert, cls.label = get_test_ct()
        # The POI has to live on the image's own grid - the renderers resample between
        # the two - and its spine must run along the image's inferior axis, which is
        # what the snapshot code interpolates along. So: build it in RAS, where the
        # builder stacks along S, then reorient onto the sample CT's PIR grid.
        ct_ras = cls.ct.reorient(("R", "A", "S"))
        ids = [v.value for v in (V.T8, V.T9, V.T10, V.T11, V.T12, V.L1, V.L2, V.L3)]
        base = spine_poi(ids=ids, sagittal_tilt={V.T11.value: 15.0}, spacing=6.0)
        centre = np.array(ct_ras.shape, dtype=float) / 2.0
        offset = centre - np.array([0.0, 0.0, 6.0 * len(ids) / 2.0])
        points = {
            vert: {sub: tuple(float(c + o) for c, o in zip(coord, offset)) for sub, coord in inner.items()}
            for vert, inner in base.centroids.items_2d()
        }
        cls.poi = POI(
            points,
            orientation=ct_ras.orientation,
            zoom=ct_ras.zoom,
            shape=ct_ras.shape,
            origin=ct_ras.origin,
            rotation=ct_ras.rotation,
            level_one_info=V,
            level_two_info=Location,
        ).reorient_(cls.ct.orientation)

    def tearDown(self):
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.close("all")

    def test_lordosis_plot_returns_the_same_numbers_as_the_bare_computation(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lordosis.png"
            values, frame = angles_mod.plot_compute_lordosis_and_kyphosis(out, self.poi.copy(), self.ct, self.vert)
            self.assertIsInstance(frame, Snapshot_Frame)
            self.assertEqual(values, compute_lordosis_and_kyphosis(self.poi.copy()))
            self.assertTrue(out.exists())

    def test_lordosis_plot_without_a_path_renders_nothing(self):
        values, _ = angles_mod.plot_compute_lordosis_and_kyphosis(None, self.poi.copy(), self.ct, self.vert)
        angle_keys = sorted(k for k in values if not k.endswith("_apex"))
        self.assertEqual(angle_keys, ["cervical_lordosis", "lumbar_lordosis", "thoracic_kyphosis"])

    def test_cobb_plot_returns_the_segments_and_a_frame(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cobb.png"
            segments, frame = angles_mod.plot_cobb_angle(out, self.poi.copy(), self.ct, self.vert)
            self.assertIsInstance(frame, Snapshot_Frame)
            self.assertIsInstance(segments, list)
            self.assertTrue(out.exists())
            for angle, *_ in segments:
                with self.subTest(angle=angle):
                    self.assertGreaterEqual(angle, 0.0)
                    self.assertLess(angle, 180.0)

    def test_plots_work_without_a_segmentation_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "no_seg.png"
            angles_mod.plot_compute_lordosis_and_kyphosis(out, self.poi.copy(), self.ct, None)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
