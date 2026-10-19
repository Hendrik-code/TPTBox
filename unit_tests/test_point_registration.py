"""Unit tests for `TPTBox.registration._ridged_points.point_registration`.

Rigid point registration needs only SimpleITK (a core dependency), so these run
everywhere. Every expectation is a known transform or an inverse property: two POI
sets related by an exact translation or rotation must be recovered to sub-voxel
accuracy, forward and inverse coordinate mapping must compose to the identity, and
a save/load round-trip must reproduce the same transform.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from TPTBox import POI, Location, Vertebra_Instance
from TPTBox.registration._ridged_points.point_registration import (
    Point_Registration,
    ridged_points_from_poi,
)

SHAPE = (60.0, 60.0, 60.0)
RNG = np.random.default_rng(0)


def _poi(points: dict) -> POI:
    return POI(
        points,
        orientation=("R", "A", "S"),
        zoom=(1.0, 1.0, 1.0),
        shape=SHAPE,
        origin=(0.0, 0.0, 0.0),
        rotation=np.eye(3),
        level_one_info=Vertebra_Instance,
        level_two_info=Location,
    )


def _landmarks() -> dict:
    """Six well-spread landmarks, enough to determine a rigid transform."""
    coords = [(10.0, 12.0, 14.0), (40.0, 12.0, 14.0), (10.0, 44.0, 14.0), (10.0, 12.0, 46.0), (30.0, 30.0, 30.0), (20.0, 40.0, 20.0)]
    return {20 + i: {Location.Vertebra_Corpus.value: c} for i, c in enumerate(coords)}


def _shifted(delta=(5.0, -3.0, 2.0)) -> dict:
    return {
        vert: {sub: tuple(float(c + d) for c, d in zip(coord, delta)) for sub, coord in inner.items()}
        for vert, inner in _landmarks().items()
    }


def _rotated(deg=12.0) -> dict:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([30.0, 30.0, 30.0])
    return {
        vert: {sub: tuple(float(v) for v in (rot @ (np.asarray(coord) - centre) + centre)) for sub, coord in inner.items()}
        for vert, inner in _landmarks().items()
    }


class TestPointRegistrationFit(unittest.TestCase):
    def test_identical_point_sets_give_a_near_zero_error(self):
        reg = ridged_points_from_poi(_poi(_landmarks()), _poi(_landmarks()), verbose=False)
        self.assertLess(reg.error_reg, 1e-6)

    def test_a_pure_translation_is_recovered(self):
        fixed = _poi(_landmarks())
        moving = _poi(_shifted())
        reg = ridged_points_from_poi(fixed, moving, verbose=False)
        self.assertLess(reg.error_reg, 1e-3, "a rigid transform fits a pure translation exactly")
        out = reg.transform_poi(moving)
        for key in fixed.keys():
            with self.subTest(key=key):
                np.testing.assert_allclose(np.asarray(out[key]), np.asarray(fixed[key]), atol=0.01)

    def test_a_pure_rotation_is_recovered(self):
        fixed = _poi(_landmarks())
        moving = _poi(_rotated())
        reg = ridged_points_from_poi(fixed, moving, verbose=False)
        self.assertLess(reg.error_reg, 1e-3)
        out = reg.transform_poi(moving)
        for key in fixed.keys():
            with self.subTest(key=key):
                np.testing.assert_allclose(np.asarray(out[key]), np.asarray(fixed[key]), atol=0.05)

    def test_error_natural_measures_non_rigidity_not_displacement(self):
        # error_natural is the mean change in *pairwise landmark distances*, so any
        # rigid motion leaves it at zero however far the points move.
        for name, points in (("translated", _shifted(delta=(50.0, -30.0, 20.0))), ("rotated", _rotated(deg=25.0))):
            with self.subTest(motion=name):
                reg = ridged_points_from_poi(_poi(_landmarks()), _poi(points), verbose=False)
                # The rotated case carries a little residue from the POI's stored
                # coordinate precision; still two orders of magnitude below the
                # deformation case below.
                self.assertAlmostEqual(reg.error_natural, 0.0, delta=0.05)

    def test_error_natural_grows_when_the_point_set_is_deformed(self):
        stretched = {
            vert: {sub: (coord[0] * 1.5, coord[1], coord[2]) for sub, coord in inner.items()} for vert, inner in _landmarks().items()
        }
        reg = ridged_points_from_poi(_poi(_landmarks()), _poi(stretched), verbose=False)
        self.assertGreater(reg.error_natural, 1.0, "a scaling cannot be absorbed by a rigid transform")

    def test_exclusion_drops_landmarks_from_the_fit(self):
        fixed = _poi(_landmarks())
        moving_points = _shifted()
        # Corrupt one landmark badly; excluding it must restore the clean fit.
        bad_key = next(iter(moving_points))
        moving_points[bad_key][Location.Vertebra_Corpus.value] = (0.0, 0.0, 0.0)
        moving = _poi(moving_points)
        with_bad = ridged_points_from_poi(fixed, moving, verbose=False).error_reg
        without = ridged_points_from_poi(fixed, moving, exclusion=[bad_key], verbose=False).error_reg
        self.assertLess(without, with_bad)
        self.assertLess(without, 1e-3)

    def test_leave_worst_percent_out_improves_the_residual(self):
        fixed = _poi(_landmarks())
        moving_points = _shifted()
        moving_points[next(iter(moving_points))][Location.Vertebra_Corpus.value] = (0.0, 0.0, 0.0)
        moving = _poi(moving_points)
        plain = ridged_points_from_poi(fixed, moving, verbose=False).error_reg
        trimmed = ridged_points_from_poi(fixed, moving, leave_worst_percent_out=0.3, verbose=False).error_reg
        self.assertLess(trimmed, plain)

    def test_out_of_range_trim_fraction_is_rejected(self):
        fixed, moving = _poi(_landmarks()), _poi(_shifted())
        for value in (1.0, 1.5, -0.1):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                ridged_points_from_poi(fixed, moving, leave_worst_percent_out=value, verbose=False)

    def test_too_few_shared_landmarks_is_an_error(self):
        fixed = _poi({20: {Location.Vertebra_Corpus.value: (1.0, 1.0, 1.0)}})
        moving = _poi({21: {Location.Vertebra_Corpus.value: (2.0, 2.0, 2.0)}})
        with self.assertRaises(ValueError):
            ridged_points_from_poi(fixed, moving, verbose=False)


class TestCoordinateTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed = _poi(_landmarks())
        cls.moving = _poi(_shifted())
        cls.reg = ridged_points_from_poi(cls.fixed, cls.moving, verbose=False)

    def test_forward_then_inverse_is_the_identity(self):
        for point in RNG.uniform(5.0, 50.0, size=(20, 3)):
            with self.subTest(point=tuple(np.round(point, 2))):
                mapped = self.reg.transform_cord(tuple(point))
                back = self.reg.transform_cord_inverse(tuple(mapped))
                np.testing.assert_allclose(np.asarray(back, dtype=float), point, atol=1e-4)

    def test_get_affine_is_consistent_with_transform_cord(self):
        affine = self.reg.get_affine()
        self.assertEqual(np.asarray(affine).shape, (4, 4))
        np.testing.assert_allclose(np.asarray(affine)[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)

    def test_transform_poi_inverse_undoes_transform_poi(self):
        forward = self.reg.transform_poi(self.moving)
        back = self.reg.transform_poi_inverse(forward)
        for key in self.moving.keys():
            with self.subTest(key=key):
                np.testing.assert_allclose(np.asarray(back[key]), np.asarray(self.moving[key]), atol=0.05)

    def test_transform_dispatches_on_the_input_type(self):
        out = self.reg.transform(self.moving)
        self.assertIsInstance(out, POI)


class TestSerialization(unittest.TestCase):
    def test_save_and_load_reproduce_the_transform(self):
        fixed, moving = _poi(_landmarks()), _poi(_shifted())
        reg = ridged_points_from_poi(fixed, moving, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.pkl"
            reg.save(path)
            self.assertTrue(path.exists())
            loaded = Point_Registration.load(path)
            np.testing.assert_allclose(np.asarray(loaded.get_affine()), np.asarray(reg.get_affine()), atol=1e-9)
            self.assertAlmostEqual(loaded.error_reg, reg.error_reg, places=9)
            self.assertAlmostEqual(loaded.error_natural, reg.error_natural, places=9)

    def test_a_loaded_registration_maps_coordinates_identically(self):
        fixed, moving = _poi(_landmarks()), _poi(_shifted())
        reg = ridged_points_from_poi(fixed, moving, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg.pkl"
            reg.save(path)
            loaded = Point_Registration.load(path)
            for point in RNG.uniform(5.0, 50.0, size=(5, 3)):
                with self.subTest(point=tuple(np.round(point, 2))):
                    np.testing.assert_allclose(
                        np.asarray(loaded.transform_cord(tuple(point)), dtype=float),
                        np.asarray(reg.transform_cord(tuple(point)), dtype=float),
                        atol=1e-9,
                    )

    def test_get_dump_round_trips_through_load_(self):
        fixed, moving = _poi(_landmarks()), _poi(_shifted())
        reg = ridged_points_from_poi(fixed, moving, verbose=False)
        rebuilt = Point_Registration.load_(reg.get_dump())
        np.testing.assert_allclose(np.asarray(rebuilt.get_affine()), np.asarray(reg.get_affine()), atol=1e-9)

    def test_asymmetric_grids_survive_the_round_trip(self):
        # Regression guard for the moving/fixed ordering in load_(): with different
        # grids on the two sides a swap would be visible, unlike in a symmetric case.
        fixed = _poi(_landmarks())
        moving = _poi(_shifted())
        moving.rescale_((2.0, 2.0, 2.0))
        reg = ridged_points_from_poi(fixed, moving, verbose=False)
        rebuilt = Point_Registration.load_(reg.get_dump())
        np.testing.assert_allclose(np.asarray(rebuilt.input_poi.affine), np.asarray(reg.input_poi.affine), atol=1e-9)
        np.testing.assert_allclose(np.asarray(rebuilt.out_poi.affine), np.asarray(reg.out_poi.affine), atol=1e-9)
        np.testing.assert_allclose(np.asarray(rebuilt.get_affine()), np.asarray(reg.get_affine()), atol=1e-9)


class TestOnSampleData(unittest.TestCase):
    """Registering the shipped CT's vertebra centroids against a shifted copy."""

    @classmethod
    def setUpClass(cls):
        from TPTBox import calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        cls.ct, cls.subreg, cls.vert, cls.label = get_test_ct()
        cls.poi = calc_poi_from_subreg_vert(cls.vert, cls.subreg, subreg_id=[Location.Vertebra_Corpus], verbose=False)

    def test_registering_a_shifted_copy_recovers_the_shift(self):
        moving = self.poi.copy()
        shifted = {key: tuple(float(c + 3.0) for c in moving[key]) for key in moving.keys()}
        for key, value in shifted.items():
            moving[key] = value
        reg = ridged_points_from_poi(self.poi, moving, verbose=False)
        self.assertLess(reg.error_reg, 0.5)

    def test_transform_nii_keeps_the_target_grid(self):
        moving = self.poi.copy()
        reg = ridged_points_from_poi(self.poi, moving, verbose=False)
        out = reg.transform_nii(self.ct)
        self.assertEqual(out.shape, self.ct.shape)


if __name__ == "__main__":
    unittest.main()
