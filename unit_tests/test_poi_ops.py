"""Extended unit tests for POI: set operations, accessors, coordinate transforms."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

import random  # noqa: E402

import numpy as np  # noqa: E402

from TPTBox.core.poi import POI  # noqa: E402
from TPTBox.tests.test_utils import get_poi, repeats  # noqa: E402


def _simple_poi(*points: tuple[int, int, float, float, float]) -> POI:
    """Build a POI with identity affine and integer coordinates.

    Points are given as (region, subregion, x, y, z) tuples.
    """
    d: dict[int, dict[int, tuple[float, float, float]]] = {}
    for region, subregion, x, y, z in points:
        d.setdefault(region, {})[subregion] = (x, y, z)
    return POI(d, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=(100, 100, 100), origin=(0, 0, 0), rotation=np.eye(3))


class Test_POI_Accessors(unittest.TestCase):
    def test_len(self):
        p = _simple_poi((1, 50, 1, 2, 3), (2, 50, 4, 5, 6), (3, 60, 7, 8, 9))
        self.assertEqual(len(p), 3)

    def test_len_empty(self):
        p = POI({}, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=(10, 10, 10), origin=(0, 0, 0), rotation=np.eye(3))
        self.assertEqual(len(p), 0)

    def test_contains_present(self):
        p = _simple_poi((1, 50, 1, 2, 3))
        self.assertIn((1, 50), p)

    def test_contains_absent(self):
        p = _simple_poi((1, 50, 1, 2, 3))
        self.assertNotIn((2, 50), p)
        self.assertNotIn((1, 99), p)

    def test_getitem(self):
        p = _simple_poi((5, 10, 1.5, 2.5, 3.5))
        coord = p[5, 10]
        self.assertAlmostEqual(coord[0], 1.5)
        self.assertAlmostEqual(coord[1], 2.5)
        self.assertAlmostEqual(coord[2], 3.5)

    def test_setitem_overwrites(self):
        p = _simple_poi((1, 1, 0, 0, 0))
        p[1, 1] = (9, 8, 7)
        coord = p[1, 1]
        self.assertAlmostEqual(coord[0], 9)
        self.assertAlmostEqual(coord[1], 8)
        self.assertAlmostEqual(coord[2], 7)

    def test_setitem_new_entry(self):
        p = _simple_poi((1, 1, 0, 0, 0))
        p[2, 5] = (3, 3, 3)
        self.assertIn((2, 5), p)
        self.assertEqual(len(p), 2)

    def test_keys_region(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 50, 1, 1, 1), (1, 60, 2, 2, 2))
        regions = p.keys_region()
        self.assertIn(1, regions)
        self.assertIn(2, regions)
        self.assertNotIn(3, regions)

    def test_keys_subregion(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 60, 1, 1, 1), (3, 50, 2, 2, 2))
        subregs = p.keys_subregion()
        self.assertIn(50, subregs)
        self.assertIn(60, subregs)
        self.assertNotIn(70, subregs)

    def test_items_2d_structure(self):
        p = _simple_poi((1, 50, 0, 0, 0), (1, 60, 1, 1, 1), (2, 50, 2, 2, 2))
        items = dict(p.items_2D())
        self.assertIn(1, items)
        self.assertIn(2, items)
        self.assertIn(50, items[1])
        self.assertIn(60, items[1])

    def test_items_flatten(self):
        # items_flatten yields (packed_int, coordinate) pairs where
        # packed_int = subregion * LABEL_MAX + region (legacy encoding).
        p = _simple_poi((1, 50, 7, 8, 9), (2, 60, 1, 2, 3))
        flat = list(p.items_flatten())
        self.assertEqual(len(flat), 2)
        # Coordinates should still be recoverable via the packed key.
        coords_by_packed = dict(flat)
        # Verify one known coord is reachable via the packed index.
        for packed, coord in coords_by_packed.items():
            retrieved = p[packed]
            np.testing.assert_allclose(retrieved, coord, atol=1e-6)


class Test_POI_SetOps(unittest.TestCase):
    def test_join_left_no_overwrite(self):
        p1 = _simple_poi((1, 50, 1, 1, 1))
        p2 = _simple_poi((1, 50, 9, 9, 9), (2, 50, 2, 2, 2))
        result = p1.join_left(p2)
        # (1,50) should remain from p1, not overwritten by p2
        coord = result[1, 50]
        self.assertAlmostEqual(coord[0], 1)
        # (2,50) should be added from p2
        self.assertIn((2, 50), result)
        self.assertEqual(len(result), 2)

    def test_join_right_overwrites(self):
        p1 = _simple_poi((1, 50, 1, 1, 1))
        p2 = _simple_poi((1, 50, 9, 9, 9), (2, 50, 2, 2, 2))
        result = p1.join_right(p2)
        # (1,50) should be overwritten by p2
        coord = result[1, 50]
        self.assertAlmostEqual(coord[0], 9)
        self.assertIn((2, 50), result)

    def test_add_operator(self):
        p1 = _simple_poi((1, 50, 0, 0, 0))
        p2 = _simple_poi((2, 50, 1, 1, 1))
        result = p1 + p2
        self.assertIn((1, 50), result)
        self.assertIn((2, 50), result)

    def test_lshift_operator(self):
        p1 = _simple_poi((1, 50, 0, 0, 0))
        p2 = _simple_poi((2, 60, 5, 5, 5))
        result = p1 << p2
        self.assertIn((2, 60), result)

    def test_intersect_keeps_common(self):
        p1 = _simple_poi((1, 50, 0, 0, 0), (2, 50, 1, 1, 1), (3, 50, 2, 2, 2))
        p2 = _simple_poi((2, 50, 9, 9, 9), (3, 50, 8, 8, 8), (4, 50, 7, 7, 7))
        result = p1.intersect(p2)
        self.assertIn((2, 50), result)
        self.assertIn((3, 50), result)
        self.assertNotIn((1, 50), result)
        self.assertNotIn((4, 50), result)

    def test_intersect_empty(self):
        p1 = _simple_poi((1, 50, 0, 0, 0))
        p2 = _simple_poi((2, 60, 1, 1, 1))
        result = p1.intersect(p2)
        self.assertEqual(len(result), 0)

    def test_subtract_removes_matching(self):
        p1 = _simple_poi((1, 50, 0, 0, 0), (2, 50, 1, 1, 1), (3, 50, 2, 2, 2))
        p2 = _simple_poi((2, 50, 9, 9, 9))
        result = p1.subtract(p2)
        self.assertIn((1, 50), result)
        self.assertNotIn((2, 50), result)
        self.assertIn((3, 50), result)

    def test_join_left_inplace(self):
        p1 = _simple_poi((1, 50, 1, 1, 1))
        p2 = _simple_poi((2, 50, 2, 2, 2))
        p1.join_left_(p2)
        self.assertIn((2, 50), p1)

    def test_intersect_inplace(self):
        p1 = _simple_poi((1, 50, 0, 0, 0), (2, 50, 1, 1, 1))
        p2 = _simple_poi((2, 50, 9, 9, 9))
        p1.intersect_(p2)
        self.assertNotIn((1, 50), p1)
        self.assertIn((2, 50), p1)


class Test_POI_Extract(unittest.TestCase):
    def test_extract_subregion(self):
        p = _simple_poi((1, 50, 0, 0, 0), (1, 60, 1, 1, 1), (2, 50, 2, 2, 2))
        p2 = p.extract_subregion(50)
        self.assertIn((1, 50), p2)
        self.assertIn((2, 50), p2)
        self.assertNotIn((1, 60), p2)

    def test_extract_subregion_multiple(self):
        p = _simple_poi((1, 50, 0, 0, 0), (1, 60, 1, 1, 1), (2, 70, 2, 2, 2))
        p2 = p.extract_subregion(50, 60)
        self.assertIn((1, 50), p2)
        self.assertIn((1, 60), p2)
        self.assertNotIn((2, 70), p2)

    def test_extract_region(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 50, 1, 1, 1), (3, 50, 2, 2, 2))
        p2 = p.extract_region(2)
        self.assertNotIn((1, 50), p2)
        self.assertIn((2, 50), p2)
        self.assertNotIn((3, 50), p2)

    def test_extract_subregion_inplace(self):
        p = _simple_poi((1, 50, 0, 0, 0), (1, 60, 1, 1, 1))
        p.extract_subregion_(50)
        self.assertIn((1, 50), p)
        self.assertNotIn((1, 60), p)

    def test_extract_region_inplace(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 50, 1, 1, 1))
        p.extract_region_(1)
        self.assertIn((1, 50), p)
        self.assertNotIn((2, 50), p)


class Test_POI_Remove(unittest.TestCase):
    def test_remove_existing(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 60, 1, 1, 1))
        p2 = p.remove((1, 50))
        self.assertNotIn((1, 50), p2)
        self.assertIn((2, 60), p2)
        # original unchanged
        self.assertIn((1, 50), p)

    def test_remove_inplace(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 60, 1, 1, 1))
        p.remove_((2, 60))
        self.assertNotIn((2, 60), p)
        self.assertIn((1, 50), p)

    def test_remove_centroid(self):
        p = _simple_poi((1, 50, 0, 0, 0), (1, 60, 1, 1, 1))
        p2 = p.remove_centroid((1, 50))
        self.assertNotIn((1, 50), p2)
        self.assertIn((1, 60), p2)


class Test_POI_Round(unittest.TestCase):
    def test_round_coordinates(self):
        p = _simple_poi((1, 50, 1.23456, 2.34567, 3.45678))
        p2 = p.round(2)
        coord = p2[1, 50]
        self.assertAlmostEqual(coord[0], 1.23, places=5)
        self.assertAlmostEqual(coord[1], 2.35, places=5)
        self.assertAlmostEqual(coord[2], 3.46, places=5)

    def test_round_inplace(self):
        p = _simple_poi((1, 50, 1.555, 2.444, 3.111))
        p.round_(1)
        coord = p[1, 50]
        self.assertAlmostEqual(coord[0], 1.6, places=5)
        self.assertAlmostEqual(coord[1], 2.4, places=5)
        self.assertAlmostEqual(coord[2], 3.1, places=5)

    def test_round_does_not_mutate_original(self):
        p = _simple_poi((1, 50, 1.23456, 2.34567, 3.45678))
        _ = p.round(2)
        coord = p[1, 50]
        self.assertAlmostEqual(coord[0], 1.23456, places=4)


class Test_POI_ApplyAll(unittest.TestCase):
    def test_apply_all_translate(self):
        p = _simple_poi((1, 50, 1, 2, 3), (2, 50, 4, 5, 6))
        p2 = p.apply_all(lambda x, y, z: (x + 10, y + 10, z + 10))
        self.assertAlmostEqual(p2[1, 50][0], 11)
        self.assertAlmostEqual(p2[2, 50][1], 15)

    def test_apply_all_inplace(self):
        p = _simple_poi((1, 50, 1, 2, 3))
        p.apply_all(lambda x, y, z: (x * 2, y * 2, z * 2), inplace=True)
        self.assertAlmostEqual(p[1, 50][0], 2)
        self.assertAlmostEqual(p[1, 50][1], 4)
        self.assertAlmostEqual(p[1, 50][2], 6)

    def test_apply_all_not_inplace_preserves_original(self):
        p = _simple_poi((1, 50, 5, 5, 5))
        _ = p.apply_all(lambda x, y, z: (x + 100, y, z))
        self.assertAlmostEqual(p[1, 50][0], 5)


class Test_POI_DistanceCord(unittest.TestCase):
    def test_distance_to_self(self):
        p = _simple_poi((1, 50, 3, 4, 0))
        dists = p.calculate_distances_cord((3, 4, 0))
        self.assertAlmostEqual(dists[(1, 50)], 0.0, places=5)

    def test_distance_known_value(self):
        p = _simple_poi((1, 50, 0, 0, 0))
        # distance to (3, 4, 0) should be 5
        dists = p.calculate_distances_cord((3, 4, 0))
        self.assertAlmostEqual(dists[(1, 50)], 5.0, places=5)

    def test_distance_multiple_points(self):
        p = _simple_poi((1, 50, 0, 0, 0), (2, 50, 10, 0, 0))
        dists = p.calculate_distances_cord((5, 0, 0))
        self.assertAlmostEqual(dists[(1, 50)], 5.0, places=5)
        self.assertAlmostEqual(dists[(2, 50)], 5.0, places=5)


class Test_POI_FilterInsideShape(unittest.TestCase):
    def test_all_inside(self):
        p = _simple_poi((1, 50, 10, 10, 10), (2, 50, 50, 50, 50))
        p2 = p.filter_points_inside_shape()
        self.assertEqual(len(p2), 2)

    def test_outside_filtered(self):
        p = _simple_poi((1, 50, 10, 10, 10), (2, 50, 200, 200, 200))
        p2 = p.filter_points_inside_shape()
        self.assertIn((1, 50), p2)
        self.assertNotIn((2, 50), p2)

    def test_inplace(self):
        p = _simple_poi((1, 50, 5, 5, 5), (2, 50, 500, 500, 500))
        p.filter_points_inside_shape(inplace=True)
        self.assertIn((1, 50), p)
        self.assertNotIn((2, 50), p)


class Test_POI_Random(unittest.TestCase):
    def test_len_matches_inserted(self):
        for _ in range(repeats):
            n = random.randint(1, 20)
            p = get_poi(num_vert=n, num_subreg=1)
            self.assertEqual(len(p), n)

    def test_keys_region_count(self):
        for _ in range(repeats):
            n = random.randint(2, 15)
            p = get_poi(num_vert=n, num_subreg=1)
            regions = p.keys_region()
            self.assertEqual(len(regions), n)

    def test_round_trip_join(self):
        for _ in range(repeats):
            p1 = get_poi(num_vert=5, num_subreg=1, max_subreg=50)
            p2 = get_poi(num_vert=5, num_subreg=1, min_subreg=51, max_subreg=100)
            p1.origin = p2.origin
            p1.rotation = p2.rotation
            p1.zoom = p2.zoom
            p1.shape = p2.shape
            combined = p1.join_left(p2)
            # intersect with p1 should recover p1 only
            recovered = combined.intersect(p1)
            for k, _ in p1.items_flatten():
                self.assertIn(k, recovered)

    def test_subtract_disjoint(self):
        p1 = get_poi(num_vert=5, num_subreg=1, max_subreg=50)
        p2 = get_poi(num_vert=5, num_subreg=1, min_subreg=51, max_subreg=100)
        p2.origin = p1.origin
        p2.rotation = p1.rotation
        p2.zoom = p1.zoom
        p2.shape = p1.shape
        result = p1.subtract(p2)
        self.assertEqual(len(result), len(p1))


if __name__ == "__main__":
    unittest.main()
