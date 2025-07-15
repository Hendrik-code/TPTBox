# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import TPTBox.core.bids_files as bids
from TPTBox import POI, POI_Global
from TPTBox.core.nii_wrapper import AX_CODES, NII
from TPTBox.core.vert_constants import conversion_poi2text

repeats = 20


def get_random_ax_code() -> AX_CODES:
    directions = [["R", "L"], ["S", "I"], ["A", "P"]]
    idx = [0, 1, 2]
    random.shuffle(idx)
    return tuple(directions[i][random.randint(0, 1)] for i in idx)  # type: ignore


def get_random_shape():
    return tuple(int(1000 * random.random() + 10) for _ in range(3))  # type: ignore


def get_centroids(x: tuple[int, int, int] = (50, 30, 40), num_point=3, default_sub=None):
    out_points: dict[tuple[int, int], tuple[float, float, float]] = {}

    for _ in range(num_point):
        point = tuple(random.randint(1, a * 100) / 100.0 for a in x)
        out_points[random.randint(1, 256), random.randint(1, 256) if default_sub is None else 50] = point
    return POI(
        centroids=out_points,
        orientation=get_random_ax_code(),
        zoom=(random.random() * 3, random.random() * 3, random.random() * 3),
        shape=x,
        origin=(0, 0, 0),
        rotation=np.eye(3),
    )


def get_centroids2(x: tuple[int, int, int] = (50, 30, 40), num_point=3):
    out_points: dict[tuple[int, int], tuple[float, float, float]] = {}
    l = list(conversion_poi2text.keys())
    for _ in range(num_point):
        point = tuple(random.randint(1, a * 100) / 100.0 for a in x)
        out_points[random.randint(1, 27), l[random.randint(0, len(l) - 1)]] = point
    return POI(centroids=out_points, orientation=get_random_ax_code(), zoom=(1, 1, 1), shape=x)


s = Path("BIDS/test/")
if not s.exists():
    s = Path()


class Test_Centroids(unittest.TestCase):
    def test_save_0(self):
        for _ in range(repeats):
            p = Path(s, "test_save_0.json")
            cdt = get_centroids(x=(500, 700, 900), num_point=99)
            cdt.save(p, verbose=False)
            cdt2 = POI.load(p)
            self.assertEqual(cdt, cdt2)
            p.unlink()

    def test_save_1(self):
        for _ in range(repeats):
            p = Path(s, "test_save_1.json")
            cdt = get_centroids2(x=get_random_shape(), num_point=99)
            cdt.save(p, verbose=False, save_hint=1)
            cdt2 = POI.load(p)
            self.assertEqual(cdt, cdt2)
            p.unlink()

    def test_save_2(self):
        for _ in range(repeats):
            p = Path(s, "test_save_2.json")
            cdt = get_centroids(x=get_random_shape(), num_point=99)
            cdt.save(p, verbose=False, save_hint=2)
            cdt2 = POI.load(p)
            self.assertEqual(cdt, cdt2)
            Path(p).unlink()

    def test_save_10(self):
        for _ in range(repeats):
            p = Path(s, "test_save_10.json")
            cdt = get_centroids(x=get_random_shape(), num_point=2)
            cdt.save(p, verbose=False, save_hint=10)
            cdt2 = POI.load(p)
            cdt = cdt.rescale((1, 1, 1), verbose=False).reorient_(("R", "P", "I"))
            cdt.shape = None  # type: ignore
            cdt.rotation = None  # type: ignore
            self.assertEqual(cdt, cdt2)
            Path(p).unlink()

    def test_save_Glob(self):
        for _ in range(repeats):
            p = Path(s, "test_save_glob.json")
            cdt = get_centroids(x=get_random_shape(), num_point=20).to_global()
            cdt.save(p, verbose=False)
            cdt2 = POI_Global.load(p)
            self.assertEqual(cdt, cdt2)
            Path(p).unlink()

    def test_save_Glob_2(self):
        for _ in range(repeats):
            p = Path(s, "test_save_glob_2.json")
            p2 = Path(s, "test_save_glob_3.json")
            cdt = get_centroids(x=get_random_shape(), num_point=20)
            glob_poi = cdt.to_global()
            cdt.save(p, verbose=False)
            glob_poi.save(p2, verbose=False)
            cdt_a = POI_Global.load(p)
            cdt_b = POI_Global.load(p2)
            self.assertEqual(cdt_a, cdt_b)
            Path(p).unlink()
            Path(p2).unlink()

    def test_save_all(self):
        for _ in range(repeats):
            p = Path(s, "test_save_all.json")
            cdt = get_centroids2(x=get_random_shape(), num_point=99)
            cdt2 = cdt
            for _ in range(5):
                cdt2.save(p, verbose=False, save_hint=random.randint(0, 2))
                cdt2 = POI.load(p)
            self.assertEqual(cdt, cdt2)
            Path(p).unlink()

    def test_save_Glob_mkr(self):
        for _ in range(repeats):
            p = Path(s, "test_save_glob.mrk.json")
            cdt = get_centroids(x=get_random_shape(), num_point=20).to_global()
            cdt.save_mrk(p)
            cdt2 = POI_Global.load(p)
            self.assertEqual(cdt, cdt2)
            Path(p).unlink()

    def test_save_Glob_2_mkr(self):
        for _ in range(repeats):
            p = Path(s, "test_save_glob_2.json")
            p2 = Path(s, "test_save_glob_3.mrk.json")
            cdt = get_centroids(x=get_random_shape(), num_point=20)
            glob_poi = cdt.to_global()
            cdt.save(p, verbose=False)
            glob_poi.save_mrk(p2)
            cdt_a = POI_Global.load(p)
            cdt_b = POI_Global.load(p2)
            self.assertEqual(cdt_a, cdt_b)
            Path(p).unlink()
            Path(p2).unlink()


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
