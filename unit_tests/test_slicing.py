# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import random
import unittest
from pathlib import Path

from TPTBox.core.compat import zip_strict
import nibabel as nib
import numpy as np

from TPTBox.core.nii_wrapper import AX_CODES, NII
from TPTBox.stitching import stitching_raw
from TPTBox.tests.test_utils import overlap

# TODO saving did not work with the test and I do not understand why.


def get_random_ax_code() -> AX_CODES:
    directions = [["R", "L"], ["S", "I"], ["A", "P"]]
    idx = [0, 1, 2]
    random.shuffle(idx)
    return tuple(directions[i][random.randint(0, 1)] for i in idx)  # type: ignore


def get_nii(x: tuple[int, int, int] | None = None, num_point=3, rotation=True):  # type: ignore
    if x is None:
        x = (random.randint(20, 40), random.randint(20, 40), random.randint(20, 40))
    a = np.zeros(x, dtype=np.uint16)
    points = []
    out_points: dict[int, dict[int, tuple[float, float, float]]] = {}
    sizes = []
    idx = 1
    while True:
        if num_point == len(points):
            break
        point = tuple(random.randint(1, a - 1) for a in x)
        size = tuple(random.randint(1, 1 + a) for a in [5, 5, 5])
        if any(a - b < 0 for a, b in zip_strict(point, size)):
            continue
        if any(a + b > c - 1 for a, b, c in zip_strict(point, size, x)):
            continue
        skip = False
        for p2, s2 in zip_strict(points, sizes):
            if overlap(point, size, p2, s2):
                skip = True
                break
        if skip:
            continue
        a[
            point[0] - size[0] : point[0] + size[0] + 1,
            point[1] - size[1] : point[1] + size[1] + 1,
            point[2] - size[2] : point[2] + size[2] + 1,
        ] = idx

        points.append(point)
        sizes.append(size)
        out_points[idx] = {50: tuple(float(a) for a in point)}

        idx += 1
    aff = np.eye(4)

    aff[0, 3] = random.randint(-100, 100)
    aff[1, 3] = random.randint(-100, 100)
    aff[2, 3] = random.randint(-100, 100)
    if rotation:
        m = 30
        from scipy.spatial.transform import Rotation

        r = Rotation.from_euler("xyz", (random.randint(-m, m), random.randint(-m, m), random.randint(-m, m)), degrees=True)
        aff[:3, :3] = r.as_matrix()
    n = NII(nib.Nifti1Image(a, aff), seg=True)
    n.reorient_(get_random_ax_code())
    return n


class TestNPInterOperability(unittest.TestCase):
    def test_get_int(self):
        # Define inputs for the function
        for _ in range(5):
            nii = get_nii()
            arr = nii.get_array()
            for _ in range(5):
                x, y, z = random.randint(0, nii.shape[0] - 1), random.randint(0, nii.shape[1] - 1), random.randint(0, nii.shape[2] - 1)
                assert nii[x, y, z] == arr[x, y, z], (nii[x, y, z], arr[x, y, z])

    def test_assign_int(self):
        # Define inputs for the function
        for _ in range(5):
            nii = get_nii()
            for _ in range(5):
                v = random.randint(0, 255)
                x, y, z = random.randint(0, nii.shape[0] - 1), random.randint(0, nii.shape[1] - 1), random.randint(0, nii.shape[2] - 1)
                v_old = nii[x, y, z]
                nii[x, y, z] = v
                assert nii[x, y, z] == v, (nii[x, y, z], v)
                assert v == v_old or nii[x, y, z] != v_old

    def test_slice(self):
        for _ in range(5):
            nii = get_nii()
            sl_nii: NII = nii[:10, :10, :10]
            assert sl_nii.shape == (10, 10, 10)
            sl_nii = nii[::2, ::3, ::4]
            assert sl_nii.zoom == (2, 3, 4)
            sl_nii = nii[::-1, ::-1, ::-1]
            assert sl_nii.zoom == (1, 1, 1)
            for axis in ("R", "A", "S"):
                ax = sl_nii.get_axis(axis)
                assert ax == nii.get_axis(axis)
                assert sl_nii.orientation[ax] != nii.orientation[ax]

    def test_slice_assign(self):
        for _ in range(5):
            nii = get_nii()
            nii[:10, :10, :10] = 99
            assert nii[5, 5, 5] == 99

    def test_np(self):
        print(np.sum(get_nii()))
        print(np.sin(get_nii()))


if __name__ == "__main__":
    unittest.main()
