# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import random
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox.core.nii_wrapper import NII
from TPTBox.stitching import stitching_raw
from TPTBox.tests.test_utils import overlap

# TODO saving did not work with the test and I do not understand why.


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
        if any(a - b < 0 for a, b in zip(point, size)):
            continue
        if any(a + b > c - 1 for a, b, c in zip(point, size, x)):
            continue
        skip = False
        for p2, s2 in zip(points, sizes):
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

    return n, out_points, n.orientation, sizes


class TestStitchingFunction(unittest.TestCase):
    def test_stitching(
        self,
        idx="C66EMBZJmy75n4XHv2YsSXVs",
        match_histogram=True,
        store_ramp=False,
        verbose=False,
        min_value=0,
        bias_field=True,
        crop_to_bias_field=False,
        crop_empty=False,
        histogram=None,
        ramp_edge_min_value=0,
        min_spacing=None,
        kick_out_fully_integrated_images=False,
        is_segmentation=False,
        dtype=float,
        save=False,
    ):
        # Define inputs for the function
        images = [get_nii()[0].nii, get_nii()[0].nii]
        output = Path(f"~/output{idx}.nii.gz")
        if save:
            print(output.absolute())
        output.unlink(missing_ok=True)

        # Call the function
        result, _ = stitching_raw(
            images,
            str(output),
            match_histogram,
            store_ramp,
            verbose,
            min_value,
            bias_field,
            crop_to_bias_field,
            crop_empty,
            histogram,
            ramp_edge_min_value,
            min_spacing,
            kick_out_fully_integrated_images,
            is_segmentation,
            dtype,
            save,
        )
        if save:
            self.assertTrue(output.parent.exists(), output)
            self.assertTrue(output.exists(), output)
            output.unlink(missing_ok=True)
        # Assertions
        self.assertIsInstance(result, nib.Nifti1Image)  # Check if result is a Nifti1Image instance
        # Add more assertions based on your requirements

    def test_stitching2(self):
        self.test_stitching(
            idx="X",
            match_histogram=False,
            store_ramp=False,
            verbose=True,
            min_value=-1024,
            bias_field=False,
            crop_to_bias_field=False,
            crop_empty=True,
            histogram=None,
            ramp_edge_min_value=20,
            min_spacing=2,
            kick_out_fully_integrated_images=True,
            is_segmentation=False,
            dtype=float,
            save=False,
        )

    def test_stitching3(self):
        self.test_stitching(
            idx="X6Fqat2JLZbJKCom6BUX7F84",
            match_histogram=False,
            min_value=0,
            bias_field=False,
            ramp_edge_min_value=0,
            is_segmentation=True,
            save=False,
            store_ramp=True,
        )


if __name__ == "__main__":
    unittest.main()
