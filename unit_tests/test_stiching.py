# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import io
import random
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox.core.compat import zip_strict
from TPTBox.core.nii_wrapper import NII
from TPTBox.stitching import GNC_stitch_T2w, stitching, stitching_raw
from TPTBox.tests.test_utils import overlap

try:
    import ants  # noqa: F401

    has_ants = True
except Exception:
    has_ants = False

# TODO saving did not work with the test and I do not understand why.


def _float_nii(shape=(28, 28, 28), translation=(0, 0, 0), blob_value=100.0) -> NII:
    """Build a small non-segmentation (float) NII with a central cuboid blob."""
    a = np.zeros(shape, dtype=np.float32)
    a[6:22, 6:22, 6:22] = blob_value
    aff = np.eye(4)
    aff[0, 3], aff[1, 3], aff[2, 3] = translation
    return NII(nib.Nifti1Image(a, aff), seg=False)


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

    return n, out_points, n.orientation, sizes


class TestStitchingFunction(unittest.TestCase):
    def test_stitching(
        self,
        idx="C66EMBZJmy75n4XHv2YsSXVs",
        match_histogram=True,
        store_ramp=False,
        verbose=False,
        min_value=0,
        bias_field=False,
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


class Test_stitching_public(unittest.TestCase):
    """Tests for the public ``stitching`` wrapper exported from TPTBox.stitching."""

    def test_stitching_two_overlapping_niis(self):
        n1 = _float_nii(translation=(0, 0, 0))
        n2 = _float_nii(translation=(8, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "stitched.nii.gz")
            result, ramp = stitching([n1, n2], out, verbose=False, kick_out_fully_integrated_images=False)
            self.assertIsInstance(result, nib.Nifti1Image)
            self.assertIsNone(ramp)
            self.assertTrue(out.exists())

    def test_stitching_store_ramp(self):
        n1 = _float_nii(translation=(0, 0, 0))
        n2 = _float_nii(translation=(8, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "stitched.nii.gz")
            result, ramp = stitching([n1, n2], out, verbose=False, store_ramp=True, kick_out_fully_integrated_images=False)
            self.assertIsInstance(result, nib.Nifti1Image)
            self.assertIsInstance(ramp, nib.Nifti1Image)

    def test_stitching_is_ct_min_value(self):
        n1 = _float_nii(translation=(0, 0, 0))
        n2 = _float_nii(translation=(8, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "ct.nii.gz")
            result, _ = stitching([n1, n2], out, is_ct=True, verbose=False, kick_out_fully_integrated_images=False)
            self.assertIsInstance(result, nib.Nifti1Image)

    def test_stitching_from_file_paths(self):
        n1 = _float_nii(translation=(0, 0, 0))
        n2 = _float_nii(translation=(8, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp, "a.nii.gz")
            n1.save(p1)
            out = Path(tmp, "stitched.nii.gz")
            result, _ = stitching([str(p1), n2], out, verbose=False, kick_out_fully_integrated_images=False)
            self.assertIsInstance(result, nib.Nifti1Image)

    def test_stitching_segmentation(self):
        s1 = get_nii(rotation=False)[0]
        s2 = get_nii(rotation=False)[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "seg.nii.gz")
            result, _ = stitching([s1, s2], out, is_seg=True, verbose=False, kick_out_fully_integrated_images=False)
            self.assertIsInstance(result, nib.Nifti1Image)


class Test_GNC_stitch(unittest.TestCase):
    def test_gnc_stitch_returns_uint16_nii(self):
        hws = _float_nii(translation=(0, 0, 0))
        bws = _float_nii(translation=(0, 12, 0))
        lws = _float_nii(translation=(0, 24, 0))
        out = GNC_stitch_T2w(hws, bws, lws)
        self.assertIsInstance(out, NII)
        self.assertEqual(out.dtype, np.uint16)


class Test_stitching_internals(unittest.TestCase):
    def test_get_rotation_and_spacing_from_affine(self):
        from TPTBox.stitching.stitching import get_rotation_and_spacing_from_affine

        aff = np.eye(4)
        aff[:3, :3] = np.diag([2.0, 3.0, 4.0])
        rot, spacing = get_rotation_and_spacing_from_affine(aff)
        np.testing.assert_allclose(spacing, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(rot, np.eye(3))

    def test_get_ras_affine_roundtrip(self):
        from TPTBox.stitching.stitching import get_ras_affine, get_rotation_and_spacing_from_affine

        rot, spacing = get_rotation_and_spacing_from_affine(np.eye(4))
        aff = get_ras_affine(rot, spacing, np.zeros(3))
        self.assertEqual(aff.shape, (4, 4))

    def test_get_array_and_set_array(self):
        from TPTBox.stitching.stitching import get_array, set_array

        nii = _float_nii().nii
        arr = get_array(nii)
        self.assertEqual(arr.shape, nii.shape)
        # set_array with a different dtype goes through the dtype-update branch
        new = set_array(nii, arr.astype(np.uint16))
        self.assertEqual(new.get_fdata().shape, nii.shape)
        # same dtype path
        same = set_array(nii, arr)
        self.assertEqual(same.get_fdata().shape, nii.shape)

    def test_argmin(self):
        from TPTBox.stitching.stitching import argmin

        self.assertEqual(argmin([3, 1, 2]), 1)

    def test_dilate_msk(self):
        from TPTBox.stitching.stitching import dilate_msk

        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[8:12, 8:12, 8:12] = 1
        out = dilate_msk(arr, mm=2)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.sum()), int(arr.sum()))

    def test_get_all_corner_points(self):
        from TPTBox.stitching.stitching import get_all_corner_points

        corners = get_all_corner_points(np.eye(4), (10, 10, 10))
        self.assertEqual(corners.shape, (8, 3))

    def test_get_max_affine_and_shape(self):
        from TPTBox.stitching.stitching import get_all_corner_points, get_max_affine_and_shape

        n1 = _float_nii(translation=(0, 0, 0))
        n2 = _float_nii(translation=(8, 0, 0))
        affines = [n1.affine, n2.affine]
        corners = np.concatenate([get_all_corner_points(n1.affine, n1.shape), get_all_corner_points(n2.affine, n2.shape)], axis=0)
        out = get_max_affine_and_shape(corners, affines, verbose=True)
        self.assertIsInstance(out, nib.Nifti1Image)

    def test_get_max_affine_and_shape_min_spacing(self):
        from TPTBox.stitching.stitching import get_all_corner_points, get_max_affine_and_shape

        n1 = _float_nii(translation=(0, 0, 0))
        n2 = _float_nii(translation=(8, 0, 0))
        affines = [n1.affine, n2.affine]
        corners = np.concatenate([get_all_corner_points(n1.affine, n1.shape), get_all_corner_points(n2.affine, n2.shape)], axis=0)
        out = get_max_affine_and_shape(corners, affines, min_spacing=2)
        self.assertIsInstance(out, nib.Nifti1Image)

    def test_compute_crop_slice(self):
        from TPTBox.stitching.stitching import compute_crop_slice

        arr = np.zeros((20, 20, 20), dtype=np.float32)
        arr[5:15, 6:14, 7:13] = 1
        nii = nib.Nifti1Image(arr, np.eye(4))
        sl = compute_crop_slice(nii, minimum=0, dist=0)
        self.assertEqual(len(sl), 3)
        self.assertEqual(sl[0].start, 5)
        # padding via dist expands the slices but stays in-bounds
        sl2 = compute_crop_slice(nii, minimum=0, dist=2)
        self.assertLessEqual(sl2[0].start, sl[0].start)

    def test_compute_crop_slice_empty_raises(self):
        from TPTBox.stitching.stitching import compute_crop_slice

        nii = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
        with self.assertRaises(ValueError):
            compute_crop_slice(nii)

    def test_buffer_reference_caches(self):
        from TPTBox.stitching.stitching import buffer_reference, buffer_references

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "ref.nii.gz"))
            _float_nii().save(path)
            arr = buffer_reference(path, bias_field=False)
            self.assertIsInstance(arr, np.ndarray)
            self.assertIn(path, buffer_references)
            arr2 = buffer_reference(path, bias_field=False)
            self.assertIs(arr2, arr)  # second call returns cached object


@unittest.skipIf(not has_ants, "requires antspyx")
class Test_n4_bias_field(unittest.TestCase):
    def test_n4_no_mask(self):
        from TPTBox.stitching.stitching import n4_bias_field_correction

        arr = (np.random.default_rng(0).random((24, 24, 24)) * 100).astype(np.float32)
        nii = nib.Nifti1Image(arr, np.eye(4))
        out = n4_bias_field_correction(nii, threshold=0)
        self.assertIsInstance(out, nib.Nifti1Image)

    def test_n4_with_auto_mask(self):
        from TPTBox.stitching.stitching import n4_bias_field_correction

        arr = (np.random.default_rng(1).random((24, 24, 24)) * 100).astype(np.float32)
        nii = nib.Nifti1Image(arr, np.eye(4))
        out = n4_bias_field_correction(nii, threshold=50, crop=False)
        self.assertIsInstance(out, nib.Nifti1Image)


# NOTE: stitching_tools.n4_bias is not tested — it calls NII.dilate_msk_(mm=3), but that
# method takes ``n_pixel`` (no ``mm`` kwarg), so the function raises TypeError before doing
# anything. This is a pre-existing source bug; covering it would require modifying source.


class Test_stitching_raw_branches(unittest.TestCase):
    """Directly exercise branch coverage in stitching.main (a.k.a. stitching_raw)."""

    def test_single_image_returns_none(self):
        result = stitching_raw([_float_nii().nii], None, save=False)
        self.assertEqual(result, (None, None))

    def test_empty_list_returns_none(self):
        result = stitching_raw([], None, save=False)
        self.assertEqual(result, (None, None))

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_match_histogram_previous_file(self, _mock_stdout):
        n1 = _float_nii(translation=(0, 0, 0)).nii
        n2 = _float_nii(translation=(8, 0, 0)).nii
        result, _ = stitching_raw([n1, n2], None, match_histogram=True, bias_field=False, save=False, verbose=True)
        self.assertIsInstance(result, nib.Nifti1Image)

    def test_match_histogram_reference_index_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp, "a.nii.gz")
            p2 = Path(tmp, "b.nii.gz")
            _float_nii(translation=(0, 0, 0)).save(p1)
            _float_nii(translation=(8, 0, 0)).save(p2)
            # histogram given as a list index "0" -> buffer_reference path
            r1, _ = stitching_raw(
                [str(p1), str(p2)], str(Path(tmp, "o1.nii.gz")), match_histogram=True, histogram="0", bias_field=False, save=True
            )
            self.assertIsInstance(r1, nib.Nifti1Image)
            # histogram given as an explicit file path
            r2, _ = stitching_raw(
                [str(p1), str(p2)], str(Path(tmp, "o2")), match_histogram=True, histogram=str(p1), bias_field=False, save=True
            )
            self.assertIsInstance(r2, nib.Nifti1Image)

    def test_ramp_edge_min_value_zero(self):
        n1 = _float_nii(translation=(0, 0, 0)).nii
        n2 = _float_nii(translation=(8, 0, 0)).nii
        result, _ = stitching_raw([n1, n2], None, ramp_edge_min_value=0, bias_field=False, save=False)
        self.assertIsInstance(result, nib.Nifti1Image)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_segmentation_verbose(self, _mock_stdout):
        s1 = get_nii(rotation=False)[0].nii
        s2 = get_nii(rotation=False)[0].nii
        result, _ = stitching_raw([s1, s2], None, is_segmentation=True, verbose=True, save=False)
        self.assertIsInstance(result, nib.Nifti1Image)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_crop_empty_and_store_ramp(self, _mock_stdout):
        n1 = _float_nii(translation=(0, 0, 0)).nii
        n2 = _float_nii(translation=(8, 0, 0)).nii
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "o.nii.gz")
            result, ramp = stitching_raw([n1, n2], str(out), store_ramp=True, crop_empty=True, bias_field=False, verbose=True, save=True)
            self.assertIsInstance(result, nib.Nifti1Image)
            self.assertIsInstance(ramp, nib.Nifti1Image)

    def test_segmentation_uint16_dtype_branch(self):
        # A label value >= 256 forces the uint16 output-dtype branch.
        def _seg_nii(translation):
            a = np.zeros((24, 24, 24), dtype=np.uint16)
            a[6:18, 6:18, 6:18] = 300
            aff = np.eye(4)
            aff[0, 3] = translation
            return nib.Nifti1Image(a, aff)

        result, _ = stitching_raw([_seg_nii(0), _seg_nii(8)], None, is_segmentation=True, save=False)
        self.assertIsInstance(result, nib.Nifti1Image)

    def test_auto_output_path_from_first_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp, "a.nii.gz")
            p2 = Path(tmp, "b.nii.gz")
            _float_nii(translation=(0, 0, 0)).save(p1)
            _float_nii(translation=(8, 0, 0)).save(p2)
            # bare filename (no path separator) -> output is placed next to images[0]
            result, _ = stitching_raw([str(p1), str(p2)], "bare_out.nii.gz", bias_field=False, save=True)
            self.assertIsInstance(result, nib.Nifti1Image)
            self.assertTrue(Path(tmp, "bare_out.nii.gz").exists())

    @unittest.skipIf(not has_ants, "requires antspyx")
    def test_n4_bias_field_correction_crop(self):
        from TPTBox.stitching.stitching import n4_bias_field_correction

        arr = (np.random.default_rng(3).random((24, 24, 24)) * 100 + 20).astype(np.float32)
        nii = nib.Nifti1Image(arr, np.eye(4))
        out = n4_bias_field_correction(nii, threshold=50, crop=True)
        self.assertIsInstance(out, nib.Nifti1Image)

    @unittest.skipIf(not has_ants, "requires antspyx")
    def test_bias_field_per_input_and_final(self):
        n1 = _float_nii(translation=(0, 0, 0)).nii
        n2 = _float_nii(translation=(8, 0, 0)).nii
        result, _ = stitching_raw([n1, n2], None, bias_field=True, save=False)
        self.assertIsInstance(result, nib.Nifti1Image)


class Test_stitching_tools_helpers(unittest.TestCase):
    def test_center_frontal(self):
        from TPTBox.stitching.stitching_tools import _center_frontal

        sl = _center_frontal(300)
        self.assertIsInstance(sl, slice)

    def test_crop_borders_valid(self):
        from TPTBox.stitching.stitching_tools import _crop_borders

        cut = {"BWS": (slice(0, 20), slice(None), slice(None))}
        out = _crop_borders(_float_nii(), "BWS", cut)
        self.assertIsInstance(out, NII)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_crop_borders_missing_chunk(self, _mock_stdout):
        from TPTBox.stitching.stitching_tools import _crop_borders

        cut = {"BWS": (slice(0, 20), slice(None), slice(None))}
        with self.assertRaises(KeyError):
            _crop_borders(_float_nii(), "ZZZ", cut)


if __name__ == "__main__":
    unittest.main()
