"""Round-trip I/O tests for NII.load, NII.save, and NII.from_numpy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from TPTBox import NII


def _make_nii(arr: np.ndarray, seg: bool = True, zoom=(1.0, 1.0, 1.0)) -> NII:
    affine = np.diag([*zoom, 1.0])
    return NII.from_numpy(arr, affine=affine, seg=seg)


class Test_NII_IO(unittest.TestCase):
    """Verify that NII survives a save→load round-trip with consistent shape and metadata."""

    def test_save_load_roundtrip_shape(self):
        arr = np.zeros((10, 12, 14), dtype=np.uint8)
        arr[3:7, 3:9, 3:11] = 1
        nii = _make_nii(arr, zoom=(1.0, 2.0, 3.0))
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            path = Path(f.name)
        try:
            nii.save(path, verbose=False)
            loaded = NII.load(path, seg=True)
            self.assertEqual(loaded.shape, nii.shape)
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_roundtrip_data(self):
        arr = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        nii = _make_nii(arr, seg=True)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            path = Path(f.name)
        try:
            nii.save(path, verbose=False)
            loaded = NII.load(path, seg=True)
            np.testing.assert_array_equal(loaded.get_array(), arr)
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_preserves_zoom(self):
        arr = np.zeros((5, 6, 7), dtype=np.uint8)
        nii = _make_nii(arr, zoom=(1.5, 2.5, 3.5))
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            path = Path(f.name)
        try:
            nii.save(path, verbose=False)
            loaded = NII.load(path, seg=True)
            for orig, restored in zip(nii.zoom, loaded.zoom):
                self.assertAlmostEqual(orig, restored, places=4)
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_seg_flag(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        nii = _make_nii(arr, seg=True)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            path = Path(f.name)
        try:
            nii.save(path, verbose=False)
            loaded = NII.load(path, seg=True)
            self.assertTrue(loaded.seg)
        finally:
            path.unlink(missing_ok=True)


class Test_NII_FromNumpy(unittest.TestCase):
    """Tests for NII.from_numpy factory method."""

    def test_shape(self):
        arr = np.zeros((8, 9, 10), dtype=np.uint8)
        nii = NII.from_numpy(arr, affine=np.eye(4), seg=True)
        self.assertEqual(nii.shape, (8, 9, 10))

    def test_seg_flag_true(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        self.assertTrue(NII.from_numpy(arr, affine=np.eye(4), seg=True).seg)

    def test_seg_flag_false(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        self.assertFalse(NII.from_numpy(arr, affine=np.eye(4), seg=False).seg)

    def test_affine_zoom_extracted(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        nii = NII.from_numpy(arr, affine=affine, seg=True)
        self.assertAlmostEqual(nii.zoom[0], 2.0)
        self.assertAlmostEqual(nii.zoom[1], 3.0)
        self.assertAlmostEqual(nii.zoom[2], 4.0)

    def test_data_round_trip(self):
        arr = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        nii = NII.from_numpy(arr, affine=np.eye(4), seg=False)
        np.testing.assert_array_equal(nii.get_array(), arr)


if __name__ == "__main__":
    unittest.main()
