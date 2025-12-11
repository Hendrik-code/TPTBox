import os
import unittest
from pathlib import Path

import numpy as np
import requests

# Import your functions
from TPTBox.core.internal.slicer_nrrd import load_slicer_nrrd, save_slicer_nrrd
from TPTBox.core.nii_wrapper import NII


class TestSlicerSegmentationIO(unittest.TestCase):
    slicerio_data = Path(__file__).parent / "slicerio_data"
    base_url = "https://raw.githubusercontent.com/lassoan/slicerio/main/slicerio/data"

    files = {  # noqa: RUF012
        "CT": "CTChest4.nrrd",
        "Seg": "Segmentation.seg.nrrd",
        "SegOverlap": "SegmentationOverlapping.seg.nrrd",
    }

    @classmethod
    def setUpClass(cls):
        """Ensure the data directory exists and download all test files."""
        os.makedirs(cls.slicerio_data, exist_ok=True)
        for filename in cls.files.values():
            url = f"{cls.base_url}/{filename}"
            out_local = cls.slicerio_data / filename
            if not out_local.exists():
                cls.download_file(url, out_local)

    @staticmethod
    def download_file(url: str, out_path: Path):
        """Download a file from a URL."""
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.writelines(resp.iter_content(chunk_size=8192))
        print(f"Downloaded {url} → {out_path}")

    def roundtrip_test(self, filename: str):
        """Helper function: load, save, reload, and compare arrays."""
        path = self.slicerio_data / filename
        seg = "seg." in filename
        nii = load_slicer_nrrd(path, seg)
        arr = nii.get_array()

        # Save to roundtrip file
        out_seg = path.with_name(path.stem + ".roundtrip.seg.nrrd")
        save_slicer_nrrd(nii, out_seg)

        # Reload saved file
        nii2 = load_slicer_nrrd(out_seg, seg)
        arr2 = nii2.get_array()

        # Compare arrays
        self.assertTrue(np.array_equal(arr, arr2), f"Round-trip arrays differ for {filename}")

        # Optional: remove roundtrip file
        out_seg.unlink(missing_ok=True)

    def roundtrip_test2(self, filename: str):
        """Helper function: load, save, reload, and compare arrays."""
        path = self.slicerio_data / filename
        seg = "seg." in filename
        nii = NII.load(path, seg)
        arr = nii.get_array()

        # Save to roundtrip file
        out_seg = path.with_name(path.stem + ".roundtrip.seg.nrrd")
        nii.save_nrrd(out_seg)

        # Reload saved file
        nii2 = NII.load(out_seg, seg)
        arr2 = nii2.get_array()

        # Compare arrays
        self.assertTrue(np.array_equal(arr, arr2), f"Round-trip arrays differ for {filename}")

        # Optional: remove roundtrip file
        out_seg.unlink(missing_ok=True)

    def test_segmentation(self):
        """Test round-trip for Segmentation.seg.nrrd."""
        self.roundtrip_test(self.files["Seg"])

    def test_segmentation_CT(self):
        """Test round-trip for Segmentation.seg.nrrd."""
        self.roundtrip_test(self.files["CT"])

    def test_segmentation_overlapping(self):
        """Test round-trip for SegmentationOverlapping.seg.nrrd."""
        self.roundtrip_test(self.files["SegOverlap"])

    def test_ct_file_exists(self):
        """Just check that CT file exists."""
        path = self.slicerio_data / self.files["CT"]
        self.assertTrue(path.exists(), f"{path} should exist")

    def test_segmentation2(self):
        """Test round-trip for Segmentation.seg.nrrd."""
        self.roundtrip_test2(self.files["Seg"])

    def test_segmentation_CT2(self):
        """Test round-trip for Segmentation.seg.nrrd."""
        self.roundtrip_test(self.files["CT"])

    def test_segmentation_overlapping2(self):
        """Test round-trip for SegmentationOverlapping.seg.nrrd."""
        self.roundtrip_test2(self.files["SegOverlap"])


if __name__ == "__main__":
    unittest.main()
