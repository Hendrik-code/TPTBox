"""Unit tests for `TPTBox.spine.spinestats.bone_density`: `ct_bone_density` and the CT json sidecar reader.

The spine is synthesised: each vertebra is a cube whose outer 2 voxels (the "cortex") are 400 HU and whose interior is a
known HU value. With a 3 mm erosion at 1 mm spacing the cortex is removed, so the expected statistics are exact.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import NII, Location, Vertebra_Instance
from TPTBox.spine.spinestats.bone_density import (
    CALIBRATIONS,
    Calibration,
    bone_density_caveats,
    ct_bone_density,
    ct_parameters,
    find_sidecar,
    qct_category,
    read_sidecar,
    resolve_calibration,
    resolve_ct_params,
)

SIDE = 20
ERODED_SIDE = SIDE - 2 * 3  # erosion of 3 mm at 1 mm spacing removes 3 voxel layers on every side
L1, L2, L3, L4 = Vertebra_Instance.L1, Vertebra_Instance.L2, Vertebra_Instance.L3, Vertebra_Instance.L4


def _nii(arr: np.ndarray, seg: bool, zoom=(1.0, 1.0, 1.0)) -> NII:
    affine = np.diag([*zoom, 1.0])
    return NII(nib.Nifti1Image(arr, affine), seg=seg)


def _spine(hu_by_vertebra: dict, zoom=(1.0, 1.0, 1.0)) -> tuple[NII, NII, NII]:
    """CT, vertebra and subregion NIIs with one cube per vertebra (stacked along axis 2, separated by 5 voxels)."""
    n = len(hu_by_vertebra)
    shape = (SIDE + 10, SIDE + 10, n * (SIDE + 5) + 10)
    ct = np.full(shape, -50.0, dtype=np.float32)
    vert = np.zeros(shape, dtype=np.uint8)
    sub = np.zeros(shape, dtype=np.uint8)
    for k, (v, hu) in enumerate(hu_by_vertebra.items()):
        z0 = 5 + k * (SIDE + 5)
        cube = (slice(5, 5 + SIDE), slice(5, 5 + SIDE), slice(z0, z0 + SIDE))
        inner = (slice(7, 3 + SIDE), slice(7, 3 + SIDE), slice(z0 + 2, z0 + SIDE - 2))
        vert[cube] = v.value
        sub[cube] = Location.Vertebra_Corpus.value
        ct[cube] = 400.0  # cortex
        ct[inner] = hu
    return _nii(ct, False, zoom), _nii(vert, True, zoom), _nii(sub, True, zoom)


class Test_ct_bone_density(unittest.TestCase):
    def test_trabecular_hu_without_cortex(self):
        ct, vert, sub = _spine({L1: 150.0})
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)])
        for key, expected in [("mean_hu", 150.0), ("median_hu", 150.0), ("std_hu", 0.0), ("n_voxels", ERODED_SIDE**3)]:
            with self.subTest(key=key):
                self.assertEqual(out["L1-L1"][key], expected)
                self.assertEqual(out["per_vertebra"]["L1"][key], expected)
        self.assertEqual(out["L1-L1"]["volume_mm3"], float(ERODED_SIDE**3))
        self.assertEqual(out["warnings"], [])

    def test_no_erosion_includes_cortex(self):
        ct, vert, sub = _spine({L1: 150.0})
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)], erode_mm=0)
        self.assertEqual(out["L1-L1"]["n_voxels"], SIDE**3)
        self.assertGreater(out["L1-L1"]["mean_hu"], 150.0)

    def test_region_pools_voxels_and_reports_missing_levels(self):
        ct, vert, sub = _spine({L1: 150.0, L2: 250.0, L4: 100.0})  # L3 is missing
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L2), (L1, L4)])
        self.assertEqual(out["L1-L2"]["mean_hu"], 200.0)
        self.assertTrue(out["L1-L2"]["complete"])
        self.assertEqual(out["L1-L2"]["vertebrae"], ["L1", "L2"])
        self.assertAlmostEqual(out["L1-L4"]["mean_hu"], 500.0 / 3, places=2)  # equal voxel counts per level
        self.assertFalse(out["L1-L4"]["complete"])
        self.assertEqual(out["L1-L4"]["vertebrae"], ["L1", "L2", "L4"])
        self.assertNotIn("L3", out["per_vertebra"])

    def test_region_without_any_vertebra_is_none(self):
        ct, vert, sub = _spine({L1: 150.0})
        self.assertIsNone(ct_bone_density(ct, vert, sub, regions=[(L3, L4)])["L3-L4"])

    def test_hu_range_excludes_voxels(self):
        ct, vert, sub = _spine({L1: 150.0})
        arr = ct.get_array()
        arr[15:17, 15:17, 15:17] = 2000.0  # 8 sclerotic voxels in the interior
        ct = _nii(arr, False)
        plain = ct_bone_density(ct, vert, sub, regions=[(L1, L1)])
        clipped = ct_bone_density(ct, vert, sub, regions=[(L1, L1)], hu_range=(-100.0, 1000.0))
        self.assertGreater(plain["L1-L1"]["mean_hu"], 150.0)
        self.assertEqual(clipped["L1-L1"]["mean_hu"], 150.0)
        self.assertEqual(clipped["L1-L1"]["n_excluded"], 8)
        self.assertEqual(clipped["L1-L1"]["n_voxels"], ERODED_SIDE**3 - 8)

    def test_calibration_is_only_added_when_given(self):
        ct, vert, sub = _spine({L1: 150.0})
        self.assertNotIn("bmd_mg_cm3", ct_bone_density(ct, vert, sub, regions=[(L1, L1)])["L1-L1"])
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)], calibration=(0.5, 10.0))
        self.assertEqual(out["L1-L1"]["bmd_mg_cm3"], 85.0)

    def test_non_hu_image_warns(self):
        ct, vert, sub = _spine({L1: 5000.0})
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)])
        self.assertTrue(any("Hounsfield" in w for w in out["warnings"]))

    def test_erosion_uses_millimetres_not_voxels(self):
        # 2 mm spacing along the last axis: a 3 mm erosion removes 2 voxel layers there (distance 2 mm, 4 mm; 4 > 3 keeps the 2nd)
        ct, vert, sub = _spine({L1: 150.0}, zoom=(1.0, 1.0, 2.0))
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)])
        iso = ct_bone_density(*_spine({L1: 150.0}), regions=[(L1, L1)])
        self.assertNotEqual(out["L1-L1"]["n_voxels"], iso["L1-L1"]["n_voxels"])
        self.assertEqual(out["L1-L1"]["volume_mm3"], out["L1-L1"]["n_voxels"] * 2.0)

    def test_per_vertebra_can_be_disabled(self):
        ct, vert, sub = _spine({L1: 150.0})
        self.assertNotIn("per_vertebra", ct_bone_density(ct, vert, sub, regions=[(L1, L1)], per_vertebra=False))


# the sidecar shipped with the VerSe dataset: DICOM tag names, everything as strings
VERSE = {
    "dataset": "VERSE  - DOI 10.17605/OSF.IO/T98FZ",
    "FilterType": "B",
    "KVP": "120",
    "Manufacturer": "Philips",
    "ManufacturerModelName": "IQon - Spectral CT",
    "PixelSpacing": "0.68359375\\0.68359375",
    "SliceThickness": "0.9",
}


class Test_ct_parameters(unittest.TestCase):
    def test_verse_sidecar_is_normalised(self):
        p = ct_parameters(VERSE)
        self.assertTrue(p["sidecar_found"])
        self.assertEqual(p["kvp"], 120.0)
        self.assertEqual(p["slice_thickness_mm"], 0.9)
        self.assertEqual(p["pixel_spacing_mm"], [0.68359375, 0.68359375])
        self.assertEqual(p["convolution_kernel"], "B")
        self.assertEqual(p["manufacturer"], "Philips")
        self.assertIsNone(p["tube_current_ma"])
        self.assertIsNone(p["contrast_agent_given"])  # no tag does not prove "no contrast"

    def test_dicom_style_values_lists_and_numbers(self):
        p = ct_parameters({"KVP": 100, "ConvolutionKernel": ["I70f", "3"], "XRayTubeCurrent": "250", "SliceThickness": [1.5]})
        self.assertEqual((p["kvp"], p["tube_current_ma"], p["slice_thickness_mm"]), (100.0, 250.0, 1.5))
        self.assertEqual(p["convolution_kernel"], "I70f 3")

    def test_missing_sidecar_gives_none_values(self):
        for sidecar in (None, {}):
            with self.subTest(sidecar=sidecar):
                p = ct_parameters(sidecar)
                self.assertEqual(p["sidecar_found"], sidecar is not None)
                self.assertIsNone(p["kvp"])
        self.assertEqual(bone_density_caveats(ct_parameters(None)), [])  # nothing to say without a sidecar

    def test_contrast_tags(self):
        for key in ("ContrastBolusAgent", "ContrastBolusVolume", "ContrastBolusTotalDose"):
            with self.subTest(key=key):
                self.assertTrue(ct_parameters({"KVP": "120", key: "Iopamidol" if key.endswith("Agent") else 80})["contrast_agent_given"])
        self.assertIsNone(ct_parameters({"KVP": "120", "ContrastBolusAgent": ""})["contrast_agent_given"])

    def test_caveats(self):
        self.assertEqual(bone_density_caveats(ct_parameters(VERSE)), [])
        self.assertTrue(any("contrast" in c for c in bone_density_caveats(ct_parameters({**VERSE, "ContrastBolusAgent": "x"}))))
        self.assertTrue(any("100 kVp" in c for c in bone_density_caveats(ct_parameters({**VERSE, "KVP": "100"}))))
        self.assertTrue(any("not in the sidecar" in c for c in bone_density_caveats(ct_parameters({"Manufacturer": "x"}))))

    def test_find_and_read_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            nii = Path(d) / "sub-1_dir-ax_ct.nii.gz"
            nii.touch()
            self.assertIsNone(find_sidecar(nii))
            self.assertIsNone(read_sidecar(nii))
            js = Path(d) / "sub-1_dir-ax_ct.json"
            js.write_text(json.dumps(VERSE))
            self.assertEqual(find_sidecar(nii), js)
            self.assertEqual(read_sidecar(str(nii)), VERSE)
            js.write_text("{broken")  # unreadable -> None, never raises
            self.assertIsNone(read_sidecar(nii))
            js.write_text("[1, 2]")  # not an object
            self.assertIsNone(read_sidecar(nii))

    def test_bone_density_flags_ct_params(self):
        ct, vert, sub = _spine({L1: 150.0})
        base = ct_bone_density(ct, vert, sub, regions=[(L1, L1)])
        self.assertEqual(base["warnings"], [])
        self.assertIsNone(base["settings"]["ct_params"])
        acq = ct_parameters({**VERSE, "KVP": "80", "ContrastBolusAgent": "x"})
        out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)], ct_params=acq)
        self.assertEqual(len(out["warnings"]), 2)
        self.assertEqual(out["settings"]["ct_params"]["kvp"], 80.0)
        self.assertEqual(out["L1-L1"], base["L1-L1"])  # flagged, never corrected

    def test_ct_params_accepts_dict_raw_sidecar_or_paths(self):
        ct, vert, sub = _spine({L1: 150.0})
        raw = {**VERSE, "KVP": "80"}  # a raw sidecar dict, not yet normalised
        with tempfile.TemporaryDirectory() as d:
            nii = Path(d) / "sub-1_dir-ax_ct.nii.gz"
            nii.touch()
            js = Path(d) / "sub-1_dir-ax_ct.json"
            js.write_text(json.dumps(raw))
            sources = {
                "normalised dict": ct_parameters(raw),
                "raw sidecar dict": raw,
                "path of the json": js,
                "path of the json (str)": str(js),
                "path of the CT": nii,
                "path of the CT (str)": str(nii),
            }
            for name, src in sources.items():
                with self.subTest(source=name):
                    out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)], ct_params=src)
                    self.assertEqual(out["settings"]["ct_params"]["kvp"], 80.0)
                    self.assertTrue(any("80 kVp" in w for w in out["warnings"]))

    def test_ct_params_path_without_sidecar_is_not_an_error(self):
        ct, vert, sub = _spine({L1: 150.0})
        with tempfile.TemporaryDirectory() as d:
            nii = Path(d) / "sub-1_dir-ax_ct.nii.gz"
            nii.touch()
            for src in (nii, Path(d) / "missing.json"):
                with self.subTest(source=str(src)):
                    out = ct_bone_density(ct, vert, sub, regions=[(L1, L1)], ct_params=src)
                    self.assertFalse(out["settings"]["ct_params"]["sidecar_found"])
                    self.assertEqual(out["warnings"], [])
        self.assertIsNone(resolve_ct_params(None))


class Test_published_calibration(unittest.TestCase):
    """The 'andresen2026' calibration inverts HU = 0.33 + 1.211 * vBMD (120 kVp, QCT K2HPO4-equivalent mg/cm3)."""

    def _run(self, hu=150.0, **kw):
        ct, vert, sub = _spine({L1: hu})
        return ct_bone_density(ct, vert, sub, regions=[(L1, L1)], **kw)

    def test_inverts_the_published_equation(self):
        cal = CALIBRATIONS["andresen2026"]
        for hu in (0.33, 99.6, 150.0, 300.0):
            with self.subTest(hu=hu):
                self.assertAlmostEqual(cal.slope * hu + cal.intercept, (hu - 0.33) / 1.211, places=9)
        # the paper's own osteoporosis cut-off (mean HU <= 99.6) lies at the QCT limit of 80 mg/cm3
        self.assertAlmostEqual(cal.slope * 99.6 + cal.intercept, 82.0, delta=0.1)
        self.assertAlmostEqual(cal.sd, 6.9 / 1.211)

    def test_bmd_and_category_for_120_kvp(self):
        out = self._run(150.0, calibration="andresen2026", ct_params={"sidecar_found": True, "kvp": 120.0})
        e = out["L1-L1"]
        self.assertAlmostEqual(e["bmd_mg_cm3"], (150.0 - 0.33) / 1.211, places=2)
        self.assertEqual(e["qct_category"], "normal")
        self.assertEqual(out["per_vertebra"]["L1"]["bmd_mg_cm3"], e["bmd_mg_cm3"])
        self.assertTrue(out["settings"]["calibration_applied"])
        self.assertEqual(out["settings"]["calibration"]["kvp"], 120.0)
        self.assertEqual(out["warnings"], [])

    def test_qct_category_boundaries(self):
        for v, expected in [(79.99, "osteoporosis"), (80.0, "osteopenia"), (120.0, "osteopenia"), (120.01, "normal")]:
            with self.subTest(vbmd=v):
                self.assertEqual(qct_category(v), expected)

    def test_other_kvp_gives_no_bmd(self):
        out = self._run(150.0, calibration="andresen2026", ct_params={"sidecar_found": True, "kvp": 100.0})
        self.assertNotIn("bmd_mg_cm3", out["L1-L1"])
        self.assertNotIn("qct_category", out["L1-L1"])
        self.assertIn("mean_hu", out["L1-L1"])  # the HU themselves are always reported
        self.assertFalse(out["settings"]["calibration_applied"])
        self.assertTrue(any("only valid at 120 kVp" in w for w in out["warnings"]))

    def test_unknown_kvp_gives_bmd_with_warning(self):
        for ct_params in (None, {"sidecar_found": False}):
            with self.subTest(ct_params=ct_params):
                out = self._run(150.0, calibration="andresen2026", ct_params=ct_params)
                self.assertIn("bmd_mg_cm3", out["L1-L1"])
                self.assertTrue(any("kVp unknown" in w for w in out["warnings"]))

    def test_own_tuple_is_not_gated_by_kvp(self):
        out = self._run(150.0, calibration=(0.5, 10.0), ct_params={"sidecar_found": True, "kvp": 80.0})
        self.assertEqual(out["L1-L1"]["bmd_mg_cm3"], 85.0)
        self.assertNotIn("qct_category", out["L1-L1"])
        self.assertTrue(out["settings"]["calibration_applied"])

    def test_calibration_object_and_unknown_name(self):
        out = self._run(
            150.0, calibration=Calibration(slope=2.0, intercept=0.0, kvp=120.0), ct_params={"sidecar_found": True, "kvp": 120.0}
        )
        self.assertEqual(out["L1-L1"]["bmd_mg_cm3"], 300.0)
        with self.assertRaises(ValueError):
            self._run(150.0, calibration="does-not-exist")
        self.assertIsNone(resolve_calibration(None))


if __name__ == "__main__":
    unittest.main()
