"""Unit tests for `TPTBox.spine.spinestats.torso_vat_sat` and `body_quadrants`.

Every input is a synthetic labelled volume with a known voxel count and a known
non-isotropic zoom, so the reported areas (mm^2) and volumes (mm^3) are exactly
predictable rather than recorded from a previous run.
"""

from __future__ import annotations

import math
import unittest

import nibabel as nib
import numpy as np

from TPTBox import NII, Location, Vertebra_Instance
from TPTBox.core.vert_constants import Full_Body_Instance_Vibe as FB
from TPTBox.spine.spinestats.body_quadrants import make_quadrants
from TPTBox.spine.spinestats.torso_vat_sat import (
    VBQ_score,
    body_composition_score,
    muscle_fat_infiltration,
    peak_centered_mean,
    torso_vat_sat_muscle_mass,
)

ZOOM = (2.0, 2.0, 5.0)
VOXEL_AREA = ZOOM[0] * ZOOM[1]  # axial in-plane area, S is the last axis
VOXEL_VOLUME = ZOOM[0] * ZOOM[1] * ZOOM[2]
SHAPE = (24, 24, 40)


def _nii(arr: np.ndarray, seg: bool) -> NII:
    affine = np.diag([*ZOOM, 1.0])
    return NII(nib.Nifti1Image(arr, affine), seg=seg)


class TestPeakCenteredMean(unittest.TestCase):
    """Robust mean around the histogram mode, used to estimate CSF signal."""

    def test_rejects_the_low_signal_tail(self):
        values = np.concatenate([np.full(1000, 100.0), np.full(50, 5.0)])
        peak = peak_centered_mean(values)
        self.assertAlmostEqual(peak, 100.0, places=9)
        self.assertLess(values.mean(), 96.0)  # the plain mean is dragged down by the tail

    def test_bimodal_sample_follows_the_dominant_mode(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(100.0, 2.0, 2000), rng.normal(20.0, 2.0, 300)])
        peak = peak_centered_mean(values)
        self.assertAlmostEqual(peak, 100.0, delta=1.0)
        self.assertGreater(peak, values.mean() + 5.0)

    def test_constant_input_returns_that_constant(self):
        self.assertAlmostEqual(peak_centered_mean(np.full(100, 7.5)), 7.5, places=9)

    def test_empty_input_is_nan(self):
        self.assertTrue(math.isnan(peak_centered_mean(np.array([]))))

    def test_non_finite_values_are_dropped(self):
        values = np.array([1.0, 1.0, 1.0, np.nan, np.inf, -np.inf])
        self.assertAlmostEqual(peak_centered_mean(values), 1.0, places=9)

    def test_lower_fraction_keeps_more_of_the_tails(self):
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(100.0, 2.0, 2000), rng.normal(60.0, 2.0, 600)])
        tight = peak_centered_mean(values, peak_frac_height=1.0)
        loose = peak_centered_mean(values, peak_frac_height=0.0)
        # peak_frac_height=0 keeps every bin, i.e. the plain mean.
        self.assertAlmostEqual(loose, float(values.mean()), places=6)
        self.assertGreater(tight, loose)

    def test_input_may_be_multidimensional(self):
        self.assertAlmostEqual(peak_centered_mean(np.full((4, 5), 3.0)), 3.0, places=9)


def _torso(include_clavicula: bool = True, include_pelvis: bool = True, vat_outside_roi: int = 0):
    """Build (vibe_seg, roi) with known voxel counts per tissue.

    muscle: 2x3x4 = 24 voxels (autochthon left/right), VAT: 2x2x4 = 16, SAT: 3x2x4 = 24.
    """
    seg = np.zeros(SHAPE, dtype=np.uint8)
    seg[2:4, 2:5, 10:14] = FB.autochthon_left.value  # 24 voxels
    seg[6:8, 2:4, 10:14] = FB.inner_fat.value  # 16 voxels
    seg[10:13, 2:4, 10:14] = FB.subcutaneous_fat.value  # 24 voxels
    if include_clavicula:
        seg[0, 0, 0] = FB.clavicula_left.value
    if include_pelvis:
        seg[1, 0, 0] = FB.pelvis_left.value

    roi = np.zeros(SHAPE, dtype=np.uint8)
    roi[:, :, 8:20] = 3  # a torso ROI label inside roi_ids=(3..8)
    if vat_outside_roi:
        # VAT voxels placed outside the ROI slab must not be counted.
        seg[6:8, 6:8, 30 : 30 + vat_outside_roi] = FB.inner_fat.value
    return _nii(seg, seg=True), _nii(roi, seg=True)


class TestTorsoVatSatMuscleMass(unittest.TestCase):
    def test_volumes_are_voxel_counts_times_the_voxel_volume(self):
        seg, roi = _torso()
        out, body_comp = torso_vat_sat_muscle_mass(seg, roi)
        self.assertAlmostEqual(out["muscle_mass"], 24 * VOXEL_VOLUME, places=4)
        self.assertAlmostEqual(out["VAT"], 16 * VOXEL_VOLUME, places=4)
        self.assertAlmostEqual(out["SAT"], 24 * VOXEL_VOLUME, places=4)
        self.assertNotIn("reason", out)
        self.assertIsInstance(body_comp, NII)

    def test_tissue_outside_the_roi_is_excluded(self):
        seg, roi = _torso(vat_outside_roi=4)
        out, _ = torso_vat_sat_muscle_mass(seg, roi)
        self.assertAlmostEqual(out["VAT"], 16 * VOXEL_VOLUME, places=4)

    def test_missing_clavicula_is_reported_as_an_incomplete_torso(self):
        seg, roi = _torso(include_clavicula=False)
        out, body_comp = torso_vat_sat_muscle_mass(seg, roi)
        self.assertTrue(math.isnan(out["VAT"]))
        self.assertIn("clavicula", out["reason"])
        self.assertIsNone(body_comp)

    def test_missing_pelvis_is_reported(self):
        seg, roi = _torso(include_pelvis=False)
        out, _ = torso_vat_sat_muscle_mass(seg, roi)
        self.assertIn("pelvis", out["reason"])

    def test_return_nii_adds_the_three_masks(self):
        seg, roi = _torso()
        out, _ = torso_vat_sat_muscle_mass(seg, roi, return_nii=True)
        self.assertEqual(sorted(out["nii"]), ["SAT", "VAT", "muscle_mass"])
        self.assertEqual(int(out["nii"]["VAT"].sum()), 16)

    def test_unsupported_dataset_id_is_reported(self):
        seg, roi = _torso()
        out, _ = torso_vat_sat_muscle_mass(seg, roi, dataset_id=7)  # type: ignore[arg-type]
        self.assertIn("Unsupported dataset_id", out["reason"])

    def test_a_roi_on_a_different_grid_is_resampled(self):
        seg, roi = _torso()
        coarse = roi.rescale((4.0, 4.0, 10.0), verbose=False)
        self.assertNotEqual(coarse.shape, seg.shape)
        out, _ = torso_vat_sat_muscle_mass(seg, coarse)
        self.assertGreater(out["muscle_mass"], 0)


def _spine_volume():
    """Vert / spine / t2w triple with 4 lumbar bodies and a spinal canal column."""
    vert = np.zeros(SHAPE, dtype=np.uint8)
    spine = np.zeros(SHAPE, dtype=np.uint8)
    t2w = np.zeros(SHAPE, dtype=np.float32)
    for i, vid in enumerate((Vertebra_Instance.L1, Vertebra_Instance.L2, Vertebra_Instance.L3, Vertebra_Instance.L4)):
        z0 = 4 + i * 8
        vert[4:13, 4:13, z0 : z0 + 7] = vid.value
        spine[4:13, 4:13, z0 : z0 + 7] = Location.Vertebra_Corpus.value
        t2w[4:13, 4:13, z0 : z0 + 7] = 200.0
    # spinal canal: a column posterior to the bodies, spanning the whole volume
    spine[15:22, 4:13, :] = Location.Spinal_Canal.value
    t2w[15:22, 4:13, :] = 400.0
    return _nii(vert, seg=True), _nii(spine, seg=True), _nii(t2w, seg=False)


class TestVbqScore(unittest.TestCase):
    REGIONS = ((Vertebra_Instance.L1, Vertebra_Instance.L4),)

    def test_vbq_is_the_vertebra_to_csf_signal_ratio(self):
        vert, spine, t2w = _spine_volume()
        out = VBQ_score(t2w, vert, spine, regions=list(self.REGIONS))
        self.assertAlmostEqual(out["mean_signal_vertebra_L1-L4"], 200.0, places=4)
        self.assertAlmostEqual(out["mean_signal_liquor_L1-L4"], 400.0, places=4)
        self.assertAlmostEqual(out["VBQ_L1-L4"], 0.5, places=4)
        self.assertAlmostEqual(out["VBQ_L1-L4_old"], 0.5, places=4)

    def test_settings_are_echoed_back(self):
        vert, spine, t2w = _spine_volume()
        out = VBQ_score(t2w, vert, spine, regions=list(self.REGIONS), n_erode=1, spinal_bins=32, spinal_peak_frac_height=0.25)
        self.assertEqual(out["n_erode"], 1)
        self.assertEqual(out["spinal_bins"], 32)
        self.assertEqual(out["spinal_peak_frac_height"], 0.25)
        self.assertFalse(out["signal_by_full_cord"])

    def test_full_cord_uses_the_whole_canal(self):
        vert, spine, t2w = _spine_volume()
        out = VBQ_score(t2w, vert, spine, regions=list(self.REGIONS), full_cord=True)
        self.assertTrue(out["signal_by_full_cord"])
        self.assertAlmostEqual(out["VBQ_L1-L4"], 0.5, places=4)

    def test_a_region_without_vertebrae_yields_none(self):
        vert, spine, t2w = _spine_volume()
        out = VBQ_score(t2w, vert, spine, regions=[(Vertebra_Instance.C3, Vertebra_Instance.C6)])
        self.assertIsNone(out["VBQ_C3-C6"])
        self.assertIsNone(out["mean_signal_vertebra_C3-C6"])

    def test_the_csf_peak_mean_ignores_a_dark_border(self):
        # Darken part of the canal: the robust estimate must stay at the 400 peak while
        # the plain mean drops.
        vert, spine, t2w = _spine_volume()
        arr = t2w.get_array()
        arr[15:17, 4:13, :] = 50.0
        t2w.set_array_(arr)
        out = VBQ_score(t2w, vert, spine, regions=list(self.REGIONS))
        self.assertAlmostEqual(out["mean_signal_liquor_L1-L4"], 400.0, places=4)
        self.assertLess(out["mean_signal_liquor_L1-L4_old"], 400.0)


class TestBodyCompositionScore(unittest.TestCase):
    def _inputs(self):
        vert, spine, _ = _spine_volume()
        seg = np.zeros(SHAPE, dtype=np.uint8)
        # 6 voxels of muscle and 4 of VAT/SAT in every axial slice of the L1-L4 extent
        seg[0:3, 0:2, :] = FB.autochthon_left.value
        seg[3:5, 0:2, :] = FB.inner_fat.value
        seg[5:7, 0:2, :] = FB.subcutaneous_fat.value
        return _nii(seg, seg=True), vert, spine

    def test_areas_are_slice_voxel_counts_times_the_in_plane_voxel_area(self):
        seg, vert, spine = self._inputs()
        out = body_composition_score(seg, vert, spine, regions=[(Vertebra_Instance.L1, Vertebra_Instance.L4)])
        self.assertAlmostEqual(out["mean_muscle_area_L1-L4"], 6 * VOXEL_AREA, places=4)
        self.assertAlmostEqual(out["max_muscle_area_L1-L4"], 6 * VOXEL_AREA, places=4)
        self.assertAlmostEqual(out["mean_VAT_area_L1-L4"], 4 * VOXEL_AREA, places=4)
        self.assertAlmostEqual(out["mean_SAT_area_L1-L4"], 4 * VOXEL_AREA, places=4)

    def test_muscle_fat_ratio_matches_the_reported_areas(self):
        seg, vert, spine = self._inputs()
        out = body_composition_score(seg, vert, spine, regions=[(Vertebra_Instance.L1, Vertebra_Instance.L4)])
        expected = out["mean_muscle_area_L1-L4"] / (out["mean_VAT_area_L1-L4"] + out["mean_SAT_area_L1-L4"])
        self.assertAlmostEqual(out["muscle_fat_ratio_L1-L4"], round(expected, 4), places=4)

    def test_muscle_index_normalises_by_height_squared(self):
        seg, vert, spine = self._inputs()
        out = body_composition_score(seg, vert, spine, regions=[(Vertebra_Instance.L1, Vertebra_Instance.L4)], height_m=2.0)
        self.assertEqual(out["muscle_index_L1-L4"], round(out["mean_muscle_area_L1-L4"] / 4.0))

    def test_n_slices_counts_the_axial_slices_of_the_region(self):
        seg, vert, spine = self._inputs()
        out = body_composition_score(seg, vert, spine, regions=[(Vertebra_Instance.L1, Vertebra_Instance.L4)])
        self.assertGreater(out["n_slices_L1-L4"], 0)

    def test_region_without_vertebrae_is_skipped(self):
        seg, vert, spine = self._inputs()
        out = body_composition_score(seg, vert, spine, regions=[(Vertebra_Instance.C3, Vertebra_Instance.C6)])
        self.assertEqual(out, {})

    def test_unsupported_dataset_id_raises(self):
        seg, vert, spine = self._inputs()
        with self.assertRaises(NotImplementedError):
            body_composition_score(seg, vert, spine, dataset_id=7)  # type: ignore[arg-type]


class TestMuscleFatInfiltration(unittest.TestCase):
    def _inputs(self, fat_fraction: float):
        seg = np.zeros(SHAPE, dtype=np.uint8)
        seg[2:8, 2:8, 10:20] = FB.autochthon_left.value
        water = np.full(SHAPE, 100.0, dtype=np.float32)
        fat = np.full(SHAPE, 100.0 * fat_fraction / (1.0 - fat_fraction), dtype=np.float32)
        return _nii(water, seg=False), _nii(fat, seg=False), _nii(seg, seg=True)

    def test_fat_fraction_is_fat_over_fat_plus_water(self):
        water, fat, seg = self._inputs(0.30)
        out = muscle_fat_infiltration(water, fat, seg, per_muscle=False)
        self.assertAlmostEqual(out["mean_fat_fraction_all_all_muscle"], 0.30, places=4)
        self.assertAlmostEqual(out["median_fat_fraction_all_all_muscle"], 0.30, places=4)

    def test_uneroded_muscle_volume_is_the_voxel_count_times_the_voxel_volume(self):
        water, fat, seg = self._inputs(0.30)
        out = muscle_fat_infiltration(water, fat, seg, per_muscle=False)
        # the mask is 6 x 6 x 10 voxels
        self.assertAlmostEqual(out["muscle_volume_no_erosion_all_all_muscle"], 6 * 6 * 10 * VOXEL_VOLUME, places=4)
        # the eroded volume used for the fat statistics is strictly smaller
        self.assertLess(out["muscle_volume_all_all_muscle"], out["muscle_volume_no_erosion_all_all_muscle"])

    def test_voxels_above_the_threshold_are_all_imat(self):
        water, fat, seg = self._inputs(0.30)
        out = muscle_fat_infiltration(water, fat, seg, threshold=0.20, per_muscle=False)
        self.assertAlmostEqual(out["IMAT_fraction_all_all_muscle"], 1.0, places=4)
        self.assertAlmostEqual(out["lean_muscle_volume_all_all_muscle"], 0.0, places=4)
        self.assertAlmostEqual(out["IMAT_volume_all_all_muscle"], out["muscle_volume_all_all_muscle"], places=4)

    def test_voxels_below_the_threshold_are_all_lean(self):
        water, fat, seg = self._inputs(0.05)
        out = muscle_fat_infiltration(water, fat, seg, threshold=0.20, per_muscle=False)
        self.assertAlmostEqual(out["IMAT_fraction_all_all_muscle"], 0.0, places=4)
        self.assertAlmostEqual(out["IMAT_volume_all_all_muscle"], 0.0, places=4)
        self.assertAlmostEqual(out["lean_muscle_volume_all_all_muscle"], out["muscle_volume_all_all_muscle"], places=4)

    def test_threshold_and_erode_settings_are_echoed_back(self):
        water, fat, seg = self._inputs(0.30)
        out = muscle_fat_infiltration(water, fat, seg, threshold=0.25, per_muscle=False)
        self.assertEqual(out["threshold"], 0.25)
        self.assertIn("all_muscle", out["erode"])

    def test_per_muscle_reports_each_group(self):
        water, fat, seg = self._inputs(0.30)
        out = muscle_fat_infiltration(water, fat, seg, per_muscle=True)
        self.assertIn("mean_fat_fraction_all_autochthon_left", out)
        self.assertIn("mean_fat_fraction_all_all_muscle", out)
        # only autochthon_left is populated, the other groups stay empty
        self.assertAlmostEqual(out["muscle_volume_no_erosion_all_autochthon_left"], 6 * 6 * 10 * VOXEL_VOLUME, places=4)

    def test_regions_require_a_vert_segmentation(self):
        water, fat, seg = self._inputs(0.30)
        with self.assertRaises(ValueError):
            muscle_fat_infiltration(water, fat, seg, regions=[(Vertebra_Instance.L1, Vertebra_Instance.L4)])

    def test_unsupported_dataset_id_raises(self):
        water, fat, seg = self._inputs(0.30)
        with self.assertRaises(NotImplementedError):
            muscle_fat_infiltration(water, fat, seg, dataset_id=7)  # type: ignore[arg-type]


class TestMakeQuadrants(unittest.TestCase):
    """Subdivides a vertebral body into 27 vertebra-local regions.

    Needs real vertebra subregions (the local axes come from POIs computed on them),
    so this runs on the shipped single-vertebra sample CT.
    """

    @classmethod
    def setUpClass(cls):
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        _, cls.subreg, cls.vert, cls.label = get_test_ct()

    def test_quadrants_are_a_subset_of_the_vertebral_body(self):
        quad = make_quadrants(self.vert, self.subreg, vert_ids=[self.label])
        self.assertIsInstance(quad, NII)
        out = quad.get_array() != 0
        self.assertGreater(out.sum(), 0)
        body = (self.vert.get_array() == self.label) & np.isin(self.subreg.get_array(), (49, 50, 52))
        self.assertTrue(np.all(body[out]), "quadrant voxels must come from the body mask")

    def test_labels_are_in_one_to_twentyseven_and_disjoint(self):
        quad = make_quadrants(self.vert, self.subreg, vert_ids=[self.label]).get_array()
        labels = [int(label) for label in np.unique(quad) if label != 0]
        self.assertGreater(len(labels), 1)
        self.assertTrue(all(1 <= label <= 27 for label in labels), labels)
        total = sum(int((quad == label).sum()) for label in labels)
        self.assertEqual(total, int((quad != 0).sum()), "labels must partition the output")

    def test_erode_shrinks_the_result(self):
        plain = int((make_quadrants(self.vert, self.subreg, vert_ids=[self.label]).get_array() != 0).sum())
        eroded = int((make_quadrants(self.vert, self.subreg, vert_ids=[self.label], erode=1).get_array() != 0).sum())
        self.assertLess(eroded, plain)


if __name__ == "__main__":
    unittest.main()
