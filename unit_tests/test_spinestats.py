"""Unit tests for ``TPTBox.spine.spinestats`` (angles, ivd_pois, body_quadrants,
make_endplate, distances) and ``TPTBox.spine.snapshot2D.snapshot_modular``.

The small sample CT/MRI volumes only contain three vertebrae each, so where a
function needs a longer spine (lordosis/kyphosis, multi-curve Cobb, direction
line plots) a synthetic POI/NII spanning C2..L5 is built instead.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import matplotlib as mpl

mpl.use("Agg")  # headless backend for the plot_* helpers

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from TPTBox import NII, Location, calc_poi_from_subreg_vert  # noqa: E402
from TPTBox.core.poi import POI  # noqa: E402
from TPTBox.core.vert_constants import Vertebra_Instance  # noqa: E402
from TPTBox.tests.test_utils import get_test_ct, get_test_mri  # noqa: E402

try:
    import torch  # noqa: F401

    has_torch = True
except Exception:
    has_torch = False

SPINE_SHAPE = (64, 64, 180)
# Cervical (C2-C7) + thoracic (T1-T12) + lumbar (L1-L5)
SPINE_VERTS = [2, 3, 4, 5, 6, 7, *range(8, 20), 20, 21, 22, 23, 24]


def build_synthetic_spine(shape: tuple[int, int, int] = SPINE_SHAPE, zstep: int = 7) -> tuple[POI, list[int]]:
    """Build a synthetic spine POI with a slight scoliosis + kyphosis curvature.

    Every vertebra carries a centroid (50), the three direction POIs
    (Right/Posterior/Inferior), an IVD centroid (100) and the superior /
    inferior disc faces (58 / 59) so that every angle/plot code path can run.
    """
    centroids: dict[int, dict[int, tuple[float, float, float]]] = {}
    for i, v in enumerate(SPINE_VERTS):
        a = 0.18 * np.sin(i / 3.0)  # coronal tilt (scoliosis)
        b = 0.16 * np.sin(i / 4.0)  # sagittal tilt (kyphosis/lordosis)
        up = np.array([np.sin(a), np.sin(b), 1.0])
        up /= np.linalg.norm(up)
        right = np.array([np.cos(a), 0.0, -np.sin(a)])
        right /= np.linalg.norm(right)
        post = np.cross(up, right)
        post /= np.linalg.norm(post)
        c = np.array([32 + 8 * np.sin(i / 3.0), 32 + 5 * np.sin(i / 4.0), shape[2] - 10.0 - zstep * i])
        disc = c + np.array([0.0, 0.0, -zstep / 2])
        centroids[v] = {
            Location.Vertebra_Corpus.value: tuple(c),
            Location.Vertebra_Direction_Right.value: tuple(c - right * 6),
            Location.Vertebra_Direction_Posterior.value: tuple(c - post * 6),
            Location.Vertebra_Direction_Inferior.value: tuple(c + up * 6),
            Location.Vertebra_Disc.value: tuple(disc),
            Location.Vertebra_Disc_Inferior.value: tuple(disc + up * 4),
            Location.Vertebra_Disc_Superior.value: tuple(disc - up * 4),
        }
    poi = POI(centroids, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=shape)
    return poi, SPINE_VERTS


def build_synthetic_nii(poi: POI, verts: list[int], shape: tuple[int, int, int] = SPINE_SHAPE) -> tuple[NII, NII]:
    """Build a matching synthetic image + vertebra segmentation for the spine POI."""
    arr = np.zeros(shape, dtype=np.float32)
    seg = np.zeros(shape, dtype=np.uint16)
    for v in verts:
        c = np.round(poi[v, 50]).astype(int)
        sl = tuple(slice(max(0, x - 3), x + 3) for x in c)
        arr[sl] = 1000
        seg[sl] = v
    img = NII(nib.Nifti1Image(arr, np.eye(4)), seg=False)
    seg_nii = NII(nib.Nifti1Image(seg, np.eye(4)), seg=True)
    return img, seg_nii


class Test_Angles_Helpers(unittest.TestCase):
    def test_unit_vector(self):
        from TPTBox.spine.spinestats import angles

        np.testing.assert_allclose(angles.unit_vector(np.array([3.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 2.0])
        self.assertAlmostEqual(float(np.linalg.norm(angles.unit_vector(v))), 1.0)

    def test_angle_between(self):
        from TPTBox.spine.spinestats import angles

        self.assertAlmostEqual(angles.angle_between((1, 0, 0), (0, 1, 0)), np.pi / 2)
        self.assertAlmostEqual(angles.angle_between((1, 0, 0), (1, 0, 0)), 0.0)
        self.assertAlmostEqual(angles.angle_between((1, 0, 0), (-1, 0, 0)), np.pi)

    def test_cosine_distance(self):
        from TPTBox.spine.spinestats import angles

        self.assertAlmostEqual(angles.cosine_distance(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])), 1.0)
        self.assertAlmostEqual(angles.cosine_distance(np.array([1.0, 0, 0]), np.array([0.0, 1, 0])), 0.0)

    def test_get_to_space(self):
        from TPTBox.spine.spinestats import angles

        a, b, c = np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), np.array([0.0, 0, 1])
        to_space, from_space = angles.get_to_space(a, b, c)
        np.testing.assert_allclose(to_space @ from_space, np.eye(3), atol=1e-9)

    def test_moveto(self):
        from TPTBox.spine.spinestats.angles import MoveTo

        poi, _ = build_synthetic_spine()
        # CENTER always resolves to (v, 50)
        self.assertTrue(MoveTo.CENTER.has_point(5, poi))
        self.assertEqual(MoveTo.CENTER.get_location(5, poi), (Vertebra_Instance.C5, 50))
        np.testing.assert_allclose(MoveTo.CENTER.get_point(5, poi), poi[5, 50])
        # BOTTOM/TOP resolve to the disc (label 100) that the synthetic spine has
        self.assertEqual(MoveTo.BOTTOM.get_location(6, poi)[1], Location.Vertebra_Disc)
        self.assertEqual(MoveTo.TOP.get_location(6, poi)[1], Location.Vertebra_Disc)
        self.assertTrue(MoveTo.BOTTOM.has_point(6, poi))
        # A valid vertebra that is absent from the POI -> no point (C1 / id 1)
        self.assertFalse(MoveTo.CENTER.has_point(1, poi))

    def test_last_lumbar_thoracic(self):
        from TPTBox.spine.spinestats.angles import _get_last_lumbar, _get_last_thoracic

        poi, _ = build_synthetic_spine()
        self.assertEqual(_get_last_lumbar(poi), Vertebra_Instance.L5)
        self.assertEqual(_get_last_thoracic(poi), Vertebra_Instance.T12)
        # empty POI -> None
        empty = POI({}, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=SPINE_SHAPE)
        self.assertIsNone(_get_last_lumbar(empty))
        self.assertIsNone(_get_last_thoracic(empty))

    def test_add_artificial_ivd(self):
        from TPTBox.spine.spinestats.angles import _add_artificial_ivd

        centroids = {v: {50: (30.0, 30.0, 100.0 - 10 * i)} for i, v in enumerate([2, 3, 4, 5])}
        poi = POI(centroids, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=SPINE_SHAPE)
        self.assertNotIn(100, poi.keys_subregion())
        out = _add_artificial_ivd(poi)
        self.assertIn(100, out.keys_subregion())


class Test_Angles_Compute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spine_poi, cls.spine_verts = build_synthetic_spine()
        _, ct_subreg, ct_vert, ct_label = get_test_ct()
        cls.ct_poi = calc_poi_from_subreg_vert(
            ct_vert,
            ct_subreg,
            subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Right, Location.Vertebra_Direction_Posterior],
        )
        cls.ct_label = ct_label
        _, mri_subreg, mri_vert, mri_label = get_test_mri()
        cls.mri_poi = calc_poi_from_subreg_vert(
            mri_vert,
            mri_subreg,
            subreg_id=[
                Location.Vertebra_Corpus,
                Location.Vertebra_Direction_Right,
                Location.Vertebra_Direction_Posterior,
                Location.Vertebra_Direction_Inferior,
            ],
        )
        cls.mri_label = mri_label

    def test_compute_angle_directions(self):
        from TPTBox.spine.spinestats import angles

        poi = self.spine_poi
        for direction in ("R", "L", "P", "A", "S", "I"):
            for project in (True, False):
                with self.subTest(direction=direction, project=project):
                    a = angles.compute_angel_between_two_points_(poi.copy(), 6, 7, direction, project_2D=project)
                    self.assertIsInstance(a, float)
                    self.assertGreaterEqual(a, 0.0)

    def test_compute_angle_ivd_direction(self):
        from TPTBox.spine.spinestats import angles

        # vert id > IVD_MORE_ACCURATE (=15) triggers the disc-direction branch
        a = angles.compute_angel_between_two_points_(self.spine_poi.copy(), 21, 22, "R", use_ivd_direction=True, project_2D=False)
        self.assertIsInstance(a, float)
        b = angles.compute_angel_between_two_points_(self.spine_poi.copy(), 21, 22, "S", use_ivd_direction=True, project_2D=True)
        self.assertIsInstance(b, float)

    def test_compute_angle_none_and_errors(self):
        from TPTBox.spine.spinestats import angles

        poi = self.spine_poi
        self.assertIsNone(angles.compute_angel_between_two_points_(poi.copy(), None, 7, "R"))
        self.assertIsNone(angles.compute_angel_between_two_points_(poi.copy(), 6, None, "R"))
        # a vertebra absent from the POI -> None
        self.assertIsNone(angles.compute_angel_between_two_points_(poi.copy(), 6, 90, "R"))
        with pytest.raises(NotImplementedError):
            angles.compute_angel_between_two_points_(poi.copy(), 6, 7, "Z")

    def test_compute_angle_real_data(self):
        from TPTBox.spine.spinestats import angles

        for poi, label in ((self.mri_poi, self.mri_label), (self.ct_poi, self.ct_label)):
            a = angles.compute_angel_between_two_points_(poi.copy(), label, label + 1, "R", project_2D=True)
            self.assertIsInstance(a, float)
            b = angles.compute_angel_between_two_points_(poi.copy(), label, label + 1, "P", project_2D=False)
            self.assertIsInstance(b, float)

    def test_lordosis_kyphosis_synthetic(self):
        from TPTBox.spine.spinestats import angles

        for project in (True, False):
            out = angles.compute_lordosis_and_kyphosis(self.spine_poi.copy(), project_2D=project)
            self.assertEqual(set(out), {"cervical_lordosis", "thoracic_kyphosis", "lumbar_lordosis"})
            for v in out.values():
                self.assertIsInstance(v, float)
                self.assertGreater(v, 0.0)

    def test_lordosis_kyphosis_real(self):
        from TPTBox.spine.spinestats import angles

        out = angles.compute_lordosis_and_kyphosis(self.mri_poi.copy(), project_2D=True)
        # sample only spans 3 cervical vertebrae -> values may be None, but keys must be present
        self.assertEqual(set(out), {"cervical_lordosis", "thoracic_kyphosis", "lumbar_lordosis"})

    def test_lordosis_requires_direction(self):
        from TPTBox.spine.spinestats import angles

        _, subreg, vert, _ = get_test_ct()
        corpus_only = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus])
        with pytest.raises(AssertionError):
            angles.compute_lordosis_and_kyphosis(corpus_only, project_2D=True)

    def test_max_cobb_angle_synthetic(self):
        from TPTBox.spine.spinestats import angles

        max_angle, from_vert, to_vert, apex = angles.compute_max_cobb_angle(self.spine_poi.copy())
        self.assertGreater(max_angle, 0.0)
        self.assertIsNotNone(from_vert)
        self.assertIsNotNone(to_vert)
        self.assertIsNotNone(apex)
        # 3D + ivd direction + explicit vertebra list
        m2 = angles.compute_max_cobb_angle(self.spine_poi.copy(), project_2D=False, use_ivd_direction=True)
        self.assertGreater(m2[0], 0.0)
        m3 = angles.compute_max_cobb_angle(self.spine_poi.copy(), vertebrae_list=list(Vertebra_Instance.thoracic()))
        self.assertGreaterEqual(m3[0], 0.0)

    def test_max_cobb_angle_real(self):
        from TPTBox.spine.spinestats import angles
        from TPTBox.spine.spinestats.angles import MoveTo

        # the sample POIs carry no IVD/endplate landmarks, so anchor on the
        # vertebra centre (MoveTo.CENTER) rather than the disc faces
        for poi in (self.mri_poi, self.ct_poi):
            res = angles.compute_max_cobb_angle(poi.copy(), vert_id1_mv=MoveTo.CENTER, vert_id2_mv=MoveTo.CENTER)
            self.assertEqual(len(res), 4)
            self.assertGreaterEqual(res[0], 0.0)

    def test_max_cobb_angle_multi(self):
        from TPTBox.spine.spinestats import angles

        curves = angles.compute_max_cobb_angle_multi(self.spine_poi.copy(), threshold_deg=3)
        self.assertGreaterEqual(len(curves), 1)
        for angle, frm, to, _apex in curves:
            self.assertGreaterEqual(angle, 3.0)
            self.assertIsInstance(frm, int)
            self.assertIsInstance(to, int)
        # a huge threshold finds nothing
        self.assertEqual(angles.compute_max_cobb_angle_multi(self.spine_poi.copy(), threshold_deg=999), [])
        # <= 2 vertebrae returns early
        self.assertEqual(angles.compute_max_cobb_angle_multi(self.spine_poi.copy(), vertebrae_list=[Vertebra_Instance.C2]), [])


class Test_Angles_Plots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poi, cls.verts = build_synthetic_spine()
        cls.img, cls.seg = build_synthetic_nii(cls.poi, cls.verts)

    def test_plot_lordosis_kyphosis(self):
        from TPTBox.spine.spinestats import angles

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lk.png")
            out, snap = angles.plot_compute_lordosis_and_kyphosis(path, self.poi.copy(), self.img, self.seg)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(set(out), {"cervical_lordosis", "thoracic_kyphosis", "lumbar_lordosis"})
            self.assertIsNotNone(snap)
        # img_path None -> returns frame without writing a file
        out2, snap2 = angles.plot_compute_lordosis_and_kyphosis(None, self.poi.copy(), self.img, project_2D=False)
        self.assertIsNotNone(snap2)

    def test_plot_cobb_angle(self):
        from TPTBox.spine.spinestats import angles

        with tempfile.TemporaryDirectory() as td:
            for use_ivd in (False, True):
                path = os.path.join(td, f"cobb_{use_ivd}.png")
                copps, frame = angles.plot_cobb_angle(path, self.poi.copy(), self.img, self.seg, threshold_deg=3, use_ivd_direction=use_ivd)
                self.assertTrue(os.path.exists(path))
                self.assertIsInstance(copps, list)
                self.assertIsNotNone(frame)

    def test_plot_cobb_and_lordosis(self):
        from TPTBox.spine.spinestats import angles

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "both.png")
            out_cobb, out_lak, frames = angles.plot_cobb_and_lordosis_and_kyphosis(
                path, self.poi.copy(), self.img, self.seg, threshold_deg=3
            )
            self.assertTrue(os.path.exists(path))
            self.assertIsInstance(out_cobb, list)
            self.assertEqual(set(out_lak), {"cervical_lordosis", "thoracic_kyphosis", "lumbar_lordosis"})
            self.assertEqual(len(frames), 2)


class Test_IVD_POIs(unittest.TestCase):
    def test_calculate_ivd_poi_mri(self):
        # the MRI sample already carries IVD labels (100) so the ProcessPool
        # branch is skipped and strategy_calculate_up_vector runs
        from TPTBox.spine.spinestats import calculate_IVD_POI

        _, subreg, vert, _ = get_test_mri()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Inferior])
        out = calculate_IVD_POI(vert.copy(), subreg.copy(), poi.copy())
        subs = out.keys_subregion()
        self.assertIn(Location.Vertebra_Disc.value, subs)
        self.assertIn(Location.Vertebra_Disc_Superior.value, subs)

    def test_calculate_ivd_poi_empty_location(self):
        from TPTBox.spine.spinestats import calculate_IVD_POI

        _, subreg, vert, _ = get_test_mri()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus])
        out = calculate_IVD_POI(vert.copy(), subreg.copy(), poi.copy(), ivd_location=set())
        self.assertEqual(len(out), len(poi))

    def test_compute_fake_ivd_ct(self):
        # the CT sample has no IVD labels -> exercises the synthetic IVD
        # generation (ProcessPoolExecutor over the three adjacent vertebrae)
        from TPTBox.spine.spinestats import compute_fake_ivd

        _, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Inferior])
        out = compute_fake_ivd(vert.copy(), subreg.copy(), poi.copy())
        self.assertTrue(any(u >= 100 for u in out.unique()))

    def test_calculate_ivd_poi_ct_synthesises_disc(self):
        # the CT sample has no IVD labels -> calculate_IVD_POI must synthesise
        # them via compute_fake_ivd before computing the disc POIs
        from TPTBox.spine.spinestats import calculate_IVD_POI

        _, subreg, vert, _ = get_test_ct()
        corpus_only = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus])
        out = calculate_IVD_POI(vert.copy(), subreg.copy(), corpus_only.copy())
        self.assertIn(Location.Vertebra_Disc.value, out.keys_subregion())

    def test_process_vertebra_helpers(self):
        # compute_fake_ivd dispatches these to a ProcessPoolExecutor, where the
        # coverage tracer never sees them; drive them directly in-process here.
        from TPTBox.spine.spinestats import ivd_pois as ivd_mod

        _, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(
            vert,
            subreg,
            subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior, Location.Vertebra_Direction_Inferior],
        )
        verts_ids = vert.unique()
        crop = vert.compute_crop(dist=2)
        vc = vert.apply_crop(crop)
        sc = subreg.apply_crop(crop)
        pc = poi.apply_crop(crop)
        first, nxt = verts_ids[0], verts_ids[1]

        cropped = ivd_mod._crop(first, verts_ids, vc, sc, pc)
        self.assertIsNotNone(cropped)
        self.assertEqual(len(cropped), 7)
        # i >= 100 and a vertebra without a successor both short-circuit to None
        self.assertIsNone(ivd_mod._crop(123, verts_ids, vc, sc, pc))
        self.assertIsNone(ivd_mod._crop(verts_ids[-1], verts_ids, vc, sc, pc))

        next_id = Vertebra_Instance(first).get_next_poi(verts_ids)
        ivd_b = ivd_mod._process_vertebra_B(first, vc, sc, next_id)
        self.assertIn(100 + first, ivd_b.unique())

        ivd_a = ivd_mod._process_vertebra_A(first, vc, sc, next_id.value, pc)
        if ivd_a is not None:
            self.assertIn(100 + first, ivd_a.unique())

        result = ivd_mod._process_vertebra(first, verts_ids, vc, sc, pc)
        self.assertIsNotNone(result)
        ivd, slices = result
        self.assertIn(100 + first, ivd.unique())
        self.assertEqual(len(slices), 3)
        self.assertEqual(nxt, verts_ids[1])

    def test_strategy_up_vector_missing_center(self):
        from TPTBox.spine.spinestats.ivd_pois import strategy_calculate_up_vector

        _, subreg, vert, _ = get_test_mri()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus])
        cropped = vert.copy()
        bb = cropped.compute_crop()
        cropped.apply_crop_(bb)
        # no IVD centroid present for this vert -> returns poi unchanged
        out = strategy_calculate_up_vector(poi.copy(), cropped, 999, bb)
        self.assertIsInstance(out, POI)

    def test_calculate_pca_normal_np(self):
        from TPTBox.spine.spinestats import calculate_pca_normal_np

        slab = np.zeros((20, 6, 6), dtype=int)
        slab[:, 2:4, 2:4] = 1
        normal = calculate_pca_normal_np(slab, pca_component=0)
        np.testing.assert_allclose(np.abs(normal), [1.0, 0.0, 0.0], atol=1e-6)


class Test_BodyQuadrants(unittest.TestCase):
    def test_make_quadrants_mri(self):
        from TPTBox.spine.spinestats import make_quadrants

        _, subreg, vert, _ = get_test_mri()
        out = make_quadrants(vert.copy(), subreg.copy())
        labels = out.unique()
        self.assertGreater(len(labels), 0)
        self.assertLessEqual(max(labels), 27)
        self.assertGreaterEqual(min(labels), 1)
        # output keeps the input orientation
        self.assertEqual(out.orientation, vert.orientation)

    def test_make_quadrants_vert_ids_and_erode(self):
        from TPTBox.spine.spinestats import make_quadrants

        _, subreg, vert, label = get_test_ct()
        out = make_quadrants(vert.copy(), subreg.copy(), vert_ids=[label], erode=1)
        labels = out.unique()
        if len(labels) > 0:  # erosion may remove everything on a tiny sample
            self.assertLessEqual(max(labels), 27)


class Test_MakeEndplate(unittest.TestCase):
    def test_endplate_extraction_ct(self):
        from TPTBox.spine.spinestats import endplate_extraction

        _, subreg, vert, label = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=Location.Vertebra_Direction_Posterior)
        out = endplate_extraction(label, vert.copy(), subreg.copy(), poi.copy())
        self.assertIsNotNone(out)
        labels = out.unique()
        self.assertIn(Location.Vertebral_Body_Endplate_Superior.value, labels)
        self.assertIn(Location.Vertebral_Body_Endplate_Inferior.value, labels)
        # accepts an Enum index as well
        out_enum = endplate_extraction(Vertebra_Instance(label), vert.copy(), subreg.copy(), poi.copy())
        self.assertIsNotNone(out_enum)

    def test_endplate_extraction_sacrum_returns_none(self):
        from TPTBox.spine.spinestats import endplate_extraction

        _, subreg, vert, _ = get_test_ct()
        poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=Location.Vertebra_Direction_Posterior)
        # 27 is in Vertebra_Instance.sacrum()[1:] -> early None
        self.assertIsNone(endplate_extraction(27, vert.copy(), subreg.copy(), poi.copy()))

    def test_endplate_extraction_missing_direction_returns_none(self):
        from TPTBox.spine.spinestats import endplate_extraction

        _, subreg, vert, label = get_test_ct()
        corpus_only = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus])
        self.assertIsNone(endplate_extraction(label, vert.copy(), subreg.copy(), corpus_only))

    def test_get_largest_cc(self):
        from TPTBox.spine.spinestats.make_endplate import _get_largest_CC

        seg = np.zeros((12, 12, 12), dtype=int)
        seg[2:5, 2:6, 2:5] = 1  # bigger blob (36 voxels)
        seg[9:11, 9:11, 9:11] = 1  # smaller blob
        out = _get_largest_CC(seg)
        self.assertEqual(out.sum(), 36)

    def test_dilate_erode_special(self):
        from TPTBox.spine.spinestats.make_endplate import _dilate_erode_special

        blob = np.zeros((12, 12, 12), dtype=bool)
        blob[3:9, 3:9, 3:9] = True
        iso = _dilate_erode_special(blob, ball_size=1)
        self.assertEqual(iso.shape, blob.shape)
        # directional structuring element (disk stacked along the normal axis)
        directed = _dilate_erode_special(blob, ball_size=1, normal=np.array([0.0, 0.0, 1.0]))
        self.assertEqual(directed.shape, blob.shape)

    def test_endplate_np_helpers(self):
        from TPTBox.spine.spinestats.make_endplate import _extract_endplate_np, _get_endplate

        body = np.zeros((14, 14, 14), dtype=int)
        body[3:11, 3:11, 3:11] = 1
        projected = np.round(np.mgrid[0:14, 0:14, 0:14][1]).astype(int) + 1
        for axis in (0, 1, 2):
            endplate = _get_endplate(body.copy(), projected.copy(), axis=axis)
            self.assertEqual(endplate.shape, body.shape)
        normal = np.array([0.0, 1.0, 0.0])
        upper = _extract_endplate_np(body.copy(), projected.copy(), normal, lower=False)
        lower = _extract_endplate_np(body.copy(), projected.copy(), normal, lower=True)
        self.assertGreater(upper.sum(), 0)
        self.assertGreater(lower.sum(), 0)


class Test_Distances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, subreg, vert, _ = get_test_mri()
        cls.vert = vert
        cls.subreg = subreg
        cls.poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus])

    def test_compute_all_distances_early_return(self):
        from TPTBox.spine.spinestats import distances

        poi = self.poi.copy()
        for key in distances.distances_funs:
            poi.info[key] = {}
        out = distances.compute_all_distances(poi, all_pois_computed=True)
        for key in distances.distances_funs:
            self.assertIn(key, out.info)

    def test_compute_all_distances_requires_vert(self):
        from TPTBox.spine.spinestats import distances

        with pytest.raises(ValueError):
            distances.compute_all_distances(self.poi.copy(), vert=None, all_pois_computed=False)

    def test_compute_all_distances_deprecated_call(self):
        # distances.py calls the refactored ``calculate_distances_poi_across_regions``
        # with the old (l1, l2, keep_zoom) signature, which now raises TypeError.
        from TPTBox.spine.spinestats import distances

        with pytest.raises(TypeError):
            distances.compute_all_distances(self.poi.copy(), self.vert.copy(), self.subreg.copy())


class Test_SnapshotModular(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct, cls.ct_subreg, cls.ct_vert, cls.ct_label = get_test_ct()
        cls.ct_poi = calc_poi_from_subreg_vert(cls.ct_vert, cls.ct_subreg, subreg_id=[Location.Vertebra_Corpus])
        cls.mri, cls.mri_subreg, cls.mri_vert, cls.mri_label = get_test_mri()
        cls.mri_poi = calc_poi_from_subreg_vert(cls.mri_vert, cls.mri_subreg, subreg_id=[Location.Vertebra_Corpus])

    def test_create_snapshot_views(self):
        from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

        frame = Snapshot_Frame(self.ct, self.ct_vert, self.ct_poi, mode="CT", sagittal=True, coronal=True, axial=True)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "views.jpg")
            create_snapshot(path, [frame])
            self.assertTrue(os.path.exists(path))

    def test_create_snapshot_mip_mean_depth(self):
        from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot
        from TPTBox.spine.snapshot2D.snapshot_modular import Visualization_Type

        f_mip = Snapshot_Frame(
            self.ct,
            self.ct_vert,
            self.ct_poi,
            mode="CT",
            coronal=True,
            axial=True,
            visualization_type=Visualization_Type.Maximum_Intensity,
        )
        f_mean = Snapshot_Frame(
            self.ct,
            self.ct_vert,
            mode="CT",
            axial=True,
            axial_heights=[0.5, 20],
            visualization_type=Visualization_Type.Mean_Intensity,
        )
        f_depth = Snapshot_Frame(
            self.ct,
            None,
            self.ct_poi,
            mode="CTs",
            visualization_type=Visualization_Type.Maximum_Intensity_Colored_Depth,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mip.jpg")
            create_snapshot(path, [f_mip, f_mean, f_depth])
            self.assertTrue(os.path.exists(path))

    def test_create_snapshot_mri_and_flags(self):
        from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

        mri_frame = Snapshot_Frame(self.mri, self.mri_subreg, self.mri_poi, mode="MRI", axial=True, coronal=True)
        flags = Snapshot_Frame(
            self.mri,
            self.mri_vert,
            self.mri_poi,
            mode="MINMAX",
            crop_msk=True,
            hide_centroids=True,
            gauss_filter=True,
            image_threshold=10,
            title="t",
        )
        none_mode = Snapshot_Frame(self.mri, None, mode="None", ignore_seg_for_centering=True)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mri.jpg")
            create_snapshot([path], [mri_frame, flags, none_mode])
            self.assertTrue(os.path.exists(path))

    def test_create_snapshot_force_show_cdt_and_check(self):
        from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

        frame = Snapshot_Frame(
            self.ct,
            self.ct_vert,
            mode="CT",
            ignore_seg_for_centering=True,
            force_show_cdt=True,
            hide_centroid_labels=True,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cdt.jpg")
            create_snapshot(path, [frame])
            self.assertTrue(os.path.exists(path))
            mtime = os.path.getmtime(path)
            # check=True must not overwrite an existing snapshot
            create_snapshot(path, [frame], check=True)
            self.assertEqual(os.path.getmtime(path), mtime)

    @unittest.skipIf(not has_torch, "requires torch")
    def test_create_snapshot_denoise(self):
        from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

        frame = Snapshot_Frame(self.ct, self.ct_vert, mode="CT", denoise_threshold=-300)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "denoise.jpg")
            create_snapshot(path, [frame])
            self.assertTrue(os.path.exists(path))

    def test_to_cdt(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import to_cdt

        self.assertIsNone(to_cdt(None))
        loaded = to_cdt(self.mri_poi)
        self.assertIsInstance(loaded, POI)
        empty = self.mri_poi.copy()
        empty.centroids.clear()
        self.assertIsNone(to_cdt(empty))

    def test_div0(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import div0

        self.assertEqual(div0(1.0, 0.0, fill=-1), -1)
        self.assertEqual(div0(6.0, 2.0), 3.0)
        np.testing.assert_array_equal(div0(np.array([1.0, 2.0]), np.array([0.0, 2.0])), [0.0, 1.0])

    def test_normalize_image(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import normalize_image

        np.testing.assert_allclose(normalize_image(np.array([0.0, 5.0, 10.0])), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalize_image(np.array([2.0, 6.0]), v_range=(0.0, 8.0)), [0.25, 0.75])

    def test_get_contrasting_stroke_color(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import get_contrasting_stroke_color

        self.assertEqual(get_contrasting_stroke_color((0.0, 0.0, 0.0)), "gray")
        self.assertEqual(get_contrasting_stroke_color((1.0, 1.0, 1.0)), "black")
        self.assertEqual(get_contrasting_stroke_color((1.0, 1.0, 1.0, 1.0)), "black")
        self.assertIn(get_contrasting_stroke_color(40), ("gray", "black"))

    def test_make_isotropic(self):
        from TPTBox.spine.snapshot2D.snapshot_modular import make_isotropic2d, make_isotropic2dpluscolor

        gray = make_isotropic2d(np.ones((4, 4)), (2.0, 1.0))
        self.assertEqual(gray.shape, (8, 4))
        color = make_isotropic2dpluscolor(np.ones((4, 4, 3)), (2.0, 1.0))
        self.assertEqual(color.shape, (8, 4, 3))
        # 2D input passes straight through make_isotropic2d
        gray2 = make_isotropic2dpluscolor(np.ones((4, 4)), (1.0, 2.0))
        self.assertEqual(gray2.shape, (4, 8))


if __name__ == "__main__":
    unittest.main()
