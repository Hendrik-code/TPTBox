"""Unit tests for `TPTBox.spine.snapshot2D`.

The pure array/colour helpers are tested directly, and `create_snapshot` plus the
`snapshot_templates` entry points are rendered to a temp directory and checked for
a non-blank PNG of the expected size. `unit_tests/test_snapshot_mip.py` already
covers the slab projection functions.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from TPTBox import NII, Location  # noqa: E402

# `create_snapshot` imports torch for a frame with `denoise_threshold` set, which the
# mip_shot template does - so that one template needs torch even though it is optional.
HAS_TORCH = importlib.util.find_spec("torch") is not None
from TPTBox.spine.snapshot2D.snapshot_modular import (  # noqa: E402
    Snapshot_Frame,
    create_snapshot,
    div0,
    get_contrasting_stroke_color,
    make_isotropic2d,
    make_isotropic2dpluscolor,
    normalize_image,
    to_cdt,
)


def _nii(arr, seg=True, zoom=(1.0, 1.0, 1.0)) -> NII:
    return NII(nib.Nifti1Image(arr, np.diag([*zoom, 1.0])), seg=seg)


class TestNormalizeImage(unittest.TestCase):
    def test_maps_to_the_unit_range(self):
        arr = np.array([[-100.0, 0.0], [100.0, 500.0]])
        out = normalize_image(arr)
        self.assertAlmostEqual(float(out.min()), 0.0, places=9)
        self.assertAlmostEqual(float(out.max()), 1.0, places=9)

    def test_is_linear(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = normalize_image(arr)
        np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-9)

    def test_explicit_range_rescales_but_does_not_clip(self):
        # The docstring calls v_range a "clip range", but the implementation only
        # rescales: values outside it land outside [0, 1]. Pinned as-is.
        arr = np.array([-50.0, 0.0, 50.0, 100.0, 150.0])
        out = normalize_image(arr, v_range=(0.0, 100.0))
        self.assertAlmostEqual(float(out[1]), 0.0, places=9)
        self.assertAlmostEqual(float(out[3]), 1.0, places=9)
        self.assertLess(float(out[0]), 0.0)
        self.assertGreater(float(out[4]), 1.0)

    def test_a_constant_image_yields_nan(self):
        # min == max, so the division is 0/0. Documented limitation, not a crash.
        out = normalize_image(np.full((4, 4), 7.0))
        self.assertTrue(np.isnan(out).all())


class TestMakeIsotropic(unittest.TestCase):
    def test_anisotropic_input_is_stretched(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[4:6, 4:6] = 1
        out = make_isotropic2d(arr, (1.0, 3.0))
        self.assertEqual(out.shape[0], 10)
        self.assertGreater(out.shape[1], 10, "the 3 mm axis must be upsampled")

    def test_isotropic_input_is_unchanged_in_shape(self):
        arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
        self.assertEqual(make_isotropic2d(arr, (1.0, 1.0)).shape, arr.shape)

    def test_mask_mode_preserves_the_exact_label_set(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:5, 2:5] = 7
        arr[6:9, 6:9] = 9
        out = make_isotropic2d(arr, (1.0, 2.5), msk=True)
        self.assertTrue(set(np.unique(out)).issubset({0, 7, 9}), "nearest neighbour must not invent labels")
        self.assertIn(7, np.unique(out))
        self.assertIn(9, np.unique(out))

    def test_plus_color_handles_rgb_per_channel(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[..., 0] = 10
        out = make_isotropic2dpluscolor(arr, (1.0, 2.0))
        self.assertEqual(out.shape[2], 3)
        self.assertEqual(out.shape[0], 8)

    def test_plus_color_falls_back_to_the_scalar_path(self):
        arr = np.zeros((8, 8), dtype=np.uint8)
        np.testing.assert_array_equal(make_isotropic2dpluscolor(arr, (1.0, 1.0)), make_isotropic2d(arr, (1.0, 1.0)))


class TestColourAndDivision(unittest.TestCase):
    def test_contrasting_stroke_for_dark_and_bright_colours(self):
        self.assertEqual(get_contrasting_stroke_color((0.0, 0.0, 0.0)), "gray")
        self.assertEqual(get_contrasting_stroke_color((1.0, 1.0, 1.0)), "black")

    def test_accepts_rgba_and_label_indices(self):
        self.assertIn(get_contrasting_stroke_color((1.0, 1.0, 1.0, 1.0)), ("gray", "black"))
        self.assertIn(get_contrasting_stroke_color([0.0, 0.0, 0.0]), ("gray", "black"))
        self.assertIn(get_contrasting_stroke_color(5), ("gray", "black"))

    def test_div0_on_arrays_and_scalars(self):
        np.testing.assert_allclose(div0(np.array([1.0, 2.0]), np.array([2.0, 0.0])), [0.5, 0.0])
        np.testing.assert_allclose(div0(np.array([1.0]), np.array([0.0]), fill=-1), [-1.0])
        self.assertEqual(div0(1.0, 0.0, fill=3), 3)
        self.assertAlmostEqual(div0(1.0, 4.0), 0.25, places=9)

    def test_div0_leaves_finite_results_alone(self):
        np.testing.assert_allclose(div0(np.array([4.0, 9.0]), np.array([2.0, 3.0])), [2.0, 3.0])


class TestToCdt(unittest.TestCase):
    def test_none_passes_through(self):
        self.assertIsNone(to_cdt(None))

    def test_a_poi_with_points_is_returned(self):
        from TPTBox import POI  # noqa: PLC0415

        poi = POI({1: {50: (1.0, 1.0, 1.0)}}, orientation=("R", "A", "S"), zoom=(1.0, 1.0, 1.0), shape=(10.0, 10.0, 10.0))
        self.assertIsNotNone(to_cdt(poi))

    def test_an_empty_poi_becomes_none(self):
        from TPTBox import POI  # noqa: PLC0415

        empty = POI({}, orientation=("R", "A", "S"), zoom=(1.0, 1.0, 1.0), shape=(10.0, 10.0, 10.0))
        self.assertIsNone(to_cdt(empty))


class TestCreateSnapshot(unittest.TestCase):
    """Renders the shipped sample CT; assertions are on the produced PNG."""

    @classmethod
    def setUpClass(cls):
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        cls.ct, cls.subreg, cls.vert, cls.label = get_test_ct()

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _png_is_not_blank(path: Path) -> bool:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
        return int(arr.std()) > 0

    def test_image_only_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            create_snapshot(out, [Snapshot_Frame(image=self.ct)])
            self.assertTrue(out.exists())
            self.assertTrue(self._png_is_not_blank(out))

    def test_segmentation_and_centroid_overlay(self):
        from TPTBox import calc_poi_from_subreg_vert  # noqa: PLC0415

        poi = calc_poi_from_subreg_vert(self.vert, self.subreg, subreg_id=[Location.Vertebra_Corpus], verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            create_snapshot(out, [Snapshot_Frame(image=self.ct, segmentation=self.vert, centroids=poi, coronal=True)])
            self.assertTrue(out.exists())
            self.assertTrue(self._png_is_not_blank(out))

    def test_all_three_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            create_snapshot(out, [Snapshot_Frame(image=self.ct, sagittal=True, coronal=True, axial=True)])
            self.assertTrue(out.exists())

    def test_two_frames_render_differently_from_one(self):
        from PIL import Image  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / "one.png", Path(tmp) / "two.png"
            create_snapshot(one, [Snapshot_Frame(image=self.ct)])
            create_snapshot(two, [Snapshot_Frame(image=self.ct), Snapshot_Frame(image=self.ct)])
            with Image.open(one) as a, Image.open(two) as b:
                self.assertNotEqual((a.width, a.height), (b.width, b.height))

    def test_writes_several_paths_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            png, jpg = Path(tmp) / "snap.png", Path(tmp) / "snap.jpg"
            create_snapshot([png, jpg], [Snapshot_Frame(image=self.ct)])
            self.assertTrue(png.exists())
            self.assertTrue(jpg.exists())

    def test_none_frames_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            create_snapshot(out, [Snapshot_Frame(image=self.ct), None])  # type: ignore[list-item]
            self.assertTrue(out.exists())

    def test_crop_shrinks_the_rendered_area(self):
        from PIL import Image  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            plain, cropped = Path(tmp) / "a.png", Path(tmp) / "b.png"
            create_snapshot(plain, [Snapshot_Frame(image=self.ct, segmentation=self.vert)])
            create_snapshot(cropped, [Snapshot_Frame(image=self.ct, segmentation=self.vert)], crop=True)
            with Image.open(plain) as a, Image.open(cropped) as b:
                self.assertNotEqual((a.width, a.height), (b.width, b.height))

    def test_rendering_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.png", Path(tmp) / "b.png"
            for path in (first, second):
                create_snapshot(path, [Snapshot_Frame(image=self.ct, segmentation=self.vert)])
            self.assertEqual(first.read_bytes(), second.read_bytes(), "the same input must render identically")


class TestSnapshotTemplates(unittest.TestCase):
    """The templates are Snapshot_Frame builders; `out_path` is their 5th argument."""

    @classmethod
    def setUpClass(cls):
        from TPTBox import calc_poi_from_subreg_vert  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

        cls.ct, cls.subreg, cls.vert, cls.label = get_test_ct()
        cls.poi = calc_poi_from_subreg_vert(cls.vert, cls.subreg, subreg_id=[Location.Vertebra_Corpus], verbose=False)

    def tearDown(self):
        plt.close("all")

    def test_snapshot_template(self):
        from TPTBox.spine.snapshot2D.snapshot_templates import snapshot  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            paths = snapshot(self.ct, self.vert, self.poi, self.subreg, out_path=out)
            self.assertTrue(out.exists())
            self.assertEqual([Path(p) for p in paths], [out])

    def test_mri_snapshot_template(self):
        from TPTBox.spine.snapshot2D.snapshot_templates import mri_snapshot  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            mri_snapshot(self.ct, self.vert, self.poi, self.subreg, out_path=out)
            self.assertTrue(out.exists())

    @unittest.skipUnless(HAS_TORCH, "mip_shot sets denoise_threshold, which makes create_snapshot import torch")
    def test_mip_shot_template(self):
        from TPTBox.spine.snapshot2D.snapshot_templates import mip_shot  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            mip_shot(self.ct, self.vert, self.poi, out_path=out)
            self.assertTrue(out.exists())

    def test_ct_mode_is_chosen_for_a_bids_ct_file(self):
        from TPTBox.core.bids_files import BIDS_FILE  # noqa: PLC0415
        from TPTBox.spine.snapshot2D.snapshot_templates import snapshot  # noqa: PLC0415
        from TPTBox.tests.test_utils import get_nii_paths_ct, get_tests_dir  # noqa: PLC0415

        ct_path = get_nii_paths_ct()[0]
        bids = BIDS_FILE(ct_path, get_tests_dir(), verbose=False)
        self.assertEqual(bids.bids_format, "ct")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            snapshot(bids, self.vert, self.poi, self.subreg, out_path=out)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
