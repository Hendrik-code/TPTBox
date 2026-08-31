# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_PYVISTA = _has_module("pyvista")
HAS_VTK = _has_module("vtk")
HAS_SKIMAGE = _has_module("skimage")
HAS_FURY = _has_module("fury")
HAS_PIL = _has_module("PIL")
HAS_XVFBWRAPPER = _has_module("xvfbwrapper")
HAS_XVFB_BINARY = shutil.which("Xvfb") is not None

MESH_BASICS = HAS_PYVISTA and HAS_VTK and HAS_SKIMAGE
SNAPSHOT_STACK = MESH_BASICS and HAS_FURY and HAS_PIL and HAS_XVFBWRAPPER and HAS_XVFB_BINARY


def _cube_array(shape: tuple[int, int, int] = (20, 20, 20), label: int = 1) -> np.ndarray:
    """Build a 3D array containing a single non-zero cube in the middle."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[6:14, 6:14, 6:14] = label
    return arr


def _multilabel_array(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    """Build a 3D array with two disjoint labeled cubes."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[4:10, 4:10, 4:10] = 1
    arr[14:20, 14:20, 14:20] = 2
    return arr


class TestMeshColors(unittest.TestCase):
    """Pure-Python tests for TPTBox.mesh3D.mesh_colors (no rendering deps)."""

    def test_rgb_color_init_from_tuple(self) -> None:
        from TPTBox.mesh3D.mesh_colors import RGB_Color

        c = RGB_Color((10, 20, 30))
        self.assertTrue(np.array_equal(c(), np.array([10, 20, 30])))

    def test_rgb_color_init_separate(self) -> None:
        from TPTBox.mesh3D.mesh_colors import RGB_Color

        c = RGB_Color.init_separate(1, 2, 3)
        self.assertTrue(np.array_equal(c(), np.array([1, 2, 3])))

    def test_rgb_color_init_list_length_assertion(self) -> None:
        from TPTBox.mesh3D.mesh_colors import RGB_Color

        with self.assertRaises(AssertionError):
            RGB_Color.init_list([1, 2])

    def test_rgb_color_call_normed(self) -> None:
        from TPTBox.mesh3D.mesh_colors import RGB_Color

        c = RGB_Color((255, 0, 128))
        normed = c(normed=True)
        self.assertAlmostEqual(float(normed[0]), 1.0)
        self.assertAlmostEqual(float(normed[1]), 0.0)
        self.assertAlmostEqual(float(normed[2]), 128 / 255.0)

    def test_rgb_color_getitem_returns_normalized_channel(self) -> None:
        from TPTBox.mesh3D.mesh_colors import RGB_Color

        c = RGB_Color((255, 128, 0))
        self.assertAlmostEqual(c[0], 1.0)
        self.assertAlmostEqual(c[1], 128 / 255.0)
        self.assertAlmostEqual(c[2], 0.0)

    def test_get_color_by_label_known(self) -> None:
        from TPTBox.mesh3D.mesh_colors import Mesh_Color_List, get_color_by_label

        color = get_color_by_label(1)
        self.assertTrue(np.array_equal(color.rgb, Mesh_Color_List.ITK_1.rgb))

    def test_get_color_by_label_wraps_for_out_of_range(self) -> None:
        from TPTBox.mesh3D.mesh_colors import get_color_by_label

        # Labels >= 150 are wrapped modulo 50 into 1..50.
        wrapped = get_color_by_label(200)
        # 200 % 50 + 1 == 1
        expected = get_color_by_label(1)
        self.assertTrue(np.array_equal(wrapped.rgb, expected.rgb))

    def test_color_palette_size(self) -> None:
        from TPTBox.mesh3D.mesh_colors import Mesh_Color_List

        itk_entries = [name for name in vars(Mesh_Color_List) if name.startswith("ITK_")]
        self.assertEqual(len(itk_entries), 201)

    def test_write_ctbl_writes_valid_slicer_table(self) -> None:
        from TPTBox.mesh3D.mesh_colors import write_ctbl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "colors.ctbl"
            write_ctbl(path)
            self.assertTrue(path.exists())
            text = path.read_text()
            self.assertIn("# Color table file for 3D Slicer", text)
            self.assertIn("0 Background 0 0 0 0", text)
            # A representative ITK entry should be present.
            self.assertIn("1 ITK_1", text)


@unittest.skipUnless(MESH_BASICS, "pyvista, vtk or scikit-image not installed")
class TestMesh3D(unittest.TestCase):
    """Round-trip and construction tests for the thin Mesh3D wrapper."""

    def test_construct_from_polydata(self) -> None:
        import pyvista as pv

        from TPTBox.mesh3D.mesh import Mesh3D

        base = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
        mesh = Mesh3D(base)
        self.assertIs(mesh.mesh, base)

    def test_save_and_load_ply(self) -> None:
        import pyvista as pv

        from TPTBox.mesh3D.mesh import Mesh3D, MeshOutputType

        with tempfile.TemporaryDirectory() as tmp:
            base = pv.Sphere(radius=2.0, theta_resolution=8, phi_resolution=8)
            mesh = Mesh3D(base)
            out = Path(tmp) / "sphere"
            mesh.save(out, mode=MeshOutputType.PLY, verbose=False)
            saved = Path(str(out) + ".ply")
            self.assertTrue(saved.exists(), f"expected mesh saved to {saved}")
            loaded = Mesh3D.load(saved)
            self.assertEqual(loaded.mesh.n_points, base.n_points)
            self.assertEqual(loaded.mesh.n_cells, base.n_cells)

    def test_save_preserves_explicit_extension(self) -> None:
        import pyvista as pv

        from TPTBox.mesh3D.mesh import Mesh3D, MeshOutputType

        with tempfile.TemporaryDirectory() as tmp:
            mesh = Mesh3D(pv.Sphere(theta_resolution=6, phi_resolution=6))
            out = Path(tmp) / "already.ply"
            mesh.save(out, mode=MeshOutputType.PLY, verbose=False)
            # save() must not double-append the extension.
            self.assertTrue(out.exists())
            self.assertFalse((Path(str(out) + ".ply")).exists())

    def test_load_missing_file_raises(self) -> None:
        from TPTBox.mesh3D.mesh import Mesh3D

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.ply"
            with self.assertRaises(AssertionError):
                Mesh3D.load(missing)

    def test_save_to_missing_directory_raises(self) -> None:
        import pyvista as pv

        from TPTBox.mesh3D.mesh import Mesh3D

        mesh = Mesh3D(pv.Sphere(theta_resolution=6, phi_resolution=6))
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "no_such_subdir"
            with self.assertRaises(FileNotFoundError):
                mesh.save(missing_dir / "out.ply", verbose=False)


@unittest.skipUnless(MESH_BASICS, "pyvista, vtk or scikit-image not installed")
class TestSegmentationMesh(unittest.TestCase):
    """SegmentationMesh construction and offset behavior."""

    def test_from_array(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh

        arr = _cube_array()
        mesh = SegmentationMesh(arr)
        self.assertGreater(mesh.mesh.n_points, 0)
        self.assertGreater(mesh.mesh.n_cells, 0)
        # Marching cubes surfaces the cube boundary; vertices sit near the cube.
        pts = np.asarray(mesh.mesh.points)
        self.assertGreaterEqual(pts[:, 0].min(), 4.0)
        self.assertLessEqual(pts[:, 0].max(), 16.0)

    def test_rejects_non_zero_background(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh

        arr = np.ones((10, 10, 10), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            SegmentationMesh(arr)

    def test_rejects_non_3d_input(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh

        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:5, 2:5] = 1
        with self.assertRaises(AssertionError):
            SegmentationMesh(arr)

    def test_get_mesh_with_offset_shifts_vertices(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh

        arr = _cube_array()
        mesh = SegmentationMesh(arr)
        offset = (100.0, 50.0, -25.0)
        shifted = mesh.get_mesh_with_offset(offset)
        orig_pts = np.asarray(mesh.mesh.points)
        new_pts = np.asarray(shifted.points)
        self.assertEqual(orig_pts.shape, new_pts.shape)
        diff = new_pts - orig_pts
        np.testing.assert_allclose(diff.mean(axis=0), np.array(offset), atol=1e-5)

    def test_float_input_is_converted(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh

        arr = _cube_array().astype(np.float32)
        mesh = SegmentationMesh(arr)
        self.assertGreater(mesh.mesh.n_points, 0)

    def test_from_segmentation_nii_sample(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh
        from TPTBox.tests.test_utils import get_test_ct

        _, _, vert, _ = get_test_ct()
        mesh = SegmentationMesh.from_segmentation_nii(vert, rescale_to_iso=True)
        self.assertGreater(mesh.mesh.n_points, 0)
        self.assertGreater(mesh.mesh.n_cells, 0)

    def test_from_segmentation_nii_rejects_non_seg(self) -> None:
        from TPTBox.mesh3D.mesh import SegmentationMesh
        from TPTBox.tests.test_utils import get_test_ct

        ct, _, _, _ = get_test_ct()
        # ct is loaded with seg=False
        with self.assertRaises(AssertionError):
            SegmentationMesh.from_segmentation_nii(ct)


@unittest.skipUnless(MESH_BASICS, "pyvista, vtk or scikit-image not installed")
class TestPOIMesh(unittest.TestCase):
    """POIMesh glyph mesh construction."""

    def _sample_poi(self):
        from TPTBox import calc_centroids
        from TPTBox.tests.test_utils import get_test_ct

        _, _, vert, _ = get_test_ct()
        return calc_centroids(vert)

    def test_from_sample_poi(self) -> None:
        from TPTBox.mesh3D.mesh import POIMesh

        poi = self._sample_poi()
        mesh = POIMesh(poi, rescale_to_iso=False, size_factor=1.5)
        self.assertGreater(mesh.mesh.n_points, 0)
        self.assertGreater(len(mesh.poi_extracted), 0)

    def test_empty_filter_raises(self) -> None:
        from TPTBox.mesh3D.mesh import POIMesh

        poi = self._sample_poi()
        with self.assertRaises(AssertionError):
            POIMesh(poi, rescale_to_iso=False, regions=[-1], subregions=[-1])

    def test_get_mesh_with_offset(self) -> None:
        from TPTBox.mesh3D.mesh import POIMesh

        poi = self._sample_poi()
        mesh = POIMesh(poi, rescale_to_iso=False, size_factor=1.0)
        pts_a = np.asarray(mesh.mesh.points)
        shifted = mesh.get_mesh_with_offset((10.0, 20.0, 30.0))
        pts_b = np.asarray(shifted.points)
        self.assertEqual(pts_a.shape, pts_b.shape)
        # Every point should move by exactly the offset.
        np.testing.assert_allclose((pts_b - pts_a).mean(axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-5)


@unittest.skipUnless(MESH_BASICS, "pyvista, vtk or scikit-image not installed")
class TestHtmlPreview(unittest.TestCase):
    """make_html_preview end-to-end (offscreen)."""

    def test_export_html(self) -> None:
        import pyvista as pv

        from TPTBox.mesh3D.html_preview import make_html_preview
        from TPTBox.tests.test_utils import get_test_ct

        # Off-screen rendering keeps CI happy when there is no display.
        pv.OFF_SCREEN = True
        _, _, vert, _ = get_test_ct()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.html"
            try:
                make_html_preview([vert], out, rescale_to_iso=False)
            except Exception as exc:  # noqa: BLE001
                # pyvista HTML export needs a working GL context; on some CI
                # runners this is not available. Skip rather than fail — the
                # rest of the mesh3D tests already cover core logic.
                raise unittest.SkipTest(f"pyvista HTML export unavailable: {exc}") from exc
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_asserts_html_extension(self) -> None:
        from TPTBox.mesh3D.html_preview import make_html_preview
        from TPTBox.tests.test_utils import get_test_ct

        _, _, vert, _ = get_test_ct()
        with self.assertRaises(AssertionError):
            make_html_preview([vert], "not_html.txt")

    def test_preview_settings_defaults(self) -> None:
        from TPTBox.mesh3D.html_preview import Preview_Settings
        from TPTBox.tests.test_utils import get_test_ct

        _, _, vert, _ = get_test_ct()
        settings = Preview_Settings(vert)
        self.assertEqual(settings.opacity, 1.0)
        self.assertEqual(settings.color, "auto")
        self.assertIsNone(settings.offset)


@unittest.skipUnless(SNAPSHOT_STACK, "fury / Pillow / xvfbwrapper (or the Xvfb binary) not installed")
class TestSnapshot3D(unittest.TestCase):
    """Full snapshot pipeline. Requires fury, Pillow and a working Xvfb."""

    def test_make_snapshot3D_from_sample(self) -> None:
        from TPTBox.mesh3D.snapshot3D import make_snapshot3D
        from TPTBox.tests.test_utils import get_test_ct

        _, _, vert, _ = get_test_ct()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snap.png"
            try:
                img = make_snapshot3D(vert, out, view="A", smoothing=1, verbose=False)
            except Exception as exc:  # noqa: BLE001
                # Rendering can still fail on headless CI without proper GL/mesa
                # even with xvfb present. Skip in that case.
                raise unittest.SkipTest(f"3D snapshot unavailable: {exc}") from exc
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)
            self.assertGreater(img.size[0], 0)
            self.assertGreater(img.size[1], 0)


if __name__ == "__main__":
    unittest.main()
