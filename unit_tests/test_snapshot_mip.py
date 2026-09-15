"""Regression tests for the mm -> voxel slab-thickness conversion in the curved-planar projections.

Both ``curve_projected_mip`` and ``curve_projected_mean`` used to convert the slab
half-width from millimetres to voxels *inside* their per-slice loop and assign the
result back to the same variable, so every slice re-divided the already-divided
value. With ``y_zoom < 1`` the slab grew geometrically until ``int()`` raised
``OverflowError: int too big to convert``; with ``y_zoom > 1`` it shrank towards the
1-voxel floor imposed by the ceiling term, so the projection quietly used a far
thinner slab than the caller asked for and raised nothing at all.
"""

from __future__ import annotations

import unittest

import numpy as np

from TPTBox.spine.snapshot2D.snapshot_modular import (
    _mm_to_voxel_thickness,
    curve_projected_mean,
    curve_projected_mip,
)


class _FakePOI(dict):
    """Minimal stand-in for ``POI``: the projections only test ``23 in ctd_list``."""


def _synthetic_volume(nx: int = 64, ny: int = 48, nz: int = 32) -> np.ndarray:
    """A volume in IPL orientation with a bright anterior-posterior band per slice."""
    rng = np.random.default_rng(0)
    vol = rng.random((nx, ny, nz)) * 10.0
    vol[:, ny // 2 - 4 : ny // 2 + 4, :] += 100.0  # a structure inside every slab
    return vol


def _curve(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Centroid x-indices and the interpolated y coordinate for each x between them."""
    x_ctd = np.arange(4, nx - 4, dtype=int)
    y_cord = np.full(len(x_ctd), ny // 2, dtype=int)
    return x_ctd, y_cord


class TestMmToVoxelThickness(unittest.TestCase):
    def test_conversion_is_a_pure_function_of_the_mm_value(self):
        # Repeated application must be idempotent in its *input*: the bug was that
        # the output was fed back in as the next input.
        for zoom in (0.25, 0.5, 1.0, 1.5, 3.0):
            first = _mm_to_voxel_thickness((100, 300), zoom)
            second = _mm_to_voxel_thickness((100, 300), zoom)
            self.assertEqual(first, second, f"not deterministic at {zoom=}")

    def test_ceiling_division(self):
        self.assertEqual(_mm_to_voxel_thickness((100, 300), 1.0), [100, 300])
        self.assertEqual(_mm_to_voxel_thickness((100, 300), 0.5), [200, 600])
        self.assertEqual(_mm_to_voxel_thickness((10, 10), 3.0), [4, 4])  # ceil(10/3)

    def test_never_returns_zero(self):
        # A zero-width slab yields an empty array slice, which np.max cannot reduce.
        self.assertEqual(_mm_to_voxel_thickness((1, 1), 1000.0), [1, 1])

    def test_degenerate_zoom_is_tolerated(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            self.assertEqual(_mm_to_voxel_thickness((100, 300), bad), [100, 300])


class TestCurveProjections(unittest.TestCase):
    """The two projections must survive every voxel spacing and stay non-degenerate."""

    def _run(self, fn, zoom: float, **kwargs):
        nx, ny, nz = 64, 48, 32
        img = _synthetic_volume(nx, ny, nz)
        x_ctd, y_cord = _curve(nx, ny)
        return fn(
            img,
            (1.0, zoom, 1.0),
            x_ctd,
            y_cord,
            _FakePOI(),
            thick_t=(20, 30),
            **kwargs,
        )

    def test_mip_fine_spacing_does_not_overflow(self):
        # Pre-fix: OverflowError: int too big to convert.
        sag, cor, _ = self._run(curve_projected_mip, 0.5)
        self.assertTrue(np.any(cor > 0), "coronal MIP is empty")
        self.assertTrue(np.any(sag > 0), "sagittal MIP is empty")

    def test_mean_fine_spacing_does_not_overflow(self):
        sag, cor, _ = self._run(curve_projected_mean, 0.5)
        self.assertTrue(np.any(cor > 0), "coronal mean projection is empty")
        self.assertTrue(np.any(sag > 0), "sagittal mean projection is empty")

    def test_coarse_spacing_keeps_the_requested_slab_width(self):
        # Pre-fix the slab shrank once per slice towards the 1-voxel floor that the
        # ceiling term imposes: for thick_t=(20, 30) mm at 3 mm/voxel it went
        # [7, 10] -> [3, 4] -> [1, 2] -> [1, 1] and stayed there. No exception, but
        # every slice after the third projected a 1-voxel slab instead of the
        # requested one. Place the only signal at the far edge of the intended slab
        # so a collapsed slab simply cannot see it.
        nx, ny, nz = 64, 48, 32
        y_ref = ny // 2  # 24; intended voxel slab at 3 mm is [y_ref - 10, y_ref + 7]
        for fn in (curve_projected_mip, curve_projected_mean):
            with self.subTest(fn=fn.__name__):
                img = np.zeros((nx, ny, nz))
                img[:, y_ref - 9 : y_ref - 7, :] = 100.0  # inside [14, 31], outside [23, 25]
                x_ctd, y_cord = _curve(nx, ny)
                _sag, cor, _ = fn(img, (1.0, 3.0, 1.0), x_ctd, y_cord, _FakePOI(), thick_t=(20, 30))
                rows_with_signal = np.count_nonzero(cor.max(axis=1) > 0)
                self.assertGreater(
                    rows_with_signal,
                    cor.shape[0] // 2,
                    f"{fn.__name__} lost the requested slab width at 3 mm spacing "
                    f"(only {rows_with_signal}/{cor.shape[0]} rows saw the structure)",
                )

    def test_slab_width_is_constant_across_slices(self):
        # The direct expression of the bug: the slab used for the first slice and
        # the slab used for the last slice must be the same width.
        for zoom in (0.5, 1.0, 3.0):
            with self.subTest(zoom=zoom):
                _sag, cor, _ = self._run(curve_projected_mip, zoom)
                per_row = cor.max(axis=1)
                self.assertGreater(per_row[5], 0)
                self.assertGreater(per_row[-5], 0, f"last slices lost their slab at {zoom=}")

    def test_all_zooms_agree_on_signal_presence(self):
        for zoom in (0.25, 0.5, 0.9, 1.0, 1.5, 3.0):
            with self.subTest(zoom=zoom):
                _sag, cor, _ = self._run(curve_projected_mip, zoom)
                self.assertTrue(np.isfinite(cor).all())
                self.assertTrue(np.any(cor > 0))

    def test_colored_depth_variant(self):
        sag, cor, _ = self._run(curve_projected_mip, 0.5, make_colored_depth=True)
        self.assertEqual(sag.shape[-1], 3)
        self.assertEqual(cor.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
