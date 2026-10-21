"""Unit tests for the segmentation wrappers, without weights, GPU or network.

The mock sits at the model boundary only: the network forward pass (or the whole
predictor) is replaced, while everything around it - RAM budgeting, chunking,
orientation handling, label post-processing, path construction, argv assembly -
runs for real. Assertions are therefore about the post-processing result, not
about whether a mock was called.

The pure numeric helpers need no mock at all and are tested directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

# `itertools.pairwise` is 3.10+; the package ships a backport for its own 3.9 support.
from TPTBox.core.compat import pairwise

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NNUNET = importlib.util.find_spec("nnunetv2") is not None
TORCH_REASON = "requires torch"
NNUNET_REASON = "requires torch + nnunetv2"


def _nii(arr: np.ndarray, seg: bool = False, zoom=(1.4, 1.4, 3.0)):
    from TPTBox import NII  # noqa: PLC0415

    return NII(nib.Nifti1Image(arr, np.diag([*zoom, 1.0])), seg=seg)


def _image(shape=(24, 24, 24)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.int16)
    arr[4:-4, 4:-4, 4:-4] = 80
    return arr


@unittest.skipUnless(HAS_NNUNET, NNUNET_REASON)
class TestRamBudgeting(unittest.TestCase):
    """Pure arithmetic from the docstring's own formula."""

    def test_estimate_matches_the_documented_formula(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import estimate_peak_ram_mb  # noqa: PLC0415

        shape, num_classes, num_channels = (64, 64, 64), 10, 1
        per_voxel = (num_classes * 2) + 2 + (num_channels * 2 * 2) + 1
        expected = int(np.prod(shape)) * per_voxel * 1.3 / (1024 * 1024)
        self.assertAlmostEqual(estimate_peak_ram_mb(shape, num_classes, num_channels), expected, places=6)

    def test_estimate_scales_linearly_in_voxels_and_classes(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import estimate_peak_ram_mb  # noqa: PLC0415

        base = estimate_peak_ram_mb((32, 32, 32), 4)
        self.assertAlmostEqual(estimate_peak_ram_mb((64, 32, 32), 4), 2 * base, places=6)
        self.assertGreater(estimate_peak_ram_mb((32, 32, 32), 8), base)

    def test_chunking_returns_one_when_the_volume_fits(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import compute_cpu_chunks_for_ram  # noqa: PLC0415

        self.assertEqual(compute_cpu_chunks_for_ram((32, 32, 32), 0, 4, 1, 4, target_mb=1e9), 1)

    def test_chunking_grows_as_the_budget_shrinks(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import compute_cpu_chunks_for_ram, estimate_peak_ram_mb  # noqa: PLC0415

        shape = (256, 256, 256)
        whole = estimate_peak_ram_mb(shape, 10, 1)
        self.assertEqual(compute_cpu_chunks_for_ram(shape, 0, 10, 1, 4, target_mb=whole * 2), 1)
        tight = compute_cpu_chunks_for_ram(shape, 0, 10, 1, 4, target_mb=whole / 4)
        self.assertGreater(tight, 1)
        self.assertLessEqual(
            estimate_peak_ram_mb((shape[0] // tight + 8, shape[1], shape[2]), 10, 1),
            whole / 4,
            "the returned chunk count must actually fit the budget",
        )

    def test_chunking_degenerate_inputs(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import compute_cpu_chunks_for_ram  # noqa: PLC0415

        self.assertEqual(compute_cpu_chunks_for_ram((1, 32, 32), 0, 4, 1, 4, target_mb=10), 1)
        self.assertEqual(compute_cpu_chunks_for_ram((32, 32, 32), 0, 4, 1, 4, target_mb=0), 1)

    def test_total_ram_is_a_positive_number(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import _get_total_ram_mb  # noqa: PLC0415

        self.assertGreater(_get_total_ram_mb(), 0.0)


@unittest.skipUnless(HAS_NNUNET, NNUNET_REASON)
class TestSplitRanges(unittest.TestCase):
    """Each entry is (read_start, read_stop, write_start, write_stop).

    The read windows overlap so every chunk has context; the write windows, taken
    relative to their own read window, must tile the axis exactly once.
    """

    def test_write_windows_tile_the_axis_exactly(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import _split_ranges  # noqa: PLC0415

        for length, n_chunks, overlap in ((100, 4, 5), (100, 1, 0), (37, 3, 2), (10, 10, 1), (64, 3, 8)):
            with self.subTest(length=length, n_chunks=n_chunks, overlap=overlap):
                written = []
                for read_start, read_stop, write_start, write_stop in _split_ranges(length, n_chunks, overlap):
                    self.assertGreaterEqual(read_start, 0)
                    self.assertLessEqual(read_stop, length)
                    self.assertLessEqual(write_stop - write_start, read_stop - read_start)
                    written.extend(range(read_start + write_start, read_start + write_stop))
                self.assertEqual(written, list(range(length)), "the write windows must tile the axis in order")

    def test_a_single_chunk_is_the_whole_axis(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import _split_ranges  # noqa: PLC0415

        self.assertEqual(list(_split_ranges(50, 1, 0)), [(0, 50, 0, 50)])

    def test_read_windows_overlap_their_neighbours(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import _split_ranges  # noqa: PLC0415

        ranges = list(_split_ranges(100, 4, 6))
        self.assertGreater(len(ranges), 1)
        for (_, stop, _, _), (start, _, _, _) in pairwise(ranges):
            with self.subTest(boundary=(stop, start)):
                self.assertLess(start, stop, "consecutive read windows must share voxels")


@unittest.skipUnless(HAS_NNUNET, NNUNET_REASON)
class TestSlidingNdSlices(unittest.TestCase):
    def test_identity_function_reconstructs_the_input(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import sliding_nd_slices  # noqa: PLC0415

        arr = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
        out = sliding_nd_slices(arr, (8, 8, 8), 2, lambda patch: patch)
        np.testing.assert_allclose(out, arr)

    def test_the_function_is_applied_to_every_voxel(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import sliding_nd_slices  # noqa: PLC0415

        arr = np.ones((12, 12, 12), dtype=np.float32)
        out = sliding_nd_slices(arr, (6, 6, 6), 2, lambda patch: patch * 3)
        np.testing.assert_allclose(out, 3.0)

    def test_shape_is_preserved_for_a_non_divisible_volume(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import sliding_nd_slices  # noqa: PLC0415

        arr = np.ones((13, 11, 7), dtype=np.float32)
        self.assertEqual(sliding_nd_slices(arr, (8, 8, 8), 2, lambda p: p).shape, arr.shape)


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class TestSquashForFloat16(unittest.TestCase):
    def test_a_large_volume_is_brought_into_range(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import squash_so_it_fits_in_float16  # noqa: PLC0415

        arr = np.full((8, 8, 8), 1e6, dtype=np.float32)
        arr[0, 0, 0] = 2e6
        out = squash_so_it_fits_in_float16(_nii(arr))
        self.assertTrue(np.isfinite(np.asarray(out.get_array(), dtype=np.float16)).all())
        # the ratio between voxels is what matters, not the absolute values
        values = out.get_array()
        self.assertAlmostEqual(float(values.max() / values[values > 0].min()), 2.0, delta=0.1)

    def test_a_small_volume_is_left_alone(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import squash_so_it_fits_in_float16  # noqa: PLC0415

        arr = np.full((8, 8, 8), 100.0, dtype=np.float32)
        np.testing.assert_allclose(squash_so_it_fits_in_float16(_nii(arr)).get_array(), arr)


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class TestExtractVertebraBodies(unittest.TestCase):
    """Pure label logic on a VIBESeg output: alternating body / disc blocks."""

    @staticmethod
    def _stack(bodies: int = 6):
        shape = (20, 20, 8 * bodies + 4)
        arr = np.zeros(shape, dtype=np.uint8)
        z = 2
        for b in range(bodies):
            arr[3:17, 3:17, z : z + 6] = 69  # vertebra body
            z += 6
            if b < bodies - 1:
                arr[3:17, 3:17, z : z + 2] = 68  # intervertebral disc
                z += 2
        return _nii(arr, seg=True, zoom=(1.0, 1.0, 1.0))

    def test_six_bodies_are_labelled_l5_upwards(self):
        from TPTBox import (
            NII,  # noqa: PLC0415
            POI,  # noqa: PLC0415
        )
        from TPTBox.segmentation.VibeSeg.vibeseg import extract_vertebra_bodies_from_VibeSeg  # noqa: PLC0415

        vert, poi = extract_vertebra_bodies_from_VibeSeg(self._stack(bodies=6))
        self.assertIsInstance(vert, NII)
        self.assertIsInstance(poi, POI)
        labels = set(np.unique(vert.get_seg_array())) - {0}
        self.assertEqual(labels, {19, 20, 21, 22, 23, 24}, "inferior to superior: L5..T12")

    def test_the_number_of_labels_follows_the_number_of_bodies(self):
        from TPTBox.segmentation.VibeSeg.vibeseg import extract_vertebra_bodies_from_VibeSeg  # noqa: PLC0415

        for bodies in (3, 5):
            with self.subTest(bodies=bodies):
                vert, _ = extract_vertebra_bodies_from_VibeSeg(self._stack(bodies=bodies))
                self.assertEqual(len(set(np.unique(vert.get_seg_array())) - {0}), bodies)

    def test_one_poi_per_body(self):
        from TPTBox.segmentation.VibeSeg.vibeseg import extract_vertebra_bodies_from_VibeSeg  # noqa: PLC0415

        vert, poi = extract_vertebra_bodies_from_VibeSeg(self._stack(bodies=4))
        self.assertEqual(len(list(poi.keys())), len(set(np.unique(vert.get_seg_array())) - {0}))


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class TestDeface(unittest.TestCase):
    def test_extend_mask_grows_only_in_the_requested_direction(self):
        from TPTBox.segmentation._deface import _extend_mask  # noqa: PLC0415

        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[8:12, 8:12, 8:12] = 1
        nii = _nii(arr, seg=True, zoom=(1.0, 1.0, 1.0)).reorient_(("R", "A", "S"))
        out = _extend_mask(nii.copy(), 4, direction="A")
        before = nii.get_seg_array()
        after = out.get_seg_array()
        self.assertGreater(int(after.sum()), int(before.sum()))
        # projections perpendicular to A are unchanged
        np.testing.assert_array_equal(after.any(axis=1), before.any(axis=1))

    def test_extend_mask_by_zero_is_a_noop(self):
        from TPTBox.segmentation._deface import _extend_mask  # noqa: PLC0415

        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[6:10, 6:10, 6:10] = 1
        nii = _nii(arr, seg=True, zoom=(1.0, 1.0, 1.0))
        np.testing.assert_array_equal(_extend_mask(nii.copy(), 0).get_seg_array(), nii.get_seg_array())

    def test_deface_img_blanks_the_masked_voxels_only(self):
        from TPTBox.segmentation._deface import deface_img  # noqa: PLC0415

        ct = _nii(_image(), seg=False)
        mask_arr = np.zeros(ct.shape, dtype=np.uint8)
        mask_arr[:8] = 1
        out = deface_img(ct, _nii(mask_arr, seg=True), min_value=-1024)
        arr = out.get_array()
        self.assertTrue((arr[:8] == -1024).all(), "masked voxels take min_value")
        np.testing.assert_array_equal(arr[8:], ct.get_array()[8:], "the rest is untouched")

    def test_deface_img_to_int_gives_an_integer_dtype(self):
        from TPTBox.segmentation._deface import deface_img  # noqa: PLC0415

        ct = _nii(_image().astype(np.float32), seg=False)
        mask_arr = np.zeros(ct.shape, dtype=np.uint8)
        mask_arr[:4] = 1
        out = deface_img(ct, _nii(mask_arr, seg=True), to_int=True)
        self.assertTrue(np.issubdtype(out.get_array().dtype, np.integer))

    def test_deface_img_writes_the_output_file(self):
        from TPTBox.segmentation._deface import deface_img  # noqa: PLC0415

        ct = _nii(_image(), seg=False)
        mask_arr = np.zeros(ct.shape, dtype=np.uint8)
        mask_arr[:4] = 1
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "defaced.nii.gz"
            deface_img(ct, _nii(mask_arr, seg=True), ct_out=out_path)
            self.assertTrue(out_path.exists())


class TestSpinepsDispatch(unittest.TestCase):
    """`spineps.py` imports spineps only inside function bodies, so a fake module is enough."""

    def _fake_spineps(self):
        fake = mock.MagicMock()
        return {
            "spineps": fake,
            "spineps.seg_run": fake.seg_run,
            "spineps.get_models": fake.get_models,
            "spineps.seg_model": fake.seg_model,
        }, fake

    def test_get_outpaths_wraps_a_plain_path_in_a_bids_file(self):
        modules, fake = self._fake_spineps()
        with mock.patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as tmp:
            from TPTBox.segmentation.spineps import get_outpaths_spineps  # noqa: PLC0415

            nii_path = Path(tmp) / "dataset-x" / "rawdata" / "sub-1" / "sub-1_T2w.nii.gz"
            nii_path.parent.mkdir(parents=True)
            nii_path.touch()
            get_outpaths_spineps(nii_path)
            self.assertTrue(fake.seg_run.output_paths_from_input.called)
            bids_arg = fake.seg_run.output_paths_from_input.call_args[0][0]
            self.assertEqual(bids_arg.bids_format, "T2w")

    def test_an_explicit_dataset_root_wins(self):
        modules, fake = self._fake_spineps()
        with mock.patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as tmp:
            from TPTBox.segmentation.spineps import get_outpaths_spineps  # noqa: PLC0415

            root = Path(tmp) / "dataset-x"
            nii_path = root / "rawdata" / "sub-1" / "sub-1_T2w.nii.gz"
            nii_path.parent.mkdir(parents=True)
            nii_path.touch()
            get_outpaths_spineps(nii_path, dataset=root)
            bids_arg = fake.seg_run.output_paths_from_input.call_args[0][0]
            self.assertEqual(Path(bids_arg.dataset), root)

    def test_run_spineps_all_invokes_one_subprocess_per_model(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run") as run:
            from TPTBox.segmentation.spineps import _run_spineps_all  # noqa: PLC0415

            _run_spineps_all(Path(tmp))
            self.assertGreaterEqual(run.call_count, 1)
            for call in run.call_args_list:
                argv = call[0][0]
                with self.subTest(argv=argv if isinstance(argv, str) else " ".join(map(str, argv))):
                    self.assertIn("spineps", " ".join(map(str, argv)) if not isinstance(argv, str) else argv)


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class TestAutoDownload(unittest.TestCase):
    def test_read_config_parses_the_shipped_json(self):
        from TPTBox.segmentation.VibeSeg.auto_download import read_config  # noqa: PLC0415

        try:
            config = read_config(85)
        except (FileNotFoundError, KeyError) as exc:  # pragma: no cover - depends on the shipped table
            self.skipTest(f"no config entry for this model id: {exc}")
        self.assertIsInstance(config, dict)

    def test_download_weights_short_circuits_when_the_folder_exists(self):
        from TPTBox.segmentation.VibeSeg import auto_download  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(auto_download, "_download") as download:
            model_path = Path(tmp)
            weights = auto_download.get_weights_dir(85, model_path=model_path)
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "plans.json").write_text(json.dumps({"configurations": {}}), encoding="utf-8")
            auto_download.download_weights(85, model_path=model_path)
            self.assertFalse(download.called, "an existing weights folder must not trigger a download")

    def test_get_weights_dir_is_under_the_given_model_path(self):
        from TPTBox.segmentation.VibeSeg.auto_download import get_weights_dir  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = get_weights_dir(85, model_path=Path(tmp))
            self.assertIn(Path(tmp), out.parents)


@unittest.skipUnless(HAS_NNUNET, NNUNET_REASON)
class TestRunInferenceWithAMockedPredictor(unittest.TestCase):
    """The mock replaces only `predict_single_npy_array`.

    Everything around it - orientation handling, the properties dict handed to the
    predictor, and the conversion of the returned array back into an NII on the input
    grid - runs for real, and that is what is asserted.
    """

    @staticmethod
    def _predictor(label_value: int = 3):
        predictor = mock.MagicMock()

        def predict(img, props, **kwargs):  # noqa: ARG001
            # img is (C, P, I, R); return a label map over the spatial dims
            out = np.zeros(img.shape[1:], dtype=np.uint8)
            out[..., : out.shape[-1] // 2] = label_value
            return out

        predictor.predict_single_npy_array.side_effect = predict
        return predictor

    def test_the_result_is_returned_on_the_input_grid(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        nii = _nii(_image(), seg=False)
        predictor = self._predictor()
        seg, logits, softmax = run_inference(nii, predictor)
        self.assertEqual(seg.shape, nii.shape)
        self.assertEqual(seg.orientation, nii.orientation)
        np.testing.assert_allclose(seg.affine, nii.affine)
        self.assertIsNone(logits)
        self.assertIsNone(softmax)

    def test_the_planted_labels_survive_the_round_trip(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        seg, _, _ = run_inference(_nii(_image(), seg=False), self._predictor(label_value=7))
        self.assertEqual(set(np.unique(seg.get_seg_array())), {0, 7})

    def test_the_predictor_receives_the_reversed_spacing(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        nii = _nii(_image(), seg=False)
        predictor = self._predictor()
        run_inference(nii, predictor)
        props = predictor.predict_single_npy_array.call_args[0][1]
        # reorient_PIR defaults to False, so the spacing is the input's own, reversed.
        np.testing.assert_allclose(props["spacing"], nii.zoom[::-1])

    def test_reorient_pir_changes_the_spacing_handed_over(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        nii = _nii(_image(), seg=False)
        predictor = self._predictor()
        run_inference(nii, predictor, reorient_PIR=True)
        props = predictor.predict_single_npy_array.call_args[0][1]
        np.testing.assert_allclose(props["spacing"], nii.reorient(("P", "I", "R")).zoom[::-1])

    def test_the_image_is_passed_with_a_leading_channel_axis(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        nii = _nii(_image(), seg=False)
        predictor = self._predictor()
        run_inference(nii, predictor)
        img = predictor.predict_single_npy_array.call_args[0][0]
        self.assertEqual(img.ndim, 4)
        self.assertEqual(img.shape[0], 1, "one channel for a single input image")

    def test_a_multi_channel_input_is_stacked(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        niis = [_nii(_image(), seg=False), _nii(_image(), seg=False)]
        predictor = self._predictor()
        run_inference(niis, predictor)
        self.assertEqual(predictor.predict_single_npy_array.call_args[0][0].shape[0], 2)

    def test_logits_are_not_implemented(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        with self.assertRaises((NotImplementedError, AssertionError)):
            run_inference(_nii(_image(), seg=False), self._predictor(), logits=True)

    def test_an_unsupported_input_type_is_rejected(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        with self.assertRaises(AssertionError):
            run_inference(5, self._predictor())  # type: ignore[arg-type]

    def test_a_path_must_be_a_nii_gz(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference  # noqa: PLC0415

        with self.assertRaises(AssertionError):
            run_inference("image.png", self._predictor())


@unittest.skipUnless(HAS_TORCH, TORCH_REASON)
class TestRunInferenceOnFile(unittest.TestCase):
    """`run_inference_on_file` orchestrates model loading, inference and output writing.

    Model 85's shipped dataset.json declares two input channels, and the function
    checks that, so the fixtures hand over two images.
    """

    @staticmethod
    def _inputs():
        return [_nii(_image(), seg=False), _nii(_image(), seg=False)]

    def _patched(self, label_value: int = 5):
        from TPTBox.segmentation.nnUnet_utils import inference_api  # noqa: PLC0415

        def fake_run_inference(input_nii, predictor, **kwargs):  # noqa: ARG001
            first = input_nii[0] if isinstance(input_nii, list) else input_nii
            arr = np.zeros(first.shape, dtype=np.uint8)
            arr[..., : arr.shape[-1] // 2] = label_value
            return first.set_array(arr, seg=True), None, None

        # run_inference_on_file imports both names inside its own body, so patching the
        # module they are defined in is what takes effect.
        return (
            mock.patch.object(inference_api, "load_inf_model", return_value=mock.MagicMock()),
            mock.patch.object(inference_api, "run_inference", side_effect=fake_run_inference),
        )

    def test_the_output_file_is_written_with_the_planted_labels(self):
        from TPTBox import to_nii  # noqa: PLC0415
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file  # noqa: PLC0415

        load_patch, run_patch = self._patched(label_value=5)
        with tempfile.TemporaryDirectory() as tmp, load_patch, run_patch:
            out_file = Path(tmp) / "seg.nii.gz"
            niis = self._inputs()
            nii = niis[0]
            run_inference_on_file(85, niis, out_file=out_file, ddevice="cpu", verbose=False)
            self.assertTrue(out_file.exists())
            written = to_nii(out_file, seg=True)
            self.assertEqual(written.shape, nii.shape)
            self.assertEqual(set(np.unique(written.get_seg_array())), {0, 5})

    def test_a_second_call_skips_inference_unless_overridden(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file  # noqa: PLC0415

        load_patch, run_patch = self._patched()
        with tempfile.TemporaryDirectory() as tmp, load_patch, run_patch as run:
            out_file = Path(tmp) / "seg.nii.gz"
            niis = self._inputs()
            run_inference_on_file(85, niis, out_file=out_file, ddevice="cpu", verbose=False)
            self.assertEqual(run.call_count, 1)
            run_inference_on_file(85, niis, out_file=out_file, ddevice="cpu", verbose=False)
            self.assertEqual(run.call_count, 1, "an existing output must short-circuit")
            run_inference_on_file(85, niis, out_file=out_file, override=True, ddevice="cpu", verbose=False)
            self.assertEqual(run.call_count, 2, "override=True must re-run")

    def test_a_label_mapping_is_applied(self):
        from TPTBox import to_nii  # noqa: PLC0415
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file  # noqa: PLC0415

        load_patch, run_patch = self._patched(label_value=5)
        with tempfile.TemporaryDirectory() as tmp, load_patch, run_patch:
            out_file = Path(tmp) / "seg.nii.gz"
            run_inference_on_file(85, self._inputs(), out_file=out_file, mapping={5: 9}, ddevice="cpu", verbose=False)
            self.assertEqual(set(np.unique(to_nii(out_file, seg=True).get_seg_array())), {0, 9})

    def test_the_output_keeps_the_input_grid(self):
        from TPTBox import to_nii  # noqa: PLC0415
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file  # noqa: PLC0415

        load_patch, run_patch = self._patched()
        with tempfile.TemporaryDirectory() as tmp, load_patch, run_patch:
            out_file = Path(tmp) / "seg.nii.gz"
            niis = self._inputs()
            nii = niis[0]
            run_inference_on_file(85, niis, out_file=out_file, ddevice="cpu", verbose=False)
            written = to_nii(out_file, seg=True)
            np.testing.assert_allclose(written.affine, nii.affine, atol=1e-6)
            self.assertEqual(written.orientation, nii.orientation)


if __name__ == "__main__":
    unittest.main()
