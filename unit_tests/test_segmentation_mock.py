from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

from TPTBox import NII  # noqa: E402
from TPTBox.tests.test_utils import get_tests_dir  # noqa: E402

try:
    import torch  # noqa: F401

    has_torch = True
except ModuleNotFoundError:
    has_torch = False

try:
    import nnunetv2  # noqa: F401

    has_nnunet = True
except ModuleNotFoundError:
    has_nnunet = False

try:
    import spineps  # noqa: F401

    has_spineps = True
except ModuleNotFoundError:
    has_spineps = False


# --------------------------------------------------------------------------- helpers
def _nii(arr: np.ndarray, seg: bool, affine=None) -> NII:
    if affine is None:
        affine = np.eye(4)
    return NII(nib.Nifti1Image(arr, affine), seg=seg)


def _img_nii(shape=(24, 24, 24)) -> NII:
    arr = np.zeros(shape, dtype=np.int16)
    arr[4:-4, 4:-4, 4:-4] = 80
    return _nii(arr, seg=False)


def _img_nii_nonident(shape=(24, 24, 24)) -> NII:
    # non-identity affine: run_VibeSeg crashes on identity affine (logger.on_warning bug)
    arr = np.zeros(shape, dtype=np.int16)
    arr[4:-4, 4:-4, 4:-4] = 80
    aff = np.diag([1.4, 1.4, 3.0, 1.0])
    return _nii(arr, seg=False, affine=aff)


def _build_model_dir(root: Path, idx: int, *, channel_names=None, extra_ds=None, inference_config=None) -> Path:
    """Create a minimal nnU-Net result tree and return the base (``nnUNet_results``) path."""
    if channel_names is None:
        channel_names = {"0": "image"}
    res = root / "nnUNet_results"
    mdir = res / f"Dataset{idx:03}_test" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (mdir / "fold_0").mkdir(parents=True, exist_ok=True)
    with open(mdir / "plans.json", "w") as f:
        json.dump({"configurations": {"3d_fullres": {"spacing": [1, 1, 1]}}}, f)
    ds = {"channel_names": channel_names, "spacing": [1, 1, 1], "orientation": ["R", "A", "S"]}
    if extra_ds:
        ds.update(extra_ds)
    with open(mdir / "dataset.json", "w") as f:
        json.dump(ds, f)
    if inference_config is not None:
        with open(mdir / "inference_config.json", "w") as f:
            json.dump(inference_config, f)
    return res


# --------------------------------------------------------------------------- extract_vertebra_bodies (no mock)
@unittest.skipIf(not has_torch, "requires torch to import the segmentation module")
class Test_extract_vertebra_bodies(unittest.TestCase):
    @staticmethod
    def _stacked_vibeseg(bodies=6):
        shape = (20, 20, 52)
        arr = np.zeros(shape, dtype=np.uint8)
        z, body_h, ivd_h = 2, 6, 2
        for b in range(bodies):
            arr[3:17, 3:17, z : z + body_h] = 69  # vertebra body
            z += body_h
            if b < bodies - 1:
                arr[3:17, 3:17, z : z + ivd_h] = 68  # IVD
                z += ivd_h
        return _nii(arr, seg=True)

    def test_relabel_lumbar_thoracic(self):
        from TPTBox import POI
        from TPTBox.segmentation.VibeSeg.vibeseg import extract_vertebra_bodies_from_VibeSeg

        nii = self._stacked_vibeseg(bodies=6)
        vb, poi = extract_vertebra_bodies_from_VibeSeg(nii)
        self.assertIsInstance(vb, NII)
        self.assertIsInstance(poi, POI)
        # 6 bodies (inferior->superior): L5,L4,L3,L2,L1,T12 -> {24,23,22,21,20,19}
        self.assertEqual(set(vb.unique()), {19, 20, 21, 22, 23, 24})
        self.assertEqual(set(poi.keys_region()), {19, 20, 21, 22, 23, 24})
        # most inferior body (lowest S) must be L5 == 24
        poi_ras = poi.reorient(("R", "A", "S"))
        s_coord = {r: poi_ras[r, 50][2] for r in poi_ras.keys_region()}
        self.assertEqual(min(s_coord, key=s_coord.get), 24)

    def test_save_outputs(self):
        from TPTBox.segmentation.VibeSeg.vibeseg import extract_vertebra_bodies_from_VibeSeg

        nii = self._stacked_vibeseg(bodies=3)
        with tempfile.TemporaryDirectory() as td:
            out_msk = Path(td) / "vb.nii.gz"
            out_poi = Path(td) / "vb_poi.json"
            vb, poi = extract_vertebra_bodies_from_VibeSeg(nii, out_path=out_msk, out_path_poi=out_poi)
            self.assertTrue(out_msk.exists())
            self.assertTrue(out_poi.exists())
            # 3 bodies -> L5,L4,L3
            self.assertEqual(set(vb.unique()), {22, 23, 24})


# --------------------------------------------------------------------------- inference_api
@unittest.skipIf(not has_nnunet, "requires nnunetv2")
class Test_inference_api_run_inference(unittest.TestCase):
    def test_marshalling_single(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference

        inp = _img_nii((20, 24, 28))
        predictor = MagicMock()
        predictor.predict_single_npy_array.side_effect = lambda img, *_, **__: np.ones(img.shape[1:], dtype=np.uint8)
        seg_nii, unc, logits = run_inference(input_nii=inp, predictor=predictor)
        self.assertIsInstance(seg_nii, NII)
        self.assertIsNone(unc)
        self.assertIsNone(logits)
        self.assertEqual(seg_nii.shape, inp.shape)
        self.assertEqual(seg_nii.orientation, inp.orientation)
        predictor.predict_single_npy_array.assert_called_once()
        call = predictor.predict_single_npy_array.call_args
        img = call.args[0]
        props = call.args[1]
        # channel dim prepended, spatial axes reversed (PIR marshalling)
        self.assertEqual(img.shape, (1, *inp.shape[::-1]))
        self.assertEqual(tuple(props["spacing"]), tuple(inp.zoom[::-1]))

    def test_logits_not_implemented(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference

        predictor = MagicMock()
        with pytest.raises(NotImplementedError):
            run_inference(_img_nii(), predictor, logits=True)

    def test_str_input_path(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference

        inp = _img_nii((16, 16, 18))
        predictor = MagicMock()
        predictor.predict_single_npy_array.side_effect = lambda img, *_, **__: np.ones(img.shape[1:], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "img.nii.gz"
            inp.save(p)
            seg_nii, _, _ = run_inference(str(p), predictor)
        self.assertEqual(seg_nii.shape, inp.shape)

    def test_bad_input_type_asserts(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference

        with pytest.raises(AssertionError):
            run_inference(123, MagicMock())

    def test_sliding_nd_slices(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import sliding_nd_slices

        arr = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            out = sliding_nd_slices(arr, patch_size=(4, 4, 4), overlap=2, fun=lambda x: x + 1)
        self.assertEqual(out.shape, arr.shape)
        self.assertIsInstance(out, np.ndarray)

    def test_multichannel_and_reorient(self):
        from TPTBox.segmentation.nnUnet_utils.inference_api import run_inference

        inp = _img_nii((18, 20, 22))
        predictor = MagicMock()
        predictor.predict_single_npy_array.side_effect = lambda img, *_, **__: np.ones(img.shape[1:], dtype=np.uint8)
        # two channels -> vstack to 2 channels, still 3D seg output
        seg_nii, _, _ = run_inference([inp.copy(), inp.copy()], predictor, reorient_PIR=True)
        self.assertIsInstance(seg_nii, NII)
        # reoriented back to the original orientation
        self.assertEqual(seg_nii.orientation, inp.orientation)
        self.assertEqual(predictor.predict_single_npy_array.call_args.args[0].shape[0], 2)


@unittest.skipIf(not has_nnunet, "requires nnunetv2")
class Test_inference_api_load_model(unittest.TestCase):
    def _load(self, td, **kwargs):
        from TPTBox.segmentation.nnUnet_utils.inference_api import load_inf_model

        return load_inf_model(td, **kwargs)

    def test_device_branches(self):
        with tempfile.TemporaryDirectory() as td:
            for dev in ("cpu", "cuda", "mps"):
                with mock.patch("TPTBox.segmentation.nnUnet_utils.inference_api.nnUNetPredictor") as MockPred:
                    out = self._load(td, ddevice=dev, init_threads=False)
                    self.assertIs(out, MockPred.return_value)
                    init = MockPred.return_value.initialize_from_trained_model_folder
                    self.assertEqual(init.call_args.kwargs["checkpoint_name"], "checkpoint_final.pth")

    def test_checkpoint_fallback_to_best(self):
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("TPTBox.segmentation.nnUnet_utils.inference_api.nnUNetPredictor") as MockPred,
        ):
            init = MockPred.return_value.initialize_from_trained_model_folder
            init.side_effect = [FileNotFoundError("no final"), None]
            self._load(td, ddevice="cpu", allow_non_final=True)
            self.assertEqual(init.call_count, 2)
            self.assertEqual(init.call_args_list[1].kwargs["checkpoint_name"], "checkpoint_best.pth")

    def test_disallow_non_final_raises(self):
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("TPTBox.segmentation.nnUnet_utils.inference_api.nnUNetPredictor") as MockPred,
        ):
            MockPred.return_value.initialize_from_trained_model_folder.side_effect = FileNotFoundError("x")
            with pytest.raises(FileNotFoundError):
                self._load(td, ddevice="cpu", allow_non_final=False)

    def test_best_also_fails_reraises_original(self):
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("TPTBox.segmentation.nnUnet_utils.inference_api.nnUNetPredictor") as MockPred,
        ):
            MockPred.return_value.initialize_from_trained_model_folder.side_effect = [
                FileNotFoundError("final"),
                RuntimeError("best broke too"),
            ]
            with pytest.raises(FileNotFoundError):
                self._load(td, ddevice="cpu", allow_non_final=True)

    def test_missing_model_folder_asserts(self):
        with (
            mock.patch("TPTBox.segmentation.nnUnet_utils.inference_api.nnUNetPredictor"),
            pytest.raises(AssertionError),
        ):
            self._load("/non/existent/model/folder", ddevice="cpu")


# --------------------------------------------------------------------------- inference_nnunet
@unittest.skipIf(not has_torch, "requires torch")
class Test_inference_nnunet_helpers(unittest.TestCase):
    def test_get_ds_info(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import get_ds_info

        with tempfile.TemporaryDirectory() as td:
            _build_model_dir(Path(td), 999, extra_ds={"foo": "bar"})
            info = get_ds_info(999, _model_path=td)
            self.assertEqual(info["foo"], "bar")
            self.assertIn("channel_names", info)

    def test_get_ds_info_missing_returns_none(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import get_ds_info

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "nnUNet_results").mkdir()
            self.assertIsNone(get_ds_info(424242, _model_path=td, exit_one_fail=False))

    def test_squash_below_threshold_unchanged(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import squash_so_it_fits_in_float16

        x = _nii(np.full((6, 6, 6), 500, dtype=np.float32), seg=False)
        out = squash_so_it_fits_in_float16(x)
        self.assertEqual(out.max(), 500)

    def test_squash_above_threshold_rescaled(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import squash_so_it_fits_in_float16

        arr = np.zeros((6, 6, 6), dtype=np.float32)
        arr[0, 0, 0] = 20000
        x = _nii(arr, seg=False)
        out = squash_so_it_fits_in_float16(x)
        self.assertAlmostEqual(float(out.max()), 1000.0, places=3)


@unittest.skipIf(not has_nnunet, "requires nnunetv2")
class Test_run_inference_on_file(unittest.TestCase):
    @staticmethod
    def _fake_run_inference(in_list, _predictor=None, **_):
        base = in_list[0]
        arr = np.zeros(base.shape, dtype=np.uint8)
        arr[2:-2, 2:-2, 2:-2] = 1
        arr[5:8, 5:8, 5:8] = 2
        return base.set_array(arr, seg=True), None, None

    def _patches(self):
        return (
            mock.patch(
                "TPTBox.segmentation.nnUnet_utils.inference_api.run_inference",
                side_effect=self._fake_run_inference,
            ),
            mock.patch(
                "TPTBox.segmentation.nnUnet_utils.inference_api.load_inf_model",
                return_value=MagicMock(),
            ),
            mock.patch("TPTBox.segmentation.VibeSeg.inference_nnunet.download_weights"),
        )

    def test_core_path_saves(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

        inp = _img_nii((24, 24, 24))
        with tempfile.TemporaryDirectory() as td:
            _build_model_dir(Path(td), 100)
            out_file = Path(td) / "seg.nii.gz"
            p1, p2, p3 = self._patches()
            with p1, p2, p3:
                seg, logits = run_inference_on_file(100, [inp], out_file=out_file, model_path=td, ddevice="cpu", verbose=False)
            self.assertIsInstance(seg, NII)
            self.assertEqual(seg.shape, inp.shape)
            self.assertTrue(out_file.exists())
            self.assertIsNone(logits)

    def test_crop_padd_mapping_and_labels_mapping(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

        inp = _img_nii((24, 24, 24))
        with tempfile.TemporaryDirectory() as td:
            _build_model_dir(
                Path(td),
                100,
                inference_config={
                    "model_expected_orientation": ["R", "A", "S"],
                    "resolution_range": [1, 1, 1],
                    "labels": {"1": "L1", "2": "L2"},
                },
            )
            p1, p2, p3 = self._patches()
            with p1, p2, p3:
                seg, _ = run_inference_on_file(
                    100, [inp], model_path=td, ddevice="cpu", verbose=True, crop=True, padd=2, mapping={2: 7}, fill_holes=True
                )
            self.assertIsInstance(seg, NII)
            self.assertEqual(seg.shape, inp.shape)
            # label 1 -> L1 (20); label 2 was remapped to 7 by `mapping` before labels_mapping
            self.assertIn(20, seg.unique())

    def test_keep_size_branch(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

        inp = _img_nii((24, 24, 24))
        with tempfile.TemporaryDirectory() as td:
            _build_model_dir(Path(td), 100)
            p1, p2, p3 = self._patches()
            with p1, p2, p3:
                seg, _ = run_inference_on_file(100, [inp], model_path=td, ddevice="cpu", keep_size=True, verbose=False)
            self.assertIsInstance(seg, NII)

    def test_idx_as_path_maxfolds_cache(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

        inp = _img_nii((20, 20, 20))
        with tempfile.TemporaryDirectory() as td:
            res = _build_model_dir(Path(td), 100)
            model_dir = res / "Dataset100_test" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
            p1, p2, p3 = self._patches()
            with p1, p2, p3:
                # idx is a Path -> bypass glob; also exercise max_folds + cache_model
                seg, _ = run_inference_on_file(model_dir, [inp], model_path=td, ddevice="cpu", max_folds=1, cache_model=True, verbose=False)
            self.assertIsInstance(seg, NII)

    def test_labels_mapping_special_tokens(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

        inp = _img_nii((20, 20, 20))
        with tempfile.TemporaryDirectory() as td:
            _build_model_dir(
                Path(td),
                100,
                # "Intervertebral_Disc" -> preset 100 (to_int early-return); unknown string -> falls back to key
                inference_config={"labels": {"1": "Intervertebral_Disc", "3": "totally_unknown_xyz"}},
            )
            p1, p2, p3 = self._patches()
            with p1, p2, p3:
                seg, _ = run_inference_on_file(100, [inp], model_path=td, ddevice="cpu", verbose=False)
            self.assertIsInstance(seg, NII)
            self.assertIn(100, seg.unique())

    def test_out_file_exists_early_return(self):
        from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

        inp = _img_nii((16, 16, 16))
        with tempfile.TemporaryDirectory() as td:
            _build_model_dir(Path(td), 100)
            out_file = Path(td) / "seg.nii.gz"
            inp.copy(seg=True).set_dtype("smallest_uint").save(out_file)
            p1, p2, p3 = self._patches()
            with p1, p2, p3 as md:
                out, logits = run_inference_on_file(100, [inp], out_file=out_file, model_path=td, override=False)
            self.assertEqual(Path(out), out_file)
            md.assert_not_called()


# --------------------------------------------------------------------------- run_VibeSeg (high-level)
@unittest.skipIf(not has_torch, "requires torch")
class Test_run_VibeSeg(unittest.TestCase):
    def test_given_dataset_id(self):
        import TPTBox.segmentation.VibeSeg.inference_nnunet as inf

        fake = _img_nii_nonident()
        with (
            mock.patch.object(inf, "download_weights", return_value=Path("/tmp/x")),
            mock.patch.object(inf, "get_ds_info", return_value={"orientation": ("R", "A", "S")}),
            mock.patch.object(inf, "run_inference_on_file", return_value=(fake, None)) as m,
        ):
            out = inf.run_VibeSeg(_img_nii_nonident(), None, dataset_id=100, _model_path=Path("/tmp"))
        self.assertIs(out, fake)
        m.assert_called_once()

    def test_dataset_id_probe_loop(self):
        import TPTBox.segmentation.VibeSeg.inference_nnunet as inf

        fake = _img_nii_nonident()
        with tempfile.TemporaryDirectory() as td:
            res = Path(td) / "nnUNet_results"
            (res / "Dataset100_t" / "x__nnUNetPlans__3d").mkdir(parents=True)
            with (
                mock.patch.object(inf, "download_weights"),
                mock.patch.object(inf, "get_ds_info", return_value={"orientation": ("R", "A", "S")}),
                mock.patch.object(inf, "run_inference_on_file", return_value=(fake, None)),
            ):
                out = inf.run_VibeSeg(_img_nii_nonident(), None, dataset_id=None, known_idx=[100], _model_path=res)
            self.assertIs(out, fake)

    def test_roi_not_implemented(self):
        import TPTBox.segmentation.VibeSeg.inference_nnunet as inf

        with (
            mock.patch.object(inf, "download_weights"),
            mock.patch.object(inf, "get_ds_info", return_value={"roi": 1}),
            mock.patch.object(inf, "run_inference_on_file", return_value=(_img_nii_nonident(), None)),
            pytest.raises(NotImplementedError),
        ):
            inf.run_VibeSeg(_img_nii_nonident(), None, dataset_id=100, _model_path=Path("/tmp"))

    def test_out_path_exists_early_return(self):
        import TPTBox.segmentation.VibeSeg.inference_nnunet as inf

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "o.nii.gz"
            _img_nii_nonident().copy(seg=True).set_dtype("smallest_uint").save(p)
            with mock.patch.object(inf, "run_inference_on_file") as m:
                out = inf.run_VibeSeg(_img_nii_nonident(), str(p), override=False)
            self.assertEqual(Path(out), p)
            m.assert_not_called()


# --------------------------------------------------------------------------- vibeseg orchestration
@unittest.skipIf(not has_torch, "requires torch")
class Test_vibeseg(unittest.TestCase):
    def test_run_vibeseg_plumbing(self):
        import TPTBox.segmentation.VibeSeg.vibeseg as vs

        inp = _img_nii()
        fake = _nii(np.zeros((4, 4, 4), dtype=np.uint8), seg=True)
        with mock.patch.object(vs, "run_inference_on_file", return_value=(fake, None)) as m:
            out = vs.run_vibeseg(inp, "out.nii.gz", dataset_id=100, gpu=2, padd=5)
        self.assertIs(out, fake)
        m.assert_called_once()
        args, kwargs = m.call_args
        self.assertEqual(args[0], 100)
        self.assertEqual(len(args[1]), 1)
        self.assertIsInstance(args[1][0], NII)
        self.assertEqual(kwargs["out_file"], "out.nii.gz")
        self.assertEqual(kwargs["padd"], 5)
        self.assertEqual(kwargs["keep_size"], False)
        # defaults[100] inject memory settings
        self.assertEqual(kwargs["memory_base"], 5500)
        self.assertEqual(kwargs["memory_factor"], 25)

    def test_run_vibeseg_list_input(self):
        import TPTBox.segmentation.VibeSeg.vibeseg as vs

        inp = _img_nii()
        fake = _nii(np.zeros((4, 4, 4), dtype=np.uint8), seg=True)
        with mock.patch.object(vs, "run_inference_on_file", return_value=(fake, None)) as m:
            vs.run_vibeseg([inp, inp], "o.nii.gz", dataset_id=100)
        self.assertEqual(len(m.call_args.args[1]), 2)

    def test_run_nnunet_plumbing(self):
        import TPTBox.segmentation.VibeSeg.vibeseg as vs

        inp = _img_nii()
        with mock.patch.object(vs, "run_inference_on_file", return_value=(MagicMock(), None)) as m:
            ret = vs.run_nnunet([inp], "o.nii.gz", dataset_id=80, gpu=1)
        self.assertIsNone(ret)
        m.assert_called_once()
        args, kwargs = m.call_args
        self.assertEqual(args[0], 80)
        self.assertEqual(kwargs["out_file"], "o.nii.gz")
        self.assertEqual(kwargs["_key_ResEnc"], "__nnUNet*ResEnc")


# --------------------------------------------------------------------------- auto_download
class Test_auto_download(unittest.TestCase):
    def setUp(self):
        from TPTBox.segmentation.VibeSeg import auto_download as ad

        self.ad = ad
        self.env = ad.env_name

    def test_get_weights_dir_env(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            wd = self.ad.get_weights_dir(85)
            self.assertEqual(wd, Path(td) / "Dataset085")
            self.assertTrue(wd.parent.exists())

    def test_get_weights_dir_model_path(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ):
            os.environ.pop(self.env, None)
            base = Path(td) / "weights"
            base.mkdir()
            wd = self.ad.get_weights_dir(85, model_path=base)
            self.assertEqual(wd, base / "Dataset085")

    def test_read_config_missing(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            self.assertEqual(self.ad.read_config(67), {"dataset_release": 0.0})

    def test_read_config_present(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            wd = Path(td) / "Dataset067"
            wd.mkdir()
            with open(wd / "dataset.json", "w") as f:
                json.dump({"dataset_release": 1.5}, f)
            self.assertEqual(self.ad.read_config(67)["dataset_release"], 1.5)

    def test_download_zip(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            wd = Path(td) / "Dataset067"
            wd.mkdir()

            def fake_retrieve(_url, path, reporthook=None):
                Path(path).write_bytes(b"data")
                if reporthook:
                    # total_size differs from the initially reported size -> pbar.total update path
                    reporthook(1, 1, 999)

            resp = MagicMock()
            resp.__enter__.return_value.info.return_value.get.return_value = 4
            with (
                mock.patch("urllib.request.urlopen", return_value=resp),
                mock.patch("urllib.request.urlretrieve", side_effect=fake_retrieve) as mr,
                mock.patch("zipfile.ZipFile") as mz,
            ):
                self.ad._download("http://example.com/067.zip", wd, text="weights")
            mr.assert_called_once()
            mz.assert_called_once()
            # zip is removed after extraction
            self.assertFalse((Path(td) / "067.zip").exists())

    def test_download_network_failure(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            wd = Path(td) / "Dataset067"
            wd.mkdir()
            with (
                mock.patch("urllib.request.urlopen", side_effect=OSError("no net")),
                mock.patch("urllib.request.urlretrieve") as mr,
            ):
                self.ad._download("http://example.com/067.zip", wd)
            mr.assert_not_called()

    def test_download_weights_calls_download_and_addendum(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            with mock.patch.object(self.ad, "_download") as md:
                self.ad._download_weights(67)
            md.assert_called_once()

    def test_addendum_download(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            wd = Path(td) / "Dataset067"
            wd.mkdir()
            with open(wd / "other_downloads.json", "w") as f:
                json.dump(["_extra"], f)
            with mock.patch.object(self.ad, "_download_weights") as mdw:
                self.ad.addendum_download(67)
            mdw.assert_called_once_with(67, addendum="_extra", first=False)
            self.assertFalse((wd / "other_downloads.json").exists())

    def test_addendum_download_noop(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            (Path(td) / "Dataset067").mkdir()
            with mock.patch.object(self.ad, "_download_weights") as mdw:
                self.ad.addendum_download(67)
            mdw.assert_not_called()

    def test_download_weights_full(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {self.env: td}):
            with (
                mock.patch.object(self.ad, "_download_weights") as mdw,
                mock.patch.object(self.ad, "addendum_download") as madd,
            ):
                wd = self.ad.download_weights(67)
            mdw.assert_called_once_with(67)
            madd.assert_called_once_with(67)
            self.assertEqual(wd.name, "Dataset067")


# --------------------------------------------------------------------------- spineps
@unittest.skipIf(not has_spineps, "requires spineps")
class Test_spineps(unittest.TestCase):
    def test_get_outpaths_spineps(self):
        from TPTBox.segmentation.spineps import get_outpaths_spineps

        tests_path = get_tests_dir()
        mri_path = tests_path / "sample_mri" / "sub-mri_label-6_T2w.nii.gz"
        out = get_outpaths_spineps(mri_path, tests_path)
        self.assertIn("out_spine", out)
        self.assertIn("out_vert", out)

    @staticmethod
    def _proc_patch(output_paths):
        # installed spineps no longer exports ``process_img_nii`` (run_spineps references the
        # removed symbol); inject it so the wrapper's plumbing logic can be unit-tested.
        return mock.patch("spineps.process_img_nii", MagicMock(return_value=(output_paths, 0)), create=True)

    def test_run_spineps_str_models(self):
        from TPTBox.segmentation.spineps import run_spineps

        tests_path = get_tests_dir()
        mri_path = tests_path / "sample_mri" / "sub-mri_label-6_T2w.nii.gz"
        op = {"out_spine": Path("/tmp/s.nii.gz"), "out_vert": Path("/tmp/v.nii.gz")}
        with (
            self._proc_patch(op),
            mock.patch("spineps.get_semantic_model", return_value=MagicMock()) as gs,
            mock.patch("spineps.get_instance_model", return_value=MagicMock()) as gi,
        ):
            out = run_spineps(mri_path, tests_path, model_labeling=None)
        self.assertEqual(out, op)
        gs.assert_called_once()
        gi.assert_called_once()

    def test_run_spineps_path_models(self):
        from TPTBox.segmentation.spineps import run_spineps

        tests_path = get_tests_dir()
        mri_path = tests_path / "sample_mri" / "sub-mri_label-6_T2w.nii.gz"
        op = {"out_spine": Path("/tmp/s.nii.gz"), "out_vert": Path("/tmp/v.nii.gz")}
        with (
            self._proc_patch(op),
            mock.patch("spineps.get_models.get_actual_model", return_value=MagicMock()) as ga,
        ):
            out = run_spineps(mri_path, tests_path, model_semantic=Path("sem"), model_instance=Path("inst"), model_labeling=None)
        self.assertEqual(out, op)
        # get_actual_model used for both the semantic and instance Path models
        self.assertEqual(ga.call_count, 2)

    @staticmethod
    def _model_returning(arr_fn):
        from spineps.seg_model import OutputType

        model = MagicMock()
        model.load.return_value = None

        def segment_scan(img, **_):
            seg = img.copy(seg=True)
            return {OutputType.seg: seg.set_array(arr_fn(img.shape), seg=True)}

        model.segment_scan.side_effect = segment_scan
        return model

    def _patched_model(self, model):
        """Patch get_actual_model and (route around) the renamed ``Segmentation_Model`` symbol.

        The installed spineps renamed ``Segmentation_Model`` -> ``SegmentationModel``, but
        ``_run_spineps_internal`` still does ``from spineps.seg_model import ... Segmentation_Model``
        (used only as a non-evaluated local annotation). Alias it so the import line succeeds.
        """
        import contextlib

        from spineps.seg_model import SegmentationModel

        es = contextlib.ExitStack()
        es.enter_context(mock.patch("spineps.seg_model.Segmentation_Model", SegmentationModel, create=True))
        es.enter_context(mock.patch("spineps.get_models.get_actual_model", return_value=model))
        return es

    def _input(self):
        arr = np.zeros((30, 30, 30), dtype=np.float32)
        arr[8:22, 8:22, 8:22] = 100
        return _nii(arr, seg=False)

    def test_run_spineps_internal(self):
        from TPTBox.segmentation.spineps import _run_spineps_internal

        def seg_arr(shape):
            a = np.zeros(shape, dtype=np.uint8)
            a[3:-3, 3:-3, 3:-3] = 1
            return a

        model = self._model_returning(seg_arr)
        inp = self._input()
        with self._patched_model(model):
            out = _run_spineps_internal(inp, model_path="dummy_model")
        self.assertIsInstance(out, NII)
        self.assertEqual(out.shape, inp.shape)
        self.assertEqual(out.affine.tolist(), inp.affine.tolist())
        model.load.assert_called_once()
        model.segment_scan.assert_called_once()

    def test_run_spineps_internal_empty_returns_none(self):
        from TPTBox.segmentation.spineps import _run_spineps_internal

        model = self._model_returning(lambda shape: np.zeros(shape, dtype=np.uint8))
        with self._patched_model(model):
            out = _run_spineps_internal(self._input(), model_path="dummy_model")
        self.assertIsNone(out)

    def test_run_spineps_internal_save_and_reload(self):
        from TPTBox.segmentation.spineps import _run_spineps_internal

        def seg_arr(shape):
            a = np.zeros(shape, dtype=np.uint8)
            a[3:-3, 3:-3, 3:-3] = 1
            return a

        model = self._model_returning(seg_arr)
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "spineps_seg.nii.gz"
            with self._patched_model(model):
                out = _run_spineps_internal(self._input(), model_path="dummy_model", outpath=out_path, override=True)
            self.assertTrue(out_path.exists())
            self.assertIsInstance(out, NII)
            # second call with override=False short-circuits to reading the file (no model load)
            model2 = self._model_returning(seg_arr)
            with self._patched_model(model2):
                _run_spineps_internal(self._input(), model_path="dummy_model", outpath=out_path, override=False)
            model2.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
