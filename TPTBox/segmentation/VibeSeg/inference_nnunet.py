from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from warnings import warn

import numpy as np
import torch

from TPTBox import NII, Image_Reference, Log_Type, Print_Logger, to_nii
from TPTBox.segmentation.VibeSeg.auto_download import download_weights

logger = Print_Logger()
out_base = Path(__file__).parent.parent / "nnUNet/"
_model_path_ = out_base / "nnUNet_results"


def get_ds_info(idx, _model_path: str | Path | None = None, exit_one_fail=True) -> dict:
    if _model_path is not None:
        _model_path = Path(_model_path)
        model_path = _model_path / "nnUNet_results"
        assert model_path.exists(), model_path
    else:
        model_path = _model_path_
    try:
        nnunet_path = next(next(iter(model_path.glob(f"*{idx}*"))).glob("*__nnUNetPlans*"))
    except StopIteration:
        try:
            nnunet_path = next(next(iter(model_path.glob(f"*{idx}*"))).glob("*__nnUNet*ResEnc*"))
        except StopIteration:
            if exit_one_fail:
                Print_Logger().print(f"Please add Dataset {idx} to {model_path}", Log_Type.FAIL)
                model_path.mkdir(exist_ok=True, parents=True)
                sys.exit()
            else:
                return None
    with open(Path(nnunet_path, "dataset.json")) as f:
        ds_info = json.load(f)
    return ds_info


def squash_so_it_fits_in_float16(x: NII):
    m = x.max()
    if m > 10000:
        x /= m / 1000  # new max will be 1000
    return x


def run_inference_on_file(
    idx: int | Path,
    input_nii: list[NII],
    out_file: str | Path | None = None,
    orientation=None,
    override=False,
    gpu=None,
    keep_size=False,
    fill_holes=False,
    logits=False,
    mapping=None,
    crop=False,
    max_folds=None,
    mode="nearest",
    padd: int = 0,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    _model_path=None,
    step_size=0.5,
    memory_base=5000,  # Base memory in MB, default is 5GB
    memory_factor=160,  # prod(shape)*memory_factor / 1000, 160 ~> 30 GB
    memory_max=160000,  # in MB, default is 160GB
    wait_till_gpu_percent_is_free=0.1,
    verbose=True,
) -> tuple[Image_Reference, np.ndarray | None]:
    if _model_path is not None:
        _model_path = Path(_model_path)
        model_path = _model_path / "nnUNet_results"
        assert model_path.exists(), model_path
    else:
        model_path = _model_path_
    if out_file is not None and Path(out_file).exists() and not override:
        return out_file, None

    from TPTBox.segmentation.nnUnet_utils.inference_api import (
        load_inf_model,
        run_inference,
    )

    if isinstance(idx, int):
        download_weights(idx, model_path)
        try:
            nnunet_path = next(next(iter(model_path.glob(f"*{idx:03}*"))).glob("*__nnUNet*ResEnc*"))
        except StopIteration:
            nnunet_path = next(next(iter(model_path.glob(f"*{idx:03}*"))).glob("*__nnUNetPlans*"))
    else:
        nnunet_path = Path(idx)
    assert nnunet_path.exists(), nnunet_path
    folds = sorted([f.name.split("fold_")[-1] for f in nnunet_path.glob("fold*")])
    if max_folds is not None:
        folds = max_folds if isinstance(max_folds, list) else folds[:max_folds]

    # if idx in _unets:
    #    nnunet = _unets[idx]
    # else:
    print("load model", nnunet_path, "; folds", folds) if verbose else None
    with open(Path(nnunet_path, "plans.json")) as f:
        plans_info = json.load(f)
    with open(Path(nnunet_path, "dataset.json")) as f:
        ds_info = json.load(f)
    inference_config = Path(nnunet_path, "inference_config.json")
    if inference_config.exists():
        with open() as f:
            ds_info2 = json.load(f)
            if "model_expected_orientation" in ds_info2:
                ds_info["orientation"] = ds_info2["model_expected_orientation"]
            if "resolution_range" in ds_info2:
                ds_info["resolution_range"] = ds_info2["resolution_range"]

    nnunet = load_inf_model(
        nnunet_path,
        allow_non_final=True,
        use_folds=tuple(folds) if len(folds) != 5 else None,
        gpu=gpu,
        ddevice=ddevice,
        step_size=step_size,
        memory_base=memory_base,
        memory_factor=memory_factor,
        memory_max=memory_max,
        wait_till_gpu_percent_is_free=wait_till_gpu_percent_is_free,
    )

    #    _unets[idx] = nnunet
    if "orientation" in ds_info:
        orientation = ds_info["orientation"]

    zoom = None
    orientation_ref = None
    og_nii = input_nii[0].copy()

    try:
        zoom = ds_info.get("spacing")
        if idx not in [527] and zoom is not None:
            zoom = zoom[::-1]

        zoom = ds_info.get("resolution_range", zoom)
        if zoom is None:
            zoom_ = plans_info["configurations"]["3d_fullres"]["spacing"]
            if all(zoom[0] == z for z in zoom_):
                zoom = zoom_
        # order = plans_info["transpose_backward"]
        ## order2 = plans_info["transpose_forward"]
        # zoom = [zoom[order[0]], zoom[order[1]], zoom[order[2]]][::-1]
        # orientation_ref = ("P", "I", "R")
        # orientation_ref = [
        #    orientation_ref[order[0]],
        #    orientation_ref[order[1]],
        #    orientation_ref[order[2]],
        # ]  # [::-1]

        # zoom_old = zoom_old[::-1]

        zoom = [float(z) for z in zoom]
    except Exception:
        pass
    assert len(ds_info["channel_names"]) == len(input_nii), (
        ds_info["channel_names"],
        len(input_nii),
        "\n",
        nnunet_path,
    )
    if orientation is not None:
        print("orientation", orientation, f"{orientation_ref=}") if verbose else None
        input_nii = [i.reorient(orientation) for i in input_nii]

    if zoom is not None:
        print("rescale", input_nii[0].orientation, f"{zoom=}") if verbose else None
        input_nii = [i.rescale_(zoom, mode=mode) for i in input_nii]
        print(input_nii)
    print("squash to float16") if verbose else None
    input_nii = [squash_so_it_fits_in_float16(i) for i in input_nii]

    if crop:
        crop = input_nii[0].compute_crop(minimum=20)
        input_nii = [i.apply_crop(crop) for i in input_nii]
    if padd != 0:
        p = (padd, padd)
        input_nii = [i.apply_pad([p, p, p], mode="reflect") for i in input_nii]

    seg_nii, uncertainty_nii, softmax_logits = run_inference(input_nii, nnunet, logits=logits)
    if padd != 0:
        seg_nii = seg_nii[padd:-padd, padd:-padd, padd:-padd]

    if mapping is not None:
        seg_nii.map_labels_(mapping)
    if not keep_size:
        seg_nii.resample_from_to_(og_nii, mode=mode)
    if fill_holes:
        seg_nii.fill_holes_()
    if out_file is not None and (not Path(out_file).exists() or override):
        seg_nii.save(out_file)
    del nnunet

    torch.cuda.empty_cache()
    return seg_nii, softmax_logits


idx_models = [100]


def run_VibeSeg(
    img: Path | str | list[Path] | list[NII] | Image_Reference,
    out_path: str | Path | None,
    override=False,
    dataset_id=None,
    gpu: int | None = None,
    logits=False,
    known_idx=idx_models,
    keep_size=False,
    fill_holes=False,
    crop=False,
    max_folds: int | None = None,
    _model_path=None,
    step_size=0.5,
    **_kargs,
):
    if isinstance(out_path, str):
        out_path = Path(out_path)
    if out_path is not None and out_path.exists() and not override:
        logger.print(out_path, "already exists. SKIP!", Log_Type.OK)
        return out_path

    model_path = _model_path if _model_path is not None else _model_path_
    if dataset_id is None:
        for idx in known_idx:
            download_weights(idx)
            try:
                next(next(iter(model_path.glob(f"*{idx}*"))).glob("*__nnUNetPlans*"))
                dataset_id = idx
                break
            except StopIteration:
                pass
        else:
            logger.print(
                f"Could not find model. Download the model an put it into {model_path.absolute()}",
                Log_Type.FAIL,
            )
            return
    else:
        weights_dir = download_weights(dataset_id)
        print("to", weights_dir)
    selected_gpu = gpu
    if gpu is None:
        gpu = "auto"  # type: ignore
    logger.print("run", f"{dataset_id=}, {gpu=}", Log_Type.STAGE)
    ds_info = get_ds_info(dataset_id)
    orientation = ds_info.get("orientation", ("R", "A", "S"))
    if not isinstance(img, Sequence) or isinstance(img, str):
        img = [img]

    if "roi" in ds_info:
        raise NotImplementedError("roi")
    else:
        in_niis = [to_nii(i) for i in img]  # type: ignore
    in_niis = [i.resample_from_to_(in_niis[0]) if i.shape != in_niis[0].shape else i for i in in_niis]
    if (in_niis[0].affine == np.eye(4)).all():
        warn(
            "Your affine matrix is the identity. Make sure that the spacing and orientation is correct. For NAKO VIBE it should be 1.40625 mm for R/L and A/P and 3 mm S/I. For UKBB R/L and A/P should be around 2.2 mm",
            stacklevel=3,
        )
    return run_inference_on_file(
        dataset_id,
        in_niis,
        override=override,
        out_file=out_path,
        gpu=selected_gpu,
        orientation=orientation,
        logits=logits,
        keep_size=keep_size,
        fill_holes=fill_holes,
        crop=crop,
        max_folds=max_folds,
        step_size=step_size,
        **_kargs,
    )[0]
