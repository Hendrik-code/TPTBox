from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from TPTBox import NII, Image_Reference, Log_Type, Print_Logger, to_nii
from TPTBox.segmentation.VibeSeg.auto_download import download_weights

logger = Print_Logger()
out_base = Path(__file__).parent.parent / "nnUNet/"
_model_path_ = out_base / "nnUNet_results"

# Opt-in cache of loaded predictors (enable via cache_model=True). Keyed by model identity plus
# the device/runtime settings that affect the loaded predictor, so repeated inference (e.g. a loop
# over many files with the same model) reuses the in-memory model instead of reloading weights from
# disk and re-uploading them to the GPU on every call.
_model_cache: dict = {}


def get_ds_info(idx: int, _model_path: str | Path | None = None, exit_one_fail: bool = True, logger=logger) -> dict:
    """Load and return the ``dataset.json`` for the model with the given dataset index.

    Args:
        idx: Numeric dataset identifier (e.g. ``100`` for Dataset100).
        _model_path: Optional base directory containing ``nnUNet_results``. If
            ``None``, the bundled default path is used.
        exit_one_fail: If ``True``, call :func:`sys.exit` when the dataset is
            not found; otherwise return ``None``.

    Returns:
        Parsed ``dataset.json`` dictionary for the requested dataset.
    """
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
                logger.print(f"Please add Dataset {idx} to {model_path}", Log_Type.FAIL)
                model_path.mkdir(exist_ok=True, parents=True)
                sys.exit()
            else:
                return None
    with open(Path(nnunet_path, "dataset.json")) as f:
        ds_info = json.load(f)
    return ds_info


def squash_so_it_fits_in_float16(x: NII) -> NII:
    """Scale image intensities so the maximum fits within the float16 range.

    Args:
        x: Input NIfTI image. Modified in-place when rescaling is necessary.

    Returns:
        The (potentially rescaled) image, with maximum capped at 1000 when the
        original maximum exceeds 10000.
    """
    m = x.max()
    if m > 10000:
        x /= m / 1000  # new max will be 1000
    return x


def run_inference_on_file(
    idx: int | Path,
    input_nii: list[NII],
    out_file: str | Path | None = None,
    orientation=None,
    override: bool = False,
    gpu=None,
    keep_size: bool = False,
    fill_holes: bool = False,
    logits: bool = False,
    mapping=None,
    crop: bool = False,
    max_folds=None,
    mode: str = "nearest",
    padd: int = 0,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    model_path=None,
    step_size: float = 0.5,
    memory_base: float | None = None,  # Base memory in MB, default is 5GB
    memory_factor: float | None = None,  # prod(shape)*memory_factor / 1000, 160 ~> 30 GB
    memory_max: int = 990000,  # in MB, default is 990GB (so it is most likely ignored and replaced by Max Memory of the GPU)
    wait_till_gpu_percent_is_free: float = 0.1,
    tile_batch_size: int = 1,
    verbose: bool = True,
    auto_download: bool = False,
    cache_model: bool = False,
    _key_ResEnc: str = "__nnUNet*ResEnc",
    fail_on_missing_memory=False,
    logger=logger,
) -> tuple[Image_Reference, np.ndarray | None]:
    """Load a VibeSeg model and run inference on the supplied NIfTI images.

    Args:
        idx: Either an integer dataset ID (model weights are auto-downloaded) or
            a :class:`~pathlib.Path` pointing directly to a trained model folder.
        input_nii: List of input NIfTI images (one per model input channel).
        out_file: Optional path to save the segmentation. If the file already
            exists and ``override`` is ``False``, returns early.
        orientation: Three-letter orientation code to reorient inputs before
            inference (e.g. ``("R", "A", "S")``). Inferred from the model's
            ``dataset.json`` when ``None``.
        override: Overwrite an existing ``out_file`` when ``True``.
        gpu: GPU index to use. ``None`` lets the loader decide automatically.
        keep_size: If ``True``, do not resample the segmentation back to the
            original image size.
        fill_holes: If ``True``, fill holes in the segmentation mask after
            inference.
        logits: If ``True``, also return the raw softmax logits array.
        mapping: Optional label remapping dict applied to the segmentation after
            inference.
        crop: If ``True``, crop the input to its foreground bounding box before
            inference and revert afterwards.
        max_folds: Maximum number of folds to average. Accepts an ``int`` (use
            the first N folds) or a list of specific fold names.
        mode: Interpolation mode used when resampling the segmentation back to
            the original space (default ``"nearest"``).
        padd: Padding (in voxels) added on all sides before inference and
            removed afterwards.
        ddevice: Device to run inference on (``"cuda"``, ``"cpu"``, or
            ``"mps"``).
        model_path: Optional path to a directory containing ``nnUNet_results``.
            Uses the bundled default when ``None``.
        step_size: Sliding-window step size fraction (see
            :class:`~TPTBox.segmentation.nnUnet_utils.predictor.nnUNetPredictor`).
        memory_base: Base GPU memory reservation in MB.
        memory_factor: Memory scaling factor per million voxels.
        memory_max: Hard cap on assumed GPU memory in MB.
        wait_till_gpu_percent_is_free: Minimum free GPU fraction to require
            before starting inference.
        tile_batch_size: Number of sliding-window tiles to run per network
            forward pass. ``1`` (default) keeps the original per-tile behaviour;
            larger values batch tiles to better saturate the GPU at the cost of
            higher peak memory.
        verbose: Print progress information.
        cache_model: If ``True``, keep the loaded predictor in a process-wide
            cache and reuse it on subsequent calls with identical model and
            device/runtime settings. Avoids reloading weights from disk and
            re-uploading them to the GPU when segmenting many files in a loop, at
            the cost of holding the model in GPU memory between calls. The GPU
            cache is also left warm (no ``empty_cache``) so the allocator can
            reuse buffers across images.

    Returns:
        A tuple ``(seg_nii, softmax_logits)`` where ``seg_nii`` is the
        segmentation (as a :class:`~TPTBox.NII` or file path) and
        ``softmax_logits`` is the raw logit array or ``None``.
    """
    if model_path is None:
        auto_download = True
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.name != "nnUNet_results":
            model_path = model_path / "nnUNet_results"
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
        if auto_download:
            download_weights(idx, model_path)
        try:
            nnunet_path = next(next(iter(model_path.glob(f"*{idx:03}*"))).glob(f"*{_key_ResEnc}*"))
        except StopIteration as e:
            try:
                nnunet_path = next(next(iter(model_path.glob(f"*{idx:03}*"))).glob("*__nnUNetPlans*"))
            except StopIteration:
                logger.on_fail(model_path, (f"*{idx:03}*"))
                raise e from None
    else:
        nnunet_path = Path(idx)
    assert nnunet_path.exists(), nnunet_path
    folds = sorted([f.name.split("fold_")[-1] for f in nnunet_path.glob("fold*")])
    if max_folds is not None:
        folds = max_folds if isinstance(max_folds, list) else folds[:max_folds]

    # if idx in _unets:
    #    nnunet = _unets[idx]
    # else:
    logger.print("load model", nnunet_path, "; folds", folds) if verbose else None
    with open(Path(nnunet_path, "plans.json")) as f:
        plans_info = json.load(f)
    with open(Path(nnunet_path, "dataset.json")) as f:
        ds_info = json.load(f)
    inference_config = Path(nnunet_path, "inference_config.json")
    if inference_config.exists():
        with open(inference_config) as f:
            ds_info2 = json.load(f)
            if "model_expected_orientation" in ds_info2:
                ds_info["orientation"] = ds_info2["model_expected_orientation"]
            if "resolution_range" in ds_info2:
                ds_info["resolution_range"] = ds_info2["resolution_range"]
            if "labels" in ds_info2:
                ds_info["labels_mapping"] = ds_info2["labels"]

    if memory_base is None:
        memory_base = float(ds_info.get("memory_base", 5000))
    if memory_factor is None:
        memory_factor = float(ds_info.get("memory_factor", 160))

    use_folds_arg = tuple(folds) if len(folds) != 5 else None
    # Include every setting that changes the loaded predictor so a cache hit is always equivalent
    # to a fresh load; differing settings simply miss the cache and reload.
    cache_key = (
        str(nnunet_path),
        use_folds_arg,
        ddevice,
        gpu,
        step_size,
        memory_base,
        memory_factor,
        memory_max,
        wait_till_gpu_percent_is_free,
        tile_batch_size,
    )
    nnunet = _model_cache.get(cache_key) if cache_model else None
    if nnunet is None:
        nnunet = load_inf_model(
            nnunet_path,
            allow_non_final=True,
            use_folds=use_folds_arg,
            gpu=gpu,
            ddevice=ddevice,
            step_size=step_size,
            memory_base=memory_base,
            memory_factor=memory_factor,
            memory_max=memory_max,
            wait_till_gpu_percent_is_free=wait_till_gpu_percent_is_free,
            tile_batch_size=tile_batch_size,
            fail_on_missing_memory=fail_on_missing_memory,
        )
        if cache_model:
            _model_cache[cache_key] = nnunet
    if "orientation" in ds_info:
        orientation = ds_info["orientation"]

    zoom = None
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
        logger.print("orientation", orientation, f"from {input_nii[0].orientation}") if verbose else None
        input_nii = [i.reorient(orientation) for i in input_nii]

    if zoom is not None:
        logger.print("rescale", f"{zoom=} from {input_nii[0].zoom}") if verbose else None
        input_nii = [i.rescale_(zoom, mode=mode, verbose=True) for i in input_nii]
        logger.print(input_nii)
    logger.print("squash to float16") if verbose else None
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
    if "labels_mapping" in ds_info:
        from TPTBox.core.vert_constants import list_of_all_enums

        mapping_ = ds_info["labels_mapping"]
        unknown_strings: dict[str, int] = {"max": seg_nii.max() + 1, "Intervertebral_Disc": 100}
        mapping = {}

        def to_int(a: str, k: None | int = None):
            if a in unknown_strings:
                return unknown_strings[a]
            try:
                return int(a)
            except Exception:
                pass

            for enum_ in list_of_all_enums:
                try:
                    return enum_[a].value
                except Exception:
                    pass
            if k is not None and k not in unknown_strings.values():
                return k
            unknown_strings[a] = unknown_strings["max"]
            unknown_strings["max"] += 1
            if unknown_strings["max"] == 100:
                unknown_strings["max"] += 1

            return unknown_strings[a]

        for k, v in mapping_.items():
            key = to_int(k)
            value = to_int(v, key)
            if k != value:
                mapping[k] = value
            unknown_strings[v] = value
        logger.print(f"{unknown_strings}")
        logger.print(f"{mapping=}")
        seg_nii.map_labels_(mapping)
    if out_file is not None and (not Path(out_file).exists() or override):
        seg_nii.set_dtype("smallest_uint").save(out_file)
    if not cache_model:
        # When caching we keep the predictor alive (it stays referenced by _model_cache, so del
        # would not free it anyway) and leave the CUDA cache warm so the next image reuses buffers.
        del nnunet
        torch.cuda.empty_cache()
    return seg_nii, softmax_logits


idx_models = [100]


def run_VibeSeg(
    img: Path | str | list[Path] | list[NII] | Image_Reference,
    out_path: str | Path | None,
    override: bool = False,
    dataset_id: int | None = None,
    gpu: int | None = None,
    logits: bool = False,
    known_idx: list[int] = idx_models,
    keep_size: bool = False,
    fill_holes: bool = False,
    crop: bool = False,
    max_folds: int | None = None,
    _model_path=None,
    step_size: float = 0.5,
    logger: Print_Logger = logger,
    **_kargs,
) -> NII | Path | None:
    """High-level entry point for running VibeSeg body-composition segmentation.

    Automatically downloads model weights when needed, selects the appropriate
    dataset, and delegates to :func:`run_inference_on_file`.

    Args:
        img: Input image(s). Accepts a single path/NII, or a list of paths/NIIs
            for multi-channel models.
        out_path: Output path for the segmentation file. Skipped when
            ``None``; skips inference when the file exists and
            ``override`` is ``False``.
        override: Overwrite existing output file.
        dataset_id: Explicit dataset ID to use. When ``None``, iterates
            through ``known_idx`` to find the first available model.
        gpu: GPU device index. ``None`` selects automatically.
        logits: If ``True``, return raw softmax logits in addition to the
            label map.
        known_idx: List of dataset IDs to probe when ``dataset_id`` is
            ``None``.
        keep_size: Skip resampling the segmentation back to the input
            resolution.
        fill_holes: Fill holes in the output segmentation.
        crop: Crop input images to their foreground bounding box.
        max_folds: Limit the number of folds used for ensemble averaging.
        _model_path: Override for the default model weights directory.
        step_size: Sliding-window step size fraction.
        **_kargs: Additional keyword arguments forwarded to
            :func:`run_inference_on_file`.

    Returns:
        Path or :class:`~TPTBox.NII` of the saved segmentation, or ``None``
        on failure.
    """
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
        logger.print("to", weights_dir)
    selected_gpu = gpu
    if gpu is None:
        gpu = "auto"  # type: ignore
    logger.print("run", f"{dataset_id=}, {gpu=}", Log_Type.STAGE)
    ds_info = get_ds_info(dataset_id, logger=logger)
    orientation = ds_info.get("orientation", ("R", "A", "S"))
    if not isinstance(img, Sequence) or isinstance(img, str):
        img = [img]

    if "roi" in ds_info:
        raise NotImplementedError("roi")
    else:
        in_niis = [to_nii(i) for i in img]  # type: ignore
    in_niis = [i.resample_from_to_(in_niis[0]) if i.shape != in_niis[0].shape else i for i in in_niis]
    if (in_niis[0].affine == np.eye(4)).all():
        logger.on_warning(
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
        logger=logger,
        **_kargs,
    )[0]
