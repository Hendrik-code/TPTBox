import hashlib
import json
import random
import sys
from functools import partial
from math import ceil
from multiprocessing.pool import Pool as Pool_type
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from TPTBox import BIDS_FILE, NII, Print_Logger, to_nii
from TPTBox.core.internal.elastic_deform import deformed_nii

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parents[1]))

logger = Print_Logger()


def run(p):
    p()


#######################################################
def set_up_dataset(
    idx: int,
    dataset_mapping: dict[str, int],
    spacing: tuple[float, ...],
    is_ct=False,
    orientation=("R", "A", "S"),
    turn_on_mirroring=False,
    turn_on_data_aug_5=False,
    nn_trainier: Literal[
        "nnUNetTrainer",
        "nnUNetTrainerNoMirroring",
        "nnUNetTrainerDA5",
        "nnUNetTrainerDAExtGPU",
    ]
    | None = None,
    AUGLAB_PARAMS_GPU_JSON="transform_params_gpu_default01-23.json",
    ignore=False,
    num_input=1,
    base="/DATA/NAS/FASTDATA/robert/nnUNet",
    **setting,
):
    out_base = Path(base, f"nnUNet_raw/Dataset{idx:03}/")
    Path(out_base).mkdir(exist_ok=True, parents=True)

    files = {"0": "ct"} if is_ct else {"0": "any"}
    for i in range(1, num_input):
        files[str(i)] = "any"
    dataset_mapping["background"] = 0

    if ignore:
        dataset_mapping["ignore"] = len(dataset_mapping)  # "ignore": 999
    data = {
        "channel_names": files,
        "labels": dataset_mapping,
        "numTraining": None,
        "file_ending": ".nii.gz",
        "reference": "deep-spine.de",
        "licence": "https://github.com/robert-graf/VibeSegmentator",
        # "regions_class_order": [i for i in sorted(dataset_mapping.values()) if i not in [0]],
        "dataset_release": "1.0",
        "orientation": orientation,
        "nnUNetTrainer": nn_trainier,
        **setting,
    }
    if nn_trainier == "nnUNetTrainerDAExtGPU":
        data["AUGLAB_PARAMS_GPU_JSON"] = AUGLAB_PARAMS_GPU_JSON
    if turn_on_mirroring:
        data["turn_on_mirroring"] = turn_on_mirroring
    if turn_on_data_aug_5:
        data["turn_on_data_aug_5"] = turn_on_data_aug_5

    if spacing is not None and -1 not in spacing:
        # data["spacing"] = tuple(str(i) for i in spacing)[::-1]
        data["resolution_range"] = tuple(str(i) for i in spacing)

    return data, out_base


def add_file(
    p: Pool_type,
    img: Path | list[Path] | str | list[str],
    seg: Path,
    dataset_settings: dict,
    root: Path,
    target_height_half=None,
    defrom=False,
    axis="S",
    deform_factor=1.0,
    defrom_count=1,
    mirror=None,
    degeneration_count=0,
    mapping=None,
    auto_crop=None,
    ignore_crop=None,
    defrom_p=1.0,
    **qargs,  # noqa: ARG001
):
    if p is not None:
        return partial(
            _add_file_async,
            img,
            seg,
            dataset_settings,
            root,
            target_height_half,
            defrom,
            axis,
            deform_factor,
            defrom_count,
            mirror,
            degeneration_count,
            mapping=mapping,
            auto_crop=auto_crop,
            ignore_crop=ignore_crop,
            defrom_p=defrom_p,
        )
    return _add_file_async(
        img,
        seg,
        dataset_settings,
        root,
        target_height_half,
        defrom,
        axis,
        deform_factor,
        defrom_count,
        mirror,
        degeneration_count,
        mapping=mapping,
        auto_crop=auto_crop,
        ignore_crop=ignore_crop,
        defrom_p=defrom_p,
    )


def finalize_ds(dataset_settings, out_base: Path):
    dataset_settings["numTraining"] = len(list((out_base / "labelsTr").iterdir()))
    with open(out_base / "dataset.json", "w") as f:
        json.dump(dataset_settings, f, indent=4)
    logger.on_ok(f"Finished dataset generation. Num Sampels is {dataset_settings['numTraining']}")


#######################################################


def _add_file_async(
    img: Path | list[Path] | str | list[str],
    seg: Path,
    dataset_settings: dict,
    root: Path,
    target_height_half=None,
    defrom=False,
    axis="S",
    deform_factor=1.0,
    defrom_count=1,
    mirror=None,
    degeneration_count=0,
    delete_brocken=True,
    mapping=None,
    auto_crop=None,
    ignore_crop=None,
    defrom_p=1.0,
):
    # check if image exists (asume on split, safe 0 split last)
    if _get_file_name(root, img, 0, defrom, defrom_count, mirror is not None, degeneration_count).exists():
        # try:
        #    if to_nii(seg, True).max() > len(dataset_settings["labels"]):
        #        seg.unlink(missing_ok=True)
        #        logger.on_debug(f"wrong label, {to_nii(seg, True).unique()}")
        #    else:
        logger.on_debug(f"Skip {seg}, exists")
        return
        # except Exception:
        #    seg.unlink(missing_ok=True)
    if delete_brocken:
        if isinstance(img, Path):
            img = [img]
        for i in img:
            try:
                to_nii(i, True).max()
            except Exception:
                [Path(i).unlink(missing_ok=True) for i in img]
                Path(seg).unlink(missing_ok=True)
                return
        try:
            to_nii(seg, True).max()
        except Exception:
            [Path(i).unlink(missing_ok=True) for i in img]
            Path(seg).unlink(missing_ok=True)
            return
    # load image
    img_nii = [to_nii(img, False)] if isinstance(img, (Path, str, BIDS_FILE)) else [to_nii(i, False) for i in img]
    seg_nii = to_nii(seg, True)

    if mapping is not None:
        seg_nii = seg_nii.map_labels(mapping)
    # spacing
    spacing = dataset_settings.get("resolution_range", (-1, -1, -1))
    spacing = tuple(float(f) for f in spacing)
    if auto_crop is not None:
        print(f"{auto_crop=}, {ignore_crop=}")
        seg_nii = seg_nii.reorient(dataset_settings["orientation"]).rescale(spacing)
        crop = seg_nii.compute_crop(0, auto_crop)
        if ignore_crop:
            crop = list(crop)
            for d in ignore_crop:
                crop[seg_nii.get_axis(d)] = slice(0, None)
            crop = tuple(crop)
        print(crop)
        seg_nii.apply_crop_(crop)
        img_nii[0] = img_nii[0].resample_from_to_(seg_nii)
    else:
        img_nii[0] = img_nii[0].reorient(dataset_settings["orientation"]).rescale(spacing)
    img_nii = [i.resample_from_to_(img_nii[0]) for i in img_nii]
    seg_nii.resample_from_to_(img_nii[0])

    for split_id, offset in _split_task(seg_nii, target_height_half, axis="S"):
        for mirror_ in [False] if mirror is None else [False, True]:
            for deg in range(degeneration_count + 1):
                c = (1 if not defrom else defrom_count + 1) if defrom_p >= 1 or defrom_p <= random.random() else 1
                for defrom_ in range(c):
                    out_name = _get_file_name(root, img, split_id, defrom_ != 0, defrom_, mirror_, deg)
                    _make_sample(
                        img_nii,
                        seg_nii,
                        out_name,
                        offset,
                        defrom=defrom_ != 0,
                        degen=deg != 0,
                        deform_factor=deform_factor,
                        target_height_half=target_height_half,
                        axis=axis,
                        mirror=mirror if mirror_ else None,
                    )


def _get_file_name(
    root,
    img: Path | list[Path] | str | list[str],
    split: int,
    defrom: bool,
    defrom_count,
    mirror,
    degeneration_count,
) -> Path:
    if not isinstance(img, (Path, str)):
        img = Path(img[0])
    addendum = _deterministic_hash(str(img))  # [:9]
    return (
        root
        / "labelsTr"
        / f"{img.name.split('.')[0]}_{addendum}_{split}{f'_d{defrom_count}' if defrom else ''}{'_m' if mirror else ''}{f'_deg{degeneration_count}' if degeneration_count != 0 else ''}.nii.gz"
    )


def _split_task(seg_nii: NII, target_height_half, axis="S"):
    if target_height_half is None:
        return [(0, 0)]
    shape = seg_nii.shape
    axis = seg_nii.get_axis(axis)  # type: ignore
    x = shape[axis]
    h = ceil(x / target_height_half)
    h = ceil(x / h)
    out = []
    for i in range(99999):
        out.append((i, i * h))
        if (i * h + target_height_half) >= x:
            break
    return reversed(out)


def _make_sample(
    img_nii: list[NII],
    seg_nii,
    outpath: Path,
    offset,
    defrom,
    degen,
    deform_factor,
    target_height_half,
    axis="S",
    mirror: list | None = None,
):
    stem = outpath.name.split(".nii.gz")[0]

    if outpath.exists() and (outpath.parent.parent / f"imagesTr/{stem}_{0:04}.nii.gz").exists():
        try:
            to_nii(outpath, True).max()
            to_nii(outpath.parent.parent / f"imagesTr/{stem}_{0:04}.nii.gz", True).max()
            logger.on_ok("Skip:", outpath.name)
            return  # noqa: TRY300
        except Exception:
            pass
    try:
        logger.on_ok(outpath.name)
        # split image
        # deform
        out_d = extract_image(
            img_nii,
            seg_nii,
            offset,
            deform=defrom,
            degen=degen,
            target_height_half=target_height_half,
            deform_factor=deform_factor,
            axis=axis,
            mirror=mirror,
        )
        img_num = 0
        for name, nii in out_d.items():
            if name == "seg":
                out = outpath
            else:
                img_num = int(name.replace("img", ""))
                out = outpath.parent.parent / f"imagesTr/{stem}_{img_num:04}.nii.gz"

            Path(out).parent.mkdir(exist_ok=True, parents=True)
            assert out_d["seg"].shape == nii.shape, (out_d, nii)
            nii.save(out)
    except Exception:
        logger.on_fail("FAILED", outpath)
        logger.print_error()
        raise


def extract_image(
    img_nii: list[NII],
    nii_seg: NII,
    offset,
    deform,
    degen,
    target_height_half=None,
    crop_top=0,
    deform_factor=1,
    axis="S",
    mirror: list | None = None,
):
    assert img_nii[0].assert_affine(nii_seg)
    axis = img_nii[0].get_axis(axis)  # type: ignore
    img_nii = [i.clone() for i in img_nii]
    nii_seg = nii_seg.clone()

    if target_height_half is None:
        offset = None
    else:
        shape = img_nii[0].shape
        offset = max(min(offset, shape[axis] - 2 * target_height_half), 0)
        max_offset = min(offset + 2 * target_height_half, shape[axis] - crop_top)
        if offset >= 0:
            crop = [slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])]
            crop[axis] = slice(offset, max_offset)
            [i.apply_crop_(tuple(crop)) for i in img_nii]
            nii_seg.apply_crop_(tuple(crop))
    if mirror is not None:
        mapping = {}
        for a, b in mirror:
            mapping[a] = b
            mapping[b] = a

        nii_seg.map_labels_(mapping)
        nii_seg = nii_seg.flip("R", keep_global_coords=False)
        img_nii = [i.flip("R", keep_global_coords=False) for i in img_nii]

    if degen:
        img_nii = [random_transform(i) for i in img_nii]
    niis: dict[str, NII] = {}
    for i, _nii in enumerate(img_nii):
        niis[f"img{i}"] = _nii
    niis["seg"] = nii_seg

    if deform:
        niis = deformed_nii(niis, deform_factor=deform_factor, normalize=True)  # type: ignore

    return niis


def _deterministic_hash(string: str) -> str:
    return hashlib.md5(string.encode()).hexdigest()


@torch.no_grad()
def random_transform(img: NII, prob=0.35):
    raise NotImplementedError("Online degeneration, has been removed")
    from training import transforms3D

    t = torch.tensor(img.get_array().astype(dtype=np.float32))
    min_v = t.min()
    max_v = t.max().item()
    tens = {"img": (t - min_v) / max_v}
    tens = transforms3D.RandomBlur(prob=prob, std=(0.5, 3), kernel_size=9)(tens)
    tens = transforms3D.RandomNoise(prob=prob, std=(0.0, 0.05))(tens)
    tens = transforms3D.RandomBiasField(coefficients=(0, 0.75), prob=prob)(tens)
    tens = transforms3D.ColorJitter3D_(prob=prob * 2)(tens)
    arr = (tens["img"] * max_v + min_v).numpy().astype(img.dtype)
    return img.set_array_(arr)
