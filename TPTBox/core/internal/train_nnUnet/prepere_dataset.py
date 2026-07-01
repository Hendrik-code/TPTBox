# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from TPTBox import Print_Logger, to_nii

logger = Print_Logger()


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """All tuneable parameters for the feet dataset build."""

    # ── Identifiers ──────────────────────────────────────────────────────────
    dataset_id: int
    files: list[tuple[Path, Path]] | list[tuple[list[Path], Path]]
    raw_label_ids: list[int | Enum] | dict[int, str | Enum] | dict[int, str] | dict[int, Enum]
    dataset_name_suffix: str = ""  # appended after Dataset<id>_ if non-empty
    nnunet_base: Path = Path()
    # ── Label IDs ─────────────────────────────────────────────────────────────

    # ── Preprocessing / spacing ───────────────────────────────────────────────
    spacing: tuple[float, float, float] = (1, 1, 1)
    is_ct: bool = True
    num_input: int = 1
    axis: str = "S"
    target_height_half: int | None = None
    auto_crop: int | None = None
    ignore_crop: Literal["R", "L", "I", "S", "A", "P"] | str | None = None  # noqa: PYI051
    # ── Augmentation ─────────────────────────────────────────────────────────
    deform_count: int = 0
    deform_factor: float = 1.0
    degeneration_count: int = 0
    mirror: list[tuple[int | Enum, int | Enum]] | None = None

    # ── Trainer ──────────────────────────────────────────────────────────────
    nn_trainer: Literal[
        "nnUNetTrainer",
        "nnUNetTrainerNoMirroring",
        "nnUNetTrainerDA5",
        "nnUNetTrainerDAExtGPU",
    ] = "nnUNetTrainer"
    auglab_params_json: str = "transform_params_gpu_default01-23.json"

    # ── Runtime ───────────────────────────────────────────────────────────────
    cpu_workers: int | None = None  # None → os.cpu_count()//2 + 3
    ignore_label: bool = False
    dry_run: bool = True  # print plan, skip actual processing


def _validate_config(cfg: DatasetConfig) -> None:
    """Raise ValueError with a clear message if the config is inconsistent."""
    errors: list[str] = []

    if cfg.mirror and "NoMirroring" not in cfg.nn_trainer:
        errors.append(
            f"use_mirror=True but nn_trainer='{cfg.nn_trainer}' does not contain "
            "'NoMirroring'. Either set use_mirror=False or use nnUNetTrainerNoMirroring."
        )
    if errors:
        logger.on_fail("Config validation failed:")
        for e in errors:
            logger.on_fail(f"  • {e}")
        raise ValueError("Invalid DatasetConfig — see errors above.")


def _build_label_mapping(
    cfg: DatasetConfig,
) -> tuple[dict[str, int], dict[int, int], dict[str, str | int], list[tuple[int, int]] | None]:
    """Returns:.
    -------
    labels_mapping:
        nnUNet label definition
        {"background": 0, "thymus": 1, ...}

    mapping_forward:
        original_label_id -> consecutive_label_id
        {17: 1, 42: 2, ...}

    labels_mapping_return:
        consecutive_label_id -> original label/name
        {"1": "thymus", "2": "femur", ...}

    mirror:
        mirror pairs remapped to consecutive ids
    """  # noqa: D205
    # ----------------------------------------------------------
    # normalize input to {original_id: name}
    # ----------------------------------------------------------
    dataset_mapping: dict[int, str]
    enums = {}

    if isinstance(cfg.raw_label_ids, dict):
        dataset_mapping = {}

        for k, v in cfg.raw_label_ids.items():
            if isinstance(v, Enum):
                dataset_mapping[int(k)] = v.name
                enums[v.name] = v.value
            else:
                dataset_mapping[int(k)] = str(v)

    else:
        dataset_mapping = {}

        for item in cfg.raw_label_ids:
            if isinstance(item, int):
                dataset_mapping[item] = str(item)
            else:
                dataset_mapping[item.value] = item.name
                enums[item.value] = item.name

    # ----------------------------------------------------------
    # create consecutive mapping
    # ----------------------------------------------------------
    labels_mapping: dict[str, int] = {"background": 0}
    mapping_forward: dict[int, int] = {}
    labels_mapping_return: dict[str, str | int] = {}

    for new_idx, (orig_idx, name) in enumerate(
        sorted(dataset_mapping.items()),
        start=1,
    ):
        labels_mapping[name] = new_idx
        mapping_forward[orig_idx] = new_idx
        labels_mapping_return[str(new_idx)] = enums.get(name, orig_idx)

    # ----------------------------------------------------------
    # remap mirror pairs
    # ----------------------------------------------------------
    mirror_out: list[tuple[int, int]] | None = None

    if cfg.mirror is not None:
        mirror_out = []

        for left, right in cfg.mirror:
            left_id = left.value if isinstance(left, Enum) else left
            right_id = right.value if isinstance(right, Enum) else right

            if left_id not in mapping_forward:
                raise ValueError(f"Mirror label {left_id} not present in raw_label_ids")

            if right_id not in mapping_forward:
                raise ValueError(f"Mirror label {right_id} not present in raw_label_ids")

            mirror_out.append((mapping_forward[left_id], mapping_forward[right_id]))

    return (labels_mapping, mapping_forward, labels_mapping_return, mirror_out)


def build_dataset(cfg: DatasetConfig) -> None:
    """Build a nnUnet dataset.

    Args:
        cfg (DatasetConfig): _description_
    """
    # ── nnUNet env MUST be set before any nnunet import ───────────────────────────
    # These are module-level so they take effect the moment this file is imported.

    f"Building Dataset {cfg.dataset_id:03}"
    _validate_config(cfg)
    os.environ["nnUNet_raw"] = str(cfg.nnunet_base / "nnUNet_raw")  # noqa: SIM112
    os.environ["nnUNet_preprocessed"] = str(cfg.nnunet_base / "nnUNet_preprocessed")  # noqa: SIM112
    os.environ["nnUNet_results"] = str(cfg.nnunet_base / "nnUNet_results")  # noqa: SIM112
    sys.path.append(str(Path(__file__).parent))
    from _prep_ds import add_file, finalize_ds, run, set_up_dataset

    labels_mapping, mapping_forward, mapping_back, mirror = _build_label_mapping(cfg)
    logger.on_text(f"Label count     : {len(mapping_forward)} classes")
    logger.on_text(f"Mirror pairs    : {len(mirror) if mirror else 0}")
    logger.on_text(f"Trainer         : {cfg.nn_trainer}")
    logger.on_text(f"Spacing         : {cfg.spacing}")
    logger.on_text(f"Deform          : (count={cfg.deform_count}, factor={cfg.deform_factor})")
    logger.on_text(f"Degeneration    : {cfg.degeneration_count}")
    logger.on_text(f"Dry run         : {cfg.dry_run}")

    if cfg.dry_run:
        logger.on_warning("Dry run — stopping before file processing.")
        logger.on_text(f"Forward mapping: {mapping_forward}")

        # Pick a random segmentation
        _, seg = random.choice(cfg.files)

        seg_nii = to_nii(seg, seg=True)

        labels_found = set(seg_nii.unique())
        labels_found.discard(0)  # ignore background

        expected_labels = set(mapping_forward.keys())

        missing_mapping = labels_found - expected_labels
        unused_mapping = expected_labels - labels_found

        logger.on_text(f"Sample segmentation: {seg}")
        logger.on_text(f"Labels found       : {sorted(labels_found)}")

        if missing_mapping:
            logger.on_fail(f"Labels present in segmentation but missing in mapping: {sorted(missing_mapping)}")

        if unused_mapping:
            logger.on_warning(f"Labels defined in mapping but not found in sample: {sorted(unused_mapping)}")

        # Test remapping
        out = seg_nii.map_labels(mapping_forward)
        remapped_labels = sorted(out.unique())

        logger.on_text(f"Remapped labels    : {remapped_labels}")

        expected_remapped = set(mapping_forward.values())
        unexpected = set(remapped_labels) - expected_remapped - {0}

        if unexpected:
            logger.on_fail(f"Unexpected labels after remapping: {sorted(unexpected)}")
        else:
            logger.on_ok("Label mapping validation successful.")

        return
    dataset_settings, out_base = set_up_dataset(
        cfg.dataset_id,
        labels_mapping,
        spacing=cfg.spacing,
        nn_trainier=cfg.nn_trainer,
        AUGLAB_PARAMS_GPU_JSON=cfg.auglab_params_json,
        ignore=cfg.ignore_label,
        num_input=cfg.num_input,
        is_ct=cfg.is_ct,
        base=cfg.nnunet_base,
    )
    dataset_settings["labels_mapping"] = mapping_back

    # ── Process files ─────────────────────────────────────────────────────────
    cpu = cfg.cpu_workers if cfg.cpu_workers is not None else (os.cpu_count() or 4) // 2 + 3
    logger.on_text(f"Worker pool     : {cpu} processes")

    results = []
    with Pool(cpu) as p:
        logger.on_log("Scheduling file processing")
        for img, seg in tqdm(cfg.files, desc="Queuing", unit="pair"):
            seg_path = Path(seg)
            if not seg_path.exists():
                logger.on_warning(f"Seg file missing, skipping: {seg_path}")
                continue

            task = add_file(
                p,
                img,
                seg_path,
                dataset_settings,
                out_base,
                target_height_half=cfg.target_height_half,
                defrom=cfg.deform_count > 0,
                axis=cfg.axis,
                deform_factor=cfg.deform_factor,
                defrom_count=cfg.deform_count,
                mirror=mirror,
                degeneration_count=cfg.degeneration_count,
                mapping=mapping_forward,
                auto_crop=cfg.auto_crop,
                ignore_crop=cfg.ignore_crop,
            )
            if task is not None:
                results.append(task)

        logger.on_text(f"Running {len(results)} async tasks …")
        p.map(run, results)

    # ── Finalise ──────────────────────────────────────────────────────────────
    finalize_ds(dataset_settings, out_base)
    logger.on_ok(f"Dataset {cfg.dataset_id:03} written to {out_base}")
    logger.on_text("Next step:")
    logger.on_text("Single Folds")
    logger.on_text(
        f"python {Path(__file__).parent}/train.py  -id {cfg.dataset_id} --gpu 0 -e 300 -el 1000 --num-folds 0 --start-fold 0 -b {cfg.nnunet_base.absolute()}"  # noqa: G004
    )  # noqa: G004

    logger.on_text("k-Folds")
    logger.on_text(
        f"python {Path(__file__).parent}/train.py  -id {cfg.dataset_id} --gpu 0 -e 300 -el 1000 --num-folds 3 --start-fold 0 -b {cfg.nnunet_base.absolute()}"  # noqa: G004
    )
    # logger.on_text(
    #    f"  conda run --live-stream --name py3.12 python "
    #    f"/DATA/NAS/ongoing_projects/robert/code/totalvibesegmentor/"
    #    f"training_nn/train_ResEnc_.py"
    # )


if __name__ == "__main__":
    from TPTBox import BIDS_FILE

    infolder = Path("/media/data/lisa/datasets/dataset-lu_dotatate_body_composition/seg_net-thymus/baseline")
    data: list[tuple[Path, Path]] = []
    for file in infolder.glob("*.nii.gz"):
        bf = BIDS_FILE(file, "/media/data/lisa/datasets/dataset-lu_dotatate_body_composition")
        sub = bf.get("sub")
        ses = bf.get("ses")
        sequ = bf.get("sequ")
        acq = bf.get("acq")
        ce = bf.get("ce")
        fn = file.name.replace("_seg-thym_msk", "_ct").replace("_seg-thym_net", "_ct")
        ct = f"/media/data/lisa/datasets/dataset-lu_dotatate_body_composition/rawdata/sub-{sub}_seg/ses-{ses}/ct/{fn}"
        ct = Path(ct)
        assert ct.exists()
        data.append((ct, file))

    build_dataset(
        DatasetConfig(
            7,
            data,
            {1: "thymus"},
            "thymus",
            nnunet_base=Path("/media/data/lisa/code/nnUnet"),
            auto_crop=150,
            ignore_crop="RA",
            is_ct=True,
            spacing=(1, 1, 1),
            dry_run=False,
        )
    )
