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
    orientation: tuple[str, str, str] = ("R", "A", "S")
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
    turn_on_mirroring: bool = False

    # ── Trainer ──────────────────────────────────────────────────────────────
    nn_trainer: Literal[
        "nnUNetTrainer",
        "nnUNetTrainerNoMirroring",
        "nnUNetTrainerDA5",
        "nnUNetTrainerDAExt",
        "nnUNetTrainerDAExtGPU",
        "nnUNetTrainerDAExtHybrid",
    ] = "nnUNetTrainer"
    # Either one of SmaugLab's bundled configs (resolved from smauglab.configs) or an absolute path to a custom JSON.
    smauglab_params_json: (
        Literal[
            "transform_params.json",  # noqa: PYI051
            "transform_params_gpu.json",  # noqa: PYI051
            "transform_params_hybrid.json",  # noqa: PYI051
            "transform_params_hybrid_TAGE.json",  # noqa: PYI051
            "transform_params_one-sequence-to-segment-them-all.json",  # noqa: PYI051
        ]
        | str
    ) = "transform_params_one-sequence-to-segment-them-all.json"

    # ── Runtime ───────────────────────────────────────────────────────────────
    cpu_workers: int | None = None  # None → os.cpu_count()//2 + 3
    ignore_label: bool = False
    dry_run: bool = True  # print plan, skip actual processing


# Filename that the SmaugLab params JSON is stored under inside the dataset folder.
# Kept in sync with train.py's DATASET_SMAUGLAB_PARAMS_FILENAME.
DATASET_SMAUGLAB_PARAMS_FILENAME = "smauglab_params.json"


def _should_strip_mirroring(cfg: DatasetConfig) -> bool:
    """Decide whether the SmaugLab params JSON should have mirror/flip stripped.

    Strip when either:
    - ``cfg.mirror`` is set (L/R paired labels — flipping would swap the pair), OR
    - ``cfg.turn_on_mirroring`` is False AND the trainer explicitly says NoMirroring.

    The trainer-name check alone is not enough: SmaugLab trainers
    (``nnUNetTrainerDAExt*``) do not carry ``NoMirroring`` in their name but must
    still disable mirroring when the dataset has anatomical L/R pairs.
    """
    if cfg.mirror:
        assert not cfg.turn_on_mirroring
        return True
    return bool("NoMirroring" in cfg.nn_trainer and not cfg.turn_on_mirroring)


def _resolve_source_params(cfg: DatasetConfig) -> Path:
    """Resolve ``cfg.smauglab_params_json`` to an absolute path.

    Relative filenames are looked up inside the shipped ``smauglab.configs``
    package; anything else is treated as a path on disk.
    """
    src = Path(cfg.smauglab_params_json)
    if src.is_absolute():
        return src
    try:
        import importlib.resources

        import smauglab.configs as _cfg_pkg  # type: ignore

        candidate = Path(str(importlib.resources.files(_cfg_pkg))) / src.name
        if candidate.is_file():
            return candidate
    except (ImportError, ModuleNotFoundError):
        pass
    return src.absolute()


def _write_dataset_params_json(cfg: DatasetConfig, out_base: Path) -> Path | None:
    """Copy ``cfg.smauglab_params_json`` into ``out_base`` as ``smauglab_params.json``.

    - When ``cfg.nn_trainer`` contains ``NoMirroring`` the copy is passed through
      :func:`_strip_mirroring` so that ``mirror_axes``/``FlipTransform``/``flip:true``
      are removed. SmaugLab has no NoMirroring trainer subclass, so this is how we
      disable mirroring for those trainers.
    - Does nothing if the file already exists (user asked: "wenn nicht bereits
      geschehen"). Returns the path either way, or ``None`` if the source JSON
      could not be read.
    """
    import json

    dst = Path(out_base) / DATASET_SMAUGLAB_PARAMS_FILENAME
    if dst.is_file():
        logger.on_text(f"SmaugLab params already present at {dst} — keeping existing file.")
        return dst

    src = _resolve_source_params(cfg)
    if not src.is_file():
        logger.on_warning(
            f"SmaugLab params source not found at {src}; cannot write {dst}. "
            "Set cfg.smauglab_params_json to an existing file or one of SmaugLab's bundled configs."
        )
        return None

    try:
        with src.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.on_warning(f"Failed to read SmaugLab params from {src} ({e}); dataset copy skipped.")
        return None

    if _should_strip_mirroring(cfg):
        _strip_mirroring(data)
        logger.on_text(
            "Stripped mirror_axes / FlipTransform / flip=true from SmaugLab params "
            f"(mirror pairs={bool(cfg.mirror)}, turn_on_mirroring={cfg.turn_on_mirroring}, trainer={cfg.nn_trainer})."
        )

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.on_warning(f"Failed to write {dst} ({e}); dataset copy skipped.")
        return None

    logger.on_ok(f"Wrote SmaugLab params to dataset folder: {dst}")
    return dst


def _strip_mirroring(data: object) -> None:
    """Recursively neutralize mirror/flip augmentations in a SmaugLab params tree."""
    if isinstance(data, dict):
        if "mirror_axes" in data:
            data["mirror_axes"] = []
        if "flip" in data and isinstance(data["flip"], bool):
            data["flip"] = False
        data.pop("FlipTransform", None)
        for v in data.values():
            _strip_mirroring(v)
    elif isinstance(data, list):
        for v in data:
            _strip_mirroring(v)


def _validate_config(cfg: DatasetConfig) -> None:
    """Raise ValueError with a clear message if the config is inconsistent."""
    errors: list[str] = []

    # Mirror pairs require the trainer to not mirror. SmaugLab DAExt trainers count as valid because
    # _write_dataset_params_json strips mirror/flip from their params JSON.
    _mirror_safe_trainers = {"nnUNetTrainerDAExt", "nnUNetTrainerDAExtGPU", "nnUNetTrainerDAExtHybrid"}
    if cfg.mirror and "NoMirroring" not in cfg.nn_trainer and cfg.nn_trainer not in _mirror_safe_trainers:
        errors.append(
            f"mirror pairs are set but nn_trainer='{cfg.nn_trainer}' does not disable mirroring. "
            "Use 'nnUNetTrainerNoMirroring', a SmaugLab DAExt trainer (mirror is stripped from its "
            "params JSON), or drop the mirror pairs."
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

    for new_idx, (orig_idx, name) in enumerate(sorted(dataset_mapping.items()), start=1):
        labels_mapping[name] = new_idx
        if orig_idx != new_idx:
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

            if left_id not in mapping_forward and left_id not in labels_mapping.values():
                raise ValueError(f"Mirror label {left_id} not present in raw_label_ids")

            if right_id not in mapping_forward and right_id not in labels_mapping.values():
                raise ValueError(f"Mirror label {right_id} not present in raw_label_ids")

            mirror_out.append((mapping_forward.get(left_id, left_id), mapping_forward.get(right_id, right_id)))

    return (labels_mapping, mapping_forward, labels_mapping_return, mirror_out)


def build_dataset(cfg: DatasetConfig) -> None:
    """Build a nnUNet dataset on disk from the configured file list.

    Sets the ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
    environment variables from ``cfg.nnunet_base`` before importing nnUNet
    helpers, builds the label mapping (including mirror pairs), and then
    delegates to ``set_up_dataset`` / ``add_file`` / ``finalize_ds`` from the
    ``_prep_ds`` module.

    Args:
        cfg (DatasetConfig): Fully populated dataset configuration (ID,
            trainer, spacing, augmentation counts, file pairs, output paths,
            ...). See :class:`DatasetConfig`.
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
    logger.on_text(f"Label count     : {len(labels_mapping) - 1} classes")
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

        expected_labels = set(labels_mapping.values())
        expected_labels.remove(0)
        logger.on_text(f"Sample segmentation: {seg}")
        logger.on_text(f"Labels found       : {sorted(labels_found)}")

        # Test remapping
        out = seg_nii.map_labels(mapping_forward)
        remapped_labels = sorted(out.unique())

        logger.on_text(f"Remapped labels    : {remapped_labels}")

        unexpected = set(remapped_labels) - expected_labels - {0}

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
        SMAUGLAB_PARAMS_GPU_JSON=cfg.smauglab_params_json,
        ignore=cfg.ignore_label,
        num_input=cfg.num_input,
        is_ct=cfg.is_ct,
        base=str(cfg.nnunet_base),
        orientation=cfg.orientation,
        turn_on_mirroring=cfg.turn_on_mirroring,
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
    # Copy SmaugLab params into the dataset folder (mirror-stripped for NoMirroring trainers).
    # train.py auto-picks this file up when a SmaugLab trainer is used.
    _write_dataset_params_json(cfg, out_base)
    logger.on_ok(f"Dataset {cfg.dataset_id:03} written to {out_base}")
    logger.on_text("Next step:")
    logger.on_text("1. Single Folds")
    logger.on_text(
        f"python {Path(__file__).parent}/train.py  -id {cfg.dataset_id} --gpu 0 -e 300 -el 1000 --num-folds 0 --start-fold 0 -b {cfg.nnunet_base.absolute()}"  # noqa: G004
    )  # noqa: G004

    logger.on_text("2. k-Folds")
    logger.on_text(
        f"python {Path(__file__).parent}/train.py  -id {cfg.dataset_id} --gpu 0 -e 300 -el 1000 --num-folds 3 --start-fold 0 -b {cfg.nnunet_base.absolute()}"  # noqa: G004
    )
    if cfg.nn_trainer in {"nnUNetTrainerDAExt", "nnUNetTrainerDAExtGPU", "nnUNetTrainerDAExtHybrid"}:
        logger.on_text(
            f"(SmaugLab trainer {cfg.nn_trainer!r} is recorded in dataset.json; train.py picks up "
            f"{DATASET_SMAUGLAB_PARAMS_FILENAME} from the dataset folder automatically.)"
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
