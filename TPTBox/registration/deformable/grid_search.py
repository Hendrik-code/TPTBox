# Original Source:
# @article{starck2024using,
#  title={Using UK Biobank data to establish population-specific atlases from whole body MRI},
#  author={Starck, Sophie and Sideri-Lampretsa, Vasiliki and Ritter, Jessica JM and Zimmer, Veronika A and Braren, Rickmer and Mueller, Tamara T and Rueckert, Daniel},
#  journal={Communications Medicine},
#  volume={4},
#  number={1},
#  pages={237},
#  year={2024},
#  publisher={Nature Publishing Group UK London}
# }
from __future__ import annotations

import argparse
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deepali.data import Image

# import wandb
from deepali.losses import HuberImageLoss
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from TPTBox.core.nii_wrapper import to_nii
from TPTBox.registration.deformable._deepali.metrics import NMILOSS, calculate_jacobian_metrics, dice
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration, Image_Reference, _load_config


@dataclass
class Example:
    name: str
    fixed_path: Image_Reference
    moving_path: Image_Reference
    fixed_seg_path: Image_Reference | None
    moving_seg_path: Image_Reference | None
    outfolder: Path


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_config(config, loss, be, stride, lr):
    config["loss"]["config"]["seg"]["name"] = loss
    config["loss"]["weights"]["be"] = be
    config["model"]["args"]["stride"] = stride
    config["optim"]["args"]["lr"] = lr
    return config


@torch.no_grad()
def pairwise(fixed, warped, fixed_seg, warped_seg, transform):
    # print(fixed.shape, warped.shape, fixed_seg.shape, warped_seg.shape)
    # print(type(fixed), type(warped), type(fixed_seg), type(warped_seg))
    fixed = np.expand_dims(fixed.numpy(), 0)
    warped = np.expand_dims(warped.numpy(), 0)

    # assert fixed.ndim == warped.ndim == 5
    folding_ratio, jacobian = calculate_jacobian_metrics(transform.numpy())
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    nmi = NMILOSS()
    # FIXME set number of labels
    if fixed_seg is not None:
        fixed_seg = np.expand_dims(fixed_seg.numpy(), 0)
        warped_seg = np.expand_dims(warped_seg.numpy(), 0)
        fixed_seg[fixed < 10] = 0
        warped_seg[fixed < 10] = 0
        c = np.unique(fixed_seg)
        dices = {label: dice(fixed_seg, warped_seg, label) for label in c if int(label) != 0}
    else:
        dices = {-1: -1}
    fixed_t = Tensor(fixed.astype(np.float32))
    warped_t = Tensor(warped.astype(np.float32))
    metrics = {
        "psnr": psnr(fixed_t, warped_t).item(),
        "ssim": ssim(fixed_t, warped_t).item(),
        "nmi": -nmi(fixed_t, warped_t).item(),
        "folding_ratio": folding_ratio,
        "jacobian": jacobian,
        "dice scores": dices,
        "avg_dice": np.mean(list(dices.values())),
    }
    return metrics


def run_all(
    e: Example,
    losses=(
        # "SSD",
        # "NMI",
        # "HuberImageLoss",
        "LNCC",
        # "MSE",
    ),
    bes=(
        1e-2,
        # 1e-3,
    ),
    strides=(4, 8, 16),
    lrs=(
        # 1e-2,
        1e-3,
        1e-4,
    ),
    spacings=([], [2], [3], [4]),
):
    tests = []
    for loss in losses:
        for be in bes:
            for sx in strides:
                for sy in strides:
                    for sz in strides:
                        for lr in lrs:
                            for spacing in spacings:
                                tests.append((loss, be, (sx, sy, sz), lr, *spacing))  # noqa: PERF401

    # Buffer laden, falls vorhanden
    try:
        with open(e.outfolder / f"buffer_{e.name}.pkl", "rb") as w:
            buffer = pickle.load(w)
    except (FileNotFoundError, EOFError):
        buffer = {}
    # print([f["avg_dice"] for f in buffer.values()])
    # print(max([f["avg_dice"] for f in buffer.values()]))
    # exit()
    random.shuffle(tests)

    for test in tqdm(tests):
        if test in buffer:
            continue
        # break
        buffer[test] = run(e, *test)

        # Buffer speichern nach jedem Testlauf
        with open(e.outfolder / f"buffer_{e.name}.pkl", "wb") as f:
            pickle.dump(buffer, f)

        # Ergebnisse als Excel speichern
        data = []
        for a, metrics in buffer.items():
            if len(a) == 4:
                (loss, be, stride, lr) = a
                spacing_type = 1
            else:
                (loss, be, stride, lr, spacing_type) = a
            row = {
                "Loss": loss,
                "BE": be,
                "Strides_X": stride[0],
                "Strides_Y": stride[1],
                "Strides_Z": stride[2],
                "LR": lr,
                "spacing_type": spacing_type,
                **metrics,
            }
            row.pop("dice scores", None)
            data.append(row)

        df_ = pd.DataFrame(data)
        df_.to_excel(e.outfolder / f"results_{e.name}.xlsx", index=False)


def run(e: Example, loss, be, stride, lr, spacing=1, save=False):
    PATH_CONFIG = Path(__file__).parent / "settings.json"  # noqa: N806
    deformable_config = _load_config(PATH_CONFIG)
    deformable_config = update_config(deformable_config, loss, be, stride, lr)

    # base_path = Path.cwd().joinpath("temp", eid)
    # TODO POINT Preregistaion
    # if not moving_path.exists():
    #    raise NotImplementedError()
    deform = Deformable_Registration(e.fixed_path, e.moving_path, device="cuda", config=deformable_config, spacing_type=spacing)
    warped = deform.transform_nii(to_nii(e.moving_path, False))
    warped_seg = deform.transform_nii(to_nii(e.moving_seg_path, True)) if e.moving_seg_path is not None else None
    deformable_trf = deform.transform
    ref = to_nii(e.fixed_path, False)  # .unsqueeze(0)

    metrics = pairwise(
        fixed=ref,
        warped=warped,
        fixed_seg=to_nii(e.fixed_seg_path, True) if e.fixed_seg_path is not None else None,
        warped_seg=warped_seg,
        transform=deformable_trf.tensor().cpu(),
    )
    if save:
        warped.save(e.outfolder / f"{e.name}_moved.nii.gz")
        ref.save(e.outfolder / f"{e.name}_fixed.nii.gz")
        if warped_seg is not None:
            warped_seg.save(e.outfolder / f"{e.name}_moved_seg.nii.gz")
            to_nii(e.fixed_seg_path, True).save(e.outfolder / f"{e.name}_fixed_seg.nii.gz")
    return metrics


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--loss", type=str, default="LNCC")
    # parser.add_argument("--be", type=float, default=0.001)
    # parser.add_argument("--stride", type=int, nargs=3, default=[8, 8, 16])
    # parser.add_argument("--lr", type=float, default=0.001)
    # args = parser.parse_args()
    PATH_DATA = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/100/100000/")

    # Set the seed
    set_seed(42)
    device = "cuda"
    e = Example(
        "vibe-mevibe-100000",
        PATH_DATA / "mevibe/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz",
        Path(
            "/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata_stitched/100/100000/vibe/sub-100000_sequ-stitched_acq-ax_part-water_vibe.nii.gz",
        ),
        fixed_seg_path=Path(
            "/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_Abdominal-Segmentation/100/100000/mevibe/sub-100000_sequ-me1_acq-ax_mod-mevibe_seg-TotalVibeSegmentator80_msk.nii.gz"
        ),
        moving_seg_path=Path(
            "/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_Abdominal-Segmentation/100/100000/vibe/sub-100000_sequ-stitched_acq-ax_mod-vibe_seg-TotalVibeSegmentator_msk.nii.gz"
        ),
        outfolder=Path("/DATA/NAS/ongoing_projects/robert/test"),
    )
    p = Path("/DATA/NAS/datasets_processed/SHIP/rawdata/10108/sub-10108001/")
    e = Example(
        "ship-vibe-T2w-10108001",
        p / "T2w/sub-10108001_sequ-T2w-2_T2w.nii.gz",
        p / "vibe/sub-10108001_sequ-stitched_part-water_vibe.nii.gz",
        None,
        None,
        outfolder=Path("/DATA/NAS/ongoing_projects/robert/test"),
    )

    # run_all(e)
    p = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata_stitched/102/102017/")
    e = Example(
        "nako-vibe-T2w-102017",
        p / "T2w/sub-102017_sequ-stitched_acq-sag_T2w.nii.gz",
        p / "vibe/sub-102017_sequ-stitched_acq-ax_part-water_vibe.nii.gz",
        Path(
            "/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_totalVibe/102/102017/T2w/sub-102017_sequ-stitched_acq-sag_seg-totalVibeSegmentor98_msk.nii.gz"
        ),
        Path(
            "/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_Abdominal-Segmentation/102/102017/vibe/sub-102017_sequ-stitched_acq-ax_mod-vibe_seg-TotalVibeSegmentator_msk.nii.gz"
        ),
        outfolder=Path("/DATA/NAS/ongoing_projects/robert/test"),
    )
    # m = run(e, "LNCC", 0.01, (4, 4, 8), 0.001, 3, True)
    # run_all(e)
    # print(m)
    # exit()
    p = Path("/DATA/NAS/datasets_processed/SHIP/rawdata/10108/sub-10108001/")
    e = Example(
        "ship-T1w-T2w-10108001",
        p / "T2w/sub-10108001_sequ-T2w-2_T2w.nii.gz",
        p / "T1w/sub-10108001_sequ-T1w-2_T1w.nii.gz",
        None,
        None,
        outfolder=Path("/DATA/NAS/ongoing_projects/robert/test"),
    )

    run(e, "LNCC", 0.001, (4, 4, 16), 0.001, True)
    # metrics = run_all(e)
    # print(metrics)
