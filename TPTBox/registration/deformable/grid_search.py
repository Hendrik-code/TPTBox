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


import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from data_utils import load_and_preprocess
from deepali.data import Image
from metrics import NMILOSS, calculate_jacobian_metrics, dice
from reg_utils import load_config, register_affine, register_deformable
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_config(config, args):
    config["energy"]["seg"] = [1, args.loss]
    config["energy"]["be"][0] = args.be
    config["energy"]["be"][2]["stride"] = [args.stride, args.stride, args.stride]
    config["optim"]["step_size"] = args.lr
    return config


@torch.no_grad()
def pairwise(fixed, warped, fixed_seg, warped_seg, transform):
    assert fixed.ndim == warped.ndim == 5
    folding_ratio, jacobian = calculate_jacobian_metrics(transform.numpy())
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    nmi = NMILOSS()
    # FIXME set number of labels
    dices = [dice(fixed_seg, warped_seg, label) for label in range(1, 4)]

    metrics = {
        "psnr": psnr(fixed, warped).item(),
        "ssim": ssim(fixed, warped).item(),
        "nmi": -nmi(fixed, warped).item(),
        "folding_ratio": folding_ratio,
        "jacobian": jacobian,
        "dice scores": dices,
        "avg_dice": np.mean(dices),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str)
    parser.add_argument("--be", type=float)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    # Set the seed
    set_seed(42)
    device = "cuda"

    PATH_CONFIG = Path("/vol/aimspace/users/staso/repositories/WholeBodyAtlas/cfg/deformable_config.yaml")
    PATH_DATA = Path("/vol/aimspace/projects/ukbb/data/whole_body/nifti/")
    PATH_SEG = Path("/vol/aimspace/projects/ukbb/data/whole_body/total_segmentator/")

    deformable_config = load_config(PATH_CONFIG)
    deformable_config = update_config(deformable_config, args)

    run = wandb.init(project="ukbb-reg", entity="starcksophie", config=args)

    ref_eid = "1197096"
    eid = "1028117"

    base_path = Path.cwd().joinpath("temp", eid)
    fixed_path = base_path.joinpath("ref_pp.nii.gz")
    moving_path = base_path.joinpath("A_wat.nii.gz")

    fixed_seg_path = PATH_SEG.joinpath(ref_eid, f"{ref_eid}_total_seg.nii.gz")
    moving_seg_path = base_path.joinpath("A_label.nii.gz")

    if not moving_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

        ref, sitk_arr = load_and_preprocess(
            path=PATH_DATA.joinpath(ref_eid, "wat.nii.gz"),
            save_path=fixed_path,
        )
        fixed_seg_path = PATH_SEG.joinpath(ref_eid, f"{ref_eid}_total_seg.nii.gz")

        target_grid = Image.read(fixed_path).grid()

        # affine reg
        _, transform, warped, warped_label = register_affine(
            ref,
            source_path=PATH_DATA.joinpath(str(eid), "wat.nii.gz"),
            source_seg=PATH_SEG.joinpath(str(eid), f"{eid!s}_total_seg.nii.gz"),
            target_grid=target_grid,
            iterations=200,
            save=(base_path, sitk_arr),
            device=device,
        )

    warped, warped_seg, deformable_trf = register_deformable(
        fixed_path, moving_path, deformable_config, "cuda", source_seg=moving_seg_path, target_seg=fixed_seg_path, save=None, verbose=1
    )

    ref = Image.read(fixed_path).unsqueeze(0)

    metrics = pairwise(
        fixed=ref,
        warped=warped.cpu().unsqueeze(0),
        fixed_seg=Image.read(fixed_seg_path).unsqueeze(0),
        warped_seg=warped_seg.cpu().unsqueeze(0).unsqueeze(0),
        transform=deformable_trf.tensor().cpu(),
    )

    wandb.log(metrics)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    f = np.rot90(ref[0, 0, :, 80], k=2)
    m = np.rot90(load_and_preprocess(PATH_DATA.joinpath(str(eid), "wat.nii.gz"))[0][:, 80].numpy(), k=2)
    w = np.rot90(warped[0, :, 80].detach().cpu().numpy(), k=2)

    s = ax[0].imshow(m - f, cmap="seismic")
    ax[0].set_title("diff map before reg")
    s.set_clim(-1, 1)
    s = ax[1].imshow(w - f, cmap="seismic")
    ax[1].set_title("diff map after reg")
    s.set_clim(-1, 1)

    wandb.log({"result": wandb.Image(fig)})
    wandb.finish()
