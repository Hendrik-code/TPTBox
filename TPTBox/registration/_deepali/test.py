import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type, cast

import matplotlib.pyplot as plt
import torch
from IPython.utils import io
from matplotlib.figure import Figure
from torch import Tensor, optim
from tqdm import tqdm

from TPTBox import NII

try:
    from deepali.core.environ import cuda_visible_devices
except ImportError:
    raise ModuleNotFoundError("git clone https://github.com/BioMedIA/deepali.git && pip install ./deepali")
import deepali.spatial as spatial
from deepali.core import functional as U
from deepali.data import Image
from deepali.losses import functional as L
from torchvision.transforms import ToTensor

ImagePyramid = dict[int, Image]
LossFunction = Callable[[Tensor, Tensor, spatial.SpatialTransform], Tensor | dict[str, Tensor]]
TransformCls = str | Type[spatial.SpatialTransform]
TransformArg = TransformCls | Tuple[TransformCls, dict[str, Any]]
OptimizerCls = str | Type[optim.Optimizer]
OptimizerArg = OptimizerCls | Tuple[OptimizerCls, dict[str, Any]]
from deepali.core import Grid
from deepali.core.enum import PaddingMode, Sampling

sim_loss = L.mse_loss


def invertible_registration_figure(target: Tensor, source: Tensor, transform: spatial.SpatialTransform, seg: Tensor):
    r"""Create figure visualizing result of diffeomorphic registration.

    Args:
        target: Fixed target image.
        source: Moving source image.
        transform: Invertible spatial transform, i.e., must implement ``SpatialTransform.inverse()``.

    Returns:
        Instance of ``matplotlib.pyplot.Figure``.

    """
    device = transform.device

    highres_grid = transform.grid().resize(512)
    grid_image = U.grid_image(highres_grid, num=1, stride=8, inverted=True, device=device)

    with torch.inference_mode():
        inverse = transform.inverse()

        source_transformer = spatial.ImageTransformer(transform)
        seg_transformer = spatial.ImageTransformer(transform, sampling=Sampling.NEAREST)
        target_transformer = spatial.ImageTransformer(inverse)

        source_grid_transformer = spatial.ImageTransformer(transform, highres_grid, padding="zeros")
        target_grid_transformer = spatial.ImageTransformer(inverse, highres_grid, padding="zeros")

        warped_source: Tensor = source_transformer(source.to(device))
        print(seg.unique())
        warped_seg: Tensor = seg_transformer(seg.to(device))
        print(warped_seg.unique())
        warped_target: Tensor = target_transformer(target.to(device))

        warped_source_grid: Tensor = source_grid_transformer(grid_image)
        warped_target_grid: Tensor = target_grid_transformer(grid_image)

    imshow(target, source, nrow=2, name="target.jpg")
    imshow(warped_source, warped_target, nrow=3, name="warped_source.jpg")
    imshow(warped_source_grid, warped_target_grid, nrow=3, name="deformation.jpg")
    return warped_source, warped_target, warped_seg


@torch.no_grad()
def imshow(*imgs: Tensor, name="registration.jpg", nrow=3):
    import torchvision
    from torchvision.utils import make_grid

    if not name.endswith(".jpg"):
        name += ".jpg"

    img = [a.squeeze().unsqueeze(0) / a.max() for a in imgs]
    if len(img[0].shape) == 4:
        x = img[0].shape[-1] // 2
        img = [a[..., x] for a in img]

    # img = [a.squeeze().unsqueeze(0) / a.max() for a in imgs]
    print([a.shape for a in imgs])
    grid = make_grid(img, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(Path(Path(__file__).parent, name))


def loss_fn(
    w_curvature: float = 0,
    w_diffusion: float = 0,
    w_bending: float = 0,
) -> Callable[[Tensor, Tensor, spatial.SpatialTransform], dict[str, Tensor]]:
    r"""Construct loss function for free-form deformation (FFD) based image registration.

    Args:
        w_curvature: Weight of curvature, i.e., sum of unmixed first order derivatives.
            When the spatial transform is parameterized by velocities, the curvature of
            the velocity vector field is computed.
        w_bending: Weight of bending energy, i.e., sum of second order derivatives.

    Returns:
        Loss function which takes as input a registered image pair, and the spatial transform
        used to register the images. The loss function evaluates the alignment of the images
        based on a similarity term and optional regularization terms (transform penalties).

    """

    def loss(
        warped: Tensor,
        target: Tensor,
        transform: spatial.SpatialTransform,
    ) -> dict[str, Tensor]:
        terms: dict[str, Tensor] = {}
        # Similarity term
        sim = sim_loss(warped, target)
        terms["sim"] = sim
        loss = sim
        # Regularization terms
        # v_or_u: dense velocity or displacement vector field, respectively.
        v_or_u = getattr(transform, "v", getattr(transform, "u", None))
        assert v_or_u is not None
        if w_curvature > 0:
            curvature = L.curvature_loss(v_or_u)
            loss = curvature.mul(w_curvature).add(loss)
            terms["curv"] = curvature
        if w_diffusion > 0:
            diffusion = L.diffusion_loss(v_or_u)
            loss = diffusion.mul(w_diffusion).add(loss)
            terms["diff"] = diffusion
        if w_bending > 0:
            if isinstance(transform, spatial.BSplineTransform):
                params = transform.params
                assert isinstance(params, Tensor)
                bending = L.bspline_bending_loss(params)
            else:
                bending = L.bending_loss(v_or_u)
            loss = bending.mul(w_bending).add(loss)
            terms["be"] = bending
        return {"loss": loss, **terms}

    return loss


def image_pyramid(
    image: Tensor | Image | ImagePyramid,
    levels: int,
    grid: Optional[Grid] = None,
    device: Optional[torch.device] = None,
) -> ImagePyramid:
    r"""Consruct image pyramid from image tensor."""
    if isinstance(image, dict):
        pyramid = {}
        for level, im in image.items():
            if type(level) is not int:
                raise TypeError("Image pyramid key values must be int")
            if level >= levels:
                break
            if type(im) is Tensor:
                im = Image(im, grid)
            if not isinstance(im, Image):
                raise TypeError("Image pyramid key values must be deepali.data.Image or torch.Tensor")
            im = cast(Image, im.float().to(device))
            pyramid[level] = im
        if len(pyramid) < levels:
            raise ValueError(f"Expected image pyramid with {levels} levels, but only got {len(pyramid)} levels")
    else:
        if not isinstance(image, Image):
            image = Image(image, grid)
        image = cast(Image, image.float().to(device))
        pyramid = image.pyramid(levels)
    return pyramid


def init_transform(transform: TransformArg, grid: Grid, device: Optional[torch.device] = None) -> spatial.SpatialTransform:
    r"""Auxiliary functiont to create spatial transform."""
    if isinstance(transform, tuple):
        cls, args = transform
    else:
        cls = transform
        args = {}
    if isinstance(cls, str):
        spatial_transform = spatial.new_spatial_transform(cls, grid, **args)
    else:
        spatial_transform = cls(grid, **args)
    return spatial_transform.to(device).train()


def init_optimizer(optimizer: OptimizerArg, transform: spatial.SpatialTransform) -> optim.Optimizer:
    r"""Auxiliary function to initialize optimizer."""
    if isinstance(optimizer, tuple):
        cls, args = optimizer
    else:
        cls = optimizer
        args = {}
    if isinstance(cls, str):
        cls = getattr(optim, cls)
    if not issubclass(cls, optim.Optimizer):
        raise TypeError("'optimizer' must be a torch.optim.Optimizer")
    return cls(transform.parameters(), **args)


def multi_resolution_registration(
    target: Tensor | Image | ImagePyramid,
    source: Tensor | Image | ImagePyramid,
    loss_fn: LossFunction,
    transform: TransformArg,
    optimizer: OptimizerArg,
    iterations: int | list[int] = 100,
    levels: int = 3,
    device: Optional[str | int | torch.device] = None,
    skip_levels=[],
) -> spatial.SpatialTransform:
    r"""Multi-resolution pairwise image registration."""
    if device is None:
        if isinstance(target, dict):
            device = next(iter(target.values())).device
        else:
            device = target.device
    device = torch.device(f"cuda:{device}" if type(device) is int else device)
    target = image_pyramid(target, levels=levels, device=device)
    levels = len(target)
    source = image_pyramid(source, levels=levels, device=device)
    model = init_transform(transform, target[levels - 1].grid(), device=device)
    bar_format = "{l_bar}{bar}{rate_fmt}{postfix}"
    if isinstance(iterations, int):
        iterations = [iterations]
    iterations = list(iterations)
    iterations += [iterations[-1]] * (levels - len(iterations))
    for level, steps in zip(reversed(range(levels)), iterations):
        model.grid_(target[level].grid())
        target_batch = target[level].batch().tensor()
        source_batch = source[level].batch().tensor()
        transformer = spatial.ImageTransformer(model)
        optim = init_optimizer(optimizer, model)
        for _ in (pbar := tqdm(range(steps), bar_format=bar_format)):
            if level in skip_levels:
                break
            warped_batch: Tensor = transformer(source_batch)
            loss = loss_fn(warped_batch, target_batch, model)
            if isinstance(loss, Tensor):
                loss = dict(loss=loss)
            pbar.set_description(f"Level {level}")
            pbar.set_postfix({k: v.item() for k, v in loss.items()})
            optim.zero_grad()
            loss["loss"].backward()
            optim.step()
    return model.eval()


import numpy as np


def diffeomorphic2D(fixed, moving, seg, cuda_id=0, levels=3):
    buffer = Path(Path(__file__).parent, "StationaryVelocityFreeFormDeformation")
    if cuda_id in cuda_visible_devices():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
        # Use first device specified in CUDA_VISIBLE_DEVICES if CUDA is available
        device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() and cuda_visible_devices() else "cpu")
    else:
        print("cuda not found!", cuda_visible_devices())
        device = torch.device("cpu")
    fixed = Tensor(fixed).float()
    moving = Tensor(moving).float()
    seg = Tensor(seg.astype(float)).float()
    if fixed.shape[0] > 3:
        fixed.unsqueeze_(0)
        moving.unsqueeze_(0)
        seg.unsqueeze_(0)
    grid = Grid(shape=fixed.shape[1:])
    target_pyramid = Image(fixed, grid).pyramid(levels)
    source_pyramid = Image(moving, grid).pyramid(levels)
    print(target_pyramid.keys())
    print("size:     ", list(grid.size()))
    print("origin:   ", grid.origin().tolist())
    print("center:   ", grid.center().tolist())
    print("spacing:  ", grid.spacing().tolist())
    print("direction:", grid.direction().tolist())
    if not buffer.exists():
        a = 1000000
        transform = multi_resolution_registration(
            target=target_pyramid,
            source=source_pyramid,
            transform=(spatial.StationaryVelocityFreeFormDeformation, {"stride": 50}),
            optimizer=(optim.Adam, {"lr": 1e-2}),
            loss_fn=loss_fn(w_bending=a, w_curvature=a, w_diffusion=0),
            device=device,
            levels=levels,
            skip_levels=[0],
        )
        torch.save(transform.state_dict(), buffer)
    else:
        transform = init_transform((spatial.StationaryVelocityFreeFormDeformation, {"stride": 50}), target_pyramid[0].grid(), device=device)
        transform.load_state_dict(torch.load(buffer))
    return *invertible_registration_figure(fixed, moving, transform, seg), transform


#

if __name__ == "__main__":
    p = "/DATA/NAS/datasets_processed/MRI_spine/20230405_NeuroPoly_Cohort/rawdata/sub-m909606/ses-20161025/anat/"
    fixed1 = NII.load(Path(p, "sub-m909606_ses-20161025_acq-ax_T2w.nii.gz"), False)
    moving = NII.load(Path(p, "sub-m909606_ses-20161025_acq-sag_T2w.nii.gz"), False).resample_from_to_(fixed1)
    p2 = "/DATA/NAS/datasets_processed/MRI_spine/20230405_NeuroPoly_Cohort/translated/derivatives/sub-m909606/ses-20161025/anat/"
    seg = NII.load(Path(p2, "sub-m909606_ses-20161025_acq-iso_mod-T2w_seg-vert_msk.nii.gz"), True).resample_from_to_(fixed1)
    crop = moving.compute_crop()
    moving.apply_crop_(crop)
    fixed = fixed1.apply_crop(crop)
    seg = seg.apply_crop(crop)
    moving = moving.get_array()  # [..., 96 // 2]
    fixed_arr = fixed.get_array()  # [..., 96 // 2]
    seg = seg.get_array()
    warped_moving, warped_fixed, warped_seg, transform = diffeomorphic2D(fixed_arr, moving, seg)

    out = Path(p, "reg.nii.gz")
    fixed_out = fixed.set_array(warped_moving.squeeze().cpu().numpy())
    fixed_out.resample_from_to(fixed1).save(out)

    out = Path(p, "seg.nii.gz")
    fixed.seg = True
    fixed_out = fixed.set_array_(warped_seg.squeeze().cpu().numpy())
    fixed_out.resample_from_to(fixed1).save(out)

    # demo()
    ## fixed.resample_from_to_(moving)
    # aligned_img, T, _ = registrate_nipy(moving, fixed)
    # aligned_img.save(p + "deformed.nii.gz")
