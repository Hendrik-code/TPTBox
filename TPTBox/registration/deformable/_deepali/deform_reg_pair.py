from __future__ import annotations

import json
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import torch
import yaml
from deepali.core import Axes, Device, Grid, PathStr
from deepali.core import functional as deepali_functional
from deepali.data import FlowField, Image
from deepali.losses import new_loss
from deepali.spatial import (
    DisplacementFieldTransform,
    HomogeneousTransform,
    NonRigidTransform,
    QuaternionRotation,
    RigidQuaternionTransform,
    SequentialTransform,
    SpatialTransform,
    Translation,
    new_spatial_transform,
)
from torch import Tensor
from torch.nn import Module

from TPTBox.core.nii_wrapper import NII, to_nii
from TPTBox.registration.deformable._deepali.engine import RegistrationEngine
from TPTBox.registration.deformable._deepali.hooks import RegistrationEvalHook, RegistrationStepHook, normalize_grad_hook, smooth_grad_hook
from TPTBox.registration.deformable._deepali.optim import new_optimizer
from TPTBox.registration.deformable._deepali.registration_losses import PairwiseImageRegistrationLoss, RegistrationResult


def get_device_config(config: dict[str, Any], device: str | torch.device | None = None) -> torch.device:
    r"""Get configured PyTorch device."""
    if device is None:
        device = config.get("device", "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        device = f"cuda:{device}"
    elif device == "cuda":
        device = "cuda:0"
    return torch.device(device)  # type: ignore


def load_config(path) -> dict[str, Any]:
    r"""Load registration parameters from configuration file."""
    config_path = Path(path).absolute()
    config_text = config_path.read_text()
    if config_path.suffix == ".json":
        return json.loads(config_text)
    return yaml.safe_load(config_text)


def read_images(sample: PathStr | NII, device: torch.device) -> tuple[Image, dict[str, tuple[int, int]]]:
    r"""Read image data from input files."""
    if not isinstance(sample, NII):
        sample = NII.load(sample, False)
    s = sample.to_deepali(dtype=torch.float32, device=device)  # type: ignore
    channels = {"img": (0, s.shape[0])}
    return s, channels  # type: ignore


def get_clamp_config(config, _channels):
    if "threshold" in config:
        return config["threshold"]["lower"], config["threshold"]["upper"]
    else:
        return None, None


def get_normalize_config(config: dict[str, Any], image: Image, channels: dict[str, tuple[int, int]]) -> dict[str, dict[str, Tensor]]:
    r"""Calculate data normalization parameters.

    Args:
        config: Configuration.
        image: Image data.
        channels: Map of image channel slices.

    Returns:
        Dictionary of normalization parameters.

    """
    scale = {}
    shift = {}
    for channel, (start, stop) in channels.items():
        data = image.tensor().narrow(0, start, stop - start)
        lower_threshold, upper_threshold = get_clamp_config(config, channel)
        scale_factor = config.get("scale")
        if channel in ("msk", "seg"):
            if lower_threshold is None:
                lower_threshold = 0
            if upper_threshold is None:
                upper_threshold = 1
        else:
            if lower_threshold is None:
                lower_threshold = data.min()
            if upper_threshold is None:
                upper_threshold = data.max()
        if scale_factor is None:
            scale_factor = upper_threshold - lower_threshold if upper_threshold > lower_threshold else 1
        else:
            scale_factor = 1 / scale_factor
        shift[channel] = lower_threshold
        scale[channel] = scale_factor
    return {"shift": shift, "scale": scale}


def append_mask(image: Image, mask_nii: NII | None, channels: dict[str, tuple[int, int]], config: dict[str, Any]) -> Image:
    r"""Append foreground mask to data tensor."""
    data = image.tensor()
    if mask_nii is None:
        if "img" in channels:
            lower_threshold, upper_threshold = get_clamp_config(config, channels)

            mask = deepali_functional.threshold(data[slice(*channels["img"])], lower_threshold, upper_threshold)
        else:
            mask = torch.ones((1,) + data.shape[1:], dtype=data.dtype, device=data.device)

    else:
        # torch.nn.functional.one_hot
        mask = mask_nii.to_deepali().tensor().long()

        # mask = torch.nn.functional.one_hot(mask)
        # print(data.shape, mask.shape)
        # mask = mask.swapaxes(-1, 0).squeeze_(-1)
        # print(data.shape, mask.shape)

        # channels["msk"] = (data.shape[0], data.shape[0] + mask.shape[0])
        # data = torch.cat([data, mask.to(device=image.device).type(data.dtype)], dim=0)
    data = torch.cat([data, mask.type(data.dtype)], dim=0)
    channels["msk"] = (data.shape[0] - 1, data.shape[0])

    return Image(data, image.grid())


def normalize_data_(
    image: Image,
    channels: dict[str, tuple[int, int]],
    shift: dict[str, Tensor] | None = None,
    scale: dict[str, Tensor] | None = None,
) -> Image:
    r"""Normalize image data."""
    if shift is None:
        shift = {}
    if scale is None:
        scale = {}
    for channel, (start, stop) in channels.items():
        data = image.tensor().narrow(0, start, stop - start)
        offset = shift.get(channel)
        if offset is not None:
            data -= offset
        norm = scale.get(channel)
        if norm is not None:
            data /= norm
        if channel in ("msk", "seg"):
            data.clamp_(min=0, max=1)
    return image


def load_transform(path: PathStr, grid: Grid) -> SpatialTransform:
    r"""Load transformation from file.

    Args:
        path: File path from which to load spatial transformation.
        grid: Target domain grid with respect to which transformation is defined.

    Returns:
        Loaded spatial transformation.

    """
    target_grid = grid

    def convert_matrix(matrix: Tensor, grid: Grid | None = None) -> Tensor:
        if grid is None:
            pre = target_grid.transform(Axes.CUBE_CORNERS, Axes.WORLD)
            post = target_grid.transform(Axes.WORLD, Axes.CUBE_CORNERS)
            matrix = deepali_functional.homogeneous_matmul(post, matrix, pre)
        elif grid != target_grid:
            pre = target_grid.transform(Axes.CUBE_CORNERS, to_grid=grid)
            post = grid.transform(Axes.CUBE_CORNERS, to_grid=target_grid)
            matrix = deepali_functional.homogeneous_matmul(post, matrix, pre)
        return matrix

    path = Path(path)
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            matrix = value.get("matrix")
            if matrix is None:
                raise KeyError("load_transform() .pt file dict must contain key 'matrix'")
            grid = value.get("grid")  # type: ignore
        elif isinstance(value, Tensor):
            matrix = value
            grid = None  # type: ignore
        else:
            raise RuntimeError("load_transform() .pt file must contain tensor or dict")
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 4):
            raise RuntimeError("load_transform() .pt file tensor must have shape (N, 3, 4)")
        params = convert_matrix(matrix, grid)
        return HomogeneousTransform(target_grid, params=params)

    flow = FlowField.read(path, axes=Axes.WORLD)
    flow = flow.axes(Axes.from_grid(target_grid))
    flow = flow.sample(target_grid)
    return DisplacementFieldTransform(target_grid, params=flow.tensor().unsqueeze(0))


def get_post_transform(
    config: dict[str, Any],
    target_grid: Grid,
    source_grid: Grid,
) -> SpatialTransform | None:
    r"""Get constant rigid transformation between image grid domains."""
    align = config.get("align", False)
    if align is False or align is None:
        return None
    if isinstance(align, (Path, str)):
        return load_transform(align, target_grid)
    if align is True:
        align_centers = True
        align_directions = True
    elif isinstance(align, dict):
        align_centers = bool(align.get("centers", True))
        align_directions = bool(align.get("directions", True))
    else:
        raise ValueError("get_post_transform() 'config' has invalid 'align' value: {align}")
    center_offset = target_grid.world_to_cube(source_grid.center()).unsqueeze(0) if align_centers else None
    rotation_matrix = source_grid.direction() @ target_grid.direction().t().unsqueeze(0) if align_directions else None
    transform = None
    if center_offset is not None and rotation_matrix is not None:
        transform = RigidQuaternionTransform(target_grid, translation=center_offset, rotation=False)
        transform.rotation.matrix_(rotation_matrix)
    elif center_offset is not None:
        transform = Translation(target_grid, params=center_offset)
    elif rotation_matrix is not None:
        transform = QuaternionRotation(target_grid, params=False)
        transform.matrix_(rotation_matrix)
    return transform


def new_loss_terms(config: dict[str, Any]) -> dict[str, Module]:
    r"""Instantiate terms of registration loss.

    Args:
        config: Preparsed configuration of loss terms.
        target_tree: Target vessel centerline tree.

    Returns:
        Mapping from channel or loss name to loss module instance.

    """
    losses = {}
    for key, value in config.items():
        kwargs = dict(value)
        name = kwargs.pop("name", None)
        _ = kwargs.pop("weight", None)
        if name is None:
            raise ValueError(f"new_loss_terms() missing 'name' for loss '{key}'")
        if not isinstance(name, str):
            raise TypeError(f"new_loss_terms() 'name' of loss '{key}' must be str")
        loss = new_loss(name, **kwargs)
        losses[key] = loss
    return losses


def register_pairwise(  # noqa: C901
    target: NII,
    source: NII,
    target_seg: NII | None,
    source_seg: NII | None,
    config: dict[str, Any] | None = None,
    verbose: bool | int = False,
    debug: bool | int = False,
    device: str | Device | None = None,
    finest_spacing=None,
) -> SpatialTransform:
    r"""Register pair of images using ``torch.autograd`` and ``torch.optim``."""
    # Configuration
    if config is None:
        config = load_config(Path(__file__).parent / "deformable_config.yaml")
    target = to_nii(target, False)
    loss_config = config["loss"]["config"]
    loss_weights = config["loss"]["weights"]
    model_name = config["model"]["name"]
    model_args = config["model"]["args"]
    model_init = config["model"]["init"]
    optim_name = config["optim"]["name"]
    optim_args = config["optim"]["args"]
    optim_loop = config["optim"]["loop"]
    levels = config["pyramid"]["levels"]
    coarsest_level = config["pyramid"]["coarsest_level"]
    finest_level = config["pyramid"]["finest_level"]
    if finest_spacing is None:
        finest_spacing = target.spacing

    min_size = config["pyramid"]["min_size"]
    pyramid_dims = config["pyramid"]["pyramid_dims"]
    device = get_device_config(config, device)
    verbose = int(verbose)
    debug = int(debug)
    if verbose > 0:
        print()

    # Read input images
    start = timer()
    target_image, target_chns = read_images(target, device=device)
    source_image, source_chns = read_images(source, device=device)
    if verbose > 3:
        print(f"Read images from files in {timer() - start:.3f}s")
    start_reg = timer()

    # Append foreground masks
    target_image = append_mask(target_image, target_seg, target_chns, config)
    source_image = append_mask(source_image, source_seg, source_chns, config)
    # Clamp and rescale images
    norm_params = get_normalize_config(config, target_image, target_chns)
    target_image = normalize_data_(target_image, target_chns, **norm_params)
    source_image = normalize_data_(source_image, source_chns, **norm_params)

    # Create Gaussian image resolution pyramids
    start = timer()
    target_pyramid = target_image.pyramid(
        levels,
        start=finest_level,
        end=coarsest_level,
        dims=pyramid_dims,
        spacing=finest_spacing,  # type: ignore
        min_size=min_size,
    )
    source_pyramid = source_image.pyramid(
        levels,
        start=finest_level,
        end=coarsest_level,
        dims=pyramid_dims,
        spacing=finest_spacing,  # type: ignore
        min_size=min_size,
    )
    if verbose > 3:
        print(f"Constructed Gaussian resolution pyramids in {timer() - start:.3f}s\n")
    if verbose > 2:
        print("Target image pyramid:")
        print_pyramid_info(target_pyramid)
        print("Source image pyramid:")
        print_pyramid_info(source_pyramid)
    # Free no longer needed images
    del target_image
    del source_image
    # Initialize transformation
    source_grid = source_pyramid[finest_level].grid()
    finest_grid = target_pyramid[finest_level].grid()
    coarsest_grid = target_pyramid[coarsest_level].grid()
    post_transform = get_post_transform(config, finest_grid, source_grid)
    transform_downsample = model_args.pop("downsample", 0)
    transform_grid = coarsest_grid.downsample(transform_downsample)
    transform = new_spatial_transform(model_name, grid=transform_grid, groups=1, **model_args)
    if model_init:
        if verbose > 1:
            print(f"Fitting '{model_init}'...")
        disp_field = FlowField.read(model_init).to(device=device)
        assert isinstance(disp_field, FlowField)
        start = timer()
        transform = transform.to(device=device).fit(disp_field.batch())
        if verbose > 0:
            print(f"Fitted initial displacement field in {timer() - start:.3f}s")
        del disp_field
    grid_transform = SequentialTransform(transform, post_transform)
    grid_transform = grid_transform.to(device=device)
    # Perform coarse-to-fine multi-resolution registration
    for level in range(coarsest_level, finest_level - 1, -1):
        target_image = target_pyramid[level]
        source_image = source_pyramid[level]
        # Initialize transformation
        if level != coarsest_level:
            start = timer()
            transform_grid = target_image.grid().downsample(transform_downsample)
            transform.grid_(transform_grid)
            if verbose > 3:
                print(f"Subdivided control point grid in {timer() - start:.3f}s")
        grid_transform.grid_(target_image.grid())
        # Registration loss function
        loss_terms = new_loss_terms(loss_config)
        loss = PairwiseImageRegistrationLoss(
            losses=loss_terms,
            source_data=source_image.tensor().unsqueeze(0),
            target_data=target_image.tensor().unsqueeze(0),
            source_grid=source_image.grid(),
            target_grid=target_image.grid(),
            source_chns=source_chns,
            target_chns=target_chns,
            transform=grid_transform,
            weights=loss_weights,
        )
        loss = loss.to(device=device)
        # result = loss.eval()
        # Initialize optimizer
        optimizer = new_optimizer(optim_name, model=grid_transform, **optim_args)
        # Perform registration at current resolution level
        engine = RegistrationEngine(
            loss=loss,
            optimizer=optimizer,
            max_steps=optim_loop.get("max_steps", 250),
            min_delta=float(optim_loop.get("min_delta", "nan")),
            max_history=optim_loop.get("max_steps", 10),
        )
        grad_sigma = float(optim_loop.get("smooth_grad", 0))
        if isinstance(transform, NonRigidTransform) and grad_sigma > 0:
            engine.register_eval_hook(smooth_grad_hook(transform, sigma=grad_sigma))
        engine.register_eval_hook(normalize_grad_hook(transform))
        if verbose > 2:
            engine.register_eval_hook(print_eval_loss_hook(level))
        elif verbose > 1:
            engine.register_step_hook(print_step_loss_hook(level))
        engine.run()

    if verbose > 3:
        print(f"Registered images in {timer() - start_reg:.3f}s")
    if verbose > 0:
        print()
    return grid_transform


def print_eval_loss_hook(_level: int) -> RegistrationEvalHook:
    r"""Get callback function for printing loss after each evaluation."""

    def fn(_: RegistrationEngine, num_steps: int, num_eval: int, result: RegistrationResult) -> None:
        loss = float(result["loss"])
        message = f"  {num_steps:>4d}:"
        message += f" {loss:>12.05f} (loss)"
        weights: dict[str, str | float] = result.get("weights", {})
        losses: dict[str, Tensor] = result["losses"]
        for name, value in losses.items():
            value = float(value)  # noqa: PLW2901
            weight = weights.get(name, 1.0)
            if not isinstance(weight, str):
                value *= weight  # noqa: PLW2901
            elif "+" in weight:
                weight = f"({weight})"
            message += f", {value:>12.05f} [{weight} * {name}]"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        print(message, flush=True)

    return fn


def print_step_loss_hook(_level: int) -> RegistrationStepHook:
    r"""Get callback function for printing loss after each step."""

    def fn(_: RegistrationEngine, num_steps: int, num_eval: int, loss: float) -> None:
        message = f"  {num_steps:>4d}: {loss:>12.05f}"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        print(message, flush=True)

    return fn


def print_pyramid_info(pyramid: dict[int, Image]) -> None:
    r"""Print information of image resolution pyramid."""
    levels = sorted(pyramid.keys())
    for level in reversed(levels):
        grid = pyramid[level].grid()
        size = ", ".join([f"{n:>3d}" for n in grid.size()])
        origin = ", ".join([f"{n:.2f}" for n in grid.origin()])
        extent = ", ".join([f"{n:.2f}" for n in grid.extent()])
        domain = ", ".join([f"{n:.2f}" for n in grid.cube_extent()])
        print(f"- Level {level}:" + f" size=({size})" + f", origin=({origin})" + f", extent=({extent})" + f", domain=({domain})")
    print()
