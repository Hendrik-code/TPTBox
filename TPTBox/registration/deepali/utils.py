import re
from collections.abc import Sequence
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar, Union

import torch
import yaml
from deepali.core import Axes, Device, Grid, PathStr
from deepali.core import functional as U
from deepali.data import FlowField, Image
from deepali.losses import RegistrationResult, new_loss
from deepali.modules import TransformImage
from deepali.spatial import (
    DisplacementFieldTransform,
    HomogeneousTransform,
    NonRigidTransform,
    QuaternionRotation,
    RigidQuaternionTransform,
    SequentialTransform,
    SpatialTransform,
    Translation,
)
from torch import Tensor, Type
from torch.nn import Module

RE_WEIGHT = re.compile(r"^((?P<mul>[0-9]+(\.[0-9]+)?)\s*[\* ])?\s*(?P<chn>[a-zA-Z0-9_-]+)\s*(\+\s*(?P<add>[0-9]+(\.[0-9]+)?))?$")
RE_TERM_VAR = re.compile(r"^[a-zA-Z0-9_-]+\((?P<var>[a-zA-Z0-9_]+)\)$")


def get_device_config(device: Union[torch.device, str, int]) -> torch.device:
    r"""Get configured PyTorch device."""
    if isinstance(device, int):
        device = f"cuda:{device}"
    elif device == "cuda":
        device = "cuda:0"
    return torch.device(device)


def get_post_transform(
    target_grid: Grid,
    source_grid: Grid,
    align=False,
) -> Optional[SpatialTransform]:
    r"""Get constant rigid transformation between image grid domains."""
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


def load_transform(path: PathStr, grid: Grid) -> SpatialTransform:
    r"""Load transformation from file.

    Args:
        path: File path from which to load spatial transformation.
        grid: Target domain grid with respect to which transformation is defined.

    Returns:
        Loaded spatial transformation.

    """
    target_grid = grid

    def convert_matrix(matrix: Tensor, grid: Optional[Grid] = None) -> Tensor:
        if grid is None:
            pre = target_grid.transform(Axes.CUBE_CORNERS, Axes.WORLD)
            post = target_grid.transform(Axes.WORLD, Axes.CUBE_CORNERS)
            matrix = U.homogeneous_matmul(post, matrix, pre)
        elif grid != target_grid:
            pre = target_grid.transform(Axes.CUBE_CORNERS, grid=grid)
            post = grid.transform(Axes.CUBE_CORNERS, grid=target_grid)
            matrix = U.homogeneous_matmul(post, matrix, pre)
        return matrix

    path = Path(path)
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            matrix = value.get("matrix")
            if matrix is None:
                raise KeyError("load_transform() .pt file dict must contain key 'matrix'")
            grid = value.get("grid")
        elif isinstance(value, Tensor):
            matrix = value
            grid = None
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


def slope_of_least_squares_fit(values: Sequence[float]) -> float:
    r"""Compute slope of least squares fit of line to last n objective function values

    See also:
    - https://www.che.udel.edu/pdf/FittingData.pdf
    - https://en.wikipedia.org/wiki/1_%2B_2_%2B_3_%2B_4_%2B_%E2%8B%AF
    - https://proofwiki.org/wiki/Sum_of_Sequence_of_Squares

    """
    n = len(values)
    if n < 2:
        return float("nan")
    if n == 2:
        return values[1] - values[0]
    # sum_x1 divided by n as a slight modified to reduce no. of operations,
    # i.e., the other terms are divided by n as well by dropping one factor n
    sum_x1 = (n + 1) / 2
    sum_x2 = n * (n + 1) * (2 * n + 1) / 6
    sum_y1 = sum(values)
    sum_xy = sum(((x + 1) * y for x, y in enumerate(values)))
    return (sum_xy - sum_x1 * sum_y1) / (sum_x2 - n * sum_x1 * sum_x1)


class OptimizerWrapper(ContextDecorator):
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __enter__(self):
        self.optimizer.zero_grad()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:  # Only step if no exception occurred
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()


def overlap_mask(source_mask: Tensor, target_mask: Tensor) -> Optional[Tensor]:
    r"""Overlap mask at which to evaluate pairwise data term."""
    mask = source_mask.type(torch.int8)
    mask &= target_mask.type(torch.int8)
    return mask


def make_foreground_mask(image: Image, foreground_lower_threshold, foreground_upper_threshold):
    data = image.tensor()
    mask = U.threshold(data, foreground_lower_threshold, foreground_upper_threshold).type(torch.int8)
    # data = torch.cat([data, mask.type(data.dtype)], dim=0)
    return Image(mask, image.grid())


def normalize_img(image: Image, normalize_strategy: Optional[Literal["auto", "CT", "MRI"]]):
    if normalize_strategy is None:
        return image
    data = image.tensor()
    if normalize_strategy == "MRI":
        max_v = torch.quantile(data[data > 0], q=0.95)
        min_v = 0
    elif normalize_strategy == "CT":
        max_v = 1024
        min_v = -1024
    elif normalize_strategy == "auto":
        max_v = image.max()
        min_v = image.min()
    else:
        raise NotImplementedError(normalize_strategy)
    scale = max_v - min_v
    data -= min_v
    data /= scale
    return Image(data, image.grid())


def clamp_mask(image: Optional[Image]):
    if image is None:
        return image
    data = image.tensor()
    data.clamp_(min=0, max=1)
    return Image(data, image.grid())


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
