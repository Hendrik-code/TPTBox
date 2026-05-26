from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from contextlib import ContextDecorator
from pathlib import Path
from typing import Literal, Union

import torch
import torch.optim
from deepali.core import Axes, Grid, PathStr
from deepali.core import functional as U  # noqa: N812
from deepali.data import FlowField, Image
from deepali.losses import (
    BSplineLoss,
    DisplacementLoss,
    LandmarkPointDistance,
    PairwiseImageLoss,
    ParamsLoss,
    PointSetDistance,
)
from deepali.spatial import (
    DisplacementFieldTransform,
    HomogeneousTransform,
    QuaternionRotation,
    RigidQuaternionTransform,
    SpatialTransform,
    Translation,
)
from torch import Tensor
from torch.nn import Module

RE_WEIGHT = re.compile(r"^((?P<mul>[0-9]+(\.[0-9]+)?)\s*[\* ])?\s*(?P<chn>[a-zA-Z0-9_-]+)\s*(\+\s*(?P<add>[0-9]+(\.[0-9]+)?))?$")
RE_TERM_VAR = re.compile(r"^[a-zA-Z0-9_-]+\((?P<var>[a-zA-Z0-9_]+)\)$")
LOSS = Union[PairwiseImageLoss, PointSetDistance, LandmarkPointDistance, DisplacementLoss, BSplineLoss, ParamsLoss]


def get_device_config(device: Union[torch.device, str, int]) -> torch.device:
    """Return a ``torch.device`` from a device specifier.

    Args:
        device: Device specifier as a ``torch.device``, a device string such as
            ``"cpu"`` or ``"cuda:0"``, or an integer GPU index.

    Returns:
        Resolved ``torch.device`` object.
    """
    if isinstance(device, int):
        device = f"cuda:{device}"
    elif device == "cuda":
        device = "cuda:0"
    return torch.device(device)


def get_post_transform(
    target_grid: Grid,
    source_grid: Grid,
    align: bool | dict | str | Path = False,
) -> SpatialTransform | None:
    """Build a constant rigid pre-alignment transform between two image grid domains.

    Args:
        target_grid: Target (fixed) image grid.
        source_grid: Source (moving) image grid.
        align: Pre-alignment strategy.  ``False`` / ``None`` disables alignment.
            ``True`` aligns both centres and directions.  A ``dict`` may contain
            ``"centers"`` and ``"directions"`` boolean flags.  A ``str`` or
            ``Path`` is treated as a file path and loaded via :func:`load_transform`.

    Returns:
        A ``SpatialTransform`` encoding the pre-alignment, or ``None`` if
        *align* is ``False`` or ``None``.

    Raises:
        ValueError: If *align* is an unrecognised type or value.
    """
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

    def convert_matrix(matrix: Tensor, grid: Grid | None = None) -> Tensor:
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
    """Compute the slope of a least-squares line fit through the given values.

    Each element index is treated as the x-coordinate.  The formula avoids
    explicit matrix construction for efficiency.

    Args:
        values: Sequence of scalar objective-function values (at least 2).

    Returns:
        Slope of the fitted line, or ``float("nan")`` when fewer than 2 values
        are provided.

    See Also:
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
    """Context manager that zeroes gradients on entry and steps the optimiser on clean exit.

    Intended as a ``with``-statement body around a ``loss.backward()`` call.

    Args:
        optimizer: PyTorch optimiser to manage.
        scheduler: Optional learning-rate scheduler stepped after each optimiser
            update.
    """

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


def overlap_mask(source_mask: Tensor | None, target_mask: Tensor | None) -> Tensor | None:
    """Compute the element-wise AND of two binary masks.

    Returns the logical overlap region at which a pairwise loss should be
    evaluated.  If either mask is ``None``, the other is returned as-is.

    Args:
        source_mask: Binary mask for the source image, or ``None``.
        target_mask: Binary mask for the target image, or ``None``.

    Returns:
        Combined binary mask as an ``int8`` tensor, or ``None`` if both inputs
        are ``None``.
    """
    if source_mask is None:
        return target_mask
    if target_mask is None:
        return source_mask
    mask = source_mask.type(torch.int8)
    mask &= target_mask.type(torch.int8)
    return mask


def make_foreground_mask(image: Image, foreground_lower_threshold: float, foreground_upper_threshold: float) -> Image:
    """Create a binary foreground mask by thresholding image intensities.

    Args:
        image: Source deepali ``Image``.
        foreground_lower_threshold: Minimum intensity value considered foreground.
        foreground_upper_threshold: Maximum intensity value considered foreground.

    Returns:
        A deepali ``Image`` of dtype ``int8`` with 1 where the intensity is
        within [*foreground_lower_threshold*, *foreground_upper_threshold*] and
        0 elsewhere.
    """
    data = image.tensor()
    mask = U.threshold(data, foreground_lower_threshold, foreground_upper_threshold).type(torch.int8)
    # data = torch.cat([data, mask.type(data.dtype)], dim=0)
    return Image(mask, image.grid())


def normalize_img(image: Image, normalize_strategy: Literal["auto", "CT", "MRI"] | None) -> Image:
    """Normalise image intensities to the [0, 1] range.

    Args:
        image: Source deepali ``Image`` to normalise.
        normalize_strategy: Normalisation strategy to apply.

            - ``None``: Return the image unchanged.
            - ``"auto"``: Shift/scale by the observed min and max.
            - ``"CT"``: Clamp and scale within [-500, 500] HU.
            - ``"MRI"``: Scale by the 95th percentile of positive voxels.

    Returns:
        Normalised deepali ``Image`` (same grid, intensity values in [0, 1]).

    Raises:
        NotImplementedError: If *normalize_strategy* is an unrecognised string.
    """
    if normalize_strategy is None:
        return image
    data = image.tensor()
    if normalize_strategy == "MRI":
        data = data.float()
        max_v = torch.quantile(data[data > 0], q=0.95)
        min_v = 0
    elif normalize_strategy == "CT":
        max_v = 500  # we dont use -1024 to 1024 to reduce the noise we see
        min_v = -500
    elif normalize_strategy == "auto":
        max_v = image.max()
        min_v = image.min()
    else:
        raise NotImplementedError(normalize_strategy)
    scale = max_v - min_v
    if abs(scale) <= 0.00000000000001:
        from warnings import warn

        warn("Detected empty image", stacklevel=6)
        scale = 1
    data -= min_v
    data /= scale
    return Image(data, image.grid())


def clamp_mask(image: Image | None) -> Image | None:
    """Clamp mask values in-place to the [0, 1] range.

    Args:
        image: Deepali ``Image`` whose tensor is clamped, or ``None``.

    Returns:
        The same image object with values clamped to [0, 1], or ``None`` if the
        input was ``None``.
    """
    if image is None:
        return image
    data = image.tensor()
    data.clamp_(min=0, max=1)
    return Image(data, image.grid())


def print_pyramid_info(pyramid: dict[int, Image]) -> None:
    """Print size, origin, extent, and domain info for each level of an image pyramid.

    Args:
        pyramid: Mapping from pyramid level index to deepali ``Image`` objects.
            Level 0 is typically the finest resolution.
    """
    levels = sorted(pyramid.keys())
    for level in reversed(levels):
        grid = pyramid[level].grid()
        size = ", ".join([f"{n:>3d}" for n in grid.size()])
        origin = ", ".join([f"{n:.2f}" for n in grid.origin()])
        extent = ", ".join([f"{n:.2f}" for n in grid.extent()])
        domain = ", ".join([f"{n:.2f}" for n in grid.cube_extent()])
        print(f"- Level {level}:" + f" size=({size})" + f", origin=({origin})" + f", extent=({extent})" + f", domain=({domain})")
    print()


def new_loss(
    name: str,
    *args,
    _remove_weight=True,
    **kwargs,
) -> LOSS:
    r"""Initialize new loss module.

    Args:
        name: Name of loss type.
        args: Loss arguments.
        kwargs: Loss keyword arguments.

    Returns:
        New loss module.

    """
    if _remove_weight:
        _ = kwargs.pop("weight", None)
    cls = getattr(sys.modules["deepali.losses"], name, None)
    if cls is None:
        raise ValueError(f"new_loss() unknown loss {name}")
    if cls is Module or not issubclass(cls, Module):
        raise TypeError(f"new_loss() '{name}' is not a subclass of torch.nn.Module")
    return cls(*args, **kwargs)  # type: ignore


def parse_loss(loss_terms: list | dict, weights: list | dict | None) -> tuple[dict, dict]:
    """Normalise loss-term and weight specifications to a pair of dictionaries.

    Accepts flexible input formats (lists of loss instances or strings, dicts
    mapping name to loss or to ``(name, args)`` tuples) and returns a
    standardised mapping from term name to loss module and from term name to
    per-pyramid-level weight.

    Args:
        loss_terms: Loss specification as a list of loss objects / names, or a
            dict mapping term names to loss objects, name strings, or
            ``(name, args)`` / ``(name, args, kwargs)`` tuples.
        weights: Corresponding weights.  When *loss_terms* is a list this should
            also be a list.  When it is a dict this should be a matching dict.
            ``None`` defaults to equal weights of 1.

    Returns:
        A 2-tuple ``(loss_terms, weights)`` where both are plain dictionaries
        keyed by term name and the values in *weights* are scalars or
        reversed-order per-pyramid-level lists.
    """
    if isinstance(loss_terms, Sequence):
        if weights is None:
            weights = {}
        else:
            assert isinstance(weights, Sequence), "when loss_terms is a list weighting must also be a list"
            weights = {l if isinstance(l, str) else type(l).__name__: i for i, l in zip(weights, loss_terms)}
        loss_terms = {l if isinstance(l, str) else type(l).__name__: l for l in loss_terms}
    if weights is None:
        weights = {}
    assert not isinstance(weights, list), "weights and loss_terms should be the same list/dict if weights not None"
    for k, v in loss_terms.items():
        if isinstance(v, str):
            loss_terms[k] = new_loss(v)  # type: ignore
        elif isinstance(v, tuple):
            if len(v) == 2:
                name, args = v
                if isinstance(args, dict):
                    loss_terms[k] = new_loss(name, **args)  # type: ignore
                else:
                    loss_terms[k] = new_loss(name, *args)  # type: ignore
            elif len(v) == 3:
                loss_terms[k] = new_loss(v[0], *v[1], **v[2])  # type: ignore
    weights = {k: v if not isinstance(v, (list, tuple)) else v[::-1] for k, v in weights.items()}
    return loss_terms, weights
