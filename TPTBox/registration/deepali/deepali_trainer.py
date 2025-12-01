from __future__ import annotations

import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Generator, Sequence
from copy import copy
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, Union

import torch
import torch.optim
from deepali.core import PaddingMode, PathStr, Sampling
from deepali.core import functional as U  # noqa: N812
from deepali.data import FlowField, Image
from deepali.losses import (
    BSplineLoss,
    DisplacementLoss,
    LandmarkPointDistance,
    PairwiseImageLoss,
    ParamsLoss,
    PointSetDistance,
    RegistrationResult,
)
from deepali.modules import SampleImage
from deepali.spatial import (
    BSplineTransform,
    CompositeTransform,
    NonRigidTransform,
    SequentialTransform,
    SpatialTransform,
    new_spatial_transform,
)
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torch.utils.hooks import RemovableHandle

from ._hooks import normalize_grad_hook, print_eval_loss_hook_tqdm, print_step_loss_hook_tqdm, smooth_grad_hook
from ._utils import (
    LOSS,
    RE_TERM_VAR,
    OptimizerWrapper,
    get_device_config,
    get_post_transform,
    normalize_img,
    overlap_mask,
    parse_loss,
    print_pyramid_info,
    slope_of_least_squares_fit,
)


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result

    return wrapper


class DeepaliPairwiseImageTrainer:
    def __init__(
        self,
        source: Union[Image, PathStr],
        target: Union[Image, PathStr],
        source_seg: Union[Image, PathStr] | None = None,
        target_seg: Union[Image, PathStr] | None = None,
        source_pset=None,
        target_pset=None,
        source_landmarks=None,
        target_landmarks=None,
        # source_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration source
        # target_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration target
        device: Union[torch.device, str, int] = "cuda",
        # foreground_mask
        source_mask: Union[Image, PathStr] | None = None,
        target_mask: Union[Image, PathStr] | None = None,
        # normalize
        normalize_strategy: (
            Literal["auto", "CT", "MRI"] | None
        ) = "auto",  # Override on_normalize for finer normalization schema or normalize before and set to None. auto: [min,max] -> [0,1]; None: Do noting
        # Pyramid
        pyramid_levels: int | None = None,  # 1/None = no pyramid; int: number of stacks, tuple from to (0 is finest)
        finest_level: int = 0,
        coarsest_level: int | None = None,
        pyramid_finest_spacing: Sequence[int] | torch.Tensor | None = None,
        pyramid_min_size=16,
        dims=("x", "y", "z"),
        align=False,
        transform_name: str = "SVFFD",  # Names that are defined in deepali.spatial.LINEAR_TRANSFORMS and deepali.spatialNONRIGID_TRANSFORMS. Override on_make_transform for finer control
        transform_args: dict | None = None,
        transform_init: PathStr | None = None,  # reload initial flowfield from file
        optim_name="Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer control
        lr: float | Sequence[float] = 0.01,  # Learning rate
        lr_end_factor: float | None = None,  # if set, will use a LinearLR scheduler to reduce the learning rate to this factor * lr
        optim_args=None,  # args of Optimizer with out lr
        smooth_grad=0.0,
        verbose=0,
        max_steps: int | Sequence[int] = 250,  # Early stopping.  override on_converged finer control
        max_history: int | None = 100,  # Used for on_converged. look at the last n sample to compute the convergence
        min_value=0.0,  # Early stopping.  override on_converged finer control
        min_delta: float | Sequence[float] = 0.0,  # Early stopping.  override on_converged finer control
        loss_terms: list[LOSS | str] | dict[str, LOSS] | dict[str, str] | dict[str, tuple[str, dict]] | None = None,
        weights: list[float] | dict[str, float | list[float]] | dict[str, list[float]] | None = None,
    ) -> None:
        """
        Initializes the DeepaliPairwiseImageTrainer for pairwise image registration.

        Args:
            source (Union[Image, PathStr]): The source image or file path.
            target (Union[Image, PathStr]): The target image or file path.
            source_pset (optional): Source point set for point-based registration. Defaults to None.
            target_pset (optional): Target point set for point-based registration. Defaults to None.
            source_landmarks (optional): Source landmark points for registration. Defaults to None.
            target_landmarks (optional): Target landmark points for registration. Defaults to None.
            device (Union[torch.device, str, int]): Computation device ("cuda" or "cpu"). Defaults to "cuda".

            source_mask : ignore regions outside the mask
            target_mask : ignore regions outside the mask

            normalize_strategy (Optional[Literal["auto", "CT", "MRI"]]): Strategy for intensity normalization.
                - "auto": Normalizes intensities to [0,1] using min-max scaling.
                - "CT": Uses a predefined normalization strategy for CT images.
                - "MRI": Uses a predefined normalization strategy for MRI images.
                - None: No normalization is applied.
                Defaults to "auto".

            pyramid_levels (Optional[int]): Number of pyramid levels.
                - None or 1: No pyramid.
                - int: Specifies the number of levels.
                - tuple: Defines a range of levels (0 is the finest level).
                Defaults to None.
            finest_level (int): The finest level of the pyramid. Defaults to 0.
            coarsest_level (Optional[int]): The coarsest level of the pyramid. Defaults to None.
            pyramid_finest_spacing (Optional[Sequence[int] | torch.Tensor]): Spacing of the finest level. Defaults to None.
                - Uses target spacing if None
            pyramid_min_size (int): Minimum image size at the coarsest level. Defaults to 16.

            dims (tuple): Spatial dimensions of the image ("x", "y", "z"). Defaults to ("x", "y", "z").
            align (bool): Whether to align images before registration. Defaults to False.

            transform_name (str): Transformation model used for registration.
                - Must be defined in `deepali.spatial.LINEAR_TRANSFORMS` or `deepali.spatial.NONRIGID_TRANSFORMS`.
                - Override `on_make_transform` for fine control.
                Defaults to "SVFFD".
            transform_args (dict | None): Additional arguments for the transformation model. Defaults to None.
            transform_init (PathStr | None): Path to an initial transformation field. Defaults to None.

            optim_name (str): Name of the optimizer (e.g., "Adam").
                - Must be defined in `torch.optim`.
                - Override `on_optimizer` for finer control.
                Defaults to "Adam".
            lr (float): Learning rate for the optimizer. Defaults to 0.01.
            optim_args (optional): Additional optimizer arguments (excluding learning rate). Defaults to None.

            smooth_grad (float): Smoothing factor applied to gradients. Defaults to 0.0.
            verbose (int): Verbosity level (higher values provide more output). Defaults to 0.

            max_steps (int | Sequence[int]): Maximum number of optimization steps.
                - Controls early stopping.
                - Override `on_converged` for finer control.
                Defaults to 250.
            max_history (int | None): Maximum number of past loss values to store for monitoring convergence. Defaults to None.
            min_value (float): Minimum loss value for early stopping. Defaults to 0.0.
            min_delta (float): Minimum required change in loss to continue optimization. Defaults to 0.0.

            loss_terms (list[LOSS] | dict[str, LOSS] | None): List or dictionary of loss terms used for optimization. Defaults to None.
            weights (list[float] | dict[str, float] | None): Weights assigned to each loss term. Defaults to None.

        Raises:
            ValueError: If incompatible parameter combinations are provided.
        """
        if loss_terms is None:
            loss_terms = {}
        if optim_args is None:
            optim_args = {}
        if transform_args is None:
            transform_args = {}

        self._dtype = torch.float16
        self.device = get_device_config(device)
        self.normalize_strategy: Literal["auto", "CT", "MRI"] | None = normalize_strategy
        self.align = align
        self.transform_name = transform_name
        self.model_args = transform_args
        self.model_init = transform_init
        self.optim_name = optim_name
        self.lr = lr if not isinstance(lr, (Sequence)) else lr[::-1]
        self.lr_end_factor = lr_end_factor
        self.optim_args = optim_args
        self.max_steps = max_steps if not isinstance(max_steps, (Sequence)) else max_steps[::-1]
        self.max_history = max_history
        self.min_value = min_value
        self.min_delta = min_delta if not isinstance(min_delta, (Sequence)) else min_delta[::-1]
        self.verbose = verbose
        self.loss_terms, self.weights = parse_loss(loss_terms, weights)

        self.pyramid_levels = pyramid_levels
        self.finest_level = finest_level
        self.coarsest_level = coarsest_level
        self.dims = dims
        self.pyramid_finest_spacing = pyramid_finest_spacing
        self.pyramid_min_size = pyramid_min_size
        self._load_all(source, target, source_seg, target_seg, source_mask, target_mask)

        self.source_pset = source_pset
        self.target_pset = target_pset
        self.source_landmarks = source_landmarks
        self.target_landmarks = target_landmarks
        self.smooth_grad = smooth_grad
        self._eval_hooks = OrderedDict()
        self._step_hooks = OrderedDict()

    def _load_all(
        self,
        source,
        target,
        source_seg,
        target_seg,
        source_mask=None,
        target_mask=None,
    ):
        # reading images
        self.source = self._read(source)
        self.target = self._read(target)

        # generate mask

        self.source_mask = self._read(source_mask, torch.int8) if source_mask is not None else None
        self.target_mask = self._read(target_mask, torch.int8) if target_mask is not None else None
        # normalize
        self.source, self.target = self.on_normalize(self.source, self.target)
        # Pyramid
        self.source_pyramid, self.target_pyramid = self.make_pyramid(
            self.source,
            self.target,
            self.pyramid_levels,
            self.finest_level,
            self.coarsest_level,
            self.dims,
            self.pyramid_finest_spacing,
            self.pyramid_min_size,
        )
        if source_seg is not None or target_seg is not None:
            with torch.no_grad():
                self.source_seg_org = self._read(source_seg, torch.long, "cpu")
                self.target_seg_org = self._read(target_seg, torch.long, "cpu")
                # Get unique labels from both source and target
                u = torch.unique(self.target_seg_org.tensor())
                u = u.detach().cpu()
                u = [a for a in u if a != 0]
                # Build a mapping from original label -> index (starting from 1)
                mapping = {int(label.item()): idx for idx, label in enumerate(u, 1)}
                num_classes = len(mapping) + 1  # Add 1 for background or assume no 0

                u2 = torch.unique(self.source_seg_org.tensor())
                u2 = u2.detach().cpu()
                u2 = [a for a in u2 if a != 0]
                for idx in u2:
                    idx = int(idx.item())  # noqa: PLW2901
                    if idx not in mapping:
                        print("Warning no matching idx found:", idx)
                        mapping[idx] = 0
                # Remap the segmentation labels according to mapping
                source_remapped = self.source_seg_org.tensor().clone()
                target_remapped = self.target_seg_org.tensor().clone()
                for orig_label, new_label in mapping.items():
                    source_remapped[self.source_seg_org.tensor() == orig_label] = new_label
                    target_remapped[self.target_seg_org.tensor() == orig_label] = new_label

                # Convert to one-hot if needed (optional)
                print(f"Found {num_classes=}, {source_remapped.unique()}, {target_remapped.unique()}; internal mapping: {mapping}")
                one_hot_source = (
                    (torch.nn.functional.one_hot(source_remapped.long(), num_classes).to(self._dtype).to(self.device))
                    .permute(0, 4, 1, 2, 3)
                    .squeeze(0)
                )
                one_hot_target = (
                    (torch.nn.functional.one_hot(target_remapped.long(), num_classes).to(self._dtype).to(self.device))
                    .permute(0, 4, 1, 2, 3)
                    .squeeze(0)
                )
                print(f"{one_hot_target.shape=}", one_hot_target.device)

                # Wrap in Image object again
                self.source_seg = Image(
                    one_hot_source.detach(),
                    self.source_seg_org.grid(),
                    dtype=self._dtype,
                    device=self.device,
                )
                self.target_seg = Image(
                    one_hot_target.detach(),
                    self.target_seg_org.grid(),
                    dtype=self._dtype,
                    device=self.device,
                )
            print("make_pyramid seg", self.source_seg.dtype, self.source_seg.device)
            self.source_pyramid_seg, self.target_pyramid_seg = self.make_pyramid(
                self.source_seg,
                self.target_seg,
                self.pyramid_levels,
                self.finest_level,
                self.coarsest_level,
                self.dims,
                self.pyramid_finest_spacing,
                self.pyramid_min_size,
            )
            print("make_pyramid seg end", self.source_seg.dtype)
        else:
            self.source_seg = None
            self.target_seg = None
            self.source_pyramid_seg = None
            self.target_pyramid_seg = None

    def on_normalize(self, source: Image, target: Image):
        return normalize_img(source, self.normalize_strategy), normalize_img(target, self.normalize_strategy)

    # def on_normalize_seg(self, source_seg: Optional[Image], target_seg: Optional[Image]):
    #    return clamp_mask(source_seg), clamp_mask(target_seg)

    def _read(self, source, dtype=None, device=None) -> Image:
        if dtype is None:
            dtype = self._dtype
        if device is None:
            device = self.device
        if isinstance(source, (str, Path)):
            return Image.read(source, dtype=dtype, device=device)
        elif hasattr(source, "to_deepali"):
            source = source.to_deepali(dtype=dtype, device=device)
        else:
            source = source.to(dtype=dtype, device=device)
        return source

    def _pyramid(self, target_image: Image):
        return target_image.pyramid(
            self._p_levels,
            start=self._p_finest_level,
            end=self._p_coarsest_level,
            dims=self._p_pyramid_dims,
            spacing=self._p_finest_spacing,  # type: ignore
            min_size=self._p_min_size,
        )

    def make_pyramid(
        self,
        source_image: Image,
        target_image: Image,
        levels: int | None = None,
        finest_level: int = 0,
        coarsest_level: int | None = None,
        pyramid_dims=("x", "y", "z"),
        finest_spacing: Sequence[int] | torch.Tensor | None = None,
        min_size=16,
    ):
        if levels is None or levels <= 1:
            return None, None
        coarsest_level = levels - 1
        assert coarsest_level < levels, f"{coarsest_level=} is smaller {levels=}"
        assert finest_level >= 0, f"{finest_level=} is negative"
        assert coarsest_level >= 0, f"{coarsest_level=} is negative"
        self.finest_level = finest_level
        self.coarsest_level = coarsest_level
        if finest_spacing is None:
            finest_spacing = target_image.spacing()
        self._p_levels = levels
        self._p_finest_level = finest_level
        self._p_coarsest_level = coarsest_level
        self._p_pyramid_dims = pyramid_dims
        self._p_finest_spacing = finest_spacing
        self._p_min_size = min_size

        target_pyramid = self._pyramid(target_image)
        source_pyramid = self._pyramid(source_image)
        if self.verbose > 2:
            print("Target image pyramid:")
            print_pyramid_info(target_pyramid)
            print("Source image pyramid:")
            print_pyramid_info(source_pyramid)
        return source_pyramid, target_pyramid

    def on_make_transform(self, transform_name, grid, groups=1, **model_args):
        return new_spatial_transform(transform_name, grid, groups=groups, **model_args)

    def on_optimizer(
        self,
        grid_transform: SequentialTransform,
        level,
        lr_end_factor: float | None,
    ) -> tuple[Optimizer, LRScheduler | None]:
        name = self.optim_name
        cls = getattr(torch.optim, name, None)
        if cls is None:
            raise ValueError(f"Unknown optimizer: {name}")
        if not issubclass(cls, Optimizer):
            raise TypeError(f"Requested type '{name}' is not a subclass of torch.optim.Optimizer")
        kwargs = self.optim_args
        kwargs["lr"] = self.lr[level] if isinstance(self.lr, (list, tuple)) else self.lr

        optimizer = cls(grid_transform.parameters(), **kwargs)
        lr_sq = None
        if lr_end_factor is not None and lr_end_factor > 0 and lr_end_factor < 1.0:
            lr_sq = LinearLR(  # type: ignore
                optimizer,
                start_factor=1.0,
                end_factor=lr_end_factor,
                total_iters=self.max_steps[level] if isinstance(self.max_steps, (list, tuple)) else self.max_steps,
            )

        return optimizer, lr_sq

    def on_converged(self, level) -> bool:
        r"""Check convergence criteria."""
        if isinstance(self.min_delta, (float, int)):
            min_delta = self.min_delta
        elif len(self.min_delta) > level:
            min_delta = self.min_delta[level]
        else:
            min_delta = self.min_delta[-1]
        if min_delta == 0 and self.min_value == 0:
            return False
        values = self.loss_values
        if not values:
            return False
        value = values[-1]
        epsilon = abs(min_delta * value) if min_delta < 0 else min_delta
        slope = slope_of_least_squares_fit(values)
        if abs(slope) < epsilon:
            return True
        return value < self.min_value

    def _loss_terms_of_type(self, loss_type: type) -> dict[str, Module]:
        r"""Get dictionary of loss terms of a specific type."""
        return {
            name: module
            for name, module in self.loss_terms.items()
            if isinstance(module, loss_type) and not (name in ["Dice", "DCE"] or module.__class__.__name__ in ["Dice", "DCE"])
        }  # type: ignore

    def _transforms_of_type(self, transform_type: type[SpatialTransform]) -> list[SpatialTransform]:
        r"""Get list of spatial transformations of a specific type."""

        def _iter_transforms(
            transform,
        ) -> Generator[SpatialTransform, None, None]:
            if isinstance(transform, transform_type):
                yield transform
            elif isinstance(transform, CompositeTransform):
                for t in transform.transforms():
                    yield from _iter_transforms(t)

        transforms = list(_iter_transforms(self.transform))
        return transforms

    def _weighted_sum(self, losses: dict[str, Tensor], level) -> Tensor:
        r"""Compute weighted sum of loss terms."""
        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        weights = self.weights
        for name, value in losses.items():
            w = weights.get(name, 1.0)
            if isinstance(w, (list, tuple)):
                w = w[level]

            if not isinstance(w, str):
                value = w * value  # noqa: PLW2901
            loss += value.sum()
        return loss

    def on_loss(  # noqa: C901
        self,
        grid_transform: SequentialTransform,
        target: Image,
        source: Image,
        target_image_seg: Image | None,
        source_image_seg: Image | None,
        level: int,
    ):  # noqa: C901
        r"""Evaluate pairwise image registration loss."""
        target_data = target.tensor()
        result = {}
        losses = {}
        # Transform target grid points
        x = target.grid().coords(device=target_data.device).unsqueeze(0)  # grid_points
        y: Tensor = grid_transform(x)
        assert len(self.loss_terms) != 0, "No losses defined"
        ### Sum of pairwise image dissimilarity terms ###
        if self.loss_pairwise_image_terms:
            moved_data = self._sample_image(y, source.tensor())
            if self.source_mask is not None and self.target_mask is not None:
                # TODO this is from the reference implantation but is need way to much GPU...
                moved_mask = self._sample_image(y, self.source_mask)
                mask = overlap_mask(moved_mask, self.target_mask)
            else:
                mask = None
            for name, term in self.loss_pairwise_image_terms.items():
                losses[name] = term(moved_data, target_data, mask=mask)
            result["source"] = moved_data
            result["target"] = target_data
            result["mask"] = mask
        if self.loss_pairwise_image_terms2:
            assert source_image_seg is not None, "Source image segmentation is required"
            moved_data: torch.Tensor = self._sample_image(y, source_image_seg.tensor())
            target_data_seg = target_image_seg.tensor()
            if self.source_mask is not None and self.target_mask is not None:
                # TODO this is from the reference implantation but is need way to much GPU...
                moved_mask = self._sample_image(y, self.source_mask)
                mask = overlap_mask(moved_mask, self.target_mask)
            else:
                mask = None
            for name, term in self.loss_pairwise_image_terms2.items():
                losses[name] = term(moved_data.unsqueeze(0), target_data_seg.unsqueeze(0), mask=mask)  # DICE
            result["source"] = moved_data
            result["target"] = target_data
            result["mask"] = mask
        ## Sum of pairwise point set distance terms
        if self.loss_dist_terms:
            if self.source_pset is None:
                raise RuntimeError(f"{type(self).__name__}() missing source point set")
            if self.target_pset is None:
                raise RuntimeError(f"{type(self).__name__}() missing target point set")
            # Points are sampled in reverse
            t = self.transform(self.target_pset)
            for name, term in self.loss_dist_terms.items():
                losses[name] = term(t, self.source_pset)
        if self.loss_ldist_terms:
            if self.source_landmarks is None:
                raise RuntimeError(f"{type(self).__name__}() missing source landmarks")
            if self.target_landmarks is None:
                raise RuntimeError(f"{type(self).__name__}() missing target landmarks")

            # t = self.transform.points(self.target_landmarks, axes=Axes.GRID, to_axes=Axes.GRID, grid=target.grid(), to_grid=source.grid())
            t = self.transform(self.target_landmarks)
            for name, term in self.loss_ldist_terms.items():
                losses[name] = term(t, self.source_landmarks)
        ### Sum of displacement field regularization terms ###
        # Buffered vector fields
        if len(self.disp_terms) != 0:
            variables = defaultdict(list)
            for name, buf in self.transform.named_buffers():
                if buf.requires_grad:
                    var = name.rsplit(".", 1)[-1]
                    variables[var].append(buf)
            variables["w"] = [U.move_dim(y - x, -1, 1)]
            for name, term in self.disp_terms.items():
                match = RE_TERM_VAR.match(name)
                if match:
                    var = match.group("var")
                elif "v" in variables:
                    var = "v"
                elif "u" in variables:
                    var = "u"
                else:
                    var = "w"
                bufs = variables.get(var)
                if not bufs:
                    raise RuntimeError(f"Unknown variable in loss term name '{name}'")
                value = torch.tensor(0, dtype=torch.float, device=self.device)
                for buf in bufs:
                    value += term(buf)
                losses[name] = value
        ### Sum of free-form deformation loss terms ###
        for name, term in self.loss_bspline_terms.items():
            value = torch.tensor(0, dtype=torch.float, device=self.device)
            for bspline_transform in self.loss_bspline_transforms:
                value += term(bspline_transform.data())  # type: ignore
            losses[name] = value
        ### Sum of parameters loss terms ###
        for name, term in self.loss_params_terms.items():
            value = torch.tensor(0, dtype=torch.float, device=self.device)
            count = 0
            for params in self.transform.parameters():
                value += term(params)
                count += 1
            if count > 1:
                value /= count
            losses[name] = value
        # Sum of other regularization terms
        for name, term in self.loss_misc_terms.items():
            losses[name] = term()
        # Calculate total loss
        result["losses"] = losses
        result["weights"] = self.weights
        result["loss"] = self._weighted_sum(losses, level)
        return result

    def on_step(
        self,
        grid_transform: SequentialTransform,
        target_image: Image,
        source_image: Image,
        target_image_seg: Image | None,
        source_image_seg: Image | None,
        opt,
        scheduler,
        num_steps,
        level,
    ) -> torch.Tensor:
        r"""Perform one registration step.

        Returns:
            Loss value prior to taking gradient step.

        """
        with OptimizerWrapper(opt, scheduler):
            result = self.on_loss(grid_transform, target_image, source_image, target_image_seg, source_image_seg, level)
            loss: Tensor = result["loss"]
            loss.backward()
            with torch.no_grad():
                for hook in self._eval_hooks.values():
                    hook(self, num_steps, 1, result)
        with torch.no_grad():
            for hook in self._step_hooks.values():
                hook(self, num_steps, 1, loss.item())
        return loss

    def _run_level(
        self,
        grid_transform: Union[SequentialTransform, SpatialTransform, CompositeTransform],
        target_image: Image,
        source_image: Image,
        target_image_seg: Image | None,
        source_image_seg: Image | None,
        level,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
    ):
        target_image = target_image.to(self.device)  # type: ignore
        source_image = source_image.to(self.device)  # type: ignore
        if isinstance(grid_transform, SpatialTransform):
            grid_transform = SequentialTransform(grid_transform)
        elif not isinstance(grid_transform, CompositeTransform):
            raise TypeError("PairwiseImageRegistrationLoss() 'transform' must be of type CompositeTransform")
        grid_transform = grid_transform.to(self.device)
        opt, lr_sq = self.on_optimizer(grid_transform, level, self.lr_end_factor)
        self.optimizer = opt
        if isinstance(self.max_steps, int):
            max_steps = self.max_steps
        elif len(self.max_steps) > level:
            max_steps = self.max_steps[level]
        else:
            max_steps = self.max_steps[-1]
        return self.run_level(
            grid_transform,
            target_image,
            source_image,
            target_image_seg,
            source_image_seg,
            opt,
            lr_sq,
            level,
            max_steps,
            sampling,
        )

    def on_split_losses(self):
        misc_excl = set()
        # self.loss_terms = {a: l.to(self.device) for a, l in self.loss_terms.items()}
        from TPTBox.registration.ridged_intensity.affine_deepali import (  # noqa: PLC0415
            PairwiseSegImageLoss,
        )

        self.loss_pairwise_image_terms2 = self._loss_terms_of_type(PairwiseSegImageLoss)
        for name, module in self.loss_terms.items():
            if name in ["Dice", "DCE"] or module.__class__.__name__ in ["Dice", "DCE"]:
                self.loss_pairwise_image_terms2[name] = module
        misc_excl |= set(self.loss_pairwise_image_terms2.keys())
        self.loss_pairwise_image_terms = self._loss_terms_of_type(PairwiseImageLoss)

        misc_excl |= set(self.loss_pairwise_image_terms.keys())
        dist_terms = self._loss_terms_of_type(PointSetDistance)
        misc_excl |= set(dist_terms.keys())
        self.loss_ldist_terms = {k: v for k, v in dist_terms.items() if isinstance(v, LandmarkPointDistance)}
        self.loss_dist_terms = {k: v for k, v in dist_terms.items() if k not in self.loss_ldist_terms}
        self.disp_terms = self._loss_terms_of_type(DisplacementLoss)
        misc_excl |= set(self.disp_terms.keys())
        self.loss_bspline_transforms = self._transforms_of_type(BSplineTransform)
        self.loss_bspline_terms = self._loss_terms_of_type(BSplineLoss)
        misc_excl |= set(self.loss_bspline_terms.keys())
        self.loss_params_terms = self._loss_terms_of_type(ParamsLoss)
        misc_excl |= set(self.loss_params_terms.keys())
        self.loss_misc_terms = {k: v for k, v in self.loss_terms.items() if k not in misc_excl}

    def on_run_start(
        self,
        grid_transform,
        target_image,
        source_image,
        target_image_seg,
        source_image_seg,
        opt,
        lr_sq,
        num_steps,
        level,
    ):
        pass

    def on_run_end(
        self,
        grid_transform,
        target_image,
        source_image,
        target_image_seg,
        source_image_seg,
        opt,
        lr_sq,
        num_steps,
        level,
    ):
        pass

    def run_level(
        self,
        grid_transform: SequentialTransform,
        target_image: Image,
        source_image: Image,
        target_image_seg: Image | None,
        source_image_seg: Image | None,
        opt: Optimizer,
        lr_sq: LRScheduler | None,
        level,
        max_steps,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
    ):
        ## Perform registration at current resolution level
        ## Initialize optimizer
        target_grid = target_image.grid()
        source_grid = source_image.grid()
        self.transform = grid_transform
        self._sample_image = SampleImage(
            target=target_grid,
            source=source_grid,
            sampling=sampling,
            padding=PaddingMode.ZEROS,
            align_centers=False,
        ).to(self.device)
        grad_sigma = self.smooth_grad
        self.loss_values = []
        self.on_split_losses()
        num_steps = 0

        _eval_hooks = copy(self._eval_hooks)
        _step_hooks = copy(self._step_hooks)
        if isinstance(self.transform, NonRigidTransform) and grad_sigma > 0:
            self.register_eval_hook(smooth_grad_hook(self.transform, sigma=grad_sigma))
        self.register_eval_hook(normalize_grad_hook(self.transform))
        if self.verbose > 2:
            self.register_eval_hook(print_eval_loss_hook_tqdm(level, max_steps))
        elif self.verbose > 1:
            self.register_step_hook(print_step_loss_hook_tqdm(level, max_steps))
        self.on_run_start(
            grid_transform,
            target_image,
            source_image,
            target_image_seg,
            source_image_seg,
            opt,
            lr_sq,
            num_steps,
            level,
        )
        while num_steps < max_steps and not self.on_converged(level):
            value = self.on_step(
                grid_transform,
                target_image,
                source_image,
                target_image_seg,
                source_image_seg,
                opt,
                lr_sq,
                num_steps,
                level,
            )
            num_steps += 1
            with torch.no_grad():
                if math.isnan(value):
                    raise RuntimeError(f"NaN value in registration loss at gradient step {num_steps}")
                if math.isinf(value):
                    raise RuntimeError(f"Inf value in registration loss at gradient step {num_steps}")
                self.loss_values.append(value)
                if self.max_history is not None and len(self.loss_values) > self.max_history:
                    self.loss_values.pop(0)

        self._eval_hooks = _eval_hooks
        self._step_hooks = _step_hooks
        self.on_run_end(
            grid_transform,
            target_image,
            source_image,
            target_image_seg,
            source_image_seg,
            opt,
            lr_sq,
            num_steps,
            level,
        )

    def _load_initial_transform(self, transform: SpatialTransform):
        if self.model_init:
            if self.verbose > 1:
                print(f"Fitting '{self.model_init}'...")
            disp_field = FlowField.read(self.model_init).to(device=self.device)
            assert isinstance(disp_field, FlowField)
            transform = transform.to(device=self.device).fit(disp_field.batch())
            del disp_field
        return transform

    def register_eval_hook(
        self,
        hook: Callable[[DeepaliPairwiseImageTrainer, int, int, RegistrationResult], None],
    ) -> RemovableHandle:
        r"""Registers a evaluation hook.

        The hook will be called every time after the registration loss has been evaluated
        during a single step of the optimizer, and the backward pass was performed, but
        before the model parameters are updated by taking a gradient step.

            hook(self, num_steps: int, num_evals: int, result: RegistrationResult) -> None

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``

        """
        handle = RemovableHandle(self._eval_hooks)
        self._eval_hooks[handle.id] = hook
        return handle

    def register_step_hook(self, hook: Callable[[DeepaliPairwiseImageTrainer, int, int, float], None]) -> RemovableHandle:
        r"""Registers a gradient step hook.

        The hook will be called every time after a gradient step of the optimizer.

            hook(self, num_steps: int, num_evals: int, loss: float) -> None

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``

        """
        handle = RemovableHandle(self._step_hooks)
        self._step_hooks[handle.id] = hook
        return handle

    def on_transform_update(self, transform: SpatialTransform):
        pass

    @time_it
    def run(self):
        start_reg = timer()
        if self.source_pyramid is None:
            target_image = self.target
            source_image = self.source
            grid = target_image.grid()
            transform = self.on_make_transform(self.transform_name, grid=grid, groups=1, **self.model_args)
            transform = self._load_initial_transform(transform)
            transform = transform.to(device=self.device)
            grid_transform = SequentialTransform(transform)
            self.on_transform_update(grid_transform)
            grid_transform = self._run_level(
                grid_transform,
                target_image,
                source_image,
                self.target_seg,
                self.source_seg,
                0,
            )
        else:
            with torch.no_grad():
                ## loop pyramid
                source_pyramid = self.source_pyramid
                target_pyramid = self.target_pyramid
                target_mask = self.target_mask
                if self.target_mask is not None:
                    target_mask_pyramid = self._pyramid(self.target_mask.to(torch.float32))  # type: ignore
                assert target_pyramid is not None
                finest_level = self.finest_level
                coarsest_level = self.coarsest_level
                source_grid = source_pyramid[finest_level].grid()
                finest_grid = target_pyramid[finest_level].grid()
                coarsest_grid = target_pyramid[coarsest_level].grid()
                post_transform = get_post_transform(finest_grid, source_grid, self.align)

                transform_downsample = self.model_args.pop("downsample", 0)
                transform_grid = coarsest_grid.downsample(transform_downsample)
                transform = self.on_make_transform(
                    self.transform_name,
                    grid=transform_grid,
                    groups=1,
                    **self.model_args,
                )
                transform = self._load_initial_transform(transform)
                grid_transform = SequentialTransform(transform, post_transform)
                grid_transform = grid_transform.to(device=self.device)
                self.on_transform_update(grid_transform)

                ## Perform coarse-to-fine multi-resolution registration

            for level in range(coarsest_level, finest_level - 1, -1):
                with torch.no_grad():
                    target_image = target_pyramid[level]
                    source_image = source_pyramid[level]
                    target_image_seg = self.target_pyramid_seg[level] if self.target_pyramid_seg is not None else None
                    source_image_seg = self.source_pyramid_seg[level] if self.source_pyramid_seg is not None else None
                    if self.target_mask is not None:
                        self.target_mask = torch.ceil(target_mask_pyramid[level]).to(dtype=torch.int8)
                    ## Initialize transformation
                    if level != coarsest_level:
                        start = timer()
                        transform_grid = target_image.grid().downsample(transform_downsample)
                        transform.grid_(transform_grid)
                        self.on_transform_update(grid_transform)
                        if self.verbose > 3:
                            print(f"Subdivided control point grid in {timer() - start:.3f}s")
                    grid_transform.grid_(target_image.grid())
                self._run_level(
                    grid_transform,
                    target_image,
                    source_image,
                    target_image_seg,
                    source_image_seg,
                    level,
                )
            if self.verbose > 3:
                print(f"Registered images in {timer() - start_reg:.3f}s")
            if self.verbose > 0:
                print()
            self.target_mask = target_mask
            return grid_transform
