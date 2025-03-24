import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Generator, Sequence
from copy import copy
from timeit import default_timer as timer
from typing import Literal, Optional, Union

import torch
import torch.optim
from deepali.core import PaddingMode, PathStr, Sampling
from deepali.core import functional as U
from deepali.data import FlowField, Image
from deepali.losses import (
    BSplineLoss,
    DisplacementLoss,
    LandmarkPointDistance,
    PairwiseImageLoss,
    ParamsLoss,
    PointSetDistance,
    RegistrationLoss,
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
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.hooks import RemovableHandle

from .hooks import normalize_grad_hook, print_eval_loss_hook_tqdm, print_step_loss_hook_tqdm, smooth_grad_hook
from .utils import (
    RE_TERM_VAR,
    OptimizerWrapper,
    get_device_config,
    get_post_transform,
    make_foreground_mask,
    normalize_img,
    overlap_mask,
    print_pyramid_info,
    slope_of_least_squares_fit,
)

LOSS = Union[PairwiseImageLoss, PointSetDistance, LandmarkPointDistance, DisplacementLoss, BSplineLoss, ParamsLoss]


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
        source_pset=None,
        target_pset=None,
        source_landmarks=None,
        target_landmarks=None,
        # source_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration source
        # target_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration target
        device: Union[torch.device, str, int] = "cuda",
        # foreground_mask
        mask_foreground=False,
        foreground_lower_threshold: Optional[int] = None,  # None means min
        foreground_upper_threshold: Optional[int] = None,  # None means max
        # normalize
        normalize_strategy: Optional[
            Literal["auto", "CT", "MRI"]
        ] = "auto",  # Override on_normalize for finer normalization schema or normalize before and set to None. auto: [min,max] -> [0,1]; None: Do noting
        # Pyramid
        pyramid_levels: Optional[int] = None,  # 1/None = no pyramid; int: number of stacks, tuple from to (0 is finest)
        finest_level: int = 0,
        coarsest_level: Optional[int] = None,
        pyramid_finest_spacing: Optional[Sequence[int] | torch.Tensor] = None,
        pyramid_min_size=16,
        dims=("x", "y", "z"),
        align=False,
        transform_name: str = "SVFFD",  # Names that are defined in deepali.spatial.LINEAR_TRANSFORMS and deepali.spatialNONRIGID_TRANSFORMS. Override on_make_transform for finer controle
        transform_args: dict | None = None,
        transform_init: PathStr | None = None,  # reload initial flowfield from file
        optim_name="Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer controle
        lr=0.01,  # Learning rate
        optim_args=None,  # args of Optimizer with out lr
        smooth_grad=0.0,
        verbose=0,
        max_steps: int | Sequence[int] = 250,  # Early stopping.  override on_converged finer controle
        max_history: int | None = None,
        min_value=0.0,  # Early stopping.  override on_converged finer controle
        min_delta=0.0,  # Early stopping.  override on_converged finer controle
        loss_terms: list[LOSS] | dict[str, LOSS] | None = None,
        weights: list[float] | dict[str, float] | None = None,
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

            mask_foreground (bool): Whether to mask the foreground in images. Defaults to True.
            foreground_lower_threshold (Optional[int]): Lower intensity threshold for foreground masking.
                If None, the minimum value is used. Defaults to None.
            foreground_upper_threshold (Optional[int]): Upper intensity threshold for foreground masking.
                If None, the maximum value is used. Defaults to None.

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

        self._dtype = torch.float32
        self.device = get_device_config(device)
        self.foreground_lower_threshold = foreground_lower_threshold
        self.foreground_upper_threshold = foreground_upper_threshold
        self.normalize_strategy: Optional[Literal["auto", "CT", "MRI"]] = normalize_strategy
        self.align = align
        self.transform_name = transform_name
        self.model_args = transform_args
        self.model_init = transform_init
        self.optim_name = optim_name
        self.lr = lr
        self.optim_args = optim_args
        self.max_steps = max_steps
        self.max_history = max_history
        self.min_value = min_value
        self.min_delta = min_delta
        self.verbose = verbose
        if isinstance(loss_terms, Sequence):
            if weights is None:
                weights = {}
            else:
                assert isinstance(weights, Sequence), "when loss_terms is a list weighting must also be a list"
                weights = {type(l).__name__: i for i, l in zip(weights, loss_terms)}
            loss_terms = {type(l).__name__: l for l in loss_terms}

        if weights is None:
            weights = {}
        assert not isinstance(weights, list), "weights and loss_terms should be the same list/dict if weights not None"
        self.loss_terms = loss_terms
        self.weights = weights
        # reading images
        self.source = self._read(source)
        self.target = self._read(target)
        # self.source_seg = self._read(source_seg)
        # self.target_seg = self._read(target_seg)
        # generate mask

        if mask_foreground:
            self.source_mask, self.target_mask = self.on_generate_masking(self.source, self.target)
            self.source_mask = self._read(self.source_mask)
            self.target_mask = self._read(self.target_mask)
        else:
            self.source_mask, self.target_mask = None, None
        # normalize
        self.source, self.target = self.on_normalize(self.source, self.target)
        # self.source_seg, self.target_seg = self.on_normalize_seg(self.source_seg, self.target_seg)
        # Pyramid

        self.source_pyramid, self.target_pyramid = self.make_pyramid(
            self.source, self.target, pyramid_levels, finest_level, coarsest_level, dims, pyramid_finest_spacing, pyramid_min_size
        )
        self.source_pset = source_pset
        self.target_pset = target_pset
        self.source_landmarks = source_landmarks
        self.target_landmarks = target_landmarks
        self.smooth_grad = smooth_grad
        self._eval_hooks = OrderedDict()
        self._step_hooks = OrderedDict()

    def on_generate_masking(self, source: Image, target: Image):
        """Append foreground mask to data tensor."""
        source_mask = make_foreground_mask(source, self.foreground_lower_threshold, self.foreground_upper_threshold)
        target_mask = make_foreground_mask(target, self.foreground_lower_threshold, self.foreground_upper_threshold)
        return source_mask, target_mask

    def on_normalize(self, source: Image, target: Image):
        return normalize_img(source, self.normalize_strategy), normalize_img(target, self.normalize_strategy)

    # def on_normalize_seg(self, source_seg: Optional[Image], target_seg: Optional[Image]):
    #    return clamp_mask(source_seg), clamp_mask(target_seg)

    def _read(self, source) -> Image:
        if isinstance(source, PathStr):
            return Image.read(source, dtype=self._dtype, device=self.device)
        elif hasattr(source, "to_deepali"):
            source = source.to_deepali(dtype=self._dtype, device=self.device)
        else:
            source = source.to(dtype=self._dtype, device=self.device)
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
        levels: Optional[int] = None,
        finest_level: int = 0,
        coarsest_level: Optional[int] = None,
        pyramid_dims=("x", "y", "z"),
        finest_spacing: Optional[Sequence[int] | torch.Tensor] = None,
        min_size=16,
    ):
        if levels is None or levels == 1:
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

    def on_optimizer(self, grid_transform: SequentialTransform) -> tuple[Optimizer, Optional[LRScheduler]]:
        name = self.optim_name
        cls = getattr(torch.optim, name, None)
        if cls is None:
            raise ValueError(f"Unknown optimizer: {name}")
        if not issubclass(cls, Optimizer):
            raise TypeError(f"Requested type '{name}' is not a subclass of torch.optim.Optimizer")
        kwargs = self.optim_args
        kwargs["lr"] = self.lr
        return cls(grid_transform.parameters(), **kwargs), None

    def on_converged(self) -> bool:
        r"""Check convergence criteria."""
        values = self.loss_values
        if not values:
            return False
        value = values[-1]
        epsilon = abs(self.min_delta * value) if self.min_delta < 0 else self.min_delta
        slope = slope_of_least_squares_fit(values)
        if abs(slope) < epsilon:
            return True
        return value < self.min_value

    def _loss_terms_of_type(self, loss_type: type) -> dict[str, Module]:
        r"""Get dictionary of loss terms of a specifictype."""
        return {name: module for name, module in self.loss_terms.items() if isinstance(module, loss_type)}  # type: ignore

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

    def _weighted_sum(self, losses: dict[str, Tensor]) -> Tensor:
        r"""Compute weighted sum of loss terms."""
        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        weights = self.weights
        for name, value in losses.items():
            w = weights.get(name, 1.0)
            if not isinstance(w, str):
                value = w * value
            loss += value.sum()
        return loss

    def on_loss(self, grid_transform: SequentialTransform, target: Image, source: Image):
        r"""Evaluate pairwise image registration loss."""
        target_data = target.tensor()
        result = {}
        losses = {}
        misc_excl = set()
        # Transform target grid points
        x = target.grid().coords(device=target_data.device).unsqueeze(0)  # grid_points
        y: Tensor = grid_transform(x, grid=True)
        assert len(self.loss_terms) != 0, "No losses defined"
        ### Sum of pairwise image dissimilarity terms ###
        data_terms = self._loss_terms_of_type(PairwiseImageLoss)
        misc_excl |= set(data_terms.keys())
        if data_terms:
            moved_data = self._sample_image(y, source.tensor())
            if self.source_mask is not None and self.target_mask is not None:
                moved_mask = self._sample_image(y, self.source_mask)
                mask = overlap_mask(moved_mask, self.target_mask)
            else:
                mask = None
            for name, term in data_terms.items():
                losses[name] = term(moved_data, target_data, mask=mask)
            result["source"] = moved_data
            result["target"] = target_data
            result["mask"] = mask
        ## Sum of pairwise point set distance terms
        dist_terms = self._loss_terms_of_type(PointSetDistance)
        misc_excl |= set(dist_terms.keys())
        ldist_terms = {k: v for k, v in dist_terms.items() if isinstance(v, LandmarkPointDistance)}
        dist_terms = {k: v for k, v in dist_terms.items() if k not in ldist_terms}
        if dist_terms:
            if self.source_pset is None:
                raise RuntimeError(f"{type(self).__name__}() missing source point set")
            if self.target_pset is None:
                raise RuntimeError(f"{type(self).__name__}() missing target point set")
            # Points are sampled in reverse
            t = self.transform(self.target_pset)
            for name, term in dist_terms.items():
                losses[name] = term(t, self.source_pset)
        if ldist_terms:
            if self.source_landmarks is None:
                raise RuntimeError(f"{type(self).__name__}() missing source landmarks")
            if self.target_landmarks is None:
                raise RuntimeError(f"{type(self).__name__}() missing target landmarks")
            t = self.transform(self.target_landmarks)
            for name, term in ldist_terms.items():
                losses[name] = term(t, self.source_landmarks)
        ### Sum of displacement field regularization terms ###
        # Buffered vector fields
        disp_terms = self._loss_terms_of_type(DisplacementLoss)
        misc_excl |= set(disp_terms.keys())
        if len(disp_terms) != 0:
            variables = defaultdict(list)
            for name, buf in self.transform.named_buffers():
                if buf.requires_grad:
                    var = name.rsplit(".", 1)[-1]
                    variables[var].append(buf)
            variables["w"] = [U.move_dim(y - x, -1, 1)]
            for name, term in disp_terms.items():
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
        bspline_transforms = self._transforms_of_type(BSplineTransform)
        bspline_terms = self._loss_terms_of_type(BSplineLoss)
        misc_excl |= set(bspline_terms.keys())
        for name, term in bspline_terms.items():
            value = torch.tensor(0, dtype=torch.float, device=self.device)
            for bspline_transform in bspline_transforms:
                value += term(bspline_transform.data())
            losses[name] = value
        ### Sum of parameters loss terms ###
        params_terms = self._loss_terms_of_type(ParamsLoss)
        misc_excl |= set(params_terms.keys())
        for name, term in params_terms.items():
            value = torch.tensor(0, dtype=torch.float, device=self.device)
            count = 0
            for params in self.transform.parameters():
                value += term(params)
                count += 1
            if count > 1:
                value /= count
            losses[name] = value
        # Sum of other regularization terms
        misc_terms = {k: v for k, v in self.loss_terms.items() if k not in misc_excl}
        for name, term in misc_terms.items():
            losses[name] = term()
        # Calculate total loss
        result["losses"] = losses
        result["weights"] = self.weights
        result["loss"] = self._weighted_sum(losses)
        return result

    def on_step(
        self,
        grid_transform: SequentialTransform,
        target_image: Image,
        source_image: Image,
        opt,
        scheduler,
        num_steps,
    ) -> torch.Tensor:
        r"""Perform one registration step.

        Returns:
            Loss value prior to taking gradient step.

        """
        with OptimizerWrapper(opt, scheduler):
            result = self.on_loss(grid_transform, target_image, source_image)
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
        opt, lr_sq = self.on_optimizer(grid_transform)
        self.optimizer = opt
        if isinstance(self.max_steps, int):
            max_steps = self.max_steps
        elif len(self.max_steps) >= level:
            max_steps = self.max_steps[-1]
        elif len(self.max_steps) >= level:
            max_steps = self.max_steps[level]

        return self.run_level(grid_transform, target_image, source_image, opt, lr_sq, level, max_steps, sampling)

    def run_level(
        self,
        grid_transform: SequentialTransform,
        target_image: Image,
        source_image: Image,
        opt: Optimizer,
        lr_sq: LRScheduler | None,
        level,
        max_steps,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
    ):
        ## Perform registration at current resolution level
        ## Initialize optimizer
        self.loss_values = []
        self.loss_terms = {a: l.to(self.device) for a, l in self.loss_terms.items()}
        num_steps = 0
        ##
        target_grid = target_image.grid()
        source_grid = source_image.grid()
        self.transform = grid_transform
        self._sample_image = SampleImage(
            target=target_grid, source=source_grid, sampling=sampling, padding=PaddingMode.ZEROS, align_centers=False
        ).to(self.device)
        grad_sigma = self.smooth_grad

        _eval_hooks = copy(self._eval_hooks)
        _step_hooks = copy(self._step_hooks)
        if isinstance(self.transform, NonRigidTransform) and grad_sigma > 0:
            self.register_eval_hook(smooth_grad_hook(self.transform, sigma=grad_sigma))
        self.register_eval_hook(normalize_grad_hook(self.transform))
        if self.verbose > 2:
            self.register_eval_hook(print_eval_loss_hook_tqdm(level, max_steps))
        elif self.verbose > 1:
            self.register_step_hook(print_step_loss_hook_tqdm(level, max_steps))

        while num_steps < max_steps and not self.on_converged():
            value = self.on_step(grid_transform, target_image, source_image, opt, lr_sq, num_steps)
            num_steps += 1
            if math.isnan(value):
                raise RuntimeError(f"NaN value in registration loss at gradient step {num_steps}")
            if math.isinf(value):
                raise RuntimeError(f"Inf value in registration loss at gradient step {num_steps}")
            self.loss_values.append(value)
            if self.max_history is not None and len(self.loss_values) > self.max_history:
                self.loss_values.pop(0)

        self._eval_hooks = _eval_hooks
        self._step_hooks = _step_hooks

    def _load_initial_transform(self, transform: SpatialTransform):
        if self.model_init:
            if self.verbose > 1:
                print(f"Fitting '{self.model_init}'...")
            disp_field = FlowField.read(self.model_init).to(device=self.device)
            assert isinstance(disp_field, FlowField)
            transform = transform.to(device=self.device).fit(disp_field.batch())
            del disp_field
        return transform

    def register_eval_hook(self, hook: Callable[["DeepaliPairwiseImageTrainer", int, int, RegistrationResult], None]) -> RemovableHandle:
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

    def register_step_hook(self, hook: Callable[["DeepaliPairwiseImageTrainer", int, int, float], None]) -> RemovableHandle:
        r"""Registers a gradient step hook.

        The hook will be called every time after a gradient step of the optimizer.

            hook(self, num_steps: int, num_evals: int, loss: float) -> None

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``

        """
        handle = RemovableHandle(self._step_hooks)
        self._step_hooks[handle.id] = hook
        return handle

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
            grid_transform = self._run_level(grid_transform, target_image, source_image, 0)
        else:
            ## loop pyramid
            source_pyramid = self.source_pyramid
            target_pyramid = self.target_pyramid
            target_mask = self.target_mask
            if self.target_mask is not None:
                target_mask_pyramid = self._pyramid(self.target_mask)
            assert target_pyramid is not None
            finest_level = self.finest_level
            coarsest_level = self.coarsest_level
            source_grid = source_pyramid[finest_level].grid()
            finest_grid = target_pyramid[finest_level].grid()
            coarsest_grid = target_pyramid[coarsest_level].grid()
            post_transform = get_post_transform(finest_grid, source_grid, self.align)

            transform_downsample = self.model_args.pop("downsample", 0)
            transform_grid = coarsest_grid.downsample(transform_downsample)
            transform = self.on_make_transform(self.transform_name, grid=transform_grid, groups=1, **self.model_args)
            transform = self._load_initial_transform(transform)
            grid_transform = SequentialTransform(transform, post_transform)
            grid_transform = grid_transform.to(device=self.device)
            ## Perform coarse-to-fine multi-resolution registration

            for level in range(coarsest_level, finest_level - 1, -1):
                target_image = target_pyramid[level]
                source_image = source_pyramid[level]
                if self.target_mask is not None:
                    self.target_mask = torch.ceil(target_mask_pyramid[level]).to(dtype=torch.int8)
                ## Initialize transformation
                if level != coarsest_level:
                    start = timer()
                    transform_grid = target_image.grid().downsample(transform_downsample)
                    transform.grid_(transform_grid)
                    if self.verbose > 3:
                        print(f"Subdivided control point grid in {timer() - start:.3f}s")
                grid_transform.grid_(target_image.grid())
                self._run_level(grid_transform, target_image, source_image, level)
            if self.verbose > 3:
                print(f"Registered images in {timer() - start_reg:.3f}s")
            if self.verbose > 0:
                print()
            self.target_mask = target_mask
            return grid_transform
