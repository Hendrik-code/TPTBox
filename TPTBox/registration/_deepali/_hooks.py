r"""Hooks for iterative optimization-based registration engine."""

from collections.abc import Callable
from ctypes import Union
from typing import TYPE_CHECKING

import torch.nn.functional as F  # noqa: N812
from deepali.core import functional as U  # noqa: N812
from deepali.core.kernels import gaussian1d
from deepali.spatial import is_linear_transform
from torch import Tensor

if TYPE_CHECKING:
    from .deepali_trainer import DeepaliPairwiseImageTrainer, RegistrationResult

RegistrationEvalHook = Callable[["DeepaliPairwiseImageTrainer", int, int, "RegistrationResult"], None]
RegistrationStepHook = Callable[["DeepaliPairwiseImageTrainer", int, int, float], None]


def noop(reg: "DeepaliPairwiseImageTrainer", *args, **kwargs) -> None:
    r"""Dummy no-op loss evaluation hook."""


def normalize_linear_grad(reg: "DeepaliPairwiseImageTrainer", *args, **kwargs) -> None:  # noqa: ARG001
    r"""Loss evaluation hook for normalization of linear transformation gradient after backward pass."""
    denom = None
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            max_abs_grad = p.grad.abs().max()
            if denom is None or denom < max_abs_grad:
                denom = max_abs_grad
    if denom is None:
        return
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            p.grad /= denom


def normalize_nonrigid_grad(reg: "DeepaliPairwiseImageTrainer", *args, **kwargs) -> None:  # noqa: ARG001
    r"""Loss evaluation hook for normalization of non-rigid transformation gradient after backward pass."""
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            F.normalize(p.grad, p=2, dim=1, out=p.grad)


def normalize_grad_hook(transform) -> RegistrationEvalHook:
    r"""Loss evaluation hook for normalization of transformation gradient after backward pass."""
    if is_linear_transform(transform):
        return normalize_linear_grad
    return normalize_nonrigid_grad


def _smooth_nonrigid_grad(reg: "DeepaliPairwiseImageTrainer", sigma: float = 1) -> None:
    r"""Loss evaluation hook for Gaussian smoothing of non-rigid transformation gradient after backward pass."""
    if sigma <= 0:
        return
    kernel = gaussian1d(sigma)
    for group in reg.optimizer.param_groups:
        for p in (p for p in group["params"] if p.grad is not None):
            p.grad.copy_(U.conv(p.grad, kernel))


def smooth_grad_hook(transform, sigma: float) -> RegistrationEvalHook:
    r"""Loss evaluation hook for Gaussian smoothing of non-rigid gradient after backward pass."""
    if is_linear_transform(transform):
        return noop

    def fn(reg: "DeepaliPairwiseImageTrainer", *args, **kwargs):  # noqa: ARG001
        return _smooth_nonrigid_grad(reg, sigma=sigma)

    return fn


def print_eval_loss_hook(level: int, max_steps: int) -> RegistrationEvalHook:  # noqa: ARG001
    r"""Get callback function for printing loss after each evaluation."""

    def fn(_: "DeepaliPairwiseImageTrainer", num_steps: int, num_eval: int, result: "RegistrationResult") -> None:
        loss = float(result["loss"])
        message = f"  {num_steps:>4d}:"
        message += f" {loss:>12.05f} (loss)"
        weights: dict[str, Union[str, float]] = result.get("weights", {})
        losses: dict[str, Tensor] = result["losses"]
        for name, value in losses.items():
            if hasattr(value, "mean"):
                value = value.mean()  # noqa: PLW2901
            value = float(value)  # noqa: PLW2901
            weight = weights.get(name, 1.0)
            if isinstance(weight, (list, tuple)):
                weight = weight[level]
            if not isinstance(weight, str):
                value *= weight  # type: ignore  # noqa: PLW2901
            elif "+" in weight:
                weight = f"({weight})"
            message += f", {value:>12.05f} [{weight} * {name}]"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        print(message, flush=True)

    return fn


def print_eval_loss_hook_tqdm(level: int, max_steps: int) -> RegistrationEvalHook:
    r"""Get callback function for printing loss after each evaluation."""
    from tqdm import tqdm

    bar = tqdm(range(max_steps))

    def fn(_: "DeepaliPairwiseImageTrainer", num_steps: int, num_eval: int, result: "RegistrationResult") -> None:
        loss = float(result["loss"])
        message = f"  {num_steps:>4d}:"
        message += f" {loss:>5.05f}4 (loss)"
        weights: dict[str, Union[str, float]] = result.get("weights", {})
        losses: dict[str, Tensor] = result["losses"]
        for name, value in losses.items():
            if hasattr(value, "mean"):
                value = value.mean()  # noqa: PLW2901
            value = float(value)  # noqa: PLW2901
            weight = weights.get(name, 1.0)
            if isinstance(weight, (list, tuple)):
                weight = weight[level]

            if not isinstance(weight, str):
                value *= weight  # type: ignore # noqa: PLW2901
            elif "+" in weight:
                weight = f"({weight})"
            message += f", {value:>5.05f}[{weight}*{name}]"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        # print("...", message, flush=True)
        bar.desc = message
        bar.update(1)

    return fn


def print_step_loss_hook(level: int, max_steps: int) -> RegistrationStepHook:  # noqa: ARG001
    r"""Get callback function for printing loss after each step."""

    def fn(_: "DeepaliPairwiseImageTrainer", num_steps: int, num_eval: int, loss: float) -> None:
        message = f"  {num_steps:>4d}: {loss:>12.05f}"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        print(message, flush=True)

    return fn


def print_step_loss_hook_tqdm(level: int, max_steps: int) -> RegistrationStepHook:  # noqa: ARG001
    r"""Get callback function for printing loss after each step."""

    from tqdm import tqdm

    bar = tqdm(range(max_steps))

    def fn(_: "DeepaliPairwiseImageTrainer", num_steps: int, num_eval: int, loss: float) -> None:
        message = f"  {num_steps:>4d}: {loss:>12.05f}"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        bar.desc = message
        bar.update(1)
        # print(message, flush=True)

    return fn
