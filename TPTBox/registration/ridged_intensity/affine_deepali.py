from __future__ import annotations

from abc import ABCMeta, abstractmethod

# pip install hf-deepali
from collections.abc import Sequence
from copy import deepcopy
from typing import Literal, Union

import torch
import torch.optim
from deepali import spatial
from deepali.core import PathStr, Sampling
from deepali.data import Image as deepaliImage
from deepali.losses import (
    PairwiseImageLoss,
)
from deepali.losses.functional import ncc_loss
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm

from TPTBox import Image_Reference
from TPTBox.core.internal.deep_learning_utils import DEVICES, get_device
from TPTBox.registration.deepali.deepali_model import General_Registration
from TPTBox.registration.deepali.deepali_trainer import PairwiseImageLoss


class PairwiseSegImageLoss(Module, metaclass=ABCMeta):
    r"""Base class of pairwise image dissimilarity criteria."""

    @abstractmethod
    def forward(self, source: Tensor, target: Tensor, mask: [Tensor] | None = None) -> Tensor:
        r"""Evaluate image dissimilarity loss."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")


def center_of_mass(tensor):
    grid = torch.meshgrid([torch.arange(s, device=tensor.device) for s in tensor.shape], indexing="ij")
    t = tensor / tensor.sum()
    com = torch.stack([(t * g).sum() for g in grid])
    return com


class Tether_single(PairwiseImageLoss):
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:  # noqa: ARG002
        com_fixed = center_of_mass(target)
        com_warped = center_of_mass(source)
        l_com = torch.norm(com_fixed - com_warped)
        if l_com < 10:
            l_com = source.sum() * 0
        l_com = torch.nan_to_num(l_com, nan=0)
        return l_com  # type: ignore


def center_of_mass_cc(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the center of mass for each channel in a (B, C, X, Y, Z) tensor.
    Returns a tensor of shape (B, C, 3) containing the (x, y, z) coordinates per channel.
    """
    dtype = tensor.dtype
    B, C, *spatial_shape = tensor.shape
    tensor = tensor.float()
    grid = torch.meshgrid([torch.arange(s, device=tensor.device) for s in spatial_shape], indexing="ij")  # each g is (X, Y, Z)
    grid = torch.stack(grid, dim=0)  # (3, X, Y, Z)

    # Flatten spatial dims
    tensor_flat = tensor.view(B, C, -1)  # (B, C, X*Y*Z)
    grid_flat = grid.view(3, -1)  # (3, X*Y*Z)

    # Normalize tensor
    norm = tensor_flat.sum(dim=-1, keepdim=True)  # (B, C, 1)
    norm[norm == 0] = 1  # avoid division by zero

    com = torch.einsum("bcn,nm->bcm", tensor_flat, grid_flat.T.to(tensor_flat.dtype)) / norm  # (B, C, 3)
    return com.to(dtype)


class Tether_Seg(PairwiseSegImageLoss):
    def __init__(self, delta=1, *args, **kwargs):
        self.delta = delta
        super().__init__(*args, **kwargs)

    def forward(
        self,
        source: torch.Tensor,  # shape: (B, C, X, Y, Z)
        target: torch.Tensor,  # shape: (B, C, X, Y, Z)
        mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        w = max(target.shape[2:])
        com_fixed = center_of_mass_cc(target)  # (B, C, 3)
        com_warped = center_of_mass_cc(source)  # (B, C, 3)

        l_com = torch.norm(com_fixed - com_warped, dim=-1) / w  # (B, C)

        # Zero out channels with small displacement (<10) or NaNs
        l_com = torch.where(l_com < self.delta, torch.zeros_like(l_com), l_com)
        l_com = torch.nan_to_num(l_com, nan=0.0)

        return l_com.mean()  # type: ignore


class Tether(PairwiseImageLoss):
    def __init__(
        self,
        delta=10,
        uniq=False,
        remember=False,
        remember_c=10,
        max_v=1,
        *args,
        **kwargs,
    ) -> None:
        self.delta = delta
        self.uniq = uniq
        self.remember = remember
        self.remember_c = remember_c
        self.count = 0
        self.max_v = max_v
        super().__init__(*args, **kwargs)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:  # noqa: ARG002
        if self.count != 0:
            self.count -= 1
            return torch.zeros(1, device=source.device)
        if self.uniq:
            loss = torch.zeros(1, device=source.device)
            k = 0
            target = (target * self.max_v).round(decimals=0)
            source = (source * self.max_v).round(decimals=0)
            u = torch.unique(target)
            for i in u:
                if i == 0:
                    continue
                com_fixed = center_of_mass(target == i)
                com_warped = center_of_mass(source == i)
                l_com = torch.norm(com_fixed - com_warped)
                l_com = torch.nan_to_num(l_com, nan=0)
                # print(l_com)
                if l_com > self.delta:
                    loss += l_com
                    k += 1
            # print(loss / k, k, len(u))
            if k == 0:
                if self.remember:
                    self.count = 10
                return loss
            return loss / k
        else:
            com_fixed = center_of_mass(target != 0)
            com_warped = center_of_mass(source != 0)
            l_com = torch.norm(com_fixed - com_warped)
            if l_com < self.delta:
                l_com = torch.zeros(1, device=source.device)
                if self.remember:
                    self.count = 10
            l_com = torch.nan_to_num(l_com, nan=0)
            return l_com  # type: ignore


def subsample_coords(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    If `coords` has more than `k` rows, return a random subset of size `k`;
    otherwise return `coords` unchanged.

    Uses sampling *without* replacement (`torch.randperm`), so every
    coordinate appears at most once.  Works entirely on-device.
    """
    n = coords.size(0)
    if n <= k:
        return coords
    idx = torch.randperm(n, device=coords.device)[:k]
    return coords[idx]


class DISTANCE_to_TARGET(PairwiseImageLoss):
    def __init__(
        self,
        max_v=1,
        res_gt=4,
        *args,
        **kwargs,
    ) -> None:
        self.max_v = max_v
        self.res_gt = res_gt
        super().__init__(*args, **kwargs)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Chamfer-style distance loss for mis-labelled voxels.

        Parameters
        ----------
        source : (D, H, W)[, …]  torch.Tensor
            Model prediction in label form (one channel per voxel).
        target : (D, H, W)[, …]  torch.Tensor
            Ground-truth labels.
        max_v  : float, default 1
            Same scale factor you use elsewhere to map the continuous range [0, 1]
            back to integer labels. Set to 1 if `source` and `target` are already
            integer encoded.

        Returns
        -------
        torch.Tensor  scalar
            The mean distance (in voxel units) from every wrongly predicted voxel
            to the nearest correct voxel of the same class in the target.
        """
        max_v = self.max_v
        device = source.device
        # Discretise
        src = (source * max_v).round().short()  # .long()
        tgt = (target * max_v).round().short()  # .long()

        classes = torch.unique(tgt)
        classes = classes[classes != 0]  # skip background label 0

        if classes.numel() == 0:
            return torch.zeros(1, device=device)

        per_class_losses = []

        for c in classes:
            wrong_mask = (src == c) & (tgt != c)  # voxels we predicted as c but shouldn't
            if not wrong_mask.any():
                continue  # no penalty if we never made that error
            res_gt = self.res_gt
            gt_mask = tgt[..., ::res_gt, ::res_gt, ::res_gt] == c
            if not gt_mask.any():
                # Optional: if the class is missing in GT you could add
                # a constant penalty or skip it. Here we skip.
                continue

            # Coordinates of voxels
            wrong_coords = torch.nonzero(wrong_mask, as_tuple=False).float()
            # print(gt_mask.shape)
            #
            gt_coords = torch.nonzero(gt_mask, as_tuple=False).float()

            # Pairwise distances (N_wrong, N_gt); differentiable
            d = torch.cdist(subsample_coords(wrong_coords, 5000), gt_coords)
            min_dists = d.min(dim=1).to_numpy()  # (N_wrong,)

            per_class_losses.append(min_dists.mean())

        if not per_class_losses:
            # Nothing to penalise - perfect overlap
            return torch.zeros(1, device=device)

        # Average over foreground classes
        return torch.stack(per_class_losses).mean()


class Rigid_Registration_with_Tether(General_Registration):
    def __init__(
        self,
        fixed_image: Image_Reference,
        moving_image: Image_Reference,
        reference_image: Image_Reference | None = None,
        device: Union[torch.device, str, int] | None = None,
        gpu=0,
        ddevice: DEVICES = "cuda",
        # foreground_mask
        fixed_mask=None,
        moving_mask=None,
        # normalize
        normalize_strategy: Literal["auto", "CT", "MRI"] | None = None,
        # Pyramid
        pyramid_levels: int | None = None,  # 1/None = no pyramid; int: number of stacks, tuple from to (0 is finest)
        finest_level: int = 0,
        coarsest_level: int | None = None,
        pyramid_finest_spacing: Sequence[int] | torch.Tensor | None = None,
        pyramid_min_size=16,
        dims=("x", "y", "z"),
        align=False,
        transform_name: str = "RigidTransform",  # Names that are defined in deepali.spatial.LINEAR_TRANSFORMS and deepali.spatialNONRIGID_TRANSFORMS. Override on_make_transform for finer control
        transform_args: dict | None = None,
        transform_init: PathStr | None = None,  # reload initial flowfield from file
        optim_name="Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer control
        lr=0.01,  # Learning rate
        optim_args=None,  # args of Optimizer with out lr
        smooth_grad=0.0,
        verbose=0,
        max_steps: int | Sequence[int] = 250,  # Early stopping.  override on_converged finer control
        max_history: int | None = None,
        min_value=0.0,  # Early stopping.  override on_converged finer control
        min_delta=0.0,  # Early stopping.  override on_converged finer control
        loss_terms=(ncc_loss, None),
        weights=(1, 0.001),
        patience=100,
        patience_delta=0.0,
        desc="RRwT",
    ) -> None:
        self.patience = patience
        self.patience_delta = patience_delta
        if device is None:
            device = get_device(ddevice, gpu)
        self.best = 1000000000
        self.best2 = 1000000000
        self.early_stopping = 0
        self.desc = desc
        super().__init__(
            fixed_image,
            moving_image,
            reference_image,
            device=device,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            normalize_strategy=normalize_strategy,
            pyramid_levels=pyramid_levels,
            finest_level=finest_level,
            coarsest_level=coarsest_level,
            pyramid_finest_spacing=pyramid_finest_spacing,
            pyramid_min_size=pyramid_min_size,
            dims=dims,
            align=align,
            transform_name=transform_name,
            transform_args=transform_args,
            transform_init=transform_init,
            optim_name=optim_name,
            lr=lr,
            optim_args=optim_args,
            smooth_grad=smooth_grad,
            verbose=verbose,
            max_steps=max_steps,
            max_history=max_history,
            min_value=min_value,
            min_delta=min_delta,
            loss_terms=loss_terms,
            weights=weights,
        )

    def run_level(
        self,
        grid_transform: spatial.SequentialTransform,
        target_image: deepaliImage,
        source_image: deepaliImage,
        opt: torch.optim.Optimizer,
        lr_sq,  # noqa: ARG002
        level,  # noqa: ARG002
        max_steps,
        sampling: Union[Sampling, str] = Sampling.LINEAR,  # noqa: ARG002
    ):
        loss_list = []
        self.loss_values = loss_list
        self.transform = grid_transform
        transformer = spatial.ImageTransformer(grid_transform).to(self.device)
        loss = next(iter(self.loss_terms.values()))
        lambda_mse, lambda_com = self.weights.values()
        pbar = tqdm(range(max_steps))
        for _ in pbar:
            if self.on_converged():
                break
            warped_batch = transformer(source_image)
            l_mse = loss(warped_batch, target_image)
            if lambda_com != 0:
                # Compute center-of-mass loss
                com_fixed = center_of_mass(target_image)
                com_warped = center_of_mass(warped_batch)
                l_com = torch.norm(com_fixed - com_warped)
                if l_com < 10:
                    l_com = 0
            else:
                l_com = 0
            l = lambda_mse * l_mse + lambda_com * l_com  # Weighted sum of losses

            loss_list.append(l.item())
            opt.zero_grad()
            l.backward()
            opt.step()
            pbar.desc = f"{self.desc} loss={l_mse.item() * lambda_mse:.5f}, center_of_mass={l_com * lambda_com:.5f}, {self.early_stopping=}, {self.best=}"

    def on_converged(self) -> bool:
        r"""Check convergence criteria."""
        values = self.loss_values
        if not values:
            return False
        value = values[-1]
        if value <= self.best - self.patience_delta:
            self.early_stopping = 0
            self.best = value
        else:
            self.early_stopping += 1

        if value <= self.best2:
            self.best_transform = deepcopy(self.transform)
            self.best2 = value

        if self.early_stopping <= self.patience:
            return False
        self.transform = self.best_transform
        return True
