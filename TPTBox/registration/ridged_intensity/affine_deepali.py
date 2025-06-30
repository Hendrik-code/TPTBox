from __future__ import annotations

# pip install hf-deepali
import json
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torch.optim
import yaml
from deepali import spatial
from deepali.core import Axes, PathStr, Sampling
from deepali.core import Grid as Deepali_Grid
from deepali.data import Image as deepaliImage
from deepali.losses import (
    BSplineLoss,
    DisplacementLoss,
    LandmarkPointDistance,
    PairwiseImageLoss,
    ParamsLoss,
    PointSetDistance,
)
from deepali.losses.functional import mse_loss, ncc_loss
from deepali.modules import TransformImage
from deepali.spatial import SpatialTransform
from tqdm import tqdm

from TPTBox import NII, POI, Image_Reference, to_nii
from TPTBox.core.compat import zip_strict
from TPTBox.core.internal.deep_learning_utils import DEVICES, get_device
from TPTBox.core.nii_poi_abstract import Grid as TPTBox_Grid
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.registration.deepali.deepali_model import General_Registration
from TPTBox.registration.deepali.deepali_trainer import PairwiseImageLoss
from TPTBox.registration.ridged_points import Point_Registration


def center_of_mass(tensor):
    grid = torch.meshgrid([torch.arange(s, device=tensor.device) for s in tensor.shape], indexing="ij")
    com = torch.stack([(tensor * g).sum() / tensor.sum() for g in grid])
    return com


class Tether(PairwiseImageLoss):
    def __init__(self, delta=10, uniq=False, remember=True, remember_c=10, max_v=1, *args, **kwargs) -> None:
        self.delta = delta
        self.uniq = uniq
        self.remember = remember
        self.remember_c = remember_c
        self.count = 0
        self.max_v = max_v
        super().__init__(*args, **kwargs)

    def forward(self, source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        if self.count != 0:
            self.count -= 1
            return torch.zeros(1, device=source.device)
        if self.uniq:
            loss = torch.zeros(1, device=source.device)
            k = 0
            target = (target * self.max_v).round(decimals=-1)
            source = (source * self.max_v).round(decimals=-1)
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
            com_fixed = center_of_mass(target)
            com_warped = center_of_mass(source)
            l_com = torch.norm(com_fixed - com_warped)
            if l_com < self.delta:
                l_com = torch.zeros(1, device=source.device)
                if self.remember:
                    self.count = 10
            return l_com  # type: ignore


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
            from copy import deepcopy

            self.best_transform = deepcopy(self.transform)
            self.best2 = value

        if self.early_stopping <= self.patience:
            return False
        self.transform = self.best_transform
        return True
