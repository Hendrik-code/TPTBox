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
from deepali.core import Axes, PathStr
from deepali.core import Grid as Deepali_Grid
from deepali.data import Image as deepaliImage
from deepali.modules import TransformImage
from deepali.spatial import SpatialTransform

from TPTBox import NII, POI, Image_Reference, to_nii
from TPTBox.core.compat import zip_strict
from TPTBox.core.internal.deep_learning_utils import DEVICES, get_device
from TPTBox.core.nii_poi_abstract import Grid as TPTBox_Grid
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.registration.deepali.deepali_trainer import LOSS, DeepaliPairwiseImageTrainer


def center_of_mass(tensor):
    grid = torch.meshgrid([torch.arange(s, device=tensor.device) for s in tensor.shape], indexing="ij")
    com = torch.stack([(tensor * g).sum() / tensor.sum() for g in grid])
    return com


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def _load_config(path):
    r"""Load registration parameters from configuration file."""
    config_path = Path(path).absolute()
    config_text = config_path.read_text()
    if config_path.suffix == ".json":
        return json.loads(config_text)
    return yaml.safe_load(config_text)


default_device = torch.device("cuda:0")


def _warp_image(
    source_image: deepaliImage, target_grid: Deepali_Grid, transform: SpatialTransform, mode="linear", device=default_device, inverse=False
) -> torch.Tensor:
    if inverse:
        transform = transform.inverse(update_buffers=True)
    warp_func = TransformImage(target=target_grid, source=target_grid, sampling=mode, padding=source_image.min()).to(device)
    with torch.inference_mode():
        data = warp_func(transform.tensor(), source_image.to(device))
    return data


def _warp_poi(
    poi_moving: POI, target_grid: TPTBox_Grid, transform: SpatialTransform, align_corners, device=default_device, inverse=True
) -> POI:
    keys: list[tuple[int, int]] = []
    points = []
    for key, key2, (x, y, z) in poi_moving.items():
        keys.append((key, key2))
        points.append((x, y, z))
        print(key, key2, (x, y, z))
    with torch.inference_mode():
        data = torch.Tensor(points)
        transform.to(device)
        # data2 = data
        if inverse:
            transform = transform.inverse(update_buffers=True)
        data = transform.points(
            data.to(device),
            axes=Axes.GRID,
            to_axes=Axes.GRID,
            grid=poi_moving.to_deepali_grid(align_corners),
            to_grid=target_grid.to_deepali_grid(align_corners),
        )

    out_poi = target_grid.make_empty_POI()
    for (key, key2), (x, y, z) in zip_strict(keys, data.cpu()):
        out_poi[key, key2] = (x.item(), y.item(), z.item())
    return out_poi


class General_Registration(DeepaliPairwiseImageTrainer):
    """
    A class for performing deformable registration between a fixed and moving image.

    Attributes:
        transform (torch.Tensor): The transformation matrix resulting from the registration.
        ref_nii (NII): Reference NII object used for registration.
        grid (torch.Tensor): Target grid for image warping.
        mov (NII): Processed version of the moving image.
    """

    def __init__(
        self,
        fixed_image: Image_Reference,
        moving_image: Image_Reference,
        reference_image: Image_Reference | None = None,
        source_pset=None,
        target_pset=None,
        source_landmarks: POI | None = None,
        target_landmarks: POI | None = None,
        # source_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration source
        # target_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration target
        device: Union[torch.device, str, int] | None = None,
        gpu=0,
        ddevice: DEVICES = "cuda",
        # foreground_mask
        fixed_mask: Image_Reference | None = None,
        moving_mask: Image_Reference | None = None,
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
        transform_name: str = "SVFFD",  # Names that are defined in deepali.spatial.LINEAR_TRANSFORMS and deepali.spatialNONRIGID_TRANSFORMS. Override on_make_transform for finer control
        transform_args: dict | None = None,
        transform_init: PathStr | None = None,  # reload initial flowfield from file
        optim_name="Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer control
        lr: float | list[float] = 0.01,  # Learning rate
        optim_args=None,  # args of Optimizer with out lr
        smooth_grad=0.0,
        verbose=99,
        max_steps: int | Sequence[int] = 250,  # Early stopping.  override on_converged finer control
        max_history: int | None = None,
        min_value=0.0,  # Early stopping.  override on_converged finer control
        min_delta=0.0,  # Early stopping.  override on_converged finer control
        loss_terms: list[LOSS | str] | dict[str, LOSS] | dict[str, str] | dict[str, tuple[str, dict]] | None = None,
        weights: list[float] | dict[str, float] | None = None,
        auto_run=True,
    ) -> None:
        if device is None:
            # self.gpu = gpu
            # self.ddevice: DEVICES = ddevice
            device = get_device(ddevice, gpu)
        fix = to_nii(fixed_image).copy()
        mov = to_nii(moving_image).copy()
        if reference_image is None:
            reference_image = fix
        else:
            fix = fix.resample_from_to(reference_image)
        ## Resample and save images
        source = mov  # .resample_from_to_(reference_image)
        ## Load configuration and perform registration
        self.target_grid = fix.to_gird()
        self.input_grid = mov.to_gird()
        self.source_landmarks_poi = source_landmarks
        self.target_landmarks_poi = target_landmarks
        super().__init__(
            source=source.to_deepali(),
            target=fix.to_deepali(),
            source_pset=source_pset,
            target_pset=target_pset,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            device=device,
            target_mask=to_nii(fixed_mask, True).to_deepali() if fixed_mask is not None else None,
            source_mask=to_nii(moving_mask, True).to_deepali() if moving_mask is not None else None,
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
        if auto_run:
            self.run()

    # def on_transform_update(self, transform: SpatialTransform):
    #    if self.source_landmarks_poi is not None and self.target_landmarks_poi is not None:
    #        lm = self.source_landmarks_poi.copy()
    #        tm = self.target_landmarks_poi.copy()
    #        for k in lm.keys().copy():
    #            if k not in tm:
    #                lm.remove_(k)
    #        for k in tm.keys().copy():
    #            if k not in lm:
    #                tm.remove_(k)
    #        self.source_landmarks = self.poi_to_deepali(lm, transform)
    #        self.target_landmarks = self.poi_to_deepali(tm, transform)
    #        assert self.source_landmarks.shape == self.target_landmarks.shape, (self.source_landmarks.shape, self.target_landmarks.shape)

    # def poi_to_deepali(self, poi: POI, transform: SpatialTransform):
    #    import torch
    #    from deepali.core import Axes
    #    keys: list[tuple[int, int]] = []
    #    points = []
    #    for key, key2, (x, y, z) in poi.items(sort=True):
    #        keys.append((key, key2))
    #        points.append((x, y, z))
    #        print(key, key2)
    #    with torch.inference_mode():
    #        data = torch.Tensor(points).unsqueeze(0)
    #        # data = (
    #        #    poi.to_deepali_grid()
    #        #    .transform_points(data, axes=Axes.GRID, to_grid=transform.grid(), to_axes=transform.axes(), decimals=None)
    #        #    .unsqueeze(0)
    #        # )
    #    return data.clone()

    @torch.no_grad()
    def transform_nii(
        self,
        img: NII,
        gpu: int | None = None,
        ddevice: DEVICES | None = None,
        target: Has_Grid | None = None,
        align_corners=True,
        inverse=False,
    ) -> NII:
        """
        Apply the computed transformation to a given NII image.

        Args:
            img (NII): The NII image to be transformed.

        Returns:
            NII: The transformed image as an NII object.
        """
        device = get_device(ddevice, 0 if gpu is None else gpu) if ddevice is not None else self.device
        target_grid_nii = self.target_grid if target is None else target
        target_grid = target_grid_nii.to_deepali_grid(align_corners)
        source_image = img.resample_from_to(self.input_grid).to_deepali()
        data = _warp_image(
            source_image, target_grid, self.transform, "nearest" if img.seg else "linear", device=device, inverse=inverse
        ).squeeze()
        data: torch.Tensor = data.permute(*torch.arange(data.ndim - 1, -1, -1))  # type: ignore
        out = target_grid_nii.make_nii(data.detach().cpu().numpy(), img.seg)
        return out

    def transform_poi(self, poi: POI, gpu: int | None = None, ddevice: DEVICES | None = None, align_corners=True, inverse=True):
        device = get_device(ddevice, 0 if gpu is None else gpu) if ddevice is not None else self.device
        source_image = poi.resample_from_to(self.target_grid)
        data = _warp_poi(source_image, self.target_grid, self.transform, align_corners, device=device, inverse=inverse)
        return data.resample_from_to(self.target_grid)

    def __call__(self, *args, **kwds) -> NII:
        """
        Call method to apply the transformation using the transform_nii method.

        Args:
            *args: Positional arguments for the transform_nii method.
            **kwds: Keyword arguments for the transform_nii method.

        Returns:
            NII: The transformed image.
        """
        return self.transform_nii(*args, **kwds)

    def get_dump(self):
        return (self.transform, self.target_grid, self.input_grid)

    def save(self, path: str | Path):
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path, gpu=0, ddevice: DEVICES = "cuda"):
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w), gpu, ddevice)

    @classmethod
    def load_(cls, w, gpu=0, ddevice: DEVICES = "cuda"):
        transform, grid, mov = w
        self = cls.__new__(cls)
        self.transform = transform
        self.target_grid = grid
        self.input_grid = mov
        self.device = get_device(ddevice, gpu)
        return self
