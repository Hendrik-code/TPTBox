# pip install hf-deepali
import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from deepali.core import Axes
from deepali.core import Grid as Deepali_Grid
from deepali.core.typing import Device
from deepali.data import Image as deepaliImage
from deepali.modules import TransformImage
from deepali.spatial import SpatialTransform
from torch import device

from TPTBox import NII, POI, Image_Reference, to_nii
from TPTBox.core.nii_poi_abstract import Grid as TPTBox_Grid
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.registration.deformable._deepali import deform_reg_pair

cuda = device("cuda")


def _load_config(path):
    r"""Load registration parameters from configuration file."""
    config_path = Path(path).absolute()
    config_text = config_path.read_text()
    if config_path.suffix == ".json":
        return json.loads(config_text)
    return yaml.safe_load(config_text)


def _warp_image(
    source_image: deepaliImage, target_grid: Deepali_Grid, transform: SpatialTransform, mode="linear", device="cuda"
) -> torch.Tensor:
    warp_func = TransformImage(target=target_grid, source=target_grid, sampling=mode, padding=source_image.min()).to(device)
    with torch.inference_mode():
        data = warp_func(transform.tensor(), source_image)
    return data


def _warp_poi(poi_moving: POI, target_grid: TPTBox_Grid, transform: SpatialTransform, align_corners, device="cuda") -> POI:
    keys: list[tuple[int, int]] = []
    points = []
    for key, key2, (x, y, z) in poi_moving.items():
        keys.append((key, key2))
        points.append((x, y, z))
        print(key, key2, (x, y, z))
    with torch.inference_mode():
        data = torch.Tensor(points)
        transform.to(device)
        data2 = data
        data = transform.inverse(update_buffers=True).points(
            data,
            axes=Axes.GRID,
            to_axes=Axes.GRID,
            grid=poi_moving.to_deepali_grid(align_corners),
            to_grid=target_grid.to_deepali_grid(align_corners),
        )
        print(data2 - data)

    out_poi = target_grid.make_empty_POI()
    for (key, key2), (x, y, z) in zip(keys, data.cpu(), strict=True):
        print(key, key2, x, y, z)
        out_poi[key, key2] = (x.item(), y.item(), z.item())
    return out_poi


class Deformable_Registration:
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
        normalize: Literal["MRI", "CT"] | None = None,
        quantile: float = 0.95,
        reference_image: Image_Reference | None = None,
        device: Device | str | None = cuda,
        align_corners: bool = False,
        verbose=1,
    ) -> None:
        """
        Initialize the deformable registration process.

        Args:
            fixed_image (Image_Reference): The fixed image to which the moving image is registered.
            moving_image (Image_Reference): The moving image to be registered.
            normalize (Literal["MRI", "CT"] | None): Normalization type; supports "MRI" or "CT" or no normalization.
            quantile (float): Quantile for intensity normalization; recommended 0.95 for MRI.
            reference_image (Image_Reference | None): Optional reference image for resampling.
            device (Device | None): The computational device for the process, default is CUDA.
            align_corners (bool): Whether to align the corners during grid resampling.
        """

        fix = to_nii(fixed_image).copy()
        mov = to_nii(moving_image).copy()

        # Preprocessing
        if normalize == "MRI":
            fix.normalize_(quantile=quantile).clamp(0, 1)
            mov.normalize_(quantile=quantile).clamp(0, 1)
        elif normalize == "CT":
            fix.clamp_(-1024, 1024).normalize_().clamp(0, 1)
            mov.clamp_(-1024, 1024).normalize_().clamp(0, 1)

        if reference_image is None:
            reference_image = fix
        else:
            fix = fix.resample_from_to(reference_image)

        # Resample and save images
        source = mov.resample_from_to_(reference_image)

        # Load configuration and perform registration
        deformable_simple = _load_config(Path(__file__).parent / "settings.json")
        self.target_grid = fix.to_gird()
        self.input_grid = mov.to_gird()
        self.align_corners = align_corners

        self.transform = deform_reg_pair.register_pairwise(
            target=fix.copy(), source=source.copy(), config=deformable_simple, device=device, verbose=verbose
        )

    @torch.no_grad()
    def transform_nii(self, img: NII, device="cuda", target: Has_Grid | None = None) -> NII:
        """
        Apply the computed transformation to a given NII image.

        Args:
            img (NII): The NII image to be transformed.

        Returns:
            NII: The transformed image as an NII object.
        """
        target_grid_nii = self.target_grid if target is None else target
        target_grid = target_grid_nii.to_deepali_grid(self.align_corners)
        source_image = img.resample_from_to(self.input_grid).to_deepali()

        data = _warp_image(source_image, target_grid, self.transform, "nearest" if img.seg else "linear", device=device).squeeze()
        data: torch.Tensor = data.permute(*torch.arange(data.ndim - 1, -1, -1))  # type: ignore
        out = target_grid_nii.make_nii(data.detach().cpu().numpy(), img.seg)
        return out

    def transform_poi(self, poi: POI, device="cuda"):
        source_image = poi.resample_from_to(self.target_grid)
        data = _warp_poi(source_image, self.target_grid, self.transform, self.align_corners, device=device)
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

    def save(self, path: str | Path):
        with open(path, "wb") as w:
            pickle.dump(
                (self.transform, self.target_grid, self.input_grid, self.align_corners),
                w,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as w:
            transform, grid, mov, align_corners = pickle.load(w)
        self = cls.__new__(cls)
        self.transform = transform
        self.target_grid = grid
        self.input_grid = mov
        self.align_corners = align_corners
        return self


def test1(run=False):
    """
    Main function to test the deformable registration process.
    Loads reference and moving images, performs registration, and saves the output.
    """
    from TPTBox import NII

    fixed = NII.load("/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-stitched_acq-ax_part-water_vibe.nii.gz", False)
    moving = NII.load("/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz", False)
    poi = moving.make_empty_POI()

    poi[123, 44] = (93 - 1, 51 - 1, 26 - 1)
    poi[123, 45] = (77 - 1, 55 - 1, 20 - 1)
    if run:
        deform = Deformable_Registration(fixed, moving, reference_image=moving)
        out = deform.transform_nii(moving.copy())
        out.save("/media/data/robert/code/TPTBox/tmp/Affine_source.nii.gz")
        deform.save("/media/data/robert/code/TPTBox/tmp/deformation.pkl")
    deform2 = Deformable_Registration.load("/media/data/robert/code/TPTBox/tmp/deformation.pkl")
    out = deform2.transform_nii(moving)
    out.save("/media/data/robert/code/TPTBox/tmp/Affine_source2.nii.gz")
    mov = moving.copy()
    mov.seg = True
    mov[mov > -10000] = 0
    mov[int(poi[123, 44][0]), int(poi[123, 44][1]), int(poi[123, 44][2])] = 1
    mov[int(poi[123, 45][0]), int(poi[123, 45][1]), int(poi[123, 45][2])] = 2
    poi = deform2.transform_poi(poi)

    out = deform2.transform_nii(mov)

    print(poi.round(1).resample_from_to(mov)[123, 44])
    print([x.item() for x in np.where(out == 1)])

    print(poi.round(1).resample_from_to(mov)[123, 45])
    print([x.item() for x in np.where(out == 2)])
    print("main")
    out.save("/media/data/robert/code/TPTBox/tmp/Affine_source2_.nii.gz")


def test2(run=False):
    """
    Main function to test the deformable registration process.
    Loads reference and moving images, performs registration, and saves the output.
    """
    from TPTBox import NII

    # ref = NII.load("/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-stitched_acq-ax_part-water_vibe.nii.gz", False)
    mov = NII.load("/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz", False)
    poi = mov.make_empty_POI()
    poi[123, 44] = (68 - 1, 61 - 1, 17 - 1)
    poi[123, 45] = (77 - 1, 55 - 1, 20 - 1)
    mov2 = NII.load("/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz", False)
    mov2[mov2 != 0] = 0
    mov2[:-3, :, :] = mov2[3:, :, :]
    # mov2[:, :-3, :] = mov2[:, 3:, :]
    # mov2[:, :, :-3] = mov2[:, :, 3:]

    # mov2 = mov2.rescale((2, 2, 2))
    if run:
        deform = Deformable_Registration(mov2, mov, reference_image=mov)
        out = deform.transform_nii(mov.copy())
        out.save("/media/data/robert/code/TPTBox/tmp/Affine_source.nii.gz")
        deform.save("/media/data/robert/code/TPTBox/tmp/deformation.pkl")
    deform2 = Deformable_Registration.load("/media/data/robert/code/TPTBox/tmp/deformation.pkl")

    mov.seg = True
    mov[mov > -10000] = 0
    mov[int(poi[123, 44][0]), int(poi[123, 44][1]), int(poi[123, 44][2])] = 1
    mov[int(poi[123, 45][0]), int(poi[123, 45][1]), int(poi[123, 45][2])] = 2
    out = deform2.transform_nii(mov)
    poi = deform2.transform_poi(poi)

    print(poi.round(1).resample_from_to(mov)[123, 44])
    print([x.item() for x in np.where(out == 1)])

    print(poi.round(1).resample_from_to(mov)[123, 45])
    print([x.item() for x in np.where(out == 2)])

    out.save("/media/data/robert/code/TPTBox/tmp/Affine_source2_.nii.gz")


if __name__ == "__main__":
    test1()
