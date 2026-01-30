from __future__ import annotations

# pip install hf-deepali
from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
import torch
from deepali.core import PathStr  # pip install hf-deepali
from deepali.losses import LNCC, BSplineBending
from torch import device

from TPTBox import Image_Reference
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.registration.deepali.deepali_model import LOSS, General_Registration

cuda = device("cuda")


class Deformable_Registration(General_Registration):
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
        fixed_seg: Image_Reference | None = None,
        moving_seg: Image_Reference | None = None,
        reference_image: Image_Reference | None = None,
        source_pset=None,
        target_pset=None,
        source_landmarks=None,
        target_landmarks=None,
        # source_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration source
        # target_seg: Optional[Union[Image, PathStr]] = None,  # Masking the registration target
        device: Union[torch.device, str, int] | None = None,
        gpu=0,
        ddevice: DEVICES = "cuda",
        # foreground_mask
        fixed_mask: Image_Reference | None = None,
        moving_mask: Image_Reference | None = None,
        # normalize
        normalize_strategy: (
            Literal["auto", "CT", "MRI"] | None
        ) = "auto",  # Override on_normalize for finer normalization schema or normalize before and set to None. auto: [min,max] -> [0,1]; None: Do noting
        # Pyramid
        pyramid_levels: int | None = 3,  # 1/None = no pyramid; int: number of stacks, tuple from to (0 is finest)
        finest_level: int = 0,
        coarsest_level: int | None = None,
        pyramid_finest_spacing: Sequence[int] | torch.Tensor | None = None,
        pyramid_min_size=16,
        dims=("x", "y", "z"),
        align=False,
        transform_name: str = "SVFFD",  # Names that are defined in deepali.spatial.LINEAR_TRANSFORMS and deepali.spatialNONRIGID_TRANSFORMS. Override on_make_transform for finer controle
        transform_args: dict | None = None,
        transform_init: PathStr | None = None,  # reload initial flowfield from file
        optim_name="Adam",  # Optimizer name defined in torch.optim. or override on_optimizer finer controle
        lr: float | Sequence[float] = 0.001,  # Learning rate
        lr_end_factor: float | None = None,  # if set, will use a LinearLR scheduler to reduce the learning rate to this factor * lr
        optim_args=None,  # args of Optimizer with out lr
        smooth_grad=0.0,
        verbose=0,
        max_steps: int | Sequence[int] = 1000,  # Early stopping.  override on_converged finer controle
        max_history: int | None = 100,
        min_value=0.0,  # Early stopping.  override on_converged finer controle
        min_delta: float | Sequence[float] = -0.0001,  # Early stopping.  override on_converged finer controle
        loss_terms: list[LOSS | str] | dict[str, LOSS] | dict[str, str] | dict[str, tuple[str, dict]] | None = None,
        weights: list[float] | dict[str, float | list[float]] | None = None,
        auto_run=True,
        stride=8,
    ):
        if transform_args is None:
            transform_args = {"stride": [stride, stride, stride], "transpose": False}
        if loss_terms is None:
            loss_terms = {
                "be": BSplineBending(stride=1),
                "lncc": LNCC(),
            }
        if weights is None:
            weights = {"be": 0.001, "lncc": 1}
        super().__init__(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_seg=fixed_seg,
            moving_seg=moving_seg,
            reference_image=reference_image,
            source_pset=source_pset,
            target_pset=target_pset,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            device=device,
            gpu=gpu,
            ddevice=ddevice,
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
            lr_end_factor=lr_end_factor,
            optim_args=optim_args,
            smooth_grad=smooth_grad,
            verbose=verbose,
            max_steps=max_steps,
            max_history=max_history,
            min_value=min_value,
            min_delta=min_delta,
            loss_terms=loss_terms,
            weights=weights,
            auto_run=auto_run,
        )


def test1(run=False):
    """
    Main function to test the deformable registration process.
    Loads reference and moving images, performs registration, and saves the output.
    """
    from TPTBox import NII

    fixed = NII.load(
        "/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-stitched_acq-ax_part-water_vibe.nii.gz",
        False,
    )
    moving = NII.load(
        "/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz",
        False,
    )
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
    mov = NII.load(
        "/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz",
        False,
    )
    poi = mov.make_empty_POI()
    poi[123, 44] = (68 - 1, 61 - 1, 17 - 1)
    poi[123, 45] = (77 - 1, 55 - 1, 20 - 1)
    mov2 = NII.load(
        "/media/data/robert/code/TPTBox/tmp/sub-100000_sequ-me1_acq-ax_part-water_mevibe.nii.gz",
        False,
    )
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
