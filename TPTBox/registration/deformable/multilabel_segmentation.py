from __future__ import annotations

import pickle
from pathlib import Path

from TPTBox import NII, POI
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.poi import calc_centroids
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration
from TPTBox.registration.ridged_intensity.affine_deepali import Tether_Seg
from TPTBox.registration.ridged_points.point_registration import Point_Registration


class Register_Multi_Seg:
    """
    Class to perform multi-stage registration between two multi-label segmentations, including optional
    landmark (point-of-interest, POI) alignment and deformable registration. If not provided they will be computed on the fly.

    This is especially useful for aligning anatomical segmentations from MRI or CT between a target and an atlas,
    optionally considering left/right flipping if segmentations are from different body sides.

    Attributes:
        same_side (bool): Whether the target and atlas represent the same anatomical side (e.g., both right sides).
        reg_point (Point_Registration): The rigid point-based registration component.
        reg_deform (Deformable_Registration): The deformable registration component.
        crop (tuple): The crop applied to both target and atlas after registration.
        target_grid_org (NII): Original spatial grid of the target.
        atlas_org (NII): Original spatial grid of the atlas.
        target_grid (NII): Cropped spatial grid used for deformable registration.
    """

    def __init__(  # noqa: C901
        self,
        target: NII,
        atlas: NII,
        poi_cms: POI | None = None,
        same_side: bool = True,
        verbose=99,
        gpu=0,
        ddevice: DEVICES = "cuda",
        loss_terms=None,  # type: ignore
        weights=None,
        lr=0.01,
        lr_end_factor=None,
        max_steps=1500,
        min_delta: float | list[float] = 1e-06,
        pyramid_levels=4,
        coarsest_level=3,
        finest_level=0,
        crop: bool = True,
        cms_ids: list | None = None,
        poi_target_cms: POI | None = None,
        max_history=100,
        change_after_point_reg=lambda x, y: (x, y),
        **args,
    ):
        """
        Initialize a multi-stage registration pipeline from an atlas to a target image.

        Args:
            target (NII): Target image with segmentation (e.g., from a subject).
            atlas (NII): Atlas image with segmentation (e.g., a reference or template).
            poi_cms (POI | None): POI centroids of the atlas, used for initial point registration.
            same_side (bool): Whether atlas and target represent the same body side.
            verbose (int): Verbosity level for logging.
            gpu (int): GPU device ID (only relevant if using GPU).
            ddevice (DEVICES): Device type ('cuda' or 'cpu').
            loss_terms (dict): Dictionary of loss terms for deformable registration.
            weights (dict): Weights for the loss terms.
            lr (float): Learning rate for deformable registration optimizer.
            max_steps (int): Maximum optimization steps.
            min_delta (float): Minimum delta for convergence.
            pyramid_levels (int): Number of resolution levels in multi-scale deformable registration.
            coarsest_level (int): Coarsest level index.
            finest_level (int): Finest level index.
            cms_ids (list | None): List of segmentation labels used to extract POI centroids.
            poi_target_cms (POI | None): Optional precomputed centroids for the target image.
            **args: Additional keyword arguments passed to Deformable_Registration.
        Raises:
            ValueError: If an invalid axis is detected during flipping.
        """
        if weights is None:
            weights = {"be": 0.0001, "seg": 1, "Dice": 0.01, "Tether": 0.001}
        if loss_terms is None:
            loss_terms = {
                "be": ("BSplineBending", {"stride": 1}),
                "seg": "MSE",
                "Dice": "Dice",
                "Tether": Tether_Seg(delta=5),
            }

        assert target.seg, target.seg
        assert atlas.seg
        target = target.copy()
        atlas = atlas.copy()
        self.same_side = same_side
        self.target_grid_org = target.to_gird()
        self.atlas_org = atlas.to_gird()
        if not same_side:
            axis = target.get_axis("R")
            if axis == 0:
                target = target.set_array(target.get_array()[::-1]).copy()
            elif axis == 1:
                target = target.set_array(target.get_array()[:, ::-1]).copy()
            elif axis == 2:
                target = target.set_array(target.get_array()[:, :, ::-1]).copy()
            else:
                raise ValueError(axis)
            if poi_target_cms is not None:
                axis = poi_target_cms.get_axis("R")
                for k1, k2, (x, y, z) in poi_target_cms.copy().items():
                    if axis == 0:
                        poi_target_cms[k1, k2] = (poi_target_cms.shape[0] - 1 - x, y, z)
                    elif axis == 1:
                        poi_target_cms[k1, k2] = (x, poi_target_cms.shape[1] - 1 - y, z)
                    elif axis == 2:
                        poi_target_cms[k1, k2] = (x, y, poi_target_cms.shape[2] - 1 - z)
        if crop:
            print("crop")
            crop = 50
            t_crop = (target).compute_crop(0, crop)
            target = target.apply_crop(t_crop)
            if atlas.is_segmentation_in_border():
                atlas = atlas.apply_pad(((1, 1), (1, 1), (1, 1)))
            for i in range(10):  # 1000,
                if i != 0:
                    target = target.apply_pad(((25, 25), (25, 25), (25, 25)))
                    crop += 50
                t_crop = (target).compute_crop(0, crop)  # if the angel is to different we need a larger crop...
                target_ = target.apply_crop(t_crop)
                # Point Registration
                print("calc_centroids")
                if poi_target_cms is None:
                    x = target_.extract_label(cms_ids, keep_label=True) if cms_ids else target_
                    poi_target = calc_centroids(x, second_stage=40, bar=True)  # TODO REMOVE
                else:
                    poi_target = poi_target_cms.resample_from_to(target_)
                if poi_cms is None:
                    x = atlas.extract_label(cms_ids, keep_label=True) if cms_ids else atlas
                    poi_cms = calc_centroids(x, second_stage=40, bar=True)  # This will be needlessly computed all the time
                if not poi_cms.assert_affine(atlas, raise_error=False):
                    poi_cms = poi_cms.resample_from_to(atlas)
                self.reg_point = Point_Registration(poi_target, poi_cms)
                atlas_reg = self.reg_point.transform_nii(atlas)
                if atlas_reg.is_segmentation_in_border():
                    print("atlas_reg does touch the border")
                else:
                    target = target_
                    break
        else:
            target_ = target
            if poi_target_cms is None:
                x = target_.extract_label(cms_ids, keep_label=True) if cms_ids else target_
                poi_target = calc_centroids(x, second_stage=40, bar=True)  # TODO REMOVE
            else:
                poi_target = poi_target_cms.resample_from_to(target_)
            self.reg_point = Point_Registration(poi_target, poi_cms)
            atlas_reg = self.reg_point.transform_nii(atlas)

        target = target_
        if crop:
            self.crop = (target + atlas_reg).compute_crop(0, 5)
            target = target.apply_crop(self.crop)
            atlas_reg = atlas_reg.apply_crop(self.crop)
        else:
            self.crop = None

        self.target_grid = target.to_gird()
        target, atlas_reg = change_after_point_reg(target, atlas_reg)
        self.reg_deform = Deformable_Registration(
            target,
            atlas_reg,
            target.copy(),
            atlas_reg.copy(),
            loss_terms=loss_terms,
            weights=weights,
            lr=lr,
            lr_end_factor=lr_end_factor,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            verbose=verbose,
            gpu=gpu,
            ddevice=ddevice,
            max_history=max_history,
            **args,
        )

    def get_dump(self):
        """
        Collect serializable state of the registration object.

        Returns:
            tuple: Serialized components including version, rigid registration state, deformable state,
                   and spatial metadata.
        """
        return (
            1,  # version
            (self.reg_point.get_dump()),
            (self.reg_deform.get_dump()),
            (
                self.same_side,
                self.atlas_org,
                self.target_grid_org,
                self.target_grid,
                self.crop,
            ),
        )

    def save(self, path: str | Path):
        """
        Save the registration state to a file.

        Args:
            path (str | Path): Path to save the pickle file.
        """
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path):
        """
        Load a previously saved registration state from a file.

        Args:
            path (str | Path): Path to the pickle file.

        Returns:
            Register_Multi_Seg: Reconstructed instance of the class.
        """
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w))

    @classmethod
    def load_(cls, w):
        """
        Load a registration object from a deserialized state (as returned by `get_dump()`).

        Args:
            w (tuple): Serialized state.

        Returns:
            Register_Multi_Seg: Reconstructed instance of the class.
        """
        (version, t0, t1, x) = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.reg_point = Point_Registration.load_(t0)
        self.reg_deform = Deformable_Registration.load_(t1)
        (
            self.same_side,
            self.atlas_org,
            self.target_grid_org,
            self.target_grid,
            self.crop,
        ) = x
        return self

    def transform_nii(self, nii_atlas: NII):
        """
        Apply both rigid and deformable registration to a new NII object.

        Args:
            nii_atlas (NII): New atlas image to be transformed.

        Returns:
            NII: Transformed image aligned with the original target image.
        """
        nii_atlas = self.reg_point.transform_nii(nii_atlas)
        nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        if nii_reg.seg:
            nii_reg.set_dtype_("smallest_uint")
        out = nii_reg.resample_from_to(self.target_grid_org)
        if self.same_side:
            return out
        axis = out.get_axis("R")
        if axis == 0:
            target = out.set_array(out.get_array()[::-1]).copy()
        elif axis == 1:
            target = out.set_array(out.get_array()[:, ::-1]).copy()
        elif axis == 2:
            target = out.set_array(out.get_array()[:, :, ::-1]).copy()
        else:
            raise ValueError(axis)

        return target

    def transform_poi(self, poi_atlas: POI_Global | POI):
        """
        Apply both rigid and deformable registration to a POI (landmark) object.

        Args:
            poi_atlas (POI_Global | POI): Atlas landmarks to be transformed.

        Returns:
            POI: Transformed POIs aligned to the target space.
        """
        poi_atlas = poi_atlas.resample_from_to(self.atlas_org)

        # Point Reg
        poi_atlas = self.reg_point.transform_poi(poi_atlas)
        # Deformable
        poi_atlas = poi_atlas.apply_crop(self.crop)

        poi_reg = self.reg_deform.transform_poi(poi_atlas)
        poi_reg = poi_reg.resample_from_to(self.target_grid_org)
        if self.same_side:
            return poi_reg
        for k1, k2, v in poi_reg.copy().items():
            k = k1  # % 100
            poi_reg[k, k2] = v
        poi_reg_flip = poi_reg.make_empty_POI()
        for k1, k2, (x, y, z) in poi_reg.copy().items():
            axis = poi_reg.get_axis("R")
            if axis == 0:
                poi_reg_flip[k1, k2] = (poi_reg.shape[0] - 1 - x, y, z)
            elif axis == 1:
                poi_reg_flip[k1, k2] = (x, poi_reg.shape[1] - 1 - y, z)
            elif axis == 2:
                poi_reg_flip[k1, k2] = (x, y, poi_reg.shape[2] - 1 - z)
            else:
                raise ValueError(axis)
        return poi_reg_flip
