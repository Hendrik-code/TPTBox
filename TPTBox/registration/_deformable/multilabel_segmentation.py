from __future__ import annotations

import pickle
from pathlib import Path

from TPTBox import NII, POI
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.poi import calc_centroids
from TPTBox.core.poi_fun.poi_global import POI_Global
from TPTBox.registration._deformable.deformable_reg import Deformable_Registration
from TPTBox.registration._ridged_intensity.affine_deepali import Tether_Seg
from TPTBox.registration._ridged_points.point_registration import Point_Registration


class Template_Registration:
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
        target_seg: NII,
        atlas_seg: NII,
        target_img: NII | None = None,
        atlas_img: NII | None = None,
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
        change_after_point_reg=lambda x, y, z, w: (x, y, z, w),
        **args,
    ):
        """
        Initialize a multi-stage registration pipeline from an atlas to a target image.

        Args:
            target (NII): Target image segmentation (e.g., from a subject).
            atlas (NII): Atlas image segmentation (e.g., a reference or template).
            target_img (NII): Target image if None the segmentation is used as an image.
            atlas_img (NII): Atlas image if None the segmentation is used as an image.
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

        assert target_seg.seg, target_seg.seg
        assert atlas_seg.seg
        target_seg = target_seg.copy()
        atlas_seg = atlas_seg.copy()
        if target_img is not None:
            target_img = target_img.resample_from_to(target_seg)
        if atlas_img is not None:
            atlas_img = atlas_img.resample_from_to(atlas_seg)
        self.same_side = same_side
        self.target_grid_org = target_seg.to_gird()
        self.atlas_org = atlas_seg.to_gird()
        if not same_side:
            axis = target_seg.get_axis("R")
            if axis == 0:
                target_seg = target_seg.set_array(target_seg.get_array()[::-1]).copy()
                target_img = target_img.set_array(target_img.get_array()[::-1]).copy() if target_img is not None else None
            elif axis == 1:
                target_seg = target_seg.set_array(target_seg.get_array()[:, ::-1]).copy()
                target_img = target_img.set_array(target_img.get_array()[:, ::-1]).copy() if target_img is not None else None
            elif axis == 2:
                target_seg = target_seg.set_array(target_seg.get_array()[:, :, ::-1]).copy()
                target_img = target_img.set_array(target_img.get_array()[:, :, ::-1]).copy() if target_img is not None else None
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
        if poi_target_cms is None:
            x = target_seg.extract_label(cms_ids, keep_label=True) if cms_ids else target_seg
            poi_target = calc_centroids(x, second_stage=40, bar=True)  # TODO REMOVE
        else:
            poi_target = poi_target_cms.resample_from_to(target_seg)
        if poi_cms is None:
            x = atlas_seg.extract_label(cms_ids, keep_label=True) if cms_ids else atlas_seg
            poi_cms = calc_centroids(x, second_stage=40, bar=True)
        if not poi_cms.assert_affine(atlas_seg, raise_error=False):
            poi_cms = poi_cms.resample_from_to(atlas_seg)
        if crop:
            print("crop")

            crop_pad_size = 50
            _step = 50
            _max_iter = 10

            resize_mode = "crop"
            resize_param: tuple | None = None
            target_tmp = target_seg

            atlas_seg_ = atlas_seg.apply_pad(((1, 1), (1, 1), (1, 1))) if atlas_seg.is_segmentation_in_border() else atlas_seg

            for i in range(_max_iter):
                if resize_mode == "crop":
                    if i != 0:
                        crop_pad_size += _step

                    # --- try crop first ---
                    t_crop = target_seg.compute_crop(0, crop_pad_size)
                    cropped = target_seg.apply_crop(t_crop)

                    if any(c < o for c, o in zip(cropped.shape, target_seg.shape)):
                        resize_mode = "crop"
                        resize_param = t_crop
                        target_tmp = cropped
                    else:
                        # --- fallback to padding ---
                        crop_pad_size = 0
                        target_tmp = target_seg
                        resize_mode = "pad"
                else:
                    if i != 0:
                        crop_pad_size += _step // 2
                    t_pad = tuple((crop_pad_size, crop_pad_size) for _ in range(3))
                    resize_param = t_pad
                    target_tmp = target_seg.apply_pad(t_pad)

                # --- Point registration ---
                print(f"iter {i}: using {resize_mode} ({crop_pad_size})")

                poi_target = poi_target.resample_from_to(target_tmp)

                if poi_cms is None:
                    x = atlas_seg_.extract_label(cms_ids, keep_label=True) if cms_ids else atlas_seg_
                    poi_cms = calc_centroids(x, second_stage=40, bar=True)

                if not poi_cms.assert_affine(atlas_seg_, raise_error=False):
                    poi_cms = poi_cms.resample_from_to(atlas_seg_)

                self.reg_point = Point_Registration(poi_target, poi_cms)
                atlas_reg = self.reg_point.transform_nii(atlas_seg_)

                if not atlas_reg.is_segmentation_in_border():
                    print("registration ok")
                    break
                else:
                    print("atlas_reg touches border → expanding")

            # --- FINAL STEP: apply once to original target ---
            if resize_mode == "crop":
                target_seg = target_seg.apply_crop(resize_param)
                target_img = target_img.apply_crop(resize_param) if target_img is not None else None
            elif resize_mode == "pad":
                target_seg = target_seg.apply_pad(resize_param)
                target_img = target_img.apply_pad(resize_param) if target_img is not None else None

        self.reg_point = Point_Registration(poi_target.resample_from_to(target_seg), poi_cms.resample_from_to(atlas_seg))
        atlas_reg = self.reg_point.transform_nii(atlas_seg)
        atlas_img_reg = self.reg_point.transform_nii(atlas_img) if atlas_img is not None else None

        if crop:
            self.crop = (target_seg + atlas_reg).compute_crop(0, 5)
            target_seg = target_seg.apply_crop(self.crop)
            target_img = target_img.apply_crop(self.crop) if target_img is not None else None
            atlas_reg = atlas_reg.apply_crop(self.crop)
            atlas_img_reg = atlas_img_reg.apply_crop(self.crop) if atlas_img_reg is not None else None
        else:
            self.crop = None

        self.target_grid = target_seg.to_gird()
        target_seg, atlas_reg, target_img, atlas_img_reg = change_after_point_reg(target_seg, atlas_reg, target_img, atlas_img_reg)
        self.reg_deform = Deformable_Registration(
            target_seg if target_img is None else target_img,
            atlas_reg if atlas_img_reg is None else atlas_img_reg,
            target_seg.copy(),
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

    def transform_nii(self, nii_atlas: NII, allow_only_same_grid_as_moving=True):
        """
        Apply both rigid and deformable registration to a new NII object.

        Args:
            nii_atlas (NII): New atlas image to be transformed.

        Returns:
            NII: Transformed image aligned with the original target image.
        """

        nii_atlas = self.reg_point.transform_nii(nii_atlas, allow_only_same_grid_as_moving=allow_only_same_grid_as_moving)
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
