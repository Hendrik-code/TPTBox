from __future__ import annotations

import gzip
import pickle
from copy import deepcopy

# Step 1
from pathlib import Path

from examples.registration.atlas_poi_transfer_leg.atlas_poi_transfer_leg_ct import parse_coordinates_to_poi, split_left_right_leg
from TPTBox import POI, POI_Global, calc_centroids, to_nii
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.core.vert_constants import Full_Body_Instance, Lower_Body
from TPTBox.logger.log_file import No_Logger
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration

# from TPTBox.registration.deformable.deformable_reg_old import Deformable_Registration as Deformable_Registration_old
from TPTBox.registration.ridged_points import Point_Registration

default_setting = {
    "loss": {"config": {"be": {"stride": 1, "name": "BSplineBending"}, "seg": {"name": "MSE"}}, "weights": {"be": 0.0001, "seg": 1}},
    "model": {"name": "SVFFD", "args": {"stride": [4, 4, 4], "transpose": False}, "init": None},
    "optim": {"name": "Adam", "args": {"lr": 0.001}, "loop": {"max_steps": 1500, "min_delta": 0.00001}},
    "pyramid": {
        "levels": 4,  # 4 for 0.8 res instead of 3
        "coarsest_level": 3,  # 3 for 0.8 res instead of 2
        "finest_level": 0,
        "finest_spacing": None,  # Auto set by the nifty
        "min_size": 16,
        "pyramid_dims": ["x", "y", "z"],
    },
}
setting_1 = default_setting

logger = No_Logger()


class Register_Point_Atlas:
    def __init__(
        self,
        target: NII,
        atlas: NII,
        atlas_left: bool = True,
        verbose=99,
        split_leg_path: Path | str | None = None,
        atlas_centroids: Path | str | None = None,
        gpu=0,
        ddevice: DEVICES = "cuda",
    ):
        # TODO Test Save and Load
        # atlas left assumed already filtered
        if split_leg_path is not None and Path(split_leg_path).exists():
            split_nii = to_nii(split_leg_path, True)
        else:
            logger.on_text("split_left_right_leg")
            split_nii = split_left_right_leg(target)
            if split_leg_path is not None:
                split_nii.save(split_leg_path)
        if atlas_centroids is not None and Path(atlas_centroids).exists():
            poi_atlas = POI.load(atlas_centroids)
        else:
            logger.on_text("calc_centroids atlas")
            poi_atlas = calc_centroids(atlas, second_stage=40)
            if atlas_centroids is not None:
                poi_atlas.save(atlas_centroids)
        if not split_nii.assert_affine(target, raise_error=False):
            split_nii.resample_from_to_(target).dilate_msk_(1)
        setting_1 = deepcopy(default_setting)
        setting_1["optim"]["args"]["lr"] *= min(target.zoom) / 2.0
        setting_1["loss"]["weights"]["be"] *= min(target.zoom) / 2.0
        # setting_1["optim"]["loop"]["min_delta"] /= min(target.zoom) / 2.0
        logger.on_text("Register_Point_Atlas_One_Leg left")
        l = split_nii.extract_label(1) * target
        r = split_nii.extract_label(2) * target
        self.left = Register_Atlas(l, atlas, poi_atlas, same_side=atlas_left, verbose=verbose, setting=setting_1, gpu=gpu, ddevice=ddevice)
        logger.on_text("Register_Point_Atlas_One_Leg right")
        self.right = Register_Atlas(
            r, atlas, poi_atlas, same_side=not atlas_left, verbose=verbose, setting=setting_1, gpu=gpu, ddevice=ddevice
        )

    def make_poi_from_txt(self, text: Path | str, out: Path | str):
        logger.on_text("make_poi")
        poi_l = self.left.forward_txt(text, left=True)
        poi_r = self.right.forward_txt(text, left=False)
        poi_l.join_left_(poi_r)
        poi_l.save(out)
        return poi_l

    def make_poi_from_poi(self, poi: POI, out: Path | str):
        logger.on_text("make_poi")
        poi_l = self.left.forward_poi(poi, left=True)
        poi_r = self.right.forward_poi(poi, left=False)
        poi_l.join_left_(poi_r)
        poi_l.save(out)
        return poi_l

    def get_dump(self):
        return (
            1,  # version
            (self.left.get_dump()),
            (self.right.get_dump()),
        )

    def save(self, path: str | Path, compress: bool = True):
        name = Path(path).name
        data = self.get_dump()
        if not compress:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            import gzip

            if not name.endswith(".pkl.gz"):
                path = str(path) + ".pkl.gz"
            with gzip.open(path, "wb") as w:
                pickle.dump(data, w)

    @classmethod
    def load(cls, path):
        try:
            with gzip.open(path) as w:
                return cls.load_(pickle.load(w))
        except Exception:
            pass
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w))

    @classmethod
    def load_(cls, w):
        (version, t0, t1) = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.left = Register_Atlas.load_(t0)
        self.right = Register_Atlas.load_(t1)
        return self


class Register_Atlas:
    def __init__(
        self,
        target_ct: NII,
        atlas_ct: NII,
        target: NII,
        atlas: NII,
        poi_cms: POI | None,
        same_side: bool,
        verbose=99,
        setting=default_setting,
        gpu=0,
        ddevice: DEVICES = "cuda",
    ):
        # Assumes that you have removed the other leg.
        assert target.seg
        assert atlas.seg
        target = target.copy()
        target_ct = target_ct.copy()
        atlas = atlas.copy()
        atlas_ct = atlas_ct.copy()
        self.same_side = same_side
        self.target_grid_org = target.to_gird()
        self.atlas_org = atlas.to_gird()
        if not same_side:
            axis = target.get_axis("R")
            if axis == 0:
                target = target.set_array(target.get_array()[::-1]).copy()
                target_ct = target.set_array(target_ct.get_array()[::-1]).copy()
            elif axis == 1:
                target = target.set_array(target.get_array()[:, ::-1]).copy()
                target_ct = target.set_array(target_ct.get_array()[:, ::-1]).copy()
            elif axis == 2:
                target = target.set_array(target.get_array()[:, :, ::-1]).copy()
                target_ct = target.set_array(target_ct.get_array()[:, :, ::-1]).copy()
            else:
                raise ValueError(axis)
        for i in [200, 100000]:
            t_crop = (target).compute_crop(0, i)  # if the angel is to different we need a larger crop...
            target_ = target.apply_crop(t_crop)
            # Point Registration
            poi_target = calc_centroids(target_, second_stage=40)
            if poi_cms is None:
                poi_cms = calc_centroids(atlas, second_stage=40)  # This will be needlessly computed all the time
            if not poi_cms.assert_affine(atlas, raise_error=False):
                poi_cms = poi_cms.resample_from_to(atlas)
            self.reg_point = Point_Registration(poi_target, poi_cms)
            atlas_reg = self.reg_point.transform_nii(atlas)
            atlas_reg_ct = self.reg_point.transform_nii(atlas_ct)
            if not atlas_reg.is_segmentation_in_border():
                target = target_
                break
            print(f"Try point reg again; padding {i=} was to small")

        # Major Bones
        # self.crop = (target + atlas_reg).compute_crop(0, 2)
        mask = target.copy() * 0
        mask.seg = True
        # mask.set_array_(mask.get_array())
        #
        mask_target = mask.copy()
        mask_target[target.compute_crop(0, 10)] = 1
        # mask_moving = mask.copy()
        # mask_moving[atlas_reg.compute_crop(0, 10)] = 1
        # target_ct = target_ct.apply_crop(self.crop)
        # atlas_reg = atlas_reg.apply_crop(self.crop)
        # atlas_reg_ct = atlas_reg_ct.apply_crop(self.crop)
        self.target_grid = target.to_gird()
        self.reg_deform = Deformable_Registration(
            target_ct,
            atlas_reg_ct,
            loss_terms={"be": ("BSplineBending", {"stride": 1}), "seg": "MSE"},  # type: ignore
            weights={"be": setting["loss"]["weights"]["be"] * 0.1, "seg": setting["loss"]["weights"]["seg"]},
            lr=setting["optim"]["args"]["lr"] * 100,
            max_steps=setting["optim"]["loop"]["max_steps"],
            min_delta=setting["optim"]["loop"]["min_delta"],
            pyramid_levels=5,
            coarsest_level=4,
            finest_level=0,
            gpu=gpu,
            ddevice=ddevice,
            verbose=verbose,
            normalize_strategy="CT",
            fixed_mask=mask_target,
            # moving_mask=mask_moving,
        )
        self.crop_patella = None
        self.crop = 0

    def get_dump(self):
        return (
            1,  # version
            (self.reg_point.get_dump()),
            (self.reg_deform.get_dump()),
            # (self.reg_deform_p.get_dump()),
            (self.same_side, self.atlas_org, self.target_grid_org, self.target_grid, self.crop, self.crop_patella),
        )

    def save(self, path: str | Path):
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w))

    @classmethod
    def load_(cls, w):
        (version, t0, t1, t2, x) = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.reg_point = Point_Registration.load_(t0)
        self.reg_deform = Deformable_Registration.load_(t1)
        # self.reg_deform_p = Deformable_Registration.load_(t2)
        self.same_side, self.atlas_org, self.target_grid_org, self.target_grid, self.crop, self.crop_patella = x
        return self

    def forward_nii(self, nii_atlas: NII):
        nii_atlas = self.reg_point.transform_nii(nii_atlas)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        nii_reg = nii_reg.resample_from_to(self.target_grid_org)
        if self.same_side:
            return nii_reg
        return nii_reg.set_array(nii_reg.get_array()[::-1])

    def forward_txt(self, file_path: str | Path, left: bool):
        poi_glob = parse_coordinates_to_poi(file_path, left)
        return self.forward_poi(poi_glob, left)

    def forward_poi(self, poi_atlas: POI_Global | POI, left):
        poi_atlas = poi_atlas.resample_from_to(self.atlas_org)
        # Point Reg
        poi_atlas = self.reg_point.transform_poi(poi_atlas)
        # Deformable
        poi_reg = self.reg_deform.transform_poi(poi_atlas)

        poi_reg = poi_reg.resample_from_to(self.target_grid_org)
        poi_reg.level_one_info = Full_Body_Instance
        poi_reg.level_two_info = Lower_Body
        if self.same_side:
            return poi_reg
        for k1, k2, v in poi_reg.copy().items():
            k = k1 % 100
            if left:
                k += 100
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
        poi_reg_flip.level_one_info = Full_Body_Instance
        poi_reg_flip.level_two_info = Lower_Body
        return poi_reg_flip


if __name__ == "__main__":
    root = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/atlas")
    atlas = to_nii(root / "atlas002_ct.nii.gz")
    reg = Register_Atlas(
        to_nii(root / "atlas001_ct.nii.gz"),
        to_nii(root / "atlas002_ct.nii.gz"),
        to_nii(root / "atlas001.nii.gz", True),
        to_nii(root / "atlas002.nii.gz", True),
        POI.load(root / "atlas002_cms_poi.json"),
        True,
        gpu=3,
    )
    reg.forward_nii(atlas).save(root.parent / "2to1.nii.gz")
    poi = reg.forward_poi(POI.load(root / "atlas002_cms_poi.json"), True)
    poi.save(root.parent / "2to1_poi.json")
    poi.make_point_cloud_nii(s=5)[1].save(root.parent / "2to1_poi.nii.gz")
