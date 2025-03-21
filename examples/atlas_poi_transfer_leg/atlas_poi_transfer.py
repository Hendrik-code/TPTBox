from __future__ import annotations

import gzip
import pickle
from copy import deepcopy
from enum import Enum
from math import ceil, floor
from pathlib import Path
from time import time

import numpy as np

from TPTBox import POI_Global, calc_centroids, to_nii
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun.poi_abstract import POI_Descriptor
from TPTBox.core.vert_constants import Full_Body_Instance, Lower_Body
from TPTBox.logger.log_file import No_Logger
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration
from TPTBox.registration.ridged_points import Point_Registration

ABBREVIATION_TO_ENUM = {
    # Patella
    "PPP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_PROXIMAL_POLE),
    "PDP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_DISTAL_POLE),
    "PMP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_MEDIAL_POLE),
    "PLP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_LATERAL_POLE),
    "PRPP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_PROXIMAL_POLE),
    "PRDP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_DISTAL_POLE),
    "PRHP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_HIGH_POINT),
    # Femur
    "TRMP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEAR_RIDGE_MEDIAL_POINT),
    "TRLP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEAR_RIDGE_LATERAL_POINT),
    "TGCP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEA_GROOVE_CENTRAL_POINT),
    "FHC": (Full_Body_Instance.femur_right, Lower_Body.HIP_CENTER),
    "FNC": (Full_Body_Instance.femur_right, Lower_Body.NECK_CENTER),
    "TGT": (Full_Body_Instance.femur_right, Lower_Body.TIP_OF_GREATER_TROCHANTER),
    "FLCP": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR),
    "FLCPC": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR_CRANIAL),
    "FMCP": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR),
    "FMCPC": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR_CRANIAL),
    "FLCD": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_DISTAL),
    "FMCD": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_DISTAL),
    "FNP": (Full_Body_Instance.femur_right, Lower_Body.NOTCH_POINT),
    "FAAP": (Full_Body_Instance.femur_right, Lower_Body.ANATOMICAL_AXIS_PROXIMAL),
    "FADP": (Full_Body_Instance.femur_right, Lower_Body.ANATOMICAL_AXIS_DISTAL),
    # Tibia
    "TKC": (Full_Body_Instance.tibia_right, Lower_Body.KNEE_CENTER),
    "TMIT": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_INTERCONDYLAR_TUBERCLE),
    "TLIT": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_INTERCONDYLAR_TUBERCLE),
    "TMCP": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR),
    "TLCP": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR),
    "TMCA": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_ANTERIOR),
    "TLCA": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_ANTERIOR),
    "TMCM": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_MEDIAL),
    "TLCL": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_LATERAL),
    "TAC": (Full_Body_Instance.tibia_right, Lower_Body.ANKLE_CENTER),
    "TMM": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_MALLEOLUS),
    "TAAP": (Full_Body_Instance.tibia_right, Lower_Body.ANATOMICAL_AXIS_PROXIMAL),
    "TADP": (Full_Body_Instance.tibia_right, Lower_Body.ANATOMICAL_AXIS_DISTAL),
    "TGPP": (Full_Body_Instance.tibia_right, Lower_Body.TGPP),
    "TTP": (Full_Body_Instance.tibia_right, Lower_Body.TTP),
    # Fibula
    "FLM": (Full_Body_Instance.fibula_right, Lower_Body.LATERAL_MALLEOLUS),
}


def parse_coordinates(file_path: str | Path) -> dict[str, tuple[float, float, float]]:
    coordinates = {}

    with open(file_path) as file:
        for line in file:
            parts = str(line).replace("(", "").replace(")", "").replace(",", "").split(" ")
            if len(parts) == 4:  # Ensure correct format
                key = parts[0]
                values = tuple(map(float, parts[1:]))
                coordinates[key] = values
            # break
    return coordinates


def parse_coordinates_to_poi(file_path: str | Path, left: bool):
    coords_dict = parse_coordinates(file_path)
    global_points = POI_Descriptor()
    for k2, v in coords_dict.items():
        k, e = ABBREVIATION_TO_ENUM[k2]
        k = k.value if isinstance(k, Enum) else k % 100
        if left:
            k += 100
        global_points[k:e] = (v[0], v[1], v[2])

    return POI_Global(global_points, itk_coords=True, level_one_info=Full_Body_Instance, level_two_info=Lower_Body)


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
# reduce be if no overfitting possible
# reduce lr if optimization platos
# increase min_delta when it stops to early

PATELLA = 14
LEGS = [13, PATELLA, 15, 16]


def split_left_right_leg(nii: NII, c=2, min_volume=50):
    cc_patella = nii.extract_label(PATELLA).get_connected_components(1)
    nii = nii.extract_label(LEGS, keep_label=True)
    m = cc_patella.max()
    if m == 0:
        print("No Leg")
        return nii * 0
    elif m == 1:
        print("Only One Leg")
        return nii.clamp(0, 1)
    a = [ceil(c * z) for z in nii.spacing]
    nii_small: NII = nii[:: a[0], :: a[1], :: a[2]]
    legs = nii_small.extract_label(LEGS)
    while True:
        legs_cc = legs.filter_connected_components(1, min_volume=min_volume, keep_label=False)
        m = legs_cc.max()
        if m == 1:
            raise ValueError("segmentation_to_small")
        elif m == 1:
            if c == 0:
                raise ValueError("Bones are touching, can not split them into into two...")
            print("reduce c")
            return split_left_right_leg(nii, floor(c / 2), min_volume=min_volume)
        elif m == 2:
            break
        legs = legs.dilate_msk(1)
    nii_left_right = nii.clamp(0, 1) * legs_cc.resample_from_to(nii).dilate_msk(c)

    axis_r = nii_left_right.get_axis("R")
    label_positions = {label: np.mean(np.where(nii_left_right == label)[axis_r]) for label in [1, 2]}
    more_right_label = label_positions[1] > label_positions[2]
    if nii_left_right.orientation[axis_r] == "R" and more_right_label:
        nii_left_right.map_labels_({1: 2, 2: 1})
    if nii_left_right.orientation[axis_r] == "L" and not more_right_label:
        nii_left_right.map_labels_({1: 2, 2: 1})
    return nii_left_right


logger = No_Logger()


def prep_Atlas(seg: NII, out_atlas: Path, atlas_centroids: Path, atlas_left=True):
    if not out_atlas.exists():
        logger.on_text("split_left_right_leg")
        split_nii = split_left_right_leg(seg)
        atlas = split_nii.extract_label(1 if atlas_left else 2) * seg
        atlas.save(out_atlas)
    if atlas_centroids is not None and not atlas_centroids.exists():
        poi_atlas = calc_centroids(out_atlas, second_stage=40)
        poi_atlas.save(atlas_centroids)


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
        self.left = Register_Point_Atlas_One_Leg(
            l,
            atlas,
            poi_atlas,
            same_side=atlas_left,
            verbose=verbose,
            setting=setting_1,
            gpu=gpu,
            ddevice=ddevice,
        )
        logger.on_text("Register_Point_Atlas_One_Leg right")
        self.right = Register_Point_Atlas_One_Leg(
            r,
            atlas,
            poi_atlas,
            same_side=not atlas_left,
            verbose=verbose,
            setting=setting_1,
            gpu=gpu,
            ddevice=ddevice,
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
        self.left = Register_Point_Atlas_One_Leg.load_(t0)
        self.right = Register_Point_Atlas_One_Leg.load_(t1)
        return self


class Register_Point_Atlas_One_Leg:
    def __init__(
        self,
        target: NII,
        atlas: NII,
        poi_atlas: POI | None,
        same_side: bool,
        verbose=99,
        setting=default_setting,
        setting_patella=default_setting,
        gpu=0,
        ddevice: DEVICES = "cuda",
    ):
        # Assumes that you have removed the other leg.
        assert target.seg
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
        for i in [200, 100000]:
            t_crop = (target).compute_crop(0, i)  # if the angel is to different we need a larger crop...
            target_ = target.apply_crop(t_crop)
            # Point Registration
            poi_target = calc_centroids(target_, second_stage=40)
            if poi_atlas is None:
                poi_atlas = calc_centroids(atlas, second_stage=40)  # This will be needlessly computed all the time
            if not poi_atlas.assert_affine(atlas, raise_error=False):
                poi_atlas = poi_atlas.resample_from_to(atlas)
            self.reg_point = Point_Registration(poi_target, poi_atlas)
            atlas_reg = self.reg_point.transform_nii(atlas)
            if not atlas_reg.is_segmentation_in_border():
                target = target_
                break
            print(f"Try point reg again; padding {i=} was to small")

        # Major Bones
        self.crop = (target + atlas_reg).compute_crop(0, 2)
        target = target.apply_crop(self.crop)
        atlas_reg = atlas_reg.apply_crop(self.crop)
        self.target_grid = target.to_gird()
        self.reg_deform = Deformable_Registration(target, atlas_reg, config=setting, verbose=verbose, gpu=gpu, ddevice=ddevice)
        atlas_reg = self.reg_deform.transform_nii(atlas_reg)
        # Patella
        patella_atlas = atlas_reg.extract_label(PATELLA)
        patella_target = target.extract_label(PATELLA)
        self.crop_patella = (patella_target + patella_atlas).compute_crop(0, 2)
        patella_atlas.apply_crop_(self.crop_patella)
        patella_target.apply_crop_(self.crop_patella)
        self.reg_deform_p = Deformable_Registration(
            patella_target, patella_atlas, config=setting_patella, verbose=verbose, gpu=gpu, ddevice=ddevice
        )

    def get_dump(self):
        return (
            1,  # version
            (self.reg_point.get_dump()),
            (self.reg_deform.get_dump()),
            (self.reg_deform_p.get_dump()),
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
        self.reg_deform_p = Deformable_Registration.load_(t2)
        self.same_side, self.atlas_org, self.target_grid_org, self.target_grid, self.crop, self.crop_patella = x
        return self

    def forward_nii(self, nii_atlas: NII):
        nii_atlas = self.reg_point.transform_nii(nii_atlas)
        nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        patella_atlas = nii_reg.extract_label(PATELLA)
        nii_reg[patella_atlas == 1] = 0
        patella_atlas.apply_crop_(self.crop_patella)
        patella_atlas_reg = self.reg_deform_p.transform_nii(patella_atlas)
        patella_atlas_reg.resample_from_to_(nii_reg)
        nii_reg[patella_atlas_reg != 0] = PATELLA
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
        poi_atlas = poi_atlas.apply_crop(self.crop)
        poi_reg = self.reg_deform.transform_poi(poi_atlas)
        # Patella
        poi_patella = poi_reg.apply_crop(self.crop_patella).extract_region(
            Full_Body_Instance.patella_left.value, Full_Body_Instance.patella_right.value
        )
        patella_poi_reg = self.reg_deform_p.transform_poi(poi_patella)
        for k1, k2, v in patella_poi_reg.resample_from_to(poi_reg).items():
            poi_reg[k1, k2] = v
        # poi_reg.save(root / "test" / "subreg_reg.json")
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
    root = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/0001/")
    atlas = root / "test" / "left.nii.gz"
    root2 = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/0000/")
    target = to_nii(root2 / "bone.nii.gz", True)
    reg = Register_Point_Atlas(
        target,
        to_nii(atlas, True),
        atlas_left=True,
        split_leg_path=root2 / "test" / "split_leg.nii.gz",
        atlas_centroids=root / "test" / "atlast_centroids.json",
    )
    poi_out = reg.make_poi_from_txt("010__left.txt", root2 / "test" / "out_points_both.json")
    nii = poi_out.resample_from_to(target).make_point_cloud_nii(s=5)[1]
    nii2 = target.clamp(0, 1) * 100
    nii[nii == 0] = nii2[nii == 0]
    nii.save(root2 / "test" / "atlas_reg_poi.nii.gz")
    poi = POI.load(root2 / "test" / "out_points_both.json")
    poi.save(root2 / "test" / "out_points_both2.json")
    reg.save("Register_Point_Atlas.pkl")
    # CUDA_VISIBLE_DEVICES=2 /opt/anaconda3/envs/py3.11/bin/python /DATA/NAS/tools/TPTBox/tmp_poi_w.py
    exit()
    st = time()

    lr = to_nii(root / "bone_lr.nii.gz", True)
    bone = to_nii("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/0001/bone.nii.gz", True)
    x = split_left_right_leg(bone)
    assert x is not None
    x.save(root / "test" / "left_right_split.nii.gz")
    print(time() - st)
    exit()
    left = root / "test" / "left.nii.gz"
    right = root / "test" / "right.nii.gz"

    if not (right).exists():
        lr[bone == 11] = 0
        lr[bone == 5] = 0

        (lr.extract_label(23) * bone).save(left)
        assert lr.get_axis("R") == 0
        (lr.extract_label(22) * bone).save(right)

    st = time()
    left_nii_atlas = to_nii(left, True)  # .rescale((2, 2, 2))
    right_nii_target = to_nii(right, True)  # .rescale((2, 2, 2))
    default_setting["optim"]["args"]["lr"] *= min(right_nii_target.zoom) / 2.0
    default_setting["loss"]["weights"]["be"] *= min(right_nii_target.zoom) / 2.0
    default_setting["optim"]["loop"]["min_delta"] /= min(right_nii_target.zoom) / 2.0
    reg_obj = Register_Point_Atlas_One_Leg(right_nii_target, left_nii_atlas, False)
    poi_out = reg_obj.forward_txt(file_path="010__left.txt")
    nii_out = reg_obj.forward_nii(left_nii_atlas)
    poi_out.save(root / "test" / "atlas_reg_poi.json")
    nii_out.save(root / "test" / "atlas_reg.nii.gz")
    nii = poi_out.make_point_cloud_nii(s=5)[1]
    nii2 = right_nii_target.clamp(0, 1) * 100
    nii[nii == 0] = nii2[nii == 0]
    nii.save(root / "test" / "atlas_reg_poi.nii.gz")
    p = poi_out.to_global(itk_coords=True)
    coords_dict = parse_coordinates("010__left.txt")
    for e, (k2, v) in enumerate(coords_dict.items(), 1):
        k = PATELLA if k2[0] == "P" else 60
        print(k2, p[k, e])

    print(time() - st)
    # reg_obj.save(root / "test" / "Register_Point_Atlas.pkl")
