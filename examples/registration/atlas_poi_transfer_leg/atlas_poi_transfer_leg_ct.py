from __future__ import annotations

import gzip
import pickle
import sys
from copy import deepcopy
from enum import Enum
from math import ceil, floor

# Step 1
from pathlib import Path
from time import time

import numpy as np
from deepali.core import Axes, PathStr
from deepali.core import Grid as Deepali_Grid

from TPTBox import POI, POI_Global, calc_centroids, to_nii
from TPTBox.core.internal.deep_learning_utils import DEVICES
from TPTBox.core.nii_poi_abstract import Has_Grid
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun.poi_abstract import POI_Descriptor
from TPTBox.core.vert_constants import Full_Body_Instance, Lower_Body
from TPTBox.logger.log_file import No_Logger
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration
from TPTBox.registration.ridged_intensity.affine_deepali import Tether

# from TPTBox.registration.deformable.deformable_reg_old import Deformable_Registration as Deformable_Registration_old
from TPTBox.registration.ridged_points import Point_Registration

logger = No_Logger()
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


def generate_atlas_from_txt(
    file_text: str | Path,
    atlas_id: int,
    text_file_is_left_leg: bool,
    segmentation_path: str | Path,
    ct_path: str | Path,
    out_folder=Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/atlas"),
    addendum="",
):
    """
    This function generated a standartiesed atlas. Right legs will be mirrored and stored as a "left" leg.
    The segmentation must be a bone-segmentation with label produced like by Model number 10 from the TotalVibeSegmentor repo.
    See: https://github.com/robert-graf/TotalVibeSegmentator

    You need segmentation and a poi file or a text file in this format:
    ```bash
    TGT (139.4188232421875, -92.69547271728516, 1138.0546875)
    FHC (93.09368133544922, -108.62625885009766, 1136.891845703125)
    FNC (107.05316162109375, -107.46392059326172, 1121.36865234375)
    FAAP (110.07427978515625, -110.30272674560547, 1027.859130859375)
    FLCD (89.51376342773438, -108.07373046875, 732.1410522460938)
    FMCD (41.474029541015625, -103.25114440917969, 731.2152709960938)
    FLCP (92.19694519042969, -84.93902587890625, 751.868896484375)
    FMCP (42.786216735839844, -74.31689453125, 753.4283447265625)
    FNP (63.380027770996094, -116.54719543457031, 739.7014770507812)
    FADP (71.667724609375, -121.2674560546875, 822.39990234375)
    TGPP (60.06828308105469, -132.88507080078125, 758.1168823242188)
    TGCP (61.84690475463867, -131.874267578125, 751.1783447265625)
    FMCPC (46.220550537109375, -83.17189025878906, 767.088623046875)
    FLCPC (89.6451187133789, -92.42149353027344, 766.2191162109375)
    TRMP (46.96009826660156, -131.4429931640625, 748.450927734375)
    TRLP (79.27147674560547, -142.57913208007812, 754.849365234375)
    FLM (73.47111511230469, -97.508544921875, 384.16046142578125)
    TMM (19.230003356933594, -129.28802490234375, 393.071533203125)
    TAC (41.175079345703125, -114.86720275878906, 392.0289001464844)
    TADP (50.04762268066406, -116.96440887451172, 459.7703857421875)
    TLCL (99.41020202636719, -108.22544860839844, 727.647705078125)
    TMCM (34.0052490234375, -97.65898132324219, 727.727783203125)
    TKC (67.44658660888672, -99.75123596191406, 734.62744140625)
    TLCA (88.45868682861328, -119.93305969238281, 729.4689331054688)
    TLCP (91.51274871826172, -91.85955047607422, 726.2589721679688)
    TMCA (42.026023864746094, -118.33468627929688, 732.6619262695312)
    TMCP (48.68177032470703, -83.51608276367188, 726.7157592773438)
    TTP (70.8026123046875, -137.39186096191406, 697.2947998046875)
    TAAP (62.44767761230469, -117.48356628417969, 654.552490234375)
    TMIT (60.547447204589844, -99.73310852050781, 736.443359375)
    TLIT (73.91001892089844, -99.01293182373047, 735.8458251953125)
    PPP (60.099151611328125, -142.22222900390625, 776.0038452148438)
    PDP (62.95762634277344, -146.806884765625, 739.1683959960938)
    PMP (43.483184814453125, -149.5174560546875, 754.05224609375)
    PLP (80.17727661132812, -150.05401611328125, 762.3734130859375)
    PRPP (59.09102249145508, -141.71621704101562, 775.1181030273438)
    PRDP (57.41969680786133, -137.11688232421875, 748.3690185546875)
    PRHP (61.70811462402344, -138.3133544921875, 751.9981079101562)
    ```
    """
    ##########################################
    ##########################################
    # Load segmentation

    # Prep atlas
    atlas_path = out_folder / f"atlas{atlas_id:03}{addendum}.nii.gz"
    atlas_ct_path = out_folder / f"atlas{atlas_id:03}{addendum}_ct.nii.gz"
    atlas_path_v = out_folder / f"atlas{atlas_id:03}{addendum}_visual.nii.gz"
    atlas_cms_poi_path = out_folder / f"atlas{atlas_id:03}{addendum}_cms_poi.json"  # Center of mass
    atlas_poi_path = out_folder / f"atlas{atlas_id:03}{addendum}_poi.json"

    prep_Atlas(segmentation_path, ct_path, atlas_path, atlas_ct_path, atlas_cms_poi_path, text_file_is_left_leg, flip=True)

    if not atlas_path_v.exists() and not atlas_poi_path.exists():
        seg = to_nii(atlas_path, True)
        poi = (
            parse_coordinates_to_poi(file_text, True).to_other(seg)
            if ".txt" in str(file_text)
            else POI.load(file_text).resample_from_to(seg)
        )
        if not text_file_is_left_leg:
            for k1, k2, (x, y, z) in poi.items():
                axis = poi.get_axis("R")
                if axis == 0:
                    poi[k1, k2] = (poi.shape[0] - 1 - x, y, z)
                elif axis == 1:
                    poi[k1, k2] = (x, poi.shape[1] - 1 - y, z)
                elif axis == 2:
                    poi[k1, k2] = (x, y, poi.shape[2] - 1 - z)
                else:
                    raise ValueError(axis)
        poi.level_one_info = Full_Body_Instance
        poi.level_two_info = Lower_Body
        poi.info["source_file_name"] = Path(file_text).name
        poi.to_global().save(atlas_poi_path)
        c = poi.make_point_cloud_nii()[1]
        c[c != 0] += 100
        a = to_nii(atlas_path, True)
        print(a, c)
        a[c != 0] = 0
        (a + c).save(atlas_path_v)


def parse_coordinates(file_path: str | Path) -> dict[str, tuple[float, float, float]]:
    coordinates = {}

    with open(file_path) as file:
        for line in file:
            parts = str(line).replace("(", "").replace(")", "").replace(",", "").replace(":", "").split(" ")
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


def parse_poi_to_coordinates(poi: POI_Global, output_path: str | Path, left_side: bool | None = None):
    poi = poi.to_global()
    left_side = ("LEFT" if "LEFT" in str(output_path).upper() else "RIGHT") if left_side is None else "LEFT" if left_side else "RIGHT"
    output_path = Path(output_path)

    lines = []
    lines.append("Points Export\n")
    lines.append(f"Side: {left_side.upper()}\n\n")
    ENUM_TO_ABBREVIATION = {k[1].value: v for v, k in ABBREVIATION_TO_ENUM.items()}
    bone_structure = {"Femur proximal": [], "Femur distal": [], "Tibia proximal": [], "Tibia distal": [], "Patella": []}

    for _, k, coords in poi.items():
        abbrev = ENUM_TO_ABBREVIATION[k]
        x, y, z = coords
        coord_str = f"{abbrev}: ({x}, {y}, {z})"
        # Assign to bone section heuristically (adjust as needed for your enums)
        if abbrev in ["TGT", "FHC", "FNC", "FAAP"]:
            bone_structure["Femur proximal"].append(coord_str)
        elif abbrev in ["FLCD", "FMCD", "FLCP", "FMCP", "FNP", "FADP", "TGPP", "TGCP", "FMCPC", "FLCPC", "TRMP", "TRLP"]:
            bone_structure["Femur distal"].append(coord_str)
        elif abbrev in ["TLCL", "TMCM", "TKC", "TLCA", "TLCP", "TMCA", "TMCP", "TTP", "TAAP", "TMIT", "TLIT"]:
            bone_structure["Tibia proximal"].append(coord_str)
        elif abbrev in ["FLM", "TMM", "TAC", "TADP"]:
            bone_structure["Tibia distal"].append(coord_str)
        elif abbrev in ["PPP", "PDP", "PMP", "PLP", "PRPP", "PRDP", "PRHP"]:
            bone_structure["Patella"].append(coord_str)
        else:
            raise ValueError(f"Unknown abbreviation '{abbrev}'")

    # Write structured output
    for section, coords in bone_structure.items():
        if coords:
            lines.append(f"{section}:\n")
            lines.extend([line + "\n" for line in coords])
            lines.append("\n")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        f.writelines(lines)


default_setting = {
    "loss": {
        "config": {"be": {"stride": 1, "name": "BSplineBending"}, "seg": {"name": "MSE"}},
        "weights": {"be": 0.0001, "seg": 1, "tether": 0.001},
    },
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

# reduce be if no overfitting possible
# reduce lr if optimization platos
# increase min_delta when it stops to early

PATELLA = [14, 114]
LEGS = [13, 113, *PATELLA, 115, 15, 116, 16]


def split_left_right_leg(seg520: NII, c=2, min_volume=50) -> NII:
    cc_patella = seg520.extract_label(PATELLA).get_connected_components(1)
    seg520 = seg520.extract_label(LEGS, keep_label=True)
    m = cc_patella.max()
    if m == 0:
        print("No Leg")
        return seg520 * 0
    elif m == 1:
        print("Only One Leg")
        return seg520.clamp(0, 1)
    a = [ceil(c * z) for z in seg520.spacing]
    nii_small: NII = seg520[:: a[0], :: a[1], :: a[2]]
    legs = nii_small.extract_label(LEGS)
    while True:
        legs_cc = legs.filter_connected_components(1, min_volume=min_volume, keep_label=False)
        m = legs_cc.max()
        if m == 0:
            raise ValueError("segmentation_to_small or no segmentation", legs.unique())
        elif m == 1:
            if c == 0:
                raise ValueError("Legs are touching, can not split them into into two...")
            print("reduce c")
            return split_left_right_leg(seg520, floor(c / 2), min_volume=min_volume)
        elif m == 2:
            break
        legs = legs.dilate_msk(1)
    nii_left_right = seg520.clamp(0, 1) * legs_cc.resample_from_to(seg520).dilate_msk(c)

    axis_r = nii_left_right.get_axis("R")
    label_positions = {label: np.mean(np.where(nii_left_right == label)[axis_r]) for label in [1, 2]}
    more_right_label = label_positions[1] > label_positions[2]
    if nii_left_right.orientation[axis_r] == "R" and more_right_label:
        nii_left_right.map_labels_({1: 2, 2: 1})
    if nii_left_right.orientation[axis_r] == "L" and not more_right_label:
        nii_left_right.map_labels_({1: 2, 2: 1})
    return nii_left_right


def prep_Atlas(
    seg: NII | str | Path,
    ct: NII | str | Path,
    out_atlas: Path,
    ct_atlas: Path,
    atlas_centroids: Path,
    atlas_left=True,
    flip=False,
    max_count_component=4,
):
    if not out_atlas.exists() or not ct_atlas.exists():
        seg = to_nii(seg, True)
        ct = to_nii(ct, False)
        logger.on_text("split_left_right_leg")
        split_nii = split_left_right_leg(seg)
        atlas = split_nii.extract_label(1 if atlas_left else 2) * seg
        if not atlas_left and flip:
            axis = atlas.get_axis("R")
            if axis == 0:
                atlas = atlas.set_array(atlas.get_array()[::-1]).copy()
                ct = ct.set_array(ct.get_array()[::-1]).copy()
            elif axis == 1:
                atlas = atlas.set_array(atlas.get_array()[:, ::-1]).copy()
                ct = ct.set_array(ct.get_array()[:, ::-1]).copy()
            elif axis == 2:
                atlas = atlas.set_array(atlas.get_array()[:, :, ::-1]).copy()
                ct = ct.set_array(ct.get_array()[:, :, ::-1]).copy()
        atlas = atlas.filter_connected_components(atlas.unique(), max_count_component=max_count_component, keep_label=True, connectivity=1)
        crop = atlas.compute_crop(1, dist=100)
        atlas.apply_crop_(crop).save(out_atlas)
        ct.apply_crop_(crop).save(ct_atlas)
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
        gaussian_sigma=0,
        ddevice: DEVICES = "cuda",
        bone_by_bone=False,
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
        if bone_by_bone:
            self.left = Register_Point_Atlas_bone_by_bone(
                l,
                atlas,
                poi_atlas,
                same_side=atlas_left,
                verbose=verbose,
                setting=setting_1,
                gpu=gpu,
                ddevice=ddevice,
                gaussian_sigma=gaussian_sigma,
            )
            logger.on_text("Register_Point_Atlas_One_Leg right")
            self.right = Register_Point_Atlas_bone_by_bone(
                r,
                atlas,
                poi_atlas,
                same_side=not atlas_left,
                verbose=verbose,
                setting=setting_1,
                gpu=gpu,
                ddevice=ddevice,
                gaussian_sigma=gaussian_sigma,
            )
        else:
            self.left = Register_Point_Atlas_One_Leg(
                l,
                atlas,
                poi_atlas,
                same_side=atlas_left,
                verbose=verbose,
                setting=setting_1,
                gpu=gpu,
                ddevice=ddevice,
                gaussian_sigma=gaussian_sigma,
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
                gaussian_sigma=gaussian_sigma,
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
        poi_cms: POI | None,
        same_side: bool,
        verbose=99,
        # setting=default_setting,
        # setting_patella=default_setting,
        gpu=0,
        ddevice: DEVICES = "cuda",
        gaussian_sigma=0,
        loss_terms=None,  # type: ignore
        weights=None,
        weights_patella=None,
        lr=0.001,
        lr_patella=0.001,
        max_steps=1500,
        min_delta=0.00001,
        pyramid_levels=4,
        coarsest_level=3,
        finest_level=0,
        satellite_structure=PATELLA,
        **args,
    ):
        self.satellite_structure = satellite_structure
        # Assumes that you have removed the other leg.
        if weights is None:
            weights = {"be": 0.0001, "seg": 1}
        if weights_patella is None:
            weights_patella = {"be": 0.0001, "seg": 1}
        if loss_terms is None:
            loss_terms = {"be": ("BSplineBending", {"stride": 1}), "seg": "MSE"}
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
            self.t_crop = t_crop
            target_ = target.apply_crop(t_crop)
            # Point Registration
            poi_target = calc_centroids(target_, second_stage=40)
            if poi_cms is None:
                poi_cms = calc_centroids(atlas, second_stage=40)  # This will be needlessly computed all the time
            if not poi_cms.assert_affine(atlas, raise_error=False):
                poi_cms = poi_cms.resample_from_to(atlas)
            self.reg_point = Point_Registration(poi_target, poi_cms)
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
        if gaussian_sigma > 0:
            target.seg = False
            target.set_dtype_(np.float32)
            atlas_reg.seg = False
            atlas_reg.set_dtype_(np.float32)
            target = target.smooth_gaussian(gaussian_sigma)
            atlas_reg = atlas_reg.smooth_gaussian(gaussian_sigma)

        self.reg_deform = Deformable_Registration(
            target.remove_labels(satellite_structure),
            atlas_reg.remove_labels(satellite_structure),
            loss_terms=loss_terms,  # type: ignore
            weights=weights,
            lr=lr,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            verbose=verbose,
            **args,
        )
        if len(self.satellite_structure) == 0:
            self.crop_patella = None
            self.reg_deform_p = None
            return
        # self.reg_deform = Deformable_Registration_old(target, atlas_reg, config=setting, verbose=verbose, gpu=gpu, ddevice=ddevice)
        atlas_reg = self.reg_deform.transform_nii(atlas_reg)
        # Patella
        patella_atlas = atlas_reg.extract_label(satellite_structure)
        patella_target = target.extract_label(satellite_structure)

        self.crop_patella = (patella_target + patella_atlas).compute_crop(0, 20)
        patella_atlas.apply_crop_(self.crop_patella)
        patella_target.apply_crop_(self.crop_patella)
        self.reg_deform_p = Deformable_Registration(
            patella_target,
            atlas_reg,
            loss_terms=loss_terms,
            weights=weights_patella,
            lr=lr_patella,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            verbose=verbose,
            gpu=gpu,
            ddevice=ddevice,
        )

    def get_dump(self):
        return (
            1,  # version
            (self.reg_point.get_dump()),
            (self.reg_deform.get_dump()),
            (self.reg_deform_p.get_dump() if self.reg_deform_p is not None else None),
            (
                self.same_side,
                self.atlas_org,
                self.target_grid_org,
                self.target_grid,
                self.crop,
                self.crop_patella,
                self.satellite_structure,
            ),
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
        (version, t0, t1, t2, x, satellite_structure) = w
        assert version == 1, f"Version mismatch {version=}"
        self = cls.__new__(cls)
        self.reg_point = Point_Registration.load_(t0)
        self.reg_deform = Deformable_Registration.load_(t1)
        self.reg_deform_p = Deformable_Registration.load_(t2)
        self.satellite_structure = satellite_structure
        self.same_side, self.atlas_org, self.target_grid_org, self.target_grid, self.crop, self.crop_patella = x
        return self

    def forward_nii(self, nii_atlas: NII):
        nii_atlas = self.reg_point.transform_nii(nii_atlas)
        nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = self.reg_deform.transform_nii(nii_atlas)
        if self.crop_patella is not None and self.reg_deform_p is not None:
            # Patella
            patella_atlas = nii_reg.extract_label(*self.satellite_structure)
            nii_reg.remove_labels_(self.satellite_structure)
            patella_atlas_reg = self.reg_deform_p.transform_nii(patella_atlas)
            patella_atlas_reg.resample_from_to_(nii_reg, mode="constant")
            nii_reg[patella_atlas_reg != 0] = patella_atlas_reg[patella_atlas_reg != 0]
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

    def forward_txt(self, file_path: str | Path, left: bool):
        poi_glob = parse_coordinates_to_poi(file_path, left)
        return self.forward_poi(poi_glob, left)

    def forward_poi(self, poi_atlas: POI_Global | POI):
        poi_atlas = poi_atlas.resample_from_to(self.atlas_org)

        # Point Reg
        poi_atlas = self.reg_point.transform_poi(poi_atlas)
        # Deformable
        poi_atlas = poi_atlas.apply_crop(self.crop)
        poi_reg = self.reg_deform.transform_poi(poi_atlas)
        if self.crop_patella is not None and self.reg_deform_p is not None:
            # Patella
            poi_patella = poi_reg.apply_crop(self.crop_patella).extract_region(*self.satellite_structure)
            patella_poi_reg = self.reg_deform_p.transform_poi(poi_patella)
            for k1, k2, v in patella_poi_reg.resample_from_to(poi_reg).items():
                poi_reg[k1, k2] = v
        poi_reg = poi_reg.resample_from_to(self.target_grid_org)
        poi_reg.level_one_info = Full_Body_Instance
        poi_reg.level_two_info = Lower_Body
        if self.same_side:
            return poi_reg
        for k1, k2, v in poi_reg.copy().items():
            k = k1 % 100
            # if left:
            #    k += 100
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

    def invert_points(
        self, points, axes: Axes, grid: Deepali_Grid | Has_Grid, to_axes: Axes = Axes.CUBE, to_grid: Deepali_Grid | Has_Grid | None = None
    ):
        raise NotImplementedError("This method is not tested and might not work as expected.")
        import torch

        reg_deform = self.reg_deform.inverse()
        # invert reg_deform
        points = reg_deform.transform_points(points, axes=axes, to_axes=Axes.GRID, grid=grid, to_grid=self.target_grid)
        # invert self.crop
        shift = torch.tensor([x.start for x in self.crop]).unsqueeze(0).to(points.device)
        points = points + shift
        # invert reg_point
        out = []
        for x, y, z in points:
            o = self.reg_point.transform_cord_inverse((x, y, z))  # type: ignore
            out.append(o)
        out = torch.tensor(out)
        # invert self.t_crop
        shift = torch.tensor([x.start for x in self.t_crop]).unsqueeze(0)
        points = points + shift
        # flip if needed
        if not self.same_side:
            grid = self.target_grid_org
            axis = grid.get_axis("R")
            if axis == 0:
                out[:, 0] = grid.shape[0] - 1 - out[:, 0]
            elif axis == 1:
                out[:, 1] = grid.shape[1] - 1 - out[:, 1]
            elif axis == 2:
                out[:, 2] = grid.shape[2] - 1 - out[:, 2]
            else:
                raise ValueError(axis)
        # transform to axes and grid
        if to_grid is not None or to_axes != Axes.CUBE:
            if to_grid is None:
                to_grid = self.target_grid_org
            if isinstance(to_grid, Has_Grid):
                to_grid = to_grid.to_deepali_grid()
            out = to_grid.transform_points(out, axes.CUBE, to_axes=to_axes, to_grid=to_grid)
        return out


class Register_Point_Atlas_bone_by_bone:
    def __init__(
        self,
        target: NII,
        atlas: NII,
        poi_cms: POI | None,
        same_side: bool,
        verbose=99,
        setting=default_setting,
        gpu=0,
        ddevice: DEVICES = "cuda",
        gaussian_sigma=0,
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
            if poi_cms is None:
                poi_cms = calc_centroids(atlas, second_stage=40)  # This will be needlessly computed all the time
            if not poi_cms.assert_affine(atlas, raise_error=False):
                poi_cms = poi_cms.resample_from_to(atlas)
            self.reg_point = Point_Registration(poi_target, poi_cms)
            atlas_reg = self.reg_point.transform_nii(atlas)
            if not atlas_reg.is_segmentation_in_border():
                target = target_
                break
            print(f"Try point reg again; padding {i=} was to small")

        # Major Bones
        self.crop = (target + atlas_reg).compute_crop(0, 2)
        target_ = target.apply_crop(self.crop)
        atlas_reg_ = atlas_reg.apply_crop(self.crop)
        self.target_grid = target_.to_gird()
        self.reg_deform = {}
        for i in target_.unique():
            logger.on_log("new Reg with id =", i)
            target = target_.extract_label(i)
            atlas_reg = atlas_reg_.extract_label(i)
            if gaussian_sigma > 0:
                target.seg = False
                target.set_dtype_(np.float32)
                atlas_reg.seg = False
                atlas_reg.set_dtype_(np.float32)
                target = target.smooth_gaussian(gaussian_sigma)
                atlas_reg = atlas_reg.smooth_gaussian(gaussian_sigma)

            self.reg_deform[i] = Deformable_Registration(
                target,
                atlas_reg,
                loss_terms={"be": ("BSplineBending", {"stride": 1}), "loss": "MSE", "tether": Tether()},  # type: ignore
                weights={
                    "be": setting["loss"]["weights"]["be"],
                    "loss": setting["loss"]["weights"]["seg"],
                    "tether": setting["loss"]["weights"]["tether"],
                },
                lr=setting["optim"]["args"]["lr"],
                max_steps=setting["optim"]["loop"]["max_steps"],
                min_delta=setting["optim"]["loop"]["min_delta"],
                pyramid_levels=4,
                coarsest_level=3,
                finest_level=0,
                verbose=verbose,
                gpu=gpu,
                ddevice=ddevice,
            )

    def get_dump(self):
        raise NotImplementedError()

    def save(self, path: str | Path):
        with open(path, "wb") as w:
            pickle.dump(self.get_dump(), w)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as w:
            return cls.load_(pickle.load(w))

    @classmethod
    def load_(cls, w):
        raise NotImplementedError()

    def forward_nii(self, nii_atlas: NII):
        nii_atlas = self.reg_point.transform_nii(nii_atlas)
        nii_atlas = nii_atlas.apply_crop(self.crop)
        nii_reg = None
        for i, deformer in self.reg_deform.items():
            a = deformer.transform_nii(nii_atlas.extract_label([i, i + 100]))
            if nii_reg is None:
                nii_reg = a
            else:
                nii_reg[np.logical_and(nii_reg == 0, a == 1)] = i
        assert nii_reg is not None
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
        poi_reg = None

        for i, deformer in self.reg_deform.items():
            a: POI = deformer.transform_poi(poi_atlas.extract_region(i, i + 100))
            if poi_reg is None:
                poi_reg = a
            else:
                poi_reg.join_left_(a)
        assert poi_reg is not None
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
    root = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/atlas/")
    atlas = root / "atlas002.nii.gz"
    root2 = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/0000/")

    target = to_nii(root2 / "bone.nii.gz", True)
    reg = Register_Point_Atlas(
        target,
        to_nii(atlas, True),
        atlas_left=True,
        split_leg_path=root2 / "test" / "split_leg.nii.gz",
        atlas_centroids=root / "atlas002_left_cms_poi.json",
        bone_by_bone=True,
    )
    poi_atlas = POI.load(root / "atlas002_poi.json", target)
    # nii = poi_out.resample_from_to(target).make_point_cloud_nii(s=5)[1]
    # nii2 = target.clamp(0, 1) * 100
    # nii[nii == 0] = nii2[nii == 0]
    # nii.save(root2 / "test" / "atlas_reg_poi3.nii.gz")
    poi = reg.make_poi_from_poi(poi_atlas, root2 / "test" / "out_points_both3.json")
    poi.to_global().save_mrk(root2 / "test" / "out_points_both3.json")
    # reg.save("Register_Point_Atlas.pkl")
    # CUDA_VISIBLE_DEVICES=2 /opt/anaconda3/envs/py3.11/bin/python /DATA/NAS/tools/TPTBox/tmp_poi_w.py
    sys.exit()
    st = time()

    lr = to_nii(root / "bone_lr.nii.gz", True)
    bone = to_nii("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/0001/bone.nii.gz", True)
    x = split_left_right_leg(bone)
    assert x is not None
    x.save(root / "test" / "left_right_split.nii.gz")
    print(time() - st)
    sys.exit()
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
    poi_atlas = reg_obj.forward_txt(file_path="010__left.txt")
    nii_out = reg_obj.forward_nii(left_nii_atlas)
    poi_atlas.save(root / "test" / "atlas_reg_poi.json")
    nii_out.save(root / "test" / "atlas_reg.nii.gz")
    nii = poi_atlas.make_point_cloud_nii(s=5)[1]
    nii2 = right_nii_target.clamp(0, 1) * 100
    nii[nii == 0] = nii2[nii == 0]
    nii.save(root / "test" / "atlas_reg_poi.nii.gz")
    p = poi_atlas.to_global(itk_coords=True)
    coords_dict = parse_coordinates("010__left.txt")
    for e, (k2, _) in enumerate(coords_dict.items(), 1):
        k = PATELLA[0] if k2[0] == "P" else 60
        print(k2, p[k, e])

    print(time() - st)
    # reg_obj.save(root / "test" / "Register_Point_Atlas.pkl")
