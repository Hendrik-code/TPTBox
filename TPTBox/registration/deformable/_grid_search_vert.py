from pathlib import Path

import numpy as np

from TPTBox import POI, Image_Reference, calc_centroids, calc_poi_from_two_segs
from TPTBox.core.nii_wrapper import NII, to_nii
from TPTBox.core.poi_fun.ray_casting import calculate_pca_normal_np
from TPTBox.registration.deformable.deformable_reg import Deformable_Registration
from TPTBox.registration.ridged_points import ridged_points_from_poi

setting = {
    "loss": {"config": {"be": {"stride": 1, "name": "BSplineBending"}, "seg": {"name": "MSE"}}, "weights": {"be": 0.001, "seg": 1}},
    "model": {"name": "SVFFD", "args": {"stride": [4, 4, 4], "transpose": False}, "init": None},
    "optim": {"name": "Adam", "args": {"lr": 0.1}, "loop": {"max_steps": 1000, "min_delta": -0.001}},
    "pyramid": {
        "levels": 4,  # 4 for 0.8 res instead of 3
        "coarsest_level": 3,  # 3 for 0.8 res instead of 2
        "finest_level": 0,
        "finest_spacing": None,  # Auto set by the nifty
        "min_size": 16,
        "pyramid_dims": ["x", "y", "z"],
    },
}


def main_vert_test():
    p = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse20/derivatives/sub-verse510/")
    sub = p / "sub-verse510_dir-ax_seg-subreg_msk.nii.gz"
    vert = p / "sub-verse510_dir-ax_seg-vert_msk.nii.gz"
    out = Path("/DATA/NAS/ongoing_projects/robert/test/seg_transplant")
    # Load and extract two vertebras
    vert = to_nii(vert, True)
    sub = to_nii(sub, True)  # .resample_from_to(vert)
    L1 = vert.extract_label(21)  # noqa: N806
    L2 = vert.extract_label(22)  # noqa: N806
    c1 = L1.compute_crop(dist=10)
    L1.apply_crop_(c1)
    c2 = L2.compute_crop(dist=10)
    L2.apply_crop_(c2)
    L1.save(out / "L1.nii.gz")
    L2.save(out / "L2.nii.gz")
    sub1 = sub.apply_crop(c1)
    sub2 = sub.apply_crop(c2)
    # Compute Points
    poi1 = calc_poi_from_two_segs(L1, sub1, out / "L1_cdt.json")
    poi2 = calc_poi_from_two_segs(L2, sub2, out / "L2_cdt.json")
    # Point registration
    reg = ridged_points_from_poi(poi1, poi2)
    L2_preg_sub = reg.transform_nii(sub2 * L2)  # noqa: N806
    L2_preg_sub.save(out / "L2_preg.nii.gz")
    L2_preg = reg.transform_nii(L2)  # noqa: N806
    # Deformable Registration
    reg_deform = Deformable_Registration(L1, L2_preg, config=setting)
    reg_deform.transform_nii(L2_preg_sub).save(out / "L2_reg_large_no_be.nii.gz")


def get_femurs(img: Image_Reference, seg_id=13):
    """Returns left (2) and right (1)

    Args:
        img (Image_Reference): _description_

    Returns:
        _type_: _description_
    """
    nii = to_nii(img, True)
    # Extract Femurs
    cc = nii.extract_label(seg_id)
    # CC
    cc = cc.get_connected_components()
    print(f"Warning more than two cc {cc.max()=}") if cc.max() >= 3 else None
    cc[cc > 2] = 0
    # Compute Poi of two largest CC (id = 1,2, cause sorted)
    femur_poi = calc_centroids(cc, second_stage=seg_id)
    # Extract left femur
    dim = femur_poi.get_axis("R")
    mirror = femur_poi.orientation[dim] == "R"
    left_id = 0 if femur_poi[1, seg_id][dim] > femur_poi[2, seg_id][dim] else 1
    if mirror:
        left_id = 1 - left_id
    left_id += 1
    get_additonal_point(femur_poi, cc, seg_id)
    if left_id == 2:
        return cc, femur_poi
    else:
        return cc.map_labels_({1: 2, 2: 1}, verbose=False), femur_poi.map_labels_(label_map_region={1: 2, 2: 1}, verbose=False)


def get_additonal_point(poi: POI, cc: NII, seg_id):
    for i in range(1, 3):
        try:
            normal_vector = calculate_pca_normal_np(cc.extract_label(i).get_array(), pca_component=0, zoom=poi.zoom)
        except ValueError:
            return
        dim = poi.get_axis("S")
        mirror = poi.orientation[dim] != "S"

        # check if it is pointing in the same direction
        flip = -1 if (normal_vector[dim] < 0 and mirror) or (normal_vector[dim] > 0 and not mirror) else 1

        direction_point = np.array(poi[i, seg_id]) + normal_vector * 20 * flip
        poi[1, seg_id + 100] = direction_point
        direction_point = np.array(poi[i, seg_id]) + normal_vector * 20 * -flip
        poi[1, seg_id + 200] = direction_point


if __name__ == "__main__":
    ### FEMUR ###
    p = Path("/DATA/NAS/ongoing_projects/robert/test/seg_transplant/bone/")
    if not (p / "femur_0001.nii.gz").exists() or True:
        femur_0, poi_0 = get_femurs(p / "bone_0000.nii.gz")
        femur_0.save(p / "femur_0000.nii.gz")
        poi_0.save(p / "femur_0000.json")
        femur_1, poi_1 = get_femurs(p / "bone_0001.nii.gz")
        femur_1.save(p / "femur_0001.nii.gz")
        poi_1.save(p / "femur_0001.json")
    femur_0_: NII = to_nii(p / "femur_0000.nii.gz", True).extract_label(1)  # TODO automatic Left/right selection
    poi_0 = POI.load(p / "femur_0000.json")
    femur_1 = to_nii(p / "femur_0001.nii.gz", True).extract_label(1)
    poi_1 = POI.load(p / "femur_0001.json").extract_region(1)
    # Point registration
    reg = ridged_points_from_poi(poi_0, poi_1)
    femur_1_reg = reg.transform_nii(femur_1)
    femur_1_reg.save(p / "femur_1_point_reg.nii.gz")
    subreg = reg.transform_nii(to_nii(p / "femur_1_subreg.nii.gz", True))
    # Crop to speed up and decreas gpu memory consumption.
    c0 = femur_0_.compute_crop(dist=10)
    c1 = femur_1_reg.compute_crop(dist=10)
    ex_slice = [slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(c1, c0, strict=False)]
    femur_0 = femur_0_.apply_crop(ex_slice)
    femur_1_reg.apply_crop_(ex_slice)

    # Deformable registration
    reg_deform = Deformable_Registration(femur_0, femur_1_reg, config=setting, verbose=99)
    reg_deform.transform_nii(femur_1_reg).resample_from_to(femur_0_).save(p / "femur_0001_reg.nii.gz")
    reg_deform.transform_nii(subreg).resample_from_to(femur_0_).save(p / "subreg_reg.nii.gz")

    # femur_1.save(p / "test_f1.nii.gz")
    # femur_0.save(p / "test_f0.nii.gz")
