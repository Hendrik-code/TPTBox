from TPTBox.mesh3D.snapshot3D import make_snapshot3D

if __name__ == "__main__":
    labels = {
        1: {"typ": "organ", "name": "spleen", "min": 1, "max": 1, "autofix": 100},
        2: {"typ": "organ", "name": "kidney_right", "min": 1, "max": 1, "autofix": 100},
        3: {"typ": "organ", "name": "kidney_left", "min": 1, "max": 1, "autofix": 100},
        4: {"typ": "organ", "name": "gallbladder", "min": 1, "max": 1, "autofix": 40},
        5: {"typ": "organ", "name": "liver", "min": 1, "max": 1, "autofix": 200},
        6: {"typ": "digenstion", "name": "stomach", "min": 1, "max": 1, "autofix": 10},
        7: {"typ": "digenstion", "name": "pancreas", "min": 1, "max": 1, "rois": [4, 3, 8, 9], "autofix": 5},
        8: {"typ": "vessel", "name": "adrenal_gland_right", "min": 1, "max": 1, "autofix": 30},
        9: {"typ": "vessel", "name": "adrenal_gland_left", "min": 1, "max": 1, "autofix": 30},
        10: {"typ": "lung", "name": "lung_upper_lobe_left", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
        11: {"typ": "lung", "name": "lung_lower_lobe_left", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
        12: {"typ": "lung", "name": "lung_upper_lobe_right", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100},
        13: {"typ": "lung", "name": "lung_middle_lobe_right", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100},
        14: {"typ": "lung", "name": "lung_lower_lobe_right", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100},
        15: {"typ": "digenstion", "name": "esophagus", "min": 1, "max": 1, "rois": [5, 9, 10], "autofix": 10},
        16: {"typ": "lung", "name": "trachea", "min": 1, "max": 1, "autofix": 10},
        17: {"typ": "organ", "name": "thyroid_gland", "min": 2, "max": 2, "autofix": 10},
        18: {"typ": "digenstion", "name": "intestine", "min": 1, "max": 1, "autofix": 20},
        19: {"typ": "digenstion", "name": "duodenum", "min": 1, "max": 1, "autofix": 10},
        20: {"typ": "digenstion", "name": "unused", "min": 1, "max": 1, "autofix": 10},
        21: {"typ": "organ", "name": "urinary_bladder", "min": 1, "max": 1, "autofix": 15},
        22: {"typ": "organ", "name": "prostate", "min": 1, "max": 1, "autofix": 10},
        23: {"typ": "bone", "name": "sacrum", "min": 1, "max": 1, "autofix": 10},
        24: {"typ": "organ", "name": "heart", "min": 1, "max": 1, "autofix": 10},
        25: {"typ": "vessel", "name": "aorta", "min": 1, "max": 1, "autofix": 10},
        26: {"typ": "vessel", "name": "pulmonary_vein", "min": 2, "max": 2, "autofix": 30},
        27: {"typ": "vessel", "name": "brachiocephalic_trunk", "min": 1, "max": 1, "autofix": 30},
        28: {"typ": "vessel", "name": "subclavian_artery_right", "min": 1, "max": 1, "autofix": 30},
        29: {"typ": "vessel", "name": "subclavian_artery_left", "min": 1, "max": 1, "autofix": 30},
        30: {"typ": "vessel", "name": "common_carotid_artery_right", "min": 1, "max": 1, "autofix": 30},
        31: {"typ": "vessel", "name": "common_carotid_artery_left", "min": 1, "max": 1, "autofix": 30},
        32: {"typ": "vessel", "name": "brachiocephalic_vein_left", "min": 1, "max": 1, "autofix": 30},
        33: {"typ": "vessel", "name": "brachiocephalic_vein_right", "min": 1, "max": 1, "autofix": 30},
        34: {"typ": "vessel", "name": "atrial_appendage_left", "min": 1, "max": 1, "autofix": 30},
        35: {"typ": "vessel", "name": "superior_vena_cava", "min": 1, "max": 1, "autofix": 30},
        36: {"typ": "vessel", "name": "inferior_vena_cava", "min": 1, "max": 1, "autofix": 30},
        37: {"typ": "vessel", "name": "portal_vein_and_splenic_vein", "min": 1, "max": 1, "autofix": 30},
        38: {"typ": "vessel", "name": "iliac_artery_left", "min": 1, "max": 1, "autofix": 30},
        39: {"typ": "vessel", "name": "iliac_artery_right", "min": 1, "max": 1, "autofix": 30},
        40: {"typ": "vessel", "name": "iliac_vena_left", "min": 1, "max": 1, "autofix": 30},
        41: {"typ": "vessel", "name": "iliac_vena_right", "min": 1, "max": 1, "autofix": 30},
        42: {"typ": "bone", "name": "humerus_left", "min": 1, "max": 1, "autofix": 200},
        43: {"typ": "bone", "name": "humerus_right", "min": 1, "max": 1, "autofix": 200},
        44: {"typ": "bone", "name": "scapula_left", "min": 1, "max": 1, "autofix": 50},
        45: {"typ": "bone", "name": "scapula_right", "min": 1, "max": 1, "autofix": 50},
        46: {"typ": "bone", "name": "clavicula_left", "min": 1, "max": 1, "autofix": 50},
        47: {"typ": "bone", "name": "clavicula_right", "min": 1, "max": 1, "autofix": 50},
        48: {"typ": "bone", "name": "femur_left", "min": 1, "max": 1, "autofix": 200},
        49: {"typ": "bone", "name": "femur_right", "min": 1, "max": 1, "autofix": 200},
        50: {"typ": "bone", "name": "hip_left", "min": 1, "max": 1, "autofix": 100},
        51: {"typ": "bone", "name": "hip_right", "min": 1, "max": 1, "autofix": 100},
        52: {"typ": "cns", "name": "spinal_cord", "min": 1, "max": 1, "autofix": 5},
        53: {"typ": "muscle", "name": "gluteus_maximus_left", "min": 1, "max": 1, "autofix": 400},
        54: {"typ": "muscle", "name": "gluteus_maximus_right", "min": 1, "max": 1, "autofix": 400},
        55: {"typ": "muscle", "name": "gluteus_medius_left", "min": 1, "max": 1, "autofix": 400},
        56: {"typ": "muscle", "name": "gluteus_medius_right", "min": 1, "max": 1, "autofix": 400},
        57: {"typ": "muscle", "name": "gluteus_minimus_left", "min": 1, "max": 1, "autofix": 400},
        58: {"typ": "muscle", "name": "gluteus_minimus_right", "min": 1, "max": 1, "autofix": 400},
        59: {"typ": "muscle", "name": "autochthon_left", "min": 1, "max": 1, "autofix": 100},
        60: {"typ": "muscle", "name": "autochthon_right", "min": 1, "max": 1, "autofix": 100},
        61: {"typ": "muscle", "name": "iliopsoas_left", "min": 1, "max": 1, "autofix": 600},
        62: {"typ": "muscle", "name": "iliopsoas_right", "min": 1, "max": 1, "autofix": 600},
        63: {"typ": "bone", "name": "sternum", "min": 1, "max": 1, "autofix": 15, "rois": [4, 5, 6, 9, 10], "rm_roi": [1, 2, 3]},
        64: {"typ": "bone", "name": "costal_cartilages", "min": 10, "max": 30, "autofix": 2, "rois": [4, 5, 6, 9, 10], "rm_roi": [1, 2, 3]},
        65: {"typ": "rest", "name": "subcutaneous_fat", "min": 1, "max": 1000, "autofix": 2},
        66: {"typ": "rest", "name": "muscle", "min": 1, "max": 1000, "autofix": 2},
        67: {"typ": "rest", "name": "inner_fat", "min": 1, "max": 1000, "autofix": 2},
        68: {"typ": "bone", "name": "IVD", "min": 1, "max": 25, "autofix": 2},
        69: {"typ": "bone", "name": "vertebra_body", "min": 1, "max": 25, "autofix": 2},
        70: {"typ": "bone", "name": "vertebra_posterior_elements", "min": 1, "max": 25},
        71: {"typ": "cns", "name": "spinal_channel", "min": 1, "max": 1, "autofix": 5},
        72: {"typ": "bone", "name": "bone_other", "min": 0, "max": 10, "autofix": 50},
    }
    l_types = ["bone", "digenstion", "lung", "muscle", "organ", "vessel", "cns", "rest"]
    idxs = []
    for t in l_types:
        idx = []
        for k, v in labels.items():
            if v["typ"] == t:
                idx.append(k)
        idxs.append(idx)
    #
    # make_sub_snapshot(
    #    NII.load(
    #        "/DATA/NAS/ongoing_projects/robert/datasets/Totalvibeseg/101_abdomen_test/sub-101529/sub-101529_sequ-stitched_acq-ax_seg-all-combinded_msk.nii.gz",
    #        True,
    #    ),
    #    "fury.png",
    #    ["A"],
    #    idxs,
    #    width_factor=0.6,
    # )
    # make_sub_snapshot(
    #    "/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_spine_inference_148_preliminary/101/101000/T2w/sub-101000_sequ-stitched_acq-sag_mod-T2w_seg-spine_msk.nii.gz",
    #    "spine.png",
    #    ["A", "L", "P", "R"],
    # )
    make_snapshot3D(
        "/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_spine_inference_148_preliminary/101/101000/T2w/sub-101000_sequ-stitched_acq-sag_mod-T2w_seg-vert_msk.nii.gz",
        "vert.png",
        orientation=["A", "L", "P", "R"],
    )  # "L", "P", "R"
    # from TPTBox.mesh3D.snapshot3D import make_snapshot3D
    #
    ## all segmentation; orientation give the rotation of an image
    # make_snapshot3D("sub-101000_msk.nii.gz", "snapshot3D.png", orientation=["A", "L", "P", "R"])
    ## Select witch segmentation per panel are chosen.
    # make_snapshot3D("sub-101000_msk.nii.gz", "snapshot3D.png", orientation=["A"], ids_list=[[1, 2], [3]])
