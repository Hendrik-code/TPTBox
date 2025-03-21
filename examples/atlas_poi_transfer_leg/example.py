import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Step 1
from pathlib import Path

from atlas_poi_transfer import parse_coordinates_to_poi, prep_Atlas

from TPTBox import POI, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance, Lower_Body

##########################################
# Settings
text_file_is_left_leg = True
file_text = "/DATA/NAS/tools/TPTBox/examples/atlas_poi_transfer_leg/010__left.txt"
segmentation_path = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/0001/bone.nii.gz"
out_folder = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/atlas")
atlas_id = 1
##########################################
# Load segmentation
seg = to_nii(segmentation_path, True)

if not text_file_is_left_leg:
    axis = seg.get_axis("R")
    if axis == 0:
        target = seg.set_array(seg.get_array()[::-1]).copy()
    elif axis == 1:
        target = seg.set_array(seg.get_array()[:, ::-1]).copy()
    elif axis == 2:
        target = seg.set_array(seg.get_array()[:, :, ::-1]).copy()
assert text_file_is_left_leg, "Not implement: Flip NII and POI"
# Prep atlas
atlas_path = out_folder / f"atlas{atlas_id:03}.nii.gz"
atlas_cms_poi_path = out_folder / f"atlas{atlas_id:03}_cms_poi.json"  # Center of mass
atlas_poi_path = out_folder / f"atlas{atlas_id:03}_poi.json"
prep_Atlas(seg, atlas_path, atlas_cms_poi_path, text_file_is_left_leg)


poi = parse_coordinates_to_poi(file_text, True).to_other(seg) if ".txt" in file_text else POI.load(file_text).resample_from_to(seg)
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
poi.to_global().save(atlas_poi_path)
# Step 1
from pathlib import Path

from atlas_poi_transfer import Register_Point_Atlas

from TPTBox import POI, to_nii

##########################################
for i in range(500):
    # Settings
    target_seg_path = (
        f"/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/Dataset001_all/{i:04}/bone.nii.gz"  # TODO Path to target seg
    )
    s = str(target_seg_path).split(".")[0]
    split_leg_path = s + "_seg-left-right-split_msk.nii.gz"
    out_new_pois = s + "_desc-leg_poi.json"
    out_new_pois_nii = s + "_desc-leg_poi.nii.gz"
    atlas_id = 1
    ddevice = "cuda"
    gpu = 0
    if not Path(target_seg_path).exists() or Path(out_new_pois_nii).exists():
        continue
    ##########################################
    # Atlas
    atlas_p = out_folder / f"atlas{atlas_id:03}.nii.gz"
    atlas_centroids = out_folder / f"atlas{atlas_id:03}_cms_poi.json"  # Center of mass
    atlas_poi_path = out_folder / f"atlas{atlas_id:03}_poi.json"
    # Load segmentation
    target = to_nii(target_seg_path, True)
    atlas = to_nii(atlas_p, True)

    # Creating this object will start the registration
    registration_obj = Register_Point_Atlas(
        target, atlas, split_leg_path=split_leg_path, atlas_centroids=atlas_centroids, gpu=gpu, ddevice=ddevice, verbose=0
    )
    out_poi = registration_obj.make_poi_from_poi(POI.load(atlas_poi_path), out_new_pois)
    nii = out_poi.make_point_cloud_nii()[1] + to_nii(split_leg_path, True) * 100
    nii.save(out_new_pois_nii)
