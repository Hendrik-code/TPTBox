from __future__ import annotations
from functools import partial

import os.path
import warnings

import numpy as np
import torch

from TPTBox import BIDS_FILE, v_idx2name, v_name2idx, to_nii, v_idx_order
from TPTBox.centroids import order_vert_name
from TPTBox.snapshot3D import cut_single_vertebra
from tqdm import tqdm
from pathlib import Path
from TPTBox.snapshot3D.mesh_utils import (
    np_mask_to_mcubes_mesh,
    combine_meshes_along_y,
    snapshot_single_mesh,
    save_mesh,
    open3d_2_pvyista_mesh,
)
from TPTBox.snapshot3D.mesh_colors import RGB_Color, subreg3d_color_dict, vert3d_color_dict
import open3d as o3d
from PIL import Image


def make_subregion_3d_and_snapshot(
    vert_ref: BIDS_FILE,
    sub_ref: BIDS_FILE,
    vert_idx_list: list[int],
    save_individual_models: bool = False,
    save_combined_model: bool = False,
    save_individual_snapshots: bool = False,
    override: bool = False,
    verbose: bool = False,
):
    """from the vertebra mask and subregion mask creates the 3d model of the subregions as well as snapshots of those models

    Args:
        vert_ref: vertebra image reference
        sub_ref: subregion image reference
        vert_idx_list: list of vertebra indices to be used
        verbose:

    Returns:
        None
    """
    axcodes_to = ("R", "P", "I")
    from_v = v_idx2name[vert_idx_list[0]]
    to_v = v_idx2name[vert_idx_list[-1]]

    snapshot_path = sub_ref.get_changed_path(file_type="png", format="snp", info={"label": f"{from_v}-{to_v}"})
    # stop if snapshot already exists and override is not set
    if os.path.exists(snapshot_path) and not override:
        print(f"Combined snapshot already exists in {snapshot_path}, will skip")
        return

    vert_nii = to_nii(vert_ref, seg=True).rescale_and_reorient_(axcodes_to=axcodes_to, voxel_spacing=(1, 1, 1))
    subreg_nii = to_nii(sub_ref, seg=True).rescale_and_reorient_(axcodes_to=axcodes_to, voxel_spacing=(1, 1, 1))

    vert_mask_arr = vert_nii.get_seg_array()
    subreg_mask_arr = subreg_nii.get_seg_array()

    # make npz cutouts
    subregion_cutouts = {}
    for vert_idx in tqdm(vert_idx_list, disable=(not verbose)):
        subreg_arr_cut, _ = make_3d_bbox_cutout_npz(
            vert_mask_arr=vert_mask_arr,
            subreg_mask_arr=subreg_mask_arr,
            bids_file=sub_ref,
            # out_filename=out_filename,
            vert_idx=vert_idx,
            verbose=verbose,
        )
        # TODO add bbox x1,y1,z1 to mesh so they fit in world coord
        if subreg_arr_cut is not None:
            subreg_arr_cut[subreg_arr_cut < 0] = 0
            subregion_cutouts[v_idx2name[vert_idx]] = subreg_arr_cut

    # Make 3d models
    models3d = make_3d_models_from_masks(subregion_cutouts, verbose=verbose)
    models3d = dict(sorted(models3d.items(), key=lambda k: order_vert_name((v_name2idx[k[0]], 0), v_idx_order=v_idx_order)))

    if len(list(models3d.keys())) == 0:
        print(f"did not get any models, will skip {vert_ref.file}")
        return

    # Save 3d models
    models3d_paths = {}
    out_path = None
    if save_individual_models:
        for vert_, model_ in models3d.items():
            out_path = sub_ref.get_changed_path(file_type="ply", format="model", info={"label": vert_}, additional_folder="models")
            if override or not os.path.exists(out_path):
                o3d.io.write_triangle_mesh(filename=str(out_path), mesh=model_)
            else:
                print(f"Found model in {out_path}")
            models3d_paths[vert_] = out_path
        assert os.path.exists(out_path)  # check if something was created.
    else:
        # convert models
        for vert_, model_ in models3d.items():
            models3d_paths[vert_] = open3d_2_pvyista_mesh(model_)
    # print(f"Saved individual models into {out_path}")
    if save_combined_model:
        out_path = sub_ref.get_changed_path(file_type="ply", format="model", info={"label": f"{from_v}-{to_v}"})
        if override or not os.path.exists(out_path):
            combined_mesh = combine_meshes_along_y(list(models3d.values()), y_distance=60)
            o3d.io.write_triangle_mesh(filename=str(out_path), mesh=combined_mesh)
            print(f"Saved combined model into {out_path}")
        else:
            print(f"Combined model already exists in {out_path}, will skip")

    # Make and save snapshots of the 3d models
    screenshot = {}

    # print(models3d_paths)
    for vert_, f_ in models3d_paths.items():
        screenshot[vert_], _, _ = snapshot_single_mesh(
            f_, vert_, bids_file=sub_ref, save_individual_views=save_individual_snapshots, zoom=0.9
        )
    # Combine screenshot of the individual combined vertebra shots
    shot_combined = np.concatenate(list(screenshot.values()), axis=0)
    im = Image.fromarray(shot_combined)
    im.save(snapshot_path)
    print(f"Saved combined snapshot into {snapshot_path}")


def make_combined_3d_model_and_snapshot(
    vert_ref: BIDS_FILE,
    vert_idx_list: list[int],
    override: bool = False,
    save_model_snap: bool = False,
    verbose: bool = False,
):
    """
    Args:
        vert_ref: BIDS file with the mask content
        vert_idx_list: the labels that should be kept in
        override: if true, will override existing files instead of skipping
        verbose:

    Returns:

    """
    axcodes_to = ("R", "P", "I")
    from_v = v_idx2name[vert_idx_list[0]]
    to_v = v_idx2name[vert_idx_list[-1]]

    out_snap_path = vert_ref.get_changed_path(file_type="png", format="snp", info={"label": f"{from_v}-{to_v}"})
    out_npz_path = vert_ref.get_changed_path(
        file_type="npz", format="msk", info={"label": f"{from_v}-{to_v}"}, additional_folder="vert_cutout_npz"
    )
    out_model_path = vert_ref.get_changed_path(file_type="ply", format="model", info={"label": f"{from_v}-{to_v}"})
    # stop if snapshot already exists and override is not set
    if os.path.exists(out_model_path) and not override and verbose:
        print(f"Combined model already exists in {out_model_path}, will skip")
        return

    vert_nii = to_nii(vert_ref, seg=True).rescale_and_reorient_(axcodes_to=axcodes_to, voxel_spacing=(1, 1, 1))

    vert_mask_arr = vert_nii.get_seg_array()
    # make everything to zero that is not on vert_idxs_list
    vert_mask_arr_extracted = np.zeros(vert_mask_arr.shape, dtype=int)
    for i in vert_idx_list:
        count = np.count_nonzero(vert_mask_arr_extracted)
        vert_mask_arr_extracted[vert_mask_arr == i] = i
        if np.count_nonzero(vert_mask_arr_extracted) <= count and verbose:
            print(f"did not find {i} in the mask, will proceed")

    # make npz cutouts
    _, _, extracted_cutout = cut_single_vertebra.make_bounding_box_cutout(
        vert_mask_arr_extracted, vert_mask_arr_extracted, cutout_size=None
    )

    np.savez_compressed(out_npz_path, extracted_cutout)

    # Make 3d model
    mesh = mask_to_mesh(extracted_cutout, cmap=vert3d_color_dict)
    # Save 3d model
    save_mesh(mesh, save_dir=None, filename=out_model_path)

    if save_model_snap:
        assert os.path.exists(out_model_path) is not None
        # Make and save snapshots of the 3d models
        shot_combined, screen_view, meshcenter = snapshot_single_mesh(
            out_model_path,
            vert_name=None,
            bids_file=vert_ref,
            save_individual_views=True,
            zoom=0.75,
        )
        im = Image.fromarray(shot_combined)
        im.save(out_snap_path)
        print(f"Saved combined snapshot into {out_snap_path}")


def make_3d_models_from_masks(
    masks: dict[str, np.ndarray],
    verbose: bool = False,
):
    models3d: dict = {}
    for vert_key, msk in masks.items():
        mesh = mask_to_mesh(msk)
        if vert_key in models3d:
            warnings.warn(
                f"Found two vertebra with the same vert_key {vert_key} in the same container in {models3d.keys()}",
                UserWarning,
            )
        models3d[vert_key] = mesh
    return models3d


def make_3d_bbox_cutout_npz(
    vert_mask_arr: np.ndarray,
    subreg_mask_arr: np.ndarray,
    bids_file: BIDS_FILE,
    vert_idx: int,
    cutout_size=(128, 128, 80),
    verbose: bool = False,
) -> tuple[np.ndarray | None, str]:
    """makes a 3d bounding box cutout and saves the array as an .npz

    Args:
        vert_mask_arr: vertebra binary mask
        subreg_mask_arr: corresponding subregion mask
        bids_file: BIDS_FILE for path reference
        vert_idx: vertebra index (19: T12 for example)
        cutout_size: size of the cutout (default is 128,128,80 which is good)
        verbose:

    Returns:
        subreg_mask_arr_cut (None if error)
    """
    vert_mask_arr_ = vert_mask_arr.copy()
    vert_mask_arr_[vert_mask_arr_ != int(vert_idx)] = 0
    vert_mask_arr_[vert_mask_arr_ == int(vert_idx)] = 1
    if vert_idx not in v_idx2name:
        print(f"skipped {vert_idx} because it is not a correct vert_idx / name") if verbose else None
        return None, None
    if np.count_nonzero(vert_mask_arr_) == 0:
        print(f"skipped {vert_idx} as it does not exist in this ct sample, only got {np.unique(vert_mask_arr)}") if verbose else None
        return None, None

    vert_name = v_idx2name[vert_idx]

    vert_mask_arr_cut, subreg_mask_arr_cut, mask_subreg_combi_cut = cut_single_vertebra.make_bounding_box_cutout(
        vert_mask_arr_, subreg_mask_arr, cutout_size=cutout_size
    )
    out_path = bids_file.get_changed_path(file_type="npz", format="msk", info={"label": vert_name}, additional_folder="vert_cutout_npz")
    np.savez_compressed(out_path, mask_subreg_combi_cut)
    return mask_subreg_combi_cut, out_path


def mask_to_mesh(
    mask_image: np.ndarray | torch.Tensor,
    verbose: bool = False,
    binary_image: bool = False,
    cmap: tuple[RGB_Color, ...] | dict[int, RGB_Color] = subreg3d_color_dict,
):
    """Converts a numpy array or torch tensor to a mesh with marching cubes

    Args:
        mask_image: input (each individual value inside represents a different color), must contain the zero!
        verbose:
        binary_image: if true, converts the input to a zero/non-zero binary image
        cmap: colormap, either a tuple of rgb colors or a dictionary mapping the input unique values to the corresponding colors

    Returns:

    """
    if isinstance(mask_image, np.ndarray):
        if mask_image.dtype != np.int16:
            mask_image = mask_image.astype(np.int16)
        mask_image = torch.from_numpy(mask_image)
    max_v = torch.max(mask_image)
    print("input min/max", torch.min(mask_image), max_v) if verbose else None
    assert torch.min(mask_image) == 0, f"min value of image is not zero, got {torch.min(mask_image)}"
    # assert 41 <= torch.max(mask_image) <= 50, f"max value of image not between 41 and 50, got {torch.max(mask_image)}"

    input_values = torch.unique(mask_image).numpy()
    if isinstance(cmap, tuple):
        # convert to dictionary
        assert len(cmap) >= len(input_values), f"more input values than given colors, got {input_values}, and cmap {cmap}"
        cmap_dict: dict = {v: cmap[idx] for idx, v in enumerate(input_values)}
    else:
        cmap_dict = cmap
    assert isinstance(cmap_dict, dict), f"cmap not a dictionary, got {cmap_dict}"

    # convert to continous array
    cmap_list = []
    mapped_image = torch.zeros_like(mask_image, dtype=torch.int32)
    for idx, v in enumerate(input_values):
        if v == 0:
            continue
        cmap_list.append(cmap_dict[v])
        mapped_image[mask_image == v] = idx

    mask_image = mapped_image.to(torch.int32)
    # mask_image[mask_image < 41] = 0
    # mask_image[mask_image >= 41] -= 40
    if binary_image:
        mask_image[mask_image != 0] = 1
        mask_image[mask_image != 1] = 0
    print(f"converted input to range {torch.min(mask_image)}, {torch.max(mask_image)}") if verbose else None

    mesh = np_mask_to_mcubes_mesh(mask_image.numpy(), cmap_list=tuple(cmap_list))
    return mesh
