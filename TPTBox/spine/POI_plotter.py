import json
import os
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import sys

sys.path.append(str(Path(__file__).parent.parent))
from TPTBox import (
    Centroids,
    NII,
    POI_Reference,
    Image_Reference,
    to_nii_seg,
)
import numpy as np
from TPTBox.core.poi import POI
from TPTBox.snapshot3D.cut_single_vertebra import calculate_cbbox_cutout_coords, make_3d_cutout
import pyvista as pv
from tqdm import tqdm

"""
This function reads the POI's saved in the "poi_space-aligASL_msk.json" file in the derivatives 
folder and plots them together with the vertebrae.

Author: Kati, Hendrik
"""

# Defined manually
conversion_poi = {
    # 81:"SSL", #this POI is not included in our POI list
    109: "ALL_CR_S",
    101: "ALL_CR",
    117: "ALL_CR_D",
    111: "ALL_CA_S",
    103: "ALL_CA",
    119: "ALL_CA_D",
    110: "PLL_CR_S",
    102: "PLL_CR",
    118: "PLL_CR_D",
    112: "PLL_CA_S",
    104: "PLL_CA",
    120: "PLL_CA_D",
    # 149:"FL_CR_S", #not calculated
    125: "FL_CR",
    # 141:"FL_CR_D", not calculated
    # 151:"FL_CA_S",
    127: "FL_CA",
    # 143:"FL_CA_D",
    # 134:"ISL_CR",
    # 136:"ISL_CA"
}


def plot_POIs_(path_to_derivatives, subject_number):  # k=subject to plot | k= index of folder in the derivatives folder
    # Get a list of all files in the derivatives folder
    """at the moment, there was no need to loop through
    the sessions, because there was only one in each subject.
    This can be added later if needed."""
    subject_dir = os.listdir(path_to_derivatives)

    # loop through the subjects (eg. WS_00)
    for i, subject in enumerate(subject_dir):
        POIs = None

        print(subject)
        # Find Json file
        files = list(Path(path_to_derivatives).joinpath(subject).rglob("*seg-poi_*source-deterministic_*poi*.json"))
        print(files)
        if len(files) > 0:
            f = files[0]
            POIs = json.load(str(f))

        if POIs is None:
            continue
        # Iterate through all vertebrae
        vert_lst = []
        i = 0
        while i < len(POIs):
            current_vert = POIs[i]["vert_label"]

            poi_lst = []
            # Loop through POIs
            for poi_ind in conversion_poi.keys():
                # If the POI is not in the list, it will be skipped
                try:
                    # read POI
                    current_POI = list(map(int, POIs[i][str(poi_ind)][1:-1].split(",")))
                    poi_lst.append(current_POI)

                except:
                    print("POI with label {} could not be found in the POI list for vert {}".format(poi_ind, i + 1))

            # saving all information for the last vert in the list of verts
            vert_lst.append(poi_lst)
            i = i + 1
        # print(vert_lst)
        # somehow return this

    # visualize POIs in a plot
    p = pv.Plotter()
    p.set_background("black", top=None)
    p.add_axes()

    all_verts = []
    # choose a subject
    vert_dir = os.path.join(path_to_derivatives, subject_dir[subject_number], "vertebrae")

    # Plot vertebrae of the chosen subject
    for stl_file in os.listdir(vert_dir):
        if stl_file.endswith("msk.stl"):
            file_path = os.path.join(vert_dir, stl_file)
            mesh = pv.read(file_path)
            all_verts.append(mesh)  # create a list with all verts
            p.add_mesh(mesh)  # add the mesh of the current vert to the plotter

    # for i in all_verts:
    #    p.add_mesh(i) # add all vertebrae of subject k=2 to the plot

    # iterate through the vertebrae to add the respective POIs and LMs to the plot
    for l, vert in enumerate(vert_lst):
        print("current vert is {}".format(POIs[l]["vert_label"]))
        # iterate through the POIs and LMs and add them to the plotter
        for j in range(0, len(vert)):
            p.add_points(np.asarray(vert[j]), render_points_as_spheres=True, point_size=10, color="orange")

    p.show()
    return ()


def visualize_pois(
    ctd_in: POI_Reference,
    seg_vert: Image_Reference,
    vert_idx_list: list[int],
    cmap: matplotlib.colors.ListedColormap = None,
    save_path: Path | str | None = None,
):
    """Visualizes a given POIs on top of a segmentation image

    Args:
        ctd: Centroid reference containing the POIs
        seg_vert: Segmentation Mask
        vert_idx_list: list of vertebra indices to plot
        cmap: ListedColormap vor the segmentation. If None, uses pyvistas default cmap

    Returns:
        None (shows the visualization)
    """
    ctd = Centroids.load(ctd_in)
    seg = to_nii_seg(seg_vert).reorient_(verbose=True)
    seg_labels = seg.unique()
    ctd.reorient_(verbose=True)

    poi_vert: dict[int, dict] = {l: {} for l in seg_labels}
    for p_id, v_id, poi in ctd.items():
        if v_id in poi_vert:
            poi_vert[v_id][p_id] = poi

    # visualize POIs in a plot
    poi_coords = []

    p = pv.Plotter()
    p.set_background("black", top=None)
    p.add_axes()
    for vert_id in tqdm(vert_idx_list):
        if vert_id in seg_labels:
            vert_mesh = make_vert_mesh(seg, vert_id)
            p.add_mesh(vert_mesh, opacity=0.95, cmap=cmap)
            for p_id, coord in poi_vert[vert_id].items():
                poi_coords.append(coord)
    n = pv.PolyData(poi_coords)
    n["radius"] = np.ones(shape=len(poi_coords)) * 5
    geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
    glyphed = n.glyph(scale="radius", geom=geom, progress_bar=False, orient=False)
    p.add_mesh(glyphed, color="red")

    if save_path is not None:
        p.export_obj(save_path)

    p.show()


def make_vert_mesh(seg: NII, vert_id: int):
    seg_arr = seg.reorient_().get_seg_array()
    vert_arr = seg_arr.copy()
    vert_arr[seg_arr != vert_id] = 0

    bbox, _ = calculate_cbbox_cutout_coords(vert_arr, cutout_size=None, cutout_margin=1.1)
    (x1, x2, y1, y2, z1, z2) = bbox

    vert_arr = make_3d_cutout(vert_arr, bbox)
    vert_verts, vert_faces, vert_normals, vert_values = marching_cubes(vert_arr, gradient_direction="ascent", step_size=1)
    vert_verts += (x1, y1, z1)  # so it has correct global coordinates

    vfaces = np.column_stack(
        (
            np.ones(
                len(vert_faces),
            )
            * 3,
            vert_faces,
        )
    ).astype(int)

    mesh = pv.PolyData(vert_verts, vfaces)
    mesh["Normals"] = vert_normals
    mesh["values"] = vert_values
    return mesh


if __name__ == "__main__":
    # plot_POIs_bids("/home/data/hendrik/spine_poi_endplate/dataset-poi-bids/")
    files = "D:/data/translated/derivatives/sub-101131/T2w"
    POIs = POI.load(Path(files, "sub-101131_acq-iso_chunk-LWS_sequ-31_mod-T2w_seg-vert_space-aligASL_poi.json"))

    p = pv.Plotter()
    p.set_background("black", top=None)  # type: ignore
    p.add_axes()  # type: ignore

    all_verts = []
    # choose a subject
    vert_dir = os.path.join(files, "vertebrae")

    # Plot vertebrae of the chosen subject
    for stl_file in os.listdir(vert_dir):
        if stl_file.endswith("msk.stl"):
            file_path = os.path.join(vert_dir, stl_file)
            mesh = pv.read(file_path)
            all_verts.append(mesh)  # create a list with all verts
            p.add_mesh(mesh, opacity=0.7)  # add the mesh of the current vert to the plotter
    vert_dir = os.path.join(files, "discs")

    # Plot vertebrae of the chosen subject
    for stl_file in os.listdir(vert_dir):
        if stl_file.endswith("msk.stl"):
            file_path = os.path.join(vert_dir, stl_file)
            mesh = pv.read(file_path)
            all_verts.append(mesh)  # create a list with all verts
            p.add_mesh(mesh, color="red", opacity=0.2)  # add the mesh of the current vert to the plotter

    # iterate through the vertebrae to add the respective POIs and LMs to the plot
    for region, subregion, cords in POIs.items():
        p.add_points(np.asarray(cords), render_points_as_spheres=True, point_size=10, color="orange")

    p.show()
