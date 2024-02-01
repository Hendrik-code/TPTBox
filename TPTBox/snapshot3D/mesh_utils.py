from __future__ import annotations

import warnings

import pyvista
import torch
import numpy as np
from torchmcubes import marching_cubes
import open3d as o3d
import pyvista as pv
from PIL import Image
import platform
from enum import auto, Enum

from TPTBox.snapshot3D.cut_single_vertebra import bbox_nd
from TPTBox.core.bids_files import BIDS_FILE
from TPTBox.snapshot3D.mesh_colors import RGB_Color, subreg3d_color_dict


def np_mask_to_mcubes_mesh(mask_image: np.ndarray, cmap_list: tuple[RGB_Color, ...]) -> o3d.geometry.TriangleMesh:
    assert np.min(mask_image) == 0, f"min value of image is not zero, got {np.min(mask_image)}"
    assert np.max(mask_image) <= len(
        cmap_list
    ), f"max value of image > number of given colors, got {np.max(mask_image)}, and cmap {cmap_list}"
    mask_image = np.pad(mask_image.copy(), pad_width=1)

    mask_image = mask_image[::-1].copy()

    N = max(np.shape(mask_image))
    n_diff = (N - np.shape(mask_image)[2]) // 2
    shape = np.shape(mask_image)
    # print("mask shape", shape)
    mask_image = np.pad(mask_image, ((0, 0), (0, 0), (n_diff, n_diff)), "constant")
    mask_image = mask_image.astype(np.int32)
    shape = np.shape(mask_image)
    # print("mask shape", shape)
    values = np.float32(mask_image.copy())

    r_label = np.zeros(shape)
    b_label = np.zeros(shape)
    g_label = np.zeros(shape)

    for i in range(1, len(cmap_list) + 1):
        r_label[values == i] = cmap_list[i - 1][0]
        g_label[values == i] = cmap_list[i - 1][1]
        b_label[values == i] = cmap_list[i - 1][2]
    rgb_label = np.float32(np.stack([r_label, g_label, b_label], axis=0))
    # print(np.unique(rgb_label[0]).shape)

    u = torch.from_numpy(values)  # .cuda()
    rgb = torch.from_numpy(rgb_label)  # .cuda()
    verts, faces = marching_cubes(u, 0.5)
    colrs = grid_interp_py(rgb, verts)
    # print(np.unique(colrs[:, 0]))
    values, _ = torch.max(colrs, dim=0)
    # print("max values", values)
    # colrs /= 255# values
    # print(np.unique(colrs[:, 0]))
    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    faces = np.stack([faces[:, 2], faces[:, 1], faces[:, 0]], axis=1)
    colrs = colrs.cpu().numpy()

    # Use Open3D for visualization
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colrs)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()
    return dec_mesh


def combine_meshes(meshes: list, translation_vector: np.ndarray):
    combined_mesh = o3d.geometry.TriangleMesh()
    for idx, m in enumerate(meshes):
        local_translation = translation_vector.copy()
        local_translation[local_translation > 0] *= idx
        local_translation = tuple(local_translation)
        m = translate_mesh_pos(m, local_translation)
        # m = translate_mesh_pos(m, translation_vector=list(translation_vector))
        combined_mesh += m
    return combined_mesh


def combine_meshes_along_y(meshes: list, y_distance: int = 55):
    return combine_meshes(meshes=meshes, translation_vector=np.array([0, y_distance, 0]))


def translate_mesh_pos(mesh, translation_vector: tuple[int, int, int]):
    assert len(translation_vector) == 3, "translation vector not 3D"
    # print("translation_vector", translation_vector)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    # print(np.shape(verts))
    verts[:, 1] += translation_vector[0]  # x
    verts[:, 0] += translation_vector[1]  # y
    verts[:, 2] += translation_vector[2]  # z
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh


def show_mesh(mesh):
    o3d.visualization.draw(mesh, show_skybox=False, raw_mode=False, ibl_intensity=0)


def save_mesh(mesh, save_dir: str | None, filename: str) -> str:
    filename = str(filename)
    if save_dir is None:
        save_dir = ""
    if not filename.endswith(".ply"):
        filename += ".ply"
    if not save_dir.endswith("/"):
        save_dir += "/"

    save_path = save_dir + filename
    o3d.io.write_triangle_mesh(f"{save_path}", mesh)
    print(f"Saved marching cubes mesh to {save_path}")
    return save_path


def open3d_2_pvyista_mesh(mesh: o3d.geometry.TriangleMesh):
    v = np.asarray(mesh.vertices)
    f = np.array(mesh.triangles)
    f = np.c_[np.full(len(f), 3), f]
    return pv.PolyData(v, f)


class VIEW(Enum):  # .value to get name
    SIDE_LEFT = auto()
    SIDE_RIGHT = auto()
    BACK = auto()
    TOP = auto()
    DIAGONAL = auto()


def snapshot_single_mesh(
    model_mesh: str | pv.PolyData,
    vert_name: str,
    bids_file: BIDS_FILE | None,
    model_axcodes: str = None,
    test_show: bool = False,
    resolution_factor: float = 1.0,
    center_crop_factor: float = 0.05,
    move_model_to_center: bool = True,
    move_model_offset: tuple[int, int, int] | None = None,
    view_keys: tuple[VIEW, ...] = (VIEW.SIDE_LEFT, VIEW.BACK, VIEW.TOP, VIEW.DIAGONAL),
    zoom: float | bool = 1.0,
    save_individual_views: bool = False,
    add_axes: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, dict, tuple[float, float, float]]:
    """This handles the model as if it were in RPI, can only be reoriented if model_axcodes is not None

    Args:
        model_mesh: filepath to the .ply file
        vert_name: name of the vertebra (if none, does not influence the output filename)
        bids_file: used for naming convention (if None, does not save anything)
        model_axcodes: If None, acts as though the model is in RPI, otherwise, may be able to reorient (experimental!)
        test_show: instead of the production, will show an interactive window of the plot with some cubes in it (default False)
        resolution_factor: factor of the resulting snapshot resolution in relation to the input (default 1.0)
        center_crop_factor: how much the snapshot should be centercropped in relation to the snap dimensions (default 0.05)
        move_model_to_center: if true, will move the model to (0,0,0) before taking the snapshot
        move_model_offset: if true, will move the model by this coordinate vector before taking the snapshot (after moving to center if that is set to true)
        view_keys: Which VIEWs are taken as snapshot
        zoom: A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.
        #override_color_list: if not set, will use the colors that were embedded in the model. This override is a index -> color translation
        save_individual_views: whether the singular views are also saved (into its own folder)
        add_axes: if True, plots axis to each snap and subsnap
        verbose:

    Returns:
        shot_combined, screen_view, meshcenter
    """

    # if "Win" not in platform.system():
    #    pv.start_xvfb()  # does not work under windows
    snap_views = {k.name: view_dict[k.value] for k in view_keys}

    if bids_file is None:
        save_individual_views = False
    pv.global_theme.background = "black"
    pv.global_theme.axes.show = True
    pv.global_theme.edge_color = "white"
    pv.global_theme.interactive = True

    default_window_size = (2048, 1536)

    if isinstance(model_mesh, str):
        reader = pv.get_reader(model_mesh)
        mesh = reader.read()
    else:
        mesh = model_mesh
    meshcenter = mesh.center
    if move_model_to_center:
        mesh = mesh.translate([-1 * i for i in meshcenter], inplace=False)
    if move_model_offset is not None:
        mesh = mesh.translate(move_model_offset, inplace=False)

    if model_axcodes == "SRP":
        rotation_point = (0, 0, 0)
        mesh = mesh.rotate_x(90, point=rotation_point, inplace=False)
        mesh = mesh.rotate_z(90, point=rotation_point, inplace=False)
        mesh = mesh.rotate_y(180, point=rotation_point, inplace=False)

    rgb = np.float32(mesh.get_array("RGB"))
    rgb_idx = rgb.copy()

    mesh_colors = rgb.copy().astype(int)
    mesh_colors = np.unique(mesh_colors, axis=0)
    mesh_cmap = []
    for idx, color in enumerate(mesh_colors):
        rgb_idx = replace_rgb_with_idx(rgb_idx, rgb, color, idx)
        mesh_cmap.append(color)

    rgb = rgb_idx
    rgb = rgb.astype(int)
    cmap = pyvista_cmap_simple(tuple(mesh_cmap))
    # MAKE SNAPSHOTs
    screen_view = {}
    if test_show:
        plotter = pv.Plotter(off_screen=False, window_size=default_window_size, lighting="three lights")
        plotter.add_mesh(
            mesh, scalars=rgb, show_edges=False, show_scalar_bar=False, cmap=cmap, interpolate_before_map=False, split_sharp_edges=True
        )
        plotter.add_axes(line_width=10, labels_off=False)
        # plotter.add_axes_at_origin(line_width=10, labels_off=True)
        plotter.add_mesh(pv.PlatonicSolid("cube", radius=15.0, center=(0, 0, 0)))
        plotter.add_mesh(pv.PlatonicSolid("cube", radius=15.0, center=(0, 50, 0)))
        plotter.add_mesh(pv.PlatonicSolid("cube", radius=15.0, center=(50, 0, 0)))
        plotter.add_mesh(pv.PlatonicSolid("cube", radius=15.0, center=(0, 0, 50)))
        plotter.show()
        print(plotter.camera_position)
        return None, None
    ####
    for view, view_coords in snap_views.items():
        pls = pv.Plotter(off_screen=True, window_size=default_window_size, lighting="three lights")
        pls.disable_shadows()
        pls.add_mesh(
            mesh,
            scalars=rgb,
            show_edges=False,
            show_scalar_bar=False,
            cmap=cmap,
            line_width=2.0,
            lighting=True,
            interpolate_before_map=False,
            split_sharp_edges=True,
        )
        pls.add_axes(line_width=10, labels_off=False) if add_axes else None

        pls.camera_position = view_coords

        if isinstance(zoom, bool) and zoom == True:
            pls.camera.zoom("tight")
        else:
            pls.camera.zoom(zoom)

        scaled_window_size = tuple(int(i * resolution_factor) for i in default_window_size)
        if scaled_window_size != default_window_size and verbose:
            print(f"changed window_size from {default_window_size} to {scaled_window_size}")
        pls.window_size = scaled_window_size
        screenshot = pls.screenshot(return_img=True)
        pls.close()
        del pls
        screenshot[screenshot == [76, 76, 76]] = 0
        screenshot_shape = screenshot.shape
        cutout_shape = (
            0 + int(center_crop_factor * screenshot_shape[0]),
            int((1 - center_crop_factor) * screenshot_shape[0]),
            int(0 + center_crop_factor * screenshot_shape[1]),
            int((1 - center_crop_factor) * screenshot_shape[1]),
        )
        # print("cropped snap to shape ", cutout_shape) if verbose else None

        bbox = bbox_nd(screenshot)
        # size_t = (bbox[1] - bbox[0], bbox[3] - bbox[2])
        # if bbox[0] < cutout_shape[0] or bbox[1] > cutout_shape[1] or bbox[2] < cutout_shape[2] or bbox[3] > cutout_shape[3]:
        #    warnings.warn(f"{vert_name}: {view} bigger bounding box than cutout_size, got {bbox}, and {cutout_shape}", UserWarning)
        # print("bbox", bbox)
        screenshot_cut = screenshot[cutout_shape[0] : cutout_shape[1], cutout_shape[2] : cutout_shape[3]]

        screen_view[view] = screenshot_cut
        if save_individual_views:
            individual_path = (
                bids_file.get_changed_path(
                    file_type="png",
                    format="snp",
                    info={"label": vert_name, "view": view},
                    additional_folder="individual_snapshots",
                )
                if vert_name is not None
                else bids_file.get_changed_path(
                    file_type="png",
                    format="snp",
                    info={"view": view},
                    additional_folder="individual_snapshots",
                )
            )
            savepath = individual_path
            im = Image.fromarray(screenshot_cut)
            im.save(savepath)

    shot_combined = np.concatenate(list(screen_view.values()), axis=1)

    if save_individual_views:
        snapshot_path = (
            bids_file.get_changed_path(file_type="png", format="snp", info={"label": vert_name}, additional_folder="individual_snapshots")
            if vert_name is not None
            else bids_file.get_changed_path(file_type="png", format="snp", additional_folder="individual_snapshots")
        )
        im = Image.fromarray(shot_combined)
        im.save(snapshot_path)
        print(f"Saved individual {vert_name} snapshot into {snapshot_path}")
    return shot_combined, screen_view, meshcenter


def replace_rgb_with_idx(rgb_new: np.ndarray, rgb: np.ndarray, ref_rgb: tuple[int, int, int], value: int):
    rgb_new[(ref_rgb[0] == rgb[..., 0]) & (ref_rgb[1] == rgb[..., 1]) & (ref_rgb[2] == rgb[..., 2])] = value
    rgb_new[..., 1] = 0
    rgb_new[..., 2] = 0
    return rgb_new


def pyvista_cmap_simple(cmap: tuple[tuple[int, int, int], ...], override_n_colors: int = None):
    from matplotlib.colors import ListedColormap

    n_colors = len(cmap)
    if override_n_colors is not None:
        n_colors = override_n_colors

    new_colors = np.empty((n_colors, 4))
    for i in range(n_colors):
        c = cmap[i % n_colors] / 255.0
        c = np.append(c, 1.0)
        new_colors[i] = c
    return ListedColormap(new_colors)  # type: ignore


view_dict = {
    VIEW.SIDE_LEFT.value: [
        (-5.66313189074029, 4.3769202199100885, 154.09802828102033),
        (1.3427934623068811, 3.327437142632535, 9.933036032449149),
        (-0.9978199656699152, 0.04441299682767809, -0.04881395111312186),
    ],
    VIEW.SIDE_RIGHT.value: [
        (-8.455737568551855, 16.96383044915939, -160.719476619939),
        (0.7103801609171445, -0.36919549857587575, 12.826516431498801),
        (-0.9986006130754632, -0.011702158481889754, 0.0515739765014664),
    ],
    VIEW.BACK.value: [
        (-8.404802078494686, 151.78997554853584, -2.6599709832453198),
        (2.8890018025198185, 7.900259381746347, -1.271097284938488),
        (-0.9969180094344114, -0.07829504962204803, -0.004936362021934786),
    ],
    VIEW.TOP.value: [
        (-193.5487287138878, 7.475026909185702, -10.11699246675732),
        (17.37132313295161, 0.03302890502040112, 0.6671438417140994),
        (-0.03515418285712232, -0.9993796918070891, -0.0021011975670570636),
    ],
    VIEW.DIAGONAL.value: [
        (-72.85390695045928, 121.12885788465334, -93.13355209619309),
        (4.885243844936774, -3.910312442930268, 0.8052233074978894),
        (-0.893089557510105, -0.3987303764396772, 0.2083389765998634),
    ],
}


def grid_interp_py(vol: torch.Tensor, points: torch.Tensor):
    assert vol.ndim == 4, "vol tensor not 4D"
    assert points.ndim == 2, "points tensor not 2D"

    Nx = vol.shape[3]
    Ny = vol.shape[2]
    Nz = vol.shape[1]
    C = vol.shape[0]
    Np = points.shape[0]

    output = torch.zeros([Np, C], dtype=torch.float32, device=vol.device)

    for i in range(Np):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]

        ix = int(x)
        iy = int(y)
        iz = int(z)
        fx = x - ix
        fy = y - iy
        fz = z - iz

        x0 = max(0, min(ix, Nx - 1))
        x1 = max(0, min(ix + 1, Nx - 1))
        y0 = max(0, min(iy, Ny - 1))
        y1 = max(0, min(iy + 1, Ny - 1))
        z0 = max(0, min(iz, Nz - 1))
        z1 = max(0, min(iz + 1, Nz - 1))

        rgb_dict = {}
        for xi in (x0, x1):
            for yi in (y0, y1):
                for zi in (z0, z1):
                    rgb = (vol[0][zi][yi][xi].item(), vol[1][zi][yi][xi].item(), vol[2][zi][yi][xi].item())
                    if rgb not in rgb_dict:
                        rgb_dict[rgb] = 1
                    else:
                        rgb_dict[rgb] += 1
        del rgb_dict[(0.0, 0.0, 0.0)]

        r, g, b = 0, 0, 0
        count = 0
        rgb_max = max(rgb_dict, key=rgb_dict.get)  # type: ignore
        r = rgb_max[0]
        g = rgb_max[1]
        b = rgb_max[2]
        output[i][0] = r
        output[i][1] = g
        output[i][2] = b
    return output
