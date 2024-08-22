import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle, FancyArrow
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.signal import savgol_filter

from TPTBox import (
    NII,
    POI,
    Image_Reference,
    Location,
    POI_Reference,
    calc_centroids,
    load_centroids,
    to_nii,
    to_nii_optional,
    v_idx2name,
    v_idx_order,
)
from TPTBox.mesh3D.mesh_colors import _color_map_in_row  # vert_color_map

NII.suppress_dtype_change_printout_in_set_array(True)
"""
Author: Maximilian Löffler, modifications: Jan Kirschke, Malek El Husseini, Robert Graf, Hendrik Möller
Create snapshots without 3d resampling. Images are scaled to isotropic pixels
before displaying. Estimate two 1d-splines through vertebral centroids and
create sagittal and coronal projection images.
include subregions to re-calculate vb centroids and create coronal images to verify last ribs
include special views for fracture rating and virtual DXA and QCT evaluations
"""


class Visualization_Type(Enum):
    Slice = auto()
    Maximum_Intensity = auto()
    Maximum_Intensity_Colored_Depth = auto()
    Mean_Intensity = auto()


#####################
# ITK-snap colormap #
# extra mappings: [255,255,255], #0 clear label;
colors_itk = (1 / 255) * np.array(
    [
        [167, 151, 255],
        [189, 143, 248],
        [95, 74, 171],
        [165, 114, 253],
        [78, 54, 158],
        [129, 56, 255],
        [56, 5, 149],  # c1-7
        [119, 194, 244],
        [67, 120, 185],
        [117, 176, 217],
        [69, 112, 158],
        [86, 172, 226],
        [48, 80, 140],  # t1-6
        [17, 150, 221],
        [14, 70, 181],
        [29, 123, 199],
        [11, 53, 144],
        [60, 125, 221],
        [16, 29, 126],  # t7-12
        [4, 159, 176],
        [106, 222, 235],
        [3, 126, 140],
        [10, 216, 239],
        [10, 75, 81],
        [108, 152, 158],  # L1-6
        [203, 160, 95],
        [149, 106, 59],
        [43, 95, 199],
        [57, 76, 76],
        [0, 128, 128],
        [188, 143, 143],
        [255, 105, 180],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 239, 213],  # 29-39 unused
        [0, 0, 205],
        [205, 133, 63],
        [210, 180, 140],
        [102, 205, 170],
        [0, 0, 128],
        [0, 139, 139],
        [46, 139, 87],
        [255, 228, 225],
        [106, 90, 205],
        [221, 160, 221],
        [233, 150, 122],  # Label 40-50 (subregions)
        [255, 250, 250],
        [147, 112, 219],
        [218, 112, 214],
        [75, 0, 130],
        [255, 182, 193],
        [60, 179, 113],
        [255, 235, 205],
        [255, 105, 180],
        [165, 42, 42],
        [188, 143, 143],
        [255, 235, 205],
        [255, 228, 196],
        [218, 165, 32],
        [0, 128, 128],  # rest unused
    ]
)

cm_itk = ListedColormap(_color_map_in_row / 255.0)  # type: ignore
cm_itk.set_bad(color="w", alpha=0)  # set NaN to full opacity for overlay
# define HU windows
wdw_sbone = Normalize(vmin=-500, vmax=1300, clip=True)
wdw_hbone = Normalize(vmin=-200, vmax=1000, clip=False)

LABEL_MAX = 256


def sag_cor_curve_projection(
    ctd_list: POI,
    img_data: np.ndarray,
    ctd_fallback: POI,
    cor_savgol_filter: bool = False,
    curve_location: Location = Location.Vertebra_Corpus,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Makes a curve projection (spline interpolation) over the spline through the given centroids

    Args:
        ctd_list: given Centroids
        img_data: given img_data
        cor_savgol_filter: If true, will perform the savgol filter also in coronal view
        curve_location: Location of the curve's centroids to be used.

    Returns:
        x_ctd: ctd x values sorted (nparray), y_cord: interpolated y coords (nparray), z_cord: interpolated z coords (nparray)
    """
    assert ctd_list is not None

    if 26 in ctd_list.centroids.keys_region() and cor_savgol_filter:
        warnings.warn(
            "Sacrum centroid present with cor_savgol_filter might overshadow the sacrum in coronal view",
            UserWarning,
            stacklevel=4,
        )
    # Sagittal and coronal projections of a curved plane defined by centroids
    # Note: Will assume IPL orientation!
    # if x-direction (=S/I) is not fully incremental, a straight, not an interpolated plane will be returned
    order = v_idx_order
    order += [i for i in range(256) if i not in v_idx_order]
    # ctd_list.sorting_list = v_idx_order
    ctd_list.round_(3)

    ctd_list.sort()
    if curve_location == Location.Vertebra_Corpus:
        l = [v for k1, k2, v in ctd_list.items() if k2 in (0, 50, 40)]
    else:
        l = [v for k1, k2, v in ctd_list.items() if k2 == curve_location.value]

    # TODO make the selected subregion flexible.
    if len(l) <= 3:
        l = list(ctd_list.values())
    if len(l) <= 3:
        l = list(ctd_fallback.values())
    # throw out all centroids that are not in correct up-down order
    x_cur = l[0][0] - 1
    throw_idx = []
    for idx, c in enumerate(l.copy()):
        if c[0] > x_cur:
            x_cur = c[0]
        else:
            throw_idx.append(idx)
    l = [i for idx, i in enumerate(l) if idx not in throw_idx] if len(l) - len(throw_idx) >= 3 else sorted(l, key=lambda x: x[0])

    ctd_arr = np.transpose(np.asarray(l))
    shp = img_data.shape
    x_ctd = np.rint(ctd_arr[0]).astype(int)
    y_ctd = np.rint(ctd_arr[1]).astype(int)
    z_ctd = np.rint(ctd_arr[2]).astype(int)
    # axl_plane = np.zeros((shp[1], shp[2]))
    try:
        f_sag = interp1d(x_ctd, y_ctd, kind="quadratic")
        f_cor = interp1d(x_ctd, z_ctd, kind="quadratic")
    except Exception:
        f_sag = interp1d(x_ctd, y_ctd, kind="linear")
        f_cor = interp1d(x_ctd, z_ctd, kind="linear")
        # print("quadratic", l, x_ctd, len(l), len(throw_idx))
        # exit()
    window_size = int((max(x_ctd) - min(x_ctd)) / 2)
    poly_order = 3
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(window_size, poly_order + 1)
    y_cord = np.array([np.rint(f_sag(x)).astype(int) for x in range(min(x_ctd), max(x_ctd))])
    y_cord = np.rint(savgol_filter(y_cord, window_size, poly_order)).astype(int) if cor_savgol_filter else y_cord

    z_cord = np.array([np.rint(f_cor(x)).astype(int) for x in range(min(x_ctd), max(x_ctd))])
    z_cord = np.rint(savgol_filter(z_cord, window_size, poly_order)).astype(int)

    y_cord[y_cord < 0] = 0  # handle out-of-volume interpolations
    y_cord[y_cord >= shp[1]] = shp[1] - 1
    z_cord[z_cord < 0] = 0
    z_cord[z_cord >= shp[2]] = shp[2] - 1
    return x_ctd, y_cord, z_cord


def curve_projected_slice(x_ctd, img_data, y_cord, z_cord, axial_heights):
    shp = img_data.shape
    cor_plane = np.zeros((shp[0], shp[2]))
    sag_plane = np.zeros((shp[0], shp[1]))
    for x in range(shp[0] - 1):
        if x < min(x_ctd):
            cor_plane[x, :] = img_data[x, y_cord[0], :]
            sag_plane[x, :] = img_data[x, :, z_cord[0]]
        elif x >= max(x_ctd):
            cor_plane[x, :] = img_data[x, y_cord[-1], :]
            sag_plane[x, :] = img_data[x, :, z_cord[-1]]
        else:
            cor_plane[x, :] = img_data[x, y_cord[x - min(x_ctd)], :]
            sag_plane[x, :] = img_data[x, :, z_cord[x - min(x_ctd)]]
    return (
        sag_plane,
        cor_plane,
        curve_projection_axial_fallback(img_data, x_ctd, heights=axial_heights),
    )


def curve_projected_mean(
    img_data: np.ndarray,
    zms: tuple[float, float, float],
    x_ctd,
    y_cord,
    ctd_list,
    thick_t: tuple[int, int] = (100, 300),
    axial_heights=None,
):
    shp = img_data.shape
    cor_plane = np.zeros((shp[0], shp[2]))
    sag_plane = np.zeros((shp[0], shp[1]))
    y_zoom = zms[1]  # 0.9 = 1px = 0.9 mm # 10cm = 112px
    thick = (*thick_t,)

    for x in range(shp[0] - 1):
        if x < min(x_ctd):  # higher
            y_ref = y_cord[0]
        elif x >= max(x_ctd):  # lower than sacrum
            y_ref = y_cord[-1]
        else:
            y_ref = y_cord[x - min(x_ctd)]

        if 23 in ctd_list and x > int(ctd_list[23][1]):
            thick = (100, 50)

        thick = [int(i // y_zoom) + int(i % y_zoom > 0) for i in thick]
        y_post_rel_to_border = y_ref + int(0.4 * (shp[1] - 1 - y_ref))  # one-third distance to border
        y_range_low = int(max(0, y_ref - thick[1]))  # sagittal left
        y_range_high = int(min(y_ref + thick[0], y_post_rel_to_border))  # sagittal right
        cor_cut = img_data[x, y_range_low:y_range_high, :]

        plane_bool = np.zeros_like(cor_cut).astype(bool)
        plane_bool[cor_cut > 0] = True
        sag = np.nansum(img_data[x, :, :], 1, where=img_data[x, :, :] > 0)
        sag_plane[x, :] = div0(sag, np.count_nonzero(img_data[x, :, :], 1), fill=0)
        cor = np.nansum(cor_cut, 0, where=plane_bool)
        cor_plane[x, :] = div0(cor, np.count_nonzero(plane_bool, 0), fill=0)
    return (
        sag_plane,
        cor_plane,
        curve_projection_axial_fallback(img_data, x_ctd, heights=axial_heights),
    )


def curve_projected_mip(
    img_data: np.ndarray,
    zms: tuple[float, float, float],
    x_ctd,
    y_cord,
    ctd_list,  # noqa: ARG001
    thick_t: tuple[int, int] = (100, 300),
    make_colored_depth: bool = False,
    axial_heights=None,
):
    shp = img_data.shape
    cor_plane = np.zeros((shp[0], shp[2]))
    cor_depth_plane = np.zeros((shp[0], shp[2]))
    sag_plane = np.zeros((shp[0], shp[1]))
    sag_depth_plane = np.zeros((shp[0], shp[1]))
    y_zoom = zms[1]  # 0.9 = 1px = 0.9 mm # 10cm = 112px
    thick = (*thick_t,)

    for x in range(shp[0] - 1):
        if x < min(x_ctd):  # higher
            y_ref = y_cord[0]
        elif x >= max(x_ctd):  # lower than sacrum
            y_ref = y_cord[-1]
        else:
            y_ref = y_cord[x - min(x_ctd)]

        # if 23 in ctd_list and x > int(ctd_list[23][1]) and not make_colored_depth:
        #    thick = (100, 50)

        # TODO set y_zoom for broken sample, see if it works
        try:
            thicke = [int(i // y_zoom) + int(i % y_zoom > 0) for i in thick]
        except Exception:
            print("thick infinity bug", y_zoom, thick_t, thick)
            thicke = (*thick_t,)
        thick = thicke
        y_post_rel_to_border = y_ref + int(0.4 * (shp[1] - 1 - y_ref))  # one-third distance to border
        y_range_low = int(max(0, y_ref - thick[1]))  # sagittal left
        y_range_high = int(min(y_ref + thick[0], y_post_rel_to_border))  # sagittal right
        # print("range", y_range_low, y_range_high)
        cor_cut = img_data[x, y_range_low:y_range_high, :]

        cor_plane[x, :] = np.max(cor_cut, axis=0)  # arr[x, mip_i, :]
        cor_depth_plane[x, :] = np.argmax(cor_cut, axis=0)
        sag_plane[x, :] = np.max(img_data[x, :, :], axis=1)  # img_data[x, :, z_ref]
        sag_depth_plane[x, :] = np.argmax(img_data[x, :, :], axis=1)

    if make_colored_depth:
        cor_depth_plane = normalize_image(cor_depth_plane)
        sag_depth_plane = normalize_image(sag_depth_plane)

        cor_plane = normalize_image(cor_plane)
        sag_plane = normalize_image(sag_plane)

        cor_m_plane = np.sqrt(cor_plane) * cor_depth_plane  # sqrt
        sag_m_plane = np.sqrt(sag_plane) * sag_depth_plane
        cor_m_plane = normalize_image(cor_m_plane)
        sag_m_plane = normalize_image(sag_m_plane)

        # convert to color image
        cmap = plt.get_cmap("inferno")
        cor_plane = cmap(cor_m_plane)[..., :3]
        sag_plane = cmap(sag_m_plane)[..., :3]

    return (
        sag_plane,
        cor_plane,
        curve_projection_axial_fallback(img_data, x_ctd, heights=axial_heights),
    )


def normalize_image(img, v_range: tuple[float, float] | None = None):
    if v_range is None:
        min_v = np.min(img)
        max_v = np.max(img)
    else:
        min_v = v_range[0]
        max_v = v_range[1]
    return (img - min_v) / (max_v - min_v)


def curve_projection_axial_fallback(img_data, x_ctd, heights: list[float] | None):
    if heights is not None:
        heights = [int(abs(h if abs(h) >= 1 else img_data.shape[0] * h)) for h in (heights) if abs(h) <= img_data.shape[0]]
        axl_plane = np.concatenate([img_data[h, :, :] for h in heights], axis=0)
        return axl_plane
    else:
        # Axial
        center = x_ctd[len(x_ctd) // 2]
        center_up = x_ctd[max(0, len(x_ctd) // 2 - 1)]
        center_down = x_ctd[min(len(x_ctd) - 1, len(x_ctd) // 2 + 1)]
        try:
            axl_plane = np.concatenate(
                [
                    img_data[(center + center_up) // 2, :, :],
                    img_data[center, :, :],
                    img_data[(center + center_down) // 2, :, :],
                ],
                axis=0,
            )
        except Exception as e:
            print(e)
            axl_plane = np.zeros((1, 1))
        return axl_plane


def make_isotropic2d(arr2d: np.ndarray, zms2d, msk=False) -> np.ndarray:
    if arr2d.dtype in (np.float64, np.float16, np.float32):
        arr2d = arr2d.astype(int)
    xs = list(range(arr2d.shape[0]))
    ys = list(range(arr2d.shape[1]))
    if msk:
        interpolator = RegularGridInterpolator((xs, ys), arr2d, method="nearest")
    else:
        interpolator = RegularGridInterpolator((xs, ys), arr2d, method="linear")
    new_shp = tuple(np.rint(np.multiply(arr2d.shape, zms2d)).astype(int))
    x_mm = np.linspace(0, arr2d.shape[0] - 1, num=new_shp[0])
    y_mm = np.linspace(0, arr2d.shape[1] - 1, num=new_shp[1])
    xx, yy = np.meshgrid(x_mm, y_mm)
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    try:
        lt = interpolator(pts)
    except Exception:
        raise ValueError(f"Needs to be casted into a other type: arr2d {arr2d.dtype}")  # noqa: B904
    img = np.reshape(lt, new_shp, order="F")
    return img


def make_isotropic2dpluscolor(arr3d, zms2d, msk=False):
    # print(arr3d.shape)
    if arr3d.ndim == 2:
        return make_isotropic2d(arr3d, zms2d, msk=msk)

    r_img = make_isotropic2d(arr3d[:, :, 0], zms2d, msk=msk)
    # print(r_img.shape)
    g_img = make_isotropic2d(arr3d[:, :, 1], zms2d, msk=msk)
    b_img = make_isotropic2d(arr3d[:, :, 2], zms2d, msk=msk)
    img = np.stack([r_img, g_img, b_img], axis=-1)
    # print(img.shape)
    return img


def create_figure(dpi, planes: list, has_title=True):
    fig_h = round(2 * planes[0].shape[0] / dpi, 2) + (0.5 if has_title else 0)
    plane_w = [p.shape[1] for p in planes]
    w = sum(plane_w)
    fig_w = round(2 * w / dpi, 2)
    x_pos = [0]
    for x in plane_w[:-1]:
        x_pos.append(x_pos[-1] + x)
    fig, axs = plt.subplots(1, len(planes), figsize=(fig_w, fig_h))

    if not isinstance(axs, np.ndarray):
        axs: list[Axes] = [axs]  # type: ignore
    for a in axs:
        a.axis("off")
        idx = list(axs).index(a)
        a.set_position([(x_pos[idx] / w), (-0.03 if has_title else 0), plane_w[idx] / w, 1])
    return fig, axs


def plot_sag_centroids(
    axs: Axes,
    ctd: POI,
    zms,
    poi_labelmap: dict[int, str],
    hide_centroid_labels: bool,
    cmap: ListedColormap = cm_itk,
    curve_location: Location = Location.Vertebra_Corpus,
    show_these_subreg_poi=None,
):
    # requires v_dict = dictionary of mask labels
    ctd2 = ctd
    if show_these_subreg_poi is not None:
        ctd2 = ctd.extract_subregion(*show_these_subreg_poi)
    for k1, k2, v in ctd2.items():
        # print(k, v, (v[1] * zms[1], v[0] * zms[0]), zms)
        try:
            axs.add_patch(Circle((v[1] * zms[1], v[0] * zms[0]), 2, color=cmap((k1 - 1) % LABEL_MAX % cmap.N)))
            if not hide_centroid_labels and k2 == curve_location.value and k1 in poi_labelmap:
                axs.text(4, v[0] * zms[0], poi_labelmap[k1], fontdict={"color": cmap(k1 - 1), "weight": "bold"})
        except Exception as e:
            print(e)
    if "line_segments_sag" in ctd.info:
        for color, x, (c, d) in ctd.info["line_segments_sag"]:
            if len(x) == 2:
                v = ctd[x]
            elif len(x) == 4:
                v = (np.array(ctd[x[0], x[1]]) + np.array(ctd[x[2], x[3]])) / 2

            axs.add_patch(FancyArrow(v[1] * zms[1], v[0] * zms[0], c, d, color=cmap(color - 1 % LABEL_MAX % cmap.N)))
    if "text_sag" in ctd.info:
        for color, x in ctd.info["text_sag"]:
            if not isinstance(color, int) and len(color) == 2:
                color, curve_location = color  # noqa: PLW2901
            if isinstance(x, str) or len(x) == 1:
                (text) = x
                a = zms[1] * ctd[color, curve_location][1]
                b = zms[0] * ctd[color, curve_location][0]
            elif len(x) == 3:
                (text, a, b) = x
            elif len(x) == 2:
                (text, a) = x
                b = zms[0] * ctd[color, curve_location][0]
            axs.text(a, b, text, fontdict={"color": cmap(color - 1), "weight": "bold"})


def plot_cor_centroids(
    axs,
    ctd: POI,
    zms,
    poi_labelmap: dict[int, str],
    hide_centroid_labels: bool,
    cmap: ListedColormap = cm_itk,
    curve_location: Location = Location.Vertebra_Corpus,
    show_these_subreg_poi=None,
):
    ctd2 = ctd
    if show_these_subreg_poi is not None:
        ctd2 = ctd.extract_subregion(*show_these_subreg_poi)

    # requires v_dict = dictionary of mask labels
    for k1, k2, v in ctd2.items():
        try:
            axs.add_patch(
                Circle(
                    (v[2] * zms[2], v[0] * zms[0]),
                    2,
                    color=cmap((k1 - 1) % LABEL_MAX % cmap.N),
                )
            )
            if not hide_centroid_labels and k2 == curve_location.value and k1 in poi_labelmap:
                axs.text(4, v[0] * zms[0], poi_labelmap[k1], fontdict={"color": cmap(k1 - 1), "weight": "bold"})
        except Exception:
            pass
    if "line_segments_cor" in ctd.info:
        for color, x, (c, d) in ctd.info["line_segments_cor"]:
            if len(x) == 2:
                v = ctd[x]
            elif len(x) == 4:
                v = (np.array(ctd[x[0], x[1]]) + np.array(ctd[x[2], x[3]])) / 2
            axs.add_patch(FancyArrow(v[2] * zms[2], v[0] * zms[0], c, d, color=cmap(color - 1 % LABEL_MAX % cmap.N)))
    if "text_cor" in ctd.info:
        for color, x in ctd.info["text_cor"]:
            if isinstance(color, Sequence) and len(color) == 2:
                color, curve_location = color  # noqa: PLW2901
            if isinstance(x, str) or len(x) == 1:
                (text) = x
                a = zms[2] * ctd[color, curve_location][2]
                b = zms[0] * ctd[color, curve_location][0]
            elif len(x) == 3:
                (text, a, b) = x
            elif len(x) == 2:
                (text, a) = x
                b = zms[0] * ctd[color, curve_location][0]
            axs.text(a, b, text, fontdict={"color": cmap(color - 1), "weight": "bold"})


def make_2d_slice(
    img: Image_Reference,
    ctd: POI,
    zms: tuple[float, float, float],
    msk: bool,
    visualization_type: Visualization_Type,
    ctd_fallback: POI,
    cor_savgol_filter: bool = False,
    to_ax=("I", "P", "L"),
    curve_location: Location = Location.Vertebra_Corpus,
    rescale_to_iso: bool = True,
    axial_heights=None,
):
    img_nii = to_nii(img)
    img_reo = img_nii.reorient_(to_ax)
    ctd_reo = ctd.reorient_centroids_to(img_reo)
    img_data = img_reo.get_array()

    if visualization_type in [
        visualization_type.Slice,
        visualization_type.Maximum_Intensity,
        visualization_type.Maximum_Intensity_Colored_Depth,
        visualization_type.Mean_Intensity,
    ]:
        # Make interpolated curve
        x_ctd, y_cord, z_cord = sag_cor_curve_projection(
            ctd_reo,
            img_data=img_data,
            ctd_fallback=ctd_fallback,
            cor_savgol_filter=cor_savgol_filter,
            curve_location=curve_location,
        )
        # Calculate snapshot data values depending on visualization type
        if visualization_type == Visualization_Type.Slice:
            sag, cor, axl = curve_projected_slice(
                x_ctd=x_ctd,
                img_data=img_data,
                y_cord=y_cord,
                z_cord=z_cord,
                axial_heights=axial_heights,
            )
        elif visualization_type == Visualization_Type.Maximum_Intensity:
            sag, cor, axl = curve_projected_mip(
                img_data=img_data,
                zms=zms,
                x_ctd=x_ctd,
                y_cord=y_cord,
                ctd_list=ctd_reo,
                axial_heights=axial_heights,
            )
        elif visualization_type == Visualization_Type.Maximum_Intensity_Colored_Depth:
            sag, cor, axl = curve_projected_mip(
                img_data=img_data,
                zms=zms,
                x_ctd=x_ctd,
                y_cord=y_cord,
                ctd_list=ctd_reo,
                make_colored_depth=not msk,
                axial_heights=axial_heights,
            )
        # make isotropic
        elif visualization_type == Visualization_Type.Mean_Intensity:
            sag, cor, axl = curve_projected_mean(
                img_data=img_data,
                zms=zms,
                x_ctd=x_ctd,
                y_cord=y_cord,
                ctd_list=ctd_reo,
                axial_heights=axial_heights,
            )

    # elif visualization_type == visualization_type.Mean_Intensity:
    #    plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
    #    idx_view = {plane_dict[s]: i for i, s in enumerate(to_ax)}
    else:
        raise NotImplementedError(visualization_type)

    if rescale_to_iso:
        if sag.ndim == 2:
            sag = make_isotropic2d(sag, (zms[0], zms[1]), msk=msk)
            cor = make_isotropic2d(cor, (zms[0], zms[2]), msk=msk)
            axl = make_isotropic2d(axl, (zms[1], zms[2]), msk=msk)
        elif sag.ndim == 3:  # color also encoded
            sag = make_isotropic2dpluscolor(sag, (zms[0], zms[1]), msk=msk)
            cor = make_isotropic2dpluscolor(cor, (zms[0], zms[2]), msk=msk)
            axl = make_isotropic2dpluscolor(axl, (zms[1], zms[1]), msk=msk)
        else:
            raise ValueError(f"make_2d_slice: expected sag to be ndim 2 or 3, but got shape {sag.shape}")

    sag = sag.astype(float)
    cor = cor.astype(float)
    axl = axl.astype(float)

    if msk:
        sag[sag == 0] = np.nan
        cor[cor == 0] = np.nan
        axl[axl == 0] = np.nan
    return sag, cor, axl


def div0(a, b, fill=0):
    """a / b, divide by 0 -> `fill`"""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
    if np.isscalar(c):
        return c if np.isfinite(c) else fill
    else:
        c[~np.isfinite(c)] = fill
        return c


Image_Modes = Literal["CT", "MRI", "CTs", "MINMAX", "None"]


@dataclass(init=True)
class Snapshot_Frame:
    # Content
    image: Image_Reference
    segmentation: Image_Reference | None = None
    centroids: POI_Reference | None = None
    # Views
    sagittal: bool = True
    coronal: bool = False
    axial: bool = False

    axial_heights: list[float | int] | None = None
    # Title: str = for all views same, list entry for each view, None = no title
    title: str | list[str] | None = None
    # Image mode, cmap
    mode: Image_Modes = "MINMAX"
    cmap: ListedColormap | str = field(default_factory=lambda: cm_itk)
    alpha: float = 0.3
    # Pre-procesing
    crop_msk: bool = False
    crop_img: bool = False
    ignore_cdt_for_centering: bool = False
    rescale_to_iso: bool = True
    ignore_seg_for_centering: bool = False
    # Type, post-processing
    visualization_type: Visualization_Type = Visualization_Type.Slice
    only_mask_area: bool = False
    image_threshold: float | None = None  # everything below this threshold is set to min value of the img
    denoise_threshold: float | None = None  # threshold like above, but set for a smoothed img version
    gauss_filter: bool = False  # applies a gauss filter to the img
    cor_savgol_filter: bool = False  # applies a savgol_filter on the curve projection spline interpolation in coronal plane

    hide_segmentation: bool = False
    hide_centroids: bool = False
    hide_centroid_labels: bool = False
    poi_labelmap: dict[int, str] = field(default_factory=lambda: v_idx2name)
    force_show_cdt: bool = False  # Shows the centroid computed by a segmentation, if no centroids are provided
    curve_location: Location | None = None  # Location.Vertebra_Corpus
    show_these_subreg_poi: list[int | Location] | None = None


def to_cdt(ctd_bids: POI_Reference | None) -> POI | None:
    if ctd_bids is None:
        return None
    ctd = load_centroids(ctd_bids)
    if len(ctd) > 2:  # handle case if empty centroid file given
        return ctd
    print("[!][snp] To few centroids", ctd)
    return None


def create_snapshot(  # noqa: C901
    snp_path: str | Path | list[str | Path] | list[str] | list[Path],
    frames: list[Snapshot_Frame],
    crop=False,
    check=False,
    to_ax=("I", "P", "L"),
    dpi=96,
    verbose: bool = False,
):
    """Create virtual dx, sagittal, and coronal curved-planar CT snapshots with mask overlay

    Args:
        snp_path (str): Path to the new jpg
        frames (List[Snapshot_Frame]): List of Images
        crop (bool): crop output to vertebral masks (seg-vert). Defaults to False.
        check (bool): if true, check if snap is present and do not re-create. Defaults to False.
        to_ax (Orientation): Sets the Orientation. Can be used for flipping the image or fixing false rotations of the original inputs.
        dpi (int): Set the resolution.
    """

    # Checks if snaps already exists, does nothing if true and check is true
    exist = all(Path(i).is_file() for i in snp_path) if isinstance(snp_path, list) else Path(snp_path).is_file()
    if check and exist:
        return None

    img_list = []
    frame_list = []
    frames = [f for f in frames if f is not None]
    for frame in frames:
        # PRE-PROCESSING
        img = to_nii(frame.image)
        assert img is not None
        seg = to_nii_optional(frame.segmentation, seg=True)  # can be None
        ctd = copy.deepcopy(to_cdt(frame.centroids))
        if (crop or frame.crop_msk) and seg is not None:  # crop to segmentation
            ex_slice = seg.compute_crop()
            img = img.copy().apply_crop_(ex_slice)
            seg = seg.copy().apply_crop_(ex_slice)
            ctd = ctd.apply_crop(ex_slice).filter_points_inside_shape() if ctd is not None else None
        if frame.crop_img:  # crops image
            ex_slice = img.compute_crop(dist=0)
            img = img.apply_crop_(ex_slice)
            seg = seg.apply_crop_(ex_slice) if seg is not None else None
            ctd = ctd.apply_crop(ex_slice).filter_points_inside_shape() if ctd is not None else None
        img = img.reorient_(to_ax)
        seg = seg.reorient_(to_ax) if seg is not None else None
        assert not isinstance(seg, tuple), "seg is a tuple"
        if ctd is not None:
            ctd = ctd.reorient(img.orientation) if ctd.shape is not None else ctd.reorient_centroids_to(img)
            if ctd.zoom not in (img.zoom, None):
                ctd.rescale_(img.zoom)

        # Preprocessing img data
        img_data = img.get_array()
        if frame.only_mask_area:
            assert seg is not None, f"only_mask_area is set but segmentation is None, got {frame}"
            seg_mask = seg.get_seg_array()
            seg_mask[seg_mask != 0] = 1
            img_data = img_data * seg_mask

        if len(img_data.shape) == 4:
            img_data = img_data[:, :, :, 0]
        if frame.gauss_filter:
            img_data = ndimage.median_filter(img_data, size=3)
        if frame.image_threshold is not None:
            img_data[img_data < frame.image_threshold] = 0  # type: ignore
        if frame.denoise_threshold is not None:
            import torch
            from torch.nn.functional import avg_pool3d

            try:
                t_arr = torch.from_numpy(img_data.copy()).unsqueeze(0).to(torch.float32)
            except Exception:
                # can't handel uint16
                t_arr = torch.from_numpy(img_data.astype(np.int32).copy()).unsqueeze(0).to(torch.float32)

            img_data_smoothed = (
                avg_pool3d(
                    t_arr,
                    kernel_size=(3, 3, 3),
                    padding=1,
                    stride=1,
                )
                .squeeze(0)
                .numpy()
                .astype(np.int32)
            )
            img_data[img_data_smoothed <= frame.denoise_threshold] = 0
        img = img.set_array(img_data, verbose=False)
        # PRE-PROCESSING Done

        zms = img.zoom
        try:
            ctd_fallback = POI(
                centroids={
                    (1, 50): (0, 0, img.shape[-1] // 2),
                    (2, 50): [img.shape[0] - 1, img.shape[1] - 1, img.shape[2] // 2],
                },
                orientation=img.orientation,
            )
            if ctd is None and seg is None:
                ctd_tmp = ctd_fallback
            elif frame.ignore_seg_for_centering:
                ctd_tmp = ctd_fallback
                if frame.force_show_cdt:
                    ctd = calc_centroids(seg)
            elif frame.ignore_cdt_for_centering:
                assert seg is not None, f"ignore_cdt_for_centering requires segmentation, but got None, {frame}"
                ctd_tmp = calc_centroids(seg)  # TODO BUFFER
            elif ctd is None:
                ctd_tmp = calc_centroids(seg)  # TODO BUFFER
                if frame.force_show_cdt:
                    ctd = ctd_tmp
            else:
                ctd_tmp = ctd
        except Exception:
            print("did not manage to calc ctd_tmp\n", frame)
            raise
        if frame.curve_location is None:
            if frame.show_these_subreg_poi is not None:
                l = frame.show_these_subreg_poi[0]
                frame.curve_location = Location(l) if isinstance(l, int) else l
            else:
                frame.curve_location = Location.Vertebra_Corpus
        try:
            sag_img, cor_img, axl_img = make_2d_slice(
                img,
                ctd_tmp,
                zms,
                msk=False,
                visualization_type=frame.visualization_type,
                ctd_fallback=ctd_fallback,
                to_ax=to_ax,
                cor_savgol_filter=frame.cor_savgol_filter,
                curve_location=frame.curve_location,
                rescale_to_iso=frame.rescale_to_iso,
                axial_heights=frame.axial_heights,
            )
            if seg is not None:
                sag_seg, cor_seg, axl_seg = make_2d_slice(
                    seg,
                    ctd_tmp,
                    zms,
                    msk=True,
                    visualization_type=frame.visualization_type,
                    ctd_fallback=ctd_fallback,
                    to_ax=to_ax,
                    cor_savgol_filter=frame.cor_savgol_filter,
                    curve_location=frame.curve_location,
                    rescale_to_iso=frame.rescale_to_iso,
                    axial_heights=frame.axial_heights,
                )
            else:
                sag_seg, cor_seg, axl_seg = (None, None, None)
        except Exception:
            print(frame)
            raise
        # Conversion to 2D image done, now normalization
        try:
            max_sag = np.percentile(sag_img[sag_img != 0], 99)
        except Exception:
            max_sag = 1
        try:
            max_cor = np.percentile(cor_img[cor_img != 0], 99)
        except Exception:
            max_cor = 1
        print("max sag/cor", max_sag, max_cor) if verbose else None
        ##MRT##
        if frame.mode == "MRI":
            max_intens = max(max_sag, max_cor)  # type: ignore
            wdw = Normalize(vmin=0, vmax=max_intens, clip=True)
        ##CT##
        elif frame.mode == "CT":
            wdw = wdw_hbone
        elif frame.mode == "CTs":
            wdw = wdw_sbone
        elif frame.mode == "MINMAX":
            max_intens = max(max_sag, max_cor)  # type: ignore
            min_intens = min(cor_img.min(), sag_img.min())
            wdw = Normalize(vmin=min_intens, vmax=max_intens, clip=True)
        elif frame.mode == "None":
            wdw = None
        else:
            raise ValueError(frame.mode + "is not implemented as a Normalize mode")
        alpha = frame.alpha
        # Colormap
        cmap = frame.cmap
        if isinstance(cmap, str):
            try:
                cmap = plt.get_cmap(str(cmap))
            except Exception:
                cmap = plt.get_cmap("viridis")
        # set segmentation to none if hide_segmentation
        if frame.hide_segmentation:
            sag_seg = None
            cor_seg = None
            axl_seg = None
        # set centroid to none if hide_centroids
        if frame.hide_centroids:
            ctd = None

        if not isinstance(frame.title, list):
            frame.title = [frame.title, frame.title, frame.title]

        if frame.sagittal:
            img_list.append(sag_img)
            frame_list.append(
                (
                    sag_img,
                    sag_seg,
                    ctd,
                    wdw,
                    True,
                    alpha,
                    cmap,
                    zms,
                    frame.curve_location,
                    frame.poi_labelmap,
                    frame.hide_centroid_labels,
                    frame.title[0],
                    frame,
                )
            )
        if frame.coronal:
            img_list.append(cor_img)
            frame_list.append(
                (
                    cor_img,
                    cor_seg,
                    ctd,
                    wdw,
                    False,
                    alpha,
                    cmap,
                    zms,
                    frame.curve_location,
                    frame.poi_labelmap,
                    frame.hide_centroid_labels,
                    frame.title[1],
                    frame,
                )
            )
        if frame.axial:
            img_list.append(axl_img)
            frame_list.append(
                (
                    axl_img,
                    axl_seg,
                    None,
                    wdw,
                    True,
                    alpha,
                    cmap,
                    zms,
                    frame.curve_location,
                    frame.poi_labelmap,
                    frame.hide_centroid_labels,
                    frame.title[2],
                    frame,
                )
            )

    fig, axs = create_figure(dpi, img_list, has_title=frame.title is None)
    for ax, (img, msk, ctd, wdw, is_sag, alpha, cmap, zms, curve_location, poi_labelmap, hide_centroid_labels, title, frame) in zip(
        axs, frame_list, strict=False
    ):
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 18, "color": "black"}, loc="center")
        if img.ndim == 3:
            ax.imshow(img, norm=wdw)  # type: ignore
        else:
            ax.imshow(img, cmap=plt.cm.gray, norm=wdw)  # type: ignore
        if msk is not None:
            ax.imshow(msk, cmap=cmap, alpha=alpha, vmin=1, vmax=cmap.N)
        frame: Snapshot_Frame
        if ctd is not None:
            if is_sag:
                plot_sag_centroids(
                    ax,
                    ctd,
                    zms,
                    poi_labelmap=poi_labelmap,
                    hide_centroid_labels=hide_centroid_labels,
                    cmap=cmap,
                    curve_location=curve_location,
                    show_these_subreg_poi=frame.show_these_subreg_poi,
                )
            else:
                plot_cor_centroids(
                    ax,
                    ctd,
                    zms,
                    poi_labelmap=poi_labelmap,
                    hide_centroid_labels=hide_centroid_labels,
                    cmap=cmap,
                    curve_location=curve_location,
                    show_these_subreg_poi=frame.show_these_subreg_poi,
                )

    if not isinstance(snp_path, list):
        snp_path = [str(snp_path)]
    for path in snp_path:
        fig.savefig(str(path))
        print("[*] Snapshot saved:", path) if verbose else None
    plt.close()
    return snp_path


if __name__ == "__main__":
    # run_on_reg("registration")
    # run_on_reg('registration2')
    # run_on_bailiang_reg()
    # run_on_bailiang_reg_NRad()
    pass
