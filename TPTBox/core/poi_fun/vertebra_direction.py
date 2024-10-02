from collections.abc import Sequence

import numpy as np
from numpy.linalg import norm

from TPTBox import NII, POI, Print_Logger, calc_poi_from_subreg_vert
from TPTBox.core.poi_fun._help import make_spine_plot, sacrum_w_o_direction
from TPTBox.core.vert_constants import DIRECTIONS, Location, _plane_dict, never_called

Vertebra_Orientation = tuple[np.ndarray, np.ndarray, np.ndarray]
_log = Print_Logger()
#### Directions ####


#### Vertebra Direction ###
def calc_orientation_of_vertebra_PIR(
    poi: POI | None,
    vert: NII,
    subreg: NII,
    spline_subreg_point_id=Location.Vertebra_Corpus,
    source_subreg_point_id=Location.Vertebra_Corpus,
    subreg_id=Location.Spinal_Canal,
    do_fill_back: bool = False,
    spine_plot_path: None | str = None,
    save_normals_in_info=False,
) -> tuple[POI, NII | None]:
    """Calculate the orientation of vertebrae using PIR (Posterior, Inferior, Right) DIRECTIONS.

    Args:
        poi (POI | None): Point of interest. If None, computed from `vert` and `subreg`.
        vert (NII): Vertebra (full).
        subreg (NII): Subregion (full).
        spline_subreg_point_id (Location, optional): Subregion point ID for spline computation. Defaults to Location.Vertebra_Corpus.
        source_subreg_point_id (Location, optional): Source subregion point ID. Defaults to Location.Vertebra_Corpus.
        subreg_id (Location, optional): Subregion ID. Defaults to Location.Spinal_Canal.
        do_fill_back (bool, optional): Whether to fill back. Defaults to False.
        spine_plot_path (None | str, optional): Path to spine plot. Defaults to None.
        save_normals_in_info (bool, optional): Whether to save normals in info. Defaults to False.

    Returns:
        Tuple[POI, NII | None]: Point of interest and filled back NII.
    """
    assert poi is None or poi.zoom is not None
    from TPTBox import calc_centroids

    # Step 1 compute the up direction
    # check if label 50 is already computed in POI
    if poi is None or spline_subreg_point_id.value not in poi.keys_subregion():
        poi = calc_poi_from_subreg_vert(vert, subreg, extend_to=poi, subreg_id=spline_subreg_point_id)
    # compute Spline in ISO space
    poi_iso = poi.rescale().reorient()
    body_spline, body_spline_der = poi_iso.fit_spline(location=spline_subreg_point_id, vertebra=True)
    # Step 2 compute the back direction by spinal channel or arcus
    intersection_target = [Location.Spinosus_Process, Location.Arcus_Vertebrae]
    # We compute everything in iso space
    subreg_iso = subreg.rescale().reorient()

    target_labels = subreg_iso.extract_label(intersection_target).get_array()
    # we want to see more of the Spinosus_Process and Arcus_Vertebrae than we cut with the plane. Should reduce randomness.
    # The ideal solution would be to make a projection onto the plane. Instead we fill values that have a vertical distanc of 10 mm up and down. This approximates the projection on to the plane.
    # Without this we have the chance to miss most of the arcus and spinosus, witch leads to instability in the direction.
    # TODO this will fail if the vertebra is not roughly aligned with S/I-direction
    for _ in range(15):
        target_labels[:, :-1] += target_labels[:, 1:]
        target_labels[:, 1:] += target_labels[:, :-1]
    target_labels = np.clip(target_labels, 0, 1)
    out = target_labels * 0
    fill_back_nii = subreg_iso.copy() if do_fill_back else None
    fill_back = out.copy() if do_fill_back else None
    down_vector: dict[int, np.ndarray] = {}
    # Draw a plain with the up_vector an cut it with intersection_target
    for reg_label, _, cords in poi_iso.extract_subregion(source_subreg_point_id).items():
        # calculate_normal_vector
        distances = np.sqrt(np.sum((body_spline - np.array(cords)) ** 2, -1))
        normal_vector_post = body_spline_der[np.argmin(distances)]
        normal_vector_post /= np.linalg.norm(normal_vector_post)
        down_vector[reg_label] = normal_vector_post.copy()
        # create_plane_coords
        # The main axis will be treated differently
        idx = [_plane_dict[i] for i in subreg_iso.orientation]
        axis = idx.index(_plane_dict["S"])
        # assert axis == np.argmax(np.abs(normal_vector)).item()
        dims = [0, 1, 2]
        dims.remove(axis)
        dim1, dim2 = dims
        # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
        start_point_np = np.array(cords)
        start_point_np[axis] = start_point_np[axis]
        shift_total = -start_point_np.dot(normal_vector_post)
        xx, yy = np.meshgrid(range(subreg_iso.shape[dim1]), range(subreg_iso.shape[dim2]))  # type: ignore
        zz = (-normal_vector_post[dim1] * xx - normal_vector_post[dim2] * yy - shift_total) * 1.0 / normal_vector_post[axis]
        z_max = subreg_iso.shape[axis] - 1
        zz[zz < 0] = 0
        zz[zz > z_max] = 0
        plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3])
        plane_coords[:, :, axis] = zz
        plane_coords[:, :, dim1] = xx
        plane_coords[:, :, dim2] = yy
        plane_coords = plane_coords.astype(int)
        # create_subregion
        # 1 where the selected subreg is, else 0
        select = subreg_iso.get_array() * 0
        select[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]] = 1
        out[out == 0] += (target_labels * select * reg_label)[out == 0]

        if fill_back is not None:
            fill_back[np.logical_and(select == 1, fill_back == 0)] = reg_label
    if fill_back is not None and fill_back_nii is not None:
        subreg_sar = subreg_iso.set_array(fill_back).reorient(("S", "A", "R"))
        fill_back = subreg_sar.get_array()
        x_slice = np.ones_like(fill_back[0]) * np.max(fill_back) + 1
        for i in range(fill_back.shape[0]):
            curr_slice = fill_back[i]
            cond = np.where(curr_slice != 0)
            x_slice[cond] = np.minimum(curr_slice[cond], x_slice[cond])
            fill_back[i] = x_slice
        subreg_sar.set_array(fill_back).reorient(poi.orientation).rescale_(poi.zoom)
        arr = subreg_sar.get_array()
        fill_back_nii.set_array_(arr)

    ret = calc_centroids(
        subreg_iso.set_array(out),
        subreg_id=subreg_id,
        extend_to=poi_iso.copy(),
        inplace=True,
    )

    poi._vert_orientation_pir = {}
    if save_normals_in_info:
        poi.info["vert_orientation_PIR"] = poi._vert_orientation_pir
    # calc posterior vector and the crossproduct
    for vert_id, normal_down in down_vector.items():
        try:
            # get two points and compute the direction:
            a = np.array(ret[vert_id : subreg_id.value]) - 1
            b = np.array(ret[vert_id : source_subreg_point_id.value]) - 1
            normal_vector_post = a - b
            normal_vector_post = normal_vector_post / norm(normal_vector_post)
            poi._vert_orientation_pir[vert_id] = (
                normal_vector_post,
                normal_down,
                np.cross(normal_vector_post, normal_down),
            )

            ### MAKE DIRECTIONS POIs ###
            ret[vert_id, Location.Vertebra_Direction_Posterior] = tuple(ret[vert_id, source_subreg_point_id] + normal_vector_post * 10)
            ret[vert_id, Location.Vertebra_Direction_Inferior] = tuple(ret[vert_id, source_subreg_point_id] + normal_down * 10)
            ret[vert_id, Location.Vertebra_Direction_Right] = tuple(
                ret[vert_id:source_subreg_point_id] + np.cross(normal_vector_post, normal_down * 10)
            )
        except KeyError as e:
            if vert_id not in sacrum_w_o_direction:
                _log.on_fail(f"calc_orientation_of_vertebra_PIR {vert_id=} - KeyError=", e)

    # if make_thicker:
    ret.remove(*ret.extract_subregion(subreg_id).keys())

    if spine_plot_path is not None:
        make_spine_plot(ret, body_spline, vert, spine_plot_path)

    ret = ret.resample_from_to(poi)  # type: ignore

    return poi.join_right_(ret), fill_back_nii


def _get_sub_array_by_direction(d: DIRECTIONS, cords: np.ndarray) -> np.ndarray:
    """Get the sub-array of coordinates along a specified direction.
    cords must be in PIR direction
    Returns:
        np.ndarray: Sub-array of coordinates along the specified direction.

    Raises:
        ValueError: If an invalid direction is provided.
    Note:
        Assumes the input `cords` array has shape (3, n), where n is the number of coordinates.
    """
    if d == "P":
        return cords[0]
    elif d == "A":
        return -cords[0]
    elif d == "I":
        return cords[1]
    elif d == "S":
        return -cords[1]
    elif d == "R":
        return cords[2]
    elif d == "L":
        return -cords[2]
    else:
        never_called(d)


def get_direction(d: DIRECTIONS, poi: POI, vert_id: int) -> np.ndarray:
    """Get the sub-array of coordinates along a specified direction.
    cords must be in PIR direction
    Returns:
        np.ndarray: Sub-array of coordinates along the specified direction.

    Raises:
        ValueError: If an invalid direction is provided.
    Note:
        Assumes the input `cords` array has shape (3, n), where n is the number of coordinates.
    """
    P, I, R = get_vert_direction_PIR(poi, vert_id, to_pir=False)  # noqa: N806
    if d == "P":
        return P
    elif d == "A":
        return -P
    elif d == "I":
        return I
    elif d == "S":
        return -I
    elif d == "R":
        return R
    elif d == "L":
        return -R
    else:
        never_called(d)


def get_vert_direction_PIR(poi: POI, vert_id, do_norm=True, to_pir=True) -> Vertebra_Orientation:
    """Retive the vertebra orientation from the POI. Must be computed by calc_orientation_of_vertebra_PIR first."""
    if vert_id in poi._vert_orientation_pir and to_pir:
        return poi._vert_orientation_pir[vert_id]  # Elusive buffer of iso/PIR directions.
    poi = poi.extract_subregion(
        Location.Vertebra_Corpus,
        Location.Vertebra_Direction_Posterior,
        Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Direction_Right,
    )
    if to_pir:
        poi = poi.rescale(verbose=False).reorient(verbose=False)

    def n(x):
        return x / norm(x) if do_norm else x

    center = np.array(poi[vert_id : Location.Vertebra_Corpus])
    post = np.array(poi[vert_id : Location.Vertebra_Direction_Posterior])
    down = np.array(poi[vert_id : Location.Vertebra_Direction_Inferior])
    right = np.array(poi[vert_id : Location.Vertebra_Direction_Right])
    out = n(post - center), n(down - center), n(right - center)
    if to_pir:
        poi._vert_orientation_pir[vert_id] = out

    return out


def get_vert_direction_matrix(poi: POI, vert_id: int, to_pir=False):
    P, I, R = get_vert_direction_PIR(poi, vert_id=vert_id, to_pir=to_pir)  # noqa: N806
    from_vert_orient = np.stack([P, I, R], axis=1)
    to_vert_orient = np.linalg.inv(from_vert_orient)
    return to_vert_orient, from_vert_orient


def calc_center_spinal_cord(
    poi: POI,
    subreg: NII,
    spline_subreg_point_id: Location | list[Location] = Location.Vertebra_Corpus,
    source_subreg_point_id: Location = Location.Vertebra_Corpus,
    subreg_id=Location.Spinal_Canal,
    intersection_target: list[Location] | None = None,
    _fill_inplace: NII | None = None,
    add_dense=False,
) -> POI:
    """
    Calculate the center of the spinal cord within a specified region.

    Parameters:
    - poi (POI): Point of Interest object containing relevant data.
    - subreg (NII): Neuroimaging subregion data for analysis.
    - spline_subreg_point_id (int, optional): The height of the region to analyze (default is Location.Vertebra_Corpus).
    - subreg_id (int, optional): The identifier for the subregion (default is Location.Spinal_Canal).
    - intersection_target (List[Location], optional): A list of locations for intersection analysis (default is [Location.Spinal_Cord, Location.Spinal_Canal]).

    Returns:
    - POI: Point of Interest object with calculated centroids.

    This function calculates the center of the spinal cord within a specified region using a spline fit to the
    given Point of Interest (POI) data and a NII. It first extracts a subregion of
    interest based on the provided spline_subreg_point_id and then calculates the center using geometric operations.

    Note:
    - The `poi` object should contain the relevant data for spline fitting.
    - The `subreg` object should be a NII subregion containing spine with number 60 and 61.
    - The `subreg_id` parameter is an optional identifier for the subregion.
    - The `intersection_target` parameter allows you to specify the locations to intersect (e.g., spinal cord and spinal canal).

    Example usage:
    ```python
    poi = POI(...)
    subreg = NII(...)
    updated_poi = calc_center_spinal_cord(
        poi,
        subreg,
        spline_subreg_point_id=Location.Vertebra_Corpus,
        subreg_id=Location.Spinal_Canal,
        intersection_target=[Location.Spinal_Cord, Location.Spinal_Canal],
    )
    ```
    """
    from TPTBox import calc_centroids

    if intersection_target is None:
        intersection_target = [Location.Spinal_Cord, Location.Spinal_Canal]
    assert _fill_inplace is None or subreg == _fill_inplace
    poi_iso = poi.rescale()
    if add_dense and (2, Location.Dens_axis) in poi_iso:
        sx = spline_subreg_point_id[0] if isinstance(spline_subreg_point_id, Sequence) else spline_subreg_point_id
        poi_iso[1, sx] = poi_iso[2, Location.Dens_axis]
        poi_iso[1, sx] = poi_iso[2, Location.Dens_axis]
    subreg_iso = subreg.rescale()
    body_spline, body_spline_der = poi_iso.fit_spline(location=spline_subreg_point_id, vertebra=True)
    target_labels = subreg_iso.extract_label(intersection_target).get_array()
    out = target_labels * 0
    fill_back = out.copy() if _fill_inplace is not None else None
    for reg_label, _, cords in poi_iso.extract_subregion(source_subreg_point_id).items():
        # calculate_normal_vector
        distances = np.sqrt(np.sum((body_spline - np.array(cords)) ** 2, -1))
        normal_vector = body_spline_der[np.argmin(distances)]
        normal_vector /= np.linalg.norm(normal_vector)
        # create_plane_coords
        # The main axis will be treated differently
        idx = [_plane_dict[i] for i in subreg_iso.orientation]
        axis = idx.index(_plane_dict["S"])
        # assert axis == np.argmax(np.abs(normal_vector)).item()
        dims = [0, 1, 2]
        dims.remove(axis)
        dim1, dim2 = dims
        # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
        start_point_np = np.array(cords)
        start_point_np[axis] = start_point_np[axis]
        shift_total = -start_point_np.dot(normal_vector)
        xx, yy = np.meshgrid(range(subreg_iso.shape[dim1]), range(subreg_iso.shape[dim2]))  # type: ignore
        zz = (-normal_vector[dim1] * xx - normal_vector[dim2] * yy - shift_total) * 1.0 / normal_vector[axis]
        z_max = subreg_iso.shape[axis] - 1
        zz[zz < 0] = 0
        zz[zz > z_max] = 0
        plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3])
        plane_coords[:, :, axis] = zz
        plane_coords[:, :, dim1] = xx
        plane_coords[:, :, dim2] = yy
        plane_coords = plane_coords.astype(int)
        # create_subregion
        # 1 where the selected subreg is, else 0
        select = subreg_iso.get_array() * 0
        select[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]] = 1
        out += target_labels * select * reg_label

        if fill_back is not None:
            fill_back[np.logical_and(select == 1, fill_back == 0)] = reg_label
    if fill_back is not None:
        assert _fill_inplace is not None
        subreg_iso = subreg_iso.set_array(fill_back).reorient(("S", "A", "R"))
        fill_back = subreg_iso.get_array()
        x_slice = np.ones_like(fill_back[0]) * np.max(fill_back) + 1
        for i in range(fill_back.shape[0]):
            curr_slice = fill_back[i]
            cond = np.where(curr_slice != 0)
            x_slice[cond] = np.minimum(curr_slice[cond], x_slice[cond])
            fill_back[i] = x_slice

        arr = subreg_iso.set_array(fill_back).reorient(poi.orientation).rescale_(poi.zoom).get_array()
        _fill_inplace.set_array_(arr)
    ret = calc_centroids(
        subreg_iso.set_array(out),
        subreg_id=subreg_id,
        extend_to=poi_iso.extract_subregion(subreg_id),
        inplace=True,
    )
    ret.rescale_(poi.zoom)
    return poi.join_left_(ret)
