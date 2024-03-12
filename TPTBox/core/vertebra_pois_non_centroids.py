import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist

from TPTBox import NII, POI, Log_Type, Logger_Interface, Print_Logger, calc_poi_from_subreg_vert
from TPTBox.core.vert_constants import Directions, Location, never_called, plane_dict, vert_directions

_log = Print_Logger()
Vertebra_Orientation = tuple[np.ndarray, np.ndarray, np.ndarray]
all_poi_functions: dict[int, "Strategy_Pattern"] = {}
pois_computed_by_side_effect: dict[int, Location] = {}


def run_poi_pipeline(vert: NII, subreg: NII, poi_path: Path, logger: Logger_Interface = _log):
    poi = calc_poi_from_subreg_vert(vert, subreg, buffer_file=poi_path, save_buffer_file=True, subreg_id=list(Location), verbose=logger)
    poi.save(poi_path)


def _strategy_side_effect(*args, **qargs):  # noqa: ARG001
    pass


class Strategy_Pattern:
    """Implements the Strategy design pattern by encapsulating different strategies as callable objects.

    Args:
        target (Location): The target location for which this strategy is defined.
        strategy (Callable): The strategy function that implements the desired behavior.
        prerequisite (set[Location] | None, optional): A set of prerequisite locations that must be satisfied before applying this strategy. Defaults to None.
        **args: Additional keyword arguments to be passed to the strategy function.

    Attributes:
        target (Location): The target location for which this strategy is defined.
        args (dict): Additional keyword arguments to be passed to the strategy function.
        prerequisite (set[Location]): A set of prerequisite locations that must be satisfied before applying this strategy.
        strategy (Callable): The strategy function that implements the desired behavior.

    Note:
        The strategy function should accept the following arguments:
        - poi (POI): The point of interest.
        - current_subreg (NII): The current subregion.
        - vert_id (int): The vertex ID.
        - bb: The bounding box.
        - log (Logger_Interface, optional): The logger interface. Defaults to _log, which should be defined globally.

    Example:
        >>> def strategy_function(poi, current_subreg, location, log, vert_id, bb, **kwargs):
        ...     # Strategy implementation
        ...     pass
        >>> strategy = Strategy_Pattern(target_location, strategy_function, prerequisite={prerequisite_location}, additional_arg=value)
        >>> result = strategy(poi, current_subreg, vert_id, bb)
    """

    def __init__(self, target: Location, strategy: Callable, prerequisite: set[Location] | None = None, **args) -> None:
        self.target = target
        self.args = args
        if prerequisite is None:
            prerequisite = set()
        if "direction" in args.keys():
            prerequisite.add(Location.Vertebra_Direction_Inferior)
        for i in args.values():
            if isinstance(i, Location):
                prerequisite.add(i)
            elif isinstance(i, Sequence):
                for j in i:
                    if isinstance(j, Location):
                        prerequisite.add(j)
        self.prerequisite = prerequisite
        self.strategy = strategy
        all_poi_functions[target.value] = self

    def __call__(self, poi: POI, current_subreg: NII, vert_id: int, bb, log: Logger_Interface = _log):
        return self.strategy(poi=poi, current_subreg=current_subreg, location=self.target, log=log, vert_id=vert_id, bb=bb, **self.args)


class Strategy_Pattern_Side_Effect(Strategy_Pattern):
    def __init__(self, target: Location, prerequisite: Location, **args) -> None:
        super().__init__(target, _strategy_side_effect, {prerequisite}, **args)
        pois_computed_by_side_effect[target.value] = prerequisite


class Strategy_Computed_Before(Strategy_Pattern):
    def __init__(self, target: Location, *prerequisite: Location, **args) -> None:
        super().__init__(target, _strategy_side_effect, set(prerequisite), **args)


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
    """Calculate the orientation of vertebrae using PIR (Posterior, Inferior, Right) directions.

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
    subreg_ids = subreg.unique()
    make_thicker = False
    if Location.Spinal_Cord.value in subreg_ids:
        intersection_target = [Location.Spinal_Cord, Location.Spinal_Canal]
    else:
        intersection_target = [Location.Spinal_Cord, Location.Spinal_Canal, Location.Spinosus_Process, Location.Arcus_Vertebrae]
        make_thicker = True
    # We compute everything in iso space
    subreg_iso = subreg.rescale().reorient()

    target_labels = subreg_iso.extract_label(intersection_target).get_array()
    if make_thicker:
        # for CT (<=> no spinal cord) we want to see more of the Spinosus_Process and Arcus_Vertebrae than we cut with the plane. Should reduce randomness.
        # The ideal solution would be to make a projection onto the plane. Instead we fill values that have a vertical distanc of 10 mm up and down. This approximates the projection on to the plane.
        # Without this we have the chance to miss most of the arcus and spinosus, witch leads to instability in the direction.
        # TODO this will fail if the vertebra is not roughly aligned with S/I-direction
        for _ in range(10):
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
        idx = [plane_dict[i] for i in subreg_iso.orientation]
        axis = idx.index(plane_dict["S"])
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
        out += target_labels * select * reg_label
        if fill_back is not None:
            fill_back[select == 1] = reg_label
    if fill_back is not None and fill_back_nii is not None:
        subreg_sar = subreg_iso.set_array(fill_back).reorient(("S", "A", "R"))
        fill_back = subreg_sar.get_array()
        x_slice = np.ones_like(fill_back[0]) * np.max(fill_back) + 1
        for i in range(fill_back.shape[0]):
            curr_slice = fill_back[i]
            cond = np.where(curr_slice != 0)
            x_slice[cond] = np.minimum(curr_slice[cond], x_slice[cond])
            fill_back[i] = x_slice
        arr = subreg_sar.set_array(fill_back).reorient(poi.orientation).rescale_(poi.zoom).get_array()
        fill_back_nii.set_array_(arr)

    ret = calc_centroids(subreg_iso.set_array(out), subreg_id=subreg_id, extend_to=poi_iso.copy(), inplace=True)
    if save_normals_in_info:
        poi.info["vert_orientation_PIR"] = {}
    # calc posterior vector and the crossproduct
    for vert_id, normal_down in down_vector.items():
        # get two points and compute the direction:
        a = np.array(ret[vert_id : subreg_id.value])
        b = np.array(ret[vert_id : source_subreg_point_id.value])
        normal_vector_post = a - b
        normal_vector_post = normal_vector_post / norm(normal_vector_post)
        if save_normals_in_info:
            poi.info["vert_orientation_PIR"][vert_id] = (normal_vector_post, normal_down, np.cross(normal_vector_post, normal_down))

        ### MAKE directions POIs ###
        # print(ret[vert_id, source_subreg_point_id], normal_vector_post)
        ret[vert_id, Location.Vertebra_Direction_Posterior] = tuple(ret[vert_id, source_subreg_point_id] + normal_vector_post)
        ret[vert_id, Location.Vertebra_Direction_Inferior] = tuple(ret[vert_id, source_subreg_point_id] + normal_down)
        ret[vert_id, Location.Vertebra_Direction_Right] = tuple(
            ret[vert_id:source_subreg_point_id] + np.cross(normal_vector_post, normal_down)
        )

    if make_thicker:
        ret.remove_centroid(*ret.extract_subregion(subreg_id).keys())
    if spine_plot_path is not None:
        _make_spine_plot(ret, body_spline, vert, spine_plot_path)

    ret = ret.resample_from_to(poi)  # type: ignore
    return poi.join_right_(ret), fill_back_nii


def _make_spine_plot(pois: POI, body_spline, vert_nii: NII, filenames):
    from matplotlib import pyplot as plt

    pois = pois.reorient()
    vert_nii = vert_nii.reorient().rescale(pois.zoom)
    body_center_list = list(np.array(pois.values()))
    # fitting a curve to the centoids and getting it's first derivative
    plt.figure(figsize=[10, 10])
    plt.imshow(np.swapaxes(np.max(vert_nii.get_array(), axis=vert_nii.get_axis(direction="R")), 0, 1), cmap=plt.cm.gray)
    plt.plot(np.asarray(body_center_list)[:, 0], np.asarray(body_center_list)[:, 1])
    plt.plot(np.asarray(body_spline[:, 0]), np.asarray(body_spline[:, 1]), "-")
    plt.savefig(filenames)


##### Extreme Points ####
def _get_sub_array_by_direction(d: Directions, cords: np.ndarray) -> np.ndarray:
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


def _get_direction(d: Directions, poi: POI, vert_id: int) -> np.ndarray:
    """Get the sub-array of coordinates along a specified direction.
    cords must be in PIR direction
    Returns:
        np.ndarray: Sub-array of coordinates along the specified direction.

    Raises:
        ValueError: If an invalid direction is provided.
    Note:
        Assumes the input `cords` array has shape (3, n), where n is the number of coordinates.
    """
    P, I, R = get_vert_direction_PIR(poi, vert_id)  # noqa: N806
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


def get_extreme_point_by_vert_direction(poi: POI, region: NII, vert_id, direction: Sequence[Directions] | Directions = "I"):
    """
    Get the extreme point in a specified direction.

    Args:
        poi (POI): The chosen point of interest represented as an array.
        region (NII): An array containing the subregion mask.
        vert_id: The ID of the vertex.
        direction (Union[Sequence[Directions], Directions], optional): The direction(s) to search for the extreme point.
            Defaults to "I" (positive direction along the secondary axis).

    Note:
        Assumes `region` contains binary values indicating the presence of points.
        Uses `_get_sub_array_by_direction` internally.
    """
    direction_: Sequence[Directions] = direction if isinstance(direction, Sequence) else (direction,)  # type: ignore

    to_reference_frame, from_reference_frame = get_vert_direction_matrix(poi, vert_id=vert_id)
    pc = np.stack(np.where(region.get_array() == 1))
    cords = to_reference_frame @ pc  # 3,n; 3 = P,I,R of vert
    a = [_get_sub_array_by_direction(d, cords) for d in direction_]
    idx = np.argmax(sum(a))
    return pc[:, idx]


def get_vert_direction_PIR(poi: POI, vert_id, do_norm=True) -> Vertebra_Orientation:
    """Retive the vertebra orientation from the POI. Must be computed by calc_orientation_of_vertebra_PIR first."""
    center = np.array(poi[vert_id : Location.Vertebra_Corpus])
    post = np.array(poi[vert_id : Location.Vertebra_Direction_Posterior])
    down = np.array(poi[vert_id : Location.Vertebra_Direction_Inferior])
    right = np.array(poi[vert_id : Location.Vertebra_Direction_Right])

    def n(x):
        if do_norm:
            return x / norm(x)
        else:
            return x

    return n(post - center), n(down - center), n(right - center)


def get_vert_direction_matrix(poi: POI, vert_id: int):
    P, I, R = get_vert_direction_PIR(poi, vert_id=vert_id)  # noqa: N806
    from_vert_orient = np.stack([P, I, R], axis=1)
    to_vert_orient = np.linalg.inv(from_vert_orient)
    return to_vert_orient, from_vert_orient


def strategy_extreme_points(
    poi: POI,
    current_subreg: NII,
    location: Location,
    direction: Sequence[Directions] | Directions,
    vert_id: int,
    subreg_id: Location,
    bb,
    log=_log,
):
    """Strategy function to update extreme points of a point of interest based on direction.

    Args:
        poi (POI): The point of interest.
        current_subreg (NII): The current subregion.
        location (Location): The location to update in the point of interest.
        direction (Union[Sequence[Directions], Directions]): Direction(s) to search for the extreme point.
        vert_id (int): The vertex ID.
        subreg_id (Location): The subregion ID.
        bb: The bounding box.
        log (Logger_Interface, optional): The logger interface. Defaults to _log.
    """
    region = current_subreg.extract_label(subreg_id)
    if region.sum() == 0:
        log.print(f"reg={vert_id},subreg={subreg_id} is missing (extreme_points)", ltype=Log_Type.FAIL)
        return
    # extreme_point = get_extreme_point(poi, region, vert_id, bb, anti_point)

    extreme_point = get_extreme_point_by_vert_direction(poi, region, vert_id, direction)
    if extreme_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip(bb, extreme_point, strict=True))


##### Ray CASTING ####


def strategy_line_cast(
    poi: POI,
    vert_id: int,
    current_subreg: NII,
    location: Location,
    start_point: Location,
    regions_loc: list[Location] | Location,
    normal_vector_points: tuple[Location, Location] | Directions,
    bb,
    log: Logger_Interface = _log,
):
    region = current_subreg.extract_label(regions_loc)
    # if legacy_code:
    #    horizontal_plane_landmarks_old(poi, region, label_id, bb, log)
    # else:
    extreme_point = max_distance_ray_cast(poi, region, vert_id, bb, normal_vector_points, start_point, log=log)
    if extreme_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip(bb, extreme_point, strict=True))


def max_distance_ray_cast(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | Directions = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    two_sided=False,
    log: Logger_Interface = _log,
):
    """Calculate the maximum distance ray cast in a region.

    Args:
        poi (POI): Point of interest.
        region (NII): Region to cast rays in.
        vert_id (int): Label of the region in `region`.
        bb (Tuple[slice, slice, slice]): Bounding box coordinates.
        normal_vector_points (Union[Tuple[Location, Location], Directions], optional):
            Points defining the normal vector or the direction. Defaults to "R".
        start_point (Location, optional): Starting point of the ray. Defaults to Location.Vertebra_Corpus.
        log (Logger_Interface, optional): Logger interface. Defaults to _log.

    Returns:
        Tuple[int, int, int]: The coordinates of the maximum distance ray cast.
    """
    plane_coords, arange = ray_cast(poi, region, vert_id, bb, normal_vector_points, start_point, log=log, two_sided=two_sided)
    if plane_coords is None:
        return None
    selected_arr = np.zeros(region.shape)
    selected_arr[plane_coords[..., 0], plane_coords[..., 1], plane_coords[..., 2]] = arange
    selected_arr = selected_arr * region.get_array()
    out = tuple(np.unravel_index(np.argmax(selected_arr, axis=None), selected_arr.shape))
    return out


def ray_cast(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | Directions = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    two_sided=False,
):
    """Perform ray casting in a region.

    Args:
        poi (POI): Point of interest.
        region (NII): Region to cast rays in.
        vert_id (int): Vertex ID.
        bb (Tuple[slice, slice, slice]): Bounding box coordinates.
        normal_vector_points (Union[Tuple[Location, Location], Directions], optional):
            Points defining the normal vector or the direction. Defaults to "R".
        start_point (Union[Location, np.ndarray], optional): Starting point of the ray.
            Defaults to Location.Vertebra_Corpus.
        log (Logger_Interface, optional): Logger interface. Defaults to _log.
        two_sided (bool, optional): Whether to perform two-sided ray casting. Defaults to False.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Plane coordinates and arange values.
    """
    start_point_np = _to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None, None
    ### Compute a normal vector, that defines the plane direction ###
    if isinstance(normal_vector_points, str):
        normal_vector = _get_direction(normal_vector_points, poi, vert_id)
    else:
        try:
            b = _to_local_np(normal_vector_points[1], bb, poi, vert_id, log)
            if b is None:
                raise TypeError()  # noqa: TRY301
            a = _to_local_np(normal_vector_points[0], bb, poi, vert_id, log)
            normal_vector = b - a
            normal_vector = normal_vector / norm(normal_vector)
            log.print(f"ray_cast used with old normal_vector_points {normal_vector_points}", Log_Type.FAIL)
        except TypeError as e:
            print("TypeError", e)
            return None, None

    def _calc_pixels(normal_vector, start_point_np):
        # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
        start_point_np = start_point_np.copy()
        num_pixel = np.abs(np.floor(np.max((np.array(region.shape) - start_point_np) / normal_vector))).item()
        arange = np.arange(0, num_pixel, step=1, dtype=float)
        coords = [start_point_np[i] + normal_vector[i] * arange for i in [0, 1, 2]]
        for i in [0, 1, 2]:
            cut_off = (region.shape[i] <= np.floor(coords[i])).sum()
            if cut_off == 0:
                cut_off = (np.floor(coords[i]) <= 0).sum()
            if cut_off != 0:
                coords = [c[:-cut_off] for c in coords]
                arange = arange[:-cut_off]
        return np.stack(coords, -1).astype(int), arange

    plane_coords, arange = _calc_pixels(normal_vector, start_point_np)
    if two_sided:
        plane_coords2, arange2 = _calc_pixels(-normal_vector, start_point_np)
        arange2 = -arange2
        plane_coords = np.concatenate([plane_coords, plane_coords2])
        arange = np.concatenate([arange, arange2]) - np.min(arange2)
    return plane_coords, arange


def _to_local_np(loc: Location, bb: tuple[slice, slice, slice], poi: POI, label, log: Logger_Interface):
    if (label, loc.value) in poi:
        return np.asarray([a - b.start for a, b in zip(poi[label, loc.value], bb, strict=True)])
    log.print(f"region={label},subregion={loc.value} is missing", ltype=Log_Type.FAIL)
    return None


def strategy_ligament_attachment(
    poi: POI,
    current_subreg: NII,
    location: Location,
    vert_id: int,
    bb,
    log=_log,
    corpus=None,
    direction: Directions = "R",
    compute_arcus_points=False,
    do_shift=False,
):
    if corpus is None:
        corpus = [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]
    corpus = current_subreg.extract_label(corpus)
    # Step 1: compute shift from center
    if do_shift:
        # The shift is dependen on the distance between Superior_Articular_Right and Superior_Articular_Left
        sup_articular_right = _to_local_np(Location.Superior_Articular_Right, bb, poi, vert_id, log)
        sup_articular_left = _to_local_np(Location.Superior_Articular_Left, bb, poi, vert_id, log)
        factor = 3.0
        if sup_articular_left is None or sup_articular_right is None:
            # fallback if a Superior is missing; TODO Test if we to readjust factor for the neck vertebra
            sup_articular_right = _to_local_np(Location.Inferior_Articular_Right, bb, poi, vert_id, log)
            sup_articular_left = _to_local_np(Location.Inferior_Articular_Left, bb, poi, vert_id, log)
            factor = 2.0
            if sup_articular_left is None or sup_articular_right is None:
                return
        vertebra_width = (sup_articular_right - sup_articular_left) ** 2  # TODO need zoom?
        vertebra_width = np.sqrt(np.sum(vertebra_width))
        shift = vertebra_width / factor
    else:
        shift = 0

    # Step 2: add corner points
    start_point_np = _to_local_np(Location.Vertebra_Corpus, bb, poi, vert_id, log=log)
    if start_point_np is None:
        return

    normal_vector = _get_direction(direction, poi, vert_id)
    # The main axis will be treated differently
    idx = [plane_dict[i] for i in current_subreg.orientation]
    axis = idx.index(plane_dict[direction])
    assert axis == np.argmax(np.abs(normal_vector)).item(), (axis, direction, normal_vector)
    dims = [0, 1, 2]
    dims.remove(axis)
    dim1, dim2 = dims
    if current_subreg.orientation[axis] != direction:
        shift *= -1
    # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
    start_point_np = start_point_np.copy()
    start_point_np[axis] = start_point_np[axis] + shift
    shift_total = -start_point_np.dot(normal_vector)
    xx, yy = np.meshgrid(range(current_subreg.shape[dim1]), range(current_subreg.shape[dim2]))
    zz = (-normal_vector[dim1] * xx - normal_vector[dim2] * yy - shift_total) * 1.0 / normal_vector[axis]
    z_max = current_subreg.shape[axis] - 1
    zz[zz < 0] = 0
    zz[zz > z_max] = 0
    # make cords to array again
    plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3])
    plane_coords[:, :, axis] = zz
    plane_coords[:, :, dim1] = xx
    plane_coords[:, :, dim2] = yy
    plane_coords = plane_coords.astype(int)
    # 1 where the selected subreg is, else 0
    corpus_arr = corpus.get_array()

    plane = corpus_arr[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]
    if plane.sum() == 0:
        log.print(vert_id, "add_vertebra_body_points, Plane empty", ltype=Log_Type.STRANGE)
        return
    ## Compute Corner Point
    out_points = _compute_vert_corners_in_reference_frame(poi, vert_id=vert_id, plane_coords=plane_coords, subregion=corpus_arr)
    for i, point in enumerate(out_points):
        # cords = plane_coords[point[0], point[1], :]
        poi[vert_id, location.value + i] = tuple(x + y.start for x, y in zip(point, bb, strict=False))
    for idx, (i, j, d) in enumerate([(0, 1, "S"), (1, 3, "P"), (2, 3, "I"), (0, 2, "A")], start=location.value + 4):  #
        point = (out_points[i] + out_points[j]) // 2
        point2 = max_distance_ray_cast(poi, corpus, vert_id, bb, d, point, two_sided=True)
        if point2 is None:
            point2 = point
        poi[vert_id, idx] = tuple(x + y.start for x, y in zip(point2, bb, strict=False))
    if compute_arcus_points:
        arcus = current_subreg.extract_label(Location.Arcus_Vertebrae).get_array()
        plane_arcus = arcus[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]
        for in_id, out_id in [
            (1, Location.Ligament_Attachment_Point_Flava_Superior_Median.value),
            (3, Location.Ligament_Attachment_Point_Flava_Inferior_Median.value),
        ]:
            loc102 = out_points[in_id]
            # Transform 3D Point in 2D point of plane
            arr_poi = arcus.copy() * 0
            arr_poi[loc102[0], loc102[1], loc102[2]] = 1
            loc102 = np.concatenate(np.where(arr_poi[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]))
            loc125 = get_nearest_neighbor(loc102, plane_arcus, 1)  # 41
            cords = plane_coords[loc125[0], loc125[1], :]
            poi[vert_id, out_id] = tuple(x + y.start for x, y in zip(cords, bb, strict=False))


def _compute_vert_corners_in_reference_frame(poi: POI, vert_id: int, plane_coords: np.ndarray, subregion: np.ndarray):
    to_reference_frame, _ = get_vert_direction_matrix(poi, vert_id)
    # plane_coords x,y,3
    pc = (
        plane_coords[subregion[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]] != 0].swapaxes(-1, 0).reshape((3, -1))
    )  # (3,n)
    # print(pc.shape, to_reference_frame.shape)
    cords = to_reference_frame @ pc  # 3,n; 3 = P,I,R of vert
    out: list[np.ndarray] = []
    p_101_ref = np.argmax(-cords[0] - cords[1])  # 0 101 A,S,*
    p_102_ref = np.argmax(cords[0] - cords[1])  # 1 102 P,S,*
    p_103_ref = np.argmax(-cords[0] + cords[1])  # 2 103 A,I,*
    p_104_ref = np.argmax(cords[0] + cords[1])  # 3 104 P,I,*
    out.append(np.array(tuple(pc[i, p_101_ref] for i in range(3))))
    out.append(np.array(tuple(pc[i, p_102_ref] for i in range(3))))
    out.append(np.array(tuple(pc[i, p_103_ref] for i in range(3))))
    out.append(np.array(tuple(pc[i, p_104_ref] for i in range(3))))
    return out


def get_nearest_neighbor(p, sr_msk, region_label):
    """
    get the coordinates of point x from sr_mask's provided region_label which is closest to point p

    inputs:
        p: the chose point of interest (as array)
        sr_msk: an array containing the subregion mask
        region_label: the label if the region of interest in sr_msk
    output:
        out_point: the point from sr_msk[region_label] closest to point p (as array)
    """
    if len(p.shape) == 1:
        p = np.expand_dims(p, 1)
    locs = np.where(sr_msk == region_label)
    locs_array = np.array(list(locs)).T
    distances = cdist(p.T, locs_array)

    return locs_array[distances.argmin()]


##### Add all Strategy to the strategy list #####
# fmt: off
L = Location
Strategy_Pattern_Side_Effect(L.Vertebra_Direction_Posterior,L.Vertebra_Direction_Inferior)
Strategy_Pattern_Side_Effect(L.Vertebra_Direction_Right,L.Vertebra_Direction_Inferior)
Strategy_Pattern_Side_Effect(L.Vertebra_Direction_Inferior,L.Vertebra_Corpus)
S = strategy_extreme_points
Strategy_Pattern(L.Muscle_Inserts_Spinosus_Process, strategy=S, subreg_id=L.Spinosus_Process, direction=("P","I"))  # 81
Strategy_Pattern(L.Muscle_Inserts_Transverse_Process_Right, strategy=S, subreg_id=L.Costal_Process_Right, direction=("P","R"))  # 82
Strategy_Pattern(L.Muscle_Inserts_Transverse_Process_Left, strategy=S, subreg_id=L.Costal_Process_Left, direction=("P","L"))  # 83
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Inferior_Left, strategy=S, subreg_id=L.Inferior_Articular_Left, direction=("I")) # 86
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Inferior_Right, strategy=S, subreg_id=L.Inferior_Articular_Right, direction=("I")) # 87
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Superior_Left, strategy=S, subreg_id=L.Superior_Articular_Left, direction=("S")) # 88
Strategy_Pattern(L.Muscle_Inserts_Articulate_Process_Superior_Right, strategy=S, subreg_id=L.Superior_Articular_Right, direction=("S")) # 89
S = strategy_line_cast
Strategy_Pattern(L.Muscle_Inserts_Vertebral_Body_Right, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="R" ) # 84
Strategy_Pattern(L.Muscle_Inserts_Vertebral_Body_Left, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="L" ) # 85
Strategy_Pattern(
    L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,
    strategy_ligament_attachment,
    compute_arcus_points=True,
    corpus=[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
    prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left}
)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Superior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Posterior_Central_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Inferior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Anterior_Central_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Flava_Superior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Flava_Inferior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median)

Strategy_Pattern(
    L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
    strategy_ligament_attachment,
    corpus=[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
    prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left},
    do_shift=True,
    direction="R"
)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Superior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Posterior_Central_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Inferior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Anterior_Central_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)

Strategy_Pattern(
    L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
    strategy_ligament_attachment,
    corpus=[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
    prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left},
    do_shift=True,
    direction="L"
)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Superior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Posterior_Central_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Inferior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Anterior_Central_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)

Strategy_Computed_Before(L.Spinal_Canal_ivd_lvl,L.Vertebra_Disc,L.Vertebra_Corpus)
Strategy_Computed_Before(L.Spinal_Canal,L.Vertebra_Corpus)
# fmt: on
def compute_non_centroid_pois(
    poi: POI,
    locations: Sequence[Location] | Location,
    vert: NII,
    subreg: NII,
    _vert_ids: Sequence[int] | None = None,
    log: Logger_Interface = _log,
):
    locations = list(locations) if isinstance(locations, Sequence) else [locations]
    ### STEP 1 Vert Direction###
    if Location.Vertebra_Direction_Inferior in locations:
        log.print("Compute Vertebra directions", locations)
        ### Calc vertebra direction; We always need them, so we just compute them. ###
        sub_regions = poi.keys_subregion()
        if any(a.value not in sub_regions for a in vert_directions):
            poi, _ = calc_orientation_of_vertebra_PIR(poi, vert, subreg, do_fill_back=False, save_normals_in_info=False)
            for i in vert_directions:
                if i in locations:
                    locations.remove(i)

    locations = [pois_computed_by_side_effect.get(l.value, l) for l in locations]
    locations = sorted(list(set(locations)), key=lambda x: x.value)  # type: ignore # noqa: C414
    log.print("Calc pois from subregion id", {l.name for l in locations})
    ### STEP 2 (Other global non centroid poi; Spinal heights ###
    if Location.Spinal_Canal in locations:
        locations.remove(Location.Spinal_Canal)
        subregs_ids = subreg.unique()
        _a = Location.Spinal_Canal.value in subregs_ids or Location.Spinal_Cord.value in subregs_ids
        if _a and Location.Spinal_Canal.value not in poi.keys_subregion():
            poi = calc_center_spinal_cord(poi, subreg)
    if Location.Spinal_Canal_ivd_lvl in locations:
        locations.remove(Location.Spinal_Canal_ivd_lvl)
        subregs_ids = subreg.unique()
        v = Location.Spinal_Canal_ivd_lvl.value
        if (v in subregs_ids or Location.Spinal_Cord.value in subregs_ids) and v not in poi.keys_subregion():
            poi = calc_center_spinal_cord(
                poi, subreg, source_subreg_point_id=Location.Vertebra_Disc, subreg_id=Location.Spinal_Canal_ivd_lvl
            )

    if _vert_ids is None:
        _vert_ids = vert.unique()
    # Step 3 Compute on individual Vertebras
    for vert_id in _vert_ids:
        if vert_id >= 39:
            continue
        current_vert = vert.extract_label(vert_id)
        bb = current_vert.compute_crop()
        current_vert.apply_crop_(bb)
        current_subreg = subreg.apply_crop(bb) * current_vert
        for location in locations:
            if location.value <= 50:
                continue
            if (vert_id, location.value) in poi:
                continue
            if location in [
                Location.Implant_Entry_Left,
                Location.Implant_Entry_Right,
                Location.Implant_Target_Left,
                Location.Implant_Target_Right,
                Location.Spinal_Canal,
            ]:
                continue
            if location.value in all_poi_functions:
                all_poi_functions[location.value](poi, current_subreg, vert_id, bb=bb, log=log)
            else:
                raise NotImplementedError(location.value)


def calc_center_spinal_cord(
    poi: POI,
    subreg: NII,
    spline_subreg_point_id=Location.Vertebra_Corpus,
    source_subreg_point_id=Location.Vertebra_Corpus,
    subreg_id=Location.Spinal_Canal,
    intersection_target: list[Location] | None = None,
    _fill_inplace: NII | None = None,
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
        idx = [plane_dict[i] for i in subreg_iso.orientation]
        axis = idx.index(plane_dict["S"])
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
            fill_back[select == 1] = reg_label

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
        # print(arr.shape, _fill_inplace, fill_back.shape)
        _fill_inplace.set_array_(arr)
    ret = calc_centroids(subreg_iso.set_array(out), subreg_id=subreg_id, extend_to=poi_iso.extract_subregion(subreg_id), inplace=True)
    ret.rescale_(poi.zoom)
    return poi.join_left_(ret)


def print_prerequisites():
    print("digraph G {")
    for source, strategy in all_poi_functions.items():
        for prereq in strategy.prerequisite:
            print(f"{source} -> {prereq.value}")
    print("}")


def add_prerequisites(locs: Sequence[Location]):
    addendum = set()
    locs2 = set(locs)
    loop_var = locs2
    i = 0
    while i != 1000:  # Prevent Deadlock
        for l in loop_var:
            if l.value in all_poi_functions:
                for prereq in all_poi_functions[l.value].prerequisite:
                    if prereq not in locs:
                        addendum.add(prereq)
        if len(addendum) == 0:
            break
        locs2 = addendum | locs2
        loop_var = addendum
        addendum = set()
        i += 1
    else:
        warnings.warn("Deadlock in add_prerequisites", stacklevel=10)
    return sorted(list(locs2), key=lambda x: x.value)  # type: ignore # noqa: C414
