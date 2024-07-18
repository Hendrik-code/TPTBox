import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from time import time
from typing import NoReturn

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

from TPTBox import NII, POI, Log_Type, Logger_Interface, Print_Logger, calc_poi_from_subreg_vert
from TPTBox.core.vert_constants import DIRECTIONS, Location, _plane_dict, never_called, vert_directions

_log = Print_Logger()
Vertebra_Orientation = tuple[np.ndarray, np.ndarray, np.ndarray]
all_poi_functions: dict[int, "Strategy_Pattern"] = {}
pois_computed_by_side_effect: dict[int, Location] = {}


def run_poi_pipeline(vert: NII, subreg: NII, poi_path: Path, logger: Logger_Interface = _log):
    poi = calc_poi_from_subreg_vert(vert, subreg, buffer_file=poi_path, save_buffer_file=True, subreg_id=list(Location), verbose=logger)
    poi.save(poi_path)


def _strategy_side_effect(*args, **qargs):  # noqa: ARG001
    pass


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__!r} took: {te-ts:2.4f} sec")
        return result

    return wrap


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

    def __init__(self, target: Location, strategy: Callable, prerequisite: set[Location] | None = None, prio=0, **args) -> None:
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
        self._prio = prio

    def __call__(self, poi: POI, current_subreg: NII, vert_id: int, bb, log: Logger_Interface = _log):
        try:
            return self.strategy(poi=poi, current_subreg=current_subreg, location=self.target, log=log, vert_id=vert_id, bb=bb, **self.args)
        except Exception:
            _log.print_error()
            return None

    def prority(self):
        return self.target.value + self._prio


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
        out += target_labels * select * reg_label
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
        arr = subreg_sar.set_array(fill_back).reorient(poi.orientation).rescale_(poi.zoom).get_array()
        fill_back_nii.set_array_(arr)

    ret = calc_centroids(subreg_iso.set_array(out), subreg_id=subreg_id, extend_to=poi_iso.copy(), inplace=True)
    poi._vert_orientation_pir = {}
    if save_normals_in_info:
        poi.info["vert_orientation_PIR"] = poi._vert_orientation_pir

    # calc posterior vector and the crossproduct
    for vert_id, normal_down in down_vector.items():
        # get two points and compute the direction:
        a = np.array(ret[vert_id : subreg_id.value]) - 1
        b = np.array(ret[vert_id : source_subreg_point_id.value]) - 1
        normal_vector_post = a - b
        normal_vector_post = normal_vector_post / norm(normal_vector_post)
        poi._vert_orientation_pir[vert_id] = (normal_vector_post, normal_down, np.cross(normal_vector_post, normal_down))

        ### MAKE DIRECTIONS POIs ###
        # print(ret[vert_id, source_subreg_point_id], normal_vector_post)
        ret[vert_id, Location.Vertebra_Direction_Posterior] = tuple(ret[vert_id, source_subreg_point_id] + normal_vector_post * 10)
        ret[vert_id, Location.Vertebra_Direction_Inferior] = tuple(ret[vert_id, source_subreg_point_id] + normal_down * 10)
        ret[vert_id, Location.Vertebra_Direction_Right] = tuple(
            ret[vert_id:source_subreg_point_id] + np.cross(normal_vector_post, normal_down * 10)
        )

    # if make_thicker:
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


def _get_direction(d: DIRECTIONS, poi: POI, vert_id: int) -> np.ndarray:
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


def get_extreme_point_by_vert_direction(poi: POI, region: NII, vert_id, direction: Sequence[DIRECTIONS] | DIRECTIONS = "I"):
    """
    Get the extreme point in a specified direction.

    Args:
        poi (POI): The chosen point of interest represented as an array.
        region (NII): An array containing the subregion mask.
        vert_id: The ID of the vertex.
        direction (Union[Sequence[DIRECTIONS], DIRECTIONS], optional): The direction(s) to search for the extreme point.
            Defaults to "I" (positive direction along the secondary axis).

    Note:
        Assumes `region` contains binary values indicating the presence of points.
        Uses `_get_sub_array_by_direction` internally.
    """
    direction_: Sequence[DIRECTIONS] = direction if isinstance(direction, Sequence) else (direction,)  # type: ignore

    to_reference_frame, from_reference_frame = get_vert_direction_matrix(poi, vert_id=vert_id)
    pc = np.stack(np.where(region.get_array() == 1))
    cords = to_reference_frame @ pc  # 3,n; 3 = P,I,R of vert
    a = [_get_sub_array_by_direction(d, cords) for d in direction_]
    idx = np.argmax(sum(a))
    return pc[:, idx]


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


def get_vert_direction_matrix(poi: POI, vert_id: int):
    P, I, R = get_vert_direction_PIR(poi, vert_id=vert_id)  # noqa: N806
    from_vert_orient = np.stack([P, I, R], axis=1)
    to_vert_orient = np.linalg.inv(from_vert_orient)
    return to_vert_orient, from_vert_orient


def strategy_extreme_points(
    poi: POI,
    current_subreg: NII,
    location: Location,
    direction: Sequence[DIRECTIONS] | DIRECTIONS,
    vert_id: int,
    subreg_id: Location | list[Location],
    bb,
    log=_log,
):
    """Strategy function to update extreme points of a point of interest based on direction.

    Args:
        poi (POI): The point of interest.
        current_subreg (NII): The current subregion.
        location (Location): The location to update in the point of interest.
        direction (Union[Sequence[DIRECTIONS], DIRECTIONS]): Direction(s) to search for the extreme point.
        vert_id (int): The vertex ID.
        subreg_id (Location): The subregion ID.
        bb: The bounding box.
        log (Logger_Interface, optional): The logger interface. Defaults to _log.
    """

    region = current_subreg.extract_label(subreg_id)
    if region.sum() == 0:
        log.print(f"reg={vert_id},subreg={subreg_id} is missing (extreme_points); {current_subreg.unique()}", ltype=Log_Type.FAIL)
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
    start_point: Location | np.ndarray,
    regions_loc: list[Location] | Location,
    normal_vector_points: tuple[Location, Location] | DIRECTIONS,
    bb,
    log: Logger_Interface = _log,
):
    region = current_subreg.extract_label(regions_loc)
    # if legacy_code:
    #    horizontal_plane_landmarks_old(poi, region, label_id, bb, log)
    # else:
    extreme_point = max_distance_ray_cast_convex(poi, region, vert_id, bb, normal_vector_points, start_point, log=log)
    if extreme_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip(bb, extreme_point, strict=True))


def max_distance_ray_cast_pixel_level(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | DIRECTIONS = "R",
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
        normal_vector_points (Union[Tuple[Location, Location], DIRECTIONS], optional):
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


def max_distance_ray_cast_convex(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | DIRECTIONS = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    acc_delta: float = 0.00005,
):
    start_point_np = _to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None

    """Convex assumption!"""
    # Compute a normal vector, that defines the plane direction
    if isinstance(normal_vector_points, str):
        normal_vector = _get_direction(normal_vector_points, poi, vert_id) / np.array(poi.zoom)

        normal_vector = normal_vector / norm(normal_vector)
    else:
        try:
            b = _to_local_np(normal_vector_points[1], bb, poi, vert_id, log)
            if b is None:
                return None
            a = _to_local_np(normal_vector_points[0], bb, poi, vert_id, log)
            normal_vector = b - a
            normal_vector = normal_vector / norm(normal_vector)
            log.print(f"ray_cast used with old normal_vector_points {normal_vector_points}", Log_Type.FAIL)
        except TypeError as e:
            print("TypeError", e)
            return None
    # Create a function to interpolate within the mask array
    interpolator = RegularGridInterpolator([np.arange(region.shape[i]) for i in range(3)], region.get_array())

    def is_inside(distance):
        coords = [start_point_np[i] + normal_vector[i] * distance for i in [0, 1, 2]]
        if any(i < 0 for i in coords):
            return 0
        if any(coords[i] > region.shape[i] - 1 for i in range(len(coords))):
            return 0
        # Evaluate the mask value at the interpolated coordinates
        mask_value = interpolator(coords)
        return mask_value > 0.5

    if not is_inside(0):
        return start_point_np
    count = 0
    min_v = 0
    max_v = sum(region.shape)
    delta = max_v * 2
    while acc_delta < delta:
        bisection = (max_v - min_v) / 2 + min_v
        if is_inside(bisection):
            min_v = bisection
        else:
            max_v = bisection
        delta = max_v - min_v
        count += 1
    return start_point_np + normal_vector * ((min_v + max_v) / 2)
    ## Golden section search
    # phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    # a, b = 0, sum(region.shape)
    #
    # while abs(b - a) > acc_delta:
    #    x1 = b - (b - a) / phi
    #    x2 = a + (b - a) / phi
    #
    #    if is_inside(x1) > is_inside(x2):
    #        a = x1
    #    else:
    #        b = x2
    #    count += 1
    # print(count)
    # return start_point_np + normal_vector * ((a + b) / 2)


def ray_cast(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | DIRECTIONS = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    two_sided=False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Perform ray casting in a region.

    Args:
        poi (POI): Point of interest.
        region (NII): Region to cast rays in.
        vert_id (int): Vertex ID.
        bb (Tuple[slice, slice, slice]): Bounding box coordinates.
        normal_vector_points (Union[Tuple[Location, Location], DIRECTIONS], optional):
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

    # Compute a normal vector, that defines the plane direction
    if isinstance(normal_vector_points, str):
        normal_vector = _get_direction(normal_vector_points, poi, vert_id) / np.array(poi.zoom)
        normal_vector = normal_vector / norm(normal_vector)
    else:
        try:
            b = _to_local_np(normal_vector_points[1], bb, poi, vert_id, log)
            if b is None:
                return None, None
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

        # Clip coordinates to region bounds
        for i in [0, 1, 2]:
            coords[i] = np.clip(coords[i], 0, region.shape[i] - 1)
        # Convert coordinates to integers for indexing
        int_coords = [c.astype(int) for c in coords]

        return np.stack(int_coords, -1), arange

    plane_coords, arange = _calc_pixels(normal_vector, start_point_np)
    if two_sided:
        plane_coords2, arange2 = _calc_pixels(-normal_vector, start_point_np)
        arange2 = -arange2
        plane_coords = np.concatenate([plane_coords, plane_coords2])
        arange = np.concatenate([arange, arange2]) - np.min(arange2)

    return plane_coords, arange


#### find corner ####


def strategy_find_corner(
    poi: POI,
    current_subreg: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    location: Location,
    vec1: Location,
    vec2: Location,
    start_point=Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    shift_direction: DIRECTIONS | None = None,
):
    start_point = shift_point(poi, vert_id, bb, start_point, direction=shift_direction, log=log)
    corner_point = _find_corner_point(poi, current_subreg, vert_id, bb, start_point, vec1=vec1, vec2=vec2, log=log, location=location)

    if corner_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip(bb, corner_point, strict=True))


# @timing
def _find_corner_point(
    poi: POI, region, vert_id, bb, start_point, vec1, vec2, log: Logger_Interface = _log, delta=0.00000005, location=None
):
    # Convert start point and vectors to local numpy coordinates
    start_point_np = _to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None
    if (vert_id, vec1.value) not in poi.keys():
        log.on_fail(f"find_corner_point - point missing {vert_id=} {vec1.value=} {location=},{poi.keys()}")
        return
    if (vert_id, vec2.value) not in poi.keys():
        log.on_fail(f"find_corner_point - point missing {(vert_id, vec2.value)=} {location=}")
        return
    v1 = _to_local_np(vec1, bb, poi, vert_id, log) - start_point_np
    v2 = _to_local_np(vec2, bb, poi, vert_id, log) - start_point_np
    if norm(v1) < 0.000001 and norm(v2) < 0.000001:
        log.on_fail(
            f"find_corner_point - Points to close {vert_id=};{start_point_np=},{vec1=},{vec2=},{norm(v1)=},{norm(v2)=}  ",
        )
        return None
    # Initialize factors and interpolator
    factor1 = factor2 = 1
    interpolator = RegularGridInterpolator([np.arange(region.shape[i]) for i in range(3)], region.get_array())

    def is_inside(f1=0.0, f2=0.0):
        coords = start_point_np + (factor1 + f1) * v1 + (factor2 + f2) * v2
        if any(i < 0 for i in coords) or any(coords[i] > region.shape[i] - 1 for i in range(len(coords))):
            return False
        return interpolator(coords) > 0.5

    # Adjust factors until inside region
    while not is_inside():
        factor1 = factor2 = factor2 * 0.98

    v1_n: float = 1 / norm(v1)  # type: ignore
    v2_n: float = 1 / norm(v2)  # type: ignore

    f1 = f2 = 0

    # Refine factors using delta
    while delta > (v1_n + v2_n) / 2:
        changed = False
        if is_inside(v1_n + f1, f2):
            f1 += v1_n
            changed = True
        if is_inside(f1, f2 + v2_n):
            f2 += v2_n
            changed = True
        if not changed:
            v1_n /= 2
            v2_n /= 2

    return start_point_np + (factor1 + f1) * v1 + (factor2 + f2) * v2


####


def strategy_ligament_attachment_point_flava(
    poi: POI,
    current_subreg: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    location: Location,
    goal: Location,
    log: Logger_Interface = _log,
    delta=0.0000001,
):
    try:
        normal_vector1 = _get_direction("S", poi, vert_id) / np.array(poi.zoom)
        v1 = normal_vector1 / norm(normal_vector1)

        normal_vector2 = _get_direction("A", poi, vert_id) / np.array(poi.zoom)
        v2 = normal_vector2 / norm(normal_vector2)
    except KeyError:
        return

    start_point_np = _to_local_np(Location.Spinosus_Process, bb, poi, vert_id, log) if isinstance(goal, Location) else goal

    goal_np = _to_local_np(goal, bb, poi, vert_id, log) if isinstance(goal, Location) else goal
    if goal_np is None or start_point_np is None:
        return

    region = current_subreg.extract_label([Location.Arcus_Vertebrae, Location.Spinosus_Process]).get_array()
    interpolator = RegularGridInterpolator([np.arange(region.shape[i]) for i in range(3)], region)
    dist_curr = norm(start_point_np - goal_np)

    def is_inside_and_closer(f1=0.0, f2=0.0):
        coords = start_point_np + f1 * v1 + f2 * v2
        if any(i < 0 for i in coords) or any(coords[i] > region.shape[i] - 1 for i in range(len(coords))):
            return False
        if interpolator(coords) > 0.5:
            dist_new = norm(coords - goal_np)
            return dist_curr > dist_new
        else:
            return False

    v1_n: float = 0.5
    v2_n: float = 0.5
    f1 = f2 = 0
    # Refine factors using delta
    while delta < (v1_n + v2_n) / 2:
        changed = False
        if is_inside_and_closer(v1_n + f1, f2):
            f1 += v1_n
            changed = True
            coords = start_point_np + f1 * v1 + f2 * v2
            dist_curr = norm(coords - goal_np)
        if is_inside_and_closer(f1, f2 + v2_n):
            f2 += v2_n
            changed = True
            coords = start_point_np + f1 * v1 + f2 * v2
            dist_curr = norm(coords - goal_np)

        if not changed:
            v1_n /= 2
            v2_n /= 2

    coords = start_point_np + f1 * v1 + f2 * v2
    poi[vert_id, location] = tuple(x + y.start for x, y in zip(coords, bb, strict=False))


def _to_local_np(loc: Location, bb: tuple[slice, slice, slice], poi: POI, label, log: Logger_Interface):
    if (label, loc.value) in poi:
        return np.asarray([a - b.start for a, b in zip(poi[label, loc.value], bb, strict=True)])
    log.print(f"region={label},subregion={loc.value} is missing", ltype=Log_Type.FAIL)
    # raise KeyError(f"region={label},subregion={loc.value} is missing")
    return None


def shift_point(
    poi: POI,
    vert_id: int,
    bb,
    start_point: Location = Location.Vertebra_Corpus,
    direction: DIRECTIONS | None = "R",
    log: Logger_Interface = _log,
):
    if direction is None:
        return _to_local_np(start_point, bb, poi, vert_id, log)
    sup_articular_right = _to_local_np(Location.Superior_Articular_Right, bb, poi, vert_id, log)
    sup_articular_left = _to_local_np(Location.Superior_Articular_Left, bb, poi, vert_id, log)
    factor = 3.0
    if sup_articular_left is None or sup_articular_right is None:
        sup_articular_right = _to_local_np(Location.Inferior_Articular_Right, bb, poi, vert_id, log)
        sup_articular_left = _to_local_np(Location.Inferior_Articular_Left, bb, poi, vert_id, log)
        factor = 2.0
        if sup_articular_left is None or sup_articular_right is None:
            return
    if vert_id <= 11:
        factor *= (12 - vert_id) / 11 + 1
    vertebra_width = np.linalg.norm(sup_articular_right - sup_articular_left)
    shift = vertebra_width / factor
    normal_vector = _get_direction(direction, poi, vert_id) / np.array(poi.zoom)
    normal_vector = normal_vector / norm(normal_vector)
    start_point_np = _to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None
    return start_point_np + normal_vector * shift


def strategy_shifted_line_cast(
    poi: POI,
    current_subreg: NII,
    location: Location,
    vert_id: int,
    bb,
    regions_loc: list[Location] | Location,
    normal_vector_points: tuple[Location, Location] | DIRECTIONS,
    start_point: Location = Location.Vertebra_Corpus,
    log=_log,
    direction: DIRECTIONS = "R",
    do_shift=True,
):
    try:
        cords = shift_point(poi, vert_id, bb, start_point, direction=direction if do_shift else None, log=log)
    except KeyError as e:
        log.print(f"region={vert_id},subregion={e} is missing", ltype=Log_Type.FAIL)
        return
    if cords is None:
        return
    strategy_line_cast(
        start_point=cords,
        regions_loc=regions_loc,
        normal_vector_points=normal_vector_points,
        poi=poi,
        vert_id=vert_id,
        current_subreg=current_subreg,
        location=location,
        bb=bb,
        log=_log,
    )


# def strategy_ligament_attachment(
#    poi: POI,
#    current_subreg: NII,
#    location: Location,
#    vert_id: int,
#    bb,
#    log=_log,
#    corpus=None,
#    direction: DIRECTIONS = "R",
#    compute_arcus_points=False,
#    do_shift=False,
# ):
#    if corpus is None:
#        corpus = [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]
#    corpus = current_subreg.extract_label(corpus)
#    org_zoom = None
#    # if max(poi.zoom) / min(poi.zoom) > 1.5:
#    #    org_zoom = poi.zoom
#    #    poi = poi.rescale_()
#    #    current_subreg = current_subreg.rescale()
#    # Step 1: compute shift from center
#    if do_shift:
#        sup_articular_right = _to_local_np(Location.Superior_Articular_Right, bb, poi, vert_id, log)
#        sup_articular_left = _to_local_np(Location.Superior_Articular_Left, bb, poi, vert_id, log)
#        factor = 3.0
#        if sup_articular_left is None or sup_articular_right is None:
#            sup_articular_right = _to_local_np(Location.Inferior_Articular_Right, bb, poi, vert_id, log)
#            sup_articular_left = _to_local_np(Location.Inferior_Articular_Left, bb, poi, vert_id, log)
#            factor = 2.0
#            if sup_articular_left is None or sup_articular_right is None:
#                return
#        vertebra_width = np.linalg.norm(sup_articular_right - sup_articular_left)
#        shift = vertebra_width / factor
#    else:
#        shift = 0
#
#    # Step 2: add corner points
#    start_point_np = _to_local_np(Location.Vertebra_Corpus, bb, poi, vert_id, log=log)
#    if start_point_np is None:
#        return
#    normal_vector = _get_direction(direction, poi, vert_id)  # / np.array(poi.zoom)
#    # normal_vector = normal_vector / norm(normal_vector)
#
#    idx = [_plane_dict[i] for i in current_subreg.orientation]
#    axis = idx.index(_plane_dict[direction])
#    assert axis == np.argmax(np.abs(normal_vector)).item(), (axis, direction, normal_vector)
#    dims = [0, 1, 2]
#    dims.remove(axis)
#    dim1, dim2 = dims
#    if current_subreg.orientation[axis] != direction:
#        shift *= -1
#    start_point_np = start_point_np.copy()
#    start_point_np[axis] = start_point_np[axis] + shift + 1
#    shift_total = -start_point_np.dot(normal_vector)
#    xx, yy = np.meshgrid(range(current_subreg.shape[dim1]), range(current_subreg.shape[dim2]))
#    zz = (-normal_vector[dim1] * xx - normal_vector[dim2] * yy - shift_total) / normal_vector[axis]
#
#    z_max = current_subreg.shape[axis] - 1
#    zz[zz < 0] = 0
#    zz[zz > z_max] = z_max
#
#    xx = xx.astype(np.float32)
#    yy = yy.astype(np.float32)
#    zz = zz.astype(np.float32)
#
#    x0 = np.floor(xx).astype(int)
#    x1 = x0 + 1
#    y0 = np.floor(yy).astype(int)
#    y1 = y0 + 1
#    z0 = np.floor(zz).astype(int)
#    z1 = z0 + 1
#
#    x1[x1 >= current_subreg.shape[dim1]] = x0[x1 >= current_subreg.shape[dim1]]
#    y1[y1 >= current_subreg.shape[dim2]] = y0[y1 >= current_subreg.shape[dim2]]
#    z1[z1 >= current_subreg.shape[axis]] = z0[z1 >= current_subreg.shape[axis]]
#
#    xd = (xx - x0).reshape(-1)
#    yd = (yy - y0).reshape(-1)
#    zd = (zz - z0).reshape(-1)
#
#    corpus_arr = corpus.get_array()
#
#    def trilinear_interpolation(v000, v001, v010, v011, v100, v101, v110, v111, xd, yd, zd):
#        c00 = v000 * (1 - xd) + v100 * xd
#        c01 = v001 * (1 - xd) + v101 * xd
#        c10 = v010 * (1 - xd) + v110 * xd
#        c11 = v011 * (1 - xd) + v111 * xd
#        c0 = c00 * (1 - yd) + c10 * yd
#        c1 = c01 * (1 - yd) + c11 * yd
#        return c0 * (1 - zd) + c1 * zd
#
#    v000 = corpus_arr[x0, y0, z0].reshape(-1)
#    v001 = corpus_arr[x0, y0, z1].reshape(-1)
#    v010 = corpus_arr[x0, y1, z0].reshape(-1)
#    v011 = corpus_arr[x0, y1, z1].reshape(-1)
#    v100 = corpus_arr[x1, y0, z0].reshape(-1)
#    v101 = corpus_arr[x1, y0, z1].reshape(-1)
#    v110 = corpus_arr[x1, y1, z0].reshape(-1)
#    v111 = corpus_arr[x1, y1, z1].reshape(-1)
#
#    plane = trilinear_interpolation(v000, v001, v010, v011, v100, v101, v110, v111, xd, yd, zd).reshape(xx.shape)
#
#    if plane.sum() == 0:
#        log.print(vert_id, "add_vertebra_body_points, Plane empty", ltype=Log_Type.STRANGE)
#        return
#
#    plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3])
#    plane_coords[:, :, axis] = zz
#    plane_coords[:, :, dim1] = xx
#    plane_coords[:, :, dim2] = yy
#    plane_coords = plane_coords.astype(int)
#
#    out_points = _compute_vert_corners_in_reference_frame(poi, vert_id=vert_id, plane_coords=plane_coords, subregion=corpus_arr)
#    for i, point in enumerate(out_points):
#        poi[vert_id, location.value + i] = tuple(x + y.start for x, y in zip(point, bb, strict=False))
#
#    for idx, (i, j, d) in enumerate([(0, 1, "S"), (1, 3, "P"), (2, 3, "I"), (0, 2, "A")], start=location.value + 4):
#        point = (out_points[i] + out_points[j]) // 2
#        point2 = max_distance_ray_cast(poi, corpus, vert_id, bb, d, point, two_sided=True)
#        if point2 is None:
#            point2 = point
#        poi[vert_id, idx] = tuple(x + y.start for x, y in zip(point2, bb, strict=False))
#
#    if compute_arcus_points:
#        arcus = current_subreg.extract_label(Location.Arcus_Vertebrae).get_array()
#        plane_arcus = arcus[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]
#        for in_id, out_id in [
#            (1, Location.Ligament_Attachment_Point_Flava_Superior_Median.value),
#            (3, Location.Ligament_Attachment_Point_Flava_Inferior_Median.value),
#        ]:
#            try:
#                loc102 = out_points[in_id]
#                arr_poi = arcus.copy() * 0
#                arr_poi[loc102[0], loc102[1], loc102[2]] = 1
#                loc102 = np.concatenate(np.where(arr_poi[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]))
#                loc125 = get_nearest_neighbor(loc102, plane_arcus, 1)  # 41
#                cords = plane_coords[loc125[0], loc125[1], :]
#                poi[vert_id, out_id] = tuple(x + y.start for x, y in zip(cords, bb, strict=False))
#            except Exception:
#                print(vert_id, out_id, "missed its target. Skipped", loc102.sum(), plane_arcus.sum(), np.unique(plane_arcus))
#    if org_zoom is not None:
#        poi.rescale_(org_zoom)


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
#Strategy_Pattern(L.Vertebra_Disc_Post, strategy=S, subreg_id=L.Vertebra_Disc, direction=("P"))

S = strategy_line_cast
Strategy_Pattern(L.Muscle_Inserts_Vertebral_Body_Right, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="R" ) # 84
Strategy_Pattern(L.Muscle_Inserts_Vertebral_Body_Left, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="L" ) # 85
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Superior_Median, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="S" ) # 105
Strategy_Pattern(L.Additional_Vertebral_Body_Posterior_Central_Median, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="P" ) # 106
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Inferior_Median, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="I" ) # 107
Strategy_Pattern(L.Additional_Vertebral_Body_Anterior_Central_Median, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus, normal_vector_points ="A" ) # 108
S = strategy_shifted_line_cast
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Superior_Right, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="R",normal_vector_points ="S",
     ) # 121
Strategy_Pattern(L.Additional_Vertebral_Body_Posterior_Central_Right, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="R",normal_vector_points ="P",
     ) # 122
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Inferior_Right, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="R",normal_vector_points ="I",
     ) # 123
Strategy_Pattern(L.Additional_Vertebral_Body_Anterior_Central_Right, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="R",normal_vector_points ="A",
     ) # 124

Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Superior_Left, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="L",normal_vector_points ="S",
     ) # 113
Strategy_Pattern(L.Additional_Vertebral_Body_Posterior_Central_Left, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="L",normal_vector_points ="P",
     ) # 114
Strategy_Pattern(L.Additional_Vertebral_Body_Middle_Inferior_Left, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="L",normal_vector_points ="I",
     ) # 115
Strategy_Pattern(L.Additional_Vertebral_Body_Anterior_Central_Left, strategy=S, regions_loc =[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
                 start_point = L.Vertebra_Corpus,prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left,L.Vertebra_Direction_Inferior},
                 direction="L",normal_vector_points ="A",
     ) # 116

S = strategy_find_corner
Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Median,prio=150) #101

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Median,prio=150) #102

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Median,prio=150) #103

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Median,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Median,prio=150) #104

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Right,prio=100,shift_direction="R") #117

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Right,prio=100,shift_direction="R") #118

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Right,prio=100,shift_direction="R") #119

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Right,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Right,prio=100,shift_direction="R") #120

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Left,prio=100,shift_direction="L") #109

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Superior_Left,prio=100,shift_direction="L") #110

Strategy_Pattern(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Anterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Left,prio=100,shift_direction="L") #111

Strategy_Pattern(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,start_point = L.Vertebra_Corpus,strategy=S,prerequisite={L.Vertebra_Direction_Inferior},
    vec1= L.Additional_Vertebral_Body_Posterior_Central_Left,vec2 = L.Additional_Vertebral_Body_Middle_Inferior_Left,prio=100,shift_direction="L") #112
S = strategy_ligament_attachment_point_flava
Strategy_Pattern(L.Ligament_Attachment_Point_Flava_Superior_Median,goal = L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,strategy=S,prio=200,
                 prerequisite={L.Spinosus_Process}
                 ) #125
Strategy_Pattern(L.Ligament_Attachment_Point_Flava_Inferior_Median,goal = L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,strategy=S,prio=200,
                 prerequisite={L.Spinosus_Process}) #127
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Flava_Superior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Flava_Inferior_Median,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)

#Strategy_Pattern(
#    L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
#    strategy_ligament_attachment,
#    corpus=[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
#    prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left},
#    do_shift=True,
#    direction="R"
#)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Superior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Posterior_Central_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Inferior_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Anterior_Central_Right,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right)

#Strategy_Pattern(
#    L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
#    strategy_ligament_attachment,
#    corpus=[L.Vertebra_Corpus, L.Vertebra_Corpus_border],
#    prerequisite={L.Superior_Articular_Right,L.Superior_Articular_Left,L.Inferior_Articular_Right,L.Inferior_Articular_Left},
#    do_shift=True,
#    direction="L"
#)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Superior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Posterior_Central_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Middle_Inferior_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)
#Strategy_Pattern_Side_Effect(L.Additional_Vertebral_Body_Anterior_Central_Left,L.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left)

Strategy_Computed_Before(L.Dens_axis,L.Vertebra_Direction_Inferior)
Strategy_Computed_Before(L.Spinal_Canal_ivd_lvl,L.Vertebra_Disc,L.Vertebra_Corpus,L.Dens_axis)
Strategy_Computed_Before(L.Spinal_Cord,L.Vertebra_Disc,L.Vertebra_Corpus,L.Dens_axis)
Strategy_Computed_Before(L.Spinal_Canal,L.Vertebra_Corpus)

# fmt: on
def compute_non_centroid_pois(  # noqa: C901
    poi: POI,
    locations: Sequence[Location] | Location,
    vert: NII,
    subreg: NII,
    _vert_ids: Sequence[int] | None = None,
    log: Logger_Interface = _log,
):
    if _vert_ids is None:
        _vert_ids = vert.unique()

    locations = list(locations) if isinstance(locations, Sequence) else [locations]
    ### STEP 1 Vert Direction###
    if Location.Vertebra_Direction_Inferior in locations:
        log.print("Compute Vertebra DIRECTIONS")
        ### Calc vertebra direction; We always need them, so we just compute them. ###
        sub_regions = poi.keys_subregion()
        if any(a.value not in sub_regions for a in vert_directions):
            poi, _ = calc_orientation_of_vertebra_PIR(poi, vert, subreg, do_fill_back=False, save_normals_in_info=False)
            for i in vert_directions:
                if i in locations:
                    locations.remove(i)

    locations = [pois_computed_by_side_effect.get(l.value, l) for l in locations]
    locations = sorted(set(locations), key=lambda x: all_poi_functions[x.value].prority() if x.value in all_poi_functions else x.value)  # type: ignore
    log.print("Calc pois from subregion id", {l.name for l in locations})
    ### DENSE###
    if Location.Dens_axis in locations and 2 in _vert_ids and (2, Location.Dens_axis.value) not in poi:
        a = subreg * vert.extract_label(2)
        bb = a.compute_crop()
        a = a.apply_crop(bb)
        s = [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]
        if a.sum() != 0:
            strategy_extreme_points(poi, a, location=Location.Dens_axis, direction=["S", "P"], vert_id=2, subreg_id=s, bb=bb)
    ### STEP 2 (Other global non centroid poi; Spinal heights ###

    if Location.Spinal_Canal in locations:
        locations.remove(Location.Spinal_Canal)
        subregs_ids = subreg.unique()
        _a = Location.Spinal_Canal.value in subregs_ids or Location.Spinal_Canal.value in subregs_ids
        if _a and Location.Spinal_Canal.value not in poi.keys_subregion():
            poi = calc_center_spinal_cord(poi, subreg, add_dense=True)
    if Location.Spinal_Cord in locations:
        locations.remove(Location.Spinal_Cord)
        subregs_ids = subreg.unique()
        v = Location.Spinal_Cord.value
        if (v in subregs_ids or Location.Spinal_Cord.value in subregs_ids) and v not in poi.keys_subregion():
            poi = calc_center_spinal_cord(
                poi,
                subreg,
                source_subreg_point_id=Location.Vertebra_Disc,
                subreg_id=Location.Spinal_Cord,
                add_dense=True,
                intersection_target=[Location.Spinal_Cord],
            )

    if Location.Spinal_Canal_ivd_lvl in locations:
        locations.remove(Location.Spinal_Canal_ivd_lvl)
        subregs_ids = subreg.unique()
        v = Location.Spinal_Canal_ivd_lvl.value
        if (v in subregs_ids or Location.Spinal_Cord.value in subregs_ids) and v not in poi.keys_subregion():
            poi = calc_center_spinal_cord(
                poi, subreg, source_subreg_point_id=Location.Vertebra_Disc, subreg_id=Location.Spinal_Canal_ivd_lvl, add_dense=True
            )

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
    body_spline, body_spline_der = poi_iso.fit_spline(location=spline_subreg_point_id, vertebra=False)
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
