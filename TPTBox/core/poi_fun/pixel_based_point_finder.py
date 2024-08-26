from collections.abc import Sequence

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from TPTBox import NII, POI, Logger_Interface, Print_Logger
from TPTBox.core.poi_fun._help import to_local_np
from TPTBox.core.poi_fun.vertebra_direction import _get_sub_array_by_direction, get_direction, get_vert_direction_matrix
from TPTBox.core.vert_constants import DIRECTIONS, Location

_log = Print_Logger()


#### Pixel Level functions ####
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
    plane_coords, arange = ray_cast_pixel_level_from_poi(
        poi,
        region,
        vert_id,
        bb,
        normal_vector_points,
        start_point,
        log=log,
        two_sided=two_sided,
    )
    if plane_coords is None:
        return None
    selected_arr = np.zeros(region.shape)
    selected_arr[plane_coords[..., 0], plane_coords[..., 1], plane_coords[..., 2]] = arange
    selected_arr = selected_arr * region.get_array()
    out = tuple(np.unravel_index(np.argmax(selected_arr, axis=None), selected_arr.shape))
    return out


def ray_cast_pixel_level_from_poi(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | DIRECTIONS = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    two_sided=False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    from TPTBox.core.poi_fun.ray_casting import ray_cast_pixel_lvl

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
    start_point_np = to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None, None

    # Compute a normal vector, that defines the plane direction
    if isinstance(normal_vector_points, str):
        normal_vector = get_direction(normal_vector_points, poi, vert_id)  # / np.array(poi.zoom)
        normal_vector = normal_vector / norm(normal_vector)
    else:
        try:
            b = to_local_np(normal_vector_points[1], bb, poi, vert_id, log)
            if b is None:
                return None, None
            a = to_local_np(normal_vector_points[0], bb, poi, vert_id, log)
            normal_vector = b - a
            normal_vector = normal_vector / norm(normal_vector)
            log.on_fail(f"ray_cast used with old normal_vector_points {normal_vector_points}")
        except TypeError as e:
            log.on_fail("TypeError", e)
            return None, None
    return ray_cast_pixel_lvl(start_point_np, normal_vector, region.shape, two_sided=two_sided)


def get_extreme_point_by_vert_direction(
    poi: POI,
    region: NII,
    vert_id,
    direction: Sequence[DIRECTIONS] | DIRECTIONS | tuple[DIRECTIONS, float] | Sequence[tuple[DIRECTIONS, float]] = "I",
):
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
    direction_: Sequence[DIRECTIONS] | Sequence[tuple[DIRECTIONS, float]] = direction if isinstance(direction, Sequence) else (direction,)  # type: ignore

    direction_weight: list[tuple[DIRECTIONS, float]] = [(d[0], 1.0) if isinstance(d, str) else (d[0], d[1]) for d in direction_]  # type: ignore
    assert poi.orientation == region.orientation, (poi.orientation, region.orientation)
    try:
        to_reference_frame, from_reference_frame = get_vert_direction_matrix(poi, vert_id=vert_id, to_pir=False)
    except KeyError:
        return None
    pc = np.stack(np.where(region.get_array() == 1))
    cords = to_reference_frame @ pc  # 3,n; 3 = P,I,R of vert
    a = [_get_sub_array_by_direction(d, cords) * poi.zoom[poi.get_axis(d)] * w for d, w in direction_weight]

    idx = np.argmax(sum(a))
    return pc[:, idx]
