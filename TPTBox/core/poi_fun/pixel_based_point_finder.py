from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from TPTBox import NII, POI, Logger_Interface, Print_Logger
from TPTBox.core.poi_fun._help import to_local_np
from TPTBox.core.poi_fun.vertebra_direction import _get_sub_array_by_direction, get_direction, get_vert_direction_matrix
from TPTBox.core.vert_constants import COORDINATE, DIRECTIONS, Location

_log = Print_Logger()


#### Pixel Level functions ####
def get_nearest_neighbor(
    p: np.ndarray,
    sr_msk: np.ndarray,
    region_label: int,
) -> np.ndarray:
    """Return the voxel in a label mask that is closest to a query point.

    Args:
        p: Query point as a 1-D or column array of shape ``(3,)`` or ``(3, 1)``.
        sr_msk: 3-D integer array containing segmentation labels.
        region_label: Label value whose voxels are searched for the nearest
            neighbour.

    Returns:
        1-D integer array ``(x, y, z)`` of the voxel in ``sr_msk`` with label
        ``region_label`` that minimises the Euclidean distance to ``p``.
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
) -> tuple[int, ...] | None:
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
    two_sided: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Perform pixel-level ray casting in a region using POI-derived direction.

    Resolves the ray start coordinate and direction vector from the ``POI``
    object, then delegates to :func:`~TPTBox.core.poi_fun.ray_casting.ray_cast_pixel_lvl`
    to enumerate all voxels along the ray at integer resolution.

    Args:
        poi: ``POI`` object providing landmark coordinates and orientation.
        region: ``NII`` image defining the volume shape for boundary clipping.
        vert_id: Vertebra identifier (integer label).
        bb: Bounding-box slices mapping global POI coordinates to local
            sub-volume indices.
        normal_vector_points: Ray direction — a ``DIRECTIONS`` string resolved
            from the vertebra orientation, or a 2-tuple of ``Location`` members
            whose difference defines the direction.  Defaults to ``"R"``.
        start_point: Ray origin — a ``Location`` member resolved from ``poi``
            or a pre-computed numpy coordinate.  Defaults to
            ``Location.Vertebra_Corpus``.
        log: Logger for warning and error messages.
        two_sided: Cast the ray in both directions.  Defaults to ``False``.

    Returns:
        ``(plane_coords, arange)`` where ``plane_coords`` is an integer array of
        shape ``(N, 3)`` with visited voxel indices and ``arange`` is a float
        array of distance values, or ``(None, None)`` on failure.
    """
    from TPTBox.core.poi_fun.ray_casting import ray_cast_pixel_lvl

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
    vert_id: int,
    direction: Sequence[DIRECTIONS] | DIRECTIONS | tuple[DIRECTIONS, float] | Sequence[tuple[DIRECTIONS, float]] = "I",
) -> np.ndarray | None:
    """Find the voxel in a binary region that is most extreme along a vertebra-relative direction.

    Projects all nonzero voxels in ``region`` into the vertebra PIR frame and
    returns the voxel index with the maximum weighted score along the requested
    direction(s).

    Args:
        poi: ``POI`` object providing the vertebra orientation matrix for
            ``vert_id``.
        region: Binary ``NII`` image (value 1 for foreground) aligned to the
            same space as ``poi``.
        vert_id: Vertebra identifier used to look up the orientation matrix.
        direction: Anatomical direction(s) to optimise.  May be:

            * A single direction letter (e.g. ``"I"``).
            * A sequence of direction letters (e.g. ``["P", "I"]``).
            * A ``(direction, weight)`` tuple or a sequence of such tuples for
              weighted combinations.

            Defaults to ``"I"``.

    Returns:
        Integer voxel index array ``(x, y, z)`` of the most extreme point, or
        ``None`` if the orientation matrix cannot be retrieved.

    Note:
        ``region`` must contain binary values (1 for foreground, 0 for
        background).  Zoom scaling from ``poi`` is applied to ensure
        physically meaningful distances.
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


def project_pois_onto_segmentation_surface(
    poi: POI,
    seg: NII,
    connectivity: int = 1,
    dilated_surface: bool = False,
) -> POI:
    """Projects points of interest (POI) onto a segmentation surface.

    This function computes the surface points of a segmentation volume and
    projects the given points of interest (POI) onto these computed surface points.

    Args:
        poi (POI): The points of interest to be projected.
        seg (NII): A segmentation volume object containing the target surface.
        connectivity (int, optional): The connectivity level for defining the surface.
            Default is 1, where 1 denotes face-connectivity.
        dilated_surface (bool, optional): Whether to compute a dilated version of the
            surface, expanding the surface area. Default is False.

    Returns:
        POI: The points of interest projected onto the segmentation surface.
    """
    point_set = seg.compute_surface_points(
        connectivity=connectivity,
        dilated_surface=dilated_surface,
    )
    return project_pois_onto_set_of_points(poi, point_set)


def project_pois_onto_set_of_points(poi: POI, point_set: list[COORDINATE]) -> POI:
    """Projects points of interest (POI) onto the nearest points in a given set.

    For each point in the POI, this function finds the closest point in the
    provided point set and updates the POI coordinates to align with the nearest points.

    Args:
        poi (POI): The points of interest to be projected.
        point_set (list[COORDINATE]): A list of coordinates representing the target points
            for projection.

    Returns:
        POI: The updated POI with coordinates projected onto the nearest points in
        the provided point set.
    """
    poi_n = poi.copy()
    point_arr = np.asarray(point_set)

    for r, s, c in poi.items():
        distance_to_point = cdist_to_point(c, point_arr)
        new_coord = point_arr[np.argmin(distance_to_point)]
        poi_n[r, s] = new_coord

    return poi_n


def cdist_to_point(
    point: COORDINATE | np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distances from a single point to every row in an array.

    Args:
        point: Query point as a sequence or 1-D array of length 3.
        a: Array of shape ``(N, 3)`` containing the candidate points.

    Returns:
        1-D float array of shape ``(N,)`` with the Euclidean distance from
        ``point`` to each row of ``a``.
    """
    return cdist([point], a)[0]
