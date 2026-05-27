from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator

from TPTBox import NII, POI, Logger_Interface, Print_Logger
from TPTBox.core.compat import zip_strict
from TPTBox.core.poi_fun._help import sacrum_w_o_arcus, to_local_np
from TPTBox.core.poi_fun.pixel_based_point_finder import get_direction, get_extreme_point_by_vert_direction
from TPTBox.core.poi_fun.ray_casting import max_distance_ray_cast_convex_poi, shift_point
from TPTBox.core.vert_constants import DIRECTIONS, Location

_log = Print_Logger()


##### Extreme Points ####
def strategy_extreme_points(
    poi: POI,
    current_subreg: NII,
    location: Location,
    direction: Sequence[DIRECTIONS] | DIRECTIONS,
    vert_id: int,
    subreg_id: Location | list[Location],
    bb: tuple[slice, slice, slice] | None,
    log: Logger_Interface = _log,
) -> None:
    """Strategy: place a POI at the anatomically extreme voxel of a subregion.

    Extracts the label(s) ``subreg_id`` from ``current_subreg``, finds the
    voxel most extreme in ``direction`` relative to the vertebra orientation,
    and stores the result in ``poi``.

    Args:
        poi: ``POI`` object that will receive the new landmark at
            ``(vert_id, location.value)``.
        current_subreg: Cropped subregion ``NII`` aligned to the vertebra
            bounding box.
        location: Target ``Location`` enum member identifying the landmark slot.
        direction: Anatomical direction(s) to optimise.  Accepts a single
            direction string, a sequence of strings, or weighted
            ``(direction, weight)`` tuples.
        vert_id: Vertebra identifier (integer label).
        subreg_id: Label(s) to extract from ``current_subreg`` before
            searching for the extreme point.
        bb: Bounding-box slices mapping local to global voxel coordinates.
        log: Logger for warning and error messages.
    """
    region = current_subreg.extract_label(subreg_id)
    if region.sum() == 0:
        log.on_fail(f"reg={vert_id},subreg={subreg_id} is missing (extreme_points); {current_subreg.unique()}")
        return
    # extreme_point = get_extreme_point(poi, region, vert_id, bb, anti_point)

    extreme_point = get_extreme_point_by_vert_direction(poi, region, vert_id, direction)
    if extreme_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip_strict(bb, extreme_point))


##### Ray CASTING ####
def strategy_line_cast(
    poi: POI,
    vert_id: int,
    current_subreg: NII,
    location: Location,
    start_point: Location | np.ndarray,
    regions_loc: list[Location] | Location,
    normal_vector_points: tuple[Location, Location] | DIRECTIONS,
    bb: tuple[slice, slice, slice] | None,
    log: Logger_Interface = _log,
) -> None:
    """Strategy: place a POI at the ray-cast exit point of a subregion surface.

    Extracts the combined label mask for ``regions_loc``, then casts a ray
    from ``start_point`` along ``normal_vector_points`` using
    :func:`~TPTBox.core.poi_fun.ray_casting.max_distance_ray_cast_convex_poi`
    and stores the resulting boundary coordinate in ``poi``.

    Args:
        poi: ``POI`` object that will receive the new landmark at
            ``(vert_id, location.value)``.
        vert_id: Vertebra identifier (integer label).
        current_subreg: Cropped subregion ``NII`` aligned to the vertebra
            bounding box.
        location: Target ``Location`` enum member identifying the landmark slot.
        start_point: Ray origin — a ``Location`` member resolved from ``poi`` or
            a pre-computed numpy coordinate.
        regions_loc: Label(s) to extract from ``current_subreg`` as the region
            to cast the ray into.
        normal_vector_points: Ray direction, given either as a ``DIRECTIONS``
            string or a pair of ``Location`` members whose difference defines
            the direction.
        bb: Bounding-box slices mapping local to global voxel coordinates.
        log: Logger for warning and error messages.
    """
    region = current_subreg.extract_label(regions_loc)
    # if legacy_code:
    #    horizontal_plane_landmarks_old(poi, region, label_id, bb, log)
    # else:
    extreme_point = max_distance_ray_cast_convex_poi(poi, region, vert_id, bb, normal_vector_points, start_point, log=log)
    if extreme_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip_strict(bb, extreme_point))


#### find corner ####
def strategy_find_corner(
    poi: POI,
    current_subreg: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    location: Location,
    vec1: Location,
    vec2: Location,
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    shift_direction: DIRECTIONS | None = None,
) -> None:
    """Strategy: place a POI at the corner of a vertebral body defined by two direction vectors.

    Shifts the start point laterally if ``shift_direction`` is given, then
    uses a 2-D bisection search within the vertebral body (``Vertebra_Corpus``
    and ``Vertebra_Corpus_border``) to find the furthest reachable corner in
    the directions of ``vec1`` and ``vec2``.

    Args:
        poi: ``POI`` object that will receive the new landmark at
            ``(vert_id, location.value)``.
        current_subreg: Cropped subregion ``NII`` aligned to the vertebra
            bounding box.
        vert_id: Vertebra identifier (integer label).
        bb: Bounding-box slices mapping local to global voxel coordinates.
        location: Target ``Location`` enum member identifying the landmark slot.
        vec1: First direction ``Location`` (e.g. anterior or superior edge
            point).
        vec2: Second direction ``Location`` (e.g. superior or inferior edge
            point).
        start_point: Ray origin — a ``Location`` member resolved from ``poi``
            or a pre-computed numpy coordinate.  Defaults to
            ``Location.Vertebra_Corpus``.
        log: Logger for warning and error messages.
        shift_direction: Optional lateral shift direction applied to
            ``start_point`` before the corner search (e.g. ``"R"`` for right).
            ``None`` skips the shift.
    """
    if vert_id in sacrum_w_o_arcus:
        return

    start_point = shift_point(poi, vert_id, bb, start_point, direction=shift_direction, log=log)
    location2 = [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]
    current_subreg = current_subreg.extract_label(location2, keep_label=False)
    if current_subreg.sum() == 0:
        return
    corner_point = _find_corner_point(poi, current_subreg, vert_id, bb, start_point, vec1=vec1, vec2=vec2, log=log, location=location)

    if corner_point is None:
        return
    poi[vert_id, location.value] = tuple(a.start + b for a, b in zip_strict(bb, corner_point))


# @timing
def _find_corner_point(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice] | None,
    start_point: Location | np.ndarray,
    vec1: Location,
    vec2: Location,
    log: Logger_Interface = _log,
    delta: float = 0.00000005,
    location: Location | None = None,
) -> np.ndarray | None:
    """Bisection search for the corner point furthest along two direction vectors inside a region."""
    # Convert start point and vectors to local numpy coordinates
    start_point_np = to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None
    if (vert_id, vec1.value) not in poi.keys():
        log.on_fail(f"find_corner_point - point missing {vert_id=} {vec1.value=} {location=},{poi.keys()}")
        return
    if (vert_id, vec2.value) not in poi.keys():
        log.on_fail(f"find_corner_point - point missing {(vert_id, vec2.value)=} {location=}")
        return
    v1 = to_local_np(vec1, bb, poi, vert_id, log) - start_point_np
    v2 = to_local_np(vec2, bb, poi, vert_id, log) - start_point_np

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
        factor1 = factor1 * 0.9
        factor2 = factor2 * 0.7
    v1_n = 1.0
    v2_n = 0.4
    f1 = f2 = 0

    # Refine factors using delta
    while delta < (v1_n + v2_n) / 2:
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


### ligament_attachment_point_flava ###
def strategy_ligament_attachment_point_flava(
    poi: POI,
    current_subreg: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    location: Location,
    goal: Location | np.ndarray,
    log: Logger_Interface = _log,
    delta: float = 0.0000001,
    shift_direction: Literal["S", "I"] = "S",
    dir2: Literal["A"] = "A",
) -> None:
    """Strategy: place a ligamentum-flavum attachment POI on the arcus / spinosus surface.

    Starting from the spinosus process (or arcus if spinosus is absent),
    performs a 2-D bisection search moving in ``shift_direction`` and ``dir2``
    while staying inside the arcus/spinosus mask and approaching ``goal``.
    The search stops when the bracket width falls below ``delta``.

    Args:
        poi: ``POI`` object that will receive the new landmark at
            ``(vert_id, location)``.
        current_subreg: Cropped subregion ``NII`` aligned to the vertebra
            bounding box (must contain arcus and spinosus labels).
        vert_id: Vertebra identifier (integer label).
        bb: Bounding-box slices mapping local to global voxel coordinates.
        location: Target ``Location`` enum member identifying the landmark slot.
        goal: Target ``Location`` or pre-computed coordinate toward which the
            search advances (e.g. an anterior longitudinal ligament point).
        log: Logger for warning and error messages.
        delta: Convergence threshold for the bisection loop.
            Defaults to 0.0000001.
        shift_direction: Primary direction of search along the arcus
            (``"S"`` = superior, ``"I"`` = inferior).  Defaults to ``"S"``.
        dir2: Secondary search direction (always anterior, ``"A"``).
            Defaults to ``"A"``.
    """
    if vert_id in sacrum_w_o_arcus:
        return
    try:
        normal_vector1 = get_direction(shift_direction, poi, vert_id)  # / np.array(poi.zoom)
        v1 = normal_vector1 / norm(normal_vector1)

        normal_vector2 = get_direction(dir2, poi, vert_id)  # / np.array(poi.zoom)
        v2 = normal_vector2 / norm(normal_vector2)
    except KeyError:
        return
    if isinstance(goal, Location):
        start_point_np = to_local_np(Location.Spinosus_Process, bb, poi, vert_id, log, verbose=False)
        if start_point_np is None:
            start_point_np = to_local_np(Location.Arcus_Vertebrae, bb, poi, vert_id, log, verbose=True)
    else:
        start_point_np = goal

    goal_np = to_local_np(goal, bb, poi, vert_id, log) if isinstance(goal, Location) else goal
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
    poi[vert_id, location] = tuple(x + y.start for x, y in zip(coords, bb))


#### shifted_line_cast ####
def strategy_shifted_line_cast(
    poi: POI,
    current_subreg: NII,
    location: Location,
    vert_id: int,
    bb: tuple[slice, slice, slice] | None,
    regions_loc: list[Location] | Location,
    normal_vector_points: tuple[Location, Location] | DIRECTIONS,
    start_point: Location = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    direction: DIRECTIONS = "R",
    do_shift: bool = True,
) -> None:
    """Strategy: lateral-shift a start point then ray-cast to the subregion surface.

    Optionally displaces ``start_point`` laterally by a vertebra-width fraction
    via :func:`~TPTBox.core.poi_fun.ray_casting.shift_point`, then delegates to
    :func:`strategy_line_cast` to find the boundary along ``normal_vector_points``.

    Args:
        poi: ``POI`` object that will receive the new landmark at
            ``(vert_id, location.value)``.
        current_subreg: Cropped subregion ``NII`` aligned to the vertebra
            bounding box.
        location: Target ``Location`` enum member identifying the landmark slot.
        vert_id: Vertebra identifier (integer label).
        bb: Bounding-box slices mapping local to global voxel coordinates.
        regions_loc: Label(s) to extract from ``current_subreg`` as the target
            region for the ray cast.
        normal_vector_points: Ray direction — a ``DIRECTIONS`` string or a pair
            of ``Location`` members.
        start_point: Ray origin before shifting.  Defaults to
            ``Location.Vertebra_Corpus``.
        log: Logger for warning and error messages.
        direction: Lateral direction for the shift (``"R"`` or ``"L"``).
            Defaults to ``"R"``.
        do_shift: When ``False``, skip the lateral shift and use ``start_point``
            directly.  Defaults to ``True``.
    """
    if vert_id in sacrum_w_o_arcus:
        return
    try:
        cords = shift_point(
            poi,
            vert_id,
            bb,
            start_point,
            direction=direction if do_shift else None,
            log=log,
        )
    except KeyError:
        log.on_fail(f"region={vert_id},subregion={location} is missing. (shifted_line_cast)")
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
