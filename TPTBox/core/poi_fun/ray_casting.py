import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator

from TPTBox import NII, POI, Print_Logger, Vertebra_Instance
from TPTBox.core.poi_fun._help import sacrum_w_o_arcus, to_local_np
from TPTBox.core.poi_fun.pixel_based_point_finder import get_direction
from TPTBox.core.vert_constants import COORDINATE, DIRECTIONS, Location
from TPTBox.logger.log_file import Logger_Interface

_log = Print_Logger()


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def max_distance_ray_cast_convex(
    region: NII,
    start_coord: COORDINATE | np.ndarray,
    direction_vector: np.ndarray,
    acc_delta: float = 0.00005,
):
    start_point_np = np.asarray(start_coord)
    if start_point_np is None:
        return None

    """Convex assumption!"""
    # Compute a normal vector, that defines the plane direction
    normal_vector = np.asarray(direction_vector)
    normal_vector = normal_vector / norm(normal_vector)
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


def ray_cast_pixel_lvl(
    start_point_np: np.ndarray,
    normal_vector: np.ndarray,
    shape: np.ndarray | tuple[int, ...],
    two_sided=False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    normal_vector = unit_vector(normal_vector)

    def _calc_pixels(normal_vector, start_point_np):
        # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
        start_point_np = start_point_np.copy()
        num_pixel = np.abs(np.floor(np.max((np.array(shape) - start_point_np) / normal_vector))).item()
        arange = np.arange(0, min(num_pixel, 1000), step=1, dtype=float)
        coords = [start_point_np[i] + normal_vector[i] * arange for i in [0, 1, 2]]

        # Clip coordinates to region bounds
        for i in [0, 1, 2]:
            cut_off = (shape[i] <= np.floor(coords[i])).sum()
            if cut_off == 0:
                cut_off = (np.floor(coords[i]) <= 0).sum()
            if cut_off != 0:
                coords = [c[:-cut_off] for c in coords]
                arange = arange[:-cut_off]
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


def add_ray_to_img(
    start_point: np.ndarray | COORDINATE,
    normal_vector: np.ndarray,
    seg: NII,
    add_to_img=True,
    inplace=False,
    value=0,
    dilate=1,
):
    start_point = np.array(start_point)
    plane_coords, arange = ray_cast_pixel_lvl(start_point, normal_vector, shape=seg.shape)
    if plane_coords is None:
        return None
    selected_arr = np.zeros(seg.shape, dtype=seg.dtype)
    selected_arr[plane_coords[..., 0], plane_coords[..., 1], plane_coords[..., 2]] = arange if value == 0 else value
    ray = seg.set_array(selected_arr)
    if dilate != 0:
        ray.dilate_msk_(dilate)
    if add_to_img:
        if not inplace:
            seg = seg.copy()
        seg[ray != 0] = ray[ray != 0]
        return seg
    return ray


def add_spline_to_img(
    seg: NII,
    poi: "POI",
    location=50,
    add_to_img=True,
    override_seg=True,
    value=100,
    dilate=2,
):
    cor, _ = poi.fit_spline(location=location, vertebra=True)
    spline = seg.copy() * 0
    # spline.rescale_()
    for x, y, z in cor:
        spline[round(x), round(y), round(z)] = value
    spline.dilate_msk_(dilate)
    # spline.resample_from_to_(a)
    if add_to_img:
        if override_seg:
            seg[spline != 0] = spline[spline != 0]
        else:
            cond = np.logical_and(spline != 0, seg == 0)
            seg[cond] = spline[cond]
        return seg
    return spline


def shift_point(
    poi: POI,
    vert_id: int,
    bb,
    start_point: Location = Location.Vertebra_Corpus,
    direction: DIRECTIONS | None = "R",
    log: Logger_Interface = _log,
):
    if vert_id in sacrum_w_o_arcus:
        return

    if direction is None:
        return to_local_np(start_point, bb, poi, vert_id, log)
    sup_articular_right = sup_articular_left = None
    if Vertebra_Instance.is_sacrum(vert_id):
        if (vert_id, Location.Costal_Process_Left) not in poi:
            return
        sup_articular_right = to_local_np(Location.Costal_Process_Right, bb, poi, vert_id, log)
        sup_articular_left = to_local_np(Location.Costal_Process_Left, bb, poi, vert_id, log)
        factor = 6.0
    if sup_articular_left is None or sup_articular_right is None:
        sup_articular_right = to_local_np(Location.Superior_Articular_Right, bb, poi, vert_id, log, verbose=False)
        sup_articular_left = to_local_np(Location.Superior_Articular_Left, bb, poi, vert_id, log, verbose=False)
        factor = 3.0
    if sup_articular_left is None or sup_articular_right is None:
        sup_articular_right = to_local_np(Location.Inferior_Articular_Right, bb, poi, vert_id, log)
        sup_articular_left = to_local_np(Location.Inferior_Articular_Left, bb, poi, vert_id, log)
        factor = 2.0
    if sup_articular_left is None or sup_articular_right is None:
        return
    if vert_id <= 11:
        factor *= (12 - vert_id) / 11 + 1
    vertebra_width = np.linalg.norm(sup_articular_right - sup_articular_left)
    shift = vertebra_width / factor
    normal_vector = get_direction(direction, poi, vert_id)  # / np.array(poi.zoom)
    normal_vector = normal_vector / norm(normal_vector)
    start_point_np = to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None
    return start_point_np + normal_vector * shift


def max_distance_ray_cast_convex_poi(
    poi: POI,
    region: NII,
    vert_id: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | DIRECTIONS = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    acc_delta: float = 0.00005,
):
    start_point_np = to_local_np(start_point, bb, poi, vert_id, log) if isinstance(start_point, Location) else start_point
    if start_point_np is None:
        return None

    """Convex assumption!"""
    # Compute a normal vector, that defines the plane direction
    if isinstance(normal_vector_points, str):
        try:
            normal_vector = get_direction(normal_vector_points, poi, vert_id)
        except KeyError:
            if vert_id not in sacrum_w_o_arcus:
                log.on_fail(f"region={vert_id},DIRECTIONS={normal_vector_points} is missing")
            return
            # raise KeyError(f"region={label},subregion={loc.value} is missing.")
        normal_vector /= norm(normal_vector)

    else:
        try:
            b = to_local_np(normal_vector_points[1], bb, poi, vert_id, log)
            if b is None:
                return None
            a = to_local_np(normal_vector_points[0], bb, poi, vert_id, log)
            normal_vector = b - a
            normal_vector = normal_vector / norm(normal_vector)
            log.on_fail(f"ray_cast used with old normal_vector_points {normal_vector_points}")
        except TypeError as e:
            log.on_fail("TypeError", e)
            return None
    return max_distance_ray_cast_convex(region, start_point_np, normal_vector, acc_delta)
