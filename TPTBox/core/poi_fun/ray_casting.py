from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

from TPTBox import NII, POI, Print_Logger, Vertebra_Instance
from TPTBox.core.poi_fun._help import sacrum_w_o_arcus, to_local_np
from TPTBox.core.poi_fun.pixel_based_point_finder import get_direction
from TPTBox.core.vert_constants import COORDINATE, DIRECTIONS, Location
from TPTBox.logger.log_file import Logger_Interface

_log = Print_Logger()


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


# @njit(fastmath=True)
def trilinear_interpolate(volume, x, y, z):
    xi, yi, zi = int(x), int(y), int(z)
    if xi < 0 or yi < 0 or zi < 0 or xi >= volume.shape[0] - 1 or yi >= volume.shape[1] - 1 or zi >= volume.shape[2] - 1:
        return 0.0

    xd, yd, zd = x - xi, y - yi, z - zi
    c000 = volume[xi, yi, zi]
    c100 = volume[xi + 1, yi, zi]
    c010 = volume[xi, yi + 1, zi]
    c110 = volume[xi + 1, yi + 1, zi]
    c001 = volume[xi, yi, zi + 1]
    c101 = volume[xi + 1, yi, zi + 1]
    c011 = volume[xi, yi + 1, zi + 1]
    c111 = volume[xi + 1, yi + 1, zi + 1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd


# @njit(fastmath=True)
def max_distance_ray_cast_convex_npfast(
    region_array: np.ndarray,
    start_coord: np.ndarray,
    direction_vector: np.ndarray,
    acc_delta=0.05,
):
    # Normalize direction
    norm_vec = direction_vector / np.sqrt((direction_vector**2).sum())

    # Quick exit if start point is outside
    if trilinear_interpolate(region_array, *start_coord) <= 0.5:
        return np.array(start_coord)

    min_v = 0.0
    max_v = np.sum(region_array.shape)
    delta = max_v - min_v

    while delta > acc_delta:
        mid = 0.5 * (max_v + min_v)
        x = start_coord[0] + norm_vec[0] * mid
        y = start_coord[1] + norm_vec[1] * mid
        z = start_coord[2] + norm_vec[2] * mid
        val = trilinear_interpolate(region_array, x, y, z)
        if val > 0.5:
            min_v = mid
        else:
            max_v = mid
        delta = max_v - min_v

    dist = 0.5 * (min_v + max_v)
    return np.array(
        [
            start_coord[0] + norm_vec[0] * dist,
            start_coord[1] + norm_vec[1] * dist,
            start_coord[2] + norm_vec[2] * dist,
        ]
    )


def max_distance_ray_cast_convex(
    region: NII,
    start_coord: COORDINATE | np.ndarray,
    direction_vector: np.ndarray,
    acc_delta: float = 0.00005,
):
    """
    Computes the maximum distance a ray can travel inside a convex region before exiting.

    Parameters:
    region (NII): The region of interest as a 3D NIfTI image.
    start_coord (COORDINATE | np.ndarray): The starting coordinate of the ray.
    direction_vector (np.ndarray): The direction vector of the ray.
    acc_delta (float, optional): The accuracy threshold for bisection search. Default is 0.00005.

    Returns:
    np.ndarray: The exit coordinate of the ray within the region.
    """
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
    poi: POI,
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
    bb: tuple[slice, slice, slice] | None,
    normal_vector_points: tuple[Location, Location] | DIRECTIONS = "R",
    start_point: Location | np.ndarray = Location.Vertebra_Corpus,
    log: Logger_Interface = _log,
    acc_delta: float = 0.00005,
):
    """
    Computes the maximum distance a ray can travel inside a convex region for a point of interest (POI).

    Parameters:
    poi (POI): The point of interest.
    region (NII): The region of interest as a 3D NIfTI image.
    vert_id (int): The vertebra identifier.
    bb (tuple[slice, slice, slice] | None): Bounding box constraints.
    normal_vector_points (tuple[Location, Location] | DIRECTIONS, optional):
        Two locations defining the normal vector or a predefined direction. Default is "R".
    start_point (Location | np.ndarray, optional):
        The starting location or coordinate of the ray. Default is Location.Vertebra_Corpus.
    log (Logger_Interface, optional): Logger instance for error handling. Default is _log.
    acc_delta (float, optional): The accuracy threshold for bisection search. Default is 0.00005.

    Returns:
    np.ndarray: The exit coordinate of the ray within the region.
    """
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


def calculate_pca_normal_np(segmentation: np.ndarray, pca_component, zoom=None, verbose=False):
    """
    Computes the normal vector of a segmented region using Principal Component Analysis (PCA).

    Parameters:
    ----------
    segmentation : np.ndarray
        A binary mask where nonzero values indicate the segmented region.
    pca_component : int, optional
        The principal component index to return as the normal vector.
        - `0`: The primary axis (direction of greatest variance, often the main elongation).
        - `1`: The secondary axis (orthogonal to the primary, capturing the second-most variance).
        - `2`: The third axis (typically the normal direction to the structure in 3D).
    zoom : tuple or array-like, optional
        If provided, scales the normal vector by the inverse of the voxel size to account for anisotropic resolution.
    verbose : bool, optional
        If True, prints the principal component vectors for debugging.

    Returns:
    -------
    normal_vector : np.ndarray
        The selected principal component as a normal vector.

    Usage:
    ------
    Use `pca_component=2` when you want the normal to a surface-like structure.
    If analyzing an elongated structure (e.g., a vessel or bone), `pca_component=0` gives the primary axis,
    while `pca_component=1` provides the secondary direction.
    """
    # Get indices of segmented region (assuming segmentation is a binary mask)
    points = np.argwhere(segmentation > 0)
    # Perform PCA to find the principal axes
    pca = PCA(n_components=3)
    pca.fit(points)
    # First, second, and third principal components
    if verbose:
        print(f"Main Axis (PC1): {pca.components_[0]}")
        print(f"Secondary Axis (PC2): {pca.components_[1]}")
        print(f"Third Axis (PC3): {pca.components_[2]}")
    normal_vector = pca.components_[pca_component]
    if zoom is not None:
        normal_vector = normal_vector / np.array(zoom)
    return normal_vector


def set_label_above_3_point_plane(
    array: NII | np.ndarray,
    p1,
    p2,
    p3,
    value=0,
    invert: Literal[-1, 1] = 1,
    mask: np.ndarray | NII | bool = True,
    inplace=False,
):
    """
    Set all values in a 3D array above a plane defined by three non-collinear points to a specified value.

    Parameters:
    -----------
    array : NII | np.ndarray
        A 3D NumPy array or an NII object representing the volume data.
    p1, p2, p3 : array-like
        Three (x, y, z) points defining the plane.
    value : int or float, optional (default=0)
        The value to set for all elements above the plane.
    inf : Literal[-1, 1], optional (default=1)
        Controls the direction of "above":
        - `1` means values superior to the plane (default).
        - `-1` means values inferior to the plane.
        If the input is an NII object with an inferior-superior orientation, this will be adjusted accordingly.

    Returns:
    --------
    np.ndarray
        The modified 3D array with values set above the plane.

    Notes:
    ------
    - The plane is defined by the equation `ax + by + cz + d = 0`, where `(a, b, c)` is the normal vector.
    - Uses `meshgrid` to construct a 3D grid and determine which values lie above the plane.

    Example:
    --------
    ```python
    import numpy as np

    data = np.random.rand(300, 300, 300)
    p1 = [100, 100, 50]
    p2 = [100, 50, 100]
    p3 = [50, 100, 100]

    result = set_above_3_point_plane(data, p1, p2, p3, value=0)
    ```
    """
    import numpy as np

    if not inplace:
        array = array.copy()
    if isinstance(array, NII) and array.orientation[array.get_axis("S")] == "I":
        # for NII inf = 1 means Superior
        invert *= -1

    # Define three 3D points that form a plane
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Compute the normal vector of the plane
    normal = np.cross(p2 - p1, p3 - p1)
    a, b, c = normal
    d = -np.dot(normal, p1)

    # Define a 3D grid of points
    shape = array.shape
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")

    # Compute the plane equation for each (x, y)
    plane_z = (-a * x - b * y - d) / c

    # Create the 3D array and set values above the plane to 0
    array[np.logical_and(mask, z * invert > plane_z)] = value
    return array
