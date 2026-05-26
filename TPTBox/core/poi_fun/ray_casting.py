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


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


# @njit(fastmath=True)
def trilinear_interpolate(volume: np.ndarray, x: float, y: float, z: float) -> float:
    """Perform trilinear interpolation of a 3D volume at a sub-voxel coordinate.

    Args:
        volume: 3D array to interpolate.
        x: Sub-voxel coordinate along the first axis.
        y: Sub-voxel coordinate along the second axis.
        z: Sub-voxel coordinate along the third axis.

    Returns:
        Interpolated scalar value at (x, y, z), or 0.0 if the point is outside
        the valid interior of the volume.
    """
    xi, yi, zi = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(z).astype(int)
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
    acc_delta: float = 0.00005,
) -> np.ndarray:
    """Find the exit point of a ray inside a convex region using bisection (fast path).

    Uses trilinear interpolation and bisection search to locate the last point
    inside the region along the given direction.  Prints a debug line at each
    bisection step (development helper — see ``max_distance_ray_cast_convex_np``
    for the production version without debug output).

    Args:
        region_array: Binary 3D numpy array where nonzero voxels define the region.
        start_coord: Starting coordinate ``(x, y, z)`` of the ray (must be inside
            the region).
        direction_vector: Direction of the ray; need not be a unit vector.
        acc_delta: Bisection stops when the bracket width falls below this value.
            Defaults to 0.00005.

    Returns:
        3-element array with the approximate exit coordinate along the ray.
        If ``start_coord`` is already outside the region the start coordinate is
        returned unchanged.
    """
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
        print(f"Raycast check at distance {mid:.2f}: value={val:.4f}")
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


def max_distance_ray_cast_convex_np(
    region: np.ndarray,
    start_coord: COORDINATE | np.ndarray,
    direction_vector: np.ndarray,
    acc_delta: float = 0.00005,
    max_v: int | None = None,
) -> np.ndarray | None:
    """Find the exit point of a ray inside a convex region (numpy array input).

    Uses ``RegularGridInterpolator`` and bisection to locate the boundary of a
    convex binary region along the given direction vector.

    Args:
        region: Binary 3D numpy array where values > 0.5 are considered inside.
        start_coord: Starting coordinate ``(x, y, z)`` of the ray.  If already
            outside the region the start coordinate is returned unchanged.
        direction_vector: Direction of the ray; will be normalised internally.
        acc_delta: Bisection convergence threshold in voxel units.
            Defaults to 0.00005.
        max_v: Upper bound for the bisection search (in voxels along the ray).
            Defaults to ``sum(region.shape)`` when ``None``.

    Returns:
        3-element numpy array with the approximate exit coordinate, or the start
        coordinate if it lies outside the region.  Returns ``None`` when
        ``start_coord`` is ``None``.
    """
    start_point_np = np.asarray(start_coord)
    if start_point_np is None:
        return None

    """Convex assumption!"""
    # Compute a normal vector, that defines the plane direction
    normal_vector = np.asarray(direction_vector)
    normal_vector = normal_vector / norm(normal_vector)
    # Create a function to interpolate within the mask array
    interpolator = RegularGridInterpolator([np.arange(region.shape[i]) for i in range(3)], region)

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
    if max_v is None:
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


def max_distance_ray_cast_convex(
    region: NII,
    start_coord: COORDINATE | np.ndarray,
    direction_vector: np.ndarray,
    acc_delta: float = 0.00005,
) -> np.ndarray | None:
    """Find the exit point of a ray inside a convex NII region.

    Wraps the underlying numpy logic by extracting the voxel array from an
    ``NII`` object and delegating to bisection-based ray casting.

    Args:
        region: ``NII`` object whose nonzero voxels define the convex region.
        start_coord: Starting coordinate ``(x, y, z)`` of the ray in voxel
            space.  If already outside the region, the start coordinate is
            returned unchanged.
        direction_vector: Direction of the ray; will be normalised internally.
        acc_delta: Bisection convergence threshold in voxel units.
            Defaults to 0.00005.

    Returns:
        3-element numpy array with the approximate exit coordinate, or the start
        coordinate if it lies outside the region.  Returns ``None`` when
        ``start_coord`` is ``None``.
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
    two_sided: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Cast a ray at pixel/voxel resolution and return all integer voxel coordinates along it.

    Traces the ray starting at ``start_point_np`` in the direction of
    ``normal_vector`` until it exits the volume defined by ``shape``.  When
    ``two_sided`` is ``True``, the ray is also cast in the opposite direction and
    the two halves are concatenated.

    Args:
        start_point_np: Starting voxel coordinate ``(x, y, z)`` of the ray.
        normal_vector: Direction vector of the ray; will be normalised to a
            unit vector internally.
        shape: Volume shape ``(dim0, dim1, dim2)`` used to clip the ray.
        two_sided: When ``True``, cast the ray in both directions and concatenate
            the results.  Defaults to ``False``.

    Returns:
        A tuple ``(plane_coords, arange)`` where

        * ``plane_coords`` is an integer array of shape ``(N, 3)`` with the
          voxel indices visited by the ray, or ``None`` on failure.
        * ``arange`` is a float array of shape ``(N,)`` with the distance
          values along the ray for each voxel, or ``None`` on failure.
    """
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
    add_to_img: bool = True,
    inplace: bool = False,
    value: int = 0,
    dilate: int = 1,
) -> NII | None:
    """Paint a ray into a segmentation NII image.

    Casts a ray from ``start_point`` along ``normal_vector`` at voxel
    resolution, fills the visited voxels with distance values (or a fixed
    ``value``), optionally dilates the ray, and optionally composites it onto
    the source segmentation.

    Args:
        start_point: Starting voxel coordinate of the ray.
        normal_vector: Direction of the ray.
        seg: Segmentation ``NII`` that defines the volume shape and is used as
            the base when ``add_to_img`` is ``True``.
        add_to_img: If ``True``, the ray is overlaid on ``seg`` and the
            composite image is returned.  If ``False``, only the ray image is
            returned.  Defaults to ``True``.
        inplace: Modify ``seg`` in place when ``add_to_img`` is ``True``.
            Defaults to ``False``.
        value: Fixed voxel value written along the ray.  When 0, distance
            values along the ray are used instead.  Defaults to 0.
        dilate: Morphological dilation radius applied to the ray image.
            Set to 0 to skip dilation.  Defaults to 1.

    Returns:
        The modified ``NII`` image, or ``None`` if the ray cast produced no
        valid coordinates.
    """
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
    location: int = 50,
    add_to_img: bool = True,
    override_seg: bool = True,
    value: int = 100,
    dilate: int = 2,
) -> NII:
    """Fit and paint a POI spline onto a segmentation NII image.

    Computes a cubic spline through the POI centroids at ``location``,
    converts it to voxel coordinates, paints each spline point with ``value``,
    dilates the result, and (optionally) composites it onto ``seg``.

    Args:
        seg: Base segmentation ``NII`` image used for shape and affine.
        poi: ``POI`` object from which the spline is computed.
        location: Subregion ID used as the spline anchor points.
            Defaults to 50 (``Vertebra_Corpus``).
        add_to_img: If ``True``, the spline is overlaid on ``seg``.
            Defaults to ``True``.
        override_seg: When overlaying, whether to overwrite existing nonzero
            voxels in ``seg``.  Defaults to ``True``.
        value: Voxel label written along the spline.  Defaults to 100.
        dilate: Morphological dilation radius.  Defaults to 2.

    Returns:
        The composite or standalone spline ``NII`` image.
    """
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
    bb: tuple[slice, slice, slice] | None,
    start_point: Location = Location.Vertebra_Corpus,
    direction: DIRECTIONS | None = "R",
    log: Logger_Interface = _log,
) -> np.ndarray | None:
    """Shift a POI start point laterally by a fraction of the vertebra width.

    Estimates the vertebra width from articular-process or costal-process POIs
    and returns a new coordinate displaced from ``start_point`` along
    ``direction`` by a scaled fraction of that width.  Sacrum vertebrae without
    arcus are skipped.

    Args:
        poi: ``POI`` object with pre-computed anatomical landmarks.
        vert_id: Vertebra identifier (integer label).
        bb: Bounding-box slices used to convert global POI coordinates to local
            coordinates within the cropped sub-volume.
        start_point: Location enum member or already-resolved numpy coordinate
            from which the shift originates.  Defaults to
            ``Location.Vertebra_Corpus``.
        direction: Anatomical direction letter (``"R"``, ``"L"``, etc.) for the
            shift, or ``None`` to return the raw local start point without
            shifting.  Defaults to ``"R"``.
        log: Logger used for warning and error messages.

    Returns:
        Shifted voxel coordinate as a 3-element numpy array, or ``None`` when
        required POIs are missing or the vertebra is excluded.
    """
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
) -> np.ndarray | None:
    """Find the boundary point of a convex region along a direction derived from POI landmarks.

    Resolves the ray start coordinate and direction vector from the ``POI``
    object, then delegates to :func:`max_distance_ray_cast_convex` to locate
    the exit point.

    Args:
        poi: ``POI`` object with pre-computed anatomical landmarks and direction
            vectors.
        region: ``NII`` image whose nonzero voxels define the convex region to
            cast the ray into.
        vert_id: Vertebra identifier (integer label).
        bb: Bounding-box slices that map between global POI coordinates and the
            local sub-volume coordinate system.
        normal_vector_points: Either a ``DIRECTIONS`` string (e.g. ``"R"``,
            ``"P"``) that is resolved from the vertebra orientation, or a
            2-tuple of ``Location`` members whose vector difference defines the
            ray direction.  Defaults to ``"R"``.
        start_point: Starting point of the ray: a ``Location`` enum member
            (resolved via the POI) or a pre-computed numpy coordinate.
            Defaults to ``Location.Vertebra_Corpus``.
        log: Logger used for warning and error messages.
        acc_delta: Bisection convergence threshold in voxel units.
            Defaults to 0.00005.

    Returns:
        3-element numpy array with the approximate exit coordinate in the local
        sub-volume coordinate system, or ``None`` when the start point or
        direction cannot be resolved.
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


def calculate_pca_normal_np(
    segmentation: np.ndarray,
    pca_component: int,
    zoom: tuple[float, ...] | np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Computes the normal vector of a segmented region using Principal Component Analysis (PCA).

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
    p1: np.ndarray | list[float],
    p2: np.ndarray | list[float],
    p3: np.ndarray | list[float],
    value: float = 0,
    invert: Literal[-1, 1] = 1,
    mask: np.ndarray | NII | bool = True,
    inplace: bool = False,
) -> NII | np.ndarray:
    """Set all values in a 3D array above a plane defined by three non-collinear points to a specified value.

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
