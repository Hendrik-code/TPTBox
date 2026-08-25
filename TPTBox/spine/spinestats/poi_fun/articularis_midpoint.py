from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from TPTBox import NII, POI, Image_Reference, Location, Logger_Interface, Print_Logger, Vertebra_Instance, to_nii

_log = Print_Logger()


def calc_all_facet_joint_pois(
    vert: Image_Reference,
    subreg: Image_Reference,
    ids: dict[str, int] | None = None,
    max_gap_mm: float = 8.0,
    surface_tolerance_mm: float = 1.5,
    log: Logger_Interface = _log,
) -> POI:
    """Compute facet-joint midpoints for every vertebra present in ``vert``.

    Iterates ids in ascending order (skipping C1) up to id 29 and delegates
    each pair to :func:`_calc_facet_joint_pois`, which places a midpoint POI
    between the Inferior_Articular process of the upper vertebra and the
    Superior_Articular process of its lower neighbour.

    Args:
        vert: Vertebra instance segmentation (image reference).
        subreg: Subregion / semantic segmentation on the same grid as ``vert``.
        ids: Optional override for the two POI subregion ids the midpoints
            are stored under. Defaults to
            ``{"left": Articular_Process_Midpoint_Left, "right": Articular_Process_Midpoint_Right}``.
        max_gap_mm: Skip a joint if the two articular surfaces are further
            apart than this in millimetres. Defaults to 8.0.
        surface_tolerance_mm: Point pairs within ``(min_distance + tolerance)``
            are averaged as the contact surface. Defaults to 1.5.
        log: Logger for status messages.

    Returns:
        POI: A fresh POI built from ``vert``'s grid containing the facet-joint
        midpoints for all successfully processed vertebra pairs.
    """
    vert_: NII = to_nii(vert, True)
    subreg_ = to_nii(subreg, True)
    poi = vert_.make_empty_POI()
    for vert_id in sorted(i for i in vert_.unique() if i < 29):
        if vert_id == 1 and vert_id != 26:
            continue

        _calc_facet_joint_pois(
            poi,
            vert_,
            subreg_,
            vert_id,
            ids=ids,
            max_gap_mm=max_gap_mm,
            surface_tolerance_mm=surface_tolerance_mm,
            log=log,
        )
    return poi


def _calc_facet_joint_pois(
    poi: POI,
    vert: NII,
    subreg: NII,
    vert_id: int,
    ids: dict[str, int] | None = None,
    max_gap_mm: float = 8.0,
    surface_tolerance_mm: float = 1.5,
    log: Logger_Interface = _log,
) -> POI:
    """Compute the facet joint point between two adjacent vertebrae, for both sides.

    The point is placed between the Inferior_Articular process of ``vert_id`` (the
    upper vertebra) and the Superior_Articular process of the next lower vertebra,
    as determined by :meth:`Vertebra_Instance.get_next_poi` (vertebra labels are not
    guaranteed to be strictly ascending, so this must be used instead of ``vert_id + 1``).

    The result lies in the middle of the (possibly extended) contact area: if the two
    surfaces touch, this is simply the middle of the contact region; if there is a
    gap, the closest facing points across the gap are averaged instead. Points that
    cannot physically belong to the contact interface (i.e. they face away from the
    other surface) are excluded before the nearest-neighbor search.

    If one of the two surfaces is missing (label not present, neighboring vertebra
    absent, etc.), the existing centroid of the other location is used as a fallback
    so the pipeline degrades gracefully. If both are missing, the point is skipped
    and a warning is logged.

    Args:
        poi: POI to extend in place.
        vert: Vertebra instance segmentation.
        subreg: Subregion/semantic segmentation, same grid as ``vert``.
        vert_id: Label of the upper vertebra (the one owning "Inferior_Articular_*").
        ids: Subregion ids under which the result is stored, e.g.
            ``{"left": Articular_Process_Midpoint_Left, "right": Articular_Process_Midpoint_Right}``. Choose ids that are free in your schema.
        max_gap_mm: If the minimal distance between the two surfaces exceeds this,
            no point is computed (joint likely absent or too far apart).
        surface_tolerance_mm: Point pairs within (min_distance + tolerance) of each
            other are averaged as "the contact surface" instead of using only the
            single closest pair.
        log: Logger for status messages.

    Returns:
        POI: the same (extended) POI object.
    """
    if ids is None:
        ids = {"left": Location.Articular_Process_Midpoint_Left.value, "right": Location.Articular_Process_Midpoint_Right.value}

    all_ids = vert.unique()
    if vert_id not in all_ids:
        log.print(f"[Facet] vert_id {vert_id} not present, skipping", verbose=True)
        return poi

    vert_id_below = _get_vert_id_below(vert_id, all_ids)

    zoom = np.array(poi.zoom if poi.zoom is not None else vert.zoom)

    for side, inf_loc, sup_loc in (
        ("left", Location.Inferior_Articular_Left, Location.Superior_Articular_Left),
        ("right", Location.Inferior_Articular_Right, Location.Superior_Articular_Right),
    ):
        target_id = ids[side]
        if (vert_id, target_id) in poi:
            continue

        pts_a, pts_b = _facet_surface_points(vert, subreg, vert_id, inf_loc, vert_id_below, sup_loc)

        point = _facet_midpoint(
            pts_a,
            pts_b,
            zoom,
            max_gap_mm=max_gap_mm,
            surface_tolerance_mm=surface_tolerance_mm,
        )

        if point is None:
            # Sensible fallback: reuse whichever centroid already exists.
            if (vert_id, inf_loc.value) in poi:
                point = poi[vert_id, inf_loc.value]
            elif vert_id_below is not None and (vert_id_below, sup_loc.value) in poi:
                point = poi[vert_id_below, sup_loc.value]

        if point is not None:
            poi[vert_id, target_id] = tuple(float(v) for v in point)
        else:
            log.print(
                f"[Facet] Could not compute facet point for vertebra {vert_id} ({side}): "
                "neither a contact surface nor a fallback centroid is available",
                verbose=True,
            )

    return poi


def _get_vert_id_below(vert_id: int, all_ids: Sequence[int]) -> int | None:
    """Return the label of the vertebra directly below ``vert_id``.

    Uses :meth:`Vertebra_Instance.get_next_poi`, since vertebra labels are not
    guaranteed to be strictly ascending (e.g. transitional or fused segments), so a
    naive ``vert_id + 1`` or "next larger label" lookup would be incorrect.

    Args:
        vert_id: Label of the current (upper) vertebra.
        all_ids: All vertebra labels present in the image.

    Returns:
        The label of the next lower vertebra, or None if there is none.
    """
    v1 = Vertebra_Instance(vert_id)
    v2 = Vertebra_Instance.get_next_poi(v1, all_ids)
    if v2 is None:
        return None
    return v2.value if isinstance(v2, Vertebra_Instance) else int(v2)


def _facet_surface_points(
    vert: NII,
    subreg: NII,
    vert_id: int,
    inf_loc: Location,
    vert_id_below: int | None,
    sup_loc: Location,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the voxel coordinates of the two candidate articular surfaces.

    Args:
        vert: Vertebra instance segmentation.
        subreg: Subregion/semantic segmentation, same grid as ``vert``.
        vert_id: Label of the upper vertebra.
        inf_loc: Inferior articular location (left or right) of the upper vertebra.
        vert_id_below: Label of the lower vertebra, or None if absent.
        sup_loc: Superior articular location (left or right) of the lower vertebra.

    Returns:
        A tuple ``(pts_a, pts_b)`` of ``(N, 3)`` arrays in full-image voxel space.
        Either array is empty if the corresponding vertebra/label is not present.
    """
    wanted = [i for i in (vert_id, vert_id_below) if i is not None]
    vert_arr_full = vert.get_seg_array()
    combined_mask = np.isin(vert_arr_full, wanted)
    if not combined_mask.any():
        return np.zeros((0, 3)), np.zeros((0, 3))

    coords = np.argwhere(combined_mask)
    mins = coords.min(0)
    maxs = coords.max(0) + 1
    sl = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))

    vert_crop = vert_arr_full[sl]
    subreg_crop = subreg.get_seg_array()[sl]

    mask_a = (vert_crop == vert_id) & (subreg_crop == inf_loc.value)
    pts_a = np.argwhere(mask_a).astype(float) + mins if mask_a.any() else np.zeros((0, 3))

    if vert_id_below is None:
        pts_b = np.zeros((0, 3))
    else:
        mask_b = (vert_crop == vert_id_below) & (subreg_crop == sup_loc.value)
        pts_b = np.argwhere(mask_b).astype(float) + mins if mask_b.any() else np.zeros((0, 3))

    return pts_a, pts_b


def _filter_facing_points(pts_a: np.ndarray, pts_b: np.ndarray, zoom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discard points that lie on the side of each surface facing away from the other.

    The articular process is a 3D blob, not a flat plate: only the hemisphere facing
    the neighboring vertebra can physically be part of the contact interface. Points
    on the outer/back side are geometrically irrelevant and, if included, could be
    picked up by the nearest-neighbor search purely by chance (e.g. on oddly shaped
    or partially segmented processes). This filter removes them before matching.

    Args:
        pts_a: Voxel coordinates of surface A.
        pts_b: Voxel coordinates of surface B.
        zoom: Voxel spacing, used to compute the facing direction in mm.

    Returns:
        The filtered ``(pts_a, pts_b)``. Falls back to the unfiltered input for a
        surface if filtering would remove all of its points (degenerate/very small
        surfaces).
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return pts_a, pts_b

    a_mm = pts_a * zoom
    b_mm = pts_b * zoom
    centroid_a = a_mm.mean(axis=0)
    centroid_b = b_mm.mean(axis=0)
    direction = centroid_b - centroid_a
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return pts_a, pts_b
    direction = direction / norm

    a_keep = (a_mm - centroid_a) @ direction >= 0
    b_keep = (b_mm - centroid_b) @ (-direction) >= 0

    pts_a_f = pts_a[a_keep] if a_keep.any() else pts_a
    pts_b_f = pts_b[b_keep] if b_keep.any() else pts_b
    return pts_a_f, pts_b_f


def _facet_midpoint(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    zoom: np.ndarray,
    max_gap_mm: float,
    surface_tolerance_mm: float,
) -> tuple[float, float, float] | None:
    """Compute the midpoint of the (possibly extended) contact area between two surfaces.

    Args:
        pts_a: Voxel coordinates of surface A.
        pts_b: Voxel coordinates of surface B.
        zoom: Voxel spacing, used to compute distances in mm.
        max_gap_mm: Maximum allowed minimal distance between the surfaces.
        surface_tolerance_mm: Band above the minimal distance that is still
            considered part of the contact surface and averaged.

    Returns:
        The midpoint in voxel coordinates, or None if either surface is empty
        (after facing-filter) or the minimal distance exceeds ``max_gap_mm``.
    """
    pts_a, pts_b = _filter_facing_points(pts_a, pts_b, zoom)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return None

    tree_b = cKDTree(pts_b * zoom)
    dist, idx_b = tree_b.query(pts_a * zoom, k=1)
    min_dist = float(dist.min())
    if min_dist > max_gap_mm:
        return None

    close = dist <= (min_dist + surface_tolerance_mm)
    a_close = pts_a[close]
    b_close = pts_b[idx_b[close]]
    midpoints = (a_close + b_close) / 2.0
    return tuple(midpoints.mean(axis=0).tolist())
