from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np

from TPTBox import POI, Image_Reference
from TPTBox.core.compat import zip_strict
from TPTBox.core.nii_wrapper import to_nii
from TPTBox.core.np_utils import np_angle_between, np_unit_vector
from TPTBox.core.vert_constants import DIRECTIONS, Location, Vertebra_Instance
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

IVD_MORE_ACCURATE = 15
VERT_START_COBB = Vertebra_Instance.C3


class MoveTo(Enum):
    """Enum selecting which endplate or centre point of a vertebra is used as the measurement anchor."""

    TOP = auto()
    BOTTOM = auto()
    CENTER = auto()

    def has_point(self, v: Vertebra_Instance | int, poi: POI) -> bool:
        """Check if the given vertebra has a specific POI at the current MoveTo position.

        Args:
            v (Vertebra_Instance | int): The vertebra instance or its ID.
            poi (POI): The point of interest data structure.

        Returns:
            bool: True if the POI exists for the specified MoveTo position, otherwise False.
        """
        if isinstance(v, int):
            v = Vertebra_Instance(v)
        try:
            self.get_point(v, poi)
        except KeyError:
            return False
        return True

    def get_location(self, v: Vertebra_Instance | int, poi: POI) -> tuple:
        """Determine the anatomical POI coordinates for the current MoveTo position in a vertebra.

        Args:
            v (Vertebra_Instance | int): The vertebra instance or its ID.
            poi (POI): The point of interest data structure.

        Returns:
            tuple: The location tuple that defines the position within the vertebra.
        """
        if isinstance(v, int):
            v = Vertebra_Instance(v)
        if self == self.CENTER:
            return (v, 50)
        elif self == self.BOTTOM:
            # Test IVD
            subreg = Location.Vertebra_Disc
            if (v, subreg) in poi:
                return (v, subreg)
            # Test if it has next
            subreg = Location.Additional_Vertebral_Body_Middle_Inferior_Median
            if (v, subreg) in poi:
                return (v, subreg)
            # Fall back to averaging v's centroid with the next vertebra's centroid.
            next_vert = v.get_next_poi(poi)
            if next_vert is not None and (v, 50) in poi and (next_vert, 50) in poi:
                return (v, 50, next_vert, 50)
        elif self == self.TOP:
            prev_vert = v.get_previous_poi(poi)
            # Test IVD
            subreg = Location.Vertebra_Disc
            if prev_vert is not None and (prev_vert, subreg) in poi:
                return (prev_vert, subreg)
            # Test if it has next
            subreg = Location.Dens_axis
            if (v, subreg) in poi:
                return (v, subreg)
            # Test if it has next
            subreg = Location.Additional_Vertebral_Body_Middle_Superior_Median
            if (v, subreg) in poi:
                return (v, subreg)
            # Fall back to averaging v's centroid with the previous vertebra's centroid.
            if prev_vert is not None and (v, 50) in poi and (prev_vert, 50) in poi:
                return (v, 50, prev_vert, 50)
        return (v, 50)

    def get_point(self, v: Vertebra_Instance | int, poi: POI) -> np.ndarray:
        """Retrieve the 3D coordinates of a POI in a vertebra at the current MoveTo position.

        Args:
            v (Vertebra_Instance | int): The vertebra instance or its ID.
            poi (POI): The point of interest data structure.

        Returns:
            np.ndarray: The 3D coordinates of the specified POI.

        Raises:
            NotImplementedError: If the POI cannot be determined.
        """
        a = self.get_location(v, poi)
        if len(a) == 2:
            return np.array(poi[a])
        elif len(a) == 4:
            return (np.array(poi[a[0], a[1]]) + np.array(poi[a[2], a[3]])) / 2
        raise NotImplementedError(v, poi)


def _get_last_lumbar(poi: POI) -> Vertebra_Instance | None:
    """Return the most inferior lumbar vertebra that has a centroid in ``poi``."""
    for i in list(reversed(Vertebra_Instance.lumbar()))[:5]:
        if (i.value, 50) in poi:
            return i
    return None


def _get_last_thoracic(poi: POI) -> Vertebra_Instance | None:
    """Return the most inferior thoracic vertebra that has a centroid in ``poi``."""
    for i in list(reversed(Vertebra_Instance.thoracic()))[:3]:
        if (i.value, 50) in poi:
            return i
    return None


@dataclass
class Def_Curvature:
    """Define the lordosis and kyposis angle."""

    start_vert: Vertebra_Instance | Literal["last_thoracic", "last_lumbar"]
    start_move: MoveTo
    stop_vert: Vertebra_Instance | Literal["last_thoracic", "last_lumbar"]
    stop_move: MoveTo

    def get_start_vert(self, poi) -> Vertebra_Instance:
        """get_start_vert."""
        if self.start_vert == "last_thoracic":
            return _get_last_thoracic(poi)  # type: ignore
        if self.start_vert == "last_lumbar":
            return _get_last_lumbar(poi)  # type: ignore
        return self.start_vert

    def get_stop_vert(self, poi) -> Vertebra_Instance:
        """get_stop_vert."""
        if self.stop_vert == "last_thoracic":
            return _get_last_thoracic(poi)  # type: ignore
        if self.stop_vert == "last_lumbar":
            return _get_last_lumbar(poi)  # type: ignore
        return self.stop_vert


curvature_definition = {
    "cervical_lordosis": Def_Curvature(Vertebra_Instance.C2, MoveTo.BOTTOM, Vertebra_Instance.C7, MoveTo.BOTTOM),
    "thoracic_kyphosis": Def_Curvature(Vertebra_Instance.T4, MoveTo.TOP, "last_thoracic", MoveTo.BOTTOM),
    "lumbar_lordosis": Def_Curvature(Vertebra_Instance.L1, MoveTo.TOP, "last_lumbar", MoveTo.BOTTOM),
}


# Canonical implementations live in np_utils; re-exported here under their historic names.
unit_vector = np_unit_vector
angle_between = np_angle_between


def get_to_space(a, b, c) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward and inverse transformation matrices for the space defined by three orthogonal vectors.

    Args:
        a (np.ndarray): First orthogonal vector.
        b (np.ndarray): Second orthogonal vector.
        c (np.ndarray): Third orthogonal vector.

    Returns:
        tuple: A tuple containing two matrices (to_space, from_space):
            - to_space (np.ndarray): Transformation matrix to the canonical space.
            - from_space (np.ndarray): Transformation matrix from the canonical space.
    """
    from_space = np.stack([a, b, c], axis=1)
    to_space = np.linalg.inv(from_space)
    return to_space, from_space


def cosine_distance(a, b) -> float:
    """Computes the cosine distance between two vectors.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.

    Returns:
        float: The cosine distance between vectors 'a' and 'b'.
    """
    return np.dot(a, b) / (np.linalg.norm(b) * np.linalg.norm(a))


def compute_angel_between_two_points_(
    poi: POI,
    vert_id1: Vertebra_Instance | int | None,
    vert_id2: Vertebra_Instance | int | None,
    direction: DIRECTIONS,
    vert_id1_mv: MoveTo = MoveTo.CENTER,
    vert_id2_mv: MoveTo = MoveTo.CENTER,
    project_2D=False,
    use_ivd_direction=False,
) -> float | None:
    """Compute the 2D or 3D angle between two anatomical landmarks.

    Useful for calculating coplanar angles, lordosis, and kyphosis depending on
    the direction specified.

    Args:
        poi (POI): An object representing a point of interest that supports indexing
            with point IDs and returns 3D coordinates. Must have methods `reorient_()`
            and `rescale_()` to prepare the data.
        vert_id1 (Vertebra_Instance | int | None): The identifier for the first point of interest.
        vert_id2 (Vertebra_Instance | int | None): The identifier for the second point of interest.
        direction (DIRECTIONS): The direction in which to compute the angle. Possible values are:
            - "P" for Posterior, used for calculating lordosis and kyphosis.
            - "A" for Anterior.
            - "R" for Right, used for calculating coplanar angles.
            - "L" for Left.
            - "S" for Superior.
            - "I" for Inferior.
        vert_id1_mv (MoveTo, optional): MoveTo instance indicating the position to consider for the first vertebra. Defaults to MoveTo.CENTER.
        vert_id2_mv (MoveTo, optional): MoveTo instance indicating the position to consider for the second vertebra. Defaults to MoveTo.CENTER.
        project_2D (bool, optional): If True, computes the 2D projection of the angle. Defaults to False.
        use_ivd_direction (bool, optional): For coronal/right-directed angles, use the IVD direction (via
            ``Location.Vertebra_Disc_Inferior``) instead of the vertebra direction for lumbar/thoracic ids
            beyond ``IVD_MORE_ACCURATE``. Defaults to False.

    Returns:
        float | None: The computed angle in degrees. Returns None if either vertebra ID is invalid.

    Raises:
        NotImplementedError: If the direction `direction` is not one of the recognized values.

    Example:
        To compute the coplanar angle between two vertebrae with IDs 20 and 21:

        >>> compute_angel_between_two_points_(poi, 20, 21, "R")

        To compute the lordosis angle between two vertebrae with IDs 20 and 21:

        >>> compute_angel_between_two_points_(poi, 20, 21, "P")
    """
    if vert_id1 is None or vert_id2 is None:
        return None
    # assert project_2D, "project_2D == True"
    id1: int = vert_id1.value if isinstance(vert_id1, Enum) else vert_id1
    id2: int = vert_id2.value if isinstance(vert_id2, Enum) else vert_id2
    if (id1, 50) not in poi or (id2, 50) not in poi:
        return None
    assert id1 != id2, id1

    # Ensure id1 is anatomically above id2 (not just numerically smaller).
    # T13 has label value 28 but sits between T12 (19) and L1 (20) in anatomical
    # order; using the label value directly would misplace it and swap the wrong
    # MoveTo semantics onto TOP/BOTTOM.
    _order = Vertebra_Instance.order_dict()
    if _order.get(id1, id1) > _order.get(id2, id2):
        id1, id2 = id2, id1
    # Reorient and rescale the POI data
    poi.reorient_().rescale_(verbose=False)
    recompute_use_ivd_direction = False
    # Determine direction-specific settings
    location2 = None
    if direction in ["P", "A"]:
        # Note: inf - The value of the direction does noting
        location = Location.Vertebra_Direction_Posterior
        inv = 1 if direction == "P" else -1
    elif direction in ["R", "L"]:
        location = Location.Vertebra_Direction_Right
        inv = 1 if direction == "R" else -1
        if use_ivd_direction:
            location = Location.Vertebra_Disc_Inferior if id1 > IVD_MORE_ACCURATE else Location.Vertebra_Direction_Right
            location2 = Location.Vertebra_Disc_Inferior if id2 > IVD_MORE_ACCURATE else Location.Vertebra_Direction_Right
            recompute_use_ivd_direction = True
    elif direction in ["S", "I"]:
        if use_ivd_direction:
            location = Location.Vertebra_Disc_Inferior if id1 > IVD_MORE_ACCURATE else Location.Vertebra_Direction_Inferior
            location2 = Location.Vertebra_Disc_Inferior if id2 > IVD_MORE_ACCURATE else Location.Vertebra_Direction_Inferior
        else:
            location = Location.Vertebra_Direction_Inferior
        inv = 1 if direction == "I" else -1
    else:
        raise NotImplementedError(f"Direction '{direction}' is not recognized.")
        # Calculate normals for the vertebrae
    if location2 is None:
        location2 = location
    norm1_vert = _get_norm(poi, id1, vert_id1_mv, location, inv=inv)
    norm2_vert = _get_norm(poi, id2, vert_id2_mv, location2, inv=inv)
    if norm1_vert is None or norm2_vert is None:
        return None
    if recompute_use_ivd_direction:
        # Compute right from Post (Vert) + Inferior (Disc)
        if Location.Vertebra_Disc_Inferior == location:
            norm1_post = _get_norm(poi, id1, vert_id1_mv, Location.Vertebra_Direction_Posterior)
            if norm1_post is None:
                return None

            norm1_vert = np.cross(norm1_vert, norm1_post)
        if Location.Vertebra_Disc_Inferior == location2:
            norm2_post = _get_norm(poi, id2, vert_id2_mv, Location.Vertebra_Direction_Posterior)
            if norm2_post is None:
                return None
            norm2_vert = np.cross(norm2_vert, norm2_post)

    assert norm1_vert is not None
    assert norm2_vert is not None
    # if _debug_plot is not None:
    #    nii_old = _debug_plot.copy()
    #    _debug_plot.reorient_().rescale_()
    #    _debug_plot = _debug_plot * 0
    #    plot_ray(poi[id1, subreg], norm1_vert, _debug_plot, inplace=True, value=1)
    #    plot_ray(poi[id2, subreg], norm2_vert, _debug_plot, inplace=True, value=2)
    # Calculate the 3D angle between the normal
    if not project_2D:
        angle_3D = angle_between(norm1_vert, norm2_vert)  # noqa: N806
        return angle_3D / np.pi * 180
    p1 = vert_id1_mv.get_point(id1, poi)
    p2 = vert_id2_mv.get_point(id2, poi)

    if direction in ["S", "I"]:
        a = unit_vector(np.array(p1) - np.array(poi[id1, Location.Vertebra_Direction_Right]))
        b = unit_vector(np.array(p2) - np.array(poi[id2, Location.Vertebra_Direction_Right]))
        norm_down = (a + b) / 2
    else:
        norm_down = unit_vector(np.array(p2) - np.array(p1))
    # Compute cross products to isolate relevant components
    norm_to_remove = np.cross(norm_down, (norm1_vert + norm2_vert))
    norm_keep_other = np.cross(norm_to_remove, norm_down)
    # Bring into a space where we can remove not considered component (like anterior/posterior is ignored for copangles)
    # Transform into a 2D space for angle calculation
    to_space, from_space = get_to_space(norm_keep_other, norm_down, norm_to_remove)
    norm1_vert = to_space @ norm1_vert
    norm2_vert = to_space @ norm2_vert
    norm1_vert[2] = 0
    norm2_vert[2] = 0
    norm1_vert = from_space @ norm1_vert
    norm2_vert = from_space @ norm2_vert
    # Calculate the 2D angle between the transformed normals
    angle_2D = angle_between(norm1_vert, norm2_vert)  # noqa: N806
    # if _debug_plot is not None:
    #    plot_ray(poi[id1, subreg], norm1_vert, _debug_plot, inplace=True, value=3)
    #    plot_ray(poi[id2, subreg], norm2_vert, _debug_plot, inplace=True, value=4)

    #    plot_ray(poi[24, 50], norm_down, _debug_plot, inplace=True, value=5)
    #    plot_ray(poi[24, 50], norm_to_removed, _debug_plot, inplace=True, value=6)
    #    plot_ray(poi[24, 50], norm_keep_other, _debug_plot, inplace=True, value=7)
    #    _debug_plot.dilate_msk_().resample_from_to_(nii_old).save(OURPATH)
    return angle_2D / np.pi * 180


def compute_lordosis_and_kyphosis(poi: POI, project_2D=True) -> dict[str, float | None]:
    """Calculates the angles of cervical lordosis, thoracic kyphosis, and lumbar lordosis based on the given points of interest (POI).

    This function determines the angles formed by specific vertebrae along the spine, which are indicative of spinal curvatures.
    The angles are calculated for three key regions: cervical, thoracic, and lumbar, representing lordosis and kyphosis.

    Args:
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae. It must include
            the vertebra direction information for proper calculation. (Location.Vertebra_Direction_Posterior)
        project_2D (bool): If True, the calculation is done in 2D projection; otherwise, in 3D. Defaults to True.

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - "cervical_lordosis": The angle of cervical lordosis, calculated between C2 and C7.
            - "thoracic_kyphosis": The angle of thoracic kyphosis, calculated between T4 and the last thoracic vertebra.
            - "lumbar_lordosis": The angle of lumbar lordosis, calculated between L1 and the last lumbar vertebra.

    Raises:
        AssertionError: If the required vertebra direction information is not present in the POI.

    Notes:
        - It is essential that the `poi` contains the posterior vertebra direction for accurate angle calculations.
        - Thoracic kyphosis is calculated from T4 to the last thoracic vertebra identified in the POI.
        - Lumbar lordosis is calculated from L1 to the last lumbar vertebra identified in the POI.

    Example:
        To compute the spinal angles for a given POI object:

        >>> angles = compute_lordosis_and_kyphosis(poi, project_2D=True)
        >>> print(angles)
        {'cervical_lordosis': 30.5, 'thoracic_kyphosis': 35.0, 'lumbar_lordosis': 45.2}
    """
    assert Location.Vertebra_Direction_Posterior.value in poi.keys_subregion(), (
        "You need to compute the Direction in the Poi (Location.Vertebra_Direction_Posterior)"
    )
    out = {}
    poi = poi.copy()

    for k, i in curvature_definition.items():
        start = i.get_start_vert(poi)
        stop = i.get_stop_vert(poi)
        angle = compute_angel_between_two_points_(poi, start, stop, "P", i.start_move, i.stop_move, project_2D)
        out[k] = round(angle, 4) if angle is not None else None
        out[f"{k}_apex"] = _find_curve_apex(poi, start, stop, i.start_move, i.stop_move, Location.Vertebra_Direction_Posterior)
    return out


def _find_curve_apex(
    poi: POI,
    from_vert: Vertebra_Instance | int | None,
    to_vert: Vertebra_Instance | int | None,
    from_mv: MoveTo,
    to_mv: MoveTo,
    location: Location,
) -> int | None:
    """Return the vertebra between ``from_vert`` and ``to_vert`` whose direction is closest to the endpoint bisector.

    Same apex heuristic as :func:`compute_max_cobb_angle`, just parameterised by
    the direction location so it also works for lordosis / kyphosis
    (``Vertebra_Direction_Posterior``). Returns ``None`` when either endpoint
    direction is unavailable or no intermediate vertebra is present.
    """
    if from_vert is None or to_vert is None:
        return None
    from_v = from_vert if isinstance(from_vert, Vertebra_Instance) else Vertebra_Instance(from_vert)
    to_v = to_vert if isinstance(to_vert, Vertebra_Instance) else Vertebra_Instance(to_vert)
    a = _get_norm(poi, from_v, from_mv, location, 1)
    b = _get_norm(poi, to_v, to_mv, location, 1)
    if a is None or b is None:
        return None
    apex_v = (a + b) / 2
    order = Vertebra_Instance.order()
    try:
        i_from = order.index(from_v)
        i_to = order.index(to_v)
    except ValueError:
        return None
    if i_from > i_to:
        i_from, i_to = i_to, i_from
    apex: int | None = None
    cos_dis = -np.inf
    for v in order[i_from : i_to + 1]:
        n = _get_norm(poi, v, to_mv, location, 1)
        if n is None:
            continue
        cos_new = cosine_distance(n, apex_v)
        if cos_new > cos_dis:
            cos_dis = cos_new
            apex = v.value
    return apex


def _single_endplate_ap_direction(poi: POI, vert: Vertebra_Instance, side: MoveTo) -> np.ndarray | None:
    """A/P direction of *one* endplate (superior for TOP, inferior for BOTTOM).

    Uses ``Vertebra_Corpus``, ``Vertebral_Body_Endplate_Superior/Inferior`` and
    ``Vertebra_Direction_Right`` to build a unit vector that lies in the endplate
    plane and points *anteriorly* (matches :func:`_get_norm`'s ``inv=1`` sign
    convention). Returns ``None`` if any required POI landmark is missing.
    """
    if side == MoveTo.TOP:
        endplate_loc = Location.Vertebral_Body_Endplate_Superior
    elif side == MoveTo.BOTTOM:
        endplate_loc = Location.Vertebral_Body_Endplate_Inferior
    else:
        return None
    if (vert, 50) not in poi or (vert, endplate_loc) not in poi or (vert, Location.Vertebra_Direction_Right) not in poi:
        return None
    corpus = np.array(poi[vert, 50], dtype=float)
    ep = np.array(poi[vert, endplate_loc], dtype=float)
    r_pt = np.array(poi[vert, Location.Vertebra_Direction_Right], dtype=float)
    n = ep - corpus
    if side == MoveTo.BOTTOM:
        n = -n  # flip inferior endplate so both cases point superior
    n_norm = np.linalg.norm(n)
    r_vec = r_pt - corpus
    r_norm = np.linalg.norm(r_vec)
    if n_norm < 1e-8 or r_norm < 1e-8:
        return None
    n /= n_norm
    r_vec /= r_norm
    # cross(right, superior-pointing) lies in the endplate plane and points posterior;
    # negate to match _get_norm's default sign (anterior for inv=1).
    p = -np.cross(r_vec, n)
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-8:
        return None
    return p / p_norm


def _endplate_ap_direction(poi: POI, vert: Vertebra_Instance, mv: MoveTo) -> np.ndarray | None:
    """Return the endplate-plane A/P direction at a vertebra-disc *boundary*.

    For a lordosis / kyphosis chain to close at every transition (e.g. so
    ``thoracic_kyphosis + lumbar_lordosis`` measures the same T4-top→L5-inferior
    end-to-end angle as the total), the "bottom of upper" and "top of lower" at
    each disc must use the *same* reference direction. This function averages the
    two flanking endplate P/A directions at the disc:

    - :attr:`MoveTo.BOTTOM` at vertebra ``V`` averages ``V``'s inferior endplate
      with the superior endplate of the next vertebra in the POI (``V.get_next_poi``).
    - :attr:`MoveTo.TOP` at vertebra ``V`` averages ``V``'s superior endplate
      with the inferior endplate of the previous vertebra (``V.get_previous_poi``).

    Falls back to the single-endplate direction when the neighbour is missing.
    Returns ``None`` when even the local endplate is unavailable.
    """
    if isinstance(vert, int):
        vert = Vertebra_Instance(vert)
    if mv == MoveTo.BOTTOM:
        neighbour = vert.get_next_poi(poi)
        neighbour_side = MoveTo.TOP
    elif mv == MoveTo.TOP:
        neighbour = vert.get_previous_poi(poi)
        neighbour_side = MoveTo.BOTTOM
    else:
        return _single_endplate_ap_direction(poi, vert, mv)
    own = _single_endplate_ap_direction(poi, vert, mv)
    other = _single_endplate_ap_direction(poi, neighbour, neighbour_side) if neighbour is not None else None
    # If the local endplate landmark is missing (calc_endplate_points_ ray-cast
    # sometimes fails to hit the mask), mirror across the disc: use the neighbour's
    # endplate as the direction proxy so both sides of the boundary agree.
    if own is None:
        return other
    if other is None:
        return own
    avg = own + other
    n = np.linalg.norm(avg)
    if n < 1e-8:
        return own
    return avg / n


def _single_endplate_r_direction(poi: POI, vert: Vertebra_Instance, side: MoveTo) -> np.ndarray | None:
    """R/L direction of *one* endplate (superior for TOP, inferior for BOTTOM).

    Builds a unit vector that lies in the endplate plane along the vertebra's
    right/left axis: takes ``corpus - Vertebra_Direction_Right`` (matching the
    WK path's ``inv=1`` sign convention, which points *left*) and projects it
    onto the plane perpendicular to the endplate normal (``endplate_point -
    corpus``). Returns ``None`` if any required POI landmark is missing.
    """
    if side == MoveTo.TOP:
        endplate_loc = Location.Vertebral_Body_Endplate_Superior
    elif side == MoveTo.BOTTOM:
        endplate_loc = Location.Vertebral_Body_Endplate_Inferior
    else:
        return None
    if (vert, 50) not in poi or (vert, endplate_loc) not in poi or (vert, Location.Vertebra_Direction_Right) not in poi:
        return None
    corpus = np.array(poi[vert, 50], dtype=float)
    ep = np.array(poi[vert, endplate_loc], dtype=float)
    r_pt = np.array(poi[vert, Location.Vertebra_Direction_Right], dtype=float)
    n = ep - corpus
    if side == MoveTo.BOTTOM:
        n = -n  # flip inferior endplate so both cases point superior
    n_norm = np.linalg.norm(n)
    # WK path uses ``corpus - right_pt`` (points anatomical LEFT); mirror that
    # convention so both paths agree when they meet in
    # ``compute_angel_between_two_points_``.
    r_vec = corpus - r_pt
    r_norm = np.linalg.norm(r_vec)
    if n_norm < 1e-8 or r_norm < 1e-8:
        return None
    n /= n_norm
    r_vec /= r_norm
    # Project the vertebra right/left vector onto the endplate plane
    r_ep = r_vec - np.dot(r_vec, n) * n
    r_ep_norm = np.linalg.norm(r_ep)
    if r_ep_norm < 1e-8:
        return None
    return r_ep / r_ep_norm


def _endplate_r_direction(poi: POI, vert: Vertebra_Instance, mv: MoveTo) -> np.ndarray | None:
    """R/L direction in the endplate plane at a vertebra-disc *boundary*.

    Analogue of :func:`_endplate_ap_direction` for the coronal (right) direction,
    used to obtain a classical endplate-line orientation for Cobb angles. Averages
    the two flanking endplate R directions at a disc:

    - :attr:`MoveTo.BOTTOM` at vertebra ``V`` averages ``V``'s inferior endplate
      with the superior endplate of the next vertebra.
    - :attr:`MoveTo.TOP` at vertebra ``V`` averages ``V``'s superior endplate
      with the inferior endplate of the previous vertebra.

    Falls back to the single-endplate direction when the neighbour is missing.
    Returns ``None`` when even the local endplate is unavailable.
    """
    if isinstance(vert, int):
        vert = Vertebra_Instance(vert)
    if mv == MoveTo.BOTTOM:
        neighbour = vert.get_next_poi(poi)
        neighbour_side = MoveTo.TOP
    elif mv == MoveTo.TOP:
        neighbour = vert.get_previous_poi(poi)
        neighbour_side = MoveTo.BOTTOM
    else:
        return _single_endplate_r_direction(poi, vert, mv)
    own = _single_endplate_r_direction(poi, vert, mv)
    other = _single_endplate_r_direction(poi, neighbour, neighbour_side) if neighbour is not None else None
    if own is None:
        return other
    if other is None:
        return own
    avg = own + other
    n = np.linalg.norm(avg)
    if n < 1e-8:
        return own
    return avg / n


def _get_norm(poi: POI, id1: int | Vertebra_Instance, mv: MoveTo, location: Location, inv: int = 1) -> np.ndarray | None:  # noqa: ARG001
    """Return the normalised direction vector from a location POI to the vertebra centroid.

    When ``location`` is :attr:`Location.Vertebra_Direction_Posterior` and ``mv`` targets
    an endplate (:attr:`MoveTo.TOP` / :attr:`MoveTo.BOTTOM`), the buffered per-endplate
    landmark (``Vertebral_Body_Endplate_Superior`` / ``_Inferior``) is preferred over
    the averaged vertebral-body posterior direction — this yields the classical
    endplate-line orientation used in Cobb-style lordosis/kyphosis measurements.
    The same automatic switch applies to ``Location.Vertebra_Direction_Right``: it
    is projected into the endplate plane so Cobb (scoliosis) angles are measured
    between endplate lines, analogous to the sagittal case. The endplate direction
    is used only when both the relevant endplate point and
    ``Vertebra_Direction_Right`` are present in ``poi`` for that vertebra; otherwise
    the code falls back to the WK-based averaged direction below.
    """
    if isinstance(id1, int):
        id1 = Vertebra_Instance(id1)
    if location == Location.Vertebra_Direction_Posterior and mv in (MoveTo.TOP, MoveTo.BOTTOM):
        ep_norm = _endplate_ap_direction(poi, id1, mv)
        if ep_norm is not None:
            return ep_norm * inv
    if location == Location.Vertebra_Direction_Right and mv in (MoveTo.TOP, MoveTo.BOTTOM):
        ep_norm = _endplate_r_direction(poi, id1, mv)
        if ep_norm is not None:
            return ep_norm * inv
    subreg = 50
    if location in [Location.Vertebra_Disc_Inferior, Location.Vertebra_Disc_Superior]:
        subreg = 100
    if (id1, subreg) not in poi or (id1, location) not in poi:
        return None
    a = np.array(poi[id1, subreg])
    b = np.array(poi[id1, location])
    if (a == b).all():
        return None
    norm1_vert = unit_vector(a - b) * inv
    next_vert = None
    if mv == MoveTo.CENTER:
        return norm1_vert
    elif mv == MoveTo.BOTTOM:
        next_vert = id1.get_next_poi(poi)
    elif mv == MoveTo.TOP:
        next_vert = id1.get_previous_poi(poi)
    if next_vert is None:
        return norm1_vert
    if (next_vert, location) in poi:
        norm1_vert_2 = unit_vector(np.array(poi[next_vert, 50]) - np.array(poi[next_vert, location])) * inv
        norm1_vert = (norm1_vert + norm1_vert_2) / 2
    return norm1_vert


def compute_max_cobb_angle(
    poi: POI,
    vertebrae_list=None,
    vert_id1_mv: MoveTo = MoveTo.TOP,
    vert_id2_mv: MoveTo = MoveTo.BOTTOM,
    project_2D=True,
    use_ivd_direction=False,
) -> tuple[float, int | None, int | None, int | None]:
    """Calculates the maximum Cobb angle from a list of vertebrae using the points of interest (POI).

    The Cobb angle is a measure commonly used to quantify the degree of spinal curvature, particularly for scoliosis.
    This function identifies the maximum Cobb angle by comparing angles between pairs of vertebrae in the specified list.
    You must compute have computed pois in the following structures:
    Version 1:
        poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Corpus,Location.Vertebra_Direction_Right])
        ivd position will be interpolated.
    Version 2:
        poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Corpus,Location.Vertebra_Direction_Right,Location.Vertebra_Disc])
        ivd (Vertebra_Disc) will be computed by the segmentation
    Version 3 (best):
        poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Corpus,Location.Vertebra_Direction_Right,Location.Vertebra_Disc,Location.Vertebra_Disc_Superior])
        ivd (Vertebra_Disc) will be computed by the segmentation
        + use_ivd_direction = True will use the disc direction and will note if there is a large shift between vertebra without rotation.

    Args:
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae.
        vertebrae_list (list, optional): A list of vertebra instances to consider for Cobb angle calculation.
            If not provided, defaults to all cervical, thoracic, and lumbar vertebrae.
        vert_id1_mv (MoveTo): Enum indicating the move direction for the first vertebra (default is MoveTo.TOP).
        vert_id2_mv (MoveTo): Enum indicating the move direction for the second vertebra (default is MoveTo.BOTTOM).
        project_2D (bool): If True, the calculation is done in 2D projection; otherwise, in 3D. Defaults to True.
        use_ivd_direction (bool, optional): For lumbar/thoracic ids beyond ``IVD_MORE_ACCURATE``, use the IVD direction
            (via ``Location.Vertebra_Disc_Inferior``) instead of the vertebra direction. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - max_angle (float): The maximum Cobb angle identified between any pair of vertebrae.
            - from_vert (int or None): The vertebra ID at which the maximum angle originates.
            - to_vert (int or None): The vertebra ID at which the maximum angle terminates.
            - apex (int or None): The vertebra ID that is the apex of the maximum Cobb angle curvature.

    Raises:
        AssertionError: If the necessary direction data for computation is not present in the POI.

    Notes:
        - The function iterates through pairs of vertebrae to calculate the angles between them.
        - It uses the `compute_angel_between_two_points_` function to determine the angle between two vertebrae.
        - The apex is determined by finding the vertebra with the largest cosine distance to the calculated apex vector.
        - Ensure that the vertebrae are provided in a correct anatomical order to avoid inaccurate results.

    Example:
        To compute the maximum Cobb angle for a given POI object:

        >>> max_angle, from_vert, to_vert, apex = compute_max_cobb_angle(poi, project_2D=True)
        >>> print(f"Max Angle: {max_angle}, From: {from_vert}, To: {to_vert}, Apex: {apex}")
        Max Angle: 35.6, From: 3, To: 12, Apex: 7
    """
    max_angle = 0
    from_vert: int | None = None
    to_vert: int | None = None
    if vertebrae_list is None:
        # Define the range of vertebrae to consider (e.g., from T1 to L5)
        vertebrae_list = list(Vertebra_Instance.cervical()) + list(Vertebra_Instance.thoracic()) + list(Vertebra_Instance.lumbar())
        vertebrae_list = vertebrae_list[vertebrae_list.index(VERT_START_COBB) :]
    # Iterate through pairs of adjacent vertebrae
    for i in range(len(vertebrae_list)):
        vert_id1: int = vertebrae_list[i].value
        for i2 in range(i):
            vert_id2: int = vertebrae_list[i2].value
            # Ensure that both vertebrae are present in the POI
            if vert_id1 in poi.keys_region() and vert_id2 in poi.keys_region():
                # Compute cobblanar angles for both right and left directions
                angle = compute_angel_between_two_points_(
                    poi,
                    vert_id1,
                    vert_id2,
                    "R",
                    vert_id1_mv,
                    vert_id2_mv,
                    project_2D,
                    use_ivd_direction=use_ivd_direction,
                )
                if angle is None:
                    continue
                # Update max_angle if a larger angle is found
                if max_angle < angle:
                    max_angle, from_vert, to_vert = angle, vert_id2, vert_id1
    apex: int | None = None
    cos_dis = 0
    if from_vert is not None and to_vert is not None:
        a = _get_norm(poi, from_vert, vert_id1_mv, Location.Vertebra_Direction_Right, 1)
        b = _get_norm(poi, to_vert, vert_id2_mv, Location.Vertebra_Direction_Right, 1)
        assert a is not None
        assert b is not None
        apex_v = (a + b) / 2
        for i in vertebrae_list[vertebrae_list.index(Vertebra_Instance(from_vert)) : vertebrae_list.index(Vertebra_Instance(to_vert)) + 1]:
            try:
                a = _get_norm(poi, i, vert_id2_mv, Location.Vertebra_Direction_Right, 1)
                if a is None:
                    continue
            except KeyError:
                continue
            cos_new = cosine_distance(a, apex_v)
            if cos_dis < cos_new:
                cos_dis = cos_new
                apex = i.value
    return round(max_angle, 4), from_vert, to_vert, apex


def compute_max_cobb_angle_multi(
    poi: POI,
    vertebrae_list=None,
    threshold_deg=10,
    out_list=None,
    vert_id1_mv: MoveTo = MoveTo.TOP,
    vert_id2_mv: MoveTo = MoveTo.BOTTOM,
    use_ivd_direction=False,
    project_2D=True,
) -> list[tuple[float, int, int, int | None]]:
    """Identifies multiple Cobb angles along the spine that exceed a given threshold.

    This function calculates Cobb angles for a list of vertebrae and recursively finds multiple
    spinal curvatures that are large enough, as determined by a threshold angle. It is useful for
    detecting and evaluating scoliosis or other spinal deformities with multiple curves.
    You must compute have computed pois in the following structures:
    Version 1:
        poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Corpus,Location.Vertebra_Direction_Right])
        ivd position will be interpolated.
    Version 2:
        poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Corpus,Location.Vertebra_Direction_Right,Location.Vertebra_Disc])
        ivd (Vertebra_Disc) will be computed by the segmentation
    Version 3 (best):
        poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Corpus,Location.Vertebra_Direction_Right,Location.Vertebra_Disc,Location.Vertebra_Disc_Inferior])
        ivd (Vertebra_Disc) will be computed by the segmentation
        + use_ivd_direction = True will use the disc direction and will note if there is a large shift between vertebra without rotation.

    Args:
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae.
        vertebrae_list (list, optional): A list of vertebra instances to consider for Cobb angle calculation.
            If not provided, defaults to all cervical, thoracic, and lumbar vertebrae.
        threshold_deg (float): The angle threshold in degrees. Only curves with angles greater than
            this value will be recorded.
        out_list (list, optional): A list to store the results of the identified Cobb angles.
            If not provided, an empty list will be created and used.
        vert_id1_mv (MoveTo): Enum indicating the move direction for the first vertebra (default is MoveTo.TOP).
        vert_id2_mv (MoveTo): Enum indicating the move direction for the second vertebra (default is MoveTo.BOTTOM).
        use_ivd_direction: Uses the IVD direction instead of the Vertebra direction for Lumbar and Thorax region.
        project_2D (bool, optional): If True, the calculation is done in 2D projection; otherwise, in 3D. Defaults to True.

    Returns:
        list: A list of tuples, each containing:
            - max_angle (float): The Cobb angle identified between two vertebrae that exceeds the threshold.
            - from_vert (int): The vertebra ID at which the angle originates.
            - to_vert (int): The vertebra ID at which the angle terminates.
            - apex (int or None): The vertebra ID that is the apex of the curvature.

    Notes:
        - The function splits the list of vertebrae and recursively calculates Cobb angles for sublists.
        - Only angles that are above the specified threshold are added to the output list.
        - It uses `compute_max_cobb_angle` to find the maximum Cobb angle within a given set of vertebrae.

    Example:
        To compute multiple Cobb angles for a given POI object with a threshold of 15 degrees:

        >>> curves = compute_max_cobb_angle_multi(poi, threshold_deg=15)
        >>> for curve in curves:
        >>>     print(f"Angle: {curve[0]}, From: {curve[1]}, To: {curve[2]}, Apex: {curve[3]}")
        Angle: 18.2, From: 2, To: 6, Apex: 4
        Angle: 12.4, From: 7, To: 10, Apex: 8
    """
    if out_list is None:
        out_list = []
    if vertebrae_list is None:
        vertebrae_list = list(Vertebra_Instance.cervical()) + list(Vertebra_Instance.thoracic()) + list(Vertebra_Instance.lumbar())
        vertebrae_list = vertebrae_list[vertebrae_list.index(VERT_START_COBB) :]
    if len(vertebrae_list) <= 2:
        return out_list
    max_angle, from_vert, to_vert, apex = compute_max_cobb_angle(
        poi,
        vertebrae_list=vertebrae_list,
        vert_id1_mv=vert_id1_mv,
        vert_id2_mv=vert_id2_mv,
        use_ivd_direction=use_ivd_direction,
        project_2D=project_2D,
    )  # type: ignore
    # split
    if threshold_deg <= max_angle:
        from_vert: int
        assert from_vert is not None
        assert to_vert is not None
        out_list.append((max_angle, from_vert, to_vert, apex))
        # Exclusive split: neither endpoint of the just-found curve may participate
        # in a subsequent curve. Prevents overlaps like (T1-T5) + (T5-T7); a
        # sibling curve below the current one starts strictly caudal to to_vert,
        # a sibling above ends strictly cranial to from_vert. Drop the ``+ 1``
        # on the ``below`` slice to restore the textbook (endpoint-shared) split.
        above = vertebrae_list[: vertebrae_list.index(Vertebra_Instance(from_vert))]
        below = vertebrae_list[vertebrae_list.index(Vertebra_Instance(to_vert)) + 1 :]
        compute_max_cobb_angle_multi(
            poi,
            above,
            vert_id1_mv=vert_id1_mv,
            vert_id2_mv=vert_id2_mv,
            threshold_deg=threshold_deg,
            out_list=out_list,
            use_ivd_direction=use_ivd_direction,
            project_2D=project_2D,
        )
        compute_max_cobb_angle_multi(
            poi,
            below,
            vert_id1_mv=vert_id1_mv,
            vert_id2_mv=vert_id2_mv,
            threshold_deg=threshold_deg,
            out_list=out_list,
            use_ivd_direction=use_ivd_direction,
            project_2D=project_2D,
        )

    return out_list


def _add_artificial_ivd(poi: POI) -> POI:
    """Insert synthetic IVD landmarks (center + superior/inferior) wherever they are missing.

    For every adjacent pair of vertebrae (upper, lower) that both have a centroid,
    a missing IVD landmark on the ``upper`` vertebra is synthesized as follows,
    using the median endplate centers when available:

    - ``Vertebra_Disc_Superior`` (upper side of the disc)  -> upper's inferior endplate median
      (fallback: upper's centroid).
    - ``Vertebra_Disc_Inferior`` (lower side of the disc)  -> lower's superior endplate median
      (fallback: lower's centroid).
    - ``Vertebra_Disc`` (disc center) -> midpoint of the two above.

    This makes the Cobb / lordosis / kyphosis paths degrade gracefully when a
    single IVD is missing (e.g. severe degeneration), instead of raising
    KeyError.
    """
    inf_med = Location.Additional_Vertebral_Body_Middle_Inferior_Median.value
    sup_med = Location.Additional_Vertebral_Body_Middle_Superior_Median.value
    disc = Location.Vertebra_Disc.value
    disc_sup = Location.Vertebra_Disc_Superior.value
    disc_inf = Location.Vertebra_Disc_Inferior.value

    ordered = [v for v in Vertebra_Instance.order() if (v.value, 50) in poi]
    for i in range(len(ordered) - 1):
        uv = ordered[i].value
        lv = ordered[i + 1].value
        up = np.array(poi[uv, inf_med]) if (uv, inf_med) in poi else np.array(poi[uv, 50])
        lo = np.array(poi[lv, sup_med]) if (lv, sup_med) in poi else np.array(poi[lv, 50])
        if (uv, disc_sup) not in poi:
            poi[uv, disc_sup] = tuple(up)
        if (uv, disc_inf) not in poi:
            poi[uv, disc_inf] = tuple(lo)
        if (uv, disc) not in poi:
            poi[uv, disc] = tuple((up + lo) / 2)
    return poi


def plot_compute_lordosis_and_kyphosis(
    img_path: str | Path | None,
    poi: POI,
    img: Image_Reference,
    seg: Image_Reference | None = None,
    line_len=100,
    project_2D=True,
    curvature_definition=curvature_definition,
) -> tuple[dict[str, float | None], Snapshot_Frame]:
    """Plots and computes the angles of lordosis and kyphosis on a spinal image.

    This function calculates cervical lordosis, thoracic kyphosis, and lumbar lordosis angles
    based on the provided Points of Interest (POI) object. It visualizes these angles on the
    specified image by drawing line segments corresponding to vertebra orientations, and adds
    annotations for the calculated angles.

    Args:
        img_path (str | Path | None): Path to save the generated image. If None, the image is not saved.
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae.
        img (Image_Reference): The reference image on which to plot the angles and lines.
        seg (Image_Reference | None): The segmentation image reference. Optional, can be None.
        line_len (int): The length of the lines representing the vertebrae directions (default is 100).
        project_2D (bool, optional): If True, the angles are computed in the 2D sagittal projection; otherwise in 3D. Defaults to True.
        curvature_definition (dict[str, Def_Curvature], optional): Mapping of output-key name
            → :class:`Def_Curvature` describing which vertebra pair defines each angle.
            Defaults to the module-level ``curvature_definition`` (cervical_lordosis,
            thoracic_kyphosis, lumbar_lordosis). Pass a custom dict to compute a different
            set of segmental angles or to override the ``last_thoracic`` / ``last_lumbar``
            resolution — the output dict's keys mirror this mapping's keys.

    Returns:
        tuple: A tuple containing:
            - out2 (dict): A dictionary with the calculated angles of lordosis and kyphosis, including:
                - "cervical_lordosis" (float): Angle of cervical lordosis.
                - "thoracic_kyphosis" (float): Angle of thoracic kyphosis.
                - "lumbar_lordosis" (float): Angle of lumbar lordosis.
            - snap (Snapshot_Frame): A Snapshot_Frame object containing the plotted image data.

    Notes:
        - Artificial intervertebral discs (IVD) in not present like for CT.
        - The lines drawn indicate vertebra orientations with specified lengths and directions.
        - The function also generates text annotations for each computed angle, positioned at the
          midpoint between the respective vertebrae.
        - If an image path is provided, the snapshot is saved as an image file.

    Example:
        To compute and visualize lordosis and kyphosis angles on a given image with a POI:

        >>> angles, snapshot = plot_compute_lordosis_and_kyphosis("output_path.png", poi, img, seg)
        >>> print(angles)
        {'cervical_lordosis': 34.5, 'thoracic_kyphosis': 42.7, 'lumbar_lordosis': 50.3}
    """
    poi = poi.reorient().rescale_(verbose=False)
    poi = _add_artificial_ivd(poi)
    out = []
    text_out = []
    for definition in curvature_definition.values():
        for id1, vert_id1_mv in [
            (definition.get_start_vert(poi), definition.start_move),
            (definition.get_stop_vert(poi), definition.stop_move),
        ]:
            vert_id1_mv: MoveTo
            if id1 is None or (id1.value, 50) not in poi:
                continue
            s = vert_id1_mv.get_location(id1, poi)
            a = _get_norm(poi, id1, vert_id1_mv, Location.Vertebra_Direction_Posterior, 1)
            if a is None:
                continue
            out.append((id1.value, s, (a[0] * line_len, a[1] * line_len)))
            out.append((id1.value, s, (-a[0] * line_len * 3, -a[1] * line_len * 3)))
    out2 = compute_lordosis_and_kyphosis(poi, project_2D=project_2D)
    for name, v in out2.items():
        if v is None or name not in curvature_definition:
            # Skip auxiliary keys like ``*_apex`` that live alongside the angles
            # in the same dict but have no curve definition of their own.
            continue
        # Apex annotation: mark the apex vertebra body with a star + label so
        # the reader can see which vertebra the ``*_apex`` json key refers to.
        apex_v = out2.get(f"{name}_apex")
        if apex_v is not None and (apex_v, 50) in poi:
            text_out.append((apex_v, ("*apex", -60)))
        id1 = curvature_definition[name].get_start_vert(poi)
        id2 = curvature_definition[name].get_stop_vert(poi)

        # Cranio-caudal midpoint via the anatomical order — arithmetic mean of
        # `.value` breaks for T13 (value 28, ordered after T12 but numbered after
        # S1/COCC), landing the annotation on L1 instead of somewhere thoracic.
        order = Vertebra_Instance.order()
        try:
            i1, i2 = order.index(id1), order.index(id2)
            mid_inst = order[(min(i1, i2) + max(i1, i2)) // 2]
            vert = mid_inst.value
        except ValueError:
            vert = round((id1.value + id2.value) / 2)
        while (vert, 50) not in poi and vert != 0:
            vert -= 1
        text_out.append((vert, (f"{v:.1f}° - {str(name).split('_')[-1]}", 25)))

    poi.info["line_segments_sag"] = out + poi.info.get("line_segments_sag", [])
    poi.info["text_sag"] = text_out + poi.info.get("text_sag", [])
    snap = Snapshot_Frame(img, seg, centroids=poi, show_these_subreg_poi=[100])
    if img_path is not None:
        create_snapshot(img_path, [snap])
    return out2, snap


def plot_cobb_angle(
    img_path: str | Path | None,
    poi: POI,
    img: Image_Reference,
    seg: Image_Reference | None = None,
    line_len=100,
    threshold_deg=10,
    vert_id1_mv: MoveTo = MoveTo.TOP,
    vert_id2_mv: MoveTo = MoveTo.BOTTOM,
    use_ivd_direction=False,
    project_2D=True,
) -> tuple[list[tuple[float, int, int, int | None]], Snapshot_Frame]:
    """Visualize Cobb angles on a spinal image by plotting the maximum angles across the spine.

    Args:
        img_path (str | Path | None): The file path where the output image with plotted angles should be saved.
            If None, the image is not saved.
        poi (POI): An object representing a point of interest that supports indexing with vertebra identifiers
            and returns 3D coordinates.
        img (Image_Reference): The image to plot the cobb angles on.
        seg (Image_Reference | None): Optional segmentation image to be used in conjunction with the main image.
        line_len (int): The length of the line segments used to visualize the direction of cobb angles.
        threshold_deg (int): The angle threshold in degrees above which cobb angles are considered for plotting.
        vert_id1_mv (MoveTo): The MoveTo option for the first vertebra in each angle calculation.
        vert_id2_mv (MoveTo): The MoveTo option for the second vertebra in each angle calculation.
        use_ivd_direction (bool, optional): For lumbar/thoracic ids beyond ``IVD_MORE_ACCURATE``, use the IVD direction
            (via ``Location.Vertebra_Disc_Inferior``) instead of the vertebra direction. Defaults to False.
        project_2D (bool, optional): If True, the underlying Cobb angles are computed as a 2D projection; otherwise in 3D. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - List of angle data and segments for plotting.
            - Snapshot_Frame object containing the final image with cobb angles plotted.

    Notes:
        - The function uses the maximum cobb angle algorithm to find angles and plots them on the sagittal view.
        - It assumes vertebrae are labeled using a standard convention (e.g., C1, T1, L1, etc.).
        - Only angles exceeding the specified threshold are plotted.

    Example:
        To plot the cobb angles and save the output image:

        >>> plot_cobb_angle("output.png", poi, img, seg, line_len=100, threshold_deg=10)
    """
    poi = poi.reorient().rescale_(verbose=False)
    poi = _add_artificial_ivd(poi)

    out = []
    text_out = []
    copps = compute_max_cobb_angle_multi(
        poi,
        threshold_deg=threshold_deg,
        vert_id1_mv=vert_id1_mv,
        vert_id2_mv=vert_id2_mv,
        use_ivd_direction=use_ivd_direction,
        project_2D=project_2D,
    )
    for max_angle, from_vert, to_vert, apex in copps:
        if from_vert is not None:
            for id1, mv in zip([from_vert, to_vert], [vert_id1_mv, vert_id2_mv]):
                c = mv.get_location(id1, poi)

                # Always go through Vertebra_Direction_Right so _get_norm routes
                # to the endplate-plane right direction (chain-closed across the
                # shared disc: `_endplate_r_direction` averages both flanking
                # endplates). The old ``use_ivd_direction`` branch pulled
                # ``Vertebra_Disc_Inferior`` from each vertebra separately —
                # T9-BOTTOM used the T9/T10 disc but T10-TOP used the T10/T11
                # disc, so the same anatomic boundary got two different lines.
                a = _get_norm(poi, id1, mv, Location.Vertebra_Direction_Right)

                assert a is not None
                out.append((apex, c, (-a[2] * line_len, a[1] * line_len)))
                out.append((apex, c, (a[2] * line_len, -a[1] * line_len)))
        if apex is not None:
            # Align the label with the disc below the apex vertebra (IVD height)
            # rather than the vertebra body centre, so the text sits at the same
            # cranio-caudal level as the drawn Cobb line at the apex.
            cord = poi[apex, Location.Vertebra_Disc.value] if (apex, Location.Vertebra_Disc.value) in poi else poi[apex, 50]
            s = f"copp angle\n{max_angle:.1f}° {Vertebra_Instance(from_vert)} - {Vertebra_Instance(to_vert)}"
            text_out.append((apex, (s, 35, cord[1])))
        poi.info["line_segments_cor"] = out + poi.info.get("line_segments_cor", [])
        poi.info["text_cor"] = text_out + poi.info.get("text_cor", [])

    axis = poi.get_axis("R")
    width = poi.shape[axis] / poi.zoom[axis] / 2
    min_half_width_mm = 80
    if width < min_half_width_mm:
        padd = [(0, 0) for _ in range(3)]
        padd[axis] = (int(min_half_width_mm - width), int(min_half_width_mm - width))
        img = to_nii(img).apply_pad(padd, verbose=False)
        seg = to_nii(seg, True).apply_pad(padd, verbose=False)
        poi = poi.resample_from_to(seg)
    frame = Snapshot_Frame(
        img,
        seg,
        centroids=poi,
        sagittal=False,
        coronal=True,
        show_these_subreg_poi=[100],
    )
    if img_path is not None:
        create_snapshot(img_path, [frame])
    return copps, frame


def plot_cobb_and_lordosis_and_kyphosis(
    jpg_path: str | Path | None,
    poi: POI | Path,
    img: Image_Reference,
    seg: Image_Reference | None = None,
    line_len=100,
    threshold_deg=10,
    project_2D=True,
) -> tuple[list, dict[str, float | None], list[Snapshot_Frame]]:
    """Plots Cobb angles and lordosis/kyphosis angles on a spinal image.

    This function calculates and visualizes both the Cobb angles for spinal curvature and the angles
    of cervical lordosis, thoracic kyphosis, and lumbar lordosis. It overlays these visualizations
    on the provided spinal image and can save the resulting image to a specified path.

    Args:
        jpg_path (str | Path | None): Path to save the generated image. If None, the image is not saved.
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae.
        img (Image_Reference): The reference image on which to plot the angles and lines.
        seg (Image_Reference | None): The segmentation image reference. Optional, can be None.
        line_len (int): The length of the lines representing the vertebrae directions (default is 100).
        threshold_deg (int): The threshold angle in degrees to identify significant Cobb angles (default is 10).
        project_2D (bool, optional): If True, the underlying angles are computed as a 2D projection; otherwise in 3D. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - out_cobb (list): A list of tuples for each significant Cobb angle found, each with:
                - max_angle (float): The maximum Cobb angle in the segment.
                - from_vert (int): The vertebra ID at the start of the Cobb angle measurement.
                - to_vert (int): The vertebra ID at the end of the Cobb angle measurement.
                - apex (int | None): The vertebra ID of the apex of the curvature.
            - out_lak (dict): A dictionary with the calculated angles of lordosis and kyphosis, including:
                - "cervical_lordosis" (float): Angle of cervical lordosis.
                - "thoracic_kyphosis" (float): Angle of thoracic kyphosis.
                - "lumbar_lordosis" (float): Angle of lumbar lordosis.
            - frames (list): A list containing the generated `Snapshot_Frame` objects for each plot.

    Notes:
        - This function internally calls `plot_cobb_angle` and `plot_compute_lordosis_and_kyphosis`
          to generate the respective plots.
        - The function combines both visualizations into a single output image if a path is specified.
        - It effectively allows for simultaneous assessment of scoliosis (via Cobb angles) and sagittal
          plane curvatures (lordosis and kyphosis).

    Example:
        To visualize and save both Cobb angles and lordosis/kyphosis angles:

        >>> cobb_angles, lordosis_kyphosis, frames = plot_cobb_and_lordosis_and_kyphosis(
        ...     "output_path.png", poi, img, seg, line_len=150, threshold_deg=15
        ... )
        >>> print(cobb_angles)
        [(22.3, 3, 12, 7), (18.5, 13, 17, 15)]
        >>> print(lordosis_kyphosis)
        {'cervical_lordosis': 35.2, 'thoracic_kyphosis': 41.5, 'lumbar_lordosis': 48.1}
    """
    if not isinstance(poi, POI):
        poi = POI.load(poi)
    out_cobb, frame1 = plot_cobb_angle(
        None,
        poi,
        img,
        seg,
        line_len=line_len,
        threshold_deg=threshold_deg,
        use_ivd_direction=True,
        project_2D=project_2D,
    )
    out_lak, frame2 = plot_compute_lordosis_and_kyphosis(None, poi, img, seg, line_len=line_len, project_2D=project_2D)
    if jpg_path is not None:
        create_snapshot(jpg_path, [frame1, frame2])
    return out_cobb, out_lak, [frame1, frame2]
