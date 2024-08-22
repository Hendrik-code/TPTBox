from enum import Enum, auto
from pathlib import Path

import numpy as np

from TPTBox import POI, Image_Reference
from TPTBox.core.nii_wrapper import to_nii
from TPTBox.core.vert_constants import DIRECTIONS, Location, Vertebra_Instance
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

IVD_MORE_ACCURATE = 15
VERT_START_COBB = Vertebra_Instance.C2


class MoveTo(Enum):
    TOP = auto()
    BOTTOM = auto()
    CENTER = auto()

    def has_point(self, v: Vertebra_Instance | int, poi: POI):
        """
        Checks if the given vertebra instance or ID has a specific point
        of interest (POI) based on the current MoveTo position.

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

    def get_location(self, v: Vertebra_Instance | int, poi: POI):
        """
        Determines the anatomical location of a point of interest (POI)
        in a vertebra based on the MoveTo position.

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
            # Test if it has next POINT
            next_vert = v.get_next_poi(poi)
            if next_vert is not None and (next_vert, 50) in poi:
                return (v, subreg, next_vert, subreg)
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
            # Test if it has next POINT
            if prev_vert is not None and (prev_vert, 50) in poi:
                return (v, subreg, prev_vert, subreg)
        return (v, 50)

    def get_point(self, v: Vertebra_Instance | int, poi: POI):
        """
        Retrieves the 3D coordinates of a specific point of interest (POI)
        in a vertebra based on the MoveTo position.

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


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Calculates the angle in radians between two vectors.

    Args:
        v1 (tuple): The first vector.
        v2 (tuple): The second vector.

    Returns:
        float: The angle in radians between vectors 'v1' and 'v2'.

    Examples:
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_to_space(a, b, c):
    """
    Computes the transformation matrices from the input coordinate space to the canonical space
    defined by orthogonal vectors a, b, and c, and vice versa.

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


def cosine_distance(a, b):
    """
    Computes the cosine distance between two vectors.

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
    project_2d=False,
    use_ivd_direction=False,
) -> float | None:
    """
    Computes the 2D and 3D angles between two points in a given anatomical structure,
    such as vertebrae, based on specified directions. This function is particularly
    useful for calculating coplanar angles, lordosis, and kyphosis, depending on the direction specified.

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
        project_2d (bool, optional): If True, computes the 2D projection of the angle. Defaults to False.

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

    id1: int = vert_id1.value if isinstance(vert_id1, Enum) else vert_id1
    id2: int = vert_id2.value if isinstance(vert_id2, Enum) else vert_id2
    if (id1, 50) not in poi or (id2, 50) not in poi:
        return None
    assert id1 != id2, id1

    # Ensure id1 is less than id2
    if id1 > id2:
        id1, id2 = id2, id1
    # Reorient and rescale the POI data
    poi.reorient_().rescale_()
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
    if not project_2d:
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


def compute_lordosis_and_kyphosis(poi: POI, project_2d=False):
    """
    Calculates the angles of cervical lordosis, thoracic kyphosis, and lumbar lordosis based on the given points of interest (POI).

    This function determines the angles formed by specific vertebrae along the spine, which are indicative of spinal curvatures.
    The angles are calculated for three key regions: cervical, thoracic, and lumbar, representing lordosis and kyphosis.

    Args:
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae. It must include
            the vertebra direction information for proper calculation. (Location.Vertebra_Direction_Posterior)
        project_2d (bool): If True, the calculation is done in 2D projection; otherwise, in 3D. Defaults to False.

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - "cervical_lordosis": The angle of cervical lordosis, calculated between C2 and C7.
            - "thoracic_kyphosis": The angle of thoracic kyphosis, calculated between T1 and the last thoracic vertebra.
            - "lumbar_lordosis": The angle of lumbar lordosis, calculated between L1 and the last lumbar vertebra.

    Raises:
        AssertionError: If the required vertebra direction information is not present in the POI.

    Notes:
        - It is essential that the `poi` contains the posterior vertebra direction for accurate angle calculations.
        - Thoracic kyphosis is calculated from T1 to the last thoracic vertebra identified in the POI.
        - Lumbar lordosis is calculated from L1 to the last lumbar vertebra identified in the POI.
    Example:
        To compute the spinal angles for a given POI object:

        >>> angles = compute_lordosis_and_kyphosis(poi, project_2d=True)
        >>> print(angles)
        {'cervical_lordosis': 30.5, 'thoracic_kyphosis': 35.0, 'lumbar_lordosis': 45.2}
    """
    assert (
        Location.Vertebra_Direction_Posterior.value in poi.keys_subregion()
    ), "You need to compute the Direction in the Poi (Location.Vertebra_Direction_Posterior)"
    last_t = _get_last_thoracic(poi)
    last_l = _get_last_lumbar(poi)
    poi = poi.copy()
    cervical = compute_angel_between_two_points_(
        poi,
        Vertebra_Instance.C2,
        Vertebra_Instance.C7,
        "P",
        MoveTo.TOP,
        MoveTo.BOTTOM,
        project_2d,
    )
    thoracic = compute_angel_between_two_points_(poi, Vertebra_Instance.T1, last_t, "P", MoveTo.TOP, MoveTo.BOTTOM, project_2d)
    lumbar = compute_angel_between_two_points_(poi, Vertebra_Instance.L1, last_l, "P", MoveTo.TOP, MoveTo.BOTTOM, project_2d)
    return {
        "cervical_lordosis": cervical,
        "thoracic_kyphosis": thoracic,
        "lumbar_lordosis": lumbar,
    }


def _get_norm(poi: POI, id1, mv: MoveTo, location: Location, inv=1):  # noqa: ARG001
    if isinstance(id1, int):
        id1 = Vertebra_Instance(id1)
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
    # if mix:
    #    # This would mix the angle of two adjacent Vertebra.
    #    if mv == MoveTo.CENTER:
    #        return norm1_vert
    #    elif mv == MoveTo.BOTTOM:
    #        next_vert = id1.get_next_poi(poi)
    #    elif mv == MoveTo.TOP:
    #        next_vert = id1.get_previous_poi(poi)
    #    if next_vert is None:
    #        return norm1_vert
    #    if (next_vert, location) in poi:
    #        norm1_vert_2 = unit_vector(np.array(poi[next_vert, 50]) - np.array(poi[next_vert, location])) * inv
    #        norm1_vert = (norm1_vert + norm1_vert_2) / 2
    return norm1_vert


def _get_last_lumbar(poi: POI):
    if Vertebra_Instance.S1.value not in poi.keys_region():
        return None
    for i in list(reversed(Vertebra_Instance.lumbar()))[:5]:
        if (i.value, 50) in poi:
            return i
    return None


def _get_last_thoracic(poi: POI):
    for i in list(reversed(Vertebra_Instance.thoracic()))[:3]:
        if (i.value, 50) in poi:
            return i
    return None


def compute_max_cobb_angle(
    poi: POI,
    vertebrae_list=None,
    vert_id1_mv: MoveTo = MoveTo.TOP,
    vert_id2_mv: MoveTo = MoveTo.BOTTOM,
    project_2d=False,
    use_ivd_direction=False,
):
    """
    Calculates the maximum Cobb angle from a list of vertebrae using the points of interest (POI).

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
        project_2d (bool): If True, the calculation is done in 2D projection; otherwise, in 3D. Defaults to False.

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

        >>> max_angle, from_vert, to_vert, apex = compute_max_cobb_angle(poi, project_2d=True)
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
                    project_2d,
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
    return max_angle, from_vert, to_vert, apex


def compute_max_cobb_angle_multi(
    poi: POI,
    vertebrae_list=None,
    threshold_deg=10,
    out_list=None,
    vert_id1_mv: MoveTo = MoveTo.TOP,
    vert_id2_mv: MoveTo = MoveTo.BOTTOM,
    use_ivd_direction=False,
):
    """
    Identifies multiple Cobb angles along the spine that exceed a given threshold.

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
    )  # type: ignore
    # split
    if threshold_deg <= max_angle:
        from_vert: int
        assert from_vert is not None
        assert to_vert is not None
        out_list.append((max_angle, from_vert, to_vert, apex))
        above = vertebrae_list[: vertebrae_list.index(Vertebra_Instance(from_vert))]
        below = vertebrae_list[vertebrae_list.index(Vertebra_Instance(to_vert)) :]
        compute_max_cobb_angle_multi(
            poi,
            above,
            vert_id1_mv=vert_id1_mv,
            vert_id2_mv=vert_id2_mv,
            threshold_deg=threshold_deg,
            out_list=out_list,
            use_ivd_direction=use_ivd_direction,
        )
        compute_max_cobb_angle_multi(
            poi,
            below,
            vert_id1_mv=vert_id1_mv,
            vert_id2_mv=vert_id2_mv,
            threshold_deg=threshold_deg,
            out_list=out_list,
            use_ivd_direction=use_ivd_direction,
        )

    return out_list


def _add_artificial_ivd(poi: POI):
    ## ADD IVD if nessasary
    if 100 not in poi.keys_subregion():
        last = None
        last_id = 1
        for j in Vertebra_Instance.order():
            if (j, 50) in poi:
                current = np.array(poi[j, 50])
                if last is not None:
                    poi[last_id, 100] = tuple((last + current) / 2)
                last = current
                last_id = j.value
    #####
    return poi


def plot_compute_lordosis_and_kyphosis(
    img_path: str | Path | None,
    poi: POI,
    img: Image_Reference,
    seg: Image_Reference | None = None,
    line_len=100,
):
    """
    Plots and computes the angles of lordosis and kyphosis on a spinal image.

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
    poi = poi.reorient().rescale_()
    poi = _add_artificial_ivd(poi)
    out = []
    text_out = []
    last_t = _get_last_thoracic(poi)
    last_l = _get_last_lumbar(poi)
    for id1, vert_id1_mv in [
        (Vertebra_Instance.C2, MoveTo.TOP),
        (Vertebra_Instance.T1, MoveTo.TOP),
        (last_t, MoveTo.BOTTOM),
        (last_l, MoveTo.BOTTOM),
    ]:
        vert_id1_mv: MoveTo
        if id1 is None or (id1.value, 50) not in poi:
            continue
        s = vert_id1_mv.get_location(id1, poi)
        a = _get_norm(poi, id1, vert_id1_mv, Location.Vertebra_Direction_Posterior, 1)
        assert a is not None
        out.append((id1.value, s, (a[0] * line_len, a[1] * line_len)))
        out.append((id1.value, s, (-a[0] * line_len * 3, -a[1] * line_len * 3)))
    out2 = compute_lordosis_and_kyphosis(poi)
    for (name, v), id1, id2 in zip(
        out2.items(),
        [Vertebra_Instance.C7, last_t, last_l],
        [Vertebra_Instance.C2, Vertebra_Instance.C7, last_t],
        strict=True,
    ):
        if v is None:
            continue
        if id1 is None or id2 is None or (id1.value, 50) not in poi:
            continue
        vert = round((id1.value + id2.value) / 2)
        while (vert, 50) not in poi and vert != 0:
            vert -= 1
        if (vert, 50) not in poi:
            cord = poi[vert, 50]
            text_out.append((vert, (f"{str(name).split('_')[-1]}: {v:.1f}°", 15, cord[1])))

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
):
    """
    Visualizes the cobb angles for a given spinal image by plotting the angle measurements on the image.
    The function calculates the maximum cobb angles across the spine and displays the results on the image.

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
    poi = poi.reorient().rescale_()
    poi = _add_artificial_ivd(poi)

    out = []
    text_out = []
    copps = compute_max_cobb_angle_multi(
        poi,
        threshold_deg=threshold_deg,
        vert_id1_mv=vert_id1_mv,
        vert_id2_mv=vert_id2_mv,
        use_ivd_direction=use_ivd_direction,
    )
    for max_angle, from_vert, to_vert, apex in copps:
        if from_vert is not None:
            for id1, mv in zip([from_vert, to_vert], [vert_id1_mv, vert_id2_mv], strict=False):
                c = mv.get_location(id1, poi)

                if use_ivd_direction and id1 > IVD_MORE_ACCURATE:
                    norm1_post = _get_norm(poi, id1, mv, Location.Vertebra_Direction_Posterior)
                    a = _get_norm(poi, id1, mv, Location.Vertebra_Disc_Inferior)
                    a = np.cross(a, norm1_post)
                else:
                    a = _get_norm(poi, id1, mv, Location.Vertebra_Direction_Right)

                # print(a, id1, mv, c)

                assert a is not None
                out.append((apex, c, (-a[2] * line_len, a[1] * line_len)))
                out.append((apex, c, (a[2] * line_len, -a[1] * line_len)))
                # a = _get_norm(poi, id1, mv, Location.Vertebra_Disc_Inferior)
                # out.append((apex, c, (a[2] * line_len, -a[1] * line_len)))
        if apex is not None:
            cord = poi[apex, 50]
            s = f"copp angle: {max_angle:.1f}° {Vertebra_Instance(from_vert)} - {Vertebra_Instance(to_vert)}"
            text_out.append((apex, (s, 25, cord[1])))
        poi.info["line_segments_cor"] = out + poi.info.get("line_segments_cor", [])
        poi.info["text_cor"] = text_out + poi.info.get("text_cor", [])
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
    img_path: str | Path | None,
    poi: POI,
    img: Image_Reference,
    seg: Image_Reference | None = None,
    line_len=100,
    threshold_deg=10,
):
    """
    Plots Cobb angles and lordosis/kyphosis angles on a spinal image.

    This function calculates and visualizes both the Cobb angles for spinal curvature and the angles
    of cervical lordosis, thoracic kyphosis, and lumbar lordosis. It overlays these visualizations
    on the provided spinal image and can save the resulting image to a specified path.

    Args:
        img_path (str | Path | None): Path to save the generated image. If None, the image is not saved.
        poi (POI): The points of interest object containing 3D coordinates for various vertebrae.
        img (Image_Reference): The reference image on which to plot the angles and lines.
        seg (Image_Reference | None): The segmentation image reference. Optional, can be None.
        line_len (int): The length of the lines representing the vertebrae directions (default is 100).
        threshold_deg (int): The threshold angle in degrees to identify significant Cobb angles (default is 10).

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
    out_cobb, frame1 = plot_cobb_angle(
        None,
        poi,
        img,
        seg,
        line_len=line_len,
        threshold_deg=threshold_deg,
        use_ivd_direction=True,
    )
    out_lak, frame2 = plot_compute_lordosis_and_kyphosis(None, poi, img, seg, line_len=line_len)
    if img_path is not None:
        create_snapshot(img_path, [frame1, frame2])
    return out_cobb, out_lak, [frame1, frame2]


if __name__ == "__main__":
    from TPTBox import POI, calc_poi_from_subreg_vert
    from TPTBox.spine.statistics.ivd_pois import compute_fake_ivd

    # poi = POI.load(
    #    "/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives_spineps/sub-mc0034/ses-20240312/sub-mc0034_ses-20240312_sequ-206_mod-ct_seg-spine_msk.nii.gz"
    # )
    nii = to_nii(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives_spineps/sub-mc0034/ses-20240312//sub-mc0034_ses-20240312_sequ-206_mod-ct_seg-vert_msk.nii.gz",
        True,
    )
    nii_subreg = to_nii(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives_spineps/sub-mc0034/ses-20240312/sub-mc0034_ses-20240312_sequ-206_mod-ct_seg-spine_msk.nii.gz",
        True,
    )
    nii2 = to_nii(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/rawdata/sub-mc0034/ses-20240312/sub-mc0034_ses-20240312_sequ-206_ct.nii.gz",
        False,
    )
    poi = calc_poi_from_subreg_vert(nii, nii_subreg, subreg_id=[Location.Vertebra_Direction_Right])

    nii = compute_fake_ivd(nii, nii_subreg, poi=poi)
    nii.save("/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives_spineps/sub-mc0034/ses-20240312/test.nii.gz")
    print(nii.unique())
    poi = calc_poi_from_subreg_vert(
        nii,
        nii_subreg,
        subreg_id=[
            Location.Vertebra_Direction_Right,
            Location.Vertebra_Disc_Inferior,
            Location.Vertebra_Disc,
        ],
    )
    idx = 23

    print(poi.extract_vert(idx))
    # print(_get_norm(poi.rescale(), 24, None, Location.Vertebra_Direction_Right))
    # plot_compute_lordosis_and_kyphosis("test_2.png", poi, nii)
    plot_cobb_angle("test.png", poi, nii2, nii, use_ivd_direction=True)
    plot_cobb_angle("test_old.png", poi, nii2, nii, use_ivd_direction=False)
    from TPTBox.core.poi_fun.ray_casting import add_ray_to_img

    cor, _ = poi.fit_spline(location=50, vertebra=False)
    print(nii.shape)
    print(
        poi[idx, 50],
        unit_vector(np.array(poi[idx, 50]) - np.array(poi[idx, Location.Vertebra_Direction_Right])),
    )
    a = add_ray_to_img(
        poi[idx, 50],
        -np.array(poi[idx, 50]) + np.array(poi[idx, Location.Vertebra_Direction_Right]),
        nii,
        True,
        value=99,
        dilate=2,
    )
    assert a is not None
    a = add_ray_to_img(
        poi[idx, 50],
        -np.array(poi[idx, 50]) + np.array(poi[idx, Location.Vertebra_Direction_Posterior]),
        a,
        True,
        value=100,
        dilate=2,
    )
    assert a is not None
    a = add_ray_to_img(
        poi[idx, 100],
        -np.array(poi[idx, 100]) + np.array(poi[idx, Location.Vertebra_Disc_Inferior]),
        a,
        True,
        value=101,
        dilate=2,
    )
    assert a is not None
    spline = a.copy() * 0
    # spline.rescale_()
    for x, y, z in cor:
        spline[round(x), round(y), round(z)] = 103
    spline.dilate_msk_(2)
    # spline.resample_from_to_(a)
    a[spline != 0] = spline[spline != 0]
    print(a.unique())
    a.save("/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives_spineps/sub-mc0034/ses-20240312/test.nii.gz")
