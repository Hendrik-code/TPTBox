import math

import numpy as np
from numpy.linalg import norm

from TPTBox import POI
from TPTBox.core.nii_wrapper import NII, to_nii
from TPTBox.core.poi_fun.vertebra_pois_non_centroids import Print_Logger, ray_cast_pixel_level_from_poi, ray_cast_pixel_lvl
from TPTBox.core.vert_constants import COORDINATE, DIRECTIONS, Location, v_idx2name, v_name2idx
from TPTBox.logger.log_file import Logger_Interface


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


# def norm(a):
#    a = np.array(a)
#    return unit_vector(a)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

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


def get_to_space(P, I, R):
    from_space = np.stack([P, I, R], axis=1)
    to_space = np.linalg.inv(from_space)
    return to_space, from_space


def compute_2D_angel_between_two_points_(poi: POI, id1, id2, direction: DIRECTIONS, subreg=100):
    """
    Computes the 2D and 3D angles between two points in a given anatomical structure,
    such as vertebrae, based on specified directions. This function is particularly
    useful for calculating coplanar angles, lordosis, and kyphosis, depending on the direction specified.

    Args:
        poi (POI): An object representing a point of interest that supports indexing
            with point IDs and returns 3D coordinates. Must have methods `reorient_()`
            and `rescale_()` to prepare the data.
        id1 (int): The identifier for the first point of interest.
        id2 (int): The identifier for the second point of interest.
        d (DIRECTIONS): The direction in which to compute the angle. Possible values are:
            - "P" for Posterior, used for calculating lordosis and kyphosis.
            - "A" for Anterior.
            - "R" for Right, used for calculating coplanar angles.
            - "L" for Left.
            - "S" for Superior.
            - "I" for Inferior.
        subreg (int, optional): The subregion index used for finer granularity in calculations.
            Defaults to 100. Determines the level of detail for calculating normals.

    Returns:
        tuple: A tuple containing two elements:
            - angle_2D (float): The 2D angle between the two points in degrees.
            - angle_3D (float): The 3D angle between the two points in degrees.

    Raises:
        NotImplementedError: If the direction `d` is not one of the recognized values.

    Notes:
        - Coplanar angles (referred to as "copangles") are computed when "R" or "L" is specified.
        - Lordosis and kyphosis measurements are computed when "P" is used.

    Example:
        To compute the coplanar angle between two vertebrae with IDs 20 and 21:

        >>> compute_2D_angel_between_two_points_(poi, 20, 21, "R")

        To compute the lordosis angle between two vertebrae with IDs 20 and 21:

        >>> compute_2D_angel_between_two_points_(poi, 20, 21, "P")
    """
    # Ensure id1 is less than id2
    if id1 > id2:
        id1, id2 = id2, id1
    # Reorient and rescale the POI data
    poi.reorient_().rescale_()
    # Determine direction-specific settings
    if direction in ["P", "A"]:
        # Note: inf - The value of the direction does noting
        location = Location.Vertebra_Direction_Posterior
        inv = 1 if direction == "P" else -1
    elif direction in ["R", "L"]:
        location = Location.Vertebra_Direction_Right
        inv = 1 if direction == "R" else -1
    elif direction in ["S", "I"]:
        location = Location.Vertebra_Direction_Inferior
        inv = 1 if direction == "I" else -1
    else:
        raise NotImplementedError(f"Direction '{direction}' is not recognized.")
        # Calculate normals for the vertebrae

    norm1_vert = norm(np.array(poi[id1, 50]) - np.array(poi[id1, location])) * inv
    norm2_vert = norm(np.array(poi[id2, 50]) - np.array(poi[id2, location])) * inv

    # Average normals across vertebrae if necessary
    if subreg == 100:
        if (id1 + 1, location) in poi:
            norm1_vert_2 = norm(np.array(poi[id1 + 1, 50]) - np.array(poi[id1 + 1, location])) * inv
            norm1_vert = (norm1_vert + norm1_vert_2) / 2
        if (id2 + 1, location) in poi:
            norm2_vert_2 = norm(np.array(poi[id2 + 1, 50]) - np.array(poi[id2 + 1, location])) * inv
            norm2_vert = (norm2_vert + norm2_vert_2) / 2

    # if _debug_plot is not None:
    #    nii_old = _debug_plot.copy()
    #    _debug_plot.reorient_().rescale_()
    #    _debug_plot = _debug_plot * 0
    #    plot_ray(poi[id1, subreg], norm1_vert, _debug_plot, inplace=True, value=1)
    #    plot_ray(poi[id2, subreg], norm2_vert, _debug_plot, inplace=True, value=2)
    # Calculate the 3D angle between the normals
    angle_3D = angle_between(norm1_vert, norm2_vert)  # noqa: N806
    if direction in ["S", "I"]:
        a = norm(np.array(poi[id1, 50]) - np.array(poi[id1, Location.Vertebra_Direction_Right]))
        b = norm(np.array(poi[id2, 50]) - np.array(poi[id2, Location.Vertebra_Direction_Right]))
        norm_down = (a + b) / 2
    if (id2, subreg) not in poi or (id1, subreg) not in poi:
        # CT does not compute IVD so we use Vertebra body instead
        # This only computed the R/L plane of the spine, so we can fall back to vertebra body
        norm_down = norm(np.array(poi[id2, 50]) - np.array(poi[id1, 50]))
    else:
        norm_down = norm(np.array(poi[id2, subreg]) - np.array(poi[id1, subreg]))
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
    return angle_2D / np.pi * 180, angle_3D / np.pi * 180


if __name__ == "__main__":
    from TPTBox import POI
    from TPTBox.core.poi_fun.vertebra_pois_non_centroids import calc_center_spinal_cord

    poi = POI.load(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives/sub-mc0001/ses-20191015/sub-mc0001_ses-20191015_sequ-20198_seg-spine_poi.json"
    )
    nii = to_nii(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-Cancer/derivatives_spine_r/sub-mc0001/ses-20191015/sub-mc0001_ses-20191015_sequ-20198_seg-vert_msk.nii.gz",
        True,
    )
    for i in poi.keys_region():
        if (i + 1, 50) in poi and (i, 50) in poi:
            poi[i, 100] = tuple((np.array(poi[i, 50]) + np.array(poi[i + 1, 50])) / 2)
    print(compute_2D_angel_between_two_points_(poi, v_name2idx["L1"], v_name2idx["L4"], "S", subreg=100))
