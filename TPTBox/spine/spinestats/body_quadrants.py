from __future__ import annotations

from pathlib import Path

import numpy as np

from TPTBox import Image_Reference, Location, calc_poi_from_subreg_vert, to_nii


def make_quadrants(
    vert: Image_Reference,
    spine: Image_Reference,
    poi_buffer: str | Path | None = None,
    vert_ids: list[int] | None = None,
    mask_ids=(49, 50, 52),
    erode=0,
):
    """
    Subdivide vertebral body masks into anatomically oriented 3×3×3 regions.

    This function computes point-of-interest (POI) landmarks for each vertebra,
    constructs a local vertebra-centric coordinate system, and partitions the
    vertebral body mask into 27 spatially consistent subregions.

    The subdivision is performed in vertebra-local coordinates rather than
    image coordinates, making it robust to subject pose, spinal curvature,
    and acquisition orientation.

    Notes
    -----
    - Only vertebral body voxels are considered (intersection of vertebra and
      spine body labels).
    - Each vertebra is processed independently.
    - Regions are encoded with integer labels from 1 to 27.
    - The output image contains all processed vertebrae combined.

    Coordinate system definition
    -----------------------------
    For each vertebra, the local axes are defined as:

    - X-axis (Left–Right):
        Vector from left to right muscle insertion POIs.
    - Z-axis (Inferior–Superior):
        Vector from inferior to superior median vertebral body POIs.
    - Y-axis (Anterior–Posterior):
        Derived via cross product to ensure orthogonality.

    The axes are optionally flipped to match the desired orientation
    convention.

    Partitioning strategy
    ---------------------
    All vertebral body voxels are projected into the local coordinate system.
    Along each axis, the voxel distribution is split into three equal-sized
    bins using quantiles, resulting in a 3×3×3 subdivision.

    Parameters
    ----------
    vert : Image_Reference
        Vertebra segmentation image. Each vertebra must have a unique label.
    spine : Image_Reference
        Spine segmentation image containing vertebral body labels
        (e.g. 49, 50, 52).
    poi_buffer : str or Path or None, optional
        Optional buffer file for caching POI computation results.
        Useful for repeated processing. Default is None.
    vert_ids : list of int or None, optional
        List of vertebra IDs to process. If None, all vertebrae found
        in the segmentation are processed.

    Returns
    -------
    Image_Reference
        An image where each vertebral body voxel is labeled with a value
        from 1 to 27, representing its anatomical subregion.

    Raises
    ------
    None
        Vertebrae missing required POIs are silently skipped.

    Examples
    --------
    >>> out = make_quadrants(vert_seg, spine_seg)
    >>> out.save("vertebra_quadrants.nii.gz")

    """  # noqa: RUF002
    vert_nii = to_nii(vert, True)
    spine_nii = to_nii(spine, True)
    orientation_org = vert_nii.orientation
    orientation = ("L", "A", "S")

    vert_nii = vert_nii.reorient(orientation)
    spine_nii = spine_nii.reorient(orientation)

    poi = calc_poi_from_subreg_vert(
        vert_nii,
        spine_nii,
        subreg_id=[
            Location.Vertebra_Corpus,
            Location.Vertebra_Direction_Inferior,
            Location.Vertebra_Direction_Inferior,
        ],
        buffer_file=poi_buffer,
    )
    out_nii = vert_nii * 0
    mask_nii = spine_nii.extract_label(mask_ids)
    if erode != 0:
        mask_nii = mask_nii.erode_msk(erode)

    for v_id in vert_nii.unique() if vert_ids is None else vert_ids:
        v21 = vert_nii.extract_label(v_id) * mask_nii
        try:
            # POIs
            center = np.array(poi[v_id, Location.Vertebra_Corpus])
            right = np.array(poi[v_id, Location.Vertebra_Direction_Right])
            inf = np.array(poi[v_id, Location.Vertebra_Direction_Inferior])

        except KeyError:
            continue

        # ant = np.array(poi[v_id, Location.Additional_Vertebral_Body_Anterior_Central_Median])
        # post = np.array(poi[v_id, Location.Additional_Vertebral_Body_Posterior_Central_Median])

        def normalize(v):
            return v / np.linalg.norm(v)

        # Local axes
        x_axis = normalize(right - center)  # left → right
        z_axis = normalize(center - inf)  # inferior → superior
        y_axis = normalize(np.cross(z_axis, x_axis))  # anterior ↔ posterior
        if "R" in orientation:
            x_axis *= -1
        if "S" in orientation:
            z_axis *= -1
        if "P" in orientation:
            y_axis *= -1

        # Re-orthogonalize (important!)
        x_axis = normalize(np.cross(y_axis, z_axis))

        mask = v21.get_array() > 0
        coords = np.array(np.nonzero(mask)).T  # voxel indices (i,j,k)

        # convert voxel indices → world coordinates
        coords_world = np.array([v21.affine @ np.append(c, 1.0) for c in coords])[:, :3]
        rel = coords_world - center

        proj_x = rel @ x_axis
        proj_y = rel @ y_axis
        proj_z = rel @ z_axis

        def split_into_3(values):
            q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
            return np.digitize(values, [q1, q2])  # → 0,1,2

        ix = split_into_3(proj_x)  # L
        iy = split_into_3(proj_y)  # A
        iz = split_into_3(proj_z)  # S
        chunk_id = ix + 3 * iy + 9 * iz + 1
        out = np.zeros_like(mask, dtype=np.uint8)
        out[coords[:, 0], coords[:, 1], coords[:, 2]] = chunk_id

        out_nii[out != 0] = out[out != 0]
    return out_nii.reorient(orientation_org)
