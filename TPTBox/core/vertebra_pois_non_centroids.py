from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist

from TPTBox import NII, POI, Log_Type, Logger_Interface, Print_Logger, calc_centroids_from_subreg_vert
from TPTBox.core.vert_constants import Directions, Location, plane_dict

__log = Print_Logger()


def run_poi_pipeline(vert: NII, subreg: NII, poi_path: Path, logger: Logger_Interface = __log):
    poi = calc_centroids_from_subreg_vert(
        vert, subreg, buffer_file=poi_path, save_buffer_file=True, subreg_id=list(range(40, 51)), use_vertebra_special_action=True
    )
    compute_non_centroid_pois(poi, [Location(i) for i in range(61)], vert, subreg, log=logger)
    compute_non_centroid_pois(poi, [Location(i) for i in range(81, 90)], vert, subreg, log=logger)
    compute_non_centroid_pois(poi, [Location(i) for i in range(101, 125)], vert, subreg, log=logger)
    poi.save(poi_path)
    # poi.extract_subregion()
    # a, b = poi.make_point_cloud_nii(s=6)
    # arr = vert.extract_label(20).get_array()
    # arr2 = a.get_array()
    # arr[arr2 != 0] = 0
    # (a.set_array(np.clip(arr, 0, 1) + a.get_array())).save("vert.nii.gz")
    # (b.set_array(np.clip(arr, 0, 1) + b.get_array())).save("subreg.nii.gz")


extreme_points: dict[int, tuple[Directions, int]] = {
    Location.Muscle_Inserts_Spinosus_Process.value: (
        "P",
        Location.Spinosus_Process.value,
    ),  # 81
    Location.Muscle_Inserts_Transverse_Process_right.value: (
        "R",
        Location.Costal_Process_Right.value,
    ),  # 82
    Location.Muscle_Inserts_Transverse_Process_left.value: (
        "L",
        Location.Costal_Process_Left.value,
    ),  # 83
    Location.Muscle_Inserts_Articulate_Process_Inferior_left.value: (
        "I",
        Location.Inferior_Articular_Left.value,
    ),  # 86
    Location.Muscle_Inserts_Articulate_Process_Inferior_right.value: (
        "I",
        Location.Inferior_Articular_Right.value,
    ),  # 87
    Location.Muscle_Inserts_Articulate_Process_Superior_left.value: (
        "S",
        Location.Superior_Articular_Left.value,
    ),  # 88
    Location.Muscle_Inserts_Articulate_Process_Superior_right.value: (
        "S",
        Location.Superior_Articular_Right.value,
    ),  # 89
}
extreme_points2: dict[int, Location] = {
    Location.Muscle_Inserts_Spinosus_Process.value: Location.Arcus_Vertebrae,  # 81
    Location.Muscle_Inserts_Transverse_Process_right.value: Location.Arcus_Vertebrae,  # 82
    Location.Muscle_Inserts_Transverse_Process_left.value: Location.Arcus_Vertebrae,  # 83
    Location.Muscle_Inserts_Articulate_Process_Inferior_left.value: Location.Arcus_Vertebrae,  # 86
    Location.Muscle_Inserts_Articulate_Process_Inferior_right.value: Location.Arcus_Vertebrae,  # 87
    Location.Muscle_Inserts_Articulate_Process_Superior_left.value: Location.Spinosus_Process,  # 88 #Arcus_Vertebrae cause it to be behind not on top
    Location.Muscle_Inserts_Articulate_Process_Superior_right.value: Location.Spinosus_Process,  # 89 #Arcus_Vertebrae cause it to be behind not on top
}
# Special 125 127 Ligament_Attachment_Point_Flava_Superior_Median
# 101 - 124 vertebra_body points
# 84,85  horizontal_plane_landmarks (vertebra_body)


line_cast: dict[int, tuple[list[Location], Location, Location, Location]] = {
    Location.Muscle_Inserts_Vertebral_Body_right.value: (
        [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border],  # possible pixel
        Location.Vertebra_Corpus,  # start point
        Location.Superior_Articular_Left,  # norm point 2
        Location.Superior_Articular_Right,  # norm point 1
    ),
    Location.Muscle_Inserts_Vertebral_Body_left.value: (
        [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border],
        Location.Vertebra_Corpus,
        Location.Superior_Articular_Right,
        Location.Superior_Articular_Left,
    ),
    # Location.Muscle_Inserts_Spinosus_Process.value: (
    #    [Location.Spinosus_Process],
    #    Location.Arcus_Vertebrae,
    #    Location.Arcus_Vertebrae,
    #    Location.Spinosus_Process,
    # ),  # 81
}


# TODO remove _old and legacy_code
def compute_non_centroid_pois(
    poi: POI,
    locations: Sequence[Location] | Location,
    vert: NII,
    subreg: NII,
    _vert_ids: tuple[int, ...] | None = None,
    log: Logger_Interface = __log,
    legacy_code=False,
):
    # TODO Test if the cropping to the vert makes it slower or faster???
    if not isinstance(locations, Sequence):
        locations = [locations]
    log.print("[*] Calc pois from subregion id", [l.name for l in locations])
    if Location.Spinal_Canal in locations:
        subregs_ids = subreg.unique()

        if (
            Location.Spinal_Canal.value in subregs_ids or Location.Spinal_Cord.value in subregs_ids
        ) and Location.Spinal_Canal.value not in poi.keys_subregion():
            calc_center_spinal_cord(poi, subreg)
    if Location.Spinal_Canal_ivd_lvl in locations:
        subregs_ids = subreg.unique()
        v = Location.Spinal_Canal_ivd_lvl.value
        if (v in subregs_ids or Location.Spinal_Cord.value in subregs_ids) and v not in poi.keys_subregion():
            calc_center_spinal_cord(poi, subreg, source_subreg_point_id=Location.Vertebra_Disc, subreg_id=Location.Spinal_Canal_ivd_lvl)
    if _vert_ids is None:
        _vert_ids = vert.unique()
    for label_id in _vert_ids:
        if label_id >= 26:
            continue
        current_vert = vert.extract_label(label_id)
        bb = current_vert.compute_crop()
        current_vert.apply_crop_(bb)
        current_subreg = subreg.apply_crop(bb) * current_vert
        for location in locations:
            if location.value <= 50:
                continue
            if location.value == Location.Spinal_Canal.value:
                continue
            if location.value == Location.Spinal_Canal_ivd_lvl.value:
                continue
            if (label_id, location.value) in poi:
                continue
            if location.value in extreme_points:  # 83
                direction, loc_id = extreme_points[location.value]  # TODO merge extreme_points and extreme_points2
                region = current_subreg.extract_label(loc_id)
                if region.sum() == 0:
                    log.print(
                        f"reg={label_id},subreg={loc_id} is missing (extreme_points)",
                        ltype=Log_Type.FAIL,
                    )
                    continue

                if legacy_code:
                    extreme_point = get_extreme_point_old(region, direction)
                else:
                    extreme_point = get_extreme_point(poi, region, label_id, bb, extreme_points2[location.value])
                    if extreme_point is None:
                        continue
                poi[label_id, location.value] = tuple(a.start + b for a, b in zip(bb, extreme_point, strict=True))
            # elif location.value == Location.Vertebra_Disc_Posterior.value:
            #    vert_arr = vert.extract_label(label_id + 100)
            #    if vert_arr.sum() == 0:
            #        continue
            #    extreme_point = get_extreme_point_old(vert_arr, "P")
            #    ids = [plane_dict[i] for i in vert_arr.orientation]
            #    axis = ids.index(plane_dict["P"])
            #    out = list(poi[label_id, 100])
            #    out[axis] = extreme_point[axis]
            #    poi[label_id, location.value] = tuple(out)  # tuple(a.start + b for a, b in zip(bb, extreme_point))
            elif location.value in line_cast:
                regions_loc, start_point, a, b = line_cast[location.value]
                normal_vector_points = (a, b)
                region = current_subreg.extract_label(regions_loc[0].value)
                for r in regions_loc[1:]:
                    region += current_subreg.extract_label(r.value)
                if legacy_code:
                    horizontal_plane_landmarks_old(poi, region, label_id, bb, log)
                else:
                    extreme_point = max_distance_ray_cast(poi, region, label_id, bb, normal_vector_points, start_point, log=log)
                    if extreme_point is None:
                        continue
                    poi[label_id, location.value] = tuple(a.start + b for a, b in zip(bb, extreme_point, strict=True))
            elif (
                location.value >= Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median.value
                and location.value <= Location.Ligament_Attachment_Point_Flava_Inferior_Median.value
                and location.value != 126
            ):
                vertebra_body(poi, current_vert, current_subreg, label_id, bb, log)

            else:
                raise NotImplementedError(location.value)


def get_extreme_point_old(region: NII, direction: Directions):
    # TODO rotation invariant ???
    idx = [plane_dict[i] for i in region.orientation]
    axis = idx.index(plane_dict[direction])
    arr = region.get_array()
    index = np.argmin(np.nonzero(arr)[axis]) if direction != region.orientation[axis] else np.argmax(np.nonzero(arr)[axis])
    return (np.nonzero(arr)[0][index], np.nonzero(arr)[1][index], np.nonzero(arr)[2][index])


def get_extreme_point(
    poi: POI,
    region: NII,
    label,
    bb: tuple[slice, slice, slice],
    anti_point: Location,
    log: Logger_Interface = __log,
):
    """
    inputs:
        p: the chose point of interest (as array)
        sr_msk: an array containing the subregion mask
        region_label: the label if the region of interest in sr_msk
    output:
        out_point: the point from sr_msk[region_label] closest to point p (as array)
    """
    p = _to_local_np(anti_point, bb, poi, label, log=log)
    if p is None:
        return None
    p = np.expand_dims(p.astype(int), 1)
    # p, sr_msk, region_label
    locs = np.where(region.get_array() == 1)
    locs_array = np.array(list(locs)).T
    distances = cdist(p.T, locs_array)

    return locs_array[distances.argmax()]


def get_closets_point(
    poi: POI,
    region: NII,
    label,
    bb: tuple[slice, slice, slice],
    close_point: Location,
    log: Logger_Interface = __log,
):
    """
    inputs:
        p: the chose point of interest (as array)
        sr_msk: an array containing the subregion mask
        region_label: the label if the region of interest in sr_msk
    output:
        out_point: the point from sr_msk[region_label] closest to point p (as array)
    """
    p = _to_local_np(close_point, bb, poi, label, log=log)
    if p is None:
        return None
    p = np.expand_dims(p.astype(int), 1)
    # p, sr_msk, region_label
    locs = np.where(region.get_array() == 1)
    locs_array = np.array(list(locs)).T
    distances = cdist(p.T, locs_array)

    return locs_array[distances.argmin()]


def horizontal_plane_landmarks_old(poi: POI, region: NII, label, bb: tuple[slice, slice, slice], log: Logger_Interface = __log):
    # TODO rotation invariant ???
    """Taking the Vertebra Corpuse, we compute the most left/right point that is still in the segmentation. The given rotation makes the 2D Plane"""
    a = (label, Location.Muscle_Inserts_Vertebral_Body_right.value) in poi
    b = (label, Location.Muscle_Inserts_Vertebral_Body_left.value) in poi
    if a and b:
        return
    idx = [plane_dict[i] for i in region.orientation]
    axis = idx.index("sag")
    centroid = _to_local_np(Location.Vertebra_Corpus, bb, poi, label, log=log).astype(int)
    arr = region.get_array()
    sli = tuple(x.item() if i != axis else slice(None) for i, x in enumerate(centroid))
    line = arr[sli]  # select a line
    # line = arr[centroid[0], centroid[1], :]  # select a line
    max_index = np.argmax(np.nonzero(line))
    min_index = np.argmin(np.nonzero(line))
    out1 = [c + b.start for c, b in zip(centroid, bb, strict=True)]
    out1[axis] = np.nonzero(line)[0][min_index] + bb[axis].start
    poi[label, Location.Muscle_Inserts_Vertebral_Body_right.value] = tuple(out1)
    out1[axis] = np.nonzero(line)[0][max_index] + bb[axis].start
    poi[label, Location.Muscle_Inserts_Vertebral_Body_left.value] = tuple(out1)


def max_distance_ray_cast(
    poi: POI,
    region: NII,
    reg_label: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | None = None,
    start_point: Location = Location.Vertebra_Corpus,
    log: Logger_Interface = __log,
):
    """Rotation independent"""
    plane_coords, arange = ray_cast(poi, region, reg_label, bb, normal_vector_points, start_point, log=log)
    if plane_coords is None:
        return None
    selected_arr = np.zeros(region.shape)
    selected_arr[plane_coords[..., 0], plane_coords[..., 1], plane_coords[..., 2]] = arange
    selected_arr = selected_arr * region.get_array()
    out = tuple(np.unravel_index(np.argmax(selected_arr, axis=None), selected_arr.shape))
    return out


def _to_local_np(loc: Location, bb: tuple[slice, slice, slice], poi: POI, label, log: Logger_Interface):
    if (label, loc.value) in poi:
        return np.asarray([a - b.start for a, b in zip(poi[label, loc.value], bb, strict=True)])
    log.print(f"region={label},subregion={loc.value} is missing", ltype=Log_Type.FAIL)
    return None


def vertebra_body(poi: POI, vert_region: NII, current_subreg: NII, reg_label: int, bb: tuple[slice, slice, slice], log: Logger_Interface):
    # vert_region = vert_region.extract_label(reg_label)
    corpus = current_subreg.extract_label(49) + current_subreg.extract_label(50)
    add_vertebra_body_points(
        poi=poi,
        vert_region=vert_region,
        current_subreg=current_subreg,
        reg_label=reg_label,
        corpus=corpus,
        shift=0,
        direction="L",
        bb=bb,
        starting_value=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median.value,
        compute_arcus_points=True,
        log=log,
    )
    sup_articular_right = _to_local_np(Location.Superior_Articular_Right, bb, poi, reg_label, log)
    sup_articular_left = _to_local_np(Location.Superior_Articular_Left, bb, poi, reg_label, log)
    factor = 3.0
    if sup_articular_left is None or sup_articular_right is None:
        # fallback if a Superior is missing; TODO Test if we to readjust factor for the neck vertebra
        sup_articular_right = _to_local_np(Location.Inferior_Articular_Right, bb, poi, reg_label, log)
        sup_articular_left = _to_local_np(Location.Inferior_Articular_Left, bb, poi, reg_label, log)
        factor = 2.0
        if sup_articular_left is None or sup_articular_right is None:
            return
    vertebra_width = (sup_articular_right - sup_articular_left) ** 2  # TODO need zoom?
    vertebra_width = np.sqrt(np.sum(vertebra_width))
    shift = vertebra_width / factor
    # print(poi)

    add_vertebra_body_points(
        poi=poi,
        vert_region=vert_region,
        current_subreg=current_subreg,
        reg_label=reg_label,
        corpus=corpus,
        shift=shift,
        direction="L",
        bb=bb,
        starting_value=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left.value,
    )
    add_vertebra_body_points(
        poi=poi,
        vert_region=vert_region,
        current_subreg=current_subreg,
        reg_label=reg_label,
        corpus=corpus,
        shift=shift,
        direction="R",
        bb=bb,
        starting_value=Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right.value,
    )


def ray_cast(
    poi: POI,
    region: NII,
    reg_label: int,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | None = None,
    start_point: Location = Location.Vertebra_Corpus,
    log: Logger_Interface = __log,
):
    start_point_np = _to_local_np(start_point, bb, poi, reg_label, log)
    if start_point_np is None:
        return None, None
    # Compute a normal vector, that defines the plane direction
    if normal_vector_points is None:
        normal_vector_points = (
            Location.Costal_Process_Right,
            Location.Costal_Process_Left,
        )
    try:
        normal_vector = _to_local_np(normal_vector_points[1], bb, poi, reg_label, log) - _to_local_np(
            normal_vector_points[0], bb, poi, reg_label, log
        )
    except TypeError as e:
        print("TypeError", e)
        return None, None
    # normal_vector *= zoom
    normal_vector = normal_vector / norm(normal_vector)
    # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
    start_point_np = start_point_np.copy()
    num_pixel = np.abs(np.floor(np.max((np.array(region.shape) - start_point_np) / normal_vector))).item()
    arange = np.arange(0, num_pixel, step=1, dtype=float)
    coords = [start_point_np[i] + normal_vector[i] * arange for i in [0, 1, 2]]
    for i in [0, 1, 2]:
        cut_off = (region.shape[i] <= np.floor(coords[i])).sum()
        if cut_off == 0:
            cut_off = (np.floor(coords[i]) <= 0).sum()
        if cut_off != 0:
            coords = [c[:-cut_off] for c in coords]
            arange = arange[:-cut_off]
    plane_coords = np.stack(coords, -1).astype(int)
    return plane_coords, arange


def add_vertebra_body_points(
    poi: POI,
    vert_region: NII,
    current_subreg: NII,
    corpus: NII,
    reg_label: int,
    shift: int,
    direction: Directions,
    bb: tuple[slice, slice, slice],
    normal_vector_points: tuple[Location, Location] | None = None,
    start_point: Location = Location.Vertebra_Corpus,
    starting_value=101,
    compute_arcus_points=False,
    log: Logger_Interface = __log,
):
    zoom = vert_region.zoom
    start_point_np = _to_local_np(start_point, bb, poi, reg_label, log=log)
    if start_point_np is None:
        return

    # Compute a normal vector, that defines the plane direction
    if normal_vector_points is None:
        normal_vector_points = (Location.Costal_Process_Right, Location.Costal_Process_Left)

    try:
        a = _to_local_np(normal_vector_points[0], bb, poi, reg_label, log=log)
        b = _to_local_np(normal_vector_points[1], bb, poi, reg_label, log=log)
        if a is None or b is None:
            raise TypeError()  # noqa: TRY301
        normal_vector = a - b
    except TypeError:
        try:
            normal_vector_points = (Location.Superior_Articular_Right, Location.Superior_Articular_Left)
            a = _to_local_np(normal_vector_points[0], bb, poi, reg_label, log=log)
            b = _to_local_np(normal_vector_points[1], bb, poi, reg_label, log=log)

            if a is None or b is None:
                raise TypeError()  # noqa: TRY301
            normal_vector = a - b
        except TypeError:
            # raise e
            return

    normal_vector *= zoom
    normal_vector = normal_vector / norm(normal_vector)
    # The main axis will be treated differently

    idx = [plane_dict[i] for i in vert_region.orientation]
    axis = idx.index(plane_dict[direction])
    assert axis == np.argmax(np.abs(normal_vector)).item()
    dims = [0, 1, 2]
    dims.remove(axis)
    dim1, dim2 = dims
    if vert_region.orientation[axis] != direction:
        shift *= -1
    # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
    start_point_np = start_point_np.copy()
    start_point_np[axis] = start_point_np[axis] + shift
    shift_total = -start_point_np.dot(normal_vector)

    xx, yy = np.meshgrid(range(vert_region.shape[dim1]), range(vert_region.shape[dim2]))
    zz = (-normal_vector[dim1] * xx - normal_vector[dim2] * yy - shift_total) * 1.0 / normal_vector[axis]
    z_max = vert_region.shape[axis] - 1
    zz[zz < 0] = 0
    zz[zz > z_max] = 0
    # make cords to array again
    plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3])
    plane_coords[:, :, axis] = zz
    plane_coords[:, :, dim1] = xx
    plane_coords[:, :, dim2] = yy
    plane_coords = plane_coords.astype(int)
    # 1 where the selected subreg is, else 0
    subregion = (vert_region * corpus).get_array()

    plane = subregion[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]

    # subregion = subregion.copy()
    # subregion[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]] += 1
    # vert_region.set_array(subregion).save("/media/data/robert/code/bids/BIDS/test/test_data/test.nii.gz")

    if plane.sum() == 0:
        log.print(reg_label, 101, "add_vertebra_body_points, Plane empty", ltype=Log_Type.STRANGE)
        return
    # compute_corners_of_plane gives 8 points (1-4) are the corers of the Bounding-box that were made by intersecting the plane with the subregion
    # 5-8 are the center points of the lines of that bounding box

    try:
        out_points = compute_corners_of_plane(plane)
        for i, point in enumerate(out_points):
            cords = plane_coords[point[0], point[1], :]
            poi[reg_label, starting_value + i] = tuple(x + y.start for x, y in zip(cords, bb, strict=False))

        if compute_arcus_points:
            loc102 = out_points[1]
            loc104 = out_points[3]
            arcus = (vert_region * current_subreg.extract_label(41)).get_array()
            plane_arcus = arcus[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]]

            loc125 = get_nearest_neighbor(loc102, plane_arcus, 1)  # 41
            cords = plane_coords[loc125[0], loc125[1], :]
            poi[reg_label, 125] = tuple(x + y.start for x, y in zip(cords, bb, strict=False))

            loc127 = get_nearest_neighbor(loc104, plane_arcus, 1)  # 41
            cords = plane_coords[loc127[0], loc127[1], :]
            poi[reg_label, 127] = tuple(x + y.start for x, y in zip(cords, bb, strict=False))
    except ValueError:
        __log.print_error()
        __log.print(vert_region.sum(), reg_label, "125-127")


def get_nearest_neighbor(p, sr_msk, region_label):
    """
    get the coordinates of point x from sr_mask's provided region_label which is closest to point p

    inputs:
        p: the chose point of interest (as array)
        sr_msk: an array containing the subregion mask
        region_label: the label if the region of interest in sr_msk
    output:
        out_point: the point from sr_msk[region_label] closest to point p (as array)
    """
    if len(p.shape) == 1:
        p = np.expand_dims(p, 1)
    locs = np.where(sr_msk == region_label)
    locs_array = np.array(list(locs)).T
    distances = cdist(p.T, locs_array)

    return locs_array[distances.argmin()]


def compute_corners_of_plane(plane: np.ndarray, region_label=1):
    """
    annotate the corners of a rectancular shape in a plane
    TODO there is a bug with sobel/simplify_coords_vw that the first point get mapped to 0,0. We fixed it by calling get_nearest_neighbor
    """
    # TODO replace the many conditions
    from simplification.cutil import simplify_coords_vw
    from skimage.filters import sobel

    plane_tmp = plane.copy()
    plane_tmp[plane_tmp != region_label] = 0
    borders: np.ndarray = sobel(plane_tmp)  # type: ignore
    borders = np.float32(borders)  # type: ignore
    border_coords = np.nonzero(borders)
    border_coords = np.asarray(list(zip(border_coords[0], border_coords[1], strict=True)))
    # simplified = simplify_coords(border_coords, 30.0)
    simplified = simplify_coords_vw(border_coords, 0.001)
    if simplified[0, 0] < 1 and simplified[0, 0] > 0:  # There is a bug that the first value is close to 0
        simplified = simplified[1:]

    if len(simplified.shape) < 2:
        raise ValueError("len < 2 of simplified" + str(simplified.shape))
    fix_point0 = np.zeros_like(simplified)
    fix_point1 = np.zeros_like(simplified)
    fix_point2 = np.zeros_like(simplified)
    fix_point3 = np.zeros_like(simplified)

    fix_point1[:, :] = (0, borders.shape[1])
    fix_point2[:, :] = (borders.shape[0], 0)
    fix_point3[:, :] = (borders.shape[0], borders.shape[1])
    simplified = simplified.astype(int)
    result = np.zeros_like(borders)
    # index starting at 1 because 0 is background
    for fix_point, idx in [(fix_point0, 1), (fix_point1, 2), (fix_point2, 3), (fix_point3, 4)]:
        dist0 = np.sum(np.abs(fix_point - simplified), axis=1)
        point0 = np.argmin(dist0)
        result[simplified[point0, 0], simplified[point0, 1]] = idx

    out = []
    for i in range(1, 5):
        # try:
        # snapped_back_point = get_nearest_neighbor(np.array([x for x in np.where(result == i)]), plane, 1)  # 41
        # except ValueError:
        snapped_back_point = np.array(list(np.where(result == i))).reshape(-1)
        out.append(snapped_back_point)
    for i, j in [(0, 1), (1, 3), (2, 3), (0, 2)]:
        point = (out[i] + out[j]) // 2
        dist0 = np.sum(np.abs(point - border_coords), axis=1)
        point = np.argmin(dist0)
        out.append(border_coords[point])
    return out


def calc_center_spinal_cord(
    poi: POI,
    subreg: NII,
    spline_subreg_point_id=Location.Vertebra_Corpus,
    source_subreg_point_id=Location.Vertebra_Corpus,
    subreg_id=Location.Spinal_Canal,
    intersection_target: list[Location] | None = None,
    _fill_inplace: NII | None = None,
) -> POI:
    """
    Calculate the center of the spinal cord within a specified region.

    Parameters:
    - poi (POI): Point of Interest object containing relevant data.
    - subreg (NII): Neuroimaging subregion data for analysis.
    - spline_subreg_point_id (int, optional): The height of the region to analyze (default is Location.Vertebra_Corpus).
    - subreg_id (int, optional): The identifier for the subregion (default is Location.Spinal_Canal).
    - intersection_target (List[Location], optional): A list of locations for intersection analysis (default is [Location.Spinal_Cord, Location.Spinal_Canal]).

    Returns:
    - POI: Point of Interest object with calculated centroids.

    This function calculates the center of the spinal cord within a specified region using a spline fit to the
    given Point of Interest (POI) data and a NII. It first extracts a subregion of
    interest based on the provided spline_subreg_point_id and then calculates the center using geometric operations.

    Note:
    - The `poi` object should contain the relevant data for spline fitting.
    - The `subreg` object should be a NII subregion containing spine with number 60 and 61.
    - The `subreg_id` parameter is an optional identifier for the subregion.
    - The `intersection_target` parameter allows you to specify the locations to intersect (e.g., spinal cord and spinal canal).

    Example usage:
    ```python
    poi = POI(...)
    subreg = NII(...)
    updated_poi = calc_center_spinal_cord(poi, subreg, spline_subreg_point_id=Location.Vertebra_Corpus, subreg_id=Location.Spinal_Canal, intersection_target=[Location.Spinal_Cord, Location.Spinal_Canal])
    ```
    """
    from TPTBox import calc_centroids

    if intersection_target is None:
        intersection_target = [Location.Spinal_Cord, Location.Spinal_Canal]
    assert _fill_inplace is None or subreg == _fill_inplace
    poi_iso = poi.rescale()
    subreg_iso = subreg.rescale()
    body_spline, body_spline_der = poi_iso.fit_spline(location=spline_subreg_point_id, vertebra=True)
    target_labels = subreg_iso.extract_label(intersection_target).get_array()
    out = target_labels * 0
    fill_back = out.copy() if _fill_inplace is not None else None
    for reg_label, _, cords in poi_iso.extract_subregion(source_subreg_point_id).items():
        # calculate_normal_vector
        distances = np.sqrt(np.sum((body_spline - np.array(cords)) ** 2, -1))
        normal_vector = body_spline_der[np.argmin(distances)]
        normal_vector /= np.linalg.norm(normal_vector)
        # create_plane_coords
        # The main axis will be treated differently
        idx = [plane_dict[i] for i in subreg_iso.orientation]
        axis = idx.index(plane_dict["S"])
        # assert axis == np.argmax(np.abs(normal_vector)).item()
        dims = [0, 1, 2]
        dims.remove(axis)
        dim1, dim2 = dims
        # Make a plane through start_point with the norm of "normal_vector", which is shifted by "shift" along the norm
        start_point_np = np.array(cords)
        start_point_np[axis] = start_point_np[axis]
        shift_total = -start_point_np.dot(normal_vector)
        xx, yy = np.meshgrid(range(subreg_iso.shape[dim1]), range(subreg_iso.shape[dim2]))  # type: ignore
        zz = (-normal_vector[dim1] * xx - normal_vector[dim2] * yy - shift_total) * 1.0 / normal_vector[axis]
        z_max = subreg_iso.shape[axis] - 1
        zz[zz < 0] = 0
        zz[zz > z_max] = 0
        plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3])
        plane_coords[:, :, axis] = zz
        plane_coords[:, :, dim1] = xx
        plane_coords[:, :, dim2] = yy
        plane_coords = plane_coords.astype(int)
        # create_subregion
        # 1 where the selected subreg is, else 0
        select = subreg_iso.get_array() * 0
        select[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]] = 1
        out += target_labels * select * reg_label
        if fill_back is not None:
            fill_back[select == 1] = reg_label

    if fill_back is not None:
        assert _fill_inplace is not None
        subreg_iso = subreg_iso.set_array(fill_back).reorient(("S", "A", "R"))
        fill_back = subreg_iso.get_array()
        x_slice = np.ones_like(fill_back[0]) * np.max(fill_back) + 1
        for i in range(fill_back.shape[0]):
            curr_slice = fill_back[i]
            cond = np.where(curr_slice != 0)
            x_slice[cond] = np.minimum(curr_slice[cond], x_slice[cond])
            fill_back[i] = x_slice

        arr = subreg_iso.set_array(fill_back).reorient(poi.orientation).rescale_(poi.zoom).get_array()
        print(arr.shape, _fill_inplace, fill_back.shape)
        _fill_inplace.set_array_(arr)
    ret = calc_centroids(subreg_iso.set_array(out), subreg_id=subreg_id, extend_to=poi_iso.extract_subregion(subreg_id), inplace=True)
    ret.rescale_(poi.zoom)
    return poi.join_left_(ret)


def make_spine_plot(pois: POI, body_spline, vert_nii: NII, filenames):
    from matplotlib import pyplot as plt

    pois = pois.reorient()
    vert_nii = vert_nii.reorient()
    body_center_list = list(np.array(pois.values()))
    # fitting a curve to the centoids and getting it's first derivative
    plt.figure(figsize=[10, 10])
    plt.imshow(np.swapaxes(np.max(vert_nii.get_array(), axis=vert_nii.get_axis(direction="R")), 0, 1), cmap=plt.cm.gray)
    plt.plot(np.asarray(body_center_list)[:, 0], np.asarray(body_center_list)[:, 1])
    plt.plot(np.asarray(body_spline[:, 0]), np.asarray(body_spline[:, 1]), "-")
    plt.savefig(filenames)


if __name__ == "__main__":
    from TPTBox import to_nii

    vert = to_nii("/media/data/robert/datasets/dataset-verse19/derivatives/sub-verse007/sub-verse007_seg-vert_msk.nii.gz", True)
    subreg = to_nii("/media/data/robert/datasets/dataset-verse19/derivatives/sub-verse007/sub-verse007_seg-subreg_msk.nii.gz", True)
    run_poi_pipeline(vert, subreg, Path("test_poi.json"))
