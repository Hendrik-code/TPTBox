from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from TPTBox import Print_Logger, Vertebra_Instance, calc_poi_from_subreg_vert
from TPTBox.core.compat import zip_strict
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun._help import to_local_np
from TPTBox.core.poi_fun.ray_casting import calculate_pca_normal_np, max_distance_ray_cast_convex
from TPTBox.core.vert_constants import Location

_log = Print_Logger()


def strategy_calculate_up_vector(
    poi: POI,
    current_vert: NII,
    vert_id: int,
    bb: tuple,
    log: object = _log,
) -> POI:
    """Compute IVD superior and inferior extreme points via PCA-based normal estimation.

    Fits a principal component axis through the IVD label (``vert_id + 100``)
    to determine the superior–inferior direction of the disc, then ray-casts to
    the extreme points along that axis.  The results are stored as
    :attr:`~TPTBox.Location.Vertebra_Disc_Inferior` and
    :attr:`~TPTBox.Location.Vertebra_Disc_Superior` in ``poi``.

    Args:
        poi: POI object that already contains the IVD centroid for ``vert_id``.
            Updated in place and returned.
        current_vert: Cropped vertebra NIfTI used for PCA and ray casting.
        vert_id: Integer vertebra region ID whose IVD label is ``vert_id + 100``.
        bb: Bounding-box slices (as returned by
            :meth:`~TPTBox.NII.compute_crop`) used to convert local
            coordinates back to the full image space.
        log: Logger instance used for progress / warning messages.

    Returns:
        The updated :class:`~TPTBox.POI` object.
    """
    center = to_local_np(Location.Vertebra_Disc, bb, poi, vert_id, log)
    if center is None:
        return poi
    try:
        normal_vector = calculate_pca_normal_np(current_vert.rescale().extract_label(vert_id + 100).get_array(), pca_component=2)
    except ValueError:
        return poi
    extreme_point = center + normal_vector * 10

    axis = poi.get_axis("S")
    s_is_poss = int(poi.orientation[axis] == "S")
    below = int(extreme_point[axis] < center[axis])

    ## INFERIOR ##
    if s_is_poss + below == 1:  # xor
        normal_vector *= -1
    # max_distance_ray_cast_convex
    extreme_point = max_distance_ray_cast_convex(current_vert.extract_label(vert_id + 100), center, normal_vector)
    extreme_point_sup = max_distance_ray_cast_convex(current_vert.extract_label(vert_id + 100), center, -normal_vector)
    # extreme_point = center + normal_vector * 10

    assert extreme_point is not None
    assert extreme_point_sup is not None
    poi[vert_id, Location.Vertebra_Disc_Inferior] = tuple(a.start + b for a, b in zip_strict(bb, extreme_point))
    poi[vert_id, Location.Vertebra_Disc_Superior] = tuple(a.start + b for a, b in zip_strict(bb, extreme_point_sup))

    return poi


def _crop(
    i: int,
    verts_ids: list[int],
    vert_full: NII,
    spine_full: NII,
    poi: POI,
) -> tuple | None:
    """Crop the vertebra and spine volumes around vertebra ``i`` and its neighbour."""
    if i >= 100 or (i + 100 in verts_ids):
        return None
    next_id = Vertebra_Instance(i).get_next_poi(verts_ids)
    if next_id is None:
        return None
    crop = vert_full.extract_label([i, next_id.value]).compute_crop(dist=3)
    vert = vert_full.apply_crop(crop)
    spine = spine_full.apply_crop(crop)
    poi = poi.apply_crop(crop)
    return i, verts_ids, vert, spine, poi, crop, next_id


def _process_vertebra_A(idx: int, vert: NII, spine: NII, next_id: int, poi: POI) -> NII | None:
    """Generate the IVD mask between vertebra ``idx`` and ``next_id`` using endplate extraction."""
    from TPTBox.spine.spinestats import endplate_extraction

    try:
        a = endplate_extraction(idx, vert, spine, poi)
        b = endplate_extraction(next_id, vert, spine, poi)
        if a is None or b is None:
            return None
        a = a.extract_label(Location.Vertebral_Body_Endplate_Inferior)
        b = b.extract_label(Location.Vertebral_Body_Endplate_Superior)
    except ValueError:
        return None
    ivd: NII = (a + b).calc_convex_hull(None)
    # ivd: NII = (
    #    (a + b).dilate_msk(1, 1, verbose=False).erode_msk(3, 1, ignore_direction="S", verbose=False)
    #    # .calc_convex_hull(None)
    # )
    # ivd[a != 0] = 2
    # ivd[b != 0] = 3
    # ivd[spine != 0] += 10
    # ivd = ivd.filter_connected_components(1, max_count_component=1, connectivity=1)
    ivd = ivd * (100 + idx)
    return ivd


def _process_vertebra_B(idx: int, vert: NII, spine: NII, next_id: Vertebra_Instance, dilate: int = 1) -> NII:
    """Generate the IVD mask between vertebra ``idx`` and ``next_id`` using convex-hull morphology."""
    spine = spine.extract_label([49, 50, 26])
    spine_full = spine.extract_label([49, 50, 26, 41, 43, 44])
    spine_full = spine_full.calc_convex_hull("S")
    above = vert.extract_label(idx) * spine
    below = vert.extract_label(next_id.value) * spine
    a = above.erode_msk(2, verbose=False) + below
    hull = a.calc_convex_hull("A")
    hull = hull.erode_msk(dilate + 1, verbose=False).dilate_msk(dilate, verbose=False)
    hull[a != 0] = 0
    b = above + below.erode_msk(2, verbose=False)
    hull_b = b.calc_convex_hull("A")
    hull_b = hull_b.erode_msk(dilate + 1, verbose=False).dilate_msk(dilate, verbose=False)
    hull_b[b != 0] = 0
    hull = hull * hull_b
    hull = hull.filter_connected_components(1, max_count_component=1)
    hull = hull.dilate_msk(dilate, verbose=False) * (-spine_full + 1)
    s2 = hull.sum()
    for j in range(2, 0, -1):
        hull2 = hull.copy()
        while True:
            hull2 = hull2.erode_msk(j, verbose=False).filter_connected_components(1, max_count_component=1, connectivity=1)
            hull2 = hull2 * (-spine_full + 1)
            hull2 = hull2.dilate_msk(j, verbose=False)
            s = hull2.sum()
            if s == s2:
                if 1 in hull2.unique():
                    hull *= hull2
                break
            s2 = s
    hull = hull.filter_connected_components(1, max_count_component=1, connectivity=1)
    # hull = hull.dilate_msk(1, verbose=False) * (-spine_full + 1)
    # out_c = out[crop]
    # out_c[out_c == 0] = hull[out_c == 0] * (100 + idx)
    return hull * (100 + idx)


def _process_vertebra(
    idx: int,
    verts_ids: list[int],
    vert: NII,
    spine: NII,
    poi: POI,
    dilate: int = 1,
) -> tuple[NII, tuple[slice, slice, slice]] | None:
    """Compute the synthetic IVD mask for a single vertebra and return it with its crop slices."""
    # POI is not cropped, as we only need the vertebra direction
    result = _crop(idx, verts_ids, vert, spine, poi)
    if result is None:
        return None
    idx, verts_ids, vert, spine, poi, crop, next_id = result
    # TODO get_endplate is not working.
    ivd_1 = None  # _process_vertebra_A(idx, vert, spine, next_id.value, poi)
    ivd_2 = _process_vertebra_B(idx, vert, spine, next_id, dilate)  # * 0

    if ivd_1 is not None:
        ivd_2[ivd_2 == 0] = ivd_1[ivd_2 == 0]
    # try:
    #    set_above_3_point_plane(
    #        ivd_2,
    #        poi[
    #            idx,
    #            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right.value,
    #        ],
    #        poi[idx, Location.Vertebra_Corpus.value],
    #        poi[
    #            idx,
    #            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left.value,
    #        ],
    #        inplace=True,
    #    )
    #    set_above_3_point_plane(
    #        ivd_2,
    #        poi[
    #            idx,
    #            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right.value,
    #        ],
    #        poi[idx, Location.Vertebra_Corpus.value],
    #        poi[
    #            idx,
    #            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left.value,
    #        ],
    #        inplace=True,
    #    )
    # except Exception as e:
    #    print("Exception", e)
    # try:
    #    set_above_3_point_plane(
    #        ivd_2,
    #        poi[
    #            next_id,
    #            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right.value,
    #        ],
    #        poi[next_id, Location.Vertebra_Corpus.value],
    #        poi[
    #            next_id,
    #            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left.value,
    #        ],
    #        inplace=True,
    #        invert=-1,
    #    )
    #    set_above_3_point_plane(
    #        ivd_2,
    #        poi[
    #            next_id,
    #            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right.value,
    #        ],
    #        poi[next_id, Location.Vertebra_Corpus.value],
    #        poi[
    #            next_id,
    #            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left.value,
    #        ],
    #        inplace=True,
    #        invert=-1,
    #    )
    # except Exception as e:
    #    print("Exception", e)
    return ivd_2, crop


def compute_fake_ivd(vert_full: NII, subreg_full: NII, poi: POI, dilate: int = 1) -> NII:
    """Add synthetic intervertebral disc (IVD) labels to the vertebra segmentation.

    For every adjacent vertebra pair a synthetic IVD mask is computed using
    morphological operations on the subregion labels.  The disc for vertebra
    ``i`` is assigned label ``100 + i`` in the output.  The computation is
    parallelised over vertebrae using a :class:`~concurrent.futures.ProcessPoolExecutor`.

    Args:
        vert_full: Vertebra segmentation NIfTI; each vertebra has a unique
            integer label.  Modified in place and returned.
        subreg_full: Subregion segmentation NIfTI containing spine body labels
            (e.g. 49, 50).
        poi: POI object used to ensure that
            :attr:`~TPTBox.Location.Vertebra_Corpus` centroids are available.
        dilate: Morphological dilation radius applied when building the IVD
            hull mask.

    Returns:
        The updated ``vert_full`` NIfTI with synthetic IVD labels added.
    """
    subreg_ids_required_for_ivd_generation = [
        # Location.Inferior_Articular_Left,
        # Location.Inferior_Articular_Right,
        # Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Corpus,
        # Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
        # Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
        # Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,
        # Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,
        # Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,
        # Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,
        # Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,
        # Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,
    ]
    _sub = poi.keys_subregion()
    if any(f.value not in _sub for f in subreg_ids_required_for_ivd_generation):
        poi = calc_poi_from_subreg_vert(
            vert_full,
            subreg_full,
            extend_to=poi,
            subreg_id=subreg_ids_required_for_ivd_generation,
        )

    verts_ids: list[int] = vert_full.unique()
    crop = vert_full.compute_crop(dist=2)
    vert_full_cropped = vert_full.apply_crop(crop)
    subreg_full = subreg_full.apply_crop(crop)
    poi_cropped = poi.apply_crop(crop)

    out = vert_full_cropped.get_array()
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_process_vertebra, i, verts_ids, vert_full_cropped, subreg_full, poi_cropped, dilate): i
            for i in vert_full.unique()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="add artifical IVD to seg"):
            result = future.result()
            if result is not None:
                out_c, slices = result
                out_c = out_c.get_array()
                out[slices][out_c != 0] = out_c[out_c != 0]
    vert_full[crop] = out
    return vert_full


def calculate_IVD_POI(
    vert: NII,
    subreg: NII,
    poi: POI,
    ivd_location: set[Location] | None = None,
) -> POI:
    """Compute IVD-related Points of Interest and add them to ``poi``.

    If the subregion image does not yet contain IVD labels (label 100),
    :func:`compute_fake_ivd` is called first to synthesise them.  Centroid
    POIs at label 100 are then computed for all vertebrae and optionally
    extended with superior / inferior extreme disc points via
    :func:`strategy_calculate_up_vector`.

    Args:
        vert: Vertebra segmentation NIfTI.
        subreg: Subregion segmentation NIfTI; updated in place when IVD labels
            are synthesised.
        poi: POI object to extend.  Updated in place and returned.
        ivd_location: Set of :class:`~TPTBox.Location` values to compute.
            Defaults to
            ``{Location.Vertebra_Disc_Superior, Location.Vertebra_Disc}``.
            Pass an empty set to skip computation entirely.

    Returns:
        The updated :class:`~TPTBox.POI` object with IVD-related locations
        added.
    """
    if ivd_location is None:
        ivd_location = {Location.Vertebra_Disc_Superior, Location.Vertebra_Disc}
    # Note: currently we always need point 100, so it is computed if any Location is in ivd_location

    if len(ivd_location) == 0:
        return poi
    if 100 not in subreg.unique():
        if Location.Vertebra_Direction_Inferior.value not in poi.keys_subregion():
            poi = calc_poi_from_subreg_vert(
                vert,
                subreg,
                extend_to=poi,
                subreg_id=Location.Vertebra_Direction_Inferior.value,
            )
        vert = compute_fake_ivd(vert, subreg, poi=poi)
        subreg[vert >= 100] = 100

    calc_poi_from_subreg_vert(vert, subreg, subreg_id=100, extend_to=poi)
    if Location.Vertebra_Disc_Superior in ivd_location or Location.Vertebra_Disc_Inferior in ivd_location:
        current_vert = vert.copy()
        bb = current_vert.compute_crop()
        current_vert.apply_crop_(bb)
        for idx in vert.unique():
            if idx >= 100:
                break
            strategy_calculate_up_vector(poi, current_vert, idx, bb)
    return poi


if __name__ == "__main__":
    from pathlib import Path

    from TPTBox import Location, calc_poi_from_subreg_vert, to_nii
    from TPTBox.core.poi_fun.ray_casting import add_spline_to_img

    idx = 20
    root = Path("/DATA/NAS/datasets_processed/CT_fullbody/dataset-watrinet/source/dataset-myelom/sub-CTFU03866_ses-20180724_sequ-202_ct/")
    # ct, subreg, vert, idx = get_test_ct()
    subreg = to_nii(root / "spine.nii.gz", True)
    vert = to_nii(root / "vert.nii.gz", True)
    poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=[50])
    subreg2 = add_spline_to_img(subreg, poi, override_seg=False, dilate=9)
    # vert_ = vert.reorient()
    # subreg.reorient_()
    #
    # poi.fit_spline(location=spline_subreg_point_id, vertebra=True)
    subreg2 = compute_fake_ivd(vert, subreg2, poi)
    # assert subreg2 is not None
    # subreg2.reorient_(vert.orientation)
    subreg2.save(root / "endplate-test.nii.gz")
