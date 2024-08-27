from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from TPTBox import Print_Logger, Vertebra_Instance, calc_poi_from_subreg_vert
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun._help import paint_into_NII, to_local_np
from TPTBox.core.poi_fun.ray_casting import max_distance_ray_cast_convex
from TPTBox.core.vert_constants import Location

_log = Print_Logger()


def calculate_up_vector_np(segmentation: np.ndarray, verbose=False):
    # Get indices of segmented region (assuming segmentation is a binary mask)
    points = np.argwhere(segmentation > 0)
    # Perform PCA to find the principal axes
    pca = PCA(n_components=3)
    pca.fit(points)
    # First, second, and third principal components
    up_vector = pca.components_[2]  # Normal to the disc plane
    if verbose:
        print(f"Main Axis (PC1): {pca.components_[0]}")
        print(f"Secondary Axis (PC2): {pca.components_[1]}")
        print(f"Up Vector (PC3): {up_vector}")
    return up_vector


def strategy_calculate_up_vector(poi: POI, current_vert: NII, vert_id: int, bb, log=_log):
    center = to_local_np(Location.Vertebra_Disc, bb, poi, vert_id, log)
    if center is None:
        return poi
    normal_vector = calculate_up_vector_np(current_vert.rescale().extract_label(vert_id + 100).get_array())
    normal_vector = normal_vector / np.array(poi.zoom)
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
    poi[vert_id, Location.Vertebra_Disc_Inferior] = tuple(a.start + b for a, b in zip(bb, extreme_point, strict=True))
    poi[vert_id, Location.Vertebra_Disc_Superior] = tuple(a.start + b for a, b in zip(bb, extreme_point_sup, strict=True))

    return poi


def _crop(i: int, verts_ids: list[int], vert_full: NII, spine_full: NII, out: np.ndarray):
    if i >= 100 or (i + 100 in verts_ids):
        return None
    next_id = Vertebra_Instance(i).get_next_poi(verts_ids)
    if next_id is None:
        return None
    crop = vert_full.extract_label([i, next_id.value]).compute_crop(dist=3)
    vert = vert_full.apply_crop(crop)
    spine = spine_full.apply_crop(crop)
    return i, verts_ids, vert, spine, out, crop, next_id


def _process_vertebra_A(idx, vert: NII, spine: NII, next_id, poi) -> NII | None:
    from TPTBox.spine.statistics import endplate_extraction

    try:
        a = endplate_extraction(idx, vert, spine, poi)
        b = endplate_extraction(next_id.value, vert, spine, poi)
        if a is None or b is None:
            return None
        a = a.extract_label(Location.Vertebral_Body_Endplate_Inferior)
        b = b.extract_label(Location.Vertebral_Body_Endplate_Superior)
    except ValueError:
        return None
    ivd: NII = (a + b).calc_convex_hull(None)
    ivd[spine != 0] = 0
    ivd = ivd.filter_connected_components(1, max_count_component=1, connectivity=1)
    ivd = ivd * (100 + idx)
    return ivd


def _process_vertebra_B(idx: int, vert: NII, spine: NII, next_id):
    spine = spine.extract_label([49, 50, 26])
    spine_full = spine.extract_label([49, 50, 26, 41, 43, 44])
    spine_full = spine_full.calc_convex_hull("S")
    above = vert.extract_label(idx) * spine
    below = vert.extract_label(next_id.value) * spine
    a = above.erode_msk(2, verbose=False) + below
    hull = a.calc_convex_hull("A")
    hull = hull.erode_msk(2, verbose=False).dilate_msk(1, verbose=False)
    hull[a != 0] = 0
    b = above + below.erode_msk(2, verbose=False)
    hull_b = b.calc_convex_hull("A")
    hull_b = hull_b.erode_msk(2, verbose=False).dilate_msk(1, verbose=False)
    hull_b[b != 0] = 0
    hull = hull * hull_b
    hull = hull.filter_connected_components(1, max_count_component=1)
    hull = hull.dilate_msk(1, verbose=False) * (-spine_full + 1)
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
    idx: int, verts_ids: list[int], vert: NII, spine: NII, poi: POI, out: np.ndarray
) -> tuple[NII, tuple[slice, slice, slice]] | None:
    """Process a single vertebra index."""
    # POI is not cropped, as we only need the vertebra direction
    result = _crop(idx, verts_ids, vert, spine, out)
    if result is None:
        return None
    idx, verts_ids, vert, spine, out, crop, next_id = result
    ivd_1 = _process_vertebra_A(idx, vert, spine, next_id, poi)
    ivd_2 = _process_vertebra_B(idx, vert, spine, next_id)
    if ivd_1 is not None:
        ivd_2[ivd_2 == 0] = ivd_1[ivd_2 == 0]
    return ivd_2, crop


def compute_fake_ivd(vert_full: NII, subreg_full: NII, poi: POI):
    verts_ids: list[int] = vert_full.unique()
    crop = vert_full.compute_crop(dist=2)
    vert_full_cropped = vert_full.apply_crop(crop)
    subreg_full = subreg_full.apply_crop(crop)

    out = vert_full_cropped.get_array()

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _process_vertebra,
                i,
                verts_ids,
                vert_full_cropped,
                subreg_full,
                poi,
                out,
            ): i
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


def calculate_IVD_POI(vert: NII, subreg: NII, poi: POI, ivd_location: set[Location] | None = None):
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

    from TPTBox import to_nii

    path = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-shockroom-without-fx/derivatives_spine_r/sub-ctsr01552/ses-20190525/")

    vert = to_nii(path / "sub-ctsr01552_ses-20190525_sequ-19_seg-vert_msk.nii.gz", True)
    subreg = to_nii(path / "sub-ctsr01552_ses-20190525_sequ-19_seg-subreg_msk.nii.gz", True)
    poi = None  # POI.load(path / "sub-ctsr00850_ses-20170927_sequ-6_seg-subreg_ctd.json")
    # TODO CROP-late
    poi = calculate_IVD_POI(vert, subreg, poi)
    print(poi)

    paint_into_NII(poi, vert, rays=None).save(path / "test.nii.gz")
