from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from TPTBox import Print_Logger, Vertebra_Instance
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import POI
from TPTBox.core.poi_fun._help import paint_into_NII, to_local_np
from TPTBox.core.poi_fun.ray_casting import add_ray_to_img, max_distance_ray_cast_convex, unit_vector
from TPTBox.core.vert_constants import DIRECTIONS, Location

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


def strategy_calculate_up_vector(poi: POI, current_subreg: NII, vert_id: int, bb, log=_log):
    center = to_local_np(Location.Vertebra_Disc, bb, poi, vert_id, log)
    if center is None:
        return poi
    normal_vector = calculate_up_vector_np(current_subreg.rescale().extract_label(vert_id + 100).get_array())
    normal_vector = normal_vector / np.array(poi.zoom)
    extreme_point = center + normal_vector * 10

    axis = poi.get_axis("S")
    s_is_poss = int(poi.orientation[axis] == "S")
    below = int(extreme_point[axis] < center[axis])

    ## INFERIOR ##
    if s_is_poss + below == 1:  # xor
        normal_vector *= -1
    # max_distance_ray_cast_convex
    extreme_point = max_distance_ray_cast_convex(current_subreg.extract_label(vert_id + 100), center, normal_vector)
    extreme_point_sup = max_distance_ray_cast_convex(current_subreg.extract_label(vert_id + 100), center, -normal_vector)
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


def _process_vertebra(
    i: int, verts_ids: list[int], vert: NII, spine: NII, out: np.ndarray
) -> tuple[np.ndarray, tuple[slice, slice, slice]] | None:
    """Process a single vertebra index."""
    result = _crop(i, verts_ids, vert, spine, out)
    if result is None:
        return None
    i, verts_ids, vert, spine, out, crop, next_id = result
    above = vert.extract_label(i) * spine
    below = vert.extract_label(next_id.value) * spine
    a = above.erode_msk(2, verbose=False) + below
    hull = a.calc_convex_hull("A")
    hull = hull.erode_msk(1, verbose=False).dilate_msk(1, verbose=False)
    hull[a != 0] = 0
    b = above + below.erode_msk(2, verbose=False)
    hull_b = b.calc_convex_hull("A")
    hull_b = hull_b.erode_msk(1, verbose=False).dilate_msk(1, verbose=False)
    hull_b[b != 0] = 0
    hull = hull * hull_b
    hull = hull.filter_connected_components(1, max_count_component=1)
    hull = hull.dilate_msk(1, verbose=False) * (-spine + 1)
    out_c = out[crop]
    out_c[out_c == 0] = hull[out_c == 0] * (100 + i)
    return out_c, crop


def compute_fake_ivd(vert_full: NII, subreg_full: NII):
    verts_ids: list[int] = vert_full.unique()
    spine_full = subreg_full.extract_label([49, 50, 26])
    out = vert_full.get_array()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_process_vertebra, i, verts_ids, vert_full, spine_full, out): i for i in vert_full.unique()}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                out_c, slices = result
                out[slices][out_c != 0] = out_c[out_c != 0]

    return vert_full.set_array(out)


def compute_fake_ivd_old(vert_full: NII, subreg_full: NII):
    from tqdm import tqdm

    vert_full = vert_full.copy()
    verts_ids: list[int] = vert_full.unique()
    spine_full = subreg_full.extract_label([49, 50, 26])
    out = vert_full.get_array()
    print(verts_ids)
    for i in tqdm(vert_full.unique()):
        if i >= 100:
            continue
        if i + 100 in verts_ids:
            continue

        next_id = Vertebra_Instance(i).get_next_poi(verts_ids)
        if next_id is None:
            continue
        crop = vert_full.extract_label([i, next_id.value]).compute_crop(dist=3)
        vert = vert_full.apply_crop(crop)
        spine = spine_full.apply_crop(crop)
        above = vert.extract_label(i) * spine
        below = vert.extract_label(next_id.value) * spine
        a = above.erode_msk(2, verbose=False) + below
        hull = a.calc_convex_hull("A")  # - a
        hull = hull.erode_msk(1, verbose=False).dilate_msk(1, verbose=False)
        hull[a != 0] = 0
        b = above + below.erode_msk(2, verbose=False)
        hull_b = b.calc_convex_hull("A")  # - a
        hull_b = hull_b.erode_msk(1, verbose=False).dilate_msk(1, verbose=False)
        hull_b[b != 0] = 0
        hull = hull * hull_b
        hull = hull.filter_connected_components(1, max_count_component=1)
        hull = hull.dilate_msk(1, verbose=False) * (-spine + 1)
        out_c = out[crop]
        out_c[out_c == 0] = hull[out_c == 0] * (100 + i)
    return vert.set_array(out)


def calculate_IVD_POI(vert: NII, subreg: NII, poi: POI, ivd_location: set[Location] | None = None):
    if ivd_location is None:
        ivd_location = {Location.Vertebra_Disc_Superior, Location.Vertebra_Disc}
    # Note: currently we always need point 100, so it is computed if any Location is in ivd_location
    from TPTBox import calc_poi_from_subreg_vert

    if len(ivd_location) == 0:
        return poi
    if 100 not in subreg.unique():
        vert = compute_fake_ivd(vert, subreg)
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
