from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from TPTBox import NII, POI, Location, No_Logger, Vertebra_Instance, calc_centroids
from TPTBox.core.vert_constants import Full_Body_Instance

logger = No_Logger(prefix="RibAssignment")


@dataclass(frozen=True)
class RibCandidate:
    cc_label: int
    volume: float
    z: float
    x: float


def _thoracic_like_labels(labels: Iterable[int]) -> list[int]:
    """Keep only vertebrae that can carry ribs and sort cranial -> caudal.

    The original implementation used a fragile hard-coded range check.
    Here we simply keep the contiguous block that was previously allowed
    (7..20 inclusive) and sort ascending = top to bottom.
    """
    return [*sorted(v for v in labels if 7 <= int(v) <= 20), 28]


def _split_ccs_by_side(
    rib_cc: NII,
    cms_cc: POI,
    cms_cc2: POI,
    cms_vert: POI,  # , ref_vert: int
) -> tuple[list[RibCandidate], list[RibCandidate]]:
    right_axis = cms_vert.get_axis("R")
    inf_axis = cms_vert.get_axis("I")
    # ref_x = cms_vert[ref_vert, 50][right_axis]

    left: list[RibCandidate] = []
    right: list[RibCandidate] = []
    vols = rib_cc.volumes()

    for cc in cms_cc2.keys_region():
        if cc == 0 or (cc, 50) not in cms_cc:
            continue
        center = cms_cc[cc, 50]

        center2 = cms_cc2[cc, 50] if (cc, 50) in cms_cc2 else center  # noqa: SIM401
        cand = RibCandidate(
            cc_label=cc,
            volume=vols.get(cc, 0),
            z=center[inf_axis],
            x=center2[right_axis],
        )
        distances = cms_vert.calculate_distances_cord(center2)
        min_key = min(distances, key=distances.get)  # type: ignore
        # x < vertebra center => patient right in RAS-like orientation
        if cand.x < cms_vert[min_key][right_axis]:
            right.append(cand)
        else:
            left.append(cand)

    # sort top -> bottom (smallest inferior coordinate first)
    left.sort(key=lambda c: c.z)
    right.sort(key=lambda c: c.z)
    return left, right


def _vert_z_spacing(cms_vert: POI) -> tuple[int, float, list[tuple[int, float]]]:
    """Return (inferior axis index, median vertebra Z-spacing in voxels, sorted [(vert_id, z)])."""
    inf_axis = cms_vert.get_axis("I")
    vz = [(int(v), float(cms_vert[v, 50][inf_axis])) for v in cms_vert.keys_region() if (v, 50) in cms_vert]
    vz.sort(key=lambda t: t[1])
    if len(vz) < 2:
        return inf_axis, 0.0, vz
    spacings = [abs(vz[i + 1][1] - vz[i][1]) for i in range(len(vz) - 1)]
    return inf_axis, float(np.median(spacings)) if spacings else 0.0, vz


def _cc_extent_on_axis(arr: np.ndarray, cc_label: int, axis: int) -> float:
    """Return the extent of a CC on a single axis."""
    coords = np.where(arr == cc_label)[axis]
    if coords.size == 0:
        return 0.0
    return float(coords.max() - coords.min())


def _try_erosion_split(
    cc_mask_template: NII,
    binary_cc: np.ndarray,
    erosion_pixels: int,
    min_volume: int,
) -> list[np.ndarray] | None:
    """Erode a single binary CC and re-run CC labelling. Return list of binary sub-CCs (infected back to the original mask) if the erode+CC produced 2+ components, else None."""
    if erosion_pixels <= 0:
        return None
    cc_nii = cc_mask_template.copy().set_array_(binary_cc.astype(np.uint8), verbose=False)
    try:
        eroded = cc_nii.erode_msk(n_pixel=erosion_pixels, verbose=False)
    except Exception:
        return None
    eroded_cc = eroded.get_connected_components(connectivity=3)
    sub_labels = [int(x) for x in eroded_cc.unique() if x != 0]
    if len(sub_labels) < 2:
        return None
    infected = eroded_cc.infect(cc_nii, verbose=False)
    infected_arr = infected.get_seg_array()
    sub_masks = []
    for sub in sub_labels:
        m = infected_arr == sub
        if int(m.sum()) >= min_volume:
            sub_masks.append(m)
    return sub_masks if len(sub_masks) >= 2 else None


def split_touching_rib_ccs(
    rib_cc: NII,
    cms_vert: POI,
    max_span_factor: float = 1.4,
    erosion_pixels: int = 2,
    min_volume: int = 100,
    max_passes: int = 3,
    verbose: bool = False,
) -> NII:
    """Split rib connected components that likely fuse the ribs of adjacent vertebrae.

    A CC whose extent on the inferior axis exceeds ``max_span_factor`` × the
    median vertebra Z-spacing is treated as merged ribs and processed by
    erosion: erode the CC, re-run CC labelling, infect the new labels back
    onto the original CC voxels. Resolves ribs bridged by a thin strip of
    voxels.

    Runs up to ``max_passes`` sweeps so a newly split sub-CC that is still
    oversized gets another chance.
    """
    inf_axis, median_spacing, _vz_sorted = _vert_z_spacing(cms_vert)
    if median_spacing <= 0:
        return rib_cc
    threshold = median_spacing * max_span_factor

    arr = rib_cc.get_seg_array()
    next_label = int(arr.max()) + 1 if arr.size and arr.max() > 0 else 1

    for _pass in range(max_passes):
        changed = False
        for cc_label in sorted({int(x) for x in np.unique(arr) if x != 0}):
            extent = _cc_extent_on_axis(arr, cc_label, inf_axis)
            if extent <= threshold:
                continue

            binary_cc = arr == cc_label
            sub_masks = _try_erosion_split(rib_cc, binary_cc, erosion_pixels, min_volume)
            if sub_masks is None:
                continue

            arr[binary_cc] = 0
            for i, m in enumerate(sub_masks):
                new_lbl = cc_label if i == 0 else next_label
                if i > 0:
                    next_label += 1
                arr[m] = new_lbl
                if verbose:
                    logger.print(f"split CC {cc_label} via erosion -> {new_lbl} (voxels={int(m.sum())})")
            changed = True
        if not changed:
            break

    rib_cc.set_array_(arr, verbose=False)
    return rib_cc


def assign_ribs_to_vert_segmentation(
    vert_seg: NII,
    sem_seg: NII,
    rib_seg: NII,
    verbose: bool = False,
    min_volume: int = 100,
    no_7=False,
    split_touching: bool = True,
    max_span_factor: float = 1.4,
    erosion_pixels: int = 2,
    left_id: int = Full_Body_Instance.rib_left.value,
    right_id: int = Full_Body_Instance.rib_right.value,
    error_value=255,
    add_error=True,
) -> tuple[NII, NII]:
    """Assign rib connected components deterministically from top to bottom.

    Refactor goals:
    - deterministic top->bottom assignment
    - no dominance / repeated while-loop logic
    - robust left/right split using centroids
    - preserve original output contract

    Args:
        split_touching: If True, run ``split_touching_rib_ccs`` on the rib CC
            map before assignment so ribs that touch front/middle/back get
            separated (erosion first, Z-band fallback).
        max_span_factor: Threshold (× median vertebra spacing) above which a rib
            CC is treated as merged and eligible for splitting.
        erosion_pixels: Voxels to erode by when attempting the erosion split.
    """
    rib_seg.assert_affine(other=vert_seg, verbose=verbose)
    rib_seg.assert_affine(other=sem_seg, verbose=verbose)

    ori = vert_seg.orientation
    vert_seg = vert_seg.reorient()
    sem_seg = sem_seg.reorient()
    rib_seg = rib_seg.reorient(verbose=verbose)
    vert_pred = vert_seg.extract_label([Vertebra_Instance.C7, *Vertebra_Instance.thoracic(), Vertebra_Instance.L1])
    rib_seg = rib_seg.extract_label([left_id, right_id], keep_label=True)  # type: ignore
    logger.on_debug(f"{rib_seg.unique()=}")
    # Remove rib voxels overlapping vertebrae
    rib_seg[vert_pred != 0] = 0

    rib_cc = rib_seg.filter_connected_components(None, min_volume=min_volume, keep_label=False)
    cms_vert = calc_centroids(vert_seg)
    if split_touching:
        rib_cc = split_touching_rib_ccs(
            rib_cc, cms_vert, max_span_factor=max_span_factor, erosion_pixels=erosion_pixels, min_volume=min_volume, verbose=verbose
        )
    cms_cc = calc_centroids(rib_cc * vert_pred.calc_convex_hull(None).dilate_msk_euclid(5))
    cms_cc2 = calc_centroids(rib_cc)  # .dilate_msk_euclid(5)

    vert_labels = _thoracic_like_labels(vert_seg.unique())
    if not vert_labels:
        logger.on_warning("No rib-bearing vertebrae found")
        return vert_seg.reorient_(ori), sem_seg.reorient_(ori)

    # Use the top-most available vertebra as side reference
    left_ccs, right_ccs = _split_ccs_by_side(rib_cc, cms_cc, cms_cc2, cms_vert)

    rib_vert_map = {cc: error_value for cc in rib_cc.unique() if cc != 0}
    rib_subreg_map = {cc: 0 for cc in rib_cc.unique() if cc != 0}

    # Deterministic cranial -> caudal assignment
    # Smarter T1 heuristic: skip the first vertebra only if there is no
    # plausible rib CC close to it on either side. This avoids the common
    # off-by-one while preserving rare valid T1 rib assignments or partial FOV
    if no_7 and 7 in vert_labels:
        vert_labels.remove(7)
    assign_vert_labels = vert_labels
    assert isinstance(vert_labels, (tuple, list)), type(vert_labels)

    if vert_labels and (
        (vert_labels[0] != 28 and vert_labels[1] != 28) or ((vert_labels[0], 50) in vert_labels and (vert_labels[1], 50) in vert_labels)
    ):
        inf_axis = cms_vert.get_axis("I")
        t1_z = cms_vert[vert_labels[0], 50][inf_axis]
        # use median vertebral spacing as adaptive threshold
        spacing = abs(cms_vert[vert_labels[1], 50][inf_axis] - t1_z) if len(vert_labels) > 1 else 30
        threshold = max(5, spacing * 0.6)

        nearest_left = abs(left_ccs[0].z - t1_z) if left_ccs else float("inf")
        nearest_right = abs(right_ccs[0].z - t1_z) if right_ccs else float("inf")
        has_t1_rib = min(nearest_left, nearest_right) < threshold

        if not has_t1_rib:
            assign_vert_labels = vert_labels[1:]

    for i, raw_vid in enumerate(assign_vert_labels):
        vid = 20 if raw_vid == 28 else raw_vid
        if i < len(left_ccs):
            cc = left_ccs[i].cc_label
            rib_vert_map[cc] = Vertebra_Instance(vid).RIB
            rib_subreg_map[cc] = Location.Rib_Right.value

        if i < len(right_ccs):
            cc = right_ccs[i].cc_label
            rib_vert_map[cc] = Vertebra_Instance(vid).RIB
            rib_subreg_map[cc] = Location.Rib_Left.value

    # rib_sem = rib_cc.map_labels(rib_subreg_map, verbose=False)
    rib_inst = rib_cc.map_labels(rib_vert_map, verbose=False)
    logger.on_debug(f"{rib_inst.unique()=}")
    # Merge rib assignments back into the original vert / sem segmentations
    # without disturbing existing (non-rib) labels. Skip unmatched CCs (sentinel error_value=255).
    matched = (rib_inst != 0) & (rib_inst != error_value)
    vert_seg[matched] = rib_inst[matched]
    if add_error:
        vert_seg[rib_inst == 255] = 255
    sem_seg[rib_seg != 0] = rib_seg.map_labels({left_id: Location.Rib_Left.value, right_id: Location.Rib_Right.value})[rib_seg != 0]  # type: ignore
    undefined = sum(v == error_value for v in rib_vert_map.values())
    logger.print(f"Unmatched rib CCs: {undefined}")

    return vert_seg.reorient_(ori), sem_seg.reorient_(ori)
