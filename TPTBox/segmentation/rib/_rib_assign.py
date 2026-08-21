"""Rib-to-vertebra assignment on already-loaded segmentations (internal module).

The functions here are implementation details of
:func:`TPTBox.segmentation.rib.add_ribs.add_ribs_to_vert_spine`, which is the
supported public entry point. They are not re-exported from the package.

* ``assign_ribs_to_vert_segmentation`` — deterministic cranial→caudal mapping
  of rib connected components to vertebrae T1..T12/L1, merging the result
  back into the passed ``vert``/``spine`` NIIs.
* ``split_touching_rib_ccs`` — erosion-based pre-processing pass that
  separates rib CCs which fuse the ribs of neighbouring vertebrae.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import partial

import numpy as np
from tqdm import tqdm

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


def _split_ccs_by_side(rib_cc: NII, cms_cc: POI, cms_cc2: POI, cms_vert: POI) -> tuple[list[RibCandidate], list[RibCandidate]]:
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
        cand = RibCandidate(cc_label=cc, volume=vols.get(cc, 0), z=center[inf_axis], x=center2[right_axis])
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


def _touching_surface(mask_a: np.ndarray, mask_b: np.ndarray, nii: NII, axis_weights=None) -> float:
    """Return weighted number of voxel faces shared by two masks.

    Touching along ``up_down_axis`` is weighted by ``up_down_weight``.
    All other axes have weight 1.0.
    """
    if axis_weights is None:
        axis_weights = {nii.get_axis("S"): 0.01, nii.get_axis("A"): 0.1, nii.get_axis("R"): 1}
    surface = 0.0

    for axis in range(mask_a.ndim):
        sl1 = [slice(None)] * mask_a.ndim
        sl2 = [slice(None)] * mask_a.ndim

        sl1[axis] = slice(None, -1)
        sl2[axis] = slice(1, None)

        touching = (
            np.logical_and(mask_a[tuple(sl1)], mask_b[tuple(sl2)]).sum() + np.logical_and(mask_b[tuple(sl1)], mask_a[tuple(sl2)]).sum()
        )

        weight = axis_weights.get(axis, 1)
        surface += weight * touching

    return surface


def _try_erosion_split(
    cc_label: int, binary_cc: NII, erosion_pixels: int, min_volume: int, _pass=0, verbose=True
) -> tuple[int, list[np.ndarray]] | None:
    """Try to split one CC by erosion.

    Returns ``(cc_label, sub_masks)`` when successful, otherwise ``None``.
    ``sub_masks`` are returned in the original image shape.
    """
    if erosion_pixels <= 0:
        return None

    # Crop to the CC's bounding box (+ padding for erosion/infection).
    cc_nii = binary_cc

    crop = cc_nii.compute_crop(0, 2)
    cc_nii = cc_nii.apply_crop(crop)
    try:
        if _pass == 0:
            eroded = cc_nii.erode_msk_euclid(n_pixel=erosion_pixels, verbose=False)
        elif _pass == 1:
            eroded = cc_nii.erode_msk(n_pixel=erosion_pixels, verbose=False)
        elif _pass == 2:
            eroded = cc_nii.erode_msk(n_pixel=erosion_pixels, verbose=False, ignore_direction="R")
        elif _pass == 3:
            eroded = cc_nii.erode_msk(n_pixel=erosion_pixels, verbose=False, ignore_direction="A")
        else:
            eroded = cc_nii.erode_msk(n_pixel=erosion_pixels, verbose=False)
    except Exception as e:
        if verbose:
            logger.on_fail(f"Erosion failed for CC {cc_label}: {e}")
        return None
    eroded_cc = eroded.get_connected_components(connectivity=3)
    sub_labels = [int(x) for x in eroded_cc.unique() if x != 0]

    if len(sub_labels) < 2:
        return None
    infected = eroded_cc.infect(cc_nii, verbose=False)
    infected_arr = infected.get_seg_array()
    full_size = cc_nii.sum()
    # remerge if to small
    for sub in sub_labels.copy():
        count = int((infected_arr == sub).sum())

        should_merge = count * cc_nii.voxel_volume() < min_volume or count / full_size < 0.2
        if not should_merge:
            continue
        # print("merge", count, full_size, count / full_size)
        remaining = [x for x in sub_labels if x != sub]
        if not remaining:
            break

        sub_mask = infected_arr == sub
        if len(remaining) == 1:
            target = remaining[0]
            # print("merge", target)
        else:
            target = max(remaining, key=lambda candidate: _touching_surface(sub_mask, infected_arr == candidate, infected))
            # print("merge of many", target)
        infected_arr[infected_arr == sub] = target
        sub_labels.remove(sub)
    if len(sub_labels) < 2:
        return None
    # Restore masks to the original shape here, so callers do not need to
    # track either the crop or the original CC label separately.
    full_masks = []
    for sub in sub_labels:
        m = infected_arr == sub
        full = np.zeros(binary_cc.shape, dtype=bool)
        full[crop] = m
        full_masks.append(full)
    if len(full_masks) < 2:
        return None
    return cc_label, full_masks


def split_touching_rib_ccs(
    rib_cc: NII,
    erosion_pixels: int = 4,
    min_volume: int = 1000,
    max_passes: int = 2,
    vert_ids=None,
    verbose: bool = False,
    num_workers: int = 1,
    short_cut=True,
) -> NII:
    """Split rib connected components that likely fuse the ribs of adjacent vertebrae.

    ``_try_erosion_split`` can optionally be evaluated in parallel for all
    connected components in a pass. Label assignment and writes to ``rib_cc``
    remain sequential to keep labels deterministic and avoid races.

    Args:
        rib_cc: Connected-component-labelled rib mask, modified in place.
        erosion_pixels: Voxels to erode per pass when attempting to split a CC.
        min_volume: Minimum voxel count for a split sub-component to survive
            the merge-back heuristic (values below are re-merged into the
            largest touching neighbour).
        max_passes: Number of erosion passes tried before giving up. Each pass
            uses a slightly different erosion strategy (Euclidean, standard,
            direction-restricted).
        vert_ids: Optional list of vertebra labels present in the case, used to
            estimate the expected number of rib CCs (thoracic vertebrae × 2).
        verbose: Emit per-split log messages.
        num_workers: Threads used to evaluate ``_try_erosion_split`` in
            parallel. ``1`` runs sequentially.
        short_cut: If True (default), stop as soon as ``rib_cc.max()`` reaches
            the expected CC count. Set to False to always run every pass.

    Returns:
        The (in-place-mutated) ``rib_cc`` NII.
    """
    # arr = rib_cc.get_seg_array()

    if vert_ids is None:
        vert_ids = []
    next_label = int(rib_cc.max()) + 1 if rib_cc.shape and rib_cc.max() > 0 else 1
    u = {int(x) for x in rib_cc.unique() if x != 0}
    # ribs counte twice (left/right)
    expected_number_of_ccs = len([a for a in Vertebra_Instance.thoracic() if a.value in vert_ids]) * 2
    if expected_number_of_ccs == 0:
        expected_number_of_ccs = 24

    for _pass in range(max_passes):
        print("Separate RIBs - Pass", _pass + 1, f"{expected_number_of_ccs=}")

        labels = sorted(u)

        binary_ccs = [(cc_label, rib_cc.extract_label(cc_label)) for cc_label in labels]
        if num_workers == 1:
            results = []
            for cc_label, binary_cc in tqdm(binary_ccs, total=len(binary_ccs), desc="Separate RIBs"):
                result = _try_erosion_split(cc_label, binary_cc, erosion_pixels, min_volume, _pass)
                if result is not None:
                    results.append(result)
        else:
            results = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_try_erosion_split, cc_label, binary_cc, erosion_pixels, min_volume, _pass)
                    for cc_label, binary_cc in binary_ccs
                ]

                # tqdm advances immediately whenever an individual future completes.
                for future in tqdm(as_completed(futures), total=len(futures), desc="Separate RIBs"):
                    result = future.result()
                    if result is not None:
                        results.append(result)

        # Apply mutations sequentially so label assignment is deterministic.
        for cc_label, sub_masks in results:
            if sub_masks is None:
                continue

            binary_cc = rib_cc.extract_label(cc_label)
            u.discard(cc_label)
            rib_cc[binary_cc] = 0

            for i, m in enumerate(sub_masks):
                new_lbl = cc_label if i == 0 else next_label
                if i > 0:
                    next_label += 1

                rib_cc[m] = new_lbl
                u.add(new_lbl)

                if verbose:
                    logger.print(f"split CC {cc_label} via erosion -> {new_lbl} (voxels={int(m.sum())})")
        if short_cut and rib_cc.max() >= expected_number_of_ccs:
            break

    return rib_cc


def assign_ribs_to_vert_segmentation(
    vert_seg: NII,
    sem_seg: NII,
    rib_seg: NII,
    verbose: bool = False,
    min_volume: int = 100,
    no_7=False,
    split_touching: bool = True,
    erosion_pixels: int = 2,
    left_id: int = Full_Body_Instance.rib_left.value,
    right_id: int = Full_Body_Instance.rib_right.value,
    error_value=255,
    add_error=True,
    short_cut=True,
) -> tuple[NII, NII]:
    """Assign rib connected components deterministically from top to bottom.

    Design:

    - deterministic top→bottom assignment
    - no dominance / repeated while-loop logic
    - robust left/right split using vertebra + rib centroids
    - merges the results back into the passed ``vert_seg`` / ``sem_seg`` so
      non-rib labels are preserved

    Args:
        vert_seg: Vertebra instance segmentation. Modified in place: matched
            rib CCs are labelled with ``Vertebra_Instance(vid).RIB`` and
            unmatched CCs get ``error_value`` (when ``add_error=True``).
        sem_seg: Spine subregion segmentation. Modified in place: rib voxels
            get ``Location.Rib_Left`` / ``Location.Rib_Right``.
        rib_seg: Raw rib segmentation containing ``left_id`` / ``right_id``.
        verbose: Verbose logging.
        min_volume: Minimum voxel volume for a connected component to be kept
            (both for the initial CC filter and inside the touching-split).
        no_7: If True, skip vertebra 7 (C7) even if it appears rib-bearing.
        split_touching: If True, run :func:`split_touching_rib_ccs` on the rib
            CC map before assignment so ribs that touch front/middle/back get
            separated via erosion.
        erosion_pixels: Voxels to erode by when attempting the erosion split.
        left_id: Label value in ``rib_seg`` that marks the left rib mask.
        right_id: Label value in ``rib_seg`` that marks the right rib mask.
        error_value: Sentinel label written into ``vert_seg`` for rib CCs that
            could not be matched to a vertebra (default 255).
        add_error: If True, propagate ``error_value`` into ``vert_seg`` and
            grow neighbouring labels into the unmatched region; if False, drop
            the unmatched CCs silently.
        short_cut: Forwarded to :func:`split_touching_rib_ccs` — stop the
            erosion loop early once the expected number of CCs is reached.

    Returns:
        ``(vert_seg, sem_seg)`` — the mutated inputs, re-oriented back to the
        original orientation of ``vert_seg``.
    """
    rib_seg.assert_affine(other=vert_seg, verbose=verbose)
    rib_seg.assert_affine(other=sem_seg, verbose=verbose)

    ori = vert_seg.orientation
    vert_ids = vert_seg.unique()

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
            rib_cc, erosion_pixels=erosion_pixels, min_volume=min_volume, vert_ids=vert_ids, short_cut=short_cut, verbose=verbose
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

    sem_seg[rib_seg != 0] = rib_seg.map_labels({left_id: Location.Rib_Left.value, right_id: Location.Rib_Right.value})[rib_seg != 0]  # type: ignore
    undefined = sum(v == error_value for v in rib_vert_map.values())
    logger.print(f"Unmatched rib CCs: {undefined}")
    if add_error:  # or split_touching:
        vert_seg[rib_inst == error_value] = error_value

        if np.any(vert_seg.get_seg_array() == error_value):
            vert_seg2 = vert_seg.remove_labels(error_value).infect(vert_seg.extract_label(error_value), verbose=False)
            vert_seg[vert_seg != vert_seg2] = vert_seg2[vert_seg != vert_seg2]
            vert_seg[np.logical_and(rib_inst == error_value, vert_seg == 0)] = error_value
        vert_seg[np.logical_and(vert_seg == 0, sem_seg.extract_label([Location.Rib_Left.value, Location.Rib_Right.value] == 1))] = (
            error_value
        )
    return vert_seg.reorient_(ori), sem_seg.reorient_(ori)
