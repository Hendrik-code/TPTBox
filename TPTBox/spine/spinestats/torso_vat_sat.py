from __future__ import annotations

from typing import Literal

import numpy as np

from TPTBox import NII
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.vert_constants import Full_Body_Instance, Full_Body_Instance_Vibe, Location, Vertebra_Instance


def peak_centered_mean(
    values: np.ndarray,
    bins: int = 64,
    peak_frac_height: float = 0.5,
) -> float:
    """Robust mean of a 1D intensity sample, centered on the histogram peak.

    Steps:
        1. Build a histogram of ``values``.
        2. Locate the mode (tallest bin).
        3. Grow a contiguous window outward from the mode as long as each
           neighbouring bin still has at least
           ``peak_frac_height`` * peak_height counts.
        4. Average only the values falling inside that window.

    Useful for estimating cerebrospinal fluid signal from a spinal canal
    mask that may include darker structures such as nerve roots or vessel
    walls at the border. Those low-signal tails sit outside the peak
    window and are excluded before averaging.

    Parameters
    ----------
    values : np.ndarray
        1D array of intensity samples already extracted from the mask.
    bins : int, default=64
        Histogram resolution used to locate the peak. Higher values give a
        more precise peak but are more sensitive to shot noise.
    peak_frac_height : float, default=0.5
        Fractional height cutoff. Bins whose count is at least this
        fraction of the peak count are kept; the default ``0.5`` matches
        the full-width half-maximum criterion. Lower values keep more of
        the tails; ``1.0`` reduces to the peak bin only.

    Returns:
    -------
    float
        Mean of the values inside the peak window. Returns ``nan`` for an
        empty input; falls back to the plain mean when the window is
        degenerate.
    """
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    hist, edges = np.histogram(values, bins=bins)
    if hist.max() == 0:
        return float(np.mean(values))
    peak_idx = int(np.argmax(hist))
    threshold = hist[peak_idx] * peak_frac_height
    left = peak_idx
    while left > 0 and hist[left - 1] >= threshold:
        left -= 1
    right = peak_idx
    while right < len(hist) - 1 and hist[right + 1] >= threshold:
        right += 1
    lo = edges[left]
    hi = edges[right + 1]
    kept = values[(values >= lo) & (values <= hi)]
    if kept.size == 0:
        return float(np.mean(values))
    return float(np.mean(kept))


def VBQ_score(
    t2w: NII,
    vert: NII,
    spine: NII,
    regions=None,
    subregs_ids=None,
    spinal_channel_id=Location.Spinal_Canal,
    n_erode=2,
    full_cord=False,
    spinal_bins: int = 64,
    spinal_peak_frac_height: float = 0.5,
) -> dict[str, int]:
    """Compute vertebral bone quality (VBQ) scores from a T2-weighted MRI.

    The VBQ score is defined as the ratio of the mean T2 signal intensity
    within the vertebral body to the mean T2 signal intensity of the
    cerebrospinal fluid (CSF) in the spinal canal over the same
    superior–inferior extent.

    For each requested spinal region, the function:
        1. Extracts the vertebral bodies of the specified vertebrae.
        2. Optionally erodes the vertebral body mask to reduce partial
           volume effects.
        3. Computes the mean T2 signal within the vertebral bodies.
        4. Computes the mean CSF signal within the spinal canal over the
           corresponding superior–inferior range.
        5. Returns the vertebral signal, CSF signal, and their ratio
           (VBQ score).

    Parameters
    ----------
    t2w : NII
        T2-weighted MRI volume.
    vert : NII
        Vertebral instance segmentation.
    spine : NII
        Spine subregion segmentation containing vertebral subregions and
        the spinal canal.
    regions : list[tuple[Vertebra_Instance, Vertebra_Instance]], optional
        Vertebral ranges over which to compute VBQ scores. Each tuple
        specifies the first and last vertebra (inclusive). Defaults to
        C3–C6, T5–T8, and L1-L4. According to https://link.springer.com/article/10.1007/s00586-022-07484-5
    subregs_ids : list[Location], optional
        Spine subregion labels defining the vertebral body. Defaults to
        ``[Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]``.
    spinal_channel_id : Location, default=Location.Spinal_Canal
        Label identifying the spinal canal (CSF) region.
    n_erode : int, default=2
        Number of erosion iterations applied to the vertebral body mask
        before signal extraction.
    spinal_bins : int, default=64
        Histogram bin count forwarded to :func:`peak_centered_mean` when
        estimating the CSF signal.
    spinal_peak_frac_height : float, default=0.5
        Fractional height cutoff forwarded to :func:`peak_centered_mean`.
        Bins with count >= ``peak_frac_height * peak`` are kept; the
        default 0.5 corresponds to FWHM.

    Returns:
    -------
    dict[str, float]
        Dictionary containing, for each region:

        - ``mean_signal_vertebra_<start>-<end>`` : mean T2 signal inside
          the (eroded) vertebral body mask.
        - ``mean_signal_liquor_<start>-<end>`` : peak-centered mean T2
          signal inside the spinal canal (robust to nerve-root voxels).
        - ``mean_signal_liquor_<start>-<end>_old`` : plain mean T2 signal
          inside the spinal canal, kept for backward comparison.
        - ``VBQ_<start>-<end>`` : ratio using the peak-centered CSF
          signal.
        - ``VBQ_<start>-<end>_old`` : ratio using the plain-mean CSF
          signal.

    Notes:
    -----
    The implementation follows the general principle of the Vertebral Bone
    Quality (VBQ) score described in the literature, using the ratio of
    vertebral body T2 signal intensity to CSF signal intensity as a proxy
    for bone quality.
    """
    if subregs_ids is None:
        subregs_ids = [
            Location.Vertebra_Corpus,
            Location.Vertebra_Corpus_border,
        ]

    if regions is None:
        regions = [
            (Vertebra_Instance.C3, Vertebra_Instance.C6),
            (Vertebra_Instance.T5, Vertebra_Instance.T8),
            (Vertebra_Instance.L1, Vertebra_Instance.L4),
        ]

    spinal_channel = spine.extract_label(spinal_channel_id).erode_msk(1, connectivity=1, verbose=False)
    corpus = spine.extract_label(subregs_ids)
    out = {}

    verts_order = Vertebra_Instance.order()

    for start, goal in regions:
        end = verts_order.index(goal) + 1
        labels = verts_order[verts_order.index(start) : end]

        # vertebrae belonging to this region
        vert_mask = vert.extract_label(labels)

        # vertebral bodies only
        bodies = vert_mask * corpus
        bodies.erode_msk_(n_erode, verbose=False)

        signal_vertebra = t2w.mean(where=bodies)

        # ---- restrict spinal canal to same S/I extent ----
        bbox = bodies.compute_crop()  # (slice_x, slice_y, slice_z)
        axis = spinal_channel.get_axis(direction="S")

        if not full_cord:
            slicer = [slice(None)] * 3
            slicer[axis] = bbox[axis]
            slicer = tuple(slicer)
            spinal_crop = spinal_channel[slicer]
            t2w_slab = t2w.get_array()[slicer]
        else:
            t2w_slab = t2w
            spinal_crop = spinal_channel

        signal_sfs_old = t2w_slab.mean(where=spinal_crop)
        spinal_arr = spinal_crop.get_array().astype(bool)
        signal_sfs = peak_centered_mean(t2w_slab[spinal_arr], bins=spinal_bins, peak_frac_height=spinal_peak_frac_height)

        out[f"mean_signal_vertebra_{start.name}-{goal.name}"] = round(signal_vertebra, 4)
        out[f"mean_signal_liquor_{start.name}-{goal.name}"] = round(signal_sfs, 4)
        out[f"mean_signal_liquor_{start.name}-{goal.name}_old"] = round(signal_sfs_old, 4)
        out[f"VBQ_{start.name}-{goal.name}"] = round(signal_vertebra / signal_sfs, 4)
        out[f"VBQ_{start.name}-{goal.name}_old"] = round(signal_vertebra / signal_sfs_old, 4)
    out["n_erode"] = n_erode
    out["signal_by_full_cord"] = full_cord
    out["spinal_bins"] = spinal_bins
    out["spinal_peak_frac_height"] = spinal_peak_frac_height
    return out


def body_composition_score(
    vibe_seg: NII,
    vert: NII,
    spine: NII,
    dataset_id: Literal[100, 12] = 100,
    regions: list[tuple[Vertebra_Instance, Vertebra_Instance]] | None = None,
    height_m: float | None = None,
) -> dict[str, float]:
    """Compute vertebral-level body composition measurements from a VIBE segmentation.

    For each vertebral region, the superior–inferior extent of the vertebral
    bodies is used to define the analysis region. Mean and maximum
    cross-sectional areas (CSA) of skeletal muscle, psoas, autochthonous
    muscles, visceral adipose tissue (VAT), and subcutaneous adipose tissue
    (SAT) are computed by averaging over all axial slices intersecting the
    vertebral body region.

    Parameters
    ----------
    vibe_seg : NII
        VIBE body composition segmentation.

    vert : NII
        Vertebral instance segmentation.

    spine : NII
        Spine subregion segmentation.
    dataset_id : {100, 12}, default=100
    Dataset definition used to determine the body composition label IDs.

    - ``100``: VIBESeg-100 label set.
    - ``12``: VIBESeg-12 label set.
    regions : list[tuple[Vertebra_Instance, Vertebra_Instance]], optional
        Vertebral ranges to analyse. Each tuple specifies the first and last
        vertebra (inclusive). Defaults to ``[(T12, L1), (L3, L4), (L3, L3)]``.

    height_m : float, optional
        Patient height in metres. If provided, the skeletal muscle index
        (CSA / m²) is additionally reported.

    Returns:
    -------
    dict[str, float]
        Dictionary containing body composition measurements for each
        region. Region tags are formatted as ``{start.name}-{goal.name}``
        (e.g. ``T12-L1``). For each region the following keys are set:

        - ``mean_{tissue}_area_{region}`` : mean cross-sectional area of
          the tissue over all axial slices intersecting the vertebral
          body region (mm²). ``tissue`` is one of
          ``muscle``, ``VAT``, ``SAT``, ``psoas``, ``autochthon``.
        - ``max_{tissue}_area_{region}`` : maximum cross-sectional area
          over the same slice range (mm²).
        - ``n_slices_{region}`` : number of axial slices contributing to
          the ``muscle`` statistic (integer).
        - ``muscle_index_{region}`` : skeletal muscle index
          (``mean_muscle_area / height_m^2``, unit mm²/m²). Only present
          when ``height_m`` is provided.
        - ``muscle_fat_ratio_{region}`` : ``mean_muscle_area /
          (mean_VAT_area + mean_SAT_area)`` (unitless). ``NaN`` when the
          denominator is zero.

        A region is silently skipped (no keys emitted) when no vertebral
        body voxels fall inside its label range.
    """
    if regions is None:
        regions = [
            (Vertebra_Instance.T12, Vertebra_Instance.L1),
            (Vertebra_Instance.L3, Vertebra_Instance.L4),
            (Vertebra_Instance.L3, Vertebra_Instance.L3),
        ]

    if vibe_seg.shape != vert.shape:
        vibe_seg = vibe_seg.resample_from_to(vert, verbose=False)

    if spine.shape != vert.shape:
        spine = spine.resample_from_to(vert, verbose=False)

    body_mask = spine.extract_label([Location.Vertebra_Corpus, Location.Vertebra_Corpus_border])

    if dataset_id == 12:
        measurements = {
            "muscle": [*Full_Body_Instance.muscle(), Full_Body_Instance.muscle_other],
            "VAT": Full_Body_Instance.inner_fat,
            "SAT": Full_Body_Instance.subcutaneous_fat,
            "psoas": [Full_Body_Instance.iliopsoas_left, Full_Body_Instance.iliopsoas_right],
            "autochthon": [Full_Body_Instance.autochthon_left, Full_Body_Instance.autochthon_right],
        }

    elif dataset_id == 100:
        measurements = {
            "muscle": [*Full_Body_Instance_Vibe.muscle_(), Full_Body_Instance_Vibe.muscle_other],
            "VAT": Full_Body_Instance_Vibe.inner_fat,
            "SAT": Full_Body_Instance_Vibe.subcutaneous_fat,
            "psoas": [Full_Body_Instance_Vibe.iliopsoas_left, Full_Body_Instance_Vibe.iliopsoas_right],
            "autochthon": [Full_Body_Instance_Vibe.autochthon_left, Full_Body_Instance_Vibe.autochthon_right],
        }

    else:
        raise NotImplementedError(f"Unsupported dataset_id={dataset_id}")

    verts_order = Vertebra_Instance.order()
    axis = vibe_seg.get_axis(direction="S")

    other_axes = tuple(i for i in range(3) if i != axis)
    voxel_area = float(vibe_seg.zoom[other_axes[0]] * vibe_seg.zoom[other_axes[1]])
    out = {}
    u = vert.unique()
    for start, goal in regions:
        end = verts_order.index(goal.get_next_poi(u))
        labels = verts_order[verts_order.index(start) : end]

        vertebral_body = vert.extract_label(labels) * body_mask

        if vertebral_body.sum() == 0:
            # print("no vertebra", labels, vert.unique())
            continue

        bbox = vertebral_body.compute_crop()

        slicer = [slice(None)] * 3
        slicer[axis] = bbox[axis]

        region: NII = vibe_seg[slicer]

        region_name = f"{start.name}-{goal.name}"

        for name, label_ids in measurements.items():
            mask = region.extract_label(label_ids)
            arr = mask.get_array().astype(bool)

            areas = arr.sum(axis=other_axes).astype(float) * voxel_area
            areas = areas[areas > 0]

            if len(areas) == 0:
                print(name, [int(a) for a in region.unique()], label_ids)
                mean_area = np.nan
                max_area = np.nan
                n_slices = 0
            else:
                mean_area = float(np.mean(areas))
                max_area = float(np.max(areas))
                n_slices = len(areas)

            out[f"mean_{name}_area_{region_name}"] = round(mean_area, 4)
            out[f"max_{name}_area_{region_name}"] = round(max_area, 4)

            if name == "muscle":
                out[f"n_slices_{region_name}"] = n_slices

                if height_m is not None and np.isfinite(mean_area):
                    out[f"muscle_index_{region_name}"] = round(mean_area / (height_m**2))

        vat = out[f"mean_VAT_area_{region_name}"]
        sat = out[f"mean_SAT_area_{region_name}"]
        muscle = out[f"mean_muscle_area_{region_name}"]

        if np.isfinite(vat) and np.isfinite(sat) and np.isfinite(muscle) and (vat + sat) > 0:
            out[f"muscle_fat_ratio_{region_name}"] = round(muscle / (vat + sat), 4)
        else:
            out[f"muscle_fat_ratio_{region_name}"] = np.nan

    return out


def muscle_fat_infiltration(
    water: NII,
    fat: NII,
    vibe_seg: NII,
    vert: NII | None = None,
    spine: NII | None = None,
    regions: list[tuple[Vertebra_Instance, Vertebra_Instance]] | list[tuple[None, None]] | None = None,
    roi: NII | None = None,
    roi_ids: tuple[int, ...] = tuple(range(3, 9)),
    dataset_id: Literal[100, 12] = 100,
    threshold: float = 0.20,
    erode: dict[str, int] | None = None,
    per_muscle: bool = True,
) -> dict[str, float]:
    """Compute muscle fat infiltration from Dixon VIBE water/fat images.

    The fat fraction (FF) is calculated voxel-wise as:

        FF = fat / (fat + water)

    within skeletal muscle masks. Voxels with FF >= threshold are classified
    as intramuscular adipose tissue (IMAT), while voxels below the threshold
    are classified as lean muscle.

    Measurements can optionally be restricted to vertebral levels and/or a
    supplied ROI. The analysis can be performed for total muscle and
    individual muscle groups. For each muscle group, muscle volumes are
    reported both with and without erosion of the muscle mask, so callers
    can compare the eroded region used for fat-fraction statistics against
    the raw segmentation volume.

    Parameters
    ----------
    water : NII
        Dixon water image.

    fat : NII
        Dixon fat image.

    vibe_seg : NII
        VIBE body composition segmentation.

    vert : NII, optional
        Vertebral instance segmentation. Required when ``regions`` is used.

    spine : NII, optional
        Spine subregion segmentation.

    regions : list[tuple[Vertebra_Instance, Vertebra_Instance]], optional
        Vertebral ranges for regional analysis. The superior-inferior extent
        of the vertebral bodies is used to restrict the calculation.

    roi : NII, optional
        Additional ROI mask restricting the analysis.

    roi_ids : tuple[int, ...], default=tuple(range(3, 9))
        ROI labels used when restricting the analysis.

    dataset_id : {100, 12}, default=100
        Body composition label definition.

    threshold : float, default=0.20
        Fat fraction threshold separating lean muscle and IMAT.

    erode : dict[str, int], optional
        Per-muscle number of erosion iterations applied to the muscle mask
        before fat-fraction statistics are computed. Defaults to
        ``{"all_muscle": 1, "iliopsoas_left": 1, "iliopsoas_right": 1,
        "autochthon_left": 2, "autochthon_right": 2, "muscle_other": 1}``.
        Erosion reduces partial-volume contamination at the muscle border
        but shrinks the mask; the pre-erosion volume is always reported
        alongside the eroded volume so both can be inspected.

    per_muscle : bool, default=True
        If True, compute values for individual muscles in addition to total
        muscle.

    Returns:
    -------
    dict[str, float]
        Muscle fat infiltration measurements. For each ``{region}_{muscle}``
        suffix the dictionary contains:

        - ``mean_fat_fraction_{suffix}`` : mean FF over the eroded mask.
        - ``median_fat_fraction_{suffix}`` : median FF over the eroded mask.
        - ``mean_lean_fat_fraction_{suffix}`` : mean FF of lean voxels
          (FF < ``threshold``).
        - ``mean_IMAT_fat_fraction_{suffix}`` : mean FF of IMAT voxels
          (FF >= ``threshold``).
        - ``muscle_volume_{suffix}`` : muscle volume after erosion (mm^3).
        - ``muscle_volume_no_erosion_{suffix}`` : muscle volume before
          erosion (mm^3); equals ``muscle_volume_{suffix}`` when no
          erosion is applied.
        - ``lean_muscle_volume_{suffix}`` : lean-muscle volume within the
          eroded mask (mm^3).
        - ``lean_muscle_volume_no_erosion_{suffix}`` : lean-muscle volume
          within the un-eroded mask (mm^3).
        - ``IMAT_volume_{suffix}`` : IMAT volume within the eroded mask
          (mm^3).
        - ``IMAT_volume_no_erosion_{suffix}`` : IMAT volume within the
          un-eroded mask (mm^3).
        - ``IMAT_fraction_{suffix}`` : IMAT voxel fraction within the
          eroded mask.
    """
    if erode is None:
        erode = {"all_muscle": 1, "iliopsoas_left": 1, "iliopsoas_right": 1, "autochthon_left": 2, "autochthon_right": 2, "muscle_other": 1}
    if water.shape != vibe_seg.shape:
        vibe_seg = vibe_seg.resample_from_to(water, verbose=False)
    if fat.shape != water.shape:
        fat = fat.resample_from_to(water, verbose=False)
    if vert is not None and vert.shape != water.shape:
        vert = vert.resample_from_to(water, verbose=False)
    if spine is not None and spine.shape != water.shape:
        spine = spine.resample_from_to(water, verbose=False)
    if roi is not None and roi.shape != water.shape:
        roi = roi.resample_from_to(water, verbose=False)

    # ------------------------------------------------------------------
    # Muscle label definitions
    # ------------------------------------------------------------------
    if dataset_id == 12:
        muscle_groups = {
            "all_muscle": [*Full_Body_Instance.muscle(), Full_Body_Instance.muscle_other],
            "iliopsoas_left": Full_Body_Instance.iliopsoas_left,
            "iliopsoas_right": Full_Body_Instance.iliopsoas_right,
            "autochthon_left": Full_Body_Instance.autochthon_left,
            "autochthon_right": Full_Body_Instance.autochthon_right,
            "muscle_other": Full_Body_Instance.muscle_other,
        }
    elif dataset_id == 100:
        muscle_groups = {
            "all_muscle": [*Full_Body_Instance_Vibe.muscle_(), Full_Body_Instance_Vibe.muscle_other],
            "iliopsoas_left": Full_Body_Instance_Vibe.iliopsoas_left,
            "iliopsoas_right": Full_Body_Instance_Vibe.iliopsoas_right,
            "autochthon_left": Full_Body_Instance_Vibe.autochthon_left,
            "autochthon_right": Full_Body_Instance_Vibe.autochthon_right,
            "muscle_other": Full_Body_Instance_Vibe.muscle_other,
        }
    else:
        raise NotImplementedError(f"Unsupported dataset_id={dataset_id}")
    if not per_muscle:
        muscle_groups = {"all_muscle": muscle_groups["all_muscle"]}
    if regions is None:
        regions = [(None, None)]
    verts = Vertebra_Instance.order()
    voxel_volume = water.voxel_volume()

    out = {}

    # ------------------------------------------------------------------
    # Vertebral regions
    # ------------------------------------------------------------------
    for start, goal in regions:
        region_name = "all"
        slicer = None

        if start is not None:
            if vert is None:
                raise ValueError("vert segmentation required when regions are provided")
            assert goal is not None
            end = verts.index(goal) + 1
            labels = verts[verts.index(start) : end]
            vertebra_region = vert.extract_label(labels)
            if spine is not None:
                vertebra_region *= spine.extract_label(Location.Vertebra_Corpus)
            bbox = vertebra_region.compute_crop()
            axis = vertebra_region.get_axis(direction="S")
            slicer = [slice(None)] * 3
            slicer[axis] = bbox[axis]
            region_name = f"{start.name}-{goal.name}"

        # --------------------------------------------------------------
        # Individual muscles
        # --------------------------------------------------------------
        for muscle_name, muscle_labels in muscle_groups.items():
            muscle_mask = vibe_seg.extract_label(muscle_labels)
            if roi is not None:
                muscle_mask *= roi.extract_label(roi_ids)
            if slicer is not None:
                muscle_mask = muscle_mask[slicer]

            mask_no_erode = muscle_mask.get_array().astype(bool)
            if erode[muscle_name] > 0:
                mask = muscle_mask.erode_msk(erode[muscle_name], verbose=False).get_array().astype(bool)
            else:
                mask = mask_no_erode

            if mask_no_erode.sum() == 0:
                continue

            water_arr = water.get_array()
            fat_arr = fat.get_array()
            if slicer is not None:
                water_arr = water_arr[slicer]  # type: ignore
                fat_arr = fat_arr[slicer]  # type: ignore
            denom = water_arr + fat_arr
            ff_full = np.zeros_like(denom, dtype=np.float32)
            valid = denom > 0
            ff_full[valid] = fat_arr[valid] / denom[valid]

            ff = ff_full[mask]
            ff_no_erode = ff_full[mask_no_erode]

            lean = ff < threshold
            imat = ff >= threshold
            lean_no_erode = ff_no_erode < threshold
            imat_no_erode = ff_no_erode >= threshold

            suffix = f"{region_name}_{muscle_name}"

            out[f"muscle_volume_no_erosion_{suffix}"] = round(float(ff_no_erode.size * voxel_volume), 4)
            out[f"lean_muscle_volume_no_erosion_{suffix}"] = round(float(np.sum(lean_no_erode) * voxel_volume), 4)
            out[f"IMAT_volume_no_erosion_{suffix}"] = round(float(np.sum(imat_no_erode) * voxel_volume), 4)

            if ff.size == 0:
                continue

            out[f"mean_fat_fraction_{suffix}"] = round(float(np.mean(ff)), 4)
            out[f"median_fat_fraction_{suffix}"] = round(float(np.median(ff)), 4)
            out[f"mean_lean_fat_fraction_{suffix}"] = round(float(np.mean(ff[lean])) if np.any(lean) else np.nan, 4)
            out[f"mean_IMAT_fat_fraction_{suffix}"] = round(float(np.mean(ff[imat])) if np.any(imat) else np.nan, 4)
            out[f"muscle_volume_{suffix}"] = round(float(ff.size * voxel_volume), 4)
            out[f"lean_muscle_volume_{suffix}"] = round(float(np.sum(lean) * voxel_volume), 4)
            out[f"IMAT_volume_{suffix}"] = round(float(np.sum(imat) * voxel_volume), 4)
            out[f"IMAT_fraction_{suffix}"] = round(float(np.mean(imat)), 4)
    out["threshold"] = threshold
    out["erode"] = erode
    return out


def torso_vat_sat_muscle_mass(
    vibe_seg: NII, roi: NII, dataset_id: Literal[100, 12] = 100, roi_ids: tuple[int, ...] = tuple(range(3, 9)), return_nii: bool = False
) -> tuple[dict, NII | None]:
    """Compute visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), and muscle volume (in mm³) from a torso segmentation.

    The segmentation is optionally restricted to the supplied ROI before
    calculating tissue volumes. Volumes are reported in physical units
    (voxel count × voxel volume, i.e. mm³).

    Parameters
    ----------
    vibe_seg : NII
        VIBE segmentation containing body composition labels.

    roi : NII
        ROI segmentation defining the torso region. If its geometry differs
        from ``vibe_seg``, it is resampled to match.

    dataset_id : {100, 12}, default=100
        Dataset definition used to determine the label IDs.

        - ``100``: VIBESeg-100 label set.
        - ``12``: VIBESeg-12 label set.

    roi_ids : tuple[int, ...], default=(3, 4, 5, 6, 7, 8)
        ROI labels that define the region in which body composition should be
        evaluated.

    return_nii : bool, default=False
        If True, additionally return binary NIfTI masks for VAT, SAT, and
        muscle.

    Returns:
    -------
    tuple[dict, NII | None]
        A tuple ``(results, body_comp)``.

        ``results`` contains:

        - ``VAT`` : float
            Visceral adipose tissue volume.
        - ``SAT`` : float
            Subcutaneous adipose tissue volume.
        - ``muscle_mass`` : float
            Muscle volume.
        - ``reason`` : str, optional
            Present only if computation failed.
        - ``nii`` : dict[str, NII], optional
            Returned only when ``return_nii=True``.

        ``body_comp`` is the ROI-restricted segmentation used for the
        computation, or ``None`` if the computation failed.

    Raises:
    ------
    NotImplementedError
        If an unsupported ``dataset_id`` is provided.

    Notes:
    -----
    The function verifies that both the clavicula and pelvis are present in
    the segmentation to ensure that the full torso is covered. If this check
    fails, NaN values are returned together with the failure reason.
    """
    if roi.shape != vibe_seg.shape:
        roi = roi.resample_from_to(vibe_seg, verbose=False)

    # Restrict computation to the requested ROI.
    body_comp = vibe_seg * roi.extract_label(roi_ids)

    results: dict[str, object] = {}

    VAT = SAT = muscle_mass = None

    try:
        labels = vibe_seg.unique()

        if dataset_id == 12:
            vat_id = Full_Body_Instance.inner_fat
            sat_id = Full_Body_Instance.subcutaneous_fat
            muscle_ids = [*Full_Body_Instance.muscle(), Full_Body_Instance.muscle_other]
            if Full_Body_Instance.clavicula_left.value not in labels:
                raise ValueError("Not the full torso visible (clavicula is missing)")

            if Full_Body_Instance.pelvis_left.value not in labels:
                raise ValueError("Not the full torso visible (pelvis is missing)")

        elif dataset_id == 100:
            if vibe_seg.max() > 80:
                raise AssertionError("Not a VIBESeg-100 (or compatible) segmentation.", vibe_seg.unique())
            vat_id = Full_Body_Instance_Vibe.inner_fat
            sat_id = Full_Body_Instance_Vibe.subcutaneous_fat
            muscle_ids = [*Full_Body_Instance_Vibe.muscle_(), Full_Body_Instance_Vibe.muscle_other]
            if Full_Body_Instance_Vibe.clavicula_left.value not in labels:
                raise ValueError("Not the full torso visible (clavicula is missing)")

            if Full_Body_Instance_Vibe.pelvis_left.value not in labels:
                raise ValueError("Not the full torso visible (pelvis is missing)")
        else:
            raise NotImplementedError(f"Unsupported dataset_id={dataset_id}")

        voxel_volume = body_comp.voxel_volume()

        VAT = body_comp.extract_label(vat_id)
        SAT = body_comp.extract_label(sat_id)
        muscle_mass = body_comp.extract_label(muscle_ids)

        results["VAT"] = round(voxel_volume * VAT.sum(), 4)
        results["SAT"] = round(voxel_volume * SAT.sum(), 4)
        results["muscle_mass"] = round(voxel_volume * muscle_mass.sum(), 4)

    except Exception as exc:
        body_comp = None
        results.update({"VAT": np.nan, "SAT": np.nan, "muscle_mass": np.nan, "reason": str(exc)})

    if return_nii:
        results["nii"] = {"VAT": VAT, "SAT": SAT, "muscle_mass": muscle_mass}

    return results, body_comp
