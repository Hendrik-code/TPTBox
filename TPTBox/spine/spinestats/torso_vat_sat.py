from typing import Literal

import numpy as np

from TPTBox import NII
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.vert_constants import Full_Body_Instance, Full_Body_Instance_Vibe, Location, Vertebra_Instance


def VBQ_score(
    t2w: NII,
    vert: NII,
    spine: NII,
    regions=None,
    subregs_ids=None,
    spinal_channel_id=Location.Spinal_Canal,
    n_erode=2,
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
        C3–C6, T5–T8, and L1.
    subregs_ids : list[Location], optional
        Spine subregion labels defining the vertebral body. Defaults to
        ``[Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]``.
    spinal_channel_id : Location, default=Location.Spinal_Canal
        Label identifying the spinal canal (CSF) region.
    n_erode : int, default=2
        Number of erosion iterations applied to the vertebral body mask
        before signal extraction.

    Returns:
    -------
    dict[str, float]
        Dictionary containing, for each region:
        - ``mean_signal_vertebra_<start>-<end>``
        - ``mean_signal_liquor_<start>-<end>``
        - ``VBQ_<start>-<end>``

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
            (Vertebra_Instance.L1, Vertebra_Instance.L1),
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

        spinal_crop = spinal_channel.copy()

        slicer = [slice(None)] * 3
        slicer[axis] = bbox[axis]
        spinal_crop = spinal_crop[slicer]

        signal_sfs = t2w.mean(where=spinal_crop)

        out[f"mean_signal_vertebra_{start.name}-{goal.name}"] = signal_vertebra
        out[f"mean_signal_liquor_{start.name}-{goal.name}"] = signal_sfs
        out[f"VBQ_{start.name}-{goal.name}"] = signal_vertebra / signal_sfs

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
        vertebra (inclusive). Defaults to T12–L1 and L3.

    height_m : float, optional
        Patient height in metres. If provided, the skeletal muscle index
        (CSA / m²) is additionally reported.

    Returns:
    -------
    dict[str, float]
        Dictionary containing body composition measurements for each region.
    """
    if regions is None:
        regions = [
            (Vertebra_Instance.T12, Vertebra_Instance.L1),
            (Vertebra_Instance.L3, Vertebra_Instance.L3),
        ]

    if vibe_seg.shape != vert.shape:
        vibe_seg = vibe_seg.resample_from_to(vert)

    if spine.shape != vert.shape:
        spine = spine.resample_from_to(vert)

    voxel_area = float(np.prod(vibe_seg.zoom[:2]))

    body_mask = spine.extract_label(Location.Vertebra_Corpus)

    if dataset_id == 100:
        measurements = {
            "muscle": [*Full_Body_Instance.muscle(), Full_Body_Instance.muscle_other],
            "VAT": Full_Body_Instance.inner_fat,
            "SAT": Full_Body_Instance.subcutaneous_fat,
            "psoas": [
                Full_Body_Instance.iliopsoas_left,
                Full_Body_Instance.iliopsoas_right,
            ],
            "autochthon": [
                Full_Body_Instance.autochthon_left,
                Full_Body_Instance.autochthon_right,
            ],
        }

    elif dataset_id == 12:
        measurements = {
            "muscle": [
                Full_Body_Instance_Vibe.muscle_(),
                Full_Body_Instance_Vibe.muscle_other,
            ],
            "VAT": Full_Body_Instance_Vibe.inner_fat,
            "SAT": Full_Body_Instance_Vibe.subcutaneous_fat,
            "psoas": [
                Full_Body_Instance_Vibe.iliopsoas_left,
                Full_Body_Instance_Vibe.iliopsoas_right,
            ],
            "autochthon": [
                Full_Body_Instance_Vibe.autochthon_left,
                Full_Body_Instance_Vibe.autochthon_right,
            ],
        }

    else:
        raise NotImplementedError(f"Unsupported dataset_id={dataset_id}")

    verts_order = Vertebra_Instance.order()
    axis = vibe_seg.get_axis(direction="S")

    out = {}

    for start, goal in regions:
        end = verts_order.index(goal) + 1
        labels = verts_order[verts_order.index(start) : end]

        vertebral_body = vert.extract_label(labels) * body_mask

        if vertebral_body.sum() == 0:
            continue

        bbox = vertebral_body.compute_crop()

        slicer = [slice(None)] * 3
        slicer[axis] = bbox[axis]

        region = vibe_seg[slicer]

        region_name = f"{start.name}-{goal.name}"

        for name, label_ids in measurements.items():
            mask = region.extract_label(label_ids)
            arr = mask.get_array().astype(bool)

            other_axes = tuple(i for i in range(3) if i != axis)
            areas = arr.sum(axis=other_axes).astype(float) * voxel_area
            areas = areas[areas > 0]

            if len(areas) == 0:
                mean_area = np.nan
                max_area = np.nan
                n_slices = 0
            else:
                mean_area = float(np.mean(areas))
                max_area = float(np.max(areas))
                n_slices = len(areas)

            out[f"mean_{name}_area_{region_name}"] = mean_area
            out[f"max_{name}_area_{region_name}"] = max_area

            if name == "muscle":
                out[f"n_slices_{region_name}"] = n_slices

                if height_m is not None and np.isfinite(mean_area):
                    out[f"muscle_index_{region_name}"] = mean_area / (height_m**2)

        vat = out[f"mean_VAT_area_{region_name}"]
        sat = out[f"mean_SAT_area_{region_name}"]
        muscle = out[f"mean_muscle_area_{region_name}"]

        if np.isfinite(vat) and np.isfinite(sat) and np.isfinite(muscle) and (vat + sat) > 0:
            out[f"muscle_fat_ratio_{region_name}"] = muscle / (vat + sat)
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
    erode: int = 0,
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
    individual muscle groups.

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

    erode : int, default=0
        Number of erosions applied to each muscle mask.

    per_muscle : bool, default=True
        If True, compute values for individual muscles in addition to total
        muscle.

    Returns:
    -------
    dict[str, float]
        Muscle fat infiltration measurements.
    """
    if water.shape != vibe_seg.shape:
        vibe_seg = vibe_seg.resample_from_to(water)
    if fat.shape != water.shape:
        fat = fat.resample_from_to(water)
    if vert is not None and vert.shape != water.shape:
        vert = vert.resample_from_to(water)
    if spine is not None and spine.shape != water.shape:
        spine = spine.resample_from_to(water)
    if roi is not None and roi.shape != water.shape:
        roi = roi.resample_from_to(water)

    # ------------------------------------------------------------------
    # Muscle label definitions
    # ------------------------------------------------------------------
    if dataset_id == 100:
        muscle_groups = {
            "all_muscle": [*Full_Body_Instance.muscle(), Full_Body_Instance.muscle_other],
            "iliopsoas_left": Full_Body_Instance.iliopsoas_left,
            "iliopsoas_right": Full_Body_Instance.iliopsoas_right,
            "autochthon_left": Full_Body_Instance.autochthon_left,
            "autochthon_right": Full_Body_Instance.autochthon_right,
            "muscle_other": Full_Body_Instance.muscle_other,
        }
    elif dataset_id == 12:
        muscle_groups = {
            "all_muscle": [Full_Body_Instance_Vibe.muscle_(), Full_Body_Instance_Vibe.muscle_other],
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
            if erode > 0:
                muscle_mask.erode_msk_(erode, verbose=False)
            mask = muscle_mask.get_array().astype(bool)
            if mask.sum() == 0:
                continue
            water_arr = water.get_array()
            fat_arr = fat.get_array()
            if slicer is not None:
                water_arr = water_arr[slicer]  # type: ignore
                fat_arr = fat_arr[slicer]  # type: ignore
            denom = water_arr + fat_arr
            ff = np.zeros_like(denom, dtype=np.float32)
            valid = denom > 0
            ff[valid] = fat_arr[valid] / denom[valid]
            ff = ff[mask]
            if ff.size == 0:
                continue

            lean = ff < threshold
            imat = ff >= threshold

            suffix = f"{region_name}_{muscle_name}"

            out[f"mean_fat_fraction_{suffix}"] = float(np.mean(ff))
            out[f"median_fat_fraction_{suffix}"] = float(np.median(ff))
            out[f"mean_lean_fat_fraction_{suffix}"] = float(np.mean(ff[lean])) if np.any(lean) else np.nan
            out[f"mean_IMAT_fat_fraction_{suffix}"] = float(np.mean(ff[imat])) if np.any(imat) else np.nan
            out[f"muscle_volume_{suffix}"] = float(ff.size * voxel_volume)
            out[f"lean_muscle_volume_{suffix}"] = float(np.sum(lean) * voxel_volume)
            out[f"IMAT_volume_{suffix}"] = float(np.sum(imat) * voxel_volume)
            out[f"IMAT_fraction_{suffix}"] = float(np.mean(imat))

    return out


def torso_vat_sat_muscle_mass(
    vibe_seg: NII, roi: NII, dataset_id: Literal[100, 12] = 100, roi_ids: tuple[int, ...] = tuple(range(3, 9)), return_nii: bool = False
) -> tuple[dict, NII | None]:
    """Compute visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), and muscle volume in mm from a torso segmentation.

    The segmentation is optionally restricted to the supplied ROI before
    calculating tissue volumes. Volumes are reported in physical units
    (voxel count × voxel volume).

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
        roi = roi.resample_from_to(vibe_seg)

    # Restrict computation to the requested ROI.
    body_comp = vibe_seg * roi.extract_label(roi_ids)

    results: dict[str, object] = {}

    VAT = SAT = muscle_mass = None

    try:
        labels = vibe_seg.unique()

        if dataset_id == 100:
            if vibe_seg.max() > 72:
                raise AssertionError("Not a VIBESeg-100 (or compatible) segmentation.")

            vat_id = Full_Body_Instance.inner_fat
            sat_id = Full_Body_Instance.subcutaneous_fat
            muscle_ids = [*Full_Body_Instance.muscle(), Full_Body_Instance.muscle_other]
            if Full_Body_Instance.clavicula_left.value not in labels:
                raise ValueError("Not the full torso visible (clavicula is missing)")

            if Full_Body_Instance.pelvis_left.value not in labels:
                raise ValueError("Not the full torso visible (pelvis is missing)")

        elif dataset_id == 12:
            vat_id = Full_Body_Instance_Vibe.inner_fat
            sat_id = Full_Body_Instance_Vibe.subcutaneous_fat
            muscle_ids = [Full_Body_Instance_Vibe.muscle_(), Full_Body_Instance_Vibe.muscle_other]
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

        results["VAT"] = voxel_volume * VAT.sum()
        results["SAT"] = voxel_volume * SAT.sum()
        results["muscle_mass"] = voxel_volume * muscle_mass.sum()

    except Exception as exc:
        body_comp = None
        results.update({"VAT": np.nan, "SAT": np.nan, "muscle_mass": np.nan, "reason": str(exc)})

    if return_nii:
        results["nii"] = {"VAT": VAT, "SAT": SAT, "muscle_mass": muscle_mass}

    return results, body_comp
