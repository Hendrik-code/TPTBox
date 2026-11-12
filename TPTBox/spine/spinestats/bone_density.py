"""Opportunistic bone density from CT: trabecular HU of the vertebral bodies (the CT counterpart of the MRI ``VBQ_score``).

Also reads the CT acquisition parameters (kVp, kernel, contrast ...) from the json sidecar to flag why HU may not be comparable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from TPTBox.core.bids_files import BIDS_FILE
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.vert_constants import Location, Vertebra_Instance

_PLAUSIBLE_HU_MEAN = (-200.0, 1500.0)  # own sanity range for "is this HU?", not from the literature


# ---------------------------------------------------------------- CT acquisition parameters (json sidecar)
_CONTRAST_KEYS = (
    "ContrastBolusAgent",
    "ContrastAgent",
    "ContrastBolusIngredient",
    "ContrastBolusVolume",
    "ContrastBolusTotalDose",
    "ContrastBolusStartTime",
)


def find_sidecar(ct: BIDS_FILE | str | Path) -> Path | None:
    """Path of the json sidecar of a CT (same name, ``.json``), or None if there is none. A path to a ``.json`` is returned as is."""
    if isinstance(ct, (str, Path)) and str(ct).endswith(".json"):
        return Path(ct) if Path(ct).exists() else None
    if isinstance(ct, BIDS_FILE):
        if "json" in ct.file:
            p = Path(ct.file["json"])
            return p if p.exists() else None
        nii = ct.file["nii.gz"] if "nii.gz" in ct.file else ct.get_nii_file()
        if nii is None:
            return None
        ct = Path(nii)
    ct = Path(ct)
    name = ct.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    p = ct.with_name(name + ".json")
    return p if p.exists() else None


def read_sidecar(ct: BIDS_FILE | str | Path) -> dict | None:
    """Parsed sidecar json, or None if it is missing, unreadable or not a json object."""
    p = find_sidecar(ct)
    if p is None:
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _floats(v: Any) -> list[float]:
    r"""'120', 120, [120], '0.68\0.68' -> list of floats (unparseable parts are dropped)."""
    if v is None:
        return []
    items = list(v) if isinstance(v, (list, tuple)) else str(v).replace("\\", " ").replace(",", " ").split()
    out = []
    for i in items:
        try:
            out.append(float(i))
        except (TypeError, ValueError):
            pass
    return out


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if d.get(k) not in (None, "", []):
            return d[k]
    return None


def _text(v: Any) -> str | None:
    if v is None:
        return None
    return " ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def ct_parameters(sidecar: dict | None) -> dict[str, Any]:
    """Normalised CT parameters from a sidecar dict (DICOM tag names as keys). Missing values are ``None``.

    ``contrast_agent_given`` is ``True`` if any contrast tag is set and ``None`` (unknown) otherwise: the absence of a
    tag does not prove a non-contrast scan.
    """
    d = sidecar or {}
    kvp = _floats(_first(d, "KVP", "kVp", "PeakKilovoltage"))
    thick = _floats(_first(d, "SliceThickness"))
    pix = _floats(_first(d, "PixelSpacing"))
    ma = _floats(_first(d, "XRayTubeCurrent", "TubeCurrent"))
    return {
        "sidecar_found": sidecar is not None,
        "kvp": kvp[0] if kvp else None,
        "tube_current_ma": ma[0] if ma else None,
        "slice_thickness_mm": thick[0] if thick else None,
        "pixel_spacing_mm": pix or None,
        "convolution_kernel": _text(_first(d, "ConvolutionKernel", "FilterType", "ReconstructionKernel")),
        "manufacturer": _text(_first(d, "Manufacturer")),
        "model": _text(_first(d, "ManufacturerModelName")),
        "series_description": _text(_first(d, "SeriesDescription")),
        "contrast_agent_given": True if any(d.get(k) not in (None, "", []) for k in _CONTRAST_KEYS) else None,
    }


def bone_density_caveats(params: dict[str, Any] | None) -> list[str]:
    # kVp: [2] (tube voltage drives the CT signal, calibrate at the scan's kVp); the equation of [3] was derived at 120 kVp.
    # Contrast: [4] (+2.5..+10.7 mg/cm3 depending on phase); [1] found no effect on classification AUC. Only a flag, not a correction.
    """Factual notes on why the CT HU of cancellous bone may not be comparable between scans."""
    if not params or not params.get("sidecar_found"):
        return []
    out = []
    if params.get("contrast_agent_given"):
        out.append("contrast agent given: vertebral body HU are raised and not comparable to non-contrast CT")
    kvp = params.get("kvp")
    if kvp is not None and kvp != 120:
        out.append(f"tube voltage is {kvp:g} kVp: HU of bone depend on kVp, values are not comparable to 120 kVp scans")
    if kvp is None:
        out.append("tube voltage (KVP) not in the sidecar")
    return out


def resolve_ct_params(ct_params: dict[str, Any] | BIDS_FILE | str | Path | None) -> dict[str, Any] | None:
    """Normalised CT parameters from whatever the caller has.

    Accepts the output of :func:`ct_parameters`, a raw sidecar dict, the path of the sidecar ``.json``, the path of the
    CT (its sibling ``.json`` is read) or a ``BIDS_FILE``.

    A path / ``BIDS_FILE`` without a readable sidecar gives ``sidecar_found=False`` (no caveats), never an error.
    """
    if ct_params is None:
        return None
    if isinstance(ct_params, dict):
        return ct_params if "sidecar_found" in ct_params else ct_parameters(ct_params)
    return ct_parameters(read_sidecar(ct_params))


@dataclass(frozen=True)
class Calibration:
    """Linear conversion ``bmd = slope * HU + intercept`` with the conditions it is valid for."""

    slope: float
    intercept: float
    unit: str = "mg/cm3"
    kvp: float | None = None  # tube voltage the equation was derived at; None = not restricted
    sd: float | None = None  # residual SD of the published fit, in `unit`
    qct_categories: bool = False  # whether `unit` is QCT vBMD for which the 80 / 120 mg/cm3 categories apply
    reference: str = ""


CALIBRATIONS: dict[str, Calibration] = {
    # [3]: HU = 0.33 + 1.211 * vBMD (residual SD 6.9 HU)  ->  vBMD = (HU - 0.33) / 1.211, SD = 6.9 / 1.211 = 5.7 mg/cm3.
    # Consistency check with [3]'s own cut-off: HU 99.6 -> 82.0 mg/cm3 (QCT osteoporosis limit: 80).
    # [3] also reports a GEE slope of 0.772 mg/cm3 per HU (no intercept) vs 1/1.211 = 0.826 here: ~7% method uncertainty.
    "andresen2026": Calibration(
        slope=1 / 1.211,
        intercept=-0.33 / 1.211,
        unit="mg/cm3 K2HPO4-equivalent (Mindways QCT)",
        kvp=120.0,
        sd=6.9 / 1.211,
        qct_categories=True,
        reference="Andresen et al., Diagnostics 2026;16(16):2640 (GE Revolution EVO 64, 120 kVp, manual cancellous ROI)",
    ),
}


def resolve_calibration(calibration: Calibration | tuple[float, float] | str | None) -> Calibration | None:
    """``None``, a name from :data:`CALIBRATIONS`, a :class:`Calibration` or a ``(slope, intercept)`` tuple -> :class:`Calibration`."""
    if calibration is None or isinstance(calibration, Calibration):
        return calibration
    if isinstance(calibration, str):
        if calibration not in CALIBRATIONS:
            raise ValueError(f"unknown calibration {calibration!r}, choose one of {sorted(CALIBRATIONS)} or pass (slope, intercept)")
        return CALIBRATIONS[calibration]
    return Calibration(slope=float(calibration[0]), intercept=float(calibration[1]))


def qct_category(vbmd_mg_cm3: float) -> str:
    """QCT vBMD category as stated in [3]: normal > 120, osteopenia 80-120, osteoporosis < 80 mg/cm3."""
    if vbmd_mg_cm3 < 80:
        return "osteoporosis"
    return "osteopenia" if vbmd_mg_cm3 <= 120 else "normal"


# ---------------------------------------------------------------- bone density
def _stats(values: np.ndarray, voxel_volume: float, calibration: Calibration | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mean_hu": round(float(values.mean()), 3),
        "median_hu": round(float(np.median(values)), 3),
        "std_hu": round(float(values.std()), 3),
        "n_voxels": int(values.size),
        "volume_mm3": round(float(values.size * voxel_volume), 2),
    }
    if calibration is not None:
        out["bmd_mg_cm3"] = round(calibration.slope * out["mean_hu"] + calibration.intercept, 3)
        if calibration.qct_categories:
            out["qct_category"] = qct_category(out["bmd_mg_cm3"])
    return out


def _aligned(img: NII, ref: NII) -> NII:
    same = img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3)
    return img if same else img.resample_from_to(ref)


def ct_bone_density(
    ct: NII,
    vert: NII,
    spine: NII,
    regions: Sequence[tuple[Vertebra_Instance, Vertebra_Instance]] | None = None,
    subregs_ids: Sequence[Location] | None = None,
    erode_mm: float = 3.0,  # own default; [3] excluded the outer ~4 mm (cube ROI), [1] used a manual central ROI
    hu_range: tuple[float, float] | None = None,
    per_vertebra: bool = True,
    calibration: Calibration | tuple[float, float] | str | None = None,
    ct_params: dict[str, Any] | BIDS_FILE | str | Path | None = None,
) -> dict[str, Any]:
    """Trabecular attenuation (HU) of the vertebral bodies in a CT; the CT counterpart of :func:`VBQ_score`.

    For every vertebra the vertebral body (``subregs_ids`` of ``spine`` inside the vertebra's instance label) is eroded
    by ``erode_mm`` millimetres (Euclidean, voxel-size aware) so that the cortical shell and partial-volume voxels are
    excluded, and the HU statistics of the remaining (mostly cancellous) voxels are reported. Regions pool the voxels of
    all their vertebrae that are present.

    Parameters
    ----------
    ct : NII
        CT image in Hounsfield units (not normalised, not a window-clipped copy).
    vert : NII
        Vertebra instance segmentation.
    spine : NII
        Spine subregion segmentation (vertebral body labels).
    regions : list[tuple[Vertebra_Instance, Vertebra_Instance]], optional
        Inclusive vertebra ranges to pool. Defaults to ``[(L1, L4)]``. Which levels to use (L1 alone, L1-L2, L1-L3 ...)
        is a protocol choice; ``per_vertebra`` always gives the single levels so any pooling can be recomputed.
    subregs_ids : list[Location], optional
        Subregion labels that make up the vertebral body. Defaults to ``[Vertebra_Corpus, Vertebra_Corpus_border]``
        (the same as ``VBQ_score``).
    erode_mm : float, default=3.0
        Euclidean erosion of the body mask in mm. A practical default, not a standard; 0 disables the erosion.
    hu_range : (float, float), optional
        If given, only voxels inside ``[low, high]`` HU are used (e.g. to drop fat/air from segmentation errors or
        sclerotic / cement voxels). The number of dropped voxels is reported as ``n_excluded``.
    per_vertebra : bool, default=True
        Add a ``"per_vertebra"`` entry with the statistics of each vertebra.
    calibration : str | (slope, intercept) | Calibration, optional
        Adds ``bmd_mg_cm3 = slope * mean_hu + intercept`` to each entry. ``"andresen2026"`` is the one published equation
        found that is usable (reference [3] in the module header): ``vBMD = (HU - 0.33) / 1.211`` in QCT K2HPO4-equivalent
        mg/cm3, derived on one GE scanner at **120 kVp** with a manual cancellous ROI; it also adds ``qct_category``
        (normal / osteopenia / osteoporosis). Its residual SD is ~5.7 mg/cm3 and, across scanners, [2] reports ~14% SD, so
        expect individual errors of that size; it has not been validated on our segmentation-based ROI. A named
        calibration is only applied if the kVp allows it: a known kVp other than the calibration's gives no BMD (warning),
        an unknown kVp gives the BMD with a warning (give ``ct_params`` to check). Your own ``(slope, intercept)`` has no
        such restriction. Published HU -> BMD equations depend on scanner, kVp and contrast phase ([2], [3], [4]); HA
        (phantom) and K2HPO4 units are not interchangeable.
    ct_params : dict | str | Path | BIDS_FILE, optional
        CT acquisition parameters (kVp, contrast ...). Accepts the output of :func:`ct_parameters`, a raw sidecar dict,
        the path of the sidecar ``.json``, the path of the CT (its sibling ``.json`` is read) or a ``BIDS_FILE``
        (see :func:`resolve_ct_params`). Contrast agent / non-120 kVp / missing kVp are added to ``"warnings"`` and the
        normalised parameters echoed in ``"settings"["ct_params"]``. Nothing is corrected, only flagged.

    Returns:
    -------
    dict
        ``"per_vertebra"``: ``{"L1": {mean_hu, median_hu, std_hu, n_voxels, volume_mm3, [n_excluded]}, ...}`` and one
        entry per region ``"L1-L4"`` with the same statistics plus ``vertebrae`` (the levels that contributed) and
        ``complete`` (all levels of the range present). Entries are ``None`` if no voxel is left.
        ``"warnings"`` lists problems (non-HU-looking values, empty ROI) and ``"settings"`` the parameters used.

    Notes:
    -----
    Valid for non-contrast or comparable-phase CT. Contrast enhancement, vertebral fractures, metastases, sclerotic
    lesions, cement and implants change the HU and are not detected here; the median and ``hu_range`` are more robust.
    """
    if regions is None:
        regions = [(Vertebra_Instance.L1, Vertebra_Instance.L4)]
    if subregs_ids is None:
        subregs_ids = [Location.Vertebra_Corpus, Location.Vertebra_Corpus_border]
    vert = _aligned(vert, ct)
    spine = _aligned(spine, ct)

    ct_arr = ct.get_array()
    vert_arr = vert.get_seg_array()
    body_arr = np.isin(spine.get_seg_array(), [int(s.value) for s in subregs_ids])
    zoom = tuple(float(z) for z in ct.zoom)
    voxel_volume = float(np.prod(zoom))
    order = Vertebra_Instance.order()

    warnings: list[str] = []
    ct_params = resolve_ct_params(ct_params)
    calibration = resolve_calibration(calibration)
    cal_used = calibration  # None if the scan's kVp rules the calibration out
    if calibration is not None and calibration.kvp is not None:
        kvp = (ct_params or {}).get("kvp")
        if kvp is None:
            warnings.append(f"kVp unknown: the calibration was derived at {calibration.kvp:g} kVp, bmd_mg_cm3 is only valid for such scans")
        elif kvp != calibration.kvp:
            cal_used = None
            warnings.append(f"scan is {kvp:g} kVp but the calibration is only valid at {calibration.kvp:g} kVp: bmd_mg_cm3 not computed")
    values: dict[Vertebra_Instance, np.ndarray] = {}
    excluded: dict[Vertebra_Instance, int] = {}
    present = {int(i) for i in np.unique(vert_arr)}
    for v in order:
        if int(v.value) not in present:
            continue
        mask_full = vert_arr == v.value
        box = ndimage.find_objects(mask_full.astype(np.uint8))[0]
        # margin so the distance transform sees the true border of the vertebra
        pad = [max(0, int(np.ceil(erode_mm / z)) + 1) for z in zoom]
        sl = tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(box, pad, ct_arr.shape))
        mask = (mask_full & body_arr)[sl]
        if erode_mm > 0 and mask.any():
            mask = ndimage.distance_transform_edt(mask, sampling=zoom) > erode_mm
        vals = ct_arr[sl][mask].astype(np.float64)
        if hu_range is not None:
            keep = (vals >= hu_range[0]) & (vals <= hu_range[1])
            excluded[v] = int((~keep).sum())
            vals = vals[keep]
        if vals.size == 0:
            warnings.append(f"{v.name}: no voxel left in the vertebral body ROI (erode_mm={erode_mm})")
            continue
        values[v] = vals

    out: dict[str, Any] = {}
    if per_vertebra:
        out["per_vertebra"] = {}
        for v, vals in values.items():
            s = _stats(vals, voxel_volume, cal_used)
            if hu_range is not None:
                s["n_excluded"] = excluded[v]
            out["per_vertebra"][v.name] = s

    for start, goal in regions:
        levels = order[order.index(start) : order.index(goal) + 1]
        used = [v for v in levels if v in values]
        key = f"{start.name}-{goal.name}"
        if not used:
            out[key] = None
            continue
        entry = _stats(np.concatenate([values[v] for v in used]), voxel_volume, cal_used)
        entry["vertebrae"] = [v.name for v in used]
        entry["complete"] = len(used) == len(levels)
        if hu_range is not None:
            entry["n_excluded"] = sum(excluded[v] for v in used)
        out[key] = entry

    for k, e in [*out.items(), *out.get("per_vertebra", {}).items()]:
        if isinstance(e, dict) and "mean_hu" in e and not _PLAUSIBLE_HU_MEAN[0] <= e["mean_hu"] <= _PLAUSIBLE_HU_MEAN[1]:
            warnings.append(f"{k}: mean {e['mean_hu']} is outside {_PLAUSIBLE_HU_MEAN}; is the image in Hounsfield units?")
    if ct_params is not None:
        warnings.extend(bone_density_caveats(ct_params))
    out["warnings"] = warnings
    out["settings"] = {
        "erode_mm": erode_mm,
        "subregions": [s.name for s in subregs_ids],
        "hu_range": list(hu_range) if hu_range is not None else None,
        "calibration": asdict(calibration) if calibration is not None else None,
        "calibration_applied": cal_used is not None,
        "ct_params": ct_params,
    }
    return out
