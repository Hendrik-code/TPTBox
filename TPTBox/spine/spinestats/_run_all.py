"""End-to-end NAKO subject processing.

For a given NAKO subject id, this module resolves the required inputs
(T2w + VIBE + spine/vertebra segmentation + VIBESeg-100 + ROI), runs the
spine and body-composition analyses, writes a single json with all
results, and (optionally) streams those results into Excel summaries in
parallel.

The full documentation of the produced json keys, their units, and the
per-function conventions used by the pipeline lives in the folder
README (``TPTBox/spine/spinestats/README.md``) and its standalone copy
``all_output_reference.md`` — both are meant to be read by clinicians
reviewing the numbers.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import queue as _queue
from pathlib import Path
from typing import Any

from tqdm import tqdm

from TPTBox import BIDS_FILE, POI, Print_Logger
from TPTBox.core.dicom.dicom2nii_utils import load_json
from TPTBox.core.internal.nii_help import save_json
from TPTBox.core.nii_wrapper import to_nii
from TPTBox.spine.spinestats._load_nako import loop_over_repaired_nako

DATASET_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako")
logger = Print_Logger()
# Version stamp written into every aggregate `_stat.json` at
# ``_provenance.version`` (built by :func:`_build_provenance`). Bump this whenever
# the produced numbers change in a way that requires already-cached files to be
# recomputed. Current bumps:
#   1 -- initial release (WK-direction based cobb/lordosis; no S1 sacrum endplate).
#   2 -- endplate-plane based lordosis / kyphosis (average of the two flanking
#        endplates at each disc), S1 sacrum-endplate landmarks, and correct
#        cranio-caudal ordering for the T13 annotation label.
CURRENT_VERSION = 2

# Top-level keys we require inside a finished json before we consider a
# subject "done" and skip recomputation. cobb/curv are optional and only
# added when run_all is called with cobb=True.
REQUIRED_MAIN_KEYS: tuple[str, ...] = (
    "ivd_geometry",
    "vert_geometry",
    "VBQ_score",
    "body_composition_score",
    "muscle_fat_infiltration",
    "torso_vat_sat_muscle_mass",
    # Extended curvature / balance / pelvic metrics (see curvature.py and pelvic_parameters.py).
    "sva",
    "coronal_balance",
    "axial_rotation",
    "segmental_endplate_angles",
    "curvature_profile",
    "multi_cobb",
    "pelvic_parameters",
)


def get_nako_paths(nako_id: str) -> dict[str, Path | None]:
    """Return a dict with all relevant paths for a given NAKO id.

    Keys:
        t2w, vibe-water, vibe-fat, vibe-inphase, vibe-outphase,
        vert, spine, roi, vibeseg100, dataset.

    Replaces the raw vibe stitched with the corrected version if both
    corrected image + json exist.
    """
    sub = str(nako_id).split("_")[0].replace("sub-", "")
    pfx = sub[:3]

    t2w_stitched = DATASET_ROOT / f"rawdata_stitched/{pfx}/{sub}/T2w/sub-{sub}_sequ-stitched_acq-sag_T2w.nii.gz"
    out = {
        f"vibe-{a}": DATASET_ROOT / f"rawdata_stitched/{pfx}/{sub}/vibe/sub-{sub}_sequ-stitched_acq-ax_part-{a}_vibe.nii.gz"
        for a in ["water", "fat", "inphase", "outphase"]
    }
    vibe_corr = (
        DATASET_ROOT
        / f"derivatives_Abdominal-Segmentation/{pfx}/{sub}/vibe/sub-{sub}_sequ-stitched_acq-ax_part-water_desc-corrected_vibe.nii.gz"
    )
    vibe_corr_json = vibe_corr.with_suffix("").with_suffix(".json")
    if vibe_corr.exists() and vibe_corr_json.exists():
        out["vibe-water"] = vibe_corr

    # current best T2w spine seg (mirrors qa_spine_shift.get_current_best_T2w_seg)
    search_folders = [
        "derivatives_spine_vert_fixed",
        "derivatives_spine_inference_combination162_148",
    ]
    vert = spine = None
    for s in search_folders:
        base = DATASET_ROOT / f"{s}/{pfx}/{sub}/T2w"
        v = base / f"sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-vert_msk.nii.gz"
        sp = base / f"sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-spine_msk.nii.gz"
        if v.exists():
            vert, spine = v, sp
            break

    fullbody_poi = (
        DATASET_ROOT / f"derivatives-fullbody-poi/{pfx}/{sub}/vibe/sub-{sub}_sequ-stitched_acq-ax_part-water_seg-fullbody_poi.json"
    )
    veridah = None
    for suffix in ("VERIDAH-label-V2", "VERIDAH-label"):
        p = (
            DATASET_ROOT / f"derivatives_spine_inference_162_sacrumfix/{pfx}/{sub}/T2w/"
            f"sub-{sub}_sequ-stitched_acq-sag_mod-T2w_seg-vert_desc-{suffix}_stat.json"
        )
        if p.exists():
            veridah = p
            break
    roi = (
        DATASET_ROOT / f"derivatives_Abdominal-Segmentation/{pfx}/{sub}/vibe/sub-{nako_id}_sequ-stitched_acq-ax_mod-vibe_seg-ROI_msk.nii.gz"
    )
    vibeseg100 = (
        DATASET_ROOT
        / f"derivatives_Abdominal-Segmentation/{pfx}/{sub}/vibe/sub-{nako_id}_sequ-stitched_acq-ax_mod-vibe_part-inphase_seg-VibeSeg-100_msk.nii.gz"
    )

    return {
        "t2w": t2w_stitched if t2w_stitched.exists() else None,
        **out,
        "vert": vert,
        "spine": spine,
        "roi": roi,
        "vibeseg100": vibeseg100 if vibeseg100.exists() else None,
        "fullbody_poi": fullbody_poi if fullbody_poi.exists() else None,
        "veridah": veridah,
        "dataset": DATASET_ROOT,
    }


def _segmentation_inputs(file_dict: dict) -> list[Path]:
    """Segmentation files whose mtime should invalidate a cached json."""
    keys = ("vert", "spine", "vibeseg100", "roi")
    return [
        file_dict[k].file["nii.gz"] if isinstance(file_dict[k], BIDS_FILE) else Path(file_dict[k])
        for k in keys
        if file_dict.get(k) is not None
    ]


def _stat_version(loaded_stat: dict | None) -> int:
    """Return the ``_provenance.version`` of a loaded stat dict (defaults to 1)."""
    if not isinstance(loaded_stat, dict):
        return 1
    prov = loaded_stat.get("_provenance")
    if isinstance(prov, dict):
        try:
            return int(prov.get("version", 1))
        except (TypeError, ValueError):
            return 1
    return 1


def _poi_is_stale_wrt_stat(stat_path: Path, loaded_stat: dict) -> bool:
    """Return True when the POI buffer sitting next to ``stat_path`` should be rebuilt.

    Staleness is derived from the stat json's ``_provenance.version``: whenever
    that reads < :data:`ANGLES_VERSION`, the sibling POI buffer is treated as
    outdated (v1 stat + v1 POI travelled together). ``loaded_stat`` is the
    already-loaded stat dict — pass ``{}`` if none exists (then nothing is stale
    since there's no v1 marker to invalidate against).
    """
    if not stat_path.exists() or not loaded_stat:
        return False
    return _stat_version(loaded_stat) < CURRENT_VERSION


def _is_cache_valid(json_path: Path, seg_files: list[Path], required_keys: tuple[str, ...]) -> tuple[bool, dict | None]:
    """Return (valid, loaded_dict).

    Cache is invalid (and needs recompute) if:
      - json does not exist,
      - json is older than any segmentation file,
      - json fails to parse,
      - any required main key is missing from the loaded dict.
    """
    # TODO(hash-invalidation): also compare each entry in data["_provenance"]["inputs"]
    # against the current file's sha1 (see _build_provenance) and force recompute on
    # mismatch. Deferred while we assume inputs are unchanged.
    if not json_path.exists():
        return False, None
    json_mtime = json_path.stat().st_mtime
    for seg in seg_files:
        if seg.exists() and seg.stat().st_mtime > json_mtime:
            return False, None
    try:
        data = load_json(json_path)
    except Exception:
        return False, None
    if not isinstance(data, dict):
        return False, None
    for k in required_keys:
        if k not in data:
            return False, None
    if _stat_version(data) < CURRENT_VERSION:
        return False, None
    return True, data


_PROVENANCE_INPUT_KEYS: tuple[str, ...] = (
    "t2w",
    "vibe_part-water",
    "vibe_part-fat",
    "vibe_part-inphase",
    "vibe_part-outphase",
    "vibe-water",
    "vibe-fat",
    "vibe-inphase",
    "vibe-outphase",
    "vert",
    "spine",
    "vibeseg100",
    "roi",
    "fullbody_poi",
    "veridah",
)


def _resolve_prov_path(v) -> Path | None:
    """Best-effort ``file_dict`` value → filesystem Path for provenance recording."""
    if v is None:
        return None
    if isinstance(v, BIDS_FILE):
        nii = v.get_nii_file()
        if nii is not None:
            return Path(nii)
        j = v.file.get("json") if hasattr(v, "file") else None
        return Path(j) if j is not None else None
    if isinstance(v, (str, Path)):
        s = str(v)
        return Path(s) if s else None
    return None


def _file_provenance(path: Path, prior: dict | None = None) -> dict:
    """Return provenance dict for ``path``.

    Shape is either ``{"path", "mtime_ns", "sha1"}`` or, when the file is
    missing, ``{"path", "missing": True}``.  When ``prior`` has the same
    ``mtime_ns`` as the current file, its ``sha1`` is reused to avoid
    re-hashing — this is what keeps reruns cheap while the "assume inputs
    unchanged" mode is in effect.
    """
    import hashlib

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "missing": True}
    st = p.stat()
    if isinstance(prior, dict) and prior.get("mtime_ns") == st.st_mtime_ns and isinstance(prior.get("sha1"), str):
        return {"path": str(p), "mtime_ns": st.st_mtime_ns, "sha1": prior["sha1"]}
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "mtime_ns": st.st_mtime_ns, "sha1": h.hexdigest()}


def _build_provenance(file_dict: dict, poi_out: Path | str | None, prior: dict | None) -> dict:
    """Build the ``_provenance`` block for an aggregated per-subject JSON.

    Records ``path`` / ``mtime_ns`` / ``sha1`` for every known input in
    ``file_dict`` plus the POI json at ``poi_out``.  Reuses
    ``prior["inputs"][k]`` to skip re-hashing files whose mtime is unchanged.
    """
    from datetime import datetime, timezone

    prior_inputs = (prior or {}).get("inputs", {}) if isinstance(prior, dict) else {}
    inputs: dict[str, dict] = {}
    for key in _PROVENANCE_INPUT_KEYS:
        if key not in file_dict:
            continue
        p = _resolve_prov_path(file_dict[key])
        if p is None:
            continue
        inputs[key] = _file_provenance(p, prior_inputs.get(key))
    if poi_out is not None:
        p = Path(poi_out)
        if p.exists():
            inputs["poi"] = _file_provenance(p, prior_inputs.get("poi"))
    return {
        "version": CURRENT_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
    }


def run_all(
    file_dict,
    override: bool = False,
    do_not_update=False,
    need_cobb=True,
    need_ivd=True,
    need_vert=True,
    need_vbq=True,
    need_bcs=True,
    need_mfi=True,
    need_torso=True,
    need_curvature=True,
    need_pelvic=True,
) -> dict[str, Any] | None:
    """Run the full pipeline for one subject and return the results dict.

    Parameters
    ----------
    file_dict : dict
        Output of :func:`get_nako_paths`.
    cobb : bool, default=False
        If True, additionally compute Cobb / lordosis / kyphosis angles
        (adds keys ``cobb`` and ``curv``).
    override : bool, default=False
        If False (default), skip recomputation and return the existing
        json when it is still valid. The cache is considered valid when:

        - the target json exists,
        - it is newer than every segmentation file listed in
          ``_segmentation_inputs(file_dict)``,
        - and it contains every key in :data:`REQUIRED_MAIN_KEYS`
          (plus ``cobb``/``curv`` when ``cobb=True``).

        If True, all algorithms run even when a valid json already
        exists and the json is overwritten.

    Returns:
    -------
    dict
        The full results dictionary (either freshly computed or loaded
        from the cached json). See ``doc/all_output_reference.md`` for
        the meaning of each key.
    """
    from TPTBox import Location, calc_poi_from_subreg_vert
    from TPTBox.spine.spinestats.angles import plot_cobb_and_lordosis_and_kyphosis
    from TPTBox.spine.spinestats.curvature import (
        compute_axial_rotation,
        compute_coronal_balance,
        compute_curvature_profile,
        compute_multi_cobb,
        compute_segmental_endplate_angles,
        compute_sva,
        compute_wedge_metrics,
    )
    from TPTBox.spine.spinestats.measure_ivd_and_vertebra_geometry import (
        measure_ivd_and_vertebra_geometry,  # structure_label: int = 100 and structure_label: int = 49
    )
    from TPTBox.spine.spinestats.pelvic_parameters import compute_pelvic_parameters
    from TPTBox.spine.spinestats.torso_vat_sat import VBQ_score, body_composition_score, muscle_fat_infiltration, torso_vat_sat_muscle_mass

    if "t2w" not in file_dict:
        return None

    t2w_bf = file_dict["t2w"] if isinstance(file_dict["t2w"], BIDS_FILE) else BIDS_FILE(file_dict["t2w"], file_dict["dataset"])
    poi_out = t2w_bf.get_changed_path(
        "json",
        "poi",
        "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2",
        info={"seg": "vert", "mod": "T2w", "desc": "vert-rotation"},
    )
    cobb_jpg_out = t2w_bf.get_changed_path(
        "jpg", "snp", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "cobb"}
    )
    final_out = t2w_bf.get_changed_path(
        "json", "stat", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "all"}
    )
    final_out = Path(final_out)

    required = (*REQUIRED_MAIN_KEYS, "cobb", "curv")
    seg_files = _segmentation_inputs(file_dict)

    out: dict[str, Any] = {}
    if not override:
        valid, cached = _is_cache_valid(final_out, seg_files, required)
        if valid and cached is not None and do_not_update:
            return cached
        # Reload existing json (if any) and only recompute the missing top-level keys.
        if final_out.exists():
            try:
                loaded = load_json(final_out)
                if isinstance(loaded, dict):
                    out = loaded
            except Exception:
                out = {}
        # Version stale -> drop the ANGLE-related keys so the corresponding _need()
        # checks trigger a recompute of just the affected metrics + JPGs, without
        # invalidating the expensive body-composition / VBQ / muscle_fat / torso
        # blocks that don't depend on the endplate-based lordosis fix.
        _cur_ver = _stat_version(out)
        if _cur_ver < 2:
            logger.on_warning("version bump", _cur_ver, "->", CURRENT_VERSION, ": recomputing angle keys")
            for _k in (
                "cobb",
                "curv",
                "curv_veridah",
                "sva",
                "coronal_balance",
                "axial_rotation",
                "segmental_endplate_angles",
                "curvature_profile",
                "multi_cobb",
                "endplate_internal_angle",
            ):
                out.pop(_k, None)

    def _need(*keys: str, compute: bool) -> bool:
        return override or (any(k not in out for k in keys) and compute)

    need_cobb = _need("cobb", "curv", compute=need_cobb)
    need_ivd = _need("ivd_geometry", compute=need_ivd)
    need_vert = _need("vert_geometry", compute=need_vert)
    need_vbq = _need("VBQ_score", compute=need_vbq)
    need_bcs = _need("body_composition_score", compute=need_bcs)
    need_mfi = _need("muscle_fat_infiltration", compute=need_mfi)
    need_torso = _need("torso_vat_sat_muscle_mass", compute=need_torso)
    _curvature_keys = ("sva", "coronal_balance", "axial_rotation", "segmental_endplate_angles", "curvature_profile", "multi_cobb")
    need_curvature = _need(*_curvature_keys, compute=need_curvature)
    need_pelvic = _need("pelvic_parameters", compute=need_pelvic)
    # Recompute area
    save = False
    if "VBQ_score" in out and "VBQ_L1-L1_old" in out["VBQ_score"]:
        logger.on_warning("redo vbq", t2w_bf.get("sub"))
        need_vbq = True
        del out["VBQ_score"]
        save = True
    if "torso_vat_sat_muscle_mass" in out and "Not a VIBESeg-100" in str(out.get("torso_vat_sat_muscle_mass", {}).get("reason", "")):
        logger.on_warning("redo torso_vat_sat_muscle_mass", t2w_bf.get("sub"))
        need_torso = True
        del out["torso_vat_sat_muscle_mass"]
        save = True
    # Redo pelvic_parameters if the cached entry is just an error stub (e.g. old runs where
    # fullbody_poi wasn't resolved), OR if it was written with the old unsigned-SS
    # implementation (detected via the PI = PT + SS invariant violated by > 0.1 deg).
    if "pelvic_parameters" in out:
        pp = out.get("pelvic_parameters", {})
        redo = False
        if isinstance(pp, dict):
            if pp.get("fullbody_poi_json") is None and "error" in pp:
                redo = True
            else:
                for variant_key in ("poi_ap", "poi_ala"):
                    v = pp.get(variant_key)
                    if not isinstance(v, dict):
                        continue
                    pi = v.get("pi_deg")
                    pt = v.get("pt_deg")
                    ss = v.get("ss_deg")
                    if pi is None or pt is None or ss is None:
                        continue
                    try:
                        if abs(float(pi) - (float(pt) + float(ss))) > 0.1:
                            redo = True
                            break
                    except (TypeError, ValueError):
                        continue
        if redo:
            need_pelvic = True
            del out["pelvic_parameters"]
            save = True
    # Retrospectively fold per-vertebra metrics into vert_geometry / ivd_geometry
    # for cached JSONs written before the merge was in place. Cheap and idempotent.
    if any(k in out for k in ("axial_rotation", "endplate_internal_angle", "segmental_endplate_angles")):
        _merge_per_vertebra_metrics(out)
        save = True
    ####
    need_veridah = override or "curv_veridah" not in out
    need_poi = need_cobb or need_ivd or need_vert or need_curvature or need_veridah
    need_t2w = need_ivd or need_vert or need_vbq
    need_vert_nii = need_poi or need_vbq or need_bcs or need_mfi
    need_spine_nii = need_vert_nii or need_vbq
    need_vibe_seg = need_bcs or need_mfi or need_torso
    need_roi = need_mfi or need_torso
    need_vibe_wf = need_mfi

    if not (
        need_cobb
        or need_ivd
        or need_vert
        or need_vbq
        or need_bcs
        or need_mfi
        or need_torso
        or need_curvature
        or need_pelvic
        or need_veridah
    ):
        if _merge_endplate_angles(out, Path(poi_out)) or save:
            out["_provenance"] = _build_provenance(file_dict, poi_out, out.get("_provenance"))
            save_json(final_out, out)
        return out

    logger.on_debug("load nii", t2w_bf.get("sub"))
    t2w = to_nii(file_dict["t2w"]) if need_t2w or need_cobb else None
    vibe_water = to_nii(file_dict["vibe_part-water"], False) if need_vibe_wf else None
    vibe_fat = to_nii(file_dict["vibe_part-fat"], False) if need_vibe_wf else None
    vert = to_nii(file_dict["vert"], True) if need_vert_nii else None
    spine = to_nii(file_dict["spine"], True) if need_spine_nii else None
    vibe_seg = to_nii(file_dict["vibeseg100"], True) if need_vibe_seg else None
    roi = to_nii(file_dict["roi"], True) if need_roi else None
    height_m = file_dict.get("height_m")

    poi = None
    if need_poi:
        logger.on_debug("calc_poi_from_subreg_vert")
        if _poi_is_stale_wrt_stat(final_out, out):
            Path(poi_out).unlink(missing_ok=True)
        if poi_out.exists():
            poi = POI.load(poi_out)
        else:
            poi = calc_poi_from_subreg_vert(
                vert,
                spine,
                subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior, Location.Endplate, Location.Vertebra_Disc],
                buffer_file=poi_out,
                save_buffer_file=True,
            )
    if need_cobb:
        try:
            project_2D = False
            threshold_deg = 10
            logger.on_debug("cobb")
            cobb_val, curv, _ = plot_cobb_and_lordosis_and_kyphosis(
                cobb_jpg_out, poi, file_dict["t2w"], file_dict["vert"], project_2D=project_2D, threshold_deg=threshold_deg
            )
            out["cobb"] = cobb_val
            out["curv"] = curv
            out["project_2D"] = project_2D
            out["min_coop_angle"] = threshold_deg
        except Exception:
            logger.on_fail("error catchted")
            logger.print_error()

    if need_ivd:
        try:
            logger.on_debug("measure_ivd_and_vertebra_geometry (ivd)")
            out["ivd_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, buffer_poi=poi_out, structure_label=100)
        except Exception:
            logger.on_fail("error catchted")
            logger.print_error()
    if need_vert:
        try:
            logger.on_debug("measure_ivd_and_vertebra_geometry (vert)")
            out["vert_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, buffer_poi=poi_out, structure_label=0)
        except Exception:
            logger.on_fail("error catchted")
            logger.print_error()
    if need_vbq:
        logger.on_debug("VBQ_score")
        out["VBQ_score"] = VBQ_score(t2w, vert, spine, full_cord=True)

    if need_bcs:
        logger.on_debug("body_composition_score")
        out["body_composition_score"] = body_composition_score(vibe_seg, vert, spine, dataset_id=100, height_m=height_m)
        if len(out["body_composition_score"]) == 0:
            logger.on_warning("body_composition_score returned empty (no vertebrae from configured regions present)")
    if need_mfi:
        logger.on_debug("muscle_fat_infiltration")
        out["muscle_fat_infiltration"] = muscle_fat_infiltration(vibe_water, vibe_fat, vibe_seg, vert, spine, roi=roi, dataset_id=100)
        out["muscle_fat_infiltration"]["physics_model"] = "2-Point-Dixon"
    if need_torso:
        # torso_vat_sat_muscle_mass returns (results_dict, body_comp_nii). Keep
        # only the serializable results dict so the whole json stays writable.
        logger.on_debug("torso_vat_sat_muscle_mass")
        torso_results, _body_comp = torso_vat_sat_muscle_mass(vibe_seg, roi, dataset_id=100)
        out["torso_vat_sat_muscle_mass"] = torso_results

    if need_curvature and poi is not None:
        try:
            logger.on_debug("curvature metrics")
            out["sva"] = compute_sva(poi)
            out["coronal_balance"] = compute_coronal_balance(poi)
            out["axial_rotation"] = compute_axial_rotation(poi)
            out["segmental_endplate_angles"] = compute_segmental_endplate_angles(poi)
            out["curvature_profile"] = compute_curvature_profile(poi)
            out["multi_cobb"] = compute_multi_cobb(poi)
        except Exception:
            logger.on_fail("curvature error caught")
            logger.print_error()

    if need_veridah and poi is not None:
        try:
            from TPTBox.spine.spinestats.veridah_angles import compute_veridah_variants, plot_veridah_variants

            logger.on_debug("veridah variants")
            out["curv_veridah"] = compute_veridah_variants(poi, file_dict.get("veridah"))
            if file_dict.get("veridah") is not None and file_dict.get("vert") is not None:
                veridah_jpg_out = t2w_bf.get_changed_path(
                    "jpg", "snp", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "cobb-veridah"}
                )
                plot_veridah_variants(
                    veridah_jpg_out,
                    poi,
                    file_dict["t2w"],
                    file_dict["vert"],
                    file_dict.get("veridah"),
                )
        except Exception:
            logger.on_fail("veridah variants error caught")
            logger.print_error()

    # Merge wedge metrics directly into the per-label vert_geometry / ivd_geometry
    # entries so they land in per_vertebra.xlsx / per_ivd.xlsx automatically.
    for geom_key in ("vert_geometry", "ivd_geometry"):
        geom = out.get(geom_key)
        if not isinstance(geom, dict):
            continue
        try:
            geom_int_keys = {int(k): v for k, v in geom.items()}
            wedge = compute_wedge_metrics(geom_int_keys)
            for label, w in wedge.items():
                target = geom.get(str(label)) or geom.get(label)
                if isinstance(target, dict):
                    for k, v in w.items():
                        target.setdefault(k, v)
        except Exception:
            logger.on_fail(f"wedge merge failed for {geom_key}")
            logger.print_error()

    # Also fold per-vertebra dicts (axial_rotation, endplate_internal_angle) into
    # vert_geometry entries and per-IVD segmental angles into ivd_geometry, so they
    # land in per_vertebra.xlsx / per_ivd.xlsx rather than exploding per_subject
    # into dozens of extra columns.
    _merge_per_vertebra_metrics(out)

    if need_pelvic:
        try:
            from TPTBox.spine.spinestats.pelvic_parameters import resolve_fullbody_poi_path

            lumbar_ll = None
            curv = out.get("curv")
            if isinstance(curv, dict):
                lumbar_ll = curv.get("lumbar_lordosis")
            # Prefer an explicit path from file_dict (get_nako_paths sets one);
            # fall back to resolving from (dataset, id) so subjects streamed from
            # loop_over_repaired_nako (which doesn't add fullbody_poi) still work.
            fb = file_dict.get("fullbody_poi")
            if fb is None:
                sub_id = file_dict.get("id")
                ds = file_dict.get("dataset", DATASET_ROOT)
                if sub_id is not None:
                    fb = resolve_fullbody_poi_path(ds, str(sub_id))
            out["pelvic_parameters"] = compute_pelvic_parameters(fb, lumbar_lordosis_deg=lumbar_ll)
        except Exception:
            logger.on_fail("pelvic_parameters error caught")
            logger.print_error()

    _merge_endplate_angles(out, Path(poi_out))
    out["_provenance"] = _build_provenance(file_dict, poi_out, out.get("_provenance"))
    out.pop("_version", None)  # TODO can beremoved
    logger.on_save("save", final_out.name)
    save_json(final_out, out)
    return out


def _read_endplate_internal_angles(poi_json_path: Path) -> dict[str, Any]:
    """Read the ``endplate_internal_angle`` dict from a POI json (if present).

    The POI json is a list; the first element is the metadata dict where the
    endplate-angle map (vertebra name -> angle in degrees) lives.
    """
    if not poi_json_path.exists():
        return {}
    try:
        data = load_json(poi_json_path)
    except Exception:
        return {}
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and isinstance(entry.get("endplate_internal_angle"), dict):
                return entry["endplate_internal_angle"]
        return {}
    if isinstance(data, dict) and isinstance(data.get("endplate_internal_angle"), dict):
        return data["endplate_internal_angle"]
    return {}


def _merge_per_vertebra_metrics(out: dict[str, Any]) -> None:
    """Fold per-vertebra top-level dicts into vert_geometry / ivd_geometry entries.

    Moves values from:
      - ``axial_rotation``           {vertebra_name: deg}  -> vert_geometry[label]["axial_rotation_deg"]
      - ``endplate_internal_angle``  {vertebra_name: deg}  -> vert_geometry[label]["endplate_internal_angle_deg"]
      - ``segmental_endplate_angles`` {"V1-V2": deg}       -> ivd_geometry[100+V1_label]["segmental_endplate_angle_deg"]

    Non-destructive on the top-level dicts (kept for backward reads), but
    the collector will exclude these keys from per_subject.xlsx.
    """
    from TPTBox.core.vert_constants import Vertebra_Instance

    def _name_to_label(n: str) -> int | None:
        try:
            return Vertebra_Instance[n].value
        except KeyError:
            return None

    def _find(geom: dict, label: int) -> dict | None:
        return geom.get(str(label)) or geom.get(label)

    vg = out.get("vert_geometry")
    if isinstance(vg, dict):
        for src_key, dst_key in (
            ("axial_rotation", "axial_rotation_deg"),
            ("endplate_internal_angle", "endplate_internal_angle_deg"),
        ):
            src = out.get(src_key)
            if not isinstance(src, dict):
                continue
            for name, val in src.items():
                lab = _name_to_label(str(name))
                if lab is None:
                    continue
                target = _find(vg, lab)
                if isinstance(target, dict):
                    target.setdefault(dst_key, val)

    ig = out.get("ivd_geometry")
    if isinstance(ig, dict):
        seg = out.get("segmental_endplate_angles")
        if isinstance(seg, dict):
            for pair, val in seg.items():
                upper = str(pair).split("-", 1)[0]
                lab = _name_to_label(upper)
                if lab is None:
                    continue
                target = _find(ig, 100 + lab)
                if isinstance(target, dict):
                    target.setdefault("segmental_endplate_angle_deg", val)


def _merge_endplate_angles(out: dict[str, Any], poi_json_path: Path) -> bool:
    """Attach the POI's per-vertebra endplate_internal_angle to ``out``.

    Adds a top-level ``endplate_internal_angle`` (vertebra-name -> angle) and,
    for every entry in ``vert_geometry``, injects the matching angle as
    ``endplate_internal_angle`` so it shows up in per-vertebra Excel rows.
    Returns True iff ``out`` was modified.
    """
    if out.get("endplate_internal_angle") is not None:
        return False
    angles = _read_endplate_internal_angles(poi_json_path)
    if not angles:
        return False
    from TPTBox.core.vert_constants import Vertebra_Instance

    changed = False
    if out.get("endplate_internal_angle") != angles:
        out["endplate_internal_angle"] = angles
        changed = True
    vg = out.get("vert_geometry")
    if isinstance(vg, dict):
        for label, metrics in vg.items():
            if not isinstance(metrics, dict):
                continue
            try:
                vname = Vertebra_Instance(int(label)).name
            except Exception:
                continue
            angle = angles.get(vname)
            if angle is None:
                continue
            if metrics.get("endplate_internal_angle") != angle:
                metrics["endplate_internal_angle"] = angle
                changed = True
    return changed


# ---------------------------------------------------------------------------
# Excel collector (parallel, producer/consumer)
# ---------------------------------------------------------------------------


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    """Flatten nested dicts into dotted keys (leaves = scalars/None)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(new_key, v, out)
    else:
        # Lists / tuples / NII placeholders end up here as-is; the writer
        # will drop non-scalar values so the sheet stays clean.
        out[prefix] = obj


def _rows_from_json(subject_id: str, data: dict) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one subject's json into (per-subject row, per-vertebra rows, per-ivd rows).

    Per-subject row: everything except the per-label geometry dicts,
    flattened to dotted keys.
    Per-vertebra rows: one row per label in ``vert_geometry``.
    Per-ivd rows: one row per label in ``ivd_geometry``.
    Split by source so each output stays well below Excel's per-sheet
    row limit (1_048_576).
    """
    per_subject: dict[str, Any] = {"subject": subject_id}
    # Exclude per-label geometry dicts (their rows live in per_vertebra / per_ivd)
    # and per-vertebra dicts that were already merged into vert_geometry / ivd_geometry.
    _PER_SUBJECT_EXCLUDE = (
        "ivd_geometry",
        "vert_geometry",
        "axial_rotation",
        "endplate_internal_angle",
        "segmental_endplate_angles",
        "_provenance",
    )
    subject_view = {k: v for k, v in data.items() if k not in _PER_SUBJECT_EXCLUDE}
    _flatten("", subject_view, per_subject)

    per_vert: list[dict[str, Any]] = []
    per_ivd: list[dict[str, Any]] = []
    for source_key, sink in (("vert_geometry", per_vert), ("ivd_geometry", per_ivd)):
        section = data.get(source_key) or {}
        if not isinstance(section, dict):
            continue
        for label, metrics in section.items():
            if not isinstance(metrics, dict):
                continue
            row: dict[str, Any] = {"subject": subject_id, "label": label}
            row.update(metrics)
            sink.append(row)
    return per_subject, per_vert, per_ivd


def _collector_worker(
    task_q: mp.Queue,
    out_folder: Path,
    per_subject_name: str,
    per_vertebra_name: str,
    per_ivd_name: str,
    flush_every: int,
) -> None:
    import pandas as pd  # local import so the main process starts fast

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    subject_rows: list[dict[str, Any]] = []
    vertebra_rows: list[dict[str, Any]] = []
    ivd_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    log_path = out_folder / "excel_collector.log"

    def _log(msg: str) -> None:
        try:
            with log_path.open("a") as f:
                from datetime import datetime as _dt

                f.write(f"[{_dt.now().isoformat(timespec='seconds')}] {msg}\n")
        except Exception:
            pass

    def _write(df_rows: list[dict[str, Any]], name: str) -> None:
        if not df_rows:
            return
        target = out_folder / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        # xlsxwriter is ~5-10x faster than openpyxl for wide sheets; fall back if unavailable.
        engine: str | None = "xlsxwriter"
        try:
            import xlsxwriter  # noqa: F401
        except ImportError:
            engine = None
        _log(f"writing {name}: rows={len(df_rows)} engine={engine or 'openpyxl'}")
        try:
            pd.DataFrame(df_rows).to_excel(tmp, index=False, engine=engine)
            tmp.replace(target)
            _log(f"  {name} done: {target.stat().st_size} bytes")
        except Exception as e:
            _log(f"  {name} FAILED: {type(e).__name__}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _flush(final: bool) -> None:
        # All three tables are now written only at shutdown. The mid-run
        # per_subject flush was killing the daemon on large runs (silent
        # xlsxwriter/openpyxl crash during full-DataFrame rewrites), so
        # nothing is written mid-run — the log below still emits a heartbeat
        # every ``flush_every`` subjects so progress stays visible.
        if not final:
            return
        _write(subject_rows, per_subject_name)
        _write(vertebra_rows, per_vertebra_name)
        _write(ivd_rows, per_ivd_name)

    _log(f"collector started; flush_every={flush_every} (heartbeat only; all writes at shutdown)")
    while True:
        try:
            item = task_q.get(timeout=1.0)
        except _queue.Empty:
            continue
        if item is None:
            _log(f"final flush triggered; seen={len(seen)} vertebra_rows={len(vertebra_rows)} ivd_rows={len(ivd_rows)}")
            _flush(final=True)
            _log("collector exiting")
            return
        subject_id, json_path = item
        if subject_id in seen:
            continue
        try:
            data = load_json(Path(json_path))
        except Exception:
            continue
        per_subj, per_vert, per_ivd = _rows_from_json(str(subject_id), data)
        subject_rows.append(per_subj)
        vertebra_rows.extend(per_vert)
        ivd_rows.extend(per_ivd)
        seen.add(subject_id)
        if flush_every and len(seen) % flush_every == 0:
            _log(f"heartbeat: seen={len(seen)} vertebra_rows={len(vertebra_rows)} ivd_rows={len(ivd_rows)}")


class ExcelCollector:
    """Background process that turns subject jsons into Excel summaries.

    Usage::

        collector = ExcelCollector(out_folder="/tmp/nako_summary")
        collector.start()
        for nako_id in ids:
            f = get_nako_paths(nako_id)
            data = run_all(f)
            collector.submit(nako_id, final_json_path_for(f))
        collector.close()  # flushes and joins
    """

    def __init__(
        self,
        out_folder: str | Path,
        per_subject_name: str = "per_subject.xlsx",
        per_vertebra_name: str = "per_vertebra.xlsx",
        per_ivd_name: str = "per_ivd.xlsx",
        flush_every: int = 200,
    ) -> None:
        self.out_folder = Path(out_folder)
        self.per_subject_name = per_subject_name
        self.per_vertebra_name = per_vertebra_name
        self.per_ivd_name = per_ivd_name
        self.flush_every = flush_every
        self._queue: mp.Queue = mp.Queue()
        self._proc: mp.Process | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = mp.Process(
            target=_collector_worker,
            args=(
                self._queue,
                self.out_folder,
                self.per_subject_name,
                self.per_vertebra_name,
                self.per_ivd_name,
                self.flush_every,
            ),
            daemon=True,
        )
        self._proc.start()

    def submit(self, subject_id: str, json_path: str | Path) -> None:
        if self._proc is None:
            raise RuntimeError("ExcelCollector not started")
        self._queue.put((str(subject_id), str(json_path)))

    def close(self, join_timeout: float = 1800.0) -> None:
        """Signal the daemon to flush + exit and wait up to ``join_timeout`` seconds.

        The final flush writes ``per_vertebra.xlsx`` and ``per_ivd.xlsx``
        from scratch; with 30k subjects that can take 5-15 minutes per
        file. The default timeout is generous (30 min) so the daemon
        has enough time to finish the shutdown flush. Progress is
        logged to ``<out_folder>/excel_collector.log``.
        """
        if self._proc is None:
            return
        self._queue.put(None)
        self._proc.join(timeout=join_timeout)
        self._proc = None


def _final_json_path(file_dict: dict) -> Path:
    """Recreate the json path run_all writes to, without re-running it."""
    t2w_bf = BIDS_FILE(file_dict["t2w"], file_dict["dataset"]) if not isinstance(file_dict["t2w"], BIDS_FILE) else file_dict["t2w"]
    return Path(
        t2w_bf.get_changed_path("json", "stat", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "all"})
    )


# Ordered list of required inputs for run_all. Order matters: for the
# missing-file report each subject is attributed to the FIRST missing
# key in this list, so a subject with several gaps is still counted once.
REQUIRED_INPUT_KEYS: tuple[str, ...] = (
    "t2w",
    "vibe_part-water",
    "vibe_part-fat",
    "vert",
    "spine",
    "vibeseg100",
    "roi",
)


def _first_missing_input(file_dict: dict) -> str | None:
    """Return the first REQUIRED_INPUT_KEYS entry not present/on disk, else None."""
    for k in REQUIRED_INPUT_KEYS:
        v = file_dict.get(k)
        if v is None:
            return k
        p = v.file["nii.gz"] if isinstance(v, BIDS_FILE) else Path(v)
        if not Path(p).exists():
            return k
    return None


def _run_one(args: tuple[dict, bool, bool]) -> tuple[str, str | None, dict]:
    """Worker: run_all for one subject; returns (subject_id, missing_key_or_None)."""
    f, override, do_not_update = args
    sub_id = str(f.get("id"))
    missing = _first_missing_input(f)
    if missing is not None:
        return sub_id, missing, f

    try:
        run_all(f, override=override, do_not_update=do_not_update)
    except Exception as e:
        logger.on_fail(f"run_all failed for {sub_id}: {e}")
        logger.print_error()
        return sub_id, f"error:{type(e).__name__}, {str(e)!s}", f
    return sub_id, None, f


if __name__ == "__main__":
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd

    from TPTBox import No_Logger

    log = No_Logger()
    os.nice(20)
    OUT_FOLDER = Path("/DATA/NAS/ongoing_projects/robert/test/NAKO-stats")
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    N_CPUS = 1  # set >1 to parallelize
    OVERRIDE = False
    aggregate = False
    do_not_update = False
    test = False
    collector: ExcelCollector | None = None
    if aggregate:
        collector = ExcelCollector(out_folder=OUT_FOLDER)
        collector.start()
    missing_rows: list[dict[str, str]] = []
    total = 30645
    try:
        if test:
            subjects = loop_over_repaired_nako(test=True)
            total = 15
            aggregate = False
        elif aggregate:
            subjects = loop_over_repaired_nako(test=False, sort=aggregate)
        else:
            # subjects = tqdm(loop_over_repaired_nako(test=False, sort=aggregate), total=30645)
            l = loop_over_repaired_nako(test=False, sort=True)  # aggregate
            total = 15
            subjects = iter([next(l) for _ in range(total)])
            # subjects = l

            # print(f"Run on {total=} random subset")
        if N_CPUS <= 1:
            for f in tqdm(subjects, total=total):
                sub_id, missing, _ = _run_one((f, OVERRIDE, do_not_update))
                if missing is not None:
                    logger.on_fail("missing", list(f.keys()), missing)
                    missing_rows.append({"subject": sub_id, "missing": missing})
                    continue
                if aggregate:
                    collector.submit(sub_id, _final_json_path(f))
        else:
            from itertools import islice

            with ProcessPoolExecutor(max_workers=N_CPUS, max_tasks_per_child=100) as ex:
                batch_size = 100
                l = tqdm(total=total)
                while True:
                    gc.collect()
                    futs = [ex.submit(_run_one, (f, OVERRIDE, do_not_update)) for f in list(islice(subjects, batch_size))]

                    if not futs:
                        break
                    for fut in as_completed(futs):
                        l.update(1)
                        sub_id, missing, f = fut.result()
                        if missing is not None:
                            logger.on_fail("missing", (sub_id), missing)
                            missing_rows.append({"subject": sub_id, "missing": missing})
                            continue
                        if aggregate:
                            collector.submit(sub_id, _final_json_path(f))

    finally:
        if aggregate and collector is not None:
            collector.close()
            if missing_rows:
                pd.DataFrame(missing_rows).to_excel(OUT_FOLDER / "missing_inputs.xlsx", index=False)
            # Auto-generate the QC report next to the aggregated tables.
            try:
                from TPTBox.spine.spinestats._qc_report import build_qc_report

                build_qc_report(OUT_FOLDER)
            except Exception as e:  # noqa: BLE001
                logger.on_fail(f"qc_report generation failed: {type(e).__name__}: {e}")
                logger.print_error()
