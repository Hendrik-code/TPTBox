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

from TPTBox import Print_Logger
from TPTBox.core.bids_files import BIDS_FILE
from TPTBox.core.dicom.dicom2nii_utils import load_json
from TPTBox.core.internal.nii_help import save_json
from TPTBox.core.nii_wrapper import to_nii
from TPTBox.spine.spinestats._load_nako import loop_over_repaired_nako

DATASET_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako")
logger = Print_Logger()
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


def _is_cache_valid(json_path: Path, seg_files: list[Path], required_keys: tuple[str, ...]) -> tuple[bool, dict | None]:
    """Return (valid, loaded_dict).

    Cache is invalid (and needs recompute) if:
      - json does not exist,
      - json is older than any segmentation file,
      - json fails to parse,
      - any required main key is missing from the loaded dict.
    """
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
    return True, data


def run_all(
    file_dict,
    override: bool = False,
    do_not_update=False,
    need_cobb=False,
    need_ivd=False,
    need_vert=False,
    need_vbq=True,
    need_bcs=True,
    need_mfi=True,
    need_torso=True,
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
    from TPTBox.spine.spinestats.measure_ivd_and_vertebra_geometry import (
        measure_ivd_and_vertebra_geometry,  # structure_label: int = 100 and structure_label: int = 49
    )
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

    def _need(*keys: str, compute: bool) -> bool:
        return override or (any(k not in out for k in keys) and compute)

    need_cobb = _need("cobb", "curv", compute=need_cobb)
    need_ivd = _need("ivd_geometry", compute=need_ivd)
    need_vert = _need("vert_geometry", compute=need_vert)
    need_vbq = _need("VBQ_score", compute=need_vbq)
    need_bcs = _need("body_composition_score", compute=need_bcs)
    need_mfi = _need("muscle_fat_infiltration", compute=need_mfi)
    need_torso = _need("torso_vat_sat_muscle_mass", compute=need_torso)
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
    ####
    need_poi = need_cobb or need_ivd or need_vert
    need_t2w = need_ivd or need_vert or need_vbq
    need_vert_nii = need_poi or need_vbq or need_bcs or need_mfi
    need_spine_nii = need_vert_nii or need_vbq
    need_vibe_seg = need_bcs or need_mfi or need_torso
    need_roi = need_mfi or need_torso
    need_vibe_wf = need_mfi

    if not (need_cobb or need_ivd or need_vert or need_vbq or need_bcs or need_mfi or need_torso):
        if _merge_endplate_angles(out, Path(poi_out)) or save:
            save_json(final_out, out)
        return out

    logger.on_debug("load nii")
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
        poi = calc_poi_from_subreg_vert(
            vert,
            spine,
            subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior, Location.Endplate, Location.Vertebra_Disc],
            buffer_file=poi_out,
            save_buffer_file=True,
        )
    if need_cobb:
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

    if need_ivd:
        logger.on_debug("measure_ivd_and_vertebra_geometry (ivd)")
        out["ivd_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, buffer_poi=poi_out, structure_label=100)
    if need_vert:
        logger.on_debug("measure_ivd_and_vertebra_geometry (vert)")
        out["vert_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, buffer_poi=poi_out, structure_label=0)

    if need_vbq:
        logger.on_debug("VBQ_score")
        out["VBQ_score"] = VBQ_score(t2w, vert, spine, full_cord=True)

    if need_bcs:
        logger.on_debug("body_composition_score")
        out["body_composition_score"] = body_composition_score(vibe_seg, vert, spine, dataset_id=100, height_m=height_m)
        assert len(out["body_composition_score"]) != 0
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
    _merge_endplate_angles(out, Path(poi_out))
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


def _rows_from_json(subject_id: str, data: dict) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split one subject's json into (per-subject row, per-vertebra rows).

    Per-subject row: everything except the per-label geometry dicts,
    flattened to dotted keys.
    Per-vertebra rows: one row per label in ``ivd_geometry`` and
    ``vert_geometry`` (source column indicates which).
    """
    per_subject: dict[str, Any] = {"subject": subject_id}
    subject_view = {k: v for k, v in data.items() if k not in ("ivd_geometry", "vert_geometry")}
    _flatten("", subject_view, per_subject)

    per_vert: list[dict[str, Any]] = []
    for source_key in ("vert_geometry", "ivd_geometry"):
        section = data.get(source_key) or {}
        if not isinstance(section, dict):
            continue
        for label, metrics in section.items():
            if not isinstance(metrics, dict):
                continue
            row: dict[str, Any] = {"subject": subject_id, "source": source_key, "label": label}
            row.update(metrics)
            per_vert.append(row)
    return per_subject, per_vert


def _collector_worker(
    task_q: mp.Queue,
    out_folder: Path,
    per_subject_name: str,
    per_vertebra_name: str,
    flush_every: int,
) -> None:
    import pandas as pd  # local import so the main process starts fast

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    subject_rows: list[dict[str, Any]] = []
    vertebra_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _flush() -> None:
        if subject_rows:
            pd.DataFrame(subject_rows).to_excel(out_folder / per_subject_name, index=False)
        if vertebra_rows:
            pd.DataFrame(vertebra_rows).to_excel(out_folder / per_vertebra_name, index=False)

    while True:
        try:
            item = task_q.get(timeout=1.0)
        except _queue.Empty:
            continue
        if item is None:
            _flush()
            return
        subject_id, json_path = item
        if subject_id in seen:
            continue
        try:
            data = load_json(Path(json_path))
        except Exception:
            continue
        per_subj, per_vert = _rows_from_json(str(subject_id), data)
        subject_rows.append(per_subj)
        vertebra_rows.extend(per_vert)
        seen.add(subject_id)
        if flush_every and len(seen) % flush_every == 0:
            _flush()


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
        flush_every: int = 200,
    ) -> None:
        self.out_folder = Path(out_folder)
        self.per_subject_name = per_subject_name
        self.per_vertebra_name = per_vertebra_name
        self.flush_every = flush_every
        self._queue: mp.Queue = mp.Queue()
        self._proc: mp.Process | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = mp.Process(
            target=_collector_worker,
            args=(self._queue, self.out_folder, self.per_subject_name, self.per_vertebra_name, self.flush_every),
            daemon=True,
        )
        self._proc.start()

    def submit(self, subject_id: str, json_path: str | Path) -> None:
        if self._proc is None:
            raise RuntimeError("ExcelCollector not started")
        self._queue.put((str(subject_id), str(json_path)))

    def close(self, join_timeout: float = 60.0) -> None:
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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd

    from TPTBox import No_Logger

    log = No_Logger()

    OUT_FOLDER = Path("/DATA/NAS/ongoing_projects/robert/test/NAKO-stats")
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    N_CPUS = 40  # set >1 to parallelize
    OVERRIDE = False
    aggregate = True
    do_not_update = False
    test = False
    if aggregate:
        collector = ExcelCollector(out_folder=OUT_FOLDER)
        collector.start()
    missing_rows: list[dict[str, str]] = []
    total = 30645
    try:
        if test:
            subjects = loop_over_repaired_nako(test=True)
            total = 10
            aggregate = False
        elif aggregate:
            subjects = loop_over_repaired_nako(test=False, sort=aggregate)
        else:
            # subjects = tqdm(loop_over_repaired_nako(test=False, sort=aggregate), total=30645)
            l = loop_over_repaired_nako(test=False, sort=aggregate)
            total = 1000
            subjects = iter([next(l) for _ in range(total)])

        if N_CPUS <= 1:
            for f in subjects:
                sub_id, missing, _ = _run_one((f, OVERRIDE, do_not_update))
                if missing is not None:
                    logger.on_fail("missing", list(f.keys()), missing)
                    missing_rows.append({"subject": sub_id, "missing": missing})
                    continue
                if aggregate:
                    collector.submit(sub_id, _final_json_path(f))
        else:
            from itertools import islice

            with ProcessPoolExecutor(max_workers=N_CPUS) as ex:
                batch_size = 1000
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
        if aggregate:
            collector.close()
            if missing_rows:
                pd.DataFrame(missing_rows).to_excel(OUT_FOLDER / "missing_inputs.xlsx", index=False)
