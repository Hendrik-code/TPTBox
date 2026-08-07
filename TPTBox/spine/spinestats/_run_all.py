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

import multiprocessing as mp
import queue as _queue
from pathlib import Path
from typing import Any

from TPTBox.core.bids_files import BIDS_FILE
from TPTBox.core.dicom.dicom2nii_utils import load_json
from TPTBox.core.internal.nii_help import save_json
from TPTBox.core.nii_wrapper import to_nii

DATASET_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako")

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
    return [Path(file_dict[k]) for k in keys if file_dict.get(k) is not None]


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


def run_all(file_dict, cobb: bool = False, override: bool = False) -> dict[str, Any]:
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

    t2w_bf = BIDS_FILE(file_dict["t2w"], file_dict["dataset"])
    poi_out = t2w_bf.get_changed_path(
        "json",
        "poi",
        "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2",
        info={"seg": "vert", "mod": "T2w", "desc": "vert-rotation-new"},
    )
    cobb_jpg_out = t2w_bf.get_changed_path(
        "jpg", "snp", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "cobb"}
    )
    final_out = t2w_bf.get_changed_path(
        "json", "stat", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "all"}
    )
    final_out = Path(final_out)

    required = REQUIRED_MAIN_KEYS + (("cobb", "curv") if cobb else ())
    seg_files = _segmentation_inputs(file_dict)

    if not override:
        valid, cached = _is_cache_valid(final_out, seg_files, required)
        if valid and cached is not None:
            return cached

    t2w = to_nii(file_dict["t2w"])
    vibe_water = to_nii(file_dict["vibe-water"], False)
    vibe_fat = to_nii(file_dict["vibe-fat"], False)
    vert = to_nii(file_dict["vert"], True)
    spine = to_nii(file_dict["spine"], True)
    vibe_seg = to_nii(file_dict["vibeseg100"], True)
    roi = to_nii(file_dict["roi"], True)
    height_m = file_dict.get("height_m")
    poi = calc_poi_from_subreg_vert(
        vert,
        spine,
        subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior, Location.Endplate, Location.Vertebra_Disc],
        buffer_file=poi_out,
        save_buffer_file=True,
    )
    out: dict[str, Any] = {}
    if cobb:
        cobb_val, curv, _ = plot_cobb_and_lordosis_and_kyphosis(
            cobb_jpg_out, poi, file_dict["t2w"], file_dict["vert"], project_2D=False
        )
        out["cobb"] = cobb_val
        out["curv"] = curv

    out["ivd_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, structure_label=100)
    out["vert_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, structure_label=50)

    out["VBQ_score"] = VBQ_score(t2w, vert, spine)
    out["body_composition_score"] = body_composition_score(vibe_seg, vert, spine, dataset_id=100, height_m=height_m)
    out["muscle_fat_infiltration"] = muscle_fat_infiltration(vibe_water, vibe_fat, vibe_seg, vert, spine, roi=roi, dataset_id=100)
    # torso_vat_sat_muscle_mass returns (results_dict, body_comp_nii). Keep
    # only the serializable results dict so the whole json stays writable.
    torso_results, _body_comp = torso_vat_sat_muscle_mass(vibe_seg, roi, dataset_id=100)
    out["torso_vat_sat_muscle_mass"] = torso_results
    save_json(final_out, out)
    return out


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
        flush_every: int = 25,
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
    t2w_bf = BIDS_FILE(file_dict["t2w"], file_dict["dataset"])
    return Path(
        t2w_bf.get_changed_path(
            "json", "stat", "derivatives_spine_inference_162_sacrumfix_subregionmeasures-v2", info={"seg": "all"}
        )
    )


if __name__ == "__main__":
    from TPTBox import No_Logger

    log = No_Logger()

    collector = ExcelCollector(out_folder="/tmp/nako_summary")
    collector.start()
    try:
        for nako_id in ["100000"]:
            f = get_nako_paths(nako_id)
            run_all(f)
            collector.submit(nako_id, _final_json_path(f))
    finally:
        collector.close()
