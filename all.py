from pathlib import Path

from TPTBox.core.bids_files import BIDS_FILE
from TPTBox.core.internal.nii_help import save_json
from TPTBox.core.nii_wrapper import to_nii

DATASET_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako")


def get_nako_paths(nako_id: str) -> dict[str, Path | None]:
    """Return a dict with all relevant paths for a given NAKO id.

    Keys:
        t2w_stitched, vibe_stitched, vert, spine, roi, vibeseg100
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
        vibe_stitched = vibe_corr

    # current best T2w spine seg (mirrors qa_spine_shift.get_current_best_T2w_seg)
    search_folders = [
        "derivatives_spine_vert_fixed",
        "derivatives_spine_inference_combination162_148",
    ]
    vert = spine = roi = None
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


def run_all(file_dict, cobb=False):
    from TPTBox import Location, calc_poi_from_subreg_vert
    from TPTBox.spine.spinestats.angles import plot_cobb_and_lordosis_and_kyphosis
    from TPTBox.spine.spinestats.measure_ivd_and_vertebra_geometry import (
        measure_ivd_and_vertebra_geometry,  # structure_label: int = 100 and structure_label: int = 49
    )
    from TPTBox.spine.spinestats.torso_vat_sat import VBQ_score, body_composition_score, muscle_fat_infiltration, torso_vat_sat_muscle_mass
    from TPTBox.spine.spinestats.vertebra_anatomical_widths import compute_all_distances

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

    t2w = to_nii(file_dict["t2w"])
    vert = to_nii(file_dict["vert"], True)
    spine = to_nii(file_dict["spine"], True)
    poi = calc_poi_from_subreg_vert(
        vert,
        spine,
        subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior, Location.Endplate, Location.Vertebra_Disc],
        buffer_file=poi_out,
        save_buffer_file=True,
    )
    out = {}
    # print(poi.centroids)
    if cobb:
        cobb, curv, _ = plot_cobb_and_lordosis_and_kyphosis(cobb_jpg_out, poi, file_dict["t2w"], file_dict["vert"], project_2D=False)
        out["cobb"] = cobb
        out["curv"] = curv

    out["ivd_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, structure_label=100)
    out["vert_geometry"] = measure_ivd_and_vertebra_geometry(t2w, vert, spine, structure_label=50)

    save_json(final_out, out)


nako_id = "100000"

if __name__ == "__main__":
    from TPTBox import No_Logger

    log = No_Logger()
    f = get_nako_paths(nako_id)
    for k, v in f.items():
        log.print(f"{k:20}: {v}") if v.exists() else log.on_warning(f"{k}: {v}")
    json_dict = run_all(f)
    # save json
