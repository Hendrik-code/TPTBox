import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from examples.dicom_select.a03_totalspineseg import black_list
from TPTBox import BIDS_Family, BIDS_Global_info, Print_Logger, calc_poi_from_subreg_vert, to_nii
from TPTBox.core.bids_constants import formats
from TPTBox.spine.snapshot2D import snapshot


def update_poi_and_snp(fam: BIDS_Family, f, new_vert=None, override=False, poi_not_younger=True):
    try:
        new_vert = new_vert or fam["msk_seg-vert"][0]
    except KeyError:
        return

    subreg = fam.get("msk_seg-subreg", fam.get("msk_seg-spine"))[0]  # type: ignore
    p = fam[f][0]
    path = p.get_nii_file()
    if path is None:
        return
    snp_paths = p.get_changed_path(
        "png",
        bids_format="snp",
        parent="derivative-spineps",
        info={"seg": "spine", "mod": p.format, "ovl": None},
        non_strict_mode=False,
        make_parent=False,
    )
    poi_path = p.get_changed_path(
        "json",
        bids_format="ctd",
        parent="derivative-spineps",
        info={"seg": "spine", "mod": p.format, "ovl": None},
        non_strict_mode=False,
        make_parent=False,
    )

    if not override and poi_path.exists() and snp_paths.exists() and poi_not_younger:
        ref_mtime = os.path.getmtime(path)
        if os.path.getmtime(poi_path) >= ref_mtime and os.path.getmtime(snp_paths) >= ref_mtime:
            return
    poi = calc_poi_from_subreg_vert(new_vert, subreg, subreg_id=[40, 50])
    poi.save(poi_path)  # type: ignore
    snapshot(p, new_vert, poi_path, subreg, out_path=snp_paths, mode="CT" if f == "ct" else "MRI")


# totalspineseg INPUT_FOLDER/ OUTPUT_FOLDER/ --step1 -k step1_levels step1_vert
def do_update_cdt_via_canada(bgi: BIDS_Global_info, select_xlsx_path: Path):
    if not select_xlsx_path.exists():
        return
    df_select = pd.read_excel(select_xlsx_path, dtype={"FileID": str})
    # exit()
    t = tqdm(bgi.iter_subjects())
    for sub, subj in t:
        t.desc = f"update cdt by {select_xlsx_path.name}; Subject {sub}"
        q = subj.new_query(flatten=True)
        q.filter_dixon_only_inphase()
        q.unflatten()
        for fam in q.loop_dict():
            for f in formats:
                if f in fam:
                    if f in black_list:
                        continue
                    update_poi_and_snp(fam, f, override=False)
        for fam in q.loop_dict():
            try:
                poi_path = fam["ctd_seg-subreg"][0] if "ctd_seg-subreg" in fam else fam["ctd_seg-spine"][0]
            except KeyError:
                poi_path = None
            if poi_path is None:
                continue
            for f in formats:
                if f in fam:
                    p = fam[f][0]
                    selected_row = df_select[df_select["FileID"] == p.BIDS_key]
                    if selected_row.empty:
                        continue
                    # selected_row_dict = selected_row.to_dict(orient="records")[0]

                    update_canada = selected_row["vert_brocken_try_canada"].to_numpy()[0]
                    update_canada = None if pd.isna(update_canada) else (update_canada)  # type: ignore
                    if update_canada is not None:
                        subreg = fam["msk_seg-subreg"][0] if "msk_seg-subreg" in fam else fam["msk_seg-spine"][0]

                        out = subreg.get_changed_path(parent=subreg.parent, info={"seg": "vert", "desc": "canada"})
                        if out.exists():
                            # if "msk_seg-vert" in fam and len(fam["msk_seg-vert"]) != 1:
                            #    print("remap", fam)
                            #    file = fam["msk_seg-vert"][0]
                            #    if file.get("desc") == "canada":
                            #        file = fam["msk_seg-vert"][1]
                            #    file = file.get_nii_file()
                            #    assert file is not None
                            #    file.rename(str(file).replace("vert_msk.", "vert_backup."))
                            #    print("rename", str(file), "to", str(file).replace("vert_msk.", "vert_backup."))
                            #    # exit()

                            continue
                        # vert =
                        poi_path = fam["ctd_seg-subreg"][0] if "ctd_seg-subreg" in fam else fam["ctd_seg-spine"][0]

                        vert_canada = subreg.get_changed_path(
                            bids_format="msk",
                            info={"seg": "vert", "desc": "totalspineseg", "mod": p.bids_format},
                            parent="derivative-canada",
                        )
                        vert_c = to_nii(vert_canada, True)
                        if subreg.get("mod") == "ct":
                            vert_c[vert_c >= 60] = 0
                            vert_c[vert_c >= 61] = 0
                            vert_c[vert_c >= 100] = 0
                        new_vert = subreg.open_nii().clamp(0, 1) * vert_c
                        new_vert.save(subreg.get_changed_path(parent=subreg.parent, info={"seg": "vert", "desc": "canada"}))
                        if "msk_seg-vert" in fam:
                            file = fam["msk_seg-vert"][0].get_nii_file()
                            assert file is not None
                            file.rename(str(file).replace("vert_msk.", "vert_backup."))
                        update_poi_and_snp(fam, f, new_vert, override=True)


if __name__ == "__main__":
    dataset_path = Path("/DATA/NAS/datasets_source/dataset-MRCT-Spine/MR-CT-Spine/dataset-MRCT-Spine")
    parent_in = ["sourcedata2", "derivative-spineps"]
    parent_out = "sourcedata3"
    info_folder = "info"

    dataset_path = Path("/DATA/NAS/datasets_processed/MRI_spine/dataset-SpineGAN")
    dataset_name = Path(dataset_path).name
    parent_in = ["rawdata", "derivative-spineps"]
    t13_vertebra_are_a_lie = True
    # parent_out = "sourcedata3"
    update_cdt_via_reference = False

    select_xlsx = dataset_name + "_select.xlsx"

    select_xlsx_path = (dataset_path) / info_folder / select_xlsx
    # random.seed(42)
    log = Print_Logger()
    bgi = BIDS_Global_info([dataset_path], parents=parent_in)
    # filter_dataset(bgi)
    do_update_cdt_via_canada(bgi, select_xlsx_path)
    ################
