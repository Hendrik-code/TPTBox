from __future__ import annotations

import json
import pickle
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import pydicom
from tqdm import tqdm

from TPTBox import BIDS_FILE, NII, BIDS_Global_info, Print_Logger

# source_folder = Path("/DATA/NAS/datasets_source/epi/NAKO/NAKO-732_Nachlieferung_20_25/")
source_folder = Path("/DATA/NAS/datasets_source/epi/NAKO/NAKO_2D_issue/")
source_folders = {
    "mevibe": {
        "rule": "part",
        "eco1-pip1": "ME_vibe_fatquant_pre_Eco1_PIP1",
        "eco4-pop1": "ME_vibe_fatquant_pre_Eco4_POP1",
        "fat-fraction": "ME_vibe_fatquant_pre_Output_FP",
        "water-fraction": "ME_vibe_fatquant_pre_Output_WP",
        "eco3-in1": "ME_vibe_fatquant_pre_Eco3_IN1",
        "fat": "ME_vibe_fatquant_pre_Output_F",
        "water": "ME_vibe_fatquant_pre_Output_W",
        "eco2-opp2": "ME_vibe_fatquant_pre_Eco2_OPP2",
        "eco5-arb1": "ME_vibe_fatquant_pre_Eco5_ARB1",
        "r2s": "ME_vibe_fatquant_pre_Output_R2s_Eff",
        "eco0-opp1": "ME_vibe_fatquant_pre_Eco0_OPP1",
    },
    "T2haste": {"rule": "acq", "ax": "T2_HASTE_TRA_COMPOSED"},
    "vibe": {
        "rule": "part",
        "fat": "3D_GRE_TRA_F",
        "inphase": "3D_GRE_TRA_in",
        "outphase": "3D_GRE_TRA_opp",
        "water": "3D_GRE_TRA_W",
    },
    "T2w": {
        "rule": "chunk",
        "LWS": "III_T2_TSE_SAG_LWS",
        "BWS": "II_T2_TSE_SAG_BWS",
        "HWS": "I_T2_TSE_SAG_HWS",
    },
    "pd": {"rule": "acq", "iso": "PD_FS_SPC_COR"},
}


def __test_nii(path: Path | str | BIDS_FILE):
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        try:
            NII.load(path, True).copy().unique()
        except Exception:
            return False
    return True


def find_all_broken(path: str = "/DATA/NAS/datasets_processed/NAKO/dataset-nako/", parents=None):
    brocken = []
    subj_id = 0
    with open("broken.pkl", "rb") as w:
        brocken = pickle.load(w)
        subj_id = int(brocken[-1].get("sub"))
        print(f"{len(brocken)=}; continue at {subj_id=}")

    if parents is None:
        parents = ["rawdata"]
    bgi = BIDS_Global_info([path], parents=parents)
    for s, subj in tqdm(bgi.enumerate_subjects(sort=True)):
        if int(s) <= subj_id:
            continue
        q = subj.new_query(flatten=True)
        q.filter_filetype("nii.gz")
        for f in q.loop_list():
            if not __test_nii(f):
                print("BROKEN:", f)
                brocken.append(f)
                with open("broken.pkl", "wb") as w:
                    pickle.dump(brocken, w)
                with open("broken2.pkl", "wb") as w:
                    pickle.dump(brocken, w)


source_folder_encrypted = Path("/media/veracrypt1/NAKO-732_MRT/")
source_folder_encrypted_alternative = Path(
    "/run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/datasets_source/epi/NAKO/NAKO-732_Nachlieferung_20_25/"
)


def __test_and_replace(out_folder="/media/data/NAKO/dataset-nako2"):
    from TPTBox.core.dicom.dicom_extract import extract_dicom_folder

    with open("/run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/tools/TPTBox/broken.pkl", "rb") as w:
        brocken = pickle.load(w)
    print(len(brocken))
    for bf in tqdm(brocken):
        bf: BIDS_FILE

        # Localize the zip
        subj = bf.get("sub")
        mod = bf.format
        sub_key = bf.get(source_folders[mod]["rule"])
        # print(source_folders[mod].keys(), sub_key)
        assert sub_key is not None, mod

        # IF MEVIBE Export all
        if mod == "mevibe":
            out_files = {}
            for i in source_folders["mevibe"].values():
                f = source_folder_encrypted / i
                f2 = Path("/NON/NON/ON")
                try:
                    f2 = next((f).glob(f"{subj}*.zip"))
                except StopIteration:
                    try:
                        f2 = next((f).glob(f"{subj}*.zip"))
                    except StopIteration:
                        continue
                if f2.exists():
                    out_files.update(extract_dicom_folder(f2, Path(out_folder), make_subject_chunks=3, verbose=False))
                else:
                    print(f, f.exists())
        else:
            try:
                zip_file = next((source_folder_encrypted / source_folders[mod][sub_key]).glob(f"{subj}*.zip"))
            except StopIteration:
                try:
                    zip_file = next((source_folder_encrypted_alternative / source_folders[mod][sub_key]).glob(f"{subj}*.zip"))
                except StopIteration:
                    print((source_folder_encrypted / source_folders[mod][sub_key]) / (f"{subj}*.zip"))
                    continue
            ## Call the extraction
            out_files = extract_dicom_folder(zip_file, Path(out_folder), make_subject_chunks=3, verbose=False)
        ## -- Testing ---
        # Save over brocken...
        for o in out_files.values():
            if o is not None and not __test_nii(o):
                Print_Logger().on_fail("Still Broken ", out_files)


def clean_dicom_data(dcm_data) -> dict:
    """Remove pixel data and specific inline binary data from DICOM dataset."""
    py_dataset = deepcopy(dcm_data)
    if not hasattr(py_dataset, "PixelData"):
        pass
        # raise NoImageError()
    else:
        del py_dataset.PixelData
    py_dict = py_dataset.to_json_dict(suppress_invalid_tags=True)
    for tag in ["00291010", "00291020"]:
        if tag in py_dict and "InlineBinary" in py_dict[tag]:
            del py_dict[tag]["InlineBinary"]
    py_dict = replace_birthdate_with_age(py_dict)
    return py_dict


def replace_birthdate_with_age(d):
    try:
        # DICOM tags
        BIRTH_TAG = "00100030"  # PatientBirthDate
        STUDY_DATE_TAG = "00080020"  # StudyDate
        AGE_TAG = "00101010"  # PatientAge

        birth_str = d.get(BIRTH_TAG, {}).get("Value", [None])[0]
        study_str = d.get(STUDY_DATE_TAG, {}).get("Value", [None])[0]

        if not birth_str:
            return d  # no birth date, nothing to do

        # Parse birth date safely
        try:
            year = int(birth_str[:4])
            month = int(birth_str[4:6]) if len(birth_str) >= 6 and birth_str[4:6] != "00" else 6
            day = int(birth_str[6:8]) if len(birth_str) == 8 and birth_str[6:8] != "00" else 15
            birth_date = date(year, month, day)
        except Exception:
            return d  # invalid date format, skip

        # Reference date (study date or today)
        try:
            ref_date = date(
                int(study_str[:4]),
                int(study_str[4:6]) if study_str[4:6] != "00" else 6,
                int(study_str[6:8]) if study_str[6:8] != "00" else 15,
            )
        except Exception:
            ref_date = date.today()

        # Compute integer age
        age = ref_date.year - birth_date.year - ((ref_date.month, ref_date.day) < (birth_date.month, birth_date.day))

        # Replace PatientBirthDate with PatientAge
        d.pop(BIRTH_TAG, None)
        d[AGE_TAG] = {
            "vr": "AS",  # Age String
            "Value": [f"{age:03d}Y"],  # DICOM age format (e.g. '034Y')
        }
    except Exception:
        pass
    return d


def get_json_from_dicom(data: list[pydicom.FileDataset] | pydicom.FileDataset):
    if isinstance(data, list):
        data = data[0]
    py_dict = clean_dicom_data(data)

    return _get_json_from_dicom(py_dict)


def _get_json_from_dicom(py_dict: dict):
    """
    takes a dictionary returned from pydicom.dataset.to_json_dict and rearranges it to be
    compatible with json encoding
    """
    data1 = {}
    for key, value in py_dict.items():
        try:
            k = pydicom.datadict.dictionary_keyword(key)  # type: ignore
        except Exception:
            continue
        if "Value" in value:
            if isinstance(value["Value"], list) and len(value["Value"]) > 0:
                if isinstance(value["Value"][0], dict):
                    data1[k] = [_get_json_from_dicom(value["Value"][i]) for i in range(len(value["Value"]))]
                elif len(value["Value"]) == 1:
                    data1[k] = value["Value"][0]
                else:
                    data1[k] = value["Value"]
            else:
                data1[k] = value["Value"]
    return data1


def test_name_conflict(json_ob, file):
    if Path(file).exists():
        with open(file) as f:
            js = json.load(f)
        return js != json_ob
    return False


def save_json(json_ob, file, check_exist=False):
    """
    recieves a json object and a path and saves the object as a json file
    """

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(type(obj))

    if check_exist and test_name_conflict(json_ob, file):
        raise FileExistsError(file)
    if Path(file).exists():
        return True
    with open(file, "w") as file_handel:
        json.dump(json_ob, file_handel, indent=4, default=convert)
    return False


def load_json(file):
    with open(file) as file_handel:
        return json.load(file_handel)


if __name__ == "__main__":
    __test_and_replace()
    # find_all_broken()
    # test_and_replace(
    #    "/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101776/mevibe/sub-101776_sequ-94_acq-ax_part-eco0-opp1_mevibe.nii.gz",
    #    "/DATA/NAS/datasets_processed/NAKO/dataset-nako/",
    # )
    # extract_folder(
    #    Path(
    #        "/DATA/NAS/datasets_source/epi/NAKO/NAKO_2D_issue/ME_vibe_fatquant_pre_Eco0_OPP1/111177_30_ME_vibe_fatquant_pre_Eco0_OPP1.zip"
    #    ),
    #    Path("/DATA/NAS/datasets_source/epi/NAKO/dataset-nako2"),
    # )

# ['python', '/DATA/NAS/tools/spineps/spineps/example/helper_parallel.py', '-i', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101509/mevibe/sub-101509_sequ-77_acq-ax_part-eco0-opp1_mevibe.nii.gz', '-ds',
# '/DATA/NAS/datasets_processed/NAKO/dataset-nako', '-der', 'derivatives_mevibe', '-ms', 'vibe', '-snap', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_mevibe/snap']
# [*] Command called with args: ['python', '/DATA/NAS/tools/spineps/spineps/example/helper_parallel.py', '-i', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101509/mevibe/sub-101509_sequ-77_acq-ax_part-ec
# o0-opp1_mevibe.nii.gz', '-ds', '/DATA/NAS/datasets_processed/NAKO/dataset-nako', '-der', 'derivatives_mevibe', '-ms', 'vibe', '-snap', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_mevibe/snap']
# [*] Fold [0]:  takes free gpu 0
# ['python', '/DATA/NAS/tools/spineps/spineps/example/helper_parallel.py', '-i', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101776/mevibe/sub-101776_sequ-94_acq-ax_part-eco0-opp1_mevibe.nii.gz', '-ds',
# '/DATA/NAS/datasets_processed/NAKO/dataset-nako', '-der', 'derivatives_mevibe', '-ms', 'vibe', '-snap', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_mevibe/snap']
# [*] Command called with args: ['python', '/DATA/NAS/tools/spineps/spineps/example/helper_parallel.py', '-i', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101776/mevibe/sub-101776_sequ-94_acq-ax_part-ec
# o0-opp1_mevibe.nii.gz', '-ds', '/DATA/NAS/datasets_processed/NAKO/dataset-nako', '-der', 'derivatives_mevibe', '-ms', 'vibe', '-snap', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_mevibe/snap']
# [*] Fold [0]:  takes free gpu 0
# ['python', '/DATA/NAS/tools/spineps/spineps/example/helper_parallel.py', '-i', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101765/mevibe/sub-101765_sequ-53_acq-ax_part-eco0-opp1_mevibe.nii.gz', '-ds',
# '/DATA/NAS/datasets_processed/NAKO/dataset-nako', '-der', 'derivatives_mevibe', '-ms', 'vibe', '-snap', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_mevibe/snap']
# [*] Command called with args: ['python', '/DATA/NAS/tools/spineps/spineps/example/helper_parallel.py', '-i', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata/101/101765/mevibe/sub-101765_sequ-53_acq-ax_part-ec
# o0-opp1_mevibe.nii.gz', '-ds', '/DATA/NAS/datasets_processed/NAKO/dataset-nako', '-der', 'derivatives_mevibe', '-ms', 'vibe', '-snap', '/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_mevibe/snap']
