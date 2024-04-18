# noqa: INP001
"""
This file contains functions that overrides parts of the dicom2nifti library from icometrix

Author: Malek El Husseini; Robert Graf
Date: 11/01/2022
"""

import csv
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pydicom
from dicom2nifti import patch_pydicom_encodings, settings

patch_pydicom_encodings.apply()


logger = logging.getLogger(__name__)


def get_json(data):
    """
    takes a dictionary returned from pydicom.dataset.to_json_dict and rearranges it to be
    compatible with json encoding
    """
    data1 = {}
    for key, value in data.items():
        try:
            k = pydicom.datadict.dictionary_keyword(key)  # type: ignore
        except Exception:
            continue
        if "Value" in value:
            if isinstance(value["Value"], list) and len(value["Value"]) > 0:
                if isinstance(value["Value"][0], dict):
                    data1[k] = [get_json(value["Value"][i]) for i in range(len(value["Value"]))]
                elif len(data[key]["Value"]) == 1:
                    data1[k] = data[key]["Value"][0]
                else:
                    data1[k] = data[key]["Value"]
            else:
                data1[k] = data[key]["Value"]
    return data1


def test_name_conflict(json_ob, file):
    if Path(file).exists():
        js = load_json(file)
        return js != json_ob
    return False


def save_json(json_ob, file):
    """
    recieves a json object and a path and saves the object as a json file
    """

    def convert(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError

    if test_name_conflict(json_ob, file):
        raise FileExistsError(file)
    if Path(file).exists():
        return True
    with open(file, "w") as file:
        json.dump(json_ob, file, indent=4, default=convert)
    return False


def load_json(file):
    with open(file) as f:
        return json.load(f)


def create_simplified_json(header, filename):
    """
    creates and save a json file given a pydicom dicom header and a filename path
    """
    py_dataset = deepcopy(header)
    py_dataset.PixelData = None
    py_dict = py_dataset.to_json_dict()
    simp_json = get_json(py_dict)
    fname = filename + ".json"
    save_json(simp_json, fname)


def create_filename_path(dcm_series, nifti_dir, prefix="", dcm_num=0):
    """
    creates the paths for saving the generated niftis and return a BIDS structured filename
    """
    modality_path = "/DATA/robert/BS/bonescreen-offline-notebook-robert/robert_dcm2nii/mr-modalities.csv"
    with open(modality_path, newline="") as csv_file:
        modalities = list(csv.reader(csv_file))

    patient_dir_name = prefix + str(dcm_series.PatientID)

    main_dir = Path(nifti_dir, patient_dir_name)

    session = "ses-" + dcm_series.StudyDate
    study_dir = Path(main_dir, session)

    if not Path(study_dir).is_dir():
        os.umask(0)
        Path(study_dir).mkdir(exist_ok=True, mode=777)

    series_no = str(dcm_series.SeriesNumber) if not dcm_num else str(dcm_num)

    contrast = dcm_series.ContrastBolusAgent if "ContrastBolusAgent" in dcm_series else ""
    if dcm_series.Modality.lower() == "mr":
        modality = "mr"
        to_look = str(dcm_series.ImageType).lower() + " " + str(dcm_series.SeriesDescription).lower()
        for sequence_ids in modalities:
            if any(sequ_id.lower() in to_look for sequ_id in sequence_ids[1:]):
                modality = sequence_ids[0]
            if modality == "t1":
                if "km" in to_look:
                    modality = "t1c"
                if len(contrast) > 5:
                    modality = "t1c"
        filename = (
            "sub-"
            + prefix
            + dcm_series.PatientID
            + "_ses-"
            + dcm_series.StudyDate
            + "_sequ-"
            + series_no
            + f"_{modality}".replace(".", "_")
        )
    else:
        filename = (
            "sub-"
            + prefix
            + dcm_series.PatientID
            + "_ses-"
            + dcm_series.StudyDate
            + "_sequ-"
            + series_no
            + "_desc-"
            + dcm_series.get_item((0x0008, 0x0008))[2]
            + f"_{dcm_series.Modality.lower()}"
        )

    return study_dir, filename


settings.disable_validate_slice_increment()
