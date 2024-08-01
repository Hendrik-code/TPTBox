import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import dicom2nifti.exceptions
import numpy as np
import pydicom
from dicom2nifti import patch_pydicom_encodings, settings

patch_pydicom_encodings.apply()


class NoImageError(ValueError):
    pass


def clean_dicom_data(dcm_data) -> dict:
    """Remove pixel data and specific inline binary data from DICOM dataset."""
    py_dataset = deepcopy(dcm_data)
    if not hasattr(py_dataset, "PixelData"):
        pass
        # raise NoImageError()
    else:
        del py_dataset.PixelData
    print(type(py_dataset))
    py_dict = py_dataset.to_json_dict(suppress_invalid_tags=True)
    for tag in ["00291010", "00291020"]:
        if tag in py_dict and "InlineBinary" in py_dict[tag]:
            del py_dict[tag]["InlineBinary"]
    return py_dict


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
    with open(file, "w") as file_handel:
        json.dump(json_ob, file_handel, indent=4, default=convert)
    return False


settings.disable_validate_slice_increment()
