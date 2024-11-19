# noqa: INP001
import glob
import os
import pprint
import random
import re
import shutil
import sys
import zipfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import dicom2nifti
import dicom2nifti.exceptions
import dill as pickle
import nibabel.orientations as nio
import numpy as np
import pydicom
from dicom2nifti import common, convert_dicom
from func_timeout import func_timeout  # pip install func_timeout
from func_timeout.exceptions import FunctionTimedOut
from joblib import Parallel, delayed

from TPTBox import BIDS_FILE, Log_Type, No_Logger
from TPTBox.core.bids_constants import formats

sys.path.append(str(Path(__file__).parent))

from TPTBox.core.dicom.dicom2nii_utils import get_json_from_dicom, save_json, test_name_conflict

logger = No_Logger()

dixon_mapping = {
    "f": "fat",
    "w": "water",
    "in": "inphase",
    "opp": "outphase",
    "opp1": "eco0-opp1",
    "pip1": "eco1-pip1",
    "opp2": "eco2-opp2",
    "in1": "eco3-in1",
    "pop1": "eco4-pop1",
    "arb1": "eco5-arb1",
    "fp": "fat-fraction",
    "eff": "r2s",
    "wp": "water-fraction",
}


def get_plane_dicom(dicoms: list[pydicom.FileDataset]) -> str | None:
    """Determines the orientation plane of the NIfTI image along the x, y, or z-axis.

    Returns:
        str: The orientation plane of the image, which can be one of the following:
            - 'ax': Axial plane (along the z-axis).
            - 'cor': Coronal plane (along the y-axis).
            - 'sag': Sagittal plane (along the x-axis).
            - 'iso': Isotropic plane (if the image has equal zoom values along all axes).
    Examples:
        >>> nii = NII(nib.load("my_image.nii.gz"))
        >>> nii.get_plane()
        'ax'
    """
    try:
        sorted_dicoms = common.sort_dicoms(dicoms)
        affine, _ = common.create_affine(sorted_dicoms)
        plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
        axc = np.array(nio.aff2axcodes(affine))
        affine = np.asarray(affine)
        q, p = affine.shape[0] - 1, affine.shape[1] - 1
        # extract the underlying rotation, zoom, shear matrix
        RZS = affine[:q, :p]  # noqa: N806
        zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
        # Zooms can be zero, in which case all elements in the column are zero, and
        # we can leave them as they are
        zooms[zooms == 0] = 1
        zms = np.around(zooms, 1)
        ix_max = np.array(zms == np.amax(zms))
        num_max = np.count_nonzero(ix_max)
        if num_max == 2:
            plane = plane_dict[axc[~ix_max][0]]
        elif num_max == 1:
            plane = plane_dict[axc[ix_max][0]]
        else:
            plane = "iso"
        return plane  # noqa: TRY300
    except Exception:
        return None


def _inc_key(keys, inc=1):
    k = "nameconflict"
    if k not in keys:
        keys[k] = 0
    v = int(keys[k])
    keys[k] = str(v + int(inc))


def generate_general_name(
    mri_format,
    template: dict[str, str | tuple[str, Callable]],
    dcm_data_l: list[pydicom.FileDataset],
    nifti_dir,
    acq=None,
    make_subject_chunks=0,
    root: Path | None = None,
    _increment_id=0,
):
    if acq is None:
        acq = get_plane_dicom(dcm_data_l)
    ### get json
    dcm_data = dcm_data_l[0]
    py_dataset = deepcopy(dcm_data)
    del py_dataset.PixelData  # = None
    py_dict = py_dataset.to_json_dict()
    if "00291010" in py_dict and "InlineBinary" in py_dict["00291010"]:
        del py_dict["00291010"]["InlineBinary"]
    if "00291020" in py_dict and "InlineBinary" in py_dict["00291010"]:
        del py_dict["00291020"]["InlineBinary"]
    simp_json = get_json_from_dicom(py_dict)
    # print(simp_json)
    #
    keys: dict[str, str] = {}
    try:
        keys["sub"] = str(simp_json["PatientID"]).split("_")[0]
    except KeyError:
        assert root is not None
        assert "study" in Path(root).parent.name, "No Patient ID" + str(root)
        keys["sub"] = Path(root).parent.name

    for key, value in template.items():
        if isinstance(value, tuple):
            v, fun = value
            if v == "path":
                v = fun(root)
            elif v in simp_json:
                v = fun(str(simp_json[v]))
            keys[key] = str(v).replace("_", "-")
        elif value.lower() == "acq":
            keys[key] = acq  # type: ignore
        elif value.lower() == "dixon_mapping":
            series_description = str(simp_json["SeriesDescription"]).lower()
            for part in reversed(series_description.split("_")):
                if part in dixon_mapping:
                    keys[key] = dixon_mapping[part]
                    break
            else:
                pass
                # keys[key] = None  # type: ignore
                # raise ValueError(SeriesDescription)
        elif value in simp_json:
            keys[key] = str(simp_json[value]).replace("_", "-")
    if _increment_id != 0:
        _inc_key(keys, inc=_increment_id)

    if make_subject_chunks != 0:
        p = Path(nifti_dir, "rawdata", str(keys["sub"][:make_subject_chunks]), keys["sub"])
    elif "ses" in keys:
        p = Path(nifti_dir, "rawdata", keys["sub"], f"ses-{keys['ses']}")
    else:
        p = Path(nifti_dir, "rawdata", keys["sub"])
    if mri_format not in formats:
        formats.append(mri_format)

    fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_path(
        info=keys,
        file_type="json",
        parent="rawdata",
        make_parent=False,
        additional_folder=mri_format,
        bids_format=mri_format,
        non_strict_mode=True,
    )
    if test_name_conflict(simp_json, fname):
        logger.print("Name conflict inclement a value by one")
        fname = generate_general_name(
            format, template, dcm_data_l, nifti_dir, make_subject_chunks=make_subject_chunks, root=root, _increment_id=_increment_id + 1
        )
    return fname


file_mapping = {
    "formats": {
        ".*t2w?_tse.*": "T2w",
        "t2w?_fse.*": "T2w",
        ".*t1w?_tse.*": "T1w",
        ".*t1w?_vibe_tra.*": "vibe",
        ".*flair.*": "flair",
        ".*stir.*": "STIR",
        ".*dti.*": "DTI",
        ".*dwi.*": "DWI",
        ".*dir.*": "DIR",
        "se": "SE",  # Spine echo
        ".* fir .*": "IR",  # fast inversion recovery
        ".*irfse.*": "IR",  # fast inversion recovery
        "ir_.*": "IR",  # inversion recovery
        ".*mp?ra?ge?.*": "MPRAGE",
        ".*mip.*": "MIP",
        "b0map": "b0map",
        ".*t2.*": "T2w",
        ".*t1.*": "T1w",
        # others
        "shim2d": "mr",
        "3-plane loc": "mr",
        "fgr": "mr",  # Fetal growth restriction (FGR) ????
        "screen save": "mr",
        ".*¶.*": "mr",
        ".*scout": "mr",
        "localizer": "mr",
        ".*pilot.*": "mr",
        ".*2d.*": "mr",
        ".*scno.*": "mr",
        ".*scano.*": "mr",
        "sys2dcard": "mr",
        "3-pl loc gr": "mr",
        re.escape("?") + "*": "mr",
        ".*": "mr",
    },
    # "templates_map": {},  # "T2w": "default", "T1w": "default", "vibe": "dixon_no_sequ"
    "templates_map": {
        "flair": "default",
        "T2w": "default",
        "T1w": "default",
        "DWI": "default",
        "DIR": "default",
        "SWIP": "default",
        "DADC": "default",
        "mr": "default",
        "MPRAGE": "default",
        "Hast": "default",
        "MIP": "default",
        "b0map": "default",
        "DTI": "default",
        "STIR": "default",
        "IR": "default",
        "SE": "default",
    },
    "templates": {
        "default": {
            "sequ": "SeriesNumber",
            "acq": "acq",
        },
        "default_ses": {
            "sequ": "SeriesNumber",
            "ses": "StudyDate",
            "acq": "acq",
        },
        "dixon": {
            "sequ": "SeriesNumber",
            "acq": "acq",
            "part": "dixon_mapping",
            "chunk": ("ProtocolName", lambda x: str(x).split("_")[-1]),
        },
        "dixon_no_sequ": {
            "acq": "acq",
            "part": "dixon_mapping",
            "chunk": ("ProtocolName", lambda x: str(x).split("_")[-1]),
        },
        "u": {
            "sequ": "SeriesNumber",
            "acq": "acq",
            "desc": ("SeriesNumber", lambda x: str(random.randint(0, 100000000))),  # noqa: ARG005
        },
        "by_zip_path": {
            "sequ": "SeriesNumber",
            "acq": "acq",
            "ses": ("path", lambda x: Path(x).parent.name),
            "sub": ("path", lambda x: Path(x).parent.parent.name),
        },
    },
}


def from_dicom_json_to_extracting_nii(  # noqa: C901
    dcm_data_l: list[pydicom.FileDataset],
    nifti_dir,
    dicom_out_path,
    make_subject_chunks=0,
    root: Path | None = None,
):
    dcm_data = dcm_data_l[0]

    py_dataset = deepcopy(dcm_data)
    if not hasattr(py_dataset, "PixelData"):
        return
    del py_dataset.PixelData  # = None
    py_dict = py_dataset.to_json_dict(suppress_invalid_tags=True)
    if "00291010" in py_dict and "InlineBinary" in py_dict["00291010"]:
        del py_dict["00291010"]["InlineBinary"]

    if "00291020" in py_dict and "InlineBinary" in py_dict["00291020"]:
        del py_dict["00291020"]["InlineBinary"]
    simp_json = get_json_from_dicom(py_dict)
    if "StudyDescription" in simp_json and "nako" in str(simp_json["StudyDescription"]).lower():
        keys = {}
        keys["sub"] = str(simp_json["PatientID"]).split("_")[0]
        p = Path(nifti_dir, "rawdata", str(keys["sub"][:3]), keys["sub"])
        logger.print(p, Log_Type.STRANGE)
        # T2w?
        series_description = str(simp_json["SeriesDescription"])
        if "T2_TSE" in series_description:
            mri_format = "T2w"
            assert "SAG" in series_description
            keys["acq"] = "sag"
            keys["chunk"] = series_description.split("_")[-1]
            assert keys["chunk"] in ["LWS", "HWS", "BWS"]
            keys["sequ"] = simp_json["SeriesNumber"]
        elif "3D_GRE_TRA" in series_description:
            mri_format = "vibe"
            keys["acq"] = "ax"
            keys["part"] = dixon_mapping[series_description.split("_")[-1].lower()]
            keys["chunk"] = str(simp_json["ProtocolName"]).split("_")[-1]
        elif "ME_vibe" in series_description:
            mri_format = "mevibe"
            keys["acq"] = "ax"
            keys["part"] = dixon_mapping[series_description.split("_")[-1].lower()]
            keys["sequ"] = simp_json["SeriesNumber"]
        elif "PD" in series_description:
            mri_format = "pd"
            keys["acq"] = "iso"
            keys["sequ"] = simp_json["SeriesNumber"]
        elif "T2_HASTE" in series_description:
            mri_format = "T2haste"
            keys["acq"] = "ax"
            keys["sequ"] = simp_json["SeriesNumber"]
        else:
            pprint.pprint(simp_json)
            raise NotImplementedError(series_description)
        fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_path(
            info=keys, file_type="json", parent="rawdata", make_parent=True, additional_folder=mri_format, bids_format=mri_format
        )
        while test_name_conflict(simp_json, fname):
            logger.print("Name conflict inclement a value by one", keys, Log_Type.FAIL)
            _inc_key(keys)
            fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_path(
                info=keys, file_type="json", parent="rawdata", make_parent=True, additional_folder=mri_format, bids_format=mri_format
            )
    else:
        if len(dcm_data_l) == 1:
            return
        if "SeriesDescription" not in simp_json:
            return
        series_description = str(simp_json["SeriesDescription"]).lower()
        mri_format = None
        #################### Understand sequence by given times ####################
        try:
            a, b = None, None
            if series_description.startswith("fse ") and "/" in series_description:
                # FSE [TR]/[TE] *
                a, b = series_description[4:].split(" ")[0].split("/")
                tr = float(a)
                te = float(b)
                if tr >= 2000 and (te < 150 and te > 80):
                    mri_format = "T2w"
                print(series_description, "Tr", tr, "te", te, "format", mri_format, tr >= 2000)
        except Exception:
            pass
        #################### Understand sequence by given times ####################
        for key in file_mapping["formats"].keys():
            if mri_format is not None:
                break
            regex = re.compile(key)

            if re.match(regex, series_description):
                mri_format = file_mapping["formats"][key]

        if mri_format is None:
            if series_description in file_mapping["formats"]:
                mri_format = file_mapping["formats"][series_description]
            else:
                while True:
                    mri_format = input(
                        f'"{series_description}" ({len(dcm_data_l)}) unknown. Known formats: {formats}; What ist the format? [format/print] '
                    )
                    if mri_format == "print":
                        pprint.pprint(simp_json)
                        continue
                    if mri_format not in formats:
                        formats.append(mri_format)
                    if mri_format in formats:
                        break
                file_mapping["formats"][re.escape(series_description)] = mri_format
        if series_description in file_mapping["templates_map"]:
            template_name = file_mapping["templates_map"][series_description]
            template = file_mapping["templates"][template_name]
        if mri_format in file_mapping["templates_map"]:
            template_name = file_mapping["templates_map"][mri_format]
            template = file_mapping["templates"][template_name]
        else:
            if mri_format not in file_mapping["templates_map"]:
                acq = get_plane_dicom(dcm_data_l)
                for key, template in file_mapping["templates"].items():
                    try:
                        file = generate_general_name(
                            mri_format, template, dcm_data_l, nifti_dir, acq=acq, make_subject_chunks=make_subject_chunks, root=root
                        )
                        print(f"{key}\t:\t", file)
                    except Exception:  # noqa: TRY203
                        raise

            while True:
                template_name = input("pick a template, or add one. (adding a template is only possible in the code): ")
                if template_name in file_mapping["templates"]:
                    template = file_mapping["templates"][template_name]
                    break
            if input(f"use for all {mri_format} (y/n)").lower() == "y":
                file_mapping["templates_map"][mri_format] = template_name
            else:
                file_mapping["templates_map"][mri_format] = series_description
        fname = generate_general_name(mri_format, template, dcm_data_l, nifti_dir, make_subject_chunks=make_subject_chunks, root=root)
        fname.parent.mkdir(exist_ok=True, parents=True)
        # raise NotImplementedError(simp_json)
    logger.print(fname, Log_Type.NEUTRAL)
    # print(simp_json)
    exist = save_json(simp_json, fname)
    nii_path = str(fname).replace(".json", "") + ".nii.gz"

    if exist and Path(nii_path).exists():
        logger.print("already exists:", fname, ltype=Log_Type.STRANGE)
        return

    if isinstance(dicom_out_path, list):
        try:
            convert_dicom.dicom_array_to_nifti(dicom_out_path, nii_path, True)
        except dicom2nifti.exceptions.ConversionValidationError as e:
            if "TOO_FEW_SLICES/LOCALIZER" in e.args[0]:
                return
            if "IMAGE_ORIENTATION_INCONSISTENT" in e.args[0]:
                return
            else:
                raise
    else:
        try:
            func_timeout(10, dicom2nifti.dicom_series_to_nifti, (dicom_out_path, nii_path, True))
        except FunctionTimedOut:
            logger.print_error()
            return False
        except ValueError:
            logger.print_error()
            return False

    # dicom2nifti.dicom_series_to_nifti(dicom_out_path, nii_path, reorient_nifti=True)
    logger.print("Save ", nii_path, Log_Type.SAVE)
    return True


def find_all_files(l: list, curr_path: Path):
    if curr_path.is_dir():
        [find_all_files(l, c) for c in curr_path.iterdir()]
        return l
    else:
        l.append(str(curr_path))
    return l


already_run = []


def extract_folder(  # noqa: C901
    dcm_dirs: Path | list[Path], nifti_dir, del_dicom=False, make_subject_chunks=0, load_settings=False, keep_settings=False
):
    global file_mapping  # noqa: PLW0603

    paths = []
    if not isinstance(dcm_dirs, list):
        dcm_dirs = [dcm_dirs]
    setting_path = Path(args.outfolder, "name_settings.pkl")

    if load_settings and setting_path.exists():
        with open(setting_path, "rb") as f:
            file_mapping = pickle.load(f)
            pprint.pprint(file_mapping)

    for dcm_dir in dcm_dirs:
        if "vera" in str(dcm_dir):
            assert del_dicom is False
        if del_dicom:
            assert "/media/data/NAKO/MRT/NAKO-732_MRT/" in str(dcm_dir)
        find_all_files(paths, dcm_dir)
        # paths += [os.path.join(dcm_dir, x) for x in os.listdir(dcm_dir)]

    for dicom_zip_path in paths:
        if str(dicom_zip_path).endswith(".pkl"):
            continue
        try:
            logger.print("Start", dicom_zip_path) if not str(dicom_zip_path).endswith(".dcm") else None
            # try:
            # Step 1 unzip files
            if str(dicom_zip_path).endswith(".zip"):
                is_temp = True
                zip_dir = Path(dicom_zip_path).parent / "tmp_zip_dir"
                print(dicom_zip_path)
                with zipfile.ZipFile(dicom_zip_path, "r") as zip_ref:
                    dicom_out_path = Path(zip_dir, os.path.basename(dicom_zip_path)[:-4])  # noqa: PTH119
                    zip_ref.extractall(dicom_out_path)
            elif str(dicom_zip_path).endswith(".dcm"):
                is_temp = False
                dicom_out_path = Path(dicom_zip_path).parent
                if dicom_out_path in already_run:
                    continue
                else:
                    already_run.append(dicom_out_path)
            else:
                is_temp = False
                dicom_out_path = dicom_zip_path

            # Step 2 convert
            dcm_data = None
            dicom_files: dict[str, list[pydicom.FileDataset]] = {}
            key = ""
            for a__ in glob.iglob(str(dicom_out_path) + "/**/*", recursive=True):  # noqa: PTH207
                a = Path(a__)
                if a.is_file():
                    try:
                        dcm_data = pydicom.dcmread(a, defer_size="1 KB", force=True)
                        try:
                            typ = str(dcm_data.get_item((0x0008, 0x0008)).value).split("\\")[2]
                        except Exception:
                            typ = ""
                        key = f"{dcm_data.SeriesInstanceUID}_{typ}"
                        if key not in dicom_files:
                            dicom_files[key] = []
                        dicom_files[key].append(dcm_data)

                    except AttributeError:
                        pass

            # easy translation, when only one file in zip
            if len(dicom_files) == 1:
                dcm_data = dicom_files[key]
                succ_all = from_dicom_json_to_extracting_nii(dcm_data, nifti_dir, dicom_out_path, make_subject_chunks, root=dicom_zip_path)

            else:
                succ_all = True
                for key in dicom_files.keys():
                    succ = from_dicom_json_to_extracting_nii(
                        dicom_files[key], nifti_dir, dicom_files[key], make_subject_chunks, root=dicom_zip_path
                    )
                    if not succ:
                        succ_all = succ
            if succ_all and del_dicom:
                logger.print("Delete dicom after translation", Log_Type.WARNING)
                Path(dicom_zip_path).unlink()
            if is_temp:
                try:
                    shutil.rmtree(dicom_out_path)
                except Exception:
                    logger.print_error()
        except Exception:
            try:
                if is_temp:
                    shutil.rmtree(dicom_out_path)
            except Exception:
                pass
            logger.print_error()
        if keep_settings:
            with open(setting_path, "wb") as f:
                pickle.dump(file_mapping, f)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--inputfolder", help="your name", required=True)
    arg_parser.add_argument("-o", "--outfolder", help="your name", required=True)
    arg_parser.add_argument("-n", "--name", default="nako", help="of the dataset")
    arg_parser.add_argument("-c", "--cpus", default=1, type=int, help="number of parallel process")
    # parser.add_argument('--feature', action='store_true')
    arg_parser.add_argument(
        "-nn", "--nonako", help="use this export script for something else or export all nako folder", action="store_true", default=False
    )
    arg_parser.add_argument("-sc", "--subjectchunks", default=3, type=int)
    args = arg_parser.parse_args()
    print(f"args={args}")
    print(f"args.inputfolder={args.inputfolder}")
    print(f"args.outfolder={args.outfolder}")
    print(f"args.name={args.name}")
    p = args.inputfolder
    assert p is not None, "use -i for a existing input folder"
    assert args.inputfolder is not None, "use -i for a existing input folder"
    assert Path(p).exists(), f"{p} does not exist."
    print(args.__dict__.keys())
    if not args.nonako:
        dcm_dir2 = [
            Path(args.inputfolder, i)
            for i in [
                "III_T2_TSE_SAG_LWS",
                "II_T2_TSE_SAG_BWS",
                "I_T2_TSE_SAG_HWS",
                "T2_HASTE_TRA_COMPOSED",
                "PD_FS_SPC_COR",
                "3D_GRE_TRA_F",
                "3D_GRE_TRA_in",
                "3D_GRE_TRA_opp",
                "3D_GRE_TRA_W",
                "ME_vibe_fatquant_pre_Eco1_PIP1",
                "ME_vibe_fatquant_pre_Eco4_POP1",
                "ME_vibe_fatquant_pre_Output_FP",
                "ME_vibe_fatquant_pre_Output_WP",
                "ME_vibe_fatquant_pre_Eco3_IN1",
                "ME_vibe_fatquant_pre_Output_F",
                "ME_vibe_fatquant_pre_Output_W",
                "ME_vibe_fatquant_pre_Eco2_OPP2",
                "ME_vibe_fatquant_pre_Eco5_ARB1",
                "ME_vibe_fatquant_pre_Output_R2s_Eff",
                "ME_vibe_fatquant_pre_Eco0_OPP1",
            ]
        ]
        for i in dcm_dir2:
            logger.override_prefix = "exist" if i.exists() else "missing"
            logger.print(i, ltype=Log_Type.OK if i.exists() else Log_Type.FAIL)
        dcm_dir = [c for c in dcm_dir2 if c.exists()]

        if len(dcm_dir2) != len(dcm_dir) and input("some folders are missing! Continue anyway? (y/n)") != "y":
            sys.exit()
    else:
        dcm_dir = list(Path(args.inputfolder).iterdir())
        for i in dcm_dir:
            logger.override_prefix = "exist" if i.exists() else "missing"
            logger.print(i, ltype=Log_Type.OK if i.exists() else Log_Type.FAIL)
    logger.override_prefix = "exp"
    # extract_folder(Path("/media/data/robert/test/dicom2nii/"), "/media/data/robert/test/dataset-out/")
    logger.print("START", ltype=Log_Type.OK)
    import os

    Parallel(n_jobs=min(os.cpu_count(), args.cpus))(
        delayed(extract_folder)(Path(path), Path(args.outfolder, f"dataset-{args.name}"), False, args.subjectchunks) for path in dcm_dir
    )
