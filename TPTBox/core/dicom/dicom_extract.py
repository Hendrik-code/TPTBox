import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import dicom2nifti
import dicom2nifti.exceptions
import pydicom
from dicom2nifti import convert_dicom
from func_timeout import func_timeout  # pip install func_timeout
from func_timeout.exceptions import FunctionTimedOut
from joblib import Parallel, delayed
from pydicom.dataset import FileDataset

from TPTBox import BIDS_FILE, Log_Type, Print_Logger
from TPTBox.core.dicom.dicom_header_to_keys import extract_keys_from_json
from TPTBox.core.nii_wrapper import NII

sys.path.append(str(Path(__file__).parent))

from TPTBox.core.dicom.dicom2nii_utils import get_json_from_dicom, save_json, test_name_conflict

logger = Print_Logger()


def _inc_key(keys, inc=1):
    k = "sequ"
    if k not in keys:
        keys[k] = 0
    try:
        v = int(keys[k])
        keys[k] = str(v + int(inc))
    except Exception:
        try:
            a, b = str(keys[k]).rsplit("-", maxsplit=2)
        except Exception:
            a = keys[k]
            b = 0
        keys[k] = a + "-" + str(int(b) + int(inc))


def generate_bids_path(
    nifti_dir, keys: dict, mri_format, simp_json, make_subject_chunks: int, parent="rawdata", _subject_folder_prefix="sub-"
):
    """Generate a BIDS-compatible file path."""
    assert "sub" in keys, keys
    p = Path(
        nifti_dir,  # dataset path
        parent,  # rawdata
        str(keys["sub"][:make_subject_chunks]) if make_subject_chunks != 0 else "",  # additional folder Optional
        keys.get("ses", ""),  # Session, if exist
        _subject_folder_prefix + keys["sub"],  # Subject folder
    )
    args = {"file_type": "json", "parent": parent, "make_parent": True, "additional_folder": mri_format, "bids_format": mri_format}
    fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_path(**args, info=keys)
    while test_name_conflict(simp_json, fname):
        _inc_key(keys)
        fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_path(**args, info=keys)
    return fname


def convert_to_nifti(dicom_out_path, nii_path):
    """Convert DICOM files to NIfTI format."""
    try:
        if isinstance(dicom_out_path, list):
            convert_dicom.dicom_array_to_nifti(dicom_out_path, nii_path, True)
        else:
            func_timeout(10, dicom2nifti.dicom_series_to_nifti, (dicom_out_path, nii_path, True))
        logger.print("Save ", nii_path, Log_Type.SAVE)
    except (dicom2nifti.exceptions.ConversionValidationError, FunctionTimedOut, ValueError):
        logger.print_error()
        return False
    return True


def get_paths(
    simp_json: dict, dcm_data_l: list[pydicom.FileDataset] | NII, nifti_dir: str | Path, make_subject_chunks=0, use_session=False
):
    (mri_format, keys) = extract_keys_from_json(simp_json, dcm_data_l, use_session)
    json_file_name = generate_bids_path(nifti_dir, keys, mri_format, simp_json, make_subject_chunks=make_subject_chunks)
    nii_path = str(json_file_name).replace(".json", "") + ".nii.gz"
    return json_file_name, nii_path


def from_dicom_to_nii(
    dcm_data_l: list[pydicom.FileDataset], nifti_dir: str | Path, make_subject_chunks: int = 0, use_session=False, verbose=True
):
    simp_json = get_json_from_dicom(dcm_data_l)
    fname, nii_path = get_paths(simp_json, dcm_data_l, nifti_dir, make_subject_chunks, use_session)
    logger.print(fname, Log_Type.NEUTRAL, verbose=verbose)
    exist = save_json(simp_json, fname)
    if exist and Path(nii_path).exists():
        logger.print("already exists:", fname, ltype=Log_Type.STRANGE, verbose=verbose)
        return nii_path
    suc = convert_to_nifti(dcm_data_l, nii_path)
    return nii_path if suc else None


def find_all_files(dcm_dirs: Path | list[Path]):
    dcm_dirs = dcm_dirs if isinstance(dcm_dirs, list) else [dcm_dirs]
    for dcm_dir in dcm_dirs:
        if dcm_dir.is_dir():
            for root, _, files in os.walk(dcm_dir):
                for file in files:
                    if str(file).endswith(".dcm"):
                        yield Path(file).parent
                        break
                    else:
                        yield Path(root) / file
        elif dcm_dir.name.endswith(".dcm"):
            yield Path(file).parent
        else:
            yield dcm_dir


def unzip_files(dicom_zip_path: Path, out_dir: str | Path) -> Path:
    with zipfile.ZipFile(dicom_zip_path, "r") as zip_ref:
        dicom_out_path = Path(out_dir, dicom_zip_path.stem)
        zip_ref.extractall(dicom_out_path)
    return dicom_out_path


def read_dicom_files(dicom_out_path: Path) -> dict[str, list[FileDataset]]:
    """Read DICOM files from a directory.

    Args:
        dicom_out_path (Path): Path to the directory containing DICOM files.

    Returns:
        Dict[str, List[FileDataset]]: Dictionary of DICOM files categorized by SeriesInstanceUID and type.
    """
    dicom_files = {}
    for _paths in dicom_out_path.rglob("*"):
        path = Path(_paths)
        if path.is_file():
            try:
                dcm_data = pydicom.dcmread(path, defer_size="1 KB", force=True)
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
    return dicom_files


def extract_folder(dicom_folder: Path | list[Path], dataset_path_out: Path, make_subject_chunks: int = 0, use_session=False, verbose=True):
    """Extract DICOM files, convert to NIfTI format, and optionally delete the original DICOM files.

    Args:
        dcm_dirs (Union[Path, List[Path]]): Directory or list of directories containing DICOM files.
        nifti_dir (Path): Directory to store the converted NIfTI files.
        del_dicom (bool, optional): Flag indicating if the original DICOM files should be deleted. Defaults to False.
        make_subject_chunks (int, optional): Parameter to control subject chunking. Defaults to 0.
        load_settings (bool, optional): Flag indicating if settings should be loaded from a file. Defaults to False.
        keep_settings (bool, optional): Flag indicating if settings should be saved to a file. Defaults to False.
    """
    outs = {}
    for p in find_all_files(dicom_folder):
        dicom_path = p
        if str(dicom_path).endswith(".pkl"):
            continue
        temp_dir = None
        try:
            # logger.print("Start", dicom_zip_path) if not str(dicom_zip_path).endswith(".dcm") else None
            if str(dicom_path).endswith(".zip"):
                temp_dir = tempfile.mkdtemp(prefix="dicom_export_from_zip_")
                dicom_path = unzip_files(dicom_path, temp_dir)
            dicom_files_dict = read_dicom_files(dicom_path)
            for key, files in dicom_files_dict.items():
                logger.print("Start", key, verbose=verbose)
                out = from_dicom_to_nii(files, dataset_path_out, make_subject_chunks, use_session, verbose=verbose)
                outs[key] = out
        except Exception:
            logger.print_error()
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir)
    return outs


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
