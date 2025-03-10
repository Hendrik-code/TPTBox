from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dicom2nifti
import dicom2nifti.exceptions
import numpy as np
import pydicom
from dicom2nifti import convert_dicom
from func_timeout import func_timeout  # pip install func_timeout
from func_timeout.exceptions import FunctionTimedOut
from joblib import Parallel, delayed
from pydicom.dataset import FileDataset

from TPTBox import BIDS_FILE, Log_Type, Print_Logger
from TPTBox.core.compat import zip_strict
from TPTBox.core.dicom.dicom_header_to_keys import extract_keys_from_json
from TPTBox.core.nii_wrapper import NII

sys.path.append(str(Path(__file__).parent))

from TPTBox.core.dicom.dicom2nii_utils import get_json_from_dicom, load_json, save_json, test_name_conflict

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


def _generate_bids_path(
    nifti_dir, keys: dict, mri_format, simp_json, make_subject_chunks: int, parent="rawdata", _subject_folder_prefix="sub-"
):
    """
    Generate a BIDS-compatible file path for NIfTI outputs based on extracted keys from DICOM headers.

    Args:
        nifti_dir (str | Path): Directory where the NIfTI file will be stored.
        keys (dict): Dictionary containing metadata keys extracted from DICOM headers.
        mri_format (str): The format or sequence type of the MRI (e.g., T1, T2).
        simp_json (dict): JSON dictionary with extracted DICOM information to avoid file naming conflicts.
        make_subject_chunks (int): Number of subject ID characters used for creating subdirectories.
        parent (str, optional): Parent folder in the BIDS structure. Defaults to "rawdata".
        _subject_folder_prefix (str, optional): Prefix for subject folders (e.g., "sub-"). Defaults to "sub-".

    Returns:
        Path: Path object for the BIDS-compliant file name.
    """

    assert "sub" in keys, keys
    ses = keys.get("ses")
    ses = f"ses-{ses}" if ses is not None else ""
    keys["sub"] = keys["sub"].replace(".", "-")
    assert keys["sub"] is not None, keys
    p = Path(
        nifti_dir,  # dataset path
        parent,  # rawdata
        str(keys["sub"][:make_subject_chunks]) if make_subject_chunks != 0 else "",  # additional folder Optional
        _subject_folder_prefix + keys["sub"],  # Subject folder
        ses,  # Session, if exist
    )
    args = {"file_type": "json", "parent": parent, "make_parent": True, "additional_folder": mri_format, "bids_format": mri_format}
    fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_bids(**args, info=keys, non_strict_mode=True)
    while test_name_conflict(simp_json, fname.file["json"]):
        _inc_key(keys)
        fname = BIDS_FILE(Path(p, "sub-000_ct.nii.gz"), nifti_dir).get_changed_bids(**args, info=keys, non_strict_mode=True)
    return fname.file["json"], fname


def _convert_to_nifti(dicom_out_path, nii_path):
    """
    Convert DICOM files to NIfTI format and handle common conversion errors.

    Args:
        dicom_out_path (list | str | Path): List of DICOM `FileDataset` objects or path to a DICOM directory.
        nii_path (str | Path): Path where the output NIfTI file should be saved.

    Returns:
        bool: True if conversion is successful, False if an error occurs.

    Raises:
        dicom2nifti.exceptions.ConversionValidationError: Raised for non-imaging or localizer DICOMs.
        FunctionTimedOut: Raised if the DICOM-to-NIfTI conversion times out.
        ValueError: Raised for generic validation failures.
    """
    try:
        if isinstance(dicom_out_path, list):
            convert_dicom.dicom_array_to_nifti(dicom_out_path, nii_path, True)
        else:
            func_timeout(10, dicom2nifti.dicom_series_to_nifti, (dicom_out_path, nii_path, True))
        logger.print("Save ", nii_path, Log_Type.SAVE)
    except dicom2nifti.exceptions.ConversionValidationError as e:
        if e.args[0] in ["NON_IMAGING_DICOM_FILES"]:
            Path(str(nii_path).replace(".nii.gz", ".json")).unlink(missing_ok=True)
            logger.on_debug(f"Not exportable '{Path(nii_path).name}':", e.args[0])
            return False
        for key, reason in [
            ("validate_orthogonal", "NON_CUBICAL_IMAGE/GANTRY_TILT"),
            ("validate_orientation", "IMAGE_ORIENTATION_INCONSISTENT"),
            ("validate_slicecount", "TOO_FEW_SLICES/LOCALIZER"),
        ]:
            if e.args[0] == reason:
                Path(str(nii_path).replace(".nii.gz", ".json")).unlink(missing_ok=True)
                logger.on_debug(
                    f"Not exported cause of {reason}, if you want try to export it anyway, set {key} = False; If it already on False, than dicom2nifti does not allow the export"
                )
                return False
        logger.print_error()
        logger.on_fail(Path(nii_path).name)

    except (FunctionTimedOut, ValueError):
        logger.print_error()

        return False
    return True


def _get_paths(
    simp_json: dict,
    dcm_data_l: list[pydicom.FileDataset],
    nifti_dir: str | Path,
    make_subject_chunks=0,
    use_session=False,
    parts: list | None = None,
    map_series_description_to_file_format=None,
    override_subject_name: Callable[[dict, Path], str] | None = None,
    chunk: int | str | None = None,
):
    (mri_format, keys) = extract_keys_from_json(
        simp_json,
        dcm_data_l,
        use_session,
        parts,
        map_series_description_to_file_format,
        override_subject_name=override_subject_name,
        chunk=chunk,
    )
    json_file_name, json_bids_name = _generate_bids_path(nifti_dir, keys, mri_format, simp_json, make_subject_chunks=make_subject_chunks)
    nii_path = str(json_file_name).replace(".json", "") + ".nii.gz"
    return json_file_name, json_bids_name, nii_path


def _filter_dicom(dcm_data_l: list[pydicom.FileDataset]):
    if len(dcm_data_l) == 1:
        return dcm_data_l
    dcm_data_l = [d for d in dcm_data_l if hasattr(d, "ImageOrientationPatient")]
    return dcm_data_l


def _from_dicom_to_nii(
    dcm_data_l: list[pydicom.FileDataset],
    nifti_dir: str | Path,
    make_subject_chunks: int = 0,
    use_session=False,
    verbose=True,
    parts: list | None = None,
    map_series_description_to_file_format=None,
    override_subject_name: Callable[[dict, Path], str] | None = None,
    chunk=None,
    skip_localizer=False,
):
    if chunk is None:
        splitted_dcm_data_l = _classic_get_grouped_dicoms(dcm_data_l)
        if len(splitted_dcm_data_l) != 1:
            outs = []
            for i, dcm in enumerate(splitted_dcm_data_l, 1):
                o = _from_dicom_to_nii(
                    dcm,
                    nifti_dir=nifti_dir,
                    make_subject_chunks=make_subject_chunks,
                    use_session=use_session,
                    verbose=verbose,
                    parts=parts,
                    map_series_description_to_file_format=map_series_description_to_file_format,
                    override_subject_name=override_subject_name,
                    chunk=i,
                    skip_localizer=skip_localizer,
                )
                outs.append(o)
            return outs
    dcm_data_l = _filter_dicom(dcm_data_l)
    if len(dcm_data_l) == 0:
        logger.on_neutral(Path(nifti_dir).name, " - not an image. Will be skipped")
        return None

    simp_json = get_json_from_dicom(dcm_data_l)
    json_file_name, json_bids, nii_path = _get_paths(
        simp_json,
        dcm_data_l,
        nifti_dir,
        make_subject_chunks,
        use_session,
        parts,
        map_series_description_to_file_format,
        override_subject_name,
        chunk=chunk,
    )
    if skip_localizer and json_bids.bids_format == "localizer":
        return
    logger.print(json_file_name, Log_Type.NEUTRAL, verbose=verbose)
    exist = save_json(simp_json, json_file_name)
    if exist and Path(nii_path).exists():
        logger.print("already exists:", json_file_name, ltype=Log_Type.STRANGE, verbose=verbose)
        return nii_path
    suc = _convert_to_nifti(dcm_data_l, nii_path)

    if suc:
        _add_grid_info_to_json(nii_path, json_file_name)
    return nii_path if suc else None


def _add_grid_info_to_json(nii_path: Path | str, simp_json: Path | str, force_update=False):
    nii = NII.load(nii_path, False)
    json_dict = load_json(simp_json) if Path(simp_json).exists() else {}
    if "grid" in json_dict and not force_update:
        return json_dict
    gird = {
        "shape": nii.shape,
        "spacing": nii.spacing,
        "orientation": nii.orientation,
        "rotation": nii.rotation.reshape(-1).tolist(),
        "origin": nii.origin,
        "dims": nii.get_num_dims(),
    }
    json_dict["grid"] = gird
    save_json(json_dict, simp_json)
    return json_dict


def _find_all_files(dcm_dirs: Path | list[Path]):
    """
    Recursively find all DICOM directories or files in the given paths.

    Args:
        dcm_dirs (Path | list[Path]): A directory or list of directories to search for DICOM files.

    Yields:
        Path: Paths to directories or individual DICOM files found during the search.
    """
    yield dcm_dirs
    dcm_dirs = dcm_dirs if isinstance(dcm_dirs, list) else [dcm_dirs]
    for dcm_dir in dcm_dirs:
        if dcm_dir.is_dir():
            for root, _, files in os.walk(dcm_dir):
                file = ""
                for file in files:
                    if Path(file).is_file():  # str(file).endswith(".dcm") or str(file).endswith(".ima")
                        yield Path(root, file).absolute().parent
                        break
                    else:
                        yield Path(root, file)
                # if "." not in str(file):
                #    yield Path(root, file).absolute().parent

        # elif Path(dcm_dir).is_file():  # dcm_dir.name.endswith(".dcm"):
        #    yield Path(root, file).absolute().parent
        # else:
        #    yield dcm_dir.absolute()


def _unzip_files(dicom_zip_path: Path, out_dir: str | Path) -> Path:
    with zipfile.ZipFile(dicom_zip_path, "r") as zip_ref:
        dicom_out_path = Path(out_dir, dicom_zip_path.stem)
        zip_ref.extractall(dicom_out_path)
    return dicom_out_path


def _read_dicom_files(dicom_out_path: Path) -> tuple[dict[str, list[FileDataset]], dict[str, list[str]]]:
    """
    Read DICOM files from a directory and categorize them based on SeriesInstanceUID and type.

    Args:
        dicom_out_path (Path): Path to the directory containing DICOM files.

    Returns:
        tuple:
            - dict[str, list[FileDataset]]: Dictionary where keys are series identifiers and values are lists of `FileDataset` objects.
            - dict[str, list[str]]: Dictionary mapping SeriesInstanceUIDs to a list of DICOM types found.
    """
    dicom_files = {}
    dicom_types: dict[str, list[str]] = {}
    for _paths in dicom_out_path.rglob("*"):
        path = Path(_paths)
        if path.is_file():
            try:
                dcm_data = pydicom.dcmread(path, defer_size="1 KB", force=True)
                try:
                    typ = (
                        str(dcm_data.get_item((0x0008, 0x0008)).value)
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace(" ", "")
                        .replace("\\\\", ",")[1:]
                    )
                except Exception:
                    typ = ""
                key1 = str(dcm_data.SeriesInstanceUID)

                key = f"{key1}_{typ}"
                if not hasattr(dcm_data, "ImageOrientationPatient"):
                    key += "_" + dcm_data.get("SOPInstanceUID", 0)

                if key1 not in dicom_types:
                    dicom_types[key1] = []
                if typ not in dicom_types[key1]:
                    dicom_types[key1].append(typ)
                if key not in dicom_files:
                    dicom_files[key] = []
                dicom_files[key].append(dcm_data)
            except AttributeError:
                pass
    for key, value in dicom_files.items():
        dicom_files[key] = sorted(value, key=lambda dcm_data: np.array(dcm_data.get("ImagePositionPatient", 0)).sum())
    return dicom_files, _filter_file_type(dicom_types)


def _classic_get_grouped_dicoms(dicom_input: list[FileDataset]) -> list[list[FileDataset]]:
    """
    Search all dicoms in the dicom directory, sort and validate them

    fast_read = True will only read the headers not the data
    """
    # Order all dicom files by InstanceNumber
    dicoms = sorted(dicom_input, key=lambda x: x.InstanceNumber)

    # now group per stack
    grouped_dicoms: list[list[FileDataset]] = [[]]  # list with first element a list
    stack_index = 0

    # loop over all sorted dicoms and sort them by stack
    # for this we use the position and direction of the slices so we can detect a new stack easily
    previous_position = None
    previous_direction = None
    for dicom_ in dicoms:
        current_direction = None
        # if the stack number decreases we moved to the next stack
        if previous_position is not None:
            current_direction = np.array(dicom_.get("ImagePositionPatient", 0)) - previous_position
            current_direction = current_direction / np.linalg.norm(current_direction)

        if (
            current_direction is not None
            and previous_direction is not None
            and not np.allclose(current_direction, previous_direction, rtol=0.05, atol=0.05)
        ):
            previous_position = np.array(dicom_.get("ImagePositionPatient", 0))
            previous_direction = None
            stack_index += 1
        else:
            previous_position = np.array(dicom_.get("ImagePositionPatient", 0))
            previous_direction = current_direction

        if stack_index >= len(grouped_dicoms):
            grouped_dicoms.append([])
        grouped_dicoms[stack_index].append(dicom_)
    others = []
    out = []
    for i in grouped_dicoms:
        if len(i) <= 3:
            others += i
        else:
            out.append(i)
    if len(others) != 0:
        out.append(others)
    return out


def _filter_file_type(dicom_types: dict[str, list[str]]):
    dicom_parts: dict[str, list[str]] = {}
    for k, v in dicom_types.items():
        if len(v) == 1:
            continue
        # Split the strings and extract the third part
        split_lists = [set(i.split(",")) for i in v]
        # Flatten the lists and count the occurrences
        all_strings = [item for sublist in split_lists for item in sublist]
        string_counts = {string: all_strings.count(string) for string in set(all_strings)}
        # Collect strings that only appear in one of the lists
        unique_strings = {string for string, count in string_counts.items() if count == 1}
        # Filter out non-unique strings in each sublist
        filtered_set = []
        for sublist in split_lists:
            filtered_set.append([item for item in sublist if item in unique_strings])  # noqa: PERF401
        # Add to dicom_parts if there are unique strings
        for l, i in zip_strict(filtered_set, v):
            dicom_parts[f"{k}_{i}"] = l
    return dicom_parts


parts_mapping = {
    "f": "fat",
    "w": "water",
    "opp": "outphase",
    "op": "outphase",
    "ip": "inphase",
    "inp": "inphase",
}


def extract_dicom_folder(
    dicom_folder: Path | list[Path],
    dataset_path_out: Path,
    make_subject_chunks: int = 0,
    use_session=False,
    verbose=True,
    parts_mapping: dict = parts_mapping,
    map_series_description_to_file_format=None,
    validate_slicecount=True,
    validate_orientation=True,
    validate_orthogonal=False,
    n_cpu: int | None = 1,
    override_subject_name: Callable[[dict, Path], str] | None = None,
    skip_localizer=True,
):
    """
    Extract DICOM files from a directory or list of directories, convert them to NIfTI format, and store the output.

    Args:
        dicom_folder (Union[Path, list[Path]]): Path to a directory or list of directories containing DICOM files or compressed DICOM archives.
        dataset_path_out (Path): Path to the directory where the converted NIfTI files will be stored.
        make_subject_chunks (int, optional): Parameter to control chunking of subject data into subdirectories. Defaults to 0 (no chunking).
        use_session (bool, optional): Flag to determine if session information should be included in the output path. Defaults to False.
        verbose (bool, optional): Whether to print detailed log information. Defaults to True.
        parts_mapping (dict, optional): A dictionary mapping DICOM part identifiers to specific descriptions (e.g., "f" -> "fat").
                                        Used for categorizing DICOM series. Defaults to a predefined mapping. The parts tag is only generated if the ImageType causes an image split.
        n_cpu (int, optional): Number of CPU cores to use for parallel processing. Defaults to 1 (sequential).

    Returns:
        dict: A dictionary with keys representing DICOM series and values as paths to the generated NIfTI files.
    """
    if not validate_slicecount:
        convert_dicom.settings.disable_validate_slicecount()
    if not validate_orientation:
        convert_dicom.settings.disable_validate_orientation()
    if not validate_orthogonal:
        convert_dicom.settings.disable_validate_orthogonal()

    outs = {}

    for p in _find_all_files(dicom_folder):
        dicom_path = p

        if str(dicom_path).endswith(".pkl"):
            continue
        temp_dir = None
        try:
            if str(dicom_path).endswith(".zip"):
                temp_dir = tempfile.mkdtemp(prefix="dicom_export_from_zip_")
                dicom_path = _unzip_files(dicom_path, temp_dir)

            dicom_files_dict, parts = _read_dicom_files(dicom_path)

            def process_series(key, files, parts):
                """Helper function for processing a single DICOM series."""
                logger.print("Start", key, verbose=verbose)
                part = parts.get(key, None)
                if part is not None:
                    part = [parts_mapping.get(p.lower(), p.lower()) for p in part]
                return key, _from_dicom_to_nii(
                    files,
                    dataset_path_out,
                    make_subject_chunks,
                    use_session,
                    verbose=verbose,
                    parts=part,
                    map_series_description_to_file_format=map_series_description_to_file_format,
                    override_subject_name=override_subject_name,
                    skip_localizer=skip_localizer,
                )

            # Process in parallel or sequentially based on n_cpu
            if n_cpu is None or n_cpu > 1:
                with ThreadPoolExecutor(max_workers=n_cpu) as executor:
                    futures = {executor.submit(process_series, key, files, parts): key for key, files in dicom_files_dict.items()}
                    for future in as_completed(futures):
                        try:
                            key, out = future.result()
                            outs[key] = out
                        except Exception:
                            logger.print_error()
            else:
                for key, files in dicom_files_dict.items():
                    try:
                        key2, out = process_series(key, files, parts)
                        outs[key2] = out
                    except Exception:
                        logger.print_error()

        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir)

    return outs


if __name__ == "__main__":
    # s = "/home/robert/Downloads/bein/dataset-oberschenkel/rawdata/sub-1-3-46-670589-11-2889201787-2305829596-303261238-2367429497/mr/sub-1-3-46-670589-11-2889201787-2305829596-303261238-2367429497_sequ-406_mr.nii.gz"
    # nii2 = NII.load(s, False)
    # print(nii2.affine, nii2.orientation)
    # nii3 = None
    # k = 1
    # for i in Path("/home/robert/Downloads/bein/").glob("ID001_*.nrrd"):
    #    nii = NII.load_nrrd(i, True)
    #    nii.reorient_(nii2.orientation)
    #    nii.affine = nii2.affine
    #    nii.save(str(i).replace(".nrrd", ".nii.gz"))
    #    if nii3 is None:
    #        nii3 = nii.copy() * 0
    #    print(nii3.unique())
    #    nii += nii3
    #    nii3[np.logical_and(nii == 1, nii3 == 0) == 1] = k
    #    k += 1
    # print(nii3.unique())
    # nii3.save("/home/robert/Downloads/bein/ID001.nii.gz")
    # extract_dicom_folder(Path("/home/robert/Downloads/bein/a"), Path("/home/robert/Downloads/bein/", "dataset-oberschenkel"), False, False)

    # exit()
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
            logger.prefix = "exist" if i.exists() else "missing"
            logger.print(i, ltype=Log_Type.OK if i.exists() else Log_Type.FAIL)
        dcm_dir = [c for c in dcm_dir2 if c.exists()]

        if len(dcm_dir2) != len(dcm_dir) and input("some folders are missing! Continue anyway? (y/n)") != "y":
            sys.exit()
    else:
        dcm_dir = list(Path(args.inputfolder).iterdir())
        for i in dcm_dir:
            logger.prefix = "exist" if i.exists() else "missing"
            logger.print(i, ltype=Log_Type.OK if i.exists() else Log_Type.FAIL)
    logger.prefix = "exp"
    # extract_folder(Path("/media/data/robert/test/dicom2nii/"), "/media/data/robert/test/dataset-out/")
    logger.print("START", ltype=Log_Type.OK)
    import os

    Parallel(n_jobs=min(os.cpu_count(), args.cpus))(
        delayed(extract_dicom_folder)(Path(path), Path(args.outfolder, f"dataset-{args.name}"), False, args.subjectchunks)
        for path in dcm_dir
    )
