from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import BIDS_FILE, NII, BIDS_Global_info, Log_Type, Print_Logger, Subject_Container, to_nii_seg

logger = Print_Logger()
SEGMENTATION_TO_SMALL_THRESHOLD = 3000


def __test_seg_file(spinal_cord_file: Path):
    if spinal_cord_file.exists():
        try:
            data = to_nii_seg(spinal_cord_file).get_seg_array()
        except Exception:
            return False
    else:
        return False
    return not (data is None or data.sum() <= SEGMENTATION_TO_SMALL_THRESHOLD)


ignore_list = ["No properties for slice"]


def run_cmd(
    cmd_ex: list[str],
    out_file: Path | None = None,
    print_ignore_list: list[str] | None = None,
    verbose=True,
    override=False,
    logger=logger,
) -> int:
    """Runs the command in the list

    Args:
        cmd_ex (list[str]): Command line argument. Instead of whitespaces us a list to separate tokens. All whitespaces must be removed or they will be part of the Token.
        ouf_file (Path, optional): Prints f'[#] Saved to {ouf_file}', if given. Defaults to None.

    Returns:
        int: Error-Code
    """
    if not override and out_file is not None and out_file.exists():
        logger.print(f"[#] Exist; Skip {out_file!s}", Log_Type.OK)
        return 0
    if print_ignore_list is None:
        print_ignore_list = []
    process = subprocess.Popen(cmd_ex, stdout=subprocess.PIPE, universal_newlines=True)

    while True:
        assert process.stdout is not None
        output = process.stdout.readline()
        if output != "":
            logger.print("[SCT]", output.strip()) if verbose else None
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                time.sleep(0.001)
                print(output.strip(), print_ignore_list) if verbose else None
            logger.print("[#] RETURN CODE", return_code, Log_Type.OK) if verbose else None
            if out_file is not None and return_code == 0:
                logger.print(f"[#] Saved to {out_file!s}", Log_Type.SAVE) if verbose else None
            break
    return return_code


def compute_spinal_cord(file: BIDS_FILE, override=False, hot=True, do_label=True, **args):
    """Runs the spinalcord file

    Args:
        bids_file (BIDS_FILE): The image in the format set in domain, default is T2
        hot (bool, optional): If not hot the cmd commands are printed instead of executed. Defaults to True.
        domain: {t1,t2,t2s,dwi}
        threshold (float, optional): Custom Threshold. Defaults to -1 and means the preset threshold is used.
        out_file (Path | None, optional): override the default out_file. Defaults to None.
        parent_folder_name: Parentfolder (like rawdata/derivatives). Defaults to "derivatives"
        override (bool): override only if set to True
    Returns:
        None
    """
    cmd_ex_cord, spinal_cord_file = get_cmd_ex_spinal_cord(file, override=override, **args)

    if hot:
        # process = subprocess.Popen(cmd_ex_cord, stdout=subprocess.PIPE, universal_newlines=True)
        run_cmd(cmd_ex_cord, print_ignore_list=ignore_list)
        if not __test_seg_file(spinal_cord_file):
            print("[!] The tool failed to segment the spinalcord. Program tries again.")

            spinal_cord_file.unlink(missing_ok=True)
            # compute_spinal_cord_with_cut_sacrum
            msk_file = file.find_changed_path(bids_format="msk")
            if msk_file is None:
                msk_file = file.find_changed_path(bids_format="msk", info={"seg": "vert"})
            suc = False
            if msk_file is not None:
                suc = compute_spinal_cord_with_cut_sacrum(file, msk_file, **args)
            if not suc:
                spinal_cord_file.unlink(missing_ok=True)
                file.get_changed_path(file_type="nii.gz", bids_format="msk", info={"label": "spinalcord"})
                run_cmd([*cmd_ex_cord, "-centerline", "cnn"], print_ignore_list=ignore_list)
                if not __test_seg_file(spinal_cord_file):
                    print("[!] The tool failed to segment the spinalcord.Program gives up.")
                    spinal_cord_file.unlink(missing_ok=True)
                    return
        if do_label:
            run_get_cmd_ex_label(file, spinal_cord_file, **args)
    else:
        print(cmd_ex_cord)


def compute_spinal_cord_with_cut_sacrum(
    mri_file: BIDS_FILE,
    msk_file: BIDS_FILE,
    parent_folder_name="derivatives",
    verbose=True,
    **args,
):
    """Runs the spinalcord file, but before it uses a point msk or vertebra segmentation to remove the sacrum.
    This function only execute when there is the L4 (id 23) present and at least 4 other vertebra segmentation.

    Args:
        mri_file (BIDS_FILE): The image in the format set in domain, default is T2
        msk_file (BIDS_FILE): A point mask file or a vertebra segmentation
        hot (bool, optional): If not hot the cmd commands are printed instead of executed. Defaults to True.
        domain: {t1,t2,t2s,dwi}
        threshold (float, optional): Custom Threshold. Defaults to -1 and means the preset threshold is used.
        out_file (Path | None, optional): override the default out_file. Defaults to None.
        parent_folder_name: Parentfolder (like rawdata/derivatives). Defaults to "derivatives"
        override (bool): override only if set to True
    Returns:
        bool: if the function tried to compute it.
    """
    if verbose:
        print("[*] Try to segment spinalcord by cutting away the vertebra.")
    # Check if the mask is applicable
    msk_nii = msk_file.open_nii_reorient(axcodes_to=("A", "S", "R"))
    msk_arr = msk_nii.get_array()
    loc = np.where(msk_arr == 23)  # L4
    if len(loc) == 0 or len(loc[0]) == 0:
        if verbose:
            print("[?] No L4 Vertebra")
        return False
    # -1 is to remove the 0 (Background)
    num_vertebras = len(np.unique(msk_arr[:, loc[1][1] :])) - 1
    if num_vertebras < 4:
        if verbose:
            print(f"[?] Not enough Vertebras. Ids:{np.unique(msk_arr[:, loc[1][1] :])}")
        return False
    # Load MRI and cut
    mri_nii = mri_file.open_nii_reorient(axcodes_to=("A", "S", "R"))
    mri_arr = mri_nii.get_array()
    out_cut = mri_arr[:, loc[1][1] :].copy()
    tmp = Path("/tmp", f"temp_{secrets.token_urlsafe(21)}_cord")
    try:
        if not Path(tmp).exists():
            Path(tmp).mkdir()
        # Run spinalcord tools
        mri_path = str(Path(tmp, "mri.nii.gz"))
        nib.save(nib.Nifti1Image(out_cut, mri_nii.affine, mri_nii.header), str(mri_path))
        out_file = __bids2spinalcord(mri_file, parent_folder_name=parent_folder_name)
        cmd_ex_cord, spinal_cord_file = get_cmd_ex_spinal_cord(mri_path, out_file=out_file, **args)
        run_cmd(cmd_ex_cord, print_ignore_list=ignore_list)
        # revert padding
        if verbose:
            print("[*] revert padding")
        arr = NII(nib.load(spinal_cord_file), seg=True).get_seg_array()
        out = mri_arr.copy()
        out *= 0
        out[:, loc[1][1] :] = arr
        if out.sum() <= 1000:
            print(f"[?] Failed. There are only {out.sum()} pixel.")
            return False
        out_nib = NII(nib.Nifti1Image(out, mri_nii.affine, mri_nii.header))
        out_nib.reorient_same_as_(mri_file.open_nii(), verbose=verbose)
        out_nib.save(spinal_cord_file, verbose=verbose)
    except Exception:  # noqa: TRY302
        raise
    finally:
        shutil.rmtree(tmp)
    return True


def _parallelized_preprocess_dixon(subj_name: str, subject: Subject_Container, hot=True):
    """Generate a spinalcord segmentation for a inphase dixon of given Subject_Container.

    Args:
        subj_name (str): Name for printing
        subject (Subject_Container): The subject container.
        hot (bool, optional): If not hot the cmd commands are printed instead of executed. Defaults to True.
    """

    query = subject.new_query(flatten=True)
    query.filter("sub", "spinegan0015")
    # It must exist a dixon
    query.filter("format", "dixon")
    # A nii.gz must exist
    query.filter("Filetype", "nii.gz")
    # Compute what dixon has a real-part image (T2 like)

    query.action(
        # Set Part key to real. Will be called if the filter = True
        action_fun=lambda x: x.set("part", "real"),
        # x is the json, because of the key="json". We return True if the json confirms that this is a real-part image
        filter_fun=lambda x: "IP" in x["ImageType"],  # type: ignore
        key="json",
        # The json is required
        required=True,
    )
    query.filter("part", "real")

    ##### ERROR FILES###
    # spinegan0107 - 303 - [!] The tool failed to segment the spinalcord. Program tries again.
    # spinegan0038 - 803
    # sub-spinegan0092_ses-20201211_sequ-303_dixon.nii.gz
    # spinegan0049
    # spinegan0084_ses-20210728_sequ-303
    # sub-spinegan0012_ses-20210723_sequ-303_e-2_dixon.nii.gz
    # sub-spinegan0071_ses-20220618_sequ-303_e-2_dixon.nii.gz
    # sub-spinegan0056_ses-20220130_sequ-303_e-1_dixon.nii.gz
    # sub-spinegan0039_ses-20220503_sequ-303_e-1_dixon.nii.gz
    # sub-spinegan0106_ses-20220430_sequ-403_e-2_dixon.nii.gz
    # sub-spinegan0032_ses-20210705_sequ-302_e-1_dixon.nii.gz
    # sub-spinegan0045_ses-20210829_sequ-303_e-1_dixon.nii.gz
    # sub-spinegan0069_ses-20220114_sequ-903_e-3_dixon.nii.gz
    query.filter(
        "sub",
        lambda x: x
        not in [
            "spinegan0107",
            "spinegan0038",
            "spinegan0092",
            "spinegan0049",
            "spinegan0084",
            "spinegan0012",
            "spinegan0071",
            "spinegan0056",
            "spinegan0039",
            "spinegan0106",
            "spinegan0032",
            "spinegan0045",
            "spinegan0069",
        ],
    )

    for bids_file in query.loop_list():
        print("##################################################################")
        print("[#] Processing", subj_name)
        print("[#] ", bids_file)

        compute_spinal_cord(bids_file, hot=hot, parent_folder_name="derivatives_spinalcord", override=False, do_label=False)
        return


def __bids2spinalcord(bids_file: BIDS_FILE | Path | str, parent_folder_name: str):
    assert not isinstance(bids_file, Path), "out_file must be set, if a Path is provided instead of a BIDS_FILE"
    assert not isinstance(bids_file, str), "out_file must be set, if a Path is provided instead of a BIDS_FILE"
    return bids_file.get_changed_path(file_type="nii.gz", bids_format="msk", info={"label": "spinalcord"}, parent=parent_folder_name)


def get_cmd_ex_spinal_cord(
    bids_file: BIDS_FILE | Path | str,
    domain: str = "t2",
    threshold: float = -1,
    out_file: Path | None = None,
    parent_folder_name: str = "derivatives",
    override=True,
) -> tuple[list[str], Path]:
    """Defines the output file and the run command to segment the spinal cord with spinalcordtoolbox

    Args:
        bids_file (BIDS_FILE): The T2/dixon image. (This can be extended to T1 when the c flag is changed)
        domain: {t1,t2,t2s,dwi}
        threshold (float, optional): Custom Threshold. Defaults to -1 and means the preset threshold is used.
        out_file (Path | None, optional): override the default out_file. Defaults to None.
        parent_folder_name: Parentfolder (like rawdata/derivatives). Defaults to "derivatives"
        override (bool): override only if set to True
    Returns:
        tuple[list[str], BIDS_FILE]: The cmd command and the output filepath
    """
    if out_file is None:
        out_file = __bids2spinalcord(bids_file, parent_folder_name=parent_folder_name)
    if not override and out_file.exists():
        return ["echo", f"[?] the spinalcord already exists {out_file.name}."], out_file
    in_file = str(bids_file) if isinstance(bids_file, Path | str) else bids_file.file["nii.gz"]
    # sct_deepseg_sc -i sub-spinegan0008_ses-20210204_sequ-302_e-1_dixon.nii.gz -c t2 -brain 0 -o text.nii.gz
    cmd_ex = ["sct_deepseg_sc", "-i", in_file, "-c", domain, "-brain", "0", "-o", str(out_file)]
    if threshold != -1:
        cmd_ex += ["-thr", str(threshold)]
    return cmd_ex, out_file


def get_cmd_ex_deepseg(bids_file: BIDS_FILE):
    """Does not work well... See: sct_deepseg -h

    Args:
        bids_file (BIDS_FILE): _description_

    Returns:
        _type_: _description_
    """
    out_file = bids_file.get_changed_path(file_type="nii.gz", bids_format="msk", info={"label": "sct_deepseg"})
    in_file = bids_file.file["nii.gz"]
    # This ting took ages for a single file because it did not work on gpu
    cmd_ex = ["sct_deepseg", "-i", in_file, "-c", "t2", "-o", out_file, "-task", "seg_exvivo_gm-wm_t2"]

    # sct_deepseg -i sub-spinegan0042_ses-20220517_sequ-301_e-1_dixon.nii.gz -c t2 -o out_file -task seg_lumbar_sc_t2w
    return cmd_ex, out_file
    # sct_merge_images


def get_cmd_ex_label(
    bids_file: BIDS_FILE,
    spinal_cord_file: Path,
    domain: str = "t2",
    out_file: Path | None = None,
    parent_folder_name: str = "derivatives",
    override=False,
    **args,  # noqa: ARG001
) -> tuple[list[str], Path]:
    """Defines the output file and the run command to split the spinal cord segmentation into vertebrae-label with spinalcordtoolbox
    The output is a folder, for easier use there is a rapping function 'run_get_cmd_ex_label'

    Args:
        bids_file (BIDS_FILE): The image.
        spinal_cord_file (BIDS_FILE): spinalcord.
        domain: {t1,t2,t2s,dwi}
        threshold (float, optional): Unused
        out_file (Path | None, optional): override the default out_file. Defaults to None.
        parent_folder_name: Parentfolder (like rawdata/derivatives). Defaults to "derivatives"
        override (bool): override only if set to True
    Returns:
        tuple[list[str], BIDS_FILE]: The cmd command and the output filepath
    """
    if out_file is None:
        out_file = bids_file.get_changed_path(
            file_type="nii.gz", bids_format="msk", info={"label": "spinalcordlabel"}, parent=parent_folder_name
        )
    if not override and out_file.exists():
        return ["echo", f"[?] the spinalcord already exists {out_file.name}."], out_file

    in_file = bids_file.file["nii.gz"]
    cmd_ex = ["sct_label_vertebrae", "-i", in_file, "-c", domain, "-s", str(spinal_cord_file), "-o", str(out_file)]
    return cmd_ex, out_file


def run_get_cmd_ex_label(bids_file: BIDS_FILE, spinal_cord_file: Path, **args) -> None:
    """runs get_cmd_ex_label and moves files from the created folder to the final destination.
    Currently only the '_labeled' is moved and the rest is deleted.

    Args:
        bids_file (BIDS_FILE): The T1/T2/dixon inphase image
        spinal_cord_file: Path to the spinal_cord_segmentation
    """
    cmd_ex, out_file = get_cmd_ex_label(bids_file, spinal_cord_file, **args)
    name = spinal_cord_file.name.split(".", maxsplit=1)[0]
    if cmd_ex[0] == "echo":
        print(cmd_ex[-1])
        return
    cmd_ex[-1] = "/tmp/" + name
    return_code = run_cmd(cmd_ex, print_ignore_list=ignore_list)

    if return_code != 0:
        return
    if not Path(cmd_ex[-1]).exists():
        print("[!] the temporary output files are missing")
        return
    file1 = Path(cmd_ex[-1], name + "_labeled.nii.gz")  # Seg same as spinalcord but with vertebrae-label
    # file2 = Path(cmd_ex[-1], name + "_labeled_discs.nii.gz")  # source points of the labels (points where the vertebra-label changes)
    # file3 = Path(cmd_ex[-1], "straight_ref.nii.gz")  # The spine straightened
    # file4 = Path(cmd_ex[-1], "warp_curve2straight.nii.gz")  # Distortion field
    # file5 = Path(cmd_ex[-1], "warp_straight2curve.nii.gz")  # Distortion field
    import shutil

    # /tmp/sub-spinegan0042_ses-20220517_sequ-301_e-1_label-cord_label_msk
    # sub-spinegan0042_ses-20220517_sequ-301_e-1_label-spinalcord_msk_labeled.nii.gz

    shutil.copy(str(file1), str(out_file))
    shutil.rmtree(cmd_ex[-1])


# Does not work
# def get_cmd_ex_spinal_greymatter(bids_file):
#    out_file = bids_file.get_changed_path(file_type="nii.gz", format="msk", info={"label": "greymatter"})
#    in_file = bids_file.file["nii.gz"]
#    cmd_ex = ["sct_deepseg_gm", "-i", in_file, "-o", out_file]
#    return cmd_ex, out_file


def parallel_execution(n_jobs, path="/media/robert/Expansion/dataset-Testset", hot=True, function=_parallelized_preprocess_dixon):
    from joblib import Parallel, delayed

    global_info = BIDS_Global_info(
        [str(path)],
        ["rawdata", "rawdata_ct", "rawdata_dixon", "derivatives"],
        additional_key=["sequ", "seg", "ovl", "e"],
    )
    print(f"Found {len(global_info.subjects)} subjects in {global_info.datasets}")

    if n_jobs > 1:
        print(f"[*] Running {n_jobs} parallel jobs. Note that stdout will not be sequential")

    Parallel(n_jobs=n_jobs)(delayed(function)(subj_name, subject, hot) for subj_name, subject in global_info.enumerate_subjects())


if __name__ == "__main__":
    parallel_execution(1, "/media/data/robert/datasets/dataset_spinegan/", function=_parallelized_preprocess_dixon)

    ###############################################
    # parallel_execution(16, "/media/data/dataset-spinegan/register_test/", function=_parallelized_preprocess_dixon)
    # parallel_execution(1, "/media/data/robert/datasets/spinedata_temp", function=_parallelized_preprocess_dixon)
    ###############################################
    # file = BIDS_FILE(
    #    "/media/data/NAKO/MRT/3D_GRE_TRA_in/100001_30/ses-20160715/sub-100001_30_ses-20160715_sequ-3_mr.nii.gz",
    #    "/media/data/NAKO/MRT/3D_GRE_TRA_in/",
    # )
    # file = BIDS_FILE(
    #    "/media/data/NAKO/MRT/3D_GRE_TRA_W_COMPOSED/100001_30/ses-20160715/sub-100001_30_ses-20160715_sequ-w_mr.nii.gz",
    #    "/media/data/NAKO/MRT/3D_GRE_TRA_W_COMPOSED/",
    # )
    # cmd_ex_cord, out_file_cord = get_cmd_ex_spinal_cord(file, domain="t1", threshold=0.7)
    # out_file = "/media/data/NAKO/MRT/sub-100001_30_ses-20160715_sequ-3_label-spinalcord_msk.nii.gz"
    # cmd_ex_cord[-1] = out_file
    # run_cmd(cmd_ex_cord, out_file_cord)
    #################################################
    # file = BIDS_FILE(
    #    "/media/data/dataset-spinegan/register_test/rawdata_dixon/spinegan0010/sub-spinegan0010_ses-20220601_sequ-301_e-1_dixon.nii.gz",
    #    "/media/data/dataset-spinegan/register_test/",
    # )
    # compute_spinal_cord(file, parent_folder_name="derivatives_spinalcord")
    # file = BIDS_FILE(
    #    "/media/data/dataset-spinegan/register_test/rawdata_dixon/spinegan0010/sub-spinegan0010_ses-20220601_sequ-302_e-1_dixon.nii.gz",
    #    "/media/data/dataset-spinegan/register_test/",
    # )
    # compute_spinal_cord(file, parent_folder_name="derivatives_spinalcord")
    # a: nib.Nifti1Image = nib.load(
    #    "/media/data/dataset-spinegan/register_test/rawdata_dixon/spinegan0010/sub-spinegan0010_ses-20220601_sequ-302_e-1_dixon.nii.gz",
    # )
    ################################################
    # file = BIDS_FILE(
    #    "/media/data/robert/test/sub-spinegan0008_ses-20210204_sequ-302_e-1_dixon.nii.gz",
    #    "/media/data/robert/test/",
    # )
    # file2 = BIDS_FILE(
    #    "/media/data/robert/test/sub-spinegan0008_ses-20210204_sequ-302_e-1_msk.nii.gz",
    #    "/media/data/robert/test/",
    # )
    # file = BIDS_FILE(
    #    "/media/data/robert/test/sub-spinegan0007_ses-20210913_sequ-303_e-3_dixon.nii.gz",
    #    "/media/data/robert/test/",
    # )
    # file2 = BIDS_FILE(
    #    "/media/data/robert/test/sub-spinegan0007_ses-20210913_sequ-303_e-3_msk.nii.gz",
    #    "/media/data/robert/test/",
    # )
    # compute_spinal_cord_with_cut_sacrum(file, file2)
