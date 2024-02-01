#!/usr/bin/python
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
import typing
import warnings
from pathlib import Path
from secrets import token_urlsafe

from TPTBox import BIDS_FILE, BIDS_Family, BIDS_Global_info, Image_Reference, to_nii
from TPTBox.logger.log_file import Log_Type, Logger, Logger_Interface, No_Logger

docker_image = "anjanys/bonescreen-segmentor:main"
docker_cmd = "/src/process_spine.py"


def run_docker_on_ds_advanced(
    bids_ds: BIDS_Global_info | str | Path,
    docker_input_ctd: bool = True,
    docker_input_vertmsk: bool = False,
    cut_sacrum_ctd: bool = True,
    save_subreg_mask: bool = True,
    save_as_sourcedocker: bool = True,
    # delete_tmp_after_done: bool = True,
    save_log: bool = True,
    logger: bool | Logger_Interface = False,
):
    """Runs the docker over a whole dataset

    Args:
        bids_ds: The BIDS_Global_info dataset (must contain only ONE dataset)
        docker_input_ctd: whether centroids are given to the docker
        docker_input_vertmsk: whether the vert-mask are given to the docker
        cut_sacrum_ctd: If true, removes sacrum centroids from a temporary copy so not to crash the docker
        save_subreg_mask: If true, saves the subregion mask created by the docker
        save_as_sourcedocker: If true, saves the data with source-docker in the name
        delete_tmp_after_done: If False, does not remove the temporary files
        save_log: Saves a log into <dataset>/logs/
        logger: given Logger. If True/False, creates a logger with default_verbose=logger

    Returns:
        bool
    """
    if isinstance(bids_ds, (str, Path)):
        bids_ds = BIDS_Global_info([bids_ds], clear=True)
    assert len(bids_ds.datasets) == 1, f"run_docker_on_ds_advanced: can only receive one dataset, not {len(bids_ds.datasets)}"
    arguments = locals()
    ds_parent = bids_ds.datasets[0]
    log_filename = "source-run-docker-advanced"
    if isinstance(logger, bool):
        logger = Logger(path=ds_parent, log_filename=log_filename, default_verbose=logger) if save_log else No_Logger()
    logger.print("Run with arguments", arguments, "\n")
    ds_raw = Path(ds_parent).joinpath("rawdata")
    ds_der = Path(ds_parent).joinpath("derivatives")
    assert os.path.exists(ds_raw), f"rawdata doesnt exist, expected in {ds_raw}"
    assert os.path.exists(ds_der), f"derivatives doesnt exist, expected in {ds_der}"
    if not docker_input_ctd and cut_sacrum_ctd:
        cut_sacrum_ctd = False
        warnings.warn(
            "run_docker: set cut_sacrum_ctd = False because docker_input_ctd is False",
            UserWarning,
        )

    tmp_raw = ds_raw.joinpath("rawdata")
    tmp_der = ds_raw.joinpath("derivatives")
    tmp_raw.mkdir(exist_ok=True, parents=True)
    tmp_der.mkdir(exist_ok=True, parents=True)
    logger.print("[*] created tmp folder for docker in:", ds_raw)

    important_family_keys = ["ct"]
    important_family_keys.append("msk_seg-vert") if docker_input_vertmsk else None
    important_family_keys.append("ctd_seg-subreg") if docker_input_ctd else None

    subject_docker_list = []

    for name, sample in bids_ds.enumerate_subjects(sort=True):
        q = sample.new_query()
        q.flatten()
        q.filter(key="filetype", filter_fun=lambda x: x == "nii.gz" or x == "json")
        q.unflatten()
        families: typing.Iterator[BIDS_Family] = q.loop_dict(key_addendum=["source", "space"])
        for family in families:
            if important_family_keys not in family:
                logger.print(
                    "[!]",
                    name,
                    f"requires {important_family_keys}, but only got {family.get_key_len()}",
                )
                continue
            family_dict: dict = family.get_bids_files_as_dict(important_family_keys)  # TODO may return string
            ct_ref = family_dict["ct"]
            dataset, parent, subpath, filename = ct_ref.get_path_decomposed()
            parentfolder = Path(subpath)
            sub_raw = tmp_raw.joinpath(parentfolder)
            sub_der = tmp_der.joinpath(parentfolder)
            sub_raw.mkdir(exist_ok=True, parents=True)
            sub_der.mkdir(exist_ok=True, parents=True)

            rawdata_file_origin: Path = ct_ref.file["nii.gz"]
            raw_to = tmp_raw.joinpath(parentfolder, rawdata_file_origin.name)
            symlink(rawdata_file_origin, raw_to)
            exit()
            given_der = [i for k, i in family_dict.items() if k != "ct"]
            copy_files_to_tmp(given_der=given_der, tmp_der=sub_der, cut_sacrum_ctd=cut_sacrum_ctd)

            subject_docker_list.append((name, rawdata_file_origin, sub_der, parentfolder))
    logger.print(f"[*] found and readied {len(subject_docker_list)} scans for the docker")
    if len(subject_docker_list) == 0:
        shutil.rmtree(tmp_raw)
        shutil.rmtree(tmp_der)
        logger.print("[*] Removed tmp folders")
        return
    # Run docker over tmp dataset
    try:
        run_docker(ds_raw)
    except BaseException as e:
        raise e

    # copy results back
    for n, rfo, sd, pf in subject_docker_list:
        sub_ses_id = str(rfo).split("/")[-1].split("_ct")[0]
        sub_der = sd
        der_to = ds_der.joinpath(pf)

        try:
            if not docker_input_ctd:
                move_back_to_original_dir(
                    f="seg-subreg_ctd.json",
                    tmp_der=sub_der,
                    sub_ses_id=sub_ses_id,
                    save_as_docker=save_as_sourcedocker,
                    derivative_dir_origin=der_to,
                    logger=logger,
                )
            if save_subreg_mask:  # parameter determines if subreg mask is saved
                move_back_to_original_dir(
                    f="seg-subreg_msk.nii.gz",
                    tmp_der=sub_der,
                    sub_ses_id=sub_ses_id,
                    save_as_docker=save_as_sourcedocker,
                    derivative_dir_origin=der_to,
                    logger=logger,
                )
            if not docker_input_vertmsk:  # only save vert_mask if it isnt given
                move_back_to_original_dir(
                    f="seg-vert_msk.nii.gz",
                    tmp_der=sub_der,
                    sub_ses_id=sub_ses_id,
                    save_as_docker=save_as_sourcedocker,
                    derivative_dir_origin=der_to,
                    logger=logger,
                )
        except Exception as e:
            print(n, rfo, sd, pf)
            logger.print(f"[!] Error with {n}, {rfo}, {sd}, {pf}")
            logger.print(e)
            continue
    shutil.rmtree(tmp_raw)
    shutil.rmtree(tmp_der)
    logger.print("[*] Removed tmp folders")
    return True


def run_docker_on_sample_advanced(
    ct_ref: BIDS_FILE,
    msk_vert_ref: BIDS_FILE | None = None,
    ctd_subreg_ref: BIDS_FILE | None = None,
    cut_sacrum_ctd: bool = True,
    save_subreg_mask: bool = True,
    save_as_sourcedocker: bool = True,
    delete_tmp_after_done: bool = True,
    save_log: bool = True,
    logger: bool | Logger_Interface = False,
) -> str | bool:
    """Runs the docker over a specific subject folder.

    Args:
        ct_ref: Given rawdata CT bids file
        msk_vert_ref: vertebra mask bids file (If none, docker will run without it and calc it)
        ctd_subreg_ref: centroid bids file (If none, docker will run without it and calc it)
        cut_sacrum_ctd: If true, will remove sacrum centroids from the temporary centroid copy the docker cannot handle (if centroids are given, will not copy the centroid file back)
        save_subreg_mask: if true the subreg mask is copied back
        save_as_sourcedocker: If true, will save the files as source-docker instead of overriding the existing files
        delete_tmp_after_done: If true, will cleanup the temporary files after being done
        save_log: If true, will save a _log.log file in the derivatives folder that contains the docker log output
        logger: the logger to log to, if true/false, will create a logger with default_verbose = logger

    Returns:
        alert (str) if something went wrong, True (bool) if it ran through
    """

    myhost = os.uname()[1]
    if myhost == "epic":
        return run_docker_on_sample_advanced_epic(
            ct_ref,
            msk_vert_ref,
            ctd_subreg_ref,
            cut_sacrum_ctd,
            save_subreg_mask,
            save_as_sourcedocker,
            delete_tmp_after_done,
            save_log,
            logger,
        )
    assert myhost != "epic"
    assert "nii.gz" in ct_ref.file
    rawdata_file_origin: Path = ct_ref.file["nii.gz"]
    if isinstance(logger, bool):
        print("[#######] logger", logger)
        log_filename = "source-run-docker-single"
        logger = (
            Logger(
                path=ct_ref.get_parent("nii.gz"),
                log_filename=log_filename,
                default_verbose=logger,
            )
            if save_log
            else No_Logger()
        )

    if msk_vert_ref is not None and ctd_subreg_ref is not None and not save_subreg_mask:
        alert = "Run_docker: got vert_msk, centroids and not save_subreg_mask. Skip"
        logger.print(alert)
        return alert

    sub_ses_id = str(rawdata_file_origin).split("/")[-1].split("_ct")[0]
    # the temporary sub dataset that the docker is run on
    tmp_file_dir = rawdata_file_origin.parent  # .joinpath("tmp_docker")
    derivative_dir_origin = Path(str(tmp_file_dir).replace("rawdata", "derivatives"))
    tmp_file_dir = tmp_file_dir.joinpath(token_urlsafe(25) + "_run_docker_on_sample_advanced")
    tmp_file_dir.mkdir(exist_ok=True)

    # make config
    config_path = Path(tmp_file_dir, "config_custom.yaml")
    f = open(config_path, "a")
    f.write(f"PROCESS_SELECTED: [{rawdata_file_origin.name}]")
    f.close()

    # the raw and derivatives below
    tmp_raw = tmp_file_dir.joinpath("rawdata")
    tmp_der = tmp_file_dir.joinpath("derivatives")
    tmp_raw.mkdir(exist_ok=True, parents=True)
    tmp_der.mkdir(exist_ok=True, parents=True)
    logger.print("[*] created tmp folder for docker:", tmp_raw)
    # Copy/symlink all files into the temp dataset
    raw_to = tmp_raw.joinpath(rawdata_file_origin.name)
    symlink(rawdata_file_origin, raw_to)
    given_der: list[BIDS_FILE] = [msk_vert_ref, ctd_subreg_ref]  # type: ignore
    given_der = [i for i in given_der if i is not None]
    copy_files_to_tmp(given_der=given_der, tmp_der=tmp_der, cut_sacrum_ctd=cut_sacrum_ctd)
    logger.print("[*] copied files into tmp folder", tmp_der)

    try:
        run_docker(tmp_file_dir.parent, log=logger, config_path=config_path)
        args = {
            "tmp_der": tmp_der,
            "sub_ses_id": sub_ses_id,
            "save_as_docker": save_as_sourcedocker,
            "derivative_dir_origin": derivative_dir_origin,
            "logger": logger,
        }
        if ctd_subreg_ref is None:  # only save centroids if they weren't given
            move_back_to_original_dir(f="seg-subreg_ctd.json", **args)
        if save_subreg_mask:  # parameter determines if subreg mask is saved
            move_back_to_original_dir(f="seg-subreg_msk.nii.gz", **args)
        if msk_vert_ref is None:  # only save vert_mask if it isn't given
            move_back_to_original_dir(f="seg-vert_msk.nii.gz", **args)

    except BaseException as e:
        if delete_tmp_after_done:
            shutil.rmtree(tmp_file_dir, ignore_errors=True)
        raise e
    if delete_tmp_after_done:
        shutil.rmtree(tmp_file_dir)
    return True


def run_docker_on_sample_advanced_epic(
    ct_ref: BIDS_FILE,
    msk_vert_ref: BIDS_FILE | None = None,
    ctd_subreg_ref: BIDS_FILE | None = None,
    cut_sacrum_ctd: bool = True,
    save_subreg_mask: bool = True,
    save_as_sourcedocker: bool = True,
    delete_tmp_after_done: bool = True,
    save_log: bool = True,
    logger: bool | Logger_Interface = False,
) -> str | bool:
    rawdata_file_origin: Path = ct_ref.file["nii.gz"]
    if isinstance(logger, bool):
        print("[#######] logger", logger)
        log_filename = "source-run-docker-single"
        logger = (
            Logger(
                path=ct_ref.get_parent("nii.gz"),
                log_filename=log_filename,
                default_verbose=logger,
            )
            if save_log
            else No_Logger()
        )

    if msk_vert_ref is not None and ctd_subreg_ref is not None and not save_subreg_mask:
        alert = "Run_docker on epic: got vert_msk, centroids and not save_subreg_mask. Skip"
        logger.print(alert)
        return alert

    sub_ses_id = str(rawdata_file_origin).split("/")[-1].split("_ct")[0]
    # the temporary sub dataset that the docker is run on
    tmp_file_dir = Path(Path.home(), "anduin", token_urlsafe(25) + "_run_docker_on_sample_advanced_epic")

    derivative_dir_origin = Path(str(rawdata_file_origin.parent).replace("rawdata", "derivatives"))

    tmp_file_dir.mkdir(exist_ok=True, parents=True)
    try:
        # make config
        config_path = Path(tmp_file_dir, "config_custom.yaml")
        f = open(config_path, "a")
        f.write(f"PROCESS_SELECTED: [{rawdata_file_origin.name}]")
        f.close()

        # the raw and derivatives below
        tmp_raw = tmp_file_dir.joinpath("rawdata")
        tmp_der = tmp_file_dir.joinpath("derivatives")
        tmp_raw.mkdir(exist_ok=True, parents=True)
        tmp_der.mkdir(exist_ok=True, parents=True)
        logger.print("[*] created tmp folder for docker:", tmp_raw)
        # Copy/symlink all files into the temp dataset
        copy_files_to_tmp([ct_ref], tmp_raw, False)
        given_der: list[BIDS_FILE] = [msk_vert_ref, ctd_subreg_ref]  # type: ignore
        given_der = [i for i in given_der if i is not None]
        copy_files_to_tmp(given_der=given_der, tmp_der=tmp_der, cut_sacrum_ctd=cut_sacrum_ctd)
        logger.print("[*] copied files into tmp folder", tmp_der)

        run_docker(tmp_file_dir.parent, log=logger, config_path=config_path)
        args = {
            "tmp_der": tmp_der,
            "sub_ses_id": sub_ses_id,
            "save_as_docker": save_as_sourcedocker,
            "derivative_dir_origin": derivative_dir_origin,
            "logger": logger,
        }
        if ctd_subreg_ref is None:  # only save centroids if they weren't given
            move_back_to_original_dir(f="seg-subreg_ctd.json", **args)
        if save_subreg_mask:  # parameter determines if subreg mask is saved
            move_back_to_original_dir(f="seg-subreg_msk.nii.gz", **args)
        if msk_vert_ref is None:  # only save vert_mask if it isn't given
            move_back_to_original_dir(f="seg-vert_msk.nii.gz", **args)

    except BaseException as e:
        if delete_tmp_after_done:
            shutil.rmtree(tmp_file_dir, ignore_errors=True)
        raise e
    if delete_tmp_after_done:
        shutil.rmtree(tmp_file_dir)
    return True


def run_docker_subreg(
    msk_vert_ref: BIDS_FILE | Path,
    ctd_subreg_ref: BIDS_FILE | None = None,
    logger=No_Logger(),
    copy_to_home: bool = False,
):
    if isinstance(msk_vert_ref, BIDS_FILE):
        msk_vert_ref = msk_vert_ref.file["nii.gz"]
    assert msk_vert_ref.exists(), msk_vert_ref

    if copy_to_home:
        tmp_file_dir = Path(Path.home(), "anduin", "run_docker_subreg_" + token_urlsafe(25))
    else:
        tmp_file_dir = Path(msk_vert_ref.parent, "run_docker_subreg_" + token_urlsafe(25))

    try:
        tmp_file_dir.mkdir(exist_ok=True, parents=True)
        # the raw and derivatives below
        tmp_raw = tmp_file_dir.joinpath("rawdata")
        tmp_der = tmp_file_dir.joinpath("derivatives")
        tmp_raw.mkdir(exist_ok=True, parents=True)
        tmp_der.mkdir(exist_ok=True, parents=True)
        logger.print("[*] created tmp folder for docker:", tmp_raw)
        # Copy/symlink all files into the temp dataset

        # "sub_ses_id" Stupid name; should be filename_stem or something like that.
        sub_ses_id = msk_vert_ref.name.rsplit("_", maxsplit=2)[0]
        raw_to = Path(tmp_raw, sub_ses_id + "_ct.nii.gz")
        if copy_to_home:
            shutil.copy(msk_vert_ref, raw_to)
        else:
            symlink(msk_vert_ref, raw_to)
        to_copy = [msk_vert_ref]
        if ctd_subreg_ref is not None:
            to_copy.append(ctd_subreg_ref)
        copy_files_to_tmp(given_der=to_copy, tmp_der=tmp_der, cut_sacrum_ctd=True)
        if ctd_subreg_ref is None:
            ctd = Path(tmp_der, sub_ses_id + "_seg-subreg_ctd.json")
            from TPTBox import calc_centroids

            calc_centroids(msk_vert_ref).save(ctd, save_hint=0)
        logger.print("[*] copied files into tmp folder", tmp_der)

        run_docker(tmp_file_dir, log=logger)
        move_back_to_original_dir(
            f="seg-subreg_msk.nii.gz",
            tmp_der=tmp_der,
            sub_ses_id=sub_ses_id,
            save_as_docker=False,
            derivative_dir_origin=msk_vert_ref.parent,
            logger=logger,
        )

    except BaseException as e:
        raise e
    finally:
        shutil.rmtree(tmp_file_dir)


def run_docker_old(
    in_path: Path,
    logger_filename: Path | None = None,
    log: Logger_Interface = No_Logger(),
    config_path: None | Path = None,
):
    """Runs the docker over a specific path

    Args:
        in_path: input directory (must contain rawdata and derivatives!)
        logger_filename: If given, creates a log file of the docker in the location: in_path + logger_filename

    Returns:
        stdout, stderr
    """
    assert platform.system() in [
        "Linux",
        "linux",
        "Darwin",
        "darwin",
    ], "Docker only works under linux and mac systems"
    log.print("Running docker...")
    command = [
        "docker",
        "run",
        "--user",
        str(os.getuid()),
        "-v",
        str(in_path) + ":/data",
    ]
    if config_path is not None:
        command += ["-v", str(config_path) + ":/config_custom.yaml"]
    command += [docker_image, docker_cmd]

    if logger_filename is not None:
        log_dir = in_path
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir.joinpath(logger_filename)
        log.print("Logging into", logger_filename, ltype=Log_Type.NEUTRAL)
        with open(log_path, "w") as output:
            process = subprocess.run(command, shell=False, stdout=output, stderr=output, check=False)
    else:
        process = subprocess.run(command, shell=False, capture_output=True, text=True, check=False)
    stdout = process.stdout
    stderr = process.stderr
    log.print(stdout, verbose=True, ignore_prefix=True) if stdout is not None else None
    log.print(stderr, verbose=True, ltype=Log_Type.FAIL) if stderr is not None and len(str(stderr)) > 2 else None
    log.print("Docker done")
    return stdout, stderr


def run_docker(in_path: Path, log: Logger_Interface = No_Logger(), config_path: None | Path = None) -> int:
    """Runs the command in the list

    Args:
        cmd_ex (list[str]): Command line argument. Instead of whitespaces us a list to separate tokens. All whitespaces must be removed or they will be part of the Token.
        ouf_file (Path, optional): Prints f'[#] Saved to {ouf_file}', if given. Defaults to None.

    Returns:
        int: Error-Code
    """
    log.print("Running docker...")
    command = ["docker", "run"]
    if platform.system() in ["Linux", "linux", "Darwin", "darwin"]:
        command += ["--user", str(os.getuid())]
    command += ["-v", str(in_path) + ":/data"]
    if config_path is not None:
        command += ["-v", str(config_path) + ":/config_custom.yaml"]
    command += [docker_image, docker_cmd]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    return_code = -1
    while True:
        assert process.stdout is not None
        time.sleep(0.001)
        output = process.stdout.readline()
        if output != "":
            log.print(output.strip(), ltype=Log_Type.DOCKER)
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                time.sleep(0.001)
                output = output.strip()
                if len(output) <= 1:
                    continue
                log.print(output, ltype=Log_Type.DOCKER)
            log.print("[#] Docker RETURN CODE", return_code, ltype=Log_Type.DOCKER)
            break
    log.print("Docker done")
    return return_code  # type:ignore


def run_exectuable(
    in_path: Path,
    log: Logger_Interface = No_Logger(),
    exectuable_path: str = "/DATA/NAS/tools/bonescreen_segmentor/",
) -> int:
    """Runs the command in the list

    Args:
        cmd_ex (list[str]): Command line argument. Instead of whitespaces us a list to separate tokens. All whitespaces must be removed or they will be part of the Token.
        ouf_file (Path, optional): Prints f'[#] Saved to {ouf_file}', if given. Defaults to None.

    Returns:
        int: Error-Code
    """
    log.print("Running docker...")
    command = [exectuable_path + "dist/" + "bonescreen-segmentor", "process"]
    # if platform.system() in ["Linux", "linux", "Darwin", "darwin"]:
    #    command += ["--user", str(os.getuid())]
    command += ["-p", str(in_path)]
    # if config_path is not None:
    #    command += ["-v", str(config_path) + ":/config_custom.yaml"]
    # command += [docker_image, docker_cmd]
    cmd_as_str = str("." + exectuable_path + "dist/" + "bonescreen-segmentor" + " process -p " + str(in_path) + "/")
    log.print(command)
    # log.print(cmd_as_str)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
    # so = os.popen(cmd_as_str).read()
    # print(so)
    # return 0
    return_code = 0
    while True:
        assert process.stdout is not None
        time.sleep(0.001)
        output = process.stdout.readline()
        if output != "":
            log.print(output.strip(), ltype=Log_Type.DOCKER)
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                time.sleep(0.001)
                output = output.strip()
                if len(output) <= 1:
                    continue
                log.print(output, ltype=Log_Type.DOCKER)
            log.print("[#] Docker RETURN CODE", return_code, ltype=Log_Type.DOCKER)
            break
    log.print("Docker done")
    return return_code  # type:ignore


def run_docker_executable_on_ctd(
    img_nii: BIDS_FILE | Path,
    ctd_subreg_ref: BIDS_FILE | Path | None = None,
    logger=No_Logger(),
    copy_to_home: bool = False,
):
    raw_dir = img_nii if isinstance(img_nii, Path) else img_nii.file["nii.gz"]
    img_nii = to_nii(img_nii)

    derivative_dir_origin = Path(str(raw_dir.parent).replace("rawdata", "derivatives_spine_r"))

    # while derivative_dir_origin.name not in ["rawdata", "derivatives_spine_r"]:
    #    derivative_dir_origin = derivative_dir_origin.parent

    if copy_to_home:
        tmp_file_dir = Path(Path.home(), "anduin", token_urlsafe(25) + "_run_executable")
    else:
        tmp_file_dir = Path(raw_dir.parent, token_urlsafe(25) + "_run_executable")

    try:
        tmp_file_dir.mkdir(exist_ok=True, parents=True)
        # the raw and derivatives below
        tmp_raw = tmp_file_dir.joinpath("rawdata")
        tmp_der = tmp_file_dir.joinpath("derivatives_spine_r")
        tmp_raw.mkdir(exist_ok=True, parents=True)
        tmp_der.mkdir(exist_ok=True, parents=True)
        logger.print("[*] created tmp folder for docker:", tmp_raw)
        # Copy/symlink all files into the temp dataset

        # "sub_ses_id" Stupid name; should be filename_stem or something like that.
        sub_ses_id = raw_dir.name.rsplit("_", maxsplit=2)[0]
        # raw_to = Path(tmp_raw, sub_ses_id + "_ct.nii.gz")
        # if copy_to_home:
        #    shutil.copy(raw_dir, raw_to)
        # else:
        #    symlink(raw_dir, raw_to)
        to_copy = []
        if ctd_subreg_ref is not None:
            to_copy.append(ctd_subreg_ref)
        copy_files_to_tmp(given_der=[raw_dir], tmp_der=tmp_raw, cut_sacrum_ctd=True)
        copy_files_to_tmp(given_der=to_copy, tmp_der=tmp_der, cut_sacrum_ctd=True)
        # if ctd_subreg_ref is None:
        #    ctd = Path(tmp_der, sub_ses_id + "_seg-subreg_ctd.json")
        #    from BIDS import calc_centroids
        #
        #            calc_centroids(msk_vert_ref).save(ctd)
        logger.print("[*] copied files into tmp folder", tmp_der)

        run_exectuable(tmp_file_dir, log=logger)
        move_back_to_original_dir_executable(
            f="seg-subreg_msk.nii.gz",
            tmp_der=tmp_der,
            sub_ses_id=sub_ses_id,
            save_as_docker=False,
            derivative_dir_origin=derivative_dir_origin,
            logger=logger,
        )
        move_back_to_original_dir_executable(
            f="seg-vert_msk.nii.gz",
            tmp_der=tmp_der,
            sub_ses_id=sub_ses_id,
            save_as_docker=False,
            derivative_dir_origin=derivative_dir_origin,
            logger=logger,
        )
        if ctd_subreg_ref is None:
            move_back_to_original_dir_executable(
                f="seg-subreg_ctd.json",
                tmp_der=tmp_der,
                sub_ses_id=sub_ses_id,
                save_as_docker=False,
                derivative_dir_origin=derivative_dir_origin,
                logger=logger,
            )

    except BaseException as e:
        raise e
    finally:
        shutil.rmtree(tmp_file_dir)


def run_executable_subreg(
    msk_vert_ref: BIDS_FILE | Path,
    ctd_subreg_ref: BIDS_FILE | None = None,
    logger=No_Logger(),
    copy_to_home: bool = False,
):
    if isinstance(msk_vert_ref, BIDS_FILE):
        msk_vert_ref = msk_vert_ref.file["nii.gz"]
    assert msk_vert_ref.exists(), msk_vert_ref

    raw_dir = msk_vert_ref
    derivative_dir_origin = Path(str(raw_dir.parent).replace("rawdata", "derivatives_spine_r"))

    if copy_to_home:
        tmp_file_dir = Path(Path.home(), "anduin", "run_docker_subreg_" + token_urlsafe(25))
    else:
        tmp_file_dir = Path(msk_vert_ref.parent, "run_docker_subreg_" + token_urlsafe(25))

    try:
        tmp_file_dir.mkdir(exist_ok=True, parents=True)
        # the raw and derivatives below
        tmp_raw = tmp_file_dir.joinpath("rawdata")
        tmp_der = tmp_file_dir.joinpath("derivatives_spine_r")
        tmp_raw.mkdir(exist_ok=True, parents=True)
        tmp_der.mkdir(exist_ok=True, parents=True)
        logger.print("[*] created tmp folder for docker:", tmp_raw)
        # Copy/symlink all files into the temp dataset

        # "sub_ses_id" Stupid name; should be filename_stem or something like that.
        sub_ses_id = msk_vert_ref.name.rsplit("_", maxsplit=2)[0]
        raw_to = Path(tmp_raw, sub_ses_id + "_ct.nii.gz")
        print("raw_to", raw_to.name)
        if copy_to_home:
            shutil.copy(msk_vert_ref, raw_to)
        else:
            symlink(msk_vert_ref, raw_to)
        to_copy = [msk_vert_ref]
        if ctd_subreg_ref is not None:
            to_copy.append(ctd_subreg_ref)
        copy_files_to_tmp(given_der=to_copy, tmp_der=tmp_der, cut_sacrum_ctd=True)
        if ctd_subreg_ref is None:
            ctd = Path(tmp_der, sub_ses_id + "_seg-subreg_ctd.json")
            from TPTBox import calc_centroids

            calc_centroids(msk_vert_ref).map_labels_(label_map_subregion={50: 0}).save(ctd, save_hint=0)
        logger.print("[*] copied files into tmp folder", tmp_der)

        run_exectuable(tmp_file_dir, log=logger)
        move_back_to_original_dir(
            f="seg-subreg_msk.nii.gz",
            tmp_der=tmp_der,
            sub_ses_id=sub_ses_id,
            save_as_docker=False,
            derivative_dir_origin=msk_vert_ref.parent,
            logger=logger,
        )

    except BaseException as e:
        raise e
    finally:
        shutil.rmtree(tmp_file_dir)


def run_executable_on_sample_advanced_epic(
    ct_ref: BIDS_FILE,
    msk_vert_ref: BIDS_FILE | None = None,
    ctd_subreg_ref: BIDS_FILE | None = None,
    cut_sacrum_ctd: bool = True,
    derivative_dir_origin: Path | None = None,
    save_subreg_mask: bool = True,
    save_as_sourcedocker: bool = True,
    save_log: bool = True,
    logger: bool | Logger_Interface = False,
    copy_to_home: bool = False,
) -> str | bool:
    rawdata_file_origin: Path = ct_ref.file["nii.gz"]
    if isinstance(logger, bool):
        print("[#######] logger", logger)
        log_filename = "source-run-docker-single"
        logger = (
            Logger(
                path=ct_ref.get_parent("nii.gz"),
                log_filename=log_filename,
                default_verbose=logger,
            )
            if save_log
            else No_Logger()
        )

    if msk_vert_ref is not None and ctd_subreg_ref is not None and not save_subreg_mask:
        alert = "Run_executable on epic: got vert_msk, centroids and not save_subreg_mask. Skip"
        logger.print(alert)
        return alert

    if copy_to_home:
        tmp_file_dir = Path(Path.home(), "anduin", "run_docker_subreg_" + token_urlsafe(25))
    else:
        tmp_file_dir = Path(msk_vert_ref.parent, "run_docker_subreg_" + token_urlsafe(25))

    # sub_ses_id = str(rawdata_file_origin).split("/")[-1].split("_ct")[0]
    # the temporary sub dataset that the docker is run on

    if derivative_dir_origin is None:
        derivative_dir_origin = Path(str(rawdata_file_origin.parent).replace("rawdata", "derivatives_spine_r"))

    try:
        tmp_file_dir.mkdir(exist_ok=True, parents=True)
        # the raw and derivatives below
        tmp_raw = tmp_file_dir.joinpath("rawdata")
        tmp_der = tmp_file_dir.joinpath("derivatives_spine_r")
        tmp_raw.mkdir(exist_ok=True, parents=True)
        tmp_der.mkdir(exist_ok=True, parents=True)
        logger.print("[*] created tmp folder for executable:", tmp_raw)

        sub_ses_id = rawdata_file_origin.name.rsplit("_", maxsplit=1)[0]
        raw_to = Path(tmp_raw, sub_ses_id + "_ct.nii.gz")
        print("raw_to", raw_to.name)

        if copy_to_home:
            shutil.copy(rawdata_file_origin, raw_to)
        else:
            symlink(rawdata_file_origin, raw_to)

        ct_nii = ct_ref.open_nii()
        zms = ct_nii.zoom
        # Copy/symlink all files into the temp dataset
        # copy_files_to_tmp([ct_ref], tmp_raw, False)
        given_der: list[BIDS_FILE] = [msk_vert_ref, ctd_subreg_ref]  # type: ignore
        given_der = [i for i in given_der if i is not None]
        copy_files_to_tmp(given_der=given_der, tmp_der=tmp_der, cut_sacrum_ctd=cut_sacrum_ctd, zoom_image=ct_nii)
        if ctd_subreg_ref is None:
            ctd = Path(tmp_der, sub_ses_id + "_seg-subreg_ctd.json")
            from TPTBox import calc_centroids

            calc_centroids(msk_vert_ref).map_labels_(label_map_subregion={50: 0}).save(ctd, save_hint=0)
        logger.print("[*] copied files into tmp folder", tmp_der)

        run_exectuable(tmp_file_dir, log=logger)
        args = {
            "tmp_der": tmp_der,
            "sub_ses_id": sub_ses_id,
            "save_as_docker": save_as_sourcedocker,
            "derivative_dir_origin": derivative_dir_origin,
            "logger": logger,
        }
        if ctd_subreg_ref is None:  # only save centroids if they weren't given
            move_back_to_original_dir(f="seg-subreg_ctd.json", **args)
        if save_subreg_mask:  # parameter determines if subreg mask is saved
            move_back_to_original_dir(f="seg-subreg_msk.nii.gz", **args)
        if msk_vert_ref is None:  # only save vert_mask if it isn't given
            move_back_to_original_dir(f="seg-vert_msk.nii.gz", **args)

    except BaseException as e:
        raise e
    finally:
        shutil.rmtree(tmp_file_dir)
    # if delete_tmp_after_done:
    #    shutil.rmtree(tmp_file_dir)
    return True


def copy_files_to_tmp(given_der: list[BIDS_FILE | Path], tmp_der: Path, cut_sacrum_ctd: bool, zoom_image: Image_Reference | None = None):
    for i in given_der:
        if isinstance(i, BIDS_FILE):
            if "nii.gz" in i.file:
                src_path: Path = i.file["nii.gz"]
                shutil.copy(src_path, tmp_der.joinpath(src_path.name))
                print(src_path, "->", tmp_der.joinpath(src_path.name))
            else:
                src_path = i.file["json"]
                ctd = i.open_ctd(zoom_image).map_labels_(label_map_subregion={50: 0})
                to_remove = [26, 29, 30, 31, 32, 33, 34, 35, 27] if cut_sacrum_ctd else []
                for r in to_remove:
                    ctd.remove_centroid(r)
                ctd.save(tmp_der.joinpath(src_path.name), verbose=False, save_hint=0)
                print("centroids to", tmp_der.joinpath(src_path.name))
        else:
            shutil.copy(i, tmp_der.joinpath(i.name))
            # print(i, "->", tmp_der.joinpath(i.name))


def move_back_to_original_dir(
    f: str,
    tmp_der: Path,
    sub_ses_id: str,
    save_as_docker: bool,
    derivative_dir_origin: Path,
    logger: Logger_Interface,
) -> bool:
    rename = None
    if save_as_docker:
        rename_o = "_" + str(f).split("_")[-1].split(".")[0] + "." + str(f).split(".")[1]
        rename_t = rename_o.replace("_", "_source-docker_")
        rename = (rename_o, rename_t)
    filepath = tmp_der.joinpath(sub_ses_id + "_" + f)

    if not os.path.exists(filepath):
        logger.print(
            filepath,
            f"does not exist, will skip. got {f}, tmp_der: {tmp_der}, sub_ses_id: {sub_ses_id}",
            ltype=Log_Type.FAIL,
        )
        return False
    p = str(filepath)
    if rename is not None:
        p = p.replace(rename[0], rename[1])
        os.rename(filepath, p)
        logger.print("[*] renamed", filepath.name, "->", Path(p).name)
    filename = Path(p).name
    to_path = derivative_dir_origin.joinpath(filename)
    to_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(p, to_path)
    logger.print(
        "[*] copied",
        "~" + p.split(str(derivative_dir_origin.parent.parent))[-1],
        "->",
        "~" + str(to_path).split(str(derivative_dir_origin.parent.parent))[-1],
    )
    return True


def move_back_to_original_dir_executable(
    f: str,
    tmp_der: Path,
    sub_ses_id: str,
    save_as_docker: bool,
    derivative_dir_origin: Path,
    logger: Logger_Interface,
) -> bool:
    rename = None
    if save_as_docker:
        rename_o = "_" + str(f).split("_")[-1].split(".")[0] + "." + str(f).split(".")[1]
        rename_t = rename_o.replace("_", "_source-docker_")
        rename = (rename_o, rename_t)

    from itertools import chain

    paths = sorted(list(chain(list(Path(f"{tmp_der}").glob(f"*{f}")))))
    if len(paths) == 1:
        filepath = paths[0]
    else:
        filepath = tmp_der.joinpath(sub_ses_id + "_" + f)

    if not os.path.exists(filepath):
        logger.print(
            filepath,
            f"does not exist, will skip. got {f}, tmp_der: {tmp_der}, sub_ses_id: {sub_ses_id}",
            ltype=Log_Type.FAIL,
        )
        return False
    p = str(filepath)
    if rename is not None:
        p = p.replace(rename[0], rename[1])
        os.rename(filepath, p)
        logger.print("[*] renamed", filepath.name, "->", Path(p).name)
    filename = Path(p).name
    to_path = derivative_dir_origin.joinpath(filename)
    to_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(p, to_path)
    logger.print(
        "[*] copied",
        "~" + p.split(str(derivative_dir_origin.parent.parent))[-1],
        "->",
        "~" + str(to_path).split(str(derivative_dir_origin.parent.parent))[-1],
    )
    return True


def symlink(src: Path | str, symlink: Path | str):
    """Creates a symlink pointing at src

    Args:
        src: the file/dir the symlink points at
        symlink: the path (including filename) where the symlink file should be created

    Returns:

    """
    assert not os.path.exists(symlink), f"symlink file already exists, got {symlink} -> {src}"
    assert os.path.exists(src), f"src file does not exist, got {symlink} -> {src}"
    subprocess.call(["ln", "-s", "-r", src, symlink])
