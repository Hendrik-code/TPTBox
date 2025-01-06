import subprocess
import time
from pathlib import Path

from TPTBox import BIDS_FILE, NII, Print_Logger

logger = Print_Logger()


def injection_function(seg_nii: NII):
    # TODO do something with semantic mask
    return seg_nii


def run_spineps_single(
    file_path: str | Path | BIDS_FILE,
    dataset=None,
    model_semantic="t2w",
    model_instance="instance",
    derivative_name="derivative",
    override_semantic=False,
    override_instance=False,
    lambda_semantic=None,
    save_debug_data=False,
    verbose=False,
    save_raw=False,
    ignore_compatibility_issues=False,
    use_cpu=False,
    **args,
):
    from spineps import get_instance_model, get_semantic_model, process_img_nii

    if not isinstance(file_path, BIDS_FILE):
        file_path = Path(file_path)
        file_path = BIDS_FILE(file_path, file_path.parent if dataset is None else dataset)
    elif dataset is not None:
        file_path.dataset = dataset
    output_paths, errcode = process_img_nii(
        img_ref=file_path,
        derivative_name=derivative_name,
        model_semantic=get_semantic_model(model_semantic, use_cpu=use_cpu),
        model_instance=get_instance_model(model_instance, use_cpu=use_cpu),
        override_semantic=override_semantic,
        override_instance=override_instance,
        lambda_semantic=lambda_semantic,
        save_debug_data=save_debug_data,
        verbose=verbose,
        save_raw=save_raw,
        ignore_compatibility_issues=ignore_compatibility_issues,
        **args,
    )
    return output_paths


def run_spineps_all(nii_dataset: Path | str):
    for model_semantic in ["t2w", "t1w", "vibe"]:
        command = [
            "spineps",
            "dataset",
            "-directory",
            str(nii_dataset),
            "-raw_name",
            "rawdata",
            "-der_name",
            "derivatives",
            "-model_semantic",
            model_semantic,
            "-model_instance",
            "instance",
        ]
        try:
            # Execute the command and capture output
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Model {model_semantic} processing complete:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during processing {model_semantic}:\n", e.stderr)
