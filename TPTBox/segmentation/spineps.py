import subprocess
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
    model_labeling: str | None = "t2w_labeling",
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

    try:
        from spineps.get_models import get_labeling_model

        label = {"model_labeling": get_labeling_model(model_labeling, use_cpu=use_cpu)}

    except Exception:
        label = {}  # TODO remove when spineps has officially adopted labeling

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
        **label,
        override_semantic=override_semantic,
        override_instance=override_instance,
        lambda_semantic=lambda_semantic,
        save_debug_data=save_debug_data,
        verbose=verbose,
        save_raw=save_raw,
        ignore_compatibility_issues=ignore_compatibility_issues,
        ignore_bids_filter=ignore_compatibility_issues,
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


def _run_spineps_internal(
    image_nii: NII | str | Path,
    model_path: str | Path,
    proc_pad_size: int = 4,
    step_size: float = 0.4,
    proc_crop_input: bool = True,
    proc_fillholes: bool = True,
    verbose: bool = False,
    outpath: str | Path | None = None,
    override=False,
    # gpu=0,
):
    # TODO select GPU
    from spineps.get_models import get_actual_model
    from spineps.seg_model import OutputType, Segmentation_Model

    from TPTBox import to_nii

    if not override and outpath is not None and outpath.exists():
        return to_nii(outpath)

    image_nii = to_nii(image_nii, False)
    model: Segmentation_Model = get_actual_model(model_path)
    model.load()
    from torch import device

    # model.predictor.network.to(device("cuda:0"))
    # model.predictor.device = device("cuda:0")
    orig_img_nii = image_nii.copy()
    logger.print("Input", orig_img_nii, verbose=verbose)
    image_nii.reorient_(verbose=verbose)
    crop = None
    if proc_crop_input:
        crop = image_nii.compute_crop(dist=2)
        image_nii.apply_crop_(crop)
        logger.print(f"Cropped down to {image_nii.shape}", verbose=verbose)
    if proc_pad_size > 0:
        image_nii = image_nii.pad_to(tuple(image_nii.shape[i] + (2 * proc_pad_size) for i in range(3)))
        # arr = image_nii.get_array()
        # arr = np.pad(arr, proc_pad_size, mode="edge")
        # image_nii.set_array_(arr)
        logger.print(f"Padded up to {image_nii.shape}", verbose=verbose)
    seg_nii_modelres = model.segment_scan(
        image_nii,
        pad_size=0,
        step_size=step_size,
        resample_output_to_input_space=False,
        verbose=verbose,
    )[OutputType.seg]
    if len(seg_nii_modelres.volumes()) == 0:
        return None
    # resample back and unpad
    logger.print("Before backsample", seg_nii_modelres, verbose=verbose)
    seg_nii = seg_nii_modelres.resample_from_to(orig_img_nii)
    logger.print("After backsample", seg_nii, verbose=verbose)
    if proc_fillholes:
        for i in seg_nii.unique():
            seg_nii.fill_holes_(labels=i, verbose=verbose)  # inferior direction (axial)
    orig_img_nii.assert_affine(shape=seg_nii.shape, orientation=seg_nii.orientation, zoom=seg_nii.zoom)
    # seg_nii.reorient(ori, verbose=verbose)
    seg_nii.affine = orig_img_nii.affine
    seg_nii.origin = orig_img_nii.origin
    if outpath is not None:
        seg_nii.save(outpath)
    return seg_nii  # , seg_nii_modelresa
