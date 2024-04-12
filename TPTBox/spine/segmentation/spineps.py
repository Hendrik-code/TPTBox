import time
from pathlib import Path

from TPTBox import BIDS_FILE, NII, Print_Logger

logger = Print_Logger()


def injection_function(seg_nii: NII):
    # TODO do something with semantic mask
    return seg_nii


model_semantic = None
model_instance = None


def segment_sagittal_image(
    ref: str | Path,
    dataset: str | Path,
    der="derivatives",
    instance_filter=injection_function,
    verbose=False,
    inst_vertebra="Inst_Vertebra",
    t2w_segmentor="T2w_Segmentor",
):
    global model_semantic, model_instance  # noqa: PLW0603
    from spineps.models import get_instance_model, get_semantic_model
    from spineps.seg_run import ErrCode, process_img_nii

    try:
        model_semantic = get_semantic_model(t2w_segmentor) if model_semantic is None else model_semantic
        model_instance = get_instance_model(inst_vertebra) if model_instance is None else model_instance
    except Exception:
        from spineps.models import modelid2folder_instance, modelid2folder_semantic

        model_semantic = get_semantic_model(next(modelid2folder_semantic().keys().__iter__())) if model_semantic is None else model_semantic
        model_instance = get_instance_model(next(modelid2folder_instance().keys().__iter__())) if model_instance is None else model_instance

    start_time = time.perf_counter()
    # Call to the pipeline
    output_paths, errcode = process_img_nii(
        img_ref=BIDS_FILE(ref, dataset),
        derivative_name=der,
        model_semantic=model_semantic,
        model_instance=model_instance,
        override_semantic=False,
        override_instance=False,
        lambda_semantic=instance_filter,
        save_debug_data=False,
        verbose=verbose,
    )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logger.print(f"Inference time is: {execution_time}", verbose=verbose)
    if errcode not in [ErrCode.OK, ErrCode.ALL_DONE]:
        logger.print(f"Pipeline threw errorcode {errcode}")
    logger.print(f"\nExecution times:{execution_time}")

    return (output_paths["out_vert"], output_paths["out_spine"], output_paths["out_ctd"])
