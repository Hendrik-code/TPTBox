from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import GPUtil
from tqdm import tqdm

from TPTBox import BIDS_FILE, NII, POI, to_nii
from TPTBox.segmentation.nnUnet_utils.inference_api import load_inf_model, run_inference
from TPTBox.segmentation.oar_segmentator.map_to_binary import class_map, class_map_9_parts, except_labels_combine, map_taskid_to_partname

class_map_inv = {v: k for k, v in class_map.items()}


def save_resampled_segmentation(seg_nii: NII, in_file: BIDS_FILE, parent, org: NII | POI, idx):
    """Helper function to resample and save NIfTI file."""
    out_path = in_file.get_changed_path("nii.gz", "msk", parent=parent, info={"seg": f"oar-{idx}"}, non_strict_mode=True)
    seg_nii.resample_from_to(org, verbose=False, mode="nearest").save(out_path)


def run_oar_segmentor(
    ct_path: Path | str | BIDS_FILE,
    dataset: Path | str | None = None,
    oar_path="/home/fercho/code/oar_segmentator/models/nnunet/results/nnUNet/3d_fullres/",
    parent="derivatives",
    gpu=None,
    override=False,
):
    ## Hard coded info ##
    zoom = 1.5
    orientation = ("R", "A", "S")
    ####
    if isinstance(ct_path, BIDS_FILE):
        in_file = ct_path
    else:
        if dataset is None:
            dataset = Path(ct_path).parent
        in_file = BIDS_FILE(ct_path, dataset)
    out_path_combined = in_file.get_changed_path("nii.gz", "msk", parent=parent, info={"seg": "oar-combined"}, non_strict_mode=True)
    if out_path_combined.exists() and not override:
        print("skip", out_path_combined.name, "    ", end="\r")
        return
    org = to_nii(in_file)
    print("resample                                                                        ")
    input_nii = org.rescale((zoom, zoom, zoom), mode="nearest").reorient(orientation)
    org = (org.shape, org.affine, org.zoom)
    segs: dict[int, NII] = {}
    futures = []
    # Create ThreadPoolExecutor for parallel saving
    print("start")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for idx in tqdm(range(251, 260), desc="Predict oar segmentation"):
            # Suppress stdout and stderr for run_inference
            nnunet_path = next(next(iter(Path(oar_path).glob(f"*{idx}*"))).glob("*__nnUNetPlans*"))
            nnunet = load_inf_model(nnunet_path, allow_non_final=True, use_folds=(0,), gpu=gpu)
            seg_nii, _, _ = run_inference(input_nii, nnunet, logits=False)
            segs[idx] = seg_nii
            # Submit the save task to the thread pool
            futures.append(executor.submit(save_resampled_segmentation, seg_nii, in_file, parent, org, idx))
        # Wait for all save tasks to complete
        for future in as_completed(futures):
            future.result()  # Ensure any exceptions in threads are raised
    seg_combined = segs[251] * 0
    for tid in range(251, 260):
        seg = segs[tid]
        for jdx, class_name in class_map_9_parts[map_taskid_to_partname[tid]].items():
            if any(class_name in s for s in except_labels_combine):
                continue
            seg_combined[seg == jdx] = class_map_inv[class_name]
    seg_combined.resample_from_to(org, verbose=False, mode="nearest").save(out_path_combined)


def check_gpu_memory(gpu_id, threshold=50):
    """Check the GPU memory utilization and return True if usage exceeds threshold."""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.id == gpu_id:
            return gpu.memoryUtil * 100 > threshold
    return False


def run_oar_segmentor_in_parallel(dataset, parents: Sequence[str] = ("rawdata",), gpu_id=3, threshold=50, max_workers=16, override=False):
    """Run the OAR segmentation in parallel and pause when GPU memory usage exceeds the threshold."""
    from TPTBox import BIDS_Global_info

    bgi = BIDS_Global_info([dataset], parents=parents)

    futures = []

    # ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _name, subject in bgi.enumerate_subjects():
            q = subject.new_query(flatten=True)
            q.filter_filetype("nii.gz")
            q.filter_format("ct")

            for i in q.loop_list():
                # Check GPU memory usage and pause if above threshold
                while check_gpu_memory(gpu_id, threshold):
                    print(f"GPU memory usage exceeded {threshold}%. Pausing submission...")
                    time.sleep(10)  # Pause for 10 seconds before checking again

                # Submit run_oar_segmentor task to the executor
                futures.append(executor.submit(run_oar_segmentor, i, gpu=gpu_id, override=override))

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions encountered
            except Exception as e:
                print(f"Error in execution: {e}")


if __name__ == "__main__":
    # Example usage
    bgi = "/DATA/NAS/datasets_processed/CT_spine/dataset-shockroom-without-fx/"

    run_oar_segmentor_in_parallel(bgi, ("rawdata_fixed",), gpu_id=0, threshold=50, max_workers=16, override=False)
