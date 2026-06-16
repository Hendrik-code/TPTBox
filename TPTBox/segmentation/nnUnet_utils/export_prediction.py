# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
from __future__ import annotations

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import \
    bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import (load_json,
                                                                  save_pickle)
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from tqdm import tqdm

from TPTBox import Print_Logger
from TPTBox.segmentation.nnUnet_utils.plans_handler import (
    ConfigurationManager, PlansManager)

SAFETY_FACTOR = 0.5  # only use 50% of VRAM


def _argmax_with_gpu_fallback(predicted_logits: torch.Tensor | np.ndarray, device: torch.device, chunk_size: int = 64,logger=Print_Logger()) -> np.ndarray:
    """Computes argmax(0).

    Tiered argmax:
      1. Full array on GPU
      2. Chunked on GPU (if full array doesn't fit)
      3. Chunked on CPU (if even a single chunk doesn't fit)
    """
    from TPTBox.segmentation.nnUnet_utils.predictor import empty_cache

    empty_cache(device)

    def _get_free_vram(device: torch.device) -> int:
        try:
            """Returns free VRAM in bytes."""
            free, _ = torch.cuda.mem_get_info(device)
            return int(free * SAFETY_FACTOR)
        except Exception:
            return 0

    def _array_bytes(shape: tuple, dtype: torch.dtype = torch.float16) -> int:
        n_elements = 1
        for s in shape:
            n_elements *= s
        return n_elements * torch.finfo(dtype).bits // 8

    def _to_cpu_tensor(arr: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        return arr.cpu()

    def _chunked_argmax_gpu(t: torch.Tensor, device: torch.device) -> np.ndarray:
        X = t.shape[1]
        out = np.empty(t.shape[1:], dtype=np.int16)
        for x in tqdm(range(0, X, chunk_size), "argmax gpu"):
            chunk = t[:, x : x + chunk_size].to(device)
            out[x : x + chunk_size] = torch.argmax(chunk, dim=0).cpu().numpy()
        del chunk
        empty_cache(device)
        return out

    def _chunked_argmax_cpu(t: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = t.numpy() if isinstance(t, torch.Tensor) else t
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        X = arr.shape[1]
        out = np.empty(arr.shape[1:], dtype=np.int16)
        for x in tqdm(range(0, X, chunk_size), "argmax cpu"):
            out[x : x + chunk_size] = arr[:, x : x + chunk_size].argmax(0)
        return out

    t = _to_cpu_tensor(predicted_logits)

    if device is None or not torch.cuda.is_available():
        return _chunked_argmax_cpu(t)

    full_bytes = _array_bytes(t.shape)
    free_vram = _get_free_vram(device)

    logger.on_debug(f"[argmax] array: {full_bytes / 1e6:.1f} MB, VRAM: {free_vram / 1e6:.1f} MB")

    # Tier 1: full GPU
    if full_bytes <= free_vram or device.type == "mps":
        try:
            return torch.argmax(t.to(device), dim=0).cpu().numpy().astype(np.int16)
        except torch.cuda.OutOfMemoryError:
            logger.on_fail("[argmax] full GPU OOM despite estimate, trying chunked GPU")
            empty_cache(device)
        except Exception as e:
            logger.on_fail(e)
            empty_cache(device)

    for i in range(10):
        chunk_shape = (t.shape[0], min(max(int(chunk_size / 2**i), 1), t.shape[1]), *t.shape[2:])
        chunk_bytes = _array_bytes(chunk_shape)
        if chunk_bytes <= free_vram:
            chunk_size = max(int(chunk_size / 2**i), 1)
            break
    logger.on_debug(f"[argmax] array chunk: {chunk_bytes / 1e6:.1f} MB, VRAM: {free_vram / 1e6:.1f} MB, {chunk_size=}")

    # Tier 2: chunked GPU
    if chunk_bytes <= free_vram:
        logger.on_log("[argmax] using chunked GPU")
        try:
            return _chunked_argmax_gpu(t, device)
        except torch.cuda.OutOfMemoryError:
            logger.on_fail("[argmax] chunked GPU OOM despite estimate, falling back to CPU")
            empty_cache(device)
    else:
        logger.on_debug("[argmax] chunk too large for VRAM, falling back to CPU")

    # Tier 3: chunked CPU
    return _chunked_argmax_cpu(t)


@torch.inference_mode()
def convert_probabilities_to_segmentation(self, predicted_probabilities: np.ndarray | torch.Tensor, device, chunk_size=64,logger=Print_Logger()) -> np.ndarray:
    """Assumes that inference_nonlinearity was already applied!

    predicted_probabilities has to have shape (c, x, y(, z)) where c is the number of classes/regions
    """
    if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
        raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor, got {type(predicted_probabilities)}")  # noqa: TRY004

    if self.has_regions:
        assert self.regions_class_order is not None, "if region-based training is requested then you need to define regions_class_order!"
        # check correct number of outputs
    assert predicted_probabilities.shape[0] == self.num_segmentation_heads, (
        f"unexpected number of channels in predicted_probabilities. Expected {self.num_segmentation_heads}, "
        f"got {predicted_probabilities.shape[0]}. Remember that predicted_probabilities should have shape "
        f"(c, x, y(, z))."
    )
    if self.has_regions:
        if isinstance(predicted_probabilities, np.ndarray):
            segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.uint16)
        else:
            # no uint16 in torch
            segmentation = torch.zeros(
                predicted_probabilities.shape[1:],
                dtype=torch.int16,
                device=predicted_probabilities.device,
            )
        for i, c in enumerate(self.regions_class_order):
            segmentation[predicted_probabilities[i] > 0.5] = c
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
    else:
        # Issensee is no longer right when saying "numpy is faster than torch" newer torch versions no longer have this issue, on GPU we even get a 20x improvment. :facepalm:
        segmentation = _argmax_with_gpu_fallback(predicted_probabilities, device, chunk_size=chunk_size,logger=logger)

    return segmentation


def convert_predicted_logits_to_segmentation_with_correct_shape(
    predicted_logits: torch.Tensor | np.ndarray,
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    label_manager: LabelManager,
    properties_dict: dict,
    return_probabilities: bool = False,
    num_threads_torch: int = 8,
    device=None,
    logger=Print_Logger()
) -> np.ndarray:
    """Revert all preprocessing steps and return a segmentation in the original image space.

    Performs (in reverse order): resampling from network output spacing back to
    the cropped shape, argmax / softmax conversion, padding to the pre-crop
    shape, and axis transposition to restore the original orientation.

    Args:
        predicted_logits: Raw network output with shape
            ``(num_classes, X', Y', Z')`` in the *preprocessed* space.
        plans_manager: Plans manager carrying transpose information.
        configuration_manager: Configuration manager carrying spacing and
            resampling function references.
        label_manager: Label manager used for converting logits to a
            segmentation map.
        properties_dict: Case properties dict as updated by
            :meth:`DefaultPreprocessor.run_case_npy` (contains cropping bbox,
            pre-crop shape, and original spacing).
        return_probabilities: Reserved for future use. Raises
            :class:`NotImplementedError` if ``True``.
        num_threads_torch: Number of threads used by PyTorch during resampling.

    Returns:
        Integer segmentation array with dtype ``np.uint8`` or ``np.uint16``
        in the original image space (before any preprocessing).

    Raises:
        NotImplementedError: If ``return_probabilities`` is ``True``.
    """
    if return_probabilities:
        raise NotImplementedError()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)
    # resample to original shape
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [properties_dict["spacing"][0], *configuration_manager.spacing]
    )
    predicted_logits = configuration_manager.resampling_fn_probabilities(
        predicted_logits, properties_dict["shape_after_cropping_and_before_resampling"], current_spacing, properties_dict["spacing"]
    )
    if label_manager.has_regions or return_probabilities:
        # Softmax does not change when we use argmax in the next step
        predicted_logits = label_manager.apply_inference_nonlin(predicted_logits)
    # segmentation: np.ndarray = label_manager.convert_probabilities_to_segmentation(predicted_logits)  # type: ignore
    segmentation: np.ndarray = convert_probabilities_to_segmentation(label_manager, predicted_logits, device,logger=logger)
    segmentation = segmentation.astype(np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    del predicted_logits
    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(
        properties_dict["shape_before_cropping"],
        dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16,
    )
    slicer = bounding_box_to_slice(properties_dict["bbox_used_for_cropping"])
    segmentation_reverted_cropping[slicer] = segmentation
    segmentation = segmentation_reverted_cropping

    # revert transpose
    segmentation = segmentation.transpose(plans_manager.transpose_backward)
    logger.print(segmentation.shape)
    # if return_probabilities:
    #    raise NotImplementedError()
    #    # revert cropping
    #    try:
    #        predicted_probabilities = predicted_logits  # noqa: F821
    #        predicted_probabilities = label_manager.revert_cropping_on_probabilities(  # type: ignore
    #            predicted_probabilities,
    #            properties_dict["bbox_used_for_cropping"],
    #            properties_dict["shape_before_cropping"],
    #        )
    #    except Exception:
    #        predicted_probabilities = label_manager.revert_cropping(  # type: ignore
    #            predicted_probabilities.numpy() if isinstance(predicted_probabilities, torch.Tensor) else predicted_probabilities,
    #            properties_dict["bbox_used_for_cropping"],
    #            properties_dict["shape_before_cropping"],
    #        )
    #    if isinstance(predicted_probabilities, torch.Tensor):
    #        predicted_probabilities = predicted_probabilities.cpu().numpy()
    #    # revert transpose
    #    predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
    #    torch.set_num_threads(old_threads)
    #    return segmentation, predicted_probabilities
    # else:
    torch.set_num_threads(old_threads)
    return segmentation


def export_prediction_from_logits(
    predicted_array_or_file: np.ndarray | torch.Tensor,
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: dict | str,
    output_file_truncated: str,
    save_probabilities: bool = False,
) -> None:
    """Convert logits to a segmentation and write it to disk.

    Delegates shape reversion to
    :func:`convert_predicted_logits_to_segmentation_with_correct_shape` and
    then writes the result using the image reader/writer defined in the plans.

    Args:
        predicted_array_or_file: Raw logits array/tensor in the preprocessed
            image space.
        properties_dict: Case properties dict containing cropping and resampling
            metadata.
        configuration_manager: Configuration manager for the active plans
            configuration.
        plans_manager: Plans manager (provides image writer class and transpose
            directions).
        dataset_json_dict_or_file: Either a parsed ``dataset.json`` dict or a
            path to the file.
        output_file_truncated: Output path **without** file extension. The
            correct extension is appended automatically from ``dataset_json``.
        save_probabilities: If ``True``, also save softmax probability maps as
            ``.npz``. Currently raises :class:`NotImplementedError` inside the
            conversion step.
    """
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)  # type: ignore
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + ".npz", probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + ".pkl")
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    out_fname = output_file_truncated + dataset_json_dict_or_file["file_ending"]  # type: ignore
    rw.write_seg(segmentation_final, out_fname, properties_dict)  # type: ignore
    print(f"Saved Segmentation into {out_fname}")
