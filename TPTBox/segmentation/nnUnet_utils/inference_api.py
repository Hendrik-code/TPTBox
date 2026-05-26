from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch

from TPTBox import NII, Log_Type, No_Logger
from TPTBox.core import sitk_utils

from .predictor import nnUNetPredictor

logger = No_Logger()
logger.prefix = "API"

_interop = False


# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
def load_inf_model(
    model_folder: str | Path,
    step_size: float = 0.5,
    ddevice: str = "cuda",
    use_folds: tuple[str | int, ...] | None = None,
    init_threads: bool = True,
    allow_non_final: bool = True,
    inference_augmentation: bool = False,
    use_gaussian: bool = True,
    verbose: bool = False,
    gpu: int | None = None,
    memory_base: int = 5000,
    memory_factor: int = 160,
    memory_max: int = 160000,
    wait_till_gpu_percent_is_free: float = 0.3,
) -> nnUNetPredictor:
    """Load and initialise an nnU-Net model predictor from a trained model folder.

    Args:
        model_folder: Path to the nnU-Net result folder containing ``fold_*``
            sub-directories.
        step_size: Sliding-window step size as a fraction of the patch size.
            Larger values are faster but may reduce accuracy.  Must be in (0, 1].
        ddevice: Inference device: ``"cuda"`` (GPU), ``"cpu"``, or ``"mps"``
            (Apple Silicon).  Do NOT use this to select a GPU index.
        use_folds: Tuple of fold indices or fold names to load.  Loads all
            available folds when ``None``.
        init_threads: If True, configure PyTorch thread counts optimally for the
            selected device.
        allow_non_final: If True, fall back to ``checkpoint_best.pth`` when
            ``checkpoint_final.pth`` is not found.
        inference_augmentation: If True, enable test-time mirroring augmentation.
        use_gaussian: If True, apply Gaussian weighting in the sliding window.
        verbose: If True, print progress information during model initialisation.
        gpu: GPU device index forwarded to the predictor.  ``None`` defaults to 0.
        memory_base: Base GPU memory reservation in MB (default 5 000 MB = 5 GB).
        memory_factor: Per-voxel memory scaling factor.  The formula is
            ``prod(shape) * memory_factor / 1000`` MB; 160 corresponds to ~30 GB
            for a 512³ volume.
        memory_max: Maximum GPU memory cap in MB (default 160 000 MB = 160 GB).
        wait_till_gpu_percent_is_free: Fraction of GPU memory that must be free
            before inference is started.

    Returns:
        Initialised ``nnUNetPredictor`` ready for inference.
    """
    if isinstance(model_folder, str):
        model_folder = Path(model_folder)
    if ddevice == "cpu":
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count()) if init_threads else None
        device = torch.device("cpu")
    elif ddevice == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        try:
            torch.set_num_threads(1) if init_threads else None
            global _interop  # noqa: PLW0603
            if not _interop:
                torch.set_num_interop_threads(1) if init_threads else None
                _interop = True
        except Exception as e:
            print(e)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    assert model_folder.exists(), f"model-folder not found: got path {model_folder}"

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=use_gaussian,
        use_mirroring=inference_augmentation,  # <- mirroring augmentation!
        perform_everything_on_gpu=ddevice != "cpu",
        device=device,
        verbose=verbose,
        verbose_preprocessing=False,
        cuda_id=0 if gpu is None else gpu,
        memory_base=memory_base,
        memory_factor=memory_factor,
        memory_max=memory_max,
        wait_till_gpu_percent_is_free=wait_till_gpu_percent_is_free,
    )
    check_name = "checkpoint_final.pth"  # if not allow_non_final else "checkpoint_best.pth"
    try:
        predictor.initialize_from_trained_model_folder(str(model_folder), checkpoint_name=check_name, use_folds=use_folds)
    except FileNotFoundError as e:
        if allow_non_final:
            try:
                predictor.initialize_from_trained_model_folder(
                    str(model_folder),
                    checkpoint_name="checkpoint_best.pth",
                    use_folds=use_folds,
                )
                logger.print("Checkpoint final not found, will load from best instead", Log_Type.WARNING)
            except Exception:
                raise e  # noqa: B904
        else:
            raise e  # noqa: TRY201
    logger.print(f"Inference Model loaded from {model_folder}") if verbose else None
    return predictor


def run_inference(
    input_nii: str | NII | list[NII],
    predictor: nnUNetPredictor,
    reorient_PIR: bool = False,  # noqa: N803
    logits: bool = False,
    verbose: bool = False,  # noqa: ARG001
) -> tuple[NII, NII | None, np.ndarray | None]:
    """Run nnU-Net inference on a single image or list of images (multi-channel).

    Args:
        input_nii: Input as a path to a ``.nii.gz`` file, a ``NII`` object, or a
            list of ``NII`` objects for multi-channel models.
        predictor: Loaded ``nnUNetPredictor`` as returned by :func:`load_inf_model`.
        reorient_PIR: If True, reorient each input image to PIR orientation before
            passing it to the model.
        logits: If True, return raw softmax logits.  Currently not implemented and
            will raise ``NotImplementedError``.
        verbose: Unused; reserved for future logging support.

    Raises:
        NotImplementedError: If *logits* is True.
        AssertionError: If *input_nii* is not a supported type, or if the output
            shape does not match the input shape.

    Returns:
        A tuple of (segmentation_NII, uncertainty_map_or_None, softmax_logits_or_None).
    """
    if logits:
        raise NotImplementedError("logits=True")
    if isinstance(input_nii, str):
        assert input_nii.endswith(".nii.gz"), f"input file is not a .nii.gz! Got {input_nii}"
        input_nii = NII.load(input_nii, seg=False)

    assert isinstance(input_nii, (NII, list)), f"input must be a NII or str or list[NII], got {type(input_nii)}"
    if isinstance(input_nii, NII):
        input_nii = [input_nii]
    orientation = input_nii[0].orientation

    img_arrs = []
    # Prepare for nnUNet behavior
    for i in input_nii:
        if reorient_PIR:
            i.reorient_()
        a = i.get_array().astype(np.float16)
        nii_img_converted = np.transpose(a, axes=a.ndim - 1 - np.arange(a.ndim))[np.newaxis, :]
        img_arrs.append(nii_img_converted)
    try:
        img = np.vstack(img_arrs)
    except Exception:
        print("could not stack images; shapes=", [a.shape for a in img_arrs])
        raise
    props = {"spacing": i.zoom[::-1]}  # PIR
    out = predictor.predict_single_npy_array(img, props, save_or_return_probabilities=False)
    segmentation: np.ndarray = out  # type: ignore
    softmax_logits = None
    segmentation = np.transpose(segmentation, axes=segmentation.ndim - 1 - np.arange(segmentation.ndim))
    assert segmentation.shape == input_nii[0].shape, (segmentation.shape, input_nii[0].shape)
    seg_nii = input_nii[0].set_array(segmentation.astype(np.uint8), seg=True)
    seg_nii.reorient_(orientation, verbose=False)
    return seg_nii, None, softmax_logits


# def predict_single_npy_array(predictor: nnUNetPredictor, img, props, logits, rescale):
#    return predictor.predict_single_npy_array(img, props, save_or_return_probabilities=logits, rescale=rescale)
#
#    def fun(x):
#        return predictor.predict_single_npy_array(x, props, save_or_return_probabilities=False)[0][None]
#
#    p = 750 if max_v % 700 > max_v % 800 else 800
#    patch_size = tuple(p for _ in img.shape)
#    overlap = min(50, max(predictor.configuration_manager.patch_size) // 2)
#    print(f"image very large ({img.shape}>1000); use sliding window", f"{patch_size=}", predictor.configuration_manager.patch_size)
#
#    return sliding_nd_slices(img, patch_size=patch_size, overlap=overlap, fun=fun)[0], None


def sliding_nd_slices(arr: np.ndarray, patch_size: tuple, overlap: int, fun) -> np.ndarray:
    """Apply *fun* to an N-D array using a sliding-window strategy with overlap.

    Args:
        arr: Input array to process.
        patch_size: Size of each patch along every dimension.
        overlap: Number of voxels to overlap between adjacent patches (symmetric).
        fun: Callable applied to each patch; must return an array with the same
            shape as its input.

    Returns:
        Reconstructed array of the same shape as *arr* with patch outputs stitched
        together (overlap regions are overwritten by the most recent patch).
    """
    print("sliding window")
    step = tuple(p - overlap for p in patch_size)
    half_overlap = overlap // 2
    shape = arr.shape

    # Compute number of steps in each dimension
    ranges = [range(0, max(s, 1), st) if s != 1 else [0] for s, st in zip(shape, step)]
    result = np.zeros_like(arr)
    for starts in np.ndindex(*[len(r) for r in ranges]):
        # Compute actual start and end indices for this patch
        idx_start = [ranges[dim][i] for dim, i in enumerate(starts)]
        idx_start2 = [ranges[dim][i] + half_overlap if ranges[dim][i] != 0 else 0 for dim, i in enumerate(starts)]
        idx_start3 = [half_overlap if ranges[dim][i] != 0 else 0 for dim, i in enumerate(starts)]
        idx_end = [min(start + size, shape[dim]) for start, size, dim in zip(idx_start, patch_size, range(len(shape)))]
        idx_end2 = [
            (start + size - half_overlap if start + size < shape[dim] else shape[dim])
            for start, size, dim in zip(idx_start, patch_size, range(len(shape)))
        ]
        idx_end3 = [(-half_overlap if a != shape[dim] else None) for a, dim in zip(idx_end2, range(len(shape)))]

        slices = tuple(slice(s, e) for s, e in zip(idx_start, idx_end))
        slices2 = tuple(slice(s, e) for s, e in zip(idx_start2, idx_end2))
        slices3 = tuple(slice(s, e) for s, e in zip(idx_start3, idx_end3))
        print("sliding window", slices)
        patch = arr[slices]
        patch = fun(patch)
        result[slices2] = patch[slices3]
    return result


# if __name__ == "__main__":
# np.zeros((1, 2243, 472, 622))
# x = sliding_nd_slices()
# max_v=2243, (1, 2243, 472, 622)
# image very large ((1, 2243, 472, 622)>1000); use sliding window patch_size=<generator object predict_single_npy_array.<locals>.<genexpr> at 0x7f89c12dfac0> [160, 192, 192]
