# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
from __future__ import annotations

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from TPTBox.segmentation.nnUnet_utils.default_preprocessor import DefaultPreprocessor
from TPTBox.segmentation.nnUnet_utils.plans_handler import ConfigurationManager, PlansManager


class PreprocessAdapterFromNpy(DataLoader):
    """DataLoader adapter that preprocesses raw numpy arrays on-the-fly for nnU-Net inference.

    Wraps a list of image arrays and their metadata into a
    :class:`~batchgenerators.dataloading.data_loader.DataLoader` that applies
    the full nnU-Net preprocessing pipeline (crop, normalise, resample) via
    :class:`~TPTBox.segmentation.nnUnet_utils.default_preprocessor.DefaultPreprocessor`.

    Args:
        list_of_images: Image arrays, each with shape ``(C, X, Y, Z)``.
        list_of_segs_from_prev_stage: Optional cascade segmentation arrays
            (one per image) to be stacked as one-hot channels. Pass ``None``
            for all entries when not using the cascade.
        list_of_image_properties: Per-image property dicts containing at
            least a ``'spacing'`` key.
        truncated_of_names: Optional output file stems for each image (used
            downstream for saving). Pass ``None`` when not needed.
        plans_manager: Plans manager for the loaded model.
        dataset_json: Parsed ``dataset.json`` dictionary.
        configuration_manager: Configuration-specific preprocessing parameters.
        num_threads_in_multithreaded: Number of worker threads (passed to the
            underlying :class:`DataLoader`).
        verbose: If ``True``, print preprocessing progress information.
    """

    def __init__(
        self,
        list_of_images: list[np.ndarray],
        list_of_segs_from_prev_stage: list[np.ndarray] | None,
        list_of_image_properties: list[dict],
        truncated_of_names: list[str] | None,
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_threads_in_multithreaded: int = 1,
        verbose: bool = False,
    ):
        preprocessor = DefaultPreprocessor(verbose=verbose)
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_of_names = (
            preprocessor,
            plans_manager,
            configuration_manager,
            dataset_json,
            truncated_of_names,
        )
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)  # type: ignore
        if truncated_of_names is None:
            truncated_of_names = [None] * len(list_of_images)  # type: ignore

        super().__init__(
            list(zip(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_of_names)),  # type: ignore
            1,
            num_threads_in_multithreaded,
            seed_for_shuffle=1,
            return_incomplete=True,
            shuffle=False,
            infinite=False,
            sampling_probabilities=None,
        )

        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self) -> dict:
        """Preprocess the next sample and return it as a data dictionary.

        Returns:
            A dict with keys:
                - ``'data'``: preprocessed image tensor ``(C, X, Y, Z)``.
                - ``'data_properites'``: updated properties dict with cropping
                  and resampling metadata.
                - ``'ofile'``: output file stem (may be ``None``).
        """
        idx = self.get_indices()[0]
        image = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        props = self._data[idx][2]
        of_name = self._data[idx][3]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg = self.preprocessor.run_case_npy(
            image, seg_prev_stage, props, self.plans_manager, self.configuration_manager, self.dataset_json
        )
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {"data": data, "data_properites": props, "ofile": of_name}


def convert_labelmap_to_one_hot(
    segmentation: np.ndarray | torch.Tensor, all_labels: list | torch.Tensor | np.ndarray | tuple, output_dtype=None
) -> np.ndarray | torch.Tensor:
    """Convert an integer label map to a one-hot encoded array.

    Args:
        segmentation: Integer label array/tensor with shape ``(X, Y, Z)``. When
            a :class:`torch.Tensor`, using ``LongTensor`` avoids an extra cast.
        all_labels: Ordered sequence of all label values to encode. Labels must
            be **consecutive integers** starting from 0 (e.g. ``[0, 1, 2, 3]``).
            Non-consecutive labels (e.g. ``[0, 32, 255]``) are **not** supported.
        output_dtype: Desired dtype for the output. Defaults to
            ``np.uint8`` / ``torch.uint8`` when ``None``.

    Returns:
        One-hot encoded array/tensor with shape
        ``(len(all_labels), X, Y, Z)``, on the same device as the input when
        ``segmentation`` is a :class:`torch.Tensor`.

    Note:
        NumPy arrays are processed faster than Torch tensors in this function.
    """
    if isinstance(segmentation, torch.Tensor):
        result = torch.zeros(
            (len(all_labels), *segmentation.shape),
            dtype=output_dtype if output_dtype is not None else torch.uint8,
            device=segmentation.device,
        )
        # variant 1, 2x faster than 2
        result.scatter_(0, segmentation[None].long(), 1)  # why does this have to be long!?
        # variant 2, slower than 1
        # for i, l in enumerate(all_labels):
        #     result[i] = segmentation == l
    else:
        result = np.zeros((len(all_labels), *segmentation.shape), dtype=output_dtype if output_dtype is not None else np.uint8)
        # variant 1, fastest in my testing
        for i, l in enumerate(all_labels):
            result[i] = segmentation == l
        # variant 2. Takes about twice as long so nah
        # result = np.eye(len(all_labels))[segmentation].transpose((3, 0, 1, 2))
    return result
