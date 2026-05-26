# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache, partial

# see https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING

import dynamic_network_architectures
import nnunetv2
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager_class_from_plans
from torch import nn

if TYPE_CHECKING:
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.label_handling.label_handling import LabelManager


class ConfigurationManager:
    """Manages a single nnU-Net configuration dictionary, exposing its fields as typed properties.

    Wraps a configuration dict (one entry from the ``configurations`` block of a
    ``plans.json`` file) and resolves all keys through typed properties so that
    callers receive properly typed values instead of raw dict lookups.
    """

    def __init__(self, configuration_dict: dict):
        self.configuration = configuration_dict

    def __repr__(self):
        return self.configuration.__repr__()

    @property
    def data_identifier(self) -> str:
        """Unique string identifier for the preprocessed data folder."""
        return self.configuration["data_identifier"]

    @property
    def preprocessor_name(self) -> str:
        """Class name of the preprocessor used for this configuration."""
        return self.configuration["preprocessor_name"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def preprocessor_class(self) -> type[DefaultPreprocessor]:
        """Resolved preprocessor class object, looked up by name inside nnunetv2."""
        preprocessor_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "preprocessing"),  # type: ignore
            self.preprocessor_name,
            current_module="nnunetv2.preprocessing",
        )
        return preprocessor_class  # type: ignore

    @property
    def batch_size(self) -> int:
        """Training batch size for this configuration."""
        return self.configuration["batch_size"]

    @property
    def patch_size(self) -> list[int]:
        """Patch size (in voxels) used during training and inference sliding-window."""
        return self.configuration["patch_size"]

    @property
    def median_image_size_in_voxels(self) -> list[int]:
        """Median image size (in voxels) of the training dataset after resampling."""
        return self.configuration["median_image_size_in_voxels"]

    @property
    def spacing(self) -> list[float]:
        """Target voxel spacing (mm) for this configuration."""
        return self.configuration["spacing"]

    @property
    def normalization_schemes(self) -> list[str]:
        """Per-channel normalization scheme names (e.g. ``CTNormalization``)."""
        return self.configuration["normalization_schemes"]

    @property
    def use_mask_for_norm(self) -> list[bool]:
        """Per-channel flags indicating whether the nonzero mask is used during normalization."""
        return self.configuration["use_mask_for_norm"]

    @property
    def UNet_class_name(self) -> str:
        """Class name of the U-Net architecture used for this configuration."""
        return self.configuration["UNet_class_name"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def UNet_class(self) -> type[nn.Module]:
        """Resolved U-Net class object, looked up inside dynamic_network_architectures.

        Raises:
            RuntimeError: If the class name cannot be found in the architectures package.
        """
        unet_class = recursive_find_python_class(
            join(dynamic_network_architectures.__path__[0], "architectures"),  # type: ignore
            self.UNet_class_name,
            current_module="dynamic_network_architectures.architectures",
        )
        if unet_class is None:
            raise RuntimeError(
                "The network architecture specified by the plans file "
                "is non-standard (maybe your own?). Fix this by not using "
                "ConfigurationManager.UNet_class to instantiate "
                "it (probably just overwrite build_network_architecture of your trainer."
            )
        return unet_class

    @property
    def UNet_base_num_features(self) -> int:
        """Base number of feature maps for the U-Net encoder/decoder."""
        return self.configuration["UNet_base_num_features"]

    @property
    def n_conv_per_stage_encoder(self) -> list[int]:
        """Number of convolutions per encoder stage."""
        return self.configuration["n_conv_per_stage_encoder"]

    @property
    def n_conv_per_stage_decoder(self) -> list[int]:
        """Number of convolutions per decoder stage."""
        return self.configuration["n_conv_per_stage_decoder"]

    @property
    def num_pool_per_axis(self) -> list[int]:
        """Number of pooling operations applied along each spatial axis."""
        return self.configuration["num_pool_per_axis"]

    @property
    def pool_op_kernel_sizes(self) -> list[list[int]]:
        """Kernel sizes for each pooling operation across all stages."""
        return self.configuration["pool_op_kernel_sizes"]

    @property
    def conv_kernel_sizes(self) -> list[list[int]]:
        """Convolution kernel sizes for each stage."""
        return self.configuration["conv_kernel_sizes"]

    @property
    def unet_max_num_features(self) -> int:
        """Maximum number of feature maps at any U-Net stage."""
        return self.configuration["unet_max_num_features"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def resampling_fn_data(
        self,
    ) -> Callable[
        [
            torch.Tensor | np.ndarray,
            tuple[int, ...] | list[int] | np.ndarray,
            tuple[float, ...] | list[float] | np.ndarray,
            tuple[float, ...] | list[float] | np.ndarray,
        ],
        torch.Tensor | np.ndarray,
    ]:
        """Resampling callable for image data, pre-bound with configuration kwargs."""
        fn = recursive_find_resampling_fn_by_name(self.configuration["resampling_fn_data"])
        fn = partial(fn, **self.configuration["resampling_fn_data_kwargs"])
        return fn

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def resampling_fn_probabilities(
        self,
    ) -> Callable[
        [
            torch.Tensor | np.ndarray,
            tuple[int, ...] | list[int] | np.ndarray,
            tuple[float, ...] | list[float] | np.ndarray,
            tuple[float, ...] | list[float] | np.ndarray,
        ],
        torch.Tensor | np.ndarray,
    ]:
        """Resampling callable for predicted probability maps, pre-bound with configuration kwargs."""
        fn = recursive_find_resampling_fn_by_name(self.configuration["resampling_fn_probabilities"])
        fn = partial(fn, **self.configuration["resampling_fn_probabilities_kwargs"])
        return fn

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def resampling_fn_seg(
        self,
    ) -> Callable[
        [
            torch.Tensor | np.ndarray,
            tuple[int, ...] | list[int] | np.ndarray,
            tuple[float, ...] | list[float] | np.ndarray,
            tuple[float, ...] | list[float] | np.ndarray,
        ],
        torch.Tensor | np.ndarray,
    ]:
        """Resampling callable for segmentation masks, pre-bound with configuration kwargs."""
        fn = recursive_find_resampling_fn_by_name(self.configuration["resampling_fn_seg"])
        fn = partial(fn, **self.configuration["resampling_fn_seg_kwargs"])
        return fn

    @property
    def batch_dice(self) -> bool:
        """Whether batch dice loss is used during training."""
        return self.configuration["batch_dice"]

    @property
    def next_stage_names(self) -> list[str] | None:
        """Names of the next cascade stage configurations, if any."""
        ret = self.configuration.get("next_stage")
        if ret is not None and isinstance(ret, str):
            ret = [ret]
        return ret

    @property
    def previous_stage_name(self) -> str | None:
        """Name of the previous cascade stage configuration, if any."""
        return self.configuration.get("previous_stage")


class PlansManager:
    """High-level interface to the nnU-Net ``plans.json`` file.

    Responsibilities:
        1. Resolve inheritance chains among configurations.
        2. Expose label manager and image I/O classes by name.
        3. Provide clearly typed access to plan fields rather than raw dict lookups.
        4. Cache expensive lookups (class resolution, label managers, etc.).

    Direct dict access is still possible via ``PlansManager.plans['key']``.
    """

    def __init__(self, plans_file_or_dict: str | dict):
        """Initialize the PlansManager from a JSON file path or a pre-loaded dict.

        Args:
            plans_file_or_dict: Either a path to a ``plans.json`` file or an
                already-loaded dictionary representing that file.
        """
        self.plans = plans_file_or_dict if isinstance(plans_file_or_dict, dict) else load_json(plans_file_or_dict)

    def __repr__(self):
        return self.plans.__repr__()

    def _internal_resolve_configuration_inheritance(self, configuration_name: str, visited: tuple[str, ...] | None = None) -> dict:
        """Recursively resolve inherited configuration keys and merge them bottom-up."""
        if configuration_name not in self.plans["configurations"].keys():
            raise ValueError(
                f"The configuration {configuration_name} does not exist in the plans I have. Valid "
                f"configuration names are {list(self.plans['configurations'].keys())}."
            )
        configuration = deepcopy(self.plans["configurations"][configuration_name])
        if "inherits_from" in configuration:
            parent_config_name = configuration["inherits_from"]

            if visited is None:
                visited = (configuration_name,)
            else:
                if parent_config_name in visited:
                    raise RuntimeError(
                        f"Circular dependency detected. The following configurations were visited "
                        f"while solving inheritance (in that order!): {visited}. "
                        f"Current configuration: {configuration_name}. Its parent configuration "
                        f"is {parent_config_name}."
                    )
                visited = (*visited, configuration_name)

            base_config = self._internal_resolve_configuration_inheritance(parent_config_name, visited)
            base_config.update(configuration)
            configuration = base_config
        return configuration

    @lru_cache(maxsize=10)  # noqa: B019
    def get_configuration(self, configuration_name: str) -> ConfigurationManager:
        """Return a :class:`ConfigurationManager` for the requested configuration.

        Args:
            configuration_name: Key into the ``configurations`` block of the plans.

        Returns:
            A :class:`ConfigurationManager` wrapping the fully-resolved configuration
            dict (inheritance already applied).

        Raises:
            RuntimeError: If ``configuration_name`` is not present in the plans.
        """
        if configuration_name not in self.plans["configurations"].keys():
            raise RuntimeError(
                f"Requested configuration {configuration_name} not found in plans. "
                f"Available configurations: {list(self.plans['configurations'].keys())}"
            )

        configuration_dict = self._internal_resolve_configuration_inheritance(configuration_name)
        return ConfigurationManager(configuration_dict)

    @property
    def dataset_name(self) -> str:
        """Name of the dataset as stored in the plans file."""
        return self.plans["dataset_name"]

    @property
    def plans_name(self) -> str:
        """Identifier string for this set of plans."""
        return self.plans["plans_name"]

    @property
    def original_median_spacing_after_transp(self) -> list[float]:
        """Median voxel spacing (mm) of the training data after applying transpose_forward."""
        return self.plans["original_median_spacing_after_transp"]

    @property
    def original_median_shape_after_transp(self) -> list[float]:
        """Median image shape (voxels) of the training data after applying transpose_forward."""
        return self.plans["original_median_shape_after_transp"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def image_reader_writer_class(self) -> type[BaseReaderWriter]:
        """Resolved image reader/writer class used for I/O during inference."""
        return recursive_find_reader_writer_by_name(self.plans["image_reader_writer"])

    @property
    def transpose_forward(self) -> list[int]:
        """Axis permutation applied to images before preprocessing (data -> network)."""
        return self.plans["transpose_forward"]

    @property
    def transpose_backward(self) -> list[int]:
        """Axis permutation applied to predictions to restore original orientation."""
        return self.plans["transpose_backward"]

    @property
    def available_configurations(self) -> list[str]:
        """Names of all configurations defined in this plans file."""
        return list(self.plans["configurations"].keys())

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def experiment_planner_class(self) -> type[ExperimentPlanner]:
        """Resolved experiment planner class that generated these plans."""
        planner_name = self.experiment_planner_name
        experiment_planner = recursive_find_python_class(
            join(nnunetv2.__path__[0], "experiment_planning"),  # type: ignore
            planner_name,
            current_module="nnunetv2.experiment_planning",
        )
        return experiment_planner  # type: ignore

    @property
    def experiment_planner_name(self) -> str:
        """Class name of the experiment planner that produced these plans."""
        return self.plans["experiment_planner_used"]

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def label_manager_class(self) -> type[LabelManager]:
        """Resolved LabelManager class appropriate for this plans file."""
        return get_labelmanager_class_from_plans(self.plans)

    def get_label_manager(self, dataset_json: dict, **kwargs) -> LabelManager:
        """Instantiate and return a LabelManager for the given dataset JSON.

        Args:
            dataset_json: Parsed ``dataset.json`` dictionary containing ``labels``
                and optionally ``regions_class_order``.
            **kwargs: Additional keyword arguments forwarded to the LabelManager
                constructor.

        Returns:
            A configured :class:`LabelManager` instance.
        """
        return self.label_manager_class(
            label_dict=dataset_json["labels"], regions_class_order=dataset_json.get("regions_class_order"), **kwargs
        )

    @property
    def foreground_intensity_properties_per_channel(self) -> dict:
        """Per-channel foreground intensity statistics (min, max, mean, std, etc.)."""
        if "foreground_intensity_properties_per_channel" not in self.plans.keys():  # noqa: SIM102
            if "foreground_intensity_properties_by_modality" in self.plans.keys():
                return self.plans["foreground_intensity_properties_by_modality"]
        return self.plans["foreground_intensity_properties_per_channel"]


if __name__ == "__main__":
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    plans = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(3), "nnUNetPlans.json"))  # type: ignore
    # build new configuration that inherits from 3d_fullres
    plans["configurations"]["3d_fullres_bs4"] = {"batch_size": 4, "inherits_from": "3d_fullres"}
    # now get plans and configuration managers
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration("3d_fullres_bs4")
    print(configuration_manager)  # look for batch size 4
