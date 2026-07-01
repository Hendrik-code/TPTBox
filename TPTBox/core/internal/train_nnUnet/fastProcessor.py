#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
from copy import deepcopy
from pathlib import Path

import blosc2
import nnunetv2.experiment_planning.plan_and_preprocess_api as pp
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, write_pickle
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def preprocess(
    dataset_ids: list[int],
    plans_identifier: str = "nnUNetPlans",
    configurations: tuple[str] | list[str] = ("2d", "3d_fullres", "3d_lowres"),  # type: ignore
    num_processes: int | tuple[int, ...] | list[int] = (8, 4, 8),
    compress=True,
    verbose: bool = False,
) -> None:
    """Run nnunet data-processing."""
    for d in dataset_ids:
        _preprocess_dataset(d, plans_identifier, configurations, num_processes, compress, verbose)


def _preprocess_dataset(
    dataset_id: int,
    plans_identifier: str = "nnUNetPlans",
    configurations: tuple[str] | list[str] = ("2d", "3d_fullres", "3d_lowres"),  # type: ignore
    num_processes: int | tuple[int, ...] | list[int] = (8, 4, 8),
    compress=True,
    verbose: bool = False,
) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)  # type: ignore
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f"The list provided with num_processes must either have len 1 or as many elements as there are "
            f"configurations (see --help). Number of configurations: {len(configurations)}, length "
            f"of num_processes: "
            f"{len(num_processes)}"
        )

    dataset_name = pp.convert_id_to_dataset_name(dataset_id)

    print(f"Preprocessing dataset {dataset_name}")
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")  # type: ignore
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations, strict=False):
        print(f"Configuration: {c}...")
        if c not in plans_manager.available_configurations:
            print(f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of dataset {dataset_name}. Skipping.")
            continue
        patch_size = plans_manager.get_configuration(c).patch_size
        preprocessor = FastPreprocessor(verbose=verbose, compress=compress, patch_size=patch_size)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    from distutils.file_util import copy_file

    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, "gt_segmentations"))  # type: ignore
    dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))  # type: ignore
    dataset = pp.get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)  # type: ignore
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(
            dataset[k]["label"],
            join(nnUNet_preprocessed, dataset_name, "gt_segmentations", k + dataset_json["file_ending"]),  # type: ignore
            update=True,  # type: ignore
        )  # type: ignore


def _comp_blosc2_params(
    image_size: tuple[int, int, int, int],
    patch_size: tuple[int, int] | tuple[int, int, int],
    bytes_per_pixel: int = 4,  # 4 byte are float32
    l1_cache_size_per_core_in_bytes=32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
    l3_cache_size_per_core_in_bytes=1441792,
    # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
    safety_factor: float = 0.8,  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
):
    """Computes a recommended block and chunk size for saving arrays with blosc v2.

    Bloscv2 NDIM doku: "Remember that having a second partition means that we have better flexibility to fit the
    different partitions at the different CPU cache levels; typically the first partition (aka chunks) should
    be made to fit in L3 cache, whereas the second partition (aka blocks) should rather fit in L2/L1 caches
    (depending on whether compression ratio or speed is desired)."
    (https://www.blosc.org/posts/blosc2-ndim-intro/)
    -> We are not 100% sure how to optimize for that. For now we try to fit the uncompressed block in L1. This
    might spill over into L2, which is fine in our books.

    Note: this is optimized for nnU-Net dataloading where each read operation is done by one core. We cannot use threading

    Cache default values computed based on old Intel 4110 CPU with 32K L1, 128K L2 and 1408K L3 cache per core.
    We cannot optimize further for more modern CPUs with more cache as the data will need be be read by the
    old ones as well.

    Args:
        patch_size: Image size, must be 4D (c, x, y, z). For 2D images, make x=1
        patch_size: Patch size, spatial dimensions only. So (x, y) or (x, y, z)
        bytes_per_pixel: Number of bytes per element. Example: float32 -> 4 bytes
        l1_cache_size_per_core_in_bytes: The size of the L1 cache per core in Bytes.
        l3_cache_size_per_core_in_bytes: The size of the L3 cache exclusively accessible by each core. Usually the global size of the L3 cache divided by the number of cores.

    Returns:
        The recommended block and the chunk size.
    """
    # Fabians code is ugly, but eh

    num_channels = image_size[0]
    if len(patch_size) == 2:
        patch_size = [1, *patch_size]
    patch_size = np.array(patch_size)
    block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

    # shrink the block size until it fits in L1
    estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
    while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
        # pick largest deviation from patch_size that is not 1
        axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
        idx = 0
        picked_axis = axis_order[idx]
        while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        # now reduce that axis to the next lowest power of 2
        block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
        block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

    block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

    # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
    # load data into L3
    # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
    chunk_size = deepcopy(block_size)
    estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
    while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
        if patch_size[0] == 1 and all(i == j for i, j in zip(chunk_size[2:], image_size[2:])):
            break
        if all(i == j for i, j in zip(chunk_size, image_size)):
            break
        # find axis that deviates from block_size the most
        axis_order = np.argsort(chunk_size[1:] / block_size[1:])
        idx = 0
        picked_axis = axis_order[idx]
        while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
        chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
            # chunk size should not exceed patch size * 1.5 on average
            chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
            break
    # better safe than sorry
    chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

    # print(image_size, chunk_size, block_size)
    return tuple(block_size), tuple(chunk_size)


class FastPreprocessor(DefaultPreprocessor):
    """Saves nnUnet data set in a mem-mappable data format. compress needs 2.5.2 or higher."""

    def __init__(self, verbose: bool = True, compress=True, patch_size=None):
        super().__init__(verbose)
        print(f"FastPreprocessor {compress=}")
        self.compress = compress
        self.patch_size = patch_size

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: list[str],
        seg_file: str,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict | str,
    ) -> None:
        """Internal nnUnet function."""
        if Path(output_filename_truncated + ".npz").exists() and Path(output_filename_truncated + ".pkl").exists():
            print("skip", output_filename_truncated, end="\r")
            return

        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        # print("dtypes", data.dtype, seg.dtype)
        # print(data.dtype, data.shape, data.max(), data.min())
        if self.compress:
            if self.patch_size is None:
                np.savez_compressed(output_filename_truncated + ".npz", data=data.astype(np.float16), seg=seg)
            else:
                # IMPORTANT
                blosc2.set_nthreads(1)

                # derive chunk/block layout
                blocks, chunks = _comp_blosc2_params(
                    image_size=data.shape,
                    patch_size=self.patch_size,
                    bytes_per_pixel=2,  # float16
                )
                cparams = {"codec": blosc2.Codec.ZSTD, "filters": [blosc2.Filter.BITSHUFFLE], "clevel": 5}
                # save image
                blosc2.asarray(
                    np.ascontiguousarray(data, dtype=np.float16),
                    urlpath=output_filename_truncated + ".b2nd",
                    chunks=chunks,
                    blocks=blocks,
                    cparams=cparams,
                )
                cparams = {"codec": blosc2.Codec.ZSTD, "filters": [blosc2.Filter.BITSHUFFLE], "clevel": 5}

                # segmentation usually compresses extremely well
                blosc2.asarray(
                    np.ascontiguousarray(seg),
                    urlpath=output_filename_truncated + "_seg.b2nd",
                    chunks=chunks,
                    blocks=blocks,
                    cparams=cparams,
                )
        else:
            np.savez(output_filename_truncated + ".npz", data=data.astype(np.float16), seg=seg)
        write_pickle(properties, output_filename_truncated + ".pkl")
