# Copyright 2024 Hartmut Häntze
# Edited by Robert Graf 2024

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)
WEIGHTS_URL_ = "https://github.com/robert-graf/VibeSegmentator/releases/download/v1.0.0/"
env_name = "VIBESEG_WEIGHTS_PATH"


def get_weights_dir(idx: int, model_path: Path | None = None) -> Path:
    """Resolve and return the directory where model weights for dataset ``idx`` are stored.

    Priority: environment variable ``VIBESEG_WEIGHTS_PATH`` > ``model_path``
    argument > default path relative to this file.

    Args:
        idx: Numeric dataset identifier (e.g. ``85`` for Dataset085).
        model_path: Optional override for the base weights directory. Ignored
            when the environment variable is set.

    Returns:
        Path to the dataset-specific weights directory (``<base>/Dataset<idx:03>``).
        The directory and its parent are created if they do not exist.
    """
    if env_name in os.environ:
        weights_dir: Path = Path(os.environ[env_name])
    elif model_path is not None and model_path.exists():
        weights_dir = model_path
    else:
        assert Path(__file__).parent.name == "VibeSeg", Path(__file__).parent

        weights_dir = Path(__file__).parent.parent / "nnUNet/nnUNet_results"

    weights_dir.parent.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    weights_dir = weights_dir / f"Dataset{idx:03}"

    return weights_dir


def read_config(idx: int) -> dict[str, float]:
    """Read the ``dataset.json`` configuration for the given dataset index.

    Args:
        idx: Numeric dataset identifier.

    Returns:
        Parsed configuration dict, or ``{'dataset_release': 0.0}`` when the
        file does not yet exist (weights not downloaded).
    """
    weights_dir = get_weights_dir(idx)
    ds_path = weights_dir / "dataset.json"
    if ds_path.exists():
        with open(ds_path) as f:
            config_info: dict[str, float] = json.load(f)
        return config_info
    else:
        return {"dataset_release": 0.0}


def _download_weights(idx: int = 85, addendum: str = "", first: bool = True) -> None:
    """Download and extract the zip archive for dataset ``idx`` from the release URL."""
    weights_dir = get_weights_dir(idx)
    weights_url = WEIGHTS_URL_ + f"{idx:03}{addendum}.zip"
    _download(weights_url, weights_dir, text="pretrained weights")
    if first:
        addendum_download(idx)


def _download(weights_url: str, weights_dir: Path, text: str = "", is_zip: bool = True) -> None:
    """Download a file from ``weights_url`` into ``weights_dir``, optionally extracting it."""
    try:
        # Retrieve file size
        with urllib.request.urlopen(str(weights_url)) as response:
            file_size = int(response.info().get("Content-Length", -1))
    except Exception:
        print("Download attempt failed:", weights_url)
        return
    print(f"Downloading {text}...")

    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=Path(weights_url).name) as pbar:

        def update_progress(block_num: int, block_size: int, total_size: int) -> None:
            if pbar.total != total_size:
                pbar.total = total_size
            pbar.update(block_num * block_size - pbar.n)

        zip_path = weights_dir.parent / Path(weights_url).name
        # Download the file
        urllib.request.urlretrieve(str(weights_url), zip_path, reporthook=update_progress)
    if is_zip:
        print(f"Extracting {text}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(weights_dir)
        os.remove(zip_path)  # noqa: PTH107


def addendum_download(idx: int) -> None:
    """Download any additional weight archives listed in ``other_downloads.json``.

    Args:
        idx: Numeric dataset identifier whose weights directory is inspected.
    """
    weights_dir = get_weights_dir(idx)
    next_zip = weights_dir / "other_downloads.json"
    if next_zip.exists():
        with open(next_zip) as f:
            add = json.load(f)
        [_download_weights(idx, addendum=a, first=False) for a in add]
        next_zip.unlink()


def download_weights(idx: int, model_path: Path | None = None) -> Path:
    """Ensure model weights for dataset ``idx`` are present locally, downloading if needed.

    Args:
        idx: Numeric dataset identifier.
        model_path: Optional base directory override for weight storage.

    Returns:
        Path to the dataset-specific weights directory.
    """
    weights_dir = get_weights_dir(idx, model_path)

    # Check if weights are downloaded
    if not weights_dir.exists():
        _download_weights(idx)
    addendum_download(idx)
    return weights_dir
