from typing import Literal

import torch

from TPTBox.core.vert_constants import never_called

DEVICES = Literal["cpu", "cuda", "mps"]


def get_device(ddevice: DEVICES, gpu_id: int) -> torch.device:
    """Construct a :class:`torch.device` from a device type string and GPU index.

    Args:
        ddevice: Target device type — one of ``"cpu"``, ``"cuda"``, or
            ``"mps"``.
        gpu_id: CUDA device index (only used when ``ddevice`` is ``"cuda"``).

    Returns:
        A :class:`torch.device` configured for the requested backend.

    Raises:
        AssertionError: Via :func:`~TPTBox.core.vert_constants.never_called`
            when ``ddevice`` is not one of the recognised values.
    """
    if ddevice == "cpu":
        # import multiprocessing

        # try:
        #    torch.set_num_threads(multiprocessing.cpu_count())
        # except Exception:
        #    pass
        device = torch.device("cpu")
    elif ddevice == "cuda":
        device = torch.device(type="cuda", index=gpu_id)
    elif ddevice == "mps":
        device = torch.device("mps")
    else:
        never_called(ddevice)
    return device
