from typing import Literal

import torch

from TPTBox.core.vert_constants import never_called

DEVICES = Literal["cpu", "cuda", "mps"]


def get_device(ddevice: DEVICES, gpu_id: int):
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
