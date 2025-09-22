import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def get_batch(x: ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[Tensor, Tensor]:
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)[:, np.newaxis]
    sub_indices = np.arange(context_length)

    input_indices = start_indices + sub_indices
    target_indices = input_indices + 1

    inputs = torch.from_numpy(x[input_indices].astype(np.int64)).to(device)
    targets = torch.from_numpy(x[target_indices].astype(np.int64)).to(device)

    device_str = str(device)
    if 'cuda' in device_str:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets