import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def get_batch(x: ndarray, batch_size: int, context_length: int, device: torch.device) -> tuple[Tensor, Tensor]:
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)[:, np.newaxis]
    sub_indices = np.arange(context_length)

    input_indices = start_indices + sub_indices
    target_indices = input_indices + 1

    inputs = torch.tensor(x[input_indices], dtype=torch.long, device=device)
    targets = torch.tensor(x[target_indices], dtype=torch.long, device=device)

    return inputs, targets