from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torchvision.utils import save_image


def save_multichannel_grayscale_image(
    tensor: Tensor, filenames: list[Union[str, Path]], normalize=False
):
    
    if tensor.shape[1] != len(filenames):
        raise ValueError("Invalid Input!")

    for i in range(tensor.shape[1]):
        tensor_per_channel = tensor[:, i, :, :].unsqueeze(1)
        save_image(tensor_per_channel, filenames[i], normalize=normalize)


def add_circle_mask_to_output_tensor(tensor: Tensor, ratio=0.9):
    bs, channels, height, width = tensor.shape
    result = torch.zeros_like(tensor)
    for i in range(height):
        for j in range(width):
            if (i - height / 2) ** 2.0 + (
                j - width / 2
            ) ** 2 <= ratio**2 * height * width / 4:
                result[:, :, i, j] = tensor[:, :, i, j]
    return result
