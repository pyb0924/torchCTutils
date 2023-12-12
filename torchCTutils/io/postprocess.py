from pathlib import Path
from typing import Union

import numpy as np
from pydicom import dcmread

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
    _, _, height, width = tensor.shape
    result = torch.zeros_like(tensor)
    for i in range(height):
        for j in range(width):
            if (i - height / 2) ** 2.0 + (
                j - width / 2
            ) ** 2 <= ratio**2 * height * width / 4:
                result[:, :, i, j] = tensor[:, :, i, j]
    return result


def save_dcm_from_output(
    output: np.ndarray,
    ds_path: Union[str, Path],
    output_path: Union[str, Path],
    model_name: str = "",
    min_value=0,
    max_value=3000,
):
    output = output * (max_value - min_value) + min_value
    output = np.clip(output, min_value, max_value).astype(np.uint16)
    ds_list = sorted(list(Path(ds_path).glob("*")), key=lambda x: int(x.stem))

    for i, slice_path in enumerate(ds_list):
        ds = dcmread(slice_path)
        ds.PatientName+=f"_{model_name}"
        ds.PixelData = output[i].tobytes()
        ds.LargestImagePixelValue = np.max(output[i])
        ds.SmallestImagePixelValue = np.min(output[i])
        ds.save_as(Path(output_path) / f"{i+1}.dcm")
