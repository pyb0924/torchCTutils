from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from pydicom import dcmread

import torch
from torch import Tensor
from torchvision.utils import save_image

from .preprocess import resample_by_size


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
    normalize=True,
    min_value=-1024,
    max_value=2048,
):
    if normalize:
        output = output * (max_value - min_value) + min_value

    output = np.clip(output, min_value, max_value)
    
    ds = dcmread(Path(ds_path) / "1.dcm")
    output_size = (ds.Rows, ds.Columns, int(ds.ImagesInAcquisition))
    output = (output - int(ds.RescaleIntercept)) / int(ds.RescaleSlope)

    output_image = sitk.GetImageFromArray(output)
    output_image = resample_by_size(
        output_image, output_size, output_type=sitk.sitkUInt16
    )
    output = sitk.GetArrayFromImage(output_image)
    
    for i, slice_path in enumerate(Path(ds_path).glob("*.dcm")):
        ds = dcmread(slice_path)
        ds.PixelData = output[i].tobytes()
        ds.save_as(Path(output_path) / f"{i+1}.dcm")
