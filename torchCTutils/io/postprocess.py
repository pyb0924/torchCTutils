from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

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


def save_image_to_dcm(image: sitk.Image, path_str: str):
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(path_str)
    writer.Execute(image)


# def save_image_to_dcm_series(image: sitk.Image, path_str: str):
#     writer = sitk.writer
#     dcm_names = [f"{path_str}/{i}.dcm" for i in range(image.GetSize()[2])]
#     writer.SetFileNames(dcm_names)
#     return writer.Execute(image)


def save_dcm_from_output(
    output: np.array,
    ds: sitk.Image,
    output_path: Union[str, Path],
    min_value=0,
    max_value=2048,
):
    output_image = sitk.GetImageFromArray(output * (max_value - min_value) + min_value)
    output_image = resample_by_size(
        output_image, ds.GetSize(), output_type=sitk.sitkUInt16
    )
    output_image.CopyInformation(ds)
    sitk.WriteImage(output_image, str(output_path / "output.dcm"))
    # save_image_to_dcm_series(output_image, str(output_path))
