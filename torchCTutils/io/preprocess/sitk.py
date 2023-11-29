from pathlib import Path

import numpy as np
import SimpleITK as sitk


def sitk_read_series_from_dcm(path_str, read_info=False):
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(path_str)
    img_names = sorted(list(img_names), key=lambda x: int(Path(x).stem))
    reader.SetFileNames(img_names)
    if read_info:
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
    return reader.Execute()


def sitk_resample_by_spacing(
    image,
    new_spacing=[1.0, 1.0, 1.0],
    resample_mode=sitk.sitkLinear,
):
    resize_factor = np.array(image.GetSpacing()) / new_spacing
    new_size = (image.GetSize() * resize_factor).astype(int)
    # print(new_size)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetInterpolator(resample_mode)
    return resampler.Execute(image)


def sitk_resample_by_size(
    image,
    new_size,
    resample_mode=sitk.sitkLinear,
    output_type=sitk.sitkFloat32,
):
    resize_factor = np.array(image.GetSize()) / new_size
    new_spacing = image.GetSpacing() * resize_factor
    # print(new_size)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetOutputPixelType(output_type)
    resampler.SetInterpolator(resample_mode)
    return resampler.Execute(image)


def sitk_window_normalize(
    image, window_level=512, window_width=1536, min_value=0.0, max_value=1.0
):
    window_filter = sitk.IntensityWindowingImageFilter()
    window_filter.SetWindowMinimum(window_level - window_width // 2)
    window_filter.SetWindowMaximum(window_level + window_width // 2)
    window_filter.SetOutputMinimum(min_value)
    window_filter.SetOutputMaximum(max_value)
    return window_filter.Execute(image)


def sitk_get_mask_and_bbox(
    image, threshold=1500, use_opening=False, kernel_size=(2, 2, 2)
):
    mask = sitk.BinaryThreshold(image, threshold, 5000)
    if use_opening:
        mask = sitk.BinaryMorphologicalOpening(mask, kernel_size)
    statFilter = sitk.LabelStatisticsImageFilter()
    statFilter.Execute(image, mask)
    return (
        sitk.GetArrayFromImage(mask),
        np.array(statFilter.GetBoundingBox(1)),
    )  # x, y, z


def sitk_get_preprocessed_from_dcm(
    path_str: str, size=[256, 256, 64], window_level=512, window_width=1536
):
    image = sitk_read_series_from_dcm(path_str)
    image = sitk_resample_by_size(image, size)
    image = sitk_window_normalize(image, window_level, window_width)
    return sitk.GetArrayFromImage(image)  # z, y, x
