from pathlib import Path
from typing import Union
import cv2
import numpy as np
from pydicom import dcmread
import SimpleITK as sitk


def dcm_crop(ds, xmin, xmax, ymin, ymax, modify_minmax_tag=True):
    img = ds.pixel_array
    res = img[xmin:xmax, ymin:ymax]
    ds.PixelData = res.tobytes()
    ds.Rows, ds.Columns = res.shape
    if modify_minmax_tag:
        ds.SmallestImagePixelValue, ds.LargestImagePixelValue = np.min(res), np.max(res)
    return ds

def dcm_clamp(ds, min_value=0, max_value=3000, modify_minmax_tag=True):
    img = ds.pixel_array
    res = np.clip(img, min_value, max_value)
    ds.PixelData = res.tobytes()
    if modify_minmax_tag:
        ds.SmallestImagePixelValue, ds.LargestImagePixelValue = np.min(res), np.max(res)

    return ds


def dcm_resize(ds, size=256, modify_minmax_tag=True):
    src_size = ds.Rows
    print(ds.Rows, ds.Columns)
    img = ds.pixel_array
    res = cv2.resize(img, (size, size))
    ds.PixelData = res.tobytes()
    ds.Rows = ds.Columns = size
    ds.PixelSpacing = list(map(lambda x: x * src_size / size, ds.PixelSpacing))

    if modify_minmax_tag:
        ds.SmallestImagePixelValue, ds.LargestImagePixelValue = np.min(res), np.max(res)

    return ds


def dcm_read_series(series_path: Union[Path, str]):
    dcm_list = sorted(Path(series_path).rglob("*.dcm"), key=lambda x: int(x.stem))
    pixel_list = []
    for dcm_file in dcm_list:
        ds = dcmread(dcm_file)
        pixel_list.append(ds.pixel_array)
    pixel_array = np.stack(pixel_list, axis=0)
    return pixel_array


def dcm_series2mhd(series_path: Path, output_path: Path):
    dcm_list = sorted(series_path.rglob("*.dcm"), key=lambda x: int(x.stem))
    ref_ds = dcmread(dcm_list[0])

    pixel_dims = (
        int(ref_ds.Rows),
        int(ref_ds.Columns),
        len(dcm_list),
    )

    pixel_array = np.zeros(pixel_dims, dtype=ref_ds.pixel_array.dtype)

    for i, dcm_file in enumerate(dcm_list):
        ds = dcmread(dcm_file)
        pixel_array[:, :, i] = ds.pixel_array
    pixel_array = np.transpose(pixel_array, (2, 0, 1))

    sitk_img = sitk.GetImageFromArray(pixel_array, isVector=False)
    sitk_img.SetSpacing(
        (
            float(ref_ds.PixelSpacing[0]),
            float(ref_ds.PixelSpacing[1]),
            float(ref_ds.SliceThickness),
        )
    )
    sitk_img.SetOrigin(ref_ds.ImagePositionPatient)
    sitk.WriteImage(sitk_img, str(output_path / f"{series_path.name}.mhd"))


if __name__ == "__main__":
    series_path = Path("/data/dat/yibo.pei/clinical/raw/GAO SHU ZHI/frame0/images")
    pixel_array = dcm_read_series(series_path)
    print(pixel_array.shape)
