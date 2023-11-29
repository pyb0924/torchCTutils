from pathlib import Path
import numpy as np

from typing import Literal


def get_bbox_from_mask(mask: np.ndarray):
    nonzero_indexs = np.nonzero(mask)
    result = []
    for index in nonzero_indexs:
        result.extend([np.min(index), np.max(index)])
    return np.array(result)


def get_mask_by_threshold(image: np.ndarray, t1: int, t2: int):
    mask = np.zeros_like(image)
    mask[(image >= t1) & (image <= t2)] = 1
    return mask


def get_mask_and_bbox(image: np.ndarray, t1, t2):
    mask = get_mask_by_threshold(image, t1, t2).astype(np.int64)
    bbox = get_bbox_from_mask(mask)
    return mask, bbox


def data_range_to_window(vmin, vmax):
    return (vmax + vmin) / 2, vmax - vmin


def window_to_data_range(wl, ww):
    return wl - ww / 2, wl + ww / 2


def window_transform(image, wl=1500, ww=3000, normalize=True):
    """Window normalization for CT image.

    Args:
        image (np.ndarray): CT image
        wl (int, optional): Window level. Defaults to 512.
        ww (int, optional): Window width. Defaults to 1536.
        normalize (bool, optional): Whether to normalize the image. Defaults to True.

    Returns:
        np.ndarray: normalized CT image
    """
    window_min, window_max = window_to_data_range(wl, ww)
    image = np.clip(image, window_min, window_max).astype(np.float32)
    if normalize:
        image = (image - window_min) / ww

    return image


def get_normalized_array(
    path: Path, method: Literal["norm01", "norm", "norm1"] = "norm01"
) -> np.ndarray:
    """Get normalized array from file path.

    Args:
        path (Path): _description_
        method (Literal['norm', 'norm1', 'norm01'], optional): Method of Normalization. Defaults to 'norm01'.
            Valid options:
                'norm' : (X - mean(X)) / var(x)
                'norm1' : 2 * (X - min(X)) / (max(X) - min(X)) - 1
                'norm01': (X - min(X)) / (max(X) - min(X))

    Returns:
        np.array: normalized Numpy array
    """
    methods = ["norm01", "norm", "norm1"]
    if method not in methods:
        raise ValueError(f"Invalid method! Availble options: {methods}")

    image = np.load(path)
    if method == "norm":
        image = (image - np.mean(image)) / np.std(image)
    elif method == "norm1":
        image = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    elif method == "norm01":
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image
