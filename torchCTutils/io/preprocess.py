from datetime import datetime
from typing import Literal
from pathlib import Path

import numpy as np


def get_normalized_array(
    path: Path, method: Literal["norm01", "norm", "norm1"] = "norm01"
) -> np.array:
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