from typing import Union, Optional
import numpy as np
import tomosipo as ts

from timm.layers import to_2tuple, to_3tuple


def get_FP_operator(
    size: Union[int, list[int], tuple[int]],
    angles: Optional[Union[int, np.ndarray]] = None,
    detector_shape=None,
) -> ts.Operator:
    """Get forward projection(FP) module in PyTorch

    Args:
        size (int): Input image size [x, y, z]
        angles (int, optional): Number of angles to do projection. Defaults to None.

    Returns:
        nn.Module: PyTorch FP module
    """

    if type(size) == int:
        size = to_3tuple(size)
    x, y, z = size
    vg = ts.volume(shape=(z, x, y), size=(1, 1, 1))
    pg = ts.parallel(angles=angles, shape=detector_shape, size=(1, 1))
    return ts.operator(vg, pg)


def get_BP_operator(
    size: Union[int, list[int], tuple[int]],
    angles: Optional[Union[int, np.ndarray]] = None,
    detector_shape=None,
) -> ts.Operator:
    return get_FP_operator(size, angles, detector_shape).T

def get_paired_CT_operator(
    size: Union[int, list[int], tuple[int]],
    angles: Optional[Union[int, np.ndarray]] = None,
    detector_shape=None,
):
    fp = get_FP_operator(size, angles, detector_shape)
    fbp = fp.T
    return fp, fbp


