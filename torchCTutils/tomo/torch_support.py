from typing import Union, Optional
import numpy as np
import tomosipo as ts
from tomosipo.torch_support import to_autograd

from torch import nn

from .operator import get_FP_operator, get_BP_operator


class TomoOperatorModule(nn.Module):
    def __init__(self, operator: ts.Operator) -> None:
        super(TomoOperatorModule, self).__init__()
        self.operator = to_autograd(operator, num_extra_dims=2)

    def forward(self, x):
        """
        x(torch.Tensor): Input image, shape [B,C,D,H,W] / [B,C,DH,A,DW].
        """
        return self.operator(x)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.operator.__class__.__name__})"


def FPlayer(
    size: Union[int, list[int], tuple[int]],
    angles: Optional[Union[int, np.ndarray]] = None,
    detector_shape=None,
) -> TomoOperatorModule:
    """Get forward projection(FP) module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        detector_shape: Detector shape. Defaults to None.

    Returns:
        nn.Module: PyTorch FP module
    """

    fp = get_FP_operator(size, angles, detector_shape)
    return TomoOperatorModule(fp)


def BPlayer(
    size: Union[int, list[int], tuple[int]],
    angles: Optional[Union[int, np.ndarray]] = None,
    detector_shape=None,
) -> TomoOperatorModule:
    """Get backward projection(BP) module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        detector_shape: Detector shape. Defaults to None.

    Returns:
        nn.Module: PyTorch BP module
    """

    bp = get_BP_operator(size, angles, detector_shape)
    return TomoOperatorModule(bp)


def paired_CTlayer(
    size: Union[int, list[int], tuple[int]],
    angles: Optional[Union[int, np.ndarray]] = None,
    detector_shape=None,
):
    """Get paired FP/BP module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        detector_shape: Detector shape. Defaults to None.

    Returns:
        Tuple[nn.Module]: paired PyTorch FP/BP module
    """
    fp = get_FP_operator(size, angles, detector_shape)
    bp = fp.T
    return TomoOperatorModule(fp), TomoOperatorModule(bp)
