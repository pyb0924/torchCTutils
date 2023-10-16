from functools import partial
from typing import Literal, Optional, Union

from odl.contrib.torch import OperatorFunction, OperatorModule
from torch import Tensor

from .operator import get_FP_operator, get_FBP_operator, get_paired_CT_operator


def odl_FP_layer(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
) -> OperatorModule:
    """Get forward projection(FP) module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        nn.Module: PyTorch FP module
    """
    fp = get_FP_operator(size, dim, angles, detector_shape)
    return OperatorModule(fp)


def odl_FBP_layer(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
) -> OperatorModule:
    """Get filtered back projection(FP) module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        nn.Module: PyTorch FBP module
    """
    fbp = get_FBP_operator(size, dim, angles, detector_shape)
    return OperatorModule(fbp)


def get_paired_CT_layer(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
) -> tuple[OperatorModule, OperatorModule]:
    """Get paired FP/FBP module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tuple[nn.Module]: paired PyTorch FP/FBP module
    """
    fp, fbp = get_paired_CT_operator(size, dim, angles, detector_shape)
    return OperatorModule(fp), OperatorModule(fbp)


def odl_FP(
    x: Tensor,
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
) -> Tensor:
    """Run forward projection(FP) implemented by ODL

    Args:
        x (Tensor): Input image to do FP.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tensor: Sinogram tensor in projection field.
    """
    if dim == 2:
        size = [x.shape[-2], x.shape[-1]]
    elif dim == 3:
        size = [x.shape[-3], x.shape[-2], x.shape[-1]]

    fp = get_FP_operator(size, dim, angles, detector_shape)
    return OperatorFunction.apply(fp, x)


def odl_FBP(
    x: Tensor,
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
) -> Tensor:
    """Run filterd back projection(FBP) implemented by ODL

    Args:
        x (Tensor): Input sinogram to do FBP.
        size (Union[int, list[int], tuple[int]]): Image size to be reconstructed.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tensor: Reconstructed image tensor in image field.
    """
    fbp = get_FBP_operator(size, dim, angles, detector_shape)
    return OperatorFunction.apply(fbp, x)


def get_paired_CT_func(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
):
    """Get paired FP/FBP PyTorch function.

    Args:
        size (Union[int, list[int], tuple[int]]): image size
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tuple[Callable[[Tensor], Tensor]]: paired (odlFP, odlFBP)
    """
    fp, fbp = get_paired_CT_operator(size, dim, angles, detector_shape)
    return partial(OperatorFunction.apply, fp), partial(OperatorFunction.apply, fbp)
