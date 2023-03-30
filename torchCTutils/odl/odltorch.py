from functools import partial
from typing import Literal, Optional

from odl.contrib.torch import OperatorFunction, OperatorModule
from torch import Tensor

from .operator import get_FP_operator, get_FBP_operator, get_paired_CT_operator


def odl_FP_layer(
    size: int, angles: Optional[int] = None, dim: Literal[2, 3] = 2
) -> OperatorModule:
    """Get forward projection(FP) module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        nn.Module: PyTorch FP module
    """
    fp = get_FP_operator(size, angles, dim=dim)
    return OperatorModule(fp)


def odl_FBP_layer(
    size: int, angles: Optional[int] = None, dim: Literal[2, 3] = 2
) -> OperatorModule:
    """Get filtered back projection(FP) module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        nn.Module: PyTorch FBP module
    """
    fbp = get_FBP_operator(size, angles, dim=dim)
    return OperatorModule(fbp)


def get_paired_CT_layer(
    size: int, angles: Optional[int] = None, dim: Literal[2, 3] = 2
) -> tuple[OperatorModule, OperatorModule]:
    """Get paired FP/FBP module in PyTorch

    Args:
        size (int): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tuple[nn.Module]: paired PyTorch FP/FBP module
    """
    fp, fbp = get_paired_CT_operator(size, angles, dim=dim)
    return OperatorModule(fp), OperatorModule(fbp)


def odlFP(x: Tensor, dim: Literal[2, 3] = 2) -> Tensor:
    """Run forward projection(FP) implemented by ODL

    Args:
        x (Tensor): Input image to do FP.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tensor: Sinogram tensor in projection field.
    """
    size = max(x.shape[-1], x.shape[-2])
    fp = get_FP_operator(size, dim=dim)
    return OperatorFunction.apply(fp, x)


def odlFBP(x: Tensor, size: int, dim: Literal[2, 3] = 2) -> Tensor:
    """Run filterd back projection(FBP) implemented by ODL

    Args:
        x (Tensor): Input sinogram to do FBP.
        size (int): Image size to be reconstructed.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tensor: Reconstructed image tensor in image field.
    """
    fbp = get_FBP_operator(size, dim=dim)
    return OperatorFunction.apply(fbp, x)


def get_paired_CT_func(size: int, dim: Literal[2, 3] = 2):
    """Get paired FP/FBP PyTorch function.

    Args:
        size (int): image size
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.

    Returns:
        Tuple[Callable[[Tensor], Tensor]]: paired (odlFP, odlFBP)
    """
    fp, fbp = get_paired_CT_operator(size, dim=dim)
    return partial(OperatorFunction.apply, fp), partial(OperatorFunction.apply, fbp)
