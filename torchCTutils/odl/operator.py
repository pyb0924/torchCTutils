from typing import Literal, Optional, Union
import numpy as np
import odl

from .utils import get_parallel_beam_geometry, get_cone_beam_geometry


def get_FP_operator(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
    mode: Literal["cone", "parallel"] = "parallel",
    src_radius: Optional[float] = None,
    det_radius: Optional[float] = None,
    nodes_on_bdry: Optional[bool] = False,
) -> odl.Operator:
    """Get forward projection(FP) operator from ODL

    Args:
        size (Union[int, list[int], tuple[int]]): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        mode (Literal[&#39;cone&#39;, &#39;parallel&#39;], optional): Geometry type. Defaults to 'parallel'.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.
        src_radius (float, optional): geometry parameter in FanBeamGeometry/ConeBeamGeometry (mode='cone'). Defaults to None.
        det_radius (float, optional): geometry parameter in FanBeamGeometry/ConeBeamGeometry (mode='cone'). Defaults to None.

    Raises:
        ValueError: Invalid dimension for FP!
        ValueError: Undefined geometry mode! Availble mode: [cone, parallel]
        ValueError: Invalid cone beam parameters!

    Returns:
        odl.Operator: FP operator
    """
    if dim != 2 and dim != 3:
        raise ValueError("Invalid dimension for FP!")
    if dim == 2:
        if type(size) == int:
            size = [size, size]
        space = odl.uniform_discr([-1, -1], [1, 1], size, dtype="float32")
    else:
        if type(size) == int:
            size = [size, size, size]
        space = odl.uniform_discr([-1, -1, -1], [1, 1, 1], size, dtype="float32")

    if mode != "parallel" and mode != "cone":
        raise ValueError("Undefined geometry mode! Availble mode: [cone, parallel]")

    if mode == "parallel":
        geometry = get_parallel_beam_geometry(
            space, angles, detector_shape, nodes_on_bdry
        )
    else:
        geometry = get_cone_beam_geometry(
            space, src_radius, det_radius, angles, detector_shape, nodes_on_bdry
        )

    return odl.tomo.RayTransform(space, geometry)


def get_FBP_operator(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
    mode: Literal["cone", "parallel"] = "parallel",
    src_radius: Optional[float] = None,
    det_radius: Optional[float] = None,
) -> odl.Operator:
    """Get filtered back projection(FP) operator from ODL

    Args:
        size (Union[int, list[int], tuple[int]]): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        mode (Literal[&#39;cone&#39;, &#39;parallel&#39;], optional): Geometry type. Defaults to 'parallel'.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.
        src_radius (float, optional): geometry parameter in FanBeamGeometry/ConeBeamGeometry (mode='cone'). Defaults to None.
        det_radius (float, optional): geometry parameter in FanBeamGeometry/ConeBeamGeometry (mode='cone'). Defaults to None.

    Raises:
        ValueError: Invalid dimension for FP!
        ValueError: Undefined geometry mode! Availble mode: [cone, parallel]
        ValueError: Invalid cone beam parameters!

    Returns:
        odl.Operator: FBP operator
    """
    fp = get_FP_operator(
        size, dim, angles, detector_shape, mode, src_radius, det_radius
    )
    return odl.tomo.fbp_op(fp)


def get_paired_CT_operator(
    size: Union[int, list[int], tuple[int]],
    dim: Literal[2, 3] = 2,
    angles: Optional[int] = None,
    detector_shape=None,
    mode: Literal["cone", "parallel"] = "parallel",
    src_radius: Optional[float] = None,
    det_radius: Optional[float] = None,
) -> tuple[odl.Operator, odl.Operator]:
    """Get filtered back projection(FBP) operator from ODL

    Args:
        size (Union[int, list[int], tuple[int]]): Input image size
        angles (int, optional): Number of angles to do projection. Defaults to None.
        mode (Literal[&#39;cone&#39;, &#39;parallel&#39;], optional): Geometry type. Defaults to 'parallel'.
        dim (Literal[2, 3], optional): 2D/3D operation. Defaults to 2.
        src_radius (float, optional): geometry parameter in FanBeamGeometry/ConeBeamGeometry (mode='cone'). Defaults to None.
        det_radius (float, optional): geometry parameter in FanBeamGeometry/ConeBeamGeometry (mode='cone'). Defaults to None.

    Raises:
        ValueError: Invalid dimension for FP!
        ValueError: Undefined geometry mode! Availble mode: [cone, parallel]
        ValueError: Invalid cone beam parameters!

    Returns:
        Tuple[odl.Operator]: paired FP/FBP operator
    """
    fp = get_FP_operator(
        size, dim, angles, detector_shape, mode, src_radius, det_radius
    )
    return fp, odl.tomo.fbp_op(fp)


def get_projections_from_3dimage(image, **kwargs):
    image = np.transpose(image, (1, 2, 0))
    fp = get_FP_operator(image.shape, dim=3, **kwargs)
    return fp(image).asarray()
