from functools import partial
from typing import Literal, Callable

import odl


def get_FP_operator(
    size: int,
    angles: int = None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: Literal[2, 3] = 2,
    src_radius: float = None,
    det_radius: float = None
) -> odl.Operator:
    """Get forward projection(FP) operator from ODL

    Args:
        size (int): Input image size
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
        raise ValueError('Invalid dimension for FP!')

    if dim == 2:
        space = odl.uniform_discr([-1, -1], [1, 1], [size, size])
    else:
        space = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [size, size, size])

    if mode != 'parallel' and mode != 'cone':
        raise ValueError(
            'Undefined geometry mode! Availble mode: [cone, parallel]')

    if mode == 'parallel':
        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=angles)
    else:
        if src_radius == None or det_radius == None:
            raise ValueError('Invalid cone beam parameters!')
        geometry = odl.tomo.cone_beam_geometry(space, src_radius, det_radius)

    return odl.tomo.RayTransform(space, geometry)


def get_FBP_operator(
    size,
    angles=None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: Literal[2, 3] = 2,
    src_radius: float = None,
    det_radius: float = None
) -> odl.Operator:
    """Get filtered back projection(FP) operator from ODL

    Args:
        size (int): Input image size
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
    fp = get_FP_operator(size, angles, mode, dim, src_radius, det_radius)
    return odl.tomo.fbp_op(fp)


def get_paired_CT_operator(
    size,
    angles=None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: Literal[2, 3] = 2,
    src_radius: float = None,
    det_radius: float = None
) -> tuple[odl.Operator]:
    """Get filtered back projection(FBP) operator from ODL

    Args:
        size (int): Input image size
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
    fp = get_FP_operator(size, angles, mode, dim, src_radius, det_radius)
    return fp, odl.tomo.fbp_op(fp)

