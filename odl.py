from functools import partial
from typing import Literal

import odl
from odl.contrib import torch as odl_torch


def getFPOperator(
    size: int,
    angles: int = None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: int = 2,
    src_radius: float = None,
    det_radius: float = None
):
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


def getFBPOperator(
    size,
    angles=None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: int = 2,
    src_radius: float = None,
    det_radius: float = None
):
    fp = getFPOperator(size, angles, mode, dim, src_radius, det_radius)
    return odl.tomo.fbp_op(fp)


def getFPLayer2D(size: int, angles: int = None):
    fp = getFPOperator(size, angles)
    return odl_torch.OperatorModule(fp)


def getFBPLayer2D(size: int, angles: int = None):
    fbp = getFBPOperator(size, angles)
    return odl_torch.OperatorModule(fbp)


def getFPfunc(size: int, angles: int = None):
    return partial(odl_torch.OperatorFunction.apply, getFPOperator(size, angles))


def getFBPfunc(size: int, angles: int = None):
    return partial(odl_torch.OperatorFunction.apply, getFBPOperator(size, angles))
