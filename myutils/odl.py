from functools import partial
from typing import Literal

import odl
from odl.contrib import torch as odl_torch


def get_FP_operator(
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


def get_FBP_operator(
    size,
    angles=None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: int = 2,
    src_radius: float = None,
    det_radius: float = None
):
    fp = get_FP_operator(size, angles, mode, dim, src_radius, det_radius)
    return odl.tomo.fbp_op(fp)


def get_paired_CT_operator(
    size,
    angles=None,
    mode: Literal['cone', 'parallel'] = 'parallel',
    dim: int = 2,
    src_radius: float = None,
    det_radius: float = None
):
    fp = get_FP_operator(size, angles, mode, dim, src_radius, det_radius)
    return fp, odl.tomo.fbp_op(fp)


def get_FP_layer(size: int, angles: int = None, dim=2):
    fp = get_FP_operator(size, angles, dim=dim)
    return odl_torch.OperatorModule(fp)


def get_FBP_layer(size: int, angles: int = None, dim=2):
    fbp = get_FBP_operator(size, angles, dim=dim)
    return odl_torch.OperatorModule(fbp)


def get_paired_CT_layer(size: int, angles: int = None, dim=2):
    fp, fbp = get_paired_CT_operator(size, angles, dim=dim)
    return odl_torch.OperatorModule(fp), odl_torch.OperatorModule(fbp)


def get_FP_func(size: int, angles: int = None, dim=2):
    return partial(odl_torch.OperatorFunction.apply, get_FP_operator(size, angles, dim=dim))


def get_FBP_func(size: int, angles: int = None, dim=2):
    return partial(odl_torch.OperatorFunction.apply, get_FBP_operator(size, angles, dim=dim))


def get_paired_CT_func(size: int, angles: int = None, dim=2):
    fp, fbp = get_paired_CT_operator(size, angles, dim=dim)
    return partial(odl_torch.OperatorFunction.apply, fp), partial(odl_torch.OperatorFunction.apply, fbp)
