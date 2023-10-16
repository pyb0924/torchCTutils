import numpy as np
from odl.discr import uniform_partition
from odl.tomo.geometry import (
    Parallel2dGeometry,
    Parallel3dAxisGeometry,
    FanBeamGeometry,
    ConeBeamGeometry,
)


def get_parallel_beam_geometry(space, num_angles, det_shape, nodes_on_bdry=False):
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1

    if space.ndim == 2:
        det_min_pt = -rho
        det_max_pt = rho
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        num_px_vert = space.shape[2]
        min_h = space.domain.min_pt[2]
        max_h = space.domain.max_pt[2]
        det_min_pt = [-rho, min_h]
        det_max_pt = [rho, max_h]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]
    angle_partition = uniform_partition(
        0, np.pi, num_angles, nodes_on_bdry=nodes_on_bdry
    )
    det_partition = uniform_partition(
        det_min_pt, det_max_pt, det_shape, nodes_on_bdry=nodes_on_bdry
    )

    if space.ndim == 2:
        return Parallel2dGeometry(angle_partition, det_partition)
    elif space.ndim == 3:
        return Parallel3dAxisGeometry(angle_partition, det_partition)
    else:
        raise ValueError("``space.ndim`` must be 2 or 3.")


def get_cone_beam_geometry(
    space, src_radius, det_radius, num_angles, det_shape, nodes_on_bdry=False
):
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side

    # Compute minimum width of the detector to cover the object. The relation
    # used here is (w/2)/(rs+rd) = rho/rs since both are equal to tan(alpha),
    # where alpha is the half fan angle.
    rs = float(src_radius)
    if rs <= rho:
        raise ValueError(
            "source too close to the object, resulting in "
            "infinite detector for full coverage"
        )
    rd = float(det_radius)
    r = src_radius + det_radius
    w = 2 * rho * (rs + rd) / rs

    # Compute minimum number of pixels given the constraint on the
    # sampling interval and the computed width
    rb = np.hypot(r, w / 2)  # length of the boundary ray to the flat detector
    num_px_horiz = 2 * int(np.ceil(w * omega * r / (2 * np.pi * rb))) + 1

    if space.ndim == 2:
        det_min_pt = -w / 2
        det_max_pt = w / 2
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        # Compute number of vertical pixels required to cover the object,
        # using the same sampling interval vertically as horizontally.
        # The reasoning is the same as for the computation of w.

        # Minimum distance of the containing cuboid edges to the source
        dist = rs - rho
        # Take angle of the rays going through the top and bottom corners
        # in that edge
        half_cone_angle = max(
            np.arctan(abs(space.partition.min_pt[2]) / dist),
            np.arctan(abs(space.partition.max_pt[2]) / dist),
        )
        h = 2 * np.sin(half_cone_angle) * (rs + rd)

        # Use the vertical spacing from the reco space, corrected for
        # magnification at the "back" of the object, i.e., where it is
        # minimal
        min_mag = (rs + rd) / (rs + rho)
        delta_h = min_mag * space.cell_sides[2]
        num_px_vert = int(np.ceil(h / delta_h))
        h = num_px_vert * delta_h  # make multiple of spacing

        det_min_pt = [-w / 2, -h / 2]
        det_max_pt = [w / 2, h / 2]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]

    max_angle = 2 * np.pi

    if num_angles is None:
        num_angles = int(np.ceil(max_angle * omega * rho / np.pi * r / (r + rho)))

    angle_partition = uniform_partition(
        0, max_angle, num_angles, nodes_on_bdry=nodes_on_bdry
    )
    det_partition = uniform_partition(
        det_min_pt, det_max_pt, det_shape, nodes_on_bdry=nodes_on_bdry
    )

    if space.ndim == 2:
        return FanBeamGeometry(angle_partition, det_partition, src_radius, det_radius)
    elif space.ndim == 3:
        return ConeBeamGeometry(angle_partition, det_partition, src_radius, det_radius)
    else:
        raise ValueError("``space.ndim`` must be 2 or 3.")
