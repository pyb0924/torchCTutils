import numpy as np
import matplotlib.pyplot as plt

from torchCTutils.odl import get_FP_operator, get_FBP_operator, get_paired_CT_operator
from torchCTutils.odl.utils import get_parallel_beam_geometry, get_cone_beam_geometry


def test_boundary_geometry_2D(space_2D, detectors):
    parallel_geometry = get_parallel_beam_geometry(space_2D, 3, detectors, True)
    assert (parallel_geometry.angles == np.array([0, np.pi / 2, np.pi])).all()

    cone_geometry = get_cone_beam_geometry(space_2D, 60, 80, 3, detectors, True)
    assert (cone_geometry.angles == np.array([0, np.pi, np.pi * 2])).all()


def test_boundary_geometry_3D(space_3D, detectors):
    parallel_geometry = get_parallel_beam_geometry(space_3D, 3, detectors, True)
    assert (parallel_geometry.angles == np.array([0, np.pi / 2, np.pi])).all()

    cone_geometry = get_cone_beam_geometry(space_3D, 60, 80, 3, detectors, True)
    assert (cone_geometry.angles == np.array([0, np.pi, np.pi * 2])).all()


def test_fpfbp_2D(phantom_2D, output_dir, size, angles, detectors):
    for mode in ["parallel", "cone"]:
        assert mode == "parallel" or mode == "cone"
        fp = get_FP_operator(size, 2, angles, detectors, mode, 60, 80)
        fbp = get_FBP_operator(size, 2, angles, detectors, mode, 60, 80)

        sinogram = fp(phantom_2D)
        plt.figure()
        plt.imshow(np.flipud(sinogram.data.T), "gray")
        plt.savefig(output_dir / f"sinogram_2D_{mode}.png")

        recon = fbp(sinogram)
        plt.figure()
        plt.imshow(np.flipud(recon.data.T), "gray")
        plt.savefig(output_dir / f"recon_2D_{mode}.png")


def test_fpfbp_paired(phantom_2D, output_dir, size, angles, detectors):
    for mode in ["parallel", "cone"]:
        assert mode == "parallel" or mode == "cone"
        fp, fbp = get_paired_CT_operator(size, 2, angles, detectors, mode, 60, 80)

        sinogram = fp(phantom_2D)
        plt.figure()
        plt.imshow(np.flipud(sinogram.data.T), "gray")
        plt.savefig(output_dir / f"sinogram_2D_{mode}_paired.png")

        recon = fbp(sinogram)
        plt.figure()
        plt.imshow(np.flipud(recon.data.T), "gray")
        plt.savefig(output_dir / f"recon_2D_{mode}_paired.png")


def test_fpfbp_3D(phantom_3D, output_dir, size, angles, detectors):
    for mode in ["parallel", "cone"]:
        assert mode == "parallel" or mode == "cone"
        fp = get_FP_operator(size, 3, angles, detectors, mode, 60, 80)
        fbp = get_FBP_operator(size, 3, angles, detectors, mode, 60, 80)

        sinogram = fp(phantom_3D)
        plt.figure()
        plt.imshow(np.flipud(sinogram.data.T)[64], "gray")
        plt.savefig(output_dir / f"sinogram_3D_{mode}.png")

        recon = fbp(sinogram)
        plt.figure()
        plt.imshow(np.flipud(recon.data.T)[64], "gray")
        plt.savefig(output_dir / f"recon_3D_{mode}.png")
