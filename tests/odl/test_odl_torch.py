import torch

from torchCTutils.odl import get_paired_CT_func, get_paired_CT_layer, odl_FP, odl_FBP


def test_fpfbp_func_2D(phantom_2D_tensor, size, angles, detectors):
    sinogram = odl_FP(phantom_2D_tensor, angles=angles, detector_shape=detectors)
    assert sinogram.shape == (2, 1, angles, detectors)

    recon = odl_FBP(sinogram, size, angles=angles, detector_shape=detectors)
    assert type(recon) == torch.Tensor
    assert phantom_2D_tensor.shape == recon.shape


def test_fpfbp_func_2D_paired(phantom_2D_tensor, size, angles, detectors):
    size = phantom_2D_tensor.shape[-1]
    fp_func, fbp_func = get_paired_CT_func(
        size, angles=angles, detector_shape=detectors
    )

    sinogram = fp_func(phantom_2D_tensor)
    assert sinogram.shape == (2, 1, angles, detectors)
    recon = fbp_func(sinogram)
    assert phantom_2D_tensor.shape == recon.shape


def test_fpfbp_module_2D(phantom_2D_tensor, size, angles, detectors):
    fp_layer, fbp_layer = get_paired_CT_layer(
        size, angles=angles, detector_shape=detectors
    )

    sinogram = fp_layer(phantom_2D_tensor)
    assert sinogram.shape == (2, 1, angles, detectors)
    recon = fbp_layer(sinogram)
    assert phantom_2D_tensor.shape == recon.shape


def test_fpfbp_func_3D(phantom_3D_tensor, size, angles, detectors):
    sinogram = odl_FP(phantom_3D_tensor, dim=3, angles=angles, detector_shape=detectors)
    assert sinogram.shape == (2, 1, angles, detectors, detectors)
    recon = odl_FBP(sinogram, size, dim=3, angles=angles, detector_shape=detectors)
    assert phantom_3D_tensor.shape == recon.shape


def test_fpfbp_func_3D_paired(phantom_3D_tensor, size, angles, detectors):
    fp_func, fbp_func = get_paired_CT_func(
        size, dim=3, angles=angles, detector_shape=detectors
    )

    sinogram = fp_func(phantom_3D_tensor)
    assert sinogram.shape == (2, 1, angles, detectors, detectors)
    recon = fbp_func(sinogram)
    assert phantom_3D_tensor.shape == recon.shape


def test_fpfbp_module_3D(phantom_3D_tensor, size, angles, detectors):
    fp_layer, fbp_layer = get_paired_CT_layer(
        size, dim=3, angles=angles, detector_shape=detectors
    )

    sinogram = fp_layer(phantom_3D_tensor)
    assert sinogram.shape == (2, 1, angles, detectors, detectors)
    recon = fbp_layer(sinogram)
    assert phantom_3D_tensor.shape == recon.shape
